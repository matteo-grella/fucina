//! Disk-streamed MoE expert tier: resolves one layer's routed experts to
//! their quantized weight blocks through a pinned set, a per-layer LRU cache
//! in RAM, and `pread` from the GGUF file — so a mixture model whose expert
//! stacks dwarf physical RAM still decodes, paying disk reads only for cache
//! misses. The design follows measured out-of-core MoE lessons: the
//! streamed tier reads with `pread` into store-owned
//! buffers rather than mmap, so resident memory is exactly dense weights +
//! this cache (mmap'd expert pages inflate RSS and let page-cache pressure
//! evict semi-randomly instead of by routing recency); misses issue
//! OS readahead hints (`POSIX_FADV_WILLNEED` / `F_RDADVISE`) for the whole
//! batch before the first synchronous read; and miss loads land in reusable
//! working-set slots that are promoted into the LRU by buffer swap after the
//! layer computes, so the cache capacity is independent of how many experts
//! one batched prefill touches.
//!
//! Concurrency contract: `acquire` locks the store until the matching
//! `release` (one MoE layer op at a time — the model forward is sequential
//! over ops). Between the two, worker threads read resolved expert pointers
//! through `StreamedMoeRhs.expertBytes` without further synchronization:
//! the pointer table is written before the op's tasks are spawned and
//! nothing evicts or swaps slots until `release`. The wave-split variant
//! (`acquireStart` / `acquireFinish`) keeps the same lock scope but returns
//! from start with the miss reads still in flight on the I/O pool, so the
//! caller can compute the already-resident experts inside the read shadow;
//! only experts whose residency bit was set at start may be touched before
//! `acquireFinish` resolves the rest.
//!
//! Layout contracts: every resolved range starts `slab_align`ed (the MoE
//! dispatch `@alignCast`s at each kernel call), and a multi-plane (PTQTP)
//! range keeps its planes contiguous in slab order (the dispatch reads
//! plane `p` at offset `p * plane_bytes`). Kernels see plain byte views
//! and `acquire` takes opaque indices. Dense/cyclic weight offload is out
//! of scope: LRU under cyclic access degenerates to zero hits.
//!
//! Layout: this file is the facade -- it owns the store state, the mutex,
//! and the acquire/release contract above; the concern bodies live in
//! `store/` (`io` platform shims + the error set, `geometry` formats and
//! layout math, `tiers` slots/LRU+heat/pinned/staging/L2, `policy`
//! routing/repin/budget, `persist` the FUCEXPT1/FUCTRCE1 formats).
const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const backend_mod = @import("../backend.zig");
const thread = @import("../thread.zig");

const io = @import("io.zig");
const geometry = @import("geometry.zig");
const tiers = @import("tiers.zig");
const policy = @import("policy.zig");
const persist = @import("persist.zig");

const Allocator = std.mem.Allocator;
const qm = backend_mod.quant;
const fd_t = io.fd_t;
const ProjGeometry = geometry.ProjGeometry;
const Slot = tiers.Slot;
const StageMeta = tiers.StageMeta;
const PilotReq = tiers.PilotReq;
const MirrorSet = policy.MirrorSet;
const invalid_eid = tiers.invalid_eid;
const slab_align = tiers.slab_align;

pub const Error = io.Error;
pub const StreamedQuant = geometry.StreamedQuant;
pub const Proj = geometry.Proj;
pub const ProjSpec = geometry.ProjSpec;
pub const LayerState = tiers.LayerState;
pub const memAvailableBytes = policy.memAvailableBytes;

pub const ExpertStore = struct {
    pub const Options = struct {
        /// Fixed LRU slots per layer; wins over `cache_bytes` when set.
        cache_slots_per_layer: ?usize = null,
        /// Total RAM budget for the streamed tiers (pinned + LRU) across all
        /// layers. Default: half of the available memory at finalize time
        /// (8 GiB fallback).
        cache_bytes: ?usize = null,
        /// Issue WILLNEED-style readahead for the whole miss set before the
        /// first synchronous read.
        readahead: bool = true,
        /// Staging slots for the prefetch worker's true async loads (0 =
        /// kernel-advice era behavior: hints enqueue but load nothing).
        /// Slabs allocate lazily to the hinted layers' sizes.
        prefetch_stage_slots: usize = 64,
        /// Demand-miss reads fan out across this many persistent I/O
        /// worker threads (the acquiring thread drains the same batch, so
        /// disk concurrency is `io_workers + 1`); 0 = sequential reads on
        /// the calling thread. Parallel misses are what let NVMe queue
        /// depth — and mirror copies on separate drives — actually add
        /// bandwidth within a single acquire.
        io_workers: usize = 8,
        /// The learning cache: when the persisted usage histogram (sidecar
        /// `<gguf>.experts` file) carries enough history, pin the hottest
        /// experts in RAM at finalize — they are read once at startup and
        /// never evicted. The engine gets faster the more it is used.
        auto_pin: bool = true,
        /// RAM for the pinned tier; default: half of the total budget when
        /// history qualifies (the LRU gets the remainder).
        pin_bytes: ?usize = null,
        /// Minimum recorded routed pairs before auto-pin trusts the history.
        auto_pin_min_history: u64 = 5000,
        /// Auto-pin must beat a FLAT histogram by this factor or it declines
        /// and hands the whole budget to the LRU. Quantile-balanced routers
        /// (Kimi-K3 class) deliberately flatten expert usage; there the
        /// "hottest" experts are noise and a pinned tier retains no more
        /// traffic than random slots would — measured upstream as 0.0%
        /// retention across a 48x cache-size sweep (kimi-k3-in-c). Only
        /// consulted when the pin budget cannot hold every used expert
        /// (when it can, pinning the whole working set is optimal
        /// regardless of skew). 1.0 disables the guard.
        auto_pin_min_advantage: f32 = 1.25,
        /// Uncached streamed reads (macOS `F_NOCACHE` on every store and
        /// mirror fd): a streamed giant reads far more than RAM per token,
        /// so page-caching those reads only churns out the mmap'd DENSE
        /// weights' pages. Already-cached pages still serve hits; the
        /// store's own pinned/LRU tiers are the re-read safety net. No-op
        /// on Linux (O_DIRECT needs aligned slabs — rig follow-up).
        uncached: bool = false,
        /// Cache-aware routing (`cacheRouteTopK`), default off — it changes
        /// expert selection, so callers opt in explicitly.
        cache_route: ?CacheRouteOptions = null,
        /// LRU-tier victim choice: evict the lowest-heat resident (the
        /// per-expert routed-pair counters the store already keeps),
        /// recency breaking ties — measured on real DeepSeek-V2-Lite
        /// routing traces at +3-10pp hit rate over pure LRU at every
        /// capacity, both via `tools/replay_experts.zig` and live.
        /// `FUCINA_MOE_LRU=1` forces pure LRU at runtime (the A/B and
        /// emergency-revert switch on one binary). Outputs are identical
        /// either way; only which expert gets re-read later changes.
        heat_eviction: bool = true,
        /// Record every routed (layer, expert) pair in request order and
        /// write the sequence here at `destroy` (see `saveTrace` for the
        /// format). The usage HISTOGRAM cannot answer policy questions —
        /// LRU and Belady both need temporal order — so this is the input
        /// to `tools/replay_experts.zig`'s offline cache replay. Purely
        /// diagnostic: recording failures are swallowed, serving never
        /// depends on it. The path is borrowed; it must outlive the store.
        trace_path: ?[]const u8 = null,
        /// Second-tier expert cache on a FASTER drive (typically the
        /// internal NVMe behind a USB-attached model): one sparse file per
        /// GGUF part holding hot experts' raw bytes AT THEIR PRIMARY
        /// OFFSETS, plus a `.idx` presence sidecar. A demand miss (or
        /// pilot stage) whose expert is present reads from the tier
        /// instead of the primary — same bytes, same kernels, identical
        /// output; only the drive changes. Derived data: rebuild any time
        /// with `l2_build_bytes`. Borrowed; must outlive the store.
        l2_path: ?[]const u8 = null,
        /// (Re)build the L2 tier at finalize before serving: rank experts
        /// by the loaded usage history (no history: natural order), skip
        /// the RAM-pinned ones (they never miss), copy primary -> tier
        /// until this many payload bytes, then write the index. Requires
        /// `l2_path`; the caller sizes this against the tier drive's free
        /// space.
        l2_build_bytes: ?u64 = null,
    };

    /// Knobs for `cacheRouteTopK` (max-rank selection, arXiv:2412.00099).
    pub const CacheRouteOptions = struct {
        /// True top ranks always taken, resident or not.
        sacred: usize = 2,
        /// Max-rank window the resident-preferring fill may draw from.
        window: usize = 12,
    };

    pub const Stats = struct {
        hits: u64 = 0,
        pin_hits: u64 = 0,
        misses: u64 = 0,
        bytes_read: u64 = 0,
        read_ns: u64 = 0,
        acquires: u64 = 0,
        /// Pilot (router lookahead): ranges enqueued to the I/O thread, and
        /// prediction recall over the acquires that followed a prediction.
        pilot_ranges: u64 = 0,
        pilot_recall_hits: u64 = 0,
        pilot_recall_total: u64 = 0,
        /// Staging tier (true async prefetch): worker loads completed,
        /// loads consumed by `acquire`, ready entries overwritten unused,
        /// and bytes read by the worker (disjoint from `bytes_read`).
        staged_loads: u64 = 0,
        staged_consumed: u64 = 0,
        staged_wasted: u64 = 0,
        staged_bytes: u64 = 0,
        /// Cache-aware routing: selection slots filled, and how many held a
        /// resident substitute instead of the true top-k expert.
        route_slots: u64 = 0,
        route_swaps: u64 = 0,

        pub fn routeSwapRate(self: Stats) f64 {
            if (self.route_slots == 0) return 0;
            return @as(f64, @floatFromInt(self.route_swaps)) / @as(f64, @floatFromInt(self.route_slots));
        }

        pub fn hitRate(self: Stats) f64 {
            const total = self.hits + self.misses;
            if (total == 0) return 0;
            return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
        }

        pub fn pilotRecall(self: Stats) f64 {
            if (self.pilot_recall_total == 0) return 0;
            return @as(f64, @floatFromInt(self.pilot_recall_hits)) / @as(f64, @floatFromInt(self.pilot_recall_total));
        }
    };

    allocator: Allocator,
    /// One read fd per split part (single-file GGUFs: one entry).
    fds: []fd_t,
    layers: []LayerState,
    registered: []bool,
    options: Options,
    /// Sidecar path of the persistent usage histogram (`<gguf>.experts`).
    usage_path: []u8,
    /// LRU slots per layer, fixed at `finalize`.
    cap: usize = 0,
    /// Pinned-tier summary, fixed at `finalize` (adapted by `repinPass`).
    pinned_experts: usize = 0,
    pinned_bytes: usize = 0,
    /// Auto-pin declined because the usage histogram was ~flat (quantile-
    /// balanced router): pinning would have retained no more traffic than
    /// random slots, so the whole budget went to the LRU tier instead.
    pins_declined_flat: bool = false,
    finalized: bool = false,
    clock: u64 = 0,
    /// `clock` at the current acquire's entry: slots stamped after it were
    /// touched by THIS acquire (hits or fresh promotions) and are never
    /// heat-eviction victims — evicting what the token just used or what
    /// this release just promoted would cannibalize the working set (pure
    /// LRU gets the same protection implicitly from recency).
    acquire_clock0: u64 = 0,
    stats: Stats = .{},
    mutex: thread.Mutex = .{},
    /// Routing trace (pairs of layer, eid in request order; see
    /// `Options.trace_path`). Grows unbounded by design — opt-in
    /// diagnostic for bounded analysis sessions.
    trace: std.ArrayList(u32) = .empty,
    // ---- per-acquire state (valid while the mutex is held) ----
    acquired_layer: ?usize = null,
    /// Unique experts of the current acquire / their miss subset (parallel
    /// to `work[0..n_miss]`).
    active: []u32 = &.{},
    n_active: usize = 0,
    miss_eids: []u32 = &.{},
    n_miss: usize = 0,
    /// Working-set slots for the current acquire's misses; promoted into the
    /// layer LRU (by slab swap) at release. Grown to the largest miss set.
    work: []Slot = &.{},
    seen: []bool = &.{},
    // ---- pilot (router-lookahead prefetch) ----
    // A dedicated I/O thread drains an SPSC ring of file ranges and issues
    // the readahead advice there: with a saturated disk queue the advice
    // call itself BLOCKS (measured ~0.5 ms each upstream), so hinting inline
    // would cost the forward thread more than the overlap earns. Ring full
    // = drop: a lost hint is not an error.
    pilot_ring: []PilotReq = &.{},
    pilot_w: std.atomic.Value(u32) = .init(0),
    pilot_r: std.atomic.Value(u32) = .init(0),
    pilot_stop: std.atomic.Value(bool) = .init(false),
    pilot_thread: ?std.Thread = null,

    // ---- staging tier (true async prefetch) ----
    // The worker claims a slot under `stage_mutex` (an empty one, else the
    // oldest ready — never a loading one), preads OUTSIDE the lock into the
    // claimed slab, then publishes `.ready`; `acquire`'s miss path consumes
    // ready entries by zero-copy slab swap. Never touches the store mutex
    // from the worker; deadlock-free by construction, and staged bytes come
    // from the same `readExpert` on the same offsets — bit-identical.
    stage_mutex: thread.Mutex = .{},
    stage_slots: []Slot = &.{},
    stage_meta: []StageMeta = &.{},
    stage_stamp: u64 = 0,

    // ---- mirror copies (N-drive expert streaming) ----
    // Extra full copies of the model on other drives (`addMirror`). Each
    // expert's reads — and its readahead hints — route to ONE copy by a
    // deterministic weighted (layer, expert) hash. Deterministic on
    // purpose: the hint and the demand pread must warm/hit the SAME
    // drive's page cache, and one expert's bytes must never occupy page
    // cache on two drives. Weighted so asymmetric drives carry
    // asymmetric shares. Aggregate read bandwidth scales with the drives
    // holding a copy; a mirror read error falls back to the primary
    // (mirrors add bandwidth, never correctness).
    mirrors: []MirrorSet = &.{},
    /// Cumulative routing thresholds over 1 << 16, primary first; empty
    /// until a mirror is attached. Built at `finalize`.
    route_cum: []u32 = &.{},
    /// Bytes served per copy (index 0 = primary) — atomic because the
    /// forward thread and the prefetch worker both read. At `finalize`.
    copy_bytes: []std.atomic.Value(u64) = &.{},
    mirror_fallbacks: std.atomic.Value(u64) = .init(0),

    // ---- L2 tier (packed expert slabs on a faster drive) ----
    /// Read fds per part; empty until `finalize` opens a tier.
    l2_fds: []fd_t = &.{},
    /// Per (layer, expert) slab-PREFIX offset into the tier file;
    /// `l2_absent` = not in the tier. Empty slice per unregistered layer.
    l2_present: [][]u64 = &.{},
    /// Per-layer striped prefix length (bytes of each covered expert's
    /// slab that live in the tier file; the suffix stays on the primary
    /// and the two are read in PARALLEL per miss). 0 = layer not striped.
    l2_prefix: []u64 = &.{},
    /// Experts served from the tier / bytes / read errors that fell back
    /// to the primary — atomics for the same two-thread reason as
    /// `copy_bytes`.
    l2_expert_hits: std.atomic.Value(u64) = .init(0),
    l2_bytes: std.atomic.Value(u64) = .init(0),
    l2_read_ns: std.atomic.Value(u64) = .init(0),
    l2_fallbacks: std.atomic.Value(u64) = .init(0),

    // ---- demand-miss I/O pool ----
    // Persistent futex-parked workers (thread.OneShotWorker) that drain
    // one read batch per acquire, started lazily on the first acquire
    // with more than one uncached miss. An atomic cursor hands each item
    // to exactly one thread; a failed item is only MARKED — the
    // acquiring thread retries it synchronously afterwards, so a
    // transient error heals and a real one surfaces with its true error
    // type. Only `io_pool[0..io_ready]` entries are initialized (a
    // partial pool after a spawn failure still drains correctly — the
    // caller always participates).
    io_pool: []thread.OneShotWorker = &.{},
    io_ready: usize = 0,
    /// Per-acquire read set (misses minus staged consumptions), sized to
    /// the largest layer at `finalize`; parallel arrays.
    read_eids: []u32 = &.{},
    read_slots: []*Slot = &.{},
    read_failed: []bool = &.{},

    // ---- wave-split acquire (hit-compute under miss-I/O overlap) ----
    /// The in-flight read batch between `acquireStart` and `acquireFinish`.
    /// Lives on the store — the I/O workers hold its address across the
    /// caller's compute window, so it must not be stack memory.
    miss_batch: IoBatch = undefined,
    /// Workers started on `miss_batch`; meaningful only while
    /// `miss_inflight`.
    miss_started: usize = 0,
    miss_inflight: bool = false,

    /// The store is heap-allocated so `StreamedMoeRhs` values and the owning
    /// model can hold stable pointers while the model struct moves by value.
    /// `gguf_paths` lists every part of a split GGUF (single-file: one
    /// entry); `ProjSpec.part` indexes into it.
    pub fn create(allocator: Allocator, gguf_paths: []const []const u8, n_layers: usize, options: Options) Error!*ExpertStore {
        std.debug.assert(gguf_paths.len > 0);
        const fds = try allocator.alloc(fd_t, gguf_paths.len);
        errdefer allocator.free(fds);
        var n_open: usize = 0;
        errdefer for (fds[0..n_open]) |fd| io.closeFd(fd);
        for (gguf_paths) |path| {
            fds[n_open] = try io.openReadOnly(allocator, path);
            if (options.uncached) io.setUncached(fds[n_open]);
            n_open += 1;
        }

        const self = try allocator.create(ExpertStore);
        errdefer allocator.destroy(self);
        const layers = try allocator.alloc(LayerState, n_layers);
        errdefer allocator.free(layers);
        const registered = try allocator.alloc(bool, n_layers);
        errdefer allocator.free(registered);
        @memset(registered, false);
        const usage_path = try std.fmt.allocPrint(allocator, "{s}.experts", .{gguf_paths[0]});

        self.* = .{
            .allocator = allocator,
            .fds = fds,
            .layers = layers,
            .registered = registered,
            .options = options,
            .usage_path = usage_path,
        };
        return self;
    }

    pub fn destroy(self: *ExpertStore) void {
        const allocator = self.allocator;
        // Best-effort: a failed trace write loses an analysis artifact,
        // nothing else.
        self.saveTrace() catch {};
        self.trace.deinit(allocator);
        for (self.layers, self.registered) |*ls, reg| {
            if (!reg) continue;
            for (ls.slots) |*slot| slot.deinit(allocator);
            allocator.free(ls.slots);
            for (ls.pinned) |*slot| slot.deinit(allocator);
            allocator.free(ls.pinned);
            allocator.free(ls.resolved);
            allocator.free(ls.usage);
            allocator.free(ls.heat);
            allocator.free(ls.pred_marks);
        }
        if (self.pilot_thread) |t| {
            self.pilot_stop.store(true, .release);
            t.join();
        }
        for (self.stage_slots) |*slot| slot.deinit(allocator);
        if (self.stage_slots.len > 0) allocator.free(self.stage_slots);
        if (self.stage_meta.len > 0) allocator.free(self.stage_meta);
        allocator.free(self.pilot_ring);
        allocator.free(self.usage_path);
        for (self.work) |*slot| slot.deinit(allocator);
        allocator.free(self.work);
        allocator.free(self.active);
        allocator.free(self.miss_eids);
        allocator.free(self.seen);
        allocator.free(self.layers);
        allocator.free(self.registered);
        for (self.io_pool[0..self.io_ready]) |*w| w.deinit();
        allocator.free(self.io_pool);
        allocator.free(self.read_eids);
        allocator.free(self.read_slots);
        allocator.free(self.read_failed);
        for (self.mirrors) |m| {
            for (m.fds) |fd| io.closeFd(fd);
            allocator.free(m.fds);
        }
        allocator.free(self.mirrors);
        allocator.free(self.route_cum);
        allocator.free(self.copy_bytes);
        for (self.l2_fds) |fd| io.closeFd(fd);
        if (self.l2_fds.len > 0) allocator.free(self.l2_fds);
        for (self.l2_present) |p| if (p.len > 0) allocator.free(p);
        if (self.l2_present.len > 0) allocator.free(self.l2_present);
        if (self.l2_prefix.len > 0) allocator.free(self.l2_prefix);
        for (self.fds) |fd| io.closeFd(fd);
        allocator.free(self.fds);
        allocator.destroy(self);
    }

    /// Attach one more full copy of the model: `paths` names the same
    /// split parts as the primary, typically on another drive, and
    /// `weight` is its read share relative to the primary's 1 (equal
    /// drives: 1; a drive half as fast: 0.5). Call any number of times
    /// before `finalize` — expert reads then split across every copy, so
    /// aggregate streaming bandwidth grows with each drive holding one.
    pub fn addMirror(self: *ExpertStore, paths: []const []const u8, weight: f32) Error!void {
        std.debug.assert(!self.finalized);
        if (paths.len != self.fds.len) return Error.InvalidExpertGeometry;
        if (!(weight > 0)) return Error.InvalidExpertGeometry;
        const fds = try self.allocator.alloc(fd_t, paths.len);
        errdefer self.allocator.free(fds);
        var n_open: usize = 0;
        errdefer for (fds[0..n_open]) |fd| io.closeFd(fd);
        for (paths) |path| {
            fds[n_open] = try io.openReadOnly(self.allocator, path);
            if (self.options.uncached) io.setUncached(fds[n_open]);
            n_open += 1;
        }
        const grown = try self.allocator.alloc(MirrorSet, self.mirrors.len + 1);
        @memcpy(grown[0..self.mirrors.len], self.mirrors);
        grown[self.mirrors.len] = .{ .fds = fds, .weight = weight };
        self.allocator.free(self.mirrors);
        self.mirrors = grown;
    }

    /// Register one MoE layer's three stacked expert tensors. Call once per
    /// layer during model load, then `finalize`.
    pub fn addLayer(self: *ExpertStore, layer_i: usize, specs: [3]ProjSpec, n_expert: usize) Error!void {
        if (layer_i >= self.layers.len) return Error.InvalidExpertGeometry;
        if (self.registered[layer_i]) return Error.LayerAlreadyRegistered;
        if (n_expert == 0 or n_expert >= invalid_eid) return Error.InvalidExpertGeometry;

        var projs: [3]ProjGeometry = undefined;
        for (specs, 0..) |spec, p| {
            if (spec.part >= self.fds.len) return Error.InvalidExpertGeometry;
            projs[p] = try geometry.ProjGeometry.init(spec, n_expert);
        }
        const layout = LayerState.expertSlabOffsets(projs);

        const resolved = try self.allocator.alloc([3]?[*]const u8, n_expert);
        errdefer self.allocator.free(resolved);
        @memset(resolved, .{ null, null, null });
        const usage = try self.allocator.alloc(u64, n_expert);
        errdefer self.allocator.free(usage);
        @memset(usage, 0);
        const heat = try self.allocator.alloc(u32, n_expert);
        @memset(heat, 0);
        const pred_marks = try self.allocator.alloc(u32, n_expert);
        @memset(pred_marks, 0);

        self.layers[layer_i] = .{
            .projs = projs,
            .n_expert = n_expert,
            .proj_off = layout.off,
            .slab_bytes = layout.total,
            .resolved = resolved,
            .usage = usage,
            .heat = heat,
            .pred_marks = pred_marks,
        };
        self.registered[layer_i] = true;
    }

    /// Register a slab-native layer: `specs` carry the three projections'
    /// GEOMETRY (their file offsets are ignored); the on-disk record at
    /// `[slab_file_offset + e*record_bytes, +record_bytes)` is the RAM slab
    /// byte-for-byte — projection sections at their 16 KiB-aligned
    /// `proj_off` positions, tail-padded to `record_bytes`. The converter
    /// writes that layout (`--repack-slab`); `record_bytes` is validated
    /// against the geometry-derived slab size, so a layout drift between
    /// producer and store fails loudly at load.
    pub fn addLayerSlab(self: *ExpertStore, layer_i: usize, specs: [3]ProjSpec, slab_part: u16, slab_file_offset: u64, record_bytes: usize, n_expert: usize) Error!void {
        if (slab_part >= self.fds.len) return Error.InvalidExpertGeometry;
        try self.addLayer(layer_i, specs, n_expert);
        const ls = &self.layers[layer_i];
        if (record_bytes != ls.slab_bytes) {
            return Error.InvalidExpertGeometry;
        }
        ls.slab_file_offset = slab_file_offset;
        ls.slab_part = slab_part;
    }

    /// Fix the tier layout from the configured budget: load the persisted
    /// usage history, carve the pinned tier out of the budget when the
    /// history qualifies (auto-pin — the learning cache), give the LRU the
    /// remainder, and allocate the bookkeeping sized to the largest
    /// registered layer. LRU slot slabs are allocated lazily on first
    /// promotion; pinned slabs are read from disk here, once.
    pub fn finalize(self: *ExpertStore) Error!void {
        var total_slab: usize = 0;
        var max_expert: usize = 0;
        var n_registered: usize = 0;
        for (self.layers, self.registered) |*ls, reg| {
            if (!reg) continue;
            n_registered += 1;
            total_slab += ls.slab_bytes;
            max_expert = @max(max_expert, ls.n_expert);
        }
        if (n_registered == 0) return Error.LayerNotRegistered;

        // The mirror routing table must exist before the pin tier loads
        // below (those reads route too): normalized cumulative weights,
        // primary first, over the 1 << 16 hash range.
        if (self.mirrors.len > 0) {
            self.route_cum = try self.allocator.alloc(u32, self.mirrors.len + 1);
            var total: f32 = 1;
            for (self.mirrors) |m| total += m.weight;
            var acc: f32 = 1;
            self.route_cum[0] = @intFromFloat(@round(acc / total * 65536.0));
            for (self.mirrors, 1..) |m, i| {
                acc += m.weight;
                self.route_cum[i] = @intFromFloat(@round(acc / total * 65536.0));
            }
            self.route_cum[self.mirrors.len] = 65536;
            self.copy_bytes = try self.allocator.alloc(std.atomic.Value(u64), self.mirrors.len + 1);
            for (self.copy_bytes) |*b| b.* = .init(0);
        }

        const history_pairs = persist.loadUsage(self);

        var budget: usize = self.options.cache_bytes orelse blk: {
            const avail = policy.memAvailableBytes() orelse (8 << 30);
            break :blk @max(avail / 2, 512 << 20);
        };
        if (self.options.auto_pin and history_pairs >= self.options.auto_pin_min_history) {
            const pin_budget = self.options.pin_bytes orelse budget / 2;
            const spent = try tiers.selectAndLoadPins(self, @min(pin_budget, budget));
            budget -= @min(spent, budget);
        }

        // L2 tier: build (explicit, reads the primary once) then open. After
        // the pin selection on purpose — pinned experts never miss, so the
        // builder spends the tier's bytes on the band below the pins.
        if (self.options.l2_path) |l2_base| {
            if (self.options.l2_build_bytes) |l2_budget| try tiers.l2Build(self, l2_base, l2_budget);
            try tiers.l2Open(self, l2_base);
        }

        var cap: usize = undefined;
        if (self.options.cache_slots_per_layer) |slots| {
            cap = @max(1, slots);
        } else {
            cap = @max(1, budget / total_slab);
        }
        cap = @min(cap, max_expert);
        self.cap = cap;

        for (self.layers, self.registered) |*ls, reg| {
            if (!reg) continue;
            const layer_cap = @min(cap, ls.n_expert);
            ls.slots = try self.allocator.alloc(Slot, layer_cap);
            @memset(ls.slots, .{});
        }
        self.active = try self.allocator.alloc(u32, max_expert);
        self.miss_eids = try self.allocator.alloc(u32, max_expert);
        self.work = try self.allocator.alloc(Slot, max_expert);
        @memset(self.work, .{});
        self.seen = try self.allocator.alloc(bool, max_expert);
        self.read_eids = try self.allocator.alloc(u32, max_expert);
        self.read_slots = try self.allocator.alloc(*Slot, max_expert);
        self.read_failed = try self.allocator.alloc(bool, max_expert);
        self.finalized = true;
    }

    /// Total bytes one layer's LRU tier may hold; times registered layers =
    /// the tier's peak footprint (plus one working set).
    pub fn perLayerCacheBytes(self: *const ExpertStore) usize {
        var max_bytes: usize = 0;
        for (self.layers, self.registered) |*ls, reg| {
            if (reg) max_bytes = @max(max_bytes, ls.slots.len * ls.slab_bytes);
        }
        return max_bytes;
    }

    /// A `MoeRhs.streamed` arm for one registered projection.
    pub fn streamedRhs(self: *ExpertStore, layer_i: usize, proj: Proj) StreamedMoeRhs {
        std.debug.assert(self.registered[layer_i]);
        const ls = &self.layers[layer_i];
        const g = &ls.projs[@intFromEnum(proj)];
        return .{
            .store = self,
            .layer_state = ls,
            .layer = layer_i,
            .proj = proj,
            .quant = g.quant,
            .k = g.in_dim,
            .out_dim = g.out_dim,
            .n_expert = ls.n_expert,
            .blocks_per_column = g.blocks_per_column,
            .plane_count = g.plane_count,
            // Folded SERVING: fill-time fold (sibling-plane files) or the
            // native pre-folded pack — the dispatch reads the same section
            // layout either way.
            .folded = g.fold or g.quant == .tq2_0_fx4,
        };
    }

    pub const max_route_window = policy.max_route_window;
    pub const trace_magic = persist.trace_magic;

    /// Cache-aware top-k expert selection (max-rank; `policy.cacheRouteTopK`
    /// holds the body and its WORKER-THREAD-ONLY contract).
    pub fn cacheRouteTopK(self: *ExpertStore, layer_i: usize, choice: []const f32, sel: []usize) bool {
        return policy.cacheRouteTopK(self, layer_i, choice, sel);
    }

    /// One pinned-tier adaptation pass (`policy.repinPass`). Call at safe
    /// boundaries, never inside an acquire.
    pub fn repinPass(self: *ExpertStore, max_swaps_per_layer: usize) usize {
        return policy.repinPass(self, max_swaps_per_layer);
    }

    /// Write the routing trace (`persist.saveTrace`; FUCTRCE1).
    pub fn saveTrace(self: *ExpertStore) Error!void {
        return persist.saveTrace(self);
    }

    /// Persist the usage histogram sidecar (`persist.saveUsage`; FUCEXPT1).
    pub fn saveUsage(self: *ExpertStore) Error!void {
        return persist.saveUsage(self);
    }

    /// Copy index (0 = primary) serving an expert's bytes
    /// (`policy.routeCopy`). Module-internal seam (`tiers` routes pin and
    /// stage reads through it); not part of the consumer contract.
    pub fn routeCopy(self: *const ExpertStore, layer_i: usize, eid: u32) usize {
        return policy.routeCopy(self, layer_i, eid);
    }

    /// Resolve every expert in `selected` (dupes fine — it is the raw routed
    /// pair list) for one layer op: pinned/cached experts resolve to their
    /// slabs; misses are read from disk into working-set slots, readahead
    /// hints for the whole miss set going out before the first synchronous
    /// read. Locks the store until `release`.
    pub fn acquire(self: *ExpertStore, layer_i: usize, selected: []const usize) Error!void {
        self.mutex.lock();
        errdefer self.mutex.unlock();
        std.debug.assert(self.acquired_layer == null);
        const rs = try self.acquireResolve(layer_i, selected);
        const ls = &self.layers[layer_i];
        const n_read = rs.n_read;
        try self.readMissSet(ls, layer_i, n_read);
        self.stats.bytes_read += @as(u64, n_read) * tiers.expertBytes(ls);
        for (self.read_eids[0..n_read], self.read_slots[0..n_read]) |eid, slot| {
            slot.eid = eid;
            self.resolveSlot(ls, eid, slot);
        }
        if (rs.read_start) |t0| {
            if (io.monotonicNanos()) |t1| self.stats.read_ns += t1 -| t0;
        }

        self.stats.acquires += 1;
        self.acquired_layer = layer_i;
    }

    const ResolveResult = struct { n_read: usize, read_start: ?u64 };

    /// Everything `acquire` does before the miss reads: trace, active-set
    /// bookkeeping, pilot scoring, hit resolution, readahead hints, staged
    /// consumption, and read-set assembly (slot capacity ensured). Caller
    /// holds `mutex`. `read_start` is captured at the same boundary the
    /// blocking path always measured from.
    fn acquireResolve(self: *ExpertStore, layer_i: usize, selected: []const usize) Error!ResolveResult {
        if (!self.finalized) return Error.StoreNotFinalized;
        if (layer_i >= self.layers.len or !self.registered[layer_i]) return Error.LayerNotRegistered;
        const ls = &self.layers[layer_i];

        // Trace what the MODEL asked for, before any cache outcome — a
        // replay at a different capacity is meaningless otherwise. Loss on
        // OOM is acceptable: the trace is diagnostic, serving is not.
        if (self.options.trace_path != null) {
            self.trace.ensureUnusedCapacity(self.allocator, 2 * selected.len) catch {};
            if (self.trace.unusedCapacitySlice().len >= 2 * selected.len) {
                for (selected) |e| {
                    self.trace.appendAssumeCapacity(@intCast(layer_i));
                    self.trace.appendAssumeCapacity(@intCast(e));
                }
            }
        }

        self.acquire_clock0 = self.clock;

        // Unique active set + usage/heat histograms (every routed pair
        // counts; `usage` persists across sessions, `heat` decays per
        // repin pass).
        @memset(self.seen[0..ls.n_expert], false);
        self.n_active = 0;
        for (selected) |e| {
            std.debug.assert(e < ls.n_expert);
            ls.usage[e] += 1;
            ls.heat[e] +|= 1;
            if (self.seen[e]) continue;
            self.seen[e] = true;
            self.active[self.n_active] = @intCast(e);
            self.n_active += 1;
        }

        // Score the pilot's prediction for this layer, once per prediction:
        // recall = predicted ∩ routed over routed (the measurement that says
        // whether router lookahead is worth its prefetch bandwidth).
        if (ls.pred_epoch != ls.pred_scored) {
            ls.pred_scored = ls.pred_epoch;
            for (self.active[0..self.n_active]) |eid| {
                if (ls.pred_marks[eid] == ls.pred_epoch) self.stats.pilot_recall_hits += 1;
            }
            self.stats.pilot_recall_total += self.n_active;
        }

        // Pinned and LRU hits resolve in place (pin first — a pinned expert
        // may transiently also sit in the LRU after a repin); misses collect.
        self.n_miss = 0;
        for (self.active[0..self.n_active]) |eid| {
            if (tiers.findPinned(ls, eid)) |slot| {
                self.resolveSlot(ls, eid, slot);
                self.stats.hits += 1;
                self.stats.pin_hits += 1;
            } else if (self.findCached(ls, eid)) |slot| {
                self.clock += 1;
                slot.used = self.clock;
                self.resolveSlot(ls, eid, slot);
                self.stats.hits += 1;
            } else {
                self.miss_eids[self.n_miss] = eid;
                self.n_miss += 1;
                self.stats.misses += 1;
            }
        }

        // Hint the whole miss set first: the kernel reads ahead while we
        // pread the earlier misses. L2-served experts are skipped — their
        // slab read never touches the primary, so a primary hint is pure
        // phantom readahead stealing the slow drive's bandwidth, and the
        // L2 fd is uncached (advise is a no-op there).
        if (self.options.readahead and self.n_miss > 1) {
            for (self.miss_eids[0..self.n_miss]) |eid| {
                if (tiers.l2Offset(self, layer_i, eid) != null) continue;
                const copy = self.routeCopy(layer_i, eid);
                for (&ls.projs) |*g| {
                    for (0..g.plane_count) |plane| io.hintWillNeed(self.copyFd(copy, g.part), g.planeFileOffset(eid, plane), g.plane_bytes);
                }
            }
        }

        const read_start = if (self.n_miss > 0) io.monotonicNanos() else null;
        // A staged load resolves its miss by slab swap; everything else
        // becomes the read set, capacity ensured here (one thread, so the
        // allocator is uncontended and an OOM surfaces before any I/O).
        var n_read: usize = 0;
        for (self.miss_eids[0..self.n_miss], 0..) |eid, w| {
            const slot = &self.work[w];
            if (tiers.stageConsume(self, layer_i, eid, slot)) {
                slot.eid = eid;
                self.resolveSlot(ls, eid, slot);
                continue;
            }
            try slot.ensureCapacity(self.allocator, ls.slab_bytes);
            // Not promotable until its read lands: an abandoned wave-split
            // acquire still releases (and promotes) — a stale eid here
            // would publish garbage bytes into the LRU.
            slot.eid = invalid_eid;
            self.read_eids[n_read] = eid;
            self.read_slots[n_read] = slot;
            n_read += 1;
        }
        return .{ .n_read = n_read, .read_start = read_start };
    }

    /// Wave-split acquire, phase 1 of 2 — decode-path overlap: resolve what
    /// is already resident, then LAUNCH the miss reads on the I/O pool and
    /// return without waiting, so the caller computes the resident experts
    /// inside the read shadow. Returns a residency bitmask over `selected`
    /// (bit i set = `selected[i]` resolved and computable now); a clear bit
    /// means `acquireFinish` must run before that expert is used. With no
    /// usable pool the reads happen synchronously here and the mask comes
    /// back all-resident. The store mutex is held from here to `release`,
    /// exactly like `acquire`; the workers never touch it.
    pub fn acquireStart(self: *ExpertStore, layer_i: usize, selected: []const usize) Error!u64 {
        std.debug.assert(selected.len <= 64);
        self.mutex.lock();
        errdefer self.mutex.unlock();
        std.debug.assert(self.acquired_layer == null);
        std.debug.assert(!self.miss_inflight);
        const rs = try self.acquireResolve(layer_i, selected);
        const ls = &self.layers[layer_i];
        const n_read = rs.n_read;
        self.stats.bytes_read += @as(u64, n_read) * tiers.expertBytes(ls);

        launch: {
            if (n_read == 0) break :launch;
            const striped = self.readSetPrepare(layer_i, n_read);
            // One worker per effective item — no `- 1` here, unlike the
            // blocking path: the caller computes instead of draining.
            const want = @min(self.options.io_workers, n_read + striped);
            if (want > 0) self.ioEnsurePool();
            const started = @min(self.io_ready, want);
            if (started == 0) {
                // Degraded (no pool): read synchronously, return all-resident.
                for (self.read_eids[0..n_read], self.read_slots[0..n_read]) |eid, slot| {
                    try self.readExpert(ls, layer_i, eid, slot);
                }
                for (self.read_eids[0..n_read], self.read_slots[0..n_read]) |eid, slot| {
                    slot.eid = eid;
                    self.resolveSlot(ls, eid, slot);
                }
                if (rs.read_start) |t0| {
                    if (io.monotonicNanos()) |t1| self.stats.read_ns += t1 -| t0;
                }
                break :launch;
            }
            @memset(self.read_failed[0..n_read], false);
            self.miss_batch = .{ .store = self, .ls = ls, .layer_i = layer_i, .count = n_read };
            for (self.io_pool[0..started]) |*w| {
                const ok = w.start(ioDrainJob, &self.miss_batch);
                std.debug.assert(ok); // dedicated pool: idle between batches
            }
            self.miss_started = started;
            self.miss_inflight = true;
        }

        self.stats.acquires += 1;
        self.acquired_layer = layer_i;
        var resident: u64 = 0;
        for (selected, 0..) |e, i| {
            if (ls.resolved[e][0] != null) resident |= @as(u64, 1) << @intCast(i);
        }
        return resident;
    }

    /// Wave-split acquire, phase 2: drain what the workers have not taken,
    /// join them, retry failed items synchronously (same healing as the
    /// blocking path), and resolve the freshly read experts. `read_ns`
    /// accounts only this blocked window — reads that completed under the
    /// caller's compute are exactly the overlap won. No-op when nothing is
    /// in flight (all-resident start, or degraded synchronous reads).
    pub fn acquireFinish(self: *ExpertStore) Error!void {
        const layer_i = self.acquired_layer.?;
        if (!self.miss_inflight) return;
        const ls = &self.layers[layer_i];
        const count = self.miss_batch.count;
        const t0 = io.monotonicNanos();
        self.miss_batch.drain();
        for (self.io_pool[0..self.miss_started]) |*w| w.wait();
        self.miss_inflight = false;
        self.miss_started = 0;
        for (0..count) |i| {
            if (!self.read_failed[i]) continue;
            try self.readExpert(ls, layer_i, self.read_eids[i], self.read_slots[i]);
        }
        for (self.read_eids[0..count], self.read_slots[0..count]) |eid, slot| {
            slot.eid = eid;
            self.resolveSlot(ls, eid, slot);
        }
        if (t0) |a| {
            if (io.monotonicNanos()) |b| self.stats.read_ns += b -| a;
        }
    }

    /// End the layer op: promote this acquire's misses into the layer LRU
    /// (slab swap — no copy; the displaced victim slab becomes the working
    /// buffer), clear the resolved pointers, unlock.
    pub fn release(self: *ExpertStore, layer_i: usize) void {
        std.debug.assert(self.acquired_layer == layer_i);
        // A failed op between `acquireStart` and `acquireFinish` lands here
        // with workers still writing slabs: join them before touching any
        // slot. The un-resolved read slots keep `invalid_eid`, so promoting
        // them below publishes nothing findable.
        if (self.miss_inflight) {
            self.miss_batch.drain();
            for (self.io_pool[0..self.miss_started]) |*w| w.wait();
            self.miss_inflight = false;
            self.miss_started = 0;
        }
        const ls = &self.layers[layer_i];

        const cap = ls.slots.len;
        const promo = @min(self.n_miss, cap);
        for (0..promo) |a| {
            const w = self.n_miss - 1 - a;
            const src = &self.work[w];
            // Abandoned wave-split acquire: this miss's read never landed
            // (`acquireFinish` did not run), so the slot still carries
            // `invalid_eid` and holds no expert. Promoting it would evict a
            // real cached expert for garbage — and the next release's heat
            // victim scan would index `heat[invalid_eid]` out of bounds.
            if (src.eid == invalid_eid) continue;
            var dst: *Slot = undefined;
            if (ls.n_slots < cap) {
                dst = &ls.slots[ls.n_slots];
                ls.n_slots += 1;
            } else if (self.options.heat_eviction and policy.envPlainLru() == false) {
                // Victim = coldest by decayed routed-pair count, recency
                // breaking ties — over slots NOT touched by this acquire
                // (see acquire_clock0). When the acquire fills the whole
                // cap, fall back to plain LRU among its own slots.
                var pick: ?*Slot = null;
                for (ls.slots[0..ls.n_slots]) |*s| {
                    if (s.used > self.acquire_clock0) continue;
                    const p = pick orelse {
                        pick = s;
                        continue;
                    };
                    const sh = ls.heat[s.eid];
                    const ph = ls.heat[p.eid];
                    if (sh < ph or (sh == ph and s.used < p.used)) pick = s;
                }
                dst = pick orelse blk: {
                    var lru = &ls.slots[0];
                    for (ls.slots[1..]) |*s| {
                        if (s.used < lru.used) lru = s;
                    }
                    break :blk lru;
                };
            } else {
                dst = &ls.slots[0];
                for (ls.slots[1..]) |*s| {
                    if (s.used < dst.used) dst = s;
                }
            }
            std.mem.swap([]align(slab_align) u8, &dst.slab, &src.slab);
            dst.eid = src.eid;
            src.eid = invalid_eid;
            self.clock += 1;
            dst.used = self.clock;
            // The promoted slab keeps resolving for this expert; pointers are
            // cleared below anyway.
        }

        for (self.active[0..self.n_active]) |eid| ls.resolved[eid] = .{ null, null, null };
        self.n_active = 0;
        self.n_miss = 0;
        self.acquired_layer = null;
        self.mutex.unlock();
    }

    /// Module-internal seam: the sibling concern files (`policy`) probe
    /// the LRU through the store. Not part of the consumer contract.
    pub fn findCached(self: *ExpertStore, ls: *LayerState, eid: u32) ?*Slot {
        _ = self;
        for (ls.slots[0..ls.n_slots]) |*slot| {
            if (slot.eid == eid) return slot;
        }
        return null;
    }

    fn resolveSlot(self: *ExpertStore, ls: *LayerState, eid: u32, slot: *Slot) void {
        _ = self;
        for (0..3) |p| ls.resolved[eid][p] = @ptrCast(slot.slab.ptr + ls.proj_off[p]);
    }

    /// One expert's gate+up+down blocks into `slot.slab` — one `pread` per
    /// projection per plane (the projections, and a PTQTP stack's plane
    /// siblings, are separate GGUF tensors, so they are not adjacent on
    /// disk; coalescing would need a converter-ordered container). A
    /// multi-plane projection's planes land contiguously in the slab
    /// section, which is the layout the MoE dispatch reads.
    /// Stats-free by design: called from the forward thread (under the
    /// store mutex) AND the prefetch worker (no store mutex) — each caller
    /// accounts bytes under its own lock (`bytes_read` / `staged_bytes`);
    /// the per-copy mirror counters and the L2 counters are atomic for the
    /// same reason.
    /// Serial both-phase read (retry, pilot staging, tier build, and
    /// single-threaded fallbacks): primary side then tier prefix. The
    /// batched miss path runs the two phases as SEPARATE work items so
    /// the slow and fast drives serve one expert concurrently.
    /// Module-internal seam: the pin loader, L2 builder, repin pass, and
    /// staging worker (`tiers`/`policy`) read experts through the store.
    /// Not part of the consumer contract.
    pub fn readExpert(self: *ExpertStore, ls: *LayerState, layer_i: usize, eid: u32, slot: *Slot) Error!void {
        try self.readExpertMain(ls, layer_i, eid, slot);
        try self.readExpertPrefix(ls, layer_i, eid, slot);
    }

    /// Primary-side phase: the whole slab when the expert is not striped,
    /// else only the suffix [prefix..slab_bytes) mapped back to the
    /// per-projection primary ranges (striped layers are never folded,
    /// so slab bytes ARE primary file bytes).
    fn readExpertMain(self: *ExpertStore, ls: *LayerState, layer_i: usize, eid: u32, slot: *Slot) Error!void {
        const copy = self.routeCopy(layer_i, eid);
        const prefix: usize = if (tiers.l2Offset(self, layer_i, eid) != null) @intCast(self.l2_prefix[layer_i]) else 0;
        // Slab-native record: the on-disk bytes ARE the slab — one
        // contiguous pread of [prefix..slab_bytes) (the tier phase serves
        // the prefix) instead of three per-projection reads.
        if (ls.slab_file_offset) |base| {
            if (prefix < ls.slab_bytes) {
                try self.preadFull(ls.slab_part, copy, slot.slab[prefix..ls.slab_bytes], base + @as(u64, eid) * ls.slab_bytes + prefix);
            }
            return;
        }
        for (&ls.projs, 0..) |*g, p| {
            if (g.fold) {
                std.debug.assert(prefix == 0);
                try self.readExpertFolded(g, eid, copy, slot.slab[ls.proj_off[p]..][0..g.expert_bytes]);
                continue;
            }
            for (0..g.plane_count) |plane| {
                const lo = ls.proj_off[p] + plane * g.plane_bytes;
                const hi = lo + g.plane_bytes;
                const start = @max(lo, prefix);
                if (start >= hi) continue;
                try self.preadFull(g.part, copy, slot.slab[start..hi], g.planeFileOffset(eid, plane) + (start - lo));
            }
        }
    }

    /// Tier-side phase: one sequential pread of the striped prefix; no-op
    /// when the expert is not covered. A tier read error falls back to
    /// reading the prefix range from the primary (the tier adds speed,
    /// never correctness).
    fn readExpertPrefix(self: *ExpertStore, ls: *LayerState, layer_i: usize, eid: u32, slot: *Slot) Error!void {
        const l2_off = tiers.l2Offset(self, layer_i, eid) orelse return;
        const prefix: usize = @intCast(self.l2_prefix[layer_i]);
        if (prefix == 0) return;
        const t0 = io.monotonicNanos();
        if (io.preadFullFd(self.l2_fds[0], slot.slab[0..prefix], l2_off)) {
            _ = self.l2_expert_hits.fetchAdd(1, .monotonic);
            _ = self.l2_bytes.fetchAdd(prefix, .monotonic);
            if (t0) |t| if (io.monotonicNanos()) |t1| {
                _ = self.l2_read_ns.fetchAdd(t1 -| t, .monotonic);
            };
            return;
        } else |_| {
            _ = self.l2_fallbacks.fetchAdd(1, .monotonic);
        }
        const copy = self.routeCopy(layer_i, eid);
        if (ls.slab_file_offset) |base| {
            try self.preadFull(ls.slab_part, copy, slot.slab[0..prefix], base + @as(u64, eid) * ls.slab_bytes);
            return;
        }
        for (&ls.projs, 0..) |*g, p| {
            for (0..g.plane_count) |plane| {
                const lo = ls.proj_off[p] + plane * g.plane_bytes;
                if (lo >= prefix) return;
                const hi = @min(lo + g.plane_bytes, prefix);
                try self.preadFull(g.part, copy, slot.slab[lo..hi], g.planeFileOffset(eid, plane));
            }
        }
    }

    /// Folded fill: both plane row-blocks bounce through a scratch, then
    /// fold into the section start as the 4-bit pack. The bounce is not
    /// optional — the pack is written faster than the plane bytes are
    /// consumed (520 pack bytes vs 264 plane-0 bytes per 4-column group),
    /// so an in-place fold would overwrite unread source blocks. The
    /// per-call alloc rides on the same thread-safe-allocator assumption
    /// the pilot worker's `ensureCapacity` already makes, and is noise
    /// next to the two disk reads it accompanies.
    fn readExpertFolded(self: *ExpertStore, g: *const ProjGeometry, eid: u32, copy: usize, section: []u8) Error!void {
        const plane_blocks = g.out_dim * g.blocks_per_column;
        const scratch = try self.allocator.alloc(dtype_mod.BlockTQ2_0, 2 * plane_blocks);
        defer self.allocator.free(scratch);
        for (0..2) |plane| {
            const dst = std.mem.sliceAsBytes(scratch[plane * plane_blocks ..][0..plane_blocks]);
            try self.preadFull(g.part, copy, dst, g.planeFileOffset(eid, plane));
        }
        var views: [2]backend_mod.quant.types.QuantizedMatmulRhsTQ2_0 = undefined;
        for (0..2) |plane| {
            views[plane] = .{
                .allocator = null,
                .blocks = scratch[plane * plane_blocks ..][0..plane_blocks],
                .blocks_per_column = g.blocks_per_column,
                .k = g.in_dim,
                .n = g.out_dim,
            };
        }
        const fg = (g.out_dim / 4) * g.blocks_per_column;
        const out = @as([*]qm.types.BlockTQ2_0Foldedx4, @ptrCast(@alignCast(section.ptr)))[0..fg];
        backend_mod.kernels.packMatmulRhsTQ2_0Foldedx4Into(out, &views[0], &views[1]) catch return Error.InvalidExpertGeometry;
    }

    /// Module-internal seam (`tiers` hints and reads through the routed
    /// copy's fd); not part of the consumer contract.
    pub fn copyFd(self: *const ExpertStore, copy: usize, part: u16) fd_t {
        return if (copy == 0) self.fds[part] else self.mirrors[copy - 1].fds[part];
    }

    /// Full positional read from the routed copy, falling back to the
    /// primary when a mirror errors mid-read: a flaky or unmounted mirror
    /// must never kill decode while the primary holds the same bytes.
    fn preadFull(self: *ExpertStore, part: u16, copy: usize, buf: []u8, offset: u64) Error!void {
        if (copy != 0) {
            if (io.preadFullFd(self.copyFd(copy, part), buf, offset)) {
                _ = self.copy_bytes[copy].fetchAdd(buf.len, .monotonic);
                return;
            } else |_| {
                _ = self.mirror_fallbacks.fetchAdd(1, .monotonic);
            }
        }
        try io.preadFullFd(self.fds[part], buf, offset);
        if (self.copy_bytes.len > 0) _ = self.copy_bytes[0].fetchAdd(buf.len, .monotonic);
    }

    // ---- demand-miss I/O pool ------------------------------------------

    /// One acquire's read fan-out: workers and the acquiring thread pull
    /// indices off `next`; the parallel arrays live on the store
    /// (`read_eids`/`read_slots`/`read_failed`). Each index is processed
    /// by exactly one thread, so the plain `read_failed` bools are
    /// race-free; they become visible to the caller through the workers'
    /// join in `readMissSet`.
    const IoBatch = struct {
        store: *ExpertStore,
        ls: *LayerState,
        layer_i: usize,
        count: usize,
        next: std.atomic.Value(usize) = .init(0),

        // Virtual items over 2*count: i < count = the expert's primary-side
        // read (full slab, or the USB suffix when striped); i >= count =
        // the striped expert's fast-tier prefix (no-op when unstriped).
        // The phases write disjoint slab ranges, so both drives serve ONE
        // miss concurrently — primary items first so the slower drive is
        // saturated from the batch's first moment.
        fn drain(self: *IoBatch) void {
            while (true) {
                const i = self.next.fetchAdd(1, .monotonic);
                if (i >= 2 * self.count) return;
                if (i < self.count) {
                    self.store.readExpertMain(self.ls, self.layer_i, self.store.read_eids[i], self.store.read_slots[i]) catch {
                        self.store.read_failed[i] = true;
                    };
                } else {
                    const j = i - self.count;
                    self.store.readExpertPrefix(self.ls, self.layer_i, self.store.read_eids[j], self.store.read_slots[j]) catch {
                        self.store.read_failed[j] = true;
                    };
                }
            }
        }
    };

    fn ioDrainJob(arg: *anyopaque) void {
        io.setIoThreadQos();
        const batch: *IoBatch = @ptrCast(@alignCast(arg));
        batch.drain();
    }

    /// Sort the read set into disk order and count the striped experts.
    /// Issue in DISK order: expert file offsets are monotone in eid within
    /// each projection, so an ascending-eid batch turns scattered seeks
    /// into (per-projection) forward sweeps the drive can merge. Striped
    /// experts contribute a second (fast-tier) work item each — the count
    /// lets a single striped miss still get a worker for the concurrent
    /// prefix read.
    fn readSetPrepare(self: *ExpertStore, layer_i: usize, count: usize) usize {
        if (count > 1) {
            const Ctx = struct {
                eids: []u32,
                slots: []*Slot,
                pub fn lessThan(c: @This(), a: usize, b: usize) bool {
                    return c.eids[a] < c.eids[b];
                }
                pub fn swap(c: @This(), a: usize, b: usize) void {
                    std.mem.swap(u32, &c.eids[a], &c.eids[b]);
                    std.mem.swap(*Slot, &c.slots[a], &c.slots[b]);
                }
            };
            std.mem.sortUnstableContext(0, count, Ctx{
                .eids = self.read_eids[0..count],
                .slots = self.read_slots[0..count],
            });
        }
        var striped: usize = 0;
        for (self.read_eids[0..count]) |eid| {
            if (tiers.l2Offset(self, layer_i, eid) != null) striped += 1;
        }
        return striped;
    }

    /// Read the collected miss set, fanning out across the I/O pool when
    /// it pays (two or more reads): disk queue depth — and mirror copies
    /// on other drives — then genuinely serve in parallel, which one
    /// synchronous pread loop never achieves. Slot capacities are already
    /// ensured, so workers only pread into disjoint slabs. Failed items
    /// are retried synchronously here.
    fn readMissSet(self: *ExpertStore, ls: *LayerState, layer_i: usize, count: usize) Error!void {
        if (count == 0) return;
        const striped = self.readSetPrepare(layer_i, count);
        const want = @min(self.options.io_workers, count + striped - 1);
        if (want > 0) self.ioEnsurePool();
        const started = @min(self.io_ready, want);
        if (started == 0) {
            for (self.read_eids[0..count], self.read_slots[0..count]) |eid, slot| {
                try self.readExpert(ls, layer_i, eid, slot);
            }
            return;
        }
        @memset(self.read_failed[0..count], false);
        var batch: IoBatch = .{ .store = self, .ls = ls, .layer_i = layer_i, .count = count };
        for (self.io_pool[0..started]) |*w| {
            const ok = w.start(ioDrainJob, &batch);
            std.debug.assert(ok); // dedicated pool: idle between batches
        }
        batch.drain();
        for (self.io_pool[0..started]) |*w| w.wait();
        for (0..count) |i| {
            if (!self.read_failed[i]) continue;
            try self.readExpert(ls, layer_i, self.read_eids[i], self.read_slots[i]);
        }
    }

    /// Bring up the worker pool (idempotent, lazy): a spawn failure just
    /// leaves a smaller — possibly empty — pool, and the batch still
    /// drains on the acquiring thread.
    fn ioEnsurePool(self: *ExpertStore) void {
        if (self.io_pool.len == 0) {
            self.io_pool = self.allocator.alloc(thread.OneShotWorker, self.options.io_workers) catch return;
            self.io_ready = 0;
        }
        while (self.io_ready < self.io_pool.len) {
            self.io_pool[self.io_ready].init() catch break;
            self.io_ready += 1;
        }
    }

    // ---- pilot: router-lookahead prefetch ----------------------------------

    /// Predicted routing for `layer_i`'s NEXT acquire (router lookahead:
    /// apply layer L+1's router to layer L's post-attention state — measured
    /// 87.6-90.5% top-8 recall on the Qwen3 MoEs vs ~41% for "same as last
    /// token" upstream). Marks the prediction for recall scoring and enqueues
    /// readahead for the experts not already pinned or cached; the dedicated
    /// I/O thread issues the actual advice. Call between ops on the forward
    /// thread — never between `acquire` and `release`.
    pub fn pilotHint(self: *ExpertStore, layer_i: usize, eids: []const usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        std.debug.assert(self.acquired_layer == null);
        if (!self.finalized or layer_i >= self.layers.len or !self.registered[layer_i]) return;
        const ls = &self.layers[layer_i];

        if (self.pilot_thread == null) tiers.pilotStart(self) catch return;

        ls.pred_epoch +%= 1;
        if (ls.pred_epoch == ls.pred_scored) ls.pred_epoch +%= 1; // skip the ambiguous wrap value
        for (eids) |e| {
            if (e >= ls.n_expert) continue;
            ls.pred_marks[e] = ls.pred_epoch;
            const eid: u32 = @intCast(e);
            if (tiers.findPinned(ls, eid) != null) continue;
            if (self.findCached(ls, eid) != null) continue;
            tiers.pilotEnqueue(self, .{ .layer_i = @intCast(layer_i), .eid = eid });
        }
    }

    /// True if the expert is pinned or currently cached — attending to it
    /// costs no disk read. Routing-time oracle for cache-aware expert
    /// dropping; call between ops (never while an acquire is open — the
    /// store mutex is not reentrant).
    pub fn isResident(self: *ExpertStore, layer_i: usize, eid: usize) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (!self.finalized or layer_i >= self.layers.len or !self.registered[layer_i]) return false;
        const ls = &self.layers[layer_i];
        if (eid >= ls.n_expert) return false;
        const e: u32 = @intCast(eid);
        return tiers.findPinned(ls, e) != null or self.findCached(ls, e) != null;
    }
};

/// The streamed counterpart of a resident stacked-expert RHS: same geometry,
/// but expert blocks resolve through the store's acquire-scoped pointer
/// table instead of a slice into one big buffer.
pub const StreamedMoeRhs = struct {
    store: *ExpertStore,
    layer_state: *LayerState,
    layer: usize,
    proj: Proj,
    quant: StreamedQuant,
    k: usize,
    out_dim: usize,
    n_expert: usize,
    blocks_per_column: usize,
    /// Trit-plane count (PTQTP stacks; 1 otherwise): expert `e`'s resolved
    /// slab section holds `plane_count` same-geometry plane row-blocks
    /// back to back. Geometry accessors stay per-plane, mirroring the
    /// resident `ptqtp` arm.
    plane_count: usize = 1,
    /// Tie-fitted K=2 stream (ProjGeometry.fold): the resolved section
    /// holds the 4-bit folded pack instead of plane row-blocks, and the
    /// MoE dispatch serves the one-pass folded kernel.
    folded: bool = false,

    /// Virtual stacked row count, mirroring the resident arms' `n`.
    pub fn rows(self: *const StreamedMoeRhs) usize {
        return self.n_expert * self.out_dim;
    }

    /// Expert `e`'s blocks for this projection. Only valid between the
    /// store's `acquire` (which resolved `e`) and `release`.
    pub inline fn expertBytes(self: *const StreamedMoeRhs, e: usize) [*]const u8 {
        const ptr = self.layer_state.resolved[e][@intFromEnum(self.proj)];
        if (ptr) |p| return p;
        @panic("streamed MoE expert used without acquire");
    }
};

test {
    _ = @import("expert_store_tests.zig");
}
