//! Disk-streamed MoE expert tier: resolves one layer's routed experts to
//! their quantized weight blocks through a pinned set, a per-layer LRU cache
//! in RAM, and `pread` from the GGUF file — so a mixture model whose expert
//! stacks dwarf physical RAM still decodes, paying disk reads only for cache
//! misses. The design follows the measured lessons of out-of-core MoE
//! engines (colibri): the streamed tier reads with `pread` into store-owned
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
//! nothing evicts or swaps slots until `release`.
const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("../backend.zig");
const thread = @import("../thread.zig");

const Allocator = std.mem.Allocator;
const qm = backend_mod.quantized_matmul;

pub const Error = error{
    LayerAlreadyRegistered,
    LayerNotRegistered,
    InvalidExpertGeometry,
    StoreNotFinalized,
    ExpertFileOpenFailed,
    ExpertFileReadFailed,
    UnexpectedEndOfFile,
} || Allocator.Error;

// ---- platform I/O shims -------------------------------------------------
// The streamed tier wants positional reads and readahead advice without an
// `std.Io` handle (the MoE ops run with `io == null` in production decode).
// Linux goes straight to the syscall layer (no libc requirement); everything
// else uses libc, which is always linked on the Apple targets.

const fd_t = std.posix.fd_t;

fn openReadOnly(allocator: Allocator, path: []const u8) Error!fd_t {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const rc = linux.openat(linux.AT.FDCWD, path_z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
            if (linux.errno(rc) != .SUCCESS) return Error.ExpertFileOpenFailed;
            return @intCast(rc);
        },
        else => {
            const rc = std.c.open(path_z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
            if (rc < 0) return Error.ExpertFileOpenFailed;
            return rc;
        },
    }
}

fn closeFd(fd: fd_t) void {
    switch (builtin.os.tag) {
        .linux => _ = std.os.linux.close(fd),
        else => _ = std.c.close(fd),
    }
}

/// Monotonic nanoseconds for the read-time stat (the exec ops' clock rides
/// on `std.Io`, which the MoE decode path legitimately runs without).
fn monotonicNanos() ?u64 {
    switch (builtin.os.tag) {
        .linux => {
            var ts: std.os.linux.timespec = undefined;
            if (std.os.linux.clock_gettime(.MONOTONIC, &ts) != 0) return null;
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        else => {
            var ts: std.c.timespec = undefined;
            if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return null;
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
    }
}

fn preadOnce(fd: fd_t, buf: []u8, offset: u64) Error!usize {
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const rc = linux.pread(fd, buf.ptr, buf.len, @intCast(offset));
            if (linux.errno(rc) != .SUCCESS) return Error.ExpertFileReadFailed;
            return rc;
        },
        else => {
            const rc = std.c.pread(fd, buf.ptr, buf.len, @intCast(offset));
            if (rc < 0) return Error.ExpertFileReadFailed;
            return @intCast(rc);
        },
    }
}

/// Quantized formats an expert stack may stream in — the K-quant family
/// every real MoE GGUF uses (matching `MoeRhs`'s resident arms).
pub const StreamedQuant = enum {
    q4_k,
    q5_k,
    q6_k,

    pub fn blockSize(self: StreamedQuant) usize {
        return switch (self) {
            .q4_k => @sizeOf(qm.BlockQ4_K),
            .q5_k => @sizeOf(qm.BlockQ5_K),
            .q6_k => @sizeOf(qm.BlockQ6_K),
        };
    }
};

/// The three projections of one MoE FFN layer, in slab order.
pub const Proj = enum(u2) { gate = 0, up = 1, down = 2 };

/// One projection's stacked expert tensor as it sits in the GGUF file:
/// expert-major contiguous, so expert `e` occupies the byte range
/// `[file_offset + e*expert_bytes, +expert_bytes)`.
pub const ProjSpec = struct {
    quant: StreamedQuant,
    /// Absolute offset of the tensor's data in the file
    /// (`gguf.File.data_offset + TensorInfo.offset`).
    file_offset: u64,
    /// Total tensor bytes (validated against `n_expert * expert_bytes`).
    byte_len: usize,
    in_dim: usize,
    out_dim: usize,
};

const ProjGeometry = struct {
    quant: StreamedQuant,
    file_offset: u64,
    in_dim: usize,
    out_dim: usize,
    blocks_per_column: usize,
    expert_bytes: usize,

    fn init(spec: ProjSpec, n_expert: usize) Error!ProjGeometry {
        const bpc = qm.qkBlockCount(spec.in_dim) catch return Error.InvalidExpertGeometry;
        const row_bytes = std.math.mul(usize, bpc, spec.quant.blockSize()) catch return Error.InvalidExpertGeometry;
        const expert_bytes = std.math.mul(usize, spec.out_dim, row_bytes) catch return Error.InvalidExpertGeometry;
        const total = std.math.mul(usize, expert_bytes, n_expert) catch return Error.InvalidExpertGeometry;
        if (total != spec.byte_len or expert_bytes == 0) return Error.InvalidExpertGeometry;
        return .{
            .quant = spec.quant,
            .file_offset = spec.file_offset,
            .in_dim = spec.in_dim,
            .out_dim = spec.out_dim,
            .blocks_per_column = bpc,
            .expert_bytes = expert_bytes,
        };
    }

    fn expertFileOffset(self: *const ProjGeometry, eid: usize) u64 {
        return self.file_offset + @as(u64, eid) * self.expert_bytes;
    }
};

/// Block slabs start 64-byte aligned so the widest kernel loads stay aligned
/// regardless of the K-quant block's own (2-byte) alignment.
const slab_align = 64;
const invalid_eid = std.math.maxInt(u32);

/// One cached expert: a single slab holding its gate+up+down blocks (loaded
/// with one logical fetch, like colibri's coalesced expert read), stamped for
/// LRU. Slots inside a layer all share that layer's slab size; the shared
/// working-set slots are re-checked per use because layers may differ.
const Slot = struct {
    eid: u32 = invalid_eid,
    used: u64 = 0,
    slab: []align(slab_align) u8 = &.{},

    fn ensureCapacity(self: *Slot, allocator: Allocator, bytes: usize) !void {
        if (self.slab.len >= bytes) return;
        if (self.slab.len > 0) allocator.free(self.slab);
        self.slab = &.{};
        self.slab = try allocator.alignedAlloc(u8, comptime std.mem.Alignment.fromByteUnits(slab_align), bytes);
    }

    fn deinit(self: *Slot, allocator: Allocator) void {
        if (self.slab.len > 0) allocator.free(self.slab);
        self.* = undefined;
    }
};

pub const LayerState = struct {
    projs: [3]ProjGeometry,
    n_expert: usize,
    /// Per-expert slab layout: projection `p`'s blocks start at `proj_off[p]`.
    proj_off: [3]usize,
    slab_bytes: usize,
    /// LRU tier (`cap` entries; slabs allocated on first promotion).
    slots: []Slot = &.{},
    n_slots: usize = 0,
    /// Resolved pointer per (expert, projection), valid between acquire and
    /// release for the acquired experts only.
    resolved: [][3]?[*]const u8 = &.{},
    /// Persistent routing histogram (one count per routed pair), the raw
    /// signal for usage-driven pinning.
    usage: []u64 = &.{},

    fn expertSlabOffsets(projs: [3]ProjGeometry) struct { off: [3]usize, total: usize } {
        var off: [3]usize = undefined;
        var at: usize = 0;
        for (projs, 0..) |g, p| {
            at = std.mem.alignForward(usize, at, slab_align);
            off[p] = at;
            at += g.expert_bytes;
        }
        return .{ .off = off, .total = std.mem.alignForward(usize, at, slab_align) };
    }
};

/// Best-effort async readahead hint for a file range: tells the kernel to
/// start reading in the background so the synchronous `pread` that follows
/// (or follows several other hinted reads) finds the page cache warm. A
/// failed hint is not an error — it only costs the overlap.
fn hintWillNeed(fd: fd_t, offset: u64, len: usize) void {
    switch (builtin.os.tag) {
        .linux => {
            _ = std.os.linux.fadvise(fd, @intCast(offset), @intCast(len), std.os.linux.POSIX_FADV.WILLNEED);
        },
        .macos, .ios => {
            // struct radvisory + F_RDADVISE (fcntl.h): macOS's readahead advice.
            const radvisory = extern struct { ra_offset: i64, ra_count: c_int };
            var adv = radvisory{
                .ra_offset = @intCast(offset),
                .ra_count = @intCast(@min(len, std.math.maxInt(c_int))),
            };
            _ = std.c.fcntl(fd, 44, @intFromPtr(&adv));
        },
        else => {},
    }
}

/// Currently-available physical memory, best effort: Linux reads
/// `MemAvailable` (free + reclaimable page cache); macOS counts only the
/// free-page pool, which understates what the OS could reclaim — acceptable
/// for a conservative default budget. `null` when undeterminable.
pub fn memAvailableBytes() ?u64 {
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const open_rc = linux.openat(linux.AT.FDCWD, "/proc/meminfo", .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
            if (linux.errno(open_rc) != .SUCCESS) return null;
            const fd: fd_t = @intCast(open_rc);
            defer closeFd(fd);
            var buf: [4096]u8 = undefined;
            const read_rc = linux.read(fd, &buf, buf.len);
            if (linux.errno(read_rc) != .SUCCESS) return null;
            const n = read_rc;
            var it = std.mem.splitScalar(u8, buf[0..n], '\n');
            while (it.next()) |line| {
                const prefix = "MemAvailable:";
                if (!std.mem.startsWith(u8, line, prefix)) continue;
                const rest = std.mem.trim(u8, line[prefix.len..], " ");
                const end = std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len;
                const kb = std.fmt.parseInt(u64, rest[0..end], 10) catch return null;
                return kb * 1024;
            }
            return null;
        },
        .macos => {
            var pages: c_int = 0;
            var page_size: c_int = 0;
            var len: usize = @sizeOf(c_int);
            if (std.c.sysctlbyname("vm.page_free_count", &pages, &len, null, 0) != 0) return null;
            len = @sizeOf(c_int);
            if (std.c.sysctlbyname("hw.pagesize", &page_size, &len, null, 0) != 0) return null;
            if (pages < 0 or page_size <= 0) return null;
            return @as(u64, @intCast(pages)) * @as(u64, @intCast(page_size));
        },
        else => return null,
    }
}

pub const ExpertStore = struct {
    pub const Options = struct {
        /// Fixed LRU slots per layer; wins over `cache_bytes` when set.
        cache_slots_per_layer: ?usize = null,
        /// Total RAM budget for the LRU tier across all layers. Default:
        /// half of the available memory at finalize time (8 GiB fallback).
        cache_bytes: ?usize = null,
        /// Issue WILLNEED-style readahead for the whole miss set before the
        /// first synchronous read.
        readahead: bool = true,
    };

    pub const Stats = struct {
        hits: u64 = 0,
        misses: u64 = 0,
        bytes_read: u64 = 0,
        read_ns: u64 = 0,
        acquires: u64 = 0,

        pub fn hitRate(self: Stats) f64 {
            const total = self.hits + self.misses;
            if (total == 0) return 0;
            return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
        }
    };

    allocator: Allocator,
    fd: fd_t,
    layers: []LayerState,
    registered: []bool,
    options: Options,
    /// LRU slots per layer, fixed at `finalize`.
    cap: usize = 0,
    finalized: bool = false,
    clock: u64 = 0,
    stats: Stats = .{},
    mutex: thread.Mutex = .{},
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

    /// The store is heap-allocated so `StreamedMoeRhs` values and the owning
    /// model can hold stable pointers while the model struct moves by value.
    pub fn create(allocator: Allocator, gguf_path: []const u8, n_layers: usize, options: Options) Error!*ExpertStore {
        const fd = try openReadOnly(allocator, gguf_path);
        errdefer closeFd(fd);

        const self = try allocator.create(ExpertStore);
        errdefer allocator.destroy(self);
        const layers = try allocator.alloc(LayerState, n_layers);
        errdefer allocator.free(layers);
        const registered = try allocator.alloc(bool, n_layers);
        @memset(registered, false);

        self.* = .{
            .allocator = allocator,
            .fd = fd,
            .layers = layers,
            .registered = registered,
            .options = options,
        };
        return self;
    }

    pub fn destroy(self: *ExpertStore) void {
        const allocator = self.allocator;
        for (self.layers, self.registered) |*ls, reg| {
            if (!reg) continue;
            for (ls.slots) |*slot| slot.deinit(allocator);
            allocator.free(ls.slots);
            allocator.free(ls.resolved);
            allocator.free(ls.usage);
        }
        for (self.work) |*slot| slot.deinit(allocator);
        allocator.free(self.work);
        allocator.free(self.active);
        allocator.free(self.miss_eids);
        allocator.free(self.seen);
        allocator.free(self.layers);
        allocator.free(self.registered);
        closeFd(self.fd);
        allocator.destroy(self);
    }

    /// Register one MoE layer's three stacked expert tensors. Call once per
    /// layer during model load, then `finalize`.
    pub fn addLayer(self: *ExpertStore, layer_i: usize, specs: [3]ProjSpec, n_expert: usize) Error!void {
        if (layer_i >= self.layers.len) return Error.InvalidExpertGeometry;
        if (self.registered[layer_i]) return Error.LayerAlreadyRegistered;
        if (n_expert == 0 or n_expert >= invalid_eid) return Error.InvalidExpertGeometry;

        var projs: [3]ProjGeometry = undefined;
        for (specs, 0..) |spec, p| projs[p] = try ProjGeometry.init(spec, n_expert);
        const layout = LayerState.expertSlabOffsets(projs);

        const resolved = try self.allocator.alloc([3]?[*]const u8, n_expert);
        errdefer self.allocator.free(resolved);
        @memset(resolved, .{ null, null, null });
        const usage = try self.allocator.alloc(u64, n_expert);
        @memset(usage, 0);

        self.layers[layer_i] = .{
            .projs = projs,
            .n_expert = n_expert,
            .proj_off = layout.off,
            .slab_bytes = layout.total,
            .resolved = resolved,
            .usage = usage,
        };
        self.registered[layer_i] = true;
    }

    /// Fix the LRU capacity from the configured budget and allocate the
    /// bookkeeping sized to the largest registered layer. Slot slabs
    /// themselves are allocated lazily on first promotion, so an unused
    /// budget costs nothing.
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

        var cap: usize = undefined;
        if (self.options.cache_slots_per_layer) |slots| {
            cap = @max(1, slots);
        } else {
            const budget = self.options.cache_bytes orelse blk: {
                const avail = memAvailableBytes() orelse (8 << 30);
                break :blk @max(avail / 2, 512 << 20);
            };
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
        };
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
        if (!self.finalized) return Error.StoreNotFinalized;
        if (layer_i >= self.layers.len or !self.registered[layer_i]) return Error.LayerNotRegistered;
        const ls = &self.layers[layer_i];

        // Unique active set + usage histogram (every routed pair counts).
        @memset(self.seen[0..ls.n_expert], false);
        self.n_active = 0;
        for (selected) |e| {
            std.debug.assert(e < ls.n_expert);
            ls.usage[e] += 1;
            if (self.seen[e]) continue;
            self.seen[e] = true;
            self.active[self.n_active] = @intCast(e);
            self.n_active += 1;
        }

        // Hits resolve in place and re-stamp; misses collect.
        self.n_miss = 0;
        for (self.active[0..self.n_active]) |eid| {
            if (self.findCached(ls, eid)) |slot| {
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
        // pread the earlier misses.
        if (self.options.readahead and self.n_miss > 1) {
            for (self.miss_eids[0..self.n_miss]) |eid| {
                for (&ls.projs) |*g| hintWillNeed(self.fd, g.expertFileOffset(eid), g.expert_bytes);
            }
        }

        const read_start = if (self.n_miss > 0) monotonicNanos() else null;
        for (self.miss_eids[0..self.n_miss], 0..) |eid, w| {
            const slot = &self.work[w];
            try slot.ensureCapacity(self.allocator, ls.slab_bytes);
            try self.readExpert(ls, eid, slot);
            slot.eid = eid;
            self.resolveSlot(ls, eid, slot);
        }
        if (read_start) |t0| {
            if (monotonicNanos()) |t1| self.stats.read_ns += t1 -| t0;
        }

        self.stats.acquires += 1;
        self.acquired_layer = layer_i;
    }

    /// End the layer op: promote this acquire's misses into the layer LRU
    /// (slab swap — no copy; the displaced victim slab becomes the working
    /// buffer), clear the resolved pointers, unlock.
    pub fn release(self: *ExpertStore, layer_i: usize) void {
        std.debug.assert(self.acquired_layer == layer_i);
        const ls = &self.layers[layer_i];

        const cap = ls.slots.len;
        const promo = @min(self.n_miss, cap);
        for (0..promo) |a| {
            const w = self.n_miss - 1 - a;
            const src = &self.work[w];
            var dst: *Slot = undefined;
            if (ls.n_slots < cap) {
                dst = &ls.slots[ls.n_slots];
                ls.n_slots += 1;
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

    fn findCached(self: *ExpertStore, ls: *LayerState, eid: u32) ?*Slot {
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

    /// One expert's gate+up+down blocks into `slot.slab` — three `pread`s
    /// (the projections are separate GGUF tensors, so they are not adjacent
    /// on disk; coalescing would need a converter-ordered container).
    fn readExpert(self: *ExpertStore, ls: *LayerState, eid: u32, slot: *Slot) Error!void {
        for (&ls.projs, 0..) |*g, p| {
            const dst = slot.slab[ls.proj_off[p]..][0..g.expert_bytes];
            try self.preadFull(dst, g.expertFileOffset(eid));
            self.stats.bytes_read += g.expert_bytes;
        }
    }

    fn preadFull(self: *ExpertStore, buf: []u8, offset: u64) Error!void {
        var done: usize = 0;
        while (done < buf.len) {
            const n = try preadOnce(self.fd, buf[done..], offset + done);
            if (n == 0) return Error.UnexpectedEndOfFile;
            done += n;
        }
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
