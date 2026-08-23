//! SCSA: self-calibrating subquadratic attention. A training-free decode
//! attention operator replacing softmax(QK^T)V per (layer, kv head) with a
//! statistically contracted sparse evaluation (docs/SUBQUADRATIC-ATTENTION.md;
//! greedy agreement with dense is calibrated, typically 94-99%, NOT exact).
//!
//! Structure per (layer, kv head): a sealed plan over cache positions
//! [sink, seal_end): semantic clusters (deterministic farthest-direction
//! splits at `cluster_size` granularity) with per-cluster summaries (count,
//! key centroid, value mean, rank-r covariance eigenpairs, diagonal
//! residual) and member keys/values packed contiguously per cluster (q8_0
//! by default, f16 fallback). Plans are maintained INCREMENTALLY with
//! frozen blocks: only the growing tail region is re-split every
//! `rebuild_interval` appends, and regions freeze permanently at
//! `block_size` rows; sinks and the unsealed suffix are always exact.
//!
//! Per decode query: cluster priorities `a + log m + b^2 q^T Sigma q / 2`
//! for all heads via per-layer GEMMs over concatenated summaries; the
//! ESTIMATE-ONLY stop decides the opened set before any read (open
//! best-first until the estimated uncaptured-mass fraction clears the
//! per-head self-calibrated tau); opened clusters are read as coalesced
//! ascending streams on catalog kernels; every unopened cluster contributes
//! its zeroth-order term `m * exp(a)` at the cluster value mean inside the
//! shared softmax (nothing is dropped). Self-calibration freezes per-head
//! taus from error curves recorded while producing exact outputs during the
//! first decode steps.
//!
//! `hierarchical` switches selection to an interval-tree frontier walk with
//! aggregated node summaries and frontier-granular tails: parity at <=32K,
//! constant selection cost at very long contexts (the long-context mode).

const std = @import("std");
const fucina = @import("fucina");
const runner_mod = @import("runner.zig");
const kv_cache_mod = @import("kv_cache.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const Tensor = fucina.internal.tensor_mod.Tensor;
const qkern = fucina.internal.backend_mod.quantized_matmul;
const BlockQ8_0 = fucina.quant.BlockQ8_0;
const q8_block = 32;

/// Storage format of the packed per-cluster K/V copies. The sparse arm is
/// bandwidth-bound on these reads, so quantized packing is a direct speed
/// lever; per-format score/accumulate kernels come from the backend catalog
/// (integer sdot path for q8_0, the query quantized once per head call).
/// Self-calibration measures its error curves through the packed
/// representation, so tau absorbs the quantization error automatically.
pub const PackedFormat = enum { f16, q8_0 };

/// Threshold grid swept during self-calibration, largest (cheapest) first.
pub const calib_grid = [_]f32{ 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01 };

pub const Config = struct {
    /// 128 measured best at 8K on 0.6B (0.946x vs 0.886x at 64: longer
    /// coalesced streams, half the scoring/sort work; 256 loses agreement).
    cluster_size: usize = 128,
    rebuild_interval: usize = 512,
    /// Frozen-block incremental maintenance: once the growing tail region
    /// reaches this many rows it freezes forever (clusters, summaries, and
    /// packed copies immutable), so rebuild cost is O(block) not O(context).
    /// Multipole Attention's W-block ablation motivates the default scale.
    block_size: usize = 4096,
    sink: usize = 4,
    recent: usize = 32,
    tau_default: f32 = 0.05,
    /// q8_0 measured best on 0.6B/8K+32K (2026-08-15 A/B vs f16: 0.947x vs
    /// 0.844x at 8K, 1.69x vs 1.36x at 32K, identical 94/96 agreement).
    /// tied-PTQTP K2 measured DEAD for K/V copies (78% agreement at 8K,
    /// 46% at 32K, taus max out opening ~everything): ~9 ternary levels are
    /// below the fidelity floor for attention rows, confirming the earlier
    /// ternary-KV falsification in the cluster-copy regime; kept only as a
    /// format-seam reference implementation.
    packed_format: PackedFormat = .q8_0,
    rank: usize = 2,
    power_iterations: usize = 4,
    /// Hierarchical frontier: selection walks a balanced interval tree over
    /// the leaf clusters (internal nodes carry aggregated count/centroid/
    /// value-mean/diagonal-variance summaries), expanding the highest
    /// estimated-mass node until the estimate-only stop clears tau; the
    /// decided leaf set is then read with the same coalesced sweep as the
    /// flat path, and unexpanded nodes contribute zeroth-order tails at
    /// their own granularity. Makes selection cost ~O(opened) instead of
    /// O(clusters). MEASURED VERDICT (2026-08-16, 0.6B/32K cold pair, with
    /// hierarchical-aware calibration): parity band at 32K (-4% speed, 94.8%
    /// vs 96.9% agreement: ~5 near-tie flips inherent to frontier-granular
    /// selection); transformative beyond ~131K where flat selection grows
    /// linearly and the walk stays constant (0.046ms vs 2.95ms at 524K
    /// rows). Default off in the <=40K regime; THE long-context mode.
    hierarchical: bool = false,
};

pub const Plan = struct {
    seal_end: usize = 0,
    tree_start: usize = 0,
    clusters: usize = 0,
    // Frozen-block incremental maintenance: everything before frozen_end
    // (absolute cache position) is immutable; rebuilds drop the growing
    // suffix (clusters/rows past the frozen counts) and re-split only
    // [frozen_end, seal_end). Arrays are capacity-managed so the frozen
    // prefix is never copied on append (amortized doubling).
    frozen_end: usize = 0,
    frozen_clusters: usize = 0,
    frozen_rows: usize = 0,
    offsets: []u32 = &.{}, // clusters + 1, into packed rows
    counts: []f32 = &.{},
    log_counts: []f32 = &.{},
    centroid: []f32 = &.{}, // clusters * d
    vmean: []f32 = &.{}, // clusters * d
    eig: []f32 = &.{}, // clusters * rank * d
    // Hierarchical frontier: balanced interval tree over leaves. Children
    // are encoded as i32: >= 0 is a node index, < 0 is ~leaf_index.
    node_count: usize = 0,
    node_child: []i32 = &.{}, // node_count * 2
    node_counts: []f32 = &.{},
    node_log_counts: []f32 = &.{},
    node_centroid: []f32 = &.{}, // node_count * d
    node_vmean: []f32 = &.{}, // node_count * d
    node_diag: []f32 = &.{}, // node_count * d (FULL variance, eig energy included)
    lam: []f32 = &.{}, // clusters * rank
    diag_resid: []f32 = &.{}, // clusters * d
    packed_k: []f16 = &.{}, // n * d, cluster-grouped (.f16 format)
    packed_v: []f16 = &.{}, // n * d, cluster-grouped (.f16 format)
    packed_k_q8: []BlockQ8_0 = &.{}, // n * d/32 blocks (.q8_0 format)
    packed_v_q8: []BlockQ8_0 = &.{},
    format: PackedFormat = .f16,

    fn free(self: *Plan, allocator: Allocator) void {
        allocator.free(self.offsets);
        allocator.free(self.counts);
        allocator.free(self.log_counts);
        allocator.free(self.centroid);
        allocator.free(self.vmean);
        allocator.free(self.eig);
        if (self.node_child.len > 0) {
            allocator.free(self.node_child);
            allocator.free(self.node_counts);
            allocator.free(self.node_log_counts);
            allocator.free(self.node_centroid);
            allocator.free(self.node_vmean);
            allocator.free(self.node_diag);
        }
        allocator.free(self.lam);
        allocator.free(self.diag_resid);

        allocator.free(self.packed_k);
        allocator.free(self.packed_v);
        if (self.packed_k_q8.len > 0) allocator.free(self.packed_k_q8);
        if (self.packed_v_q8.len > 0) allocator.free(self.packed_v_q8);
        self.* = .{};
    }
};

/// Per-layer concatenation of every kv head's cluster summaries, kept as
/// persistent f32 tensors so the per-step priority pass and moment tail run
/// as single BLAS-grade matmuls for all query heads at once.
const LayerCat = struct {
    total_c: usize = 0,
    head_off: []u32 = &.{}, // kv_heads + 1 cluster offsets into the concat
    lam: []f32 = &.{}, // total_c * rank
    log_counts: []f32 = &.{}, // total_c
    centroid_t: ?Tensor = null, // (total_c, d)
    eig_t: ?Tensor = null, // (total_c * rank, d)
    diag_t: ?Tensor = null, // (total_c, d)

    fn free(self: *LayerCat, allocator: Allocator) void {
        if (self.head_off.len > 0) allocator.free(self.head_off);
        if (self.lam.len > 0) allocator.free(self.lam);
        if (self.log_counts.len > 0) allocator.free(self.log_counts);
        if (self.centroid_t) |*t| t.deinit();
        if (self.eig_t) |*t| t.deinit();
        if (self.diag_t) |*t| t.deinit();

        self.* = .{};
    }
};

pub const State = struct {
    allocator: Allocator,
    config: Config,
    num_layers: usize,
    q_heads: usize,
    kv_heads: usize,
    d: usize,
    taus: []f32, // num_layers * q_heads
    plans: []Plan, // num_layers * kv_heads
    rebuild_count: u64 = 0,
    /// Diagnostics (atomic; summed across kv-head tasks): exact rows read
    /// from opened clusters, exact suffix/sink rows, clusters scored, and
    /// attendHead invocations.
    stat_opened_rows: u64 = 0,
    stat_exact_rows: u64 = 0,
    stat_scored: u64 = 0,
    stat_calls: u64 = 0,
    /// Force the per-kv-head loop serial even when a work pool exists
    /// (diagnostic: isolates pool spawn/barrier overhead).
    serial: bool = false,
    // Self-calibration: while `calibrating`, attend() opens every cluster,
    // returns the exact output, and records per-head error-vs-threshold
    // curves; finishCalibration() freezes per-head taus from them. This is
    // the recipe's few-run calibration, native and self-contained.
    calibrating: bool = false,
    calib_tol: f32 = 0.025,
    calib_max_rows: usize = 64,
    calib_errs: []f32 = &.{}, // layers*q_heads*grid*max_rows
    calib_counts: []u32 = &.{}, // layers*q_heads
    // Per-kv-head scratch slots (parallel attend), sized to the largest plan.
    numer: []f32 = &.{}, // q_heads * d
    scores: []f32 = &.{}, // q_heads * scr_rows exact-batch score scratch
    scr_rows: usize = 0,
    prio_a: []f32 = &.{}, // q_heads * cap
    hier_prio: []f64 = &.{}, // q_heads * cap frontier heap (log-mass priority)
    hier_a0: []f32 = &.{}, // zeroth-order log-mass per frontier entry
    hier_child: []i32 = &.{}, // child encoding per frontier entry
    prio_est: []f64 = &.{},
    sorted: []u32 = &.{},
    scratch_cap: usize = 0,
    // Layer-level GEMM machinery: concatenated summaries, per-head tail
    // weight rows (q_heads * max total_c, reused across layers), per-head
    // total estimated mass and normalizer partials.
    cats: []LayerCat = &.{},
    head_total_est: []f64 = &.{},
    /// Per-head max cluster priority (the estimate gauge): est[] values are
    /// exp(prio - head_prio_max), so the stop rule's absolute exact mass is
    /// shifted by the same gauge before entering the denominator.
    head_prio_max: []f64 = &.{},
    head_z: []f64 = &.{},
    qq: []f32 = &.{}, // q_heads * d scratch for the diagonal variance GEMM
    /// Persistent model-bridge buffers (query flatten fallback + attention
    /// output), so the per-layer glue performs no allocations.
    bridge_q: []f32 = &.{},
    bridge_out: []f32 = &.{},

    pub fn init(
        allocator: Allocator,
        num_layers: usize,
        q_heads: usize,
        kv_heads: usize,
        d: usize,
        config: Config,
    ) !State {
        if (kv_heads == 0 or q_heads == 0 or q_heads % kv_heads != 0) return error.InvalidHeadGeometry;
        if (kv_heads > 64 or q_heads > 128) return error.TooManyHeads;
        if (d == 0 or d > 512 or d % 8 != 0) return error.InvalidHeadDim;
        if (config.rank > 8 or config.cluster_size == 0 or config.rebuild_interval == 0) return error.InvalidConfig;
        const taus = try allocator.alloc(f32, num_layers * q_heads);
        errdefer allocator.free(taus);
        @memset(taus, config.tau_default);
        const plans = try allocator.alloc(Plan, num_layers * kv_heads);
        errdefer allocator.free(plans);
        @memset(plans, .{});
        const numer = try allocator.alloc(f32, q_heads * d);
        errdefer allocator.free(numer);
        // Largest exact batch: the unsealed suffix right before a rebuild
        // (sink + recent + rebuild_interval rows); clusters are at most
        // cluster_size rows by construction.
        const scr_rows = config.sink + config.recent + config.rebuild_interval + config.cluster_size + 8;
        const scores = try allocator.alloc(f32, q_heads * scr_rows);
        errdefer allocator.free(scores);
        const cats = try allocator.alloc(LayerCat, num_layers);
        @memset(cats, .{});
        const head_total_est = try allocator.alloc(f64, q_heads);
        const head_prio_max = try allocator.alloc(f64, q_heads);
        errdefer allocator.free(head_prio_max);
        const head_z = try allocator.alloc(f64, q_heads);
        const qq = try allocator.alloc(f32, q_heads * d);
        const bridge_q = try allocator.alloc(f32, q_heads * d);
        const bridge_out = try allocator.alloc(f32, q_heads * d);
        return .{
            .allocator = allocator,
            .config = config,
            .num_layers = num_layers,
            .q_heads = q_heads,
            .kv_heads = kv_heads,
            .d = d,
            .taus = taus,
            .plans = plans,
            .numer = numer,
            .scores = scores,
            .scr_rows = scr_rows,
            .cats = cats,
            .head_total_est = head_total_est,
            .head_prio_max = head_prio_max,
            .head_z = head_z,
            .qq = qq,
            .bridge_q = bridge_q,
            .bridge_out = bridge_out,
        };
    }

    pub fn startCalibration(self: *State, tol: f32) !void {
        const grid = calib_grid.len;
        if (self.calib_errs.len == 0) {
            self.calib_errs = try self.allocator.alloc(f32, self.num_layers * self.q_heads * grid * self.calib_max_rows);
            self.calib_counts = try self.allocator.alloc(u32, self.num_layers * self.q_heads);
        }
        @memset(self.calib_counts, 0);
        self.calib_tol = tol;
        self.calibrating = true;
    }

    /// Freeze per-head taus: the largest grid threshold whose median recorded
    /// error meets the tolerance (falls back to the smallest threshold).
    pub fn finishCalibration(self: *State) void {
        const grid = calib_grid.len;
        var row_buf: [256]f32 = undefined;
        for (0..self.num_layers * self.q_heads) |slot| {
            const count = self.calib_counts[slot];
            if (count == 0) continue;
            var chosen: f32 = calib_grid[grid - 1];
            for (0..grid) |g| {
                const base = (slot * grid + g) * self.calib_max_rows;
                const rows = row_buf[0..count];
                @memcpy(rows, self.calib_errs[base..][0..count]);
                std.mem.sort(f32, rows, {}, std.sort.asc(f32));
                const median = rows[count / 2];
                if (median <= self.calib_tol) {
                    chosen = calib_grid[g];
                    break;
                }
            }
            self.taus[slot] = chosen;
        }
        self.calibrating = false;
    }

    pub fn deinit(self: *State) void {
        for (self.plans) |*plan| plan.free(self.allocator);
        self.allocator.free(self.plans);
        self.allocator.free(self.taus);
        self.allocator.free(self.numer);
        self.allocator.free(self.scores);
        for (self.cats) |*cat| cat.free(self.allocator);
        self.allocator.free(self.cats);
        self.allocator.free(self.head_total_est);
        self.allocator.free(self.head_prio_max);
        self.allocator.free(self.head_z);
        self.allocator.free(self.qq);
        self.allocator.free(self.bridge_q);
        self.allocator.free(self.bridge_out);
        if (self.calib_errs.len > 0) {
            self.allocator.free(self.calib_errs);
            self.allocator.free(self.calib_counts);
        }
        if (self.prio_a.len > 0) self.allocator.free(self.prio_a);
        if (self.hier_prio.len > 0) {
            self.allocator.free(self.hier_prio);
            self.allocator.free(self.hier_a0);
            self.allocator.free(self.hier_child);
        }
        if (self.prio_est.len > 0) self.allocator.free(self.prio_est);
        if (self.sorted.len > 0) self.allocator.free(self.sorted);
        self.* = undefined;
    }

    /// Load per-head thresholds from the calibration JSON: {"layer:head": tau}.
    pub fn loadTausJson(self: *State, io: std.Io, path: []const u8) !void {
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .limited(16 * 1024 * 1024));
        defer self.allocator.free(raw);
        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, raw, .{});
        defer parsed.deinit();
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            const colon = std.mem.indexOfScalar(u8, key, ':') orelse return error.InvalidTauKey;
            const layer_i = try std.fmt.parseInt(usize, key[0..colon], 10);
            const head_i = try std.fmt.parseInt(usize, key[colon + 1 ..], 10);
            if (layer_i >= self.num_layers or head_i >= self.q_heads) return error.InvalidTauKey;
            self.taus[layer_i * self.q_heads + head_i] = switch (entry.value_ptr.*) {
                .float => |x| @floatCast(x),
                .integer => |x| @floatFromInt(x),
                else => return error.InvalidTauKey,
            };
        }
    }

    fn ensureScratch(self: *State, clusters: usize) !void {
        if (self.scratch_cap >= clusters) return;
        if (self.prio_a.len > 0) {
            self.allocator.free(self.prio_a);
            self.allocator.free(self.prio_est);
            self.allocator.free(self.sorted);
        }
        // One slot per (kv head, local q head): priorities for a GQA pair are
        // computed in one stream over the summary arrays, so each q head
        // keeps its own copy.
        const slots = self.kv_heads * (self.q_heads / self.kv_heads);
        self.prio_a = try self.allocator.alloc(f32, slots * clusters);
        if (self.hier_prio.len > 0) {
            self.allocator.free(self.hier_prio);
            self.allocator.free(self.hier_a0);
            self.allocator.free(self.hier_child);
        }
        self.hier_prio = try self.allocator.alloc(f64, slots * (clusters + 2));
        self.hier_a0 = try self.allocator.alloc(f32, slots * (clusters + 2));
        self.hier_child = try self.allocator.alloc(i32, slots * (clusters + 2));
        self.prio_est = try self.allocator.alloc(f64, slots * clusters);
        self.sorted = try self.allocator.alloc(u32, slots * clusters);
        self.scratch_cap = clusters;
    }

    /// Decode-step attention for one layer: `q` is the single query row
    /// (q_heads * d, post QK-norm/RoPE), `k`/`v` are the cache's f16 data
    /// (cached_len * kv_heads * d), `out` receives q_heads * d.
    pub fn attend(
        self: *State,
        ctx: *ExecContext,
        layer_i: usize,
        q: []const f32,
        k: []const f16,
        v: []const f16,
        cached_len: usize,
        out: []f32,
    ) !void {
        const cfg = self.config;
        var rebuilt = false;
        for (0..self.kv_heads) |kv_head| {
            const plan = &self.plans[layer_i * self.kv_heads + kv_head];
            // Rewind safety: a cache truncated below the sealed region makes
            // the plan (and its frozen prefix) stale; drop it and replan.
            if (cached_len < plan.seal_end) {
                plan.free(self.allocator);
                rebuilt = true;
            }
            if (cached_len >= cfg.sink + cfg.recent + cfg.rebuild_interval and
                cached_len - cfg.recent >= plan.seal_end + cfg.rebuild_interval)
            {
                try self.rebuild(ctx, plan, k, v, kv_head, cached_len - cfg.recent);
                if (cfg.hierarchical) try self.buildHierarchy(plan);
                self.rebuild_count += 1;
                rebuilt = true;
            }
            try self.ensureScratch(plan.clusters);
        }
        if (rebuilt) try self.rebuildCat(ctx, layer_i);
        if (self.cats[layer_i].total_c > 0 and (!cfg.hierarchical or self.calibrating)) try self.computePrioritiesGemm(ctx, layer_i, q);
        const heads_per_kv = self.q_heads / self.kv_heads;
        var tasks: [128]HeadTask = undefined;
        for (0..self.q_heads) |head_i| {
            tasks[head_i] = .{
                .state = self,
                .layer_i = layer_i,
                .kv_head = head_i / heads_per_kv,
                .head_i = head_i,
                .q = q,
                .k = k,
                .v = v,
                .cached_len = cached_len,
            };
        }
        if (if (self.serial) null else ctx.workPool()) |pool| {
            var wait_group: fucina.internal.thread_mod.WaitGroup = .{};
            for (tasks[1..self.q_heads]) |*task| {
                _ = pool.spawnWg(&wait_group, runHeadTask, .{task});
            }
            runHeadTask(&tasks[0]);
            pool.waitAndWork(&wait_group);
        } else {
            for (tasks[0..self.q_heads]) |*task| runHeadTask(task);
        }
        try self.combineTail(ctx, layer_i, out);
    }

    /// Moment tail as one masked (q_heads x total_c) x (total_c, d) matmul,
    /// then the shared-normalizer combine: out = (numer + tail) / z.
    /// Normalize each head's numerator by its accumulated mass.
    fn combineTail(self: *State, ctx: *ExecContext, layer_i: usize, out: []f32) !void {
        _ = ctx;
        _ = layer_i;
        const d = self.d;
        for (0..self.q_heads) |h| {
            const numer = self.numer[h * d ..][0..d];
            const inv: f32 = @floatCast(1.0 / @max(self.head_z[h], 1e-300));
            for (out[h * d ..][0..d], numer) |*o, nvalue| o.* = nvalue * inv;
        }
    }

    /// Rebuild the layer's concatenated summary tensors after any plan
    /// rebuild (all kv heads of a layer reseal at the same append count).
    fn rebuildCat(self: *State, ctx: *ExecContext, layer_i: usize) !void {
        const cat = &self.cats[layer_i];
        cat.free(self.allocator);
        const d = self.d;
        const rank = self.config.rank;
        const head_off = try self.allocator.alloc(u32, self.kv_heads + 1);
        errdefer self.allocator.free(head_off);
        var total: usize = 0;
        for (0..self.kv_heads) |h| {
            head_off[h] = @intCast(total);
            total += self.plans[layer_i * self.kv_heads + h].clusters;
        }
        head_off[self.kv_heads] = @intCast(total);
        cat.head_off = head_off;
        cat.total_c = total;
        // Coalesced offset-order reads can span every packed row of a plan
        // in one batch: grow the per-head score scratch to cover it.
        var max_rows: usize = 0;
        for (0..self.kv_heads) |h| {
            const plan = &self.plans[layer_i * self.kv_heads + h];
            if (plan.clusters > 0) max_rows = @max(max_rows, plan.offsets[plan.clusters]);
        }
        if (max_rows > self.scr_rows) {
            self.allocator.free(self.scores);
            self.scores = try self.allocator.alloc(f32, self.q_heads * max_rows);
            self.scr_rows = max_rows;
        }
        if (total == 0) return;
        cat.lam = try self.allocator.alloc(f32, total * rank);
        cat.log_counts = try self.allocator.alloc(f32, total);
        const centroid = try self.allocator.alloc(f32, total * d);
        defer self.allocator.free(centroid);
        const eig = try self.allocator.alloc(f32, total * rank * d);
        defer self.allocator.free(eig);
        const diag = try self.allocator.alloc(f32, total * d);
        defer self.allocator.free(diag);
        const vmean = try self.allocator.alloc(f32, total * d);
        defer self.allocator.free(vmean);
        for (0..self.kv_heads) |h| {
            const plan = &self.plans[layer_i * self.kv_heads + h];
            const off: usize = head_off[h];
            const n = plan.clusters;
            if (n == 0) continue;
            @memcpy(cat.lam[off * rank ..][0 .. n * rank], plan.lam[0 .. n * rank]);
            @memcpy(cat.log_counts[off..][0..n], plan.log_counts[0..n]);
            @memcpy(centroid[off * d ..][0 .. n * d], plan.centroid[0 .. n * d]);
            @memcpy(eig[off * rank * d ..][0 .. n * rank * d], plan.eig[0 .. n * rank * d]);
            @memcpy(diag[off * d ..][0 .. n * d], plan.diag_resid[0 .. n * d]);
            @memcpy(vmean[off * d ..][0 .. n * d], plan.vmean[0 .. n * d]);
        }
        cat.centroid_t = try ctx.fromSliceRank(2, .{ total, d }, centroid);
        if (rank > 0) cat.eig_t = try ctx.fromSliceRank(2, .{ total * rank, d }, eig);
        cat.diag_t = try ctx.fromSliceRank(2, .{ total, d }, diag);
    }

    /// Second-cumulant priorities for every query head of the layer in three
    /// BLAS matmuls against the concatenated summaries: a = Q C^T, the rank-r
    /// eigenprojections Q E^T, and the diagonal residual (Q o Q) D^T.
    fn computePrioritiesGemm(self: *State, ctx: *ExecContext, layer_i: usize, q: []const f32) !void {
        const d = self.d;
        const rank = self.config.rank;
        const cat = &self.cats[layer_i];
        const total = cat.total_c;
        var qt = try ctx.fromSliceRank(2, .{ self.q_heads, d }, q[0 .. self.q_heads * d]);
        defer qt.deinit();
        var a_t = try ctx.matmulTransB(&qt, &cat.centroid_t.?);
        defer a_t.deinit();
        var p_t: ?Tensor = null;
        defer if (p_t) |*t| t.deinit();
        if (rank > 0) p_t = try ctx.matmulTransB(&qt, &cat.eig_t.?);
        const qq = self.qq[0 .. self.q_heads * d];
        for (qq, q[0 .. self.q_heads * d]) |*o, x| o.* = x * x;
        var qq_t = try ctx.fromSliceRank(2, .{ self.q_heads, d }, qq);
        defer qq_t.deinit();
        var dv_t = try ctx.matmulTransB(&qq_t, &cat.diag_t.?);
        defer dv_t.deinit();
        const a_all = a_t.dataConst();
        const p_all: []const f32 = if (p_t) |*t| t.dataConst() else &.{};
        const dv_all = dv_t.dataConst();
        const beta: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        const cap = self.scratch_cap;
        const heads_per_kv = self.q_heads / self.kv_heads;
        for (0..self.q_heads) |h| {
            const kv = h / heads_per_kv;
            const off: usize = cat.head_off[kv];
            const cnum: usize = cat.head_off[kv + 1] - off;
            const prio_a = self.prio_a[h * cap ..][0..cnum];
            const est = self.prio_est[h * cap ..][0..cnum];
            const prio_buf = self.scores[h * self.scr_rows ..][0..cnum];
            var maxp: f32 = -std.math.inf(f32);
            for (0..cnum) |c| {
                const g = off + c;
                const a = beta * a_all[h * total + g];
                var variance: f32 = 0;
                for (0..rank) |r| {
                    const proj = p_all[h * (total * rank) + g * rank + r];
                    variance += cat.lam[g * rank + r] * proj * proj;
                }
                variance += dv_all[h * total + g];
                variance *= beta * beta;
                const prio = a + cat.log_counts[g] + 0.5 * variance;
                prio_a[c] = a;
                prio_buf[c] = prio;
                maxp = @max(maxp, prio);
            }
            // Gauge-shifted estimates: one vexpf pass instead of a scalar
            // f64 exp per cluster (the dominant priority cost at long
            // context: ~900K exps/step at 32K before this).
            var total_est: f64 = 0;
            {
                const V = @Vector(8, f32);
                const mv: V = @splat(maxp);
                var c: usize = 0;
                while (c + 8 <= cnum) : (c += 8) {
                    const e = fucina.simd.vexpf(8, @as(V, prio_buf[c..][0..8].*) - mv);
                    inline for (0..8) |j| {
                        est[c + j] = e[j];
                        total_est += e[j];
                    }
                }
                while (c < cnum) : (c += 1) {
                    const e = @exp(prio_buf[c] - maxp);
                    est[c] = e;
                    total_est += e;
                }
            }
            self.head_total_est[h] = total_est;
            self.head_prio_max[h] = maxp;
        }
    }

    /// Second-cumulant priorities for one query over the plan's cluster
    /// summaries; returns the total estimated mass.
    fn computePriorities(self: *State, plan: *const Plan, query: []const f32, beta: f32, prio_a: []f32, est: []f64) f64 {
        const d = self.d;
        const rank = self.config.rank;
        var total: f64 = 0;
        for (0..plan.clusters) |c| {
            const a = beta * dotF32(query, plan.centroid[c * d ..][0..d]);
            var variance: f32 = 0;
            for (0..rank) |r| {
                const proj = dotF32(query, plan.eig[(c * rank + r) * d ..][0..d]);
                variance += plan.lam[c * rank + r] * proj * proj;
            }
            variance += weightedSquareDot(query, plan.diag_resid[c * d ..][0..d]);
            variance *= beta * beta;
            const prio = a + plan.log_counts[c] + 0.5 * variance;
            prio_a[c] = a;
            est[c] = @exp(@as(f64, @min(prio, 700.0)));
            total += est[c];
        }
        return total;
    }

    /// Paired variant for a GQA pair: one stream over centroid/eig/diag
    /// arrays computes both heads' priorities (identical numbers to two
    /// single passes; the summary bytes are read once instead of twice).
    fn computePriorities2(
        self: *State,
        plan: *const Plan,
        q0: []const f32,
        q1: []const f32,
        beta: f32,
        prio_a0: []f32,
        prio_a1: []f32,
        est0: []f64,
        est1: []f64,
        totals: []f64,
    ) void {
        const d = self.d;
        const rank = self.config.rank;
        var total0: f64 = 0;
        var total1: f64 = 0;
        for (0..plan.clusters) |c| {
            const a = dot2F32(q0, q1, plan.centroid[c * d ..][0..d]);
            var var0: f32 = 0;
            var var1: f32 = 0;
            for (0..rank) |r| {
                const proj = dot2F32(q0, q1, plan.eig[(c * rank + r) * d ..][0..d]);
                const lam = plan.lam[c * rank + r];
                var0 += lam * proj[0] * proj[0];
                var1 += lam * proj[1] * proj[1];
            }
            const diag = weightedSquareDot2(q0, q1, plan.diag_resid[c * d ..][0..d]);
            var0 = (var0 + diag[0]) * beta * beta;
            var1 = (var1 + diag[1]) * beta * beta;
            const log_m = plan.log_counts[c];
            const a0 = beta * a[0];
            const a1 = beta * a[1];
            prio_a0[c] = a0;
            prio_a1[c] = a1;
            est0[c] = @exp(@as(f64, @min(a0 + log_m + 0.5 * var0, 700.0)));
            est1[c] = @exp(@as(f64, @min(a1 + log_m + 0.5 * var1, 700.0)));
            total0 += est0[c];
            total1 += est1[c];
        }
        totals[0] = total0;
        totals[1] = total1;
    }

    /// Frontier entry score: the second-cumulant log-mass priority (ranker)
    /// and the zeroth-order log-mass (tail weight), for a node (>= 0) or a
    /// leaf (~index). Internal nodes use the aggregated full-diagonal
    /// variance; leaves use their full rank + diagonal model.
    fn scoreEntry(self: *State, plan: *const Plan, child: i32, query: []const f32, beta: f32) HeapEntry {
        const d = self.d;
        const rank = self.config.rank;
        if (child >= 0) {
            const ni: usize = @intCast(child);
            const a = beta * dotF32(query, plan.node_centroid[ni * d ..][0..d]);
            const lm = plan.node_log_counts[ni];
            const variance = weightedSquareDot(query, plan.node_diag[ni * d ..][0..d]) * beta * beta;
            return .{ .prio = @min(@as(f64, a + lm + 0.5 * variance), 700), .a0 = a + lm, .child = child };
        }
        const li: usize = @intCast(~child);
        const a = beta * dotF32(query, plan.centroid[li * d ..][0..d]);
        const lm = plan.log_counts[li];
        var variance: f32 = 0;
        for (0..rank) |r| {
            const proj = dotF32(query, plan.eig[(li * rank + r) * d ..][0..d]);
            variance += plan.lam[li * rank + r] * proj * proj;
        }
        variance += weightedSquareDot(query, plan.diag_resid[li * d ..][0..d]);
        variance *= beta * beta;
        return .{ .prio = @min(@as(f64, a + lm + 0.5 * variance), 700), .a0 = a + lm, .child = child };
    }

    fn attendHead(
        self: *State,
        plan: *const Plan,
        query: []const f32,
        k: []const f16,
        v: []const f16,
        kv_head: usize,
        cached_len: usize,
        beta: f32,
        tau: f32,
        layer_i: usize,
        head_i: usize,
        prio_a: []const f32,
        est: []const f64,
        sorted_scratch: []u32,
        total_est: f64,
    ) void {
        _ = layer_i;
        const cfg = self.config;
        const d = self.d;
        const row_stride = self.kv_heads * d;
        const numer = self.numer[head_i * d ..][0..d];
        @memset(numer, 0);
        const scores_all = self.scores[head_i * self.scr_rows ..][0..self.scr_rows];
        // q8_0 packed format: the query is quantized once per head call and
        // scored on the integer path, like the dense q8-KV decode kernels.
        const use_q8 = plan.format == .q8_0;
        const bpr = if (use_q8) d / q8_block else 0;
        var q_q8_buf: [16]BlockQ8_0 = undefined;
        var q_scales_buf: [16]f32 = undefined;
        if (use_q8) {
            qkern.q8k.quantizeRowQ8_0Into(q_q8_buf[0..bpr], query) catch unreachable;
            qkern.q8_0.q8RowScalesInto(q_scales_buf[0..bpr], q_q8_buf[0..bpr]);
        }

        // Exact part under an online softmax gauge (numer, exact_w, and
        // tail_z are all relative to exp(gauge); the gauge cancels in the
        // final normalization, and the stop rule converts to absolute mass
        // with one scalar exp where it compares against the estimates):
        // sinks plus the unsealed suffix (cache rows, strided).
        var gauge: f32 = -std.math.inf(f32);
        var exact_w: f64 = 0;
        const sink_end = @min(cfg.sink, cached_len);
        if (sink_end > 0) {
            const base = kv_head * d;
            exactBatch(numer, &gauge, &exact_w, scores_all[0..sink_end], query, k[base..], v[base..], row_stride, d, beta);
        }
        const suffix_start = @max(plan.seal_end, sink_end);
        if (cached_len > suffix_start) {
            const m = cached_len - suffix_start;
            const base = suffix_start * row_stride + kv_head * d;
            exactBatch(numer, &gauge, &exact_w, scores_all[0..m], query, k[base..], v[base..], row_stride, d, beta);
        }
        if (gauge == -std.math.inf(f32)) gauge = 0;
        _ = @atomicRmw(u64, &self.stat_exact_rows, .Add, @as(u64, @intCast(sink_end + (cached_len - suffix_start))), .monotonic);
        _ = @atomicRmw(u64, &self.stat_calls, .Add, 1, .monotonic);
        _ = @atomicRmw(u64, &self.stat_scored, .Add, @as(u64, @intCast(plan.clusters)), .monotonic);

        var tail_z: f64 = 0;
        if (plan.clusters > 0) {
            const clusters = plan.clusters;
            const sorted = sorted_scratch[0..clusters];
            const hier_mode = cfg.hierarchical and plan.node_count > 0;
            var hier_frontier: usize = 0;
            const hstride = self.scratch_cap + 2;
            const hp = self.hier_prio[head_i * hstride ..][0..hstride];
            const ha = self.hier_a0[head_i * hstride ..][0..hstride];
            const hc = self.hier_child[head_i * hstride ..][0..hstride];
            var opened: usize = 0;
            if (hier_mode) {
                // Hierarchical selection: expand the highest estimated-
                // mass frontier node until the estimate-only stop holds;
                // no reads happen inside the walk.
                var hn: usize = 0;
                const root = self.scoreEntry(plan, 0, query, beta);
                heapPush(hp, ha, hc, &hn, root.prio, root.a0, root.child);
                var rem: f64 = @exp(root.prio);
                var captured: f64 = 0;
                const exact_abs_h = exact_w * @exp(@as(f64, gauge));
                while (hn > 0) {
                    if (rem <= @as(f64, tau) * (rem + captured + exact_abs_h)) break;
                    const e = heapPop(hp, ha, hc, &hn);
                    rem -= @exp(e.prio);
                    if (rem < 0) rem = 0;
                    if (e.child < 0) {
                        sorted[opened] = @intCast(~e.child);
                        opened += 1;
                        captured += @exp(e.prio);
                    } else {
                        const ni: usize = @intCast(e.child);
                        inline for (0..2) |side| {
                            const cs = self.scoreEntry(plan, plan.node_child[ni * 2 + side], query, beta);
                            heapPush(hp, ha, hc, &hn, cs.prio, cs.a0, cs.child);
                            rem += @exp(cs.prio);
                        }
                    }
                }
                hier_frontier = hn;
            } else {
                for (sorted, 0..) |*s, c| s.* = @intCast(c);
                std.mem.sort(u32, sorted, est, estDesc);

                // Estimate-only stop: captured mass is counted by the same
                // estimates that rank the clusters, so the denominator is
                // constant and the opened set is decided before any read
                // (self-calibration freezes tau under this exact rule).
                const exact_abs = exact_w * @exp(@as(f64, gauge) - self.head_prio_max[head_i]);
                const target = @as(f64, tau) * (total_est + exact_abs);
                var rem_est = total_est;
                while (opened < clusters and rem_est > target) : (opened += 1) {
                    rem_est -= est[sorted[opened]];
                }
            }
            // Read the decided set in ascending cluster order; adjacent
            // opened clusters are contiguous in the packed arrays and
            // coalesce into single streaming batches.
            std.mem.sort(u32, sorted[0..opened], {}, std.sort.asc(u32));
            var oi: usize = 0;
            while (oi < opened) {
                var oj = oi + 1;
                while (oj < opened and sorted[oj] == sorted[oj - 1] + 1) : (oj += 1) {}
                const off: usize = plan.offsets[sorted[oi]];
                const m: usize = plan.offsets[sorted[oj - 1] + 1] - off;
                _ = @atomicRmw(u64, &self.stat_opened_rows, .Add, @as(u64, @intCast(m)), .monotonic);
                switch (plan.format) {
                    .q8_0 => exactBatchQ8(numer, &gauge, &exact_w, scores_all[0..m], q_q8_buf[0..bpr], q_scales_buf[0..bpr], plan.packed_k_q8[off * bpr ..], plan.packed_v_q8[off * bpr ..], bpr, beta),
                    .f16 => exactBatch(numer, &gauge, &exact_w, scores_all[0..m], query, plan.packed_k[off * d ..], plan.packed_v[off * d ..], d, d, beta),
                }
                oi = oj;
            }

            if (hier_mode) {
                // Zeroth-order tails at frontier granularity: every
                // unexpanded node (or unopened leaf) contributes its
                // aggregated m * exp(a) at its own value mean.
                for (0..hier_frontier) |fi| {
                    const w = @exp(@as(f64, ha[fi]) - @as(f64, gauge));
                    tail_z += w;
                    const w32: f32 = @floatCast(w);
                    const ch = hc[fi];
                    if (ch >= 0) {
                        accumulateF32(numer, plan.node_vmean[@as(usize, @intCast(ch)) * d ..][0..d], w32);
                    } else {
                        accumulateF32(numer, plan.vmean[@as(usize, @intCast(~ch)) * d ..][0..d], w32);
                    }
                }
            } else {
                // Zeroth-order tail over the unopened clusters (m * exp(a)
                // at the value mean): DELIBERATE. The second-cumulant
                // estimate exp(a + s^2/2) is the right RANKER but a badly
                // biased mass weight (measured 2026-08-16: agreement
                // collapse to 27-65% when est[] weighted the tail).
                for (sorted[opened..]) |c| {
                    const w = @as(f64, plan.counts[c]) * @exp(@as(f64, prio_a[c]) - @as(f64, gauge));
                    tail_z += w;
                    accumulateF32(numer, plan.vmean[@as(usize, c) * d ..][0..d], @floatCast(w));
                }
            }
        }

        self.head_z[head_i] = exact_w + tail_z;
    }

    /// Calibration variant of attendHead: opens every cluster (the output is
    /// therefore EXACT), and records for each grid threshold the error the
    /// mass-stop would have made had it stopped there. Cold path.
    fn calibrateHead(
        self: *State,
        plan: *const Plan,
        query: []const f32,
        k: []const f16,
        v: []const f16,
        kv_head: usize,
        cached_len: usize,
        beta: f32,
        layer_i: usize,
        head_i: usize,
        prio_a: []const f32,
        est: []const f64,
        sorted_scratch: []u32,
        total_est: f64,
    ) !void {
        const cfg = self.config;
        const d = self.d;
        const row_stride = self.kv_heads * d;
        const allocator = self.allocator;
        const numer = self.numer[head_i * d ..][0..d];
        @memset(numer, 0);
        // Calibration output is exact: numer carries the full numerator and
        // the shared combine normalizes.

        var exact_w: f64 = 0;
        const sink_end = @min(cfg.sink, cached_len);
        for (0..sink_end) |pos| {
            const base = pos * row_stride + kv_head * d;
            const w = @exp(@as(f64, beta * dotF16(query, k[base..][0..d])));
            accumulateF16(numer, v[base..][0..d], @floatCast(w));
            exact_w += w;
        }
        var pos = @max(plan.seal_end, sink_end);
        while (pos < cached_len) : (pos += 1) {
            const base = pos * row_stride + kv_head * d;
            const w = @exp(@as(f64, beta * dotF16(query, k[base..][0..d])));
            accumulateF16(numer, v[base..][0..d], @floatCast(w));
            exact_w += w;
        }

        if (plan.clusters == 0) {
            self.head_z[head_i] = exact_w;
            return;
        }
        const clusters = plan.clusters;
        const sorted = sorted_scratch[0..clusters];
        for (sorted, 0..) |*s_, c| s_.* = @intCast(c);
        std.mem.sort(u32, sorted, est, estDesc);

        // Per-cluster exact partials.
        const wc = try allocator.alloc(f64, clusters);
        defer allocator.free(wc);
        const nc = try allocator.alloc(f32, clusters * d);
        defer allocator.free(nc);
        @memset(nc, 0);
        var kbuf: [512]f32 = undefined;
        var vbuf: [512]f32 = undefined;
        const cal_bpr = if (plan.format == .q8_0) d / q8_block else 0;
        for (0..clusters) |c| {
            var w_sum: f64 = 0;
            const cn = nc[c * d ..][0..d];
            var row: usize = plan.offsets[c];
            const end: usize = plan.offsets[c + 1];
            while (row < end) : (row += 1) {
                if (plan.format == .q8_0) {
                    qkern.q8k.dequantizeRowQ8_0Into(kbuf[0..d], plan.packed_k_q8[row * cal_bpr ..][0..cal_bpr]) catch unreachable;
                    qkern.q8k.dequantizeRowQ8_0Into(vbuf[0..d], plan.packed_v_q8[row * cal_bpr ..][0..cal_bpr]) catch unreachable;
                    const w = @exp(@as(f64, beta * dotF32(query, kbuf[0..d])));
                    accumulateF32(cn, vbuf[0..d], @floatCast(w));
                    w_sum += w;
                } else {
                    const w = @exp(@as(f64, beta * dotF16(query, plan.packed_k[row * d ..][0..d])));
                    accumulateF16(cn, plan.packed_v[row * d ..][0..d], @floatCast(w));
                    w_sum += w;
                }
            }
            wc[c] = w_sum;
        }
        var full_w = exact_w;
        var exact_out_buf: [256]f32 = undefined;
        const exact_out = exact_out_buf[0..d];
        @memcpy(exact_out, numer);
        for (0..clusters) |c| {
            full_w += wc[c];
            accumulateF32(exact_out, nc[c * d ..][0..d], 1.0);
        }
        const inv_full: f32 = @floatCast(1.0 / @max(full_w, 1e-300));
        for (exact_out) |*o| o.* *= inv_full;
        var exact_norm: f32 = @sqrt(dotF32(exact_out, exact_out));
        exact_norm = @max(exact_norm, 1e-12);

        // Zeroth-order tail suffix sums over the sorted order.
        const tail_z = try allocator.alloc(f64, clusters + 1);
        defer allocator.free(tail_z);
        const tail_n = try allocator.alloc(f32, (clusters + 1) * d);
        defer allocator.free(tail_n);
        tail_z[clusters] = 0;
        @memset(tail_n[clusters * d ..][0..d], 0);
        var ci = clusters;
        while (ci > 0) {
            ci -= 1;
            const c = sorted[ci];
            const w = @as(f64, plan.counts[c]) * @exp(@as(f64, prio_a[c]));
            tail_z[ci] = tail_z[ci + 1] + w;
            const dst = tail_n[ci * d ..][0..d];
            @memcpy(dst, tail_n[(ci + 1) * d ..][0..d]);
            const w32: f32 = @floatCast(w);
            accumulateF32(dst, plan.vmean[@as(usize, c) * d ..][0..d], w32);
        }

        // Prefix walk: record the mass-stop error at each grid threshold.
        const slot = layer_i * self.q_heads + head_i;
        const row_i = self.calib_counts[slot];
        const grid = calib_grid.len;
        var cur_w = exact_w;
        var cur_numer_buf: [256]f32 = undefined;
        const cur_numer = cur_numer_buf[0..d];
        @memcpy(cur_numer, numer);
        var rem_est = total_est;
        var g: usize = 0;
        var approx_buf: [256]f32 = undefined;
        const approx = approx_buf[0..d];
        const exact_gauged = exact_w * @exp(-self.head_prio_max[head_i]);
        for (0..clusters + 1) |step| {
            const frac = rem_est / @max(total_est + exact_gauged, 1e-300);
            while (g < grid and frac <= calib_grid[g]) : (g += 1) {
                if (row_i < self.calib_max_rows) {
                    const z = cur_w + tail_z[step];
                    const invz: f32 = @floatCast(1.0 / @max(z, 1e-300));
                    var err_sq: f32 = 0;
                    for (0..d) |dim| {
                        approx[dim] = (cur_numer[dim] + tail_n[step * d + dim]) * invz;
                        const delta = approx[dim] - exact_out[dim];
                        err_sq += delta * delta;
                    }
                    self.calib_errs[(slot * grid + g) * self.calib_max_rows + row_i] = @sqrt(err_sq) / exact_norm;
                }
            }
            if (g >= grid or step == clusters) break;
            const c = sorted[step];
            cur_w += wc[c];
            accumulateF32(cur_numer, nc[@as(usize, c) * d ..][0..d], 1.0);
            rem_est -= est[c];
        }
        while (g < grid) : (g += 1) {
            if (row_i < self.calib_max_rows) {
                self.calib_errs[(slot * grid + g) * self.calib_max_rows + row_i] = 0;
            }
        }
        if (row_i < self.calib_max_rows) self.calib_counts[slot] = row_i + 1;

        // Hand off the exact numerator and normalizer to the shared combine.
        for (0..clusters) |c| accumulateF32(numer, nc[c * d ..][0..d], 1.0);
        self.head_z[head_i] = full_w;
    }

    /// Build the balanced interval tree over the plan's leaf clusters and
    /// aggregate internal summaries bottom-up (exact for count/centroid/
    /// value mean; diagonal variance via the within+between decomposition,
    /// with leaf eig energy folded back so internal variance is FULL).
    fn buildHierarchy(self: *State, plan: *Plan) !void {
        const allocator = self.allocator;
        const d = self.d;
        const rank = self.config.rank;
        if (plan.node_child.len > 0) {
            allocator.free(plan.node_child);
            allocator.free(plan.node_counts);
            allocator.free(plan.node_log_counts);
            allocator.free(plan.node_centroid);
            allocator.free(plan.node_vmean);
            allocator.free(plan.node_diag);
            plan.node_child = &.{};
            plan.node_count = 0;
        }
        const leaves = plan.clusters;
        if (leaves < 2) return;
        const max_nodes = leaves - 1;
        plan.node_child = try allocator.alloc(i32, max_nodes * 2);
        plan.node_counts = try allocator.alloc(f32, max_nodes);
        plan.node_log_counts = try allocator.alloc(f32, max_nodes);
        plan.node_centroid = try allocator.alloc(f32, max_nodes * d);
        plan.node_vmean = try allocator.alloc(f32, max_nodes * d);
        plan.node_diag = try allocator.alloc(f32, max_nodes * d);
        var next: usize = 0;
        _ = try self.buildNode(plan, &next, 0, leaves, d, rank);
        plan.node_count = next;
    }

    /// Recursive helper: builds the subtree over leaf range [lo, hi) and
    /// returns its child encoding (node index >= 0, ~leaf for singletons).
    fn buildNode(self: *State, plan: *Plan, next: *usize, lo: usize, hi: usize, d: usize, rank: usize) !i32 {
        if (hi - lo == 1) return ~@as(i32, @intCast(lo));
        const idx = next.*;
        next.* += 1;
        const mid = lo + (hi - lo) / 2;
        const c0 = try self.buildNode(plan, next, lo, mid, d, rank);
        const c1 = try self.buildNode(plan, next, mid, hi, d, rank);
        plan.node_child[idx * 2] = c0;
        plan.node_child[idx * 2 + 1] = c1;
        // Aggregate the two children (leaf or node summaries).
        var m0: f32 = undefined;
        var m1: f32 = undefined;
        var cen0: []const f32 = undefined;
        var cen1: []const f32 = undefined;
        var vm0: []const f32 = undefined;
        var vm1: []const f32 = undefined;
        inline for (.{ c0, c1 }, 0..) |ch, side| {
            var m_: f32 = undefined;
            var cen_: []const f32 = undefined;
            var vm_: []const f32 = undefined;
            if (ch >= 0) {
                const ni: usize = @intCast(ch);
                m_ = plan.node_counts[ni];
                cen_ = plan.node_centroid[ni * d ..][0..d];
                vm_ = plan.node_vmean[ni * d ..][0..d];
            } else {
                const li: usize = @intCast(~ch);
                m_ = plan.counts[li];
                cen_ = plan.centroid[li * d ..][0..d];
                vm_ = plan.vmean[li * d ..][0..d];
            }
            if (side == 0) {
                m0 = m_;
                cen0 = cen_;
                vm0 = vm_;
            } else {
                m1 = m_;
                cen1 = cen_;
                vm1 = vm_;
            }
        }
        const m = m0 + m1;
        plan.node_counts[idx] = m;
        plan.node_log_counts[idx] = @log(m);
        const cen = plan.node_centroid[idx * d ..][0..d];
        const vm = plan.node_vmean[idx * d ..][0..d];
        const dg = plan.node_diag[idx * d ..][0..d];
        for (0..d) |j| {
            cen[j] = (m0 * cen0[j] + m1 * cen1[j]) / m;
            vm[j] = (m0 * vm0[j] + m1 * vm1[j]) / m;
        }
        // Full diagonal variance: E[x^2] mixes children's (var + mean^2).
        for (0..d) |j| {
            const e2_0 = childFullVar(plan, c0, j, d, rank) + cen0[j] * cen0[j];
            const e2_1 = childFullVar(plan, c1, j, d, rank) + cen1[j] * cen1[j];
            dg[j] = @max((m0 * e2_0 + m1 * e2_1) / m - cen[j] * cen[j], 0);
        }
        return @intCast(idx);
    }

    /// Hierarchical-aware calibration: identical error-vs-threshold curves
    /// to calibrateHead, but the counterfactual replays the HIERARCHICAL
    /// expansion walk (frontier-granular estimates and tails), so frozen
    /// taus match the rule the decode path actually runs.
    fn calibrateHeadHier(
        self: *State,
        plan: *const Plan,
        query: []const f32,
        k: []const f16,
        v: []const f16,
        kv_head: usize,
        cached_len: usize,
        beta: f32,
        layer_i: usize,
        head_i: usize,
    ) !void {
        const cfg = self.config;
        const d = self.d;
        const row_stride = self.kv_heads * d;
        const allocator = self.allocator;
        const numer = self.numer[head_i * d ..][0..d];
        @memset(numer, 0);

        // Exact sinks + suffix (absolute f64 weights, like calibrateHead).
        var exact_w: f64 = 0;
        const sink_end = @min(cfg.sink, cached_len);
        for (0..sink_end) |pos| {
            const base = pos * row_stride + kv_head * d;
            const w = @exp(@as(f64, beta * dotF16(query, k[base..][0..d])));
            accumulateF16(numer, v[base..][0..d], @floatCast(w));
            exact_w += w;
        }
        var pos = @max(plan.seal_end, sink_end);
        while (pos < cached_len) : (pos += 1) {
            const base = pos * row_stride + kv_head * d;
            const w = @exp(@as(f64, beta * dotF16(query, k[base..][0..d])));
            accumulateF16(numer, v[base..][0..d], @floatCast(w));
            exact_w += w;
        }
        const clusters = plan.clusters;
        if (clusters == 0 or plan.node_count == 0) {
            self.head_z[head_i] = @max(exact_w, 1e-300);
            return;
        }

        // Exact per-leaf partials (dequant-consistent reads).
        const wc = try allocator.alloc(f64, clusters);
        defer allocator.free(wc);
        const nc = try allocator.alloc(f32, clusters * d);
        defer allocator.free(nc);
        @memset(nc, 0);
        var kbuf: [512]f32 = undefined;
        var vbuf: [512]f32 = undefined;
        const cal_bpr = if (plan.format == .q8_0) d / q8_block else 0;
        for (0..clusters) |c| {
            var w_sum: f64 = 0;
            const cn = nc[c * d ..][0..d];
            var row: usize = plan.offsets[c];
            const end: usize = plan.offsets[c + 1];
            while (row < end) : (row += 1) {
                if (plan.format == .q8_0) {
                    qkern.q8k.dequantizeRowQ8_0Into(kbuf[0..d], plan.packed_k_q8[row * cal_bpr ..][0..cal_bpr]) catch unreachable;
                    qkern.q8k.dequantizeRowQ8_0Into(vbuf[0..d], plan.packed_v_q8[row * cal_bpr ..][0..cal_bpr]) catch unreachable;
                    const w = @exp(@as(f64, beta * dotF32(query, kbuf[0..d])));
                    accumulateF32(cn, vbuf[0..d], @floatCast(w));
                    w_sum += w;
                } else {
                    const w = @exp(@as(f64, beta * dotF16(query, plan.packed_k[row * d ..][0..d])));
                    accumulateF16(cn, plan.packed_v[row * d ..][0..d], @floatCast(w));
                    w_sum += w;
                }
            }
            wc[c] = w_sum;
        }
        var full_w = exact_w;
        var exact_out_buf: [512]f32 = undefined;
        const exact_out = exact_out_buf[0..d];
        @memcpy(exact_out, numer);
        for (0..clusters) |c| {
            full_w += wc[c];
            accumulateF32(exact_out, nc[c * d ..][0..d], 1.0);
        }
        const inv_full: f32 = @floatCast(1.0 / @max(full_w, 1e-300));
        for (exact_out) |*o| o.* *= inv_full;
        var exact_norm: f32 = @sqrt(dotF32(exact_out, exact_out));
        exact_norm = @max(exact_norm, 1e-12);

        // Counterfactual hierarchical walk: expand fully, recording the
        // approximation error at each grid threshold's first crossing.
        const hstride = self.scratch_cap + 2;
        const hp = self.hier_prio[head_i * hstride ..][0..hstride];
        const ha = self.hier_a0[head_i * hstride ..][0..hstride];
        const hc = self.hier_child[head_i * hstride ..][0..hstride];
        var hn: usize = 0;
        const root = self.scoreEntry(plan, 0, query, beta);
        heapPush(hp, ha, hc, &hn, root.prio, root.a0, root.child);
        var rem: f64 = @exp(root.prio);
        var captured: f64 = 0;
        // Frontier zeroth-order tail sums, maintained incrementally.
        var tail_zf: f64 = @exp(@as(f64, root.a0));
        var tail_nf_buf: [512]f32 = undefined;
        const tail_nf = tail_nf_buf[0..d];
        for (tail_nf, plan.node_vmean[0..d]) |*o, m_| o.* = @floatCast(@exp(@as(f64, root.a0)) * @as(f64, m_));
        var cur_w = exact_w;
        var cur_numer_buf: [512]f32 = undefined;
        const cur_numer = cur_numer_buf[0..d];
        @memcpy(cur_numer, numer);

        const slot = layer_i * self.q_heads + head_i;
        const row_i = self.calib_counts[slot];
        const grid = calib_grid.len;
        var g: usize = 0;
        var approx_buf: [512]f32 = undefined;
        const approx = approx_buf[0..d];
        while (true) {
            const frac = rem / @max(rem + captured + exact_w, 1e-300);
            while (g < grid and frac <= calib_grid[g]) : (g += 1) {
                if (row_i < self.calib_max_rows) {
                    const z = cur_w + tail_zf;
                    const invz: f32 = @floatCast(1.0 / @max(z, 1e-300));
                    var err_sq: f32 = 0;
                    for (0..d) |dim| {
                        approx[dim] = (cur_numer[dim] + tail_nf[dim]) * invz;
                        const delta = approx[dim] - exact_out[dim];
                        err_sq += delta * delta;
                    }
                    self.calib_errs[(slot * grid + g) * self.calib_max_rows + row_i] = @sqrt(err_sq) / exact_norm;
                }
            }
            if (g >= grid or hn == 0) break;
            const e = heapPop(hp, ha, hc, &hn);
            rem -= @exp(e.prio);
            if (rem < 0) rem = 0;
            const w_zero = @exp(@as(f64, e.a0));
            tail_zf -= w_zero;
            if (e.child >= 0) {
                const ni: usize = @intCast(e.child);
                accumulateF32(tail_nf, plan.node_vmean[ni * d ..][0..d], @floatCast(-w_zero));
                inline for (0..2) |side| {
                    const cs = self.scoreEntry(plan, plan.node_child[ni * 2 + side], query, beta);
                    heapPush(hp, ha, hc, &hn, cs.prio, cs.a0, cs.child);
                    rem += @exp(cs.prio);
                    const wz = @exp(@as(f64, cs.a0));
                    tail_zf += wz;
                    if (cs.child >= 0) {
                        accumulateF32(tail_nf, plan.node_vmean[@as(usize, @intCast(cs.child)) * d ..][0..d], @floatCast(wz));
                    } else {
                        accumulateF32(tail_nf, plan.vmean[@as(usize, @intCast(~cs.child)) * d ..][0..d], @floatCast(wz));
                    }
                }
            } else {
                const li: usize = @intCast(~e.child);
                accumulateF32(tail_nf, plan.vmean[li * d ..][0..d], @floatCast(-w_zero));
                captured += @exp(e.prio);
                cur_w += wc[li];
                accumulateF32(cur_numer, nc[li * d ..][0..d], 1.0);
            }
        }
        while (g < grid) : (g += 1) {
            if (row_i < self.calib_max_rows) {
                self.calib_errs[(slot * grid + g) * self.calib_max_rows + row_i] = 0;
            }
        }
        if (row_i < self.calib_max_rows) self.calib_counts[slot] = row_i + 1;

        // Hand off the exact output through the shared combine.
        for (0..clusters) |c| accumulateF32(numer, nc[c * d ..][0..d], 1.0);
        self.head_z[head_i] = full_w;
    }

    /// Rebuild the sealed plan over cache positions [sink, seal_end). Cold
    /// path: once per rebuild_interval appended positions.
    fn rebuild(
        self: *State,
        ctx: *ExecContext,
        plan: *Plan,
        k: []const f16,
        v: []const f16,
        kv_head: usize,
        seal_end: usize,
    ) !void {
        const allocator = self.allocator;
        const cfg = self.config;
        const d = self.d;
        const row_stride = self.kv_heads * d;
        const tree_start = @min(cfg.sink, seal_end);
        const fmt_resolved: PackedFormat = if (cfg.packed_format == .q8_0 and d % q8_block == 0) .q8_0 else .f16;
        // A format change mid-run forces a full reset.
        if (plan.frozen_rows > 0 and plan.format != fmt_resolved) plan.free(allocator);
        // Drop the growing suffix; the frozen prefix stays in place.
        plan.clusters = plan.frozen_clusters;
        const old_rows = plan.frozen_rows;
        const region_start = @max(tree_start, plan.frozen_end);
        const n = seal_end - region_start;
        plan.seal_end = seal_end;
        plan.tree_start = tree_start;
        if (n < 2) return;

        const keys = try allocator.alloc(f32, n * d);
        defer allocator.free(keys);
        for (0..n) |i| {
            const base = (region_start + i) * row_stride + kv_head * d;
            for (0..d) |c| keys[i * d + c] = @floatCast(k[base + c]);
        }

        // Farthest-direction splitting down to cluster granularity; the
        // recursion frontier IS the cluster partition, contiguous in `order`.
        const order = try allocator.alloc(u32, n);
        defer allocator.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        var offsets_list: std.ArrayList(u32) = .empty;
        defer offsets_list.deinit(allocator);
        try offsets_list.append(allocator, 0);
        const norms = try allocator.alloc(f32, n);
        defer allocator.free(norms);
        for (0..n) |i| norms[i] = dotF32(keys[i * d ..][0..d], keys[i * d ..][0..d]);
        var builder = Builder{
            .keys = keys,
            .norms = norms,
            .d = d,
            .cluster_size = cfg.cluster_size,
            .order = order,
            .offsets = &offsets_list,
            .allocator = allocator,
        };
        try builder.split(0, n);
        const clusters = offsets_list.items.len - 1;

        // Packed per-cluster keys/values in order sequence; format decides
        // whether the f16 or q8_0 arrays are materialized.
        const fmt = fmt_resolved;
        const packed_k = try allocator.alloc(f16, if (fmt == .f16) n * d else 0);
        defer allocator.free(packed_k);
        const packed_v = try allocator.alloc(f16, if (fmt == .f16) n * d else 0);
        defer allocator.free(packed_v);
        const ordered_k = try allocator.alloc(f32, n * d);
        defer allocator.free(ordered_k);
        const ordered_k2 = try allocator.alloc(f32, n * d);
        defer allocator.free(ordered_k2);
        const ordered_v = try allocator.alloc(f32, n * d);
        defer allocator.free(ordered_v);
        for (order, 0..) |member, i| {
            const base = (region_start + @as(usize, member)) * row_stride + kv_head * d;
            for (0..d) |c| {
                const kv_k = k[base + c];
                const kv_v = v[base + c];
                if (fmt == .f16) {
                    packed_k[i * d + c] = kv_k;
                    packed_v[i * d + c] = kv_v;
                }
                const kf: f32 = @floatCast(kv_k);
                ordered_k[i * d + c] = kf;
                ordered_k2[i * d + c] = kf * kf;
                ordered_v[i * d + c] = @floatCast(kv_v);
            }
        }
        const bpr = d / q8_block;
        const packed_k_q8 = try allocator.alloc(BlockQ8_0, if (fmt == .q8_0) n * bpr else 0);
        defer allocator.free(packed_k_q8);
        const packed_v_q8 = try allocator.alloc(BlockQ8_0, if (fmt == .q8_0) n * bpr else 0);
        defer allocator.free(packed_v_q8);
        if (fmt == .q8_0) {
            for (0..n) |i| {
                try qkern.q8k.quantizeRowQ8_0Into(packed_k_q8[i * bpr ..][0..bpr], ordered_k[i * d ..][0..d]);
                try qkern.q8k.quantizeRowQ8_0Into(packed_v_q8[i * bpr ..][0..bpr], ordered_v[i * d ..][0..d]);
            }
        }
        // Cluster moment sums via the segmentSum core op.
        const seg_offsets = try allocator.alloc(usize, clusters + 1);
        defer allocator.free(seg_offsets);
        for (offsets_list.items, 0..) |off, i| seg_offsets[i] = off;
        const sum_k = try segmentSumRows(ctx, ordered_k, n, d, seg_offsets, allocator);
        defer allocator.free(sum_k);
        const sum_v = try segmentSumRows(ctx, ordered_v, n, d, seg_offsets, allocator);
        defer allocator.free(sum_v);
        const sum_k2 = try segmentSumRows(ctx, ordered_k2, n, d, seg_offsets, allocator);
        defer allocator.free(sum_k2);

        const counts = try allocator.alloc(f32, clusters);
        defer allocator.free(counts);
        const log_counts = try allocator.alloc(f32, clusters);
        defer allocator.free(log_counts);
        const centroid = try allocator.alloc(f32, clusters * d);
        defer allocator.free(centroid);
        const vmean = try allocator.alloc(f32, clusters * d);
        defer allocator.free(vmean);
        const eig = try allocator.alloc(f32, clusters * cfg.rank * d);
        defer allocator.free(eig);
        const lam = try allocator.alloc(f32, clusters * cfg.rank);
        defer allocator.free(lam);
        const diag_resid = try allocator.alloc(f32, clusters * d);
        defer allocator.free(diag_resid);

        const z_scratch = try allocator.alloc(f32, (2 * cfg.cluster_size + 4) * d);
        defer allocator.free(z_scratch);
        const u_scratch = try allocator.alloc(f32, d);
        defer allocator.free(u_scratch);
        const w_scratch = try allocator.alloc(f32, d);
        defer allocator.free(w_scratch);

        for (0..clusters) |c| {
            const start = seg_offsets[c];
            const end = seg_offsets[c + 1];
            const m = end - start;
            const mf: f32 = @floatFromInt(m);
            counts[c] = mf;
            log_counts[c] = @log(mf);
            const cd = c * d;
            for (0..d) |dim| {
                const mean = sum_k[cd + dim] / mf;
                centroid[cd + dim] = mean;
                vmean[cd + dim] = sum_v[cd + dim] / mf;
                diag_resid[cd + dim] = @max(sum_k2[cd + dim] / mf - mean * mean, 0);
            }
            @memset(lam[c * cfg.rank ..][0..cfg.rank], 0);
            @memset(eig[cd * cfg.rank ..][0 .. cfg.rank * d], 0);
            if (m < 2) continue;
            // Centered member rows for power iteration.
            for (0..m) |zi| {
                const src = ordered_k[(start + zi) * d ..][0..d];
                for (0..d) |dim| z_scratch[zi * d + dim] = src[dim] - centroid[cd + dim];
            }
            const rows = z_scratch[0 .. m * d];
            for (0..cfg.rank) |r| {
                const seed: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
                @memset(u_scratch, seed);
                for (0..cfg.power_iterations) |_| {
                    @memset(w_scratch, 0);
                    for (0..m) |zi| {
                        const zrow = rows[zi * d ..][0..d];
                        accumulateF32(w_scratch, zrow, dotF32(u_scratch, zrow));
                    }
                    const norm = @sqrt(dotF32(w_scratch, w_scratch));
                    if (norm < 1e-20) break;
                    for (0..d) |dim| u_scratch[dim] = w_scratch[dim] / norm;
                }
                var energy: f32 = 0;
                for (0..m) |zi| {
                    const p = dotF32(u_scratch, rows[zi * d ..][0..d]);
                    energy += p * p;
                }
                const lam_r = energy / mf;
                lam[c * cfg.rank + r] = lam_r;
                const urow = eig[(c * cfg.rank + r) * d ..][0..d];
                @memcpy(urow, u_scratch);
                for (0..m) |zi| {
                    const zrow = rows[zi * d ..][0..d];
                    accumulateF32(zrow, u_scratch, -dotF32(u_scratch, zrow));
                }
                for (0..d) |dim| {
                    diag_resid[cd + dim] = @max(diag_resid[cd + dim] - lam_r * urow[dim] * urow[dim], 0);
                }
            }
        }

        // Append the region after the frozen prefix (capacity-managed; the
        // frozen prefix is never copied except on amortized capacity growth).
        const oldc = plan.clusters;
        const newc = oldc + clusters;
        const new_rows = old_rows + n;
        try growCap(u32, allocator, &plan.offsets, if (oldc > 0) oldc + 1 else 0, newc + 1);
        plan.offsets[0] = 0;
        for (0..clusters) |c| plan.offsets[oldc + 1 + c] = @intCast(old_rows + offsets_list.items[c + 1]);
        try growCap(f32, allocator, &plan.counts, oldc, newc);
        @memcpy(plan.counts[oldc..][0..clusters], counts);
        try growCap(f32, allocator, &plan.log_counts, oldc, newc);
        @memcpy(plan.log_counts[oldc..][0..clusters], log_counts);
        try growCap(f32, allocator, &plan.centroid, oldc * d, newc * d);
        @memcpy(plan.centroid[oldc * d ..][0 .. clusters * d], centroid);
        try growCap(f32, allocator, &plan.vmean, oldc * d, newc * d);
        @memcpy(plan.vmean[oldc * d ..][0 .. clusters * d], vmean);
        try growCap(f32, allocator, &plan.diag_resid, oldc * d, newc * d);
        @memcpy(plan.diag_resid[oldc * d ..][0 .. clusters * d], diag_resid);
        try growCap(f32, allocator, &plan.eig, oldc * cfg.rank * d, newc * cfg.rank * d);
        @memcpy(plan.eig[oldc * cfg.rank * d ..][0 .. clusters * cfg.rank * d], eig);
        try growCap(f32, allocator, &plan.lam, oldc * cfg.rank, newc * cfg.rank);
        @memcpy(plan.lam[oldc * cfg.rank ..][0 .. clusters * cfg.rank], lam);
        switch (fmt) {
            .f16 => {
                try growCap(f16, allocator, &plan.packed_k, old_rows * d, new_rows * d);
                @memcpy(plan.packed_k[old_rows * d ..][0 .. n * d], packed_k);
                try growCap(f16, allocator, &plan.packed_v, old_rows * d, new_rows * d);
                @memcpy(plan.packed_v[old_rows * d ..][0 .. n * d], packed_v);
            },
            .q8_0 => {
                try growCap(BlockQ8_0, allocator, &plan.packed_k_q8, old_rows * bpr, new_rows * bpr);
                @memcpy(plan.packed_k_q8[old_rows * bpr ..][0 .. n * bpr], packed_k_q8);
                try growCap(BlockQ8_0, allocator, &plan.packed_v_q8, old_rows * bpr, new_rows * bpr);
                @memcpy(plan.packed_v_q8[old_rows * bpr ..][0 .. n * bpr], packed_v_q8);
            },
        }
        plan.clusters = newc;
        plan.format = fmt;
        if (n >= cfg.block_size) {
            plan.frozen_end = seal_end;
            plan.frozen_clusters = newc;
            plan.frozen_rows = new_rows;
        }
    }
};

const HeadTask = struct {
    state: *State,
    layer_i: usize,
    kv_head: usize,
    head_i: usize,
    q: []const f32,
    k: []const f16,
    v: []const f16,
    cached_len: usize,
};

fn runHeadTask(task: *HeadTask) void {
    const self = task.state;
    const d = self.d;
    const beta: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
    const plan = &self.plans[task.layer_i * self.kv_heads + task.kv_head];
    const cap = self.scratch_cap;
    const clusters = plan.clusters;
    const head_i = task.head_i;
    const query = task.q[head_i * d ..][0..d];
    const base = head_i * cap;
    const prio_a = self.prio_a[base..][0..clusters];
    const est = self.prio_est[base..][0..clusters];
    const sorted = self.sorted[base..][0..clusters];
    const total_est: f64 = if (clusters > 0) self.head_total_est[head_i] else 0;
    const tau = self.taus[task.layer_i * self.q_heads + head_i];
    if (self.calibrating) {
        if (self.config.hierarchical and plan.node_count > 0) {
            self.calibrateHeadHier(plan, query, task.k, task.v, task.kv_head, task.cached_len, beta, task.layer_i, head_i) catch {
                self.attendHead(plan, query, task.k, task.v, task.kv_head, task.cached_len, beta, tau, task.layer_i, head_i, prio_a, est, sorted, total_est);
            };
        } else self.calibrateHead(plan, query, task.k, task.v, task.kv_head, task.cached_len, beta, task.layer_i, head_i, prio_a, est, sorted, total_est) catch {
            self.attendHead(plan, query, task.k, task.v, task.kv_head, task.cached_len, beta, tau, task.layer_i, head_i, prio_a, est, sorted, total_est);
        };
    } else {
        self.attendHead(plan, query, task.k, task.v, task.kv_head, task.cached_len, beta, tau, task.layer_i, head_i, prio_a, est, sorted, total_est);
    }
}

const HeapEntry = struct { prio: f64, a0: f32, child: i32 };

fn heapPush(hp: []f64, ha: []f32, hc: []i32, n: *usize, prio: f64, a0: f32, child: i32) void {
    var i = n.*;
    hp[i] = prio;
    ha[i] = a0;
    hc[i] = child;
    n.* += 1;
    while (i > 0) {
        const parent = (i - 1) / 2;
        if (hp[parent] >= hp[i]) break;
        std.mem.swap(f64, &hp[parent], &hp[i]);
        std.mem.swap(f32, &ha[parent], &ha[i]);
        std.mem.swap(i32, &hc[parent], &hc[i]);
        i = parent;
    }
}

fn heapPop(hp: []f64, ha: []f32, hc: []i32, n: *usize) HeapEntry {
    const top = HeapEntry{ .prio = hp[0], .a0 = ha[0], .child = hc[0] };
    n.* -= 1;
    hp[0] = hp[n.*];
    ha[0] = ha[n.*];
    hc[0] = hc[n.*];
    var i: usize = 0;
    while (true) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        var big = i;
        if (l < n.* and hp[l] > hp[big]) big = l;
        if (r < n.* and hp[r] > hp[big]) big = r;
        if (big == i) break;
        std.mem.swap(f64, &hp[big], &hp[i]);
        std.mem.swap(f32, &ha[big], &ha[i]);
        std.mem.swap(i32, &hc[big], &hc[i]);
        i = big;
    }
    return top;
}

/// Full per-coordinate variance of a child (leaf: diag residual plus the
/// rank-r eigen energy; node: stored full diag).
fn childFullVar(plan: *const Plan, child: i32, j: usize, d: usize, rank: usize) f32 {
    if (child >= 0) return plan.node_diag[@as(usize, @intCast(child)) * d + j];
    const li: usize = @intCast(~child);
    var v = plan.diag_resid[li * d + j];
    for (0..rank) |r| {
        const u = plan.eig[(li * rank + r) * d + j];
        v += plan.lam[li * rank + r] * u * u;
    }
    return v;
}

fn estDesc(est: []const f64, a: u32, b: u32) bool {
    return est[a] > est[b];
}

const Builder = struct {
    keys: []const f32,
    norms: []const f32,
    d: usize,
    cluster_size: usize,
    order: []u32,
    offsets: *std.ArrayList(u32),
    allocator: Allocator,

    fn distance2(self: *const Builder, a: u32, b: u32) f64 {
        const d = self.d;
        const ka = self.keys[@as(usize, a) * d ..][0..d];
        const kb = self.keys[@as(usize, b) * d ..][0..d];
        return @as(f64, self.norms[a]) + @as(f64, self.norms[b]) - 2.0 * @as(f64, dotF32(ka, kb));
    }

    fn split(self: *Builder, start: usize, end: usize) !void {
        if (end - start <= self.cluster_size) {
            try self.offsets.append(self.allocator, @intCast(end));
            return;
        }
        const tokens = self.order[start..end];
        var seed_a: u32 = tokens[0];
        var farthest: f64 = -1;
        for (tokens) |token| {
            const value = self.distance2(tokens[0], token);
            if (value > farthest) {
                farthest = value;
                seed_a = token;
            }
        }
        var seed_b: u32 = seed_a;
        farthest = -1;
        for (tokens) |token| {
            const value = self.distance2(seed_a, token);
            if (value > farthest) {
                farthest = value;
                seed_b = token;
            }
        }
        if (farthest > 1e-20) {
            const d = self.d;
            const ka = self.keys[@as(usize, seed_a) * d ..][0..d];
            const kb = self.keys[@as(usize, seed_b) * d ..][0..d];
            const diff = try self.allocator.alloc(f32, d);
            defer self.allocator.free(diff);
            for (diff, ka, kb) |*x, a_c, b_c| x.* = b_c - a_c;
            const proj = try self.allocator.alloc(f64, tokens.len);
            defer self.allocator.free(proj);
            for (tokens, proj) |token, *p| {
                p.* = dotF32(self.keys[@as(usize, token) * d ..][0..d], diff);
            }
            const Ctx = struct {
                proj: []const f64,
                tokens: []const u32,
                fn less(ctx: @This(), ai: usize, bi: usize) bool {
                    if (ctx.proj[ai] != ctx.proj[bi]) return ctx.proj[ai] < ctx.proj[bi];
                    return ctx.tokens[ai] < ctx.tokens[bi];
                }
            };
            const idx = try self.allocator.alloc(usize, tokens.len);
            defer self.allocator.free(idx);
            for (idx, 0..) |*x, i| x.* = i;
            std.mem.sort(usize, idx, Ctx{ .proj = proj, .tokens = tokens }, Ctx.less);
            const sorted = try self.allocator.alloc(u32, tokens.len);
            defer self.allocator.free(sorted);
            for (idx, 0..) |src, i| sorted[i] = tokens[src];
            @memcpy(tokens, sorted);
        }
        const middle = start + (end - start) / 2;
        try self.split(start, middle);
        try self.split(middle, end);
    }
};

/// Cluster-level segmented row sums via the segmentSum core op.
/// Amortized-doubling append support: grow `arr` to at least `need`
/// elements, copying the first `used` (the frozen prefix) into the new
/// allocation. No-op when capacity suffices.
fn growCap(comptime T: type, allocator: Allocator, arr: *[]T, used: usize, need: usize) !void {
    if (arr.len >= need) return;
    const cap = @max(arr.len * 2, need);
    const next = try allocator.alloc(T, cap);
    @memcpy(next[0..used], arr.*[0..used]);
    if (arr.len > 0) allocator.free(arr.*);
    arr.* = next;
}

fn segmentSumRows(
    ctx: *ExecContext,
    rows: []const f32,
    n: usize,
    d: usize,
    offsets: []const usize,
    allocator: Allocator,
) ![]f32 {
    var x = try ctx.fromSliceRank(2, .{ n, d }, rows);
    defer x.deinit();
    var summed = try ctx.segmentSumAxisRank(2, &x, 0, offsets);
    defer summed.deinit();
    const out = try allocator.alloc(f32, (offsets.len - 1) * d);
    @memcpy(out, summed.dataConst());
    return out;
}

fn dotF32(a: []const f32, b: []const f32) f32 {
    const V = @Vector(8, f32);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var i: usize = 0;
    while (i + 16 <= a.len) : (i += 16) {
        acc0 = @mulAdd(V, a[i..][0..8].*, b[i..][0..8].*, acc0);
        acc1 = @mulAdd(V, a[i + 8 ..][0..8].*, b[i + 8 ..][0..8].*, acc1);
    }
    while (i + 8 <= a.len) : (i += 8) {
        acc0 = @mulAdd(V, a[i..][0..8].*, b[i..][0..8].*, acc0);
    }
    var total: f32 = @reduce(.Add, acc0 + acc1);
    while (i < a.len) : (i += 1) total += a[i] * b[i];
    return total;
}

/// Two dots against a shared right-hand row, loaded once: {a0.b, a1.b}.
fn dot2F32(a0: []const f32, a1: []const f32, b: []const f32) [2]f32 {
    const V = @Vector(8, f32);
    var x00: V = @splat(0);
    var x01: V = @splat(0);
    var x10: V = @splat(0);
    var x11: V = @splat(0);
    var i: usize = 0;
    while (i + 16 <= b.len) : (i += 16) {
        const b0: V = b[i..][0..8].*;
        const b1: V = b[i + 8 ..][0..8].*;
        x00 = @mulAdd(V, a0[i..][0..8].*, b0, x00);
        x10 = @mulAdd(V, a1[i..][0..8].*, b0, x10);
        x01 = @mulAdd(V, a0[i + 8 ..][0..8].*, b1, x01);
        x11 = @mulAdd(V, a1[i + 8 ..][0..8].*, b1, x11);
    }
    while (i + 8 <= b.len) : (i += 8) {
        const bv: V = b[i..][0..8].*;
        x00 = @mulAdd(V, a0[i..][0..8].*, bv, x00);
        x10 = @mulAdd(V, a1[i..][0..8].*, bv, x10);
    }
    var t0: f32 = @reduce(.Add, x00 + x01);
    var t1: f32 = @reduce(.Add, x10 + x11);
    while (i < b.len) : (i += 1) {
        t0 += a0[i] * b[i];
        t1 += a1[i] * b[i];
    }
    return .{ t0, t1 };
}

const dotF16 = fucina.internal.backend_mod.vector_impl.primitives.dotF32F16;

/// One exact-read batch under the shared online softmax gauge: score all
/// rows, raise the gauge if the batch's max score exceeds it (rescaling the
/// partial numerator and mass), exponentiate in place, and accumulate the
/// weighted values. This is the dense kernel's score-buffer + vexpf shape
/// applied per batch (sinks, unsealed suffix, one opened cluster each).
/// q8_0 sibling of exactBatch over packed cluster blocks: integer-path
/// scoring (query pre-quantized to q8_0 once per head call) and paired
/// weighted dequant-accumulate, both from the backend quant catalog.
fn exactBatchQ8(
    numer: []f32,
    gauge: *f32,
    exact_w: *f64,
    scores: []f32,
    q_q8: []const BlockQ8_0,
    q_scales: []const f32,
    k_blocks: []const BlockQ8_0,
    v_blocks: []const BlockQ8_0,
    bpr: usize,
    beta: f32,
) void {
    const m = scores.len;
    if (m == 0) return;
    const simd = fucina.internal.backend_mod.vector_impl;
    var i: usize = 0;
    while (i + 2 <= m) : (i += 2) {
        const pair = qkern.q8_0.vecDotQ8_0Q8_0x2(q_q8, q_scales, k_blocks[i * bpr ..][0..bpr], k_blocks[(i + 1) * bpr ..][0..bpr]);
        scores[i] = pair[0];
        scores[i + 1] = pair[1];
    }
    if (i < m) scores[i] = qkern.q8_0.vecDotQ8_0Q8_0(q_q8, k_blocks[i * bpr ..][0..bpr]);
    const mx = beta * simd.primitives.vecMaxReduce(scores);
    if (mx > gauge.*) {
        if (gauge.* != -std.math.inf(f32)) {
            const rescale: f32 = @exp(gauge.* - mx);
            simd.primitives.vecScale(numer, numer, rescale);
            exact_w.* *= rescale;
        }
        gauge.* = mx;
    }
    exact_w.* += simd.primitives.vecExpAffineSumInPlace(scores, beta, -gauge.*);
    i = 0;
    while (i + 2 <= m) : (i += 2) {
        qkern.q8_0.weightedQ8_0Row2(true, numer, v_blocks[i * bpr ..][0..bpr], scores[i], v_blocks[(i + 1) * bpr ..][0..bpr], scores[i + 1]);
    }
    if (i < m) qkern.q8_0.weightedQ8_0Row(true, numer, v_blocks[i * bpr ..][0..bpr], scores[i]);
}

fn exactBatch(
    numer: []f32,
    gauge: *f32,
    exact_w: *f64,
    scores: []f32,
    query: []const f32,
    k_rows: []const f16,
    v_rows: []const f16,
    stride: usize,
    d: usize,
    beta: f32,
) void {
    if (scores.len == 0) return;
    const simd = fucina.internal.backend_mod.vector_impl;
    simd.primitives.scoreRows4F16(scores, query[0..d], k_rows, stride);
    const m = beta * simd.primitives.vecMaxReduce(scores);
    if (m > gauge.*) {
        if (gauge.* != -std.math.inf(f32)) {
            const rescale: f32 = @exp(gauge.* - m);
            simd.primitives.vecScale(numer, numer, rescale);
            exact_w.* *= rescale;
        }
        gauge.* = m;
    }
    exact_w.* += simd.primitives.vecExpAffineSumInPlace(scores, beta, -gauge.*);
    simd.primitives.weightedAccumRows4F16(numer[0..d], scores, v_rows, stride);
}

fn accumulateF32(acc: []f32, row: []const f32, scale: f32) void {
    const V = @Vector(8, f32);
    const sv: V = @splat(scale);
    var i: usize = 0;
    while (i + 8 <= acc.len) : (i += 8) {
        const av: V = acc[i..][0..8].*;
        const rv: V = row[i..][0..8].*;
        acc[i..][0..8].* = av + sv * rv;
    }
    while (i < acc.len) : (i += 1) acc[i] += scale * row[i];
}

fn accumulateF16(acc: []f32, row: []const f16, scale: f32) void {
    const V = @Vector(8, f32);
    const H = @Vector(8, f16);
    const sv: V = @splat(scale);
    var i: usize = 0;
    while (i + 8 <= acc.len) : (i += 8) {
        const av: V = acc[i..][0..8].*;
        const hv: H = row[i..][0..8].*;
        const rv: V = @floatCast(hv);
        acc[i..][0..8].* = av + sv * rv;
    }
    while (i < acc.len) : (i += 1) acc[i] += scale * @as(f32, @floatCast(row[i]));
}

/// {sum q0[i]^2 w[i], sum q1[i]^2 w[i]} with the weight row loaded once.
fn weightedSquareDot2(q0: []const f32, q1: []const f32, w: []const f32) [2]f32 {
    const V = @Vector(8, f32);
    var x0: V = @splat(0);
    var x1: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= w.len) : (i += 8) {
        const wv: V = w[i..][0..8].*;
        const q0v: V = q0[i..][0..8].*;
        const q1v: V = q1[i..][0..8].*;
        x0 = @mulAdd(V, q0v * q0v, wv, x0);
        x1 = @mulAdd(V, q1v * q1v, wv, x1);
    }
    var t0: f32 = @reduce(.Add, x0);
    var t1: f32 = @reduce(.Add, x1);
    while (i < w.len) : (i += 1) {
        t0 += q0[i] * q0[i] * w[i];
        t1 += q1[i] * q1[i] * w[i];
    }
    return .{ t0, t1 };
}

fn weightedSquareDot(q: []const f32, w: []const f32) f32 {
    const V = @Vector(8, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= q.len) : (i += 8) {
        const qv: V = q[i..][0..8].*;
        const wv: V = w[i..][0..8].*;
        acc += qv * qv * wv;
    }
    var total: f32 = @reduce(.Add, acc);
    while (i < q.len) : (i += 1) total += q[i] * q[i] * w[i];
    return total;
}

/// The descriptor runner's research seam adapter
/// (`runner.AttentionOverride`): install with
/// `model.attention_override = subq.attentionOverride(&state)` and drive
/// the stock `forwardStep`. The override takes single query rows over f16
/// KV caches (the operator's domain) and returns null otherwise, so the
/// stock kernels keep every other call.
pub fn attentionOverride(sq: *State) runner_mod.AttentionOverride {
    return .{ .ctx = sq, .call = attendOverride };
}

fn attendOverride(
    ptr: *anyopaque,
    ctx: *ExecContext,
    config: runner_mod.Descriptor,
    layer_i: usize,
    q: *const fucina.Tensor(.{ .seq, .head, .d }),
    kv: *kv_cache_mod.KvCache,
    cached_len: usize,
) anyerror!?fucina.Tensor(.{ .seq, .attn }) {
    const sq: *State = @ptrCast(@alignCast(ptr));
    if (kv.dtype != .f16 or q.dim(.seq) != 1) return null;
    const heads = config.num_attention_heads;
    const d = config.head_dim;
    // Borrow the contiguous query row when possible; the state's persistent
    // bridge buffers cover the fallback and the output (no per-token
    // allocations in this glue).
    const q_flat: []const f32 = blk: {
        if (q.dataConst()) |qd| {
            if (qd.len == heads * d) break :blk qd;
        } else |_| {}
        try q.copyTo(sq.bridge_q);
        break :blk sq.bridge_q;
    };
    const out = sq.bridge_out;
    const row_len = cached_len * config.num_key_value_heads * d;
    try sq.attend(
        ctx,
        layer_i,
        q_flat,
        (try kv.k[layer_i].dataConst())[0..row_len],
        (try kv.v[layer_i].dataConst())[0..row_len],
        cached_len,
        out,
    );
    return try fucina.Tensor(.{ .seq, .attn }).fromSlice(ctx, .{ 1, heads * d }, out);
}

test {
    _ = @import("subq_tests.zig");
}
