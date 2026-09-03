//! Fused attention: grouped causal forward (f32/f16/q8_0 KV arms, head- and
//! query-tile task shapes), the tiled backward, its BLAS-strip variant
//! (behind `backend.blas.available` and the `attn_bwd_blas` gate), and the
//! GPU-offload dispatch. Domain module: every op receives an explicit
//! `*ExecContext`; `exec.zig`'s struct body carries the public aliases.
const std = @import("std");
const backend_mod = @import("../backend.zig");
const offload = backend_mod.offload;
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");
const parallel = @import("../parallel.zig");
const tuning = @import("../tuning.zig");
const storage = @import("../storage.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const DType = tensor.DType;
const Tensor = tensor.Tensor;

// The attention kernel bodies, their Task payloads (pure slices/dims) and
// run*Task pool adapters live in the backend leaf (backend/vector/
// attention.zig): serial kernel entries through the conformed
// `backend.kernels` table, tasks/adapters/constants through the
// `backend.attention` seam. This file keeps validation, layout,
// allocation and dispatch.
const attn = backend_mod.attention;
const kernels = backend_mod.kernels;
const kvDtypeOf = attn.kvDtypeOf;
const BlockQ8_0 = dtype_mod.BlockQ8_0;
const q8_0_block_size = dtype_mod.blockSize(.q8_0);
const attention_q8_max_d = attn.attention_q8_max_d;
const GroupedCausalAttentionTask = attn.GroupedCausalAttentionTask;
const GroupedCausalAttentionPairTask = attn.GroupedCausalAttentionPairTask;
const attention_tile_rows = attn.attention_tile_rows;
const attention_tile_max_d = attn.attention_tile_max_d;
const attention_tiled_min_q_seq = attn.attention_tiled_min_q_seq;
const GroupedCausalAttentionTiledTask = attn.GroupedCausalAttentionTiledTask;
const GroupedCausalAttentionBackwardTask = attn.GroupedCausalAttentionBackwardTask;
const runGroupedCausalAttentionTask = attn.runGroupedCausalAttentionTask;
const runGroupedCausalAttentionPairTask = attn.runGroupedCausalAttentionPairTask;
const runGroupedCausalAttentionTiledTask = attn.runGroupedCausalAttentionTiledTask;
const runGroupedCausalAttentionBackwardTask = attn.runGroupedCausalAttentionBackwardTask;
const groupedCausalAttentionHeads = kernels.groupedCausalAttentionHeads;
const groupedCausalAttentionHeadPairs = kernels.groupedCausalAttentionHeadPairs;
const attentionTileKeyCount = attn.attentionTileKeyCount;
const groupedCausalAttentionBackwardKvHeads = kernels.groupedCausalAttentionBackwardKvHeads;
const attention_bwd_tile_rows = attn.attention_bwd_tile_rows;
const GroupedCausalAttentionBackwardTiledTask = attn.GroupedCausalAttentionBackwardTiledTask;
const runGroupedCausalAttentionBackwardTiledTask = attn.runGroupedCausalAttentionBackwardTiledTask;
const groupedCausalAttentionBackwardTiles = kernels.groupedCausalAttentionBackwardTiles;
const attention_bwd_blas_tile_rows = attn.attention_bwd_blas_tile_rows;
const runGroupedCausalAttentionBackwardBlasTiledTask = attn.runGroupedCausalAttentionBackwardBlasTiledTask;
const groupedCausalAttentionBackwardBlasTiles = kernels.groupedCausalAttentionBackwardBlasTiles;
const AttentionBackwardReduceTask = attn.AttentionBackwardReduceTask;
const runAttentionBackwardReduceTask = attn.runAttentionBackwardReduceTask;
const attentionBackwardReduceRows = kernels.attentionBackwardReduceRows;
const hasAdjacentKvHeadPairs = attn.hasAdjacentKvHeadPairs;
const GroupedCausalAttentionMultiTask = attn.GroupedCausalAttentionMultiTask;
const runGroupedCausalAttentionMultiTask = attn.runGroupedCausalAttentionMultiTask;
const groupedCausalAttentionMultiUnits = kernels.groupedCausalAttentionMultiUnits;
const kvRowElems = attn.kvRowElems;

pub const GroupedCausalAttentionBackwardResult = struct {
    q: ?tensor.Tensor = null,
    k: ?tensor.Tensor = null,
    v: ?tensor.Tensor = null,

    pub fn deinit(self: *GroupedCausalAttentionBackwardResult) void {
        if (self.q) |*value| value.deinit();
        if (self.k) |*value| value.deinit();
        if (self.v) |*value| value.deinit();
        self.* = undefined;
    }
};

const grouped_attention_backward_gemm_work_threshold: usize = 16 * 1024;

/// The BLAS-strip backward route is the default on BLAS builds — the same
/// kernel-selection contract as `shouldUseBlas` on the plain matmuls.
/// FUCINA_ATTN_BWD_BLAS=0 (`tuning.Table.attn_bwd_blas`) reverts to the
/// register-tiled route (the A/B switch and the escape hatch for non-BLAS
/// parity work).
fn attentionBackwardBlasEnabled() bool {
    if (comptime !(backend_mod.active_kind == .native and backend_mod.native_uses_blas)) return false;
    return tuning.get().attn_bwd_blas;
}

/// Forward-saved stats route for the backward softmax reconstruction:
/// FUCINA_ATTN_BWD_STATS=0 pins the 3-pass recompute (the A/B and
/// emergency-revert switch), =1 forces on. Read once, cached
/// (`tuning.Table.attn_bwd_stats`). Only consulted when the caller has
/// stats at all (the autograd record); the stats-less exec path always
/// recomputes.
fn attnBwdStatsEnabled() bool {
    return tuning.get().attn_bwd_stats;
}

/// Mask mode of `groupedAttention`: causal (each query attends its prefix)
/// or bidirectional (every query attends every key: the block-diffusion
/// canvas and OmniVoice encoder attention).
pub const AttentionMask = enum { causal, bidirectional };

/// K/V operand of `groupedAttention`: one variant per cache representation
/// the attention kernels accept.
pub const KvView = union(enum) {
    /// f32 K/V tensors, `[kv_seq, kv_heads, d]` (training graphs, f32 caches).
    f32: struct { k: *const Tensor, v: *const Tensor },
    /// f16 K/V tensors with the same layout (the decode KV cache): half the
    /// bandwidth, widened to f32 in the kernel. Q and the output stay f32.
    f16: struct { k: *const tensor.TensorOf(.f16), v: *const tensor.TensorOf(.f16) },
    /// q8_0 block cache (~quarter the f32 bandwidth and half f16's):
    /// `kv_seq * kv_heads * d/32` BlockQ8_0 laid out `[kv_seq, kv_heads, d/32]`,
    /// so each (position, kv_head) row segment is `d/32` consecutive blocks.
    /// Raw blocks carry no shape, so the view names `kv_seq` and `kv_heads`.
    /// Kernels dequantize each row into per-task L1 scratch as they stream;
    /// traffic from the cache stays the quantized 34 bytes/block. Requires
    /// `d % 32 == 0` and `d <= attention_q8_max_d`. Causal only.
    q8: struct { k: []const BlockQ8_0, v: []const BlockQ8_0, kv_seq: usize, kv_heads: usize },
    /// Ragged multi-stream decode over per-stream f16 caches (the batch-N
    /// decode shape): `q` is `[n_streams, heads, d]`, exactly one query per
    /// stream, and query row `s` attends ALL `lens[s]` cached positions of
    /// stream `s`. `k[s]`/`v[s]` hold at least `lens[s]` leading
    /// `[kv_heads, d]` rows of that stream's cache layer. Dispatch schedules
    /// flattened (stream, head-unit) items weighted by stream length over
    /// the SAME per-query kernels m=1 decode uses, so each stream's output
    /// is bit-identical to its own single-stream `.f16` call. Causal only,
    /// full reach.
    multi_f16: struct { k: []const []const f16, v: []const []const f16, lens: []const usize, kv_heads: usize },
    /// As `.multi_f16` for q8_0 caches: `k[s]`/`v[s]` hold
    /// `lens[s] * kv_heads * d/32` leading BlockQ8_0 laid out
    /// `[len, kv_heads, d/32]` (the `kBlocks`/`vBlocks` shape). Requires
    /// `d % 32 == 0` and `d <= attention_q8_max_d`.
    multi_q8: struct { k: []const []const BlockQ8_0, v: []const []const BlockQ8_0, lens: []const usize, kv_heads: usize },
};

/// Options of `groupedAttention`. Combinations with no kernel return
/// `error.UnsupportedAttentionVariant` at runtime; the `Tensor.groupedAttention`
/// facade rejects them at comptime.
pub const AttentionOptions = struct {
    mask: AttentionMask = .causal,
    /// Sliding-window attention: 0 = full reach; else a query at absolute
    /// position `p` attends only keys in `[max(0, p-window+1), p]` (Gemma's
    /// local SWA layers). Causal only: a bidirectional reach limit is
    /// realized by narrowing the K/V views instead.
    window: usize = 0,
    /// Additive f32 bias of shape `[q_seq, kv_seq]` added to the SCALED
    /// scores before the softmax: score(query s, key kv) =
    /// dot(q_s, k_kv) * scale_value + bias[s][kv] (OmniVoice's uncond CFG
    /// +1.0/0.0 row, ggml_soft_max_ext mask semantics). An additive soft
    /// bias, NOT -inf masking: a -inf bias value poisons its query row on
    /// the tiled kernel like a NaN logit does. `.bidirectional` + `.f32`
    /// KV only.
    bias: ?*const Tensor = null,
    /// Per-(head, query) softmax {max, sum_exp} capture, interleaved f32
    /// pairs of length heads * q_seq * 2. Each kernel records the
    /// normalizer IT used, and the output is BITWISE identical with or
    /// without capture (write-only). The stats feed
    /// `groupedAttentionBackward`, which then rebuilds this forward's
    /// probabilities in one pass instead of three. `.f32` KV only.
    stats_out: ?[]f32 = null,
};

/// Grouped-query (GQA) attention over every KV representation: `q` is
/// `[q_seq, heads, d]` f32, `kv_head_for_head[h]` maps each query head to
/// its KV head, and the result is `[q_seq, heads * d]` f32. `kv` selects
/// the cache representation (see `KvView`), `opts` the mask/window/bias/
/// stats variant (see `AttentionOptions`). Requires `q_seq <= kv_seq`
/// (validated, like the shapes and the head map). Kernel selection is by
/// shape: long prefill runs the query-tiled online-softmax kernel (same
/// math as the per-query kernels, different summation grouping, ~1e-6
/// relative), decode and short prefill stay on the bit-identical per-query
/// kernels. On GPU builds the f32/f16 arms offload behind the provider's
/// attention gate; every other variant stays on CPU.
///
/// Option combinations with no kernel return
/// `error.UnsupportedAttentionVariant`: `.bidirectional` with a window,
/// `.bias` with `.causal` or a non-`.f32` view, `.stats_out` on a
/// non-`.f32` view, `.bidirectional` on the `.q8` or multi-stream views,
/// and a window on the multi-stream views.
pub fn groupedAttention(
    self: *ExecContext,
    q: *const Tensor,
    kv: KvView,
    kv_head_for_head: []const usize,
    scale_value: f32,
    opts: AttentionOptions,
) !Tensor {
    if (opts.mask == .bidirectional and opts.window != 0) return error.UnsupportedAttentionVariant;
    if (opts.bias != null and (opts.mask == .causal or kv != .f32)) return error.UnsupportedAttentionVariant;
    if (opts.stats_out != null and kv != .f32) return error.UnsupportedAttentionVariant;
    switch (kv) {
        .f32 => |view| return groupedCausalAttentionImpl(self, f32, q, view.k, view.v, kv_head_for_head, scale_value, opts.window, opts.mask == .causal, opts.bias, opts.stats_out),
        .f16 => |view| return groupedCausalAttentionImpl(self, f16, q, view.k, view.v, kv_head_for_head, scale_value, opts.window, opts.mask == .causal, null, null),
        .q8 => |view| {
            if (opts.mask != .causal) return error.UnsupportedAttentionVariant;
            return groupedCausalAttentionQ8KvImpl(self, q, view.k, view.v, view.kv_seq, view.kv_heads, kv_head_for_head, scale_value, opts.window);
        },
        .multi_f16 => |view| {
            if (opts.mask != .causal or opts.window != 0) return error.UnsupportedAttentionVariant;
            return groupedCausalAttentionMultiImpl(self, f16, q, view.k, view.v, view.lens, view.kv_heads, kv_head_for_head, scale_value);
        },
        .multi_q8 => |view| {
            if (opts.mask != .causal or opts.window != 0) return error.UnsupportedAttentionVariant;
            return groupedCausalAttentionMultiImpl(self, BlockQ8_0, q, view.k, view.v, view.lens, view.kv_heads, kv_head_for_head, scale_value);
        },
    }
}

/// The fused grouped attention backward's request: the forward operands
/// and head map, the upstream gradient, the softmax parameters, the
/// forward's optional saved row stats, and which gradients to produce.
pub const AttentionBackwardRequest = struct {
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    /// Upstream gradient, `[q_seq, heads * d]`.
    gy: *const Tensor,
    kv_head_for_head: []const usize,
    scale: f32,
    window: usize = 0,
    causal: bool = true,
    /// The forward's saved per-(head, query) {max, sum_exp} pairs from
    /// `groupedAttention`'s `.stats_out` capture (length heads * q_seq * 2),
    /// or null to recompute them.
    stats: ?[]const f32 = null,
    need: struct { q: bool = true, k: bool = true, v: bool = true } = .{},
};

/// With `request.stats` the tiled route rebuilds the forward's
/// probabilities in one pass instead of the max/sum recompute (gated by
/// FUCINA_ATTN_BWD_STATS=0); the small-shape per-kv-head route always
/// recomputes. The forward output is not an input: the tiled route builds
/// full-width probability and dO·Vᵀ panels per query tile and takes the
/// softmax-backward row dot from them, so the flash-style rowsum(dO ∘ O)
/// term would only replace that in-panel reduction, not a pass.
pub fn groupedAttentionBackward(self: *ExecContext, request: AttentionBackwardRequest) !GroupedCausalAttentionBackwardResult {
    const q = request.q;
    const k = request.k;
    const v = request.v;
    const gy = request.gy;
    const kv_head_for_head = request.kv_head_for_head;
    const scale_value = request.scale;
    const window = request.window;
    const causal = request.causal;
    const stats = request.stats;
    const need_q = request.need.q;
    const need_k = request.need.k;
    const need_v = request.need.v;

    const q_view = try q.rankView(3);
    const k_view = try k.rankView(3);
    const v_view = try v.rankView(3);
    const gy_view = try gy.rankView(2);

    const q_seq = q_view.shape[0];
    const kv_seq = k_view.shape[0];
    const heads = q_view.shape[1];
    const d = q_view.shape[2];
    const kv_heads = k_view.shape[1];
    if (kv_head_for_head.len != heads) return tensor.TensorError.InvalidShape;
    if (q_seq == 0 or q_seq > kv_seq) return tensor.TensorError.InvalidShape;
    if (v_view.shape[0] != kv_seq) return tensor.TensorError.InvalidShape;
    if (k_view.shape[2] != d or v_view.shape[2] != d) return tensor.TensorError.InvalidShape;
    if (v_view.shape[1] != kv_heads) return tensor.TensorError.InvalidShape;
    if (gy_view.shape[0] != q_seq or gy_view.shape[1] != heads * d) return tensor.TensorError.ShapeMismatch;
    for (kv_head_for_head) |kv_head_i| {
        if (kv_head_i >= kv_heads) return tensor.TensorError.IndexOutOfBounds;
    }

    var qq = try self.prepareContiguous(.f32, q);
    defer qq.deinit();
    var kk = try self.prepareContiguous(.f32, k);
    defer kk.deinit();
    var vv = try self.prepareContiguous(.f32, v);
    defer vv.deinit();
    var ggy = try self.prepareContiguous(.f32, gy);
    defer ggy.deinit();

    const q_data = qq.tensor().dataConst();
    const k_data = kk.tensor().dataConst();
    const v_data = vv.tensor().dataConst();
    const gy_data = ggy.tensor().dataConst();

    var result = GroupedCausalAttentionBackwardResult{};
    errdefer result.deinit();
    if (need_q) result.q = try self.zeros(.f32, .{ q_seq, heads, d });
    if (need_k) result.k = try self.zeros(.f32, .{ kv_seq, kv_heads, d });
    if (need_v) result.v = try self.zeros(.f32, .{ kv_seq, kv_heads, d });
    const q_grad: ?[]f32 = if (result.q) |*value| value.data() else null;
    const k_grad: ?[]f32 = if (result.k) |*value| value.data() else null;
    const v_grad: ?[]f32 = if (result.v) |*value| value.data() else null;
    const source_offset = kv_seq - q_seq;

    const attention_work = parallel.saturatedMul3(q_seq, kv_seq, heads * d);
    const stats_active: ?[]const f32 = if (stats != null and attnBwdStatsEnabled()) stats else null;
    if (stats_active) |values| {
        if (values.len != heads * q_seq * 2) return tensor.TensorError.InvalidDataLength;
    }
    if ((need_q or need_k or need_v) and
        q_seq >= 8 and kv_seq >= 8 and d >= 4 and d <= attention_tile_max_d and
        attention_work >= grouped_attention_backward_gemm_work_threshold)
    {
        // Direct dK/dV writes need exactly one query head per kv head;
        // any sharing switches to per-head planes + the fixed-order reduce.
        var shared_kv_head = heads != kv_heads;
        if (!shared_kv_head) {
            const seen = try self.allocator().alloc(bool, kv_heads);
            defer self.allocator().free(seen);
            @memset(seen, false);
            for (kv_head_for_head) |kv_head_i| {
                if (seen[kv_head_i]) shared_kv_head = true;
                seen[kv_head_i] = true;
            }
        }
        const partial_mode = shared_kv_head and (need_k or need_v);

        var dk_plane_storage: ?*storage.Buffer = null;
        defer if (dk_plane_storage) |buffer| buffer.release();
        var dv_plane_storage: ?*storage.Buffer = null;
        defer if (dv_plane_storage) |buffer| buffer.release();
        var dk_target: ?[]f32 = k_grad;
        var dv_target: ?[]f32 = v_grad;
        if (partial_mode) {
            const plane_len = heads * kv_seq * d;
            if (need_k) {
                const buffer = try self.rt.buffers.acquire(plane_len);
                dk_plane_storage = buffer;
                @memset(buffer.data[0..plane_len], 0);
                dk_target = buffer.data[0..plane_len];
            }
            if (need_v) {
                const buffer = try self.rt.buffers.acquire(plane_len);
                dv_plane_storage = buffer;
                @memset(buffer.data[0..plane_len], 0);
                dv_target = buffer.data[0..plane_len];
            }
        }

        const base_task: GroupedCausalAttentionBackwardTiledTask = .{
            .q_data = q_data,
            .k_data = k_data,
            .v_data = v_data,
            .gy_data = gy_data,
            .stats = stats_active,
            .q_grad = q_grad,
            .dk_target = if (need_k) dk_target else null,
            .dv_target = if (need_v) dv_target else null,
            .partial_mode = partial_mode,
            .kv_head_for_head = kv_head_for_head,
            .q_seq = q_seq,
            .kv_seq = kv_seq,
            .source_offset = source_offset,
            .heads = heads,
            .d = d,
            .kv_heads = kv_heads,
            .scale_value = scale_value,
            .window = window,
            .causal = causal,
            .scratch = &.{},
            .head_start = 0,
            .head_end = heads,
        };
        const blas_route = attentionBackwardBlasEnabled();
        const route_tile_rows = if (blas_route) attention_bwd_blas_tile_rows else attention_bwd_tile_rows;
        const panel_len = 2 * route_tile_rows * kv_seq;
        var dispatched = false;
        if (attention_work >= parallel.attention_work_threshold) {
            if (self.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), heads);
                if (task_count > 1) {
                    const task_scratch = try self.allocator().alloc(f32, task_count * panel_len);
                    defer self.allocator().free(task_scratch);
                    var task_storage: [parallel.vector_max_threads]GroupedCausalAttentionBackwardTiledTask = undefined;
                    for (0..task_count) |task_i| {
                        task_storage[task_i] = base_task;
                        task_storage[task_i].head_start = task_i * heads / task_count;
                        task_storage[task_i].head_end = (task_i + 1) * heads / task_count;
                        task_storage[task_i].scratch = task_scratch[task_i * panel_len ..][0..panel_len];
                    }
                    if (blas_route) {
                        pool.parallelChunks(GroupedCausalAttentionBackwardTiledTask, task_storage[0..task_count], runGroupedCausalAttentionBackwardBlasTiledTask);
                    } else {
                        pool.parallelChunks(GroupedCausalAttentionBackwardTiledTask, task_storage[0..task_count], runGroupedCausalAttentionBackwardTiledTask);
                    }
                    dispatched = true;
                }
            }
        }
        if (!dispatched) {
            const task_scratch = try self.allocator().alloc(f32, panel_len);
            defer self.allocator().free(task_scratch);
            var serial_task = base_task;
            serial_task.scratch = task_scratch;
            if (blas_route) {
                groupedCausalAttentionBackwardBlasTiles(serial_task);
            } else {
                groupedCausalAttentionBackwardTiles(serial_task);
            }
        }

        if (partial_mode) {
            const reduce_base: AttentionBackwardReduceTask = .{
                .partials = &.{},
                .grad = &.{},
                .kv_head_for_head = kv_head_for_head,
                .kv_seq = kv_seq,
                .d = d,
                .kv_heads = kv_heads,
                .source_start = 0,
                .source_end = kv_seq,
            };
            const reduce_targets = [2]struct { partials: ?[]const f32, grad: ?[]f32 }{
                .{ .partials = if (need_k) dk_target else null, .grad = k_grad },
                .{ .partials = if (need_v) dv_target else null, .grad = v_grad },
            };
            for (reduce_targets) |pair| {
                var pair_base = reduce_base;
                pair_base.partials = pair.partials orelse continue;
                pair_base.grad = pair.grad orelse continue;
                const reduced = parallel.saturatedMul3(kv_seq, heads, d) >= parallel.vector_elementwise_len_threshold and
                    self.dispatchRange(AttentionBackwardReduceTask, "source_start", "source_end", pair_base, kv_seq, runAttentionBackwardReduceTask);
                if (!reduced) attentionBackwardReduceRows(pair_base);
            }
        }
        return result;
    }

    if (attention_work >= parallel.attention_work_threshold and kv_heads > 1) {
        if (self.workPool()) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), kv_heads);
            var task_storage: [parallel.vector_max_threads]GroupedCausalAttentionBackwardTask = undefined;
            const task_scratch = try self.allocator().alloc(f32, task_count * kv_seq * 2);
            defer self.allocator().free(task_scratch);
            const base: GroupedCausalAttentionBackwardTask = .{
                .q_data = q_data,
                .k_data = k_data,
                .v_data = v_data,
                .gy_data = gy_data,
                .q_grad = q_grad,
                .k_grad = k_grad,
                .v_grad = v_grad,
                .kv_head_for_head = kv_head_for_head,
                .q_seq = q_seq,
                .kv_seq = kv_seq,
                .source_offset = source_offset,
                .heads = heads,
                .d = d,
                .kv_heads = kv_heads,
                .scale_value = scale_value,
                .window = window,
                .causal = causal,
                .kv_head_start = 0,
                .kv_head_end = kv_heads,
                .scores = task_scratch[0..0],
                .dprob = task_scratch[0..0],
            };
            for (0..task_count) |task_i| {
                task_storage[task_i] = base;
                task_storage[task_i].kv_head_start = task_i * kv_heads / task_count;
                task_storage[task_i].kv_head_end = (task_i + 1) * kv_heads / task_count;
                const scratch = task_scratch[task_i * kv_seq * 2 ..][0 .. kv_seq * 2];
                task_storage[task_i].scores = scratch[0..kv_seq];
                task_storage[task_i].dprob = scratch[kv_seq..][0..kv_seq];
            }
            pool.parallelChunks(GroupedCausalAttentionBackwardTask, task_storage[0..task_count], runGroupedCausalAttentionBackwardTask);
            return result;
        }
    }

    var stack_scores: [4096]f32 = undefined;
    var stack_dprob: [4096]f32 = undefined;
    var heap_scratch: ?[]f32 = null;
    defer if (heap_scratch) |values| self.allocator().free(values);
    const scores = if (kv_seq <= stack_scores.len) stack_scores[0..kv_seq] else blk: {
        const values = try self.allocator().alloc(f32, kv_seq * 2);
        heap_scratch = values;
        break :blk values[0..kv_seq];
    };
    const dprob = if (kv_seq <= stack_dprob.len) stack_dprob[0..kv_seq] else heap_scratch.?[kv_seq..][0..kv_seq];
    groupedCausalAttentionBackwardKvHeads(.{
        .q_data = q_data,
        .k_data = k_data,
        .v_data = v_data,
        .gy_data = gy_data,
        .q_grad = q_grad,
        .k_grad = k_grad,
        .v_grad = v_grad,
        .kv_head_for_head = kv_head_for_head,
        .q_seq = q_seq,
        .kv_seq = kv_seq,
        .source_offset = source_offset,
        .heads = heads,
        .d = d,
        .kv_heads = kv_heads,
        .scale_value = scale_value,
        .window = window,
        .causal = causal,
        .kv_head_start = 0,
        .kv_head_end = kv_heads,
        .scores = scores,
        .dprob = dprob,
    });

    return result;
}

/// The `.q8` KvView arm of `groupedAttention`: shape validation for the
/// raw-block cache, then the shared kernel dispatch.
fn groupedCausalAttentionQ8KvImpl(
    self: *ExecContext,
    q: *const Tensor,
    k_blocks: []const BlockQ8_0,
    v_blocks: []const BlockQ8_0,
    kv_seq: usize,
    kv_heads: usize,
    kv_head_for_head: []const usize,
    scale_value: f32,
    window: usize,
) !Tensor {
    const q_view = try q.rankView(3);
    const q_seq = q_view.shape[0];
    const heads = q_view.shape[1];
    const d = q_view.shape[2];
    if (kv_head_for_head.len != heads) return tensor.TensorError.InvalidShape;
    if (q_seq == 0 or q_seq > kv_seq) return tensor.TensorError.InvalidShape;
    if (d == 0 or d % q8_0_block_size != 0 or d > attention_q8_max_d) return tensor.TensorError.InvalidShape;
    const row_blocks = kv_heads * (d / q8_0_block_size);
    if (k_blocks.len != kv_seq * row_blocks) return tensor.TensorError.InvalidShape;
    if (v_blocks.len != k_blocks.len) return tensor.TensorError.InvalidShape;
    for (kv_head_for_head) |kv_head_i| {
        if (kv_head_i >= kv_heads) return tensor.TensorError.IndexOutOfBounds;
    }

    var qq = try self.prepareContiguous(.f32, q);
    defer qq.deinit();

    var out = try self.empty(.f32, .{ q_seq, heads * d });
    errdefer out.deinit();
    try groupedCausalAttentionDispatch(self, BlockQ8_0, qq.tensor().dataConst(), k_blocks, v_blocks, out.data(), kv_head_for_head, q_seq, kv_seq, heads, d, kv_heads, scale_value, window, true, null, null);
    return out;
}

/// The multi-stream KvView arms of `groupedAttention`: ragged-shape
/// validation, then length-weighted (stream, head-unit) dispatch over the
/// per-query decode kernels.
fn groupedCausalAttentionMultiImpl(
    self: *ExecContext,
    comptime KvElem: type,
    q: *const Tensor,
    ks: []const []const KvElem,
    vs: []const []const KvElem,
    lens: []const usize,
    kv_heads: usize,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !Tensor {
    const q_view = try q.rankView(3);
    const n = q_view.shape[0];
    const heads = q_view.shape[1];
    const d = q_view.shape[2];
    if (n == 0 or ks.len != n or vs.len != n or lens.len != n) return tensor.TensorError.InvalidShape;
    if (kv_head_for_head.len != heads) return tensor.TensorError.InvalidShape;
    if (comptime KvElem == BlockQ8_0) {
        if (d == 0 or d % q8_0_block_size != 0 or d > attention_q8_max_d) return tensor.TensorError.InvalidShape;
    }
    for (kv_head_for_head) |kv_head_i| {
        if (kv_head_i >= kv_heads) return tensor.TensorError.IndexOutOfBounds;
    }
    const row = kvRowElems(KvElem, kv_heads, d);
    var max_len: usize = 0;
    var lens_sum: usize = 0;
    for (ks, vs, lens) |k_s, v_s, len_s| {
        if (len_s == 0) return tensor.TensorError.InvalidShape;
        if (k_s.len < len_s * row or v_s.len < len_s * row) return tensor.TensorError.InvalidShape;
        max_len = @max(max_len, len_s);
        lens_sum +|= len_s;
    }

    var qq = try self.prepareContiguous(.f32, q);
    defer qq.deinit();

    var out = try self.empty(.f32, .{ n, heads * d });
    errdefer out.deinit();

    const can_pair = hasAdjacentKvHeadPairs(kv_head_for_head, heads, kv_heads);
    const n_units = if (can_pair) kv_heads else heads;
    const total_work = n * n_units;
    const scores_per_task = max_len * 2;

    const base = GroupedCausalAttentionMultiTask(KvElem){
        .q_data = qq.tensor().dataConst(),
        .out_data = out.data(),
        .ks = ks,
        .vs = vs,
        .lens = lens,
        .kv_head_for_head = kv_head_for_head,
        .heads = heads,
        .d = d,
        .kv_heads = kv_heads,
        .scale_value = scale_value,
        .n_units = n_units,
        .work_start = 0,
        .work_end = total_work,
        .scores = &.{},
    };

    const attention_work = parallel.saturatedMul3(lens_sum, heads, d);
    if (attention_work >= parallel.attention_work_threshold and total_work > 1) {
        if (self.workPool()) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), total_work);
            var task_storage: [parallel.vector_max_threads]GroupedCausalAttentionMultiTask(KvElem) = undefined;
            const scores_storage = try self.rt.buffers.acquire(task_count * scores_per_task);
            defer scores_storage.release();
            const task_scores = scores_storage.data[0 .. task_count * scores_per_task];

            // Length-weighted ranges (the tiled kernel's partition pattern):
            // every head unit of stream s costs ~lens[s] key rows, so ragged
            // streams spread evenly instead of by item count.
            const grand_total: u64 = @as(u64, lens_sum) * n_units;
            var task_i: usize = 0;
            var cumulative: u64 = 0;
            var range_start: usize = 0;
            for (0..total_work) |work_i| {
                cumulative += lens[work_i / n_units];
                while (task_i < task_count and cumulative * task_count >= grand_total * (task_i + 1)) {
                    task_storage[task_i] = base;
                    task_storage[task_i].work_start = range_start;
                    task_storage[task_i].work_end = if (task_i + 1 == task_count) total_work else work_i + 1;
                    task_storage[task_i].scores = task_scores[task_i * scores_per_task ..][0..scores_per_task];
                    range_start = task_storage[task_i].work_end;
                    task_i += 1;
                }
            }
            std.debug.assert(task_i == task_count);
            if (can_pair) {
                pool.parallelChunks(GroupedCausalAttentionMultiTask(KvElem), task_storage[0..task_count], runGroupedCausalAttentionMultiTask(KvElem, true));
            } else {
                pool.parallelChunks(GroupedCausalAttentionMultiTask(KvElem), task_storage[0..task_count], runGroupedCausalAttentionMultiTask(KvElem, false));
            }
            return out;
        }
    }

    var stack_scores: [8192]f32 = undefined;
    var scores_storage: ?*storage.Buffer = null;
    defer if (scores_storage) |buffer| buffer.release();
    const scores = if (scores_per_task <= stack_scores.len) stack_scores[0..scores_per_task] else blk: {
        const buffer = try self.rt.buffers.acquire(scores_per_task);
        scores_storage = buffer;
        break :blk buffer.data[0..scores_per_task];
    };
    var task = base;
    task.scores = scores;
    if (can_pair) {
        groupedCausalAttentionMultiUnits(KvElem, true, task);
    } else {
        groupedCausalAttentionMultiUnits(KvElem, false, task);
    }
    return out;
}

/// Runs the query-tiled attention kernel over flattened (head-unit, tile)
/// work. Task boundaries are key-count-weighted (see
/// `attentionTileKeyCount`) so causally heavier late tiles spread evenly.
/// Allocation-free: tile accumulators live on the kernel's stack and no
/// score scratch exists, so nothing is acquired from the buffer pool.
pub fn groupedCausalAttentionTiledRun(
    self: *ExecContext,
    comptime KvElem: type,
    comptime head_group: usize,
    base_in: GroupedCausalAttentionTiledTask(KvElem),
) void {
    const q_tile = attention_tile_rows / head_group;
    var base = base_in;
    // `window` arrives unvalidated from GGUF metadata (e.g. gemma4's
    // sliding_window), and the tiled kernel computes the SWA bound in i32
    // — `@intCast(window)` is UB for window >= 2^31 in ReleaseFast. Clamp
    // once here; semantics-preserving: the per-query lower bound
    // `(p+1) -| window` saturates at 0 for any window > p, and every
    // query position p <= kv_seq - 1, so any window >= kv_seq is full
    // causal either way (window == 0, the no-window sentinel, is
    // unaffected by the @min).
    base.window = @min(base.window, base.kv_seq);
    base.n_tiles = (base.q_seq + q_tile - 1) / q_tile;

    // GPU prefill-attention seam: stateless
    // blocking offload of the whole fused op — no residency involved, Q/K/V
    // stream per call, and a false return falls through to the CPU tiled
    // kernel below. f16-KV common case only; biased/oversized variants stay CPU.
    if (comptime KvElem == f16) {
        if (offload.attnPrefillF16(base.q_data, base.k_data, base.v_data, base.out_data, base.kv_head_for_head, head_group, base.q_seq, base.kv_seq, base.heads, base.kv_heads, base.d, base.source_offset, base.scale_value, base.window, base.causal, base.bias != null)) return;
    }

    const head_units = if (head_group == 2) base.kv_heads else base.heads;
    const total_work = head_units * base.n_tiles;
    const run_fn = runGroupedCausalAttentionTiledTask(KvElem, head_group);

    // Same pool gate as the per-query attention paths in
    // groupedCausalAttentionImpl: below it the whole job runs as a single
    // task on the calling thread — no lazy pool init, no dispatch
    // barrier. The output is identical either way: work items are
    // independent and each (head, query) output row is written by exactly
    // the one task whose [work_start, work_end) range holds its work
    // item, so the partitioning cannot change any result.
    const attention_work = parallel.saturatedMul3(base.q_seq, base.kv_seq, base.heads * base.d);
    const gate_ok = attention_work >= parallel.attention_work_threshold;
    const pool = (if (gate_ok) self.workPool() else null) orelse {
        var task = base;
        task.work_start = 0;
        task.work_end = total_work;
        run_fn(&task);
        return;
    };

    const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), total_work);
    var task_storage: [parallel.vector_max_threads]GroupedCausalAttentionTiledTask(KvElem) = undefined;

    var tile_weight_sum: u64 = 0;
    for (0..base.n_tiles) |tile_i| {
        tile_weight_sum += attentionTileKeyCount(q_tile, base.q_seq, base.kv_seq, base.source_offset, base.window, base.causal, tile_i);
    }
    const grand_total = tile_weight_sum * head_units;

    var task_i: usize = 0;
    var cumulative: u64 = 0;
    var range_start: usize = 0;
    for (0..total_work) |work_i| {
        cumulative += attentionTileKeyCount(q_tile, base.q_seq, base.kv_seq, base.source_offset, base.window, base.causal, work_i % base.n_tiles);
        while (task_i < task_count and cumulative * task_count >= grand_total * (task_i + 1)) {
            task_storage[task_i] = base;
            task_storage[task_i].work_start = range_start;
            task_storage[task_i].work_end = if (task_i + 1 == task_count) total_work else work_i + 1;
            range_start = task_storage[task_i].work_end;
            task_i += 1;
        }
    }
    std.debug.assert(task_i == task_count);
    pool.parallelChunks(GroupedCausalAttentionTiledTask(KvElem), task_storage[0..task_count], run_fn);
}

fn groupedCausalAttentionImpl(
    self: *ExecContext,
    comptime KvElem: type,
    q: *const Tensor,
    k: *const tensor.TensorOf(kvDtypeOf(KvElem)),
    v: *const tensor.TensorOf(kvDtypeOf(KvElem)),
    kv_head_for_head: []const usize,
    scale_value: f32,
    window: usize,
    causal: bool,
    bias: ?*const Tensor,
    stats: ?[]f32,
) !Tensor {
    std.debug.assert(causal or window == 0);
    std.debug.assert(bias == null or (!causal and window == 0));
    const kv_dtype = comptime kvDtypeOf(KvElem);
    const q_view = try q.rankView(3);
    const k_view = try k.rankView(3);
    const v_view = try v.rankView(3);

    const q_seq = q_view.shape[0];
    const kv_seq = k_view.shape[0];
    const heads = q_view.shape[1];
    const d = q_view.shape[2];
    const kv_heads = k_view.shape[1];
    if (kv_head_for_head.len != heads) return tensor.TensorError.InvalidShape;
    if (q_seq == 0 or q_seq > kv_seq) return tensor.TensorError.InvalidShape;
    if (v_view.shape[0] != kv_seq) return tensor.TensorError.InvalidShape;
    if (k_view.shape[2] != d or v_view.shape[2] != d) return tensor.TensorError.InvalidShape;
    if (v_view.shape[1] != kv_heads) return tensor.TensorError.InvalidShape;
    for (kv_head_for_head) |kv_head_i| {
        if (kv_head_i >= kv_heads) return tensor.TensorError.IndexOutOfBounds;
    }

    var qq = try self.prepareContiguous(.f32, q);
    defer qq.deinit();
    var kk = try self.prepareContiguous(kv_dtype, k);
    defer kk.deinit();
    var vv = try self.prepareContiguous(kv_dtype, v);
    defer vv.deinit();

    // The optional additive bias is validated to [q_seq, kv_seq] here and
    // handed to the kernels as a row-contiguous slice.
    var bias_prepared: ?ExecContext.PreparedTensor = null;
    defer if (bias_prepared) |*prepared| prepared.deinit();
    const bias_data: ?[]const f32 = if (bias) |bias_tensor| blk: {
        const bias_view = try bias_tensor.rankView(2);
        if (bias_view.shape[0] != q_seq or bias_view.shape[1] != kv_seq) return tensor.TensorError.InvalidShape;
        bias_prepared = try self.prepareContiguous(.f32, bias_tensor);
        break :blk bias_prepared.?.tensor().dataConst();
    } else null;

    const q_data = qq.tensor().dataConst();
    const k_data = kk.tensor().dataConst();
    const v_data = vv.tensor().dataConst();

    if (stats) |values| {
        if (values.len != heads * q_seq * 2) return tensor.TensorError.InvalidDataLength;
    }

    var out = try self.empty(.f32, .{ q_seq, heads * d });
    errdefer out.deinit();

    // GPU tier: providers with the attention-forward arm take
    // prefill-length rows over the uniform GQA mapping — f32 K/V (training
    // graphs) and the f16 KV cache (inference prefill) instantiate the same
    // kernel; bias rows, exotic head maps, quantized caches, and any
    // provider decline stay on the CPU tiers. Same contract,
    // summation-order tolerance class.
    if (offload.attentionFwd(KvElem, q_data, k_data, v_data, out.data(), stats, kv_head_for_head, q_seq, kv_seq, heads, kv_heads, d, window, causal, scale_value, bias_data != null)) return out;

    try groupedCausalAttentionDispatch(self, KvElem, q_data, k_data, v_data, out.data(), kv_head_for_head, q_seq, kv_seq, heads, d, kv_heads, scale_value, window, causal, bias_data, stats);
    return out;
}

/// Kernel/parallelism dispatch shared by every grouped-causal-attention
/// cache element type (f32, f16, q8_0 blocks): the query-tiled kernel for
/// long prefill, else the per-query (pair or general) kernels, threaded
/// over the work pool when the job is big enough. `k_data`/`v_data` hold
/// `[kv_seq, kv_heads, d]` rows — element-typed for f32/f16, `d/32`
/// BlockQ8_0 blocks per (position, kv_head) row segment for q8_0.
/// `causal == false` = bidirectional (every query attends every key).
/// `bias` is the optional row-contiguous [q_seq, kv_seq] additive score
/// bias (see GroupedCausalAttentionTask), threaded to every kernel tier.
fn groupedCausalAttentionDispatch(
    self: *ExecContext,
    comptime KvElem: type,
    q_data: []const f32,
    k_data: []const KvElem,
    v_data: []const KvElem,
    out_data: []f32,
    kv_head_for_head: []const usize,
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    d: usize,
    kv_heads: usize,
    scale_value: f32,
    window: usize,
    causal: bool,
    bias: ?[]const f32,
    stats: ?[]f32,
) !void {
    const source_offset = kv_seq - q_seq;
    const attention_work = parallel.saturatedMul3(q_seq, kv_seq, heads * d);
    const can_pair_heads = hasAdjacentKvHeadPairs(kv_head_for_head, heads, kv_heads);

    // Long prefill: query-tiled online-softmax kernel — each K/V row is
    // loaded once per query tile instead of once per query. Its results
    // differ from the per-query kernels in summation order only (~1e-6
    // relative); decode and short prefill below stay bit-identical.
    if (q_seq >= attention_tiled_min_q_seq and d <= attention_tile_max_d) {
        const base = GroupedCausalAttentionTiledTask(KvElem){
            .q_data = q_data,
            .k_data = k_data,
            .v_data = v_data,
            .out_data = out_data,
            .kv_head_for_head = kv_head_for_head,
            .q_seq = q_seq,
            .kv_seq = kv_seq,
            .source_offset = source_offset,
            .heads = heads,
            .d = d,
            .kv_heads = kv_heads,
            .scale_value = scale_value,
            .window = window,
            .causal = causal,
            .bias = bias,
            .n_tiles = 0, // set by groupedCausalAttentionTiledRun
            .work_start = 0,
            .work_end = 0,
            .stats = stats,
        };
        if (can_pair_heads) {
            groupedCausalAttentionTiledRun(self, KvElem, 2, base);
        } else {
            groupedCausalAttentionTiledRun(self, KvElem, 1, base);
        }
        return;
    }
    if (attention_work >= parallel.attention_work_threshold and heads > 1) {
        if (self.workPool()) |pool| {
            if (can_pair_heads) {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), kv_heads);
                var task_storage: [parallel.vector_max_threads]GroupedCausalAttentionPairTask(KvElem) = undefined;
                // Per-task softmax scratch from the buffer pool: at long
                // contexts this crosses the allocator's mmap threshold, so a
                // plain alloc would pay a syscall pair + page faults per
                // layer per token.
                const scores_storage = try self.rt.buffers.acquire(task_count * kv_seq * 2);
                defer scores_storage.release();
                const task_scores = scores_storage.data[0 .. task_count * kv_seq * 2];

                const base: GroupedCausalAttentionPairTask(KvElem) = .{
                    .q_data = q_data,
                    .k_data = k_data,
                    .v_data = v_data,
                    .out_data = out_data,
                    .q_seq = q_seq,
                    .kv_seq = kv_seq,
                    .source_offset = source_offset,
                    .heads = heads,
                    .d = d,
                    .kv_heads = kv_heads,
                    .scale_value = scale_value,
                    .window = window,
                    .causal = causal,
                    .bias = bias,
                    .kv_head_start = 0,
                    .kv_head_end = kv_heads,
                    .scores = task_scores[0..0],
                    .stats = stats,
                };
                for (0..task_count) |task_i| {
                    task_storage[task_i] = base;
                    task_storage[task_i].kv_head_start = task_i * kv_heads / task_count;
                    task_storage[task_i].kv_head_end = (task_i + 1) * kv_heads / task_count;
                    task_storage[task_i].scores = task_scores[task_i * kv_seq * 2 ..][0 .. kv_seq * 2];
                }
                pool.parallelChunks(GroupedCausalAttentionPairTask(KvElem), task_storage[0..task_count], runGroupedCausalAttentionPairTask(KvElem));
                return;
            }

            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), heads);
            var task_storage: [parallel.vector_max_threads]GroupedCausalAttentionTask(KvElem) = undefined;
            const scores_storage = try self.rt.buffers.acquire(task_count * kv_seq);
            defer scores_storage.release();
            const task_scores = scores_storage.data[0 .. task_count * kv_seq];

            const base: GroupedCausalAttentionTask(KvElem) = .{
                .q_data = q_data,
                .k_data = k_data,
                .v_data = v_data,
                .out_data = out_data,
                .kv_head_for_head = kv_head_for_head,
                .q_seq = q_seq,
                .kv_seq = kv_seq,
                .source_offset = source_offset,
                .heads = heads,
                .d = d,
                .kv_heads = kv_heads,
                .scale_value = scale_value,
                .window = window,
                .causal = causal,
                .bias = bias,
                .head_start = 0,
                .head_end = heads,
                .scores = task_scores[0..0],
                .stats = stats,
            };
            for (0..task_count) |task_i| {
                task_storage[task_i] = base;
                task_storage[task_i].head_start = task_i * heads / task_count;
                task_storage[task_i].head_end = (task_i + 1) * heads / task_count;
                task_storage[task_i].scores = task_scores[task_i * kv_seq ..][0..kv_seq];
            }
            pool.parallelChunks(GroupedCausalAttentionTask(KvElem), task_storage[0..task_count], runGroupedCausalAttentionTask(KvElem));
            return;
        }
    }

    if (can_pair_heads) {
        var stack_pair_scores: [8192]f32 = undefined;
        var pair_scores_storage: ?*storage.Buffer = null;
        defer if (pair_scores_storage) |buffer| buffer.release();
        const pair_scores = if (kv_seq * 2 <= stack_pair_scores.len) stack_pair_scores[0 .. kv_seq * 2] else blk: {
            const buffer = try self.rt.buffers.acquire(kv_seq * 2);
            pair_scores_storage = buffer;
            break :blk buffer.data[0 .. kv_seq * 2];
        };

        groupedCausalAttentionHeadPairs(KvElem, .{
            .q_data = q_data,
            .k_data = k_data,
            .v_data = v_data,
            .out_data = out_data,
            .q_seq = q_seq,
            .kv_seq = kv_seq,
            .source_offset = source_offset,
            .heads = heads,
            .d = d,
            .kv_heads = kv_heads,
            .scale_value = scale_value,
            .window = window,
            .causal = causal,
            .bias = bias,
            .kv_head_start = 0,
            .kv_head_end = kv_heads,
            .scores = pair_scores,
            .stats = stats,
        });
        return;
    }

    var stack_scores: [4096]f32 = undefined;
    var scores_storage: ?*storage.Buffer = null;
    defer if (scores_storage) |buffer| buffer.release();
    const scores = if (kv_seq <= stack_scores.len) stack_scores[0..kv_seq] else blk: {
        const buffer = try self.rt.buffers.acquire(kv_seq);
        scores_storage = buffer;
        break :blk buffer.data[0..kv_seq];
    };

    groupedCausalAttentionHeads(KvElem, .{
        .q_data = q_data,
        .k_data = k_data,
        .v_data = v_data,
        .out_data = out_data,
        .kv_head_for_head = kv_head_for_head,
        .q_seq = q_seq,
        .kv_seq = kv_seq,
        .source_offset = source_offset,
        .heads = heads,
        .d = d,
        .kv_heads = kv_heads,
        .scale_value = scale_value,
        .window = window,
        .causal = causal,
        .bias = bias,
        .head_start = 0,
        .head_end = heads,
        .scores = scores,
        .stats = stats,
    });
}

test {
    // Group-B tiled-attention parity tests (they drive this module's pub
    // tiled kernel + Task; kept out of exec.zig's inline Group-A tests).
    _ = @import("attention_tests.zig");
}
