const std = @import("std");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");
const parallel = @import("../parallel.zig");
const storage = @import("../storage.zig");
const thread = @import("../thread.zig");
const Runtime = @import("runtime.zig").Runtime;

const DType = tensor.DType;
const Tensor = tensor.Tensor;
const vexpf = @import("../backend/vector/primitives.zig").vexpf;

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

fn zerosRank(self: *Runtime, comptime rank: usize, shape: [rank]usize) !tensor.Tensor {
    var out = try self.emptyRank(rank, shape);
    @memset(out.data(), 0);
    return out;
}

fn workPool(self: *Runtime) ?*thread.Pool {
    return self.tryWorkPool() catch null;
}

// K/V come in either as f32 (prefill, no cache), f16 (the decode KV cache,
// half the bandwidth), or q8_0 blocks (the quantized KV cache, ~quarter the
// bandwidth of f32). The attention kernels are generic over that element type:
// f16 lanes are widened to f32 in-register (the f32 instantiation compiles to
// the same loads as before — the comptime branches below collapse to a plain
// load); q8_0 rows are dequantized into per-task L1 scratch as they stream
// (see `kvRowSelect`), so the inner dot/accumulate loops always run on f32/f16
// lanes.
inline fn widenKvVec(comptime KvElem: type, comptime width: usize, data: []const KvElem, offset: usize) @Vector(width, f32) {
    const chunk: @Vector(width, KvElem) = data[offset..][0..width].*;
    return if (KvElem == f32) chunk else @floatCast(chunk);
}

inline fn widenKvScalar(comptime KvElem: type, value: KvElem) f32 {
    return if (KvElem == f32) value else @floatCast(value);
}

pub fn kvDtypeOf(comptime KvElem: type) DType {
    return if (KvElem == f32) .f32 else .f16;
}

pub const BlockQ8_0 = dtype_mod.BlockQ8_0;
pub const q8_0_block_size = dtype_mod.q8_0_block_size;

/// Lane type the attention inner loops consume: q8_0 rows are dequantized to
/// f32 scratch first, f32/f16 are read (and widened) directly.
fn KvLane(comptime KvElem: type) type {
    return if (KvElem == BlockQ8_0) f32 else KvElem;
}

/// Head dims the per-query q8_0 attention kernels support: the per-task
/// dequant scratch row is this many f32s on the stack (Gemma's widest global
/// head is 512). Validated at the `groupedCausalAttentionQ8Kv*` entries.
pub const attention_q8_max_d: usize = 512;

/// Resolve the K or V row that starts at element offset `elem_base` for the
/// dot/accumulate loops. f32/f16 pass the raw cache slice straight through
/// (zero-cost after inlining — codegen is the pre-q8 load sequence). q8_0
/// dequantizes the row's `d/32` blocks into `scratch` (per-task, L1-resident)
/// and indexes it at 0: traffic from the cache stays the quantized 34
/// bytes/block, only scratch traffic is added. Row bases are always multiples
/// of `d` (and `d % 32 == 0` is validated at the q8 entries), so the block
/// index `elem_base / 32` is exact.
inline fn kvRowSelect(
    comptime KvElem: type,
    data: []const KvElem,
    elem_base: usize,
    d: usize,
    scratch: []f32,
) struct { []const KvLane(KvElem), usize } {
    if (comptime KvElem == BlockQ8_0) {
        backend_mod.quantized_matmul.dequantizeRowQ8_0Into(
            scratch[0..d],
            data[elem_base / q8_0_block_size ..][0 .. d / q8_0_block_size],
        ) catch unreachable;
        return .{ scratch[0..d], 0 };
    }
    return .{ data, elem_base };
}

pub fn GroupedCausalAttentionTask(comptime KvElem: type) type {
    return struct {
        q_data: []const f32,
        k_data: []const KvElem,
        v_data: []const KvElem,
        out_data: []f32,
        kv_head_for_head: []const usize,
        q_seq: usize,
        kv_seq: usize,
        source_offset: usize,
        heads: usize,
        d: usize,
        kv_heads: usize,
        scale_value: f32,
        // Sliding-window attention: 0 = full causal; else a query at absolute
        // position `p` attends keys in `[max(0, p-window+1), p]` (local SWA).
        window: usize,
        // false = bidirectional: every query attends ALL kv_seq keys (block
        // diffusion canvas attention). Requires window == 0 — a canvas window
        // is realized by narrowing the K/V views instead.
        causal: bool = true,
        // Optional additive f32 bias, row-contiguous [q_seq, kv_seq]: added
        // to the SCALED score before the softmax — score(query_i, source_i)
        // = dot * scale_value + bias[query_i * kv_seq + source_i]. Null = no
        // bias; non-null only via the bidirectional biased entry today.
        bias: ?[]const f32 = null,
        head_start: usize,
        head_end: usize,
        scores: []f32,
        // Optional per-(head, query) softmax statistics out: {max, sum_exp}
        // interleaved f32 pairs at (head_i * q_seq + query_i) * 2. Each
        // kernel records the normalizer IT used (two-pass max/sum here and
        // on the pair kernel; the tiled kernel's online-rescaled running
        // pair), so a backward fed with them reconstructs that kernel's
        // probabilities. Disjoint writes across tasks.
        stats: ?[]f32 = null,
    };
}

pub fn GroupedCausalAttentionPairTask(comptime KvElem: type) type {
    return struct {
        q_data: []const f32,
        k_data: []const KvElem,
        v_data: []const KvElem,
        out_data: []f32,
        q_seq: usize,
        kv_seq: usize,
        source_offset: usize,
        heads: usize,
        d: usize,
        kv_heads: usize,
        scale_value: f32,
        // Sliding-window attention: 0 = full causal (see GroupedCausalAttentionTask).
        window: usize,
        // false = bidirectional (see GroupedCausalAttentionTask).
        causal: bool = true,
        // Optional additive [q_seq, kv_seq] bias (see GroupedCausalAttentionTask).
        bias: ?[]const f32 = null,
        kv_head_start: usize,
        kv_head_end: usize,
        scores: []f32,
        // Optional {max, sum_exp} stats out (see GroupedCausalAttentionTask).
        stats: ?[]f32 = null,
    };
}

/// Query rows in flight per tile in the query-tiled attention forward:
/// queries per tile = `attention_tile_rows / head_group` (head_group = 2 on
/// the adjacent-pair GQA path shares each K/V row across the pair as well).
/// With `attention_key_block` keys per inner step the score microkernel keeps
/// rows x keys = 16 @Vector(4, f32) dot accumulators plus both operand rows
/// in registers — the NEON file's sweet spot; 8 rows would spill.
/// Picked by benchmark on M1 Max (see docs/BENCHMARK.md protocol).
pub const attention_tile_rows: usize = 4;
/// Keys processed per inner step. Blocking keys amortizes the q-row reloads
/// of the score pass and the accumulator load/stores of the value pass (the
/// L1-port bottleneck of a 1-key-at-a-time online-softmax kernel) and means
/// one running-max check per block instead of per key.
pub const attention_key_block: usize = 4;
/// The tiled kernel keeps per-row output accumulators on the stack
/// (`rows * attention_tile_max_d` floats, 4 KiB at 4 rows x 256): L1-resident
/// and allocation-free. Heads wider than this stay on the per-query kernels.
pub const attention_tile_max_d: usize = 256;
/// Prefill length at which the tiled kernel takes over. Decode (q_seq == 1)
/// and short prefill stay on the per-query kernels above, whose results
/// remain bit-identical. Tuned by benchmark: below this the K/V re-stream
/// fits in cache and the per-query kernels' simpler inner loop wins.
pub const attention_tiled_min_q_seq: usize = 48;

pub fn GroupedCausalAttentionTiledTask(comptime KvElem: type) type {
    return struct {
        q_data: []const f32,
        k_data: []const KvElem,
        v_data: []const KvElem,
        out_data: []f32,
        // Only read on the general (head_group == 1) path; the pair path maps
        // head_unit -> kv head implicitly like GroupedCausalAttentionPairTask.
        kv_head_for_head: []const usize,
        q_seq: usize,
        kv_seq: usize,
        source_offset: usize,
        heads: usize,
        d: usize,
        kv_heads: usize,
        scale_value: f32,
        // Sliding-window attention: 0 = full causal (see GroupedCausalAttentionTask).
        window: usize,
        // false = bidirectional (see GroupedCausalAttentionTask).
        causal: bool = true,
        // Optional additive [q_seq, kv_seq] bias (see GroupedCausalAttentionTask).
        bias: ?[]const f32 = null,
        n_tiles: usize,
        // Work items are flattened head-major: `head_unit * n_tiles + tile_i`.
        work_start: usize,
        work_end: usize,
        // Optional {max, sum_exp} stats out (see GroupedCausalAttentionTask);
        // this kernel records its online-rescaled running pair.
        stats: ?[]f32 = null,
    };
}

pub const GroupedCausalAttentionBackwardTask = struct {
    q_data: []const f32,
    k_data: []const f32,
    v_data: []const f32,
    gy_data: []const f32,
    q_grad: ?[]f32,
    k_grad: ?[]f32,
    v_grad: ?[]f32,
    kv_head_for_head: []const usize,
    q_seq: usize,
    kv_seq: usize,
    source_offset: usize,
    heads: usize,
    d: usize,
    kv_heads: usize,
    scale_value: f32,
    window: usize,
    // false = bidirectional (see GroupedCausalAttentionTask).
    causal: bool = true,
    kv_head_start: usize,
    kv_head_end: usize,
    scores: []f32,
    dprob: []f32,
};

pub fn runGroupedCausalAttentionTask(comptime KvElem: type) fn (*const GroupedCausalAttentionTask(KvElem)) void {
    return struct {
        fn run(task: *const GroupedCausalAttentionTask(KvElem)) void {
            groupedCausalAttentionHeads(KvElem, task.*);
        }
    }.run;
}

pub fn runGroupedCausalAttentionPairTask(comptime KvElem: type) fn (*const GroupedCausalAttentionPairTask(KvElem)) void {
    return struct {
        fn run(task: *const GroupedCausalAttentionPairTask(KvElem)) void {
            groupedCausalAttentionHeadPairs(KvElem, task.*);
        }
    }.run;
}

pub fn runGroupedCausalAttentionTiledTask(comptime KvElem: type, comptime head_group: usize) fn (*const GroupedCausalAttentionTiledTask(KvElem)) void {
    return struct {
        fn run(task: *const GroupedCausalAttentionTiledTask(KvElem)) void {
            groupedCausalAttentionQueryTiles(KvElem, head_group, task.*);
        }
    }.run;
}

pub fn runGroupedCausalAttentionBackwardTask(task: *const GroupedCausalAttentionBackwardTask) void {
    groupedCausalAttentionBackwardKvHeads(task.*);
}

pub fn groupedCausalAttentionHeads(comptime KvElem: type, task: GroupedCausalAttentionTask(KvElem)) void {
    const q_head_stride = task.d;
    const q_seq_stride = task.heads * task.d;
    const kv_head_stride = task.d;
    const kv_seq_stride = task.kv_heads * task.d;
    const out_seq_stride = task.heads * task.d;
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    const Lane = KvLane(KvElem);
    // q8_0 dequant scratch (one K or V row); zero-sized (and unused) otherwise.
    var q8_scratch: [if (KvElem == BlockQ8_0) attention_q8_max_d else 0]f32 = undefined;

    for (task.head_start..task.head_end) |head_i| {
        const kv_head_i = task.kv_head_for_head[head_i];
        for (0..task.q_seq) |query_i| {
            const active = if (task.causal) task.source_offset + query_i + 1 else task.kv_seq;
            const lo = if (!task.causal or task.window == 0) 0 else active -| task.window;
            const q_base = query_i * q_seq_stride + head_i * q_head_stride;
            const bias_row: ?[]const f32 = if (task.bias) |bias_data| bias_data[query_i * task.kv_seq ..][0..task.kv_seq] else null;

            var max_score = -std.math.inf(f32);
            for (lo..active) |source_i| {
                const k_row, const k_base = kvRowSelect(KvElem, task.k_data, source_i * kv_seq_stride + kv_head_i * kv_head_stride, task.d, &q8_scratch);
                var dot_vec: Vec = @splat(0);
                var feature_i: usize = 0;
                while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                    const qv: Vec = task.q_data[q_base + feature_i ..][0..vector_width].*;
                    const kv: Vec = widenKvVec(Lane, vector_width, k_row, k_base + feature_i);
                    dot_vec += qv * kv;
                }
                var dot_value: f32 = @reduce(.Add, dot_vec);
                while (feature_i < task.d) : (feature_i += 1) {
                    dot_value += task.q_data[q_base + feature_i] * widenKvScalar(Lane, k_row[k_base + feature_i]);
                }
                var score = dot_value * task.scale_value;
                if (bias_row) |row| score += row[source_i];
                task.scores[source_i] = score;
                max_score = @max(max_score, score);
            }

            var sum_exp: f32 = 0;
            for (lo..active) |source_i| {
                const weight = @exp(task.scores[source_i] - max_score);
                task.scores[source_i] = weight;
                sum_exp += weight;
            }
            const inv_sum = 1 / sum_exp;
            if (task.stats) |stats| {
                const stat_base = (head_i * task.q_seq + query_i) * 2;
                stats[stat_base] = max_score;
                stats[stat_base + 1] = sum_exp;
            }

            const out_base = query_i * out_seq_stride + head_i * task.d;
            {
                const weight: Vec = @splat(task.scores[lo] * inv_sum);
                const v_row, const v_base = kvRowSelect(KvElem, task.v_data, lo * kv_seq_stride + kv_head_i * kv_head_stride, task.d, &q8_scratch);
                var feature_i: usize = 0;
                while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                    const v_vec: Vec = widenKvVec(Lane, vector_width, v_row, v_base + feature_i);
                    task.out_data[out_base + feature_i ..][0..vector_width].* = weight * v_vec;
                }
                while (feature_i < task.d) : (feature_i += 1) {
                    task.out_data[out_base + feature_i] = task.scores[lo] * inv_sum * widenKvScalar(Lane, v_row[v_base + feature_i]);
                }
            }
            for (lo + 1..active) |source_i| {
                const weight: Vec = @splat(task.scores[source_i] * inv_sum);
                const scalar_weight = task.scores[source_i] * inv_sum;
                const v_row, const v_base = kvRowSelect(KvElem, task.v_data, source_i * kv_seq_stride + kv_head_i * kv_head_stride, task.d, &q8_scratch);
                var feature_i: usize = 0;
                while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                    const current: Vec = task.out_data[out_base + feature_i ..][0..vector_width].*;
                    const v_vec: Vec = widenKvVec(Lane, vector_width, v_row, v_base + feature_i);
                    task.out_data[out_base + feature_i ..][0..vector_width].* = current + weight * v_vec;
                }
                while (feature_i < task.d) : (feature_i += 1) {
                    task.out_data[out_base + feature_i] += scalar_weight * widenKvScalar(Lane, v_row[v_base + feature_i]);
                }
            }
        }
    }
}

pub fn groupedCausalAttentionHeadPairs(comptime KvElem: type, task: GroupedCausalAttentionPairTask(KvElem)) void {
    const q_head_stride = task.d;
    const q_seq_stride = task.heads * task.d;
    const kv_head_stride = task.d;
    const kv_seq_stride = task.kv_heads * task.d;
    const out_seq_stride = task.heads * task.d;
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    const Lane = KvLane(KvElem);
    // q8_0 dequant scratch (one K or V row, shared by the head pair);
    // zero-sized (and unused) otherwise.
    var q8_scratch: [if (KvElem == BlockQ8_0) attention_q8_max_d else 0]f32 = undefined;
    const scores0 = task.scores[0..task.kv_seq];
    const scores1 = task.scores[task.kv_seq..][0..task.kv_seq];

    for (task.kv_head_start..task.kv_head_end) |kv_head_i| {
        const head0 = kv_head_i * 2;
        const head1 = head0 + 1;
        for (0..task.q_seq) |query_i| {
            const active = if (task.causal) task.source_offset + query_i + 1 else task.kv_seq;
            const lo = if (!task.causal or task.window == 0) 0 else active -| task.window;
            const q_base0 = query_i * q_seq_stride + head0 * q_head_stride;
            const q_base1 = query_i * q_seq_stride + head1 * q_head_stride;
            const bias_row: ?[]const f32 = if (task.bias) |bias_data| bias_data[query_i * task.kv_seq ..][0..task.kv_seq] else null;

            var max_score0 = -std.math.inf(f32);
            var max_score1 = -std.math.inf(f32);
            for (lo..active) |source_i| {
                const k_row, const k_base = kvRowSelect(KvElem, task.k_data, source_i * kv_seq_stride + kv_head_i * kv_head_stride, task.d, &q8_scratch);
                var dot_vec0: Vec = @splat(0);
                var dot_vec1: Vec = @splat(0);
                var feature_i: usize = 0;
                while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                    const kv: Vec = widenKvVec(Lane, vector_width, k_row, k_base + feature_i);
                    const q0v: Vec = task.q_data[q_base0 + feature_i ..][0..vector_width].*;
                    const q1v: Vec = task.q_data[q_base1 + feature_i ..][0..vector_width].*;
                    dot_vec0 += q0v * kv;
                    dot_vec1 += q1v * kv;
                }
                var dot_value0: f32 = @reduce(.Add, dot_vec0);
                var dot_value1: f32 = @reduce(.Add, dot_vec1);
                while (feature_i < task.d) : (feature_i += 1) {
                    const k_value = widenKvScalar(Lane, k_row[k_base + feature_i]);
                    dot_value0 += task.q_data[q_base0 + feature_i] * k_value;
                    dot_value1 += task.q_data[q_base1 + feature_i] * k_value;
                }
                var score0 = dot_value0 * task.scale_value;
                var score1 = dot_value1 * task.scale_value;
                if (bias_row) |row| {
                    score0 += row[source_i];
                    score1 += row[source_i];
                }
                scores0[source_i] = score0;
                scores1[source_i] = score1;
                max_score0 = @max(max_score0, score0);
                max_score1 = @max(max_score1, score1);
            }

            var sum_exp0: f32 = 0;
            var sum_exp1: f32 = 0;
            for (lo..active) |source_i| {
                const weight0 = @exp(scores0[source_i] - max_score0);
                const weight1 = @exp(scores1[source_i] - max_score1);
                scores0[source_i] = weight0;
                scores1[source_i] = weight1;
                sum_exp0 += weight0;
                sum_exp1 += weight1;
            }
            const inv_sum0 = 1 / sum_exp0;
            const inv_sum1 = 1 / sum_exp1;
            if (task.stats) |stats| {
                const stat_base0 = (head0 * task.q_seq + query_i) * 2;
                stats[stat_base0] = max_score0;
                stats[stat_base0 + 1] = sum_exp0;
                const stat_base1 = (head1 * task.q_seq + query_i) * 2;
                stats[stat_base1] = max_score1;
                stats[stat_base1 + 1] = sum_exp1;
            }

            const out_base0 = query_i * out_seq_stride + head0 * task.d;
            const out_base1 = query_i * out_seq_stride + head1 * task.d;
            {
                const weight0: Vec = @splat(scores0[lo] * inv_sum0);
                const weight1: Vec = @splat(scores1[lo] * inv_sum1);
                const v_row, const v_base = kvRowSelect(KvElem, task.v_data, lo * kv_seq_stride + kv_head_i * kv_head_stride, task.d, &q8_scratch);
                var feature_i: usize = 0;
                while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                    const v_vec: Vec = widenKvVec(Lane, vector_width, v_row, v_base + feature_i);
                    task.out_data[out_base0 + feature_i ..][0..vector_width].* = weight0 * v_vec;
                    task.out_data[out_base1 + feature_i ..][0..vector_width].* = weight1 * v_vec;
                }
                const scalar_weight0 = scores0[lo] * inv_sum0;
                const scalar_weight1 = scores1[lo] * inv_sum1;
                while (feature_i < task.d) : (feature_i += 1) {
                    const v_value = widenKvScalar(Lane, v_row[v_base + feature_i]);
                    task.out_data[out_base0 + feature_i] = scalar_weight0 * v_value;
                    task.out_data[out_base1 + feature_i] = scalar_weight1 * v_value;
                }
            }
            for (lo + 1..active) |source_i| {
                const weight0: Vec = @splat(scores0[source_i] * inv_sum0);
                const weight1: Vec = @splat(scores1[source_i] * inv_sum1);
                const scalar_weight0 = scores0[source_i] * inv_sum0;
                const scalar_weight1 = scores1[source_i] * inv_sum1;
                const v_row, const v_base = kvRowSelect(KvElem, task.v_data, source_i * kv_seq_stride + kv_head_i * kv_head_stride, task.d, &q8_scratch);
                var feature_i: usize = 0;
                while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                    const current0: Vec = task.out_data[out_base0 + feature_i ..][0..vector_width].*;
                    const current1: Vec = task.out_data[out_base1 + feature_i ..][0..vector_width].*;
                    const v_vec: Vec = widenKvVec(Lane, vector_width, v_row, v_base + feature_i);
                    task.out_data[out_base0 + feature_i ..][0..vector_width].* = current0 + weight0 * v_vec;
                    task.out_data[out_base1 + feature_i ..][0..vector_width].* = current1 + weight1 * v_vec;
                }
                while (feature_i < task.d) : (feature_i += 1) {
                    const v_value = widenKvScalar(Lane, v_row[v_base + feature_i]);
                    task.out_data[out_base0 + feature_i] += scalar_weight0 * v_value;
                    task.out_data[out_base1 + feature_i] += scalar_weight1 * v_value;
                }
            }
        }
    }
}

/// Query-tiled online-softmax attention forward (long prefill).
///
/// The per-query kernels above re-stream K and V for every query: K/V traffic
/// is ~q_seq * kv_seq * d per head, which is bandwidth-bound at long context.
/// This kernel processes a tile of consecutive query positions (times the
/// `head_group` query heads sharing the KV head) per pass over the keys,
/// so each K/V row is loaded once per tile instead of once per query, with an
/// online softmax (running max `m`, normalizer `l`, accumulator rescaled when
/// `m` rises) so no q*kv score matrix and no score scratch exist.
///
/// Ragged causal bounds inside a tile (each query attends a different prefix,
/// and SWA adds a per-query lower bound) are handled with a per-key row mask
/// instead of region splitting: the mask costs two vector compares + selects
/// per key, amortized to noise against the rows * d multiply-adds of the key
/// body, and keeps a single branch-free inner loop for every key. Masked rows
/// get score = -inf (so `m` is untouched) and p forced to 0 (so a row whose
/// `m` is still -inf is not poisoned by exp(-inf - -inf) = NaN). A NaN logit
/// still poisons its row: @max drops the NaN from `m` but vexpf propagates it
/// through `p` into `l` and the accumulator, matching the 3-pass kernels.
/// One divergence from the 3-pass kernels: they never read V rows outside a
/// query's [lo, active) range, while this kernel walks the tile's union key
/// range and folds masked keys in with p = 0 — so a NON-FINITE V value at a
/// key position masked for a row (causally future within the tile span, below
/// the row's SWA bound, or a clamped tail duplicate) poisons that row on this
/// path only (0 * inf = NaN through the fmadd). Finite V at masked positions
/// contributes an exact 0 and is unaffected; this is inherent to flash-style
/// union-range accumulation.
///
/// Precondition: kv_seq < 2^31 — relative positions (and with them the SWA
/// window bound) are computed in i32. Positions are validated upstream; the
/// tiled dispatch clamps `window` to kv_seq so it fits the same budget.
///
/// The last partial tile clamps the query index, so duplicate rows recompute
/// the final query (valid loads, identical math) and are not written back.
///
/// Numerics: per query the keys are visited in the same order as the 3-pass
/// kernels, but the summation grouping differs (online rescale, normalization
/// after accumulation, fused multiply-adds), so results agree to ~1e-6
/// relative rather than bitwise. Dispatch keeps decode and short prefill on
/// the bit-identical kernels above.
pub fn groupedCausalAttentionQueryTiles(
    comptime KvElem: type,
    comptime head_group: usize,
    task: GroupedCausalAttentionTiledTask(KvElem),
) void {
    const rows = attention_tile_rows;
    const q_tile = rows / head_group;
    const kb = attention_key_block;
    // Score pass: rows x kb dot accumulators must stay in registers, so the
    // d-chunk is one NEON register wide. Value pass: the accumulator rows are
    // load/store traffic either way, so a doubled chunk halves loop overhead.
    const DotVec = @Vector(4, f32);
    const dot_width = 4;
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    const RVec = @Vector(rows, f32);
    const RVecI = @Vector(rows, i32);
    const ones: RVec = @splat(1);
    const zeros: RVec = @splat(0);
    const neg_inf: RVec = @splat(-std.math.inf(f32));

    const q_seq_stride = task.heads * task.d;
    const kv_seq_stride = task.kv_heads * task.d;
    const out_seq_stride = task.heads * task.d;
    const Lane = KvLane(KvElem);

    // Per-row f32 output accumulators; L1-resident stack scratch.
    var acc: [rows][attention_tile_max_d]f32 = undefined;
    // q8_0 dequant scratch for one key block's kb rows; the score pass fully
    // consumes the K rows before the value pass dequantizes the V rows over
    // them. Zero-sized (and unused) for f32/f16.
    var q8_scratch: [if (KvElem == BlockQ8_0) kb * attention_tile_max_d else 0]f32 = undefined;

    for (task.work_start..task.work_end) |work_i| {
        const head_unit = work_i / task.n_tiles;
        const tile_i = work_i % task.n_tiles;
        const kv_head_i = if (head_group == 2) head_unit else task.kv_head_for_head[head_unit];
        const kv_head_base = kv_head_i * task.d;
        const q0 = tile_i * q_tile;
        const rows_active = @min(q_tile, task.q_seq - q0);

        // Row r covers query q0 + min(r / head_group, rows_active - 1) and
        // query head head_unit * head_group + (r % head_group).
        var q_base: [rows]usize = undefined;
        var qr_arr: [rows]i32 = undefined;
        var bias_base: [rows]usize = undefined;
        inline for (0..rows) |r| {
            const qr = @min(r / head_group, rows_active - 1);
            qr_arr[r] = @intCast(qr);
            const head_i = head_unit * head_group + (r % head_group);
            q_base[r] = (q0 + qr) * q_seq_stride + head_i * task.d;
            bias_base[r] = (q0 + qr) * task.kv_seq;
        }
        const qr_vec: RVecI = qr_arr;

        var m_run = neg_inf;
        var l_run = zeros;
        inline for (0..rows) |r| @memset(acc[r][0..task.d], 0);

        const p_first = task.source_offset + q0;
        // Bidirectional: every query scans the full [0, kv_seq) key range;
        // only the unconditional in-range mask below applies.
        const active_last = if (task.causal) p_first + rows_active else task.kv_seq;
        const lo_first = if (!task.causal or task.window == 0) 0 else (p_first + 1) -| task.window;

        var block_start = lo_first;
        while (block_start < active_last) : (block_start += kb) {
            // Key ki addresses source position block_start + ki; positions at
            // or past active_last reload the last real key (valid memory) and
            // are killed by the causal mask below, so the tail needs no
            // separate code path.
            var k_base: [kb]usize = undefined;
            inline for (0..kb) |ki| {
                const source_i = @min(block_start + ki, active_last - 1);
                k_base[ki] = source_i * kv_seq_stride + kv_head_base;
            }
            var k_rows: [kb][]const Lane = undefined;
            var k_offs: [kb]usize = undefined;
            inline for (0..kb) |ki| {
                const scratch_off = comptime if (KvElem == BlockQ8_0) ki * attention_tile_max_d else 0;
                const row, const off = kvRowSelect(KvElem, task.k_data, k_base[ki], task.d, q8_scratch[scratch_off..]);
                k_rows[ki] = row;
                k_offs[ki] = off;
            }

            // Score microkernel: rows x kb dots off one pass over d, both
            // operand rows loaded once per chunk (outer-product shape).
            var dot: [kb][rows]DotVec = undefined;
            inline for (0..kb) |ki| inline for (0..rows) |r| {
                dot[ki][r] = @splat(0);
            };
            var feature_i: usize = 0;
            while (feature_i + dot_width <= task.d) : (feature_i += dot_width) {
                var kv: [kb]DotVec = undefined;
                inline for (0..kb) |ki| kv[ki] = widenKvVec(Lane, dot_width, k_rows[ki], k_offs[ki] + feature_i);
                inline for (0..rows) |r| {
                    const qv: DotVec = task.q_data[q_base[r] + feature_i ..][0..dot_width].*;
                    inline for (0..kb) |ki| dot[ki][r] = @mulAdd(DotVec, qv, kv[ki], dot[ki][r]);
                }
            }
            var score_arr: [kb][rows]f32 = undefined;
            inline for (0..kb) |ki| inline for (0..rows) |r| {
                score_arr[ki][r] = @reduce(.Add, dot[ki][r]);
            };
            while (feature_i < task.d) : (feature_i += 1) {
                inline for (0..kb) |ki| {
                    const k_value = widenKvScalar(Lane, k_rows[ki][k_offs[ki] + feature_i]);
                    inline for (0..rows) |r| score_arr[ki][r] += task.q_data[q_base[r] + feature_i] * k_value;
                }
            }

            // Row mask per key: query at position p = source_offset + q0 + qr
            // attends source_i iff source_i <= p and (window == 0 or
            // source_i >= p + 1 - window), i.e. qr >= s_rel and
            // qr <= s_rel + window - 1 with s_rel = source_i - p_first. The
            // mask uses the unclamped position, so clamped tail keys drop out.
            // Bidirectional: every in-range key (source < kv_seq) is attended
            // by every row; only the tail-clamped reloads are masked.
            const scale_splat: RVec = @splat(task.scale_value);
            const s_rel0: i32 = @as(i32, @intCast(block_start)) - @as(i32, @intCast(p_first));
            var scores: [kb]RVec = undefined;
            var mask: [kb]@Vector(rows, bool) = undefined;
            var block_max = neg_inf;
            inline for (0..kb) |ki| {
                const s_rel: i32 = s_rel0 + @as(i32, @intCast(ki));
                const causal_ok = qr_vec >= @as(RVecI, @splat(s_rel));
                mask[ki] = if (!task.causal) blk: {
                    // Scalar branch between comptime-known splats instead of
                    // @splat(runtime bool): the self-hosted x86_64 backend
                    // (Debug builds) miscompiles runtime-bool vector splats —
                    // a FALSE input broadcasts stray index bits instead
                    // (verified minimal repro, zig 0.16.0), which unmasked
                    // the bidirectional tail-clamp keys and corrupted every
                    // kv_seq % kb != 0 bidirectional result under test
                    // builds. LLVM folds this branch back to a broadcast.
                    const trues: @Vector(rows, bool) = @splat(true);
                    const falses: @Vector(rows, bool) = @splat(false);
                    break :blk if (block_start + ki < task.kv_seq) trues else falses;
                } else if (task.window == 0) causal_ok else blk: {
                    const window_ok = qr_vec <= @as(RVecI, @splat(s_rel + @as(i32, @intCast(task.window)) - 1));
                    break :blk @select(bool, causal_ok, window_ok, @as(@Vector(rows, bool), @splat(false)));
                };
                var scaled = @as(RVec, score_arr[ki]) * scale_splat;
                if (task.bias) |bias_data| {
                    // Same tail clamp as k_base above: a clamped duplicate
                    // key reloads the last real key's (valid) bias value and
                    // is masked to -inf below, so it never contributes.
                    const source_i = @min(block_start + ki, active_last - 1);
                    var bias_arr: [rows]f32 = undefined;
                    inline for (0..rows) |r| bias_arr[r] = bias_data[bias_base[r] + source_i];
                    scaled += @as(RVec, bias_arr);
                }
                scores[ki] = @select(f32, mask[ki], scaled, neg_inf);
                block_max = @max(block_max, scores[ki]);
            }

            // Online-softmax update, once per key block: a NaN logit slips
            // past @max (which drops NaN) but vexpf propagates it through p
            // into l and the accumulator, poisoning the row like the 3-pass
            // kernels.
            const m_new = @max(m_run, block_max);
            var p: [kb]RVec = undefined;
            var p_sum = zeros;
            inline for (0..kb) |ki| {
                p[ki] = @select(f32, mask[ki], vexpf(rows, scores[ki] - m_new), zeros);
                p_sum += p[ki];
            }
            var p_arr: [kb][rows]f32 = undefined;
            inline for (0..kb) |ki| p_arr[ki] = p[ki];
            // K and V rows share the same layout, so the offsets carry over.
            // The score pass is done with the K rows, so the q8_0 dequant
            // scratch is safely reused for the V rows here.
            var v_rows: [kb][]const Lane = undefined;
            var v_offs: [kb]usize = undefined;
            inline for (0..kb) |ki| {
                const scratch_off = comptime if (KvElem == BlockQ8_0) ki * attention_tile_max_d else 0;
                const row, const off = kvRowSelect(KvElem, task.v_data, k_base[ki], task.d, q8_scratch[scratch_off..]);
                v_rows[ki] = row;
                v_offs[ki] = off;
            }

            // Value pass: kb keys' p*V folded into the accumulator rows per
            // chunk, amortizing the acc load/store over the block. The
            // running max rarely rises after the first blocks, so the common
            // path skips the accumulator rescale entirely.
            const m_same = m_new == m_run;
            if (@reduce(.And, m_same)) {
                l_run += p_sum;
                feature_i = 0;
                while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                    var vv: [kb]Vec = undefined;
                    inline for (0..kb) |ki| vv[ki] = widenKvVec(Lane, vector_width, v_rows[ki], v_offs[ki] + feature_i);
                    inline for (0..rows) |r| {
                        var current: Vec = acc[r][feature_i..][0..vector_width].*;
                        inline for (0..kb) |ki| current = @mulAdd(Vec, @as(Vec, @splat(p_arr[ki][r])), vv[ki], current);
                        acc[r][feature_i..][0..vector_width].* = current;
                    }
                }
                while (feature_i < task.d) : (feature_i += 1) {
                    inline for (0..kb) |ki| {
                        const v_value = widenKvScalar(Lane, v_rows[ki][v_offs[ki] + feature_i]);
                        inline for (0..rows) |r| acc[r][feature_i] += p_arr[ki][r] * v_value;
                    }
                }
            } else {
                // exp(0) == 1 exactly, but the select keeps unchanged lanes
                // (including m == -inf, whose difference is NaN) at exactly 1.
                const correction = @select(f32, m_same, ones, vexpf(rows, m_run - m_new));
                m_run = m_new;
                l_run = @mulAdd(RVec, l_run, correction, p_sum);
                const c_arr: [rows]f32 = correction;
                feature_i = 0;
                while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                    var vv: [kb]Vec = undefined;
                    inline for (0..kb) |ki| vv[ki] = widenKvVec(Lane, vector_width, v_rows[ki], v_offs[ki] + feature_i);
                    inline for (0..rows) |r| {
                        var current: Vec = acc[r][feature_i..][0..vector_width].*;
                        current *= @as(Vec, @splat(c_arr[r]));
                        inline for (0..kb) |ki| current = @mulAdd(Vec, @as(Vec, @splat(p_arr[ki][r])), vv[ki], current);
                        acc[r][feature_i..][0..vector_width].* = current;
                    }
                }
                while (feature_i < task.d) : (feature_i += 1) {
                    inline for (0..kb) |ki| {
                        const v_value = widenKvScalar(Lane, v_rows[ki][v_offs[ki] + feature_i]);
                        inline for (0..rows) |r| {
                            const rescaled = if (ki == 0) acc[r][feature_i] * c_arr[r] else acc[r][feature_i];
                            acc[r][feature_i] = rescaled + p_arr[ki][r] * v_value;
                        }
                    }
                }
            }
        }

        const inv_l = ones / l_run;
        const inv_arr: [rows]f32 = inv_l;
        const m_arr: [rows]f32 = m_run;
        const l_arr: [rows]f32 = l_run;
        inline for (0..rows) |r| {
            if (r / head_group < rows_active) {
                const head_i = head_unit * head_group + (r % head_group);
                if (task.stats) |stats| {
                    const stat_base = (head_i * task.q_seq + (q0 + r / head_group)) * 2;
                    stats[stat_base] = m_arr[r];
                    stats[stat_base + 1] = l_arr[r];
                }
                const out_base = (q0 + r / head_group) * out_seq_stride + head_i * task.d;
                const scale_vec: Vec = @splat(inv_arr[r]);
                var feature_i: usize = 0;
                while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                    const current: Vec = acc[r][feature_i..][0..vector_width].*;
                    task.out_data[out_base + feature_i ..][0..vector_width].* = current * scale_vec;
                }
                while (feature_i < task.d) : (feature_i += 1) {
                    task.out_data[out_base + feature_i] = acc[r][feature_i] * inv_arr[r];
                }
            }
        }
    }
}

/// Keys scanned by one query tile — the load-balance weight of a work item in
/// the tiled attention split (late tiles attend more keys under causal
/// masking, so equal-count task ranges would be lopsided). Bidirectional
/// tiles all scan the full key range, so their weights are uniform.
pub fn attentionTileKeyCount(q_tile: usize, q_seq: usize, kv_seq: usize, source_offset: usize, window: usize, causal: bool, tile_i: usize) u64 {
    if (!causal) return kv_seq;
    const q0 = tile_i * q_tile;
    const rows_active = @min(q_tile, q_seq - q0);
    const p_first = source_offset + q0;
    const active_last = p_first + rows_active;
    const lo_first = if (window == 0) 0 else (p_first + 1) -| window;
    return active_last - lo_first;
}

pub fn groupedCausalAttentionBackwardKvHeads(task: GroupedCausalAttentionBackwardTask) void {
    const q_head_stride = task.d;
    const q_seq_stride = task.heads * task.d;
    const kv_head_stride = task.d;
    const kv_seq_stride = task.kv_heads * task.d;
    const out_seq_stride = task.heads * task.d;
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (0..task.heads) |head_i| {
        const kv_head_i = task.kv_head_for_head[head_i];
        if (kv_head_i < task.kv_head_start or kv_head_i >= task.kv_head_end) continue;

        for (0..task.q_seq) |query_i| {
            const active = if (task.causal) task.source_offset + query_i + 1 else task.kv_seq;
            const lo = if (!task.causal or task.window == 0) 0 else active -| task.window;
            const q_base = query_i * q_seq_stride + head_i * q_head_stride;
            const gy_base = query_i * out_seq_stride + head_i * task.d;

            var max_score = -std.math.inf(f32);
            for (lo..active) |source_i| {
                const k_base = source_i * kv_seq_stride + kv_head_i * kv_head_stride;
                var dot_vec: Vec = @splat(0);
                var feature_i: usize = 0;
                while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                    const qv: Vec = task.q_data[q_base + feature_i ..][0..vector_width].*;
                    const kv: Vec = task.k_data[k_base + feature_i ..][0..vector_width].*;
                    dot_vec += qv * kv;
                }
                var dot_value: f32 = @reduce(.Add, dot_vec);
                while (feature_i < task.d) : (feature_i += 1) {
                    dot_value += task.q_data[q_base + feature_i] * task.k_data[k_base + feature_i];
                }
                const score = dot_value * task.scale_value;
                task.scores[source_i] = score;
                max_score = @max(max_score, score);
            }

            var sum_exp: f32 = 0;
            for (lo..active) |source_i| {
                const prob_unnormalized = @exp(task.scores[source_i] - max_score);
                task.scores[source_i] = prob_unnormalized;
                sum_exp += prob_unnormalized;
            }
            const inv_sum = 1 / sum_exp;
            for (lo..active) |source_i| {
                task.scores[source_i] *= inv_sum;
            }

            var dprob_dot: f32 = 0;
            for (lo..active) |source_i| {
                const v_base = source_i * kv_seq_stride + kv_head_i * kv_head_stride;
                var dprob_vec: Vec = @splat(0);
                var feature_i: usize = 0;
                while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                    const gy_vec: Vec = task.gy_data[gy_base + feature_i ..][0..vector_width].*;
                    const v_vec: Vec = task.v_data[v_base + feature_i ..][0..vector_width].*;
                    dprob_vec += gy_vec * v_vec;
                }
                var grad_value: f32 = @reduce(.Add, dprob_vec);
                while (feature_i < task.d) : (feature_i += 1) {
                    grad_value += task.gy_data[gy_base + feature_i] * task.v_data[v_base + feature_i];
                }
                task.dprob[source_i] = grad_value;
                dprob_dot += task.scores[source_i] * grad_value;
            }

            for (lo..active) |source_i| {
                const prob = task.scores[source_i];
                const dscore = prob * (task.dprob[source_i] - dprob_dot);
                const scaled_dscore = task.scale_value * dscore;
                const prob_vec: Vec = @splat(prob);
                const scaled_dscore_vec: Vec = @splat(scaled_dscore);
                const k_base = source_i * kv_seq_stride + kv_head_i * kv_head_stride;
                const v_base = source_i * kv_seq_stride + kv_head_i * kv_head_stride;

                if (task.v_grad) |grad| {
                    var feature_i: usize = 0;
                    while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                        const current: Vec = grad[v_base + feature_i ..][0..vector_width].*;
                        const gy_vec: Vec = task.gy_data[gy_base + feature_i ..][0..vector_width].*;
                        grad[v_base + feature_i ..][0..vector_width].* = current + prob_vec * gy_vec;
                    }
                    while (feature_i < task.d) : (feature_i += 1) {
                        grad[v_base + feature_i] += prob * task.gy_data[gy_base + feature_i];
                    }
                }

                if (task.q_grad) |grad| {
                    var feature_i: usize = 0;
                    while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                        const current: Vec = grad[q_base + feature_i ..][0..vector_width].*;
                        const k_vec: Vec = task.k_data[k_base + feature_i ..][0..vector_width].*;
                        grad[q_base + feature_i ..][0..vector_width].* = current + scaled_dscore_vec * k_vec;
                    }
                    while (feature_i < task.d) : (feature_i += 1) {
                        grad[q_base + feature_i] += scaled_dscore * task.k_data[k_base + feature_i];
                    }
                }

                if (task.k_grad) |grad| {
                    var feature_i: usize = 0;
                    while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                        const current: Vec = grad[k_base + feature_i ..][0..vector_width].*;
                        const q_vec: Vec = task.q_data[q_base + feature_i ..][0..vector_width].*;
                        grad[k_base + feature_i ..][0..vector_width].* = current + scaled_dscore_vec * q_vec;
                    }
                    while (feature_i < task.d) : (feature_i += 1) {
                        grad[k_base + feature_i] += scaled_dscore * task.q_data[q_base + feature_i];
                    }
                }
            }
        }
    }
}

pub fn hasAdjacentKvHeadPairs(kv_head_for_head: []const usize, heads: usize, kv_heads: usize) bool {
    if (heads != kv_heads * 2) return false;
    for (0..kv_heads) |kv_head_i| {
        if (kv_head_for_head[kv_head_i * 2] != kv_head_i) return false;
        if (kv_head_for_head[kv_head_i * 2 + 1] != kv_head_i) return false;
    }
    return true;
}

pub fn addSliceInPlace(dest: []f32, src: []const f32) void {
    std.debug.assert(dest.len == src.len);
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    var i: usize = 0;
    while (i + vector_width <= src.len) : (i += vector_width) {
        const dest_vec: Vec = dest[i..][0..vector_width].*;
        const src_vec: Vec = src[i..][0..vector_width].*;
        dest[i..][0..vector_width].* = dest_vec + src_vec;
    }
    while (i < src.len) : (i += 1) dest[i] += src[i];
}

/// Forward-saved stats route for the backward softmax reconstruction:
/// FUCINA_NO_ATTN_BWD_STATS=1 pins the 3-pass recompute (the A/B and
/// emergency-revert switch), FUCINA_ATTN_BWD_STATS=1 forces on. Read once,
/// cached. Only consulted when the caller has stats at all (the autograd
/// record); the stats-less exec path always recomputes.
var attn_bwd_stats_state = std.atomic.Value(u8).init(0); // 0 = unread, 1 = on, 2 = off
fn attnBwdStatsEnabled() bool {
    const state = attn_bwd_stats_state.load(.acquire);
    if (state != 0) return state == 1;
    const on = if (parallel.envPositiveUsize("FUCINA_NO_ATTN_BWD_STATS") != null)
        false
    else if (parallel.envPositiveUsize("FUCINA_ATTN_BWD_STATS") != null)
        true
    else
        true;
    attn_bwd_stats_state.store(if (on) 1 else 2, .release);
    return on;
}

/// Score panel -> probability panel. With forward-saved `stats` ({max,
/// sum_exp} pairs indexed by GLOBAL head via `head_indices`) each row is ONE
/// fused pass p = exp(s*scale - max) * inv_sum; without them the historical
/// 3-pass recompute (scale+max scan, exp+sum, normalize) runs. The stats
/// route reconstructs the FORWARD kernel's probabilities (its normalizer),
/// where the recompute route re-derives them from the GEMM scores — the two
/// agree to f32 roundoff, not bitwise (the route-parity test pins this).
pub fn groupedCausalAttentionBackwardSoftmaxRows(
    scores: []f32,
    stats: ?[]const f32,
    head_indices: []const usize,
    head_count: usize,
    q_seq: usize,
    kv_seq: usize,
    source_offset: usize,
    scale_value: f32,
    window: usize,
    causal: bool,
) void {
    std.debug.assert(scores.len == head_count * q_seq * kv_seq);
    for (0..head_count) |local_head_i| {
        for (0..q_seq) |query_i| {
            const row = scores[(local_head_i * q_seq + query_i) * kv_seq ..][0..kv_seq];
            const active = if (causal) source_offset + query_i + 1 else kv_seq;
            const lo = if (!causal or window == 0) 0 else active -| window;

            if (lo > 0) @memset(row[0..lo], 0);

            if (stats) |values| {
                const stat_base = (head_indices[local_head_i] * q_seq + query_i) * 2;
                const max_score = values[stat_base];
                const inv_sum = 1 / values[stat_base + 1];
                for (lo..active) |source_i| {
                    row[source_i] = @exp(row[source_i] * scale_value - max_score) * inv_sum;
                }
                if (active < kv_seq) @memset(row[active..kv_seq], 0);
                continue;
            }

            var max_score = -std.math.inf(f32);
            for (lo..active) |source_i| {
                const scaled_score = row[source_i] * scale_value;
                row[source_i] = scaled_score;
                max_score = @max(max_score, scaled_score);
            }

            var sum_exp: f32 = 0;
            for (lo..active) |source_i| {
                const probability = @exp(row[source_i] - max_score);
                row[source_i] = probability;
                sum_exp += probability;
            }

            const inv_sum = 1 / sum_exp;
            for (lo..active) |source_i| row[source_i] *= inv_sum;
            if (active < kv_seq) @memset(row[active..kv_seq], 0);
        }
    }
}

pub fn groupedCausalAttention(
    self: *Runtime,
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !Tensor {
    return groupedCausalAttentionImpl(self, f32, q, k, v, kv_head_for_head, scale_value, 0, true, null, null);
}

/// As `groupedCausalAttention` with a sliding-window `window` (0 = full
/// causal; else a query at absolute position `p` attends only keys in
/// `[max(0, p-window+1), p]`). Used by Gemma's local SWA layers.
pub fn groupedCausalAttentionWindowed(
    self: *Runtime,
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    kv_head_for_head: []const usize,
    scale_value: f32,
    window: usize,
) !Tensor {
    return groupedCausalAttentionImpl(self, f32, q, k, v, kv_head_for_head, scale_value, window, true, null, null);
}

/// Bidirectional (non-causal) grouped attention: every query row attends
/// EVERY key row. The block-diffusion canvas attention — canvas queries
/// attend the cached prefix plus the whole canvas; an SWA reach limit is
/// realized by narrowing the K/V views, not by a window here, so no
/// window parameter exists. Same GQA mapping/shapes as
/// `groupedCausalAttention` (the shared validation keeps
/// `q_seq <= kv_seq`, which every prefix+canvas layout satisfies).
pub fn groupedBidirectionalAttention(
    self: *Runtime,
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !Tensor {
    return groupedCausalAttentionImpl(self, f32, q, k, v, kv_head_for_head, scale_value, 0, false, null, null);
}

/// As `groupedBidirectionalAttention` with an additive f32 `bias` of shape
/// `[q_seq, kv_seq]` added to the SCALED scores before the softmax:
/// score(query s, key kv) = dot(q_s, k_kv) * scale_value + bias[s][kv]
/// (OmniVoice's uncond CFG +1.0/0.0 row — ggml_soft_max_ext mask semantics).
/// This is an additive soft bias, NOT -inf masking: a -inf bias value would
/// poison its query row on the tiled kernel like a NaN logit does.
pub fn groupedBidirectionalAttentionBiased(
    self: *Runtime,
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    kv_head_for_head: []const usize,
    scale_value: f32,
    bias: *const Tensor,
) !Tensor {
    return groupedCausalAttentionImpl(self, f32, q, k, v, kv_head_for_head, scale_value, 0, false, bias, null);
}

/// As the f32 grouped attention forwards (`causal`/`window` select the
/// causal, windowed, or bidirectional variant) additionally recording the
/// per-(head, query) softmax {max, sum_exp} statistics into `stats`
/// (interleaved f32 pairs, length heads * q_seq * 2). The output is
/// BITWISE identical to the stats-less entries — capture is write-only.
/// The stats feed `groupedCausalAttentionBackward`, which then rebuilds
/// this forward's probabilities in one pass instead of three.
pub fn groupedCausalAttentionStatsOut(
    self: *Runtime,
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    kv_head_for_head: []const usize,
    scale_value: f32,
    window: usize,
    causal: bool,
    stats: []f32,
) !Tensor {
    return groupedCausalAttentionImpl(self, f32, q, k, v, kv_head_for_head, scale_value, window, causal, null, stats);
}

/// D[head * q_seq + query] = dot over d of gy row and forward-output row —
/// the softmax-backward row dot via the output identity sum(P*dP) = gy.O.
/// Both operands are [q_seq, heads * d] row-major.
fn attentionBackwardRowDots(gy_data: []const f32, out_data: []const f32, row_dots: []f32, q_seq: usize, heads: usize, d: usize) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    for (0..heads) |head_i| {
        for (0..q_seq) |query_i| {
            const base = query_i * heads * d + head_i * d;
            var dot_vec: Vec = @splat(0);
            var feature_i: usize = 0;
            while (feature_i + vector_width <= d) : (feature_i += vector_width) {
                const gv: Vec = gy_data[base + feature_i ..][0..vector_width].*;
                const ov: Vec = out_data[base + feature_i ..][0..vector_width].*;
                dot_vec += gv * ov;
            }
            var dot = @reduce(.Add, dot_vec);
            while (feature_i < d) : (feature_i += 1) {
                dot += gy_data[base + feature_i] * out_data[base + feature_i];
            }
            row_dots[head_i * q_seq + query_i] = dot;
        }
    }
}

fn groupedCausalAttentionBackwardGemm(
    self: *Runtime,
    q_data: []const f32,
    k_data: []const f32,
    v_data: []const f32,
    gy_data: []const f32,
    stats: ?[]const f32,
    row_dots: ?[]const f32,
    q_grad: ?[]f32,
    k_grad: ?[]f32,
    v_grad: ?[]f32,
    kv_head_for_head: []const usize,
    q_seq: usize,
    kv_seq: usize,
    source_offset: usize,
    heads: usize,
    d: usize,
    kv_heads: usize,
    scale_value: f32,
    window: usize,
    causal: bool,
) !void {
    const q_seq_stride = heads * d;
    const kv_seq_stride = kv_heads * d;
    const need_score_grad = q_grad != null or k_grad != null;

    for (0..kv_heads) |kv_head_i| {
        var head_count: usize = 0;
        for (kv_head_for_head) |mapped_kv_head| {
            if (mapped_kv_head == kv_head_i) head_count += 1;
        }
        if (head_count == 0) continue;

        const head_indices = try self.allocator.alloc(usize, head_count);
        defer self.allocator.free(head_indices);
        {
            var local_head_i: usize = 0;
            for (kv_head_for_head, 0..) |mapped_kv_head, head_i| {
                if (mapped_kv_head == kv_head_i) {
                    head_indices[local_head_i] = head_i;
                    local_head_i += 1;
                }
            }
        }

        const rows = head_count * q_seq;
        var q_panel = try self.emptyRank(2, .{ rows, d });
        defer q_panel.deinit();
        var gy_panel = try self.emptyRank(2, .{ rows, d });
        defer gy_panel.deinit();
        var k_panel = try self.emptyRank(2, .{ kv_seq, d });
        defer k_panel.deinit();
        var prob_panel = try self.emptyRank(2, .{ rows, kv_seq });
        defer prob_panel.deinit();

        const q_panel_data = q_panel.data();
        const gy_panel_data = gy_panel.data();
        const k_panel_data = k_panel.data();

        for (head_indices, 0..) |head_i, local_head_i| {
            for (0..q_seq) |query_i| {
                const row = local_head_i * q_seq + query_i;
                const q_base = query_i * q_seq_stride + head_i * d;
                const panel_base = row * d;
                @memcpy(q_panel_data[panel_base..][0..d], q_data[q_base..][0..d]);
                @memcpy(gy_panel_data[panel_base..][0..d], gy_data[q_base..][0..d]);
            }
        }

        for (0..kv_seq) |source_i| {
            const source_base = source_i * kv_seq_stride + kv_head_i * d;
            @memcpy(k_panel_data[source_i * d ..][0..d], k_data[source_base..][0..d]);
        }

        self.enableNativeMatmulPoolForWork(rows, kv_seq, d);
        self.backend.matmulTransB2DIntoUnchecked(&prob_panel, &q_panel, &k_panel, rows, kv_seq, d);
        const probabilities = prob_panel.data();
        groupedCausalAttentionBackwardSoftmaxRows(
            probabilities,
            stats,
            head_indices,
            head_count,
            q_seq,
            kv_seq,
            source_offset,
            scale_value,
            window,
            causal,
        );

        if (v_grad) |grad| {
            var dv_panel = try self.emptyRank(2, .{ kv_seq, d });
            defer dv_panel.deinit();
            self.enableNativeMatmulPoolForWork(kv_seq, d, rows);
            self.backend.matmulTransA2DIntoUnchecked(&dv_panel, &prob_panel, &gy_panel, kv_seq, d, rows);

            const dv_panel_data = dv_panel.dataConst();
            for (0..kv_seq) |source_i| {
                const grad_base = source_i * kv_seq_stride + kv_head_i * d;
                addSliceInPlace(grad[grad_base..][0..d], dv_panel_data[source_i * d ..][0..d]);
            }
        }

        if (need_score_grad) {
            var v_panel = try self.emptyRank(2, .{ kv_seq, d });
            defer v_panel.deinit();
            const v_panel_data = v_panel.data();
            for (0..kv_seq) |source_i| {
                const source_base = source_i * kv_seq_stride + kv_head_i * d;
                @memcpy(v_panel_data[source_i * d ..][0..d], v_data[source_base..][0..d]);
            }

            var dscore_panel = try self.emptyRank(2, .{ rows, kv_seq });
            defer dscore_panel.deinit();
            self.enableNativeMatmulPoolForWork(rows, kv_seq, d);
            self.backend.matmulTransB2DIntoUnchecked(&dscore_panel, &gy_panel, &v_panel, rows, kv_seq, d);
            const dscore_data = dscore_panel.data();
            groupedCausalAttentionBackwardDScoreRows(
                probabilities,
                dscore_data,
                row_dots,
                head_indices,
                head_count,
                q_seq,
                kv_seq,
                source_offset,
                scale_value,
                window,
                causal,
            );

            if (q_grad) |grad| {
                var dq_panel = try self.emptyRank(2, .{ rows, d });
                defer dq_panel.deinit();
                self.enableNativeMatmulPoolForWork(rows, d, kv_seq);
                self.backend.matmul2DIntoUnchecked(&dq_panel, &dscore_panel, &k_panel, rows, d, kv_seq);

                const dq_panel_data = dq_panel.dataConst();
                for (head_indices, 0..) |head_i, local_head_i| {
                    for (0..q_seq) |query_i| {
                        const row = local_head_i * q_seq + query_i;
                        const grad_base = query_i * q_seq_stride + head_i * d;
                        addSliceInPlace(grad[grad_base..][0..d], dq_panel_data[row * d ..][0..d]);
                    }
                }
            }

            if (k_grad) |grad| {
                var dk_panel = try self.emptyRank(2, .{ kv_seq, d });
                defer dk_panel.deinit();
                self.enableNativeMatmulPoolForWork(kv_seq, d, rows);
                self.backend.matmulTransA2DIntoUnchecked(&dk_panel, &dscore_panel, &q_panel, kv_seq, d, rows);

                const dk_panel_data = dk_panel.dataConst();
                for (0..kv_seq) |source_i| {
                    const grad_base = source_i * kv_seq_stride + kv_head_i * d;
                    addSliceInPlace(grad[grad_base..][0..d], dk_panel_data[source_i * d ..][0..d]);
                }
            }
        }
    }
}

/// `stats` (optional): the forward's saved per-(head, query) {max, sum_exp}
/// pairs from `groupedCausalAttentionStatsOut` — length heads * q_seq * 2.
/// `out` (optional, requires stats): the forward's OUTPUT [q_seq, heads*d];
/// the softmax-backward row dot then comes from the identity
/// sum(P*dP) = gy.O (length-d dots) instead of a kv_seq-length pass over
/// both panels. Both are consumed only by the GEMM route (gated together by
/// FUCINA_NO_ATTN_BWD_STATS); the direct per-kv-head route always recomputes.
pub fn groupedCausalAttentionBackward(
    self: *Runtime,
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    gy: *const Tensor,
    kv_head_for_head: []const usize,
    scale_value: f32,
    window: usize,
    causal: bool,
    stats: ?[]const f32,
    out: ?*const Tensor,
    need_q: bool,
    need_k: bool,
    need_v: bool,
) !GroupedCausalAttentionBackwardResult {
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

    var qq = try self.prepareContiguous(q);
    defer qq.deinit();
    var kk = try self.prepareContiguous(k);
    defer kk.deinit();
    var vv = try self.prepareContiguous(v);
    defer vv.deinit();
    var ggy = try self.prepareContiguous(gy);
    defer ggy.deinit();

    const q_data = qq.tensor().dataConst();
    const k_data = kk.tensor().dataConst();
    const v_data = vv.tensor().dataConst();
    const gy_data = ggy.tensor().dataConst();

    var result = GroupedCausalAttentionBackwardResult{};
    errdefer result.deinit();
    if (need_q) result.q = try zerosRank(self, 3, .{ q_seq, heads, d });
    if (need_k) result.k = try zerosRank(self, 3, .{ kv_seq, kv_heads, d });
    if (need_v) result.v = try zerosRank(self, 3, .{ kv_seq, kv_heads, d });
    const q_grad: ?[]f32 = if (result.q) |*value| value.data() else null;
    const k_grad: ?[]f32 = if (result.k) |*value| value.data() else null;
    const v_grad: ?[]f32 = if (result.v) |*value| value.data() else null;
    const source_offset = kv_seq - q_seq;

    const attention_work = parallel.saturatedMul3(q_seq, kv_seq, heads * d);
    const stats_active: ?[]const f32 = if (stats != null and attnBwdStatsEnabled()) stats else null;
    if (stats_active) |values| {
        if (values.len != heads * q_seq * 2) return tensor.TensorError.InvalidDataLength;
    }
    // D = gy.O per (head, query) — computed ONCE for all kv heads (the dot
    // depends only on the query row). Rides the stats gate: both shortcuts
    // reconstruct forward-side values instead of re-deriving panel-side ones.
    var row_dots_storage: ?*storage.Buffer = null;
    defer if (row_dots_storage) |buffer| buffer.release();
    var row_dots: ?[]const f32 = null;
    if (stats_active != null) if (out) |out_tensor| {
        const out_view = try out_tensor.rankView(2);
        if (out_view.shape[0] != q_seq or out_view.shape[1] != heads * d) return tensor.TensorError.ShapeMismatch;
        var oo = try self.prepareContiguous(out_tensor);
        defer oo.deinit();
        const buffer = try self.buffers.acquire(heads * q_seq);
        row_dots_storage = buffer;
        const dots = buffer.data[0 .. heads * q_seq];
        attentionBackwardRowDots(gy_data, oo.tensor().dataConst(), dots, q_seq, heads, d);
        row_dots = dots;
    };
    if ((need_q or need_k or need_v) and
        q_seq >= 8 and kv_seq >= 8 and d >= 4 and
        attention_work >= grouped_attention_backward_gemm_work_threshold)
    {
        try groupedCausalAttentionBackwardGemm(
            self,
            q_data,
            k_data,
            v_data,
            gy_data,
            stats_active,
            row_dots,
            q_grad,
            k_grad,
            v_grad,
            kv_head_for_head,
            q_seq,
            kv_seq,
            source_offset,
            heads,
            d,
            kv_heads,
            scale_value,
            window,
            causal,
        );
        return result;
    }

    if (attention_work >= parallel.vector_matmul_work_threshold / 2 and kv_heads > 1) {
        if (workPool(self)) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), kv_heads);
            var task_storage: [parallel.vector_max_threads]GroupedCausalAttentionBackwardTask = undefined;
            const task_scratch = try self.allocator.alloc(f32, task_count * kv_seq * 2);
            defer self.allocator.free(task_scratch);
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
    defer if (heap_scratch) |values| self.allocator.free(values);
    const scores = if (kv_seq <= stack_scores.len) stack_scores[0..kv_seq] else blk: {
        const values = try self.allocator.alloc(f32, kv_seq * 2);
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

/// Same as `groupedCausalAttention` but the cached K/V are f16 (decode KV
/// cache): half the bandwidth, widened to f32 in the kernel. Q and the
/// output stay f32.
pub fn groupedCausalAttentionF16Kv(
    self: *Runtime,
    q: *const Tensor,
    k: *const tensor.TensorOf(.f16),
    v: *const tensor.TensorOf(.f16),
    kv_head_for_head: []const usize,
    scale_value: f32,
) !Tensor {
    return groupedCausalAttentionImpl(self, f16, q, k, v, kv_head_for_head, scale_value, 0, true, null, null);
}

/// f16-KV bidirectional attention (see `groupedBidirectionalAttention`):
/// the block-diffusion canvas pass over a prefix+canvas f16 KV cache.
pub fn groupedBidirectionalAttentionF16Kv(
    self: *Runtime,
    q: *const Tensor,
    k: *const tensor.TensorOf(.f16),
    v: *const tensor.TensorOf(.f16),
    kv_head_for_head: []const usize,
    scale_value: f32,
) !Tensor {
    return groupedCausalAttentionImpl(self, f16, q, k, v, kv_head_for_head, scale_value, 0, false, null, null);
}

/// f16-KV decode attention with a sliding `window` (see
/// `groupedCausalAttentionWindowed`).
pub fn groupedCausalAttentionF16KvWindowed(
    self: *Runtime,
    q: *const Tensor,
    k: *const tensor.TensorOf(.f16),
    v: *const tensor.TensorOf(.f16),
    kv_head_for_head: []const usize,
    scale_value: f32,
    window: usize,
) !Tensor {
    return groupedCausalAttentionImpl(self, f16, q, k, v, kv_head_for_head, scale_value, window, true, null, null);
}

/// Same as `groupedCausalAttention` but the cached K/V are q8_0 blocks
/// (the quantized decode KV cache, ~quarter the f32 bandwidth and half
/// f16's): `k_blocks`/`v_blocks` hold `kv_seq * kv_heads * d/32`
/// BlockQ8_0 laid out `[kv_seq, kv_heads, d/32]`, so each (position,
/// kv_head) row segment is `d/32` consecutive blocks. Kernels dequantize
/// each row into per-task L1 scratch as they stream — traffic from the
/// cache stays the quantized 34 bytes/block. Q and the output stay f32.
/// Requires `d % 32 == 0` and `d <= attention_q8_max_d`.
pub fn groupedCausalAttentionQ8Kv(
    self: *Runtime,
    q: *const Tensor,
    k_blocks: []const BlockQ8_0,
    v_blocks: []const BlockQ8_0,
    kv_seq: usize,
    kv_heads: usize,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !Tensor {
    return groupedCausalAttentionQ8KvImpl(self, q, k_blocks, v_blocks, kv_seq, kv_heads, kv_head_for_head, scale_value, 0);
}

/// q8_0-KV attention with a sliding `window` (see
/// `groupedCausalAttentionWindowed`).
pub fn groupedCausalAttentionQ8KvWindowed(
    self: *Runtime,
    q: *const Tensor,
    k_blocks: []const BlockQ8_0,
    v_blocks: []const BlockQ8_0,
    kv_seq: usize,
    kv_heads: usize,
    kv_head_for_head: []const usize,
    scale_value: f32,
    window: usize,
) !Tensor {
    return groupedCausalAttentionQ8KvImpl(self, q, k_blocks, v_blocks, kv_seq, kv_heads, kv_head_for_head, scale_value, window);
}

fn groupedCausalAttentionQ8KvImpl(
    self: *Runtime,
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

    var qq = try self.prepareContiguous(q);
    defer qq.deinit();

    var out = try self.emptyRank(2, .{ q_seq, heads * d });
    errdefer out.deinit();
    try groupedCausalAttentionDispatch(self, BlockQ8_0, qq.tensor().dataConst(), k_blocks, v_blocks, out.data(), kv_head_for_head, q_seq, kv_seq, heads, d, kv_heads, scale_value, window, true, null, null);
    return out;
}

/// One (stream × head-unit) work range of the ragged multi-stream decode
/// attention (`groupedCausalAttentionMulti*Kv`): query row `s` of `q_data`
/// attends the leading `lens[s]` cached rows of stream `s`'s K/V slices.
/// Work items are stream-major (`stream_i * n_units + unit_i`); each item
/// runs ONE head unit of ONE stream through the per-query decode kernels,
/// so per-stream results are bit-identical to N single-stream calls.
pub fn GroupedCausalAttentionMultiTask(comptime KvElem: type) type {
    return struct {
        q_data: []const f32,
        out_data: []f32,
        ks: []const []const KvElem,
        vs: []const []const KvElem,
        lens: []const usize,
        kv_head_for_head: []const usize,
        heads: usize,
        d: usize,
        kv_heads: usize,
        scale_value: f32,
        /// Head units per stream: kv-head pairs on the pair path, single
        /// query heads on the general path.
        n_units: usize,
        work_start: usize,
        work_end: usize,
        /// Softmax scratch, `2 * max(lens)` floats (the pair kernel needs
        /// two rows; the general path uses the first `lens[s]`).
        scores: []f32,
    };
}

pub fn runGroupedCausalAttentionMultiTask(comptime KvElem: type, comptime pair: bool) fn (*const GroupedCausalAttentionMultiTask(KvElem)) void {
    return struct {
        fn run(task: *const GroupedCausalAttentionMultiTask(KvElem)) void {
            groupedCausalAttentionMultiUnits(KvElem, pair, task.*);
        }
    }.run;
}

pub fn groupedCausalAttentionMultiUnits(comptime KvElem: type, comptime pair: bool, task: GroupedCausalAttentionMultiTask(KvElem)) void {
    const row = task.heads * task.d;
    for (task.work_start..task.work_end) |work_i| {
        const stream_i = work_i / task.n_units;
        const unit_i = work_i % task.n_units;
        const kv_seq = task.lens[stream_i];
        const q_row = task.q_data[stream_i * row ..][0..row];
        const out_row = task.out_data[stream_i * row ..][0..row];
        if (pair) {
            groupedCausalAttentionHeadPairs(KvElem, .{
                .q_data = q_row,
                .k_data = task.ks[stream_i],
                .v_data = task.vs[stream_i],
                .out_data = out_row,
                .q_seq = 1,
                .kv_seq = kv_seq,
                .source_offset = kv_seq - 1,
                .heads = task.heads,
                .d = task.d,
                .kv_heads = task.kv_heads,
                .scale_value = task.scale_value,
                .window = 0,
                .kv_head_start = unit_i,
                .kv_head_end = unit_i + 1,
                .scores = task.scores[0 .. kv_seq * 2],
            });
        } else {
            groupedCausalAttentionHeads(KvElem, .{
                .q_data = q_row,
                .k_data = task.ks[stream_i],
                .v_data = task.vs[stream_i],
                .out_data = out_row,
                .kv_head_for_head = task.kv_head_for_head,
                .q_seq = 1,
                .kv_seq = kv_seq,
                .source_offset = kv_seq - 1,
                .heads = task.heads,
                .d = task.d,
                .kv_heads = task.kv_heads,
                .scale_value = task.scale_value,
                .window = 0,
                .head_start = unit_i,
                .head_end = unit_i + 1,
                .scores = task.scores[0..kv_seq],
            });
        }
    }
}

/// Elements per cached (position) row: `kv_heads * d` f16/f32 lanes, or
/// `kv_heads * d/32` BlockQ8_0.
fn kvRowElems(comptime KvElem: type, kv_heads: usize, d: usize) usize {
    return if (KvElem == BlockQ8_0) kv_heads * (d / q8_0_block_size) else kv_heads * d;
}

/// Ragged multi-stream decode attention over per-stream f16 KV caches
/// (the batch-N decode shape): query row `s` of `q` — exactly one query
/// per stream, `[n_streams, heads, d]` — attends ALL `lens[s]` cached
/// positions of stream `s`. `ks[s]`/`vs[s]` hold at least `lens[s]`
/// leading `[kv_heads, d]` rows of that stream's cache layer. Dispatch
/// schedules flattened (stream, head-unit) items weighted by stream
/// length over the SAME per-query kernels m=1 decode uses, so each
/// stream's output is bit-identical to its own single-stream
/// `groupedCausalAttentionF16Kv` call.
pub fn groupedCausalAttentionMultiF16Kv(
    self: *Runtime,
    q: *const Tensor,
    ks: []const []const f16,
    vs: []const []const f16,
    lens: []const usize,
    kv_heads: usize,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !Tensor {
    return groupedCausalAttentionMultiImpl(self, f16, q, ks, vs, lens, kv_heads, kv_head_for_head, scale_value);
}

/// As `groupedCausalAttentionMultiF16Kv` for q8_0 caches: `ks[s]`/`vs[s]`
/// hold `lens[s] * kv_heads * d/32` leading BlockQ8_0 laid out
/// `[len, kv_heads, d/32]` (the `kBlocks`/`vBlocks` shape). Requires
/// `d % 32 == 0` and `d <= attention_q8_max_d`.
pub fn groupedCausalAttentionMultiQ8Kv(
    self: *Runtime,
    q: *const Tensor,
    ks: []const []const BlockQ8_0,
    vs: []const []const BlockQ8_0,
    lens: []const usize,
    kv_heads: usize,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !Tensor {
    return groupedCausalAttentionMultiImpl(self, BlockQ8_0, q, ks, vs, lens, kv_heads, kv_head_for_head, scale_value);
}

fn groupedCausalAttentionMultiImpl(
    self: *Runtime,
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

    var qq = try self.prepareContiguous(q);
    defer qq.deinit();

    var out = try self.emptyRank(2, .{ n, heads * d });
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
    if (attention_work >= parallel.vector_matmul_work_threshold / 2 and total_work > 1) {
        if (workPool(self)) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), total_work);
            var task_storage: [parallel.vector_max_threads]GroupedCausalAttentionMultiTask(KvElem) = undefined;
            const scores_storage = try self.buffers.acquire(task_count * scores_per_task);
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
        const buffer = try self.buffers.acquire(scores_per_task);
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
    self: *Runtime,
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
    if (comptime backend_mod.gpu_impl.enabled and KvElem == f16) attn_gpu: {
        const gpu = backend_mod.gpu_impl;
        if (base.bias != null or base.d > 256 or base.heads > 64) break :attn_gpu;
        if (!gpu.shouldUseGpuAttn(base.q_seq, base.kv_seq, base.heads, base.d)) break :attn_gpu;
        var map_buf: [64]i32 = undefined;
        for (0..base.heads) |h| {
            // The pair path (head_group == 2) maps two adjacent q heads onto
            // one kv head implicitly; the general path carries the explicit map.
            map_buf[h] = if (head_group == 2) @intCast(h / 2) else @intCast(base.kv_head_for_head[h]);
        }
        if (gpu.attnPrefillF16(
            base.q_data,
            base.k_data,
            base.v_data,
            base.out_data,
            map_buf[0..base.heads],
            base.q_seq,
            base.kv_seq,
            base.heads,
            base.kv_heads,
            base.d,
            base.source_offset,
            base.scale_value,
            base.window,
            base.causal,
        )) return;
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
    const gate_ok = attention_work >= parallel.vector_matmul_work_threshold / 2;
    const pool = (if (gate_ok) workPool(self) else null) orelse {
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
    self: *Runtime,
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

    var qq = try self.prepareContiguous(q);
    defer qq.deinit();
    var kk = try self.prepareContiguousTyped(kv_dtype, k);
    defer kk.deinit();
    var vv = try self.prepareContiguousTyped(kv_dtype, v);
    defer vv.deinit();

    // The optional additive bias is validated to [q_seq, kv_seq] here and
    // handed to the kernels as a row-contiguous slice.
    var bias_prepared: ?Runtime.PreparedTensor = null;
    defer if (bias_prepared) |*prepared| prepared.deinit();
    const bias_data: ?[]const f32 = if (bias) |bias_tensor| blk: {
        const bias_view = try bias_tensor.rankView(2);
        if (bias_view.shape[0] != q_seq or bias_view.shape[1] != kv_seq) return tensor.TensorError.InvalidShape;
        bias_prepared = try self.prepareContiguous(bias_tensor);
        break :blk bias_prepared.?.tensor().dataConst();
    } else null;

    const q_data = qq.tensor().dataConst();
    const k_data = kk.tensor().dataConst();
    const v_data = vv.tensor().dataConst();

    if (stats) |values| {
        if (values.len != heads * q_seq * 2) return tensor.TensorError.InvalidDataLength;
    }

    var out = try self.emptyRank(2, .{ q_seq, heads * d });
    errdefer out.deinit();
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
    self: *Runtime,
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
    if (attention_work >= parallel.vector_matmul_work_threshold / 2 and heads > 1) {
        if (workPool(self)) |pool| {
            if (can_pair_heads) {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), kv_heads);
                var task_storage: [parallel.vector_max_threads]GroupedCausalAttentionPairTask(KvElem) = undefined;
                // Per-task softmax scratch from the buffer pool: at long
                // contexts this crosses the allocator's mmap threshold, so a
                // plain alloc would pay a syscall pair + page faults per
                // layer per token.
                const scores_storage = try self.buffers.acquire(task_count * kv_seq * 2);
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
            const scores_storage = try self.buffers.acquire(task_count * kv_seq);
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
            const buffer = try self.buffers.acquire(kv_seq * 2);
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
        const buffer = try self.buffers.acquire(kv_seq);
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

/// Probability + dP panels -> score-gradient panel: dS = scale * P * (dP - D)
/// with D = row dot of P and dP. When `row_dots` is provided (the forward's
/// output identity D = gy.O, indexed by GLOBAL head via `head_indices` —
/// length-d dots instead of length-kv_seq ones), the per-row kv_seq dot pass
/// over BOTH panels is skipped entirely; the values agree with the in-panel
/// dot to f32 roundoff (same tolerance class as the stats softmax route,
/// same FUCINA_NO_ATTN_BWD_STATS gate upstream).
pub fn groupedCausalAttentionBackwardDScoreRows(
    probabilities: []const f32,
    dscore: []f32,
    row_dots: ?[]const f32,
    head_indices: []const usize,
    head_count: usize,
    q_seq: usize,
    kv_seq: usize,
    source_offset: usize,
    scale_value: f32,
    window: usize,
    causal: bool,
) void {
    std.debug.assert(probabilities.len == head_count * q_seq * kv_seq);
    std.debug.assert(dscore.len == probabilities.len);
    for (0..head_count) |local_head_i| {
        for (0..q_seq) |query_i| {
            const row_offset = (local_head_i * q_seq + query_i) * kv_seq;
            const prob_row = probabilities[row_offset..][0..kv_seq];
            const dscore_row = dscore[row_offset..][0..kv_seq];
            const active = if (causal) source_offset + query_i + 1 else kv_seq;
            const lo = if (!causal or window == 0) 0 else active -| window;

            var dprob_dot: f32 = 0;
            if (row_dots) |dots| {
                dprob_dot = dots[head_indices[local_head_i] * q_seq + query_i];
            } else {
                for (lo..active) |source_i| dprob_dot += prob_row[source_i] * dscore_row[source_i];
            }

            if (lo > 0) @memset(dscore_row[0..lo], 0);
            for (lo..active) |source_i| {
                dscore_row[source_i] = scale_value * prob_row[source_i] * (dscore_row[source_i] - dprob_dot);
            }
            if (active < kv_seq) @memset(dscore_row[active..kv_seq], 0);
        }
    }
}

test {
    // Group-B tiled-attention parity tests (they drive this module's pub
    // tiled kernel + Task; kept out of exec.zig's inline Group-A tests).
    _ = @import("attention_tests.zig");
}
