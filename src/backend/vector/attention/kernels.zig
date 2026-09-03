//! The grouped-causal attention kernels: per-query units (decode, short
//! prefill; f32/f16/q8_0 KV), the query-tiled online-softmax prefill, the
//! tiled backward and its BLAS-strip variant, the partial reduce, and the
//! multi-stream decode, each with its scalar reference twin in `scalar`.
//! Task payloads, adapters and tile constants are the seam one level up
//! (`../attention.zig`); the bodies here are reached through
//! `backend.kernels` only.
const std = @import("std");
const common = @import("../common.zig");
const build_options = @import("build_options");
const dtype_mod = @import("../../../dtype.zig");
const isa = @import("../../isa.zig");
const quantized_matmul = @import("../../quant.zig");
const vexpf = @import("../primitives.zig").vexpf;
const tasks = @import("../attention.zig");

const DType = dtype_mod.DType;
const BlockQ8_0 = tasks.BlockQ8_0;
const q8_0_block_size = tasks.q8_0_block_size;
const attention_q8_max_d = tasks.attention_q8_max_d;
const GroupedCausalAttentionTask = tasks.GroupedCausalAttentionTask;
const GroupedCausalAttentionPairTask = tasks.GroupedCausalAttentionPairTask;
const attention_tile_rows = tasks.attention_tile_rows;
const attention_key_block = tasks.attention_key_block;
const attention_tile_max_d = tasks.attention_tile_max_d;
const GroupedCausalAttentionTiledTask = tasks.GroupedCausalAttentionTiledTask;
const GroupedCausalAttentionBackwardTask = tasks.GroupedCausalAttentionBackwardTask;
const attention_bwd_tile_rows = tasks.attention_bwd_tile_rows;
const GroupedCausalAttentionBackwardTiledTask = tasks.GroupedCausalAttentionBackwardTiledTask;
const attention_bwd_blas_tile_rows = tasks.attention_bwd_blas_tile_rows;
const AttentionBackwardReduceTask = tasks.AttentionBackwardReduceTask;
const GroupedCausalAttentionMultiTask = tasks.GroupedCausalAttentionMultiTask;

// The CBLAS provider, reachable only on BLAS-backed native builds (the
// same comptime gate as `backend.blas`); the BLAS-strip backward guards
// every use behind `blas_available`.
const blas_available = build_options.backend_kind == .native and build_options.use_blas;
const blas = if (blas_available) @import("../../blas.zig") else struct {};

// K/V come in either as f32 (prefill, no cache), f16 (the decode KV cache,
// half the bandwidth), or q8_0 blocks (the quantized KV cache, ~quarter the
// bandwidth of f32). The attention kernels are generic over that element type:
// f16 lanes are widened to f32 in-register (the f32 instantiation compiles to
// the same loads as before — the comptime branches below collapse to a plain
// load). q8_0 splits by kernel: the per-query kernels (decode, short prefill)
// run the INTEGER path — the query row is quantized once to q8_0 and scores
// are q8xq8 sdot dots straight on the cached K blocks, with the V dequant
// fused into the weighted accumulate — so the KV sweep reads only quantized
// bytes and never materializes an f32 row. The query-tiled prefill kernel
// keeps the dequant-scratch path (`kvRowSelect`): its per-tile row reuse
// already amortizes the dequant, and its f32 scores stay bit-exact vs the
// f32 kernel on a dequantized cache.
inline fn widenKvVec(comptime KvElem: type, comptime width: usize, data: []const KvElem, offset: usize) @Vector(width, f32) {
    const chunk: @Vector(width, KvElem) = data[offset..][0..width].*;
    return if (KvElem == f32) chunk else @floatCast(chunk);
}

inline fn widenKvScalar(comptime KvElem: type, value: KvElem) f32 {
    return if (KvElem == f32) value else @floatCast(value);
}

/// Lane type the attention inner loops consume: q8_0 rows are dequantized to
/// f32 scratch first, f32/f16 are read (and widened) directly.
fn KvLane(comptime KvElem: type) type {
    return if (KvElem == BlockQ8_0) f32 else KvElem;
}

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
        quantized_matmul.q8k.dequantizeRowQ8_0Into(
            scratch[0..d],
            data[elem_base / q8_0_block_size ..][0 .. d / q8_0_block_size],
        ) catch unreachable;
        return .{ scratch[0..d], 0 };
    }
    return .{ data, elem_base };
}

/// The per-query attention kernel family (decode, short prefill), one body
/// for both work-unit shapes:
///
/// - `head_group == 1` (`groupedCausalAttentionHeads`): one query head per
///   work unit, arbitrary `kv_head_for_head` mapping.
/// - `head_group == 2` (`groupedCausalAttentionHeadPairs`): the
///   heads == 2*kv_heads adjacent-pair GQA grouping — TWO query heads sharing
///   one KV head walk the cache together, so each K/V row is loaded once for
///   both (the GQA pairing win).
///
/// Three phases per query row: score the group's heads into their scratch
/// rows tracking each max, exp-normalize in place, then the weighted V pass —
/// fused (pair-)row Q8_0 kernels when V is block-quantized, SIMD
/// widen-per-lane otherwise. Every arm unrolls by `head_group` with the same
/// per-head operations and accumulator per head, so each instantiation is
/// bit-identical to the historical hand-written kernel it replaces.
fn groupedCausalAttentionUnits(
    comptime KvElem: type,
    comptime head_group: usize,
    task: if (head_group == 2) GroupedCausalAttentionPairTask(KvElem) else GroupedCausalAttentionTask(KvElem),
) void {
    comptime std.debug.assert(head_group == 1 or head_group == 2);
    const q_head_stride = task.d;
    const q_seq_stride = task.heads * task.d;
    const kv_head_stride = task.d;
    const kv_seq_stride = task.kv_heads * task.d;
    const out_seq_stride = task.heads * task.d;
    const Vec = common.RowVec;
    const vector_width = common.row_lanes;
    const Lane = KvLane(KvElem);
    // q8_0 KV: each of the group's query rows is quantized ONCE per (query,
    // unit) and the score pass runs the integer q8xq8 dot straight on the
    // cached K blocks — no dequant scratch, the sweep reads only the
    // quantized bytes (the shared K row is dotted for the whole group in one
    // pass over its blocks on the pair path). The V pass fuses dequant into
    // the weighted accumulate the same way. (The tiled prefill kernel keeps
    // the dequant-scratch path: its per-tile row reuse already amortizes the
    // dequant.)
    const q8_blocks = comptime attention_q8_max_d / q8_0_block_size;
    var q_q8: [if (KvElem == BlockQ8_0) head_group * q8_blocks else 0]BlockQ8_0 = undefined;
    var scores: [head_group][]f32 = undefined;
    inline for (0..head_group) |j| scores[j] = task.scores[j * task.kv_seq ..][0..task.kv_seq];

    const unit_start = if (head_group == 2) task.kv_head_start else task.head_start;
    const unit_end = if (head_group == 2) task.kv_head_end else task.head_end;
    for (unit_start..unit_end) |unit_i| {
        const kv_head_i = if (head_group == 2) unit_i else task.kv_head_for_head[unit_i];
        const head_base = unit_i * head_group;
        for (0..task.q_seq) |query_i| {
            const active = if (task.causal) task.source_offset + query_i + 1 else task.kv_seq;
            const lo = if (!task.causal or task.window == 0) 0 else active -| task.window;
            var q_base: [head_group]usize = undefined;
            inline for (0..head_group) |j| q_base[j] = query_i * q_seq_stride + (head_base + j) * q_head_stride;
            const bias_row: ?[]const f32 = if (task.bias) |bias_data| bias_data[query_i * task.kv_seq ..][0..task.kv_seq] else null;

            var q_scales: [if (KvElem == BlockQ8_0) head_group * q8_blocks else 0]f32 = undefined;
            if (comptime KvElem == BlockQ8_0) {
                const qm = quantized_matmul;
                inline for (0..head_group) |j| {
                    qm.q8k.quantizeRowQ8_0IntoUnchecked(q_q8[j * q8_blocks ..][0 .. task.d / q8_0_block_size], task.q_data[q_base[j]..][0..task.d]);
                    qm.q8_0.q8RowScalesInto(q_scales[j * q8_blocks ..][0 .. task.d / q8_0_block_size], q_q8[j * q8_blocks ..][0 .. task.d / q8_0_block_size]);
                }
            }
            var max_score: [head_group]f32 = @splat(-std.math.inf(f32));
            if (comptime KvElem == BlockQ8_0) {
                // 2-key step: 2*head_group independent accumulator chains
                // hide the per-block fma latency the single-key dot
                // serializes on; K blocks are loaded once per step for the
                // whole group.
                const qm = quantized_matmul;
                const bpr = task.d / q8_0_block_size;
                const row_stride = kv_seq_stride / q8_0_block_size;
                const head_off = kv_head_i * kv_head_stride / q8_0_block_size;
                var source_i = lo;
                while (source_i + 2 <= active) : (source_i += 2) {
                    const k0 = task.k_data[source_i * row_stride + head_off ..][0..bpr];
                    const k1 = task.k_data[(source_i + 1) * row_stride + head_off ..][0..bpr];
                    // dots is 2-key x head_group, key-major.
                    const dots = if (head_group == 2)
                        qm.q8_0.vecDotQ8_0Q8_0Pairx2(q_q8[0..bpr], q_q8[q8_blocks..][0..bpr], q_scales[0..bpr], q_scales[q8_blocks..][0..bpr], k0, k1)
                    else
                        qm.q8_0.vecDotQ8_0Q8_0x2(q_q8[0..bpr], q_scales[0..bpr], k0, k1);
                    inline for (0..2) |i| {
                        inline for (0..head_group) |j| {
                            var score = dots[head_group * i + j] * task.scale_value;
                            if (bias_row) |row| score += row[source_i + i];
                            scores[j][source_i + i] = score;
                            max_score[j] = @max(max_score[j], score);
                        }
                    }
                }
                if (source_i < active) {
                    const k0 = task.k_data[source_i * row_stride + head_off ..][0..bpr];
                    const dots: [head_group]f32 = if (head_group == 2)
                        qm.q8_0.vecDotQ8_0Q8_0Pair(q_q8[0..bpr], q_q8[q8_blocks..][0..bpr], k0)
                    else
                        .{qm.q8_0.vecDotQ8_0Q8_0(q_q8[0..bpr], k0)};
                    inline for (0..head_group) |j| {
                        var score = dots[j] * task.scale_value;
                        if (bias_row) |row| score += row[source_i];
                        scores[j][source_i] = score;
                        max_score[j] = @max(max_score[j], score);
                    }
                }
            } else for (lo..active) |source_i| {
                var dot_value: [head_group]f32 = undefined;
                {
                    const k_row = task.k_data;
                    const k_base = source_i * kv_seq_stride + kv_head_i * kv_head_stride;
                    var dot_vec: [head_group]Vec = @splat(@splat(0));
                    var feature_i: usize = 0;
                    while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                        const kv: Vec = widenKvVec(Lane, vector_width, k_row, k_base + feature_i);
                        inline for (0..head_group) |j| {
                            const qv: Vec = task.q_data[q_base[j] + feature_i ..][0..vector_width].*;
                            dot_vec[j] += qv * kv;
                        }
                    }
                    inline for (0..head_group) |j| dot_value[j] = @reduce(.Add, dot_vec[j]);
                    while (feature_i < task.d) : (feature_i += 1) {
                        const k_value = widenKvScalar(Lane, k_row[k_base + feature_i]);
                        inline for (0..head_group) |j| dot_value[j] += task.q_data[q_base[j] + feature_i] * k_value;
                    }
                }
                inline for (0..head_group) |j| {
                    var score = dot_value[j] * task.scale_value;
                    if (bias_row) |row| score += row[source_i];
                    scores[j][source_i] = score;
                    max_score[j] = @max(max_score[j], score);
                }
            }

            var sum_exp: [head_group]f32 = @splat(0);
            for (lo..active) |source_i| {
                inline for (0..head_group) |j| {
                    const weight = @exp(scores[j][source_i] - max_score[j]);
                    scores[j][source_i] = weight;
                    sum_exp[j] += weight;
                }
            }
            var inv_sum: [head_group]f32 = undefined;
            inline for (0..head_group) |j| inv_sum[j] = 1 / sum_exp[j];
            if (task.stats) |stats| {
                inline for (0..head_group) |j| {
                    const stat_base = ((head_base + j) * task.q_seq + query_i) * 2;
                    stats[stat_base] = max_score[j];
                    stats[stat_base + 1] = sum_exp[j];
                }
            }

            var out_base: [head_group]usize = undefined;
            inline for (0..head_group) |j| out_base[j] = query_i * out_seq_stride + (head_base + j) * task.d;
            if (comptime KvElem == BlockQ8_0) {
                // 2-row fused V pass: the group's out rows load and store
                // once per two source rows.
                const qm = quantized_matmul;
                var out_rows: [head_group][]f32 = undefined;
                inline for (0..head_group) |j| out_rows[j] = task.out_data[out_base[j]..][0..task.d];
                const bpr = task.d / q8_0_block_size;
                const row_stride = kv_seq_stride / q8_0_block_size;
                const head_off = kv_head_i * kv_head_stride / q8_0_block_size;
                var source_i = lo;
                if (active - lo >= 2) {
                    const v0 = task.v_data[lo * row_stride + head_off ..][0..bpr];
                    const v1 = task.v_data[(lo + 1) * row_stride + head_off ..][0..bpr];
                    if (head_group == 2) {
                        qm.q8_0.weightedQ8_0RowPair2(false, out_rows[0], out_rows[1], v0, scores[0][lo] * inv_sum[0], scores[1][lo] * inv_sum[1], v1, scores[0][lo + 1] * inv_sum[0], scores[1][lo + 1] * inv_sum[1]);
                    } else {
                        qm.q8_0.weightedQ8_0Row2(false, out_rows[0], v0, scores[0][lo] * inv_sum[0], v1, scores[0][lo + 1] * inv_sum[0]);
                    }
                    source_i = lo + 2;
                } else {
                    const v0 = task.v_data[lo * row_stride + head_off ..][0..bpr];
                    if (head_group == 2) {
                        qm.q8_0.weightedQ8_0RowPair(false, out_rows[0], out_rows[1], v0, scores[0][lo] * inv_sum[0], scores[1][lo] * inv_sum[1]);
                    } else {
                        qm.q8_0.weightedQ8_0Row(false, out_rows[0], v0, scores[0][lo] * inv_sum[0]);
                    }
                    source_i = lo + 1;
                }
                while (source_i + 2 <= active) : (source_i += 2) {
                    const v0 = task.v_data[source_i * row_stride + head_off ..][0..bpr];
                    const v1 = task.v_data[(source_i + 1) * row_stride + head_off ..][0..bpr];
                    if (head_group == 2) {
                        qm.q8_0.weightedQ8_0RowPair2(true, out_rows[0], out_rows[1], v0, scores[0][source_i] * inv_sum[0], scores[1][source_i] * inv_sum[1], v1, scores[0][source_i + 1] * inv_sum[0], scores[1][source_i + 1] * inv_sum[1]);
                    } else {
                        qm.q8_0.weightedQ8_0Row2(true, out_rows[0], v0, scores[0][source_i] * inv_sum[0], v1, scores[0][source_i + 1] * inv_sum[0]);
                    }
                }
                if (source_i < active) {
                    const v0 = task.v_data[source_i * row_stride + head_off ..][0..bpr];
                    if (head_group == 2) {
                        qm.q8_0.weightedQ8_0RowPair(true, out_rows[0], out_rows[1], v0, scores[0][source_i] * inv_sum[0], scores[1][source_i] * inv_sum[1]);
                    } else {
                        qm.q8_0.weightedQ8_0Row(true, out_rows[0], v0, scores[0][source_i] * inv_sum[0]);
                    }
                }
            } else {
                {
                    var weight: [head_group]f32 = undefined;
                    var weight_vec: [head_group]Vec = undefined;
                    inline for (0..head_group) |j| {
                        weight[j] = scores[j][lo] * inv_sum[j];
                        weight_vec[j] = @splat(weight[j]);
                    }
                    const v_row = task.v_data;
                    const v_base = lo * kv_seq_stride + kv_head_i * kv_head_stride;
                    var feature_i: usize = 0;
                    while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                        const v_vec: Vec = widenKvVec(Lane, vector_width, v_row, v_base + feature_i);
                        inline for (0..head_group) |j| {
                            task.out_data[out_base[j] + feature_i ..][0..vector_width].* = weight_vec[j] * v_vec;
                        }
                    }
                    while (feature_i < task.d) : (feature_i += 1) {
                        const v_value = widenKvScalar(Lane, v_row[v_base + feature_i]);
                        inline for (0..head_group) |j| {
                            task.out_data[out_base[j] + feature_i] = weight[j] * v_value;
                        }
                    }
                }
                for (lo + 1..active) |source_i| {
                    var weight: [head_group]f32 = undefined;
                    var weight_vec: [head_group]Vec = undefined;
                    inline for (0..head_group) |j| {
                        weight[j] = scores[j][source_i] * inv_sum[j];
                        weight_vec[j] = @splat(weight[j]);
                    }
                    const v_row = task.v_data;
                    const v_base = source_i * kv_seq_stride + kv_head_i * kv_head_stride;
                    var feature_i: usize = 0;
                    while (feature_i + vector_width <= task.d) : (feature_i += vector_width) {
                        const v_vec: Vec = widenKvVec(Lane, vector_width, v_row, v_base + feature_i);
                        inline for (0..head_group) |j| {
                            const current: Vec = task.out_data[out_base[j] + feature_i ..][0..vector_width].*;
                            task.out_data[out_base[j] + feature_i ..][0..vector_width].* = current + weight_vec[j] * v_vec;
                        }
                    }
                    while (feature_i < task.d) : (feature_i += 1) {
                        const v_value = widenKvScalar(Lane, v_row[v_base + feature_i]);
                        inline for (0..head_group) |j| {
                            task.out_data[out_base[j] + feature_i] += weight[j] * v_value;
                        }
                    }
                }
            }
        }
    }
}

/// General per-query kernel: one query head per work unit, arbitrary
/// `kv_head_for_head` mapping (see `groupedCausalAttentionUnits`).
pub fn groupedCausalAttentionHeads(comptime KvElem: type, task: GroupedCausalAttentionTask(KvElem)) void {
    // The q8_0 KV arm is reference-by-composition: its integer dot and
    // weighted-row kernels select their own scalar tiers.
    if (comptime isa.reference and KvElem != BlockQ8_0) return scalar.groupedCausalAttentionHeads(KvElem, task);
    groupedCausalAttentionUnits(KvElem, 1, task);
}

/// Adjacent-pair GQA per-query kernel: TWO query heads sharing one KV head
/// per work unit (see `groupedCausalAttentionUnits`).
pub fn groupedCausalAttentionHeadPairs(comptime KvElem: type, task: GroupedCausalAttentionPairTask(KvElem)) void {
    // See groupedCausalAttentionHeads on the q8_0 arm.
    if (comptime isa.reference and KvElem != BlockQ8_0) return scalar.groupedCausalAttentionHeadPairs(KvElem, task);
    groupedCausalAttentionUnits(KvElem, 2, task);
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
    if (comptime isa.reference) return scalar.groupedCausalAttentionQueryTiles(KvElem, head_group, task);
    const rows = attention_tile_rows;
    const q_tile = rows / head_group;
    const kb = attention_key_block;
    // Score pass: rows x kb dot accumulators must stay in registers, so the
    // d-chunk is one NEON register wide. Value pass: the accumulator rows are
    // load/store traffic either way, so a doubled chunk halves loop overhead.
    const DotVec = @Vector(4, f32);
    const dot_width = 4;
    const Vec = common.RowVec;
    const vector_width = common.row_lanes;
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

pub const groupedCausalAttentionBackwardKvHeads = if (isa.reference) scalar.groupedCausalAttentionBackwardKvHeads else nativeGroupedCausalAttentionBackwardKvHeads;
fn nativeGroupedCausalAttentionBackwardKvHeads(task: GroupedCausalAttentionBackwardTask) void {
    const q_head_stride = task.d;
    const q_seq_stride = task.heads * task.d;
    const kv_head_stride = task.d;
    const kv_seq_stride = task.kv_heads * task.d;
    const out_seq_stride = task.heads * task.d;
    const Vec = common.RowVec;
    const vector_width = common.row_lanes;

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

/// Phase 2 of the tiled backward, shared by the register-tiled and the
/// BLAS-strip routes: per row of the tile, probabilities in place (stats or
/// recompute), the softmax-backward row dot, then dS in place. Dead cells
/// (outside the row's causal/window span) are zeroed in BOTH panels, so the
/// gradient contractions that follow need no masks.
inline fn attentionBackwardSoftmaxTileRows(
    task: *const GroupedCausalAttentionBackwardTiledTask,
    q0: usize,
    rows_active: usize,
    tile_lo: usize,
    live: usize,
    p_panel_all: []f32,
    ds_panel_all: []f32,
    stats_head: ?[]const f32,
) void {
    const Vec = common.RowVec;
    const vector_width = common.row_lanes;
    const neg_inf = -std.math.inf(f32);

    for (0..rows_active) |r| {
        const query_i = q0 + r;
        const active = if (task.causal) task.source_offset + query_i + 1 else task.kv_seq;
        const lo_r = if (!task.causal or task.window == 0) 0 else active -| task.window;
        const c_lo = lo_r - tile_lo;
        const c_hi = active - tile_lo;
        const p_row = p_panel_all[r * task.kv_seq ..][0..live];
        const ds_row = ds_panel_all[r * task.kv_seq ..][0..live];

        var max_score: f32 = neg_inf;
        var inv_sum: f32 = undefined;
        if (stats_head) |values| {
            max_score = values[query_i * 2];
            inv_sum = 1 / values[query_i * 2 + 1];
        } else {
            var max_vec: Vec = @splat(neg_inf);
            var column = c_lo;
            while (column + vector_width <= c_hi) : (column += vector_width) {
                max_vec = @max(max_vec, @as(Vec, p_row[column..][0..vector_width].*));
            }
            max_score = @reduce(.Max, max_vec);
            while (column < c_hi) : (column += 1) max_score = @max(max_score, p_row[column]);
        }

        const max_splat: Vec = @splat(max_score);
        var sum_vec: Vec = @splat(0);
        var column = c_lo;
        while (column + vector_width <= c_hi) : (column += vector_width) {
            const p = vexpf(vector_width, @as(Vec, p_row[column..][0..vector_width].*) - max_splat);
            p_row[column..][0..vector_width].* = p;
            sum_vec += p;
        }
        var sum_exp = @reduce(.Add, sum_vec);
        while (column < c_hi) : (column += 1) {
            const p = vexpf(1, @splat(p_row[column] - max_score))[0];
            p_row[column] = p;
            sum_exp += p;
        }
        if (stats_head == null) inv_sum = 1 / sum_exp;

        @memset(p_row[0..c_lo], 0);
        @memset(p_row[c_hi..live], 0);

        const inv_splat: Vec = @splat(inv_sum);
        var dot_vec: Vec = @splat(0);
        column = c_lo;
        while (column + vector_width <= c_hi) : (column += vector_width) {
            const p = @as(Vec, p_row[column..][0..vector_width].*) * inv_splat;
            p_row[column..][0..vector_width].* = p;
            dot_vec += p * @as(Vec, ds_row[column..][0..vector_width].*);
        }
        var row_dot = @reduce(.Add, dot_vec);
        while (column < c_hi) : (column += 1) {
            const p = p_row[column] * inv_sum;
            p_row[column] = p;
            row_dot += p * ds_row[column];
        }

        const dot_splat: Vec = @splat(row_dot);
        const scale_splat: Vec = @splat(task.scale_value);
        @memset(ds_row[0..c_lo], 0);
        @memset(ds_row[c_hi..live], 0);
        column = c_lo;
        while (column + vector_width <= c_hi) : (column += vector_width) {
            const p: Vec = p_row[column..][0..vector_width].*;
            const dp: Vec = ds_row[column..][0..vector_width].*;
            ds_row[column..][0..vector_width].* = scale_splat * p * (dp - dot_splat);
        }
        while (column < c_hi) : (column += 1) {
            ds_row[column] = task.scale_value * p_row[column] * (ds_row[column] - row_dot);
        }
    }
}

/// Fused tile-blocked attention backward: per (query head, q-tile) the
/// scaled-score and dO·Vᵀ panels are built once over the tile's LIVE key
/// range only (causal/window dead cells never compute beyond the sub-block
/// fringe), the probability/dS rows are formed in place, and all three
/// gradient contractions read the cache-resident panels. Work splits by
/// whole query heads: dQ rows have a single writer, and dK/dV go to
/// per-head planes (or straight to the gradient in direct mode), so the
/// result is bitwise identical for any task count.
pub const groupedCausalAttentionBackwardTiles = if (isa.reference) scalar.groupedCausalAttentionBackwardTiles else nativeGroupedCausalAttentionBackwardTiles;
fn nativeGroupedCausalAttentionBackwardTiles(task: GroupedCausalAttentionBackwardTiledTask) void {
    const tile_rows = attention_bwd_tile_rows;
    const kb = attention_key_block;
    const qr_block = 4;
    const DotVec = @Vector(4, f32);
    const dot_width = 4;
    const Vec = common.RowVec;
    const vector_width = common.row_lanes;

    const d = task.d;
    const q_seq_stride = task.heads * d;
    const kv_seq_stride = task.kv_heads * d;
    const p_panel_all = task.scratch[0 .. tile_rows * task.kv_seq];
    const ds_panel_all = task.scratch[tile_rows * task.kv_seq ..][0 .. tile_rows * task.kv_seq];

    for (task.head_start..task.head_end) |head_i| {
        const kv_head_i = task.kv_head_for_head[head_i];
        const kv_head_base = kv_head_i * d;
        const acc_base = if (task.partial_mode) head_i * task.kv_seq * d else kv_head_base;
        const acc_stride = if (task.partial_mode) d else kv_seq_stride;
        const stats_head: ?[]const f32 = if (task.stats) |values| values[head_i * task.q_seq * 2 ..][0 .. task.q_seq * 2] else null;

        var q0: usize = 0;
        while (q0 < task.q_seq) : (q0 += tile_rows) {
            const rows_active = @min(tile_rows, task.q_seq - q0);
            const p_first = task.source_offset + q0;
            const tile_hi = if (task.causal) @min(p_first + rows_active, task.kv_seq) else task.kv_seq;
            const tile_lo = if (!task.causal or task.window == 0) 0 else (p_first + 1) -| task.window;
            const live = tile_hi - tile_lo;

            // Phase 1: scaled scores and dO·Vᵀ into the panels, kb keys x
            // qr_block queries per microkernel step (the forward's
            // outer-product shape). Tail keys/rows clamp to valid indices;
            // clamped keys skip their store, clamped rows are never stored.
            var block_start = tile_lo;
            while (block_start < tile_hi) : (block_start += kb) {
                var k_base: [kb]usize = undefined;
                inline for (0..kb) |ki| {
                    const source_i = @min(block_start + ki, tile_hi - 1);
                    k_base[ki] = source_i * kv_seq_stride + kv_head_base;
                }

                var r0: usize = 0;
                while (r0 < rows_active) : (r0 += qr_block) {
                    const qr_active = @min(qr_block, rows_active - r0);
                    var q_base: [qr_block]usize = undefined;
                    inline for (0..qr_block) |r| {
                        q_base[r] = (q0 + @min(r0 + r, rows_active - 1)) * q_seq_stride + head_i * d;
                    }

                    // Two separate kb x qr_block microkernels — fusing the
                    // score and dO·Vᵀ accumulators into one pass needs 32
                    // dot registers and spills; K/V rows re-read from L1.
                    var score_arr: [kb][qr_block]f32 = undefined;
                    {
                        var dot: [kb][qr_block]DotVec = undefined;
                        inline for (0..kb) |ki| inline for (0..qr_block) |r| {
                            dot[ki][r] = @splat(0);
                        };
                        var feature_i: usize = 0;
                        while (feature_i + dot_width <= d) : (feature_i += dot_width) {
                            var k_vec: [kb]DotVec = undefined;
                            inline for (0..kb) |ki| k_vec[ki] = task.k_data[k_base[ki] + feature_i ..][0..dot_width].*;
                            inline for (0..qr_block) |r| {
                                const qv: DotVec = task.q_data[q_base[r] + feature_i ..][0..dot_width].*;
                                inline for (0..kb) |ki| dot[ki][r] = @mulAdd(DotVec, qv, k_vec[ki], dot[ki][r]);
                            }
                        }
                        inline for (0..kb) |ki| inline for (0..qr_block) |r| {
                            score_arr[ki][r] = @reduce(.Add, dot[ki][r]);
                        };
                        while (feature_i < d) : (feature_i += 1) {
                            inline for (0..kb) |ki| {
                                const k_value = task.k_data[k_base[ki] + feature_i];
                                inline for (0..qr_block) |r| score_arr[ki][r] += task.q_data[q_base[r] + feature_i] * k_value;
                            }
                        }
                    }
                    var dp_arr: [kb][qr_block]f32 = undefined;
                    {
                        var dot: [kb][qr_block]DotVec = undefined;
                        inline for (0..kb) |ki| inline for (0..qr_block) |r| {
                            dot[ki][r] = @splat(0);
                        };
                        var feature_i: usize = 0;
                        while (feature_i + dot_width <= d) : (feature_i += dot_width) {
                            var v_vec: [kb]DotVec = undefined;
                            inline for (0..kb) |ki| v_vec[ki] = task.v_data[k_base[ki] + feature_i ..][0..dot_width].*;
                            inline for (0..qr_block) |r| {
                                const gyv: DotVec = task.gy_data[q_base[r] + feature_i ..][0..dot_width].*;
                                inline for (0..kb) |ki| dot[ki][r] = @mulAdd(DotVec, gyv, v_vec[ki], dot[ki][r]);
                            }
                        }
                        inline for (0..kb) |ki| inline for (0..qr_block) |r| {
                            dp_arr[ki][r] = @reduce(.Add, dot[ki][r]);
                        };
                        while (feature_i < d) : (feature_i += 1) {
                            inline for (0..kb) |ki| {
                                const v_value = task.v_data[k_base[ki] + feature_i];
                                inline for (0..qr_block) |r| dp_arr[ki][r] += task.gy_data[q_base[r] + feature_i] * v_value;
                            }
                        }
                    }

                    inline for (0..kb) |ki| {
                        if (block_start + ki < tile_hi) {
                            const column = block_start + ki - tile_lo;
                            for (0..qr_active) |r| {
                                const panel_row = (r0 + r) * task.kv_seq;
                                p_panel_all[panel_row + column] = score_arr[ki][r] * task.scale_value;
                                ds_panel_all[panel_row + column] = dp_arr[ki][r];
                            }
                        }
                    }
                }
            }

            // Phase 2 per row: probabilities in place (stats or recompute),
            // the softmax-backward row dot, then dS in place. Dead cells hold
            // P = 0, so their dS is exactly 0 and phase 3 needs no masks.
            attentionBackwardSoftmaxTileRows(&task, q0, rows_active, tile_lo, live, p_panel_all, ds_panel_all, stats_head);

            // Phase 3a: dQ — 4-row register kernel per d-chunk; each K row
            // vector is loaded once and shared by the 4 dS splats. Single
            // writer per (head, query) row.
            if (task.q_grad) |grad| {
                var r0: usize = 0;
                while (r0 < rows_active) : (r0 += qr_block) {
                    const qr_active = @min(qr_block, rows_active - r0);
                    var panel_rows: [qr_block]usize = undefined;
                    inline for (0..qr_block) |r| {
                        panel_rows[r] = @min(r0 + r, rows_active - 1) * task.kv_seq;
                    }
                    var feature_i: usize = 0;
                    while (feature_i + vector_width <= d) : (feature_i += vector_width) {
                        var acc: [qr_block]Vec = undefined;
                        inline for (0..qr_block) |r| acc[r] = @splat(0);
                        for (0..live) |column| {
                            const k_vec: Vec = task.k_data[(tile_lo + column) * kv_seq_stride + kv_head_base + feature_i ..][0..vector_width].*;
                            inline for (0..qr_block) |r| {
                                acc[r] = @mulAdd(Vec, @as(Vec, @splat(ds_panel_all[panel_rows[r] + column])), k_vec, acc[r]);
                            }
                        }
                        for (0..qr_active) |r| {
                            grad[(q0 + r0 + r) * q_seq_stride + head_i * d + feature_i ..][0..vector_width].* = acc[r];
                        }
                    }
                    while (feature_i < d) : (feature_i += 1) {
                        for (0..qr_active) |r| {
                            var tail: f32 = 0;
                            const ds_row = ds_panel_all[(r0 + r) * task.kv_seq ..][0..live];
                            for (0..live) |column| {
                                tail += ds_row[column] * task.k_data[(tile_lo + column) * kv_seq_stride + kv_head_base + feature_i];
                            }
                            grad[(q0 + r0 + r) * q_seq_stride + head_i * d + feature_i] = tail;
                        }
                    }
                }
            }

            // Phase 3b: dV/dK — 4-column register kernel per d-chunk, dV and
            // dK fused so each gy/q row vector is loaded once. Accumulators
            // live in registers for the whole row sweep; the panel splats
            // read 4 contiguous scalars per row.
            const col_block = 4;
            var c0: usize = 0;
            while (c0 + col_block <= live) : (c0 += col_block) {
                var feature_i: usize = 0;
                while (feature_i + vector_width <= d) : (feature_i += vector_width) {
                    var dv_acc: [col_block]Vec = undefined;
                    var dk_acc: [col_block]Vec = undefined;
                    inline for (0..col_block) |ci| {
                        dv_acc[ci] = @splat(0);
                        dk_acc[ci] = @splat(0);
                    }
                    for (0..rows_active) |r| {
                        const row_base = (q0 + r) * q_seq_stride + head_i * d + feature_i;
                        const panel_row = r * task.kv_seq + c0;
                        if (task.dv_target != null) {
                            const gy_vec: Vec = task.gy_data[row_base..][0..vector_width].*;
                            inline for (0..col_block) |ci| {
                                dv_acc[ci] = @mulAdd(Vec, @as(Vec, @splat(p_panel_all[panel_row + ci])), gy_vec, dv_acc[ci]);
                            }
                        }
                        if (task.dk_target != null) {
                            const q_vec: Vec = task.q_data[row_base..][0..vector_width].*;
                            inline for (0..col_block) |ci| {
                                dk_acc[ci] = @mulAdd(Vec, @as(Vec, @splat(ds_panel_all[panel_row + ci])), q_vec, dk_acc[ci]);
                            }
                        }
                    }
                    inline for (0..col_block) |ci| {
                        const target_row = acc_base + (tile_lo + c0 + ci) * acc_stride + feature_i;
                        if (task.dv_target) |target| {
                            target[target_row..][0..vector_width].* = @as(Vec, target[target_row..][0..vector_width].*) + dv_acc[ci];
                        }
                        if (task.dk_target) |target| {
                            target[target_row..][0..vector_width].* = @as(Vec, target[target_row..][0..vector_width].*) + dk_acc[ci];
                        }
                    }
                }
                while (feature_i < d) : (feature_i += 1) {
                    inline for (0..col_block) |ci| {
                        const target_row = acc_base + (tile_lo + c0 + ci) * acc_stride + feature_i;
                        var dv_tail: f32 = 0;
                        var dk_tail: f32 = 0;
                        for (0..rows_active) |r| {
                            const value = task.q_data[(q0 + r) * q_seq_stride + head_i * d + feature_i];
                            const gy_value = task.gy_data[(q0 + r) * q_seq_stride + head_i * d + feature_i];
                            dv_tail += p_panel_all[r * task.kv_seq + c0 + ci] * gy_value;
                            dk_tail += ds_panel_all[r * task.kv_seq + c0 + ci] * value;
                        }
                        if (task.dv_target) |target| target[target_row] += dv_tail;
                        if (task.dk_target) |target| target[target_row] += dk_tail;
                    }
                }
            }
            // Column tail: the last live % 4 keys, one column at a time.
            while (c0 < live) : (c0 += 1) {
                const target_row_base = acc_base + (tile_lo + c0) * acc_stride;
                var feature_i: usize = 0;
                while (feature_i + vector_width <= d) : (feature_i += vector_width) {
                    var dv_acc: Vec = @splat(0);
                    var dk_acc: Vec = @splat(0);
                    for (0..rows_active) |r| {
                        const row_base = (q0 + r) * q_seq_stride + head_i * d + feature_i;
                        const panel_row = r * task.kv_seq + c0;
                        if (task.dv_target != null) {
                            dv_acc = @mulAdd(Vec, @as(Vec, @splat(p_panel_all[panel_row])), @as(Vec, task.gy_data[row_base..][0..vector_width].*), dv_acc);
                        }
                        if (task.dk_target != null) {
                            dk_acc = @mulAdd(Vec, @as(Vec, @splat(ds_panel_all[panel_row])), @as(Vec, task.q_data[row_base..][0..vector_width].*), dk_acc);
                        }
                    }
                    if (task.dv_target) |target| {
                        target[target_row_base + feature_i ..][0..vector_width].* = @as(Vec, target[target_row_base + feature_i ..][0..vector_width].*) + dv_acc;
                    }
                    if (task.dk_target) |target| {
                        target[target_row_base + feature_i ..][0..vector_width].* = @as(Vec, target[target_row_base + feature_i ..][0..vector_width].*) + dk_acc;
                    }
                }
                while (feature_i < d) : (feature_i += 1) {
                    var dv_tail: f32 = 0;
                    var dk_tail: f32 = 0;
                    for (0..rows_active) |r| {
                        const base_idx = (q0 + r) * q_seq_stride + head_i * d + feature_i;
                        dv_tail += p_panel_all[r * task.kv_seq + c0] * task.gy_data[base_idx];
                        dk_tail += ds_panel_all[r * task.kv_seq + c0] * task.q_data[base_idx];
                    }
                    if (task.dv_target) |target| target[target_row_base + feature_i] += dv_tail;
                    if (task.dk_target) |target| target[target_row_base + feature_i] += dk_tail;
                }
            }
        }
    }
}

/// BLAS-strip attention backward: the register-tiled route's tile walk and
/// task contract, with the five per-tile contractions issued as strided
/// sgemm strips — the scaled-score and dO·Vᵀ panels (nt), dQ (nn), and the
/// dK/dV accumulations (tn, beta=1) — so the contraction FLOPs run on the
/// BLAS engine while the softmax/dS rows stay on the vector units (shared
/// phase 2, which zeroes every dead cell the full-rectangle strips filled).
/// Same head split and single-writer discipline as the tiled route; results
/// are deterministic for any task count but differ from the register route
/// in the last ulps (different contraction associations).
pub fn groupedCausalAttentionBackwardBlasTiles(task: GroupedCausalAttentionBackwardTiledTask) void {
    if (comptime !blas_available) {
        return groupedCausalAttentionBackwardTiles(task);
    } else {
        const sgemm = blas.gemmStrided;
        // One task per head range, so this scope covers every strip sgemm this
        // worker issues; without it a self-threading BLAS starts an engine team
        // per call inside our own parallel region. The token restores whatever
        // this thread had.
        const blas_scope = blas.beginNestedScope();
        defer blas.endNestedScope(blas_scope);
        const tile_rows = attention_bwd_blas_tile_rows;
        const d = task.d;
        const q_seq_stride = task.heads * d;
        const kv_seq_stride = task.kv_heads * d;
        const p_panel_all = task.scratch[0 .. tile_rows * task.kv_seq];
        const ds_panel_all = task.scratch[tile_rows * task.kv_seq ..][0 .. tile_rows * task.kv_seq];

        for (task.head_start..task.head_end) |head_i| {
            const kv_head_i = task.kv_head_for_head[head_i];
            const kv_head_base = kv_head_i * d;
            const acc_base = if (task.partial_mode) head_i * task.kv_seq * d else kv_head_base;
            const acc_stride = if (task.partial_mode) d else kv_seq_stride;
            const stats_head: ?[]const f32 = if (task.stats) |values| values[head_i * task.q_seq * 2 ..][0 .. task.q_seq * 2] else null;

            var q0: usize = 0;
            while (q0 < task.q_seq) : (q0 += tile_rows) {
                const rows_active = @min(tile_rows, task.q_seq - q0);
                const p_first = task.source_offset + q0;
                const tile_hi = if (task.causal) @min(p_first + rows_active, task.kv_seq) else task.kv_seq;
                const tile_lo = if (!task.causal or task.window == 0) 0 else (p_first + 1) -| task.window;
                if (tile_hi <= tile_lo) continue;
                const live = tile_hi - tile_lo;

                const q_tile = task.q_data[q0 * q_seq_stride + head_i * d ..];
                const gy_tile = task.gy_data[q0 * q_seq_stride + head_i * d ..];
                const k_live = task.k_data[tile_lo * kv_seq_stride + kv_head_base ..];
                const v_live = task.v_data[tile_lo * kv_seq_stride + kv_head_base ..];

                sgemm(false, true, rows_active, live, d, task.scale_value, q_tile, q_seq_stride, k_live, kv_seq_stride, 0.0, p_panel_all, task.kv_seq);
                sgemm(false, true, rows_active, live, d, 1.0, gy_tile, q_seq_stride, v_live, kv_seq_stride, 0.0, ds_panel_all, task.kv_seq);

                attentionBackwardSoftmaxTileRows(&task, q0, rows_active, tile_lo, live, p_panel_all, ds_panel_all, stats_head);

                if (task.q_grad) |grad| {
                    sgemm(false, false, rows_active, d, live, 1.0, ds_panel_all, task.kv_seq, k_live, kv_seq_stride, 0.0, grad[q0 * q_seq_stride + head_i * d ..], q_seq_stride);
                }
                if (task.dv_target) |target| {
                    sgemm(true, false, live, d, rows_active, 1.0, p_panel_all, task.kv_seq, gy_tile, q_seq_stride, 1.0, target[acc_base + tile_lo * acc_stride ..], acc_stride);
                }
                if (task.dk_target) |target| {
                    sgemm(true, false, live, d, rows_active, 1.0, ds_panel_all, task.kv_seq, q_tile, q_seq_stride, 1.0, target[acc_base + tile_lo * acc_stride ..], acc_stride);
                }
            }
        }
    }
}

pub fn attentionBackwardReduceRows(task: AttentionBackwardReduceTask, source_start: usize, source_end: usize) void {
    for (source_start..source_end) |source_i| {
        for (task.kv_head_for_head, 0..) |kv_head_i, head_i| {
            const grad_row = task.grad[source_i * task.kv_heads * task.d + kv_head_i * task.d ..][0..task.d];
            const partial_row = task.partials[head_i * task.kv_seq * task.d + source_i * task.d ..][0..task.d];
            addSliceInPlace(grad_row, partial_row);
        }
    }
}

pub fn addSliceInPlace(dest: []f32, src: []const f32) void {
    std.debug.assert(dest.len == src.len);
    // The reference arm: the plain serial accumulate (same per-element
    // order as the vector body, so this is a spelling, not a numeric fork).
    if (comptime isa.reference) {
        for (dest, src) |*d, s| d.* += s;
        return;
    }
    const Vec = common.RowVec;
    const vector_width = common.row_lanes;
    var i: usize = 0;
    while (i + vector_width <= src.len) : (i += vector_width) {
        const dest_vec: Vec = dest[i..][0..vector_width].*;
        const src_vec: Vec = src[i..][0..vector_width].*;
        dest[i..][0..vector_width].* = dest_vec + src_vec;
    }
    while (i < src.len) : (i += 1) dest[i] += src[i];
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

/// The scalar reference twins of the attention entries: serial three-pass
/// per-(head, query) loops — no `@Vector`, no tiles, no online softmax.
/// On `-Dbackend=scalar` builds the f32/f16 forward entries and both
/// backward routes dispatch here (the q8_0 per-query arms compose the
/// quant kernels' own scalar tiers instead; the tiled twin dequantizes
/// q8_0 rows serially); on native builds the twins stay reachable for
/// `backend/parity_test.zig`. The per-query forward twin matches the
/// per-query entries' per-element arithmetic (`@exp`, first-write V
/// accumulate) and differs in reduction order only; the tiled entries
/// state a ~1e-6 relative summation-order class against exactly this
/// serial reference.
pub const scalar = struct {
    /// One (head, query) row: serial dot scores over [lo, active), @exp
    /// softmax with running max, optional stats write, serial weighted V.
    fn forwardRow(
        comptime KvElem: type,
        q_row: []const f32,
        k_data: []const KvElem,
        v_data: []const KvElem,
        out_row: []f32,
        scores: []f32,
        kv_head_i: usize,
        kv_heads: usize,
        d: usize,
        lo: usize,
        active: usize,
        scale_value: f32,
        bias_row: ?[]const f32,
        stats_slot: ?[]f32,
    ) void {
        const kv_seq_stride = kv_heads * d;
        const Lane = KvLane(KvElem);
        var q8_row: [attention_tile_max_d]f32 = undefined;

        var max_score = -std.math.inf(f32);
        for (lo..active) |source_i| {
            const elem_base = source_i * kv_seq_stride + kv_head_i * d;
            const k_row, const k_off = kvRowSelect(KvElem, k_data, elem_base, d, &q8_row);
            var dot_value: f32 = 0;
            for (0..d) |feature_i| {
                dot_value += q_row[feature_i] * widenKvScalar(Lane, k_row[k_off + feature_i]);
            }
            var score = dot_value * scale_value;
            if (bias_row) |row| score += row[source_i];
            scores[source_i] = score;
            max_score = @max(max_score, score);
        }

        var sum_exp: f32 = 0;
        for (lo..active) |source_i| {
            const weight = @exp(scores[source_i] - max_score);
            scores[source_i] = weight;
            sum_exp += weight;
        }
        const inv_sum = 1 / sum_exp;
        if (stats_slot) |slot| {
            slot[0] = max_score;
            slot[1] = sum_exp;
        }

        for (lo..active) |source_i| {
            const elem_base = source_i * kv_seq_stride + kv_head_i * d;
            const v_row, const v_off = kvRowSelect(KvElem, v_data, elem_base, d, &q8_row);
            const weight = scores[source_i] * inv_sum;
            if (source_i == lo) {
                for (0..d) |feature_i| {
                    out_row[feature_i] = weight * widenKvScalar(Lane, v_row[v_off + feature_i]);
                }
            } else {
                for (0..d) |feature_i| {
                    out_row[feature_i] += weight * widenKvScalar(Lane, v_row[v_off + feature_i]);
                }
            }
        }
    }

    pub fn groupedCausalAttentionHeads(comptime KvElem: type, task: GroupedCausalAttentionTask(KvElem)) void {
        for (task.head_start..task.head_end) |head_i| {
            const kv_head_i = task.kv_head_for_head[head_i];
            for (0..task.q_seq) |query_i| {
                const active = if (task.causal) task.source_offset + query_i + 1 else task.kv_seq;
                const lo = if (!task.causal or task.window == 0) 0 else active -| task.window;
                forwardRow(
                    KvElem,
                    task.q_data[query_i * task.heads * task.d + head_i * task.d ..][0..task.d],
                    task.k_data,
                    task.v_data,
                    task.out_data[query_i * task.heads * task.d + head_i * task.d ..][0..task.d],
                    task.scores[0..task.kv_seq],
                    kv_head_i,
                    task.kv_heads,
                    task.d,
                    lo,
                    active,
                    task.scale_value,
                    if (task.bias) |bias_data| bias_data[query_i * task.kv_seq ..][0..task.kv_seq] else null,
                    if (task.stats) |stats| stats[(head_i * task.q_seq + query_i) * 2 ..][0..2] else null,
                );
            }
        }
    }

    pub fn groupedCausalAttentionHeadPairs(comptime KvElem: type, task: GroupedCausalAttentionPairTask(KvElem)) void {
        for (task.kv_head_start..task.kv_head_end) |kv_head_i| {
            for (0..2) |j| {
                const head_i = kv_head_i * 2 + j;
                for (0..task.q_seq) |query_i| {
                    const active = if (task.causal) task.source_offset + query_i + 1 else task.kv_seq;
                    const lo = if (!task.causal or task.window == 0) 0 else active -| task.window;
                    forwardRow(
                        KvElem,
                        task.q_data[query_i * task.heads * task.d + head_i * task.d ..][0..task.d],
                        task.k_data,
                        task.v_data,
                        task.out_data[query_i * task.heads * task.d + head_i * task.d ..][0..task.d],
                        task.scores[0..task.kv_seq],
                        kv_head_i,
                        task.kv_heads,
                        task.d,
                        lo,
                        active,
                        task.scale_value,
                        if (task.bias) |bias_data| bias_data[query_i * task.kv_seq ..][0..task.kv_seq] else null,
                        if (task.stats) |stats| stats[(head_i * task.q_seq + query_i) * 2 ..][0..2] else null,
                    );
                }
            }
        }
    }

    /// One (head, query) row without score scratch (the tiled entry is
    /// allocation-free): pass A finds the max, pass B recomputes each
    /// score, accumulates sum and the weighted V row unnormalized, pass C
    /// scales by 1/sum. Serial three-pass semantics; the dot is computed
    /// twice per key.
    fn forwardRowNoScratch(
        comptime KvElem: type,
        q_row: []const f32,
        k_data: []const KvElem,
        v_data: []const KvElem,
        out_row: []f32,
        kv_head_i: usize,
        kv_heads: usize,
        d: usize,
        lo: usize,
        active: usize,
        scale_value: f32,
        bias_row: ?[]const f32,
        stats_slot: ?[]f32,
    ) void {
        const kv_seq_stride = kv_heads * d;
        const Lane = KvLane(KvElem);
        var q8_row: [attention_tile_max_d]f32 = undefined;

        var max_score = -std.math.inf(f32);
        for (lo..active) |source_i| {
            const elem_base = source_i * kv_seq_stride + kv_head_i * d;
            const k_row, const k_off = kvRowSelect(KvElem, k_data, elem_base, d, &q8_row);
            var dot_value: f32 = 0;
            for (0..d) |feature_i| {
                dot_value += q_row[feature_i] * widenKvScalar(Lane, k_row[k_off + feature_i]);
            }
            var score = dot_value * scale_value;
            if (bias_row) |row| score += row[source_i];
            max_score = @max(max_score, score);
        }

        var sum_exp: f32 = 0;
        for (lo..active) |source_i| {
            const elem_base = source_i * kv_seq_stride + kv_head_i * d;
            const k_row, const k_off = kvRowSelect(KvElem, k_data, elem_base, d, &q8_row);
            var dot_value: f32 = 0;
            for (0..d) |feature_i| {
                dot_value += q_row[feature_i] * widenKvScalar(Lane, k_row[k_off + feature_i]);
            }
            var score = dot_value * scale_value;
            if (bias_row) |row| score += row[source_i];
            const weight = @exp(score - max_score);
            sum_exp += weight;
            const v_row, const v_off = kvRowSelect(KvElem, v_data, elem_base, d, &q8_row);
            if (source_i == lo) {
                for (0..d) |feature_i| {
                    out_row[feature_i] = weight * widenKvScalar(Lane, v_row[v_off + feature_i]);
                }
            } else {
                for (0..d) |feature_i| {
                    out_row[feature_i] += weight * widenKvScalar(Lane, v_row[v_off + feature_i]);
                }
            }
        }

        const inv_sum = 1 / sum_exp;
        for (0..d) |feature_i| out_row[feature_i] *= inv_sum;
        if (stats_slot) |slot| {
            slot[0] = max_score;
            slot[1] = sum_exp;
        }
    }

    pub fn groupedCausalAttentionQueryTiles(
        comptime KvElem: type,
        comptime head_group: usize,
        task: GroupedCausalAttentionTiledTask(KvElem),
    ) void {
        const q_tile = attention_tile_rows / head_group;
        for (task.work_start..task.work_end) |work_i| {
            const head_unit = work_i / task.n_tiles;
            const tile_i = work_i % task.n_tiles;
            const kv_head_i = if (head_group == 2) head_unit else task.kv_head_for_head[head_unit];
            const q0 = tile_i * q_tile;
            const rows_active = @min(q_tile, task.q_seq - q0);
            for (0..rows_active) |qr| {
                const query_i = q0 + qr;
                const p = task.source_offset + query_i;
                const active = if (task.causal) p + 1 else task.kv_seq;
                const lo = if (!task.causal or task.window == 0) 0 else (p + 1) -| @min(task.window, task.kv_seq);
                for (0..head_group) |j| {
                    const head_i = head_unit * head_group + j;
                    forwardRowNoScratch(
                        KvElem,
                        task.q_data[query_i * task.heads * task.d + head_i * task.d ..][0..task.d],
                        task.k_data,
                        task.v_data,
                        task.out_data[query_i * task.heads * task.d + head_i * task.d ..][0..task.d],
                        kv_head_i,
                        task.kv_heads,
                        task.d,
                        lo,
                        active,
                        task.scale_value,
                        if (task.bias) |bias_data| bias_data[query_i * task.kv_seq ..][0..task.kv_seq] else null,
                        if (task.stats) |stats| stats[(head_i * task.q_seq + query_i) * 2 ..][0..2] else null,
                    );
                }
            }
        }
    }

    pub fn groupedCausalAttentionBackwardKvHeads(task: GroupedCausalAttentionBackwardTask) void {
        const q_seq_stride = task.heads * task.d;
        const kv_seq_stride = task.kv_heads * task.d;
        for (0..task.heads) |head_i| {
            const kv_head_i = task.kv_head_for_head[head_i];
            if (kv_head_i < task.kv_head_start or kv_head_i >= task.kv_head_end) continue;
            for (0..task.q_seq) |query_i| {
                const active = if (task.causal) task.source_offset + query_i + 1 else task.kv_seq;
                const lo = if (!task.causal or task.window == 0) 0 else active -| task.window;
                const q_base = query_i * q_seq_stride + head_i * task.d;
                const gy_base = query_i * q_seq_stride + head_i * task.d;

                var max_score = -std.math.inf(f32);
                for (lo..active) |source_i| {
                    const k_base = source_i * kv_seq_stride + kv_head_i * task.d;
                    var dot_value: f32 = 0;
                    for (0..task.d) |feature_i| {
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
                for (lo..active) |source_i| task.scores[source_i] *= inv_sum;

                var dprob_dot: f32 = 0;
                for (lo..active) |source_i| {
                    const v_base = source_i * kv_seq_stride + kv_head_i * task.d;
                    var grad_value: f32 = 0;
                    for (0..task.d) |feature_i| {
                        grad_value += task.gy_data[gy_base + feature_i] * task.v_data[v_base + feature_i];
                    }
                    task.dprob[source_i] = grad_value;
                    dprob_dot += task.scores[source_i] * grad_value;
                }

                for (lo..active) |source_i| {
                    const prob = task.scores[source_i];
                    const dscore = prob * (task.dprob[source_i] - dprob_dot);
                    const scaled_dscore = task.scale_value * dscore;
                    const k_base = source_i * kv_seq_stride + kv_head_i * task.d;
                    if (task.v_grad) |grad| {
                        for (0..task.d) |feature_i| {
                            grad[k_base + feature_i] += prob * task.gy_data[gy_base + feature_i];
                        }
                    }
                    if (task.q_grad) |grad| {
                        for (0..task.d) |feature_i| {
                            grad[q_base + feature_i] += scaled_dscore * task.k_data[k_base + feature_i];
                        }
                    }
                    if (task.k_grad) |grad| {
                        for (0..task.d) |feature_i| {
                            grad[k_base + feature_i] += scaled_dscore * task.q_data[q_base + feature_i];
                        }
                    }
                }
            }
        }
    }

    pub fn groupedCausalAttentionBackwardTiles(task: GroupedCausalAttentionBackwardTiledTask) void {
        const d = task.d;
        const q_seq_stride = task.heads * d;
        const kv_seq_stride = task.kv_heads * d;
        const p_row = task.scratch[0..task.kv_seq];
        const ds_row = task.scratch[task.kv_seq..][0..task.kv_seq];

        for (task.head_start..task.head_end) |head_i| {
            const kv_head_i = task.kv_head_for_head[head_i];
            const kv_head_base = kv_head_i * d;
            const acc_base = if (task.partial_mode) head_i * task.kv_seq * d else kv_head_base;
            const acc_stride = if (task.partial_mode) d else kv_seq_stride;
            const stats_head: ?[]const f32 = if (task.stats) |values| values[head_i * task.q_seq * 2 ..][0 .. task.q_seq * 2] else null;

            for (0..task.q_seq) |query_i| {
                const p_first = task.source_offset + query_i;
                const active = if (task.causal) @min(p_first + 1, task.kv_seq) else task.kv_seq;
                const lo = if (!task.causal or task.window == 0) 0 else (p_first + 1) -| task.window;
                if (active <= lo) continue;
                const q_base = query_i * q_seq_stride + head_i * d;

                // Scaled scores and the dO.V^T row.
                for (lo..active) |source_i| {
                    const k_base = source_i * kv_seq_stride + kv_head_base;
                    var dot_value: f32 = 0;
                    var dp_value: f32 = 0;
                    for (0..d) |feature_i| {
                        dot_value += task.q_data[q_base + feature_i] * task.k_data[k_base + feature_i];
                        dp_value += task.gy_data[q_base + feature_i] * task.v_data[k_base + feature_i];
                    }
                    p_row[source_i] = dot_value * task.scale_value;
                    ds_row[source_i] = dp_value;
                }

                // Probabilities (stats or recompute), row dot, dS in place.
                var max_score = -std.math.inf(f32);
                var inv_sum: f32 = undefined;
                if (stats_head) |values| {
                    max_score = values[query_i * 2];
                    inv_sum = 1 / values[query_i * 2 + 1];
                } else {
                    for (lo..active) |source_i| max_score = @max(max_score, p_row[source_i]);
                }
                var sum_exp: f32 = 0;
                for (lo..active) |source_i| {
                    const p = @exp(p_row[source_i] - max_score);
                    p_row[source_i] = p;
                    sum_exp += p;
                }
                if (stats_head == null) inv_sum = 1 / sum_exp;
                var row_dot: f32 = 0;
                for (lo..active) |source_i| {
                    const p = p_row[source_i] * inv_sum;
                    p_row[source_i] = p;
                    row_dot += p * ds_row[source_i];
                }
                for (lo..active) |source_i| {
                    ds_row[source_i] = task.scale_value * p_row[source_i] * (ds_row[source_i] - row_dot);
                }

                // dQ (single writer per row), then the dK/dV accumulations.
                if (task.q_grad) |grad| {
                    for (0..d) |feature_i| {
                        var acc: f32 = 0;
                        for (lo..active) |source_i| {
                            acc += ds_row[source_i] * task.k_data[source_i * kv_seq_stride + kv_head_base + feature_i];
                        }
                        grad[q_base + feature_i] = acc;
                    }
                }
                for (lo..active) |source_i| {
                    const target_row = acc_base + source_i * acc_stride;
                    if (task.dv_target) |target| {
                        for (0..d) |feature_i| {
                            target[target_row + feature_i] += p_row[source_i] * task.gy_data[q_base + feature_i];
                        }
                    }
                    if (task.dk_target) |target| {
                        for (0..d) |feature_i| {
                            target[target_row + feature_i] += ds_row[source_i] * task.q_data[q_base + feature_i];
                        }
                    }
                }
            }
        }
    }
};
