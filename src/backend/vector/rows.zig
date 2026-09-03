//! The row-kernel seam: the Task payloads the fused row kernels take
//! (pure slices and dims, plus a ranked shape/stride array or an optional
//! RankedTensor mask where a kernel walks a strided view), the fused
//! activation+quantize task factory, and the small helpers the exec domain
//! modules and the kernels share (`coordinateForLinear`, `rowSumSq`,
//! `dropoutKeepCutoff`, the weight-grad block rows). The kernel bodies live
//! in `rows/kernels.zig` beneath this file and are reached through
//! `backend.kernels` only; each takes its row, lane, block, column or
//! vector range as two trailing parameters, and `ExecContext.forRange`
//! splits the range.
const std = @import("std");
const common = @import("common.zig");
const dtype_mod = @import("../../dtype.zig");
const tensor = @import("../../tensor.zig");
const isa = @import("../isa.zig");
const quantized_matmul = @import("../quant.zig");
const elementwise = @import("elementwise.zig");
const kernels = @import("rows/kernels.zig");

const DType = dtype_mod.DType;

/// Coordinate along `axis` of the contiguous linear index `linear` under
/// `shape`/`strides` (row-major contiguous strides). Shared index math of
/// the strided row kernels; `exec/gather_scatter.zig` reaches it for the exec
/// band's own strided walks.
pub fn coordinateForLinear(comptime rank: usize, shape: [rank]usize, strides: [rank]usize, linear: usize, axis: usize) usize {
    return (linear / strides[axis]) % shape[axis];
}

pub const SplitSwiGluTask = struct {
    input: []const f32,
    output: []f32,
    axis_dim: usize,
    half: usize,
};

/// The glu twin carries the same payload: one task type, two kernels.
pub const SplitGluTask = SplitSwiGluTask;

// Activation flavor for the fused activation+quantize+packed-GEMM ops.
pub const FusedActKind = enum { split_swiglu, geglu_quant, rms_norm_mul };

// LHS block layout the fused quantizer produces.
pub const FusedLhsFormat = enum { q8_kx4, q8_k_rows, q8_0x4 };

/// Row-group worker for the fused ops: activates up to 4 rows into task-private
/// scratch with the exact kernels the unfused path uses, then quantizes the
/// scratch with the exact packers the unfused matmul path uses — so results
/// stay bit-identical while the m*k activation tensor is never materialized.
pub fn FusedActQuantTask(comptime act: FusedActKind, comptime format: FusedLhsFormat) type {
    return struct {
        // split_swiglu: `gate` holds fused gate_up rows of width 2*cols (gate
        // first half, up second). geglu_quant: `gate`/`up` hold separate rows
        // of width cols; gated = up * geluQuant(gate). rms_norm_mul: `gate`
        // holds the PRE-norm rows of width cols and `up` the [cols] norm
        // weight row (eps/inv_cols below).
        gate: []const f32,
        up: []const f32,
        scratch: []f32,
        rows: usize,
        cols: usize,
        blocks_per_row: usize,
        // rms_norm_mul only. `rows_kernel` mirrors the unfused
        // rmsNormMul dispatch (rows kernel at/above the elementwise
        // work threshold, the scalar loop below it) so the fused route stays
        // BITWISE identical to rmsNormMul + quantize at every shape.
        eps: f32 = 0,
        inv_cols: f32 = 0,
        rows_kernel: bool = true,
        row_group_start: usize,
        row_group_end: usize,
        x4_blocks: []quantized_matmul.types.BlockQ8_Kx4 = &.{},
        row_blocks: []dtype_mod.BlockQ8_K = &.{},
        q8_0x4_blocks: []quantized_matmul.types.BlockQ8_0x4 = &.{},

        pub fn run(task: *const @This()) void {
            const cols = task.cols;
            for (task.row_group_start..task.row_group_end) |row_group| {
                const row0 = row_group * 4;
                const rows_in_group = @min(task.rows - row0, 4);
                const group_scratch = task.scratch[0 .. rows_in_group * cols];
                switch (act) {
                    .split_swiglu => kernels.splitSwiGluRows(.{
                        .input = task.gate[row0 * cols * 2 ..],
                        .output = group_scratch,
                        .axis_dim = cols * 2,
                        .half = cols,
                    }, 0, rows_in_group),
                    .geglu_quant => for (0..rows_in_group) |r| {
                        const dst = task.scratch[r * cols ..][0..cols];
                        elementwise.unaryRowSlice(.gelu_quant, dst, task.gate[(row0 + r) * cols ..][0..cols]);
                        elementwise.mulRowSlice(dst, dst, task.up[(row0 + r) * cols ..][0..cols]);
                    },
                    .rms_norm_mul => if (task.rows_kernel) kernels.rmsNormMulRows(.{
                        .input = task.gate[row0 * cols ..],
                        .weights = task.up,
                        .output = group_scratch,
                        .axis_dim = cols,
                        .inv_axis_dim = task.inv_cols,
                        .eps = task.eps,
                    }, 0, rows_in_group) else for (0..rows_in_group) |r| {
                        // The unfused sub-threshold scalar loop, verbatim.
                        const row_in = task.gate[(row0 + r) * cols ..][0..cols];
                        const row_out = task.scratch[r * cols ..][0..cols];
                        var sumsq: f32 = 0;
                        for (row_in) |value| sumsq += value * value;
                        const scale_value = 1 / @sqrt(sumsq * task.inv_cols + task.eps);
                        for (row_out, row_in, task.up) |*dst, value, weight| {
                            dst.* = value * scale_value * weight;
                        }
                    },
                }
                switch (format) {
                    .q8_kx4 => quantized_matmul.q8k.quantizeRowGroupQ8_Kx4Into(
                        task.x4_blocks[row_group * task.blocks_per_row ..][0..task.blocks_per_row],
                        group_scratch,
                        rows_in_group,
                        cols,
                    ),
                    .q8_k_rows => for (0..rows_in_group) |r| {
                        quantized_matmul.q8k.quantizeRowQ8_KIntoUnchecked(
                            task.row_blocks[(row0 + r) * task.blocks_per_row ..][0..task.blocks_per_row],
                            task.scratch[r * cols ..][0..cols],
                        );
                    },
                    .q8_0x4 => {
                        // The plain group packer reads all 4 lanes; zero rows
                        // quantize to d=0/qs=0, matching padded-lane semantics.
                        if (rows_in_group < 4) @memset(task.scratch[rows_in_group * cols .. 4 * cols], 0);
                        quantized_matmul.q8_0.quantizeRowsQ8_0x4GroupsInto(
                            task.q8_0x4_blocks[row_group * task.blocks_per_row ..][0..task.blocks_per_row],
                            task.scratch[0 .. 4 * cols],
                            4,
                            cols,
                            task.blocks_per_row,
                            0,
                            1,
                        );
                    },
                }
            }
        }
    };
}

pub const SplitSwiGluBackwardTask = struct {
    input: []const f32,
    grad: []const f32,
    output: []f32,
    axis_dim: usize,
    half: usize,
};

pub const SplitGluBackwardTask = SplitSwiGluBackwardTask;

pub const RmsNormMulRopeHalfTask = struct {
    input: []const f32,
    weights: []const f32,
    output: []f32,
    sin_values: []const f32,
    cos_values: []const f32,
    shape: [tensor.max_rank]usize,
    input_strides: [tensor.max_rank]usize,
    output_strides: [tensor.max_rank]usize,
    input_offset: usize,
    rank: usize,
    position_axis: usize,
    feature_axis: usize,
    feature_dim: usize,
    pair_count: usize,
    inv_feature_dim: f32,
    eps: f32,
};

pub const RmsNormMulRowsTask = struct {
    input: []const f32,
    weights: []const f32,
    output: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
};

pub const RmsNormMulAddRowsTask = struct {
    input: []const f32,
    weights: []const f32,
    residual: []const f32,
    output: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
};

pub const RmsNormMulBackwardInputRowsTask = struct {
    input: []const f32,
    weights: []const f32,
    grad: []const f32,
    output: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
};

pub const RmsNormMulBackwardWeightRowsTask = struct {
    input: []const f32,
    grad: []const f32,
    output: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
};

/// Row-block grid of the large-input rmsNorm weight gradient: a fixed
/// 64 rows per block, independent of the thread count, so which rows a
/// block's partial covers and the order the partials are reduced in are
/// properties of the shape alone.
pub const rms_weight_grad_block_rows: usize = 64;

pub const RmsNormWeightGradBlocksTask = struct {
    input: []const f32,
    grad: []const f32,
    /// `[block_count][axis_dim]` partials. Block b owns rows
    /// `[b * block_rows, min((b + 1) * block_rows, rows))` and accumulates
    /// them in row order from zero, so its partial is a pure function of
    /// its rows whichever task computes it.
    partials: []f32,
    rows: usize,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
};

pub const RmsNormWeightGradReduceTask = struct {
    partials: []const f32,
    /// Pre-zeroed `[axis_dim]`. Each task owns the column range
    /// `[col_start, col_end)` and adds the block partials in block order;
    /// the vector body and the scalar tail evaluate the same
    /// `acc + partial` per column, so the column split never moves a bit.
    output: []f32,
    block_count: usize,
    axis_dim: usize,
};

pub const LayerNormRowsTask = struct {
    input: []const f32,
    // Affine parameters (rank-1 [axis_dim]); both null = plain normalize.
    // One task type covers layerNorm and layerNormAffine — the branch sits
    // outside the vector loops.
    weights: ?[]const f32,
    biases: ?[]const f32,
    output: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
};

pub const LayerNormBackwardInputRowsTask = struct {
    input: []const f32,
    // Affine weight (rank-1 [axis_dim]); null = plain layerNorm (g' = gy).
    weights: ?[]const f32,
    grad: []const f32,
    output: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
};

pub const LayerNormRowStatsTask = struct {
    input: []const f32,
    // Per-row {mean, 1/σ} pairs (mean at [2·row], 1/σ at [2·row+1]) —
    // disjoint writes by row, and each value is a pure function of its row,
    // so the scratch is bitwise identical for any thread count.
    stats: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
};

pub const LayerNormParamGradColumnsTask = struct {
    input: []const f32,
    grad: []const f32,
    stats: []const f32,
    // Each task owns the contiguous DESTINATION column range
    // [col_start, col_end) and accumulates over ALL rows in row order
    // (the ScatterAddRows destination-partition pattern): per-column
    // accumulation order equals the serial row order, so dweight/dbias are
    // bitwise identical for any thread count.
    dweight: ?[]f32,
    dbias: ?[]f32,
    rows: usize,
    axis_dim: usize,
};

pub const SoftmaxRowsTask = struct {
    input: []const f32,
    output: []f32,
    axis_dim: usize,
};

/// The fused log-domain row kernels share the softmax payload:
/// `logsumexpRows` writes ONE output slot per row, `logSoftmaxRows` a
/// full row.
pub const LogRowsTask = SoftmaxRowsTask;

pub fn SoftmaxExtRowsTask(comptime rank: usize) type {
    return struct {
        input: []const f32,
        output: []f32,
        shape: [rank]usize,
        strides: [rank]usize,
        mask: ?tensor.RankedTensor(rank),
        sinks: ?[]const f32,
        slopes: ?[]const f32,
        scale: f32,
        head_axis: ?usize,
        causal_query_axis: ?usize,
        causal_source_offset: usize,
        axis_dim: usize,
        inner: usize,
        // inner == 1 and the mask (if any) is unit-stride along the softmax
        // axis: rows are contiguous, so they take the SIMD body.
        simd_rows: bool,
    };
}

pub const SoftmaxBackwardRowsTask = struct {
    y: []const f32,
    gy: []const f32,
    output: []f32,
    axis_dim: usize,
    scale: f32,
};

pub const CrossEntropyLossRowsTask = struct {
    input: []const f32,
    labels: []const usize,
    // Per-row losses indexed by row — disjoint writes across tasks; the
    // dispatcher does ONE serial sum in row order, so the reduced loss is
    // bitwise identical for any thread count (same policy as optim.zig's
    // parallel maps).
    row_losses: []f32,
    // When set (length 2 * rows), receives the per-row softmax statistics
    // {max, sum_exp} interleaved — exactly the f32 values this kernel
    // computes for the loss, so a backward fed with them is bitwise
    // identical to one that recomputes them (identical reduction shapes).
    // Ignored rows get {0, 1}. Disjoint writes across tasks, like
    // `row_losses`.
    row_stats: ?[]f32,
    class_count: usize,
    ignore_index: ?usize,
    label_smoothing: f32,
};

pub const CrossEntropyBackwardRowsTask = struct {
    input: []const f32,
    labels: []const usize,
    output: []f32,
    // Upstream gradient per position for `.none` reduction (null for the
    // scalar mean/sum upstream, which is folded into `grad_common`).
    per_row_scale: ?[]const f32,
    // Forward-saved per-row {max, sum_exp} (the CrossEntropyLossRowsTask
    // layout). When set, the kernel skips the max scan and the exp-sum
    // reduction and emits final gradients in ONE pass over the row —
    // bitwise identical to the recompute path (the stats are the exact f32
    // values that path would recompute, and the per-element op order is
    // unchanged).
    row_stats: ?[]const f32,
    class_count: usize,
    ignore_index: ?usize,
    label_smoothing: f32,
    grad_common: f32,
};

pub const DropoutRangeTask = struct {
    input: []const f32,
    output: []f32,
    // Element i keeps its value iff the 53-bit uniform of rng.at(seed, i) is
    // < 1 - p. The hot loop compares integers: with k = rng.at >> 11 and
    // t = 1 - p, `k * 2^-53 < t` iff `k < ceil(t * 2^53)` (both conversions
    // are exact — k has <= 53 bits and t * 2^53 is a power-of-two scaling),
    // so the mask is bit-identical to the historical f64 comparison; the
    // (seed, i) -> mask mapping is a checkpoint contract. Counter-based —
    // a pure function of (seed, i) — so flat element ranges partition freely
    // across tasks and the result is bitwise identical for any thread count.
    keep_cutoff: u64,
    scale: f32,
    seed: u64,
};

/// Integer form of the dropout keep predicate (see `DropoutRangeTask`).
/// Requires `0 <= p < 1`.
pub fn dropoutKeepCutoff(p: f32) u64 {
    const keep_threshold = 1.0 - @as(f64, p);
    return @intFromFloat(@ceil(keep_threshold * 0x1.0p53));
}

pub const ScatterAddRowsTask = struct {
    grad: []const f32,
    // Each task owns the contiguous DESTINATION row range [row_start, row_end):
    // it zeroes that range, then scans the full index list and accumulates only
    // the grad rows that land inside it. Writes are disjoint across tasks by
    // construction, and per-destination accumulation order equals the serial
    // index order, so the result is bitwise identical for any thread count.
    // (Pre-binning source rows per task was measured-declined 2026-07-10:
    // interleaved paired medians -0.8%..+3.4% on M1 and neutral on x86 — the
    // dense zero-fill dominates, so the serial bin build only added
    // critical-path work. Re-open with sparse embedding gradients, where a
    // per-destination index structure is required anyway.)
    output: []f32,
    indices: []const usize,
    row_len: usize,
};

/// Sum of squares of one contiguous row, the RMS-norm statistic every
/// rmsNorm row kernel shares: four 8-lane accumulators over the bulk,
/// combined as `(acc0 + acc1) + (acc2 + acc3)`, one 8-lane loop over the
/// remainder, a horizontal reduce, then the scalar tail. The order is the
/// bit contract of the rmsNorm family (goldens pin it); keep it. Public
/// for the tests' bit oracles.
pub inline fn rowSumSq(row: []const f32) f32 {
    // The reference arm: one serial accumulator in index order (the rms
    // twins and the tests' bit oracles both resolve to this on
    // `-Dbackend=scalar` builds).
    if (comptime isa.reference) {
        var serial_sumsq: f32 = 0;
        for (row) |value| serial_sumsq += value * value;
        return serial_sumsq;
    }
    const Vec = common.RowVec;
    const vector_width = common.row_lanes;
    var i: usize = 0;
    var acc0: Vec = @splat(0);
    var acc1: Vec = @splat(0);
    var acc2: Vec = @splat(0);
    var acc3: Vec = @splat(0);
    while (i + 4 * vector_width <= row.len) : (i += 4 * vector_width) {
        const v0: Vec = row[i..][0..vector_width].*;
        const v1: Vec = row[i + vector_width ..][0..vector_width].*;
        const v2: Vec = row[i + 2 * vector_width ..][0..vector_width].*;
        const v3: Vec = row[i + 3 * vector_width ..][0..vector_width].*;
        acc0 += v0 * v0;
        acc1 += v1 * v1;
        acc2 += v2 * v2;
        acc3 += v3 * v3;
    }
    var sumsq_vec: Vec = (acc0 + acc1) + (acc2 + acc3);
    while (i + vector_width <= row.len) : (i += vector_width) {
        const values: Vec = row[i..][0..vector_width].*;
        sumsq_vec += values * values;
    }
    var sumsq: f32 = @reduce(.Add, sumsq_vec);
    while (i < row.len) : (i += 1) {
        const value = row[i];
        sumsq += value * value;
    }
    return sumsq;
}

pub const SoftmaxInnerTask = struct {
    input: []const f32,
    output: []f32,
    axis_dim: usize,
    inner: usize,
    /// Pooled f32 scratch of length `2 * inner`: max row, then sum row.
    /// Tasks touch only their own `[inner_start, inner_end)` columns, so a
    /// single scratch allocation serves every task disjointly.
    scratch: []f32,
    outer: usize,
};

pub const SoftmaxBackwardInnerTask = struct {
    y: []const f32,
    gy: []const f32,
    output: []f32,
    axis_dim: usize,
    inner: usize,
    /// Pooled f32 scratch of length `inner`: the per-lane gy·y dot row.
    /// Tasks touch only their own `[inner_start, inner_end)` columns.
    scratch: []f32,
    scale: f32,
    outer: usize,
};

pub const LayerNormBackwardInnerTask = struct {
    input: []const f32,
    grad: []const f32,
    /// Per-axis weight row (`[axis_dim]`), or null for the un-affine form.
    weights: ?[]const f32,
    dx: ?[]f32,
    dweight: ?[]f32,
    dbias: ?[]f32,
    axis_dim: usize,
    inner: usize,
    /// Pooled f32 scratch of length `4 * inner`: mean, gsum, sumsq/inv_sigma,
    /// dot/correction rows (the last two pairs share slots).
    scratch: []f32,
    inv_axis_dim: f32,
    eps: f32,
    outer: usize,
};

pub const VarianceInnerTask = struct {
    input: []const f32,
    /// One `inner`-wide row per outer block (the axis removed).
    output: []f32,
    axis_dim: usize,
    inner: usize,
    /// Pooled f32 scratch of length `2 * inner`: sum (then mean) row, then
    /// the centered sum-of-squares row. Tasks touch only their own
    /// `[inner_start, inner_end)` columns.
    scratch: []f32,
    outer: usize,
    inv_axis_dim: f32,
    /// `1 / (axis_dim - ddof)`.
    inv_denom: f32,
};

pub fn StandardizeInnerTask(comptime Acc: type) type {
    return struct {
        input: []const f32,
        output: []f32,
        axis_dim: usize,
        inner: usize,
        /// Standardized prefix of the axis; positions past it are written 0.
        valid_count: usize,
        ddof_count: usize,
        eps: Acc,
        /// `denom = sqrt(var + eps)` when set, `sqrt(var) + eps` otherwise.
        eps_inside_sqrt: bool,
        /// Pooled `Acc` scratch of length `2 * inner`: mean row, then the
        /// variance (finalized in place to the denominator) row.
        scratch: []Acc,
        outer: usize,
    };
}

pub fn StandardizeBackwardInnerTask(comptime Acc: type) type {
    return struct {
        input: []const f32,
        grad: []const f32,
        /// Pre-zeroed: `valid_count == 0` blocks are left untouched.
        output: []f32,
        axis_dim: usize,
        inner: usize,
        valid_count: usize,
        ddof_count: usize,
        eps: Acc,
        eps_inside_sqrt: bool,
        /// Pooled `Acc` scratch of length `5 * inner`: mean, variance,
        /// denominator (1/denominator after the finalize), gradient sum
        /// (then mean gradient), centered-gradient dot (then second scale).
        scratch: []Acc,
        outer: usize,
    };
}

pub const RmsNormInnerTask = struct {
    input: []const f32,
    /// Per-axis weight row (`[axis_dim]`) and same-shape residual; null =
    /// absent. One kernel covers the plain / weighted / weighted+residual
    /// forward arms with their exact expression order.
    weights: ?[]const f32,
    residual: ?[]const f32,
    output: []f32,
    axis_dim: usize,
    inner: usize,
    /// Pooled f32 scratch of length `inner`: sum-of-squares, then 1/rms row.
    scratch: []f32,
    outer: usize,
    inv_axis_dim: f32,
    eps: f32,
};

pub const RmsNormBackwardInputInnerTask = struct {
    input: []const f32,
    /// Per-axis weight row (`[axis_dim]`); null = the plain (unweighted)
    /// rmsNorm, whose correction term is spelled in its own order.
    weights: ?[]const f32,
    grad: []const f32,
    output: []f32,
    axis_dim: usize,
    inner: usize,
    /// Pooled f32 scratch of length `2 * inner`: sum-of-squares (then
    /// 1/rms) row, then the grad·x dot (then correction) row.
    scratch: []f32,
    outer: usize,
    inv_axis_dim: f32,
    eps: f32,
};

pub const RmsNormBackwardWeightInnerTask = struct {
    input: []const f32,
    grad: []const f32,
    /// `[axis_dim]` accumulator (pre-zeroed): every (outer, lane) pair is
    /// chained into it in the scalar loop's order, so this kernel has no
    /// lane split and runs serially.
    output: []f32,
    axis_dim: usize,
    inner: usize,
    /// f32 scratch of length `inner`: sum-of-squares, then 1/rms row.
    scratch: []f32,
    outer: usize,
    inv_axis_dim: f32,
    eps: f32,
};

pub const LayerNormInnerTask = struct {
    input: []const f32,
    /// Affine terms (rank-1 `[axis_dim]`); null = absent.
    weights: ?[]const f32,
    biases: ?[]const f32,
    output: []f32,
    axis_dim: usize,
    inner: usize,
    /// Pooled f32 scratch of length `2 * inner`: sum (then mean) row, then
    /// the centered sum-of-squares (then 1/sigma) row.
    scratch: []f32,
    outer: usize,
    inv_axis_dim: f32,
    eps: f32,
};
