const std = @import("std");
const backend_mod = @import("../backend.zig");
const backend_ops = backend_mod.ops;
const rng = @import("../rng.zig");
const tensor = @import("../tensor.zig");
const shape = @import("shape.zig");

const vexpf = @import("../backend/vector/primitives.zig").vexpf;
const coordinateForLinear = shape.coordinateForLinear;
const physicalOffsetExcludingAxis = shape.physicalOffsetExcludingAxis;
const preSoftmaxValue = shape.preSoftmaxValue;

pub const SplitSwiGluTask = struct {
    input: []const f32,
    output: []f32,
    axis_dim: usize,
    half: usize,
    outer_start: usize,
    outer_end: usize,
};

pub const SplitGluTask = struct {
    input: []const f32,
    output: []f32,
    axis_dim: usize,
    half: usize,
    outer_start: usize,
    outer_end: usize,
};

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
        backend: *const backend_mod.Backend,
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
        // rmsNormMulAxisRank dispatch (rows kernel at/above the elementwise
        // work threshold, the scalar loop below it) so the fused route stays
        // BITWISE identical to rmsNormMul + quantize at every shape.
        eps: f32 = 0,
        inv_cols: f32 = 0,
        rows_kernel: bool = true,
        row_group_start: usize,
        row_group_end: usize,
        x4_blocks: []backend_mod.quantized_matmul.BlockQ8_Kx4 = &.{},
        row_blocks: []backend_mod.quantized_matmul.BlockQ8_K = &.{},
        q8_0x4_blocks: []backend_mod.quantized_matmul.BlockQ8_0x4 = &.{},

        pub fn run(task: *const @This()) void {
            const cols = task.cols;
            for (task.row_group_start..task.row_group_end) |row_group| {
                const row0 = row_group * 4;
                const rows_in_group = @min(task.rows - row0, 4);
                const group_scratch = task.scratch[0 .. rows_in_group * cols];
                switch (act) {
                    .split_swiglu => splitSwiGluRows(.{
                        .input = task.gate[row0 * cols * 2 ..],
                        .output = group_scratch,
                        .axis_dim = cols * 2,
                        .half = cols,
                        .outer_start = 0,
                        .outer_end = rows_in_group,
                    }),
                    .geglu_quant => for (0..rows_in_group) |r| {
                        const dst = task.scratch[r * cols ..][0..cols];
                        task.backend.unaryRowSliceUnchecked(.gelu_quant, dst, task.gate[(row0 + r) * cols ..][0..cols]);
                        task.backend.mulRowSliceUnchecked(dst, dst, task.up[(row0 + r) * cols ..][0..cols]);
                    },
                    .rms_norm_mul => if (task.rows_kernel) rmsNormMulRows(.{
                        .input = task.gate[row0 * cols ..],
                        .weights = task.up,
                        .output = group_scratch,
                        .axis_dim = cols,
                        .inv_axis_dim = task.inv_cols,
                        .eps = task.eps,
                        .row_start = 0,
                        .row_end = rows_in_group,
                    }) else for (0..rows_in_group) |r| {
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
                    .q8_kx4 => backend_mod.quantized_matmul.quantizeRowGroupQ8_Kx4Into(
                        task.x4_blocks[row_group * task.blocks_per_row ..][0..task.blocks_per_row],
                        group_scratch,
                        rows_in_group,
                        cols,
                    ),
                    .q8_k_rows => for (0..rows_in_group) |r| {
                        backend_mod.quantized_matmul.quantizeRowQ8_KInto(
                            task.row_blocks[(row0 + r) * task.blocks_per_row ..][0..task.blocks_per_row],
                            task.scratch[r * cols ..][0..cols],
                        ) catch unreachable;
                    },
                    .q8_0x4 => {
                        // The plain group packer reads all 4 lanes; zero rows
                        // quantize to d=0/qs=0, matching padded-lane semantics.
                        if (rows_in_group < 4) @memset(task.scratch[rows_in_group * cols .. 4 * cols], 0);
                        backend_mod.quantized_matmul.quantizeRowsQ8_0x4GroupsInto(
                            task.q8_0x4_blocks[row_group * task.blocks_per_row ..][0..task.blocks_per_row],
                            task.scratch[0 .. 4 * cols],
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

pub const SplitSwiGluQuantQ8_0x4Task = struct {
    input: []const f32,
    blocks: []backend_mod.quantized_matmul.BlockQ8_0x4,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    row_group_start: usize,
    row_group_end: usize,
};

pub const SplitSwiGluBackwardTask = struct {
    input: []const f32,
    grad: []const f32,
    output: []f32,
    axis_dim: usize,
    half: usize,
    outer_start: usize,
    outer_end: usize,
};

pub const SplitGluBackwardTask = struct {
    input: []const f32,
    grad: []const f32,
    output: []f32,
    axis_dim: usize,
    half: usize,
    outer_start: usize,
    outer_end: usize,
};

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
    vector_start: usize,
    vector_end: usize,
};

pub const RmsNormMulRowsTask = struct {
    input: []const f32,
    weights: []const f32,
    output: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
    row_start: usize,
    row_end: usize,
};

pub const RmsNormMulAddRowsTask = struct {
    input: []const f32,
    weights: []const f32,
    residual: []const f32,
    output: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
    row_start: usize,
    row_end: usize,
};

pub const RmsNormMulBackwardInputRowsTask = struct {
    input: []const f32,
    weights: []const f32,
    grad: []const f32,
    output: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
    row_start: usize,
    row_end: usize,
};

pub const RmsNormMulBackwardWeightRowsTask = struct {
    input: []const f32,
    grad: []const f32,
    output: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
    row_start: usize,
    row_end: usize,
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
    row_start: usize,
    row_end: usize,
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
    row_start: usize,
    row_end: usize,
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
    row_start: usize,
    row_end: usize,
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
    col_start: usize,
    col_end: usize,
};

pub const SoftmaxRowsTask = struct {
    input: []const f32,
    output: []f32,
    axis_dim: usize,
    row_start: usize,
    row_end: usize,
};

/// Shared task shape for the fused log-domain row kernels: `logsumexpRows`
/// writes ONE output slot per row; `logSoftmaxRows` writes a full row.
pub const LogRowsTask = struct {
    input: []const f32,
    output: []f32,
    axis_dim: usize,
    row_start: usize,
    row_end: usize,
};

pub fn runLogsumexpRowsTask(task: *const LogRowsTask) void {
    logsumexpRows(task.*);
}

pub fn runLogSoftmaxRowsTask(task: *const LogRowsTask) void {
    logSoftmaxRows(task.*);
}

/// SIMD row max with the log-domain non-finite guard: rows whose max is
/// ±inf shift by 0 instead (torch logsumexp/log_softmax convention — an
/// all(-inf) row yields -inf, a +inf entry yields +inf, never NaN).
inline fn rowMaxSafe(row_in: []const f32) f32 {
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    var axis_i: usize = 0;
    var max_vec: Vec = @splat(-std.math.inf(f32));
    while (axis_i + vector_width <= row_in.len) : (axis_i += vector_width) {
        max_vec = @max(max_vec, @as(Vec, row_in[axis_i..][0..vector_width].*));
    }
    var max_value = @reduce(.Max, max_vec);
    while (axis_i < row_in.len) : (axis_i += 1) {
        max_value = @max(max_value, row_in[axis_i]);
    }
    return if (std.math.isFinite(max_value)) max_value else 0;
}

/// Fused log-sum-exp rows (torch.logsumexp over the last axis):
/// `output[row] = m + log(Σ exp(x - m))` with `m` the guarded row max —
/// one SIMD max scan and one SIMD vexpf-sum per row, no materialized
/// intermediate. Output holds one slot per row.
pub fn logsumexpRows(task: LogRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row_i| {
        const row_in = task.input[row_i * task.axis_dim ..][0..task.axis_dim];
        const max_safe = rowMaxSafe(row_in);

        const max_splat: Vec = @splat(max_safe);
        var sum_vec: Vec = @splat(0);
        var axis_i: usize = 0;
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            sum_vec += vexpf(vector_width, @as(Vec, row_in[axis_i..][0..vector_width].*) - max_splat);
        }
        var sum_exp = @reduce(.Add, sum_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            sum_exp += vexpf(1, @splat(row_in[axis_i] - max_safe))[0];
        }
        task.output[row_i] = max_safe + @log(sum_exp);
    }
}

/// Fused log-softmax rows (torch.log_softmax over the last axis):
/// `output[i] = (x[i] - m) - log(Σ exp(x - m))` with the same guarded
/// max — two SIMD passes per row (vexpf-sum, then shift), no
/// materialized intermediate.
pub fn logSoftmaxRows(task: LogRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row_i| {
        const base = row_i * task.axis_dim;
        const row_in = task.input[base..][0..task.axis_dim];
        const row_out = task.output[base..][0..task.axis_dim];
        const max_safe = rowMaxSafe(row_in);

        const max_splat: Vec = @splat(max_safe);
        var sum_vec: Vec = @splat(0);
        var axis_i: usize = 0;
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            sum_vec += vexpf(vector_width, @as(Vec, row_in[axis_i..][0..vector_width].*) - max_splat);
        }
        var sum_exp = @reduce(.Add, sum_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            sum_exp += vexpf(1, @splat(row_in[axis_i] - max_safe))[0];
        }

        const shift = max_safe + @log(sum_exp);
        const shift_splat: Vec = @splat(shift);
        axis_i = 0;
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            row_out[axis_i..][0..vector_width].* = @as(Vec, row_in[axis_i..][0..vector_width].*) - shift_splat;
        }
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            row_out[axis_i] = row_in[axis_i] - shift;
        }
    }
}

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
        row_start: usize,
        row_end: usize,
    };
}

pub const SoftmaxBackwardRowsTask = struct {
    y: []const f32,
    gy: []const f32,
    output: []f32,
    axis_dim: usize,
    scale: f32,
    row_start: usize,
    row_end: usize,
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
    row_start: usize,
    row_end: usize,
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
    row_start: usize,
    row_end: usize,
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
    start: usize,
    end: usize,
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
    row_start: usize,
    row_end: usize,
};

pub fn runSplitSwiGluTask(task: *const SplitSwiGluTask) void {
    splitSwiGluRows(task.*);
}

pub fn runSplitGluTask(task: *const SplitGluTask) void {
    splitGluRows(task.*);
}

pub fn runSplitSwiGluQuantQ8_0x4Task(task: *const SplitSwiGluQuantQ8_0x4Task) void {
    backend_mod.quantized_matmul.quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto(
        task.blocks,
        task.input,
        task.rows,
        task.cols,
        task.blocks_per_row,
        task.row_group_start,
        task.row_group_end,
    );
}

pub fn runSplitSwiGluBackwardTask(task: *const SplitSwiGluBackwardTask) void {
    splitSwiGluBackwardRows(task.*);
}

pub fn runSplitGluBackwardTask(task: *const SplitGluBackwardTask) void {
    splitGluBackwardRows(task.*);
}

pub fn runRmsNormMulRopeHalfTask(task: *const RmsNormMulRopeHalfTask) void {
    rmsNormMulRopeHalfVectors(task.*);
}

pub fn runRmsNormMulRowsTask(task: *const RmsNormMulRowsTask) void {
    rmsNormMulRows(task.*);
}

pub fn runRmsNormMulAddRowsTask(task: *const RmsNormMulAddRowsTask) void {
    rmsNormMulAddRows(task.*);
}

pub fn runRmsNormMulBackwardInputRowsTask(task: *const RmsNormMulBackwardInputRowsTask) void {
    rmsNormMulBackwardInputRows(task.*);
}

pub fn runRmsNormMulBackwardWeightRowsTask(task: *const RmsNormMulBackwardWeightRowsTask) void {
    rmsNormMulBackwardWeightRows(task.*);
}

pub fn runLayerNormRowsTask(task: *const LayerNormRowsTask) void {
    layerNormRows(task.*);
}

pub fn runLayerNormBackwardInputRowsTask(task: *const LayerNormBackwardInputRowsTask) void {
    layerNormBackwardInputRows(task.*);
}

pub fn runLayerNormRowStatsTask(task: *const LayerNormRowStatsTask) void {
    layerNormRowStats(task.*);
}

pub fn runLayerNormParamGradColumnsTask(task: *const LayerNormParamGradColumnsTask) void {
    layerNormParamGradColumns(task.*);
}

pub fn runSoftmaxRowsTask(task: *const SoftmaxRowsTask) void {
    softmaxRows(task.*);
}

pub fn runSoftmaxExtRowsTask(comptime rank: usize, comptime axis: usize) fn (*const SoftmaxExtRowsTask(rank)) void {
    return struct {
        fn run(task: *const SoftmaxExtRowsTask(rank)) void {
            softmaxExtRows(rank, axis, task.*);
        }
    }.run;
}

pub fn runSoftmaxBackwardRowsTask(task: *const SoftmaxBackwardRowsTask) void {
    softmaxBackwardRows(task.*);
}

pub fn runCrossEntropyLossRowsTask(task: *const CrossEntropyLossRowsTask) void {
    crossEntropyLossRows(task.*);
}

pub fn runCrossEntropyBackwardRowsTask(task: *const CrossEntropyBackwardRowsTask) void {
    crossEntropyBackwardRows(task.*);
}

pub fn runScatterAddRowsTask(task: *const ScatterAddRowsTask) void {
    scatterAddRows(task.*);
}

pub fn runDropoutRangeTask(task: *const DropoutRangeTask) void {
    dropoutRange(task.*);
}

pub fn splitSwiGluRows(task: SplitSwiGluTask) void {
    const Vec = @Vector(4, f32);
    const one: Vec = @splat(1);
    for (task.outer_start..task.outer_end) |outer_i| {
        const in_base = outer_i * task.axis_dim;
        const out_base = outer_i * task.half;
        var i: usize = 0;
        while (i + 4 <= task.half) : (i += 4) {
            const gate: Vec = task.input[in_base + i ..][0..4].*;
            const up: Vec = task.input[in_base + task.half + i ..][0..4].*;
            task.output[out_base + i ..][0..4].* = up * gate * (one / (one + vexpf(4, -gate)));
        }
        while (i < task.half) : (i += 1) {
            const gate = task.input[in_base + i];
            const up = task.input[in_base + task.half + i];
            task.output[out_base + i] = up * gate / (1 + @exp(-gate));
        }
    }
}

pub fn splitGluRows(task: SplitGluTask) void {
    const Vec = @Vector(4, f32);
    const one: Vec = @splat(1);
    for (task.outer_start..task.outer_end) |outer_i| {
        const in_base = outer_i * task.axis_dim;
        const out_base = outer_i * task.half;
        var i: usize = 0;
        while (i + 4 <= task.half) : (i += 4) {
            const up: Vec = task.input[in_base + i ..][0..4].*;
            const gate: Vec = task.input[in_base + task.half + i ..][0..4].*;
            task.output[out_base + i ..][0..4].* = up * (one / (one + vexpf(4, -gate)));
        }
        while (i < task.half) : (i += 1) {
            const up = task.input[in_base + i];
            const gate = task.input[in_base + task.half + i];
            task.output[out_base + i] = up / (1 + @exp(-gate));
        }
    }
}

pub fn splitSwiGluBackwardRows(task: SplitSwiGluBackwardTask) void {
    const Vec = @Vector(4, f32);
    const one: Vec = @splat(1);
    for (task.outer_start..task.outer_end) |outer_i| {
        const in_base = outer_i * task.axis_dim;
        const grad_base = outer_i * task.half;
        var i: usize = 0;
        while (i + 4 <= task.half) : (i += 4) {
            const gate: Vec = task.input[in_base + i ..][0..4].*;
            const up: Vec = task.input[in_base + task.half + i ..][0..4].*;
            const grad: Vec = task.grad[grad_base + i ..][0..4].*;
            const sigmoid_value = one / (one + vexpf(4, -gate));
            const silu_value = gate * sigmoid_value;
            const silu_deriv = sigmoid_value * (one + gate * (one - sigmoid_value));
            task.output[in_base + i ..][0..4].* = grad * up * silu_deriv;
            task.output[in_base + task.half + i ..][0..4].* = grad * silu_value;
        }
        while (i < task.half) : (i += 1) {
            const gate = task.input[in_base + i];
            const up = task.input[in_base + task.half + i];
            const grad_value = task.grad[grad_base + i];
            const sigmoid_value = backend_ops.sigmoidScalar(gate);
            const silu_value = gate * sigmoid_value;
            const silu_deriv = sigmoid_value * (1 + gate * (1 - sigmoid_value));
            task.output[in_base + i] = grad_value * up * silu_deriv;
            task.output[in_base + task.half + i] = grad_value * silu_value;
        }
    }
}

pub fn splitGluBackwardRows(task: SplitGluBackwardTask) void {
    const Vec = @Vector(4, f32);
    const one: Vec = @splat(1);
    for (task.outer_start..task.outer_end) |outer_i| {
        const in_base = outer_i * task.axis_dim;
        const grad_base = outer_i * task.half;
        var i: usize = 0;
        while (i + 4 <= task.half) : (i += 4) {
            const up: Vec = task.input[in_base + i ..][0..4].*;
            const gate: Vec = task.input[in_base + task.half + i ..][0..4].*;
            const grad: Vec = task.grad[grad_base + i ..][0..4].*;
            const sigmoid_value = one / (one + vexpf(4, -gate));
            task.output[in_base + i ..][0..4].* = grad * sigmoid_value;
            task.output[in_base + task.half + i ..][0..4].* = grad * up * sigmoid_value * (one - sigmoid_value);
        }
        while (i < task.half) : (i += 1) {
            const up = task.input[in_base + i];
            const gate = task.input[in_base + task.half + i];
            const grad = task.grad[grad_base + i];
            const sigmoid_value = 1 / (1 + @exp(-gate));
            task.output[in_base + i] = grad * sigmoid_value;
            task.output[in_base + task.half + i] = grad * up * sigmoid_value * (1 - sigmoid_value);
        }
    }
}

pub fn rmsNormMulRopeHalfVectors(task: RmsNormMulRopeHalfTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.vector_start..task.vector_end) |vector_i| {
        var remainder = vector_i;
        var input_base: usize = task.input_offset;
        var output_base: usize = 0;
        var position_coord: usize = 0;
        var dim = task.rank;
        while (dim > 0) {
            dim -= 1;
            if (dim != task.feature_axis) {
                const coord = remainder % task.shape[dim];
                remainder /= task.shape[dim];
                input_base += coord * task.input_strides[dim];
                output_base += coord * task.output_strides[dim];
                if (dim == task.position_axis) position_coord = coord;
            }
        }

        var feature_i: usize = 0;
        var sumsq_vec: Vec = @splat(0);
        while (feature_i + vector_width <= task.feature_dim) : (feature_i += vector_width) {
            const values: Vec = task.input[input_base + feature_i ..][0..vector_width].*;
            sumsq_vec += values * values;
        }
        var sumsq: f32 = @reduce(.Add, sumsq_vec);
        while (feature_i < task.feature_dim) : (feature_i += 1) {
            const value = task.input[input_base + feature_i];
            sumsq += value * value;
        }
        const rms_scale = 1 / @sqrt(sumsq * task.inv_feature_dim + task.eps);
        const scale_vec: Vec = @splat(rms_scale);

        var pair_i: usize = 0;
        while (pair_i + vector_width <= task.pair_count) : (pair_i += vector_width) {
            const angle_i = position_coord * task.pair_count + pair_i;
            const sin_vec: Vec = task.sin_values[angle_i..][0..vector_width].*;
            const cos_vec: Vec = task.cos_values[angle_i..][0..vector_width].*;
            const first = @as(Vec, task.input[input_base + pair_i ..][0..vector_width].*) * scale_vec * @as(Vec, task.weights[pair_i..][0..vector_width].*);
            const second = @as(Vec, task.input[input_base + pair_i + task.pair_count ..][0..vector_width].*) * scale_vec * @as(Vec, task.weights[pair_i + task.pair_count ..][0..vector_width].*);
            task.output[output_base + pair_i ..][0..vector_width].* = first * cos_vec - second * sin_vec;
            task.output[output_base + pair_i + task.pair_count ..][0..vector_width].* = first * sin_vec + second * cos_vec;
        }
        while (pair_i < task.pair_count) : (pair_i += 1) {
            const angle_i = position_coord * task.pair_count + pair_i;
            const first = task.input[input_base + pair_i] * rms_scale * task.weights[pair_i];
            const second = task.input[input_base + pair_i + task.pair_count] * rms_scale * task.weights[pair_i + task.pair_count];
            task.output[output_base + pair_i] = first * task.cos_values[angle_i] - second * task.sin_values[angle_i];
            task.output[output_base + pair_i + task.pair_count] = first * task.sin_values[angle_i] + second * task.cos_values[angle_i];
        }
    }
}

pub fn rmsNormMulRows(task: RmsNormMulRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row_i| {
        const base = row_i * task.axis_dim;
        var axis_i: usize = 0;
        var sumsq_vec: Vec = @splat(0);
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const values: Vec = task.input[base + axis_i ..][0..vector_width].*;
            sumsq_vec += values * values;
        }
        var sumsq: f32 = @reduce(.Add, sumsq_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            const value = task.input[base + axis_i];
            sumsq += value * value;
        }
        const scale_value = 1 / @sqrt(sumsq * task.inv_axis_dim + task.eps);
        const scale_vec: Vec = @splat(scale_value);

        axis_i = 0;
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const values: Vec = task.input[base + axis_i ..][0..vector_width].*;
            const weights: Vec = task.weights[axis_i..][0..vector_width].*;
            task.output[base + axis_i ..][0..vector_width].* = values * scale_vec * weights;
        }
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            task.output[base + axis_i] = task.input[base + axis_i] * scale_value * task.weights[axis_i];
        }
    }
}

pub fn rmsNormMulAddRows(task: RmsNormMulAddRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row_i| {
        const base = row_i * task.axis_dim;
        var axis_i: usize = 0;
        var sumsq_vec: Vec = @splat(0);
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const values: Vec = task.input[base + axis_i ..][0..vector_width].*;
            sumsq_vec += values * values;
        }
        var sumsq: f32 = @reduce(.Add, sumsq_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            const value = task.input[base + axis_i];
            sumsq += value * value;
        }
        const scale_value = 1 / @sqrt(sumsq * task.inv_axis_dim + task.eps);
        const scale_vec: Vec = @splat(scale_value);

        axis_i = 0;
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const values: Vec = task.input[base + axis_i ..][0..vector_width].*;
            const weights: Vec = task.weights[axis_i..][0..vector_width].*;
            const residual: Vec = task.residual[base + axis_i ..][0..vector_width].*;
            task.output[base + axis_i ..][0..vector_width].* = residual + values * scale_vec * weights;
        }
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            task.output[base + axis_i] = task.residual[base + axis_i] + task.input[base + axis_i] * scale_value * task.weights[axis_i];
        }
    }
}

pub fn rmsNormMulBackwardInputRows(task: RmsNormMulBackwardInputRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row_i| {
        const base = row_i * task.axis_dim;
        var axis_i: usize = 0;
        var sumsq_vec: Vec = @splat(0);
        var dot_vec: Vec = @splat(0);
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const values: Vec = task.input[base + axis_i ..][0..vector_width].*;
            const grad: Vec = task.grad[base + axis_i ..][0..vector_width].*;
            const weights: Vec = task.weights[axis_i..][0..vector_width].*;
            sumsq_vec += values * values;
            dot_vec += grad * weights * values;
        }
        var sumsq: f32 = @reduce(.Add, sumsq_vec);
        var dot_acc: f32 = @reduce(.Add, dot_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            const value = task.input[base + axis_i];
            sumsq += value * value;
            dot_acc += task.grad[base + axis_i] * task.weights[axis_i] * value;
        }

        const rms_scale = 1 / @sqrt(sumsq * task.inv_axis_dim + task.eps);
        const correction_scale = rms_scale * rms_scale * rms_scale * task.inv_axis_dim * dot_acc;
        const rms_vec: Vec = @splat(rms_scale);
        const correction_vec: Vec = @splat(correction_scale);

        axis_i = 0;
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const values: Vec = task.input[base + axis_i ..][0..vector_width].*;
            const grad: Vec = task.grad[base + axis_i ..][0..vector_width].*;
            const weights: Vec = task.weights[axis_i..][0..vector_width].*;
            task.output[base + axis_i ..][0..vector_width].* = grad * weights * rms_vec - values * correction_vec;
        }
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            task.output[base + axis_i] = task.grad[base + axis_i] * task.weights[axis_i] * rms_scale - task.input[base + axis_i] * correction_scale;
        }
    }
}

pub fn rmsNormMulBackwardWeightRows(task: RmsNormMulBackwardWeightRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row_i| {
        const base = row_i * task.axis_dim;
        var axis_i: usize = 0;
        var sumsq_vec: Vec = @splat(0);
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const values: Vec = task.input[base + axis_i ..][0..vector_width].*;
            sumsq_vec += values * values;
        }
        var sumsq: f32 = @reduce(.Add, sumsq_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            const value = task.input[base + axis_i];
            sumsq += value * value;
        }

        const rms_scale = 1 / @sqrt(sumsq * task.inv_axis_dim + task.eps);
        const rms_vec: Vec = @splat(rms_scale);
        axis_i = 0;
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const current: Vec = task.output[axis_i..][0..vector_width].*;
            const values: Vec = task.input[base + axis_i ..][0..vector_width].*;
            const grad: Vec = task.grad[base + axis_i ..][0..vector_width].*;
            task.output[axis_i..][0..vector_width].* = current + grad * values * rms_vec;
        }
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            task.output[axis_i] += task.grad[base + axis_i] * task.input[base + axis_i] * rms_scale;
        }
    }
}

pub fn layerNormRows(task: LayerNormRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row_i| {
        const base = row_i * task.axis_dim;
        const row_in = task.input[base..][0..task.axis_dim];
        const row_out = task.output[base..][0..task.axis_dim];

        // Pass 1: row mean.
        var axis_i: usize = 0;
        var sum_vec: Vec = @splat(0);
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            sum_vec += @as(Vec, row_in[axis_i..][0..vector_width].*);
        }
        var sum = @reduce(.Add, sum_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            sum += row_in[axis_i];
        }
        const mean_value = sum * task.inv_axis_dim;
        const mean_vec: Vec = @splat(mean_value);

        // Pass 2 (ggml_norm-style): write centered values, accumulate the
        // centered sum of squares — the BIASED variance numerator.
        axis_i = 0;
        var sumsq_vec: Vec = @splat(0);
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const centered = @as(Vec, row_in[axis_i..][0..vector_width].*) - mean_vec;
            row_out[axis_i..][0..vector_width].* = centered;
            sumsq_vec += centered * centered;
        }
        var sumsq = @reduce(.Add, sumsq_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            const centered = row_in[axis_i] - mean_value;
            row_out[axis_i] = centered;
            sumsq += centered * centered;
        }
        const inv_sigma = 1 / @sqrt(sumsq * task.inv_axis_dim + task.eps);
        const sigma_vec: Vec = @splat(inv_sigma);

        // Pass 3: scale (and the fused affine weight/bias when present).
        axis_i = 0;
        if (task.weights) |weights| {
            const biases = task.biases.?;
            while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
                const w: Vec = weights[axis_i..][0..vector_width].*;
                const b: Vec = biases[axis_i..][0..vector_width].*;
                row_out[axis_i..][0..vector_width].* = @as(Vec, row_out[axis_i..][0..vector_width].*) * sigma_vec * w + b;
            }
            while (axis_i < task.axis_dim) : (axis_i += 1) {
                row_out[axis_i] = row_out[axis_i] * inv_sigma * weights[axis_i] + biases[axis_i];
            }
        } else {
            while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
                row_out[axis_i..][0..vector_width].* = @as(Vec, row_out[axis_i..][0..vector_width].*) * sigma_vec;
            }
            while (axis_i < task.axis_dim) : (axis_i += 1) {
                row_out[axis_i] *= inv_sigma;
            }
        }
    }
}

pub fn layerNormBackwardInputRows(task: LayerNormBackwardInputRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row_i| {
        const base = row_i * task.axis_dim;
        const row_in = task.input[base..][0..task.axis_dim];
        const row_gy = task.grad[base..][0..task.axis_dim];
        const row_out = task.output[base..][0..task.axis_dim];

        // Pass 1: row mean of x and the sum of g' = gy [* weight].
        var axis_i: usize = 0;
        var sum_vec: Vec = @splat(0);
        var gsum_vec: Vec = @splat(0);
        if (task.weights) |weights| {
            while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
                sum_vec += @as(Vec, row_in[axis_i..][0..vector_width].*);
                gsum_vec += @as(Vec, row_gy[axis_i..][0..vector_width].*) * @as(Vec, weights[axis_i..][0..vector_width].*);
            }
        } else {
            while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
                sum_vec += @as(Vec, row_in[axis_i..][0..vector_width].*);
                gsum_vec += @as(Vec, row_gy[axis_i..][0..vector_width].*);
            }
        }
        var sum = @reduce(.Add, sum_vec);
        var gsum = @reduce(.Add, gsum_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            sum += row_in[axis_i];
            gsum += row_gy[axis_i] * (if (task.weights) |weights| weights[axis_i] else 1);
        }
        const mean_value = sum * task.inv_axis_dim;
        const mean_vec: Vec = @splat(mean_value);

        // Pass 2: centered sum of squares (biased variance numerator) and
        // the centered dot Σ g'·(x−μ).
        axis_i = 0;
        var sumsq_vec: Vec = @splat(0);
        var dot_vec: Vec = @splat(0);
        if (task.weights) |weights| {
            while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
                const centered = @as(Vec, row_in[axis_i..][0..vector_width].*) - mean_vec;
                sumsq_vec += centered * centered;
                dot_vec += @as(Vec, row_gy[axis_i..][0..vector_width].*) * @as(Vec, weights[axis_i..][0..vector_width].*) * centered;
            }
        } else {
            while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
                const centered = @as(Vec, row_in[axis_i..][0..vector_width].*) - mean_vec;
                sumsq_vec += centered * centered;
                dot_vec += @as(Vec, row_gy[axis_i..][0..vector_width].*) * centered;
            }
        }
        var sumsq = @reduce(.Add, sumsq_vec);
        var dot_acc = @reduce(.Add, dot_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            const centered = row_in[axis_i] - mean_value;
            sumsq += centered * centered;
            dot_acc += row_gy[axis_i] * (if (task.weights) |weights| weights[axis_i] else 1) * centered;
        }

        // dx = (1/σ)(g' − mean(g') − x̂·mean(g'·x̂)) with x̂ = (x−μ)/σ,
        // rearranged per element to dx_i = g'_i/σ − shift − (x_i−μ)·correction.
        const inv_sigma = 1 / @sqrt(sumsq * task.inv_axis_dim + task.eps);
        const shift = gsum * task.inv_axis_dim * inv_sigma;
        const correction = dot_acc * task.inv_axis_dim * inv_sigma * inv_sigma * inv_sigma;
        const sigma_vec: Vec = @splat(inv_sigma);
        const shift_vec: Vec = @splat(shift);
        const correction_vec: Vec = @splat(correction);

        // Pass 3: write dx.
        axis_i = 0;
        if (task.weights) |weights| {
            while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
                const upstream = @as(Vec, row_gy[axis_i..][0..vector_width].*) * @as(Vec, weights[axis_i..][0..vector_width].*);
                const centered = @as(Vec, row_in[axis_i..][0..vector_width].*) - mean_vec;
                row_out[axis_i..][0..vector_width].* = upstream * sigma_vec - shift_vec - centered * correction_vec;
            }
            while (axis_i < task.axis_dim) : (axis_i += 1) {
                row_out[axis_i] = row_gy[axis_i] * weights[axis_i] * inv_sigma - shift - (row_in[axis_i] - mean_value) * correction;
            }
        } else {
            while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
                const upstream: Vec = row_gy[axis_i..][0..vector_width].*;
                const centered = @as(Vec, row_in[axis_i..][0..vector_width].*) - mean_vec;
                row_out[axis_i..][0..vector_width].* = upstream * sigma_vec - shift_vec - centered * correction_vec;
            }
            while (axis_i < task.axis_dim) : (axis_i += 1) {
                row_out[axis_i] = row_gy[axis_i] * inv_sigma - shift - (row_in[axis_i] - mean_value) * correction;
            }
        }
    }
}

/// Serial dweight/dbias fallback for the affine LayerNorm backward (no pool /
/// small inputs): one pass over the rows in row order, SIMD within each row.
/// Deliberately NOT row-task-parallel: per-task partial buffers combined in
/// task order (the rmsNormMulBackwardWeight pattern) change the float
/// accumulation order when the task count changes, so results would differ
/// across thread counts. Large inputs instead take the column-partitioned
/// parallel path (layerNormParamGradColumns), which keeps the same per-column
/// row-order accumulation and the same per-element expressions — so both
/// paths produce bitwise-identical dweight/dbias for any thread count.
pub fn layerNormAffineParamGradRows(
    input: []const f32,
    grad: []const f32,
    dweight: ?[]f32,
    dbias: ?[]f32,
    rows: usize,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (0..rows) |row_i| {
        const base = row_i * axis_dim;
        const row_in = input[base..][0..axis_dim];
        const row_gy = grad[base..][0..axis_dim];

        if (dweight) |weight_out| {
            // dweight needs x̂, so recompute the row statistics (mean, then
            // the centered sum of squares — same two-pass as the forward).
            var axis_i: usize = 0;
            var sum_vec: Vec = @splat(0);
            while (axis_i + vector_width <= axis_dim) : (axis_i += vector_width) {
                sum_vec += @as(Vec, row_in[axis_i..][0..vector_width].*);
            }
            var sum = @reduce(.Add, sum_vec);
            while (axis_i < axis_dim) : (axis_i += 1) {
                sum += row_in[axis_i];
            }
            const mean_value = sum * inv_axis_dim;
            const mean_vec: Vec = @splat(mean_value);

            axis_i = 0;
            var sumsq_vec: Vec = @splat(0);
            while (axis_i + vector_width <= axis_dim) : (axis_i += vector_width) {
                const centered = @as(Vec, row_in[axis_i..][0..vector_width].*) - mean_vec;
                sumsq_vec += centered * centered;
            }
            var sumsq = @reduce(.Add, sumsq_vec);
            while (axis_i < axis_dim) : (axis_i += 1) {
                const centered = row_in[axis_i] - mean_value;
                sumsq += centered * centered;
            }
            const inv_sigma = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            const sigma_vec: Vec = @splat(inv_sigma);

            // dweight += gy ⊙ x̂ (and dbias += gy in the same pass). The
            // per-column accumulation order is the row order — independent of
            // the vector width split across columns.
            axis_i = 0;
            if (dbias) |bias_out| {
                while (axis_i + vector_width <= axis_dim) : (axis_i += vector_width) {
                    const gy_vec: Vec = row_gy[axis_i..][0..vector_width].*;
                    const normalized = (@as(Vec, row_in[axis_i..][0..vector_width].*) - mean_vec) * sigma_vec;
                    weight_out[axis_i..][0..vector_width].* = @as(Vec, weight_out[axis_i..][0..vector_width].*) + gy_vec * normalized;
                    bias_out[axis_i..][0..vector_width].* = @as(Vec, bias_out[axis_i..][0..vector_width].*) + gy_vec;
                }
                while (axis_i < axis_dim) : (axis_i += 1) {
                    weight_out[axis_i] += row_gy[axis_i] * ((row_in[axis_i] - mean_value) * inv_sigma);
                    bias_out[axis_i] += row_gy[axis_i];
                }
            } else {
                while (axis_i + vector_width <= axis_dim) : (axis_i += vector_width) {
                    const gy_vec: Vec = row_gy[axis_i..][0..vector_width].*;
                    const normalized = (@as(Vec, row_in[axis_i..][0..vector_width].*) - mean_vec) * sigma_vec;
                    weight_out[axis_i..][0..vector_width].* = @as(Vec, weight_out[axis_i..][0..vector_width].*) + gy_vec * normalized;
                }
                while (axis_i < axis_dim) : (axis_i += 1) {
                    weight_out[axis_i] += row_gy[axis_i] * ((row_in[axis_i] - mean_value) * inv_sigma);
                }
            }
        } else if (dbias) |bias_out| {
            // Bias-only: a plain column sum of gy — no row statistics needed.
            var axis_i: usize = 0;
            while (axis_i + vector_width <= axis_dim) : (axis_i += vector_width) {
                bias_out[axis_i..][0..vector_width].* = @as(Vec, bias_out[axis_i..][0..vector_width].*) + @as(Vec, row_gy[axis_i..][0..vector_width].*);
            }
            while (axis_i < axis_dim) : (axis_i += 1) {
                bias_out[axis_i] += row_gy[axis_i];
            }
        }
    }
}

pub fn layerNormRowStats(task: LayerNormRowStatsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row_i| {
        const row_in = task.input[row_i * task.axis_dim ..][0..task.axis_dim];

        var axis_i: usize = 0;
        var sum_vec: Vec = @splat(0);
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            sum_vec += @as(Vec, row_in[axis_i..][0..vector_width].*);
        }
        var sum_acc = @reduce(.Add, sum_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            sum_acc += row_in[axis_i];
        }
        const mean_value = sum_acc * task.inv_axis_dim;
        const mean_vec: Vec = @splat(mean_value);

        axis_i = 0;
        var sumsq_vec: Vec = @splat(0);
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const centered = @as(Vec, row_in[axis_i..][0..vector_width].*) - mean_vec;
            sumsq_vec += centered * centered;
        }
        var sumsq = @reduce(.Add, sumsq_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            const centered = row_in[axis_i] - mean_value;
            sumsq += centered * centered;
        }
        task.stats[2 * row_i] = mean_value;
        task.stats[2 * row_i + 1] = 1 / @sqrt(sumsq * task.inv_axis_dim + task.eps);
    }
}

/// Column-partitioned dweight/dbias accumulation (the large-input path): the
/// task accumulates its own column range over ALL rows in row order, reading
/// per-row {mean, 1/σ} from the precomputed stats scratch. The vector body
/// and the scalar tail evaluate the exact same per-element expression
/// gy·((x−μ)·(1/σ)), so a column produces the same bits whether it lands in
/// a vector lane or the tail — the column split can move with the task count
/// without changing any result bit.
pub fn layerNormParamGradColumns(task: LayerNormParamGradColumnsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (0..task.rows) |row_i| {
        const base = row_i * task.axis_dim;

        var col = task.col_start;
        if (task.dweight) |dweight| {
            // Stats are only filled (and only needed) when dweight exists;
            // bias-only runs skip the stats pass entirely.
            const mean_value = task.stats[2 * row_i];
            const inv_sigma = task.stats[2 * row_i + 1];
            const mean_vec: Vec = @splat(mean_value);
            const sigma_vec: Vec = @splat(inv_sigma);
            if (task.dbias) |dbias| {
                while (col + vector_width <= task.col_end) : (col += vector_width) {
                    const gy_vec: Vec = task.grad[base + col ..][0..vector_width].*;
                    const normalized = (@as(Vec, task.input[base + col ..][0..vector_width].*) - mean_vec) * sigma_vec;
                    dweight[col..][0..vector_width].* = @as(Vec, dweight[col..][0..vector_width].*) + gy_vec * normalized;
                    dbias[col..][0..vector_width].* = @as(Vec, dbias[col..][0..vector_width].*) + gy_vec;
                }
                while (col < task.col_end) : (col += 1) {
                    dweight[col] += task.grad[base + col] * ((task.input[base + col] - mean_value) * inv_sigma);
                    dbias[col] += task.grad[base + col];
                }
            } else {
                while (col + vector_width <= task.col_end) : (col += vector_width) {
                    const gy_vec: Vec = task.grad[base + col ..][0..vector_width].*;
                    const normalized = (@as(Vec, task.input[base + col ..][0..vector_width].*) - mean_vec) * sigma_vec;
                    dweight[col..][0..vector_width].* = @as(Vec, dweight[col..][0..vector_width].*) + gy_vec * normalized;
                }
                while (col < task.col_end) : (col += 1) {
                    dweight[col] += task.grad[base + col] * ((task.input[base + col] - mean_value) * inv_sigma);
                }
            }
        } else if (task.dbias) |dbias| {
            while (col + vector_width <= task.col_end) : (col += vector_width) {
                dbias[col..][0..vector_width].* = @as(Vec, dbias[col..][0..vector_width].*) + @as(Vec, task.grad[base + col ..][0..vector_width].*);
            }
            while (col < task.col_end) : (col += 1) {
                dbias[col] += task.grad[base + col];
            }
        }
    }
}

pub fn softmaxRows(task: SoftmaxRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row_i| {
        const base = row_i * task.axis_dim;
        const row_in = task.input[base..][0..task.axis_dim];
        const row_out = task.output[base..][0..task.axis_dim];

        var axis_i: usize = 0;
        var max_vec: Vec = @splat(-std.math.inf(f32));
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            max_vec = @max(max_vec, @as(Vec, row_in[axis_i..][0..vector_width].*));
        }
        var max_value = @reduce(.Max, max_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            max_value = @max(max_value, row_in[axis_i]);
        }

        const max_splat: Vec = @splat(max_value);
        var sum_vec: Vec = @splat(0);
        axis_i = 0;
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const value = vexpf(vector_width, @as(Vec, row_in[axis_i..][0..vector_width].*) - max_splat);
            row_out[axis_i..][0..vector_width].* = value;
            sum_vec += value;
        }
        var sum_exp = @reduce(.Add, sum_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            const value = vexpf(1, @splat(row_in[axis_i] - max_value))[0];
            row_out[axis_i] = value;
            sum_exp += value;
        }

        const inv_sum = 1 / sum_exp;
        const inv_vec: Vec = @splat(inv_sum);
        axis_i = 0;
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            row_out[axis_i..][0..vector_width].* = @as(Vec, row_out[axis_i..][0..vector_width].*) * inv_vec;
        }
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            row_out[axis_i] *= inv_sum;
        }
    }
}

pub fn softmaxExtRows(comptime rank: usize, comptime axis: usize, task: SoftmaxExtRowsTask(rank)) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row| {
        const outer_i = row / task.inner;
        const inner_i = row % task.inner;
        const base = outer_i * task.axis_dim * task.inner;
        const row_linear = base + inner_i;
        const head_i = if (task.head_axis) |head_axis| coordinateForLinear(rank, task.shape, task.strides, row_linear, head_axis) else 0;
        const slope = if (task.slopes) |values| values[head_i] else 1;
        const causal_query_i = if (task.causal_query_axis) |query_axis| coordinateForLinear(rank, task.shape, task.strides, row_linear, query_axis) else null;
        const active_axis_dim = if (causal_query_i) |query_i| task.causal_source_offset + query_i + 1 else task.axis_dim;
        const mask_base = if (task.mask) |mask| physicalOffsetExcludingAxis(rank, task.shape, task.strides, mask.strides, mask.tensor.offset, row_linear, axis) else 0;
        const mask_axis_stride = if (task.mask) |mask| mask.strides[axis] else 0;

        if (task.simd_rows) {
            const row_in = task.input[base..][0..task.axis_dim];
            const row_out = task.output[base..][0..task.axis_dim];
            const mask_row: ?[]const f32 = if (task.mask) |mask| mask.tensor.buffer.data[mask_base..] else null;
            const scale_vec: Vec = @splat(task.scale);
            const slope_vec: Vec = @splat(slope);

            var axis_i: usize = 0;
            var max_vec: Vec = @splat(-std.math.inf(f32));
            var max_value = -std.math.inf(f32);
            if (mask_row) |mask_values| {
                while (axis_i + vector_width <= active_axis_dim) : (axis_i += vector_width) {
                    const pre = @as(Vec, row_in[axis_i..][0..vector_width].*) * scale_vec + slope_vec * @as(Vec, mask_values[axis_i..][0..vector_width].*);
                    max_vec = @max(max_vec, pre);
                }
                max_value = @reduce(.Max, max_vec);
                while (axis_i < active_axis_dim) : (axis_i += 1) {
                    max_value = @max(max_value, row_in[axis_i] * task.scale + slope * mask_values[axis_i]);
                }
            } else {
                while (axis_i + vector_width <= active_axis_dim) : (axis_i += vector_width) {
                    max_vec = @max(max_vec, @as(Vec, row_in[axis_i..][0..vector_width].*) * scale_vec);
                }
                max_value = @reduce(.Max, max_vec);
                while (axis_i < active_axis_dim) : (axis_i += 1) {
                    max_value = @max(max_value, row_in[axis_i] * task.scale);
                }
            }
            if (task.sinks) |sinks| {
                max_value = @max(max_value, sinks[head_i]);
            }

            const max_splat: Vec = @splat(max_value);
            var sum_vec: Vec = @splat(0);
            var sum_exp: f32 = 0;
            axis_i = 0;
            if (mask_row) |mask_values| {
                while (axis_i + vector_width <= active_axis_dim) : (axis_i += vector_width) {
                    const pre = @as(Vec, row_in[axis_i..][0..vector_width].*) * scale_vec + slope_vec * @as(Vec, mask_values[axis_i..][0..vector_width].*);
                    const value = vexpf(vector_width, pre - max_splat);
                    row_out[axis_i..][0..vector_width].* = value;
                    sum_vec += value;
                }
                sum_exp = @reduce(.Add, sum_vec);
                while (axis_i < active_axis_dim) : (axis_i += 1) {
                    const value = vexpf(1, @splat(row_in[axis_i] * task.scale + slope * mask_values[axis_i] - max_value))[0];
                    row_out[axis_i] = value;
                    sum_exp += value;
                }
            } else {
                while (axis_i + vector_width <= active_axis_dim) : (axis_i += vector_width) {
                    const value = vexpf(vector_width, @as(Vec, row_in[axis_i..][0..vector_width].*) * scale_vec - max_splat);
                    row_out[axis_i..][0..vector_width].* = value;
                    sum_vec += value;
                }
                sum_exp = @reduce(.Add, sum_vec);
                while (axis_i < active_axis_dim) : (axis_i += 1) {
                    const value = vexpf(1, @splat(row_in[axis_i] * task.scale - max_value))[0];
                    row_out[axis_i] = value;
                    sum_exp += value;
                }
            }
            if (task.sinks) |sinks| {
                sum_exp += @exp(sinks[head_i] - max_value);
            }

            const inv_sum = 1 / sum_exp;
            const inv_vec: Vec = @splat(inv_sum);
            axis_i = 0;
            while (axis_i + vector_width <= active_axis_dim) : (axis_i += vector_width) {
                row_out[axis_i..][0..vector_width].* = @as(Vec, row_out[axis_i..][0..vector_width].*) * inv_vec;
            }
            while (axis_i < active_axis_dim) : (axis_i += 1) {
                row_out[axis_i] *= inv_sum;
            }
            @memset(row_out[active_axis_dim..task.axis_dim], 0);
            continue;
        }

        // Strided / inner>1 rows: scalar body, unchanged from the serial kernel.
        var max_value = preSoftmaxValue(rank, task.input[base + inner_i], task.scale, task.mask, mask_base, mask_axis_stride, 0, slope);
        for (1..active_axis_dim) |axis_i| {
            const offset = base + axis_i * task.inner + inner_i;
            max_value = @max(max_value, preSoftmaxValue(rank, task.input[offset], task.scale, task.mask, mask_base, mask_axis_stride, axis_i, slope));
        }
        if (task.sinks) |sinks| {
            max_value = @max(max_value, sinks[head_i]);
        }

        var sum_exp: f32 = 0;
        for (0..active_axis_dim) |axis_i| {
            const offset = base + axis_i * task.inner + inner_i;
            const value = @exp(preSoftmaxValue(rank, task.input[offset], task.scale, task.mask, mask_base, mask_axis_stride, axis_i, slope) - max_value);
            task.output[offset] = value;
            sum_exp += value;
        }
        if (task.sinks) |sinks| {
            sum_exp += @exp(sinks[head_i] - max_value);
        }

        const inv_sum = 1 / sum_exp;
        for (0..active_axis_dim) |axis_i| {
            task.output[base + axis_i * task.inner + inner_i] *= inv_sum;
        }
        for (active_axis_dim..task.axis_dim) |axis_i| {
            task.output[base + axis_i * task.inner + inner_i] = 0;
        }
    }
}

pub fn softmaxBackwardRows(task: SoftmaxBackwardRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;

    for (task.row_start..task.row_end) |row_i| {
        const base = row_i * task.axis_dim;
        const row_y = task.y[base..][0..task.axis_dim];
        const row_gy = task.gy[base..][0..task.axis_dim];
        const row_out = task.output[base..][0..task.axis_dim];

        var axis_i: usize = 0;
        var dot_vec: Vec = @splat(0);
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            dot_vec += @as(Vec, row_gy[axis_i..][0..vector_width].*) * @as(Vec, row_y[axis_i..][0..vector_width].*);
        }
        var dot_acc = @reduce(.Add, dot_vec);
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            dot_acc += row_gy[axis_i] * row_y[axis_i];
        }

        const scale_vec: Vec = @splat(task.scale);
        const dot_splat: Vec = @splat(dot_acc);
        axis_i = 0;
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            const yv: Vec = row_y[axis_i..][0..vector_width].*;
            const gyv: Vec = row_gy[axis_i..][0..vector_width].*;
            row_out[axis_i..][0..vector_width].* = scale_vec * yv * (gyv - dot_splat);
        }
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            row_out[axis_i] = task.scale * row_y[axis_i] * (row_gy[axis_i] - dot_acc);
        }
    }
}

pub fn crossEntropyLossRows(task: CrossEntropyLossRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    const eps = task.label_smoothing;
    const eps_uniform = eps / @as(f32, @floatFromInt(task.class_count));

    for (task.row_start..task.row_end) |row_i| {
        const label = task.labels[row_i];
        if (task.ignore_index) |ignore_index| {
            if (label == ignore_index) {
                task.row_losses[row_i] = 0;
                if (task.row_stats) |stats| {
                    stats[2 * row_i] = 0;
                    stats[2 * row_i + 1] = 1;
                }
                continue;
            }
        }
        const row_in = task.input[row_i * task.class_count ..][0..task.class_count];

        var class_i: usize = 0;
        var max_vec: Vec = @splat(-std.math.inf(f32));
        while (class_i + vector_width <= task.class_count) : (class_i += vector_width) {
            max_vec = @max(max_vec, @as(Vec, row_in[class_i..][0..vector_width].*));
        }
        var max_value = @reduce(.Max, max_vec);
        while (class_i < task.class_count) : (class_i += 1) {
            max_value = @max(max_value, row_in[class_i]);
        }

        const max_splat: Vec = @splat(max_value);
        var sum_vec: Vec = @splat(0);
        class_i = 0;
        while (class_i + vector_width <= task.class_count) : (class_i += vector_width) {
            sum_vec += vexpf(vector_width, @as(Vec, row_in[class_i..][0..vector_width].*) - max_splat);
        }
        var sum_exp = @reduce(.Add, sum_vec);
        while (class_i < task.class_count) : (class_i += 1) {
            sum_exp += vexpf(1, @splat(row_in[class_i] - max_value))[0];
        }
        if (task.row_stats) |stats| {
            stats[2 * row_i] = max_value;
            stats[2 * row_i + 1] = sum_exp;
        }

        var loss = @log(sum_exp) + max_value - (1 - eps) * row_in[label];
        if (eps > 0) {
            var logit_vec: Vec = @splat(0);
            class_i = 0;
            while (class_i + vector_width <= task.class_count) : (class_i += vector_width) {
                logit_vec += @as(Vec, row_in[class_i..][0..vector_width].*);
            }
            var logit_sum = @reduce(.Add, logit_vec);
            while (class_i < task.class_count) : (class_i += 1) {
                logit_sum += row_in[class_i];
            }
            loss -= eps_uniform * logit_sum;
        }
        task.row_losses[row_i] = loss;
    }
}

pub fn crossEntropyBackwardRows(task: CrossEntropyBackwardRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    const eps = task.label_smoothing;
    const eps_uniform = eps / @as(f32, @floatFromInt(task.class_count));

    for (task.row_start..task.row_end) |row_i| {
        const label = task.labels[row_i];
        const row_out = task.output[row_i * task.class_count ..][0..task.class_count];
        if (task.ignore_index) |ignore_index| {
            if (label == ignore_index) {
                @memset(row_out, 0);
                continue;
            }
        }
        const row_scale = if (task.per_row_scale) |values| task.grad_common * values[row_i] else task.grad_common;
        const row_in = task.input[row_i * task.class_count ..][0..task.class_count];

        if (task.row_stats) |stats| {
            // Forward-saved stats: emit final gradients in one pass — same
            // f32 values and per-element op order as the recompute path
            // below (the exp is multiplied in-register instead of after a
            // store/load round-trip, which is value-preserving for f32).
            const max_value = stats[2 * row_i];
            const sum_exp = stats[2 * row_i + 1];
            const prob_scale = row_scale / sum_exp;
            const smooth_term = eps_uniform * row_scale;
            const stat_max_splat: Vec = @splat(max_value);
            const prob_scale_vec: Vec = @splat(prob_scale);
            const smooth_vec: Vec = @splat(smooth_term);
            var class_i: usize = 0;
            while (class_i + vector_width <= task.class_count) : (class_i += vector_width) {
                const value = vexpf(vector_width, @as(Vec, row_in[class_i..][0..vector_width].*) - stat_max_splat);
                row_out[class_i..][0..vector_width].* = value * prob_scale_vec - smooth_vec;
            }
            while (class_i < task.class_count) : (class_i += 1) {
                const value = vexpf(1, @splat(row_in[class_i] - max_value))[0];
                row_out[class_i] = value * prob_scale - smooth_term;
            }
            row_out[label] -= (1 - eps) * row_scale;
            continue;
        }

        var class_i: usize = 0;
        var max_vec: Vec = @splat(-std.math.inf(f32));
        while (class_i + vector_width <= task.class_count) : (class_i += vector_width) {
            max_vec = @max(max_vec, @as(Vec, row_in[class_i..][0..vector_width].*));
        }
        var max_value = @reduce(.Max, max_vec);
        while (class_i < task.class_count) : (class_i += 1) {
            max_value = @max(max_value, row_in[class_i]);
        }

        const max_splat: Vec = @splat(max_value);
        var sum_vec: Vec = @splat(0);
        class_i = 0;
        while (class_i + vector_width <= task.class_count) : (class_i += vector_width) {
            const value = vexpf(vector_width, @as(Vec, row_in[class_i..][0..vector_width].*) - max_splat);
            row_out[class_i..][0..vector_width].* = value;
            sum_vec += value;
        }
        var sum_exp = @reduce(.Add, sum_vec);
        while (class_i < task.class_count) : (class_i += 1) {
            const value = vexpf(1, @splat(row_in[class_i] - max_value))[0];
            row_out[class_i] = value;
            sum_exp += value;
        }

        // grad_c = (p_c - (1-eps)*1{c==label} - eps/K) * row_scale
        const prob_scale = row_scale / sum_exp;
        const smooth_term = eps_uniform * row_scale;
        const prob_scale_vec: Vec = @splat(prob_scale);
        const smooth_vec: Vec = @splat(smooth_term);
        class_i = 0;
        while (class_i + vector_width <= task.class_count) : (class_i += vector_width) {
            row_out[class_i..][0..vector_width].* = @as(Vec, row_out[class_i..][0..vector_width].*) * prob_scale_vec - smooth_vec;
        }
        while (class_i < task.class_count) : (class_i += 1) {
            row_out[class_i] = row_out[class_i] * prob_scale - smooth_term;
        }
        row_out[label] -= (1 - eps) * row_scale;
    }
}

pub fn dropoutRange(task: DropoutRangeTask) void {
    for (task.start..task.end) |i| {
        task.output[i] = if (rng.at(task.seed, i) >> 11 < task.keep_cutoff) task.input[i] * task.scale else 0;
    }
}

pub fn scatterAddRows(task: ScatterAddRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    const row_len = task.row_len;

    @memset(task.output[task.row_start * row_len .. task.row_end * row_len], 0);
    for (task.indices, 0..) |index, row| {
        if (index < task.row_start or index >= task.row_end) continue;
        const src = task.grad[row * row_len ..][0..row_len];
        const dst = task.output[index * row_len ..][0..row_len];
        var i: usize = 0;
        while (i + vector_width <= row_len) : (i += vector_width) {
            dst[i..][0..vector_width].* = @as(Vec, dst[i..][0..vector_width].*) + @as(Vec, src[i..][0..vector_width].*);
        }
        while (i < row_len) : (i += 1) {
            dst[i] += src[i];
        }
    }
}
