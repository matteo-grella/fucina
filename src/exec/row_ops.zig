//! Row-kernel task bodies: the worker-thread forms of the fused row passes
//! (softmax/log-softmax/logsumexp rows, layer/RMS-norm rows and their
//! backward stats, cross-entropy rows, dropout, scatter-add, gated
//! activations, and the fused activation+re-quantize passes the quantized
//! matmuls chain onto). The `Task` structs here are dispatched by the
//! domain modules that own the ops (`softmax.zig`, `norm.zig`, `loss.zig`,
//! `quant_matmul.zig`, ...); this file owns the per-row loop bodies they
//! share, not the ops themselves.
const std = @import("std");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const kernels = backend_mod.kernels;
const backend_ops = backend_mod.ops;
const rng = @import("../rng.zig");
const tensor = @import("../tensor.zig");
const shape = @import("shape.zig");

const vexpf = backend_mod.simd.vexpf;
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
        x4_blocks: []backend_mod.quantized_matmul.BlockQ8_Kx4 = &.{},
        row_blocks: []dtype_mod.BlockQ8_K = &.{},
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
                        kernels.unaryRowSlice(.gelu_quant, dst, task.gate[(row0 + r) * cols ..][0..cols]);
                        kernels.mulRowSlice(dst, dst, task.up[(row0 + r) * cols ..][0..cols]);
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
                    .q8_kx4 => backend_mod.quantized_matmul.q8k.quantizeRowGroupQ8_Kx4Into(
                        task.x4_blocks[row_group * task.blocks_per_row ..][0..task.blocks_per_row],
                        group_scratch,
                        rows_in_group,
                        cols,
                    ),
                    .q8_k_rows => for (0..rows_in_group) |r| {
                        backend_mod.quantized_matmul.q8k.quantizeRowQ8_KIntoUnchecked(
                            task.row_blocks[(row0 + r) * task.blocks_per_row ..][0..task.blocks_per_row],
                            task.scratch[r * cols ..][0..cols],
                        );
                    },
                    .q8_0x4 => {
                        // The plain group packer reads all 4 lanes; zero rows
                        // quantize to d=0/qs=0, matching padded-lane semantics.
                        if (rows_in_group < 4) @memset(task.scratch[rows_in_group * cols .. 4 * cols], 0);
                        backend_mod.quantized_matmul.q8_0.quantizeRowsQ8_0x4GroupsInto(
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

pub const RmsNormRowStatsTask = struct {
    input: []const f32,
    /// Per-row 1/rms (`[rows]`): disjoint writes by row, each a pure
    /// function of its row through the same sum-of-squares tree as
    /// `rmsNormMulBackwardWeightRows`, so the scratch is bitwise identical
    /// for any thread count and to the serial row kernel.
    stats: []f32,
    axis_dim: usize,
    inv_axis_dim: f32,
    eps: f32,
    row_start: usize,
    row_end: usize,
};

pub const RmsNormWeightGradColumnsTask = struct {
    input: []const f32,
    grad: []const f32,
    stats: []const f32,
    // Each task owns the contiguous DESTINATION column range
    // [col_start, col_end) and accumulates over ALL rows in row order (the
    // LayerNormParamGradColumnsTask pattern): per-column accumulation
    // order equals the serial row kernel's, so dweight is bitwise
    // identical for any thread count.
    dweight: []f32,
    rows: usize,
    axis_dim: usize,
    col_start: usize,
    col_end: usize,
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

pub const DistillStatsRowsTask = struct {
    input: []const f32,
    // Per-row softmax statistics {max, sum_exp} interleaved (the
    // CrossEntropyLossRowsTask layout) — disjoint writes across tasks, so
    // the result is bitwise identical for any thread count.
    row_stats: []f32,
    class_count: usize,
    row_start: usize,
    row_end: usize,
};

pub fn runDistillStatsRowsTask(task: *const DistillStatsRowsTask) void {
    distillStatsRows(task.*);
}

/// Per-row {max, sum_exp} over the class axis — the softmax statistics the
/// sparse-soft-target losses need (no labels: every row in range counts).
/// Same vector kernels (and therefore the same f32 values) as
/// `crossEntropyLossRows`.
pub fn distillStatsRows(task: DistillStatsRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    for (task.row_start..task.row_end) |row_i| {
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
        task.row_stats[2 * row_i] = max_value;
        task.row_stats[2 * row_i + 1] = sum_exp;
    }
}

pub const DistillBackwardRowsTask = struct {
    // May alias `output` (the destructive in-place arm over saved logits).
    input: []const f32,
    output: []f32,
    // Forward-saved per-row {max, sum_exp} (DistillStatsRowsTask layout).
    row_stats: []const f32,
    // Per-row total teacher mass, pre-multiplied by the common gradient
    // scale: output[r, v] = row_mass[r] * softmax(input[r])[v]. The sparse
    // per-entry subtractions are applied by the caller afterwards (entry
    // lists are tiny next to rows x classes).
    row_mass: []const f32,
    class_count: usize,
    row_start: usize,
    row_end: usize,
};

pub fn runDistillBackwardRowsTask(task: *const DistillBackwardRowsTask) void {
    distillBackwardRows(task.*);
}

pub fn distillBackwardRows(task: DistillBackwardRowsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    for (task.row_start..task.row_end) |row_i| {
        const row_in = task.input[row_i * task.class_count ..][0..task.class_count];
        const row_out = task.output[row_i * task.class_count ..][0..task.class_count];
        const max_value = task.row_stats[2 * row_i];
        const inv_sum = 1 / task.row_stats[2 * row_i + 1];
        const scale = task.row_mass[row_i];
        const max_splat: Vec = @splat(max_value);
        const factor: Vec = @splat(scale * inv_sum);
        var class_i: usize = 0;
        while (class_i + vector_width <= task.class_count) : (class_i += vector_width) {
            const p = vexpf(vector_width, @as(Vec, row_in[class_i..][0..vector_width].*) - max_splat);
            row_out[class_i..][0..vector_width].* = p * factor;
        }
        while (class_i < task.class_count) : (class_i += 1) {
            row_out[class_i] = vexpf(1, @splat(row_in[class_i] - max_value))[0] * scale * inv_sum;
        }
    }
}

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
    backend_mod.quantized_matmul.q8_0.quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto(
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

pub fn runRmsNormRowStatsTask(task: *const RmsNormRowStatsTask) void {
    rmsNormRowStats(task.*);
}

pub fn runRmsNormWeightGradColumnsTask(task: *const RmsNormWeightGradColumnsTask) void {
    rmsNormWeightGradColumns(task.*);
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

/// Sum of squares of one contiguous row, the RMS-norm statistic every
/// rmsNorm row kernel shares: four 8-lane accumulators over the bulk,
/// combined as `(acc0 + acc1) + (acc2 + acc3)`, one 8-lane loop over the
/// remainder, a horizontal reduce, then the scalar tail. The order is the
/// bit contract of the rmsNorm family (goldens pin it); keep it.
inline fn rowSumSq(row: []const f32) f32 {
    const Vec = @Vector(8, f32);
    const vector_width = 8;
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

        const sumsq = rowSumSq(task.input[input_base..][0..task.feature_dim]);
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
        const sumsq = rowSumSq(task.input[base..][0..task.axis_dim]);
        const scale_value = 1 / @sqrt(sumsq * task.inv_axis_dim + task.eps);
        const scale_vec: Vec = @splat(scale_value);

        var axis_i: usize = 0;
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
        const sumsq = rowSumSq(task.input[base..][0..task.axis_dim]);
        const scale_value = 1 / @sqrt(sumsq * task.inv_axis_dim + task.eps);
        const scale_vec: Vec = @splat(scale_value);

        var axis_i: usize = 0;
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
        var sumsq0: Vec = @splat(0);
        var sumsq1: Vec = @splat(0);
        var dot0: Vec = @splat(0);
        var dot1: Vec = @splat(0);
        while (axis_i + 2 * vector_width <= task.axis_dim) : (axis_i += 2 * vector_width) {
            const v0: Vec = task.input[base + axis_i ..][0..vector_width].*;
            const g0: Vec = task.grad[base + axis_i ..][0..vector_width].*;
            const w0: Vec = task.weights[axis_i..][0..vector_width].*;
            const v1: Vec = task.input[base + axis_i + vector_width ..][0..vector_width].*;
            const g1: Vec = task.grad[base + axis_i + vector_width ..][0..vector_width].*;
            const w1: Vec = task.weights[axis_i + vector_width ..][0..vector_width].*;
            sumsq0 += v0 * v0;
            dot0 += g0 * w0 * v0;
            sumsq1 += v1 * v1;
            dot1 += g1 * w1 * v1;
        }
        var sumsq_vec: Vec = sumsq0 + sumsq1;
        var dot_vec: Vec = dot0 + dot1;
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
        const sumsq = rowSumSq(task.input[base..][0..task.axis_dim]);
        const rms_scale = 1 / @sqrt(sumsq * task.inv_axis_dim + task.eps);
        const rms_vec: Vec = @splat(rms_scale);
        var axis_i: usize = 0;
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

pub fn rmsNormRowStats(task: RmsNormRowStatsTask) void {
    for (task.row_start..task.row_end) |row_i| {
        const sumsq = rowSumSq(task.input[row_i * task.axis_dim ..][0..task.axis_dim]);
        task.stats[row_i] = 1 / @sqrt(sumsq * task.inv_axis_dim + task.eps);
    }
}

/// Column-partitioned dweight accumulation (the large-input pooled path):
/// the task accumulates its own column range over ALL rows in row order,
/// reading per-row 1/rms from the precomputed stats. The vector body and
/// the scalar tail evaluate the exact per-element expression of
/// `rmsNormMulBackwardWeightRows`, `dweight + gy * x * (1/rms)`, so a
/// column produces the same bits whether it lands in a vector lane or the
/// tail: the column split can move with the task count without changing
/// any result bit.
pub fn rmsNormWeightGradColumns(task: RmsNormWeightGradColumnsTask) void {
    const Vec = @Vector(8, f32);
    const vector_width = 8;
    // A task's column slice is one short run per row, `axis_dim` floats
    // apart, which the hardware prefetcher does not follow. Rows are
    // walked in blocks: the column loop sweeps the block so each cache
    // line of the block is fetched once and consumed vector by vector,
    // and the next block's lines are requested explicitly while this one
    // computes. Per column the rows are still added in row order (rows
    // within a block, blocks in sequence), so the bytes do not depend on
    // the block size.
    const row_block = 16;
    const prefetch_line_floats = 16;

    var row0: usize = 0;
    while (row0 < task.rows) : (row0 += row_block) {
        const row_end = @min(row0 + row_block, task.rows);
        const next_end = @min(row_end + row_block, task.rows);
        for (row_end..next_end) |row_i| {
            const ahead = row_i * task.axis_dim;
            var line = task.col_start;
            while (line < task.col_end) : (line += prefetch_line_floats) {
                @prefetch(&task.input[ahead + line], .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(&task.grad[ahead + line], .{ .rw = .read, .locality = 3, .cache = .data });
            }
        }

        var col = task.col_start;
        while (col + vector_width <= task.col_end) : (col += vector_width) {
            var acc: Vec = task.dweight[col..][0..vector_width].*;
            for (row0..row_end) |row_i| {
                const base = row_i * task.axis_dim;
                const values: Vec = task.input[base + col ..][0..vector_width].*;
                const grad: Vec = task.grad[base + col ..][0..vector_width].*;
                const rms_vec: Vec = @splat(task.stats[row_i]);
                acc = acc + grad * values * rms_vec;
            }
            task.dweight[col..][0..vector_width].* = acc;
        }
        while (col < task.col_end) : (col += 1) {
            var acc = task.dweight[col];
            for (row0..row_end) |row_i| {
                const base = row_i * task.axis_dim;
                acc += task.grad[base + col] * task.input[base + col] * task.stats[row_i];
            }
            task.dweight[col] = acc;
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

        // Pass 3: scale, then the affine weight and bias when present. The
        // terms apply left to right in one expression per element, so each
        // combination is the same rounding sequence as the composed ops.
        axis_i = 0;
        while (axis_i + vector_width <= task.axis_dim) : (axis_i += vector_width) {
            var v = @as(Vec, row_out[axis_i..][0..vector_width].*) * sigma_vec;
            if (task.weights) |weights| v = v * @as(Vec, weights[axis_i..][0..vector_width].*);
            if (task.biases) |biases| v = v + @as(Vec, biases[axis_i..][0..vector_width].*);
            row_out[axis_i..][0..vector_width].* = v;
        }
        while (axis_i < task.axis_dim) : (axis_i += 1) {
            var v = row_out[axis_i] * inv_sigma;
            if (task.weights) |weights| v = v * weights[axis_i];
            if (task.biases) |biases| v = v + biases[axis_i];
            row_out[axis_i] = v;
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

// ---------------- Strided-axis (inner > 1) softmax family ----------------
//
// The reduced axis has stride `inner` and the inner index is contiguous, so
// these kernels run vector lanes across `inner` and stream every pass
// row-major. The memory-order fix matters as much as the SIMD: the scalar
// strided loops they replace column-walked with an `inner * 4`-byte stride —
// one cache/TLB miss per element at LLM widths. Per-lane reduction order
// along the axis matches those scalar loops exactly; `vexpf` replaces `@exp`,
// matching the last-axis row kernels' arithmetic.

const InnerVec = @Vector(8, f32);
const inner_vec_width = 8;

/// Lane-wise `acc = max(acc, row)`.
fn maxIntoLanes(acc: []f32, row: []const f32) void {
    var i: usize = 0;
    while (i + inner_vec_width <= acc.len) : (i += inner_vec_width) {
        acc[i..][0..inner_vec_width].* = @max(@as(InnerVec, acc[i..][0..inner_vec_width].*), @as(InnerVec, row[i..][0..inner_vec_width].*));
    }
    while (i < acc.len) : (i += 1) acc[i] = @max(acc[i], row[i]);
}

/// Lane-wise non-finite guard: ±inf/NaN maxima shift by 0, exactly the
/// `std.math.isFinite` guard of the scalar loops.
fn guardFiniteLanes(row: []f32) void {
    const zero: InnerVec = @splat(0);
    const inf: InnerVec = @splat(std.math.inf(f32));
    var i: usize = 0;
    while (i + inner_vec_width <= row.len) : (i += inner_vec_width) {
        const value: InnerVec = row[i..][0..inner_vec_width].*;
        row[i..][0..inner_vec_width].* = @select(f32, @abs(value) < inf, value, zero);
    }
    while (i < row.len) : (i += 1) {
        if (!std.math.isFinite(row[i])) row[i] = 0;
    }
}

/// Lane-wise `out = exp(in - max)`, accumulating `sum += out`.
fn expSubStoreLanes(out: []f32, in: []const f32, max_row: []const f32, sum_row: []f32) void {
    var i: usize = 0;
    while (i + inner_vec_width <= out.len) : (i += inner_vec_width) {
        const value = vexpf(inner_vec_width, @as(InnerVec, in[i..][0..inner_vec_width].*) - @as(InnerVec, max_row[i..][0..inner_vec_width].*));
        out[i..][0..inner_vec_width].* = value;
        sum_row[i..][0..inner_vec_width].* = @as(InnerVec, sum_row[i..][0..inner_vec_width].*) + value;
    }
    while (i < out.len) : (i += 1) {
        const value = vexpf(1, @splat(in[i] - max_row[i]))[0];
        out[i] = value;
        sum_row[i] += value;
    }
}

/// Lane-wise `sum += exp(in - max)` with no materialized output.
fn expSubSumLanes(in: []const f32, max_row: []const f32, sum_row: []f32) void {
    var i: usize = 0;
    while (i + inner_vec_width <= in.len) : (i += inner_vec_width) {
        sum_row[i..][0..inner_vec_width].* = @as(InnerVec, sum_row[i..][0..inner_vec_width].*) +
            vexpf(inner_vec_width, @as(InnerVec, in[i..][0..inner_vec_width].*) - @as(InnerVec, max_row[i..][0..inner_vec_width].*));
    }
    while (i < in.len) : (i += 1) {
        sum_row[i] += vexpf(1, @splat(in[i] - max_row[i]))[0];
    }
}

/// Lane-wise `out *= 1 / sum` (reciprocal first, matching the scalar loops).
fn scaleByInvLanes(out: []f32, sum_row: []const f32) void {
    const one: InnerVec = @splat(1);
    var i: usize = 0;
    while (i + inner_vec_width <= out.len) : (i += inner_vec_width) {
        const inv = one / @as(InnerVec, sum_row[i..][0..inner_vec_width].*);
        out[i..][0..inner_vec_width].* = @as(InnerVec, out[i..][0..inner_vec_width].*) * inv;
    }
    while (i < out.len) : (i += 1) out[i] *= 1 / sum_row[i];
}

/// Lane-wise `out = in - shift`.
fn subLanes(out: []f32, in: []const f32, shift_row: []const f32) void {
    var i: usize = 0;
    while (i + inner_vec_width <= out.len) : (i += inner_vec_width) {
        out[i..][0..inner_vec_width].* = @as(InnerVec, in[i..][0..inner_vec_width].*) - @as(InnerVec, shift_row[i..][0..inner_vec_width].*);
    }
    while (i < out.len) : (i += 1) out[i] = in[i] - shift_row[i];
}

/// Lane-wise `dot += gy * y`.
fn mulIntoLanes(dot_row: []f32, gy: []const f32, y: []const f32) void {
    var i: usize = 0;
    while (i + inner_vec_width <= dot_row.len) : (i += inner_vec_width) {
        dot_row[i..][0..inner_vec_width].* = @as(InnerVec, dot_row[i..][0..inner_vec_width].*) +
            @as(InnerVec, gy[i..][0..inner_vec_width].*) * @as(InnerVec, y[i..][0..inner_vec_width].*);
    }
    while (i < dot_row.len) : (i += 1) dot_row[i] += gy[i] * y[i];
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
    // Lane sub-range this task owns. Lanes are independent, so any split is
    // bitwise identical to the serial full-range call.
    inner_start: usize,
    inner_end: usize,
};

pub fn runSoftmaxInnerTask(task: *const SoftmaxInnerTask) void {
    softmaxInner(task.*);
}

pub fn runLogsumexpInnerTask(task: *const SoftmaxInnerTask) void {
    logsumexpInner(task.*);
}

pub fn runLogSoftmaxInnerTask(task: *const SoftmaxInnerTask) void {
    logSoftmaxInner(task.*);
}

pub fn softmaxInner(task: SoftmaxInnerTask) void {
    const inner = task.inner;
    const lane0 = task.inner_start;
    const lanes = task.inner_end - task.inner_start;
    const max_row = task.scratch[lane0..][0..lanes];
    const sum_row = task.scratch[inner + lane0 ..][0..lanes];
    for (0..task.outer) |outer_i| {
        const base = outer_i * task.axis_dim * inner + lane0;
        @memcpy(max_row, task.input[base..][0..lanes]);
        for (1..task.axis_dim) |axis_i| {
            maxIntoLanes(max_row, task.input[base + axis_i * inner ..][0..lanes]);
        }
        @memset(sum_row, 0);
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            expSubStoreLanes(task.output[offset..][0..lanes], task.input[offset..][0..lanes], max_row, sum_row);
        }
        for (0..task.axis_dim) |axis_i| {
            scaleByInvLanes(task.output[base + axis_i * inner ..][0..lanes], sum_row);
        }
    }
}

/// Output holds one `inner`-wide row per outer block (the axis removed).
pub fn logsumexpInner(task: SoftmaxInnerTask) void {
    const inner = task.inner;
    const lane0 = task.inner_start;
    const lanes = task.inner_end - task.inner_start;
    const max_row = task.scratch[lane0..][0..lanes];
    const sum_row = task.scratch[inner + lane0 ..][0..lanes];
    for (0..task.outer) |outer_i| {
        const base = outer_i * task.axis_dim * inner + lane0;
        @memcpy(max_row, task.input[base..][0..lanes]);
        for (1..task.axis_dim) |axis_i| {
            maxIntoLanes(max_row, task.input[base + axis_i * inner ..][0..lanes]);
        }
        guardFiniteLanes(max_row);
        @memset(sum_row, 0);
        for (0..task.axis_dim) |axis_i| {
            expSubSumLanes(task.input[base + axis_i * inner ..][0..lanes], max_row, sum_row);
        }
        const out_row = task.output[outer_i * inner + lane0 ..][0..lanes];
        for (out_row, max_row, sum_row) |*out, max_safe, sum_exp| {
            out.* = max_safe + @log(sum_exp);
        }
    }
}

pub fn logSoftmaxInner(task: SoftmaxInnerTask) void {
    const inner = task.inner;
    const lane0 = task.inner_start;
    const lanes = task.inner_end - task.inner_start;
    const max_row = task.scratch[lane0..][0..lanes];
    const sum_row = task.scratch[inner + lane0 ..][0..lanes];
    for (0..task.outer) |outer_i| {
        const base = outer_i * task.axis_dim * inner + lane0;
        @memcpy(max_row, task.input[base..][0..lanes]);
        for (1..task.axis_dim) |axis_i| {
            maxIntoLanes(max_row, task.input[base + axis_i * inner ..][0..lanes]);
        }
        guardFiniteLanes(max_row);
        @memset(sum_row, 0);
        for (0..task.axis_dim) |axis_i| {
            expSubSumLanes(task.input[base + axis_i * inner ..][0..lanes], max_row, sum_row);
        }
        // Shift row reuses the max slots: shift = max_safe + log(sum_exp).
        for (max_row, sum_row) |*max_safe, sum_exp| {
            max_safe.* = max_safe.* + @log(sum_exp);
        }
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            subLanes(task.output[offset..][0..lanes], task.input[offset..][0..lanes], max_row);
        }
    }
}

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
    inner_start: usize,
    inner_end: usize,
};

pub fn runSoftmaxBackwardInnerTask(task: *const SoftmaxBackwardInnerTask) void {
    softmaxBackwardInner(task.*);
}

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

/// Streaming inner-lane layerNorm backward for non-last axes (serial: the
/// dweight/dbias accumulation crosses lanes, so a lane split would race).
/// `dx` reproduces the scalar fallback's per-lane expressions exactly;
/// dweight/dbias accumulate one deterministic vector-reduce per axis row
/// (a fixed reduction tree — deterministic, though ordered differently
/// from the retired lane-serial fallback).
pub fn layerNormBackwardInner(task: LayerNormBackwardInnerTask) void {
    const inner = task.inner;
    const mean_row = task.scratch[0..inner];
    const gsum_row = task.scratch[inner..][0..inner];
    // After the stats passes these two hold inv_sigma and correction.
    const sumsq_row = task.scratch[2 * inner ..][0..inner];
    const dot_row = task.scratch[3 * inner ..][0..inner];
    const inv_axis_splat: InnerVec = @splat(task.inv_axis_dim);
    const eps_splat: InnerVec = @splat(task.eps);
    const one: InnerVec = @splat(1);

    for (0..task.outer) |outer_i| {
        const base = outer_i * task.axis_dim * inner;

        // Pass A: per-lane Σ input and Σ grad·w over the axis.
        @memset(mean_row, 0);
        @memset(gsum_row, 0);
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            const w_axis: f32 = if (task.weights) |w| w[axis_i] else 1;
            const w_splat: InnerVec = @splat(w_axis);
            const row_in = task.input[offset..][0..inner];
            const row_g = task.grad[offset..][0..inner];
            var i: usize = 0;
            while (i + inner_vec_width <= inner) : (i += inner_vec_width) {
                mean_row[i..][0..inner_vec_width].* = @as(InnerVec, mean_row[i..][0..inner_vec_width].*) + @as(InnerVec, row_in[i..][0..inner_vec_width].*);
                gsum_row[i..][0..inner_vec_width].* = @as(InnerVec, gsum_row[i..][0..inner_vec_width].*) + @as(InnerVec, row_g[i..][0..inner_vec_width].*) * w_splat;
            }
            while (i < inner) : (i += 1) {
                mean_row[i] += row_in[i];
                gsum_row[i] += row_g[i] * w_axis;
            }
        }
        // mean = Σx / n (in place).
        {
            var i: usize = 0;
            while (i + inner_vec_width <= inner) : (i += inner_vec_width) {
                mean_row[i..][0..inner_vec_width].* = @as(InnerVec, mean_row[i..][0..inner_vec_width].*) * inv_axis_splat;
            }
            while (i < inner) : (i += 1) mean_row[i] *= task.inv_axis_dim;
        }

        // Pass B: per-lane Σ centered² and Σ grad·w·centered.
        @memset(sumsq_row, 0);
        @memset(dot_row, 0);
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            const w_axis: f32 = if (task.weights) |w| w[axis_i] else 1;
            const w_splat: InnerVec = @splat(w_axis);
            const row_in = task.input[offset..][0..inner];
            const row_g = task.grad[offset..][0..inner];
            var i: usize = 0;
            while (i + inner_vec_width <= inner) : (i += inner_vec_width) {
                const centered = @as(InnerVec, row_in[i..][0..inner_vec_width].*) - @as(InnerVec, mean_row[i..][0..inner_vec_width].*);
                sumsq_row[i..][0..inner_vec_width].* = @as(InnerVec, sumsq_row[i..][0..inner_vec_width].*) + centered * centered;
                dot_row[i..][0..inner_vec_width].* = @as(InnerVec, dot_row[i..][0..inner_vec_width].*) + @as(InnerVec, row_g[i..][0..inner_vec_width].*) * w_splat * centered;
            }
            while (i < inner) : (i += 1) {
                const centered = row_in[i] - mean_row[i];
                sumsq_row[i] += centered * centered;
                dot_row[i] += row_g[i] * w_axis * centered;
            }
        }
        // inv_sigma and correction in place: inv_sigma = 1/√(sumsq/n + eps);
        // shift = gsum/n·inv_sigma; correction = dot/n·inv_sigma³.
        {
            var i: usize = 0;
            while (i + inner_vec_width <= inner) : (i += inner_vec_width) {
                const inv_sigma = one / @sqrt(@as(InnerVec, sumsq_row[i..][0..inner_vec_width].*) * inv_axis_splat + eps_splat);
                sumsq_row[i..][0..inner_vec_width].* = inv_sigma;
                gsum_row[i..][0..inner_vec_width].* = @as(InnerVec, gsum_row[i..][0..inner_vec_width].*) * inv_axis_splat * inv_sigma;
                dot_row[i..][0..inner_vec_width].* = @as(InnerVec, dot_row[i..][0..inner_vec_width].*) * inv_axis_splat * inv_sigma * inv_sigma * inv_sigma;
            }
            while (i < inner) : (i += 1) {
                const inv_sigma = 1 / @sqrt(sumsq_row[i] * task.inv_axis_dim + task.eps);
                sumsq_row[i] = inv_sigma;
                gsum_row[i] = gsum_row[i] * task.inv_axis_dim * inv_sigma;
                dot_row[i] = dot_row[i] * task.inv_axis_dim * inv_sigma * inv_sigma * inv_sigma;
            }
        }

        // Pass C: dx rows plus one deterministic reduce per axis row for the
        // parameter gradients.
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            const w_axis: f32 = if (task.weights) |w| w[axis_i] else 1;
            const w_splat: InnerVec = @splat(w_axis);
            const row_in = task.input[offset..][0..inner];
            const row_g = task.grad[offset..][0..inner];
            var dweight_vec: InnerVec = @splat(0);
            var dbias_vec: InnerVec = @splat(0);
            var dweight_tail: f32 = 0;
            var dbias_tail: f32 = 0;
            var i: usize = 0;
            while (i + inner_vec_width <= inner) : (i += inner_vec_width) {
                const centered = @as(InnerVec, row_in[i..][0..inner_vec_width].*) - @as(InnerVec, mean_row[i..][0..inner_vec_width].*);
                const gv: InnerVec = row_g[i..][0..inner_vec_width].*;
                const inv_sigma: InnerVec = sumsq_row[i..][0..inner_vec_width].*;
                if (task.dx) |dx| {
                    dx[offset + i ..][0..inner_vec_width].* = gv * w_splat * inv_sigma -
                        @as(InnerVec, gsum_row[i..][0..inner_vec_width].*) -
                        centered * @as(InnerVec, dot_row[i..][0..inner_vec_width].*);
                }
                if (task.dweight != null) dweight_vec += gv * centered * inv_sigma;
                if (task.dbias != null) dbias_vec += gv;
            }
            while (i < inner) : (i += 1) {
                const centered = row_in[i] - mean_row[i];
                if (task.dx) |dx| {
                    dx[offset + i] = row_g[i] * w_axis * sumsq_row[i] - gsum_row[i] - centered * dot_row[i];
                }
                if (task.dweight != null) dweight_tail += row_g[i] * centered * sumsq_row[i];
                if (task.dbias != null) dbias_tail += row_g[i];
            }
            if (task.dweight) |dweight| dweight[axis_i] += @reduce(.Add, dweight_vec) + dweight_tail;
            if (task.dbias) |dbias| dbias[axis_i] += @reduce(.Add, dbias_vec) + dbias_tail;
        }
    }
}

pub fn softmaxBackwardInner(task: SoftmaxBackwardInnerTask) void {
    const inner = task.inner;
    const lane0 = task.inner_start;
    const lanes = task.inner_end - task.inner_start;
    const dot_row = task.scratch[lane0..][0..lanes];
    const scale_vec: InnerVec = @splat(task.scale);
    for (0..task.outer) |outer_i| {
        const base = outer_i * task.axis_dim * inner + lane0;
        @memset(dot_row, 0);
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            mulIntoLanes(dot_row, task.gy[offset..][0..lanes], task.y[offset..][0..lanes]);
        }
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            const row_y = task.y[offset..][0..lanes];
            const row_gy = task.gy[offset..][0..lanes];
            const row_out = task.output[offset..][0..lanes];
            var i: usize = 0;
            while (i + inner_vec_width <= lanes) : (i += inner_vec_width) {
                const yv: InnerVec = row_y[i..][0..inner_vec_width].*;
                const gyv: InnerVec = row_gy[i..][0..inner_vec_width].*;
                const dv: InnerVec = dot_row[i..][0..inner_vec_width].*;
                row_out[i..][0..inner_vec_width].* = scale_vec * yv * (gyv - dv);
            }
            while (i < lanes) : (i += 1) {
                row_out[i] = task.scale * row_y[i] * (row_gy[i] - dot_row[i]);
            }
        }
    }
}

// ---- Stats/norm inner-lane kernels: the non-last-axis arms of
// variance/standardize (fwd/bwd) and the rms/layer norms (fwd, rms bwd).
// Same layout rule as the softmax family above: lanes across `inner`, every
// pass streamed row-major, and per-lane accumulation order along the axis
// identical to the scalar strided loops they replace (bitwise-neutral: for
// one output element the sequence of adds is unchanged). The per-lane
// finalize steps (mean, 1/sigma, correction, ...) are plain scalar loops
// over the lane row that evaluate the retired loops' expressions verbatim,
// so the vector/scalar split of the streaming passes cannot move a bit.
// `Acc` (f32 or f64) is the standardize family's accumulation type; inputs
// and outputs stay f32.

fn laneWidth(comptime Acc: type) comptime_int {
    return switch (Acc) {
        f32 => inner_vec_width,
        f64 => inner_vec_width / 2,
        else => @compileError("lane accumulation type must be f32 or f64"),
    };
}

fn LaneVec(comptime Acc: type) type {
    return @Vector(laneWidth(Acc), Acc);
}

/// Lane-wise `acc += row` (f32 rows widened to `Acc`).
fn addIntoLanes(comptime Acc: type, acc: []Acc, row: []const f32) void {
    const width = laneWidth(Acc);
    var i: usize = 0;
    while (i + width <= acc.len) : (i += width) {
        const values: LaneVec(Acc) = @floatCast(@as(@Vector(width, f32), row[i..][0..width].*));
        acc[i..][0..width].* = @as(LaneVec(Acc), acc[i..][0..width].*) + values;
    }
    while (i < acc.len) : (i += 1) acc[i] += @as(Acc, @floatCast(row[i]));
}

/// Lane-wise `acc += (row - mean)^2`.
fn addCenteredSqIntoLanes(comptime Acc: type, acc: []Acc, row: []const f32, mean_row: []const Acc) void {
    const width = laneWidth(Acc);
    var i: usize = 0;
    while (i + width <= acc.len) : (i += width) {
        const values: LaneVec(Acc) = @floatCast(@as(@Vector(width, f32), row[i..][0..width].*));
        const centered = values - @as(LaneVec(Acc), mean_row[i..][0..width].*);
        acc[i..][0..width].* = @as(LaneVec(Acc), acc[i..][0..width].*) + centered * centered;
    }
    while (i < acc.len) : (i += 1) {
        const centered = @as(Acc, @floatCast(row[i])) - mean_row[i];
        acc[i] += centered * centered;
    }
}

/// Lane-wise `acc += row^2`.
fn addSqIntoLanes(acc: []f32, row: []const f32) void {
    var i: usize = 0;
    while (i + inner_vec_width <= acc.len) : (i += inner_vec_width) {
        const values: InnerVec = row[i..][0..inner_vec_width].*;
        acc[i..][0..inner_vec_width].* = @as(InnerVec, acc[i..][0..inner_vec_width].*) + values * values;
    }
    while (i < acc.len) : (i += 1) acc[i] += row[i] * row[i];
}

/// Lane-wise `out = (row - mean) / denom` in `Acc`, stored as f32.
fn centeredDivStoreLanes(comptime Acc: type, out: []f32, row: []const f32, mean_row: []const Acc, denom_row: []const Acc) void {
    const width = laneWidth(Acc);
    var i: usize = 0;
    while (i + width <= out.len) : (i += width) {
        const values: LaneVec(Acc) = @floatCast(@as(@Vector(width, f32), row[i..][0..width].*));
        const centered = values - @as(LaneVec(Acc), mean_row[i..][0..width].*);
        out[i..][0..width].* = @as(@Vector(width, f32), @floatCast(centered / @as(LaneVec(Acc), denom_row[i..][0..width].*)));
    }
    while (i < out.len) : (i += 1) {
        const centered = @as(Acc, @floatCast(row[i])) - mean_row[i];
        out[i] = @floatCast(centered / denom_row[i]);
    }
}

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
    inner_start: usize,
    inner_end: usize,
};

pub fn runVarianceInnerTask(task: *const VarianceInnerTask) void {
    varianceInner(task.*);
}

pub fn varianceInner(task: VarianceInnerTask) void {
    const inner = task.inner;
    const lane0 = task.inner_start;
    const lanes = task.inner_end - task.inner_start;
    const mean_row = task.scratch[lane0..][0..lanes];
    const sumsq_row = task.scratch[inner + lane0 ..][0..lanes];
    for (0..task.outer) |outer_i| {
        const base = outer_i * task.axis_dim * inner + lane0;
        @memset(mean_row, 0);
        for (0..task.axis_dim) |axis_i| {
            addIntoLanes(f32, mean_row, task.input[base + axis_i * inner ..][0..lanes]);
        }
        for (mean_row) |*mean| mean.* = mean.* * task.inv_axis_dim;
        @memset(sumsq_row, 0);
        for (0..task.axis_dim) |axis_i| {
            addCenteredSqIntoLanes(f32, sumsq_row, task.input[base + axis_i * inner ..][0..lanes], mean_row);
        }
        const out_row = task.output[outer_i * inner + lane0 ..][0..lanes];
        for (out_row, sumsq_row) |*out, sumsq| out.* = sumsq * task.inv_denom;
    }
}

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
        inner_start: usize,
        inner_end: usize,
    };
}

pub fn runStandardizeInnerTask(comptime Acc: type) fn (task: *const StandardizeInnerTask(Acc)) void {
    return struct {
        fn run(task: *const StandardizeInnerTask(Acc)) void {
            standardizeInner(Acc, task.*);
        }
    }.run;
}

pub fn standardizeInner(comptime Acc: type, task: StandardizeInnerTask(Acc)) void {
    const inner = task.inner;
    const lane0 = task.inner_start;
    const lanes = task.inner_end - task.inner_start;
    const mean_row = task.scratch[lane0..][0..lanes];
    const denom_row = task.scratch[inner + lane0 ..][0..lanes];
    const count: Acc = @floatFromInt(task.valid_count);
    for (0..task.outer) |outer_i| {
        const base = outer_i * task.axis_dim * inner + lane0;
        if (task.valid_count == 0) {
            for (0..task.axis_dim) |axis_i| @memset(task.output[base + axis_i * inner ..][0..lanes], 0);
            continue;
        }

        @memset(mean_row, 0);
        for (0..task.valid_count) |axis_i| {
            addIntoLanes(Acc, mean_row, task.input[base + axis_i * inner ..][0..lanes]);
        }
        for (mean_row) |*mean| mean.* = mean.* / count;

        @memset(denom_row, 0);
        if (task.valid_count > task.ddof_count) {
            for (0..task.valid_count) |axis_i| {
                addCenteredSqIntoLanes(Acc, denom_row, task.input[base + axis_i * inner ..][0..lanes], mean_row);
            }
            const denom_count: Acc = @floatFromInt(task.valid_count - task.ddof_count);
            for (denom_row) |*variance| variance.* = variance.* / denom_count;
        }
        for (denom_row) |*variance| {
            variance.* = if (task.eps_inside_sqrt) @sqrt(variance.* + task.eps) else @sqrt(variance.*) + task.eps;
        }

        for (0..task.valid_count) |axis_i| {
            const offset = base + axis_i * inner;
            centeredDivStoreLanes(Acc, task.output[offset..][0..lanes], task.input[offset..][0..lanes], mean_row, denom_row);
        }
        for (task.valid_count..task.axis_dim) |axis_i| @memset(task.output[base + axis_i * inner ..][0..lanes], 0);
    }
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
        inner_start: usize,
        inner_end: usize,
    };
}

pub fn runStandardizeBackwardInnerTask(comptime Acc: type) fn (task: *const StandardizeBackwardInnerTask(Acc)) void {
    return struct {
        fn run(task: *const StandardizeBackwardInnerTask(Acc)) void {
            standardizeBackwardInner(Acc, task.*);
        }
    }.run;
}

/// Lane-wise `gsum += g; dot += g * (row - mean)`.
fn addGradMomentsIntoLanes(comptime Acc: type, gsum_row: []Acc, dot_row: []Acc, grad: []const f32, row: []const f32, mean_row: []const Acc) void {
    const width = laneWidth(Acc);
    var i: usize = 0;
    while (i + width <= gsum_row.len) : (i += width) {
        const g: LaneVec(Acc) = @floatCast(@as(@Vector(width, f32), grad[i..][0..width].*));
        const values: LaneVec(Acc) = @floatCast(@as(@Vector(width, f32), row[i..][0..width].*));
        const centered = values - @as(LaneVec(Acc), mean_row[i..][0..width].*);
        gsum_row[i..][0..width].* = @as(LaneVec(Acc), gsum_row[i..][0..width].*) + g;
        dot_row[i..][0..width].* = @as(LaneVec(Acc), dot_row[i..][0..width].*) + g * centered;
    }
    while (i < gsum_row.len) : (i += 1) {
        const g = @as(Acc, @floatCast(grad[i]));
        const centered = @as(Acc, @floatCast(row[i])) - mean_row[i];
        gsum_row[i] += g;
        dot_row[i] += g * centered;
    }
}

pub fn standardizeBackwardInner(comptime Acc: type, task: StandardizeBackwardInnerTask(Acc)) void {
    const inner = task.inner;
    const lane0 = task.inner_start;
    const lanes = task.inner_end - task.inner_start;
    const mean_row = task.scratch[lane0..][0..lanes];
    const var_row = task.scratch[inner + lane0 ..][0..lanes];
    const denom_row = task.scratch[2 * inner + lane0 ..][0..lanes];
    const mean_grad_row = task.scratch[3 * inner + lane0 ..][0..lanes];
    const second_row = task.scratch[4 * inner + lane0 ..][0..lanes];
    const count: Acc = @floatFromInt(task.valid_count);
    const width = laneWidth(Acc);
    for (0..task.outer) |outer_i| {
        if (task.valid_count == 0) continue;
        const base = outer_i * task.axis_dim * inner + lane0;

        @memset(mean_row, 0);
        for (0..task.valid_count) |axis_i| {
            addIntoLanes(Acc, mean_row, task.input[base + axis_i * inner ..][0..lanes]);
        }
        for (mean_row) |*mean| mean.* = mean.* / count;

        @memset(var_row, 0);
        if (task.valid_count > task.ddof_count) {
            for (0..task.valid_count) |axis_i| {
                addCenteredSqIntoLanes(Acc, var_row, task.input[base + axis_i * inner ..][0..lanes], mean_row);
            }
            const denom_count: Acc = @floatFromInt(task.valid_count - task.ddof_count);
            for (var_row) |*variance| variance.* = variance.* / denom_count;
        }
        for (denom_row, var_row) |*denom, variance| {
            denom.* = if (task.eps_inside_sqrt) @sqrt(variance + task.eps) else @sqrt(variance) + task.eps;
        }

        @memset(mean_grad_row, 0);
        @memset(second_row, 0);
        for (0..task.valid_count) |axis_i| {
            const offset = base + axis_i * inner;
            addGradMomentsIntoLanes(Acc, mean_grad_row, second_row, task.grad[offset..][0..lanes], task.input[offset..][0..lanes], mean_row);
        }
        for (0..lanes) |i| {
            mean_grad_row[i] = mean_grad_row[i] / count;
            var second_scale: Acc = 0;
            if (task.valid_count > task.ddof_count and var_row[i] > 0) {
                const denom_count: Acc = @floatFromInt(task.valid_count - task.ddof_count);
                const std_value = @sqrt(var_row[i]);
                const denom = denom_row[i];
                second_scale = if (task.eps_inside_sqrt)
                    second_row[i] / (denom_count * denom * denom * denom)
                else
                    second_row[i] / (denom_count * std_value * denom * denom);
            }
            second_row[i] = second_scale;
            denom_row[i] = 1 / denom_row[i];
        }

        for (0..task.valid_count) |axis_i| {
            const offset = base + axis_i * inner;
            const row_g = task.grad[offset..][0..lanes];
            const row_in = task.input[offset..][0..lanes];
            const row_out = task.output[offset..][0..lanes];
            var i: usize = 0;
            while (i + width <= lanes) : (i += width) {
                const g: LaneVec(Acc) = @floatCast(@as(@Vector(width, f32), row_g[i..][0..width].*));
                const values: LaneVec(Acc) = @floatCast(@as(@Vector(width, f32), row_in[i..][0..width].*));
                const centered = values - @as(LaneVec(Acc), mean_row[i..][0..width].*);
                row_out[i..][0..width].* = @as(@Vector(width, f32), @floatCast((g - @as(LaneVec(Acc), mean_grad_row[i..][0..width].*)) * @as(LaneVec(Acc), denom_row[i..][0..width].*) -
                    centered * @as(LaneVec(Acc), second_row[i..][0..width].*)));
            }
            while (i < lanes) : (i += 1) {
                const centered = @as(Acc, @floatCast(row_in[i])) - mean_row[i];
                row_out[i] = @floatCast((@as(Acc, @floatCast(row_g[i])) - mean_grad_row[i]) * denom_row[i] - centered * second_row[i]);
            }
        }
    }
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
    inner_start: usize,
    inner_end: usize,
};

pub fn runRmsNormInnerTask(task: *const RmsNormInnerTask) void {
    rmsNormInner(task.*);
}

/// Lane-wise `out = in * scale [* w] [+ r]` (the residual added last, on
/// the left, as the scalar arms wrote it).
fn rmsApplyLanes(out: []f32, in: []const f32, scale_row: []const f32, weight: ?f32, residual: ?[]const f32) void {
    var i: usize = 0;
    while (i + inner_vec_width <= out.len) : (i += inner_vec_width) {
        var value = @as(InnerVec, in[i..][0..inner_vec_width].*) * @as(InnerVec, scale_row[i..][0..inner_vec_width].*);
        if (weight) |w| value = value * @as(InnerVec, @splat(w));
        if (residual) |r| value = @as(InnerVec, r[i..][0..inner_vec_width].*) + value;
        out[i..][0..inner_vec_width].* = value;
    }
    while (i < out.len) : (i += 1) {
        var value = in[i] * scale_row[i];
        if (weight) |w| value = value * w;
        if (residual) |r| value = r[i] + value;
        out[i] = value;
    }
}

pub fn rmsNormInner(task: RmsNormInnerTask) void {
    const inner = task.inner;
    const lane0 = task.inner_start;
    const lanes = task.inner_end - task.inner_start;
    const scale_row = task.scratch[lane0..][0..lanes];
    for (0..task.outer) |outer_i| {
        const base = outer_i * task.axis_dim * inner + lane0;
        @memset(scale_row, 0);
        for (0..task.axis_dim) |axis_i| {
            addSqIntoLanes(scale_row, task.input[base + axis_i * inner ..][0..lanes]);
        }
        for (scale_row) |*scale| scale.* = 1 / @sqrt(scale.* * task.inv_axis_dim + task.eps);
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            rmsApplyLanes(
                task.output[offset..][0..lanes],
                task.input[offset..][0..lanes],
                scale_row,
                if (task.weights) |w| w[axis_i] else null,
                if (task.residual) |r| r[offset..][0..lanes] else null,
            );
        }
    }
}

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
    inner_start: usize,
    inner_end: usize,
};

pub fn runRmsNormBackwardInputInnerTask(task: *const RmsNormBackwardInputInnerTask) void {
    rmsNormBackwardInputInner(task.*);
}

/// Lane-wise `sumsq += x^2; dot += (g * w) * x`.
fn addSqAndWeightedDotIntoLanes(sumsq_row: []f32, dot_row: []f32, row: []const f32, grad: []const f32, w_axis: f32) void {
    const w_splat: InnerVec = @splat(w_axis);
    var i: usize = 0;
    while (i + inner_vec_width <= sumsq_row.len) : (i += inner_vec_width) {
        const values: InnerVec = row[i..][0..inner_vec_width].*;
        const g: InnerVec = grad[i..][0..inner_vec_width].*;
        sumsq_row[i..][0..inner_vec_width].* = @as(InnerVec, sumsq_row[i..][0..inner_vec_width].*) + values * values;
        dot_row[i..][0..inner_vec_width].* = @as(InnerVec, dot_row[i..][0..inner_vec_width].*) + g * w_splat * values;
    }
    while (i < sumsq_row.len) : (i += 1) {
        sumsq_row[i] += row[i] * row[i];
        dot_row[i] += grad[i] * w_axis * row[i];
    }
}

pub fn rmsNormBackwardInputInner(task: RmsNormBackwardInputInnerTask) void {
    const inner = task.inner;
    const lane0 = task.inner_start;
    const lanes = task.inner_end - task.inner_start;
    const rms_row = task.scratch[lane0..][0..lanes];
    const corr_row = task.scratch[inner + lane0 ..][0..lanes];
    for (0..task.outer) |outer_i| {
        const base = outer_i * task.axis_dim * inner + lane0;
        @memset(rms_row, 0);
        @memset(corr_row, 0);
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            const w_axis: f32 = if (task.weights) |w| w[axis_i] else 1;
            addSqAndWeightedDotIntoLanes(rms_row, corr_row, task.input[offset..][0..lanes], task.grad[offset..][0..lanes], w_axis);
        }
        // Two spellings of the same correction, kept verbatim from the
        // weighted and plain scalar arms (they round differently).
        if (task.weights != null) {
            for (rms_row, corr_row) |*rms, *corr| {
                const rms_scale = 1 / @sqrt(rms.* * task.inv_axis_dim + task.eps);
                rms.* = rms_scale;
                corr.* = rms_scale * rms_scale * rms_scale * task.inv_axis_dim * corr.*;
            }
        } else {
            for (rms_row, corr_row) |*rms, *corr| {
                const inv_rms = 1 / @sqrt(rms.* * task.inv_axis_dim + task.eps);
                rms.* = inv_rms;
                corr.* = corr.* * task.inv_axis_dim * inv_rms * inv_rms * inv_rms;
            }
        }
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            const w_axis: f32 = if (task.weights) |w| w[axis_i] else 1;
            const w_splat: InnerVec = @splat(w_axis);
            const row_in = task.input[offset..][0..lanes];
            const row_g = task.grad[offset..][0..lanes];
            const row_out = task.output[offset..][0..lanes];
            var i: usize = 0;
            while (i + inner_vec_width <= lanes) : (i += inner_vec_width) {
                row_out[i..][0..inner_vec_width].* = @as(InnerVec, row_g[i..][0..inner_vec_width].*) * w_splat * @as(InnerVec, rms_row[i..][0..inner_vec_width].*) -
                    @as(InnerVec, row_in[i..][0..inner_vec_width].*) * @as(InnerVec, corr_row[i..][0..inner_vec_width].*);
            }
            while (i < lanes) : (i += 1) {
                row_out[i] = row_g[i] * w_axis * rms_row[i] - row_in[i] * corr_row[i];
            }
        }
    }
}

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

pub fn rmsNormBackwardWeightInner(task: RmsNormBackwardWeightInnerTask) void {
    const inner = task.inner;
    const scale_row = task.scratch[0..inner];
    for (0..task.outer) |outer_i| {
        const base = outer_i * task.axis_dim * inner;
        @memset(scale_row, 0);
        for (0..task.axis_dim) |axis_i| {
            addSqIntoLanes(scale_row, task.input[base + axis_i * inner ..][0..inner]);
        }
        for (scale_row) |*scale| scale.* = 1 / @sqrt(scale.* * task.inv_axis_dim + task.eps);
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            var acc = task.output[axis_i];
            for (task.grad[offset..][0..inner], task.input[offset..][0..inner], scale_row) |g, x, scale| {
                acc += g * x * scale;
            }
            task.output[axis_i] = acc;
        }
    }
}

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
    inner_start: usize,
    inner_end: usize,
};

pub fn runLayerNormInnerTask(task: *const LayerNormInnerTask) void {
    layerNormInner(task.*);
}

/// Lane-wise `out = (in - mean) * inv_sigma [* w] [+ b]`.
fn layerNormApplyLanes(out: []f32, in: []const f32, mean_row: []const f32, sigma_row: []const f32, weight: ?f32, bias: ?f32) void {
    var i: usize = 0;
    while (i + inner_vec_width <= out.len) : (i += inner_vec_width) {
        var value = (@as(InnerVec, in[i..][0..inner_vec_width].*) - @as(InnerVec, mean_row[i..][0..inner_vec_width].*)) * @as(InnerVec, sigma_row[i..][0..inner_vec_width].*);
        if (weight) |w| value = value * @as(InnerVec, @splat(w));
        if (bias) |b| value = value + @as(InnerVec, @splat(b));
        out[i..][0..inner_vec_width].* = value;
    }
    while (i < out.len) : (i += 1) {
        var value = (in[i] - mean_row[i]) * sigma_row[i];
        if (weight) |w| value = value * w;
        if (bias) |b| value = value + b;
        out[i] = value;
    }
}

pub fn layerNormInner(task: LayerNormInnerTask) void {
    const inner = task.inner;
    const lane0 = task.inner_start;
    const lanes = task.inner_end - task.inner_start;
    const mean_row = task.scratch[lane0..][0..lanes];
    const sigma_row = task.scratch[inner + lane0 ..][0..lanes];
    for (0..task.outer) |outer_i| {
        const base = outer_i * task.axis_dim * inner + lane0;
        @memset(mean_row, 0);
        for (0..task.axis_dim) |axis_i| {
            addIntoLanes(f32, mean_row, task.input[base + axis_i * inner ..][0..lanes]);
        }
        for (mean_row) |*mean| mean.* = mean.* * task.inv_axis_dim;
        @memset(sigma_row, 0);
        for (0..task.axis_dim) |axis_i| {
            addCenteredSqIntoLanes(f32, sigma_row, task.input[base + axis_i * inner ..][0..lanes], mean_row);
        }
        for (sigma_row) |*sigma| sigma.* = 1 / @sqrt(sigma.* * task.inv_axis_dim + task.eps);
        for (0..task.axis_dim) |axis_i| {
            const offset = base + axis_i * inner;
            layerNormApplyLanes(
                task.output[offset..][0..lanes],
                task.input[offset..][0..lanes],
                mean_row,
                sigma_row,
                if (task.weights) |w| w[axis_i] else null,
                if (task.biases) |b| b[axis_i] else null,
            );
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
