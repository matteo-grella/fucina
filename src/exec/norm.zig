//! RMSNorm, LayerNorm and GroupNorm forward and backward, one entry each:
//! the optional affine terms (`weight`, `bias`, `residual`) and the requested
//! gradients are options, not name variants. The fused rms-norm+rope kernel
//! is the one deliberate extra entry (its numerics differ from the composed
//! pair).
//!
//! Domain module: every op receives an explicit `*ExecContext`. Per-row SIMD
//! kernels + Task structs stay in the `row_ops` leaf; the fused rms-norm+rope
//! kernel reads the rope table's pub `sinValues()`/`cosValues()`. Home of
//! the options and result types (re-exported by `exec.zig`).

const backend_mod = @import("../backend.zig");
const kernels = backend_mod.kernels;
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

const exec_row_ops = @import("row_ops.zig");
const exec_shape = @import("shape.zig");
const exec_rope = @import("rope.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const Tensor = tensor.Tensor;
const DType = tensor.DType;
const RopeTable = exec_rope.RopeTable;
const RopeMode = exec_rope.RopeMode;

const productAfterAxis = exec_shape.productAfterAxis;
const productBeforeAxis = exec_shape.productBeforeAxis;
const contiguousStridesArray = exec_shape.contiguousStridesArray;

const RmsNormMulRopeHalfTask = exec_row_ops.RmsNormMulRopeHalfTask;
const RmsNormMulRowsTask = exec_row_ops.RmsNormMulRowsTask;
const RmsNormMulAddRowsTask = exec_row_ops.RmsNormMulAddRowsTask;
const RmsNormMulBackwardInputRowsTask = exec_row_ops.RmsNormMulBackwardInputRowsTask;
const RmsNormMulBackwardWeightRowsTask = exec_row_ops.RmsNormMulBackwardWeightRowsTask;
const RmsNormWeightGradBlocksTask = exec_row_ops.RmsNormWeightGradBlocksTask;
const RmsNormWeightGradReduceTask = exec_row_ops.RmsNormWeightGradReduceTask;
const LayerNormRowsTask = exec_row_ops.LayerNormRowsTask;
const LayerNormBackwardInputRowsTask = exec_row_ops.LayerNormBackwardInputRowsTask;
const LayerNormRowStatsTask = exec_row_ops.LayerNormRowStatsTask;
const LayerNormParamGradColumnsTask = exec_row_ops.LayerNormParamGradColumnsTask;
const RmsNormInnerTask = exec_row_ops.RmsNormInnerTask;
const RmsNormBackwardInputInnerTask = exec_row_ops.RmsNormBackwardInputInnerTask;
const LayerNormInnerTask = exec_row_ops.LayerNormInnerTask;
const runRmsNormMulRopeHalfTask = exec_row_ops.runRmsNormMulRopeHalfTask;
const runRmsNormMulRowsTask = exec_row_ops.runRmsNormMulRowsTask;
const runRmsNormMulAddRowsTask = exec_row_ops.runRmsNormMulAddRowsTask;
const runRmsNormMulBackwardInputRowsTask = exec_row_ops.runRmsNormMulBackwardInputRowsTask;
const runRmsNormMulBackwardWeightRowsTask = exec_row_ops.runRmsNormMulBackwardWeightRowsTask;
const runRmsNormWeightGradBlocksTask = exec_row_ops.runRmsNormWeightGradBlocksTask;
const runRmsNormWeightGradReduceTask = exec_row_ops.runRmsNormWeightGradReduceTask;
const runLayerNormRowsTask = exec_row_ops.runLayerNormRowsTask;
const runLayerNormBackwardInputRowsTask = exec_row_ops.runLayerNormBackwardInputRowsTask;
const runLayerNormRowStatsTask = exec_row_ops.runLayerNormRowStatsTask;
const runLayerNormParamGradColumnsTask = exec_row_ops.runLayerNormParamGradColumnsTask;
const runRmsNormInnerTask = exec_row_ops.runRmsNormInnerTask;
const runRmsNormBackwardInputInnerTask = exec_row_ops.runRmsNormBackwardInputInnerTask;
const runLayerNormInnerTask = exec_row_ops.runLayerNormInnerTask;
const rmsNormMulRopeHalfVectors = exec_row_ops.rmsNormMulRopeHalfVectors;
const rmsNormMulRows = exec_row_ops.rmsNormMulRows;
const rmsNormMulAddRows = exec_row_ops.rmsNormMulAddRows;
const rmsNormMulBackwardInputRows = exec_row_ops.rmsNormMulBackwardInputRows;
const rmsNormMulBackwardWeightRows = exec_row_ops.rmsNormMulBackwardWeightRows;
const layerNormBackwardInputRows = exec_row_ops.layerNormBackwardInputRows;
const layerNormAffineParamGradRows = exec_row_ops.layerNormAffineParamGradRows;
const layerNormRowStats = exec_row_ops.layerNormRowStats;
const layerNormParamGradColumns = exec_row_ops.layerNormParamGradColumns;

/// Optional per-feature affine terms of layerNorm and groupNorm: `weight`
/// scales the normalized row, `bias` adds to it (each rank-1 `[axis_dim]`
/// of the op's storage dtype, each independently optional). Applied in the
/// same row pass.
pub fn AffineOptions(comptime dtype: DType) type {
    return struct {
        weight: ?*const tensor.TensorOf(dtype) = null,
        bias: ?*const tensor.TensorOf(dtype) = null,
    };
}

/// The same terms as `[]const f32` slices, for the slice-level entries.
pub const AffineSlices = struct {
    weight: ?[]const f32 = null,
    bias: ?[]const f32 = null,
};

/// Which gradients an affine normalization backward computes. `weight` must
/// be the forward's weight when it had one (it feeds dx) and null otherwise.
pub const AffineBackwardOptions = struct {
    weight: ?*const Tensor = null,
    need_input: bool = true,
    need_weight: bool = false,
    need_bias: bool = false,
};

pub const AffineBackwardResult = struct {
    input: ?Tensor = null,
    weight: ?Tensor = null,
    bias: ?Tensor = null,

    pub fn deinit(self: *AffineBackwardResult) void {
        if (self.input) |*value| value.deinit();
        if (self.weight) |*value| value.deinit();
        if (self.bias) |*value| value.deinit();
        self.* = undefined;
    }
};

/// Optional terms of rmsNorm: `weight` (rank-1 `[axis_dim]`) scales the
/// normalized row, `residual` (same shape as `x`) is added after it, both in
/// the same row pass; both of the op's storage dtype.
pub fn RmsNormOptions(comptime dtype: DType) type {
    return struct {
        weight: ?*const tensor.TensorOf(dtype) = null,
        residual: ?*const tensor.TensorOf(dtype) = null,
    };
}

/// An optional operand widened to the compute dtype (a borrow, a copy, or
/// one cast); `null` stays `null`.
fn OptionalPrepared(comptime compute: DType) type {
    return struct {
        prepared: ?ExecContext.PreparedTensorOf(compute) = null,

        fn tensorPtr(self: *@This()) ?*const tensor.TensorOf(compute) {
            return if (self.prepared) |*p| p.tensor() else null;
        }

        fn deinit(self: *@This()) void {
            if (self.prepared) |*p| p.deinit();
        }
    };
}

fn prepareOptionalAs(ctx: *ExecContext, comptime dtype: DType, comptime compute: DType, x: ?*const tensor.TensorOf(dtype)) !OptionalPrepared(compute) {
    const t = x orelse return .{};
    return .{ .prepared = try ctx.prepareAs(dtype, compute, t) };
}

/// Which gradients rmsNormBackward computes. `weight` must be the forward's
/// weight when it had one (it feeds dx) and null otherwise; the weight
/// gradient `sum_rows gy * x_hat` does not depend on it.
pub const RmsNormBackwardOptions = struct {
    weight: ?*const Tensor = null,
    need_input: bool = true,
    need_weight: bool = false,
};

pub const RmsNormBackwardResult = struct {
    input: ?Tensor = null,
    weight: ?Tensor = null,

    pub fn deinit(self: *RmsNormBackwardResult) void {
        if (self.input) |*value| value.deinit();
        if (self.weight) |*value| value.deinit();
        self.* = undefined;
    }
};

/// RMSNorm over `axis`: y = x / sqrt(mean(x^2) + eps), then `* weight` and
/// `+ residual` when given. The weighted rows go through the row kernels
/// (`inner == 1`, large inputs); a non-last axis (`inner > 1`) through the
/// inner-lane kernel with its lane range split across the pool; every other
/// combination through the scalar loop. Each path computes the same
/// expression in the same order, so neither an option nor the layout is
/// ever a different numeric result. One f32 kernel set;
/// 16-bit inputs (and their weight/residual) follow the `.widened` policy.
pub fn rmsNorm(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize, eps: f32, options: RmsNormOptions(dtype)) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "rmsNorm");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    var ww = try prepareOptionalAs(ctx, dtype, compute, options.weight);
    defer ww.deinit();
    var rr = try prepareOptionalAs(ctx, dtype, compute, options.residual);
    defer rr.deinit();
    var out = try rmsNormF32(ctx, rank, xx.tensor(), axis, eps, .{ .weight = ww.tensorPtr(), .residual = rr.tensorPtr() });
    errdefer out.deinit();
    return ctx.storeAs(compute, dtype, out);
}

fn rmsNormF32(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, eps: f32, options: RmsNormOptions(.f32)) !Tensor {
    if (options.weight) |weight| {
        if (options.residual) |residual| return rmsNormMulAdd(ctx, rank, x, weight, residual, axis, eps);
        return rmsNormMul(ctx, rank, x, weight, axis, eps);
    }
    return rmsNormPlain(ctx, rank, x, options.residual, axis, eps);
}

/// VJP of rmsNorm; computes only the requested gradients. dx recomputes the
/// row rms from `x` (weighted or plain per `options.weight`); dweight is
/// `sum_rows gy * x_hat`. Small inputs accumulate it per column in row
/// order (the row kernel). Large inputs accumulate fixed 64-row blocks in
/// row order (`rms_weight_grad_block_rows`) and reduce the block partials
/// in block order, whichever threads computed them, so dweight is bitwise
/// identical for any thread count: block-ordered, a property of the shape
/// alone, not the single-chain row order of the small-input path.
pub fn rmsNormBackward(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize, eps: f32, options: RmsNormBackwardOptions) !RmsNormBackwardResult {
    var result = RmsNormBackwardResult{};
    errdefer result.deinit();
    if (options.need_input) {
        result.input = if (options.weight) |weight|
            try rmsNormBackwardInputWeighted(ctx, rank, x, weight, gy, axis, eps)
        else
            try rmsNormBackwardInputPlain(ctx, rank, x, gy, axis, eps);
    }
    if (options.need_weight) result.weight = try rmsNormBackwardWeight(ctx, rank, x, gy, axis, eps);
    return result;
}

fn rmsNormPlain(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, residual: ?*const Tensor, comptime axis: usize, eps: f32) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    if (residual) |r| try tensor.requireSameShape(x, r);

    const source = try x.rankView(rank);
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();
    var rr: ?ExecContext.PreparedTensor = null;
    defer if (rr) |*p| p.deinit();
    var residual_data: ?[]const f32 = null;
    if (residual) |r| {
        rr = try ctx.prepareContiguous(.f32, r);
        residual_data = rr.?.tensor().dataConst();
    }

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner > 1) {
        // Non-last axis: the inner-lane kernel (lane ranges split across
        // the pool; per-lane order equals the scalar loop below).
        var scratch = try ctx.empty(.f32, .{inner});
        defer scratch.deinit();
        ctx.dispatchInnerLanes(RmsNormInnerTask, .{
            .input = input,
            .weights = null,
            .residual = residual_data,
            .output = output,
            .axis_dim = axis_dim,
            .inner = inner,
            .scratch = scratch.data(),
            .outer = outer,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .inner_start = 0,
            .inner_end = inner,
        }, source.len(), inner, runRmsNormInnerTask);
        return out;
    }
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const value = input[base + axis_i * inner + inner_i];
                sumsq += value * value;
            }
            const scale_value = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const value = input[offset] * scale_value;
                output[offset] = if (residual_data) |r| r[offset] + value else value;
            }
        }
    }
    return out;
}

fn rmsNormMul(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, weight: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const weight_view = try weight.rankView(1);
    const axis_dim = source.shape[axis];
    if (weight_view.dim(0) != axis_dim) return tensor.TensorError.ShapeMismatch;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var ww = try ctx.prepareContiguous(.f32, weight);
    defer ww.deinit();
    const input = xx.tensor().dataConst();
    const weights = ww.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner == 1 and source.len() >= parallel.row_kernel_len_threshold) {
        const base_task: RmsNormMulRowsTask = .{
            .input = input,
            .weights = weights,
            .output = output,
            .axis_dim = axis_dim,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .row_start = 0,
            .row_end = outer,
        };
        if (outer > 1) {
            if (ctx.dispatchRange(RmsNormMulRowsTask, "row_start", "row_end", base_task, outer, runRmsNormMulRowsTask)) {
                return out;
            }
        }

        rmsNormMulRows(base_task);
        return out;
    }

    if (inner > 1) {
        // Non-last axis: the inner-lane kernel (lane ranges split across
        // the pool; per-lane order equals the scalar loop below).
        var scratch = try ctx.empty(.f32, .{inner});
        defer scratch.deinit();
        ctx.dispatchInnerLanes(RmsNormInnerTask, .{
            .input = input,
            .weights = weights,
            .residual = null,
            .output = output,
            .axis_dim = axis_dim,
            .inner = inner,
            .scratch = scratch.data(),
            .outer = outer,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .inner_start = 0,
            .inner_end = inner,
        }, source.len(), inner, runRmsNormInnerTask);
        return out;
    }
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const value = input[base + axis_i * inner + inner_i];
                sumsq += value * value;
            }
            const scale_value = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                output[offset] = input[offset] * scale_value * weights[axis_i];
            }
        }
    }
    return out;
}

fn rmsNormMulAdd(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, weight: *const Tensor, residual: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    try tensor.requireSameShape(x, residual);

    const source = try x.rankView(rank);
    const weight_view = try weight.rankView(1);
    const axis_dim = source.shape[axis];
    if (weight_view.dim(0) != axis_dim) return tensor.TensorError.ShapeMismatch;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var ww = try ctx.prepareContiguous(.f32, weight);
    defer ww.deinit();
    var rr = try ctx.prepareContiguous(.f32, residual);
    defer rr.deinit();
    const input = xx.tensor().dataConst();
    const weights = ww.tensor().dataConst();
    const residual_data = rr.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner == 1 and source.len() >= parallel.row_kernel_len_threshold) {
        const base_task: RmsNormMulAddRowsTask = .{
            .input = input,
            .weights = weights,
            .residual = residual_data,
            .output = output,
            .axis_dim = axis_dim,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .row_start = 0,
            .row_end = outer,
        };
        if (outer > 1) {
            if (ctx.dispatchRange(RmsNormMulAddRowsTask, "row_start", "row_end", base_task, outer, runRmsNormMulAddRowsTask)) {
                return out;
            }
        }

        rmsNormMulAddRows(base_task);
        return out;
    }

    if (inner > 1) {
        // Non-last axis: the inner-lane kernel (lane ranges split across
        // the pool; per-lane order equals the scalar loop below).
        var scratch = try ctx.empty(.f32, .{inner});
        defer scratch.deinit();
        ctx.dispatchInnerLanes(RmsNormInnerTask, .{
            .input = input,
            .weights = weights,
            .residual = residual_data,
            .output = output,
            .axis_dim = axis_dim,
            .inner = inner,
            .scratch = scratch.data(),
            .outer = outer,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .inner_start = 0,
            .inner_end = inner,
        }, source.len(), inner, runRmsNormInnerTask);
        return out;
    }
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const value = input[base + axis_i * inner + inner_i];
                sumsq += value * value;
            }
            const scale_value = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                output[offset] = residual_data[offset] + input[offset] * scale_value * weights[axis_i];
            }
        }
    }
    return out;
}

fn rmsNormBackwardInputWeighted(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    weight: *const Tensor,
    gy: *const Tensor,
    comptime axis: usize,
    eps: f32,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    try tensor.requireSameShape(x, gy);

    const source = try x.rankView(rank);
    const weight_view = try weight.rankView(1);
    const axis_dim = source.shape[axis];
    if (weight_view.dim(0) != axis_dim) return tensor.TensorError.ShapeMismatch;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var ww = try ctx.prepareContiguous(.f32, weight);
    defer ww.deinit();
    var ggy = try ctx.prepareContiguous(.f32, gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const weights = ww.tensor().dataConst();
    const grad = ggy.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner == 1 and source.len() >= parallel.row_kernel_len_threshold) {
        const base_task: RmsNormMulBackwardInputRowsTask = .{
            .input = input,
            .weights = weights,
            .grad = grad,
            .output = output,
            .axis_dim = axis_dim,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .row_start = 0,
            .row_end = outer,
        };
        if (outer > 1) {
            if (ctx.dispatchRange(RmsNormMulBackwardInputRowsTask, "row_start", "row_end", base_task, outer, runRmsNormMulBackwardInputRowsTask)) {
                return out;
            }
        }

        rmsNormMulBackwardInputRows(base_task);
        return out;
    }

    if (inner > 1) {
        var scratch = try ctx.empty(.f32, .{2 * inner});
        defer scratch.deinit();
        ctx.dispatchInnerLanes(RmsNormBackwardInputInnerTask, .{
            .input = input,
            .weights = weights,
            .grad = grad,
            .output = output,
            .axis_dim = axis_dim,
            .inner = inner,
            .scratch = scratch.data(),
            .outer = outer,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .inner_start = 0,
            .inner_end = inner,
        }, source.len(), inner, runRmsNormBackwardInputInnerTask);
        return out;
    }
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            var dot_acc: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const value = input[offset];
                sumsq += value * value;
                dot_acc += grad[offset] * weights[axis_i] * value;
            }
            const rms_scale = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            const correction_scale = rms_scale * rms_scale * rms_scale * inv_axis_dim * dot_acc;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                output[offset] = grad[offset] * weights[axis_i] * rms_scale - input[offset] * correction_scale;
            }
        }
    }
    return out;
}

fn rmsNormBackwardWeight(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    gy: *const Tensor,
    comptime axis: usize,
    eps: f32,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    try tensor.requireSameShape(x, gy);

    const source = try x.rankView(rank);
    const axis_dim = source.shape[axis];

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var ggy = try ctx.prepareContiguous(.f32, gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const grad = ggy.tensor().dataConst();

    var out = try ctx.zeros(.f32, .{axis_dim});
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner == 1 and source.len() >= parallel.row_kernel_len_threshold) {
        const base_task: RmsNormMulBackwardWeightRowsTask = .{
            .input = input,
            .grad = grad,
            .output = output,
            .axis_dim = axis_dim,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .row_start = 0,
            .row_end = outer,
        };
        // Fixed row-block grid: each 64-row block accumulates its own
        // partial in row order, the partials are reduced in block order.
        // Blocks are distributed over the tasks and the grid depends on
        // `outer` alone, so one thread and N produce the same bytes (the
        // serial fallbacks below walk the same grid). A single block is
        // the row kernel itself (0 + partial is the partial).
        const block_rows = exec_row_ops.rms_weight_grad_block_rows;
        const block_count = (outer + block_rows - 1) / block_rows;
        if (block_count == 1) {
            rmsNormMulBackwardWeightRows(base_task);
            return out;
        }
        const partials_buffer = try ctx.buffers.acquire(block_count * axis_dim);
        defer partials_buffer.release();
        const partials = partials_buffer.data[0 .. block_count * axis_dim];
        const blocks_task: RmsNormWeightGradBlocksTask = .{
            .input = input,
            .grad = grad,
            .partials = partials,
            .rows = outer,
            .axis_dim = axis_dim,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .block_start = 0,
            .block_end = block_count,
        };
        if (!ctx.dispatchRange(RmsNormWeightGradBlocksTask, "block_start", "block_end", blocks_task, block_count, runRmsNormWeightGradBlocksTask)) {
            exec_row_ops.rmsNormWeightGradBlocks(blocks_task);
        }
        const reduce_task: RmsNormWeightGradReduceTask = .{
            .partials = partials,
            .output = output,
            .block_count = block_count,
            .axis_dim = axis_dim,
            .col_start = 0,
            .col_end = axis_dim,
        };
        const reduce_pooled = partials.len >= parallel.row_kernel_len_threshold and
            ctx.dispatchRange(RmsNormWeightGradReduceTask, "col_start", "col_end", reduce_task, axis_dim, runRmsNormWeightGradReduceTask);
        if (!reduce_pooled) exec_row_ops.rmsNormWeightGradReduce(reduce_task);
        return out;
    }

    if (inner > 1) {
        // Non-last axis: lane-vectorized row statistics; the dweight
        // accumulation itself chains every (outer, lane) pair in the scalar
        // loop's order, so this arm stays serial.
        var scratch = try ctx.empty(.f32, .{inner});
        defer scratch.deinit();
        exec_row_ops.rmsNormBackwardWeightInner(.{
            .input = input,
            .grad = grad,
            .output = output,
            .axis_dim = axis_dim,
            .inner = inner,
            .scratch = scratch.data(),
            .outer = outer,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
        });
        return out;
    }

    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const value = input[base + axis_i * inner + inner_i];
                sumsq += value * value;
            }
            const rms_scale = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                output[axis_i] += grad[offset] * input[offset] * rms_scale;
            }
        }
    }
    return out;
}

/// Fused rmsnorm·weight + rotary embedding in one pass over each vector:
/// compute the row's rms once, then emit each rotated half-pair from the
/// normalized-and-scaled values directly (no intermediate normalized
/// tensor). Positions come from the precomputed sin/cos table; the fused
/// form matches the composed pair to f32 roundoff, not bitwise.
pub fn rmsNormMulRopeWithTable(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    weight: *const Tensor,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    eps: f32,
    table: *const RopeTable,
    comptime mode: RopeMode,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (position_axis >= rank or feature_axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const weight_view = try weight.rankView(1);
    const feature_dim = source.shape[feature_axis];
    if (weight_view.dim(0) != feature_dim) return tensor.TensorError.ShapeMismatch;
    if (table.feature_dim != feature_dim or table.positions.len != source.shape[position_axis]) return tensor.TensorError.ShapeMismatch;
    if (feature_dim % 2 != 0) return tensor.TensorError.InvalidShape;

    var ww = try ctx.prepareContiguous(.f32, weight);
    defer ww.deinit();
    x.buffer.waitReady();
    const input = x.buffer.data;
    const weights = ww.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const pair_count = feature_dim / 2;
    const sin_values = table.sinValues();
    const cos_values = table.cosValues();
    const output_strides = contiguousStridesArray(rank, source.shape);
    const input_feature_stride = source.strides[feature_axis];
    const output_feature_stride = output_strides[feature_axis];
    const total_vectors = source.len() / feature_dim;
    const inv_feature_dim = 1 / @as(f32, @floatFromInt(feature_dim));

    // Kernel choice is LAYOUT-ONLY: the vectorized kernel and the
    // generic fallback reduce a row's sum-of-squares in different orders,
    // so any batch-size input to this choice would make a row's bytes
    // depend on how many rows share the call — and k rows land in the KV
    // cache, where a batch-size dependence leaks into every subsequent
    // token. Same row shape in, same bytes out, at every row count (the
    // lossless-speculation contract, speculative/core.zig). The length
    // threshold below gates POOLING only — a row split never changes
    // per-row math.
    if (input_feature_stride == 1 and output_feature_stride == 1 and mode == .half) {
        var shape_dyn = [_]usize{1} ** tensor.max_rank;
        var input_strides_dyn = [_]usize{0} ** tensor.max_rank;
        var output_strides_dyn = [_]usize{0} ** tensor.max_rank;
        inline for (0..rank) |dim_i| {
            shape_dyn[dim_i] = source.shape[dim_i];
            input_strides_dyn[dim_i] = source.strides[dim_i];
            output_strides_dyn[dim_i] = output_strides[dim_i];
        }

        const base_task: RmsNormMulRopeHalfTask = .{
            .input = input,
            .weights = weights,
            .output = output,
            .sin_values = sin_values,
            .cos_values = cos_values,
            .shape = shape_dyn,
            .input_strides = input_strides_dyn,
            .output_strides = output_strides_dyn,
            .input_offset = x.offset,
            .rank = rank,
            .position_axis = position_axis,
            .feature_axis = feature_axis,
            .feature_dim = feature_dim,
            .pair_count = pair_count,
            .inv_feature_dim = inv_feature_dim,
            .eps = eps,
            .vector_start = 0,
            .vector_end = total_vectors,
        };

        if (total_vectors > 1 and source.len() >= parallel.vector_elementwise_len_threshold / 8) {
            if (ctx.dispatchRange(RmsNormMulRopeHalfTask, "vector_start", "vector_end", base_task, total_vectors, runRmsNormMulRopeHalfTask)) {
                return out;
            }
        }

        rmsNormMulRopeHalfVectors(base_task);
        return out;
    }

    for (0..total_vectors) |vector_i| {
        var remainder = vector_i;
        var input_base: usize = x.offset;
        var output_base: usize = 0;
        var position_coord: usize = 0;
        comptime var dim = rank;
        inline while (dim > 0) {
            dim -= 1;
            if (dim != feature_axis) {
                const coord = remainder % source.shape[dim];
                remainder /= source.shape[dim];
                input_base += coord * source.strides[dim];
                output_base += coord * output_strides[dim];
                if (dim == position_axis) position_coord = coord;
            }
        }

        var sumsq: f32 = 0;
        for (0..feature_dim) |feature_i| {
            const value = input[input_base + feature_i * input_feature_stride];
            sumsq += value * value;
        }
        const rms_scale = 1 / @sqrt(sumsq * inv_feature_dim + eps);

        if (input_feature_stride == 1 and output_feature_stride == 1 and mode == .half) {
            const Vec = @Vector(4, f32);
            const vector_width = 4;
            const scale_vec: Vec = @splat(rms_scale);
            var pair_i: usize = 0;
            while (pair_i + vector_width <= pair_count) : (pair_i += vector_width) {
                const angle_i = position_coord * pair_count + pair_i;
                const sin_vec: Vec = sin_values[angle_i..][0..vector_width].*;
                const cos_vec: Vec = cos_values[angle_i..][0..vector_width].*;
                const input_first_offset = input_base + pair_i;
                const input_second_offset = input_base + pair_i + pair_count;
                const output_first_offset = output_base + pair_i;
                const output_second_offset = output_base + pair_i + pair_count;
                const first = @as(Vec, input[input_first_offset..][0..vector_width].*) * scale_vec * @as(Vec, weights[pair_i..][0..vector_width].*);
                const second = @as(Vec, input[input_second_offset..][0..vector_width].*) * scale_vec * @as(Vec, weights[pair_i + pair_count ..][0..vector_width].*);
                output[output_first_offset..][0..vector_width].* = first * cos_vec - second * sin_vec;
                output[output_second_offset..][0..vector_width].* = first * sin_vec + second * cos_vec;
            }
            while (pair_i < pair_count) : (pair_i += 1) {
                const angle_i = position_coord * pair_count + pair_i;
                const sin_value = sin_values[angle_i];
                const cos_value = cos_values[angle_i];
                const input_first_offset = input_base + pair_i;
                const input_second_offset = input_base + pair_i + pair_count;
                const output_first_offset = output_base + pair_i;
                const output_second_offset = output_base + pair_i + pair_count;
                const first = input[input_first_offset] * rms_scale * weights[pair_i];
                const second = input[input_second_offset] * rms_scale * weights[pair_i + pair_count];
                output[output_first_offset] = first * cos_value - second * sin_value;
                output[output_second_offset] = first * sin_value + second * cos_value;
            }
            continue;
        }

        for (0..pair_count) |pair_i| {
            const angle_i = position_coord * pair_count + pair_i;
            const sin_value = sin_values[angle_i];
            const cos_value = cos_values[angle_i];

            // Full-width tables only in the fused kernel (checked above), so
            // .interleaved_tail degenerates to plain interleaved pairing.
            const first_feature = switch (mode) {
                .interleaved, .interleaved_tail => 2 * pair_i,
                .half => pair_i,
            };
            const second_feature = switch (mode) {
                .interleaved, .interleaved_tail => 2 * pair_i + 1,
                .half => pair_i + pair_count,
            };
            const input_first_offset = input_base + first_feature * input_feature_stride;
            const input_second_offset = input_base + second_feature * input_feature_stride;
            const output_first_offset = output_base + first_feature * output_feature_stride;
            const output_second_offset = output_base + second_feature * output_feature_stride;
            const first = input[input_first_offset] * rms_scale * weights[first_feature];
            const second = input[input_second_offset] * rms_scale * weights[second_feature];
            output[output_first_offset] = first * cos_value - second * sin_value;
            output[output_second_offset] = first * sin_value + second * cos_value;
        }
    }
    return out;
}

fn rmsNormBackwardInputPlain(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    try tensor.requireSameShape(x, gy);

    const source = try x.rankView(rank);
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var ggy = try ctx.prepareContiguous(.f32, gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const gyd = ggy.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner > 1) {
        var scratch = try ctx.empty(.f32, .{2 * inner});
        defer scratch.deinit();
        ctx.dispatchInnerLanes(RmsNormBackwardInputInnerTask, .{
            .input = input,
            .weights = null,
            .grad = gyd,
            .output = output,
            .axis_dim = axis_dim,
            .inner = inner,
            .scratch = scratch.data(),
            .outer = outer,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .inner_start = 0,
            .inner_end = inner,
        }, source.len(), inner, runRmsNormBackwardInputInnerTask);
        return out;
    }
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            var dot_acc: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const value = input[offset];
                sumsq += value * value;
                dot_acc += gyd[offset] * value;
            }
            const inv_rms = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            const correction = dot_acc * inv_axis_dim * inv_rms * inv_rms * inv_rms;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                output[offset] = gyd[offset] * inv_rms - input[offset] * correction;
            }
        }
    }
    return out;
}

/// LayerNorm with PyTorch semantics over `axis`: y = (x − μ)/√(σ² + eps),
/// where μ is the row mean and σ² is the BIASED variance (divide by N —
/// matches torch.nn.LayerNorm and ggml_norm). Statistics are two-pass per
/// row: mean first, then the centered sum of squares (ggml-style; one
/// extra subtraction over the E[x²]−μ² shortcut but immune to its
/// catastrophic cancellation).
pub fn layerNormRows(
    ctx: *ExecContext,
    input: []const f32,
    rows: usize,
    cols: usize,
    eps: f32,
    affine: AffineSlices,
) !Tensor {
    if (input.len != rows * cols) return tensor.TensorError.InvalidDataLength;
    if (affine.weight) |w| if (w.len != cols) return tensor.TensorError.ShapeMismatch;
    if (affine.bias) |b| if (b.len != cols) return tensor.TensorError.ShapeMismatch;

    var out = try ctx.empty(.f32, .{ rows, cols });
    errdefer out.deinit();
    const base_task: LayerNormRowsTask = .{
        .input = input,
        .weights = affine.weight,
        .biases = affine.bias,
        .output = out.data(),
        .axis_dim = cols,
        .inv_axis_dim = 1 / @as(f32, @floatFromInt(cols)),
        .eps = eps,
        .row_start = 0,
        .row_end = rows,
    };
    if (input.len >= parallel.row_kernel_len_threshold and rows > 1) {
        if (ctx.dispatchRange(LayerNormRowsTask, "row_start", "row_end", base_task, rows, runLayerNormRowsTask)) {
            return out;
        }
    }
    exec_row_ops.layerNormRows(base_task);
    return out;
}

/// LayerNorm over `axis` with the optional affine terms of `options` applied
/// in the same row pass (`* weight`, then `+ bias`). One f32 kernel set;
/// 16-bit inputs (and their affine terms) follow the `.widened` policy.
pub fn layerNorm(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize, eps: f32, options: AffineOptions(dtype)) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "layerNorm");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    var ww = try prepareOptionalAs(ctx, dtype, compute, options.weight);
    defer ww.deinit();
    var bb = try prepareOptionalAs(ctx, dtype, compute, options.bias);
    defer bb.deinit();
    var out = try layerNormF32(ctx, rank, xx.tensor(), axis, eps, .{ .weight = ww.tensorPtr(), .bias = bb.tensorPtr() });
    errdefer out.deinit();
    return ctx.storeAs(compute, dtype, out);
}

fn layerNormF32(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, eps: f32, options: AffineOptions(.f32)) !Tensor {
    const source = try x.rankView(rank);
    const axis_dim = source.shape[axis];
    var ww = try prepareAffineTerm(ctx, options.weight, axis_dim);
    defer ww.deinit();
    var bb = try prepareAffineTerm(ctx, options.bias, axis_dim);
    defer bb.deinit();
    return layerNormDispatchAxisRank(ctx, rank, x, ww.slice(), bb.slice(), axis, eps);
}

/// An optional rank-1 `[axis_dim]` affine term made contiguous f32.
const PreparedAffineTerm = struct {
    prepared: ?ExecContext.PreparedTensor = null,

    fn slice(self: *PreparedAffineTerm) ?[]const f32 {
        return if (self.prepared) |*p| p.tensor().dataConst() else null;
    }

    fn deinit(self: *PreparedAffineTerm) void {
        if (self.prepared) |*p| p.deinit();
    }
};

fn prepareAffineTerm(ctx: *ExecContext, term: ?*const Tensor, axis_dim: usize) !PreparedAffineTerm {
    const t = term orelse return .{};
    const view = try t.rankView(1);
    if (view.dim(0) != axis_dim) return tensor.TensorError.ShapeMismatch;
    return .{ .prepared = try ctx.prepareContiguous(.f32, t) };
}

fn layerNormDispatchAxisRank(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    weights: ?[]const f32,
    biases: ?[]const f32,
    comptime axis: usize,
    eps: f32,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner == 1 and source.len() >= parallel.row_kernel_len_threshold) {
        const base_task: LayerNormRowsTask = .{
            .input = input,
            .weights = weights,
            .biases = biases,
            .output = output,
            .axis_dim = axis_dim,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .row_start = 0,
            .row_end = outer,
        };
        if (outer > 1) {
            if (ctx.dispatchRange(LayerNormRowsTask, "row_start", "row_end", base_task, outer, runLayerNormRowsTask)) {
                return out;
            }
        }

        exec_row_ops.layerNormRows(base_task);
        return out;
    }

    if (inner > 1) {
        var scratch = try ctx.empty(.f32, .{2 * inner});
        defer scratch.deinit();
        ctx.dispatchInnerLanes(LayerNormInnerTask, .{
            .input = input,
            .weights = weights,
            .biases = biases,
            .output = output,
            .axis_dim = axis_dim,
            .inner = inner,
            .scratch = scratch.data(),
            .outer = outer,
            .inv_axis_dim = inv_axis_dim,
            .eps = eps,
            .inner_start = 0,
            .inner_end = inner,
        }, source.len(), inner, runLayerNormInnerTask);
        return out;
    }

    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sum_acc: f32 = 0;
            for (0..axis_dim) |axis_i| {
                sum_acc += input[base + axis_i * inner + inner_i];
            }
            const mean_value = sum_acc * inv_axis_dim;
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const centered = input[base + axis_i * inner + inner_i] - mean_value;
                sumsq += centered * centered;
            }
            const inv_sigma = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                var value = (input[offset] - mean_value) * inv_sigma;
                if (weights) |w| value = value * w[axis_i];
                if (biases) |b| value = value + b[axis_i];
                output[offset] = value;
            }
        }
    }
    return out;
}

/// GroupNorm over `[T, C]` rows (ggml_compute_forward_group_norm semantics):
/// `groups` must divide C; per group g the mean and BIASED variance are
/// accumulated in f64 over the T × C/groups elements of channel columns
/// `[g*C/groups, (g+1)*C/groups)`, then `y = (x − mean)/sqrt(var + eps)` is
/// applied in f32 (eps INSIDE the sqrt; the 1/sqrt scale is computed in f32,
/// matching ggml). Optional per-channel affine `y = y*weight[c] + bias[c]`
/// (`[C]` each) is applied AFTER normalization.
pub fn groupNorm(ctx: *ExecContext, x: *const Tensor, groups: usize, eps: f32, options: AffineOptions(.f32)) !Tensor {
    const source = try x.rankView(2);
    const rows = source.shape[0];
    const cols = source.shape[1];
    if (rows == 0 or cols == 0) return tensor.TensorError.InvalidShape;
    if (groups == 0 or cols % groups != 0) return tensor.TensorError.InvalidShape;

    var ww = try prepareAffineTerm(ctx, options.weight, cols);
    defer ww.deinit();
    var bb = try prepareAffineTerm(ctx, options.bias, cols);
    defer bb.deinit();

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();

    var out = try ctx.empty(.f32, .{ rows, cols });
    errdefer out.deinit();
    ctx.enableNativeVectorPoolForWork(rows * cols, parallel.vector_elementwise_len_threshold);
    kernels.groupNormInto(ctx.pc(), &out, xx.tensor(), ww.slice(), bb.slice(), rows, cols, groups, eps);
    return out;
}

/// VJP of groupNorm. Computes only the requested gradients:
///   dx = (1/σ_g)·(ĝ − mean_G(ĝ) − x̂·mean_G(ĝ·x̂)) per group, with
///        ĝ = gy⊙weight (or gy when the forward had no affine weight) and
///        the group statistics RECOMPUTED from `x` with the forward's exact
///        f64-accumulate / f32-apply policy (the layerNorm VJP convention —
///        nothing is saved from the forward);
///   dweight[c] = Σ_t gy[t,c]·x̂[t,c];  dbias[c] = Σ_t gy[t,c].
/// One backend kernel fills all requested outputs, parallel over whole
/// groups (disjoint column slices ⇒ bitwise identical for any thread count).
pub fn groupNormBackward(ctx: *ExecContext, x: *const Tensor, gy: *const Tensor, groups: usize, eps: f32, options: AffineBackwardOptions) !AffineBackwardResult {
    const source = try x.rankView(2);
    const rows = source.shape[0];
    const cols = source.shape[1];
    if (rows == 0 or cols == 0) return tensor.TensorError.InvalidShape;
    if (groups == 0 or cols % groups != 0) return tensor.TensorError.InvalidShape;
    try tensor.requireSameShape(x, gy);

    var ww = try prepareAffineTerm(ctx, options.weight, cols);
    defer ww.deinit();

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var gg = try ctx.prepareContiguous(.f32, gy);
    defer gg.deinit();

    var result = AffineBackwardResult{};
    errdefer result.deinit();
    if (options.need_input) result.input = try ctx.empty(.f32, .{ rows, cols });
    if (options.need_weight) result.weight = try ctx.empty(.f32, .{cols});
    if (options.need_bias) result.bias = try ctx.empty(.f32, .{cols});
    if (!options.need_input and !options.need_weight and !options.need_bias) return result;

    ctx.enableNativeVectorPoolForWork(rows * cols, parallel.vector_elementwise_len_threshold);
    kernels.groupNormBackwardInto(
        ctx.pc(),
        if (result.input) |*t| t else null,
        if (result.weight) |*t| t else null,
        if (result.bias) |*t| t else null,
        xx.tensor(),
        gg.tensor(),
        ww.slice(),
        rows,
        cols,
        groups,
        eps,
    );
    return result;
}

/// VJP of layerNorm. Computes only the requested gradients:
/// dx = (1/σ)(g' − mean(g') − x̂·mean(g'·x̂)) with x̂ = (x−μ)/σ and
/// g' = gy⊙weight (gy when the forward had no weight), dweight = Σ_rows gy⊙x̂,
/// dbias = Σ_rows gy. The dweight/dbias row reduction always accumulates each
/// column in row order — serially for small inputs
/// (layerNormAffineParamGradRows), column-partitioned across the pool for
/// large ones (layerNormParamGradColumns) — so it is bitwise identical for
/// any thread count.
pub fn layerNormBackward(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize, eps: f32, options: AffineBackwardOptions) !AffineBackwardResult {
    const source = try x.rankView(rank);
    var ww = try prepareAffineTerm(ctx, options.weight, source.shape[axis]);
    defer ww.deinit();
    return layerNormBackwardDispatchAxisRank(ctx, rank, x, ww.slice(), gy, axis, eps, options.need_input, options.need_weight, options.need_bias);
}

fn layerNormBackwardDispatchAxisRank(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    weights: ?[]const f32,
    gy: *const Tensor,
    comptime axis: usize,
    eps: f32,
    need_input: bool,
    need_weight: bool,
    need_bias: bool,
) !AffineBackwardResult {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    try tensor.requireSameShape(x, gy);

    const source = try x.rankView(rank);
    const axis_dim = source.shape[axis];

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var ggy = try ctx.prepareContiguous(.f32, gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const grad = ggy.tensor().dataConst();

    var result = AffineBackwardResult{};
    errdefer result.deinit();
    if (need_input) result.input = try ctx.empty(.f32, source.shape);
    if (need_weight) result.weight = try ctx.zeros(.f32, .{axis_dim});
    if (need_bias) result.bias = try ctx.zeros(.f32, .{axis_dim});
    if (!need_input and !need_weight and !need_bias) return result;

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));

    if (inner == 1) {
        if (result.input) |*dx| {
            const base_task: LayerNormBackwardInputRowsTask = .{
                .input = input,
                .weights = weights,
                .grad = grad,
                .output = dx.data(),
                .axis_dim = axis_dim,
                .inv_axis_dim = inv_axis_dim,
                .eps = eps,
                .row_start = 0,
                .row_end = outer,
            };
            var dispatched = false;
            if (outer > 1 and source.len() >= parallel.row_kernel_len_threshold) {
                if (ctx.dispatchRange(LayerNormBackwardInputRowsTask, "row_start", "row_end", base_task, outer, runLayerNormBackwardInputRowsTask)) {
                    dispatched = true;
                }
            }
            if (!dispatched) layerNormBackwardInputRows(base_task);
        }

        if (need_weight or need_bias) {
            var dispatched = false;
            if (outer > 1 and axis_dim > 1 and source.len() >= parallel.row_kernel_len_threshold) {
                if (ctx.workPool()) |pool| {
                    // Per-row {mean, 1/σ} scratch, then the
                    // column-partitioned accumulation: both stages are
                    // bitwise identical for any thread count (see the
                    // task structs / kernels), unlike per-task row
                    // partials combined in task order.
                    var stats: []f32 = &.{};
                    defer if (stats.len > 0) ctx.allocator.free(stats);
                    if (need_weight) {
                        stats = try ctx.allocator.alloc(f32, 2 * outer);
                        const stats_base: LayerNormRowStatsTask = .{
                            .input = input,
                            .stats = stats,
                            .axis_dim = axis_dim,
                            .inv_axis_dim = inv_axis_dim,
                            .eps = eps,
                            .row_start = 0,
                            .row_end = outer,
                        };
                        const row_task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                        var row_tasks: [parallel.vector_max_threads]LayerNormRowStatsTask = undefined;
                        for (0..row_task_count) |task_i| {
                            row_tasks[task_i] = stats_base;
                            row_tasks[task_i].row_start = task_i * outer / row_task_count;
                            row_tasks[task_i].row_end = (task_i + 1) * outer / row_task_count;
                        }
                        pool.parallelChunks(LayerNormRowStatsTask, row_tasks[0..row_task_count], runLayerNormRowStatsTask);
                    }

                    const col_base: LayerNormParamGradColumnsTask = .{
                        .input = input,
                        .grad = grad,
                        .stats = stats,
                        .dweight = if (result.weight) |*value| value.data() else null,
                        .dbias = if (result.bias) |*value| value.data() else null,
                        .rows = outer,
                        .axis_dim = axis_dim,
                        .col_start = 0,
                        .col_end = axis_dim,
                    };
                    const col_task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), axis_dim);
                    var col_tasks: [parallel.vector_max_threads]LayerNormParamGradColumnsTask = undefined;
                    for (0..col_task_count) |task_i| {
                        col_tasks[task_i] = col_base;
                        col_tasks[task_i].col_start = task_i * axis_dim / col_task_count;
                        col_tasks[task_i].col_end = (task_i + 1) * axis_dim / col_task_count;
                    }
                    pool.parallelChunks(LayerNormParamGradColumnsTask, col_tasks[0..col_task_count], runLayerNormParamGradColumnsTask);
                    dispatched = true;
                }
            }
            if (!dispatched) {
                layerNormAffineParamGradRows(
                    input,
                    grad,
                    if (result.weight) |*value| value.data() else null,
                    if (result.bias) |*value| value.data() else null,
                    outer,
                    axis_dim,
                    inv_axis_dim,
                    eps,
                );
            }
        }
        return result;
    }

    // Generic inner>1 arm: the streaming inner-lane kernel (serial — the
    // dweight/dbias accumulation crosses lanes).
    var scratch = try ctx.empty(.f32, .{4 * inner});
    defer scratch.deinit();
    exec_row_ops.layerNormBackwardInner(.{
        .input = input,
        .grad = grad,
        .weights = weights,
        .dx = if (result.input) |*value| value.data() else null,
        .dweight = if (result.weight) |*value| value.data() else null,
        .dbias = if (result.bias) |*value| value.data() else null,
        .axis_dim = axis_dim,
        .inner = inner,
        .scratch = scratch.data(),
        .inv_axis_dim = inv_axis_dim,
        .eps = eps,
        .outer = outer,
    });
    return result;
}
