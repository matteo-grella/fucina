//! Softmax (plain + masked/sink/ALiBi/causal "ext") forward and backward.
//!
//! Domain module: every op receives an explicit `*ExecContext`; the per-row SIMD
//! kernels + Task structs stay in the `row_ops` leaf (imported), so the hot
//! loops are untouched. Home of `SoftmaxExtOptions` (re-exported by `exec.zig`).

const std = @import("std");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");

const exec_row_ops = @import("row_ops.zig");
const exec_shape = @import("shape.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const Tensor = tensor.Tensor;
const DType = dtype_mod.DType;

const productAfterAxis = exec_shape.productAfterAxis;
const productBeforeAxis = exec_shape.productBeforeAxis;
const contiguousStridesArray = exec_shape.contiguousStridesArray;
const floorPowerOfTwo = exec_shape.floorPowerOfTwo;
const alibiSlope = exec_shape.alibiSlope;

const SoftmaxRowsTask = exec_row_ops.SoftmaxRowsTask;
const LogRowsTask = exec_row_ops.LogRowsTask;
const runLogsumexpRowsTask = exec_row_ops.runLogsumexpRowsTask;
const runLogSoftmaxRowsTask = exec_row_ops.runLogSoftmaxRowsTask;
const logsumexpRows = exec_row_ops.logsumexpRows;
const logSoftmaxRows = exec_row_ops.logSoftmaxRows;
const shapeWithoutAxis = exec_shape.shapeWithoutAxis;
const SoftmaxExtRowsTask = exec_row_ops.SoftmaxExtRowsTask;
const SoftmaxBackwardRowsTask = exec_row_ops.SoftmaxBackwardRowsTask;
const SoftmaxInnerTask = exec_row_ops.SoftmaxInnerTask;
const SoftmaxBackwardInnerTask = exec_row_ops.SoftmaxBackwardInnerTask;
const runSoftmaxInnerTask = exec_row_ops.runSoftmaxInnerTask;
const runLogsumexpInnerTask = exec_row_ops.runLogsumexpInnerTask;
const runLogSoftmaxInnerTask = exec_row_ops.runLogSoftmaxInnerTask;
const runSoftmaxBackwardInnerTask = exec_row_ops.runSoftmaxBackwardInnerTask;

const runSoftmaxRowsTask = exec_row_ops.runSoftmaxRowsTask;
const runSoftmaxExtRowsTask = exec_row_ops.runSoftmaxExtRowsTask;
const runSoftmaxBackwardRowsTask = exec_row_ops.runSoftmaxBackwardRowsTask;
const softmaxRows = exec_row_ops.softmaxRows;
const softmaxExtRows = exec_row_ops.softmaxExtRows;
const softmaxBackwardRows = exec_row_ops.softmaxBackwardRows;

pub const SoftmaxExtOptions = struct {
    mask: ?*const Tensor = null,
    sinks: ?[]const f32 = null,
    scale: f32 = 1,
    max_bias: f32 = 0,
    head_axis: ?usize = null,
    causal_query_axis: ?usize = null,
    causal_source_offset: usize = 0,
};

/// Log-sum-exp along `axis` (torch.logsumexp), the axis removed:
/// max-shifted with the non-finite guard (±inf maxima shift by 0, so
/// all(-inf) rows give -inf and +inf entries give +inf, never NaN). The
/// last-axis path runs the fused SIMD row kernel (`logsumexpRows`,
/// task-parallel over rows like `softmax`); other axes run the streaming
/// inner-lane kernel (`logsumexpInner`) with identical semantics.
/// One f32 kernel; 16-bit inputs follow the `.widened` policy.
pub fn logsumexp(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
    const compute = comptime ExecContext.widenedCompute(dtype, "logsumexp");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    var out = try logsumexpF32(ctx, rank, xx.tensor(), axis);
    errdefer out.deinit();
    return ctx.storeAs(compute, comptime dtype_mod.outputDType(.reduction, dtype), out);
}

fn logsumexpF32(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try ctx.empty(.f32, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    if (inner == 1) {
        const base_task: LogRowsTask = .{
            .input = input,
            .output = output,
            .axis_dim = axis_dim,
            .row_start = 0,
            .row_end = outer,
        };
        if (outer > 1 and source.len() >= parallel.row_kernel_len_threshold) {
            if (ctx.dispatchRange(LogRowsTask, "row_start", "row_end", base_task, outer, runLogsumexpRowsTask)) {
                return out;
            }
        }
        logsumexpRows(base_task);
        return out;
    }

    var scratch = try ctx.empty(.f32, .{2 * inner});
    defer scratch.deinit();
    ctx.dispatchInnerLanes(SoftmaxInnerTask, .{
        .input = input,
        .output = output,
        .axis_dim = axis_dim,
        .inner = inner,
        .scratch = scratch.data(),
        .outer = outer,
        .inner_start = 0,
        .inner_end = inner,
    }, source.len(), inner, runLogsumexpInnerTask);
    return out;
}

/// Log-softmax along `axis` (torch.log_softmax), shape-preserving:
/// `(x - m) - log(Σ exp(x - m))` with the same guarded max as
/// `logsumexp`. Last-axis path is the fused SIMD row kernel
/// (`logSoftmaxRows`, task-parallel over rows); other axes run the
/// streaming inner-lane kernel (`logSoftmaxInner`).
/// One f32 kernel; 16-bit inputs follow the `.widened` policy.
pub fn logSoftmax(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "logSoftmax");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    var out = try logSoftmaxF32(ctx, rank, xx.tensor(), axis);
    errdefer out.deinit();
    return ctx.storeAs(compute, dtype, out);
}

fn logSoftmaxF32(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
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
    if (inner == 1) {
        const base_task: LogRowsTask = .{
            .input = input,
            .output = output,
            .axis_dim = axis_dim,
            .row_start = 0,
            .row_end = outer,
        };
        if (outer > 1 and source.len() >= parallel.row_kernel_len_threshold) {
            if (ctx.dispatchRange(LogRowsTask, "row_start", "row_end", base_task, outer, runLogSoftmaxRowsTask)) {
                return out;
            }
        }
        logSoftmaxRows(base_task);
        return out;
    }

    var scratch = try ctx.empty(.f32, .{2 * inner});
    defer scratch.deinit();
    ctx.dispatchInnerLanes(SoftmaxInnerTask, .{
        .input = input,
        .output = output,
        .axis_dim = axis_dim,
        .inner = inner,
        .scratch = scratch.data(),
        .outer = outer,
        .inner_start = 0,
        .inner_end = inner,
    }, source.len(), inner, runLogSoftmaxInnerTask);
    return out;
}

/// One f32 kernel; 16-bit inputs follow the `.widened` policy.
pub fn softmax(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize) !tensor.TensorOf(dtype) {
    const compute = comptime ExecContext.widenedCompute(dtype, "softmax");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    var out = try softmaxF32(ctx, rank, xx.tensor(), axis);
    errdefer out.deinit();
    return ctx.storeAs(compute, dtype, out);
}

fn softmaxF32(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
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
    if (inner == 1) {
        const base_task: SoftmaxRowsTask = .{
            .input = input,
            .output = output,
            .axis_dim = axis_dim,
            .row_start = 0,
            .row_end = outer,
        };
        if (outer > 1 and source.len() >= parallel.row_kernel_len_threshold) {
            if (ctx.dispatchRange(SoftmaxRowsTask, "row_start", "row_end", base_task, outer, runSoftmaxRowsTask)) {
                return out;
            }
        }

        softmaxRows(base_task);
        return out;
    }

    var scratch = try ctx.empty(.f32, .{2 * inner});
    defer scratch.deinit();
    ctx.dispatchInnerLanes(SoftmaxInnerTask, .{
        .input = input,
        .output = output,
        .axis_dim = axis_dim,
        .inner = inner,
        .scratch = scratch.data(),
        .outer = outer,
        .inner_start = 0,
        .inner_end = inner,
    }, source.len(), inner, runSoftmaxInnerTask);
    return out;
}

pub fn softmaxExt(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, options: SoftmaxExtOptions) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    if (options.head_axis) |head_axis| {
        if (head_axis >= rank) return tensor.TensorError.InvalidShape;
        if ((options.max_bias > 0 or options.sinks != null) and head_axis == axis) return tensor.TensorError.InvalidShape;
        if ((options.max_bias > 0 or options.sinks != null) and source.shape[head_axis] == 0) return tensor.TensorError.InvalidShape;
    } else if (options.sinks) |sinks| {
        if (sinks.len != 1) return tensor.TensorError.InvalidShape;
    }
    if (options.causal_query_axis) |query_axis| {
        if (query_axis >= rank or query_axis == axis) return tensor.TensorError.InvalidShape;
        if (options.causal_source_offset > source.shape[axis]) return tensor.TensorError.InvalidShape;
        if (source.shape[query_axis] > source.shape[axis] - options.causal_source_offset) return tensor.TensorError.InvalidShape;
    } else if (options.causal_source_offset != 0) {
        return tensor.TensorError.InvalidShape;
    }
    if (options.max_bias > 0 and options.head_axis == null) return tensor.TensorError.InvalidArgument;
    if (options.max_bias > 0 and options.mask == null) return tensor.TensorError.InvalidArgument;

    const head_count = if (options.head_axis) |head_axis| source.shape[head_axis] else 1;
    if (options.sinks) |sinks| {
        if (sinks.len != head_count) return tensor.TensorError.InvalidShape;
    }

    var mask_value: ?Tensor = null;
    defer if (mask_value) |*mask| mask.deinit();
    if (options.mask) |mask| {
        mask_value = try mask.broadcastToRank(rank, source.shape);
    }
    if (mask_value) |*mask| mask.buffer.waitReady();
    const mask_ranked = if (mask_value) |*mask| try mask.rankView(rank) else null;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const source_strides = contiguousStridesArray(rank, source.shape);
    const head_log2 = floorPowerOfTwo(head_count);
    var slopes: ?[]f32 = null;
    defer if (slopes) |values| ctx.allocator.free(values);
    if (options.max_bias > 0) {
        const values = try ctx.allocator.alloc(f32, head_count);
        errdefer ctx.allocator.free(values);
        for (values, 0..) |*value, head_i| {
            value.* = alibiSlope(head_i, head_log2, options.max_bias);
        }
        slopes = values;
    }

    // Every (outer, inner) row is independent: one task type covers all
    // option combinations, with a SIMD body for contiguous rows and the
    // scalar per-row body for strided/exotic layouts.
    const rows = outer * inner;
    const simd_rows = inner == 1 and (mask_ranked == null or mask_ranked.?.strides[axis] == 1);
    const base_task: SoftmaxExtRowsTask(rank) = .{
        .input = input,
        .output = output,
        .shape = source.shape,
        .strides = source_strides,
        .mask = mask_ranked,
        .sinks = options.sinks,
        .slopes = slopes,
        .scale = options.scale,
        .head_axis = options.head_axis,
        .causal_query_axis = options.causal_query_axis,
        .causal_source_offset = options.causal_source_offset,
        .axis_dim = axis_dim,
        .inner = inner,
        .simd_rows = simd_rows,
        .row_start = 0,
        .row_end = rows,
    };
    if (rows > 1 and source.len() >= parallel.row_kernel_len_threshold) {
        if (ctx.dispatchRange(SoftmaxExtRowsTask(rank), "row_start", "row_end", base_task, rows, runSoftmaxExtRowsTask(rank, axis))) {
            return out;
        }
    }

    softmaxExtRows(rank, axis, base_task);
    return out;
}

/// Softmax VJP along `axis`: `scale_value` folds the forward's logit
/// scaling into the same pass (`1` for the plain softmax).
pub fn softmaxBackward(ctx: *ExecContext, comptime rank: usize, y: *const Tensor, gy: *const Tensor, comptime axis: usize, scale_value: f32) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
    if (axis >= rank) @compileError("axis out of bounds");
    try tensor.requireSameShape(y, gy);

    const source = try y.rankView(rank);
    var yy = try ctx.prepareContiguous(.f32, y);
    defer yy.deinit();
    var ggy = try ctx.prepareContiguous(.f32, gy);
    defer ggy.deinit();
    const yd = yy.tensor().dataConst();
    const gyd = ggy.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    if (inner == 1) {
        const base_task: SoftmaxBackwardRowsTask = .{
            .y = yd,
            .gy = gyd,
            .output = output,
            .axis_dim = axis_dim,
            .scale = scale_value,
            .row_start = 0,
            .row_end = outer,
        };
        if (outer > 1 and source.len() >= parallel.row_kernel_len_threshold) {
            if (ctx.dispatchRange(SoftmaxBackwardRowsTask, "row_start", "row_end", base_task, outer, runSoftmaxBackwardRowsTask)) {
                return out;
            }
        }

        softmaxBackwardRows(base_task);
        return out;
    }

    var scratch = try ctx.empty(.f32, .{inner});
    defer scratch.deinit();
    ctx.dispatchInnerLanes(SoftmaxBackwardInnerTask, .{
        .y = yd,
        .gy = gyd,
        .output = output,
        .axis_dim = axis_dim,
        .inner = inner,
        .scratch = scratch.data(),
        .scale = scale_value,
        .outer = outer,
        .inner_start = 0,
        .inner_end = inner,
    }, source.len(), inner, runSoftmaxBackwardInnerTask);
    return out;
}
