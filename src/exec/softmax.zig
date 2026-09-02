//! Softmax (plain + masked/sink/ALiBi/causal "ext") forward and backward.
//!
//! Domain module: every op receives an explicit `*ExecContext`; the per-row SIMD
//! kernels + Task structs live in the backend row-kernel leaf (`backend.rows`,
//! dispatched through `backend.kernels`). Home of `SoftmaxExtOptions`
//! (re-exported by `exec.zig`).

const std = @import("std");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");

const backend_mod = @import("../backend.zig");
const exec_row_ops = backend_mod.rows;
const shape_mod = @import("../shape.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const Tensor = tensor.Tensor;
const DType = dtype_mod.DType;

const productAfterAxis = shape_mod.productAfterAxis;
const productBeforeAxis = shape_mod.productBeforeAxis;
const contiguousStridesArray = shape_mod.contiguousStrides;

const SoftmaxRowsTask = exec_row_ops.SoftmaxRowsTask;
const logsumexpRows = backend_mod.kernels.logsumexpRows;
const logSoftmaxRows = backend_mod.kernels.logSoftmaxRows;
const shapeWithoutAxis = shape_mod.withoutAxis;
const SoftmaxExtRowsTask = exec_row_ops.SoftmaxExtRowsTask;
const SoftmaxBackwardRowsTask = exec_row_ops.SoftmaxBackwardRowsTask;
const SoftmaxInnerTask = exec_row_ops.SoftmaxInnerTask;
const SoftmaxBackwardInnerTask = exec_row_ops.SoftmaxBackwardInnerTask;
const runSoftmaxInnerTask = exec_row_ops.runSoftmaxInnerTask;
const runLogsumexpInnerTask = exec_row_ops.runLogsumexpInnerTask;
const runLogSoftmaxInnerTask = exec_row_ops.runLogSoftmaxInnerTask;
const runSoftmaxBackwardInnerTask = exec_row_ops.runSoftmaxBackwardInnerTask;

const softmaxRows = backend_mod.kernels.softmaxRows;
const softmaxExtRows = backend_mod.kernels.softmaxExtRows;
const softmaxBackwardRows = backend_mod.kernels.softmaxBackwardRows;

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
    return rowFamilyF32(ctx, rank, x, axis, .{ .reduces = true, .rows = logsumexpRows, .inner = runLogsumexpInnerTask });
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
    return rowFamilyF32(ctx, rank, x, axis, .{ .reduces = false, .rows = logSoftmaxRows, .inner = runLogSoftmaxInnerTask });
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
    return rowFamilyF32(ctx, rank, x, axis, .{ .reduces = false, .rows = softmaxRows, .inner = runSoftmaxInnerTask });
}

/// What distinguishes the three row ops over one driver: whether the axis
/// is removed, the fused last-axis row kernel, and the inner-lane kernel.
const RowFamily = struct {
    reduces: bool,
    rows: fn (SoftmaxRowsTask) void,
    inner: fn (*const SoftmaxInnerTask) void,
};

/// The one driver of the softmax family: last-axis rows through the fused
/// row kernel (pool-split above the row threshold), any other axis through
/// the inner-lane kernel over a `2 * inner` scratch.
fn rowFamilyF32(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, comptime family: RowFamily) !Tensor {
    const source = try x.rankView(rank);
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    const out_rank = if (rank == 1) 1 else rank - 1;
    const reduced_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);
    const out_shape: []const usize = if (family.reduces) &reduced_shape else &source.shape;
    var out = try ctx.empty(.f32, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const g = shape_mod.AxisGeometry.of(rank, source.shape, axis);
    if (g.inner == 1) {
        ctx.dispatchRangeOr(SoftmaxRowsTask, "row_start", "row_end", .{
            .input = input,
            .output = output,
            .axis_dim = g.axis_dim,
            .row_start = 0,
            .row_end = g.outer,
        }, g.outer, g.outer > 1 and source.len() >= parallel.row_kernel_len_threshold, family.rows);
        return out;
    }

    var scratch = try ctx.empty(.f32, .{2 * g.inner});
    defer scratch.deinit();
    ctx.dispatchInnerLanes(SoftmaxInnerTask, .{
        .input = input,
        .output = output,
        .axis_dim = g.axis_dim,
        .inner = g.inner,
        .scratch = scratch.data(),
        .outer = g.outer,
        .inner_start = 0,
        .inner_end = g.inner,
    }, source.len(), g.inner, family.inner);
    return out;
}

pub fn softmaxExt(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, options: SoftmaxExtOptions) !Tensor {
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
    const head_log2 = std.math.floorPowerOfTwo(usize, head_count);
    var slopes: ?[]f32 = null;
    defer if (slopes) |values| ctx.allocator().free(values);
    if (options.max_bias > 0) {
        const values = try ctx.allocator().alloc(f32, head_count);
        errdefer ctx.allocator().free(values);
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
    const Kernel = struct {
        fn run(task: SoftmaxExtRowsTask(rank)) void {
            softmaxExtRows(rank, axis, task);
        }
    };
    ctx.dispatchRangeOr(SoftmaxExtRowsTask(rank), "row_start", "row_end", base_task, rows, rows > 1 and source.len() >= parallel.row_kernel_len_threshold, Kernel.run);
    return out;
}

/// Softmax VJP along `axis`: `scale_value` folds the forward's logit
/// scaling into the same pass (`1` for the plain softmax).
pub fn softmaxBackward(ctx: *ExecContext, comptime rank: usize, y: *const Tensor, gy: *const Tensor, comptime axis: usize, scale_value: f32) !Tensor {
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
        ctx.dispatchRangeOr(SoftmaxBackwardRowsTask, "row_start", "row_end", base_task, outer, outer > 1 and source.len() >= parallel.row_kernel_len_threshold, softmaxBackwardRows);
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

/// The ggml ALiBi slope schedule: head `i` of `head_log2` (the head count
/// rounded down to a power of two) takes `2^(-max_bias/head_log2)` powers,
/// the overflow heads the half-step schedule.
fn alibiSlope(head_i: usize, head_log2: usize, max_bias: f32) f32 {
    const head_log2_f: f32 = @floatFromInt(head_log2);
    const m0 = std.math.pow(f32, 2, -max_bias / head_log2_f);
    const m1 = std.math.pow(f32, 2, -(max_bias / 2) / head_log2_f);
    if (head_i < head_log2) {
        const exponent: f32 = @floatFromInt(head_i + 1);
        return std.math.pow(f32, m0, exponent);
    }
    const exponent: f32 = @floatFromInt(2 * (head_i - head_log2) + 1);
    return std.math.pow(f32, m1, exponent);
}
