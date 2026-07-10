//! Softmax (plain + masked/sink/ALiBi/causal "ext") forward and backward.
//!
//! Domain module: every op receives an explicit `*Runtime`; the per-row SIMD
//! kernels + Task structs stay in the `row_ops` leaf (imported), so the hot
//! loops are untouched. Home of `SoftmaxExtOptions` (re-exported by `exec.zig`).

const std = @import("std");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

const exec_row_ops = @import("row_ops.zig");
const exec_shape = @import("shape.zig");
const Runtime = @import("runtime.zig").Runtime;

const Tensor = tensor.Tensor;

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
/// task-parallel over rows like `softmax`); other axes fall back to the
/// scalar strided loop with identical semantics.
pub fn logsumexpAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(out_rank, out_shape);
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
        if (outer > 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]LogRowsTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].row_start = task_i * outer / task_count;
                    tasks[task_i].row_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(LogRowsTask, tasks[0..task_count], runLogsumexpRowsTask);
                return out;
            }
        }
        logsumexpRows(base_task);
        return out;
    }

    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var max_value = input[base + inner_i];
            for (1..axis_dim) |axis_i| {
                max_value = @max(max_value, input[base + axis_i * inner + inner_i]);
            }
            const max_safe = if (std.math.isFinite(max_value)) max_value else 0;
            var sum_exp: f32 = 0;
            for (0..axis_dim) |axis_i| {
                sum_exp += @exp(input[base + axis_i * inner + inner_i] - max_safe);
            }
            output[outer_i * inner + inner_i] = max_safe + @log(sum_exp);
        }
    }
    return out;
}

/// Log-softmax along `axis` (torch.log_softmax), shape-preserving:
/// `(x - m) - log(Σ exp(x - m))` with the same guarded max as
/// `logsumexpAxisRank`. Last-axis path is the fused SIMD row kernel
/// (`logSoftmaxRows`, task-parallel over rows); other axes fall back to
/// the scalar strided loop.
pub fn logSoftmaxAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
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
        if (outer > 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]LogRowsTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].row_start = task_i * outer / task_count;
                    tasks[task_i].row_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(LogRowsTask, tasks[0..task_count], runLogSoftmaxRowsTask);
                return out;
            }
        }
        logSoftmaxRows(base_task);
        return out;
    }

    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var max_value = input[base + inner_i];
            for (1..axis_dim) |axis_i| {
                max_value = @max(max_value, input[base + axis_i * inner + inner_i]);
            }
            const max_safe = if (std.math.isFinite(max_value)) max_value else 0;
            var sum_exp: f32 = 0;
            for (0..axis_dim) |axis_i| {
                sum_exp += @exp(input[base + axis_i * inner + inner_i] - max_safe);
            }
            const shift = max_safe + @log(sum_exp);
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                output[offset] = input[offset] - shift;
            }
        }
    }
    return out;
}

pub fn softmaxAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
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
        if (outer > 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]SoftmaxRowsTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].row_start = task_i * outer / task_count;
                    tasks[task_i].row_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(SoftmaxRowsTask, tasks[0..task_count], runSoftmaxRowsTask);
                return out;
            }
        }

        softmaxRows(base_task);
        return out;
    }

    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var max_value = input[base + inner_i];
            for (1..axis_dim) |axis_i| {
                max_value = @max(max_value, input[base + axis_i * inner + inner_i]);
            }

            var sum_exp: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const value = @exp(input[offset] - max_value);
                output[offset] = value;
                sum_exp += value;
            }

            const inv_sum = 1 / sum_exp;
            for (0..axis_dim) |axis_i| {
                output[base + axis_i * inner + inner_i] *= inv_sum;
            }
        }
    }
    return out;
}

pub fn softmaxExtAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, options: SoftmaxExtOptions) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
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
    if (options.max_bias > 0 and options.head_axis == null) return tensor.TensorError.InvalidShape;
    if (options.max_bias > 0 and options.mask == null) return tensor.TensorError.InvalidShape;

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

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const source_strides = contiguousStridesArray(rank, source.shape);
    const head_log2 = floorPowerOfTwo(head_count);
    var slopes: ?[]f32 = null;
    defer if (slopes) |values| rt.allocator.free(values);
    if (options.max_bias > 0) {
        const values = try rt.allocator.alloc(f32, head_count);
        errdefer rt.allocator.free(values);
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
    if (rows > 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
        if (rt.workPool()) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), rows);
            var tasks: [parallel.vector_max_threads]SoftmaxExtRowsTask(rank) = undefined;
            for (0..task_count) |task_i| {
                tasks[task_i] = base_task;
                tasks[task_i].row_start = task_i * rows / task_count;
                tasks[task_i].row_end = (task_i + 1) * rows / task_count;
            }
            pool.parallelChunks(SoftmaxExtRowsTask(rank), tasks[0..task_count], runSoftmaxExtRowsTask(rank, axis));
            return out;
        }
    }

    softmaxExtRows(rank, axis, base_task);
    return out;
}

pub fn softmaxBackwardAxisRank(rt: *Runtime, comptime rank: usize, y: *const Tensor, gy: *const Tensor, comptime axis: usize) !Tensor {
    return softmaxExtBackwardAxisRank(rt, rank, y, gy, axis, 1);
}

pub fn softmaxExtBackwardAxisRank(rt: *Runtime, comptime rank: usize, y: *const Tensor, gy: *const Tensor, comptime axis: usize, scale_value: f32) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    try tensor.requireSameShape(y, gy);

    const source = try y.rankView(rank);
    var yy = try rt.prepareContiguous(y);
    defer yy.deinit();
    var ggy = try rt.prepareContiguous(gy);
    defer ggy.deinit();
    const yd = yy.tensor().dataConst();
    const gyd = ggy.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
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
        if (outer > 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]SoftmaxBackwardRowsTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].row_start = task_i * outer / task_count;
                    tasks[task_i].row_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(SoftmaxBackwardRowsTask, tasks[0..task_count], runSoftmaxBackwardRowsTask);
                return out;
            }
        }

        softmaxBackwardRows(base_task);
        return out;
    }

    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var dot_acc: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                dot_acc += gyd[offset] * yd[offset];
            }
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                output[offset] = scale_value * yd[offset] * (gyd[offset] - dot_acc);
            }
        }
    }
    return out;
}
