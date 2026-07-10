//! RMSNorm (+ weighted / +residual / fused-rope variants) and LayerNorm
//! (plain + affine) forward and backward.
//!
//! Domain module: every op receives an explicit `*Runtime`. Per-row SIMD
//! kernels + Task structs stay in the `row_ops` leaf; the fused rms-norm+rope
//! kernel reads the rope table's pub `sinValues()`/`cosValues()`. Home of
//! `LayerNormAffineBackwardResult` (re-exported by `exec.zig`).

const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

const exec_row_ops = @import("row_ops.zig");
const exec_shape = @import("shape.zig");
const exec_rope = @import("rope.zig");
const Runtime = @import("runtime.zig").Runtime;

const Tensor = tensor.Tensor;
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
const LayerNormRowsTask = exec_row_ops.LayerNormRowsTask;
const LayerNormBackwardInputRowsTask = exec_row_ops.LayerNormBackwardInputRowsTask;
const LayerNormRowStatsTask = exec_row_ops.LayerNormRowStatsTask;
const LayerNormParamGradColumnsTask = exec_row_ops.LayerNormParamGradColumnsTask;
const runRmsNormMulRopeHalfTask = exec_row_ops.runRmsNormMulRopeHalfTask;
const runRmsNormMulRowsTask = exec_row_ops.runRmsNormMulRowsTask;
const runRmsNormMulAddRowsTask = exec_row_ops.runRmsNormMulAddRowsTask;
const runRmsNormMulBackwardInputRowsTask = exec_row_ops.runRmsNormMulBackwardInputRowsTask;
const runRmsNormMulBackwardWeightRowsTask = exec_row_ops.runRmsNormMulBackwardWeightRowsTask;
const runLayerNormRowsTask = exec_row_ops.runLayerNormRowsTask;
const runLayerNormBackwardInputRowsTask = exec_row_ops.runLayerNormBackwardInputRowsTask;
const runLayerNormRowStatsTask = exec_row_ops.runLayerNormRowStatsTask;
const runLayerNormParamGradColumnsTask = exec_row_ops.runLayerNormParamGradColumnsTask;
const rmsNormMulRopeHalfVectors = exec_row_ops.rmsNormMulRopeHalfVectors;
const rmsNormMulRows = exec_row_ops.rmsNormMulRows;
const rmsNormMulAddRows = exec_row_ops.rmsNormMulAddRows;
const rmsNormMulBackwardInputRows = exec_row_ops.rmsNormMulBackwardInputRows;
const rmsNormMulBackwardWeightRows = exec_row_ops.rmsNormMulBackwardWeightRows;
const layerNormRows = exec_row_ops.layerNormRows;
const layerNormBackwardInputRows = exec_row_ops.layerNormBackwardInputRows;
const layerNormAffineParamGradRows = exec_row_ops.layerNormAffineParamGradRows;
const layerNormRowStats = exec_row_ops.layerNormRowStats;
const layerNormParamGradColumns = exec_row_ops.layerNormParamGradColumns;

pub const LayerNormAffineBackwardResult = struct {
    input: ?Tensor = null,
    weight: ?Tensor = null,
    bias: ?Tensor = null,

    pub fn deinit(self: *LayerNormAffineBackwardResult) void {
        if (self.input) |*value| value.deinit();
        if (self.weight) |*value| value.deinit();
        if (self.bias) |*value| value.deinit();
        self.* = undefined;
    }
};

pub fn rmsNormAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
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
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
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
                output[offset] = input[offset] * scale_value;
            }
        }
    }
    return out;
}

pub fn rmsNormMulAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, weight: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const weight_view = try weight.rankView(1);
    const axis_dim = source.shape[axis];
    if (weight_view.dim(0) != axis_dim) return tensor.TensorError.ShapeMismatch;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();
    const input = xx.tensor().dataConst();
    const weights = ww.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner == 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
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
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]RmsNormMulRowsTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].row_start = task_i * outer / task_count;
                    tasks[task_i].row_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(RmsNormMulRowsTask, tasks[0..task_count], runRmsNormMulRowsTask);
                return out;
            }
        }

        rmsNormMulRows(base_task);
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

pub fn rmsNormMulAddAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, weight: *const Tensor, residual: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    try tensor.requireSameShape(x, residual);

    const source = try x.rankView(rank);
    const weight_view = try weight.rankView(1);
    const axis_dim = source.shape[axis];
    if (weight_view.dim(0) != axis_dim) return tensor.TensorError.ShapeMismatch;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();
    var rr = try rt.prepareContiguous(residual);
    defer rr.deinit();
    const input = xx.tensor().dataConst();
    const weights = ww.tensor().dataConst();
    const residual_data = rr.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner == 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
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
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]RmsNormMulAddRowsTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].row_start = task_i * outer / task_count;
                    tasks[task_i].row_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(RmsNormMulAddRowsTask, tasks[0..task_count], runRmsNormMulAddRowsTask);
                return out;
            }
        }

        rmsNormMulAddRows(base_task);
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

pub fn rmsNormMulBackwardInputAxisRank(
    rt: *Runtime,
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

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();
    var ggy = try rt.prepareContiguous(gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const weights = ww.tensor().dataConst();
    const grad = ggy.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner == 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
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
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]RmsNormMulBackwardInputRowsTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].row_start = task_i * outer / task_count;
                    tasks[task_i].row_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(RmsNormMulBackwardInputRowsTask, tasks[0..task_count], runRmsNormMulBackwardInputRowsTask);
                return out;
            }
        }

        rmsNormMulBackwardInputRows(base_task);
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

pub fn rmsNormMulBackwardWeightAxisRank(
    rt: *Runtime,
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

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var ggy = try rt.prepareContiguous(gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const grad = ggy.tensor().dataConst();

    var out = try rt.zerosRank(1, .{axis_dim});
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner == 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
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
        if (outer > 1) {
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]RmsNormMulBackwardWeightRowsTask = undefined;
                var partials = try rt.allocator.alloc(f32, task_count * axis_dim);
                defer rt.allocator.free(partials);
                @memset(partials, 0);
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].output = partials[task_i * axis_dim ..][0..axis_dim];
                    tasks[task_i].row_start = task_i * outer / task_count;
                    tasks[task_i].row_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(RmsNormMulBackwardWeightRowsTask, tasks[0..task_count], runRmsNormMulBackwardWeightRowsTask);

                const Vec = @Vector(8, f32);
                const vector_width = 8;
                for (0..task_count) |task_i| {
                    const partial = partials[task_i * axis_dim ..][0..axis_dim];
                    var axis_i: usize = 0;
                    while (axis_i + vector_width <= axis_dim) : (axis_i += vector_width) {
                        const current: Vec = output[axis_i..][0..vector_width].*;
                        const addend: Vec = partial[axis_i..][0..vector_width].*;
                        output[axis_i..][0..vector_width].* = current + addend;
                    }
                    while (axis_i < axis_dim) : (axis_i += 1) {
                        output[axis_i] += partial[axis_i];
                    }
                }
                return out;
            }
        }

        rmsNormMulBackwardWeightRows(base_task);
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

pub fn rmsNormMulRopeAxisRankWithTable(
    rt: *Runtime,
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

    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();
    @constCast(x.buffer).waitReady();
    const input = x.buffer.data;
    const weights = ww.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
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

    if (input_feature_stride == 1 and output_feature_stride == 1 and mode == .half and source.len() >= parallel.vector_elementwise_len_threshold / 8) {
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

        if (total_vectors > 1) {
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), total_vectors);
                var tasks: [parallel.vector_max_threads]RmsNormMulRopeHalfTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].vector_start = task_i * total_vectors / task_count;
                    tasks[task_i].vector_end = (task_i + 1) * total_vectors / task_count;
                }
                pool.parallelChunks(RmsNormMulRopeHalfTask, tasks[0..task_count], runRmsNormMulRopeHalfTask);
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

            const first_feature = switch (mode) {
                .interleaved => 2 * pair_i,
                .half => pair_i,
            };
            const second_feature = switch (mode) {
                .interleaved => 2 * pair_i + 1,
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

pub fn rmsNormBackwardAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    try tensor.requireSameShape(x, gy);

    const source = try x.rankView(rank);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var ggy = try rt.prepareContiguous(gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const gyd = ggy.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
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
pub fn layerNormAffineRows(
    rt: *Runtime,
    input: []const f32,
    rows: usize,
    cols: usize,
    weight: []const f32,
    bias: []const f32,
    eps: f32,
) !Tensor {
    if (input.len != rows * cols) return tensor.TensorError.InvalidDataLength;
    if (weight.len != cols or bias.len != cols) return tensor.TensorError.ShapeMismatch;

    var out = try rt.emptyRank(2, .{ rows, cols });
    errdefer out.deinit();
    const base_task: LayerNormRowsTask = .{
        .input = input,
        .weights = weight,
        .biases = bias,
        .output = out.data(),
        .axis_dim = cols,
        .inv_axis_dim = 1 / @as(f32, @floatFromInt(cols)),
        .eps = eps,
        .row_start = 0,
        .row_end = rows,
    };
    if (input.len >= parallel.vector_elementwise_len_threshold / 2 and rows > 1) {
        if (rt.workPool()) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), rows);
            var tasks: [parallel.vector_max_threads]LayerNormRowsTask = undefined;
            for (0..task_count) |task_i| {
                tasks[task_i] = base_task;
                tasks[task_i].row_start = task_i * rows / task_count;
                tasks[task_i].row_end = (task_i + 1) * rows / task_count;
            }
            pool.parallelChunks(LayerNormRowsTask, tasks[0..task_count], runLayerNormRowsTask);
            return out;
        }
    }
    layerNormRows(base_task);
    return out;
}

pub fn layerNormAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
    return layerNormDispatchAxisRank(rt, rank, x, null, null, axis, eps);
}

/// Fused affine LayerNorm: layerNormAxisRank followed by `* weight + bias`
/// (both rank-1 of the axis length) in the same row pass.
pub fn layerNormAffineAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    weight: *const Tensor,
    bias: *const Tensor,
    comptime axis: usize,
    eps: f32,
) !Tensor {
    const source = try x.rankView(rank);
    const axis_dim = source.shape[axis];
    const weight_view = try weight.rankView(1);
    if (weight_view.dim(0) != axis_dim) return tensor.TensorError.ShapeMismatch;
    const bias_view = try bias.rankView(1);
    if (bias_view.dim(0) != axis_dim) return tensor.TensorError.ShapeMismatch;

    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();
    var bb = try rt.prepareContiguous(bias);
    defer bb.deinit();
    return layerNormDispatchAxisRank(rt, rank, x, ww.tensor().dataConst(), bb.tensor().dataConst(), axis, eps);
}

fn layerNormDispatchAxisRank(
    rt: *Runtime,
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
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    if (inner == 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
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
            if (rt.workPool()) |pool| {
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                var tasks: [parallel.vector_max_threads]LayerNormRowsTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].row_start = task_i * outer / task_count;
                    tasks[task_i].row_end = (task_i + 1) * outer / task_count;
                }
                pool.parallelChunks(LayerNormRowsTask, tasks[0..task_count], runLayerNormRowsTask);
                return out;
            }
        }

        layerNormRows(base_task);
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
                if (weights) |w| value = value * w[axis_i] + biases.?[axis_i];
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
pub fn groupNormAxisRank(
    rt: *Runtime,
    x: *const Tensor,
    groups: usize,
    eps: f32,
    weight: ?*const Tensor,
    bias: ?*const Tensor,
) !Tensor {
    const source = try x.rankView(2);
    const rows = source.shape[0];
    const cols = source.shape[1];
    if (rows == 0 or cols == 0) return tensor.TensorError.InvalidShape;
    if (groups == 0 or cols % groups != 0) return tensor.TensorError.InvalidShape;

    var ww: ?Runtime.PreparedTensor = null;
    defer if (ww) |*p| p.deinit();
    var weight_slice: ?[]const f32 = null;
    if (weight) |w| {
        const wv = try w.rankView(1);
        if (wv.shape[0] != cols) return tensor.TensorError.ShapeMismatch;
        ww = try rt.prepareContiguous(w);
        weight_slice = ww.?.tensor().dataConst();
    }
    var bb: ?Runtime.PreparedTensor = null;
    defer if (bb) |*p| p.deinit();
    var bias_slice: ?[]const f32 = null;
    if (bias) |b| {
        const bv = try b.rankView(1);
        if (bv.shape[0] != cols) return tensor.TensorError.ShapeMismatch;
        bb = try rt.prepareContiguous(b);
        bias_slice = bb.?.tensor().dataConst();
    }

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();

    var out = try rt.emptyRank(2, .{ rows, cols });
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(rows * cols, parallel.vector_elementwise_len_threshold);
    rt.backend.groupNormInto(&out, xx.tensor(), weight_slice, bias_slice, rows, cols, groups, eps);
    return out;
}

pub const GroupNormBackwardResult = struct {
    input: ?Tensor = null,
    weight: ?Tensor = null,
    bias: ?Tensor = null,

    pub fn deinit(self: *GroupNormBackwardResult) void {
        if (self.input) |*value| value.deinit();
        if (self.weight) |*value| value.deinit();
        if (self.bias) |*value| value.deinit();
        self.* = undefined;
    }
};

/// VJP of groupNormAxisRank. Computes only the requested gradients:
///   dx = (1/σ_g)·(ĝ − mean_G(ĝ) − x̂·mean_G(ĝ·x̂)) per group, with
///        ĝ = gy⊙weight (or gy when the forward had no affine weight) and
///        the group statistics RECOMPUTED from `x` with the forward's exact
///        f64-accumulate / f32-apply policy (the layerNorm VJP convention —
///        nothing is saved from the forward);
///   dweight[c] = Σ_t gy[t,c]·x̂[t,c];  dbias[c] = Σ_t gy[t,c].
/// `weight` must be the forward's weight when it had one (it feeds dx) and
/// null otherwise. One backend kernel fills all requested outputs, parallel
/// over whole groups (disjoint column slices ⇒ bitwise identical for any
/// thread count).
pub fn groupNormBackwardAxisRank(
    rt: *Runtime,
    x: *const Tensor,
    gy: *const Tensor,
    groups: usize,
    eps: f32,
    weight: ?*const Tensor,
    need_input: bool,
    need_weight: bool,
    need_bias: bool,
) !GroupNormBackwardResult {
    const source = try x.rankView(2);
    const rows = source.shape[0];
    const cols = source.shape[1];
    if (rows == 0 or cols == 0) return tensor.TensorError.InvalidShape;
    if (groups == 0 or cols % groups != 0) return tensor.TensorError.InvalidShape;
    try tensor.requireSameShape(x, gy);

    var ww: ?Runtime.PreparedTensor = null;
    defer if (ww) |*p| p.deinit();
    var weight_slice: ?[]const f32 = null;
    if (weight) |w| {
        const wv = try w.rankView(1);
        if (wv.shape[0] != cols) return tensor.TensorError.ShapeMismatch;
        ww = try rt.prepareContiguous(w);
        weight_slice = ww.?.tensor().dataConst();
    }

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();

    var result = GroupNormBackwardResult{};
    errdefer result.deinit();
    if (need_input) result.input = try rt.emptyRank(2, .{ rows, cols });
    if (need_weight) result.weight = try rt.emptyRank(1, .{cols});
    if (need_bias) result.bias = try rt.emptyRank(1, .{cols});
    if (!need_input and !need_weight and !need_bias) return result;

    rt.enableNativeVectorPoolForWork(rows * cols, parallel.vector_elementwise_len_threshold);
    rt.backend.groupNormBackwardInto(
        if (result.input) |*t| t else null,
        if (result.weight) |*t| t else null,
        if (result.bias) |*t| t else null,
        xx.tensor(),
        gg.tensor(),
        weight_slice,
        rows,
        cols,
        groups,
        eps,
    );
    return result;
}

/// VJP of layerNormAxisRank (dx only):
/// dx = (1/σ)(gy − mean(gy) − x̂·mean(gy·x̂)) with x̂ = (x−μ)/σ.
pub fn layerNormBackwardAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, gy: *const Tensor, comptime axis: usize, eps: f32) !Tensor {
    const result = try layerNormBackwardDispatchAxisRank(rt, rank, x, null, gy, axis, eps, true, false, false);
    return result.input.?;
}

/// VJP of layerNormAffineAxisRank. Computes only the requested gradients:
/// dx as in layerNormBackwardAxisRank with g' = gy⊙weight in place of gy,
/// dweight = Σ_rows gy⊙x̂, dbias = Σ_rows gy. The dweight/dbias row
/// reduction always accumulates each column in row order — serially for
/// small inputs (layerNormAffineParamGradRows), column-partitioned across
/// the pool for large ones (layerNormParamGradColumns) — so it is bitwise
/// identical for any thread count.
pub fn layerNormAffineBackwardAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    weight: *const Tensor,
    gy: *const Tensor,
    comptime axis: usize,
    eps: f32,
    need_input: bool,
    need_weight: bool,
    need_bias: bool,
) !LayerNormAffineBackwardResult {
    const source = try x.rankView(rank);
    const weight_view = try weight.rankView(1);
    if (weight_view.dim(0) != source.shape[axis]) return tensor.TensorError.ShapeMismatch;

    var ww = try rt.prepareContiguous(weight);
    defer ww.deinit();
    return layerNormBackwardDispatchAxisRank(rt, rank, x, ww.tensor().dataConst(), gy, axis, eps, need_input, need_weight, need_bias);
}

fn layerNormBackwardDispatchAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    weights: ?[]const f32,
    gy: *const Tensor,
    comptime axis: usize,
    eps: f32,
    need_input: bool,
    need_weight: bool,
    need_bias: bool,
) !LayerNormAffineBackwardResult {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    try tensor.requireSameShape(x, gy);

    const source = try x.rankView(rank);
    const axis_dim = source.shape[axis];

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var ggy = try rt.prepareContiguous(gy);
    defer ggy.deinit();
    const input = xx.tensor().dataConst();
    const grad = ggy.tensor().dataConst();

    var result = LayerNormAffineBackwardResult{};
    errdefer result.deinit();
    if (need_input) result.input = try rt.emptyRank(rank, source.shape);
    if (need_weight) result.weight = try rt.zerosRank(1, .{axis_dim});
    if (need_bias) result.bias = try rt.zerosRank(1, .{axis_dim});
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
            if (outer > 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
                if (rt.workPool()) |pool| {
                    const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
                    var tasks: [parallel.vector_max_threads]LayerNormBackwardInputRowsTask = undefined;
                    for (0..task_count) |task_i| {
                        tasks[task_i] = base_task;
                        tasks[task_i].row_start = task_i * outer / task_count;
                        tasks[task_i].row_end = (task_i + 1) * outer / task_count;
                    }
                    pool.parallelChunks(LayerNormBackwardInputRowsTask, tasks[0..task_count], runLayerNormBackwardInputRowsTask);
                    dispatched = true;
                }
            }
            if (!dispatched) layerNormBackwardInputRows(base_task);
        }

        if (need_weight or need_bias) {
            var dispatched = false;
            if (outer > 1 and axis_dim > 1 and source.len() >= parallel.vector_elementwise_len_threshold / 2) {
                if (rt.workPool()) |pool| {
                    // Per-row {mean, 1/σ} scratch, then the
                    // column-partitioned accumulation: both stages are
                    // bitwise identical for any thread count (see the
                    // task structs / kernels), unlike per-task row
                    // partials combined in task order.
                    var stats: []f32 = &.{};
                    defer if (stats.len > 0) rt.allocator.free(stats);
                    if (need_weight) {
                        stats = try rt.allocator.alloc(f32, 2 * outer);
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

    // Generic inner>1 scalar fallback: everything serial.
    const dx_data: ?[]f32 = if (result.input) |*value| value.data() else null;
    const dweight: ?[]f32 = if (result.weight) |*value| value.data() else null;
    const dbias: ?[]f32 = if (result.bias) |*value| value.data() else null;
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sum_acc: f32 = 0;
            var gsum: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                sum_acc += input[offset];
                gsum += grad[offset] * (if (weights) |w| w[axis_i] else 1);
            }
            const mean_value = sum_acc * inv_axis_dim;
            var sumsq: f32 = 0;
            var dot_acc: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const centered = input[offset] - mean_value;
                sumsq += centered * centered;
                dot_acc += grad[offset] * (if (weights) |w| w[axis_i] else 1) * centered;
            }
            const inv_sigma = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            const shift = gsum * inv_axis_dim * inv_sigma;
            const correction = dot_acc * inv_axis_dim * inv_sigma * inv_sigma * inv_sigma;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const centered = input[offset] - mean_value;
                if (dx_data) |dx| {
                    dx[offset] = grad[offset] * (if (weights) |w| w[axis_i] else 1) * inv_sigma - shift - centered * correction;
                }
                if (dweight) |weight_out| weight_out[axis_i] += grad[offset] * centered * inv_sigma;
                if (dbias) |bias_out| bias_out[axis_i] += grad[offset];
            }
        }
    }
    return result;
}
