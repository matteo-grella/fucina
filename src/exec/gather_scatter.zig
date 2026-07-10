//! Indexing / structural ops: narrow, concat (incl. quantized rows), pad,
//! gather, set-slice/set-rows, zero-slice/zero-rows, slice-gradient,
//! scatter-add, and the relative-position shift.
//!
//! Domain module: every op receives an explicit `*Runtime`. The scatter-add
//! embedding-gradient kernel + Task stay in the `row_ops` leaf; the private
//! `writeSlice*`/`writeRows*`/`validateUniqueIndices` helpers move here with
//! their callers.

const std = @import("std");
const parallel = @import("../parallel.zig");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");

const exec_row_ops = @import("row_ops.zig");
const exec_shape = @import("shape.zig");
const Runtime = @import("runtime.zig").Runtime;

const DType = tensor.DType;
const Tensor = tensor.Tensor;

const productAfterAxis = exec_shape.productAfterAxis;
const productBeforeAxis = exec_shape.productBeforeAxis;
const contiguousStridesArray = exec_shape.contiguousStridesArray;
const coordinateForLinear = exec_shape.coordinateForLinear;

const ScatterAddRowsTask = exec_row_ops.ScatterAddRowsTask;
const runScatterAddRowsTask = exec_row_ops.runScatterAddRowsTask;

pub fn relposShiftRank3(rt: *Runtime, bd: *const Tensor, t_k: usize) !Tensor {
    const v = try bd.rankView(3);
    const h = v.shape[0];
    const t_q = v.shape[1];
    const p = v.shape[2];
    if (t_q == 0 or t_k == 0 or p < t_k + t_q - 1) return tensor.TensorError.InvalidShape;

    var bb = try rt.prepareContiguous(bd);
    defer bb.deinit();
    const input = bb.tensor().dataConst();

    var out = try rt.emptyRank(3, .{ h, t_q, t_k });
    errdefer out.deinit();
    const output = out.data();

    for (0..h) |hh| {
        for (0..t_q) |qi| {
            const in_row = (hh * t_q + qi) * p + ((t_q - 1) - qi); // base offset for this query
            const out_row = (hh * t_q + qi) * t_k;
            for (0..t_k) |kj| output[out_row + kj] = input[in_row + kj];
        }
    }
    return out;
}

pub fn narrowAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, start: usize, length: usize) !Tensor {
    return narrowAxisRankTyped(rt, .f32, rank, x, axis, start, length);
}

pub fn narrowAxisRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    x: *const tensor.TensorOf(dtype),
    comptime axis: usize,
    start: usize,
    length: usize,
) !tensor.TensorOf(dtype) {
    _ = rt;
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    if (length == 0) return tensor.TensorError.InvalidShape;

    const source = try x.rankView(rank);
    if (start > source.shape[axis] or length > source.shape[axis] - start) return tensor.TensorError.IndexOutOfBounds;

    var out_shape = source.shape;
    out_shape[axis] = length;
    const offset_delta = try std.math.mul(usize, start, source.strides[axis]);
    return x.viewWithStridesOffset(out_shape[0..], source.strides[0..], offset_delta);
}

pub fn concatAxisRank(rt: *Runtime, comptime rank: usize, inputs: []const *const Tensor, comptime axis: usize) !Tensor {
    return concatAxisRankTyped(rt, .f32, rank, inputs, axis);
}

pub fn concatAxisRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    inputs: []const *const tensor.TensorOf(dtype),
    comptime axis: usize,
) !tensor.TensorOf(dtype) {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    if (inputs.len == 0) return tensor.TensorError.InvalidShape;

    const first = try inputs[0].rankView(rank);
    var out_shape = first.shape;
    out_shape[axis] = 0;
    for (inputs) |input| {
        const view = try input.rankView(rank);
        inline for (0..rank) |dim| {
            if (dim != axis and view.shape[dim] != first.shape[dim]) return tensor.TensorError.ShapeMismatch;
        }
        out_shape[axis] = try std.math.add(usize, out_shape[axis], view.shape[axis]);
    }
    if (out_shape[axis] == 0) return tensor.TensorError.InvalidShape;

    var out = try rt.emptyRankTyped(dtype, rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, out_shape, axis);
    const outer = productBeforeAxis(rank, out_shape, axis);
    if (comptime axis == rank - 1) {
        var all_contiguous = true;
        for (inputs) |input| {
            if (!input.isContiguous()) {
                all_contiguous = false;
                break;
            }
        }

        if (all_contiguous) {
            for (0..outer) |outer_i| {
                var dst_base = outer_i * out_shape[axis];
                for (inputs) |input| {
                    const view = try input.rankView(rank);
                    const copy_len = view.shape[axis];
                    const src_base = outer_i * copy_len;
                    @memcpy(output[dst_base..][0..copy_len], input.dataConst()[src_base..][0..copy_len]);
                    dst_base += copy_len;
                }
            }
            return out;
        }
    }

    var axis_offset: usize = 0;
    for (inputs) |input| {
        const view = try input.rankView(rank);
        var prepared = try rt.prepareContiguousTyped(dtype, input);
        defer prepared.deinit();
        const input_data = prepared.tensor().dataConst();
        const copy_len = view.shape[axis] * inner;
        for (0..outer) |outer_i| {
            const dst_base = outer_i * out_shape[axis] * inner + axis_offset * inner;
            const src_base = outer_i * view.shape[axis] * inner;
            @memcpy(output[dst_base..][0..copy_len], input_data[src_base..][0..copy_len]);
        }
        axis_offset += view.shape[axis];
    }

    return out;
}

pub fn concatQuantizedRowsTyped(
    rt: *Runtime,
    comptime dtype: DType,
    inputs: []const *const tensor.TensorOf(dtype),
) !tensor.TensorOf(dtype) {
    comptime if (!dtype_mod.isBlockQuantized(dtype)) @compileError("concatQuantizedRowsTyped requires a block-quantized dtype");
    if (inputs.len == 0) return tensor.TensorError.InvalidShape;

    const first = try inputs[0].rankView(2);
    const cols = first.dim(1);
    const blocks_per_row = try backend_mod.quantized_matmul.blockCountForDType(dtype, cols);
    var rows: usize = 0;
    for (inputs) |input| {
        const view = try input.rankView(2);
        if (view.dim(1) != cols) return tensor.TensorError.ShapeMismatch;
        if (!input.isContiguous()) return tensor.TensorError.UnsupportedView;
        rows = try std.math.add(usize, rows, view.dim(0));
    }
    if (rows == 0) return tensor.TensorError.InvalidShape;

    var out = try rt.emptyRankTyped(dtype, 2, .{ rows, cols });
    errdefer out.deinit();
    const output = out.data();

    var dst_base: usize = 0;
    for (inputs) |input| {
        const view = try input.rankView(2);
        const copy_len = view.dim(0) * blocks_per_row;
        @memcpy(output[dst_base..][0..copy_len], input.dataConst()[0..copy_len]);
        dst_base += copy_len;
    }

    return out;
}

/// Constant padding along `axis` (torch F.pad with mode='constant' on one
/// dim): the output grows by `before + after` on that axis, the body is
/// copied at offset `before`, and the pad positions hold `fill`. The VJP is
/// a narrow of the upstream gradient at offset `before` (`PadBackward`).
pub fn padAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    comptime axis: usize,
    before: usize,
    after: usize,
    fill: f32,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var out_shape = source.shape;
    out_shape[axis] = try std.math.add(usize, source.shape[axis], try std.math.add(usize, before, after));

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(rank, out_shape);
    errdefer out.deinit();
    const output = out.data();
    @memset(output, fill);

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const copy_len = source.shape[axis] * inner;
    for (0..outer) |outer_i| {
        const dst_base = outer_i * out_shape[axis] * inner + before * inner;
        const src_base = outer_i * copy_len;
        @memcpy(output[dst_base..][0..copy_len], input[src_base..][0..copy_len]);
    }
    return out;
}

pub fn gatherAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, indices: []const usize) !Tensor {
    return gatherAxisRankTyped(rt, .f32, rank, x, axis, indices);
}

pub fn gatherAxisRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    x: *const tensor.TensorOf(dtype),
    comptime axis: usize,
    indices: []const usize,
) !tensor.TensorOf(dtype) {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    if (indices.len == 0) return tensor.TensorError.InvalidShape;

    const source = try x.rankView(rank);
    for (indices) |index| {
        if (index >= source.shape[axis]) return tensor.TensorError.IndexOutOfBounds;
    }

    var out_shape = source.shape;
    out_shape[axis] = indices.len;

    var xx = try rt.prepareContiguousTyped(dtype, x);
    defer xx.deinit();
    const xp = xx.tensor();
    const input = xp.dataConst();

    var out = try rt.emptyRankTyped(dtype, rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    if (comptime axis == 0) {
        const row_len = productAfterAxis(rank, source.shape, 0);
        for (indices, 0..) |index, row| {
            const src = input[index * row_len ..][0..row_len];
            const dst = output[row * row_len ..][0..row_len];
            @memcpy(dst, src);
        }
        return out;
    }

    const source_strides = contiguousStridesArray(rank, source.shape);
    const out_strides = contiguousStridesArray(rank, out_shape);
    for (output, 0..) |*dst, linear| {
        var remainder = linear;
        var src_linear: usize = 0;
        inline for (0..rank) |dim| {
            const coord = remainder / out_strides[dim];
            remainder %= out_strides[dim];
            const src_coord = if (dim == axis) indices[coord] else coord;
            src_linear += src_coord * source_strides[dim];
        }
        dst.* = input[src_linear];
    }

    return out;
}

pub fn setSliceAxisRank(rt: *Runtime, comptime rank: usize, base: *const Tensor, update: *const Tensor, comptime axis: usize, start: usize) !Tensor {
    return setSliceAxisRankTyped(rt, .f32, rank, base, update, axis, start);
}

pub fn setSliceAxisRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    base: *const tensor.TensorOf(dtype),
    update: *const tensor.TensorOf(dtype),
    comptime axis: usize,
    start: usize,
) !tensor.TensorOf(dtype) {
    const source = try base.rankView(rank);
    const uv = try update.rankView(rank);
    if (axis >= rank) @compileError("axis out of bounds");
    if (start > source.shape[axis] or uv.shape[axis] > source.shape[axis] - start) return tensor.TensorError.IndexOutOfBounds;
    inline for (0..rank) |dim| {
        if (dim != axis and uv.shape[dim] != source.shape[dim]) return tensor.TensorError.ShapeMismatch;
    }

    var out = try rt.materializeTyped(dtype, base);
    errdefer out.deinit();
    try writeSliceAxisRankTyped(rt, dtype, rank, &out, update, axis, start);
    return out;
}

pub fn setRowsAxisRank(rt: *Runtime, comptime rank: usize, base: *const Tensor, update: *const Tensor, comptime axis: usize, indices: []const usize) !Tensor {
    return setRowsAxisRankTyped(rt, .f32, rank, base, update, axis, indices);
}

pub fn setRowsAxisRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    base: *const tensor.TensorOf(dtype),
    update: *const tensor.TensorOf(dtype),
    comptime axis: usize,
    indices: []const usize,
) !tensor.TensorOf(dtype) {
    const source = try base.rankView(rank);
    const uv = try update.rankView(rank);
    if (axis >= rank) @compileError("axis out of bounds");
    if (indices.len == 0 or uv.shape[axis] != indices.len) return tensor.TensorError.InvalidShape;
    inline for (0..rank) |dim| {
        if (dim != axis and uv.shape[dim] != source.shape[dim]) return tensor.TensorError.ShapeMismatch;
    }
    try validateUniqueIndices(rt, indices, source.shape[axis]);

    var out = try rt.materializeTyped(dtype, base);
    errdefer out.deinit();
    try writeRowsAxisRankTyped(rt, dtype, rank, &out, update, axis, indices);
    return out;
}

const TakeAlongTask = struct {
    input: []const f32,
    output: []f32,
    indices: []const usize,
    axis_dim: usize,
    out_axis_len: usize,
    inner: usize,
    outer_start: usize,
    outer_end: usize,
};

fn runTakeAlongTask(task: *const TakeAlongTask) void {
    // Indices are pre-validated; each task owns disjoint outer slices of
    // the output, so parallel writes never overlap and the result is
    // bitwise identical for any thread count.
    for (task.outer_start..task.outer_end) |outer_i| {
        const in_base = outer_i * task.axis_dim * task.inner;
        const out_base = outer_i * task.out_axis_len * task.inner;
        for (0..task.out_axis_len) |i| {
            for (0..task.inner) |inner_i| {
                const out_pos = out_base + i * task.inner + inner_i;
                task.output[out_pos] = task.input[in_base + task.indices[out_pos] * task.inner + inner_i];
            }
        }
    }
}

/// Elementwise gather along `axis` (torch.gather / np.take_along_axis):
/// `out[..., i, ...] = x[..., indices[..., i, ...], ...]`, where `indices`
/// is the flat row-major index buffer of the OUTPUT shape — the source
/// shape with `axis` replaced by `out_axis_len`. Parallel over outer
/// slices (disjoint writes; bitwise identical for any thread count).
/// Out-of-range entries error with `IndexOutOfBounds` (validated up
/// front, before dispatch).
pub fn takeAlongAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    comptime axis: usize,
    indices: []const usize,
    out_axis_len: usize,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    if (out_axis_len == 0) return tensor.TensorError.InvalidShape;
    const source = try x.rankView(rank);
    var out_shape = source.shape;
    out_shape[axis] = out_axis_len;
    const out_len = try tensor.elementCountArray(rank, out_shape);
    if (indices.len != out_len) return tensor.TensorError.InvalidShape;

    const axis_dim = source.shape[axis];
    for (indices) |index| {
        if (index >= axis_dim) return tensor.TensorError.IndexOutOfBounds;
    }

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const base_task: TakeAlongTask = .{
        .input = input,
        .output = output,
        .indices = indices,
        .axis_dim = axis_dim,
        .out_axis_len = out_axis_len,
        .inner = inner,
        .outer_start = 0,
        .outer_end = outer,
    };
    if (outer > 1 and out_len >= parallel.vector_elementwise_len_threshold) {
        if (rt.workPool()) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
            var tasks: [parallel.vector_max_threads]TakeAlongTask = undefined;
            for (0..task_count) |task_i| {
                tasks[task_i] = base_task;
                tasks[task_i].outer_start = task_i * outer / task_count;
                tasks[task_i].outer_end = (task_i + 1) * outer / task_count;
            }
            pool.parallelChunks(TakeAlongTask, tasks[0..task_count], runTakeAlongTask);
            return out;
        }
    }
    runTakeAlongTask(&base_task);
    return out;
}

/// Copy of `base` with `src` accumulated at per-element positions along
/// `axis` (torch.scatter_add): `out[..., indices[..., i, ...], ...] +=
/// src[..., i, ...]`. `src` matches `base` except along `axis`; `indices`
/// is the flat row-major index buffer of `src`'s shape. Duplicate indices
/// accumulate. Serial in row-major order (deterministic).
pub fn scatterAddAlongAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    base: *const Tensor,
    src: *const Tensor,
    comptime axis: usize,
    indices: []const usize,
) !Tensor {
    return scatterAlongAxisRankImpl(rt, rank, base, src, axis, indices, .add);
}

/// Copy of `base` with `src` written at per-element positions along
/// `axis` (torch.scatter with a tensor source): like
/// `scatterAddAlongAxisRank` but overwriting — duplicate indices resolve
/// deterministically to the LAST write in row-major `src` order (torch
/// leaves duplicate order unspecified; this pins it).
pub fn scatterAlongAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    base: *const Tensor,
    src: *const Tensor,
    comptime axis: usize,
    indices: []const usize,
) !Tensor {
    return scatterAlongAxisRankImpl(rt, rank, base, src, axis, indices, .write);
}

fn scatterAlongAxisRankImpl(
    rt: *Runtime,
    comptime rank: usize,
    base: *const Tensor,
    src: *const Tensor,
    comptime axis: usize,
    indices: []const usize,
    comptime mode: ScatterMode,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    const dest = try base.rankView(rank);
    const sv = try src.rankView(rank);
    inline for (0..rank) |dim| {
        if (dim != axis and sv.shape[dim] != dest.shape[dim]) return tensor.TensorError.ShapeMismatch;
    }
    const src_len = try tensor.elementCountArray(rank, sv.shape);
    if (indices.len != src_len) return tensor.TensorError.InvalidShape;

    const axis_dim = dest.shape[axis];
    for (indices) |index| {
        if (index >= axis_dim) return tensor.TensorError.IndexOutOfBounds;
    }

    var ss = try rt.prepareContiguous(src);
    defer ss.deinit();
    const source = ss.tensor().dataConst();

    var out = try rt.materialize(base);
    errdefer out.deinit();
    const output = out.data();

    const src_axis_len = sv.shape[axis];
    const inner = productAfterAxis(rank, dest.shape, axis);
    const outer = productBeforeAxis(rank, dest.shape, axis);
    const base_task: ScatterAlongTask(mode) = .{
        .source = source,
        .output = output,
        .indices = indices,
        .axis_dim = axis_dim,
        .src_axis_len = src_axis_len,
        .inner = inner,
        .outer_start = 0,
        .outer_end = outer,
    };
    if (outer > 1 and src_len >= parallel.vector_elementwise_len_threshold) {
        if (rt.workPool()) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), outer);
            var tasks: [parallel.vector_max_threads]ScatterAlongTask(mode) = undefined;
            for (0..task_count) |task_i| {
                tasks[task_i] = base_task;
                tasks[task_i].outer_start = task_i * outer / task_count;
                tasks[task_i].outer_end = (task_i + 1) * outer / task_count;
            }
            pool.parallelChunks(ScatterAlongTask(mode), tasks[0..task_count], runScatterAlongTask(mode));
            return out;
        }
    }
    runScatterAlongTask(mode)(&base_task);
    return out;
}

const ScatterMode = enum { add, write };

fn ScatterAlongTask(comptime mode: ScatterMode) type {
    _ = mode;
    return struct {
        source: []const f32,
        output: []f32,
        indices: []const usize,
        axis_dim: usize,
        src_axis_len: usize,
        inner: usize,
        outer_start: usize,
        outer_end: usize,
    };
}

fn runScatterAlongTask(comptime mode: ScatterMode) fn (*const ScatterAlongTask(mode)) void {
    return struct {
        // Indices are pre-validated. Each task owns disjoint outer slices
        // of the output — duplicates only collide WITHIN a slice, where
        // the serial row-major order is preserved, so accumulation order
        // and last-write-wins are bitwise identical for any thread count.
        fn run(task: *const ScatterAlongTask(mode)) void {
            for (task.outer_start..task.outer_end) |outer_i| {
                const out_base = outer_i * task.axis_dim * task.inner;
                const src_base = outer_i * task.src_axis_len * task.inner;
                for (0..task.src_axis_len) |i| {
                    for (0..task.inner) |inner_i| {
                        const src_pos = src_base + i * task.inner + inner_i;
                        const out_pos = out_base + task.indices[src_pos] * task.inner + inner_i;
                        switch (comptime mode) {
                            .add => task.output[out_pos] += task.source[src_pos],
                            .write => task.output[out_pos] = task.source[src_pos],
                        }
                    }
                }
            }
        }
    }.run;
}

pub fn zeroSliceAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, start: usize, length: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    if (length == 0) return tensor.TensorError.InvalidShape;
    const source = try x.rankView(rank);
    if (start > source.shape[axis] or length > source.shape[axis] - start) return tensor.TensorError.IndexOutOfBounds;

    var out = try rt.materialize(x);
    errdefer out.deinit();
    const output = out.data();
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const zero_len = length * inner;
    for (0..outer) |outer_i| {
        const base = outer_i * source.shape[axis] * inner + start * inner;
        @memset(output[base..][0..zero_len], 0);
    }
    return out;
}

pub fn zeroRowsAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, indices: []const usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    if (indices.len == 0) return tensor.TensorError.InvalidShape;
    const source = try x.rankView(rank);
    for (indices) |index| {
        if (index >= source.shape[axis]) return tensor.TensorError.IndexOutOfBounds;
    }

    var out = try rt.materialize(x);
    errdefer out.deinit();
    const output = out.data();
    if (comptime axis == 0) {
        const row_len = productAfterAxis(rank, source.shape, 0);
        for (indices) |index| {
            @memset(output[index * row_len ..][0..row_len], 0);
        }
        return out;
    }
    if (comptime axis == rank - 1) {
        const row_len = source.shape[axis];
        const row_count = output.len / row_len;
        for (0..row_count) |row| {
            const row_base = row * row_len;
            for (indices) |index| output[row_base + index] = 0;
        }
        return out;
    }

    var zero_mask = try rt.allocator.alloc(bool, source.shape[axis]);
    defer rt.allocator.free(zero_mask);
    @memset(zero_mask, false);
    for (indices) |index| zero_mask[index] = true;
    const source_strides = contiguousStridesArray(rank, source.shape);
    for (output, 0..) |*dst, linear| {
        const coord = coordinateForLinear(rank, source.shape, source_strides, linear, axis);
        if (zero_mask[coord]) dst.* = 0;
    }
    return out;
}

pub fn sliceGradientAxisRank(rt: *Runtime, comptime rank: usize, grad: *const Tensor, source_shape: [rank]usize, comptime axis: usize, start: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    _ = try tensor.elementCountArray(rank, source_shape);
    const gv = try grad.rankView(rank);
    if (start > source_shape[axis] or gv.shape[axis] > source_shape[axis] - start) return tensor.TensorError.IndexOutOfBounds;
    inline for (0..rank) |dim| {
        if (dim != axis and gv.shape[dim] != source_shape[dim]) return tensor.TensorError.ShapeMismatch;
    }

    var out = try rt.zerosRank(rank, source_shape);
    errdefer out.deinit();
    try writeSliceAxisRank(rt, rank, &out, grad, axis, start);
    return out;
}

pub fn scatterAddAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    grad: *const Tensor,
    source_shape: [rank]usize,
    comptime axis: usize,
    indices: []const usize,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    if (indices.len == 0) return tensor.TensorError.InvalidShape;
    const source_len = try tensor.elementCountArray(rank, source_shape);

    var expected_grad_shape = source_shape;
    expected_grad_shape[axis] = indices.len;
    const gv = try grad.rankView(rank);
    if (!std.mem.eql(usize, gv.shape[0..], expected_grad_shape[0..])) return tensor.TensorError.ShapeMismatch;
    for (indices) |index| {
        if (index >= source_shape[axis]) return tensor.TensorError.IndexOutOfBounds;
    }

    var gg = try rt.prepareContiguous(grad);
    defer gg.deinit();
    const input = gg.tensor().dataConst();

    if (comptime axis == 0) {
        const rows = source_shape[0];
        const row_len = productAfterAxis(rank, source_shape, 0);

        // Embedding-gradient hot path: the dominant cost is touching every
        // output element once (the dense zero-fill of the source-shaped
        // result) plus the streaming row accumulates — the same bytes/cycle
        // profile as the elementwise kernels, so the same work threshold
        // applies (softmax/CE divide it because of exp; there is no exp
        // here). Each task zeroes and accumulates its own destination row
        // range, scanning the full index list, so the result is bitwise
        // identical to the serial path below (see ScatterAddRowsTask).
        const total_work = source_len +| input.len;
        if (rows > 1 and total_work >= parallel.vector_elementwise_len_threshold) {
            if (rt.workPool()) |pool| {
                var out = try rt.emptyRank(rank, source_shape);
                errdefer out.deinit();
                const base_task: ScatterAddRowsTask = .{
                    .grad = input,
                    .output = out.data(),
                    .indices = indices,
                    .row_len = row_len,
                    .row_start = 0,
                    .row_end = rows,
                };
                const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), rows);
                var tasks: [parallel.vector_max_threads]ScatterAddRowsTask = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = base_task;
                    tasks[task_i].row_start = task_i * rows / task_count;
                    tasks[task_i].row_end = (task_i + 1) * rows / task_count;
                }
                pool.parallelChunks(ScatterAddRowsTask, tasks[0..task_count], runScatterAddRowsTask);
                return out;
            }
        }

        var out = try rt.zerosRank(rank, source_shape);
        errdefer out.deinit();
        const output = out.data();
        for (indices, 0..) |index, row| {
            const src = input[row * row_len ..][0..row_len];
            const dst = output[index * row_len ..][0..row_len];
            for (dst, src) |*d, v| d.* += v;
        }
        return out;
    }

    var out = try rt.zerosRank(rank, source_shape);
    errdefer out.deinit();
    const output = out.data();

    const source_strides = contiguousStridesArray(rank, source_shape);
    const grad_strides = contiguousStridesArray(rank, expected_grad_shape);
    for (input, 0..) |value, linear| {
        var remainder = linear;
        var out_linear: usize = 0;
        inline for (0..rank) |dim| {
            const coord = remainder / grad_strides[dim];
            remainder %= grad_strides[dim];
            const out_coord = if (dim == axis) indices[coord] else coord;
            out_linear += out_coord * source_strides[dim];
        }
        output[out_linear] += value;
    }

    return out;
}

fn writeSliceAxisRank(rt: *Runtime, comptime rank: usize, target: *Tensor, update: *const Tensor, comptime axis: usize, start: usize) !void {
    return writeSliceAxisRankTyped(rt, .f32, rank, target, update, axis, start);
}

fn writeSliceAxisRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    target: *tensor.TensorOf(dtype),
    update: *const tensor.TensorOf(dtype),
    comptime axis: usize,
    start: usize,
) !void {
    const tv = try target.rankView(rank);
    const uv = try update.rankView(rank);
    if (!target.isContiguous()) return tensor.TensorError.UnsupportedView;
    if (start > tv.shape[axis] or uv.shape[axis] > tv.shape[axis] - start) return tensor.TensorError.IndexOutOfBounds;
    inline for (0..rank) |dim| {
        if (dim != axis and uv.shape[dim] != tv.shape[dim]) return tensor.TensorError.ShapeMismatch;
    }

    var uu = try rt.prepareContiguousTyped(dtype, update);
    defer uu.deinit();
    const input = uu.tensor().dataConst();
    const output = target.data();
    const inner = productAfterAxis(rank, tv.shape, axis);
    const outer = productBeforeAxis(rank, tv.shape, axis);
    const copy_len = uv.shape[axis] * inner;
    if (comptime axis == rank - 1) {
        const row_len = tv.shape[axis];
        const update_len = uv.shape[axis];
        for (0..outer) |outer_i| {
            const dst_base = outer_i * row_len + start;
            const src_base = outer_i * update_len;
            @memcpy(output[dst_base..][0..update_len], input[src_base..][0..update_len]);
        }
        return;
    }

    for (0..outer) |outer_i| {
        const dst_base = outer_i * tv.shape[axis] * inner + start * inner;
        const src_base = outer_i * uv.shape[axis] * inner;
        @memcpy(output[dst_base..][0..copy_len], input[src_base..][0..copy_len]);
    }
}

fn writeRowsAxisRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    target: *tensor.TensorOf(dtype),
    update: *const tensor.TensorOf(dtype),
    comptime axis: usize,
    indices: []const usize,
) !void {
    const tv = try target.rankView(rank);
    const uv = try update.rankView(rank);
    if (!target.isContiguous()) return tensor.TensorError.UnsupportedView;
    if (indices.len == 0 or uv.shape[axis] != indices.len) return tensor.TensorError.InvalidShape;
    inline for (0..rank) |dim| {
        if (dim != axis and uv.shape[dim] != tv.shape[dim]) return tensor.TensorError.ShapeMismatch;
    }
    for (indices) |index| {
        if (index >= tv.shape[axis]) return tensor.TensorError.IndexOutOfBounds;
    }

    var uu = try rt.prepareContiguousTyped(dtype, update);
    defer uu.deinit();
    const input = uu.tensor().dataConst();
    const output = target.data();
    if (comptime axis == 0) {
        const row_len = productAfterAxis(rank, tv.shape, 0);
        for (indices, 0..) |index, row| {
            @memcpy(output[index * row_len ..][0..row_len], input[row * row_len ..][0..row_len]);
        }
        return;
    }
    if (comptime axis == rank - 1) {
        const row_len = tv.shape[axis];
        const update_len = uv.shape[axis];
        const row_count = output.len / row_len;
        for (0..row_count) |row| {
            const dst_base = row * row_len;
            const src_base = row * update_len;
            for (indices, 0..) |index, update_i| {
                output[dst_base + index] = input[src_base + update_i];
            }
        }
        return;
    }

    const target_strides = contiguousStridesArray(rank, tv.shape);
    const update_strides = contiguousStridesArray(rank, uv.shape);
    for (input, 0..) |value, linear| {
        var remainder = linear;
        var out_linear: usize = 0;
        inline for (0..rank) |dim| {
            const coord = remainder / update_strides[dim];
            remainder %= update_strides[dim];
            const out_coord = if (dim == axis) indices[coord] else coord;
            out_linear += out_coord * target_strides[dim];
        }
        output[out_linear] = value;
    }
}

fn validateUniqueIndices(rt: *Runtime, indices: []const usize, limit: usize) !void {
    var seen = try rt.allocator.alloc(bool, limit);
    defer rt.allocator.free(seen);
    @memset(seen, false);
    for (indices) |index| {
        if (index >= limit) return tensor.TensorError.IndexOutOfBounds;
        if (seen[index]) return tensor.TensorError.InvalidShape;
        seen[index] = true;
    }
}
