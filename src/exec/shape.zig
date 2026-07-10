const std = @import("std");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

const DType = tensor.DType;
const Tensor = tensor.Tensor;

pub fn coordinateForLinear(comptime rank: usize, shape: [rank]usize, strides: [rank]usize, linear: usize, axis: usize) usize {
    return (linear / strides[axis]) % shape[axis];
}

pub fn physicalOffsetExcludingAxis(
    comptime rank: usize,
    shape: [rank]usize,
    source_strides: [rank]usize,
    target_strides: [rank]usize,
    target_offset: usize,
    linear: usize,
    comptime axis: usize,
) usize {
    var physical = target_offset;
    inline for (0..rank) |dim| {
        if (dim != axis) {
            physical += coordinateForLinear(rank, shape, source_strides, linear, dim) * target_strides[dim];
        }
    }
    return physical;
}

pub fn preSoftmaxValue(
    comptime rank: usize,
    value: f32,
    scale_value: f32,
    mask: ?tensor.RankedTensor(rank),
    mask_base: usize,
    mask_axis_stride: usize,
    axis_i: usize,
    slope: f32,
) f32 {
    var out = value * scale_value;
    if (mask) |mask_view| {
        out += slope * mask_view.tensor.buffer.data[mask_base + axis_i * mask_axis_stride];
    }
    return out;
}

pub fn dispatchRank(comptime F: anytype, rank: usize, args: anytype) !Tensor {
    return switch (rank) {
        1 => @call(.auto, F, .{1} ++ args),
        2 => @call(.auto, F, .{2} ++ args),
        3 => @call(.auto, F, .{3} ++ args),
        4 => @call(.auto, F, .{4} ++ args),
        5 => @call(.auto, F, .{5} ++ args),
        6 => @call(.auto, F, .{6} ++ args),
        7 => @call(.auto, F, .{7} ++ args),
        8 => @call(.auto, F, .{8} ++ args),
        else => tensor.TensorError.InvalidShape,
    };
}

pub fn requireSameRankShape(comptime rank: usize, a: *const Tensor, b: *const Tensor) ![rank]usize {
    if (a.shape.len != rank or b.shape.len != rank) return tensor.TensorError.ShapeMismatch;
    const av = try a.rankView(rank);
    const bv = try b.rankView(rank);
    if (!std.mem.eql(usize, av.shape[0..], bv.shape[0..])) return tensor.TensorError.ShapeMismatch;
    return av.shape;
}

pub fn requireSameRankShapeOf(
    comptime dtype: DType,
    comptime rank: usize,
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) ![rank]usize {
    if (a.shape.len != rank or b.shape.len != rank) return tensor.TensorError.ShapeMismatch;
    const av = try a.rankView(rank);
    const bv = try b.rankView(rank);
    if (!std.mem.eql(usize, av.shape[0..], bv.shape[0..])) return tensor.TensorError.ShapeMismatch;
    return av.shape;
}

pub fn shapeArrayFromSlice(comptime rank: usize, shape: []const usize) ![rank]usize {
    if (shape.len != rank) return tensor.TensorError.InvalidShape;

    var out: [rank]usize = undefined;
    inline for (0..rank) |i| {
        out[i] = shape[i];
    }
    _ = try tensor.elementCountArray(rank, out);
    return out;
}

pub fn validateBroadcastRank(
    comptime target_rank: usize,
    comptime source_rank: usize,
    target_shape: [target_rank]usize,
    source_shape: [source_rank]usize,
) !void {
    if (target_rank > source_rank) return tensor.TensorError.ShapeMismatch;
    _ = try tensor.elementCountArray(target_rank, target_shape);

    const rank_diff = source_rank - target_rank;
    inline for (0..target_rank) |i| {
        const target_dim = target_shape[i];
        const source_dim = source_shape[rank_diff + i];
        if (target_dim != source_dim and target_dim != 1) {
            return tensor.TensorError.ShapeMismatch;
        }
    }
}

pub fn isExactSuffixRank(
    comptime target_rank: usize,
    comptime source_rank: usize,
    target_shape: [target_rank]usize,
    source_shape: [source_rank]usize,
) bool {
    if (target_rank > source_rank) return false;
    const rank_diff = source_rank - target_rank;
    inline for (0..target_rank) |i| {
        if (target_shape[i] != source_shape[rank_diff + i]) return false;
    }
    return true;
}

pub fn contiguousStridesArray(comptime rank: usize, shape: [rank]usize) [rank]usize {
    var strides: [rank]usize = undefined;
    var stride: usize = 1;
    comptime var i = rank;
    inline while (i > 0) {
        i -= 1;
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

pub fn shapeWithoutAxis(
    comptime rank: usize,
    comptime out_rank: usize,
    shape: [rank]usize,
    comptime axis: usize,
) [out_rank]usize {
    var out: [out_rank]usize = undefined;
    if (rank == 1) {
        out[0] = 1;
        return out;
    }
    inline for (0..rank) |i| {
        if (i != axis) {
            const out_i = if (i < axis) i else i - 1;
            out[out_i] = shape[i];
        }
    }
    return out;
}

pub fn productBeforeAxis(comptime rank: usize, shape: [rank]usize, comptime axis: usize) usize {
    var n: usize = 1;
    inline for (0..axis) |i| n *= shape[i];
    return n;
}

pub fn productAfterAxis(comptime rank: usize, shape: [rank]usize, comptime axis: usize) usize {
    var n: usize = 1;
    inline for ((axis + 1)..rank) |i| n *= shape[i];
    return n;
}

pub fn validateCausalDepthwiseState(state: ?[]const f32, channels: usize, taps: usize) !void {
    if (taps == 0) return tensor.TensorError.InvalidShape;
    const expected = try std.math.mul(usize, taps - 1, channels);
    if (state) |values| {
        if (values.len != expected) return tensor.TensorError.InvalidDataLength;
    }
}

pub fn validateCausalConvState(state: ?[]const f32, in_channels: usize, taps: usize, dilation: usize) !void {
    if (taps == 0 or dilation == 0) return tensor.TensorError.InvalidShape;
    const pad = try std.math.mul(usize, dilation, taps - 1);
    const expected = try std.math.mul(usize, pad, in_channels);
    if (state) |values| {
        if (values.len != expected) return tensor.TensorError.InvalidDataLength;
    }
}

pub fn validateGroupedCausalConv(
    state: ?[]const f32,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
) !usize {
    if (groups == 0) return tensor.TensorError.InvalidShape;
    if (in_channels % groups != 0 or out_channels % groups != 0) return tensor.TensorError.ShapeMismatch;
    try validateCausalConvState(state, in_channels, taps, dilation);
    return in_channels / groups;
}

pub fn causalConvWork(seq: usize, in_channels: usize, out_channels: usize, taps: usize) usize {
    return std.math.mul(usize, parallel.saturatedMul3(seq, in_channels, out_channels), taps) catch std.math.maxInt(usize);
}

pub fn floorPowerOfTwo(value: usize) usize {
    std.debug.assert(value > 0);
    var out: usize = 1;
    while (out <= value / 2) out *= 2;
    return out;
}

pub fn alibiSlope(head_i: usize, head_log2: usize, max_bias: f32) f32 {
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
