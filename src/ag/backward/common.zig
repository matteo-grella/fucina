//! Shared VJP helpers: gradient/tag reduction and expansion,
//! shape/stride scratch arrays, mask gating, rope-table cloning.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const tag_ops = @import("../../tag_ops.zig");
const tags_mod = @import("../../tags.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const Tag = tags_mod.Tag;
const rawRank = tags_mod.rawRank;
const tagIndex = tags_mod.tagIndex;
const removeTags = tags_mod.removeTags;
const tagsEqual = tags_mod.tagsEqual;

pub const PointwiseOp = tag_ops.PointwiseOp;

pub fn rawShapeArray(comptime tags: anytype, value: *const RawTensor) [rawRank(tags.len)]usize {
    return rawShapeArrayOf(.f32, tags, value);
}

pub fn rawShapeArrayOf(comptime tensor_dtype: tensor_mod.DType, comptime tags: anytype, value: *const tensor_mod.TensorOf(tensor_dtype)) [rawRank(tags.len)]usize {
    const rank = comptime rawRank(tags.len);
    var out: [rank]usize = undefined;
    inline for (0..rank) |i| {
        out[i] = value.shape.at(i);
    }
    return out;
}

pub fn rawStrideArray(comptime tags: anytype, value: *const RawTensor) [rawRank(tags.len)]usize {
    const rank = comptime rawRank(tags.len);
    var out: [rank]usize = undefined;
    inline for (0..rank) |i| {
        out[i] = value.strides.at(i);
    }
    return out;
}

pub fn taggedShapeArray(comptime tags: anytype, raw_shape: [rawRank(tags.len)]usize) [tags.len]usize {
    var out: [tags.len]usize = undefined;
    inline for (0..tags.len) |i| out[i] = raw_shape[i];
    return out;
}

pub fn tagsDifference(comptime tags: anytype, comptime keep_tags: anytype) [tagsDifferenceLen(tags, keep_tags)]Tag {
    var out: [tagsDifferenceLen(tags, keep_tags)]Tag = undefined;
    var out_i: usize = 0;
    inline for (tags) |tag| {
        if (comptime tagIndex(keep_tags, tag) == null) {
            out[out_i] = tag;
            out_i += 1;
        }
    }
    return out;
}

pub fn tagsDifferenceLen(comptime tags: anytype, comptime keep_tags: anytype) usize {
    var len: usize = 0;
    inline for (tags) |tag| {
        if (comptime tagIndex(keep_tags, tag) == null) len += 1;
    }
    return len;
}

/// Sum-reduce a result-tagged gradient back to an operand's tags/shape (the
/// pointwise broadcast-backward rule). Shared with `elemental.zig`.
pub fn reduceGradientToTags(
    comptime grad_tags: anytype,
    comptime target_tags: anytype,
    ctx: *ExecContext,
    grad: *const RawTensor,
    target_shape: [rawRank(target_tags.len)]usize,
) !RawTensor {
    if (comptime tagsEqual(grad_tags, target_tags)) {
        if (std.mem.eql(usize, grad.shape.slice(), target_shape[0..])) {
            return grad.cloneView();
        }
    }

    const reduce_tags = tagsDifference(grad_tags, target_tags);
    var reduced = try tag_ops.sumManyTensor(grad_tags, grad, ctx, reduce_tags);
    defer reduced.deinit();
    const remaining_tags = removeTags(grad_tags, reduce_tags);
    var aligned = try tag_ops.permuteTensorTo(remaining_tags, &reduced, target_tags);
    defer aligned.deinit();

    return ctx.reduceBroadcast(&aligned, target_shape[0..]);
}

/// Borrow-if-contiguous read access: a retained view when the layout is
/// already contiguous, else a materialized copy. Shared with `elemental.zig`.
pub fn contiguousForRead(ctx: *ExecContext, value: *const RawTensor) !RawTensor {
    if (value.isContiguous()) return value.cloneView();
    return ctx.materialize(value);
}

pub fn expandGradientToTags(
    comptime grad_tags: anytype,
    comptime target_tags: anytype,
    ctx: *ExecContext,
    grad: *const RawTensor,
    target_shape: [rawRank(target_tags.len)]usize,
) !RawTensor {
    const tagged_shape = taggedShapeArray(target_tags, target_shape);
    _ = ctx;
    return tag_ops.broadcastTensorTo(grad_tags, grad, target_tags, tagged_shape);
}

test "reduceGradientToTags uses direct view when tags and shape already match" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var grad = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer grad.deinit();

    const outstanding_before = ctx.buffers.outstandingBuffers();
    var reduced = try reduceGradientToTags(.{ .batch, .hidden }, .{ .batch, .hidden }, &ctx, &grad, .{ 2, 3 });
    defer reduced.deinit();

    try std.testing.expectEqualSlices(f32, grad.dataConst(), reduced.dataConst());
    try std.testing.expectEqual(grad.dataConst().ptr, reduced.dataConst().ptr);
    try std.testing.expectEqual(outstanding_before, ctx.buffers.outstandingBuffers());
}

/// VJP for `maskedFill(x, mask, value)`: grad passes through where the mask is
/// clear, zero where it is set (`value` is a constant — no grad).
/// Contiguous read of a saved typed mask/cond tensor (any scalar dtype).
pub fn contiguousForReadTyped(comptime mask_dtype: tensor_mod.DType, ctx: *ExecContext, value: *const tensor_mod.TensorOf(mask_dtype)) !tensor_mod.TensorOf(mask_dtype) {
    if (value.isContiguous()) return value.cloneView();
    return ctx.materializeTyped(mask_dtype, value);
}

/// Shared tail of the masked-reduction VJPs: broadcast a result-shaped
/// gradient back over the reduced axis and zero it wherever the mask excluded
/// the element. The broadcast alone is a stride-0 view (what the unmasked sum
/// returns), but gating makes the result genuinely per-element, so this
/// materializes — one allocation at source shape, the `MaskedFillBackward`
/// shape.
pub fn gateGradientByMask(
    comptime mask_dtype: tensor_mod.DType,
    comptime result_tags: anytype,
    comptime source_tags: anytype,
    ctx: *ExecContext,
    gy: *const RawTensor,
    mask: *const tensor_mod.TensorOf(mask_dtype),
    source_shape: [rawRank(source_tags.len)]usize,
) !RawTensor {
    var m = try contiguousForReadTyped(mask_dtype, ctx, mask);
    defer m.deinit();

    var expanded_view = try expandGradientToTags(result_tags, source_tags, ctx, gy, source_shape);
    defer expanded_view.deinit();
    var expanded = try contiguousForRead(ctx, &expanded_view);
    defer expanded.deinit();

    var gx = try ctx.empty(m.shape.slice());
    errdefer gx.deinit();
    for (m.dataConst(), expanded.dataConst(), gx.data()) |mv, grad, *dst| {
        dst.* = if (dtype_mod.isTruthy(mask_dtype, mv)) grad else 0;
    }
    return gx;
}

/// Row geometry for axis-wise scans: element (outer, i, inner) of a
/// contiguous `shape` tensor lives at `outer·(shape[axis]·inner_len) +
/// i·inner_len + inner`.
pub fn axisGeometry(comptime rank: usize, shape: [rank]usize, comptime axis: usize) struct { outer: usize, inner: usize } {
    var outer: usize = 1;
    for (0..axis) |i| outer *= shape[i];
    var inner: usize = 1;
    for (axis + 1..rank) |i| inner *= shape[i];
    return .{ .outer = outer, .inner = inner };
}

/// Owned copy of a RoPE table with the sin half negated: applying the forward
/// rotation kernel with this table is the exact inverse (transpose) rotation,
/// i.e. the RoPE VJP. Cloning the table (instead of rebuilding from positions
/// and theta) preserves `freq_factors` scaling baked into the angles.
pub fn cloneInverseRopeTable(allocator: std.mem.Allocator, table: *const exec_mod.RopeTable) !exec_mod.RopeTable {
    const positions = try allocator.dupe(i32, table.positions);
    errdefer allocator.free(positions);
    const values = try allocator.dupe(f32, table.values);
    const angle_count = table.positions.len * table.pair_count;
    for (values[0..angle_count]) |*value| value.* = -value.*;
    return .{
        .allocator = allocator,
        .positions = positions,
        .theta_base = table.theta_base,
        .feature_dim = table.feature_dim,
        .pair_count = table.pair_count,
        .values = values,
    };
}
