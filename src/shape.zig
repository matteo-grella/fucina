//! Shape and stride arithmetic over inline dimension arrays: the `Shape`
//! value, element counting (logical, and block-storage over the last
//! axis), contiguous strides, comptime-rank dispatch, and the axis
//! geometry every axis-wise kernel decomposes into. The one home for
//! this arithmetic: the raw tensor, the tag-ops library and the exec
//! runtime all count and stride through here. No tensor type, no dtype
//! policy beyond the block length a storage count needs.
//! Layer stack: docs/ARCHITECTURE.md.
const std = @import("std");
const dtype_mod = @import("dtype.zig");

pub const max_rank = 8;

/// Shared comptime-guard texts: the messages name the limit so the compile
/// error carries it to the call site.
pub const invalid_rank_msg = std.fmt.comptimePrint("invalid tensor rank (1..{d})", .{max_rank});
pub const too_many_tags_msg = std.fmt.comptimePrint("too many tensor tags (max {d})", .{max_rank});

/// The shape error domain: `InvalidShape` for a shape that is not
/// representable (zero rank, rank past `max_rank`, a zero dimension, an
/// element count that overflows), `ShapeMismatch` for two shapes that
/// were required to agree. `tensor.TensorError` extends it.
/// The shape domain, one rule per name. `InvalidShape`: a shape, rank,
/// axis or index-structure argument is malformed on its own (rank 0 or
/// beyond `max_rank`, a zero dimension, an element count that overflows
/// `usize`, an axis beyond the rank, an empty index list, offsets that
/// disagree with their section count, a dimension not aligned to its
/// block size). `ShapeMismatch`: two tensors, or a tensor and a table
/// carrying a shape, disagree.
pub const ShapeError = error{
    InvalidShape,
    ShapeMismatch,
};

/// Inline dimension array with a runtime rank: the metadata every raw
/// tensor carries twice (shape and strides). `init` validates a shape
/// (every dimension >= 1); `initStrides` stores strides, which may be 0
/// (broadcast axes).
pub const Shape = struct {
    len: u8,
    dims: [max_rank]usize = undefined,

    pub fn init(values: []const usize) !Shape {
        if (values.len == 0 or values.len > max_rank) return ShapeError.InvalidShape;

        var out = Shape{ .len = @intCast(values.len) };
        for (values, 0..) |value, i| {
            if (value == 0) return ShapeError.InvalidShape;
            out.dims[i] = value;
        }
        return out;
    }

    pub fn initStrides(values: []const usize) !Shape {
        if (values.len == 0 or values.len > max_rank) return ShapeError.InvalidShape;

        var out = Shape{ .len = @intCast(values.len) };
        for (values, 0..) |value, i| {
            out.dims[i] = value;
        }
        return out;
    }

    /// A validated `Shape` from any shape spelling an op takes: a
    /// `[n]usize` array (by value or by pointer), a tuple of sizes, or a
    /// `[]const usize` slice. One normalization, so every allocation
    /// primitive has a single arm.
    pub fn from(values: anytype) !Shape {
        const V = @TypeOf(values);
        const invalid = "shape must be a [n]usize array, a tuple of sizes, or a []const usize slice";
        switch (@typeInfo(V)) {
            .array => return init(&values),
            .pointer => |info| switch (info.size) {
                .one => return from(values.*),
                .slice => return init(values),
                else => @compileError(invalid),
            },
            .@"struct" => |info| {
                if (!info.is_tuple) @compileError(invalid);
                var dims: [info.fields.len]usize = undefined;
                inline for (values, 0..) |dim, i| dims[i] = dim;
                return init(&dims);
            },
            else => @compileError(invalid),
        }
    }

    pub fn slice(self: *const Shape) []const usize {
        return self.dims[0..self.len];
    }

    pub fn at(self: *const Shape, i: usize) usize {
        return self.dims[i];
    }
};

/// Element count of a validated shape (rank 1..`max_rank`, every
/// dimension >= 1, no overflow); `InvalidShape` otherwise. Arrays coerce
/// through `&shape`.
pub fn elementCount(shape: []const usize) !usize {
    if (shape.len == 0 or shape.len > max_rank) return ShapeError.InvalidShape;
    var n: usize = 1;
    for (shape) |dim| {
        if (dim == 0) return ShapeError.InvalidShape;
        n = std.math.mul(usize, n, dim) catch return ShapeError.InvalidShape;
    }
    return n;
}

/// Storage element count: the block count for a block-quantized dtype
/// (the last axis must be a whole number of blocks), the element count
/// for a scalar dtype.
pub fn storageElementCount(comptime dtype: dtype_mod.DType, shape: []const usize) !usize {
    if (comptime dtype_mod.isScalar(dtype)) return elementCount(shape);
    if (shape.len == 0 or shape.len > max_rank) return ShapeError.InvalidShape;

    const block = comptime dtype_mod.blockSize(dtype);
    const last_dim = shape[shape.len - 1];
    if (last_dim == 0 or last_dim % block != 0) return ShapeError.InvalidShape;
    var n: usize = last_dim / block;
    for (shape[0 .. shape.len - 1]) |dim| {
        if (dim == 0) return ShapeError.InvalidShape;
        n = std.math.mul(usize, n, dim) catch return ShapeError.InvalidShape;
    }
    return n;
}

/// Product of the dimensions of a shape already validated by `elementCount`
/// (a tensor's own metadata).
pub fn elementCountAssumeValid(shape: []const usize) usize {
    return product(shape);
}

pub fn storageElementCountAssumeValid(comptime dtype: dtype_mod.DType, shape: []const usize) usize {
    if (comptime dtype_mod.isScalar(dtype)) return product(shape);
    return product(shape[0 .. shape.len - 1]) * (shape[shape.len - 1] / dtype_mod.blockSize(dtype));
}

/// Plain product of a dimension range (no validation, no overflow check).
pub fn product(dims: []const usize) usize {
    var n: usize = 1;
    for (dims) |dim| n *= dim;
    return n;
}

/// A `[]const usize` shape as a fixed-rank array, validated.
pub fn arrayFromSlice(comptime rank: usize, shape: []const usize) ![rank]usize {
    if (shape.len != rank) return ShapeError.InvalidShape;
    var out: [rank]usize = undefined;
    inline for (0..rank) |i| out[i] = shape[i];
    _ = try elementCount(&out);
    return out;
}

/// Row-major strides of `shape` into `out` (same length).
pub fn contiguousStridesInto(out: []usize, shape: []const usize) void {
    var stride: usize = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        out[i] = stride;
        stride *= shape[i];
    }
}

pub fn contiguousStrides(comptime rank: usize, shape: [rank]usize) [rank]usize {
    var strides: [rank]usize = undefined;
    contiguousStridesInto(&strides, &shape);
    return strides;
}

/// True when `strides` are exactly the row-major strides of `shape`.
pub fn isContiguous(shape: []const usize, strides: []const usize) bool {
    var expected: usize = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        if (strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

/// Call `F(rank, args...)` with the runtime rank made comptime, for the
/// kernels specialized per rank; `InvalidShape` past `max_rank`. `F`'s
/// return type is an error union whose payload does not depend on the
/// rank.
pub fn dispatchRank(comptime F: anytype, rank: usize, args: anytype) !DispatchPayload(F) {
    return switch (rank) {
        inline 1...max_rank => |r| @call(.auto, F, .{r} ++ args),
        else => ShapeError.InvalidShape,
    };
}

fn DispatchPayload(comptime F: anytype) type {
    const Ret = @typeInfo(@TypeOf(F)).@"fn".return_type orelse
        @compileError("dispatchRank needs a rank-independent return type on " ++ @typeName(@TypeOf(F)));
    return @typeInfo(Ret).error_union.payload;
}

/// The `(outer, axis_dim, inner)` decomposition of a rank-`rank` shape
/// around `axis`: every axis-wise kernel walks `outer` groups of
/// `axis_dim` runs of `inner` elements.
pub const AxisGeometry = struct {
    outer: usize,
    axis_dim: usize,
    inner: usize,

    pub fn of(comptime rank: usize, shape: [rank]usize, comptime axis: usize) AxisGeometry {
        comptime if (axis >= rank) @compileError("axis out of bounds");
        return .{
            .outer = product(shape[0..axis]),
            .axis_dim = shape[axis],
            .inner = product(shape[axis + 1 ..]),
        };
    }
};

pub fn productBeforeAxis(comptime rank: usize, shape: [rank]usize, comptime axis: usize) usize {
    return product(shape[0..axis]);
}

pub fn productAfterAxis(comptime rank: usize, shape: [rank]usize, comptime axis: usize) usize {
    return product(shape[axis + 1 ..]);
}

/// `shape` with `axis` removed; a rank-1 input reports the scalar shape
/// `{1}` (the raw layer has no rank 0).
pub fn withoutAxis(
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

/// `ShapeMismatch` unless both tensors (anything with `shape.slice()`)
/// carry the same shape.
pub fn requireSameShape(a: anytype, b: anytype) !void {
    if (!std.mem.eql(usize, a.shape.slice(), b.shape.slice())) return ShapeError.ShapeMismatch;
}

/// `requireSameShape` at a comptime rank, returning the shared shape.
pub fn requireSameRankShape(comptime rank: usize, a: anytype, b: anytype) ![rank]usize {
    if (a.shape.len != rank or b.shape.len != rank) return ShapeError.ShapeMismatch;
    const av = try a.rankView(rank);
    const bv = try b.rankView(rank);
    if (!std.mem.eql(usize, av.shape[0..], bv.shape[0..])) return ShapeError.ShapeMismatch;
    return av.shape;
}

/// `target_shape` must be a trailing broadcast of `source_shape`: same
/// rank or lower, every target dim equal to its source dim or 1.
pub fn validateBroadcastRank(
    comptime target_rank: usize,
    comptime source_rank: usize,
    target_shape: [target_rank]usize,
    source_shape: [source_rank]usize,
) !void {
    if (target_rank > source_rank) return ShapeError.ShapeMismatch;
    _ = try elementCount(&target_shape);

    const rank_diff = source_rank - target_rank;
    inline for (0..target_rank) |i| {
        const target_dim = target_shape[i];
        const source_dim = source_shape[rank_diff + i];
        if (target_dim != source_dim and target_dim != 1) {
            return ShapeError.ShapeMismatch;
        }
    }
}

/// True when `target_shape` equals the trailing `target_rank` dims of
/// `source_shape` exactly.
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

test "Shape.from accepts arrays, pointers to arrays, tuples and slices" {
    const from_array = try Shape.from([_]usize{ 2, 3 });
    const from_ptr = try Shape.from(&[_]usize{ 2, 3 });
    const from_tuple = try Shape.from(.{ 2, 3 });
    const as_slice: []const usize = &.{ 2, 3 };
    const from_slice = try Shape.from(as_slice);
    for ([_]Shape{ from_array, from_ptr, from_tuple, from_slice }) |s| {
        try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, s.slice());
    }
    try std.testing.expectError(ShapeError.InvalidShape, Shape.from(.{ 2, 0 }));
    try std.testing.expectError(ShapeError.InvalidShape, Shape.from(&[_]usize{}));
}

test "element counts validate and overflow-check" {
    try std.testing.expectEqual(@as(usize, 24), try elementCount(&.{ 2, 3, 4 }));
    try std.testing.expectError(ShapeError.InvalidShape, elementCount(&.{ 2, 0 }));
    try std.testing.expectError(ShapeError.InvalidShape, elementCount(&.{}));
    try std.testing.expectError(ShapeError.InvalidShape, elementCount(&.{ std.math.maxInt(usize), 2 }));
    try std.testing.expectEqual(@as(usize, 2), try storageElementCount(.q8_0, &.{ 2, 32 }));
    try std.testing.expectError(ShapeError.InvalidShape, storageElementCount(.q8_0, &.{ 2, 33 }));
}

test "contiguous strides and the axis geometry" {
    const strides = contiguousStrides(3, .{ 2, 3, 4 });
    try std.testing.expectEqualSlices(usize, &.{ 12, 4, 1 }, &strides);
    try std.testing.expect(isContiguous(&.{ 2, 3, 4 }, &strides));
    try std.testing.expect(!isContiguous(&.{ 2, 3, 4 }, &.{ 1, 2, 6 }));
    const g = AxisGeometry.of(3, .{ 2, 3, 4 }, 1);
    try std.testing.expectEqual(@as(usize, 2), g.outer);
    try std.testing.expectEqual(@as(usize, 3), g.axis_dim);
    try std.testing.expectEqual(@as(usize, 4), g.inner);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, &withoutAxis(3, 2, .{ 2, 3, 4 }, 1));
    try std.testing.expectEqualSlices(usize, &.{1}, &withoutAxis(1, 1, .{7}, 0));
}

test "dispatchRank makes the rank comptime and rejects rank 0 and past max_rank" {
    const Probe = struct {
        fn f(comptime rank: usize, scale: usize) !usize {
            return rank * scale;
        }
    };
    try std.testing.expectEqual(@as(usize, 6), try dispatchRank(Probe.f, 3, .{@as(usize, 2)}));
    try std.testing.expectError(ShapeError.InvalidShape, dispatchRank(Probe.f, 0, .{@as(usize, 2)}));
    try std.testing.expectError(ShapeError.InvalidShape, dispatchRank(Probe.f, max_rank + 1, .{@as(usize, 2)}));
}
