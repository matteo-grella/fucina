const std = @import("std");
const dtype_mod = @import("dtype.zig");
const storage = @import("storage.zig");

const Allocator = std.mem.Allocator;
pub const DType = dtype_mod.DType;
pub const Scalar = dtype_mod.Scalar;
pub const Storage = dtype_mod.Storage;
pub const max_rank = 8;

pub const TensorError = error{
    ShapeMismatch,
    InvalidShape,
    InvalidDataLength,
    IndexOutOfBounds,
    UnsupportedView,
    EmptySelection,
};

pub const Shape = struct {
    len: u8,
    dims: [max_rank]usize = undefined,

    pub fn init(values: []const usize) !Shape {
        if (values.len == 0 or values.len > max_rank) return TensorError.InvalidShape;

        var out = Shape{ .len = @intCast(values.len) };
        for (values, 0..) |value, i| {
            if (value == 0) return TensorError.InvalidShape;
            out.dims[i] = value;
        }
        return out;
    }

    pub fn initStrides(values: []const usize) !Shape {
        if (values.len == 0 or values.len > max_rank) return TensorError.InvalidShape;

        var out = Shape{ .len = @intCast(values.len) };
        for (values, 0..) |value, i| {
            out.dims[i] = value;
        }
        return out;
    }

    pub fn slice(self: *const Shape) []const usize {
        return self.dims[0..self.len];
    }

    pub fn at(self: *const Shape, i: usize) usize {
        return self.dims[i];
    }
};

pub fn RankedTensorOf(comptime tensor_dtype: DType, comptime rank: usize) type {
    if (rank == 0 or rank > max_rank) @compileError("invalid tensor rank");

    return struct {
        tensor: *const TensorOf(tensor_dtype),
        shape: [rank]usize,
        strides: [rank]usize,

        pub fn dim(self: @This(), comptime axis: usize) usize {
            if (axis >= rank) @compileError("axis out of bounds");
            return self.shape[axis];
        }

        pub fn len(self: @This()) usize {
            return elementCountArrayAssumeValid(rank, self.shape);
        }

        pub fn isContiguous(self: @This()) bool {
            var expected: usize = 1;
            comptime var i = rank;
            inline while (i > 0) : (i -= 1) {
                if (self.strides[i - 1] != expected) return false;
                expected *= self.shape[i - 1];
            }
            return true;
        }
    };
}

pub fn RankedTensor(comptime rank: usize) type {
    return RankedTensorOf(.f32, rank);
}

pub fn TensorOf(comptime tensor_dtype: DType) type {
    const Elem = dtype_mod.Storage(tensor_dtype);
    const is_scalar_dtype = dtype_mod.isScalar(tensor_dtype);
    const ScalarElem = if (is_scalar_dtype) dtype_mod.Scalar(tensor_dtype) else void;
    const Buffer = storage.BufferOf(tensor_dtype);

    return struct {
        buffer: *Buffer,
        shape: Shape,
        strides: Shape,
        offset: usize = 0,

        const Self = @This();
        pub const dtype = tensor_dtype;
        pub const Element = Elem;

        pub fn zeros(allocator: Allocator, shape: []const usize) !Self {
            comptime if (!is_scalar_dtype) @compileError("zeros is only defined for scalar tensor dtypes");
            const size = try storageElementCount(tensor_dtype, shape);
            const buffer = try Buffer.create(allocator, size);
            @memset(buffer.data, dtype_mod.zero(tensor_dtype));
            errdefer buffer.release();

            return initFromBuffer(tensor_dtype, buffer, shape, 0);
        }

        pub fn ones(allocator: Allocator, shape: []const usize) !Self {
            comptime if (!is_scalar_dtype) @compileError("ones is only defined for scalar tensor dtypes");
            var out = try zeros(allocator, shape);
            @memset(out.data(), dtype_mod.one(tensor_dtype));
            return out;
        }

        pub fn fromSlice(allocator: Allocator, shape: []const usize, values: []const ScalarElem) !Self {
            comptime if (!is_scalar_dtype) @compileError("fromSlice is only defined for scalar tensor dtypes; use fromStorageSlice for block-quantized tensors");
            const size = try elementCount(shape);
            if (size != values.len) return TensorError.InvalidDataLength;

            const buffer = try Buffer.fromSlice(allocator, values);
            errdefer buffer.release();

            return initFromBuffer(tensor_dtype, buffer, shape, 0);
        }

        pub fn fromBorrowedSlice(allocator: Allocator, shape: []const usize, values: []ScalarElem) !Self {
            comptime if (!is_scalar_dtype) @compileError("fromBorrowedSlice is only defined for scalar tensor dtypes");
            const size = try elementCount(shape);
            if (size != values.len) return TensorError.InvalidDataLength;

            const buffer = try Buffer.fromBorrowedSlice(allocator, values);
            errdefer buffer.release();

            return initFromBuffer(tensor_dtype, buffer, shape, 0);
        }

        pub fn fromStorageSlice(allocator: Allocator, shape: []const usize, values: []const Elem) !Self {
            const size = try storageElementCount(tensor_dtype, shape);
            if (size != values.len) return TensorError.InvalidDataLength;

            const buffer = try Buffer.fromSlice(allocator, values);
            errdefer buffer.release();

            return initFromBuffer(tensor_dtype, buffer, shape, 0);
        }

        pub fn fromBorrowedStorageSlice(allocator: Allocator, shape: []const usize, values: []Elem) !Self {
            const size = try storageElementCount(tensor_dtype, shape);
            if (size != values.len) return TensorError.InvalidDataLength;

            const buffer = try Buffer.fromBorrowedSlice(allocator, values);
            errdefer buffer.release();

            return initFromBuffer(tensor_dtype, buffer, shape, 0);
        }

        // Takes ownership of one reference to buffer. Callers must not release that
        // reference after this succeeds; Tensor.deinit releases it.
        pub fn fromOwnedBuffer(buffer: *Buffer, shape: []const usize) !Self {
            const size = try storageElementCount(tensor_dtype, shape);
            if (buffer.data.len < size) return TensorError.InvalidDataLength;
            return initFromBuffer(tensor_dtype, buffer, shape, 0);
        }

        pub fn scalar(allocator: Allocator, value: ScalarElem) !Self {
            comptime if (!is_scalar_dtype) @compileError("scalar is only defined for scalar tensor dtypes");
            return fromSlice(allocator, &.{1}, &.{value});
        }

        pub fn deinit(self: *Self) void {
            self.buffer.release();
            self.* = undefined;
        }

        pub fn clone(self: *const Self, allocator: Allocator) !Self {
            const buffer = try Buffer.create(allocator, self.storageLen());
            errdefer buffer.release();

            var out = try initFromBuffer(tensor_dtype, buffer, self.shape.slice(), 0);
            try self.copyTo(out.data());
            return out;
        }

        pub fn cloneView(self: *const Self) !Self {
            self.buffer.retain();
            errdefer self.buffer.release();
            return initFromBufferWithStrides(tensor_dtype, self.buffer, self.shape.slice(), self.strides.slice(), self.offset);
        }

        pub fn viewWithStrides(self: *const Self, shape: []const usize, strides: []const usize) !Self {
            return self.viewWithStridesOffset(shape, strides, 0);
        }

        pub fn viewWithStridesOffset(self: *const Self, shape: []const usize, strides: []const usize, offset_delta: usize) !Self {
            if (comptime !is_scalar_dtype) {
                const same_shape = std.mem.eql(usize, shape, self.shape.slice());
                const same_strides = std.mem.eql(usize, strides, self.strides.slice());
                if (!same_shape or !same_strides or offset_delta != 0) return TensorError.UnsupportedView;
                return self.cloneView();
            }

            _ = try elementCount(shape);
            if (strides.len != shape.len) return TensorError.InvalidShape;

            const view_offset = try std.math.add(usize, self.offset, offset_delta);
            var max_index = view_offset;
            for (shape, strides) |dim, stride| {
                const span = try std.math.mul(usize, dim - 1, stride);
                max_index = try std.math.add(usize, max_index, span);
            }
            if (max_index >= self.buffer.data.len) return TensorError.InvalidDataLength;

            self.buffer.retain();
            errdefer self.buffer.release();
            return initFromBufferWithStrides(tensor_dtype, self.buffer, shape, strides, view_offset);
        }

        pub fn reshape(self: *const Self, new_shape: []const usize) !Self {
            if (comptime !is_scalar_dtype) {
                if (!std.mem.eql(usize, new_shape, self.shape.slice())) return TensorError.UnsupportedView;
                return self.cloneView();
            }

            if (!self.isContiguous()) return TensorError.UnsupportedView;
            if (try elementCount(new_shape) != self.len()) return TensorError.InvalidShape;

            self.buffer.retain();
            errdefer self.buffer.release();
            return initFromBuffer(tensor_dtype, self.buffer, new_shape, self.offset);
        }

        pub fn broadcastTo(self: *const Self, target_shape: []const usize) !Self {
            return dispatchRank(tensor_dtype, broadcastToDispatched, target_shape.len, .{ self, target_shape });
        }

        pub fn broadcastToRank(self: *const Self, comptime target_rank: usize, target_shape: [target_rank]usize) !Self {
            return dispatchRank(tensor_dtype, broadcastFromRankDispatched, self.shape.len, .{ self, target_rank, target_shape });
        }

        pub fn rank(self: *const Self) usize {
            return self.shape.len;
        }

        pub fn rankView(self: *const Self, comptime rank_value: usize) !RankedTensorOf(tensor_dtype, rank_value) {
            if (self.shape.len != rank_value) return TensorError.InvalidShape;

            var shape: [rank_value]usize = undefined;
            var strides: [rank_value]usize = undefined;
            inline for (0..rank_value) |i| {
                shape[i] = self.shape.at(i);
                strides[i] = self.strides.at(i);
            }

            return .{
                .tensor = self,
                .shape = shape,
                .strides = strides,
            };
        }

        pub fn len(self: *const Self) usize {
            return elementCountAssumeValid(self.shape.slice());
        }

        pub fn storageLen(self: *const Self) usize {
            return storageElementCountAssumeValid(tensor_dtype, self.shape.slice());
        }

        pub fn rows(self: *const Self) !usize {
            if (self.shape.len != 2) return TensorError.InvalidShape;
            return self.shape.at(0);
        }

        pub fn cols(self: *const Self) !usize {
            if (self.shape.len != 2) return TensorError.InvalidShape;
            return self.shape.at(1);
        }

        pub fn isScalar(self: *const Self) bool {
            return self.len() == 1;
        }

        pub fn isContiguous(self: *const Self) bool {
            var expected: usize = 1;
            var i = self.shape.len;
            while (i > 0) {
                i -= 1;
                if (self.strides.at(i) != expected) return false;
                expected *= self.shape.at(i);
            }
            return true;
        }

        // Safe only when the caller owns exclusive access to this Tensor value.
        // The refcount proves no other retained Tensor aliases the buffer now; it
        // is not a lock against another thread retaining the same Tensor later.
        pub fn canTakeInPlace(self: *const Self) bool {
            return self.offset == 0 and self.isContiguous() and self.buffer.isUnique();
        }

        pub fn dataChecked(self: *Self) ![]Elem {
            if (!self.isContiguous()) return TensorError.UnsupportedView;
            return self.buffer.data[self.offset .. self.offset + self.storageLen()];
        }

        pub fn dataConstChecked(self: *const Self) ![]const Elem {
            if (!self.isContiguous()) return TensorError.UnsupportedView;
            return self.buffer.data[self.offset .. self.offset + self.storageLen()];
        }

        pub fn data(self: *Self) []Elem {
            self.requireContiguousData();
            return self.buffer.data[self.offset .. self.offset + self.storageLen()];
        }

        pub fn dataConst(self: *const Self) []const Elem {
            self.requireContiguousData();
            return self.buffer.data[self.offset .. self.offset + self.storageLen()];
        }

        pub fn copyTo(self: *const Self, dst: []Elem) !void {
            if (dst.len != self.storageLen()) return TensorError.InvalidDataLength;
            if (comptime !is_scalar_dtype) {
                if (!self.isContiguous()) return TensorError.UnsupportedView;
                @memcpy(dst, self.dataConst());
                return;
            }
            if (self.isContiguous()) {
                @memcpy(dst, self.dataConst());
                return;
            }
            self.copyRangeTo(dst, 0, dst.len);
        }

        /// Copies `count` elements of the row-major linearization, starting
        /// at `linear_start`, into `dst[0..count]`. Strided views advance by
        /// an odometer (one coordinate decode per CALL, incremental stride
        /// arithmetic per run — never a per-element division); the maximal
        /// row-major-contiguous axis suffix copies as whole `@memcpy` runs,
        /// and a strided innermost axis copies as a simple strided loop.
        /// Disjoint ranges may be copied concurrently (read-only source).
        pub fn copyRangeTo(self: *const Self, dst: []Elem, linear_start: usize, count: usize) void {
            comptime if (!is_scalar_dtype) @compileError("copyRangeTo is only defined for scalar tensor dtypes");
            std.debug.assert(dst.len >= count);
            std.debug.assert(linear_start + count <= self.storageLen());
            if (count == 0) return;

            // Maximal suffix of axes laid out row-major-contiguously in the
            // view (dim-1 axes are absorbed regardless of their stride).
            var run_len: usize = 1;
            var outer_rank: usize = self.shape.len;
            while (outer_rank > 0) {
                const i = outer_rank - 1;
                const dim = self.shape.at(i);
                if (dim != 1 and self.strides.at(i) != run_len) break;
                run_len *= dim;
                outer_rank -= 1;
            }
            // A strided innermost axis still copies per run (incremental
            // stride arithmetic only), just not as a memcpy.
            var inner_stride: usize = 1;
            if (run_len == 1 and outer_rank > 0) {
                outer_rank -= 1;
                run_len = self.shape.at(outer_rank);
                inner_stride = self.strides.at(outer_rank);
            }

            // Decode the starting coordinates once.
            var coords: [max_rank]usize = undefined;
            var src_base: usize = self.offset;
            var within_run = linear_start % run_len;
            var remainder = linear_start / run_len;
            var i = outer_rank;
            while (i > 0) {
                i -= 1;
                const dim = self.shape.at(i);
                coords[i] = remainder % dim;
                remainder /= dim;
                src_base += coords[i] * self.strides.at(i);
            }

            var copied: usize = 0;
            while (copied < count) {
                const n = @min(run_len - within_run, count - copied);
                const src_off = src_base + within_run * inner_stride;
                if (inner_stride == 1) {
                    @memcpy(dst[copied..][0..n], self.buffer.data[src_off..][0..n]);
                } else {
                    var j: usize = 0;
                    var off = src_off;
                    while (j < n) : (j += 1) {
                        dst[copied + j] = self.buffer.data[off];
                        off += inner_stride;
                    }
                }
                copied += n;
                within_run = 0;

                // Odometer over the outer axes with incremental offsets.
                var axis = outer_rank;
                while (axis > 0) {
                    axis -= 1;
                    coords[axis] += 1;
                    src_base += self.strides.at(axis);
                    if (coords[axis] < self.shape.at(axis)) break;
                    src_base -= coords[axis] * self.strides.at(axis);
                    coords[axis] = 0;
                }
            }
        }

        pub fn item(self: *const Self) Elem {
            comptime if (!is_scalar_dtype) @compileError("item is only defined for scalar tensor dtypes");
            std.debug.assert(self.isScalar());
            return self.dataConst()[0];
        }

        pub fn addInPlace(self: *Self, other: *const Self) !void {
            comptime if (!is_scalar_dtype) @compileError("addInPlace is only defined for scalar tensor dtypes");
            try requireSameShapeOf(tensor_dtype, self, other);
            const x = self.data();
            const y = other.dataConst();
            for (x, y) |*a, b| a.* += b;
        }

        pub fn scaleInPlace(self: *Self, scalar_value: Elem) void {
            comptime if (!is_scalar_dtype) @compileError("scaleInPlace is only defined for scalar tensor dtypes");
            for (self.data()) |*v| v.* *= scalar_value;
        }

        pub fn fill(self: *Self, value: Elem) void {
            comptime if (!is_scalar_dtype) @compileError("fill is only defined for scalar tensor dtypes");
            @memset(self.data(), value);
        }

        fn requireContiguousData(self: *const Self) void {
            if (!self.isContiguous()) @panic("Tensor.data requires a contiguous tensor; materialize or use dataChecked");
        }

        fn broadcastFromRankToRank(
            self: *const Self,
            comptime source_rank: usize,
            comptime target_rank: usize,
            target_shape: [target_rank]usize,
        ) !Self {
            _ = try elementCountArray(target_rank, target_shape);
            if (source_rank > target_rank) return TensorError.ShapeMismatch;

            const source = try self.rankView(source_rank);
            const rank_diff = target_rank - source_rank;
            var target_strides: [target_rank]usize = undefined;

            inline for (0..target_rank) |target_i| {
                if (target_i < rank_diff) {
                    target_strides[target_i] = 0;
                } else {
                    const source_i = target_i - rank_diff;
                    const source_dim = source.shape[source_i];
                    const target_dim = target_shape[target_i];
                    if (source_dim == target_dim) {
                        target_strides[target_i] = source.strides[source_i];
                    } else if (source_dim == 1) {
                        target_strides[target_i] = 0;
                    } else {
                        return TensorError.ShapeMismatch;
                    }
                }
            }

            self.buffer.retain();
            errdefer self.buffer.release();
            return initFromBufferWithStrides(tensor_dtype, self.buffer, target_shape[0..], target_strides[0..], self.offset);
        }

        fn broadcastToDispatched(comptime target_rank: usize, self: *const Self, target_shape: []const usize) !Self {
            return self.broadcastToRank(target_rank, try shapeArrayFromSlice(target_rank, target_shape));
        }

        fn broadcastFromRankDispatched(
            comptime source_rank: usize,
            self: *const Self,
            comptime target_rank: usize,
            target_shape: [target_rank]usize,
        ) !Self {
            return self.broadcastFromRankToRank(source_rank, target_rank, target_shape);
        }
    };
}

pub const Tensor = TensorOf(.f32);

fn dispatchRank(comptime tensor_dtype: DType, comptime F: anytype, rank: usize, args: anytype) !TensorOf(tensor_dtype) {
    return switch (rank) {
        1 => @call(.auto, F, .{1} ++ args),
        2 => @call(.auto, F, .{2} ++ args),
        3 => @call(.auto, F, .{3} ++ args),
        4 => @call(.auto, F, .{4} ++ args),
        5 => @call(.auto, F, .{5} ++ args),
        6 => @call(.auto, F, .{6} ++ args),
        7 => @call(.auto, F, .{7} ++ args),
        8 => @call(.auto, F, .{8} ++ args),
        else => TensorError.InvalidShape,
    };
}

pub fn requireSameShape(a: *const Tensor, b: *const Tensor) !void {
    return requireSameShapeOf(.f32, a, b);
}

pub fn requireSameShapeOf(comptime tensor_dtype: DType, a: *const TensorOf(tensor_dtype), b: *const TensorOf(tensor_dtype)) !void {
    if (!std.mem.eql(usize, a.shape.slice(), b.shape.slice())) return TensorError.ShapeMismatch;
}

pub fn elementCount(shape: []const usize) !usize {
    if (shape.len == 0 or shape.len > max_rank) return TensorError.InvalidShape;
    var n: usize = 1;
    for (shape) |dim| {
        if (dim == 0) return TensorError.InvalidShape;
        n = try std.math.mul(usize, n, dim);
    }
    return n;
}

pub fn storageElementCount(comptime tensor_dtype: DType, shape: []const usize) !usize {
    if (comptime dtype_mod.isScalar(tensor_dtype)) return elementCount(shape);
    if (shape.len == 0 or shape.len > max_rank) return TensorError.InvalidShape;

    var prefix: usize = 1;
    for (shape[0 .. shape.len - 1]) |dim| {
        if (dim == 0) return TensorError.InvalidShape;
        prefix = try std.math.mul(usize, prefix, dim);
    }

    const last_dim = shape[shape.len - 1];
    if (last_dim == 0 or last_dim % dtype_mod.blockSize(tensor_dtype) != 0) return TensorError.InvalidShape;
    return try std.math.mul(usize, prefix, last_dim / dtype_mod.blockSize(tensor_dtype));
}

pub fn elementCountArray(comptime rank: usize, shape: [rank]usize) !usize {
    if (rank == 0 or rank > max_rank) return TensorError.InvalidShape;
    var n: usize = 1;
    inline for (shape) |dim| {
        if (dim == 0) return TensorError.InvalidShape;
        n = try std.math.mul(usize, n, dim);
    }
    return n;
}

pub fn storageElementCountArray(comptime tensor_dtype: DType, comptime rank: usize, shape: [rank]usize) !usize {
    if (comptime dtype_mod.isScalar(tensor_dtype)) return elementCountArray(rank, shape);
    if (rank == 0 or rank > max_rank) return TensorError.InvalidShape;

    var prefix: usize = 1;
    inline for (0..rank - 1) |i| {
        if (shape[i] == 0) return TensorError.InvalidShape;
        prefix = try std.math.mul(usize, prefix, shape[i]);
    }

    const last_dim = shape[rank - 1];
    if (last_dim == 0 or last_dim % dtype_mod.blockSize(tensor_dtype) != 0) return TensorError.InvalidShape;
    return try std.math.mul(usize, prefix, last_dim / dtype_mod.blockSize(tensor_dtype));
}

fn elementCountAssumeValid(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |dim| n *= dim;
    return n;
}

fn storageElementCountAssumeValid(comptime tensor_dtype: DType, shape: []const usize) usize {
    if (comptime dtype_mod.isScalar(tensor_dtype)) return elementCountAssumeValid(shape);
    var n: usize = 1;
    for (shape[0 .. shape.len - 1]) |dim| n *= dim;
    n *= shape[shape.len - 1] / dtype_mod.blockSize(tensor_dtype);
    return n;
}

pub fn elementCountArrayAssumeValid(comptime rank: usize, shape: [rank]usize) usize {
    var n: usize = 1;
    inline for (shape) |dim| n *= dim;
    return n;
}

fn shapeArrayFromSlice(comptime rank: usize, shape: []const usize) ![rank]usize {
    if (shape.len != rank) return TensorError.InvalidShape;

    var out: [rank]usize = undefined;
    inline for (0..rank) |i| {
        out[i] = shape[i];
    }
    _ = try elementCountArray(rank, out);
    return out;
}

fn initFromBuffer(comptime tensor_dtype: DType, buffer: *storage.BufferOf(tensor_dtype), shape: []const usize, offset: usize) !TensorOf(tensor_dtype) {
    const tensor_shape = try Shape.init(shape);
    var tensor_strides = try Shape.init(shape);
    writeContiguousStrides(tensor_strides.dims[0..tensor_strides.len], tensor_shape.slice());

    return .{
        .buffer = buffer,
        .shape = tensor_shape,
        .strides = tensor_strides,
        .offset = offset,
    };
}

fn initFromBufferWithStrides(
    comptime tensor_dtype: DType,
    buffer: *storage.BufferOf(tensor_dtype),
    shape: []const usize,
    strides: []const usize,
    offset: usize,
) !TensorOf(tensor_dtype) {
    if (strides.len != shape.len) return TensorError.InvalidShape;

    const tensor_shape = try Shape.init(shape);
    const tensor_strides = try Shape.initStrides(strides);

    return .{
        .buffer = buffer,
        .shape = tensor_shape,
        .strides = tensor_strides,
        .offset = offset,
    };
}

fn writeContiguousStrides(out: []usize, shape: []const usize) void {
    var stride: usize = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        out[i] = stride;
        stride *= shape[i];
    }
}

test {
    _ = @import("tensor_tests.zig");
}
