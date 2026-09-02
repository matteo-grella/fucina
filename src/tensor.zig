//! The raw runtime tensor: an untagged, reference-counted n-d f32/typed
//! buffer with strides — the single runtime currency every band below the
//! public facade trades in. In-tree code names it `fucina.internal.RawTensor`;
//! the public API is the tagged autograd `Tensor` (`ag.zig`). Views alias
//! their parent's storage with independent lifetimes. Shape and stride
//! arithmetic is `shape.zig`'s; this file owns the buffer binding.
//! Layer stack: docs/ARCHITECTURE.md.
const std = @import("std");
const dtype_mod = @import("dtype.zig");
const shape_mod = @import("shape.zig");
const storage = @import("storage.zig");

const Allocator = std.mem.Allocator;
pub const DType = dtype_mod.DType;
pub const Scalar = dtype_mod.Scalar;
pub const Storage = dtype_mod.Storage;
pub const Shape = shape_mod.Shape;
pub const max_rank = shape_mod.max_rank;
pub const invalid_rank_msg = shape_mod.invalid_rank_msg;
pub const too_many_tags_msg = shape_mod.too_many_tags_msg;
pub const requireSameShape = shape_mod.requireSameShape;

/// The shape/data error domain: `shape.ShapeError` plus the data and view
/// errors a tensor adds.
pub const TensorError = shape_mod.ShapeError || error{
    /// A non-shape argument fails its own validity check: a probability
    /// outside its interval, a non-positive cap, `min > max`, a zero step,
    /// duplicate scatter indices, an option combination that names no
    /// target. Shape and layout problems stay `InvalidShape`/`ShapeMismatch`.
    InvalidArgument,
    InvalidDataLength,
    IndexOutOfBounds,
    UnsupportedView,
    EmptySelection,
    DivisionByZero,
};

/// Type-erased releaser for a heap-retained `*TensorOf(dtype)` handle
/// (`deinit` then `destroy`), shared by the registries that hold facade
/// tensors behind `*anyopaque` (`ParamRegistry`, the ES slots).
pub fn makeRetainedRelease(comptime Raw: type) *const fn (*anyopaque, Allocator) void {
    return struct {
        fn release(ptr: *anyopaque, allocator: Allocator) void {
            const v: *Raw = @ptrCast(@alignCast(ptr));
            v.deinit();
            allocator.destroy(v);
        }
    }.release;
}

pub fn RankedTensorOf(comptime tensor_dtype: DType, comptime rank: usize) type {
    if (rank == 0 or rank > max_rank) @compileError(invalid_rank_msg);

    return struct {
        tensor: *const TensorOf(tensor_dtype),
        shape: [rank]usize,
        strides: [rank]usize,

        pub fn dim(self: @This(), comptime axis: usize) usize {
            if (axis >= rank) @compileError("axis out of bounds");
            return self.shape[axis];
        }

        pub fn len(self: @This()) usize {
            return shape_mod.product(&self.shape);
        }

        pub fn isContiguous(self: @This()) bool {
            return shape_mod.isContiguous(&self.shape, &self.strides);
        }
    };
}

pub fn RankedTensor(comptime rank: usize) type {
    return RankedTensorOf(.f32, rank);
}

/// The raw tensor type for one storage dtype: a refcounted `Buffer` plus
/// shape/strides metadata (views share the buffer; `deinit` releases one
/// reference). Scalar dtypes get element accessors (`data`, `at`, ...);
/// block-quantized dtypes carry block storage and reject per-element access
/// at comptime. `Tensor` is `TensorOf(.f32)`, the runtime's currency.
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
            const size = try shape_mod.storageElementCount(tensor_dtype, shape);
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
            const size = try shape_mod.elementCount(shape);
            if (size != values.len) return TensorError.InvalidDataLength;

            const buffer = try Buffer.fromSlice(allocator, values);
            errdefer buffer.release();

            return initFromBuffer(tensor_dtype, buffer, shape, 0);
        }

        pub fn fromBorrowedSlice(allocator: Allocator, shape: []const usize, values: []ScalarElem) !Self {
            comptime if (!is_scalar_dtype) @compileError("fromBorrowedSlice is only defined for scalar tensor dtypes");
            const size = try shape_mod.elementCount(shape);
            if (size != values.len) return TensorError.InvalidDataLength;

            const buffer = try Buffer.fromBorrowedSlice(allocator, values);
            errdefer buffer.release();

            return initFromBuffer(tensor_dtype, buffer, shape, 0);
        }

        pub fn fromStorageSlice(allocator: Allocator, shape: []const usize, values: []const Elem) !Self {
            const size = try shape_mod.storageElementCount(tensor_dtype, shape);
            if (size != values.len) return TensorError.InvalidDataLength;

            const buffer = try Buffer.fromSlice(allocator, values);
            errdefer buffer.release();

            return initFromBuffer(tensor_dtype, buffer, shape, 0);
        }

        pub fn fromBorrowedStorageSlice(allocator: Allocator, shape: []const usize, values: []Elem) !Self {
            const size = try shape_mod.storageElementCount(tensor_dtype, shape);
            if (size != values.len) return TensorError.InvalidDataLength;

            const buffer = try Buffer.fromBorrowedSlice(allocator, values);
            errdefer buffer.release();

            return initFromBuffer(tensor_dtype, buffer, shape, 0);
        }

        // Takes ownership of one reference to buffer. Callers must not release that
        // reference after this succeeds; Tensor.deinit releases it.
        pub fn fromOwnedBuffer(buffer: *Buffer, shape: []const usize) !Self {
            const size = try shape_mod.storageElementCount(tensor_dtype, shape);
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

            _ = try shape_mod.elementCount(shape);
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
            if (try shape_mod.elementCount(new_shape) != self.len()) return TensorError.InvalidShape;

            self.buffer.retain();
            errdefer self.buffer.release();
            return initFromBuffer(tensor_dtype, self.buffer, new_shape, self.offset);
        }

        pub fn broadcastTo(self: *const Self, target_shape: []const usize) !Self {
            return shape_mod.dispatchRank(broadcastToDispatched, target_shape.len, .{ self, target_shape });
        }

        pub fn broadcastToRank(self: *const Self, comptime target_rank: usize, target_shape: [target_rank]usize) !Self {
            return shape_mod.dispatchRank(broadcastFromRankDispatched, self.shape.len, .{ self, target_rank, target_shape });
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
            return shape_mod.elementCountAssumeValid(self.shape.slice());
        }

        pub fn storageLen(self: *const Self) usize {
            return shape_mod.storageElementCountAssumeValid(tensor_dtype, self.shape.slice());
        }

        pub fn isScalar(self: *const Self) bool {
            return self.len() == 1;
        }

        pub fn isContiguous(self: *const Self) bool {
            return shape_mod.isContiguous(self.shape.slice(), self.strides.slice());
        }

        // Safe only when the caller owns exclusive access to this Tensor value.
        // The refcount proves no other retained Tensor aliases the buffer now; it
        // is not a lock against another thread retaining the same Tensor later.
        pub fn canTakeInPlace(self: *const Self) bool {
            return self.offset == 0 and self.isContiguous() and self.buffer.isUnique();
        }

        /// Recoverable mutable element view: `error.UnsupportedView` on a
        /// non-contiguous view. The public facade's `data` reaches storage
        /// only through this pair, so the panicking fast path below never
        /// surfaces through the public API.
        pub fn dataChecked(self: *Self) ![]Elem {
            if (!self.isContiguous()) return TensorError.UnsupportedView;
            self.buffer.waitMutable();
            return self.buffer.data[self.offset .. self.offset + self.storageLen()];
        }

        pub fn dataConstChecked(self: *const Self) ![]const Elem {
            if (!self.isContiguous()) return TensorError.UnsupportedView;
            self.buffer.waitReady();
            return self.buffer.data[self.offset .. self.offset + self.storageLen()];
        }

        /// INTERNAL fast path: PANICS on a non-contiguous view. For call
        /// sites that already hold the contiguity invariant (kernels behind
        /// `prepareContiguous`, freshly allocated outputs); everything else
        /// takes `dataChecked`, and the public facade always does.
        pub fn data(self: *Self) []Elem {
            self.requireContiguousData();
            self.buffer.waitMutable();
            return self.buffer.data[self.offset .. self.offset + self.storageLen()];
        }

        /// Const sibling of `data`: same PANICKING contiguity contract;
        /// `dataConstChecked` is the recoverable form.
        pub fn dataConst(self: *const Self) []const Elem {
            self.requireContiguousData();
            self.buffer.waitReady();
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
            self.buffer.waitReady();

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
            try requireSameShape(self, other);
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
            _ = try shape_mod.elementCount(&target_shape);
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
            return self.broadcastToRank(target_rank, try shape_mod.arrayFromSlice(target_rank, target_shape));
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

fn initFromBuffer(comptime tensor_dtype: DType, buffer: *storage.BufferOf(tensor_dtype), shape: []const usize, offset: usize) !TensorOf(tensor_dtype) {
    const tensor_shape = try Shape.init(shape);
    var tensor_strides = tensor_shape;
    shape_mod.contiguousStridesInto(tensor_strides.dims[0..tensor_strides.len], tensor_shape.slice());

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

test {
    _ = @import("tensor_tests.zig");
}
