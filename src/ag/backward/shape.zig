//! VJPs for shape/view ops: reshape, narrow, pad, concat,
//! broadcast, axis and strided views.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const exec_mod = @import("../../exec.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const inserted_axis = tags_mod.inserted_axis;
const rawRank = tags_mod.rawRank;
const tagIndex = tags_mod.tagIndex;

const common = @import("common.zig");
const rawShapeArray = common.rawShapeArray;
const rawStrideArray = common.rawStrideArray;
const reduceGradientToTags = common.reduceGradientToTags;

pub fn BroadcastBackward(comptime source_tags: anytype, comptime result_tags: anytype) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            out[0] = try reduceGradientToTags(result_tags, source_tags, ctx, gy, self.source_shape);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn NarrowBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,
        start: usize,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            out[0] = try ctx.sliceGradient(rawRank(source_tags.len), gy, self.source_shape, axis, self.start);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP for `pad` (constant padding along one axis): the source gradient is
/// the narrow of the upstream gradient at offset `before` with the source
/// axis length — pad positions hold a constant, so their gradient is dropped.
pub fn PadBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,
        before: usize,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            var view = try ctx.narrowAxis(.f32, rawRank(source_tags.len), gy, axis, self.before, self.source_shape[axis]);
            defer view.deinit();
            out[0] = try ctx.materialize(.f32, &view);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn ConcatBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: []?*GradState,
        sizes: []usize,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            var start: usize = 0;
            for (self.sizes, 0..) |size, i| {
                defer start += size;
                if (!core.needs(self, i)) continue;
                var view = try ctx.narrowAxis(.f32, rawRank(tags.len), gy, axis, start, size);
                defer view.deinit();
                out[i] = try ctx.materialize(.f32, &view);
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.parents);
            allocator.free(self.sizes);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub const ReshapeBackward = struct {
    const Self = @This();

    parents: [1]?*GradState,
    source_shape: []usize,

    pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
        if (!core.needs(self, 0)) return;

        var ready = if (gy.isContiguous()) try gy.cloneView() else try ctx.materialize(.f32, gy);
        defer ready.deinit();
        out[0] = try ready.reshape(self.source_shape);
    }

    pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.source_shape);
    }

    pub const vtable = core.recordVTable(Self);
};

pub fn AxisViewBackward(comptime source_tags: anytype, comptime axes: anytype) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;

            if (comptime source_tags.len == 0) {
                out[0] = try gy.clone(ctx.allocator);
                return;
            }

            var strides: [rawRank(source_tags.len)]usize = undefined;
            @memset(&strides, 0);
            inline for (axes, 0..) |source_axis, target_axis| {
                if (source_axis != inserted_axis) {
                    strides[source_axis] = gy.strides.at(target_axis);
                }
            }

            var view = try gy.viewWithStrides(self.source_shape[0..], strides[0..]);
            defer view.deinit();
            out[0] = try ctx.materialize(.f32, &view);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn StridedViewBackward(comptime source_tags: anytype, comptime view_tags: anytype) type {
    return struct {
        const source_rank = rawRank(source_tags.len);
        const view_rank = rawRank(view_tags.len);

        parents: [1]?*GradState,
        source_shape: [source_rank]usize,
        source_strides: [source_rank]usize,
        source_axis_order: [source_rank]usize,
        source_offset: usize,
        view_shape: [view_rank]usize,
        view_strides: [view_rank]usize,
        view_offset: usize,
        aliases_source: bool,
        order_preserving: bool,

        const Self = @This();
        const view_to_source_axis = viewToSourceAxis();

        /// The record for `view` over `source`: pure geometry, no resource
        /// is taken (the VJP scatters by strides, it never reads a value).
        pub fn of(parent: ?*GradState, source: *const RawTensor, view: *const RawTensor) Self {
            const source_shape = rawShapeArray(source_tags, source);
            const source_strides = rawStrideArray(source_tags, source);
            const view_shape = rawShapeArray(view_tags, view);
            const view_strides = rawStrideArray(view_tags, view);
            const aliases_source = source.buffer == view.buffer;
            return .{
                .parents = .{parent},
                .source_shape = source_shape,
                .source_strides = source_strides,
                .source_axis_order = sourceAxisOrder(source_strides),
                .source_offset = source.offset,
                .view_shape = view_shape,
                .view_strides = view_strides,
                .view_offset = view.offset,
                .aliases_source = aliases_source,
                .order_preserving = aliases_source and source.offset == view.offset and source.len() == view.len() and logicalOrderPreserving(source, view),
            };
        }

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;

            if (!self.gyShapeMatches(gy)) return tensor_mod.TensorError.ShapeMismatch;

            var ready = if (gy.isContiguous()) try gy.cloneView() else try ctx.materialize(.f32, gy);
            defer ready.deinit();
            if (!self.aliases_source or self.order_preserving) {
                out[0] = try ready.reshape(self.source_shape[0..]);
                return;
            }

            var gx = try ctx.empty(.f32, self.source_shape);
            errdefer gx.deinit();
            @memset(gx.data(), 0);

            const gyd = ready.dataConst();
            const gxd = gx.data();
            for (gyd, 0..) |g, linear| {
                const source_linear = try self.sourceLinearForViewLinear(linear);
                if (source_linear >= gxd.len) return tensor_mod.TensorError.InvalidShape;
                gxd[source_linear] += g;
            }
            out[0] = gx;
        }

        fn gyShapeMatches(self: *const Self, gy: *const RawTensor) bool {
            if (gy.shape.len != view_rank) return false;
            inline for (0..view_rank) |i| {
                if (gy.shape.at(i) != self.view_shape[i]) return false;
            }
            return true;
        }

        fn sourceLinearForViewLinear(self: *const Self, linear: usize) !usize {
            var view_coords: [view_rank]usize = undefined;
            const physical = try self.viewPhysicalOffset(linear, &view_coords);
            if (physical < self.source_offset) return tensor_mod.TensorError.InvalidShape;

            var known: [source_rank]bool = [_]bool{false} ** source_rank;
            var coords: [source_rank]usize = undefined;
            @memset(&coords, 0);
            inline for (0..view_rank) |view_axis| {
                if (comptime view_to_source_axis[view_axis]) |source_axis| {
                    const coord = view_coords[view_axis];
                    if (coord >= self.source_shape[source_axis]) return tensor_mod.TensorError.InvalidShape;
                    if (known[source_axis] and coords[source_axis] != coord) return tensor_mod.TensorError.InvalidShape;
                    coords[source_axis] = coord;
                    known[source_axis] = true;
                }
            }

            var remaining = physical - self.source_offset;
            inline for (0..source_rank) |axis| {
                if (known[axis]) {
                    const span = try std.math.mul(usize, coords[axis], self.source_strides[axis]);
                    if (span > remaining) return tensor_mod.TensorError.InvalidShape;
                    remaining -= span;
                }
            }

            for (self.source_axis_order) |axis| {
                if (known[axis]) continue;
                const dim = self.source_shape[axis];
                if (dim == 1) {
                    coords[axis] = 0;
                    known[axis] = true;
                    continue;
                }
                const stride = self.source_strides[axis];
                if (stride == 0) return tensor_mod.TensorError.UnsupportedView;
                const coord = remaining / stride;
                if (coord >= dim) return tensor_mod.TensorError.InvalidShape;
                coords[axis] = coord;
                known[axis] = true;
                remaining -= coord * stride;
            }
            if (remaining != 0) return tensor_mod.TensorError.InvalidShape;

            var out_linear: usize = 0;
            var contiguous_stride: usize = 1;
            var axis = source_rank;
            while (axis > 0) {
                axis -= 1;
                const span = try std.math.mul(usize, coords[axis], contiguous_stride);
                out_linear = try std.math.add(usize, out_linear, span);
                contiguous_stride = try std.math.mul(usize, contiguous_stride, self.source_shape[axis]);
            }
            return out_linear;
        }

        fn viewPhysicalOffset(self: *const Self, linear: usize, coords: *[view_rank]usize) !usize {
            var remaining = linear;
            var physical = self.view_offset;
            var axis = view_rank;
            while (axis > 0) {
                axis -= 1;
                const dim = self.view_shape[axis];
                const coord = remaining % dim;
                remaining /= dim;
                coords[axis] = coord;
                const span = try std.math.mul(usize, coord, self.view_strides[axis]);
                physical = try std.math.add(usize, physical, span);
            }
            return physical;
        }

        fn sourceAxisOrder(strides: [source_rank]usize) [source_rank]usize {
            var order: [source_rank]usize = undefined;
            inline for (0..source_rank) |i| order[i] = i;
            var i: usize = 1;
            while (i < source_rank) : (i += 1) {
                const axis = order[i];
                const stride = strides[axis];
                var j = i;
                while (j > 0 and stride > strides[order[j - 1]]) : (j -= 1) {
                    order[j] = order[j - 1];
                }
                order[j] = axis;
            }
            return order;
        }

        fn viewToSourceAxis() [view_rank]?usize {
            var out: [view_rank]?usize = [_]?usize{null} ** view_rank;
            inline for (view_tags, 0..) |tag, view_axis| {
                out[view_axis] = comptime tagIndex(source_tags, tag);
            }
            return out;
        }

        const LayoutChunks = struct {
            len: usize = 0,
            sizes: [tensor_mod.max_rank]usize = undefined,
            strides: [tensor_mod.max_rank]usize = undefined,
        };

        fn logicalOrderPreserving(source: *const RawTensor, view: *const RawTensor) bool {
            const source_chunks = layoutChunks(source.shape.slice(), source.strides.slice()) orelse return false;
            const view_chunks = layoutChunks(view.shape.slice(), view.strides.slice()) orelse return false;
            if (source_chunks.len != view_chunks.len) return false;
            for (0..source_chunks.len) |i| {
                if (source_chunks.sizes[i] != view_chunks.sizes[i]) return false;
                if (source_chunks.strides[i] != view_chunks.strides[i]) return false;
            }
            return true;
        }

        fn layoutChunks(shape: []const usize, strides: []const usize) ?LayoutChunks {
            var out: LayoutChunks = .{};
            var have_chunk = false;
            var chunk_size: usize = 1;
            var chunk_stride: usize = 0;

            var axis = shape.len;
            while (axis > 0) {
                axis -= 1;
                const dim = shape[axis];
                if (dim == 1) continue;
                const stride = strides[axis];
                if (!have_chunk) {
                    have_chunk = true;
                    chunk_size = dim;
                    chunk_stride = stride;
                    continue;
                }
                const expected_outer_stride = std.math.mul(usize, chunk_size, chunk_stride) catch return null;
                if (stride == expected_outer_stride) {
                    chunk_size = std.math.mul(usize, chunk_size, dim) catch return null;
                } else {
                    out.sizes[out.len] = chunk_size;
                    out.strides[out.len] = chunk_stride;
                    out.len += 1;
                    chunk_size = dim;
                    chunk_stride = stride;
                }
            }

            if (have_chunk) {
                out.sizes[out.len] = chunk_size;
                out.strides[out.len] = chunk_stride;
                out.len += 1;
            }
            return out;
        }

        pub const vtable = core.recordVTable(Self);
    };
}
