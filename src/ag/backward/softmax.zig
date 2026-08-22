//! VJPs for softmax-family ops: softmax, log-softmax, logsumexp.

const std = @import("std");
const backend_ops = @import("../../backend.zig").ops;
const backend_quant = @import("../../backend.zig").quantized_matmul;
const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const parallel = @import("../../parallel.zig");
const tag_ops = @import("../../tagged.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");
const vector_primitives = @import("../../backend/vector/primitives.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const inserted_axis = tags_mod.inserted_axis;
const rawRank = tags_mod.rawRank;
const tagIndex = tags_mod.tagIndex;
const removeTags = tags_mod.removeTags;
const dotResultTags = tags_mod.dotResultTags;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const intersectTags = tags_mod.intersectTags;
const tagsEqual = tags_mod.tagsEqual;

const common = @import("common.zig");
const PointwiseOp = tag_ops.PointwiseOp;
const rawShapeArray = common.rawShapeArray;
const rawShapeArrayOf = common.rawShapeArrayOf;
const rawStrideArray = common.rawStrideArray;
const taggedShapeArray = common.taggedShapeArray;
const tagsDifference = common.tagsDifference;
const tagsDifferenceLen = common.tagsDifferenceLen;
const reduceGradientToTags = common.reduceGradientToTags;
const contiguousForRead = common.contiguousForRead;
const expandGradientToTags = common.expandGradientToTags;
const contiguousForReadTyped = common.contiguousForReadTyped;
const axisGeometry = common.axisGeometry;
const gateGradientByMask = common.gateGradientByMask;
const cloneInverseRopeTable = common.cloneInverseRopeTable;

pub fn LogsumexpBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,
        output: RawTensor,

        const Self = @This();
        const rank = rawRank(source_tags.len);

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor, output: *const RawTensor) !void {
            _ = allocator;
            var in_view = try input.cloneView();
            errdefer in_view.deinit();
            self.* = .{
                .parents = .{parent},
                .input = in_view,
                .output = try output.cloneView(),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            var x_ready = try contiguousForRead(ctx, &self.input);
            defer x_ready.deinit();
            const x = x_ready.dataConst();
            var lse_ready = try contiguousForRead(ctx, &self.output);
            defer lse_ready.deinit();
            const lse = lse_ready.dataConst();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            const g = gy_ready.dataConst();

            const source_shape = rawShapeArray(source_tags, &self.input);
            var gx = try ctx.emptyRank(rank, source_shape);
            errdefer gx.deinit();
            const gxd = gx.data();

            // d(logsumexp)/dx = exp(x - lse), scaled by the reduced-shape
            // upstream gradient (the softmax of the row).
            const axis_dim = source_shape[axis];
            const geo = axisGeometry(rank, source_shape, axis);
            for (0..geo.outer) |outer_i| {
                const base = outer_i * axis_dim * geo.inner;
                for (0..geo.inner) |inner_i| {
                    const reduced = outer_i * geo.inner + inner_i;
                    const shift = lse[reduced];
                    const upstream = g[reduced];
                    for (0..axis_dim) |i| {
                        const offset = base + i * geo.inner + inner_i;
                        gxd[offset] = @exp(x[offset] - shift) * upstream;
                    }
                }
            }
            out[0] = gx;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
            self.output.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn LogSoftmaxBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        output: RawTensor,

        const Self = @This();
        const rank = rawRank(source_tags.len);

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, output: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .output = try output.cloneView(),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            var y_ready = try contiguousForRead(ctx, &self.output);
            defer y_ready.deinit();
            const y = y_ready.dataConst();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            const g = gy_ready.dataConst();

            const source_shape = rawShapeArray(source_tags, &self.output);
            var gx = try ctx.emptyRank(rank, source_shape);
            errdefer gx.deinit();
            const gxd = gx.data();

            // d(log_softmax)/dx: g - softmax·Σg, with softmax = exp(y)
            // (torch saves the output for exactly this identity).
            const axis_dim = source_shape[axis];
            const geo = axisGeometry(rank, source_shape, axis);
            for (0..geo.outer) |outer_i| {
                const base = outer_i * axis_dim * geo.inner;
                for (0..geo.inner) |inner_i| {
                    var g_sum: f32 = 0;
                    for (0..axis_dim) |i| g_sum += g[base + i * geo.inner + inner_i];
                    for (0..axis_dim) |i| {
                        const offset = base + i * geo.inner + inner_i;
                        gxd[offset] = g[offset] - @exp(y[offset]) * g_sum;
                    }
                }
            }
            out[0] = gx;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.output.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn SoftmaxBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        output: RawTensor,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, output: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .output = try output.cloneView(),
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.softmaxBackwardAxisRank(rawRank(tags.len), &self.output, gy, axis);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.output.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn SoftmaxExtBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        output: RawTensor,
        scale: f32,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, output: *const RawTensor, scale: f32) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .output = try output.cloneView(),
                .scale = scale,
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.softmaxExtBackwardAxisRank(rawRank(tags.len), &self.output, gy, axis, self.scale);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.output.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}
