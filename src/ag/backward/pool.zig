//! VJPs for pooling and nearest-neighbor upsampling.

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

/// VJP of the channel-last max pool2d: `gy` routes to each window's argmax
/// tap, recomputed in the exec backward kernel from the saved forward input
/// (no index tensor is stored).
pub const MaxPool2dBackward = struct {
    parents: [1]?*GradState,
    kernel: [2]usize,
    stride: [2]usize,
    pad: [2]usize,
    input_value: RawTensor,

    const Self = @This();

    pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor, kernel: [2]usize, stride: [2]usize, pad: [2]usize) !void {
        _ = allocator;
        self.* = .{
            .parents = .{parent},
            .kernel = kernel,
            .stride = stride,
            .pad = pad,
            .input_value = try input.cloneView(),
        };
    }

    pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        if (needs_grad.len == 0 or !needs_grad[0]) return;
        out[0] = try ctx.maxPool2dBackward(&self.input_value, gy, self.kernel, self.stride, self.pad);
    }

    pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.input_value.deinit();
    }

    pub const vtable = core.recordVTable(Self);
};

/// VJP of the channel-last avg pool2d: scatter `gy / valid_count` over each
/// window (only the input geometry is retained).
pub const AvgPool2dBackward = struct {
    parents: [1]?*GradState,
    in_h: usize,
    in_w: usize,
    kernel: [2]usize,
    stride: [2]usize,
    pad: [2]usize,

    const Self = @This();

    pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, in_h: usize, in_w: usize, kernel: [2]usize, stride: [2]usize, pad: [2]usize) !void {
        _ = allocator;
        self.* = .{ .parents = .{parent}, .in_h = in_h, .in_w = in_w, .kernel = kernel, .stride = stride, .pad = pad };
    }

    pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        if (needs_grad.len == 0 or !needs_grad[0]) return;
        out[0] = try ctx.avgPool2dBackward(gy, self.in_h, self.in_w, self.kernel, self.stride, self.pad);
    }

    pub const vtable = core.recordVTable(Self);
};

/// VJP of the 2× nearest upsample: a 2×2 stride-2 sum-pool of `gy`.
pub const Upsample2xNearestBackward = struct {
    parents: [1]?*GradState,

    const Self = @This();

    pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState) !void {
        _ = allocator;
        self.* = .{ .parents = .{parent} };
    }

    pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        _ = self;
        if (needs_grad.len == 0 or !needs_grad[0]) return;
        out[0] = try ctx.upsample2xNearestBackward(gy);
    }

    pub const vtable = core.recordVTable(Self);
};
