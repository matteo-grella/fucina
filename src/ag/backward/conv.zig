//! VJPs for convolutions (1d/2d, causal, transpose) and
//! unfold/fold.

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

pub fn CausalDepthwiseConv1dBackward(
    comptime input_tags: anytype,
    comptime kernel_tags: anytype,
    comptime time_axis: usize,
    comptime channel_axis: usize,
) type {
    return struct {
        parents: [2]?*GradState,
        input_shape: [rawRank(input_tags.len)]usize,
        kernel_shape: [rawRank(kernel_tags.len)]usize,
        estimated_work: usize,
        input_value: RawTensor,
        kernel_value: RawTensor,
        dilation: usize,
        state: ?[]f32 = null,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            kernel_parent: ?*GradState,
            input: *const RawTensor,
            kernel: *const RawTensor,
            dilation: usize,
            state: ?[]const f32,
        ) !void {
            self.* = .{
                .parents = .{ input_parent, kernel_parent },
                .input_shape = rawShapeArray(input_tags, input),
                .kernel_shape = rawShapeArray(kernel_tags, kernel),
                .estimated_work = workEstimate(input_parent, kernel_parent, input, kernel),
                .input_value = try input.cloneView(),
                .kernel_value = undefined,
                .dilation = dilation,
            };
            errdefer self.input_value.deinit();
            self.kernel_value = try kernel.cloneView();
            errdefer self.kernel_value.deinit();
            if (state) |values| {
                self.state = try allocator.dupe(f32, values);
            }
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.causalDepthwiseConv1dBackwardInputAxisRank(
                    rawRank(input_tags.len),
                    gy,
                    &self.kernel_value,
                    time_axis,
                    channel_axis,
                    self.dilation,
                );
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.causalDepthwiseConv1dBackwardKernelAxisRank(
                    rawRank(input_tags.len),
                    &self.input_value,
                    gy,
                    time_axis,
                    channel_axis,
                    self.kernel_shape[1],
                    self.dilation,
                    self.state,
                );
            }
        }

        fn workEstimate(input_parent: ?*GradState, kernel_parent: ?*GradState, input: *const RawTensor, kernel: *const RawTensor) usize {
            var branches: usize = 0;
            if (input_parent != null) branches += 1;
            if (kernel_parent != null) branches += 1;
            if (branches == 0) return 0;
            const seq = input.shape.at(time_axis);
            const channels = input.shape.at(channel_axis);
            const taps = kernel.shape.at(1);
            const base = parallel.saturatedMul3(seq, channels, taps);
            return std.math.mul(usize, base, branches) catch std.math.maxInt(usize);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            self.input_value.deinit();
            self.kernel_value.deinit();
            if (self.state) |values| allocator.free(values);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn CausalConv1dBackward(
    comptime input_tags: anytype,
    comptime weight_tags: anytype,
    comptime time_axis: usize,
    comptime channel_axis: usize,
) type {
    return struct {
        parents: [2]?*GradState,
        input_shape: [rawRank(input_tags.len)]usize,
        weight_shape: [rawRank(weight_tags.len)]usize,
        dilation: usize,
        estimated_work: usize,
        input_value: RawTensor,
        weight_value: RawTensor,
        state: ?[]f32 = null,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            weight_parent: ?*GradState,
            input: *const RawTensor,
            weight: *const RawTensor,
            dilation: usize,
            state: ?[]const f32,
        ) !void {
            self.* = .{
                .parents = .{ input_parent, weight_parent },
                .input_shape = rawShapeArray(input_tags, input),
                .weight_shape = rawShapeArray(weight_tags, weight),
                .dilation = dilation,
                .estimated_work = workEstimate(input_parent, weight_parent, input, weight),
                .input_value = try input.cloneView(),
                .weight_value = undefined,
            };
            errdefer self.input_value.deinit();
            self.weight_value = try weight.cloneView();
            errdefer self.weight_value.deinit();
            if (state) |values| {
                self.state = try allocator.dupe(f32, values);
            }
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.causalConv1dBackwardInputAxisRank(
                    rawRank(input_tags.len),
                    gy,
                    &self.weight_value,
                    time_axis,
                    channel_axis,
                    self.dilation,
                );
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.causalConv1dBackwardWeightAxisRank(
                    rawRank(input_tags.len),
                    &self.input_value,
                    gy,
                    time_axis,
                    channel_axis,
                    self.weight_shape[0],
                    self.dilation,
                    self.state,
                );
            }
        }

        fn workEstimate(input_parent: ?*GradState, weight_parent: ?*GradState, input: *const RawTensor, weight: *const RawTensor) usize {
            var branches: usize = 0;
            if (input_parent != null) branches += 1;
            if (weight_parent != null) branches += 1;
            if (branches == 0) return 0;
            const seq = input.shape.at(time_axis);
            const in_channels = input.shape.at(channel_axis);
            const taps = weight.shape.at(0);
            const out_channels = weight.shape.at(2);
            const base = std.math.mul(usize, parallel.saturatedMul3(seq, in_channels, out_channels), taps) catch std.math.maxInt(usize);
            return std.math.mul(usize, base, branches) catch std.math.maxInt(usize);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            self.input_value.deinit();
            self.weight_value.deinit();
            if (self.state) |values| allocator.free(values);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn GroupedCausalConv1dBackward(
    comptime input_tags: anytype,
    comptime weight_tags: anytype,
    comptime time_axis: usize,
    comptime channel_axis: usize,
) type {
    return struct {
        parents: [2]?*GradState,
        input_shape: [rawRank(input_tags.len)]usize,
        weight_shape: [rawRank(weight_tags.len)]usize,
        dilation: usize,
        groups: usize,
        estimated_work: usize,
        input_value: RawTensor,
        weight_value: RawTensor,
        state: ?[]f32 = null,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            weight_parent: ?*GradState,
            input: *const RawTensor,
            weight: *const RawTensor,
            dilation: usize,
            groups: usize,
            state: ?[]const f32,
        ) !void {
            self.* = .{
                .parents = .{ input_parent, weight_parent },
                .input_shape = rawShapeArray(input_tags, input),
                .weight_shape = rawShapeArray(weight_tags, weight),
                .dilation = dilation,
                .groups = groups,
                .estimated_work = workEstimate(input_parent, weight_parent, input, weight),
                .input_value = try input.cloneView(),
                .weight_value = undefined,
            };
            errdefer self.input_value.deinit();
            self.weight_value = try weight.cloneView();
            errdefer self.weight_value.deinit();
            if (state) |values| {
                self.state = try allocator.dupe(f32, values);
            }
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.groupedCausalConv1dBackwardInputAxisRank(
                    rawRank(input_tags.len),
                    gy,
                    &self.weight_value,
                    time_axis,
                    channel_axis,
                    self.dilation,
                    self.groups,
                );
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.groupedCausalConv1dBackwardWeightAxisRank(
                    rawRank(input_tags.len),
                    &self.input_value,
                    gy,
                    time_axis,
                    channel_axis,
                    self.weight_shape[0],
                    self.dilation,
                    self.groups,
                    self.state,
                );
            }
        }

        fn workEstimate(input_parent: ?*GradState, weight_parent: ?*GradState, input: *const RawTensor, weight: *const RawTensor) usize {
            var branches: usize = 0;
            if (input_parent != null) branches += 1;
            if (weight_parent != null) branches += 1;
            if (branches == 0) return 0;
            const seq = input.shape.at(time_axis);
            const in_per_group = weight.shape.at(1);
            const taps = weight.shape.at(0);
            const out_channels = weight.shape.at(2);
            const base = std.math.mul(usize, parallel.saturatedMul3(seq, in_per_group, out_channels), taps) catch std.math.maxInt(usize);
            return std.math.mul(usize, base, branches) catch std.math.maxInt(usize);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            self.input_value.deinit();
            self.weight_value.deinit();
            if (self.state) |values| allocator.free(values);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP of the general non-causal conv1d (stride/pad/dilation/groups).
/// Modeled on CausalConv1dBackward: clones views of both operands, dispatches
/// to the dedicated exec backward kernels. The forward has no state operand,
/// so only input + weight receive gradients.
/// VJP of the channel-last conv2d — rank-3 `[h,w,cin]` input, rank-4
/// `[cout,kh,kw,cin/groups]` weight, optional rank-1 `[cout]` bias. grad_input /
/// grad_weight run the exec backward kernels; grad_bias = Σ gy over the spatial
/// (oh,ow) axes. Fixed rank (no comptime tag parameterization needed); no
/// dilation in 2-D.
pub const Conv2dBackward = struct {
    parents: [3]?*GradState,
    input_shape: [3]usize,
    weight_shape: [4]usize,
    stride: [2]usize,
    pad: [2]usize,
    groups: usize,
    estimated_work: usize,
    input_value: RawTensor,
    weight_value: RawTensor,

    const Self = @This();

    pub fn init(
        self: *Self,
        allocator: std.mem.Allocator,
        input_parent: ?*GradState,
        weight_parent: ?*GradState,
        bias_parent: ?*GradState,
        input: *const RawTensor,
        weight: *const RawTensor,
        stride: [2]usize,
        pad: [2]usize,
        groups: usize,
    ) !void {
        _ = allocator;
        const work = std.math.mul(usize, parallel.saturatedMul3(input.shape.at(0) * input.shape.at(1), weight.shape.at(3), weight.shape.at(0)), weight.shape.at(1) * weight.shape.at(2)) catch std.math.maxInt(usize);
        self.* = .{
            .parents = .{ input_parent, weight_parent, bias_parent },
            .input_shape = .{ input.shape.at(0), input.shape.at(1), input.shape.at(2) },
            .weight_shape = .{ weight.shape.at(0), weight.shape.at(1), weight.shape.at(2), weight.shape.at(3) },
            .stride = stride,
            .pad = pad,
            .groups = groups,
            .estimated_work = work,
            .input_value = try input.cloneView(),
            .weight_value = undefined,
        };
        errdefer self.input_value.deinit();
        self.weight_value = try weight.cloneView();
    }

    pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        if (needs_grad.len > 0 and needs_grad[0]) {
            out[0] = try ctx.conv2dBackwardInput(gy, &self.weight_value, self.input_shape[0], self.input_shape[1], self.stride, self.pad, self.groups);
        }
        if (needs_grad.len > 1 and needs_grad[1]) {
            out[1] = try ctx.conv2dBackwardWeight(&self.input_value, gy, self.weight_shape[1], self.weight_shape[2], self.stride, self.pad, self.groups);
        }
        if (needs_grad.len > 2 and needs_grad[2]) {
            var s0 = try ctx.sumAxisRank(3, gy, 0); // Σ over oh -> [ow, cout]
            defer s0.deinit();
            out[2] = try ctx.sumAxisRank(2, &s0, 0); // Σ over ow -> [cout]
        }
    }

    pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.input_value.deinit();
        self.weight_value.deinit();
    }

    pub const vtable = core.recordVTable(Self);
};

/// VJP for `unfold` (im2col patch extraction): the input gradient is the
/// overlap-accumulating `fold` (col2im) of the upstream column gradient —
/// the exact adjoint (pad-tap gradients drop, overlapping taps sum).
pub const UnfoldBackward = struct {
    parents: [1]?*GradState,
    output_size: [2]usize,
    kernel: [2]usize,
    stride: [2]usize,
    pad: [2]usize,

    const Self = @This();

    pub fn init(
        self: *Self,
        allocator: std.mem.Allocator,
        parent: ?*GradState,
        input: *const RawTensor,
        kernel: [2]usize,
        stride: [2]usize,
        pad: [2]usize,
    ) !void {
        _ = allocator;
        self.* = .{
            .parents = .{parent},
            .output_size = .{ input.shape.at(0), input.shape.at(1) },
            .kernel = kernel,
            .stride = stride,
            .pad = pad,
        };
    }

    pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        if (needs_grad.len == 0 or !needs_grad[0]) return;
        out[0] = try ctx.fold(gy, self.output_size, self.kernel, self.stride, self.pad);
    }

    pub const vtable = core.recordVTable(Self);
};

/// VJP for `fold` (col2im patch scatter): the column gradient is the
/// `unfold` (im2col) of the upstream image gradient — each patch tap reads
/// the image position it accumulated into (pad taps read 0).
pub const FoldBackward = struct {
    parents: [1]?*GradState,
    kernel: [2]usize,
    stride: [2]usize,
    pad: [2]usize,

    const Self = @This();

    pub fn init(
        self: *Self,
        allocator: std.mem.Allocator,
        parent: ?*GradState,
        kernel: [2]usize,
        stride: [2]usize,
        pad: [2]usize,
    ) !void {
        _ = allocator;
        self.* = .{
            .parents = .{parent},
            .kernel = kernel,
            .stride = stride,
            .pad = pad,
        };
    }

    pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        if (needs_grad.len == 0 or !needs_grad[0]) return;
        out[0] = try ctx.unfold(gy, self.kernel, self.stride, self.pad);
    }

    pub const vtable = core.recordVTable(Self);
};

pub fn Conv1dBackward(
    comptime input_tags: anytype,
    comptime weight_tags: anytype,
    comptime time_axis: usize,
    comptime channel_axis: usize,
) type {
    return struct {
        parents: [2]?*GradState,
        input_shape: [rawRank(input_tags.len)]usize,
        weight_shape: [rawRank(weight_tags.len)]usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        groups: usize,
        estimated_work: usize,
        input_value: RawTensor,
        weight_value: RawTensor,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            weight_parent: ?*GradState,
            input: *const RawTensor,
            weight: *const RawTensor,
            stride: usize,
            pad: usize,
            dilation: usize,
            groups: usize,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ input_parent, weight_parent },
                .input_shape = rawShapeArray(input_tags, input),
                .weight_shape = rawShapeArray(weight_tags, weight),
                .stride = stride,
                .pad = pad,
                .dilation = dilation,
                .groups = groups,
                .estimated_work = workEstimate(input_parent, weight_parent, input, weight),
                .input_value = try input.cloneView(),
                .weight_value = undefined,
            };
            errdefer self.input_value.deinit();
            self.weight_value = try weight.cloneView();
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.conv1dBackwardInputAxisRank(
                    rawRank(input_tags.len),
                    gy,
                    &self.weight_value,
                    time_axis,
                    channel_axis,
                    self.input_shape[time_axis],
                    self.stride,
                    self.pad,
                    self.dilation,
                    self.groups,
                );
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.conv1dBackwardWeightAxisRank(
                    rawRank(input_tags.len),
                    &self.input_value,
                    gy,
                    time_axis,
                    channel_axis,
                    self.weight_shape[0],
                    self.stride,
                    self.pad,
                    self.dilation,
                    self.groups,
                );
            }
        }

        fn workEstimate(input_parent: ?*GradState, weight_parent: ?*GradState, input: *const RawTensor, weight: *const RawTensor) usize {
            var branches: usize = 0;
            if (input_parent != null) branches += 1;
            if (weight_parent != null) branches += 1;
            if (branches == 0) return 0;
            const seq = input.shape.at(time_axis);
            const in_per_group = weight.shape.at(1);
            const taps = weight.shape.at(0);
            const out_channels = weight.shape.at(2);
            const base = std.math.mul(usize, parallel.saturatedMul3(seq, in_per_group, out_channels), taps) catch std.math.maxInt(usize);
            return std.math.mul(usize, base, branches) catch std.math.maxInt(usize);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input_value.deinit();
            self.weight_value.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP of ConvTranspose1d (the GEMM + col2im decomposition), COMPOSED from
/// existing exec ops: gcol = col2im1dBackward(gy) (the im2col-style gather),
/// then g_input = gcol·weight2, g_weight2 = gcolᵀ·input (both plain 2-D
/// GEMMs), g_bias = per-column sums of the FULL gy — including the
/// `output_pad` rows, onto which the forward broadcast the bias. The weight
/// gradient is wrt the PACKED `[K*OC, IC]` weight2 layout exactly as passed
/// to the forward. The bias operand is optional: its parent slot is null when
/// the forward had no bias, so its needs_grad flag can never be set.
pub fn ConvTranspose1dBackward(comptime input_tags: anytype) type {
    return struct {
        parents: [3]?*GradState,
        input_shape: [rawRank(input_tags.len)]usize,
        out_channels: usize,
        taps: usize,
        stride: usize,
        pad: usize,
        input_value: RawTensor,
        weight_value: RawTensor,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            weight_parent: ?*GradState,
            bias_parent: ?*GradState,
            input: *const RawTensor,
            weight2: *const RawTensor,
            out_channels: usize,
            taps: usize,
            stride: usize,
            pad: usize,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ input_parent, weight_parent, bias_parent },
                .input_shape = rawShapeArray(input_tags, input),
                .out_channels = out_channels,
                .taps = taps,
                .stride = stride,
                .pad = pad,
                .input_value = try input.cloneView(),
                .weight_value = undefined,
            };
            errdefer self.input_value.deinit();
            self.weight_value = try weight2.cloneView();
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const need_input = needs_grad.len > 0 and needs_grad[0];
            const need_weight = needs_grad.len > 1 and needs_grad[1];
            const need_bias = needs_grad.len > 2 and needs_grad[2];

            if (need_input or need_weight) {
                var gcol = try ctx.col2im1dBackwardAxisRank(
                    gy,
                    self.input_shape[0],
                    self.out_channels,
                    self.taps,
                    self.stride,
                    self.pad,
                );
                defer gcol.deinit();
                if (need_input) out[0] = try ctx.matmul2D(&gcol, &self.weight_value);
                if (need_weight) out[1] = try ctx.matmulTransA(&gcol, &self.input_value);
            }
            if (need_bias) out[2] = try ctx.sumAxisRank(2, gy, 0);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input_value.deinit();
            self.weight_value.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}
