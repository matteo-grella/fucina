//! VJPs for pooling and nearest-neighbor upsampling.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const exec_mod = @import("../../exec.zig");
const core = @import("../core.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;

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

    pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
        if (!core.needs(self, 0)) return;
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

    pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
        if (!core.needs(self, 0)) return;
        out[0] = try ctx.avgPool2dBackward(gy, self.in_h, self.in_w, self.kernel, self.stride, self.pad);
    }

    pub const vtable = core.recordVTable(Self);
};

/// VJP of the 2× nearest upsample: a 2×2 stride-2 sum-pool of `gy`.
pub const Upsample2xNearestBackward = struct {
    parents: [1]?*GradState,

    const Self = @This();

    pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
        if (!core.needs(self, 0)) return;
        out[0] = try ctx.upsample2xNearestBackward(gy);
    }

    pub const vtable = core.recordVTable(Self);
};
