//! VJPs for normalization ops: rmsnorm (+fused mul/add/rope),
//! layernorm, groupnorm.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const exec_mod = @import("../../exec.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const rawRank = tags_mod.rawRank;

const common = @import("common.zig");
const cloneInverseRopeTable = common.cloneInverseRopeTable;

pub fn RmsNormBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,
        eps: f32,

        const Self = @This();

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = (try ctx.rmsNormBackward(rawRank(tags.len), &self.input, gy, axis, self.eps, .{})).input;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn RmsNormMulBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [2]?*GradState,
        input: RawTensor,
        weight: RawTensor,
        eps: f32,

        const Self = @This();

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const need_input = needs_grad.len > 0 and needs_grad[0];
            const need_weight = needs_grad.len > 1 and needs_grad[1];
            if (!need_input and !need_weight) return;
            const result = try ctx.rmsNormBackward(rawRank(tags.len), &self.input, gy, axis, self.eps, .{ .weight = &self.weight, .need_input = need_input, .need_weight = need_weight });
            out[0] = result.input;
            if (need_weight) out[1] = result.weight;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
            self.weight.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn RmsNormMulAddBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [3]?*GradState,
        input: RawTensor,
        weight: RawTensor,
        eps: f32,

        const Self = @This();

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const need_input = needs_grad.len > 0 and needs_grad[0];
            const need_weight = needs_grad.len > 1 and needs_grad[1];
            if (need_input or need_weight) {
                const result = try ctx.rmsNormBackward(rawRank(tags.len), &self.input, gy, axis, self.eps, .{ .weight = &self.weight, .need_input = need_input, .need_weight = need_weight });
                out[0] = result.input;
                if (need_weight) out[1] = result.weight;
            }
            if (needs_grad.len > 2 and needs_grad[2]) {
                out[2] = try gy.cloneView();
            }
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
            self.weight.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn LayerNormBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,
        eps: f32,

        const Self = @This();

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = (try ctx.layerNormBackward(rawRank(tags.len), &self.input, gy, axis, self.eps, .{})).input;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn LayerNormAffineBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [3]?*GradState,
        input: RawTensor,
        weight: RawTensor,
        eps: f32,

        const Self = @This();

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const need_input = needs_grad.len > 0 and needs_grad[0];
            const need_weight = needs_grad.len > 1 and needs_grad[1];
            const need_bias = needs_grad.len > 2 and needs_grad[2];
            if (!need_input and !need_weight and !need_bias) return;

            const result = try ctx.layerNormBackward(rawRank(tags.len), &self.input, gy, axis, self.eps, .{ .weight = &self.weight, .need_input = need_input, .need_weight = need_weight, .need_bias = need_bias });
            if (need_input) out[0] = result.input.?;
            if (need_weight) out[1] = result.weight.?;
            if (need_bias) out[2] = result.bias.?;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
            self.weight.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

pub fn RmsNormMulRopeBackward(
    comptime tags: anytype,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    comptime mode: exec_mod.RopeMode,
) type {
    return struct {
        parents: [2]?*GradState,
        input: RawTensor,
        weight: RawTensor,
        eps: f32,
        inverse_table: exec_mod.RopeTable,

        const Self = @This();

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const need_input = needs_grad.len > 0 and needs_grad[0];
            const need_weight = needs_grad.len > 1 and needs_grad[1];
            if (!need_input and !need_weight) return;

            const rank = comptime rawRank(tags.len);
            var unrotated = try ctx.ropeWithTable(rank, gy, position_axis, feature_axis, &self.inverse_table, mode);
            defer unrotated.deinit();
            const result = try ctx.rmsNormBackward(rank, &self.input, &unrotated, feature_axis, self.eps, .{ .weight = &self.weight, .need_input = need_input, .need_weight = need_weight });
            out[0] = result.input;
            if (need_weight) out[1] = result.weight;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input.deinit();
            self.weight.deinit();
            self.inverse_table.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// VJP of GroupNorm. Three operands: input, weight, bias — the affine
/// operands are optional exactly like ConvTranspose1dBackward's bias (null
/// parent slots when the forward had none). The forward's weight is cloned
/// when present because it feeds dx (ĝ = gy⊙weight); statistics are
/// recomputed from the input inside the exec backward (the layerNorm VJP
/// convention).
pub fn GroupNormBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [3]?*GradState,
        groups: usize,
        eps: f32,
        input_value: RawTensor,
        weight_value: ?RawTensor = null,

        const Self = @This();

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const need_input = needs_grad.len > 0 and needs_grad[0];
            const need_weight = needs_grad.len > 1 and needs_grad[1];
            const need_bias = needs_grad.len > 2 and needs_grad[2];
            if (!need_input and !need_weight and !need_bias) return;

            const result = try ctx.groupNormBackward(&self.input_value, gy, self.groups, self.eps, .{ .weight = if (self.weight_value) |*w| w else null, .need_input = need_input, .need_weight = need_weight, .need_bias = need_bias });
            if (need_input) out[0] = result.input.?;
            if (need_weight) out[1] = result.weight.?;
            if (need_bias) out[2] = result.bias.?;
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.input_value.deinit();
            if (self.weight_value) |*w| w.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}
