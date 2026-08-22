//! VJPs for normalization ops: rmsnorm (+fused mul/add/rope),
//! layernorm, groupnorm.

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

pub fn RmsNormBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,
        eps: f32,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor, eps: f32) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .input = try input.cloneView(),
                .eps = eps,
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.rmsNormBackwardAxisRank(rawRank(tags.len), &self.input, gy, axis, self.eps);
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

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            weight_parent: ?*GradState,
            input: *const RawTensor,
            weight: *const RawTensor,
            eps: f32,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ input_parent, weight_parent },
                .input = try input.cloneView(),
                .weight = undefined,
                .eps = eps,
            };
            errdefer self.input.deinit();
            self.weight = try weight.cloneView();
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.rmsNormMulBackwardInputAxisRank(rawRank(tags.len), &self.input, &self.weight, gy, axis, self.eps);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.rmsNormMulBackwardWeightAxisRank(rawRank(tags.len), &self.input, gy, axis, self.eps);
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

pub fn RmsNormMulAddBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [3]?*GradState,
        input: RawTensor,
        weight: RawTensor,
        eps: f32,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            weight_parent: ?*GradState,
            residual_parent: ?*GradState,
            input: *const RawTensor,
            weight: *const RawTensor,
            eps: f32,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ input_parent, weight_parent, residual_parent },
                .input = try input.cloneView(),
                .weight = undefined,
                .eps = eps,
            };
            errdefer self.input.deinit();
            self.weight = try weight.cloneView();
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.rmsNormMulBackwardInputAxisRank(rawRank(tags.len), &self.input, &self.weight, gy, axis, self.eps);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.rmsNormMulBackwardWeightAxisRank(rawRank(tags.len), &self.input, gy, axis, self.eps);
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

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor, eps: f32) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .input = try input.cloneView(),
                .eps = eps,
            };
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.layerNormBackwardAxisRank(rawRank(tags.len), &self.input, gy, axis, self.eps);
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

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            weight_parent: ?*GradState,
            bias_parent: ?*GradState,
            input: *const RawTensor,
            weight: *const RawTensor,
            eps: f32,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ input_parent, weight_parent, bias_parent },
                .input = try input.cloneView(),
                .weight = undefined,
                .eps = eps,
            };
            errdefer self.input.deinit();
            self.weight = try weight.cloneView();
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const need_input = needs_grad.len > 0 and needs_grad[0];
            const need_weight = needs_grad.len > 1 and needs_grad[1];
            const need_bias = needs_grad.len > 2 and needs_grad[2];
            if (!need_input and !need_weight and !need_bias) return;

            const result = try ctx.layerNormAffineBackwardAxisRank(
                rawRank(tags.len),
                &self.input,
                &self.weight,
                gy,
                axis,
                self.eps,
                need_input,
                need_weight,
                need_bias,
            );
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

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            weight_parent: ?*GradState,
            input: *const RawTensor,
            weight: *const RawTensor,
            eps: f32,
            table: *const exec_mod.RopeTable,
        ) !void {
            self.* = .{
                .parents = .{ input_parent, weight_parent },
                .input = try input.cloneView(),
                .weight = undefined,
                .eps = eps,
                .inverse_table = undefined,
            };
            errdefer self.input.deinit();
            self.weight = try weight.cloneView();
            errdefer self.weight.deinit();
            self.inverse_table = try cloneInverseRopeTable(allocator, table);
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const need_input = needs_grad.len > 0 and needs_grad[0];
            const need_weight = needs_grad.len > 1 and needs_grad[1];
            if (!need_input and !need_weight) return;

            const rank = comptime rawRank(tags.len);
            var unrotated = try ctx.ropeAxisRankWithTable(rank, gy, position_axis, feature_axis, &self.inverse_table, mode);
            defer unrotated.deinit();
            if (need_input) {
                out[0] = try ctx.rmsNormMulBackwardInputAxisRank(rank, &self.input, &self.weight, &unrotated, feature_axis, self.eps);
            }
            if (need_weight) {
                out[1] = try ctx.rmsNormMulBackwardWeightAxisRank(rank, &self.input, &unrotated, feature_axis, self.eps);
            }
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

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            weight_parent: ?*GradState,
            bias_parent: ?*GradState,
            input: *const RawTensor,
            weight: ?*const RawTensor,
            groups: usize,
            eps: f32,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ input_parent, weight_parent, bias_parent },
                .groups = groups,
                .eps = eps,
                .input_value = try input.cloneView(),
            };
            errdefer self.input_value.deinit();
            if (weight) |w| {
                self.weight_value = try w.cloneView();
            }
        }

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const need_input = needs_grad.len > 0 and needs_grad[0];
            const need_weight = needs_grad.len > 1 and needs_grad[1];
            const need_bias = needs_grad.len > 2 and needs_grad[2];
            if (!need_input and !need_weight and !need_bias) return;

            const result = try ctx.groupNormBackwardAxisRank(
                &self.input_value,
                gy,
                self.groups,
                self.eps,
                if (self.weight_value) |*w| w else null,
                need_input,
                need_weight,
                need_bias,
            );
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
