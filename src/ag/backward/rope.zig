//! VJPs for rotary position embedding.

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

pub fn RopeBackward(
    comptime tags: anytype,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    comptime mode: exec_mod.RopeMode,
) type {
    return struct {
        parents: [1]?*GradState,
        positions: []i32,
        theta_base: f32,

        const Self = @This();

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.rope(rawRank(tags.len), gy, position_axis, feature_axis, .{ .positions = self.positions, .theta_base = self.theta_base }, mode, true);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.positions);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// Backward for table-prepared RoPE (full or partial rotation). Clones the
/// forward table with negated sin instead of rebuilding from positions/theta,
/// so tables built with `freq_factors` (Llama-3 long-context, Gemma global
/// layers) get the exact inverse rotation in the VJP.
pub fn RopeTableBackward(
    comptime tags: anytype,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    comptime mode: exec_mod.RopeMode,
) type {
    return struct {
        parents: [1]?*GradState,
        inverse_table: exec_mod.RopeTable,

        const Self = @This();

        pub fn vjp(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            // Mirrors the forward: the partial entry self-falls-back to the
            // full kernel when the table spans the whole feature axis.
            out[0] = try ctx.ropeWithTable(rawRank(tags.len), gy, position_axis, feature_axis, &self.inverse_table, mode);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.inverse_table.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}
