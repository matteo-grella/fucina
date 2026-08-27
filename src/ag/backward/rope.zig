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

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            out[0] = try ctx.rope(rawRank(tags.len), gy, position_axis, feature_axis, .{ .positions = self.positions, .theta_base = self.theta_base }, mode, true);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.positions);
        }

        pub const vtable = core.recordVTable(Self);
    };
}

/// Backward for table-prepared RoPE (full or partial rotation). RETAINS the
/// forward table (refcounted handle, no positions/values copy) and applies
/// `ropeWithTableInverse`, which negates sin at apply time — bitwise
/// identical to applying a negated-table clone, and tables built with
/// `freq_factors` (Llama-3 long-context, Gemma global layers) keep the
/// exact inverse rotation in the VJP.
pub fn RopeTableBackward(
    comptime tags: anytype,
    comptime position_axis: usize,
    comptime feature_axis: usize,
    comptime mode: exec_mod.RopeMode,
) type {
    return struct {
        parents: [1]?*GradState,
        table: exec_mod.RopeTable,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const RawTensor, out: []?RawTensor) !void {
            if (!core.needs(self, 0)) return;
            // Mirrors the forward: the partial entry self-falls-back to the
            // full kernel when the table spans the whole feature axis.
            out[0] = try ctx.ropeWithTableInverse(rawRank(tags.len), gy, position_axis, feature_axis, &self.table, mode);
        }

        pub fn deinitFields(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.table.deinit();
        }

        pub const vtable = core.recordVTable(Self);
    };
}
