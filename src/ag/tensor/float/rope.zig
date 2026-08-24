//! f32 tensor methods: rotary position embedding. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const std = @import("std");
const exec_mod = @import("../../../exec.zig");
const tags_mod = @import("../../../tags.zig");
const backward_rope = @import("../../backward/rope.zig");

const ExecContext = exec_mod.ExecContext;
const RopeMode = exec_mod.RopeMode;
const Tag = tags_mod.Tag;
const RopeBackward = backward_rope.RopeBackward;
const RopeTableBackward = backward_rope.RopeTableBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const finishOp = plumbing.finishOp;

        /// Rotary position embedding over (`position_tag`, `feature_tag`).
        /// `source` selects the factor source at comptime (a closed set):
        ///
        ///   - `*const exec.RopeTable` (or `*RopeTable`) — prepared factors
        ///     (freq_factors/NTK scaling live there; the production path).
        ///     The table's `feature_dim` is the AUTHORITATIVE rotary span:
        ///     equal to `dim(feature_tag)` rotates fully; SMALLER rotates the
        ///     leading `feature_dim` dims and passes the tail through
        ///     unchanged (partial NEOX RoPE). Validate the span at the call
        ///     site if a mismatched table would be a bug (qwen35's
        ///     `partialRope` is the precedent).
        ///   - `exec.RopeTheta` (or `.{ .positions = p, .theta_base = t }`)
        ///     — on-the-fly factors, full rotation only.
        ///
        /// `mode` (.half | .interleaved) is comptime: the backward record
        /// types are parameterized on it. Differentiable in `self` (the
        /// backward applies the inverse rotation).
        pub fn rope(
            self: *const Self,
            ctx: *ExecContext,
            comptime position_tag: Tag,
            comptime feature_tag: Tag,
            source: anytype,
            comptime mode: RopeMode,
        ) !Self {
            const position_axis = comptime axis(position_tag);
            const feature_axis = comptime axis(feature_tag);
            const SourceT = @TypeOf(source);
            const info = @typeInfo(SourceT);
            if (comptime (info == .pointer and info.pointer.size == .one and info.pointer.child == exec_mod.RopeTable)) {
                // The table's feature_dim is the rotary span (full or
                // partial); the backward mirrors it with the inverse table.
                var value = try ctx.ropeWithTable(tag_rank, self.asRawTensor(), position_axis, feature_axis, source, mode);
                errdefer value.deinit();
                return finishOp(tags, ctx, value, self.requiresGrad(), RopeTableBackward(tags, position_axis, feature_axis, mode), .{ ctx.allocator, self.grad_state, source });
            }
            if (comptime info == .@"struct") {
                comptime {
                    for (info.@"struct".fields) |field| {
                        if (!std.mem.eql(u8, field.name, "positions") and !std.mem.eql(u8, field.name, "theta_base"))
                            @compileError("rope: unknown RopeTheta field ." ++ field.name);
                    }
                    if (!@hasField(SourceT, "positions") or !@hasField(SourceT, "theta_base"))
                        @compileError("rope: an on-the-fly source needs both .positions and .theta_base");
                }
                const theta = exec_mod.RopeTheta{ .positions = source.positions, .theta_base = source.theta_base };
                var value = try ctx.rope(tag_rank, self.asRawTensor(), position_axis, feature_axis, theta, mode, false);
                errdefer value.deinit();
                return finishOp(tags, ctx, value, self.requiresGrad(), RopeBackward(tags, position_axis, feature_axis, mode), .{ ctx.allocator, self.grad_state, theta.positions, theta.theta_base });
            }
            @compileError("rope: source must be a *const exec.RopeTable or an exec.RopeTheta (.{ .positions, .theta_base }); got " ++ @typeName(SourceT));
        }
    };
}
