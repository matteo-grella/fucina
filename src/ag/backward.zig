const std = @import("std");
const backend_ops = @import("../backend.zig").ops;
const backend_quant = @import("../backend.zig").quantized_matmul;
const tensor_mod = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");
const exec_mod = @import("../exec.zig");
const parallel = @import("../parallel.zig");
const tag_ops = @import("../tagged.zig");
const core = @import("core.zig");
const tags_mod = @import("../tags.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const BackwardFunction = core.BackwardFunction;
const Tag = tags_mod.Tag;
const inserted_axis = tags_mod.inserted_axis;
const rawRank = tags_mod.rawRank;
const tagIndex = tags_mod.tagIndex;
const removeTags = tags_mod.removeTags;
const dotResultTags = tags_mod.dotResultTags;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const intersectTags = tags_mod.intersectTags;
const tagsEqual = tags_mod.tagsEqual;

pub const PointwiseOp = tag_ops.PointwiseOp;

fn rawShapeArray(comptime tags: anytype, value: *const RawTensor) [rawRank(tags.len)]usize {
    const rank = comptime rawRank(tags.len);
    var out: [rank]usize = undefined;
    inline for (0..rank) |i| {
        out[i] = value.shape.at(i);
    }
    return out;
}

fn rawStrideArray(comptime tags: anytype, value: *const RawTensor) [rawRank(tags.len)]usize {
    const rank = comptime rawRank(tags.len);
    var out: [rank]usize = undefined;
    inline for (0..rank) |i| {
        out[i] = value.strides.at(i);
    }
    return out;
}

fn taggedShapeArray(comptime tags: anytype, raw_shape: [rawRank(tags.len)]usize) [tags.len]usize {
    var out: [tags.len]usize = undefined;
    inline for (0..tags.len) |i| out[i] = raw_shape[i];
    return out;
}

fn tagsDifference(comptime tags: anytype, comptime keep_tags: anytype) [tagsDifferenceLen(tags, keep_tags)]Tag {
    var out: [tagsDifferenceLen(tags, keep_tags)]Tag = undefined;
    var out_i: usize = 0;
    inline for (tags) |tag| {
        if (comptime tagIndex(keep_tags, tag) == null) {
            out[out_i] = tag;
            out_i += 1;
        }
    }
    return out;
}

fn tagsDifferenceLen(comptime tags: anytype, comptime keep_tags: anytype) usize {
    var len: usize = 0;
    inline for (tags) |tag| {
        if (comptime tagIndex(keep_tags, tag) == null) len += 1;
    }
    return len;
}

/// Sum-reduce a result-tagged gradient back to an operand's tags/shape (the
/// pointwise broadcast-backward rule). Shared with `elemental.zig`.
pub fn reduceGradientToTags(
    comptime grad_tags: anytype,
    comptime target_tags: anytype,
    ctx: *ExecContext,
    grad: *const RawTensor,
    target_shape: [rawRank(target_tags.len)]usize,
) !RawTensor {
    if (comptime tagsEqual(grad_tags, target_tags)) {
        if (std.mem.eql(usize, grad.shape.slice(), target_shape[0..])) {
            return grad.cloneView();
        }
    }

    const reduce_tags = tagsDifference(grad_tags, target_tags);
    var reduced = try tag_ops.sumManyTensor(grad_tags, grad, ctx, reduce_tags);
    defer reduced.deinit();
    const remaining_tags = removeTags(grad_tags, reduce_tags);
    var aligned = try tag_ops.permuteTensorTo(remaining_tags, &reduced, target_tags);
    defer aligned.deinit();

    return ctx.reduceBroadcast(&aligned, target_shape[0..]);
}

/// Borrow-if-contiguous read access: a retained view when the layout is
/// already contiguous, else a materialized copy. Shared with `elemental.zig`.
pub fn contiguousForRead(ctx: *ExecContext, value: *const RawTensor) !RawTensor {
    if (value.isContiguous()) return value.cloneView();
    return ctx.materialize(value);
}

fn expandGradientToTags(
    comptime grad_tags: anytype,
    comptime target_tags: anytype,
    ctx: *ExecContext,
    grad: *const RawTensor,
    target_shape: [rawRank(target_tags.len)]usize,
) !RawTensor {
    const tagged_shape = taggedShapeArray(target_tags, target_shape);
    _ = ctx;
    return tag_ops.broadcastTensorTo(grad_tags, grad, target_tags, tagged_shape);
}

pub fn PointwiseBackward(
    comptime op: PointwiseOp,
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    comptime result_tags: anytype,
) type {
    return struct {
        parents: [2]?*GradState,
        left_shape: [rawRank(left_tags.len)]usize,
        right_shape: [rawRank(right_tags.len)]usize,
        left_value: ?RawTensor = null,
        right_value: ?RawTensor = null,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            left_parent: ?*GradState,
            right_parent: ?*GradState,
            left: *const RawTensor,
            right: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ left_parent, right_parent },
                .left_shape = rawShapeArray(left_tags, left),
                .right_shape = rawShapeArray(right_tags, right),
            };
            errdefer self.deinitFields();

            if (comptime op == .mul or op == .div or op == .max or op == .min) {
                self.left_value = try left.cloneView();
                self.right_value = try right.cloneView();
            }
        }

        const Side = enum { left, right };

        /// max/min gradient at the broadcast result shape: gy weighted by
        /// 1 where `favored` wins, 0.5 on exact ties (torch's subgradient,
        /// ±inf ties included), 0 where it loses — NaN positions weigh 0
        /// on both sides (IEEE compares are false; the torch formula).
        fn winnerWeightGrad(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, comptime favored: Side) !RawTensor {
            const result_shape = taggedShapeArray(result_tags, rawShapeArray(result_tags, gy));
            var left_view = try tag_ops.broadcastTensorTo(left_tags, &self.left_value.?, result_tags, result_shape);
            defer left_view.deinit();
            var right_view = try tag_ops.broadcastTensorTo(right_tags, &self.right_value.?, result_tags, result_shape);
            defer right_view.deinit();
            const win_op: exec_mod.CompareOp = comptime if ((op == .max) == (favored == .left)) .gt else .lt;
            var wins = try ctx.compare(win_op, &left_view, &right_view);
            defer wins.deinit();
            var ties = try ctx.compare(.eq, &left_view, &right_view);
            defer ties.deinit();
            var half_ties = try ctx.scale(&ties, 0.5);
            defer half_ties.deinit();
            var weight = try ctx.add(&wins, &half_ties);
            defer weight.deinit();
            return tag_ops.pointwise(.mul, result_tags, gy, ctx, result_tags, &weight);
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(
            ptr: *const anyopaque,
            ctx: *ExecContext,
            gy: *const RawTensor,
            needs_grad: []const bool,
            out: []?RawTensor,
        ) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len > 0 and needs_grad[0]) {
                var g = switch (op) {
                    .add, .sub => try gy.cloneView(),
                    .mul => try tag_ops.pointwise(.mul, result_tags, gy, ctx, right_tags, &self.right_value.?),
                    .div => try tag_ops.pointwise(.div, result_tags, gy, ctx, right_tags, &self.right_value.?),
                    .max, .min => try self.winnerWeightGrad(ctx, gy, .left),
                };
                defer g.deinit();
                out[0] = try reduceGradientToTags(result_tags, left_tags, ctx, &g, self.left_shape);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                var g = switch (op) {
                    .add => try gy.cloneView(),
                    .sub => try ctx.scale(gy, -1),
                    .mul => try tag_ops.pointwise(.mul, result_tags, gy, ctx, left_tags, &self.left_value.?),
                    .div => blk: {
                        const num_tags = comptime pointwiseResultTags(result_tags, left_tags);
                        var numerator = try tag_ops.pointwise(.mul, result_tags, gy, ctx, left_tags, &self.left_value.?);
                        defer numerator.deinit();
                        var denominator = try tag_ops.pointwise(.mul, right_tags, &self.right_value.?, ctx, right_tags, &self.right_value.?);
                        defer denominator.deinit();
                        var quotient = try tag_ops.pointwise(.div, num_tags, &numerator, ctx, right_tags, &denominator);
                        defer quotient.deinit();
                        const neg = try ctx.scale(&quotient, -1);
                        break :blk neg;
                    },
                    .max, .min => try self.winnerWeightGrad(ctx, gy, .right),
                };
                defer g.deinit();
                out[1] = try reduceGradientToTags(result_tags, right_tags, ctx, &g, self.right_shape);
            }
        }

        fn deinitFields(self: *Self) void {
            if (self.left_value) |*value| value.deinit();
            if (self.right_value) |*value| value.deinit();
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.deinitFields();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn CastBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState) !void {
            _ = allocator;
            self.* = .{ .parents = .{parent} };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            _ = ptr;
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try contiguousForRead(ctx, gy);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn IdentityBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState) !void {
            _ = allocator;
            self.* = .{ .parents = .{parent} };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            _ = ptr;
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try contiguousForRead(ctx, gy);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn Matmul2DBackward(comptime trans_b: bool) type {
    return struct {
        parents: [2]?*GradState,
        left: RawTensor,
        right: RawTensor,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            left_parent: ?*GradState,
            right_parent: ?*GradState,
            left: *const RawTensor,
            right: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ left_parent, right_parent },
                .left = try left.cloneView(),
                .right = undefined,
            };
            errdefer self.left.deinit();
            self.right = try right.cloneView();
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = if (comptime trans_b)
                    try ctx.matmul2D(gy, &self.right)
                else
                    try ctx.matmulTransB(gy, &self.right);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = if (comptime trans_b)
                    try ctx.matmulTransA(gy, &self.left)
                else
                    try ctx.matmulTransA(&self.left, gy);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.left.deinit();
            self.right.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn BmmBackward(comptime kind: exec_mod.BmmKind) type {
    return struct {
        parents: [2]?*GradState,
        left: RawTensor,
        right: RawTensor,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            left_parent: ?*GradState,
            right_parent: ?*GradState,
            left: *const RawTensor,
            right: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ left_parent, right_parent },
                .left = try left.cloneView(),
                .right = undefined,
            };
            errdefer self.left.deinit();
            self.right = try right.cloneView();
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));

            if (needs_grad.len > 0 and needs_grad[0]) {
                var full = switch (kind) {
                    .plain => try ctx.bmmTransB(gy, &self.right),
                    .trans_a => try ctx.bmmTransB(&self.right, gy),
                    .trans_b => try ctx.bmm(gy, &self.right),
                };
                defer full.deinit();
                out[0] = try ctx.reduceBroadcast(&full, self.left.shape.slice());
            }

            if (needs_grad.len > 1 and needs_grad[1]) {
                var full = switch (kind) {
                    .plain => try ctx.bmmTransA(&self.left, gy),
                    .trans_a => try ctx.bmm(&self.left, gy),
                    .trans_b => try ctx.bmmTransA(gy, &self.left),
                };
                defer full.deinit();
                out[1] = try ctx.reduceBroadcast(&full, self.right.shape.slice());
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.left.deinit();
            self.right.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

test "reduceGradientToTags uses direct view when tags and shape already match" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var grad = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer grad.deinit();

    const outstanding_before = ctx.rt.buffers.outstandingBuffers();
    var reduced = try reduceGradientToTags(.{ .batch, .hidden }, .{ .batch, .hidden }, &ctx, &grad, .{ 2, 3 });
    defer reduced.deinit();

    try std.testing.expectEqualSlices(f32, grad.dataConst(), reduced.dataConst());
    try std.testing.expectEqual(grad.dataConst().ptr, reduced.dataConst().ptr);
    try std.testing.expectEqual(outstanding_before, ctx.rt.buffers.outstandingBuffers());
}

pub const ReluBackward = struct {
    parents: [1]?*GradState,
    input: RawTensor,

    pub fn init(self: *ReluBackward, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor) !void {
        _ = allocator;
        self.* = .{
            .parents = .{parent},
            .input = try input.cloneView(),
        };
    }

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const ReluBackward = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const ReluBackward = @ptrCast(@alignCast(ptr));
        if (needs_grad.len == 0 or !needs_grad[0]) return;

        var x = try contiguousForRead(ctx, &self.input);
        defer x.deinit();
        var gy_ready = try contiguousForRead(ctx, gy);
        defer gy_ready.deinit();

        var gx = try ctx.empty(x.shape.slice());
        errdefer gx.deinit();
        for (x.dataConst(), gy_ready.dataConst(), gx.data()) |value, grad, *dst| {
            dst.* = if (value > 0) grad else 0;
        }
        out[0] = gx;
    }

    fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *ReluBackward = @ptrCast(@alignCast(ptr));
        self.input.deinit();
        core.destroyNode(ReluBackward, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinit,
    };
};

/// VJP of `relposShiftRank3` (S.2 skew). Forward is a per-query gather
/// `out[h,qi,kj] = bd[h,qi, kj+(Tq-1)-qi]`; the cotangent scatters back to the
/// gathered `P`-axis index (`kj→r` is a bijection within a query row, so no
/// intra-row accumulation, but unused relpos entries correctly get 0). Saves
/// only `p` (the input relpos-table dim) to size the gradient.
pub const RelposShiftBackward = struct {
    parents: [1]?*GradState,
    p: usize,

    pub fn init(self: *RelposShiftBackward, allocator: std.mem.Allocator, parent: ?*GradState, p: usize) !void {
        _ = allocator;
        self.* = .{ .parents = .{parent}, .p = p };
    }

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const RelposShiftBackward = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const RelposShiftBackward = @ptrCast(@alignCast(ptr));
        if (needs_grad.len == 0 or !needs_grad[0]) return;

        var gy_ready = try contiguousForRead(ctx, gy);
        defer gy_ready.deinit();
        const gv = try gy_ready.rankView(3); // [H, Tq, Tk]
        const h = gv.shape[0];
        const t_q = gv.shape[1];
        const t_k = gv.shape[2];
        const gyd = gy_ready.dataConst();

        var gbd = try ctx.emptyRank(3, .{ h, t_q, self.p });
        errdefer gbd.deinit();
        const gd = gbd.data();
        @memset(gd, 0);
        for (0..h) |hh| {
            for (0..t_q) |qi| {
                const in_row = (hh * t_q + qi) * self.p + ((t_q - 1) - qi);
                const out_row = (hh * t_q + qi) * t_k;
                for (0..t_k) |kj| gd[in_row + kj] += gyd[out_row + kj];
            }
        }
        out[0] = gbd;
    }

    fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *RelposShiftBackward = @ptrCast(@alignCast(ptr));
        core.destroyNode(RelposShiftBackward, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinit,
    };
};

pub const LeakyReluBackward = struct {
    parents: [1]?*GradState,
    input: RawTensor,
    negative_slope: f32,

    pub fn init(self: *LeakyReluBackward, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor, negative_slope: f32) !void {
        _ = allocator;
        self.* = .{
            .parents = .{parent},
            .input = try input.cloneView(),
            .negative_slope = negative_slope,
        };
    }

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const LeakyReluBackward = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const LeakyReluBackward = @ptrCast(@alignCast(ptr));
        if (needs_grad.len == 0 or !needs_grad[0]) return;

        var x = try contiguousForRead(ctx, &self.input);
        defer x.deinit();
        var gy_ready = try contiguousForRead(ctx, gy);
        defer gy_ready.deinit();

        var gx = try ctx.empty(x.shape.slice());
        errdefer gx.deinit();
        for (x.dataConst(), gy_ready.dataConst(), gx.data()) |value, grad, *dst| {
            dst.* = if (value > 0) grad else grad * self.negative_slope;
        }
        out[0] = gx;
    }

    fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *LeakyReluBackward = @ptrCast(@alignCast(ptr));
        self.input.deinit();
        core.destroyNode(LeakyReluBackward, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinit,
    };
};

/// Ops whose derivative is cheaper in terms of the forward OUTPUT t
/// (tanh' = 1 − t²): the node stores the output view instead of the input, so
/// the VJP never re-evaluates the transcendental and differentiates exactly
/// the value the vectorized forward kernel produced.
pub fn unaryUsesOutput(comptime op: exec_mod.UnaryOp) bool {
    return switch (op) {
        .tanh, .softcap_15, .reciprocal => true,
        else => false,
    };
}

pub fn UnaryBackward(comptime op: exec_mod.UnaryOp, comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,
        /// The forward input — or the forward OUTPUT for unaryUsesOutput ops.
        input: RawTensor,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .input = try input.cloneView(),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            var x = try contiguousForRead(ctx, &self.input);
            defer x.deinit();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();

            var gx = try ctx.empty(x.shape.slice());
            errdefer gx.deinit();
            const xs = x.dataConst();
            const gys = gy_ready.dataConst();
            const dsts = gx.data();
            // Elementwise map: chunking is partition-invariant (bitwise-equal
            // to the serial loop), so parallelize like the forward kernels —
            // transcendental derivatives (tanh/gelu/…) over vocab-sized
            // gradients otherwise serialize the whole backward pass.
            const total = dsts.len;
            if (total >= parallel.vector_elementwise_len_threshold) {
                if (ctx.workPool()) |pool| {
                    const task_count = parallel.cpuThreadCount(parallel.vector_max_threads);
                    var tasks: [parallel.vector_max_threads]ChunkTask = undefined;
                    for (0..task_count) |i| {
                        const s = i * total / task_count;
                        const e = (i + 1) * total / task_count;
                        tasks[i] = .{ .xs = xs[s..e], .gys = gys[s..e], .dsts = dsts[s..e] };
                    }
                    pool.parallelChunks(ChunkTask, tasks[0..task_count], ChunkTask.run);
                    out[0] = gx;
                    return;
                }
            }
            ChunkTask.run(&.{ .xs = xs, .gys = gys, .dsts = dsts });
            out[0] = gx;
        }

        const ChunkTask = struct {
            xs: []const f32,
            gys: []const f32,
            dsts: []f32,

            fn run(t: *const ChunkTask) void {
                if (comptime unaryUsesOutput(op)) {
                    for (t.xs, t.gys, t.dsts) |value, grad, *dst| {
                        dst.* = grad * unaryDerivativeFromOutput(op, value);
                    }
                } else {
                    for (t.xs, t.gys, t.dsts) |value, grad, *dst| {
                        dst.* = grad * unaryDerivative(op, value);
                    }
                }
            }
        };

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn ScaleBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,
        scalar_value: f32,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, scalar_value: f32) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .scalar_value = scalar_value,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.scale(gy, self.scalar_value);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// Dropout VJP: stores only {p, seed} — the mask is NEVER materialized. The
/// exec backward kernel regenerates the identical counter-based (seed, i)
/// mask, so the node is as cheap as ScaleBackward and recompute-safe under
/// activation checkpointing.
pub fn DropoutBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,
        p: f32,
        seed: u64,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, p: f32, seed: u64) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .p = p,
                .seed = seed,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.dropoutBackward(gy, self.p, self.seed);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub const ClampBackward = struct {
    parents: [1]?*GradState,
    input: RawTensor,
    min_value: f32,
    max_value: f32,

    pub fn init(self: *ClampBackward, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor, min_value: f32, max_value: f32) !void {
        _ = allocator;
        self.* = .{
            .parents = .{parent},
            .input = try input.cloneView(),
            .min_value = min_value,
            .max_value = max_value,
        };
    }

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const ClampBackward = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const ClampBackward = @ptrCast(@alignCast(ptr));
        if (needs_grad.len == 0 or !needs_grad[0]) return;

        var x = try contiguousForRead(ctx, &self.input);
        defer x.deinit();
        var gy_ready = try contiguousForRead(ctx, gy);
        defer gy_ready.deinit();

        var gx = try ctx.empty(x.shape.slice());
        errdefer gx.deinit();
        for (x.dataConst(), gy_ready.dataConst(), gx.data()) |value, grad, *dst| {
            dst.* = if (value >= self.min_value and value <= self.max_value) grad else 0;
        }
        out[0] = gx;
    }

    fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *ClampBackward = @ptrCast(@alignCast(ptr));
        self.input.deinit();
        core.destroyNode(ClampBackward, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinit,
    };
};

pub fn GatedBackward(
    comptime op: exec_mod.GatedOp,
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    comptime result_tags: anytype,
) type {
    return struct {
        parents: [2]?*GradState,
        left_shape: [rawRank(left_tags.len)]usize,
        right_shape: [rawRank(right_tags.len)]usize,
        result_shape: [rawRank(result_tags.len)]usize,
        left_value: RawTensor,
        right_value: RawTensor,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            left_parent: ?*GradState,
            right_parent: ?*GradState,
            left: *const RawTensor,
            right: *const RawTensor,
            result: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ left_parent, right_parent },
                .left_shape = rawShapeArray(left_tags, left),
                .right_shape = rawShapeArray(right_tags, right),
                .result_shape = rawShapeArray(result_tags, result),
                .left_value = try left.cloneView(),
                .right_value = undefined,
            };
            errdefer self.left_value.deinit();
            self.right_value = try right.cloneView();
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if ((needs_grad.len == 0 or !needs_grad[0]) and (needs_grad.len < 2 or !needs_grad[1])) return;

            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            const gyd = gy_ready.dataConst();

            const result_tagged_shape = taggedShapeArray(result_tags, self.result_shape);
            var left_b = try tag_ops.broadcastTensorTo(left_tags, &self.left_value, result_tags, result_tagged_shape);
            defer left_b.deinit();
            var right_b = try tag_ops.broadcastTensorTo(right_tags, &self.right_value, result_tags, result_tagged_shape);
            defer right_b.deinit();
            var left_ready = try contiguousForRead(ctx, &left_b);
            defer left_ready.deinit();
            var right_ready = try contiguousForRead(ctx, &right_b);
            defer right_ready.deinit();
            const left_data = left_ready.dataConst();
            const right_data = right_ready.dataConst();

            if (needs_grad.len > 0 and needs_grad[0]) {
                var g = try ctx.empty(self.result_shape[0..]);
                for (gyd, right_data, g.data()) |grad, gate, *dst| {
                    dst.* = grad * gatedActivation(op, gate);
                }
                defer g.deinit();
                out[0] = try reduceGradientToTags(result_tags, left_tags, ctx, &g, self.left_shape);
            }

            if (needs_grad.len > 1 and needs_grad[1]) {
                var g = try ctx.empty(self.result_shape[0..]);
                for (gyd, left_data, right_data, g.data()) |grad, left, gate, *dst| {
                    dst.* = grad * left * gatedActivationDerivative(op, gate);
                }
                defer g.deinit();
                out[1] = try reduceGradientToTags(result_tags, right_tags, ctx, &g, self.right_shape);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.left_value.deinit();
            self.right_value.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn SplitSwiGluBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .input = try input.cloneView(),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.splitSwiGluBackwardAxisRank(rawRank(tags.len), &self.input, gy, axis);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn SplitGluBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .input = try input.cloneView(),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.splitGluBackwardAxisRank(rawRank(tags.len), &self.input, gy, axis);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// VJP for `addScalar` (and `subScalar`): `d/dx (x + c) = 1`, so grad passes
/// through unchanged.
pub fn AddScalarBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState) !void {
            _ = allocator;
            self.* = .{ .parents = .{parent} };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            _ = ptr;
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.scale(gy, 1); // identity passthrough as a fresh owned tensor
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// VJP for `powScalar`: `d/dx (x^c) = c·x^(c-1)`, so `grad_x = gy · c · x^(c-1)`.
pub fn PowScalarBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,
        exponent: f32,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor, exponent: f32) !void {
            _ = allocator;
            self.* = .{ .parents = .{parent}, .input = try input.cloneView(), .exponent = exponent };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            var x = try contiguousForRead(ctx, &self.input);
            defer x.deinit();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();

            var gx = try ctx.empty(x.shape.slice());
            errdefer gx.deinit();
            const c = self.exponent;
            for (x.dataConst(), gy_ready.dataConst(), gx.data()) |value, grad, *dst| {
                dst.* = grad * c * std.math.pow(f32, value, c - 1);
            }
            out[0] = gx;
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// VJP for `maskedFill(x, mask, value)`: grad passes through where the mask is
/// clear, zero where it is set (`value` is a constant — no grad).
pub fn MaskedFillBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [1]?*GradState,
        mask: RawTensor,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, mask: *const RawTensor) !void {
            _ = allocator;
            self.* = .{ .parents = .{parent}, .mask = try mask.cloneView() };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            var m = try contiguousForRead(ctx, &self.mask);
            defer m.deinit();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            var gx = try ctx.empty(m.shape.slice());
            errdefer gx.deinit();
            for (m.dataConst(), gy_ready.dataConst(), gx.data()) |mv, grad, *dst| {
                dst.* = if (mv != 0) 0 else grad;
            }
            out[0] = gx;
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.mask.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// VJP for `where(x, cond, y)` = `cond ? x : y`: `grad_x = cond ? gy : 0` and
/// `grad_y = cond ? 0 : gy` (`cond` is a non-grad mask).
pub fn WhereBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [2]?*GradState,
        cond: RawTensor,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, x_parent: ?*GradState, y_parent: ?*GradState, cond: *const RawTensor) !void {
            _ = allocator;
            self.* = .{ .parents = .{ x_parent, y_parent }, .cond = try cond.cloneView() };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            var c = try contiguousForRead(ctx, &self.cond);
            defer c.deinit();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            if (needs_grad.len > 0 and needs_grad[0]) {
                var gx = try ctx.empty(c.shape.slice());
                errdefer gx.deinit();
                for (c.dataConst(), gy_ready.dataConst(), gx.data()) |cv, grad, *dst| {
                    dst.* = if (cv != 0) grad else 0;
                }
                out[0] = gx;
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                var gyy = try ctx.empty(c.shape.slice());
                errdefer gyy.deinit();
                for (c.dataConst(), gy_ready.dataConst(), gyy.data()) |cv, grad, *dst| {
                    dst.* = if (cv != 0) 0 else grad;
                }
                out[1] = gyy;
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.cond.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// Derivative expressed in the forward OUTPUT t (unaryUsesOutput ops only).
fn unaryDerivativeFromOutput(comptime op: exec_mod.UnaryOp, t: f32) f32 {
    return switch (op) {
        .tanh => 1 - t * t,
        // out = 15·tanh(x/15) ⇒ d/dx = 1 − (out/15)².
        .softcap_15 => blk: {
            const u = t * (1.0 / 15.0);
            break :blk 1 - u * u;
        },
        // out = 1/x ⇒ d/dx = -1/x² = -out².
        .reciprocal => -t * t,
        else => @compileError("unaryDerivativeFromOutput: op is not output-derivative"),
    };
}

fn unaryDerivative(comptime op: exec_mod.UnaryOp, value: f32) f32 {
    return switch (op) {
        .relu => if (value > 0) 1 else 0,
        .exp => @exp(value),
        .sqrt => 0.5 / @sqrt(value),
        .rsqrt => -0.5 / (value * @sqrt(value)),
        .sigmoid => blk: {
            const s = sigmoid(value);
            break :blk s * (1 - s);
        },
        .silu => blk: {
            const s = sigmoid(value);
            break :blk s * (1 + value * (1 - s));
        },
        .log => 1 / value,
        .log1p => 1 / (1 + value),
        .neg => -1,
        .abs => if (value > 0) 1 else if (value < 0) -1 else 0,
        .sin => @cos(value),
        .cos => -@sin(value),
        .tanh => blk: {
            // Only reached via input-based callers; the autograd node stores
            // the OUTPUT for tanh (unaryUsesOutput) and uses 1 - t².
            const t = std.math.tanh(value);
            break :blk 1 - t * t;
        },
        .fast_tanh => fastTanhDerivative(value),
        .gelu => geluDerivative(value),
        .quick_gelu => quickGeluDerivative(value),
        .softcap_15 => blk: {
            const t = std.math.tanh(value * (1.0 / 15.0));
            break :blk 1 - t * t;
        },
        .softcap_30 => blk: {
            const t = std.math.tanh(value * (1.0 / 30.0));
            break :blk 1 - t * t;
        },
        .gelu_quant => geluDerivative(value), // inference-only; exact-gelu derivative
        .elu => if (value > 0) 1 else @exp(value),
        .gelu_erf => geluErfDerivative(value),
        // Piecewise-constant ops: zero gradient almost everywhere (the
        // torch convention — jump points get the a.e. value, 0).
        .floor, .ceil, .round, .sign => 0,
        .reciprocal => -1 / (value * value),
    };
}

fn geluErfDerivative(value: f32) f32 {
    // d/dx [0.5·x·(1 + erf(x/√2))] = 0.5·(1 + erf(x/√2)) + x·φ(x),
    // with the standard-normal pdf φ(x) = exp(-x²/2)/√(2π).
    const inv_sqrt_2pi: f32 = 0.3989422804014327; // 1/√(2π)
    const cdf = 0.5 * (1 + backend_ops.erff(value * 0.70710678118654752440084436210484));
    return cdf + value * @exp(-0.5 * value * value) * inv_sqrt_2pi;
}

fn fastTanhDerivative(value: f32) f32 {
    const a: f32 = 2.45550750702956;
    const b: f32 = 0.893229853513558;
    const c: f32 = 0.821226666969744;
    const d: f32 = 2.44506634652299;
    const e: f32 = 0.814642734961073;

    const ax = @abs(value);
    const dax: f32 = if (value > 0) 1 else if (value < 0) -1 else 0;
    const x2 = value * value;
    const p = a + a * ax + (b + c * ax) * x2;
    const dp = a * dax + c * dax * x2 + (b + c * ax) * 2 * value;
    const numerator = value * p;
    const dnumerator = p + value * dp;

    const q = value + e * value * ax;
    const dq = 1 + e * (ax + value * dax);
    const r = @abs(q);
    const dr: f32 = if (q > 0) dq else if (q < 0) -dq else 0;
    const denominator = d + (d + x2) * r;
    const ddenominator = 2 * value * r + (d + x2) * dr;

    return (dnumerator * denominator - numerator * ddenominator) / (denominator * denominator);
}

fn sigmoid(value: f32) f32 {
    if (value >= 0) {
        const z = @exp(-value);
        return 1 / (1 + z);
    }
    const z = @exp(value);
    return z / (1 + z);
}

fn geluDerivative(value: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const x2 = value * value;
    const u = sqrt_2_over_pi * (value + 0.044715 * value * x2);
    const t = std.math.tanh(u);
    const du = sqrt_2_over_pi * (1 + 3 * 0.044715 * x2);
    return 0.5 * (1 + t) + 0.5 * value * (1 - t * t) * du;
}

fn quickGeluDerivative(value: f32) f32 {
    const s = sigmoid(1.702 * value);
    return s + value * 1.702 * s * (1 - s);
}

fn gatedActivation(comptime op: exec_mod.GatedOp, value: f32) f32 {
    return switch (op) {
        .glu => sigmoid(value),
        .swiglu => value * sigmoid(value),
        .geglu => 0.5 * value * (1 + std.math.tanh(0.7978845608028654 * (value + 0.044715 * value * value * value))),
    };
}

fn gatedActivationDerivative(comptime op: exec_mod.GatedOp, value: f32) f32 {
    return switch (op) {
        .glu => blk: {
            const s = sigmoid(value);
            break :blk s * (1 - s);
        },
        .swiglu => blk: {
            const s = sigmoid(value);
            break :blk s * (1 + value * (1 - s));
        },
        .geglu => geluDerivative(value),
    };
}

pub fn SumBackward(comptime source_tags: anytype, comptime result_tags: anytype) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try expandGradientToTags(result_tags, source_tags, ctx, gy, self.source_shape);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn MeanBackward(comptime source_tags: anytype, comptime result_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            var expanded = try expandGradientToTags(result_tags, source_tags, ctx, gy, self.source_shape);
            defer expanded.deinit();
            out[0] = try ctx.scale(&expanded, 1 / @as(f32, @floatFromInt(self.source_shape[axis])));
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// VJP for variance over an axis: dx = gy·2(x−μ)/(N−ddof) with μ the row
/// mean, recomputed here from the saved input.
pub fn VarBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,
        ddof: u1,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor, ddof: u1) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .input = try input.cloneView(),
                .ddof = ddof,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            const rank = comptime rawRank(source_tags.len);
            var x_ready = try contiguousForRead(ctx, &self.input);
            defer x_ready.deinit();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();

            const source_shape = rawShapeArray(source_tags, &self.input);
            var gx = try ctx.emptyRank(rank, source_shape);
            errdefer gx.deinit();
            const xd = x_ready.dataConst();
            const gyd = gy_ready.dataConst();
            const gxd = gx.data();

            const axis_dim = source_shape[axis];
            var inner: usize = 1;
            inline for (axis + 1..rank) |i| inner *= source_shape[i];
            var outer: usize = 1;
            inline for (0..axis) |i| outer *= source_shape[i];

            const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
            const scale_base = 2 / (@as(f32, @floatFromInt(axis_dim)) - @as(f32, @floatFromInt(self.ddof)));
            for (0..outer) |outer_i| {
                const base = outer_i * axis_dim * inner;
                for (0..inner) |inner_i| {
                    var sum: f32 = 0;
                    for (0..axis_dim) |axis_i| {
                        sum += xd[base + axis_i * inner + inner_i];
                    }
                    const mean_value = sum * inv_axis_dim;
                    const upstream = gyd[outer_i * inner + inner_i] * scale_base;
                    for (0..axis_dim) |axis_i| {
                        const offset = base + axis_i * inner + inner_i;
                        gxd[offset] = upstream * (xd[offset] - mean_value);
                    }
                }
            }
            out[0] = gx;
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn StandardizeBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,
        valid_len: ?usize,
        options: exec_mod.StandardizeOptions,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            input: *const RawTensor,
            valid_len: ?usize,
            options: exec_mod.StandardizeOptions,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .input = try input.cloneView(),
                .valid_len = valid_len,
                .options = options,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.standardizeBackwardAxisRank(rawRank(tags.len), &self.input, gy, axis, self.valid_len, self.options);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn BroadcastBackward(comptime source_tags: anytype, comptime result_tags: anytype) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try reduceGradientToTags(result_tags, source_tags, ctx, gy, self.source_shape);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn GatherBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,
        // Source element count: the scatter-add VJP touches every element of
        // the dense source-shaped gradient (zero-fill + accumulate), so this
        // is the work the engine weighs against
        // parallel.backward_async_work_threshold.
        estimated_work: usize,
        indices: []usize,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            source: *const RawTensor,
            indices: []const usize,
        ) !void {
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
                .estimated_work = if (parent != null) source.len() else 0,
                .indices = try allocator.dupe(usize, indices),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn estimatedWork(ptr: *const anyopaque) usize {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.estimated_work;
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.scatterAddAxisRank(rawRank(source_tags.len), gy, self.source_shape, axis, self.indices);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            allocator.free(self.indices);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
            .estimated_work = estimatedWork,
        };
    };
}

pub fn TopKBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,
        indices: RawTensor,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            source: *const RawTensor,
            indices: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
                .indices = try indices.cloneView(),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            const rank = comptime rawRank(source_tags.len);
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            var idx_ready = try contiguousForRead(ctx, &self.indices);
            defer idx_ready.deinit();

            var gx = try ctx.zeros(self.source_shape[0..]);
            errdefer gx.deinit();
            const gyd = gy_ready.dataConst();
            const idxd = idx_ready.dataConst();
            const gxd = gx.data();

            const axis_dim = self.source_shape[axis];
            const k = gy_ready.shape.at(axis);
            var inner: usize = 1;
            inline for (axis + 1..rank) |i| inner *= self.source_shape[i];
            var outer: usize = 1;
            inline for (0..axis) |i| outer *= self.source_shape[i];

            for (0..outer) |outer_i| {
                const gy_base = outer_i * k * inner;
                const gx_base = outer_i * axis_dim * inner;
                for (0..k) |slot| {
                    for (0..inner) |inner_i| {
                        const flat = gy_base + slot * inner + inner_i;
                        const index = @min(@as(usize, @intFromFloat(idxd[flat])), axis_dim - 1);
                        gxd[gx_base + index * inner + inner_i] += gyd[flat];
                    }
                }
            }
            out[0] = gx;
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.indices.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// VJP for max/min over an axis: the gradient flows ONLY to the first
/// occurrence of the extremum along the axis — the index captured by the
/// forward kernel (strict comparison tie-break), matching PyTorch's
/// single-index routing for torch.max/torch.min over a dim.
pub fn MinMaxBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,
        // First-extremum indices from the forward pass (out-shaped, f32 like
        // argmax/topK indices) — stored rather than recomputed.
        indices: RawTensor,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            source: *const RawTensor,
            indices: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
                .indices = try indices.cloneView(),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            const rank = comptime rawRank(source_tags.len);
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            var idx_ready = try contiguousForRead(ctx, &self.indices);
            defer idx_ready.deinit();

            var gx = try ctx.zeros(self.source_shape[0..]);
            errdefer gx.deinit();
            const gyd = gy_ready.dataConst();
            const idxd = idx_ready.dataConst();
            const gxd = gx.data();

            const axis_dim = self.source_shape[axis];
            var inner: usize = 1;
            inline for (axis + 1..rank) |i| inner *= self.source_shape[i];
            var outer: usize = 1;
            inline for (0..axis) |i| outer *= self.source_shape[i];

            for (0..outer) |outer_i| {
                const gx_base = outer_i * axis_dim * inner;
                for (0..inner) |inner_i| {
                    const flat = outer_i * inner + inner_i;
                    // Forward indices are stored as f32 (the repo-wide index
                    // convention, same as TopKBackward/argmax): exact only
                    // for indices < 2^24, so on a >2^24 axis the rounded
                    // float can land at axis_dim. Clamp so the routed index
                    // degrades (off-by-rounding) instead of writing out of
                    // bounds; extremumAxisRank documents the same contract
                    // at the store.
                    const index = @min(@as(usize, @intFromFloat(idxd[flat])), axis_dim - 1);
                    gxd[gx_base + index * inner + inner_i] = gyd[flat];
                }
            }
            out[0] = gx;
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.indices.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn NarrowBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,
        start: usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor, start: usize) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
                .start = start,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.sliceGradientAxisRank(rawRank(source_tags.len), gy, self.source_shape, axis, self.start);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// VJP for `cumsum` over an axis: gx[i] = Σ_{j >= i} gy[j] along the axis —
/// the REVERSED cumulative (suffix) sum of the upstream gradient, computed by
/// the dedicated serial `cumsumReverseAxisRank` exec helper (deterministic:
/// one serial pass per row, same order for any thread count).
pub fn CumsumBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState) !void {
            _ = allocator;
            self.* = .{ .parents = .{parent} };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            _ = ptr;
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.cumsumReverseAxisRank(rawRank(source_tags.len), gy, axis);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
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

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor, before: usize) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
                .before = before,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            var view = try ctx.narrowAxisRank(rawRank(source_tags.len), gy, axis, self.before, self.source_shape[axis]);
            defer view.deinit();
            out[0] = try ctx.materialize(&view);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn ConcatBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: []?*GradState,
        sizes: []usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parents: []const ?*GradState, sizes: []const usize) !void {
            if (parents.len != sizes.len) return tensor_mod.TensorError.InvalidShape;
            self.* = .{
                .parents = try allocator.dupe(?*GradState, parents),
                .sizes = undefined,
            };
            errdefer allocator.free(self.parents);
            self.sizes = try allocator.dupe(usize, sizes);
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents;
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            var start: usize = 0;
            for (self.sizes, 0..) |size, i| {
                defer start += size;
                if (i >= needs_grad.len or !needs_grad[i]) continue;
                var view = try ctx.narrowAxisRank(rawRank(tags.len), gy, axis, start, size);
                defer view.deinit();
                out[i] = try ctx.materialize(&view);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            allocator.free(self.parents);
            allocator.free(self.sizes);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn SetSliceBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [2]?*GradState,
        update_shape: [rawRank(tags.len)]usize,
        start: usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, base_parent: ?*GradState, update_parent: ?*GradState, update: *const RawTensor, start: usize) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ base_parent, update_parent },
                .update_shape = rawShapeArray(tags, update),
                .start = start,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.zeroSliceAxisRank(rawRank(tags.len), gy, axis, self.start, self.update_shape[axis]);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                var view = try ctx.narrowAxisRank(rawRank(tags.len), gy, axis, self.start, self.update_shape[axis]);
                defer view.deinit();
                out[1] = try ctx.materialize(&view);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn SetRowsBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [2]?*GradState,
        indices: []usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, base_parent: ?*GradState, update_parent: ?*GradState, indices: []const usize) !void {
            self.* = .{
                .parents = .{ base_parent, update_parent },
                .indices = try allocator.dupe(usize, indices),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.zeroRowsAxisRank(rawRank(tags.len), gy, axis, self.indices);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.gatherAxisRank(rawRank(tags.len), gy, axis, self.indices);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            allocator.free(self.indices);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// Row geometry for axis-wise scans: element (outer, i, inner) of a
/// contiguous `shape` tensor lives at `outer·(shape[axis]·inner_len) +
/// i·inner_len + inner`.
fn axisGeometry(comptime rank: usize, shape: [rank]usize, comptime axis: usize) struct { outer: usize, inner: usize } {
    var outer: usize = 1;
    for (0..axis) |i| outer *= shape[i];
    var inner: usize = 1;
    for (axis + 1..rank) |i| inner *= shape[i];
    return .{ .outer = outer, .inner = inner };
}

pub fn ProdBackward(comptime source_tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        input: RawTensor,

        const Self = @This();
        const rank = rawRank(source_tags.len);

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .input = try input.cloneView(),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            var x_ready = try contiguousForRead(ctx, &self.input);
            defer x_ready.deinit();
            const x = x_ready.dataConst();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            const g = gy_ready.dataConst();

            const source_shape = rawShapeArray(source_tags, &self.input);
            var gx = try ctx.emptyRank(rank, source_shape);
            errdefer gx.deinit();
            const gxd = gx.data();

            // torch.prod zero-handling: no zeros → g·(prod/x_i); exactly
            // one zero → its slot gets g·Π(nonzero), the rest 0; two or
            // more zeros → all 0. Computed division-free per row.
            const axis_dim = source_shape[axis];
            const geo = axisGeometry(rank, source_shape, axis);
            for (0..geo.outer) |outer_i| {
                const base = outer_i * axis_dim * geo.inner;
                for (0..geo.inner) |inner_i| {
                    var zero_count: usize = 0;
                    var nonzero_prod: f32 = 1;
                    for (0..axis_dim) |i| {
                        const v = x[base + i * geo.inner + inner_i];
                        if (v == 0) zero_count += 1 else nonzero_prod *= v;
                    }
                    const upstream = g[outer_i * geo.inner + inner_i];
                    for (0..axis_dim) |i| {
                        const offset = base + i * geo.inner + inner_i;
                        const v = x[offset];
                        gxd[offset] = switch (zero_count) {
                            0 => upstream * (nonzero_prod / v),
                            1 => if (v == 0) upstream * nonzero_prod else 0,
                            else => 0,
                        };
                    }
                }
            }
            out[0] = gx;
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn CumprodBackward(comptime source_tags: anytype, comptime axis: usize) type {
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            var x_ready = try contiguousForRead(ctx, &self.input);
            defer x_ready.deinit();
            const x = x_ready.dataConst();
            var y_ready = try contiguousForRead(ctx, &self.output);
            defer y_ready.deinit();
            const y = y_ready.dataConst();
            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            const g = gy_ready.dataConst();

            const source_shape = rawShapeArray(source_tags, &self.input);
            var gx = try ctx.emptyRank(rank, source_shape);
            errdefer gx.deinit();
            const gxd = gx.data();

            // Zero-free rows use the O(n) reverse-scan closed form
            // grad_i = (Σ_{j≥i} g_j·y_j)/x_i; rows containing a zero fall
            // back to the exact division-free O(n²) expansion
            // grad_i = Σ_{j≥i} g_j·Π_{k≤j, k≠i} x_k (torch semantics).
            const axis_dim = source_shape[axis];
            const geo = axisGeometry(rank, source_shape, axis);
            for (0..geo.outer) |outer_i| {
                const base = outer_i * axis_dim * geo.inner;
                for (0..geo.inner) |inner_i| {
                    var has_zero = false;
                    for (0..axis_dim) |i| {
                        if (x[base + i * geo.inner + inner_i] == 0) {
                            has_zero = true;
                            break;
                        }
                    }
                    if (!has_zero) {
                        var suffix: f32 = 0;
                        var i: usize = axis_dim;
                        while (i > 0) {
                            i -= 1;
                            const offset = base + i * geo.inner + inner_i;
                            suffix += g[offset] * y[offset];
                            gxd[offset] = suffix / x[offset];
                        }
                    } else {
                        for (0..axis_dim) |i| {
                            var prefix: f32 = 1;
                            for (0..i) |k| prefix *= x[base + k * geo.inner + inner_i];
                            var run = prefix;
                            var acc = g[base + i * geo.inner + inner_i] * run;
                            for (i + 1..axis_dim) |j| {
                                run *= x[base + j * geo.inner + inner_i];
                                acc += g[base + j * geo.inner + inner_i] * run;
                            }
                            gxd[base + i * geo.inner + inner_i] = acc;
                        }
                    }
                }
            }
            out[0] = gx;
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            self.output.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            self.output.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.output.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn TakeAlongBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        indices: []usize,
        source_shape: [rank]usize,

        const Self = @This();
        const rank = rawRank(tags.len);

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor, indices: []const usize) !void {
            self.* = .{
                .parents = .{parent},
                .indices = try allocator.dupe(usize, indices),
                .source_shape = rawShapeArray(tags, source),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            // Adjoint of the elementwise gather: scatter-add gy into zeros.
            var zeros_base = try ctx.zeros(self.source_shape[0..]);
            defer zeros_base.deinit();
            out[0] = try ctx.scatterAddAlongAxisRank(rank, &zeros_base, gy, axis, self.indices);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            allocator.free(self.indices);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn ScatterAlongBackward(comptime tags: anytype, comptime axis: usize, comptime accumulate: bool) type {
    return struct {
        parents: [2]?*GradState,
        indices: []usize,
        src_axis_len: usize,

        const Self = @This();
        const rank = rawRank(tags.len);

        pub fn init(self: *Self, allocator: std.mem.Allocator, base_parent: ?*GradState, src_parent: ?*GradState, indices: []const usize, src_axis_len: usize) !void {
            self.* = .{
                .parents = .{ base_parent, src_parent },
                .indices = try allocator.dupe(usize, indices),
                .src_axis_len = src_axis_len,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len > 0 and needs_grad[0]) {
                if (comptime accumulate) {
                    // scatter-add: d/dbase is the identity.
                    out[0] = try contiguousForRead(ctx, gy);
                } else {
                    // scatter (overwrite): base positions that were written
                    // lose their gradient — zero every addressed slot.
                    var gb = try ctx.materialize(gy);
                    errdefer gb.deinit();
                    const gbd = gb.data();
                    const out_shape = rawShapeArray(tags, gy);
                    const axis_dim = out_shape[axis];
                    const geo = axisGeometry(rank, out_shape, axis);
                    for (0..geo.outer) |outer_i| {
                        const out_base = outer_i * axis_dim * geo.inner;
                        const src_base = outer_i * self.src_axis_len * geo.inner;
                        for (0..self.src_axis_len) |i| {
                            for (0..geo.inner) |inner_i| {
                                const index = self.indices[src_base + i * geo.inner + inner_i];
                                gbd[out_base + index * geo.inner + inner_i] = 0;
                            }
                        }
                    }
                    out[0] = gb;
                }
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                // d/dsrc gathers the written slots (the torch formula; on
                // overwrite-duplicates every writer reads the winner's
                // gradient, matching torch.scatter's backward).
                out[1] = try ctx.takeAlongAxisRank(rank, gy, axis, self.indices, self.src_axis_len);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            allocator.free(self.indices);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn IndexAddBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [2]?*GradState,
        indices: []usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, base_parent: ?*GradState, update_parent: ?*GradState, indices: []const usize) !void {
            self.* = .{
                .parents = .{ base_parent, update_parent },
                .indices = try allocator.dupe(usize, indices),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            // out = self + scatterAdd(update): d/dself is the identity;
            // d/dupdate gathers the addressed rows (duplicate indices each
            // receive their position's gradient — the accumulation adjoint).
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try contiguousForRead(ctx, gy);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.gatherAxisRank(rawRank(tags.len), gy, axis, self.indices);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            allocator.free(self.indices);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn ZeroSliceBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        start: usize,
        length: usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, start: usize, length: usize) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .start = start,
                .length = length,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.zeroSliceAxisRank(rawRank(tags.len), gy, axis, self.start, self.length);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn ZeroRowsBackward(comptime tags: anytype, comptime axis: usize) type {
    return struct {
        parents: [1]?*GradState,
        indices: []usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, indices: []const usize) !void {
            self.* = .{
                .parents = .{parent},
                .indices = try allocator.dupe(usize, indices),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.zeroRowsAxisRank(rawRank(tags.len), gy, axis, self.indices);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            allocator.free(self.indices);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.softmaxBackwardAxisRank(rawRank(tags.len), &self.output, gy, axis);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.output.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.softmaxExtBackwardAxisRank(rawRank(tags.len), &self.output, gy, axis, self.scale);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.output.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.rmsNormBackwardAxisRank(rawRank(tags.len), &self.input, gy, axis, self.eps);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.rmsNormMulBackwardInputAxisRank(rawRank(tags.len), &self.input, &self.weight, gy, axis, self.eps);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.rmsNormMulBackwardWeightAxisRank(rawRank(tags.len), &self.input, gy, axis, self.eps);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            self.weight.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            self.weight.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.layerNormBackwardAxisRank(rawRank(tags.len), &self.input, gy, axis, self.eps);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            self.weight.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// Owned copy of a RoPE table with the sin half negated: applying the forward
/// rotation kernel with this table is the exact inverse (transpose) rotation,
/// i.e. the RoPE VJP. Cloning the table (instead of rebuilding from positions
/// and theta) preserves `freq_factors` scaling baked into the angles.
fn cloneInverseRopeTable(allocator: std.mem.Allocator, table: *const exec_mod.RopeTable) !exec_mod.RopeTable {
    const positions = try allocator.dupe(i32, table.positions);
    errdefer allocator.free(positions);
    const values = try allocator.dupe(f32, table.values);
    const angle_count = table.positions.len * table.pair_count;
    for (values[0..angle_count]) |*value| value.* = -value.*;
    return .{
        .allocator = allocator,
        .positions = positions,
        .theta_base = table.theta_base,
        .feature_dim = table.feature_dim,
        .pair_count = table.pair_count,
        .values = values,
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            self.weight.deinit();
            self.inverse_table.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub const GroupedCausalAttentionBackward = struct {
    parents: [3]?*GradState,
    q: RawTensor,
    k: RawTensor,
    v: RawTensor,
    kv_head_for_head: []usize,
    // Forward-saved per-(head, query) softmax {max, sum_exp} pairs (8 bytes
    // per row; empty = none): the backward's GEMM route rebuilds this
    // forward's probabilities in ONE pass instead of three (see
    // `groupedCausalAttentionBackwardSoftmaxRows`).
    row_stats: []f32,
    // The forward's output (refcounted view, no copy): with stats, the
    // softmax-backward row dot comes from sum(P*dP) = gy.O — length-d dots
    // instead of a kv_seq pass over both panels.
    out: RawTensor,
    scale_value: f32,
    window: usize,
    // false = bidirectional attention (block-diffusion canvas); the backward
    // re-masks to the same full key range the forward used.
    causal: bool,
    estimated_work: usize,

    pub fn init(
        self: *GroupedCausalAttentionBackward,
        allocator: std.mem.Allocator,
        q_parent: ?*GradState,
        k_parent: ?*GradState,
        v_parent: ?*GradState,
        q: *const RawTensor,
        k: *const RawTensor,
        v: *const RawTensor,
        kv_head_for_head: []const usize,
        scale_value: f32,
        window: usize,
        causal: bool,
        row_stats: []const f32,
        out: *const RawTensor,
    ) !void {
        self.* = .{
            .parents = .{ q_parent, k_parent, v_parent },
            .q = try q.cloneView(),
            .k = undefined,
            .v = undefined,
            .kv_head_for_head = undefined,
            .row_stats = undefined,
            .out = undefined,
            .scale_value = scale_value,
            .window = window,
            .causal = causal,
            .estimated_work = workEstimate(q_parent, k_parent, v_parent, q, k),
        };
        errdefer self.q.deinit();
        self.k = try k.cloneView();
        errdefer self.k.deinit();
        self.v = try v.cloneView();
        errdefer self.v.deinit();
        self.out = try out.cloneView();
        errdefer self.out.deinit();
        self.kv_head_for_head = try allocator.dupe(usize, kv_head_for_head);
        errdefer allocator.free(self.kv_head_for_head);
        self.row_stats = try allocator.dupe(f32, row_stats);
    }

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const GroupedCausalAttentionBackward = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn estimatedWork(ptr: *const anyopaque) usize {
        const self: *const GroupedCausalAttentionBackward = @ptrCast(@alignCast(ptr));
        return self.estimated_work;
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const GroupedCausalAttentionBackward = @ptrCast(@alignCast(ptr));
        const need_q = needs_grad.len > 0 and needs_grad[0];
        const need_k = needs_grad.len > 1 and needs_grad[1];
        const need_v = needs_grad.len > 2 and needs_grad[2];
        var grads = try ctx.groupedCausalAttentionBackward(
            &self.q,
            &self.k,
            &self.v,
            gy,
            self.kv_head_for_head,
            self.scale_value,
            self.window,
            self.causal,
            if (self.row_stats.len == 0) null else self.row_stats,
            if (self.row_stats.len == 0) null else &self.out,
            need_q,
            need_k,
            need_v,
        );
        defer grads.deinit();
        if (need_q) {
            out[0] = grads.q.?;
            grads.q = null;
        }
        if (need_k) {
            out[1] = grads.k.?;
            grads.k = null;
        }
        if (need_v) {
            out[2] = grads.v.?;
            grads.v = null;
        }
    }

    fn workEstimate(q_parent: ?*GradState, k_parent: ?*GradState, v_parent: ?*GradState, q: *const RawTensor, k: *const RawTensor) usize {
        var branches: usize = 0;
        if (q_parent != null) branches += 1;
        if (k_parent != null) branches += 1;
        if (v_parent != null) branches += 1;
        if (branches == 0) return 0;
        const q_seq = q.shape.at(0);
        const heads = q.shape.at(1);
        const d = q.shape.at(2);
        const kv_seq = k.shape.at(0);
        const base = parallel.saturatedMul3(q_seq, heads, kv_seq);
        return std.math.mul(usize, base, d * branches) catch std.math.maxInt(usize);
    }

    fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *GroupedCausalAttentionBackward = @ptrCast(@alignCast(ptr));
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.out.deinit();
        allocator.free(self.kv_head_for_head);
        allocator.free(self.row_stats);
        core.destroyNode(GroupedCausalAttentionBackward, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinit,
        .estimated_work = estimatedWork,
    };
};

/// VJP record of the fused `linearCrossEntropy` (loss = CE(x·Wᵀ, labels)):
/// saves x, W, the forward's logits, and the per-row {max, sum_exp} stats.
/// SINGLE-USE: the backward overwrites the saved logits in place with the
/// logit gradient (the record owns the buffer exclusively), so the full
/// [rows, classes] gradient never costs a second buffer; a repeat backward
/// over the same record errors loudly instead of computing garbage (the
/// scheduler otherwise permits re-walking a retained graph).
/// Differentiable in BOTH operands.
pub fn LinearCrossEntropyBackward(comptime options: exec_mod.CrossEntropyOptions) type {
    return struct {
        parents: [2]?*GradState,
        x: RawTensor,
        weight: RawTensor,
        logits: RawTensor,
        labels: []usize,
        row_stats: []f32,
        estimated_work: usize,
        consumed: bool = false,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            x_parent: ?*GradState,
            weight_parent: ?*GradState,
            x: *const RawTensor,
            weight: *const RawTensor,
            logits: *const RawTensor,
            labels: []const usize,
            row_stats: []const f32,
        ) !void {
            var branches: usize = 0;
            if (x_parent != null) branches += 1;
            if (weight_parent != null) branches += 1;
            self.* = .{
                .parents = .{ x_parent, weight_parent },
                .x = try x.cloneView(),
                .weight = undefined,
                .logits = undefined,
                .labels = undefined,
                .row_stats = undefined,
                .estimated_work = std.math.mul(usize, x.len(), weight.shape.at(0) * branches) catch std.math.maxInt(usize),
            };
            errdefer self.x.deinit();
            self.weight = try weight.cloneView();
            errdefer self.weight.deinit();
            self.logits = try logits.cloneView();
            errdefer self.logits.deinit();
            self.labels = try allocator.dupe(usize, labels);
            errdefer allocator.free(self.labels);
            self.row_stats = try allocator.dupe(f32, row_stats);
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn estimatedWork(ptr: *const anyopaque) usize {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.estimated_work;
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            const need_x = needs_grad.len > 0 and needs_grad[0];
            const need_weight = needs_grad.len > 1 and needs_grad[1];
            // The record exclusively owns its saved logits and the VJP
            // consumes them in place; the const-cast is the scheduler's
            // `*const` record pointer meeting that single-writer ownership.
            const mut_self: *Self = @constCast(self);
            if (self.consumed) return error.LinearCrossEntropyBackwardConsumed;
            mut_self.consumed = true;
            var grads = try ctx.linearCrossEntropyBackwardUpstream(
                &mut_self.x,
                &mut_self.weight,
                &mut_self.logits,
                self.labels,
                options,
                gy,
                self.row_stats,
                need_x,
                need_weight,
            );
            defer grads.deinit();
            if (need_x) {
                out[0] = grads.dx.?;
                grads.dx = null;
            }
            if (need_weight) {
                out[1] = grads.dweight.?;
                grads.dweight = null;
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.x.deinit();
            self.weight.deinit();
            self.logits.deinit();
            allocator.free(self.labels);
            allocator.free(self.row_stats);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
            .estimated_work = estimatedWork,
        };
    };
}

pub fn CrossEntropyBackward(comptime tags: anytype, comptime axis: usize) type {
    return CrossEntropyExtBackward(tags, axis, .{});
}

pub fn CrossEntropyExtBackward(comptime tags: anytype, comptime axis: usize, comptime options: exec_mod.CrossEntropyOptions) type {
    return struct {
        parents: [1]?*GradState,
        logits: RawTensor,
        labels: []usize,
        // Forward-saved per-position {max, sum_exp} (8 bytes per position):
        // the backward emits final gradients in ONE pass over the logits,
        // bitwise identical to recomputing the statistics (see
        // `crossEntropyBackwardExStatsAxisRank`).
        row_stats: []f32,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            logits: *const RawTensor,
            labels: []const usize,
            row_stats: []const f32,
        ) !void {
            self.* = .{
                .parents = .{parent},
                .logits = try logits.cloneView(),
                .labels = undefined,
                .row_stats = undefined,
            };
            errdefer self.logits.deinit();
            self.labels = try allocator.dupe(usize, labels);
            errdefer allocator.free(self.labels);
            self.row_stats = try allocator.dupe(f32, row_stats);
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            // For `.mean`/`.sum` the upstream gy must be a scalar; for `.none`
            // it's the per-position gradient tensor (class axis removed).
            out[0] = try ctx.crossEntropyBackwardExUpstreamStatsAxisRank(rawRank(tags.len), &self.logits, axis, self.labels, options, gy, self.row_stats);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.logits.deinit();
            allocator.free(self.labels);
            allocator.free(self.row_stats);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// Shared two-parent VJP template for the whole-tensor elementwise losses
/// (`mseLoss`/`huberLoss`/`bceLoss`/`klDivLoss`): saves views of input and
/// target and routes the upstream gradient (scalar for `.mean`/`.sum`,
/// per-element for `.none`) through the loss's `*BackwardUpstream` exec arm
/// once per operand that needs a gradient.
fn ElementwiseLossBackward(comptime Options: type, comptime options: Options, comptime upstream_fn: anytype) type {
    return struct {
        parents: [2]?*GradState,
        input: RawTensor,
        target: RawTensor,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            target_parent: ?*GradState,
            input: *const RawTensor,
            target: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ input_parent, target_parent },
                .input = try input.cloneView(),
                .target = undefined,
            };
            errdefer self.input.deinit();
            self.target = try target.cloneView();
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try upstream_fn(ctx, &self.input, &self.target, options, gy, .input);
            }
            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try upstream_fn(ctx, &self.input, &self.target, options, gy, .target);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input.deinit();
            self.target.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn MseLossBackward(comptime tags: anytype, comptime options: exec_mod.MseOptions) type {
    _ = tags;
    return ElementwiseLossBackward(exec_mod.MseOptions, options, ExecContext.mseBackwardUpstream);
}

pub fn HuberLossBackward(comptime tags: anytype, comptime options: exec_mod.HuberOptions) type {
    _ = tags;
    return ElementwiseLossBackward(exec_mod.HuberOptions, options, ExecContext.huberBackwardUpstream);
}

pub fn BceLossBackward(comptime tags: anytype, comptime options: exec_mod.BceOptions) type {
    _ = tags;
    return ElementwiseLossBackward(exec_mod.BceOptions, options, ExecContext.bceBackwardUpstream);
}

pub fn KlDivLossBackward(comptime tags: anytype, comptime options: exec_mod.KlDivOptions) type {
    _ = tags;
    return ElementwiseLossBackward(exec_mod.KlDivOptions, options, ExecContext.klDivBackwardUpstream);
}

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

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            positions: []const i32,
            theta_base: f32,
        ) !void {
            self.* = .{
                .parents = .{parent},
                .positions = try allocator.dupe(i32, positions),
                .theta_base = theta_base,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            out[0] = try ctx.ropeAxisRank(rawRank(tags.len), gy, position_axis, feature_axis, self.positions, self.theta_base, mode, true);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            allocator.free(self.positions);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
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

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            parent: ?*GradState,
            table: *const exec_mod.RopeTable,
        ) !void {
            self.* = .{
                .parents = .{parent},
                .inverse_table = try cloneInverseRopeTable(allocator, table),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;
            // Mirrors the forward: the partial entry self-falls-back to the
            // full kernel when the table spans the whole feature axis.
            out[0] = try ctx.ropePartialAxisRankWithTable(rawRank(tags.len), gy, position_axis, feature_axis, &self.inverse_table, mode);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.inverse_table.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub const ReshapeBackward = struct {
    parents: [1]?*GradState,
    source_shape: []usize,

    pub fn init(self: *ReshapeBackward, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor) !void {
        self.* = .{
            .parents = .{parent},
            .source_shape = try allocator.dupe(usize, source.shape.slice()),
        };
    }

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const ReshapeBackward = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const ReshapeBackward = @ptrCast(@alignCast(ptr));
        if (needs_grad.len == 0 or !needs_grad[0]) return;

        var ready = if (gy.isContiguous()) try gy.cloneView() else try ctx.materialize(gy);
        defer ready.deinit();
        out[0] = try ready.reshape(self.source_shape);
    }

    fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *ReshapeBackward = @ptrCast(@alignCast(ptr));
        allocator.free(self.source_shape);
        core.destroyNode(ReshapeBackward, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinit,
    };
};

pub fn AxisViewBackward(comptime source_tags: anytype, comptime axes: anytype) type {
    return struct {
        parents: [1]?*GradState,
        source_shape: [rawRank(source_tags.len)]usize,

        const Self = @This();

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor) !void {
            _ = allocator;
            self.* = .{
                .parents = .{parent},
                .source_shape = rawShapeArray(source_tags, source),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;

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
            out[0] = try ctx.materialize(&view);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
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

        pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState, source: *const RawTensor, view: *const RawTensor) !void {
            _ = allocator;
            const source_shape = rawShapeArray(source_tags, source);
            const source_strides = rawStrideArray(source_tags, source);
            const view_shape = rawShapeArray(view_tags, view);
            const view_strides = rawStrideArray(view_tags, view);
            const aliases_source = source.buffer == view.buffer;
            self.* = .{
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            if (!self.gyShapeMatches(gy)) return tensor_mod.TensorError.ShapeMismatch;

            var ready = if (gy.isContiguous()) try gy.cloneView() else try ctx.materialize(gy);
            defer ready.deinit();
            if (!self.aliases_source or self.order_preserving) {
                out[0] = try ready.reshape(self.source_shape[0..]);
                return;
            }

            var gx = try ctx.emptyRank(source_rank, self.source_shape);
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
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

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

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
        state: ?[]f32 = null,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            kernel_parent: ?*GradState,
            input: *const RawTensor,
            kernel: *const RawTensor,
            state: ?[]const f32,
        ) !void {
            self.* = .{
                .parents = .{ input_parent, kernel_parent },
                .input_shape = rawShapeArray(input_tags, input),
                .kernel_shape = rawShapeArray(kernel_tags, kernel),
                .estimated_work = workEstimate(input_parent, kernel_parent, input, kernel),
                .input_value = try input.cloneView(),
                .kernel_value = undefined,
            };
            errdefer self.input_value.deinit();
            self.kernel_value = try kernel.cloneView();
            errdefer self.kernel_value.deinit();
            if (state) |values| {
                self.state = try allocator.dupe(f32, values);
            }
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn estimatedWork(ptr: *const anyopaque) usize {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.estimated_work;
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.causalDepthwiseConv1dBackwardInputAxisRank(
                    rawRank(input_tags.len),
                    gy,
                    &self.kernel_value,
                    time_axis,
                    channel_axis,
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input_value.deinit();
            self.kernel_value.deinit();
            if (self.state) |values| allocator.free(values);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
            .estimated_work = estimatedWork,
        };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn estimatedWork(ptr: *const anyopaque) usize {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.estimated_work;
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input_value.deinit();
            self.weight_value.deinit();
            if (self.state) |values| allocator.free(values);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
            .estimated_work = estimatedWork,
        };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn estimatedWork(ptr: *const anyopaque) usize {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.estimated_work;
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input_value.deinit();
            self.weight_value.deinit();
            if (self.state) |values| allocator.free(values);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
            .estimated_work = estimatedWork,
        };
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

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn estimatedWork(ptr: *const anyopaque) usize {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        return self.estimated_work;
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const Self = @ptrCast(@alignCast(ptr));
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

    fn deinitFn(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.input_value.deinit();
        self.weight_value.deinit();
        core.destroyNode(Self, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinitFn,
        .estimated_work = estimatedWork,
    };
};

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

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        if (needs_grad.len == 0 or !needs_grad[0]) return;
        out[0] = try ctx.maxPool2dBackward(&self.input_value, gy, self.kernel, self.stride, self.pad);
    }

    fn deinitFn(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.input_value.deinit();
        core.destroyNode(Self, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinitFn,
    };
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

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        if (needs_grad.len == 0 or !needs_grad[0]) return;
        out[0] = try ctx.avgPool2dBackward(gy, self.in_h, self.in_w, self.kernel, self.stride, self.pad);
    }

    fn deinitFn(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        core.destroyNode(Self, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinitFn,
    };
};

/// VJP of the 2× nearest upsample: a 2×2 stride-2 sum-pool of `gy`.
pub const Upsample2xNearestBackward = struct {
    parents: [1]?*GradState,

    const Self = @This();

    pub fn init(self: *Self, allocator: std.mem.Allocator, parent: ?*GradState) !void {
        _ = allocator;
        self.* = .{ .parents = .{parent} };
    }

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        _ = ptr;
        if (needs_grad.len == 0 or !needs_grad[0]) return;
        out[0] = try ctx.upsample2xNearestBackward(gy);
    }

    fn deinitFn(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        core.destroyNode(Self, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinitFn,
    };
};

/// VJP of the per-channel PReLU: `gx = x > 0 ? gy : α[c]·gy`;
/// `gα[c] = Σ gy·min(x,0)` over the leading (row) axes.
pub const PreluChannelsBackward = struct {
    parents: [2]?*GradState,
    channels: usize,
    input_value: RawTensor,
    alpha_value: RawTensor,

    const Self = @This();

    pub fn init(self: *Self, allocator: std.mem.Allocator, input_parent: ?*GradState, alpha_parent: ?*GradState, input: *const RawTensor, alpha: *const RawTensor) !void {
        _ = allocator;
        self.* = .{
            .parents = .{ input_parent, alpha_parent },
            .channels = alpha.len(),
            .input_value = try input.cloneView(),
            .alpha_value = undefined,
        };
        errdefer self.input_value.deinit();
        self.alpha_value = try alpha.cloneView();
    }

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        if (needs_grad.len > 0 and needs_grad[0]) {
            out[0] = try ctx.preluChannelsBackwardInput(gy, &self.input_value, &self.alpha_value);
        }
        if (needs_grad.len > 1 and needs_grad[1]) {
            out[1] = try ctx.preluChannelsBackwardAlpha(gy, &self.input_value, self.channels);
        }
    }

    fn deinitFn(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.input_value.deinit();
        self.alpha_value.deinit();
        core.destroyNode(Self, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinitFn,
    };
};

/// VJP of the per-channel affine `y = x·scale[c] + shift[c]`:
/// `gx = gy·scale[c]` (the same kernel, shift-less), `gscale[c] = Σ gy·x` and
/// `gshift[c] = Σ gy` over the leading axes (suffix reduce).
pub const ChannelAffineBackward = struct {
    parents: [3]?*GradState,
    channels: usize,
    input_value: RawTensor,
    scale_value: RawTensor,

    const Self = @This();

    pub fn init(self: *Self, allocator: std.mem.Allocator, input_parent: ?*GradState, scale_parent: ?*GradState, shift_parent: ?*GradState, input: *const RawTensor, scale: *const RawTensor) !void {
        _ = allocator;
        self.* = .{
            .parents = .{ input_parent, scale_parent, shift_parent },
            .channels = scale.len(),
            .input_value = try input.cloneView(),
            .scale_value = undefined,
        };
        errdefer self.input_value.deinit();
        self.scale_value = try scale.cloneView();
    }

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        if (needs_grad.len > 0 and needs_grad[0]) {
            out[0] = try ctx.channelAffine(gy, &self.scale_value, null);
        }
        if (needs_grad.len > 1 and needs_grad[1]) {
            var prod = try ctx.mul(gy, &self.input_value);
            defer prod.deinit();
            out[1] = try ctx.reduceBroadcast(&prod, &.{self.channels});
        }
        if (needs_grad.len > 2 and needs_grad[2]) {
            out[2] = try ctx.reduceBroadcast(gy, &.{self.channels});
        }
    }

    fn deinitFn(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.input_value.deinit();
        self.scale_value.deinit();
        core.destroyNode(Self, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinitFn,
    };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn estimatedWork(ptr: *const anyopaque) usize {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.estimated_work;
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input_value.deinit();
            self.weight_value.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
            .estimated_work = estimatedWork,
        };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input_value.deinit();
            self.weight_value.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// VJP of the per-channel Snake activation. Three operands: input, alpha,
/// inv_b — alpha and inv_b are INDEPENDENT tensor inputs at this level (the
/// caller ties `inv_b = 1/(alpha+1e-9)` at load time; a trainer wanting a
/// single alpha parameter must chain through that relation itself). The two
/// per-channel parameter gradients come from one fused kernel pass; the one
/// that was not requested is dropped.
pub fn SnakeBackward(comptime tags: anytype) type {
    _ = tags;
    return struct {
        parents: [3]?*GradState,
        input_value: RawTensor,
        alpha_value: RawTensor,
        inv_b_value: RawTensor,

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            input_parent: ?*GradState,
            alpha_parent: ?*GradState,
            inv_b_parent: ?*GradState,
            input: *const RawTensor,
            alpha: *const RawTensor,
            inv_b: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ input_parent, alpha_parent, inv_b_parent },
                .input_value = try input.cloneView(),
                .alpha_value = undefined,
                .inv_b_value = undefined,
            };
            errdefer self.input_value.deinit();
            self.alpha_value = try alpha.cloneView();
            errdefer self.alpha_value.deinit();
            self.inv_b_value = try inv_b.cloneView();
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len > 0 and needs_grad[0]) {
                out[0] = try ctx.snakeRowsBackwardInput(&self.input_value, gy, &self.alpha_value, &self.inv_b_value);
            }
            const need_alpha = needs_grad.len > 1 and needs_grad[1];
            const need_inv_b = needs_grad.len > 2 and needs_grad[2];
            if (need_alpha or need_inv_b) {
                var params = try ctx.snakeRowsBackwardParams(&self.input_value, gy, &self.alpha_value, &self.inv_b_value);
                if (need_alpha) out[1] = params.alpha else params.alpha.deinit();
                if (need_inv_b) out[2] = params.inv_b else params.inv_b.deinit();
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input_value.deinit();
            self.alpha_value.deinit();
            self.inv_b_value.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
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

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
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

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.input_value.deinit();
            if (self.weight_value) |*w| w.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

pub fn DotBackward(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) type {
    // `dot` is the single-contract-tag einsum; its VJP record is the einsum
    // one with the canonical dot result order as the equation.
    return EinsumBackward(left_tags, right_tags, dotResultTags(left_tags, right_tags, contract_tag));
}

/// Backward for `einsum` (multi-index tagged contraction). Contractions are
/// closed under differentiation: the gradient w.r.t. one operand is another
/// einsum — the output gradient contracted with the other operand, keeping
/// exactly this operand's recoverable tags — broadcast over any axes the
/// forward summed away. Both operand gradients therefore lower onto GEMM/BMM
/// kernels; there is no pointwise fallback.
pub fn EinsumBackward(comptime left_tags: anytype, comptime right_tags: anytype, comptime out_tags: anytype) type {
    // Unions here are membership sets, not tensor tag tuples: they may
    // legally exceed max_rank (`unionTags`, not `pointwiseResultTags`).
    const left_recover_tags = intersectTags(left_tags, tags_mod.unionTags(out_tags, right_tags));
    const right_recover_tags = intersectTags(right_tags, tags_mod.unionTags(out_tags, left_tags));
    const dropped_tags = tagsDifference(tags_mod.unionTags(left_tags, right_tags), out_tags);

    return struct {
        parents: [2]?*GradState,
        left_shape: [rawRank(left_tags.len)]usize,
        right_shape: [rawRank(right_tags.len)]usize,
        estimated_work: usize,
        left_value: RawTensor,
        right_value: RawTensor,

        const Self = @This();

        const BranchTask = struct {
            self: *const Self,
            ctx: *ExecContext,
            gy: *const RawTensor,
            out: *?RawTensor,
            err: ?anyerror = null,
        };

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            left_parent: ?*GradState,
            right_parent: ?*GradState,
            left: *const RawTensor,
            right: *const RawTensor,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ left_parent, right_parent },
                .left_shape = rawShapeArray(left_tags, left),
                .right_shape = rawShapeArray(right_tags, right),
                .estimated_work = einsumBackwardWorkEstimate(left_parent, right_parent, left, right),
                .left_value = try left.cloneView(),
                .right_value = undefined,
            };
            errdefer self.left_value.deinit();
            self.right_value = try right.cloneView();
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn estimatedWork(ptr: *const anyopaque) usize {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.estimated_work;
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            const need_left = needs_grad.len > 0 and needs_grad[0];
            const need_right = needs_grad.len > 1 and needs_grad[1];

            if (comptime exec_mod.parallel_dot_backward_branches) {
                if (need_left and need_right) {
                    if (ctx.dotBackwardWorker()) |worker| {
                        var right_task = BranchTask{
                            .self = self,
                            .ctx = ctx,
                            .gy = gy,
                            .out = &out[1],
                        };
                        if (worker.start(runEinsumBackwardRightBranch, &right_task)) {
                            self.backwardLeft(ctx, gy, &out[0]) catch |err| {
                                worker.wait();
                                return err;
                            };
                            worker.wait();
                            if (right_task.err) |err| return err;
                            return;
                        }
                    }
                }
            }

            if (need_left) {
                try self.backwardLeft(ctx, gy, &out[0]);
            }
            if (need_right) {
                try self.backwardRight(ctx, gy, &out[1]);
            }
        }

        fn backwardLeft(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, out: *?RawTensor) !void {
            var grad = try tag_ops.taggedEinsum(out_tags, gy, ctx, right_tags, &self.right_value, left_recover_tags);
            defer grad.deinit();
            out.* = try expandGradientToTags(left_recover_tags, left_tags, ctx, &grad, self.left_shape);
        }

        fn backwardRight(self: *const Self, ctx: *ExecContext, gy: *const RawTensor, out: *?RawTensor) !void {
            var grad = try tag_ops.taggedEinsum(out_tags, gy, ctx, left_tags, &self.left_value, right_recover_tags);
            defer grad.deinit();
            out.* = try expandGradientToTags(right_recover_tags, right_tags, ctx, &grad, self.right_shape);
        }

        fn runEinsumBackwardRightBranch(ptr: *anyopaque) void {
            const task: *BranchTask = @ptrCast(@alignCast(ptr));
            task.self.backwardRight(task.ctx, task.gy, task.out) catch |err| {
                task.err = err;
            };
        }

        fn einsumBackwardWorkEstimate(left_parent: ?*GradState, right_parent: ?*GradState, left: *const RawTensor, right: *const RawTensor) usize {
            var branches: usize = 0;
            if (left_parent != null) branches += 1;
            if (right_parent != null) branches += 1;
            if (branches == 0) return 0;

            var result_elems: usize = 1;
            inline for (out_tags) |tag| {
                result_elems = saturatedMul(result_elems, dimForTag(tag, left, right));
            }
            var dropped_elems: usize = 1;
            inline for (dropped_tags) |tag| {
                dropped_elems = saturatedMul(dropped_elems, dimForTag(tag, left, right));
            }
            return parallel.saturatedMul3(result_elems, dropped_elems, branches);
        }

        fn dimForTag(comptime tag: Tag, left: *const RawTensor, right: *const RawTensor) usize {
            if (comptime tagIndex(left_tags, tag)) |axis| return left.shape.at(axis);
            if (comptime tagIndex(right_tags, tag)) |axis| return right.shape.at(axis);
            unreachable;
        }

        fn saturatedMul(a: usize, b: usize) usize {
            return std.math.mul(usize, a, b) catch std.math.maxInt(usize);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.left_value.deinit();
            self.right_value.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
            .estimated_work = estimatedWork,
        };
    };
}

/// Backward for `dot` against a constant quantized, f16, or bf16 RHS (frozen
/// weights): only the f32 LHS (activation) gradient flows, computed against
/// the RHS widened to f32 at backward time. The widened copy is transient —
/// weight memory stays quantized between steps — which makes fine-tuning
/// through frozen GGUF weights possible without duplicating them in f32
/// permanently.
pub fn ConstRhsDotBackward(
    comptime rhs_dtype: tensor_mod.DType,
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    comptime contract_tag: Tag,
) type {
    // Same delegation as DotBackward: dot is the single-contract-tag einsum
    // (every left tag is recoverable, so no broadcast expansion happens).
    return ConstRhsEinsumBackward(rhs_dtype, left_tags, right_tags, dotResultTags(left_tags, right_tags, contract_tag));
}

/// Backward for `einsum` against a constant quantized, f16, or bf16 RHS:
/// only the f32 LHS gradient flows, computed against the RHS widened to f32
/// at backward time (the widened copy is transient -- weight memory stays
/// narrow between steps). The gradient is itself an einsum, broadcast over
/// any LHS axes the forward summed away.
pub fn ConstRhsEinsumBackward(
    comptime rhs_dtype: tensor_mod.DType,
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    comptime out_tags: anytype,
) type {
    const left_recover_tags = intersectTags(left_tags, tags_mod.unionTags(out_tags, right_tags));

    return struct {
        parents: [1]?*GradState,
        left_shape: [rawRank(left_tags.len)]usize,
        right_value: tensor_mod.TensorOf(rhs_dtype),

        const Self = @This();

        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            left_parent: ?*GradState,
            left: *const RawTensor,
            right: *const tensor_mod.TensorOf(rhs_dtype),
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{left_parent},
                .left_shape = rawShapeArray(left_tags, left),
                .right_value = try right.cloneView(),
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad.len == 0 or !needs_grad[0]) return;

            var right_f32 = if (comptime dtype_mod.isBlockQuantized(rhs_dtype))
                try ctx.dequantizeTensorTyped(rhs_dtype, &self.right_value)
            else
                try ctx.castTyped(rhs_dtype, .f32, &self.right_value);
            defer right_f32.deinit();

            // dL/dleft is itself a contraction (einsum closure); axes the
            // forward summed away come back as a broadcast.
            var grad = try tag_ops.taggedEinsum(out_tags, gy, ctx, right_tags, &right_f32, left_recover_tags);
            defer grad.deinit();
            out[0] = try expandGradientToTags(left_recover_tags, left_tags, ctx, &grad, self.left_shape);
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.right_value.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };
}

/// Backward for `dotTernarySte`: forward ran the flattened activation
/// `x2d [m, k]` against a TQ2_0-encoded snapshot of the latent weight.
/// dx flows through the QUANTIZED weight — the encoded rows are dequantized
/// transiently (scale-correct, mirroring `ConstRhsDotBackward`) and gy is
/// contracted against them. dW is the straight-through estimate: the plain
/// trans_b matmul VJP `gyᵀ @ x` against the latent weight (identity through
/// the quantizer — no clipping or masking).
pub fn TernarySteDotBackward(comptime left_tags: anytype) type {
    return struct {
        parents: [2]?*GradState,
        // Flattened contiguous [m, k] activation view the forward contracted.
        left: RawTensor,
        left_shape: [rawRank(left_tags.len)]usize,
        estimated_work: usize,
        // Encoded weight snapshot; owned (freed by deinit).
        rhs: backend_quant.QuantizedMatmulRhsTQ2_0,

        const Self = @This();

        /// Takes ownership of `rhs` on success; on any failure — including a
        /// node-allocation failure in `core.createNode`, which happens before
        /// this runs — it stays with the caller (who holds the errdefer).
        /// `left_full` only provides the original activation shape; `left2d`
        /// is cloned as a view.
        pub fn init(
            self: *Self,
            allocator: std.mem.Allocator,
            left_parent: ?*GradState,
            right_parent: ?*GradState,
            left_full: *const RawTensor,
            left2d: *const RawTensor,
            rhs: backend_quant.QuantizedMatmulRhsTQ2_0,
        ) !void {
            _ = allocator;
            self.* = .{
                .parents = .{ left_parent, right_parent },
                .left = try left2d.cloneView(),
                .left_shape = rawShapeArray(left_tags, left_full),
                .estimated_work = workEstimate(left_parent, right_parent, left2d, rhs.n),
                .rhs = rhs,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn estimatedWork(ptr: *const anyopaque) usize {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.estimated_work;
        }

        /// DotBackward's accounting adapted to this op's fixed shapes: each
        /// live branch is one [m, n] x [n, k]-shaped dense contraction, so
        /// work = result_elems (m*n) * contract (k) * branches. The dx
        /// branch additionally dequantizes the [n, k] snapshot — one more
        /// n*k pass, dominated by the contractions above — so the dot-shaped
        /// estimate stays representative.
        fn workEstimate(left_parent: ?*GradState, right_parent: ?*GradState, left2d: *const RawTensor, n: usize) usize {
            var branches: usize = 0;
            if (left_parent != null) branches += 1;
            if (right_parent != null) branches += 1;
            if (branches == 0) return 0;

            const m = left2d.shape.at(0);
            const k = left2d.shape.at(1);
            const result_elems = std.math.mul(usize, m, n) catch std.math.maxInt(usize);
            return parallel.saturatedMul3(result_elems, k, branches);
        }

        fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            const m = self.left.shape.at(0);
            const k = self.left.shape.at(1);
            const n = self.rhs.n;

            var gy_ready = try contiguousForRead(ctx, gy);
            defer gy_ready.deinit();
            var gy2d = try gy_ready.reshape(&.{ m, n });
            defer gy2d.deinit();

            if (needs_grad.len > 0 and needs_grad[0]) {
                var right_f32 = try ctx.emptyRankTyped(.f32, 2, .{ n, k });
                defer right_f32.deinit();
                const rows = right_f32.data();
                for (0..n) |row| {
                    try backend_quant.dequantizeRowTQ2_0Into(rows[row * k ..][0..k], self.rhs.columnBlocks(row));
                }
                var dx = try ctx.matmul2D(&gy2d, &right_f32);
                errdefer dx.deinit();
                if (!std.mem.eql(usize, dx.shape.slice(), self.left_shape[0..])) {
                    const reshaped = try dx.reshape(self.left_shape[0..]);
                    dx.deinit();
                    dx = reshaped;
                }
                out[0] = dx;
            }

            if (needs_grad.len > 1 and needs_grad[1]) {
                out[1] = try ctx.matmulTransA(&gy2d, &self.left);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.left.deinit();
            self.rhs.deinit();
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
            .estimated_work = estimatedWork,
        };
    };
}
