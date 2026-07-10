//! Behavioral tests for the autograd scheduling core (`core.zig`): the
//! backward scheduler's handling of wide operand fan-out (beyond the stack
//! scratch capacity) and gradient-accumulation copy-on-write protection of
//! shared view contributions.
const std = @import("std");
const core = @import("core.zig");
const exec_mod = @import("../exec.zig");
const tensor = @import("../tensor.zig");

const Allocator = std.mem.Allocator;
const ExecContext = exec_mod.ExecContext;
const Tensor = tensor.Tensor;

const BackwardFunction = core.BackwardFunction;
const GradState = core.GradState;
const backwardGrad = core.backwardGrad;
const backwardGradOne = core.backwardGradOne;

test "backward scheduler handles more operands than stack scratch capacity" {
    const WideBackward = struct {
        parents: []?*GradState,

        const Self = @This();

        pub fn init(self: *Self, allocator: Allocator, parents: []const ?*GradState) !void {
            const owned_parents = try allocator.dupe(?*GradState, parents);
            self.* = .{ .parents = owned_parents };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents;
        }

        fn backward(
            ptr: *const anyopaque,
            ctx: *ExecContext,
            gy: *const Tensor,
            needs_grad: []const bool,
            out: []?Tensor,
        ) anyerror!void {
            _ = gy;
            const self: *const Self = @ptrCast(@alignCast(ptr));
            try std.testing.expectEqual(self.parents.len, needs_grad.len);
            try std.testing.expectEqual(self.parents.len, out.len);

            for (needs_grad, out, 0..) |need, *slot, i| {
                if (need) {
                    slot.* = try ctx.scalar(@floatFromInt(i + 1));
                }
            }
        }

        fn deinit(ptr: *anyopaque, allocator: Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            allocator.free(self.parents);
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const operand_count = 12;
    var parents: [operand_count]*GradState = undefined;
    var parent_operands: [operand_count]?*GradState = undefined;
    var initialized: usize = 0;
    errdefer {
        for (parents[0..initialized]) |parent| {
            parent.deinit();
        }
    }
    for (0..operand_count) |i| {
        const parent = try GradState.leaf(ctx.allocator);
        parents[i] = parent;
        parent_operands[i] = parent;
        initialized += 1;
    }
    defer {
        for (&parents) |parent| {
            parent.deinit();
        }
    }

    var output_value = try ctx.scalar(0);
    defer output_value.deinit();

    const output = try core.createNode(WideBackward, .{ ctx.allocator, &parent_operands });
    defer output.deinit();

    try backwardGradOne(&ctx, output, &output_value);

    for (&parents, 0..) |parent, i| {
        var grad = (try parent.gradClone(ctx.allocator)).?;
        defer grad.deinit();
        try std.testing.expectEqual(@as(f32, @floatFromInt(i + 1)), grad.item());
    }
}

test "gradient accumulation copy-on-write protects shared view contributions" {
    const DuplicateViewBackward = struct {
        parents: [3]?*GradState,

        const Self = @This();

        pub fn init(self: *Self, allocator: Allocator, a: *GradState, b: *GradState) !void {
            _ = allocator;
            self.* = .{ .parents = .{ a, b, a } };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parents[0..];
        }

        fn backward(
            ptr: *const anyopaque,
            ctx: *ExecContext,
            gy: *const Tensor,
            needs_grad: []const bool,
            out: []?Tensor,
        ) anyerror!void {
            _ = ptr;
            _ = ctx;
            try std.testing.expectEqual(@as(usize, 3), needs_grad.len);
            try std.testing.expectEqual(@as(usize, 3), out.len);

            for (needs_grad, out) |need, *slot| {
                if (need) {
                    slot.* = try gy.cloneView();
                }
            }
        }

        fn deinit(ptr: *anyopaque, allocator: Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const a = try GradState.leaf(ctx.allocator);
    defer a.deinit();
    const b = try GradState.leaf(ctx.allocator);
    defer b.deinit();

    var output_value = try ctx.scalar(0);
    defer output_value.deinit();

    const output = try core.createNode(DuplicateViewBackward, .{ ctx.allocator, a, b });
    defer output.deinit();

    try backwardGradOne(&ctx, output, &output_value);

    var ga = (try a.gradClone(ctx.allocator)).?;
    defer ga.deinit();
    var gb = (try b.gradClone(ctx.allocator)).?;
    defer gb.deinit();
    try std.testing.expectEqual(@as(f32, 2), ga.item());
    try std.testing.expectEqual(@as(f32, 1), gb.item());

    var ga_view = (try a.gradView()).?;
    defer ga_view.deinit();
    var gb_view = (try b.gradView()).?;
    defer gb_view.deinit();
    try std.testing.expect(ga_view.buffer != gb_view.buffer);
}

test "multi-output backward adds a seed when a prior output already touched that grad" {
    const ScaleToParentBackward = struct {
        parent: [1]?*GradState,
        factor: f32,

        const Self = @This();

        pub fn init(self: *Self, allocator: Allocator, parent: *GradState, factor: f32) !void {
            _ = allocator;
            self.* = .{
                .parent = .{parent},
                .factor = factor,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parent[0..];
        }

        fn backward(
            ptr: *const anyopaque,
            ctx: *ExecContext,
            gy: *const Tensor,
            needs_grad: []const bool,
            out: []?Tensor,
        ) anyerror!void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad[0]) {
                out[0] = try ctx.scalar(gy.item() * self.factor);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x = try GradState.leaf(ctx.allocator);
    defer x.deinit();

    const z = try core.createNode(ScaleToParentBackward, .{ ctx.allocator, x, 3 });
    defer z.deinit();

    const y = try core.createNode(ScaleToParentBackward, .{ ctx.allocator, z, 2 });
    defer y.deinit();

    var y_value = try ctx.scalar(0);
    defer y_value.deinit();
    var z_value = try ctx.scalar(0);
    defer z_value.deinit();

    try backwardGrad(&ctx, &.{ y, z }, &.{ &y_value, &z_value });

    var gz = (try z.gradClone(ctx.allocator)).?;
    defer gz.deinit();
    try std.testing.expectEqual(@as(f32, 3), gz.item());

    var gx = (try x.gradClone(ctx.allocator)).?;
    defer gx.deinit();
    try std.testing.expectEqual(@as(f32, 9), gx.item());
}

test "failed output seeding leaves the graph re-runnable" {
    const ScaleToParentBackward = struct {
        parent: [1]?*GradState,
        factor: f32,

        const Self = @This();

        pub fn init(self: *Self, allocator: Allocator, parent: *GradState, factor: f32) !void {
            _ = allocator;
            self.* = .{
                .parent = .{parent},
                .factor = factor,
            };
        }

        fn operands(ptr: *const anyopaque) []const ?*GradState {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            return self.parent[0..];
        }

        fn backward(
            ptr: *const anyopaque,
            ctx: *ExecContext,
            gy: *const Tensor,
            needs_grad: []const bool,
            out: []?Tensor,
        ) anyerror!void {
            const self: *const Self = @ptrCast(@alignCast(ptr));
            if (needs_grad[0]) {
                out[0] = try ctx.scalar(gy.item() * self.factor);
            }
        }

        fn deinit(ptr: *anyopaque, allocator: Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            core.destroyNode(Self, allocator, self);
        }

        pub const vtable = BackwardFunction.VTable{
            .operands = operands,
            .backward = backward,
            .deinit = deinit,
        };
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x = try GradState.leaf(ctx.allocator);
    defer x.deinit();

    const y = try core.createNode(ScaleToParentBackward, .{ ctx.allocator, x, 2 });
    defer y.deinit();

    const z = try core.createNode(ScaleToParentBackward, .{ ctx.allocator, x, 3 });
    defer z.deinit();

    var y_value = try ctx.scalar(0);
    defer y_value.deinit();
    var z_value = try ctx.fromSlice(&.{2}, &.{ 0, 0 });
    defer z_value.deinit();

    // z is non-scalar and unseeded: the failure must surface before any
    // pending counter is installed or any node runs.
    try std.testing.expectError(
        core.AgError.MissingOutputGradient,
        backwardGrad(&ctx, &.{ y, z }, &.{ &y_value, &z_value }),
    );
    try std.testing.expect((try x.gradClone(ctx.allocator)) == null);

    // Seeding z explicitly repairs the SAME graph: the retry must deliver
    // both contributions to x (stale counters from the failed pass would
    // silently skip z's backward).
    z.setGrad(try ctx.scalar(1));
    try backwardGrad(&ctx, &.{ y, z }, &.{ &y_value, &z_value });

    var gx = (try x.gradClone(ctx.allocator)).?;
    defer gx.deinit();
    try std.testing.expectEqual(@as(f32, 5), gx.item());

    // The successful retry consumed the graph: a third pass fails loudly
    // instead of silently compounding interior gradients — for a single
    // consumed output within the batch too.
    try std.testing.expectError(
        core.AgError.BackwardAlreadyRun,
        backwardGrad(&ctx, &.{ y, z }, &.{ &y_value, &z_value }),
    );
    try std.testing.expectError(
        core.AgError.BackwardAlreadyRun,
        backwardGradOne(&ctx, y, &y_value),
    );
}
