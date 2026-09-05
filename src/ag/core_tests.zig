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

const GradState = core.GradState;
const backwardGrad = core.backwardGrad;
const backwardGradOne = core.backwardGradOne;

test "backward scheduler handles more operands than stack scratch capacity" {
    const WideBackward = struct {
        parents: []?*GradState,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            _ = gy;
            try std.testing.expectEqual(self.parents.len, out.len);
            for (out, 0..) |*slot, i| {
                if (core.needs(self, i)) slot.* = try ctx.scalar(.f32, @floatFromInt(i + 1));
            }
        }

        pub fn deinitFields(self: *Self, allocator: Allocator) void {
            allocator.free(self.parents);
        }

        pub const vtable = core.recordVTable(Self);
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
            parent.release();
        }
    }
    for (0..operand_count) |i| {
        const parent = try GradState.leaf(ctx.allocator());
        parents[i] = parent;
        parent_operands[i] = parent;
        initialized += 1;
    }
    defer {
        for (&parents) |parent| {
            parent.release();
        }
    }

    var output_value = try ctx.scalar(.f32, 0);
    defer output_value.deinit();

    const owned_parents = try ctx.allocator().dupe(?*GradState, &parent_operands);
    errdefer ctx.allocator().free(owned_parents);
    const output = try core.createNode(ctx.allocator(), WideBackward{ .parents = owned_parents });
    defer output.release();

    try backwardGradOne(&ctx, output, &output_value);

    for (&parents, 0..) |parent, i| {
        var grad = (try parent.gradClone(ctx.allocator())).?;
        defer grad.deinit();
        try std.testing.expectEqual(@as(f32, @floatFromInt(i + 1)), grad.item());
    }
}

test "gradient accumulation copy-on-write protects shared view contributions" {
    const DuplicateViewBackward = struct {
        parents: [3]?*GradState,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            _ = ctx;
            try std.testing.expectEqual(@as(usize, 3), out.len);
            for (out, 0..) |*slot, i| {
                if (core.needs(self, i)) slot.* = try gy.cloneView();
            }
        }

        pub const vtable = core.recordVTable(Self);
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const a = try GradState.leaf(ctx.allocator());
    defer a.release();
    const b = try GradState.leaf(ctx.allocator());
    defer b.release();

    var output_value = try ctx.scalar(.f32, 0);
    defer output_value.deinit();

    const output = try core.createNode(ctx.allocator(), DuplicateViewBackward{ .parents = .{ a, b, a } });
    defer output.release();

    try backwardGradOne(&ctx, output, &output_value);

    var ga = (try a.gradClone(ctx.allocator())).?;
    defer ga.deinit();
    var gb = (try b.gradClone(ctx.allocator())).?;
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
        parents: [1]?*GradState,
        factor: f32,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            if (core.needs(self, 0)) out[0] = try ctx.scalar(.f32, gy.item() * self.factor);
        }

        pub const vtable = core.recordVTable(Self);
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x = try GradState.leaf(ctx.allocator());
    defer x.release();

    const z = try core.createNode(ctx.allocator(), ScaleToParentBackward{ .parents = .{x}, .factor = 3 });
    defer z.release();

    const y = try core.createNode(ctx.allocator(), ScaleToParentBackward{ .parents = .{z}, .factor = 2 });
    defer y.release();

    var y_value = try ctx.scalar(.f32, 0);
    defer y_value.deinit();
    var z_value = try ctx.scalar(.f32, 0);
    defer z_value.deinit();

    try backwardGrad(&ctx, &.{ y, z }, &.{ &y_value, &z_value });

    var gz = (try z.gradClone(ctx.allocator())).?;
    defer gz.deinit();
    try std.testing.expectEqual(@as(f32, 3), gz.item());

    var gx = (try x.gradClone(ctx.allocator())).?;
    defer gx.deinit();
    try std.testing.expectEqual(@as(f32, 9), gx.item());
}

test "failed output seeding leaves the graph re-runnable" {
    const ScaleToParentBackward = struct {
        parents: [1]?*GradState,
        factor: f32,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            if (core.needs(self, 0)) out[0] = try ctx.scalar(.f32, gy.item() * self.factor);
        }

        pub const vtable = core.recordVTable(Self);
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x = try GradState.leaf(ctx.allocator());
    defer x.release();

    const y = try core.createNode(ctx.allocator(), ScaleToParentBackward{ .parents = .{x}, .factor = 2 });
    defer y.release();

    const z = try core.createNode(ctx.allocator(), ScaleToParentBackward{ .parents = .{x}, .factor = 3 });
    defer z.release();

    var y_value = try ctx.scalar(.f32, 0);
    defer y_value.deinit();
    var z_value = try ctx.fromSlice(.f32, &.{2}, &.{ 0, 0 });
    defer z_value.deinit();

    // z is non-scalar and unseeded: the failure must surface before any
    // pending counter is installed or any node runs.
    try std.testing.expectError(
        core.AgError.MissingOutputGradient,
        backwardGrad(&ctx, &.{ y, z }, &.{ &y_value, &z_value }),
    );
    try std.testing.expect((try x.gradClone(ctx.allocator())) == null);

    // Seeding z explicitly repairs the SAME graph: the retry must deliver
    // both contributions to x (stale counters from the failed pass would
    // silently skip z's backward).
    z.setGrad(try ctx.scalar(.f32, 1));
    try backwardGrad(&ctx, &.{ y, z }, &.{ &y_value, &z_value });

    var gx = (try x.gradClone(ctx.allocator())).?;
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

test "backward rejects a consumed output reached as an interior node" {
    // z = 2x, backward(z); then y = 3z, backward(y). z's retained result
    // gradient (1) would compound with y's contribution (3) and propagate
    // 2·4 = 8 into x on top of the 2 already there (10, not the 8 a fresh
    // graph would give). The second pass must fail before any gradient
    // moves and leave the scheduling state clean.
    const ScaleToParentBackward = struct {
        parents: [1]?*GradState,
        factor: f32,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            if (core.needs(self, 0)) out[0] = try ctx.scalar(.f32, gy.item() * self.factor);
        }

        pub const vtable = core.recordVTable(Self);
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x = try GradState.leaf(ctx.allocator());
    defer x.release();
    const z = try core.createNode(ctx.allocator(), ScaleToParentBackward{ .parents = .{x}, .factor = 2 });
    defer z.release();
    var z_value = try ctx.scalar(.f32, 0);
    defer z_value.deinit();

    try backwardGradOne(&ctx, z, &z_value);
    var gx = (try x.gradClone(ctx.allocator())).?;
    defer gx.deinit();
    try std.testing.expectEqual(@as(f32, 2), gx.item());

    const y = try core.createNode(ctx.allocator(), ScaleToParentBackward{ .parents = .{z}, .factor = 3 });
    defer y.release();
    var y_value = try ctx.scalar(.f32, 0);
    defer y_value.deinit();

    try std.testing.expectError(core.AgError.BackwardAlreadyRun, backwardGradOne(&ctx, y, &y_value));

    // Untouched: x's gradient, z's result gradient, and the scheduling
    // state of every node the preparation reached.
    var gx_after = (try x.gradClone(ctx.allocator())).?;
    defer gx_after.deinit();
    try std.testing.expectEqual(@as(f32, 2), gx_after.item());
    var gz = (try z.gradClone(ctx.allocator())).?;
    defer gz.deinit();
    try std.testing.expectEqual(@as(f32, 1), gz.item());
    try std.testing.expect((try y.gradClone(ctx.allocator())) == null);
    for ([_]*GradState{ x, z, y }) |state| {
        try std.testing.expectEqual(@as(u32, 0), state.pending_grads.load(.acquire));
        try std.testing.expectEqual(@as(u8, 0), state.state.load(.acquire)); // .idle
    }
    try std.testing.expect(!y.backward_done);
    try std.testing.expect(!y.pass_output);
}

test "a node reached without a gradient releases its whole subgraph" {
    // top -> mid -> deep -> x. mid's VJP delivers nothing on the first
    // pass, so deep is scheduled with no gradient: it must still release
    // deep's operand counters (x), or the retry over the repaired graph
    // stops at the stranded state and reports success with x untouched.
    const ScaleToParentBackward = struct {
        parents: [1]?*GradState,
        factor: f32,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            if (core.needs(self, 0)) out[0] = try ctx.scalar(.f32, gy.item() * self.factor);
        }

        pub const vtable = core.recordVTable(Self);
    };
    const SwitchableBackward = struct {
        parents: [1]?*GradState,
        deliver: bool,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            if (self.deliver and core.needs(self, 0)) out[0] = try ctx.scalar(.f32, gy.item());
        }

        pub const vtable = core.recordVTable(Self);
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x = try GradState.leaf(ctx.allocator());
    defer x.release();
    const deep = try core.createNode(ctx.allocator(), ScaleToParentBackward{ .parents = .{x}, .factor = 2 });
    defer deep.release();
    const mid = try core.createNode(ctx.allocator(), SwitchableBackward{ .parents = .{deep}, .deliver = false });
    defer mid.release();
    const top = try core.createNode(ctx.allocator(), ScaleToParentBackward{ .parents = .{mid}, .factor = 5 });
    defer top.release();
    var top_value = try ctx.scalar(.f32, 0);
    defer top_value.deinit();

    try std.testing.expectError(core.AgError.MissingBackwardGradient, backwardGradOne(&ctx, top, &top_value));

    // Every reachable state is back to idle with a zero counter, no
    // non-leaf holds a gradient (the failed pass dropped top's seed), and
    // the graph is unconsumed.
    for ([_]*GradState{ x, deep, mid, top }) |state| {
        try std.testing.expectEqual(@as(u32, 0), state.pending_grads.load(.acquire));
        try std.testing.expectEqual(@as(u8, 0), state.state.load(.acquire)); // .idle
    }
    for ([_]*GradState{ deep, mid, top }) |state| {
        try std.testing.expect((try state.gradClone(ctx.allocator())) == null);
    }
    try std.testing.expect((try x.gradClone(ctx.allocator())) == null);
    try std.testing.expect(!top.backward_done);
    try std.testing.expect(!top.pass_output);

    // Repair the VJP and retry over the SAME graph: the gradient reaches x.
    const mid_record: *SwitchableBackward = @ptrCast(@alignCast(mid.grad_fn.?.ptr));
    mid_record.deliver = true;
    try backwardGradOne(&ctx, top, &top_value);
    var gx = (try x.gradClone(ctx.allocator())).?;
    defer gx.deinit();
    try std.testing.expectEqual(@as(f32, 10), gx.item());
}

test "backward and teardown of a deep chain use a bounded stack" {
    // 50k nodes in one chain: preparation, execution and the release
    // cascade all walk explicit worklists, so graph depth is not a
    // call-stack resource (the recursive engine overflowed near 10k).
    const PassThrough = struct {
        parents: [1]?*GradState,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            if (core.needs(self, 0)) out[0] = try ctx.scalar(.f32, gy.item());
        }

        pub const vtable = core.recordVTable(Self);
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const depth = 50_000;
    const x = try GradState.leaf(ctx.allocator());
    defer x.release();
    var top: *GradState = x.retain();
    for (0..depth) |_| {
        const next = try core.createNode(ctx.allocator(), PassThrough{ .parents = .{top} });
        top.release();
        top = next;
    }
    var top_value = try ctx.scalar(.f32, 0);
    defer top_value.deinit();

    try backwardGradOne(&ctx, top, &top_value);
    var gx = (try x.gradClone(ctx.allocator())).?;
    defer gx.deinit();
    try std.testing.expectEqual(@as(f32, 1), gx.item());
    // The last handle frees the whole chain: the cascade, not recursion.
    top.release();
}

test "a throwing interior VJP leaves no gradient behind for the retry" {
    // top -> mid -> x, mid's VJP fails on the first pass. mid received its
    // gradient before failing; the retry must not accumulate the fresh
    // contribution onto that stale value (x would receive 5·(5+5) = 50
    // instead of 5·5 = 25).
    const ScaleToParentBackward = struct {
        parents: [1]?*GradState,
        factor: f32,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            if (core.needs(self, 0)) out[0] = try ctx.scalar(.f32, gy.item() * self.factor);
        }

        pub const vtable = core.recordVTable(Self);
    };
    const ThrowOnceBackward = struct {
        parents: [1]?*GradState,
        throws: bool,

        const Self = @This();
        const Failure = error{VjpFailed};

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            if (self.throws) return Failure.VjpFailed;
            if (core.needs(self, 0)) out[0] = try ctx.scalar(.f32, gy.item());
        }

        pub const vtable = core.recordVTable(Self);
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x = try GradState.leaf(ctx.allocator());
    defer x.release();
    const mid = try core.createNode(ctx.allocator(), ThrowOnceBackward{ .parents = .{x}, .throws = true });
    defer mid.release();
    const top = try core.createNode(ctx.allocator(), ScaleToParentBackward{ .parents = .{mid}, .factor = 5 });
    defer top.release();
    var top_value = try ctx.scalar(.f32, 0);
    defer top_value.deinit();

    try std.testing.expectError(ThrowOnceBackward.Failure.VjpFailed, backwardGradOne(&ctx, top, &top_value));
    for ([_]*GradState{ x, mid, top }) |state| {
        try std.testing.expectEqual(@as(u32, 0), state.pending_grads.load(.acquire));
        try std.testing.expectEqual(@as(u8, 0), state.state.load(.acquire)); // .idle
    }
    // No non-leaf keeps a gradient from the failed pass: not the output,
    // and not the node whose VJP threw after receiving its gradient.
    try std.testing.expect((try mid.gradClone(ctx.allocator())) == null);
    try std.testing.expect((try top.gradClone(ctx.allocator())) == null);
    try std.testing.expect((try x.gradClone(ctx.allocator())) == null);

    const mid_record: *ThrowOnceBackward = @ptrCast(@alignCast(mid.grad_fn.?.ptr));
    mid_record.throws = false;
    try backwardGradOne(&ctx, top, &top_value);
    var gx = (try x.gradClone(ctx.allocator())).?;
    defer gx.deinit();
    try std.testing.expectEqual(@as(f32, 5), gx.item());
}

test "a failing VJP that consumed its saved state marks the graph consumed" {
    // A record whose body consumes its saved tensors in place calls
    // `consumeRecord` before its first fallible step. When that step
    // fails, the pass fails as usual, but the graph must not be retryable:
    // the retry fails at the preflight, before any gradient moves.
    const ConsumingBackward = struct {
        parents: [1]?*GradState,

        const Self = @This();
        const Failure = error{ConsumedThenFailed};

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            _ = ctx;
            _ = gy;
            _ = out;
            core.consumeRecord(self);
            return Failure.ConsumedThenFailed;
        }

        pub const vtable = core.recordVTable(Self);
    };
    const ScaleToParentBackward = struct {
        parents: [1]?*GradState,
        factor: f32,

        const Self = @This();

        pub fn vjp(self: *Self, ctx: *ExecContext, gy: *const Tensor, out: []?Tensor) !void {
            if (core.needs(self, 0)) out[0] = try ctx.scalar(.f32, gy.item() * self.factor);
        }

        pub const vtable = core.recordVTable(Self);
    };

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x = try GradState.leaf(ctx.allocator());
    defer x.release();
    const consuming = try core.createNode(ctx.allocator(), ConsumingBackward{ .parents = .{x} });
    defer consuming.release();
    const top = try core.createNode(ctx.allocator(), ScaleToParentBackward{ .parents = .{consuming}, .factor = 3 });
    defer top.release();
    var top_value = try ctx.scalar(.f32, 0);
    defer top_value.deinit();

    try std.testing.expectError(ConsumingBackward.Failure.ConsumedThenFailed, backwardGradOne(&ctx, top, &top_value));
    for ([_]*GradState{ x, consuming, top }) |state| {
        try std.testing.expectEqual(@as(u32, 0), state.pending_grads.load(.acquire));
        try std.testing.expectEqual(@as(u8, 0), state.state.load(.acquire)); // .idle
    }
    try std.testing.expect(consuming.backward_done);
    try std.testing.expect(!top.backward_done);

    try std.testing.expectError(core.AgError.BackwardAlreadyRun, backwardGradOne(&ctx, top, &top_value));
    try std.testing.expect((try x.gradClone(ctx.allocator())) == null);
    try std.testing.expect((try top.gradClone(ctx.allocator())) == null);
}
