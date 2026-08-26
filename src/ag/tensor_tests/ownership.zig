//! Graph ownership without an exec scope: a facade handle may be released
//! at any point before `backward` because every consumer record retains
//! its operands' `GradState`s (ag/core.zig). Pinned here for the
//! hand-written case (an intermediate deinit'ed before backward, a
//! multi-consumer intermediate) and for every library-composed op that
//! releases its own intermediates on return: unscoped and scoped runs
//! must produce the same gradients, under a leak-checking allocator.

const std = @import("std");
const exec_mod = @import("../../exec.zig");
const ag_tensor = @import("../tensor.zig");

const ExecContext = exec_mod.ExecContext;
const Tensor = ag_tensor.Tensor;

/// The DebugAllocator configuration that keeps freed memory mapped and
/// its metadata around, so a use-after-free of a released graph node is
/// reported (double free / invalid free) instead of silently reading
/// recycled memory.
const StrictAllocator = std.heap.DebugAllocator(.{ .safety = true, .never_unmap = true, .retain_metadata = true });

test "unscoped: an intermediate released before backward still receives and forwards its gradient" {
    var gpa = StrictAllocator{};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{.n});
    var x = try T.variableFromSlice(&ctx, .{4}, &.{ 1, -2, 3, -4 });
    defer x.deinit();

    var y = try x.relu(&ctx); // interior node, parents = {x}
    var z = try y.exp(&ctx); // interior node, parents = {y}
    defer z.deinit();
    var loss = try z.sumAll(&ctx); // interior node, parents = {z}
    defer loss.deinit();

    y.deinit(); // the handle goes; z's record keeps y's state alive

    try loss.backward(&ctx); // loss -> z -> y -> x
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    // d/dx sum(exp(relu(x))) = relu'(x) * exp(relu(x)).
    const g = try gx.dataConst();
    try std.testing.expectApproxEqRel(@as(f32, @exp(1.0)), g[0], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), g[1]);
    try std.testing.expectApproxEqRel(@as(f32, @exp(3.0)), g[2], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), g[3]);
}

test "unscoped: a multi-consumer intermediate released before backward accumulates from every consumer" {
    var gpa = StrictAllocator{};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{.n});
    var x = try T.variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    var y = try x.scale(&ctx, 2); // y = 2x, consumed twice below
    var a = try y.mul(&ctx, &y); // a = 4x^2
    defer a.deinit();
    var b = try y.scale(&ctx, 3); // b = 6x
    defer b.deinit();
    y.deinit(); // two records hold y now
    var s = try a.add(&ctx, &b);
    defer s.deinit();
    var loss = try s.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    // d/dx (4x^2 + 6x) = 8x + 6.
    try std.testing.expectEqualSlices(f32, &.{ 14, 22, 30 }, try gx.dataConst());
}

test "unscoped: conv2dRelu with a variable weight differentiates through its released conv intermediate" {
    var gpa = StrictAllocator{};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // input [H=3, W=3, Cin=1], weight [Cout=1, kH=2, kW=2, 1] = x(i,j) - x(i+1,j+1):
    // conv = [2, -3, -5, 3] -> relu keeps positions (0,0) and (1,1) only.
    var x = try Tensor(.{ .h, .w, .cin }).variableFromSlice(&ctx, .{ 3, 3, 1 }, &.{ 5, 1, 2, 1, 3, 4, 2, 6, 0 });
    defer x.deinit();
    var w = try Tensor(.{ .cout, .kh, .kw, .cinpg }).variableFromSlice(&ctx, .{ 1, 2, 2, 1 }, &.{ 1, 0, 0, -1 });
    defer w.deinit();

    var y = try x.conv2dRelu(&ctx, w, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .oh, .ow, .cout });
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 0, 0, 3 }, try y.dataConst());
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 8, 5, 7, 3 }, try gw.dataConst());
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 0, 0, 0, 0, 0, -1 }, try gx.dataConst());
}

// ---------------------------------------------------------------------
// Every library-composed op, unscoped vs scoped.
// ---------------------------------------------------------------------

/// `sum(y * ramp)` with `ramp = 1, 2, 3, ...` over `y`'s elements: a
/// position-dependent upstream gradient, so a wrong scatter shows.
fn weightedSum(ctx: *ExecContext, y: anytype) !Tensor(.{}) {
    const T = @TypeOf(y.*);
    const shape = y.shape();
    var n: usize = 1;
    for (shape) |d| n *= d;
    var ramp: [64]f32 = undefined;
    for (ramp[0..n], 0..) |*r, i| r.* = @floatFromInt(i + 1);
    var w = try T.fromSlice(ctx, shape, ramp[0..n]);
    defer w.deinit();
    var yw = try y.mul(ctx, &w);
    defer yw.deinit();
    return yw.sumAll(ctx);
}

const V = Tensor(.{.d});
const M = Tensor(.{ .h, .w });
const RC = Tensor(.{ .row, .col });
const BS = Tensor(.{ .batch, .seq });

/// One scalar loss per composed op. Every intermediate is released on
/// return (the inference idiom); the op's own intermediates are released
/// inside the op.
const Losses = struct {
    fn reshape(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
        var y = try x.reshape(ctx, .{ .a, .b }, .{ 3, 2 });
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn select(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
        var y = try x.select(ctx, .w, 1);
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn stack(ctx: *ExecContext, x: *const V) !Tensor(.{}) {
        var other = try V.fromSlice(ctx, .{2}, &.{ 5, 6 });
        defer other.deinit();
        var y = try x.stack(ctx, .s, 0, &.{&other});
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn trace(ctx: *ExecContext, x: *const RC) !Tensor(.{}) {
        return x.trace(ctx, .row, .col);
    }

    fn diag(ctx: *ExecContext, x: *const V) !Tensor(.{}) {
        var y = try x.diag(ctx, .{ .row, .col });
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn l2Normalize(ctx: *ExecContext, x: *const V) !Tensor(.{}) {
        var y = try x.l2Normalize(ctx, .d, 1e-6);
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn norm(ctx: *ExecContext, x: *const V) !Tensor(.{}) {
        return x.norm(ctx, .d, .l2);
    }

    fn normAll(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
        return x.normAll(ctx, .l2);
    }

    fn cosineSimilarity(ctx: *ExecContext, x: *const V) !Tensor(.{}) {
        var other = try V.fromSlice(ctx, .{4}, &.{ 1, 1, 0, 0 });
        defer other.deinit();
        return x.cosineSimilarity(ctx, &other, .d, 1e-8);
    }

    fn nllLoss(ctx: *ExecContext, x: *const Tensor(.{ .pos, .class })) !Tensor(.{}) {
        return x.nllLoss(ctx, .class, &.{ 2, 0 }, .mean);
    }

    fn maskedSelect(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
        var mask = try M.fromSlice(ctx, .{ 2, 2 }, &.{ 1, 0, 0, 1 });
        defer mask.deinit();
        var y = try x.maskedSelect(ctx, mask, .m);
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn maskedScatter(ctx: *ExecContext, x: *const V) !Tensor(.{}) {
        var mask = try V.fromSlice(ctx, .{4}, &.{ 1, 0, 1, 0 });
        defer mask.deinit();
        var vals = try Tensor(.{.nz}).fromSlice(ctx, .{2}, &.{ 7, 9 });
        defer vals.deinit();
        var y = try x.maskedScatter(ctx, mask, .nz, &vals);
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn rollBy(ctx: *ExecContext, x: *const BS) !Tensor(.{}) {
        var y = try x.rollBy(ctx, .seq, &.{ 1, -1 });
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn shiftBy(ctx: *ExecContext, x: *const BS) !Tensor(.{}) {
        var y = try x.shiftBy(ctx, .seq, &.{ 1, -1 }, 0.5);
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn constantPad2d(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
        var y = try x.constantPad2d(ctx, .h, .w, 1, 0.5);
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn zeroPad2d(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
        var y = try x.zeroPad2d(ctx, .h, .w, 1);
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn unbindInto(ctx: *ExecContext, x: *const Tensor(.{ .s, .d })) !Tensor(.{}) {
        var parts: [2]V = undefined;
        try x.unbindInto(ctx, .s, &parts);
        defer for (&parts) |*part| part.deinit();
        var first = try weightedSum(ctx, &parts[0]);
        defer first.deinit();
        var second = try parts[1].sumAll(ctx);
        defer second.deinit();
        var second3 = try second.scale(ctx, 3);
        defer second3.deinit();
        return first.add(ctx, &second3);
    }

    fn diagEmbed(ctx: *ExecContext, x: *const Tensor(.{ .b, .d })) !Tensor(.{}) {
        var y = try x.diagEmbed(ctx, .d, .{ .row, .col });
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn slice(ctx: *ExecContext, x: *const RC) !Tensor(.{}) {
        var y = try x.slice(ctx, .{ .row = .{ .start = 1 }, .col = .{ .end = 2 } });
        defer y.deinit();
        return weightedSum(ctx, &y);
    }

    fn einsumMany(ctx: *ExecContext, x: *const Tensor(.{ .s, .i })) !Tensor(.{}) {
        var a = try Tensor(.{ .r, .i }).fromSlice(ctx, .{ 2, 3 }, &.{ 1, 0, -1, 2, 1, 0 });
        defer a.deinit();
        var b = try Tensor(.{ .o, .r }).fromSlice(ctx, .{ 2, 2 }, &.{ 1, 2, -1, 3 });
        defer b.deinit();
        var y = try ag_tensor.einsumMany(ctx, .{ .s, .o }, .{ x, &a, &b });
        defer y.deinit();
        return weightedSum(ctx, &y);
    }
};

/// Run `loss_fn` twice from the same input: under an exec scope (the
/// reference) and unscoped, where every intermediate handle is released
/// before `backward`. The gradients must agree exactly (same kernels,
/// same order).
fn expectUnscopedMatchesScoped(
    ctx: *ExecContext,
    comptime In: type,
    shape: [In.tensor_rank]usize,
    values: []const f32,
    comptime loss_fn: anytype,
) !void {
    var x_scoped = try In.variableFromSlice(ctx, shape, values);
    defer x_scoped.deinit();
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try loss_fn(ctx, &x_scoped);
        try loss.backward(ctx);
    }
    var g_scoped = (try x_scoped.grad(ctx)).?;
    defer g_scoped.deinit();

    var x = try In.variableFromSlice(ctx, shape, values);
    defer x.deinit();
    var loss = try loss_fn(ctx, &x);
    defer loss.deinit();
    try loss.backward(ctx);
    var g = (try x.grad(ctx)).?;
    defer g.deinit();

    try std.testing.expectEqualSlices(f32, try g_scoped.dataConst(), try g.dataConst());
}

test "every composed op differentiates unscoped exactly as it does under an exec scope" {
    var gpa = StrictAllocator{};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const m23 = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const m22 = [_]f32{ 1, 2, 3, 4 };
    const v2 = [_]f32{ 3, -4 };
    const v3 = [_]f32{ 1, 2, 3 };
    const v4 = [_]f32{ 1, 0, 1, 0 };
    const m24 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const m34 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const logp = [_]f32{ -1, -2, -0.5, -0.3, -1.2, -2.3 };

    try expectUnscopedMatchesScoped(&ctx, M, .{ 2, 3 }, &m23, Losses.reshape);
    try expectUnscopedMatchesScoped(&ctx, M, .{ 2, 3 }, &m23, Losses.select);
    try expectUnscopedMatchesScoped(&ctx, V, .{2}, &v2, Losses.stack);
    try expectUnscopedMatchesScoped(&ctx, RC, .{ 2, 2 }, &m22, Losses.trace);
    try expectUnscopedMatchesScoped(&ctx, V, .{3}, &v3, Losses.diag);
    try expectUnscopedMatchesScoped(&ctx, V, .{2}, &v2, Losses.l2Normalize);
    try expectUnscopedMatchesScoped(&ctx, V, .{2}, &v2, Losses.norm);
    try expectUnscopedMatchesScoped(&ctx, M, .{ 2, 2 }, &m22, Losses.normAll);
    try expectUnscopedMatchesScoped(&ctx, V, .{4}, &v4, Losses.cosineSimilarity);
    try expectUnscopedMatchesScoped(&ctx, Tensor(.{ .pos, .class }), .{ 2, 3 }, &logp, Losses.nllLoss);
    try expectUnscopedMatchesScoped(&ctx, M, .{ 2, 2 }, &m22, Losses.maskedSelect);
    try expectUnscopedMatchesScoped(&ctx, V, .{4}, &v4, Losses.maskedScatter);
    try expectUnscopedMatchesScoped(&ctx, BS, .{ 2, 4 }, &m24, Losses.rollBy);
    try expectUnscopedMatchesScoped(&ctx, BS, .{ 2, 4 }, &m24, Losses.shiftBy);
    try expectUnscopedMatchesScoped(&ctx, M, .{ 2, 2 }, &m22, Losses.constantPad2d);
    try expectUnscopedMatchesScoped(&ctx, M, .{ 2, 2 }, &m22, Losses.zeroPad2d);
    try expectUnscopedMatchesScoped(&ctx, Tensor(.{ .s, .d }), .{ 2, 2 }, &m22, Losses.unbindInto);
    try expectUnscopedMatchesScoped(&ctx, Tensor(.{ .b, .d }), .{ 2, 3 }, &m23, Losses.diagEmbed);
    try expectUnscopedMatchesScoped(&ctx, RC, .{ 3, 4 }, &m34, Losses.slice);
    try expectUnscopedMatchesScoped(&ctx, Tensor(.{ .s, .i }), .{ 2, 3 }, &m23, Losses.einsumMany);
}
