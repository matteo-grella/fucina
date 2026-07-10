//! Behavioral tests for the elemental scalar-op adapter (`elemental.zig`)
//! through the public facade methods `elementalUnary`/`elementalBinary`.
const std = @import("std");
const exec_mod = @import("../exec.zig");
const ag_tensor = @import("tensor.zig");
const gradcheck_mod = @import("gradcheck.zig");

const ExecContext = exec_mod.ExecContext;
const Tensor = ag_tensor.Tensor;

const Square = struct {
    pub fn forward(x: f32, extra: void) f32 {
        _ = extra;
        return x * x;
    }

    pub fn backward(x: f32, y: f32, grad_y: f32, extra: void) f32 {
        _ = y;
        _ = extra;
        return 2.0 * x * grad_y;
    }
};

const PowScale = struct {
    scale: f32,

    pub fn forward(x: f32, extra: @This()) f32 {
        return extra.scale * x * x * x;
    }

    pub fn backward(x: f32, y: f32, grad_y: f32, extra: @This()) f32 {
        _ = y;
        return 3.0 * extra.scale * x * x * grad_y;
    }
};

const AddLike = struct {
    pub fn forward(a: f32, b: f32, extra: void) f32 {
        _ = extra;
        return a + b;
    }

    pub fn backwardA(a: f32, b: f32, y: f32, grad_y: f32, extra: void) f32 {
        _ = a;
        _ = b;
        _ = y;
        _ = extra;
        return grad_y;
    }

    pub fn backwardB(a: f32, b: f32, y: f32, grad_y: f32, extra: void) f32 {
        _ = a;
        _ = b;
        _ = y;
        _ = extra;
        return grad_y;
    }
};

const SmoothMax = struct {
    // max-like blend: y = a·s + b·(1-s), s = sigmoid(k(a-b)); smooth in both.
    k: f32,

    fn sig(z: f32) f32 {
        return 1.0 / (1.0 + @exp(-z));
    }

    pub fn forward(a: f32, b: f32, extra: @This()) f32 {
        const s = sig(extra.k * (a - b));
        return a * s + b * (1 - s);
    }

    pub fn backwardA(a: f32, b: f32, y: f32, grad_y: f32, extra: @This()) f32 {
        _ = y;
        const s = sig(extra.k * (a - b));
        const ds = extra.k * s * (1 - s);
        return grad_y * (s + (a - b) * ds);
    }

    pub fn backwardB(a: f32, b: f32, y: f32, grad_y: f32, extra: @This()) f32 {
        _ = y;
        const s = sig(extra.k * (a - b));
        const ds = extra.k * s * (1 - s);
        return grad_y * ((1 - s) - (a - b) * ds);
    }
};

test "elementalUnary computes forward values and propagated gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, -2, 3 });
    defer x.deinit();

    var y = try x.elementalUnary(&ctx, Square, {});
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 9 }, try y.dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, -4, 6 }, try gx.dataConst());
}

test "elementalUnary passes extra by value and stays no-grad for constants" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var c = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 2, -1 });
    defer c.deinit();
    var y = try c.elementalUnary(&ctx, PowScale, PowScale{ .scale = 0.5 });
    defer y.deinit();
    try std.testing.expect(!y.requiresGrad());
    try std.testing.expectEqualSlices(f32, &.{ 4, -0.5 }, try y.dataConst());
}

test "elementalUnary accepts strided views and routes gradients to the source" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{ .row, .col }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var t = try x.permuteTo(&ctx, .{ .col, .row });
    defer t.deinit();

    var y = try t.elementalUnary(&ctx, PowScale, PowScale{ .scale = 1.0 });
    defer y.deinit();
    // Transposed logical order: {1, 4, 2, 5, 3, 6} cubed.
    try std.testing.expectEqualSlices(f32, &.{ 1, 64, 8, 125, 27, 216 }, try y.dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    // dL/dx = 3x^2 in x's own (row-major) layout.
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 12, 27, 48, 75, 108 }, try gx.dataConst());
}

test "elementalBinary broadcasts like add and reduces the broadcast gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer b.deinit();

    var y = try x.elementalBinary(&ctx, &b, AddLike, {});
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, y.asRawTensor().shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, try y.dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1 }, try gx.dataConst());

    // The broadcast operand's gradient sum-reduces over .batch, like `add`.
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2 }, try gb.dataConst());
}

test "elementalBinary needs_grad prunes constant operands" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer x.deinit();
    var c = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 5, 5 });
    defer c.deinit();

    var y = try x.elementalBinary(&ctx, &c, AddLike, {});
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1 }, try gx.dataConst());
}

fn squareGcLoss(ctx: *ExecContext, x: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try x.elementalUnary(ctx, Square, {});
    defer y.deinit();
    return y.sumAll(ctx);
}

test "gradcheck: elementalUnary VJP" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 0.5, -1.25, 2.0, 0.75 });
    defer x.deinit();

    const result = try gradcheck_mod.gradcheck(&ctx, squareGcLoss, .{&x}, .{});
    try std.testing.expectEqual(@as(usize, 4), result.checked);
    try std.testing.expect(result.max_abs_error <= 2e-3);
}

fn smoothMaxGcLoss(ctx: *ExecContext, a: *const Tensor(.{ .batch, .d }), b: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try a.elementalBinary(ctx, b, SmoothMax, SmoothMax{ .k = 2.0 });
    defer y.deinit();
    return y.sumAll(ctx);
}

test "gradcheck: elementalBinary broadcast VJP with extra" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -0.75, 1.5, 0.25 });
    defer a.deinit();
    var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 0.1, 0.9 });
    defer b.deinit();

    const result = try gradcheck_mod.gradcheck(&ctx, smoothMaxGcLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 6), result.checked);
    try std.testing.expect(result.max_abs_error <= 2e-3);
}
