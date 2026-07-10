//! Behavioral tests for the public custom VJP adapter.
const std = @import("std");
const exec_mod = @import("../exec.zig");
const tensor_mod = @import("../tensor.zig");
const custom = @import("custom.zig");
const ag_tensor = @import("tensor.zig");

const ExecContext = exec_mod.ExecContext;
const RawTensor = tensor_mod.Tensor;
const Tensor = ag_tensor.Tensor;

const Scale = struct {
    value: f32,
};

const NoExtra = struct {};

const ScaledMul = struct {
    pub const Output = Tensor(.{.d});

    pub fn forward(ctx: *ExecContext, extra: Scale, inputs: []const *const RawTensor) !RawTensor {
        var product = try ctx.mulRank(1, inputs[0], inputs[1]);
        defer product.deinit();
        return ctx.scale(&product, extra.value);
    }

    pub fn backward(
        ctx: *ExecContext,
        extra: Scale,
        inputs: []const *const RawTensor,
        output: *const RawTensor,
        gy: *const RawTensor,
        needs_grad: []const bool,
        out: []?RawTensor,
    ) !void {
        _ = output;
        if (needs_grad[0]) {
            var scaled_rhs = try ctx.scale(inputs[1], extra.value);
            defer scaled_rhs.deinit();
            out[0] = try ctx.mulRank(1, gy, &scaled_rhs);
        }
        if (needs_grad[1]) {
            var scaled_lhs = try ctx.scale(inputs[0], extra.value);
            defer scaled_lhs.deinit();
            out[1] = try ctx.mulRank(1, gy, &scaled_lhs);
        }
    }
};

const BadShapeGradient = struct {
    pub const Output = Tensor(.{.d});

    pub fn forward(ctx: *ExecContext, extra: NoExtra, inputs: []const *const RawTensor) !RawTensor {
        _ = extra;
        return ctx.materialize(inputs[0]);
    }

    pub fn backward(
        ctx: *ExecContext,
        extra: NoExtra,
        inputs: []const *const RawTensor,
        output: *const RawTensor,
        gy: *const RawTensor,
        needs_grad: []const bool,
        out: []?RawTensor,
    ) !void {
        _ = extra;
        _ = inputs;
        _ = output;
        _ = gy;
        if (needs_grad[0]) {
            out[0] = try ctx.scalar(1);
        }
    }
};

test "custom VJP over public tagged tensors computes gradients for all variable inputs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 2, -3, 4 });
    defer a.deinit();
    var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 5, 7, -11 });
    defer b.deinit();

    var y = try custom.customVjp(&ctx, ScaledMul, Scale{ .value = 0.5 }, .{ &a, &b });
    defer y.deinit();
    try std.testing.expect(y.requiresGrad());
    try std.testing.expectEqualSlices(f32, &.{ 5, -10.5, -22 }, try y.dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2.5, 3.5, -5.5 }, try ga.dataConst());

    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, -1.5, 2 }, try gb.dataConst());
}

test "custom VJP needs_grad skips constant inputs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 2, -3, 4 });
    defer a.deinit();
    var b = try Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 5, 7, -11 });
    defer b.deinit();

    var y = try custom.customVjp(&ctx, ScaledMul, Scale{ .value = 0.25 }, .{ &a, &b });
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1.25, 1.75, -2.75 }, try ga.dataConst());

    try std.testing.expect(!b.requiresGrad());
    try std.testing.expect((try b.grad(&ctx)) == null);
}

test "custom VJP returns no-grad output when all inputs are constants" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer a.deinit();
    var b = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 5, 6 });
    defer b.deinit();

    var y = try custom.customVjp(&ctx, ScaledMul, Scale{ .value = 2 }, .{ &a, &b });
    defer y.deinit();

    try std.testing.expect(!y.requiresGrad());
    try std.testing.expectEqualSlices(f32, &.{ 30, 48 }, try y.dataConst());
}

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 2, -3, 4 });
    defer a.deinit();
    var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 5, 7, -11 });
    defer b.deinit();

    var y = try custom.customVjp(&ctx, ScaledMul, Scale{ .value = 0.5 }, .{ &a, &b });
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
}

test "custom VJP releases exactly once under induced allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}

test "custom VJP rejects a backward tensor with the wrong input shape" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    var y = try custom.customVjp(&ctx, BadShapeGradient, NoExtra{}, .{&x});
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try std.testing.expectError(tensor_mod.TensorError.ShapeMismatch, loss.backward(&ctx));
}
