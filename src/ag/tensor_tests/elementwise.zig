//! Pointwise math on the public Tensor: unary and binary families,
//! activations and gated ops, clamps, dropout, scalar convenience ops, and
//! their gradients.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const ag_tensor = @import("../tensor.zig");

const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tensor = ag_tensor.Tensor;
const RawTensor = @import("../../tensor.zig").Tensor;

const util = @import("util.zig");
const expectCloseSlices = util.expectCloseSlices;

test "public situ matches the composed soft-clamp/gate ops in values and gradients" {
    // situ(up, gate) = 25·tanh(up/25) · 4·tanh(gate/4)·sigmoid(gate).
    // The reference composes the same math from scale/tanh/sigmoid/mul, so
    // both the fused forward (vector bodies) and the fused backward are
    // checked against independently differentiated ops.
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const n = 103; // odd length: vector body + scalar tail both covered
    const T = Tensor(.{.d});
    const up_values = try allocator.alloc(f32, n);
    defer allocator.free(up_values);
    const gate_values = try allocator.alloc(f32, n);
    defer allocator.free(gate_values);
    for (up_values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 41)) - 20)) * 1.7; // spans past the ±25 clamp knee
    for (gate_values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 29)) - 14)) * 0.9;

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);

    var up_f = try T.variableFromSlice(&ctx, .{n}, up_values);
    defer up_f.deinit();
    var gate_f = try T.variableFromSlice(&ctx, .{n}, gate_values);
    defer gate_f.deinit();
    var fused = try up_f.situ(&ctx, &gate_f);
    var fused_loss = try fused.sumAll(&ctx);
    try fused_loss.backward(&ctx);

    var up_r = try T.variableFromSlice(&ctx, .{n}, up_values);
    defer up_r.deinit();
    var gate_r = try T.variableFromSlice(&ctx, .{n}, gate_values);
    defer gate_r.deinit();
    var up_soft = try (try (try up_r.scale(&ctx, 0.04)).unary(&ctx, .tanh)).scale(&ctx, 25.0);
    var gate_tanh = try (try (try gate_r.scale(&ctx, 0.25)).unary(&ctx, .tanh)).scale(&ctx, 4.0);
    var gate_sig = try gate_r.unary(&ctx, .sigmoid);
    var gate_act = try gate_tanh.mul(&ctx, &gate_sig);
    var expected = try up_soft.mul(&ctx, &gate_act);
    var expected_loss = try expected.sumAll(&ctx);
    try expected_loss.backward(&ctx);

    for (expected.asRawTensor().dataConst(), fused.asRawTensor().dataConst()) |e, actual| {
        try std.testing.expectApproxEqAbs(e, actual, 1e-4);
    }
    inline for (.{ .{ &up_r, &up_f }, .{ &gate_r, &gate_f } }) |pair| {
        var g_ref = (try pair[0].grad(&ctx)).?;
        defer g_ref.deinit();
        var g_fused = (try pair[1].grad(&ctx)).?;
        defer g_fused.deinit();
        for (g_ref.asRawTensor().dataConst(), g_fused.asRawTensor().dataConst()) |e, actual| {
            try std.testing.expectApproxEqAbs(e, actual, 1e-4);
        }
    }
}

test "tagged public tensor leakyRelu forward and backward" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variable(
        &ctx,
        try ctx.fromSlice(.f32, &.{3}, &.{ -2, 0, 3 }),
    );
    defer x.deinit();

    var y = try x.leakyRelu(&ctx, 0.25);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -0.5, 0, 3 }, y.asRawTensor().dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.25, 1 }, gx.asRawTensor().dataConst());
}

fn fastTanhDerivativeForTest(value: f32) f32 {
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

test "tagged public tensor fastTanh forward and backward" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variable(
        &ctx,
        try ctx.fromSlice(.f32, &.{4}, &.{ -2, -0.25, 0, 3 }),
    );
    defer x.deinit();

    var y = try x.fastTanh(&ctx);
    defer y.deinit();
    for (x.asRawTensor().dataConst(), y.asRawTensor().dataConst()) |value, actual| {
        try std.testing.expectApproxEqAbs(backend_mod.ops.fastTanhScalar(value), actual, 1e-6);
    }

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    for (x.asRawTensor().dataConst(), gx.asRawTensor().dataConst()) |value, actual| {
        try std.testing.expectApproxEqAbs(fastTanhDerivativeForTest(value), actual, 1e-5);
    }
}

test "tagged autograd differentiates div mean and scalar unary math" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 1, 4 });
    defer x.deinit();
    var denom = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 2, 8 });
    defer denom.deinit();

    var q = try x.div(&ctx, &denom);
    defer q.deinit();
    var exp_x = try x.exp(&ctx);
    defer exp_x.deinit();
    var sqrt_x = try x.sqrt(&ctx);
    defer sqrt_x.deinit();
    var rsqrt_x = try x.rsqrt(&ctx);
    defer rsqrt_x.deinit();
    var sigmoid_x = try x.sigmoid(&ctx);
    defer sigmoid_x.deinit();
    var silu_x = try x.silu(&ctx);
    defer silu_x.deinit();
    var log_x = try x.log(&ctx);
    defer log_x.deinit();

    var total = try q.add(&ctx, &exp_x);
    defer total.deinit();
    var total2 = try total.add(&ctx, &sqrt_x);
    defer total2.deinit();
    var total3 = try total2.add(&ctx, &rsqrt_x);
    defer total3.deinit();
    var total4 = try total3.add(&ctx, &sigmoid_x);
    defer total4.deinit();
    var total5 = try total4.add(&ctx, &silu_x);
    defer total5.deinit();
    var total6 = try total5.add(&ctx, &log_x);
    defer total6.deinit();
    var loss = try total6.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    const data = grad.asRawTensor().dataConst();
    const s0 = testSigmoid(1);
    const s1 = testSigmoid(4);
    const expected = [_]f32{
        1.0 / 2.0 + @exp(@as(f32, 1)) + 0.5 / @sqrt(@as(f32, 1)) - 0.5 / (@as(f32, 1) * @sqrt(@as(f32, 1))) + s0 * (1 - s0) + s0 * (1 + @as(f32, 1) * (1 - s0)) + 1.0 / 1.0,
        1.0 / 8.0 + @exp(@as(f32, 4)) + 0.5 / @sqrt(@as(f32, 4)) - 0.5 / (@as(f32, 4) * @sqrt(@as(f32, 4))) + s1 * (1 - s1) + s1 * (1 + @as(f32, 4) * (1 - s1)) + 1.0 / 4.0,
    };
    try expectCloseSlices(&expected, data, 1e-5);

    var x2 = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x2.deinit();
    var mean = try x2.mean(&ctx, .d, .{});
    defer mean.deinit();
    var mean_loss = try mean.sumAll(&ctx);
    defer mean_loss.deinit();
    try mean_loss.backward(&ctx);
    var mean_grad = (try x2.grad(&ctx)).?;
    defer mean_grad.deinit();
    try expectCloseSlices(&.{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 }, mean_grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd differentiates extended unary ops and clamp" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ -0.75, 1.25 });
    defer x.deinit();

    var neg_x = try x.neg(&ctx);
    defer neg_x.deinit();
    var abs_x = try x.abs(&ctx);
    defer abs_x.deinit();
    var sin_x = try x.sin(&ctx);
    defer sin_x.deinit();
    var cos_x = try x.cos(&ctx);
    defer cos_x.deinit();
    var tanh_x = try x.tanh(&ctx);
    defer tanh_x.deinit();
    var gelu_x = try x.gelu(&ctx);
    defer gelu_x.deinit();
    var quick_x = try x.quickGelu(&ctx);
    defer quick_x.deinit();
    var clamp_x = try x.clamp(&ctx, -0.5, 1.0);
    defer clamp_x.deinit();

    var total = try neg_x.add(&ctx, &abs_x);
    defer total.deinit();
    var total2 = try total.add(&ctx, &sin_x);
    defer total2.deinit();
    var total3 = try total2.add(&ctx, &cos_x);
    defer total3.deinit();
    var total4 = try total3.add(&ctx, &tanh_x);
    defer total4.deinit();
    var total5 = try total4.add(&ctx, &gelu_x);
    defer total5.deinit();
    var total6 = try total5.add(&ctx, &quick_x);
    defer total6.deinit();
    var total7 = try total6.add(&ctx, &clamp_x);
    defer total7.deinit();
    var loss = try total7.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();

    const x0: f32 = -0.75;
    const x1: f32 = 1.25;
    const expected = [_]f32{
        -1 - 1 + @cos(x0) - @sin(x0) + testTanhDerivative(x0) + testGeluDerivative(x0) + testQuickGeluDerivative(x0) + 0,
        -1 + 1 + @cos(x1) - @sin(x1) + testTanhDerivative(x1) + testGeluDerivative(x1) + testQuickGeluDerivative(x1) + 0,
    };
    try expectCloseSlices(&expected, grad.asRawTensor().dataConst(), 1e-5);
}

test "clampMin/clampMax clamp one side and pass gradient through the open side" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ -2.0, 0.5, 3.0 });
    defer x.deinit();

    var lo = try x.clampMin(&ctx, 0.0);
    defer lo.deinit();
    try expectCloseSlices(&.{ 0.0, 0.5, 3.0 }, lo.asRawTensor().dataConst(), 0);
    var hi = try x.clampMax(&ctx, 1.0);
    defer hi.deinit();
    try expectCloseSlices(&.{ -2.0, 0.5, 1.0 }, hi.asRawTensor().dataConst(), 0);

    // d(clampMin)/dx = [0,1,1], d(clampMax)/dx = [1,1,0] — the open
    // (infinite) side never clips values or gradients.
    var total = try lo.add(&ctx, &hi);
    defer total.deinit();
    var loss = try total.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(&.{ 1.0, 2.0, 1.0 }, grad.asRawTensor().dataConst(), 0);
}

test "tagged autograd differentiates fused glu and swiglu" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 2, -3 });
    defer a.deinit();
    var gate = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 0, 1 });
    defer gate.deinit();

    var y = try a.swiglu(&ctx, &gate);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gg = (try gate.grad(&ctx)).?;
    defer gg.deinit();

    const s0 = testSigmoid(0);
    const s1 = testSigmoid(1);
    try expectCloseSlices(&.{ 0 * s0, 1 * s1 }, ga.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ 2 * s0 * (1 + 0 * (1 - s0)), -3 * s1 * (1 + 1 * (1 - s1)) }, gg.asRawTensor().dataConst(), 1e-6);

    var left = try Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 2, -3 });
    defer left.deinit();
    var y_glu = try left.glu(&ctx, &gate);
    defer y_glu.deinit();
    try expectCloseSlices(&.{ 2 * s0, -3 * s1 }, y_glu.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd f32 cast preserves the graph" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, -2, 4 });
    defer x.deinit();

    var y = try x.to(&ctx, .f32);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(&.{ 1, 1, 1 }, grad.asRawTensor().dataConst(), 1e-6);
    // f16/bf16 narrows are differentiable (the mixed-precision seam);
    // f64 stays a no-grad-only cast.
    var narrowed = try x.to(&ctx, .f16);
    defer narrowed.deinit();
    try std.testing.expect(narrowed.requiresGrad());
    try std.testing.expectError(error.GradientCastUnsupported, x.to(&ctx, .f64));
}

test "tagged autograd differentiates splitSwiGlu along the fused axis" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .ff }).variableFromSlice(&ctx, .{ 1, 4 }, &.{ 0, 1, 2, -3 });
    defer x.deinit();

    var y = try x.splitGated(&ctx, .swiglu, .ff, .d);
    defer y.deinit();
    const s0 = testSigmoid(0);
    const s1 = testSigmoid(1);
    try expectCloseSlices(&.{ 2 * 0 * s0, -3 * 1 * s1 }, y.asRawTensor().dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(
        &.{ 2 * s0 * (1 + 0 * (1 - s0)), -3 * s1 * (1 + 1 * (1 - s1)), 0 * s0, 1 * s1 },
        grad.asRawTensor().dataConst(),
        1e-6,
    );
}

test "tagged autograd differentiates splitGlu along the fused axis" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .ff }).variableFromSlice(&ctx, .{ 2, 4 }, &.{
        2, -3, 0, 0,
        4, 6,  0, 0,
    });
    defer x.deinit();

    var y = try x.splitGated(&ctx, .glu, .ff, .d);
    defer y.deinit();
    try expectCloseSlices(&.{ 1, -1.5, 2, 3 }, y.asRawTensor().dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(
        &.{ 0.5, 0.5, 0.5, -0.75, 0.5, 0.5, 1, 1.5 },
        grad.asRawTensor().dataConst(),
        1e-6,
    );
}

fn testGelu(x: f32) f32 {
    return 0.5 * x * (1 + std.math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

fn testGeluDeriv(x: f32) f32 {
    const a: f32 = 0.7978845608028654;
    const x2 = x * x;
    const u = a * (x + 0.044715 * x * x2);
    const t = std.math.tanh(u);
    const du = a * (1 + 3 * 0.044715 * x2);
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
}

test "tagged tanh/gelu/geglu match scalar reference over the SIMD path" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const N = 40; // > 4*vector_len so the vectorized tanh/gelu path runs, plus a tail
    var vals: [N]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) - 20.0) * 0.7; // ~ -14 .. 13

    var x = try Tensor(.{.d}).fromSlice(&ctx, .{N}, &vals);
    defer x.deinit();
    var th = try x.tanh(&ctx);
    defer th.deinit();
    var ge = try x.gelu(&ctx);
    defer ge.deinit();
    var gg = try x.geglu(&ctx, &x); // x * gelu(x)
    defer gg.deinit();

    for (0..N) |i| {
        try std.testing.expectApproxEqAbs(std.math.tanh(vals[i]), th.asRawTensor().dataConst()[i], 1e-4);
        try std.testing.expectApproxEqAbs(testGelu(vals[i]), ge.asRawTensor().dataConst()[i], 1e-3);
        try std.testing.expectApproxEqAbs(vals[i] * testGelu(vals[i]), gg.asRawTensor().dataConst()[i], 2e-3);
    }
}

test "tagged autograd differentiates fused geglu (gelu gate)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 2, -3 });
    defer a.deinit();
    var gate = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 0, 1 });
    defer gate.deinit();

    // geglu(a, gate) = a * gelu(gate) (GELU tanh approximation), matching Gemma.
    var y = try a.geglu(&ctx, &gate);
    defer y.deinit();
    const g0 = testGelu(0);
    const g1 = testGelu(1);
    try expectCloseSlices(&.{ 2 * g0, -3 * g1 }, y.asRawTensor().dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    var gg = (try gate.grad(&ctx)).?;
    defer gg.deinit();
    try expectCloseSlices(&.{ g0, g1 }, ga.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ 2 * testGeluDeriv(0), -3 * testGeluDeriv(1) }, gg.asRawTensor().dataConst(), 1e-6);
}

fn testSigmoid(value: f32) f32 {
    if (value >= 0) {
        const z = @exp(-value);
        return 1 / (1 + z);
    }
    const z = @exp(value);
    return z / (1 + z);
}

fn testTanhDerivative(value: f32) f32 {
    const t = std.math.tanh(value);
    return 1 - t * t;
}

fn testGeluDerivative(value: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const x2 = value * value;
    const u = sqrt_2_over_pi * (value + 0.044715 * value * x2);
    const t = std.math.tanh(u);
    return 0.5 * (1 + t) + 0.5 * value * (1 - t * t) * sqrt_2_over_pi * (1 + 3 * 0.044715 * x2);
}

fn testQuickGeluDerivative(value: f32) f32 {
    const s = testSigmoid(1.702 * value);
    return s + value * 1.702 * s * (1 - s);
}

test "tagged autograd dropout regenerates the mask and matches mul-by-mask gradients" {
    const rng = @import("../../rng.zig");

    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const len = 64;
    const seed: u64 = 0xd20b;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const random = prng.random();
    var x_data: [len]f32 = undefined;
    var w_data: [len]f32 = undefined;
    for (&x_data) |*value| value.* = random.floatNorm(f32) + 0.25;
    for (&w_data) |*value| value.* = random.floatNorm(f32);

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 8, 8 }, &x_data);
    defer x.deinit();
    var w = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 8, 8 }, &w_data);
    defer w.deinit();

    for ([_]f32{ 0.25, 0.5 }) |p| {
        const scale = 1.0 / (1.0 - p);
        // Test-side mask from the same counter-based stream: m[i] = scale if
        // the 53-bit uniform of rng.at(seed, i) is < 1-p, else 0 — so
        // dropout(x) must equal x .* m exactly, forward and backward.
        var mask_data: [len]f32 = undefined;
        for (&mask_data, 0..) |*value, i| {
            const uniform = @as(f64, @floatFromInt(rng.at(seed, i) >> 11)) * 0x1.0p-53;
            value.* = if (uniform < 1.0 - @as(f64, p)) scale else 0;
        }
        var mask = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 8, 8 }, &mask_data);
        defer mask.deinit();

        // Forward: kept positions scaled exactly by 1/(1-p), zeros elsewhere.
        var y = try x.dropout(&ctx, p, seed);
        defer y.deinit();
        var kept: usize = 0;
        for (y.asRawTensor().dataConst(), x_data, mask_data) |out_value, in_value, m| {
            if (m != 0) {
                try std.testing.expectEqual(in_value * scale, out_value);
                kept += 1;
            } else {
                try std.testing.expectEqual(@as(f32, 0), out_value);
            }
        }
        try std.testing.expect(kept > 0 and kept < len);

        // Same seed -> bitwise identical output; different seed -> different.
        var y_same = try x.dropout(&ctx, p, seed);
        defer y_same.deinit();
        try std.testing.expectEqualSlices(f32, y.asRawTensor().dataConst(), y_same.asRawTensor().dataConst());
        var y_other = try x.dropout(&ctx, p, seed + 1);
        defer y_other.deinit();
        try std.testing.expect(!std.mem.eql(f32, y.asRawTensor().dataConst(), y_other.asRawTensor().dataConst()));

        // Gradients equal the x .* mask composition bitwise (non-trivial
        // upstream gradient via the constant weights).
        const dropout_grad = grad: {
            var weighted = try y.mul(&ctx, &w);
            defer weighted.deinit();
            var loss = try weighted.sumAll(&ctx);
            defer loss.deinit();
            try loss.backward(&ctx);
            var g = (try x.grad(&ctx)).?;
            defer g.deinit();
            break :grad (try allocator.dupe(f32, try g.dataConst()));
        };
        defer allocator.free(dropout_grad);
        x.zeroGrad();

        const mask_grad = grad: {
            var masked = try x.mul(&ctx, &mask);
            defer masked.deinit();
            var weighted = try masked.mul(&ctx, &w);
            defer weighted.deinit();
            var loss = try weighted.sumAll(&ctx);
            defer loss.deinit();
            try loss.backward(&ctx);
            var g = (try x.grad(&ctx)).?;
            defer g.deinit();
            break :grad (try allocator.dupe(f32, try g.dataConst()));
        };
        defer allocator.free(mask_grad);
        x.zeroGrad();

        try std.testing.expectEqualSlices(f32, mask_grad, dropout_grad);
    }

    // p == 0: identity view (no copy), bitwise-equal data, gradient flows.
    {
        var y = try x.dropout(&ctx, 0, seed);
        defer y.deinit();
        try std.testing.expect(y.asRawTensor().buffer == x.asRawTensor().buffer);
        try std.testing.expectEqualSlices(f32, &x_data, y.asRawTensor().dataConst());
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        var g = (try x.grad(&ctx)).?;
        defer g.deinit();
        for (try g.dataConst()) |value| try std.testing.expectEqual(@as(f32, 1), value);
        x.zeroGrad();
    }

    try std.testing.expectError(error.InvalidArgument, x.dropout(&ctx, 1.0, seed));
    try std.testing.expectError(error.InvalidArgument, x.dropout(&ctx, -0.5, seed));
}

test "public Tensor biasAdd / addAxisVectorInPlace / addScaledInPlace" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .batch, .n });
    const bias = [_]f32{ 10, 20, 30 };

    // out-of-place biasAdd: [2,3] + bias[3] along .n; source unchanged.
    var x = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var y = try x.biasAdd(&ctx, &bias, .n);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, y.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, x.asRawTensor().dataConst()); // unchanged

    // in-place addAxisVectorInPlace mutates x to match the out-of-place result.
    try x.addAxisVectorInPlace(&ctx, &bias, .n);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, x.asRawTensor().dataConst());

    // in-place addScaledInPlace (self += 0.5·other) matches ctx.addScaledInPlace.
    const S = Tensor(.{ .r, .c });
    var a = try S.fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try S.fromSlice(&ctx, .{ 2, 2 }, &.{ 10, 20, 30, 40 });
    defer b.deinit();
    var raw = try a.asRawTensor().clone(ctx.allocator);
    defer raw.deinit();
    try a.addScaledInPlace(&ctx, b, 0.5);
    try ctx.addScaledInPlace(&raw, b.asRawTensor(), 0.5);
    try std.testing.expectEqualSlices(f32, &.{ 6, 12, 18, 24 }, a.asRawTensor().dataConst());
    try std.testing.expectEqualSlices(f32, raw.dataConst(), a.asRawTensor().dataConst());
}

test "public Tensor scalar convenience ops (addScalar/subScalar/divScalar/powScalar/log1p) values" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{.d});
    var x = try T.fromSlice(&ctx, .{4}, &.{ 1, 2, 4, 8 });
    defer x.deinit();

    var a = try x.addScalar(&ctx, 10);
    defer a.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 12, 14, 18 }, a.asRawTensor().dataConst());

    var s = try x.subScalar(&ctx, 1);
    defer s.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 3, 7 }, s.asRawTensor().dataConst());

    var d = try x.divScalar(&ctx, 2);
    defer d.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 1, 2, 4 }, d.asRawTensor().dataConst());

    var p = try x.powScalar(&ctx, 2);
    defer p.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 16, 64 }, p.asRawTensor().dataConst());

    var l = try x.log1p(&ctx);
    defer l.deinit();
    for (l.asRawTensor().dataConst(), [_]f32{ 1, 2, 4, 8 }) |got, xv| {
        try std.testing.expectApproxEqAbs(@log(1 + xv), got, 1e-6);
    }
}

test "tagged autograd differentiates elu and geluErf" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ -1.0, 1.25 });
    defer x.deinit();

    var elu_x = try x.elu(&ctx);
    defer elu_x.deinit();
    var gelu_x = try x.geluErf(&ctx);
    defer gelu_x.deinit();

    // Forward values: elu(-1) = expm1(-1); geluErf(x) = x*Phi(x) with the
    // standard normal CDF: 1.25*Phi(1.25) = 1.1179378, -1*Phi(-1) = -0.15865526.
    try expectCloseSlices(&.{ -0.6321206, 1.25 }, elu_x.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ -0.15865526, 1.1179378 }, gelu_x.asRawTensor().dataConst(), 1e-5);

    var total = try elu_x.add(&ctx, &gelu_x);
    defer total.deinit();
    var loss = try total.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    // elu' = exp(x) for x <= 0, 1 for x > 0;
    // geluErf' = 0.5*(1+erf(x/sqrt(2))) + x*exp(-x*x/2)/sqrt(2*pi).
    const x0: f32 = -1.0;
    const x1: f32 = 1.25;
    const expected = [_]f32{
        @exp(x0) + testGeluErfDerivative(x0),
        1.0 + testGeluErfDerivative(x1),
    };
    try expectCloseSlices(&expected, grad.asRawTensor().dataConst(), 1e-5);

    // Finite-difference cross-check of the analytic geluErf derivative.
    inline for (.{ x0, x1 }) |point| {
        const h: f32 = 1e-3;
        const fd = (geluErfValue(point + h) - geluErfValue(point - h)) / (2 * h);
        try std.testing.expectApproxEqAbs(fd, testGeluErfDerivative(point), 1e-3);
    }
}

fn geluErfValue(value: f32) f32 {
    return 0.5 * value * (1 + backend_mod.ops.erff(value * 0.70710678118654752440084436210484));
}

fn testGeluErfDerivative(value: f32) f32 {
    const inv_sqrt_2pi: f32 = 0.3989422804014327;
    const cdf = 0.5 * (1 + backend_mod.ops.erff(value * 0.70710678118654752440084436210484));
    return cdf + value * @exp(-0.5 * value * value) * inv_sqrt_2pi;
}

test "public Tensor maximum minimum route gradients with even tie split" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var a = try V.variableFromSlice(&ctx, .{4}, &.{ 1, 5, 2, 2 });
    defer a.deinit();
    var b = try V.variableFromSlice(&ctx, .{4}, &.{ 3, 4, 2, -1 });
    defer b.deinit();

    var hi = try a.maximum(&ctx, &b);
    defer hi.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 5, 2, 2 }, try hi.dataConst());
    var lo = try a.minimum(&ctx, &b);
    defer lo.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, -1 }, try lo.dataConst());

    var loss = try hi.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var ga = (try a.grad(&ctx)).?;
    defer ga.deinit();
    try expectCloseSlices(&.{ 0, 1, 0.5, 1 }, try ga.dataConst(), 0);
    var gb = (try b.grad(&ctx)).?;
    defer gb.deinit();
    try expectCloseSlices(&.{ 1, 0, 0.5, 0 }, try gb.dataConst(), 0);

    // NaN in either operand propagates (torch.maximum, NOT IEEE maxNum).
    var with_nan = try V.fromSlice(&ctx, .{4}, &.{ std.math.nan(f32), 1, 2, 3 });
    defer with_nan.deinit();
    var c = try V.fromSlice(&ctx, .{4}, &.{ 0, 1, 2, 3 });
    defer c.deinit();
    var nan_out = try c.maximum(&ctx, &with_nan);
    defer nan_out.deinit();
    try std.testing.expect(std.math.isNan((try nan_out.dataConst())[0]));
}

test "public Tensor pow with tensor exponent and both gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var base = try V.variableFromSlice(&ctx, .{2}, &.{ 2, 3 });
    defer base.deinit();
    var expo = try V.variableFromSlice(&ctx, .{2}, &.{ 3, 2 });
    defer expo.deinit();

    var y = try base.pow(&ctx, &expo);
    defer y.deinit();
    try expectCloseSlices(&.{ 8, 9 }, try y.dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gb = (try base.grad(&ctx)).?;
    defer gb.deinit();
    // b·a^(b-1): {3·4, 2·3} = {12, 6}.
    try expectCloseSlices(&.{ 12, 6 }, try gb.dataConst(), 1e-5);
    var ge = (try expo.grad(&ctx)).?;
    defer ge.deinit();
    // ln(a)·a^b: {8·ln2, 9·ln3}.
    try expectCloseSlices(&.{ 8 * 0.6931472, 9 * 1.0986123 }, try ge.dataConst(), 1e-4);
}

test "public Tensor floor ceil round sign reciprocal" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const V = Tensor(.{.d});
    var x = try V.fromSlice(&ctx, .{6}, &.{ -1.5, -0.5, 0.5, 1.5, 2.5, 2.3 });
    defer x.deinit();

    var fl = try x.floor(&ctx);
    defer fl.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -2, -1, 0, 1, 2, 2 }, try fl.dataConst());
    var ce = try x.ceil(&ctx);
    defer ce.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -1, 0, 1, 2, 3, 3 }, try ce.dataConst());
    // Round-half-to-even (torch.round): ties go to the even neighbor.
    var ro = try x.round(&ctx);
    defer ro.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -2, -0.0, 0, 2, 2, 2 }, try ro.dataConst());
    var sg = try x.sign(&ctx);
    defer sg.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, 1, 1, 1, 1 }, try sg.dataConst());

    // sign preserves ±0 and propagates NaN; round passes big/NaN through.
    var edge = try V.fromSlice(&ctx, .{4}, &.{ 0.0, -0.0, std.math.nan(f32), 8388609.0 });
    defer edge.deinit();
    var sg_edge = try edge.sign(&ctx);
    defer sg_edge.deinit();
    const sge = try sg_edge.dataConst();
    try std.testing.expectEqual(@as(f32, 0), sge[0]);
    try std.testing.expect(std.math.signbit(sge[1]));
    try std.testing.expect(std.math.isNan(sge[2]));
    var ro_edge = try edge.round(&ctx);
    defer ro_edge.deinit();
    try std.testing.expectEqual(@as(f32, 8388609.0), (try ro_edge.dataConst())[3]);

    var r = try V.fromSlice(&ctx, .{3}, &.{ 2, -4, 0.5 });
    defer r.deinit();
    var rec = try r.reciprocal(&ctx);
    defer rec.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.5, -0.25, 2 }, try rec.dataConst());

    // Gradients: piecewise-constant ops get exact zero; reciprocal -1/x².
    var xv = try V.variableFromSlice(&ctx, .{2}, &.{ 1.4, -2.6 });
    defer xv.deinit();
    var yv = try xv.round(&ctx);
    defer yv.deinit();
    var loss = try yv.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, try gx.dataConst());

    var rv = try V.variableFromSlice(&ctx, .{2}, &.{ 2, -4 });
    defer rv.deinit();
    var ry = try rv.reciprocal(&ctx);
    defer ry.deinit();
    var rloss = try ry.sumAll(&ctx);
    defer rloss.deinit();
    try rloss.backward(&ctx);
    var gr = (try rv.grad(&ctx)).?;
    defer gr.deinit();
    try expectCloseSlices(&.{ -0.25, -0.0625 }, try gr.dataConst(), 1e-7);
}
