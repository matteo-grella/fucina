//! Behavioral tests for the public gradcheck helper.
const std = @import("std");
const exec_mod = @import("../exec.zig");
const tensor_mod = @import("../tensor.zig");
const custom = @import("custom.zig");
const gradcheck_mod = @import("gradcheck.zig");
const ag_tensor = @import("tensor.zig");

const ExecContext = exec_mod.ExecContext;
const RawTensor = tensor_mod.Tensor;
const Tensor = ag_tensor.Tensor;

const Scale = struct {
    value: f32,
};

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

const BadScaledMul = struct {
    pub const Output = Tensor(.{.d});

    pub const forward = ScaledMul.forward;

    pub fn backward(
        ctx: *ExecContext,
        extra: Scale,
        inputs: []const *const RawTensor,
        output: *const RawTensor,
        gy: *const RawTensor,
        needs_grad: []const bool,
        out: []?RawTensor,
    ) !void {
        _ = extra;
        _ = output;
        if (needs_grad[0]) out[0] = try ctx.mulRank(1, gy, inputs[1]);
        if (needs_grad[1]) out[1] = try ctx.mulRank(1, gy, inputs[0]);
    }
};

fn scaledMulLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try custom.customVjp(ctx, ScaledMul, Scale{ .value = 0.5 }, .{ a, b });
    defer y.deinit();
    return y.sumAll(ctx);
}

fn badScaledMulLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try custom.customVjp(ctx, BadScaledMul, Scale{ .value = 0.5 }, .{ a, b });
    defer y.deinit();
    return y.sumAll(ctx);
}

fn log1pLoss(ctx: *ExecContext, a: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try a.log1p(ctx);
    defer y.deinit();
    return y.sumAll(ctx);
}

fn powScalarLoss(ctx: *ExecContext, a: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try a.powScalar(ctx, 2.5);
    defer y.deinit();
    return y.sumAll(ctx);
}

fn whereLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    var cond = try Tensor(.{.d}).fromSlice(ctx, .{4}, &.{ 1, 0, 1, 0 });
    defer cond.deinit();
    var y = try a.where(ctx, cond, b);
    defer y.deinit();
    return y.sumAll(ctx);
}

fn maskedFillLoss(ctx: *ExecContext, a: *const Tensor(.{.d})) !Tensor(.{}) {
    var mask = try Tensor(.{.d}).fromSlice(ctx, .{4}, &.{ 0, 1, 0, 1 });
    defer mask.deinit();
    var y = try a.maskedFill(ctx, mask, 5.0);
    defer y.deinit();
    return y.sumAll(ctx);
}

test "gradcheck validates a custom VJP against central finite differences" {
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

    const result = try gradcheck_mod.gradcheck(&ctx, scaledMulLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 6), result.checked);
    try std.testing.expect(result.max_abs_error <= 2e-3);

    try std.testing.expect((try a.grad(&ctx)) == null);
    try std.testing.expect((try b.grad(&ctx)) == null);
}

fn conv2dGcLoss(ctx: *ExecContext, x: *const Tensor(.{ .h, .w, .c }), w: *const Tensor(.{ .oc, .kh, .kw, .c }), b: *const Tensor(.{.oc})) !Tensor(.{}) {
    var y = try x.conv2d(ctx, w, b, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .oh, .ow, .oc });
    defer y.deinit();
    return y.sumAll(ctx);
}

test "gradcheck: conv2d input+weight+bias VJP (channel-last)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // 3x3x2 input, weight [cout2,kh2,kw2,cin2], bias[2], stride 1 pad 0 -> out 2x2x2.
    var x = try Tensor(.{ .h, .w, .c }).variableFromSlice(&ctx, .{ 3, 3, 2 }, &.{
        0.5, -1.0, 0.3, 0.8, -0.2, 1.1, 0.9, -0.7, 0.4, -0.5, 1.2, 0.1, -0.9, 0.6, 0.2, -0.3, 0.7, -1.1,
    });
    defer x.deinit();
    var w = try Tensor(.{ .oc, .kh, .kw, .c }).variableFromSlice(&ctx, .{ 2, 2, 2, 2 }, &.{
        0.2, -0.4, 0.1, 0.5, -0.3, 0.6, 0.7, -0.1, 0.3, 0.2, -0.5, 0.4, 0.8, -0.6, 0.1, -0.2,
    });
    defer w.deinit();
    var b = try Tensor(.{.oc}).variableFromSlice(&ctx, .{2}, &.{ 0.1, -0.2 });
    defer b.deinit();

    const result = try gradcheck_mod.gradcheck(&ctx, conv2dGcLoss, .{ &x, &w, &b }, .{});
    try std.testing.expectEqual(@as(usize, 18 + 16 + 2), result.checked);
    try std.testing.expect(result.max_abs_error <= 2e-3);
}

fn conv2dStridedGcLoss(ctx: *ExecContext, x: *const Tensor(.{ .h, .w, .c }), w: *const Tensor(.{ .oc, .kh, .kw, .c }), b: *const Tensor(.{.oc})) !Tensor(.{}) {
    var y = try x.conv2d(ctx, w, b, .{ 2, 2 }, .{ 1, 1 }, 1, .{ .oh, .ow, .oc });
    defer y.deinit();
    return y.sumAll(ctx);
}

test "gradcheck: conv2d 3x3 s2 p1 input+weight+bias VJP (GEMM backward route)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // 5x5x2 input, weight [cout2,kh3,kw3,cin2], bias[2], stride 2 pad 1 ->
    // out 3x3x2. Strided+padded 3x3 exercises the col2im/im2col GEMM
    // backward decomposition end to end through the autograd VJP.
    var xd: [5 * 5 * 2]f32 = undefined;
    for (&xd, 0..) |*v, i| v.* = 0.1 * (@as(f32, @floatFromInt((i * 7) % 11)) - 5.0);
    var x = try Tensor(.{ .h, .w, .c }).variableFromSlice(&ctx, .{ 5, 5, 2 }, &xd);
    defer x.deinit();
    var wd: [2 * 3 * 3 * 2]f32 = undefined;
    for (&wd, 0..) |*v, i| v.* = 0.1 * (@as(f32, @floatFromInt((i * 5) % 13)) - 6.0);
    var w = try Tensor(.{ .oc, .kh, .kw, .c }).variableFromSlice(&ctx, .{ 2, 3, 3, 2 }, &wd);
    defer w.deinit();
    var b = try Tensor(.{.oc}).variableFromSlice(&ctx, .{2}, &.{ 0.1, -0.2 });
    defer b.deinit();

    const result = try gradcheck_mod.gradcheck(&ctx, conv2dStridedGcLoss, .{ &x, &w, &b }, .{});
    try std.testing.expectEqual(@as(usize, 50 + 36 + 2), result.checked);
    try std.testing.expect(result.max_abs_error <= 2e-3);
}

fn maxPoolGcLoss(ctx: *ExecContext, x: *const Tensor(.{ .h, .w, .c })) !Tensor(.{}) {
    var y = try x.maxPool2d(ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 });
    defer y.deinit();
    return y.sumAll(ctx);
}

fn avgPoolGcLoss(ctx: *ExecContext, x: *const Tensor(.{ .h, .w, .c })) !Tensor(.{}) {
    // pad 1 < kernel 2 exercises the valid-tap (count-exclude-pad) arm.
    var y = try x.avgPool2d(ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 1, 1 });
    defer y.deinit();
    return y.sumAll(ctx);
}

fn upsampleGcLoss(ctx: *ExecContext, x: *const Tensor(.{ .h, .w, .c })) !Tensor(.{}) {
    var y = try x.upsample2xNearest(ctx);
    defer y.deinit();
    // Weight each output cell differently through a CONSTANT so the loss stays
    // linear in x (f32 FD stays exact) while still catching a wrong sum-pool
    // scatter in the VJP.
    var wdata: [8 * 8 * 2]f32 = undefined;
    for (&wdata, 0..) |*v, i| v.* = 0.01 * (@as(f32, @floatFromInt(i % 13)) - 6.0);
    var w = try Tensor(.{ .h, .w, .c }).fromSlice(ctx, .{ 8, 8, 2 }, &wdata);
    defer w.deinit();
    var zw = try y.mul(ctx, &w);
    defer zw.deinit();
    return zw.sumAll(ctx);
}

test "gradcheck: maxPool2d / avgPool2d / upsample2xNearest VJPs (channel-last)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // 4x4x2, values well separated so the max-pool argmax is FD-stable.
    var x = try Tensor(.{ .h, .w, .c }).variableFromSlice(&ctx, .{ 4, 4, 2 }, &.{
        0.9,  -1.2, 0.3, 0.7,  -0.5, 1.4,  0.2,  -0.8,
        -0.3, 0.6,  1.1, -0.9, 0.4,  -1.5, 0.85, 0.15,
        1.3,  -0.4, 0.5, 0.95, -1.1, 0.25, -0.6, 1.05,
        0.1,  -0.7, 1.2, -0.2, 0.65, -1.3, 0.45, 0.75,
    });
    defer x.deinit();

    const rmax = try gradcheck_mod.gradcheck(&ctx, maxPoolGcLoss, .{&x}, .{});
    try std.testing.expectEqual(@as(usize, 32), rmax.checked);
    try std.testing.expect(rmax.max_abs_error <= 2e-3);

    const ravg = try gradcheck_mod.gradcheck(&ctx, avgPoolGcLoss, .{&x}, .{});
    try std.testing.expectEqual(@as(usize, 32), ravg.checked);
    try std.testing.expect(ravg.max_abs_error <= 2e-3);

    const rup = try gradcheck_mod.gradcheck(&ctx, upsampleGcLoss, .{&x}, .{});
    try std.testing.expectEqual(@as(usize, 32), rup.checked);
    try std.testing.expect(rup.max_abs_error <= 2e-3);
}

fn preluGcLoss(ctx: *ExecContext, x: *const Tensor(.{ .h, .w, .c }), alpha: *const Tensor(.{.c})) !Tensor(.{}) {
    var y = try x.prelu(ctx, alpha);
    defer y.deinit();
    return y.sumAll(ctx);
}

fn channelAffineGcLoss(ctx: *ExecContext, x: *const Tensor(.{ .h, .w, .c }), s: *const Tensor(.{.c}), t: *const Tensor(.{.c})) !Tensor(.{}) {
    var y = try x.channelAffine(ctx, s, t);
    defer y.deinit();
    var sq = try y.mul(ctx, &y); // square so gscale/gshift depend on x
    defer sq.deinit();
    return sq.sumAll(ctx);
}

test "gradcheck: prelu (input+alpha) and channelAffine (input+scale+shift) VJPs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Values away from 0 so the PReLU kink is FD-stable.
    var x = try Tensor(.{ .h, .w, .c }).variableFromSlice(&ctx, .{ 2, 3, 2 }, &.{
        0.9, -1.2, 0.3, 0.7, -0.5, 1.4, -0.35, 0.62, 1.15, -0.95, 0.44, -1.5,
    });
    defer x.deinit();
    var alpha = try Tensor(.{.c}).variableFromSlice(&ctx, .{2}, &.{ 0.25, -0.4 });
    defer alpha.deinit();

    const rp = try gradcheck_mod.gradcheck(&ctx, preluGcLoss, .{ &x, &alpha }, .{});
    try std.testing.expectEqual(@as(usize, 12 + 2), rp.checked);
    try std.testing.expect(rp.max_abs_error <= 2e-3);

    var s = try Tensor(.{.c}).variableFromSlice(&ctx, .{2}, &.{ 1.3, -0.7 });
    defer s.deinit();
    var t = try Tensor(.{.c}).variableFromSlice(&ctx, .{2}, &.{ -0.2, 0.55 });
    defer t.deinit();

    const ra = try gradcheck_mod.gradcheck(&ctx, channelAffineGcLoss, .{ &x, &s, &t }, .{});
    try std.testing.expectEqual(@as(usize, 12 + 2 + 2), ra.checked);
    try std.testing.expect(ra.max_abs_error <= 2e-3);
}

test "gradcheck catches a wrong custom VJP" {
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

    try std.testing.expectError(
        error.GradientMismatch,
        gradcheck_mod.gradcheck(&ctx, badScaledMulLoss, .{ &a, &b }, .{ .print_mismatch = false }),
    );

    try std.testing.expect((try a.grad(&ctx)) == null);
    try std.testing.expect((try b.grad(&ctx)) == null);
}

test "gradcheck validates log1p VJP (1/(1+x))" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 0.5, 1.0, 2.0, 3.5 });
    defer a.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, log1pLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 4), result.checked);
    try std.testing.expect(result.max_abs_error <= 2e-3);
}

test "gradcheck validates powScalar VJP (c·x^(c-1))" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 0.5, 1.0, 2.0, 3.0 });
    defer a.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, powScalarLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 4), result.checked);
    try std.testing.expect(result.max_abs_error <= 2e-3);
}

test "gradcheck validates where VJP (cond routes grad to taken branch)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 5, 6, 7, 8 });
    defer b.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, whereLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 8), result.checked);
    try std.testing.expect(result.max_abs_error <= 2e-3);
}

test "gradcheck validates maskedFill VJP (grad zeroed where filled)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, maskedFillLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 4), result.checked);
    try std.testing.expect(result.max_abs_error <= 2e-3);
}

// --- Shape/scan/sort ops ----------------------------------------------------
// Every wrapper weights the op output by a FIXED asymmetric constant before
// summing, so index-routing mistakes (a wrong permutation/scatter target)
// change the gradient instead of hiding under an all-ones upstream.

fn maskedSelectLoss(ctx: *ExecContext, a: *const Tensor(.{.d})) !Tensor(.{}) {
    var mask = try Tensor(.{.d}).fromSlice(ctx, .{4}, &.{ 1, 0, 0, 1 });
    defer mask.deinit();
    var y = try a.maskedSelect(ctx, mask, .m);
    defer y.deinit();
    var w = try Tensor(.{.m}).fromSlice(ctx, .{2}, &.{ 1, -2 });
    defer w.deinit();
    var z = try y.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

fn stackLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try a.stack(ctx, .s, 0, &.{b});
    defer y.deinit();
    var w = try Tensor(.{ .s, .d }).fromSlice(ctx, .{ 2, 4 }, &.{ 1, -2, 3, 0.5, -1, 2, -3, 0.25 });
    defer w.deinit();
    var z = try y.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

fn unbindLoss(ctx: *ExecContext, a: *const Tensor(.{ .s, .d })) !Tensor(.{}) {
    var parts: [2]Tensor(.{.d}) = undefined;
    try a.unbindInto(ctx, .s, &parts);
    defer for (&parts) |*part| part.deinit();
    var w0 = try Tensor(.{.d}).fromSlice(ctx, .{3}, &.{ 1, -2, 3 });
    defer w0.deinit();
    var w1 = try Tensor(.{.d}).fromSlice(ctx, .{3}, &.{ 0.5, 2, -1 });
    defer w1.deinit();
    var z0 = try parts[0].mul(ctx, &w0);
    defer z0.deinit();
    var z1 = try parts[1].mul(ctx, &w1);
    defer z1.deinit();
    var s0 = try z0.sumAll(ctx);
    defer s0.deinit();
    var s1 = try z1.sumAll(ctx);
    defer s1.deinit();
    return s0.add(ctx, &s1);
}

fn flipLoss(ctx: *ExecContext, a: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try a.flip(ctx, .d);
    defer y.deinit();
    var w = try Tensor(.{.d}).fromSlice(ctx, .{4}, &.{ 1, -2, 3, 0.5 });
    defer w.deinit();
    var z = try y.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

fn rollLoss(ctx: *ExecContext, a: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try a.roll(ctx, .d, 1);
    defer y.deinit();
    var w = try Tensor(.{.d}).fromSlice(ctx, .{4}, &.{ 1, -2, 3, 0.5 });
    defer w.deinit();
    var z = try y.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

fn repeatAxisLoss(ctx: *ExecContext, a: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try a.repeatAxis(ctx, .d, 2);
    defer y.deinit();
    var w = try Tensor(.{.d}).fromSlice(ctx, .{8}, &.{ 1, -2, 3, 0.5, -1, 2, -3, 0.25 });
    defer w.deinit();
    var z = try y.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

fn cumsumLoss(ctx: *ExecContext, a: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try a.cumsum(ctx, .d);
    defer y.deinit();
    var w = try Tensor(.{.d}).fromSlice(ctx, .{4}, &.{ 1, -2, 3, 0.5 });
    defer w.deinit();
    var z = try y.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

fn padLoss(ctx: *ExecContext, a: *const Tensor(.{.d})) !Tensor(.{}) {
    var y = try a.pad(ctx, .d, 1, 2, 3.5);
    defer y.deinit();
    var w = try Tensor(.{.d}).fromSlice(ctx, .{7}, &.{ 1, -2, 3, 0.5, -1, 2, -3 });
    defer w.deinit();
    var z = try y.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

fn sortValuesLoss(ctx: *ExecContext, a: *const Tensor(.{.d})) !Tensor(.{}) {
    var result = try a.sort(ctx, .d, false);
    defer result.deinit();
    var w = try Tensor(.{.d}).fromSlice(ctx, .{4}, &.{ 1, -2, 3, 0.5 });
    defer w.deinit();
    var z = try result.values.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

test "gradcheck validates the composed maskedSelect gradient (flatten + gather)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 2, -3, 4, 1 });
    defer a.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, maskedSelectLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 4), result.checked);
}

test "gradcheck validates the composed stack gradient (both parents)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 2, -3, 4, 1 });
    defer a.deinit();
    var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 5, 7, -11, 2 });
    defer b.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, stackLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 8), result.checked);
}

test "gradcheck validates the composed unbindInto gradients (narrow + squeeze)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{ .s, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 2, -3, 4, 1, 5, -2 });
    defer a.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, unbindLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 6), result.checked);
}

test "gradcheck validates flip and roll permutation gradients (gather VJP)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 2, -3, 4, 1 });
    defer a.deinit();
    const flip_result = try gradcheck_mod.gradcheck(&ctx, flipLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 4), flip_result.checked);
    const roll_result = try gradcheck_mod.gradcheck(&ctx, rollLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 4), roll_result.checked);
}

test "gradcheck validates the repeatAxis multi-copy gradient accumulation" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 2, -3, 4, 1 });
    defer a.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, repeatAxisLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 4), result.checked);
}

test "gradcheck validates the cumsum VJP (reversed cumulative sum)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 2, -3, 4, 1 });
    defer a.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, cumsumLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 4), result.checked);
}

test "gradcheck validates the pad VJP (narrow of the upstream gradient)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 2, -3, 4, 1 });
    defer a.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, padLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 4), result.checked);
}

test "gradcheck validates the sort values VJP (scatter by saved indices)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // Distinct values well away from ties: the ±1e-3 finite-difference
    // perturbation must not change the sort permutation.
    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 2, -3, 4, 1 });
    defer a.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, sortValuesLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 4), result.checked);
}

fn relposShiftLoss(ctx: *ExecContext, a: *const Tensor(.{ .h, .q, .p })) !Tensor(.{}) {
    // [H=1, Tq=2, P=3] -> [H=1, Tq=2, Tk=2]; skew gather out[0,qi,kj] = a[0,qi,kj+1-qi].
    var y = try a.relposShift(ctx, 2, .{ .h, .q, .k });
    defer y.deinit();
    return y.sumAll(ctx);
}

// --- Elementwise losses + normalization convenience ops --------------------

fn mseMeanLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    return a.mseLoss(ctx, b, .{});
}

fn mseNoneLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    // `.none` + sumAll exercises the per-element upstream-gradient arm.
    var y = try a.mseLoss(ctx, b, .{ .reduction = .none });
    defer y.deinit();
    return y.sumAll(ctx);
}

fn huberMeanLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    return a.huberLoss(ctx, b, .{ .delta = 1.0 });
}

fn bceLogitsLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    return a.bceLoss(ctx, b, .{ .from_logits = true });
}

fn bceProbLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    return a.bceLoss(ctx, b, .{ .reduction = .sum });
}

fn klDivMeanLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    return a.klDivLoss(ctx, b, .{});
}

fn klDivLogTargetLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    return a.klDivLoss(ctx, b, .{ .log_target = true, .reduction = .sum });
}

fn nllMeanLoss(ctx: *ExecContext, a: *const Tensor(.{ .pos, .class })) !Tensor(.{}) {
    return a.nllLoss(ctx, .class, &.{ 2, 0 }, .mean);
}

fn l2NormalizeLoss(ctx: *ExecContext, a: *const Tensor(.{.d})) !Tensor(.{}) {
    // Fixed asymmetric weights so distinct gradient components are checked.
    var w = try Tensor(.{.d}).fromSlice(ctx, .{4}, &.{ 1, -2, 3, 0.5 });
    defer w.deinit();
    var y = try a.l2Normalize(ctx, .d, 1e-6);
    defer y.deinit();
    var z = try y.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

fn cosineSimilarityLoss(ctx: *ExecContext, a: *const Tensor(.{.d}), b: *const Tensor(.{.d})) !Tensor(.{}) {
    return a.cosineSimilarity(ctx, b, .d, 1e-8);
}

test "gradcheck validates mseLoss VJPs (mean and none reductions, both parents)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 2, -3, 4, 1 });
    defer a.deinit();
    var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 0.5, 0.7, -1.1, 1 });
    defer b.deinit();

    const mean_result = try gradcheck_mod.gradcheck(&ctx, mseMeanLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 8), mean_result.checked);
    const none_result = try gradcheck_mod.gradcheck(&ctx, mseNoneLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 8), none_result.checked);
}

test "gradcheck validates huberLoss VJP across the quadratic/linear arms" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // Diffs {0.4, -2, 1.7, 0.1} straddle delta = 1 while staying clear of
    // the |d| == delta kink (finite differences would be wrong there).
    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 0.4, -2, 1.7, 0.1 });
    defer a.deinit();
    var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 0, 0, 0, 0 });
    defer b.deinit();

    const result = try gradcheck_mod.gradcheck(&ctx, huberMeanLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 8), result.checked);
}

test "gradcheck validates bceLoss VJPs (logits arm and clamped probability arm)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var logits = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 0.5, -1.5, 2, -0.25 });
    defer logits.deinit();
    var target = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 0, 1, 0.5, 0.25 });
    defer target.deinit();
    const logits_result = try gradcheck_mod.gradcheck(&ctx, bceLogitsLoss, .{ &logits, &target }, .{});
    try std.testing.expectEqual(@as(usize, 8), logits_result.checked);

    // Boundary-adjacent probabilities (0.02 / 0.98): inside the bce_eps
    // clamp even under the ±1e-3 finite-difference perturbation.
    var probs = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 0.02, 0.3, 0.75, 0.98 });
    defer probs.deinit();
    var prob_target = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 1, 0.5, 0, 0.25 });
    defer prob_target.deinit();
    const prob_result = try gradcheck_mod.gradcheck(&ctx, bceProbLoss, .{ &probs, &prob_target }, .{});
    try std.testing.expectEqual(@as(usize, 8), prob_result.checked);
}

test "gradcheck validates klDivLoss VJPs (probability and log targets, both parents)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var logp = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ -0.36, -1.61, -2.3, -1.2 });
    defer logp.deinit();
    // Strictly positive so the ±1e-3 target perturbation stays in t > 0.
    var target = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 0.4, 0.3, 0.2, 0.1 });
    defer target.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, klDivMeanLoss, .{ &logp, &target }, .{});
    try std.testing.expectEqual(@as(usize, 8), result.checked);

    var log_target = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ -0.9, -1.2, -1.6, -2.3 });
    defer log_target.deinit();
    const log_result = try gradcheck_mod.gradcheck(&ctx, klDivLogTargetLoss, .{ &logp, &log_target }, .{});
    try std.testing.expectEqual(@as(usize, 8), log_result.checked);
}

test "gradcheck validates the composed nllLoss gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{ .pos, .class }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ -1, -2, -0.5, -0.3, -1.2, -2.3 });
    defer a.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, nllMeanLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 6), result.checked);
}

test "gradcheck validates the composed l2Normalize gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 2, -3, 4, 1 });
    defer a.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, l2NormalizeLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 4), result.checked);
}

test "gradcheck validates the composed cosineSimilarity gradient (both parents)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var a = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 2, -3, 4, 1 });
    defer a.deinit();
    var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 5, 7, -11, 2 });
    defer b.deinit();
    const result = try gradcheck_mod.gradcheck(&ctx, cosineSimilarityLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 8), result.checked);
}

test "gradcheck validates the relposShift (Transformer-XL skew) scatter VJP" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // a[0,0,0] and a[0,1,2] are NOT gathered (grad must be exactly 0); the other
    // four entries gather 1-to-1 into the [1,2,2] output.
    var a = try Tensor(.{ .h, .q, .p }).variableFromSlice(&ctx, .{ 1, 2, 3 }, &.{ 2, -3, 4, 5, -6, 7 });
    defer a.deinit();

    const result = try gradcheck_mod.gradcheck(&ctx, relposShiftLoss, .{&a}, .{});
    try std.testing.expectEqual(@as(usize, 6), result.checked);
    try std.testing.expect(result.max_abs_error <= 2e-3);
}

fn einsumBatchLoss(ctx: *ExecContext, a: *const Tensor(.{ .b, .i, .k }), b: *const Tensor(.{ .b, .k, .j })) !Tensor(.{}) {
    var y = try a.einsum(ctx, b, .{ .b, .i, .j });
    defer y.deinit();
    return y.sumAll(ctx);
}

fn einsumMultiContractLoss(ctx: *ExecContext, a: *const Tensor(.{ .i, .k1, .k2 }), b: *const Tensor(.{ .j, .k1, .k2 })) !Tensor(.{}) {
    var y = try a.einsum(ctx, b, .{ .i, .j });
    defer y.deinit();
    return y.sumAll(ctx);
}

fn einsumSummedLoss(ctx: *ExecContext, a: *const Tensor(.{ .i, .s, .k }), b: *const Tensor(.{ .k, .t, .j })) !Tensor(.{}) {
    var y = try a.einsum(ctx, b, .{ .i, .j });
    defer y.deinit();
    return y.sumAll(ctx);
}

fn einsumSwappedOutLoss(ctx: *ExecContext, a: *const Tensor(.{ .i, .k }), b: *const Tensor(.{ .k, .j })) !Tensor(.{}) {
    var y = try a.einsum(ctx, b, .{ .j, .i });
    defer y.deinit();
    return y.sumAll(ctx);
}

fn einsumOuterLoss(ctx: *ExecContext, a: *const Tensor(.{.i}), b: *const Tensor(.{.j})) !Tensor(.{}) {
    var y = try a.einsum(ctx, b, .{ .i, .j });
    defer y.deinit();
    var sq = try y.mul(ctx, &y);
    defer sq.deinit();
    return sq.sumAll(ctx);
}

fn einsumScalarLoss(ctx: *ExecContext, a: *const Tensor(.{ .i, .j }), b: *const Tensor(.{ .j, .i })) !Tensor(.{}) {
    return a.einsum(ctx, b, .{});
}

test "gradcheck: einsum batched and multi-contract VJPs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .b, .i, .k }).variableFromSlice(&ctx, .{ 2, 2, 3 }, &.{
        0.5, -1.0, 0.3, 0.8, -0.2, 1.1, 0.9, -0.7, 0.4, -0.5, 1.2, 0.1,
    });
    defer a.deinit();
    var b = try Tensor(.{ .b, .k, .j }).variableFromSlice(&ctx, .{ 2, 3, 2 }, &.{
        0.2, -0.4, 0.1, 0.5, -0.3, 0.6, 0.7, -0.1, 0.3, 0.2, -0.5, 0.4,
    });
    defer b.deinit();
    const batched = try gradcheck_mod.gradcheck(&ctx, einsumBatchLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 24), batched.checked);
    try std.testing.expect(batched.max_abs_error <= 2e-3);

    var c = try Tensor(.{ .i, .k1, .k2 }).variableFromSlice(&ctx, .{ 2, 2, 3 }, &.{
        0.5, -1.0, 0.3, 0.8, -0.2, 1.1, 0.9, -0.7, 0.4, -0.5, 1.2, 0.1,
    });
    defer c.deinit();
    var d = try Tensor(.{ .j, .k1, .k2 }).variableFromSlice(&ctx, .{ 2, 2, 3 }, &.{
        0.2, -0.4, 0.1, 0.5, -0.3, 0.6, 0.7, -0.1, 0.3, 0.2, -0.5, 0.4,
    });
    defer d.deinit();
    const multi = try gradcheck_mod.gradcheck(&ctx, einsumMultiContractLoss, .{ &c, &d }, .{});
    try std.testing.expectEqual(@as(usize, 24), multi.checked);
    try std.testing.expect(multi.max_abs_error <= 2e-3);
}

test "gradcheck: einsum summed-axis broadcast and permuted-output VJPs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{ .i, .s, .k }).variableFromSlice(&ctx, .{ 2, 2, 3 }, &.{
        0.5, -1.0, 0.3, 0.8, -0.2, 1.1, 0.9, -0.7, 0.4, -0.5, 1.2, 0.1,
    });
    defer a.deinit();
    var b = try Tensor(.{ .k, .t, .j }).variableFromSlice(&ctx, .{ 3, 2, 2 }, &.{
        0.2, -0.4, 0.1, 0.5, -0.3, 0.6, 0.7, -0.1, 0.3, 0.2, -0.5, 0.4,
    });
    defer b.deinit();
    const summed = try gradcheck_mod.gradcheck(&ctx, einsumSummedLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 24), summed.checked);
    try std.testing.expect(summed.max_abs_error <= 2e-3);

    var c = try Tensor(.{ .i, .k }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 0.5, -1.0, 0.3, 0.8, -0.2, 1.1 });
    defer c.deinit();
    var d = try Tensor(.{ .k, .j }).variableFromSlice(&ctx, .{ 3, 2 }, &.{ 0.2, -0.4, 0.1, 0.5, -0.3, 0.6 });
    defer d.deinit();
    const swapped = try gradcheck_mod.gradcheck(&ctx, einsumSwappedOutLoss, .{ &c, &d }, .{});
    try std.testing.expectEqual(@as(usize, 12), swapped.checked);
    try std.testing.expect(swapped.max_abs_error <= 2e-3);
}

test "gradcheck: einsum outer-product and full-contraction VJPs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try Tensor(.{.i}).variableFromSlice(&ctx, .{3}, &.{ 0.5, -1.0, 0.3 });
    defer a.deinit();
    var b = try Tensor(.{.j}).variableFromSlice(&ctx, .{4}, &.{ 0.2, -0.4, 0.1, 0.5 });
    defer b.deinit();
    const outer = try gradcheck_mod.gradcheck(&ctx, einsumOuterLoss, .{ &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 7), outer.checked);
    try std.testing.expect(outer.max_abs_error <= 2e-3);

    var c = try Tensor(.{ .i, .j }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 0.5, -1.0, 0.3, 0.8, -0.2, 1.1 });
    defer c.deinit();
    var d = try Tensor(.{ .j, .i }).variableFromSlice(&ctx, .{ 3, 2 }, &.{ 0.2, -0.4, 0.1, 0.5, -0.3, 0.6 });
    defer d.deinit();
    const scalar = try gradcheck_mod.gradcheck(&ctx, einsumScalarLoss, .{ &c, &d }, .{});
    try std.testing.expectEqual(@as(usize, 12), scalar.checked);
    try std.testing.expect(scalar.max_abs_error <= 2e-3);
}

fn einsumManyLoss(ctx: *ExecContext, x: *const Tensor(.{ .s, .i }), a: *const Tensor(.{ .r, .i }), b: *const Tensor(.{ .o, .r })) !Tensor(.{}) {
    var y = try ag_tensor.einsumMany(ctx, .{ .s, .o }, .{ x, a, b });
    defer y.deinit();
    return y.sumAll(ctx);
}

test "gradcheck: einsumMany three-operand chain VJPs through the fold" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .s, .i }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 0.5, -1.0, 0.3, 0.8, -0.2, 1.1 });
    defer x.deinit();
    var a = try Tensor(.{ .r, .i }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 0.2, -0.4, 0.1, 0.5, -0.3, 0.6 });
    defer a.deinit();
    var b = try Tensor(.{ .o, .r }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.7, -0.1, 0.3, 0.2 });
    defer b.deinit();

    const result = try gradcheck_mod.gradcheck(&ctx, einsumManyLoss, .{ &x, &a, &b }, .{});
    try std.testing.expectEqual(@as(usize, 6 + 6 + 4), result.checked);
    try std.testing.expect(result.max_abs_error <= 2e-3);
}
