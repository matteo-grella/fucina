//! Normalization: rmsNorm and its fused mul/add forms, layerNorm and
//! layerNormAffine against f64 references and finite differences, groupNorm,
//! l2Normalize/cosineSimilarity, and the norm variants.

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
const fdCheckGrad = util.fdCheckGrad;
const fdFillPattern = util.fdFillPattern;
const fdWeightedSum = util.fdWeightedSum;

test "tagged autograd rmsNormMul matches rmsNorm followed by weight multiply" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x_values = [_]f32{ 1, 2, 3, 2, 0, 4 };
    const w_values = [_]f32{ 0.5, -1.5, 2 };

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &x_values);
    defer x.deinit();
    var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &w_values);
    defer w.deinit();
    var fused = try x.rmsNormMul(&ctx, .d, &w, 1e-5);
    defer fused.deinit();
    var fused_loss = try fused.sumAll(&ctx);
    defer fused_loss.deinit();
    try fused_loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();

    var x_ref = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &x_values);
    defer x_ref.deinit();
    var w_ref = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &w_values);
    defer w_ref.deinit();
    var norm = try x_ref.rmsNorm(&ctx, .d, 1e-5);
    defer norm.deinit();
    var ref_y = try norm.mul(&ctx, &w_ref);
    defer ref_y.deinit();
    var ref_loss = try ref_y.sumAll(&ctx);
    defer ref_loss.deinit();
    try ref_loss.backward(&ctx);

    var gx_ref = (try x_ref.grad(&ctx)).?;
    defer gx_ref.deinit();
    var gw_ref = (try w_ref.grad(&ctx)).?;
    defer gw_ref.deinit();
    try expectCloseSlices(gx_ref.asRawTensor().dataConst(), gx.asRawTensor().dataConst(), 1e-5);
    try expectCloseSlices(gw_ref.asRawTensor().dataConst(), gw.asRawTensor().dataConst(), 1e-5);
}

test "tagged autograd rmsNormMulAdd matches rmsNormMul plus residual add" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x_values = [_]f32{ 1, -2, 3, 4, 0.5, -1 };
    const w_values = [_]f32{ 0.5, -1.5, 2 };
    const r_values = [_]f32{ 0.25, 0.5, -0.75, 1.25, -1, 0.125 };
    const gy_values = [_]f32{ 1, -0.5, 0.25, -1.5, 2, 0.75 };

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &x_values);
    defer x.deinit();
    var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &w_values);
    defer w.deinit();
    var r = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &r_values);
    defer r.deinit();
    var gy = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &gy_values);
    defer gy.deinit();

    var fused = try x.rmsNormMulAdd(&ctx, .d, &w, &r, 1e-5);
    defer fused.deinit();
    var fused_weighted = try fused.mul(&ctx, &gy);
    defer fused_weighted.deinit();
    var fused_loss = try fused_weighted.sumAll(&ctx);
    defer fused_loss.deinit();
    try fused_loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    var gr = (try r.grad(&ctx)).?;
    defer gr.deinit();

    var x_ref = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &x_values);
    defer x_ref.deinit();
    var w_ref = try Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &w_values);
    defer w_ref.deinit();
    var r_ref = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &r_values);
    defer r_ref.deinit();
    var norm = try x_ref.rmsNormMul(&ctx, .d, &w_ref, 1e-5);
    defer norm.deinit();
    var ref_y = try r_ref.add(&ctx, &norm);
    defer ref_y.deinit();
    var ref_weighted = try ref_y.mul(&ctx, &gy);
    defer ref_weighted.deinit();
    var ref_loss = try ref_weighted.sumAll(&ctx);
    defer ref_loss.deinit();
    try ref_loss.backward(&ctx);

    var gx_ref = (try x_ref.grad(&ctx)).?;
    defer gx_ref.deinit();
    var gw_ref = (try w_ref.grad(&ctx)).?;
    defer gw_ref.deinit();
    var gr_ref = (try r_ref.grad(&ctx)).?;
    defer gr_ref.deinit();

    try expectCloseSlices(ref_y.asRawTensor().dataConst(), fused.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(gx_ref.asRawTensor().dataConst(), gx.asRawTensor().dataConst(), 1e-5);
    try expectCloseSlices(gw_ref.asRawTensor().dataConst(), gw.asRawTensor().dataConst(), 1e-5);
    try expectCloseSlices(gr_ref.asRawTensor().dataConst(), gr.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd rms norm backward matches row-wise formula" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 2, 0, 4 });
    defer x.deinit();

    var y = try x.rmsNorm(&ctx, .d, 1e-5);
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();

    var expected: [6]f32 = undefined;
    expectedRmsNormSumGrad(.{ 1, 2, 3 }, 1e-5, expected[0..3]);
    expectedRmsNormSumGrad(.{ 2, 0, 4 }, 1e-5, expected[3..6]);
    try expectCloseSlices(&expected, grad.asRawTensor().dataConst(), 1e-5);
}

/// Scalar probe for the layerNorm finite-difference checks:
/// loss = Σ (layerNorm[Affine](x) ⊙ r). `w_values == null` exercises the
/// plain (non-affine) op.
fn layerNormLossForTest(
    ctx: *ExecContext,
    x_values: []const f32,
    w_values: ?[]const f32,
    b_values: ?[]const f32,
    r_values: []const f32,
    rows: usize,
    cols: usize,
    eps: f32,
) !f32 {
    var x = try Tensor(.{ .token, .d }).fromSlice(ctx, .{ rows, cols }, x_values);
    defer x.deinit();
    var r = try Tensor(.{ .token, .d }).fromSlice(ctx, .{ rows, cols }, r_values);
    defer r.deinit();

    var y = blk: {
        if (w_values) |wv| {
            var w = try Tensor(.{.d}).fromSlice(ctx, .{cols}, wv);
            defer w.deinit();
            var b = try Tensor(.{.d}).fromSlice(ctx, .{cols}, b_values.?);
            defer b.deinit();
            break :blk try x.layerNorm(ctx, .d, eps, .{ .weight = &w, .bias = &b });
        }
        break :blk try x.layerNorm(ctx, .d, eps, .{});
    };
    defer y.deinit();
    var weighted = try y.mul(ctx, &r);
    defer weighted.deinit();
    var loss = try weighted.sumAll(ctx);
    defer loss.deinit();
    return loss.asRawTensor().item();
}

test "tagged autograd layerNorm matches the f64 closed form (PyTorch golden)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Golden spot-checks: the expected values come from evaluating the exact
    // closed form torch.nn.LayerNorm implements — y = (x−μ)/√(σ²+eps)·w + b
    // with the BIASED σ² — in f64 inside the test (the f64 closed form is
    // the PyTorch golden; torch evaluates the same formula).
    const Case = struct {
        rows: usize,
        cols: usize,
        x: []const f32,
        w: ?[]const f32 = null,
        b: ?[]const f32 = null,
        eps: f32,
    };
    const cases = [_]Case{
        .{ .rows = 1, .cols = 4, .x = &.{ 1, 2, 3, 4 }, .eps = 1e-5 },
        .{ .rows = 1, .cols = 3, .x = &.{ 0.5, -1.5, 2.0 }, .w = &.{ 2, 0.5, -1 }, .b = &.{ 0.1, -0.2, 0.3 }, .eps = 1e-6 },
        .{ .rows = 2, .cols = 2, .x = &.{ 1, 1, -3, 5 }, .w = &.{ -0.5, 1.5 }, .b = &.{ 0.25, -1 }, .eps = 1e-5 },
    };

    for (cases) |case| {
        var x = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ case.rows, case.cols }, case.x);
        defer x.deinit();

        var y = blk: {
            if (case.w) |wv| {
                var w = try Tensor(.{.d}).fromSlice(&ctx, .{case.cols}, wv);
                defer w.deinit();
                var b = try Tensor(.{.d}).fromSlice(&ctx, .{case.cols}, case.b.?);
                defer b.deinit();
                break :blk try x.layerNorm(&ctx, .d, case.eps, .{ .weight = &w, .bias = &b });
            }
            break :blk try x.layerNorm(&ctx, .d, case.eps, .{});
        };
        defer y.deinit();

        const yd = y.asRawTensor().dataConst();
        const n = @as(f64, @floatFromInt(case.cols));
        for (0..case.rows) |row| {
            var sum: f64 = 0;
            for (case.x[row * case.cols ..][0..case.cols]) |value| sum += value;
            const mean = sum / n;
            var sumsq: f64 = 0;
            for (case.x[row * case.cols ..][0..case.cols]) |value| {
                const centered = @as(f64, value) - mean;
                sumsq += centered * centered;
            }
            const inv_sigma = 1 / @sqrt(sumsq / n + @as(f64, case.eps));
            for (0..case.cols) |col| {
                var want = (@as(f64, case.x[row * case.cols + col]) - mean) * inv_sigma;
                if (case.w) |wv| want = want * wv[col] + case.b.?[col];
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), yd[row * case.cols + col], 1e-5);
            }
        }
    }
}

test "tagged autograd layerNorm and layerNormAffine match finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows = 3;
    const cols = 4;
    const x_values = [_]f32{
        0.4,  -1.2, 0.8,  1.5,
        1.1,  0.2,  -0.7, 0.5,
        -0.6, 1.3,  0.0,  -0.2,
    };
    const w_values = [_]f32{ 0.5, -1.25, 2, 0.75 };
    const b_values = [_]f32{ -0.3, 0.6, 0.1, -1 };
    const r_values = [_]f32{
        0.7,  -0.4, 1.1, 0.3,
        -0.9, 0.5,  0.2, -1.3,
        0.6,  -0.8, 1.4, 0.1,
    };
    const eps: f32 = 1e-5;
    const h: f32 = 1e-2;

    // Plain layerNorm: dx via finite differences.
    {
        var x = try Tensor(.{ .token, .d }).variableFromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var r = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ rows, cols }, &r_values);
        defer r.deinit();
        var y = try x.layerNorm(&ctx, .d, eps, .{});
        defer y.deinit();
        var weighted = try y.mul(&ctx, &r);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var grad = (try x.grad(&ctx)).?;
        defer grad.deinit();
        const gd = grad.asRawTensor().dataConst();
        var work = x_values;
        for (x_values, 0..) |_, i| {
            work = x_values;
            work[i] += h;
            const plus = try layerNormLossForTest(&ctx, &work, null, null, &r_values, rows, cols, eps);
            work[i] -= 2 * h;
            const minus = try layerNormLossForTest(&ctx, &work, null, null, &r_values, rows, cols, eps);
            try std.testing.expectApproxEqAbs((plus - minus) / (2 * h), gd[i], 2e-3);
        }
    }

    // Affine layerNorm: dx, dweight, dbias via finite differences.
    {
        var x = try Tensor(.{ .token, .d }).variableFromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &w_values);
        defer w.deinit();
        var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &b_values);
        defer b.deinit();
        var r = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ rows, cols }, &r_values);
        defer r.deinit();
        var y = try x.layerNorm(&ctx, .d, eps, .{ .weight = &w, .bias = &b });
        defer y.deinit();
        var weighted = try y.mul(&ctx, &r);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        const gxd = gx.asRawTensor().dataConst();
        var x_work = x_values;
        for (x_values, 0..) |_, i| {
            x_work = x_values;
            x_work[i] += h;
            const plus = try layerNormLossForTest(&ctx, &x_work, &w_values, &b_values, &r_values, rows, cols, eps);
            x_work[i] -= 2 * h;
            const minus = try layerNormLossForTest(&ctx, &x_work, &w_values, &b_values, &r_values, rows, cols, eps);
            try std.testing.expectApproxEqAbs((plus - minus) / (2 * h), gxd[i], 2e-3);
        }

        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        const gwd = gw.asRawTensor().dataConst();
        var w_work = w_values;
        for (w_values, 0..) |_, i| {
            w_work = w_values;
            w_work[i] += h;
            const plus = try layerNormLossForTest(&ctx, &x_values, &w_work, &b_values, &r_values, rows, cols, eps);
            w_work[i] -= 2 * h;
            const minus = try layerNormLossForTest(&ctx, &x_values, &w_work, &b_values, &r_values, rows, cols, eps);
            try std.testing.expectApproxEqAbs((plus - minus) / (2 * h), gwd[i], 2e-3);
        }

        var gb = (try b.grad(&ctx)).?;
        defer gb.deinit();
        const gbd = gb.asRawTensor().dataConst();
        var b_work = b_values;
        for (b_values, 0..) |_, i| {
            b_work = b_values;
            b_work[i] += h;
            const plus = try layerNormLossForTest(&ctx, &x_values, &w_values, &b_work, &r_values, rows, cols, eps);
            b_work[i] -= 2 * h;
            const minus = try layerNormLossForTest(&ctx, &x_values, &w_values, &b_work, &r_values, rows, cols, eps);
            try std.testing.expectApproxEqAbs((plus - minus) / (2 * h), gbd[i], 2e-3);
        }
    }
}

test "tagged autograd layerNormAffine prunes gradients per operand" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows = 2;
    const cols = 3;
    const x_values = [_]f32{ 0.4, -1.2, 0.8, 1.1, 0.2, -0.7 };
    const w_values = [_]f32{ 0.5, -1.25, 2 };
    const b_values = [_]f32{ -0.3, 0.6, 0.1 };

    // Full run: every operand is a variable.
    var gw_full: [cols]f32 = undefined;
    var gb_full: [cols]f32 = undefined;
    {
        var x = try Tensor(.{ .token, .d }).variableFromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &w_values);
        defer w.deinit();
        var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &b_values);
        defer b.deinit();
        var y = try x.layerNorm(&ctx, .d, 1e-5, .{ .weight = &w, .bias = &b });
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        @memcpy(&gw_full, gw.asRawTensor().dataConst());
        var gb = (try b.grad(&ctx)).?;
        defer gb.deinit();
        @memcpy(&gb_full, gb.asRawTensor().dataConst());
    }

    // Weight-only: x and bias are constants; the weight grad is bitwise the
    // same as the full run (same serial param pass on the same inputs).
    {
        var x = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var w = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &w_values);
        defer w.deinit();
        var b = try Tensor(.{.d}).fromSlice(&ctx, .{cols}, &b_values);
        defer b.deinit();
        var y = try x.layerNorm(&ctx, .d, 1e-5, .{ .weight = &w, .bias = &b });
        defer y.deinit();
        try std.testing.expect(y.requiresGrad());
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        try std.testing.expectEqualSlices(f32, &gw_full, gw.asRawTensor().dataConst());
    }

    // Bias-only.
    {
        var x = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var w = try Tensor(.{.d}).fromSlice(&ctx, .{cols}, &w_values);
        defer w.deinit();
        var b = try Tensor(.{.d}).variableFromSlice(&ctx, .{cols}, &b_values);
        defer b.deinit();
        var y = try x.layerNorm(&ctx, .d, 1e-5, .{ .weight = &w, .bias = &b });
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
        var gb = (try b.grad(&ctx)).?;
        defer gb.deinit();
        try std.testing.expectEqualSlices(f32, &gb_full, gb.asRawTensor().dataConst());
    }

    // All-constant inputs stay grad-free.
    {
        var x = try Tensor(.{ .token, .d }).fromSlice(&ctx, .{ rows, cols }, &x_values);
        defer x.deinit();
        var w = try Tensor(.{.d}).fromSlice(&ctx, .{cols}, &w_values);
        defer w.deinit();
        var b = try Tensor(.{.d}).fromSlice(&ctx, .{cols}, &b_values);
        defer b.deinit();
        var y = try x.layerNorm(&ctx, .d, 1e-5, .{ .weight = &w, .bias = &b });
        defer y.deinit();
        try std.testing.expect(!y.requiresGrad());
    }
}

fn expectedRmsNormSumGrad(comptime row: [3]f32, eps: f32, out: []f32) void {
    var sumsq: f32 = 0;
    var dot: f32 = 0;
    inline for (0..3) |i| {
        sumsq += row[i] * row[i];
        dot += row[i];
    }
    const inv_n = 1.0 / 3.0;
    const inv_rms = 1 / @sqrt(sumsq * inv_n + eps);
    const correction = dot * inv_n * inv_rms * inv_rms * inv_rms;
    inline for (0..3) |i| {
        out[i] = inv_rms - row[i] * correction;
    }
}

test "public Tensor groupNorm facade normalizes per group" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const eps: f32 = 1e-5;
    var x = try Tensor(.{ .time, .ch }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    // G=C: per-channel InstanceNorm over time; col0 {1,3} and col1 {2,4} both
    // have biased var 1.
    var wt = try Tensor(.{.ch}).fromSlice(&ctx, .{2}, &.{ 2.0, 3.0 });
    defer wt.deinit();
    var bt = try Tensor(.{.ch}).fromSlice(&ctx, .{2}, &.{ 10.0, 20.0 });
    defer bt.deinit();
    var y = try x.groupNorm(&ctx, .ch, 2, eps, .{ .weight = &wt, .bias = &bt });
    defer y.deinit();
    const inv = 1.0 / @sqrt(@as(f32, 1.0) + eps);
    try expectCloseSlices(&.{ -inv * 2 + 10, -inv * 3 + 20, inv * 2 + 10, inv * 3 + 20 }, y.asRawTensor().dataConst(), 1e-5);

    // No-affine arm.
    var y_plain = try x.groupNorm(&ctx, .ch, 2, eps, .{});
    defer y_plain.deinit();
    try expectCloseSlices(&.{ -inv, -inv, inv, inv }, y_plain.asRawTensor().dataConst(), 1e-6);
}

const GroupNormFdContext = struct {
    ctx: *ExecContext,
    x_vals: []f32,
    w_vals: ?[]f32,
    b_vals: ?[]f32,
    coef: []const f32,
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
};

fn groupNormFdLoss(c: GroupNormFdContext) anyerror!f32 {
    var x = try c.ctx.fromSlice(.f32, &.{ c.rows, c.cols }, c.x_vals);
    defer x.deinit();
    var weight: ?RawTensor = if (c.w_vals) |wv| try c.ctx.fromSlice(.f32, &.{c.cols}, wv) else null;
    defer if (weight) |*w| w.deinit();
    var bias: ?RawTensor = if (c.b_vals) |bv| try c.ctx.fromSlice(.f32, &.{c.cols}, bv) else null;
    defer if (bias) |*b| b.deinit();
    var y = try c.ctx.groupNorm(&x, c.groups, c.eps, .{ .weight = if (weight) |*w| w else null, .bias = if (bias) |*b| b else null });
    defer y.deinit();
    return fdWeightedSum(y.dataConst(), c.coef);
}

test "tagged autograd groupNorm matches finite differences across group configurations" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const rows: usize = 3;
    const cols: usize = 8;
    const eps: f32 = 1e-5;

    // {groups, affine}: G=1, G=C, G=4 with affine, G=4 without.
    const configs = [_]struct { groups: usize, affine: bool }{
        .{ .groups = 1, .affine = false },
        .{ .groups = cols, .affine = false },
        .{ .groups = 4, .affine = true },
        .{ .groups = 4, .affine = false },
    };
    for (configs) |cfg| {
        const x_vals = try allocator.alloc(f32, rows * cols);
        defer allocator.free(x_vals);
        fdFillPattern(x_vals, 1.1);
        const w_vals = try allocator.alloc(f32, cols);
        defer allocator.free(w_vals);
        fdFillPattern(w_vals, 2.4);
        for (w_vals) |*wv| wv.* += 1.0; // keep the affine weight away from 0
        const b_vals = try allocator.alloc(f32, cols);
        defer allocator.free(b_vals);
        fdFillPattern(b_vals, 3.6);
        const coef = try allocator.alloc(f32, rows * cols);
        defer allocator.free(coef);
        fdFillPattern(coef, 4.7);

        var x = try Tensor(.{ .time, .ch }).variable(&ctx, try ctx.fromSlice(.f32, &.{ rows, cols }, x_vals));
        defer x.deinit();
        var weight: ?Tensor(.{.ch}) = if (cfg.affine) try Tensor(.{.ch}).variable(&ctx, try ctx.fromSlice(.f32, &.{cols}, w_vals)) else null;
        defer if (weight) |*w| w.deinit();
        var bias: ?Tensor(.{.ch}) = if (cfg.affine) try Tensor(.{.ch}).variable(&ctx, try ctx.fromSlice(.f32, &.{cols}, b_vals)) else null;
        defer if (bias) |*b| b.deinit();
        var coef_t = try Tensor(.{ .time, .ch }).fromTensor(&ctx, try ctx.fromSlice(.f32, &.{ rows, cols }, coef));
        defer coef_t.deinit();

        var y = try x.groupNorm(&ctx, .ch, cfg.groups, eps, .{ .weight = if (weight) |*w| w else null, .bias = if (bias) |*b| b else null });
        defer y.deinit();
        var weighted = try y.mul(&ctx, &coef_t);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();

        const fd_ctx = GroupNormFdContext{
            .ctx = &ctx,
            .x_vals = x_vals,
            .w_vals = if (cfg.affine) w_vals else null,
            .b_vals = if (cfg.affine) b_vals else null,
            .coef = coef,
            .rows = rows,
            .cols = cols,
            .groups = cfg.groups,
            .eps = eps,
        };
        try fdCheckGrad(x_vals, gx.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, groupNormFdLoss);
        if (cfg.affine) {
            var gw = (try weight.?.grad(&ctx)).?;
            defer gw.deinit();
            var gb = (try bias.?.grad(&ctx)).?;
            defer gb.deinit();
            try fdCheckGrad(w_vals, gw.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, groupNormFdLoss);
            try fdCheckGrad(b_vals, gb.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, groupNormFdLoss);
        }
    }
}

test "public Tensor l2Normalize and cosineSimilarity compositions" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // y = x·rsqrt(sum(x²) + eps): {3, 4} -> {0.6, 0.8} (eps = 0).
    const V = Tensor(.{.d});
    var v = try V.fromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer v.deinit();
    var vn = try v.l2Normalize(&ctx, .d, 0);
    defer vn.deinit();
    try expectCloseSlices(&.{ 0.6, 0.8 }, try vn.dataConst(), 1e-6);

    // Per-row normalization along the tag axis of a rank-2 tensor.
    const M = Tensor(.{ .row, .d });
    var m = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 3, 4, 6, 8 });
    defer m.deinit();
    var mn = try m.l2Normalize(&ctx, .d, 0);
    defer mn.deinit();
    try expectCloseSlices(&.{ 0.6, 0.8, 0.6, 0.8 }, try mn.dataConst(), 1e-6);

    // cos({1,0,1,0}, {1,1,0,0}) = 1/(√2·√2) = 0.5 (torch F.cosine_similarity).
    const W = Tensor(.{.d});
    var a = try W.fromSlice(&ctx, .{4}, &.{ 1, 0, 1, 0 });
    defer a.deinit();
    var b = try W.fromSlice(&ctx, .{4}, &.{ 1, 1, 0, 0 });
    defer b.deinit();
    var cos = try a.cosineSimilarity(&ctx, &b, .d, 1e-8);
    defer cos.deinit();
    try std.testing.expectApproxEqAbs(0.5, try cos.item(), 1e-6);

    // Zero vector: the eps clamp keeps the quotient finite -> similarity 0.
    var zero = try W.fromSlice(&ctx, .{4}, &.{ 0, 0, 0, 0 });
    defer zero.deinit();
    var cos_zero = try zero.cosineSimilarity(&ctx, &b, .d, 1e-8);
    defer cos_zero.deinit();
    try std.testing.expectApproxEqAbs(0.0, try cos_zero.item(), 1e-6);

    // Rank-2 reduces the tag away: rows {3,4}·{3,4} colinear -> 1,
    // {1,0}·{0,1} orthogonal -> 0.
    var p = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 3, 4, 1, 0 });
    defer p.deinit();
    var q = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 3, 4, 0, 1 });
    defer q.deinit();
    var cos_rows = try p.cosineSimilarity(&ctx, &q, .d, 1e-8);
    defer cos_rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, cos_rows.asRawTensor().shape.slice());
    try expectCloseSlices(&.{ 1, 0 }, try cos_rows.dataConst(), 1e-6);
}

test "public Tensor norm variants and the l2 gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 3, -4, 1, -1 });
    defer x.deinit();

    var l1 = try x.norm(&ctx, .col, .l1);
    defer l1.deinit();
    try expectCloseSlices(&.{ 7, 2 }, try l1.dataConst(), 0);
    var l2 = try x.norm(&ctx, .col, .l2);
    defer l2.deinit();
    try expectCloseSlices(&.{ 5, std.math.sqrt2 }, try l2.dataConst(), 1e-6);
    var linf = try x.norm(&ctx, .col, .inf);
    defer linf.deinit();
    try expectCloseSlices(&.{ 4, 1 }, try linf.dataConst(), 0);

    var total = try x.normAll(&ctx, .l1);
    defer total.deinit();
    try std.testing.expectEqual(@as(f32, 9), try total.item());

    // l2 gradient is x/‖x‖.
    var xv = try Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 3, -4 });
    defer xv.deinit();
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var n = try xv.norm(&ctx, .d, .l2);
        defer n.deinit();
        var loss = try n.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 0.6, -0.8 }, try gx.dataConst(), 1e-6);
}
