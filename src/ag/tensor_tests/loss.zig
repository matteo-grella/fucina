//! Loss ops: fused cross entropy and crossEntropyExt options, elementwise
//! mse/huber/bce/klDiv losses, and nllLoss compositions with reductions and
//! gradients.

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

test "tagged autograd cross entropy fuses stable loss and logits gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var logits = try Tensor(.{ .token, .vocab }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 2, 0, -1 });
    defer logits.deinit();

    var loss = try logits.crossEntropy(&ctx, .vocab, &.{ 2, 0 }, .{});
    defer loss.deinit();

    const expected_loss = (expectedCrossEntropy(.{ 1, 2, 3 }, 2) + expectedCrossEntropy(.{ 2, 0, -1 }, 0)) / 2;
    try std.testing.expectApproxEqAbs(expected_loss, loss.asRawTensor().item(), 1e-6);

    try loss.backward(&ctx);
    var grad = (try logits.grad(&ctx)).?;
    defer grad.deinit();

    var expected: [6]f32 = undefined;
    expectedCrossEntropyGrad(.{ 1, 2, 3 }, 2, 0.5, expected[0..3]);
    expectedCrossEntropyGrad(.{ 2, 0, -1 }, 0, 0.5, expected[3..6]);
    try expectCloseSlices(&expected, grad.asRawTensor().dataConst(), 1e-6);
}

fn crossEntropyExtScalarLossForTest(
    ctx: *ExecContext,
    data: []const f32,
    labels: []const usize,
    comptime options: exec_mod.CrossEntropyOptions,
    weights: []const f32,
) !f32 {
    var logits = try ctx.fromSlice(.f32, .{ 4, 7 }, data);
    defer logits.deinit();
    var loss = try ctx.crossEntropyLoss(2, &logits, 1, labels, options);
    defer loss.deinit();
    if (comptime options.reduction == .none) {
        var acc: f32 = 0;
        for (loss.dataConst(), weights) |value, weight| acc += value * weight;
        return acc;
    }
    return loss.item();
}

test "tagged autograd crossEntropyExt matches finite differences across options" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const logit_values = [_]f32{
        0.4,  -1.2, 0.8,  1.5,  -0.3, 0.1,  -0.9,
        1.1,  0.2,  -0.7, 0.5,  -1.4, 0.9,  0.3,
        -0.6, 1.3,  0.0,  -0.2, 0.7,  -1.1, 0.6,
        0.9,  -0.5, 1.2,  -0.8, 0.3,  0.4,  -0.1,
    };
    const weights = [_]f32{ 0.5, -1.25, 2, 0.75 };

    inline for (.{
        exec_mod.CrossEntropyOptions{},
        exec_mod.CrossEntropyOptions{ .reduction = .sum },
        exec_mod.CrossEntropyOptions{ .reduction = .none },
        exec_mod.CrossEntropyOptions{ .ignore_index = 9, .label_smoothing = 0.1 },
        exec_mod.CrossEntropyOptions{ .reduction = .sum, .ignore_index = 9, .label_smoothing = 0.1 },
        exec_mod.CrossEntropyOptions{ .reduction = .none, .ignore_index = 9 },
    }) |options| {
        const labels: []const usize = if (comptime options.ignore_index != null) &.{ 2, 9, 0, 6 } else &.{ 2, 5, 0, 6 };

        var logits = try Tensor(.{ .token, .vocab }).variableFromSlice(&ctx, .{ 4, 7 }, &logit_values);
        defer logits.deinit();

        var loss_value: f32 = undefined;
        if (comptime options.reduction == .none) {
            var losses = try logits.crossEntropy(&ctx, .vocab, labels, options);
            defer losses.deinit();
            // The class tag is removed like sum/mean over an axis.
            try std.testing.expect(@TypeOf(losses).axis_tags.len == 1);
            try std.testing.expect(@TypeOf(losses).axis_tags[0] == .token);
            var w = try Tensor(.{.token}).fromSlice(&ctx, .{4}, &weights);
            defer w.deinit();
            var weighted = try losses.mul(&ctx, &w);
            defer weighted.deinit();
            var loss = try weighted.sumAll(&ctx);
            defer loss.deinit();
            loss_value = loss.asRawTensor().item();
            try loss.backward(&ctx);
        } else {
            var loss = try logits.crossEntropy(&ctx, .vocab, labels, options);
            defer loss.deinit();
            loss_value = loss.asRawTensor().item();
            try loss.backward(&ctx);
        }

        // The facade forward agrees with the exec-level kernel.
        try std.testing.expectApproxEqAbs(
            try crossEntropyExtScalarLossForTest(&ctx, &logit_values, labels, options, &weights),
            loss_value,
            1e-6,
        );

        var grad = (try logits.grad(&ctx)).?;
        defer grad.deinit();
        const gd = grad.asRawTensor().dataConst();

        const h: f32 = 1e-2;
        var work = logit_values;
        for (logit_values, 0..) |_, i| {
            work = logit_values;
            work[i] += h;
            const plus = try crossEntropyExtScalarLossForTest(&ctx, &work, labels, options, &weights);
            work[i] -= 2 * h;
            const minus = try crossEntropyExtScalarLossForTest(&ctx, &work, labels, options, &weights);
            const expected = (plus - minus) / (2 * h);
            try std.testing.expectApproxEqAbs(expected, gd[i], 2e-3);
        }

        // Ignored positions (row 1 when ignore_index is set) get exactly zero.
        if (comptime options.ignore_index != null) {
            for (gd[7..14]) |value| try std.testing.expectEqual(@as(f32, 0), value);
        }
    }
}

test "tagged autograd crossEntropyExt default options match crossEntropy" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var logits = try Tensor(.{ .token, .vocab }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 2, 0, -1 });
    defer logits.deinit();

    var legacy = try logits.crossEntropy(&ctx, .vocab, &.{ 2, 0 }, .{});
    defer legacy.deinit();
    var ext = try logits.crossEntropy(&ctx, .vocab, &.{ 2, 0 }, .{});
    defer ext.deinit();
    try std.testing.expectEqual(legacy.asRawTensor().item(), ext.asRawTensor().item());
}

fn expectedCrossEntropy(comptime logits: [3]f32, comptime label: usize) f32 {
    var max_value = logits[0];
    inline for (1..3) |i| max_value = @max(max_value, logits[i]);
    var sum_exp: f32 = 0;
    inline for (0..3) |i| sum_exp += @exp(logits[i] - max_value);
    return @log(sum_exp) + max_value - logits[label];
}

fn expectedCrossEntropyGrad(comptime logits: [3]f32, comptime label: usize, scale_value: f32, out: []f32) void {
    var max_value = logits[0];
    inline for (1..3) |i| max_value = @max(max_value, logits[i]);

    var probs: [3]f32 = undefined;
    var sum_exp: f32 = 0;
    inline for (0..3) |i| {
        probs[i] = @exp(logits[i] - max_value);
        sum_exp += probs[i];
    }

    inline for (0..3) |i| {
        var grad = probs[i] / sum_exp;
        if (i == label) grad -= 1;
        out[i] = grad * scale_value;
    }
}

test "public Tensor mseLoss/huberLoss/bceLoss/klDivLoss values and two-parent gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{ .batch, .d });
    var x = try T.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var t = try T.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, 2.5, 1.5, 4 });
    defer t.deinit();

    // torch F.mse_loss: mean((x - t)^2) with d = {0.5, -0.5, 1.5, 0}.
    var mse = try x.mseLoss(&ctx, &t, .{});
    defer mse.deinit();
    try std.testing.expectApproxEqAbs(0.6875, try mse.item(), 1e-6);
    try mse.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var gt = (try t.grad(&ctx)).?;
    defer gt.deinit();
    // d/dx = 2·d/4 = {0.25, -0.25, 0.75, 0}; d/dt = -d/dx.
    try expectCloseSlices(&.{ 0.25, -0.25, 0.75, 0 }, gx.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(&.{ -0.25, 0.25, -0.75, 0 }, gt.asRawTensor().dataConst(), 1e-6);

    // `.none` keeps the input tags/shape (per-element losses).
    var mse_none = try x.mseLoss(&ctx, &t, .{ .reduction = .none });
    defer mse_none.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, mse_none.asRawTensor().shape.slice());
    try expectCloseSlices(&.{ 0.25, 0.25, 2.25, 0 }, try mse_none.dataConst(), 1e-6);

    // torch F.huber_loss(delta=1): d = {0.5, -0.5, 1.5, 0} ->
    // {0.125, 0.125, 1·(1.5 - 0.5), 0}; sum = 1.25.
    var huber = try x.huberLoss(&ctx, &t, .{ .reduction = .sum });
    defer huber.deinit();
    try std.testing.expectApproxEqAbs(1.25, try huber.item(), 1e-6);

    // torch F.binary_cross_entropy_with_logits on logits {0, 2} vs
    // targets {0, 1}: (ln 2 + log1p(e^-2))/2 = 0.4100375958014592.
    const V = Tensor(.{.d});
    var logits = try V.variableFromSlice(&ctx, .{2}, &.{ 0, 2 });
    defer logits.deinit();
    var bce_target = try V.fromSlice(&ctx, .{2}, &.{ 0, 1 });
    defer bce_target.deinit();
    var bce = try logits.bceLoss(&ctx, &bce_target, .{ .from_logits = true });
    defer bce.deinit();
    try std.testing.expectApproxEqAbs(0.4100375958014592, try bce.item(), 1e-6);
    try bce.backward(&ctx);
    var g_logits = (try logits.grad(&ctx)).?;
    defer g_logits.deinit();
    // d/dx = (sigmoid(x) - y)/2 = {0.25, -0.059601461}.
    try expectCloseSlices(&.{ 0.25, -0.05960146101105877 }, g_logits.asRawTensor().dataConst(), 1e-6);

    // torch F.kl_div(x, t, reduction='sum'): x = ln{0.7, 0.3}, t = {0.5, 0.5}:
    // sum t·(ln t - x) = 0.5·(ln 0.5 - ln 0.7) + 0.5·(ln 0.5 - ln 0.3).
    var logp = try V.variableFromSlice(&ctx, .{2}, &.{ -0.35667494393873245, -1.2039728043259361 });
    defer logp.deinit();
    var kl_target = try V.fromSlice(&ctx, .{2}, &.{ 0.5, 0.5 });
    defer kl_target.deinit();
    var kl = try logp.klDivLoss(&ctx, &kl_target, .{ .reduction = .sum });
    defer kl.deinit();
    try std.testing.expectApproxEqAbs(0.08717669357238897, try kl.item(), 1e-6);
    try kl.backward(&ctx);
    var g_logp = (try logp.grad(&ctx)).?;
    defer g_logp.deinit();
    // d/dx = -t.
    try expectCloseSlices(&.{ -0.5, -0.5 }, g_logp.asRawTensor().dataConst(), 1e-6);
}

test "public Tensor nllLoss composes -logp[label] with reductions and gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const T = Tensor(.{ .pos, .class });
    const logp = [_]f32{ -1, -2, -0.5, -0.3, -1.2, -2.3 };
    const labels = [_]usize{ 2, 0 };

    // No-grad values work unscoped (the intermediates are freed eagerly).
    var c = try T.fromSlice(&ctx, .{ 2, 3 }, &logp);
    defer c.deinit();

    // torch F.nll_loss on log-probs: mean of {-x[0,2], -x[1,0]} = (0.5 + 0.3)/2.
    var mean_loss = try c.nllLoss(&ctx, .class, &labels, .mean);
    defer mean_loss.deinit();
    try std.testing.expectApproxEqAbs(0.4, try mean_loss.item(), 1e-6);

    var sum_loss = try c.nllLoss(&ctx, .class, &labels, .sum);
    defer sum_loss.deinit();
    try std.testing.expectApproxEqAbs(0.8, try sum_loss.item(), 1e-6);

    // `.none` removes the class tag: per-position losses {0.5, 0.3}.
    var none_loss = try c.nllLoss(&ctx, .class, &labels, .none);
    defer none_loss.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, none_loss.asRawTensor().shape.slice());
    try expectCloseSlices(&.{ 0.5, 0.3 }, try none_loss.dataConst(), 1e-6);

    // Validation: label count and range.
    try std.testing.expectError(error.InvalidDataLength, c.nllLoss(&ctx, .class, &.{2}, .mean));
    try std.testing.expectError(error.IndexOutOfBounds, c.nllLoss(&ctx, .class, &.{ 3, 0 }, .mean));

    // Grad tracking without an exec scope is a LOUD error (the composed
    // intermediates would dangle) — the training pattern below is scoped.
    var x = try T.variableFromSlice(&ctx, .{ 2, 3 }, &logp);
    defer x.deinit();
    try std.testing.expectError(error.ActiveExecScopeRequired, x.nllLoss(&ctx, .class, &labels, .mean));

    // mean gradient: -onehot/positions.
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try x.nllLoss(&ctx, .class, &labels, .mean);
        defer loss.deinit();
        try std.testing.expectApproxEqAbs(0.4, try loss.item(), 1e-6);
        try loss.backward(&ctx);
    }
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 0, 0, -0.5, -0.5, 0, 0 }, gx.asRawTensor().dataConst(), 1e-6);
}
