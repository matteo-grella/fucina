//! Softmax family: row-wise softmax backward, softmaxExt masks, sinks, and
//! ALiBi slopes, logsumexp/logSoftmax, and ggml-inspired axis coverage for
//! softmax, rmsnorm, and cross entropy.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const ag_tensor = @import("../tensor.zig");
const gradcheck_mod = @import("../gradcheck.zig");

const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tensor = ag_tensor.Tensor;
const RawTensor = @import("../../tensor.zig").Tensor;

const util = @import("util.zig");
const expectCloseSlices = util.expectCloseSlices;
const expectedSoftmaxExtProbs = util.expectedSoftmaxExtProbs;
const expectedSoftmaxExtWeighted = util.expectedSoftmaxExtWeighted;

test "tagged autograd softmax backward follows stable row-wise VJP" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 2, 4, 8 });
    defer x.deinit();
    var w = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0.5, -1, 2, 1, 0, -0.5 });
    defer w.deinit();

    var y = try x.softmax(&ctx, .d, .{});
    defer y.deinit();
    var weighted = try y.mul(&ctx, &w);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();

    var expected: [6]f32 = undefined;
    expectedSoftmaxWeightedGrad(.{ 1, 2, 3 }, .{ 0.5, -1, 2 }, expected[0..3]);
    expectedSoftmaxWeightedGrad(.{ 2, 4, 8 }, .{ 1, 0, -0.5 }, expected[3..6]);
    try expectCloseSlices(&expected, grad.asRawTensor().dataConst(), 1e-5);
}

test "tagged autograd softmaxExt applies scaled additive masks and gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 2, 4, 8 });
    defer x.deinit();
    var mask = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0, -1, 0.5, -0.5, 0, -2 });
    defer mask.deinit();
    var w = try Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0.5, -1, 2, 1, 0, -0.5 });
    defer w.deinit();

    var y = try x.softmax(&ctx, .d, .{ .mask = &mask, .scale = 0.5 });
    defer y.deinit();
    var weighted = try y.mul(&ctx, &w);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var expected_y: [6]f32 = undefined;
    var expected_grad: [6]f32 = undefined;
    expectedSoftmaxExtWeighted(.{ 1, 2, 3 }, .{ 0, -1, 0.5 }, .{ 0.5, -1, 2 }, 0.5, 1, null, expected_y[0..3], expected_grad[0..3]);
    expectedSoftmaxExtWeighted(.{ 2, 4, 8 }, .{ -0.5, 0, -2 }, .{ 1, 0, -0.5 }, 0.5, 1, null, expected_y[3..6], expected_grad[3..6]);
    try expectCloseSlices(&expected_y, y.asRawTensor().dataConst(), 1e-6);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(&expected_grad, grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd softmaxExt applies causal masks and gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .seq, .src }).variableFromSlice(&ctx, .{ 3, 3 }, &.{
        1, 2, 3,
        2, 4, 8,
        3, 1, 0,
    });
    defer x.deinit();
    var w = try Tensor(.{ .seq, .src }).fromSlice(&ctx, .{ 3, 3 }, &.{
        0.5, -1,   2,
        1,   0,    -0.5,
        -1,  0.25, 2,
    });
    defer w.deinit();

    var y = try x.softmax(&ctx, .src, .{ .causal = .{ .query_tag = .seq }, .scale = 0.5 });
    defer y.deinit();
    var weighted = try y.mul(&ctx, &w);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    const neg_inf = -std.math.inf(f32);
    var expected_y: [9]f32 = undefined;
    var expected_grad: [9]f32 = undefined;
    expectedSoftmaxExtWeighted(.{ 1, 2, 3 }, .{ 0, neg_inf, neg_inf }, .{ 0.5, -1, 2 }, 0.5, 1, null, expected_y[0..3], expected_grad[0..3]);
    expectedSoftmaxExtWeighted(.{ 2, 4, 8 }, .{ 0, 0, neg_inf }, .{ 1, 0, -0.5 }, 0.5, 1, null, expected_y[3..6], expected_grad[3..6]);
    expectedSoftmaxExtWeighted(.{ 3, 1, 0 }, .{ 0, 0, 0 }, .{ -1, 0.25, 2 }, 0.5, 1, null, expected_y[6..9], expected_grad[6..9]);
    try expectCloseSlices(&expected_y, y.asRawTensor().dataConst(), 1e-6);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(&expected_grad, grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd softmaxExt applies causal source offsets and gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .seq, .src }).variableFromSlice(&ctx, .{ 2, 4 }, &.{
        1, 2, 3, 4,
        2, 4, 8, 1,
    });
    defer x.deinit();
    var w = try Tensor(.{ .seq, .src }).fromSlice(&ctx, .{ 2, 4 }, &.{
        0.5, -1, 2,    3,
        1,   0,  -0.5, 2,
    });
    defer w.deinit();

    var y = try x.softmax(&ctx, .src, .{ .causal = .{ .query_tag = .seq, .source_offset = 2 }, .scale = 0.5 });
    defer y.deinit();
    var weighted = try y.mul(&ctx, &w);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    const neg_inf = -std.math.inf(f32);
    var expected_y: [8]f32 = undefined;
    var expected_grad: [8]f32 = undefined;
    expectedSoftmaxExtWeighted(.{ 1, 2, 3, 4 }, .{ 0, 0, 0, neg_inf }, .{ 0.5, -1, 2, 3 }, 0.5, 1, null, expected_y[0..4], expected_grad[0..4]);
    expectedSoftmaxExtWeighted(.{ 2, 4, 8, 1 }, .{ 0, 0, 0, 0 }, .{ 1, 0, -0.5, 2 }, 0.5, 1, null, expected_y[4..8], expected_grad[4..8]);
    try expectCloseSlices(&expected_y, y.asRawTensor().dataConst(), 1e-6);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(&expected_grad, grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd softmaxExt supports sink denominator mass" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .head, .key }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 0, 0, 0, 0, 0, 0 });
    defer x.deinit();
    var sinks = [_]f32{ 0, @log(@as(f32, 3)) };

    var y = try x.softmax(&ctx, .key, .{ .sinks = sinks[0..], .head_tag = .head });
    defer y.deinit();
    try expectCloseSlices(&.{ 0.25, 0.25, 0.25, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0 }, y.asRawTensor().dataConst(), 1e-6);

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var grad = (try x.grad(&ctx)).?;
    defer grad.deinit();
    try expectCloseSlices(&.{ 0.0625, 0.0625, 0.0625, 1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0 }, grad.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd softmaxExt applies ggml-style ALiBi max_bias to broadcast masks" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .head, .key }).variableFromSlice(&ctx, .{ 3, 4 }, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    defer x.deinit();
    var mask = try Tensor(.{.key}).fromSlice(&ctx, .{4}, &.{ 0, -1, -2, -3 });
    defer mask.deinit();

    var y = try x.softmax(&ctx, .key, .{ .mask = &mask, .max_bias = 8.0, .head_tag = .head });
    defer y.deinit();

    var expected: [12]f32 = undefined;
    expectedSoftmaxExtProbs(.{ 0, 0, 0, 0 }, .{ 0, -1, -2, -3 }, 1, 1.0 / 16.0, null, expected[0..4]);
    expectedSoftmaxExtProbs(.{ 0, 0, 0, 0 }, .{ 0, -1, -2, -3 }, 1, 1.0 / 256.0, null, expected[4..8]);
    expectedSoftmaxExtProbs(.{ 0, 0, 0, 0 }, .{ 0, -1, -2, -3 }, 1, 1.0 / 4.0, null, expected[8..12]);
    try expectCloseSlices(&expected, y.asRawTensor().dataConst(), 1e-6);
}

test "tagged autograd softmaxExt rejects differentiable masks" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 1, 3 }, &.{ 1, 2, 3 });
    defer x.deinit();
    var mask = try Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 1, 3 }, &.{ 0, -1, 0 });
    defer mask.deinit();

    try std.testing.expectError(error.UnsupportedGradient, x.softmax(&ctx, .d, .{ .mask = &mask }));
}

test "tagged ggml-inspired axis coverage for softmax rmsnorm and cross entropy" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var logits = try Tensor(.{ .vocab, .token, .batch }).variableFromSlice(
        &ctx,
        .{ 4, 2, 2 },
        &.{
            1, 2,  3,  4,
            2, 0,  -1, 3,
            0, -2, 1,  2,
            4, 3,  2,  1,
        },
    );
    defer logits.deinit();

    var probs = try logits.softmax(&ctx, .vocab, .{});
    defer probs.deinit();
    try expectSoftmaxAxisSumsClose(probs.asRawTensor().dataConst(), 4, 4, 1e-6);

    var loss = try logits.crossEntropy(&ctx, .vocab, &.{ 3, 0, 2, 1 }, .{});
    defer loss.deinit();
    try loss.backward(&ctx);
    var grad = (try logits.grad(&ctx)).?;
    defer grad.deinit();
    try expectCrossEntropyGradAxis0SumsClose(grad.asRawTensor().dataConst(), 4, 4, 1e-6);

    var x = try Tensor(.{ .d, .token, .batch }).variableFromSlice(
        &ctx,
        .{ 4, 2, 1 },
        &.{ 1, 2, 3, 4, -1, -2, -3, -4 },
    );
    defer x.deinit();
    var y = try x.rmsNorm(&ctx, .d, 1e-6);
    defer y.deinit();
    try expectRmsNormAxisMeanSquareClose(y.asRawTensor().dataConst(), 4, 2, 1, 1e-5);
}

fn expectSoftmaxAxisSumsClose(values: []const f32, class_count: usize, inner: usize, tolerance: f32) !void {
    try std.testing.expectEqual(@as(usize, 0), values.len % (class_count * inner));
    const outer = values.len / (class_count * inner);
    for (0..outer) |outer_i| {
        const base = outer_i * class_count * inner;
        for (0..inner) |inner_i| {
            var sum: f32 = 0;
            for (0..class_count) |class_i| {
                sum += values[base + class_i * inner + inner_i];
            }
            try std.testing.expectApproxEqAbs(@as(f32, 1), sum, tolerance);
        }
    }
}

fn expectCrossEntropyGradAxis0SumsClose(values: []const f32, class_count: usize, inner: usize, tolerance: f32) !void {
    try std.testing.expectEqual(@as(usize, 0), values.len % (class_count * inner));
    const outer = values.len / (class_count * inner);
    for (0..outer) |outer_i| {
        const base = outer_i * class_count * inner;
        for (0..inner) |inner_i| {
            var sum: f32 = 0;
            for (0..class_count) |class_i| {
                sum += values[base + class_i * inner + inner_i];
            }
            try std.testing.expectApproxEqAbs(@as(f32, 0), sum, tolerance);
        }
    }
}

fn expectRmsNormAxisMeanSquareClose(values: []const f32, axis_dim: usize, inner: usize, expected: f32, tolerance: f32) !void {
    try std.testing.expectEqual(@as(usize, 0), values.len % (axis_dim * inner));
    const outer = values.len / (axis_dim * inner);
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const value = values[base + axis_i * inner + inner_i];
                sumsq += value * value;
            }
            try std.testing.expectApproxEqAbs(expected, sumsq / @as(f32, @floatFromInt(axis_dim)), tolerance);
        }
    }
}

fn expectedSoftmaxWeightedGrad(comptime logits: [3]f32, comptime weights: [3]f32, out: []f32) void {
    var max_value = logits[0];
    inline for (1..3) |i| max_value = @max(max_value, logits[i]);

    var probs: [3]f32 = undefined;
    var sum_exp: f32 = 0;
    inline for (0..3) |i| {
        probs[i] = @exp(logits[i] - max_value);
        sum_exp += probs[i];
    }
    inline for (0..3) |i| probs[i] /= sum_exp;

    var dot: f32 = 0;
    inline for (0..3) |i| dot += probs[i] * weights[i];
    inline for (0..3) |i| out[i] = probs[i] * (weights[i] - dot);
}

test "public Tensor logsumexp and logSoftmax match the analytic form" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const M = Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ 0, @log(@as(f32, 3)), 1, 1 });
    defer x.deinit();

    // Row 0: log(1 + 3) = log 4; row 1: log(2e) = 1 + log 2.
    var lse = try x.logsumexp(&ctx, .col);
    defer lse.deinit();
    try expectCloseSlices(&.{ @log(@as(f32, 4)), 1 + @log(@as(f32, 2)) }, try lse.dataConst(), 1e-6);

    var lsm = try x.logSoftmax(&ctx, .col);
    defer lsm.deinit();
    // Row 0 probabilities {0.25, 0.75}; row 1 {0.5, 0.5}.
    try expectCloseSlices(&.{ @log(@as(f32, 0.25)), @log(@as(f32, 0.75)), @log(@as(f32, 0.5)), @log(@as(f32, 0.5)) }, try lsm.dataConst(), 1e-6);

    // Non-finite rows follow torch: all -inf → -inf, a +inf entry → +inf.
    const inf = std.math.inf(f32);
    var edge = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ -inf, -inf, 0, inf });
    defer edge.deinit();
    var edge_lse = try edge.logsumexp(&ctx, .col);
    defer edge_lse.deinit();
    try std.testing.expect(std.math.isNegativeInf((try edge_lse.dataConst())[0]));
    try std.testing.expect(std.math.isPositiveInf((try edge_lse.dataConst())[1]));

    // d(logsumexp)/dx is the softmax (the shift's gradient cancels).
    var xv = try M.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0, @log(@as(f32, 3)), 1, 1 });
    defer xv.deinit();
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var lv = try xv.logsumexp(&ctx, .col);
        defer lv.deinit();
        var loss = try lv.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gx = (try xv.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 0.25, 0.75, 0.5, 0.5 }, try gx.dataConst(), 1e-6);
}
