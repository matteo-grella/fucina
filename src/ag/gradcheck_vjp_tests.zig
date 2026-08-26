//! Table-driven finite-difference checks over the VJP records that have no
//! other gradcheck in the tree: one small f32 case per record family,
//! every op output weighted by a fixed asymmetric coefficient pattern
//! before the scalar sum (an index-routing mistake changes the gradient
//! instead of hiding under an all-ones upstream). The harness
//! (`gradcheck.zig`) opens an exec scope around every forward, so composed
//! intermediates are scope-owned. Ops with kinks (relu, leakyRelu, clamp,
//! min/max, topK) take inputs kept away from the kink so the central
//! difference never straddles it.
const std = @import("std");
const exec_mod = @import("../exec.zig");
const gradcheck_mod = @import("gradcheck.zig");
const ag_tensor = @import("tensor.zig");

const ExecContext = exec_mod.ExecContext;
const Tensor = ag_tensor.Tensor;
const Options = gradcheck_mod.Options;

// --- fixtures ---------------------------------------------------------------

const max_elems = 128;

/// Deterministic smooth fill in roughly [-0.75, 0.75]; `phase` keeps
/// operands distinct.
fn fill(values: []f32, phase: f32) void {
    for (values, 0..) |*v, i| {
        const x = @as(f32, @floatFromInt(i)) + phase;
        v.* = @sin(x * 0.7) * 0.5 + @cos(x * 0.3) * 0.25;
    }
}

/// `fill`, then every magnitude pushed to at least `margin` (sign kept):
/// inputs for ops with a kink at zero (relu, leakyRelu) or a pole
/// (cumprod, prod).
fn fillAway(values: []f32, phase: f32, margin: f32) void {
    fill(values, phase);
    for (values) |*v| {
        const sign: f32 = if (v.* < 0) -1 else 1;
        v.* = sign * (@abs(v.*) + margin);
    }
}

/// Fixed asymmetric coefficients: alternating sign, magnitudes 0.5..1.28,
/// no two neighbours alike.
fn coef(i: usize) f32 {
    const sign: f32 = if (i % 2 == 0) 1 else -1;
    return sign * (0.5 + 0.13 * @as(f32, @floatFromInt((i * 5) % 7)));
}

/// `sum(y * coef)` for any f32 facade tensor.
fn weightedSum(ctx: *ExecContext, y: anytype) !Tensor(.{}) {
    const Y = @TypeOf(y.*);
    const len = y.asRawTensor().len();
    var coefs: [max_elems]f32 = undefined;
    for (0..len) |i| coefs[i] = coef(i);
    var w = try Y.fromSlice(ctx, y.shape(), coefs[0..len]);
    defer w.deinit();
    var z = try y.mul(ctx, &w);
    defer z.deinit();
    return z.sumAll(ctx);
}

fn variable(ctx: *ExecContext, comptime T: type, shape: [T.axis_tags.len]usize, phase: f32) !T {
    var values: [max_elems]f32 = undefined;
    var len: usize = 1;
    for (shape) |d| len *= d;
    fill(values[0..len], phase);
    return T.variableFromSlice(ctx, shape, values[0..len]);
}

fn variableAway(ctx: *ExecContext, comptime T: type, shape: [T.axis_tags.len]usize, phase: f32, margin: f32) !T {
    var values: [max_elems]f32 = undefined;
    var len: usize = 1;
    for (shape) |d| len *= d;
    fillAway(values[0..len], phase, margin);
    return T.variableFromSlice(ctx, shape, values[0..len]);
}

fn check(ctx: *ExecContext, comptime loss_fn: anytype, inputs: anytype, expected_checked: usize, options: Options) !void {
    const result = try gradcheck_mod.gradcheck(ctx, loss_fn, inputs, options);
    try std.testing.expectEqual(expected_checked, result.checked);
}

const M = Tensor(.{ .row, .col });
const V = Tensor(.{.col});

// --- softmax family ---------------------------------------------------------

fn softmaxLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.softmax(ctx, .col, .{});
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn softmaxExtLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    // Additive mask: one fully masked lane per row, one mildly biased.
    var mask = try M.fromSlice(ctx, .{ 3, 4 }, &.{ 0, -0.7, 0, -1e9, -1e9, 0, 0.4, 0, 0, 0, -0.3, -1e9 });
    defer mask.deinit();
    var y = try x.softmax(ctx, .col, .{ .scale = 0.7, .mask = &mask });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn logSoftmaxLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.logSoftmax(ctx, .col);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn logsumexpLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.logsumexp(ctx, .col);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn runSoftmax(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 0.1);
    defer x.deinit();
    try check(ctx, softmaxLoss, .{&x}, 12, .{});
}

fn runSoftmaxExt(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 0.2);
    defer x.deinit();
    try check(ctx, softmaxExtLoss, .{&x}, 12, .{});
}

fn runLogSoftmax(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 0.3);
    defer x.deinit();
    try check(ctx, logSoftmaxLoss, .{&x}, 12, .{});
}

fn runLogsumexp(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 0.4);
    defer x.deinit();
    try check(ctx, logsumexpLoss, .{&x}, 12, .{});
}

// --- rms norm family ----------------------------------------------------------

fn rmsNormLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.rmsNorm(ctx, .col, 1e-5);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn rmsNormMulLoss(ctx: *ExecContext, x: *const M, w: *const V) !Tensor(.{}) {
    var y = try x.rmsNormMul(ctx, .col, w, 1e-5);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn rmsNormMulAddLoss(ctx: *ExecContext, x: *const M, w: *const V, residual: *const M) !Tensor(.{}) {
    var y = try x.rmsNormMulAdd(ctx, .col, w, residual, 1e-5);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

const S = Tensor(.{ .seq, .d });
const D = Tensor(.{.d});

fn rmsNormMulRopeLoss(ctx: *ExecContext, x: *const S, w: *const D) !Tensor(.{}) {
    var table = try ctx.prepareRopeTable(.{ .positions = .{ .explicit = &.{ 0, 3, 5 } }, .feature_dim = 4, .freqs = .{ .theta = .{ .base = 100 } } });
    defer table.deinit();
    var y = try x.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, w, 1e-5, &table);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn runRmsNorm(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 0.5);
    defer x.deinit();
    try check(ctx, rmsNormLoss, .{&x}, 12, .{});
}

fn runRmsNormMul(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 0.6);
    defer x.deinit();
    var w = try variable(ctx, V, .{4}, 1.1);
    defer w.deinit();
    try check(ctx, rmsNormMulLoss, .{ &x, &w }, 16, .{});
}

fn runRmsNormMulAdd(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 0.7);
    defer x.deinit();
    var w = try variable(ctx, V, .{4}, 1.2);
    defer w.deinit();
    var residual = try variable(ctx, M, .{ 3, 4 }, 2.3);
    defer residual.deinit();
    try check(ctx, rmsNormMulAddLoss, .{ &x, &w, &residual }, 28, .{});
}

fn runRmsNormMulRope(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, S, .{ 3, 4 }, 0.8);
    defer x.deinit();
    var w = try variable(ctx, D, .{4}, 1.3);
    defer w.deinit();
    try check(ctx, rmsNormMulRopeLoss, .{ &x, &w }, 16, .{});
}

// --- rope (on-the-fly factors) ------------------------------------------------

const rope_positions = [_]i32{ 0, 2, 7 };

fn ropeHalfLoss(ctx: *ExecContext, x: *const S) !Tensor(.{}) {
    var y = try x.rope(ctx, .seq, .d, .{ .positions = &rope_positions, .theta_base = 100 }, .half);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn ropeInterleavedLoss(ctx: *ExecContext, x: *const S) !Tensor(.{}) {
    var y = try x.rope(ctx, .seq, .d, .{ .positions = &rope_positions, .theta_base = 100 }, .interleaved);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn runRope(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, S, .{ 3, 4 }, 0.9);
    defer x.deinit();
    try check(ctx, ropeHalfLoss, .{&x}, 12, .{});
    try check(ctx, ropeInterleavedLoss, .{&x}, 12, .{});
}

// --- contractions -------------------------------------------------------------

const A = Tensor(.{ .m, .k });
const B = Tensor(.{ .k, .n });
const Bt = Tensor(.{ .n, .k });
const C = Tensor(.{ .m, .n });
const BA = Tensor(.{ .b, .m, .k });
const BB = Tensor(.{ .b, .k, .n });
const BBt = Tensor(.{ .b, .n, .k });
const BAt = Tensor(.{ .b, .k, .m });

fn matmulPlainLoss(ctx: *ExecContext, a: *const A, b: *const B) !Tensor(.{}) {
    var y = try a.matmul(ctx, b, .plain, .{ .m, .n });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn matmulTransBLoss(ctx: *ExecContext, a: *const A, b: *const Bt) !Tensor(.{}) {
    var y = try a.matmul(ctx, b, .trans_b, .{ .m, .n });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn bmmPlainLoss(ctx: *ExecContext, a: *const BA, b: *const BB) !Tensor(.{}) {
    var y = try a.matmul(ctx, b, .plain, .{ .b, .m, .n });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn bmmTransBLoss(ctx: *ExecContext, a: *const BA, b: *const BBt) !Tensor(.{}) {
    var y = try a.matmul(ctx, b, .trans_b, .{ .b, .m, .n });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn bmmTransALoss(ctx: *ExecContext, a: *const BAt, b: *const BB) !Tensor(.{}) {
    var y = try a.matmul(ctx, b, .trans_a, .{ .b, .m, .n });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn bmmBroadcastRhsLoss(ctx: *ExecContext, a: *const BA, b: *const B) !Tensor(.{}) {
    var y = try a.matmul(ctx, b, .plain, .{ .b, .m, .n });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn addDotLoss(ctx: *ExecContext, base: *const C, a: *const A, b: *const B) !Tensor(.{}) {
    var y = try base.addDot(ctx, a, b, .k);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn dotLoss(ctx: *ExecContext, a: *const A, b: *const B) !Tensor(.{}) {
    var y = try a.dot(ctx, b, .k);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn dotBatchedLoss(ctx: *ExecContext, a: *const BA, b: *const BB) !Tensor(.{}) {
    var y = try a.dot(ctx, b, .k);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn runMatmul2D(ctx: *ExecContext) anyerror!void {
    var a = try variable(ctx, A, .{ 3, 4 }, 1.0);
    defer a.deinit();
    var b = try variable(ctx, B, .{ 4, 2 }, 2.0);
    defer b.deinit();
    try check(ctx, matmulPlainLoss, .{ &a, &b }, 20, .{});
    var bt = try variable(ctx, Bt, .{ 2, 4 }, 2.5);
    defer bt.deinit();
    try check(ctx, matmulTransBLoss, .{ &a, &bt }, 20, .{});
}

fn runBmm(ctx: *ExecContext) anyerror!void {
    var a = try variable(ctx, BA, .{ 2, 3, 4 }, 1.1);
    defer a.deinit();
    var b = try variable(ctx, BB, .{ 2, 4, 2 }, 2.1);
    defer b.deinit();
    try check(ctx, bmmPlainLoss, .{ &a, &b }, 40, .{});
    var bt = try variable(ctx, BBt, .{ 2, 2, 4 }, 2.6);
    defer bt.deinit();
    try check(ctx, bmmTransBLoss, .{ &a, &bt }, 40, .{});
    var at = try variable(ctx, BAt, .{ 2, 4, 3 }, 1.6);
    defer at.deinit();
    try check(ctx, bmmTransALoss, .{ &at, &b }, 40, .{});
    var shared = try variable(ctx, B, .{ 4, 2 }, 3.1);
    defer shared.deinit();
    try check(ctx, bmmBroadcastRhsLoss, .{ &a, &shared }, 32, .{});
}

fn runAddDot(ctx: *ExecContext) anyerror!void {
    var base = try variable(ctx, C, .{ 3, 2 }, 0.4);
    defer base.deinit();
    var a = try variable(ctx, A, .{ 3, 4 }, 1.2);
    defer a.deinit();
    var b = try variable(ctx, B, .{ 4, 2 }, 2.2);
    defer b.deinit();
    try check(ctx, addDotLoss, .{ &base, &a, &b }, 26, .{});
}

fn runDot(ctx: *ExecContext) anyerror!void {
    var a = try variable(ctx, A, .{ 3, 4 }, 1.3);
    defer a.deinit();
    var b = try variable(ctx, B, .{ 4, 2 }, 2.3);
    defer b.deinit();
    try check(ctx, dotLoss, .{ &a, &b }, 20, .{});
    var ba = try variable(ctx, BA, .{ 2, 3, 4 }, 1.4);
    defer ba.deinit();
    var bb = try variable(ctx, BB, .{ 2, 4, 2 }, 2.4);
    defer bb.deinit();
    try check(ctx, dotBatchedLoss, .{ &ba, &bb }, 40, .{});
}

// --- pointwise ------------------------------------------------------------------

fn addScalarLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.addScalar(ctx, 0.75);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn reluLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.relu(ctx);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn leakyReluLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.leakyRelu(ctx, 0.1);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn softcapLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.softcap(ctx, 2.0);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn clampLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.clamp(ctx, -0.5, 0.5);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn dropoutLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.dropout(ctx, 0.5, 42);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn runAddScalar(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 1.5);
    defer x.deinit();
    try check(ctx, addScalarLoss, .{&x}, 12, .{});
}

fn runRelu(ctx: *ExecContext) anyerror!void {
    var x = try variableAway(ctx, M, .{ 3, 4 }, 1.6, 0.3);
    defer x.deinit();
    try check(ctx, reluLoss, .{&x}, 12, .{});
}

fn runLeakyRelu(ctx: *ExecContext) anyerror!void {
    var x = try variableAway(ctx, M, .{ 3, 4 }, 1.7, 0.3);
    defer x.deinit();
    try check(ctx, leakyReluLoss, .{&x}, 12, .{});
}

fn runSoftcap(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 1.8);
    defer x.deinit();
    try check(ctx, softcapLoss, .{&x}, 12, .{});
}

fn runClamp(ctx: *ExecContext) anyerror!void {
    // Values inside (-0.5, 0.5) and outside, none within 0.1 of the bounds.
    var x = try M.variableFromSlice(ctx, .{ 3, 4 }, &.{ 0.2, -0.3, 0.8, -0.9, 0.1, 0.65, -0.15, -0.7, 0.35, 1.2, -1.1, 0.05 });
    defer x.deinit();
    try check(ctx, clampLoss, .{&x}, 12, .{});
}

fn runDropout(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 1.9);
    defer x.deinit();
    try check(ctx, dropoutLoss, .{&x}, 12, .{});
}

// --- gated units ------------------------------------------------------------------

fn splitSwiGluLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.splitGated(ctx, .swiglu, .col, .h);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn splitGluLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.splitGated(ctx, .glu, .col, .h);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn swigluLoss(ctx: *ExecContext, a: *const M, b: *const M) !Tensor(.{}) {
    var y = try a.swiglu(ctx, b);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn gluLoss(ctx: *ExecContext, a: *const M, b: *const M) !Tensor(.{}) {
    var y = try a.glu(ctx, b);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn gegluLoss(ctx: *ExecContext, a: *const M, b: *const M) !Tensor(.{}) {
    var y = try a.geglu(ctx, b);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn situLoss(ctx: *ExecContext, a: *const M, b: *const M) !Tensor(.{}) {
    var y = try a.situ(ctx, b);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn runSplitGated(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 2.0);
    defer x.deinit();
    try check(ctx, splitSwiGluLoss, .{&x}, 12, .{});
    try check(ctx, splitGluLoss, .{&x}, 12, .{});
}

fn runGated(ctx: *ExecContext) anyerror!void {
    var a = try variable(ctx, M, .{ 3, 4 }, 2.1);
    defer a.deinit();
    var b = try variable(ctx, M, .{ 3, 4 }, 3.2);
    defer b.deinit();
    try check(ctx, swigluLoss, .{ &a, &b }, 24, .{});
    try check(ctx, gluLoss, .{ &a, &b }, 24, .{});
    try check(ctx, gegluLoss, .{ &a, &b }, 24, .{});
    try check(ctx, situLoss, .{ &a, &b }, 24, .{});
}

// --- reductions and scans ---------------------------------------------------------

fn cumprodLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.cumprod(ctx, .col);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn prodLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.prod(ctx, .col);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn maxLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.max(ctx, .col, .{});
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn minLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.min(ctx, .col, .{});
    defer y.deinit();
    return weightedSum(ctx, &y);
}

const BoolM = Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
const reduce_mask = [_]bool{ true, false, true, true, false, true, false, false, true, true, true, false };

fn maskedMaxLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var mask = try BoolM.fromSlice(ctx, .{ 3, 4 }, &reduce_mask);
    defer mask.deinit();
    var y = try x.max(ctx, .col, .{ .mask = &mask });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn maskedMinLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var mask = try BoolM.fromSlice(ctx, .{ 3, 4 }, &reduce_mask);
    defer mask.deinit();
    var y = try x.min(ctx, .col, .{ .mask = &mask });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn topKLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var result = try x.topK(ctx, .col, 2, .k);
    defer result.deinit();
    return weightedSum(ctx, &result.values);
}

/// Distinct values, pairwise at least 0.1 apart: the ±1e-3 perturbation
/// never changes an extremum or a top-k selection.
const distinct_values = [_]f32{ 0.3, -0.6, 0.9, 0.1, -0.2, 0.7, -0.8, 0.5, 0.4, -0.4, 0.2, 0.6 };

fn runCumprod(ctx: *ExecContext) anyerror!void {
    var x = try variableAway(ctx, M, .{ 3, 4 }, 2.2, 0.3);
    defer x.deinit();
    try check(ctx, cumprodLoss, .{&x}, 12, .{});
}

fn runProd(ctx: *ExecContext) anyerror!void {
    var x = try variableAway(ctx, M, .{ 3, 4 }, 2.3, 0.3);
    defer x.deinit();
    try check(ctx, prodLoss, .{&x}, 12, .{});
}

fn runMinMax(ctx: *ExecContext) anyerror!void {
    var x = try M.variableFromSlice(ctx, .{ 3, 4 }, &distinct_values);
    defer x.deinit();
    try check(ctx, maxLoss, .{&x}, 12, .{});
    try check(ctx, minLoss, .{&x}, 12, .{});
}

fn runMaskedMinMax(ctx: *ExecContext) anyerror!void {
    var x = try M.variableFromSlice(ctx, .{ 3, 4 }, &distinct_values);
    defer x.deinit();
    try check(ctx, maskedMaxLoss, .{&x}, 12, .{});
    try check(ctx, maskedMinLoss, .{&x}, 12, .{});
}

fn runTopK(ctx: *ExecContext) anyerror!void {
    var x = try M.variableFromSlice(ctx, .{ 3, 4 }, &distinct_values);
    defer x.deinit();
    try check(ctx, topKLoss, .{&x}, 12, .{});
}

// --- functional row/slice updates ---------------------------------------------------

const index_add_rows = [_]usize{ 1, 3, 1 };
const set_rows = [_]usize{ 0, 2 };
const zero_rows = [_]usize{ 1, 3 };

fn indexAddLoss(ctx: *ExecContext, x: *const M, update: *const M) !Tensor(.{}) {
    var y = try x.indexAdd(ctx, .row, &index_add_rows, update);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn setRowsLoss(ctx: *ExecContext, x: *const M, update: *const M) !Tensor(.{}) {
    var y = try x.setRows(ctx, .row, &set_rows, update);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn setSliceLoss(ctx: *ExecContext, x: *const M, update: *const M) !Tensor(.{}) {
    var y = try x.setSlice(ctx, .row, 1, update);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn zeroRowsLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.zeroRows(ctx, .row, &zero_rows);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn zeroSliceLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.zeroSlice(ctx, .row, 1, 2);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn runIndexAdd(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 4, 3 }, 2.4);
    defer x.deinit();
    var update = try variable(ctx, M, .{ 3, 3 }, 3.4);
    defer update.deinit();
    try check(ctx, indexAddLoss, .{ &x, &update }, 21, .{});
}

fn runSetRows(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 4, 3 }, 2.5);
    defer x.deinit();
    var update = try variable(ctx, M, .{ 2, 3 }, 3.5);
    defer update.deinit();
    try check(ctx, setRowsLoss, .{ &x, &update }, 18, .{});
}

fn runSetSlice(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 4, 3 }, 2.6);
    defer x.deinit();
    var update = try variable(ctx, M, .{ 2, 3 }, 3.6);
    defer update.deinit();
    try check(ctx, setSliceLoss, .{ &x, &update }, 18, .{});
}

fn runZeroRows(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 4, 3 }, 2.7);
    defer x.deinit();
    try check(ctx, zeroRowsLoss, .{&x}, 12, .{});
}

fn runZeroSlice(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 4, 3 }, 2.8);
    defer x.deinit();
    try check(ctx, zeroSliceLoss, .{&x}, 12, .{});
}

// --- views ------------------------------------------------------------------------------

fn broadcastLoss(ctx: *ExecContext, x: *const V) !Tensor(.{}) {
    var y = try x.broadcastTo(ctx, .{ .row, .col }, .{ 2, 3 });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn diagonalLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.diagonal(ctx, .row, .col, .d);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn mergeLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.merge(ctx, .flat, .{ .row, .col });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn splitLoss(ctx: *ExecContext, x: *const V) !Tensor(.{}) {
    var y = try x.split(ctx, .col, .{ .a, .b }, .{ 2, 3 });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn materializeLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var permuted = try x.permuteTo(ctx, .{ .col, .row });
    defer permuted.deinit();
    var y = try permuted.materialize(ctx);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn viewWithStridesLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    // [3, 4] row-major: shape {3}, stride {5} reads x[0,0], x[1,1], x[2,2].
    var y = try x.viewWithStrides(ctx, .{.v}, .{3}, .{5});
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn permuteLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.permuteTo(ctx, .{ .col, .row });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn insertAxisLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.insertAxis(ctx, .one, 1);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn contiguousLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.contiguous(ctx);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn biasAddLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.biasAdd(ctx, &.{ 0.5, -1.0, 0.25, 2.0 }, .col);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn runBroadcast(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, V, .{3}, 2.9);
    defer x.deinit();
    try check(ctx, broadcastLoss, .{&x}, 3, .{});
}

fn runStridedView(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 3.0);
    defer x.deinit();
    try check(ctx, diagonalLoss, .{&x}, 12, .{});
    try check(ctx, mergeLoss, .{&x}, 12, .{});
    try check(ctx, materializeLoss, .{&x}, 12, .{});
    try check(ctx, viewWithStridesLoss, .{&x}, 12, .{});
    var v = try variable(ctx, V, .{6}, 3.1);
    defer v.deinit();
    try check(ctx, splitLoss, .{&v}, 6, .{});
}

fn runAxisView(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 3.2);
    defer x.deinit();
    try check(ctx, permuteLoss, .{&x}, 12, .{});
    try check(ctx, insertAxisLoss, .{&x}, 12, .{});
}

fn runIdentity(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 3.3);
    defer x.deinit();
    try check(ctx, contiguousLoss, .{&x}, 12, .{});
    try check(ctx, biasAddLoss, .{&x}, 12, .{});
}

// --- casts ------------------------------------------------------------------------------

fn castF32Loss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var y = try x.to(ctx, .f32);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn castF16RoundTripLoss(ctx: *ExecContext, x: *const M) !Tensor(.{}) {
    var narrow = try x.to(ctx, .f16);
    defer narrow.deinit();
    var y = try narrow.to(ctx, .f32);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn runCast(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, M, .{ 3, 4 }, 3.4);
    defer x.deinit();
    try check(ctx, castF32Loss, .{&x}, 12, .{});
    // The narrow is a staircase in x (f16 spacing 2^-11 below 1): the
    // straight-through VJP is the slope of the staircase envelope, so the
    // central difference uses a step spanning many f16 ulps.
    var half = try M.variableFromSlice(ctx, .{ 3, 4 }, &.{ 0.55, 0.6, 0.7, 0.9, 0.52, 0.66, 0.78, 0.84, 0.58, 0.72, 0.95, 0.63 });
    defer half.deinit();
    try check(ctx, castF16RoundTripLoss, .{&half}, 12, .{ .eps = 0.05 });
}

// --- convolutions ------------------------------------------------------------------------

const T2 = Tensor(.{ .t, .in });
const W3 = Tensor(.{ .tap, .in, .out });
const Tc = Tensor(.{ .t, .c });
const Kd = Tensor(.{ .c, .tap });
const Wg = Tensor(.{ .tap, .ipg, .out });
const Img = Tensor(.{ .h, .w, .c });
const Patches = Tensor(.{ .p, .e });

fn causalConv1dLoss(ctx: *ExecContext, x: *const T2, w: *const W3) !Tensor(.{}) {
    var y = try x.causalConv1d(ctx, .t, .in, .tap, .out, w, 1, null);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn causalConv1dDilatedLoss(ctx: *ExecContext, x: *const T2, w: *const W3) !Tensor(.{}) {
    var y = try x.causalConv1d(ctx, .t, .in, .tap, .out, w, 2, null);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn causalDepthwiseConv1dLoss(ctx: *ExecContext, x: *const Tc, k: *const Kd) !Tensor(.{}) {
    var y = try x.causalDepthwiseConv1d(ctx, .t, .c, .tap, k, 1, null);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn groupedCausalConv1dLoss(ctx: *ExecContext, x: *const T2, w: *const Wg) !Tensor(.{}) {
    var y = try x.groupedCausalConv1d(ctx, .t, .in, .tap, .ipg, .out, w, 1, 2, null);
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn unfoldLoss(ctx: *ExecContext, x: *const Img) !Tensor(.{}) {
    var y = try x.unfold(ctx, .{ 2, 2 }, .{ 1, 1 }, .{ 0, 0 }, .{ .p, .e });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn unfoldPaddedLoss(ctx: *ExecContext, x: *const Img) !Tensor(.{}) {
    var y = try x.unfold(ctx, .{ 2, 2 }, .{ 2, 1 }, .{ 1, 1 }, .{ .p, .e });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn foldLoss(ctx: *ExecContext, x: *const Patches) !Tensor(.{}) {
    var y = try x.fold(ctx, .{ 3, 3 }, .{ 2, 2 }, .{ 1, 1 }, .{ 0, 0 }, .{ .h, .w, .c });
    defer y.deinit();
    return weightedSum(ctx, &y);
}

fn runCausalConv1d(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, T2, .{ 5, 2 }, 3.5);
    defer x.deinit();
    var w = try variable(ctx, W3, .{ 3, 2, 3 }, 4.5);
    defer w.deinit();
    try check(ctx, causalConv1dLoss, .{ &x, &w }, 28, .{});
    try check(ctx, causalConv1dDilatedLoss, .{ &x, &w }, 28, .{});
}

fn runCausalDepthwiseConv1d(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, Tc, .{ 5, 3 }, 3.6);
    defer x.deinit();
    var k = try variable(ctx, Kd, .{ 3, 2 }, 4.6);
    defer k.deinit();
    try check(ctx, causalDepthwiseConv1dLoss, .{ &x, &k }, 21, .{});
}

fn runGroupedCausalConv1d(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, T2, .{ 5, 4 }, 3.7);
    defer x.deinit();
    var w = try variable(ctx, Wg, .{ 2, 2, 4 }, 4.7);
    defer w.deinit();
    try check(ctx, groupedCausalConv1dLoss, .{ &x, &w }, 36, .{});
}

fn runUnfold(ctx: *ExecContext) anyerror!void {
    var x = try variable(ctx, Img, .{ 3, 3, 2 }, 3.8);
    defer x.deinit();
    try check(ctx, unfoldLoss, .{&x}, 18, .{});
    try check(ctx, unfoldPaddedLoss, .{&x}, 18, .{});
}

fn runFold(ctx: *ExecContext) anyerror!void {
    // [oH*oW = 4, kH*kW*C = 8] patches for a 3x3x2 image, 2x2 kernel, stride 1.
    var x = try variable(ctx, Patches, .{ 4, 8 }, 3.9);
    defer x.deinit();
    try check(ctx, foldLoss, .{&x}, 32, .{});
}

// --- the table ----------------------------------------------------------------------------

const Case = struct {
    /// The VJP record(s) the case exercises.
    name: []const u8,
    run: *const fn (*ExecContext) anyerror!void,
    /// A documented, reproduced gradient defect: the case is skipped (and
    /// printed) instead of failing the suite. Empty = the case runs.
    known: []const u8 = "",
};

const cases = [_]Case{
    .{ .name = "SoftmaxBackward", .run = runSoftmax },
    .{ .name = "SoftmaxExtBackward (scale + mask)", .run = runSoftmaxExt },
    .{ .name = "LogSoftmaxBackward", .run = runLogSoftmax },
    .{ .name = "LogsumexpBackward", .run = runLogsumexp },
    .{ .name = "RmsNormBackward", .run = runRmsNorm },
    .{ .name = "RmsNormMulBackward", .run = runRmsNormMul },
    .{ .name = "RmsNormMulAddBackward", .run = runRmsNormMulAdd },
    .{ .name = "RmsNormMulRopeBackward", .run = runRmsNormMulRope },
    .{ .name = "RopeBackward (on-the-fly, half + interleaved)", .run = runRope },
    .{ .name = "Matmul2DBackward (plain, trans_b)", .run = runMatmul2D },
    .{ .name = "BmmBackward (plain, trans_b, trans_a, broadcast rhs)", .run = runBmm },
    .{ .name = "AddDotBackward", .run = runAddDot },
    .{ .name = "DotBackward (2-D, batched)", .run = runDot },
    .{ .name = "AddScalarBackward", .run = runAddScalar },
    .{ .name = "ReluBackward", .run = runRelu },
    .{ .name = "LeakyReluBackward", .run = runLeakyRelu },
    .{ .name = "SoftcapBackward", .run = runSoftcap },
    .{ .name = "ClampBackward", .run = runClamp },
    .{ .name = "DropoutBackward (fixed seed)", .run = runDropout },
    .{ .name = "SplitSwiGluBackward / SplitGluBackward", .run = runSplitGated },
    .{ .name = "GatedBackward (swiglu, glu, geglu, situ)", .run = runGated },
    .{ .name = "CumprodBackward", .run = runCumprod },
    .{ .name = "ProdBackward", .run = runProd },
    .{ .name = "MinMaxBackward", .run = runMinMax },
    .{ .name = "MaskedMinMaxBackward", .run = runMaskedMinMax },
    .{ .name = "TopKBackward (values)", .run = runTopK },
    .{ .name = "IndexAddBackward", .run = runIndexAdd },
    .{ .name = "SetRowsBackward", .run = runSetRows },
    .{ .name = "SetSliceBackward", .run = runSetSlice },
    .{ .name = "ZeroRowsBackward", .run = runZeroRows },
    .{ .name = "ZeroSliceBackward", .run = runZeroSlice },
    .{ .name = "BroadcastBackward", .run = runBroadcast },
    .{ .name = "StridedViewBackward (diagonal, merge, split, materialize, viewWithStrides)", .run = runStridedView },
    .{ .name = "AxisViewBackward (permuteTo, insertAxis)", .run = runAxisView },
    .{ .name = "IdentityBackward (contiguous, biasAdd)", .run = runIdentity },
    .{ .name = "CastBackward (f32, f16 round trip)", .run = runCast },
    .{ .name = "CausalConv1dBackward (dilation 1, 2)", .run = runCausalConv1d },
    .{ .name = "CausalDepthwiseConv1dBackward", .run = runCausalDepthwiseConv1d },
    .{ .name = "GroupedCausalConv1dBackward", .run = runGroupedCausalConv1d },
    .{ .name = "UnfoldBackward (unpadded, padded strided)", .run = runUnfold },
    .{ .name = "FoldBackward", .run = runFold },
};

test "finite differences over the VJP records without a dedicated gradcheck" {
    var failures: usize = 0;
    for (cases) |case| {
        if (case.known.len != 0) {
            std.debug.print("gradcheck case skipped ({s}): KNOWN {s}\n", .{ case.name, case.known });
            continue;
        }
        var gpa = std.heap.DebugAllocator(.{}){};
        defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
        var ctx: ExecContext = undefined;
        ctx.init(gpa.allocator());
        defer ctx.deinit();
        case.run(&ctx) catch |err| {
            std.debug.print("gradcheck case failed ({s}): {s}\n", .{ case.name, @errorName(err) });
            failures += 1;
        };
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
}
