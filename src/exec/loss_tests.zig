//! Numeric tests for the whole-tensor elementwise losses in `loss.zig`
//! (MSE / Huber / BCE / KL divergence): hand-computed goldens against the
//! torch formulas (documented per test), reduction plumbing, the
//! upstream-gradient arms for BOTH operands, and validation errors.
//! Force-imported by `loss.zig`'s `test` block. Excluded from arch-check
//! (a `_tests.zig` file).

const std = @import("std");

const exec_mod = @import("../exec.zig");
const tensor_mod = @import("../tensor.zig");

const ExecContext = exec_mod.ExecContext;
const TensorError = tensor_mod.TensorError;

/// f64 golden vs f32 kernel output: absolute-plus-relative tolerance sized
/// for a handful of f32 ops per element.
fn expectClose(want: f64, got: f32) !void {
    const tol = 1e-5 + 1e-5 * @abs(want);
    try std.testing.expect(@abs(@as(f64, got) - want) <= tol);
}

fn expectSliceClose(want: []const f64, got: []const f32) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try expectClose(w, g);
}

test "exec mseLoss matches torch F.mse_loss across reductions and both gradients" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // torch: F.mse_loss(x, t, reduction) with per-element (x - t)^2;
    // d = x - t = {0.5, -0.5, 1.5, 0} -> losses {0.25, 0.25, 2.25, 0}.
    var input = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    var target = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 0.5, 2.5, 1.5, 4 });
    defer target.deinit();

    var mean_loss = try ctx.mseLoss(&input, &target, .{});
    defer mean_loss.deinit();
    try expectClose(0.6875, mean_loss.item());

    var sum_loss = try ctx.mseLoss(&input, &target, .{ .reduction = .sum });
    defer sum_loss.deinit();
    try expectClose(2.75, sum_loss.item());

    var none_loss = try ctx.mseLoss(&input, &target, .{ .reduction = .none });
    defer none_loss.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, none_loss.shape.slice());
    try expectSliceClose(&.{ 0.25, 0.25, 2.25, 0 }, none_loss.dataConst());

    // .mean upstream 0.75: d/dx = 2·d·0.75/4 = d·0.375; d/dt = -d/dx.
    var gy = try ctx.fromSliceRank(1, .{1}, &.{0.75});
    defer gy.deinit();
    var gx = try ctx.mseBackwardUpstream(&input, &target, .{}, &gy, .input);
    defer gx.deinit();
    try expectSliceClose(&.{ 0.1875, -0.1875, 0.5625, 0 }, gx.dataConst());
    var gt = try ctx.mseBackwardUpstream(&input, &target, .{}, &gy, .target);
    defer gt.deinit();
    try expectSliceClose(&.{ -0.1875, 0.1875, -0.5625, 0 }, gt.dataConst());

    // .none upstream {1, 2, 3, 4}: d/dx_i = 2·d_i·gy_i.
    var gy_none = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer gy_none.deinit();
    var gx_none = try ctx.mseBackwardUpstream(&input, &target, .{ .reduction = .none }, &gy_none, .input);
    defer gx_none.deinit();
    try expectSliceClose(&.{ 1, -2, 9, 0 }, gx_none.dataConst());
}

test "exec huberLoss matches torch F.huber_loss (quadratic/linear arms) and validates delta" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // torch: F.huber_loss(x, t, delta=1.5): 0.5·d² for |d| <= delta,
    // delta·(|d| - 0.5·delta) otherwise. d = {0.5, -2, 1.5, 0}:
    // {0.125, 1.5·(2 - 0.75) = 1.875, 0.5·2.25 = 1.125 (boundary is
    // quadratic), 0}. sum = 3.125, mean = 0.78125.
    var input = try ctx.fromSliceRank(1, .{4}, &.{ 0.5, -2, 1.5, 0 });
    defer input.deinit();
    var target = try ctx.fromSliceRank(1, .{4}, &.{ 0, 0, 0, 0 });
    defer target.deinit();
    const options = exec_mod.HuberOptions{ .delta = 1.5 };

    var mean_loss = try ctx.huberLoss(&input, &target, options);
    defer mean_loss.deinit();
    try expectClose(0.78125, mean_loss.item());

    var sum_loss = try ctx.huberLoss(&input, &target, .{ .delta = 1.5, .reduction = .sum });
    defer sum_loss.deinit();
    try expectClose(3.125, sum_loss.item());

    var none_loss = try ctx.huberLoss(&input, &target, .{ .delta = 1.5, .reduction = .none });
    defer none_loss.deinit();
    try expectSliceClose(&.{ 0.125, 1.875, 1.125, 0 }, none_loss.dataConst());

    // .sum upstream 1: d/dx = d for |d| <= delta else delta·sign(d).
    var gy = try ctx.fromSliceRank(1, .{1}, &.{1});
    defer gy.deinit();
    var gx = try ctx.huberBackwardUpstream(&input, &target, .{ .delta = 1.5, .reduction = .sum }, &gy, .input);
    defer gx.deinit();
    try expectSliceClose(&.{ 0.5, -1.5, 1.5, 0 }, gx.dataConst());
    var gt = try ctx.huberBackwardUpstream(&input, &target, .{ .delta = 1.5, .reduction = .sum }, &gy, .target);
    defer gt.deinit();
    try expectSliceClose(&.{ -0.5, 1.5, -1.5, 0 }, gt.dataConst());

    // delta must be positive and finite.
    try std.testing.expectError(TensorError.InvalidShape, ctx.huberLoss(&input, &target, .{ .delta = 0 }));
    try std.testing.expectError(TensorError.InvalidShape, ctx.huberLoss(&input, &target, .{ .delta = -1 }));
    try std.testing.expectError(TensorError.InvalidShape, ctx.huberBackwardUpstream(&input, &target, .{ .delta = 0 }, &gy, .input));
}

test "exec bceLoss from_logits matches torch F.binary_cross_entropy_with_logits" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // torch: loss_i = max(x,0) - x·y + log1p(exp(-|x|)) for
    // x = {0, 2, -3, 1}, y = {0, 1, 0, 0.5}:
    //   {ln 2, log1p(e^-2), log1p(e^-3), 0.5 + log1p(e^-1)}.
    var input = try ctx.fromSliceRank(1, .{4}, &.{ 0, 2, -3, 1 });
    defer input.deinit();
    var target = try ctx.fromSliceRank(1, .{4}, &.{ 0, 1, 0, 0.5 });
    defer target.deinit();
    const options = exec_mod.BceOptions{ .from_logits = true };

    var none_loss = try ctx.bceLoss(&input, &target, .{ .from_logits = true, .reduction = .none });
    defer none_loss.deinit();
    try expectSliceClose(&.{
        0.6931471805599453, // ln 2
        0.126928011042973, // log1p(e^-2)
        0.048587351573742244, // log1p(e^-3)
        0.8132616875182229, // 0.5 + log1p(e^-1)
    }, none_loss.dataConst());

    var mean_loss = try ctx.bceLoss(&input, &target, options);
    defer mean_loss.deinit();
    try expectClose(0.42048105767372086, mean_loss.item());

    // .mean upstream 1: d/dx = (sigmoid(x) - y)/4; d/dy = -x/4.
    var gy = try ctx.fromSliceRank(1, .{1}, &.{1});
    defer gy.deinit();
    var gx = try ctx.bceBackwardUpstream(&input, &target, options, &gy, .input);
    defer gx.deinit();
    try expectSliceClose(&.{
        0.125,
        -0.029800730505529426, // (sigmoid(2) - 1)/4
        0.011856468294391696, // sigmoid(-3)/4
        0.05776464465750122, // (sigmoid(1) - 0.5)/4
    }, gx.dataConst());
    var gt = try ctx.bceBackwardUpstream(&input, &target, options, &gy, .target);
    defer gt.deinit();
    try expectSliceClose(&.{ 0, -0.5, 0.75, -0.25 }, gt.dataConst());
}

test "exec bceLoss probability arm matches torch F.binary_cross_entropy and clamps at bce_eps" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // torch: loss_i = -(y·ln p + (1-y)·ln(1-p)) for p = {0.25, 0.5, 0.9, 0.02},
    // y = {0, 1, 0.5, 1} (all p inside the clamp interval).
    var input = try ctx.fromSliceRank(1, .{4}, &.{ 0.25, 0.5, 0.9, 0.02 });
    defer input.deinit();
    var target = try ctx.fromSliceRank(1, .{4}, &.{ 0, 1, 0.5, 1 });
    defer target.deinit();

    var none_loss = try ctx.bceLoss(&input, &target, .{ .reduction = .none });
    defer none_loss.deinit();
    try expectSliceClose(&.{
        0.2876820724517809, // -ln 0.75
        0.6931471805599453, // -ln 0.5
        1.2039728043259361, // -(0.5·ln 0.9 + 0.5·ln 0.1)
        3.912023005428146, // -ln 0.02
    }, none_loss.dataConst());

    var sum_loss = try ctx.bceLoss(&input, &target, .{ .reduction = .sum });
    defer sum_loss.deinit();
    try expectClose(6.096825062766, sum_loss.item());

    // .sum upstream 1: d/dp = (p - y)/(p·(1-p)); d/dy = ln(1-p) - ln p.
    var gy = try ctx.fromSliceRank(1, .{1}, &.{1});
    defer gy.deinit();
    var gx = try ctx.bceBackwardUpstream(&input, &target, .{ .reduction = .sum }, &gy, .input);
    defer gx.deinit();
    try expectSliceClose(&.{ 4.0 / 3.0, -2, 4.444444444444445, -50 }, gx.dataConst());
    var gt = try ctx.bceBackwardUpstream(&input, &target, .{ .reduction = .sum }, &gy, .target);
    defer gt.deinit();
    try expectSliceClose(&.{
        1.0986122886681098, // ln 0.75 - ln 0.25
        0,
        -2.1972245773362196, // ln 0.1 - ln 0.9
        3.8918202981106265, // ln 0.98 - ln 0.02
    }, gt.dataConst());

    // Boundary probabilities: the forward clamps to [bce_eps, 1-bce_eps]
    // (loss ~ -ln(1-eps) ~ 1.2e-7, not inf/NaN) and the input gradient is
    // exactly 0 where the clamp saturates.
    var boundary = try ctx.fromSliceRank(1, .{2}, &.{ 0, 1 });
    defer boundary.deinit();
    var boundary_target = try ctx.fromSliceRank(1, .{2}, &.{ 0, 1 });
    defer boundary_target.deinit();
    var boundary_loss = try ctx.bceLoss(&boundary, &boundary_target, .{ .reduction = .none });
    defer boundary_loss.deinit();
    for (boundary_loss.dataConst()) |value| {
        try std.testing.expect(std.math.isFinite(value));
        try std.testing.expect(@abs(value) <= 1e-6);
    }
    var boundary_gx = try ctx.bceBackwardUpstream(&boundary, &boundary_target, .{ .reduction = .sum }, &gy, .input);
    defer boundary_gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, boundary_gx.dataConst());
}

test "exec klDivLoss matches torch F.kl_div for probability and log targets" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // torch: F.kl_div(x, t) with x = log-probs of {0.7, 0.2, 0.1} and
    // t = {0.5, 0.25, 0.25}: loss_i = t·(ln t - x).
    var input = try ctx.fromSliceRank(1, .{3}, &.{ -0.35667494393873245, -1.6094379124341003, -2.3025850929940455 });
    defer input.deinit();
    var target = try ctx.fromSliceRank(1, .{3}, &.{ 0.5, 0.25, 0.25 });
    defer target.deinit();

    const row_losses = [_]f64{ -0.16823611831060643, 0.05578588782855243, 0.22907268296853872 };
    var none_loss = try ctx.klDivLoss(&input, &target, .{ .reduction = .none });
    defer none_loss.deinit();
    try expectSliceClose(&row_losses, none_loss.dataConst());

    var sum_loss = try ctx.klDivLoss(&input, &target, .{ .reduction = .sum });
    defer sum_loss.deinit();
    try expectClose(0.11662245248648472, sum_loss.item());

    // Deliberate divergence from torch's batchmean: .mean divides by the
    // TOTAL element count (torch's own `.mean`).
    var mean_loss = try ctx.klDivLoss(&input, &target, .{});
    defer mean_loss.deinit();
    try expectClose(0.03887415082882824, mean_loss.item());

    // .sum upstream 1: d/dx = -t; d/dt = ln t + 1 - x.
    var gy = try ctx.fromSliceRank(1, .{1}, &.{1});
    defer gy.deinit();
    var gx = try ctx.klDivBackwardUpstream(&input, &target, .{ .reduction = .sum }, &gy, .input);
    defer gx.deinit();
    try expectSliceClose(&.{ -0.5, -0.25, -0.25 }, gx.dataConst());
    var gt = try ctx.klDivBackwardUpstream(&input, &target, .{ .reduction = .sum }, &gy, .target);
    defer gt.deinit();
    try expectSliceClose(&.{ 0.6635277633787872, 1.2231435513142097, 1.916290731874155 }, gt.dataConst());

    // Zero-mass target entry: zero forward contribution (xlogy convention)
    // and zero gradients for both operands.
    var zero_target = try ctx.fromSliceRank(1, .{3}, &.{ 0, 0.5, 0.5 });
    defer zero_target.deinit();
    var zero_loss = try ctx.klDivLoss(&input, &zero_target, .{ .reduction = .none });
    defer zero_loss.deinit();
    try std.testing.expectEqual(@as(f32, 0), zero_loss.dataConst()[0]);
    var zero_gt = try ctx.klDivBackwardUpstream(&input, &zero_target, .{ .reduction = .sum }, &gy, .target);
    defer zero_gt.deinit();
    try std.testing.expectEqual(@as(f32, 0), zero_gt.dataConst()[0]);

    // log_target arm: t holds ln{0.5, 0.25, 0.25}; identical losses,
    // d/dx = -exp(t), d/dt = exp(t)·(t - x + 1).
    var log_target = try ctx.fromSliceRank(1, .{3}, &.{ -0.6931471805599453, -1.3862943611198906, -1.3862943611198906 });
    defer log_target.deinit();
    const log_options = exec_mod.KlDivOptions{ .log_target = true, .reduction = .sum };
    var log_none = try ctx.klDivLoss(&input, &log_target, .{ .log_target = true, .reduction = .none });
    defer log_none.deinit();
    try expectSliceClose(&row_losses, log_none.dataConst());
    var log_gx = try ctx.klDivBackwardUpstream(&input, &log_target, log_options, &gy, .input);
    defer log_gx.deinit();
    try expectSliceClose(&.{ -0.5, -0.25, -0.25 }, log_gx.dataConst());
    var log_gt = try ctx.klDivBackwardUpstream(&input, &log_target, log_options, &gy, .target);
    defer log_gt.deinit();
    try expectSliceClose(&.{ 0.3317638816893936, 0.30578588782855243, 0.4790726829685387 }, log_gt.dataConst());
}

test "exec elementwise losses validate operand and upstream shapes" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var input = try ctx.fromSliceRank(1, .{4}, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    var short = try ctx.fromSliceRank(1, .{3}, &.{ 1, 2, 3 });
    defer short.deinit();
    var target = try ctx.fromSliceRank(1, .{4}, &.{ 0, 0, 0, 0 });
    defer target.deinit();

    try std.testing.expectError(TensorError.ShapeMismatch, ctx.mseLoss(&input, &short, .{}));
    try std.testing.expectError(TensorError.ShapeMismatch, ctx.bceLoss(&input, &short, .{}));

    // `.mean`/`.sum` upstream must be scalar; `.none` upstream must be
    // input-shaped.
    var gy_vec = try ctx.fromSliceRank(1, .{4}, &.{ 1, 1, 1, 1 });
    defer gy_vec.deinit();
    var gy_scalar = try ctx.fromSliceRank(1, .{1}, &.{1});
    defer gy_scalar.deinit();
    try std.testing.expectError(TensorError.ShapeMismatch, ctx.mseBackwardUpstream(&input, &target, .{}, &gy_vec, .input));
    try std.testing.expectError(TensorError.ShapeMismatch, ctx.mseBackwardUpstream(&input, &target, .{ .reduction = .none }, &gy_scalar, .input));
}
