const std = @import("std");
const fucina = @import("fucina");
const nn = @import("nn.zig");
const testlog = @import("testlog.zig");

fn expectClose(actual: f32, expected: f32) !void {
    try std.testing.expect(@abs(actual - expected) <= 1e-5);
}

// 4×4×1 map, values h*4+w:
//    0  1  2  3
//    4  5  6  7
//    8  9 10 11
//   12 13 14 15
// 2×2 s2 windows -> avg {2.5, 4.5, 10.5, 12.5}, max {5, 7, 13, 15}, layout [oh,ow,c].
test "avgPool2x2 / maxPool2x2 on a 4x4x1 map (hand-computed)" {
    var ctx: fucina.ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var data: [16]f32 = undefined;
    for (0..16) |i| data[i] = @floatFromInt(i);
    var x = try fucina.Tensor(.{ .h, .w, .c }).variable(&ctx, try ctx.fromSlice(&.{ 4, 4, 1 }, &data));
    defer x.deinit();

    var avg = try nn.avgPool2x2(&ctx, &x);
    defer avg.deinit();
    try std.testing.expectEqual(@as(usize, 2), avg.dim(.h));
    try std.testing.expectEqual(@as(usize, 2), avg.dim(.w));
    const ad = try avg.dataConst();
    const avg_expected = [_]f32{ 2.5, 4.5, 10.5, 12.5 };
    for (avg_expected, 0..) |e, i| try expectClose(ad[i], e);

    var mx = try nn.maxPool2x2(&ctx, &x);
    defer mx.deinit();
    const md = try mx.dataConst();
    const max_expected = [_]f32{ 5, 7, 13, 15 };
    for (max_expected, 0..) |e, i| try expectClose(md[i], e);
}

// 2×2×1 map [[1,2],[3,4]] -> nearest-2x -> 4×4 each pixel replicated 2×2.
test "upsample2xNearest on a 2x2x1 map (hand-computed)" {
    var ctx: fucina.ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var data = [_]f32{ 1, 2, 3, 4 };
    var x = try fucina.Tensor(.{ .h, .w, .c }).variable(&ctx, try ctx.fromSlice(&.{ 2, 2, 1 }, &data));
    defer x.deinit();

    var up = try nn.upsample2xNearest(&ctx, &x);
    defer up.deinit();
    try std.testing.expectEqual(@as(usize, 4), up.dim(.h));
    try std.testing.expectEqual(@as(usize, 4), up.dim(.w));
    const ud = try up.dataConst();
    const expected = [_]f32{
        1, 1, 2, 2,
        1, 1, 2, 2,
        3, 3, 4, 4,
        3, 3, 4, 4,
    };
    for (expected, 0..) |e, i| try expectClose(ud[i], e);
}

// PReLU per-channel: pixels [[-4,3],[2,-6]] (h=2,w=1,c=2), alpha [0.25,0.5].
// prelu = x>0 ? x : alpha[c]*x  ->  [-1, 3, 2, -3].
test "prelu with per-channel learnable slope (vs max(x,0)+a*min(x,0))" {
    var ctx: fucina.ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var data = [_]f32{ -4, 3, 2, -6 };
    var x = try fucina.Tensor(.{ .h, .w, .c }).variable(&ctx, try ctx.fromSlice(&.{ 2, 1, 2 }, &data));
    defer x.deinit();
    var adata = [_]f32{ 0.25, 0.5 };
    var alpha = try fucina.Tensor(.{.c}).variable(&ctx, try ctx.fromSlice(&.{2}, &adata));
    defer alpha.deinit();

    var y = try nn.prelu(&ctx, &x, &alpha);
    defer y.deinit();
    const yd = try y.dataConst();
    const expected = [_]f32{ -1, 3, 2, -3 };
    for (expected, 0..) |e, i| try expectClose(yd[i], e);
}

// Inference BatchNorm, per-node eps. Two channels:
//   c0: mu=1 var=3 gamma=2 beta=0.5 eps=1 -> scale=2/sqrt(4)=1, shift=0.5-1 = -0.5 -> y=x-0.5
//   c1: mu=0 var=8 gamma=6 beta=1   eps=1 -> scale=6/sqrt(9)=2, shift=1        -> y=2x+1
// pixels [[3,2],[-1,4]] (h=2,w=1,c=2) -> [2.5, 5, -1.5, 9].
test "batchNormInfer per-channel affine vs (x-mu)/sqrt(var+eps)*gamma+beta" {
    var ctx: fucina.ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var xd = [_]f32{ 3, 2, -1, 4 };
    var x = try fucina.Tensor(.{ .h, .w, .c }).variable(&ctx, try ctx.fromSlice(&.{ 2, 1, 2 }, &xd));
    defer x.deinit();
    var gd = [_]f32{ 2, 6 };
    var gamma = try fucina.Tensor(.{.c}).variable(&ctx, try ctx.fromSlice(&.{2}, &gd));
    defer gamma.deinit();
    var bd = [_]f32{ 0.5, 1 };
    var beta = try fucina.Tensor(.{.c}).variable(&ctx, try ctx.fromSlice(&.{2}, &bd));
    defer beta.deinit();
    var md = [_]f32{ 1, 0 };
    var mean = try fucina.Tensor(.{.c}).variable(&ctx, try ctx.fromSlice(&.{2}, &md));
    defer mean.deinit();
    var vd = [_]f32{ 3, 8 };
    var variance = try fucina.Tensor(.{.c}).variable(&ctx, try ctx.fromSlice(&.{2}, &vd));
    defer variance.deinit();

    var y = try nn.batchNormInfer(&ctx, &x, &gamma, &beta, &mean, &variance, 1.0);
    defer y.deinit();
    const yd = try y.dataConst();
    const expected = [_]f32{ 2.5, 5, -1.5, 9 };
    for (expected, 0..) |e, i| try expectClose(yd[i], e);
}

fn preluGcLoss(ctx: *fucina.ExecContext, x: *const nn.Map, a: *const nn.Channels) !fucina.Tensor(.{}) {
    var y = try nn.prelu(ctx, x, a);
    defer y.deinit();
    return y.sumAll(ctx);
}

// PReLU backward: grad wrt x and per-channel alpha. Values kept away from the
// x=0 kink so finite differences are valid; each channel has a negative value
// so the alpha gradient is non-trivial.
test "gradcheck: prelu (x + per-channel alpha)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .h, .w, .c }).variableFromSlice(&ctx, .{ 2, 2, 2 }, &.{ 1.0, -1.5, 0.8, -0.6, -1.2, 0.9, 0.7, -1.1 });
    defer x.deinit();
    var a = try fucina.Tensor(.{.c}).variableFromSlice(&ctx, .{2}, &.{ 0.25, -0.5 });
    defer a.deinit();

    const result = try fucina.gradcheck(&ctx, preluGcLoss, .{ &x, &a }, .{});
    try std.testing.expectEqual(@as(usize, 10), result.checked); // 8 x + 2 alpha
    try std.testing.expect(result.max_abs_error <= 2e-3);
}

fn bnTrainGcLoss(ctx: *fucina.ExecContext, x: *const nn.Map, g: *const nn.Channels, b: *const nn.Channels) !fucina.Tensor(.{}) {
    var y = try nn.batchNormTrain(ctx, x, g, b, 1e-3);
    defer y.deinit();
    return y.sumAll(ctx);
}

// Train-mode BatchNorm backward: grad wrt x, gamma, beta (batch stats couple all
// spatial positions of a channel). Values well-spread so variance is far from 0.
test "gradcheck: batchNormTrain (x, gamma, beta)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .h, .w, .c }).variableFromSlice(&ctx, .{ 2, 2, 2 }, &.{ 1.0, 2.0, -0.5, 0.5, 0.3, -1.0, 0.8, 1.5 });
    defer x.deinit();
    var g = try fucina.Tensor(.{.c}).variableFromSlice(&ctx, .{2}, &.{ 1.2, 0.8 });
    defer g.deinit();
    var b = try fucina.Tensor(.{.c}).variableFromSlice(&ctx, .{2}, &.{ 0.1, -0.3 });
    defer b.deinit();

    const result = try fucina.gradcheck(&ctx, bnTrainGcLoss, .{ &x, &g, &b }, .{});
    try std.testing.expectEqual(@as(usize, 12), result.checked); // 8 x + 2 g + 2 b
    try std.testing.expect(result.max_abs_error <= 3e-3);
}

fn avgPoolGcLoss(ctx: *fucina.ExecContext, x: *const nn.Map) !fucina.Tensor(.{}) {
    var y = try nn.avgPool2x2(ctx, x);
    defer y.deinit();
    return y.sumAll(ctx);
}
fn maxPoolGcLoss(ctx: *fucina.ExecContext, x: *const nn.Map) !fucina.Tensor(.{}) {
    var y = try nn.maxPool2x2(ctx, x);
    defer y.deinit();
    return y.sumAll(ctx);
}
fn upsampleGcLoss(ctx: *fucina.ExecContext, x: *const nn.Map) !fucina.Tensor(.{}) {
    var y = try nn.upsample2xNearest(ctx, x);
    defer y.deinit();
    return y.sumAll(ctx);
}

// Pool + upsample backward are inherited (pool = split+mean/max; upsample =
// split+broadcast+merge). Values 0..15 are distinct so max-pool has no ties and
// finite differences are valid.
test "gradcheck: avg/max pool + upsample backward" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var d16: [16]f32 = undefined;
    for (0..16) |i| d16[i] = @floatFromInt(i);
    var x4 = try fucina.Tensor(.{ .h, .w, .c }).variableFromSlice(&ctx, .{ 4, 4, 1 }, &d16);
    defer x4.deinit();

    const avg = try fucina.gradcheck(&ctx, avgPoolGcLoss, .{&x4}, .{});
    try std.testing.expectEqual(@as(usize, 16), avg.checked);
    try std.testing.expect(avg.max_abs_error <= 2e-3);

    const mx = try fucina.gradcheck(&ctx, maxPoolGcLoss, .{&x4}, .{});
    try std.testing.expectEqual(@as(usize, 16), mx.checked);
    try std.testing.expect(mx.max_abs_error <= 2e-3);

    var x2 = try fucina.Tensor(.{ .h, .w, .c }).variableFromSlice(&ctx, .{ 2, 2, 1 }, &.{ 1, 2, 3, 4 });
    defer x2.deinit();
    const up = try fucina.gradcheck(&ctx, upsampleGcLoss, .{&x2}, .{});
    try std.testing.expectEqual(@as(usize, 4), up.checked);
    // grad magnitude is 4 (each pixel -> 4 outputs), so the absolute FD error
    // scales accordingly (~2.2e-3 = ~5e-4 relative — correct, just larger).
    try std.testing.expect(up.max_abs_error <= 3e-3);
}

// ArcFace additive-angular-margin loss (identity head + margin), composed:
// l2-normalize embedding + per-class weights -> cosine logits -> add angular
// margin m to the TARGET class (cos(θ+m) = cosθ·cos m − sinθ·sin m) -> scale s ->
// softmax cross-entropy. `where(one_hot, cos_m, cos)` applies the margin only to
// the target. Differentiable in the embedding and the weight (the head).
fn arcfaceGcLoss(ctx: *fucina.ExecContext, emb: *const fucina.Tensor(.{ .batch, .d }), weight: *const fucina.Tensor(.{ .cls, .d })) !fucina.Tensor(.{}) {
    const s: f32 = 8.0;
    const cm: f32 = @cos(0.3); // cos(margin), margin = 0.3 rad
    const sm: f32 = @sin(0.3);
    var xn = try emb.l2Normalize(ctx, .d, 1e-6);
    defer xn.deinit();
    var wn = try weight.l2Normalize(ctx, .d, 1e-6);
    defer wn.deinit();
    var cos = try xn.dot(ctx, &wn, .d); // [batch, cls]
    defer cos.deinit();
    var cos_sq = try cos.mul(ctx, &cos);
    defer cos_sq.deinit();
    var neg = try cos_sq.scale(ctx, -1.0);
    defer neg.deinit();
    var one_minus = try neg.addScalar(ctx, 1.0); // 1 − cos²
    defer one_minus.deinit();
    var sin_t = try one_minus.sqrt(ctx); // sinθ = √(1−cos²)
    defer sin_t.deinit();
    var cc = try cos.scale(ctx, cm);
    defer cc.deinit();
    var ss = try sin_t.scale(ctx, sm);
    defer ss.deinit();
    var cos_m = try cc.sub(ctx, &ss); // cosθ·cos m − sinθ·sin m
    defer cos_m.deinit();
    // one-hot target mask (batch 1, class 1) — a constant.
    var one_hot = try fucina.Tensor(.{ .batch, .cls }).fromSlice(ctx, .{ 1, 3 }, &.{ 0, 1, 0 });
    defer one_hot.deinit();
    var logit = try cos_m.where(ctx, &one_hot, &cos); // target gets margin, else cos
    defer logit.deinit();
    var scaled = try logit.scale(ctx, s);
    defer scaled.deinit();
    return scaled.crossEntropy(ctx, .cls, &.{1});
}

test "gradcheck: ArcFace additive-angular-margin loss (embedding + head)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // embedding [1,4], weight [3 classes, 4] — values give moderate cosines (off ±1).
    var emb = try fucina.Tensor(.{ .batch, .d }).variableFromSlice(&ctx, .{ 1, 4 }, &.{ 0.6, -0.3, 0.9, 0.2 });
    defer emb.deinit();
    var weight = try fucina.Tensor(.{ .cls, .d }).variableFromSlice(&ctx, .{ 3, 4 }, &.{
        0.4, 0.5, -0.2, 0.7, -0.6, 0.3, 0.8, 0.1, 0.2, -0.4, 0.5, 0.9,
    });
    defer weight.deinit();

    const result = try fucina.gradcheck(&ctx, arcfaceGcLoss, .{ &emb, &weight }, .{});
    try std.testing.expectEqual(@as(usize, 16), result.checked); // 4 emb + 12 weight
    try std.testing.expect(result.max_abs_error <= 3e-3);
}

// Training capstone: a tiny conv -> relu -> avg-pool -> FC classifier
// trained with AdamW + exec scopes, overfitting two separable 4x4 images
// (bright-left = class 0, bright-right = class 1). Proves the differentiable
// conv2d + pool primitives train end-to-end on CPU — loss falls sharply.
test "training capstone: tiny conv classifier converges (AdamW + exec scopes)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var cw_buf: [4 * 3 * 3 * 1]f32 = undefined;
    fucina.rng.gaussianFill(1, &cw_buf, 0.3);
    var conv_w = try fucina.Tensor(.{ .oc, .kh, .kw, .c }).variableFromSlice(&ctx, .{ 4, 3, 3, 1 }, &cw_buf);
    defer conv_w.deinit();
    var cb_buf = [_]f32{0} ** 4;
    var conv_b = try fucina.Tensor(.{.oc}).variableFromSlice(&ctx, .{4}, &cb_buf);
    defer conv_b.deinit();
    var fw_buf: [2 * 16]f32 = undefined;
    fucina.rng.gaussianFill(2, &fw_buf, 0.3);
    var fc_w = try fucina.Tensor(.{ .class, .flat }).variableFromSlice(&ctx, .{ 2, 16 }, &fw_buf);
    defer fc_w.deinit();
    var fb_buf = [_]f32{0} ** 2;
    var fc_b = try fucina.Tensor(.{.class}).variableFromSlice(&ctx, .{2}, &fb_buf);
    defer fc_b.deinit();

    const img0 = [_]f32{ 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0 }; // bright left
    const img1 = [_]f32{ 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1 }; // bright right

    var opt = fucina.optim.AdamW.init(allocator, .{ .lr = 0.05, .weight_decay = 0.0 });
    defer opt.deinit();
    try opt.addParam(&conv_w);
    try opt.addParam(&conv_b);
    try opt.addParam(&fc_w);
    try opt.addParam(&fc_b);

    var first_loss: f32 = 0;
    var last_loss: f32 = 0;
    var step: usize = 0;
    while (step < 80) : (step += 1) {
        const use0 = (step % 2 == 0);
        const img: []const f32 = if (use0) &img0 else &img1;
        const label: usize = if (use0) 0 else 1;

        var x = try fucina.Tensor(.{ .h, .w, .c }).fromSlice(&ctx, .{ 4, 4, 1 }, img);
        defer x.deinit();

        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);

        var cv = try x.conv2d(&ctx, &conv_w, &conv_b, .{ 1, 1 }, .{ 1, 1 }, 1, .{ .h, .w, .c });
        var rl = try cv.relu(&ctx);
        var pl = try nn.avgPool2x2(&ctx, &rl);
        var fl = try pl.merge(&ctx, .flat, .{ .h, .w, .c });
        var lg = try fl.dot(&ctx, &fc_w, .flat);
        var lb = try lg.add(&ctx, &fc_b);
        var loss = try lb.crossEntropy(&ctx, .class, &.{label});

        const lv = try loss.item();
        if (step == 0) first_loss = lv;
        last_loss = lv;
        if (step == 0 or step == 40 or step == 79) testlog.print("[capstone] step {d} loss {d:.4}\n", .{ step, lv });

        try loss.backward(&ctx);
        try opt.step(&ctx);
        opt.zeroGrad();
    }
    // Overfitting two separable examples -> loss collapses well below the start.
    try std.testing.expect(last_loss < first_loss * 0.3);
    try std.testing.expect(last_loss < 0.2);
}
