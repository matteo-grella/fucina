//! 2D spatial ops: conv2d facades, winograd and prepared-weight parity,
//! pooling and upsample, prelu/channelAffine, and unfold/fold patch
//! extraction with adjoint gradients.

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

test "public Tensor conv2d facade matches ctx.conv2d (with and without bias)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // input [H=3, W=3, Cin=1], weight [Cout=1, kH=2, kW=2, Cin/g=1] -> [oH=2, oW=2, Cout=1]
    var inp = try Tensor(.{ .h, .w, .cin }).fromSlice(&ctx, .{ 3, 3, 1 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer inp.deinit();
    var w = try Tensor(.{ .cout, .kh, .kw, .cinpg }).fromSlice(&ctx, .{ 1, 2, 2, 1 }, &.{ 1, 2, 3, 4 });
    defer w.deinit();

    // no bias
    var got = try inp.conv2d(&ctx, w, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .oh, .ow, .cout });
    defer got.deinit();
    var want = try ctx.conv2d(inp.asRawTensor(), w.asRawTensor(), null, .{ 1, 1 }, .{ 0, 0 }, 1);
    defer want.deinit();
    try std.testing.expectEqualSlices(f32, want.dataConst(), got.asRawTensor().dataConst());

    // with a [Cout] bias
    var bias = try Tensor(.{.cout}).fromSlice(&ctx, .{1}, &.{10});
    defer bias.deinit();
    var got_b = try inp.conv2d(&ctx, w, bias, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .oh, .ow, .cout });
    defer got_b.deinit();
    var want_b = try ctx.conv2d(inp.asRawTensor(), w.asRawTensor(), bias.asRawTensor(), .{ 1, 1 }, .{ 0, 0 }, 1);
    defer want_b.deinit();
    try std.testing.expectEqualSlices(f32, want_b.dataConst(), got_b.asRawTensor().dataConst());
}

test "public Tensor conv2dRelu grad path is composed: gradients pass the relu mask, scoped or not" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // input [H=3, W=3, Cin=1], weight [Cout=1, kH=2, kW=2, 1] = x(i,j) - x(i+1,j+1):
    // conv = [2, -3, -5, 3] -> relu keeps positions (0,0) and (1,1) only.
    var x = try Tensor(.{ .h, .w, .cin }).variableFromSlice(&ctx, .{ 3, 3, 1 }, &.{ 5, 1, 2, 1, 3, 4, 2, 6, 0 });
    defer x.deinit();
    var w = try Tensor(.{ .cout, .kh, .kw, .cinpg }).variableFromSlice(&ctx, .{ 1, 2, 2, 1 }, &.{ 1, 0, 0, -1 });
    defer w.deinit();

    // Unscoped: the composition's conv intermediate is released inside
    // conv2dRelu; the relu record keeps its state alive until backward.
    {
        var y = try x.conv2dRelu(&ctx, w, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .oh, .ow, .cout });
        defer y.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 2, 0, 0, 3 }, try y.dataConst());
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    {
        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();
        // dL/dw[k] = sum of the input patch over the two surviving positions.
        try std.testing.expectEqualSlices(f32, &.{ 8, 5, 7, 3 }, try gw.dataConst());
        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 0, 0, 0, 0, 0, -1 }, try gx.dataConst());
    }
    x.zeroGrad();
    w.zeroGrad();

    // Scoped: the same gradients.
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var y = try x.conv2dRelu(&ctx, w, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .oh, .ow, .cout });
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 8, 5, 7, 3 }, try gw.dataConst());
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 0, 0, 0, 0, 0, -1 }, try gx.dataConst());

    // The no-grad path stays unscoped.
    var xc = try Tensor(.{ .h, .w, .cin }).fromSlice(&ctx, .{ 3, 3, 1 }, &.{ 5, 1, 2, 1, 3, 4, 2, 6, 0 });
    defer xc.deinit();
    var wc = try Tensor(.{ .cout, .kh, .kw, .cinpg }).fromSlice(&ctx, .{ 1, 2, 2, 1 }, &.{ 1, 0, 0, -1 });
    defer wc.deinit();
    var yc = try xc.conv2dRelu(&ctx, wc, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .oh, .ow, .cout });
    defer yc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 0, 0, 3 }, try yc.dataConst());
}

test "public Tensor maxPool2d avgPool2d upsample2xNearest prelu channelAffine forward values" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const Map = Tensor(.{ .h, .w, .c });
    const Chan = Tensor(.{.c});

    // 4x4x1 fixture.
    var x = try Map.fromSlice(&ctx, .{ 4, 4, 1 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    });
    defer x.deinit();

    // max 2x2 s2 p0: window maxima.
    var mp = try x.maxPool2d(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 });
    defer mp.deinit();
    try expectCloseSlices(&.{ 6, 8, 14, 16 }, mp.asRawTensor().dataConst(), 0);

    // max 3x3 s2 p1: −inf border (out-of-range taps skipped).
    var mp3 = try x.maxPool2d(&ctx, .{ 3, 3 }, .{ 2, 2 }, .{ 1, 1 });
    defer mp3.deinit();
    try expectCloseSlices(&.{ 6, 8, 14, 16 }, mp3.asRawTensor().dataConst(), 0);

    // avg 2x2 s2 p1: corner windows hold ONE valid tap (count excludes pad).
    var ap = try x.avgPool2d(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 1, 1 });
    defer ap.deinit();
    try expectCloseSlices(&.{ 1, 2.5, 4, 7, 8.5, 10, 13, 14.5, 16 }, ap.asRawTensor().dataConst(), 0);

    // avg 2x2 s2 p0: plain window means.
    var ap0 = try x.avgPool2d(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 });
    defer ap0.deinit();
    try expectCloseSlices(&.{ 3.5, 5.5, 11.5, 13.5 }, ap0.asRawTensor().dataConst(), 0);

    // upsample 2x on 1x2x2: each pixel becomes a 2x2 block.
    var u = try Map.fromSlice(&ctx, .{ 1, 2, 2 }, &.{ 1, 2, 3, 4 });
    defer u.deinit();
    var up = try u.upsample2xNearest(&ctx);
    defer up.deinit();
    try expectCloseSlices(&.{ 1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4 }, up.asRawTensor().dataConst(), 0);

    // prelu: negative lanes scale by the channel alpha.
    var px = try Map.fromSlice(&ctx, .{ 1, 2, 2 }, &.{ 2, -2, -4, 4 });
    defer px.deinit();
    var alpha = try Chan.fromSlice(&ctx, .{2}, &.{ 0.5, 0.25 });
    defer alpha.deinit();
    var py = try px.prelu(&ctx, &alpha);
    defer py.deinit();
    try expectCloseSlices(&.{ 2, -0.5, -2, 4 }, py.asRawTensor().dataConst(), 0);

    // channelAffine: y = x*s + t per channel.
    var s = try Chan.fromSlice(&ctx, .{2}, &.{ 2, -1 });
    defer s.deinit();
    var t = try Chan.fromSlice(&ctx, .{2}, &.{ 1, 0.5 });
    defer t.deinit();
    var ay = try px.channelAffine(&ctx, &s, &t);
    defer ay.deinit();
    try expectCloseSlices(&.{ 5, 2.5, -7, -3.5 }, ay.asRawTensor().dataConst(), 0);
}

test "conv2d winograd route matches the direct kernel (3x3 s1, pad 0/1, odd shapes)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Pin the Winograd route ON so the check is meaningful on every build
    // config (BLAS builds default the route off).
    const tuning = @import("../../tuning.zig");
    tuning.setField("winograd", true);
    defer tuning.setField("winograd", null);

    const cases = [_][5]usize{
        // h, w, cin, cout, pad — even/odd spatial exercise full and partial
        // tiles; min(oh,ow) >= 14 shapes take the F4 tier, the rest F2.
        .{ 8, 8, 8, 8, 1 },
        .{ 5, 7, 17, 5, 1 },
        .{ 4, 4, 4, 8, 0 },
        .{ 9, 5, 8, 3, 1 },
        .{ 16, 16, 8, 8, 1 },
        .{ 17, 15, 12, 5, 1 },
        .{ 18, 14, 8, 8, 0 },
    };
    var seed: u64 = 1;
    for (cases) |cs| {
        const h = cs[0];
        const w = cs[1];
        const cin = cs[2];
        const cout = cs[3];
        const p = cs[4];
        const oh = h + 2 * p - 2;
        const ow = w + 2 * p - 2;

        const xd = try allocator.alloc(f32, h * w * cin);
        defer allocator.free(xd);
        const wd = try allocator.alloc(f32, cout * 9 * cin);
        defer allocator.free(wd);
        const bd = try allocator.alloc(f32, cout);
        defer allocator.free(bd);
        const rng_mod = @import("../../rng.zig");
        rng_mod.gaussianFill(seed, xd, 1.0);
        rng_mod.gaussianFill(seed + 1, wd, 0.5);
        rng_mod.gaussianFill(seed + 2, bd, 0.5);
        seed += 3;

        // Facade conv2d — the eligible shape takes the Winograd route.
        var x = try Tensor(.{ .h, .w, .c }).fromSlice(&ctx, .{ h, w, cin }, xd);
        defer x.deinit();
        var wt = try Tensor(.{ .oc, .kh, .kw, .c }).fromSlice(&ctx, .{ cout, 3, 3, cin }, wd);
        defer wt.deinit();
        var bt = try Tensor(.{.oc}).fromSlice(&ctx, .{cout}, bd);
        defer bt.deinit();
        var y = try x.conv2d(&ctx, &wt, &bt, .{ 1, 1 }, .{ p, p }, 1, .{ .h, .w, .c });
        defer y.deinit();

        // Reference: the independent scalar reference arm (the conv2d twin).
        var xr = try RawTensor.fromSlice(allocator, &[_]usize{ h, w, cin }, xd);
        defer xr.deinit();
        var wr = try RawTensor.fromSlice(allocator, &[_]usize{ cout, 3, 3, cin }, wd);
        defer wr.deinit();
        var expected = try RawTensor.zeros(allocator, &[_]usize{ oh, ow, cout });
        defer expected.deinit();
        backend_mod.vector_impl.conv.scalar.conv2dInto(&expected, &xr, &wr, bd, .{
            .h = h,
            .w = w,
            .cin = cin,
            .oh = oh,
            .ow = ow,
            .cout = cout,
            .kh = 3,
            .kw = 3,
            .stride_h = 1,
            .stride_w = 1,
            .pad_h = p,
            .pad_w = p,
            .groups = 1,
        });

        // Winograd reassociates the 3x3 reduction: ~1e-6 relative vs direct.
        const yd = try y.dataConst();
        const ed = expected.dataConst();
        try std.testing.expectEqual(ed.len, yd.len);
        for (ed, yd) |e, a| {
            const tol = 1e-4 * @max(@as(f32, 1.0), @abs(e));
            try std.testing.expect(@abs(e - a) <= tol);
        }
    }
}

test "conv2dPrepared matches conv2d bitwise (winograd F2/F4 tiers, odd tails, cin gate, stride-2 fallback)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Pin the Winograd route ON so preparation is exercised on every build
    // config (BLAS builds default the route off).
    const tuning = @import("../../tuning.zig");
    tuning.setField("winograd", true);
    defer tuning.setField("winograd", null);

    const cases = [_][6]usize{
        // h, w, cin, cout, pad, stride — F2-tier small maps (even/odd, full
        // and partial tiles), F4-tier large maps (min(oh,ow) >= 14, cin <=
        // 56), a large map with cin > 56 (F4-ineligible at prepare AND call
        // time: the prepared set carries f2 only and serves the call), and a
        // stride-2 call where the prepared planes are inert (non-Winograd
        // route, identical code path).
        .{ 8, 8, 8, 8, 1, 1 },
        .{ 5, 7, 17, 5, 1, 1 },
        .{ 9, 5, 8, 3, 0, 1 },
        .{ 16, 16, 8, 8, 1, 1 },
        .{ 17, 15, 12, 5, 1, 1 },
        .{ 16, 16, 60, 5, 1, 1 },
        .{ 9, 9, 8, 4, 1, 2 },
    };
    var seed: u64 = 101;
    for (cases) |cs| {
        const h = cs[0];
        const w = cs[1];
        const cin = cs[2];
        const cout = cs[3];
        const p = cs[4];
        const s = cs[5];

        const xd = try allocator.alloc(f32, h * w * cin);
        defer allocator.free(xd);
        const wd = try allocator.alloc(f32, cout * 9 * cin);
        defer allocator.free(wd);
        const bd = try allocator.alloc(f32, cout);
        defer allocator.free(bd);
        const rng_mod = @import("../../rng.zig");
        rng_mod.gaussianFill(seed, xd, 1.0);
        rng_mod.gaussianFill(seed + 1, wd, 0.5);
        rng_mod.gaussianFill(seed + 2, bd, 0.5);
        seed += 3;

        var x = try Tensor(.{ .h, .w, .c }).fromSlice(&ctx, .{ h, w, cin }, xd);
        defer x.deinit();
        var wt = try Tensor(.{ .oc, .kh, .kw, .c }).fromSlice(&ctx, .{ cout, 3, 3, cin }, wd);
        defer wt.deinit();
        var bt = try Tensor(.{.oc}).fromSlice(&ctx, .{cout}, bd);
        defer bt.deinit();

        var prep = try wt.prepareConv2dWeights(&ctx);
        defer prep.deinit();
        // 3x3 with cin >= 4 and the route pinned on: F2 always prepared, F4
        // iff cin passes the max-cin gate (default 56); f4 => f2.
        try std.testing.expect(prep.f2 != null);
        try std.testing.expect((prep.f4 != null) == (cin <= 56));

        var y_ref = try x.conv2d(&ctx, &wt, &bt, .{ s, s }, .{ p, p }, 1, .{ .h, .w, .c });
        defer y_ref.deinit();
        var y_prep = try x.conv2dPrepared(&ctx, &wt, &prep, &bt, .{ s, s }, .{ p, p }, 1, .{ .h, .w, .c });
        defer y_prep.deinit();
        try std.testing.expectEqualSlices(f32, try y_ref.dataConst(), try y_prep.dataConst());

        var yr_ref = try x.conv2dRelu(&ctx, &wt, &bt, .{ s, s }, .{ p, p }, 1, .{ .h, .w, .c });
        defer yr_ref.deinit();
        var yr_prep = try x.conv2dPreparedRelu(&ctx, &wt, &prep, &bt, .{ s, s }, .{ p, p }, 1, .{ .h, .w, .c });
        defer yr_prep.deinit();
        try std.testing.expectEqualSlices(f32, try yr_ref.dataConst(), try yr_prep.dataConst());
    }
}

test "conv2dPrepared: 1x1 and .empty preparations are inert; grad operands are rejected" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const tuning = @import("../../tuning.zig");
    tuning.setField("winograd", true);
    defer tuning.setField("winograd", null);

    const rng_mod = @import("../../rng.zig");
    const xd = try allocator.alloc(f32, 8 * 8 * 8);
    defer allocator.free(xd);
    rng_mod.gaussianFill(11, xd, 1.0);
    var x = try Tensor(.{ .h, .w, .c }).fromSlice(&ctx, .{ 8, 8, 8 }, xd);
    defer x.deinit();

    // 1x1 weight: preparation is naturally `.empty` (not Winograd-shaped)
    // and the call takes the pointwise fast path — bitwise equal.
    const w1d = try allocator.alloc(f32, 4 * 8);
    defer allocator.free(w1d);
    rng_mod.gaussianFill(12, w1d, 0.5);
    var w1 = try Tensor(.{ .oc, .kh, .kw, .c }).fromSlice(&ctx, .{ 4, 1, 1, 8 }, w1d);
    defer w1.deinit();
    var prep1 = try w1.prepareConv2dWeights(&ctx);
    defer prep1.deinit();
    try std.testing.expect(prep1.f2 == null and prep1.f4 == null);
    var y1_ref = try x.conv2d(&ctx, &w1, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .h, .w, .c });
    defer y1_ref.deinit();
    var y1_prep = try x.conv2dPrepared(&ctx, &w1, &prep1, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .h, .w, .c });
    defer y1_prep.deinit();
    try std.testing.expectEqualSlices(f32, try y1_ref.dataConst(), try y1_prep.dataConst());

    // Explicit `.empty` on a Winograd-eligible call: per-call transform
    // fallback, bitwise equal to the unprepared conv.
    const w3d = try allocator.alloc(f32, 4 * 9 * 8);
    defer allocator.free(w3d);
    rng_mod.gaussianFill(13, w3d, 0.5);
    var w3 = try Tensor(.{ .oc, .kh, .kw, .c }).fromSlice(&ctx, .{ 4, 3, 3, 8 }, w3d);
    defer w3.deinit();
    const empty = exec_mod.ExecContext.PreparedConvWeights.empty;
    var y3_ref = try x.conv2d(&ctx, &w3, null, .{ 1, 1 }, .{ 1, 1 }, 1, .{ .h, .w, .c });
    defer y3_ref.deinit();
    var y3_empty = try x.conv2dPrepared(&ctx, &w3, &empty, null, .{ 1, 1 }, .{ 1, 1 }, 1, .{ .h, .w, .c });
    defer y3_empty.deinit();
    try std.testing.expectEqualSlices(f32, try y3_ref.dataConst(), try y3_empty.dataConst());

    // Grad guards (the dotPacked policy): a grad-carrying weight cannot be
    // prepared, and no operand of conv2dPrepared may require grad.
    var wg = try Tensor(.{ .oc, .kh, .kw, .c }).variableFromSlice(&ctx, .{ 4, 3, 3, 8 }, w3d);
    defer wg.deinit();
    try std.testing.expectError(error.GradientPreparedConv2dUnsupported, wg.prepareConv2dWeights(&ctx));
    var xg = try Tensor(.{ .h, .w, .c }).variableFromSlice(&ctx, .{ 8, 8, 8 }, xd);
    defer xg.deinit();
    try std.testing.expectError(
        error.GradientPreparedConv2dUnsupported,
        xg.conv2dPrepared(&ctx, &w3, &empty, null, .{ 1, 1 }, .{ 1, 1 }, 1, .{ .h, .w, .c }),
    );
    try std.testing.expectError(
        error.GradientPreparedConv2dUnsupported,
        x.conv2dPreparedRelu(&ctx, &wg, &empty, null, .{ 1, 1 }, .{ 1, 1 }, 1, .{ .h, .w, .c }),
    );
}

test "public Tensor unfold extracts im2col patches with the fold-adjoint gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const Img = Tensor(.{ .h, .w, .c });
    var x = try Img.variableFromSlice(&ctx, .{ 3, 3, 1 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer x.deinit();

    // 2x2 windows, stride 1, no pad: 4 patches, each (ky, kx, c)-ordered.
    var col = try x.unfold(&ctx, .{ 2, 2 }, .{ 1, 1 }, .{ 0, 0 }, .{ .patch, .elem });
    defer col.deinit();
    try std.testing.expectEqual(@as(usize, 4), col.dim(.patch));
    try std.testing.expectEqual(@as(usize, 4), col.dim(.elem));
    try expectCloseSlices(&.{
        1, 2, 4, 5,
        2, 3, 5, 6,
        4, 5, 7, 8,
        5, 6, 8, 9,
    }, try col.dataConst(), 0);

    // VJP = fold: each position receives its window count.
    var loss = try col.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try expectCloseSlices(&.{ 1, 2, 1, 2, 4, 2, 1, 2, 1 }, try gx.dataConst(), 0);

    // Padding: out-of-range taps read 0 (and receive no gradient).
    var small = try Img.fromSlice(&ctx, .{ 2, 2, 1 }, &.{ 1, 2, 3, 4 });
    defer small.deinit();
    var padded = try small.unfold(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 1, 1 }, .{ .patch, .elem });
    defer padded.deinit();
    try expectCloseSlices(&.{
        0, 0, 0, 1,
        0, 0, 2, 0,
        0, 3, 0, 0,
        4, 0, 0, 0,
    }, try padded.dataConst(), 0);

    // Channel-fastest element order (the conv2d GEMM col layout).
    var two_c = try Img.fromSlice(&ctx, .{ 1, 2, 2 }, &.{ 1, 2, 3, 4 });
    defer two_c.deinit();
    var c_col = try two_c.unfold(&ctx, .{ 1, 2 }, .{ 1, 1 }, .{ 0, 0 }, .{ .patch, .elem });
    defer c_col.deinit();
    try expectCloseSlices(&.{ 1, 2, 3, 4 }, try c_col.dataConst(), 0);

    // stride = kernel, pad 0 is non-overlapping patchify (the ViT stem).
    var big = try Img.fromSlice(&ctx, .{ 4, 4, 1 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    });
    defer big.deinit();
    var patches = try big.unfold(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 }, .{ .patch, .elem });
    defer patches.deinit();
    try expectCloseSlices(&.{
        1,  2,  5,  6,
        3,  4,  7,  8,
        9,  10, 13, 14,
        11, 12, 15, 16,
    }, try patches.dataConst(), 0);
}

test "public Tensor fold accumulates overlaps with the unfold-adjoint gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const Img = Tensor(.{ .h, .w, .c });
    const Col = Tensor(.{ .patch, .elem });

    // fold(unfold(x)) multiplies each position by its window count.
    var x = try Img.fromSlice(&ctx, .{ 3, 3, 1 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer x.deinit();
    var col = try x.unfold(&ctx, .{ 2, 2 }, .{ 1, 1 }, .{ 0, 0 }, .{ .patch, .elem });
    defer col.deinit();
    var back = try col.fold(&ctx, .{ 3, 3 }, .{ 2, 2 }, .{ 1, 1 }, .{ 0, 0 }, .{ .h, .w, .c });
    defer back.deinit();
    try expectCloseSlices(&.{ 1, 4, 3, 8, 20, 12, 7, 16, 9 }, try back.dataConst(), 0);

    // VJP = unfold of the upstream image gradient.
    var vcol = try Col.variableFromSlice(&ctx, .{ 4, 4 }, &([_]f32{1} ** 16));
    defer vcol.deinit();
    var img = try vcol.fold(&ctx, .{ 3, 3 }, .{ 2, 2 }, .{ 1, 1 }, .{ 0, 0 }, .{ .h, .w, .c });
    defer img.deinit();
    var weights = try Img.fromSlice(&ctx, .{ 3, 3, 1 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer weights.deinit();
    var weighted = try img.mul(&ctx, &weights);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gcol = (try vcol.grad(&ctx)).?;
    defer gcol.deinit();
    try expectCloseSlices(&.{
        1, 2, 4, 5,
        2, 3, 5, 6,
        4, 5, 7, 8,
        5, 6, 8, 9,
    }, try gcol.dataConst(), 0);

    // Geometry mismatches are recoverable errors.
    var bad = try Col.fromSlice(&ctx, .{ 3, 4 }, &([_]f32{0} ** 12));
    defer bad.deinit();
    try std.testing.expectError(error.ShapeMismatch, bad.fold(&ctx, .{ 3, 3 }, .{ 2, 2 }, .{ 1, 1 }, .{ 0, 0 }, .{ .h, .w, .c }));
    var bad_width = try Col.fromSlice(&ctx, .{ 4, 3 }, &([_]f32{0} ** 12));
    defer bad_width.deinit();
    try std.testing.expectError(error.ShapeMismatch, bad_width.fold(&ctx, .{ 3, 3 }, .{ 2, 2 }, .{ 1, 1 }, .{ 0, 0 }, .{ .h, .w, .c }));
}
