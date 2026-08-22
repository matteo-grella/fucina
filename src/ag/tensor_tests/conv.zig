//! Causal and 1D convolutions: depthwise/general/grouped causal convs with
//! history state, conv1d, convTranspose1d, and snake, against hand values
//! and finite differences.

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
const fdCheckGrad = util.fdCheckGrad;
const fdFillPattern = util.fdFillPattern;
const fdWeightedSum = util.fdWeightedSum;

test "tagged public tensor causal depthwise conv uses optional history state" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .channel }).fromSlice(&ctx, .{ 3, 2 }, &.{
        1, 10,
        2, 20,
        3, 30,
    });
    defer input.deinit();
    var kernel = try Tensor(.{ .channel, .tap }).fromSlice(&ctx, .{ 2, 3 }, &.{
        1,  2,  3,
        10, 20, 30,
    });
    defer kernel.deinit();

    var no_state = try input.causalDepthwiseConv1d(&ctx, .time, .channel, .tap, &kernel, 1, null);
    defer no_state.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        3,  300,
        8,  800,
        14, 1400,
    }, no_state.asRawTensor().dataConst());

    var state = [_]f32{
        -2, -20,
        4,  40,
    };
    var with_state = try input.causalDepthwiseConv1d(&ctx, .time, .channel, .tap, &kernel, 1, &state);
    defer with_state.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        9,  900,
        12, 1200,
        14, 1400,
    }, with_state.asRawTensor().dataConst());
}

test "tagged public tensor causal depthwise conv propagates input and kernel gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .channel }).variableFromSlice(&ctx, .{ 3, 2 }, &.{
        1, 10,
        2, 20,
        3, 30,
    });
    defer input.deinit();
    var kernel = try Tensor(.{ .channel, .tap }).variableFromSlice(&ctx, .{ 2, 3 }, &.{
        1,  2,  3,
        10, 20, 30,
    });
    defer kernel.deinit();
    var state = [_]f32{
        -1,  -10,
        0.5, 5,
    };

    var out = try input.causalDepthwiseConv1d(&ctx, .time, .channel, .tap, &kernel, 1, &state);
    defer out.deinit();
    var loss = try out.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try input.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        6, 60,
        5, 50,
        3, 30,
    }, gx.asRawTensor().dataConst());

    var gk = (try kernel.grad(&ctx)).?;
    defer gk.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        0.5, 3.5, 6,
        5,   35,  60,
    }, gk.asRawTensor().dataConst());
}

test "dilated causal depthwise conv: hand values, state, grouped equivalence with gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Hand case: dilation=2, taps=2 (tap 1 = newest):
    // y[t] = 10·x[t-2] + 1·x[t].
    var x1 = try Tensor(.{ .time, .channel }).fromSlice(&ctx, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer x1.deinit();
    var k1 = try Tensor(.{ .channel, .tap }).fromSlice(&ctx, .{ 1, 2 }, &.{ 10, 1 });
    defer k1.deinit();
    var y1 = try x1.causalDepthwiseConv1d(&ctx, .time, .channel, .tap, &k1, 2, null);
    defer y1.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 13, 24 }, y1.asRawTensor().dataConst());

    // Streaming state supplies the dilation·(taps−1) = 2 preceding rows.
    var s1 = [_]f32{ 7, 9 };
    var y1s = try x1.causalDepthwiseConv1d(&ctx, .time, .channel, .tap, &k1, 2, &s1);
    defer y1s.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 71, 92, 13, 24 }, y1s.asRawTensor().dataConst());

    // Wrong state length (taps−1 rows, the undilated count) is rejected.
    var short_state = [_]f32{7};
    try std.testing.expectError(error.InvalidDataLength, x1.causalDepthwiseConv1d(&ctx, .time, .channel, .tap, &k1, 2, &short_state));

    // Equivalence against groupedCausalConv1d (groups == channels ==
    // depthwise) at dilation=3, including both gradients.
    const xs = [_]f32{
        0.5,  -1.0,
        2.0,  0.25,
        -0.5, 1.5,
        1.0,  -2.0,
        3.0,  0.75,
        -1.5, 0.5,
        0.25, 2.5,
    };
    var input_dw = try Tensor(.{ .time, .channel }).variableFromSlice(&ctx, .{ 7, 2 }, &xs);
    defer input_dw.deinit();
    // Depthwise kernel [channel, tap].
    var kernel_dw = try Tensor(.{ .channel, .tap }).variableFromSlice(&ctx, .{ 2, 3 }, &.{
        1,  -2, 3,
        -4, 5,  6,
    });
    defer kernel_dw.deinit();
    var out_dw = try input_dw.causalDepthwiseConv1d(&ctx, .time, .channel, .tap, &kernel_dw, 3, null);
    defer out_dw.deinit();
    var loss_dw = try out_dw.sumAll(&ctx);
    defer loss_dw.deinit();
    try loss_dw.backward(&ctx);
    var gx_dw = (try input_dw.grad(&ctx)).?;
    defer gx_dw.deinit();
    var gk_dw = (try kernel_dw.grad(&ctx)).?;
    defer gk_dw.deinit();

    // Grouped arm: weight [tap, in_per_group=1, out] with
    // w[k][0][c] = kernel_dw[c][k].
    var input_g = try Tensor(.{ .time, .in }).variableFromSlice(&ctx, .{ 7, 2 }, &xs);
    defer input_g.deinit();
    var weight_g = try Tensor(.{ .tap, .ipg, .out }).variableFromSlice(&ctx, .{ 3, 1, 2 }, &.{
        1,  -4,
        -2, 5,
        3,  6,
    });
    defer weight_g.deinit();
    var out_g = try input_g.groupedCausalConv1d(&ctx, .time, .in, .tap, .ipg, .out, &weight_g, 3, 2, null);
    defer out_g.deinit();
    try std.testing.expectEqualSlices(f32, out_g.asRawTensor().dataConst(), out_dw.asRawTensor().dataConst());

    var loss_g = try out_g.sumAll(&ctx);
    defer loss_g.deinit();
    try loss_g.backward(&ctx);
    var gx_g = (try input_g.grad(&ctx)).?;
    defer gx_g.deinit();
    try std.testing.expectEqualSlices(f32, gx_g.asRawTensor().dataConst(), gx_dw.asRawTensor().dataConst());
    var gw_g = (try weight_g.grad(&ctx)).?;
    defer gw_g.deinit();
    // gk_dw is [channel, tap]; gw_g is [tap, 1, channel] — same values
    // transposed.
    const gk = gk_dw.asRawTensor().dataConst();
    const gw = gw_g.asRawTensor().dataConst();
    for (0..2) |c| {
        for (0..3) |k| {
            try std.testing.expectEqual(gk[c * 3 + k], gw[k * 2 + c]);
        }
    }
}

test "tagged public tensor general causal conv mixes channels with optional state" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .in }).fromSlice(&ctx, .{ 3, 2 }, &.{
        1, 10,
        2, 20,
        3, 30,
    });
    defer input.deinit();
    // w[k][i][o], k=1 is the newest tap.
    var weight = try Tensor(.{ .tap, .in, .out }).fromSlice(&ctx, .{ 2, 2, 2 }, &.{
        10, 20,
        30, 40,
        1,  2,
        3,  4,
    });
    defer weight.deinit();

    var no_state = try input.causalConv1d(&ctx, .time, .in, .tap, .out, &weight, 1, null);
    defer no_state.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        31,  42,
        372, 504,
        713, 966,
    }, no_state.asRawTensor().dataConst());

    var state = [_]f32{ 5, 7 };
    var with_state = try input.causalConv1d(&ctx, .time, .in, .tap, .out, &weight, 1, &state);
    defer with_state.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        291, 422,
        372, 504,
        713, 966,
    }, with_state.asRawTensor().dataConst());

    // Dilation 2, in=out=1: y[t] = 10*x[t-2] + x[t], state covers t<2 history.
    var mono = try Tensor(.{ .time, .in }).fromSlice(&ctx, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer mono.deinit();
    var taps2 = try Tensor(.{ .tap, .in, .out }).fromSlice(&ctx, .{ 2, 1, 1 }, &.{ 10, 1 });
    defer taps2.deinit();
    var dilated_state = [_]f32{ 100, 200 };
    var dilated = try mono.causalConv1d(&ctx, .time, .in, .tap, .out, &taps2, 2, &dilated_state);
    defer dilated.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1001, 2002, 13, 24 }, dilated.asRawTensor().dataConst());
}

test "tagged public tensor general causal conv propagates input and weight gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .in }).variableFromSlice(&ctx, .{ 3, 2 }, &.{
        1, 10,
        2, 20,
        3, 30,
    });
    defer input.deinit();
    var weight = try Tensor(.{ .tap, .in, .out }).variableFromSlice(&ctx, .{ 2, 2, 2 }, &.{
        10, 20,
        30, 40,
        1,  2,
        3,  4,
    });
    defer weight.deinit();
    var state = [_]f32{ 5, 7 };

    var out = try input.causalConv1d(&ctx, .time, .in, .tap, .out, &weight, 1, &state);
    defer out.deinit();
    var loss = try out.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try input.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        33, 77,
        33, 77,
        3,  7,
    }, gx.asRawTensor().dataConst());

    var gw = (try weight.grad(&ctx)).?;
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        8,  8,
        37, 37,
        6,  6,
        60, 60,
    }, gw.asRawTensor().dataConst());
}

test "tagged public tensor grouped causal conv partitions channels with state" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .in }).fromSlice(&ctx, .{ 3, 4 }, &.{
        1, 10, 100, 1000,
        2, 20, 200, 2000,
        3, 30, 300, 3000,
    });
    defer input.deinit();
    // w[k][local_input][out]; groups=2, so out 0/1 read input 0/1 and
    // out 2/3 read input 2/3.
    var weight = try Tensor(.{ .tap, .in_group, .out }).fromSlice(&ctx, .{ 2, 2, 4 }, &.{
        10,  20,   30,    40,
        1,   2,    3,     4,
        5,   6,    7,     8,
        0.5, 0.25, 0.125, 0.0625,
    });
    defer weight.deinit();
    var state = [_]f32{ 9, 90, 900, 9000 };

    var out = try input.groupedCausalConv1d(&ctx, .time, .in, .tap, .in_group, .out, &weight, 1, 2, &state);
    defer out.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        190, 368.5, 54825, 72862.5,
        40,  57,    7650,  9725,
        70,  105.5, 14475, 18587.5,
    }, out.asRawTensor().dataConst());
}

test "tagged public tensor grouped causal conv propagates input and weight gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .in }).variableFromSlice(&ctx, .{ 3, 4 }, &.{
        1, 10, 100, 1000,
        2, 20, 200, 2000,
        3, 30, 300, 3000,
    });
    defer input.deinit();
    var weight = try Tensor(.{ .tap, .in_group, .out }).variableFromSlice(&ctx, .{ 2, 2, 4 }, &.{
        10,  20,   30,    40,
        1,   2,    3,     4,
        5,   6,    7,     8,
        0.5, 0.25, 0.125, 0.0625,
    });
    defer weight.deinit();
    var state = [_]f32{ 9, 90, 900, 9000 };

    var out = try input.groupedCausalConv1d(&ctx, .time, .in, .tap, .in_group, .out, &weight, 1, 2, &state);
    defer out.deinit();
    var loss = try out.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try input.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        41, 3.75, 85, 7.1875,
        41, 3.75, 85, 7.1875,
        11, 0.75, 15, 0.1875,
    }, gx.asRawTensor().dataConst());

    var gw = (try weight.grad(&ctx)).?;
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        12,  12,  1200,  1200,
        120, 120, 12000, 12000,
        6,   6,   600,   600,
        60,  60,  6000,  6000,
    }, gw.asRawTensor().dataConst());
}

test "tagged public tensor general causal conv dilated gradients" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var input = try Tensor(.{ .time, .in }).variableFromSlice(&ctx, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    var weight = try Tensor(.{ .tap, .in, .out }).variableFromSlice(&ctx, .{ 2, 1, 1 }, &.{ 10, 1 });
    defer weight.deinit();
    var state = [_]f32{ 100, 200 };

    var out = try input.causalConv1d(&ctx, .time, .in, .tap, .out, &weight, 2, &state);
    defer out.deinit();
    var loss = try out.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);

    var gx = (try input.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 11, 1, 1 }, gx.asRawTensor().dataConst());

    var gw = (try weight.grad(&ctx)).?;
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 303, 10 }, gw.asRawTensor().dataConst());
}

test "public Tensor conv1d facade matches ctx.conv1dAxisRank" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{ .time, .cin }).fromSlice(&ctx, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w = try Tensor(.{ .tap, .cin, .cout }).fromSlice(&ctx, .{ 3, 1, 1 }, &.{ 1, 2, 3 });
    defer w.deinit();

    var y = try x.conv1d(&ctx, .time, .cin, .tap, .cout, &w, 1, 1, 1, 1);
    defer y.deinit();
    // PyTorch: F.conv1d([1,2,3,4], [1,2,3], padding=1) = [8, 14, 20, 11].
    try expectCloseSlices(&.{ 8, 14, 20, 11 }, y.asRawTensor().dataConst(), 1e-6);
}

const Conv1dFdContext = struct {
    ctx: *ExecContext,
    x_vals: []f32,
    w_vals: []f32,
    coef: []const f32,
    seq: usize,
    in_ch: usize,
    ipg: usize,
    out_ch: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
};

fn conv1dFdLoss(c: Conv1dFdContext) anyerror!f32 {
    var x = try c.ctx.fromSlice(&.{ c.seq, c.in_ch }, c.x_vals);
    defer x.deinit();
    var w = try c.ctx.fromSlice(&.{ c.taps, c.ipg, c.out_ch }, c.w_vals);
    defer w.deinit();
    var y = try c.ctx.conv1dAxisRank(2, &x, &w, 0, 1, c.stride, c.pad, c.dilation, c.groups);
    defer y.deinit();
    return fdWeightedSum(y.dataConst(), c.coef);
}

test "tagged autograd conv1d matches finite differences across stride/pad/dilation/groups" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // {stride, pad, dilation, groups, taps}: plain, strided+padded, dilated
    // with long pad, the grouped k=8 p=64 shape, the stride>1 AND dilation>1
    // interaction (exercises the backward-input divisibility skip), and a
    // grouped case with in_per_group > 1.
    const configs = [_][5]usize{
        .{ 1, 0, 1, 1, 3 },
        .{ 2, 3, 1, 1, 3 },
        .{ 1, 6, 3, 1, 5 },
        .{ 1, 64, 1, 4, 8 },
        .{ 2, 3, 3, 1, 3 },
        .{ 3, 2, 2, 2, 3 },
    };
    const seq: usize = 10;
    for (configs) |cfg| {
        const stride = cfg[0];
        const pad = cfg[1];
        const dilation = cfg[2];
        const groups = cfg[3];
        const taps = cfg[4];
        const in_ch: usize = switch (groups) {
            4 => 4,
            2 => 4,
            else => 2,
        };
        const out_ch: usize = switch (groups) {
            4 => 4,
            2 => 6,
            else => 3,
        };
        const ipg = in_ch / groups;
        const out_len = (seq + 2 * pad - (dilation * (taps - 1) + 1)) / stride + 1;

        const x_vals = try allocator.alloc(f32, seq * in_ch);
        defer allocator.free(x_vals);
        fdFillPattern(x_vals, 0.3);
        const w_vals = try allocator.alloc(f32, taps * ipg * out_ch);
        defer allocator.free(w_vals);
        fdFillPattern(w_vals, 1.7);
        const coef = try allocator.alloc(f32, out_len * out_ch);
        defer allocator.free(coef);
        fdFillPattern(coef, 2.9);

        var x = try Tensor(.{ .time, .cin }).variable(&ctx, try ctx.fromSlice(&.{ seq, in_ch }, x_vals));
        defer x.deinit();
        var w = try Tensor(.{ .tap, .cin, .cout }).variable(&ctx, try ctx.fromSlice(&.{ taps, ipg, out_ch }, w_vals));
        defer w.deinit();
        var coef_t = try Tensor(.{ .time, .cout }).fromTensor(&ctx, try ctx.fromSlice(&.{ out_len, out_ch }, coef));
        defer coef_t.deinit();

        var y = try x.conv1d(&ctx, .time, .cin, .tap, .cout, &w, stride, pad, dilation, groups);
        defer y.deinit();
        var weighted = try y.mul(&ctx, &coef_t);
        defer weighted.deinit();
        var loss = try weighted.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);

        var gx = (try x.grad(&ctx)).?;
        defer gx.deinit();
        var gw = (try w.grad(&ctx)).?;
        defer gw.deinit();

        const fd_ctx = Conv1dFdContext{
            .ctx = &ctx,
            .x_vals = x_vals,
            .w_vals = w_vals,
            .coef = coef,
            .seq = seq,
            .in_ch = in_ch,
            .ipg = ipg,
            .out_ch = out_ch,
            .taps = taps,
            .stride = stride,
            .pad = pad,
            .dilation = dilation,
            .groups = groups,
        };
        try fdCheckGrad(x_vals, gx.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, conv1dFdLoss);
        try fdCheckGrad(w_vals, gw.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, conv1dFdLoss);
    }
}

test "public Tensor convTranspose1d facade upsamples with bias" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    // t_in=2, IC=1, OC=1, K=2, stride=2, pad=0, output_pad=1: y (before bias)
    // = [x0*w_k0, x0*w_k1, x1*w_k0, x1*w_k1, 0] with weight2 rows [(oc*K + k), IC].
    var x = try Tensor(.{ .time, .cin }).fromSlice(&ctx, .{ 2, 1 }, &.{ 1, 2 });
    defer x.deinit();
    var w2 = try Tensor(.{ .kout, .cin }).fromSlice(&ctx, .{ 2, 1 }, &.{ 10, 20 });
    defer w2.deinit();
    var bias = try Tensor(.{.cout}).fromSlice(&ctx, .{1}, &.{100});
    defer bias.deinit();

    var y = try x.convTranspose1d(&ctx, .time, .cin, .kout, .cout, &w2, &bias, 1, 2, 2, 0, 1);
    defer y.deinit();
    try expectCloseSlices(&.{ 110, 120, 120, 140, 100 }, y.asRawTensor().dataConst(), 1e-6);
}

const ConvTranspose1dFdContext = struct {
    ctx: *ExecContext,
    x_vals: []f32,
    w_vals: []f32,
    b_vals: ?[]f32,
    coef: []const f32,
    t_in: usize,
    in_ch: usize,
    out_ch: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    output_pad: usize,
};

fn convTranspose1dFdLoss(c: ConvTranspose1dFdContext) anyerror!f32 {
    var x = try c.ctx.fromSlice(&.{ c.t_in, c.in_ch }, c.x_vals);
    defer x.deinit();
    var w2 = try c.ctx.fromSlice(&.{ c.taps * c.out_ch, c.in_ch }, c.w_vals);
    defer w2.deinit();
    var bias: ?RawTensor = if (c.b_vals) |bv| try c.ctx.fromSlice(&.{c.out_ch}, bv) else null;
    defer if (bias) |*b| b.deinit();
    var y = try c.ctx.convTranspose1d(&x, &w2, if (bias) |*b| b else null, c.out_ch, c.taps, c.stride, c.pad, c.output_pad);
    defer y.deinit();
    return fdWeightedSum(y.dataConst(), c.coef);
}

test "tagged autograd convTranspose1d matches finite differences at DAC configs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // {stride, taps, pad, output_pad}: two of the DAC decoder configs.
    const configs = [_][4]usize{
        .{ 8, 16, 4, 0 },
        .{ 5, 10, 3, 1 },
    };
    const t_in: usize = 3;
    const in_ch: usize = 2;
    const out_ch: usize = 2;
    for (configs) |cfg| {
        const stride = cfg[0];
        const taps = cfg[1];
        const pad = cfg[2];
        const output_pad = cfg[3];
        const out_len = (t_in - 1) * stride + taps - 2 * pad + output_pad;

        for ([_]bool{ true, false }) |with_bias| {
            const x_vals = try allocator.alloc(f32, t_in * in_ch);
            defer allocator.free(x_vals);
            fdFillPattern(x_vals, 0.9);
            const w_vals = try allocator.alloc(f32, taps * out_ch * in_ch);
            defer allocator.free(w_vals);
            fdFillPattern(w_vals, 4.2);
            const b_vals = try allocator.alloc(f32, out_ch);
            defer allocator.free(b_vals);
            fdFillPattern(b_vals, 6.1);
            const coef = try allocator.alloc(f32, out_len * out_ch);
            defer allocator.free(coef);
            fdFillPattern(coef, 7.4);

            var x = try Tensor(.{ .time, .cin }).variable(&ctx, try ctx.fromSlice(&.{ t_in, in_ch }, x_vals));
            defer x.deinit();
            var w2 = try Tensor(.{ .kout, .cin }).variable(&ctx, try ctx.fromSlice(&.{ taps * out_ch, in_ch }, w_vals));
            defer w2.deinit();
            var bias: ?Tensor(.{.cout}) = if (with_bias) try Tensor(.{.cout}).variable(&ctx, try ctx.fromSlice(&.{out_ch}, b_vals)) else null;
            defer if (bias) |*b| b.deinit();
            var coef_t = try Tensor(.{ .time, .cout }).fromTensor(&ctx, try ctx.fromSlice(&.{ out_len, out_ch }, coef));
            defer coef_t.deinit();

            var y = try x.convTranspose1d(&ctx, .time, .cin, .kout, .cout, &w2, if (bias) |*b| b else null, out_ch, taps, stride, pad, output_pad);
            defer y.deinit();
            var weighted = try y.mul(&ctx, &coef_t);
            defer weighted.deinit();
            var loss = try weighted.sumAll(&ctx);
            defer loss.deinit();
            try loss.backward(&ctx);

            var gx = (try x.grad(&ctx)).?;
            defer gx.deinit();
            var gw = (try w2.grad(&ctx)).?;
            defer gw.deinit();

            const fd_ctx = ConvTranspose1dFdContext{
                .ctx = &ctx,
                .x_vals = x_vals,
                .w_vals = w_vals,
                .b_vals = if (with_bias) b_vals else null,
                .coef = coef,
                .t_in = t_in,
                .in_ch = in_ch,
                .out_ch = out_ch,
                .taps = taps,
                .stride = stride,
                .pad = pad,
                .output_pad = output_pad,
            };
            try fdCheckGrad(x_vals, gx.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, convTranspose1dFdLoss);
            try fdCheckGrad(w_vals, gw.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, convTranspose1dFdLoss);
            if (with_bias) {
                var gb = (try bias.?.grad(&ctx)).?;
                defer gb.deinit();
                // The bias broadcast onto ALL output rows including the
                // output_pad ones — the FD loss sees exactly that.
                try fdCheckGrad(b_vals, gb.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, convTranspose1dFdLoss);
            }
        }
    }
}

test "public Tensor snake facade applies per-channel activation" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    var x = try Tensor(.{ .time, .ch }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -1.0, 2.0, 0.0 });
    defer x.deinit();
    var alpha = try Tensor(.{.ch}).fromSlice(&ctx, .{2}, &.{ 1.0, 2.0 });
    defer alpha.deinit();
    var inv_b = try Tensor(.{.ch}).fromSlice(&ctx, .{2}, &.{ 1.0, 0.5 });
    defer inv_b.deinit();

    var y = try x.snake(&ctx, .ch, &alpha, &inv_b);
    defer y.deinit();
    const s05 = @sin(@as(f32, 0.5));
    const sm2 = @sin(@as(f32, -2.0));
    const s4 = @sin(@as(f32, 2.0));
    try expectCloseSlices(&.{
        0.5 + s05 * s05,
        -1.0 + 0.5 * (sm2 * sm2),
        2.0 + s4 * s4,
        0.0,
    }, y.asRawTensor().dataConst(), 1e-6);
}

const SnakeFdContext = struct {
    ctx: *ExecContext,
    x_vals: []f32,
    a_vals: []f32,
    ib_vals: []f32,
    coef: []const f32,
    rows: usize,
    cols: usize,
};

fn snakeFdLoss(c: SnakeFdContext) anyerror!f32 {
    var x = try c.ctx.fromSlice(&.{ c.rows, c.cols }, c.x_vals);
    defer x.deinit();
    var alpha = try c.ctx.fromSlice(&.{c.cols}, c.a_vals);
    defer alpha.deinit();
    var inv_b = try c.ctx.fromSlice(&.{c.cols}, c.ib_vals);
    defer inv_b.deinit();
    var y = try c.ctx.snakeRows(&x, &alpha, &inv_b);
    defer y.deinit();
    return fdWeightedSum(y.dataConst(), c.coef);
}

test "tagged autograd snake matches finite differences for input alpha and inv_b" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // cols=13: not a multiple of any SIMD width.
    const rows: usize = 3;
    const cols: usize = 13;

    const x_vals = try allocator.alloc(f32, rows * cols);
    defer allocator.free(x_vals);
    fdFillPattern(x_vals, 0.5);
    const a_vals = try allocator.alloc(f32, cols);
    defer allocator.free(a_vals);
    fdFillPattern(a_vals, 3.3);
    for (a_vals) |*a| a.* = @abs(a.*) + 0.25;
    const ib_vals = try allocator.alloc(f32, cols);
    defer allocator.free(ib_vals);
    for (ib_vals, a_vals) |*ib, a| ib.* = 1.0 / (a + 1e-9);
    const coef = try allocator.alloc(f32, rows * cols);
    defer allocator.free(coef);
    fdFillPattern(coef, 5.8);

    var x = try Tensor(.{ .time, .ch }).variable(&ctx, try ctx.fromSlice(&.{ rows, cols }, x_vals));
    defer x.deinit();
    var alpha = try Tensor(.{.ch}).variable(&ctx, try ctx.fromSlice(&.{cols}, a_vals));
    defer alpha.deinit();
    var inv_b = try Tensor(.{.ch}).variable(&ctx, try ctx.fromSlice(&.{cols}, ib_vals));
    defer inv_b.deinit();
    var coef_t = try Tensor(.{ .time, .ch }).fromTensor(&ctx, try ctx.fromSlice(&.{ rows, cols }, coef));
    defer coef_t.deinit();

    var y = try x.snake(&ctx, .ch, &alpha, &inv_b);
    defer y.deinit();
    var weighted = try y.mul(&ctx, &coef_t);
    defer weighted.deinit();
    var loss = try weighted.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    var ga = (try alpha.grad(&ctx)).?;
    defer ga.deinit();
    var gib = (try inv_b.grad(&ctx)).?;
    defer gib.deinit();

    const fd_ctx = SnakeFdContext{
        .ctx = &ctx,
        .x_vals = x_vals,
        .a_vals = a_vals,
        .ib_vals = ib_vals,
        .coef = coef,
        .rows = rows,
        .cols = cols,
    };
    try fdCheckGrad(x_vals, gx.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, snakeFdLoss);
    try fdCheckGrad(a_vals, ga.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, snakeFdLoss);
    try fdCheckGrad(ib_vals, gib.asRawTensor().dataConst(), 1e-2, 1e-2, fd_ctx, snakeFdLoss);
}
