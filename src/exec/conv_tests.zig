//! Convolution kernels through ExecContext: conv2d with winograd prepared
//! weights, conv1d, col2im1d, and convTranspose1d, forward and backward,
//! with geometry rejection. Force-imported by `exec.zig`.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const exec = @import("../exec.zig");
const exec_elementwise = @import("elementwise.zig");
const exec_row_ops = @import("row_ops.zig");
const exec_moe_chain = @import("moe_chain.zig");
const dtype_mod = @import("../dtype.zig");
const fpenv = @import("../fpenv.zig");
const parallel = @import("../parallel.zig");
const rng = @import("../rng.zig");
const tensor = @import("../tensor.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;
const ExecContext = exec.ExecContext;
const LayoutClass = exec.LayoutClass;
const CrossEntropyOptions = exec.CrossEntropyOptions;
const Reduction = exec.Reduction;

// --- conv2d: hand-computed cases; run under native and -Dbackend=scalar ---

test "conv2d: pointwise 1x1 (groups=1) with bias" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // input [H=2, W=1, Cin=2]: h0=[1,2], h1=[3,4]
    var input = try ctx.fromSlice(.f32, .{ 2, 1, 2 }, &[_]f32{ 1, 2, 3, 4 });
    defer input.deinit();
    // weight [Cout=1, KH=1, KW=1, Cinpg=2] = [5,6]
    var weight = try ctx.fromSlice(.f32, .{ 1, 1, 1, 2 }, &[_]f32{ 5, 6 });
    defer weight.deinit();
    var bias = try ctx.fromSlice(.f32, .{1}, &[_]f32{100});
    defer bias.deinit();

    var out = try ctx.conv2d(&input, &weight, &bias, .{ 1, 1 }, .{ 0, 0 }, 1);
    defer out.deinit();
    // out[h] = in[h,0]*5 + in[h,1]*6 + 100
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1 * 5 + 2 * 6 + 100, 3 * 5 + 4 * 6 + 100 }, out.dataConst());
}

test "conv2d: 3x3 same-padding (stride 1, pad 1), ones kernel on ones image" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // input [3,3,1] all ones; weight [1,3,3,1] all ones; pad 1 -> output 3x3.
    var input = try ctx.fromSlice(.f32, .{ 3, 3, 1 }, &[_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1 });
    defer input.deinit();
    var weight = try ctx.fromSlice(.f32, .{ 1, 3, 3, 1 }, &[_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1 });
    defer weight.deinit();

    var out = try ctx.conv2d(&input, &weight, null, .{ 1, 1 }, .{ 1, 1 }, 1);
    defer out.deinit();
    // neighborhood sums with zero pad: corners 4, edges 6, center 9.
    try std.testing.expectEqualSlices(f32, &[_]f32{ 4, 6, 4, 6, 9, 6, 4, 6, 4 }, out.dataConst());
}

test "conv2d: depthwise (groups=Cin) with stride 2" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // input [H=4, W=1, Cin=2], interleaved h*2+c: ch0=[1,2,3,4], ch1=[10,20,30,40]
    var input = try ctx.fromSlice(.f32, .{ 4, 1, 2 }, &[_]f32{ 1, 10, 2, 20, 3, 30, 4, 40 });
    defer input.deinit();
    // weight [Cout=2, KH=2, KW=1, Cinpg=1] all ones; oc0->ch0, oc1->ch1 (depthwise).
    var weight = try ctx.fromSlice(.f32, .{ 2, 2, 1, 1 }, &[_]f32{ 1, 1, 1, 1 });
    defer weight.deinit();

    var out = try ctx.conv2d(&input, &weight, null, .{ 2, 1 }, .{ 0, 0 }, 2);
    defer out.deinit();
    // OH=(4-2)/2+1=2. oh0: ch0=1+2=3, ch1=10+20=30 ; oh1: ch0=3+4=7, ch1=30+40=70.
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3, 30, 7, 70 }, out.dataConst());
}

test "conv2d: rejects mismatched groups / shapes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var input = try ctx.fromSlice(.f32, .{ 2, 1, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();
    var weight = try ctx.fromSlice(.f32, .{ 1, 1, 1, 2 }, &[_]f32{ 1, 1 }); // Cinpg=2 != Cin/groups=3
    defer weight.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv2d(&input, &weight, null, .{ 1, 1 }, .{ 0, 0 }, 1));
}

test "conv2d prepared winograd weights: exec-level parity, .empty fallback, shape mismatch" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Pin the Winograd route ON (BLAS builds default it off).
    const exec_conv = @import("conv.zig");
    exec_conv.setWinogradForTest(true);
    defer exec_conv.setWinogradForTest(null);

    // 8x8x8 -> F2-eligible 3x3 s1 p1 conv.
    const h = 8;
    const w = 8;
    const cin = 8;
    const cout = 4;
    const xd = try allocator.alloc(f32, h * w * cin);
    defer allocator.free(xd);
    const wd = try allocator.alloc(f32, cout * 9 * cin);
    defer allocator.free(wd);
    const bd = try allocator.alloc(f32, cout);
    defer allocator.free(bd);
    rng.gaussianFill(7, xd, 1.0);
    rng.gaussianFill(8, wd, 0.5);
    rng.gaussianFill(9, bd, 0.5);

    var input = try ctx.fromSlice(.f32, .{ h, w, cin }, xd);
    defer input.deinit();
    var weight = try ctx.fromSlice(.f32, .{ cout, 3, 3, cin }, wd);
    defer weight.deinit();
    var bias = try ctx.fromSlice(.f32, .{cout}, bd);
    defer bias.deinit();

    var ref = try ctx.conv2d(&input, &weight, &bias, .{ 1, 1 }, .{ 1, 1 }, 1);
    defer ref.deinit();

    var prep = try ctx.prepareConv2dWeights(&weight);
    defer prep.deinit();
    try std.testing.expect(prep.f2 != null);
    var got = try ctx.conv2dPrepared(&input, &weight, &prep, &bias, .{ 1, 1 }, .{ 1, 1 }, 1);
    defer got.deinit();
    try std.testing.expectEqualSlices(f32, ref.dataConst(), got.dataConst());

    // Fused-relu entry against the same planes.
    var ref_relu = try ctx.conv2dRelu(&input, &weight, &bias, .{ 1, 1 }, .{ 1, 1 }, 1);
    defer ref_relu.deinit();
    var got_relu = try ctx.conv2dPreparedRelu(&input, &weight, &prep, &bias, .{ 1, 1 }, .{ 1, 1 }, 1);
    defer got_relu.deinit();
    try std.testing.expectEqualSlices(f32, ref_relu.dataConst(), got_relu.dataConst());

    // `.empty` is inert: conv2dPrepared falls back to the per-call
    // transform and produces the same bytes.
    var empty = ExecContext.PreparedConvWeights.empty;
    defer empty.deinit();
    var got_empty = try exec_conv.conv2dPrepared(&ctx, &input, &weight, &empty, &bias, .{ 1, 1 }, .{ 1, 1 }, 1);
    defer got_empty.deinit();
    try std.testing.expectEqualSlices(f32, ref.dataConst(), got_empty.dataConst());

    // Planes prepared for a DIFFERENT weight shape are rejected on the
    // Winograd route (caller wiring bug, not a fallback case).
    const wd2 = try allocator.alloc(f32, cout * 9 * cin * 2);
    defer allocator.free(wd2);
    rng.gaussianFill(10, wd2, 0.5);
    var weight2 = try ctx.fromSlice(.f32, .{ cout, 3, 3, cin * 2 }, wd2);
    defer weight2.deinit();
    var prep2 = try ctx.prepareConv2dWeights(&weight2);
    defer prep2.deinit();
    try std.testing.expect(prep2.f2 != null and prep2.cin == cin * 2);
    try std.testing.expectError(
        tensor.TensorError.ShapeMismatch,
        ctx.conv2dPrepared(&input, &weight, &prep2, &bias, .{ 1, 1 }, .{ 1, 1 }, 1),
    );
}

test "conv1d: hand-computed same-pad, stride+dilation, and grouped cases" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // k=3, s=1, p=1, d=1: x=[1,2,3,4], w=[1,2,3].
    // PyTorch: F.conv1d(x, w, padding=1) = [8, 14, 20, 11].
    var x = try ctx.fromSlice(.f32, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w = try ctx.fromSlice(.f32, .{ 3, 1, 1 }, &.{ 1, 2, 3 });
    defer w.deinit();
    var y = try ctx.conv1d(2, &x, &w, 0, 1, 1, 1, 1, 1);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 8, 14, 20, 11 }, y.dataConst());

    // k=2, s=2, p=0, d=2: T_out = (5 - 2 - 1)/2 + 1 = 2; y[t] = 10*x[2t] + x[2t+2].
    // PyTorch: F.conv1d([1..5], [10,1], stride=2, dilation=2) = [13, 35].
    var xs = try ctx.fromSlice(.f32, .{ 5, 1 }, &.{ 1, 2, 3, 4, 5 });
    defer xs.deinit();
    var ws = try ctx.fromSlice(.f32, .{ 2, 1, 1 }, &.{ 10, 1 });
    defer ws.deinit();
    var ys = try ctx.conv1d(2, &xs, &ws, 0, 1, 2, 0, 2, 1);
    defer ys.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 13, 35 }, ys.dataConst());

    // groups=2, k=1: in=[a,b] rows, w [1, 1, 2] = [[10, 100]] so
    // y[t] = [10*a, 100*b] (each output channel sees only its group's input).
    var xg = try ctx.fromSlice(.f32, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer xg.deinit();
    var wg = try ctx.fromSlice(.f32, .{ 1, 1, 2 }, &.{ 10, 100 });
    defer wg.deinit();
    var yg = try ctx.conv1d(2, &xg, &wg, 0, 1, 1, 0, 1, 2);
    defer yg.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 200, 30, 400 }, yg.dataConst());

    // Channel mixing, no pad: in=2 channels, out=1, k=2.
    // y[t] = 1*x[t,0] + 2*x[t,1] + 3*x[t+1,0] + 4*x[t+1,1]
    // (tap k reads xpad[t*s + k*d], so k=0 is the OLDEST sample — non-causal
    // orientation, unlike causalConv1d where the last tap is newest).
    var xm = try ctx.fromSlice(.f32, .{ 3, 2 }, &.{ 1, 10, 2, 20, 3, 30 });
    defer xm.deinit();
    var wm = try ctx.fromSlice(.f32, .{ 2, 2, 1 }, &.{ 1, 2, 3, 4 });
    defer wm.deinit();
    var ym = try ctx.conv1d(2, &xm, &wm, 0, 1, 1, 0, 1, 1);
    defer ym.deinit();
    // t=0: 1*1 + 2*10 + 3*2 + 4*20 = 107 ; t=1: 1*2 + 2*20 + 3*3 + 4*30 = 171.
    try std.testing.expectEqualSlices(f32, &.{ 107, 171 }, ym.dataConst());
}

test "conv1d: rejects invalid geometry and mismatched shapes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, .{ 4, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer x.deinit();
    var w = try ctx.fromSlice(.f32, .{ 2, 2, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer w.deinit();

    // stride/dilation must be positive.
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.conv1d(2, &x, &w, 0, 1, 0, 0, 1, 1));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.conv1d(2, &x, &w, 0, 1, 1, 0, 0, 1));
    // groups must divide in and out channels.
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv1d(2, &x, &w, 0, 1, 1, 0, 1, 3));
    // weight in_per_group mismatch: groups=2 wants weight.shape[1] == 1.
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv1d(2, &x, &w, 0, 1, 1, 0, 1, 2));
    // Receptive field longer than the padded input: T + 2p < d*(K-1)+1.
    var wide = try ctx.fromSlice(.f32, .{ 6, 2, 2 }, &([_]f32{0.5} ** 24));
    defer wide.deinit();
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.conv1d(2, &x, &wide, 0, 1, 1, 0, 1, 1));
}

test "col2im1d: hand-computed gather with crop and output_pad" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // t_in=2, taps=2, oc=1, stride=2: col rows [10,20],[30,40] (column = k).
    var col = try ctx.fromSlice(.f32, .{ 2, 2 }, &.{ 10, 20, 30, 40 });
    defer col.deinit();

    // pad=0: T_out = 4, disjoint scatter.
    var y = try ctx.col2im1d(&col, 1, 2, 2, 0, 0);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30, 40 }, y.dataConst());

    // pad=1: crops one frame each side -> [20, 30]; output_pad=1 appends a zero row.
    var yc = try ctx.col2im1d(&col, 1, 2, 2, 1, 1);
    defer yc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 20, 30, 0 }, yc.dataConst());

    // Rejects col width != taps*out_channels and degenerate T_out.
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.col2im1d(&col, 2, 2, 2, 0, 0));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.col2im1d(&col, 1, 2, 1, 2, 0));
}

test "convTranspose1d: matches a naive direct scatter reference (DAC combos)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // (stride, taps, pad, output_pad): two of the DAC decoder combos plus a
    // small overlapping case.
    const combos = [_][4]usize{
        .{ 2, 4, 1, 0 },
        .{ 3, 6, 2, 1 },
        .{ 1, 3, 1, 0 },
    };
    for (combos) |combo| {
        const stride = combo[0];
        const taps = combo[1];
        const pad = combo[2];
        const output_pad = combo[3];
        const t_in: usize = 5;
        const in_channels: usize = 3;
        const out_channels: usize = 2;
        const t_out = (t_in - 1) * stride + taps - 2 * pad;
        const out_len = t_out + output_pad;

        var x = try ctx.zeros(.f32, .{ t_in, in_channels });
        defer x.deinit();
        for (x.data()) |*v| v.* = random.float(f32) * 2 - 1;
        // weight2[(oc*K + k)*IC + ic] — the reference's load-time repack.
        var w2 = try ctx.zeros(.f32, .{ taps * out_channels, in_channels });
        defer w2.deinit();
        for (w2.data()) |*v| v.* = random.float(f32) * 2 - 1;
        var bias = try ctx.zeros(.f32, .{out_channels});
        defer bias.deinit();
        for (bias.data()) |*v| v.* = random.float(f32);

        var got = try ctx.convTranspose1d(&x, &w2, &bias, out_channels, taps, stride, pad, output_pad);
        defer got.deinit();
        try std.testing.expectEqualSlices(usize, &.{ out_len, out_channels }, got.shape.slice());

        // Naive direct ConvTranspose1d: every input frame scatters its taps to
        // t_in*stride + k - pad, dropping positions outside [0, T_out); the
        // output_pad rows stay at bias.
        const want = try allocator.alloc(f32, out_len * out_channels);
        defer allocator.free(want);
        for (0..out_len) |t| {
            for (0..out_channels) |oc| want[t * out_channels + oc] = bias.dataConst()[oc];
        }
        const xd = x.dataConst();
        const wd = w2.dataConst();
        for (0..t_in) |ti| {
            for (0..taps) |k| {
                const pos = ti * stride + k;
                if (pos < pad) continue;
                const t = pos - pad;
                if (t >= t_out) continue;
                for (0..out_channels) |oc| {
                    var acc: f32 = 0;
                    for (0..in_channels) |ic| {
                        acc += xd[ti * in_channels + ic] * wd[(oc * taps + k) * in_channels + ic];
                    }
                    want[t * out_channels + oc] += acc;
                }
            }
        }
        for (want, got.dataConst()) |wv, gv| {
            try std.testing.expectApproxEqAbs(wv, gv, 1e-5);
        }
    }
}

test "convTranspose1d: rejects mismatched weight2 and bias shapes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w2 = try ctx.fromSlice(.f32, .{ 4, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer w2.deinit();

    // weight2 rows must be taps*out_channels (here 2*1=2 != 4 for taps=2, oc=1).
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.convTranspose1d(&x, &w2, null, 1, 2, 2, 0, 0));
    // bias length must equal out_channels.
    var bad_bias = try ctx.fromSlice(.f32, .{3}, &.{ 1, 2, 3 });
    defer bad_bias.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.convTranspose1d(&x, &w2, &bad_bias, 2, 2, 2, 0, 0));
    // stride 0 rejected.
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.convTranspose1d(&x, &w2, null, 2, 2, 0, 0, 0));
}

// NOTE: this test stays EXACT (expectEqualSlices) across the direct-gather
// and GEMM+im2col/col2im backward routes: every operand is an integer-valued
// f32 and every reduction is a sum of integer products well below 2^24, so
// f32 accumulation is exact in ANY association order. Keep the values
// integer if you edit the cases — fractional values would make the
// route-dependent summation order observable.
test "conv2d backward: hand-computed input/weight gradients (channel-last)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Case A: 3x3x1 input, weight [cout1,kh2,kw2,cin1]=[[1,0],[0,2]], gy=ones 2x2, s1 p0 g1.
    var x = try ctx.fromSlice(.f32, .{ 3, 3, 1 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer x.deinit();
    var w = try ctx.fromSlice(.f32, .{ 1, 2, 2, 1 }, &.{ 1, 0, 0, 2 });
    defer w.deinit();
    var gy = try ctx.fromSlice(.f32, .{ 2, 2, 1 }, &.{ 1, 1, 1, 1 });
    defer gy.deinit();
    var gx = try ctx.conv2dBackwardInput(&gy, &w, 3, 3, .{ 1, 1 }, .{ 0, 0 }, 1);
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 0, 1, 3, 2, 0, 2, 2 }, gx.dataConst());
    var gw = try ctx.conv2dBackwardWeight(&x, &gy, 2, 2, .{ 1, 1 }, .{ 0, 0 }, 1);
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 12, 16, 24, 28 }, gw.dataConst());

    // Case B: stride 2. 4x4x1 input [1..16], weight ones 2x2, gy=ones 2x2, s2 p0 g1.
    // Every input pixel is read by exactly one output window -> gx all ones;
    // gw[kh,kw] = sum of in[2oh+kh, 2ow+kw].
    var x2 = try ctx.fromSlice(.f32, .{ 4, 4, 1 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    defer x2.deinit();
    var w2 = try ctx.fromSlice(.f32, .{ 1, 2, 2, 1 }, &.{ 1, 1, 1, 1 });
    defer w2.deinit();
    var gy2 = try ctx.fromSlice(.f32, .{ 2, 2, 1 }, &.{ 1, 1, 1, 1 });
    defer gy2.deinit();
    var gx2 = try ctx.conv2dBackwardInput(&gy2, &w2, 4, 4, .{ 2, 2 }, .{ 0, 0 }, 1);
    defer gx2.deinit();
    try std.testing.expectEqualSlices(f32, &([_]f32{1} ** 16), gx2.dataConst());
    var gw2 = try ctx.conv2dBackwardWeight(&x2, &gy2, 2, 2, .{ 2, 2 }, .{ 0, 0 }, 1);
    defer gw2.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 24, 28, 40, 44 }, gw2.dataConst());

    // Case C: depthwise 1x1 (groups=cin=cout=2). in [1,1,2]=[3,5], w[c]=[2,4], gy=[1,1].
    // gx[c] = gy[c]*w[c] = [2,4]; gw[c] = gy[c]*in[c] = [3,5].
    var x3 = try ctx.fromSlice(.f32, .{ 1, 1, 2 }, &.{ 3, 5 });
    defer x3.deinit();
    var w3 = try ctx.fromSlice(.f32, .{ 2, 1, 1, 1 }, &.{ 2, 4 });
    defer w3.deinit();
    var gy3 = try ctx.fromSlice(.f32, .{ 1, 1, 2 }, &.{ 1, 1 });
    defer gy3.deinit();
    var gx3 = try ctx.conv2dBackwardInput(&gy3, &w3, 1, 1, .{ 1, 1 }, .{ 0, 0 }, 2);
    defer gx3.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4 }, gx3.dataConst());
    var gw3 = try ctx.conv2dBackwardWeight(&x3, &gy3, 1, 1, .{ 1, 1 }, .{ 0, 0 }, 2);
    defer gw3.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 5 }, gw3.dataConst());
}

test "conv2d backward GEMM routes match the direct gather kernels" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Pin the GEMM route ON so every groups == 1 case below exercises it
    // (the depthwise case stays on the direct kernel either way).
    const exec_conv = @import("conv.zig");
    exec_conv.setConvBwdGemmForTest(true);
    defer exec_conv.setConvBwdGemmForTest(null);

    const vector_conv = @import("../backend/vector/conv.zig");

    const Case = struct {
        h: usize,
        w: usize,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
        groups: usize,
    };
    const cases = [_]Case{
        .{ .h = 6, .w = 5, .cin = 8, .cout = 4, .k = 1, .stride = 1, .pad = 0, .groups = 1 }, // 1x1 s1 p0 plain-GEMM route
        .{ .h = 7, .w = 5, .cin = 6, .cout = 3, .k = 1, .stride = 2, .pad = 0, .groups = 1 }, // 1x1 s2: general GEMM route
        .{ .h = 6, .w = 6, .cin = 5, .cout = 4, .k = 2, .stride = 1, .pad = 0, .groups = 1 },
        .{ .h = 8, .w = 7, .cin = 4, .cout = 6, .k = 3, .stride = 1, .pad = 1, .groups = 1 },
        .{ .h = 9, .w = 8, .cin = 4, .cout = 5, .k = 3, .stride = 2, .pad = 1, .groups = 1 },
        .{ .h = 9, .w = 9, .cin = 3, .cout = 4, .k = 5, .stride = 1, .pad = 2, .groups = 1 },
        .{ .h = 8, .w = 8, .cin = 6, .cout = 6, .k = 3, .stride = 1, .pad = 1, .groups = 6 }, // depthwise: direct route
    };

    for (cases, 0..) |case, case_i| {
        const cin_pg = case.cin / case.groups;
        const oh = (case.h + 2 * case.pad - case.k) / case.stride + 1;
        const ow = (case.w + 2 * case.pad - case.k) / case.stride + 1;

        var input = try ctx.empty(.f32, .{ case.h, case.w, case.cin });
        defer input.deinit();
        rng.gaussianFill(100 + case_i, input.data(), 1.0);
        var weight = try ctx.empty(.f32, .{ case.cout, case.k, case.k, cin_pg });
        defer weight.deinit();
        rng.gaussianFill(200 + case_i, weight.data(), 1.0);
        var gy = try ctx.empty(.f32, .{ oh, ow, case.cout });
        defer gy.deinit();
        rng.gaussianFill(300 + case_i, gy.data(), 1.0);

        const d: backend_mod.Conv2dDims = .{
            .h = case.h,
            .w = case.w,
            .cin = case.cin,
            .oh = oh,
            .ow = ow,
            .cout = case.cout,
            .kh = case.k,
            .kw = case.k,
            .stride_h = case.stride,
            .stride_w = case.stride,
            .pad_h = case.pad,
            .pad_w = case.pad,
            .groups = case.groups,
        };

        // Reference: the direct gather cores on the raw tensors, run serially.
        var gx_ref = try ctx.empty(.f32, .{ case.h, case.w, case.cin });
        defer gx_ref.deinit();
        vector_conv.conv2dBackwardInputInto(.{}, &gx_ref, &gy, &weight, d);
        var gw_ref = try ctx.empty(.f32, .{ case.cout, case.k, case.k, cin_pg });
        defer gw_ref.deinit();
        vector_conv.conv2dBackwardWeightInto(.{}, &gw_ref, &input, &gy, d);

        var gx = try ctx.conv2dBackwardInput(&gy, &weight, case.h, case.w, .{ case.stride, case.stride }, .{ case.pad, case.pad }, case.groups);
        defer gx.deinit();
        for (gx_ref.dataConst(), gx.dataConst()) |expected, got| {
            const tol = 1e-4 * @max(1.0, @abs(expected));
            try std.testing.expectApproxEqAbs(expected, got, tol);
        }

        var gw = try ctx.conv2dBackwardWeight(&input, &gy, case.k, case.k, .{ case.stride, case.stride }, .{ case.pad, case.pad }, case.groups);
        defer gw.deinit();
        for (gw_ref.dataConst(), gw.dataConst()) |expected, got| {
            const tol = 1e-4 * @max(1.0, @abs(expected));
            try std.testing.expectApproxEqAbs(expected, got, tol);
        }
    }
}

test "conv1d backward: hand-computed input/weight gradients + geometry rejection" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Adjoint of the k=3, s=1, p=1 forward hand case with gy = ones:
    // gx[ti] = sum of the taps that touch x[ti], gw[k] = sum of the x rows
    // tap k reads.
    var x = try ctx.fromSlice(.f32, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w = try ctx.fromSlice(.f32, .{ 3, 1, 1 }, &.{ 1, 2, 3 });
    defer w.deinit();
    var gy = try ctx.fromSlice(.f32, .{ 4, 1 }, &.{ 1, 1, 1, 1 });
    defer gy.deinit();

    var gx = try ctx.conv1dBackwardInput(2, &gy, &w, 0, 1, 4, 1, 1, 1, 1);
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 6, 6, 5 }, gx.dataConst());

    var gw = try ctx.conv1dBackwardWeight(2, &x, &gy, 0, 1, 3, 1, 1, 1, 1);
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 10, 9 }, gw.dataConst());

    // Adjoint of the k=2, s=2, d=2 case (y[t] = 10*x[2t] + x[2t+2], out_len 2).
    var xs = try ctx.fromSlice(.f32, .{ 5, 1 }, &.{ 1, 2, 3, 4, 5 });
    defer xs.deinit();
    var ws = try ctx.fromSlice(.f32, .{ 2, 1, 1 }, &.{ 10, 1 });
    defer ws.deinit();
    var gys = try ctx.fromSlice(.f32, .{ 2, 1 }, &.{ 1, 1 });
    defer gys.deinit();
    var gxs = try ctx.conv1dBackwardInput(2, &gys, &ws, 0, 1, 5, 2, 0, 2, 1);
    defer gxs.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 0, 11, 0, 1 }, gxs.dataConst());
    var gws = try ctx.conv1dBackwardWeight(2, &xs, &gys, 0, 1, 2, 2, 0, 2, 1);
    defer gws.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 4, 8 }, gws.dataConst());

    // seq inconsistent with gy's out_len is rejected (both directions).
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv1dBackwardInput(2, &gy, &w, 0, 1, 5, 1, 1, 1, 1));
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv1dBackwardWeight(2, &xs, &gy, 0, 1, 3, 1, 1, 1, 1));
    // Degenerate geometry is rejected.
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.conv1dBackwardInput(2, &gy, &w, 0, 1, 4, 0, 1, 1, 1));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.conv1dBackwardWeight(2, &x, &gy, 0, 1, 3, 1, 1, 0, 1));
    // groups must divide the channel counts.
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv1dBackwardInput(2, &gy, &w, 0, 1, 4, 1, 1, 1, 3));
}

test "col2im1d backward: hand-computed gather transpose + rejects short gy" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Adjoint of the s=2, k=2, pad=0 forward case: gcol[ti, k] = gy[2*ti + k].
    var gy = try ctx.fromSlice(.f32, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer gy.deinit();
    var gcol = try ctx.col2im1dBackward(&gy, 2, 1, 2, 2, 0);
    defer gcol.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, gcol.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, gcol.dataConst());

    // pad=1, output_pad=1 (t_conv = 2, gy has 3 rows): cropped taps and the
    // output_pad row read back as zero.
    var gy_pad = try ctx.fromSlice(.f32, .{ 3, 1 }, &.{ 5, 6, 7 });
    defer gy_pad.deinit();
    var gcol_pad = try ctx.col2im1dBackward(&gy_pad, 2, 1, 2, 2, 1);
    defer gcol_pad.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 5, 6, 0 }, gcol_pad.dataConst());

    // gy channel count must match; gy must cover t_conv rows; stride > 0.
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.col2im1dBackward(&gy, 2, 2, 2, 2, 0));
    var short_gy = try ctx.fromSlice(.f32, .{ 3, 1 }, &.{ 1, 2, 3 });
    defer short_gy.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.col2im1dBackward(&short_gy, 2, 1, 2, 2, 0));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.col2im1dBackward(&gy, 2, 1, 2, 0, 0));
}
