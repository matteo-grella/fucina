//! Tests for dac.zig: a tiny synthetic 5-block decoder (real strides
//! 8·5·4·2·3, tiny channels) checked stage-by-stage against a naive host
//! reference (snake / padded-dilated conv1d / PyTorch ConvTranspose1d
//! scatter / res-unit skip), including the per-stage taps and the final
//! waveform length algebra (T → 960·T).

const std = @import("std");
const fucina = @import("fucina");

const codec = @import("codec.zig");
const dac = @import("dac.zig");

/// Deterministic pseudo-random fill in [-0.5, 0.5].
fn fill(values: []f32, seed: u64) void {
    var state = seed *% 0x9E3779B97F4A7C15 +% 1;
    for (values) |*value| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const bits: u32 = @truncate(state >> 33);
        value.* = (@as(f32, @floatFromInt(bits % 2000)) - 1000.0) / 2000.0;
    }
}

/// Deterministic alpha fill in [0.5, 1.0] (keeps 1/(alpha+eps) tame).
fn fillAlpha(values: []f32, seed: u64) void {
    fill(values, seed);
    for (values) |*value| value.* = 0.75 + value.* / 2.0;
}

// --- naive host reference ---------------------------------------------------

const Stage = struct {
    t: usize,
    c: usize,
    data: []f32, // [t, c] row-major
};

fn refSnake(s: *const Stage, alpha: []const f32, inv_b: []const f32) void {
    for (0..s.t) |ti| {
        for (0..s.c) |ci| {
            const x = s.data[ti * s.c + ci];
            const sn = @sin(alpha[ci] * x);
            s.data[ti * s.c + ci] = x + sn * sn * inv_b[ci];
        }
    }
}

/// Cross-correlation conv1d, stride 1, symmetric zero pad, dilation;
/// `w_pt` is the PyTorch/GGUF flat layout `w[(oc*IC + ic)*K + k]`.
fn refConv1d(
    arena: std.mem.Allocator,
    s: *const Stage,
    w_pt: []const f32,
    bias: []const f32,
    out_ch: usize,
    k: usize,
    pad: usize,
    dilation: usize,
) !Stage {
    const t_out = s.t + 2 * pad - dilation * (k - 1);
    const out = try arena.alloc(f32, t_out * out_ch);
    for (0..t_out) |ti| {
        for (0..out_ch) |oc| {
            var sum: f32 = bias[oc];
            for (0..s.c) |ic| {
                for (0..k) |kk| {
                    const pos = ti + kk * dilation;
                    if (pos < pad or pos - pad >= s.t) continue;
                    sum += w_pt[(oc * s.c + ic) * k + kk] * s.data[(pos - pad) * s.c + ic];
                }
            }
            out[ti * out_ch + oc] = sum;
        }
    }
    return .{ .t = t_out, .c = out_ch, .data = out };
}

/// PyTorch ConvTranspose1d as a scatter; `wt_pt` is the PT flat layout
/// `w[(ic*OC + oc)*K + k]`. Appends `output_pad` zero rows, then adds bias.
fn refConvT(
    arena: std.mem.Allocator,
    s: *const Stage,
    wt_pt: []const f32,
    bias: []const f32,
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
    output_pad: usize,
) !Stage {
    const t_conv = (s.t - 1) * stride + k - 2 * pad;
    const t_out = t_conv + output_pad;
    const out = try arena.alloc(f32, t_out * out_ch);
    @memset(out, 0.0);
    for (0..s.t) |ti| {
        for (0..k) |kk| {
            const upsampled = ti * stride + kk;
            if (upsampled < pad or upsampled - pad >= t_conv) continue;
            const pos = upsampled - pad;
            for (0..out_ch) |oc| {
                var sum: f32 = 0.0;
                for (0..s.c) |ic| {
                    sum += s.data[ti * s.c + ic] * wt_pt[(ic * out_ch + oc) * k + kk];
                }
                out[pos * out_ch + oc] += sum;
            }
        }
    }
    for (0..t_out) |ti| {
        for (0..out_ch) |oc| out[ti * out_ch + oc] += bias[oc];
    }
    return .{ .t = t_out, .c = out_ch, .data = out };
}

fn expectStageEqual(want: *const Stage, got_t: usize, got_c: usize, got: []const f32) !void {
    try std.testing.expectEqual(want.t, got_t);
    try std.testing.expectEqual(want.c, got_c);
    try std.testing.expectEqual(want.data.len, got.len);
    for (want.data, got) |w, g| {
        const tol = 1e-3 + 1e-3 * @abs(w);
        if (@abs(w - g) > tol) {
            std.debug.print("stage mismatch: want {d} got {d}\n", .{ w, g });
            return error.TestExpectedApproxEq;
        }
    }
}

// --- synthetic decoder builder ----------------------------------------------

/// PT-layout weight copies for the naive reference side.
const SynResUnit = struct {
    s1_a: []f32,
    s1_inv: []f32,
    conv1_w: []f32,
    conv1_b: []f32,
    s2_a: []f32,
    s2_inv: []f32,
    conv2_w: []f32,
    conv2_b: []f32,
};

const SynBlock = struct {
    s1_a: []f32,
    s1_inv: []f32,
    convt_w: []f32,
    convt_b: []f32,
    res: [3]SynResUnit,
};

const SynDac = struct {
    conv1_w: []f32,
    conv1_b: []f32,
    blocks: [5]SynBlock,
    final_a: []f32,
    final_inv: []f32,
    conv2_w: []f32,
    conv2_b: []f32,
};

fn makeAlphaPair(arena: std.mem.Allocator, c: usize, seed: u64) !struct { a: []f32, inv: []f32 } {
    const a = try arena.alloc(f32, c);
    fillAlpha(a, seed);
    const inv = try arena.alloc(f32, c);
    for (inv, a) |*dst, av| dst.* = codec.snakeInvB(av);
    return .{ .a = a, .inv = inv };
}

fn makeChannelVec(ctx: *fucina.ExecContext, values: []const f32) !codec.ChannelVec {
    return codec.ChannelVec.fromSlice(ctx, .{values.len}, values);
}

fn makeConvWeight(ctx: *fucina.ExecContext, arena: std.mem.Allocator, w_pt: []const f32, k: usize, in_ch: usize, out_ch: usize) !codec.ConvWeight {
    const repacked = try arena.alloc(f32, w_pt.len);
    codec.repackConv1dWeight(repacked, w_pt, k, in_ch, out_ch);
    return codec.ConvWeight.fromSlice(ctx, .{ k, in_ch, out_ch }, repacked);
}

fn makeGemmWeight(ctx: *fucina.ExecContext, arena: std.mem.Allocator, w_pt: []const f32, k: usize, in_ch: usize, out_ch: usize) !codec.GemmConvWeight {
    const repacked = try arena.alloc(f32, w_pt.len);
    codec.repackConv1dGemmWeight(repacked, w_pt, k, in_ch, out_ch);
    return codec.GemmConvWeight.fromSlice(ctx, .{ out_ch, k * in_ch }, repacked);
}

fn makeResUnit(
    ctx: *fucina.ExecContext,
    arena: std.mem.Allocator,
    test_alloc: std.mem.Allocator,
    c: usize,
    dilation: usize,
    seed: u64,
) !struct { syn: SynResUnit, ru: codec.ResUnit } {
    const s1 = try makeAlphaPair(arena, c, seed);
    const s2 = try makeAlphaPair(arena, c, seed + 1);
    const conv1_w = try arena.alloc(f32, c * c * 7);
    fill(conv1_w, seed + 2);
    const conv1_b = try arena.alloc(f32, c);
    fill(conv1_b, seed + 3);
    const conv2_w = try arena.alloc(f32, c * c * 1);
    fill(conv2_w, seed + 4);
    const conv2_b = try arena.alloc(f32, c);
    fill(conv2_b, seed + 5);

    return .{
        .syn = .{
            .s1_a = s1.a,
            .s1_inv = s1.inv,
            .conv1_w = conv1_w,
            .conv1_b = conv1_b,
            .s2_a = s2.a,
            .s2_inv = s2.inv,
            .conv2_w = conv2_w,
            .conv2_b = conv2_b,
        },
        .ru = .{
            .snake1_a = try makeChannelVec(ctx, s1.a),
            .snake1_inv_b = try makeChannelVec(ctx, s1.inv),
            .conv1_w = try makeConvWeight(ctx, arena, conv1_w, 7, c, c),
            .conv1_gw = try makeGemmWeight(ctx, arena, conv1_w, 7, c, c),
            .conv1_b = try test_alloc.dupe(f32, conv1_b),
            .snake2_a = try makeChannelVec(ctx, s2.a),
            .snake2_inv_b = try makeChannelVec(ctx, s2.inv),
            .conv2_w = try makeConvWeight(ctx, arena, conv2_w, 1, c, c),
            .conv2_gw = try makeGemmWeight(ctx, arena, conv2_w, 1, c, c),
            .conv2_b = try test_alloc.dupe(f32, conv2_b),
            .dilation = dilation,
        },
    };
}

test "dac decodeForward matches naive reference stage by stage" {
    const test_alloc = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(test_alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var ctx: fucina.ExecContext = undefined;
    ctx.init(test_alloc);
    defer ctx.deinit();

    // Tiny channels, REAL strides: total upsample 8·5·4·2·3 = 960.
    const in_dim = 2; // fc2 output channels
    const chans = [6]usize{ 3, 3, 2, 2, 2, 2 }; // conv1 out + per-block out
    const strides = [5]usize{ 8, 5, 4, 2, 3 };

    var syn: SynDac = undefined;
    var dec: codec.DacDecoder = undefined;

    // conv1: in_dim → chans[0], k=7 pad 3.
    syn.conv1_w = try arena.alloc(f32, chans[0] * in_dim * 7);
    fill(syn.conv1_w, 1000);
    syn.conv1_b = try arena.alloc(f32, chans[0]);
    fill(syn.conv1_b, 1001);
    dec.conv1_w = try makeConvWeight(&ctx, arena, syn.conv1_w, 7, in_dim, chans[0]);
    dec.conv1_gw = try makeGemmWeight(&ctx, arena, syn.conv1_w, 7, in_dim, chans[0]);
    dec.conv1_b = try test_alloc.dupe(f32, syn.conv1_b);

    for (0..5) |bi| {
        const ic = chans[bi];
        const oc = chans[bi + 1];
        const spec = codec.BlockSpec.init(ic, oc, strides[bi]);
        const seed: u64 = 2000 + 100 * bi;

        const s1 = try makeAlphaPair(arena, ic, seed);
        const convt_w = try arena.alloc(f32, ic * oc * spec.taps);
        fill(convt_w, seed + 2);
        const convt_b = try arena.alloc(f32, oc);
        fill(convt_b, seed + 3);
        syn.blocks[bi] = .{
            .s1_a = s1.a,
            .s1_inv = s1.inv,
            .convt_w = convt_w,
            .convt_b = convt_b,
            .res = undefined,
        };

        const repacked = try arena.alloc(f32, convt_w.len);
        codec.repackConvT1dWeight(repacked, convt_w, spec.taps, oc, ic);
        dec.blocks[bi] = .{
            .spec = spec,
            .snake1_a = try makeChannelVec(&ctx, s1.a),
            .snake1_inv_b = try makeChannelVec(&ctx, s1.inv),
            .conv_t_w2 = try codec.ConvTWeight.fromSlice(&ctx, .{ spec.taps * oc, ic }, repacked),
            .conv_t_b = try codec.OutVec.fromSlice(&ctx, .{oc}, convt_b),
            .res = undefined,
        };
        for (0..3) |ri| {
            const made = try makeResUnit(&ctx, arena, test_alloc, oc, codec.res_unit_dilations[ri], seed + 10 * (ri + 1));
            syn.blocks[bi].res[ri] = made.syn;
            dec.blocks[bi].res[ri] = made.ru;
        }
    }

    const final = try makeAlphaPair(arena, chans[5], 9000);
    syn.final_a = final.a;
    syn.final_inv = final.inv;
    syn.conv2_w = try arena.alloc(f32, 1 * chans[5] * 7);
    fill(syn.conv2_w, 9001);
    syn.conv2_b = try arena.alloc(f32, 1);
    fill(syn.conv2_b, 9002);
    dec.final_snake_a = try makeChannelVec(&ctx, final.a);
    dec.final_snake_inv_b = try makeChannelVec(&ctx, final.inv);
    dec.conv2_w = try makeConvWeight(&ctx, arena, syn.conv2_w, 7, chans[5], 1);
    dec.conv2_gw = try makeGemmWeight(&ctx, arena, syn.conv2_w, 7, chans[5], 1);
    dec.conv2_b = try test_alloc.dupe(f32, syn.conv2_b);
    defer dec.deinit(test_alloc);

    // Input [T=2, in_dim].
    const t0 = 2;
    const input = try arena.alloc(f32, t0 * in_dim);
    fill(input, 42);

    // --- naive reference forward, capturing the tap stages ---
    var stage = Stage{ .t = t0, .c = in_dim, .data = try arena.dupe(f32, input) };
    stage = try refConv1d(arena, &stage, syn.conv1_w, syn.conv1_b, chans[0], 7, 3, 1);
    const ref_after_conv1 = Stage{ .t = stage.t, .c = stage.c, .data = try arena.dupe(f32, stage.data) };

    var ref_after_blk: [5]Stage = undefined;
    for (0..5) |bi| {
        const sb = &syn.blocks[bi];
        const spec = dec.blocks[bi].spec;
        refSnake(&stage, sb.s1_a, sb.s1_inv);
        stage = try refConvT(arena, &stage, sb.convt_w, sb.convt_b, spec.out_ch, spec.taps, spec.stride, spec.pad, spec.output_pad);
        for (&sb.res, codec.res_unit_dilations) |*ru, dil| {
            const skip = try arena.dupe(f32, stage.data);
            refSnake(&stage, ru.s1_a, ru.s1_inv);
            stage = try refConv1d(arena, &stage, ru.conv1_w, ru.conv1_b, spec.out_ch, 7, 3 * dil, dil);
            refSnake(&stage, ru.s2_a, ru.s2_inv);
            stage = try refConv1d(arena, &stage, ru.conv2_w, ru.conv2_b, spec.out_ch, 1, 0, 1);
            for (stage.data, skip) |*x, s| x.* += s;
        }
        ref_after_blk[bi] = .{ .t = stage.t, .c = stage.c, .data = try arena.dupe(f32, stage.data) };
    }
    refSnake(&stage, syn.final_a, syn.final_inv);
    stage = try refConv1d(arena, &stage, syn.conv2_w, syn.conv2_b, 1, 7, 3, 1);
    try std.testing.expectEqual(@as(usize, t0 * 960), stage.t);

    // --- decodeForward under test ---
    var fc2_out = try dac.Act.fromSlice(&ctx, .{ t0, in_dim }, input);
    defer fc2_out.deinit();
    var taps = dac.Taps.init(test_alloc);
    defer taps.deinit();
    const audio = try dac.decodeForward(&ctx, test_alloc, &dec, &fc2_out, &taps);
    defer test_alloc.free(audio);

    try std.testing.expectEqual(@as(usize, t0 * 960), audio.len);
    const conv1_tap = taps.after_conv1.?;
    try expectStageEqual(&ref_after_conv1, conv1_tap.t, conv1_tap.c, conv1_tap.data);
    for (0..5) |bi| {
        const tap = taps.after_blk[bi].?;
        try expectStageEqual(&ref_after_blk[bi], tap.t, tap.c, tap.data);
    }
    try expectStageEqual(&stage, audio.len, 1, audio);

    // The direct-kernel escape hatch must agree with the default im2col+GEMM
    // path (same math; f32 GEMM reassociation noise only).
    dac.use_direct_conv = true;
    defer dac.use_direct_conv = false;
    const audio_direct = try dac.decodeForward(&ctx, test_alloc, &dec, &fc2_out, null);
    defer test_alloc.free(audio_direct);
    try std.testing.expectEqual(audio.len, audio_direct.len);
    for (audio, audio_direct) |g, d| {
        const tol = 1e-4 + 1e-4 * @abs(g);
        if (@abs(g - d) > tol) {
            std.debug.print("gemm/direct mismatch: gemm {d} direct {d}\n", .{ g, d });
            return error.TestExpectedApproxEq;
        }
    }
}

test "encodeOutputLength matches the reference block formulas" {
    // Hop-multiple inputs land exactly at n/960.
    try std.testing.expectEqual(@as(isize, 1), dac.encodeOutputLength(960));
    try std.testing.expectEqual(@as(isize, 2), dac.encodeOutputLength(1920));
    try std.testing.expectEqual(@as(isize, 110), dac.encodeOutputLength(105600));
    // Chain the 5 block formulas by hand for a non-multiple.
    var t: isize = 1000;
    for ([_][3]isize{ .{ 16, 8, 4 }, .{ 10, 5, 3 }, .{ 8, 4, 2 }, .{ 4, 2, 1 }, .{ 6, 3, 2 } }) |ksp| {
        t = @divTrunc(t + 2 * ksp[2] - ksp[0], ksp[1]) + 1;
    }
    try std.testing.expectEqual(t, dac.encodeOutputLength(1000));
}
