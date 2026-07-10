//! Tests for codec.zig: repack functions on synthetic buffers, snake inv_b
//! arithmetic, the DAC block geometry table, and an env-guarded real-file
//! load (set OMNIVOICE_TOKENIZER_GGUF to run it).

const std = @import("std");
const fucina = @import("fucina");

const codec = @import("codec.zig");

test "repackConv1dWeight maps ggml ne=(K,IC,OC) to [tap,in,out]" {
    const taps = 3;
    const in_ch = 2;
    const out_ch = 4;
    var src: [taps * in_ch * out_ch]f32 = undefined;
    // GGUF buffer: src[((oc*IC)+ic)*K + k] = f(oc, ic, k).
    for (0..out_ch) |oc| {
        for (0..in_ch) |ic| {
            for (0..taps) |k| {
                src[(oc * in_ch + ic) * taps + k] = value(oc, ic, k);
            }
        }
    }
    var dst: [src.len]f32 = undefined;
    codec.repackConv1dWeight(&dst, &src, taps, in_ch, out_ch);
    // Our layout: dst[(k*IC + ic)*OC + oc].
    for (0..taps) |k| {
        for (0..in_ch) |ic| {
            for (0..out_ch) |oc| {
                try std.testing.expectEqual(value(oc, ic, k), dst[(k * in_ch + ic) * out_ch + oc]);
            }
        }
    }
}

test "repackConv1dGemmWeight maps ggml ne=(K,IC,OC) to [OC, K*IC] ic-fastest" {
    const taps = 3;
    const in_ch = 2;
    const out_ch = 4;
    var src: [taps * in_ch * out_ch]f32 = undefined;
    // GGUF buffer: src[((oc*IC)+ic)*K + k] = f(oc, ic, k).
    for (0..out_ch) |oc| {
        for (0..in_ch) |ic| {
            for (0..taps) |k| {
                src[(oc * in_ch + ic) * taps + k] = value(oc, ic, k);
            }
        }
    }
    var dst: [src.len]f32 = undefined;
    codec.repackConv1dGemmWeight(&dst, &src, taps, in_ch, out_ch);
    // GEMM layout: dst[oc*(K*IC) + k*IC + ic] (im2col column index k*IC+ic).
    for (0..out_ch) |oc| {
        for (0..taps) |k| {
            for (0..in_ch) |ic| {
                try std.testing.expectEqual(value(oc, ic, k), dst[oc * (taps * in_ch) + k * in_ch + ic]);
            }
        }
    }
}

test "repackConvT1dWeight maps ggml ne=(K,OC,IC) to [K*OC, IC] k-fastest" {
    const taps = 4;
    const out_ch = 3;
    const in_ch = 2;
    var src: [taps * out_ch * in_ch]f32 = undefined;
    // GGUF buffer: src[((ic*OC)+oc)*K + k] = f(ic, oc, k).
    for (0..in_ch) |ic| {
        for (0..out_ch) |oc| {
            for (0..taps) |k| {
                src[(ic * out_ch + oc) * taps + k] = value(ic, oc, k);
            }
        }
    }
    var dst: [src.len]f32 = undefined;
    codec.repackConvT1dWeight(&dst, &src, taps, out_ch, in_ch);
    // weight2 layout: dst[(oc*K + k)*IC + ic], k fastest inside each oc block.
    for (0..out_ch) |oc| {
        for (0..taps) |k| {
            for (0..in_ch) |ic| {
                try std.testing.expectEqual(value(ic, oc, k), dst[(oc * taps + k) * in_ch + ic]);
            }
        }
    }
}

fn value(a: usize, b: usize, c: usize) f32 {
    return @floatFromInt(a * 100 + b * 10 + c);
}

test "snakeInvB is 1/(alpha + 1e-9) in f32" {
    const alphas = [_]f32{ 0.5, 1.0, 3.25, 1e-3, 42.0 };
    for (alphas) |a| {
        const expected: f32 = 1.0 / (a + comptime @as(f32, 1.0e-9));
        try std.testing.expectEqual(expected, codec.snakeInvB(a));
    }
}

test "dac block specs match the reference table and the length algebra" {
    const expected = [5]struct { in_ch: usize, out_ch: usize, s: usize, k: usize, p: usize, op: usize }{
        .{ .in_ch = 1024, .out_ch = 512, .s = 8, .k = 16, .p = 4, .op = 0 },
        .{ .in_ch = 512, .out_ch = 256, .s = 5, .k = 10, .p = 3, .op = 1 },
        .{ .in_ch = 256, .out_ch = 128, .s = 4, .k = 8, .p = 2, .op = 0 },
        .{ .in_ch = 128, .out_ch = 64, .s = 2, .k = 4, .p = 1, .op = 0 },
        .{ .in_ch = 64, .out_ch = 32, .s = 3, .k = 6, .p = 2, .op = 1 },
    };
    var total_upsample: usize = 1;
    for (codec.dac_block_specs, expected) |spec, e| {
        try std.testing.expectEqual(e.in_ch, spec.in_ch);
        try std.testing.expectEqual(e.out_ch, spec.out_ch);
        try std.testing.expectEqual(e.s, spec.stride);
        try std.testing.expectEqual(e.k, spec.taps);
        try std.testing.expectEqual(e.p, spec.pad);
        try std.testing.expectEqual(e.op, spec.output_pad);
        // ConvTranspose1d length: (T-1)*S + K - 2*pad + output_pad == S*T.
        for ([_]usize{ 1, 2, 7, 110 }) |t| {
            const t_out = (t - 1) * spec.stride + spec.taps - 2 * spec.pad + spec.output_pad;
            try std.testing.expectEqual(spec.stride * t, t_out);
        }
        total_upsample *= spec.stride;
    }
    try std.testing.expectEqual(@as(usize, 960), total_upsample);
    try std.testing.expectEqual(@as(usize, 3), codec.res_unit_dilations.len);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 9 }, &codec.res_unit_dilations);
}

test "real tokenizer GGUF loads (set OMNIVOICE_TOKENIZER_GGUF to enable)" {
    const path_z = std.c.getenv("OMNIVOICE_TOKENIZER_GGUF") orelse return error.SkipZigTest;
    const path = std.mem.span(path_z);

    const allocator = std.testing.allocator;
    var file = try fucina.gguf.File.loadMmap(allocator, std.testing.io, path);
    defer file.deinit();

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var cdc = try codec.Codec.load(&ctx, &file);
    defer cdc.deinit();

    try std.testing.expectEqual(@as(usize, 24000), cdc.config.sample_rate);
    try std.testing.expectEqual(@as(usize, 960), cdc.config.hop_length);
    for (&cdc.rvq.quantizers) |*q| {
        try std.testing.expectEqual(@as(usize, 1024), q.embed.dim(.code));
        try std.testing.expectEqual(@as(usize, 64), q.embed.dim(.cdim));
        try std.testing.expectEqual(@as(usize, 1024), q.project_out.outDim());
        try std.testing.expectEqual(@as(usize, 64), q.project_out.inDim());
        try std.testing.expectEqual(@as(usize, 1024), q.embed_sq.len);
    }
    try std.testing.expectEqual(@as(usize, 256), cdc.rvq.fc2.outDim());
    try std.testing.expectEqual(@as(usize, 1024), cdc.rvq.fc2.inDim());
    try std.testing.expectEqual(@as(usize, 7 * 256 * 1024), (try cdc.dac.conv1_w.dataConst()).len);
    for (&cdc.dac.blocks) |*blk| {
        try std.testing.expectEqual(blk.spec.taps * blk.spec.out_ch, blk.conv_t_w2.dim(.kout));
        try std.testing.expectEqual(blk.spec.in_ch, blk.conv_t_w2.dim(.in));
    }
    try std.testing.expectEqual(@as(usize, 32), cdc.dac.final_snake_a.dim(.in));

    // Encode side.
    var enc = try codec.Encoder.load(&ctx, &file);
    defer enc.deinit();

    try std.testing.expectEqual(@as(usize, 10), enc.hubert.feat[0].conv_w.taps);
    try std.testing.expectEqual(@as(usize, 1), enc.hubert.feat[0].conv_w.in_per_group);
    try std.testing.expectEqual(@as(usize, 512), enc.hubert.feat[0].conv_w.out_ch);
    try std.testing.expect(enc.hubert.feat[0].gn_w != null);
    try std.testing.expect(enc.hubert.feat[1].gn_w == null);
    try std.testing.expectEqual(@as(usize, 128), enc.hubert.pos_conv_w.taps);
    try std.testing.expectEqual(@as(usize, 48), enc.hubert.pos_conv_w.in_per_group);
    try std.testing.expectEqual(@as(usize, 768), enc.hubert.pos_conv_w.out_ch);
    try std.testing.expectEqual(@as(usize, 16), enc.hubert.pos_conv_w.groups);
    try std.testing.expectEqual(@as(usize, 768), enc.hubert.fp_proj.outDim());
    try std.testing.expectEqual(@as(usize, 512), enc.hubert.fp_proj.inDim());
    for (&enc.hubert.layers) |*layer| {
        try std.testing.expectEqual(@as(usize, 768), layer.q_proj.outDim());
        try std.testing.expectEqual(@as(usize, 3072), layer.fc1.outDim());
        try std.testing.expectEqual(@as(usize, 3072), layer.fc2.inDim());
        try std.testing.expectEqual(@as(usize, 768), layer.ln_attn_w.len);
        try std.testing.expectEqual(@as(usize, 768), layer.ln_final_b.len);
    }
    try std.testing.expectEqual(@as(usize, 3), enc.semantic.conv_w.taps);
    try std.testing.expectEqual(@as(usize, 768), enc.semantic.conv_w.out_ch);
    try std.testing.expectEqual(@as(usize, 7), enc.dac.conv1_w.taps);
    try std.testing.expectEqual(@as(usize, 64), enc.dac.conv1_w.out_ch);
    for (&enc.dac.blocks, codec.dac_enc_block_specs) |*blk, spec| {
        try std.testing.expectEqual(spec.taps, blk.conv_w.taps);
        try std.testing.expectEqual(spec.in_ch, blk.conv_w.in_per_group);
        try std.testing.expectEqual(spec.out_ch, blk.conv_w.out_ch);
        try std.testing.expectEqual(spec.in_ch, blk.snake_a.len);
    }
    try std.testing.expectEqual(@as(usize, 2048), enc.dac.final_snake_a.len);
    try std.testing.expectEqual(@as(usize, 256), enc.dac.conv2_w.out_ch);
    for (&enc.project_in) |*p| {
        try std.testing.expectEqual(@as(usize, 64), p.weight.outDim());
        try std.testing.expectEqual(@as(usize, 1024), p.weight.inDim());
        try std.testing.expectEqual(@as(usize, 64), p.bias.len);
    }
    try std.testing.expectEqual(@as(usize, 1024), enc.fc.outDim());
    try std.testing.expectEqual(@as(usize, 1024), enc.fc.inDim());
}

// --- ggml-parity host ops ----------------------------------------------------

test "dac encoder block specs match the reference table" {
    const expected = [5]struct { in_ch: usize, out_ch: usize, k: usize, s: usize, p: usize }{
        .{ .in_ch = 64, .out_ch = 128, .k = 16, .s = 8, .p = 4 },
        .{ .in_ch = 128, .out_ch = 256, .k = 10, .s = 5, .p = 3 },
        .{ .in_ch = 256, .out_ch = 512, .k = 8, .s = 4, .p = 2 },
        .{ .in_ch = 512, .out_ch = 1024, .k = 4, .s = 2, .p = 1 },
        .{ .in_ch = 1024, .out_ch = 2048, .k = 6, .s = 3, .p = 2 },
    };
    var total_downsample: usize = 1;
    for (codec.dac_enc_block_specs, expected) |spec, e| {
        try std.testing.expectEqual(e.in_ch, spec.in_ch);
        try std.testing.expectEqual(e.out_ch, spec.out_ch);
        try std.testing.expectEqual(e.k, spec.taps);
        try std.testing.expectEqual(e.s, spec.stride);
        try std.testing.expectEqual(e.p, spec.pad);
        total_downsample *= spec.stride;
    }
    try std.testing.expectEqual(@as(usize, 960), total_downsample);
}

test "vecDotF16Ggml matches a scalar reference within f16 accumulation noise" {
    var x: [100]f16 = undefined;
    var y: [100]f16 = undefined;
    var state: u64 = 7;
    for (0..100) |i| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        x[i] = @floatCast(@as(f32, @floatFromInt(@as(i64, @intCast(state >> 33 & 0xFFF)) - 2048)) / 2048.0);
        state = state *% 6364136223846793005 +% 1442695040888963407;
        y[i] = @floatCast(@as(f32, @floatFromInt(@as(i64, @intCast(state >> 33 & 0xFFF)) - 2048)) / 2048.0);
    }
    for ([_]usize{ 1, 7, 10, 31, 32, 33, 64, 100 }) |n| {
        var want: f64 = 0.0;
        for (x[0..n], y[0..n]) |xv, yv| want += @as(f64, xv) * @as(f64, yv);
        const got = codec.vecDotF16Ggml(x[0..n], y[0..n]);
        // f16 lane accumulation noise stays ~1e-3 relative at these sizes.
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got, 2e-2);
    }
    // n < 32 runs the exact scalar f64-of-f32-products path.
    var want_small: f64 = 0.0;
    for (x[0..10], y[0..10]) |xv, yv| want_small += @as(f64, @as(f32, xv) * @as(f32, yv));
    try std.testing.expectEqual(@as(f32, @floatCast(want_small)), codec.vecDotF16Ggml(x[0..10], y[0..10]));
}

test "ggmlConv1d matches a naive conv on f16-exact inputs" {
    const allocator = std.testing.allocator;
    const t_in = 9;
    const in_ch = 2;
    const out_ch = 3;
    const k = 3;

    var w_data: [out_ch * in_ch * k]f16 = undefined;
    for (&w_data, 0..) |*dst, i| dst.* = @floatFromInt(@as(i64, @intCast(i % 5)) - 2);
    var w = codec.GgmlConvWeight{
        .data = try allocator.dupe(f16, &w_data),
        .taps = k,
        .in_per_group = in_ch,
        .out_ch = out_ch,
        .groups = 1,
    };
    defer w.deinit(allocator);

    var x: [t_in * in_ch]f32 = undefined;
    for (&x, 0..) |*dst, i| dst.* = @floatFromInt(@as(i64, @intCast(i % 7)) - 3);
    const bias = [_]f32{ 0.5, -1.0, 2.0 };

    // stride 2, pad 1, dilation 1: t_out = (9 + 2 - 3)/2 + 1 = 5.
    const out = try codec.ggmlConv1d(allocator, &x, t_in, in_ch, &w, &bias, 2, 1, 1);
    defer allocator.free(out.data);
    try std.testing.expectEqual(@as(usize, 5), out.t);
    try std.testing.expectEqual(@as(usize, out_ch), out.c);

    for (0..out.t) |ot| {
        for (0..out_ch) |oc| {
            var want: f32 = bias[oc];
            for (0..in_ch) |ic| {
                for (0..k) |kk| {
                    const pos = ot * 2 + kk;
                    if (pos < 1 or pos - 1 >= t_in) continue;
                    want += @as(f32, @floatCast(w_data[(oc * in_ch + ic) * k + kk])) * x[(pos - 1) * in_ch + ic];
                }
            }
            // Small integers: exact in f16, so the dot is exact.
            try std.testing.expectApproxEqAbs(want, out.data[ot * out_ch + oc], 1e-5);
        }
    }
}

test "snakeGgml computes x + fma(sin^2(a*x), inv_b, x)" {
    var values = [_]f32{ 0.5, -1.25, 2.0, 0.0, 3.5, -0.75 };
    const alpha = [_]f32{ 0.8, 1.5 };
    const inv_b = [_]f32{ 2.0, 0.5 };
    var expected: [values.len]f32 = undefined;
    for (&expected, values, 0..) |*e, x, i| {
        const s = @sin(alpha[i % 2] * x);
        e.* = @mulAdd(f32, s * s, inv_b[i % 2], x);
    }
    codec.snakeGgml(&values, 3, 2, &alpha, &inv_b);
    for (values, expected) |got, want| {
        try std.testing.expectApproxEqAbs(want, got, 1e-6);
    }
}

test "geluErfGgml and eluGgml match the reference formulas" {
    var g = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    codec.geluErfGgml(&g);
    // gelu_erf(1) = 0.8413447; gelu_erf(-1) = -0.15865526 (exact-erf GELU).
    try std.testing.expectApproxEqAbs(@as(f32, 0.8413447), g[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.15865526), g[1], 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), g[2]);

    var e = [_]f32{ -1.0, -0.5, 0.0, 0.5, 2.0 };
    codec.eluGgml(&e);
    try std.testing.expectApproxEqAbs(@as(f32, -0.63212055), e[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.39346934), e[1], 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), e[2]);
    try std.testing.expectEqual(@as(f32, 0.5), e[3]);
    try std.testing.expectEqual(@as(f32, 2.0), e[4]);
}

test "groupNormGgml normalizes per channel group with affine" {
    // G == C: per-channel instance norm over time (the HuBERT layer-0 case).
    const t = 4;
    const c = 2;
    var values = [_]f32{ 1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0 };
    const weight = [_]f32{ 2.0, 1.0 };
    const bias = [_]f32{ 0.5, -1.0 };
    codec.groupNormGgml(&values, t, c, c, 1e-5, &weight, &bias);

    // Channel 0: x = 1..4, mean 2.5, biased var 1.25.
    const std0 = @sqrt(1.25 + 1e-5);
    for (0..t) |ti| {
        const x: f32 = @floatFromInt(ti + 1);
        const want = (x - 2.5) / std0 * 2.0 + 0.5;
        try std.testing.expectApproxEqAbs(want, values[ti * c], 1e-4);
    }
    // Channel 1: x = 10..40, mean 25, biased var 125.
    const std1 = @sqrt(125.0 + 1e-5);
    for (0..t) |ti| {
        const x: f32 = @floatFromInt((ti + 1) * 10);
        const want = (x - 25.0) / std1 * 1.0 - 1.0;
        try std.testing.expectApproxEqAbs(want, values[ti * c + 1], 1e-4);
    }
}

test "layerNormGgml matches the torch LayerNorm formula" {
    const t = 2;
    const c = 4;
    var values = [_]f32{ 1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0 };
    const weight = [_]f32{ 1.0, 2.0, 0.5, 1.0 };
    const bias = [_]f32{ 0.0, 0.1, -0.1, 0.0 };
    var expected: [t * c]f32 = undefined;
    for (0..t) |ti| {
        const row = values[ti * c ..][0..c];
        var mean: f64 = 0.0;
        for (row) |v| mean += v;
        mean /= c;
        var variance: f64 = 0.0;
        for (row) |v| variance += (v - mean) * (v - mean);
        variance /= c;
        for (0..c) |ch| {
            const norm = (row[ch] - mean) / @sqrt(variance + 1e-5);
            expected[ti * c + ch] = @floatCast(norm * weight[ch] + bias[ch]);
        }
    }
    codec.layerNormGgml(&values, t, c, 1e-5, &weight, &bias);
    for (values, expected) |got, want| {
        try std.testing.expectApproxEqAbs(want, got, 1e-4);
    }
}

test "softMaxRowGgml matches a reference softmax and sums to 1" {
    var row = [_]f32{ 1.0, 3.0, -2.0, 0.5, 3.0, -10.0, 7.25, 0.0 };
    var expected: [row.len]f32 = undefined;
    var sum: f64 = 0.0;
    for (row) |v| sum += @exp(@as(f64, v) - 7.25);
    for (&expected, row) |*e, v| e.* = @floatCast(@exp(@as(f64, v) - 7.25) / sum);
    codec.softMaxRowGgml(&row);
    var got_sum: f64 = 0.0;
    for (row, expected) |got, want| {
        got_sum += got;
        // ggml's v_expf is a ~1-ulp polynomial approximation of expf.
        try std.testing.expectApproxEqAbs(want, got, 1e-6);
    }
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), got_sum, 1e-5);
}
