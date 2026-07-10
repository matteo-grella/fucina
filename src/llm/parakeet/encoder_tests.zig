//! Tests for the FastConformer encoder: the sinusoidal rel-pos table, the
//! Transformer-XL rel-shift closed form (validated against the literal ggml
//! pad/reshape/slice sequence), and the full relpos attention vs a naive f64
//! reference on a synthetic GGUF.
const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;
const loader = @import("loader.zig");
const enc = @import("encoder.zig");
const ParakeetWeights = @import("weights.zig").ParakeetWeights;

const ExecContext = fucina.ExecContext;
const Writer = gguf.Writer;

test "relPosEncoding matches the NeMo sinusoid (T=2, d_model=4)" {
    const allocator = std.testing.allocator;
    const pe = try enc.relPosEncoding(allocator, 2, 4);
    defer allocator.free(pe);
    // P=3, half=2; div[0]=1, div[1]=exp(2*-(ln1e4/4))=1e4^(-1/2)=0.01.
    // positions: p0->+1, p1->0, p2->-1. row = [sin,cos (i=0), sin,cos (i=1)].
    const e = [_]f64{
        @sin(1.0),  @cos(1.0),  @sin(0.01),  @cos(0.01),
        0.0,        1.0,        0.0,         1.0,
        @sin(-1.0), @cos(-1.0), @sin(-0.01), @cos(-0.01),
    };
    for (e, 0..) |x, i| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(x)), pe[i], 1e-5);
}

// The closed form used by relposAttention (result[kj][qi] = bd[kj-qi+T-1][qi])
// must equal parakeet.cpp's literal pad/reshape/view rel-shift skew. Implement
// the literal ggml op sequence on flat arrays and compare (single head).
test "rel-shift closed form == literal ggml pad/reshape/slice skew" {
    inline for (.{ 2, 3, 5 }) |TC| {
        const PC = 2 * TC - 1;
        // bd[r*TC + qi], r in [0,PC) — arbitrary distinct values.
        var bd: [PC * TC]f64 = undefined;
        for (0..PC) |r| for (0..TC) |qi| {
            bd[r * TC + qi] = @floatFromInt(r * 100 + qi);
        };

        // ---- literal ggml sequence (ne0 fastest) ----
        // bd ne [PC, TC]; pad ne0 left 1 -> [2TC, TC]; flat[r' + 2TC*qi].
        var padded: [2 * TC * TC]f64 = undefined;
        for (0..2 * TC) |rp| for (0..TC) |qi| {
            padded[rp + 2 * TC * qi] = if (rp == 0) 0 else bd[(rp - 1) * TC + qi];
        };
        // reshape [TC, 2TC]; view drop b=0 + cont -> J[a + TC*b'] = padded[a + TC*(b'+1)].
        var jbuf: [TC * (2 * TC - 1)]f64 = undefined;
        for (0..TC) |a| for (0..2 * TC - 1) |bp| {
            jbuf[a + TC * bp] = padded[a + TC * (bp + 1)];
        };
        // reshape [2TC-1, TC]: K[c + (2TC-1)*qi]; take first TC of c.
        var literal: [TC * TC]f64 = undefined; // literal[kj*TC + qi]
        for (0..TC) |kj| for (0..TC) |qi| {
            literal[kj * TC + qi] = jbuf[kj + (2 * TC - 1) * qi];
        };

        // ---- closed form ----
        for (0..TC) |kj| for (0..TC) |qi| {
            const r = kj + TC - 1 - qi;
            try std.testing.expectEqual(bd[r * TC + qi], literal[kj * TC + qi]);
        };
    }
}

// --- full relpos attention vs naive f64 reference (synthetic GGUF, f32 weights) ---

const D = 8;
const H = 2;
const DK = D / H;
const T = 3;
const P = 2 * T - 1;

fn gen(i: usize, salt: usize) f32 {
    return @floatCast(0.1 * @sin(@as(f64, @floatFromInt(i * 7 + salt * 13 + 1)) * 0.9));
}

fn tinyConfig() loader.Config {
    return .{
        .arch = .hybrid_tdt_ctc,
        .d_model = D,
        .n_layers = 1,
        .n_heads = H,
        .head_dim = DK,
        .ff_dim = 16,
        .feat_in = 8,
        .conv_kernel = 9,
        .conv_norm = .batch_norm,
        .subsampling_factor = 8,
        .subsampling_stages = 3,
        .subsampling_conv_channels = 4,
        .pos_emb_max_len = 100,
        .xscaling = false,
        .sample_rate = 16000,
        .n_mels = 8,
        .n_fft = 512,
        .win_length = 400,
        .hop_length = 160,
        .preemph = 0.97,
        .mag_power = 2.0,
        .log_zero_guard = 1e-5,
        .normalize = .per_feature,
        .vocab_size = 10,
        .blank_id = 10,
        .pred_hidden = 6,
        .pred_rnn_layers = 1,
        .max_symbols = 10,
        .joint_hidden = 6,
        .joint_activation = .relu,
        .num_durations = 3,
        .durations = [_]i32{ 0, 1, 2 } ++ [_]i32{0} ** 13,
    };
}

// addTensor borrows the data until finish(); pass the live weight slices
// (little-endian host == GGUF) directly, never a stack-local buffer.
fn addW(w: *Writer, name: []const u8, data: []const f32, ne0: usize, ne1: usize) !void {
    try w.addTensor(name, .f32, &.{ ne0, ne1 }, std.mem.sliceAsBytes(data));
}
fn addV(w: *Writer, name: []const u8, data: []const f32) !void {
    try w.addTensor(name, .f32, &.{data.len}, std.mem.sliceAsBytes(data));
}
fn addT(w: *Writer, name: []const u8, data: []const f32, dims: []const usize) !void {
    try w.addTensor(name, .f32, dims, std.mem.sliceAsBytes(data));
}

const Weights = struct {
    wq: [D * D]f32,
    bq: [D]f32,
    wk: [D * D]f32,
    bk: [D]f32,
    wv: [D * D]f32,
    bv: [D]f32,
    wo: [D * D]f32,
    bo: [D]f32,
    wp: [D * D]f32,
    bu: [H * DK]f32,
    bvb: [H * DK]f32,
};

fn genWeights() Weights {
    var w: Weights = undefined;
    for (&w.wq, 0..) |*v, i| v.* = gen(i, 1);
    for (&w.bq, 0..) |*v, i| v.* = gen(i, 2);
    for (&w.wk, 0..) |*v, i| v.* = gen(i, 3);
    for (&w.bk, 0..) |*v, i| v.* = gen(i, 4);
    for (&w.wv, 0..) |*v, i| v.* = gen(i, 5);
    for (&w.bv, 0..) |*v, i| v.* = gen(i, 6);
    for (&w.wo, 0..) |*v, i| v.* = gen(i, 7);
    for (&w.bo, 0..) |*v, i| v.* = gen(i, 8);
    for (&w.wp, 0..) |*v, i| v.* = gen(i, 9);
    for (&w.bu, 0..) |*v, i| v.* = gen(i, 10);
    for (&w.bvb, 0..) |*v, i| v.* = gen(i, 11);
    return w;
}

fn proj(out: []f64, x: []const f32, w: []const f32, b: ?[]const f32, m: usize) void {
    // out[t,o] = sum_i x[t,i]*w[o*D+i] + b[o]
    for (0..m) |t| for (0..D) |o| {
        var acc: f64 = if (b) |bb| bb[o] else 0;
        for (0..D) |i| acc += @as(f64, x[t * D + i]) * @as(f64, w[o * D + i]);
        out[t * D + o] = acc;
    };
}

fn naiveAttn(out: []f64, x: []const f32, pos_emb: []const f32, w: *const Weights) void {
    var q: [T * D]f64 = undefined;
    var k: [T * D]f64 = undefined;
    var v: [T * D]f64 = undefined;
    var pp: [P * D]f64 = undefined;
    proj(&q, x, &w.wq, &w.bq, T);
    proj(&k, x, &w.wk, &w.bk, T);
    proj(&v, x, &w.wv, &w.bv, T);
    proj(&pp, pos_emb, &w.wp, null, P);
    const scale = 1.0 / @sqrt(@as(f64, DK));
    var ctxb: [T * D]f64 = undefined;
    var scores: [T]f64 = undefined;
    for (0..H) |h| {
        const hoff = h * DK;
        for (0..T) |qi| {
            var maxs: f64 = -std.math.inf(f64);
            for (0..T) |kj| {
                const r = kj + T - 1 - qi;
                var ac: f64 = 0;
                var bd: f64 = 0;
                for (0..DK) |d| {
                    const qv = q[qi * D + hoff + d];
                    ac += (qv + w.bu[hoff + d]) * k[kj * D + hoff + d];
                    bd += (qv + w.bvb[hoff + d]) * pp[r * D + hoff + d];
                }
                scores[kj] = scale * (ac + bd);
                if (scores[kj] > maxs) maxs = scores[kj];
            }
            var sum: f64 = 0;
            for (0..T) |kj| {
                scores[kj] = @exp(scores[kj] - maxs);
                sum += scores[kj];
            }
            for (0..DK) |d| {
                var acc: f64 = 0;
                for (0..T) |kj| acc += scores[kj] * v[kj * D + hoff + d];
                ctxb[qi * D + hoff + d] = acc / sum;
            }
        }
    }
    // out projection
    for (0..T) |t| for (0..D) |o| {
        var acc: f64 = w.bo[o];
        for (0..D) |i| acc += ctxb[t * D + i] * @as(f64, w.wo[o * D + i]);
        out[t * D + o] = acc;
    };
}

test "relpos attention matches a naive f64 reference (synthetic GGUF)" {
    const allocator = std.testing.allocator;
    const w = genWeights();

    var writer = Writer.init(allocator);
    defer writer.deinit();
    // weights ggml ne [in=D, out=D]; our data row-major [out,in] (wq[o*D+i]) ==
    // ggml memory (i + D*o). pos_bias ne [dk, H], data[h*dk+d].
    try addW(&writer, "encoder.layers.0.self_attn.linear_q.weight", &w.wq, D, D);
    try addV(&writer, "encoder.layers.0.self_attn.linear_q.bias", &w.bq);
    try addW(&writer, "encoder.layers.0.self_attn.linear_k.weight", &w.wk, D, D);
    try addV(&writer, "encoder.layers.0.self_attn.linear_k.bias", &w.bk);
    try addW(&writer, "encoder.layers.0.self_attn.linear_v.weight", &w.wv, D, D);
    try addV(&writer, "encoder.layers.0.self_attn.linear_v.bias", &w.bv);
    try addW(&writer, "encoder.layers.0.self_attn.linear_out.weight", &w.wo, D, D);
    try addV(&writer, "encoder.layers.0.self_attn.linear_out.bias", &w.bo);
    try addW(&writer, "encoder.layers.0.self_attn.linear_pos.weight", &w.wp, D, D);
    try addW(&writer, "encoder.layers.0.self_attn.pos_bias_u", &w.bu, DK, H);
    try addW(&writer, "encoder.layers.0.self_attn.pos_bias_v", &w.bvb, DK, H);

    var buf: [64 * 1024]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try writer.finish(&sink);
    var file = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer file.deinit();

    var x: [T * D]f32 = undefined;
    for (&x, 0..) |*vv, i| vv.* = gen(i, 20);
    var pe: [P * D]f32 = undefined;
    for (&pe, 0..) |*vv, i| vv.* = gen(i, 21);

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var xt = try fucina.Tensor(2).fromSlice(&ctx, .{ T, D }, &x);
    defer xt.deinit();
    var pet = try fucina.Tensor(2).fromSlice(&ctx, .{ P, D }, &pe);
    defer pet.deinit();

    var pw = ParakeetWeights.init(&ctx, &file);
    defer pw.deinit();
    var got = try enc.relposAttention(&ctx, &pw, tinyConfig(), 0, &xt, &pet, null, null);
    defer got.deinit();

    var want: [T * D]f64 = undefined;
    naiveAttn(&want, &x, &pe, &w);

    const gd = try got.dataConst();
    for (0..T * D) |i| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want[i])), gd[i], 1e-4);
}

test "relpos attention rejects qkv projection width mismatch before packing" {
    const allocator = std.testing.allocator;
    const w = genWeights();

    var wq_bad: [D * (D - 1)]f32 = undefined;
    for (&wq_bad, 0..) |*v, i| v.* = gen(i, 70);
    var bq_bad: [D - 1]f32 = undefined;
    for (&bq_bad, 0..) |*v, i| v.* = gen(i, 71);

    var writer = Writer.init(allocator);
    defer writer.deinit();
    try addW(&writer, "encoder.layers.0.self_attn.linear_q.weight", &wq_bad, D, D - 1);
    try addV(&writer, "encoder.layers.0.self_attn.linear_q.bias", &bq_bad);
    try addW(&writer, "encoder.layers.0.self_attn.linear_k.weight", &w.wk, D, D);
    try addV(&writer, "encoder.layers.0.self_attn.linear_k.bias", &w.bk);
    try addW(&writer, "encoder.layers.0.self_attn.linear_v.weight", &w.wv, D, D);
    try addV(&writer, "encoder.layers.0.self_attn.linear_v.bias", &w.bv);
    try addW(&writer, "encoder.layers.0.self_attn.linear_out.weight", &w.wo, D, D);
    try addV(&writer, "encoder.layers.0.self_attn.linear_out.bias", &w.bo);
    try addW(&writer, "encoder.layers.0.self_attn.linear_pos.weight", &w.wp, D, D);
    try addW(&writer, "encoder.layers.0.self_attn.pos_bias_u", &w.bu, DK, H);
    try addW(&writer, "encoder.layers.0.self_attn.pos_bias_v", &w.bvb, DK, H);

    var buf: [64 * 1024]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try writer.finish(&sink);
    var file = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer file.deinit();

    var x: [T * D]f32 = undefined;
    for (&x, 0..) |*vv, i| vv.* = gen(i, 72);
    var pe: [P * D]f32 = undefined;
    for (&pe, 0..) |*vv, i| vv.* = gen(i, 73);

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var xt = try fucina.Tensor(2).fromSlice(&ctx, .{ T, D }, &x);
    defer xt.deinit();
    var pet = try fucina.Tensor(2).fromSlice(&ctx, .{ P, D }, &pe);
    defer pet.deinit();

    var pw = ParakeetWeights.init(&ctx, &file);
    defer pw.deinit();
    try std.testing.expectError(error.InvalidWeightShape, enc.relposAttention(&ctx, &pw, tinyConfig(), 0, &xt, &pet, null, null));
}

// --- conv module vs naive f64 reference (synthetic GGUF) ---

const Dc = 4;
const Kc = 3; // odd; pad = 1
const Tc = 5;

fn convCfg() loader.Config {
    var c = tinyConfig();
    c.d_model = Dc;
    c.n_heads = 2;
    c.head_dim = Dc / 2;
    c.conv_kernel = Kc;
    return c;
}

const ConvW = struct {
    pw1: [2 * Dc * Dc]f32, // ggml [1, Dc, 2Dc]; W[o*Dc+i]
    pw1b: [2 * Dc]f32,
    dww: [Dc * Kc]f32, // ggml [Kc,1,Dc]; w[c*Kc+k]
    dwb: [Dc]f32,
    g: [Dc]f32,
    b: [Dc]f32,
    m: [Dc]f32,
    v: [Dc]f32, // batch_norm
    pw2: [Dc * Dc]f32, // ggml [1, Dc, Dc]
    pw2b: [Dc]f32,
};

fn genConvW() ConvW {
    var w: ConvW = undefined;
    for (&w.pw1, 0..) |*x, i| x.* = gen(i, 30);
    for (&w.pw1b, 0..) |*x, i| x.* = gen(i, 31);
    for (&w.dww, 0..) |*x, i| x.* = gen(i, 32);
    for (&w.dwb, 0..) |*x, i| x.* = gen(i, 33);
    for (&w.g, 0..) |*x, i| x.* = 0.5 + gen(i, 34); // keep variance positive-ish
    for (&w.b, 0..) |*x, i| x.* = gen(i, 35);
    for (&w.m, 0..) |*x, i| x.* = gen(i, 36);
    for (&w.v, 0..) |*x, i| x.* = 0.3 + @abs(gen(i, 37)); // var > 0
    for (&w.pw2, 0..) |*x, i| x.* = gen(i, 38);
    for (&w.pw2b, 0..) |*x, i| x.* = gen(i, 39);
    return w;
}

fn sig(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

fn naiveConv(out: []f64, cin: []const f32, w: *const ConvW) void {
    const pad = (Kc - 1) / 2;
    // pw1: y[t,o] = sum_i cin[t,i]*pw1[o*Dc+i] + pw1b[o]
    var y: [Tc * 2 * Dc]f64 = undefined;
    for (0..Tc) |t| for (0..2 * Dc) |o| {
        var acc: f64 = w.pw1b[o];
        for (0..Dc) |i| acc += @as(f64, cin[t * Dc + i]) * @as(f64, w.pw1[o * Dc + i]);
        y[t * 2 * Dc + o] = acc;
    };
    // glu
    var glu: [Tc * Dc]f64 = undefined;
    for (0..Tc) |t| for (0..Dc) |cc| {
        glu[t * Dc + cc] = y[t * 2 * Dc + cc] * sig(y[t * 2 * Dc + Dc + cc]);
    };
    // depthwise + bn + silu
    var s: [Tc * Dc]f64 = undefined;
    for (0..Dc) |cc| {
        const scale = @as(f64, w.g[cc]) / @sqrt(@as(f64, w.v[cc]) + 1e-5);
        const shift = @as(f64, w.b[cc]) - @as(f64, w.m[cc]) * scale;
        for (0..Tc) |t| {
            var acc: f64 = w.dwb[cc];
            for (0..Kc) |k| {
                const tt = @as(i64, @intCast(t + k)) - pad;
                if (tt < 0 or tt >= Tc) continue;
                acc += glu[@as(usize, @intCast(tt)) * Dc + cc] * @as(f64, w.dww[cc * Kc + k]);
            }
            const bn = acc * scale + shift;
            s[t * Dc + cc] = bn * sig(bn);
        }
    }
    // pw2
    for (0..Tc) |t| for (0..Dc) |o| {
        var acc: f64 = w.pw2b[o];
        for (0..Dc) |i| acc += s[t * Dc + i] * @as(f64, w.pw2[o * Dc + i]);
        out[t * Dc + o] = acc;
    };
}

test "conv module matches a naive f64 reference (synthetic GGUF)" {
    const allocator = std.testing.allocator;
    const w = genConvW();

    var writer = Writer.init(allocator);
    defer writer.deinit();
    try addT(&writer, "encoder.layers.0.conv.pointwise_conv1.weight", &w.pw1, &.{ 1, Dc, 2 * Dc });
    try addV(&writer, "encoder.layers.0.conv.pointwise_conv1.bias", &w.pw1b);
    try addT(&writer, "encoder.layers.0.conv.depthwise_conv.weight", &w.dww, &.{ Kc, 1, Dc });
    try addV(&writer, "encoder.layers.0.conv.depthwise_conv.bias", &w.dwb);
    try addV(&writer, "encoder.layers.0.conv.batch_norm.weight", &w.g);
    try addV(&writer, "encoder.layers.0.conv.batch_norm.bias", &w.b);
    try addV(&writer, "encoder.layers.0.conv.batch_norm.running_mean", &w.m);
    try addV(&writer, "encoder.layers.0.conv.batch_norm.running_var", &w.v);
    try addT(&writer, "encoder.layers.0.conv.pointwise_conv2.weight", &w.pw2, &.{ 1, Dc, Dc });
    try addV(&writer, "encoder.layers.0.conv.pointwise_conv2.bias", &w.pw2b);

    var buf: [64 * 1024]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try writer.finish(&sink);
    var file = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer file.deinit();

    var cin: [Tc * Dc]f32 = undefined;
    for (&cin, 0..) |*v, i| v.* = gen(i, 40);

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var cint = try fucina.Tensor(2).fromSlice(&ctx, .{ Tc, Dc }, &cin);
    defer cint.deinit();

    var pw = ParakeetWeights.init(&ctx, &file);
    defer pw.deinit();
    var got = try enc.convModule(&ctx, &pw, convCfg(), 0, &cint);
    defer got.deinit();

    var want: [Tc * Dc]f64 = undefined;
    naiveConv(&want, &cin, &w);
    const gd = try got.dataConst();
    for (0..Tc * Dc) |i| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want[i])), gd[i], 1e-4);
}

test "layerNorm: per-row normalize + affine vs naive f64" {
    const allocator = std.testing.allocator;
    const Tn = 4;
    const Dn = 6;
    var in: [Tn * Dn]f32 = undefined;
    for (&in, 0..) |*v, i| v.* = gen(i, 50);
    var g: [Dn]f32 = undefined;
    var b: [Dn]f32 = undefined;
    for (&g, 0..) |*v, i| v.* = 0.5 + gen(i, 51);
    for (&b, 0..) |*v, i| v.* = gen(i, 52);

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var got = try enc.layerNorm(&ctx, &in, Tn, Dn, &g, &b);
    defer got.deinit();
    const gd = try got.dataConst();

    for (0..Tn) |t| {
        var mean: f64 = 0;
        for (0..Dn) |d| mean += in[t * Dn + d];
        mean /= Dn;
        var v: f64 = 0;
        for (0..Dn) |d| {
            const e = @as(f64, in[t * Dn + d]) - mean;
            v += e * e;
        }
        v /= Dn;
        const inv = 1.0 / @sqrt(v + 1e-5);
        for (0..Dn) |d| {
            const want = (@as(f64, in[t * Dn + d]) - mean) * inv * @as(f64, g[d]) + @as(f64, b[d]);
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), gd[t * Dn + d], 1e-5);
        }
    }
}
