//! Tests for the parakeet decoders (CTC greedy collapse). The CTC head +
//! full decode are validated end-to-end by the `--compare ctc` token-id parity
//! gate (exact match vs parakeet.cpp); the collapse rule is locked here.
const std = @import("std");
const dec = @import("decoder.zig");

test "ctcCollapse: fold consecutive + drop blanks (blank=0)" {
    const allocator = std.testing.allocator;
    // blank=0. "5 5 . 5 3 3 . . 7" -> 5 (run), 5 (new after blank), 3, 7.
    const am = [_]i32{ 0, 5, 5, 0, 5, 3, 3, 0, 0, 7 };
    const out = try dec.ctcCollapse(allocator, &am, 0);
    defer allocator.free(out);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 5, 5, 3, 7 }, out);
}

test "ctcCollapse: leading repeats and all-blank edge cases" {
    const allocator = std.testing.allocator;
    // leading run of a token (no preceding blank) emits once.
    {
        const am = [_]i32{ 3, 3, 3 };
        const out = try dec.ctcCollapse(allocator, &am, 0);
        defer allocator.free(out);
        try std.testing.expectEqualSlices(i32, &[_]i32{3}, out);
    }
    // all blanks -> empty.
    {
        const am = [_]i32{ 0, 0, 0 };
        const out = try dec.ctcCollapse(allocator, &am, 0);
        defer allocator.free(out);
        try std.testing.expectEqual(@as(usize, 0), out.len);
    }
    // nonzero blank id (e.g. 1024); token equal to blank never emits.
    {
        const am = [_]i32{ 1024, 7, 1024, 7, 7, 1024 };
        const out = try dec.ctcCollapse(allocator, &am, 1024);
        defer allocator.free(out);
        try std.testing.expectEqualSlices(i32, &[_]i32{ 7, 7 }, out);
    }
}

// --- Predictor (LSTM) + Joint vs inline naive f64 (synthetic GGUF) ---
const fucina = @import("fucina");
const gguf = fucina.gguf;
const loader = @import("loader.zig");
const ParakeetWeights = @import("weights.zig").ParakeetWeights;
const Writer = gguf.Writer;

const PH = 2; // pred_hidden
const EH = 3; // d_model (enc hidden)
const JH = 2; // joint_hidden
const VOC = 2; // vocab
const VP1 = VOC + 1; // embed rows
const ND = 2; // durations
const VP = VOC + 1 + ND; // V_plus = 5

fn dgen(i: usize, salt: usize) f32 {
    return @floatCast(0.2 * @sin(@as(f64, @floatFromInt(i * 5 + salt * 11 + 1)) * 0.7));
}
fn addT(w: *Writer, name: []const u8, data: []const f32, dims: []const usize) !void {
    try w.addTensor(name, .f32, dims, std.mem.sliceAsBytes(data));
}
fn sg(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

fn cfgD() loader.Config {
    return .{
        .arch = .hybrid_tdt_ctc,
        .d_model = EH,
        .n_layers = 1,
        .n_heads = 1,
        .head_dim = EH,
        .ff_dim = 4,
        .feat_in = 4,
        .conv_kernel = 3,
        .conv_norm = .batch_norm,
        .subsampling_factor = 8,
        .subsampling_stages = 3,
        .subsampling_conv_channels = 4,
        .pos_emb_max_len = 100,
        .xscaling = false,
        .sample_rate = 16000,
        .n_mels = 4,
        .n_fft = 512,
        .win_length = 400,
        .hop_length = 160,
        .preemph = 0.97,
        .mag_power = 2.0,
        .log_zero_guard = 1e-5,
        .normalize = .per_feature,
        .vocab_size = VOC,
        .blank_id = VOC,
        .pred_hidden = PH,
        .pred_rnn_layers = 1,
        .max_symbols = 10,
        .joint_hidden = JH,
        .joint_activation = .relu,
        .num_durations = ND,
        .durations = [_]i32{ 0, 1 } ++ [_]i32{0} ** 14,
    };
}

test "Predictor SOS step + Joint step vs inline naive f64" {
    const allocator = std.testing.allocator;
    // weights
    var wih: [PH * 4 * PH]f32 = undefined; // ggml [PH, 4PH]
    var whh: [PH * 4 * PH]f32 = undefined;
    var bih: [4 * PH]f32 = undefined;
    var bhh: [4 * PH]f32 = undefined;
    var embed: [PH * VP1]f32 = undefined; // ggml [PH, VP1]
    var encw: [EH * JH]f32 = undefined; // ggml [EH, JH]
    var encb: [JH]f32 = undefined;
    var predw: [PH * JH]f32 = undefined; // ggml [PH, JH]
    var predb: [JH]f32 = undefined;
    var outw: [JH * VP]f32 = undefined; // ggml [JH, VP]
    var outb: [VP]f32 = undefined;
    for (&wih, 0..) |*v, i| v.* = dgen(i, 1);
    for (&whh, 0..) |*v, i| v.* = dgen(i, 2);
    for (&bih, 0..) |*v, i| v.* = dgen(i, 3);
    for (&bhh, 0..) |*v, i| v.* = dgen(i, 4);
    for (&embed, 0..) |*v, i| v.* = dgen(i, 5);
    for (&encw, 0..) |*v, i| v.* = dgen(i, 6);
    for (&encb, 0..) |*v, i| v.* = dgen(i, 7);
    for (&predw, 0..) |*v, i| v.* = dgen(i, 8);
    for (&predb, 0..) |*v, i| v.* = dgen(i, 9);
    for (&outw, 0..) |*v, i| v.* = dgen(i, 10);
    for (&outb, 0..) |*v, i| v.* = dgen(i, 11);

    var w = Writer.init(allocator);
    defer w.deinit();
    try addT(&w, "decoder.prediction.dec_rnn.lstm.weight_ih_l0", &wih, &.{ PH, 4 * PH });
    try addT(&w, "decoder.prediction.dec_rnn.lstm.weight_hh_l0", &whh, &.{ PH, 4 * PH });
    try addT(&w, "decoder.prediction.dec_rnn.lstm.bias_ih_l0", &bih, &.{4 * PH});
    try addT(&w, "decoder.prediction.dec_rnn.lstm.bias_hh_l0", &bhh, &.{4 * PH});
    try addT(&w, "decoder.prediction.embed.weight", &embed, &.{ PH, VP1 });
    try addT(&w, "joint.enc.weight", &encw, &.{ EH, JH });
    try addT(&w, "joint.enc.bias", &encb, &.{JH});
    try addT(&w, "joint.pred.weight", &predw, &.{ PH, JH });
    try addT(&w, "joint.pred.bias", &predb, &.{JH});
    try addT(&w, "joint.joint_net.2.weight", &outw, &.{ JH, VP });
    try addT(&w, "joint.joint_net.2.bias", &outb, &.{VP});

    var buf: [16 * 1024]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try w.finish(&sink);
    var file = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer file.deinit();

    const cfg = cfgD();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var pw = ParakeetWeights.init(&ctx, &file);
    defer pw.deinit();
    var pred = try dec.Predictor.init(allocator, &pw, cfg);
    defer pred.deinit();
    var joint = try dec.Joint.init(allocator, &pw, cfg);
    defer joint.deinit();

    // Predictor SOS step (x=0, h=0, c=0).
    var zeros = [_]f32{0} ** PH;
    var g: [PH]f32 = undefined;
    var ho: [PH]f32 = undefined;
    var co: [PH]f32 = undefined;
    try pred.step(&ctx, -1, true, &zeros, &zeros, &g, &ho, &co);

    // naive SOS: z[gate] = bih+bhh (x=0,h=0). gate order i,f,g,o.
    var ng: [PH]f64 = undefined;
    for (0..PH) |o| {
        const zi = @as(f64, bih[0 * PH + o]) + @as(f64, bhh[0 * PH + o]);
        const zf = @as(f64, bih[1 * PH + o]) + @as(f64, bhh[1 * PH + o]);
        const zg = @as(f64, bih[2 * PH + o]) + @as(f64, bhh[2 * PH + o]);
        const zo = @as(f64, bih[3 * PH + o]) + @as(f64, bhh[3 * PH + o]);
        const cv = sg(zf) * 0.0 + sg(zi) * std.math.tanh(zg);
        ng[o] = sg(zo) * std.math.tanh(cv);
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(ng[o])), g[o], 1e-4);
    }

    // Joint step on a known enc frame ([1, EH] → batched encProjAll → [1, JH]).
    var enc_t = [_]f32{ 0.3, -0.1, 0.5 }; // [EH]
    var enc_t_tensor = try fucina.Tensor(2).fromSlice(&ctx, .{ 1, EH }, &enc_t);
    defer enc_t_tensor.deinit();
    var ep = try joint.encProjAll(&enc_t_tensor);
    defer ep.deinit();
    var logits: [VP]f32 = undefined;
    try joint.step(&ctx, (try ep.dataConst())[0..JH], &g, &logits);

    // naive joint
    var nf: [JH]f64 = undefined;
    for (0..JH) |h| {
        var epa: f64 = encb[h];
        for (0..EH) |e| epa += @as(f64, encw[h * EH + e]) * @as(f64, enc_t[e]);
        var pp: f64 = predb[h];
        for (0..PH) |pi| pp += @as(f64, predw[h * PH + pi]) * ng[pi];
        const s = epa + pp;
        nf[h] = if (s > 0) s else 0;
    }
    for (0..VP) |v| {
        var acc: f64 = outb[v];
        for (0..JH) |h| acc += @as(f64, outw[v * JH + h]) * nf[h];
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), logits[v], 1e-4);
    }
}
