//! Hermetic plumbing test for the subsampling stem. Builds a minimal GGUF with
//! the `encoder.pre_encode.*` tensors (tiny dims), zero conv weights + a known
//! `out.bias`, so the output is exactly the broadcast bias — this locks the
//! output shape, the channel-major flatten, and the linear + bias path without a
//! model. Full numeric parity is the `--compare subsampling` gate (Accept).
const std = @import("std");
const fucina = @import("fucina");
const gguf_mod = fucina.gguf;
const loader = @import("loader.zig");
const sub = @import("subsampling.zig");

const ExecContext = fucina.ExecContext;
const Writer = gguf_mod.Writer;

const C = 4; // conv channels
const NM = 8; // n_mels
const DM = 8; // d_model
const SUBFREQ = 1; // subsampledFreq(8, stages=3) = 8->4->2->1
const K = C * SUBFREQ; // 4

fn addZ(w: *Writer, zeros: []const u8, name: []const u8, dims: []const usize) !void {
    var cnt: usize = 1;
    for (dims) |x| cnt *= x;
    try w.addTensor(name, .f32, dims, zeros[0 .. cnt * 4]);
}

fn tinyConfig() loader.Config {
    return .{
        .arch = .hybrid_tdt_ctc,
        .d_model = DM,
        .n_layers = 1,
        .n_heads = 2,
        .head_dim = DM / 2,
        .ff_dim = 16,
        .feat_in = NM,
        .conv_kernel = 9,
        .conv_norm = .batch_norm,
        .subsampling_factor = 8,
        .subsampling_stages = 3,
        .subsampling_conv_channels = C,
        .pos_emb_max_len = 100,
        .xscaling = false,
        .sample_rate = 16000,
        .n_mels = NM,
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

test "subsampling stem: shape + bias-only output (zero conv weights)" {
    const allocator = std.testing.allocator;

    var w = Writer.init(allocator);
    defer w.deinit();
    const zeros = try allocator.alloc(u8, 1024);
    defer allocator.free(zeros);
    @memset(zeros, 0);

    // pre_encode conv stack (all-zero weights/biases).
    try addZ(&w, zeros, "encoder.pre_encode.conv.0.weight", &.{ 3, 3, 1, C });
    try addZ(&w, zeros, "encoder.pre_encode.conv.0.bias", &.{C});
    try addZ(&w, zeros, "encoder.pre_encode.conv.2.weight", &.{ 3, 3, 1, C });
    try addZ(&w, zeros, "encoder.pre_encode.conv.2.bias", &.{C});
    try addZ(&w, zeros, "encoder.pre_encode.conv.3.weight", &.{ 1, 1, C, C });
    try addZ(&w, zeros, "encoder.pre_encode.conv.3.bias", &.{C});
    try addZ(&w, zeros, "encoder.pre_encode.conv.5.weight", &.{ 3, 3, 1, C });
    try addZ(&w, zeros, "encoder.pre_encode.conv.5.bias", &.{C});
    try addZ(&w, zeros, "encoder.pre_encode.conv.6.weight", &.{ 1, 1, C, C });
    try addZ(&w, zeros, "encoder.pre_encode.conv.6.bias", &.{C});
    // out.weight [K, DM] (ggml ne), zero; out.bias known.
    try addZ(&w, zeros, "encoder.pre_encode.out.weight", &.{ K, DM });
    const bias = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var bias_bytes: [DM * 4]u8 = undefined;
    for (bias, 0..) |v, i| std.mem.writeInt(u32, bias_bytes[i * 4 ..][0..4], @bitCast(v), .little);
    try w.addTensor("encoder.pre_encode.out.bias", .f32, &.{DM}, &bias_bytes);

    var buf: [64 * 1024]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try w.finish(&sink);
    var file = try gguf_mod.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer file.deinit();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const T = 16; // subsample_len(16) = 16->8->4->2 -> T'=2
    const mel = try allocator.alloc(f32, NM * T);
    defer allocator.free(mel);
    @memset(mel, 0.5); // arbitrary; conv weights are zero so it cannot affect output

    var out = try sub.subsample(&ctx, &file, tinyConfig(), mel, NM, T);
    defer out.deinit();

    const view = out.shape();
    try std.testing.expectEqual(@as(usize, 2), view[0]); // T'
    try std.testing.expectEqual(@as(usize, DM), view[1]); // d_model
    // Zero conv weights -> flat == 0 -> y == out.bias broadcast over T'.
    const data = try out.dataConst();
    for (0..view[0]) |t| {
        for (0..DM) |d| try std.testing.expectApproxEqAbs(bias[d], data[t * DM + d], 1e-6);
    }
}
