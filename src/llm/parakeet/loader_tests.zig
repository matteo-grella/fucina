//! Hermetic tests for the parakeet GGUF loader. Build a tiny but structurally
//! complete `hybrid_tdt_ctc` GGUF in memory (via gguf.Writer), parse it back, and
//! assert Config parsing + full tensor resolution/validation + featurizer borrow +
//! piece decode. No dependency on the (gitignored) real models.
const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;
const loader = @import("loader.zig");

const Writer = gguf.Writer;

// Tiny config the synthetic GGUF encodes (keeps the test fast + self-contained).
const D = 8; // d_model
const HEADS = 2; // -> head_dim = 4
const HD = D / HEADS;
const FF = 16; // ff_dim
const NM = 8; // n_mels / feat_in
const NFFT = 8; // -> fb bins = 5
const FB_BINS = NFFT / 2 + 1;
const WIN = 6;
const CC = 4; // subsampling conv channels
const CK = 4; // conformer conv kernel
const SUB = 8; // subsampling_factor (stages=3) -> subsampledFreq(8)=1
const SUBFREQ = 1;
const NLAYERS = 2;
const VOCAB = 10;
const V1 = VOCAB + 1;
const PH = 6; // pred_hidden
const JH = 6; // joint_hidden
const NDUR = 3;
const VP = VOCAB + 1 + NDUR; // 14

fn addZ(w: *Writer, zeros: []const u8, name: []const u8, dims: []const usize) !void {
    var cnt: usize = 1;
    for (dims) |x| cnt *= x;
    try w.addTensor(name, .f32, dims, zeros[0 .. cnt * 4]);
}

fn addPair(w: *Writer, zeros: []const u8, base: []const u8, w_dims: []const usize, bias_dim: usize) !void {
    var buf: [288]u8 = undefined;
    try addZ(w, zeros, try std.fmt.bufPrint(&buf, "{s}.weight", .{base}), w_dims);
    try addZ(w, zeros, try std.fmt.bufPrint(&buf, "{s}.bias", .{base}), &.{bias_dim});
}

fn addNorm(w: *Writer, zeros: []const u8, base: []const u8) !void {
    try addPair(w, zeros, base, &.{D}, D);
}

fn addMeta(w: *Writer) !void {
    try w.addMetaString("general.architecture", "parakeet");
    try w.addMetaString("parakeet.arch", "hybrid_tdt_ctc");
    try w.addMetaInt("parakeet.encoder.d_model", u32, D);
    try w.addMetaInt("parakeet.encoder.n_layers", u32, NLAYERS);
    try w.addMetaInt("parakeet.encoder.n_heads", u32, HEADS);
    try w.addMetaInt("parakeet.encoder.ff_dim", u32, FF);
    try w.addMetaInt("parakeet.encoder.feat_in", u32, NM);
    try w.addMetaInt("parakeet.encoder.conv_kernel", u32, CK);
    try w.addMetaString("parakeet.encoder.conv_norm_type", "batch_norm");
    try w.addMetaInt("parakeet.encoder.subsampling_factor", u32, SUB);
    try w.addMetaInt("parakeet.encoder.subsampling_conv_channels", u32, CC);
    try w.addMetaInt("parakeet.encoder.pos_emb_max_len", u32, 100);
    try w.addMetaBool("parakeet.encoder.xscaling", false);
    try w.addMetaInt("parakeet.preprocessor.sample_rate", u32, 16000);
    try w.addMetaInt("parakeet.preprocessor.n_mels", u32, NM);
    try w.addMetaInt("parakeet.preprocessor.n_fft", u32, NFFT);
    try w.addMetaInt("parakeet.preprocessor.win_length", u32, WIN);
    try w.addMetaInt("parakeet.preprocessor.hop_length", u32, 4);
    try w.addMetaFloat("parakeet.preprocessor.preemph", f32, 0.97);
    try w.addMetaFloat("parakeet.preprocessor.mag_power", f32, 2.0);
    try w.addMetaFloat("parakeet.preprocessor.log_zero_guard", f32, 5.9604645e-8);
    try w.addMetaString("parakeet.preprocessor.normalize", "per_feature");
    try w.addMetaInt("parakeet.vocab_size", u32, VOCAB);
    try w.addMetaInt("parakeet.blank_id", u32, VOCAB);
    try w.addMetaInt("parakeet.decoder.pred_hidden", u32, PH);
    try w.addMetaInt("parakeet.decoder.pred_rnn_layers", u32, 1);
    try w.addMetaInt("parakeet.decoding.max_symbols", u32, 10);
    try w.addMetaInt("parakeet.joint.joint_hidden", u32, JH);
    try w.addMetaString("parakeet.joint.activation", "relu");
    try w.addMetaArray("parakeet.tdt.durations", i32, &.{ 0, 1, 2 });
    try w.addMetaStringArray("parakeet.tokenizer.pieces", &.{
        "<unk>", "a", "b", "c", "d", "e", "f", "g", "h", "i",
    });
}

fn addTensors(w: *Writer, zeros: []const u8) !void {
    var nb: [288]u8 = undefined;

    // Featurizer.
    try addZ(w, zeros, "preprocessor.featurizer.window", &.{WIN});
    try addZ(w, zeros, "preprocessor.featurizer.fb", &.{ FB_BINS, NM, 1 });

    // Subsampling stem.
    try addPair(w, zeros, "encoder.pre_encode.conv.0", &.{ 3, 3, 1, CC }, CC);
    try addPair(w, zeros, "encoder.pre_encode.conv.2", &.{ 3, 3, 1, CC }, CC);
    try addPair(w, zeros, "encoder.pre_encode.conv.3", &.{ 1, 1, CC, CC }, CC);
    try addPair(w, zeros, "encoder.pre_encode.conv.5", &.{ 3, 3, 1, CC }, CC);
    try addPair(w, zeros, "encoder.pre_encode.conv.6", &.{ 1, 1, CC, CC }, CC);
    try addPair(w, zeros, "encoder.pre_encode.out", &.{ CC * SUBFREQ, D }, D);

    // Conformer blocks.
    for (0..NLAYERS) |il| {
        try addNorm(w, zeros, try ln(&nb, il, "norm_feed_forward1"));
        try addPair(w, zeros, try ln(&nb, il, "feed_forward1.linear1"), &.{ D, FF }, FF);
        try addPair(w, zeros, try ln(&nb, il, "feed_forward1.linear2"), &.{ FF, D }, D);
        try addNorm(w, zeros, try ln(&nb, il, "norm_self_att"));
        try addPair(w, zeros, try ln(&nb, il, "self_attn.linear_q"), &.{ D, D }, D);
        try addPair(w, zeros, try ln(&nb, il, "self_attn.linear_k"), &.{ D, D }, D);
        try addPair(w, zeros, try ln(&nb, il, "self_attn.linear_v"), &.{ D, D }, D);
        try addPair(w, zeros, try ln(&nb, il, "self_attn.linear_out"), &.{ D, D }, D);
        try addZ(w, zeros, try ln(&nb, il, "self_attn.linear_pos.weight"), &.{ D, D });
        try addZ(w, zeros, try ln(&nb, il, "self_attn.pos_bias_u"), &.{ HD, HEADS });
        try addZ(w, zeros, try ln(&nb, il, "self_attn.pos_bias_v"), &.{ HD, HEADS });
        try addNorm(w, zeros, try ln(&nb, il, "norm_conv"));
        try addZ(w, zeros, try ln(&nb, il, "conv.pointwise_conv1.weight"), &.{ 1, D, 2 * D });
        try addZ(w, zeros, try ln(&nb, il, "conv.pointwise_conv1.bias"), &.{2 * D});
        try addZ(w, zeros, try ln(&nb, il, "conv.depthwise_conv.weight"), &.{ CK, 1, D });
        try addZ(w, zeros, try ln(&nb, il, "conv.depthwise_conv.bias"), &.{D});
        try addZ(w, zeros, try ln(&nb, il, "conv.batch_norm.weight"), &.{D});
        try addZ(w, zeros, try ln(&nb, il, "conv.batch_norm.bias"), &.{D});
        try addZ(w, zeros, try ln(&nb, il, "conv.batch_norm.running_mean"), &.{D});
        try addZ(w, zeros, try ln(&nb, il, "conv.batch_norm.running_var"), &.{D});
        try addZ(w, zeros, try ln(&nb, il, "conv.pointwise_conv2.weight"), &.{ 1, D, D });
        try addZ(w, zeros, try ln(&nb, il, "conv.pointwise_conv2.bias"), &.{D});
        try addNorm(w, zeros, try ln(&nb, il, "norm_feed_forward2"));
        try addPair(w, zeros, try ln(&nb, il, "feed_forward2.linear1"), &.{ D, FF }, FF);
        try addPair(w, zeros, try ln(&nb, il, "feed_forward2.linear2"), &.{ FF, D }, D);
        try addNorm(w, zeros, try ln(&nb, il, "norm_out"));
    }

    // Transducer: LSTM predictor + joint.
    try addZ(w, zeros, "decoder.prediction.embed.weight", &.{ PH, V1 });
    try addZ(w, zeros, "decoder.prediction.dec_rnn.lstm.weight_ih_l0", &.{ PH, 4 * PH });
    try addZ(w, zeros, "decoder.prediction.dec_rnn.lstm.weight_hh_l0", &.{ PH, 4 * PH });
    try addZ(w, zeros, "decoder.prediction.dec_rnn.lstm.bias_ih_l0", &.{4 * PH});
    try addZ(w, zeros, "decoder.prediction.dec_rnn.lstm.bias_hh_l0", &.{4 * PH});
    try addPair(w, zeros, "joint.enc", &.{ D, JH }, JH);
    try addPair(w, zeros, "joint.pred", &.{ PH, JH }, JH);
    try addPair(w, zeros, "joint.joint_net.2", &.{ JH, VP }, VP);

    // CTC head.
    try addZ(w, zeros, "ctc_decoder.decoder_layers.0.weight", &.{ 1, D, V1 });
    try addZ(w, zeros, "ctc_decoder.decoder_layers.0.bias", &.{V1});
}

fn ln(buf: []u8, il: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "encoder.layers.{d}.{s}", .{ il, suffix });
}

fn buildSyntheticGguf(allocator: std.mem.Allocator) ![]u8 {
    var w = Writer.init(allocator);
    defer w.deinit();

    const zeros = try allocator.alloc(u8, 4096);
    defer allocator.free(zeros);
    @memset(zeros, 0);

    try addMeta(&w);
    try addTensors(&w, zeros);

    const buf = try allocator.alloc(u8, 512 * 1024);
    defer allocator.free(buf);
    var sink = std.Io.Writer.fixed(buf);
    try w.finish(&sink);
    return allocator.dupe(u8, sink.buffered());
}

test "parakeet loader: config + full tensor validation on a synthetic GGUF" {
    const allocator = std.testing.allocator;

    const bytes = try buildSyntheticGguf(allocator);
    var file = try gguf.File.parseOwned(allocator, bytes); // takes ownership of `bytes`
    defer file.deinit();

    const cfg = try loader.Config.fromGguf(&file);

    // Config.
    try std.testing.expectEqual(loader.DecoderArch.hybrid_tdt_ctc, cfg.arch);
    try std.testing.expect(cfg.arch.hasCtc());
    try std.testing.expect(cfg.arch.hasTransducer());
    try std.testing.expect(cfg.arch.isTdt());
    try std.testing.expectEqual(@as(usize, D), cfg.d_model);
    try std.testing.expectEqual(@as(usize, NLAYERS), cfg.n_layers);
    try std.testing.expectEqual(@as(usize, HEADS), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, HD), cfg.head_dim);
    try std.testing.expectEqual(@as(usize, FF), cfg.ff_dim);
    try std.testing.expectEqual(@as(usize, NM), cfg.n_mels);
    try std.testing.expectEqual(@as(usize, NM), cfg.feat_in);
    try std.testing.expectEqual(@as(usize, NFFT), cfg.n_fft);
    try std.testing.expectEqual(@as(usize, WIN), cfg.win_length);
    try std.testing.expectEqual(@as(usize, CK), cfg.conv_kernel);
    try std.testing.expectEqual(loader.ConvNorm.batch_norm, cfg.conv_norm);
    try std.testing.expectEqual(@as(usize, SUB), cfg.subsampling_factor);
    try std.testing.expectEqual(@as(usize, 3), cfg.subsampling_stages);
    try std.testing.expectEqual(@as(usize, SUBFREQ), cfg.subsampledFreq());
    try std.testing.expectEqual(@as(usize, CC), cfg.subsampling_conv_channels);
    try std.testing.expectEqual(@as(usize, VOCAB), cfg.vocab_size);
    try std.testing.expectEqual(@as(usize, VOCAB), cfg.blank_id);
    try std.testing.expectEqual(@as(usize, PH), cfg.pred_hidden);
    try std.testing.expectEqual(@as(usize, 1), cfg.pred_rnn_layers);
    try std.testing.expectEqual(@as(usize, JH), cfg.joint_hidden);
    try std.testing.expectEqual(loader.JointActivation.relu, cfg.joint_activation);
    try std.testing.expectEqual(@as(usize, NDUR), cfg.num_durations);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2 }, cfg.durationsSlice());
    try std.testing.expectEqual(@as(usize, VP), cfg.vPlus());
    try std.testing.expectEqual(@as(f32, 0.97), cfg.preemph);

    // Every expected tensor resolves + validates. Count is deterministic:
    // 2 featurizer + 12 stem + 39*NLAYERS layer + 5 predictor + 6 joint + 2 ctc.
    const expected_count: usize = 2 + 12 + 39 * NLAYERS + 5 + 6 + 2;
    try std.testing.expectEqual(expected_count, try loader.validateTensors(&file, cfg));

    // Featurizer borrow.
    const feat = try loader.loadFeaturizer(&file, cfg);
    try std.testing.expectEqual(@as(usize, FB_BINS * NM), feat.fb.len);
    try std.testing.expectEqual(@as(usize, FB_BINS), feat.fb_bins);
    try std.testing.expectEqual(@as(usize, NM), feat.fb_mels);
    try std.testing.expectEqual(@as(usize, WIN), feat.window.len);

    // Pieces.
    const pieces = try loader.loadPieces(&file, allocator);
    defer allocator.free(pieces);
    try std.testing.expectEqual(@as(usize, VOCAB), pieces.len);
    try std.testing.expectEqualStrings("<unk>", pieces[0]);
    try std.testing.expectEqualStrings("i", pieces[9]);
}

test "parakeet loader: rejects non-parakeet and missing tensors" {
    const allocator = std.testing.allocator;

    // Wrong general.architecture -> NotParakeet.
    {
        var w = Writer.init(allocator);
        defer w.deinit();
        try w.addMetaString("general.architecture", "llama");
        const buf = try allocator.alloc(u8, 4096);
        defer allocator.free(buf);
        var sink = std.Io.Writer.fixed(buf);
        try w.finish(&sink);
        var file = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
        defer file.deinit();
        try std.testing.expectError(loader.Error.NotParakeet, loader.Config.fromGguf(&file));
    }

    // Valid config but a tensor with the wrong shape -> TensorShapeMismatch.
    {
        const bytes = try buildSyntheticGgufMissing(allocator);
        var file = try gguf.File.parseOwned(allocator, bytes);
        defer file.deinit();
        const cfg = try loader.Config.fromGguf(&file);
        try std.testing.expectError(loader.Error.TensorShapeMismatch, loader.validateTensors(&file, cfg));
    }
}

test "parakeet loader: streaming config parse + absent on offline models" {
    const allocator = std.testing.allocator;

    // (a) A streaming model's metadata -> StreamingConfig with the realtime_eou_120m-v1
    //     values (att_context [70,1] chunked_limited; chunk/shift [9,16]; pre-enc [0,9]).
    {
        var w = Writer.init(allocator);
        defer w.deinit();
        try w.addMetaString("general.architecture", "parakeet");
        try w.addMetaInt("parakeet.encoder.att_context_left", i32, 70);
        try w.addMetaInt("parakeet.encoder.att_context_right", i32, 1);
        try w.addMetaString("parakeet.encoder.att_context_style", "chunked_limited");
        try w.addMetaArray("parakeet.streaming.chunk_size", i32, &.{ 9, 16 });
        try w.addMetaArray("parakeet.streaming.shift_size", i32, &.{ 9, 16 });
        try w.addMetaArray("parakeet.streaming.pre_encode_cache_size", i32, &.{ 0, 9 });
        try w.addMetaInt("parakeet.streaming.cache_drop_size", i32, 0);
        try w.addMetaInt("parakeet.streaming.last_channel_cache_size", i32, 70);
        try w.addMetaInt("parakeet.streaming.valid_out_len", i32, 2);
        try w.addMetaInt("parakeet.streaming.drop_extra_pre_encoded", i32, 2);
        const buf = try allocator.alloc(u8, 16 * 1024);
        defer allocator.free(buf);
        var sink = std.Io.Writer.fixed(buf);
        try w.finish(&sink);
        var file = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
        defer file.deinit();

        const sc = (try loader.StreamingConfig.fromGguf(&file)) orelse return error.TestUnexpectedNull;
        try std.testing.expectEqual(@as(i32, 70), sc.att_context_left);
        try std.testing.expectEqual(@as(i32, 1), sc.att_context_right);
        try std.testing.expectEqual(loader.AttContextStyle.chunked_limited, sc.att_context_style);
        try std.testing.expectEqualSlices(i32, &.{ 9, 16 }, &sc.chunk_size);
        try std.testing.expectEqualSlices(i32, &.{ 9, 16 }, &sc.shift_size);
        try std.testing.expectEqualSlices(i32, &.{ 0, 9 }, &sc.pre_encode_cache_size);
        try std.testing.expectEqual(@as(i32, 0), sc.cache_drop_size);
        try std.testing.expectEqual(@as(i32, 70), sc.last_channel_cache_size);
        try std.testing.expectEqual(@as(i32, 2), sc.valid_out_len);
        try std.testing.expectEqual(@as(i32, 2), sc.drop_extra_pre_encoded);
        // step schedule indexing
        try std.testing.expectEqual(@as(i32, 9), sc.chunk_size[loader.StreamingConfig.stepIdx(0)]);
        try std.testing.expectEqual(@as(i32, 16), sc.chunk_size[loader.StreamingConfig.stepIdx(3)]);
    }

    // (b) The offline synthetic GGUF carries no streaming metadata -> null.
    {
        const bytes = try buildSyntheticGguf(allocator);
        var file = try gguf.File.parseOwned(allocator, bytes);
        defer file.deinit();
        try std.testing.expectEqual(@as(?loader.StreamingConfig, null), try loader.StreamingConfig.fromGguf(&file));
    }
}

test "parakeet loader: PromptConfig parse + locale resolution" {
    const allocator = std.testing.allocator;

    // (a) A prompt-conditioned model's metadata -> PromptConfig + resolveLang.
    {
        var w = Writer.init(allocator);
        defer w.deinit();
        try w.addMetaString("general.architecture", "parakeet");
        try w.addMetaBool("parakeet.prompt.present", true);
        try w.addMetaInt("parakeet.prompt.num_prompts", u32, 8);
        try w.addMetaString("parakeet.prompt.default_lang", "en");
        try w.addMetaStringArray("parakeet.prompt.dictionary.keys", &.{ "en", "de", "fr" });
        try w.addMetaArray("parakeet.prompt.dictionary.values", i32, &.{ 0, 3, 5 });
        const buf = try allocator.alloc(u8, 16 * 1024);
        defer allocator.free(buf);
        var sink = std.Io.Writer.fixed(buf);
        try w.finish(&sink);
        var file = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
        defer file.deinit();

        const pc = loader.PromptConfig.fromGguf(&file) orelse return error.TestUnexpectedNull;
        try std.testing.expect(pc.present);
        try std.testing.expectEqual(@as(i32, 8), pc.num_prompts);
        try std.testing.expectEqualStrings("en", pc.default_lang);
        try std.testing.expectEqual(@as(?i32, 3), try pc.resolveLang(&file, allocator, "de"));
        try std.testing.expectEqual(@as(?i32, 5), try pc.resolveLang(&file, allocator, "fr"));
        try std.testing.expectEqual(@as(?i32, 0), try pc.resolveLang(&file, allocator, "")); // empty -> default "en" -> 0
        try std.testing.expectEqual(@as(?i32, null), try pc.resolveLang(&file, allocator, "xx")); // unknown locale
    }

    // (b) A non-prompt model (no prompt.present) -> null.
    {
        const bytes = try buildSyntheticGguf(allocator);
        var file = try gguf.File.parseOwned(allocator, bytes);
        defer file.deinit();
        try std.testing.expectEqual(@as(?loader.PromptConfig, null), loader.PromptConfig.fromGguf(&file));
    }
}

// Same as buildSyntheticGguf but the CTC head weight has a deliberately wrong
// shape, to exercise validation failure.
fn buildSyntheticGgufMissing(allocator: std.mem.Allocator) ![]u8 {
    var w = Writer.init(allocator);
    defer w.deinit();
    const zeros = try allocator.alloc(u8, 4096);
    defer allocator.free(zeros);
    @memset(zeros, 0);
    try addMeta(&w);
    try addTensorsExceptCtcWeight(&w, zeros);
    try addZ(&w, zeros, "ctc_decoder.decoder_layers.0.weight", &.{ 1, D, V1 + 1 }); // wrong V1
    try addZ(&w, zeros, "ctc_decoder.decoder_layers.0.bias", &.{V1});
    const buf = try allocator.alloc(u8, 512 * 1024);
    defer allocator.free(buf);
    var sink = std.Io.Writer.fixed(buf);
    try w.finish(&sink);
    return allocator.dupe(u8, sink.buffered());
}

fn addTensorsExceptCtcWeight(w: *Writer, zeros: []const u8) !void {
    // Reuse addTensors but skip the two ctc tensors by adding everything else,
    // then the caller adds the (wrong) ctc tensors. Simplest: add all, but the
    // ctc names would clash; instead add all non-ctc via a trimmed copy.
    var nb: [288]u8 = undefined;
    try addZ(w, zeros, "preprocessor.featurizer.window", &.{WIN});
    try addZ(w, zeros, "preprocessor.featurizer.fb", &.{ FB_BINS, NM, 1 });
    try addPair(w, zeros, "encoder.pre_encode.conv.0", &.{ 3, 3, 1, CC }, CC);
    try addPair(w, zeros, "encoder.pre_encode.conv.2", &.{ 3, 3, 1, CC }, CC);
    try addPair(w, zeros, "encoder.pre_encode.conv.3", &.{ 1, 1, CC, CC }, CC);
    try addPair(w, zeros, "encoder.pre_encode.conv.5", &.{ 3, 3, 1, CC }, CC);
    try addPair(w, zeros, "encoder.pre_encode.conv.6", &.{ 1, 1, CC, CC }, CC);
    try addPair(w, zeros, "encoder.pre_encode.out", &.{ CC * SUBFREQ, D }, D);
    for (0..NLAYERS) |il| {
        try addNorm(w, zeros, try ln(&nb, il, "norm_feed_forward1"));
        try addPair(w, zeros, try ln(&nb, il, "feed_forward1.linear1"), &.{ D, FF }, FF);
        try addPair(w, zeros, try ln(&nb, il, "feed_forward1.linear2"), &.{ FF, D }, D);
        try addNorm(w, zeros, try ln(&nb, il, "norm_self_att"));
        try addPair(w, zeros, try ln(&nb, il, "self_attn.linear_q"), &.{ D, D }, D);
        try addPair(w, zeros, try ln(&nb, il, "self_attn.linear_k"), &.{ D, D }, D);
        try addPair(w, zeros, try ln(&nb, il, "self_attn.linear_v"), &.{ D, D }, D);
        try addPair(w, zeros, try ln(&nb, il, "self_attn.linear_out"), &.{ D, D }, D);
        try addZ(w, zeros, try ln(&nb, il, "self_attn.linear_pos.weight"), &.{ D, D });
        try addZ(w, zeros, try ln(&nb, il, "self_attn.pos_bias_u"), &.{ HD, HEADS });
        try addZ(w, zeros, try ln(&nb, il, "self_attn.pos_bias_v"), &.{ HD, HEADS });
        try addNorm(w, zeros, try ln(&nb, il, "norm_conv"));
        try addZ(w, zeros, try ln(&nb, il, "conv.pointwise_conv1.weight"), &.{ 1, D, 2 * D });
        try addZ(w, zeros, try ln(&nb, il, "conv.pointwise_conv1.bias"), &.{2 * D});
        try addZ(w, zeros, try ln(&nb, il, "conv.depthwise_conv.weight"), &.{ CK, 1, D });
        try addZ(w, zeros, try ln(&nb, il, "conv.depthwise_conv.bias"), &.{D});
        try addZ(w, zeros, try ln(&nb, il, "conv.batch_norm.weight"), &.{D});
        try addZ(w, zeros, try ln(&nb, il, "conv.batch_norm.bias"), &.{D});
        try addZ(w, zeros, try ln(&nb, il, "conv.batch_norm.running_mean"), &.{D});
        try addZ(w, zeros, try ln(&nb, il, "conv.batch_norm.running_var"), &.{D});
        try addZ(w, zeros, try ln(&nb, il, "conv.pointwise_conv2.weight"), &.{ 1, D, D });
        try addZ(w, zeros, try ln(&nb, il, "conv.pointwise_conv2.bias"), &.{D});
        try addNorm(w, zeros, try ln(&nb, il, "norm_feed_forward2"));
        try addPair(w, zeros, try ln(&nb, il, "feed_forward2.linear1"), &.{ D, FF }, FF);
        try addPair(w, zeros, try ln(&nb, il, "feed_forward2.linear2"), &.{ FF, D }, D);
        try addNorm(w, zeros, try ln(&nb, il, "norm_out"));
    }
    try addZ(w, zeros, "decoder.prediction.embed.weight", &.{ PH, V1 });
    try addZ(w, zeros, "decoder.prediction.dec_rnn.lstm.weight_ih_l0", &.{ PH, 4 * PH });
    try addZ(w, zeros, "decoder.prediction.dec_rnn.lstm.weight_hh_l0", &.{ PH, 4 * PH });
    try addZ(w, zeros, "decoder.prediction.dec_rnn.lstm.bias_ih_l0", &.{4 * PH});
    try addZ(w, zeros, "decoder.prediction.dec_rnn.lstm.bias_hh_l0", &.{4 * PH});
    try addPair(w, zeros, "joint.enc", &.{ D, JH }, JH);
    try addPair(w, zeros, "joint.pred", &.{ PH, JH }, JH);
    try addPair(w, zeros, "joint.joint_net.2", &.{ JH, VP }, VP);
}
