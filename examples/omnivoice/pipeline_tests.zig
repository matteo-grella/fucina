//! Tests for pipeline.zig — TTS pipeline orchestration.
//!
//! The end-to-end parity gates against the captured reference goldens
//! (refs/omnivoice-research/goldens/{maskgit,tts-design}/) need the multi-GB
//! model files under models/omnivoice/ and are gated behind OMNIVOICE_PARITY
//! (run them in an optimized build — Debug takes far too long):
//!
//!   OMNIVOICE_PARITY=1 zig build test -Doptimize=ReleaseSafe

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const codec = @import("codec.zig");
const dump = @import("dump.zig");
const lm = @import("lm.zig");
const maskgit = @import("maskgit.zig");
const pipeline = @import("pipeline.zig");
const wav = @import("wav.zig");

// ---------------------------------------------------------------------------
// Fast unit tests (no model files)
// ---------------------------------------------------------------------------

test "thresholdFrames: f32 multiply + C truncation" {
    // Reference defaults: 30 s * 25 fps = 750; the chunked golden run used
    // --chunk-threshold 5 -> 125.
    try std.testing.expectEqual(@as(i32, 750), pipeline.thresholdFrames(30.0, 25));
    try std.testing.expectEqual(@as(i32, 125), pipeline.thresholdFrames(5.0, 25));
    // Truncation toward zero, not rounding.
    try std.testing.expectEqual(@as(i32, 22), pipeline.thresholdFrames(0.9, 25));
}

test "chunkLenCodepoints: f64 math, truncation, floor 1" {
    // avg = 750/300 = 2.5 tokens/char; 15 s * 25 fps / 2.5 = 150 codepoints.
    try std.testing.expectEqual(@as(i32, 150), pipeline.chunkLenCodepoints(15.0, 25, 750, 300));
    // Tiny budget floors at 1.
    try std.testing.expectEqual(@as(i32, 1), pipeline.chunkLenCodepoints(0.01, 25, 1000, 10));
}

test "postFilter: quiet-ref rescale runs even with postproc=false" {
    const allocator = std.testing.allocator;
    var audio = try allocator.dupe(f32, &[_]f32{ 0.8, -0.4, 0.2, -0.6 });
    defer allocator.free(audio);

    try pipeline.postFilter(allocator, &audio, 24000, 0.05, false);
    // k = 0.05 / 0.1 = 0.5; length untouched (no silence removal, no pad).
    try std.testing.expectEqual(@as(usize, 4), audio.len);
    const expected = [_]f32{ 0.4, -0.2, 0.1, -0.3 };
    try std.testing.expectEqualSlices(f32, &expected, audio);
}

test "postFilter: no-ref and loud-ref leave audio untouched when postproc=false" {
    const allocator = std.testing.allocator;
    const original = [_]f32{ 0.8, -0.4, 0.2, -0.6 };

    var no_ref = try allocator.dupe(f32, &original);
    defer allocator.free(no_ref);
    try pipeline.postFilter(allocator, &no_ref, 24000, -1.0, false);
    try std.testing.expectEqualSlices(f32, &original, no_ref);

    var loud_ref = try allocator.dupe(f32, &original);
    defer allocator.free(loud_ref);
    try pipeline.postFilter(allocator, &loud_ref, 24000, 0.2, false);
    try std.testing.expectEqualSlices(f32, &original, loud_ref);
}

test "postFilter: postproc chain on a loud constant buffer (no-ref branch)" {
    const allocator = std.testing.allocator;
    // 1 s of constant full-scale audio: nothing is silent, so remove_silence
    // is a no-op; peak/0.5 halves; fade_and_pad adds 0.1 s on each side.
    var audio = try allocator.alloc(f32, 24000);
    defer allocator.free(audio);
    @memset(audio, 1.0);

    try pipeline.postFilter(allocator, &audio, 24000, -1.0, true);
    try std.testing.expectEqual(@as(usize, 24000 + 2 * 2400), audio.len);
    // Interior sample: outside both pads and fades, peak-normalized to 0.5
    // (the s16 round trip in remove_silence quantizes 1.0 -> 32767/32768,
    // which peak/0.5 then rescales to exactly 0.5).
    const interior = audio[audio.len / 2];
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), interior, 1e-6);
    // Pads are exact zeros.
    try std.testing.expectEqual(@as(f32, 0.0), audio[0]);
    try std.testing.expectEqual(@as(f32, 0.0), audio[audio.len - 1]);
}

// ---------------------------------------------------------------------------
// Shared helpers for the env-gated parity gates
// ---------------------------------------------------------------------------

fn readFileBytes(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotAFile;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}

/// Raw [K, T] i32 row-major, NO header (the reference --maskgit-test output).
fn readRawTokens(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expect_len: usize) ![]i32 {
    const bytes = try readFileBytes(allocator, io, path);
    defer allocator.free(bytes);
    if (bytes.len != expect_len * 4) return error.CorruptGolden;
    const values = try allocator.alloc(i32, expect_len);
    errdefer allocator.free(values);
    for (values, 0..) |*dst, i| {
        dst.* = std.mem.readInt(i32, bytes[i * 4 ..][0..4], .little);
    }
    return values;
}

fn expectDumpExact(allocator: std.mem.Allocator, io: std.Io, ours_path: []const u8, golden_path: []const u8) !void {
    const ours = try dump.readFile(allocator, io, ours_path);
    defer {
        allocator.free(ours.shape);
        allocator.free(ours.data);
    }
    const golden = try dump.readFile(allocator, io, golden_path);
    defer {
        allocator.free(golden.shape);
        allocator.free(golden.data);
    }
    try std.testing.expectEqualSlices(i32, golden.shape, ours.shape);
    try std.testing.expectEqualSlices(f32, golden.data, ours.data);
}

// ---------------------------------------------------------------------------
// Gate 2: greedy --maskgit-test path, byte-identical tokens (F32)
// ---------------------------------------------------------------------------

test "OMNIVOICE_PARITY: pipeline greedy generateTokens byte-identical to maskgit golden" {
    if (std.c.getenv("OMNIVOICE_PARITY") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const target_len = 75; // --duration 3 at the hardcoded 25 fps
    const num_k = 8;

    var file = fucina.gguf.File.loadMmap(allocator, io, "models/omnivoice/omnivoice-base-F32.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    var tok = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tok.deinit();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try lm.loadModel(&ctx, &file);
    defer model.deinit();

    // --maskgit-test forces both temperatures to zero (greedy, RNG-free).
    const cfg = maskgit.Config{ .class_temperature = 0.0, .position_temperature = 0.0, .seed = 42 };
    var ctr_lo: u32 = 0;
    const tokens = try pipeline.generateTokens(
        allocator,
        io,
        &ctx,
        &model,
        &tok,
        "The quick brown fox jumps over the lazy dog.",
        "English",
        "",
        target_len,
        true,
        cfg,
        "",
        null,
        0,
        null,
        &ctr_lo,
        null,
    );
    defer allocator.free(tokens);
    try std.testing.expectEqual(@as(u32, 0), ctr_lo);

    const want = try readRawTokens(allocator, io, "refs/omnivoice-research/goldens/maskgit/tokens-F32.bin", num_k * target_len);
    defer allocator.free(want);
    try std.testing.expectEqualSlices(i32, want, tokens);
}

// ---------------------------------------------------------------------------
// Gate 3: voice design end-to-end synthesize vs tts-design goldens
// ---------------------------------------------------------------------------

test "OMNIVOICE_PARITY: voice-design synthesize end-to-end vs tts-design goldens" {
    if (std.c.getenv("OMNIVOICE_PARITY") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const goldens_dir = "refs/omnivoice-research/goldens/tts-design";
    const scratch_dir = "/tmp/fucina-omnivoice-pipeline-test";

    var lm_file = fucina.gguf.File.loadMmap(allocator, io, "models/omnivoice/omnivoice-base-F32.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer lm_file.deinit();
    var codec_file = fucina.gguf.File.loadMmap(allocator, io, "models/omnivoice/omnivoice-tokenizer-F32.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer codec_file.deinit();

    std.Io.Dir.cwd().createDirPath(io, scratch_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    var tok = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &lm_file, .{});
    defer tok.deinit();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try lm.loadModel(&ctx, &lm_file);
    defer model.deinit();
    var cdc = try codec.Codec.load(&ctx, &codec_file);
    defer cdc.deinit();

    const tts = pipeline.Tts{
        .allocator = allocator,
        .io = io,
        .ctx = &ctx,
        .model = &model,
        .tok = &tok,
        .cdc = &cdc,
        .enc = null,
    };
    // The reference tts-design capture: instruct "male, young adult,
    // moderate pitch", lang English, seed 42, all other params default.
    const audio = try pipeline.synthesize(&tts, .{
        .text = "The quick brown fox jumps over the lazy dog.",
        .lang = "English",
        .instruct = "male, young adult, moderate pitch",
        .mg = .{ .seed = 42 },
        .dump_dir = scratch_dir,
    });
    defer allocator.free(audio);

    // Prompt dumps + mg-tokens: exact vs the goldens.
    try expectDumpExact(allocator, io, scratch_dir ++ "/prompt-cond-ids.bin", goldens_dir ++ "/prompt-cond-ids.bin");
    try expectDumpExact(allocator, io, scratch_dir ++ "/prompt-uncond-ids.bin", goldens_dir ++ "/prompt-uncond-ids.bin");
    try expectDumpExact(allocator, io, scratch_dir ++ "/mg-tokens.bin", goldens_dir ++ "/mg-tokens.bin");

    // Final waveform: same length (postproc is deterministic; a length
    // mismatch means a silence-trim boundary flipped) + cosine >= 0.999.
    const golden_wav = try wav.readMono(io, allocator, goldens_dir ++ "/out.wav", 24000);
    defer allocator.free(golden_wav);
    try std.testing.expectEqual(golden_wav.len, audio.len);
    const stats = dump.compare(audio, golden_wav);
    std.debug.print("omnivoice pipeline design e2e: {d} samples, wav cos={d:.7} max_abs={e:.3}\n", .{
        audio.len, stats.cosine, stats.max_abs_diff,
    });
    try std.testing.expect(stats.cosine >= 0.999);
}
