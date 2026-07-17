//! Cross-module smoke tests for the OmniVoice example root, plus the
//! env-gated end-to-end STREAMING parity gates (vs the reference captures
//! under refs/omnivoice-research/goldens/tts-stream/, produced by the
//! reference CPU build with base-F32 + tokenizer-F32, seed 42):
//!
//!   OMNIVOICE_PARITY=1 zig build test -Doptimize=ReleaseSafe

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const chunker = @import("chunker.zig");
const chunker_stream = @import("chunker_stream.zig");
const codec = @import("codec.zig");
const dump = @import("dump.zig");
const lm = @import("lm.zig");
const pipeline = @import("pipeline.zig");
const wav = @import("wav.zig");

test {
    _ = @import("lm.zig");
    _ = @import("prompt.zig");
    _ = @import("mg_decode.zig");
}

test "placeholder: stage A modules register their own sibling tests" {
    // Each module (philox, resample, wav, dump, rvq_file) carries its tests in
    // its own <name>_tests.zig, imported from the production file per house
    // convention. This file exists for cross-module integration tests added in
    // later stages.
    try std.testing.expect(true);
}

// ---------------------------------------------------------------------------
// Streaming end-to-end parity gates
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

fn wavSinkEmit(ctx: *anyopaque, samples: []const f32) anyerror!void {
    const sink: *wav.StreamSink = @ptrCast(@alignCast(ctx));
    try sink.writeSamples(samples);
}

/// In-process mirror of the CLI streaming session (examples/omnivoice/main.zig
/// runTtsStream): incremental chunker at chunk_len (int)((24000/480)*5) = 250
/// codepoints, one full synthesizeStream per ready chunk (fresh Philox
/// counter per call, seed 42), --chunk-threshold 5 --chunk-duration 5,
/// wav16 stream sink. `by_line` drains the chunker at every newline and
/// re-arms a fresh RIFF header for the next line's audio.
fn runStreamSession(
    allocator: std.mem.Allocator,
    tts: *const pipeline.Tts,
    input: []const u8,
    by_line: bool,
    out: *std.Io.Writer,
) !void {
    var sink = wav.StreamSink{ .writer = out, .sample_rate = 24000, .format = .s16 };
    try sink.writeHeader();

    var chk = chunker_stream.Stream.init(allocator, (24000 / 480) * 5, chunker.min_chunk_len_default);
    defer chk.deinit();

    var need_header = false;

    const Synth = struct {
        fn one(tts_: *const pipeline.Tts, sink_: *wav.StreamSink, need_header_: *bool, chunk_text: []const u8) !void {
            if (need_header_.*) {
                try sink_.writeHeader();
                need_header_.* = false;
            }
            try pipeline.synthesizeStream(tts_, .{
                .text = chunk_text,
                .lang = "English",
                .chunk_duration_sec = 5.0,
                .chunk_threshold_sec = 5.0,
                .mg = .{ .seed = 42 },
            }, .{ .ctx = sink_, .func = wavSinkEmit });
        }
    };

    var pos: usize = 0;
    while (pos < input.len) {
        var end = input.len;
        var flush_line = false;
        if (by_line) {
            if (std.mem.indexOfScalarPos(u8, input, pos, '\n')) |nl| {
                end = nl + 1;
                flush_line = true;
            }
        } else {
            end = @min(pos + 4096, input.len);
        }

        const ready = try chk.pushBytes(input[pos..end]);
        defer chunker.freeChunks(allocator, ready);
        for (ready) |ct| try Synth.one(tts, &sink, &need_header, ct);

        if (flush_line) {
            const line_tail = try chk.flushEof();
            defer chunker.freeChunks(allocator, line_tail);
            for (line_tail) |ct| try Synth.one(tts, &sink, &need_header, ct);
            need_header = true;
        }
        pos = end;
    }

    const tail = try chk.flushEof();
    defer chunker.freeChunks(allocator, tail);
    for (tail) |ct| try Synth.one(tts, &sink, &need_header, ct);
    try out.flush();
}

/// Offsets of every RIFF header in a live byte stream (RIFF....WAVEfmt
/// signature; plain "RIFF" could collide with audio payload bytes).
fn riffOffsets(allocator: std.mem.Allocator, bytes: []const u8) ![]usize {
    var offsets: std.ArrayList(usize) = .empty;
    defer offsets.deinit(allocator);
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, bytes, pos, "RIFF")) |i| {
        if (i + 16 <= bytes.len and std.mem.eql(u8, bytes[i + 8 ..][0..8], "WAVEfmt ")) {
            try offsets.append(allocator, i);
        }
        pos = i + 4;
    }
    return offsets.toOwnedSlice(allocator);
}

/// Decodes the s16 payload of one live-header WAV segment to f32.
fn decodeSegment(allocator: std.mem.Allocator, segment: []const u8) ![]f32 {
    var audio = try wav.parse(allocator, segment);
    defer audio.deinit();
    if (audio.channels.len != 1) return error.CorruptGolden;
    return allocator.dupe(f32, audio.channels[0]);
}

test "OMNIVOICE_PARITY: streaming e2e (-o -) vs tts-stream golden: header bytes, sample count, cosine" {
    if (std.c.getenv("OMNIVOICE_PARITY") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const goldens_dir = "refs/omnivoice-research/goldens/tts-stream";

    const golden = readFileBytes(allocator, io, goldens_dir ++ "/ref_stream.wav") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(golden);
    const text_raw = try readFileBytes(allocator, io, goldens_dir ++ "/text.txt");
    defer allocator.free(text_raw);

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

    var aw = std.Io.Writer.Allocating.init(allocator);
    defer aw.deinit();
    try runStreamSession(allocator, &tts, text_raw, false, &aw.writer);
    const ours = aw.written();

    // Identical 44-byte live header (incl. the 0x7FFFFFFF sizes).
    try std.testing.expect(golden.len > 44 and ours.len > 44);
    try std.testing.expectEqualSlices(u8, golden[0..44], ours[0..44]);

    // Same payload sample count, cosine >= 0.999 (tokens are byte-exact at
    // F32 + seed; only codec f32 noise differs).
    const golden_audio = try decodeSegment(allocator, golden);
    defer allocator.free(golden_audio);
    const our_audio = try decodeSegment(allocator, ours);
    defer allocator.free(our_audio);
    try std.testing.expectEqual(golden_audio.len, our_audio.len);
    const stats = dump.compare(our_audio, golden_audio);
    std.debug.print("omnivoice stream e2e: {d} samples, cos={d:.7} max_abs={e:.3}\n", .{
        our_audio.len, stats.cosine, stats.max_abs_diff,
    });
    try std.testing.expect(stats.cosine >= 0.999);
}

test "OMNIVOICE_PARITY: --stream-by-line vs tts-stream golden: RIFF offsets + per-segment cosine" {
    if (std.c.getenv("OMNIVOICE_PARITY") == null) return error.SkipZigTest;

    // NOTE on the lines.txt input: the seeded (pos_temp 5.0) MaskGIT chain
    // is only cross-implementation token-exact when no CFG/top-k boundary
    // sits within f32-reassociation distance of a tie; that is
    // text-dependent. The first line-1 candidate ("Hello there, this is
    // line one. It has two sentences.") flips tokens between the reference
    // and this port EVEN IN BUFFERED MODE (verified 2026-07-02: mg-tokens
    // differ, audio cos 0.987), so the capture uses a token-stable pair
    // (verified token-identical buffered) to isolate the STREAMING
    // machinery, which is what this gate is about.

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const goldens_dir = "refs/omnivoice-research/goldens/tts-stream";

    const golden = readFileBytes(allocator, io, goldens_dir ++ "/ref_stream_lines.wav") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(golden);
    const lines_raw = try readFileBytes(allocator, io, goldens_dir ++ "/lines.txt");
    defer allocator.free(lines_raw);

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

    var aw = std.Io.Writer.Allocating.init(allocator);
    defer aw.deinit();
    try runStreamSession(allocator, &tts, lines_raw, true, &aw.writer);
    const ours = aw.written();

    // Two RIFF headers at identical byte offsets (one per line; the second
    // segment's offset equality implies byte-equal first-payload length).
    const golden_offsets = try riffOffsets(allocator, golden);
    defer allocator.free(golden_offsets);
    const our_offsets = try riffOffsets(allocator, ours);
    defer allocator.free(our_offsets);
    try std.testing.expectEqual(@as(usize, 2), golden_offsets.len);
    try std.testing.expectEqualSlices(usize, golden_offsets, our_offsets);
    try std.testing.expectEqual(golden.len, ours.len);

    // Per-segment payload cosine >= 0.999.
    for (0..2) |seg_i| {
        const g_start = golden_offsets[seg_i];
        const g_end = if (seg_i + 1 < golden_offsets.len) golden_offsets[seg_i + 1] else golden.len;
        const golden_audio = try decodeSegment(allocator, golden[g_start..g_end]);
        defer allocator.free(golden_audio);
        const our_audio = try decodeSegment(allocator, ours[g_start..g_end]);
        defer allocator.free(our_audio);
        try std.testing.expectEqual(golden_audio.len, our_audio.len);
        const stats = dump.compare(our_audio, golden_audio);
        std.debug.print("omnivoice stream-by-line seg {d}: {d} samples, cos={d:.7}\n", .{ seg_i, our_audio.len, stats.cosine });
        try std.testing.expect(stats.cosine >= 0.999);
    }
}
