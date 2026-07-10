//! Tests for postproc_stream.zig — streaming post-proc stages vs their
//! buffered equivalents in postproc.zig (parity gate 1 of the streaming
//! task). The reference's own equivalence claims (audio-postproc-stream.h
//! header comments) are verified sample-exactly on synthetic signals shaped
//! like real OmniVoice output; the documented divergences (FadePad between
//! fade_n and 2*fade_n, voice-design peak/0.5 skip) are asserted as such.

const std = @import("std");

const postproc = @import("postproc.zig");
const postproc_stream = @import("postproc_stream.zig");

const Emit = postproc_stream.Emit;

/// Collects everything a stage emits.
const Collector = struct {
    allocator: std.mem.Allocator,
    samples: std.ArrayList(f32),

    fn init(allocator: std.mem.Allocator) Collector {
        return .{ .allocator = allocator, .samples = .empty };
    }

    fn deinit(self: *Collector) void {
        self.samples.deinit(self.allocator);
    }

    fn emitFn(ctx: *anyopaque, samples: []const f32) anyerror!void {
        const self: *Collector = @ptrCast(@alignCast(ctx));
        try std.testing.expect(samples.len > 0); // stages never emit empty
        try self.samples.appendSlice(self.allocator, samples);
    }

    fn emit(self: *Collector) Emit {
        return .{ .ctx = self, .func = emitFn };
    }
};

/// Deterministic "speech-like" fill: bounded pseudo-random values well above
/// the -50 dBFS silence threshold, no exact zeros (so index bugs cannot hide
/// behind zero samples).
fn fillSpeech(samples: []f32, seed: u64) void {
    var state = seed | 1;
    for (samples) |*s| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const r: f32 = @floatFromInt(@as(u16, @truncate(state >> 33)));
        s.* = 0.05 + 0.4 * (r / 65535.0); // in [0.05, 0.45]
    }
}

/// Quiet noise below the silence threshold (s16 magnitude <= 3, RMS far
/// below thresh_lin ~= 103.6), but nonzero.
fn fillQuiet(samples: []f32, seed: u64) void {
    var state = seed | 1;
    for (samples) |*s| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const r: f32 = @floatFromInt(@as(u16, @truncate(state >> 40)) & 0x7);
        s.* = (r - 3.5) * 2.0e-5; // |s16| <= 3
    }
}

// ---------------------------------------------------------------------------
// Crossfader vs crossFadeChunks
// ---------------------------------------------------------------------------

test "Crossfader over 3 chunks is sample-exact vs crossFadeChunks" {
    const allocator = std.testing.allocator;
    const sr = 1000; // fade_n = silence_n = (int)(0.3*1000)/3 = 100

    var c0: [450]f32 = undefined;
    var c1: [377]f32 = undefined;
    var c2: [512]f32 = undefined;
    fillSpeech(&c0, 11);
    fillSpeech(&c1, 22);
    fillSpeech(&c2, 33);
    const chunks = [_][]const f32{ &c0, &c1, &c2 };

    const want = try postproc.crossFadeChunks(allocator, &chunks, sr, 0.3);
    defer allocator.free(want);

    var got = Collector.init(allocator);
    defer got.deinit();
    var cf = postproc_stream.Crossfader.init(allocator, sr, 0.3);
    defer cf.deinit();
    for (&chunks) |c| try cf.push(c, got.emit());
    try cf.flush(got.emit());

    try std.testing.expectEqualSlices(f32, want, got.samples.items);
}

test "Crossfader single chunk passes through verbatim" {
    const allocator = std.testing.allocator;
    var c0: [450]f32 = undefined;
    fillSpeech(&c0, 5);

    var got = Collector.init(allocator);
    defer got.deinit();
    var cf = postproc_stream.Crossfader.init(allocator, 1000, 0.3);
    defer cf.deinit();
    try cf.push(&c0, got.emit());
    try cf.flush(got.emit());

    try std.testing.expectEqualSlices(f32, &c0, got.samples.items);
}

// ---------------------------------------------------------------------------
// SilenceRemover vs removeSilence
// ---------------------------------------------------------------------------

fn expectSilenceRemoverMatchesBatch(signal: []const f32, push_sizes: []const usize) !void {
    const allocator = std.testing.allocator;
    const sr = 1000; // min_sil_n = keep_n = 500, seek_step = chunk_n = 10

    var want = try allocator.dupe(f32, signal);
    defer allocator.free(want);
    try postproc.removeSilence(allocator, &want, sr, 500, 100, 100, -50.0);

    var got = Collector.init(allocator);
    defer got.deinit();
    var srm = postproc_stream.SilenceRemover.init(allocator, sr, 500, 100, 100, -50.0);
    defer srm.deinit();

    var pos: usize = 0;
    var k: usize = 0;
    while (pos < signal.len) {
        const n = @min(push_sizes[k % push_sizes.len], signal.len - pos);
        try srm.push(signal[pos..][0..n], got.emit());
        pos += n;
        k += 1;
    }
    try srm.flush(got.emit());

    // removeSilence returns the s16-quantized round trip of the kept
    // samples; the streaming stage emits the original f32 samples. Compare
    // through the same quantization for exact index equivalence.
    const want_s16 = try postproc.f32ToS16(allocator, want);
    defer allocator.free(want_s16);
    const got_s16 = try postproc.f32ToS16(allocator, got.samples.items);
    defer allocator.free(got_s16);
    try std.testing.expectEqualSlices(i16, want_s16, got_s16);
}

test "SilenceRemover: speech / long mid silence / speech == removeSilence" {
    // Mid silence of 1200 samples >= 2*keep_n = 1000: the middle drops and
    // keep_n stays on each side (the split_on_silence drop branch).
    var signal: [800 + 1200 + 700]f32 = undefined;
    fillSpeech(signal[0..800], 1);
    fillQuiet(signal[800..2000], 2);
    fillSpeech(signal[2000..], 3);
    try expectSilenceRemoverMatchesBatch(&signal, &.{ 257, 100, 1024 });
}

test "SilenceRemover: short mid silence keeps everything (midpoint split)" {
    // Mid silence of 600 samples: 600 < 2*keep_n=1000 -> pydub keeps all
    // samples, splitting the overlap at the midpoint (concat-identical).
    var signal: [900 + 600 + 800]f32 = undefined;
    fillSpeech(signal[0..900], 4);
    fillQuiet(signal[900..1500], 5);
    fillSpeech(signal[1500..], 6);
    try expectSilenceRemoverMatchesBatch(&signal, &.{ 64, 333 });
}

test "SilenceRemover: leading and trailing silence trim == removeSilence" {
    var signal: [700 + 1500 + 900]f32 = undefined;
    fillQuiet(signal[0..700], 7);
    fillSpeech(signal[700..2200], 8);
    fillQuiet(signal[2200..], 9);
    try expectSilenceRemoverMatchesBatch(&signal, &.{ 411, 89, 1000 });
}

test "SilenceRemover: no silence at all is a pass-through == removeSilence" {
    var signal: [2100]f32 = undefined;
    fillSpeech(&signal, 10);
    try expectSilenceRemoverMatchesBatch(&signal, &.{523});
}

test "SilenceRemover: all-silent input follows the STREAMING reference (documented divergence)" {
    // The reference streaming flush does not special-case a never-started
    // lead phase: the whole buffer is trail-trimmed as one suffix, leaving
    // trail_keep samples (100 at sr=1000). The batch removeSilence returns
    // EMPTY for all-silent input (detect_nonsilent finds nothing). Never
    // hit in practice — synthesized chunks are never fully silent.
    const allocator = std.testing.allocator;
    var signal: [1500]f32 = undefined;
    fillQuiet(&signal, 11);

    var got = Collector.init(allocator);
    defer got.deinit();
    var srm = postproc_stream.SilenceRemover.init(allocator, 1000, 500, 100, 100, -50.0);
    defer srm.deinit();
    try srm.push(&signal, got.emit());
    try srm.flush(got.emit());
    try std.testing.expectEqualSlices(f32, signal[0..100], got.samples.items);

    var batch = try allocator.dupe(f32, &signal);
    defer allocator.free(batch);
    try postproc.removeSilence(allocator, &batch, 1000, 500, 100, 100, -50.0);
    try std.testing.expectEqual(@as(usize, 0), batch.len);
}

test "SilenceRemover: push-size invariance (byte-split independence)" {
    const allocator = std.testing.allocator;
    var signal: [600 + 1300 + 500]f32 = undefined;
    fillSpeech(signal[0..600], 12);
    fillQuiet(signal[600..1900], 13);
    fillSpeech(signal[1900..], 14);

    var whole = Collector.init(allocator);
    defer whole.deinit();
    {
        var srm = postproc_stream.SilenceRemover.init(allocator, 1000, 500, 100, 100, -50.0);
        defer srm.deinit();
        try srm.push(&signal, whole.emit());
        try srm.flush(whole.emit());
    }

    var split = Collector.init(allocator);
    defer split.deinit();
    {
        var srm = postproc_stream.SilenceRemover.init(allocator, 1000, 500, 100, 100, -50.0);
        defer srm.deinit();
        var pos: usize = 0;
        while (pos < signal.len) {
            const n = @min(@as(usize, 173), signal.len - pos);
            try srm.push(signal[pos..][0..n], split.emit());
            pos += n;
        }
        try srm.flush(split.emit());
    }

    try std.testing.expectEqualSlices(f32, whole.samples.items, split.samples.items);
}

// ---------------------------------------------------------------------------
// FadePad vs fadeAndPad
// ---------------------------------------------------------------------------

fn expectFadePadMatchesBatch(len: usize, push_size: usize) !void {
    const allocator = std.testing.allocator;
    const sr = 1000; // fade_n = pad_n = 100

    const signal = try allocator.alloc(f32, len);
    defer allocator.free(signal);
    fillSpeech(signal, 21);

    var want = try allocator.dupe(f32, signal);
    defer allocator.free(want);
    try postproc.fadeAndPad(allocator, &want, sr, 0.1, 0.1);

    var got = Collector.init(allocator);
    defer got.deinit();
    var fp = postproc_stream.FadePad.init(allocator, sr, 0.1, 0.1);
    defer fp.deinit();
    var pos: usize = 0;
    while (pos < signal.len) {
        const n = @min(push_size, signal.len - pos);
        try fp.push(signal[pos..][0..n], got.emit());
        pos += n;
    }
    try fp.flush(got.emit());

    try std.testing.expectEqualSlices(f32, want, got.samples.items);
}

test "FadePad is sample-exact vs fadeAndPad for inputs >= 2*fade_n" {
    try expectFadePadMatchesBatch(200, 77); // exactly 2*fade_n
    try expectFadePadMatchesBatch(731, 64);
    try expectFadePadMatchesBatch(731, 1024); // single push
}

test "FadePad matches fadeAndPad for inputs shorter than fade_n" {
    // total < fade_n: flush applies the same degraded fade (k = total/2)
    // as the batch function.
    try expectFadePadMatchesBatch(61, 13);
    try expectFadePadMatchesBatch(1, 1);
}

test "FadePad between fade_n and 2*fade_n follows the STREAMING reference (documented divergence)" {
    // For fade_n <= len < 2*fade_n the reference streaming stage fades in
    // over fade_n and out over len - fade_n, while the batch function fades
    // both ends over len/2. Port the streaming behaviour verbatim.
    const allocator = std.testing.allocator;
    const sr = 1000;
    const fade_n = 100;
    const len = 150;

    var signal: [len]f32 = undefined;
    fillSpeech(&signal, 31);

    var got = Collector.init(allocator);
    defer got.deinit();
    var fp = postproc_stream.FadePad.init(allocator, sr, 0.1, 0.1);
    defer fp.deinit();
    try fp.push(&signal, got.emit());
    try fp.flush(got.emit());

    // Expected: pad_n zeros | fade-in over fade_n (denom fade_n-1) |
    // fade-out over len-fade_n (denom max(len-fade_n-1,1)) | pad_n zeros.
    var expected: [100 + len + 100]f32 = undefined;
    @memset(expected[0..100], 0.0);
    @memset(expected[100 + len ..], 0.0);
    for (0..fade_n) |j| {
        expected[100 + j] = signal[j] * (@as(f32, @floatFromInt(j)) / @as(f32, fade_n - 1));
    }
    const k = len - fade_n; // 50
    for (0..k) |j| {
        expected[100 + fade_n + j] = signal[fade_n + j] * (1.0 - @as(f32, @floatFromInt(j)) / @as(f32, k - 1));
    }
    try std.testing.expectEqualSlices(f32, &expected, got.samples.items);

    // ... and it genuinely diverges from the batch function there.
    var batch = try allocator.dupe(f32, &signal);
    defer allocator.free(batch);
    try postproc.fadeAndPad(allocator, &batch, sr, 0.1, 0.1);
    try std.testing.expectEqual(expected.len, batch.len);
    try std.testing.expect(!std.mem.eql(f32, &expected, batch));
}

// ---------------------------------------------------------------------------
// Full Pipeline vs the buffered chain
// ---------------------------------------------------------------------------

fn runBufferedChain(allocator: std.mem.Allocator, chunks: []const []const f32, volume_scale: f32) ![]f32 {
    var audio = try postproc.crossFadeChunks(allocator, chunks, 1000, 0.3);
    errdefer allocator.free(audio);
    try postproc.removeSilence(allocator, &audio, 1000, 500, 100, 100, -50.0);
    for (audio) |*s| s.* *= volume_scale;
    try postproc.fadeAndPad(allocator, &audio, 1000, 0.1, 0.1);
    return audio;
}

fn expectPipelineMatchesBufferedChain(volume_scale: f32) !void {
    const allocator = std.testing.allocator;

    // Three "chunks" with interior structure: speech, some mid silence in
    // chunk 1, trailing quiet in chunk 2 (all >> fade_n = 100 samples).
    var c0: [1800]f32 = undefined;
    fillSpeech(&c0, 41);
    var c1: [2600]f32 = undefined;
    fillSpeech(c1[0..800], 42);
    fillQuiet(c1[800..1900], 43);
    fillSpeech(c1[1900..], 44);
    var c2: [1500]f32 = undefined;
    fillSpeech(c2[0..1100], 45);
    fillQuiet(c2[1100..], 46);
    const chunks = [_][]const f32{ &c0, &c1, &c2 };

    const want = try runBufferedChain(allocator, &chunks, volume_scale);
    defer allocator.free(want);

    var got = Collector.init(allocator);
    defer got.deinit();
    var pipe = postproc_stream.Pipeline.init(allocator, 1000, volume_scale, got.emit());
    defer pipe.deinit();
    for (&chunks) |c| try pipe.pushChunk(c);
    try pipe.finish();

    // The buffered removeSilence output went through the f32->s16->f32
    // round trip before the scale and fades; the streaming stage keeps the
    // original f32 (like the reference: buf_f holds raw samples). Compare
    // in the s16 domain, where the fade/scale double-rounding costs at most
    // one count; interior (unfaded, unscaled) samples are exactly equal.
    const want_s16 = try postproc.f32ToS16(allocator, want);
    defer allocator.free(want_s16);
    const got_s16 = try postproc.f32ToS16(allocator, got.samples.items);
    defer allocator.free(got_s16);
    try std.testing.expectEqual(want_s16.len, got_s16.len);
    for (want_s16, got_s16) |a, b| {
        const d = @abs(@as(i32, a) - @as(i32, b));
        try std.testing.expect(d <= 1);
    }
}

test "Pipeline (cf -> sr -> fp) matches the buffered chain, scale = 1" {
    try expectPipelineMatchesBufferedChain(1.0);
}

test "Pipeline volume-scale branch matches the buffered chain within s16 rounding" {
    try expectPipelineMatchesBufferedChain(0.53);
}
