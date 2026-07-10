//! TTS waveform post-processing for the OmniVoice example.
//!
//! Port of `refs/omnivoice.cpp/src/audio-postproc.h`, itself a strict math
//! port of omnivoice/utils/audio.py and pydub.silence. All public functions
//! take and return float32 mono PCM in [-1, 1] at the pipeline sample rate
//! (24 kHz). Silence detection runs on int16 samples to match pydub
//! bit-for-bit: the f32 -> s16 -> f32 quantizing round trip inside
//! `removeSilence` is load-bearing for parity.
//!
//! Integer parameters are `i32` to mirror the reference's C `int` arithmetic
//! (`@divTrunc` = C integer division, `@intFromFloat` = C cast truncation
//! toward zero).

const std = @import("std");

/// Inclusive-start/exclusive-end sample range as pydub builds them
/// (`end` = last hit + min_silence_len for silence ranges).
pub const Range = struct { start: i32, end: i32 };

/// RMS of an int16 slice [start, start + n) clamped to s16.len. A slice that
/// extends past the end shrinks accordingly. Empty slices return 0.0,
/// matching pydub's AudioSegment.rms on empty segments.
/// (reference `postproc_slice_rms_s16`)
pub fn sliceRmsS16(s16: []const i16, start: usize, n: usize) f64 {
    var end = start + n;
    if (end > s16.len) {
        end = s16.len;
    }

    if (start >= end) {
        return 0.0;
    }

    var ssq: i64 = 0;
    for (s16[start..end]) |s| {
        const v: i64 = s;
        ssq += v * v;
    }

    const cnt = end - start;
    return @sqrt(@as(f64, @floatFromInt(ssq)) / @as(f64, @floatFromInt(cnt)));
}

/// One sample of the exact pydub f32 -> s16 recipe: multiply by 32768.0 in
/// f64, clamp to the int16 range, truncate toward zero. Shared by the
/// buffered `f32ToS16` and the streaming silence remover's incremental
/// append (`silence_remover_stream::append`), which must quantize
/// bit-identically.
pub fn f32ToS16Sample(x: f32) i16 {
    var v = @as(f64, x) * 32768.0;
    if (v > 32767.0) {
        v = 32767.0;
    }

    if (v < -32768.0) {
        v = -32768.0;
    }

    return @intFromFloat(v);
}

/// Converts float32 [-1, 1] to int16 with the exact pydub recipe:
/// (audio * 32768.0).clip(-32768, 32767).astype(int16). Truncation toward 0,
/// matching numpy's astype(int16). Caller frees.
/// (reference `postproc_f32_to_s16`)
pub fn f32ToS16(allocator: std.mem.Allocator, a: []const f32) ![]i16 {
    const out = try allocator.alloc(i16, a.len);
    for (out, a) |*dst, x| {
        dst.* = f32ToS16Sample(x);
    }

    return out;
}

/// Inverse of f32ToS16: int16 -> float32 via f64 division by 32768.0.
/// Caller frees. (reference `postproc_s16_to_f32`)
pub fn s16ToF32(allocator: std.mem.Allocator, s16: []const i16) ![]f32 {
    const out = try allocator.alloc(f32, s16.len);
    for (out, s16) |*dst, s| {
        dst.* = @floatCast(@as(f64, @floatFromInt(s)) / 32768.0);
    }

    return out;
}

/// pydub.silence.detect_silence ported to int16 samples. seek_step and
/// min_silence_len are in samples. Returns ranges [start, end] in samples
/// where end = start + min_silence_len of the last hit, exactly as pydub
/// builds them. Caller frees. (reference `postproc_detect_silence`)
pub fn detectSilence(
    allocator: std.mem.Allocator,
    s16: []const i16,
    min_silence_len: i32,
    thresh_lin: f64,
    seek_step: i32,
) ![]Range {
    const seg_len: i32 = @intCast(s16.len);

    if (seg_len < min_silence_len) {
        return try allocator.alloc(Range, 0);
    }

    const last_slice_start = seg_len - min_silence_len;

    var starts: std.ArrayList(i32) = .empty;
    defer starts.deinit(allocator);
    var i: i32 = 0;
    while (i <= last_slice_start) : (i += seek_step) {
        try starts.append(allocator, i);
    }

    if (@rem(last_slice_start, seek_step) != 0) {
        try starts.append(allocator, last_slice_start);
    }

    var silence_starts: std.ArrayList(i32) = .empty;
    defer silence_starts.deinit(allocator);
    for (starts.items) |s| {
        const r = sliceRmsS16(s16, @intCast(s), @intCast(min_silence_len));
        if (r <= thresh_lin) {
            try silence_starts.append(allocator, s);
        }
    }

    var ranges: std.ArrayList(Range) = .empty;
    errdefer ranges.deinit(allocator);

    if (silence_starts.items.len == 0) {
        return try ranges.toOwnedSlice(allocator);
    }

    var prev_i = silence_starts.items[0];
    var range_start = prev_i;

    for (silence_starts.items[1..]) |si| {
        const continuous = si == prev_i + seek_step;
        const has_gap = si > prev_i + min_silence_len;

        if (!continuous and has_gap) {
            try ranges.append(allocator, .{ .start = range_start, .end = prev_i + min_silence_len });
            range_start = si;
        }

        prev_i = si;
    }

    try ranges.append(allocator, .{ .start = range_start, .end = prev_i + min_silence_len });
    return try ranges.toOwnedSlice(allocator);
}

/// pydub.silence.detect_nonsilent: invert detectSilence over [0, seg_len].
/// Caller frees. (reference `postproc_detect_nonsilent`)
pub fn detectNonsilent(
    allocator: std.mem.Allocator,
    s16: []const i16,
    min_silence_len: i32,
    thresh_lin: f64,
    seek_step: i32,
) ![]Range {
    const seg_len: i32 = @intCast(s16.len);
    const silent = try detectSilence(allocator, s16, min_silence_len, thresh_lin, seek_step);
    defer allocator.free(silent);

    var nonsilent: std.ArrayList(Range) = .empty;
    errdefer nonsilent.deinit(allocator);

    if (silent.len == 0) {
        try nonsilent.append(allocator, .{ .start = 0, .end = seg_len });
        return try nonsilent.toOwnedSlice(allocator);
    }

    if (silent[0].start == 0 and silent[0].end == seg_len) {
        return try nonsilent.toOwnedSlice(allocator);
    }

    var prev_end: i32 = 0;
    var last_end: i32 = 0;

    for (silent) |r| {
        try nonsilent.append(allocator, .{ .start = prev_end, .end = r.start });
        prev_end = r.end;
        last_end = r.end;
    }

    if (last_end != seg_len) {
        try nonsilent.append(allocator, .{ .start = prev_end, .end = seg_len });
    }

    if (nonsilent.items.len != 0 and nonsilent.items[0].start == 0 and nonsilent.items[0].end == 0) {
        _ = nonsilent.orderedRemove(0);
    }

    return try nonsilent.toOwnedSlice(allocator);
}

/// pydub.silence.detect_leading_silence ported to int16. chunk_n is in
/// samples. Returns the sample index where the leading silence ends
/// (clamped to len). pydub compares dBFS < threshold; in linear amplitude
/// that is r < thresh_lin (strict), since dBFS is monotonic in r and r=0
/// gives -inf which is always below any finite threshold.
/// (reference `postproc_detect_leading_silence`)
pub fn detectLeadingSilence(s16: []const i16, thresh_lin: f64, chunk_n: i32) i32 {
    var trim: i32 = 0;
    const seg_len: i32 = @intCast(s16.len);

    while (trim < seg_len) {
        const slice_end = @min(trim + chunk_n, seg_len);
        const n = slice_end - trim;
        const r = sliceRmsS16(s16, @intCast(trim), @intCast(n));

        if (r >= thresh_lin) {
            break;
        }

        trim += chunk_n;
    }

    if (trim > seg_len) {
        trim = seg_len;
    }

    return trim;
}

/// removeSilence: strict 1:1 port of omnivoice/utils/audio.py:remove_silence
/// (reference `remove_silence`). Removes mid silences longer than mid_sil_ms
/// (kept down to mid_sil_ms via pydub split_on_silence with keep_silence ==
/// mid_sil_ms), then trims the leading and trailing silences leaving
/// lead_sil_ms / trail_sil_ms intact. thresh_db is the dBFS threshold
/// (default -50 dBFS in upstream).
///
/// `a.*` must be owned by `allocator`; on return it may have been freed and
/// replaced by a shorter buffer.
///
/// DEVIATION: the reference computes thresh_lin with libm pow; Zig's
/// std.math.pow(f64, 10, -2.5) lands 2 ulps below Apple libm. The comparison
/// flips only if a slice RMS falls within those 2 ulps of the threshold —
/// slice RMS is sqrt(int/int), so the golden tests (which straddle the
/// threshold by orders of magnitude) remain bit-exact.
pub fn removeSilence(
    allocator: std.mem.Allocator,
    a: *[]f32,
    sr: i32,
    mid_sil_ms: i32,
    lead_sil_ms: i32,
    trail_sil_ms: i32,
    thresh_db: f64,
) !void {
    if (a.len == 0) {
        return;
    }

    var s16_owned = try f32ToS16(allocator, a.*);
    defer allocator.free(s16_owned);
    var s16: []i16 = s16_owned;

    const thresh_lin = 32768.0 * std.math.pow(f64, 10.0, thresh_db / 20.0);
    const seek_step = @divTrunc(sr, 100); // 10 ms

    // Mid silence removal via split_on_silence + concat.
    if (mid_sil_ms > 0) {
        const min_sil_n = @divTrunc(sr * mid_sil_ms, 1000);
        const keep_n = min_sil_n;

        const nonsilent = try detectNonsilent(allocator, s16, min_sil_n, thresh_lin, seek_step);
        defer allocator.free(nonsilent);

        const output_ranges = try allocator.alloc(Range, nonsilent.len);
        defer allocator.free(output_ranges);
        for (output_ranges, nonsilent) |*dst, r| {
            dst.* = .{ .start = r.start - keep_n, .end = r.end + keep_n };
        }

        // pydub pairwise overlap dedup: split overlap at the midpoint.
        var i: usize = 0;
        while (i + 1 < output_ranges.len) : (i += 1) {
            const last_end = output_ranges[i].end;
            const next_start = output_ranges[i + 1].start;
            if (next_start < last_end) {
                const mid = @divTrunc(last_end + next_start, 2);
                output_ranges[i].end = mid;
                output_ranges[i + 1].start = mid;
            }
        }

        // Concat clipped slices. Empty slices contribute nothing, matching
        // AudioSegment.silent(0) += seg semantics.
        var out: std.ArrayList(i16) = .empty;
        defer out.deinit(allocator);
        try out.ensureTotalCapacity(allocator, s16.len);
        const seg_len: i32 = @intCast(s16.len);

        for (output_ranges) |r| {
            const cs = @max(0, r.start);
            const ce = @min(seg_len, r.end);
            if (cs < ce) {
                try out.appendSlice(allocator, s16[@intCast(cs)..@intCast(ce)]);
            }
        }

        const new_owned = try out.toOwnedSlice(allocator);
        allocator.free(s16_owned);
        s16_owned = new_owned;
        s16 = s16_owned;
    }

    // Edge trimming: leading then trailing via reverse trick.
    const chunk_n = @divTrunc(sr, 100); // 10 ms

    var trim_lead = detectLeadingSilence(s16, thresh_lin, chunk_n);
    trim_lead = @max(0, trim_lead - @divTrunc(sr * lead_sil_ms, 1000));
    if (trim_lead > 0) {
        s16 = s16[@intCast(@min(trim_lead, @as(i32, @intCast(s16.len))))..];
    }

    std.mem.reverse(i16, s16);

    var trim_trail = detectLeadingSilence(s16, thresh_lin, chunk_n);
    trim_trail = @max(0, trim_trail - @divTrunc(sr * trail_sil_ms, 1000));
    if (trim_trail > 0) {
        s16 = s16[@intCast(@min(trim_trail, @as(i32, @intCast(s16.len))))..];
    }

    std.mem.reverse(i16, s16);

    const trimmed = try s16ToF32(allocator, s16);
    allocator.free(a.*);
    a.* = trimmed;
}

/// refPreprocessAudio: reference waveform preprocessing shared by the TTS
/// reference path and the codec CLI encode path (reference
/// `ref_preprocess_audio`). Mirrors the upstream Python chain: RMS
/// measurement (f64 accumulate), auto-gain to RMS 0.1 when the original RMS
/// sits in (0, 0.1) — UNCONDITIONAL, even when trim_silence is false — then
/// silence-trim with mid=200ms / lead=100ms / trail=200ms at -50 dBFS when
/// trim_silence is set. Returns the ORIGINAL pre-gain RMS so the TTS
/// post-proc can rescale generated audio back to the reference loudness; an
/// empty buffer returns -1.
///
/// `a.*` must be owned by `allocator`; the trim may free and replace it.
pub fn refPreprocessAudio(allocator: std.mem.Allocator, a: *[]f32, sr: i32, trim_silence: bool) !f32 {
    if (a.len == 0) {
        return -1.0;
    }

    var sumsq: f64 = 0.0;
    for (a.*) |v| {
        sumsq += @as(f64, v) * @as(f64, v);
    }
    const ref_rms = @sqrt(sumsq / @as(f64, @floatFromInt(a.len)));

    if (ref_rms > 0.0 and ref_rms < 0.1) {
        const gain: f32 = @floatCast(0.1 / ref_rms);
        for (a.*) |*v| {
            v.* *= gain;
        }
    }

    if (trim_silence) {
        try removeSilence(allocator, a, sr, 200, 100, 200, -50.0);
    }

    return @floatCast(ref_rms);
}

/// peakNormalizeHalf: rescale so peak amplitude becomes 0.5 (-6 dBFS).
/// Mirrors the no-ref branch of _post_process_audio in omnivoice.py.
/// (reference `peak_normalize_half`)
pub fn peakNormalizeHalf(a: []f32) void {
    if (a.len == 0) {
        return;
    }

    var peak: f32 = 0.0;
    for (a) |s| {
        const v = @abs(s);
        if (v > peak) {
            peak = v;
        }
    }

    if (peak > 1e-6) {
        const k: f32 = 0.5 / peak;
        for (a) |*s| {
            s.* *= k;
        }
    }
}

/// fadeAndPad: linear fade-in / fade-out on the first and last fade_dur
/// seconds, then pad pad_dur seconds of silence on each side. 1:1 port of
/// fade_and_pad_audio in omnivoice/utils/audio.py (reference `fade_and_pad`).
///
/// `a.*` must be owned by `allocator`; padding frees and replaces it.
pub fn fadeAndPad(allocator: std.mem.Allocator, a: *[]f32, sr: i32, fade_dur: f64, pad_dur: f64) !void {
    if (a.len == 0) {
        return;
    }

    const fade_n: i32 = @intFromFloat(fade_dur * @as(f64, @floatFromInt(sr)));
    const pad_n: i32 = @intFromFloat(pad_dur * @as(f64, @floatFromInt(sr)));

    if (fade_n > 0) {
        const k = @min(fade_n, @divTrunc(@as(i32, @intCast(a.len)), 2));
        if (k > 0) {
            const denom: f32 = @floatFromInt(@max(k - 1, 1));

            var i: i32 = 0;
            while (i < k) : (i += 1) {
                const w = @as(f32, @floatFromInt(i)) / denom;
                a.*[@intCast(i)] *= w;
            }

            i = 0;
            while (i < k) : (i += 1) {
                const w = 1.0 - @as(f32, @floatFromInt(i)) / denom;
                a.*[a.len - @as(usize, @intCast(k)) + @as(usize, @intCast(i))] *= w;
            }
        }
    }

    if (pad_n > 0) {
        const pn: usize = @intCast(pad_n);
        const padded = try allocator.alloc(f32, pn + a.len + pn);
        @memset(padded, 0.0);
        @memcpy(padded[pn..][0..a.len], a.*);
        allocator.free(a.*);
        a.* = padded;
    }
}

/// crossFadeChunks: concatenate audio chunks with a silence_dur gap split
/// into fade_out, pure silence, fade_in. The fade-out endpoint hits exactly
/// 0. 1:1 port of cross_fade_chunks in omnivoice/utils/audio.py (reference
/// `cross_fade_chunks`). Caller frees the returned buffer.
pub fn crossFadeChunks(
    allocator: std.mem.Allocator,
    chunks: []const []const f32,
    sr: i32,
    silence_dur: f64,
) ![]f32 {
    if (chunks.len == 0) {
        return try allocator.alloc(f32, 0);
    }

    if (chunks.len == 1) {
        return try allocator.dupe(f32, chunks[0]);
    }

    const total_n: i32 = @intFromFloat(silence_dur * @as(f64, @floatFromInt(sr)));
    const fade_n = @divTrunc(total_n, 3);
    const silence_n = fade_n;

    var merged: std.ArrayList(f32) = .empty;
    defer merged.deinit(allocator);
    try merged.appendSlice(allocator, chunks[0]);

    for (chunks[1..]) |chunk| {
        // Fade-out tail of merged.
        const fout_n = @min(fade_n, @as(i32, @intCast(merged.items.len)));
        if (fout_n > 0) {
            const denom: f32 = @floatFromInt(@max(fout_n - 1, 1));
            const tail = merged.items[merged.items.len - @as(usize, @intCast(fout_n)) ..];
            for (tail, 0..) |*s, j| {
                const w = 1.0 - @as(f32, @floatFromInt(@as(i32, @intCast(j)))) / denom;
                s.* *= w;
            }
        }

        // Silence gap.
        if (silence_n > 0) {
            var j: i32 = 0;
            while (j < silence_n) : (j += 1) {
                try merged.append(allocator, 0.0);
            }
        }

        // Fade-in head of the appended chunk (the reference fades a copy of
        // the chunk then appends; fading in place after the append is the
        // same arithmetic).
        const head_start = merged.items.len;
        try merged.appendSlice(allocator, chunk);
        const fin_n = @min(fade_n, @as(i32, @intCast(chunk.len)));
        if (fin_n > 0) {
            const denom: f32 = @floatFromInt(@max(fin_n - 1, 1));
            const head = merged.items[head_start..][0..@intCast(fin_n)];
            for (head, 0..) |*s, j| {
                const w = @as(f32, @floatFromInt(@as(i32, @intCast(j)))) / denom;
                s.* *= w;
            }
        }
    }

    return try merged.toOwnedSlice(allocator);
}

test {
    _ = @import("postproc_tests.zig");
}
