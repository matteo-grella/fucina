//! WAV parse/encode for the OmniVoice example.
//!
//! Reader: port of `refs/omnivoice.cpp/src/wav.h` (`read_wav_buf`) — classic
//! RIFF and WAVE_FORMAT_EXTENSIBLE (tag 0xFFFE, subformat at fmt offset +24),
//! payloads PCM16 (s/32768.0f), PCM24 (sign-extended s24le / 8388608.0f) and
//! IEEE float32 (verbatim); mono or stereo; the chunk walk skips unknown
//! chunks, honors the odd-size pad byte and clips a lying chunk size to the
//! file size.
//! Writer: port of the writer half of `refs/omnivoice.cpp/src/audio-io.h`
//! (`audio_encode_wav_{s16,s24,f32}` + `audio_parse_format`) — mono only,
//! 44-byte classic header, NaN/Inf sanitized to 0, S16/S24 clamp then
//! truncate toward zero exactly like the reference's C casts, F32 raw bits.
//! `readMono` mirrors the reference `audio_read_mono` orchestration:
//! per-channel resample to the target rate FIRST, then the `0.5*(L+R)`
//! downmix (mono sources passthrough-resample, bit-identical to the
//! reference's duplicate-then-downmix).

const std = @import("std");

const resample = @import("resample.zig");

pub const Error = error{
    NotWav,
    UnsupportedWavFormat,
    NoAudioData,
};

/// Decoded WAV payload as planar channels.
///
/// Mono sources yield ONE channel. The reference duplicates mono into L/R
/// and later downmixes 0.5*(L+R), which is bit-identical to using the single
/// channel directly, so no information is lost by keeping it planar-mono.
/// Multi-channel sources yield two channels (the first two of each frame,
/// exactly as the reference reads them).
pub const Audio = struct {
    sample_rate: u32,
    channels: [][]f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Audio) void {
        for (self.channels) |ch| self.allocator.free(ch);
        self.allocator.free(self.channels);
        self.* = undefined;
    }

    pub fn frames(self: *const Audio) usize {
        return if (self.channels.len == 0) 0 else self.channels[0].len;
    }
};

fn readS24le(p: *const [3]u8) i32 {
    var u: u32 = @as(u32, p[0]) | (@as(u32, p[1]) << 8) | (@as(u32, p[2]) << 16);
    if (u & 0x00800000 != 0) u |= 0xFF000000;
    return @bitCast(u);
}

/// Parses a whole WAV image from memory. Mirrors `read_wav_buf`: the walk
/// stops at the first data chunk seen after a valid fmt chunk; unsupported
/// format/bit combinations error only when that data chunk is reached.
pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Audio {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) {
        return Error.NotWav;
    }

    var n_channels: u16 = 0;
    var sample_rate: u32 = 0;
    var bits_per_sample: u16 = 0;
    var audio_format: u16 = 0;

    var pos: usize = 12;
    while (pos + 8 <= bytes.len) {
        const chunk_id = bytes[pos .. pos + 4];
        var chunk_size: usize = std.mem.readInt(u32, bytes[pos + 4 ..][0..4], .little);
        pos += 8;
        if (chunk_size > bytes.len - pos) chunk_size = bytes.len - pos;

        if (std.mem.eql(u8, chunk_id, "fmt ") and chunk_size >= 16) {
            audio_format = std.mem.readInt(u16, bytes[pos..][0..2], .little);
            n_channels = std.mem.readInt(u16, bytes[pos + 2 ..][0..2], .little);
            sample_rate = std.mem.readInt(u32, bytes[pos + 4 ..][0..4], .little);
            bits_per_sample = std.mem.readInt(u16, bytes[pos + 14 ..][0..2], .little);

            if (audio_format == 0xFFFE and chunk_size >= 40) {
                const subformat = std.mem.readInt(u16, bytes[pos + 24 ..][0..2], .little);
                if (subformat == 1 or subformat == 3) audio_format = subformat;
            }

            pos += chunk_size;
        } else if (std.mem.eql(u8, chunk_id, "data") and n_channels > 0) {
            const channels = try decodeData(allocator, bytes[pos .. pos + chunk_size], audio_format, bits_per_sample, n_channels);
            return .{
                .sample_rate = sample_rate,
                .channels = channels,
                .allocator = allocator,
            };
        } else {
            pos += chunk_size;
        }

        if (chunk_size & 1 != 0) pos += 1;
    }

    return Error.NoAudioData;
}

fn decodeData(allocator: std.mem.Allocator, payload: []const u8, audio_format: u16, bits_per_sample: u16, n_channels: u16) ![][]f32 {
    const bytes_per_sample: usize = blk: {
        if (audio_format == 1 and bits_per_sample == 16) break :blk 2;
        if (audio_format == 1 and bits_per_sample == 24) break :blk 3;
        if (audio_format == 3 and bits_per_sample == 32) break :blk 4;
        return Error.UnsupportedWavFormat;
    };
    const frame_stride = @as(usize, n_channels) * bytes_per_sample;
    const n_samples = payload.len / frame_stride;
    const out_channels: usize = if (n_channels == 1) 1 else 2;

    const channels = try allocator.alloc([]f32, out_channels);
    var allocated: usize = 0;
    errdefer {
        for (channels[0..allocated]) |ch| allocator.free(ch);
        allocator.free(channels);
    }
    for (channels) |*ch| {
        ch.* = try allocator.alloc(f32, n_samples);
        allocated += 1;
    }

    for (channels, 0..) |ch, c| {
        const base = c * bytes_per_sample;
        for (ch, 0..) |*dst, t| {
            const p = payload[t * frame_stride + base ..];
            dst.* = switch (bytes_per_sample) {
                2 => @as(f32, @floatFromInt(std.mem.readInt(i16, p[0..2], .little))) / 32768.0,
                3 => @as(f32, @floatFromInt(readS24le(p[0..3]))) / 8388608.0,
                4 => @bitCast(std.mem.readInt(u32, p[0..4], .little)),
                else => unreachable,
            };
        }
    }

    return channels;
}

/// Reads and parses a WAV file.
pub fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Audio {
    const bytes = try readFileBytes(io, allocator, path);
    defer allocator.free(bytes);
    return parse(allocator, bytes);
}

pub fn readFileBytes(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotAFile;
    const byte_len: usize = @intCast(stat.size);
    const bytes = try allocator.alloc(u8, byte_len);
    errdefer allocator.free(bytes);

    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}

/// torchaudio-parity resample with the SHIPPED reference binary's exact
/// arithmetic. clang -O2 compiles `audio_resample_apply_mono`'s
/// `sum += x[k]*w[k]` loop as a strict vectorized reduction — the first
/// `K & ~15` products are individually rounded (fmul.4s) and added to the
/// running sum IN ORDER, while the scalar epilogue contracts to fmadd
/// (verified in the disassembly). `resample.resample`'s uniform
/// two-rounding port differs by 1 ulp on a few % of samples — enough to
/// flip s16 quantization boundaries in the silence-trim round-trip and
/// desync the RVQ codes. Kernel construction is shared with resample.zig
/// (all f64, no contractible mul+add pairs). Caller frees.
pub fn resampleFma(allocator: std.mem.Allocator, in: []const f32, sr_in: u32, sr_out: u32) ![]f32 {
    if (in.len == 0 or sr_in == 0 or sr_out == 0) return error.InvalidInput;
    if (sr_in == sr_out) return allocator.dupe(f32, in);

    const g = resample.gcd(sr_in, sr_out);
    const orig = sr_in / g;
    const newf = sr_out / g;

    var kernel = try resample.buildKernel(allocator, orig, newf);
    defer kernel.deinit(allocator);
    const k_size = kernel.kernel_size;

    const target_f = @ceil(@as(f64, @floatFromInt(sr_out)) * @as(f64, @floatFromInt(in.len)) / @as(f64, @floatFromInt(sr_in)));
    const target: usize = @intFromFloat(target_f);
    if (target == 0) return error.InvalidInput;

    const np = in.len + 2 * kernel.width + orig;
    const padded = try allocator.alloc(f32, np);
    defer allocator.free(padded);
    @memset(padded, 0);
    @memcpy(padded[kernel.width..][0..in.len], in);

    const n_per_chan = (np - k_size) / orig + 1;
    const out_len = @min(target, n_per_chan * newf);
    const out = try allocator.alloc(f32, target);
    errdefer allocator.free(out);
    if (out_len < target) @memset(out[out_len..], 0);

    const nv = k_size & ~@as(usize, 15);
    for (0..out_len) |t_out| {
        const chan = t_out % newf;
        const pos = t_out / newf;
        const w = kernel.weights[chan * k_size ..][0..k_size];
        const x = padded[pos * orig ..][0..k_size];
        var sum: f32 = 0.0;
        var k: usize = 0;
        while (k < nv) : (k += 1) {
            sum += x[k] * w[k]; // rounded product, ordered add
        }
        while (k < k_size) : (k += 1) {
            sum = @mulAdd(f32, x[k], w[k], sum); // contracted scalar tail
        }
        out[t_out] = sum;
    }
    return out;
}

/// Resamples the decoded channels to `target_sr` (per channel, keeping them
/// coherent) and downmixes to mono `0.5*(L+R)` (reference `audio_read_mono`).
/// A mono source resamples its single channel and skips the downmix — the
/// reference duplicates mono into L/R and downmixes `0.5*(x+x)`, which is
/// bit-identical. Caller frees the returned samples.
pub fn monoFromAudio(allocator: std.mem.Allocator, audio: *const Audio, target_sr: u32) ![]f32 {
    if (audio.channels.len == 0 or audio.frames() == 0) return Error.NoAudioData;

    if (audio.channels.len == 1) {
        return resampleFma(allocator, audio.channels[0], audio.sample_rate, target_sr);
    }

    const left = try resampleFma(allocator, audio.channels[0], audio.sample_rate, target_sr);
    defer allocator.free(left);
    const right = try resampleFma(allocator, audio.channels[1], audio.sample_rate, target_sr);
    defer allocator.free(right);

    const n = @min(left.len, right.len);
    const mono = try allocator.alloc(f32, n);
    for (mono, left[0..n], right[0..n]) |*dst, l, r| {
        dst.* = 0.5 * (l + r);
    }
    return mono;
}

/// Reads a WAV file, resamples to `target_sr` and downmixes to mono
/// (reference `audio_read_mono`). Caller frees.
pub fn readMono(io: std.Io, allocator: std.mem.Allocator, path: []const u8, target_sr: u32) ![]f32 {
    var audio = try readFile(io, allocator, path);
    defer audio.deinit();
    return monoFromAudio(allocator, &audio, target_sr);
}

/// Output sample format (reference `WavFormat`): s16/s24 = integer PCM
/// (fmt_tag 1), f32 = IEEE float (fmt_tag 3).
pub const Format = enum { s16, s24, f32 };

/// Parses a CLI format string (reference `audio_parse_format`). Accepts
/// "wav16", "wav24", "wav32"; null on anything else.
pub fn parseFormat(s: []const u8) ?Format {
    if (std.mem.eql(u8, s, "wav16")) return .s16;
    if (std.mem.eql(u8, s, "wav24")) return .s24;
    if (std.mem.eql(u8, s, "wav32")) return .f32;
    return null;
}

fn sanitize(x: f32) f32 {
    return if (std.math.isFinite(x)) x else 0.0;
}

fn clamp1(x: f32) f32 {
    return if (x < -1.0) -1.0 else if (x > 1.0) 1.0 else x;
}

/// Encodes mono f32 samples to an in-memory WAV image, byte-identical to the
/// reference `audio_encode_wav`: 44-byte classic header, NaN/Inf coerced to
/// zero; s16 = trunc(clamp1(x)*32767.0f), s24 = trunc(clamp1(x)*8388607.0f)
/// (C-cast truncation toward zero, which `@intFromFloat` matches), f32 = raw
/// bits with no clamping.
pub fn encodeMono(allocator: std.mem.Allocator, samples: []const f32, sample_rate: u32, format: Format) ![]u8 {
    const bytes_per_sample: usize = switch (format) {
        .s16 => 2,
        .s24 => 3,
        .f32 => 4,
    };
    const fmt_tag: u16 = switch (format) {
        .s16, .s24 => 1,
        .f32 => 3,
    };
    const data_size: u32 = @intCast(samples.len * bytes_per_sample);
    const out = try allocator.alloc(u8, 44 + data_size);
    errdefer allocator.free(out);

    @memcpy(out[0..4], "RIFF");
    std.mem.writeInt(u32, out[4..8], 36 + data_size, .little);
    @memcpy(out[8..12], "WAVE");
    @memcpy(out[12..16], "fmt ");
    std.mem.writeInt(u32, out[16..20], 16, .little);
    std.mem.writeInt(u16, out[20..22], fmt_tag, .little);
    std.mem.writeInt(u16, out[22..24], 1, .little);
    std.mem.writeInt(u32, out[24..28], sample_rate, .little);
    std.mem.writeInt(u32, out[28..32], sample_rate * @as(u32, @intCast(bytes_per_sample)), .little);
    std.mem.writeInt(u16, out[32..34], @intCast(bytes_per_sample), .little);
    std.mem.writeInt(u16, out[34..36], @intCast(bytes_per_sample * 8), .little);
    @memcpy(out[36..40], "data");
    std.mem.writeInt(u32, out[40..44], data_size, .little);

    var pos: usize = 44;
    switch (format) {
        .s16 => for (samples) |x| {
            const s: i16 = @intFromFloat(clamp1(sanitize(x)) * 32767.0);
            std.mem.writeInt(u16, out[pos..][0..2], @bitCast(s), .little);
            pos += 2;
        },
        .s24 => for (samples) |x| {
            const s: i32 = @intFromFloat(clamp1(sanitize(x)) * 8388607.0);
            const u: u32 = @bitCast(s);
            out[pos] = @truncate(u);
            out[pos + 1] = @truncate(u >> 8);
            out[pos + 2] = @truncate(u >> 16);
            pos += 3;
        },
        .f32 => for (samples) |x| {
            std.mem.writeInt(u32, out[pos..][0..4], @bitCast(sanitize(x)), .little);
            pos += 4;
        },
    }

    return out;
}

/// The 44-byte "live" streaming WAV header (reference
/// `wav_stream_write_header`): both the RIFF chunk size and the data chunk
/// size advertise 0x7FFFFFFF, the conventional "unknown / live" marker that
/// aplay, ffmpeg and most players accept by reading until EOF. Never patched
/// afterwards — the stream is one-shot and non-seekable.
pub fn streamHeader(sample_rate: u32, format: Format) [44]u8 {
    const bytes_per_sample: u32 = switch (format) {
        .s16 => 2,
        .s24 => 3,
        .f32 => 4,
    };
    const fmt_tag: u16 = switch (format) {
        .s16, .s24 => 1,
        .f32 => 3,
    };

    var header: [44]u8 = undefined;
    @memcpy(header[0..4], "RIFF");
    std.mem.writeInt(u32, header[4..8], 0x7FFFFFFF, .little);
    @memcpy(header[8..12], "WAVE");
    @memcpy(header[12..16], "fmt ");
    std.mem.writeInt(u32, header[16..20], 16, .little);
    std.mem.writeInt(u16, header[20..22], fmt_tag, .little);
    std.mem.writeInt(u16, header[22..24], 1, .little);
    std.mem.writeInt(u32, header[24..28], sample_rate, .little);
    std.mem.writeInt(u32, header[28..32], sample_rate * bytes_per_sample, .little);
    std.mem.writeInt(u16, header[32..34], @intCast(bytes_per_sample), .little);
    std.mem.writeInt(u16, header[34..36], @intCast(bytes_per_sample * 8), .little);
    @memcpy(header[36..40], "data");
    std.mem.writeInt(u32, header[40..44], 0x7FFFFFFF, .little);
    return header;
}

/// Minimal streaming WAV sink over a byte writer (reference `wav_stream`).
/// `writeHeader` emits a fresh live header (called once at open, and again
/// at every utterance boundary by line-oriented streaming so a client can
/// split the stream into standalone WAV clips on the RIFF magic).
/// `writeSamples` encodes with the exact buffered-writer arithmetic (NaN/Inf
/// -> 0; S16/S24 clamp then truncate toward zero; F32 raw bits) and flushes
/// after every write so a downstream pipe sees the bytes immediately.
pub const StreamSink = struct {
    writer: *std.Io.Writer,
    sample_rate: u32,
    format: Format,

    pub fn writeHeader(self: *StreamSink) !void {
        const header = streamHeader(self.sample_rate, self.format);
        try self.writer.writeAll(&header);
        try self.writer.flush();
    }

    pub fn writeSamples(self: *StreamSink, samples: []const f32) !void {
        if (samples.len == 0) {
            return;
        }
        const w = self.writer;
        switch (self.format) {
            .s16 => for (samples) |x| {
                const s: i16 = @intFromFloat(clamp1(sanitize(x)) * 32767.0);
                try w.writeInt(u16, @bitCast(s), .little);
            },
            .s24 => for (samples) |x| {
                const s: i32 = @intFromFloat(clamp1(sanitize(x)) * 8388607.0);
                const u: u32 = @bitCast(s);
                const b = [3]u8{ @truncate(u), @truncate(u >> 8), @truncate(u >> 16) };
                try w.writeAll(&b);
            },
            .f32 => for (samples) |x| {
                try w.writeInt(u32, @bitCast(sanitize(x)), .little);
            },
        }
        try w.flush();
    }
};

/// Writes mono f32 samples to a WAV file in the requested format
/// (reference `audio_write_wav`).
pub fn writeMono(io: std.Io, allocator: std.mem.Allocator, path: []const u8, samples: []const f32, sample_rate: u32, format: Format) !void {
    const image = try encodeMono(allocator, samples, sample_rate, format);
    defer allocator.free(image);

    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var write_buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &write_buffer);
    try writer.interface.writeAll(image);
    try writer.interface.flush();
}

test {
    _ = @import("wav_tests.zig");
}
