//! Minimal WAV reader/writer for the NAM example.
//!
//! Read: RIFF/WAVE with PCM 16/24/32-bit int, 32-bit IEEE float, including
//! WAVE_FORMAT_EXTENSIBLE wrappers; any channel count (NAM needs mono — use
//! `requireMono`). Integer samples normalize by 2^(bits-1), matching the
//! upstream trainer's wavio convention (neural-amp-modeler nam/data.py:137).
//! Write: mono 24-bit PCM (the upstream trainer's output format,
//! nam/data.py:241-258) or mono 32-bit float (what upstream tools/render
//! emits).

const std = @import("std");

pub const Error = error{
    NotWav,
    UnsupportedWavFormat,
    CorruptWav,
    NotMono,
};

pub const Wav = struct {
    /// Interleaved samples in [-1, 1) for int sources, raw for float sources.
    samples: []f32,
    channels: usize,
    sample_rate: u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Wav) void {
        self.allocator.free(self.samples);
        self.* = undefined;
    }

    pub fn frames(self: *const Wav) usize {
        return self.samples.len / self.channels;
    }

    /// Returns the sample slice if the file is mono, error otherwise.
    pub fn requireMono(self: *const Wav) Error![]f32 {
        if (self.channels != 1) return Error.NotMono;
        return self.samples;
    }
};

const FormatTag = enum { pcm, ieee_float };

/// Parses a whole WAV file image from memory.
pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Wav {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) {
        return Error.NotWav;
    }

    var format: ?FormatTag = null;
    var bits: u16 = 0;
    var channels: u16 = 0;
    var sample_rate: u32 = 0;
    var data: ?[]const u8 = null;

    var pos: usize = 12;
    while (pos + 8 <= bytes.len) {
        const chunk_id = bytes[pos .. pos + 4];
        const chunk_len: usize = std.mem.readInt(u32, bytes[pos + 4 ..][0..4], .little);
        pos += 8;
        if (chunk_len > bytes.len - pos) return Error.CorruptWav;
        const chunk = bytes[pos .. pos + chunk_len];

        if (std.mem.eql(u8, chunk_id, "fmt ")) {
            if (chunk.len < 16) return Error.CorruptWav;
            var tag = std.mem.readInt(u16, chunk[0..2], .little);
            channels = std.mem.readInt(u16, chunk[2..4], .little);
            sample_rate = std.mem.readInt(u32, chunk[4..8], .little);
            bits = std.mem.readInt(u16, chunk[14..16], .little);
            if (tag == 0xFFFE) {
                // WAVE_FORMAT_EXTENSIBLE: the real tag is the first two bytes
                // of the SubFormat GUID at offset 24.
                if (chunk.len < 26) return Error.CorruptWav;
                tag = std.mem.readInt(u16, chunk[24..26], .little);
            }
            format = switch (tag) {
                1 => .pcm,
                3 => .ieee_float,
                else => return Error.UnsupportedWavFormat,
            };
        } else if (std.mem.eql(u8, chunk_id, "data")) {
            data = chunk;
        }

        pos += chunk_len + (chunk_len & 1); // chunks are word-aligned
    }

    const fmt = format orelse return Error.CorruptWav;
    const payload = data orelse return Error.CorruptWav;
    if (channels == 0 or sample_rate == 0) return Error.CorruptWav;

    const bytes_per_sample: usize = switch (fmt) {
        .pcm => switch (bits) {
            16 => 2,
            24 => 3,
            32 => 4,
            else => return Error.UnsupportedWavFormat,
        },
        .ieee_float => switch (bits) {
            32 => 4,
            else => return Error.UnsupportedWavFormat,
        },
    };
    const count = payload.len / bytes_per_sample;

    const samples = try allocator.alloc(f32, count);
    errdefer allocator.free(samples);

    switch (fmt) {
        .pcm => switch (bits) {
            16 => for (samples, 0..) |*dst, i| {
                const v = std.mem.readInt(i16, payload[i * 2 ..][0..2], .little);
                dst.* = @as(f32, @floatFromInt(v)) / 32768.0;
            },
            24 => for (samples, 0..) |*dst, i| {
                const lo: u32 = payload[i * 3];
                const mid: u32 = payload[i * 3 + 1];
                const hi: u32 = payload[i * 3 + 2];
                const raw: u32 = lo | (mid << 8) | (hi << 16);
                const v: i32 = if (raw & 0x800000 != 0) @as(i32, @bitCast(raw | 0xFF000000)) else @as(i32, @bitCast(raw));
                dst.* = @as(f32, @floatFromInt(v)) / 8388608.0;
            },
            32 => for (samples, 0..) |*dst, i| {
                const v = std.mem.readInt(i32, payload[i * 4 ..][0..4], .little);
                dst.* = @floatCast(@as(f64, @floatFromInt(v)) / 2147483648.0);
            },
            else => unreachable,
        },
        .ieee_float => for (samples, 0..) |*dst, i| {
            const raw = std.mem.readInt(u32, payload[i * 4 ..][0..4], .little);
            dst.* = @bitCast(raw);
        },
    }

    return .{
        .samples = samples,
        .channels = channels,
        .sample_rate = sample_rate,
        .allocator = allocator,
    };
}

pub fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Wav {
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

pub const SampleFormat = enum { pcm24, float32 };

/// Serializes mono samples to an in-memory WAV image.
pub fn encodeMono(allocator: std.mem.Allocator, samples: []const f32, sample_rate: u32, format: SampleFormat) ![]u8 {
    const bytes_per_sample: usize = switch (format) {
        .pcm24 => 3,
        .float32 => 4,
    };
    const data_len = samples.len * bytes_per_sample;
    // RIFF(12) + fmt(8+16) + [fact(8+4) for float] + data(8+len)
    const fact_len: usize = if (format == .float32) 12 else 0;
    const total = 12 + 24 + fact_len + 8 + data_len;
    const out = try allocator.alloc(u8, total);
    errdefer allocator.free(out);

    @memcpy(out[0..4], "RIFF");
    std.mem.writeInt(u32, out[4..8], @intCast(total - 8), .little);
    @memcpy(out[8..12], "WAVE");

    @memcpy(out[12..16], "fmt ");
    std.mem.writeInt(u32, out[16..20], 16, .little);
    const tag: u16 = switch (format) {
        .pcm24 => 1,
        .float32 => 3,
    };
    std.mem.writeInt(u16, out[20..22], tag, .little);
    std.mem.writeInt(u16, out[22..24], 1, .little); // channels
    std.mem.writeInt(u32, out[24..28], sample_rate, .little);
    const block_align: u16 = @intCast(bytes_per_sample);
    std.mem.writeInt(u32, out[28..32], sample_rate * @as(u32, block_align), .little); // byte rate
    std.mem.writeInt(u16, out[32..34], block_align, .little);
    std.mem.writeInt(u16, out[34..36], @intCast(bytes_per_sample * 8), .little);

    var pos: usize = 36;
    if (format == .float32) {
        @memcpy(out[pos .. pos + 4], "fact");
        std.mem.writeInt(u32, out[pos + 4 ..][0..4], 4, .little);
        std.mem.writeInt(u32, out[pos + 8 ..][0..4], @intCast(samples.len), .little);
        pos += 12;
    }

    @memcpy(out[pos .. pos + 4], "data");
    std.mem.writeInt(u32, out[pos + 4 ..][0..4], @intCast(data_len), .little);
    pos += 8;

    switch (format) {
        .pcm24 => for (samples) |s| {
            const scaled = @round(@as(f64, s) * 8388608.0);
            const clamped = std.math.clamp(scaled, -8388608.0, 8388607.0);
            const v: i32 = @intFromFloat(clamped);
            const raw: u32 = @bitCast(v);
            out[pos] = @truncate(raw);
            out[pos + 1] = @truncate(raw >> 8);
            out[pos + 2] = @truncate(raw >> 16);
            pos += 3;
        },
        .float32 => for (samples) |s| {
            std.mem.writeInt(u32, out[pos..][0..4], @bitCast(s), .little);
            pos += 4;
        },
    }

    return out;
}

pub fn writeMono(io: std.Io, allocator: std.mem.Allocator, path: []const u8, samples: []const f32, sample_rate: u32, format: SampleFormat) !void {
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
