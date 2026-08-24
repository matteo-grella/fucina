//! Behavioral tests for the NAM WAV reader/writer (`wav.zig`): encode/parse
//! round-trips for PCM24 and float32, and rejection of garbage input plus the
//! non-mono `requireMono` guard.

const std = @import("std");
const wav = @import("wav.zig");

const Error = wav.Error;
const SampleFormat = wav.SampleFormat;
const encodeMono = wav.encodeMono;
const parse = wav.parse;

test "wav round-trip pcm24 and float32" {
    const allocator = std.testing.allocator;
    const samples = [_]f32{ 0.0, 0.5, -0.5, 0.999, -1.0, 0.000244140625 };

    inline for (.{ SampleFormat.pcm24, SampleFormat.float32 }) |format| {
        const image = try encodeMono(allocator, &samples, 48000, format);
        defer allocator.free(image);

        var decoded = try parse(allocator, image);
        defer decoded.deinit();
        try std.testing.expectEqual(@as(usize, 1), decoded.channels);
        try std.testing.expectEqual(@as(u32, 48000), decoded.sample_rate);
        const mono = try decoded.requireMono();
        try std.testing.expectEqual(samples.len, mono.len);
        for (samples, mono) |expected, got| {
            const tol: f32 = if (format == .pcm24) 1.0 / 8388608.0 else 0.0;
            try std.testing.expect(@abs(expected - got) <= tol);
        }
    }
}

test "wav rejects garbage and non-mono requireMono" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(Error.NotWav, parse(allocator, "definitely not a wav"));

    // Hand-build a 2-channel 16-bit file: 2 frames.
    var image: [12 + 24 + 8 + 8]u8 = undefined;
    @memcpy(image[0..4], "RIFF");
    std.mem.writeInt(u32, image[4..8], image.len - 8, .little);
    @memcpy(image[8..12], "WAVE");
    @memcpy(image[12..16], "fmt ");
    std.mem.writeInt(u32, image[16..20], 16, .little);
    std.mem.writeInt(u16, image[20..22], 1, .little);
    std.mem.writeInt(u16, image[22..24], 2, .little);
    std.mem.writeInt(u32, image[24..28], 44100, .little);
    std.mem.writeInt(u32, image[28..32], 44100 * 4, .little);
    std.mem.writeInt(u16, image[32..34], 4, .little);
    std.mem.writeInt(u16, image[34..36], 16, .little);
    @memcpy(image[36..40], "data");
    std.mem.writeInt(u32, image[40..44], 8, .little);
    std.mem.writeInt(i16, image[44..46], 16384, .little);
    std.mem.writeInt(i16, image[46..48], -16384, .little);
    std.mem.writeInt(i16, image[48..50], 0, .little);
    std.mem.writeInt(i16, image[50..52], 8192, .little);

    var decoded = try parse(allocator, &image);
    defer decoded.deinit();
    try std.testing.expectEqual(@as(usize, 2), decoded.channels);
    try std.testing.expectEqual(@as(usize, 2), decoded.frames());
    try std.testing.expectError(Error.NotMono, decoded.requireMono());
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), decoded.samples[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), decoded.samples[1], 1e-7);
}
