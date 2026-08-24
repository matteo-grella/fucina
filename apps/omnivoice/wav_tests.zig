//! Golden-parity tests for the OmniVoice WAV module (`wav.zig`) vs the C++
//! reference (refs/omnivoice.cpp/src/wav.h reader + audio-io.h writer).
//! Golden bytes / bit patterns were produced by a harness compiled against
//! the reference headers (scratchpad wav_golden.cpp: audio_encode_wav for
//! writer bytes, read_wav_buf for decoded f32 bits). All comparisons are
//! bit-exact.

const std = @import("std");
const resample = @import("resample.zig");
const wav = @import("wav.zig");

// 25 test samples as f32 bit patterns: ±0, ±1, ±1.5 (clip), NaN/Inf/-NaN,
// ~±1/3 (truncation-vs-round), ±0.5, ~±1e-8, FLT_MIN, smallest denormal,
// just-inside-±1, ~1/sqrt(2), 0.25, ±0.1, ~0.999992.
const sample_bits = [_]u32{
    0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3FC00000, 0xBFC00000,
    0x7FC00000, 0x7F800000, 0xFF800000, 0xFFC00000, 0x3EAAAAAB, 0xBEAAAAAB,
    0x3F000000, 0xBF000000, 0x322BCC77, 0xB22BCC77, 0x00800000, 0x00000001,
    0x3F7FFFFF, 0xBF7FFFFF, 0x3F3504F3, 0x3E800000, 0x3DCCCCCD, 0xBDCCCCCD,
    0x3F7FFF58,
};

fn testSamples() [sample_bits.len]f32 {
    var s: [sample_bits.len]f32 = undefined;
    for (&s, sample_bits) |*dst, bits| dst.* = @bitCast(bits);
    return s;
}

const golden_s16_wav = [_]u8{
    0x52, 0x49, 0x46, 0x46, 0x56, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
    0x66, 0x6D, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
    0xC0, 0x5D, 0x00, 0x00, 0x80, 0xBB, 0x00, 0x00, 0x02, 0x00, 0x10, 0x00,
    0x64, 0x61, 0x74, 0x61, 0x32, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xFF, 0x7F, 0x01, 0x80, 0xFF, 0x7F, 0x01, 0x80, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xAA, 0x2A, 0x56, 0xD5, 0xFF, 0x3F, 0x01, 0xC0,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFE, 0x7F, 0x02, 0x80,
    0x81, 0x5A, 0xFF, 0x1F, 0xCC, 0x0C, 0x34, 0xF3, 0xFE, 0x7F,
};
const golden_s24_wav = [_]u8{
    0x52, 0x49, 0x46, 0x46, 0x6F, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
    0x66, 0x6D, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
    0xC0, 0x5D, 0x00, 0x00, 0x40, 0x19, 0x01, 0x00, 0x03, 0x00, 0x18, 0x00,
    0x64, 0x61, 0x74, 0x61, 0x4B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0xFF, 0xFF, 0x7F, 0x01, 0x00, 0x80, 0xFF, 0xFF, 0x7F, 0x01,
    0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0xAA, 0xAA, 0x2A, 0x56, 0x55, 0xD5, 0xFF, 0xFF, 0x3F, 0x01,
    0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0xFE, 0xFF, 0x7F, 0x02, 0x00, 0x80, 0x79, 0x82, 0x5A, 0xFF,
    0xFF, 0x1F, 0xCC, 0xCC, 0x0C, 0x34, 0x33, 0xF3, 0xAB, 0xFF, 0x7F,
};
const golden_f32_wav = [_]u8{
    0x52, 0x49, 0x46, 0x46, 0x88, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
    0x66, 0x6D, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x01, 0x00,
    0xC0, 0x5D, 0x00, 0x00, 0x00, 0x77, 0x01, 0x00, 0x04, 0x00, 0x20, 0x00,
    0x64, 0x61, 0x74, 0x61, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0xBF,
    0x00, 0x00, 0xC0, 0x3F, 0x00, 0x00, 0xC0, 0xBF, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xAB, 0xAA, 0xAA, 0x3E, 0xAB, 0xAA, 0xAA, 0xBE, 0x00, 0x00, 0x00, 0x3F,
    0x00, 0x00, 0x00, 0xBF, 0x77, 0xCC, 0x2B, 0x32, 0x77, 0xCC, 0x2B, 0xB2,
    0x00, 0x00, 0x80, 0x00, 0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x7F, 0x3F,
    0xFF, 0xFF, 0x7F, 0xBF, 0xF3, 0x04, 0x35, 0x3F, 0x00, 0x00, 0x80, 0x3E,
    0xCD, 0xCC, 0xCC, 0x3D, 0xCD, 0xCC, 0xCC, 0xBD, 0x58, 0xFF, 0x7F, 0x3F,
};

// read_wav_buf output for the three encodings above (mono: L==R, L listed).
const golden_s16_decoded = [_]u32{
    0x00000000, 0x00000000, 0x3F7FFE00, 0xBF7FFE00, 0x3F7FFE00, 0xBF7FFE00,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3EAAA800, 0xBEAAA800,
    0x3EFFFC00, 0xBEFFFC00, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x3F7FFC00, 0xBF7FFC00, 0x3F350200, 0x3E7FF800, 0x3DCCC000, 0xBDCCC000,
    0x3F7FFC00,
};
const golden_s24_decoded = [_]u32{
    0x00000000, 0x00000000, 0x3F7FFFFE, 0xBF7FFFFE, 0x3F7FFFFE, 0xBF7FFFFE,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3EAAAAA8, 0xBEAAAAA8,
    0x3EFFFFFC, 0xBEFFFFFC, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x3F7FFFFC, 0xBF7FFFFC, 0x3F3504F2, 0x3E7FFFF8, 0x3DCCCCC0, 0xBDCCCCC0,
    0x3F7FFF56,
};
const golden_f32_decoded = [_]u32{
    0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3FC00000, 0xBFC00000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3EAAAAAB, 0xBEAAAAAB,
    0x3F000000, 0xBF000000, 0x322BCC77, 0xB22BCC77, 0x00800000, 0x00000001,
    0x3F7FFFFF, 0xBF7FFFFF, 0x3F3504F3, 0x3E800000, 0x3DCCCCCD, 0xBDCCCCCD,
    0x3F7FFF58,
};

// WAVE_FORMAT_EXTENSIBLE PCM16 stereo (44.1 kHz), preceded by an odd-size
// "junk" chunk + pad byte. Frames: (32767,-32768) (8192,-8192) (1,-1)
// (-32767,32766).
const ext_pcm16_wav = [_]u8{
    0x52, 0x49, 0x46, 0x46, 0x58, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
    0x6A, 0x75, 0x6E, 0x6B, 0x03, 0x00, 0x00, 0x00, 0xAA, 0xBB, 0xCC, 0x00,
    0x66, 0x6D, 0x74, 0x20, 0x28, 0x00, 0x00, 0x00, 0xFE, 0xFF, 0x02, 0x00,
    0x44, 0xAC, 0x00, 0x00, 0x10, 0xB1, 0x02, 0x00, 0x04, 0x00, 0x10, 0x00,
    0x16, 0x00, 0x10, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
    0x64, 0x61, 0x74, 0x61, 0x10, 0x00, 0x00, 0x00, 0xFF, 0x7F, 0x00, 0x80,
    0x00, 0x20, 0x00, 0xE0, 0x01, 0x00, 0xFF, 0xFF, 0x01, 0x80, 0xFE, 0x7F,
};
const ext_pcm16_left = [_]u32{ 0x3F7FFE00, 0x3E800000, 0x38000000, 0xBF7FFE00 };
const ext_pcm16_right = [_]u32{ 0xBF800000, 0xBE800000, 0xB8000000, 0x3F7FFC00 };

// WAVE_FORMAT_EXTENSIBLE IEEE-float mono (16 kHz): 1.0, -0.5, +Inf (the
// reader passes float payloads through verbatim, no sanitizing).
const ext_f32_wav = [_]u8{
    0x52, 0x49, 0x46, 0x46, 0x48, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
    0x66, 0x6D, 0x74, 0x20, 0x28, 0x00, 0x00, 0x00, 0xFE, 0xFF, 0x01, 0x00,
    0x80, 0x3E, 0x00, 0x00, 0x00, 0xFA, 0x00, 0x00, 0x04, 0x00, 0x20, 0x00,
    0x16, 0x00, 0x20, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
    0x64, 0x61, 0x74, 0x61, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F,
    0x00, 0x00, 0x00, 0xBF, 0x00, 0x00, 0x80, 0x7F,
};
const ext_f32_decoded = [_]u32{ 0x3F800000, 0xBF000000, 0x7F800000 };

// Classic PCM16 mono whose data chunk declares 100 bytes but only 6 are
// present: the reader clips the chunk to the file size (3 samples).
const clipped_wav = [_]u8{
    0x52, 0x49, 0x46, 0x46, 0x2A, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
    0x66, 0x6D, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
    0x40, 0x1F, 0x00, 0x00, 0x80, 0x3E, 0x00, 0x00, 0x02, 0x00, 0x10, 0x00,
    0x64, 0x61, 0x74, 0x61, 0x64, 0x00, 0x00, 0x00, 0x64, 0x00, 0x9C, 0xFF,
    0x2C, 0x01,
};
const clipped_decoded = [_]u32{ 0x3B480000, 0xBB480000, 0x3C160000 };

// Classic PCM24 stereo (48 kHz), 2 frames covering s24 sign extension:
// (+1, -1) then (-8388608, +8388607).
const s24_stereo_wav = [_]u8{
    0x52, 0x49, 0x46, 0x46, 0x30, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
    0x66, 0x6D, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00,
    0x80, 0xBB, 0x00, 0x00, 0x00, 0x65, 0x04, 0x00, 0x06, 0x00, 0x18, 0x00,
    0x64, 0x61, 0x74, 0x61, 0x0C, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0xFF,
    0xFF, 0xFF, 0x00, 0x00, 0x80, 0xFF, 0xFF, 0x7F,
};
const s24_stereo_left = [_]u32{ 0x34000000, 0xBF800000 };
const s24_stereo_right = [_]u32{ 0xB4000000, 0x3F7FFFFE };

fn expectBitsEqual(expected: []const u32, got: []const f32) !void {
    try std.testing.expectEqual(expected.len, got.len);
    for (expected, got) |e, g| {
        try std.testing.expectEqual(e, @as(u32, @bitCast(g)));
    }
}

test "encodeMono is byte-identical to reference audio_encode_wav (S16/S24/F32)" {
    const allocator = std.testing.allocator;
    const samples = testSamples();

    const cases = .{
        .{ wav.Format.s16, &golden_s16_wav },
        .{ wav.Format.s24, &golden_s24_wav },
        .{ wav.Format.f32, &golden_f32_wav },
    };
    inline for (cases) |case| {
        const image = try wav.encodeMono(allocator, &samples, 24000, case[0]);
        defer allocator.free(image);
        try std.testing.expectEqualSlices(u8, case[1], image);
    }
}

test "parse recovers reference read_wav_buf values bit-exactly (mono S16/S24/F32)" {
    const allocator = std.testing.allocator;

    const cases = .{
        .{ &golden_s16_wav, &golden_s16_decoded },
        .{ &golden_s24_wav, &golden_s24_decoded },
        .{ &golden_f32_wav, &golden_f32_decoded },
    };
    inline for (cases) |case| {
        var audio = try wav.parse(allocator, case[0]);
        defer audio.deinit();
        try std.testing.expectEqual(@as(u32, 24000), audio.sample_rate);
        try std.testing.expectEqual(@as(usize, 1), audio.channels.len);
        try std.testing.expectEqual(sample_bits.len, audio.frames());
        try expectBitsEqual(case[1], audio.channels[0]);
    }
}

test "parse handles WAVE_FORMAT_EXTENSIBLE PCM16 stereo with odd junk chunk" {
    const allocator = std.testing.allocator;
    var audio = try wav.parse(allocator, &ext_pcm16_wav);
    defer audio.deinit();
    try std.testing.expectEqual(@as(u32, 44100), audio.sample_rate);
    try std.testing.expectEqual(@as(usize, 2), audio.channels.len);
    try expectBitsEqual(&ext_pcm16_left, audio.channels[0]);
    try expectBitsEqual(&ext_pcm16_right, audio.channels[1]);
}

test "parse handles WAVE_FORMAT_EXTENSIBLE float32 mono verbatim (incl. Inf)" {
    const allocator = std.testing.allocator;
    var audio = try wav.parse(allocator, &ext_f32_wav);
    defer audio.deinit();
    try std.testing.expectEqual(@as(u32, 16000), audio.sample_rate);
    try std.testing.expectEqual(@as(usize, 1), audio.channels.len);
    try expectBitsEqual(&ext_f32_decoded, audio.channels[0]);
}

test "parse clips a lying data chunk size to the file size" {
    const allocator = std.testing.allocator;
    var audio = try wav.parse(allocator, &clipped_wav);
    defer audio.deinit();
    try std.testing.expectEqual(@as(u32, 8000), audio.sample_rate);
    try std.testing.expectEqual(@as(usize, 1), audio.channels.len);
    try expectBitsEqual(&clipped_decoded, audio.channels[0]);
}

test "parse sign-extends PCM24 stereo" {
    const allocator = std.testing.allocator;
    var audio = try wav.parse(allocator, &s24_stereo_wav);
    defer audio.deinit();
    try std.testing.expectEqual(@as(u32, 48000), audio.sample_rate);
    try std.testing.expectEqual(@as(usize, 2), audio.channels.len);
    try expectBitsEqual(&s24_stereo_left, audio.channels[0]);
    try expectBitsEqual(&s24_stereo_right, audio.channels[1]);
}

test "parseFormat mirrors audio_parse_format" {
    try std.testing.expectEqual(wav.Format.s16, wav.parseFormat("wav16").?);
    try std.testing.expectEqual(wav.Format.s24, wav.parseFormat("wav24").?);
    try std.testing.expectEqual(wav.Format.f32, wav.parseFormat("wav32").?);
    try std.testing.expectEqual(@as(?wav.Format, null), wav.parseFormat("wav"));
    try std.testing.expectEqual(@as(?wav.Format, null), wav.parseFormat("wav16 "));
    try std.testing.expectEqual(@as(?wav.Format, null), wav.parseFormat(""));
}

test "parse rejects non-WAV, missing data, and unsupported payloads" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(wav.Error.NotWav, wav.parse(allocator, "definitely not a wav"));
    try std.testing.expectError(wav.Error.NotWav, wav.parse(allocator, "RIFF"));
    try std.testing.expectError(wav.Error.NoAudioData, wav.parse(allocator, "RIFF\x04\x00\x00\x00WAVE"));

    // A data chunk before any fmt chunk is skipped (n_channels == 0), like
    // the reference.
    var no_fmt: [12 + 8 + 4]u8 = undefined;
    @memcpy(no_fmt[0..4], "RIFF");
    std.mem.writeInt(u32, no_fmt[4..8], no_fmt.len - 8, .little);
    @memcpy(no_fmt[8..12], "WAVE");
    @memcpy(no_fmt[12..16], "data");
    std.mem.writeInt(u32, no_fmt[16..20], 4, .little);
    @memset(no_fmt[20..24], 0);
    try std.testing.expectError(wav.Error.NoAudioData, wav.parse(allocator, &no_fmt));

    // PCM 8-bit is unsupported; the error surfaces at the data chunk, as in
    // the reference.
    var pcm8: [44 + 2]u8 = undefined;
    @memcpy(pcm8[0..44], golden_s16_wav[0..44]);
    std.mem.writeInt(u16, pcm8[32..34], 1, .little); // block align
    std.mem.writeInt(u16, pcm8[34..36], 8, .little); // bits
    std.mem.writeInt(u32, pcm8[40..44], 2, .little); // data size
    pcm8[44] = 0x80;
    pcm8[45] = 0x80;
    try std.testing.expectError(wav.Error.UnsupportedWavFormat, wav.parse(allocator, &pcm8));
}

test "writeMono/readFile round-trip through the filesystem" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const samples = testSamples();

    const path = "/tmp/fucina-omnivoice-wav-test.wav";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    try wav.writeMono(io, allocator, path, &samples, 24000, .s24);

    const bytes = try wav.readFileBytes(io, allocator, path);
    defer allocator.free(bytes);
    try std.testing.expectEqualSlices(u8, &golden_s24_wav, bytes);

    var audio = try wav.readFile(io, allocator, path);
    defer audio.deinit();
    try std.testing.expectEqual(@as(u32, 24000), audio.sample_rate);
    try std.testing.expectEqual(@as(usize, 1), audio.channels.len);
    try expectBitsEqual(&golden_s24_decoded, audio.channels[0]);
}

test "monoFromAudio: stereo same-rate downmix is exactly 0.5*(L+R)" {
    const allocator = std.testing.allocator;
    var left = [_]f32{ 0.5, -0.25, 0.125, 1.0 };
    var right = [_]f32{ -0.5, 0.75, 0.25, -1.0 };
    var channels = [_][]f32{ try allocator.dupe(f32, &left), try allocator.dupe(f32, &right) };
    var audio = wav.Audio{ .sample_rate = 24000, .channels = &channels, .allocator = allocator };
    defer for (channels) |ch| allocator.free(ch);

    const mono = try wav.monoFromAudio(allocator, &audio, 24000);
    defer allocator.free(mono);
    try std.testing.expectEqual(@as(usize, 4), mono.len);
    for (mono, left, right) |m, l, r| {
        try std.testing.expectEqual(0.5 * (l + r), m);
    }
}

test "monoFromAudio: mono passthrough-resample copies bit-exactly at same rate" {
    const allocator = std.testing.allocator;
    var samples = [_]f32{ 0.1, -0.2, 0.3, -0.4, 0.5 };
    var channels = [_][]f32{try allocator.dupe(f32, &samples)};
    var audio = wav.Audio{ .sample_rate = 24000, .channels = &channels, .allocator = allocator };
    defer allocator.free(channels[0]);

    const mono = try wav.monoFromAudio(allocator, &audio, 24000);
    defer allocator.free(mono);
    try std.testing.expectEqualSlices(f32, &samples, mono);
}

test "resampleFma matches resample.resample within 1 ulp (48k -> 24k)" {
    // The FMA-tail hybrid (the shipped reference binary's arithmetic) and
    // the strict two-rounding port agree to ~1-ulp; lengths are identical.
    const allocator = std.testing.allocator;
    var in: [480]f32 = undefined;
    var state: u64 = 42;
    for (&in) |*v| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        v.* = (@as(f32, @floatFromInt(@as(i64, @intCast(state >> 33 & 0xFFFF)))) - 32768.0) / 32768.0;
    }
    const a = try wav.resampleFma(allocator, &in, 48000, 24000);
    defer allocator.free(a);
    const b = try resample.resample(allocator, &in, 48000, 24000);
    defer allocator.free(b);
    try std.testing.expectEqual(b.len, a.len);
    try std.testing.expectEqual(@as(usize, 240), a.len);
    for (a, b) |av, bv| {
        try std.testing.expectApproxEqAbs(bv, av, 1e-6);
    }
    // Same-rate passthrough copies.
    const c = try wav.resampleFma(allocator, &in, 24000, 24000);
    defer allocator.free(c);
    try std.testing.expectEqualSlices(f32, &in, c);
}

test "streamHeader mirrors wav_stream_write_header: 0x7FFFFFFF sizes, per-format fields" {
    // Reference wav_stream_write_header: RIFF/data sizes = 0x7FFFFFFF (the
    // live "unknown size" marker), otherwise identical field layout to the
    // buffered 44-byte header.
    const cases = .{
        .{ wav.Format.s16, @as(u16, 1), @as(u16, 16), @as(u32, 2) },
        .{ wav.Format.s24, @as(u16, 1), @as(u16, 24), @as(u32, 3) },
        .{ wav.Format.f32, @as(u16, 3), @as(u16, 32), @as(u32, 4) },
    };
    inline for (cases) |case| {
        const h = wav.streamHeader(24000, case[0]);
        try std.testing.expectEqualSlices(u8, "RIFF", h[0..4]);
        try std.testing.expectEqual(@as(u32, 0x7FFFFFFF), std.mem.readInt(u32, h[4..8], .little));
        try std.testing.expectEqualSlices(u8, "WAVE", h[8..12]);
        try std.testing.expectEqualSlices(u8, "fmt ", h[12..16]);
        try std.testing.expectEqual(@as(u32, 16), std.mem.readInt(u32, h[16..20], .little));
        try std.testing.expectEqual(case[1], std.mem.readInt(u16, h[20..22], .little)); // fmt tag
        try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, h[22..24], .little)); // channels
        try std.testing.expectEqual(@as(u32, 24000), std.mem.readInt(u32, h[24..28], .little));
        try std.testing.expectEqual(24000 * case[3], std.mem.readInt(u32, h[28..32], .little)); // byte rate
        try std.testing.expectEqual(@as(u16, @intCast(case[3])), std.mem.readInt(u16, h[32..34], .little)); // block align
        try std.testing.expectEqual(case[2], std.mem.readInt(u16, h[34..36], .little)); // bits
        try std.testing.expectEqualSlices(u8, "data", h[36..40]);
        try std.testing.expectEqual(@as(u32, 0x7FFFFFFF), std.mem.readInt(u32, h[40..44], .little));
    }
}

test "StreamSink payload encode is byte-identical to encodeMono's payload (S16/S24/F32)" {
    const allocator = std.testing.allocator;
    const samples = testSamples();

    inline for (.{ wav.Format.s16, wav.Format.s24, wav.Format.f32 }) |fmt| {
        const image = try wav.encodeMono(allocator, &samples, 24000, fmt);
        defer allocator.free(image);

        var aw = std.Io.Writer.Allocating.init(allocator);
        defer aw.deinit();
        var sink = wav.StreamSink{ .writer = &aw.writer, .sample_rate = 24000, .format = fmt };
        try sink.writeHeader();
        // Split writes: flush-per-write must not change the byte stream.
        try sink.writeSamples(samples[0..7]);
        try sink.writeSamples(samples[7..]);
        const got = aw.written();

        try std.testing.expectEqual(image.len, got.len);
        try std.testing.expectEqualSlices(u8, image[44..], got[44..]);
        // Header differs from the buffered one ONLY in the two size fields.
        try std.testing.expectEqualSlices(u8, image[0..4], got[0..4]);
        try std.testing.expectEqualSlices(u8, image[8..40], got[8..40]);
    }
}
