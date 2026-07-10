//! Tests for resample.zig against golden outputs of the C++ reference
//! (refs/omnivoice.cpp/src/audio-resample.h). Goldens were produced by a
//! standalone harness that feeds 977 LCG-generated samples through
//! audio_resample and prints the results as u32 bit patterns.
//!
//! The harness is compiled with `clang++ -O2 -ffp-contract=off` so the goldens
//! carry the reference's SOURCE semantics (sequential f32 mul+add). Default
//! clang -O2 on aarch64 contracts only the scalar remainder loop of the dot
//! product to fmadd (the vectorized 16-tap chunks stay unfused), a
//! target/compiler artifact that perturbs low-magnitude samples by ulps. The
//! Zig port matches the contract=off build bit for bit on the full outputs of
//! all four rate pairs (verified exhaustively, 3139 samples).

const std = @import("std");
const resample = @import("resample.zig");

/// Deterministic test signal: LCG state*1664525+1013904223 from seed
/// 0x12345678, top 24 bits mapped to [-1, 1) via exact power-of-two scaling
/// (integer arithmetic + exact f64 ops only — replicated in the C++ harness).
fn makeInput(buf: []f32) void {
    var state: u32 = 0x12345678;
    for (buf) |*s| {
        state = state *% 1664525 +% 1013904223;
        const u: u32 = state >> 8;
        s.* = @floatCast(@as(f64, @floatFromInt(u)) * (1.0 / 8388608.0) - 1.0);
    }
}

fn expectBits(expected: []const u32, got: []const f32) !void {
    for (expected, got) |e, g| {
        try std.testing.expectEqual(e, @as(u32, @bitCast(g)));
    }
}

const Golden = struct {
    sr_in: u32,
    sr_out: u32,
    n_out: usize,
    first32: [32]u32,
    last8: [8]u32,
};

const goldens = [_]Golden{
    .{
        .sr_in = 24000,
        .sr_out = 16000,
        .n_out = 652,
        .first32 = .{
            0x3E39F0F9, 0xBD086337, 0xBF17403A, 0x3EF6D8A6, 0x3F6305D2, 0x3F2C39F5, 0xBE528CC2, 0xBED62E89,
            0xBF05D265, 0xBD49FE62, 0xBDF0D3C4, 0x3E227C42, 0x3EA31129, 0xBE95B472, 0xBDE78A04, 0x3EFB4836,
            0x3F0F2E63, 0x3D840CF9, 0xBD646C1A, 0xBEA1BB79, 0x3F4491F2, 0x3EDC8B62, 0xBED7BE91, 0xBE1D9A0A,
            0x3EA3826C, 0x3EF40F77, 0xBE484BAD, 0xBF16F45B, 0x3F327F2E, 0x3E32E041, 0xBEB13904, 0xBF55EB50,
        },
        .last8 = .{ 0xBF2ED7FC, 0xBF0162DD, 0xBF0AEEF8, 0x3E1E0FD3, 0x3E0E8463, 0xBD715020, 0x3F49CA2E, 0xBED5473B },
    },
    .{
        .sr_in = 16000,
        .sr_out = 24000,
        .n_out = 1466,
        .first32 = .{
            0xBD96FDA9, 0x3F0ED7E7, 0x3E886E9E, 0xBF314F86, 0xBF2D56DB, 0xBE63CECF, 0x3D410DBF, 0x3F0D5EB8,
            0x3F580465, 0x3F55DC51, 0x3F704007, 0x3F36176C, 0x3E9CF808, 0x3E2A73F2, 0xBE8ED81C, 0xBF3D7D93,
            0xBEDEF8F4, 0xBDDD7D04, 0xBEF1D06E, 0xBF41221D, 0xBE845082, 0x3EC81F55, 0x3E64E523, 0xBEED1B89,
            0xBED0A6D5, 0x3EEAF575, 0x3F4B8761, 0x3E4F1D51, 0xBE9A4845, 0xBE9547C8, 0xBDC85F14, 0xBC85ED62,
        },
        .last8 = .{ 0x3E8DF881, 0xBF6E34B4, 0xBEE88A8C, 0x3F763E98, 0x3F99D99F, 0x3E2EC9B4, 0xBEF6BA6F, 0xBE587776 },
    },
    .{
        .sr_in = 48000,
        .sr_out = 24000,
        .n_out = 489,
        .first32 = .{
            0x3E4B33C6, 0xBEA66E07, 0xBBB1315D, 0x3F7FA8D0, 0x3EAD19C0, 0xBEE0001F, 0xBED2F62B, 0xBDDB1128,
            0x3D78ADFC, 0x3E87B50E, 0xBEA68C23, 0x3EA2F633, 0x3F05E18A, 0x3D1826A3, 0xBE8E9ABD, 0x3F254C29,
            0x3E17EB48, 0xBECB9500, 0x3EC627AE, 0x3E903ABF, 0xBF0D6F72, 0x3EFBE34D, 0x3D74B06D, 0xBF340327,
            0xBEED3192, 0xBF0567A3, 0xBE5F73B1, 0xBE9321BC, 0x3EBAD4D4, 0x3EAFCAA3, 0x3E44AE0D, 0xBF51A33B,
        },
        .last8 = .{ 0xBE7662B3, 0x3E80D75C, 0xBF264DC7, 0xBF17D595, 0xBD31292D, 0x3D3E8900, 0x3EB1F1DA, 0x3DE94976 },
    },
    .{
        .sr_in = 44100,
        .sr_out = 24000,
        .n_out = 532,
        .first32 = .{
            0x3E530A07, 0xBE8815DD, 0xBE5095A8, 0x3F6686A6, 0x3F2B2CDA, 0xBE59504A, 0xBF0476C2, 0xBE7F4255,
            0xBDA0190F, 0x3E333124, 0x3E328439, 0xBEABCCEF, 0x3EB1DB06, 0x3F09C57F, 0x3DE22409, 0xBE960D3C,
            0x3EA63092, 0x3F2016CD, 0xBEDA09E6, 0xBD0793BB, 0x3F038050, 0xBC210261, 0xBEFBCE11, 0x3F1F1059,
            0xBC97A720, 0xBF3BAAEC, 0xBEC33CCA, 0xBF339DE8, 0xBD136C54, 0xBF19BF10, 0x3ECB162C, 0x3DAFB551,
        },
        .last8 = .{ 0x3EC12D3A, 0xBE3F6536, 0xBF2583F1, 0xBF109CC7, 0x3D0283C9, 0x3D044986, 0x3EA27464, 0x3E6DF01C },
    },
};

test "LCG input matches the C++ harness bit for bit" {
    var in: [977]f32 = undefined;
    makeInput(&in);
    try std.testing.expectEqual(@as(u32, 0xBDABCD90), @as(u32, @bitCast(in[0])));
    try std.testing.expectEqual(@as(u32, 0xBEF6961C), @as(u32, @bitCast(in[976])));
}

test "gcd" {
    try std.testing.expectEqual(@as(u32, 8000), resample.gcd(24000, 16000));
    try std.testing.expectEqual(@as(u32, 300), resample.gcd(44100, 24000));
    try std.testing.expectEqual(@as(u32, 24000), resample.gcd(48000, 24000));
    try std.testing.expectEqual(@as(u32, 7), resample.gcd(7, 0));
}

test "kernel geometry (width, kernel_size)" {
    const allocator = std.testing.allocator;
    // 24000->16000: orig=3 newf=2, base=1.98, width=ceil(18/1.98)=10, K=23.
    var k1 = try resample.buildKernel(allocator, 3, 2);
    defer k1.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 10), k1.width);
    try std.testing.expectEqual(@as(usize, 23), k1.kernel_size);
    try std.testing.expectEqual(@as(usize, 2 * 23), k1.weights.len);
    // 44100->24000: orig=147 newf=80, base=79.2, width=ceil(882/79.2)=12, K=171.
    var k2 = try resample.buildKernel(allocator, 147, 80);
    defer k2.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 12), k2.width);
    try std.testing.expectEqual(@as(usize, 171), k2.kernel_size);
}

test "same rate returns a copy" {
    const allocator = std.testing.allocator;
    var in: [977]f32 = undefined;
    makeInput(&in);
    const out = try resample.resample(allocator, &in, 24000, 24000);
    defer allocator.free(out);
    try std.testing.expectEqualSlices(f32, &in, out);
    try std.testing.expect(out.ptr != @as([]const f32, &in).ptr);
}

test "invalid input" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidInput, resample.resample(allocator, &.{}, 24000, 16000));
    const one = [_]f32{0.5};
    try std.testing.expectError(error.InvalidInput, resample.resample(allocator, &one, 0, 16000));
    try std.testing.expectError(error.InvalidInput, resample.resample(allocator, &one, 24000, 0));
}

test "golden parity vs C++ reference (bit exact)" {
    const allocator = std.testing.allocator;
    var in: [977]f32 = undefined;
    makeInput(&in);

    for (goldens) |g| {
        const out = try resample.resample(allocator, &in, g.sr_in, g.sr_out);
        defer allocator.free(out);
        try std.testing.expectEqual(g.n_out, out.len);
        try expectBits(&g.first32, out[0..32]);
        try expectBits(&g.last8, out[out.len - 8 ..]);
    }
}
