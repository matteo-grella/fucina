//! Golden tests for rvq_file.zig against the C++ reference rvq-file.h.
//!
//! Golden bytes produced by scratchpad/golden_rvq.cpp (clang++ -O2, includes
//! refs/omnivoice.cpp/src/rvq-file.h directly) with RVQ_CODE_BITS = 11.

const std = @import("std");
const rvq_file = @import("rvq_file.zig");

// [K=8, T=5] row-major k-slow: 40 codes, 440 bits = 55 bytes exactly.
const golden_codes = [_]i32{
    0,     1,     2,     3,     4,
    1023,  1024,  2047,  2046,  512,
    0x555, 0x2AA, 0x7FF, 0x400, 0x001,
    7,     77,    777,   1777,  2000,
    2047,  0,     2047,  0,     2047,
    100,   200,   300,   400,   500,
    1,     10,    100,   1000,  2000,
    999,   42,    0,     1,     1365,
};
const golden_packed_hex =
    "00088000064080ff01f0fffe07505555f57f0006e0004d4858bca1ff7f00fc1f00ff270332580219fa04400164401ff4cfa7020004a0aa";

// 3 codes -> 33 bits -> 5 bytes (ceil), exercising the partial tail byte.
const tail_codes = [_]i32{ 2047, 0, 1234 };
const tail_packed_hex = "ff07803401";

// Out-of-range codes are masked to 11 bits on pack: 4095 -> 2047, -1 -> 2047,
// 2048 -> 0.
const oob_codes = [_]i32{ 4095, -1, 2048 };
const oob_packed_hex = "ffff3f0000";

fn hexBytes(comptime hex: []const u8) [hex.len / 2]u8 {
    var out: [hex.len / 2]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, hex) catch unreachable;
    return out;
}

test "pack reproduces the reference rvq_pack_codes bytes for [8,5]" {
    const allocator = std.testing.allocator;
    const bytes = try rvq_file.pack(allocator, &golden_codes);
    defer allocator.free(bytes);
    try std.testing.expectEqualSlices(u8, &hexBytes(golden_packed_hex), bytes);
}

test "unpack round-trips the [8,5] golden bytes with K=8" {
    const allocator = std.testing.allocator;
    const codes = try rvq_file.unpack(allocator, &hexBytes(golden_packed_hex), 8);
    defer allocator.free(codes);
    try std.testing.expectEqual(@as(usize, 40), codes.len);
    try std.testing.expectEqual(@as(usize, 5), codes.len / 8);
    try std.testing.expectEqualSlices(i32, &golden_codes, codes);
}

test "pack emits a ceil-sized partial tail byte" {
    const allocator = std.testing.allocator;
    const bytes = try rvq_file.pack(allocator, &tail_codes);
    defer allocator.free(bytes);
    try std.testing.expectEqualSlices(u8, &hexBytes(tail_packed_hex), bytes);

    const codes = try rvq_file.unpack(allocator, bytes, 3);
    defer allocator.free(codes);
    try std.testing.expectEqualSlices(i32, &tail_codes, codes);
}

test "pack masks out-of-range codes to 11 bits like the reference" {
    const allocator = std.testing.allocator;
    const bytes = try rvq_file.pack(allocator, &oob_codes);
    defer allocator.free(bytes);
    try std.testing.expectEqualSlices(u8, &hexBytes(oob_packed_hex), bytes);

    const codes = try rvq_file.unpack(allocator, bytes, 3);
    defer allocator.free(codes);
    try std.testing.expectEqualSlices(i32, &.{ 2047, 2047, 0 }, codes);
}

test "unpack rejects streams whose code count does not divide by K" {
    const allocator = std.testing.allocator;
    const bytes = hexBytes(golden_packed_hex);
    try std.testing.expectError(error.InvalidCodeStream, rvq_file.unpack(allocator, &bytes, 7));
    try std.testing.expectError(error.InvalidCodeStream, rvq_file.unpack(allocator, &.{}, 8));
    try std.testing.expectError(error.InvalidCodeStream, rvq_file.unpack(allocator, &bytes, 0));
}

test "rvq file round-trips through writeFile and readFile" {
    const allocator = std.testing.allocator;
    const path = "/tmp/fucina-omnivoice-rvq-roundtrip.rvq";
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    try rvq_file.writeFile(allocator, std.testing.io, path, &golden_codes);
    const codes = try rvq_file.readFile(allocator, std.testing.io, path, 8);
    defer allocator.free(codes);
    try std.testing.expectEqualSlices(i32, &golden_codes, codes);
}
