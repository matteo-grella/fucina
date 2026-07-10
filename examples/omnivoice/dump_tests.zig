//! Golden tests for dump.zig against the C++ reference debug.h.
//!
//! Golden bytes and f64 bit patterns produced by scratchpad/golden_dump.cpp
//! (clang++ -O2, includes refs/omnivoice.cpp/src/debug.h directly): a [2,3]
//! debug_dump, an [8] debug_dump_i32_as_f32, and debug_cosine_sim values.

const std = @import("std");
const dump = @import("dump.zig");

// debug_dump(shape=[2,3], {0.0, -0.0, 1.5, -2.25, 3.14159265f, FLT_MIN}).
const golden_f32_hex = "02000000020000000300000000000000000000800000c03f000010c0db0f494000008000";
const golden_f32_bits = [_]u32{ 0x00000000, 0x80000000, 0x3fc00000, 0xc0100000, 0x40490fdb, 0x00800000 };

// debug_dump_i32_as_f32(shape=[8], {0, 1, -1, 1023, 2047, 16777217, -2147483648, 2147483647}).
const golden_i32_hex = "0100000008000000000000000000803f000080bf00c07f4400e0ff440000804b000000cf0000004f";
const golden_i32_values = [_]i32{ 0, 1, -1, 1023, 2047, 16777217, -2147483648, 2147483647 };

const vec_a = [_]f32{ 1.0, 2.0, -3.5, 0.25, 4.0 };
const vec_b = [_]f32{ 0.5, -1.0, 2.0, 8.0, -0.125 };
const vec_nz = [_]f32{ 1.0, 2.0, 3.0 };

fn hexBytes(comptime hex: []const u8) [hex.len / 2]u8 {
    var out: [hex.len / 2]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, hex) catch unreachable;
    return out;
}

fn readAllBytes(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const io = std.testing.io;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
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

fn writeAllBytes(path: []const u8, bytes: []const u8) !void {
    const io = std.testing.io;
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var writer = file.writer(io, &buf);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
}

test "writeFile reproduces the reference debug_dump bytes" {
    const allocator = std.testing.allocator;
    const path = "/tmp/fucina-omnivoice-dump-f32-test.bin";
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    var data: [6]f32 = undefined;
    for (&data, golden_f32_bits) |*d, bits| d.* = @bitCast(bits);
    try dump.writeFile(std.testing.io, path, &.{ 2, 3 }, &data);

    const got = try readAllBytes(allocator, path);
    defer allocator.free(got);
    try std.testing.expectEqualSlices(u8, &hexBytes(golden_f32_hex), got);
}

test "writeI32AsF32File reproduces the reference debug_dump_i32_as_f32 bytes" {
    const allocator = std.testing.allocator;
    const path = "/tmp/fucina-omnivoice-dump-i32-test.bin";
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    try dump.writeI32AsF32File(std.testing.io, path, &.{8}, &golden_i32_values);

    const got = try readAllBytes(allocator, path);
    defer allocator.free(got);
    try std.testing.expectEqualSlices(u8, &hexBytes(golden_i32_hex), got);
}

test "readFile parses the reference debug_dump bytes bit-exactly" {
    const allocator = std.testing.allocator;
    const path = "/tmp/fucina-omnivoice-dump-read-test.bin";
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    try writeAllBytes(path, &hexBytes(golden_f32_hex));

    const d = try dump.readFile(allocator, std.testing.io, path);
    defer allocator.free(d.shape);
    defer allocator.free(d.data);
    try std.testing.expectEqualSlices(i32, &.{ 2, 3 }, d.shape);
    try std.testing.expectEqual(@as(usize, 6), d.data.len);
    for (d.data, golden_f32_bits) |v, bits| {
        try std.testing.expectEqual(bits, @as(u32, @bitCast(v)));
    }
}

test "parse rejects corrupt dumps" {
    const allocator = std.testing.allocator;
    const golden = hexBytes(golden_f32_hex);
    try std.testing.expectError(error.CorruptDump, dump.parse(allocator, golden[0..2]));
    try std.testing.expectError(error.CorruptDump, dump.parse(allocator, golden[0 .. golden.len - 4]));

    var negative_ndims = golden;
    negative_ndims[3] = 0x80;
    try std.testing.expectError(error.CorruptDump, dump.parse(allocator, &negative_ndims));

    var negative_dim = golden;
    negative_dim[7] = 0x80;
    try std.testing.expectError(error.CorruptDump, dump.parse(allocator, &negative_dim));
}

test "writeFile rejects a shape/data length mismatch" {
    try std.testing.expectError(
        error.ShapeMismatch,
        dump.writeFile(std.testing.io, "/tmp/fucina-omnivoice-dump-mismatch.bin", &.{ 2, 3 }, vec_a[0..5]),
    );
}

test "compare matches the reference debug_cosine_sim goldens" {
    const stats = dump.compare(&vec_a, &vec_b);
    try std.testing.expectEqual(@as(u64, 0xbfc2a72214120b7b), @as(u64, @bitCast(stats.cosine)));
    try std.testing.expectEqual(@as(u64, 0x401f000000000000), @as(u64, @bitCast(stats.max_abs_diff)));
    try std.testing.expectEqual(@as(u64, 0x4010b33333333333), @as(u64, @bitCast(stats.mean_abs_diff)));
    try std.testing.expectEqual(@as(usize, 5), stats.len_a);
    try std.testing.expectEqual(@as(usize, 5), stats.len_b);
}

test "compare runs over min(len) like the reference tooling" {
    const stats = dump.compare(&vec_a, &vec_nz);
    try std.testing.expectEqual(@as(u64, 0xbfd6a69cb5e7bcdb), @as(u64, @bitCast(stats.cosine)));
    try std.testing.expectEqual(@as(u64, 0x401a000000000000), @as(u64, @bitCast(stats.max_abs_diff)));
    try std.testing.expectEqual(@as(u64, 0x4001555555555555), @as(u64, @bitCast(stats.mean_abs_diff)));
    try std.testing.expectEqual(@as(usize, 5), stats.len_a);
    try std.testing.expectEqual(@as(usize, 3), stats.len_b);
}

test "compare of a zero vector has cosine 0" {
    const zero = [_]f32{ 0.0, 0.0, 0.0 };
    const stats = dump.compare(&zero, &vec_nz);
    try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(stats.cosine)));
    try std.testing.expectEqual(@as(f64, 3.0), stats.max_abs_diff);
    try std.testing.expectEqual(@as(f64, 2.0), stats.mean_abs_diff);
}

test "compare of empty inputs is all zeros" {
    const stats = dump.compare(&.{}, &.{});
    try std.testing.expectEqual(@as(f64, 0.0), stats.cosine);
    try std.testing.expectEqual(@as(f64, 0.0), stats.max_abs_diff);
    try std.testing.expectEqual(@as(f64, 0.0), stats.mean_abs_diff);
    try std.testing.expectEqual(@as(usize, 0), stats.len_a);
    try std.testing.expectEqual(@as(usize, 0), stats.len_b);
}

test "dump file round-trips through writeFile and readFile" {
    const allocator = std.testing.allocator;
    const path = "/tmp/fucina-omnivoice-dump-roundtrip.bin";
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    const shape = [_]i32{ 1, 2, 2 };
    const data = [_]f32{ -1.25, 0.0, 42.0, 3.5 };
    try dump.writeFile(std.testing.io, path, &shape, &data);

    const d = try dump.readFile(allocator, std.testing.io, path);
    defer allocator.free(d.shape);
    defer allocator.free(d.data);
    try std.testing.expectEqualSlices(i32, &shape, d.shape);
    try std.testing.expectEqualSlices(f32, &data, d.data);
}
