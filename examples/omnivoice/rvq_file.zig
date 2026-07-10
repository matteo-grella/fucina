//! Packed RVQ code stream (.rvq) file I/O. Ports
//! refs/omnivoice.cpp/src/rvq-file.h (rvq_pack_codes, rvq_unpack_codes,
//! rvq_read_file, rvq_write_file) with RVQ_CODE_BITS = 11 from
//! refs/omnivoice.cpp/tools/omnivoice-codec.cpp.
//!
//! Headerless bit stream, `code_bits` per code LSB-first (acc |= code << bits,
//! low bytes emitted first); layout [K, T] row-major k-slow. The read side
//! infers T = filesize*8 / (K*code_bits) and requires the code count to divide
//! exactly by K.

const std = @import("std");

/// 11 bits per code (codebook size V <= 2048).
pub const code_bits = 11;

pub const Error = error{InvalidCodeStream};

/// Packs a flat code stream at `code_bits` per code, LSB-first. Codes are
/// masked to `code_bits` bits like the reference. Output length is
/// ceil(codes.len * code_bits / 8); caller frees with the same allocator.
pub fn pack(allocator: std.mem.Allocator, codes: []const i32) ![]u8 {
    const mask: u32 = (1 << code_bits) - 1;
    const total_bits = codes.len * code_bits;
    const out = try allocator.alloc(u8, (total_bits + 7) / 8);

    var acc: u64 = 0;
    var bits_in_acc: u6 = 0;
    var out_pos: usize = 0;
    for (codes) |c| {
        acc |= @as(u64, @as(u32, @bitCast(c)) & mask) << bits_in_acc;
        bits_in_acc += code_bits;
        while (bits_in_acc >= 8) {
            out[out_pos] = @truncate(acc);
            out_pos += 1;
            acc >>= 8;
            bits_in_acc -= 8;
        }
    }
    if (bits_in_acc > 0) {
        out[out_pos] = @truncate(acc);
        out_pos += 1;
    }
    std.debug.assert(out_pos == out.len);
    return out;
}

/// Unpacks a whole packed byte stream into K*T codes. The code count is
/// inferred from the byte length (n = len*8 / code_bits) and must be nonzero
/// and divide exactly by `k`. Caller frees with the same allocator.
pub fn unpack(allocator: std.mem.Allocator, bytes: []const u8, k: usize) ![]i32 {
    if (k == 0) return Error.InvalidCodeStream;
    const n_codes = bytes.len * 8 / code_bits;
    if (n_codes == 0 or n_codes % k != 0) return Error.InvalidCodeStream;
    return unpackCodes(allocator, bytes, n_codes);
}

/// Reference rvq_unpack_codes loop. `n_codes` must fit in the bit budget of
/// `bytes` (guaranteed by `unpack`'s size derivation).
fn unpackCodes(allocator: std.mem.Allocator, bytes: []const u8, n_codes: usize) ![]i32 {
    const mask: u32 = (1 << code_bits) - 1;
    const out = try allocator.alloc(i32, n_codes);

    var acc: u64 = 0;
    var bits_in_acc: u6 = 0;
    var in_pos: usize = 0;
    for (out) |*code| {
        while (bits_in_acc < code_bits and in_pos < bytes.len) {
            acc |= @as(u64, bytes[in_pos]) << bits_in_acc;
            in_pos += 1;
            bits_in_acc += 8;
        }
        code.* = @intCast(@as(u32, @truncate(acc)) & mask);
        acc >>= code_bits;
        bits_in_acc -= code_bits;
    }
    return out;
}

/// Reads a .rvq file and unpacks it into K*T codes (T = codes.len / k).
/// Caller frees with the same allocator.
pub fn readFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, k: usize) ![]i32 {
    const bytes = try readFileBytes(allocator, io, path);
    defer allocator.free(bytes);
    return unpack(allocator, bytes, k);
}

/// Packs and writes a .rvq file.
pub fn writeFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, codes: []const i32) !void {
    const packed_bytes = try pack(allocator, codes);
    defer allocator.free(packed_bytes);

    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var write_buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &write_buffer);
    try writer.interface.writeAll(packed_bytes);
    try writer.interface.flush();
}

fn readFileBytes(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
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

test {
    _ = @import("rvq_file_tests.zig");
}
