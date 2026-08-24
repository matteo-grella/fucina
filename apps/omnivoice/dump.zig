//! Tensor dump file I/O + compare helper for parity validation against the
//! C++ reference. Ports refs/omnivoice.cpp/src/debug.h (debug_dump,
//! debug_dump_i32_as_f32, debug_cosine_sim).
//!
//! File format (little-endian): [ndims:i32][shape:i32 x ndims][data:f32 x prod(shape)].

const std = @import("std");

const wav = @import("wav.zig");

pub const Error = error{ CorruptDump, ShapeMismatch };

pub const Dump = struct {
    shape: []i32,
    data: []f32,
};

/// Reads and parses a dump file. Caller frees `.shape` and `.data` with the
/// same allocator.
pub fn readFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Dump {
    const bytes = try wav.readFileBytes(io, allocator, path);
    defer allocator.free(bytes);
    return parse(allocator, bytes);
}

/// Parses a whole dump file image from memory. Requires the byte length to
/// match the header exactly.
pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Dump {
    if (bytes.len < 4) return Error.CorruptDump;
    const ndims_raw = std.mem.readInt(i32, bytes[0..4], .little);
    if (ndims_raw < 0) return Error.CorruptDump;
    const ndims: usize = @intCast(ndims_raw);
    if ((bytes.len - 4) / 4 < ndims) return Error.CorruptDump;

    const shape = try allocator.alloc(i32, ndims);
    errdefer allocator.free(shape);
    var numel: usize = 1;
    for (shape, 0..) |*dim, i| {
        const v = std.mem.readInt(i32, bytes[4 + i * 4 ..][0..4], .little);
        if (v < 0) return Error.CorruptDump;
        dim.* = v;
        numel = std.math.mul(usize, numel, @intCast(v)) catch return Error.CorruptDump;
    }
    const body = bytes[4 + ndims * 4 ..];
    if (body.len != numel * 4) return Error.CorruptDump;

    const data = try allocator.alloc(f32, numel);
    for (data, 0..) |*dst, i| {
        dst.* = @bitCast(std.mem.readInt(u32, body[i * 4 ..][0..4], .little));
    }
    return .{ .shape = shape, .data = data };
}

/// Writes a dump file, mirroring the reference debug_dump byte layout.
/// `data.len` must equal the product of `shape`.
pub fn writeFile(io: std.Io, path: []const u8, shape: []const i32, data: []const f32) !void {
    if (numelOf(shape) != data.len) return Error.ShapeMismatch;

    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var write_buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &write_buffer);
    const w = &writer.interface;

    try writeHeader(w, shape);
    for (data) |v| {
        try w.writeInt(u32, @bitCast(v), .little);
    }
    try w.flush();
}

/// Writes a dump file with each i32 value cast to f32, mirroring the
/// reference debug_dump_i32_as_f32 (C `(float)` cast = round to nearest).
pub fn writeI32AsF32File(io: std.Io, path: []const u8, shape: []const i32, values: []const i32) !void {
    if (numelOf(shape) != values.len) return Error.ShapeMismatch;

    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var write_buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &write_buffer);
    const w = &writer.interface;

    try writeHeader(w, shape);
    for (values) |v| {
        const f: f32 = @floatFromInt(v);
        try w.writeInt(u32, @bitCast(f), .little);
    }
    try w.flush();
}

fn writeHeader(w: *std.Io.Writer, shape: []const i32) !void {
    try w.writeInt(i32, @intCast(shape.len), .little);
    for (shape) |s| {
        try w.writeInt(i32, s, .little);
    }
}

fn numelOf(shape: []const i32) ?usize {
    var numel: usize = 1;
    for (shape) |s| {
        if (s < 0) return null;
        numel = std.math.mul(usize, numel, @intCast(s)) catch return null;
    }
    return numel;
}

pub const CompareStats = struct {
    cosine: f64,
    max_abs_diff: f64,
    mean_abs_diff: f64,
    len_a: usize,
    len_b: usize,
};

/// Compares two f32 arrays over the first min(len) elements with f64
/// accumulation, matching the reference debug_cosine_sim (cosine of a
/// zero-norm vector — threshold 1e-30 on the squared norms — is 0).
pub fn compare(a: []const f32, b: []const f32) CompareStats {
    const n = @min(a.len, b.len);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var max_abs: f64 = 0;
    var sum_abs: f64 = 0;
    for (0..n) |i| {
        const av: f64 = a[i];
        const bv: f64 = b[i];
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
        const diff = @abs(av - bv);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
    }
    const cosine: f64 = if (na < 1e-30 or nb < 1e-30) 0.0 else dot / (@sqrt(na) * @sqrt(nb));
    const mean_abs: f64 = if (n == 0) 0.0 else sum_abs / @as(f64, @floatFromInt(n));
    return .{
        .cosine = cosine,
        .max_abs_diff = max_abs,
        .mean_abs_diff = mean_abs,
        .len_a = a.len,
        .len_b = b.len,
    };
}

test {
    _ = @import("dump_tests.zig");
}
