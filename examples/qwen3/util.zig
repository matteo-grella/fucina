//! Shared plumbing for the Qwen3 runner modules: clocks, small file I/O,
//! tokenizer round-trips, and the KV-cache banner.
const std = @import("std");
const models = @import("fucina_models");

pub fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

pub fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
}

pub fn millis(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

pub fn millisI128(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

pub fn printCacheInfo(stdout: anytype, cache: *const models.text.kv_cache.KvCache) !void {
    const bytes = cache.byteSize();
    const per_token = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(cache.capacity));
    try stdout.print("kv cache: {s}, capacity {d} tok, {d:.1} MiB ({d:.1} KiB/token, {d:.1} MiB per 1k tok)\n", .{
        @tagName(cache.dtype),
        cache.capacity,
        @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0),
        per_token / 1024.0,
        per_token * 1000.0 / (1024.0 * 1024.0),
    });
}

pub fn writeLogits(io: std.Io, path: []const u8, values: []const f32) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);

    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    defer writer.interface.flush() catch {};
    try writer.interface.writeAll(std.mem.sliceAsBytes(values));
}

pub fn writeTokenIdsU32(io: std.Io, allocator: std.mem.Allocator, path: []const u8, prompt: []const usize, generated: []const usize) !void {
    const ids = try allocator.alloc(u32, prompt.len + generated.len);
    defer allocator.free(ids);
    for (prompt, ids[0..prompt.len]) |src, *dst| dst.* = @intCast(src);
    for (generated, ids[prompt.len..]) |src, *dst| dst.* = @intCast(src);
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(ids));
}

/// Resolve a grammar flag value: `@PATH` reads the file (ownership parked in
/// `owned` for the caller's deferred free); anything else is the inline text.
pub fn grammarValue(io: std.Io, allocator: std.mem.Allocator, value: []const u8, owned: *?[]u8) ![]const u8 {
    if (!std.mem.startsWith(u8, value, "@")) return value;
    const text = try readTextFile(io, allocator, value[1..]);
    owned.* = text;
    return text;
}

/// Read a small text file (grammar/schema/reference documents), with a size
/// cap so a mistyped path (a GGUF, a core dump, ...) fails fast and clearly
/// instead of ballooning memory.
pub fn readTextFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.IsDir;
    const max_bytes = 64 * 1024 * 1024;
    if (stat.size > max_bytes) return error.FileTooLarge;
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

/// Read a UTF-8 text file and tokenize it (no BOS/EOS policy) for use as a
/// speculation reference document or the `--tokenize` parity harness.
pub fn tokenizeFile(io: std.Io, allocator: std.mem.Allocator, tok: *const models.text.tokenizer.Tokenizer, path: []const u8) ![]usize {
    const bytes = try readTextFile(io, allocator, path);
    defer allocator.free(bytes);
    const ids32 = try tok.encodeRaw(allocator, bytes);
    defer allocator.free(ids32);
    const ids = try allocator.alloc(usize, ids32.len);
    errdefer allocator.free(ids);
    for (ids, ids32) |*d, s| d.* = s;
    return ids;
}

pub fn decodeIds(allocator: std.mem.Allocator, tok: *const models.text.tokenizer.Tokenizer, ids: []const usize) ![]u8 {
    const ids32 = try allocator.alloc(u32, ids.len);
    defer allocator.free(ids32);
    for (ids32, ids) |*d, s| d.* = @intCast(s);
    return tok.decode(allocator, ids32);
}
