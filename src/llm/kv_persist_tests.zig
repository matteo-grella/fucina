//! Behavioral tests for KV-cache persistence (`kv_persist.zig`): bit-exact
//! save/load roundtrip in both cache dtypes, incremental (multi-turn)
//! appends, wholesale rejection of foreign geometry, and the crash-safety
//! contract (a torn tail yields the consistent prefix, never garbage).
const std = @import("std");
const fucina = @import("fucina");

const ExecContext = fucina.ExecContext;
const kv_cache = @import("kv_cache.zig");
const kv_persist = @import("kv_persist.zig");
const KvCache = kv_cache.KvCache;
const KvInput = kv_cache.KvInput;

const n_layers = 3;
const kv_heads = 2;
const head_dim = 32; // q8_0-block aligned
const capacity = 16;

fn tmpPath(buf: []u8, comptime tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "kv_persist_test_{s}_{d}.kv", .{ tag, std.Io.Clock.real.now(std.testing.io).nanoseconds });
}

/// Append `tokens.len` deterministic positions (seeded per position) across
/// all layers, mirroring one model step per token.
fn fillPositions(ctx: *ExecContext, kv: *KvCache, base: usize, count: usize) !void {
    var vals: [kv_heads * head_dim]f32 = undefined;
    for (0..count) |i| {
        const pos = base + i;
        for (0..n_layers) |layer_i| {
            for (&vals, 0..) |*v, j| v.* = @sin(@as(f32, @floatFromInt(pos * 131 + layer_i * 17 + j)) * 0.13) * 2.0;
            var k = try KvInput.fromSlice(ctx, .{ 1, kv_heads, head_dim }, &vals);
            defer k.deinit();
            for (&vals, 0..) |*v, j| v.* = @cos(@as(f32, @floatFromInt(pos * 71 + layer_i * 29 + j)) * 0.19);
            var v = try KvInput.fromSlice(ctx, .{ 1, kv_heads, head_dim }, &vals);
            defer v.deinit();
            try kv.appendLayer(ctx, layer_i, &k, &v);
        }
        kv.advance(1);
    }
}

fn expectCachesEqual(a: *const KvCache, b: *const KvCache) !void {
    try std.testing.expectEqual(a.len, b.len);
    for (0..n_layers) |i| {
        switch (a.dtype) {
            .f16 => {
                try std.testing.expectEqualSlices(f16, try a.kSlice(i, a.len), try b.kSlice(i, b.len));
                try std.testing.expectEqualSlices(f16, try a.vSlice(i, a.len), try b.vSlice(i, b.len));
            },
            .q8_0 => {
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a.kBlocks(i, a.len)), std.mem.sliceAsBytes(b.kBlocks(i, b.len)));
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a.vBlocks(i, a.len)), std.mem.sliceAsBytes(b.vBlocks(i, b.len)));
            },
        }
    }
}

fn roundtripCase(dtype: kv_cache.KvDtype, comptime tag: []const u8) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var path_buf: [128]u8 = undefined;
    const path = try tmpPath(&path_buf, tag);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var kv = try KvCache.initWithDtype(&ctx, n_layers, kv_heads, head_dim, capacity, dtype);
    defer kv.deinit();

    // Turn 1: five positions, persist all.
    try fillPositions(&ctx, &kv, 0, 5);
    const tokens1 = [_]usize{ 11, 22, 33, 44, 55 };
    try kv_persist.appendRange(io, allocator, path, &kv, &tokens1);

    // Turn 2: three more, persist only the delta.
    try fillPositions(&ctx, &kv, 5, 3);
    const tokens2 = [_]usize{ 11, 22, 33, 44, 55, 66, 77, 88 };
    try kv_persist.appendRange(io, allocator, path, &kv, &tokens2);

    // Resume into a fresh cache: same tokens, bit-identical K/V.
    var kv2 = try KvCache.initWithDtype(&ctx, n_layers, kv_heads, head_dim, capacity, dtype);
    defer kv2.deinit();
    const resumed = (try kv_persist.load(io, allocator, path, &kv2)) orelse return error.TestExpectedResume;
    defer allocator.free(resumed);
    try std.testing.expectEqualSlices(usize, &tokens2, resumed);
    try expectCachesEqual(&kv, &kv2);
}

test "kv persistence roundtrip is bit-exact across incremental appends (f16)" {
    try roundtripCase(.f16, "f16");
}

test "kv persistence roundtrip is bit-exact across incremental appends (q8_0)" {
    try roundtripCase(.q8_0, "q8");
}

test "kv persistence rejects foreign geometry and recovers the prefix of a torn append" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var path_buf: [128]u8 = undefined;
    const path = try tmpPath(&path_buf, "torn");
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var kv = try KvCache.init(&ctx, n_layers, kv_heads, head_dim, capacity);
    defer kv.deinit();
    try fillPositions(&ctx, &kv, 0, 4);
    const tokens = [_]usize{ 1, 2, 3, 4 };
    try kv_persist.appendRange(io, allocator, path, &kv, &tokens);

    // Foreign geometry (different head_dim): ignored wholesale.
    var other = try KvCache.init(&ctx, n_layers, kv_heads, head_dim * 2, capacity);
    defer other.deinit();
    try std.testing.expectEqual(@as(?[]usize, null), try kv_persist.load(io, allocator, path, &other));

    // Torn tail: truncate the file mid-record 3 while nrec still says 4 —
    // the load stops at the consistent prefix (3 positions), exactly the
    // state before the interrupted append of position 4.
    {
        var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write });
        defer file.close(io);
        const size = try file.length(io);
        try file.setLength(io, size - 10);
    }
    var kv2 = try KvCache.init(&ctx, n_layers, kv_heads, head_dim, capacity);
    defer kv2.deinit();
    const resumed = (try kv_persist.load(io, allocator, path, &kv2)) orelse return error.TestExpectedResume;
    defer allocator.free(resumed);
    try std.testing.expectEqual(@as(usize, 3), resumed.len);
    try std.testing.expectEqualSlices(usize, tokens[0..3], resumed);
    try std.testing.expectEqual(@as(usize, 3), kv2.len);
    for (0..n_layers) |i| {
        try std.testing.expectEqualSlices(f16, try kv.kSlice(i, 3), try kv2.kSlice(i, 3));
        try std.testing.expectEqualSlices(f16, try kv.vSlice(i, 3), try kv2.vSlice(i, 3));
    }
}
