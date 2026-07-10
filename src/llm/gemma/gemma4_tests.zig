//! Behavioral tests for the Gemma 4 inference module (`gemma4.zig`):
//! per-layer SWA/KV geometry derivation, shared-KV reuse mapping, and the
//! f16-only KV-cache forward-seam guard.
const std = @import("std");
const fucina = @import("fucina");
const kv_cache = @import("../kv_cache.zig");
const gemma4 = @import("gemma4.zig");

const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;
const Error = gemma4.Error;
const deriveGeometry = gemma4.deriveGeometry;
const requireF16KvCache = gemma4.requireF16KvCache;

test "gemma4 per-layer geometry maps an explicit SWA + KV pattern" {
    const allocator = std.testing.allocator;
    const n_layer = 30;
    const global_layers = [_]usize{ 5, 11, 17, 23, 29 };
    var pattern = [_]bool{true} ** n_layer;
    var kv = [_]usize{8} ** n_layer;
    for (global_layers) |g| {
        pattern[g] = false; // global
        kv[g] = 2;
    }

    var geom = try deriveGeometry(allocator, n_layer, &pattern, &kv, 0, 512, 256);
    defer geom.deinit(allocator);

    var globals: usize = 0;
    for (0..n_layer) |il| {
        var is_global = false;
        for (global_layers) |g| {
            if (g == il) is_global = true;
        }
        try std.testing.expectEqual(!is_global, geom.is_swa[il]);
        try std.testing.expectEqual(@as(usize, if (is_global) 512 else 256), geom.head_dim[il]);
        try std.testing.expectEqual(@as(usize, if (is_global) 2 else 8), geom.kv_heads[il]);
        try std.testing.expect(geom.has_kv[il]); // shared_kv_layers = 0
        try std.testing.expectEqual(il, geom.kv_ref[il]);
        if (is_global) globals += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), globals);
}

test "gemma4 rejects a q8_0 KV cache at the forward seam" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var q8 = try KvCache.initWithDtype(&ctx, 1, 2, 64, 4, .q8_0);
    defer q8.deinit();
    try std.testing.expectError(Error.UnsupportedKvCacheDtype, requireF16KvCache(&q8));

    var f16_cache = try KvCache.initWithDtype(&ctx, 1, 2, 64, 4, .f16);
    defer f16_cache.deinit();
    try requireF16KvCache(&f16_cache);
}

test "gemma4 shared-KV reuse map is in range" {
    const allocator = std.testing.allocator;
    var pattern = [_]bool{ true, true, false, true, true, true };
    var kv = [_]usize{ 8, 8, 2, 8, 8, 8 };
    var geom = try deriveGeometry(allocator, 6, &pattern, &kv, 2, 256, 128);
    defer geom.deinit(allocator);
    for (0..4) |il| try std.testing.expect(geom.has_kv[il] and geom.kv_ref[il] == il);
    for (4..6) |il| {
        try std.testing.expect(!geom.has_kv[il]);
        try std.testing.expect(geom.kv_ref[il] < 4);
    }
}
