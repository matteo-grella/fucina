//! Tests for parakeet/weights.zig. The guarded f32 borrow helper `borrowF32`
//! is the single home for the four parakeet f32 byte-cast helpers; it must
//! reject misaligned / odd-length untrusted GGUF bytes instead of being UB at
//! the `@alignCast` under ReleaseFast.
const std = @import("std");
const fucina = @import("fucina");
const pweights = @import("weights.zig");

test "borrowF32 rejects misaligned and odd-length GGUF bytes" {
    var buf: [16]u8 align(4) = undefined;
    for (&buf, 0..) |*b, i| b.* = @intCast(i);

    // 2-byte-misaligned start (4-aligned buf, +2): a multiple-of-4 length but a
    // non-f32-aligned pointer — would be illegal behaviour at the @alignCast.
    try std.testing.expectError(error.InvalidWeightShape, pweights.borrowF32(buf[2..14]));
    // odd length (not a multiple of @sizeOf(f32)).
    try std.testing.expectError(error.InvalidWeightShape, pweights.borrowF32(buf[0..10]));
    // valid: 4-aligned start + multiple-of-4 length -> 3 f32 elements.
    const ok = try pweights.borrowF32(buf[0..12]);
    try std.testing.expectEqual(@as(usize, 3), ok.len);
}

test "resident byte registry: non-gpu identity + no growth; gpu cache-hit; deinit leak-free (G2.2)" {
    var ctx: fucina.ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();
    // The resident registry touches only ctx.allocator (not `file`), so a
    // direct minimal struct suffices — no synthetic GGUF needed. `defer deinit`
    // under the testing allocator is the registry-cleanup check; GPU builds also
    // release any resident shim buffer slot.
    var pw: pweights.ParakeetWeights = .{
        .ctx = &ctx,
        .file = undefined,
        .allocator = std.testing.allocator,
        .resident = @import("../weights.zig").ResidentByteRegistry.init(std.testing.allocator),
    };
    defer pw.deinit();

    const src = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const a = pw.resident.bytes(&src);
    const b = pw.resident.bytes(&src); // second call: stable buffer-key must hit the cache

    if (comptime !fucina.internal.gpu.enabled) {
        try std.testing.expectEqual(@intFromPtr(&src[0]), @intFromPtr(a.ptr)); // identity
        try std.testing.expectEqual(@as(usize, 0), pw.resident.map.count()); // registry stays empty
    } else {
        // gpu build: same buffer on the 2nd call (a device copy when a GPU is present,
        // or both fall back to `src` in a headless environment — either way, stable).
        try std.testing.expectEqual(@intFromPtr(a.ptr), @intFromPtr(b.ptr));
    }
}
