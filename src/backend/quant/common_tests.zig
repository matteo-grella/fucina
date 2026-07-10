//! Behavioral tests for the shared quant primitives (`common.zig`): the
//! portable u8xi8 / i8xi8 dot constructions validated against scalar references
//! across the nibble, stress, and saturation-edge domains.
const std = @import("std");
const common = @import("common.zig");

const dotU8I8x16Portable = common.dotU8I8x16Portable;
const dotI8x16Portable = common.dotI8x16Portable;

fn refDotU8I8(a: *const [16]u8, b: *const [16]i8) i32 {
    var s: i32 = 0;
    for (a, b) |x, y| s += @as(i32, x) * @as(i32, y);
    return s;
}

fn refDotI8I8(a: *const [16]i8, b: *const [16]i8) i32 {
    var s: i32 = 0;
    for (a, b) |x, y| s += @as(i32, x) * @as(i32, y);
    return s;
}

test "dotU8I8x16Portable matches the scalar reference (nibble + stress domains)" {
    // Deterministic random coverage. Two domains, both saturation-free by the
    // proofs above: nibble weights (the Q4_K call-site domain) against the
    // FULL i8 range incl. -128/127, and a <= 127 against the full i8 range.
    var prng = std.Random.DefaultPrng.init(0x9e3779b97f4a7c15);
    const random = prng.random();
    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        var aa: [16]u8 = undefined;
        var bb: [16]i8 = undefined;
        for (&aa) |*x| x.* = random.uintLessThan(u8, 16); // 0..15
        for (&bb) |*y| y.* = @bitCast(random.int(u8)); // -128..127
        try std.testing.expectEqual(refDotU8I8(&aa, &bb), dotU8I8x16Portable(aa, bb));
        for (&aa) |*x| x.* = random.uintLessThan(u8, 128); // 0..127
        try std.testing.expectEqual(refDotU8I8(&aa, &bb), dotU8I8x16Portable(aa, bb));
    }
    // Saturation-edge stress: pair sums of exactly ±32512 (the maddubs i16
    // ceiling is 32767; these must come through exact, not clamped).
    const a_edge: [16]u8 = @splat(128);
    const b_hi: [16]i8 = @splat(127);
    const b_lo: [16]i8 = @splat(-127);
    try std.testing.expectEqual(@as(i32, 16 * 128 * 127), dotU8I8x16Portable(a_edge, b_hi));
    try std.testing.expectEqual(@as(i32, 16 * 128 * -127), dotU8I8x16Portable(a_edge, b_lo));
    // -128 on the signed side with nibble weights (the Q4_K worst case).
    const a_nib: [16]u8 = @splat(15);
    const b_min: [16]i8 = @splat(-128);
    try std.testing.expectEqual(@as(i32, 16 * 15 * -128), dotU8I8x16Portable(a_nib, b_min));
}

test "dotI8x16Portable sign-trick domain: extremes and saturation stress" {
    // The sign-trick exactness domain: a unrestricted (incl. -128), b in
    // [-127,127] — exactly what every kernel call site guarantees (b is
    // quantizeToI8 output). On non-AVX2 targets this trivially exercises the
    // plain/bias forms; cross-compiled to x86-64-v3 and run under qemu it
    // executes the vpsignb/vpmaddubsw path against the same expectations.
    var prng = std.Random.DefaultPrng.init(0xdeadbeefcafef00d);
    const random = prng.random();
    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        var aa: [16]i8 = undefined;
        var bb: [16]i8 = undefined;
        for (&aa) |*x| x.* = @bitCast(random.int(u8)); // -128..127
        for (&bb) |*y| y.* = @intCast(@as(i32, random.uintLessThan(u8, 255)) - 127); // -127..127
        try std.testing.expectEqual(refDotI8I8(&aa, &bb), dotI8x16Portable(aa, bb));
    }
    // Extremes: a = -128 lanes against b = ±127 produce pair sums of ±32512 —
    // the exact saturation-free maxima of the maddubs construction.
    const a_min: [16]i8 = @splat(-128);
    const b_hi: [16]i8 = @splat(127);
    const b_lo: [16]i8 = @splat(-127);
    try std.testing.expectEqual(@as(i32, 16 * -128 * 127), dotI8x16Portable(a_min, b_hi));
    try std.testing.expectEqual(@as(i32, 16 * -128 * -127), dotI8x16Portable(a_min, b_lo));
    // Mixed signs alternating, worst-case magnitude per pair.
    var am: [16]i8 = undefined;
    var bm: [16]i8 = undefined;
    for (&am, 0..) |*x, i| x.* = if (i % 2 == 0) -128 else 127;
    for (&bm, 0..) |*y, i| y.* = if (i % 3 == 0) -127 else 127;
    try std.testing.expectEqual(refDotI8I8(&am, &bm), dotI8x16Portable(am, bm));
}
