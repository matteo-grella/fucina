//! Fake-quant grid parity tests. The arithmetic RNE rounding in
//! `fakequant.zig` is pinned bit-for-bit against literal grid-search oracles
//! (ported verbatim from the deepseek4 host implementation it replaced):
//! every grid boundary, midpoints ± ulps, dense randoms, and the group
//! kernels on reference-shaped rows. Force-imported by `fakequant.zig`'s
//! `test` block. Excluded from arch-check (a `_tests.zig` file).

const std = @import("std");

const fakequant = @import("fakequant.zig");

// ---------------------------------------------------------------------------
// Oracles: the original grid-search round trips, verbatim.
// ---------------------------------------------------------------------------

fn e4m3Value(i: i32) f32 {
    const exp_scale = [16]f32{ 0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0 };
    const exp: usize = @intCast((i >> 3) & 0x0f);
    const mant: f32 = @floatFromInt(i & 0x07);
    return if (exp == 0) mant * 0.001953125 else (1.0 + mant * 0.125) * exp_scale[exp];
}

fn e4m3RoundOracle(x: f32) f32 {
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const ax = @min(@abs(x), 448.0);
    var lo: i32 = 0;
    var hi: i32 = 126;
    while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (e4m3Value(mid) <= ax) lo = mid else hi = mid - 1;
    }
    var best = lo;
    if (best < 126) {
        const best_diff = @abs(ax - e4m3Value(best));
        const next_diff = @abs(ax - e4m3Value(best + 1));
        if (next_diff < best_diff or (next_diff == best_diff and ((best + 1) & 1) == 0 and (best & 1) != 0)) {
            best += 1;
        }
    }
    return sign * e4m3Value(best);
}

fn e2m1Value(i: usize) f32 {
    const values = [8]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };
    return values[i & 7];
}

fn e2m1RoundOracle(x: f32) f32 {
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const ax = @min(@abs(x), 6.0);
    var best: usize = 0;
    var best_diff = @abs(ax - e2m1Value(0));
    for (1..8) |i| {
        const diff = @abs(ax - e2m1Value(i));
        if (diff < best_diff or (diff == best_diff and (i & 1) == 0 and (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * e2m1Value(best);
}

fn fp8GroupOracle(x: []f32, group: usize, amax_floor: f32) void {
    var off: usize = 0;
    while (off < x.len) : (off += group) {
        var amax: f32 = 0;
        for (x[off..][0..group]) |v| amax = @max(amax, @abs(v));
        if (amax < amax_floor) amax = amax_floor;
        const scale = std.math.ldexp(@as(f32, 1.0), @intFromFloat(@ceil(@log2(amax / 448.0))));
        for (x[off..][0..group]) |*v| {
            const clamped = @min(@max(v.* / scale, -448.0), 448.0);
            v.* = e4m3RoundOracle(clamped) * scale;
        }
    }
}

fn fp4GroupOracle(x: []f32, group: usize, amax_floor: f32) void {
    var off: usize = 0;
    while (off < x.len) : (off += group) {
        var amax: f32 = 0;
        for (x[off..][0..group]) |v| amax = @max(amax, @abs(v));
        if (amax < amax_floor) amax = amax_floor;
        const scale = std.math.ldexp(@as(f32, 1.0), @intFromFloat(@ceil(@log2(amax / 6.0))));
        for (x[off..][0..group]) |*v| {
            const clamped = @min(@max(v.* / scale, -6.0), 6.0);
            v.* = e2m1RoundOracle(clamped) * scale;
        }
    }
}

fn hadamard128Oracle(x: *[128]f32) void {
    var stride: usize = 1;
    while (stride < 128) : (stride <<= 1) {
        var base: usize = 0;
        while (base < 128) : (base += 2 * stride) {
            for (0..stride) |i| {
                const a = x[base + i];
                const b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
    }
    const scale = 0.08838834764831845;
    for (x) |*v| v.* *= scale;
}

// ---------------------------------------------------------------------------
// Scalar rounding: bit parity with the oracles.
// ---------------------------------------------------------------------------

fn expectBitEqual(expected: f32, actual: f32) !void {
    try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(actual)));
}

test "roundE4m3 matches the grid-search oracle at every boundary and midpoint" {
    // Every grid value, every midpoint, and ± a few ulps around each.
    for (0..127) |i| {
        const gv = e4m3Value(@intCast(i));
        var probes: [8]f32 = .{ gv, 0, 0, 0, 0, 0, 0, 0 };
        if (i < 126) {
            const next = e4m3Value(@intCast(i + 1));
            const mid = (gv + next) / 2.0;
            probes[1] = mid;
            probes[2] = std.math.nextAfter(f32, mid, 0);
            probes[3] = std.math.nextAfter(f32, mid, 1000);
            probes[4] = std.math.nextAfter(f32, gv, 1000);
            probes[5] = std.math.nextAfter(f32, next, 0);
        }
        probes[6] = gv * 1.0001;
        probes[7] = gv * 0.9999;
        for (probes) |p| {
            try expectBitEqual(e4m3RoundOracle(p), fakequant.roundE4m3(p));
            try expectBitEqual(e4m3RoundOracle(-p), fakequant.roundE4m3(-p));
        }
    }
    // Saturation + zero.
    for ([_]f32{ 0.0, -0.0, 448.0, 449.0, 1.0e9, std.math.floatMin(f32), std.math.floatMax(f32) }) |p| {
        try expectBitEqual(e4m3RoundOracle(p), fakequant.roundE4m3(p));
    }
}

test "roundE2m1 matches the grid-search oracle at every boundary and midpoint" {
    for (0..8) |i| {
        const gv = e2m1Value(i);
        var probes: [8]f32 = .{ gv, 0, 0, 0, 0, 0, 0, 0 };
        if (i < 7) {
            const next = e2m1Value(i + 1);
            const mid = (gv + next) / 2.0;
            probes[1] = mid;
            probes[2] = std.math.nextAfter(f32, mid, 0);
            probes[3] = std.math.nextAfter(f32, mid, 100);
            probes[4] = std.math.nextAfter(f32, gv, 100);
            probes[5] = std.math.nextAfter(f32, next, 0);
        }
        probes[6] = gv * 1.0001;
        probes[7] = gv * 0.9999;
        for (probes) |p| {
            try expectBitEqual(e2m1RoundOracle(p), fakequant.roundE2m1(p));
            try expectBitEqual(e2m1RoundOracle(-p), fakequant.roundE2m1(-p));
        }
    }
    for ([_]f32{ 0.0, -0.0, 6.0, 6.5, 7.0, 1.0e9 }) |p| {
        try expectBitEqual(e2m1RoundOracle(p), fakequant.roundE2m1(p));
    }
}

test "rounding matches the oracle on dense randoms across the whole range" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const random = prng.random();
    for (0..200_000) |_| {
        // Log-uniform magnitude covering denormal-of-grid through saturation.
        const mag = std.math.pow(f32, 10.0, random.float(f32) * 9.0 - 6.0);
        const v = if (random.boolean()) mag else -mag;
        try expectBitEqual(e4m3RoundOracle(v), fakequant.roundE4m3(v));
        try expectBitEqual(e2m1RoundOracle(@min(v, 100.0)), fakequant.roundE2m1(@min(v, 100.0)));
    }
}

// ---------------------------------------------------------------------------
// Group kernels: bit parity on reference-shaped rows (SIMD + scalar tails).
// ---------------------------------------------------------------------------

test "groupRoundTripE4m3InPlace matches the reference row recipe bit-for-bit" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    // 448-wide row of 64-groups (the deepseek4 KV row shape) + a 24-wide
    // 8-group row exercising the scalar tail (group smaller than the vector).
    inline for (.{ .{ 448, 64 }, .{ 24, 8 }, .{ 12, 4 } }) |shape| {
        const len = shape[0];
        const group = shape[1];
        for (0..200) |round_i| {
            var row: [len]f32 = undefined;
            for (&row) |*v| {
                const mag = std.math.pow(f32, 10.0, random.float(f32) * 8.0 - 5.0);
                v.* = if (random.boolean()) mag else -mag;
            }
            if (round_i == 0) @memset(&row, 0); // all-zero group: floor path
            var expected = row;
            fp8GroupOracle(&expected, group, 1.0e-4);
            fakequant.groupRoundTripE4m3InPlace(&row, group, 1.0e-4);
            for (expected, row) |e, a| try expectBitEqual(e, a);
            // No idempotence assertion: the recipe is not a projection — a
            // second pass may pick a smaller power-of-two scale (the group
            // amax shrinks through rounding) and re-clamp near the grid max.
        }
    }
}

test "groupRoundTripE2m1InPlace matches the reference row recipe bit-for-bit" {
    var prng = std.Random.DefaultPrng.init(43);
    const random = prng.random();
    const floor: f32 = 7.052966104933725e-38;
    for (0..200) |round_i| {
        var row: [128]f32 = undefined;
        for (&row) |*v| {
            const mag = std.math.pow(f32, 10.0, random.float(f32) * 8.0 - 5.0);
            v.* = if (random.boolean()) mag else -mag;
        }
        if (round_i == 0) @memset(&row, 0);
        var expected = row;
        fp4GroupOracle(&expected, 32, floor);
        fakequant.groupRoundTripE2m1InPlace(&row, 32, floor);
        for (expected, row) |e, a| try expectBitEqual(e, a);
    }
}

// ---------------------------------------------------------------------------
// Hadamard + f16 round trip.
// ---------------------------------------------------------------------------

test "hadamardInPlace matches the 128-wide reference and involutes" {
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    var row: [128]f32 = undefined;
    for (&row) |*v| v.* = random.floatNorm(f32) * 10.0;

    var expected = row;
    hadamard128Oracle(&expected);
    var actual = row;
    fakequant.hadamardInPlace(&actual);
    for (expected, actual) |e, a| try expectBitEqual(e, a);

    // Involution up to f32 rounding, at a non-reference length too.
    var small: [16]f32 = undefined;
    for (&small, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast(i % 5)) - 2);
    const orig = small;
    fakequant.hadamardInPlace(&small);
    fakequant.hadamardInPlace(&small);
    for (orig, small) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-5);
}

test "roundF16InPlace equals the scalar f16 cast round trip" {
    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();
    var row: [37]f32 = undefined; // odd length: SIMD body + scalar tail
    for (&row) |*v| v.* = random.floatNorm(f32) * 100.0;
    var expected = row;
    for (&expected) |*v| v.* = @floatCast(@as(f16, @floatCast(v.*)));
    fakequant.roundF16InPlace(&row);
    for (expected, row) |e, a| try expectBitEqual(e, a);
}
