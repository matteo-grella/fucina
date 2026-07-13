//! Fake-quantization round-trips: values pass through a low-precision grid
//! (FP8 E4M3, FP4 E2M1, or f16) and come back as f32. This is
//! quantization-aware *inference* numerics — models whose reference stores
//! activations or cache rows through such grids (DeepSeek V4's FP8 KV rows
//! and FP4/Hadamard indexer QAT) need the exact same round trip, because the
//! quantized values feed back into the graph across steps.
//!
//! The grid rounding is round-to-nearest with ties-to-even (the IEEE default
//! and what a hardware cast does): implemented arithmetically on the f32 bit
//! pattern rather than by grid search, so it vectorizes; the tests pin it
//! bit-for-bit against a literal grid-search oracle over every grid boundary.
//! The group kernels reproduce the microscaling recipe exactly: per group,
//! `amax = max |x|` (floored at `amax_floor`), a power-of-two scale
//! `2^ceil(log2(amax / grid_max))`, clamp to ±grid_max, grid round trip,
//! rescale. All kernels are in place over host slices, and the SIMD and
//! scalar paths evaluate the same per-element expression, so results are
//! bit-identical for any length.
//!
//! Domain module: pure slice kernels, no `*Runtime` (nothing allocates).
//! Re-exported as `exec.fakequant` / `fucina.fakequant`.

const std = @import("std");

const vec_width = 8;
const F32Vec = @Vector(vec_width, f32);
const U32Vec = @Vector(vec_width, u32);

/// FP8 E4M3 round trip: nearest grid value (ties-to-even mantissa) among
/// sign x {0, denormals k·2^-9, (1+m/8)·2^e for e in [-6, 8]}, saturated at
/// ±448 (the OCP E4M3 finite range; NaN/inf encodings are never produced —
/// out-of-range magnitudes clamp to ±448 first, matching a saturating cast).
pub fn roundE4m3(x: f32) f32 {
    const q = roundE4m3Mag(@min(@abs(x), 448.0));
    return if (x < 0) -q else q;
}

inline fn roundE4m3Mag(ax: f32) f32 {
    if (ax < 0.015625) {
        // Below the E4M3 normal range (2^-6): the grid is k·2^-9 — round
        // ax·2^9 to an integer (RNE via the 2^23 magic number) and rescale.
        const big: f32 = 8388608.0;
        return ((ax * 512.0 + big) - big) * (1.0 / 512.0);
    }
    // Normal range: RNE to 3 kept mantissa bits directly on the bit pattern
    // (carry into the exponent handles mantissa overflow, e.g. 1.96 -> 2.0).
    var bits: u32 = @bitCast(ax);
    bits += 0x0007FFFF + ((bits >> 20) & 1);
    return @bitCast(bits & 0xFFF0_0000);
}

/// FP4 E2M1 round trip: nearest of sign x {0, 0.5, 1, 1.5, 2, 3, 4, 6}
/// (ties-to-even mantissa), saturated at ±6.
pub fn roundE2m1(x: f32) f32 {
    const q = roundE2m1Mag(@min(@abs(x), 6.0));
    return if (x < 0) -q else q;
}

inline fn roundE2m1Mag(ax: f32) f32 {
    if (ax < 1.0) {
        // Below the E2M1 normal range: the grid is k·0.5.
        const big: f32 = 8388608.0;
        return ((ax * 2.0 + big) - big) * 0.5;
    }
    // RNE to 1 kept mantissa bit.
    var bits: u32 = @bitCast(ax);
    bits += 0x001F_FFFF + ((bits >> 22) & 1);
    return @bitCast(bits & 0xFFC0_0000);
}

inline fn roundE4m3MagVec(ax: F32Vec) F32Vec {
    const big: F32Vec = @splat(8388608.0);
    const sub = (ax * @as(F32Vec, @splat(512.0)) + big - big) * @as(F32Vec, @splat(1.0 / 512.0));
    var bits: U32Vec = @bitCast(ax);
    bits += @as(U32Vec, @splat(0x0007FFFF)) + ((bits >> @as(@Vector(vec_width, u5), @splat(20))) & @as(U32Vec, @splat(1)));
    const norm: F32Vec = @bitCast(bits & @as(U32Vec, @splat(0xFFF0_0000)));
    return @select(f32, ax < @as(F32Vec, @splat(0.015625)), sub, norm);
}

inline fn roundE2m1MagVec(ax: F32Vec) F32Vec {
    const big: F32Vec = @splat(8388608.0);
    const sub = (ax * @as(F32Vec, @splat(2.0)) + big - big) * @as(F32Vec, @splat(0.5));
    var bits: U32Vec = @bitCast(ax);
    bits += @as(U32Vec, @splat(0x001F_FFFF)) + ((bits >> @as(@Vector(vec_width, u5), @splat(22))) & @as(U32Vec, @splat(1)));
    const norm: F32Vec = @bitCast(bits & @as(U32Vec, @splat(0xFFC0_0000)));
    return @select(f32, ax < @as(F32Vec, @splat(1.0)), sub, norm);
}

const GridKind = enum { e4m3, e2m1 };

/// Microscaling fake-quant of one contiguous slice, in place: per
/// `group_size` values, power-of-two scale from the group amax (floored at
/// `amax_floor` so the scale stays finite on all-zero groups), clamp to the
/// grid range, grid round trip, rescale. `x.len` must be a multiple of
/// `group_size`. DeepSeek V4 uses (64, 1e-4) for its FP8 KV rows.
pub fn groupRoundTripE4m3InPlace(x: []f32, group_size: usize, amax_floor: f32) void {
    groupRoundTrip(.e4m3, x, group_size, amax_floor);
}

/// As `groupRoundTripE4m3InPlace` on the E2M1 grid. DeepSeek V4 uses
/// (32, 6·2^-126) for its FP4 indexer activations (the floor keeps the
/// power-of-two scale a normal f32).
pub fn groupRoundTripE2m1InPlace(x: []f32, group_size: usize, amax_floor: f32) void {
    groupRoundTrip(.e2m1, x, group_size, amax_floor);
}

fn groupRoundTrip(comptime kind: GridKind, x: []f32, group_size: usize, amax_floor: f32) void {
    std.debug.assert(group_size > 0 and x.len % group_size == 0);
    const grid_max: f32 = switch (kind) {
        .e4m3 => 448.0,
        .e2m1 => 6.0,
    };
    var off: usize = 0;
    while (off < x.len) : (off += group_size) {
        const group = x[off..][0..group_size];

        var amax: f32 = 0;
        var i: usize = 0;
        if (group_size >= vec_width) {
            var amax_vec: F32Vec = @splat(0.0);
            while (i + vec_width <= group.len) : (i += vec_width) {
                amax_vec = @max(amax_vec, @abs(@as(F32Vec, group[i..][0..vec_width].*)));
            }
            amax = @reduce(.Max, amax_vec);
        }
        while (i < group.len) : (i += 1) amax = @max(amax, @abs(group[i]));
        if (amax < amax_floor) amax = amax_floor;

        const scale = std.math.ldexp(@as(f32, 1.0), @intFromFloat(@ceil(@log2(amax / grid_max))));

        i = 0;
        while (i + vec_width <= group.len) : (i += vec_width) {
            const v: F32Vec = group[i..][0..vec_width].*;
            const clamped = @min(@max(v / @as(F32Vec, @splat(scale)), @as(F32Vec, @splat(-grid_max))), @as(F32Vec, @splat(grid_max)));
            const mag = switch (kind) {
                .e4m3 => roundE4m3MagVec(@abs(clamped)),
                .e2m1 => roundE2m1MagVec(@abs(clamped)),
            };
            const rounded = @select(f32, clamped < @as(F32Vec, @splat(0.0)), -mag, mag);
            group[i..][0..vec_width].* = rounded * @as(F32Vec, @splat(scale));
        }
        while (i < group.len) : (i += 1) {
            const clamped = @min(@max(group[i] / scale, -grid_max), grid_max);
            const rounded = switch (kind) {
                .e4m3 => roundE4m3(clamped),
                .e2m1 => roundE2m1(clamped),
            };
            group[i] = rounded * scale;
        }
    }
}

/// In-place fast Walsh-Hadamard transform scaled by 1/sqrt(len): the
/// orthonormal Hadamard rotation used by rotation-based quantization schemes
/// (QuaRot/SpinQuant-style; DeepSeek V4's indexer QAT rotates 128-wide rows
/// before the FP4 round trip). `x.len` must be a power of two. Involution:
/// applying it twice restores the input up to f32 rounding. Every butterfly
/// output is exactly `a + b` / `a - b` (no accumulation), so the SIMD and
/// scalar passes are bit-identical.
pub fn hadamardInPlace(x: []f32) void {
    std.debug.assert(x.len >= 1 and std.math.isPowerOfTwo(x.len));
    var stride: usize = 1;
    while (stride < x.len) : (stride <<= 1) {
        var base: usize = 0;
        while (base < x.len) : (base += 2 * stride) {
            var i: usize = 0;
            if (stride >= vec_width) {
                while (i + vec_width <= stride) : (i += vec_width) {
                    const a: F32Vec = x[base + i ..][0..vec_width].*;
                    const b: F32Vec = x[base + stride + i ..][0..vec_width].*;
                    x[base + i ..][0..vec_width].* = a + b;
                    x[base + stride + i ..][0..vec_width].* = a - b;
                }
            }
            while (i < stride) : (i += 1) {
                const a = x[base + i];
                const b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
    }
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(x.len))));
    var i: usize = 0;
    while (i + vec_width <= x.len) : (i += vec_width) {
        x[i..][0..vec_width].* = @as(F32Vec, x[i..][0..vec_width].*) * @as(F32Vec, @splat(scale));
    }
    while (i < x.len) : (i += 1) x[i] *= scale;
}

/// In-place f32 -> f16 -> f32 round trip (RNE both ways, saturating per
/// IEEE): what storing a row into an f16 cache and reading it back does.
pub fn roundF16InPlace(x: []f32) void {
    const F16Vec = @Vector(vec_width, f16);
    var i: usize = 0;
    while (i + vec_width <= x.len) : (i += vec_width) {
        const narrowed: F16Vec = @floatCast(@as(F32Vec, x[i..][0..vec_width].*));
        x[i..][0..vec_width].* = @as(F32Vec, @floatCast(narrowed));
    }
    while (i < x.len) : (i += 1) x[i] = @floatCast(@as(f16, @floatCast(x[i])));
}

test {
    _ = @import("fakequant_tests.zig");
}
