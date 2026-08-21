//! Behavioral tests for the dtype module (`dtype.zig`): scalar/accumulator
//! storage mapping, compute/output dtype promotion, and bf16<->f32 rounding.
const std = @import("std");
const dtype = @import("dtype.zig");

const Scalar = dtype.Scalar;
const Accumulator = dtype.Accumulator;
const one = dtype.one;
const supportsGrad = dtype.supportsGrad;
const computeDType = dtype.computeDType;
const outputDType = dtype.outputDType;
const bf16ToF32 = dtype.bf16ToF32;
const f32ToBf16 = dtype.f32ToBf16;

test "dtype maps scalar and accumulator storage" {
    try std.testing.expect(Scalar(.bool) == bool);
    try std.testing.expect(Scalar(.u16) == u16);
    try std.testing.expect(Scalar(.bf16) == u16);
    try std.testing.expect(Scalar(.f32) == f32);
    try std.testing.expect(Accumulator(.f16) == f32);
    try std.testing.expect(Accumulator(.f64) == f64);
    try std.testing.expectEqual(@as(u16, 0x3f80), one(.bf16));
    try std.testing.expect(supportsGrad(.f32));
    try std.testing.expect(!supportsGrad(.u16));
    try std.testing.expect(computeDType(.matmul, .bf16) == .f32);
    try std.testing.expect(outputDType(.matmul, .bf16) == .bf16);
    try std.testing.expect(outputDType(.reduction, .bf16) == .f32);
    try std.testing.expect(computeDType(.pointwise, .f16) == .f16);
    try std.testing.expect(computeDType(.reduction, .f64) == .f64);
}

test "bf16 conversion rounds through f32" {
    try std.testing.expectEqual(@as(f32, 1), bf16ToF32(f32ToBf16(1)));
    try std.testing.expectEqual(@as(f32, -2), bf16ToF32(f32ToBf16(-2)));
    try std.testing.expectApproxEqAbs(@as(f32, 3.140625), bf16ToF32(f32ToBf16(3.14159)), 0);
}

test "f8 decode matches the OCP format definitions" {
    const f8e4m3ToF32 = dtype.f8e4m3ToF32;
    const f8e5m2ToF32 = dtype.f8e5m2ToF32;
    try std.testing.expectEqual(@as(f32, 0), f8e4m3ToF32(0x00));
    try std.testing.expect(std.math.signbit(f8e4m3ToF32(0x80)));
    try std.testing.expectEqual(@as(f32, 1), f8e4m3ToF32(0x38));
    try std.testing.expectEqual(@as(f32, -2), f8e4m3ToF32(0xc0));
    try std.testing.expectEqual(@as(f32, 448), f8e4m3ToF32(0x7e));
    try std.testing.expectEqual(@as(f32, 0.001953125), f8e4m3ToF32(0x01)); // 2^-9 min subnormal
    try std.testing.expectEqual(@as(f32, 0.015625), f8e4m3ToF32(0x08)); // 2^-6 min normal
    try std.testing.expect(std.math.isNan(f8e4m3ToF32(0x7f)));
    try std.testing.expect(std.math.isNan(f8e4m3ToF32(0xff)));

    try std.testing.expectEqual(@as(f32, 1), f8e5m2ToF32(0x3c));
    try std.testing.expectEqual(@as(f32, 57344), f8e5m2ToF32(0x7b));
    try std.testing.expectEqual(std.math.inf(f32), f8e5m2ToF32(0x7c));
    try std.testing.expectEqual(-std.math.inf(f32), f8e5m2ToF32(0xfc));
    try std.testing.expectEqual(@as(f32, 0.0000152587890625), f8e5m2ToF32(0x01)); // 2^-16
    try std.testing.expect(std.math.isNan(f8e5m2ToF32(0x7d)));
    try std.testing.expect(std.math.isNan(f8e5m2ToF32(0xfe)));
}

test "f8 encode round-trips every finite code" {
    for (0..256) |code| {
        const bits: u8 = @intCast(code);
        const e4 = dtype.f8e4m3ToF32(bits);
        if (!std.math.isNan(e4)) {
            try std.testing.expectEqual(bits, dtype.f32ToF8e4m3(e4));
        }
        const e5 = dtype.f8e5m2ToF32(bits);
        if (!std.math.isNan(e5)) {
            try std.testing.expectEqual(bits, dtype.f32ToF8e5m2(e5));
        }
    }
}

fn bruteForceF8(comptime decode: fn (u8) f32, x: f32) u8 {
    // Nearest finite value with ties to even code (the code LSB is the
    // mantissa LSB, also across exponent steps); candidates share x's sign
    // so signed zero resolves correctly.
    const sign_bit: u8 = if (std.math.signbit(x)) 0x80 else 0;
    var best: u8 = sign_bit;
    var best_dist: f64 = std.math.inf(f64);
    for (0..128) |low| {
        const code: u8 = sign_bit | @as(u8, @intCast(low));
        const v = decode(code);
        if (std.math.isNan(v) or std.math.isInf(v)) continue;
        const dist = @abs(@as(f64, x) - @as(f64, v));
        if (dist < best_dist or (dist == best_dist and code & 1 == 0)) {
            best = code;
            best_dist = dist;
        }
    }
    return best;
}

test "f8 encode matches brute-force nearest-even over every f16 value" {
    for (0..65536) |u| {
        const h: f16 = @bitCast(@as(u16, @intCast(u)));
        const x: f32 = @floatCast(h);
        const sign_bit: u8 = if (std.math.signbit(x)) 0x80 else 0;

        const got4 = dtype.f32ToF8e4m3(x);
        if (std.math.isNan(x)) {
            try std.testing.expect(std.math.isNan(dtype.f8e4m3ToF32(got4)));
        } else if (@abs(x) > 464) {
            // beyond the 448-to-next halfway point: e4m3 overflows to NaN
            try std.testing.expectEqual(sign_bit | 0x7f, got4);
        } else {
            try std.testing.expectEqual(bruteForceF8(dtype.f8e4m3ToF32, x), got4);
        }

        const got5 = dtype.f32ToF8e5m2(x);
        if (std.math.isNan(x)) {
            try std.testing.expect(std.math.isNan(dtype.f8e5m2ToF32(got5)));
        } else if (@abs(x) >= 61440) {
            // 61440 is the halfway point; its round-half-even lands on inf
            try std.testing.expectEqual(sign_bit | 0x7c, got5);
        } else {
            try std.testing.expectEqual(bruteForceF8(dtype.f8e5m2ToF32, x), got5);
        }
    }
}

test "f8 overflow and special-value semantics" {
    try std.testing.expectEqual(@as(u8, 0x7f), dtype.f32ToF8e4m3(std.math.inf(f32)));
    try std.testing.expectEqual(@as(u8, 0xff), dtype.f32ToF8e4m3(-std.math.inf(f32)));
    try std.testing.expectEqual(@as(u8, 0x7f), dtype.f32ToF8e4m3(1000));
    try std.testing.expectEqual(@as(u8, 0x7e), dtype.f32ToF8e4m3(464)); // tie rounds to even mantissa 6 = 448
    try std.testing.expectEqual(@as(u8, 0x7c), dtype.f32ToF8e5m2(std.math.inf(f32)));
    try std.testing.expectEqual(@as(u8, 0xfc), dtype.f32ToF8e5m2(-1e30));
    try std.testing.expectEqual(@as(u8, 0x80), dtype.f32ToF8e4m3(-0.0));
    try std.testing.expect(std.math.isNan(dtype.f8e5m2ToF32(dtype.f32ToF8e5m2(std.math.nan(f32)))));
}

test "f8 dtype wiring" {
    try std.testing.expect(Scalar(.f8_e4m3) == u8);
    try std.testing.expect(Scalar(.f8_e5m2) == u8);
    try std.testing.expect(Accumulator(.f8_e4m3) == f32);
    try std.testing.expectEqual(@as(f32, 1), dtype.f8e4m3ToF32(one(.f8_e4m3)));
    try std.testing.expectEqual(@as(f32, 1), dtype.f8e5m2ToF32(one(.f8_e5m2)));
    try std.testing.expect(!supportsGrad(.f8_e4m3));
    try std.testing.expect(!dtype.supportsForwardFloatMath(.f8_e5m2));
    try std.testing.expect(dtype.supportsToFloat(.f8_e4m3));
    try std.testing.expect(dtype.logicalDType(.f8_e4m3) == .f32);

    // castScalar bridges through f32 instead of the integer paths
    try std.testing.expectEqual(@as(i32, 448), dtype.castScalar(.f8_e4m3, .i32, 0x7e));
    try std.testing.expectEqual(@as(i32, 0), dtype.castScalar(.f8_e4m3, .i32, 0x7f)); // NaN -> 0
    try std.testing.expectEqual(@as(u8, 0x38), dtype.castScalar(.i32, .f8_e4m3, 1));
    try std.testing.expectEqual(@as(f32, -2), dtype.castScalar(.f8_e4m3, .f32, 0xc0));
    try std.testing.expectEqual(@as(u8, 0x3c), dtype.castScalar(.f8_e4m3, .f8_e5m2, 0x38));
    try std.testing.expect(!dtype.isTruthy(.f8_e4m3, 0x80)); // -0 is falsy
    try std.testing.expect(dtype.isTruthy(.f8_e5m2, 0x01));
}

test "bf16 conversion quiets NaNs like ggml" {
    const pos_nan: u32 = 0x7fff_ffff;
    const neg_nan: u32 = 0xffff_ffff;
    try std.testing.expectEqual(@as(u16, 0x7fff), f32ToBf16(@as(f32, @bitCast(pos_nan))));
    try std.testing.expectEqual(@as(u16, 0xffff), f32ToBf16(@as(f32, @bitCast(neg_nan))));
    try std.testing.expect(std.math.isNan(bf16ToF32(f32ToBf16(@as(f32, @bitCast(pos_nan))))));
}
