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

test "bf16 conversion quiets NaNs like ggml" {
    const pos_nan: u32 = 0x7fff_ffff;
    const neg_nan: u32 = 0xffff_ffff;
    try std.testing.expectEqual(@as(u16, 0x7fff), f32ToBf16(@as(f32, @bitCast(pos_nan))));
    try std.testing.expectEqual(@as(u16, 0xffff), f32ToBf16(@as(f32, @bitCast(neg_nan))));
    try std.testing.expect(std.math.isNan(bf16ToF32(f32ToBf16(@as(f32, @bitCast(pos_nan))))));
}
