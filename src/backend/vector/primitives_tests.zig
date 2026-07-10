//! Behavioral tests for the low-level @Vector primitives (`primitives.zig`):
//! the SIMD polynomial expf (`vexpf`) numeric accuracy + special-case handling.

const std = @import("std");
const primitives = @import("primitives.zig");

const vexpf = primitives.vexpf;

test "vexpf matches scalar exp over a dense sweep and handles extremes" {
    // Exact special cases.
    try std.testing.expectEqual(@as(f32, 1), vexpf(1, @splat(0))[0]);
    try std.testing.expectEqual(@as(f32, 0), vexpf(1, @splat(-std.math.inf(f32)))[0]);
    try std.testing.expectEqual(@as(f32, 0), vexpf(1, @splat(-200))[0]);
    try std.testing.expectEqual(std.math.inf(f32), vexpf(1, @splat(200))[0]);
    try std.testing.expectEqual(std.math.inf(f32), vexpf(1, @splat(std.math.inf(f32)))[0]);

    // NaN propagates (like @exp); it must not flush to exp(clamp(NaN)) = 0.
    const nan = std.math.nan(f32);
    try std.testing.expect(std.math.isNan(vexpf(1, @splat(nan))[0]));
    {
        const mixed: @Vector(8, f32) = .{ 0, -1, nan, 1, -200, nan, 200, 0.5 };
        const got = vexpf(8, mixed);
        inline for (0..8) |lane| {
            if (std.math.isNan(mixed[lane])) {
                try std.testing.expect(std.math.isNan(got[lane]));
            } else {
                try std.testing.expect(!std.math.isNan(got[lane]));
                const want = @exp(mixed[lane]);
                if (std.math.isInf(want)) {
                    try std.testing.expectEqual(want, got[lane]);
                } else {
                    try std.testing.expect(@abs(got[lane] - want) <= 2e-6 * @abs(want));
                }
            }
        }
    }

    // Dense sweep of [-90, 89]: relative tolerance 2e-6, absolute 1e-42 near
    // zero (covers the subnormal range below ~ -87.3).
    const W = 8;
    const point_count: usize = 100_000;
    const lo: f64 = -90;
    const hi: f64 = 89;
    const step = (hi - lo) / @as(f64, @floatFromInt(point_count - 1));
    var i: usize = 0;
    while (i < point_count) : (i += W) {
        var x: @Vector(W, f32) = undefined;
        inline for (0..W) |lane| {
            const point = @min(i + lane, point_count - 1);
            x[lane] = @floatCast(lo + step * @as(f64, @floatFromInt(point)));
        }
        const got = vexpf(W, x);
        inline for (0..W) |lane| {
            const want = @exp(x[lane]);
            const err = @abs(got[lane] - want);
            if (err > 1e-42 and err > 2e-6 * @abs(want)) {
                std.debug.print("vexpf({d}) = {d}, want {d}\n", .{ x[lane], got[lane], want });
                return error.TestUnexpectedResult;
            }
        }
    }
}

const ops = @import("../ops.zig");

test "erff matches known values and special cases (musl port)" {
    // Reference values (double-precision erf, rounded): erf(1) = 0.8427007929,
    // erf(0.5) = 0.5204998778, erf(2) = 0.9953222650, erf(3) = 0.9999779095.
    try std.testing.expectEqual(@as(f32, 0), ops.erff(0));
    try std.testing.expectApproxEqAbs(@as(f32, 0.8427008), ops.erff(1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.8427008), ops.erff(-1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5204999), ops.erff(0.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9953223), ops.erff(2.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9999779), ops.erff(3.0), 1e-6);
    // |x| >= 6 branch and infinities saturate to +-1.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ops.erff(10.0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), ops.erff(-10.0), 1e-7);
    try std.testing.expectEqual(@as(f32, 1.0), ops.erff(std.math.inf(f32)));
    try std.testing.expectEqual(@as(f32, -1.0), ops.erff(-std.math.inf(f32)));
    try std.testing.expect(std.math.isNan(ops.erff(std.math.nan(f32))));
    // Tiny-argument branch: erf(x) ~= 2x/sqrt(pi).
    const tiny: f32 = 1e-10;
    try std.testing.expectApproxEqRel(@as(f32, 2.0 / @sqrt(std.math.pi)) * tiny, ops.erff(tiny), 1e-5);
    // Odd symmetry on a sweep.
    var x: f32 = -5.0;
    while (x <= 5.0) : (x += 0.173) {
        try std.testing.expectEqual(ops.erff(x), -ops.erff(-x));
    }
}

test "elu and gelu_erf scalar values match known-good constants" {
    // elu(-1) = expm1(-1) = -0.63212055; identity for x > 0.
    try std.testing.expectApproxEqAbs(@as(f32, -0.6321206), ops.unaryScalar(.elu, -1.0), 1e-6);
    try std.testing.expectEqual(@as(f32, 2.0), ops.unaryScalar(.elu, 2.0));
    try std.testing.expectEqual(@as(f32, 0.0), ops.unaryScalar(.elu, 0.0));
    try std.testing.expectApproxEqAbs(@as(f32, -0.0951626), ops.unaryScalar(.elu, -0.1), 1e-6);

    // gelu_erf(1) = 0.5*(1 + erf(1/sqrt(2))) = 0.8413447 (the exact-erf GELU,
    // NOT the tanh approximation).
    try std.testing.expectApproxEqAbs(@as(f32, 0.8413447), ops.unaryScalar(.gelu_erf, 1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.15865526), ops.unaryScalar(.gelu_erf, -1.0), 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), ops.unaryScalar(.gelu_erf, 0.0));
    try std.testing.expectApproxEqAbs(@as(f32, 1.9544997), ops.unaryScalar(.gelu_erf, 2.0), 1e-6);
}

test "vecUnary elu and gelu_erf match the scalar path bit-for-bit" {
    // 19 elements: exercises the 4x-unrolled body, the single-vector loop, and
    // the scalar tail; the per-lane fallback must agree with unaryScalar exactly.
    var x: [19]f32 = undefined;
    for (&x, 0..) |*v, i| {
        v.* = (@as(f32, @floatFromInt(i)) - 9.0) * 0.37;
    }
    var got: [19]f32 = undefined;

    primitives.vecUnary(.elu, &got, &x);
    for (x, got) |v, g| try std.testing.expectEqual(ops.unaryScalar(.elu, v), g);

    primitives.vecUnary(.gelu_erf, &got, &x);
    for (x, got) |v, g| try std.testing.expectEqual(ops.unaryScalar(.gelu_erf, v), g);
}
