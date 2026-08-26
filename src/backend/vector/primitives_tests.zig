//! Behavioral tests for the low-level @Vector primitives (`primitives.zig`):
//! the SIMD polynomial expf (`vexpf`) numeric accuracy + special-case handling.

const std = @import("std");
const primitives = @import("primitives.zig");

test "primitives.vexpf matches scalar exp over a dense sweep and handles extremes" {
    // Exact special cases.
    try std.testing.expectEqual(@as(f32, 1), primitives.vexpf(1, @splat(0))[0]);
    try std.testing.expectEqual(@as(f32, 0), primitives.vexpf(1, @splat(-std.math.inf(f32)))[0]);
    try std.testing.expectEqual(@as(f32, 0), primitives.vexpf(1, @splat(-200))[0]);
    try std.testing.expectEqual(std.math.inf(f32), primitives.vexpf(1, @splat(200))[0]);
    try std.testing.expectEqual(std.math.inf(f32), primitives.vexpf(1, @splat(std.math.inf(f32)))[0]);

    // NaN propagates (like @exp); it must not flush to exp(clamp(NaN)) = 0.
    const nan = std.math.nan(f32);
    try std.testing.expect(std.math.isNan(primitives.vexpf(1, @splat(nan))[0]));
    {
        const mixed: @Vector(8, f32) = .{ 0, -1, nan, 1, -200, nan, 200, 0.5 };
        const got = primitives.vexpf(8, mixed);
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
        const got = primitives.vexpf(W, x);
        inline for (0..W) |lane| {
            const want = @exp(x[lane]);
            const err = @abs(got[lane] - want);
            if (err > 1e-42 and err > 2e-6 * @abs(want)) {
                std.debug.print("primitives.vexpf({d}) = {d}, want {d}\n", .{ x[lane], got[lane], want });
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

    // erf: torch.erf reference values
    try std.testing.expectApproxEqAbs(@as(f32, 0.8427008), ops.unaryScalar(.erf, 1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.8427008), ops.unaryScalar(.erf, -1.0), 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), ops.unaryScalar(.erf, 0.0));
    try std.testing.expectApproxEqAbs(@as(f32, 0.9953223), ops.unaryScalar(.erf, 2.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5204999), ops.unaryScalar(.erf, 0.5), 1e-6);
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

    primitives.vecUnary(.erf, &got, &x);
    for (x, got) |v, g| try std.testing.expectEqual(ops.unaryScalar(.erf, v), g);
}

test "vecUnary gelu_quant matches ops.geluQuantScalar bit-for-bit over every f16 in [-10, 10]" {
    // ggml's GGML_GELU_FP16 is a 65536-entry table indexed by the f16-rounded
    // input, and `gelu_quant` promises that table's bytes (reference ch. 9,
    // the Gemma GeGLU parity contract). Every finite f16 inside the clamps is
    // an input the table has an entry for, so this sweep IS the contract: the
    // vector lanes must agree with the scalar form (`std.math.tanh`) on all
    // of them, whatever tanh the lanes use.
    var inputs: [1 << 16]f32 = undefined;
    var count: usize = 0;
    var bits: u16 = 0;
    while (true) : (bits += 1) {
        const h: f16 = @bitCast(bits);
        if (std.math.isFinite(h)) {
            const v: f32 = @floatCast(h);
            if (v >= -10 and v <= 10) {
                inputs[count] = v;
                count += 1;
            }
        }
        if (bits == 0xffff) break;
    }
    // Off-grid values on both sides of the clamps (rounded to f16 inside).
    for ([_]f32{ -10.5, -9.999, -9.99, 9.99, 9.999, 10.5, 0.1, -0.1, 1e-3, -1e-3 }) |v| {
        inputs[count] = v;
        count += 1;
    }
    try std.testing.expect(count > 30_000);

    var got: [1 << 16]f32 = undefined;
    primitives.vecUnary(.gelu_quant, got[0..count], inputs[0..count]);
    var mismatches: usize = 0;
    var first: ?usize = null;
    for (inputs[0..count], got[0..count], 0..) |v, g, i| {
        const want = ops.geluQuantScalar(v);
        if (@as(u32, @bitCast(want)) != @as(u32, @bitCast(g))) {
            mismatches += 1;
            if (first == null) first = i;
        }
    }
    if (mismatches != 0) {
        const i = first.?;
        std.debug.print("gelu_quant lane/scalar mismatch: {d} of {d} inputs; first x={e} want={e} got={e}\n", .{ mismatches, count, inputs[i], ops.geluQuantScalar(inputs[i]), got[i] });
        return error.TestUnexpectedResult;
    }
}
