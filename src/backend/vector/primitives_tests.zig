//! Behavioral tests for the low-level @Vector primitives (`primitives.zig`):
//! the SIMD polynomial expf (`vexpf`) numeric accuracy + special-case
//! handling, and the byte-parity contracts of the musl-faithful lane ports
//! (`vexpm1f`, `vtanhf`, and the `gelu_quant`/`elu` bodies riding them).

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

test "primitives.vexpm1f matches std.math.expm1 bit-for-bit over dense sweeps and specials" {
    // The musl expm1f lane port's contract is BYTE equality with the scalar
    // `std.math.expm1` (same musl code): every branch — NaN
    // canonicalization, saturation to -1, overflow past o_threshold, the
    // 27*ln2 fall-through, the k = ±1 / far-k reductions, |k| > 56 with the
    // k = 128 special, tiny, and subnormal — must produce the scalar bits.
    const specials = [_]f32{
        0.0,      -0.0,     std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32),
        88.72168, 88.72169, 89.0,              104.0,              3.4e38,
        -18.714973,   -18.714975,    -20.0,          -1e38,           88.375, // k = 128
        0x1.62e42p-1, -0x1.62e42p-1, 0x1.0a2b24p+0,  -0x1.0a2b24p+0,  0x1.62e44p-1,
        0x1p-25,      -0x1p-25,      0x1.fffffep-26, -0x1.fffffep-26, 1e-30,
        -1e-30,       0x1p-126,      -0x1p-126,      0x1p-140,        -0x1p-140,
    };
    for (specials) |v| {
        const got = primitives.vexpm1f(1, @splat(v))[0];
        const want = std.math.expm1(v);
        try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got)));
    }
    // Dense sweeps: the full finite band [-104, 89] (every k arm, the
    // saturation/overflow tails) and a fine band across the reduction
    // boundaries around zero.
    const W = 8;
    const bands = [_][2]f64{ .{ -104, 89 }, .{ -1.1, 1.1 } };
    for (bands) |band| {
        const point_count: usize = 100_003;
        const step = (band[1] - band[0]) / @as(f64, @floatFromInt(point_count - 1));
        var i: usize = 0;
        while (i < point_count) : (i += W) {
            var x: @Vector(W, f32) = undefined;
            inline for (0..W) |lane| {
                const point = @min(i + lane, point_count - 1);
                x[lane] = @floatCast(band[0] + step * @as(f64, @floatFromInt(point)));
            }
            const got = primitives.vexpm1f(W, x);
            inline for (0..W) |lane| {
                const want = std.math.expm1(x[lane]);
                if (@as(u32, @bitCast(want)) != @as(u32, @bitCast(got[lane]))) {
                    std.debug.print("vexpm1f({e}) = {e}, want {e}\n", .{ x[lane], got[lane], want });
                    return error.TestUnexpectedResult;
                }
            }
        }
    }
}

test "primitives.vtanhf matches std.math.tanh bit-for-bit over dense sweeps and specials" {
    // Same byte contract as vexpm1f, for the tanhf port every `gelu_quant`
    // lane calls: all four scalar branches (saturated |x| > 10, the two
    // expm1 quotient arms, the subnormal pass-through) plus NaN/±inf, at
    // the exact musl branch-cut bit patterns.
    const cut_hi: f32 = @bitCast(@as(u32, 0x3F0C9F54)); // log(3)/2 cut
    const cut_lo: f32 = @bitCast(@as(u32, 0x3E82C578)); // log(5/3)/2 cut
    const specials = [_]f32{
        0.0,                            -0.0,     std.math.inf(f32),              -std.math.inf(f32),                        std.math.nan(f32),
        10.0,                           -10.0,    @bitCast(@as(u32, 0x41200001)), -@as(f32, @bitCast(@as(u32, 0x41200001))), 12.5,
        cut_hi,                         -cut_hi,  @bitCast(@as(u32, 0x3F0C9F55)), cut_lo,                                    -cut_lo,
        @bitCast(@as(u32, 0x3E82C579)), 0x1p-126, -0x1p-126,                      0x1p-135,                                  -0x1p-135,
        1e-40,                          -1e-40,   0.125,                          -0.125,                                    20.0,
    };
    for (specials) |v| {
        const got = primitives.vtanhf(1, @splat(v))[0];
        const want = std.math.tanh(v);
        try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got)));
    }
    const W = 8;
    const bands = [_][2]f64{ .{ -12, 12 }, .{ -0.6, 0.6 } };
    for (bands) |band| {
        const point_count: usize = 100_003;
        const step = (band[1] - band[0]) / @as(f64, @floatFromInt(point_count - 1));
        var i: usize = 0;
        while (i < point_count) : (i += W) {
            var x: @Vector(W, f32) = undefined;
            inline for (0..W) |lane| {
                const point = @min(i + lane, point_count - 1);
                x[lane] = @floatCast(band[0] + step * @as(f64, @floatFromInt(point)));
            }
            const got = primitives.vtanhf(W, x);
            inline for (0..W) |lane| {
                const want = std.math.tanh(x[lane]);
                if (@as(u32, @bitCast(want)) != @as(u32, @bitCast(got[lane]))) {
                    std.debug.print("vtanhf({e}) = {e}, want {e}\n", .{ x[lane], got[lane], want });
                    return error.TestUnexpectedResult;
                }
            }
        }
    }
}

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

const GeluQuantSweep = struct {
    fn check(inputs: []const f32, got: []f32, label: []const u8) !void {
        primitives.vecUnary(.gelu_quant, got[0..inputs.len], inputs);
        var mismatches: usize = 0;
        var first: ?usize = null;
        for (inputs, got[0..inputs.len], 0..) |v, g, i| {
            const want = ops.geluQuantScalar(v);
            if (@as(u32, @bitCast(want)) != @as(u32, @bitCast(g))) {
                mismatches += 1;
                if (first == null) first = i;
            }
        }
        if (mismatches != 0) {
            const i = first.?;
            std.debug.print("gelu_quant lane/scalar mismatch ({s}): {d} of {d} inputs; first x={e} want={e} got={e}\n", .{ label, mismatches, inputs.len, inputs[i], ops.geluQuantScalar(inputs[i]), got[i] });
            return error.TestUnexpectedResult;
        }
    }
};

test "vecUnary gelu_quant matches ops.geluQuantScalar bit-for-bit over every f16 in [-10, 10]" {
    // ggml's GGML_GELU_FP16 is a 65536-entry table indexed by the f16-rounded
    // input, and `gelu_quant` promises that table's bytes (reference ch. 9,
    // the Gemma GeGLU parity contract). Every finite f16 inside the clamps is
    // an input the table has an entry for, so this sweep IS the contract: the
    // vector lanes must agree with the scalar form (`std.math.tanh`) on all
    // of them, whatever tanh the lanes use.
    var inputs: [1 << 16]f32 = undefined;
    var got: [1 << 16]f32 = undefined;
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
    try GeluQuantSweep.check(inputs[0..count], &got, "f16 grid");

    // The rounding half of the contract: the input is f16-rounded FIRST, so
    // f32 inputs that are NOT f16-representable must land on the same table
    // entry as the scalar. The midpoint between every adjacent pair of f16
    // grid points is exact in f32 and ties under round-to-nearest-even —
    // the hardest case the rounding step has.
    count = 0;
    bits = 0;
    while (true) : (bits += 1) {
        const h: f16 = @bitCast(bits);
        const h2: f16 = @bitCast(bits +% 1);
        if (std.math.isFinite(h) and std.math.isFinite(h2)) {
            const a: f32 = @floatCast(h);
            const b: f32 = @floatCast(h2);
            const m = (a + b) * 0.5; // exact: adjacent f16 pairs sum exactly in f32
            if (m >= -10.5 and m <= 10.5) {
                inputs[count] = m;
                count += 1;
            }
        }
        if (bits == 0xffff) break;
    }
    try std.testing.expect(count > 30_000);
    try GeluQuantSweep.check(inputs[0..count], &got, "f16 midpoints");

    // A uniform off-grid band across the clamps (generic non-representable
    // f32 inputs, no tie bias).
    count = 0;
    const point_count: usize = 60_013;
    var i: usize = 0;
    while (i < point_count) : (i += 1) {
        inputs[count] = @floatCast(-10.6 + 21.2 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(point_count - 1)));
        count += 1;
    }
    try GeluQuantSweep.check(inputs[0..count], &got, "uniform off-grid");
}

test "vecUnary elu matches the scalar path bit-for-bit over every finite f16 and a dense f32 band" {
    // .elu is ggml_vec_elu_f32 parity (omnivoice's SemanticEncoder): the
    // vector arm rides `vexpm1f`, and this sweep pins its bytes to the
    // scalar `std.math.expm1` form across the negative branch (saturation
    // below -18.71, every reduction arm, tiny, subnormal), the identity
    // positive arm, and the specials.
    var inputs: [70_000]f32 = undefined;
    var got: [70_000]f32 = undefined;
    var count: usize = 0;
    var bits: u16 = 0;
    while (true) : (bits += 1) {
        const h: f16 = @bitCast(bits);
        if (std.math.isFinite(h)) {
            inputs[count] = @floatCast(h);
            count += 1;
        }
        if (bits == 0xffff) break;
    }
    for ([_]f32{ std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32), -1e38, 1e38, -88.7, -104.5, 0x1p-140, -0x1p-140 }) |v| {
        inputs[count] = v;
        count += 1;
    }
    // Dense off-grid negative band through the -18.71 saturation cut.
    const point_count: usize = 6_007;
    var i: usize = 0;
    while (i < point_count) : (i += 1) {
        inputs[count] = @floatCast(-21.0 + 21.5 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(point_count - 1)));
        count += 1;
    }

    primitives.vecUnary(.elu, got[0..count], inputs[0..count]);
    var mismatches: usize = 0;
    var first: ?usize = null;
    for (inputs[0..count], got[0..count], 0..) |v, g, idx| {
        const want = ops.unaryScalar(.elu, v);
        if (@as(u32, @bitCast(want)) != @as(u32, @bitCast(g))) {
            mismatches += 1;
            if (first == null) first = idx;
        }
    }
    if (mismatches != 0) {
        const idx = first.?;
        std.debug.print("elu lane/scalar mismatch: {d} of {d} inputs; first x={e} want={e} got={e}\n", .{ mismatches, count, inputs[idx], ops.unaryScalar(.elu, inputs[idx]), got[idx] });
        return error.TestUnexpectedResult;
    }
}
