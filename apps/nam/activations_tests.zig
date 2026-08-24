//! Behavioral tests for the NAM activations module (`activations.zig`):
//! per-channel PReLU slopes and the gated/blended A2 row-split formulas,
//! exercised through the public `applyRows`/`applyGated`/`applyBlended` API.
const std = @import("std");
const activations = @import("activations.zig");
const nam_file = @import("nam_file.zig");

const Activation = nam_file.Activation;
const applyRows = activations.applyRows;
const applyGated = activations.applyGated;
const applyBlended = activations.applyBlended;

test "prelu applies per-channel slopes" {
    const slopes = [_]f32{ 0.1, 0.5 };
    const prelu = Activation{ .kind = .prelu, .negative_slopes = &slopes };
    var data = [_]f32{ -1.0, -1.0, 2.0, -2.0 };
    applyRows(&prelu, &data, 2);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1), data[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), data[1], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data[2], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), data[3], 1e-7);
}

test "gated and blended activations use the A2 row split formulas" {
    const primary = Activation{ .kind = .relu };
    const secondary = Activation{ .kind = .hardtanh };
    const z = [_]f32{
        -2.0, 4.0,  0.25, 0.75,
        3.0,  -4.0, -0.5, 1.0,
    };

    var gated: [4]f32 = undefined;
    applyGated(&primary, &secondary, &z, &gated, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gated[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), gated[1], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), gated[2], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gated[3], 1e-7);

    var blended: [4]f32 = undefined;
    applyBlended(&primary, &secondary, &z, &blended, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), blended[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), blended[1], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), blended[2], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), blended[3], 1e-7);
}

test "tanhF32 tracks correctly rounded tanh across the full range" {
    // Dense sweep + branch-boundary neighborhoods, vs the f64 reference.
    // Absolute bound 3e-7 ≈ 2 ulp of tanh's [-1, 1] range — same class as
    // libm tanhf, ~20x inside the 5e-6 golden render gates.
    var max_abs: f64 = 0;
    var x: f64 = -20.0;
    while (x <= 20.0) : (x += 0.00317) {
        const xf: f32 = @floatCast(x);
        const want: f32 = @floatCast(std.math.tanh(@as(f64, xf)));
        const got = activations.tanhF32(xf);
        max_abs = @max(max_abs, @abs(@as(f64, want) - @as(f64, got)));
    }
    for ([_]f32{ 0.35, 9.01, 18.02 / 2.0, 18.5 / 2.0 }) |b| {
        var i: i32 = -64;
        while (i <= 64) : (i += 1) {
            const xf = b + @as(f32, @floatFromInt(i)) * 1e-6;
            for ([_]f32{ xf, -xf }) |v| {
                const want: f32 = @floatCast(std.math.tanh(@as(f64, v)));
                const got = activations.tanhF32(v);
                max_abs = @max(max_abs, @abs(@as(f64, want) - @as(f64, got)));
            }
        }
    }
    try std.testing.expect(max_abs <= 3e-7);
}

test "tanhF32 special values" {
    try std.testing.expectEqual(@as(f32, 0.0), activations.tanhF32(0.0));
    try std.testing.expect(std.math.isNegativeZero(activations.tanhF32(-0.0)));
    try std.testing.expectEqual(@as(f32, 1.0), activations.tanhF32(std.math.inf(f32)));
    try std.testing.expectEqual(@as(f32, -1.0), activations.tanhF32(-std.math.inf(f32)));
    try std.testing.expectEqual(@as(f32, 1.0), activations.tanhF32(50.0));
    try std.testing.expectEqual(@as(f32, -1.0), activations.tanhF32(-50.0));
    try std.testing.expect(std.math.isNan(activations.tanhF32(std.math.nan(f32))));
    // Subnormals pass through (tanh(x) = x - x^3/3, the cube underflows).
    const tiny: f32 = @bitCast(@as(u32, 0x0000_0001));
    try std.testing.expectEqual(tiny, activations.tanhF32(tiny));
    try std.testing.expectEqual(-tiny, activations.tanhF32(-tiny));
}

test "bulk tanh path handles special values in vector lanes" {
    // Long enough to hit the SIMD path on any vector width; checked builds
    // verify the lane arithmetic never traps on non-finite inputs.
    var data = [_]f32{
        std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), 50.0,
        -50.0,             0.0,               -0.0,               1.0,
        std.math.nan(f32), -1.0,              9.02,               -9.02,
        1e-30,             -1e-30,            0.35,               -0.35,
        18.5,
    };
    const act = Activation{ .kind = .tanh };
    applyRows(&act, &data, 1);
    try std.testing.expect(std.math.isNan(data[0]));
    try std.testing.expectEqual(@as(f32, 1.0), data[1]);
    try std.testing.expectEqual(@as(f32, -1.0), data[2]);
    try std.testing.expectEqual(@as(f32, 1.0), data[3]);
    try std.testing.expectEqual(@as(f32, -1.0), data[4]);
    try std.testing.expectEqual(@as(f32, 0.0), data[5]);
    try std.testing.expect(std.math.isNegativeZero(data[6]));
    try std.testing.expect(std.math.isNan(data[8]));
    try std.testing.expectEqual(@as(f32, 1e-30), data[12]);
    try std.testing.expectEqual(@as(f32, -1e-30), data[13]);
}

test "bulk tanh path is bit-identical to the scalar contract at every offset" {
    // Catches lane-position or tail-padding dependence: any data length from 1
    // to 67 must produce, per element, exactly tanhF32(v).
    var rng_state: u64 = 0x9E3779B97F4A7C15;
    var data: [67]f32 = undefined;
    for (&data) |*v| {
        rng_state = rng_state *% 6364136223846793005 +% 1442695040888963407;
        const bits: u32 = @truncate(rng_state >> 32);
        // Map to roughly [-12, 12): covers both branches and saturation.
        v.* = (@as(f32, @floatFromInt(bits)) / 4294967296.0 - 0.5) * 24.0;
    }
    const act = Activation{ .kind = .tanh };
    var len: usize = 1;
    while (len <= data.len) : (len += 1) {
        var buf: [67]f32 = data;
        applyRows(&act, buf[0..len], 1);
        for (buf[0..len], data[0..len]) |got, src| {
            try std.testing.expectEqual(activations.tanhF32(src), got);
        }
    }
}
