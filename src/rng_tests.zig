//! Tests for the repo-owned deterministic RNG (`rng.zig`): counter-based vs
//! sequential splitmix64 parity, the half-open uniform bound (incl. the
//! rounding-boundary clamp), the kaiming bound, and normal-fill moments.

const std = @import("std");
const rng = @import("rng.zig");

const splitmix64 = rng.splitmix64;
const at = rng.at;
const gaussianFill = rng.gaussianFill;
const uniformFill = rng.uniformFill;
const kaimingUniformFill = rng.kaimingUniformFill;
const normalFill = rng.normalFill;

test "counter-based at matches the sequential splitmix64 stream" {
    const seeds = [_]u64{ 0, 1, 42, 0x123456789ABCDEF, std.math.maxInt(u64) };
    for (seeds) |seed| {
        var state = seed;
        for (0..1000) |i| {
            const sequential = splitmix64(&state);
            try std.testing.expectEqual(sequential, at(seed, i));
        }
    }
}

test "uniformFill stays in [lo, hi) and is deterministic" {
    var a: [4096]f32 = undefined;
    var b: [4096]f32 = undefined;
    uniformFill(99, &a, -0.25, 0.75);
    uniformFill(99, &b, -0.25, 0.75);
    try std.testing.expectEqualSlices(f32, &a, &b);
    var mean: f64 = 0;
    for (a) |value| {
        try std.testing.expect(value >= -0.25 and value < 0.75);
        mean += value;
    }
    mean /= a.len;
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), mean, 0.02);

    var c: [4096]f32 = undefined;
    uniformFill(100, &c, -0.25, 0.75);
    try std.testing.expect(!std.mem.eql(f32, &a, &c));
}

test "uniformFill never returns exactly hi at the rounding boundary" {
    // This seed's first splitmix64 output is 0xFFFFFFFFFFFFFFFF (found by
    // inverting the mix function), so u = (2^53 - 1) * 2^-53: the largest
    // representable u, whose f64 -> f32 cast rounds UP to exactly `hi`
    // without the clamp.
    const boundary_seed: u64 = 0x31628AF67B2131AB;
    {
        var state = boundary_seed;
        try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), splitmix64(&state));
    }
    const bounds = [_][2]f32{ .{ -0.25, 0.75 }, .{ 0, 1 }, .{ -1, 1 } };
    for (bounds) |b| {
        var out: [1]f32 = undefined;
        uniformFill(boundary_seed, &out, b[0], b[1]);
        try std.testing.expect(out[0] >= b[0] and out[0] < b[1]);
        try std.testing.expectEqual(std.math.nextAfter(f32, b[1], b[0]), out[0]);
    }
}

test "kaimingUniformFill uses the 1/sqrt(fan_in) bound" {
    const fan_in = 64;
    const bound = @sqrt(1.0 / @as(f32, fan_in));
    var values: [8192]f32 = undefined;
    kaimingUniformFill(7, &values, fan_in);
    var max_abs: f32 = 0;
    for (values) |value| {
        try std.testing.expect(value >= -bound and value < bound);
        max_abs = @max(max_abs, @abs(value));
    }
    // The draw should actually exercise the range, not collapse near zero.
    try std.testing.expect(max_abs > 0.9 * bound);

    var reference: [8192]f32 = undefined;
    uniformFill(7, &reference, -bound, bound);
    try std.testing.expectEqualSlices(f32, &reference, &values);
}

test "gaussianFillAt reproduces every gaussianFill range bitwise" {
    const seeds = [_]u64{ 0, 42, 0xE5E5E5E5E5E5E5E5 };
    for (seeds) |seed| {
        var full: [257]f32 = undefined; // odd length: the discarded-sin tail case
        rng.gaussianFillAt(seed, 0, &full, 1.5);
        var sequential: [257]f32 = undefined;
        gaussianFill(seed, &sequential, 1.5);
        try std.testing.expectEqualSlices(f32, &sequential, &full);

        // Every (first, len) window — odd starts, odd lengths, single elements.
        const windows = [_][2]usize{ .{ 0, 1 }, .{ 1, 1 }, .{ 1, 2 }, .{ 3, 5 }, .{ 2, 254 }, .{ 255, 2 }, .{ 256, 1 }, .{ 7, 250 } };
        for (windows) |w| {
            var out: [256]f32 = undefined;
            rng.gaussianFillAt(seed, w[0], out[0..w[1]], 1.5);
            try std.testing.expectEqualSlices(f32, full[w[0]..][0..w[1]], out[0..w[1]]);
        }

        // Chunked decomposition == one-shot fill (the parallel-kernel contract).
        var chunked: [257]f32 = undefined;
        var start: usize = 0;
        const chunk_lens = [_]usize{ 3, 64, 1, 100, 89 };
        for (chunk_lens) |len| {
            rng.gaussianFillAt(seed, start, chunked[start..][0..len], 1.5);
            start += len;
        }
        try std.testing.expectEqual(full.len, start);
        try std.testing.expectEqualSlices(f32, &full, &chunked);
    }
}

test "gaussianFillAtFast: chunking-invariant, close to the scalar mapping, sane moments" {
    const seeds = [_]u64{ 1, 42, 0xDEADBEEFCAFEF00D };
    for (seeds) |seed| {
        // Chunking invariance (vector body vs scalar edges included): any
        // window decomposition must be bitwise identical to the full fill.
        var full: [1027]f32 = undefined; // odd length, forces scalar tail
        rng.gaussianFillAtFast(seed, 0, &full, 2.0);
        var chunked: [1027]f32 = undefined;
        var start: usize = 0;
        const lens = [_]usize{ 1, 7, 16, 3, 500, 499, 1 };
        for (lens) |len| {
            rng.gaussianFillAtFast(seed, start, chunked[start..][0..len], 2.0);
            start += len;
        }
        try std.testing.expectEqual(full.len, start);
        try std.testing.expectEqualSlices(f32, &full, &chunked);

        // Odd-start window.
        var window: [33]f32 = undefined;
        rng.gaussianFillAtFast(seed, 11, &window, 2.0);
        try std.testing.expectEqualSlices(f32, full[11..][0..33], &window);

        // Same-stream closeness to the scalar f64 mapping: identical
        // uniforms, polynomial-vs-libm transcendentals — a few f32 ulps.
        // A wrong coefficient would blow past this bound by orders of
        // magnitude.
        var scalar: [1027]f32 = undefined;
        gaussianFill(seed, &scalar, 2.0);
        for (scalar, full) |expected, actual| {
            const tolerance = 1e-5 * @max(1.0, @abs(expected));
            try std.testing.expectApproxEqAbs(expected, actual, tolerance);
        }
    }

    // Moments over a larger draw.
    var big: [65536]f32 = undefined;
    rng.gaussianFillAtFast(9, 0, &big, 1.0);
    var mean: f64 = 0;
    for (big) |x| mean += x;
    mean /= big.len;
    var variance: f64 = 0;
    for (big) |x| variance += (x - mean) * (x - mean);
    variance /= big.len;
    try std.testing.expectApproxEqAbs(@as(f64, 0), mean, 0.02);
    try std.testing.expectApproxEqAbs(@as(f64, 1), variance, 0.02);
}

test "normalFill applies mean and std to the gaussianFill stream" {
    var z: [4096]f32 = undefined;
    gaussianFill(3, &z, 1.0);
    var values: [4096]f32 = undefined;
    normalFill(3, &values, 2.0, 0.5);
    for (z, values) |zi, vi| {
        try std.testing.expectEqual(2.0 + 0.5 * zi, vi);
    }

    var plain: [4096]f32 = undefined;
    normalFill(3, &plain, 0.0, 1.0);
    try std.testing.expectEqualSlices(f32, &z, &plain);
}
