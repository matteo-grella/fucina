//! Behavioral tests for the PTQTP trit-plane solver (`ptqtp.zig`): exact
//! recovery on lattice inputs, error ordering vs the single-plane baselines
//! (PTQTP-K2 < PTQTP-K1 < blind absmean b1.58), byte-level layout parity
//! with the ggml-parity TQ2_0 encoder, matmul-path equivalence through the
//! borrowed RHS views, NaN benignity, determinism, the reference-path G
//! ablation, and option/shape validation.
const std = @import("std");
const backend = @import("backend.zig");
const exec = @import("exec.zig");
const ptqtp = @import("ptqtp.zig");

const ExecContext = exec.ExecContext;
const quant = backend.quantized_matmul;
const BlockTQ2_0 = backend.BlockTQ2_0;

fn relErr(w: []const f32, rec: []const f32) f64 {
    var err: f64 = 0;
    var w2: f64 = 0;
    for (w, rec) |a, b| {
        const d = @as(f64, a) - @as(f64, b);
        err += d * d;
        w2 += @as(f64, a) * @as(f64, a);
    }
    return if (w2 > 0) @sqrt(err / w2) else 0;
}

fn fillGaussian(prng: *std.Random.DefaultPrng, values: []f32, scale: f32) void {
    const random = prng.random();
    for (values) |*v| v.* = random.floatNorm(f32) * scale;
}

fn randomTrits(prng: *std.Random.DefaultPrng, trits: []i8) void {
    const random = prng.random();
    for (trits) |*t| t.* = @as(i8, @intCast(random.intRangeAtMost(i2, -1, 1)));
}

test "K=1 recovers an exactly ternary matrix bit-perfectly" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const n = 3;
    const k = 512;
    var prng = std.Random.DefaultPrng.init(42);
    var trits: [n * k]i8 = undefined;
    randomTrits(&prng, &trits);
    var w: [n * k]f32 = undefined;
    for (&w, trits) |*v, t| v.* = 0.5 * @as(f32, @floatFromInt(t));

    var pair = try ptqtp.quantizeMatrix(&ctx, &w, n, k, .{ .planes = 1 });
    defer pair.deinit(ctx.allocator);

    try std.testing.expectEqual(@as(usize, 1), pair.planeCount());
    try std.testing.expectEqual(@as(f64, 0), pair.stats.rel_frob_err);

    var rec: [n * k]f32 = undefined;
    try pair.reconstructInto(&rec);
    for (w, rec) |a, b| try std.testing.expectEqual(a, b);
}

test "packed blocks are byte-identical to the scaled TQ2_0 encoder" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const k = 256;
    var prng = std.Random.DefaultPrng.init(7);
    var trits: [k]i8 = undefined;
    randomTrits(&prng, &trits);
    var w: [k]f32 = undefined;
    for (&w, trits) |*v, t| v.* = 0.5 * @as(f32, @floatFromInt(t));

    var pair = try ptqtp.quantizeMatrix(&ctx, &w, 1, k, .{ .planes = 1 });
    defer pair.deinit(ctx.allocator);

    var reference: [1]BlockTQ2_0 = undefined;
    try quant.quantizeRowTQ2_0ScaledInto(&reference, &w, 0.5);

    try std.testing.expectEqual(reference[0].d, pair.plane1[0].d);
    try std.testing.expect(std.mem.eql(u8, &reference[0].qs, &pair.plane1[0].qs));
}

test "dual planes beat one plane beat blind absmean on gaussian weights" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const n = 4;
    const k = 512;
    var prng = std.Random.DefaultPrng.init(1234);
    var w: [n * k]f32 = undefined;
    fillGaussian(&prng, &w, 0.02);

    var pair2 = try ptqtp.quantizeMatrix(&ctx, &w, n, k, .{ .planes = 2 });
    defer pair2.deinit(ctx.allocator);
    var pair1 = try ptqtp.quantizeMatrix(&ctx, &w, n, k, .{ .planes = 1 });
    defer pair1.deinit(ctx.allocator);

    // Blind b1.58: one absmean scale for the whole matrix, round-clip.
    const absmean_blocks = try std.testing.allocator.alloc(BlockTQ2_0, n * k / ptqtp.block_len);
    defer std.testing.allocator.free(absmean_blocks);
    const d = quant.ternaryAbsmeanScale(&w);
    const blocks_per_row = k / ptqtp.block_len;
    for (0..n) |r| {
        try quant.quantizeRowTQ2_0ScaledInto(
            absmean_blocks[r * blocks_per_row ..][0..blocks_per_row],
            w[r * k ..][0..k],
            d,
        );
    }
    var absmean_pair = ptqtp.PlanePair{
        .plane1 = absmean_blocks,
        .plane2 = &.{},
        .plane3 = &.{},
        .rows = n,
        .cols = k,
        .stats = undefined,
    };
    var rec: [n * k]f32 = undefined;
    try absmean_pair.reconstructInto(&rec);
    const err_absmean = relErr(&w, &rec);

    try std.testing.expect(pair2.stats.rel_frob_err < pair1.stats.rel_frob_err);
    try std.testing.expect(pair1.stats.rel_frob_err < err_absmean);
    // Stats must agree with the actual packed reconstruction.
    try pair2.reconstructInto(&rec);
    try std.testing.expectApproxEqAbs(pair2.stats.rel_frob_err, relErr(&w, &rec), 1e-12);
    // Both planes carry sparsity and the planes differ (symmetric init broken).
    try std.testing.expect(pair2.stats.zero_frac[0] > 0);
    try std.testing.expect(pair2.stats.zero_frac[1] > 0);
    try std.testing.expect(!std.mem.eql(
        u8,
        std.mem.sliceAsBytes(pair2.plane1),
        std.mem.sliceAsBytes(pair2.plane2),
    ));
    try std.testing.expect(pair2.stats.mean_iterations <= 50);
}

test "borrowed RHS views multiply like the reconstruction" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const n = 8;
    const k = 256;
    const m = 2;
    var prng = std.Random.DefaultPrng.init(99);
    var w: [n * k]f32 = undefined;
    fillGaussian(&prng, &w, 0.05);
    var x: [m * k]f32 = undefined;
    fillGaussian(&prng, &x, 1.0);

    var pair = try ptqtp.quantizeMatrix(&ctx, &w, n, k, .{});
    defer pair.deinit(ctx.allocator);

    var y = [_]f32{0} ** (m * n);
    var y_plane = [_]f32{0} ** (m * n);
    var rhs0 = try pair.rhs(0);
    var rhs1 = try pair.rhs(1);
    quant.matmulTQ2_0F32RhsRange(&y, &x, &rhs0, m, n, 0, m);
    quant.matmulTQ2_0F32RhsRange(&y_plane, &x, &rhs1, m, n, 0, m);
    for (&y, y_plane) |*acc, v| acc.* += v;

    var rec: [n * k]f32 = undefined;
    try pair.reconstructInto(&rec);
    for (0..m) |r| {
        for (0..n) |c| {
            var want: f64 = 0;
            for (0..k) |j| want += @as(f64, x[r * k + j]) * @as(f64, rec[c * k + j]);
            try std.testing.expectApproxEqAbs(want, @as(f64, y[r * n + c]), 1e-3);
        }
    }
}

test "NaN weights degrade to zero trits without poisoning their group" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const k = 256;
    var prng = std.Random.DefaultPrng.init(5);
    var w_nan: [k]f32 = undefined;
    fillGaussian(&prng, &w_nan, 0.02);
    var w_zero = w_nan;
    w_nan[7] = std.math.nan(f32);
    w_zero[7] = 0;

    var pair_nan = try ptqtp.quantizeMatrix(&ctx, &w_nan, 1, k, .{});
    defer pair_nan.deinit(ctx.allocator);
    var pair_zero = try ptqtp.quantizeMatrix(&ctx, &w_zero, 1, k, .{});
    defer pair_zero.deinit(ctx.allocator);

    try std.testing.expect(std.mem.eql(
        u8,
        std.mem.sliceAsBytes(pair_nan.plane1),
        std.mem.sliceAsBytes(pair_zero.plane1),
    ));
    try std.testing.expect(std.mem.eql(
        u8,
        std.mem.sliceAsBytes(pair_nan.plane2),
        std.mem.sliceAsBytes(pair_zero.plane2),
    ));
    try std.testing.expect(std.math.isFinite(pair_nan.stats.rel_frob_err));
}

test "quantizeMatrix is deterministic across runs" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const n = 16;
    const k = 512;
    var prng = std.Random.DefaultPrng.init(2024);
    const w = try std.testing.allocator.alloc(f32, n * k);
    defer std.testing.allocator.free(w);
    fillGaussian(&prng, w, 0.03);

    var a = try ptqtp.quantizeMatrix(&ctx, w, n, k, .{});
    defer a.deinit(ctx.allocator);
    var b = try ptqtp.quantizeMatrix(&ctx, w, n, k, .{});
    defer b.deinit(ctx.allocator);

    try std.testing.expect(std.mem.eql(u8, std.mem.sliceAsBytes(a.plane1), std.mem.sliceAsBytes(b.plane1)));
    try std.testing.expect(std.mem.eql(u8, std.mem.sliceAsBytes(a.plane2), std.mem.sliceAsBytes(b.plane2)));
    try std.testing.expectEqual(a.stats.rel_frob_err, b.stats.rel_frob_err);
    try std.testing.expectEqual(a.stats.mean_iterations, b.stats.mean_iterations);
}

test "reference path: finer groups do not lose accuracy" {
    const n = 2;
    const k = 512;
    var prng = std.Random.DefaultPrng.init(77);
    var w: [n * k]f32 = undefined;
    fillGaussian(&prng, &w, 0.02);
    var rec: [n * k]f32 = undefined;

    const stats_128 = try ptqtp.reconstructReference(std.testing.allocator, &w, n, k, .{ .group_size = 128 }, &rec);
    try std.testing.expectApproxEqAbs(stats_128.rel_frob_err, relErr(&w, &rec), 1e-12);
    const stats_256 = try ptqtp.reconstructReference(std.testing.allocator, &w, n, k, .{ .group_size = 256 }, &rec);
    try std.testing.expectApproxEqAbs(stats_256.rel_frob_err, relErr(&w, &rec), 1e-12);

    try std.testing.expect(stats_128.rel_frob_err > 0);
    try std.testing.expect(stats_256.rel_frob_err > 0);
    // Halved groups double the scale freedom; alternating solves are local,
    // so allow a hair of slack instead of asserting strict dominance.
    try std.testing.expect(stats_128.rel_frob_err <= stats_256.rel_frob_err * 1.02);
    try std.testing.expectEqual(@as(usize, n * k / 128), stats_128.group_count);
}

test "all-zero matrix packs to zero scales and zero trits" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const k = 256;
    const w = [_]f32{0} ** (2 * k);
    var pair = try ptqtp.quantizeMatrix(&ctx, &w, 2, k, .{});
    defer pair.deinit(ctx.allocator);

    try std.testing.expectEqual(@as(f64, 0), pair.stats.rel_frob_err);
    try std.testing.expectEqual(@as(f64, 1), pair.stats.zero_frac[0]);
    try std.testing.expectEqual(@as(f64, 1), pair.stats.zero_frac[1]);
    for (pair.plane1) |block| {
        try std.testing.expectEqual(@as(u16, 0), block.d);
        for (block.qs) |q| try std.testing.expectEqual(@as(u8, 0b01_01_01_01), q);
    }
}

test "shape and option validation" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var w: [256]f32 = undefined;
    @memset(&w, 0.5);
    try std.testing.expectError(ptqtp.Error.InvalidOptions, ptqtp.quantizeMatrix(&ctx, &w, 1, 256, .{ .planes = 4 }));
    try std.testing.expectError(ptqtp.Error.InvalidOptions, ptqtp.quantizeMatrix(&ctx, &w, 1, 256, .{ .group_size = 128 }));
    try std.testing.expectError(ptqtp.Error.InvalidShape, ptqtp.quantizeMatrix(&ctx, w[0..128], 1, 128, .{}));
    try std.testing.expectError(ptqtp.Error.InvalidShape, ptqtp.quantizeMatrix(&ctx, w[0..128], 1, 256, .{}));

    var rec: [256]f32 = undefined;
    try std.testing.expectError(
        ptqtp.Error.InvalidShape,
        ptqtp.reconstructReference(std.testing.allocator, &w, 1, 256, .{ .group_size = 96 }, &rec),
    );
}

test "solveGroup handles a tiny group and reports signed scales faithfully" {
    // G = 4, single plane: t settles to sign(w) and alpha to the absmean of
    // the support, so reconstruction of a symmetric group is exact.
    const w = [_]f32{ 0.25, -0.25, 0.25, -0.25 };
    var t1: [4]i8 = undefined;
    var t2: [0]i8 = undefined;
    const res = ptqtp.solveGroup(&w, &t1, &t2, &t2, .{ .planes = 1, .group_size = 4 });
    try std.testing.expect(res.converged);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), res.alpha[0], 1e-6);
    const expected = [_]i8{ 1, -1, 1, -1 };
    for (t1, expected) |got, want| try std.testing.expectEqual(want, got);
}

test "tie_scales: exact ratio-3 alphas, valid balanced-trit decomposition" {
    // The tied fit is the optimal uniform 9-level quantizer: alphas must be
    // exactly [3s, s] pre-f16 and every code pair must reconstruct
    // c = 3*t1 + t2 in {-4..4} — the folding identity's preconditions.
    var prng = std.Random.DefaultPrng.init(0x600df01d);
    var w: [256]f32 = undefined;
    for (&w) |*v| v.* = prng.random().floatNorm(f32) * 0.04;

    var t1: [256]i8 = undefined;
    var t2: [256]i8 = undefined;
    var t3: [0]i8 = undefined;
    const res = ptqtp.solveGroup(&w, &t1, &t2, &t3, .{ .planes = 2, .tie_scales = true });
    try std.testing.expect(res.converged);
    try std.testing.expect(res.alpha[1] > 0);
    try std.testing.expectEqual(res.alpha[1] * 3, res.alpha[0]); // exact in f32
    for (t1, t2, w) |a, b, x| {
        try std.testing.expect(a >= -1 and a <= 1 and b >= -1 and b <= 1);
        const c = 3 * @as(i32, a) + b;
        try std.testing.expect(c >= -4 and c <= 4);
        // The code is the nearest 9-level point at step alpha[1].
        const want: i32 = @intFromFloat(std.math.clamp(@round(x / res.alpha[1]), -4, 4));
        try std.testing.expectEqual(want, c);
    }
    // Reconstruction error can't exceed half a step over the clamp-free range.
    for (t1, t2, w) |a, b, x| {
        const rec = res.alpha[0] * @as(f32, @floatFromInt(a)) + res.alpha[1] * @as(f32, @floatFromInt(b));
        if (@abs(x) < 4 * res.alpha[1]) {
            try std.testing.expect(@abs(x - rec) <= res.alpha[1] * 0.5 + 1e-6);
        }
    }
}

test "triple planes beat dual planes on gaussian weights, roughly threefold" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const n = 6;
    const k = 512;
    var prng = std.Random.DefaultPrng.init(4242);
    var w: [n * k]f32 = undefined;
    fillGaussian(&prng, &w, 0.03);

    var pair3 = try ptqtp.quantizeMatrix(&ctx, &w, n, k, .{ .planes = 3 });
    defer pair3.deinit(ctx.allocator);
    var pair2 = try ptqtp.quantizeMatrix(&ctx, &w, n, k, .{ .planes = 2 });
    defer pair2.deinit(ctx.allocator);

    try std.testing.expectEqual(@as(usize, 3), pair3.planeCount());
    // The 27-level high-rate bound sits ~3x below the 9-level one; the
    // alternating solve is local, so assert a conservative 2x.
    try std.testing.expect(pair3.stats.rel_frob_err < pair2.stats.rel_frob_err / 2.0);

    // Stats agree with the served three-plane reconstruction.
    var rec: [n * k]f32 = undefined;
    try pair3.reconstructInto(&rec);
    try std.testing.expectApproxEqAbs(pair3.stats.rel_frob_err, relErr(&w, &rec), 1e-12);
    // All three planes carry sparsity and the third has real support.
    try std.testing.expect(pair3.stats.zero_frac[2] > 0);
    try std.testing.expect(pair3.stats.zero_frac[2] < 1);

    // The third RHS view multiplies like the third plane's reconstruction.
    var rhs2 = try pair3.rhs(2);
    var x: [k]f32 = undefined;
    fillGaussian(&prng, &x, 1.0);
    var y = [_]f32{0} ** n;
    quant.matmulTQ2_0F32RhsRange(&y, &x, &rhs2, 1, n, 0, 1);
    var rec1: [n * k]f32 = undefined;
    @memset(&rec1, 0);
    var only3 = ptqtp.PlanePair{
        .plane1 = pair3.plane3,
        .plane2 = &.{},
        .plane3 = &.{},
        .rows = n,
        .cols = k,
        .stats = undefined,
    };
    try only3.reconstructInto(&rec1);
    for (0..n) |c| {
        var want: f64 = 0;
        for (0..k) |j| want += @as(f64, x[j]) * @as(f64, rec1[c * k + j]);
        try std.testing.expectApproxEqAbs(want, @as(f64, y[c]), 1e-3);
    }
}

test "planes=3 deterministic; planes=4 rejected" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const n = 4;
    const k = 256;
    var prng = std.Random.DefaultPrng.init(88);
    var w: [n * k]f32 = undefined;
    fillGaussian(&prng, &w, 0.05);

    var a = try ptqtp.quantizeMatrix(&ctx, &w, n, k, .{ .planes = 3 });
    defer a.deinit(ctx.allocator);
    var b = try ptqtp.quantizeMatrix(&ctx, &w, n, k, .{ .planes = 3 });
    defer b.deinit(ctx.allocator);
    try std.testing.expect(std.mem.eql(u8, std.mem.sliceAsBytes(a.plane3), std.mem.sliceAsBytes(b.plane3)));

    try std.testing.expectError(ptqtp.Error.InvalidOptions, ptqtp.quantizeMatrix(&ctx, &w, n, k, .{ .planes = 4 }));
}
