//! Behavioral tests for the cache-blocked packed f32 GEMM (`gemm_blocked.zig`):
//! tail-combination correctness vs a naive f64 reference, kc-boundary parity,
//! serial/parallel equivalence, transposed-orientation parity, the k == 0
//! zeroing path, and the dispatch-gate thresholds.
const std = @import("std");
const gemm_blocked = @import("gemm_blocked.zig");
const common = @import("common.zig");
const thread = @import("../../thread.zig");

const ParallelConfig = common.ParallelConfig;

const testing = std.testing;

fn fillPattern(values: []f32, seed: usize) void {
    for (values, 0..) |*value, idx| {
        const centered: i64 = @intCast((idx * 7 + seed * 13) % 23);
        value.* = @as(f32, @floatFromInt(centered - 11)) * 0.0625;
    }
}

// Naive f64-accumulated reference; `orient` follows the same source layouts
// as the kernel (nn: A[m,k] B[k,n]; tn: A[k,m] B[k,n]; nt: A[m,k] B[n,k]).
fn naiveGemm(
    comptime orient: gemm_blocked.Orientation,
    cd: []f32,
    ad: []const f32,
    bd: []const f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f64 = 0;
            for (0..k) |p| {
                const a_val: f64 = switch (orient) {
                    .plain, .trans_b => ad[i * k + p],
                    .trans_a => ad[p * m + i],
                };
                const b_val: f64 = switch (orient) {
                    .plain, .trans_a => bd[p * n + j],
                    .trans_b => bd[j * k + p],
                };
                acc += a_val * b_val;
            }
            cd[i * n + j] = @floatCast(acc);
        }
    }
}

fn expectBlockedMatchesNaive(
    pc: ParallelConfig,
    comptime orient: gemm_blocked.Orientation,
    allocator: std.mem.Allocator,
    m: usize,
    n: usize,
    k: usize,
    params: gemm_blocked.BlockParams,
    tolerance: f32,
) !void {
    const a_len = m * k;
    const b_len = k * n;
    const ad = try allocator.alloc(f32, a_len);
    defer allocator.free(ad);
    const bd = try allocator.alloc(f32, b_len);
    defer allocator.free(bd);
    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    const want = try allocator.alloc(f32, m * n);
    defer allocator.free(want);

    fillPattern(ad, m + 3 * k);
    fillPattern(bd, n + 5 * k);
    @memset(got, std.math.nan(f32));

    gemm_blocked.gemmBlockedWithParams(pc, orient, got, ad, bd, m, n, k, params);
    naiveGemm(orient, want, ad, bd, m, n, k);

    for (want, got) |w, g| {
        try testing.expectApproxEqAbs(w, g, tolerance);
    }
}

test "blocked gemm matches naive reference across every tail combination" {
    const allocator = testing.allocator;
    // m crosses mr (8 on aarch64, 6 elsewhere) and the tiny mc below; n
    // crosses nr (12 / 16) and the tiny nc; k crosses both kc choices.
    const ms = [_]usize{ 1, 7, 8, 9, 23 };
    const ns = [_]usize{ 1, 11, 12, 13, 37 };
    const ks = [_]usize{ 1, 63, 64, 65 };
    // Tiny blocking factors so the jc/pc/ic loops and the accumulate path all
    // run even at test sizes (k=65 -> three pc blocks, n=37 -> two jc blocks).
    const tiny: gemm_blocked.BlockParams = .{ .kc = 32, .mc = 16, .nc = 24 };

    inline for (.{ gemm_blocked.Orientation.plain, gemm_blocked.Orientation.trans_a, gemm_blocked.Orientation.trans_b }) |orient| {
        for (ms) |m| {
            for (ns) |n| {
                for (ks) |k| {
                    try expectBlockedMatchesNaive(.{}, orient, allocator, m, n, k, tiny, 1e-4);
                }
            }
        }
    }
}

test "blocked gemm matches naive reference at kc boundaries with default params" {
    const allocator = testing.allocator;
    const ks = [_]usize{ 255, 256, 257, 511, 512, 513 };
    for (ks) |k| {
        try expectBlockedMatchesNaive(.{}, .plain, allocator, 9, 13, k, .{}, 1e-3);
    }
}

test "blocked gemm parallel result matches serial result exactly" {
    const allocator = testing.allocator;
    const m = 70;
    const n = 50;
    const k = 96;

    const ad = try allocator.alloc(f32, m * k);
    defer allocator.free(ad);
    const bd = try allocator.alloc(f32, k * n);
    defer allocator.free(bd);
    const serial = try allocator.alloc(f32, m * n);
    defer allocator.free(serial);
    const pooled = try allocator.alloc(f32, m * n);
    defer allocator.free(pooled);
    fillPattern(ad, 1);
    fillPattern(bd, 2);

    const tiny: gemm_blocked.BlockParams = .{ .kc = 32, .mc = 16, .nc = 24 };
    gemm_blocked.gemmBlockedWithParams(.{}, .plain, serial, ad, bd, m, n, k, tiny);

    var pool: thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = 3 });
    defer pool.deinit();
    gemm_blocked.gemmBlockedWithParams(.{ .pool = &pool }, .plain, pooled, ad, bd, m, n, k, tiny);

    // Disjoint ic ownership means the split cannot change the arithmetic.
    try testing.expectEqualSlices(f32, serial, pooled);
}

test "blocked gemm large random shape stays within f64-reference tolerance" {
    const allocator = testing.allocator;
    const m = 160;
    const n = 131;
    const k = 300;

    const ad = try allocator.alloc(f32, m * k);
    defer allocator.free(ad);
    const bd = try allocator.alloc(f32, k * n);
    defer allocator.free(bd);
    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    const want = try allocator.alloc(f32, m * n);
    defer allocator.free(want);

    var prng = std.Random.DefaultPrng.init(0xb10c4ed);
    const rng = prng.random();
    for (ad) |*v| v.* = rng.float(f32) * 2 - 1;
    for (bd) |*v| v.* = rng.float(f32) * 2 - 1;

    gemm_blocked.gemmBlockedWithParams(.{}, .plain, got, ad, bd, m, n, k, .{ .kc = 128, .mc = 64, .nc = 96 });
    naiveGemm(.plain, want, ad, bd, m, n, k);

    // k = 300 values in [-1, 1): partial sums stay O(sqrt(k)); sequential
    // f32 FMA accumulation vs the f64 reference stays well under 1e-3.
    for (want, got) |w, g| {
        try testing.expectApproxEqAbs(w, g, 1e-3);
    }
}

test "blocked gemm transposed orientations match the plain orientation bitwise" {
    const allocator = testing.allocator;
    const m = 40;
    const n = 29;
    const k = 70;

    const ad = try allocator.alloc(f32, m * k);
    defer allocator.free(ad);
    const at = try allocator.alloc(f32, k * m);
    defer allocator.free(at);
    const bd = try allocator.alloc(f32, k * n);
    defer allocator.free(bd);
    const bt = try allocator.alloc(f32, n * k);
    defer allocator.free(bt);
    fillPattern(ad, 11);
    fillPattern(bd, 12);
    for (0..m) |i| {
        for (0..k) |p| at[p * m + i] = ad[i * k + p];
    }
    for (0..k) |p| {
        for (0..n) |j| bt[j * k + p] = bd[p * n + j];
    }

    const nn = try allocator.alloc(f32, m * n);
    defer allocator.free(nn);
    const tn = try allocator.alloc(f32, m * n);
    defer allocator.free(tn);
    const nt = try allocator.alloc(f32, m * n);
    defer allocator.free(nt);

    const tiny: gemm_blocked.BlockParams = .{ .kc = 32, .mc = 16, .nc = 24 };
    gemm_blocked.gemmBlockedWithParams(.{}, .plain, nn, ad, bd, m, n, k, tiny);
    gemm_blocked.gemmBlockedWithParams(.{}, .trans_a, tn, at, bd, m, n, k, tiny);
    gemm_blocked.gemmBlockedWithParams(.{}, .trans_b, nt, ad, bt, m, n, k, tiny);

    // Packing absorbs the transposes without changing the arithmetic order.
    try testing.expectEqualSlices(f32, nn, tn);
    try testing.expectEqualSlices(f32, nn, nt);
}

test "blocked gemm handles k == 0 by zeroing the output" {
    var cd = [_]f32{ 1, 2, 3, 4, 5, 6 };
    gemm_blocked.gemmBlockedWithParams(.{}, .plain, &cd, &.{}, &.{}, 2, 3, 0, .{});
    try testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, &cd);
}

test "blocked dispatch gate respects the work threshold and minimum dims" {
    // 768 x 512 x 512 == blocked_work_threshold exactly.
    try testing.expect(gemm_blocked.shouldUseBlocked(768, 512, 512));
    try testing.expect(!gemm_blocked.shouldUseBlocked(767, 512, 512));
    try testing.expect(!gemm_blocked.shouldUseBlocked(768, 512, 511));
    try testing.expect(!gemm_blocked.shouldUseBlocked(gemm_blocked.blocked_min_m - 1, 1 << 13, 1 << 13));
    try testing.expect(!gemm_blocked.shouldUseBlocked(1 << 13, gemm_blocked.blocked_min_n - 1, 1 << 13));
    try testing.expect(!gemm_blocked.shouldUseBlocked(1 << 13, 1 << 13, gemm_blocked.blocked_min_k - 1));
    try testing.expect(gemm_blocked.shouldUseBlocked(gemm_blocked.blocked_min_m, 1 << 13, 1 << 13));
}
