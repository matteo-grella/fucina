//! Behavioral tests for the shaped elementwise kernels in `elementwise.zig`:
//! the per-channel Snake activation and the ggml-semantics GroupNorm, checked
//! against hand-rolled naive references at SIMD-awkward widths.

const std = @import("std");
const elementwise = @import("elementwise.zig");
const tensor = @import("../../tensor.zig");

const Tensor = tensor.Tensor;

const snakeIntoWithConfig = elementwise.snakeIntoWithConfig;
const snakeBackwardInputIntoWithConfig = elementwise.snakeBackwardInputIntoWithConfig;
const snakeBackwardParamsIntoWithConfig = elementwise.snakeBackwardParamsIntoWithConfig;
const groupNormIntoWithConfig = elementwise.groupNormIntoWithConfig;
const groupNormBackwardIntoWithConfig = elementwise.groupNormBackwardIntoWithConfig;

/// Deterministic pseudo-random fill (splitmix-style) in [-2, 2).
fn fillPseudoRandom(values: []f32, seed: u64) void {
    var state = seed;
    for (values) |*v| {
        state +%= 0x9e3779b97f4a7c15;
        var z = state;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        z ^= z >> 31;
        const unit = @as(f32, @floatFromInt(z >> 40)) / @as(f32, @floatFromInt(@as(u64, 1) << 24));
        v.* = unit * 4.0 - 2.0;
    }
}

fn naiveSnake(out: []f32, x: []const f32, alpha: []const f32, inv_b: []const f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        for (0..cols) |c| {
            const v = x[r * cols + c];
            const s = @sin(alpha[c] * v);
            out[r * cols + c] = v + inv_b[c] * s * s;
        }
    }
}

test "snake vector kernel matches naive reference at SIMD-awkward widths" {
    const allocator = std.testing.allocator;

    // cols not a multiple of any SIMD width (13), plus a wide case (37).
    const cases = [_][2]usize{ .{ 5, 13 }, .{ 3, 1 }, .{ 7, 37 }, .{ 1, 8 } };
    for (cases, 0..) |case, case_i| {
        const rows = case[0];
        const cols = case[1];

        var x = try Tensor.zeros(allocator, &.{ rows, cols });
        defer x.deinit();
        fillPseudoRandom(x.data(), 100 + case_i);
        const alpha = try allocator.alloc(f32, cols);
        defer allocator.free(alpha);
        fillPseudoRandom(alpha, 200 + case_i);
        // Shift alpha away from 0 so inv_b = 1/(alpha + 1e-9) stays bounded.
        for (alpha) |*a| a.* = @abs(a.*) + 0.25;
        const inv_b = try allocator.alloc(f32, cols);
        defer allocator.free(inv_b);
        for (inv_b, alpha) |*ib, a| ib.* = 1.0 / (a + 1e-9);

        var out = try Tensor.zeros(allocator, &.{ rows, cols });
        defer out.deinit();
        snakeIntoWithConfig(&out, &x, alpha, inv_b, rows, cols, .{});

        const want = try allocator.alloc(f32, rows * cols);
        defer allocator.free(want);
        naiveSnake(want, x.dataConst(), alpha, inv_b, rows, cols);

        for (want, out.dataConst()) |w, g| {
            try std.testing.expectApproxEqAbs(w, g, 1e-6);
        }
    }
}

test "snake vector kernel hand-computed values (alpha=1, inv_b=1)" {
    const allocator = std.testing.allocator;

    var x = try Tensor.fromSlice(allocator, &.{ 2, 2 }, &.{ 0.5, -1.0, 2.0, 0.0 });
    defer x.deinit();
    var out = try Tensor.zeros(allocator, &.{ 2, 2 });
    defer out.deinit();
    snakeIntoWithConfig(&out, &x, &.{ 1, 1 }, &.{ 1, 1 }, 2, 2, .{});

    // y = x + sin(x)^2
    const inputs = [_]f32{ 0.5, -1.0, 2.0, 0.0 };
    for (inputs, out.dataConst()) |v, g| {
        const s = @sin(v);
        try std.testing.expectApproxEqAbs(v + s * s, g, 1e-6);
    }
}

fn naiveGroupNorm(
    out: []f32,
    x: []const f32,
    weight: ?[]const f32,
    bias: ?[]const f32,
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
) void {
    const cpg = cols / groups;
    const count: f64 = @floatFromInt(rows * cpg);
    for (0..groups) |g| {
        var sum: f64 = 0;
        for (0..rows) |r| {
            for (0..cpg) |lc| sum += x[r * cols + g * cpg + lc];
        }
        const mean: f32 = @floatCast(sum / count);
        var sum2: f64 = 0;
        for (0..rows) |r| {
            for (0..cpg) |lc| {
                const centered = x[r * cols + g * cpg + lc] - mean;
                sum2 += @as(f64, centered) * @as(f64, centered);
            }
        }
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatCast(sum2 / count + eps)));
        for (0..rows) |r| {
            for (0..cpg) |lc| {
                const c = g * cpg + lc;
                var value = (x[r * cols + c] - mean) * scale;
                if (weight) |w| value *= w[c];
                if (bias) |b| value += b[c];
                out[r * cols + c] = value;
            }
        }
    }
}

test "groupNorm vector kernel matches naive f64 two-pass reference" {
    const allocator = std.testing.allocator;

    // (rows, cols, groups): G=1, G=C (instance norm over time), G in between,
    // plus a SIMD-awkward channel count.
    const cases = [_][3]usize{ .{ 3, 8, 1 }, .{ 4, 6, 6 }, .{ 3, 12, 4 }, .{ 5, 13, 13 }, .{ 2, 10, 2 } };
    for (cases, 0..) |case, case_i| {
        const rows = case[0];
        const cols = case[1];
        const groups = case[2];
        const eps: f32 = 1e-5;

        var x = try Tensor.zeros(allocator, &.{ rows, cols });
        defer x.deinit();
        fillPseudoRandom(x.data(), 300 + case_i);

        // Plain (no affine).
        var out = try Tensor.zeros(allocator, &.{ rows, cols });
        defer out.deinit();
        groupNormIntoWithConfig(&out, &x, null, null, rows, cols, groups, eps, .{});

        const want = try allocator.alloc(f32, rows * cols);
        defer allocator.free(want);
        naiveGroupNorm(want, x.dataConst(), null, null, rows, cols, groups, eps);
        try std.testing.expectEqualSlices(f32, want, out.dataConst());

        // Affine (weight + bias applied after normalization).
        const weight = try allocator.alloc(f32, cols);
        defer allocator.free(weight);
        fillPseudoRandom(weight, 400 + case_i);
        const bias = try allocator.alloc(f32, cols);
        defer allocator.free(bias);
        fillPseudoRandom(bias, 500 + case_i);

        var out_affine = try Tensor.zeros(allocator, &.{ rows, cols });
        defer out_affine.deinit();
        groupNormIntoWithConfig(&out_affine, &x, weight, bias, rows, cols, groups, eps, .{});
        naiveGroupNorm(want, x.dataConst(), weight, bias, rows, cols, groups, eps);
        try std.testing.expectEqualSlices(f32, want, out_affine.dataConst());
    }
}

test "groupNorm vector kernel hand-computed G=1 statistics" {
    const allocator = std.testing.allocator;

    // All four elements are one group: mean 2.5, biased var 1.25.
    var x = try Tensor.fromSlice(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var out = try Tensor.zeros(allocator, &.{ 2, 2 });
    defer out.deinit();
    const eps: f32 = 1e-5;
    groupNormIntoWithConfig(&out, &x, null, null, 2, 2, 1, eps, .{});

    const inv = 1.0 / @sqrt(@as(f32, 1.25) + eps);
    const expected = [_]f32{ -1.5 * inv, -0.5 * inv, 0.5 * inv, 1.5 * inv };
    for (expected, out.dataConst()) |w, g| {
        try std.testing.expectApproxEqAbs(w, g, 1e-6);
    }
}

fn naiveSnakeBackward(
    gx: []f32,
    ga: []f32,
    gib: []f32,
    x: []const f32,
    gy: []const f32,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
) void {
    @memset(ga, 0);
    @memset(gib, 0);
    for (0..rows) |r| {
        for (0..cols) |c| {
            const v = x[r * cols + c];
            const g = gy[r * cols + c];
            const s = @sin(alpha[c] * v);
            const s2 = @sin(2 * alpha[c] * v);
            gx[r * cols + c] = g * (1 + inv_b[c] * alpha[c] * s2);
            ga[c] += g * inv_b[c] * v * s2;
            gib[c] += g * s * s;
        }
    }
}

test "snake backward vector kernels match naive reference at SIMD-awkward widths" {
    const allocator = std.testing.allocator;

    // cols = 13 exercises SIMD body + tail; 1 and 37 the degenerate/wide arms.
    const cases = [_][2]usize{ .{ 5, 13 }, .{ 3, 1 }, .{ 7, 37 }, .{ 1, 8 } };
    for (cases, 0..) |case, case_i| {
        const rows = case[0];
        const cols = case[1];

        var x = try Tensor.zeros(allocator, &.{ rows, cols });
        defer x.deinit();
        fillPseudoRandom(x.data(), 600 + case_i);
        var gy = try Tensor.zeros(allocator, &.{ rows, cols });
        defer gy.deinit();
        fillPseudoRandom(gy.data(), 700 + case_i);
        const alpha = try allocator.alloc(f32, cols);
        defer allocator.free(alpha);
        fillPseudoRandom(alpha, 800 + case_i);
        for (alpha) |*a| a.* = @abs(a.*) + 0.25;
        const inv_b = try allocator.alloc(f32, cols);
        defer allocator.free(inv_b);
        for (inv_b, alpha) |*ib, a| ib.* = 1.0 / (a + 1e-9);

        var gx = try Tensor.zeros(allocator, &.{ rows, cols });
        defer gx.deinit();
        snakeBackwardInputIntoWithConfig(&gx, &x, &gy, alpha, inv_b, rows, cols, .{});

        var ga = try Tensor.zeros(allocator, &.{cols});
        defer ga.deinit();
        var gib = try Tensor.zeros(allocator, &.{cols});
        defer gib.deinit();
        snakeBackwardParamsIntoWithConfig(&ga, &gib, &x, &gy, alpha, inv_b, rows, cols, .{});

        const want_gx = try allocator.alloc(f32, rows * cols);
        defer allocator.free(want_gx);
        const want_ga = try allocator.alloc(f32, cols);
        defer allocator.free(want_ga);
        const want_gib = try allocator.alloc(f32, cols);
        defer allocator.free(want_gib);
        naiveSnakeBackward(want_gx, want_ga, want_gib, x.dataConst(), gy.dataConst(), alpha, inv_b, rows, cols);

        for (want_gx, gx.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);
        for (want_ga, ga.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);
        for (want_gib, gib.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);
    }
}

/// Naive GroupNorm VJP with the forward's f64 two-pass statistics: recompute
/// mean/scale, then dx = scale*(ĝ − mean(ĝ) − x̂·mean(ĝ·x̂)),
/// dw = Σ gy·x̂, db = Σ gy.
fn naiveGroupNormBackward(
    gx: []f32,
    gw: []f32,
    gb: []f32,
    x: []const f32,
    gy: []const f32,
    weight: ?[]const f32,
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
) void {
    const cpg = cols / groups;
    const count: f64 = @floatFromInt(rows * cpg);
    for (0..groups) |g| {
        const col_start = g * cpg;
        var sum: f64 = 0;
        for (0..rows) |r| {
            for (0..cpg) |lc| sum += x[r * cols + col_start + lc];
        }
        const mean: f32 = @floatCast(sum / count);
        var sum2: f64 = 0;
        for (0..rows) |r| {
            for (0..cpg) |lc| {
                const centered = x[r * cols + col_start + lc] - mean;
                sum2 += @as(f64, centered) * @as(f64, centered);
            }
        }
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatCast(sum2 / count + eps)));

        var sum_g: f64 = 0;
        var sum_gx: f64 = 0;
        for (0..rows) |r| {
            for (0..cpg) |lc| {
                const c = col_start + lc;
                const wv: f32 = if (weight) |w| w[c] else 1.0;
                const gh = gy[r * cols + c] * wv;
                const xh = (x[r * cols + c] - mean) * scale;
                sum_g += gh;
                sum_gx += @as(f64, gh) * @as(f64, xh);
            }
        }
        const mean_g: f32 = @floatCast(sum_g / count);
        const mean_gx: f32 = @floatCast(sum_gx / count);
        for (0..cpg) |lc| {
            const c = col_start + lc;
            var acc_w: f32 = 0;
            var acc_b: f32 = 0;
            for (0..rows) |r| {
                const wv: f32 = if (weight) |w| w[c] else 1.0;
                const gh = gy[r * cols + c] * wv;
                const xh = (x[r * cols + c] - mean) * scale;
                gx[r * cols + c] = scale * (gh - mean_g - xh * mean_gx);
                acc_w += gy[r * cols + c] * xh;
                acc_b += gy[r * cols + c];
            }
            gw[c] = acc_w;
            gb[c] = acc_b;
        }
    }
}

test "groupNorm backward vector kernel matches naive f64 two-pass reference" {
    const allocator = std.testing.allocator;

    // (rows, cols, groups) incl. SIMD-awkward channel counts; plain + affine.
    const cases = [_][3]usize{ .{ 3, 8, 1 }, .{ 4, 6, 6 }, .{ 3, 12, 4 }, .{ 5, 13, 13 }, .{ 2, 10, 2 } };
    for (cases, 0..) |case, case_i| {
        const rows = case[0];
        const cols = case[1];
        const groups = case[2];
        const eps: f32 = 1e-5;

        var x = try Tensor.zeros(allocator, &.{ rows, cols });
        defer x.deinit();
        fillPseudoRandom(x.data(), 900 + case_i);
        var gy = try Tensor.zeros(allocator, &.{ rows, cols });
        defer gy.deinit();
        fillPseudoRandom(gy.data(), 1000 + case_i);
        const weight = try allocator.alloc(f32, cols);
        defer allocator.free(weight);
        fillPseudoRandom(weight, 1100 + case_i);

        const want_gx = try allocator.alloc(f32, rows * cols);
        defer allocator.free(want_gx);
        const want_gw = try allocator.alloc(f32, cols);
        defer allocator.free(want_gw);
        const want_gb = try allocator.alloc(f32, cols);
        defer allocator.free(want_gb);

        for ([_]bool{ false, true }) |affine| {
            const w_opt: ?[]const f32 = if (affine) weight else null;

            var gx = try Tensor.zeros(allocator, &.{ rows, cols });
            defer gx.deinit();
            var gw = try Tensor.zeros(allocator, &.{cols});
            defer gw.deinit();
            var gb = try Tensor.zeros(allocator, &.{cols});
            defer gb.deinit();
            groupNormBackwardIntoWithConfig(&gx, &gw, &gb, &x, &gy, w_opt, rows, cols, groups, eps, .{});

            naiveGroupNormBackward(want_gx, want_gw, want_gb, x.dataConst(), gy.dataConst(), w_opt, rows, cols, groups, eps);

            for (want_gx, gx.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);
            for (want_gw, gw.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);
            for (want_gb, gb.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);

            // Null outputs are skipped without touching the others.
            var gx_only = try Tensor.zeros(allocator, &.{ rows, cols });
            defer gx_only.deinit();
            groupNormBackwardIntoWithConfig(&gx_only, null, null, &x, &gy, w_opt, rows, cols, groups, eps, .{});
            try std.testing.expectEqualSlices(f32, gx.dataConst(), gx_only.dataConst());
        }
    }
}
