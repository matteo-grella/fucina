//! Stats kernels through ExecContext: max/min/var against naive references,
//! NaN contracts for extrema and argmax/topK, and sort with exact indices.
//! Force-imported by `exec.zig`.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const exec = @import("../exec.zig");
const exec_elementwise = @import("elementwise.zig");
const exec_row_ops = @import("../backend.zig").rows;
const dtype_mod = @import("../dtype.zig");
const fpenv = @import("../fpenv.zig");
const parallel = @import("../parallel.zig");
const rng = @import("../rng.zig");
const tensor = @import("../tensor.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;
const ExecContext = exec.ExecContext;
const CrossEntropyOptions = exec.CrossEntropyOptions;
const Reduction = exec.Reduction;

const util = @import("test_util.zig");
const expectCloseToF64 = util.expectCloseToF64;

test "exec context max min var match naive references" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x3a7c);
    const random = prng.random();

    // inner == 1 SIMD rows (axis_dim 11 exercises the scalar tail; 4099 the
    // vector body) and the inner>1 scalar layout.
    const cases = [_][3]usize{
        .{ 4, 11, 1 },
        .{ 2, 4099, 1 },
        .{ 3, 7, 2 },
        .{ 1, 1, 1 },
    };
    inline for (.{ 2, 3 }) |rank| {
        for (cases) |case| {
            const outer = case[0];
            const axis_dim = case[1];
            const inner = case[2];
            if ((rank == 2) != (inner == 1)) continue;
            const data = try allocator.alloc(f32, outer * axis_dim * inner);
            defer allocator.free(data);
            for (data) |*value| value.* = random.floatNorm(f32) * 3;

            var shape: [rank]usize = undefined;
            shape[0] = outer;
            shape[1] = axis_dim;
            if (rank == 3) shape[2] = inner;
            var x = try ctx.fromSlice(.f32, shape, data);
            defer x.deinit();

            var max_result = try ctx.maxAxis(.f32, rank, &x, 1);
            defer max_result.deinit();
            var min_result = try ctx.minAxis(.f32, rank, &x, 1);
            defer min_result.deinit();

            for (0..outer) |outer_i| {
                const base = outer_i * axis_dim * inner;
                for (0..inner) |inner_i| {
                    var max_i: usize = 0;
                    var min_i: usize = 0;
                    var max_value = data[base + inner_i];
                    var min_value = data[base + inner_i];
                    for (1..axis_dim) |axis_i| {
                        const value = data[base + axis_i * inner + inner_i];
                        if (value > max_value) {
                            max_value = value;
                            max_i = axis_i;
                        }
                        if (value < min_value) {
                            min_value = value;
                            min_i = axis_i;
                        }
                    }
                    const flat = outer_i * inner + inner_i;
                    try std.testing.expectEqual(max_value, max_result.values.dataConst()[flat]);
                    try std.testing.expectEqual(@as(i64, @intCast(max_i)), max_result.indices.dataConst()[flat]);
                    try std.testing.expectEqual(min_value, min_result.values.dataConst()[flat]);
                    try std.testing.expectEqual(@as(i64, @intCast(min_i)), min_result.indices.dataConst()[flat]);
                }
            }

            for ([_]u1{ 0, 1 }) |ddof| {
                var v = try ctx.varAxis(.f32, rank, &x, 1, ddof);
                defer v.deinit();
                const n = @as(f64, @floatFromInt(axis_dim));
                for (0..outer) |outer_i| {
                    const base = outer_i * axis_dim * inner;
                    for (0..inner) |inner_i| {
                        var sum_acc: f64 = 0;
                        for (0..axis_dim) |axis_i| sum_acc += data[base + axis_i * inner + inner_i];
                        const mean_value = sum_acc / n;
                        var sumsq: f64 = 0;
                        for (0..axis_dim) |axis_i| {
                            const centered = @as(f64, data[base + axis_i * inner + inner_i]) - mean_value;
                            sumsq += centered * centered;
                        }
                        const got = v.dataConst()[outer_i * inner + inner_i];
                        if (axis_dim == 1 and ddof == 1) {
                            // torch.var on one element with Bessel: 0/0 = NaN.
                            try std.testing.expect(std.math.isNan(got));
                        } else {
                            try expectCloseToF64(sumsq / (n - @as(f64, @floatFromInt(ddof))), got, 5e-4, 5e-6);
                        }
                    }
                }
            }
        }
    }

    // Tie-break: duplicate extrema report the FIRST index on both layouts.
    var ties = try ctx.fromSlice(.f32, .{ 2, 4 }, &.{
        1, 3,  3, 2,
        5, -1, 5, 4,
    });
    defer ties.deinit();
    var tie_max = try ctx.maxAxis(.f32, 2, &ties, 1);
    defer tie_max.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 5 }, tie_max.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, tie_max.indices.dataConst());
    var tie_min = try ctx.fromSlice(.f32, .{ 1, 5 }, &.{ 4, -2, 7, -2, 0 });
    defer tie_min.deinit();
    var tie_min_result = try ctx.minAxis(.f32, 2, &tie_min, 1);
    defer tie_min_result.deinit();
    try std.testing.expectEqualSlices(f32, &.{-2}, tie_min_result.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{1}, tie_min_result.indices.dataConst());

    // Tie on axis 0 with inner > 1 (the scalar layout).
    var ties_inner = try ctx.fromSlice(.f32, .{ 3, 2 }, &.{
        2, 1,
        2, 5,
        0, 5,
    });
    defer ties_inner.deinit();
    var tie_axis0 = try ctx.maxAxis(.f32, 2, &ties_inner, 0);
    defer tie_axis0.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 5 }, tie_axis0.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 0, 1 }, tie_axis0.indices.dataConst());
}

test "max/min over an axis: NaN drops and all-NaN rows degrade identically on both layouts" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // One NaN contract for both layouts (see maxAxis): NaN never wins —
    // regardless of position (a leading NaN used to poison the inner > 1
    // path, which seeded from input[first]); an all-NaN row degrades to
    // -inf/+inf with index 0.
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    const row_a = [_]f32{ nan, 2, -1, 7, 7, 0, 3, -5, 1, 2, 6 };
    const row_b = [_]f32{ 4, nan, nan, -2, 9, nan, 9, -8, nan, 5, -8 };
    const row_nan = [_]f32{nan} ** 11;

    // inner == 1 layout (SIMD body + scalar tail at axis_dim 11).
    var rows: [3 * 11]f32 = undefined;
    @memcpy(rows[0..11], &row_a);
    @memcpy(rows[11..22], &row_b);
    @memcpy(rows[22..33], &row_nan);
    var x = try ctx.fromSlice(.f32, .{ 3, 11 }, &rows);
    defer x.deinit();
    var mx = try ctx.maxAxis(.f32, 2, &x, 1);
    defer mx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 7, 9, -inf }, mx.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 3, 4, 0 }, mx.indices.dataConst());
    var mn = try ctx.minAxis(.f32, 2, &x, 1);
    defer mn.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -5, -8, inf }, mn.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 7, 7, 0 }, mn.indices.dataConst());

    // inner > 1 layout (generic strided path): the same rows as columns of a
    // {11, 3} tensor reduced over axis 0 — value and index semantics must be
    // identical to the inner == 1 results above.
    var cols: [11 * 3]f32 = undefined;
    for (0..11) |i| {
        cols[i * 3 + 0] = row_a[i];
        cols[i * 3 + 1] = row_b[i];
        cols[i * 3 + 2] = row_nan[i];
    }
    var xt = try ctx.fromSlice(.f32, .{ 11, 3 }, &cols);
    defer xt.deinit();
    var mxt = try ctx.maxAxis(.f32, 2, &xt, 0);
    defer mxt.deinit();
    try std.testing.expectEqualSlices(f32, mx.values.dataConst(), mxt.values.dataConst());
    try std.testing.expectEqualSlices(i64, mx.indices.dataConst(), mxt.indices.dataConst());
    var mnt = try ctx.minAxis(.f32, 2, &xt, 0);
    defer mnt.deinit();
    try std.testing.expectEqualSlices(f32, mn.values.dataConst(), mnt.values.dataConst());
    try std.testing.expectEqualSlices(i64, mn.indices.dataConst(), mnt.indices.dataConst());
}

test "argmax/topK over an axis: NaN never places, matching the max contract" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Same rows and contract as the max/min test above: a NaN never wins
    // (a leading NaN used to poison argmax, which seeded from input[first],
    // and used to land in topK slot 0, which admitted via `value <= min`);
    // an all-NaN row degrades to index 0 (argmax) / (-inf, 0) slots (topK).
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    const row_a = [_]f32{ nan, 2, -1, 7, 7, 0, 3, -5, 1, 2, 6 };
    const row_b = [_]f32{ 4, nan, nan, -2, 9, nan, 9, -8, nan, 5, -8 };
    const row_nan = [_]f32{nan} ** 11;

    var rows: [3 * 11]f32 = undefined;
    @memcpy(rows[0..11], &row_a);
    @memcpy(rows[11..22], &row_b);
    @memcpy(rows[22..33], &row_nan);
    var x = try ctx.fromSlice(.f32, .{ 3, 11 }, &rows);
    defer x.deinit();

    var arg = try ctx.argmax(.f32, 2, &x, 1);
    defer arg.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 3, 4, 0 }, arg.dataConst());

    var top = try ctx.topK(2, &x, 1, 3);
    defer top.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 7, 7, 6, 9, 9, 5, -inf, -inf, -inf }, top.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 3, 4, 10, 4, 6, 9, 0, 0, 0 }, top.indices.dataConst());

    // The same rows as columns of a {11, 3} tensor reduced over axis 0:
    // identical winners in the strided layout.
    var cols: [11 * 3]f32 = undefined;
    for (0..11) |i| {
        cols[i * 3 + 0] = row_a[i];
        cols[i * 3 + 1] = row_b[i];
        cols[i * 3 + 2] = row_nan[i];
    }
    var xt = try ctx.fromSlice(.f32, .{ 11, 3 }, &cols);
    defer xt.deinit();

    var argt = try ctx.argmax(.f32, 2, &xt, 0);
    defer argt.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 3, 4, 0 }, argt.dataConst());

    var topt = try ctx.topK(2, &xt, 0, 3);
    defer topt.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 7, 9, -inf, 7, 9, -inf, 6, 5, -inf }, topt.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 3, 4, 0, 4, 6, 0, 10, 9, 0 }, topt.indices.dataConst());
}

test "exec sort orders rows both directions with NaN last and exact indices" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // torch.sort(x, dim=1): values {{1,3,4,7},{-2,0,5,8}},
    // indices {{2,1,0,3},{1,3,2,0}} (all values distinct, no tie ambiguity).
    var x = try ctx.fromSlice(.f32, .{ 2, 4 }, &.{ 4, 3, 1, 7, 8, -2, 5, 0 });
    defer x.deinit();
    var asc = try ctx.sort(2, &x, 1, false);
    defer asc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 4, 7, -2, 0, 5, 8 }, asc.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 0, 3, 1, 3, 2, 0 }, asc.indices.dataConst());

    // torch.sort(x, dim=1, descending=True).
    var desc = try ctx.sort(2, &x, 1, true);
    defer desc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 7, 4, 3, 1, 8, 5, 0, -2 }, desc.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 3, 0, 1, 2, 0, 2, 3, 1 }, desc.indices.dataConst());

    // Axis-0 sort exercises the inner > 1 (strided) layout: columns sorted
    // independently — torch.sort(x, dim=0).
    var cols = try ctx.sort(2, &x, 0, false);
    defer cols.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 4, -2, 1, 0, 8, 3, 5, 7 }, cols.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 1, 1, 0, 1, 0 }, cols.indices.dataConst());

    // NaN contract: NaN sorts LAST in BOTH directions (diverges from torch,
    // which puts NaN first when descending).
    const nan = std.math.nan(f32);
    var with_nan = try ctx.fromSlice(.f32, .{4}, &.{ 2, nan, 1, 3 });
    defer with_nan.deinit();
    var nan_asc = try ctx.sort(1, &with_nan, 0, false);
    defer nan_asc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, nan_asc.values.dataConst()[0..3]);
    try std.testing.expect(std.math.isNan(nan_asc.values.dataConst()[3]));
    try std.testing.expectEqualSlices(i64, &.{ 2, 0, 3, 1 }, nan_asc.indices.dataConst());
    var nan_desc = try ctx.sort(1, &with_nan, 0, true);
    defer nan_desc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 2, 1 }, nan_desc.values.dataConst()[0..3]);
    try std.testing.expect(std.math.isNan(nan_desc.values.dataConst()[3]));
    try std.testing.expectEqualSlices(i64, &.{ 3, 0, 2, 1 }, nan_desc.indices.dataConst());
}

// ---- Non-last-axis lane kernels vs the retired scalar strided arms ----
//
// The oracles below are the strided scalar loops the inner-lane kernels
// replaced (one scalar accumulator per output lane, stride `inner`), kept
// verbatim: the kernels promise the same per-element accumulation order,
// so the comparison is bitwise.

fn oracleVarianceStrided(data: []const f32, out: []f32, outer: usize, axis_dim: usize, inner: usize, ddof: u1) void {
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    const inv_denom = 1 / (@as(f32, @floatFromInt(axis_dim)) - @as(f32, @floatFromInt(ddof)));
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sum_acc: f32 = 0;
            for (0..axis_dim) |axis_i| {
                sum_acc += data[base + axis_i * inner + inner_i];
            }
            const mean_value = sum_acc * inv_axis_dim;
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const centered = data[base + axis_i * inner + inner_i] - mean_value;
                sumsq += centered * centered;
            }
            out[outer_i * inner + inner_i] = sumsq * inv_denom;
        }
    }
}

fn oracleStandardizeStrided(comptime Acc: type, input: []const f32, output: []f32, outer: usize, axis_dim: usize, inner: usize, valid_count: usize, options: exec.StandardizeOptions) void {
    const ddof_count: usize = options.ddof;
    const eps: Acc = @floatCast(options.eps);
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            if (valid_count == 0) {
                for (0..axis_dim) |axis_i| output[base + axis_i * inner + inner_i] = 0;
                continue;
            }
            var sum_acc: Acc = 0;
            for (0..valid_count) |axis_i| {
                sum_acc += @floatCast(input[base + axis_i * inner + inner_i]);
            }
            const count: Acc = @floatFromInt(valid_count);
            const mean_value = sum_acc / count;
            var variance: Acc = 0;
            if (valid_count > ddof_count) {
                var sumsq: Acc = 0;
                for (0..valid_count) |axis_i| {
                    const centered = @as(Acc, @floatCast(input[base + axis_i * inner + inner_i])) - mean_value;
                    sumsq += centered * centered;
                }
                variance = sumsq / @as(Acc, @floatFromInt(valid_count - ddof_count));
            }
            const std_value = @sqrt(variance);
            const denom = switch (options.eps_mode) {
                .outside_sqrt => std_value + eps,
                .inside_sqrt => @sqrt(variance + eps),
            };
            for (0..valid_count) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const centered = @as(Acc, @floatCast(input[offset])) - mean_value;
                output[offset] = @floatCast(centered / denom);
            }
            for (valid_count..axis_dim) |axis_i| {
                output[base + axis_i * inner + inner_i] = 0;
            }
        }
    }
}

fn oracleStandardizeBackwardStrided(comptime Acc: type, input: []const f32, upstream: []const f32, output: []f32, outer: usize, axis_dim: usize, inner: usize, valid_count: usize, options: exec.StandardizeOptions) void {
    @memset(output, 0);
    const ddof_count: usize = options.ddof;
    const eps: Acc = @floatCast(options.eps);
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            if (valid_count == 0) continue;
            var sum_acc: Acc = 0;
            for (0..valid_count) |axis_i| {
                sum_acc += @floatCast(input[base + axis_i * inner + inner_i]);
            }
            const count: Acc = @floatFromInt(valid_count);
            const mean_value = sum_acc / count;
            var variance: Acc = 0;
            if (valid_count > ddof_count) {
                var sumsq: Acc = 0;
                for (0..valid_count) |axis_i| {
                    const centered = @as(Acc, @floatCast(input[base + axis_i * inner + inner_i])) - mean_value;
                    sumsq += centered * centered;
                }
                variance = sumsq / @as(Acc, @floatFromInt(valid_count - ddof_count));
            }
            const std_value = @sqrt(variance);
            const denom = switch (options.eps_mode) {
                .outside_sqrt => std_value + eps,
                .inside_sqrt => @sqrt(variance + eps),
            };
            const inv_denom = 1 / denom;
            var grad_sum: Acc = 0;
            var centered_grad_dot: Acc = 0;
            for (0..valid_count) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const g = @as(Acc, @floatCast(upstream[offset]));
                const centered = @as(Acc, @floatCast(input[offset])) - mean_value;
                grad_sum += g;
                centered_grad_dot += g * centered;
            }
            const mean_grad = grad_sum / count;
            var second_scale: Acc = 0;
            if (valid_count > ddof_count and variance > 0) {
                const denom_count: Acc = @floatFromInt(valid_count - ddof_count);
                second_scale = switch (options.eps_mode) {
                    .outside_sqrt => centered_grad_dot / (denom_count * std_value * denom * denom),
                    .inside_sqrt => centered_grad_dot / (denom_count * denom * denom * denom),
                };
            }
            for (0..valid_count) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const centered = @as(Acc, @floatCast(input[offset])) - mean_value;
                output[offset] = @floatCast((@as(Acc, @floatCast(upstream[offset])) - mean_grad) * inv_denom - centered * second_scale);
            }
        }
    }
}

fn testFillRandom(data: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (data) |*value| value.* = random.floatNorm(f32) * 3;
}

fn expectBitwiseF32(expected: []const f32, actual: []const f32) !void {
    try std.testing.expectEqualSlices(u32, @ptrCast(expected), @ptrCast(actual));
}

/// Every non-last axis of one shape, through varAxis / standardize /
/// standardizeBackward, against the strided oracles.
fn expectStatsLaneKernelsMatchOracle(ctx: *ExecContext, comptime rank: usize, shape: [rank]usize, seed: u64) !void {
    const allocator = std.testing.allocator;
    var len: usize = 1;
    for (shape) |dim| len *= dim;
    const data = try allocator.alloc(f32, len);
    defer allocator.free(data);
    testFillRandom(data, seed);
    const grad = try allocator.alloc(f32, len);
    defer allocator.free(grad);
    testFillRandom(grad, seed + 1);
    const expected = try allocator.alloc(f32, len);
    defer allocator.free(expected);

    var x = try ctx.fromSlice(.f32, shape, data);
    defer x.deinit();
    var gy = try ctx.fromSlice(.f32, shape, grad);
    defer gy.deinit();

    inline for (0..rank - 1) |axis| {
        const axis_dim = shape[axis];
        var outer: usize = 1;
        for (0..axis) |dim_i| outer *= shape[dim_i];
        const inner = len / (outer * axis_dim);

        for ([_]u1{ 0, 1 }) |ddof| {
            var got = try ctx.varAxis(.f32, rank, &x, axis, ddof);
            defer got.deinit();
            oracleVarianceStrided(data, expected[0 .. outer * inner], outer, axis_dim, inner, ddof);
            try expectBitwiseF32(expected[0 .. outer * inner], got.dataConst());
        }

        const option_set = [_]exec.StandardizeOptions{
            .{},
            .{ .ddof = 1, .eps = 1e-5, .eps_mode = .outside_sqrt },
            .{ .ddof = 0, .eps = 1e-5, .eps_mode = .inside_sqrt },
            .{ .ddof = 1, .eps = 1e-3, .eps_mode = .inside_sqrt, .accumulation = .f64 },
            .{ .eps = 1e-5, .accumulation = .f64 },
        };
        for (option_set) |options| {
            for ([_]?usize{ null, axis_dim / 2, 1, 0 }) |valid_len| {
                const valid_count = valid_len orelse axis_dim;
                var got = if (valid_len) |prefix|
                    try ctx.standardizeValidPrefix(rank, &x, axis, prefix, options)
                else
                    try ctx.standardize(rank, &x, axis, options);
                defer got.deinit();
                switch (options.accumulation) {
                    .f32 => oracleStandardizeStrided(f32, data, expected, outer, axis_dim, inner, valid_count, options),
                    .f64 => oracleStandardizeStrided(f64, data, expected, outer, axis_dim, inner, valid_count, options),
                }
                try expectBitwiseF32(expected, got.dataConst());

                var got_bwd = try ctx.standardizeBackward(rank, &x, &gy, axis, valid_len, options);
                defer got_bwd.deinit();
                switch (options.accumulation) {
                    .f32 => oracleStandardizeBackwardStrided(f32, data, grad, expected, outer, axis_dim, inner, valid_count, options),
                    .f64 => oracleStandardizeBackwardStrided(f64, data, grad, expected, outer, axis_dim, inner, valid_count, options),
                }
                try expectBitwiseF32(expected, got_bwd.dataConst());
            }
        }
    }
}

test "stats non-last-axis lane kernels are bitwise the strided scalar arms" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // Below the row-kernel dispatch threshold (serial lane kernel, odd lane
    // counts for the vector tails) and above it (lane ranges split across
    // the pool, tails again).
    try expectStatsLaneKernelsMatchOracle(&ctx, 3, .{ 3, 5, 7 }, 0x51a7);
    try expectStatsLaneKernelsMatchOracle(&ctx, 4, .{ 2, 3, 4, 5 }, 0x51a8);
    try expectStatsLaneKernelsMatchOracle(&ctx, 3, .{ 2, 64, 1027 }, 0x51a9);
    try expectStatsLaneKernelsMatchOracle(&ctx, 4, .{ 2, 3, 32, 701 }, 0x51aa);
}
