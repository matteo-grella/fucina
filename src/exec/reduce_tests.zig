//! Reduction kernels through ExecContext: compile-time axis and contiguous
//! sum paths, segmentSum with broadcast expansion, and cumsum in both
//! directions. Force-imported by `exec.zig`.

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

test "exec context reduces a compile-time axis" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, .{ 2, 2, 3 }, &.{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    });
    defer x.deinit();

    var rows = try ctx.sumAxis(.f32, 3, &x, 1);
    defer rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, rows.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9, 17, 19, 21 }, rows.dataConst());

    var vector = try ctx.fromSlice(.f32, .{3}, &.{ 2, 4, 6 });
    defer vector.deinit();

    var all = try ctx.sumAxis(.f32, 1, &vector, 0);
    defer all.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, all.shape.slice());
    try std.testing.expectEqual(@as(f32, 12), all.item());
}

test "exec context uses contiguous reduction paths for rank-one and last-axis sums" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var v = try ctx.fromSlice(.f32, .{4}, &.{ 1, 2, 3, 4 });
    defer v.deinit();
    var total = try ctx.sumAxis(.f32, 1, &v, 0);
    defer total.deinit();
    try std.testing.expectEqualSlices(f32, &.{10}, total.dataConst());

    var x = try ctx.fromSlice(.f32, .{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();
    var rows = try ctx.sumAxis(.f32, 3, &x, 2);
    defer rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, rows.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 6, 15, 24, 33 }, rows.dataConst());
}

test "exec segmentSum reduces contiguous ranges and its broadcast expands them back" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, .{ 5, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
    defer x.deinit();

    // segments [0,2), [2,2) empty, [2,5): rows {4,6}, {0,0}, {21,24}.
    const offsets = [_]usize{ 0, 2, 2, 5 };
    var summed = try ctx.segmentSum(2, &x, 0, &offsets);
    defer summed.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 4, 6, 0, 0, 21, 24 }, summed.dataConst());

    var expanded = try ctx.segmentBroadcast(2, &summed, 0, &offsets, 5);
    defer expanded.deinit();
    try std.testing.expectEqualSlices(
        f32,
        &.{ 4, 6, 4, 6, 21, 24, 21, 24, 21, 24 },
        expanded.dataConst(),
    );

    // Along the last axis: segments [0,1), [1,2) of a (2, 2) tensor are the
    // identity partition.
    var y = try ctx.fromSlice(.f32, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer y.deinit();
    const unit = [_]usize{ 0, 1, 2 };
    var same = try ctx.segmentSum(2, &y, 1, &unit);
    defer same.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, same.dataConst());

    const bad = [_]usize{ 0, 3 };
    try std.testing.expectError(
        tensor.TensorError.InvalidShape,
        ctx.segmentSum(2, &x, 0, &bad),
    );
}

test "exec cumsum forward and reverse match torch.cumsum along both axes" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    // torch.cumsum(x, dim=1): rows {1,3,6} and {4,9,15}.
    var last = try ctx.cumsum(.f32, 2, &x, 1);
    defer last.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 6, 4, 9, 15 }, last.dataConst());

    // torch.cumsum(x, dim=0): {1,2,3} then {5,7,9} (inner > 1 layout).
    var first = try ctx.cumsum(.f32, 2, &x, 0);
    defer first.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 5, 7, 9 }, first.dataConst());

    // Reverse (suffix) sums — the cumsum VJP: torch.cumsum(x.flip(1), 1).flip(1).
    var rev = try ctx.cumsumReverse(.f32, 2, &x, 1);
    defer rev.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 5, 3, 15, 11, 6 }, rev.dataConst());
    var rev0 = try ctx.cumsumReverse(.f32, 2, &x, 0);
    defer rev0.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9, 4, 5, 6 }, rev0.dataConst());
}

const Fold = enum { sum, prod };

/// Naive reference for a non-last-axis fold: every output element meets its
/// axis values in index order, one at a time, with the entry's per-step
/// dtype policy. This is the accumulation order the streaming arm promises.
fn naiveFoldAxis(
    comptime dtype: tensor.DType,
    comptime fold: Fold,
    input: []const dtype_mod.Scalar(dtype),
    output: anytype,
    outer: usize,
    axis_dim: usize,
    inner: usize,
) void {
    const compute = comptime dtype_mod.computeDType(.reduction, dtype);
    const out_dtype = comptime dtype_mod.outputDType(.reduction, dtype);
    for (0..outer) |o| {
        for (0..inner) |i| {
            var acc: dtype_mod.Scalar(out_dtype) = if (fold == .prod) 1 else 0;
            for (0..axis_dim) |a| {
                const value = input[(o * axis_dim + a) * inner + i];
                switch (comptime fold) {
                    .sum => if (comptime dtype == .f32) {
                        acc += value;
                    } else {
                        acc = dtype_mod.castFloat(compute, out_dtype, dtype_mod.castFloat(out_dtype, compute, acc) + dtype_mod.castFloat(dtype, compute, value));
                    },
                    .prod => acc *= value,
                }
            }
            output[o * inner + i] = acc;
        }
    }
}

fn checkFoldAxis(
    ctx: *ExecContext,
    comptime dtype: tensor.DType,
    comptime rank: usize,
    shape: [rank]usize,
    comptime axis: usize,
    comptime fold: Fold,
) !void {
    const allocator = std.testing.allocator;
    var x = try ctx.zeros(dtype, shape);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(0x5eed + axis + rank);
    const random = prng.random();
    for (x.data()) |*v| {
        const f = random.float(f32) * 2 - 1;
        v.* = switch (comptime dtype) {
            .f32 => f,
            .f16 => @floatCast(f),
            .bf16 => dtype_mod.f32ToBf16(f),
            else => comptime unreachable,
        };
    }
    var got = if (comptime fold == .sum) try ctx.sumAxis(dtype, rank, &x, axis) else try ctx.prod(dtype, rank, &x, axis);
    defer got.deinit();

    var outer: usize = 1;
    var inner: usize = 1;
    inline for (0..rank) |d| {
        if (d < axis) outer *= shape[d];
        if (d > axis) inner *= shape[d];
    }
    const Out = @TypeOf(got.dataConst()[0]);
    const want = try allocator.alloc(Out, outer * inner);
    defer allocator.free(want);
    naiveFoldAxis(dtype, fold, x.dataConst(), want, outer, shape[axis], inner);
    try std.testing.expectEqualSlices(Out, want, got.dataConst());
}

test "exec non-last-axis reductions stream (outer, axis, inner) bitwise like the index-order reference" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // Above the row-kernel work threshold (131072 elements) so the pool
    // dispatch runs: [4, 64, 512] splits by outer block, [1, 256, 512] by
    // inner lane, and the rank-4 shape reaches every non-last axis.
    try checkFoldAxis(&ctx, .f32, 3, .{ 4, 64, 512 }, 0, .sum);
    try checkFoldAxis(&ctx, .f32, 3, .{ 4, 64, 512 }, 1, .sum);
    try checkFoldAxis(&ctx, .f32, 3, .{ 1, 256, 512 }, 1, .sum);
    try checkFoldAxis(&ctx, .f32, 4, .{ 4, 8, 8, 512 }, 0, .sum);
    try checkFoldAxis(&ctx, .f32, 4, .{ 4, 8, 8, 512 }, 1, .sum);
    try checkFoldAxis(&ctx, .f32, 4, .{ 4, 8, 8, 512 }, 2, .sum);
    try checkFoldAxis(&ctx, .f32, 3, .{ 4, 64, 512 }, 1, .prod);
    try checkFoldAxis(&ctx, .f32, 4, .{ 4, 8, 8, 512 }, 1, .prod);

    // Below the threshold: the serial task, same order.
    try checkFoldAxis(&ctx, .f32, 3, .{ 3, 5, 7 }, 0, .sum);
    try checkFoldAxis(&ctx, .f32, 3, .{ 3, 5, 7 }, 1, .sum);
    try checkFoldAxis(&ctx, .f32, 4, .{ 2, 3, 5, 7 }, 2, .prod);

    // The widened 16-bit entries keep their per-step cast policy.
    try checkFoldAxis(&ctx, .f16, 3, .{ 4, 64, 512 }, 1, .sum);
    try checkFoldAxis(&ctx, .f16, 4, .{ 4, 8, 8, 512 }, 0, .sum);
    try checkFoldAxis(&ctx, .bf16, 3, .{ 4, 64, 512 }, 0, .sum);
    try checkFoldAxis(&ctx, .bf16, 4, .{ 4, 8, 8, 512 }, 2, .sum);
    try checkFoldAxis(&ctx, .f16, 3, .{ 3, 5, 7 }, 1, .sum);
    try checkFoldAxis(&ctx, .bf16, 3, .{ 3, 5, 7 }, 0, .sum);
}

test "exec non-last-axis integer sums stream like the index-order reference" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var x = try ctx.zeros(.i32, .{ 4, 64, 512 });
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();
    for (x.data()) |*v| v.* = random.intRangeAtMost(i32, -1000, 1000);
    var got = try ctx.sumAxis(.i32, 3, &x, 1);
    defer got.deinit();

    const want = try std.testing.allocator.alloc(i64, 4 * 512);
    defer std.testing.allocator.free(want);
    for (0..4) |o| for (0..512) |i| {
        var acc: i64 = 0;
        for (0..64) |a| acc +%= x.dataConst()[(o * 64 + a) * 512 + i];
        want[o * 512 + i] = acc;
    };
    try std.testing.expectEqualSlices(i64, want, got.dataConst());
}
