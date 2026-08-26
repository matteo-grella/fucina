//! Reduction kernels through ExecContext: compile-time axis and contiguous
//! sum paths, segmentSum with broadcast expansion, and cumsum in both
//! directions. Force-imported by `exec.zig`.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const exec = @import("../exec.zig");
const exec_elementwise = @import("elementwise.zig");
const exec_row_ops = @import("row_ops.zig");
const exec_moe_chain = @import("moe_chain.zig");
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
