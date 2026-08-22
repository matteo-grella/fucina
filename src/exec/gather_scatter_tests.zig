//! Gather/scatter kernels through ExecContext: typed data movement, last-axis
//! concat and updates, row zeroing, scatter-add parallel parity, and pad
//! placement. Force-imported by `exec.zig`.

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
const LayoutClass = exec.LayoutClass;
const CrossEntropyOptions = exec.CrossEntropyOptions;
const Reduction = exec.Reduction;

/// Serial reference for `scatterAddAxisRank` with axis == 0: dense zeros plus
/// row accumulation in index order — the exact algorithm of the serial path,
/// so parity assertions against it are BITWISE.
fn scatterAddAxis0Reference(expected: []f32, grad: []const f32, row_len: usize, indices: []const usize) void {
    @memset(expected, 0);
    for (indices, 0..) |index, row| {
        for (expected[index * row_len ..][0..row_len], grad[row * row_len ..][0..row_len]) |*d, v| {
            d.* += v;
        }
    }
}

test "exec context optimizes last-axis concat and updates" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer b.deinit();

    var inputs = [_]*const Tensor{ &a, &b };
    var joined = try ctx.concatAxisRank(2, &inputs, 1);
    defer joined.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, joined.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 10, 20, 30, 3, 4, 40, 50, 60 }, joined.dataConst());

    var update = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 7, 8, 9, 10 });
    defer update.deinit();
    var sliced = try ctx.setSliceAxisRank(2, &joined, &update, 1, 1);
    defer sliced.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 7, 8, 20, 30, 3, 9, 10, 50, 60 }, sliced.dataConst());

    var row_update = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 100, 200, 300, 400 });
    defer row_update.deinit();
    var rows = try ctx.setRowsAxisRank(2, &joined, &row_update, 1, &.{ 4, 0 });
    defer rows.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 200, 2, 10, 20, 100, 400, 4, 40, 50, 300 }, rows.dataConst());
}

test "exec context zeros indexed rows on non-leading axes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(3, .{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    var middle = try ctx.zeroRowsAxisRank(3, &x, 1, &.{ 2, 0 });
    defer middle.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 3, 4, 0, 0, 0, 0, 9, 10, 0, 0 }, middle.dataConst());

    var last = try ctx.zeroRowsAxisRank(3, &x, 2, &.{1});
    defer last.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0 }, last.dataConst());
}

test "exec context runs typed data movement and indexing kernels" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var ids = try ctx.fromSliceRankTyped(.u16, 2, .{ 3, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer ids.deinit();

    var rows = try ctx.gatherAxisRankTyped(.u16, 2, &ids, 0, &.{ 2, 0 });
    defer rows.deinit();
    try std.testing.expectEqualSlices(u16, &.{ 7, 8, 9, 1, 2, 3 }, rows.dataConst());

    var narrowed = try ctx.narrowAxisRankTyped(.u16, 2, &ids, 1, 1, 2);
    defer narrowed.deinit();
    var narrowed_data: [6]u16 = undefined;
    try narrowed.copyTo(&narrowed_data);
    try std.testing.expectEqualSlices(u16, &.{ 2, 3, 5, 6, 8, 9 }, &narrowed_data);

    var extra = try ctx.fromSliceRankTyped(.u16, 2, .{ 3, 1 }, &.{ 10, 11, 12 });
    defer extra.deinit();
    var concat_inputs = [_]*const tensor.TensorOf(.u16){ &ids, &extra };
    var joined = try ctx.concatAxisRankTyped(.u16, 2, &concat_inputs, 1);
    defer joined.deinit();
    try std.testing.expectEqualSlices(u16, &.{ 1, 2, 3, 10, 4, 5, 6, 11, 7, 8, 9, 12 }, joined.dataConst());

    var update = try ctx.fromSliceRankTyped(.u16, 2, .{ 3, 2 }, &.{ 20, 21, 22, 23, 24, 25 });
    defer update.deinit();
    var sliced = try ctx.setSliceAxisRankTyped(.u16, 2, &joined, &update, 1, 1);
    defer sliced.deinit();
    try std.testing.expectEqualSlices(u16, &.{ 1, 20, 21, 10, 4, 22, 23, 11, 7, 24, 25, 12 }, sliced.dataConst());

    var row_update = try ctx.fromSliceRankTyped(.u16, 2, .{ 2, 4 }, &.{ 30, 31, 32, 33, 40, 41, 42, 43 });
    defer row_update.deinit();
    var replaced = try ctx.setRowsAxisRankTyped(.u16, 2, &joined, &row_update, 0, &.{ 2, 0 });
    defer replaced.deinit();
    try std.testing.expectEqualSlices(u16, &.{ 40, 41, 42, 43, 4, 5, 6, 11, 30, 31, 32, 33 }, replaced.dataConst());
}

test "scatter add axis0 parallel path matches serial reference bitwise" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // 4096x96 source + 1024-row grad = 491520 elements of work: above the
    // parallel threshold, so the pool path runs (duplicates included).
    const rows = 4096;
    const row_len = 96;
    const index_count = 1024;

    var prng = std.Random.DefaultPrng.init(0x5ca77e);
    const random = prng.random();
    const grad_data = try allocator.alloc(f32, index_count * row_len);
    defer allocator.free(grad_data);
    for (grad_data) |*value| value.* = random.floatNorm(f32);
    const indices = try allocator.alloc(usize, index_count);
    defer allocator.free(indices);
    // Heavy duplicates: half the indices land in a 13-row band.
    for (indices, 0..) |*index, i| {
        index.* = if (i % 2 == 0) random.uintLessThan(usize, rows) else 100 + random.uintLessThan(usize, 13);
    }

    var grad = try ctx.fromSliceRank(2, .{ index_count, row_len }, grad_data);
    defer grad.deinit();

    const expected = try allocator.alloc(f32, rows * row_len);
    defer allocator.free(expected);
    scatterAddAxis0Reference(expected, grad_data, row_len, indices);

    var out = try ctx.scatterAddAxisRank(2, &grad, .{ rows, row_len }, 0, indices);
    defer out.deinit();
    try std.testing.expectEqualSlices(f32, expected, out.dataConst());

    // Determinism: a second run is bitwise identical.
    var out2 = try ctx.scatterAddAxisRank(2, &grad, .{ rows, row_len }, 0, indices);
    defer out2.deinit();
    try std.testing.expectEqualSlices(f32, out.dataConst(), out2.dataConst());
}

test "scatter add axis0 single repeated index and more indices than rows" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0xd0011ca7e);
    const random = prng.random();

    // Every index hits the same destination row (worst-case duplicate skew on
    // the parallel path: one task accumulates everything, the rest only zero).
    {
        const rows = 2048;
        const row_len = 192;
        const index_count = 2048;
        const grad_data = try allocator.alloc(f32, index_count * row_len);
        defer allocator.free(grad_data);
        for (grad_data) |*value| value.* = random.floatNorm(f32);
        const indices = try allocator.alloc(usize, index_count);
        defer allocator.free(indices);
        @memset(indices, 7);

        var grad = try ctx.fromSliceRank(2, .{ index_count, row_len }, grad_data);
        defer grad.deinit();
        const expected = try allocator.alloc(f32, rows * row_len);
        defer allocator.free(expected);
        scatterAddAxis0Reference(expected, grad_data, row_len, indices);

        var out = try ctx.scatterAddAxisRank(2, &grad, .{ rows, row_len }, 0, indices);
        defer out.deinit();
        try std.testing.expectEqualSlices(f32, expected, out.dataConst());
    }

    // indices.len far above the source row count: every row is hit many times.
    {
        const rows = 8;
        const row_len = 64;
        const index_count = 4096;
        const grad_data = try allocator.alloc(f32, index_count * row_len);
        defer allocator.free(grad_data);
        for (grad_data) |*value| value.* = random.floatNorm(f32);
        const indices = try allocator.alloc(usize, index_count);
        defer allocator.free(indices);
        for (indices) |*index| index.* = random.uintLessThan(usize, rows);

        var grad = try ctx.fromSliceRank(2, .{ index_count, row_len }, grad_data);
        defer grad.deinit();
        const expected = try allocator.alloc(f32, rows * row_len);
        defer allocator.free(expected);
        scatterAddAxis0Reference(expected, grad_data, row_len, indices);

        var out = try ctx.scatterAddAxisRank(2, &grad, .{ rows, row_len }, 0, indices);
        defer out.deinit();
        try std.testing.expectEqualSlices(f32, expected, out.dataConst());
    }
}

test "scatter add axis0 parallel threshold boundary is bitwise seamless" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // row_len == 1 makes total work = rows + indices.len exactly, so the three
    // index counts straddle the threshold: -1 stays serial, 0/+1 go parallel.
    const rows = 131072;
    const base_index_count = parallel.vector_elementwise_len_threshold - rows;

    var prng = std.Random.DefaultPrng.init(0xb0a2d);
    const random = prng.random();

    for ([_]usize{ base_index_count - 1, base_index_count, base_index_count + 1 }) |index_count| {
        const grad_data = try allocator.alloc(f32, index_count);
        defer allocator.free(grad_data);
        for (grad_data) |*value| value.* = random.floatNorm(f32);
        const indices = try allocator.alloc(usize, index_count);
        defer allocator.free(indices);
        for (indices) |*index| index.* = random.uintLessThan(usize, rows);

        var grad = try ctx.fromSliceRank(2, .{ index_count, 1 }, grad_data);
        defer grad.deinit();
        const expected = try allocator.alloc(f32, rows);
        defer allocator.free(expected);
        scatterAddAxis0Reference(expected, grad_data, 1, indices);

        var out = try ctx.scatterAddAxisRank(2, &grad, .{ rows, 1 }, 0, indices);
        defer out.deinit();
        try std.testing.expectEqualSlices(f32, expected, out.dataConst());
    }
}

test "scatter add rank3 axis0 parallel and axis1 generic paths" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0xa715);
    const random = prng.random();

    // Rank 3, axis 0, above threshold: rows are (32*256)-element planes.
    {
        const rows = 64;
        const row_len = 32 * 256;
        const index_count = 96;
        const grad_data = try allocator.alloc(f32, index_count * row_len);
        defer allocator.free(grad_data);
        for (grad_data) |*value| value.* = random.floatNorm(f32);
        const indices = try allocator.alloc(usize, index_count);
        defer allocator.free(indices);
        for (indices) |*index| index.* = random.uintLessThan(usize, rows);

        var grad = try ctx.fromSliceRank(3, .{ index_count, 32, 256 }, grad_data);
        defer grad.deinit();
        const expected = try allocator.alloc(f32, rows * row_len);
        defer allocator.free(expected);
        scatterAddAxis0Reference(expected, grad_data, row_len, indices);

        var out = try ctx.scatterAddAxisRank(3, &grad, .{ rows, 32, 256 }, 0, indices);
        defer out.deinit();
        try std.testing.expectEqualSlices(f32, expected, out.dataConst());
    }

    // axis != 0 keeps the generic strided path (small reference check).
    {
        var grad = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 10, 20, 30 });
        defer grad.deinit();
        var out = try ctx.scatterAddAxisRank(2, &grad, .{ 2, 2 }, 1, &.{ 1, 0, 1 });
        defer out.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 2, 4, 20, 40 }, out.dataConst());
    }
}

test "exec pad places the body at offset before and fills the rest" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    // torch F.pad(x, (1, 2), value=9): last axis grows 2 -> 5.
    var last = try ctx.padAxisRank(2, &x, 1, 1, 2, 9);
    defer last.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, last.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 9, 1, 2, 9, 9, 9, 3, 4, 9, 9 }, last.dataConst());

    // torch F.pad(x, (0, 0, 2, 1), value=0): first axis 2 -> 5.
    var first = try ctx.padAxisRank(2, &x, 0, 2, 1, 0);
    defer first.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 5, 2 }, first.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 1, 2, 3, 4, 0, 0 }, first.dataConst());

    // before == after == 0 is an identity copy.
    var same = try ctx.padAxisRank(2, &x, 1, 0, 0, 7);
    defer same.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, same.dataConst());
}
