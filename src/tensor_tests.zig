//! Behavioral tests for the raw internal tensor (`tensor.zig`): independent
//! shape/data ownership, borrowed-slice aliasing, dtype-preserving views and
//! copies, fixed-rank views, reshape/stride views, broadcasting, and clone
//! materialization.
const std = @import("std");
const tensor = @import("tensor.zig");

const Tensor = tensor.Tensor;
const TensorOf = tensor.TensorOf;
const TensorError = tensor.TensorError;

test "tensor owns shape and data independently" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    try std.testing.expectEqual(@as(usize, 4), x.len());
    try std.testing.expectEqual(@as(f32, 3), x.dataConst()[2]);
}

test "tensor borrowed slice aliases caller-owned data" {
    const allocator = std.testing.allocator;
    var values = [_]f32{ 1, 2, 3, 4 };

    var x = try Tensor.fromBorrowedSlice(allocator, &.{ 2, 2 }, values[0..]);
    defer x.deinit();

    values[2] = 30;
    try std.testing.expectEqual(@as(f32, 30), x.dataConst()[2]);
    x.data()[1] = 20;
    try std.testing.expectEqual(@as(f32, 20), values[1]);
}

test "typed tensors preserve dtype through views and copies" {
    const allocator = std.testing.allocator;
    var ids = try TensorOf(.u16).fromSlice(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ids.deinit();

    try std.testing.expect(@TypeOf(ids).dtype == .u16);
    try std.testing.expect(@TypeOf(ids.data()[0]) == u16);

    var row = try ids.viewWithStridesOffset(&.{3}, &.{1}, 3);
    defer row.deinit();

    var copied = [_]u16{0} ** 3;
    try row.copyTo(&copied);
    try std.testing.expectEqualSlices(u16, &.{ 4, 5, 6 }, &copied);

    var mask = try TensorOf(.bool).ones(allocator, &.{2});
    defer mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true }, mask.dataConst());
}

test "rankView exposes fixed-rank shape and stride metadata" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    const view = try x.rankView(2);
    try std.testing.expectEqual(@as(usize, 2), view.dim(0));
    try std.testing.expectEqual(@as(usize, 3), view.dim(1));
    try std.testing.expectEqual(@as(usize, 6), view.len());
    try std.testing.expect(view.isContiguous());
    try std.testing.expectError(TensorError.InvalidShape, x.rankView(3));
}

test "reshape is a retained view over storage" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    var y = try x.reshape(&.{4});
    defer y.deinit();

    y.data()[1] = 9;
    try std.testing.expectEqual(@as(f32, 9), x.dataConst()[1]);
}

test "viewWithStrides creates retained checked views" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var y = try x.viewWithStrides(&.{ 3, 2 }, &.{ 1, 3 });
    defer y.deinit();

    try std.testing.expect(y.buffer == x.buffer);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, y.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, y.strides.slice());

    var copied = [_]f32{0} ** 6;
    try y.copyTo(&copied);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, &copied);

    try std.testing.expectError(TensorError.InvalidShape, x.viewWithStrides(&.{ 2, 3 }, &.{3}));
    try std.testing.expectError(TensorError.InvalidDataLength, x.viewWithStrides(&.{ 3, 3 }, &.{ 1, 3 }));
}

test "viewWithStridesOffset creates retained checked subviews" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{ 3, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer x.deinit();

    var y = try x.viewWithStridesOffset(&.{ 2, 3 }, &.{ 3, 1 }, 3);
    defer y.deinit();

    try std.testing.expect(y.buffer == x.buffer);
    try std.testing.expectEqual(@as(usize, 3), y.offset);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, y.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6, 7, 8, 9 }, y.dataConst());

    try std.testing.expectError(TensorError.InvalidDataLength, x.viewWithStridesOffset(&.{ 2, 3 }, &.{ 3, 1 }, 6));
}

test "dataChecked rejects non-contiguous views" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var y = try x.viewWithStrides(&.{ 3, 2 }, &.{ 1, 3 });
    defer y.deinit();

    try std.testing.expectError(TensorError.UnsupportedView, y.dataChecked());
    try std.testing.expectError(TensorError.UnsupportedView, y.dataConstChecked());
}

test "broadcastTo creates a zero-stride view" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{3}, &.{ 1, 2, 3 });
    defer x.deinit();

    var y = try x.broadcastTo(&.{ 2, 3 });
    defer y.deinit();

    try std.testing.expect(y.buffer == x.buffer);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, y.strides.slice());

    var copied = [_]f32{0} ** 6;
    try y.copyTo(&copied);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 1, 2, 3 }, &copied);
}

test "broadcastTo supports scalar and singleton dimensions" {
    const allocator = std.testing.allocator;
    var scalar_value = try Tensor.scalar(allocator, 7);
    defer scalar_value.deinit();

    var scalar_b = try scalar_value.broadcastTo(&.{ 2, 3 });
    defer scalar_b.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 0 }, scalar_b.strides.slice());

    var scalar_copied = [_]f32{0} ** 6;
    try scalar_b.copyTo(&scalar_copied);
    try std.testing.expectEqualSlices(f32, &.{ 7, 7, 7, 7, 7, 7 }, &scalar_copied);

    var x = try Tensor.fromSlice(allocator, &.{ 2, 1, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var y = try x.broadcastTo(&.{ 2, 4, 3 });
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 0, 1 }, y.strides.slice());

    var copied = [_]f32{0} ** 24;
    try y.copyTo(&copied);
    try std.testing.expectEqualSlices(f32, &.{
        1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
        4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
    }, &copied);
}

test "broadcastToRank handles leading and singleton dimensions" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{ 1, 3 }, &.{ 1, 2, 3 });
    defer x.deinit();

    var y = try x.broadcastToRank(3, .{ 2, 4, 3 });
    defer y.deinit();

    try std.testing.expect(y.buffer == x.buffer);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 1 }, y.strides.slice());

    var copied = [_]f32{0} ** 24;
    try y.copyTo(&copied);
    try std.testing.expectEqualSlices(f32, &.{
        1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
        1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
    }, &copied);
}

test "broadcastToRank handles higher-rank targets and errors" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{2}, &.{ 4, 5 });
    defer x.deinit();

    var y = try x.broadcastToRank(5, .{ 2, 1, 2, 1, 2 });
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 0, 0, 1 }, y.strides.slice());

    var copied = [_]f32{0} ** 8;
    try y.copyTo(&copied);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 4, 5, 4, 5, 4, 5 }, &copied);

    try std.testing.expectError(TensorError.ShapeMismatch, y.broadcastToRank(1, .{2}));
    try std.testing.expectError(TensorError.InvalidShape, x.broadcastToRank(2, .{ 0, 2 }));
}

test "clone materializes non-contiguous views" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var y = try x.broadcastTo(&.{ 2, 3 });
    defer y.deinit();

    var z = try y.clone(allocator);
    defer z.deinit();

    try std.testing.expect(z.buffer != x.buffer);
    try std.testing.expect(z.isContiguous());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 1, 2, 3 }, z.dataConst());
}

test "broadcastTo rejects incompatible shapes" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{2}, &.{ 1, 2 });
    defer x.deinit();

    try std.testing.expectError(TensorError.ShapeMismatch, x.broadcastTo(&.{ 3, 4 }));
}

test "copyRangeTo matches full copy across run boundaries on a permuted view" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{ 3, 4 }, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });
    defer x.deinit();
    // [4, 3] transposed view: strided innermost axis (stride 4).
    var t = try x.viewWithStrides(&.{ 4, 3 }, &.{ 1, 4 });
    defer t.deinit();

    var full = [_]f32{0} ** 12;
    try t.copyTo(&full);
    try std.testing.expectEqualSlices(f32, &.{ 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11 }, &full);

    // Every (start, count) split must agree with the full linearization,
    // including splits that land mid-run.
    var start: usize = 0;
    while (start < 12) : (start += 1) {
        var count: usize = 1;
        while (start + count <= 12) : (count += 1) {
            var part = [_]f32{99} ** 12;
            t.copyRangeTo(part[0..count], start, count);
            try std.testing.expectEqualSlices(f32, full[start .. start + count], part[0..count]);
        }
    }
}

test "copyRangeTo handles broadcast, offset, and absorbed singleton views" {
    const allocator = std.testing.allocator;
    var x = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    // Zero-stride broadcast axis (dim > 1, stride 0).
    var b = try x.viewWithStrides(&.{ 2, 2, 3 }, &.{ 0, 3, 1 });
    defer b.deinit();
    var bd = [_]f32{0} ** 12;
    try b.copyTo(&bd);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 }, &bd);

    // Offset view (second row) with a leading stride-0 singleton: materially
    // contiguous even though isContiguous() is strictly false.
    var o = try x.viewWithStridesOffset(&.{ 1, 3 }, &.{ 0, 1 }, 3);
    defer o.deinit();
    try std.testing.expect(!o.isContiguous());
    var od = [_]f32{0} ** 3;
    try o.copyTo(&od);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6 }, &od);
}
