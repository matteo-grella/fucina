//! Behavioral tests for the packed-RHS matmul module (`packed.zig`): f16/bf16
//! RHS packing to f32 and the single-row packed GEMV fast path that writes
//! output without falling back to the f32 temp matmul.
const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");
const packed_mod = @import("packed.zig");

const Tensor = tensor.Tensor;

const packRhs = packed_mod.packRhs;
const packDenseRhs = packed_mod.packDenseRhs;
const matmul2DIntoUncheckedPackedRhsTypedWithConfig = packed_mod.matmul2DIntoUncheckedPackedRhsTypedWithConfig;

test "pack f16 RHS as f32" {
    const allocator = std.testing.allocator;
    const values = [_]f16{ 1, 2, 3, 4, 5, 6 };
    var rhs = try tensor.TensorOf(.f16).fromSlice(allocator, &.{ 3, 2 }, &values);
    defer rhs.deinit();

    var packed_rhs = try packRhs(allocator, .f16, &rhs);
    defer packed_rhs.deinit();

    try std.testing.expectEqual(@as(usize, 3), packed_rhs.k);
    try std.testing.expectEqual(@as(usize, 2), packed_rhs.n);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, packed_rhs.rhs.dataConst());
}

test "pack bf16 RHS as f32" {
    const allocator = std.testing.allocator;
    const values = [_]u16{
        dtype_mod.f32ToBf16(1), dtype_mod.f32ToBf16(2), dtype_mod.f32ToBf16(3),
        dtype_mod.f32ToBf16(4), dtype_mod.f32ToBf16(5), dtype_mod.f32ToBf16(6),
    };
    var rhs = try tensor.TensorOf(.bf16).fromSlice(allocator, &.{ 3, 2 }, &values);
    defer rhs.deinit();

    var packed_rhs = try packRhs(allocator, .bf16, &rhs);
    defer packed_rhs.deinit();

    try std.testing.expectEqual(@as(usize, 3), packed_rhs.k);
    try std.testing.expectEqual(@as(usize, 2), packed_rhs.n);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, packed_rhs.rhs.dataConst());
}

test "dense pack snapshots f32 rows and zero-pads the output panel" {
    const allocator = std.testing.allocator;
    const values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var rhs = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &values);
    defer rhs.deinit();

    var packed_rhs = try packDenseRhs(allocator, .f32, &rhs);
    defer packed_rhs.deinit();
    try std.testing.expectEqual(@as(usize, 3), packed_rhs.n);
    try std.testing.expectEqual(@as(usize, 2), packed_rhs.k);
    try std.testing.expectEqual(@as(usize, 4), packed_rhs.padded_n);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 0, 0 }, packed_rhs.rhs.dataConst());

    rhs.data()[0] = 99;
    try std.testing.expectEqual(@as(f32, 1), packed_rhs.rhs.dataConst()[0]);
}

test "dense pack widens f16 and bf16 rows" {
    const allocator = std.testing.allocator;
    var f16_rhs = try tensor.TensorOf(.f16).fromSlice(allocator, &.{ 1, 3 }, &[_]f16{ 1, -2, 3.5 });
    defer f16_rhs.deinit();
    var f16_packed = try packDenseRhs(allocator, .f16, &f16_rhs);
    defer f16_packed.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, -2, 3.5 }, f16_packed.rhs.dataConst()[0..3]);

    var bf16_rhs = try tensor.TensorOf(.bf16).fromSlice(allocator, &.{ 1, 3 }, &.{
        dtype_mod.f32ToBf16(1),
        dtype_mod.f32ToBf16(-2),
        dtype_mod.f32ToBf16(3.5),
    });
    defer bf16_rhs.deinit();
    var bf16_packed = try packDenseRhs(allocator, .bf16, &bf16_rhs);
    defer bf16_packed.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, -2, 3.5 }, bf16_packed.rhs.dataConst()[0..3]);
}

test "packed single-row f16 matmul writes output without f32 temp fallback" {
    const allocator = std.testing.allocator;
    var lhs = try tensor.TensorOf(.f16).fromSlice(allocator, &.{ 1, 3 }, &[_]f16{ 1, 2, 3 });
    defer lhs.deinit();
    var rhs_tensor = try tensor.TensorOf(.f16).fromSlice(allocator, &.{ 3, 2 }, &[_]f16{ 7, 8, 9, 10, 11, 12 });
    defer rhs_tensor.deinit();
    var rhs = try packRhs(allocator, .f16, &rhs_tensor);
    defer rhs.deinit();
    var out = try tensor.TensorOf(.f16).zeros(allocator, &.{ 1, 2 });
    defer out.deinit();

    const Fallback = struct {
        fn run(_: *Tensor, _: *const Tensor, _: *const Tensor, _: usize, _: usize, _: usize, _: anytype) void {
            unreachable;
        }
    }.run;

    try matmul2DIntoUncheckedPackedRhsTypedWithConfig(allocator, .f16, &out, &lhs, &rhs, 1, 2, 3, .{ .pool = null }, Fallback);
    try std.testing.expectEqualSlices(f16, &[_]f16{ 58, 64 }, out.dataConst());
}

test "packed single-row bf16 matmul writes output without f32 temp fallback" {
    const allocator = std.testing.allocator;
    var lhs = try tensor.TensorOf(.bf16).fromSlice(allocator, &.{ 1, 3 }, &.{
        dtype_mod.f32ToBf16(1),
        dtype_mod.f32ToBf16(2),
        dtype_mod.f32ToBf16(3),
    });
    defer lhs.deinit();
    var rhs_tensor = try tensor.TensorOf(.bf16).fromSlice(allocator, &.{ 3, 2 }, &.{
        dtype_mod.f32ToBf16(7),
        dtype_mod.f32ToBf16(8),
        dtype_mod.f32ToBf16(9),
        dtype_mod.f32ToBf16(10),
        dtype_mod.f32ToBf16(11),
        dtype_mod.f32ToBf16(12),
    });
    defer rhs_tensor.deinit();
    var rhs = try packRhs(allocator, .bf16, &rhs_tensor);
    defer rhs.deinit();
    var out = try tensor.TensorOf(.bf16).zeros(allocator, &.{ 1, 2 });
    defer out.deinit();

    const Fallback = struct {
        fn run(_: *Tensor, _: *const Tensor, _: *const Tensor, _: usize, _: usize, _: usize, _: anytype) void {
            unreachable;
        }
    }.run;

    try matmul2DIntoUncheckedPackedRhsTypedWithConfig(allocator, .bf16, &out, &lhs, &rhs, 1, 2, 3, .{ .pool = null }, Fallback);
    try std.testing.expectEqualSlices(u16, &.{ dtype_mod.f32ToBf16(58), dtype_mod.f32ToBf16(64) }, out.dataConst());
}
