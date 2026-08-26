//! Behavioral tests for the packed-RHS matmul module (`packed.zig`): the
//! dense f32 output-row panel and its exact f16/bf16 widening.
const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");
const packed_mod = @import("packed.zig");

const Tensor = tensor.Tensor;

const packDenseRhs = packed_mod.packDenseRhs;

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
