//! Behavioral tests for the quantized-matmul module (`quant.zig`): the W8A8
//! container's group-size policy and storage shape, the block byte sizes
//! and block lengths `dtype.block_formats` registers per format, Q8_K x4
//! packed-vs-direct quantization parity, the exact i8 block-wise
//! quantize-and-matmul path when the scale is 1, and the TQ2_0
//! borrowed-blocks RHS view (no copy, no-op deinit).
const std = @import("std");
const quant = @import("quant.zig");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");

const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

const QuantizedMatmulRhsI8 = quant.types.QuantizedMatmulRhsI8;
const qk_k_block_size = dtype_mod.qk_k_block_size;
const BlockQ8_Kx4 = quant.types.BlockQ8_Kx4;
const quantizeRowsQ8_K = quant.q8k.quantizeRowsQ8_K;
const packRowsQ8_Kx4 = quant.q8k.packRowsQ8_Kx4;
const quantizeRowsQ8_Kx4Into = quant.q8k.quantizeRowsQ8_Kx4Into;
const quantizeRhsBlockwiseI8 = quant.quantizeRhsBlockwiseI8;
const quantizeActivationsPerRowI8 = quant.quantizeActivationsPerRowI8;
const matmulI8BlockwiseRange = quant.matmulI8BlockwiseRange;

test "i8 block-wise quantized matmul RHS resolves group size and storage shape" {
    try std.testing.expectEqual(@as(usize, 32), QuantizedMatmulRhsI8.default_group_size);
    try std.testing.expectEqual(@as(usize, 32), QuantizedMatmulRhsI8.effectiveGroupSize(0));
    try std.testing.expectEqual(@as(usize, 16), QuantizedMatmulRhsI8.effectiveGroupSize(16));
    try std.testing.expectEqual(@as(usize, 3), QuantizedMatmulRhsI8.groupCountForSize(65, 32));

    // Storage is [n][k] i8 and scales are [n][num_groups] f32: column 2's
    // k-vector starts at 2 * k and its group-1 scale sits at 2 * 3 + 1.
    const allocator = std.testing.allocator;
    const values = [_]f32{1} ** (65 * 7);
    var rhs = try Tensor.fromSlice(allocator, &.{ 65, 7 }, &values);
    defer rhs.deinit();
    var qrhs = try quant.quantizeRhsBlockwiseI8(allocator, &rhs, 0);
    defer qrhs.deinit();
    try std.testing.expectEqual(@as(usize, 32), qrhs.group_size);
    try std.testing.expectEqual(@as(usize, 3), qrhs.num_groups);
    try std.testing.expectEqualSlices(usize, &.{ 7, 65 }, qrhs.qw.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 7, 3 }, qrhs.scales.shape.slice());
}

test "block formats register GGML block lengths and byte sizes" {
    const cases = [_]struct { dtype: DType, block_size: usize, byte_size: usize }{
        .{ .dtype = .q8_0, .block_size = 32, .byte_size = 34 },
        .{ .dtype = .q4_0, .block_size = 32, .byte_size = 18 },
        .{ .dtype = .q2_k, .block_size = 256, .byte_size = 84 },
        .{ .dtype = .q3_k, .block_size = 256, .byte_size = 110 },
        .{ .dtype = .q4_k, .block_size = 256, .byte_size = 144 },
        .{ .dtype = .q5_k, .block_size = 256, .byte_size = 176 },
        .{ .dtype = .q6_k, .block_size = 256, .byte_size = 210 },
        .{ .dtype = .q8_k, .block_size = 256, .byte_size = 292 },
        .{ .dtype = .iq1_s, .block_size = 256, .byte_size = 50 },
        .{ .dtype = .iq1_m, .block_size = 256, .byte_size = 56 },
        .{ .dtype = .iq2_xxs, .block_size = 256, .byte_size = 66 },
        .{ .dtype = .iq2_xs, .block_size = 256, .byte_size = 74 },
        .{ .dtype = .iq2_s, .block_size = 256, .byte_size = 82 },
        .{ .dtype = .iq3_xxs, .block_size = 256, .byte_size = 98 },
        .{ .dtype = .iq3_s, .block_size = 256, .byte_size = 110 },
        .{ .dtype = .iq4_nl, .block_size = 32, .byte_size = 18 },
        .{ .dtype = .iq4_xs, .block_size = 256, .byte_size = 136 },
        .{ .dtype = .tq1_0, .block_size = 256, .byte_size = 54 },
        .{ .dtype = .tq2_0, .block_size = 256, .byte_size = 66 },
        .{ .dtype = .mxfp4, .block_size = 32, .byte_size = 17 },
        .{ .dtype = .nvfp4, .block_size = 64, .byte_size = 36 },
    };
    inline for (cases) |case| {
        try std.testing.expectEqual(case.block_size, dtype_mod.blockSize(case.dtype));
        try std.testing.expectEqual(case.byte_size, dtype_mod.blockByteSize(case.dtype));
    }
    // q8_1 and q8_k are registered block formats without a matmul RHS kernel.
    try std.testing.expect(!dtype_mod.supportsQuantizedMatmulRhs(.q8_1));
    try std.testing.expect(!dtype_mod.supportsQuantizedMatmulRhs(.q8_k));
    inline for (cases) |case| {
        if (case.dtype != .q8_k) try std.testing.expect(dtype_mod.supportsQuantizedMatmulRhs(case.dtype));
    }
}

test "ggml_q8_k x4 direct quantization matches packed rows" {
    const allocator = std.testing.allocator;

    var values: [4 * qk_k_block_size]f32 = undefined;
    for (&values, 0..) |*v, i| {
        const row = i / qk_k_block_size;
        const col = i % qk_k_block_size;
        const value: i32 = if (row == 2)
            0
        else
            @as(i32, @intCast((col * 17 + row * 5) % 251)) - 125;
        v.* = @floatFromInt(value);
    }

    var dense = try Tensor.fromSlice(allocator, &.{ 4, qk_k_block_size }, &values);
    defer dense.deinit();

    const qrows = try quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qrows);

    const packed_rows = try packRowsQ8_Kx4(allocator, qrows, 4, qk_k_block_size, 1);
    defer allocator.free(packed_rows);

    var direct: [1]BlockQ8_Kx4 = undefined;
    try quantizeRowsQ8_Kx4Into(&direct, &dense);

    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(packed_rows), std.mem.sliceAsBytes(direct[0..]));
}

test "quantize and matmul i8 block-wise is exact when scale is 1" {
    const allocator = std.testing.allocator;

    // RHS [k=4, n=2]; each column's group has amax 127 so the scale is exactly 1.
    const w_vals = [_]f32{ 127, 3, 0, 4, 1, 127, 2, 5 };
    var rhs = try Tensor.fromSlice(allocator, &.{ 4, 2 }, &w_vals);
    defer rhs.deinit();

    var qrhs = try quantizeRhsBlockwiseI8(allocator, &rhs, 32);
    defer qrhs.deinit();

    try std.testing.expectEqual(@as(usize, 4), qrhs.k);
    try std.testing.expectEqual(@as(usize, 2), qrhs.n);
    try std.testing.expectEqual(@as(usize, 1), qrhs.num_groups);
    // Stored transposed [n][k]: column 0 then column 1.
    try std.testing.expectEqualSlices(i8, &.{ 127, 0, 1, 2, 3, 4, 127, 5 }, qrhs.qw.dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 1, 1 }, qrhs.scales.dataConst());

    // Activations [m=2, k=4]; each row has amax 127 so the row scale is exactly 1.
    const a_vals = [_]f32{ 127, 1, 2, 3, 4, 127, 5, 6 };
    var qa: [8]i8 = undefined;
    var a_scales: [2]f32 = undefined;
    quantizeActivationsPerRowI8(&qa, &a_scales, &a_vals, 2, 4);
    try std.testing.expectEqualSlices(i8, &.{ 127, 1, 2, 3, 4, 127, 5, 6 }, &qa);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1 }, &a_scales);

    var out: [4]f32 = undefined;
    matmulI8BlockwiseRange(&out, &qa, &a_scales, qrhs.qw.dataConst(), qrhs.scales.dataConst(), 2, 2, 4, qrhs.group_size, qrhs.num_groups, 0, 2);
    try std.testing.expectEqualSlices(f32, &.{ 16137, 654, 525, 1185 }, &out);
}

test "tq2_0 borrowed-blocks RHS: no copy, matmul parity with the owning constructor, no-op deinit" {
    const allocator = std.testing.allocator;
    const k = qk_k_block_size;
    const n = 3;

    var weights: [n * k]f32 = undefined;
    for (&weights, 0..) |*w, i| w.* = @floatFromInt(@as(i32, @intCast(i % 5)) - 2);
    var owned = try quant.ternary.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, &weights);
    defer owned.deinit();

    // Borrow the owning container's blocks: same storage, no dupe; deinit
    // frees nothing (allocator = null), so both containers may deinit.
    var borrowed = try quant.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, owned.blocks);
    defer borrowed.deinit();
    try std.testing.expectEqual(owned.blocks.ptr, borrowed.blocks.ptr);
    try std.testing.expectEqual(@as(usize, 1), borrowed.blocks_per_column);

    // One matmul through each container agrees bitwise.
    var lhs: [2 * k]f32 = undefined;
    for (&lhs, 0..) |*x, i| x.* = @floatFromInt(@as(i32, @intCast(i % 7)) - 3);
    var out_owned: [2 * n]f32 = undefined;
    var out_borrowed: [2 * n]f32 = undefined;
    quant.ternary.matmulTQ2_0F32RhsRange(&out_owned, &lhs, &owned, 2, n, 0, 2);
    quant.ternary.matmulTQ2_0F32RhsRange(&out_borrowed, &lhs, &borrowed, 2, n, 0, 2);
    try std.testing.expectEqualSlices(f32, &out_owned, &out_borrowed);

    // Length validation still applies to borrowed views.
    try std.testing.expectError(
        quant.types.QuantizedFormatError.InvalidQuantizedLength,
        quant.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n + 1, owned.blocks),
    );
}
