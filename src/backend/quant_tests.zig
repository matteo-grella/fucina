//! Behavioral tests for the quantized-matmul module (`quant.zig`): traits
//! describing storage/scale layout per format (i8 W8A8, GGML row-block,
//! K-quant, IQ/TQ/FP4), Q8_K x4 packed-vs-direct quantization parity, the
//! exact i8 block-wise quantize-and-matmul path when the scale is 1, and
//! the TQ2_0 borrowed-blocks RHS view (no copy, no-op deinit).
const std = @import("std");
const quant = @import("quant.zig");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");

const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

const QuantizedMatmulRhsI8 = quant.QuantizedMatmulRhsI8;
const QuantizedMatmulFormat = quant.QuantizedMatmulFormat;
const QuantizedMatmulKernel = quant.QuantizedMatmulKernel;
const QuantizedStorageLayout = quant.QuantizedStorageLayout;
const QuantizedScaleLayout = quant.QuantizedScaleLayout;
const matmulTraits = quant.matmulTraits;
const matmulTraitsRuntime = quant.matmulTraitsRuntime;
const supportsMatmul = quant.supportsMatmul;
const qk_k_block_size = quant.qk_k_block_size;
const BlockQ8_Kx4 = quant.BlockQ8_Kx4;
const quantizeRowsQ8_K = quant.quantizeRowsQ8_K;
const packRowsQ8_Kx4 = quant.packRowsQ8_Kx4;
const quantizeRowsQ8_Kx4Into = quant.quantizeRowsQ8_Kx4Into;
const quantizeRhsBlockwiseI8 = quant.quantizeRhsBlockwiseI8;
const quantizeActivationsPerRowI8 = quant.quantizeActivationsPerRowI8;
const matmulI8BlockwiseRange = quant.matmulI8BlockwiseRange;

test "i8 block-wise quantized matmul traits describe storage layout" {
    const traits = QuantizedMatmulRhsI8.traits;

    try std.testing.expectEqual(QuantizedMatmulFormat.fucina_w8a8_rhs, traits.format);
    try std.testing.expectEqual(DType.f32, traits.source_dtype);
    try std.testing.expectEqual(DType.i8, traits.storage_dtype);
    try std.testing.expectEqual(DType.f32, traits.scale_dtype);
    try std.testing.expectEqual(@as(usize, 32), traits.effectiveGroupSize(0));
    try std.testing.expectEqual(@as(usize, 16), traits.effectiveGroupSize(16));
    try std.testing.expectEqual(@as(usize, 3), traits.groupCount(65, 32));
    try std.testing.expectEqual([2]usize{ 7, 65 }, traits.storageShape(65, 7));
    try std.testing.expectEqual([2]usize{ 7, 3 }, traits.scaleShape(65, 7, 32));
    try std.testing.expectEqual(@as(usize, 2 * 65 + 4), traits.storageIndex(2, 4, 65));
    try std.testing.expectEqual(@as(usize, 2 * 3 + 1), traits.scaleIndex(2, 1, 3));
    try std.testing.expectEqual(QuantizedMatmulKernel.fucina_w8a8_f32, traits.matmul_kernel);
}

test "ggml_q8_0 traits describe GGML block layout" {
    const traits = matmulTraits(.ggml_q8_0);

    try std.testing.expectEqual(QuantizedMatmulFormat.ggml_q8_0, traits.format);
    try std.testing.expectEqual(DType.f32, traits.source_dtype);
    try std.testing.expectEqual(DType.i8, traits.storage_dtype);
    try std.testing.expectEqual(DType.f16, traits.scale_dtype);
    try std.testing.expectEqual(@as(usize, 32), traits.block_size);
    try std.testing.expectEqual(@as(?usize, 34), traits.block_byte_size);
    try std.testing.expectEqual(QuantizedStorageLayout.ggml_blocks, traits.storage_layout);
    try std.testing.expectEqual(QuantizedScaleLayout.inline_block_scale, traits.scale_layout);
    try std.testing.expectEqual(@as(usize, 3), traits.storageRowSize(96));
    try std.testing.expectEqual([2]usize{ 7, 3 }, traits.storageShape(96, 7));
    try std.testing.expectEqual(QuantizedMatmulKernel.ggml_q8_0, traits.matmul_kernel);
}

test "ggml_q4_0 traits describe GGML block layout" {
    const traits = matmulTraits(.ggml_q4_0);

    try std.testing.expectEqual(QuantizedMatmulFormat.ggml_q4_0, traits.format);
    try std.testing.expectEqual(DType.f32, traits.source_dtype);
    try std.testing.expectEqual(DType.u8, traits.storage_dtype);
    try std.testing.expectEqual(DType.f16, traits.scale_dtype);
    try std.testing.expectEqual(@as(usize, 32), traits.block_size);
    try std.testing.expectEqual(@as(?usize, 18), traits.block_byte_size);
    try std.testing.expectEqual(QuantizedStorageLayout.ggml_blocks, traits.storage_layout);
    try std.testing.expectEqual(QuantizedScaleLayout.inline_block_scale, traits.scale_layout);
    try std.testing.expectEqual(@as(usize, 3), traits.storageRowSize(96));
    try std.testing.expectEqual([2]usize{ 7, 3 }, traits.storageShape(96, 7));
    try std.testing.expectEqual(QuantizedMatmulKernel.ggml_q4_0, traits.matmul_kernel);
}

test "GGML K-quant traits register block formats" {
    const cases = [_]struct {
        format: QuantizedMatmulFormat,
        size: usize,
        kernel: QuantizedMatmulKernel,
        supports_from_float: bool,
        supports_to_float: bool,
        supports_matmul: bool,
    }{
        .{ .format = .ggml_q2_k, .size = 84, .kernel = .ggml_q2_k, .supports_from_float = false, .supports_to_float = true, .supports_matmul = true },
        .{ .format = .ggml_q3_k, .size = 110, .kernel = .ggml_q3_k, .supports_from_float = false, .supports_to_float = true, .supports_matmul = true },
        .{ .format = .ggml_q4_k, .size = 144, .kernel = .ggml_q4_k, .supports_from_float = true, .supports_to_float = true, .supports_matmul = true },
        .{ .format = .ggml_q5_k, .size = 176, .kernel = .ggml_q5_k, .supports_from_float = true, .supports_to_float = true, .supports_matmul = true },
        .{ .format = .ggml_q6_k, .size = 210, .kernel = .ggml_q6_k, .supports_from_float = true, .supports_to_float = true, .supports_matmul = true },
        .{ .format = .ggml_q8_k, .size = 292, .kernel = .unsupported, .supports_from_float = true, .supports_to_float = true, .supports_matmul = false },
    };

    for (cases) |case| {
        const traits = matmulTraitsRuntime(case.format);
        try std.testing.expectEqual(case.format, traits.format);
        try std.testing.expectEqual(@as(usize, 256), traits.block_size);
        try std.testing.expectEqual(@as(?usize, case.size), traits.block_byte_size);
        try std.testing.expectEqual(QuantizedStorageLayout.ggml_blocks, traits.storage_layout);
        try std.testing.expectEqual(QuantizedScaleLayout.inline_block_scale, traits.scale_layout);
        try std.testing.expectEqual(case.supports_from_float, traits.supports_from_float);
        try std.testing.expectEqual(case.supports_to_float, traits.supports_to_float);
        try std.testing.expectEqual(case.supports_matmul, traits.supports_matmul);
        try std.testing.expectEqual(case.supports_matmul, supportsMatmul(case.format));
        try std.testing.expectEqual(case.kernel, traits.matmul_kernel);
    }
}

test "GGML IQ, TQ, and FP4 traits register implemented kernels" {
    const cases = [_]struct {
        format: QuantizedMatmulFormat,
        block_size: usize,
        size: usize,
        kernel: QuantizedMatmulKernel,
        supports_from_float: bool,
    }{
        .{ .format = .ggml_iq1_s, .block_size = 256, .size = 50, .kernel = .ggml_iq1_s, .supports_from_float = false },
        .{ .format = .ggml_iq1_m, .block_size = 256, .size = 56, .kernel = .ggml_iq1_m, .supports_from_float = false },
        .{ .format = .ggml_iq2_xxs, .block_size = 256, .size = 66, .kernel = .ggml_iq2_xxs, .supports_from_float = false },
        .{ .format = .ggml_iq2_xs, .block_size = 256, .size = 74, .kernel = .ggml_iq2_xs, .supports_from_float = false },
        .{ .format = .ggml_iq2_s, .block_size = 256, .size = 82, .kernel = .ggml_iq2_s, .supports_from_float = false },
        .{ .format = .ggml_iq3_xxs, .block_size = 256, .size = 98, .kernel = .ggml_iq3_xxs, .supports_from_float = false },
        .{ .format = .ggml_iq3_s, .block_size = 256, .size = 110, .kernel = .ggml_iq3_s, .supports_from_float = false },
        .{ .format = .ggml_iq4_nl, .block_size = 32, .size = 18, .kernel = .ggml_iq4_nl, .supports_from_float = false },
        .{ .format = .ggml_iq4_xs, .block_size = 256, .size = 136, .kernel = .ggml_iq4_xs, .supports_from_float = false },
        .{ .format = .ggml_tq1_0, .block_size = 256, .size = 54, .kernel = .ggml_tq1_0, .supports_from_float = false },
        .{ .format = .ggml_tq2_0, .block_size = 256, .size = 66, .kernel = .ggml_tq2_0, .supports_from_float = true },
        .{ .format = .ggml_mxfp4, .block_size = 32, .size = 17, .kernel = .ggml_mxfp4, .supports_from_float = false },
        .{ .format = .ggml_nvfp4, .block_size = 64, .size = 36, .kernel = .ggml_nvfp4, .supports_from_float = false },
    };

    for (cases) |case| {
        const traits = matmulTraitsRuntime(case.format);
        try std.testing.expectEqual(case.format, traits.format);
        try std.testing.expectEqual(case.block_size, traits.block_size);
        try std.testing.expectEqual(@as(?usize, case.size), traits.block_byte_size);
        try std.testing.expectEqual(QuantizedStorageLayout.ggml_blocks, traits.storage_layout);
        try std.testing.expectEqual(QuantizedScaleLayout.inline_block_scale, traits.scale_layout);
        try std.testing.expectEqual(case.supports_from_float, traits.supports_from_float);
        try std.testing.expectEqual(true, traits.supports_to_float);
        try std.testing.expectEqual(true, traits.supports_matmul);
        try std.testing.expectEqual(true, supportsMatmul(case.format));
        try std.testing.expectEqual(case.kernel, traits.matmul_kernel);
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
    var owned = try quant.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, &weights);
    defer owned.deinit();

    // Borrow the owning container's blocks: same storage, no dupe; deinit
    // frees nothing (allocator = null), so both containers may deinit.
    var borrowed = try quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, owned.rows.blocks);
    defer borrowed.deinit();
    try std.testing.expectEqual(owned.rows.blocks.ptr, borrowed.rows.blocks.ptr);
    try std.testing.expectEqual(@as(usize, 1), borrowed.rows.blocks_per_row);

    // One matmul through each container agrees bitwise.
    var lhs: [2 * k]f32 = undefined;
    for (&lhs, 0..) |*x, i| x.* = @floatFromInt(@as(i32, @intCast(i % 7)) - 3);
    var out_owned: [2 * n]f32 = undefined;
    var out_borrowed: [2 * n]f32 = undefined;
    quant.matmulTQ2_0F32RhsRange(&out_owned, &lhs, &owned, 2, n, 0, 2);
    quant.matmulTQ2_0F32RhsRange(&out_borrowed, &lhs, &borrowed, 2, n, 0, 2);
    try std.testing.expectEqualSlices(f32, &out_owned, &out_borrowed);

    // Length validation still applies to borrowed views.
    try std.testing.expectError(
        quant.QuantizedFormatError.InvalidQuantizedLength,
        quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n + 1, owned.rows.blocks),
    );
}
