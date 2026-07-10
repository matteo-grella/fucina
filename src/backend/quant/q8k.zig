//! Q8_K / Q8_0 encode-quantize-dequantize helpers + the K-quant `*FromBlocks`
//! RHS constructors. They live in this child-neutral leaf rather than the
//! quant.zig parent barrel so the quant kernel children can import them
//! without a quant.zig<->children import cycle. quant.zig re-exports these
//! so `quant.<sym>` callers are unchanged.

const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("../../tensor.zig");
const common = @import("common.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;

const has_aarch64_i8mm = common.has_aarch64_i8mm;
const quantizeToI8 = common.quantizeToI8;
const roundNearestEven = common.roundNearestEven;
const roundNearestEvenVec4ToI32 = common.roundNearestEvenVec4ToI32;
const roundHalfAwayFromZeroVec4ToI32 = common.roundHalfAwayFromZeroVec4ToI32;
const f32ToF16Bits = common.f32ToF16Bits;
const f16BitsToF32 = common.f16BitsToF32;
const f16x4BitsToF32 = common.f16x4BitsToF32;
const i8DotI32 = common.i8DotI32;
const QKV4i32 = common.QKV4i32;
const QKV4f32 = common.QKV4f32;

const qk_k_block_size = types.qk_k_block_size;
const q8_0_block_size = types.q8_0_block_size;
const k_scale_size = types.k_scale_size;
const BlockQ8_0 = types.BlockQ8_0;
const BlockQ8_K = types.BlockQ8_K;
const BlockQ8_Kx4 = types.BlockQ8_Kx4;
const BlockQ8_Kx2Mmla = types.BlockQ8_Kx2Mmla;
const BlockQ2_K = types.BlockQ2_K;
const BlockQ3_K = types.BlockQ3_K;
const BlockQ4_K = types.BlockQ4_K;
const BlockQ5_K = types.BlockQ5_K;
const BlockQ6_K = types.BlockQ6_K;
const QuantizedFormatError = types.QuantizedFormatError;
const QuantizedRowsQ8_0 = types.QuantizedRowsQ8_0;
const QuantizedMatmulRhsQ2_K = types.QuantizedMatmulRhsQ2_K;
const QuantizedMatmulRhsQ3_K = types.QuantizedMatmulRhsQ3_K;
const QuantizedMatmulRhsQ4_K = types.QuantizedMatmulRhsQ4_K;
const QuantizedMatmulRhsQ5_K = types.QuantizedMatmulRhsQ5_K;
const QuantizedMatmulRhsQ6_K = types.QuantizedMatmulRhsQ6_K;
const checkedProduct = types.checkedProduct;

pub fn quantizeRowQ8_0Into(dst: []BlockQ8_0, src: []const f32) !void {
    const block_count = try q8_0BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    if (comptime builtin.cpu.arch == .aarch64) {
        return quantizeRowQ8_0IntoAarch64(dst, src);
    }

    var block_index: usize = 0;
    while (block_index < block_count) : (block_index += 1) {
        const row = src[block_index * q8_0_block_size ..][0..q8_0_block_size];
        var amax: f32 = 0;
        for (row) |v| amax = @max(amax, @abs(v));

        const d = amax / 127.0;
        const inv_d: f32 = if (d == 0) 0 else 1.0 / d;

        dst[block_index].d = f32ToF16Bits(d);
        for (&dst[block_index].qs, row) |*q, v| q.* = quantizeToI8(v * inv_d);
    }
}
fn quantizeRowQ8_0IntoAarch64(dst: []BlockQ8_0, src: []const f32) void {
    var block_index: usize = 0;
    while (block_index < dst.len) : (block_index += 1) {
        const row = src[block_index * q8_0_block_size ..][0..q8_0_block_size];

        var amaxv: QKV4f32 = @splat(0);
        inline for (0..8) |j| {
            const v: QKV4f32 = row[j * 4 ..][0..4].*;
            amaxv = @max(amaxv, @abs(v));
        }

        const amax = @reduce(.Max, amaxv);
        const d = amax / 127.0;
        const inv_d: f32 = if (d == 0) 0 else 1.0 / d;

        dst[block_index].d = f32ToF16Bits(d);
        inline for (0..8) |j| {
            const v: QKV4f32 = row[j * 4 ..][0..4].*;
            const scaled = v * @as(QKV4f32, @splat(inv_d));
            const clamped = @max(@as(QKV4f32, @splat(-127.0)), @min(@as(QKV4f32, @splat(127.0)), scaled));
            const q = roundHalfAwayFromZeroVec4ToI32(clamped);
            inline for (0..4) |lane| dst[block_index].qs[j * 4 + lane] = @intCast(q[lane]);
        }
    }
}
pub fn dequantizeRowQ8_0Into(dst: []f32, src: []const BlockQ8_0) !void {
    if (dst.len != try checkedProduct(src.len, q8_0_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    // Explicit 8-lane vectors: the q8_0 KV-cache attention path dequantizes
    // every K/V row it streams through this function, and the scalar
    // element loop compiles to per-element converts (~2.4x decode attention
    // cost vs f16). Widen-multiply in vector chunks instead; same values
    // bit-for-bit (each element is one exact i8->f32 convert and one f32
    // multiply in both forms).
    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const dv: @Vector(8, f32) = @splat(d);
        const out = dst[block_index * q8_0_block_size ..][0..q8_0_block_size];
        inline for (0..q8_0_block_size / 8) |j| {
            const q: @Vector(8, i8) = block.qs[j * 8 ..][0..8].*;
            const w: @Vector(8, f32) = @floatFromInt(q);
            out[j * 8 ..][0..8].* = w * dv;
        }
    }
}
pub fn quantizeRowsQ8_0(allocator: Allocator, src: *const Tensor) !QuantizedRowsQ8_0 {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    const blocks_per_row = try q8_0BlockCount(cols);

    const blocks = try allocator.alloc(BlockQ8_0, try checkedProduct(rows, blocks_per_row));
    errdefer allocator.free(blocks);

    try quantizeRowsQ8_0Into(blocks, src);

    return .{
        .allocator = allocator,
        .blocks = blocks,
        .rows = rows,
        .cols = cols,
        .blocks_per_row = blocks_per_row,
    };
}
pub fn quantizeRowsQ8_0Into(blocks: []BlockQ8_0, src: *const Tensor) !void {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    const blocks_per_row = try q8_0BlockCount(cols);
    if (blocks.len != try checkedProduct(rows, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;

    const data = try src.dataConstChecked();
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        try quantizeRowQ8_0Into(
            blocks[row * blocks_per_row ..][0..blocks_per_row],
            data[row * cols ..][0..cols],
        );
    }
}
pub fn dequantizeRowsQ8_0Into(dst: *Tensor, src: *const QuantizedRowsQ8_0) !void {
    const view = try dst.rankView(2);
    if (view.dim(0) != src.rows or view.dim(1) != src.cols) return tensor.TensorError.ShapeMismatch;

    const out = try dst.dataChecked();
    var row: usize = 0;
    while (row < src.rows) : (row += 1) {
        try dequantizeRowQ8_0Into(out[row * src.cols ..][0..src.cols], src.rowBlocks(row));
    }
}
pub fn getRowsQ8_0Into(dst: *Tensor, table: *const QuantizedRowsQ8_0, indices: []const usize) !void {
    if (indices.len == 0) return tensor.TensorError.InvalidShape;
    const view = try dst.rankView(2);
    if (view.dim(0) != indices.len or view.dim(1) != table.cols) return tensor.TensorError.ShapeMismatch;

    const out = try dst.dataChecked();
    for (indices, 0..) |index, row| {
        if (index >= table.rows) return tensor.TensorError.IndexOutOfBounds;
        try dequantizeRowQ8_0Into(out[row * table.cols ..][0..table.cols], table.rowBlocks(index));
    }
}
pub fn q8_0BlockCount(len: usize) !usize {
    if (len % q8_0_block_size != 0) return QuantizedFormatError.InvalidQuantizedLength;
    return len / q8_0_block_size;
}
pub fn qkBlockCount(len: usize) !usize {
    if (len % qk_k_block_size != 0) return QuantizedFormatError.InvalidQuantizedLength;
    return len / qk_k_block_size;
}
pub fn blockCountExact(comptime block_size: usize, len: usize) !usize {
    if (len % block_size != 0) return QuantizedFormatError.InvalidQuantizedLength;
    return len / block_size;
}
pub fn quantizedMatmulRhsQ2_KFromBlocks(
    allocator: Allocator,
    k: usize,
    n: usize,
    blocks: []const BlockQ2_K,
) !QuantizedMatmulRhsQ2_K {
    const blocks_per_column = try qkBlockCount(k);
    if (blocks.len != try checkedProduct(n, blocks_per_column)) return QuantizedFormatError.InvalidQuantizedLength;
    const owned = try allocator.dupe(BlockQ2_K, blocks);
    return .{
        .allocator = allocator,
        .blocks = owned,
        .k = k,
        .n = n,
        .blocks_per_column = blocks_per_column,
    };
}
pub fn quantizedMatmulRhsQ3_KFromBlocks(
    allocator: Allocator,
    k: usize,
    n: usize,
    blocks: []const BlockQ3_K,
) !QuantizedMatmulRhsQ3_K {
    const blocks_per_column = try qkBlockCount(k);
    if (blocks.len != try checkedProduct(n, blocks_per_column)) return QuantizedFormatError.InvalidQuantizedLength;
    const owned = try allocator.dupe(BlockQ3_K, blocks);
    return .{
        .allocator = allocator,
        .blocks = owned,
        .k = k,
        .n = n,
        .blocks_per_column = blocks_per_column,
    };
}
pub fn quantizedMatmulRhsQ4_KFromBlocks(
    allocator: Allocator,
    k: usize,
    n: usize,
    blocks: []const BlockQ4_K,
) !QuantizedMatmulRhsQ4_K {
    const blocks_per_column = try qkBlockCount(k);
    if (blocks.len != try checkedProduct(n, blocks_per_column)) return QuantizedFormatError.InvalidQuantizedLength;
    const owned = try allocator.dupe(BlockQ4_K, blocks);
    return .{
        .allocator = allocator,
        .blocks = owned,
        .k = k,
        .n = n,
        .blocks_per_column = blocks_per_column,
    };
}
pub fn quantizedMatmulRhsQ5_KFromBlocks(
    allocator: Allocator,
    k: usize,
    n: usize,
    blocks: []const BlockQ5_K,
) !QuantizedMatmulRhsQ5_K {
    const blocks_per_column = try qkBlockCount(k);
    if (blocks.len != try checkedProduct(n, blocks_per_column)) return QuantizedFormatError.InvalidQuantizedLength;
    const owned = try allocator.dupe(BlockQ5_K, blocks);
    return .{
        .allocator = allocator,
        .blocks = owned,
        .k = k,
        .n = n,
        .blocks_per_column = blocks_per_column,
    };
}
pub fn quantizedMatmulRhsQ6_KFromBlocks(
    allocator: Allocator,
    k: usize,
    n: usize,
    blocks: []const BlockQ6_K,
) !QuantizedMatmulRhsQ6_K {
    const blocks_per_column = try qkBlockCount(k);
    if (blocks.len != try checkedProduct(n, blocks_per_column)) return QuantizedFormatError.InvalidQuantizedLength;
    const owned = try allocator.dupe(BlockQ6_K, blocks);
    return .{
        .allocator = allocator,
        .blocks = owned,
        .k = k,
        .n = n,
        .blocks_per_column = blocks_per_column,
    };
}
pub fn quantizeRowQ8_KInto(dst: []BlockQ8_K, src: []const f32) !void {
    const block_count = try qkBlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;
    if (comptime builtin.cpu.arch == .aarch64) {
        return quantizeRowQ8_KIntoAarch64(dst, src);
    }

    var block_index: usize = 0;
    while (block_index < block_count) : (block_index += 1) {
        const row = src[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var amaxv: QKV4f32 = @splat(0);
        var vec_index: usize = 0;
        while (vec_index < qk_k_block_size / 4) : (vec_index += 1) {
            const v: QKV4f32 = row[vec_index * 4 ..][0..4].*;
            amaxv = @max(amaxv, @abs(v));
        }
        const amax = @reduce(.Max, amaxv);
        var max_value: f32 = 0;
        for (row) |v| {
            if (@abs(v) == amax) {
                max_value = v;
                break;
            }
        }

        if (amax == 0) {
            dst[block_index].d = 0;
            @memset(&dst[block_index].qs, 0);
            @memset(&dst[block_index].bsums, 0);
            continue;
        }

        const inv_scale = -127.0 / max_value;
        for (&dst[block_index].qs, row) |*q, v| {
            const quantized = roundNearestEven(inv_scale * v);
            q.* = @intFromFloat(@min(127.0, quantized));
        }

        for (&dst[block_index].bsums, 0..) |*sum, group| {
            var acc: i32 = 0;
            for (dst[block_index].qs[group * 16 ..][0..16]) |q| acc += q;
            sum.* = @intCast(acc);
        }
        dst[block_index].d = 1.0 / inv_scale;
    }
}
fn quantizeRowQ8_KIntoAarch64(dst: []BlockQ8_K, src: []const f32) void {
    var block_index: usize = 0;
    while (block_index < dst.len) : (block_index += 1) {
        const row = src[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var block = &dst[block_index];

        var amaxv: QKV4f32 = @splat(0);
        var vec_index: usize = 0;
        while (vec_index < qk_k_block_size / 4) : (vec_index += 1) {
            const v: QKV4f32 = row[vec_index * 4 ..][0..4].*;
            amaxv = @max(amaxv, @abs(v));
        }
        const amax = @reduce(.Max, amaxv);
        var max_value: f32 = 0;
        for (row) |v| {
            if (@abs(v) == amax) {
                max_value = v;
                break;
            }
        }

        if (amax == 0) {
            block.d = 0;
            @memset(&block.qs, 0);
            @memset(&block.bsums, 0);
            continue;
        }

        const inv_scale = -127.0 / max_value;
        var bsums = [_]i32{0} ** 16;
        vec_index = 0;
        while (vec_index < qk_k_block_size / 4) : (vec_index += 1) {
            const v: QKV4f32 = row[vec_index * 4 ..][0..4].*;
            const q = @min(roundNearestEvenVec4ToI32(v * @as(QKV4f32, @splat(inv_scale))), @as(QKV4i32, @splat(127)));
            inline for (0..4) |lane| block.qs[vec_index * 4 + lane] = @intCast(q[lane]);
            bsums[vec_index / 4] += @reduce(.Add, q);
        }

        for (&block.bsums, bsums) |*sum, value| sum.* = @intCast(value);
        block.d = 1.0 / inv_scale;
    }
}
pub fn quantizeRowsQ8_K(allocator: Allocator, src: *const Tensor) ![]BlockQ8_K {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    const blocks_per_row = try qkBlockCount(cols);
    const data = try src.dataConstChecked();

    const blocks = try allocator.alloc(BlockQ8_K, try checkedProduct(rows, blocks_per_row));
    errdefer allocator.free(blocks);

    var row: usize = 0;
    while (row < rows) : (row += 1) {
        try quantizeRowQ8_KInto(
            blocks[row * blocks_per_row ..][0..blocks_per_row],
            data[row * cols ..][0..cols],
        );
    }
    return blocks;
}
pub fn quantizeRowsQ8_Kx4Into(blocks: []BlockQ8_Kx4, src: *const Tensor) !void {
    return quantizeRowsQ8_Kx4IntoImpl(blocks, src, false);
}
pub fn quantizeRowsQ8_Kx4PaddedInto(blocks: []BlockQ8_Kx4, src: *const Tensor) !void {
    return quantizeRowsQ8_Kx4IntoImpl(blocks, src, true);
}
pub fn quantizeRowsQ8_Kx4IntoImpl(blocks: []BlockQ8_Kx4, src: *const Tensor, comptime pad_rows: bool) !void {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    if (!pad_rows and rows % 4 != 0) return tensor.TensorError.InvalidShape;

    const blocks_per_row = try qkBlockCount(cols);
    const row_groups = if (pad_rows) (rows + 3) / 4 else rows / 4;
    if (blocks.len != try checkedProduct(row_groups, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;

    const data = try src.dataConstChecked();
    for (0..row_groups) |row_group| {
        const rows_in_group = @min(rows - row_group * 4, 4);
        quantizeRowGroupQ8_Kx4Into(
            blocks[row_group * blocks_per_row ..][0..blocks_per_row],
            data[row_group * 4 * cols ..][0 .. rows_in_group * cols],
            rows_in_group,
            cols,
        );
    }
}
pub fn quantizeRowGroupQ8_Kx4Into(blocks: []BlockQ8_Kx4, data: []const f32, rows_in_group: usize, cols: usize) void {
    const blocks_per_row = blocks.len;
    {
        for (0..blocks_per_row) |block_index| {
            var dst = &blocks[block_index];
            inline for (0..4) |row_lane| {
                const row = row_lane;
                if (row >= rows_in_group) {
                    zeroQ8Kx4Lane(dst, row_lane);
                } else {
                    const source = data[row * cols + block_index * qk_k_block_size ..][0..qk_k_block_size];

                    var amaxv: QKV4f32 = @splat(0);
                    var vec_index: usize = 0;
                    while (vec_index < qk_k_block_size / 4) : (vec_index += 1) {
                        const v: QKV4f32 = source[vec_index * 4 ..][0..4].*;
                        amaxv = @max(amaxv, @abs(v));
                    }
                    const amax = @reduce(.Max, amaxv);
                    var max_value: f32 = 0;
                    for (source) |v| {
                        if (@abs(v) == amax) {
                            max_value = v;
                            break;
                        }
                    }

                    if (amax == 0) {
                        zeroQ8Kx4Lane(dst, row_lane);
                    } else {
                        const inv_scale = -127.0 / max_value;
                        var bsums = [_]i32{0} ** 16;
                        vec_index = 0;
                        while (vec_index < qk_k_block_size / 4) : (vec_index += 1) {
                            const v: QKV4f32 = source[vec_index * 4 ..][0..4].*;
                            const q = @min(roundNearestEvenVec4ToI32(v * @as(QKV4f32, @splat(inv_scale))), @as(QKV4i32, @splat(127)));
                            inline for (0..4) |lane| {
                                dst.qs[vec_index * 16 + row_lane * 4 + lane] = @intCast(q[lane]);
                            }
                            bsums[vec_index / 4] += @reduce(.Add, q);
                        }

                        inline for (0..16) |subblock| {
                            dst.bsums[(subblock / 4) * 16 + row_lane * 4 + subblock % 4] = @intCast(bsums[subblock]);
                        }
                        dst.d[row_lane] = 1.0 / inv_scale;
                    }
                }
            }
        }
    }
}
pub fn zeroQ8Kx4Lane(block: *BlockQ8_Kx4, comptime row_lane: usize) void {
    block.d[row_lane] = 0;
    var feature_group: usize = 0;
    while (feature_group < qk_k_block_size / 4) : (feature_group += 1) {
        inline for (0..4) |lane| {
            block.qs[feature_group * 16 + row_lane * 4 + lane] = 0;
        }
    }
    var subblock: usize = 0;
    while (subblock < 16) : (subblock += 1) {
        block.bsums[(subblock / 4) * 16 + row_lane * 4 + subblock % 4] = 0;
    }
}
pub fn quantizeRowsQ8_Kx2MmlaInto(blocks: []BlockQ8_Kx2Mmla, src: *const Tensor) !void {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    if (rows % 2 != 0) return tensor.TensorError.InvalidShape;

    const blocks_per_row = try qkBlockCount(cols);
    if (blocks.len != try checkedProduct(rows / 2, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;

    const data = try src.dataConstChecked();
    for (0..rows / 2) |row_group| {
        for (0..blocks_per_row) |block_index| {
            var dst = &blocks[row_group * blocks_per_row + block_index];
            var row_qs: [2][qk_k_block_size]i8 = undefined;
            var row_bsums: [2][8]i32 = .{ [_]i32{0} ** 8, [_]i32{0} ** 8 };

            inline for (0..2) |row_lane| {
                const row = row_group * 2 + row_lane;
                const source = data[row * cols + block_index * qk_k_block_size ..][0..qk_k_block_size];

                var amaxv: QKV4f32 = @splat(0);
                var vec_index: usize = 0;
                while (vec_index < qk_k_block_size / 4) : (vec_index += 1) {
                    const v: QKV4f32 = source[vec_index * 4 ..][0..4].*;
                    amaxv = @max(amaxv, @abs(v));
                }
                const amax = @reduce(.Max, amaxv);
                var max_value: f32 = 0;
                for (source) |v| {
                    if (@abs(v) == amax) {
                        max_value = v;
                        break;
                    }
                }

                if (amax == 0) {
                    dst.d[row_lane] = 0;
                    @memset(row_qs[row_lane][0..], 0);
                } else {
                    const inv_scale = -127.0 / max_value;
                    vec_index = 0;
                    while (vec_index < qk_k_block_size / 4) : (vec_index += 1) {
                        const v: QKV4f32 = source[vec_index * 4 ..][0..4].*;
                        const q = @min(roundNearestEvenVec4ToI32(v * @as(QKV4f32, @splat(inv_scale))), @as(QKV4i32, @splat(127)));
                        inline for (0..4) |lane| {
                            row_qs[row_lane][vec_index * 4 + lane] = @intCast(q[lane]);
                        }
                        row_bsums[row_lane][vec_index / 8] += @reduce(.Add, q);
                    }
                    dst.d[row_lane] = 1.0 / inv_scale;
                }

                inline for (0..8) |subblock| {
                    dst.bsums[subblock * 2 + row_lane] = @intCast(row_bsums[row_lane][subblock]);
                }
            }

            inline for (0..8) |subblock| {
                inline for (0..2) |half| {
                    const dst_base = subblock * 64 + half * 32;
                    const src_base = subblock * 32 + half * 16;
                    inline for (0..8) |lane| {
                        dst.qs[dst_base + lane] = row_qs[0][src_base + lane];
                        dst.qs[dst_base + 8 + lane] = row_qs[1][src_base + lane];
                        dst.qs[dst_base + 16 + lane] = row_qs[0][src_base + 8 + lane];
                        dst.qs[dst_base + 24 + lane] = row_qs[1][src_base + 8 + lane];
                    }
                }
            }
        }
    }
}
pub fn packRowsQ8_Kx4(
    allocator: Allocator,
    blocks: []const BlockQ8_K,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
) ![]BlockQ8_Kx4 {
    if (rows % 4 != 0) return tensor.TensorError.InvalidShape;
    if (blocks_per_row != try qkBlockCount(cols)) return tensor.TensorError.InvalidShape;
    if (blocks.len != try checkedProduct(rows, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;

    const row_groups = rows / 4;
    const packed_blocks = try allocator.alloc(BlockQ8_Kx4, try checkedProduct(row_groups, blocks_per_row));
    errdefer allocator.free(packed_blocks);

    for (0..row_groups) |group| {
        for (0..blocks_per_row) |block_index| {
            const src = [_]*const BlockQ8_K{
                &blocks[(group * 4 + 0) * blocks_per_row + block_index],
                &blocks[(group * 4 + 1) * blocks_per_row + block_index],
                &blocks[(group * 4 + 2) * blocks_per_row + block_index],
                &blocks[(group * 4 + 3) * blocks_per_row + block_index],
            };
            var dst = &packed_blocks[group * blocks_per_row + block_index];
            inline for (0..4) |row| dst.d[row] = src[row].d;

            for (0..qk_k_block_size / 4) |feature_group| {
                inline for (0..4) |row| {
                    inline for (0..4) |lane| {
                        dst.qs[feature_group * 16 + row * 4 + lane] = src[row].qs[feature_group * 4 + lane];
                    }
                }
            }

            inline for (0..4) |row| {
                inline for (0..16) |subblock| {
                    dst.bsums[(subblock / 4) * 16 + row * 4 + subblock % 4] = src[row].bsums[subblock];
                }
            }
        }
    }

    return packed_blocks;
}
pub fn packRowsQ8_Kx4PaddedInto(
    dst: []BlockQ8_Kx4,
    blocks: []const BlockQ8_K,
    m: usize,
    blocks_per_row: usize,
) void {
    const row_groups = (m + 3) / 4;
    for (0..row_groups) |group| {
        for (0..blocks_per_row) |block_index| {
            var d = &dst[group * blocks_per_row + block_index];
            inline for (0..4) |row_lane| {
                const row = group * 4 + row_lane;
                if (row >= m) {
                    zeroQ8Kx4Lane(d, row_lane);
                } else {
                    const src = &blocks[row * blocks_per_row + block_index];
                    d.d[row_lane] = src.d;
                    for (0..qk_k_block_size / 4) |feature_group| {
                        inline for (0..4) |lane| {
                            d.qs[feature_group * 16 + row_lane * 4 + lane] = src.qs[feature_group * 4 + lane];
                        }
                    }
                    inline for (0..16) |subblock| {
                        d.bsums[(subblock / 4) * 16 + row_lane * 4 + subblock % 4] = src.bsums[subblock];
                    }
                }
            }
        }
    }
}
pub fn q4Kx8D(bits: [8]u16, comptime group: usize) QKV4f32 {
    return f16x4BitsToF32(.{
        bits[group * 4 + 0],
        bits[group * 4 + 1],
        bits[group * 4 + 2],
        bits[group * 4 + 3],
    });
}
pub fn q4Kx8Scales(values: *const [8 * 8]u8, comptime subblock: usize, comptime group: usize) QKV4i32 {
    const offset = subblock * 8 + group * 4;
    return .{
        values[offset + 0],
        values[offset + 1],
        values[offset + 2],
        values[offset + 3],
    };
}
pub fn dequantizeBlockQ8_KInto(dst: *[qk_k_block_size]f32, src: *const BlockQ8_K) void {
    for (dst, src.qs) |*out, q| out.* = src.d * @as(f32, @floatFromInt(q));
}
const ScaleMinK4 = struct {
    scale: u8,
    min: u8,
};
pub fn getScaleMinK4(q: *const [k_scale_size]u8, index: usize) ScaleMinK4 {
    if (index < 4) {
        return .{
            .scale = q[index] & 63,
            .min = q[index + 4] & 63,
        };
    }
    return .{
        .scale = (q[index + 4] & 0x0f) | ((q[index - 4] >> 6) << 4),
        .min = (q[index + 4] >> 4) | ((q[index] >> 6) << 4),
    };
}
pub const group_max_eps: f32 = 1e-15;
pub fn nearestInt(fval: f32) i32 {
    std.debug.assert(@abs(fval) <= 4194303.0);
    const val: f32 = fval + 12582912.0;
    const bits: i32 = @bitCast(val);
    return (bits & 0x007fffff) - 0x00400000;
}
fn qxQuantWeight(rmse_type: i32, v: f32, qw: ?[]const f32, i: usize) f32 {
    if (qw) |w| return w[i];
    return switch (rmse_type) {
        1 => v * v,
        2 => 1,
        3 => @abs(v),
        else => @sqrt(@abs(v)),
    };
}
pub fn makeQxQuants(nmax: i32, x: []const f32, L: []i8, rmse_type_in: i32, qw: ?[]const f32) f32 {
    std.debug.assert(L.len == x.len);
    var max: f32 = 0;
    var amax: f32 = 0;
    for (x) |v| {
        const ax = @abs(v);
        if (ax > amax) {
            amax = ax;
            max = v;
        }
    }
    if (amax < group_max_eps) { // all zero
        @memset(L, 0);
        return 0;
    }
    const nmax_f: f32 = @floatFromInt(nmax);
    var iscale: f32 = -nmax_f / max;
    var rmse_type = rmse_type_in;
    if (rmse_type == 0) {
        for (x, L) |v, *l_out| {
            const l = nearestInt(iscale * v);
            l_out.* = @intCast(nmax + @max(-nmax, @min(nmax - 1, l)));
        }
        return 1 / iscale;
    }
    var return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    var sumlx: f32 = 0;
    var suml2: f32 = 0;
    for (x, L, 0..) |v, *l_out, i| {
        var l = nearestInt(iscale * v);
        l = @max(-nmax, @min(nmax - 1, l));
        l_out.* = @intCast(l + nmax);
        const w = qxQuantWeight(rmse_type, v, qw, i);
        const lf: f32 = @floatFromInt(l);
        sumlx += w * v * lf;
        suml2 += w * lf * lf;
    }
    var scale: f32 = if (suml2 != 0) sumlx / suml2 else 0.0;
    if (return_early) return if (suml2 > 0) 0.5 * (scale + 1 / iscale) else 1 / iscale;
    var best = scale * sumlx;
    var is: i32 = -9;
    while (is <= 9) : (is += 1) {
        if (is == 0) continue;
        iscale = -(nmax_f + 0.1 * @as(f32, @floatFromInt(is))) / max;
        sumlx = 0;
        suml2 = 0;
        for (x, 0..) |v, i| {
            var l = nearestInt(iscale * v);
            l = @max(-nmax, @min(nmax - 1, l));
            const w = qxQuantWeight(rmse_type, v, qw, i);
            const lf: f32 = @floatFromInt(l);
            sumlx += w * v * lf;
            suml2 += w * lf * lf;
        }
        if (suml2 > 0 and sumlx * sumlx > best * suml2) {
            for (x, L) |v, *l_out| {
                const l = nearestInt(iscale * v);
                l_out.* = @intCast(nmax + @max(-nmax, @min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}
pub fn makeQkx2Quants(
    nmax: i32,
    x: []const f32,
    weights: []const f32,
    L: []u8,
    the_min: *f32,
    Laux: []u8,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) f32 {
    const n = x.len;
    std.debug.assert(L.len == n and weights.len == n and Laux.len == n);
    var min = x[0];
    var max = x[0];
    var sum_w = weights[0];
    var sum_x = sum_w * x[0];
    for (x[1..], weights[1..]) |v, w| {
        if (v < min) min = v;
        if (v > max) max = v;
        sum_w += w;
        sum_x += w * v;
    }
    if (min > 0) min = 0;
    if (max == min) {
        @memset(L, 0);
        the_min.* = -min;
        return 0;
    }
    const nmax_f: f32 = @floatFromInt(nmax);
    var iscale = nmax_f / (max - min);
    var scale = 1 / iscale;
    var best_error: f32 = 0;
    for (x, L, weights) |v, *l_out, w| {
        const l = nearestInt(iscale * (v - min));
        l_out.* = @intCast(@max(0, @min(nmax, l)));
        var diff = scale * @as(f32, @floatFromInt(l_out.*)) + min - v;
        diff = if (use_mad) @abs(diff) else diff * diff;
        best_error += w * diff;
    }
    if (nstep < 1) {
        the_min.* = -min;
        return scale;
    }
    var is: i32 = 0;
    while (is <= nstep) : (is += 1) {
        iscale = (rmin + rdelta * @as(f32, @floatFromInt(is)) + nmax_f) / (max - min);
        var sum_l: f32 = 0;
        var sum_l2: f32 = 0;
        var sum_xl: f32 = 0;
        for (x, Laux, weights) |v, *laux, w| {
            var l = nearestInt(iscale * (v - min));
            l = @max(0, @min(nmax, l));
            laux.* = @intCast(l);
            const lf: f32 = @floatFromInt(l);
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * v;
        }
        const determinant = sum_w * sum_l2 - sum_l * sum_l;
        if (determinant > 0) {
            var this_scale = (sum_w * sum_xl - sum_x * sum_l) / determinant;
            var this_min = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            var cur_error: f32 = 0;
            for (x, Laux, weights) |v, laux, w| {
                var diff = this_scale * @as(f32, @floatFromInt(laux)) + this_min - v;
                diff = if (use_mad) @abs(diff) else diff * diff;
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                @memcpy(L, Laux);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    the_min.* = -min;
    return scale;
}
pub fn fillQ8KPattern(block: *BlockQ8_K) void {
    block.d = 1;
    for (&block.qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast(i % 17)) - 8);
    for (&block.bsums, 0..) |*sum, group| {
        var acc: i32 = 0;
        for (block.qs[group * 16 ..][0..16]) |q| acc += q;
        sum.* = @intCast(acc);
    }
}
