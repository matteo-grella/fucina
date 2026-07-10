//! Cold (rarely-used, generic-format) quantization code relocated out of quant.zig.
//! See quant.zig for the hot/live kernels and all shared type/helper definitions.
//! This module holds only cold *function bodies*; every shared symbol it references
//! is aliased from quant.zig (`qm`) below so the moved bodies compile unchanged.

const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const tensor = @import("../../tensor.zig");
const tables = @import("../quant_tables.zig");
const q8k_mod = @import("q8k.zig");
const types_mod = @import("types.zig");
const common = @import("common.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

// Shared symbols defined in quant.zig, aliased here so moved cold bodies compile unchanged.
const BlockIQ1_M = types_mod.BlockIQ1_M;
const BlockIQ1_S = types_mod.BlockIQ1_S;
const BlockIQ2_S = types_mod.BlockIQ2_S;
const BlockIQ2_XS = types_mod.BlockIQ2_XS;
const BlockIQ2_XXS = types_mod.BlockIQ2_XXS;
const BlockIQ3_S = types_mod.BlockIQ3_S;
const BlockIQ3_XXS = types_mod.BlockIQ3_XXS;
const BlockIQ4_NL = types_mod.BlockIQ4_NL;
const BlockIQ4_XS = types_mod.BlockIQ4_XS;
const BlockMXFP4 = types_mod.BlockMXFP4;
const BlockNVFP4 = types_mod.BlockNVFP4;
const BlockQ1_0 = types_mod.BlockQ1_0;
const BlockQ2_K = types_mod.BlockQ2_K;
const BlockQ3_K = types_mod.BlockQ3_K;
const BlockQ4_0 = types_mod.BlockQ4_0;
const BlockQ4_1 = types_mod.BlockQ4_1;
const BlockQ5_0 = types_mod.BlockQ5_0;
const BlockQ5_1 = types_mod.BlockQ5_1;
const BlockQ8_0 = types_mod.BlockQ8_0;
const BlockQ8_1 = types_mod.BlockQ8_1;
const BlockQ8_K = types_mod.BlockQ8_K;
const BlockTQ1_0 = types_mod.BlockTQ1_0;
const BlockTQ2_0 = types_mod.BlockTQ2_0;
const Q4V16i16 = common.Q4V16i16;
const Q4V16i8 = common.Q4V16i8;
const Q4V16u8 = common.Q4V16u8;
const QKV16i16 = common.QKV16i16;
const QKV16i32 = common.QKV16i32;
const QKV16i8 = common.QKV16i8;
const QKV16u16 = common.QKV16u16;
const QKV16u8 = common.QKV16u8;
const QKV8i16 = common.QKV8i16;
const QKV8i32 = common.QKV8i32;
const QKV8i8 = common.QKV8i8;
const QKV8u8 = common.QKV8u8;
const QuantizedFormatError = types_mod.QuantizedFormatError;
const QuantizedMatmulRhsQ1_0 = types_mod.QuantizedMatmulRhsQ1_0;
const QuantizedMatmulRhsQ2_K = types_mod.QuantizedMatmulRhsQ2_K;
const QuantizedMatmulRhsQ3_K = types_mod.QuantizedMatmulRhsQ3_K;
const QuantizedMatmulRhsQ4_0 = types_mod.QuantizedMatmulRhsQ4_0;
const QuantizedMatmulRhsQ4_1 = types_mod.QuantizedMatmulRhsQ4_1;
const QuantizedMatmulRhsQ5_0 = types_mod.QuantizedMatmulRhsQ5_0;
const QuantizedMatmulRhsQ5_1 = types_mod.QuantizedMatmulRhsQ5_1;
const QuantizedMatmulRhsRowsFor = types_mod.QuantizedMatmulRhsRowsFor;
const QuantizedRowsQ4_0 = types_mod.QuantizedRowsQ4_0;
const QuantizedRowsQ8_1 = types_mod.QuantizedRowsQ8_1;
const checkedProduct = types_mod.checkedProduct;
const dequantizeBlockQ8_KInto = q8k_mod.dequantizeBlockQ8_KInto;
const dequantizeRowQ8_0Into = q8k_mod.dequantizeRowQ8_0Into;
const dotDense = common.dotDense;
const f16BitsToF32 = common.f16BitsToF32;
const f32ToF16Bits = common.f32ToF16Bits;
const fillQ8KPattern = q8k_mod.fillQ8KPattern;
const iq4_nl_block_size = types_mod.iq4_nl_block_size;
const mxfp4_block_size = types_mod.mxfp4_block_size;
const nvfp4_block_size = types_mod.nvfp4_block_size;
const nvfp4_subblock_size = types_mod.nvfp4_subblock_size;
const q1_0_block_size = types_mod.q1_0_block_size;
const q4_0_block_size = types_mod.q4_0_block_size;
const q4_1_block_size = types_mod.q4_1_block_size;
const q5_0_block_size = types_mod.q5_0_block_size;
const q5_1_block_size = types_mod.q5_1_block_size;
const q8_0_block_size = types_mod.q8_0_block_size;
const q8_1_block_size = types_mod.q8_1_block_size;
const qk_col_block = common.qk_col_block;
const qk_k_block_size = types_mod.qk_k_block_size;
const quantizeToI8 = common.quantizeToI8;
const quantizedMatmulRhsQ2_KFromBlocks = q8k_mod.quantizedMatmulRhsQ2_KFromBlocks;
const quantizedMatmulRhsQ3_KFromBlocks = q8k_mod.quantizedMatmulRhsQ3_KFromBlocks;

pub fn quantizeRowQ4_0Into(dst: []BlockQ4_0, src: []const f32) !void {
    const block_count = try q4_0BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    var block_index: usize = 0;
    while (block_index < block_count) : (block_index += 1) {
        const row = src[block_index * q4_0_block_size ..][0..q4_0_block_size];
        var amax: f32 = 0;
        var max_value: f32 = 0;
        for (row) |v| {
            const abs_v = @abs(v);
            if (amax < abs_v) {
                amax = abs_v;
                max_value = v;
            }
        }

        const d = max_value / -8.0;
        var inv_d: f32 = if (d == 0) 0 else 1.0 / d;
        // Degenerate-but-finite blocks (subnormal spread) overflow 1/d to
        // inf, and 0*inf = NaN would reach @intFromFloat (UB in ReleaseFast).
        // In-contract blocks have finite inv_d, so goldens are unchanged.
        if (!std.math.isFinite(inv_d)) inv_d = 0;

        dst[block_index].d = f32ToF16Bits(d);
        for (&dst[block_index].qs, 0..) |*q, j| {
            const x0 = quantizeToQ4_0Nibble(row[j] * inv_d);
            const x1 = quantizeToQ4_0Nibble(row[q4_0_block_size / 2 + j] * inv_d);
            q.* = x0 | (x1 << 4);
        }
    }
}

pub fn dequantizeRowQ4_0Into(dst: []f32, src: []const BlockQ4_0) !void {
    if (dst.len != try checkedProduct(src.len, q4_0_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * q4_0_block_size ..][0..q4_0_block_size];
        for (block.qs, 0..) |q, j| {
            const x0: i32 = @as(i32, q & 0x0f) - 8;
            const x1: i32 = @as(i32, q >> 4) - 8;
            out[j] = @as(f32, @floatFromInt(x0)) * d;
            out[q4_0_block_size / 2 + j] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

pub fn quantizeRowsQ4_0(allocator: Allocator, src: *const Tensor) !QuantizedRowsQ4_0 {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    const blocks_per_row = try q4_0BlockCount(cols);
    const data = try src.dataConstChecked();

    const blocks = try allocator.alloc(BlockQ4_0, try checkedProduct(rows, blocks_per_row));
    errdefer allocator.free(blocks);

    var row: usize = 0;
    while (row < rows) : (row += 1) {
        try quantizeRowQ4_0Into(
            blocks[row * blocks_per_row ..][0..blocks_per_row],
            data[row * cols ..][0..cols],
        );
    }

    return .{
        .allocator = allocator,
        .blocks = blocks,
        .rows = rows,
        .cols = cols,
        .blocks_per_row = blocks_per_row,
    };
}

pub fn quantizeMatmulRhsQ4_0(allocator: Allocator, rhs: *const Tensor) !QuantizedMatmulRhsQ4_0 {
    const view = try rhs.rankView(2);
    const k = view.dim(0);
    const n = view.dim(1);
    const blocks_per_column = try q4_0BlockCount(k);
    const data = try rhs.dataConstChecked();

    const blocks = try allocator.alloc(BlockQ4_0, try checkedProduct(n, blocks_per_column));
    errdefer allocator.free(blocks);
    const scratch = try allocator.alloc(f32, k);
    defer allocator.free(scratch);

    var col: usize = 0;
    while (col < n) : (col += 1) {
        var p: usize = 0;
        while (p < k) : (p += 1) scratch[p] = data[p * n + col];
        try quantizeRowQ4_0Into(
            blocks[col * blocks_per_column ..][0..blocks_per_column],
            scratch,
        );
    }

    return .{
        .rows = .{
            .allocator = allocator,
            .blocks = blocks,
            .rows = n,
            .cols = k,
            .blocks_per_row = blocks_per_column,
        },
        .k = k,
        .n = n,
    };
}

pub fn dequantizeRowsQ4_0Into(dst: *Tensor, src: *const QuantizedRowsQ4_0) !void {
    const view = try dst.rankView(2);
    if (view.dim(0) != src.rows or view.dim(1) != src.cols) return tensor.TensorError.ShapeMismatch;

    const out = try dst.dataChecked();
    var row: usize = 0;
    while (row < src.rows) : (row += 1) {
        try dequantizeRowQ4_0Into(out[row * src.cols ..][0..src.cols], src.rowBlocks(row));
    }
}

pub fn getRowsQ4_0Into(dst: *Tensor, table: *const QuantizedRowsQ4_0, indices: []const usize) !void {
    if (indices.len == 0) return tensor.TensorError.InvalidShape;
    const view = try dst.rankView(2);
    if (view.dim(0) != indices.len or view.dim(1) != table.cols) return tensor.TensorError.ShapeMismatch;

    const out = try dst.dataChecked();
    for (indices, 0..) |index, row| {
        if (index >= table.rows) return tensor.TensorError.IndexOutOfBounds;
        try dequantizeRowQ4_0Into(out[row * table.cols ..][0..table.cols], table.rowBlocks(index));
    }
}

pub fn q4_0BlockCount(len: usize) !usize {
    if (len % q4_0_block_size != 0) return QuantizedFormatError.InvalidQuantizedLength;
    return len / q4_0_block_size;
}

fn quantizeToQ4_0Nibble(x: f32) u8 {
    // ggml: MIN(15, (int8_t)(x + 8.5f)). Clamp in float space with Zig's
    // NaN-collapsing @min/@max so a degenerate x (NaN/inf) never reaches
    // @intFromFloat (UB in ReleaseFast). In-contract x is in [-8, 8], where
    // trunc(x + 8.5) is in [0, 16] and both formulations agree byte-exactly.
    return @intFromFloat(@min(@as(f32, 15.0), @max(@as(f32, 0.0), x + 8.5)));
}

pub fn q1_0BlockCount(len: usize) !usize {
    if (len % q1_0_block_size != 0) return QuantizedFormatError.InvalidQuantizedLength;
    return len / q1_0_block_size;
}

pub fn q4_1BlockCount(len: usize) !usize {
    if (len % q4_1_block_size != 0) return QuantizedFormatError.InvalidQuantizedLength;
    return len / q4_1_block_size;
}

pub fn q5_0BlockCount(len: usize) !usize {
    if (len % q5_0_block_size != 0) return QuantizedFormatError.InvalidQuantizedLength;
    return len / q5_0_block_size;
}

pub fn q5_1BlockCount(len: usize) !usize {
    if (len % q5_1_block_size != 0) return QuantizedFormatError.InvalidQuantizedLength;
    return len / q5_1_block_size;
}

pub fn q8_1BlockCount(len: usize) !usize {
    if (len % q8_1_block_size != 0) return QuantizedFormatError.InvalidQuantizedLength;
    return len / q8_1_block_size;
}

pub fn dequantizeRowQ1_0Into(dst: []f32, src: []const BlockQ1_0) !void {
    if (dst.len != try checkedProduct(src.len, q1_0_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * q1_0_block_size ..][0..q1_0_block_size];
        for (out, 0..) |*y, j| {
            const mask: u8 = @as(u8, 1) << @intCast(j % 8);
            y.* = if ((block.qs[j / 8] & mask) != 0) d else -d;
        }
    }
}

/// f32 -> Q4_1 row encoder; faithful port of ggml's quantize_row_q4_1_ref
/// (byte-exact, see quant/encode_golden_test.zig). Assumes finite input;
/// degenerate-but-finite blocks (subnormal/overflowing spreads) produce
/// defined clamped output instead of @intFromFloat UB.
pub fn quantizeRowQ4_1Into(dst: []BlockQ4_1, src: []const f32) !void {
    const block_count = try q4_1BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    for (dst, 0..) |*block, block_index| {
        const row = src[block_index * q4_1_block_size ..][0..q4_1_block_size];
        var min: f32 = std.math.floatMax(f32);
        var max: f32 = -std.math.floatMax(f32);
        for (row) |v| {
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const d = (max - min) / 15.0;
        var inv_d: f32 = if (d != 0) 1.0 / d else 0.0;
        // Degenerate-but-finite blocks (subnormal spread -> 1/d overflows to
        // inf; max-min overflow -> d = inf) would feed NaN (0*inf / inf*0)
        // into @intFromFloat (UB in ReleaseFast). In-contract blocks have
        // finite inv_d, so goldens are unchanged.
        if (!std.math.isFinite(inv_d)) inv_d = 0;

        block.dm = .{ f32ToF16Bits(d), f32ToF16Bits(min) };
        for (&block.qs, 0..) |*q, j| {
            const x0 = (row[j] - min) * inv_d;
            const x1 = (row[q4_1_block_size / 2 + j] - min) * inv_d;
            // ggml: MIN(15, (int8_t)(x + 0.5f)); the float-space clamp with
            // NaN-collapsing @min/@max is byte-identical for in-contract
            // x in [0, 15] and keeps NaN away from @intFromFloat.
            const xi0: u8 = @intFromFloat(@min(@as(f32, 15.0), @max(@as(f32, 0.0), x0 + 0.5)));
            const xi1: u8 = @intFromFloat(@min(@as(f32, 15.0), @max(@as(f32, 0.0), x1 + 0.5)));
            q.* = xi0 | (xi1 << 4);
        }
    }
}

pub fn dequantizeRowQ4_1Into(dst: []f32, src: []const BlockQ4_1) !void {
    if (dst.len != try checkedProduct(src.len, q4_1_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.dm[0]);
        const m = f16BitsToF32(block.dm[1]);
        const out = dst[block_index * q4_1_block_size ..][0..q4_1_block_size];
        for (block.qs, 0..) |q, j| {
            out[j] = @as(f32, @floatFromInt(q & 0x0f)) * d + m;
            out[q4_1_block_size / 2 + j] = @as(f32, @floatFromInt(q >> 4)) * d + m;
        }
    }
}

/// f32 -> Q5_0 row encoder; faithful port of ggml's quantize_row_q5_0_ref
/// (byte-exact, see quant/encode_golden_test.zig). Assumes finite input;
/// degenerate-but-finite blocks (subnormal spreads) produce defined clamped
/// output instead of @intFromFloat UB.
pub fn quantizeRowQ5_0Into(dst: []BlockQ5_0, src: []const f32) !void {
    const block_count = try q5_0BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    for (dst, 0..) |*block, block_index| {
        const row = src[block_index * q5_0_block_size ..][0..q5_0_block_size];
        var amax: f32 = 0;
        var max_value: f32 = 0;
        for (row) |v| {
            const abs_v = @abs(v);
            if (amax < abs_v) {
                amax = abs_v;
                max_value = v;
            }
        }

        const d = max_value / -16.0;
        var inv_d: f32 = if (d != 0) 1.0 / d else 0.0;
        // Degenerate-but-finite blocks (subnormal spread) overflow 1/d to
        // inf, and 0*inf = NaN would reach @intFromFloat (UB in ReleaseFast).
        // In-contract blocks have finite inv_d, so goldens are unchanged.
        if (!std.math.isFinite(inv_d)) inv_d = 0;

        block.d = f32ToF16Bits(d);

        var qh: u32 = 0;
        for (&block.qs, 0..) |*q, j| {
            const x0 = row[j] * inv_d;
            const x1 = row[q5_0_block_size / 2 + j] * inv_d;
            // ggml: MIN(31, (int8_t)(x + 16.5f)); the float-space clamp with
            // NaN-collapsing @min/@max is byte-identical for in-contract
            // x in [-16, 16] and keeps NaN away from @intFromFloat.
            const xi0: u8 = @intFromFloat(@min(@as(f32, 31.0), @max(@as(f32, 0.0), x0 + 16.5)));
            const xi1: u8 = @intFromFloat(@min(@as(f32, 31.0), @max(@as(f32, 0.0), x1 + 16.5)));
            q.* = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
            // the 5-th bit, stored across the packed qh word
            qh |= @as(u32, (xi0 & 0x10) >> 4) << @intCast(j);
            qh |= @as(u32, (xi1 & 0x10) >> 4) << @intCast(j + q5_0_block_size / 2);
        }
        writeQh(&block.qh, qh);
    }
}

pub fn dequantizeRowQ5_0Into(dst: []f32, src: []const BlockQ5_0) !void {
    if (dst.len != try checkedProduct(src.len, q5_0_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const qh = readQh(&block.qh);
        const out = dst[block_index * q5_0_block_size ..][0..q5_0_block_size];
        for (block.qs, 0..) |q, j| {
            const xh0: u8 = @intCast(((qh >> @intCast(j)) << 4) & 0x10);
            const xh1: u8 = @intCast((qh >> @intCast(j + 12)) & 0x10);
            const x0: i32 = @as(i32, (q & 0x0f) | xh0) - 16;
            const x1: i32 = @as(i32, (q >> 4) | xh1) - 16;
            out[j] = @as(f32, @floatFromInt(x0)) * d;
            out[q5_0_block_size / 2 + j] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

/// f32 -> Q5_1 row encoder; faithful port of ggml's quantize_row_q5_1_ref
/// (byte-exact, see quant/encode_golden_test.zig). Assumes finite input;
/// degenerate-but-finite blocks (subnormal/overflowing spreads) produce
/// defined clamped output instead of @intFromFloat UB.
pub fn quantizeRowQ5_1Into(dst: []BlockQ5_1, src: []const f32) !void {
    const block_count = try q5_1BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    for (dst, 0..) |*block, block_index| {
        const row = src[block_index * q5_1_block_size ..][0..q5_1_block_size];
        var min: f32 = std.math.floatMax(f32);
        var max: f32 = -std.math.floatMax(f32);
        for (row) |v| {
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const d = (max - min) / 31.0;
        var inv_d: f32 = if (d != 0) 1.0 / d else 0.0;
        // Degenerate-but-finite blocks (subnormal spread -> 1/d overflows to
        // inf; max-min overflow -> d = inf) would feed NaN (0*inf / inf*0)
        // into @intFromFloat (UB in ReleaseFast). In-contract blocks have
        // finite inv_d, so goldens are unchanged.
        if (!std.math.isFinite(inv_d)) inv_d = 0;

        block.dm = .{ f32ToF16Bits(d), f32ToF16Bits(min) };

        var qh: u32 = 0;
        for (&block.qs, 0..) |*q, j| {
            const x0 = (row[j] - min) * inv_d;
            const x1 = (row[q5_1_block_size / 2 + j] - min) * inv_d;
            // ggml does not clamp here: in-contract x0/x1 are in [0, 31] by
            // construction, where the NaN-collapsing float-space clamp is
            // byte-identical; it only exists to keep degenerate NaN away
            // from @intFromFloat.
            const xi0: u8 = @intFromFloat(@min(@as(f32, 31.0), @max(@as(f32, 0.0), x0 + 0.5)));
            const xi1: u8 = @intFromFloat(@min(@as(f32, 31.0), @max(@as(f32, 0.0), x1 + 0.5)));
            q.* = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
            // the 5-th bit, stored across the packed qh word
            qh |= @as(u32, (xi0 & 0x10) >> 4) << @intCast(j);
            qh |= @as(u32, (xi1 & 0x10) >> 4) << @intCast(j + q5_1_block_size / 2);
        }
        writeQh(&block.qh, qh);
    }
}

pub fn dequantizeRowQ5_1Into(dst: []f32, src: []const BlockQ5_1) !void {
    if (dst.len != try checkedProduct(src.len, q5_1_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.dm[0]);
        const m = f16BitsToF32(block.dm[1]);
        const qh = readQh(&block.qh);
        const out = dst[block_index * q5_1_block_size ..][0..q5_1_block_size];
        for (block.qs, 0..) |q, j| {
            const xh0: u8 = @intCast(((qh >> @intCast(j)) << 4) & 0x10);
            const xh1: u8 = @intCast((qh >> @intCast(j + 12)) & 0x10);
            const x0: u8 = (q & 0x0f) | xh0;
            const x1: u8 = (q >> 4) | xh1;
            out[j] = @as(f32, @floatFromInt(x0)) * d + m;
            out[q5_1_block_size / 2 + j] = @as(f32, @floatFromInt(x1)) * d + m;
        }
    }
}

pub fn quantizeRowQ8_1Into(dst: []BlockQ8_1, src: []const f32) !void {
    const block_count = try q8_1BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    var block_index: usize = 0;
    while (block_index < block_count) : (block_index += 1) {
        const row = src[block_index * q8_1_block_size ..][0..q8_1_block_size];
        var amax: f32 = 0;
        for (row) |v| amax = @max(amax, @abs(v));

        const d = amax / 127.0;
        const inv_d: f32 = if (d == 0) 0 else 1.0 / d;
        var sum: i32 = 0;
        for (&dst[block_index].qs, row) |*q, v| {
            q.* = quantizeToI8(v * inv_d);
            sum += q.*;
        }
        dst[block_index].ds = .{ f32ToF16Bits(d), f32ToF16Bits(@as(f32, @floatFromInt(sum)) * d) };
    }
}

pub fn quantizeRowsQ8_1(allocator: Allocator, src: *const Tensor) !QuantizedRowsQ8_1 {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    const blocks_per_row = try q8_1BlockCount(cols);
    const data = try src.dataConstChecked();

    const blocks = try allocator.alloc(BlockQ8_1, try checkedProduct(rows, blocks_per_row));
    errdefer allocator.free(blocks);

    var row: usize = 0;
    while (row < rows) : (row += 1) {
        try quantizeRowQ8_1Into(
            blocks[row * blocks_per_row ..][0..blocks_per_row],
            data[row * cols ..][0..cols],
        );
    }

    return .{
        .allocator = allocator,
        .blocks = blocks,
        .rows = rows,
        .cols = cols,
        .blocks_per_row = blocks_per_row,
    };
}

pub fn dequantizeRowQ8_1Into(dst: []f32, src: []const BlockQ8_1) !void {
    if (dst.len != try checkedProduct(src.len, q8_1_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.ds[0]);
        const out = dst[block_index * q8_1_block_size ..][0..q8_1_block_size];
        for (out, block.qs) |*y, q| y.* = @as(f32, @floatFromInt(q)) * d;
    }
}

pub fn dequantizeRowMXFP4Into(dst: []f32, src: []const BlockMXFP4) !void {
    if (dst.len != try checkedProduct(src.len, mxfp4_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = e8m0ToF32Half(block.e);
        const out = dst[block_index * mxfp4_block_size ..][0..mxfp4_block_size];
        for (block.qs, 0..) |q, j| {
            out[j] = @as(f32, @floatFromInt(tables.kvalues_mxfp4[q & 0x0f])) * d;
            out[j + mxfp4_block_size / 2] = @as(f32, @floatFromInt(tables.kvalues_mxfp4[q >> 4])) * d;
        }
    }
}

pub fn dequantizeRowNVFP4Into(dst: []f32, src: []const BlockNVFP4) !void {
    if (dst.len != try checkedProduct(src.len, nvfp4_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const out = dst[block_index * nvfp4_block_size ..][0..nvfp4_block_size];
        for (0..nvfp4_block_size / nvfp4_subblock_size) |subblock| {
            const d = ue4m3ToF32(block.d[subblock]);
            const sub_out = out[subblock * nvfp4_subblock_size ..][0..nvfp4_subblock_size];
            const qs = block.qs[subblock * (nvfp4_subblock_size / 2) ..][0 .. nvfp4_subblock_size / 2];
            for (qs, 0..) |q, j| {
                sub_out[j] = @as(f32, @floatFromInt(tables.kvalues_mxfp4[q & 0x0f])) * d;
                sub_out[j + nvfp4_subblock_size / 2] = @as(f32, @floatFromInt(tables.kvalues_mxfp4[q >> 4])) * d;
            }
        }
    }
}

pub fn dequantizeRowTQ1_0Into(dst: []f32, src: []const BlockTQ1_0) !void {
    if (dst.len != try checkedProduct(src.len, qk_k_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    const pow3 = [_]u8{ 1, 3, 9, 27, 81, 243 };
    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var offset: usize = 0;
        const full_qs = block.qs.len - block.qs.len % 32;

        var j: usize = 0;
        while (j < full_qs) : (j += 32) {
            for (0..5) |n| {
                for (0..32) |m| {
                    const q = block.qs[j + m] *% pow3[n];
                    const xi: i16 = @intCast((@as(u16, q) * 3) >> 8);
                    out[offset] = @as(f32, @floatFromInt(xi - 1)) * d;
                    offset += 1;
                }
            }
        }

        while (j < block.qs.len) : (j += 16) {
            for (0..5) |n| {
                for (0..16) |m| {
                    const q = block.qs[j + m] *% pow3[n];
                    const xi: i16 = @intCast((@as(u16, q) * 3) >> 8);
                    out[offset] = @as(f32, @floatFromInt(xi - 1)) * d;
                    offset += 1;
                }
            }
        }

        for (0..4) |n| {
            for (block.qh) |qh| {
                const q = qh *% pow3[n];
                const xi: i16 = @intCast((@as(u16, q) * 3) >> 8);
                out[offset] = @as(f32, @floatFromInt(xi - 1)) * d;
                offset += 1;
            }
        }
        std.debug.assert(offset == qk_k_block_size);
    }
}

pub fn dequantizeRowTQ2_0Into(dst: []f32, src: []const BlockTQ2_0) !void {
    if (dst.len != try checkedProduct(src.len, qk_k_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var offset: usize = 0;
        var j: usize = 0;
        while (j < block.qs.len) : (j += 32) {
            for (0..4) |lane| {
                for (0..32) |m| {
                    const q: i32 = @intCast((block.qs[j + m] >> @intCast(lane * 2)) & 3);
                    out[offset] = @as(f32, @floatFromInt(q - 1)) * d;
                    offset += 1;
                }
            }
        }
        std.debug.assert(offset == qk_k_block_size);
    }
}

pub fn dequantizeRowIQ2_XXSInto(dst: []f32, src: []const BlockIQ2_XXS) !void {
    if (dst.len != try checkedProduct(src.len, qk_k_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var offset: usize = 0;
        for (0..qk_k_block_size / 32) |ib32| {
            const aux0 = readU32FromU16s(block.qs[4 * ib32 ..][0..2]);
            const aux1 = readU32FromU16s(block.qs[4 * ib32 + 2 ..][0..2]);
            const db = d * (0.5 + @as(f32, @floatFromInt(aux1 >> 28))) * 0.25;
            for (0..4) |lane| {
                const grid_index = byteFromU32(aux0, lane);
                const signs = tables.ksigns_iq2xs[(aux1 >> @intCast(7 * lane)) & 127];
                for (0..8) |j| {
                    const grid = gridU64Byte(&tables.iq2xxs_grid, grid_index, j);
                    const sign: f32 = if ((signs & tables.kmask_iq2xs[j]) != 0) -1 else 1;
                    out[offset + j] = db * @as(f32, @floatFromInt(grid)) * sign;
                }
                offset += 8;
            }
        }
    }
}

pub fn dequantizeRowIQ2_XSInto(dst: []f32, src: []const BlockIQ2_XS) !void {
    if (dst.len != try checkedProduct(src.len, qk_k_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var offset: usize = 0;
        for (0..qk_k_block_size / 32) |ib32| {
            const scales = block.scales[ib32];
            const db = [_]f32{
                d * (0.5 + @as(f32, @floatFromInt(scales & 0x0f))) * 0.25,
                d * (0.5 + @as(f32, @floatFromInt(scales >> 4))) * 0.25,
            };
            for (0..4) |lane| {
                const qword = block.qs[4 * ib32 + lane];
                const grid_index: usize = qword & 511;
                const signs = tables.ksigns_iq2xs[qword >> 9];
                for (0..8) |j| {
                    const grid = gridU64Byte(&tables.iq2xs_grid, grid_index, j);
                    const sign: f32 = if ((signs & tables.kmask_iq2xs[j]) != 0) -1 else 1;
                    out[offset + j] = db[lane / 2] * @as(f32, @floatFromInt(grid)) * sign;
                }
                offset += 8;
            }
        }
    }
}

pub fn dequantizeRowIQ2_SInto(dst: []f32, src: []const BlockIQ2_S) !void {
    if (dst.len != try checkedProduct(src.len, qk_k_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var qs_index: usize = 0;
        var signs_index: usize = qk_k_block_size / 8;
        var offset: usize = 0;
        for (0..qk_k_block_size / 32) |ib32| {
            const scales = block.scales[ib32];
            const db = [_]f32{
                d * (0.5 + @as(f32, @floatFromInt(scales & 0x0f))) * 0.25,
                d * (0.5 + @as(f32, @floatFromInt(scales >> 4))) * 0.25,
            };
            for (0..4) |lane| {
                const grid_index: usize = @as(usize, block.qs[qs_index + lane]) |
                    ((@as(usize, block.qh[ib32]) << @intCast(8 - 2 * lane)) & 0x300);
                const signs = block.qs[signs_index + lane];
                for (0..8) |j| {
                    const grid = gridU64Byte(&tables.iq2s_grid, grid_index, j);
                    const sign: f32 = if ((signs & tables.kmask_iq2xs[j]) != 0) -1 else 1;
                    out[offset + j] = db[lane / 2] * @as(f32, @floatFromInt(grid)) * sign;
                }
                offset += 8;
            }
            qs_index += 4;
            signs_index += 4;
        }
    }
}

pub fn dequantizeRowIQ3_XXSInto(dst: []f32, src: []const BlockIQ3_XXS) !void {
    if (dst.len != try checkedProduct(src.len, qk_k_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var qs_index: usize = 0;
        var offset: usize = 0;
        const scales_and_signs = qk_k_block_size / 4;
        for (0..qk_k_block_size / 32) |ib32| {
            const aux = readU32Bytes(block.qs[scales_and_signs + 4 * ib32 ..][0..4]);
            const db = d * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.5;
            for (0..4) |lane| {
                const signs = tables.ksigns_iq2xs[(aux >> @intCast(7 * lane)) & 127];
                const grid1 = block.qs[qs_index + 2 * lane];
                const grid2 = block.qs[qs_index + 2 * lane + 1];
                for (0..4) |j| {
                    const sign1: f32 = if ((signs & tables.kmask_iq2xs[j]) != 0) -1 else 1;
                    const sign2: f32 = if ((signs & tables.kmask_iq2xs[j + 4]) != 0) -1 else 1;
                    out[offset + j] = db * @as(f32, @floatFromInt(gridU32Byte(&tables.iq3xxs_grid, grid1, j))) * sign1;
                    out[offset + j + 4] = db * @as(f32, @floatFromInt(gridU32Byte(&tables.iq3xxs_grid, grid2, j))) * sign2;
                }
                offset += 8;
            }
            qs_index += 8;
        }
    }
}

pub fn dequantizeRowIQ3_SInto(dst: []f32, src: []const BlockIQ3_S) !void {
    if (dst.len != try checkedProduct(src.len, qk_k_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var qs_index: usize = 0;
        var qh_index: usize = 0;
        var signs_index: usize = 0;
        var offset: usize = 0;
        var ib32: usize = 0;
        while (ib32 < qk_k_block_size / 32) : (ib32 += 2) {
            const scale = block.scales[ib32 / 2];
            const db1 = d * @as(f32, @floatFromInt(1 + 2 * @as(u16, scale & 0x0f)));
            const db2 = d * @as(f32, @floatFromInt(1 + 2 * @as(u16, scale >> 4)));

            for (0..4) |lane| {
                const grid1: usize = @as(usize, block.qs[qs_index + 2 * lane]) | ((@as(usize, block.qh[qh_index]) << @intCast(8 - 2 * lane)) & 256);
                const grid2: usize = @as(usize, block.qs[qs_index + 2 * lane + 1]) | ((@as(usize, block.qh[qh_index]) << @intCast(7 - 2 * lane)) & 256);
                const signs = block.signs[signs_index + lane];
                for (0..4) |j| {
                    const sign1: f32 = if ((signs & tables.kmask_iq2xs[j]) != 0) -1 else 1;
                    const sign2: f32 = if ((signs & tables.kmask_iq2xs[j + 4]) != 0) -1 else 1;
                    out[offset + j] = db1 * @as(f32, @floatFromInt(gridU32Byte(&tables.iq3s_grid, grid1, j))) * sign1;
                    out[offset + j + 4] = db1 * @as(f32, @floatFromInt(gridU32Byte(&tables.iq3s_grid, grid2, j))) * sign2;
                }
                offset += 8;
            }
            qs_index += 8;
            signs_index += 4;

            for (0..4) |lane| {
                const grid1: usize = @as(usize, block.qs[qs_index + 2 * lane]) | ((@as(usize, block.qh[qh_index + 1]) << @intCast(8 - 2 * lane)) & 256);
                const grid2: usize = @as(usize, block.qs[qs_index + 2 * lane + 1]) | ((@as(usize, block.qh[qh_index + 1]) << @intCast(7 - 2 * lane)) & 256);
                const signs = block.signs[signs_index + lane];
                for (0..4) |j| {
                    const sign1: f32 = if ((signs & tables.kmask_iq2xs[j]) != 0) -1 else 1;
                    const sign2: f32 = if ((signs & tables.kmask_iq2xs[j + 4]) != 0) -1 else 1;
                    out[offset + j] = db2 * @as(f32, @floatFromInt(gridU32Byte(&tables.iq3s_grid, grid1, j))) * sign1;
                    out[offset + j + 4] = db2 * @as(f32, @floatFromInt(gridU32Byte(&tables.iq3s_grid, grid2, j))) * sign2;
                }
                offset += 8;
            }
            qh_index += 2;
            qs_index += 8;
            signs_index += 4;
        }
    }
}

pub fn dequantizeRowIQ1_SInto(dst: []f32, src: []const BlockIQ1_S) !void {
    if (dst.len != try checkedProduct(src.len, qk_k_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var qs_index: usize = 0;
        var offset: usize = 0;
        for (0..qk_k_block_size / 32) |ib| {
            const qh = block.qh[ib];
            const dl = d * @as(f32, @floatFromInt(2 * ((qh >> 12) & 7) + 1));
            const delta: f32 = if ((qh & 0x8000) != 0) -0.125 else 0.125;
            for (0..4) |lane| {
                const grid_index: usize = @as(usize, block.qs[qs_index + lane]) | @as(usize, ((qh >> @intCast(3 * lane)) & 7) << 8);
                for (0..8) |j| {
                    out[offset + j] = dl * (@as(f32, @floatFromInt(gridI8Byte(&tables.iq1s_grid, grid_index, j))) + delta);
                }
                offset += 8;
            }
            qs_index += 4;
        }
    }
}

pub fn dequantizeRowIQ1_MInto(dst: []f32, src: []const BlockIQ1_M) !void {
    if (dst.len != try checkedProduct(src.len, qk_k_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const sc0 = readU16Bytes(block.scales[0..2]);
        const sc1 = readU16Bytes(block.scales[2..4]);
        const sc2 = readU16Bytes(block.scales[4..6]);
        const sc3 = readU16Bytes(block.scales[6..8]);
        const scale_bits: u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
        const d = f16BitsToF32(scale_bits);
        const out = dst[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var qs_index: usize = 0;
        var qh_index: usize = 0;
        var offset: usize = 0;

        for (0..qk_k_block_size / 32) |ib| {
            const sc = readU16Bytes(block.scales[2 * (ib / 2) ..][0..2]);
            const dl1 = d * @as(f32, @floatFromInt(2 * ((sc >> @intCast(6 * (ib % 2))) & 0x7) + 1));
            const dl2 = d * @as(f32, @floatFromInt(2 * ((sc >> @intCast(6 * (ib % 2) + 3)) & 0x7) + 1));
            const qh0 = block.qh[qh_index];
            const qh1 = block.qh[qh_index + 1];
            const idx = [_]usize{
                @as(usize, block.qs[qs_index + 0]) | ((@as(usize, qh0) << 8) & 0x700),
                @as(usize, block.qs[qs_index + 1]) | ((@as(usize, qh0) << 4) & 0x700),
                @as(usize, block.qs[qs_index + 2]) | ((@as(usize, qh1) << 8) & 0x700),
                @as(usize, block.qs[qs_index + 3]) | ((@as(usize, qh1) << 4) & 0x700),
            };
            const delta = [_]f32{
                if ((qh0 & 0x08) != 0) -0.125 else 0.125,
                if ((qh0 & 0x80) != 0) -0.125 else 0.125,
                if ((qh1 & 0x08) != 0) -0.125 else 0.125,
                if ((qh1 & 0x80) != 0) -0.125 else 0.125,
            };
            for (0..2) |lane| {
                for (0..8) |j| {
                    out[offset + j] = dl1 * (@as(f32, @floatFromInt(gridI8Byte(&tables.iq1s_grid, idx[lane], j))) + delta[lane]);
                }
                offset += 8;
            }
            for (2..4) |lane| {
                for (0..8) |j| {
                    out[offset + j] = dl2 * (@as(f32, @floatFromInt(gridI8Byte(&tables.iq1s_grid, idx[lane], j))) + delta[lane]);
                }
                offset += 8;
            }
            qs_index += 4;
            qh_index += 2;
        }
    }
}

pub fn dequantizeRowIQ4_NLInto(dst: []f32, src: []const BlockIQ4_NL) !void {
    if (dst.len != try checkedProduct(src.len, iq4_nl_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * iq4_nl_block_size ..][0..iq4_nl_block_size];
        for (block.qs, 0..) |q, j| {
            out[j] = d * @as(f32, @floatFromInt(tables.kvalues_iq4nl[q & 0x0f]));
            out[j + iq4_nl_block_size / 2] = d * @as(f32, @floatFromInt(tables.kvalues_iq4nl[q >> 4]));
        }
    }
}

pub fn dequantizeRowIQ4_XSInto(dst: []f32, src: []const BlockIQ4_XS) !void {
    if (dst.len != try checkedProduct(src.len, qk_k_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var qs_index: usize = 0;
        for (0..qk_k_block_size / 32) |ib| {
            const low = (block.scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0x0f;
            const high = ((block.scales_h >> @intCast(2 * ib)) & 3) << 4;
            const ls: i32 = @intCast(@as(u16, low) | high);
            const dl = d * @as(f32, @floatFromInt(ls - 32));
            const sub_out = out[ib * 32 ..][0..32];
            for (0..16) |j| {
                const q = block.qs[qs_index + j];
                sub_out[j] = dl * @as(f32, @floatFromInt(tables.kvalues_iq4nl[q & 0x0f]));
                sub_out[j + 16] = dl * @as(f32, @floatFromInt(tables.kvalues_iq4nl[q >> 4]));
            }
            qs_index += 16;
        }
    }
}

fn e8m0ToF32Half(x: u8) f32 {
    const bits: u32 = if (x < 2)
        @as(u32, 0x00200000) << @intCast(x)
    else
        @as(u32, x - 1) << 23;
    return @bitCast(bits);
}

fn ue4m3ToF32(x: u8) f32 {
    if (x == 0 or x == 0x7f) return 0;
    const exp: i32 = @intCast((x >> 3) & 0x0f);
    const man: u8 = x & 0x07;
    if (exp == 0) return @as(f32, @floatFromInt(man)) * std.math.pow(f32, 2, -10);
    return (1.0 + @as(f32, @floatFromInt(man)) / 8.0) * std.math.pow(f32, 2, @floatFromInt(exp - 8));
}

fn readU32FromU16s(values: *const [2]u16) u32 {
    return @as(u32, values[0]) | (@as(u32, values[1]) << 16);
}

fn readU32Bytes(values: *const [4]u8) u32 {
    return @as(u32, values[0]) |
        (@as(u32, values[1]) << 8) |
        (@as(u32, values[2]) << 16) |
        (@as(u32, values[3]) << 24);
}

fn readU16Bytes(values: *const [2]u8) u16 {
    return @as(u16, values[0]) | (@as(u16, values[1]) << 8);
}

fn byteFromU32(value: u32, index: usize) u8 {
    return @intCast((value >> @intCast(index * 8)) & 0xff);
}

fn gridU64Byte(comptime table: []const u64, index: usize, byte_index: usize) u8 {
    return @intCast((table[index] >> @intCast(byte_index * 8)) & 0xff);
}

fn gridU32Byte(comptime table: []const u32, index: usize, byte_index: usize) u8 {
    return @intCast((table[index] >> @intCast(byte_index * 8)) & 0xff);
}

fn gridI8Byte(comptime table: []const u64, index: usize, byte_index: usize) i8 {
    return @bitCast(gridU64Byte(table, index, byte_index));
}

fn readQh(qh: *const [4]u8) u32 {
    return std.mem.readInt(u32, qh, .little);
}

fn writeQh(qh: []u8, value: u32) void {
    std.debug.assert(qh.len == 4);
    qh[0] = @intCast(value & 0xff);
    qh[1] = @intCast((value >> 8) & 0xff);
    qh[2] = @intCast((value >> 16) & 0xff);
    qh[3] = @intCast((value >> 24) & 0xff);
}

pub fn matmulQ4_0RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ4_0,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.rows.blocks_per_row;
    var i = r0;
    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;

        while (j + q4_0_col_block <= c1) : (j += q4_0_col_block) {
            var acc = [_]f32{0} ** q4_0_col_block;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const a_block = prepareQ8_0Block(&lhs_row[block_index]);
                inline for (0..q4_0_col_block) |c| {
                    const rhs_block = &rhs.rows.blocks[(j + c) * blocks_per_row + block_index];
                    acc[c] += dotQ4_0PreparedQ8_0(rhs_block, a_block);
                }
            }
            inline for (0..q4_0_col_block) |c| out[i * n + j + c] = acc[c];
        }

        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc += dotQ4_0Q8_0(&rhs_col[block_index], &lhs_row[block_index]);
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn matmulQ4_0RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ4_0,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ4_0RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ1_0RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ1_0,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.rows.blocks_per_row;
    const lhs_blocks_per_row = blocks_per_row * (q1_0_block_size / q8_0_block_size);
    var i = r0;
    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * lhs_blocks_per_row ..][0..lhs_blocks_per_row];
        var j = c0;
        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const lhs_base = block_index * (q1_0_block_size / q8_0_block_size);
                acc += dotQ1_0Q8_0(&rhs_col[block_index], lhs_row[lhs_base..][0..4]);
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn matmulQ1_0RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ1_0,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ1_0RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ4_1RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_1,
    rhs: *const QuantizedMatmulRhsQ4_1,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.rows.blocks_per_row;
    var i = r0;
    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;
        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc += dotQ4_1Q8_1(&rhs_col[block_index], &lhs_row[block_index]);
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn matmulQ4_1RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_1,
    rhs: *const QuantizedMatmulRhsQ4_1,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ4_1RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ5_0RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ5_0,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.rows.blocks_per_row;
    var i = r0;
    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;
        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc += dotQ5_0Q8_0(&rhs_col[block_index], &lhs_row[block_index]);
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn matmulQ5_0RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ5_0,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ5_0RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ5_1RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_1,
    rhs: *const QuantizedMatmulRhsQ5_1,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.rows.blocks_per_row;
    var i = r0;
    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;
        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc += dotQ5_1Q8_1(&rhs_col[block_index], &lhs_row[block_index]);
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn matmulQ5_1RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_1,
    rhs: *const QuantizedMatmulRhsQ5_1,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ5_1RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ2_KRhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ2_K,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.blocks_per_column;
    var i = r0;
    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;

        while (j + qk_col_block <= c1) : (j += qk_col_block) {
            var acc = [_]f32{0} ** qk_col_block;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const lhs_block = &lhs_row[block_index];
                inline for (0..qk_col_block) |c| {
                    const rhs_block = &rhs.blocks[(j + c) * blocks_per_row + block_index];
                    acc[c] += dotQ2_KQ8_K(rhs_block, lhs_block);
                }
            }
            inline for (0..qk_col_block) |c| out[i * n + j + c] = acc[c];
        }

        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc += dotQ2_KQ8_K(&rhs_col[block_index], &lhs_row[block_index]);
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn matmulQ2_KRhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ2_K,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ2_KRhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ3_KRhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ3_K,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.blocks_per_column;
    var i = r0;
    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;

        while (j + qk_col_block <= c1) : (j += qk_col_block) {
            var acc = [_]f32{0} ** qk_col_block;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const lhs_block = &lhs_row[block_index];
                inline for (0..qk_col_block) |c| {
                    const rhs_block = &rhs.blocks[(j + c) * blocks_per_row + block_index];
                    acc[c] += dotQ3_KQ8_K(rhs_block, lhs_block);
                }
            }
            inline for (0..qk_col_block) |c| out[i * n + j + c] = acc[c];
        }

        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc += dotQ3_KQ8_K(&rhs_col[block_index], &lhs_row[block_index]);
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn matmulQ3_KRhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ3_K,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ3_KRhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulTableQ8_0RhsRange(
    comptime rhs_dtype: DType,
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsRowsFor(rhs_dtype),
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulTableQ8_0RhsTile(rhs_dtype, out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulTableQ8_0RhsTile(
    comptime rhs_dtype: DType,
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsRowsFor(rhs_dtype),
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const rhs_block_size = dtype_mod.blockSize(rhs_dtype);
    const lhs_blocks_per_rhs_block = rhs_block_size / q8_0_block_size;
    const blocks_per_row = rhs.rows.blocks_per_row;
    const lhs_blocks_per_row = blocks_per_row * lhs_blocks_per_rhs_block;

    var row = r0;
    while (row < r1) : (row += 1) {
        const lhs_row = lhs_blocks[row * lhs_blocks_per_row ..][0..lhs_blocks_per_row];
        var col = c0;

        while (col + table_q8_0_col_block <= c1) : (col += table_q8_0_col_block) {
            var acc = [_]f32{0} ** table_q8_0_col_block;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const lhs_offset = block_index * lhs_blocks_per_rhs_block;
                const lhs_block = lhs_row[lhs_offset..][0..lhs_blocks_per_rhs_block];
                inline for (0..table_q8_0_col_block) |c| {
                    const rhs_block = &rhs.rows.blocks[(col + c) * blocks_per_row + block_index];
                    acc[c] += dotTableQ8_0(rhs_dtype, rhs_block, lhs_block);
                }
            }
            inline for (0..table_q8_0_col_block) |c| out[row * n + col + c] = acc[c];
        }

        while (col < c1) : (col += 1) {
            const rhs_col = rhs.columnBlocks(col);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const lhs_offset = block_index * lhs_blocks_per_rhs_block;
                acc += dotTableQ8_0(rhs_dtype, &rhs_col[block_index], lhs_row[lhs_offset..][0..lhs_blocks_per_rhs_block]);
            }
            out[row * n + col] = acc;
        }
    }
}

pub fn matmulTableQ8_KRhsRange(
    comptime rhs_dtype: DType,
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsRowsFor(rhs_dtype),
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulTableQ8_KRhsTile(rhs_dtype, out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulTableQ8_KRhsTile(
    comptime rhs_dtype: DType,
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsRowsFor(rhs_dtype),
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.rows.blocks_per_row;

    var row = r0;
    while (row < r1) : (row += 1) {
        const lhs_row = lhs_blocks[row * blocks_per_row ..][0..blocks_per_row];
        var col = c0;

        while (col + table_q8_k_col_block <= c1) : (col += table_q8_k_col_block) {
            var acc = [_]f32{0} ** table_q8_k_col_block;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const lhs_block = &lhs_row[block_index];
                inline for (0..table_q8_k_col_block) |c| {
                    const rhs_block = &rhs.rows.blocks[(col + c) * blocks_per_row + block_index];
                    acc[c] += dotTableQ8_K(rhs_dtype, rhs_block, lhs_block);
                }
            }
            inline for (0..table_q8_k_col_block) |c| out[row * n + col + c] = acc[c];
        }

        while (col < c1) : (col += 1) {
            const rhs_col = rhs.columnBlocks(col);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc += dotTableQ8_K(rhs_dtype, &rhs_col[block_index], &lhs_row[block_index]);
            }
            out[row * n + col] = acc;
        }
    }
}

fn dotTableQ8_0(comptime rhs_dtype: DType, w: *const dtype_mod.Storage(rhs_dtype), a: []const BlockQ8_0) f32 {
    return switch (rhs_dtype) {
        .iq4_nl => dotIQ4_NLQ8_0(w, a),
        .mxfp4 => dotMXFP4Q8_0(w, a),
        .nvfp4 => dotNVFP4Q8_0(w, a),
        else => @compileError("dtype does not use Q8_0 table dot matmul"),
    };
}

fn dotTableQ8_K(comptime rhs_dtype: DType, w: *const dtype_mod.Storage(rhs_dtype), a: *const BlockQ8_K) f32 {
    return switch (rhs_dtype) {
        .iq1_s => dotIQ1_SQ8_K(w, a),
        .iq1_m => dotIQ1_MQ8_K(w, a),
        .iq2_xxs => dotIQ2_XXSQ8_K(w, a),
        .iq2_xs => dotIQ2_XSQ8_K(w, a),
        .iq2_s => dotIQ2_SQ8_K(w, a),
        .iq3_xxs => dotIQ3_XXSQ8_K(w, a),
        .iq3_s => dotIQ3_SQ8_K(w, a),
        .iq4_xs => dotIQ4_XSQ8_K(w, a),
        .tq1_0 => dotTQ1_0Q8_K(w, a),
        .tq2_0 => dotTQ2_0Q8_K(w, a),
        else => @compileError("dtype does not use Q8_K table dot matmul"),
    };
}

fn dotIQ4_NLQ8_0(w: *const BlockIQ4_NL, a: []const BlockQ8_0) f32 {
    std.debug.assert(a.len == 1);
    const ab = &a[0];
    const isum = dotNibbleTable32Q8(&tables.kvalues_iq4nl, &w.qs, ab.qs[0..16], ab.qs[16..32]);
    return f16BitsToF32(w.d) * f16BitsToF32(ab.d) * @as(f32, @floatFromInt(isum));
}

fn dotMXFP4Q8_0(w: *const BlockMXFP4, a: []const BlockQ8_0) f32 {
    std.debug.assert(a.len == 1);
    const ab = &a[0];
    const isum = dotNibbleTable32Q8(&tables.kvalues_mxfp4, &w.qs, ab.qs[0..16], ab.qs[16..32]);
    return e8m0ToF32Half(w.e) * f16BitsToF32(ab.d) * @as(f32, @floatFromInt(isum));
}

fn dotNVFP4Q8_0(w: *const BlockNVFP4, a: []const BlockQ8_0) f32 {
    std.debug.assert(a.len == nvfp4_block_size / q8_0_block_size);
    var sum: f32 = 0;
    for (0..nvfp4_block_size / nvfp4_subblock_size) |subblock| {
        const ab = &a[(subblock * nvfp4_subblock_size) / q8_0_block_size];
        const a_offset = (subblock * nvfp4_subblock_size) % q8_0_block_size;
        const qs = w.qs[subblock * (nvfp4_subblock_size / 2) ..][0 .. nvfp4_subblock_size / 2];
        const isum = dotNibbleTable16Q8(&tables.kvalues_mxfp4, qs, ab.qs[a_offset..][0..nvfp4_subblock_size]);
        sum += ue4m3ToF32(w.d[subblock]) * f16BitsToF32(ab.d) * @as(f32, @floatFromInt(isum));
    }
    return sum;
}

fn dotIQ2_XXSQ8_K(w: *const BlockIQ2_XXS, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.d) * a.d;
    var sum: f32 = 0;
    var offset: usize = 0;
    for (0..qk_k_block_size / 32) |ib32| {
        const aux0 = readU32FromU16s(w.qs[4 * ib32 ..][0..2]);
        const aux1 = readU32FromU16s(w.qs[4 * ib32 + 2 ..][0..2]);
        const db = d * (0.5 + @as(f32, @floatFromInt(aux1 >> 28))) * 0.25;
        var isum: i32 = 0;
        for (0..4) |lane| {
            const grid_index = byteFromU32(aux0, lane);
            const signs = tables.ksigns_iq2xs[(aux1 >> @intCast(7 * lane)) & 127];
            isum += dotIQ2GridLaneQ8_K(&tables.iq2xxs_grid, grid_index, signs, a, offset);
            offset += 8;
        }
        sum += db * @as(f32, @floatFromInt(isum));
    }
    return sum;
}

fn dotIQ2_XSQ8_K(w: *const BlockIQ2_XS, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.d) * a.d;
    var sum: f32 = 0;
    var offset: usize = 0;
    for (0..qk_k_block_size / 32) |ib32| {
        const scales = w.scales[ib32];
        const db = [_]f32{
            d * (0.5 + @as(f32, @floatFromInt(scales & 0x0f))) * 0.25,
            d * (0.5 + @as(f32, @floatFromInt(scales >> 4))) * 0.25,
        };
        for (0..4) |lane| {
            const qword = w.qs[4 * ib32 + lane];
            const grid_index: usize = qword & 511;
            const signs = tables.ksigns_iq2xs[qword >> 9];
            const isum = dotIQ2GridLaneQ8_K(&tables.iq2xs_grid, grid_index, signs, a, offset);
            sum += db[lane / 2] * @as(f32, @floatFromInt(isum));
            offset += 8;
        }
    }
    return sum;
}

fn dotIQ2_SQ8_K(w: *const BlockIQ2_S, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.d) * a.d;
    var sum: f32 = 0;
    var qs_index: usize = 0;
    var signs_index: usize = qk_k_block_size / 8;
    var offset: usize = 0;
    for (0..qk_k_block_size / 32) |ib32| {
        const scales = w.scales[ib32];
        const db = [_]f32{
            d * (0.5 + @as(f32, @floatFromInt(scales & 0x0f))) * 0.25,
            d * (0.5 + @as(f32, @floatFromInt(scales >> 4))) * 0.25,
        };
        for (0..4) |lane| {
            const grid_index: usize = @as(usize, w.qs[qs_index + lane]) |
                ((@as(usize, w.qh[ib32]) << @intCast(8 - 2 * lane)) & 0x300);
            const signs = w.qs[signs_index + lane];
            const isum = dotIQ2GridLaneQ8_K(&tables.iq2s_grid, grid_index, signs, a, offset);
            sum += db[lane / 2] * @as(f32, @floatFromInt(isum));
            offset += 8;
        }
        qs_index += 4;
        signs_index += 4;
    }
    return sum;
}

fn dotIQ2GridLaneQ8_K(comptime table: []const u64, grid_index: usize, signs: u8, a: *const BlockQ8_K, offset: usize) i32 {
    const grid_i16: QKV8i16 = @intCast(gridU64Vector(table, grid_index));
    return dotI16I8x8(grid_i16 * signsVector8(signs), a.qs[offset..][0..8]);
}

fn dotIQ3_XXSQ8_K(w: *const BlockIQ3_XXS, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.d) * a.d;
    var sum: f32 = 0;
    var qs_index: usize = 0;
    var offset: usize = 0;
    const scales_and_signs = qk_k_block_size / 4;
    for (0..qk_k_block_size / 32) |ib32| {
        const aux = readU32Bytes(w.qs[scales_and_signs + 4 * ib32 ..][0..4]);
        const db = d * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.5;
        var isum: i32 = 0;
        for (0..4) |lane| {
            const signs = tables.ksigns_iq2xs[(aux >> @intCast(7 * lane)) & 127];
            isum += dotIQ3GridLaneQ8_K(&tables.iq3xxs_grid, w.qs[qs_index + 2 * lane], w.qs[qs_index + 2 * lane + 1], signs, a, offset);
            offset += 8;
        }
        sum += db * @as(f32, @floatFromInt(isum));
        qs_index += 8;
    }
    return sum;
}

fn dotIQ3_SQ8_K(w: *const BlockIQ3_S, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.d) * a.d;
    var sum: f32 = 0;
    var qs_index: usize = 0;
    var qh_index: usize = 0;
    var signs_index: usize = 0;
    var offset: usize = 0;
    var ib32: usize = 0;
    while (ib32 < qk_k_block_size / 32) : (ib32 += 2) {
        const scale = w.scales[ib32 / 2];
        const db1 = d * @as(f32, @floatFromInt(1 + 2 * @as(u16, scale & 0x0f)));
        const db2 = d * @as(f32, @floatFromInt(1 + 2 * @as(u16, scale >> 4)));

        var isum1: i32 = 0;
        for (0..4) |lane| {
            const grid1: usize = @as(usize, w.qs[qs_index + 2 * lane]) | ((@as(usize, w.qh[qh_index]) << @intCast(8 - 2 * lane)) & 256);
            const grid2: usize = @as(usize, w.qs[qs_index + 2 * lane + 1]) | ((@as(usize, w.qh[qh_index]) << @intCast(7 - 2 * lane)) & 256);
            isum1 += dotIQ3GridLaneQ8_K(&tables.iq3s_grid, grid1, grid2, w.signs[signs_index + lane], a, offset);
            offset += 8;
        }
        sum += db1 * @as(f32, @floatFromInt(isum1));
        qs_index += 8;
        signs_index += 4;

        var isum2: i32 = 0;
        for (0..4) |lane| {
            const grid1: usize = @as(usize, w.qs[qs_index + 2 * lane]) | ((@as(usize, w.qh[qh_index + 1]) << @intCast(8 - 2 * lane)) & 256);
            const grid2: usize = @as(usize, w.qs[qs_index + 2 * lane + 1]) | ((@as(usize, w.qh[qh_index + 1]) << @intCast(7 - 2 * lane)) & 256);
            isum2 += dotIQ3GridLaneQ8_K(&tables.iq3s_grid, grid1, grid2, w.signs[signs_index + lane], a, offset);
            offset += 8;
        }
        sum += db2 * @as(f32, @floatFromInt(isum2));
        qh_index += 2;
        qs_index += 8;
        signs_index += 4;
    }
    return sum;
}

fn dotIQ3GridLaneQ8_K(comptime table: []const u32, grid1: usize, grid2: usize, signs: u8, a: *const BlockQ8_K, offset: usize) i32 {
    const grid_i16: QKV8i16 = @intCast(gridU32PairVector(table, grid1, grid2));
    return dotI16I8x8(grid_i16 * signsVector8(signs), a.qs[offset..][0..8]);
}

fn dotIQ1_SQ8_K(w: *const BlockIQ1_S, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.d) * a.d;
    var sum: f32 = 0;
    var qs_index: usize = 0;
    var offset: usize = 0;
    for (0..qk_k_block_size / 32) |ib| {
        const qh = w.qh[ib];
        const dl = d * @as(f32, @floatFromInt(2 * ((qh >> 12) & 7) + 1));
        const delta: f32 = if ((qh & 0x8000) != 0) -0.125 else 0.125;
        var isum: i32 = 0;
        var qsum: i32 = 0;
        for (0..4) |lane| {
            const grid_index: usize = @as(usize, w.qs[qs_index + lane]) | @as(usize, ((qh >> @intCast(3 * lane)) & 7) << 8);
            const lane_dot = dotIQ1GridLaneQ8_K(grid_index, a, offset);
            isum += lane_dot.grid;
            qsum += lane_dot.q;
            offset += 8;
        }
        sum += dl * (@as(f32, @floatFromInt(isum)) + delta * @as(f32, @floatFromInt(qsum)));
        qs_index += 4;
    }
    return sum;
}

fn dotIQ1_MQ8_K(w: *const BlockIQ1_M, a: *const BlockQ8_K) f32 {
    const sc0 = readU16Bytes(w.scales[0..2]);
    const sc1 = readU16Bytes(w.scales[2..4]);
    const sc2 = readU16Bytes(w.scales[4..6]);
    const sc3 = readU16Bytes(w.scales[6..8]);
    const scale_bits: u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
    const d = f16BitsToF32(scale_bits) * a.d;
    var sum: f32 = 0;
    var qs_index: usize = 0;
    var qh_index: usize = 0;
    var offset: usize = 0;

    for (0..qk_k_block_size / 32) |ib| {
        const sc = readU16Bytes(w.scales[2 * (ib / 2) ..][0..2]);
        const dl1 = d * @as(f32, @floatFromInt(2 * ((sc >> @intCast(6 * (ib % 2))) & 0x7) + 1));
        const dl2 = d * @as(f32, @floatFromInt(2 * ((sc >> @intCast(6 * (ib % 2) + 3)) & 0x7) + 1));
        const qh0 = w.qh[qh_index];
        const qh1 = w.qh[qh_index + 1];
        const idx = [_]usize{
            @as(usize, w.qs[qs_index + 0]) | ((@as(usize, qh0) << 8) & 0x700),
            @as(usize, w.qs[qs_index + 1]) | ((@as(usize, qh0) << 4) & 0x700),
            @as(usize, w.qs[qs_index + 2]) | ((@as(usize, qh1) << 8) & 0x700),
            @as(usize, w.qs[qs_index + 3]) | ((@as(usize, qh1) << 4) & 0x700),
        };
        const delta = [_]f32{
            if ((qh0 & 0x08) != 0) -0.125 else 0.125,
            if ((qh0 & 0x80) != 0) -0.125 else 0.125,
            if ((qh1 & 0x08) != 0) -0.125 else 0.125,
            if ((qh1 & 0x80) != 0) -0.125 else 0.125,
        };

        for (0..2) |lane| {
            const lane_dot = dotIQ1GridLaneQ8_K(idx[lane], a, offset);
            sum += dl1 * (@as(f32, @floatFromInt(lane_dot.grid)) + delta[lane] * @as(f32, @floatFromInt(lane_dot.q)));
            offset += 8;
        }
        for (2..4) |lane| {
            const lane_dot = dotIQ1GridLaneQ8_K(idx[lane], a, offset);
            sum += dl2 * (@as(f32, @floatFromInt(lane_dot.grid)) + delta[lane] * @as(f32, @floatFromInt(lane_dot.q)));
            offset += 8;
        }
        qs_index += 4;
        qh_index += 2;
    }
    return sum;
}

const IQ1LaneDot = struct {
    grid: i32,
    q: i32,
};

fn dotIQ1GridLaneQ8_K(grid_index: usize, a: *const BlockQ8_K, offset: usize) IQ1LaneDot {
    const q_i8: QKV8i8 = @bitCast(a.qs[offset..][0..8].*);
    const q_i16: QKV8i16 = @intCast(q_i8);
    return .{
        .grid = dotI16I16x8(gridI8Vector(&tables.iq1s_grid, grid_index), q_i16),
        .q = reduceI16x8(q_i16),
    };
}

fn dotIQ4_XSQ8_K(w: *const BlockIQ4_XS, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.d) * a.d;
    var sum: f32 = 0;
    var qs_index: usize = 0;
    for (0..qk_k_block_size / 32) |ib| {
        const low = (w.scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0x0f;
        const high = ((w.scales_h >> @intCast(2 * ib)) & 3) << 4;
        const ls: i32 = @intCast(@as(u16, low) | high);
        const isum = dotNibbleTable32Q8(&tables.kvalues_iq4nl, w.qs[qs_index..][0..16], a.qs[ib * 32 ..][0..16], a.qs[ib * 32 + 16 ..][0..16]);
        sum += d * @as(f32, @floatFromInt(ls - 32)) * @as(f32, @floatFromInt(isum));
        qs_index += 16;
    }
    return sum;
}

fn dotTQ1_0Q8_K(w: *const BlockTQ1_0, a: *const BlockQ8_K) f32 {
    const pow3 = [_]u8{ 1, 3, 9, 27, 81, 243 };
    var offset: usize = 0;
    var isum: i32 = 0;
    const full_qs = w.qs.len - w.qs.len % 32;

    var j: usize = 0;
    while (j < full_qs) : (j += 32) {
        inline for (0..5) |n| {
            isum += dotTernaryLane16(w.qs[j..][0..16], a.qs[offset..][0..16], pow3[n]);
            offset += 16;
            isum += dotTernaryLane16(w.qs[j + 16 ..][0..16], a.qs[offset..][0..16], pow3[n]);
            offset += 16;
        }
    }

    while (j < w.qs.len) : (j += 16) {
        inline for (0..5) |n| {
            isum += dotTernaryLane16(w.qs[j..][0..16], a.qs[offset..][0..16], pow3[n]);
            offset += 16;
        }
    }

    for (0..4) |n| {
        for (w.qh) |qh| {
            isum += ternaryValue(qh, pow3[n]) * @as(i32, @intCast(a.qs[offset]));
            offset += 1;
        }
    }
    std.debug.assert(offset == qk_k_block_size);
    return f16BitsToF32(w.d) * a.d * @as(f32, @floatFromInt(isum));
}

fn dotTQ2_0Q8_K(w: *const BlockTQ2_0, a: *const BlockQ8_K) f32 {
    var isum: i32 = 0;
    var j: usize = 0;
    var offset: usize = 0;
    while (j < w.qs.len) : (j += 32) {
        inline for (0..4) |lane| {
            inline for (0..2) |chunk| {
                isum += dotTQ2Lane16(w.qs[j + chunk * 16 ..][0..16], a.qs[offset + chunk * 16 ..][0..16], lane);
            }
            offset += 32;
        }
    }
    std.debug.assert(offset == qk_k_block_size);
    return f16BitsToF32(w.d) * a.d * @as(f32, @floatFromInt(isum));
}

fn ternaryValue(qs: u8, pow3: u8) i32 {
    const q = qs *% pow3;
    const xi: i16 = @intCast((@as(u16, q) * 3) >> 8);
    return @intCast(xi - 1);
}

fn dotQ1_0Q8_0(w: *const BlockQ1_0, a: []const BlockQ8_0) f32 {
    std.debug.assert(a.len == q1_0_block_size / q8_0_block_size);
    const d0 = f16BitsToF32(w.d);
    var sum: f32 = 0;
    for (a, 0..) |*ab, block_index| {
        var isum: i32 = 0;
        const bits = w.qs[block_index * 4 ..][0..4];
        var offset: usize = 0;
        for (bits) |mask| {
            inline for (0..8) |bit| {
                const q = @as(i32, ab.qs[offset + bit]);
                isum += if ((mask & (@as(u8, 1) << bit)) != 0) q else -q;
            }
            offset += 8;
        }
        sum += d0 * f16BitsToF32(ab.d) * @as(f32, @floatFromInt(isum));
    }
    return sum;
}

fn dotQ4_1Q8_1(w: *const BlockQ4_1, a: *const BlockQ8_1) f32 {
    var isum: i32 = 0;
    for (w.qs, 0..) |q, j| {
        isum += @as(i32, q & 0x0f) * @as(i32, a.qs[j]);
        isum += @as(i32, q >> 4) * @as(i32, a.qs[q4_1_block_size / 2 + j]);
    }
    const d = f16BitsToF32(w.dm[0]) * f16BitsToF32(a.ds[0]);
    const m = f16BitsToF32(w.dm[1]) * f16BitsToF32(a.ds[1]);
    return d * @as(f32, @floatFromInt(isum)) + m;
}

fn dotQ5_0Q8_0(w: *const BlockQ5_0, a: *const BlockQ8_0) f32 {
    const qh = readQh(&w.qh);
    var isum: i32 = 0;
    for (w.qs, 0..) |q, j| {
        const xh0: u8 = @intCast(((qh >> @intCast(j)) << 4) & 0x10);
        const xh1: u8 = @intCast((qh >> @intCast(j + 12)) & 0x10);
        const x0: i32 = @as(i32, (q & 0x0f) | xh0) - 16;
        const x1: i32 = @as(i32, (q >> 4) | xh1) - 16;
        isum += x0 * @as(i32, a.qs[j]);
        isum += x1 * @as(i32, a.qs[q5_0_block_size / 2 + j]);
    }
    const d = f16BitsToF32(w.d) * f16BitsToF32(a.d);
    return d * @as(f32, @floatFromInt(isum));
}

fn dotQ5_1Q8_1(w: *const BlockQ5_1, a: *const BlockQ8_1) f32 {
    const qh = readQh(&w.qh);
    var isum: i32 = 0;
    for (w.qs, 0..) |q, j| {
        const xh0: u8 = @intCast(((qh >> @intCast(j)) << 4) & 0x10);
        const xh1: u8 = @intCast((qh >> @intCast(j + 12)) & 0x10);
        const x0: u8 = (q & 0x0f) | xh0;
        const x1: u8 = (q >> 4) | xh1;
        isum += @as(i32, x0) * @as(i32, a.qs[j]);
        isum += @as(i32, x1) * @as(i32, a.qs[q5_1_block_size / 2 + j]);
    }
    const d = f16BitsToF32(w.dm[0]) * f16BitsToF32(a.ds[0]);
    const m = f16BitsToF32(w.dm[1]) * f16BitsToF32(a.ds[1]);
    return d * @as(f32, @floatFromInt(isum)) + m;
}

// Reuses each Q8 activation block across a few quantized columns without the
// register pressure observed with wider portable tiles.

const q4_0_col_block: usize = 2;

const table_q8_0_col_block: usize = 2;

const table_q8_k_col_block: usize = 2;

const qk_v8_sign_masks: QKV8u8 = .{ 1, 2, 4, 8, 16, 32, 64, 128 };

const NibbleTableVectors = struct {
    lo: QKV16i16,
    hi: QKV16i16,
};

fn dotI16I8x16(w: QKV16i16, a_qs: *const [16]i8) i32 {
    const a_i8: QKV16i8 = @bitCast(a_qs.*);
    const a_i16: QKV16i16 = @intCast(a_i8);
    return dotI16I16x16(w, a_i16);
}

fn dotI16I8x8(w: QKV8i16, a_qs: *const [8]i8) i32 {
    const a_i8: QKV8i8 = @bitCast(a_qs.*);
    const a_i16: QKV8i16 = @intCast(a_i8);
    return dotI16I16x8(w, a_i16);
}

fn dotI16I16x16(w: QKV16i16, a: QKV16i16) i32 {
    const product_i16 = w * a;
    const product_i32: QKV16i32 = @intCast(product_i16);
    return @reduce(.Add, product_i32);
}

fn dotI16I16x8(w: QKV8i16, a: QKV8i16) i32 {
    const product_i16 = w * a;
    const product_i32: QKV8i32 = @intCast(product_i16);
    return @reduce(.Add, product_i32);
}

fn reduceI16x8(a: QKV8i16) i32 {
    const a_i32: QKV8i32 = @intCast(a);
    return @reduce(.Add, a_i32);
}

fn signsVector8(signs: u8) QKV8i16 {
    const active = (@as(QKV8u8, @splat(signs)) & qk_v8_sign_masks) != @as(QKV8u8, @splat(0));
    return @select(i16, active, @as(QKV8i16, @splat(-1)), @as(QKV8i16, @splat(1)));
}

fn gridU64Vector(comptime table: []const u64, index: usize) QKV8u8 {
    var out: [8]u8 = undefined;
    inline for (0..8) |j| out[j] = gridU64Byte(table, index, j);
    return @bitCast(out);
}

fn gridU32PairVector(comptime table: []const u32, grid1: usize, grid2: usize) QKV8u8 {
    var out: [8]u8 = undefined;
    inline for (0..4) |j| {
        out[j] = gridU32Byte(table, grid1, j);
        out[j + 4] = gridU32Byte(table, grid2, j);
    }
    return @bitCast(out);
}

fn gridI8Vector(comptime table: []const u64, index: usize) QKV8i16 {
    var out: [8]i8 = undefined;
    inline for (0..8) |j| out[j] = gridI8Byte(table, index, j);
    const v_i8: QKV8i8 = @bitCast(out);
    return @intCast(v_i8);
}

fn nibbleTableVectors(comptime table: *const [16]i8, qs: *const [16]u8) NibbleTableVectors {
    var lo: [16]i16 = undefined;
    var hi: [16]i16 = undefined;
    for (qs, 0..) |q, j| {
        lo[j] = @intCast(table[q & 0x0f]);
        hi[j] = @intCast(table[q >> 4]);
    }
    return .{ .lo = @bitCast(lo), .hi = @bitCast(hi) };
}

fn dotNibbleTable32Q8(
    comptime table: *const [16]i8,
    qs: *const [16]u8,
    a_lo: *const [16]i8,
    a_hi: *const [16]i8,
) i32 {
    const w = nibbleTableVectors(table, qs);
    return dotI16I8x16(w.lo, a_lo) + dotI16I8x16(w.hi, a_hi);
}

fn dotNibbleTable16Q8(comptime table: *const [16]i8, qs: *const [8]u8, a_qs: *const [16]i8) i32 {
    var decoded: [16]i16 = undefined;
    for (qs, 0..) |q, j| {
        decoded[j] = @intCast(table[q & 0x0f]);
        decoded[j + 8] = @intCast(table[q >> 4]);
    }
    return dotI16I8x16(@bitCast(decoded), a_qs);
}

fn dotTQ2Lane16(qs: *const [16]u8, a_qs: *const [16]i8, comptime lane: usize) i32 {
    const q: QKV16u8 = @bitCast(qs.*);
    const vals = (q >> @as(QKV16u8, @splat(lane * 2))) & @as(QKV16u8, @splat(0x03));
    const q_i16: QKV16i16 = @as(QKV16i16, @intCast(vals)) - @as(QKV16i16, @splat(1));
    return dotI16I8x16(q_i16, a_qs);
}

fn dotTernaryLane16(qs: *const [16]u8, a_qs: *const [16]i8, comptime pow3: u8) i32 {
    const q: QKV16u8 = @bitCast(qs.*);
    const wrapped = q *% @as(QKV16u8, @splat(pow3));
    const xi_u16 = (@as(QKV16u16, @intCast(wrapped)) * @as(QKV16u16, @splat(3))) >> @as(QKV16u16, @splat(8));
    const xi_i16: QKV16i16 = @intCast(xi_u16);
    return dotI16I8x16(xi_i16 - @as(QKV16i16, @splat(1)), a_qs);
}

const PreparedQ8_0Block = struct {
    lo: Q4V16i16,
    hi: Q4V16i16,
    scale: f32,
};

fn prepareQ8_0Block(a: *const BlockQ8_0) PreparedQ8_0Block {
    const a_lo_i8: Q4V16i8 = @bitCast(a.qs[0 .. q4_0_block_size / 2].*);
    const a_hi_i8: Q4V16i8 = @bitCast(a.qs[q4_0_block_size / 2 .. q4_0_block_size].*);
    return .{
        .lo = @intCast(a_lo_i8),
        .hi = @intCast(a_hi_i8),
        .scale = f16BitsToF32(a.d),
    };
}

fn dotQ4_0Q8_0(w: *const BlockQ4_0, a: *const BlockQ8_0) f32 {
    return dotQ4_0PreparedQ8_0(w, prepareQ8_0Block(a));
}

fn dotQ4_0PreparedQ8_0(w: *const BlockQ4_0, a: PreparedQ8_0Block) f32 {
    const q: Q4V16u8 = @bitCast(w.qs);
    const lo_i16: Q4V16i16 = @intCast(q & @as(Q4V16u8, @splat(0x0f)));
    const hi_i16: Q4V16i16 = @intCast(q >> @as(Q4V16u8, @splat(4)));
    const w_lo = lo_i16 - @as(Q4V16i16, @splat(8));
    const w_hi = hi_i16 - @as(Q4V16i16, @splat(8));

    const acc_lo: i16 = @reduce(.Add, w_lo * a.lo);
    const acc_hi: i16 = @reduce(.Add, w_hi * a.hi);
    const acc: i32 = @as(i32, acc_lo) + @as(i32, acc_hi);
    const d = f16BitsToF32(w.d) * a.scale;
    return @as(f32, @floatFromInt(acc)) * d;
}

fn dotQ2_KQ8_K(w: *const BlockQ2_K, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.dm[0]) * a.d;
    const dmin = f16BitsToF32(w.dm[1]) * a.d;

    var sum_min: i32 = 0;
    inline for (0..16) |group| {
        sum_min += @as(i32, a.bsums[group]) * @as(i32, w.scales[group] >> 4);
    }

    var sum: i32 = 0;
    inline for (0..2) |chunk| {
        inline for (0..4) |section| {
            inline for (0..2) |half| {
                const group = chunk * 8 + section * 2 + half;
                const acc = dotQ2_KGroupI32(w, a, chunk, section, half);
                sum += @as(i32, w.scales[group] & 0x0f) * acc;
            }
        }
    }
    return d * @as(f32, @floatFromInt(sum)) - dmin * @as(f32, @floatFromInt(sum_min));
}

fn dotQ2_KGroupI32(w: *const BlockQ2_K, a: *const BlockQ8_K, comptime chunk: usize, comptime section: usize, comptime half: usize) i32 {
    const q_offset = chunk * 32 + half * 16;
    const a_offset = chunk * 128 + section * 32 + half * 16;
    const q: QKV16u8 = @bitCast(w.qs[q_offset..][0..16].*);
    const vals = (q >> @as(QKV16u8, @splat(section * 2))) & @as(QKV16u8, @splat(0x03));
    const q_i16: QKV16i16 = @intCast(vals);
    const a_i8: QKV16i8 = @bitCast(a.qs[a_offset..][0..16].*);
    const a_i16: QKV16i16 = @intCast(a_i8);
    const product_i16 = q_i16 * a_i16;
    const product_i32: QKV16i32 = @intCast(product_i16);
    return @reduce(.Add, product_i32);
}

pub fn dequantizeBlockQ2_KInto(dst: *[qk_k_block_size]f32, src: *const BlockQ2_K) void {
    const d = f16BitsToF32(src.dm[0]);
    const dmin = f16BitsToF32(src.dm[1]);
    var index: usize = 0;
    while (index < qk_k_block_size) : (index += 1) {
        const group = index / 16;
        const q: f32 = @floatFromInt(q2KValue(src, index));
        dst[index] = d * @as(f32, @floatFromInt(src.scales[group] & 0x0f)) * q -
            dmin * @as(f32, @floatFromInt(src.scales[group] >> 4));
    }
}

fn dotQ3_KQ8_K(w: *const BlockQ3_K, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.d) * a.d;
    var sum: f32 = 0;
    inline for (0..16) |group| {
        const acc = dotQ3_KGroupI32(w, a, group);
        sum += @as(f32, @floatFromInt(acc)) * @as(f32, @floatFromInt(q3KScale(w, group)));
    }
    return sum * d;
}

fn dotQ3_KGroupI32(w: *const BlockQ3_K, a: *const BlockQ8_K, comptime group: usize) i32 {
    const section = (group / 2) % 4;
    const half = group % 2;
    const chunk = group / 8;
    const q_offset = chunk * 32 + half * 16;
    const a_offset = group * 16;
    const mask: u8 = @as(u8, 1) << @intCast(group / 2);

    const q: QKV16u8 = @bitCast(w.qs[q_offset..][0..16].*);
    const hm_offset = half * 16;
    const hm: QKV16u8 = @bitCast(w.hmask[hm_offset..][0..16].*);
    const low = (q >> @as(QKV16u8, @splat(section * 2))) & @as(QKV16u8, @splat(0x03));
    const has_high = (hm & @as(QKV16u8, @splat(mask))) != @as(QKV16u8, @splat(0));
    const subtract = @select(i16, has_high, @as(QKV16i16, @splat(0)), @as(QKV16i16, @splat(4)));
    const q_i16: QKV16i16 = @as(QKV16i16, @intCast(low)) - subtract;
    const a_i8: QKV16i8 = @bitCast(a.qs[a_offset..][0..16].*);
    const a_i16: QKV16i16 = @intCast(a_i8);
    const product_i16 = q_i16 * a_i16;
    const product_i32: QKV16i32 = @intCast(product_i16);
    return @reduce(.Add, product_i32);
}

pub fn dequantizeBlockQ3_KInto(dst: *[qk_k_block_size]f32, src: *const BlockQ3_K) void {
    const d = f16BitsToF32(src.d);
    var index: usize = 0;
    while (index < qk_k_block_size) : (index += 1) {
        const group = index / 16;
        dst[index] = d * @as(f32, @floatFromInt(q3KScale(src, group))) *
            @as(f32, @floatFromInt(q3KValue(src, index)));
    }
}

fn q2KValue(w: *const BlockQ2_K, index: usize) u8 {
    const chunk = index / 128;
    const local = index % 128;
    const section = local / 32;
    const half = (local % 32) / 16;
    const offset = local % 16;
    const byte = w.qs[chunk * 32 + half * 16 + offset];
    return (byte >> @intCast(section * 2)) & 0x03;
}

fn q3KScale(w: *const BlockQ3_K, index: usize) i8 {
    const low = if (index < 8)
        w.scales[index] & 0x0f
    else
        w.scales[index - 8] >> 4;
    const high = (w.scales[8 + index % 4] >> @intCast(2 * (index / 4))) & 0x03;
    const combined: i16 = @as(i16, low) | (@as(i16, high) << 4);
    return @intCast(combined - 32);
}

fn q3KValue(w: *const BlockQ3_K, index: usize) i8 {
    const chunk = index / 128;
    const local = index % 128;
    const section = local / 32;
    const offset = local % 32;
    const byte_index = chunk * 32 + offset;
    const low = (w.qs[byte_index] >> @intCast(section * 2)) & 0x03;
    const high_mask: u8 = @as(u8, 1) << @intCast(chunk * 4 + section);
    const combined: i16 = @intCast(low);
    const hmask_index = offset;
    const subtract: i16 = if ((w.hmask[hmask_index] & high_mask) != 0) 0 else 4;
    return @intCast(combined - subtract);
}

fn fillQ8_0Pattern(block: *BlockQ8_0) void {
    block.d = f32ToF16Bits(1);
    for (&block.qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast(i % 17)) - 8);
}

fn fillQ8_1Pattern(block: *BlockQ8_1) void {
    var sum: i32 = 0;
    for (&block.qs, 0..) |*q, i| {
        q.* = @intCast(@as(i32, @intCast(i % 17)) - 8);
        sum += q.*;
    }
    block.ds = .{ f32ToF16Bits(1), f32ToF16Bits(@floatFromInt(sum)) };
}

fn fillQ1_0Pattern(block: *BlockQ1_0) void {
    block.d = f32ToF16Bits(1);
    for (&block.qs, 0..) |*q, i| q.* = if (i % 2 == 0) 0b1010_0101 else 0b0101_1010;
}

fn fillQ4_1Pattern(block: *BlockQ4_1) void {
    block.dm = .{ f32ToF16Bits(1), f32ToF16Bits(0) };
    for (&block.qs, 0..) |*q, i| {
        const lo: u8 = @intCast(i % 16);
        const hi: u8 = @intCast((i + 5) % 16);
        q.* = lo | (hi << 4);
    }
}

fn setQ5_0Value(block: *BlockQ5_0, index: usize, value: i8) void {
    const encoded: u8 = @intCast(@as(i16, value) + 16);
    const byte_index = index % (q5_0_block_size / 2);
    if (index < q5_0_block_size / 2) {
        block.qs[byte_index] = (block.qs[byte_index] & 0xf0) | (encoded & 0x0f);
    } else {
        block.qs[byte_index] = (block.qs[byte_index] & 0x0f) | ((encoded & 0x0f) << 4);
    }
    const bit: u5 = @intCast(index);
    if ((encoded & 0x10) != 0) {
        writeQh(block.qh[0..], readQh(&block.qh) | (@as(u32, 1) << bit));
    } else {
        writeQh(block.qh[0..], readQh(&block.qh) & ~(@as(u32, 1) << bit));
    }
}

fn fillQ5_0Pattern(block: *BlockQ5_0) void {
    block.d = f32ToF16Bits(1);
    @memset(&block.qh, 0);
    @memset(&block.qs, 0);
    for (0..q5_0_block_size) |i| setQ5_0Value(block, i, @intCast(@as(i32, @intCast(i % 23)) - 11));
}

fn setQ5_1Value(block: *BlockQ5_1, index: usize, value: u8) void {
    const byte_index = index % (q5_1_block_size / 2);
    if (index < q5_1_block_size / 2) {
        block.qs[byte_index] = (block.qs[byte_index] & 0xf0) | (value & 0x0f);
    } else {
        block.qs[byte_index] = (block.qs[byte_index] & 0x0f) | ((value & 0x0f) << 4);
    }
    const bit: u5 = @intCast(index);
    if ((value & 0x10) != 0) {
        writeQh(block.qh[0..], readQh(&block.qh) | (@as(u32, 1) << bit));
    } else {
        writeQh(block.qh[0..], readQh(&block.qh) & ~(@as(u32, 1) << bit));
    }
}

fn fillQ5_1Pattern(block: *BlockQ5_1) void {
    block.dm = .{ f32ToF16Bits(1), f32ToF16Bits(0) };
    @memset(&block.qh, 0);
    @memset(&block.qs, 0);
    for (0..q5_1_block_size) |i| setQ5_1Value(block, i, @intCast((i * 7) % 32));
}

fn fillQ2KPattern(block: *BlockQ2_K) void {
    block.dm = .{ f32ToF16Bits(1), f32ToF16Bits(0) };
    for (&block.scales, 0..) |*scale, i| scale.* = @intCast((i % 7) + 1);
    for (&block.qs, 0..) |*q, i| {
        q.* = @intCast((i % 4) | (((i + 1) % 4) << 2) | (((i + 2) % 4) << 4) | (((i + 3) % 4) << 6));
    }
}

fn setQ3KScale(block: *BlockQ3_K, index: usize, scale: i8) void {
    const encoded: u8 = @intCast(@as(i16, scale) + 32);
    if (index < 8) {
        block.scales[index] = (block.scales[index] & 0xf0) | (encoded & 0x0f);
    } else {
        block.scales[index - 8] = (block.scales[index - 8] & 0x0f) | ((encoded & 0x0f) << 4);
    }
    const high_index = 8 + index % 4;
    const shift: u3 = @intCast(2 * (index / 4));
    block.scales[high_index] = (block.scales[high_index] & ~(@as(u8, 0x03) << shift)) | (((encoded >> 4) & 0x03) << shift);
}

fn setQ3KValue(block: *BlockQ3_K, index: usize, value: i8) void {
    const chunk = index / 128;
    const local = index % 128;
    const section = local / 32;
    const offset = local % 32;
    const byte_index = chunk * 32 + offset;
    const shift: u3 = @intCast(section * 2);
    const encoded: u8 = if (value >= 0) @intCast(value) else @intCast(@as(i16, value) + 4);
    block.qs[byte_index] = (block.qs[byte_index] & ~(@as(u8, 0x03) << shift)) | ((encoded & 0x03) << shift);
    const high_mask: u8 = @as(u8, 1) << @intCast(chunk * 4 + section);
    if (value >= 0) {
        block.hmask[offset] |= high_mask;
    } else {
        block.hmask[offset] &= ~high_mask;
    }
}

fn fillQ3KPattern(block: *BlockQ3_K) void {
    @memset(&block.hmask, 0);
    @memset(&block.qs, 0);
    @memset(&block.scales, 0);
    block.d = f32ToF16Bits(1);
    for (0..qk_k_block_size / 16) |i| {
        const scale: i8 = @intCast(@as(i32, @intCast(i % 5)) + 1);
        setQ3KScale(block, i, scale);
    }
    for (0..qk_k_block_size) |i| {
        const value: i8 = @intCast(@as(i32, @intCast(i % 8)) - 4);
        setQ3KValue(block, i, value);
    }
}

test "ggml_q1_0 dot and matmul consume loaded blocks" {
    var q1: BlockQ1_0 = undefined;
    fillQ1_0Pattern(&q1);
    var q8 = [_]BlockQ8_0{undefined} ** (q1_0_block_size / q8_0_block_size);
    for (&q8) |*block| fillQ8_0Pattern(block);

    var dense_w: [q1_0_block_size]f32 = undefined;
    try dequantizeRowQ1_0Into(&dense_w, &.{q1});
    var dense_a: [q1_0_block_size]f32 = undefined;
    for (&q8, 0..) |*block, i| {
        for (block.qs, 0..) |v, j| {
            dense_a[i * q8_0_block_size + j] = @as(f32, @floatFromInt(v)) * f16BitsToF32(block.d);
        }
    }

    try std.testing.expectEqual(dotDense(&dense_w, &dense_a), dotQ1_0Q8_0(&q1, &q8));

    var rhs_blocks = [_]BlockQ1_0{ q1, q1 };
    var rhs = QuantizedMatmulRhsQ1_0{
        .rows = .{ .allocator = std.testing.allocator, .blocks = &rhs_blocks, .rows = 2, .cols = q1_0_block_size, .blocks_per_row = 1 },
        .k = q1_0_block_size,
        .n = 2,
    };
    var out: [2]f32 = undefined;
    matmulQ1_0RhsRange(&out, &q8, &rhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ1_0Q8_0(&q1, &q8), out[0]);
}

test "ggml_q4_1 dot and matmul consume loaded blocks" {
    var q4: BlockQ4_1 = undefined;
    fillQ4_1Pattern(&q4);
    var q8: BlockQ8_1 = undefined;
    fillQ8_1Pattern(&q8);

    var dense_w: [q4_1_block_size]f32 = undefined;
    try dequantizeRowQ4_1Into(&dense_w, &.{q4});
    var dense_a: [q8_1_block_size]f32 = undefined;
    try dequantizeRowQ8_1Into(&dense_a, &.{q8});

    try std.testing.expectEqual(dotDense(&dense_w, &dense_a), dotQ4_1Q8_1(&q4, &q8));

    var rhs_blocks = [_]BlockQ4_1{ q4, q4 };
    var rhs = QuantizedMatmulRhsQ4_1{
        .rows = .{ .allocator = std.testing.allocator, .blocks = &rhs_blocks, .rows = 2, .cols = q4_1_block_size, .blocks_per_row = 1 },
        .k = q4_1_block_size,
        .n = 2,
    };
    var out: [2]f32 = undefined;
    matmulQ4_1RhsRange(&out, &.{q8}, &rhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ4_1Q8_1(&q4, &q8), out[0]);
}

test "ggml_q5_0 dot and matmul consume loaded blocks" {
    var q5: BlockQ5_0 = undefined;
    fillQ5_0Pattern(&q5);
    var q8: BlockQ8_0 = undefined;
    fillQ8_0Pattern(&q8);

    var dense_w: [q5_0_block_size]f32 = undefined;
    try dequantizeRowQ5_0Into(&dense_w, &.{q5});
    var dense_a: [q8_0_block_size]f32 = undefined;
    try dequantizeRowQ8_0Into(&dense_a, &.{q8});

    try std.testing.expectEqual(dotDense(&dense_w, &dense_a), dotQ5_0Q8_0(&q5, &q8));

    var rhs_blocks = [_]BlockQ5_0{ q5, q5 };
    var rhs = QuantizedMatmulRhsQ5_0{
        .rows = .{ .allocator = std.testing.allocator, .blocks = &rhs_blocks, .rows = 2, .cols = q5_0_block_size, .blocks_per_row = 1 },
        .k = q5_0_block_size,
        .n = 2,
    };
    var out: [2]f32 = undefined;
    matmulQ5_0RhsRange(&out, &.{q8}, &rhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ5_0Q8_0(&q5, &q8), out[0]);
}

test "ggml_q5_1 dot and matmul consume loaded blocks" {
    var q5: BlockQ5_1 = undefined;
    fillQ5_1Pattern(&q5);
    var q8: BlockQ8_1 = undefined;
    fillQ8_1Pattern(&q8);

    var dense_w: [q5_1_block_size]f32 = undefined;
    try dequantizeRowQ5_1Into(&dense_w, &.{q5});
    var dense_a: [q8_1_block_size]f32 = undefined;
    try dequantizeRowQ8_1Into(&dense_a, &.{q8});

    try std.testing.expectEqual(dotDense(&dense_w, &dense_a), dotQ5_1Q8_1(&q5, &q8));

    var rhs_blocks = [_]BlockQ5_1{ q5, q5 };
    var rhs = QuantizedMatmulRhsQ5_1{
        .rows = .{ .allocator = std.testing.allocator, .blocks = &rhs_blocks, .rows = 2, .cols = q5_1_block_size, .blocks_per_row = 1 },
        .k = q5_1_block_size,
        .n = 2,
    };
    var out: [2]f32 = undefined;
    matmulQ5_1RhsRange(&out, &.{q8}, &rhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ5_1Q8_1(&q5, &q8), out[0]);
}

test "ggml_q2_k dot and matmul consume loaded blocks" {
    const allocator = std.testing.allocator;

    var q2: BlockQ2_K = undefined;
    fillQ2KPattern(&q2);
    var q8: BlockQ8_K = undefined;
    fillQ8KPattern(&q8);

    var dense_w: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ2_KInto(&dense_w, &q2);
    var dense_a: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ8_KInto(&dense_a, &q8);

    try std.testing.expectEqual(dotDense(&dense_w, &dense_a), dotQ2_KQ8_K(&q2, &q8));

    var rhs_blocks = [_]BlockQ2_K{ q2, q2 };
    var qrhs = try quantizedMatmulRhsQ2_KFromBlocks(allocator, qk_k_block_size, 2, &rhs_blocks);
    defer qrhs.deinit();
    var out: [2]f32 = undefined;
    matmulQ2_KRhsRange(&out, &.{q8}, &qrhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ2_KQ8_K(&q2, &q8), out[0]);
}

test "ggml_q3_k dot and matmul consume loaded blocks" {
    const allocator = std.testing.allocator;

    var q3: BlockQ3_K = undefined;
    fillQ3KPattern(&q3);
    var q8: BlockQ8_K = undefined;
    fillQ8KPattern(&q8);

    var dense_w: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ3_KInto(&dense_w, &q3);
    var dense_a: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ8_KInto(&dense_a, &q8);

    try std.testing.expectEqual(dotDense(&dense_w, &dense_a), dotQ3_KQ8_K(&q3, &q8));

    var rhs_blocks = [_]BlockQ3_K{ q3, q3 };
    var qrhs = try quantizedMatmulRhsQ3_KFromBlocks(allocator, qk_k_block_size, 2, &rhs_blocks);
    defer qrhs.deinit();
    var out: [2]f32 = undefined;
    matmulQ3_KRhsRange(&out, &.{q8}, &qrhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ3_KQ8_K(&q3, &q8), out[0]);
}

test {
    _ = @import("cold_tests.zig");
}
