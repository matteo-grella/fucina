//! Cold-format quantized kernels: the rarely-served GGML formats, each with a
//! straightforward per-block dot skeleton — correctness-first, no per-ISA
//! tuning (the hot per-ISA kernels live in the sibling per-format modules).
//! Block/RHS types come from `types.zig` and `common.zig`, dequant tables
//! from `../quant_tables.zig`; the naming grammar is in `quant.zig`.

const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const tensor = @import("../../tensor.zig");
const tables = @import("../quant_tables.zig");
const q8k = @import("q8k.zig");
const types = @import("types.zig");
const common = @import("common.zig");
const ops = @import("../ops.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

const BlockQ8_0 = dtype_mod.BlockQ8_0;
const BlockQ8_K = dtype_mod.BlockQ8_K;
const QKV16i16 = common.QKV16i16;
const QKV16u8 = common.QKV16u8;
const QuantizedFormatError = types.QuantizedFormatError;
const checkedProduct = types.checkedProduct;
const f16BitsToF32 = common.f16BitsToF32;
const f32ToF16Bits = common.f32ToF16Bits;
const qk_k_block_size = types.qk_k_block_size;

pub fn quantizeRowQ4_0Into(dst: []dtype_mod.BlockQ4_0, src: []const f32) !void {
    const block_count = try q4_0BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    var block_index: usize = 0;
    while (block_index < block_count) : (block_index += 1) {
        const row = src[block_index * types.q4_0_block_size ..][0..types.q4_0_block_size];
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
            const x1 = quantizeToQ4_0Nibble(row[types.q4_0_block_size / 2 + j] * inv_d);
            q.* = x0 | (x1 << 4);
        }
    }
}

pub fn dequantizeRowQ4_0Into(dst: []f32, src: []const dtype_mod.BlockQ4_0) !void {
    if (dst.len != try checkedProduct(src.len, types.q4_0_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * types.q4_0_block_size ..][0..types.q4_0_block_size];
        for (block.qs, 0..) |q, j| {
            const x0: i32 = @as(i32, q & 0x0f) - 8;
            const x1: i32 = @as(i32, q >> 4) - 8;
            out[j] = @as(f32, @floatFromInt(x0)) * d;
            out[types.q4_0_block_size / 2 + j] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

pub fn quantizeRowsQ4_0(allocator: Allocator, src: *const Tensor) !types.QuantizedRowsQ4_0 {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    const blocks_per_row = try q4_0BlockCount(cols);
    const data = try src.dataConstChecked();

    const blocks = try allocator.alloc(dtype_mod.BlockQ4_0, try checkedProduct(rows, blocks_per_row));
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

pub fn quantizeMatmulRhsQ4_0(allocator: Allocator, rhs: *const Tensor) !types.QuantizedMatmulRhsQ4_0 {
    const view = try rhs.rankView(2);
    const k = view.dim(0);
    const n = view.dim(1);
    const blocks_per_column = try q4_0BlockCount(k);
    const data = try rhs.dataConstChecked();

    const blocks = try allocator.alloc(dtype_mod.BlockQ4_0, try checkedProduct(n, blocks_per_column));
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

pub fn dequantizeRowsQ4_0Into(dst: *Tensor, src: *const types.QuantizedRowsQ4_0) !void {
    const view = try dst.rankView(2);
    if (view.dim(0) != src.rows or view.dim(1) != src.cols) return tensor.TensorError.ShapeMismatch;

    const out = try dst.dataChecked();
    var row: usize = 0;
    while (row < src.rows) : (row += 1) {
        try dequantizeRowQ4_0Into(out[row * src.cols ..][0..src.cols], src.rowBlocks(row));
    }
}

pub fn getRowsQ4_0Into(dst: *Tensor, table: *const types.QuantizedRowsQ4_0, indices: []const usize) !void {
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
    return types.blockCountExact(types.q4_0_block_size, len);
}

fn quantizeToQ4_0Nibble(x: f32) u8 {
    // ggml: MIN(15, (int8_t)(x + 8.5f)). Clamp in float space with Zig's
    // NaN-collapsing @min/@max so a degenerate x (NaN/inf) never reaches
    // @intFromFloat (UB in ReleaseFast). In-contract x is in [-8, 8], where
    // trunc(x + 8.5) is in [0, 16] and both formulations agree byte-exactly.
    return @intFromFloat(@min(@as(f32, 15.0), @max(@as(f32, 0.0), x + 8.5)));
}

pub fn q1_0BlockCount(len: usize) !usize {
    return types.blockCountExact(types.q1_0_block_size, len);
}

pub fn q2_0BlockCount(len: usize) !usize {
    return types.blockCountExact(types.q2_0_block_size, len);
}

pub fn q4_1BlockCount(len: usize) !usize {
    return types.blockCountExact(types.q4_1_block_size, len);
}

pub fn q5_0BlockCount(len: usize) !usize {
    return types.blockCountExact(types.q5_0_block_size, len);
}

pub fn q5_1BlockCount(len: usize) !usize {
    return types.blockCountExact(types.q5_1_block_size, len);
}

pub fn q8_1BlockCount(len: usize) !usize {
    return types.blockCountExact(types.q8_1_block_size, len);
}

pub fn dequantizeRowQ1_0Into(dst: []f32, src: []const dtype_mod.BlockQ1_0) !void {
    if (dst.len != try checkedProduct(src.len, types.q1_0_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * types.q1_0_block_size ..][0..types.q1_0_block_size];
        for (out, 0..) |*y, j| {
            const mask: u8 = @as(u8, 1) << @intCast(j % 8);
            y.* = if ((block.qs[j / 8] & mask) != 0) d else -d;
        }
    }
}

/// ggml dequantize_row_q2_0 parity (PrismML fork / Bonsai Q2_0_g128): four
/// 2-bit codes per byte, LSB-first, code q in {0,1,2,3} decodes to (q-1)*d.
/// Ternary files only ever carry {0,1,2} (the encoder rounds against the
/// block absmax, so |w/d| <= 1); code 3 = +2d is part of the wire contract.
pub fn dequantizeRowQ2_0Into(dst: []f32, src: []const dtype_mod.BlockQ2_0) !void {
    if (dst.len != try checkedProduct(src.len, types.q2_0_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * types.q2_0_block_size ..][0..types.q2_0_block_size];
        for (out, 0..) |*y, j| {
            const shift: u3 = @intCast((j % 4) * 2);
            const q: i32 = (block.qs[j / 4] >> shift) & 0x3;
            y.* = @as(f32, @floatFromInt(q - 1)) * d;
        }
    }
}

/// f32 -> Q2_0 row encoder; faithful port of the fork's quantize_row_q2_0_ref
/// (per-block absmax d, round-half-away, codes clamped to [0,3] — in-range
/// finite input only ever produces {0,1,2}, i.e. ternary weights). Assumes
/// finite input (guarded at the gguf.encodeF32 seam); degenerate-but-finite
/// blocks (subnormal absmax overflowing 1/d) produce defined clamped output
/// instead of @intFromFloat UB.
pub fn quantizeRowQ2_0Into(dst: []dtype_mod.BlockQ2_0, src: []const f32) !void {
    const block_count = try q2_0BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    for (dst, 0..) |*block, block_index| {
        const x = src[block_index * types.q2_0_block_size ..][0..types.q2_0_block_size];
        var amax: f32 = 0;
        for (x) |v| amax = @max(amax, @abs(v));

        const d = amax;
        var inv_d: f32 = if (d != 0) 1.0 / d else 0.0;
        if (!std.math.isFinite(inv_d)) inv_d = 0;
        block.d = f32ToF16Bits(d);
        block.qs = @splat(0);

        for (x, 0..) |v, j| {
            // Clamp in the float domain: round(v*inv_d) is in [-1, 1] for
            // in-contract blocks, so the clamps never bite there.
            const r = @max(0.0, @min(3.0, common.roundHalfAwayFromZero(v * inv_d) + 1.0));
            const q: u8 = @intFromFloat(r);
            const shift: u3 = @intCast((j % 4) * 2);
            block.qs[j / 4] |= q << shift;
        }
    }
}

/// f32 -> Q4_1 row encoder; faithful port of ggml's quantize_row_q4_1_ref
/// (byte-exact, see quant/encode_golden_test.zig). Assumes finite input;
/// degenerate-but-finite blocks (subnormal/overflowing spreads) produce
/// defined clamped output instead of @intFromFloat UB.
pub fn quantizeRowQ4_1Into(dst: []dtype_mod.BlockQ4_1, src: []const f32) !void {
    const block_count = try q4_1BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    for (dst, 0..) |*block, block_index| {
        const row = src[block_index * types.q4_1_block_size ..][0..types.q4_1_block_size];
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
            const x1 = (row[types.q4_1_block_size / 2 + j] - min) * inv_d;
            // ggml: MIN(15, (int8_t)(x + 0.5f)); the float-space clamp with
            // NaN-collapsing @min/@max is byte-identical for in-contract
            // x in [0, 15] and keeps NaN away from @intFromFloat.
            const xi0: u8 = @intFromFloat(@min(@as(f32, 15.0), @max(@as(f32, 0.0), x0 + 0.5)));
            const xi1: u8 = @intFromFloat(@min(@as(f32, 15.0), @max(@as(f32, 0.0), x1 + 0.5)));
            q.* = xi0 | (xi1 << 4);
        }
    }
}

pub fn dequantizeRowQ4_1Into(dst: []f32, src: []const dtype_mod.BlockQ4_1) !void {
    if (dst.len != try checkedProduct(src.len, types.q4_1_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.dm[0]);
        const m = f16BitsToF32(block.dm[1]);
        const out = dst[block_index * types.q4_1_block_size ..][0..types.q4_1_block_size];
        for (block.qs, 0..) |q, j| {
            out[j] = @as(f32, @floatFromInt(q & 0x0f)) * d + m;
            out[types.q4_1_block_size / 2 + j] = @as(f32, @floatFromInt(q >> 4)) * d + m;
        }
    }
}

/// f32 -> Q5_0 row encoder; faithful port of ggml's quantize_row_q5_0_ref
/// (byte-exact, see quant/encode_golden_test.zig). Assumes finite input;
/// degenerate-but-finite blocks (subnormal spreads) produce defined clamped
/// output instead of @intFromFloat UB.
pub fn quantizeRowQ5_0Into(dst: []dtype_mod.BlockQ5_0, src: []const f32) !void {
    const block_count = try q5_0BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    for (dst, 0..) |*block, block_index| {
        const row = src[block_index * types.q5_0_block_size ..][0..types.q5_0_block_size];
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
            const x1 = row[types.q5_0_block_size / 2 + j] * inv_d;
            // ggml: MIN(31, (int8_t)(x + 16.5f)); the float-space clamp with
            // NaN-collapsing @min/@max is byte-identical for in-contract
            // x in [-16, 16] and keeps NaN away from @intFromFloat.
            const xi0: u8 = @intFromFloat(@min(@as(f32, 31.0), @max(@as(f32, 0.0), x0 + 16.5)));
            const xi1: u8 = @intFromFloat(@min(@as(f32, 31.0), @max(@as(f32, 0.0), x1 + 16.5)));
            q.* = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
            // the 5-th bit, stored across the packed qh word
            qh |= @as(u32, (xi0 & 0x10) >> 4) << @intCast(j);
            qh |= @as(u32, (xi1 & 0x10) >> 4) << @intCast(j + types.q5_0_block_size / 2);
        }
        writeQh(&block.qh, qh);
    }
}

pub fn dequantizeRowQ5_0Into(dst: []f32, src: []const dtype_mod.BlockQ5_0) !void {
    if (dst.len != try checkedProduct(src.len, types.q5_0_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const qh = readQh(&block.qh);
        const out = dst[block_index * types.q5_0_block_size ..][0..types.q5_0_block_size];
        for (block.qs, 0..) |q, j| {
            const xh0: u8 = @intCast(((qh >> @intCast(j)) << 4) & 0x10);
            const xh1: u8 = @intCast((qh >> @intCast(j + 12)) & 0x10);
            const x0: i32 = @as(i32, (q & 0x0f) | xh0) - 16;
            const x1: i32 = @as(i32, (q >> 4) | xh1) - 16;
            out[j] = @as(f32, @floatFromInt(x0)) * d;
            out[types.q5_0_block_size / 2 + j] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

/// f32 -> Q5_1 row encoder; faithful port of ggml's quantize_row_q5_1_ref
/// (byte-exact, see quant/encode_golden_test.zig). Assumes finite input;
/// degenerate-but-finite blocks (subnormal/overflowing spreads) produce
/// defined clamped output instead of @intFromFloat UB.
pub fn quantizeRowQ5_1Into(dst: []dtype_mod.BlockQ5_1, src: []const f32) !void {
    const block_count = try q5_1BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    for (dst, 0..) |*block, block_index| {
        const row = src[block_index * types.q5_1_block_size ..][0..types.q5_1_block_size];
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
            const x1 = (row[types.q5_1_block_size / 2 + j] - min) * inv_d;
            // ggml does not clamp here: in-contract x0/x1 are in [0, 31] by
            // construction, where the NaN-collapsing float-space clamp is
            // byte-identical; it only exists to keep degenerate NaN away
            // from @intFromFloat.
            const xi0: u8 = @intFromFloat(@min(@as(f32, 31.0), @max(@as(f32, 0.0), x0 + 0.5)));
            const xi1: u8 = @intFromFloat(@min(@as(f32, 31.0), @max(@as(f32, 0.0), x1 + 0.5)));
            q.* = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
            // the 5-th bit, stored across the packed qh word
            qh |= @as(u32, (xi0 & 0x10) >> 4) << @intCast(j);
            qh |= @as(u32, (xi1 & 0x10) >> 4) << @intCast(j + types.q5_1_block_size / 2);
        }
        writeQh(&block.qh, qh);
    }
}

pub fn dequantizeRowQ5_1Into(dst: []f32, src: []const dtype_mod.BlockQ5_1) !void {
    if (dst.len != try checkedProduct(src.len, types.q5_1_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.dm[0]);
        const m = f16BitsToF32(block.dm[1]);
        const qh = readQh(&block.qh);
        const out = dst[block_index * types.q5_1_block_size ..][0..types.q5_1_block_size];
        for (block.qs, 0..) |q, j| {
            const xh0: u8 = @intCast(((qh >> @intCast(j)) << 4) & 0x10);
            const xh1: u8 = @intCast((qh >> @intCast(j + 12)) & 0x10);
            const x0: u8 = (q & 0x0f) | xh0;
            const x1: u8 = (q >> 4) | xh1;
            out[j] = @as(f32, @floatFromInt(x0)) * d + m;
            out[types.q5_1_block_size / 2 + j] = @as(f32, @floatFromInt(x1)) * d + m;
        }
    }
}

pub fn quantizeRowQ8_1Into(dst: []dtype_mod.BlockQ8_1, src: []const f32) !void {
    const block_count = try q8_1BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    var block_index: usize = 0;
    while (block_index < block_count) : (block_index += 1) {
        const row = src[block_index * types.q8_1_block_size ..][0..types.q8_1_block_size];
        var amax: f32 = 0;
        for (row) |v| amax = @max(amax, @abs(v));

        const d = amax / 127.0;
        const inv_d: f32 = if (d == 0) 0 else 1.0 / d;
        var sum: i32 = 0;
        for (&dst[block_index].qs, row) |*q, v| {
            q.* = common.quantizeToI8(v * inv_d);
            sum += q.*;
        }
        dst[block_index].ds = .{ f32ToF16Bits(d), f32ToF16Bits(@as(f32, @floatFromInt(sum)) * d) };
    }
}

pub fn quantizeRowsQ8_1(allocator: Allocator, src: *const Tensor) !types.QuantizedRowsQ8_1 {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    const blocks_per_row = try q8_1BlockCount(cols);
    const data = try src.dataConstChecked();

    const blocks = try allocator.alloc(dtype_mod.BlockQ8_1, try checkedProduct(rows, blocks_per_row));
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

pub fn dequantizeRowQ8_1Into(dst: []f32, src: []const dtype_mod.BlockQ8_1) !void {
    if (dst.len != try checkedProduct(src.len, types.q8_1_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.ds[0]);
        const out = dst[block_index * types.q8_1_block_size ..][0..types.q8_1_block_size];
        for (out, block.qs) |*y, q| y.* = @as(f32, @floatFromInt(q)) * d;
    }
}

pub fn dequantizeRowMXFP4Into(dst: []f32, src: []const dtype_mod.BlockMXFP4) !void {
    if (dst.len != try checkedProduct(src.len, types.mxfp4_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = common.e8m0ToF32Half(block.e);
        const out = dst[block_index * types.mxfp4_block_size ..][0..types.mxfp4_block_size];
        for (block.qs, 0..) |q, j| {
            out[j] = @as(f32, @floatFromInt(tables.kvalues_mxfp4[q & 0x0f])) * d;
            out[j + types.mxfp4_block_size / 2] = @as(f32, @floatFromInt(tables.kvalues_mxfp4[q >> 4])) * d;
        }
    }
}

pub fn dequantizeRowNVFP4Into(dst: []f32, src: []const dtype_mod.BlockNVFP4) !void {
    if (dst.len != try checkedProduct(src.len, types.nvfp4_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const out = dst[block_index * types.nvfp4_block_size ..][0..types.nvfp4_block_size];
        for (0..types.nvfp4_block_size / types.nvfp4_subblock_size) |subblock| {
            const d = ue4m3ToF32(block.d[subblock]);
            const sub_out = out[subblock * types.nvfp4_subblock_size ..][0..types.nvfp4_subblock_size];
            const qs = block.qs[subblock * (types.nvfp4_subblock_size / 2) ..][0 .. types.nvfp4_subblock_size / 2];
            for (qs, 0..) |q, j| {
                sub_out[j] = @as(f32, @floatFromInt(tables.kvalues_mxfp4[q & 0x0f])) * d;
                sub_out[j + types.nvfp4_subblock_size / 2] = @as(f32, @floatFromInt(tables.kvalues_mxfp4[q >> 4])) * d;
            }
        }
    }
}

pub fn dequantizeRowTQ1_0Into(dst: []f32, src: []const dtype_mod.BlockTQ1_0) !void {
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

pub fn dequantizeRowTQ2_0Into(dst: []f32, src: []const dtype_mod.BlockTQ2_0) !void {
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

pub fn dequantizeRowIQ2_XXSInto(dst: []f32, src: []const dtype_mod.BlockIQ2_XXS) !void {
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

pub fn dequantizeRowIQ2_XSInto(dst: []f32, src: []const dtype_mod.BlockIQ2_XS) !void {
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

pub fn dequantizeRowIQ2_SInto(dst: []f32, src: []const dtype_mod.BlockIQ2_S) !void {
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

pub fn dequantizeRowIQ3_XXSInto(dst: []f32, src: []const dtype_mod.BlockIQ3_XXS) !void {
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

pub fn dequantizeRowIQ3_SInto(dst: []f32, src: []const dtype_mod.BlockIQ3_S) !void {
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

pub fn dequantizeRowIQ1_SInto(dst: []f32, src: []const dtype_mod.BlockIQ1_S) !void {
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

pub fn dequantizeRowIQ1_MInto(dst: []f32, src: []const dtype_mod.BlockIQ1_M) !void {
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

pub fn dequantizeRowIQ4_NLInto(dst: []f32, src: []const dtype_mod.BlockIQ4_NL) !void {
    if (dst.len != try checkedProduct(src.len, types.iq4_nl_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    for (src, 0..) |block, block_index| {
        const d = f16BitsToF32(block.d);
        const out = dst[block_index * types.iq4_nl_block_size ..][0..types.iq4_nl_block_size];
        for (block.qs, 0..) |q, j| {
            out[j] = d * @as(f32, @floatFromInt(tables.kvalues_iq4nl[q & 0x0f]));
            out[j + types.iq4_nl_block_size / 2] = d * @as(f32, @floatFromInt(tables.kvalues_iq4nl[q >> 4]));
        }
    }
}

pub fn dequantizeRowIQ4_XSInto(dst: []f32, src: []const dtype_mod.BlockIQ4_XS) !void {
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

/// The Q8_0 activation block is widened once per K step (`prepareQ8_0Block`)
/// and shared across the column block.
const matmulQ4_0RhsTile = common.RowOuterTile(BlockQ8_0, types.QuantizedMatmulRhsQ4_0, q4_0_col_block, .{ .prepare = prepareQ8_0Block, .dot = dotQ4_0PreparedQ8_0 });

pub const matmulQ4_0RhsRange = common.RangeFromTile(matmulQ4_0RhsTile);

/// One Q1_0 block (128 weights) spans four Q8_0 activation blocks.
const matmulQ1_0RhsTile = common.RowOuterTile(BlockQ8_0, types.QuantizedMatmulRhsQ1_0, 1, .{ .lhs_per_rhs = types.q1_0_block_size / types.q8_0_block_size, .dot = dotQ1_0Q8_0 });

pub const matmulQ1_0RhsRange = common.RangeFromTile(matmulQ1_0RhsTile);

const matmulQ4_1RhsTile = common.RowOuterTile(dtype_mod.BlockQ8_1, types.QuantizedMatmulRhsQ4_1, 1, .{ .dot = dotQ4_1Q8_1 });

pub const matmulQ4_1RhsRange = common.RangeFromTile(matmulQ4_1RhsTile);

const matmulQ5_0RhsTile = common.RowOuterTile(BlockQ8_0, types.QuantizedMatmulRhsQ5_0, 1, .{ .dot = dotQ5_0Q8_0 });

pub const matmulQ5_0RhsRange = common.RangeFromTile(matmulQ5_0RhsTile);

const matmulQ5_1RhsTile = common.RowOuterTile(dtype_mod.BlockQ8_1, types.QuantizedMatmulRhsQ5_1, 1, .{ .dot = dotQ5_1Q8_1 });

pub const matmulQ5_1RhsRange = common.RangeFromTile(matmulQ5_1RhsTile);

const matmulQ2_KRhsTile = common.RowOuterTile(BlockQ8_K, types.QuantizedMatmulRhsQ2_K, common.qk_col_block, .{ .dot = dotQ2_KQ8_K });

pub const matmulQ2_KRhsRange = common.RangeFromTile(matmulQ2_KRhsTile);

const matmulQ3_KRhsTile = common.RowOuterTile(BlockQ8_K, types.QuantizedMatmulRhsQ3_K, common.qk_col_block, .{ .dot = dotQ3_KQ8_K });

pub const matmulQ3_KRhsRange = common.RangeFromTile(matmulQ3_KRhsTile);

pub fn matmulTableQ8_0RhsRange(
    comptime rhs_dtype: DType,
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const types.QuantizedMatmulRhsRowsFor(rhs_dtype),
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulTableQ8_0RhsTile(rhs_dtype, out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

fn matmulTableQ8_0RhsTile(
    comptime rhs_dtype: DType,
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const types.QuantizedMatmulRhsRowsFor(rhs_dtype),
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    TableQ8_0Tile(rhs_dtype)(out, lhs_blocks, rhs, n, r0, r1, c0, c1);
}

/// The Q8_0-activation table tile of `rhs_dtype`: one RHS block spans
/// `blockSize(rhs_dtype) / 32` activation blocks.
fn TableQ8_0Tile(comptime rhs_dtype: DType) common.TileFn(BlockQ8_0, types.QuantizedMatmulRhsRowsFor(rhs_dtype)) {
    return common.RowOuterTile(BlockQ8_0, types.QuantizedMatmulRhsRowsFor(rhs_dtype), table_q8_0_col_block, .{
        .lhs_per_rhs = dtype_mod.blockSize(rhs_dtype) / types.q8_0_block_size,
        .dot = struct {
            fn dot(w: *const dtype_mod.Storage(rhs_dtype), a: []const BlockQ8_0) f32 {
                return dotTableQ8_0(rhs_dtype, w, a);
            }
        }.dot,
    });
}

pub fn matmulTableQ8_KRhsRange(
    comptime rhs_dtype: DType,
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const types.QuantizedMatmulRhsRowsFor(rhs_dtype),
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
    rhs: *const types.QuantizedMatmulRhsRowsFor(rhs_dtype),
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    // Multi-row LHS on iq4_xs: the generic tile re-decodes every weight
    // block once per LHS row; the decoded tile hoists the table decode out
    // of the row loop (identical float sequence — bitwise equal, pinned by
    // the randomized tile parity test in this file).
    if (comptime rhs_dtype == .iq4_xs) {
        if (r1 - r0 > 1) return matmulIQ4_XSTileDecoded(out, lhs_blocks, rhs, n, r0, r1, c0, c1);
    }
    TableQ8_KTile(rhs_dtype)(out, lhs_blocks, rhs, n, r0, r1, c0, c1);
}

/// The Q8_K-activation table tile of `rhs_dtype`.
fn TableQ8_KTile(comptime rhs_dtype: DType) common.TileFn(BlockQ8_K, types.QuantizedMatmulRhsRowsFor(rhs_dtype)) {
    return common.RowOuterTile(BlockQ8_K, types.QuantizedMatmulRhsRowsFor(rhs_dtype), table_q8_k_col_block, .{
        .dot = struct {
            fn dot(w: *const dtype_mod.Storage(rhs_dtype), a: *const BlockQ8_K) f32 {
                return dotTableQ8_K(rhs_dtype, w, a);
            }
        }.dot,
    });
}

/// LHS rows per decoded-tile pass: bounds the strided Q8_K activation
/// window a decoded weight block is reused across (64 rows x 292 B ≈ 18 KB).
const iq4_xs_row_tile: usize = 64;

/// iq4_xs tile with the weight-block table decode hoisted out of the LHS row
/// loop: per (column, superblock) the 8 sub-block nibble lanes decode ONCE
/// (two `tbl` each) and serve every row of the tile, instead of once per
/// row. Float ops replicate the generic path's exact sequence — per
/// (row, col): out starts at 0 and accumulates one per-superblock partial
/// per step, each partial summing `(d·ls)·isum` terms in sub-block order —
/// so results are bitwise identical to the generic loop.
fn matmulIQ4_XSTileDecoded(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const types.QuantizedMatmulRhsRowsFor(.iq4_xs),
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.rows.blocks_per_row;
    var rt = r0;
    while (rt < r1) : (rt += iq4_xs_row_tile) {
        const rt_end = @min(rt + iq4_xs_row_tile, r1);
        var col = c0;
        // 4 columns per pass, sub-block-outer: per sub-block only the four
        // columns' two decoded lanes are live (8 vectors — fits the NEON
        // register file), and every streamed activation load feeds four
        // accumulator chains. The f32 partial tile accumulates each
        // (row, col)'s eight sub-block terms in sub-block order, so the
        // float sequence — and therefore the result — is bitwise identical
        // to the generic path.
        while (col + 4 <= c1) : (col += 4) {
            for (rt..rt_end) |row| {
                inline for (0..4) |c| out[row * n + col + c] = 0;
            }
            var bsum_tile: [iq4_xs_row_tile][4]f32 = undefined;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                var wb: [4]*const dtype_mod.BlockIQ4_XS = undefined;
                var wd: [4]f32 = undefined;
                inline for (0..4) |c| {
                    wb[c] = &rhs.rows.blocks[(col + c) * blocks_per_row + block_index];
                    wd[c] = f16BitsToF32(wb[c].d);
                }
                for (rt..rt_end) |row| {
                    bsum_tile[row - rt] = .{ 0, 0, 0, 0 };
                }
                inline for (0..8) |ib| {
                    var wlo: [4]common.QKV16i8 = undefined;
                    var whi: [4]common.QKV16i8 = undefined;
                    var lsf: [4]f32 = undefined;
                    inline for (0..4) |c| {
                        const nib = nibbleTableBytes(&tables.kvalues_iq4nl, wb[c].qs[16 * ib ..][0..16]);
                        wlo[c] = nib.lo;
                        whi[c] = nib.hi;
                        const low = (wb[c].scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0x0f;
                        const high = ((wb[c].scales_h >> @intCast(2 * ib)) & 3) << 4;
                        const ls: i32 = @intCast(@as(u16, low) | high);
                        lsf[c] = @floatFromInt(ls - 32);
                    }
                    for (rt..rt_end) |row| {
                        const ab = &lhs_blocks[row * blocks_per_row + block_index];
                        const a_lo: common.QKV16i8 = @bitCast(ab.qs[32 * ib ..][0..16].*);
                        const a_hi: common.QKV16i8 = @bitCast(ab.qs[32 * ib + 16 ..][0..16].*);
                        const bt = &bsum_tile[row - rt];
                        inline for (0..4) |c| {
                            var acc: common.QKV4i32 = @splat(0);
                            acc = common.sdotI8x16(acc, wlo[c], a_lo);
                            acc = common.sdotI8x16(acc, whi[c], a_hi);
                            bt[c] += (wd[c] * ab.d) * lsf[c] * @as(f32, @floatFromInt(@reduce(.Add, acc)));
                        }
                    }
                }
                for (rt..rt_end) |row| {
                    inline for (0..4) |c| out[row * n + col + c] += bsum_tile[row - rt][c];
                }
            }
        }
        while (col < c1) : (col += 1) {
            const col_blocks = rhs.rows.blocks[col * blocks_per_row ..][0..blocks_per_row];
            for (rt..rt_end) |row| out[row * n + col] = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const wb = &col_blocks[block_index];
                const wd = f16BitsToF32(wb.d);
                var wlo: [8]common.QKV16i8 = undefined;
                var whi: [8]common.QKV16i8 = undefined;
                var lsf: [8]f32 = undefined;
                inline for (0..8) |ib| {
                    const nib = nibbleTableBytes(&tables.kvalues_iq4nl, wb.qs[16 * ib ..][0..16]);
                    wlo[ib] = nib.lo;
                    whi[ib] = nib.hi;
                    const low = (wb.scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0x0f;
                    const high = ((wb.scales_h >> @intCast(2 * ib)) & 3) << 4;
                    const ls: i32 = @intCast(@as(u16, low) | high);
                    lsf[ib] = @floatFromInt(ls - 32);
                }
                for (rt..rt_end) |row| {
                    const ab = &lhs_blocks[row * blocks_per_row + block_index];
                    const d = wd * ab.d;
                    var bsum: f32 = 0;
                    inline for (0..8) |ib| {
                        var acc: common.QKV4i32 = @splat(0);
                        acc = common.sdotI8x16(acc, wlo[ib], @bitCast(ab.qs[32 * ib ..][0..16].*));
                        acc = common.sdotI8x16(acc, whi[ib], @bitCast(ab.qs[32 * ib + 16 ..][0..16].*));
                        bsum += d * lsf[ib] * @as(f32, @floatFromInt(@reduce(.Add, acc)));
                    }
                    out[row * n + col] += bsum;
                }
            }
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

fn dotIQ4_NLQ8_0(w: *const dtype_mod.BlockIQ4_NL, a: []const BlockQ8_0) f32 {
    std.debug.assert(a.len == 1);
    const ab = &a[0];
    const isum = dotNibbleTable32Q8(&tables.kvalues_iq4nl, &w.qs, ab.qs[0..16], ab.qs[16..32]);
    return f16BitsToF32(w.d) * f16BitsToF32(ab.d) * @as(f32, @floatFromInt(isum));
}

fn dotMXFP4Q8_0(w: *const dtype_mod.BlockMXFP4, a: []const BlockQ8_0) f32 {
    std.debug.assert(a.len == 1);
    const ab = &a[0];
    const isum = dotNibbleTable32Q8(&tables.kvalues_mxfp4, &w.qs, ab.qs[0..16], ab.qs[16..32]);
    return common.e8m0ToF32Half(w.e) * f16BitsToF32(ab.d) * @as(f32, @floatFromInt(isum));
}

fn dotNVFP4Q8_0(w: *const dtype_mod.BlockNVFP4, a: []const BlockQ8_0) f32 {
    std.debug.assert(a.len == types.nvfp4_block_size / types.q8_0_block_size);
    var sum: f32 = 0;
    for (0..types.nvfp4_block_size / types.nvfp4_subblock_size) |subblock| {
        const ab = &a[(subblock * types.nvfp4_subblock_size) / types.q8_0_block_size];
        const a_offset = (subblock * types.nvfp4_subblock_size) % types.q8_0_block_size;
        const qs = w.qs[subblock * (types.nvfp4_subblock_size / 2) ..][0 .. types.nvfp4_subblock_size / 2];
        const isum = dotNibbleTable16Q8(&tables.kvalues_mxfp4, qs, ab.qs[a_offset..][0..types.nvfp4_subblock_size]);
        sum += ue4m3ToF32(w.d[subblock]) * f16BitsToF32(ab.d) * @as(f32, @floatFromInt(isum));
    }
    return sum;
}

fn dotIQ2_XXSQ8_K(w: *const dtype_mod.BlockIQ2_XXS, a: *const BlockQ8_K) f32 {
    // Hot path: the four 8-value lanes of each 32-subblock assemble into one
    // 32-byte signed-weight vector and dot through sdot/vpdpbusd (two 16-byte
    // granules, one horizontal reduce) instead of four widen-multiply-reduce
    // rounds. `isum` is integer-exact either way and the float combination
    // below is unchanged, so the result is BITWISE identical to the lane
    // formulation (pinned by the parity test in quant_tests.zig).
    const d = f16BitsToF32(w.d) * a.d;
    var sum: f32 = 0;
    var offset: usize = 0;
    for (0..qk_k_block_size / 32) |ib32| {
        const aux0 = readU32FromU16s(w.qs[4 * ib32 ..][0..2]);
        const aux1 = readU32FromU16s(w.qs[4 * ib32 + 2 ..][0..2]);
        const db = d * (0.5 + @as(f32, @floatFromInt(aux1 >> 28))) * 0.25;
        var wbytes: [32]i8 = undefined;
        inline for (0..4) |lane| {
            const grid = gridU64Vector(&tables.iq2xxs_grid, byteFromU32(aux0, lane));
            const signs = tables.ksigns_iq2xs[(aux1 >> @intCast(7 * lane)) & 127];
            wbytes[8 * lane ..][0..8].* = signedGridBytes8(grid, signs);
        }
        const isum = dotSignedWeights32(&wbytes, a.qs[offset..][0..32]);
        offset += 32;
        sum += db * @as(f32, @floatFromInt(isum));
    }
    return sum;
}

/// Apply a ksigns septet to 8 unsigned grid magnitudes: (g ^ m) - m with
/// m in {0, -1} per element (two's-complement negate where the bit is set).
/// Grid magnitudes are < 128, so the i8 negate cannot overflow.
fn signedGridBytes8(grid: common.QKV8u8, signs: u8) [8]i8 {
    const active = (@as(common.QKV8u8, @splat(signs)) & qk_v8_sign_masks) != @as(common.QKV8u8, @splat(0));
    const mask = @select(i8, active, @as(common.QKV8i8, @splat(-1)), @as(common.QKV8i8, @splat(0)));
    const g: common.QKV8i8 = @intCast(grid);
    return (g ^ mask) - mask;
}

/// isum = w . a over 32 signed bytes: two sdot granules, one reduce.
fn dotSignedWeights32(wbytes: *const [32]i8, a_qs: *const [32]i8) i32 {
    var acc: common.QKV4i32 = @splat(0);
    acc = common.sdotI8x16(acc, @bitCast(wbytes[0..16].*), @bitCast(a_qs[0..16].*));
    acc = common.sdotI8x16(acc, @bitCast(wbytes[16..32].*), @bitCast(a_qs[16..32].*));
    return @reduce(.Add, acc);
}

fn dotIQ2_XSQ8_K(w: *const dtype_mod.BlockIQ2_XS, a: *const BlockQ8_K) f32 {
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

fn dotIQ2_SQ8_K(w: *const dtype_mod.BlockIQ2_S, a: *const BlockQ8_K) f32 {
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
    const grid_i16: common.QKV8i16 = @intCast(gridU64Vector(table, grid_index));
    return dotI16I8x8(grid_i16 * signsVector8(signs), a.qs[offset..][0..8]);
}

fn dotIQ3_XXSQ8_K(w: *const dtype_mod.BlockIQ3_XXS, a: *const BlockQ8_K) f32 {
    // Same sdot restructure as dotIQ2_XXSQ8_K — integer isum, unchanged
    // float combination, bitwise identical to the lane formulation.
    const d = f16BitsToF32(w.d) * a.d;
    var sum: f32 = 0;
    var qs_index: usize = 0;
    var offset: usize = 0;
    const scales_and_signs = qk_k_block_size / 4;
    for (0..qk_k_block_size / 32) |ib32| {
        const aux = readU32Bytes(w.qs[scales_and_signs + 4 * ib32 ..][0..4]);
        const db = d * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.5;
        var wbytes: [32]i8 = undefined;
        inline for (0..4) |lane| {
            const grid = gridU32PairVector(&tables.iq3xxs_grid, w.qs[qs_index + 2 * lane], w.qs[qs_index + 2 * lane + 1]);
            const signs = tables.ksigns_iq2xs[(aux >> @intCast(7 * lane)) & 127];
            wbytes[8 * lane ..][0..8].* = signedGridBytes8(grid, signs);
        }
        const isum = dotSignedWeights32(&wbytes, a.qs[offset..][0..32]);
        offset += 32;
        sum += db * @as(f32, @floatFromInt(isum));
        qs_index += 8;
    }
    return sum;
}

fn dotIQ3_SQ8_K(w: *const dtype_mod.BlockIQ3_S, a: *const BlockQ8_K) f32 {
    // Hot path: the four 8-value lanes of each 32-subblock assemble into one
    // 32-byte signed-weight vector (grid magnitudes ±sign via the two's-
    // complement mask trick) and dot through sdot/vpdpbusd. The per-subblock
    // integer sum and the float combination are unchanged, so the result is
    // BITWISE identical to the per-lane reference below (pinned by the
    // randomized parity test in this file).
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

        inline for (0..2) |half| {
            var wbytes: [32]i8 = undefined;
            inline for (0..4) |lane| {
                const qh = w.qh[qh_index + half];
                const grid1: usize = @as(usize, w.qs[qs_index + 2 * lane]) | ((@as(usize, qh) << @intCast(8 - 2 * lane)) & 256);
                const grid2: usize = @as(usize, w.qs[qs_index + 2 * lane + 1]) | ((@as(usize, qh) << @intCast(7 - 2 * lane)) & 256);
                const grid = gridU32PairVector(&tables.iq3s_grid, grid1, grid2);
                wbytes[8 * lane ..][0..8].* = signedGridBytes8(grid, w.signs[signs_index + lane]);
            }
            const isum = dotSignedWeights32(&wbytes, a.qs[offset..][0..32]);
            sum += (if (half == 0) db1 else db2) * @as(f32, @floatFromInt(isum));
            offset += 32;
            qs_index += 8;
            signs_index += 4;
        }
        qh_index += 2;
    }
    return sum;
}

/// Per-lane reference formulation of `dotIQ3_SQ8_K` (the parity oracle; the
/// integer subblock sums are exact in both, so the two must agree bitwise).
fn dotIQ3_SQ8_KRef(w: *const dtype_mod.BlockIQ3_S, a: *const BlockQ8_K) f32 {
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
    const grid_i16: common.QKV8i16 = @intCast(gridU32PairVector(table, grid1, grid2));
    return dotI16I8x8(grid_i16 * signsVector8(signs), a.qs[offset..][0..8]);
}

fn dotIQ1_SQ8_K(w: *const dtype_mod.BlockIQ1_S, a: *const BlockQ8_K) f32 {
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

fn dotIQ1_MQ8_K(w: *const dtype_mod.BlockIQ1_M, a: *const BlockQ8_K) f32 {
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
    const q_i8: common.QKV8i8 = @bitCast(a.qs[offset..][0..8].*);
    const q_i16: common.QKV8i16 = @intCast(q_i8);
    return .{
        .grid = dotI16I16x8(gridI8Vector(&tables.iq1s_grid, grid_index), q_i16),
        .q = reduceI16x8(q_i16),
    };
}

fn dotIQ4_XSQ8_K(w: *const dtype_mod.BlockIQ4_XS, a: *const BlockQ8_K) f32 {
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

fn dotTQ1_0Q8_K(w: *const dtype_mod.BlockTQ1_0, a: *const BlockQ8_K) f32 {
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

fn dotTQ2_0Q8_K(w: *const dtype_mod.BlockTQ2_0, a: *const BlockQ8_K) f32 {
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

/// One Q2_0 weight row against one Q8_0 activation row — the scalar
/// reference for the hot kernels in ternary.zig. Accumulation contract:
/// a FIXED 4-lane structure (lane k carries sub-block k of every block:
/// lane_k += d0 * d1_k * isum_k) folded pairwise ((l0+l1)+(l2+l3)) once at
/// the end — the same shape the hot kernel computes with one vector FMA per
/// 128-block, so hot and cold are bitwise identical. isum_k is the exact
/// per-32 integer sum((q-1)*a).
pub fn dotQ2_0RowQ8_0(wblocks: []const dtype_mod.BlockQ2_0, arow: []const BlockQ8_0) f32 {
    const sub_per_block = types.q2_0_block_size / types.q8_0_block_size; // 4
    std.debug.assert(arow.len == wblocks.len * sub_per_block);
    var lanes = [4]f32{ 0, 0, 0, 0 };
    for (wblocks, 0..) |*w, bi| {
        const d0 = f16BitsToF32(w.d);
        inline for (0..sub_per_block) |k| {
            const ab = &arow[bi * sub_per_block + k];
            var isum: i32 = 0;
            const qs = w.qs[k * 8 ..][0..8];
            var offset: usize = 0;
            for (qs) |byte| {
                inline for (0..4) |slot| {
                    const q = @as(i32, (byte >> (2 * slot)) & 0x3) - 1;
                    isum += q * @as(i32, ab.qs[offset + slot]);
                }
                offset += 4;
            }
            lanes[k] += d0 * f16BitsToF32(ab.d) * @as(f32, @floatFromInt(isum));
        }
    }
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
}

/// Reference Q2_0 matmul tile (scalar-backend path; the hot kernel's
/// bitwise oracle). RHS convention: rhs row c is output column c.
fn matmulQ2_0RhsRefTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const types.QuantizedMatmulRhsQ2_0,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const sub_blocks_per_row = rhs.rows.blocks_per_row * (types.q2_0_block_size / types.q8_0_block_size);
    var r = r0;
    while (r < r1) : (r += 1) {
        const arow = lhs_blocks[r * sub_blocks_per_row ..][0..sub_blocks_per_row];
        var c = c0;
        while (c < c1) : (c += 1) {
            out[r * n + c] = dotQ2_0RowQ8_0(rhs.columnBlocks(c), arow);
        }
    }
}

pub const matmulQ2_0RhsRefRange = common.RangeFromTile(matmulQ2_0RhsRefTile);

fn dotQ1_0Q8_0(w: *const dtype_mod.BlockQ1_0, a: []const BlockQ8_0) f32 {
    std.debug.assert(a.len == types.q1_0_block_size / types.q8_0_block_size);
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

fn dotQ4_1Q8_1(w: *const dtype_mod.BlockQ4_1, a: *const dtype_mod.BlockQ8_1) f32 {
    var isum: i32 = 0;
    for (w.qs, 0..) |q, j| {
        isum += @as(i32, q & 0x0f) * @as(i32, a.qs[j]);
        isum += @as(i32, q >> 4) * @as(i32, a.qs[types.q4_1_block_size / 2 + j]);
    }
    const d = f16BitsToF32(w.dm[0]) * f16BitsToF32(a.ds[0]);
    const m = f16BitsToF32(w.dm[1]) * f16BitsToF32(a.ds[1]);
    return d * @as(f32, @floatFromInt(isum)) + m;
}

fn dotQ5_0Q8_0(w: *const dtype_mod.BlockQ5_0, a: *const BlockQ8_0) f32 {
    const qh = readQh(&w.qh);
    var isum: i32 = 0;
    for (w.qs, 0..) |q, j| {
        const xh0: u8 = @intCast(((qh >> @intCast(j)) << 4) & 0x10);
        const xh1: u8 = @intCast((qh >> @intCast(j + 12)) & 0x10);
        const x0: i32 = @as(i32, (q & 0x0f) | xh0) - 16;
        const x1: i32 = @as(i32, (q >> 4) | xh1) - 16;
        isum += x0 * @as(i32, a.qs[j]);
        isum += x1 * @as(i32, a.qs[types.q5_0_block_size / 2 + j]);
    }
    const d = f16BitsToF32(w.d) * f16BitsToF32(a.d);
    return d * @as(f32, @floatFromInt(isum));
}

fn dotQ5_1Q8_1(w: *const dtype_mod.BlockQ5_1, a: *const dtype_mod.BlockQ8_1) f32 {
    const qh = readQh(&w.qh);
    var isum: i32 = 0;
    for (w.qs, 0..) |q, j| {
        const xh0: u8 = @intCast(((qh >> @intCast(j)) << 4) & 0x10);
        const xh1: u8 = @intCast((qh >> @intCast(j + 12)) & 0x10);
        const x0: u8 = (q & 0x0f) | xh0;
        const x1: u8 = (q >> 4) | xh1;
        isum += @as(i32, x0) * @as(i32, a.qs[j]);
        isum += @as(i32, x1) * @as(i32, a.qs[types.q5_1_block_size / 2 + j]);
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

const qk_v8_sign_masks: common.QKV8u8 = .{ 1, 2, 4, 8, 16, 32, 64, 128 };

const NibbleTableVectors = struct {
    lo: QKV16i16,
    hi: QKV16i16,
};

fn dotI16I8x16(w: QKV16i16, a_qs: *const [16]i8) i32 {
    const a_i8: common.QKV16i8 = @bitCast(a_qs.*);
    const a_i16: QKV16i16 = @intCast(a_i8);
    return dotI16I16x16(w, a_i16);
}

fn dotI16I8x8(w: common.QKV8i16, a_qs: *const [8]i8) i32 {
    const a_i8: common.QKV8i8 = @bitCast(a_qs.*);
    const a_i16: common.QKV8i16 = @intCast(a_i8);
    return dotI16I16x8(w, a_i16);
}

fn dotI16I16x16(w: QKV16i16, a: QKV16i16) i32 {
    const product_i16 = w * a;
    const product_i32: common.QKV16i32 = @intCast(product_i16);
    return @reduce(.Add, product_i32);
}

fn dotI16I16x8(w: common.QKV8i16, a: common.QKV8i16) i32 {
    const product_i16 = w * a;
    const product_i32: common.QKV8i32 = @intCast(product_i16);
    return @reduce(.Add, product_i32);
}

fn reduceI16x8(a: common.QKV8i16) i32 {
    const a_i32: common.QKV8i32 = @intCast(a);
    return @reduce(.Add, a_i32);
}

fn signsVector8(signs: u8) common.QKV8i16 {
    const active = (@as(common.QKV8u8, @splat(signs)) & qk_v8_sign_masks) != @as(common.QKV8u8, @splat(0));
    return @select(i16, active, @as(common.QKV8i16, @splat(-1)), @as(common.QKV8i16, @splat(1)));
}

fn gridU64Vector(comptime table: []const u64, index: usize) common.QKV8u8 {
    var out: [8]u8 = undefined;
    inline for (0..8) |j| out[j] = gridU64Byte(table, index, j);
    return @bitCast(out);
}

fn gridU32PairVector(comptime table: []const u32, grid1: usize, grid2: usize) common.QKV8u8 {
    var out: [8]u8 = undefined;
    inline for (0..4) |j| {
        out[j] = gridU32Byte(table, grid1, j);
        out[j + 4] = gridU32Byte(table, grid2, j);
    }
    return @bitCast(out);
}

fn gridI8Vector(comptime table: []const u64, index: usize) common.QKV8i16 {
    var out: [8]i8 = undefined;
    inline for (0..8) |j| out[j] = gridI8Byte(table, index, j);
    const v_i8: common.QKV8i8 = @bitCast(out);
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

/// Decode 16 packed nibbles through a 16-entry signed byte table into the
/// low-nibble and high-nibble i8 lanes: one `tbl` per lane on aarch64
/// instead of 32 scalar table loads. Table values fit i8, so the lanes feed
/// sdot directly.
fn nibbleTableBytes(comptime table: *const [16]i8, qs: *const [16]u8) struct { lo: common.QKV16i8, hi: common.QKV16i8 } {
    const tv: common.QKV16i8 = @bitCast(table.*);
    const q: QKV16u8 = @bitCast(qs.*);
    const lo_idx = q & @as(QKV16u8, @splat(0x0f));
    const hi_idx = q >> @as(QKV16u8, @splat(4));
    return .{ .lo = common.tblI8x16(tv, lo_idx), .hi = common.tblI8x16(tv, hi_idx) };
}

// Hot path: vector table decode + two sdot granules. `isum` is integer-exact
// either way, so the result is BITWISE identical to the scalar-decode i16
// reference below (pinned by the randomized parity test in this file).
fn dotNibbleTable32Q8(
    comptime table: *const [16]i8,
    qs: *const [16]u8,
    a_lo: *const [16]i8,
    a_hi: *const [16]i8,
) i32 {
    const w = nibbleTableBytes(table, qs);
    var acc: common.QKV4i32 = @splat(0);
    acc = common.sdotI8x16(acc, w.lo, @bitCast(a_lo.*));
    acc = common.sdotI8x16(acc, w.hi, @bitCast(a_hi.*));
    return @reduce(.Add, acc);
}

/// Scalar-decode reference formulation of `dotNibbleTable32Q8` (the parity
/// oracle; integer-exact, so the two must agree bitwise).
fn dotNibbleTable32Q8Ref(
    comptime table: *const [16]i8,
    qs: *const [16]u8,
    a_lo: *const [16]i8,
    a_hi: *const [16]i8,
) i32 {
    const w = nibbleTableVectors(table, qs);
    return dotI16I8x16(w.lo, a_lo) + dotI16I8x16(w.hi, a_hi);
}

fn dotNibbleTable16Q8(comptime table: *const [16]i8, qs: *const [8]u8, a_qs: *const [16]i8) i32 {
    const tv: common.QKV16i8 = @bitCast(table.*);
    var idx: [16]u8 = undefined;
    inline for (0..8) |j| {
        idx[j] = qs[j] & 0x0f;
        idx[j + 8] = qs[j] >> 4;
    }
    var acc: common.QKV4i32 = @splat(0);
    acc = common.sdotI8x16(acc, common.tblI8x16(tv, @bitCast(idx)), @bitCast(a_qs.*));
    return @reduce(.Add, acc);
}

/// Scalar-decode reference formulation of `dotNibbleTable16Q8`.
fn dotNibbleTable16Q8Ref(comptime table: *const [16]i8, qs: *const [8]u8, a_qs: *const [16]i8) i32 {
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
    const xi_u16 = (@as(common.QKV16u16, @intCast(wrapped)) * @as(common.QKV16u16, @splat(3))) >> @as(common.QKV16u16, @splat(8));
    const xi_i16: QKV16i16 = @intCast(xi_u16);
    return dotI16I8x16(xi_i16 - @as(QKV16i16, @splat(1)), a_qs);
}

const PreparedQ8_0Block = struct {
    lo: common.Q4V16i16,
    hi: common.Q4V16i16,
    scale: f32,
};

fn prepareQ8_0Block(a: *const BlockQ8_0) PreparedQ8_0Block {
    const a_lo_i8: common.Q4V16i8 = @bitCast(a.qs[0 .. types.q4_0_block_size / 2].*);
    const a_hi_i8: common.Q4V16i8 = @bitCast(a.qs[types.q4_0_block_size / 2 .. types.q4_0_block_size].*);
    return .{
        .lo = @intCast(a_lo_i8),
        .hi = @intCast(a_hi_i8),
        .scale = f16BitsToF32(a.d),
    };
}

fn dotQ4_0Q8_0(w: *const dtype_mod.BlockQ4_0, a: *const BlockQ8_0) f32 {
    return dotQ4_0PreparedQ8_0(w, prepareQ8_0Block(a));
}

fn dotQ4_0PreparedQ8_0(w: *const dtype_mod.BlockQ4_0, a: PreparedQ8_0Block) f32 {
    const q: common.Q4V16u8 = @bitCast(w.qs);
    const lo_i16: common.Q4V16i16 = @intCast(q & @as(common.Q4V16u8, @splat(0x0f)));
    const hi_i16: common.Q4V16i16 = @intCast(q >> @as(common.Q4V16u8, @splat(4)));
    const w_lo = lo_i16 - @as(common.Q4V16i16, @splat(8));
    const w_hi = hi_i16 - @as(common.Q4V16i16, @splat(8));

    const acc_lo: i16 = @reduce(.Add, w_lo * a.lo);
    const acc_hi: i16 = @reduce(.Add, w_hi * a.hi);
    const acc: i32 = @as(i32, acc_lo) + @as(i32, acc_hi);
    const d = f16BitsToF32(w.d) * a.scale;
    return @as(f32, @floatFromInt(acc)) * d;
}

fn dotQ2_KQ8_K(w: *const dtype_mod.BlockQ2_K, a: *const BlockQ8_K) f32 {
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

fn dotQ2_KGroupI32(w: *const dtype_mod.BlockQ2_K, a: *const BlockQ8_K, comptime chunk: usize, comptime section: usize, comptime half: usize) i32 {
    const q_offset = chunk * 32 + half * 16;
    const a_offset = chunk * 128 + section * 32 + half * 16;
    const q: QKV16u8 = @bitCast(w.qs[q_offset..][0..16].*);
    const vals = (q >> @as(QKV16u8, @splat(section * 2))) & @as(QKV16u8, @splat(0x03));
    const q_i16: QKV16i16 = @intCast(vals);
    const a_i8: common.QKV16i8 = @bitCast(a.qs[a_offset..][0..16].*);
    const a_i16: QKV16i16 = @intCast(a_i8);
    const product_i16 = q_i16 * a_i16;
    const product_i32: common.QKV16i32 = @intCast(product_i16);
    return @reduce(.Add, product_i32);
}

pub fn dequantizeBlockQ2_KInto(dst: *[qk_k_block_size]f32, src: *const dtype_mod.BlockQ2_K) void {
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

fn dotQ3_KQ8_K(w: *const dtype_mod.BlockQ3_K, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.d) * a.d;
    var sum: f32 = 0;
    inline for (0..16) |group| {
        const acc = dotQ3_KGroupI32(w, a, group);
        sum += @as(f32, @floatFromInt(acc)) * @as(f32, @floatFromInt(q3KScale(w, group)));
    }
    return sum * d;
}

fn dotQ3_KGroupI32(w: *const dtype_mod.BlockQ3_K, a: *const BlockQ8_K, comptime group: usize) i32 {
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
    const a_i8: common.QKV16i8 = @bitCast(a.qs[a_offset..][0..16].*);
    const a_i16: QKV16i16 = @intCast(a_i8);
    const product_i16 = q_i16 * a_i16;
    const product_i32: common.QKV16i32 = @intCast(product_i16);
    return @reduce(.Add, product_i32);
}

pub fn dequantizeBlockQ3_KInto(dst: *[qk_k_block_size]f32, src: *const dtype_mod.BlockQ3_K) void {
    const d = f16BitsToF32(src.d);
    var index: usize = 0;
    while (index < qk_k_block_size) : (index += 1) {
        const group = index / 16;
        dst[index] = d * @as(f32, @floatFromInt(q3KScale(src, group))) *
            @as(f32, @floatFromInt(q3KValue(src, index)));
    }
}

fn q2KValue(w: *const dtype_mod.BlockQ2_K, index: usize) u8 {
    const chunk = index / 128;
    const local = index % 128;
    const section = local / 32;
    const half = (local % 32) / 16;
    const offset = local % 16;
    const byte = w.qs[chunk * 32 + half * 16 + offset];
    return (byte >> @intCast(section * 2)) & 0x03;
}

fn q3KScale(w: *const dtype_mod.BlockQ3_K, index: usize) i8 {
    const low = if (index < 8)
        w.scales[index] & 0x0f
    else
        w.scales[index - 8] >> 4;
    const high = (w.scales[8 + index % 4] >> @intCast(2 * (index / 4))) & 0x03;
    const combined: i16 = @as(i16, low) | (@as(i16, high) << 4);
    return @intCast(combined - 32);
}

fn q3KValue(w: *const dtype_mod.BlockQ3_K, index: usize) i8 {
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

fn fillQ8_1Pattern(block: *dtype_mod.BlockQ8_1) void {
    var sum: i32 = 0;
    for (&block.qs, 0..) |*q, i| {
        q.* = @intCast(@as(i32, @intCast(i % 17)) - 8);
        sum += q.*;
    }
    block.ds = .{ f32ToF16Bits(1), f32ToF16Bits(@floatFromInt(sum)) };
}

fn fillQ1_0Pattern(block: *dtype_mod.BlockQ1_0) void {
    block.d = f32ToF16Bits(1);
    for (&block.qs, 0..) |*q, i| q.* = if (i % 2 == 0) 0b1010_0101 else 0b0101_1010;
}

fn fillQ2_0Pattern(block: *dtype_mod.BlockQ2_0) void {
    block.d = f32ToF16Bits(1);
    // Walks every 2-bit code including 3 (+2d) — the wire contract allows it
    // even though the reference encoder only emits {0,1,2}.
    for (&block.qs, 0..) |*q, i| q.* = @truncate(i *% 57 +% 0b11_10_01_00);
}

fn fillQ4_1Pattern(block: *dtype_mod.BlockQ4_1) void {
    block.dm = .{ f32ToF16Bits(1), f32ToF16Bits(0) };
    for (&block.qs, 0..) |*q, i| {
        const lo: u8 = @intCast(i % 16);
        const hi: u8 = @intCast((i + 5) % 16);
        q.* = lo | (hi << 4);
    }
}

fn setQ5_0Value(block: *dtype_mod.BlockQ5_0, index: usize, value: i8) void {
    const encoded: u8 = @intCast(@as(i16, value) + 16);
    const byte_index = index % (types.q5_0_block_size / 2);
    if (index < types.q5_0_block_size / 2) {
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

fn fillQ5_0Pattern(block: *dtype_mod.BlockQ5_0) void {
    block.d = f32ToF16Bits(1);
    @memset(&block.qh, 0);
    @memset(&block.qs, 0);
    for (0..types.q5_0_block_size) |i| setQ5_0Value(block, i, @intCast(@as(i32, @intCast(i % 23)) - 11));
}

fn setQ5_1Value(block: *dtype_mod.BlockQ5_1, index: usize, value: u8) void {
    const byte_index = index % (types.q5_1_block_size / 2);
    if (index < types.q5_1_block_size / 2) {
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

fn fillQ5_1Pattern(block: *dtype_mod.BlockQ5_1) void {
    block.dm = .{ f32ToF16Bits(1), f32ToF16Bits(0) };
    @memset(&block.qh, 0);
    @memset(&block.qs, 0);
    for (0..types.q5_1_block_size) |i| setQ5_1Value(block, i, @intCast((i * 7) % 32));
}

fn fillQ2KPattern(block: *dtype_mod.BlockQ2_K) void {
    block.dm = .{ f32ToF16Bits(1), f32ToF16Bits(0) };
    for (&block.scales, 0..) |*scale, i| scale.* = @intCast((i % 7) + 1);
    for (&block.qs, 0..) |*q, i| {
        q.* = @intCast((i % 4) | (((i + 1) % 4) << 2) | (((i + 2) % 4) << 4) | (((i + 3) % 4) << 6));
    }
}

fn setQ3KScale(block: *dtype_mod.BlockQ3_K, index: usize, scale: i8) void {
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

fn setQ3KValue(block: *dtype_mod.BlockQ3_K, index: usize, value: i8) void {
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

fn fillQ3KPattern(block: *dtype_mod.BlockQ3_K) void {
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
    var q1: dtype_mod.BlockQ1_0 = undefined;
    fillQ1_0Pattern(&q1);
    var q8 = [_]BlockQ8_0{undefined} ** (types.q1_0_block_size / types.q8_0_block_size);
    for (&q8) |*block| fillQ8_0Pattern(block);

    var dense_w: [types.q1_0_block_size]f32 = undefined;
    try dequantizeRowQ1_0Into(&dense_w, &.{q1});
    var dense_a: [types.q1_0_block_size]f32 = undefined;
    for (&q8, 0..) |*block, i| {
        for (block.qs, 0..) |v, j| {
            dense_a[i * types.q8_0_block_size + j] = @as(f32, @floatFromInt(v)) * f16BitsToF32(block.d);
        }
    }

    try std.testing.expectEqual(common.dotDense(&dense_w, &dense_a), dotQ1_0Q8_0(&q1, &q8));

    var rhs_blocks = [_]dtype_mod.BlockQ1_0{ q1, q1 };
    var rhs = types.QuantizedMatmulRhsQ1_0{
        .rows = .{ .allocator = std.testing.allocator, .blocks = &rhs_blocks, .rows = 2, .cols = types.q1_0_block_size, .blocks_per_row = 1 },
        .k = types.q1_0_block_size,
        .n = 2,
    };
    var out: [2]f32 = undefined;
    matmulQ1_0RhsRange(&out, &q8, &rhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ1_0Q8_0(&q1, &q8), out[0]);
}

test "ggml_q2_0 dot and matmul consume loaded blocks" {
    var q2: dtype_mod.BlockQ2_0 = undefined;
    fillQ2_0Pattern(&q2);
    var q8 = [_]BlockQ8_0{undefined} ** (types.q2_0_block_size / types.q8_0_block_size);
    for (&q8) |*block| fillQ8_0Pattern(block);

    var dense_w: [types.q2_0_block_size]f32 = undefined;
    try dequantizeRowQ2_0Into(&dense_w, &.{q2});
    var dense_a: [types.q2_0_block_size]f32 = undefined;
    for (&q8, 0..) |*block, i| {
        for (block.qs, 0..) |v, j| {
            dense_a[i * types.q8_0_block_size + j] = @as(f32, @floatFromInt(v)) * f16BitsToF32(block.d);
        }
    }

    try std.testing.expectEqual(common.dotDense(&dense_w, &dense_a), dotQ2_0RowQ8_0(&.{q2}, &q8));

    var rhs_blocks = [_]dtype_mod.BlockQ2_0{ q2, q2 };
    var rhs = types.QuantizedMatmulRhsQ2_0{
        .rows = .{ .allocator = std.testing.allocator, .blocks = &rhs_blocks, .rows = 2, .cols = types.q2_0_block_size, .blocks_per_row = 1 },
        .k = types.q2_0_block_size,
        .n = 2,
    };
    var out: [2]f32 = undefined;
    matmulQ2_0RhsRefRange(&out, &q8, &rhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ2_0RowQ8_0(&.{q2}, &q8), out[0]);
}

test "ggml_q4_1 dot and matmul consume loaded blocks" {
    var q4: dtype_mod.BlockQ4_1 = undefined;
    fillQ4_1Pattern(&q4);
    var q8: dtype_mod.BlockQ8_1 = undefined;
    fillQ8_1Pattern(&q8);

    var dense_w: [types.q4_1_block_size]f32 = undefined;
    try dequantizeRowQ4_1Into(&dense_w, &.{q4});
    var dense_a: [types.q8_1_block_size]f32 = undefined;
    try dequantizeRowQ8_1Into(&dense_a, &.{q8});

    try std.testing.expectEqual(common.dotDense(&dense_w, &dense_a), dotQ4_1Q8_1(&q4, &q8));

    var rhs_blocks = [_]dtype_mod.BlockQ4_1{ q4, q4 };
    var rhs = types.QuantizedMatmulRhsQ4_1{
        .rows = .{ .allocator = std.testing.allocator, .blocks = &rhs_blocks, .rows = 2, .cols = types.q4_1_block_size, .blocks_per_row = 1 },
        .k = types.q4_1_block_size,
        .n = 2,
    };
    var out: [2]f32 = undefined;
    matmulQ4_1RhsRange(&out, &.{q8}, &rhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ4_1Q8_1(&q4, &q8), out[0]);
}

test "ggml_q5_0 dot and matmul consume loaded blocks" {
    var q5: dtype_mod.BlockQ5_0 = undefined;
    fillQ5_0Pattern(&q5);
    var q8: BlockQ8_0 = undefined;
    fillQ8_0Pattern(&q8);

    var dense_w: [types.q5_0_block_size]f32 = undefined;
    try dequantizeRowQ5_0Into(&dense_w, &.{q5});
    var dense_a: [types.q8_0_block_size]f32 = undefined;
    try q8k.dequantizeRowQ8_0Into(&dense_a, &.{q8});

    try std.testing.expectEqual(common.dotDense(&dense_w, &dense_a), dotQ5_0Q8_0(&q5, &q8));

    var rhs_blocks = [_]dtype_mod.BlockQ5_0{ q5, q5 };
    var rhs = types.QuantizedMatmulRhsQ5_0{
        .rows = .{ .allocator = std.testing.allocator, .blocks = &rhs_blocks, .rows = 2, .cols = types.q5_0_block_size, .blocks_per_row = 1 },
        .k = types.q5_0_block_size,
        .n = 2,
    };
    var out: [2]f32 = undefined;
    matmulQ5_0RhsRange(&out, &.{q8}, &rhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ5_0Q8_0(&q5, &q8), out[0]);
}

test "ggml_q5_1 dot and matmul consume loaded blocks" {
    var q5: dtype_mod.BlockQ5_1 = undefined;
    fillQ5_1Pattern(&q5);
    var q8: dtype_mod.BlockQ8_1 = undefined;
    fillQ8_1Pattern(&q8);

    var dense_w: [types.q5_1_block_size]f32 = undefined;
    try dequantizeRowQ5_1Into(&dense_w, &.{q5});
    var dense_a: [types.q8_1_block_size]f32 = undefined;
    try dequantizeRowQ8_1Into(&dense_a, &.{q8});

    try std.testing.expectEqual(common.dotDense(&dense_w, &dense_a), dotQ5_1Q8_1(&q5, &q8));

    var rhs_blocks = [_]dtype_mod.BlockQ5_1{ q5, q5 };
    var rhs = types.QuantizedMatmulRhsQ5_1{
        .rows = .{ .allocator = std.testing.allocator, .blocks = &rhs_blocks, .rows = 2, .cols = types.q5_1_block_size, .blocks_per_row = 1 },
        .k = types.q5_1_block_size,
        .n = 2,
    };
    var out: [2]f32 = undefined;
    matmulQ5_1RhsRange(&out, &.{q8}, &rhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ5_1Q8_1(&q5, &q8), out[0]);
}

test "ggml_q2_k dot and matmul consume loaded blocks" {
    const allocator = std.testing.allocator;

    var q2: dtype_mod.BlockQ2_K = undefined;
    fillQ2KPattern(&q2);
    var q8: BlockQ8_K = undefined;
    q8k.fillQ8KPattern(&q8);

    var dense_w: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ2_KInto(&dense_w, &q2);
    var dense_a: [qk_k_block_size]f32 = undefined;
    q8k.dequantizeBlockQ8_KInto(&dense_a, &q8);

    try std.testing.expectEqual(common.dotDense(&dense_w, &dense_a), dotQ2_KQ8_K(&q2, &q8));

    var rhs_blocks = [_]dtype_mod.BlockQ2_K{ q2, q2 };
    var qrhs = try q8k.quantizedMatmulRhsQ2_KFromBlocks(allocator, qk_k_block_size, 2, &rhs_blocks);
    defer qrhs.deinit();
    var out: [2]f32 = undefined;
    matmulQ2_KRhsRange(&out, &.{q8}, &qrhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ2_KQ8_K(&q2, &q8), out[0]);
}

test "ggml_q3_k dot and matmul consume loaded blocks" {
    const allocator = std.testing.allocator;

    var q3: dtype_mod.BlockQ3_K = undefined;
    fillQ3KPattern(&q3);
    var q8: BlockQ8_K = undefined;
    q8k.fillQ8KPattern(&q8);

    var dense_w: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ3_KInto(&dense_w, &q3);
    var dense_a: [qk_k_block_size]f32 = undefined;
    q8k.dequantizeBlockQ8_KInto(&dense_a, &q8);

    try std.testing.expectEqual(common.dotDense(&dense_w, &dense_a), dotQ3_KQ8_K(&q3, &q8));

    var rhs_blocks = [_]dtype_mod.BlockQ3_K{ q3, q3 };
    var qrhs = try q8k.quantizedMatmulRhsQ3_KFromBlocks(allocator, qk_k_block_size, 2, &rhs_blocks);
    defer qrhs.deinit();
    var out: [2]f32 = undefined;
    matmulQ3_KRhsRange(&out, &.{q8}, &qrhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ3_KQ8_K(&q3, &q8), out[0]);
}

test {
    _ = @import("cold_tests.zig");
}

test "iq2_xxs and iq3_xxs sdot dots match the lane-based reference bitwise" {
    // The hot kernels assemble each 32-subblock's four grid lanes into one
    // signed byte vector and dot through sdot; the integer isum is exact and
    // the float combination order is unchanged, so equality must be BITWISE
    // against the original per-lane widen-multiply-reduce formulation.
    const RefLane = struct {
        fn dot(gb: [8]u8, signs: u8, a_qs: []const i8) i32 {
            var isum: i32 = 0;
            for (0..8) |j| {
                const sign: i32 = if ((signs & (@as(u8, 1) << @intCast(j))) != 0) -1 else 1;
                isum += @as(i32, gb[j]) * sign * @as(i32, a_qs[j]);
            }
            return isum;
        }
    };

    var prng = std.Random.DefaultPrng.init(0x51D07);
    const random = prng.random();
    for (0..64) |_| {
        var a: BlockQ8_K = undefined;
        a.d = (random.float(f32) - 0.5) * 0.2;
        for (&a.qs) |*v| v.* = random.intRangeAtMost(i8, -127, 127);
        for (&a.bsums, 0..) |*b, gi| {
            var t: i32 = 0;
            for (a.qs[gi * 16 ..][0..16]) |v| t += v;
            b.* = @intCast(t);
        }

        var w2: dtype_mod.BlockIQ2_XXS = undefined;
        w2.d = @intCast(random.intRangeAtMost(u16, 0x2c00, 0x3c00)); // sane f16 scale bits
        for (&w2.qs) |*q| q.* = random.int(u16);
        var w3: dtype_mod.BlockIQ3_XXS = undefined;
        w3.d = w2.d;
        for (&w3.qs) |*q| q.* = random.int(u8);

        // iq2_xxs reference
        {
            const d = f16BitsToF32(w2.d) * a.d;
            var want: f32 = 0;
            var offset: usize = 0;
            for (0..qk_k_block_size / 32) |ib32| {
                const aux0 = readU32FromU16s(w2.qs[4 * ib32 ..][0..2]);
                const aux1 = readU32FromU16s(w2.qs[4 * ib32 + 2 ..][0..2]);
                const db = d * (0.5 + @as(f32, @floatFromInt(aux1 >> 28))) * 0.25;
                var isum: i32 = 0;
                for (0..4) |lane| {
                    var gb: [8]u8 = undefined;
                    for (0..8) |j| gb[j] = gridU64Byte(&tables.iq2xxs_grid, byteFromU32(aux0, lane), j);
                    const signs = tables.ksigns_iq2xs[(aux1 >> @intCast(7 * lane)) & 127];
                    isum += RefLane.dot(gb, signs, a.qs[offset..][0..8]);
                    offset += 8;
                }
                want += db * @as(f32, @floatFromInt(isum));
            }
            try std.testing.expectEqual(want, dotIQ2_XXSQ8_K(&w2, &a));
        }
        // iq3_xxs reference
        {
            const d = f16BitsToF32(w3.d) * a.d;
            var want: f32 = 0;
            var qs_index: usize = 0;
            var offset: usize = 0;
            const scales_and_signs = qk_k_block_size / 4;
            for (0..qk_k_block_size / 32) |ib32| {
                const aux = readU32Bytes(w3.qs[scales_and_signs + 4 * ib32 ..][0..4]);
                const db = d * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.5;
                var isum: i32 = 0;
                for (0..4) |lane| {
                    var gb: [8]u8 = undefined;
                    const gv = gridU32PairVector(&tables.iq3xxs_grid, w3.qs[qs_index + 2 * lane], w3.qs[qs_index + 2 * lane + 1]);
                    const garr: [8]u8 = @bitCast(gv);
                    gb = garr;
                    const signs = tables.ksigns_iq2xs[(aux >> @intCast(7 * lane)) & 127];
                    isum += RefLane.dot(gb, signs, a.qs[offset..][0..8]);
                    offset += 8;
                }
                want += db * @as(f32, @floatFromInt(isum));
                qs_index += 8;
            }
            try std.testing.expectEqual(want, dotIQ3_XXSQ8_K(&w3, &a));
        }
    }
}

// ---------------------------------------------------------------------------
// Hot q4_0 row kernels: fused nibble-unpack dot and weighted accumulate
// without materializing the row (the q4 centroid-residual attention read
// path is DRAM-bound only if dequantization stays in registers).

/// dot(x, dequant(blocks)); x.len == blocks.len * 32.
pub fn vecDotQ4_0F32(blocks: []const dtype_mod.BlockQ4_0, x: []const f32) f32 {
    const V = @Vector(8, f32);
    const U = @Vector(8, u8);
    const I = @Vector(8, i16);
    var acc: V = @splat(0);
    for (blocks, 0..) |*block, bi| {
        const scale = f16BitsToF32(block.d);
        const base = bi * types.q4_0_block_size;
        var blk: V = @splat(0);
        inline for (0..2) |h| {
            const qv: U = block.qs[h * 8 ..][0..8].*;
            const lo: V = @floatFromInt(@as(I, @intCast(qv & @as(U, @splat(0xF)))) - @as(I, @splat(8)));
            const hi: V = @floatFromInt(@as(I, @intCast(qv >> @splat(4))) - @as(I, @splat(8)));
            blk = @mulAdd(V, lo, x[base + h * 8 ..][0..8].*, blk);
            blk = @mulAdd(V, hi, x[base + 16 + h * 8 ..][0..8].*, blk);
        }
        acc = @mulAdd(V, blk, @as(V, @splat(scale)), acc);
    }
    return @reduce(.Add, acc);
}

/// out (+)= weight * dequant(blocks); out.len == blocks.len * 32.
pub fn weightedQ4_0Row(comptime accumulate: bool, out: []f32, blocks: []const dtype_mod.BlockQ4_0, weight: f32) void {
    const V = @Vector(8, f32);
    const U = @Vector(8, u8);
    const I = @Vector(8, i16);
    for (blocks, 0..) |*block, bi| {
        const ws: V = @splat(weight * f16BitsToF32(block.d));
        const base = bi * types.q4_0_block_size;
        inline for (0..2) |h| {
            const qv: U = block.qs[h * 8 ..][0..8].*;
            const lo: V = @floatFromInt(@as(I, @intCast(qv & @as(U, @splat(0xF)))) - @as(I, @splat(8)));
            const hi: V = @floatFromInt(@as(I, @intCast(qv >> @splat(4))) - @as(I, @splat(8)));
            const dst_lo = out[base + h * 8 ..][0..8];
            const dst_hi = out[base + 16 + h * 8 ..][0..8];
            dst_lo.* = if (accumulate) @mulAdd(V, lo, ws, @as(V, dst_lo.*)) else lo * ws;
            dst_hi.* = if (accumulate) @mulAdd(V, hi, ws, @as(V, dst_hi.*)) else hi * ws;
        }
    }
}

test "nibble-table sdot dots match the scalar-decode reference bitwise" {
    var prng = std.Random.DefaultPrng.init(0x14C0FFEE);
    const rnd = prng.random();
    for (0..256) |_| {
        var qs32: [16]u8 = undefined;
        rnd.bytes(&qs32);
        var a_lo: [16]i8 = undefined;
        rnd.bytes(std.mem.asBytes(&a_lo));
        var a_hi: [16]i8 = undefined;
        rnd.bytes(std.mem.asBytes(&a_hi));
        try std.testing.expectEqual(
            dotNibbleTable32Q8Ref(&tables.kvalues_iq4nl, &qs32, &a_lo, &a_hi),
            dotNibbleTable32Q8(&tables.kvalues_iq4nl, &qs32, &a_lo, &a_hi),
        );
        try std.testing.expectEqual(
            dotNibbleTable32Q8Ref(&tables.kvalues_mxfp4, &qs32, &a_lo, &a_hi),
            dotNibbleTable32Q8(&tables.kvalues_mxfp4, &qs32, &a_lo, &a_hi),
        );

        var qs16: [8]u8 = undefined;
        rnd.bytes(&qs16);
        try std.testing.expectEqual(
            dotNibbleTable16Q8Ref(&tables.kvalues_mxfp4, &qs16, &a_lo),
            dotNibbleTable16Q8(&tables.kvalues_mxfp4, &qs16, &a_lo),
        );
    }
}

test "iq3_s sdot dot matches the per-lane reference bitwise" {
    var prng = std.Random.DefaultPrng.init(0x13C0FFEE);
    const rnd = prng.random();
    for (0..128) |_| {
        var w: dtype_mod.BlockIQ3_S = undefined;
        rnd.bytes(std.mem.asBytes(&w));
        w.d = f32ToF16Bits(rnd.float(f32) * 2.0 - 1.0);
        var a: BlockQ8_K = undefined;
        rnd.bytes(std.mem.asBytes(&a));
        a.d = rnd.float(f32) * 2.0 - 1.0;
        try std.testing.expectEqual(dotIQ3_SQ8_KRef(&w, &a), dotIQ3_SQ8_K(&w, &a));
    }
}

test "iq4_xs decoded tile matmul matches the per-row generic path bitwise" {
    var prng = std.Random.DefaultPrng.init(0x44C0FFEE);
    const rnd = prng.random();
    const m = 7;
    const cols = 5;
    const bpr = 3;

    var wblocks: [cols * bpr]dtype_mod.BlockIQ4_XS = undefined;
    rnd.bytes(std.mem.sliceAsBytes(wblocks[0..]));
    for (&wblocks) |*wb| wb.d = f32ToF16Bits(rnd.float(f32) * 2.0 - 1.0);
    var ablocks: [m * bpr]BlockQ8_K = undefined;
    rnd.bytes(std.mem.sliceAsBytes(ablocks[0..]));
    for (&ablocks) |*ab| ab.d = rnd.float(f32) * 2.0 - 1.0;

    const rhs = types.QuantizedMatmulRhsRowsFor(.iq4_xs){
        .rows = .{ .allocator = std.testing.allocator, .blocks = &wblocks, .rows = cols, .cols = bpr * qk_k_block_size, .blocks_per_row = bpr },
        .k = bpr * qk_k_block_size,
        .n = cols,
    };

    var out_tiled: [m * cols]f32 = undefined;
    matmulTableQ8_KRhsTile(.iq4_xs, &out_tiled, &ablocks, &rhs, cols, 0, m, 0, cols);

    // Single-row calls take the generic per-row loop (the decoded tile only
    // arms for multi-row LHS), so this is the reference formulation.
    var out_ref: [m * cols]f32 = undefined;
    for (0..m) |row| {
        matmulTableQ8_KRhsTile(.iq4_xs, &out_ref, &ablocks, &rhs, cols, row, row + 1, 0, cols);
    }
    try std.testing.expectEqualSlices(f32, &out_ref, &out_tiled);
}

/// The cold-family GEMM entry (`ops.QuantGemm`): the direct 32-block and
/// K-quant reference tiles, and the table-decoded families over Q8_0 or
/// Q8_K activations (the tile binds `g.weight` as the table dtype).
/// Output row stride is `rhs.n`. The Q2_0 scalar reference twin
/// (`matmulQ2_0RhsRefRange`) is the scalar provider's bespoke arm, outside
/// the seam.
pub fn gemm(comptime g: ops.QuantGemm, out: []f32, lhs: ops.LhsOf(g), rhs: ops.RhsOf(g), tile: ops.Tile) void {
    comptime g.check();
    const n = rhs.n;
    switch (comptime g.weight) {
        .q1_0 => matmulQ1_0RhsTile(out, lhs, rhs, n, tile.r0, tile.r1, tile.c0, tile.c1),
        .q4_0 => matmulQ4_0RhsTile(out, lhs, rhs, n, tile.r0, tile.r1, tile.c0, tile.c1),
        .q4_1 => matmulQ4_1RhsTile(out, lhs, rhs, n, tile.r0, tile.r1, tile.c0, tile.c1),
        .q5_0 => matmulQ5_0RhsTile(out, lhs, rhs, n, tile.r0, tile.r1, tile.c0, tile.c1),
        .q5_1 => matmulQ5_1RhsTile(out, lhs, rhs, n, tile.r0, tile.r1, tile.c0, tile.c1),
        .q2_k => matmulQ2_KRhsTile(out, lhs, rhs, n, tile.r0, tile.r1, tile.c0, tile.c1),
        .q3_k => matmulQ3_KRhsTile(out, lhs, rhs, n, tile.r0, tile.r1, tile.c0, tile.c1),
        .iq4_nl, .mxfp4, .nvfp4 => matmulTableQ8_0RhsTile(g.weight, out, lhs, rhs, n, tile.r0, tile.r1, tile.c0, tile.c1),
        .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_xs, .tq1_0 => matmulTableQ8_KRhsTile(g.weight, out, lhs, rhs, n, tile.r0, tile.r1, tile.c0, tile.c1),
        else => comptime unreachable,
    }
}
