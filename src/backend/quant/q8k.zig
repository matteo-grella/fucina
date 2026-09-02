//! Q8_K / Q8_0 encode-quantize-dequantize helpers, the Q8_Kx4 and Q8_Kx2Mmla
//! LHS packers, the K-quant `*FromBlocks` RHS constructors and the ggml
//! reference scale-search helpers. A leaf of the quant group that every
//! kernel child imports.

const std = @import("std");
const builtin = @import("builtin");
const dtype_mod = @import("../../dtype.zig");
const tensor = @import("../../tensor.zig");
const common = @import("common.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;

const QKV4f32 = common.QKV4f32;

const qk_k_block_size = types.qk_k_block_size;

/// The shared shell of the lane-pack RHS constructors
/// (`packMatmulRhs*x4/x8/x2Mmla` in the format files): validation, group
/// allocation, the group/block walk and the container literal. `packGroup`
/// packs one destination block from its `lanes` source column blocks; the
/// per-format layout stays beside its kernels. Byte-for-byte the walk each
/// constructor spelled out.
pub fn packLaneGroups(
    comptime lanes: usize,
    comptime SrcBlock: type,
    comptime Rhs: type,
    comptime packGroup: anytype,
    allocator: Allocator,
    blocks: []const SrcBlock,
    n: usize,
    k: usize,
    blocks_per_row: usize,
) !Rhs {
    if (n % lanes != 0) return tensor.TensorError.InvalidShape;
    const expected = if (comptime SrcBlock == dtype_mod.BlockQ8_0) try q8_0BlockCount(k) else try qkBlockCount(k);
    if (blocks_per_row != expected) return tensor.TensorError.InvalidShape;
    if (blocks.len != try types.checkedProduct(n, blocks_per_row)) return types.QuantizedFormatError.InvalidQuantizedLength;

    const group_count = n / lanes;
    const DstBlock = @typeInfo(@FieldType(Rhs, "blocks")).pointer.child;
    const packed_blocks = try allocator.alloc(DstBlock, try types.checkedProduct(group_count, blocks_per_row));
    errdefer allocator.free(packed_blocks);

    for (0..group_count) |group_i| {
        for (0..blocks_per_row) |block_i| {
            var cols: [lanes]*const SrcBlock = undefined;
            inline for (0..lanes) |lane| cols[lane] = &blocks[(lanes * group_i + lane) * blocks_per_row + block_i];
            packGroup(&packed_blocks[group_i * blocks_per_row + block_i], cols);
        }
    }

    return .{
        .allocator = allocator,
        .blocks = packed_blocks,
        .k = k,
        .n = n,
        .blocks_per_group = blocks_per_row,
    };
}

/// The shared x8 lane-pack body of `packGroupQ4_Kx8` / `packGroupQ5_Kx8`:
/// the eight columns' d/dmin and per-sub-block (scale, min), then `groups`
/// 256-byte qs groups (8 feature groups x 8 columns x 4 lanes) filled from
/// `value(block, group, feature_offset)`. Q4_K keeps its nibble pairs
/// packed (4 groups of two sub-blocks each); Q5_K stores the unpacked
/// 5-bit values (8 groups).
pub fn packGroupKx8(dst: anytype, cols: anytype, comptime groups: usize, comptime value: anytype) void {
    inline for (0..8) |col| {
        dst.d[col] = cols[col].dm[0];
        dst.dmin[col] = cols[col].dm[1];
    }

    for (0..8) |subblock| {
        inline for (0..8) |col| {
            const scale_min = getScaleMinK4(&cols[col].scales, subblock);
            dst.scales[subblock * 8 + col] = scale_min.scale;
            dst.mins[subblock * 8 + col] = scale_min.min;
        }
    }

    for (0..groups) |group| {
        for (0..8) |feature_group| {
            inline for (0..8) |col| {
                const block = cols[col];
                inline for (0..4) |lane| {
                    const feature_offset = feature_group * 4 + lane;
                    dst.qs[group * 256 + feature_group * 32 + col * 4 + lane] = @intCast(value(block, group, feature_offset));
                }
            }
        }
    }
}

pub fn quantizeRowQ8_0Into(dst: []dtype_mod.BlockQ8_0, src: []const f32) !void {
    const block_count = try q8_0BlockCount(src.len);
    if (dst.len != block_count) return types.QuantizedFormatError.InvalidQuantizedLength;
    quantizeRowQ8_0IntoUnchecked(dst, src);
}

/// `quantizeRowQ8_0Into` for callers that have already proven the lengths:
/// `src` is a whole number of q8_0 blocks and `dst` covers exactly that
/// count (asserted in safe builds). The validate-then-unchecked entry, so
/// pre-validated call sites (thread-task bodies) never suppress the checked
/// twin's error set with `catch unreachable`.
///
/// One `quantizeBlockQ8_0` per block.
pub fn quantizeRowQ8_0IntoUnchecked(dst: []dtype_mod.BlockQ8_0, src: []const f32) void {
    std.debug.assert(src.len % types.q8_0_block_size == 0);
    std.debug.assert(dst.len == src.len / types.q8_0_block_size);

    var block_index: usize = 0;
    while (block_index < dst.len) : (block_index += 1) {
        dst[block_index] = quantizeBlockQ8_0(src[block_index * types.q8_0_block_size ..][0..types.q8_0_block_size]);
    }
}

/// One Q8_0 block from 32 values, the body behind every Q8_0 activation
/// layout (rows, Q8_0x4 lanes, the fused SwiGLU quantizer). One `@Vector`
/// body on every architecture: 4-lane amax, scale, clamp, then
/// `common.roundHalfAwayFromZeroVec4ToI32` (fcvtas on aarch64, the portable
/// vector round elsewhere). Clamping before the round is equivalent to
/// rounding first because the bounds are integral, so every byte equals the
/// scalar `common.quantizeToI8` form (`x86dot_check.zig` pins the two
/// against each other on every ISA it runs on).
pub fn quantizeBlockQ8_0(src: *const [types.q8_0_block_size]f32) dtype_mod.BlockQ8_0 {
    var amaxv: QKV4f32 = @splat(0);
    inline for (0..8) |j| {
        const v: QKV4f32 = src[j * 4 ..][0..4].*;
        amaxv = @max(amaxv, @abs(v));
    }
    return quantizeBlockQ8_0Scaled(src, @reduce(.Max, amaxv));
}

/// The scale-and-round half of `quantizeBlockQ8_0` for a caller that
/// already holds the block's amax.
pub fn quantizeBlockQ8_0Scaled(src: *const [types.q8_0_block_size]f32, amax: f32) dtype_mod.BlockQ8_0 {
    const d = amax / 127.0;
    const inv_d: f32 = if (d == 0) 0 else 1.0 / d;

    var block: dtype_mod.BlockQ8_0 = undefined;
    block.d = common.f32ToF16Bits(d);
    inline for (0..8) |j| {
        const v: QKV4f32 = src[j * 4 ..][0..4].*;
        const scaled = v * @as(QKV4f32, @splat(inv_d));
        const clamped = @max(@as(QKV4f32, @splat(-127.0)), @min(@as(QKV4f32, @splat(127.0)), scaled));
        const q = common.roundHalfAwayFromZeroVec4ToI32(clamped);
        inline for (0..4) |lane| block.qs[j * 4 + lane] = @intCast(q[lane]);
    }
    return block;
}

pub fn dequantizeRowQ8_0Into(dst: []f32, src: []const dtype_mod.BlockQ8_0) !void {
    if (dst.len != try types.checkedProduct(src.len, types.q8_0_block_size)) return types.QuantizedFormatError.InvalidQuantizedLength;

    // Explicit 8-lane vectors: the q8_0 KV-cache attention path dequantizes
    // every K/V row it streams through this function, and the scalar
    // element loop compiles to per-element converts (~2.4x decode attention
    // cost vs f16). Widen-multiply in vector chunks instead; same values
    // bit-for-bit (each element is one exact i8->f32 convert and one f32
    // multiply in both forms).
    for (src, 0..) |block, block_index| {
        const d = common.f16BitsToF32(block.d);
        const dv: @Vector(8, f32) = @splat(d);
        const out = dst[block_index * types.q8_0_block_size ..][0..types.q8_0_block_size];
        inline for (0..types.q8_0_block_size / 8) |j| {
            const q: @Vector(8, i8) = block.qs[j * 8 ..][0..8].*;
            const w: @Vector(8, f32) = @floatFromInt(q);
            out[j * 8 ..][0..8].* = w * dv;
        }
    }
}
pub fn quantizeRowsQ8_0(allocator: Allocator, src: *const Tensor) !types.QuantizedRowsQ8_0 {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    const blocks_per_row = try q8_0BlockCount(cols);

    const blocks = try allocator.alloc(dtype_mod.BlockQ8_0, try types.checkedProduct(rows, blocks_per_row));
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
pub fn quantizeRowsQ8_0Into(blocks: []dtype_mod.BlockQ8_0, src: *const Tensor) !void {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    const blocks_per_row = try q8_0BlockCount(cols);
    if (blocks.len != try types.checkedProduct(rows, blocks_per_row)) return types.QuantizedFormatError.InvalidQuantizedLength;

    const data = try src.dataConstChecked();
    quantizeRowsQ8_0RangeInto(blocks, data, rows, cols, blocks_per_row, 0, rows);
}
/// Rows `[row_start, row_end)` of the `[rows, cols]` activation `data` into
/// their q8_0 blocks (`blocks` covers all `rows * blocks_per_row`; the
/// lengths are the caller's proof). Rows own disjoint blocks, so a pool
/// split over row ranges produces the serial call's bytes; allocation-free.
pub fn quantizeRowsQ8_0RangeInto(blocks: []dtype_mod.BlockQ8_0, data: []const f32, rows: usize, cols: usize, blocks_per_row: usize, row_start: usize, row_end: usize) void {
    std.debug.assert(row_end <= rows and data.len == rows * cols and blocks.len == rows * blocks_per_row);
    for (row_start..row_end) |row| {
        quantizeRowQ8_0IntoUnchecked(
            blocks[row * blocks_per_row ..][0..blocks_per_row],
            data[row * cols ..][0..cols],
        );
    }
}
pub fn dequantizeRowsQ8_0Into(dst: *Tensor, src: *const types.QuantizedRowsQ8_0) !void {
    const view = try dst.rankView(2);
    if (view.dim(0) != src.rows or view.dim(1) != src.cols) return tensor.TensorError.ShapeMismatch;

    const out = try dst.dataChecked();
    var row: usize = 0;
    while (row < src.rows) : (row += 1) {
        try dequantizeRowQ8_0Into(out[row * src.cols ..][0..src.cols], src.rowBlocks(row));
    }
}
pub fn getRowsQ8_0Into(dst: *Tensor, table: *const types.QuantizedRowsQ8_0, indices: []const usize) !void {
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
    return types.blockCountExact(types.q8_0_block_size, len);
}
pub fn qkBlockCount(len: usize) !usize {
    return types.blockCountExact(qk_k_block_size, len);
}
pub const blockCountExact = types.blockCountExact;

/// The owning compact-RHS constructor of a K-quant format: `n` columns of
/// `k/256` blocks each, copied into the container.
fn KQuantRhsFromBlocks(comptime Rhs: type) type {
    const Block = @typeInfo(@FieldType(Rhs, "blocks")).pointer.child;
    return struct {
        fn fromBlocks(allocator: Allocator, k: usize, n: usize, blocks: []const Block) !Rhs {
            const blocks_per_column = try qkBlockCount(k);
            if (blocks.len != try types.checkedProduct(n, blocks_per_column)) return types.QuantizedFormatError.InvalidQuantizedLength;
            const owned = try allocator.dupe(Block, blocks);
            return .{
                .allocator = allocator,
                .blocks = owned,
                .k = k,
                .n = n,
                .blocks_per_column = blocks_per_column,
            };
        }
    };
}

pub const quantizedMatmulRhsQ2_KFromBlocks = KQuantRhsFromBlocks(types.QuantizedMatmulRhsQ2_K).fromBlocks;
pub const quantizedMatmulRhsQ3_KFromBlocks = KQuantRhsFromBlocks(types.QuantizedMatmulRhsQ3_K).fromBlocks;
pub const quantizedMatmulRhsQ4_KFromBlocks = KQuantRhsFromBlocks(types.QuantizedMatmulRhsQ4_K).fromBlocks;
pub const quantizedMatmulRhsQ5_KFromBlocks = KQuantRhsFromBlocks(types.QuantizedMatmulRhsQ5_K).fromBlocks;
pub const quantizedMatmulRhsQ6_KFromBlocks = KQuantRhsFromBlocks(types.QuantizedMatmulRhsQ6_K).fromBlocks;

pub fn quantizeRowQ8_KInto(dst: []dtype_mod.BlockQ8_K, src: []const f32) !void {
    const block_count = try qkBlockCount(src.len);
    if (dst.len != block_count) return types.QuantizedFormatError.InvalidQuantizedLength;
    quantizeRowQ8_KIntoUnchecked(dst, src);
}

/// `quantizeRowQ8_KInto` for callers that have already proven the lengths
/// (same contract as `quantizeRowQ8_0IntoUnchecked`): one
/// `quantizeBlockQ8_K` per block.
pub fn quantizeRowQ8_KIntoUnchecked(dst: []dtype_mod.BlockQ8_K, src: []const f32) void {
    std.debug.assert(src.len % qk_k_block_size == 0);
    std.debug.assert(dst.len == src.len / qk_k_block_size);

    var block_index: usize = 0;
    while (block_index < dst.len) : (block_index += 1) {
        dst[block_index] = quantizeBlockQ8_K(src[block_index * qk_k_block_size ..][0..qk_k_block_size]);
    }
}

/// One Q8_K block from 256 values, the body behind every Q8_K activation
/// layout (rows, Q8_Kx4 lanes, Q8_Kx2Mmla pairs). One `@Vector` body on
/// every architecture: 4-lane amax and the signed max it belongs to
/// (`-127 / max` scale, so `qs` never holds -128), then per lane
/// `common.roundNearestEvenVec4ToI32` (fcvtns on aarch64, the 2^23
/// magic-number round elsewhere; |scaled| <= 127 keeps it exact) capped at
/// 127, and the 16-group bsums from the same lanes. An all-zero block
/// (d = 0) when every value is zero. Byte-equal to the scalar
/// `common.roundNearestEven` form (`x86dot_check.zig` pins the two).
pub fn quantizeBlockQ8_K(src: *const [qk_k_block_size]f32) dtype_mod.BlockQ8_K {
    var amaxv: QKV4f32 = @splat(0);
    var vec_index: usize = 0;
    while (vec_index < qk_k_block_size / 4) : (vec_index += 1) {
        const v: QKV4f32 = src[vec_index * 4 ..][0..4].*;
        amaxv = @max(amaxv, @abs(v));
    }
    const amax = @reduce(.Max, amaxv);
    var max_value: f32 = 0;
    for (src) |v| {
        if (@abs(v) == amax) {
            max_value = v;
            break;
        }
    }

    if (amax == 0) return std.mem.zeroes(dtype_mod.BlockQ8_K);

    var block: dtype_mod.BlockQ8_K = undefined;
    const inv_scale = -127.0 / max_value;
    var bsums = [_]i32{0} ** 16;
    vec_index = 0;
    while (vec_index < qk_k_block_size / 4) : (vec_index += 1) {
        const v: QKV4f32 = src[vec_index * 4 ..][0..4].*;
        const q = @min(common.roundNearestEvenVec4ToI32(v * @as(QKV4f32, @splat(inv_scale))), @as(common.QKV4i32, @splat(127)));
        inline for (0..4) |lane| block.qs[vec_index * 4 + lane] = @intCast(q[lane]);
        bsums[vec_index / 4] += @reduce(.Add, q);
    }

    for (&block.bsums, bsums) |*sum, value| sum.* = @intCast(value);
    block.d = 1.0 / inv_scale;
    return block;
}

pub fn quantizeRowsQ8_K(allocator: Allocator, src: *const Tensor) ![]dtype_mod.BlockQ8_K {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    const blocks_per_row = try qkBlockCount(cols);
    const data = try src.dataConstChecked();

    const blocks = try allocator.alloc(dtype_mod.BlockQ8_K, try types.checkedProduct(rows, blocks_per_row));
    errdefer allocator.free(blocks);

    quantizeRowsQ8_KRangeInto(blocks, data, rows, cols, blocks_per_row, 0, rows);
    return blocks;
}
/// Rows `[row_start, row_end)` of the `[rows, cols]` activation `data` into
/// their Q8_K blocks; same contract as `quantizeRowsQ8_0RangeInto`.
pub fn quantizeRowsQ8_KRangeInto(blocks: []dtype_mod.BlockQ8_K, data: []const f32, rows: usize, cols: usize, blocks_per_row: usize, row_start: usize, row_end: usize) void {
    std.debug.assert(row_end <= rows and data.len == rows * cols and blocks.len == rows * blocks_per_row);
    for (row_start..row_end) |row| {
        quantizeRowQ8_KIntoUnchecked(
            blocks[row * blocks_per_row ..][0..blocks_per_row],
            data[row * cols ..][0..cols],
        );
    }
}
pub fn quantizeRowsQ8_Kx4Into(blocks: []types.BlockQ8_Kx4, src: *const Tensor) !void {
    return quantizeRowsQ8_Kx4IntoImpl(blocks, src, false);
}
pub fn quantizeRowsQ8_Kx4PaddedInto(blocks: []types.BlockQ8_Kx4, src: *const Tensor) !void {
    return quantizeRowsQ8_Kx4IntoImpl(blocks, src, true);
}
pub fn quantizeRowsQ8_Kx4IntoImpl(blocks: []types.BlockQ8_Kx4, src: *const Tensor, comptime pad_rows: bool) !void {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    if (!pad_rows and rows % 4 != 0) return tensor.TensorError.InvalidShape;

    const blocks_per_row = try qkBlockCount(cols);
    const row_groups = if (pad_rows) (rows + 3) / 4 else rows / 4;
    if (blocks.len != try types.checkedProduct(row_groups, blocks_per_row)) return types.QuantizedFormatError.InvalidQuantizedLength;

    const data = try src.dataConstChecked();
    quantizeRowsQ8_Kx4GroupsInto(blocks, data, rows, cols, blocks_per_row, 0, row_groups);
}
/// 4-row groups `[group_start, group_end)` of the `[rows, cols]` activation
/// `data` into their lane-packed Q8_Kx4 blocks (a final partial group pads
/// its missing lanes); same contract as `quantizeRowsQ8_0RangeInto`.
pub fn quantizeRowsQ8_Kx4GroupsInto(blocks: []types.BlockQ8_Kx4, data: []const f32, rows: usize, cols: usize, blocks_per_row: usize, group_start: usize, group_end: usize) void {
    std.debug.assert(group_end <= (rows + 3) / 4 and data.len == rows * cols and blocks.len == ((rows + 3) / 4) * blocks_per_row);
    for (group_start..group_end) |row_group| {
        const rows_in_group = @min(rows - row_group * 4, 4);
        quantizeRowGroupQ8_Kx4Into(
            blocks[row_group * blocks_per_row ..][0..blocks_per_row],
            data[row_group * 4 * cols ..][0 .. rows_in_group * cols],
            rows_in_group,
            cols,
        );
    }
}
/// One 4-row group (`rows_in_group` real rows, the rest zero lanes) of
/// `[rows_in_group, cols]` activation `data` into its `blocks.len`
/// lane-packed Q8_Kx4 blocks.
pub fn quantizeRowGroupQ8_Kx4Into(blocks: []types.BlockQ8_Kx4, data: []const f32, rows_in_group: usize, cols: usize) void {
    const blocks_per_row = blocks.len;
    for (0..blocks_per_row) |block_index| {
        const dst = &blocks[block_index];
        inline for (0..4) |row_lane| {
            if (row_lane >= rows_in_group) {
                zeroQ8Kx4Lane(dst, row_lane);
            } else {
                const block = quantizeBlockQ8_K(data[row_lane * cols + block_index * qk_k_block_size ..][0..qk_k_block_size]);
                storeQ8Kx4Lane(dst, row_lane, &block);
            }
        }
    }
}

/// Row `row_lane` of a Q8_Kx4 group from one Q8_K block: the 4-feature
/// interleave (`qs[fg*16 + row*4 + lane]`) and the bsums interleave
/// (`bsums[(sb/4)*16 + row*4 + sb%4]`) the Q8_Kx4 kernels read.
pub fn storeQ8Kx4Lane(dst: *types.BlockQ8_Kx4, comptime row_lane: usize, src: *const dtype_mod.BlockQ8_K) void {
    dst.d[row_lane] = src.d;
    for (0..qk_k_block_size / 4) |feature_group| {
        inline for (0..4) |lane| {
            dst.qs[feature_group * 16 + row_lane * 4 + lane] = src.qs[feature_group * 4 + lane];
        }
    }
    inline for (0..16) |subblock| {
        dst.bsums[(subblock / 4) * 16 + row_lane * 4 + subblock % 4] = src.bsums[subblock];
    }
}

pub fn zeroQ8Kx4Lane(block: *types.BlockQ8_Kx4, comptime row_lane: usize) void {
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
pub fn quantizeRowsQ8_Kx2MmlaInto(blocks: []types.BlockQ8_Kx2Mmla, src: *const Tensor) !void {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    if (rows % 2 != 0) return tensor.TensorError.InvalidShape;

    const blocks_per_row = try qkBlockCount(cols);
    if (blocks.len != try types.checkedProduct(rows / 2, blocks_per_row)) return types.QuantizedFormatError.InvalidQuantizedLength;

    const data = try src.dataConstChecked();
    quantizeRowsQ8_Kx2MmlaGroupsInto(blocks, data, rows, cols, blocks_per_row, 0, rows / 2);
}
/// Row pairs `[group_start, group_end)` of the `[rows, cols]` activation
/// `data` (`rows` even) into their smmla-interleaved Q8_Kx2Mmla blocks;
/// same contract as `quantizeRowsQ8_0RangeInto`.
pub fn quantizeRowsQ8_Kx2MmlaGroupsInto(blocks: []types.BlockQ8_Kx2Mmla, data: []const f32, rows: usize, cols: usize, blocks_per_row: usize, group_start: usize, group_end: usize) void {
    std.debug.assert(rows % 2 == 0 and group_end <= rows / 2 and data.len == rows * cols and blocks.len == (rows / 2) * blocks_per_row);
    for (group_start..group_end) |row_group| {
        for (0..blocks_per_row) |block_index| {
            var dst = &blocks[row_group * blocks_per_row + block_index];
            var pair: [2]dtype_mod.BlockQ8_K = undefined;
            inline for (0..2) |row_lane| {
                const row = row_group * 2 + row_lane;
                pair[row_lane] = quantizeBlockQ8_K(data[row * cols + block_index * qk_k_block_size ..][0..qk_k_block_size]);
                dst.d[row_lane] = pair[row_lane].d;
                // One smmla sub-block spans two Q8_K 16-groups.
                inline for (0..8) |subblock| {
                    dst.bsums[subblock * 2 + row_lane] = @intCast(@as(i32, pair[row_lane].bsums[subblock * 2]) + @as(i32, pair[row_lane].bsums[subblock * 2 + 1]));
                }
            }

            inline for (0..8) |subblock| {
                inline for (0..2) |half| {
                    const dst_base = subblock * 64 + half * 32;
                    const src_base = subblock * 32 + half * 16;
                    inline for (0..8) |lane| {
                        dst.qs[dst_base + lane] = pair[0].qs[src_base + lane];
                        dst.qs[dst_base + 8 + lane] = pair[1].qs[src_base + lane];
                        dst.qs[dst_base + 16 + lane] = pair[0].qs[src_base + 8 + lane];
                        dst.qs[dst_base + 24 + lane] = pair[1].qs[src_base + 8 + lane];
                    }
                }
            }
        }
    }
}

pub fn packRowsQ8_Kx4(
    allocator: Allocator,
    blocks: []const dtype_mod.BlockQ8_K,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
) ![]types.BlockQ8_Kx4 {
    if (rows % 4 != 0) return tensor.TensorError.InvalidShape;
    if (blocks_per_row != try qkBlockCount(cols)) return tensor.TensorError.InvalidShape;
    if (blocks.len != try types.checkedProduct(rows, blocks_per_row)) return types.QuantizedFormatError.InvalidQuantizedLength;

    const row_groups = rows / 4;
    const packed_blocks = try allocator.alloc(types.BlockQ8_Kx4, try types.checkedProduct(row_groups, blocks_per_row));
    errdefer allocator.free(packed_blocks);

    for (0..row_groups) |group| {
        for (0..blocks_per_row) |block_index| {
            const dst = &packed_blocks[group * blocks_per_row + block_index];
            inline for (0..4) |row| storeQ8Kx4Lane(dst, row, &blocks[(group * 4 + row) * blocks_per_row + block_index]);
        }
    }

    return packed_blocks;
}

/// `ceil(m/4)` Q8_Kx4 groups from `m` rows of Q8_K blocks, the lanes past
/// `m` zeroed.
pub fn packRowsQ8_Kx4PaddedInto(
    dst: []types.BlockQ8_Kx4,
    blocks: []const dtype_mod.BlockQ8_K,
    m: usize,
    blocks_per_row: usize,
) void {
    const row_groups = (m + 3) / 4;
    for (0..row_groups) |group| {
        for (0..blocks_per_row) |block_index| {
            const d = &dst[group * blocks_per_row + block_index];
            inline for (0..4) |row_lane| {
                const row = group * 4 + row_lane;
                if (row >= m) {
                    zeroQ8Kx4Lane(d, row_lane);
                } else {
                    storeQ8Kx4Lane(d, row_lane, &blocks[row * blocks_per_row + block_index]);
                }
            }
        }
    }
}

pub fn q4Kx8D(bits: [8]u16, comptime group: usize) QKV4f32 {
    return common.f16x4BitsToF32(.{
        bits[group * 4 + 0],
        bits[group * 4 + 1],
        bits[group * 4 + 2],
        bits[group * 4 + 3],
    });
}
pub fn q4Kx8Scales(values: *const [8 * 8]u8, comptime subblock: usize, comptime group: usize) common.QKV4i32 {
    const offset = subblock * 8 + group * 4;
    return .{
        values[offset + 0],
        values[offset + 1],
        values[offset + 2],
        values[offset + 3],
    };
}
pub fn dequantizeBlockQ8_KInto(dst: *[qk_k_block_size]f32, src: *const dtype_mod.BlockQ8_K) void {
    for (dst, src.qs) |*out, q| out.* = src.d * @as(f32, @floatFromInt(q));
}
pub const getScaleMinK4 = common.getScaleMinK4;
// ---------------------------------------------------------------------------
// f32 -> K-quant reference encoders: shared iterative scale-search helpers,
// ported operation-for-operation (f32 arithmetic, same rounding) from ggml's
// ggml-quants.c so the encoded bytes match quantize_row_{q4,q5,q6}_K_ref
// bit-for-bit (verified against embedded goldens in encode_golden_test.zig).
//
// Finite-input contract (same as ggml): inputs must be free of NaN/inf. A
// sub-block whose value spread underflows f32 (so nmax/(max-min) overflows to
// inf) feeds a non-finite value into nearestInt; ggml's own assert in
// nearest_int rejects that too, and we mirror it with a debug assert.
// ---------------------------------------------------------------------------

/// ggml GROUP_MAX_EPS: sub-blocks with amax below this encode as all-zero.
pub const group_max_eps: f32 = 1e-15;
/// ggml nearest_int: round-to-nearest-even via the 1.5*2^23 magic constant.
/// Valid for |fval| <= 4194303 (ggml asserts the same bound).
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
/// ggml make_qx_quants: symmetric scale search over +/-9 tenth-steps around
/// -nmax/max (the sign rides on the scale). `L` receives nmax-biased levels in
/// [0, 2*nmax-1]. The Q6_K encoder uses rmse_type = 1 with qw = null; the full
/// reference behavior (rmse_type 0/negative/2/3/default, explicit qw) is kept.
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
/// ggml make_qkx2_quants: asymmetric (scale, min) grid search used by the
/// Q4_K/Q5_K encoders. `iscale` candidates sweep (rmin + rdelta*is + nmax) /
/// (max - min) for is in [0, nstep]; a weighted least-squares (scale, min) fit
/// is accepted when it lowers the weighted error (MAD when use_mad). `L`
/// receives levels in [0, nmax]; `Laux` is caller-supplied scratch.
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
pub fn fillQ8KPattern(block: *dtype_mod.BlockQ8_K) void {
    block.d = 1;
    for (&block.qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast(i % 17)) - 8);
    for (&block.bsums, 0..) |*sum, group| {
        var acc: i32 = 0;
        for (block.qs[group * 16 ..][0..16]) |q| acc += q;
        sum.* = @intCast(acc);
    }
}
