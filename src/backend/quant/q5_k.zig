//! Hot Q5_K / Q5_Kx8 quantized matmul kernels relocated out of quant.zig.
//! See quant.zig for shared type/helper definitions; every shared symbol this
//! module references is aliased from quant.zig (`qm`) below so the moved bodies
//! compile unchanged.

const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("../../tensor.zig");
const q8k_mod = @import("q8k.zig");
const types_mod = @import("types.zig");
const common = @import("common.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;

// Shared symbols defined in quant.zig, aliased here so the moved bodies compile unchanged.
const BlockQ5_K = types_mod.BlockQ5_K;
const BlockQ5_Kx8 = types_mod.BlockQ5_Kx8;
const BlockQ8_K = types_mod.BlockQ8_K;
const BlockQ8_Kx4 = types_mod.BlockQ8_Kx4;
const QKV16i16 = common.QKV16i16;
const QKV16i8 = common.QKV16i8;
const QKV16u8 = common.QKV16u8;
const QKV32i8 = common.QKV32i8;
const QKV32u8 = common.QKV32u8;
const QKV4f32 = common.QKV4f32;
const QKV4i32 = common.QKV4i32;
const QKV8i32 = common.QKV8i32;
const QuantizedFormatError = types_mod.QuantizedFormatError;
const addHalvesI32x8 = common.addHalvesI32x8;
const QuantizedMatmulRhsQ5_K = types_mod.QuantizedMatmulRhsQ5_K;
const QuantizedMatmulRhsQ5_Kx8 = types_mod.QuantizedMatmulRhsQ5_Kx8;
const checkedProduct = types_mod.checkedProduct;
const dequantizeBlockQ8_KInto = q8k_mod.dequantizeBlockQ8_KInto;
const dotDense = common.dotDense;
const dotI8GroupsWidenI32x8 = common.dotI8GroupsWidenI32x8;
const dpbusdI32x8 = common.dpbusdI32x8;
const f16BitsToF32 = common.f16BitsToF32;
const f32ToF16Bits = common.f32ToF16Bits;
const fillQ8KPattern = q8k_mod.fillQ8KPattern;
const getScaleMinK4 = q8k_mod.getScaleMinK4;
const has_x86_avx2 = common.has_x86_avx2;
const has_x86_vnni_ymm = common.has_x86_vnni_ymm;
const maddubsDotGroupsI32x8 = common.maddubsDotGroupsI32x8;
const makeQkx2Quants = q8k_mod.makeQkx2Quants;
const nearestInt = q8k_mod.nearestInt;
const packRowsQ8_Kx4 = q8k_mod.packRowsQ8_Kx4;
const q4Kx8D = q8k_mod.q4Kx8D;
const q4Kx8Scales = q8k_mod.q4Kx8Scales;
const q4_kx8_row_block = common.q4_kx8_row_block;
const qkBlockCount = q8k_mod.qkBlockCount;
const qk_col_block = common.qk_col_block;
const qk_k_block_size = types_mod.qk_k_block_size;
const quantizedMatmulRhsQ5_KFromBlocks = q8k_mod.quantizedMatmulRhsQ5_KFromBlocks;
const sdotI8x16Lane = common.sdotI8x16Lane;

pub fn packMatmulRhsQ5_Kx8(
    allocator: Allocator,
    blocks: []const BlockQ5_K,
    n: usize,
    k: usize,
    blocks_per_row: usize,
) !QuantizedMatmulRhsQ5_Kx8 {
    if (n % 8 != 0) return tensor.TensorError.InvalidShape;
    if (blocks_per_row != try qkBlockCount(k)) return tensor.TensorError.InvalidShape;
    if (blocks.len != try checkedProduct(n, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;

    const group_count = n / 8;
    const packed_blocks = try allocator.alloc(BlockQ5_Kx8, try checkedProduct(group_count, blocks_per_row));
    errdefer allocator.free(packed_blocks);

    for (0..group_count) |group_i| {
        for (0..blocks_per_row) |block_i| {
            const cols = [_]*const BlockQ5_K{
                &blocks[(8 * group_i + 0) * blocks_per_row + block_i],
                &blocks[(8 * group_i + 1) * blocks_per_row + block_i],
                &blocks[(8 * group_i + 2) * blocks_per_row + block_i],
                &blocks[(8 * group_i + 3) * blocks_per_row + block_i],
                &blocks[(8 * group_i + 4) * blocks_per_row + block_i],
                &blocks[(8 * group_i + 5) * blocks_per_row + block_i],
                &blocks[(8 * group_i + 6) * blocks_per_row + block_i],
                &blocks[(8 * group_i + 7) * blocks_per_row + block_i],
            };
            var dst = &packed_blocks[group_i * blocks_per_row + block_i];

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

            for (0..8) |subblock| {
                for (0..8) |feature_group| {
                    inline for (0..8) |col| {
                        const block = cols[col];
                        inline for (0..4) |lane| {
                            const feature_offset = feature_group * 4 + lane;
                            dst.qs[subblock * 256 + feature_group * 32 + col * 4 + lane] =
                                @intCast(q5KValue(block, subblock, feature_offset));
                        }
                    }
                }
            }
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

pub fn matmulQ5_Kx8RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ5_Kx8,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    std.debug.assert(c0 % 8 == 0);
    std.debug.assert(c1 % 8 == 0);

    const blocks_per_row = rhs.blocks_per_group;
    var i = r0;
    while (i + q4_kx8_row_block <= r1) : (i += q4_kx8_row_block) {
        var j = c0;
        while (j < c1) : (j += 8) {
            const rhs_group = rhs.groupBlocks(j / 8);
            var acc: [q4_kx8_row_block][2]QKV4f32 = undefined;
            inline for (0..q4_kx8_row_block) |r| {
                acc[r][0] = @splat(0);
                acc[r][1] = @splat(0);
            }

            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const rhs_block = &rhs_group[block_index];
                accumulateQ5_Kx8Rows(lhs_blocks, i, blocks_per_row, block_index, rhs_block, &acc);
            }

            inline for (0..q4_kx8_row_block) |r| {
                out[(i + r) * n + j + 0] = acc[r][0][0];
                out[(i + r) * n + j + 1] = acc[r][0][1];
                out[(i + r) * n + j + 2] = acc[r][0][2];
                out[(i + r) * n + j + 3] = acc[r][0][3];
                out[(i + r) * n + j + 4] = acc[r][1][0];
                out[(i + r) * n + j + 5] = acc[r][1][1];
                out[(i + r) * n + j + 6] = acc[r][1][2];
                out[(i + r) * n + j + 7] = acc[r][1][3];
            }
        }
    }

    while (i + 8 <= r1) : (i += 8) {
        matmulQ5_Kx8RhsTailRows(8, out, lhs_blocks, rhs, n, i, c0, c1, blocks_per_row);
    }

    while (i + 4 <= r1) : (i += 4) {
        matmulQ5_Kx8RhsTailRows(4, out, lhs_blocks, rhs, n, i, c0, c1, blocks_per_row);
    }

    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;
        while (j < c1) : (j += 8) {
            const rhs_group = rhs.groupBlocks(j / 8);
            var acc: [2]QKV4f32 = .{ @splat(0), @splat(0) };
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                accumulateQ5_Kx8(&lhs_row[block_index], &rhs_group[block_index], &acc);
            }
            out[i * n + j + 0] = acc[0][0];
            out[i * n + j + 1] = acc[0][1];
            out[i * n + j + 2] = acc[0][2];
            out[i * n + j + 3] = acc[0][3];
            out[i * n + j + 4] = acc[1][0];
            out[i * n + j + 5] = acc[1][1];
            out[i * n + j + 6] = acc[1][2];
            out[i * n + j + 7] = acc[1][3];
        }
    }
}

fn matmulQ5_Kx8RhsTailRows(
    comptime row_block: usize,
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ5_Kx8,
    n: usize,
    row_start: usize,
    c0: usize,
    c1: usize,
    blocks_per_row: usize,
) void {
    var j = c0;
    while (j < c1) : (j += 8) {
        const rhs_group = rhs.groupBlocks(j / 8);
        var acc: [row_block][2]QKV4f32 = undefined;
        inline for (0..row_block) |r| {
            acc[r][0] = @splat(0);
            acc[r][1] = @splat(0);
        }

        var block_index: usize = 0;
        while (block_index < blocks_per_row) : (block_index += 1) {
            const rhs_block = &rhs_group[block_index];
            inline for (0..row_block) |r| {
                const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
                accumulateQ5_Kx8(lhs, rhs_block, &acc[r]);
            }
        }

        inline for (0..row_block) |r| {
            out[(row_start + r) * n + j + 0] = acc[r][0][0];
            out[(row_start + r) * n + j + 1] = acc[r][0][1];
            out[(row_start + r) * n + j + 2] = acc[r][0][2];
            out[(row_start + r) * n + j + 3] = acc[r][0][3];
            out[(row_start + r) * n + j + 4] = acc[r][1][0];
            out[(row_start + r) * n + j + 5] = acc[r][1][1];
            out[(row_start + r) * n + j + 6] = acc[r][1][2];
            out[(row_start + r) * n + j + 7] = acc[r][1][3];
        }
    }
}

pub fn matmulQ5_Kx8RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ5_Kx8,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ5_Kx8RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ5_Kx8Q8_Kx4RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_Kx4,
    rhs: *const QuantizedMatmulRhsQ5_Kx8,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    std.debug.assert(r0 % 4 == 0);
    std.debug.assert(r1 % 4 == 0);
    std.debug.assert(c0 % 8 == 0);
    std.debug.assert(c1 % 8 == 0);

    const blocks_per_row = rhs.blocks_per_group;
    var i = r0;
    while (i < r1) : (i += 4) {
        var j = c0;
        while (j < c1) : (j += 8) {
            const rhs_group = rhs.groupBlocks(j / 8);
            const lhs_row_group = (i / 4) * blocks_per_row;
            var acc: [4][2]QKV4f32 = undefined;
            inline for (0..4) |row| {
                acc[row][0] = @splat(0);
                acc[row][1] = @splat(0);
            }

            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                accumulateQ5_Kx8Q8_Kx4(&lhs_blocks[lhs_row_group + block_index], &rhs_group[block_index], &acc);
            }

            inline for (0..4) |row| {
                out[(i + row) * n + j + 0] = acc[row][0][0];
                out[(i + row) * n + j + 1] = acc[row][0][1];
                out[(i + row) * n + j + 2] = acc[row][0][2];
                out[(i + row) * n + j + 3] = acc[row][0][3];
                out[(i + row) * n + j + 4] = acc[row][1][0];
                out[(i + row) * n + j + 5] = acc[row][1][1];
                out[(i + row) * n + j + 6] = acc[row][1][2];
                out[(i + row) * n + j + 7] = acc[row][1][3];
            }
        }
    }
}

pub fn matmulQ5_Kx8Q8_Kx4RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_Kx4,
    rhs: *const QuantizedMatmulRhsQ5_Kx8,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ5_Kx8Q8_Kx4RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ5_KRhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ5_K,
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
                    acc[c] += dotQ5_KQ8_K(rhs_block, lhs_block);
                }
            }
            inline for (0..qk_col_block) |c| out[i * n + j + c] = acc[c];
        }

        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc += dotQ5_KQ8_K(&rhs_col[block_index], &lhs_row[block_index]);
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn matmulQ5_KRhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ5_K,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ5_KRhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

const moe_row_tile = 4;

/// Unpack one Q5_K sub-block (32 weights) to two i8 lanes for sdot. Same
/// extraction as `dotQ5_KSubblockI32`, but emitted once so it can be reused
/// across a batch of LHS rows.
fn unpackQ5_KSubblock(w: *const BlockQ5_K, comptime subblock: usize) [2]QKV16i8 {
    const q_offset = (subblock / 2) * 32;
    const high_mask: u8 = @as(u8, 1) << @intCast(subblock);
    const q0: QKV16u8 = @bitCast(w.qs[q_offset..][0..16].*);
    const q1: QKV16u8 = @bitCast(w.qs[q_offset + 16 ..][0..16].*);
    const h0: QKV16u8 = @bitCast(w.qh[0..16].*);
    const h1: QKV16u8 = @bitCast(w.qh[16..32].*);
    var qs0 = if (subblock % 2 == 0) q0 & @as(QKV16u8, @splat(0x0f)) else q0 >> @as(QKV16u8, @splat(4));
    var qs1 = if (subblock % 2 == 0) q1 & @as(QKV16u8, @splat(0x0f)) else q1 >> @as(QKV16u8, @splat(4));
    qs0 += @select(u8, (h0 & @as(QKV16u8, @splat(high_mask))) != @as(QKV16u8, @splat(0)), @as(QKV16u8, @splat(16)), @as(QKV16u8, @splat(0)));
    qs1 += @select(u8, (h1 & @as(QKV16u8, @splat(high_mask))) != @as(QKV16u8, @splat(0)), @as(QKV16u8, @splat(16)), @as(QKV16u8, @splat(0)));
    return .{ @bitCast(qs0), @bitCast(qs1) };
}

fn dotUnpackedI8x32(w0: QKV16i8, w1: QKV16i8, a0: QKV16i8, a1: QKV16i8) i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        var acc: QKV4i32 = @splat(0);
        acc = common.sdotI8x16(acc, w0, a0);
        acc = common.sdotI8x16(acc, w1, a1);
        return @reduce(.Add, acc);
    }
    // non-aarch64: VNNI lowers each to vpdpbusd (via the +128 bias), AVX2 takes
    // the sign-trick vpmaddubsw path (w ∈ [0,31] here; a comes from BlockQ8_K,
    // i.e. quantizeRowQ8_KInto's -127/max scale construction, so a ∈ [-127,127]
    // — inside the sign-trick exactness domain; see common.zig).
    return common.dotI8x16Portable(w0, a0) + common.dotI8x16Portable(w1, a1);
}

/// Column-outer Q5_K matmul for the m>1 (batched MoE prefill) case: unpack each
/// weight block's 5-bit values ONCE, then sdot them against a tile of LHS rows —
/// amortizing the unpack over the batch instead of re-unpacking per row like the
/// row-outer `matmulQ5_KRhsTile`. Numerically identical to it (same per-block
/// deferred-f32 reduction, same cross-block accumulation order).
pub fn matmulQ5_KRhsCompactColOuter(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ5_K,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const bpc = rhs.blocks_per_column;
    var j = c0;
    while (j < c1) : (j += 1) {
        const col = rhs.columnBlocks(j);
        var row0 = r0;
        while (row0 < r1) : (row0 += moe_row_tile) {
            const tn = @min(moe_row_tile, r1 - row0);
            var acc_f32 = [_]f32{0} ** moe_row_tile;
            var bi: usize = 0;
            while (bi < bpc) : (bi += 1) {
                const w = &col[bi];
                var iscale = [_]i32{0} ** moe_row_tile;
                var imin = [_]i32{0} ** moe_row_tile;
                inline for (0..8) |subblock| {
                    const wv = unpackQ5_KSubblock(w, subblock);
                    const sm = getScaleMinK4(&w.scales, subblock);
                    var r: usize = 0;
                    while (r < tn) : (r += 1) {
                        const a = &lhs_blocks[(row0 + r) * bpc + bi];
                        const a0: QKV16i8 = @bitCast(a.qs[subblock * 32 ..][0..16].*);
                        const a1: QKV16i8 = @bitCast(a.qs[subblock * 32 + 16 ..][0..16].*);
                        const acc = dotUnpackedI8x32(wv[0], wv[1], a0, a1);
                        const bsum = @as(i32, a.bsums[subblock * 2]) + @as(i32, a.bsums[subblock * 2 + 1]);
                        iscale[r] += @as(i32, sm.scale) * acc;
                        imin[r] += @as(i32, sm.min) * bsum;
                    }
                }
                const d = f16BitsToF32(w.dm[0]);
                const dmin = f16BitsToF32(w.dm[1]);
                var r: usize = 0;
                while (r < tn) : (r += 1) {
                    const ad = lhs_blocks[(row0 + r) * bpc + bi].d;
                    acc_f32[r] += (d * ad) * @as(f32, @floatFromInt(iscale[r])) - (dmin * ad) * @as(f32, @floatFromInt(imin[r]));
                }
            }
            var r: usize = 0;
            while (r < tn) : (r += 1) out[(row0 + r) * n + j] = acc_f32[r];
        }
    }
}

/// Per-row activation sum of one Q5_K sub-block (= two Q8_K 16-groups, `2*subblock`
/// and `2*subblock+1`) for all four rows of a `BlockQ8_Kx4`, lane = row. Mirrors the
/// `bsums[(g/4)*16 + row*4 + g%4]` interleave that `quantizeRowsQ8_Kx4*Into` writes.
inline fn bsumPairQ8_Kx4(a: *const BlockQ8_Kx4, comptime subblock: usize) QKV4i32 {
    const g0 = subblock * 2;
    const g1 = subblock * 2 + 1;
    var v: QKV4i32 = undefined;
    inline for (0..4) |row| {
        v[row] = @as(i32, a.bsums[(g0 / 4) * 16 + row * 4 + (g0 % 4)]) +
            @as(i32, a.bsums[(g1 / 4) * 16 + row * 4 + (g1 % 4)]);
    }
    return v;
}

/// Four rows' i8 dot of one 32-wide Q5_K sub-block against pre-unpacked weights
/// `wv` (`wv[0]` = feature-groups 0..3, `wv[1]` = 4..7), returning the four row dots
/// in the four lanes of one i32x4 (lane = row). On aarch64 each `sdot …4b[g]` reuses
/// the single unpacked weight register across all four rows, so the whole sub-block
/// is 8 `sdot`s and **zero** horizontal reductions. Integer dots are
/// order-independent, so this equals the per-row `dotUnpackedI8x32` path exactly.
inline fn dot4RowsSubblockQ8_Kx4(a: *const BlockQ8_Kx4, comptime subblock: usize, wv: [2]QKV16i8) QKV4i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        var dot: QKV4i32 = @splat(0);
        inline for (0..4) |g| {
            const ag: QKV16i8 = @bitCast(a.qs[subblock * 128 + g * 16 ..][0..16].*);
            dot = sdotI8x16Lane(g, dot, ag, wv[0]);
        }
        inline for (0..4) |g| {
            const ag: QKV16i8 = @bitCast(a.qs[subblock * 128 + (g + 4) * 16 ..][0..16].*);
            dot = sdotI8x16Lane(g, dot, ag, wv[1]);
        }
        return dot;
    }
    if (comptime has_x86_vnni_ymm) return dot4RowsSubblockQ8_Kx4Simd(.vnni, a, subblock, wv);
    if (comptime has_x86_avx2) return dot4RowsSubblockQ8_Kx4Simd(.avx2, a, subblock, wv);
    return dot4RowsSubblockQ8_Kx4Simd(.widen, a, subblock, wv);
}

// vpermd-class (cross-lane): broadcast dword 2c to the low 128-bit lane and
// dword 2c+1 to the high lane of a 4-dword source — aligns one pre-unpacked
// 16-byte weight half against the [fg | fg+1] activation halves of a 32-byte
// Q8_Kx4 load (the 4-dword-source analog of `broadcastGroupI32x8`).
inline fn broadcastPairGroupsI32x4(comptime c: comptime_int, v: QKV4i32) QKV8i32 {
    return @shuffle(i32, v, undefined, [8]i32{ 2 * c, 2 * c, 2 * c, 2 * c, 2 * c + 1, 2 * c + 1, 2 * c + 1, 2 * c + 1 });
}

/// x86/portable ymm arms of `dot4RowsSubblockQ8_Kx4` (the MoE-prefill 4-row
/// lane dot): each 32-byte Q8_Kx4 activation load already holds [fg: 4
/// rows × 4 features | fg+1: …] dword-per-row, so broadcasting the
/// matching weight dword pair (`broadcastPairGroupsI32x4`) turns the
/// 32-feature × 4-row sub-block dot into four grouped-dot ops + one
/// half-fold — no per-row rebuild. OPERAND SHAPE: `wv` holds UNSIGNED 5-bit
/// values [0,31] (nibble + qh bit, see `unpackQ5_KSubblock`) — natively
/// vpdpbusd's u8 side, dotted directly (no bias, no correction, no sign
/// trick); activations are unrestricted i8. SATURATION (avx2 tier): w ≤ 31 →
/// vpmaddubsw pair sums ≤ 2·31·128 = 7936 < 2^15. NO OVERFLOW: |sum8 lane| ≤
/// 4·4·31·128 < 2^17. Integer sums are order-independent → bit-identical to
/// `dot4RowsSubblockQ8_Kx4Scalar` (q5_k_tests.zig). pub for the sibling
/// exact-parity tests.
pub fn dot4RowsSubblockQ8_Kx4Simd(comptime tier: X86DotTier, a: *const BlockQ8_Kx4, comptime subblock: usize, wv: [2]QKV16i8) QKV4i32 {
    var sum8: QKV8i32 = @splat(0);
    inline for (0..4) |c| {
        const act: QKV32i8 = @bitCast(a.qs[subblock * 128 + c * 32 ..][0..32].*);
        const wb: QKV32u8 = @bitCast(broadcastPairGroupsI32x4(c % 2, @as(QKV4i32, @bitCast(wv[c / 2]))));
        sum8 = dotQ5GroupsI32x8(tier, sum8, wb, act);
    }
    return addHalvesI32x8(sum8);
}

// pub: the bit-exactness reference for dot4RowsSubblockQ8_Kx4Simd
// (q5_k_tests.zig) — the plain per-row rebuild over the interleaved Q8_Kx4
// layout (row r's 4 features for feature-group g live at
// qs[subblock*128 + g*16 + r*4 ..][0..4]).
pub fn dot4RowsSubblockQ8_Kx4Scalar(a: *const BlockQ8_Kx4, comptime subblock: usize, wv: [2]QKV16i8) QKV4i32 {
    var dot: QKV4i32 = @splat(0);
    inline for (0..4) |row| {
        var acc: i32 = 0;
        inline for (0..8) |g| {
            inline for (0..4) |t| {
                acc += @as(i32, wv[g / 4][(g % 4) * 4 + t]) * @as(i32, a.qs[subblock * 128 + g * 16 + row * 4 + t]);
            }
        }
        dot[row] = acc;
    }
    return dot;
}

/// Column-outer Q5_K matmul over **4-row-interleaved Q8_Kx4** activations. Like
/// `matmulQ5_KRhsCompactColOuter` it unpacks each weight sub-block once and reuses it
/// across the row tile, but it packs the four rows into the `sdot` lanes
/// (`dot4RowsSubblockQ8_Kx4`) so the four rows share one i32x4 accumulator with no
/// per-row horizontal reduction, and the deferred-f32 epilogue runs vector-wide over
/// the four rows. `lhs_blocks` holds `ceil(m/4)` Q8_Kx4 groups per K-block (tail rows
/// zero-padded, e.g. via `quantizeRowsQ8_Kx4PaddedInto`); `m` is the real row count so
/// padded lanes are never stored. Bit-identical to the per-row column-outer / row-outer
/// tile (same integer reduction, same cross-block f32 accumulation order).
pub fn matmulQ5_KCompactQ8_Kx4ColOuter(
    out: []f32,
    lhs_blocks: []const BlockQ8_Kx4,
    rhs: *const QuantizedMatmulRhsQ5_K,
    n: usize,
    m: usize,
    c0: usize,
    c1: usize,
) void {
    const bpc = rhs.blocks_per_column;
    const row_groups = (m + 3) / 4;
    var j = c0;
    while (j < c1) : (j += 1) {
        const col = rhs.columnBlocks(j);
        var rg: usize = 0;
        while (rg < row_groups) : (rg += 1) {
            const row0 = rg * 4;
            const tn = @min(@as(usize, 4), m - row0);
            var acc_f32: QKV4f32 = @splat(0);
            var bi: usize = 0;
            while (bi < bpc) : (bi += 1) {
                const w = &col[bi];
                const a = &lhs_blocks[rg * bpc + bi];
                var iscale: QKV4i32 = @splat(0);
                var imin: QKV4i32 = @splat(0);
                inline for (0..8) |subblock| {
                    const wv = unpackQ5_KSubblock(w, subblock);
                    const sm = getScaleMinK4(&w.scales, subblock);
                    const dot = dot4RowsSubblockQ8_Kx4(a, subblock, wv);
                    iscale += @as(QKV4i32, @splat(@as(i32, sm.scale))) * dot;
                    imin += @as(QKV4i32, @splat(@as(i32, sm.min))) * bsumPairQ8_Kx4(a, subblock);
                }
                const d = f16BitsToF32(w.dm[0]);
                const dmin = f16BitsToF32(w.dm[1]);
                const ad: QKV4f32 = a.d;
                acc_f32 += (@as(QKV4f32, @splat(d)) * ad) * @as(QKV4f32, @floatFromInt(iscale)) -
                    (@as(QKV4f32, @splat(dmin)) * ad) * @as(QKV4f32, @floatFromInt(imin));
            }
            const acc_arr: [4]f32 = acc_f32;
            var r: usize = 0;
            while (r < tn) : (r += 1) out[(row0 + r) * n + j] = acc_arr[r];
        }
    }
}

fn accumulateQ5_Kx8(lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ5_Kx8Aarch64(lhs, rhs, acc);
    }
    if (comptime has_x86_vnni_ymm) {
        return accumulateQ5_Kx8Vnni(lhs, rhs, acc);
    }
    if (comptime has_x86_avx2) {
        return accumulateQ5_Kx8Avx2(lhs, rhs, acc);
    }
    return accumulateQ5_Kx8Widen(lhs, rhs, acc);
}

fn accumulateQ5_Kx8Rows(
    lhs_blocks: []const BlockQ8_K,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: *const BlockQ5_Kx8,
    acc: *[q4_kx8_row_block][2]QKV4f32,
) void {
    inline for (0..q4_kx8_row_block) |r| {
        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
        accumulateQ5_Kx8(lhs, rhs, &acc[r]);
    }
}

fn accumulateQ5_Kx8Q8_Kx4(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ5_Kx8Q8_Kx4Sdot(lhs, rhs, acc);
    }
    if (comptime has_x86_vnni_ymm) {
        return accumulateQ5_Kx8Q8_Kx4Vnni(lhs, rhs, acc);
    }
    if (comptime has_x86_avx2) {
        return accumulateQ5_Kx8Q8_Kx4Avx2(lhs, rhs, acc);
    }
    return accumulateQ5_Kx8Q8_Kx4Widen(lhs, rhs, acc);
}

// pub: exercised directly by q5_k_tests.zig (bit-exact vs the scalar reference
// on every host — integer sums are order-independent, epilogue is identical).
pub fn accumulateQ5_Kx8Q8_Kx4Sdot(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
    const rhs_d0 = q4Kx8D(rhs.d, 0);
    const rhs_d1 = q4Kx8D(rhs.d, 1);
    const rhs_dmin0 = q4Kx8D(rhs.dmin, 0);
    const rhs_dmin1 = q4Kx8D(rhs.dmin, 1);
    var row_d0: [4]QKV4f32 = undefined;
    var row_d1: [4]QKV4f32 = undefined;
    var row_dmin0: [4]QKV4f32 = undefined;
    var row_dmin1: [4]QKV4f32 = undefined;
    inline for (0..4) |row| {
        const lhs_d: QKV4f32 = @splat(lhs.d[row]);
        row_d0[row] = rhs_d0 * lhs_d;
        row_d1[row] = rhs_d1 * lhs_d;
        row_dmin0[row] = rhs_dmin0 * lhs_d;
        row_dmin1[row] = rhs_dmin1 * lhs_d;
    }

    var iscale0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var iscale1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var imin0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var imin1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };

    inline for (0..8) |subblock| {
        var dot0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
        var dot1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };

        inline for (0..8) |feature_group| {
            const rhs_offset = subblock * 256 + feature_group * 32;
            const rhs0: QKV16i8 = @bitCast(rhs.qs[rhs_offset..][0..16].*);
            const rhs1: QKV16i8 = @bitCast(rhs.qs[rhs_offset + 16 ..][0..16].*);
            const q8_vec: QKV16i8 = @bitCast(lhs.qs[subblock * 128 + feature_group * 16 ..][0..16].*);
            inline for (0..4) |row| {
                dot0[row] = sdotI8x16Lane(row, dot0[row], rhs0, q8_vec);
                dot1[row] = sdotI8x16Lane(row, dot1[row], rhs1, q8_vec);
            }
        }

        const scales0 = q4Kx8Scales(&rhs.scales, subblock, 0);
        const scales1 = q4Kx8Scales(&rhs.scales, subblock, 1);
        const mins0 = q4Kx8Scales(&rhs.mins, subblock, 0);
        const mins1 = q4Kx8Scales(&rhs.mins, subblock, 1);

        inline for (0..4) |row| {
            const bsum0 = subblock * 2;
            const bsum: QKV4i32 = @splat(
                @as(i32, lhs.bsums[(bsum0 / 4) * 16 + row * 4 + bsum0 % 4]) +
                    @as(i32, lhs.bsums[((bsum0 + 1) / 4) * 16 + row * 4 + (bsum0 + 1) % 4]),
            );
            iscale0[row] += dot0[row] * scales0;
            iscale1[row] += dot1[row] * scales1;
            imin0[row] += bsum * mins0;
            imin1[row] += bsum * mins1;
        }
    }

    inline for (0..4) |row| {
        acc[row][0] += @as(QKV4f32, @floatFromInt(iscale0[row])) * row_d0[row] -
            @as(QKV4f32, @floatFromInt(imin0[row])) * row_dmin0[row];
        acc[row][1] += @as(QKV4f32, @floatFromInt(iscale1[row])) * row_d1[row] -
            @as(QKV4f32, @floatFromInt(imin1[row])) * row_dmin1[row];
    }
}

fn accumulateQ5_Kx8Aarch64(lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
    const d0 = q4Kx8D(rhs.d, 0) * @as(QKV4f32, @splat(lhs.d));
    const d1 = q4Kx8D(rhs.d, 1) * @as(QKV4f32, @splat(lhs.d));
    const dmin0 = q4Kx8D(rhs.dmin, 0) * @as(QKV4f32, @splat(lhs.d));
    const dmin1 = q4Kx8D(rhs.dmin, 1) * @as(QKV4f32, @splat(lhs.d));
    // i32-accumulate scale/min across the 8 subblocks, single float convert at
    // the end (same strategy as the ColOuter kernels and dotQ5_KQ8_K).
    var iscale0: QKV4i32 = @splat(0);
    var iscale1: QKV4i32 = @splat(0);
    var imin0: QKV4i32 = @splat(0);
    var imin1: QKV4i32 = @splat(0);

    inline for (0..8) |subblock| {
        var dot0: QKV4i32 = @splat(0);
        var dot1: QKV4i32 = @splat(0);
        inline for (0..2) |half| {
            const lhs_vec: QKV16i8 = @bitCast(lhs.qs[subblock * 32 + half * 16 ..][0..16].*);
            inline for (0..4) |feature_group| {
                const rhs_offset = subblock * 256 + (half * 4 + feature_group) * 32;
                const rhs_vec0: QKV16i8 = @bitCast(rhs.qs[rhs_offset..][0..16].*);
                const rhs_vec1: QKV16i8 = @bitCast(rhs.qs[rhs_offset + 16 ..][0..16].*);
                dot0 = sdotI8x16Lane(feature_group, dot0, rhs_vec0, lhs_vec);
                dot1 = sdotI8x16Lane(feature_group, dot1, rhs_vec1, lhs_vec);
            }
        }

        const scales0 = q4Kx8Scales(&rhs.scales, subblock, 0);
        const scales1 = q4Kx8Scales(&rhs.scales, subblock, 1);
        const mins0 = q4Kx8Scales(&rhs.mins, subblock, 0);
        const mins1 = q4Kx8Scales(&rhs.mins, subblock, 1);
        const bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[subblock * 2]) + @as(i32, lhs.bsums[subblock * 2 + 1]));
        iscale0 += dot0 * scales0;
        iscale1 += dot1 * scales1;
        imin0 += bsum * mins0;
        imin1 += bsum * mins1;
    }

    acc[0] += @as(QKV4f32, @floatFromInt(iscale0)) * d0 - @as(QKV4f32, @floatFromInt(imin0)) * dmin0;
    acc[1] += @as(QKV4f32, @floatFromInt(iscale1)) * d1 - @as(QKV4f32, @floatFromInt(imin1)) * dmin1;
}

// pub: the bit-exactness reference for the x86/portable SIMD arms below
// (q5_k_tests.zig); production non-aarch64 dispatch goes to those arms.
pub fn accumulateQ5_Kx8Scalar(lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
    @setEvalBranchQuota(10000); // fully unrolls 8*32*4 inline iters
    const d0 = q4Kx8D(rhs.d, 0) * @as(QKV4f32, @splat(lhs.d));
    const d1 = q4Kx8D(rhs.d, 1) * @as(QKV4f32, @splat(lhs.d));
    const dmin0 = q4Kx8D(rhs.dmin, 0) * @as(QKV4f32, @splat(lhs.d));
    const dmin1 = q4Kx8D(rhs.dmin, 1) * @as(QKV4f32, @splat(lhs.d));
    var iscale0: QKV4i32 = @splat(0);
    var iscale1: QKV4i32 = @splat(0);
    var imin0: QKV4i32 = @splat(0);
    var imin1: QKV4i32 = @splat(0);

    inline for (0..8) |subblock| {
        var dot0: QKV4i32 = @splat(0);
        var dot1: QKV4i32 = @splat(0);
        inline for (0..32) |feature_offset| {
            const lhs_value: i32 = lhs.qs[subblock * 32 + feature_offset];
            inline for (0..4) |col| {
                const q_offset = subblock * 256 + (feature_offset / 4) * 32 + col * 4 + feature_offset % 4;
                dot0[col] += lhs_value * @as(i32, rhs.qs[q_offset]);
                dot1[col] += lhs_value * @as(i32, rhs.qs[q_offset + 16]);
            }
        }

        const scales0 = q4Kx8Scales(&rhs.scales, subblock, 0);
        const scales1 = q4Kx8Scales(&rhs.scales, subblock, 1);
        const mins0 = q4Kx8Scales(&rhs.mins, subblock, 0);
        const mins1 = q4Kx8Scales(&rhs.mins, subblock, 1);
        const bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[subblock * 2]) + @as(i32, lhs.bsums[subblock * 2 + 1]));
        iscale0 += dot0 * scales0;
        iscale1 += dot1 * scales1;
        imin0 += bsum * mins0;
        imin1 += bsum * mins1;
    }

    acc[0] += @as(QKV4f32, @floatFromInt(iscale0)) * d0 - @as(QKV4f32, @floatFromInt(imin0)) * dmin0;
    acc[1] += @as(QKV4f32, @floatFromInt(iscale1)) * d1 - @as(QKV4f32, @floatFromInt(imin1)) * dmin1;
}

// --- x86 / portable-SIMD arms of the Q5_Kx8 accumulates ----------------------
//
// Packed Q5_Kx8 qs bytes are UNSIGNED 5-bit values (low nibble + qh high bit,
// 0..31 — see q5KValue / packMatmulRhsQ5_Kx8) and the Q8_K activations are
// signed i8 — natively vpdpbusd's u8·i8 shape, so the VNNI arm dots them
// DIRECTLY (no bias, no correction, no sign trick). Every arm computes the
// SAME i32 subblock sums as the scalar arm (exactness bounds per tier at
// dotQ5GroupsI32x8) and applies the f16 scales with the scalar arm's exact
// f32 association, so results are bit-for-bit equal to the scalar reference —
// asserted by q5_k_tests.zig. The arms are written over the comptime-gated
// primitives in common.zig (vpdpbusd / vpmaddubsw / portable widening), so
// each also compiles and runs on any target via the portable twins (that is
// how the aarch64 dev machine exercises them).

// pub: q5_k_tests.zig iterates the tiers when asserting the exact-parity
// suites of the tier-parameterized arms below.
pub const X86DotTier = enum { vnni, avx2, widen };

// One grouped u8(0..31)·i8 dot-accumulate step, tier-selected:
//   vnni : vpdpbusd directly — exact i32 for all inputs, no saturation.
//   avx2 : vpmaddubsw+vpmaddwd — saturation-free here: pair sums are bounded
//          by 2·31·128 = 7936 < 2^15 (weights ≤ 31 by the 5-bit format).
//   widen: universal @Vector form (no arch gate) — q5 values ≤ 31 fit i8, so
//          the signed widening grouped dot is exact; the production floor for
//          every ISA that is neither aarch64 nor gated x86.
inline fn dotQ5GroupsI32x8(comptime tier: X86DotTier, acc: QKV8i32, w: QKV32u8, a: QKV32i8) QKV8i32 {
    return switch (tier) {
        .vnni => dpbusdI32x8(acc, w, a),
        .avx2 => maddubsDotGroupsI32x8(acc, w, a),
        .widen => dotI8GroupsWidenI32x8(acc, @bitCast(w), a),
    };
}

// vpermd/vpbroadcastd-class: broadcast dword `g` (one 4-byte feature group) to
// all 8 dword lanes — aligns one LHS feature group against the 8 packed RHS
// columns of a 32-byte chunk.
inline fn broadcastGroupI32x8(comptime g: comptime_int, v: QKV8i32) QKV8i32 {
    return @shuffle(i32, v, undefined, [8]i32{ g, g, g, g, g, g, g, g });
}

// Split the 8-lane grouped accumulator into its column halves: RHS bytes
// 0..16 of each chunk are columns 0..3 (dot0, low half), bytes 16..32 are
// columns 4..7 (dot1, high half).
inline fn lowHalfI32x8(v: QKV8i32) QKV4i32 {
    return @shuffle(i32, v, undefined, [4]i32{ 0, 1, 2, 3 });
}

inline fn highHalfI32x8(v: QKV8i32) QKV4i32 {
    return @shuffle(i32, v, undefined, [4]i32{ 4, 5, 6, 7 });
}

fn accumulateQ5_Kx8Tier(comptime tier: X86DotTier, lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
    const d0 = q4Kx8D(rhs.d, 0) * @as(QKV4f32, @splat(lhs.d));
    const d1 = q4Kx8D(rhs.d, 1) * @as(QKV4f32, @splat(lhs.d));
    const dmin0 = q4Kx8D(rhs.dmin, 0) * @as(QKV4f32, @splat(lhs.d));
    const dmin1 = q4Kx8D(rhs.dmin, 1) * @as(QKV4f32, @splat(lhs.d));
    var iscale0: QKV4i32 = @splat(0);
    var iscale1: QKV4i32 = @splat(0);
    var imin0: QKV4i32 = @splat(0);
    var imin1: QKV4i32 = @splat(0);

    inline for (0..8) |subblock| {
        // Two independent chains (even/odd feature groups) hide the dot
        // latency; i32 adds are order-independent, so the merged sum equals
        // the scalar arm's. Per-lane bound: 8·4·31·127 < 2^17 — exact i32.
        var sum_e: QKV8i32 = @splat(0);
        var sum_o: QKV8i32 = @splat(0);
        const lhs_groups: QKV8i32 = @bitCast(lhs.qs[subblock * 32 ..][0..32].*);
        inline for (0..4) |pair| {
            const w_e: QKV32u8 = @bitCast(rhs.qs[subblock * 256 + (pair * 2) * 32 ..][0..32].*);
            const w_o: QKV32u8 = @bitCast(rhs.qs[subblock * 256 + (pair * 2 + 1) * 32 ..][0..32].*);
            const b_e: QKV32i8 = @bitCast(broadcastGroupI32x8(pair * 2, lhs_groups));
            const b_o: QKV32i8 = @bitCast(broadcastGroupI32x8(pair * 2 + 1, lhs_groups));
            sum_e = dotQ5GroupsI32x8(tier, sum_e, w_e, b_e);
            sum_o = dotQ5GroupsI32x8(tier, sum_o, w_o, b_o);
        }
        const sum = sum_e + sum_o;

        const scales0 = q4Kx8Scales(&rhs.scales, subblock, 0);
        const scales1 = q4Kx8Scales(&rhs.scales, subblock, 1);
        const mins0 = q4Kx8Scales(&rhs.mins, subblock, 0);
        const mins1 = q4Kx8Scales(&rhs.mins, subblock, 1);
        const bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[subblock * 2]) + @as(i32, lhs.bsums[subblock * 2 + 1]));
        iscale0 += lowHalfI32x8(sum) * scales0;
        iscale1 += highHalfI32x8(sum) * scales1;
        imin0 += bsum * mins0;
        imin1 += bsum * mins1;
    }

    acc[0] += @as(QKV4f32, @floatFromInt(iscale0)) * d0 - @as(QKV4f32, @floatFromInt(imin0)) * dmin0;
    acc[1] += @as(QKV4f32, @floatFromInt(iscale1)) * d1 - @as(QKV4f32, @floatFromInt(imin1)) * dmin1;
}

pub fn accumulateQ5_Kx8Vnni(lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
    accumulateQ5_Kx8Tier(.vnni, lhs, rhs, acc);
}

pub fn accumulateQ5_Kx8Avx2(lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
    accumulateQ5_Kx8Tier(.avx2, lhs, rhs, acc);
}

pub fn accumulateQ5_Kx8Widen(lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
    accumulateQ5_Kx8Tier(.widen, lhs, rhs, acc);
}

fn accumulateQ5_Kx8Q8_Kx4Tier(comptime tier: X86DotTier, lhs: *const BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
    const rhs_d0 = q4Kx8D(rhs.d, 0);
    const rhs_d1 = q4Kx8D(rhs.d, 1);
    const rhs_dmin0 = q4Kx8D(rhs.dmin, 0);
    const rhs_dmin1 = q4Kx8D(rhs.dmin, 1);
    var row_d0: [4]QKV4f32 = undefined;
    var row_d1: [4]QKV4f32 = undefined;
    var row_dmin0: [4]QKV4f32 = undefined;
    var row_dmin1: [4]QKV4f32 = undefined;
    inline for (0..4) |row| {
        const lhs_d: QKV4f32 = @splat(lhs.d[row]);
        row_d0[row] = rhs_d0 * lhs_d;
        row_d1[row] = rhs_d1 * lhs_d;
        row_dmin0[row] = rhs_dmin0 * lhs_d;
        row_dmin1[row] = rhs_dmin1 * lhs_d;
    }

    var iscale0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var iscale1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var imin0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var imin1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };

    inline for (0..8) |subblock| {
        // Eight independent chains (4 rows × even/odd feature groups); the
        // interleaved Kx4 layout puts feature groups 2·pair (dwords 0..3 =
        // rows 0..3) and 2·pair+1 (dwords 4..7) in one 32-byte LHS load.
        var sums_e: [4]QKV8i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
        var sums_o: [4]QKV8i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
        inline for (0..4) |pair| {
            const lhs_groups: QKV8i32 = @bitCast(lhs.qs[subblock * 128 + pair * 32 ..][0..32].*);
            const w_e: QKV32u8 = @bitCast(rhs.qs[subblock * 256 + (pair * 2) * 32 ..][0..32].*);
            const w_o: QKV32u8 = @bitCast(rhs.qs[subblock * 256 + (pair * 2 + 1) * 32 ..][0..32].*);
            inline for (0..4) |row| {
                sums_e[row] = dotQ5GroupsI32x8(tier, sums_e[row], w_e, @bitCast(broadcastGroupI32x8(row, lhs_groups)));
                sums_o[row] = dotQ5GroupsI32x8(tier, sums_o[row], w_o, @bitCast(broadcastGroupI32x8(4 + row, lhs_groups)));
            }
        }

        const scales0 = q4Kx8Scales(&rhs.scales, subblock, 0);
        const scales1 = q4Kx8Scales(&rhs.scales, subblock, 1);
        const mins0 = q4Kx8Scales(&rhs.mins, subblock, 0);
        const mins1 = q4Kx8Scales(&rhs.mins, subblock, 1);

        inline for (0..4) |row| {
            const sum = sums_e[row] + sums_o[row];
            const bsum0 = subblock * 2;
            const bsum: QKV4i32 = @splat(
                @as(i32, lhs.bsums[(bsum0 / 4) * 16 + row * 4 + bsum0 % 4]) +
                    @as(i32, lhs.bsums[((bsum0 + 1) / 4) * 16 + row * 4 + (bsum0 + 1) % 4]),
            );
            iscale0[row] += lowHalfI32x8(sum) * scales0;
            iscale1[row] += highHalfI32x8(sum) * scales1;
            imin0[row] += bsum * mins0;
            imin1[row] += bsum * mins1;
        }
    }

    inline for (0..4) |row| {
        acc[row][0] += @as(QKV4f32, @floatFromInt(iscale0[row])) * row_d0[row] -
            @as(QKV4f32, @floatFromInt(imin0[row])) * row_dmin0[row];
        acc[row][1] += @as(QKV4f32, @floatFromInt(iscale1[row])) * row_d1[row] -
            @as(QKV4f32, @floatFromInt(imin1[row])) * row_dmin1[row];
    }
}

pub fn accumulateQ5_Kx8Q8_Kx4Vnni(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
    accumulateQ5_Kx8Q8_Kx4Tier(.vnni, lhs, rhs, acc);
}

pub fn accumulateQ5_Kx8Q8_Kx4Avx2(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
    accumulateQ5_Kx8Q8_Kx4Tier(.avx2, lhs, rhs, acc);
}

pub fn accumulateQ5_Kx8Q8_Kx4Widen(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
    accumulateQ5_Kx8Q8_Kx4Tier(.widen, lhs, rhs, acc);
}

// pub: the bit-exactness reference for the Q8_Kx4 SIMD arms (q5_k_tests.zig).
// Plain integer loops over the interleaved layouts; identical i32 sums and
// identical f32 epilogue association as the production arms.
pub fn accumulateQ5_Kx8Q8_Kx4Scalar(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
    const rhs_d0 = q4Kx8D(rhs.d, 0);
    const rhs_d1 = q4Kx8D(rhs.d, 1);
    const rhs_dmin0 = q4Kx8D(rhs.dmin, 0);
    const rhs_dmin1 = q4Kx8D(rhs.dmin, 1);
    var row_d0: [4]QKV4f32 = undefined;
    var row_d1: [4]QKV4f32 = undefined;
    var row_dmin0: [4]QKV4f32 = undefined;
    var row_dmin1: [4]QKV4f32 = undefined;
    inline for (0..4) |row| {
        const lhs_d: QKV4f32 = @splat(lhs.d[row]);
        row_d0[row] = rhs_d0 * lhs_d;
        row_d1[row] = rhs_d1 * lhs_d;
        row_dmin0[row] = rhs_dmin0 * lhs_d;
        row_dmin1[row] = rhs_dmin1 * lhs_d;
    }

    var iscale0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var iscale1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var imin0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var imin1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };

    inline for (0..8) |subblock| {
        var dot0: [4][4]i32 = .{ .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } };
        var dot1: [4][4]i32 = .{ .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } };
        for (0..8) |feature_group| {
            for (0..4) |row| {
                for (0..4) |col| {
                    for (0..4) |lane| {
                        const a: i32 = lhs.qs[subblock * 128 + feature_group * 16 + row * 4 + lane];
                        const q_offset = subblock * 256 + feature_group * 32 + col * 4 + lane;
                        dot0[row][col] += a * @as(i32, rhs.qs[q_offset]);
                        dot1[row][col] += a * @as(i32, rhs.qs[q_offset + 16]);
                    }
                }
            }
        }

        const scales0 = q4Kx8Scales(&rhs.scales, subblock, 0);
        const scales1 = q4Kx8Scales(&rhs.scales, subblock, 1);
        const mins0 = q4Kx8Scales(&rhs.mins, subblock, 0);
        const mins1 = q4Kx8Scales(&rhs.mins, subblock, 1);

        inline for (0..4) |row| {
            const bsum0 = subblock * 2;
            const bsum: QKV4i32 = @splat(
                @as(i32, lhs.bsums[(bsum0 / 4) * 16 + row * 4 + bsum0 % 4]) +
                    @as(i32, lhs.bsums[((bsum0 + 1) / 4) * 16 + row * 4 + (bsum0 + 1) % 4]),
            );
            iscale0[row] += @as(QKV4i32, dot0[row]) * scales0;
            iscale1[row] += @as(QKV4i32, dot1[row]) * scales1;
            imin0[row] += bsum * mins0;
            imin1[row] += bsum * mins1;
        }
    }

    inline for (0..4) |row| {
        acc[row][0] += @as(QKV4f32, @floatFromInt(iscale0[row])) * row_d0[row] -
            @as(QKV4f32, @floatFromInt(imin0[row])) * row_dmin0[row];
        acc[row][1] += @as(QKV4f32, @floatFromInt(iscale1[row])) * row_d1[row] -
            @as(QKV4f32, @floatFromInt(imin1[row])) * row_dmin1[row];
    }
}

fn dotQ5_KQ8_K(w: *const BlockQ5_K, a: *const BlockQ8_K) f32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        const d = f16BitsToF32(w.dm[0]) * a.d;
        const dmin = f16BitsToF32(w.dm[1]) * a.d;
        // d/dmin are constant for this block, so accumulate scale*acc and min*bsum in
        // i32 across the 8 subblocks and apply f32 once at the end — fewer (and more
        // accurate) float ops than a per-subblock f32 multiply-add chain.
        var iscale: i32 = 0;
        var imin: i32 = 0;
        inline for (0..8) |subblock| {
            const scale_min = getScaleMinK4(&w.scales, subblock);
            const acc = dotQ5_KSubblockI32(w, a, subblock);
            const bsum = @as(i32, a.bsums[subblock * 2]) + @as(i32, a.bsums[subblock * 2 + 1]);
            iscale += @as(i32, scale_min.scale) * acc;
            imin += @as(i32, scale_min.min) * bsum;
        }
        return d * @as(f32, @floatFromInt(iscale)) - dmin * @as(f32, @floatFromInt(imin));
    }
    if (comptime has_x86_vnni_ymm) return dotQ5_KQ8_KSimd(.vnni, w, a);
    if (comptime has_x86_avx2) return dotQ5_KQ8_KSimd(.avx2, w, a);
    return dotQ5_KQ8_KSimd(.widen, w, a);
}

/// x86/portable ymm arm of the Q5_K row dot (the decode/GEMV path of
/// `matmulQ5_KRhsTile`): one sub-block step covers 32 contiguous
/// features — 32 qs bytes (nibble + qh bit expanded), the shared 32
/// qh bytes, 32 activation bytes — so the whole 256-feature block is
/// 8 grouped-dot ops. OPERAND SHAPE: the expanded weights are
/// UNSIGNED 5-bit values [0,31], natively vpdpbusd's u8 side (no bias, no
/// correction, no sign trick); activations are unrestricted i8. The
/// per-sub-block scale rides the 8-lane accumulator; one horizontal reduce
/// per block; the mins path is the same scalar bsums fold as the reference.
/// SATURATION (avx2 tier): w ≤ 31 → vpmaddubsw pair sums ≤ 2·31·128 = 7936
/// < 2^15. NO OVERFLOW: |iacc8 lane| ≤ 8·4·31·128·63 < 2^24, reduce < 2^27.
/// Identical i32 totals (order-independent integer adds) and identical f32
/// epilogue as the scalar reference → bit-exact (q5_k_tests.zig). pub for
/// the sibling exact-parity tests.
pub fn dotQ5_KQ8_KSimd(comptime tier: X86DotTier, w: *const BlockQ5_K, a: *const BlockQ8_K) f32 {
    @setEvalBranchQuota(10000);
    var iacc8: QKV8i32 = @splat(0);
    var imin: i32 = 0;
    inline for (0..8) |subblock| {
        const q32: QKV32u8 = @bitCast(w.qs[(subblock / 2) * 32 ..][0..32].*);
        const h32: QKV32u8 = @bitCast(w.qh[0..32].*);
        const high_mask: u8 = @as(u8, 1) << @intCast(subblock);
        const low = if (subblock % 2 == 0) q32 & @as(QKV32u8, @splat(0x0f)) else q32 >> @as(QKV32u8, @splat(4));
        const qs = low + @select(u8, (h32 & @as(QKV32u8, @splat(high_mask))) != @as(QKV32u8, @splat(0)), @as(QKV32u8, @splat(16)), @as(QKV32u8, @splat(0)));
        const act: QKV32i8 = @bitCast(a.qs[subblock * 32 ..][0..32].*);
        const sm = getScaleMinK4(&w.scales, subblock);
        const sum = dotQ5GroupsI32x8(tier, @splat(0), qs, act);
        iacc8 += sum * @as(QKV8i32, @splat(@as(i32, sm.scale)));
        imin += @as(i32, sm.min) * (@as(i32, a.bsums[subblock * 2]) + @as(i32, a.bsums[subblock * 2 + 1]));
    }
    const iscale = @reduce(.Add, iacc8);
    const d = f16BitsToF32(w.dm[0]) * a.d;
    const dmin = f16BitsToF32(w.dm[1]) * a.d;
    return d * @as(f32, @floatFromInt(iscale)) - dmin * @as(f32, @floatFromInt(imin));
}

// pub: the plain-scalar bit-exactness reference for dotQ5_KQ8_KSimd AND the
// aarch64 row-dot arm (q5_k_tests.zig): same integer totals
// (order-independent adds), same f32 epilogue expression.
pub fn dotQ5_KQ8_KScalar(w: *const BlockQ5_K, a: *const BlockQ8_K) f32 {
    var iscale: i32 = 0;
    var imin: i32 = 0;
    var subblock: usize = 0;
    while (subblock < 8) : (subblock += 1) {
        const sm = getScaleMinK4(&w.scales, subblock);
        var dot: i32 = 0;
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            dot += @as(i32, q5KValue(w, subblock, i)) * @as(i32, a.qs[subblock * 32 + i]);
        }
        const bsum = @as(i32, a.bsums[subblock * 2]) + @as(i32, a.bsums[subblock * 2 + 1]);
        iscale += @as(i32, sm.scale) * dot;
        imin += @as(i32, sm.min) * bsum;
    }
    const d = f16BitsToF32(w.dm[0]) * a.d;
    const dmin = f16BitsToF32(w.dm[1]) * a.d;
    return d * @as(f32, @floatFromInt(iscale)) - dmin * @as(f32, @floatFromInt(imin));
}

fn dotQ5_KSubblockI32(w: *const BlockQ5_K, a: *const BlockQ8_K, comptime subblock: usize) i32 {
    const q_offset = (subblock / 2) * 32;
    const a_offset = subblock * 32;
    const high_mask: u8 = @as(u8, 1) << @intCast(subblock);
    const q0: QKV16u8 = @bitCast(w.qs[q_offset..][0..16].*);
    const q1: QKV16u8 = @bitCast(w.qs[q_offset + 16 ..][0..16].*);
    const h0: QKV16u8 = @bitCast(w.qh[0..16].*);
    const h1: QKV16u8 = @bitCast(w.qh[16..32].*);
    var qs0 = if (subblock % 2 == 0)
        q0 & @as(QKV16u8, @splat(0x0f))
    else
        q0 >> @as(QKV16u8, @splat(4));
    var qs1 = if (subblock % 2 == 0)
        q1 & @as(QKV16u8, @splat(0x0f))
    else
        q1 >> @as(QKV16u8, @splat(4));
    const high0 = (h0 & @as(QKV16u8, @splat(high_mask))) != @as(QKV16u8, @splat(0));
    const high1 = (h1 & @as(QKV16u8, @splat(high_mask))) != @as(QKV16u8, @splat(0));
    qs0 += @select(u8, high0, @as(QKV16u8, @splat(16)), @as(QKV16u8, @splat(0)));
    qs1 += @select(u8, high1, @as(QKV16u8, @splat(16)), @as(QKV16u8, @splat(0)));

    const a0_i8: QKV16i8 = @bitCast(a.qs[a_offset..][0..16].*);
    const a1_i8: QKV16i8 = @bitCast(a.qs[a_offset + 16 ..][0..16].*);
    // q5 values are in [0,31] so they fit i8; dot in i32 — NEON sdot where
    // available, i32 multiply-reduce otherwise. Both accumulate in i32 (the old
    // i16 reduce could overflow on the 16-wide sum) and sdot is faster.
    const w0_i8: QKV16i8 = @bitCast(qs0);
    const w1_i8: QKV16i8 = @bitCast(qs1);
    if (comptime builtin.cpu.arch == .aarch64) {
        var acc: QKV4i32 = @splat(0);
        acc = common.sdotI8x16(acc, w0_i8, a0_i8);
        acc = common.sdotI8x16(acc, w1_i8, a1_i8);
        return @reduce(.Add, acc);
    }
    const w0: @Vector(16, i32) = @intCast(qs0);
    const w1: @Vector(16, i32) = @intCast(qs1);
    const a0: @Vector(16, i32) = @intCast(a0_i8);
    const a1: @Vector(16, i32) = @intCast(a1_i8);
    return @reduce(.Add, w0 * a0) + @reduce(.Add, w1 * a1);
}

pub fn dequantizeBlockQ5_KInto(dst: *[qk_k_block_size]f32, src: *const BlockQ5_K) void {
    const d = f16BitsToF32(src.dm[0]);
    const dmin = f16BitsToF32(src.dm[1]);
    var subblock: usize = 0;
    while (subblock < 8) : (subblock += 1) {
        const scale_min = getScaleMinK4(&src.scales, subblock);
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            const q: f32 = @floatFromInt(q5KValue(src, subblock, i));
            dst[subblock * 32 + i] = d * @as(f32, @floatFromInt(scale_min.scale)) * q -
                dmin * @as(f32, @floatFromInt(scale_min.min));
        }
    }
}

/// f32 -> Q5_K encoder for one 256-element block; faithful port of ggml's
/// quantize_row_q5_K_ref (byte-exact, see quant/encode_golden_test.zig).
/// Assumes finite input (no NaN/inf) — see the encoder contract in quant.zig.
pub fn quantizeBlockQ5_KInto(dst: *BlockQ5_K, src: *const [qk_k_block_size]f32) void {
    var L: [qk_k_block_size]u8 = undefined;
    var Laux: [32]u8 = undefined;
    var weights: [32]f32 = undefined;
    var mins: [8]f32 = undefined;
    var scales: [8]f32 = undefined;

    var max_scale: f32 = 0; // as we are deducting the min, scales are always positive
    var max_min: f32 = 0;
    var j: usize = 0;
    while (j < 8) : (j += 1) {
        const xs = src[32 * j ..][0..32];
        var sum_x2: f32 = 0;
        for (xs) |v| sum_x2 += v * v;
        const av_x = @sqrt(sum_x2 / 32);
        for (&weights, xs) |*w, v| w.* = av_x + @abs(v);
        scales[j] = makeQkx2Quants(31, xs, &weights, L[32 * j ..][0..32], &mins[j], &Laux, -0.5, 0.1, 15, false);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    const inv_scale: f32 = if (max_scale > 0) 63.0 / max_scale else 0.0;
    const inv_min: f32 = if (max_min > 0) 63.0 / max_min else 0.0;
    j = 0;
    while (j < 8) : (j += 1) {
        // ggml truncates the rounded int to uint8_t before the 63 clamp.
        var ls: u8 = @truncate(@as(u32, @bitCast(nearestInt(inv_scale * scales[j]))));
        var lm: u8 = @truncate(@as(u32, @bitCast(nearestInt(inv_min * mins[j]))));
        ls = @min(63, ls);
        lm = @min(63, lm);
        if (j < 4) {
            dst.scales[j] = ls;
            dst.scales[j + 4] = lm;
        } else {
            dst.scales[j + 4] = (ls & 0x0f) | ((lm & 0x0f) << 4);
            dst.scales[j - 4] |= (ls >> 4) << 6;
            dst.scales[j] |= (lm >> 4) << 6;
        }
    }
    dst.dm = .{ f32ToF16Bits(max_scale / 63.0), f32ToF16Bits(max_min / 63.0) };

    j = 0;
    while (j < 8) : (j += 1) {
        const sm = getScaleMinK4(&dst.scales, j);
        const d = f16BitsToF32(dst.dm[0]) * @as(f32, @floatFromInt(sm.scale));
        if (d == 0) continue; // keeps the makeQkx2Quants levels, like ggml
        const dm = f16BitsToF32(dst.dm[1]) * @as(f32, @floatFromInt(sm.min));
        for (src[32 * j ..][0..32], L[32 * j ..][0..32]) |v, *l_out| {
            const l = nearestInt((v + dm) / d);
            l_out.* = @intCast(@max(0, @min(31, l)));
        }
    }

    @memset(&dst.qh, 0);
    var qs_offset: usize = 0;
    var n: usize = 0;
    while (n < qk_k_block_size) : (n += 64) {
        const shift: u3 = @intCast((n / 64) * 2);
        const m1 = @as(u8, 1) << shift;
        const m2 = @as(u8, 2) << shift;
        var jj: usize = 0;
        while (jj < 32) : (jj += 1) {
            var l1 = L[n + jj];
            if (l1 > 15) {
                l1 -= 16;
                dst.qh[jj] |= m1;
            }
            var l2 = L[n + jj + 32];
            if (l2 > 15) {
                l2 -= 16;
                dst.qh[jj] |= m2;
            }
            dst.qs[qs_offset + jj] = l1 | (l2 << 4);
        }
        qs_offset += 32;
    }
}

/// f32 -> Q5_K row encoder (caller supplies the output blocks).
pub fn quantizeRowQ5_KInto(dst: []BlockQ5_K, src: []const f32) !void {
    const block_count = try qkBlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;
    for (dst, 0..) |*block, block_index| {
        quantizeBlockQ5_KInto(block, src[block_index * qk_k_block_size ..][0..qk_k_block_size]);
    }
}

fn q5KValue(w: *const BlockQ5_K, subblock: usize, offset: usize) u8 {
    const byte = w.qs[(subblock / 2) * 32 + offset];
    const low = if (subblock % 2 == 0) byte & 0x0f else byte >> 4;
    const high_mask: u8 = @as(u8, 1) << @intCast(subblock);
    return low + if ((w.qh[offset] & high_mask) != 0) @as(u8, 16) else @as(u8, 0);
}

fn setQ5KValue(block: *BlockQ5_K, subblock: usize, offset: usize, value: u8) void {
    const byte_index = (subblock / 2) * 32 + offset;
    if (subblock % 2 == 0) {
        block.qs[byte_index] = (block.qs[byte_index] & 0xf0) | (value & 0x0f);
    } else {
        block.qs[byte_index] = (block.qs[byte_index] & 0x0f) | ((value & 0x0f) << 4);
    }
    const high_mask: u8 = @as(u8, 1) << @intCast(subblock);
    if (value >= 16) {
        block.qh[offset] |= high_mask;
    } else {
        block.qh[offset] &= ~high_mask;
    }
}

fn fillQ5KPattern(block: *BlockQ5_K) void {
    block.dm = .{ f32ToF16Bits(1), f32ToF16Bits(0) };
    block.scales = .{ 1, 2, 3, 4, 0, 0, 0, 0, 1, 2, 3, 4 };
    @memset(&block.qh, 0);
    @memset(&block.qs, 0);
    for (0..8) |subblock| {
        for (0..32) |offset| {
            setQ5KValue(block, subblock, offset, @intCast((subblock * 7 + offset) % 32));
        }
    }
}

test "ggml_q5_k dot and matmul consume loaded blocks" {
    const allocator = std.testing.allocator;

    var q5: BlockQ5_K = undefined;
    fillQ5KPattern(&q5);
    var q8: BlockQ8_K = undefined;
    fillQ8KPattern(&q8);

    var dense_w: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ5_KInto(&dense_w, &q5);
    var dense_a: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ8_KInto(&dense_a, &q8);

    try std.testing.expectEqual(dotDense(&dense_w, &dense_a), dotQ5_KQ8_K(&q5, &q8));

    var rhs_blocks = [_]BlockQ5_K{ q5, q5 };
    var qrhs = try quantizedMatmulRhsQ5_KFromBlocks(allocator, qk_k_block_size, 2, &rhs_blocks);
    defer qrhs.deinit();
    var out: [2]f32 = undefined;
    matmulQ5_KRhsRange(&out, &.{q8}, &qrhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ5_KQ8_K(&q5, &q8), out[0]);
}

test {
    _ = @import("q5_k_tests.zig");
}
