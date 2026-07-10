//! Hot Q4_K quantized matmul kernels (Q4_Kx8 and the comptime-gated Q4_Kx2Mmla
//! smmla path) relocated out of quant.zig. See quant.zig for shared type/helper
//! definitions; every shared symbol this module references is aliased from
//! quant.zig (`qm`) below so the moved bodies compile unchanged.

const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("../../tensor.zig");
const q8k_mod = @import("q8k.zig");
const types_mod = @import("types.zig");
const common = @import("common.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;

// Shared symbols defined in quant.zig, aliased here so the moved bodies compile unchanged.
const BlockQ4_K = types_mod.BlockQ4_K;
const BlockQ4_Kx2Mmla = types_mod.BlockQ4_Kx2Mmla;
const BlockQ4_Kx4 = types_mod.BlockQ4_Kx4;
const BlockQ4_Kx8 = types_mod.BlockQ4_Kx8;
const BlockQ8_K = types_mod.BlockQ8_K;
const BlockQ8_Kx2Mmla = types_mod.BlockQ8_Kx2Mmla;
const BlockQ8_Kx4 = types_mod.BlockQ8_Kx4;
const QKV16i16 = common.QKV16i16;
const QKV16i8 = common.QKV16i8;
const QKV16u8 = common.QKV16u8;
const QKV32i8 = common.QKV32i8;
const QKV32u8 = common.QKV32u8;
const QKV4f32 = common.QKV4f32;
const QKV4i32 = common.QKV4i32;
const QKV8i32 = common.QKV8i32;
const addHalvesI32x8 = common.addHalvesI32x8;
const dotI8GroupsWidenI32x8 = common.dotI8GroupsWidenI32x8;
const dpbusdI32x8 = common.dpbusdI32x8;
const has_x86_vnni_ymm = common.has_x86_vnni_ymm;
const maddubsDotGroupsI32x8 = common.maddubsDotGroupsI32x8;
const QuantizedFormatError = types_mod.QuantizedFormatError;
const QuantizedMatmulRhsQ4_K = types_mod.QuantizedMatmulRhsQ4_K;
const QuantizedMatmulRhsQ4_Kx2Mmla = types_mod.QuantizedMatmulRhsQ4_Kx2Mmla;
const QuantizedMatmulRhsQ4_Kx4 = types_mod.QuantizedMatmulRhsQ4_Kx4;
const QuantizedMatmulRhsQ4_Kx8 = types_mod.QuantizedMatmulRhsQ4_Kx8;
const checkedProduct = types_mod.checkedProduct;
const dequantizeBlockQ8_KInto = q8k_mod.dequantizeBlockQ8_KInto;
const dotDense = common.dotDense;
const dotU8I8x16Portable = common.dotU8I8x16Portable;
const f16BitsToF32 = common.f16BitsToF32;
const f16x4BitsToF32 = common.f16x4BitsToF32;
const f32ToF16Bits = common.f32ToF16Bits;
const fillQ8KPattern = q8k_mod.fillQ8KPattern;
const getScaleMinK4 = q8k_mod.getScaleMinK4;
const has_aarch64_i8mm = common.has_aarch64_i8mm;
const has_x86_avx2 = common.has_x86_avx2;
const makeQkx2Quants = q8k_mod.makeQkx2Quants;
const nearestInt = q8k_mod.nearestInt;
const q4HighNibbleI8 = common.q4HighNibbleI8;
const q4Kx8D = q8k_mod.q4Kx8D;
const q4Kx8Scales = q8k_mod.q4Kx8Scales;
const q4LowNibbleI8 = common.q4LowNibbleI8;
const q4_kx8_row_block = common.q4_kx8_row_block;
const q8_0_row_block = common.q8_0_row_block;
const qkBlockCount = q8k_mod.qkBlockCount;
const qk_col_block = common.qk_col_block;
const qk_k_block_size = types_mod.qk_k_block_size;
const quantizeRowsQ8_K = q8k_mod.quantizeRowsQ8_K;
const quantizeRowsQ8_Kx2MmlaInto = q8k_mod.quantizeRowsQ8_Kx2MmlaInto;
const quantizeRowsQ8_Kx4PaddedInto = q8k_mod.quantizeRowsQ8_Kx4PaddedInto;
const quantizedMatmulRhsQ4_KFromBlocks = q8k_mod.quantizedMatmulRhsQ4_KFromBlocks;
const sdotI8x16Lane = common.sdotI8x16Lane;
const smmlaI8x16 = common.smmlaI8x16;

pub fn packMatmulRhsQ4_Kx4(
    allocator: Allocator,
    blocks: []const BlockQ4_K,
    n: usize,
    k: usize,
    blocks_per_row: usize,
) !QuantizedMatmulRhsQ4_Kx4 {
    if (n % 4 != 0) return tensor.TensorError.InvalidShape;
    if (blocks_per_row != try qkBlockCount(k)) return tensor.TensorError.InvalidShape;
    if (blocks.len != try checkedProduct(n, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;

    const group_count = n / 4;
    const packed_blocks = try allocator.alloc(BlockQ4_Kx4, try checkedProduct(group_count, blocks_per_row));
    errdefer allocator.free(packed_blocks);

    for (0..group_count) |group_i| {
        for (0..blocks_per_row) |block_i| {
            const b0 = &blocks[(4 * group_i + 0) * blocks_per_row + block_i];
            const b1 = &blocks[(4 * group_i + 1) * blocks_per_row + block_i];
            const b2 = &blocks[(4 * group_i + 2) * blocks_per_row + block_i];
            const b3 = &blocks[(4 * group_i + 3) * blocks_per_row + block_i];
            const cols = [_]*const BlockQ4_K{ b0, b1, b2, b3 };
            var dst = &packed_blocks[group_i * blocks_per_row + block_i];
            dst.d = .{ b0.dm[0], b1.dm[0], b2.dm[0], b3.dm[0] };
            dst.dmin = .{ b0.dm[1], b1.dm[1], b2.dm[1], b3.dm[1] };

            for (0..8) |subblock| {
                inline for (0..4) |col| {
                    const scale_min = getScaleMinK4(&cols[col].scales, subblock);
                    dst.scales[subblock * 4 + col] = scale_min.scale;
                    dst.mins[subblock * 4 + col] = scale_min.min;
                }

                for (0..8) |feature_group| {
                    for (0..4) |col| {
                        const block = cols[col];
                        for (0..4) |lane| {
                            const feature_offset = feature_group * 4 + lane;
                            dst.qs[subblock * 128 + feature_group * 16 + col * 4 + lane] =
                                @intCast(q4KValue(block, subblock, feature_offset));
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

pub fn packMatmulRhsQ4_Kx8(
    allocator: Allocator,
    blocks: []const BlockQ4_K,
    n: usize,
    k: usize,
    blocks_per_row: usize,
) !QuantizedMatmulRhsQ4_Kx8 {
    if (n % 8 != 0) return tensor.TensorError.InvalidShape;
    if (blocks_per_row != try qkBlockCount(k)) return tensor.TensorError.InvalidShape;
    if (blocks.len != try checkedProduct(n, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;

    const group_count = n / 8;
    const packed_blocks = try allocator.alloc(BlockQ4_Kx8, try checkedProduct(group_count, blocks_per_row));
    errdefer allocator.free(packed_blocks);

    for (0..group_count) |group_i| {
        for (0..blocks_per_row) |block_i| {
            const cols = [_]*const BlockQ4_K{
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

            for (0..4) |subblock_pair| {
                for (0..8) |feature_group| {
                    inline for (0..8) |col| {
                        const block = cols[col];
                        inline for (0..4) |lane| {
                            const feature_offset = feature_group * 4 + lane;
                            dst.qs[subblock_pair * 256 + feature_group * 32 + col * 4 + lane] =
                                block.qs[subblock_pair * 32 + feature_offset];
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

pub fn packMatmulRhsQ4_Kx2Mmla(
    allocator: Allocator,
    blocks: []const BlockQ4_K,
    n: usize,
    k: usize,
    blocks_per_row: usize,
) !QuantizedMatmulRhsQ4_Kx2Mmla {
    if (n % 2 != 0) return tensor.TensorError.InvalidShape;
    if (blocks_per_row != try qkBlockCount(k)) return tensor.TensorError.InvalidShape;
    if (blocks.len != try checkedProduct(n, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;

    const group_count = n / 2;
    const packed_blocks = try allocator.alloc(BlockQ4_Kx2Mmla, try checkedProduct(group_count, blocks_per_row));
    errdefer allocator.free(packed_blocks);

    for (0..group_count) |group_i| {
        for (0..blocks_per_row) |block_i| {
            const cols = [_]*const BlockQ4_K{
                &blocks[(2 * group_i + 0) * blocks_per_row + block_i],
                &blocks[(2 * group_i + 1) * blocks_per_row + block_i],
            };
            var dst = &packed_blocks[group_i * blocks_per_row + block_i];

            inline for (0..2) |col| {
                dst.d[col] = cols[col].dm[0];
                dst.dmin[col] = cols[col].dm[1];
            }

            for (0..8) |subblock| {
                inline for (0..2) |col| {
                    const scale_min = getScaleMinK4(&cols[col].scales, subblock);
                    dst.scales[subblock * 2 + col] = scale_min.scale;
                    dst.mins[subblock * 2 + col] = scale_min.min;
                }

                inline for (0..2) |half| {
                    const base = subblock * 64 + half * 32;
                    inline for (0..8) |lane| {
                        dst.qs[base + lane] = @intCast(q4KValue(cols[0], subblock, half * 16 + lane));
                        dst.qs[base + 8 + lane] = @intCast(q4KValue(cols[1], subblock, half * 16 + lane));
                        dst.qs[base + 16 + lane] = @intCast(q4KValue(cols[0], subblock, half * 16 + 8 + lane));
                        dst.qs[base + 24 + lane] = @intCast(q4KValue(cols[1], subblock, half * 16 + 8 + lane));
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

const moe_row_tile = 4;

/// Unpack one 32-wide Q4_K sub-block's nibbles into two i8x16 vectors
/// (values in [0,15]). The q5_k unpack minus the qh high bit.
inline fn unpackQ4_KSubblock(w: *const BlockQ4_K, comptime subblock: usize) [2]QKV16i8 {
    const q_offset = (subblock / 2) * 32;
    const q0: QKV16u8 = @bitCast(w.qs[q_offset..][0..16].*);
    const q1: QKV16u8 = @bitCast(w.qs[q_offset + 16 ..][0..16].*);
    const qs0 = if (subblock % 2 == 0) q0 & @as(QKV16u8, @splat(0x0f)) else q0 >> @as(QKV16u8, @splat(4));
    const qs1 = if (subblock % 2 == 0) q1 & @as(QKV16u8, @splat(0x0f)) else q1 >> @as(QKV16u8, @splat(4));
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
    // the sign-trick vpmaddubsw path (w ∈ [0,15] here; a comes from BlockQ8_K,
    // i.e. quantizeRowQ8_KInto's -127/max scale construction, so a ∈ [-127,127]
    // — inside the sign-trick exactness domain; see common.zig).
    return common.dotI8x16Portable(w0, a0) + common.dotI8x16Portable(w1, a1);
}

/// Column-outer Q4_K matmul for the m>1 (batched MoE prefill) case: unpack each
/// weight block's nibbles ONCE, then sdot them against a tile of LHS rows —
/// amortizing the unpack over the batch instead of re-unpacking per row like the
/// row-outer `matmulQ4_KRhsTile`. Numerically identical to it (same per-block
/// deferred-f32 reduction, same cross-block accumulation order).
pub fn matmulQ4_KRhsCompactColOuter(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ4_K,
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
                    const wv = unpackQ4_KSubblock(w, subblock);
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

/// Per-row activation sum of one Q4_K sub-block (= two Q8_K 16-groups, `2*subblock`
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

/// Four rows' i8 dot of one 32-wide Q4_K sub-block against pre-unpacked weights
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
// Q8_Kx4 load (the 4-dword-source analog of `broadcastPairGroupsI32x8`).
inline fn broadcastPairGroupsI32x4(comptime c: comptime_int, v: QKV4i32) QKV8i32 {
    return @shuffle(i32, v, undefined, [8]i32{ 2 * c, 2 * c, 2 * c, 2 * c, 2 * c + 1, 2 * c + 1, 2 * c + 1, 2 * c + 1 });
}

/// x86/portable ymm arms of `dot4RowsSubblockQ8_Kx4` (the MoE-prefill
/// 4-row lane dot): each 32-byte Q8_Kx4 activation load already holds
/// [fg: 4 rows × 4 features | fg+1: …] dword-per-row, so
/// broadcasting the matching weight dword pair (`broadcastPairGroupsI32x4`)
/// turns the 32-feature × 4-row sub-block dot into four grouped-dot ops + one
/// half-fold — no per-row rebuild. OPERAND SHAPE: `wv` holds UNSIGNED nibble
/// values [0,15] (see `unpackQ4_KSubblock`) — natively vpdpbusd's u8 side,
/// dotted directly (no bias, no correction, no sign trick); activations are
/// unrestricted i8. SATURATION (avx2 tier): w ≤ 15 → vpmaddubsw pair sums ≤
/// 2·15·128 = 3840 << 2^15. NO OVERFLOW: |sum8 lane| ≤ 4·4·15·128 < 2^15.
/// Integer sums are order-independent → bit-identical to
/// `dot4RowsSubblockQ8_Kx4Scalar` (q4_k_tests.zig). pub for the sibling
/// exact-parity tests.
pub fn dot4RowsSubblockQ8_Kx4Simd(comptime tier: Q4DotTier, a: *const BlockQ8_Kx4, comptime subblock: usize, wv: [2]QKV16i8) QKV4i32 {
    var sum8: QKV8i32 = @splat(0);
    inline for (0..4) |c| {
        const act: QKV32i8 = @bitCast(a.qs[subblock * 128 + c * 32 ..][0..32].*);
        const wb: QKV32u8 = @bitCast(broadcastPairGroupsI32x4(c % 2, @as(QKV4i32, @bitCast(wv[c / 2]))));
        sum8 = dotNibbleGroupsI32x8(tier, sum8, wb, act);
    }
    return addHalvesI32x8(sum8);
}

// pub: the bit-exactness reference for dot4RowsSubblockQ8_Kx4Simd
// (q4_k_tests.zig) — the plain per-row rebuild over the interleaved Q8_Kx4
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

/// Column-outer Q4_K matmul over **4-row-interleaved Q8_Kx4** activations. Like
/// `matmulQ4_KRhsCompactColOuter` it unpacks each weight sub-block once and reuses it
/// across the row tile, but it packs the four rows into the `sdot` lanes
/// (`dot4RowsSubblockQ8_Kx4`) so the four rows share one i32x4 accumulator with no
/// per-row horizontal reduction, and the deferred-f32 epilogue runs vector-wide over
/// the four rows. `lhs_blocks` holds `ceil(m/4)` Q8_Kx4 groups per K-block (tail rows
/// zero-padded, e.g. via `quantizeRowsQ8_Kx4PaddedInto`); `m` is the real row count so
/// padded lanes are never stored. Bit-identical to the per-row column-outer / row-outer
/// tile (same integer reduction, same cross-block f32 accumulation order).
pub fn matmulQ4_KCompactQ8_Kx4ColOuter(
    out: []f32,
    lhs_blocks: []const BlockQ8_Kx4,
    rhs: *const QuantizedMatmulRhsQ4_K,
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
                    const wv = unpackQ4_KSubblock(w, subblock);
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

pub fn matmulQ4_KRhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ4_K,
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
                    acc[c] += dotQ4_KQ8_K(rhs_block, lhs_block);
                }
            }
            inline for (0..qk_col_block) |c| out[i * n + j + c] = acc[c];
        }

        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc += dotQ4_KQ8_K(&rhs_col[block_index], &lhs_row[block_index]);
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn matmulQ4_KRhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ4_K,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ4_KRhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ4_Kx4RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ4_Kx4,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    std.debug.assert(c0 % 4 == 0);
    std.debug.assert(c1 % 4 == 0);

    const blocks_per_row = rhs.blocks_per_group;
    var i = r0;
    while (i + q8_0_row_block <= r1) : (i += q8_0_row_block) {
        var j = c0;
        while (j < c1) : (j += 4) {
            const rhs_group = rhs.groupBlocks(j / 4);
            var acc: [q8_0_row_block]QKV4f32 = undefined;
            inline for (0..q8_0_row_block) |r| acc[r] = @splat(0);

            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const rhs_block = &rhs_group[block_index];
                accumulateQ4_Kx4Rows(lhs_blocks, i, blocks_per_row, block_index, rhs_block, &acc);
            }

            inline for (0..q8_0_row_block) |r| {
                out[(i + r) * n + j + 0] = acc[r][0];
                out[(i + r) * n + j + 1] = acc[r][1];
                out[(i + r) * n + j + 2] = acc[r][2];
                out[(i + r) * n + j + 3] = acc[r][3];
            }
        }
    }

    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;
        while (j < c1) : (j += 4) {
            const rhs_group = rhs.groupBlocks(j / 4);
            var acc: QKV4f32 = @splat(0);
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc = accumulateQ4_Kx4(&lhs_row[block_index], &rhs_group[block_index], acc);
            }
            out[i * n + j + 0] = acc[0];
            out[i * n + j + 1] = acc[1];
            out[i * n + j + 2] = acc[2];
            out[i * n + j + 3] = acc[3];
        }
    }
}

pub fn matmulQ4_Kx4RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ4_Kx4,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ4_Kx4RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ4_Kx8RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ4_Kx8,
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
                accumulateQ4_Kx8Rows(lhs_blocks, i, blocks_per_row, block_index, rhs_block, &acc);
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
        matmulQ4_Kx8RhsTailRows(8, out, lhs_blocks, rhs, n, i, c0, c1, blocks_per_row);
    }

    while (i + 4 <= r1) : (i += 4) {
        matmulQ4_Kx8RhsTailRows(4, out, lhs_blocks, rhs, n, i, c0, c1, blocks_per_row);
    }

    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;
        while (j < c1) : (j += 8) {
            const rhs_group = rhs.groupBlocks(j / 8);
            var acc: [2]QKV4f32 = .{ @splat(0), @splat(0) };
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                accumulateQ4_Kx8(&lhs_row[block_index], &rhs_group[block_index], &acc);
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

fn matmulQ4_Kx8RhsTailRows(
    comptime row_block: usize,
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ4_Kx8,
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
                accumulateQ4_Kx8(lhs, rhs_block, &acc[r]);
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

pub fn matmulQ4_Kx8RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ4_Kx8,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ4_Kx8RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ4_Kx8Q8_Kx4RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_Kx4,
    rhs: *const QuantizedMatmulRhsQ4_Kx8,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    std.debug.assert(r0 % 4 == 0);
    std.debug.assert(c0 % 8 == 0);
    std.debug.assert(c1 % 8 == 0);

    const blocks_per_row = rhs.blocks_per_group;
    var i = r0;
    while (i < r1) : (i += 4) {
        const full_row_group = i + 4 <= r1;
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
                accumulateQ4_Kx8Q8_Kx4(&lhs_blocks[lhs_row_group + block_index], &rhs_group[block_index], &acc);
            }

            if (full_row_group) {
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
            } else {
                const valid_rows = r1 - i;
                inline for (0..4) |row| {
                    if (row < valid_rows) {
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
    }
}

pub fn matmulQ4_Kx8Q8_Kx4RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_Kx4,
    rhs: *const QuantizedMatmulRhsQ4_Kx8,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ4_Kx8Q8_Kx4RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ4_Kx2MmlaRhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ4_Kx2Mmla,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    std.debug.assert(c0 % 2 == 0);
    std.debug.assert(c1 % 2 == 0);

    const blocks_per_row = rhs.blocks_per_group;
    var i = r0;
    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;
        while (j < c1) : (j += 2) {
            const rhs_group = rhs.groupBlocks(j / 2);
            var acc = [_]f32{ 0, 0 };
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                accumulateQ4_Kx2MmlaRow(&lhs_row[block_index], &rhs_group[block_index], &acc);
            }
            out[i * n + j + 0] = acc[0];
            out[i * n + j + 1] = acc[1];
        }
    }
}

pub fn matmulQ4_Kx2MmlaRhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ4_Kx2Mmla,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ4_Kx2MmlaRhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_Kx2Mmla,
    rhs: *const QuantizedMatmulRhsQ4_Kx2Mmla,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    std.debug.assert(r0 % 2 == 0);
    std.debug.assert(r1 % 2 == 0);
    std.debug.assert(c0 % 2 == 0);
    std.debug.assert(c1 % 2 == 0);

    const blocks_per_row = rhs.blocks_per_group;
    var i = r0;
    while (i < r1) : (i += 2) {
        var j = c0;
        while (j < c1) : (j += 2) {
            const rhs_group = rhs.groupBlocks(j / 2);
            const lhs_row_group = (i / 2) * blocks_per_row;
            var acc: QKV4f32 = @splat(0);

            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                accumulateQ4_Kx2Mmla(&lhs_blocks[lhs_row_group + block_index], &rhs_group[block_index], &acc);
            }

            out[(i + 0) * n + j + 0] = acc[0];
            out[(i + 1) * n + j + 0] = acc[1];
            out[(i + 0) * n + j + 1] = acc[2];
            out[(i + 1) * n + j + 1] = acc[3];
        }
    }
}

pub fn matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_Kx2Mmla,
    rhs: *const QuantizedMatmulRhsQ4_Kx2Mmla,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

fn dotQ4_KQ8_K(w: *const BlockQ4_K, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.dm[0]) * a.d;
    const dmin = f16BitsToF32(w.dm[1]) * a.d;
    // d/dmin are constant for this block, so accumulate scale*acc and min*bsum in
    // i32 across the 8 subblocks and apply f32 once at the end — fewer (and more
    // accurate) float ops than a per-subblock f32 multiply-add chain (the same
    // structure as dotQ5_KQ8_K). Bounds: |acc| <= 32*15*127 ≈ 61k, x scale <= 63,
    // x 8 subblocks ≈ 31M — comfortably i32.
    var iscale: i32 = 0;
    var imin: i32 = 0;
    inline for (0..8) |subblock| {
        const scale_min = getScaleMinK4(&w.scales, subblock);
        const acc = dotQ4_KSubblockI32(w, a, subblock);
        const bsum = @as(i32, a.bsums[subblock * 2]) + @as(i32, a.bsums[subblock * 2 + 1]);
        iscale += @as(i32, scale_min.scale) * acc;
        imin += @as(i32, scale_min.min) * bsum;
    }
    return d * @as(f32, @floatFromInt(iscale)) - dmin * @as(f32, @floatFromInt(imin));
}

fn dotQ4_KSubblockI32(w: *const BlockQ4_K, a: *const BlockQ8_K, comptime subblock: usize) i32 {
    const q_offset = (subblock / 2) * 32;
    const a_offset = subblock * 32;
    const q0: QKV16u8 = @bitCast(w.qs[q_offset..][0..16].*);
    const q1: QKV16u8 = @bitCast(w.qs[q_offset + 16 ..][0..16].*);
    const qs0 = if (subblock % 2 == 0)
        q0 & @as(QKV16u8, @splat(0x0f))
    else
        q0 >> @as(QKV16u8, @splat(4));
    const qs1 = if (subblock % 2 == 0)
        q1 & @as(QKV16u8, @splat(0x0f))
    else
        q1 >> @as(QKV16u8, @splat(4));

    const a0_i8: QKV16i8 = @bitCast(a.qs[a_offset..][0..16].*);
    const a1_i8: QKV16i8 = @bitCast(a.qs[a_offset + 16 ..][0..16].*);

    if (comptime has_x86_avx2) {
        // u8 nibbles (0..15) x i8 activations on the AVX2 vpmaddubsw path:
        // pair sums are bounded by 2*15*128 = 3840 << 32767, so the saturating
        // u8·i8 multiply-add cannot saturate — exact i32, bit-equal to the
        // reduce below. Under VNNI the same call lowers to vpdpbusd instead.
        return dotU8I8x16Portable(qs0, a0_i8) + dotU8I8x16Portable(qs1, a1_i8);
    }

    // q4 nibbles are in [0,15] so they fit i8; dot in i32 — NEON sdot where
    // available (the missing arm that left the q4_k GEMV i16-widen-bound on
    // Apple Silicon; q5_k/q6_k already had it), i32 multiply-reduce otherwise.
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

fn accumulateQ4_Kx4(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx4, acc: QKV4f32) QKV4f32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ4_Kx4Aarch64(lhs, rhs, acc);
    }
    if (comptime has_x86_vnni_ymm) {
        return accumulateQ4_Kx4Vnni(lhs, rhs, acc);
    }
    if (comptime has_x86_avx2) {
        return accumulateQ4_Kx4Avx2(lhs, rhs, acc);
    }
    return accumulateQ4_Kx4Widen(lhs, rhs, acc);
}

fn accumulateQ4_Kx4Rows(
    lhs_blocks: []const BlockQ8_K,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: *const BlockQ4_Kx4,
    acc: *[q8_0_row_block]QKV4f32,
) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ4_Kx4RowsAarch64(lhs_blocks, row_start, blocks_per_row, block_index, rhs, acc);
    }
    inline for (0..q8_0_row_block) |r| {
        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
        acc[r] = accumulateQ4_Kx4(lhs, rhs, acc[r]);
    }
}

fn accumulateQ4_Kx4Aarch64(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx4, acc: QKV4f32) QKV4f32 {
    const d = f16x4BitsToF32(rhs.d) * @as(QKV4f32, @splat(lhs.d));
    const dmin = f16x4BitsToF32(rhs.dmin) * @as(QKV4f32, @splat(lhs.d));
    var out = acc;

    inline for (0..8) |subblock| {
        var dot: QKV4i32 = @splat(0);
        inline for (0..2) |half| {
            const lhs_vec: QKV16i8 = @bitCast(lhs.qs[subblock * 32 + half * 16 ..][0..16].*);
            inline for (0..4) |feature_group| {
                const rhs_vec: QKV16i8 = @bitCast(rhs.qs[subblock * 128 + half * 64 + feature_group * 16 ..][0..16].*);
                dot = sdotI8x16Lane(feature_group, dot, rhs_vec, lhs_vec);
            }
        }

        const scales: QKV4i32 = .{
            rhs.scales[subblock * 4 + 0],
            rhs.scales[subblock * 4 + 1],
            rhs.scales[subblock * 4 + 2],
            rhs.scales[subblock * 4 + 3],
        };
        const mins: QKV4i32 = .{
            rhs.mins[subblock * 4 + 0],
            rhs.mins[subblock * 4 + 1],
            rhs.mins[subblock * 4 + 2],
            rhs.mins[subblock * 4 + 3],
        };
        const bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[subblock * 2]) + @as(i32, lhs.bsums[subblock * 2 + 1]));
        out += @as(QKV4f32, @floatFromInt(dot * scales)) * d -
            @as(QKV4f32, @floatFromInt(bsum * mins)) * dmin;
    }

    return out;
}

fn accumulateQ4_Kx4RowsAarch64(
    lhs_blocks: []const BlockQ8_K,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: *const BlockQ4_Kx4,
    acc: *[q8_0_row_block]QKV4f32,
) void {
    const rhs_d = f16x4BitsToF32(rhs.d);
    const rhs_dmin = f16x4BitsToF32(rhs.dmin);

    inline for (0..8) |subblock| {
        const rhs00: QKV16i8 = @bitCast(rhs.qs[subblock * 128 + 0 * 64 + 0 * 16 ..][0..16].*);
        const rhs01: QKV16i8 = @bitCast(rhs.qs[subblock * 128 + 0 * 64 + 1 * 16 ..][0..16].*);
        const rhs02: QKV16i8 = @bitCast(rhs.qs[subblock * 128 + 0 * 64 + 2 * 16 ..][0..16].*);
        const rhs03: QKV16i8 = @bitCast(rhs.qs[subblock * 128 + 0 * 64 + 3 * 16 ..][0..16].*);
        const rhs10: QKV16i8 = @bitCast(rhs.qs[subblock * 128 + 1 * 64 + 0 * 16 ..][0..16].*);
        const rhs11: QKV16i8 = @bitCast(rhs.qs[subblock * 128 + 1 * 64 + 1 * 16 ..][0..16].*);
        const rhs12: QKV16i8 = @bitCast(rhs.qs[subblock * 128 + 1 * 64 + 2 * 16 ..][0..16].*);
        const rhs13: QKV16i8 = @bitCast(rhs.qs[subblock * 128 + 1 * 64 + 3 * 16 ..][0..16].*);
        const scales: QKV4i32 = .{
            rhs.scales[subblock * 4 + 0],
            rhs.scales[subblock * 4 + 1],
            rhs.scales[subblock * 4 + 2],
            rhs.scales[subblock * 4 + 3],
        };
        const mins: QKV4i32 = .{
            rhs.mins[subblock * 4 + 0],
            rhs.mins[subblock * 4 + 1],
            rhs.mins[subblock * 4 + 2],
            rhs.mins[subblock * 4 + 3],
        };

        inline for (0..q8_0_row_block) |r| {
            const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
            const lhs0: QKV16i8 = @bitCast(lhs.qs[subblock * 32 + 0 * 16 ..][0..16].*);
            const lhs1: QKV16i8 = @bitCast(lhs.qs[subblock * 32 + 1 * 16 ..][0..16].*);
            var dot: QKV4i32 = @splat(0);
            dot = sdotI8x16Lane(0, dot, rhs00, lhs0);
            dot = sdotI8x16Lane(1, dot, rhs01, lhs0);
            dot = sdotI8x16Lane(2, dot, rhs02, lhs0);
            dot = sdotI8x16Lane(3, dot, rhs03, lhs0);
            dot = sdotI8x16Lane(0, dot, rhs10, lhs1);
            dot = sdotI8x16Lane(1, dot, rhs11, lhs1);
            dot = sdotI8x16Lane(2, dot, rhs12, lhs1);
            dot = sdotI8x16Lane(3, dot, rhs13, lhs1);

            const d = rhs_d * @as(QKV4f32, @splat(lhs.d));
            const dmin = rhs_dmin * @as(QKV4f32, @splat(lhs.d));
            const bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[subblock * 2]) + @as(i32, lhs.bsums[subblock * 2 + 1]));
            acc[r] += @as(QKV4f32, @floatFromInt(dot * scales)) * d -
                @as(QKV4f32, @floatFromInt(bsum * mins)) * dmin;
        }
    }
}

// pub: the bit-exactness reference for the x86/portable SIMD arms below (q4_k_tests.zig).
pub fn accumulateQ4_Kx4Scalar(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx4, acc: QKV4f32) QKV4f32 {
    @setEvalBranchQuota(10000); // non-aarch64 fallback fully unrolls 8*32*4 inline iters
    const d = f16x4BitsToF32(rhs.d) * @as(QKV4f32, @splat(lhs.d));
    const dmin = f16x4BitsToF32(rhs.dmin) * @as(QKV4f32, @splat(lhs.d));
    var out = acc;

    inline for (0..8) |subblock| {
        var dot: QKV4i32 = @splat(0);
        inline for (0..32) |feature_offset| {
            const lhs_value: i32 = lhs.qs[subblock * 32 + feature_offset];
            inline for (0..4) |col| {
                dot[col] += lhs_value * @as(i32, rhs.qs[subblock * 128 + (feature_offset / 4) * 16 + col * 4 + feature_offset % 4]);
            }
        }

        const scales: QKV4i32 = .{
            rhs.scales[subblock * 4 + 0],
            rhs.scales[subblock * 4 + 1],
            rhs.scales[subblock * 4 + 2],
            rhs.scales[subblock * 4 + 3],
        };
        const mins: QKV4i32 = .{
            rhs.mins[subblock * 4 + 0],
            rhs.mins[subblock * 4 + 1],
            rhs.mins[subblock * 4 + 2],
            rhs.mins[subblock * 4 + 3],
        };
        const bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[subblock * 2]) + @as(i32, lhs.bsums[subblock * 2 + 1]));
        out += @as(QKV4f32, @floatFromInt(dot * scales)) * d -
            @as(QKV4f32, @floatFromInt(bsum * mins)) * dmin;
    }

    return out;
}

fn accumulateQ4_Kx8(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx8, acc: *[2]QKV4f32) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ4_Kx8Aarch64(lhs, rhs, acc);
    }
    if (comptime has_x86_vnni_ymm) {
        return accumulateQ4_Kx8Vnni(lhs, rhs, acc);
    }
    if (comptime has_x86_avx2) {
        return accumulateQ4_Kx8Avx2(lhs, rhs, acc);
    }
    return accumulateQ4_Kx8Widen(lhs, rhs, acc);
}

fn accumulateQ4_Kx8Rows(
    lhs_blocks: []const BlockQ8_K,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: *const BlockQ4_Kx8,
    acc: *[q4_kx8_row_block][2]QKV4f32,
) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ4_Kx8RowsAarch64(lhs_blocks, row_start, blocks_per_row, block_index, rhs, acc);
    }
    inline for (0..q4_kx8_row_block) |r| {
        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
        accumulateQ4_Kx8(lhs, rhs, &acc[r]);
    }
}

fn q4Kx2MmlaScales(values: *const [8 * 2]u8, comptime subblock: usize) QKV4i32 {
    return .{
        values[subblock * 2 + 0],
        values[subblock * 2 + 0],
        values[subblock * 2 + 1],
        values[subblock * 2 + 1],
    };
}

fn q4Kx2MmlaD(rhs: *const BlockQ4_Kx2Mmla, lhs: *const BlockQ8_Kx2Mmla) QKV4f32 {
    const d0 = f16BitsToF32(rhs.d[0]);
    const d1 = f16BitsToF32(rhs.d[1]);
    return .{
        d0 * lhs.d[0],
        d0 * lhs.d[1],
        d1 * lhs.d[0],
        d1 * lhs.d[1],
    };
}

fn q4Kx2MmlaDmin(rhs: *const BlockQ4_Kx2Mmla, lhs: *const BlockQ8_Kx2Mmla) QKV4f32 {
    const dmin0 = f16BitsToF32(rhs.dmin[0]);
    const dmin1 = f16BitsToF32(rhs.dmin[1]);
    return .{
        dmin0 * lhs.d[0],
        dmin0 * lhs.d[1],
        dmin1 * lhs.d[0],
        dmin1 * lhs.d[1],
    };
}

fn q4Kx2MmlaValue(qs: *const [qk_k_block_size * 2]i8, subblock: usize, feature: usize, lane: usize) i8 {
    const half = feature / 16;
    const local = feature % 16;
    const base = subblock * 64 + half * 32;
    return if (local < 8)
        qs[base + lane * 8 + local]
    else
        qs[base + 16 + lane * 8 + local - 8];
}

fn accumulateQ4_Kx2Mmla(lhs: *const BlockQ8_Kx2Mmla, rhs: *const BlockQ4_Kx2Mmla, acc: *QKV4f32) void {
    var scaled: QKV4i32 = @splat(0);
    var bias: QKV4i32 = @splat(0);

    inline for (0..8) |subblock| {
        var dot: QKV4i32 = @splat(0);
        if (comptime has_aarch64_i8mm) {
            inline for (0..2) |half| {
                const base = subblock * 64 + half * 32;
                const q4_lo: QKV16i8 = @bitCast(rhs.qs[base..][0..16].*);
                const q4_hi: QKV16i8 = @bitCast(rhs.qs[base + 16 ..][0..16].*);
                const q8_lo: QKV16i8 = @bitCast(lhs.qs[base..][0..16].*);
                const q8_hi: QKV16i8 = @bitCast(lhs.qs[base + 16 ..][0..16].*);
                dot = smmlaI8x16(dot, q4_lo, q8_lo);
                dot = smmlaI8x16(dot, q4_hi, q8_hi);
            }
        } else {
            inline for (0..32) |feature| {
                const q40: i32 = q4Kx2MmlaValue(&rhs.qs, subblock, feature, 0);
                const q41: i32 = q4Kx2MmlaValue(&rhs.qs, subblock, feature, 1);
                const q80: i32 = q4Kx2MmlaValue(&lhs.qs, subblock, feature, 0);
                const q81: i32 = q4Kx2MmlaValue(&lhs.qs, subblock, feature, 1);
                dot[0] += q40 * q80;
                dot[1] += q40 * q81;
                dot[2] += q41 * q80;
                dot[3] += q41 * q81;
            }
        }

        const scales = q4Kx2MmlaScales(&rhs.scales, subblock);
        const mins = q4Kx2MmlaScales(&rhs.mins, subblock);
        const bsums: QKV4i32 = .{
            lhs.bsums[subblock * 2 + 0],
            lhs.bsums[subblock * 2 + 1],
            lhs.bsums[subblock * 2 + 0],
            lhs.bsums[subblock * 2 + 1],
        };
        scaled += dot * scales;
        bias += bsums * mins;
    }

    acc.* += @as(QKV4f32, @floatFromInt(scaled)) * q4Kx2MmlaD(rhs, lhs) -
        @as(QKV4f32, @floatFromInt(bias)) * q4Kx2MmlaDmin(rhs, lhs);
}

fn accumulateQ4_Kx2MmlaRow(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx2Mmla, acc: *[2]f32) void {
    const d = [_]f32{
        f16BitsToF32(rhs.d[0]) * lhs.d,
        f16BitsToF32(rhs.d[1]) * lhs.d,
    };
    const dmin = [_]f32{
        f16BitsToF32(rhs.dmin[0]) * lhs.d,
        f16BitsToF32(rhs.dmin[1]) * lhs.d,
    };

    inline for (0..8) |subblock| {
        var dot = [_]i32{ 0, 0 };
        inline for (0..32) |feature| {
            const q8: i32 = lhs.qs[subblock * 32 + feature];
            inline for (0..2) |col| {
                dot[col] += @as(i32, q4Kx2MmlaValue(&rhs.qs, subblock, feature, col)) * q8;
            }
        }

        const bsum: i32 = @as(i32, lhs.bsums[subblock * 2]) + @as(i32, lhs.bsums[subblock * 2 + 1]);
        inline for (0..2) |col| {
            const scale: f32 = @floatFromInt(rhs.scales[subblock * 2 + col]);
            const min: f32 = @floatFromInt(rhs.mins[subblock * 2 + col]);
            acc[col] += @as(f32, @floatFromInt(dot[col])) * scale * d[col] -
                @as(f32, @floatFromInt(bsum)) * min * dmin[col];
        }
    }
}

fn accumulateQ4_Kx8Q8_Kx4(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ4_Kx8, acc: *[4][2]QKV4f32) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ4_Kx8Q8_Kx4Aarch64(lhs, rhs, acc);
    }
    if (comptime has_x86_vnni_ymm) {
        return accumulateQ4_Kx8Q8_Kx4Vnni(lhs, rhs, acc);
    }
    if (comptime has_x86_avx2) {
        return accumulateQ4_Kx8Q8_Kx4Avx2(lhs, rhs, acc);
    }
    return accumulateQ4_Kx8Q8_Kx4Widen(lhs, rhs, acc);
}

fn accumulateQ4_Kx8Q8_Kx4Aarch64(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ4_Kx8, acc: *[4][2]QKV4f32) void {
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
    var bias0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var bias1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };

    inline for (0..4) |pair| {
        const even_subblock = pair * 2;
        const odd_subblock = even_subblock + 1;
        var dot_even0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
        var dot_even1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
        var dot_odd0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
        var dot_odd1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };

        inline for (0..8) |feature_group| {
            const rhs_offset = pair * 256 + feature_group * 32;
            const rhs0: QKV16u8 = @bitCast(rhs.qs[rhs_offset..][0..16].*);
            const rhs1: QKV16u8 = @bitCast(rhs.qs[rhs_offset + 16 ..][0..16].*);
            const rhs0_lo = q4LowNibbleI8(rhs0);
            const rhs1_lo = q4LowNibbleI8(rhs1);
            const rhs0_hi = q4HighNibbleI8(rhs0);
            const rhs1_hi = q4HighNibbleI8(rhs1);
            const q8_even: QKV16i8 = @bitCast(lhs.qs[pair * 256 + feature_group * 16 ..][0..16].*);
            const q8_odd: QKV16i8 = @bitCast(lhs.qs[pair * 256 + 128 + feature_group * 16 ..][0..16].*);
            inline for (0..4) |row| {
                dot_even0[row] = sdotI8x16Lane(row, dot_even0[row], rhs0_lo, q8_even);
                dot_even1[row] = sdotI8x16Lane(row, dot_even1[row], rhs1_lo, q8_even);
                dot_odd0[row] = sdotI8x16Lane(row, dot_odd0[row], rhs0_hi, q8_odd);
                dot_odd1[row] = sdotI8x16Lane(row, dot_odd1[row], rhs1_hi, q8_odd);
            }
        }

        const even_scales0 = q4Kx8Scales(&rhs.scales, even_subblock, 0);
        const even_scales1 = q4Kx8Scales(&rhs.scales, even_subblock, 1);
        const odd_scales0 = q4Kx8Scales(&rhs.scales, odd_subblock, 0);
        const odd_scales1 = q4Kx8Scales(&rhs.scales, odd_subblock, 1);
        const even_mins0 = q4Kx8Scales(&rhs.mins, even_subblock, 0);
        const even_mins1 = q4Kx8Scales(&rhs.mins, even_subblock, 1);
        const odd_mins0 = q4Kx8Scales(&rhs.mins, odd_subblock, 0);
        const odd_mins1 = q4Kx8Scales(&rhs.mins, odd_subblock, 1);

        inline for (0..4) |row| {
            const even_bsum0 = even_subblock * 2;
            const odd_bsum0 = odd_subblock * 2;
            const even_bsum: QKV4i32 = @splat(
                @as(i32, lhs.bsums[(even_bsum0 / 4) * 16 + row * 4 + even_bsum0 % 4]) +
                    @as(i32, lhs.bsums[((even_bsum0 + 1) / 4) * 16 + row * 4 + (even_bsum0 + 1) % 4]),
            );
            const odd_bsum: QKV4i32 = @splat(
                @as(i32, lhs.bsums[(odd_bsum0 / 4) * 16 + row * 4 + odd_bsum0 % 4]) +
                    @as(i32, lhs.bsums[((odd_bsum0 + 1) / 4) * 16 + row * 4 + (odd_bsum0 + 1) % 4]),
            );
            const sum0 = dot_even0[row] * even_scales0 + dot_odd0[row] * odd_scales0;
            const sum1 = dot_even1[row] * even_scales1 + dot_odd1[row] * odd_scales1;
            acc[row][0] += @as(QKV4f32, @floatFromInt(sum0)) * row_d0[row];
            acc[row][1] += @as(QKV4f32, @floatFromInt(sum1)) * row_d1[row];
            bias0[row] += even_bsum * even_mins0 + odd_bsum * odd_mins0;
            bias1[row] += even_bsum * even_mins1 + odd_bsum * odd_mins1;
        }
    }

    inline for (0..4) |row| {
        acc[row][0] -= @as(QKV4f32, @floatFromInt(bias0[row])) * row_dmin0[row];
        acc[row][1] -= @as(QKV4f32, @floatFromInt(bias1[row])) * row_dmin1[row];
    }
}

fn accumulateQ4_Kx8Aarch64(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx8, acc: *[2]QKV4f32) void {
    const d0 = q4Kx8D(rhs.d, 0) * @as(QKV4f32, @splat(lhs.d));
    const d1 = q4Kx8D(rhs.d, 1) * @as(QKV4f32, @splat(lhs.d));
    const dmin0 = q4Kx8D(rhs.dmin, 0) * @as(QKV4f32, @splat(lhs.d));
    const dmin1 = q4Kx8D(rhs.dmin, 1) * @as(QKV4f32, @splat(lhs.d));

    inline for (0..4) |pair| {
        const even_subblock = pair * 2;
        const odd_subblock = even_subblock + 1;
        var dot_even0: QKV4i32 = @splat(0);
        var dot_even1: QKV4i32 = @splat(0);
        var dot_odd0: QKV4i32 = @splat(0);
        var dot_odd1: QKV4i32 = @splat(0);
        inline for (0..2) |half| {
            const lhs_even: QKV16i8 = @bitCast(lhs.qs[even_subblock * 32 + half * 16 ..][0..16].*);
            const lhs_odd: QKV16i8 = @bitCast(lhs.qs[odd_subblock * 32 + half * 16 ..][0..16].*);
            inline for (0..4) |feature_group| {
                const rhs_offset = pair * 256 + (half * 4 + feature_group) * 32;
                const rhs_vec0: QKV16u8 = @bitCast(rhs.qs[rhs_offset..][0..16].*);
                const rhs_vec1: QKV16u8 = @bitCast(rhs.qs[rhs_offset + 16 ..][0..16].*);
                dot_even0 = sdotI8x16Lane(feature_group, dot_even0, q4LowNibbleI8(rhs_vec0), lhs_even);
                dot_even1 = sdotI8x16Lane(feature_group, dot_even1, q4LowNibbleI8(rhs_vec1), lhs_even);
                dot_odd0 = sdotI8x16Lane(feature_group, dot_odd0, q4HighNibbleI8(rhs_vec0), lhs_odd);
                dot_odd1 = sdotI8x16Lane(feature_group, dot_odd1, q4HighNibbleI8(rhs_vec1), lhs_odd);
            }
        }

        const even_scales0 = q4Kx8Scales(&rhs.scales, even_subblock, 0);
        const even_scales1 = q4Kx8Scales(&rhs.scales, even_subblock, 1);
        const odd_scales0 = q4Kx8Scales(&rhs.scales, odd_subblock, 0);
        const odd_scales1 = q4Kx8Scales(&rhs.scales, odd_subblock, 1);
        const even_mins0 = q4Kx8Scales(&rhs.mins, even_subblock, 0);
        const even_mins1 = q4Kx8Scales(&rhs.mins, even_subblock, 1);
        const odd_mins0 = q4Kx8Scales(&rhs.mins, odd_subblock, 0);
        const odd_mins1 = q4Kx8Scales(&rhs.mins, odd_subblock, 1);
        const even_bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[even_subblock * 2]) + @as(i32, lhs.bsums[even_subblock * 2 + 1]));
        const odd_bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[odd_subblock * 2]) + @as(i32, lhs.bsums[odd_subblock * 2 + 1]));
        acc[0] += @as(QKV4f32, @floatFromInt(dot_even0 * even_scales0)) * d0 -
            @as(QKV4f32, @floatFromInt(even_bsum * even_mins0)) * dmin0;
        acc[1] += @as(QKV4f32, @floatFromInt(dot_even1 * even_scales1)) * d1 -
            @as(QKV4f32, @floatFromInt(even_bsum * even_mins1)) * dmin1;
        acc[0] += @as(QKV4f32, @floatFromInt(dot_odd0 * odd_scales0)) * d0 -
            @as(QKV4f32, @floatFromInt(odd_bsum * odd_mins0)) * dmin0;
        acc[1] += @as(QKV4f32, @floatFromInt(dot_odd1 * odd_scales1)) * d1 -
            @as(QKV4f32, @floatFromInt(odd_bsum * odd_mins1)) * dmin1;
    }
}

fn accumulateQ4_Kx8RowsAarch64(
    lhs_blocks: []const BlockQ8_K,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: *const BlockQ4_Kx8,
    acc: *[q4_kx8_row_block][2]QKV4f32,
) void {
    const rhs_d0 = q4Kx8D(rhs.d, 0);
    const rhs_d1 = q4Kx8D(rhs.d, 1);
    const rhs_dmin0 = q4Kx8D(rhs.dmin, 0);
    const rhs_dmin1 = q4Kx8D(rhs.dmin, 1);
    var row_d0: [q4_kx8_row_block]QKV4f32 = undefined;
    var row_d1: [q4_kx8_row_block]QKV4f32 = undefined;
    var row_dmin0: [q4_kx8_row_block]QKV4f32 = undefined;
    var row_dmin1: [q4_kx8_row_block]QKV4f32 = undefined;
    inline for (0..q4_kx8_row_block) |r| {
        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
        const lhs_d: QKV4f32 = @splat(lhs.d);
        row_d0[r] = rhs_d0 * lhs_d;
        row_d1[r] = rhs_d1 * lhs_d;
        row_dmin0[r] = rhs_dmin0 * lhs_d;
        row_dmin1[r] = rhs_dmin1 * lhs_d;
    }

    inline for (0..4) |pair| {
        const even_subblock = pair * 2;
        const odd_subblock = even_subblock + 1;
        const rhs000: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 0 * 32 + 0 ..][0..16].*);
        const rhs001: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 0 * 32 + 16 ..][0..16].*);
        const rhs010: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 1 * 32 + 0 ..][0..16].*);
        const rhs011: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 1 * 32 + 16 ..][0..16].*);
        const rhs020: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 2 * 32 + 0 ..][0..16].*);
        const rhs021: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 2 * 32 + 16 ..][0..16].*);
        const rhs030: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 3 * 32 + 0 ..][0..16].*);
        const rhs031: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 3 * 32 + 16 ..][0..16].*);
        const rhs100: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 4 * 32 + 0 ..][0..16].*);
        const rhs101: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 4 * 32 + 16 ..][0..16].*);
        const rhs110: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 5 * 32 + 0 ..][0..16].*);
        const rhs111: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 5 * 32 + 16 ..][0..16].*);
        const rhs120: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 6 * 32 + 0 ..][0..16].*);
        const rhs121: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 6 * 32 + 16 ..][0..16].*);
        const rhs130: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 7 * 32 + 0 ..][0..16].*);
        const rhs131: QKV16u8 = @bitCast(rhs.qs[pair * 256 + 7 * 32 + 16 ..][0..16].*);
        const rhs000_lo = q4LowNibbleI8(rhs000);
        const rhs001_lo = q4LowNibbleI8(rhs001);
        const rhs010_lo = q4LowNibbleI8(rhs010);
        const rhs011_lo = q4LowNibbleI8(rhs011);
        const rhs020_lo = q4LowNibbleI8(rhs020);
        const rhs021_lo = q4LowNibbleI8(rhs021);
        const rhs030_lo = q4LowNibbleI8(rhs030);
        const rhs031_lo = q4LowNibbleI8(rhs031);
        const rhs100_lo = q4LowNibbleI8(rhs100);
        const rhs101_lo = q4LowNibbleI8(rhs101);
        const rhs110_lo = q4LowNibbleI8(rhs110);
        const rhs111_lo = q4LowNibbleI8(rhs111);
        const rhs120_lo = q4LowNibbleI8(rhs120);
        const rhs121_lo = q4LowNibbleI8(rhs121);
        const rhs130_lo = q4LowNibbleI8(rhs130);
        const rhs131_lo = q4LowNibbleI8(rhs131);
        const rhs000_hi = q4HighNibbleI8(rhs000);
        const rhs001_hi = q4HighNibbleI8(rhs001);
        const rhs010_hi = q4HighNibbleI8(rhs010);
        const rhs011_hi = q4HighNibbleI8(rhs011);
        const rhs020_hi = q4HighNibbleI8(rhs020);
        const rhs021_hi = q4HighNibbleI8(rhs021);
        const rhs030_hi = q4HighNibbleI8(rhs030);
        const rhs031_hi = q4HighNibbleI8(rhs031);
        const rhs100_hi = q4HighNibbleI8(rhs100);
        const rhs101_hi = q4HighNibbleI8(rhs101);
        const rhs110_hi = q4HighNibbleI8(rhs110);
        const rhs111_hi = q4HighNibbleI8(rhs111);
        const rhs120_hi = q4HighNibbleI8(rhs120);
        const rhs121_hi = q4HighNibbleI8(rhs121);
        const rhs130_hi = q4HighNibbleI8(rhs130);
        const rhs131_hi = q4HighNibbleI8(rhs131);
        const even_scales0 = q4Kx8Scales(&rhs.scales, even_subblock, 0);
        const even_scales1 = q4Kx8Scales(&rhs.scales, even_subblock, 1);
        const odd_scales0 = q4Kx8Scales(&rhs.scales, odd_subblock, 0);
        const odd_scales1 = q4Kx8Scales(&rhs.scales, odd_subblock, 1);
        const even_mins0 = q4Kx8Scales(&rhs.mins, even_subblock, 0);
        const even_mins1 = q4Kx8Scales(&rhs.mins, even_subblock, 1);
        const odd_mins0 = q4Kx8Scales(&rhs.mins, odd_subblock, 0);
        const odd_mins1 = q4Kx8Scales(&rhs.mins, odd_subblock, 1);

        inline for (0..q4_kx8_row_block) |r| {
            const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
            const lhs_even0: QKV16i8 = @bitCast(lhs.qs[even_subblock * 32 + 0 * 16 ..][0..16].*);
            const lhs_even1: QKV16i8 = @bitCast(lhs.qs[even_subblock * 32 + 1 * 16 ..][0..16].*);
            const lhs_odd0: QKV16i8 = @bitCast(lhs.qs[odd_subblock * 32 + 0 * 16 ..][0..16].*);
            const lhs_odd1: QKV16i8 = @bitCast(lhs.qs[odd_subblock * 32 + 1 * 16 ..][0..16].*);
            var dot_even0: QKV4i32 = @splat(0);
            var dot_even1: QKV4i32 = @splat(0);
            var dot_odd0: QKV4i32 = @splat(0);
            var dot_odd1: QKV4i32 = @splat(0);
            dot_even0 = sdotI8x16Lane(0, dot_even0, rhs000_lo, lhs_even0);
            dot_even1 = sdotI8x16Lane(0, dot_even1, rhs001_lo, lhs_even0);
            dot_odd0 = sdotI8x16Lane(0, dot_odd0, rhs000_hi, lhs_odd0);
            dot_odd1 = sdotI8x16Lane(0, dot_odd1, rhs001_hi, lhs_odd0);
            dot_even0 = sdotI8x16Lane(1, dot_even0, rhs010_lo, lhs_even0);
            dot_even1 = sdotI8x16Lane(1, dot_even1, rhs011_lo, lhs_even0);
            dot_odd0 = sdotI8x16Lane(1, dot_odd0, rhs010_hi, lhs_odd0);
            dot_odd1 = sdotI8x16Lane(1, dot_odd1, rhs011_hi, lhs_odd0);
            dot_even0 = sdotI8x16Lane(2, dot_even0, rhs020_lo, lhs_even0);
            dot_even1 = sdotI8x16Lane(2, dot_even1, rhs021_lo, lhs_even0);
            dot_odd0 = sdotI8x16Lane(2, dot_odd0, rhs020_hi, lhs_odd0);
            dot_odd1 = sdotI8x16Lane(2, dot_odd1, rhs021_hi, lhs_odd0);
            dot_even0 = sdotI8x16Lane(3, dot_even0, rhs030_lo, lhs_even0);
            dot_even1 = sdotI8x16Lane(3, dot_even1, rhs031_lo, lhs_even0);
            dot_odd0 = sdotI8x16Lane(3, dot_odd0, rhs030_hi, lhs_odd0);
            dot_odd1 = sdotI8x16Lane(3, dot_odd1, rhs031_hi, lhs_odd0);
            dot_even0 = sdotI8x16Lane(0, dot_even0, rhs100_lo, lhs_even1);
            dot_even1 = sdotI8x16Lane(0, dot_even1, rhs101_lo, lhs_even1);
            dot_odd0 = sdotI8x16Lane(0, dot_odd0, rhs100_hi, lhs_odd1);
            dot_odd1 = sdotI8x16Lane(0, dot_odd1, rhs101_hi, lhs_odd1);
            dot_even0 = sdotI8x16Lane(1, dot_even0, rhs110_lo, lhs_even1);
            dot_even1 = sdotI8x16Lane(1, dot_even1, rhs111_lo, lhs_even1);
            dot_odd0 = sdotI8x16Lane(1, dot_odd0, rhs110_hi, lhs_odd1);
            dot_odd1 = sdotI8x16Lane(1, dot_odd1, rhs111_hi, lhs_odd1);
            dot_even0 = sdotI8x16Lane(2, dot_even0, rhs120_lo, lhs_even1);
            dot_even1 = sdotI8x16Lane(2, dot_even1, rhs121_lo, lhs_even1);
            dot_odd0 = sdotI8x16Lane(2, dot_odd0, rhs120_hi, lhs_odd1);
            dot_odd1 = sdotI8x16Lane(2, dot_odd1, rhs121_hi, lhs_odd1);
            dot_even0 = sdotI8x16Lane(3, dot_even0, rhs130_lo, lhs_even1);
            dot_even1 = sdotI8x16Lane(3, dot_even1, rhs131_lo, lhs_even1);
            dot_odd0 = sdotI8x16Lane(3, dot_odd0, rhs130_hi, lhs_odd1);
            dot_odd1 = sdotI8x16Lane(3, dot_odd1, rhs131_hi, lhs_odd1);

            const even_bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[even_subblock * 2]) + @as(i32, lhs.bsums[even_subblock * 2 + 1]));
            const odd_bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[odd_subblock * 2]) + @as(i32, lhs.bsums[odd_subblock * 2 + 1]));
            acc[r][0] += @as(QKV4f32, @floatFromInt(dot_even0 * even_scales0)) * row_d0[r] -
                @as(QKV4f32, @floatFromInt(even_bsum * even_mins0)) * row_dmin0[r];
            acc[r][1] += @as(QKV4f32, @floatFromInt(dot_even1 * even_scales1)) * row_d1[r] -
                @as(QKV4f32, @floatFromInt(even_bsum * even_mins1)) * row_dmin1[r];
            acc[r][0] += @as(QKV4f32, @floatFromInt(dot_odd0 * odd_scales0)) * row_d0[r] -
                @as(QKV4f32, @floatFromInt(odd_bsum * odd_mins0)) * row_dmin0[r];
            acc[r][1] += @as(QKV4f32, @floatFromInt(dot_odd1 * odd_scales1)) * row_d1[r] -
                @as(QKV4f32, @floatFromInt(odd_bsum * odd_mins1)) * row_dmin1[r];
        }
    }
}

// pub: the bit-exactness reference for the x86/portable SIMD arms below (q4_k_tests.zig).
pub fn accumulateQ4_Kx8Scalar(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx8, acc: *[2]QKV4f32) void {
    const d0 = q4Kx8D(rhs.d, 0) * @as(QKV4f32, @splat(lhs.d));
    const d1 = q4Kx8D(rhs.d, 1) * @as(QKV4f32, @splat(lhs.d));
    const dmin0 = q4Kx8D(rhs.dmin, 0) * @as(QKV4f32, @splat(lhs.d));
    const dmin1 = q4Kx8D(rhs.dmin, 1) * @as(QKV4f32, @splat(lhs.d));

    inline for (0..4) |pair| {
        const even_subblock = pair * 2;
        const odd_subblock = even_subblock + 1;
        var dot_even0: QKV4i32 = @splat(0);
        var dot_even1: QKV4i32 = @splat(0);
        var dot_odd0: QKV4i32 = @splat(0);
        var dot_odd1: QKV4i32 = @splat(0);
        inline for (0..32) |feature_offset| {
            const lhs_even: i32 = lhs.qs[even_subblock * 32 + feature_offset];
            const lhs_odd: i32 = lhs.qs[odd_subblock * 32 + feature_offset];
            inline for (0..4) |col| {
                const q_offset = pair * 256 + (feature_offset / 4) * 32 + col * 4 + feature_offset % 4;
                const q0 = rhs.qs[q_offset];
                const q1 = rhs.qs[q_offset + 16];
                dot_even0[col] += lhs_even * @as(i32, @intCast(q0 & 0x0f));
                dot_even1[col] += lhs_even * @as(i32, @intCast(q1 & 0x0f));
                dot_odd0[col] += lhs_odd * @as(i32, @intCast(q0 >> 4));
                dot_odd1[col] += lhs_odd * @as(i32, @intCast(q1 >> 4));
            }
        }

        const even_scales0 = q4Kx8Scales(&rhs.scales, even_subblock, 0);
        const even_scales1 = q4Kx8Scales(&rhs.scales, even_subblock, 1);
        const odd_scales0 = q4Kx8Scales(&rhs.scales, odd_subblock, 0);
        const odd_scales1 = q4Kx8Scales(&rhs.scales, odd_subblock, 1);
        const even_mins0 = q4Kx8Scales(&rhs.mins, even_subblock, 0);
        const even_mins1 = q4Kx8Scales(&rhs.mins, even_subblock, 1);
        const odd_mins0 = q4Kx8Scales(&rhs.mins, odd_subblock, 0);
        const odd_mins1 = q4Kx8Scales(&rhs.mins, odd_subblock, 1);
        const even_bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[even_subblock * 2]) + @as(i32, lhs.bsums[even_subblock * 2 + 1]));
        const odd_bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[odd_subblock * 2]) + @as(i32, lhs.bsums[odd_subblock * 2 + 1]));
        acc[0] += @as(QKV4f32, @floatFromInt(dot_even0 * even_scales0)) * d0 -
            @as(QKV4f32, @floatFromInt(even_bsum * even_mins0)) * dmin0;
        acc[1] += @as(QKV4f32, @floatFromInt(dot_even1 * even_scales1)) * d1 -
            @as(QKV4f32, @floatFromInt(even_bsum * even_mins1)) * dmin1;
        acc[0] += @as(QKV4f32, @floatFromInt(dot_odd0 * odd_scales0)) * d0 -
            @as(QKV4f32, @floatFromInt(odd_bsum * odd_mins0)) * dmin0;
        acc[1] += @as(QKV4f32, @floatFromInt(dot_odd1 * odd_scales1)) * d1 -
            @as(QKV4f32, @floatFromInt(odd_bsum * odd_mins1)) * dmin1;
    }
}

// --- x86 / portable-SIMD arms of the packed Q4_K accumulates -----------------
//
// Every arm below computes the SAME i32 group sums as the scalar arms
// (bit-identical integer accumulation) and applies the f16 scales/mins with
// the scalar arms' exact f32 expression order, so results are bit-for-bit
// equal to the scalar reference — asserted by q4_k_tests.zig. Operand shape
// (differs from q8_0): the packed weight side is UNSIGNED nibble-expanded
// values in [0,15] and the Q8_K activation side is signed i8 — natively
// vpdpbusd's u8·i8 shape, so the VNNI tier needs no bias and no sign trick.
// The arms are written over the comptime-gated primitives in common.zig, so
// each also compiles and runs on any target via the portable twins (that is
// how the aarch64 dev machine exercises them). They are pub for the sibling
// exact-parity tests.

// The three x86 tiers differ ONLY in which grouped u8·i8 dot primitive they
// use; loads, broadcasts, and epilogues are shared.
// pub: q4_k_tests.zig iterates the tiers when asserting the exact-parity
// suites of the tier-parameterized arms.
pub const Q4DotTier = enum { vnni, avx2, widen };

/// One grouped weight·activation dot-accumulate step; `w` holds nibble values
/// in [0,15], `a` is unrestricted i8. Exactness per tier:
///   vnni : vpdpbusd — exact i32 for all u8·i8 inputs, no saturation.
///   avx2 : vpmaddubsw+vpmaddwd — pair sums bounded by 2·15·128 = 3840
///          << 32767, saturation-free, exact.
///   widen: nibble values fit i8 (@bitCast is value-preserving), and i8·i8
///          products are exact in the widening dot on every target.
inline fn dotNibbleGroupsI32x8(comptime tier: Q4DotTier, acc: QKV8i32, w: QKV32u8, a: QKV32i8) QKV8i32 {
    return switch (tier) {
        .vnni => dpbusdI32x8(acc, w, a),
        .avx2 => maddubsDotGroupsI32x8(acc, w, a),
        .widen => dotI8GroupsWidenI32x8(acc, @bitCast(w), a),
    };
}

// vpermd-class (cross-lane): broadcast dword 2g to the low 128-bit lane and
// dword 2g+1 to the high lane — aligns plain-LHS feature groups 2g/2g+1
// against the two 16-byte column chunks of one 32-byte packed-RHS load (the
// same shape as q8_0.zig's helper of the same name).
inline fn broadcastPairGroupsI32x8(comptime g: comptime_int, v: QKV8i32) QKV8i32 {
    return @shuffle(i32, v, undefined, [8]i32{ 2 * g, 2 * g, 2 * g, 2 * g, 2 * g + 1, 2 * g + 1, 2 * g + 1, 2 * g + 1 });
}

// vpbroadcastd-class: splat one 4-byte activation group across all eight
// dword groups of a ymm (a single broadcast load from memory).
inline fn broadcastGroupI8x32(bytes: *const [4]i8) QKV32i8 {
    const dword: i32 = @bitCast(bytes.*);
    return @bitCast(@as(QKV8i32, @splat(dword)));
}

// Split the eight i32 group sums of a ymm accumulator into the two 4-column
// halves of the Q4_Kx8 chunk layout (cols 0..3 / cols 4..7).
inline fn lowHalfI32x8(v: QKV8i32) QKV4i32 {
    return @shuffle(i32, v, undefined, [4]i32{ 0, 1, 2, 3 });
}

inline fn highHalfI32x8(v: QKV8i32) QKV4i32 {
    return @shuffle(i32, v, undefined, [4]i32{ 4, 5, 6, 7 });
}

pub fn accumulateQ4_Kx4Vnni(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx4, acc: QKV4f32) QKV4f32 {
    return accumulateQ4_Kx4X86(.vnni, lhs, rhs, acc);
}

pub fn accumulateQ4_Kx4Avx2(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx4, acc: QKV4f32) QKV4f32 {
    return accumulateQ4_Kx4X86(.avx2, lhs, rhs, acc);
}

pub fn accumulateQ4_Kx4Widen(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx4, acc: QKV4f32) QKV4f32 {
    return accumulateQ4_Kx4X86(.widen, lhs, rhs, acc);
}

fn accumulateQ4_Kx4X86(comptime tier: Q4DotTier, lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx4, acc: QKV4f32) QKV4f32 {
    const d = f16x4BitsToF32(rhs.d) * @as(QKV4f32, @splat(lhs.d));
    const dmin = f16x4BitsToF32(rhs.dmin) * @as(QKV4f32, @splat(lhs.d));
    var out = acc;

    inline for (0..8) |subblock| {
        const lhs_groups: QKV8i32 = @bitCast(lhs.qs[subblock * 32 ..][0..32].*);
        var sum: QKV8i32 = @splat(0);
        inline for (0..4) |g| {
            // 32-byte chunk = feature groups 2g/2g+1, cols 0..3 each; the x4
            // pack stores nibble-EXPANDED values in [0,15], one byte each.
            const w: QKV32u8 = @bitCast(rhs.qs[subblock * 128 + g * 32 ..][0..32].*);
            const bcast: QKV32i8 = @bitCast(broadcastPairGroupsI32x8(g, lhs_groups));
            sum = dotNibbleGroupsI32x8(tier, sum, w, bcast);
        }
        const dot = addHalvesI32x8(sum);

        const scales: QKV4i32 = .{
            rhs.scales[subblock * 4 + 0],
            rhs.scales[subblock * 4 + 1],
            rhs.scales[subblock * 4 + 2],
            rhs.scales[subblock * 4 + 3],
        };
        const mins: QKV4i32 = .{
            rhs.mins[subblock * 4 + 0],
            rhs.mins[subblock * 4 + 1],
            rhs.mins[subblock * 4 + 2],
            rhs.mins[subblock * 4 + 3],
        };
        const bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[subblock * 2]) + @as(i32, lhs.bsums[subblock * 2 + 1]));
        out += @as(QKV4f32, @floatFromInt(dot * scales)) * d -
            @as(QKV4f32, @floatFromInt(bsum * mins)) * dmin;
    }

    return out;
}

pub fn accumulateQ4_Kx8Vnni(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx8, acc: *[2]QKV4f32) void {
    return accumulateQ4_Kx8X86(.vnni, lhs, rhs, acc);
}

pub fn accumulateQ4_Kx8Avx2(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx8, acc: *[2]QKV4f32) void {
    return accumulateQ4_Kx8X86(.avx2, lhs, rhs, acc);
}

pub fn accumulateQ4_Kx8Widen(lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx8, acc: *[2]QKV4f32) void {
    return accumulateQ4_Kx8X86(.widen, lhs, rhs, acc);
}

fn accumulateQ4_Kx8X86(comptime tier: Q4DotTier, lhs: *const BlockQ8_K, rhs: *const BlockQ4_Kx8, acc: *[2]QKV4f32) void {
    const d0 = q4Kx8D(rhs.d, 0) * @as(QKV4f32, @splat(lhs.d));
    const d1 = q4Kx8D(rhs.d, 1) * @as(QKV4f32, @splat(lhs.d));
    const dmin0 = q4Kx8D(rhs.dmin, 0) * @as(QKV4f32, @splat(lhs.d));
    const dmin1 = q4Kx8D(rhs.dmin, 1) * @as(QKV4f32, @splat(lhs.d));

    inline for (0..4) |pair| {
        const even_subblock = pair * 2;
        const odd_subblock = even_subblock + 1;
        var sum_even: QKV8i32 = @splat(0);
        var sum_odd: QKV8i32 = @splat(0);
        inline for (0..8) |g| {
            // 32-byte chunk = one feature group across all 8 columns (cols
            // 0..3 then 4..7); low nibbles = even sub-block, high = odd.
            const q: QKV32u8 = @bitCast(rhs.qs[pair * 256 + g * 32 ..][0..32].*);
            const w_even = q & @as(QKV32u8, @splat(0x0f));
            const w_odd = q >> @as(QKV32u8, @splat(4));
            const a_even = broadcastGroupI8x32(lhs.qs[even_subblock * 32 + g * 4 ..][0..4]);
            const a_odd = broadcastGroupI8x32(lhs.qs[odd_subblock * 32 + g * 4 ..][0..4]);
            sum_even = dotNibbleGroupsI32x8(tier, sum_even, w_even, a_even);
            sum_odd = dotNibbleGroupsI32x8(tier, sum_odd, w_odd, a_odd);
        }
        const dot_even0 = lowHalfI32x8(sum_even);
        const dot_even1 = highHalfI32x8(sum_even);
        const dot_odd0 = lowHalfI32x8(sum_odd);
        const dot_odd1 = highHalfI32x8(sum_odd);

        const even_scales0 = q4Kx8Scales(&rhs.scales, even_subblock, 0);
        const even_scales1 = q4Kx8Scales(&rhs.scales, even_subblock, 1);
        const odd_scales0 = q4Kx8Scales(&rhs.scales, odd_subblock, 0);
        const odd_scales1 = q4Kx8Scales(&rhs.scales, odd_subblock, 1);
        const even_mins0 = q4Kx8Scales(&rhs.mins, even_subblock, 0);
        const even_mins1 = q4Kx8Scales(&rhs.mins, even_subblock, 1);
        const odd_mins0 = q4Kx8Scales(&rhs.mins, odd_subblock, 0);
        const odd_mins1 = q4Kx8Scales(&rhs.mins, odd_subblock, 1);
        const even_bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[even_subblock * 2]) + @as(i32, lhs.bsums[even_subblock * 2 + 1]));
        const odd_bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[odd_subblock * 2]) + @as(i32, lhs.bsums[odd_subblock * 2 + 1]));
        acc[0] += @as(QKV4f32, @floatFromInt(dot_even0 * even_scales0)) * d0 -
            @as(QKV4f32, @floatFromInt(even_bsum * even_mins0)) * dmin0;
        acc[1] += @as(QKV4f32, @floatFromInt(dot_even1 * even_scales1)) * d1 -
            @as(QKV4f32, @floatFromInt(even_bsum * even_mins1)) * dmin1;
        acc[0] += @as(QKV4f32, @floatFromInt(dot_odd0 * odd_scales0)) * d0 -
            @as(QKV4f32, @floatFromInt(odd_bsum * odd_mins0)) * dmin0;
        acc[1] += @as(QKV4f32, @floatFromInt(dot_odd1 * odd_scales1)) * d1 -
            @as(QKV4f32, @floatFromInt(odd_bsum * odd_mins1)) * dmin1;
    }
}

pub fn accumulateQ4_Kx8Q8_Kx4Vnni(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ4_Kx8, acc: *[4][2]QKV4f32) void {
    return accumulateQ4_Kx8Q8_Kx4X86(.vnni, lhs, rhs, acc);
}

pub fn accumulateQ4_Kx8Q8_Kx4Avx2(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ4_Kx8, acc: *[4][2]QKV4f32) void {
    return accumulateQ4_Kx8Q8_Kx4X86(.avx2, lhs, rhs, acc);
}

pub fn accumulateQ4_Kx8Q8_Kx4Widen(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ4_Kx8, acc: *[4][2]QKV4f32) void {
    return accumulateQ4_Kx8Q8_Kx4X86(.widen, lhs, rhs, acc);
}

fn accumulateQ4_Kx8Q8_Kx4X86(comptime tier: Q4DotTier, lhs: *const BlockQ8_Kx4, rhs: *const BlockQ4_Kx8, acc: *[4][2]QKV4f32) void {
    @setEvalBranchQuota(4000);
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
    var bias0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var bias1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };

    inline for (0..4) |pair| {
        const even_subblock = pair * 2;
        const odd_subblock = even_subblock + 1;
        var sum_even: [4]QKV8i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
        var sum_odd: [4]QKV8i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };

        inline for (0..8) |g| {
            const q: QKV32u8 = @bitCast(rhs.qs[pair * 256 + g * 32 ..][0..32].*);
            const w_even = q & @as(QKV32u8, @splat(0x0f));
            const w_odd = q >> @as(QKV32u8, @splat(4));
            inline for (0..4) |row| {
                const a_even = broadcastGroupI8x32(lhs.qs[pair * 256 + g * 16 + row * 4 ..][0..4]);
                const a_odd = broadcastGroupI8x32(lhs.qs[pair * 256 + 128 + g * 16 + row * 4 ..][0..4]);
                sum_even[row] = dotNibbleGroupsI32x8(tier, sum_even[row], w_even, a_even);
                sum_odd[row] = dotNibbleGroupsI32x8(tier, sum_odd[row], w_odd, a_odd);
            }
        }

        const even_scales0 = q4Kx8Scales(&rhs.scales, even_subblock, 0);
        const even_scales1 = q4Kx8Scales(&rhs.scales, even_subblock, 1);
        const odd_scales0 = q4Kx8Scales(&rhs.scales, odd_subblock, 0);
        const odd_scales1 = q4Kx8Scales(&rhs.scales, odd_subblock, 1);
        const even_mins0 = q4Kx8Scales(&rhs.mins, even_subblock, 0);
        const even_mins1 = q4Kx8Scales(&rhs.mins, even_subblock, 1);
        const odd_mins0 = q4Kx8Scales(&rhs.mins, odd_subblock, 0);
        const odd_mins1 = q4Kx8Scales(&rhs.mins, odd_subblock, 1);

        inline for (0..4) |row| {
            const even_bsum0 = even_subblock * 2;
            const odd_bsum0 = odd_subblock * 2;
            const even_bsum: QKV4i32 = @splat(
                @as(i32, lhs.bsums[(even_bsum0 / 4) * 16 + row * 4 + even_bsum0 % 4]) +
                    @as(i32, lhs.bsums[((even_bsum0 + 1) / 4) * 16 + row * 4 + (even_bsum0 + 1) % 4]),
            );
            const odd_bsum: QKV4i32 = @splat(
                @as(i32, lhs.bsums[(odd_bsum0 / 4) * 16 + row * 4 + odd_bsum0 % 4]) +
                    @as(i32, lhs.bsums[((odd_bsum0 + 1) / 4) * 16 + row * 4 + (odd_bsum0 + 1) % 4]),
            );
            const sum0 = lowHalfI32x8(sum_even[row]) * even_scales0 + lowHalfI32x8(sum_odd[row]) * odd_scales0;
            const sum1 = highHalfI32x8(sum_even[row]) * even_scales1 + highHalfI32x8(sum_odd[row]) * odd_scales1;
            acc[row][0] += @as(QKV4f32, @floatFromInt(sum0)) * row_d0[row];
            acc[row][1] += @as(QKV4f32, @floatFromInt(sum1)) * row_d1[row];
            bias0[row] += even_bsum * even_mins0 + odd_bsum * odd_mins0;
            bias1[row] += even_bsum * even_mins1 + odd_bsum * odd_mins1;
        }
    }

    inline for (0..4) |row| {
        acc[row][0] -= @as(QKV4f32, @floatFromInt(bias0[row])) * row_dmin0[row];
        acc[row][1] -= @as(QKV4f32, @floatFromInt(bias1[row])) * row_dmin1[row];
    }
}

// pub: the bit-exactness reference for the SIMD arms above AND for the aarch64
// arm (same integer dots, same f32 expression order) — q4_k_tests.zig.
pub fn accumulateQ4_Kx8Q8_Kx4Scalar(lhs: *const BlockQ8_Kx4, rhs: *const BlockQ4_Kx8, acc: *[4][2]QKV4f32) void {
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
    var bias0: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    var bias1: [4]QKV4i32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };

    inline for (0..4) |pair| {
        const even_subblock = pair * 2;
        const odd_subblock = even_subblock + 1;
        var dot_even0 = [_][4]i32{.{ 0, 0, 0, 0 }} ** 4;
        var dot_even1 = [_][4]i32{.{ 0, 0, 0, 0 }} ** 4;
        var dot_odd0 = [_][4]i32{.{ 0, 0, 0, 0 }} ** 4;
        var dot_odd1 = [_][4]i32{.{ 0, 0, 0, 0 }} ** 4;

        for (0..8) |feature_group| {
            inline for (0..4) |row| {
                inline for (0..4) |col| {
                    inline for (0..4) |lane| {
                        const q0 = rhs.qs[pair * 256 + feature_group * 32 + col * 4 + lane];
                        const q1 = rhs.qs[pair * 256 + feature_group * 32 + 16 + col * 4 + lane];
                        const a_even: i32 = lhs.qs[pair * 256 + feature_group * 16 + row * 4 + lane];
                        const a_odd: i32 = lhs.qs[pair * 256 + 128 + feature_group * 16 + row * 4 + lane];
                        dot_even0[row][col] += a_even * @as(i32, @intCast(q0 & 0x0f));
                        dot_even1[row][col] += a_even * @as(i32, @intCast(q1 & 0x0f));
                        dot_odd0[row][col] += a_odd * @as(i32, @intCast(q0 >> 4));
                        dot_odd1[row][col] += a_odd * @as(i32, @intCast(q1 >> 4));
                    }
                }
            }
        }

        const even_scales0 = q4Kx8Scales(&rhs.scales, even_subblock, 0);
        const even_scales1 = q4Kx8Scales(&rhs.scales, even_subblock, 1);
        const odd_scales0 = q4Kx8Scales(&rhs.scales, odd_subblock, 0);
        const odd_scales1 = q4Kx8Scales(&rhs.scales, odd_subblock, 1);
        const even_mins0 = q4Kx8Scales(&rhs.mins, even_subblock, 0);
        const even_mins1 = q4Kx8Scales(&rhs.mins, even_subblock, 1);
        const odd_mins0 = q4Kx8Scales(&rhs.mins, odd_subblock, 0);
        const odd_mins1 = q4Kx8Scales(&rhs.mins, odd_subblock, 1);

        inline for (0..4) |row| {
            const even_bsum0 = even_subblock * 2;
            const odd_bsum0 = odd_subblock * 2;
            const even_bsum: QKV4i32 = @splat(
                @as(i32, lhs.bsums[(even_bsum0 / 4) * 16 + row * 4 + even_bsum0 % 4]) +
                    @as(i32, lhs.bsums[((even_bsum0 + 1) / 4) * 16 + row * 4 + (even_bsum0 + 1) % 4]),
            );
            const odd_bsum: QKV4i32 = @splat(
                @as(i32, lhs.bsums[(odd_bsum0 / 4) * 16 + row * 4 + odd_bsum0 % 4]) +
                    @as(i32, lhs.bsums[((odd_bsum0 + 1) / 4) * 16 + row * 4 + (odd_bsum0 + 1) % 4]),
            );
            const sum0 = @as(QKV4i32, dot_even0[row]) * even_scales0 + @as(QKV4i32, dot_odd0[row]) * odd_scales0;
            const sum1 = @as(QKV4i32, dot_even1[row]) * even_scales1 + @as(QKV4i32, dot_odd1[row]) * odd_scales1;
            acc[row][0] += @as(QKV4f32, @floatFromInt(sum0)) * row_d0[row];
            acc[row][1] += @as(QKV4f32, @floatFromInt(sum1)) * row_d1[row];
            bias0[row] += even_bsum * even_mins0 + odd_bsum * odd_mins0;
            bias1[row] += even_bsum * even_mins1 + odd_bsum * odd_mins1;
        }
    }

    inline for (0..4) |row| {
        acc[row][0] -= @as(QKV4f32, @floatFromInt(bias0[row])) * row_dmin0[row];
        acc[row][1] -= @as(QKV4f32, @floatFromInt(bias1[row])) * row_dmin1[row];
    }
}

pub fn dequantizeBlockQ4_KInto(dst: *[qk_k_block_size]f32, src: *const BlockQ4_K) void {
    const d = f16BitsToF32(src.dm[0]);
    const dmin = f16BitsToF32(src.dm[1]);
    var subblock: usize = 0;
    while (subblock < 8) : (subblock += 1) {
        const scale_min = getScaleMinK4(&src.scales, subblock);
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            const q: f32 = @floatFromInt(q4KValue(src, subblock, i));
            dst[subblock * 32 + i] = d * @as(f32, @floatFromInt(scale_min.scale)) * q -
                dmin * @as(f32, @floatFromInt(scale_min.min));
        }
    }
}

fn q4KValue(w: *const BlockQ4_K, subblock: usize, offset: usize) u8 {
    const byte = w.qs[(subblock / 2) * 32 + offset];
    return if (subblock % 2 == 0) byte & 0x0f else byte >> 4;
}

/// f32 -> Q4_K encoder for one 256-element block; faithful port of ggml's
/// quantize_row_q4_K_ref (byte-exact, see quant/encode_golden_test.zig).
/// Assumes finite input (no NaN/inf) — see the encoder contract in quant.zig.
pub fn quantizeBlockQ4_KInto(dst: *BlockQ4_K, src: *const [qk_k_block_size]f32) void {
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
        scales[j] = makeQkx2Quants(15, xs, &weights, L[32 * j ..][0..32], &mins[j], &Laux, -1.0, 0.1, 20, false);
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
            l_out.* = @intCast(@max(0, @min(15, l)));
        }
    }

    var qs_offset: usize = 0;
    j = 0;
    while (j < qk_k_block_size) : (j += 64) {
        var l: usize = 0;
        while (l < 32) : (l += 1) {
            dst.qs[qs_offset + l] = L[j + l] | (L[j + l + 32] << 4);
        }
        qs_offset += 32;
    }
}

/// f32 -> Q4_K row encoder (caller supplies the output blocks).
pub fn quantizeRowQ4_KInto(dst: []BlockQ4_K, src: []const f32) !void {
    const block_count = try qkBlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;
    for (dst, 0..) |*block, block_index| {
        quantizeBlockQ4_KInto(block, src[block_index * qk_k_block_size ..][0..qk_k_block_size]);
    }
}

fn fillQ4KPattern(block: *BlockQ4_K) void {
    block.dm = .{ f32ToF16Bits(1), f32ToF16Bits(1) };
    block.scales = .{ 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 };
    for (&block.qs, 0..) |*q, i| {
        const lo: u8 = @intCast(i % 16);
        const hi: u8 = @intCast((i + 3) % 16);
        q.* = lo | (hi << 4);
    }
}

test "ggml_q4_k dot and matmul consume loaded blocks" {
    const allocator = std.testing.allocator;

    var q4: BlockQ4_K = undefined;
    fillQ4KPattern(&q4);
    var q8: BlockQ8_K = undefined;
    fillQ8KPattern(&q8);

    var dense_w: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ4_KInto(&dense_w, &q4);
    var dense_a: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ8_KInto(&dense_a, &q8);

    try std.testing.expectEqual(dotDense(&dense_w, &dense_a), dotQ4_KQ8_K(&q4, &q8));

    var rhs_blocks = [_]BlockQ4_K{ q4, q4 };
    var qrhs = try quantizedMatmulRhsQ4_KFromBlocks(allocator, qk_k_block_size, 2, &rhs_blocks);
    defer qrhs.deinit();
    var out: [2]f32 = undefined;
    matmulQ4_KRhsRange(&out, &.{q8}, &qrhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ4_KQ8_K(&q4, &q8), out[0]);
}

// Scalar reference replica of dotQ4_KQ8_K: same nibble extraction, same i32
// integer accumulation (deferred scale*acc / min*bsum reduction), same f32
// expression order — so the comparison below is BIT-EXACT on every target
// (the integer dot is exact on all paths; Zig never contracts the identical
// f32 ops). Used by the randomized parity test and mirrored in
// src/x86dot_check.zig for the cross-ISA (Rosetta/qemu) runs.
fn refDotQ4_KQ8_K(w: *const BlockQ4_K, a: *const BlockQ8_K) f32 {
    const d = f16BitsToF32(w.dm[0]) * a.d;
    const dmin = f16BitsToF32(w.dm[1]) * a.d;
    var iscale: i32 = 0;
    var imin: i32 = 0;
    var subblock: usize = 0;
    while (subblock < 8) : (subblock += 1) {
        const scale_min = getScaleMinK4(&w.scales, subblock);
        var acc: i32 = 0;
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            const byte = w.qs[(subblock / 2) * 32 + i];
            const q: i32 = if (subblock % 2 == 0) (byte & 0x0f) else (byte >> 4);
            acc += q * @as(i32, a.qs[subblock * 32 + i]);
        }
        const bsum = @as(i32, a.bsums[subblock * 2]) + @as(i32, a.bsums[subblock * 2 + 1]);
        iscale += @as(i32, scale_min.scale) * acc;
        imin += @as(i32, scale_min.min) * bsum;
    }
    return d * @as(f32, @floatFromInt(iscale)) - dmin * @as(f32, @floatFromInt(imin));
}

fn fillRandomBlockQ4_K(block: *BlockQ4_K, random: std.Random) void {
    // Any byte pattern is a valid Q4_K payload (nibbles/scales are unsigned
    // bitfields); keep d/dmin finite & small so f32 compares stay meaningful.
    block.dm = .{ f32ToF16Bits(0.25 + random.float(f32)), f32ToF16Bits(random.float(f32) * 0.5) };
    for (&block.scales) |*s| s.* = random.int(u8);
    for (&block.qs) |*q| q.* = random.int(u8);
}

fn fillRandomBlockQ8_K(block: *BlockQ8_K, random: std.Random, extreme: bool) void {
    // BlockQ8_K activations come from quantizeRowQ8_KInto, whose -127/max
    // scale construction bounds qs to [-127,127] (never -128); that is the
    // exactness domain of the AVX2 sign-trick path and is reproduced here.
    block.d = 0.25 + random.float(f32);
    for (&block.qs, 0..) |*q, i| {
        if (extreme) {
            q.* = if (i % 2 == 0) 127 else -127; // saturation-stress pattern
        } else {
            q.* = @intCast(@as(i32, random.uintLessThan(u8, 255)) - 127); // -127..127
        }
    }
    for (&block.bsums, 0..) |*sum, group| {
        var acc: i32 = 0;
        for (block.qs[group * 16 ..][0..16]) |q| acc += q;
        sum.* = @intCast(acc);
    }
}

test "ggml_q4_k randomized blocks: dot kernel matches scalar reference bit-exactly" {
    // Randomized + extreme-value parity for the per-block Q4_K·Q8_K dot — the
    // kernel the non-aarch64 matmul path reduces to. On x86-64-v3 this
    // exercises the vpmaddubsw construction (run via qemu or real hardware);
    // on aarch64/baseline it pins the portable forms to the same reference.
    var prng = std.Random.DefaultPrng.init(0x517cc1b727220a95);
    const random = prng.random();

    var w: BlockQ4_K = undefined;
    var a: BlockQ8_K = undefined;
    var iter: usize = 0;
    while (iter < 50) : (iter += 1) {
        fillRandomBlockQ4_K(&w, random);
        fillRandomBlockQ8_K(&a, random, false);
        try std.testing.expectEqual(refDotQ4_KQ8_K(&w, &a), dotQ4_KQ8_K(&w, &a));
    }

    // Saturation stress: all-0xFF nibbles (15s) x alternating ±127 activations
    // — pair sums of 2*15*127 = 3810. The absolute call-site bound is
    // 2*15*128 = 3840 (an arbitrary i8 b operand can be -128, see common.zig's
    // dotU8I8x16WidenForm note), but BlockQ8_K activations never reach -128;
    // either way far below the 32767 i16 saturation limit of the AVX2 path.
    w.dm = .{ f32ToF16Bits(1.0), f32ToF16Bits(0.5) };
    for (&w.scales) |*s| s.* = 0xff;
    for (&w.qs) |*q| q.* = 0xff;
    fillRandomBlockQ8_K(&a, random, true);
    try std.testing.expectEqual(refDotQ4_KQ8_K(&w, &a), dotQ4_KQ8_K(&w, &a));
}

test {
    _ = @import("q4_k_tests.zig");
}
