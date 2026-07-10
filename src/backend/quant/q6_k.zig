//! Hot Q6_K / Q6_Kx4 quantized matmul kernels relocated out of quant.zig.
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
const BlockQ6_K = types_mod.BlockQ6_K;
const BlockQ6_Kx4 = types_mod.BlockQ6_Kx4;
const BlockQ8_K = types_mod.BlockQ8_K;
const BlockQ8_Kx4 = types_mod.BlockQ8_Kx4;
const QKV16i16 = common.QKV16i16;
const QKV16i32 = common.QKV16i32;
const QKV16i8 = common.QKV16i8;
const QKV16u8 = common.QKV16u8;
const QKV32i8 = common.QKV32i8;
const QKV32u8 = common.QKV32u8;
const QKV4f32 = common.QKV4f32;
const QKV4i32 = common.QKV4i32;
const QKV8i32 = common.QKV8i32;
const QuantizedFormatError = types_mod.QuantizedFormatError;
const QuantizedMatmulRhsQ6_K = types_mod.QuantizedMatmulRhsQ6_K;
const QuantizedMatmulRhsQ6_Kx4 = types_mod.QuantizedMatmulRhsQ6_Kx4;
const addHalvesI32x8 = common.addHalvesI32x8;
const checkedProduct = types_mod.checkedProduct;
const dequantizeBlockQ8_KInto = q8k_mod.dequantizeBlockQ8_KInto;
const dotDense = common.dotDense;
const dotI8GroupsWidenI32x8 = common.dotI8GroupsWidenI32x8;
const dpbusdI32x8 = common.dpbusdI32x8;
const f16BitsToF32 = common.f16BitsToF32;
const f16x4BitsToF32 = common.f16x4BitsToF32;
const f32ToF16Bits = common.f32ToF16Bits;
const fillQ8KPattern = q8k_mod.fillQ8KPattern;
const group_max_eps = q8k_mod.group_max_eps;
const has_x86_avx2 = common.has_x86_avx2;
const has_x86_vnni_ymm = common.has_x86_vnni_ymm;
const maddubsDotGroupsI32x8 = common.maddubsDotGroupsI32x8;
const makeQxQuants = q8k_mod.makeQxQuants;
const nearestInt = q8k_mod.nearestInt;
const q8_0_row_block = common.q8_0_row_block;
const qkBlockCount = q8k_mod.qkBlockCount;
const qk_col_block = common.qk_col_block;
const qk_k_block_size = types_mod.qk_k_block_size;
const quantizedMatmulRhsQ6_KFromBlocks = q8k_mod.quantizedMatmulRhsQ6_KFromBlocks;
const sdotI8x16Lane = common.sdotI8x16Lane;

pub fn packMatmulRhsQ6_Kx4(
    allocator: Allocator,
    blocks: []const BlockQ6_K,
    n: usize,
    k: usize,
    blocks_per_row: usize,
) !QuantizedMatmulRhsQ6_Kx4 {
    if (n % 4 != 0) return tensor.TensorError.InvalidShape;
    if (blocks_per_row != try qkBlockCount(k)) return tensor.TensorError.InvalidShape;
    if (blocks.len != try checkedProduct(n, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;

    const group_count = n / 4;
    const packed_blocks = try allocator.alloc(BlockQ6_Kx4, try checkedProduct(group_count, blocks_per_row));
    errdefer allocator.free(packed_blocks);

    for (0..group_count) |group_i| {
        for (0..blocks_per_row) |block_i| {
            const b0 = &blocks[(4 * group_i + 0) * blocks_per_row + block_i];
            const b1 = &blocks[(4 * group_i + 1) * blocks_per_row + block_i];
            const b2 = &blocks[(4 * group_i + 2) * blocks_per_row + block_i];
            const b3 = &blocks[(4 * group_i + 3) * blocks_per_row + block_i];
            const cols = [_]*const BlockQ6_K{ b0, b1, b2, b3 };
            var dst = &packed_blocks[group_i * blocks_per_row + block_i];
            dst.d = .{ b0.d, b1.d, b2.d, b3.d };

            for (0..16) |scale_group| {
                dst.scales[scale_group * 4 + 0] = b0.scales[scale_group];
                dst.scales[scale_group * 4 + 1] = b1.scales[scale_group];
                dst.scales[scale_group * 4 + 2] = b2.scales[scale_group];
                dst.scales[scale_group * 4 + 3] = b3.scales[scale_group];

                for (0..4) |feature_group| {
                    for (0..4) |col| {
                        const block = cols[col];
                        for (0..4) |lane| {
                            const feature = scale_group * 16 + feature_group * 4 + lane;
                            dst.qs[scale_group * 64 + feature_group * 16 + col * 4 + lane] = q6KValue(block, feature);
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

pub fn matmulQ6_KRhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ6_K,
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
                    acc[c] += dotQ6_KQ8_K(rhs_block, lhs_block);
                }
            }
            inline for (0..qk_col_block) |c| out[i * n + j + c] = acc[c];
        }

        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc += dotQ6_KQ8_K(&rhs_col[block_index], &lhs_row[block_index]);
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn matmulQ6_KRhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ6_K,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ6_KRhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

const moe_row_tile = 4;

/// Unpack one Q6_K group (16 weights) to a centered i8 lane for sdot. Same
/// extraction as `dotQ6_KGroupI32`, emitted once for reuse across a row batch.
fn unpackQ6_KGroup(w: *const BlockQ6_K, comptime chunk: usize, comptime section: usize, comptime half: usize) QKV16i8 {
    const ql_offset = chunk * 64 + (if (section == 1 or section == 3) 32 else 0) + half * 16;
    const qh_offset = chunk * 32 + half * 16;
    const ql_vec: QKV16u8 = @bitCast(w.ql[ql_offset..][0..16].*);
    const qh_vec: QKV16u8 = @bitCast(w.qh[qh_offset..][0..16].*);
    const low = if (section < 2) ql_vec & @as(QKV16u8, @splat(0x0f)) else ql_vec >> @as(QKV16u8, @splat(4));
    const high = (qh_vec >> @as(QKV16u8, @splat(section * 2))) & @as(QKV16u8, @splat(0x03));
    const combined = low | (high << @as(QKV16u8, @splat(4)));
    return @as(QKV16i8, @intCast(combined)) - @as(QKV16i8, @splat(32));
}

fn dotUnpackedI8x16(w: QKV16i8, a: QKV16i8) i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        return @reduce(.Add, common.sdotI8x16(@as(QKV4i32, @splat(0)), w, a));
    }
    // non-aarch64: VNNI lowers to vpdpbusd (via the +128 bias), AVX2 takes the
    // sign-trick vpmaddubsw path (w ∈ [-32,31] here; a comes from BlockQ8_K,
    // i.e. quantizeRowQ8_KInto's -127/max scale construction, so a ∈ [-127,127]
    // — inside the sign-trick exactness domain; see common.zig).
    return common.dotI8x16Portable(w, a);
}

/// Column-outer Q6_K matmul for m>1 (batched MoE prefill): unpack each weight
/// group's 6-bit values ONCE, then sdot against a tile of LHS rows — amortizing
/// the (costlier) 6-bit unpack over the batch. Numerically identical to the
/// row-outer `matmulQ6_KRhsTile`.
pub fn matmulQ6_KRhsCompactColOuter(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ6_K,
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
                var iacc = [_]i32{0} ** moe_row_tile;
                inline for (0..2) |chunk| {
                    inline for (0..4) |section| {
                        inline for (0..2) |half| {
                            const group = chunk * 8 + section * 2 + half;
                            const wv = unpackQ6_KGroup(w, chunk, section, half);
                            const scale: i32 = w.scales[group];
                            const a_offset = chunk * 128 + section * 32 + half * 16;
                            var r: usize = 0;
                            while (r < tn) : (r += 1) {
                                const a = &lhs_blocks[(row0 + r) * bpc + bi];
                                const av: QKV16i8 = @bitCast(a.qs[a_offset..][0..16].*);
                                iacc[r] += dotUnpackedI8x16(wv, av) * scale;
                            }
                        }
                    }
                }
                const d = f16BitsToF32(w.d);
                var r: usize = 0;
                while (r < tn) : (r += 1) {
                    acc_f32[r] += @as(f32, @floatFromInt(iacc[r])) * (d * lhs_blocks[(row0 + r) * bpc + bi].d);
                }
            }
            var r: usize = 0;
            while (r < tn) : (r += 1) out[(row0 + r) * n + j] = acc_f32[r];
        }
    }
}

/// Four rows' 16-wide i8 dot of one Q6_K group against pre-unpacked weights `wv`
/// (16 centered i8), returning the four row dots in the four lanes of one i32x4
/// (lane = row). `fg_base` is the group's first Q8_Kx4 feature-group index. On
/// aarch64 each `sdot …4b[g]` reuses the unpacked weight across all four rows, so
/// the group is 4 `sdot`s and **zero** horizontal reductions. Integer dots are
/// order-independent, so this equals the per-row `dotUnpackedI8x16` path exactly.
inline fn dot16Group4RowsQ8_Kx4(a: *const BlockQ8_Kx4, fg_base: usize, wv: QKV16i8) QKV4i32 {
    @setEvalBranchQuota(10000); // raises the caller's running quota for the unrolled arms
    if (comptime builtin.cpu.arch == .aarch64) {
        var dot: QKV4i32 = @splat(0);
        inline for (0..4) |g| {
            const ag: QKV16i8 = @bitCast(a.qs[(fg_base + g) * 16 ..][0..16].*);
            dot = sdotI8x16Lane(g, dot, ag, wv);
        }
        return dot;
    }
    if (comptime has_x86_vnni_ymm) return dot16Group4RowsQ8_Kx4Simd(.vnni, a, fg_base, wv);
    if (comptime has_x86_avx2) return dot16Group4RowsQ8_Kx4Simd(.avx2, a, fg_base, wv);
    return dot16Group4RowsQ8_Kx4Simd(.widen, a, fg_base, wv);
}

/// Prepared pre-unpacked ColOuter weight-group type per tier: +32-biased u8
/// for the u8·i8 tiers, untouched signed i8 for the widening tier (the
/// 16-byte analog of `Q6Kx4WeightChunk`).
fn Q6Weights16(comptime tier: Q6Kx4SimdTier) type {
    return if (tier == .widen) QKV16i8 else QKV16u8;
}

/// +32-bias one pre-unpacked 16-weight group for the u8·i8 tiers (exact:
/// w ∈ [-32,31] → u8 ∈ [0,63]); pass-through for the widening tier.
inline fn prepQ6Weights16(comptime tier: Q6Kx4SimdTier, w: QKV16i8) Q6Weights16(tier) {
    if (comptime tier == .widen) return w;
    return @bitCast(w +% @as(QKV16i8, @splat(32)));
}

/// x86/portable ymm arms of `dot16Group4RowsQ8_Kx4` (the MoE-prefill 4-row
/// lane dot): each 32-byte Q8_Kx4 activation load
/// already holds [fg: 4 rows × 4 features | fg+1: …] dword-per-row, so
/// broadcasting the matching weight dword pair (`broadcastPairGroupsI32x4`)
/// turns the 16-feature × 4-row dot into two grouped-dot ops + one half-fold
/// — no per-row rebuild. OPERAND SHAPE: `wv` is SIGNED [-32,31] (centered
/// Q6_K values); the u8·i8 tiers dot the +32-biased form (u8 ∈ [0,63]) and
/// subtract 32·(per-row 16-feature activation sums) taken from the Q8_Kx4
/// bsums contract (`bsums[(sg/4)*16 + row*4 + sg%4]`, sg = fg_base/4 — the
/// interleave `quantizeRowsQ8_Kx4*Into` writes and the Q4_K/Q5_K mins
/// kernels already rely on), so every tier returns the true signed dot.
/// SATURATION (avx2 tier): biased w ≤ 63 → vpmaddubsw pair sums ≤ 2·63·128 =
/// 16128 < 2^15 for all i8 activations. NO OVERFLOW: |sum8 lane| ≤ 2·4·63·128
/// < 2^17, |32·bsum| ≤ 32·16·128 = 2^16. Integer sums are order-independent →
/// bit-identical to `dot16Group4RowsQ8_Kx4Scalar` (q6_k_tests.zig). pub for
/// the sibling exact-parity tests.
pub fn dot16Group4RowsQ8_Kx4Simd(comptime tier: Q6Kx4SimdTier, a: *const BlockQ8_Kx4, fg_base: usize, wv: QKV16i8) QKV4i32 {
    const w4: QKV4i32 = @bitCast(prepQ6Weights16(tier, wv));
    var sum8: QKV8i32 = @splat(0);
    inline for (0..2) |c| {
        const act: QKV32i8 = @bitCast(a.qs[(fg_base + 2 * c) * 16 ..][0..32].*);
        sum8 = dotQ6Kx4Groups(tier, sum8, @bitCast(broadcastPairGroupsI32x4(c, w4)), act);
    }
    var dot = addHalvesI32x8(sum8);
    if (comptime tier != .widen) {
        const sg = fg_base / 4;
        var bs: QKV4i32 = undefined;
        inline for (0..4) |row| bs[row] = a.bsums[(sg / 4) * 16 + row * 4 + sg % 4];
        dot -= bs * @as(QKV4i32, @splat(32));
    }
    return dot;
}

// pub: the bit-exactness reference for dot16Group4RowsQ8_Kx4Simd
// (q6_k_tests.zig) — the plain per-row rebuild over the interleaved Q8_Kx4
// layout (row r's 4 features for feature-group fg live at
// qs[fg*16 + r*4 ..][0..4]).
pub fn dot16Group4RowsQ8_Kx4Scalar(a: *const BlockQ8_Kx4, fg_base: usize, wv: QKV16i8) QKV4i32 {
    var dot: QKV4i32 = @splat(0);
    inline for (0..4) |row| {
        var acc: i32 = 0;
        inline for (0..4) |g| {
            inline for (0..4) |t| {
                acc += @as(i32, wv[g * 4 + t]) * @as(i32, a.qs[(fg_base + g) * 16 + row * 4 + t]);
            }
        }
        dot[row] = acc;
    }
    return dot;
}

/// Column-outer Q6_K matmul over **4-row-interleaved Q8_Kx4** activations. Like
/// `matmulQ6_KRhsCompactColOuter` it unpacks each 6-bit weight group once and reuses
/// it across the row tile, but it packs the four rows into the `sdot` lanes
/// (`dot16Group4RowsQ8_Kx4`) so the four rows share one i32x4 accumulator with no
/// per-row horizontal reduction, and the f32 epilogue runs vector-wide. `lhs_blocks`
/// holds `ceil(m/4)` Q8_Kx4 groups per K-block (tail rows zero-padded, e.g. via
/// `packRowsQ8_Kx4PaddedInto`); `m` is the real row count so padded lanes are never
/// stored. Bit-identical to the per-row column-outer / row-outer tile.
pub fn matmulQ6_KCompactQ8_Kx4ColOuter(
    out: []f32,
    lhs_blocks: []const BlockQ8_Kx4,
    rhs: *const QuantizedMatmulRhsQ6_K,
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
                var iacc: QKV4i32 = @splat(0);
                inline for (0..2) |chunk| {
                    inline for (0..4) |section| {
                        inline for (0..2) |half| {
                            const group = chunk * 8 + section * 2 + half;
                            const wv = unpackQ6_KGroup(w, chunk, section, half);
                            const scale: i32 = w.scales[group];
                            const fg_base = chunk * 32 + section * 8 + half * 4;
                            const dot = dot16Group4RowsQ8_Kx4(a, fg_base, wv);
                            iacc += @as(QKV4i32, @splat(scale)) * dot;
                        }
                    }
                }
                const d = f16BitsToF32(w.d);
                const ad: QKV4f32 = a.d;
                acc_f32 += @as(QKV4f32, @floatFromInt(iacc)) * (@as(QKV4f32, @splat(d)) * ad);
            }
            const acc_arr: [4]f32 = acc_f32;
            var r: usize = 0;
            while (r < tn) : (r += 1) out[(row0 + r) * n + j] = acc_arr[r];
        }
    }
}

pub fn matmulQ6_Kx4RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ6_Kx4,
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
                accumulateQ6_Kx4Rows(lhs_blocks, i, blocks_per_row, block_index, rhs_block, &acc);
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
                acc = accumulateQ6_Kx4(&lhs_row[block_index], &rhs_group[block_index], acc);
            }
            out[i * n + j + 0] = acc[0];
            out[i * n + j + 1] = acc[1];
            out[i * n + j + 2] = acc[2];
            out[i * n + j + 3] = acc[3];
        }
    }
}

pub fn matmulQ6_Kx4RhsPairTile(
    gate_out: []f32,
    up_out: []f32,
    lhs_blocks: []const BlockQ8_K,
    gate_rhs: *const QuantizedMatmulRhsQ6_Kx4,
    up_rhs: *const QuantizedMatmulRhsQ6_Kx4,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    std.debug.assert(c0 % 4 == 0);
    std.debug.assert(c1 % 4 == 0);
    std.debug.assert(gate_rhs.blocks_per_group == up_rhs.blocks_per_group);
    std.debug.assert(gate_rhs.n == up_rhs.n);
    std.debug.assert(gate_rhs.k == up_rhs.k);

    const blocks_per_row = gate_rhs.blocks_per_group;
    var i = r0;
    while (i + q8_0_row_block <= r1) : (i += q8_0_row_block) {
        var j = c0;
        while (j < c1) : (j += 4) {
            const gate_group = gate_rhs.groupBlocks(j / 4);
            const up_group = up_rhs.groupBlocks(j / 4);
            var gate_acc: [q8_0_row_block]QKV4f32 = undefined;
            var up_acc: [q8_0_row_block]QKV4f32 = undefined;
            inline for (0..q8_0_row_block) |r| {
                gate_acc[r] = @splat(0);
                up_acc[r] = @splat(0);
            }

            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                accumulateQ6_Kx4RowsPair(
                    lhs_blocks,
                    i,
                    blocks_per_row,
                    block_index,
                    &gate_group[block_index],
                    &up_group[block_index],
                    &gate_acc,
                    &up_acc,
                );
            }

            inline for (0..q8_0_row_block) |r| {
                gate_out[(i + r) * n + j + 0] = gate_acc[r][0];
                gate_out[(i + r) * n + j + 1] = gate_acc[r][1];
                gate_out[(i + r) * n + j + 2] = gate_acc[r][2];
                gate_out[(i + r) * n + j + 3] = gate_acc[r][3];
                up_out[(i + r) * n + j + 0] = up_acc[r][0];
                up_out[(i + r) * n + j + 1] = up_acc[r][1];
                up_out[(i + r) * n + j + 2] = up_acc[r][2];
                up_out[(i + r) * n + j + 3] = up_acc[r][3];
            }
        }
    }

    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;
        while (j < c1) : (j += 4) {
            const gate_group = gate_rhs.groupBlocks(j / 4);
            const up_group = up_rhs.groupBlocks(j / 4);
            var gate_acc: QKV4f32 = @splat(0);
            var up_acc: QKV4f32 = @splat(0);
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const pair = accumulateQ6_Kx4Pair(&lhs_row[block_index], &gate_group[block_index], &up_group[block_index], gate_acc, up_acc);
                gate_acc = pair.gate;
                up_acc = pair.up;
            }
            gate_out[i * n + j + 0] = gate_acc[0];
            gate_out[i * n + j + 1] = gate_acc[1];
            gate_out[i * n + j + 2] = gate_acc[2];
            gate_out[i * n + j + 3] = gate_acc[3];
            up_out[i * n + j + 0] = up_acc[0];
            up_out[i * n + j + 1] = up_acc[1];
            up_out[i * n + j + 2] = up_acc[2];
            up_out[i * n + j + 3] = up_acc[3];
        }
    }
}

pub fn matmulQ6_Kx4RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsQ6_Kx4,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ6_Kx4RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

fn dotQ6_KQ8_K(w: *const BlockQ6_K, a: *const BlockQ8_K) f32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        const d = f16BitsToF32(w.d) * a.d;
        // Accumulate acc*scale in i32 across the 16 groups, apply d once at the end.
        var iacc: i32 = 0;
        inline for (0..2) |chunk| {
            inline for (0..4) |section| {
                inline for (0..2) |half| {
                    const group = chunk * 8 + section * 2 + half;
                    const acc = dotQ6_KGroupI32(w, a, chunk, section, half);
                    iacc += acc * @as(i32, w.scales[group]);
                }
            }
        }
        return @as(f32, @floatFromInt(iacc)) * d;
    }
    if (comptime has_x86_vnni_ymm) return dotQ6_KQ8_KSimd(.vnni, w, a);
    if (comptime has_x86_avx2) return dotQ6_KQ8_KSimd(.avx2, w, a);
    return dotQ6_KQ8_KSimd(.widen, w, a);
}

/// One grouped dot-accumulate of a 32-byte +32-biased Q6_K weight run
/// (u8 ∈ [0,63]) against 32 activations, tier-selected — same primitives (and
/// per-tier exactness arguments) as `dotQ6Kx4Groups`; the widening tier
/// re-centers to signed [-32,31] in-register (exact, no wrap) so it needs no
/// bias correction.
inline fn dotQ6BiasedGroups(comptime tier: Q6Kx4SimdTier, acc: QKV8i32, biased: QKV32u8, act: QKV32i8) QKV8i32 {
    return switch (tier) {
        .vnni => dpbusdI32x8(acc, biased, act),
        .avx2 => maddubsDotGroupsI32x8(acc, biased, act),
        .widen => dotI8GroupsWidenI32x8(acc, @as(QKV32i8, @bitCast(biased)) -% @as(QKV32i8, @splat(32)), act),
    };
}

/// x86/portable ymm arm of the Q6_K row dot (the decode/GEMV path of
/// `matmulQ6_KRhsTile`): one (chunk, section) step covers 32 contiguous
/// features — 32 ql bytes, 32 qh bytes, 32 activation bytes — so the whole
/// 256-feature block is 8 grouped-dot ops. OPERAND SHAPE: the raw 6-bit
/// value `low | high<<4` ∈ [0,63] IS the +32-biased weight, so the u8·i8
/// tiers dot it directly and remove the bias once per block through the
/// BlockQ8_K bsums contract (Σ_g scale_g·bsums[g], ×32); the widening tier
/// re-centers in-register and needs no correction. The two per-section
/// scales ride the 8-lane accumulator ([s0×4 | s1×4]); one horizontal reduce
/// per block. SATURATION (avx2 tier): w ≤ 63 → vpmaddubsw pair sums ≤
/// 2·63·128 = 16128 < 2^15 for all i8 activations. NO OVERFLOW: |iacc8 lane|
/// ≤ 8·4·63·128·128 < 2^26, reduce ≤ 2^29; |32·Σ scale·bsum| ≤ 32·16·128·2048
/// = 2^27. Identical i32 total (order-independent integer adds) and identical
/// f32 epilogue as the scalar reference → bit-exact (q6_k_tests.zig). pub for
/// the sibling exact-parity tests.
pub fn dotQ6_KQ8_KSimd(comptime tier: Q6Kx4SimdTier, w: *const BlockQ6_K, a: *const BlockQ8_K) f32 {
    @setEvalBranchQuota(10000);
    var iacc8: QKV8i32 = @splat(0);
    var ibias: i32 = 0;
    inline for (0..2) |chunk| {
        const qh32: QKV32u8 = @bitCast(w.qh[chunk * 32 ..][0..32].*);
        inline for (0..4) |section| {
            const ql32: QKV32u8 = @bitCast(w.ql[chunk * 64 + (if (section == 1 or section == 3) 32 else 0) ..][0..32].*);
            const low = if (section < 2) ql32 & @as(QKV32u8, @splat(0x0f)) else ql32 >> @as(QKV32u8, @splat(4));
            const high = (qh32 >> @as(QKV32u8, @splat(section * 2))) & @as(QKV32u8, @splat(0x03));
            const biased = low | (high << @as(QKV32u8, @splat(4))); // w+32 ∈ [0,63]
            const act: QKV32i8 = @bitCast(a.qs[chunk * 128 + section * 32 ..][0..32].*);
            const sum = dotQ6BiasedGroups(tier, @splat(0), biased, act);
            const g0 = chunk * 8 + section * 2;
            const s0: i32 = w.scales[g0];
            const s1: i32 = w.scales[g0 + 1];
            iacc8 += sum * @as(QKV8i32, .{ s0, s0, s0, s0, s1, s1, s1, s1 });
            if (comptime tier != .widen) {
                ibias += s0 * @as(i32, a.bsums[g0]) + s1 * @as(i32, a.bsums[g0 + 1]);
            }
        }
    }
    var iacc = @reduce(.Add, iacc8);
    if (comptime tier != .widen) iacc -= 32 * ibias;
    const d = f16BitsToF32(w.d) * a.d;
    return @as(f32, @floatFromInt(iacc)) * d;
}

// pub: the plain-scalar bit-exactness reference for dotQ6_KQ8_KSimd AND the
// aarch64 row-dot arm (q6_k_tests.zig): same integer total (order-independent
// adds), same f32 epilogue expression.
pub fn dotQ6_KQ8_KScalar(w: *const BlockQ6_K, a: *const BlockQ8_K) f32 {
    var iacc: i32 = 0;
    var group: usize = 0;
    while (group < 16) : (group += 1) {
        var dot: i32 = 0;
        var i: usize = 0;
        while (i < 16) : (i += 1) {
            const feature = group * 16 + i;
            dot += @as(i32, q6KValue(w, feature)) * @as(i32, a.qs[feature]);
        }
        iacc += dot * @as(i32, w.scales[group]);
    }
    const d = f16BitsToF32(w.d) * a.d;
    return @as(f32, @floatFromInt(iacc)) * d;
}

fn accumulateQ6_Kx4(lhs: *const BlockQ8_K, rhs: *const BlockQ6_Kx4, acc: QKV4f32) QKV4f32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ6_Kx4Aarch64(lhs, rhs, acc);
    }
    if (comptime has_x86_vnni_ymm) {
        return accumulateQ6_Kx4Simd(.vnni, lhs, rhs, acc);
    }
    if (comptime has_x86_avx2) {
        return accumulateQ6_Kx4Simd(.avx2, lhs, rhs, acc);
    }
    return accumulateQ6_Kx4Simd(.widen, lhs, rhs, acc);
}

const Q6Kx4PairAcc = struct {
    gate: QKV4f32,
    up: QKV4f32,
};

fn accumulateQ6_Kx4Pair(
    lhs: *const BlockQ8_K,
    gate_rhs: *const BlockQ6_Kx4,
    up_rhs: *const BlockQ6_Kx4,
    gate_acc: QKV4f32,
    up_acc: QKV4f32,
) Q6Kx4PairAcc {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ6_Kx4PairAarch64(lhs, gate_rhs, up_rhs, gate_acc, up_acc);
    }
    if (comptime has_x86_vnni_ymm) {
        return accumulateQ6_Kx4PairSimd(.vnni, lhs, gate_rhs, up_rhs, gate_acc, up_acc);
    }
    if (comptime has_x86_avx2) {
        return accumulateQ6_Kx4PairSimd(.avx2, lhs, gate_rhs, up_rhs, gate_acc, up_acc);
    }
    return accumulateQ6_Kx4PairSimd(.widen, lhs, gate_rhs, up_rhs, gate_acc, up_acc);
}

fn accumulateQ6_Kx4Rows(
    lhs_blocks: []const BlockQ8_K,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: *const BlockQ6_Kx4,
    acc: *[q8_0_row_block]QKV4f32,
) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ6_Kx4RowsAarch64(lhs_blocks, row_start, blocks_per_row, block_index, rhs, acc);
    }
    if (comptime has_x86_vnni_ymm) {
        return accumulateQ6_Kx4RowsSimd(.vnni, lhs_blocks, row_start, blocks_per_row, block_index, rhs, acc);
    }
    if (comptime has_x86_avx2) {
        return accumulateQ6_Kx4RowsSimd(.avx2, lhs_blocks, row_start, blocks_per_row, block_index, rhs, acc);
    }
    return accumulateQ6_Kx4RowsSimd(.widen, lhs_blocks, row_start, blocks_per_row, block_index, rhs, acc);
}

fn accumulateQ6_Kx4RowsPair(
    lhs_blocks: []const BlockQ8_K,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    gate_rhs: *const BlockQ6_Kx4,
    up_rhs: *const BlockQ6_Kx4,
    gate_acc: *[q8_0_row_block]QKV4f32,
    up_acc: *[q8_0_row_block]QKV4f32,
) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ6_Kx4RowsPairAarch64(lhs_blocks, row_start, blocks_per_row, block_index, gate_rhs, up_rhs, gate_acc, up_acc);
    }
    inline for (0..q8_0_row_block) |r| {
        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
        const pair = accumulateQ6_Kx4Pair(lhs, gate_rhs, up_rhs, gate_acc[r], up_acc[r]);
        gate_acc[r] = pair.gate;
        up_acc[r] = pair.up;
    }
}

fn accumulateQ6_Kx4Aarch64(lhs: *const BlockQ8_K, rhs: *const BlockQ6_Kx4, acc: QKV4f32) QKV4f32 {
    const d = f16x4BitsToF32(rhs.d) * @as(QKV4f32, @splat(lhs.d));
    // Accumulate scale-weighted dots in i32 across the 16 groups (per-group
    // products peak ~8.3M, the block sum ~132M, both well under i32), then do a
    // single float convert+scale per block instead of one per group — same
    // i32-accumulate strategy as dotQ6_KQ8_K. The 64 sdots are unchanged.
    var iacc: QKV4i32 = @splat(0);

    inline for (0..16) |scale_group| {
        const lhs_vec: QKV16i8 = @bitCast(lhs.qs[scale_group * 16 ..][0..16].*);
        var dot: QKV4i32 = @splat(0);
        inline for (0..4) |feature_group| {
            const rhs_vec: QKV16i8 = @bitCast(rhs.qs[scale_group * 64 + feature_group * 16 ..][0..16].*);
            dot = sdotI8x16Lane(feature_group, dot, rhs_vec, lhs_vec);
        }

        const scales: QKV4i32 = .{
            rhs.scales[scale_group * 4 + 0],
            rhs.scales[scale_group * 4 + 1],
            rhs.scales[scale_group * 4 + 2],
            rhs.scales[scale_group * 4 + 3],
        };
        iacc += dot * scales;
    }

    return acc + @as(QKV4f32, @floatFromInt(iacc)) * d;
}

fn accumulateQ6_Kx4PairAarch64(
    lhs: *const BlockQ8_K,
    gate_rhs: *const BlockQ6_Kx4,
    up_rhs: *const BlockQ6_Kx4,
    gate_acc: QKV4f32,
    up_acc: QKV4f32,
) Q6Kx4PairAcc {
    const gate_d = f16x4BitsToF32(gate_rhs.d) * @as(QKV4f32, @splat(lhs.d));
    const up_d = f16x4BitsToF32(up_rhs.d) * @as(QKV4f32, @splat(lhs.d));
    var gate_iacc: QKV4i32 = @splat(0);
    var up_iacc: QKV4i32 = @splat(0);

    inline for (0..16) |scale_group| {
        const lhs_vec: QKV16i8 = @bitCast(lhs.qs[scale_group * 16 ..][0..16].*);

        const gate_rhs0: QKV16i8 = @bitCast(gate_rhs.qs[scale_group * 64 + 0 * 16 ..][0..16].*);
        const gate_rhs1: QKV16i8 = @bitCast(gate_rhs.qs[scale_group * 64 + 1 * 16 ..][0..16].*);
        const gate_rhs2: QKV16i8 = @bitCast(gate_rhs.qs[scale_group * 64 + 2 * 16 ..][0..16].*);
        const gate_rhs3: QKV16i8 = @bitCast(gate_rhs.qs[scale_group * 64 + 3 * 16 ..][0..16].*);
        const gate_scales: QKV4i32 = .{
            gate_rhs.scales[scale_group * 4 + 0],
            gate_rhs.scales[scale_group * 4 + 1],
            gate_rhs.scales[scale_group * 4 + 2],
            gate_rhs.scales[scale_group * 4 + 3],
        };
        var gate_dot: QKV4i32 = @splat(0);
        gate_dot = sdotI8x16Lane(0, gate_dot, gate_rhs0, lhs_vec);
        gate_dot = sdotI8x16Lane(1, gate_dot, gate_rhs1, lhs_vec);
        gate_dot = sdotI8x16Lane(2, gate_dot, gate_rhs2, lhs_vec);
        gate_dot = sdotI8x16Lane(3, gate_dot, gate_rhs3, lhs_vec);
        gate_iacc += gate_dot * gate_scales;

        const up_rhs0: QKV16i8 = @bitCast(up_rhs.qs[scale_group * 64 + 0 * 16 ..][0..16].*);
        const up_rhs1: QKV16i8 = @bitCast(up_rhs.qs[scale_group * 64 + 1 * 16 ..][0..16].*);
        const up_rhs2: QKV16i8 = @bitCast(up_rhs.qs[scale_group * 64 + 2 * 16 ..][0..16].*);
        const up_rhs3: QKV16i8 = @bitCast(up_rhs.qs[scale_group * 64 + 3 * 16 ..][0..16].*);
        const up_scales: QKV4i32 = .{
            up_rhs.scales[scale_group * 4 + 0],
            up_rhs.scales[scale_group * 4 + 1],
            up_rhs.scales[scale_group * 4 + 2],
            up_rhs.scales[scale_group * 4 + 3],
        };
        var up_dot: QKV4i32 = @splat(0);
        up_dot = sdotI8x16Lane(0, up_dot, up_rhs0, lhs_vec);
        up_dot = sdotI8x16Lane(1, up_dot, up_rhs1, lhs_vec);
        up_dot = sdotI8x16Lane(2, up_dot, up_rhs2, lhs_vec);
        up_dot = sdotI8x16Lane(3, up_dot, up_rhs3, lhs_vec);
        up_iacc += up_dot * up_scales;
    }

    return .{
        .gate = gate_acc + @as(QKV4f32, @floatFromInt(gate_iacc)) * gate_d,
        .up = up_acc + @as(QKV4f32, @floatFromInt(up_iacc)) * up_d,
    };
}

fn accumulateQ6_Kx4RowsAarch64(
    lhs_blocks: []const BlockQ8_K,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: *const BlockQ6_Kx4,
    acc: *[q8_0_row_block]QKV4f32,
) void {
    const rhs_d = f16x4BitsToF32(rhs.d);
    var iacc = [_]QKV4i32{@splat(0)} ** q8_0_row_block;

    inline for (0..16) |scale_group| {
        const rhs0: QKV16i8 = @bitCast(rhs.qs[scale_group * 64 + 0 * 16 ..][0..16].*);
        const rhs1: QKV16i8 = @bitCast(rhs.qs[scale_group * 64 + 1 * 16 ..][0..16].*);
        const rhs2: QKV16i8 = @bitCast(rhs.qs[scale_group * 64 + 2 * 16 ..][0..16].*);
        const rhs3: QKV16i8 = @bitCast(rhs.qs[scale_group * 64 + 3 * 16 ..][0..16].*);
        const scales: QKV4i32 = .{
            rhs.scales[scale_group * 4 + 0],
            rhs.scales[scale_group * 4 + 1],
            rhs.scales[scale_group * 4 + 2],
            rhs.scales[scale_group * 4 + 3],
        };

        inline for (0..q8_0_row_block) |r| {
            const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
            const lhs_vec: QKV16i8 = @bitCast(lhs.qs[scale_group * 16 ..][0..16].*);
            var dot: QKV4i32 = @splat(0);
            dot = sdotI8x16Lane(0, dot, rhs0, lhs_vec);
            dot = sdotI8x16Lane(1, dot, rhs1, lhs_vec);
            dot = sdotI8x16Lane(2, dot, rhs2, lhs_vec);
            dot = sdotI8x16Lane(3, dot, rhs3, lhs_vec);
            iacc[r] += dot * scales;
        }
    }

    inline for (0..q8_0_row_block) |r| {
        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
        const d = rhs_d * @as(QKV4f32, @splat(lhs.d));
        acc[r] += @as(QKV4f32, @floatFromInt(iacc[r])) * d;
    }
}

fn accumulateQ6_Kx4RowsPairAarch64(
    lhs_blocks: []const BlockQ8_K,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    gate_rhs: *const BlockQ6_Kx4,
    up_rhs: *const BlockQ6_Kx4,
    gate_acc: *[q8_0_row_block]QKV4f32,
    up_acc: *[q8_0_row_block]QKV4f32,
) void {
    const gate_rhs_d = f16x4BitsToF32(gate_rhs.d);
    const up_rhs_d = f16x4BitsToF32(up_rhs.d);
    var gate_iacc = [_]QKV4i32{@splat(0)} ** q8_0_row_block;
    var up_iacc = [_]QKV4i32{@splat(0)} ** q8_0_row_block;

    inline for (0..16) |scale_group| {
        const gate_rhs0: QKV16i8 = @bitCast(gate_rhs.qs[scale_group * 64 + 0 * 16 ..][0..16].*);
        const gate_rhs1: QKV16i8 = @bitCast(gate_rhs.qs[scale_group * 64 + 1 * 16 ..][0..16].*);
        const gate_rhs2: QKV16i8 = @bitCast(gate_rhs.qs[scale_group * 64 + 2 * 16 ..][0..16].*);
        const gate_rhs3: QKV16i8 = @bitCast(gate_rhs.qs[scale_group * 64 + 3 * 16 ..][0..16].*);
        const gate_scales: QKV4i32 = .{
            gate_rhs.scales[scale_group * 4 + 0],
            gate_rhs.scales[scale_group * 4 + 1],
            gate_rhs.scales[scale_group * 4 + 2],
            gate_rhs.scales[scale_group * 4 + 3],
        };

        const up_rhs0: QKV16i8 = @bitCast(up_rhs.qs[scale_group * 64 + 0 * 16 ..][0..16].*);
        const up_rhs1: QKV16i8 = @bitCast(up_rhs.qs[scale_group * 64 + 1 * 16 ..][0..16].*);
        const up_rhs2: QKV16i8 = @bitCast(up_rhs.qs[scale_group * 64 + 2 * 16 ..][0..16].*);
        const up_rhs3: QKV16i8 = @bitCast(up_rhs.qs[scale_group * 64 + 3 * 16 ..][0..16].*);
        const up_scales: QKV4i32 = .{
            up_rhs.scales[scale_group * 4 + 0],
            up_rhs.scales[scale_group * 4 + 1],
            up_rhs.scales[scale_group * 4 + 2],
            up_rhs.scales[scale_group * 4 + 3],
        };

        inline for (0..q8_0_row_block) |r| {
            const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
            const lhs_vec: QKV16i8 = @bitCast(lhs.qs[scale_group * 16 ..][0..16].*);

            var gate_dot: QKV4i32 = @splat(0);
            gate_dot = sdotI8x16Lane(0, gate_dot, gate_rhs0, lhs_vec);
            gate_dot = sdotI8x16Lane(1, gate_dot, gate_rhs1, lhs_vec);
            gate_dot = sdotI8x16Lane(2, gate_dot, gate_rhs2, lhs_vec);
            gate_dot = sdotI8x16Lane(3, gate_dot, gate_rhs3, lhs_vec);
            gate_iacc[r] += gate_dot * gate_scales;

            var up_dot: QKV4i32 = @splat(0);
            up_dot = sdotI8x16Lane(0, up_dot, up_rhs0, lhs_vec);
            up_dot = sdotI8x16Lane(1, up_dot, up_rhs1, lhs_vec);
            up_dot = sdotI8x16Lane(2, up_dot, up_rhs2, lhs_vec);
            up_dot = sdotI8x16Lane(3, up_dot, up_rhs3, lhs_vec);
            up_iacc[r] += up_dot * up_scales;
        }
    }

    inline for (0..q8_0_row_block) |r| {
        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
        const lhs_d: QKV4f32 = @splat(lhs.d);
        gate_acc[r] += @as(QKV4f32, @floatFromInt(gate_iacc[r])) * (gate_rhs_d * lhs_d);
        up_acc[r] += @as(QKV4f32, @floatFromInt(up_iacc[r])) * (up_rhs_d * lhs_d);
    }
}

// --- x86 / portable-SIMD arms of the x4 accumulates --------------------------
//
// Every arm below computes the SAME i32 per-scale-group dots as
// accumulateQ6_Kx4Scalar (bit-identical integer accumulation — bounds proven
// at accumulateQ6_Kx4Simd) and applies the f16 scales with the scalar arm's
// exact f32 association (acc + float(iacc) · (rhs_d · splat(lhs_d))), so
// results are bit-for-bit equal to the scalar reference — asserted by
// q6_k_tests.zig. The arms are written over the comptime-gated primitives in
// common.zig (vpdpbusd / maddubs / portable widening), so each also compiles
// and runs on any target via the portable twins (that is how the aarch64 dev
// machine exercises them); Debug x86 builds (self-hosted backend, no ymm asm)
// run the twins too via the has_llvm_asm gate inside the primitives.
//
// OPERAND SHAPE: unlike the unsigned nibble expansions of Q4_K, the packed
// Q6_Kx4 weights are SIGNED, centered to [-32,31] by packMatmulRhsQ6_Kx4
// (q6KValue subtracts 32). The u8·i8 tiers therefore re-bias each 32-byte
// weight chunk by +32 (u8 ∈ [0,63]) and remove the bias per scale group with
// the Q8_K block sums:
//     Σ w·a = Σ (w+32)·a − 32·bsums[sg]
// where bsums[sg] = Σ qs[sg·16..][0..16] is part of the BlockQ8_K format
// contract (quantizeRowQ8_KInto always writes it; the Q4_K mins kernels
// already rely on it). Activations are unrestricted i8 in every tier — no
// sign-trick domain restriction anywhere on this path.

/// SIMD tier of the non-aarch64 Q6_Kx4 arms: which grouped dot primitive the
/// arm is built on. `vnni`/`avx2` are the +32-weight-bias u8·i8 forms
/// (vpdpbusd / vpmaddubsw+vpmaddwd); `widen` is the universal exact i8·i8
/// floor (no bias, no correction, no arch gate).
pub const Q6Kx4SimdTier = enum { vnni, avx2, widen };

/// Prepared weight-chunk type per tier: +32-biased u8 for the u8·i8 tiers,
/// untouched signed i8 for the widening tier.
fn Q6Kx4WeightChunk(comptime tier: Q6Kx4SimdTier) type {
    return if (tier == .widen) QKV32i8 else QKV32u8;
}

/// +32-bias one 32-byte packed weight chunk for the u8·i8 tiers. Exact u8 for
/// all w ≥ −32 (w+32 ≤ 159 < 256; wrapping add ≡ mod-256); the packed format
/// guarantees w ∈ [-32,31] → u8 ∈ [0,63]. Pass-through for the widening tier.
inline fn prepQ6Kx4WeightChunk(comptime tier: Q6Kx4SimdTier, w: QKV32i8) Q6Kx4WeightChunk(tier) {
    if (comptime tier == .widen) return w;
    return @bitCast(w +% @as(QKV32i8, @splat(32)));
}

/// Grouped dot of one prepared weight chunk against one broadcast activation
/// chunk. SATURATION (avx2 tier only): vpmaddubsw pair sums are bounded by
/// 2·63·128 = 16128 < 2^15 in the packed weight domain (biased u8 ≤ 63), so
/// the maddubs form never saturates; the vnni and widen forms are exact for
/// all inputs by construction (see common.zig).
inline fn dotQ6Kx4Groups(comptime tier: Q6Kx4SimdTier, acc: QKV8i32, w: Q6Kx4WeightChunk(tier), a: QKV32i8) QKV8i32 {
    return switch (tier) {
        .vnni => dpbusdI32x8(acc, w, a),
        .avx2 => maddubsDotGroupsI32x8(acc, w, a),
        .widen => dotI8GroupsWidenI32x8(acc, w, a),
    };
}

// vpermd-class (cross-lane): broadcast activation feature-group dword 2c to
// the low 128-bit lane and dword 2c+1 to the high lane — aligns one 16-byte
// Q8_K scale-group load against the two feature-group halves of the 32-byte
// Q6_Kx4 chunk c (chunk layout: [fg 2c: col0..col3 | fg 2c+1: col0..col3]).
inline fn broadcastPairGroupsI32x4(comptime c: comptime_int, v: QKV4i32) QKV8i32 {
    return @shuffle(i32, v, undefined, [8]i32{ 2 * c, 2 * c, 2 * c, 2 * c, 2 * c + 1, 2 * c + 1, 2 * c + 1, 2 * c + 1 });
}

inline fn q6Kx4Scales(rhs: *const BlockQ6_Kx4, comptime scale_group: usize) QKV4i32 {
    return .{
        rhs.scales[scale_group * 4 + 0],
        rhs.scales[scale_group * 4 + 1],
        rhs.scales[scale_group * 4 + 2],
        rhs.scales[scale_group * 4 + 3],
    };
}

/// x86/portable SIMD arm of accumulateQ6_Kx4. Per scale group the two weight
/// chunks are dotted against the cross-lane broadcast activations and the 8
/// partial column sums are scale-multiplied in-register — the half-fold is
/// deferred to the block epilogue (integer sums are order-independent, so the
/// folded i32 equals the scalar arm's exactly). NO OVERFLOW: per lane one
/// scale group contributes ≤ 8·63·128 = 64512 (biased partials), ×|scale| ≤
/// 128 → ≤ 8.26M, ×16 groups ≤ 133M < 2^31; the bias accumulator is bounded
/// by 16·2048·128 ≈ 4.2M (|bsums| ≤ 16·128 by the format contract), ×32 ≤
/// 135M.
pub fn accumulateQ6_Kx4Simd(comptime tier: Q6Kx4SimdTier, lhs: *const BlockQ8_K, rhs: *const BlockQ6_Kx4, acc: QKV4f32) QKV4f32 {
    @setEvalBranchQuota(10000); // fully unrolls 16 scale groups x 2 chunks
    var iacc8: QKV8i32 = @splat(0);
    var bias4: QKV4i32 = @splat(0);

    inline for (0..16) |scale_group| {
        const lhs4: QKV4i32 = @bitCast(lhs.qs[scale_group * 16 ..][0..16].*);
        const scales = q6Kx4Scales(rhs, scale_group);
        var sum: QKV8i32 = @splat(0);
        inline for (0..2) |chunk| {
            const w = prepQ6Kx4WeightChunk(tier, @bitCast(rhs.qs[scale_group * 64 + chunk * 32 ..][0..32].*));
            sum = dotQ6Kx4Groups(tier, sum, w, @bitCast(broadcastPairGroupsI32x4(chunk, lhs4)));
        }
        iacc8 += sum * @shuffle(i32, scales, undefined, [8]i32{ 0, 1, 2, 3, 0, 1, 2, 3 });
        if (comptime tier != .widen) bias4 += @as(QKV4i32, @splat(lhs.bsums[scale_group])) * scales;
    }

    var iacc = addHalvesI32x8(iacc8);
    if (comptime tier != .widen) iacc -= bias4 * @as(QKV4i32, @splat(32));
    const d = f16x4BitsToF32(rhs.d) * @as(QKV4f32, @splat(lhs.d));
    return acc + @as(QKV4f32, @floatFromInt(iacc)) * d;
}

/// Rows-tile SIMD arm (q8_0_row_block LHS rows against one packed column
/// group): the biased weight chunks and scale vectors are prepared once per
/// scale group and reused across the row tile — otherwise row-for-row the
/// same integer/f32 algebra (and bounds) as accumulateQ6_Kx4Simd, so results
/// are bit-identical to per-row calls.
pub fn accumulateQ6_Kx4RowsSimd(
    comptime tier: Q6Kx4SimdTier,
    lhs_blocks: []const BlockQ8_K,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: *const BlockQ6_Kx4,
    acc: *[q8_0_row_block]QKV4f32,
) void {
    @setEvalBranchQuota(10000); // fully unrolls 16 scale groups x 4 rows
    const rhs_d = f16x4BitsToF32(rhs.d);
    var iacc8 = [_]QKV8i32{@splat(0)} ** q8_0_row_block;
    var bias4 = [_]QKV4i32{@splat(0)} ** q8_0_row_block;

    inline for (0..16) |scale_group| {
        const w0 = prepQ6Kx4WeightChunk(tier, @bitCast(rhs.qs[scale_group * 64 ..][0..32].*));
        const w1 = prepQ6Kx4WeightChunk(tier, @bitCast(rhs.qs[scale_group * 64 + 32 ..][0..32].*));
        const scales = q6Kx4Scales(rhs, scale_group);
        const scales8 = @shuffle(i32, scales, undefined, [8]i32{ 0, 1, 2, 3, 0, 1, 2, 3 });

        inline for (0..q8_0_row_block) |r| {
            const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
            const lhs4: QKV4i32 = @bitCast(lhs.qs[scale_group * 16 ..][0..16].*);
            var sum: QKV8i32 = @splat(0);
            sum = dotQ6Kx4Groups(tier, sum, w0, @bitCast(broadcastPairGroupsI32x4(0, lhs4)));
            sum = dotQ6Kx4Groups(tier, sum, w1, @bitCast(broadcastPairGroupsI32x4(1, lhs4)));
            iacc8[r] += sum * scales8;
            if (comptime tier != .widen) bias4[r] += @as(QKV4i32, @splat(lhs.bsums[scale_group])) * scales;
        }
    }

    inline for (0..q8_0_row_block) |r| {
        var iacc = addHalvesI32x8(iacc8[r]);
        if (comptime tier != .widen) iacc -= bias4[r] * @as(QKV4i32, @splat(32));
        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
        const d = rhs_d * @as(QKV4f32, @splat(lhs.d));
        acc[r] += @as(QKV4f32, @floatFromInt(iacc)) * d;
    }
}

/// Paired gate/up SIMD arm: one activation load + broadcast shared by both
/// weight matrices; per matrix the same integer/f32 algebra (and bounds) as
/// accumulateQ6_Kx4Simd, so results are bit-identical to two independent
/// calls.
pub fn accumulateQ6_Kx4PairSimd(
    comptime tier: Q6Kx4SimdTier,
    lhs: *const BlockQ8_K,
    gate_rhs: *const BlockQ6_Kx4,
    up_rhs: *const BlockQ6_Kx4,
    gate_acc: QKV4f32,
    up_acc: QKV4f32,
) Q6Kx4PairAcc {
    @setEvalBranchQuota(10000); // fully unrolls 16 scale groups x 2 matrices
    var gate_iacc8: QKV8i32 = @splat(0);
    var up_iacc8: QKV8i32 = @splat(0);
    var gate_bias4: QKV4i32 = @splat(0);
    var up_bias4: QKV4i32 = @splat(0);

    inline for (0..16) |scale_group| {
        const lhs4: QKV4i32 = @bitCast(lhs.qs[scale_group * 16 ..][0..16].*);
        const a0: QKV32i8 = @bitCast(broadcastPairGroupsI32x4(0, lhs4));
        const a1: QKV32i8 = @bitCast(broadcastPairGroupsI32x4(1, lhs4));

        const gate_scales = q6Kx4Scales(gate_rhs, scale_group);
        var gate_sum: QKV8i32 = @splat(0);
        gate_sum = dotQ6Kx4Groups(tier, gate_sum, prepQ6Kx4WeightChunk(tier, @bitCast(gate_rhs.qs[scale_group * 64 ..][0..32].*)), a0);
        gate_sum = dotQ6Kx4Groups(tier, gate_sum, prepQ6Kx4WeightChunk(tier, @bitCast(gate_rhs.qs[scale_group * 64 + 32 ..][0..32].*)), a1);
        gate_iacc8 += gate_sum * @shuffle(i32, gate_scales, undefined, [8]i32{ 0, 1, 2, 3, 0, 1, 2, 3 });
        if (comptime tier != .widen) gate_bias4 += @as(QKV4i32, @splat(lhs.bsums[scale_group])) * gate_scales;

        const up_scales = q6Kx4Scales(up_rhs, scale_group);
        var up_sum: QKV8i32 = @splat(0);
        up_sum = dotQ6Kx4Groups(tier, up_sum, prepQ6Kx4WeightChunk(tier, @bitCast(up_rhs.qs[scale_group * 64 ..][0..32].*)), a0);
        up_sum = dotQ6Kx4Groups(tier, up_sum, prepQ6Kx4WeightChunk(tier, @bitCast(up_rhs.qs[scale_group * 64 + 32 ..][0..32].*)), a1);
        up_iacc8 += up_sum * @shuffle(i32, up_scales, undefined, [8]i32{ 0, 1, 2, 3, 0, 1, 2, 3 });
        if (comptime tier != .widen) up_bias4 += @as(QKV4i32, @splat(lhs.bsums[scale_group])) * up_scales;
    }

    var gate_iacc = addHalvesI32x8(gate_iacc8);
    var up_iacc = addHalvesI32x8(up_iacc8);
    if (comptime tier != .widen) {
        gate_iacc -= gate_bias4 * @as(QKV4i32, @splat(32));
        up_iacc -= up_bias4 * @as(QKV4i32, @splat(32));
    }
    const gate_d = f16x4BitsToF32(gate_rhs.d) * @as(QKV4f32, @splat(lhs.d));
    const up_d = f16x4BitsToF32(up_rhs.d) * @as(QKV4f32, @splat(lhs.d));
    return .{
        .gate = gate_acc + @as(QKV4f32, @floatFromInt(gate_iacc)) * gate_d,
        .up = up_acc + @as(QKV4f32, @floatFromInt(up_iacc)) * up_d,
    };
}

// pub: the bit-exactness reference for the SIMD arms above (q6_k_tests.zig).
pub fn accumulateQ6_Kx4Scalar(lhs: *const BlockQ8_K, rhs: *const BlockQ6_Kx4, acc: QKV4f32) QKV4f32 {
    @setEvalBranchQuota(10000); // non-aarch64 fallback fully unrolls 16*16*4 inline iters
    const d = f16x4BitsToF32(rhs.d) * @as(QKV4f32, @splat(lhs.d));
    var iacc: QKV4i32 = @splat(0);

    inline for (0..16) |scale_group| {
        var dot: QKV4i32 = @splat(0);
        inline for (0..16) |feature_offset| {
            const lhs_value: i32 = lhs.qs[scale_group * 16 + feature_offset];
            inline for (0..4) |col| {
                dot[col] += lhs_value * @as(i32, rhs.qs[scale_group * 64 + (feature_offset / 4) * 16 + col * 4 + feature_offset % 4]);
            }
        }

        const scales: QKV4i32 = .{
            rhs.scales[scale_group * 4 + 0],
            rhs.scales[scale_group * 4 + 1],
            rhs.scales[scale_group * 4 + 2],
            rhs.scales[scale_group * 4 + 3],
        };
        iacc += dot * scales;
    }

    return acc + @as(QKV4f32, @floatFromInt(iacc)) * d;
}

fn dotQ6_KGroupI32(w: *const BlockQ6_K, a: *const BlockQ8_K, comptime chunk: usize, comptime section: usize, comptime half: usize) i32 {
    const ql_offset = chunk * 64 + (if (section == 1 or section == 3) 32 else 0) + half * 16;
    const qh_offset = chunk * 32 + half * 16;
    const a_offset = chunk * 128 + section * 32 + half * 16;

    const ql_vec: QKV16u8 = @bitCast(w.ql[ql_offset..][0..16].*);
    const qh_vec: QKV16u8 = @bitCast(w.qh[qh_offset..][0..16].*);
    const low = if (section < 2)
        ql_vec & @as(QKV16u8, @splat(0x0f))
    else
        ql_vec >> @as(QKV16u8, @splat(4));
    const high = (qh_vec >> @as(QKV16u8, @splat(section * 2))) & @as(QKV16u8, @splat(0x03));
    const combined = low | (high << @as(QKV16u8, @splat(4)));

    const a_i8: QKV16i8 = @bitCast(a.qs[a_offset..][0..16].*);
    // q6 values center to [-32,31] (fit i8); dot via NEON sdot where available.
    if (comptime builtin.cpu.arch == .aarch64) {
        const w_i8: QKV16i8 = @as(QKV16i8, @intCast(combined)) - @as(QKV16i8, @splat(32));
        return @reduce(.Add, common.sdotI8x16(@splat(0), w_i8, a_i8));
    }
    const w_i16: QKV16i16 = @as(QKV16i16, @intCast(combined)) - @as(QKV16i16, @splat(32));
    const a_i16: QKV16i16 = @intCast(a_i8);
    const product_i32: QKV16i32 = @intCast(w_i16 * a_i16);
    return @reduce(.Add, product_i32);
}

pub fn dequantizeBlockQ6_KInto(dst: *[qk_k_block_size]f32, src: *const BlockQ6_K) void {
    const d = f16BitsToF32(src.d);
    var index: usize = 0;
    while (index < qk_k_block_size) : (index += 1) {
        const scale_index = q6KScaleIndex(index);
        dst[index] = d * @as(f32, @floatFromInt(src.scales[scale_index])) *
            @as(f32, @floatFromInt(q6KValue(src, index)));
    }
}

/// f32 -> Q6_K encoder for one 256-element block; faithful port of ggml's
/// quantize_row_q6_K_ref (byte-exact, see quant/encode_golden_test.zig).
/// Assumes finite input (no NaN/inf) — see the encoder contract in quant.zig.
pub fn quantizeBlockQ6_KInto(dst: *BlockQ6_K, src: *const [qk_k_block_size]f32) void {
    var L: [qk_k_block_size]i8 = undefined;
    var scales: [16]f32 = undefined;

    var max_scale: f32 = 0;
    var max_abs_scale: f32 = 0;
    var ib: usize = 0;
    while (ib < 16) : (ib += 1) {
        const scale = makeQxQuants(32, src[16 * ib ..][0..16], L[16 * ib ..][0..16], 1, null);
        scales[ib] = scale;
        const abs_scale = @abs(scale);
        if (abs_scale > max_abs_scale) {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if (max_abs_scale < group_max_eps) {
        dst.* = std.mem.zeroes(BlockQ6_K);
        return;
    }

    const iscale: f32 = -128.0 / max_scale;
    dst.d = f32ToF16Bits(1 / iscale);
    ib = 0;
    while (ib < 16) : (ib += 1) {
        dst.scales[ib] = @intCast(@min(127, nearestInt(iscale * scales[ib])));
    }

    var j: usize = 0;
    while (j < 16) : (j += 1) {
        const d = f16BitsToF32(dst.d) * @as(f32, @floatFromInt(dst.scales[j]));
        if (d == 0) continue; // keeps the makeQxQuants levels, like ggml
        for (src[16 * j ..][0..16], L[16 * j ..][0..16]) |v, *l_out| {
            const l = nearestInt(v / d);
            l_out.* = @intCast(@max(-32, @min(31, l)) + 32);
        }
    }

    var ql_offset: usize = 0;
    var qh_offset: usize = 0;
    j = 0;
    while (j < qk_k_block_size) : (j += 128) {
        var l: usize = 0;
        while (l < 32) : (l += 1) {
            const q1: u8 = @intCast(L[j + l]);
            const q2: u8 = @intCast(L[j + l + 32]);
            const q3: u8 = @intCast(L[j + l + 64]);
            const q4: u8 = @intCast(L[j + l + 96]);
            dst.ql[ql_offset + l] = (q1 & 0x0f) | ((q3 & 0x0f) << 4);
            dst.ql[ql_offset + l + 32] = (q2 & 0x0f) | ((q4 & 0x0f) << 4);
            dst.qh[qh_offset + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
        ql_offset += 64;
        qh_offset += 32;
    }
}

/// f32 -> Q6_K row encoder (caller supplies the output blocks).
pub fn quantizeRowQ6_KInto(dst: []BlockQ6_K, src: []const f32) !void {
    const block_count = try qkBlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;
    for (dst, 0..) |*block, block_index| {
        quantizeBlockQ6_KInto(block, src[block_index * qk_k_block_size ..][0..qk_k_block_size]);
    }
}

fn q6KValue(w: *const BlockQ6_K, index: usize) i8 {
    const chunk = index / 128;
    const local = index % 128;
    const section = local / 32;
    const l = local % 32;
    const ql_base = chunk * 64;
    const qh_base = chunk * 32;
    const ql_index = ql_base + if (section == 1 or section == 3) 32 + l else l;
    const low = if (section < 2) w.ql[ql_index] & 0x0f else w.ql[ql_index] >> 4;
    const high = (w.qh[qh_base + l] >> @intCast(section * 2)) & 0x03;
    const combined: i16 = @as(i16, low) | (@as(i16, high) << 4);
    return @intCast(combined - 32);
}

fn q6KScaleIndex(index: usize) usize {
    const chunk = index / 128;
    const local = index % 128;
    const section = local / 32;
    const l = local % 32;
    return chunk * 8 + section * 2 + l / 16;
}

fn setQ6KValue(block: *BlockQ6_K, index: usize, value: i8) void {
    const encoded: u8 = @intCast(@as(i16, value) + 32);
    const chunk = index / 128;
    const local = index % 128;
    const section = local / 32;
    const l = local % 32;
    const ql_base = chunk * 64;
    const qh_base = chunk * 32;
    const ql_index = ql_base + if (section == 1 or section == 3) 32 + l else l;
    if (section < 2) {
        block.ql[ql_index] = (block.ql[ql_index] & 0xf0) | (encoded & 0x0f);
    } else {
        block.ql[ql_index] = (block.ql[ql_index] & 0x0f) | ((encoded & 0x0f) << 4);
    }
    const shift: u3 = @intCast(section * 2);
    block.qh[qh_base + l] = (block.qh[qh_base + l] & ~(@as(u8, 0x03) << shift)) | (((encoded >> 4) & 0x03) << shift);
}

fn fillQ6KPattern(block: *BlockQ6_K) void {
    @memset(&block.ql, 0);
    @memset(&block.qh, 0);
    block.d = f32ToF16Bits(1);
    for (&block.scales, 0..) |*scale, i| scale.* = @intCast(@as(i32, @intCast(i % 5)) - 2);
    block.scales[0] = 1;
    for (0..qk_k_block_size) |i| setQ6KValue(block, i, @intCast(@as(i32, @intCast(i % 23)) - 11));
}

test "ggml_q6_k dot and matmul consume loaded blocks" {
    const allocator = std.testing.allocator;

    var q6: BlockQ6_K = undefined;
    fillQ6KPattern(&q6);
    var q8: BlockQ8_K = undefined;
    fillQ8KPattern(&q8);

    var dense_w: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ6_KInto(&dense_w, &q6);
    var dense_a: [qk_k_block_size]f32 = undefined;
    dequantizeBlockQ8_KInto(&dense_a, &q8);

    try std.testing.expectEqual(dotDense(&dense_w, &dense_a), dotQ6_KQ8_K(&q6, &q8));

    var rhs_blocks = [_]BlockQ6_K{ q6, q6 };
    var qrhs = try quantizedMatmulRhsQ6_KFromBlocks(allocator, qk_k_block_size, 2, &rhs_blocks);
    defer qrhs.deinit();
    var out: [2]f32 = undefined;
    matmulQ6_KRhsRange(&out, &.{q8}, &qrhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ6_KQ8_K(&q6, &q8), out[0]);
}

test {
    _ = @import("q6_k_tests.zig");
}
