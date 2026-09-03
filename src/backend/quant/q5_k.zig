//! Hot Q5_K / Q5_Kx8 quantized matmul kernels (per-ISA accumulate tiers).
//! Block/RHS types come from `types.zig`, shared accumulate/dispatch helpers
//! from `common.zig`, Q8_K LHS quantization from `q8k.zig`; the naming
//! grammar is in `quant.zig`.

const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const tensor = @import("../../tensor.zig");
const isa = @import("../isa.zig");
const q8k = @import("q8k.zig");
const types = @import("types.zig");
const common = @import("common.zig");
const ops = @import("../ops.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;

const BlockQ5_Kx8 = types.BlockQ5_Kx8;
const BlockQ8_K = dtype_mod.BlockQ8_K;
const QKV16i8 = common.QKV16i8;
const QKV16u8 = common.QKV16u8;
const QKV4f32 = common.QKV4f32;
const QKV4i32 = common.QKV4i32;
const QKV8i32 = common.QKV8i32;
const q4Kx8D = q8k.q4Kx8D;
const q4Kx8Scales = q8k.q4Kx8Scales;

pub fn packMatmulRhsQ5_Kx8(allocator: Allocator, blocks: []const dtype_mod.BlockQ5_K, n: usize, k: usize, blocks_per_row: usize) !types.QuantizedMatmulRhsQ5_Kx8 {
    return q8k.packLaneGroups(8, dtype_mod.BlockQ5_K, types.QuantizedMatmulRhsQ5_Kx8, packGroupQ5_Kx8, allocator, blocks, n, k, blocks_per_row);
}

fn packGroupQ5_Kx8(dst: *BlockQ5_Kx8, cols: [8]*const dtype_mod.BlockQ5_K) void {
    q8k.packGroupKx8(dst, cols, 8, q5KValue);
}

const matmulQ5_Kx8RhsTile = common.LaneX8Tile(BlockQ8_K, types.QuantizedMatmulRhsQ5_Kx8, .{ .rows = accumulateQ5_Kx8Rows, .one = accumulateQ5_Kx8 });

pub const matmulQ5_Kx8RhsRange = common.RangeFromTile(matmulQ5_Kx8RhsTile);

/// Requires whole row groups (`.full`: `r1 % 4 == 0`).
const matmulQ5_Kx8Q8_Kx4RhsTile = common.LaneX8Q8_Kx4Tile(types.QuantizedMatmulRhsQ5_Kx8, accumulateQ5_Kx8Q8_Kx4, .full);

pub const matmulQ5_Kx8Q8_Kx4RhsRange = common.RangeFromTile(matmulQ5_Kx8Q8_Kx4RhsTile);

pub const matmulQ5_KRhsTile = common.RowOuterTile(BlockQ8_K, types.QuantizedMatmulRhsQ5_K, common.qk_col_block, .{ .dot = dotQ5_KQ8_K });

pub const matmulQ5_KRhsRange = common.RangeFromTile(matmulQ5_KRhsTile);

/// Unpack one Q5_K sub-block (32 weights) to two i8 lanes for sdot. Same
/// extraction as `dotQ5_KSubblockI32`, but emitted once so it can be reused
/// across a batch of LHS rows.
fn unpackQ5_KSubblock(w: *const dtype_mod.BlockQ5_K, comptime subblock: usize) [2]QKV16i8 {
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

/// Column-outer Q5_K matmul for the m>1 (batched MoE prefill) case: the
/// shared K-quant skeleton over `unpackQ5_KSubblock` (5-bit values unpacked
/// once per weight block, reused across the row tile).
pub const matmulQ5_KRhsCompactColOuter = common.KQuantColOuterTile(types.QuantizedMatmulRhsQ5_K, unpackQ5_KSubblock);

/// The Q8_Kx4-activation twin: `common.KQuantColOuterQ8_Kx4Tile` over
/// `unpackQ5_KSubblock`.
pub const matmulQ5_KCompactQ8_Kx4ColOuter = common.KQuantColOuterQ8_Kx4Tile(types.QuantizedMatmulRhsQ5_K, unpackQ5_KSubblock);

fn accumulateQ5_Kx8(lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
    return common.accumulateTier(.{
        .scalar = accumulateQ5_Kx8Scalar,
        .aarch64 = accumulateQ5_Kx8Aarch64,
        .x86 = accumulateQ5_Kx8Tier,
    }, .{ lhs, rhs, acc });
}

// No fused rows body for q5_k: every tier walks the shell's generic loop.
fn accumulateQ5_Kx8Rows(
    lhs_blocks: []const BlockQ8_K,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: *const BlockQ5_Kx8,
    acc: *[common.q4_kx8_row_block][2]QKV4f32,
) void {
    common.accumulateLaneRows(common.q4_kx8_row_block, .{
        .one = accumulateQ5_Kx8,
    }, lhs_blocks, row_start, blocks_per_row, block_index, rhs, acc);
}

fn accumulateQ5_Kx8Q8_Kx4(lhs: *const types.BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
    return common.accumulateTier(.{
        .scalar = accumulateQ5_Kx8Q8_Kx4Scalar,
        .aarch64 = accumulateQ5_Kx8Q8_Kx4Sdot,
        .x86 = accumulateQ5_Kx8Q8_Kx4Tier,
    }, .{ lhs, rhs, acc });
}

// pub: exercised directly by q5_k_tests.zig (bit-exact vs the scalar reference
// on every host — integer sums are order-independent, epilogue is identical).
pub fn accumulateQ5_Kx8Q8_Kx4Sdot(lhs: *const types.BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
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
                dot0[row] = common.sdotI8x16Lane(row, dot0[row], rhs0, q8_vec);
                dot1[row] = common.sdotI8x16Lane(row, dot1[row], rhs1, q8_vec);
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
                dot0 = common.sdotI8x16Lane(feature_group, dot0, rhs_vec0, lhs_vec);
                dot1 = common.sdotI8x16Lane(feature_group, dot1, rhs_vec1, lhs_vec);
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
// common.dotGroupsI32x8) and applies the f16 scales with the scalar arm's exact
// f32 association, so results are bit-for-bit equal to the scalar reference —
// asserted by q5_k_tests.zig. The arms are written over the comptime-gated
// primitives in common.zig (vpdpbusd / vpmaddubsw / portable widening), so
// each also compiles and runs on any target via the portable twins (that is
// how the aarch64 dev machine exercises them).

// vpermd/vpbroadcastd-class: broadcast dword `g` (one 4-byte feature group) to
// all 8 dword lanes — aligns one LHS feature group against the 8 packed RHS
// columns of a 32-byte chunk.
inline fn broadcastGroupI32x8(comptime g: comptime_int, v: QKV8i32) QKV8i32 {
    return @shuffle(i32, v, undefined, [8]i32{ g, g, g, g, g, g, g, g });
}

fn accumulateQ5_Kx8Tier(comptime tier: isa.Tier, lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
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
            const w_e: common.QKV32u8 = @bitCast(rhs.qs[subblock * 256 + (pair * 2) * 32 ..][0..32].*);
            const w_o: common.QKV32u8 = @bitCast(rhs.qs[subblock * 256 + (pair * 2 + 1) * 32 ..][0..32].*);
            const b_e: common.QKV32i8 = @bitCast(broadcastGroupI32x8(pair * 2, lhs_groups));
            const b_o: common.QKV32i8 = @bitCast(broadcastGroupI32x8(pair * 2 + 1, lhs_groups));
            sum_e = common.dotGroupsI32x8(tier, sum_e, w_e, b_e);
            sum_o = common.dotGroupsI32x8(tier, sum_o, w_o, b_o);
        }
        const sum = sum_e + sum_o;

        const scales0 = q4Kx8Scales(&rhs.scales, subblock, 0);
        const scales1 = q4Kx8Scales(&rhs.scales, subblock, 1);
        const mins0 = q4Kx8Scales(&rhs.mins, subblock, 0);
        const mins1 = q4Kx8Scales(&rhs.mins, subblock, 1);
        const bsum: QKV4i32 = @splat(@as(i32, lhs.bsums[subblock * 2]) + @as(i32, lhs.bsums[subblock * 2 + 1]));
        iscale0 += common.lowHalfI32x8(sum) * scales0;
        iscale1 += common.highHalfI32x8(sum) * scales1;
        imin0 += bsum * mins0;
        imin1 += bsum * mins1;
    }

    acc[0] += @as(QKV4f32, @floatFromInt(iscale0)) * d0 - @as(QKV4f32, @floatFromInt(imin0)) * dmin0;
    acc[1] += @as(QKV4f32, @floatFromInt(iscale1)) * d1 - @as(QKV4f32, @floatFromInt(imin1)) * dmin1;
}

pub fn accumulateQ5_Kx8Vnni(lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
    accumulateQ5_Kx8Tier(.x86_vnni, lhs, rhs, acc);
}

pub fn accumulateQ5_Kx8Avx2(lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
    accumulateQ5_Kx8Tier(.x86_avx2, lhs, rhs, acc);
}

pub fn accumulateQ5_Kx8Widen(lhs: *const BlockQ8_K, rhs: *const BlockQ5_Kx8, acc: *[2]QKV4f32) void {
    accumulateQ5_Kx8Tier(.portable, lhs, rhs, acc);
}

fn accumulateQ5_Kx8Q8_Kx4Tier(comptime tier: isa.Tier, lhs: *const types.BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
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
            const w_e: common.QKV32u8 = @bitCast(rhs.qs[subblock * 256 + (pair * 2) * 32 ..][0..32].*);
            const w_o: common.QKV32u8 = @bitCast(rhs.qs[subblock * 256 + (pair * 2 + 1) * 32 ..][0..32].*);
            inline for (0..4) |row| {
                sums_e[row] = common.dotGroupsI32x8(tier, sums_e[row], w_e, @bitCast(broadcastGroupI32x8(row, lhs_groups)));
                sums_o[row] = common.dotGroupsI32x8(tier, sums_o[row], w_o, @bitCast(broadcastGroupI32x8(4 + row, lhs_groups)));
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
            iscale0[row] += common.lowHalfI32x8(sum) * scales0;
            iscale1[row] += common.highHalfI32x8(sum) * scales1;
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

pub fn accumulateQ5_Kx8Q8_Kx4Vnni(lhs: *const types.BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
    accumulateQ5_Kx8Q8_Kx4Tier(.x86_vnni, lhs, rhs, acc);
}

pub fn accumulateQ5_Kx8Q8_Kx4Avx2(lhs: *const types.BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
    accumulateQ5_Kx8Q8_Kx4Tier(.x86_avx2, lhs, rhs, acc);
}

pub fn accumulateQ5_Kx8Q8_Kx4Widen(lhs: *const types.BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
    accumulateQ5_Kx8Q8_Kx4Tier(.portable, lhs, rhs, acc);
}

// pub: the bit-exactness reference for the Q8_Kx4 SIMD arms (q5_k_tests.zig).
// Plain integer loops over the interleaved layouts; identical i32 sums and
// identical f32 epilogue association as the production arms.
pub fn accumulateQ5_Kx8Q8_Kx4Scalar(lhs: *const types.BlockQ8_Kx4, rhs: *const BlockQ5_Kx8, acc: *[4][2]QKV4f32) void {
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

fn dotQ5_KQ8_K(w: *const dtype_mod.BlockQ5_K, a: *const BlockQ8_K) f32 {
    switch (isa.tier) {
        .neon_i8mm, .neon_sdot => {
            const d = common.f16BitsToF32(w.dm[0]) * a.d;
            const dmin = common.f16BitsToF32(w.dm[1]) * a.d;
            // d/dmin are constant for this block, so accumulate scale*acc and min*bsum in
            // i32 across the 8 subblocks and apply f32 once at the end — fewer (and more
            // accurate) float ops than a per-subblock f32 multiply-add chain.
            var iscale: i32 = 0;
            var imin: i32 = 0;
            inline for (0..8) |subblock| {
                const scale_min = q8k.getScaleMinK4(&w.scales, subblock);
                const acc = dotQ5_KSubblockI32(w, a, subblock);
                const bsum = @as(i32, a.bsums[subblock * 2]) + @as(i32, a.bsums[subblock * 2 + 1]);
                iscale += @as(i32, scale_min.scale) * acc;
                imin += @as(i32, scale_min.min) * bsum;
            }
            return d * @as(f32, @floatFromInt(iscale)) - dmin * @as(f32, @floatFromInt(imin));
        },
        .x86_vnni, .x86_avx2, .portable => return dotQ5_KQ8_KSimd(isa.tier, w, a),
        .scalar => return dotQ5_KQ8_KScalar(w, a),
    }
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
pub fn dotQ5_KQ8_KSimd(comptime tier: isa.Tier, w: *const dtype_mod.BlockQ5_K, a: *const BlockQ8_K) f32 {
    @setEvalBranchQuota(10000);
    var iacc8: QKV8i32 = @splat(0);
    var imin: i32 = 0;
    inline for (0..8) |subblock| {
        const q32: common.QKV32u8 = @bitCast(w.qs[(subblock / 2) * 32 ..][0..32].*);
        const h32: common.QKV32u8 = @bitCast(w.qh[0..32].*);
        const high_mask: u8 = @as(u8, 1) << @intCast(subblock);
        const low = if (subblock % 2 == 0) q32 & @as(common.QKV32u8, @splat(0x0f)) else q32 >> @as(common.QKV32u8, @splat(4));
        const qs = low + @select(u8, (h32 & @as(common.QKV32u8, @splat(high_mask))) != @as(common.QKV32u8, @splat(0)), @as(common.QKV32u8, @splat(16)), @as(common.QKV32u8, @splat(0)));
        const act: common.QKV32i8 = @bitCast(a.qs[subblock * 32 ..][0..32].*);
        const sm = q8k.getScaleMinK4(&w.scales, subblock);
        const sum = common.dotGroupsI32x8(tier, @splat(0), qs, act);
        iacc8 += sum * @as(QKV8i32, @splat(@as(i32, sm.scale)));
        imin += @as(i32, sm.min) * (@as(i32, a.bsums[subblock * 2]) + @as(i32, a.bsums[subblock * 2 + 1]));
    }
    const iscale = @reduce(.Add, iacc8);
    const d = common.f16BitsToF32(w.dm[0]) * a.d;
    const dmin = common.f16BitsToF32(w.dm[1]) * a.d;
    return d * @as(f32, @floatFromInt(iscale)) - dmin * @as(f32, @floatFromInt(imin));
}

// pub: the plain-scalar bit-exactness reference for dotQ5_KQ8_KSimd AND the
// aarch64 row-dot arm (q5_k_tests.zig): same integer totals
// (order-independent adds), same f32 epilogue expression.
pub fn dotQ5_KQ8_KScalar(w: *const dtype_mod.BlockQ5_K, a: *const BlockQ8_K) f32 {
    var iscale: i32 = 0;
    var imin: i32 = 0;
    var subblock: usize = 0;
    while (subblock < 8) : (subblock += 1) {
        const sm = q8k.getScaleMinK4(&w.scales, subblock);
        var dot: i32 = 0;
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            dot += @as(i32, q5KValue(w, subblock, i)) * @as(i32, a.qs[subblock * 32 + i]);
        }
        const bsum = @as(i32, a.bsums[subblock * 2]) + @as(i32, a.bsums[subblock * 2 + 1]);
        iscale += @as(i32, sm.scale) * dot;
        imin += @as(i32, sm.min) * bsum;
    }
    const d = common.f16BitsToF32(w.dm[0]) * a.d;
    const dmin = common.f16BitsToF32(w.dm[1]) * a.d;
    return d * @as(f32, @floatFromInt(iscale)) - dmin * @as(f32, @floatFromInt(imin));
}

fn dotQ5_KSubblockI32(w: *const dtype_mod.BlockQ5_K, a: *const BlockQ8_K, comptime subblock: usize) i32 {
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
    switch (isa.tier) {
        .neon_i8mm, .neon_sdot => {
            const w0_i8: QKV16i8 = @bitCast(qs0);
            const w1_i8: QKV16i8 = @bitCast(qs1);
            var acc: QKV4i32 = @splat(0);
            acc = common.sdotI8x16(acc, w0_i8, a0_i8);
            acc = common.sdotI8x16(acc, w1_i8, a1_i8);
            return @reduce(.Add, acc);
        },
        .x86_vnni, .x86_avx2, .portable => {
            const w0: @Vector(16, i32) = @intCast(qs0);
            const w1: @Vector(16, i32) = @intCast(qs1);
            const a0: @Vector(16, i32) = @intCast(a0_i8);
            const a1: @Vector(16, i32) = @intCast(a1_i8);
            return @reduce(.Add, w0 * a0) + @reduce(.Add, w1 * a1);
        },
        .scalar => @compileError("dotQ5_KSubblockI32: the scalar tier takes dotQ5_KQ8_KScalar"),
    }
}

pub fn dequantizeBlockQ5_KInto(dst: *[dtype_mod.qk_k_block_size]f32, src: *const dtype_mod.BlockQ5_K) void {
    const d = common.f16BitsToF32(src.dm[0]);
    const dmin = common.f16BitsToF32(src.dm[1]);
    var subblock: usize = 0;
    while (subblock < 8) : (subblock += 1) {
        const scale_min = q8k.getScaleMinK4(&src.scales, subblock);
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
/// Assumes finite input (no NaN/inf); see the encoder contract in q8k.zig.
pub fn quantizeBlockQ5_KInto(dst: *dtype_mod.BlockQ5_K, src: *const [dtype_mod.qk_k_block_size]f32) void {
    var L: [dtype_mod.qk_k_block_size]u8 = undefined;
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
        scales[j] = q8k.makeQkx2Quants(31, xs, &weights, L[32 * j ..][0..32], &mins[j], &Laux, -0.5, 0.1, 15, false);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    const inv_scale: f32 = if (max_scale > 0) 63.0 / max_scale else 0.0;
    const inv_min: f32 = if (max_min > 0) 63.0 / max_min else 0.0;
    j = 0;
    while (j < 8) : (j += 1) {
        // ggml truncates the rounded int to uint8_t before the 63 clamp.
        var ls: u8 = @truncate(@as(u32, @bitCast(q8k.nearestInt(inv_scale * scales[j]))));
        var lm: u8 = @truncate(@as(u32, @bitCast(q8k.nearestInt(inv_min * mins[j]))));
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
    dst.dm = .{ common.f32ToF16Bits(max_scale / 63.0), common.f32ToF16Bits(max_min / 63.0) };

    j = 0;
    while (j < 8) : (j += 1) {
        const sm = q8k.getScaleMinK4(&dst.scales, j);
        const d = common.f16BitsToF32(dst.dm[0]) * @as(f32, @floatFromInt(sm.scale));
        if (d == 0) continue; // keeps the makeQkx2Quants levels, like ggml
        const dm = common.f16BitsToF32(dst.dm[1]) * @as(f32, @floatFromInt(sm.min));
        for (src[32 * j ..][0..32], L[32 * j ..][0..32]) |v, *l_out| {
            const l = q8k.nearestInt((v + dm) / d);
            l_out.* = @intCast(@max(0, @min(31, l)));
        }
    }

    @memset(&dst.qh, 0);
    var qs_offset: usize = 0;
    var n: usize = 0;
    while (n < dtype_mod.qk_k_block_size) : (n += 64) {
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
pub fn quantizeRowQ5_KInto(dst: []dtype_mod.BlockQ5_K, src: []const f32) !void {
    const block_count = try q8k.qkBlockCount(src.len);
    if (dst.len != block_count) return types.QuantizedFormatError.InvalidQuantizedLength;
    for (dst, 0..) |*block, block_index| {
        quantizeBlockQ5_KInto(block, src[block_index * dtype_mod.qk_k_block_size ..][0..dtype_mod.qk_k_block_size]);
    }
}

fn q5KValue(w: *const dtype_mod.BlockQ5_K, subblock: usize, offset: usize) u8 {
    const byte = w.qs[(subblock / 2) * 32 + offset];
    const low = if (subblock % 2 == 0) byte & 0x0f else byte >> 4;
    const high_mask: u8 = @as(u8, 1) << @intCast(subblock);
    return low + if ((w.qh[offset] & high_mask) != 0) @as(u8, 16) else @as(u8, 0);
}

fn setQ5KValue(block: *dtype_mod.BlockQ5_K, subblock: usize, offset: usize, value: u8) void {
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

fn fillQ5KPattern(block: *dtype_mod.BlockQ5_K) void {
    block.dm = .{ common.f32ToF16Bits(1), common.f32ToF16Bits(0) };
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

    var q5: dtype_mod.BlockQ5_K = undefined;
    fillQ5KPattern(&q5);
    var q8: BlockQ8_K = undefined;
    q8k.fillQ8KPattern(&q8);

    var dense_w: [dtype_mod.qk_k_block_size]f32 = undefined;
    dequantizeBlockQ5_KInto(&dense_w, &q5);
    var dense_a: [dtype_mod.qk_k_block_size]f32 = undefined;
    q8k.dequantizeBlockQ8_KInto(&dense_a, &q8);

    try std.testing.expectEqual(common.dotDense(&dense_w, &dense_a), dotQ5_KQ8_K(&q5, &q8));

    var rhs_blocks = [_]dtype_mod.BlockQ5_K{ q5, q5 };
    var qrhs = try q8k.quantizedMatmulRhsQ5_KFromBlocks(allocator, dtype_mod.qk_k_block_size, 2, &rhs_blocks);
    defer qrhs.deinit();
    var out: [2]f32 = undefined;
    matmulQ5_KRhsRange(&out, &.{q8}, &qrhs, 1, 2, 0, 1);
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(dotQ5_KQ8_K(&q5, &q8), out[0]);
}

test {
    _ = @import("q5_k_tests.zig");
}

/// The one Q5_K GEMM entry (`ops.QuantGemm`): comptime-selects the tile
/// body for `.{ g.rhs, g.lhs, g.order }`. Output row stride is `rhs.n`.
/// The Q5_K kernels (see `q4_k.kernels`).
pub const kernels = .{
    .{ .g = ops.QuantGemm{ .weight = .q5_k, .rhs = .rows, .lhs = .q8_k, .order = .row_outer }, .tile = matmulQ5_KRhsTile },
    .{ .g = ops.QuantGemm{ .weight = .q5_k, .rhs = .rows, .lhs = .q8_k, .order = .col_outer }, .tile = matmulQ5_KRhsCompactColOuter },
    .{ .g = ops.QuantGemm{ .weight = .q5_k, .rhs = .rows, .lhs = .q8_kx4, .order = .col_outer }, .tile = q8Kx4ColOuterTile },
    .{ .g = ops.QuantGemm{ .weight = .q5_k, .rhs = .x8, .lhs = .q8_k, .order = .row_outer }, .tile = matmulQ5_Kx8RhsTile },
    .{ .g = ops.QuantGemm{ .weight = .q5_k, .rhs = .x8, .lhs = .q8_kx4, .order = .row_outer }, .tile = matmulQ5_Kx8Q8_Kx4RhsTile },
};

/// The col-outer Q8_Kx4 body covers the whole row range (`r0 == 0`).
fn q8Kx4ColOuterTile(out: []f32, lhs: anytype, rhs: anytype, n: usize, r0: usize, r1: usize, c0: usize, c1: usize) void {
    std.debug.assert(r0 == 0);
    matmulQ5_KCompactQ8_Kx4ColOuter(out, lhs, rhs, n, r1, c0, c1);
}
