//! Hot MXFP4 matmul kernel: native fp4-e2m1 weights against Q8_0
//! activations.
//!
//! The nibble table stores DOUBLED e2m1 values ({0,1,2,3,4,6,8,12} and
//! negatives — exact int8), so each 32-element block dots as a pure integer
//! `tbl`+`sdot` pass; the block's e8m0 scale is applied halved
//! (`e8m0ToF32Half`), folding the /2 back without a rounding step. One f32
//! multiply-add per block per column is the only float work, and both
//! integer-dot arms (NEON asm / portable) are exact, so the kernel's result
//! is bitwise identical across architectures by construction.
//!
//! Layouts match `dequantizeRowMXFP4Into` (cold.zig): `qs[j]` low nibble is
//! element `j`, high nibble element `j + 16`, so the low/high code vectors
//! pair with the Q8_0 activation halves `qs[0..16]` / `qs[16..32]`.

const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const isa = @import("../isa.zig");
const types = @import("types.zig");
const common = @import("common.zig");
const tables = @import("../quant_tables.zig");

const kvalues: common.QKV16i8 = tables.kvalues_mxfp4;

// x86 ymm path (the `x86_vnni`/`x86_avx2` tiers): the 16-entry signed table
// duplicated across both vpshufb lanes, plus its magnitudes for vpdpbusd's
// unsigned side (sign transfers to the activation via vpsignb; |code| <= 12
// keeps the AVX2 maddubs pair sums at <= 3048, far inside i16 — exact).
// DOMAIN: activation codes must be in [-127, 127] — vpsignb's negation wraps
// -128 — which the Q8_0 quantizer guarantees (it never emits -128); same
// proven-domain contract as the q8_0 sign-trick kernel.
const kvalues_x2: common.QKV32i8 = blk: {
    var v: [32]i8 = undefined;
    for (0..16) |i| {
        v[i] = tables.kvalues_mxfp4[i];
        v[16 + i] = tables.kvalues_mxfp4[i];
    }
    break :blk v;
};
const kmag_x2: common.QKV32i8 = blk: {
    var v: [32]i8 = undefined;
    for (0..32) |i| v[i] = if (kvalues_x2[i] < 0) -kvalues_x2[i] else kvalues_x2[i];
    break :blk v;
};

const e8m0_half_lut: [256]f32 = blk: {
    var lut: [256]f32 = undefined;
    for (0..256) |i| lut[i] = common.e8m0ToF32Half(@intCast(i));
    break :blk lut;
};

const DecodedMXFP4 = struct { lo: common.QKV16i8, hi: common.QKV16i8 };

inline fn decodeBlock(qs: *const [16]u8) DecodedMXFP4 {
    const packed_v: common.QKV16u8 = @bitCast(qs.*);
    const lo_idx = packed_v & @as(common.QKV16u8, @splat(0x0f));
    const hi_idx = packed_v >> @as(common.QKV16u8, @splat(4));
    return .{ .lo = common.tblI8x16(kvalues, lo_idx), .hi = common.tblI8x16(kvalues, hi_idx) };
}

/// Integer block dot as FOUR lane sums, arch-split but integer-exact and
/// lane-identical everywhere: NEON's two sdots give lane l = lo-group l +
/// hi-group l; the x86 ymm path reproduces the same four sums by folding
/// dpbusd's eight 4-byte-group lanes halfwise (lane l + lane l+4). The
/// float tail below is therefore BITWISE identical across architectures.
inline fn blockDotI32(w: *const dtype_mod.BlockMXFP4, a_lo: common.QKV16i8, a_hi: common.QKV16i8) common.QKV4i32 {
    switch (isa.tier) {
        .x86_vnni, .x86_avx2 => {
            const packed_v: common.QKV16u8 = @bitCast(w.qs);
            const lo_idx = packed_v & @as(common.QKV16u8, @splat(0x0f));
            const hi_idx = packed_v >> @as(common.QKV16u8, @splat(4));
            const idx: common.QKV32u8 = std.simd.join(lo_idx, hi_idx);
            const a32: common.QKV32i8 = std.simd.join(a_lo, a_hi);
            const svec = common.pshufbI8x32(kvalues_x2, idx);
            const mag: common.QKV32u8 = @bitCast(common.pshufbI8x32(kmag_x2, idx));
            // vpsignb zeroes where the code is 0 — magnitude 0 there anyway.
            const adj = common.psignI8x32(a32, svec);
            var acc8: common.QKV8i32 = @splat(0);
            acc8 = if (comptime isa.tier == .x86_vnni)
                common.dpbusdI32x8(acc8, mag, adj)
            else
                common.maddubsDotGroupsI32x8(acc8, mag, adj);
            return common.addHalvesI32x8(acc8);
        },
        // NEON `tbl`+`sdot`, or their exact portable twins.
        .neon_i8mm, .neon_sdot, .portable, .scalar => {
            const d = decodeBlock(&w.qs);
            var iacc: common.QKV4i32 = @splat(0);
            iacc = common.sdotI8x16(iacc, d.lo, a_lo);
            iacc = common.sdotI8x16(iacc, d.hi, a_hi);
            return iacc;
        },
    }
}

inline fn blockContribution(w: *const dtype_mod.BlockMXFP4, a_d: f32, a_lo: common.QKV16i8, a_hi: common.QKV16i8) common.QKV4f32 {
    const iacc = blockDotI32(w, a_lo, a_hi);
    const scale: common.QKV4f32 = @splat(e8m0_half_lut[w.e] * a_d);
    return @as(common.QKV4f32, @floatFromInt(iacc)) * scale;
}

inline fn accumulateBlock(acc: common.QKV4f32, w: *const dtype_mod.BlockMXFP4, a_d: f32, a_lo: common.QKV16i8, a_hi: common.QKV16i8) common.QKV4f32 {
    return acc + blockContribution(w, a_d, a_lo, a_hi);
}

const col_block = 4;

/// out[r0..r1, c0..c1] over an MXFP4 rhs, Q8_0 lhs rows. Columns advance in
/// fused four-column groups (four independent f32 accumulator chains per
/// activation block load) with a single-column tail.
pub fn matmulMXFP4RhsTile(
    out: []f32,
    lhs_blocks: []const dtype_mod.BlockQ8_0,
    rhs: *const types.QuantizedMatmulRhsMXFP4,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const bpc = rhs.rows.blocks_per_row;
    var i = r0;
    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * bpc ..][0..bpc];
        var j = c0;

        while (j + col_block <= c1) : (j += col_block) {
            var acc: [col_block]common.QKV4f32 = undefined;
            inline for (0..col_block) |c| acc[c] = @splat(0);
            var cols: [col_block][]const dtype_mod.BlockMXFP4 = undefined;
            inline for (0..col_block) |c| cols[c] = rhs.rows.blocks[(j + c) * bpc ..][0..bpc];

            // Block pairs per column: independent contributions pipeline the
            // integer dots; the per-column float adds stay in serial order,
            // keeping results bitwise identical to the one-block loop.
            var b: usize = 0;
            while (b + 2 <= bpc) : (b += 2) {
                const a0 = &lhs_row[b];
                const a1 = &lhs_row[b + 1];
                const a0_d = common.f16BitsToF32(a0.d);
                const a1_d = common.f16BitsToF32(a1.d);
                const a0_lo: common.QKV16i8 = @bitCast(a0.qs[0..16].*);
                const a0_hi: common.QKV16i8 = @bitCast(a0.qs[16..32].*);
                const a1_lo: common.QKV16i8 = @bitCast(a1.qs[0..16].*);
                const a1_hi: common.QKV16i8 = @bitCast(a1.qs[16..32].*);
                inline for (0..col_block) |c| {
                    const t0 = blockContribution(&cols[c][b], a0_d, a0_lo, a0_hi);
                    const t1 = blockContribution(&cols[c][b + 1], a1_d, a1_lo, a1_hi);
                    acc[c] += t0;
                    acc[c] += t1;
                }
            }
            if (b < bpc) {
                const a = &lhs_row[b];
                const a_d = common.f16BitsToF32(a.d);
                const a_lo: common.QKV16i8 = @bitCast(a.qs[0..16].*);
                const a_hi: common.QKV16i8 = @bitCast(a.qs[16..32].*);
                inline for (0..col_block) |c| {
                    acc[c] = accumulateBlock(acc[c], &cols[c][b], a_d, a_lo, a_hi);
                }
            }
            inline for (0..col_block) |c| out[i * n + j + c] = @reduce(.Add, acc[c]);
        }

        while (j < c1) : (j += 1) {
            const col = rhs.columnBlocks(j);
            var acc: common.QKV4f32 = @splat(0);
            for (0..bpc) |b| {
                const a = &lhs_row[b];
                acc = accumulateBlock(acc, &col[b], common.f16BitsToF32(a.d), @bitCast(a.qs[0..16].*), @bitCast(a.qs[16..32].*));
            }
            out[i * n + j] = @reduce(.Add, acc);
        }
    }
}

test {
    _ = @import("mxfp4_tests.zig");
}
