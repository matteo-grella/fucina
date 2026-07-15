//! Hot TQ2_0 ternary kernels: {-1,0,+1} weights (BitNet b1.58 class) at
//! 2.0625 bits/weight in the ggml TQ2_0 block layout (BlockTQ2_0: qs[64]
//! 2-bit crumbs storing w+1 in {0,1,2}, fp16 d, 256 elements — GGUF type 35).
//!
//! Int8 flagship (Q8_K activations): the crumbs multiply as unsigned codes,
//! dot = sum((w+1)*a) - sum(a); the Q8_K bsums supply sum(a), so the hot loop
//! is shift/mask unpack + int8 group dots with ONE subtraction per block:
//!   aarch64: sdot (i32 exact), x86: vpdpbusd (VNNI) or vpmaddubsw+vpmaddwd
//!   (AVX2 — a maddubs pair sum is at most 2*127*2 = 508, no i16 saturation),
//!   elsewhere: the portable @Vector twins of those primitives.
//! Every arm accumulates the exact block integer (no saturation anywhere), so
//! results are cross-ISA bitwise identical to the cold.zig scalar reference.
//!
//! F32 path (no activation quantization — the training/STE forward): ternary
//! multiply is two bitwise ops and one add per vector, exact in IEEE fp32:
//!   w*x = (x XOR s) AND m,  s = signbit where w == -1,  m = ~0 where w != 0.
//! Fixed 4-lane accumulation order keeps this path bitwise identical across
//! ISAs too (bitwise ops + fp adds in one association order).
//!
//! Encoder: quantize_row_tq2_0 parity with the ggml reference (per-block
//! absmax d, round-half-away, crumb n of byte m covers element m + n*32 of
//! each 128-group), plus the BitNet b1.58 recipe (per-tensor absmean scale,
//! round-clip) for straight-through-estimator training and ternary-native ES.

const std = @import("std");
const builtin = @import("builtin");
const types_mod = @import("types.zig");
const common = @import("common.zig");

const Allocator = std.mem.Allocator;

const BlockTQ2_0 = types_mod.BlockTQ2_0;
const BlockQ2_0 = types_mod.BlockQ2_0;
const BlockQ8_0 = types_mod.BlockQ8_0;
const BlockQ8_K = types_mod.BlockQ8_K;
const QuantizedFormatError = types_mod.QuantizedFormatError;
const QuantizedMatmulRhsTQ2_0 = types_mod.QuantizedMatmulRhsTQ2_0;
const QuantizedMatmulRhsQ2_0 = types_mod.QuantizedMatmulRhsQ2_0;
const checkedProduct = types_mod.checkedProduct;
const q2_0_block_size = types_mod.q2_0_block_size;
const q8_0_block_size = types_mod.q8_0_block_size;
const qk_k_block_size = types_mod.qk_k_block_size;

const QKV4f32 = common.QKV4f32;
const QKV4i32 = common.QKV4i32;
const QKV8i32 = common.QKV8i32;
const QKV16i8 = common.QKV16i8;
const QKV16u8 = common.QKV16u8;
const QKV32i8 = common.QKV32i8;
const QKV32u8 = common.QKV32u8;
const dpbusdI32x8 = common.dpbusdI32x8;
const f16BitsToF32 = common.f16BitsToF32;
const f32ToF16Bits = common.f32ToF16Bits;
const has_x86_vnni_ymm = common.has_x86_vnni_ymm;
const maddubsDotGroupsI32x8 = common.maddubsDotGroupsI32x8;
const roundHalfAwayFromZero = common.roundHalfAwayFromZero;
const sdotI8x16 = common.sdotI8x16;

/// Weight rows processed together per tile step: the four columns share every
/// activation vector load and the per-block bsum total.
pub const ternary_col_block: usize = 4;

// ---------------- encode / decode ----------------

fn tq2_0BlockCount(len: usize) !usize {
    if (len == 0 or len % qk_k_block_size != 0) return QuantizedFormatError.InvalidQuantizedLength;
    return len / qk_k_block_size;
}

/// ggml quantize_row_tq2_0 parity: per-block absmax d (fp16), crumbs store
/// round(x/d) + 1. Element m + n*32 of each 128-group lands in bits 2n+1:2n
/// of byte m (group 0 -> qs[0..32], group 1 -> qs[32..64]).
pub fn quantizeRowTQ2_0Into(dst: []BlockTQ2_0, src: []const f32) !void {
    const block_count = try tq2_0BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;

    for (dst, 0..) |*block, block_index| {
        const x = src[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var amax: f32 = 0;
        for (x) |v| amax = @max(amax, @abs(v));

        const d = amax;
        const id: f32 = if (d != 0) 1.0 / d else 0.0;
        block.d = f32ToF16Bits(d);
        encodeCrumbs(block, x, id);
    }
}

/// BitNet b1.58 per-tensor scale: d = clamp(mean(|w|), 1e-5, inf). The row
/// encoder below round-CLIPS against an explicit d, so encode(w, d) realizes
/// W_q = clamp(round(w/d), -1, +1) with dequantized values W_q * d.
pub fn ternaryAbsmeanScale(src: []const f32) f32 {
    var sum: f64 = 0;
    for (src) |v| sum += @abs(v);
    const mean: f32 = if (src.len == 0) 0 else @floatCast(sum / @as(f64, @floatFromInt(src.len)));
    return @max(mean, 1e-5);
}

/// Encode one row against an explicit (per-tensor) scale d, clipping to the
/// ternary range. Every block stores the same d, so the blocks stay valid
/// standalone TQ2_0 (dequantize/matmul need no side channel).
pub fn quantizeRowTQ2_0ScaledInto(dst: []BlockTQ2_0, src: []const f32, d: f32) !void {
    const block_count = try tq2_0BlockCount(src.len);
    if (dst.len != block_count) return QuantizedFormatError.InvalidQuantizedLength;
    if (!(d > 0)) return QuantizedFormatError.InvalidQuantizedLength;

    const id: f32 = 1.0 / d;
    for (dst, 0..) |*block, block_index| {
        const x = src[block_index * qk_k_block_size ..][0..qk_k_block_size];
        block.d = f32ToF16Bits(d);
        encodeCrumbs(block, x, id);
    }
}

fn encodeCrumbs(block: *BlockTQ2_0, x: *const [qk_k_block_size]f32, id: f32) void {
    for (0..2) |group| {
        const base = group * 128;
        for (0..32) |m| {
            var q: u8 = 0;
            inline for (0..4) |n| {
                // Clamp in the FLOAT domain before the int conversion:
                // @intFromFloat of NaN/inf is safety-checked illegal behavior,
                // and divergent latent weights reach this encoder through
                // dotTernarySte with no upstream finite guard (NaN maps to
                // code 1 = zero weight; a saturated inf/amax=inf row encodes
                // as x*id = NaN -> zeros). Finite ggml-parity inputs satisfy
                // |round(x*id)| <= 1, so the clamp never fires for them, and
                // the round-CLIP of the explicit-scale variant is realized by
                // the same clamp.
                const rounded = roundHalfAwayFromZero(x[base + n * 32 + m] * id);
                const bounded: f32 = if (std.math.isNan(rounded)) 0.0 else @min(1.0, @max(-1.0, rounded));
                const xi: i32 = @intFromFloat(bounded);
                q += @as(u8, @intCast((xi + 1) & 3)) << (2 * n);
            }
            block.qs[group * 32 + m] = q;
        }
    }
}

pub fn quantizedMatmulRhsTQ2_0FromBlocks(
    allocator: Allocator,
    k: usize,
    n: usize,
    blocks: []const BlockTQ2_0,
) !QuantizedMatmulRhsTQ2_0 {
    const blocks_per_row = try tq2_0BlockCount(k);
    if (blocks.len != try checkedProduct(n, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;
    const owned = try allocator.dupe(BlockTQ2_0, blocks);
    return .{
        .rows = .{
            .allocator = allocator,
            .blocks = owned,
            .rows = n,
            .cols = k,
            .blocks_per_row = blocks_per_row,
        },
        .k = k,
        .n = n,
    };
}

/// Wrap caller-owned TQ2_0 blocks (row-major [n] weight rows of k/256
/// blocks each; view row c = weight row c = output column c) as a matmul
/// RHS WITHOUT copying: the container borrows `blocks` (allocator = null),
/// so `deinit` frees nothing — the caller keeps ownership and the blocks
/// must outlive the view.
pub fn quantizedMatmulRhsTQ2_0FromBorrowedBlocks(
    k: usize,
    n: usize,
    blocks: []BlockTQ2_0,
) !QuantizedMatmulRhsTQ2_0 {
    const blocks_per_row = try tq2_0BlockCount(k);
    if (blocks.len != try checkedProduct(n, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;
    return .{
        .rows = .{
            .allocator = null,
            .blocks = blocks,
            .rows = n,
            .cols = k,
            .blocks_per_row = blocks_per_row,
        },
        .k = k,
        .n = n,
    };
}

/// Encode a dense row-major [n][k] f32 weight matrix (ggml per-block absmax
/// semantics; pass an absmean scale via the Scaled variant for b1.58 tensors).
pub fn quantizedMatmulRhsTQ2_0FromF32(
    allocator: Allocator,
    k: usize,
    n: usize,
    weights: []const f32,
) !QuantizedMatmulRhsTQ2_0 {
    return rhsFromF32(allocator, k, n, weights, null);
}

/// Encode with the BitNet b1.58 per-tensor absmean scale (computed over the
/// whole matrix), round-clipped to {-1, 0, +1}.
pub fn quantizedMatmulRhsTQ2_0FromF32Absmean(
    allocator: Allocator,
    k: usize,
    n: usize,
    weights: []const f32,
) !QuantizedMatmulRhsTQ2_0 {
    return rhsFromF32(allocator, k, n, weights, ternaryAbsmeanScale(weights));
}

fn rhsFromF32(allocator: Allocator, k: usize, n: usize, weights: []const f32, scale: ?f32) !QuantizedMatmulRhsTQ2_0 {
    const blocks_per_row = try tq2_0BlockCount(k);
    if (weights.len != try checkedProduct(n, k)) return QuantizedFormatError.InvalidQuantizedLength;
    const blocks = try allocator.alloc(BlockTQ2_0, n * blocks_per_row);
    errdefer allocator.free(blocks);
    for (0..n) |row| {
        const dst = blocks[row * blocks_per_row ..][0..blocks_per_row];
        const src = weights[row * k ..][0..k];
        if (scale) |d| {
            try quantizeRowTQ2_0ScaledInto(dst, src, d);
        } else {
            try quantizeRowTQ2_0Into(dst, src);
        }
    }
    return .{
        .rows = .{
            .allocator = allocator,
            .blocks = blocks,
            .rows = n,
            .cols = k,
            .blocks_per_row = blocks_per_row,
        },
        .k = k,
        .n = n,
    };
}

// ---------------- int8 flagship: block dots ----------------

inline fn crumb16(q: QKV16u8, comptime lane: usize) QKV16i8 {
    const shift: @Vector(16, u3) = @splat(2 * lane);
    return @bitCast((q >> shift) & @as(QKV16u8, @splat(3)));
}

inline fn crumb32(q: QKV32u8, comptime lane: usize) QKV32u8 {
    const shift: @Vector(32, u3) = @splat(2 * lane);
    return (q >> shift) & @as(QKV32u8, @splat(3));
}

inline fn dotGroups32(acc: QKV8i32, codes: QKV32u8, a: QKV32i8) QKV8i32 {
    if (comptime has_x86_vnni_ymm) return dpbusdI32x8(acc, codes, a);
    return maddubsDotGroupsI32x8(acc, codes, a);
}

/// sum over the 16 per-16-element activation sums = sum(a) for the block.
inline fn blockBsumTotal(a: *const BlockQ8_K) i32 {
    const sums: @Vector(16, i16) = a.bsums;
    return @reduce(.Add, @as(@Vector(16, i32), sums));
}

/// sum((w+1) * a) for `width` weight blocks against one activation block —
/// the unsigned-code dot; activation vectors are shared across all columns.
/// aarch64 shape: 16-byte granules through sdot (crumbs {0,1,2} are valid
/// signed bytes). Elsewhere: 32-byte granules through vpdpbusd/maddubs or
/// their portable twins. Every arm accumulates the exact i32 (max |isum| is
/// 256*2*127 = 65024), so all arms and widths agree bitwise — the fused
/// 4-column tile and the width-1 tail take the same body.
inline fn blockCodeDotW(comptime width: usize, w: [width]*const BlockTQ2_0, a: *const BlockQ8_K) [width]i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        var acc: [width]QKV4i32 = @splat(@splat(0));
        inline for ([_]usize{ 0, 32 }) |j| {
            var q0: [width]QKV16u8 = undefined;
            var q1: [width]QKV16u8 = undefined;
            inline for (0..width) |ci| {
                q0[ci] = w[ci].qs[j..][0..16].*;
                q1[ci] = w[ci].qs[j + 16 ..][0..16].*;
            }
            inline for (0..4) |lane| {
                const a0: QKV16i8 = a.qs[j * 4 + lane * 32 ..][0..16].*;
                const a1: QKV16i8 = a.qs[j * 4 + lane * 32 + 16 ..][0..16].*;
                inline for (0..width) |ci| {
                    acc[ci] = sdotI8x16(acc[ci], crumb16(q0[ci], lane), a0);
                    acc[ci] = sdotI8x16(acc[ci], crumb16(q1[ci], lane), a1);
                }
            }
        }
        var out: [width]i32 = undefined;
        inline for (0..width) |ci| out[ci] = @reduce(.Add, acc[ci]);
        return out;
    }
    var acc: [width]QKV8i32 = @splat(@splat(0));
    inline for ([_]usize{ 0, 32 }) |j| {
        var g: [width]QKV32u8 = undefined;
        inline for (0..width) |ci| g[ci] = w[ci].qs[j..][0..32].*;
        inline for (0..4) |lane| {
            const av: QKV32i8 = a.qs[j * 4 + lane * 32 ..][0..32].*;
            inline for (0..width) |ci| {
                acc[ci] = dotGroups32(acc[ci], crumb32(g[ci], lane), av);
            }
        }
    }
    var out: [width]i32 = undefined;
    inline for (0..width) |ci| out[ci] = @reduce(.Add, acc[ci]);
    return out;
}

// ---------------- int8 flagship: tile / range kernels ----------------

/// Row blocks a bsum cache covers without allocating: 256 blocks = k up to
/// 65536. Longer rows fall back to recomputing per column group — the values
/// are identical either way, so the cache is bitwise-invisible.
const bsum_cache_blocks: usize = 256;

/// out[r, c] tiles of LHS Q8_K activation rows x TQ2_0 weight rows (RHS
/// convention: rhs row c is output column c). Allocation-free, unchecked:
/// callers validate shapes (out is m*n, lhs_blocks is m*blocks_per_row).
/// PRECONDITION: each activation block's bsums must equal the exact per-16
/// sums of its qs (quantizeRowsQ8_K guarantees this) — the kernel computes
/// dot = sum(codes*a) - sum(bsums), unlike the cold table path which reads
/// only qs, so stale/foreign bsums silently corrupt results.
pub fn matmulTQ2_0RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsTQ2_0,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.rows.blocks_per_row;
    const cached = blocks_per_row <= bsum_cache_blocks;
    var bsum_cache: [bsum_cache_blocks]i32 = undefined;
    var r = r0;
    while (r < r1) : (r += 1) {
        const arow = lhs_blocks[r * blocks_per_row ..][0..blocks_per_row];
        const orow = out[r * n ..][0..n];
        if (cached) {
            for (arow, 0..) |*a, bi| bsum_cache[bi] = blockBsumTotal(a);
        }
        var c = c0;
        while (c + ternary_col_block <= c1) : (c += ternary_col_block) {
            var wcols: [ternary_col_block][]const BlockTQ2_0 = undefined;
            inline for (0..ternary_col_block) |ci| wcols[ci] = rhs.columnBlocks(c + ci);
            var sums: [ternary_col_block]f32 = @splat(0);
            for (arow, 0..) |*a, bi| {
                const bsum = if (cached) bsum_cache[bi] else blockBsumTotal(a);
                const w: [ternary_col_block]*const BlockTQ2_0 = .{
                    &wcols[0][bi], &wcols[1][bi], &wcols[2][bi], &wcols[3][bi],
                };
                const dots = blockCodeDotW(ternary_col_block, w, a);
                inline for (0..ternary_col_block) |ci| {
                    const isum = dots[ci] - bsum;
                    sums[ci] += f16BitsToF32(w[ci].d) * a.d * @as(f32, @floatFromInt(isum));
                }
            }
            inline for (0..ternary_col_block) |ci| orow[c + ci] = sums[ci];
        }
        while (c < c1) : (c += 1) {
            const wcol = rhs.columnBlocks(c);
            var sum: f32 = 0;
            for (arow, 0..) |*a, bi| {
                const bsum = if (cached) bsum_cache[bi] else blockBsumTotal(a);
                const dots = blockCodeDotW(1, .{&wcol[bi]}, a);
                const isum = dots[0] - bsum;
                sum += f16BitsToF32(wcol[bi].d) * a.d * @as(f32, @floatFromInt(isum));
            }
            orow[c] = sum;
        }
    }
}

pub fn matmulTQ2_0RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_K,
    rhs: *const QuantizedMatmulRhsTQ2_0,
    m: usize,
    n: usize,
    r0: usize,
    r1: usize,
) void {
    _ = m;
    matmulTQ2_0RhsTile(out, lhs_blocks, rhs, n, r0, r1, 0, n);
}

// ---------------- Q2_0 (Bonsai g128) int8 kernels ----------------
//
// Q2_0 is the PrismML/Bonsai ternary container: 128-element blocks, four
// sequential LSB-first 2-bit codes per byte, one f16 scale, codes q in
// {0,1,2,3} decoding to (q-1)*d (files written by the reference encoder are
// pure ternary — code 3 never occurs). Activations are Q8_0 rows (32-element
// sub-blocks), so dot = d0 * sum_k d1_k * isum_k with
// isum_k = sum(q*a) - sum(a): the codes multiply UNSIGNED through
// sdot/vpdpbusd/maddubs and one bsum subtraction per 32-group replaces the
// per-lane "-1" — the bsums are computed once per LHS row and shared across
// every output column. Every arm accumulates the exact i32
// (|sum(q*a)| <= 32*3*127 = 12192, maddubs pair sums <= 2*127*3 = 762, no
// i16 saturation), so all arms are bitwise identical to cold.zig's
// dotQ2_0Q8_0 reference.

/// Unsigned codes for 16 consecutive elements of one Q2_0 block: bytes
/// `first_byte..first_byte+4` each replicated 4x (tbl/pshufb), lanes shifted
/// by {0,2,4,6} and masked to 2 bits.
inline fn q2_0Codes16(qs: QKV32u8, comptime first_byte: usize) QKV16u8 {
    const mask = comptime blk: {
        @setEvalBranchQuota(4000);
        var idx: [16]i32 = undefined;
        for (0..16) |lane| idx[lane] = @intCast(first_byte + lane / 4);
        break :blk idx;
    };
    const shifts = comptime blk: {
        @setEvalBranchQuota(4000);
        var sv: [16]u3 = undefined;
        for (0..16) |lane| sv[lane] = @intCast((lane % 4) * 2);
        break :blk sv;
    };
    const repl: QKV16u8 = @shuffle(u8, qs, undefined, @as(@Vector(16, i32), mask));
    return (repl >> shifts) & @as(QKV16u8, @splat(3));
}

/// 32-lane twin of `q2_0Codes16` (bytes `first_byte..first_byte+8`).
inline fn q2_0Codes32(qs: QKV32u8, comptime first_byte: usize) QKV32u8 {
    const mask = comptime blk: {
        @setEvalBranchQuota(4000);
        var idx: [32]i32 = undefined;
        for (0..32) |lane| idx[lane] = @intCast(first_byte + lane / 4);
        break :blk idx;
    };
    const shifts = comptime blk: {
        @setEvalBranchQuota(4000);
        var sv: [32]u3 = undefined;
        for (0..32) |lane| sv[lane] = @intCast((lane % 4) * 2);
        break :blk sv;
    };
    const repl: QKV32u8 = @shuffle(u8, qs, undefined, @as(@Vector(32, i32), mask));
    return (repl >> shifts) & @as(QKV32u8, @splat(3));
}

/// sum(a) over one Q8_0 sub-block, exact i32.
inline fn q8_0BlockSum(a: *const BlockQ8_0) i32 {
    const v: QKV32i8 = a.qs;
    return @reduce(.Add, @as(@Vector(32, i32), v));
}

/// Unpacked unsigned codes for one 32-element sub-block of a Q2_0 block, in
/// the granule shape the ISA's dot arm wants: two 16-byte vectors on aarch64
/// (sdot), one 32-byte vector elsewhere (vpdpbusd/maddubs or portable twins).
const Q2_0Codes = if (builtin.cpu.arch == .aarch64)
    struct { lo: QKV16i8, hi: QKV16i8 }
else
    QKV32u8;

/// Unpack sub-block `k` (elements k*32..k*32+32) of one Q2_0 block. Codes
/// stay unsigned {0..3}; the matmul subtracts the activation bsum instead of
/// materializing the "-1" per lane (dot = sum(q*a) - sum(a)).
inline fn q2_0UnpackCodes(qs: QKV32u8, comptime k: usize) Q2_0Codes {
    if (comptime builtin.cpu.arch == .aarch64) {
        return .{
            .lo = @bitCast(q2_0Codes16(qs, k * 8)),
            .hi = @bitCast(q2_0Codes16(qs, k * 8 + 4)),
        };
    }
    return q2_0Codes32(qs, k * 8);
}

/// sum(q * a) of unpacked codes against one Q8_0 sub-block, exact i32
/// (|sum(q*a)| <= 32*3*127 = 12192; maddubs pair sums <= 2*127*3 = 762, no
/// i16 saturation) — all arms bitwise identical to cold.zig's dotQ2_0Q8_0.
inline fn q2_0CodeDot(codes: Q2_0Codes, a: *const BlockQ8_0) i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        const a0: QKV16i8 = a.qs[0..16].*;
        const a1: QKV16i8 = a.qs[16..32].*;
        var acc: QKV4i32 = @splat(0);
        acc = sdotI8x16(acc, codes.lo, a0);
        acc = sdotI8x16(acc, codes.hi, a1);
        return @reduce(.Add, acc);
    }
    const av: QKV32i8 = a.qs;
    var acc: QKV8i32 = @splat(0);
    acc = dotGroups32(acc, codes, av);
    return @reduce(.Add, acc);
}

/// LHS rows sharing one weight-block unpack (comptime, so the code vectors
/// and accumulators stay in registers): with rows-inner-of-columns the RHS
/// is streamed once per row pair instead of once per row, halving prefill
/// weight traffic. Decode (m = 1) takes the width-1 instantiation.
const q2_0_row_block: usize = 2;

/// LHS Q8_0 sub-blocks a row-side bsum cache covers without allocating:
/// 2048 sub-blocks = k up to 65536 (16 KiB of stack per cached row). Longer
/// rows recompute per column group — identical values either way, the cache
/// is bitwise-invisible.
const q2_0_bsum_cache_subs: usize = 2048;

/// One (row-width x 4-column) micro-tile over a full k reduction: weights
/// unpacked once per (column, sub-block) and dotted against `rw` activation
/// rows, everything comptime-shaped so codes and accumulators stay in
/// registers. Accumulation order per output element matches cold.zig
/// dotQ2_0Q8_0 exactly (blocks in order, sum += d0 * d1_k * isum_k).
inline fn q2_0MicroTile(
    comptime rw: usize,
    comptime cw: usize,
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ2_0,
    n: usize,
    r: usize,
    c: usize,
    bsums: *const [q2_0_row_block][q2_0_bsum_cache_subs]i32,
    cached: bool,
) void {
    const sub_per_block = q2_0_block_size / q8_0_block_size; // 4
    const blocks_per_row = rhs.rows.blocks_per_row;
    const sub_blocks_per_row = blocks_per_row * sub_per_block;

    var wcols: [cw][]const BlockQ2_0 = undefined;
    inline for (0..cw) |ci| wcols[ci] = rhs.columnBlocks(c + ci);
    var arows: [rw][]const BlockQ8_0 = undefined;
    inline for (0..rw) |r2| arows[r2] = lhs_blocks[(r + r2) * sub_blocks_per_row ..][0..sub_blocks_per_row];

    var sums: [rw][cw]f32 = @splat(@splat(0));
    var bi: usize = 0;
    while (bi < blocks_per_row) : (bi += 1) {
        var d0: [cw]f32 = undefined;
        inline for (0..cw) |ci| d0[ci] = f16BitsToF32(wcols[ci][bi].d);
        // Per-block partial sums, folded into the running total only after
        // the block's four sub-blocks — the exact nesting of the cold
        // reference (matmulTableQ8_0RhsTile adds one dotQ2_0Q8_0 per block).
        var part: [rw][cw]f32 = @splat(@splat(0));
        inline for (0..sub_per_block) |k| {
            const si = bi * sub_per_block + k;
            var codes: [cw]Q2_0Codes = undefined;
            inline for (0..cw) |ci| codes[ci] = q2_0UnpackCodes(wcols[ci][bi].qs, k);
            inline for (0..rw) |r2| {
                const a = &arows[r2][si];
                const bsum = if (cached) bsums[r2][si] else q8_0BlockSum(a);
                const d1 = f16BitsToF32(a.d);
                inline for (0..cw) |ci| {
                    const isum = q2_0CodeDot(codes[ci], a) - bsum;
                    part[r2][ci] += d0[ci] * d1 * @as(f32, @floatFromInt(isum));
                }
            }
        }
        inline for (0..rw) |r2| {
            inline for (0..cw) |ci| sums[r2][ci] += part[r2][ci];
        }
    }
    inline for (0..rw) |r2| {
        inline for (0..cw) |ci| out[(r + r2) * n + c + ci] = sums[r2][ci];
    }
}

/// out[r, c] tiles of LHS Q8_0 activation rows x Q2_0 weight rows (RHS
/// convention: rhs row c is output column c). Allocation-free, unchecked:
/// callers validate shapes (out is m*n, lhs_blocks is m*4*blocks_per_row).
/// Row pairs share each weight unpack (`q2_0MicroTile`); every micro-tile
/// width is a comptime instantiation, so all shapes take register-resident
/// bodies and every path is bitwise identical to cold.zig's dotQ2_0Q8_0.
pub fn matmulQ2_0RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ2_0,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const sub_per_block = q2_0_block_size / q8_0_block_size; // 4
    const blocks_per_row = rhs.rows.blocks_per_row;
    const sub_blocks_per_row = blocks_per_row * sub_per_block;
    const cached = sub_blocks_per_row <= q2_0_bsum_cache_subs;
    var bsum_cache: [q2_0_row_block][q2_0_bsum_cache_subs]i32 = undefined;

    var r = r0;
    while (r < r1) : (r += q2_0_row_block) {
        const rw_live = @min(q2_0_row_block, r1 - r);
        if (cached) {
            for (0..rw_live) |r2| {
                const arow = lhs_blocks[(r + r2) * sub_blocks_per_row ..][0..sub_blocks_per_row];
                for (arow, 0..) |*a, si| bsum_cache[r2][si] = q8_0BlockSum(a);
            }
        }
        var c = c0;
        inline for (.{ q2_0_row_block, 1 }) |rw| {
            if (rw_live == rw) {
                while (c + ternary_col_block <= c1) : (c += ternary_col_block) {
                    q2_0MicroTile(rw, ternary_col_block, out, lhs_blocks, rhs, n, r, c, &bsum_cache, cached);
                }
                while (c < c1) : (c += 1) {
                    q2_0MicroTile(rw, 1, out, lhs_blocks, rhs, n, r, c, &bsum_cache, cached);
                }
            }
        }
    }
}

pub fn matmulQ2_0RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ2_0,
    m: usize,
    n: usize,
    r0: usize,
    r1: usize,
) void {
    _ = m;
    matmulQ2_0RhsTile(out, lhs_blocks, rhs, n, r0, r1, 0, n);
}

// ---------------- f32-activation path (mul-free, IEEE-exact) ----------------

/// One weight row against one dense f32 activation row:
/// y = sum_blocks d_block * sum((x XOR s) AND m). The fixed 4-lane
/// accumulator makes the result bitwise reproducible on every ISA.
/// Two deliberate semantic deltas vs dequantize-then-multiply: a NaN/inf
/// activation is masked to +0 where the weight is 0 (the AND eats it, where
/// NaN*0 would propagate), and the invalid crumb code 3 — never emitted by
/// the encoders, rejected by es.addTernaryParam — reads as +1 here vs +2 on
/// the int8 code-dot path (garbage in, differently-shaped garbage out).
pub fn dotTQ2_0F32(wblocks: []const BlockTQ2_0, x: []const f32) f32 {
    std.debug.assert(x.len == wblocks.len * qk_k_block_size);
    var total: f32 = 0;
    for (wblocks, 0..) |*w, block_index| {
        const xb = x[block_index * qk_k_block_size ..][0..qk_k_block_size];
        var acc: QKV4f32 = @splat(0);
        inline for ([_]usize{ 0, 32 }) |j| {
            inline for (0..4) |lane| {
                var m: usize = 0;
                while (m < 32) : (m += 4) {
                    const qb: @Vector(4, u8) = w.qs[j + m ..][0..4].*;
                    const shift: @Vector(4, u3) = @splat(2 * lane);
                    const q: @Vector(4, u32) = (qb >> shift) & @as(@Vector(4, u8), @splat(3));
                    const sgn = @select(u32, q == @as(@Vector(4, u32), @splat(0)), @as(@Vector(4, u32), @splat(0x8000_0000)), @as(@Vector(4, u32), @splat(0)));
                    const msk = @select(u32, q == @as(@Vector(4, u32), @splat(1)), @as(@Vector(4, u32), @splat(0)), @as(@Vector(4, u32), @splat(0xFFFF_FFFF)));
                    const xv: QKV4f32 = xb[j * 4 + lane * 32 + m ..][0..4].*;
                    const bits: @Vector(4, u32) = @bitCast(xv);
                    acc += @as(QKV4f32, @bitCast((bits ^ sgn) & msk));
                }
            }
        }
        const lane_sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        total += f16BitsToF32(w.d) * lane_sum;
    }
    return total;
}

/// Dense f32 LHS x TQ2_0 RHS tiles — the no-activation-quantization forward.
pub fn matmulTQ2_0F32RhsTile(
    out: []f32,
    lhs: []const f32,
    rhs: *const QuantizedMatmulRhsTQ2_0,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const k = rhs.k;
    var r = r0;
    while (r < r1) : (r += 1) {
        const xrow = lhs[r * k ..][0..k];
        const orow = out[r * n ..][0..n];
        var c = c0;
        while (c < c1) : (c += 1) {
            orow[c] = dotTQ2_0F32(rhs.columnBlocks(c), xrow);
        }
    }
}

pub fn matmulTQ2_0F32RhsRange(
    out: []f32,
    lhs: []const f32,
    rhs: *const QuantizedMatmulRhsTQ2_0,
    m: usize,
    n: usize,
    r0: usize,
    r1: usize,
) void {
    _ = m;
    matmulTQ2_0F32RhsTile(out, lhs, rhs, n, r0, r1, 0, n);
}

test {
    _ = @import("ternary_tests.zig");
}
