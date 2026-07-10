//! Behavioral tests for the Q6_K / Q6_Kx4 matmul kernels (`q6_k.zig`): paired
//! gate/up tiling vs. two independent tiles, column-outer vs. row-outer tiling,
//! the lane-packed Q8_Kx4 column-outer kernel (aligned + tail-padded), and the
//! exact-parity suite of the x86/portable SIMD accumulate arms vs. the scalar
//! reference.
const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("../../tensor.zig");
const qm = @import("../quant.zig");
const common = @import("common.zig");
const q6_k = @import("q6_k.zig");

const Tensor = tensor.Tensor;

const BlockQ6_K = qm.BlockQ6_K;
const BlockQ6_Kx4 = qm.BlockQ6_Kx4;
const BlockQ8_K = qm.BlockQ8_K;
const BlockQ8_Kx4 = qm.BlockQ8_Kx4;
const quantizedMatmulRhsQ6_KFromBlocks = qm.quantizedMatmulRhsQ6_KFromBlocks;
const qk_k_block_size = qm.qk_k_block_size;
const QKV4f32 = common.QKV4f32;
const f32ToF16Bits = common.f32ToF16Bits;
const q8_0_row_block = common.q8_0_row_block;

const packMatmulRhsQ6_Kx4 = q6_k.packMatmulRhsQ6_Kx4;
const matmulQ6_Kx4RhsTile = q6_k.matmulQ6_Kx4RhsTile;
const matmulQ6_Kx4RhsRange = q6_k.matmulQ6_Kx4RhsRange;
const matmulQ6_Kx4RhsPairTile = q6_k.matmulQ6_Kx4RhsPairTile;
const matmulQ6_KRhsTile = q6_k.matmulQ6_KRhsTile;
const matmulQ6_KRhsRange = q6_k.matmulQ6_KRhsRange;
const matmulQ6_KRhsCompactColOuter = q6_k.matmulQ6_KRhsCompactColOuter;
const matmulQ6_KCompactQ8_Kx4ColOuter = q6_k.matmulQ6_KCompactQ8_Kx4ColOuter;

test "Q6_Kx4 paired gate/up tile matches two independent tiles" {
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 8;
    const m = 5;
    const bpc = k / qk_k_block_size;

    const gate_blocks = try allocator.alloc(BlockQ6_K, n * bpc);
    defer allocator.free(gate_blocks);
    const up_blocks = try allocator.alloc(BlockQ6_K, n * bpc);
    defer allocator.free(up_blocks);
    for (gate_blocks, up_blocks, 0..) |*g, *u, bi| {
        g.d = f32ToF16Bits(0.021 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        u.d = f32ToF16Bits(0.017 + 0.001 * @as(f32, @floatFromInt((bi + 3) % 5)));
        for (&g.scales, &u.scales, 0..) |*gs, *us, i| {
            gs.* = @intCast(@as(i32, @intCast((i * 7 + bi * 5) % 96)) - 48);
            us.* = @intCast(@as(i32, @intCast((i * 11 + bi * 3) % 96)) - 48);
        }
        for (&g.ql, &u.ql, 0..) |*gq, *uq, i| {
            gq.* = @intCast((i * 19 + bi * 13) % 256);
            uq.* = @intCast((i * 23 + bi * 17) % 256);
        }
        for (&g.qh, &u.qh, 0..) |*gq, *uq, i| {
            gq.* = @intCast((i * 29 + bi * 7) % 256);
            uq.* = @intCast((i * 31 + bi * 11) % 256);
        }
    }

    var gate_rhs = try packMatmulRhsQ6_Kx4(allocator, gate_blocks, n, k, bpc);
    defer gate_rhs.deinit();
    var up_rhs = try packMatmulRhsQ6_Kx4(allocator, up_blocks, n, k, bpc);
    defer up_rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 37) % 257)) - 128)) * 0.03125;
    }
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();
    const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);

    const gate_ref = try allocator.alloc(f32, m * n);
    defer allocator.free(gate_ref);
    const up_ref = try allocator.alloc(f32, m * n);
    defer allocator.free(up_ref);
    const gate_pair = try allocator.alloc(f32, m * n);
    defer allocator.free(gate_pair);
    const up_pair = try allocator.alloc(f32, m * n);
    defer allocator.free(up_pair);

    matmulQ6_Kx4RhsTile(gate_ref, qlhs, &gate_rhs, n, 0, m, 0, n);
    matmulQ6_Kx4RhsTile(up_ref, qlhs, &up_rhs, n, 0, m, 0, n);
    matmulQ6_Kx4RhsPairTile(gate_pair, up_pair, qlhs, &gate_rhs, &up_rhs, n, 0, m, 0, n);

    try std.testing.expectEqualSlices(f32, gate_ref, gate_pair);
    try std.testing.expectEqualSlices(f32, up_ref, up_pair);
}

test "Q6_K compact-vs-packed cross-layout matmul is bit-identical at decode shapes (m=1,2,3)" {
    // The E4 decode-route proof (weights.linearSeqQ6_K m<4 gate, the Q5_K
    // gate's ride-along): the compact GGUF-native kernel family
    // (matmulQ6_KRhsRange over BlockQ6_K, 6.5625 bpw) and the byte-expanded
    // packed family (matmulQ6_Kx4RhsRange over BlockQ6_Kx4) must agree
    // BITWISE on the same blocks and the same quantized activations: both
    // compute the exact order-independent i32 total iacc = sum_g dot_g*scale_g
    // (Q6_K has no separate mins path; the -32 centering and the biased-dot
    // bsums correction reach the same integer) and both apply the identical
    // f32 epilogue association per block — acc + float(iacc)*(f16(d_w)*a.d) —
    // in ascending block order, so no tolerance is needed on any host.
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 16; // x4-multiple (packed-layout requirement); four column groups
    const bpc = k / qk_k_block_size;

    var prng = std.Random.DefaultPrng.init(0x36d8be51f04a97c2);
    const random = prng.random();

    const blocks = try allocator.alloc(BlockQ6_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks) |*b| fillRandomBlockQ6_K(b, random);

    var rhs_plain = try quantizedMatmulRhsQ6_KFromBlocks(allocator, k, n, blocks);
    defer rhs_plain.deinit();
    var rhs_packed = try packMatmulRhsQ6_Kx4(allocator, blocks, n, k, bpc);
    defer rhs_packed.deinit();

    inline for ([_]usize{ 1, 2, 3 }) |m| {
        const lhs_vals = try allocator.alloc(f32, m * k);
        defer allocator.free(lhs_vals);
        for (lhs_vals) |*v| v.* = (random.float(f32) - 0.5) * 8.0;
        var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
        defer dense.deinit();
        const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
        defer allocator.free(qlhs);

        const out_compact = try allocator.alloc(f32, m * n);
        defer allocator.free(out_compact);
        const out_packed = try allocator.alloc(f32, m * n);
        defer allocator.free(out_packed);
        matmulQ6_KRhsRange(out_compact, qlhs, &rhs_plain, m, n, 0, m);
        matmulQ6_Kx4RhsRange(out_packed, qlhs, &rhs_packed, m, n, 0, m);
        try std.testing.expectEqualSlices(f32, out_packed, out_compact);
    }
}

test "Q6_K column-outer matmul matches row-outer tile" {
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 8;
    const m = 17;
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ6_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.d = f32ToF16Bits(0.03 + 0.001 * @as(f32, @floatFromInt(bi % 5)));
        for (&b.scales, 0..) |*s, i| s.* = @intCast(@as(i32, @intCast((i * 5 + bi * 3) % 64)) - 32);
        for (&b.ql, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
    }
    var rhs = try quantizedMatmulRhsQ6_KFromBlocks(allocator, k, n, blocks);
    defer rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();
    const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);

    const out_tile = try allocator.alloc(f32, m * n);
    defer allocator.free(out_tile);
    const out_col = try allocator.alloc(f32, m * n);
    defer allocator.free(out_col);
    matmulQ6_KRhsTile(out_tile, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ6_KRhsCompactColOuter(out_col, qlhs, &rhs, n, 0, m, 0, n);
    for (out_tile, out_col) |t, c| try std.testing.expect(@abs(t - c) <= 1e-3 * @max(@as(f32, 1), @abs(t)));
}

test "Q6_K lane-packed Q8_Kx4 col-outer is bit-identical on the same activations" {
    // Same Q8_K activations, repacked into Q8_Kx4: the lane-packed kernel must
    // reproduce the per-row column-outer result exactly (integer dots are
    // order-independent; the f32 epilogue ops are identical).
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 9;
    const m = 16; // multiple of 4 so packRowsQ8_Kx4 applies (no padding)
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ6_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.d = f32ToF16Bits(0.03 + 0.001 * @as(f32, @floatFromInt(bi % 5)));
        for (&b.scales, 0..) |*s, i| s.* = @intCast(@as(i32, @intCast((i * 5 + bi * 3) % 64)) - 32);
        for (&b.ql, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
    }
    var rhs = try quantizedMatmulRhsQ6_KFromBlocks(allocator, k, n, blocks);
    defer rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();
    const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);
    const qlhs_x4 = try qm.packRowsQ8_Kx4(allocator, qlhs, m, k, bpc);
    defer allocator.free(qlhs_x4);

    const out_col = try allocator.alloc(f32, m * n);
    defer allocator.free(out_col);
    const out_x4 = try allocator.alloc(f32, m * n);
    defer allocator.free(out_x4);
    matmulQ6_KRhsCompactColOuter(out_col, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ6_KCompactQ8_Kx4ColOuter(out_x4, qlhs_x4, &rhs, n, m, 0, n);
    for (out_col, out_x4) |c, x| try std.testing.expectEqual(c, x);
}

test "Q6_K packRowsQ8_Kx4PaddedInto + lane-packed col-outer matches row-outer tile (tail)" {
    const allocator = std.testing.allocator;
    const k = 768;
    const n = 7;
    const m = 13; // not a multiple of 4
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ6_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.d = f32ToF16Bits(0.025 + 0.002 * @as(f32, @floatFromInt(bi % 4)));
        for (&b.scales, 0..) |*s, i| s.* = @intCast(@as(i32, @intCast((i * 7 + bi * 5) % 64)) - 32);
        for (&b.ql, 0..) |*q, i| q.* = @intCast((i * 29 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 17 + bi * 7) % 256);
    }
    var rhs = try quantizedMatmulRhsQ6_KFromBlocks(allocator, k, n, blocks);
    defer rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 13) % 241)) - 120);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();
    const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);
    const row_groups = (m + 3) / 4;
    const qlhs_x4 = try allocator.alloc(BlockQ8_Kx4, row_groups * bpc);
    defer allocator.free(qlhs_x4);
    qm.packRowsQ8_Kx4PaddedInto(qlhs_x4, qlhs, m, bpc);

    const out_tile = try allocator.alloc(f32, m * n);
    defer allocator.free(out_tile);
    const out_x4 = try allocator.alloc(f32, m * n);
    defer allocator.free(out_x4);
    matmulQ6_KRhsTile(out_tile, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ6_KCompactQ8_Kx4ColOuter(out_x4, qlhs_x4, &rhs, n, m, 0, n);
    for (out_tile, out_x4) |t, c| try std.testing.expect(@abs(t - c) <= 1e-3 * @max(@as(f32, 1), @abs(t)));
}

test "Q6_K col-outer kernels: split column ranges are bit-identical to full range" {
    // The batched-MoE phase chunking (exec/moe_chain.zig) hands these kernels
    // [c0, c1) column ranges at 256-column boundaries; a column's bits must
    // not depend on which range computed it, so one full-range call must
    // equal two chunked calls writing into the same-shaped buffer.
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 512; // two 256-column phase chunks
    const split = 256;
    const m = 5;
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ6_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.d = f32ToF16Bits(0.03 + 0.001 * @as(f32, @floatFromInt(bi % 5)));
        for (&b.scales, 0..) |*s, i| s.* = @intCast(@as(i32, @intCast((i * 5 + bi * 3) % 64)) - 32);
        for (&b.ql, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
    }
    var rhs = try quantizedMatmulRhsQ6_KFromBlocks(allocator, k, n, blocks);
    defer rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();
    const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);

    const out_full = try allocator.alloc(f32, m * n);
    defer allocator.free(out_full);
    const out_split = try allocator.alloc(f32, m * n);
    defer allocator.free(out_split);

    // Row-outer tile: the m < 4 arm of moeExpertTileDotRange.
    for ([_]usize{ 1, 3 }) |mt| {
        matmulQ6_KRhsTile(out_full, qlhs, &rhs, n, 0, mt, 0, n);
        matmulQ6_KRhsTile(out_split, qlhs, &rhs, n, 0, mt, 0, split);
        matmulQ6_KRhsTile(out_split, qlhs, &rhs, n, 0, mt, split, n);
        try std.testing.expectEqualSlices(f32, out_full[0 .. mt * n], out_split[0 .. mt * n]);
    }

    // Per-row column-outer: the m >= 4 arm.
    matmulQ6_KRhsCompactColOuter(out_full, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ6_KRhsCompactColOuter(out_split, qlhs, &rhs, n, 0, m, 0, split);
    matmulQ6_KRhsCompactColOuter(out_split, qlhs, &rhs, n, 0, m, split, n);
    try std.testing.expectEqualSlices(f32, out_full, out_split);

    // Lane-packed Q8_Kx4 column-outer: the phased-prefill arm (padded m=5 tail).
    const row_groups = (m + 3) / 4;
    const qlhs_x4 = try allocator.alloc(BlockQ8_Kx4, row_groups * bpc);
    defer allocator.free(qlhs_x4);
    qm.packRowsQ8_Kx4PaddedInto(qlhs_x4, qlhs, m, bpc);
    matmulQ6_KCompactQ8_Kx4ColOuter(out_full, qlhs_x4, &rhs, n, m, 0, n);
    matmulQ6_KCompactQ8_Kx4ColOuter(out_split, qlhs_x4, &rhs, n, m, 0, split);
    matmulQ6_KCompactQ8_Kx4ColOuter(out_split, qlhs_x4, &rhs, n, m, split, n);
    try std.testing.expectEqualSlices(f32, out_full, out_split);
}

// --- exact-parity tests: SIMD arms vs the scalar reference arm ---------------
//
// The VNNI / AVX2 / portable-widening arms promise BIT-IDENTICAL results to
// accumulateQ6_Kx4Scalar (equal i32 group dots + identical f32 association).
// On aarch64 hosts these tests execute the arms' portable primitive twins
// (validating the +32-bias/bsums-correction algebra and the f32 expression
// shape); on x86 hosts the same tests execute the real vpdpbusd /
// vpmaddubsw instructions (ReleaseFast/ReleaseSafe — Debug's self-hosted
// backend runs the twins via the has_llvm_asm gate).

const simd_tiers = [_]q6_k.Q6Kx4SimdTier{ .vnni, .avx2, .widen };

// Packed-weight domain: packMatmulRhsQ6_Kx4 stores q6KValue outputs, i.e.
// SIGNED centered values in [-32,31] (see the OPERAND SHAPE note in q6_k.zig).
fn fillRandomBlockQ6_Kx4(block: *BlockQ6_Kx4, random: std.Random) void {
    for (&block.d) |*d| d.* = f32ToF16Bits(0.25 + random.float(f32));
    for (&block.scales) |*s| s.* = @bitCast(random.int(u8)); // full i8 incl. -128
    for (&block.qs) |*q| q.* = @intCast(@as(i32, random.uintLessThan(u8, 64)) - 32);
}

// bsums[g] = Σ qs[g*16..][0..16] is part of the BlockQ8_K format contract
// (quantizeRowQ8_KInto always writes it); the biased SIMD arms rely on it.
fn setQ8KBsums(block: *BlockQ8_K) void {
    for (&block.bsums, 0..) |*sum, group| {
        var acc: i32 = 0;
        for (block.qs[group * 16 ..][0..16]) |q| acc += q;
        sum.* = @intCast(acc);
    }
}

// Activations over the FULL i8 range incl. -128: every Q6_Kx4 SIMD arm is
// activation-unrestricted (no sign-trick on this path), so the tests assert
// a wider domain than quantizeRowQ8_KInto's [-127,127] production values.
fn fillRandomBlockQ8_K(block: *BlockQ8_K, random: std.Random) void {
    block.d = 0.25 + random.float(f32);
    for (&block.qs) |*q| q.* = @bitCast(random.int(u8));
    setQ8KBsums(block);
}

fn randomVec4(random: std.Random) QKV4f32 {
    var vals: [4]f32 = undefined;
    for (&vals) |*v| v.* = (random.float(f32) - 0.5) * 64.0;
    return vals;
}

fn expectVec4BitEqual(expected: QKV4f32, got: QKV4f32) !void {
    const e: [4]f32 = expected;
    const g: [4]f32 = got;
    for (e, g) |ev, gv| {
        try std.testing.expectEqual(@as(u32, @bitCast(ev)), @as(u32, @bitCast(gv)));
    }
}

fn checkQ6Kx4Arms(lhs: *const BlockQ8_K, rhs: *const BlockQ6_Kx4, acc: QKV4f32) !void {
    const ref = q6_k.accumulateQ6_Kx4Scalar(lhs, rhs, acc);
    inline for (simd_tiers) |tier| {
        try expectVec4BitEqual(ref, q6_k.accumulateQ6_Kx4Simd(tier, lhs, rhs, acc));
    }
}

test "ggml_q6_kx4 SIMD arms match the scalar arm bit-exactly" {
    var prng = std.Random.DefaultPrng.init(0x6b1cf3a29d47e805);
    const random = prng.random();

    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        var lhs: BlockQ8_K = undefined;
        var rhs: BlockQ6_Kx4 = undefined;
        fillRandomBlockQ8_K(&lhs, random);
        fillRandomBlockQ6_Kx4(&rhs, random);
        try checkQ6Kx4Arms(&lhs, &rhs, randomVec4(random));
    }

    // Edge blocks: weight extremes (-32 / +31 / alternating) against
    // activation extremes (±127 and -128) under scale extremes (-128 / +127
    // / zero) — covers the bias-correction maxima (all-(-32)·all-(-128)·
    // (-128) peaks |iacc| ≈ 134M) and the all-zero-scales block.
    var lhs: BlockQ8_K = undefined;
    var rhs: BlockQ6_Kx4 = undefined;
    lhs.d = 1.0;
    for (&rhs.d) |*d| d.* = f32ToF16Bits(1.0);

    const edges = [_]struct { w: [2]i8, a: [2]i8, s: [2]i8 }{
        .{ .w = .{ -32, -32 }, .a = .{ -128, -128 }, .s = .{ -128, -128 } },
        .{ .w = .{ 31, 31 }, .a = .{ 127, 127 }, .s = .{ 127, 127 } },
        .{ .w = .{ -32, 31 }, .a = .{ 127, -128 }, .s = .{ -128, 127 } },
        .{ .w = .{ 31, -32 }, .a = .{ -127, 127 }, .s = .{ 127, -128 } },
        .{ .w = .{ -32, 31 }, .a = .{ -128, 127 }, .s = .{ 0, 0 } },
    };
    for (edges) |edge| {
        for (&rhs.qs, 0..) |*q, i| q.* = edge.w[i % 2];
        for (&rhs.scales, 0..) |*s, i| s.* = edge.s[i % 2];
        for (&lhs.qs, 0..) |*q, i| q.* = edge.a[i % 2];
        setQ8KBsums(&lhs);
        try checkQ6Kx4Arms(&lhs, &rhs, @splat(0));
    }

    // Zero-d block (the quantizer's all-zero escape): d bits = 0 on both sides.
    lhs.d = 0;
    for (&rhs.d) |*d| d.* = 0;
    try checkQ6Kx4Arms(&lhs, &rhs, randomVec4(random));
}

test "ggml_q6_kx4 SIMD rows arm matches per-row scalar accumulation" {
    var prng = std.Random.DefaultPrng.init(0x1d0c5e83b9f6a247);
    const random = prng.random();
    const blocks_per_row = 2;

    var iter: usize = 0;
    while (iter < 50) : (iter += 1) {
        var lhs_blocks: [q8_0_row_block * blocks_per_row]BlockQ8_K = undefined;
        for (&lhs_blocks) |*b| fillRandomBlockQ8_K(b, random);
        var rhs: BlockQ6_Kx4 = undefined;
        fillRandomBlockQ6_Kx4(&rhs, random);

        const block_index = iter % blocks_per_row;
        var acc0: [q8_0_row_block]QKV4f32 = undefined;
        for (&acc0) |*row| row.* = randomVec4(random);
        var want = acc0;
        for (&want, 0..) |*row, r| {
            row.* = q6_k.accumulateQ6_Kx4Scalar(&lhs_blocks[r * blocks_per_row + block_index], &rhs, row.*);
        }

        inline for (simd_tiers) |tier| {
            var got = acc0;
            q6_k.accumulateQ6_Kx4RowsSimd(tier, &lhs_blocks, 0, blocks_per_row, block_index, &rhs, &got);
            inline for (0..q8_0_row_block) |r| try expectVec4BitEqual(want[r], got[r]);
        }
    }
}

test "ggml_q6_kx4 SIMD pair arm matches two scalar accumulates" {
    var prng = std.Random.DefaultPrng.init(0x8842f0d15c3ab967);
    const random = prng.random();

    var iter: usize = 0;
    while (iter < 50) : (iter += 1) {
        var lhs: BlockQ8_K = undefined;
        var gate_rhs: BlockQ6_Kx4 = undefined;
        var up_rhs: BlockQ6_Kx4 = undefined;
        fillRandomBlockQ8_K(&lhs, random);
        fillRandomBlockQ6_Kx4(&gate_rhs, random);
        fillRandomBlockQ6_Kx4(&up_rhs, random);
        const gate_acc = randomVec4(random);
        const up_acc = randomVec4(random);

        const gate_ref = q6_k.accumulateQ6_Kx4Scalar(&lhs, &gate_rhs, gate_acc);
        const up_ref = q6_k.accumulateQ6_Kx4Scalar(&lhs, &up_rhs, up_acc);

        inline for (simd_tiers) |tier| {
            const pair = q6_k.accumulateQ6_Kx4PairSimd(tier, &lhs, &gate_rhs, &up_rhs, gate_acc, up_acc);
            try expectVec4BitEqual(gate_ref, pair.gate);
            try expectVec4BitEqual(up_ref, pair.up);
        }
    }
}

test "ggml_q6_kx4 matmul entry points match the scalar-arm reference" {
    // Dispatcher-level: production-shaped inputs (quantized Q8_K activations,
    // packed weights from real Q6_K blocks) through matmulQ6_Kx4RhsTile and
    // matmulQ6_Kx4RhsPairTile, with m covering the 4-row main loop AND the
    // row tail. On non-aarch64 the dispatched arms are bit-identical to the
    // scalar arm → exact compare; on aarch64 the sdot arm's f32 association
    // differs → tolerance.
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 8;
    const m = 6; // 1 full row block + 2 tail rows
    const bpc = k / qk_k_block_size;

    const gate_blocks = try allocator.alloc(BlockQ6_K, n * bpc);
    defer allocator.free(gate_blocks);
    const up_blocks = try allocator.alloc(BlockQ6_K, n * bpc);
    defer allocator.free(up_blocks);
    for (gate_blocks, up_blocks, 0..) |*g, *u, bi| {
        g.d = f32ToF16Bits(0.021 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        u.d = f32ToF16Bits(0.017 + 0.001 * @as(f32, @floatFromInt((bi + 3) % 5)));
        for (&g.scales, &u.scales, 0..) |*gs, *us, i| {
            gs.* = @intCast(@as(i32, @intCast((i * 7 + bi * 5) % 256)) - 128);
            us.* = @intCast(@as(i32, @intCast((i * 11 + bi * 3) % 256)) - 128);
        }
        for (&g.ql, &u.ql, 0..) |*gq, *uq, i| {
            gq.* = @intCast((i * 19 + bi * 13) % 256);
            uq.* = @intCast((i * 23 + bi * 17) % 256);
        }
        for (&g.qh, &u.qh, 0..) |*gq, *uq, i| {
            gq.* = @intCast((i * 29 + bi * 7) % 256);
            uq.* = @intCast((i * 31 + bi * 11) % 256);
        }
    }
    var gate_rhs = try packMatmulRhsQ6_Kx4(allocator, gate_blocks, n, k, bpc);
    defer gate_rhs.deinit();
    var up_rhs = try packMatmulRhsQ6_Kx4(allocator, up_blocks, n, k, bpc);
    defer up_rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 37) % 257)) - 128)) * 0.03125;
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();
    const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);

    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    q6_k.matmulQ6_Kx4RhsTile(got, qlhs, &gate_rhs, n, 0, m, 0, n);

    const ref = try allocator.alloc(f32, m * n);
    defer allocator.free(ref);
    for (0..m) |i| {
        var j: usize = 0;
        while (j < n) : (j += 4) {
            const rhs_group = gate_rhs.groupBlocks(j / 4);
            var acc: QKV4f32 = @splat(0);
            for (0..bpc) |block_index| {
                acc = q6_k.accumulateQ6_Kx4Scalar(&qlhs[i * bpc + block_index], &rhs_group[block_index], acc);
            }
            const vals: [4]f32 = acc;
            for (0..4) |c| ref[i * n + j + c] = vals[c];
        }
    }

    for (ref, got) |expected, actual| {
        if (comptime builtin.cpu.arch == .aarch64) {
            try std.testing.expectApproxEqAbs(expected, actual, 1e-2);
        } else {
            try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(actual)));
        }
    }

    // Paired gate/up entry point against two independent scalar-composed refs.
    const gate_got = try allocator.alloc(f32, m * n);
    defer allocator.free(gate_got);
    const up_got = try allocator.alloc(f32, m * n);
    defer allocator.free(up_got);
    q6_k.matmulQ6_Kx4RhsPairTile(gate_got, up_got, qlhs, &gate_rhs, &up_rhs, n, 0, m, 0, n);

    const up_ref = try allocator.alloc(f32, m * n);
    defer allocator.free(up_ref);
    for (0..m) |i| {
        var j: usize = 0;
        while (j < n) : (j += 4) {
            const rhs_group = up_rhs.groupBlocks(j / 4);
            var acc: QKV4f32 = @splat(0);
            for (0..bpc) |block_index| {
                acc = q6_k.accumulateQ6_Kx4Scalar(&qlhs[i * bpc + block_index], &rhs_group[block_index], acc);
            }
            const vals: [4]f32 = acc;
            for (0..4) |c| up_ref[i * n + j + c] = vals[c];
        }
    }

    for (0..m * n) |idx| {
        if (comptime builtin.cpu.arch == .aarch64) {
            try std.testing.expectApproxEqAbs(ref[idx], gate_got[idx], 1e-2);
            try std.testing.expectApproxEqAbs(up_ref[idx], up_got[idx], 1e-2);
        } else {
            try std.testing.expectEqual(@as(u32, @bitCast(ref[idx])), @as(u32, @bitCast(gate_got[idx])));
            try std.testing.expectEqual(@as(u32, @bitCast(up_ref[idx])), @as(u32, @bitCast(up_got[idx])));
        }
    }
}

// --- exact-parity tests: the MoE ColOuter 4-row lane dot + the row dot ------
//
// Both new SIMD surfaces promise BIT-IDENTICAL i32 dots (and, for the row
// dot, bit-identical f32 results) to their plain-scalar references on every
// host: the aarch64 dev machine executes the arms' portable primitive twins
// (validating the +32-bias/bsums-correction algebra), x86 ReleaseFast/Safe
// executes the real vpdpbusd / vpmaddubsw instructions.

// Interleave 4 plain Q8_K rows exactly like packRowsQ8_Kx4 (incl. the bsums
// interleave `bsums[(g/4)*16 + row*4 + g%4]` the biased arms rely on).
fn packBlockQ8_Kx4(rows: *const [4]BlockQ8_K) BlockQ8_Kx4 {
    var dst: BlockQ8_Kx4 = undefined;
    inline for (0..4) |row| dst.d[row] = rows[row].d;
    for (0..qk_k_block_size / 4) |feature_group| {
        inline for (0..4) |row| {
            inline for (0..4) |lane| {
                dst.qs[feature_group * 16 + row * 4 + lane] = rows[row].qs[feature_group * 4 + lane];
            }
        }
    }
    inline for (0..4) |row| {
        inline for (0..16) |group| {
            dst.bsums[(group / 4) * 16 + row * 4 + group % 4] = rows[row].bsums[group];
        }
    }
    return dst;
}

test "q6_k 4-row lane dot SIMD arms match the scalar reference" {
    var prng = std.Random.DefaultPrng.init(0x7e2a91c4d5f60b38);
    const random = prng.random();

    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        var rows: [4]BlockQ8_K = undefined;
        for (&rows) |*r| fillRandomBlockQ8_K(r, random);
        const a = packBlockQ8_Kx4(&rows);
        // Pre-unpacked weight-group domain: unpackQ6_KGroup emits centered
        // values in [-32,31].
        var wa: [16]i8 = undefined;
        for (&wa) |*w| w.* = @intCast(@as(i32, random.uintLessThan(u8, 64)) - 32);
        const wv: common.QKV16i8 = wa;
        const fg_base = (iter % 16) * 4;
        const ref: [4]i32 = q6_k.dot16Group4RowsQ8_Kx4Scalar(&a, fg_base, wv);
        inline for (simd_tiers) |tier| {
            try std.testing.expectEqual(ref, @as([4]i32, q6_k.dot16Group4RowsQ8_Kx4Simd(tier, &a, fg_base, wv)));
        }
    }

    // Edge blocks: weight extremes (-32 / +31 / zero) against activation
    // extremes incl. the out-of-domain -128 (the +32-bias arms stay exact:
    // pair sums <= 2*63*128 < 2^15) — covers the bias-correction maxima —
    // over every scale group's fg_base.
    const edge_a = [_][2]i8{ .{ -128, -128 }, .{ 127, 127 }, .{ 127, -128 }, .{ -127, 127 } };
    const edge_w = [_]i8{ -32, 31, 0 };
    var rows: [4]BlockQ8_K = undefined;
    for (edge_a) |pattern| {
        for (&rows, 0..) |*r, ri| {
            r.d = 1.0;
            for (&r.qs, 0..) |*q, i| q.* = pattern[(i + ri) % 2];
            setQ8KBsums(r);
        }
        const a = packBlockQ8_Kx4(&rows);
        for (edge_w) |wval| {
            const wv: common.QKV16i8 = @splat(wval);
            var sg: usize = 0;
            while (sg < 16) : (sg += 1) {
                const ref: [4]i32 = q6_k.dot16Group4RowsQ8_Kx4Scalar(&a, sg * 4, wv);
                inline for (simd_tiers) |tier| {
                    try std.testing.expectEqual(ref, @as([4]i32, q6_k.dot16Group4RowsQ8_Kx4Simd(tier, &a, sg * 4, wv)));
                }
            }
        }
    }
}

fn fillRandomBlockQ6_K(b: *BlockQ6_K, random: std.Random) void {
    b.d = f32ToF16Bits(0.25 + random.float(f32));
    for (&b.scales) |*s| s.* = @bitCast(random.int(u8)); // full i8 incl. -128
    for (&b.ql) |*q| q.* = random.int(u8);
    for (&b.qh) |*q| q.* = random.int(u8);
}

fn checkQ6RowDotArms(w: *const BlockQ6_K, a: *const BlockQ8_K) !void {
    const ref = q6_k.dotQ6_KQ8_KScalar(w, a);
    inline for (simd_tiers) |tier| {
        const got = q6_k.dotQ6_KQ8_KSimd(tier, w, a);
        try std.testing.expectEqual(@as(u32, @bitCast(ref)), @as(u32, @bitCast(got)));
    }
}

test "q6_k row dot SIMD arms match the scalar reference bit-exactly" {
    var prng = std.Random.DefaultPrng.init(0x40cbe17f5a9d2c63);
    const random = prng.random();

    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        var w: BlockQ6_K = undefined;
        var a: BlockQ8_K = undefined;
        fillRandomBlockQ6_K(&w, random);
        fillRandomBlockQ8_K(&a, random);
        try checkQ6RowDotArms(&w, &a);
    }

    // Edge blocks: raw-byte weight extremes (ql/qh all-0x00 -> w = -32,
    // all-0xff -> w = +31) x scale extremes (-128 / +127 / 0) x activation
    // extremes incl. -128 (bsums per contract) — the bias-correction maxima.
    const edge_a = [_][2]i8{ .{ -128, -128 }, .{ 127, 127 }, .{ 127, -128 }, .{ -127, 127 } };
    const edge_bytes = [_]u8{ 0x00, 0xff };
    const edge_scales = [_]i8{ -128, 127, 0 };
    var w: BlockQ6_K = undefined;
    var a: BlockQ8_K = undefined;
    w.d = f32ToF16Bits(1.0);
    a.d = 1.0;
    for (edge_a) |pattern| {
        for (&a.qs, 0..) |*q, i| q.* = pattern[i % 2];
        setQ8KBsums(&a);
        for (edge_bytes) |byte| {
            @memset(&w.ql, byte);
            @memset(&w.qh, byte);
            for (edge_scales) |s| {
                @memset(&w.scales, s);
                try checkQ6RowDotArms(&w, &a);
            }
        }
    }

    // Zero-d block (the quantizer's all-zero escape).
    w.d = 0;
    a.d = 0;
    try checkQ6RowDotArms(&w, &a);
}

test "q6_k row-outer tile matches the scalar row dot bit-exactly" {
    // Dispatcher-level: every dotQ6_KQ8_K arm (sdot / vpdpbusd / maddubs /
    // widen) computes the same i32 totals and the same f32 epilogue, so the
    // row-outer tile must match a dotQ6_KQ8_KScalar-composed reference
    // bit-exactly on EVERY host. n=3 exercises the paired-column loop and
    // the single-column tail.
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 3;
    const m = 2;
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ6_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.d = f32ToF16Bits(0.03 + 0.001 * @as(f32, @floatFromInt(bi % 5)));
        for (&b.scales, 0..) |*s, i| s.* = @intCast(@as(i32, @intCast((i * 5 + bi * 3) % 256)) - 128);
        for (&b.ql, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
    }
    var rhs = try quantizedMatmulRhsQ6_KFromBlocks(allocator, k, n, blocks);
    defer rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();
    const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);

    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    matmulQ6_KRhsTile(got, qlhs, &rhs, n, 0, m, 0, n);

    for (0..m) |i| {
        for (0..n) |j| {
            const col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            for (0..bpc) |bi| {
                acc += q6_k.dotQ6_KQ8_KScalar(&col[bi], &qlhs[i * bpc + bi]);
            }
            try std.testing.expectEqual(@as(u32, @bitCast(acc)), @as(u32, @bitCast(got[i * n + j])));
        }
    }
}
