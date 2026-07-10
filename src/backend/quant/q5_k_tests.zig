//! Behavioral tests for the Q5_K / Q5_Kx8 quantized matmul kernels in `q5_k.zig`:
//! x8-packed vs plain matmul parity, and the column-outer / lane-packed Q8_Kx4
//! kernels against the trusted row-outer tile (incl. padded-tail row groups).

const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("../../tensor.zig");
const qm = @import("../quant.zig");
const common = @import("common.zig");
const q5_k = @import("q5_k.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;

const BlockQ5_K = qm.BlockQ5_K;
const BlockQ8_K = qm.BlockQ8_K;
const BlockQ8_Kx4 = qm.BlockQ8_Kx4;
const f32ToF16Bits = common.f32ToF16Bits;
const fillQ8KPattern = qm.fillQ8KPattern;
const packRowsQ8_Kx4 = qm.packRowsQ8_Kx4;
const quantizedMatmulRhsQ5_KFromBlocks = qm.quantizedMatmulRhsQ5_KFromBlocks;
const qk_k_block_size = qm.qk_k_block_size;

const packMatmulRhsQ5_Kx8 = q5_k.packMatmulRhsQ5_Kx8;
const matmulQ5_Kx8RhsRange = q5_k.matmulQ5_Kx8RhsRange;
const matmulQ5_Kx8Q8_Kx4RhsRange = q5_k.matmulQ5_Kx8Q8_Kx4RhsRange;
const matmulQ5_KRhsRange = q5_k.matmulQ5_KRhsRange;
const matmulQ5_KRhsTile = q5_k.matmulQ5_KRhsTile;
const matmulQ5_KRhsCompactColOuter = q5_k.matmulQ5_KRhsCompactColOuter;
const matmulQ5_KCompactQ8_Kx4ColOuter = q5_k.matmulQ5_KCompactQ8_Kx4ColOuter;

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

test "ggml_q5_k x8 packed matmul matches plain q5_k matmul" {
    const allocator = std.testing.allocator;

    var q5: BlockQ5_K = undefined;
    fillQ5KPattern(&q5);
    var q8: BlockQ8_K = undefined;
    fillQ8KPattern(&q8);

    var lhs_blocks = [_]BlockQ8_K{ q8, q8, q8, q8 };
    var rhs_blocks = [_]BlockQ5_K{ q5, q5, q5, q5, q5, q5, q5, q5 };
    var rhs_plain = try quantizedMatmulRhsQ5_KFromBlocks(allocator, qk_k_block_size, 8, &rhs_blocks);
    defer rhs_plain.deinit();
    var rhs_packed = try packMatmulRhsQ5_Kx8(allocator, &rhs_blocks, 8, qk_k_block_size, 1);
    defer rhs_packed.deinit();

    var expected: [32]f32 = undefined;
    var actual: [32]f32 = undefined;
    matmulQ5_KRhsRange(&expected, &lhs_blocks, &rhs_plain, 4, 8, 0, 4);
    matmulQ5_Kx8RhsRange(&actual, &lhs_blocks, &rhs_packed, 4, 8, 0, 4);

    for (expected, actual) |ref, got| {
        try std.testing.expectApproxEqAbs(ref, got, 1e-3);
    }

    const lhs_x4 = try packRowsQ8_Kx4(allocator, &lhs_blocks, 4, qk_k_block_size, 1);
    defer allocator.free(lhs_x4);
    var actual_x4: [32]f32 = undefined;
    matmulQ5_Kx8Q8_Kx4RhsRange(&actual_x4, lhs_x4, &rhs_packed, 4, 8, 0, 4);
    for (expected, actual_x4) |ref, got| {
        try std.testing.expectApproxEqAbs(ref, got, 1e-3);
    }
}

test "Q5_K compact-vs-packed cross-layout matmul is bit-identical at decode shapes (m=1,2,3)" {
    // The P0 decode-route proof (weights.linearSeqQ5_K m<4 gate): the compact
    // GGUF-native kernel family (matmulQ5_KRhsRange over BlockQ5_K, 5.5 bpw)
    // and the byte-expanded packed family (matmulQ5_Kx8RhsRange over
    // BlockQ5_Kx8, 8.625 bpw) must agree BITWISE on the same blocks and the
    // same quantized activations: both compute exact order-independent i32
    // subblock sums (iscale = sum scale*dot, imin = sum min*bsum) and both
    // apply the identical f32 epilogue association per block —
    // float(iscale)*(d_w*a.d) - float(imin)*(dmin_w*a.d), accumulated in
    // ascending block order — so no tolerance is needed on any host.
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 16; // x8-multiple (packed-layout requirement); two column groups
    const bpc = k / qk_k_block_size;

    var prng = std.Random.DefaultPrng.init(0x7a52c90de13fb864);
    const random = prng.random();

    const blocks = try allocator.alloc(BlockQ5_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks) |*b| fillRandomBlockQ5_K(b, random);

    var rhs_plain = try quantizedMatmulRhsQ5_KFromBlocks(allocator, k, n, blocks);
    defer rhs_plain.deinit();
    var rhs_packed = try packMatmulRhsQ5_Kx8(allocator, blocks, n, k, bpc);
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
        matmulQ5_KRhsRange(out_compact, qlhs, &rhs_plain, m, n, 0, m);
        matmulQ5_Kx8RhsRange(out_packed, qlhs, &rhs_packed, m, n, 0, m);
        try std.testing.expectEqualSlices(f32, out_packed, out_compact);
    }
}

test "Q5_K column-outer matmul matches row-outer tile" {
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 8;
    const m = 17;
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ5_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = f32ToF16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f32ToF16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    var rhs = try quantizedMatmulRhsQ5_KFromBlocks(allocator, k, n, blocks);
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
    matmulQ5_KRhsTile(out_tile, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ5_KRhsCompactColOuter(out_col, qlhs, &rhs, n, 0, m, 0, n);
    for (out_tile, out_col) |t, c| try std.testing.expect(@abs(t - c) <= 1e-3 * @max(@as(f32, 1), @abs(t)));
}

test "Q5_K lane-packed Q8_Kx4 col-outer is bit-identical on the same activations" {
    // Same quantized Q8_K activations, just repacked into the 4-row-interleaved
    // Q8_Kx4 layout: the lane-packed kernel must reproduce the per-row column-outer
    // result exactly (integer dots are order-independent; epilogue ops are identical).
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 9; // non-multiple of any col tile, exercises the column loop tail
    const m = 16; // multiple of 4 so packRowsQ8_Kx4 applies (no padding here)
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ5_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = f32ToF16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f32ToF16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    var rhs = try quantizedMatmulRhsQ5_KFromBlocks(allocator, k, n, blocks);
    defer rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();

    const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);
    const qlhs_x4 = try packRowsQ8_Kx4(allocator, qlhs, m, k, bpc);
    defer allocator.free(qlhs_x4);

    const out_col = try allocator.alloc(f32, m * n);
    defer allocator.free(out_col);
    const out_x4 = try allocator.alloc(f32, m * n);
    defer allocator.free(out_x4);
    matmulQ5_KRhsCompactColOuter(out_col, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ5_KCompactQ8_Kx4ColOuter(out_x4, qlhs_x4, &rhs, n, m, 0, n);
    for (out_col, out_x4) |c, x| try std.testing.expectEqual(c, x);
}

test "Q5_K lane-packed Q8_Kx4 col-outer matches row-outer tile with padded tail" {
    // m not a multiple of 4: quantizeRowsQ8_Kx4PaddedInto zero-pads the tail group,
    // and the kernel must still match the trusted row-outer tile over the real rows.
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 8;
    const m = 17;
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ5_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = f32ToF16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f32ToF16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    var rhs = try quantizedMatmulRhsQ5_KFromBlocks(allocator, k, n, blocks);
    defer rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();

    const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);
    const row_groups = (m + 3) / 4;
    const qlhs_x4 = try allocator.alloc(BlockQ8_Kx4, row_groups * bpc);
    defer allocator.free(qlhs_x4);
    try qm.quantizeRowsQ8_Kx4PaddedInto(qlhs_x4, &dense);

    const out_tile = try allocator.alloc(f32, m * n);
    defer allocator.free(out_tile);
    const out_x4 = try allocator.alloc(f32, m * n);
    defer allocator.free(out_x4);
    matmulQ5_KRhsTile(out_tile, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ5_KCompactQ8_Kx4ColOuter(out_x4, qlhs_x4, &rhs, n, m, 0, n);
    for (out_tile, out_x4) |t, c| try std.testing.expect(@abs(t - c) <= 1e-3 * @max(@as(f32, 1), @abs(t)));
}

test "Q5_K packRowsQ8_Kx4PaddedInto + lane-packed col-outer matches row-outer tile (tail)" {
    // Exercises the allocation-free pack helper the MoE prefill path uses: pack the
    // per-row Q8_K into Q8_Kx4 with a padded tail, then run the lane-packed kernel.
    const allocator = std.testing.allocator;
    const k = 768;
    const n = 7;
    const m = 13; // not a multiple of 4
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ5_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = f32ToF16Bits(0.04 + 0.002 * @as(f32, @floatFromInt(bi % 5)));
        b.dm[1] = f32ToF16Bits(0.015);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 11 + bi * 5) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 17 + bi * 7) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 29 + bi * 3) % 256);
    }
    var rhs = try quantizedMatmulRhsQ5_KFromBlocks(allocator, k, n, blocks);
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
    matmulQ5_KRhsTile(out_tile, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ5_KCompactQ8_Kx4ColOuter(out_x4, qlhs_x4, &rhs, n, m, 0, n);
    for (out_tile, out_x4) |t, c| try std.testing.expect(@abs(t - c) <= 1e-3 * @max(@as(f32, 1), @abs(t)));
}

test "Q5_K col-outer kernels: split column ranges are bit-identical to full range" {
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
    const blocks = try allocator.alloc(BlockQ5_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = f32ToF16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f32ToF16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    var rhs = try quantizedMatmulRhsQ5_KFromBlocks(allocator, k, n, blocks);
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
        matmulQ5_KRhsTile(out_full, qlhs, &rhs, n, 0, mt, 0, n);
        matmulQ5_KRhsTile(out_split, qlhs, &rhs, n, 0, mt, 0, split);
        matmulQ5_KRhsTile(out_split, qlhs, &rhs, n, 0, mt, split, n);
        try std.testing.expectEqualSlices(f32, out_full[0 .. mt * n], out_split[0 .. mt * n]);
    }

    // Per-row column-outer: the m >= 4 arm.
    matmulQ5_KRhsCompactColOuter(out_full, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ5_KRhsCompactColOuter(out_split, qlhs, &rhs, n, 0, m, 0, split);
    matmulQ5_KRhsCompactColOuter(out_split, qlhs, &rhs, n, 0, m, split, n);
    try std.testing.expectEqualSlices(f32, out_full, out_split);

    // Lane-packed Q8_Kx4 column-outer: the phased-prefill arm (padded m=5 tail).
    const row_groups = (m + 3) / 4;
    const qlhs_x4 = try allocator.alloc(BlockQ8_Kx4, row_groups * bpc);
    defer allocator.free(qlhs_x4);
    qm.packRowsQ8_Kx4PaddedInto(qlhs_x4, qlhs, m, bpc);
    matmulQ5_KCompactQ8_Kx4ColOuter(out_full, qlhs_x4, &rhs, n, m, 0, n);
    matmulQ5_KCompactQ8_Kx4ColOuter(out_split, qlhs_x4, &rhs, n, m, 0, split);
    matmulQ5_KCompactQ8_Kx4ColOuter(out_split, qlhs_x4, &rhs, n, m, split, n);
    try std.testing.expectEqualSlices(f32, out_full, out_split);
}

// --- exact-parity tests: SIMD arms vs the scalar reference arms --------------
//
// The VNNI / AVX2 / portable-widening arms (and the sdot arm) promise
// BIT-IDENTICAL results to accumulateQ5_Kx8Scalar /
// accumulateQ5_Kx8Q8_Kx4Scalar: the i32 subblock sums are exact in every arm
// (order-independent integer adds) and the f32 epilogue expressions are
// element-for-element the same. On aarch64 hosts the x86 arms execute their
// portable primitive twins (validating the u8·i8 algebra and the epilogue
// shape) while the sdot arm executes real sdot; on x86 hosts the same tests
// execute the real vpdpbusd / vpmaddubsw instructions.

const QKV4f32 = common.QKV4f32;

fn fillRandomBlockQ5_Kx8(block: *qm.BlockQ5_Kx8, random: std.Random) void {
    for (&block.d) |*d| d.* = f32ToF16Bits(0.25 + random.float(f32));
    for (&block.dmin) |*d| d.* = f32ToF16Bits(0.05 + 0.5 * random.float(f32));
    // packMatmulRhsQ5_Kx8 stores getScaleMinK4 outputs (6-bit) and q5KValue
    // outputs (5-bit) — the valid packed domain.
    for (&block.scales) |*s| s.* = random.uintLessThan(u8, 64);
    for (&block.mins) |*m| m.* = random.uintLessThan(u8, 64);
    for (&block.qs) |*q| q.* = @intCast(random.uintLessThan(u8, 32));
}

fn fillRandomBlockQ8_KConsistent(block: *BlockQ8_K, random: std.Random) void {
    block.d = 0.25 + random.float(f32);
    // quantizeRowQ8_KInto domain: qs in [-127,127], bsums = 16-wide group sums.
    for (&block.qs) |*q| q.* = @intCast(@as(i32, random.uintLessThan(u8, 255)) - 127);
    recomputeBsums(block);
}

fn recomputeBsums(block: *BlockQ8_K) void {
    for (&block.bsums, 0..) |*b, g| {
        var s: i32 = 0;
        for (block.qs[g * 16 ..][0..16]) |q| s += q;
        b.* = @intCast(s);
    }
}

// Interleave 4 plain Q8_K rows exactly like packRowsQ8_Kx4.
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
        inline for (0..16) |subblock| {
            dst.bsums[(subblock / 4) * 16 + row * 4 + subblock % 4] = rows[row].bsums[subblock];
        }
    }
    return dst;
}

fn randomAcc2(random: std.Random) [2]QKV4f32 {
    var acc: [2]QKV4f32 = undefined;
    for (&acc) |*half| {
        var vals: [4]f32 = undefined;
        for (&vals) |*v| v.* = (random.float(f32) - 0.5) * 64.0;
        half.* = vals;
    }
    return acc;
}

fn expectAcc2BitEqual(expected: [2]QKV4f32, got: [2]QKV4f32) !void {
    inline for (0..2) |h| {
        const e: [4]f32 = expected[h];
        const g: [4]f32 = got[h];
        for (e, g) |ev, gv| {
            try std.testing.expectEqual(@as(u32, @bitCast(ev)), @as(u32, @bitCast(gv)));
        }
    }
}

fn expectAcc4x2BitEqual(expected: [4][2]QKV4f32, got: [4][2]QKV4f32) !void {
    inline for (0..4) |r| try expectAcc2BitEqual(expected[r], got[r]);
}

// Run all plain-LHS arms against the scalar arm for one (lhs, rhs, acc) triple.
fn checkPlainArms(lhs: *const BlockQ8_K, rhs: *const qm.BlockQ5_Kx8, acc0: [2]QKV4f32) !void {
    var ref = acc0;
    q5_k.accumulateQ5_Kx8Scalar(lhs, rhs, &ref);

    var got = acc0;
    q5_k.accumulateQ5_Kx8Vnni(lhs, rhs, &got);
    try expectAcc2BitEqual(ref, got);

    got = acc0;
    q5_k.accumulateQ5_Kx8Avx2(lhs, rhs, &got);
    try expectAcc2BitEqual(ref, got);

    got = acc0;
    q5_k.accumulateQ5_Kx8Widen(lhs, rhs, &got);
    try expectAcc2BitEqual(ref, got);
}

// Run all Q8_Kx4 packed-LHS arms (incl. the sdot arm — bit-exact on every
// host) against the scalar reference.
fn checkPackedArms(lhs: *const BlockQ8_Kx4, rhs: *const qm.BlockQ5_Kx8, acc0: [4][2]QKV4f32) !void {
    var ref = acc0;
    q5_k.accumulateQ5_Kx8Q8_Kx4Scalar(lhs, rhs, &ref);

    var got = acc0;
    q5_k.accumulateQ5_Kx8Q8_Kx4Vnni(lhs, rhs, &got);
    try expectAcc4x2BitEqual(ref, got);

    got = acc0;
    q5_k.accumulateQ5_Kx8Q8_Kx4Avx2(lhs, rhs, &got);
    try expectAcc4x2BitEqual(ref, got);

    got = acc0;
    q5_k.accumulateQ5_Kx8Q8_Kx4Widen(lhs, rhs, &got);
    try expectAcc4x2BitEqual(ref, got);

    got = acc0;
    q5_k.accumulateQ5_Kx8Q8_Kx4Sdot(lhs, rhs, &got);
    try expectAcc4x2BitEqual(ref, got);
}

// Edge activation patterns: extremes of the quantizeRowQ8_KInto domain plus
// the out-of-domain -128 (the u8·i8 arms stay exact there: maddubs pair sums
// are bounded by 2·31·128 = 7936 < 2^15).
const edge_activations = [_][2]i8{
    .{ 127, -127 },
    .{ -127, -127 },
    .{ 127, 127 },
    .{ -128, -128 },
};

fn fillEdgeBlockQ5_Kx8(block: *qm.BlockQ5_Kx8, qs_value: u8, scale_value: u8) void {
    for (&block.d) |*d| d.* = f32ToF16Bits(1.0);
    for (&block.dmin) |*d| d.* = f32ToF16Bits(1.0);
    for (&block.scales) |*s| s.* = scale_value;
    for (&block.mins) |*m| m.* = scale_value;
    for (&block.qs) |*q| q.* = @intCast(qs_value);
}

test "ggml_q5_kx8 plain-lhs SIMD arms match the scalar arm bit-exactly" {
    var prng = std.Random.DefaultPrng.init(0x3d1f9a5b7c42e680);
    const random = prng.random();

    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        var lhs: BlockQ8_K = undefined;
        var rhs: qm.BlockQ5_Kx8 = undefined;
        fillRandomBlockQ8_KConsistent(&lhs, random);
        fillRandomBlockQ5_Kx8(&rhs, random);
        try checkPlainArms(&lhs, &rhs, randomAcc2(random));
    }

    // Edge blocks: max 5-bit weights / max scales, zero scales, zero weights;
    // activations pinned to their extremes.
    const acc_zero: [2]QKV4f32 = .{ @splat(0), @splat(0) };
    var lhs: BlockQ8_K = undefined;
    var rhs: qm.BlockQ5_Kx8 = undefined;
    lhs.d = 1.0;
    for (edge_activations) |pattern| {
        for (&lhs.qs, 0..) |*q, i| q.* = pattern[i % 2];
        recomputeBsums(&lhs);
        fillEdgeBlockQ5_Kx8(&rhs, 31, 63);
        try checkPlainArms(&lhs, &rhs, acc_zero);
        fillEdgeBlockQ5_Kx8(&rhs, 31, 0);
        try checkPlainArms(&lhs, &rhs, acc_zero);
        fillEdgeBlockQ5_Kx8(&rhs, 0, 63);
        try checkPlainArms(&lhs, &rhs, acc_zero);
    }
}

test "ggml_q5_kx8 packed Q8_Kx4 SIMD arms match the scalar arm bit-exactly" {
    var prng = std.Random.DefaultPrng.init(0x91c4e2a06f5d3b78);
    const random = prng.random();

    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        var rows: [4]BlockQ8_K = undefined;
        for (&rows) |*r| fillRandomBlockQ8_KConsistent(r, random);
        const lhs = packBlockQ8_Kx4(&rows);
        var rhs: qm.BlockQ5_Kx8 = undefined;
        fillRandomBlockQ5_Kx8(&rhs, random);
        const acc0: [4][2]QKV4f32 = .{
            randomAcc2(random),
            randomAcc2(random),
            randomAcc2(random),
            randomAcc2(random),
        };
        try checkPackedArms(&lhs, &rhs, acc0);
    }

    // Edge blocks, as in the plain-lhs test.
    const acc_zero: [4][2]QKV4f32 = .{
        .{ @splat(0), @splat(0) },
        .{ @splat(0), @splat(0) },
        .{ @splat(0), @splat(0) },
        .{ @splat(0), @splat(0) },
    };
    var rows: [4]BlockQ8_K = undefined;
    var rhs: qm.BlockQ5_Kx8 = undefined;
    for (edge_activations) |pattern| {
        for (&rows, 0..) |*r, ri| {
            r.d = 1.0;
            for (&r.qs, 0..) |*q, i| q.* = pattern[(i + ri) % 2];
            recomputeBsums(r);
        }
        const lhs = packBlockQ8_Kx4(&rows);
        fillEdgeBlockQ5_Kx8(&rhs, 31, 63);
        try checkPackedArms(&lhs, &rhs, acc_zero);
        fillEdgeBlockQ5_Kx8(&rhs, 31, 0);
        try checkPackedArms(&lhs, &rhs, acc_zero);
        fillEdgeBlockQ5_Kx8(&rhs, 0, 63);
        try checkPackedArms(&lhs, &rhs, acc_zero);
    }
}

test "ggml_q5_kx8 matmul entry points match the scalar-arm reference bit-exactly" {
    // End-to-end over the public x8 entry points with production-shaped inputs
    // (randomized valid Q5_K encodings packed by packMatmulRhsQ5_Kx8, real
    // quantized activations). m=29 routes through the 16-row block, the 8- and
    // 4-row tails, AND the single-row loop of matmulQ5_Kx8RhsTile; m=8 covers
    // the Q8_Kx4 packed path. Every dispatcher arm (sdot / vpdpbusd / maddubs
    // / widen / scalar) shares exact i32 sums and the same f32 epilogue, so
    // the comparison is bit-exact on every host.
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 16;
    const m = 29;
    const bpc = k / qk_k_block_size;

    const blocks = try allocator.alloc(BlockQ5_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = f32ToF16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f32ToF16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    var rhs_packed = try packMatmulRhsQ5_Kx8(allocator, blocks, n, k, bpc);
    defer rhs_packed.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();
    const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);

    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    matmulQ5_Kx8RhsRange(got, qlhs, &rhs_packed, m, n, 0, m);

    for (0..m) |i| {
        var j: usize = 0;
        while (j < n) : (j += 8) {
            const rhs_group = rhs_packed.groupBlocks(j / 8);
            var acc: [2]QKV4f32 = .{ @splat(0), @splat(0) };
            for (0..bpc) |bi| {
                q5_k.accumulateQ5_Kx8Scalar(&qlhs[i * bpc + bi], &rhs_group[bi], &acc);
            }
            inline for (0..2) |h| {
                const vals: [4]f32 = acc[h];
                for (0..4) |c| {
                    const expected = vals[c];
                    const actual = got[i * n + j + h * 4 + c];
                    try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(actual)));
                }
            }
        }
    }

    // Q8_Kx4 packed-LHS entry point over the first 8 rows.
    const m4 = 8;
    const qlhs_x4 = try packRowsQ8_Kx4(allocator, qlhs[0 .. m4 * bpc], m4, k, bpc);
    defer allocator.free(qlhs_x4);
    const got_x4 = try allocator.alloc(f32, m4 * n);
    defer allocator.free(got_x4);
    matmulQ5_Kx8Q8_Kx4RhsRange(got_x4, qlhs_x4, &rhs_packed, m4, n, 0, m4);

    var row_group: usize = 0;
    while (row_group < m4 / 4) : (row_group += 1) {
        var j: usize = 0;
        while (j < n) : (j += 8) {
            const rhs_group = rhs_packed.groupBlocks(j / 8);
            var acc: [4][2]QKV4f32 = .{
                .{ @splat(0), @splat(0) },
                .{ @splat(0), @splat(0) },
                .{ @splat(0), @splat(0) },
                .{ @splat(0), @splat(0) },
            };
            for (0..bpc) |bi| {
                q5_k.accumulateQ5_Kx8Q8_Kx4Scalar(&qlhs_x4[row_group * bpc + bi], &rhs_group[bi], &acc);
            }
            inline for (0..4) |r| {
                inline for (0..2) |h| {
                    const vals: [4]f32 = acc[r][h];
                    for (0..4) |c| {
                        const expected = vals[c];
                        const actual = got_x4[(row_group * 4 + r) * n + j + h * 4 + c];
                        try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(actual)));
                    }
                }
            }
        }
    }
}

// --- exact-parity tests: the MoE ColOuter 4-row lane dot + the row dot ------
//
// Both new SIMD surfaces promise BIT-IDENTICAL i32 dots (and, for the row
// dot, bit-identical f32 results) to their plain-scalar references on every
// host: the aarch64 dev machine executes the arms' portable primitive twins,
// x86 ReleaseFast/Safe executes the real vpdpbusd / vpmaddubsw instructions.

const dot_tiers = [_]q5_k.X86DotTier{ .vnni, .avx2, .widen };

test "q5_k 4-row lane dot SIMD arms match the scalar reference" {
    var prng = std.Random.DefaultPrng.init(0x5c4d3e2f1a0b9887);
    const random = prng.random();

    var iter: usize = 0;
    while (iter < 100) : (iter += 1) {
        var rows: [4]BlockQ8_K = undefined;
        for (&rows) |*r| fillRandomBlockQ8_KConsistent(r, random);
        const a = packBlockQ8_Kx4(&rows);
        // Pre-unpacked sub-block domain: unpackQ5_KSubblock emits unsigned
        // 5-bit values in [0,31].
        var wa: [2][16]i8 = undefined;
        for (&wa) |*half| {
            for (half) |*w| w.* = @intCast(random.uintLessThan(u8, 32));
        }
        const wv: [2]common.QKV16i8 = .{ wa[0], wa[1] };
        inline for (0..8) |subblock| {
            const ref: [4]i32 = q5_k.dot4RowsSubblockQ8_Kx4Scalar(&a, subblock, wv);
            inline for (dot_tiers) |tier| {
                try std.testing.expectEqual(ref, @as([4]i32, q5_k.dot4RowsSubblockQ8_Kx4Simd(tier, &a, subblock, wv)));
            }
        }
    }

    // Edge blocks: weight extremes (31 / 0) against activation extremes incl.
    // the out-of-domain -128 (the u8*i8 arms stay exact: pair sums are
    // bounded by 2*31*128 = 7936 < 2^15).
    var rows: [4]BlockQ8_K = undefined;
    const edge_w = [_]i8{ 31, 0 };
    for (edge_activations) |pattern| {
        for (&rows, 0..) |*r, ri| {
            r.d = 1.0;
            for (&r.qs, 0..) |*q, i| q.* = pattern[(i + ri) % 2];
            recomputeBsums(r);
        }
        const a = packBlockQ8_Kx4(&rows);
        for (edge_w) |wval| {
            const wv: [2]common.QKV16i8 = .{ @splat(wval), @splat(wval) };
            inline for (0..8) |subblock| {
                const ref: [4]i32 = q5_k.dot4RowsSubblockQ8_Kx4Scalar(&a, subblock, wv);
                inline for (dot_tiers) |tier| {
                    try std.testing.expectEqual(ref, @as([4]i32, q5_k.dot4RowsSubblockQ8_Kx4Simd(tier, &a, subblock, wv)));
                }
            }
        }
    }
}

fn fillRandomBlockQ5_K(b: *BlockQ5_K, random: std.Random) void {
    b.dm = .{ f32ToF16Bits(0.25 + random.float(f32)), f32ToF16Bits(0.05 + 0.5 * random.float(f32)) };
    for (&b.scales) |*s| s.* = random.int(u8);
    for (&b.qs) |*q| q.* = random.int(u8);
    for (&b.qh) |*q| q.* = random.int(u8);
}

fn checkQ5RowDotArms(w: *const BlockQ5_K, a: *const BlockQ8_K) !void {
    const ref = q5_k.dotQ5_KQ8_KScalar(w, a);
    inline for (dot_tiers) |tier| {
        const got = q5_k.dotQ5_KQ8_KSimd(tier, w, a);
        try std.testing.expectEqual(@as(u32, @bitCast(ref)), @as(u32, @bitCast(got)));
    }
}

test "q5_k row dot SIMD arms match the scalar reference bit-exactly" {
    var prng = std.Random.DefaultPrng.init(0x2b8f61ad0c47e935);
    const random = prng.random();

    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        var w: BlockQ5_K = undefined;
        var a: BlockQ8_K = undefined;
        fillRandomBlockQ5_K(&w, random);
        fillRandomBlockQ8_KConsistent(&a, random);
        try checkQ5RowDotArms(&w, &a);
    }

    // Edge blocks: raw-byte weight extremes (qs/qh all-0xff -> w = 31,
    // all-0x00 -> w = 0) x raw scale-byte extremes x activation extremes
    // incl. -128 (bsums per contract).
    const edge_bytes = [_]u8{ 0x00, 0xff };
    const edge_scale_bytes = [_]u8{ 0x00, 0xff };
    var w: BlockQ5_K = undefined;
    var a: BlockQ8_K = undefined;
    w.dm = .{ f32ToF16Bits(1.0), f32ToF16Bits(1.0) };
    a.d = 1.0;
    for (edge_activations) |pattern| {
        for (&a.qs, 0..) |*q, i| q.* = pattern[i % 2];
        recomputeBsums(&a);
        for (edge_bytes) |byte| {
            @memset(&w.qs, byte);
            @memset(&w.qh, byte);
            for (edge_scale_bytes) |sb| {
                @memset(&w.scales, sb);
                try checkQ5RowDotArms(&w, &a);
            }
        }
    }

    // Zero-d block (the quantizer's all-zero escape).
    w.dm = .{ 0, 0 };
    a.d = 0;
    try checkQ5RowDotArms(&w, &a);
}

test "q5_k row-outer tile matches the scalar row dot bit-exactly" {
    // Dispatcher-level: every dotQ5_KQ8_K arm (sdot / vpdpbusd / maddubs /
    // widen) computes the same i32 totals and the same f32 epilogue, so the
    // row-outer tile must match a dotQ5_KQ8_KScalar-composed reference
    // bit-exactly on EVERY host. n=3 exercises the paired-column loop and
    // the single-column tail.
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 3;
    const m = 2;
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ5_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = f32ToF16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f32ToF16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    var rhs = try quantizedMatmulRhsQ5_KFromBlocks(allocator, k, n, blocks);
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
    matmulQ5_KRhsTile(got, qlhs, &rhs, n, 0, m, 0, n);

    for (0..m) |i| {
        for (0..n) |j| {
            const col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            for (0..bpc) |bi| {
                acc += q5_k.dotQ5_KQ8_KScalar(&col[bi], &qlhs[i * bpc + bi]);
            }
            try std.testing.expectEqual(@as(u32, @bitCast(acc)), @as(u32, @bitCast(got[i * n + j])));
        }
    }
}
