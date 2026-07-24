//! Behavioral tests for the hot Q4_K quantized matmul kernels (`q4_k.zig`):
//! packed x4/x8/x2-mmla matmul parity vs the plain row-outer tile, padded
//! Q8_Kx4 lhs parity, randomized scalar-reference parity, and the column-outer
//! / lane-packed col-outer kernels matching the trusted row-outer tile.
const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("../../tensor.zig");
const qm = @import("../quant.zig");
const common = @import("common.zig");
const q4_k = @import("q4_k.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;

// Shared symbols defined in quant.zig / common.zig, aliased here so the moved
// test bodies compile unchanged.
const BlockQ4_K = qm.BlockQ4_K;
const BlockQ8_K = qm.BlockQ8_K;
const BlockQ8_Kx2Mmla = qm.BlockQ8_Kx2Mmla;
const BlockQ8_Kx4 = qm.BlockQ8_Kx4;
const dequantizeBlockQ8_KInto = qm.dequantizeBlockQ8_KInto;
const dotDense = common.dotDense;
const f16BitsToF32 = common.f16BitsToF32;
const f32ToF16Bits = common.f32ToF16Bits;
const getScaleMinK4 = qm.getScaleMinK4;
const qk_k_block_size = qm.qk_k_block_size;
const quantizeRowsQ8_K = qm.quantizeRowsQ8_K;
const quantizeRowsQ8_Kx2MmlaInto = qm.quantizeRowsQ8_Kx2MmlaInto;
const quantizeRowsQ8_Kx4PaddedInto = qm.quantizeRowsQ8_Kx4PaddedInto;
const quantizedMatmulRhsQ4_KFromBlocks = qm.quantizedMatmulRhsQ4_KFromBlocks;

// Public symbols of q4_k.zig used by the moved tests.
const dequantizeBlockQ4_KInto = q4_k.dequantizeBlockQ4_KInto;
const matmulQ4_KRhsRange = q4_k.matmulQ4_KRhsRange;
const matmulQ4_KRhsTile = q4_k.matmulQ4_KRhsTile;
const matmulQ4_KRhsCompactColOuter = q4_k.matmulQ4_KRhsCompactColOuter;
const matmulQ4_KCompactQ8_Kx4ColOuter = q4_k.matmulQ4_KCompactQ8_Kx4ColOuter;
const matmulQ4_Kx4RhsRange = q4_k.matmulQ4_Kx4RhsRange;
const matmulQ4_Kx8RhsRange = q4_k.matmulQ4_Kx8RhsRange;
const matmulQ4_Kx8Q8_Kx4RhsRange = q4_k.matmulQ4_Kx8Q8_Kx4RhsRange;
const matmulQ4_Kx2MmlaRhsRange = q4_k.matmulQ4_Kx2MmlaRhsRange;
const matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsRange = q4_k.matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsRange;
const packMatmulRhsQ4_Kx4 = q4_k.packMatmulRhsQ4_Kx4;
const packMatmulRhsQ4_Kx8 = q4_k.packMatmulRhsQ4_Kx8;
const packMatmulRhsQ4_Kx2Mmla = q4_k.packMatmulRhsQ4_Kx2Mmla;

// Scalar reference replica of dotQ4_KQ8_K: same nibble extraction, same i32
// integer accumulation (deferred scale*acc / min*bsum reduction), same f32
// expression order — so the comparison below is BIT-EXACT on every target
// (the integer dot is exact on all paths; Zig never contracts the identical
// f32 ops). Used by the randomized parity test and mirrored in
// src/x86dot_check.zig for the cross-ISA attestation runs.
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

fn expectPackedApprox(ref: f32, got: f32) !void {
    const diff = @abs(ref - got);
    const scale = @max(@abs(ref), @as(f32, 1));
    try std.testing.expect(diff <= 5e-3 or diff <= scale * 1e-5);
}

test "ggml_q4_k x4 packed matmul matches plain q4_k matmul" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6f2a9b4d8c713501);
    const random = prng.random();

    const blocks_per_row = 2;
    const k = blocks_per_row * qk_k_block_size;

    var lhs_blocks: [4 * blocks_per_row]BlockQ8_K = undefined;
    for (&lhs_blocks) |*b| fillRandomBlockQ8_K(b, random, false);
    var rhs_blocks: [4 * blocks_per_row]BlockQ4_K = undefined;
    for (&rhs_blocks) |*b| fillRandomBlockQ4_K(b, random);

    var rhs_plain = try quantizedMatmulRhsQ4_KFromBlocks(allocator, k, 4, &rhs_blocks);
    defer rhs_plain.deinit();
    var rhs_packed = try packMatmulRhsQ4_Kx4(allocator, &rhs_blocks, 4, k, blocks_per_row);
    defer rhs_packed.deinit();

    var expected: [16]f32 = undefined;
    var actual: [16]f32 = undefined;
    matmulQ4_KRhsRange(&expected, &lhs_blocks, &rhs_plain, 4, 4, 0, 4);
    matmulQ4_Kx4RhsRange(&actual, &lhs_blocks, &rhs_packed, 4, 4, 0, 4);

    for (expected, actual) |ref, got| try expectPackedApprox(ref, got);
}

test "ggml_q4_k x8 packed matmul matches plain q4_k matmul" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x9d3e57a1c4820b6f);
    const random = prng.random();

    const blocks_per_row = 2;
    const k = blocks_per_row * qk_k_block_size;

    var lhs_blocks: [4 * blocks_per_row]BlockQ8_K = undefined;
    for (&lhs_blocks) |*b| fillRandomBlockQ8_K(b, random, false);
    var rhs_blocks: [8 * blocks_per_row]BlockQ4_K = undefined;
    for (&rhs_blocks) |*b| fillRandomBlockQ4_K(b, random);

    var rhs_plain = try quantizedMatmulRhsQ4_KFromBlocks(allocator, k, 8, &rhs_blocks);
    defer rhs_plain.deinit();
    var rhs_packed = try packMatmulRhsQ4_Kx8(allocator, &rhs_blocks, 8, k, blocks_per_row);
    defer rhs_packed.deinit();

    var expected: [32]f32 = undefined;
    var actual: [32]f32 = undefined;
    matmulQ4_KRhsRange(&expected, &lhs_blocks, &rhs_plain, 4, 8, 0, 4);
    matmulQ4_Kx8RhsRange(&actual, &lhs_blocks, &rhs_packed, 4, 8, 0, 4);

    for (expected, actual) |ref, got| try expectPackedApprox(ref, got);
}

test "ggml_q4_k x8 padded q8_k x4 lhs matches row lhs matmul" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x48cf1750ab62d39e);
    const random = prng.random();

    const blocks_per_row = 2;
    const k = blocks_per_row * qk_k_block_size;

    var lhs_values: [3 * k]f32 = undefined;
    for (&lhs_values, 0..) |*value, i| {
        const signed: i32 = @as(i32, @intCast((i * 13 + 11) % 251)) - 125;
        value.* = @as(f32, @floatFromInt(signed)) / 7.0;
    }
    var lhs = try Tensor.fromSlice(allocator, &.{ 3, k }, &lhs_values);
    defer lhs.deinit();

    const lhs_rows = try quantizeRowsQ8_K(allocator, &lhs);
    defer allocator.free(lhs_rows);

    var lhs_x4: [blocks_per_row]BlockQ8_Kx4 = undefined;
    try quantizeRowsQ8_Kx4PaddedInto(&lhs_x4, &lhs);

    var rhs_blocks: [8 * blocks_per_row]BlockQ4_K = undefined;
    for (&rhs_blocks) |*b| fillRandomBlockQ4_K(b, random);
    var rhs_packed = try packMatmulRhsQ4_Kx8(allocator, &rhs_blocks, 8, k, blocks_per_row);
    defer rhs_packed.deinit();

    var expected: [3 * 8]f32 = undefined;
    var actual: [3 * 8]f32 = undefined;
    matmulQ4_Kx8RhsRange(&expected, lhs_rows, &rhs_packed, 3, 8, 0, 3);
    matmulQ4_Kx8Q8_Kx4RhsRange(&actual, &lhs_x4, &rhs_packed, 3, 8, 0, 3);

    for (expected, actual) |ref, got| try expectPackedApprox(ref, got);
}

test "ggml_q4_k x2 mmla packed matmul matches plain q4_k matmul" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xb13524fc0e69d78a);
    const random = prng.random();

    const blocks_per_row = 2;
    const k = blocks_per_row * qk_k_block_size;

    var lhs_values: [2 * k]f32 = undefined;
    for (&lhs_values, 0..) |*value, i| {
        const signed: i32 = @as(i32, @intCast((i * 11 + 5) % 251)) - 125;
        value.* = @as(f32, @floatFromInt(signed)) / 9.0;
    }
    var lhs = try Tensor.fromSlice(allocator, &.{ 2, k }, &lhs_values);
    defer lhs.deinit();

    const lhs_blocks = try quantizeRowsQ8_K(allocator, &lhs);
    defer allocator.free(lhs_blocks);

    var rhs_blocks: [2 * blocks_per_row]BlockQ4_K = undefined;
    for (&rhs_blocks) |*b| fillRandomBlockQ4_K(b, random);
    var rhs_plain = try quantizedMatmulRhsQ4_KFromBlocks(allocator, k, 2, &rhs_blocks);
    defer rhs_plain.deinit();
    var rhs_packed = try packMatmulRhsQ4_Kx2Mmla(allocator, &rhs_blocks, 2, k, blocks_per_row);
    defer rhs_packed.deinit();

    var expected: [4]f32 = undefined;
    var actual_rows: [4]f32 = undefined;
    var actual_x2: [4]f32 = undefined;
    matmulQ4_KRhsRange(&expected, lhs_blocks, &rhs_plain, 2, 2, 0, 2);
    matmulQ4_Kx2MmlaRhsRange(&actual_rows, lhs_blocks, &rhs_packed, 2, 2, 0, 2);

    var lhs_x2: [blocks_per_row]BlockQ8_Kx2Mmla = undefined;
    try quantizeRowsQ8_Kx2MmlaInto(&lhs_x2, &lhs);
    matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsRange(&actual_x2, &lhs_x2, &rhs_packed, 2, 2, 0, 2);

    for (expected, actual_rows) |ref, got| try expectPackedApprox(ref, got);
    for (expected, actual_x2) |ref, got| try expectPackedApprox(ref, got);
}

test "ggml_q4_k randomized blocks: row-outer matmul matches scalar reference" {
    // End-to-end over the public entry point (matmulQ4_KRhsRange → RhsTile →
    // dotQ4_KQ8_K), random m x n x k with an odd column count to cover the
    // qk_col_block tail loop. Accumulation order is replicated, so bit-exact.
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x2545f4914f6cdd1d);
    const random = prng.random();

    const m = 3;
    const n = 5;
    const blocks_per_row = 2;
    const k = blocks_per_row * qk_k_block_size;

    var lhs_blocks: [m * blocks_per_row]BlockQ8_K = undefined;
    for (&lhs_blocks) |*b| fillRandomBlockQ8_K(b, random, false);
    var rhs_blocks: [n * blocks_per_row]BlockQ4_K = undefined;
    for (&rhs_blocks) |*b| fillRandomBlockQ4_K(b, random);

    var rhs = try quantizedMatmulRhsQ4_KFromBlocks(allocator, k, n, &rhs_blocks);
    defer rhs.deinit();

    var out: [m * n]f32 = undefined;
    matmulQ4_KRhsRange(&out, &lhs_blocks, &rhs, m, n, 0, m);

    for (0..m) |i| {
        for (0..n) |j| {
            var expected: f32 = 0;
            for (0..blocks_per_row) |bi| {
                expected += refDotQ4_KQ8_K(&rhs_blocks[j * blocks_per_row + bi], &lhs_blocks[i * blocks_per_row + bi]);
            }
            try std.testing.expectEqual(expected, out[i * n + j]);
        }
    }
}

test "Q4_K column-outer matmul matches row-outer tile" {
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 8;
    const m = 17;
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ4_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = f32ToF16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f32ToF16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    var rhs = try quantizedMatmulRhsQ4_KFromBlocks(allocator, k, n, blocks);
    defer rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();
    const qlhs = try quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);

    const out_tile = try allocator.alloc(f32, m * n);
    defer allocator.free(out_tile);
    const out_col = try allocator.alloc(f32, m * n);
    defer allocator.free(out_col);
    matmulQ4_KRhsTile(out_tile, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ4_KRhsCompactColOuter(out_col, qlhs, &rhs, n, 0, m, 0, n);
    for (out_tile, out_col) |t, c| try std.testing.expect(@abs(t - c) <= 1e-3 * @max(@as(f32, 1), @abs(t)));
}

test "Q4_K lane-packed Q8_Kx4 col-outer is bit-identical on the same activations" {
    // Same quantized Q8_K activations, just repacked into the 4-row-interleaved
    // Q8_Kx4 layout: the lane-packed kernel must reproduce the per-row column-outer
    // result exactly (integer dots are order-independent; epilogue ops are identical).
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 9; // non-multiple of any col tile, exercises the column loop tail
    const m = 16; // multiple of 4 so packRowsQ8_Kx4 applies (no padding here)
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ4_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = f32ToF16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f32ToF16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    var rhs = try quantizedMatmulRhsQ4_KFromBlocks(allocator, k, n, blocks);
    defer rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();

    const qlhs = try quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);
    const qlhs_x4 = try qm.packRowsQ8_Kx4(allocator, qlhs, m, k, bpc);
    defer allocator.free(qlhs_x4);

    const out_col = try allocator.alloc(f32, m * n);
    defer allocator.free(out_col);
    const out_x4 = try allocator.alloc(f32, m * n);
    defer allocator.free(out_x4);
    matmulQ4_KRhsCompactColOuter(out_col, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ4_KCompactQ8_Kx4ColOuter(out_x4, qlhs_x4, &rhs, n, m, 0, n);
    for (out_col, out_x4) |c, x| try std.testing.expectEqual(c, x);
}

test "Q4_K lane-packed Q8_Kx4 col-outer matches row-outer tile with padded tail" {
    // m not a multiple of 4: quantizeRowsQ8_Kx4PaddedInto zero-pads the tail group,
    // and the kernel must still match the trusted row-outer tile over the real rows.
    const allocator = std.testing.allocator;
    const k = 512;
    const n = 8;
    const m = 17;
    const bpc = k / qk_k_block_size;
    const blocks = try allocator.alloc(BlockQ4_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = f32ToF16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f32ToF16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    var rhs = try quantizedMatmulRhsQ4_KFromBlocks(allocator, k, n, blocks);
    defer rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();

    const qlhs = try quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);
    const row_groups = (m + 3) / 4;
    const qlhs_x4 = try allocator.alloc(BlockQ8_Kx4, row_groups * bpc);
    defer allocator.free(qlhs_x4);
    try quantizeRowsQ8_Kx4PaddedInto(qlhs_x4, &dense);

    const out_tile = try allocator.alloc(f32, m * n);
    defer allocator.free(out_tile);
    const out_x4 = try allocator.alloc(f32, m * n);
    defer allocator.free(out_x4);
    matmulQ4_KRhsTile(out_tile, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ4_KCompactQ8_Kx4ColOuter(out_x4, qlhs_x4, &rhs, n, m, 0, n);
    for (out_tile, out_x4) |t, c| try std.testing.expect(@abs(t - c) <= 1e-3 * @max(@as(f32, 1), @abs(t)));
}

test "Q4_K col-outer kernels: split column ranges are bit-identical to full range" {
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
    const blocks = try allocator.alloc(BlockQ4_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = f32ToF16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f32ToF16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    var rhs = try quantizedMatmulRhsQ4_KFromBlocks(allocator, k, n, blocks);
    defer rhs.deinit();

    const lhs_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_vals);
    for (lhs_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
    defer dense.deinit();
    const qlhs = try quantizeRowsQ8_K(allocator, &dense);
    defer allocator.free(qlhs);

    const out_full = try allocator.alloc(f32, m * n);
    defer allocator.free(out_full);
    const out_split = try allocator.alloc(f32, m * n);
    defer allocator.free(out_split);

    // Row-outer tile: the m < 4 arm of moeExpertTileDotRange.
    for ([_]usize{ 1, 3 }) |mt| {
        matmulQ4_KRhsTile(out_full, qlhs, &rhs, n, 0, mt, 0, n);
        matmulQ4_KRhsTile(out_split, qlhs, &rhs, n, 0, mt, 0, split);
        matmulQ4_KRhsTile(out_split, qlhs, &rhs, n, 0, mt, split, n);
        try std.testing.expectEqualSlices(f32, out_full[0 .. mt * n], out_split[0 .. mt * n]);
    }

    // Per-row column-outer: the m >= 4 arm.
    matmulQ4_KRhsCompactColOuter(out_full, qlhs, &rhs, n, 0, m, 0, n);
    matmulQ4_KRhsCompactColOuter(out_split, qlhs, &rhs, n, 0, m, 0, split);
    matmulQ4_KRhsCompactColOuter(out_split, qlhs, &rhs, n, 0, m, split, n);
    try std.testing.expectEqualSlices(f32, out_full, out_split);

    // Lane-packed Q8_Kx4 column-outer: the phased-prefill arm (padded m=5 tail).
    const row_groups = (m + 3) / 4;
    const qlhs_x4 = try allocator.alloc(BlockQ8_Kx4, row_groups * bpc);
    defer allocator.free(qlhs_x4);
    try quantizeRowsQ8_Kx4PaddedInto(qlhs_x4, &dense);
    matmulQ4_KCompactQ8_Kx4ColOuter(out_full, qlhs_x4, &rhs, n, m, 0, n);
    matmulQ4_KCompactQ8_Kx4ColOuter(out_split, qlhs_x4, &rhs, n, m, 0, split);
    matmulQ4_KCompactQ8_Kx4ColOuter(out_split, qlhs_x4, &rhs, n, m, split, n);
    try std.testing.expectEqualSlices(f32, out_full, out_split);
}

// --- exact-parity tests: x86/portable SIMD arms vs the scalar reference arms -
//
// The VNNI / AVX2 / portable-widening arms promise BIT-IDENTICAL results to
// accumulateQ4_Kx4Scalar / accumulateQ4_Kx8Scalar / accumulateQ4_Kx8Q8_Kx4Scalar
// (equal i32 sums + identical f32 expression order). On aarch64 hosts these
// tests execute the arms' portable primitive twins (validating the layout
// walks and the f32 expression shape); on x86 hosts the same tests execute
// the real vpdpbusd / vpmaddubsw instructions.

const QKV4f32 = common.QKV4f32;

fn fillRandomBlockQ4_Kx4(block: *qm.BlockQ4_Kx4, random: std.Random, max_nibbles: bool) void {
    for (&block.d) |*d| d.* = f32ToF16Bits(0.25 + random.float(f32));
    for (&block.dmin) |*d| d.* = f32ToF16Bits(random.float(f32) * 0.5);
    for (&block.scales) |*s| s.* = random.uintLessThan(u8, 64); // getScaleMinK4 output domain
    for (&block.mins) |*m| m.* = random.uintLessThan(u8, 64);
    // the x4 pack stores nibble-EXPANDED values in [0,15], one byte each
    for (&block.qs) |*q| q.* = if (max_nibbles) 15 else @intCast(random.uintLessThan(u8, 16));
}

fn fillRandomBlockQ4_Kx8(block: *qm.BlockQ4_Kx8, random: std.Random, max_nibbles: bool) void {
    for (&block.d) |*d| d.* = f32ToF16Bits(0.25 + random.float(f32));
    for (&block.dmin) |*d| d.* = f32ToF16Bits(random.float(f32) * 0.5);
    for (&block.scales) |*s| s.* = random.uintLessThan(u8, 64);
    for (&block.mins) |*m| m.* = random.uintLessThan(u8, 64);
    // the x8 pack stores raw nibble PAIRS: any byte is a valid payload
    for (&block.qs) |*q| q.* = if (max_nibbles) 0xff else random.int(u8);
}

// Random BlockQ8_Kx4 with the bsums interleave quantizeRowsQ8_Kx4*Into writes:
// bsums[(G/4)*16 + row*4 + G%4] = row's 16-group-G sum, whose bytes live at
// qs[subblock*128 + (h*4 + j)*16 + row*4 + t] with G = subblock*2 + h.
fn fillRandomBlockQ8_Kx4(block: *qm.BlockQ8_Kx4, random: std.Random, extreme: bool) void {
    for (&block.d) |*d| d.* = 0.25 + random.float(f32);
    for (&block.qs, 0..) |*q, i| {
        if (extreme) {
            q.* = if (i % 2 == 0) 127 else -127;
        } else {
            q.* = @intCast(@as(i32, random.uintLessThan(u8, 255)) - 127); // -127..127
        }
    }
    for (0..16) |g| {
        const subblock = g / 2;
        const h = g % 2;
        for (0..4) |row| {
            var acc: i32 = 0;
            for (0..4) |j| {
                for (0..4) |t| {
                    acc += block.qs[subblock * 128 + (h * 4 + j) * 16 + row * 4 + t];
                }
            }
            block.bsums[(g / 4) * 16 + row * 4 + g % 4] = @intCast(acc);
        }
    }
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

fn expectAcc2BitEqual(expected: [2]QKV4f32, got: [2]QKV4f32) !void {
    inline for (0..2) |half| try expectVec4BitEqual(expected[half], got[half]);
}

fn expectAcc42BitEqual(expected: [4][2]QKV4f32, got: [4][2]QKV4f32) !void {
    inline for (0..4) |row| try expectAcc2BitEqual(expected[row], got[row]);
}

fn checkQ4_Kx4Arms(lhs: *const BlockQ8_K, rhs: *const qm.BlockQ4_Kx4, acc: QKV4f32) !void {
    const ref = q4_k.accumulateQ4_Kx4Scalar(lhs, rhs, acc);
    try expectVec4BitEqual(ref, q4_k.accumulateQ4_Kx4Vnni(lhs, rhs, acc));
    try expectVec4BitEqual(ref, q4_k.accumulateQ4_Kx4Avx2(lhs, rhs, acc));
    try expectVec4BitEqual(ref, q4_k.accumulateQ4_Kx4Widen(lhs, rhs, acc));
}

fn checkQ4_Kx8Arms(lhs: *const BlockQ8_K, rhs: *const qm.BlockQ4_Kx8, acc: [2]QKV4f32) !void {
    var ref = acc;
    q4_k.accumulateQ4_Kx8Scalar(lhs, rhs, &ref);
    var got = acc;
    q4_k.accumulateQ4_Kx8Vnni(lhs, rhs, &got);
    try expectAcc2BitEqual(ref, got);
    got = acc;
    q4_k.accumulateQ4_Kx8Avx2(lhs, rhs, &got);
    try expectAcc2BitEqual(ref, got);
    got = acc;
    q4_k.accumulateQ4_Kx8Widen(lhs, rhs, &got);
    try expectAcc2BitEqual(ref, got);
}

fn checkQ4_Kx8Q8_Kx4Arms(lhs: *const qm.BlockQ8_Kx4, rhs: *const qm.BlockQ4_Kx8, acc: [4][2]QKV4f32) !void {
    var ref = acc;
    q4_k.accumulateQ4_Kx8Q8_Kx4Scalar(lhs, rhs, &ref);
    var got = acc;
    q4_k.accumulateQ4_Kx8Q8_Kx4Vnni(lhs, rhs, &got);
    try expectAcc42BitEqual(ref, got);
    got = acc;
    q4_k.accumulateQ4_Kx8Q8_Kx4Avx2(lhs, rhs, &got);
    try expectAcc42BitEqual(ref, got);
    got = acc;
    q4_k.accumulateQ4_Kx8Q8_Kx4Widen(lhs, rhs, &got);
    try expectAcc42BitEqual(ref, got);
}

test "ggml_q4_kx4 SIMD arms match the scalar arm bit-exactly" {
    var prng = std.Random.DefaultPrng.init(0x3c6ef372fe94f82b);
    const random = prng.random();

    var lhs: BlockQ8_K = undefined;
    var rhs: qm.BlockQ4_Kx4 = undefined;
    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        fillRandomBlockQ8_K(&lhs, random, false);
        fillRandomBlockQ4_Kx4(&rhs, random, false);
        try checkQ4_Kx4Arms(&lhs, &rhs, randomVec4(random));
    }

    // Edges: max nibbles x alternating ±127 activations (largest dots), then
    // zero scales/mins and max (63) scales/mins on the same payload.
    fillRandomBlockQ8_K(&lhs, random, true);
    fillRandomBlockQ4_Kx4(&rhs, random, true);
    rhs.d = .{ f32ToF16Bits(1.0), f32ToF16Bits(1.0), f32ToF16Bits(1.0), f32ToF16Bits(1.0) };
    rhs.dmin = .{ f32ToF16Bits(0.5), f32ToF16Bits(0.5), f32ToF16Bits(0.5), f32ToF16Bits(0.5) };
    for (&rhs.scales) |*s| s.* = 63;
    for (&rhs.mins) |*m| m.* = 63;
    try checkQ4_Kx4Arms(&lhs, &rhs, @splat(0));
    for (&rhs.scales) |*s| s.* = 0;
    for (&rhs.mins) |*m| m.* = 0;
    try checkQ4_Kx4Arms(&lhs, &rhs, randomVec4(random));
}

test "ggml_q4_kx8 plain-lhs SIMD arms match the scalar arm bit-exactly" {
    var prng = std.Random.DefaultPrng.init(0x9e3779b97f4a7c15);
    const random = prng.random();

    var lhs: BlockQ8_K = undefined;
    var rhs: qm.BlockQ4_Kx8 = undefined;
    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        fillRandomBlockQ8_K(&lhs, random, false);
        fillRandomBlockQ4_Kx8(&rhs, random, false);
        try checkQ4_Kx8Arms(&lhs, &rhs, .{ randomVec4(random), randomVec4(random) });
    }

    fillRandomBlockQ8_K(&lhs, random, true);
    fillRandomBlockQ4_Kx8(&rhs, random, true);
    for (&rhs.d) |*d| d.* = f32ToF16Bits(1.0);
    for (&rhs.dmin) |*d| d.* = f32ToF16Bits(0.5);
    for (&rhs.scales) |*s| s.* = 63;
    for (&rhs.mins) |*m| m.* = 63;
    try checkQ4_Kx8Arms(&lhs, &rhs, .{ @splat(0), @splat(0) });
    for (&rhs.scales) |*s| s.* = 0;
    for (&rhs.mins) |*m| m.* = 0;
    try checkQ4_Kx8Arms(&lhs, &rhs, .{ randomVec4(random), randomVec4(random) });
}

test "ggml_q4_kx8 q8_kx4-lhs SIMD arms match the scalar arm bit-exactly" {
    var prng = std.Random.DefaultPrng.init(0xc2b2ae3d27d4eb4f);
    const random = prng.random();

    const zero_acc: [4][2]QKV4f32 = .{
        .{ @splat(0), @splat(0) },
        .{ @splat(0), @splat(0) },
        .{ @splat(0), @splat(0) },
        .{ @splat(0), @splat(0) },
    };

    var lhs: qm.BlockQ8_Kx4 = undefined;
    var rhs: qm.BlockQ4_Kx8 = undefined;
    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        fillRandomBlockQ8_Kx4(&lhs, random, false);
        fillRandomBlockQ4_Kx8(&rhs, random, false);
        var acc = zero_acc;
        inline for (0..4) |row| acc[row] = .{ randomVec4(random), randomVec4(random) };
        try checkQ4_Kx8Q8_Kx4Arms(&lhs, &rhs, acc);
    }

    fillRandomBlockQ8_Kx4(&lhs, random, true);
    fillRandomBlockQ4_Kx8(&rhs, random, true);
    for (&rhs.d) |*d| d.* = f32ToF16Bits(1.0);
    for (&rhs.dmin) |*d| d.* = f32ToF16Bits(0.5);
    for (&rhs.scales) |*s| s.* = 63;
    for (&rhs.mins) |*m| m.* = 63;
    try checkQ4_Kx8Q8_Kx4Arms(&lhs, &rhs, zero_acc);
    for (&rhs.scales) |*s| s.* = 0;
    for (&rhs.mins) |*m| m.* = 0;
    try checkQ4_Kx8Q8_Kx4Arms(&lhs, &rhs, zero_acc);
}

test "ggml_q4_k packed matmul entry points match the scalar-arm references bit-exactly" {
    // Dispatcher-level exactness over the public entry points. The aarch64
    // sdot arms share the scalar arms' exact integer sums and f32 expression
    // order, so the comparison is bit-exact on EVERY target (x86 arms by the
    // parity contract above, aarch64 by construction).
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1f83d9abfb41bd6b);
    const random = prng.random();

    const blocks_per_row = 2;
    const k = blocks_per_row * qk_k_block_size;

    // x4 pack: m=6 covers the 4-row block and the single-row tail.
    {
        const m = 6;
        const n = 8;
        var lhs_blocks: [m * blocks_per_row]BlockQ8_K = undefined;
        for (&lhs_blocks) |*b| fillRandomBlockQ8_K(b, random, false);
        var rhs_cols: [n * blocks_per_row]BlockQ4_K = undefined;
        for (&rhs_cols) |*b| fillRandomBlockQ4_K(b, random);
        var rhs = try packMatmulRhsQ4_Kx4(allocator, &rhs_cols, n, k, blocks_per_row);
        defer rhs.deinit();

        var got: [m * n]f32 = undefined;
        matmulQ4_Kx4RhsRange(&got, &lhs_blocks, &rhs, m, n, 0, m);

        for (0..m) |i| {
            var j: usize = 0;
            while (j < n) : (j += 4) {
                const rhs_group = rhs.groupBlocks(j / 4);
                var acc: QKV4f32 = @splat(0);
                for (0..blocks_per_row) |bi| {
                    acc = q4_k.accumulateQ4_Kx4Scalar(&lhs_blocks[i * blocks_per_row + bi], &rhs_group[bi], acc);
                }
                const vals: [4]f32 = acc;
                for (0..4) |c| {
                    try std.testing.expectEqual(
                        @as(u32, @bitCast(vals[c])),
                        @as(u32, @bitCast(got[i * n + j + c])),
                    );
                }
            }
        }
    }

    // x8 pack, plain lhs: m=30 covers the 16-row block, the 8- and 4-row
    // tails, and single rows.
    {
        const m = 30;
        const n = 16;
        const lhs_blocks = try allocator.alloc(BlockQ8_K, m * blocks_per_row);
        defer allocator.free(lhs_blocks);
        for (lhs_blocks) |*b| fillRandomBlockQ8_K(b, random, false);
        const rhs_cols = try allocator.alloc(BlockQ4_K, n * blocks_per_row);
        defer allocator.free(rhs_cols);
        for (rhs_cols) |*b| fillRandomBlockQ4_K(b, random);
        var rhs = try packMatmulRhsQ4_Kx8(allocator, rhs_cols, n, k, blocks_per_row);
        defer rhs.deinit();

        const got = try allocator.alloc(f32, m * n);
        defer allocator.free(got);
        matmulQ4_Kx8RhsRange(got, lhs_blocks, &rhs, m, n, 0, m);

        for (0..m) |i| {
            var j: usize = 0;
            while (j < n) : (j += 8) {
                const rhs_group = rhs.groupBlocks(j / 8);
                var acc: [2]QKV4f32 = .{ @splat(0), @splat(0) };
                for (0..blocks_per_row) |bi| {
                    q4_k.accumulateQ4_Kx8Scalar(&lhs_blocks[i * blocks_per_row + bi], &rhs_group[bi], &acc);
                }
                inline for (0..2) |half| {
                    const vals: [4]f32 = acc[half];
                    for (0..4) |c| {
                        try std.testing.expectEqual(
                            @as(u32, @bitCast(vals[c])),
                            @as(u32, @bitCast(got[i * n + j + half * 4 + c])),
                        );
                    }
                }
            }
        }
    }

    // x8 pack, interleaved Q8_Kx4 lhs (the batched prefill path): m=7 covers
    // a full row-group and the zero-padded tail group.
    {
        const m = 7;
        const n = 8;
        var lhs_values: [m * k]f32 = undefined;
        for (&lhs_values, 0..) |*value, i| {
            const signed: i32 = @as(i32, @intCast((i * 19 + 3) % 251)) - 125;
            value.* = @as(f32, @floatFromInt(signed)) / 6.0;
        }
        var lhs = try Tensor.fromSlice(allocator, &.{ m, k }, &lhs_values);
        defer lhs.deinit();
        var lhs_x4: [2 * blocks_per_row]qm.BlockQ8_Kx4 = undefined;
        try quantizeRowsQ8_Kx4PaddedInto(&lhs_x4, &lhs);

        const rhs_cols = try allocator.alloc(BlockQ4_K, n * blocks_per_row);
        defer allocator.free(rhs_cols);
        for (rhs_cols) |*b| fillRandomBlockQ4_K(b, random);
        var rhs = try packMatmulRhsQ4_Kx8(allocator, rhs_cols, n, k, blocks_per_row);
        defer rhs.deinit();

        var got: [m * n]f32 = undefined;
        matmulQ4_Kx8Q8_Kx4RhsRange(&got, &lhs_x4, &rhs, m, n, 0, m);

        var row_group: usize = 0;
        while (row_group < 2) : (row_group += 1) {
            var j: usize = 0;
            while (j < n) : (j += 8) {
                const rhs_group = rhs.groupBlocks(j / 8);
                var acc: [4][2]QKV4f32 = .{
                    .{ @splat(0), @splat(0) },
                    .{ @splat(0), @splat(0) },
                    .{ @splat(0), @splat(0) },
                    .{ @splat(0), @splat(0) },
                };
                for (0..blocks_per_row) |bi| {
                    q4_k.accumulateQ4_Kx8Q8_Kx4Scalar(&lhs_x4[row_group * blocks_per_row + bi], &rhs_group[bi], &acc);
                }
                inline for (0..4) |row| {
                    const i = row_group * 4 + row;
                    if (i < m) {
                        inline for (0..2) |half| {
                            const vals: [4]f32 = acc[row][half];
                            for (0..4) |c| {
                                try std.testing.expectEqual(
                                    @as(u32, @bitCast(vals[c])),
                                    @as(u32, @bitCast(got[i * n + j + half * 4 + c])),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// --- exact-parity tests: the MoE ColOuter 4-row lane dot ---------------------
//
// The ymm arms of dot4RowsSubblockQ8_Kx4 promise BIT-IDENTICAL i32 dots to
// the plain-scalar reference on every host: the aarch64 dev machine executes
// the arms' portable primitive twins, x86 ReleaseFast/Safe executes the real
// vpdpbusd / vpmaddubsw instructions.

const lane_dot_tiers = [_]q4_k.Q4DotTier{ .vnni, .avx2, .widen };

test "q4_k 4-row lane dot SIMD arms match the scalar reference" {
    var prng = std.Random.DefaultPrng.init(0x9d3c5b17e2a4f086);
    const random = prng.random();

    var iter: usize = 0;
    while (iter < 100) : (iter += 1) {
        var a: qm.BlockQ8_Kx4 = undefined;
        fillRandomBlockQ8_Kx4(&a, random, iter % 7 == 0);
        // Pre-unpacked sub-block domain: unpackQ4_KSubblock emits unsigned
        // nibble values in [0,15].
        var wa: [2][16]i8 = undefined;
        for (&wa) |*half| {
            for (half) |*w| w.* = @intCast(random.uintLessThan(u8, 16));
        }
        const wv: [2]common.QKV16i8 = .{ wa[0], wa[1] };
        inline for (0..8) |subblock| {
            const ref: [4]i32 = q4_k.dot4RowsSubblockQ8_Kx4Scalar(&a, subblock, wv);
            inline for (lane_dot_tiers) |tier| {
                try std.testing.expectEqual(ref, @as([4]i32, q4_k.dot4RowsSubblockQ8_Kx4Simd(tier, &a, subblock, wv)));
            }
        }
    }

    // Edge blocks: weight extremes (15 / 0) against activation extremes incl.
    // the out-of-domain -128 (the u8*i8 arms stay exact: pair sums are
    // bounded by 2*15*128 = 3840 << 2^15).
    const edge_a = [_][2]i8{ .{ -128, -128 }, .{ 127, 127 }, .{ 127, -128 }, .{ -127, 127 } };
    const edge_w = [_]i8{ 15, 0 };
    var a: qm.BlockQ8_Kx4 = undefined;
    for (edge_a) |pattern| {
        fillRandomBlockQ8_Kx4(&a, random, false); // sets d + the bsums interleave shape
        for (&a.qs, 0..) |*q, i| q.* = pattern[i % 2];
        // Recompute the interleaved bsums for the pinned activations.
        for (0..16) |g| {
            const subblock = g / 2;
            const h = g % 2;
            for (0..4) |row| {
                var acc: i32 = 0;
                for (0..4) |j| {
                    for (0..4) |t| {
                        acc += a.qs[subblock * 128 + (h * 4 + j) * 16 + row * 4 + t];
                    }
                }
                a.bsums[(g / 4) * 16 + row * 4 + g % 4] = @intCast(acc);
            }
        }
        for (edge_w) |wval| {
            const wv: [2]common.QKV16i8 = .{ @splat(wval), @splat(wval) };
            inline for (0..8) |subblock| {
                const ref: [4]i32 = q4_k.dot4RowsSubblockQ8_Kx4Scalar(&a, subblock, wv);
                inline for (lane_dot_tiers) |tier| {
                    try std.testing.expectEqual(ref, @as([4]i32, q4_k.dot4RowsSubblockQ8_Kx4Simd(tier, &a, subblock, wv)));
                }
            }
        }
    }
}
