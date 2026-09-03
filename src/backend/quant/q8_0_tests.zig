//! Behavioral tests for the Q8_0 / Q8_0x4 quantized matmul kernels
//! (`q8_0.zig`): GGML block-semantic quantize/dequantize parity, roundf-style
//! activation quantization, partial-block rejection, row gather, packed-vs-plain
//! matmul parity, and the randomized row-outer matmul scalar-reference check.
const std = @import("std");
const builtin = @import("builtin");
const dtype_mod = @import("../../dtype.zig");
const tensor = @import("../../tensor.zig");
const qm = @import("../quant.zig");
const q8k = @import("q8k.zig");
const types = @import("types.zig");
const common = @import("common.zig");
const q8_0 = @import("q8_0.zig");

const Tensor = tensor.Tensor;

const BlockQ8_0 = dtype_mod.BlockQ8_0;
const q8_0_block_size = types.q8_0_block_size;

test "ggml_q8_0 quantize and dequantize match GGML block semantics" {
    var src: [q8_0_block_size]f32 = undefined;
    var expected_qs: [q8_0_block_size]i8 = undefined;
    for (&src, &expected_qs, 0..) |*v, *q, i| {
        const value: i32 = @as(i32, @intCast(i)) - 16;
        v.* = @floatFromInt(value);
        q.* = @intCast(value);
    }
    src[0] = -127;
    expected_qs[0] = -127;
    src[q8_0_block_size - 1] = 127;
    expected_qs[q8_0_block_size - 1] = 127;

    var blocks: [1]BlockQ8_0 = undefined;
    try q8k.quantizeRowQ8_0Into(&blocks, &src);

    try std.testing.expectEqual(@as(usize, 34), @sizeOf(BlockQ8_0));
    try std.testing.expectEqual(common.f32ToF16Bits(1.0), blocks[0].d);
    try std.testing.expectEqualSlices(i8, &expected_qs, &blocks[0].qs);

    var dequantized: [q8_0_block_size]f32 = undefined;
    try q8k.dequantizeRowQ8_0Into(&dequantized, &blocks);
    try std.testing.expectEqualSlices(f32, &src, &dequantized);
}

test "ggml_q8_0 activation quantization rounds ties away from zero" {
    var src = [_]f32{0} ** q8_0_block_size;
    src[0] = 127;
    src[1] = 0.5;
    src[2] = 1.5;
    src[3] = 2.5;
    src[4] = -0.5;
    src[5] = -1.5;
    src[6] = -2.5;

    var blocks: [1]BlockQ8_0 = undefined;
    try q8k.quantizeRowQ8_0Into(&blocks, &src);

    try std.testing.expectEqual(@as(i8, 1), blocks[0].qs[1]);
    try std.testing.expectEqual(@as(i8, 2), blocks[0].qs[2]);
    try std.testing.expectEqual(@as(i8, 3), blocks[0].qs[3]);
    try std.testing.expectEqual(@as(i8, -1), blocks[0].qs[4]);
    try std.testing.expectEqual(@as(i8, -2), blocks[0].qs[5]);
    try std.testing.expectEqual(@as(i8, -3), blocks[0].qs[6]);
}

test "ggml_q8_0 rejects partial GGML blocks" {
    var src = [_]f32{0} ** (q8_0_block_size - 1);
    var blocks: [1]BlockQ8_0 = undefined;

    try std.testing.expectError(types.QuantizedFormatError.InvalidQuantizedLength, q8k.quantizeRowQ8_0Into(&blocks, &src));
    try std.testing.expectError(types.QuantizedFormatError.InvalidQuantizedLength, types.blockCountForDType(.q8_0, src.len));
}

test "ggml_q8_0 quantized rows dequantize and gather rows" {
    const allocator = std.testing.allocator;

    var values: [3 * q8_0_block_size]f32 = undefined;
    for (&values, 0..) |*v, i| {
        const col = i % q8_0_block_size;
        const row = i / q8_0_block_size;
        const value: i32 = if (col == 0)
            -127
        else if (col == q8_0_block_size - 1)
            127
        else
            @as(i32, @intCast(col)) - 16 + @as(i32, @intCast(row));
        v.* = @floatFromInt(value);
    }

    var dense = try Tensor.fromSlice(allocator, &.{ 3, q8_0_block_size }, &values);
    defer dense.deinit();

    var qrows = try q8k.quantizeRowsQ8_0(allocator, &dense);
    defer qrows.deinit();

    var dequantized = try Tensor.zeros(allocator, &.{ 3, q8_0_block_size });
    defer dequantized.deinit();
    try q8k.dequantizeRowsQ8_0Into(&dequantized, &qrows);
    try std.testing.expectEqualSlices(f32, &values, dequantized.dataConst());

    var gathered = try Tensor.zeros(allocator, &.{ 2, q8_0_block_size });
    defer gathered.deinit();
    try q8k.getRowsQ8_0Into(&gathered, &qrows, &.{ 2, 0 });
    try std.testing.expectEqualSlices(f32, values[2 * q8_0_block_size ..][0..q8_0_block_size], gathered.dataConst()[0..q8_0_block_size]);
    try std.testing.expectEqualSlices(f32, values[0..q8_0_block_size], gathered.dataConst()[q8_0_block_size..][0..q8_0_block_size]);

    var invalid_index_out = try Tensor.zeros(allocator, &.{ 1, q8_0_block_size });
    defer invalid_index_out.deinit();
    try std.testing.expectError(tensor.TensorError.IndexOutOfBounds, q8k.getRowsQ8_0Into(&invalid_index_out, &qrows, &.{3}));
}
test "ggml_q8_0x4 packed lhs matches plain lhs matmul" {
    const allocator = std.testing.allocator;
    const m = 4;
    const n = 8;
    const k = 64;
    const blocks_per_row = k / q8_0_block_size;

    var lhs_values: [m * k]f32 = undefined;
    for (&lhs_values, 0..) |*value, i| {
        const signed: i32 = @as(i32, @intCast((i * 13 + 7) % 251)) - 125;
        value.* = @as(f32, @floatFromInt(signed)) / 8.0;
    }
    var lhs = try Tensor.fromSlice(allocator, &.{ m, k }, &lhs_values);
    defer lhs.deinit();

    var rhs_values: [n * k]f32 = undefined;
    for (&rhs_values, 0..) |*value, i| {
        const signed: i32 = @as(i32, @intCast((i * 17 + 3) % 251)) - 125;
        value.* = @as(f32, @floatFromInt(signed)) / 7.0;
    }
    var rhs_dense = try Tensor.fromSlice(allocator, &.{ n, k }, &rhs_values);
    defer rhs_dense.deinit();

    var lhs_plain: [m * blocks_per_row]BlockQ8_0 = undefined;
    try q8k.quantizeRowsQ8_0Into(&lhs_plain, &lhs);

    var lhs_packed: [(m / 4) * blocks_per_row]types.BlockQ8_0x4 = undefined;
    try q8_0.quantizeRowsQ8_0x4PaddedInto(&lhs_packed, &lhs);

    var rhs_rows = try q8k.quantizeRowsQ8_0(allocator, &rhs_dense);
    defer rhs_rows.deinit();
    var rhs = try q8_0.packMatmulRhsQ8_0x4(allocator, rhs_rows.blocks, n, k, blocks_per_row);
    defer rhs.deinit();

    var ref: [m * n]f32 = undefined;
    var got: [m * n]f32 = undefined;
    q8_0.matmulQ8_0x4RhsRange(&ref, &lhs_plain, &rhs, m, n, 0, m);
    q8_0.matmulQ8_0x4PackedRhsRange(&got, &lhs_packed, &rhs, m, n, 0, m);

    for (ref, got) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-3);
    }
}

test "ggml_q8_0x4 packed ColsFirst dual-row path matches plain matmul" {
    const allocator = std.testing.allocator;
    // m=204 -> 51 row-groups: with row_group_tile=8 the last ColsFirst block
    // holds 3 row-groups, exercising both the dual-row pair loop and the odd
    // trailing row-group. m>=128 selects the ColsFirst path.
    const m = 204;
    const n = 12;
    const k = 64;
    const blocks_per_row = k / q8_0_block_size;

    const lhs_values = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_values);
    for (lhs_values, 0..) |*value, i| {
        const signed: i32 = @as(i32, @intCast((i * 13 + 7) % 251)) - 125;
        value.* = @as(f32, @floatFromInt(signed)) / 8.0;
    }
    var lhs = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_values);
    defer lhs.deinit();

    const rhs_values = try allocator.alloc(f32, n * k);
    defer allocator.free(rhs_values);
    for (rhs_values, 0..) |*value, i| {
        const signed: i32 = @as(i32, @intCast((i * 17 + 3) % 251)) - 125;
        value.* = @as(f32, @floatFromInt(signed)) / 7.0;
    }
    var rhs_dense = try Tensor.fromSlice(allocator, &.{ n, k }, rhs_values);
    defer rhs_dense.deinit();

    const lhs_plain = try allocator.alloc(BlockQ8_0, m * blocks_per_row);
    defer allocator.free(lhs_plain);
    try q8k.quantizeRowsQ8_0Into(lhs_plain, &lhs);

    const lhs_packed = try allocator.alloc(types.BlockQ8_0x4, (m / 4) * blocks_per_row);
    defer allocator.free(lhs_packed);
    try q8_0.quantizeRowsQ8_0x4PaddedInto(lhs_packed, &lhs);

    var rhs_rows = try q8k.quantizeRowsQ8_0(allocator, &rhs_dense);
    defer rhs_rows.deinit();
    var rhs = try q8_0.packMatmulRhsQ8_0x4(allocator, rhs_rows.blocks, n, k, blocks_per_row);
    defer rhs.deinit();

    const ref = try allocator.alloc(f32, m * n);
    defer allocator.free(ref);
    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    q8_0.matmulQ8_0x4RhsRange(ref, lhs_plain, &rhs, m, n, 0, m);
    q8_0.matmulQ8_0x4PackedRhsRange(got, lhs_packed, &rhs, m, n, 0, m);

    for (ref, got) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-3);
    }
}

// --- exact-parity tests: SIMD arms vs the scalar reference arms --------------
//
// The VNNI / AVX2 / portable-widening arms promise BIT-IDENTICAL results to
// accumulateQ8_0x4PackedScalar / accumulateQ8_0x4Scalar (equal i32 sums +
// identical f32 association). On aarch64 hosts these tests execute the arms'
// portable primitive twins (validating the bias/sign-trick/widening algebra
// and the f32 expression shape); on x86 hosts the same tests execute the real
// vpdpbusd / vpmaddubsw+vpsignb instructions.

fn fillRandomBlockQ8_0x4(block: *qm.BlockQ8_0x4, random: std.Random, allow_m128: bool) void {
    for (&block.d) |*d| d.* = common.f32ToF16Bits(0.25 + random.float(f32));
    for (&block.qs) |*q| {
        if (allow_m128) {
            q.* = @bitCast(random.int(u8)); // full i8 range incl. -128 (GGUF weight side)
        } else {
            q.* = @intCast(@as(i32, random.uintLessThan(u8, 255)) - 127); // quantizeToI8 domain
        }
    }
}

fn randomAcc4(random: std.Random) [4]common.QKV4f32 {
    var acc: [4]common.QKV4f32 = undefined;
    for (&acc) |*row| {
        var vals: [4]f32 = undefined;
        for (&vals) |*v| v.* = (random.float(f32) - 0.5) * 64.0;
        row.* = vals;
    }
    return acc;
}

fn expectAcc4BitEqual(expected: [4]common.QKV4f32, got: [4]common.QKV4f32) !void {
    inline for (0..4) |r| {
        const e: [4]f32 = expected[r];
        const g: [4]f32 = got[r];
        for (e, g) |ev, gv| {
            try std.testing.expectEqual(@as(u32, @bitCast(ev)), @as(u32, @bitCast(gv)));
        }
    }
}

fn expectVec4BitEqual(expected: common.QKV4f32, got: common.QKV4f32) !void {
    const e: [4]f32 = expected;
    const g: [4]f32 = got;
    for (e, g) |ev, gv| {
        try std.testing.expectEqual(@as(u32, @bitCast(ev)), @as(u32, @bitCast(gv)));
    }
}

// Run all packed arms (+ the Dual variants) against the scalar arm for one
// (lhs_full, lhs_act, rhs) triple. lhs_full exercises the unrestricted arms
// (VNNI bias form, widening) incl. -128; lhs_act stays in the [-127,127]
// activation domain the AVX2 sign-trick arm documents.
fn checkPackedArms(lhs_full: *const qm.BlockQ8_0x4, lhs_act: *const qm.BlockQ8_0x4, rhs: *const qm.BlockQ8_0x4, acc0: [4]common.QKV4f32) !void {
    var ref_full = acc0;
    q8_0.accumulateQ8_0x4PackedScalar(lhs_full, rhs, &ref_full);
    var ref_act = acc0;
    q8_0.accumulateQ8_0x4PackedScalar(lhs_act, rhs, &ref_act);

    var got = acc0;
    q8_0.accumulateQ8_0x4PackedVnni(lhs_full, rhs, &got);
    try expectAcc4BitEqual(ref_full, got);

    got = acc0;
    q8_0.accumulateQ8_0x4PackedWiden(lhs_full, rhs, &got);
    try expectAcc4BitEqual(ref_full, got);

    got = acc0;
    q8_0.accumulateQ8_0x4PackedAvx2(lhs_act, rhs, &got);
    try expectAcc4BitEqual(ref_act, got);

    // Dual arms: bit-for-bit the same math as two single-group calls.
    var got_a = acc0;
    var got_b = acc0;
    q8_0.accumulateQ8_0x4PackedDualVnni(lhs_full, lhs_act, rhs, &got_a, &got_b);
    try expectAcc4BitEqual(ref_full, got_a);
    try expectAcc4BitEqual(ref_act, got_b);

    got_a = acc0;
    got_b = acc0;
    q8_0.accumulateQ8_0x4PackedDualAvx2(lhs_act, lhs_act, rhs, &got_a, &got_b);
    try expectAcc4BitEqual(ref_act, got_a);
    try expectAcc4BitEqual(ref_act, got_b);
}

test "ggml_q8_0x4 packed SIMD arms match the scalar arm bit-exactly" {
    var prng = std.Random.DefaultPrng.init(0x8f3b7a11c2d94e05);
    const random = prng.random();

    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        var lhs_full: qm.BlockQ8_0x4 = undefined;
        var lhs_act: qm.BlockQ8_0x4 = undefined;
        var rhs: qm.BlockQ8_0x4 = undefined;
        fillRandomBlockQ8_0x4(&lhs_full, random, true);
        fillRandomBlockQ8_0x4(&lhs_act, random, false);
        fillRandomBlockQ8_0x4(&rhs, random, true);
        try checkPackedArms(&lhs_full, &lhs_act, &rhs, randomAcc4(random));
    }

    // Edge blocks: all -128 / all +127 / alternating extremes on the weight
    // side; the activation side pinned to its clamped extremes. Covers the
    // VNNI bias-correction maxima and the sign-trick |rhs|=128 corner.
    var lhs_full: qm.BlockQ8_0x4 = undefined;
    var lhs_act: qm.BlockQ8_0x4 = undefined;
    var rhs: qm.BlockQ8_0x4 = undefined;
    for (&lhs_full.d) |*d| d.* = common.f32ToF16Bits(1.0);
    lhs_act.d = lhs_full.d;
    rhs.d = lhs_full.d;

    const edges = [_]struct { w: [2]i8, a: [2]i8 }{
        .{ .w = .{ -128, -128 }, .a = .{ -127, -127 } },
        .{ .w = .{ 127, 127 }, .a = .{ 127, 127 } },
        .{ .w = .{ -128, 127 }, .a = .{ 127, -127 } },
        .{ .w = .{ -128, -128 }, .a = .{ 127, 127 } },
    };
    const acc_zero: [4]common.QKV4f32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
    for (edges) |edge| {
        for (&lhs_full.qs, 0..) |*q, i| q.* = edge.w[i % 2];
        for (&lhs_act.qs, 0..) |*q, i| q.* = edge.a[i % 2];
        for (&rhs.qs, 0..) |*q, i| q.* = edge.w[(i + 1) % 2];
        try checkPackedArms(&lhs_full, &lhs_act, &rhs, acc_zero);
    }
}

test "ggml_q8_0x4 plain-lhs SIMD arms match the scalar arm bit-exactly" {
    var prng = std.Random.DefaultPrng.init(0x517cc1b727220a95);
    const random = prng.random();

    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        var lhs_full: BlockQ8_0 = undefined;
        var lhs_act: BlockQ8_0 = undefined;
        var rhs: qm.BlockQ8_0x4 = undefined;
        fillRandomBlockQ8_0(&lhs_full, random, true);
        fillRandomBlockQ8_0(&lhs_act, random, false);
        fillRandomBlockQ8_0x4(&rhs, random, true);

        var vals: [4]f32 = undefined;
        for (&vals) |*v| v.* = (random.float(f32) - 0.5) * 64.0;
        const acc: common.QKV4f32 = vals;

        const ref_full = q8_0.accumulateQ8_0x4Scalar(&lhs_full, &rhs, acc);
        const ref_act = q8_0.accumulateQ8_0x4Scalar(&lhs_act, &rhs, acc);

        try expectVec4BitEqual(ref_full, q8_0.accumulateQ8_0x4Vnni(&lhs_full, &rhs, acc));
        try expectVec4BitEqual(ref_full, q8_0.accumulateQ8_0x4Widen(&lhs_full, &rhs, acc));
        try expectVec4BitEqual(ref_act, q8_0.accumulateQ8_0x4Avx2(&lhs_act, &rhs, acc));
    }

    // Edges: weights all -128 against activation extremes ±127.
    var lhs_act: BlockQ8_0 = undefined;
    var rhs: qm.BlockQ8_0x4 = undefined;
    lhs_act.d = common.f32ToF16Bits(1.0);
    for (&rhs.d) |*d| d.* = common.f32ToF16Bits(1.0);
    for (&rhs.qs) |*q| q.* = -128;
    for (&lhs_act.qs, 0..) |*q, i| q.* = if (i % 2 == 0) 127 else -127;
    const acc_zero: common.QKV4f32 = @splat(0);
    const ref = q8_0.accumulateQ8_0x4Scalar(&lhs_act, &rhs, acc_zero);
    try expectVec4BitEqual(ref, q8_0.accumulateQ8_0x4Vnni(&lhs_act, &rhs, acc_zero));
    try expectVec4BitEqual(ref, q8_0.accumulateQ8_0x4Widen(&lhs_act, &rhs, acc_zero));
    try expectVec4BitEqual(ref, q8_0.accumulateQ8_0x4Avx2(&lhs_act, &rhs, acc_zero));
}

test "ggml_q8_0x4 packed matmul entry points match the scalar-arm reference" {
    // End-to-end over the public packed entry point with production-shaped
    // inputs (quantized activations, packed weights with injected -128 bytes),
    // m large enough to route through the ColsFirst dual-row path AND the odd
    // trailing row-group, plus a small-m run for the row-major path. On
    // non-aarch64 the dispatcher arms are bit-identical to the scalar arm →
    // exact compare; on aarch64 the sdot arm's f32 association differs →
    // tolerance.
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x2b992ddfa23249d6);
    const random = prng.random();

    const m = 204;
    const n = 12;
    const k = 64;
    const blocks_per_row = k / q8_0_block_size;

    const lhs_values = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs_values);
    for (lhs_values) |*value| value.* = (random.float(f32) - 0.5) * 8.0;
    var lhs = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_values);
    defer lhs.deinit();

    const rhs_values = try allocator.alloc(f32, n * k);
    defer allocator.free(rhs_values);
    for (rhs_values) |*value| value.* = (random.float(f32) - 0.5) * 4.0;
    var rhs_dense = try Tensor.fromSlice(allocator, &.{ n, k }, rhs_values);
    defer rhs_dense.deinit();

    const lhs_packed = try allocator.alloc(qm.BlockQ8_0x4, (m / 4) * blocks_per_row);
    defer allocator.free(lhs_packed);
    try q8_0.quantizeRowsQ8_0x4PaddedInto(lhs_packed, &lhs);

    var rhs_rows = try q8k.quantizeRowsQ8_0(allocator, &rhs_dense);
    defer rhs_rows.deinit();
    var rhs = try q8_0.packMatmulRhsQ8_0x4(allocator, rhs_rows.blocks, n, k, blocks_per_row);
    defer rhs.deinit();
    // Inject -128 weight bytes (legal in the GGUF format, never produced by
    // our encoder) so the exactness claim covers the full weight domain.
    for (rhs.blocks, 0..) |*block, bi| {
        block.qs[(bi * 7) % block.qs.len] = -128;
        block.qs[(bi * 13 + 41) % block.qs.len] = -128;
    }

    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    q8_0.matmulQ8_0x4PackedRhsRange(got, lhs_packed, &rhs, m, n, 0, m);

    const ref = try allocator.alloc(f32, m * n);
    defer allocator.free(ref);
    var row_group: usize = 0;
    while (row_group < m / 4) : (row_group += 1) {
        var j: usize = 0;
        while (j < n) : (j += 4) {
            const lhs_group = lhs_packed[row_group * blocks_per_row ..][0..blocks_per_row];
            const rhs_group = rhs.groupBlocks(j / 4);
            var acc: [4]common.QKV4f32 = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
            for (0..blocks_per_row) |block_index| {
                q8_0.accumulateQ8_0x4PackedScalar(&lhs_group[block_index], &rhs_group[block_index], &acc);
            }
            inline for (0..4) |r| {
                const row = row_group * 4 + r;
                const vals: [4]f32 = acc[r];
                for (0..4) |c| ref[row * n + j + c] = vals[c];
            }
        }
    }

    for (ref, got) |expected, actual| {
        if (comptime builtin.cpu.arch == .aarch64) {
            try std.testing.expectApproxEqAbs(expected, actual, 1e-2);
        } else {
            try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(actual)));
        }
    }

    // Plain-lhs entry point (accumulateQ8_0x4Rows / accumulateQ8_0x4 path).
    const lhs_plain = try allocator.alloc(BlockQ8_0, m * blocks_per_row);
    defer allocator.free(lhs_plain);
    try q8k.quantizeRowsQ8_0Into(lhs_plain, &lhs);

    const got_plain = try allocator.alloc(f32, m * n);
    defer allocator.free(got_plain);
    q8_0.matmulQ8_0x4RhsRange(got_plain, lhs_plain, &rhs, m, n, 0, m);

    for (0..m) |i| {
        var j: usize = 0;
        while (j < n) : (j += 4) {
            const rhs_group = rhs.groupBlocks(j / 4);
            var acc: common.QKV4f32 = @splat(0);
            for (0..blocks_per_row) |block_index| {
                acc = q8_0.accumulateQ8_0x4Scalar(&lhs_plain[i * blocks_per_row + block_index], &rhs_group[block_index], acc);
            }
            const vals: [4]f32 = acc;
            for (0..4) |c| {
                if (comptime builtin.cpu.arch == .aarch64) {
                    try std.testing.expectApproxEqAbs(vals[c], got_plain[i * n + j + c], 1e-2);
                } else {
                    try std.testing.expectEqual(@as(u32, @bitCast(vals[c])), @as(u32, @bitCast(got_plain[i * n + j + c])));
                }
            }
        }
    }
}

// Scalar reference replica of dotQ8_0Q8_0: exact i32 integer dot, identical
// f32 expression shape — comparisons against it are BIT-EXACT on every target.
// Mirrored in src/x86dot_check.zig for the cross-ISA attestation runs.
fn refDotQ8_0Q8_0(a: *const BlockQ8_0, b: *const BlockQ8_0) f32 {
    var acc: i32 = 0;
    for (a.qs, b.qs) |x, y| acc += @as(i32, x) * @as(i32, y);
    return @as(f32, @floatFromInt(acc)) * (common.f16BitsToF32(a.d) * common.f16BitsToF32(b.d));
}

fn fillRandomBlockQ8_0(block: *BlockQ8_0, random: std.Random, allow_m128: bool) void {
    block.d = common.f32ToF16Bits(0.25 + random.float(f32));
    for (&block.qs) |*q| {
        if (allow_m128) {
            q.* = @bitCast(random.int(u8)); // full i8 range incl. -128 (GGUF weight side)
        } else {
            q.* = @intCast(@as(i32, random.uintLessThan(u8, 255)) - 127); // quantizeToI8 domain
        }
    }
}

test "ggml_q8_0 randomized blocks: row-outer matmul matches scalar reference" {
    // End-to-end over the public entry point (matmulQ8_0RhsTile → dotQ8_0Q8_0)
    // with n chosen to cover both the q8_0_col_block loop and the column tail.
    // On non-aarch64 the accumulation order is replicated, so the comparison
    // is BIT-EXACT (this is the path the new AVX2 branch lives on). On aarch64
    // matmulQ8_0RhsTile routes to the sdot tile kernel, which accumulates in
    // four f32 lanes and reduces once at the end — a different (but equally
    // valid) f32 association — so the comparison is tolerance-based there.
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xd1b54a32d192ed03);
    const random = prng.random();

    const m = 3;
    const n = 5;
    const blocks_per_row = 2;
    const k = blocks_per_row * q8_0_block_size;

    var lhs_blocks: [m * blocks_per_row]BlockQ8_0 = undefined;
    for (&lhs_blocks) |*blk| fillRandomBlockQ8_0(blk, random, false);

    const rhs_blocks = try allocator.alloc(BlockQ8_0, n * blocks_per_row);
    for (rhs_blocks) |*blk| fillRandomBlockQ8_0(blk, random, true);

    var rhs = types.QuantizedMatmulRhsQ8_0{
        .allocator = allocator,
        .blocks = rhs_blocks,
        .blocks_per_column = blocks_per_row,
        .k = k,
        .n = n,
    };
    defer rhs.deinit();

    var out: [m * n]f32 = undefined;
    q8_0.matmulQ8_0RhsTile(&out, &lhs_blocks, &rhs, n, 0, m, 0, n);

    for (0..m) |i| {
        for (0..n) |j| {
            var expected: f32 = 0;
            for (0..blocks_per_row) |bi| {
                expected += refDotQ8_0Q8_0(&lhs_blocks[i * blocks_per_row + bi], &rhs_blocks[j * blocks_per_row + bi]);
            }
            if (comptime builtin.cpu.arch == .aarch64) {
                try std.testing.expectApproxEqRel(expected, out[i * n + j], 1e-5);
            } else {
                try std.testing.expectEqual(expected, out[i * n + j]);
            }
        }
    }
}

test "ggml_q8_0 single-row chunked sdot path matches scalar reference" {
    // The aarch64 single-row (decode GEMV) path processes four blocks per
    // step with a pre-converted row-scale strip and a batched rhs-scale
    // convert. Shapes chosen to hit the four-block chunk loop, the
    // block-remainder loop, and the column tail (n % 4 != 0).
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x9e3779b97f4a7c15);
    const random = prng.random();

    const configs = [_]struct { blocks_per_row: usize, n: usize }{
        .{ .blocks_per_row = 5, .n = 6 },
        .{ .blocks_per_row = 8, .n = 4 },
    };

    inline for (configs) |config| {
        const blocks_per_row = config.blocks_per_row;
        const n = config.n;
        const k = blocks_per_row * q8_0_block_size;

        var lhs_blocks: [blocks_per_row]BlockQ8_0 = undefined;
        for (&lhs_blocks) |*blk| fillRandomBlockQ8_0(blk, random, false);

        const rhs_blocks = try allocator.alloc(BlockQ8_0, n * blocks_per_row);
        for (rhs_blocks) |*blk| fillRandomBlockQ8_0(blk, random, true);

        var rhs = types.QuantizedMatmulRhsQ8_0{
            .allocator = allocator,
            .blocks = rhs_blocks,
            .blocks_per_column = blocks_per_row,
            .k = k,
            .n = n,
        };
        defer rhs.deinit();

        var out: [n]f32 = undefined;
        q8_0.matmulQ8_0RhsTile(&out, &lhs_blocks, &rhs, n, 0, 1, 0, n);

        for (0..n) |j| {
            var expected: f32 = 0;
            for (0..blocks_per_row) |bi| {
                expected += refDotQ8_0Q8_0(&lhs_blocks[bi], &rhs_blocks[j * blocks_per_row + bi]);
            }
            if (comptime builtin.cpu.arch == .aarch64) {
                try std.testing.expectApproxEqRel(expected, out[j], 1e-5);
            } else {
                try std.testing.expectEqual(expected, out[j]);
            }
        }
    }
}

test "ggml_q8_0x4 padded ColsFirst matches per-row reference and guards the tail rows" {
    // m=5 -> two row groups, three padding lanes in the last: routes through
    // the guarded column-outer path. `out` is sized exactly m*n, so a write
    // to any padding row would trip the out-of-bounds safety check.
    const allocator = std.testing.allocator;
    const m = 5;
    const n = 8;
    const k = 64;
    const blocks_per_row = k / q8_0_block_size;

    var lhs_values: [m * k]f32 = undefined;
    for (&lhs_values, 0..) |*value, i| {
        const signed: i32 = @as(i32, @intCast((i * 29 + 5) % 251)) - 125;
        value.* = @as(f32, @floatFromInt(signed)) / 9.0;
    }
    var lhs = try Tensor.fromSlice(allocator, &.{ m, k }, &lhs_values);
    defer lhs.deinit();

    var rhs_values: [n * k]f32 = undefined;
    for (&rhs_values, 0..) |*value, i| {
        const signed: i32 = @as(i32, @intCast((i * 23 + 11) % 251)) - 125;
        value.* = @as(f32, @floatFromInt(signed)) / 6.0;
    }
    var rhs_dense = try Tensor.fromSlice(allocator, &.{ n, k }, &rhs_values);
    defer rhs_dense.deinit();

    var lhs_plain: [m * blocks_per_row]BlockQ8_0 = undefined;
    try q8k.quantizeRowsQ8_0Into(&lhs_plain, &lhs);

    var lhs_padded: [((m + 3) / 4) * blocks_per_row]types.BlockQ8_0x4 = undefined;
    try q8_0.quantizeRowsQ8_0x4PaddedInto(&lhs_padded, &lhs);

    var rhs_rows = try q8k.quantizeRowsQ8_0(allocator, &rhs_dense);
    defer rhs_rows.deinit();
    var rhs = try q8_0.packMatmulRhsQ8_0x4(allocator, rhs_rows.blocks, n, k, blocks_per_row);
    defer rhs.deinit();

    var ref: [m * n]f32 = undefined;
    var got: [m * n]f32 = undefined;
    q8_0.matmulQ8_0x4RhsRange(&ref, &lhs_plain, &rhs, m, n, 0, m);
    q8_0.matmulQ8_0x4PackedPaddedRhsRange(&got, &lhs_padded, &rhs, m, n);

    for (ref, got) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-3);
    }
}
