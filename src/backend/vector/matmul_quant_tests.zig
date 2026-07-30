//! Behavioral tests for the pooled quantized-matmul dispatch
//! (`matmul_quant.zig`): the QuantizedRhsParallel row- and column-split paths
//! of the TQ2_0 entries must reproduce the serial Range kernels bitwise —
//! every output element is one complete dot, so the split can never change
//! the accumulation order.
const std = @import("std");
const matmul_quant = @import("matmul_quant.zig");
const parallel = @import("../../parallel.zig");
const qm = @import("../quant.zig");
const tensor = @import("../../tensor.zig");
const thread = @import("../../thread.zig");

const testing = std.testing;

const Tensor = tensor.Tensor;
const qk_k = qm.qk_k_block_size;

// Shapes pinned to the dispatch gates (common.zig matmulThreadCount /
// i8ColumnThreadCount): the row split needs m >= vector_column_min_m and
// m·n·k >= vector_matmul_work_threshold; the column split needs
// m < vector_column_min_m, n >= vector_column_min_n, the same work bound,
// and at least two vector_column_chunk column chunks. Both sit exactly at
// the work threshold so the tests stay fast. On a single-core runner the
// cpu-count factor degrades both to the serial path and the comparison
// still holds (it just stops exercising the pool).
const row_m: usize = parallel.vector_column_min_m;
const row_n: usize = 128;
const row_k: usize = qk_k;
const col_m: usize = 1;
const col_n: usize = 4096;
const col_k: usize = qk_k;

comptime {
    std.debug.assert(row_m >= parallel.vector_column_min_m);
    std.debug.assert(row_m * row_n * row_k >= parallel.vector_matmul_work_threshold);
    std.debug.assert(col_m < parallel.vector_column_min_m);
    std.debug.assert(col_n >= parallel.vector_column_min_n);
    std.debug.assert(col_m * col_n * col_k >= parallel.vector_matmul_work_threshold);
    std.debug.assert(col_n / parallel.vector_column_chunk >= 2);
}

fn fillUniform(prng: *std.Random.DefaultPrng, values: []f32, scale: f32) void {
    const random = prng.random();
    for (values) |*v| v.* = (random.float(f32) * 2.0 - 1.0) * scale;
}

fn expectPooledTQ2_0F32MatchesSerial(m: usize, n: usize, k: usize, seed: u64) !void {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const w = try allocator.alloc(f32, n * k);
    defer allocator.free(w);
    fillUniform(&prng, w, 1.0);
    const lhs = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs);
    fillUniform(&prng, lhs, 2.0);

    var rhs = try qm.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w);
    defer rhs.deinit();

    const serial = try allocator.alloc(f32, m * n);
    defer allocator.free(serial);
    const pooled = try allocator.alloc(f32, m * n);
    defer allocator.free(pooled);
    @memset(pooled, std.math.nan(f32));

    qm.matmulTQ2_0F32RhsRange(serial, lhs, &rhs, m, n, 0, m);

    var pool: thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = 3 });
    defer pool.deinit();
    matmul_quant.matmul2DTQ2_0F32RhsIntoWithConfig(pooled, lhs, &rhs, m, n, k, .{ .pool = &pool });

    try testing.expectEqualSlices(f32, serial, pooled);
}

fn expectPooledTQ2_0Int8MatchesSerial(m: usize, n: usize, k: usize, seed: u64) !void {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const w = try allocator.alloc(f32, n * k);
    defer allocator.free(w);
    fillUniform(&prng, w, 1.0);
    const lhs = try allocator.alloc(f32, m * k);
    defer allocator.free(lhs);
    fillUniform(&prng, lhs, 2.0);

    var rhs = try qm.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w);
    defer rhs.deinit();

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, lhs);
    defer a.deinit();
    const qlhs = try qm.quantizeRowsQ8_K(allocator, &a);
    defer allocator.free(qlhs);

    const serial = try allocator.alloc(f32, m * n);
    defer allocator.free(serial);
    const pooled = try allocator.alloc(f32, m * n);
    defer allocator.free(pooled);
    @memset(pooled, std.math.nan(f32));

    qm.matmulTQ2_0RhsRange(serial, qlhs, &rhs, m, n, 0, m);

    var pool: thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = 3 });
    defer pool.deinit();
    matmul_quant.matmul2DTQ2_0RhsIntoWithConfig(pooled, qlhs, &rhs, m, n, k, .{ .pool = &pool });

    try testing.expectEqualSlices(f32, serial, pooled);
}

test "pooled tq2_0 f32 matmul row split matches serial bitwise" {
    try expectPooledTQ2_0F32MatchesSerial(row_m, row_n, row_k, 0x7e60);
}

test "pooled tq2_0 f32 matmul column split matches serial bitwise" {
    try expectPooledTQ2_0F32MatchesSerial(col_m, col_n, col_k, 0x7e61);
}

test "pooled tq2_0 int8 matmul row split matches serial bitwise" {
    try expectPooledTQ2_0Int8MatchesSerial(row_m, row_n, row_k, 0x7e62);
}

test "pooled tq2_0 int8 matmul column split matches serial bitwise" {
    try expectPooledTQ2_0Int8MatchesSerial(col_m, col_n, col_k, 0x7e63);
}

// ---- Pooled-vs-serial parity across every SplitPolicy entry ----
//
// Each entry's row/column split must reproduce its serial Range kernel
// bitwise — every output element is one whole-k dot, so task boundaries can
// never change the accumulation order, and a wrong lane-grouped boundary
// (c0/r0 not a multiple of the pack group) shows up as wrong values. Block
// payloads are random BYTES (any pattern decodes, and both paths read the
// same bytes); outputs compare as raw bit patterns so a random f16 scale
// that decodes to NaN or inf still compares exactly.

fn randomBlocks(comptime B: type, allocator: std.mem.Allocator, prng: *std.Random.DefaultPrng, count: usize) ![]B {
    const blocks = try allocator.alloc(B, count);
    prng.random().bytes(std.mem.sliceAsBytes(blocks));
    return blocks;
}

fn expectPooledMatchesSerialBits(
    comptime entryFn: anytype,
    comptime rangeFn: anytype,
    lhs_blocks: anytype,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
) !void {
    const allocator = testing.allocator;
    const serial = try allocator.alloc(f32, m * n);
    defer allocator.free(serial);
    const pooled = try allocator.alloc(f32, m * n);
    defer allocator.free(pooled);
    @memset(pooled, std.math.nan(f32));

    rangeFn(serial, lhs_blocks, rhs, m, n, 0, m);

    var pool: thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = 3 });
    defer pool.deinit();
    entryFn(pooled, lhs_blocks, rhs, m, n, k, .{ .pool = &pool });

    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(serial), std.mem.sliceAsBytes(pooled));
}

test "pooled dispatch matches serial bitwise: plain-block formats" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51CA7);
    inline for (.{
        .{ qm.BlockQ8_K, qk_k, qm.QuantizedMatmulRhsQ2_K, qm.BlockQ2_K, qk_k, matmul_quant.matmul2DQ2_KRhsIntoWithConfig, qm.matmulQ2_KRhsRange },
        .{ qm.BlockQ8_K, qk_k, qm.QuantizedMatmulRhsQ3_K, qm.BlockQ3_K, qk_k, matmul_quant.matmul2DQ3_KRhsIntoWithConfig, qm.matmulQ3_KRhsRange },
        .{ qm.BlockQ8_K, qk_k, qm.QuantizedMatmulRhsQ4_K, qm.BlockQ4_K, qk_k, matmul_quant.matmul2DQ4_KRhsIntoWithConfig, qm.matmulQ4_KRhsRange },
        .{ qm.BlockQ8_K, qk_k, qm.QuantizedMatmulRhsQ5_K, qm.BlockQ5_K, qk_k, matmul_quant.matmul2DQ5_KRhsIntoWithConfig, qm.matmulQ5_KRhsRange },
        .{ qm.BlockQ8_K, qk_k, qm.QuantizedMatmulRhsQ6_K, qm.BlockQ6_K, qk_k, matmul_quant.matmul2DQ6_KRhsIntoWithConfig, qm.matmulQ6_KRhsRange },
    }) |spec| {
        inline for (.{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k } }) |shape| {
            const m, const n, const k = shape;
            const lhs = try randomBlocks(spec[0], allocator, &prng, m * (k / spec[1]));
            defer allocator.free(lhs);
            const rblocks = try randomBlocks(spec[3], allocator, &prng, n * (k / spec[4]));
            defer allocator.free(rblocks);
            const rhs: spec[2] = .{ .allocator = null, .blocks = rblocks, .k = k, .n = n, .blocks_per_column = k / spec[4] };
            try expectPooledMatchesSerialBits(spec[5], spec[6], lhs, &rhs, m, n, k);
        }
    }
    // Q8_0/Q4_0 wrap their blocks in a rows container instead (Q8_0's may
    // borrow; Q4_0's owns its blocks).
    inline for (.{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k } }) |shape| {
        const m, const n, const k = shape;
        const lhs = try randomBlocks(qm.BlockQ8_0, allocator, &prng, m * (k / 32));
        defer allocator.free(lhs);

        const q80_blocks = try randomBlocks(qm.BlockQ8_0, allocator, &prng, n * (k / 32));
        defer allocator.free(q80_blocks);
        const q80_rhs: qm.QuantizedMatmulRhsQ8_0 = .{ .rows = .{
            .allocator = null,
            .blocks = q80_blocks,
            .rows = n,
            .cols = k,
            .blocks_per_row = k / 32,
        }, .k = k, .n = n };
        try expectPooledMatchesSerialBits(matmul_quant.matmul2DQ8_0RhsIntoWithConfig, qm.matmulQ8_0RhsRange, lhs, &q80_rhs, m, n, k);

        var q40_rhs: qm.QuantizedMatmulRhsQ4_0 = .{ .rows = .{
            .allocator = allocator,
            .blocks = try randomBlocks(qm.BlockQ4_0, allocator, &prng, n * (k / 32)),
            .rows = n,
            .cols = k,
            .blocks_per_row = k / 32,
        }, .k = k, .n = n };
        defer q40_rhs.deinit();
        try expectPooledMatchesSerialBits(matmul_quant.matmul2DQ4_0RhsIntoWithConfig, qm.matmulQ4_0RhsRange, lhs, &q40_rhs, m, n, k);
    }
}

test "pooled dispatch matches serial bitwise: interleaved-pack formats" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51CA8);
    inline for (.{
        // {LhsBlock, lhs elems/block, PlainBlock, packFn, entry, range, shapes}
        // Shapes cover the column arm (m = 1), the row arm at the work
        // threshold (m = 32), and — for the x8/cap-3 tiles — m = 128.
        .{ qm.BlockQ8_0, 32, qm.BlockQ8_0, qm.packMatmulRhsQ8_0x4, matmul_quant.matmul2DQ8_0x4RhsIntoWithConfig, qm.matmulQ8_0x4RhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k } } },
        .{ qm.BlockQ8_K, qk_k, qm.BlockQ4_K, qm.packMatmulRhsQ4_Kx4, matmul_quant.matmul2DQ4_Kx4RhsIntoWithConfig, qm.matmulQ4_Kx4RhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k } } },
        .{ qm.BlockQ8_K, qk_k, qm.BlockQ4_K, qm.packMatmulRhsQ4_Kx8, matmul_quant.matmul2DQ4_Kx8RhsIntoWithConfig, qm.matmulQ4_Kx8RhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k }, .{ 128, 128, qk_k } } },
        .{ qm.BlockQ8_K, qk_k, qm.BlockQ4_K, qm.packMatmulRhsQ4_Kx2Mmla, matmul_quant.matmul2DQ4_Kx2MmlaRhsIntoWithConfig, qm.matmulQ4_Kx2MmlaRhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k }, .{ 128, 128, qk_k } } },
        .{ qm.BlockQ8_K, qk_k, qm.BlockQ5_K, qm.packMatmulRhsQ5_Kx8, matmul_quant.matmul2DQ5_Kx8RhsIntoWithConfig, qm.matmulQ5_Kx8RhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k }, .{ 128, 128, qk_k } } },
        .{ qm.BlockQ8_K, qk_k, qm.BlockQ6_K, qm.packMatmulRhsQ6_Kx4, matmul_quant.matmul2DQ6_Kx4RhsIntoWithConfig, qm.matmulQ6_Kx4RhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k } } },
    }) |spec| {
        inline for (spec[6]) |shape| {
            const m, const n, const k = shape;
            const lhs = try randomBlocks(spec[0], allocator, &prng, m * (k / spec[1]));
            defer allocator.free(lhs);
            const plain = try randomBlocks(spec[2], allocator, &prng, n * (k / qk_k) * (qk_k / blockElems(spec[2])));
            defer allocator.free(plain);
            var rhs = try spec[3](allocator, plain, n, k, k / blockElems(spec[2]));
            defer rhs.deinit();
            try expectPooledMatchesSerialBits(spec[4], spec[5], lhs, &rhs, m, n, k);
        }
    }
}

fn blockElems(comptime B: type) usize {
    return if (B == qm.BlockQ8_0) 32 else qk_k;
}

test "pooled dispatch matches serial bitwise: lane-packed LHS entries" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51CA9);
    inline for (.{
        // {LhsBlock, rows/lhs group, PlainBlock, packFn, entry, range, shapes}
        // (64, 64, 256) forces the ROW arm (n below the column gate);
        // m stays a group multiple, the packed-LHS entries' contract.
        .{ qm.BlockQ8_Kx4, 4, qm.BlockQ4_K, qm.packMatmulRhsQ4_Kx8, matmul_quant.matmul2DQ4_Kx8Q8_Kx4RhsIntoWithConfig, qm.matmulQ4_Kx8Q8_Kx4RhsRange, .{ .{ 64, 64, qk_k }, .{ 4, col_n, col_k }, .{ row_m, row_n, row_k } } },
        .{ qm.BlockQ8_Kx4, 4, qm.BlockQ5_K, qm.packMatmulRhsQ5_Kx8, matmul_quant.matmul2DQ5_Kx8Q8_Kx4RhsIntoWithConfig, qm.matmulQ5_Kx8Q8_Kx4RhsRange, .{ .{ 64, 64, qk_k }, .{ 4, col_n, col_k }, .{ row_m, row_n, row_k } } },
        .{ qm.BlockQ8_Kx2Mmla, 2, qm.BlockQ4_K, qm.packMatmulRhsQ4_Kx2Mmla, matmul_quant.matmul2DQ4_Kx2MmlaQ8_Kx2MmlaRhsIntoWithConfig, qm.matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsRange, .{ .{ 64, 64, qk_k }, .{ 2, col_n, col_k }, .{ 128, 128, qk_k } } },
        .{ qm.BlockQ8_0x4, 4, qm.BlockQ8_0, qm.packMatmulRhsQ8_0x4, matmul_quant.matmul2DQ8_0x4PackedRhsIntoWithConfig, qm.matmulQ8_0x4PackedRhsRange, .{ .{ row_m, row_n, row_k }, .{ 4, col_n, col_k }, .{ 128, col_n, col_k } } },
    }) |spec| {
        inline for (spec[6]) |shape| {
            const m, const n, const k = shape;
            const groups = (m + spec[1] - 1) / spec[1];
            const lhs = try randomBlocks(spec[0], allocator, &prng, groups * (k / qk_k) * (qk_k / blockElems(spec[2])));
            defer allocator.free(lhs);
            const plain = try randomBlocks(spec[2], allocator, &prng, n * (k / blockElems(spec[2])));
            defer allocator.free(plain);
            var rhs = try spec[3](allocator, plain, n, k, k / blockElems(spec[2]));
            defer rhs.deinit();
            try expectPooledMatchesSerialBits(spec[4], spec[5], lhs, &rhs, m, n, k);
        }
    }
}
