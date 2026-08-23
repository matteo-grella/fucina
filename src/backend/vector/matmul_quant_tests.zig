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
const qk_k = qm.types.qk_k_block_size;

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

    var rhs = try qm.ternary.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w);
    defer rhs.deinit();

    const serial = try allocator.alloc(f32, m * n);
    defer allocator.free(serial);
    const pooled = try allocator.alloc(f32, m * n);
    defer allocator.free(pooled);
    @memset(pooled, std.math.nan(f32));

    qm.ternary.matmulTQ2_0F32RhsRange(serial, lhs, &rhs, m, n, 0, m);

    var pool: thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = 3 });
    defer pool.deinit();
    matmul_quant.matmul2DTQ2_0F32RhsInto(.{ .pool = &pool }, pooled, lhs, &rhs, m, n, k);

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

    var rhs = try qm.ternary.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w);
    defer rhs.deinit();

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, lhs);
    defer a.deinit();
    const qlhs = try qm.q8k.quantizeRowsQ8_K(allocator, &a);
    defer allocator.free(qlhs);

    const serial = try allocator.alloc(f32, m * n);
    defer allocator.free(serial);
    const pooled = try allocator.alloc(f32, m * n);
    defer allocator.free(pooled);
    @memset(pooled, std.math.nan(f32));

    qm.ternary.matmulTQ2_0RhsRange(serial, qlhs, &rhs, m, n, 0, m);

    var pool: thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = 3 });
    defer pool.deinit();
    matmul_quant.matmul2DTQ2_0RhsInto(.{ .pool = &pool }, pooled, qlhs, &rhs, m, n, k);

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
// Each entry's row/column split must agree with its serial Range kernel.
// Bitwise equality is NOT the contract here: kernels may anchor their
// row/column micro-tiling at (r0, c0) (the aarch64 q8_0 tile does), so a
// split range's group phase and tail differ from the full range's and the
// per-cell rounding drifts in the last ulps — the same reassociation
// stance the backend states for GEMMs (§9.3). What a mis-transcribed
// SplitPolicy produces instead is order-of-magnitude wrong values (a lane
// group bisected, a column attributed to the wrong task), so the check is
// bitwise-first with a tight per-element tolerance fallback. Inputs are
// real activations/weights run through the real quantizers, so every
// derived field (Q8_K bsums included) is self-consistent and every kernel
// identity computes the same finite quantities.

fn randomBlocks(comptime B: type, allocator: std.mem.Allocator, prng: *std.Random.DefaultPrng, count: usize) ![]B {
    const blocks = try allocator.alloc(B, count);
    prng.random().bytes(std.mem.sliceAsBytes(blocks));
    return blocks;
}

fn f16Bits(v: f16) u16 {
    return @bitCast(v);
}

fn expectPooledMatchesSerial(
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
    entryFn(.{ .pool = &pool }, pooled, lhs_blocks, rhs, m, n, k);

    if (std.mem.eql(u8, std.mem.sliceAsBytes(serial), std.mem.sliceAsBytes(pooled))) return;
    for (serial, pooled) |s, p| {
        if (@abs(s - p) <= 1e-4) continue;
        try testing.expectApproxEqRel(s, p, 2e-3);
    }
}

test "pooled dispatch matches serial: plain-block formats" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51CA7);
    inline for (.{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k } }) |shape| {
        const m, const n, const k = shape;

        const acts = try allocator.alloc(f32, m * k);
        defer allocator.free(acts);
        fillUniform(&prng, acts, 1.0);
        var a = try Tensor.fromSlice(allocator, &.{ m, k }, acts);
        defer a.deinit();
        const lhs_q8k = try qm.q8k.quantizeRowsQ8_K(allocator, &a);
        defer allocator.free(lhs_q8k);
        const lhs_q80 = try allocator.alloc(qm.BlockQ8_0, m * (k / 32));
        defer allocator.free(lhs_q80);
        try qm.q8k.quantizeRowsQ8_0Into(lhs_q80, &a);

        const w = try allocator.alloc(f32, n * k);
        defer allocator.free(w);
        fillUniform(&prng, w, 1.0);

        // Encoder-backed K-quants against the Q8_K activations.
        inline for (.{
            .{ .q4_k, qm.QuantizedMatmulRhsQ4_K, qm.BlockQ4_K, matmul_quant.matmul2DQ4_KRhsInto, qm.q4_k.matmulQ4_KRhsRange },
            .{ .q5_k, qm.QuantizedMatmulRhsQ5_K, qm.BlockQ5_K, matmul_quant.matmul2DQ5_KRhsInto, qm.q5_k.matmulQ5_KRhsRange },
            .{ .q6_k, qm.QuantizedMatmulRhsQ6_K, qm.BlockQ6_K, matmul_quant.matmul2DQ6_KRhsInto, qm.q6_k.matmulQ6_KRhsRange },
        }) |spec| {
            const rblocks = try allocator.alloc(spec[2], n * (k / qk_k));
            defer allocator.free(rblocks);
            try qm.quantizeRowForDType(spec[0], rblocks, w);
            const rhs: spec[1] = .{ .allocator = null, .blocks = rblocks, .k = k, .n = n, .blocks_per_column = k / qk_k };
            try expectPooledMatchesSerial(spec[3], spec[4], lhs_q8k, &rhs, m, n, k);
        }

        // Q2_K/Q3_K have no encoder: random payloads with the f16 scale
        // fields pinned finite (the quantities stay well-defined; only the
        // trit/scale patterns are arbitrary).
        {
            const rblocks = try randomBlocks(qm.BlockQ2_K, allocator, &prng, n * (k / qk_k));
            defer allocator.free(rblocks);
            for (rblocks) |*bl| bl.dm = .{ f16Bits(0.01), f16Bits(0.002) };
            const rhs: qm.QuantizedMatmulRhsQ2_K = .{ .allocator = null, .blocks = rblocks, .k = k, .n = n, .blocks_per_column = k / qk_k };
            try expectPooledMatchesSerial(matmul_quant.matmul2DQ2_KRhsInto, qm.cold.matmulQ2_KRhsRange, lhs_q8k, &rhs, m, n, k);
        }
        {
            const rblocks = try randomBlocks(qm.BlockQ3_K, allocator, &prng, n * (k / qk_k));
            defer allocator.free(rblocks);
            for (rblocks) |*bl| bl.d = f16Bits(0.01);
            const rhs: qm.QuantizedMatmulRhsQ3_K = .{ .allocator = null, .blocks = rblocks, .k = k, .n = n, .blocks_per_column = k / qk_k };
            try expectPooledMatchesSerial(matmul_quant.matmul2DQ3_KRhsInto, qm.cold.matmulQ3_KRhsRange, lhs_q8k, &rhs, m, n, k);
        }

        // Q8_0/Q4_0 against the Q8_0 activations (rows containers).
        {
            const rblocks = try allocator.alloc(qm.BlockQ8_0, n * (k / 32));
            defer allocator.free(rblocks);
            try qm.quantizeRowForDType(.q8_0, rblocks, w);
            const rhs: qm.QuantizedMatmulRhsQ8_0 = .{ .rows = .{
                .allocator = null,
                .blocks = rblocks,
                .rows = n,
                .cols = k,
                .blocks_per_row = k / 32,
            }, .k = k, .n = n };
            try expectPooledMatchesSerial(matmul_quant.matmul2DQ8_0RhsInto, qm.q8_0.matmulQ8_0RhsRange, lhs_q80, &rhs, m, n, k);
        }
        {
            var rhs: qm.QuantizedMatmulRhsQ4_0 = .{ .rows = .{
                .allocator = allocator,
                .blocks = try allocator.alloc(qm.BlockQ4_0, n * (k / 32)),
                .rows = n,
                .cols = k,
                .blocks_per_row = k / 32,
            }, .k = k, .n = n };
            defer rhs.deinit();
            try qm.quantizeRowForDType(.q4_0, rhs.rows.blocks, w);
            try expectPooledMatchesSerial(matmul_quant.matmul2DQ4_0RhsInto, qm.cold.matmulQ4_0RhsRange, lhs_q80, &rhs, m, n, k);
        }
    }
}

test "pooled dispatch matches serial: interleaved-pack formats" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51CA8);
    inline for (.{
        // {plain dtype, PlainBlock, lhs kind (q8k/q80), packFn, entry, range, shapes}
        // Shapes cover the column arm (m = 1), the row arm at the work
        // threshold (m = 32), and — for the x8/cap-3 tiles — m = 128.
        .{ .q8_0, qm.BlockQ8_0, false, qm.q8_0.packMatmulRhsQ8_0x4, matmul_quant.matmul2DQ8_0x4RhsInto, qm.q8_0.matmulQ8_0x4RhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k } } },
        .{ .q4_k, qm.BlockQ4_K, true, qm.q4_k.packMatmulRhsQ4_Kx4, matmul_quant.matmul2DQ4_Kx4RhsInto, qm.q4_k.matmulQ4_Kx4RhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k } } },
        .{ .q4_k, qm.BlockQ4_K, true, qm.q4_k.packMatmulRhsQ4_Kx8, matmul_quant.matmul2DQ4_Kx8RhsInto, qm.q4_k.matmulQ4_Kx8RhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k }, .{ 128, 128, qk_k } } },
        .{ .q4_k, qm.BlockQ4_K, true, qm.q4_k.packMatmulRhsQ4_Kx2Mmla, matmul_quant.matmul2DQ4_Kx2MmlaRhsInto, qm.q4_k.matmulQ4_Kx2MmlaRhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k }, .{ 128, 128, qk_k } } },
        .{ .q5_k, qm.BlockQ5_K, true, qm.q5_k.packMatmulRhsQ5_Kx8, matmul_quant.matmul2DQ5_Kx8RhsInto, qm.q5_k.matmulQ5_Kx8RhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k }, .{ 128, 128, qk_k } } },
        .{ .q6_k, qm.BlockQ6_K, true, qm.q6_k.packMatmulRhsQ6_Kx4, matmul_quant.matmul2DQ6_Kx4RhsInto, qm.q6_k.matmulQ6_Kx4RhsRange, .{ .{ row_m, row_n, row_k }, .{ col_m, col_n, col_k } } },
    }) |spec| {
        inline for (spec[6]) |shape| {
            const m, const n, const k = shape;

            const acts = try allocator.alloc(f32, m * k);
            defer allocator.free(acts);
            fillUniform(&prng, acts, 1.0);
            var a = try Tensor.fromSlice(allocator, &.{ m, k }, acts);
            defer a.deinit();
            const lhs_q8k = try qm.q8k.quantizeRowsQ8_K(allocator, &a);
            defer allocator.free(lhs_q8k);
            const lhs_q80 = try allocator.alloc(qm.BlockQ8_0, m * (k / 32));
            defer allocator.free(lhs_q80);
            try qm.q8k.quantizeRowsQ8_0Into(lhs_q80, &a);

            const w = try allocator.alloc(f32, n * k);
            defer allocator.free(w);
            fillUniform(&prng, w, 1.0);
            const plain = try allocator.alloc(spec[1], n * (k / blockElems(spec[1])));
            defer allocator.free(plain);
            try qm.quantizeRowForDType(spec[0], plain, w);

            var rhs = try spec[3](allocator, plain, n, k, k / blockElems(spec[1]));
            defer rhs.deinit();
            if (spec[2]) {
                try expectPooledMatchesSerial(spec[4], spec[5], lhs_q8k, &rhs, m, n, k);
            } else {
                try expectPooledMatchesSerial(spec[4], spec[5], lhs_q80, &rhs, m, n, k);
            }
        }
    }
}

fn blockElems(comptime B: type) usize {
    return if (B == qm.BlockQ8_0) 32 else qk_k;
}

test "pooled dispatch matches serial: lane-packed LHS entries" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51CA9);
    inline for (.{
        // {plain dtype, PlainBlock, LhsBlock, lhs quantizer, packFn, entry, range, shapes}
        // (64, 64, 256) forces the ROW arm (n below the column gate);
        // m stays a group multiple, the packed-LHS entries' contract.
        .{ .q4_k, qm.BlockQ4_K, qm.BlockQ8_Kx4, qm.q8k.quantizeRowsQ8_Kx4PaddedInto, qm.q4_k.packMatmulRhsQ4_Kx8, matmul_quant.matmul2DQ4_Kx8Q8_Kx4RhsInto, qm.q4_k.matmulQ4_Kx8Q8_Kx4RhsRange, .{ .{ 64, 64, qk_k }, .{ 4, col_n, col_k }, .{ row_m, row_n, row_k } } },
        .{ .q5_k, qm.BlockQ5_K, qm.BlockQ8_Kx4, qm.q8k.quantizeRowsQ8_Kx4PaddedInto, qm.q5_k.packMatmulRhsQ5_Kx8, matmul_quant.matmul2DQ5_Kx8Q8_Kx4RhsInto, qm.q5_k.matmulQ5_Kx8Q8_Kx4RhsRange, .{ .{ 64, 64, qk_k }, .{ 4, col_n, col_k }, .{ row_m, row_n, row_k } } },
        .{ .q4_k, qm.BlockQ4_K, qm.BlockQ8_Kx2Mmla, qm.q8k.quantizeRowsQ8_Kx2MmlaInto, qm.q4_k.packMatmulRhsQ4_Kx2Mmla, matmul_quant.matmul2DQ4_Kx2MmlaQ8_Kx2MmlaRhsInto, qm.q4_k.matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsRange, .{ .{ 64, 64, qk_k }, .{ 2, col_n, col_k }, .{ 128, 128, qk_k } } },
        .{ .q8_0, qm.BlockQ8_0, qm.BlockQ8_0x4, qm.q8_0.quantizeRowsQ8_0x4PaddedInto, qm.q8_0.packMatmulRhsQ8_0x4, matmul_quant.matmul2DQ8_0x4PackedRhsInto, qm.q8_0.matmulQ8_0x4PackedRhsRange, .{ .{ row_m, row_n, row_k }, .{ 4, col_n, col_k }, .{ 128, col_n, col_k } } },
    }) |spec| {
        inline for (spec[7]) |shape| {
            const m, const n, const k = shape;

            const acts = try allocator.alloc(f32, m * k);
            defer allocator.free(acts);
            fillUniform(&prng, acts, 1.0);
            var a = try Tensor.fromSlice(allocator, &.{ m, k }, acts);
            defer a.deinit();
            const group: usize = if (spec[2] == qm.BlockQ8_Kx2Mmla) 2 else 4;
            const groups = (m + group - 1) / group;
            const lhs = try allocator.alloc(spec[2], groups * (k / blockElems(spec[1])));
            defer allocator.free(lhs);
            try spec[3](lhs, &a);

            const w = try allocator.alloc(f32, n * k);
            defer allocator.free(w);
            fillUniform(&prng, w, 1.0);
            const plain = try allocator.alloc(spec[1], n * (k / blockElems(spec[1])));
            defer allocator.free(plain);
            try qm.quantizeRowForDType(spec[0], plain, w);

            var rhs = try spec[4](allocator, plain, n, k, k / blockElems(spec[1]));
            defer rhs.deinit();
            try expectPooledMatchesSerial(spec[5], spec[6], lhs, &rhs, m, n, k);
        }
    }
}
