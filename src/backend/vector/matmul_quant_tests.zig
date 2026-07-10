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
