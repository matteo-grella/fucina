//! Behavioral tests for the backend dispatch facade (`backend.zig`):
//! scalar elementwise ops, the unary/clamp/gated contiguous kernels, the
//! basic dense matmul path, kernel dispatch with an attached work pool, and
//! the native quantized-matmul bulk/remainder row split for off-multiple m.
const std = @import("std");
const backend_mod = @import("backend.zig");
const native = @import("backend/native.zig");
const quant = @import("backend/quant.zig");
const thread = @import("thread.zig");
const vector = @import("backend/vector.zig");
pub const ops = @import("backend/ops.zig");

const Backend = backend_mod.Backend;
const Tensor = backend_mod.Tensor;

test "backend executes scalar ops" {
    const allocator = std.testing.allocator;
    var backend = Backend.init();

    var a = try Tensor.fromSlice(allocator, &.{3}, &.{ 1, 2, 3 });
    defer a.deinit();
    var b = try Tensor.fromSlice(allocator, &.{3}, &.{ 4, 5, 6 });
    defer b.deinit();

    var c = try Tensor.zeros(allocator, &.{3});
    defer c.deinit();
    try backend.addInto(&c, &a, &b);

    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, c.dataConst());
}

test "backend executes unary clamp and gated contiguous kernels" {
    const allocator = std.testing.allocator;
    var backend = Backend.init();

    var x = try Tensor.fromSlice(allocator, &.{4}, &.{ -2, -0.5, 0.5, 2 });
    defer x.deinit();
    var gate = try Tensor.fromSlice(allocator, &.{4}, &.{ -1, 0, 1, 2 });
    defer gate.deinit();

    var unary = try Tensor.zeros(allocator, &.{4});
    defer unary.deinit();
    backend.unaryContiguousIntoUnchecked(.silu, &unary, &x, x.len());

    for (x.dataConst(), unary.dataConst()) |value, actual| {
        const expected = value * ops.sigmoidScalar(value);
        try std.testing.expectApproxEqAbs(expected, actual, 1e-6);
    }

    var fast_tanh = try Tensor.zeros(allocator, &.{4});
    defer fast_tanh.deinit();
    backend.unaryContiguousIntoUnchecked(.fast_tanh, &fast_tanh, &x, x.len());
    for (x.dataConst(), fast_tanh.dataConst()) |value, actual| {
        try std.testing.expectApproxEqAbs(ops.fastTanhScalar(value), actual, 1e-6);
    }

    var clamped = try Tensor.zeros(allocator, &.{4});
    defer clamped.deinit();
    backend.clampContiguousIntoUnchecked(&clamped, &x, x.len(), -1, 1);
    try std.testing.expectEqualSlices(f32, &.{ -1, -0.5, 0.5, 1 }, clamped.dataConst());

    var leaky = try Tensor.zeros(allocator, &.{4});
    defer leaky.deinit();
    backend.leakyReluContiguousIntoUnchecked(&leaky, &x, x.len(), 0.1);
    try std.testing.expectEqualSlices(f32, &.{ -0.2, -0.05, 0.5, 2 }, leaky.dataConst());

    var gated = try Tensor.zeros(allocator, &.{4});
    defer gated.deinit();
    backend.gatedContiguousIntoUnchecked(.swiglu, &gated, &x, &gate, x.len());
    for (x.dataConst(), gate.dataConst(), gated.dataConst()) |left, gate_value, actual| {
        const expected = left * gate_value * ops.sigmoidScalar(gate_value);
        try std.testing.expectApproxEqAbs(expected, actual, 1e-6);
    }
}

test "backend kernels dispatch through an attached work pool" {
    const allocator = std.testing.allocator;
    var backend = Backend.init();

    var pool: thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = 4 });
    defer pool.deinit();
    backend.setWorkPool(&pool);
    defer backend.setWorkPool(null);

    const n = 4096;
    var a = try Tensor.zeros(allocator, &.{n});
    defer a.deinit();
    var b = try Tensor.zeros(allocator, &.{n});
    defer b.deinit();
    for (a.data(), b.data(), 0..) |*av, *bv, i| {
        av.* = @floatFromInt(i);
        bv.* = @floatFromInt(2 * i);
    }

    var c = try Tensor.zeros(allocator, &.{n});
    defer c.deinit();
    backend.addContiguousIntoUnchecked(&c, &a, &b, n);

    for (c.dataConst(), 0..) |actual, i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(3 * i)), actual);
    }
}

test "backend matmul" {
    const allocator = std.testing.allocator;
    var backend = Backend.init();

    var a = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();

    var c = try Tensor.zeros(allocator, &.{ 2, 2 });
    defer c.deinit();
    try backend.matmulInto(&c, &a, &b);

    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, c.dataConst());
}

// ---------------- native quantized dispatch: off-multiple-m row split ----------------
//
// For a non-multiple-of-4 m the native dispatchers run every row through the
// padded x4 kernel (q4_k, m >= 4), or split a multiple-of-4 bulk through the
// x4 kernel plus a 1-3-row remainder through the row kernel (q8_0 m >= 32,
// q5_k m >= 128; below the floor the row path takes all rows in one pass).
// The x4 and row kernels fold scales in different f32 association orders, so
// the contract is split-invariance, checked bitwise per row range:
//   - bulk rows match the plain m % 4 == 0 dispatch of the same rows;
//   - remainder rows match the kernel that owns them (row kernel for q8_0 and
//     q5_k, padded x4 group for q4_k) run standalone on just those rows;
//   - below-floor m matches the row-kernel reference on every row;
//   - pooled dispatch matches unpooled bit-for-bit (row/column splits never
//     change per-element math).

const split_test_n: usize = 32;
const split_test_k: usize = 512;
const split_test_max_m: usize = 129;
const split_test_ms = [_]usize{ 5, 6, 7, 9, 15, 33, 127, 129 };

fn expectBitEqualF32(expected: []const f32, actual: []const f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |ref, got, i| {
        if (@as(u32, @bitCast(ref)) != @as(u32, @bitCast(got))) {
            std.debug.print("bit mismatch at {d}: {x:0>8} vs {x:0>8}\n", .{ i, @as(u32, @bitCast(ref)), @as(u32, @bitCast(got)) });
            return error.TestExpectedEqual;
        }
    }
}

fn expectSplitApprox(expected: []const f32, actual: []const f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |ref, got| {
        const diff = @abs(ref - got);
        const scale = @max(@abs(ref), @as(f32, 1));
        try std.testing.expect(diff <= 5e-3 or diff <= scale * 1e-5);
    }
}

fn fillSplitTestValues(values: []f32, random: std.Random) void {
    for (values) |*value| value.* = (random.float(f32) - 0.5) * 4.0;
}

// Runs a native matmul2DQuantizedRhs* dispatcher over the first m rows of
// lhs_values; caller frees the returned output.
fn runSplitDispatch(
    comptime dispatchFn: anytype,
    allocator: std.mem.Allocator,
    rhs: anytype,
    lhs_values: []const f32,
    m: usize,
    config: vector.ParallelConfig,
) ![]f32 {
    var lhs = try Tensor.fromSlice(allocator, &.{ m, split_test_k }, lhs_values[0 .. m * split_test_k]);
    defer lhs.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, split_test_n });
    defer out.deinit();
    try dispatchFn(allocator, &out, &lhs, rhs, m, split_test_n, split_test_k, config);
    return allocator.dupe(f32, out.dataConst());
}

// Row-kernel reference over m rows (the exact path the dispatcher's remainder
// takes for q8_k-lhs formats); caller frees the returned output.
fn rowsRefQ8_K(
    comptime rowsKernel: anytype,
    allocator: std.mem.Allocator,
    rhs: anytype,
    lhs_values: []const f32,
    m: usize,
) ![]f32 {
    var lhs = try Tensor.fromSlice(allocator, &.{ m, split_test_k }, lhs_values[0 .. m * split_test_k]);
    defer lhs.deinit();
    const qlhs = try quant.quantizeRowsQ8_K(allocator, &lhs);
    defer allocator.free(qlhs);
    const out = try allocator.alloc(f32, m * split_test_n);
    errdefer allocator.free(out);
    rowsKernel(out, qlhs, rhs, m, split_test_n, split_test_k, vector.ParallelConfig{});
    return out;
}

fn rowsRefQ8_0(
    allocator: std.mem.Allocator,
    rhs: *const quant.QuantizedMatmulRhsQ8_0x4,
    lhs_values: []const f32,
    m: usize,
) ![]f32 {
    var lhs = try Tensor.fromSlice(allocator, &.{ m, split_test_k }, lhs_values[0 .. m * split_test_k]);
    defer lhs.deinit();
    const blocks_per_row = try quant.q8_0BlockCount(split_test_k);
    const qlhs = try allocator.alloc(quant.BlockQ8_0, m * blocks_per_row);
    defer allocator.free(qlhs);
    try quant.quantizeRowsQ8_0Into(qlhs, &lhs);
    const out = try allocator.alloc(f32, m * split_test_n);
    errdefer allocator.free(out);
    vector.matmul2DQ8_0x4RhsIntoWithConfig(out, qlhs, rhs, m, split_test_n, split_test_k, .{});
    return out;
}

// Standalone padded x4 group over 1-3 rows: the exact math the q4_k padded
// dispatch applies to its final (padded) row group; caller frees the output.
fn paddedTailRefQ4_Kx8(
    allocator: std.mem.Allocator,
    rhs: *const quant.QuantizedMatmulRhsQ4_Kx8,
    tail_values: []const f32,
    tail_rows: usize,
) ![]f32 {
    var lhs = try Tensor.fromSlice(allocator, &.{ tail_rows, split_test_k }, tail_values[0 .. tail_rows * split_test_k]);
    defer lhs.deinit();
    const blocks_per_row = try quant.blockCountForDType(.q8_k, split_test_k);
    const qlhs = try allocator.alloc(quant.BlockQ8_Kx4, ((tail_rows + 3) / 4) * blocks_per_row);
    defer allocator.free(qlhs);
    try quant.quantizeRowsQ8_Kx4PaddedInto(qlhs, &lhs);
    const out = try allocator.alloc(f32, tail_rows * split_test_n);
    errdefer allocator.free(out);
    vector.matmul2DQ4_Kx8Q8_Kx4RhsIntoWithConfig(out, qlhs, rhs, tail_rows, split_test_n, split_test_k, .{});
    return out;
}

fn buildSplitRhsQ4_Kx8(allocator: std.mem.Allocator, random: std.Random) !quant.QuantizedMatmulRhsQ4_Kx8 {
    const blocks_per_row = try quant.blockCountForDType(.q4_k, split_test_k);
    const blocks = try allocator.alloc(quant.BlockQ4_K, split_test_n * blocks_per_row);
    defer allocator.free(blocks);
    var values: [256]f32 = undefined;
    for (blocks) |*block| {
        fillSplitTestValues(&values, random);
        quant.quantizeBlockQ4_KInto(block, &values);
    }
    return quant.packMatmulRhsQ4_Kx8(allocator, blocks, split_test_n, split_test_k, blocks_per_row);
}

fn buildSplitRhsQ5_Kx8(allocator: std.mem.Allocator, random: std.Random) !quant.QuantizedMatmulRhsQ5_Kx8 {
    const blocks_per_row = try quant.blockCountForDType(.q5_k, split_test_k);
    const blocks = try allocator.alloc(quant.BlockQ5_K, split_test_n * blocks_per_row);
    defer allocator.free(blocks);
    var values: [256]f32 = undefined;
    for (blocks) |*block| {
        fillSplitTestValues(&values, random);
        quant.quantizeBlockQ5_KInto(block, &values);
    }
    return quant.packMatmulRhsQ5_Kx8(allocator, blocks, split_test_n, split_test_k, blocks_per_row);
}

fn buildSplitRhsQ8_0x4(allocator: std.mem.Allocator, random: std.Random) !quant.QuantizedMatmulRhsQ8_0x4 {
    const blocks_per_row = try quant.q8_0BlockCount(split_test_k);
    const values = try allocator.alloc(f32, split_test_n * split_test_k);
    defer allocator.free(values);
    fillSplitTestValues(values, random);
    var weights = try Tensor.fromSlice(allocator, &.{ split_test_n, split_test_k }, values);
    defer weights.deinit();
    const blocks = try allocator.alloc(quant.BlockQ8_0, split_test_n * blocks_per_row);
    defer allocator.free(blocks);
    try quant.quantizeRowsQ8_0Into(blocks, &weights);
    return quant.packMatmulRhsQ8_0x4(allocator, blocks, split_test_n, split_test_k, blocks_per_row);
}

test "native q5_k x8 dispatch splits off-multiple m into x4 bulk plus row-kernel tail" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51c4d0ffbeef5501);
    const random = prng.random();

    var rhs = try buildSplitRhsQ5_Kx8(allocator, random);
    defer rhs.deinit();
    const lhs_values = try allocator.alloc(f32, split_test_max_m * split_test_k);
    defer allocator.free(lhs_values);
    fillSplitTestValues(lhs_values, random);

    for (split_test_ms) |m| {
        const full = try runSplitDispatch(native.matmul2DQuantizedRhsQ5_Kx8WithConfig, allocator, &rhs, lhs_values, m, .{});
        defer allocator.free(full);

        const all_rows = try rowsRefQ8_K(vector.matmul2DQ5_Kx8RhsIntoWithConfig, allocator, &rhs, lhs_values, m);
        defer allocator.free(all_rows);

        if (m < 128) {
            // Below the q5_k prefix floor the row path takes every row.
            try expectBitEqualF32(all_rows, full);
            continue;
        }

        const bulk_rows = m - m % 4;
        const prefix = try runSplitDispatch(native.matmul2DQuantizedRhsQ5_Kx8WithConfig, allocator, &rhs, lhs_values, bulk_rows, .{});
        defer allocator.free(prefix);
        try expectBitEqualF32(prefix, full[0 .. bulk_rows * split_test_n]);

        const tail_ref = try rowsRefQ8_K(vector.matmul2DQ5_Kx8RhsIntoWithConfig, allocator, &rhs, lhs_values[bulk_rows * split_test_k ..], m - bulk_rows);
        defer allocator.free(tail_ref);
        try expectBitEqualF32(tail_ref, full[bulk_rows * split_test_n .. m * split_test_n]);

        try expectSplitApprox(all_rows, full);
    }

    var pool: thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = 4 });
    defer pool.deinit();
    const serial = try runSplitDispatch(native.matmul2DQuantizedRhsQ5_Kx8WithConfig, allocator, &rhs, lhs_values, 129, .{});
    defer allocator.free(serial);
    const pooled = try runSplitDispatch(native.matmul2DQuantizedRhsQ5_Kx8WithConfig, allocator, &rhs, lhs_values, 129, .{ .pool = &pool });
    defer allocator.free(pooled);
    try expectBitEqualF32(serial, pooled);
}

test "native q4_k x8 dispatch runs every off-multiple m through the padded x4 kernel" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x41c4d0ffbeef4401);
    const random = prng.random();

    var rhs = try buildSplitRhsQ4_Kx8(allocator, random);
    defer rhs.deinit();
    const lhs_values = try allocator.alloc(f32, split_test_max_m * split_test_k);
    defer allocator.free(lhs_values);
    fillSplitTestValues(lhs_values, random);

    for (split_test_ms) |m| {
        const full = try runSplitDispatch(native.matmul2DQuantizedRhsQ4_Kx8WithConfig, allocator, &rhs, lhs_values, m, .{});
        defer allocator.free(full);
        const bulk_rows = m - m % 4;

        const prefix = try runSplitDispatch(native.matmul2DQuantizedRhsQ4_Kx8WithConfig, allocator, &rhs, lhs_values, bulk_rows, .{});
        defer allocator.free(prefix);
        try expectBitEqualF32(prefix, full[0 .. bulk_rows * split_test_n]);

        const tail_ref = try paddedTailRefQ4_Kx8(allocator, &rhs, lhs_values[bulk_rows * split_test_k ..], m - bulk_rows);
        defer allocator.free(tail_ref);
        try expectBitEqualF32(tail_ref, full[bulk_rows * split_test_n .. m * split_test_n]);

        const all_rows = try rowsRefQ8_K(vector.matmul2DQ4_Kx8RhsIntoWithConfig, allocator, &rhs, lhs_values, m);
        defer allocator.free(all_rows);
        try expectSplitApprox(all_rows, full);
    }

    var pool: thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = 4 });
    defer pool.deinit();
    const serial = try runSplitDispatch(native.matmul2DQuantizedRhsQ4_Kx8WithConfig, allocator, &rhs, lhs_values, 129, .{});
    defer allocator.free(serial);
    const pooled = try runSplitDispatch(native.matmul2DQuantizedRhsQ4_Kx8WithConfig, allocator, &rhs, lhs_values, 129, .{ .pool = &pool });
    defer allocator.free(pooled);
    try expectBitEqualF32(serial, pooled);
}

test "native q8_0 x4 dispatch splits off-multiple m >= 32 into packed bulk plus row-kernel tail" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x80c4d0ffbeef8001);
    const random = prng.random();

    var rhs = try buildSplitRhsQ8_0x4(allocator, random);
    defer rhs.deinit();
    const lhs_values = try allocator.alloc(f32, split_test_max_m * split_test_k);
    defer allocator.free(lhs_values);
    fillSplitTestValues(lhs_values, random);

    for (split_test_ms) |m| {
        const full = try runSplitDispatch(native.matmul2DQuantizedRhsQ8_0x4WithConfig, allocator, &rhs, lhs_values, m, .{});
        defer allocator.free(full);

        if (m < 12) {
            // Whole matmul on the row path: bit-exact against the row kernel.
            const all_rows = try rowsRefQ8_0(allocator, &rhs, lhs_values, m);
            defer allocator.free(all_rows);
            try expectBitEqualF32(all_rows, full);
            continue;
        }

        if (m >= 32) {
            const bulk_rows = m - m % 4;
            const prefix = try runSplitDispatch(native.matmul2DQuantizedRhsQ8_0x4WithConfig, allocator, &rhs, lhs_values, bulk_rows, .{});
            defer allocator.free(prefix);
            try expectBitEqualF32(prefix, full[0 .. bulk_rows * split_test_n]);

            const tail_ref = try rowsRefQ8_0(allocator, &rhs, lhs_values[bulk_rows * split_test_k ..], m - bulk_rows);
            defer allocator.free(tail_ref);
            try expectBitEqualF32(tail_ref, full[bulk_rows * split_test_n .. m * split_test_n]);
        }

        // 12 <= m < 32 keeps the pre-existing padded x4 path; approx only.
        const all_rows = try rowsRefQ8_0(allocator, &rhs, lhs_values, m);
        defer allocator.free(all_rows);
        try expectSplitApprox(all_rows, full);
    }

    var pool: thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = 4 });
    defer pool.deinit();
    const serial = try runSplitDispatch(native.matmul2DQuantizedRhsQ8_0x4WithConfig, allocator, &rhs, lhs_values, 129, .{});
    defer allocator.free(serial);
    const pooled = try runSplitDispatch(native.matmul2DQuantizedRhsQ8_0x4WithConfig, allocator, &rhs, lhs_values, 129, .{ .pool = &pool });
    defer allocator.free(pooled);
    try expectBitEqualF32(serial, pooled);
}
