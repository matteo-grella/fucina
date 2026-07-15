//! Behavioral tests for the hot TQ2_0 ternary kernels (`ternary.zig`):
//! encoder parity vs an independent scalar replica of the ggml reference
//! algorithm, exact round-trips for genuinely ternary rows, b1.58 absmean
//! encoding, hot-vs-cold bitwise matmul parity (the cold table path is the
//! ggml-parity scalar reference), tile-split equivalence (what the parallel
//! dispatch executes), and the mul-free f32 path against both an
//! order-matched scalar replica (bit-exact) and a dequantized dot (tolerance).
const std = @import("std");
const tensor = @import("../../tensor.zig");
const qm = @import("../quant.zig");
const common = @import("common.zig");
const ternary = @import("ternary.zig");

const Tensor = tensor.Tensor;

const BlockTQ2_0 = qm.BlockTQ2_0;
const BlockQ8_K = qm.BlockQ8_K;
const f16BitsToF32 = common.f16BitsToF32;
const qk_k_block_size = qm.qk_k_block_size;
const quantizeRowsQ8_K = qm.quantizeRowsQ8_K;

fn fillUniform(prng: *std.Random.DefaultPrng, values: []f32, scale: f32) void {
    const random = prng.random();
    for (values) |*v| v.* = (random.float(f32) * 2.0 - 1.0) * scale;
}

// Independent scalar replica of the ggml quantize_row_tq2_0 packing: walks
// elements in linear order and computes byte/crumb positions from the element
// index (the encoder walks bytes and gathers elements), so a transposition
// bug in either cannot cancel out.
fn refEncodeRow(dst: []BlockTQ2_0, src: []const f32) void {
    for (dst, 0..) |*block, bi| {
        const x = src[bi * qk_k_block_size ..][0..qk_k_block_size];
        var amax: f32 = 0;
        for (x) |v| amax = @max(amax, @abs(v));
        const id: f32 = if (amax != 0) 1.0 / amax else 0.0;
        block.d = common.f32ToF16Bits(amax);
        @memset(&block.qs, 0);
        for (x, 0..) |v, e| {
            const group = e / 128;
            const within = e % 128;
            const lane = within / 32;
            const byte = within % 32;
            const xi: i32 = @intFromFloat(@round(v * id));
            const code: u8 = @intCast((xi + 1) & 3);
            block.qs[group * 32 + byte] |= code << @intCast(2 * lane);
        }
    }
}

test "tq2_0 encoder matches an independent scalar replica" {
    const allocator = std.testing.allocator;
    const k = 2 * qk_k_block_size;
    var prng = std.Random.DefaultPrng.init(0x7e51);
    const x = try allocator.alloc(f32, k);
    defer allocator.free(x);
    fillUniform(&prng, x, 2.5);

    var got: [2]BlockTQ2_0 = undefined;
    var want: [2]BlockTQ2_0 = undefined;
    try ternary.quantizeRowTQ2_0Into(&got, x);
    refEncodeRow(&want, x);

    for (got, want) |g, w| {
        try std.testing.expectEqual(w.d, g.d);
        try std.testing.expectEqualSlices(u8, &w.qs, &g.qs);
    }
}

test "tq2_0 encoder round-trips exact ternary rows" {
    const allocator = std.testing.allocator;
    const k = qk_k_block_size;
    const scale: f32 = 0.03125; // exactly representable in f16
    var prng = std.Random.DefaultPrng.init(0x7e52);
    const random = prng.random();

    const x = try allocator.alloc(f32, k);
    defer allocator.free(x);
    for (x) |*v| {
        const trit: f32 = @floatFromInt(random.intRangeAtMost(i8, -1, 1));
        v.* = trit * scale;
    }

    var blocks: [1]BlockTQ2_0 = undefined;
    try ternary.quantizeRowTQ2_0Into(&blocks, x);

    const decoded = try allocator.alloc(f32, k);
    defer allocator.free(decoded);
    try qm.dequantizeRowForDType(.tq2_0, decoded, &blocks);
    try std.testing.expectEqualSlices(f32, x, decoded);
}

test "tq2_0 scaled encoder clips to the ternary range" {
    const k = qk_k_block_size;
    var x: [k]f32 = undefined;
    for (&x, 0..) |*v, i| {
        v.* = switch (i % 5) {
            0 => 4.0, // clips to +1
            1 => -7.5, // clips to -1
            2 => 0.2, // rounds to 0 at d=1
            3 => -0.8, // rounds to -1
            else => 0.0,
        };
    }
    var blocks: [1]BlockTQ2_0 = undefined;
    try ternary.quantizeRowTQ2_0ScaledInto(&blocks, &x, 1.0);

    var decoded: [k]f32 = undefined;
    try qm.dequantizeRowForDType(.tq2_0, &decoded, &blocks);
    for (decoded, 0..) |v, i| {
        const want: f32 = switch (i % 5) {
            0 => 1.0,
            1 => -1.0,
            2 => 0.0,
            3 => -1.0,
            else => 0.0,
        };
        try std.testing.expectEqual(want, v);
    }
}

test "tq2_0 absmean rhs produces b1.58 blocks" {
    const allocator = std.testing.allocator;
    const k = qk_k_block_size;
    const n = 3;
    var prng = std.Random.DefaultPrng.init(0x7e53);
    const w = try allocator.alloc(f32, n * k);
    defer allocator.free(w);
    fillUniform(&prng, w, 1.0);

    var rhs = try ternary.quantizedMatmulRhsTQ2_0FromF32Absmean(allocator, k, n, w);
    defer rhs.deinit();

    const d_want = common.f32ToF16Bits(ternary.ternaryAbsmeanScale(w));
    const d_f32 = f16BitsToF32(d_want);
    const decoded = try allocator.alloc(f32, k);
    defer allocator.free(decoded);
    for (0..n) |row| {
        const blocks = rhs.columnBlocks(row);
        for (blocks) |b| try std.testing.expectEqual(d_want, b.d);
        try qm.dequantizeRowForDType(.tq2_0, decoded, blocks);
        for (decoded) |v| {
            try std.testing.expect(v == 0.0 or v == d_f32 or v == -d_f32);
        }
    }
}

test "hot tq2_0 matmul matches the cold table path bitwise" {
    const allocator = std.testing.allocator;
    const m = 3;
    const k = 2 * qk_k_block_size;
    const n = 5; // exercises the 4-column block and the tail column

    var prng = std.Random.DefaultPrng.init(0x7e54);
    const w = try allocator.alloc(f32, n * k);
    defer allocator.free(w);
    fillUniform(&prng, w, 1.5);
    const a_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(a_vals);
    fillUniform(&prng, a_vals, 3.0);

    var rhs = try ternary.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w);
    defer rhs.deinit();

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, a_vals);
    defer a.deinit();
    const qlhs = try quantizeRowsQ8_K(allocator, &a);
    defer allocator.free(qlhs);

    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    const want = try allocator.alloc(f32, m * n);
    defer allocator.free(want);

    ternary.matmulTQ2_0RhsRange(got, qlhs, &rhs, m, n, 0, m);
    qm.matmulTableQ8_KRhsRange(.tq2_0, want, qlhs, &rhs, m, n, 0, m);

    try std.testing.expectEqualSlices(f32, want, got);
}

test "tq2_0 tile splits reproduce the full range bitwise" {
    const allocator = std.testing.allocator;
    const m = 4;
    const k = qk_k_block_size;
    const n = 7;

    var prng = std.Random.DefaultPrng.init(0x7e55);
    const w = try allocator.alloc(f32, n * k);
    defer allocator.free(w);
    fillUniform(&prng, w, 1.0);
    const a_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(a_vals);
    fillUniform(&prng, a_vals, 2.0);

    var rhs = try ternary.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w);
    defer rhs.deinit();
    var a = try Tensor.fromSlice(allocator, &.{ m, k }, a_vals);
    defer a.deinit();
    const qlhs = try quantizeRowsQ8_K(allocator, &a);
    defer allocator.free(qlhs);

    const full = try allocator.alloc(f32, m * n);
    defer allocator.free(full);
    const split = try allocator.alloc(f32, m * n);
    defer allocator.free(split);

    ternary.matmulTQ2_0RhsRange(full, qlhs, &rhs, m, n, 0, m);

    // Row split (the m >= vector_column_min_m parallel shape) ...
    ternary.matmulTQ2_0RhsTile(split, qlhs, &rhs, n, 0, 2, 0, n);
    ternary.matmulTQ2_0RhsTile(split, qlhs, &rhs, n, 2, m, 0, n);
    try std.testing.expectEqualSlices(f32, full, split);

    // ... and column split (the decode GEMV shape).
    ternary.matmulTQ2_0RhsTile(split, qlhs, &rhs, n, 0, m, 0, 3);
    ternary.matmulTQ2_0RhsTile(split, qlhs, &rhs, n, 0, m, 3, n);
    try std.testing.expectEqualSlices(f32, full, split);
}

// Order-matched scalar replica of dotTQ2_0F32: same 4-lane accumulator
// structure and the same lane-fold order, so the comparison is bit-exact.
fn refDotF32(wblocks: []const BlockTQ2_0, x: []const f32) f32 {
    var total: f32 = 0;
    for (wblocks, 0..) |*w, bi| {
        const xb = x[bi * qk_k_block_size ..][0..qk_k_block_size];
        var acc = [4]f32{ 0, 0, 0, 0 };
        for ([_]usize{ 0, 32 }) |j| {
            for (0..4) |lane| {
                var m: usize = 0;
                while (m < 32) : (m += 1) {
                    const code: u8 = (w.qs[j + m] >> @intCast(2 * lane)) & 3;
                    const xv = xb[j * 4 + lane * 32 + m];
                    const term: f32 = switch (code) {
                        0 => -xv,
                        1 => 0.0,
                        else => xv,
                    };
                    acc[m % 4] += term;
                }
            }
        }
        const lane_sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        total += f16BitsToF32(w.d) * lane_sum;
    }
    return total;
}

test "mul-free f32 dot matches the order-matched scalar replica bitwise" {
    const allocator = std.testing.allocator;
    const k = 2 * qk_k_block_size;
    var prng = std.Random.DefaultPrng.init(0x7e56);
    const w = try allocator.alloc(f32, k);
    defer allocator.free(w);
    fillUniform(&prng, w, 1.0);
    const x = try allocator.alloc(f32, k);
    defer allocator.free(x);
    fillUniform(&prng, x, 4.0);

    var blocks: [2]BlockTQ2_0 = undefined;
    try ternary.quantizeRowTQ2_0Into(&blocks, w);

    const got = ternary.dotTQ2_0F32(&blocks, x);
    const want = refDotF32(&blocks, x);
    try std.testing.expectEqual(want, got);
}

test "mul-free f32 matmul matches the dequantized reference within tolerance" {
    const allocator = std.testing.allocator;
    const m = 2;
    const k = qk_k_block_size;
    const n = 6;

    var prng = std.Random.DefaultPrng.init(0x7e57);
    const w = try allocator.alloc(f32, n * k);
    defer allocator.free(w);
    fillUniform(&prng, w, 1.0);
    const x = try allocator.alloc(f32, m * k);
    defer allocator.free(x);
    fillUniform(&prng, x, 2.0);

    var rhs = try ternary.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w);
    defer rhs.deinit();

    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    ternary.matmulTQ2_0F32RhsRange(got, x, &rhs, m, n, 0, m);

    const dense = try allocator.alloc(f32, k);
    defer allocator.free(dense);
    for (0..m) |r| {
        for (0..n) |c| {
            try qm.dequantizeRowForDType(.tq2_0, dense, rhs.columnBlocks(c));
            var want: f64 = 0;
            for (dense, x[r * k ..][0..k]) |wv, xv| want += @as(f64, wv) * @as(f64, xv);
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got[r * n + c], 1e-3);
        }
    }
}

test "tq2_0 encoders stay defined on non-finite input" {
    const k = qk_k_block_size;
    var x: [k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = if (i % 2 == 0) 0.5 else -0.5;
    x[0] = std.math.nan(f32);
    x[1] = std.math.inf(f32);
    x[2] = -std.math.inf(f32);

    // ggml variant: amax = inf -> id = 0 -> every product is 0 or NaN; NaN
    // maps to the zero code, so the whole block encodes as zeros (no UB).
    var blocks: [1]BlockTQ2_0 = undefined;
    try ternary.quantizeRowTQ2_0Into(&blocks, &x);
    for (blocks[0].qs) |q| try std.testing.expectEqual(@as(u8, 0b01_01_01_01), q);

    // Explicit-scale variant: NaN -> 0, +/-inf clamp to the rails, finite
    // values round-clip as usual; every crumb stays in {0,1,2}.
    try ternary.quantizeRowTQ2_0ScaledInto(&blocks, &x, 1.0);
    var decoded: [k]f32 = undefined;
    try qm.dequantizeRowForDType(.tq2_0, &decoded, &blocks);
    try std.testing.expectEqual(@as(f32, 0.0), decoded[0]);
    try std.testing.expectEqual(@as(f32, 1.0), decoded[1]);
    try std.testing.expectEqual(@as(f32, -1.0), decoded[2]);
    for (blocks[0].qs) |q| {
        inline for (0..4) |lane| {
            try std.testing.expect(((q >> @intCast(2 * lane)) & 3) <= 2);
        }
    }
}

test "quantizeRowForDType routes tq2_0" {
    var x: [qk_k_block_size]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) - 1.0;
    var blocks: [1]BlockTQ2_0 = undefined;
    try qm.quantizeRowForDType(.tq2_0, &blocks, &x);
    try std.testing.expectEqual(common.f32ToF16Bits(1.0), blocks[0].d);
}

// ---------------- Q2_0 (Bonsai g128) ----------------

const BlockQ2_0 = qm.BlockQ2_0;
const q2_0_block_size = qm.q2_0_block_size;

// Independent scalar replica of the fork's quantize_row_q2_0_ref packing:
// walks elements in linear order and computes byte/shift from the element
// index (the encoder walks the block layout), so a transposition bug in
// either cannot cancel out.
fn refEncodeQ2_0Row(dst: []BlockQ2_0, src: []const f32) void {
    for (dst, 0..) |*block, bi| {
        const x = src[bi * q2_0_block_size ..][0..q2_0_block_size];
        var amax: f32 = 0;
        for (x) |v| amax = @max(amax, @abs(v));
        const id: f32 = if (amax != 0) 1.0 / amax else 0.0;
        block.d = common.f32ToF16Bits(amax);
        @memset(&block.qs, 0);
        for (x, 0..) |v, e| {
            var q: i32 = @intFromFloat(@round(v * id));
            q += 1;
            if (q < 0) q = 0;
            if (q > 3) q = 3;
            block.qs[e / 4] |= @as(u8, @intCast(q)) << @intCast((e % 4) * 2);
        }
    }
}

test "q2_0 encoder matches an independent scalar replica" {
    const allocator = std.testing.allocator;
    const k = 4 * q2_0_block_size;
    var prng = std.Random.DefaultPrng.init(0x2b01);
    const x = try allocator.alloc(f32, k);
    defer allocator.free(x);
    fillUniform(&prng, x, 2.5);

    var got: [4]BlockQ2_0 = undefined;
    var want: [4]BlockQ2_0 = undefined;
    try qm.quantizeRowQ2_0Into(&got, x);
    refEncodeQ2_0Row(&want, x);
    for (got, want) |g, w| {
        try std.testing.expectEqual(w.d, g.d);
        try std.testing.expectEqualSlices(u8, &w.qs, &g.qs);
    }
}

test "q2_0 encoder round-trips exact ternary rows" {
    const k = q2_0_block_size;
    var x: [k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 3)) - 1.0) * 0.5; // {-0.5, 0, +0.5}
    var blocks: [1]BlockQ2_0 = undefined;
    try qm.quantizeRowQ2_0Into(&blocks, &x);
    var decoded: [k]f32 = undefined;
    try qm.dequantizeRowQ2_0Into(&decoded, &blocks);
    try std.testing.expectEqualSlices(f32, &x, &decoded);
}

fn q2_0RhsFromF32(allocator: std.mem.Allocator, k: usize, n: usize, w: []const f32) !struct {
    blocks: []BlockQ2_0,
    rhs: qm.QuantizedMatmulRhsQ2_0,
} {
    const blocks_per_row = k / q2_0_block_size;
    const blocks = try allocator.alloc(BlockQ2_0, n * blocks_per_row);
    errdefer allocator.free(blocks);
    for (0..n) |row| {
        try qm.quantizeRowQ2_0Into(
            blocks[row * blocks_per_row ..][0..blocks_per_row],
            w[row * k ..][0..k],
        );
    }
    return .{
        .blocks = blocks,
        .rhs = .{
            .rows = .{ .allocator = null, .blocks = blocks, .rows = n, .cols = k, .blocks_per_row = blocks_per_row },
            .k = k,
            .n = n,
        },
    };
}

test "hot q2_0 matmul matches the cold table path bitwise" {
    const allocator = std.testing.allocator;
    const m = 3; // exercises the 2-row micro-tile and the 1-row tail
    const k = 3 * q2_0_block_size;
    const n = 5; // exercises the 4-column block and the tail column

    var prng = std.Random.DefaultPrng.init(0x2b02);
    const w = try allocator.alloc(f32, n * k);
    defer allocator.free(w);
    fillUniform(&prng, w, 1.5);
    const a_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(a_vals);
    fillUniform(&prng, a_vals, 3.0);

    var packed_rhs = try q2_0RhsFromF32(allocator, k, n, w);
    defer allocator.free(packed_rhs.blocks);

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, a_vals);
    defer a.deinit();
    var qlhs = try qm.quantizeRowsQ8_0(allocator, &a);
    defer qlhs.deinit();

    const got = try allocator.alloc(f32, m * n);
    defer allocator.free(got);
    const want = try allocator.alloc(f32, m * n);
    defer allocator.free(want);

    ternary.matmulQ2_0RhsRange(got, qlhs.blocks, &packed_rhs.rhs, m, n, 0, m);
    qm.matmulTableQ8_0RhsRange(.q2_0, want, qlhs.blocks, &packed_rhs.rhs, m, n, 0, m);

    try std.testing.expectEqualSlices(f32, want, got);
}

test "q2_0 tile splits reproduce the full range bitwise" {
    const allocator = std.testing.allocator;
    const m = 5;
    const k = q2_0_block_size;
    const n = 7;

    var prng = std.Random.DefaultPrng.init(0x2b03);
    const w = try allocator.alloc(f32, n * k);
    defer allocator.free(w);
    fillUniform(&prng, w, 1.0);
    const a_vals = try allocator.alloc(f32, m * k);
    defer allocator.free(a_vals);
    fillUniform(&prng, a_vals, 2.0);

    var packed_rhs = try q2_0RhsFromF32(allocator, k, n, w);
    defer allocator.free(packed_rhs.blocks);
    var a = try Tensor.fromSlice(allocator, &.{ m, k }, a_vals);
    defer a.deinit();
    var qlhs = try qm.quantizeRowsQ8_0(allocator, &a);
    defer qlhs.deinit();

    const full = try allocator.alloc(f32, m * n);
    defer allocator.free(full);
    const split = try allocator.alloc(f32, m * n);
    defer allocator.free(split);

    ternary.matmulQ2_0RhsRange(full, qlhs.blocks, &packed_rhs.rhs, m, n, 0, m);

    // Row split landing mid row-pair (the parallel dispatch's range shape).
    ternary.matmulQ2_0RhsTile(split, qlhs.blocks, &packed_rhs.rhs, n, 0, 3, 0, n);
    ternary.matmulQ2_0RhsTile(split, qlhs.blocks, &packed_rhs.rhs, n, 3, m, 0, n);
    try std.testing.expectEqualSlices(f32, full, split);

    // ... and column split (the decode GEMV shape).
    ternary.matmulQ2_0RhsTile(split, qlhs.blocks, &packed_rhs.rhs, n, 0, m, 0, 3);
    ternary.matmulQ2_0RhsTile(split, qlhs.blocks, &packed_rhs.rhs, n, 0, m, 3, n);
    try std.testing.expectEqualSlices(f32, full, split);
}

test "q2_0 code 3 (+2d) agrees between hot and cold paths" {
    const allocator = std.testing.allocator;
    const k = q2_0_block_size;
    const n = 2;
    const m = 2;

    // Hand-built blocks walking all four codes, including 3 (+2d): the wire
    // contract allows it even though the reference encoder never emits it.
    var blocks: [2]BlockQ2_0 = undefined;
    for (&blocks, 0..) |*b, row| {
        b.d = common.f32ToF16Bits(1.0);
        for (&b.qs, 0..) |*q, i| q.* = @truncate(i *% 57 +% row *% 31 +% 0b11_10_01_00);
    }
    var rhs = qm.QuantizedMatmulRhsQ2_0{
        .rows = .{ .allocator = null, .blocks = &blocks, .rows = n, .cols = k, .blocks_per_row = 1 },
        .k = k,
        .n = n,
    };

    var prng = std.Random.DefaultPrng.init(0x2b04);
    var a_vals: [m * k]f32 = undefined;
    fillUniform(&prng, &a_vals, 2.0);
    var a = try Tensor.fromSlice(allocator, &.{ m, k }, &a_vals);
    defer a.deinit();
    var qlhs = try qm.quantizeRowsQ8_0(allocator, &a);
    defer qlhs.deinit();

    var got: [m * n]f32 = undefined;
    var want: [m * n]f32 = undefined;
    ternary.matmulQ2_0RhsRange(&got, qlhs.blocks, &rhs, m, n, 0, m);
    qm.matmulTableQ8_0RhsRange(.q2_0, &want, qlhs.blocks, &rhs, m, n, 0, m);
    try std.testing.expectEqualSlices(f32, &want, &got);
}

test "quantizeRowForDType and gguf transcode route q2_0" {
    var x: [q2_0_block_size]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) - 1.0;
    var blocks: [1]BlockQ2_0 = undefined;
    try qm.quantizeRowForDType(.q2_0, &blocks, &x);
    try std.testing.expectEqual(common.f32ToF16Bits(1.0), blocks[0].d);
    var decoded: [q2_0_block_size]f32 = undefined;
    try qm.dequantizeRowForDType(.q2_0, &decoded, &blocks);
    try std.testing.expectEqualSlices(f32, &x, &decoded);
}
