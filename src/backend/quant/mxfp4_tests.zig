//! MXFP4 hot-kernel tests. The tile is pinned BITWISE against an
//! independent scalar re-derivation of its sdot lane structure (integer
//! lane sums recomputed from nibbles, then the identical f32 vector
//! schedule), and cross-checked against the cold dequantizer through an
//! f64 dense reference within tolerance.

const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const types = @import("types.zig");
const common = @import("common.zig");
const cold = @import("cold.zig");
const mxfp4 = @import("mxfp4.zig");
const tables = @import("../quant_tables.zig");

fn refCell(lhs_row: []const dtype_mod.BlockQ8_0, col: []const dtype_mod.BlockMXFP4) f32 {
    var acc: common.QKV4f32 = @splat(0);
    for (lhs_row, col) |*a, *w| {
        var iacc: [4]i32 = .{ 0, 0, 0, 0 };
        for (0..4) |l| {
            for (0..4) |t| {
                const j = 4 * l + t;
                const lo: i32 = tables.kvalues_mxfp4[w.qs[j] & 0x0f];
                const hi: i32 = tables.kvalues_mxfp4[w.qs[j] >> 4];
                iacc[l] += lo * @as(i32, a.qs[j]) + hi * @as(i32, a.qs[16 + j]);
            }
        }
        const scale: common.QKV4f32 = @splat(common.e8m0ToF32Half(w.e) * common.f16BitsToF32(a.d));
        acc += @as(common.QKV4f32, @floatFromInt(@as(common.QKV4i32, iacc))) * scale;
    }
    return @reduce(.Add, acc);
}

fn randomBlocks(random: std.Random, allocator: std.mem.Allocator, n: usize, bpc: usize) ![]dtype_mod.BlockMXFP4 {
    const blocks = try allocator.alloc(dtype_mod.BlockMXFP4, n * bpc);
    for (blocks, 0..) |*b, i| {
        // Scale exponents span the useful range plus the subnormal edge
        // (e < 2) so the halved-scale fold is exercised where it bends.
        b.e = if (i % 17 == 0) @intCast(i % 2) else 100 + random.uintLessThan(u8, 55);
        for (&b.qs) |*q| q.* = random.int(u8);
    }
    return blocks;
}

fn randomLhs(random: std.Random, allocator: std.mem.Allocator, m: usize, bpc: usize) ![]dtype_mod.BlockQ8_0 {
    const blocks = try allocator.alloc(dtype_mod.BlockQ8_0, m * bpc);
    for (blocks) |*b| {
        b.d = common.f32ToF16Bits(0.001 + random.float(f32) * 0.05);
        // Quantizer domain: our Q8_0 encoder bounds codes to [-127, 127];
        // -128 never occurs in serving, and the x86 sign-transfer path
        // (vpsignb wraps -128) relies on that — same proven-domain
        // contract as the q8_0 sign-trick kernel.
        for (&b.qs) |*q| q.* = @max(@as(i8, -127), random.int(i8));
    }
    return blocks;
}

test "mxfp4 tile matches scalar lane re-derivation bitwise" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x4d584650);
    const random = prng.random();

    // n = 6 exercises the fused four-column body plus the two-column tail;
    // the second case covers the minimal single-block geometry.
    for ([_][3]usize{ .{ 3, 6, 3 }, .{ 1, 1, 1 }, .{ 2, 9, 4 } }) |case| {
        const m, const n, const bpc = .{ case[0], case[1], case[2] };
        const k = bpc * 32;

        const rhs_blocks = try randomBlocks(random, allocator, n, bpc);
        defer allocator.free(rhs_blocks);
        const lhs_blocks = try randomLhs(random, allocator, m, bpc);
        defer allocator.free(lhs_blocks);

        const rhs = types.QuantizedMatmulRhsMXFP4{
            .rows = .{ .allocator = null, .blocks = rhs_blocks, .rows = n, .cols = k, .blocks_per_row = bpc },
            .k = k,
            .n = n,
        };

        const out = try allocator.alloc(f32, m * n);
        defer allocator.free(out);
        mxfp4.matmulMXFP4RhsTile(out, lhs_blocks, &rhs, n, 0, m, 0, n);

        for (0..m) |i| {
            for (0..n) |j| {
                const expected = refCell(lhs_blocks[i * bpc ..][0..bpc], rhs_blocks[j * bpc ..][0..bpc]);
                try std.testing.expectEqual(expected, out[i * n + j]);
            }
        }
    }
}

test "mxfp4 tile matches cold dequantizer through f64 dense reference" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6d786670);
    const random = prng.random();

    const m = 2;
    const n = 5;
    const bpc = 8;
    const k = bpc * 32;

    const rhs_blocks = try randomBlocks(random, allocator, n, bpc);
    defer allocator.free(rhs_blocks);
    const lhs_blocks = try randomLhs(random, allocator, m, bpc);
    defer allocator.free(lhs_blocks);

    const rhs = types.QuantizedMatmulRhsMXFP4{
        .rows = .{ .allocator = null, .blocks = rhs_blocks, .rows = n, .cols = k, .blocks_per_row = bpc },
        .k = k,
        .n = n,
    };

    const out = try allocator.alloc(f32, m * n);
    defer allocator.free(out);
    mxfp4.matmulMXFP4RhsTile(out, lhs_blocks, &rhs, n, 0, m, 0, n);

    const w_dense = try allocator.alloc(f32, k);
    defer allocator.free(w_dense);
    for (0..m) |i| {
        for (0..n) |j| {
            try cold.dequantizeRowMXFP4Into(w_dense, rhs_blocks[j * bpc ..][0..bpc]);
            var expected: f64 = 0;
            for (0..bpc) |b| {
                const a = &lhs_blocks[i * bpc + b];
                const d: f64 = common.f16BitsToF32(a.d);
                for (0..32) |t| {
                    expected += @as(f64, w_dense[b * 32 + t]) * (d * @as(f64, @floatFromInt(a.qs[t])));
                }
            }
            try std.testing.expectApproxEqRel(@as(f32, @floatCast(expected)), out[i * n + j], 1e-4);
        }
    }
}
