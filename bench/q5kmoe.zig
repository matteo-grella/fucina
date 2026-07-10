// Focused microbenchmark for the K-quant compact MoE-expert matmul. Compares the
// per-row column-outer kernel against the 4-row lane-packed Q8_Kx4 column-outer,
// which packs four LHS rows into the sdot lanes so one i32x4 accumulator holds all
// four rows with no per-row horizontal reduction. Runs at the tiny per-expert batch
// sizes that dominate MoE prefill (avg m≈8 at pp128, ≈16 at pp256), single-threaded,
// to isolate the kernel's per-FLOP / L2 behavior from the model and threading.
// Covers both Q5_K and Q6_K experts.
//
//   zig build bench-q5kmoe -Doptimize=ReleaseFast -- [--iters N]

const std = @import("std");
const Timer = @import("timer.zig").Timer;
const raw_backend = @import("raw_backend");

const Tensor = raw_backend.Tensor;
const qm = raw_backend.quantized_matmul;
const BlockQ5_K = qm.BlockQ5_K;
const BlockQ6_K = qm.BlockQ6_K;
const BlockQ8_K = qm.BlockQ8_K;
const BlockQ8_Kx4 = qm.BlockQ8_Kx4;

var io: std.Io = undefined;

const Shape = struct { name: []const u8, n: usize, k: usize };

// Qwen3-30B-A3B expert projections: hidden=2048, moe_intermediate=768.
const shapes = [_]Shape{
    .{ .name = "gate/up (n=768,k=2048)", .n = 768, .k = 2048 },
    .{ .name = "down    (n=2048,k=768)", .n = 2048, .k = 768 },
};

const ms = [_]usize{ 4, 8, 16, 32 };

fn makeRhsQ5(allocator: std.mem.Allocator, k: usize, n: usize) !qm.QuantizedMatmulRhsQ5_K {
    const bpc = k / 256;
    const blocks = try allocator.alloc(BlockQ5_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.dm[0] = @bitCast(@as(f16, 0.05));
        b.dm[1] = @bitCast(@as(f16, 0.02));
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    return qm.quantizedMatmulRhsQ5_KFromBlocks(allocator, k, n, blocks);
}

fn makeRhsQ6(allocator: std.mem.Allocator, k: usize, n: usize) !qm.QuantizedMatmulRhsQ6_K {
    const bpc = k / 256;
    const blocks = try allocator.alloc(BlockQ6_K, n * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, bi| {
        b.d = @bitCast(@as(f16, 0.03));
        for (&b.scales, 0..) |*s, i| s.* = @intCast(@as(i32, @intCast((i * 5 + bi * 3) % 64)) - 32);
        for (&b.ql, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
    }
    return qm.quantizedMatmulRhsQ6_KFromBlocks(allocator, k, n, blocks);
}

fn checksum(out: []const f32) f64 {
    var s: f64 = 0;
    for (out) |v| s += v;
    return s;
}

fn runVariant(
    out: anytype,
    allocator: std.mem.Allocator,
    label: []const u8,
    iters: usize,
    warmup: usize,
    comptime Rhs: type,
    comptime makeRhs: fn (std.mem.Allocator, usize, usize) anyerror!Rhs,
    comptime perRow: fn ([]f32, []const BlockQ8_K, *const Rhs, usize, usize, usize, usize, usize) void,
    comptime x4: fn ([]f32, []const BlockQ8_Kx4, *const Rhs, usize, usize, usize, usize) void,
) !void {
    try out.print("== {s} ==\n", .{label});
    for (shapes) |shape| {
        const k = shape.k;
        const n = shape.n;
        const bpc = k / 256;
        var rhs = try makeRhs(allocator, k, n);
        defer rhs.deinit();

        for (ms) |m| {
            const lhs_vals = try allocator.alloc(f32, m * k);
            defer allocator.free(lhs_vals);
            for (lhs_vals, 0..) |*v, idx| v.* = @floatFromInt(@as(i32, @intCast((idx * 17) % 251)) - 125);
            var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
            defer dense.deinit();

            const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
            defer allocator.free(qlhs);
            const qlhs_x4 = try qm.packRowsQ8_Kx4(allocator, qlhs, m, k, bpc);
            defer allocator.free(qlhs_x4);

            const out_row = try allocator.alloc(f32, m * n);
            defer allocator.free(out_row);
            const out_kx4 = try allocator.alloc(f32, m * n);
            defer allocator.free(out_kx4);

            var sink: f64 = 0;
            var w: usize = 0;
            while (w < warmup) : (w += 1) {
                perRow(out_row, qlhs, &rhs, n, 0, m, 0, n);
                x4(out_kx4, qlhs_x4, &rhs, n, m, 0, n);
            }

            var t = try Timer.start(io);
            var it: usize = 0;
            while (it < iters) : (it += 1) {
                perRow(out_row, qlhs, &rhs, n, 0, m, 0, n);
                sink += out_row[it % out_row.len];
            }
            const row_ns = t.read();

            t.reset();
            it = 0;
            while (it < iters) : (it += 1) {
                x4(out_kx4, qlhs_x4, &rhs, n, m, 0, n);
                sink += out_kx4[it % out_kx4.len];
            }
            const kx4_ns = t.read();

            const cs_row = checksum(out_row);
            const cs_kx4 = checksum(out_kx4);
            const rel = @abs(cs_row - cs_kx4) / @max(1.0, @abs(cs_row));

            const flops = 2.0 * @as(f64, @floatFromInt(m * n * k));
            const row_us = @as(f64, @floatFromInt(row_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
            const kx4_us = @as(f64, @floatFromInt(kx4_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
            const row_gf = flops / (@as(f64, @floatFromInt(row_ns)) / @as(f64, @floatFromInt(iters)));
            const kx4_gf = flops / (@as(f64, @floatFromInt(kx4_ns)) / @as(f64, @floatFromInt(iters)));

            try out.print("{s:<24} | {d:>3} | {d:>11.3} | {d:>11.3} | {d:>7.2}x | {d:>9.1} | {d:>9.1}{s}\n", .{
                shape.name, m, row_us, kx4_us, row_us / kx4_us, row_gf, kx4_gf,
                if (rel > 1e-3) " MISMATCH" else "",
            });
            std.mem.doNotOptimizeAway(sink);
        }
        try out.print("{s}\n", .{"-" ** 92});
    }
}

pub fn main(init: std.process.Init) !void {
    io = init.io;
    const allocator = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var iters: usize = 2000;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--iters") and i + 1 < args.len) {
            i += 1;
            iters = try std.fmt.parseInt(usize, args[i], 10);
        }
    }
    const warmup = @max(iters / 20, 10);

    var buf: [4096]u8 = undefined;
    var sw = std.Io.File.stdout().writer(io, &buf);
    const out = &sw.interface;
    defer out.flush() catch {};

    try out.print("K-quant compact MoE-expert matmul microbench  iters={d} (single-thread)\n", .{iters});
    try out.print("{s:<24} | {s:>3} | {s:>11} | {s:>11} | {s:>8} | {s:>9} | {s:>9}\n", .{
        "shape", "m", "perrow us", "kx4 us", "speedup", "GF/s row", "GF/s kx4",
    });
    try out.print("{s}\n", .{"-" ** 92});

    try runVariant(out, allocator, "Q5_K", iters, warmup, qm.QuantizedMatmulRhsQ5_K, makeRhsQ5, qm.matmulQ5_KRhsCompactColOuter, qm.matmulQ5_KCompactQ8_Kx4ColOuter);
    try runVariant(out, allocator, "Q6_K", iters, warmup, qm.QuantizedMatmulRhsQ6_K, makeRhsQ6, qm.matmulQ6_KRhsCompactColOuter, qm.matmulQ6_KCompactQ8_Kx4ColOuter);
}
