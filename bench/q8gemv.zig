// Focused microbenchmark for the q8_0 skinny-m (decode GEMV) matmul kernels.
// Compares the three live q8_0 paths at the row counts that dominate decode:
//
//   row  matmulQ8_0RhsTile             column-major rhs, per-row lhs — the
//                                      MoE-expert / unpacked path.
//   x4   matmulQ8_0x4RhsTile           interleaved 4-column rhs, per-row lhs —
//                                      what PackedRhs(.q8_0) serves for
//                                      generic linears.
//   x4p  matmulQ8_0x4PackedPaddedRhsRange  4-column rhs AND 4-row lane-packed
//                                      lhs (rows padded to 4) — the split-
//                                      SwiGLU down-projection path.
//
// Single-threaded on purpose: decode GEMV is memory-bound per core, so the
// number that matters is the weight-stream GB/s one core sustains against the
// measured single-thread DRAM wall (bench-membw).
//
//   zig build bench-q8gemv -Doptimize=ReleaseFast -- [--iters N]

const std = @import("std");
const Timer = @import("timer.zig").Timer;
const raw_backend = @import("raw_backend");

const Tensor = raw_backend.Tensor;
const qm = raw_backend.quantized_matmul;
const BlockQ8_0 = qm.BlockQ8_0;
const BlockQ8_0x4 = qm.BlockQ8_0x4;

var io: std.Io = undefined;

const Shape = struct { name: []const u8, n: usize, k: usize };

// Dense-model decode linears: square attention-sized and wide MLP-sized.
const shapes = [_]Shape{
    .{ .name = "n=4096  k=4096", .n = 4096, .k = 4096 },
    .{ .name = "n=11008 k=4096", .n = 11008, .k = 4096 },
};

const ms = [_]usize{ 1, 2, 4, 8 };

fn checksum(out: []const f32) f64 {
    var s: f64 = 0;
    for (out) |v| s += v;
    return s;
}

pub fn main(init: std.process.Init) !void {
    io = init.io;
    const allocator = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var iters: usize = 500;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--iters") and i + 1 < args.len) {
            i += 1;
            iters = try std.fmt.parseInt(usize, args[i], 10);
        }
    }
    const warmup = @max(iters / 20, 5);

    var buf: [4096]u8 = undefined;
    var sw = std.Io.File.stdout().writer(io, &buf);
    const out = &sw.interface;
    defer out.flush() catch {};

    try out.print("q8_0 skinny-m GEMV microbench  iters={d} (single-thread, GB/s = weight stream)\n", .{iters});
    try out.print("{s:<16} | {s:>2} | {s:>10} | {s:>10} | {s:>10} | {s:>8} | {s:>8} | {s:>8}\n", .{
        "shape", "m", "row us", "x4 us", "x4p us", "row GB/s", "x4 GB/s", "x4p GB/s",
    });
    try out.print("{s}\n", .{"-" ** 96});

    for (shapes) |shape| {
        const n = shape.n;
        const k = shape.k;
        const bpr = k / 32;

        const rhs_vals = try allocator.alloc(f32, k * n);
        defer allocator.free(rhs_vals);
        var prng = std.Random.DefaultPrng.init(42);
        const rand = prng.random();
        for (rhs_vals) |*v| v.* = (rand.float(f32) - 0.5) * 0.2;
        var rhs_dense = try Tensor.fromSlice(allocator, &.{ k, n }, rhs_vals);
        defer rhs_dense.deinit();

        var rhs_row = try qm.quantizeMatmulRhsQ8_0(allocator, &rhs_dense);
        defer rhs_row.deinit();
        var rhs_x4 = try qm.q8_0.packMatmulRhsQ8_0x4(allocator, rhs_row.rows.blocks, n, k, bpr);
        defer rhs_x4.deinit();

        for (ms) |m| {
            const lhs_vals = try allocator.alloc(f32, m * k);
            defer allocator.free(lhs_vals);
            for (lhs_vals) |*v| v.* = (rand.float(f32) - 0.5) * 2.0;
            var lhs_dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
            defer lhs_dense.deinit();

            const qlhs = try allocator.alloc(BlockQ8_0, m * bpr);
            defer allocator.free(qlhs);
            for (0..m) |r| try qm.q8k.quantizeRowQ8_0Into(qlhs[r * bpr ..][0..bpr], lhs_vals[r * k ..][0..k]);

            const row_groups = (m + 3) / 4;
            const qlhs_x4 = try allocator.alloc(BlockQ8_0x4, row_groups * bpr);
            defer allocator.free(qlhs_x4);
            try qm.q8_0.quantizeRowsQ8_0x4PaddedInto(qlhs_x4, &lhs_dense);

            const out_row = try allocator.alloc(f32, m * n);
            defer allocator.free(out_row);
            const out_x4 = try allocator.alloc(f32, m * n);
            defer allocator.free(out_x4);
            const out_x4p = try allocator.alloc(f32, m * n);
            defer allocator.free(out_x4p);

            var sink: f64 = 0;
            var w: usize = 0;
            while (w < warmup) : (w += 1) {
                qm.q8_0.matmulQ8_0RhsTile(out_row, qlhs, &rhs_row, n, 0, m, 0, n);
                qm.q8_0.matmulQ8_0x4RhsTile(out_x4, qlhs, &rhs_x4, n, 0, m, 0, n);
                qm.q8_0.matmulQ8_0x4PackedPaddedRhsRange(out_x4p, qlhs_x4, &rhs_x4, m, n);
            }

            var t = try Timer.start(io);
            var it: usize = 0;
            while (it < iters) : (it += 1) {
                qm.q8_0.matmulQ8_0RhsTile(out_row, qlhs, &rhs_row, n, 0, m, 0, n);
                sink += out_row[it % out_row.len];
            }
            const row_ns = t.read();

            t.reset();
            it = 0;
            while (it < iters) : (it += 1) {
                qm.q8_0.matmulQ8_0x4RhsTile(out_x4, qlhs, &rhs_x4, n, 0, m, 0, n);
                sink += out_x4[it % out_x4.len];
            }
            const x4_ns = t.read();

            t.reset();
            it = 0;
            while (it < iters) : (it += 1) {
                qm.q8_0.matmulQ8_0x4PackedPaddedRhsRange(out_x4p, qlhs_x4, &rhs_x4, m, n);
                sink += out_x4p[it % out_x4p.len];
            }
            const x4p_ns = t.read();
            std.mem.doNotOptimizeAway(sink);

            const cs_row = checksum(out_row);
            const rel_x4 = @abs(checksum(out_x4) - cs_row) / @max(1.0, @abs(cs_row));
            const rel_x4p = @abs(checksum(out_x4p) - cs_row) / @max(1.0, @abs(cs_row));

            const weight_bytes = @as(f64, @floatFromInt(n * bpr * @sizeOf(BlockQ8_0)));
            const fiters: f64 = @floatFromInt(iters);
            const row_us = @as(f64, @floatFromInt(row_ns)) / fiters / 1000.0;
            const x4_us = @as(f64, @floatFromInt(x4_ns)) / fiters / 1000.0;
            const x4p_us = @as(f64, @floatFromInt(x4p_ns)) / fiters / 1000.0;

            try out.print("{s:<16} | {d:>2} | {d:>10.1} | {d:>10.1} | {d:>10.1} | {d:>8.1} | {d:>8.1} | {d:>8.1}{s}{s}\n", .{
                shape.name,                          m,
                row_us,                              x4_us,
                x4p_us,                              weight_bytes / (row_us * 1000.0),
                weight_bytes / (x4_us * 1000.0),     weight_bytes / (x4p_us * 1000.0),
                if (rel_x4 > 1e-3) " X4-MISMATCH" else "",
                if (rel_x4p > 1e-3) " X4P-MISMATCH" else "",
            });
        }
        try out.print("{s}\n", .{"-" ** 96});
    }
}
