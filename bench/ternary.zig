// Focused microbenchmark for the TQ2_0 ternary matmul kernels. Compares the
// hot sdot/vpdpbusd tile path against the cold generic table path it replaced,
// the mul-free f32-activation path (the STE training forward), the Q4_K row
// kernel (the 4-bit workhorse at the same shapes), and the dense f32 GEMM.
// Single-threaded to isolate per-kernel behavior; hot and cold outputs are
// compared element-wise bitwise, and any mismatch fails the run (nonzero
// exit) — the bench doubles as a real ReleaseFast parity gate.
//
//   zig build bench-ternary -Doptimize=ReleaseFast -- [--iters N]

const std = @import("std");
const Timer = @import("timer.zig").Timer;
const raw_backend = @import("raw_backend");

const Tensor = raw_backend.Tensor;
const native = raw_backend.native_impl;
const qm = raw_backend.quantized_matmul;
const BlockQ4_K = qm.BlockQ4_K;
const BlockQ8_K = qm.BlockQ8_K;
const BlockTQ2_0 = qm.BlockTQ2_0;

var io: std.Io = undefined;

const Shape = struct { name: []const u8, n: usize, k: usize };

const shapes = [_]Shape{
    .{ .name = "n=4096 k=4096", .n = 4096, .k = 4096 },
    .{ .name = "n=11008 k=4096", .n = 11008, .k = 4096 },
};

const ms = [_]usize{ 1, 4, 32, 128 };

fn fillWeights(vals: []f32) void {
    for (vals, 0..) |*v, idx| {
        v.* = (@as(f32, @floatFromInt(@as(i32, @intCast((idx * 37) % 2003)) - 1001))) / 1001.0;
    }
}

fn measure(iters: usize, warmup: usize, ctx: anytype, comptime runOne: fn (@TypeOf(ctx)) void) !f64 {
    var w: usize = 0;
    while (w < warmup) : (w += 1) runOne(ctx);
    var t = try Timer.start(io);
    var it: usize = 0;
    while (it < iters) : (it += 1) runOne(ctx);
    const ns = t.read();
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
}

pub fn main(init: std.process.Init) !void {
    io = init.io;
    const allocator = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var iters: usize = 100;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--iters") and i + 1 < args.len) {
            i += 1;
            iters = try std.fmt.parseInt(usize, args[i], 10);
        }
    }

    var buf: [4096]u8 = undefined;
    var sw = std.Io.File.stdout().writer(io, &buf);
    const out = &sw.interface;
    defer out.flush() catch {};

    try out.print("TQ2_0 ternary matmul microbench  iters={d} (single-thread)\n", .{iters});
    try out.print("{s:<15} | {s:>4} | {s:>10} | {s:>10} | {s:>6} | {s:>10} | {s:>10} | {s:>10} | {s:>8}\n", .{
        "shape", "m", "cold us", "hot us", "hot x", "f32act us", "q4_k us", "f32 us", "w GB/s",
    });
    try out.print("{s}\n", .{"-" ** 108});

    var any_mismatch = false;
    for (shapes) |shape| {
        const n = shape.n;
        const k = shape.k;
        const bpr = k / 256;

        const w_vals = try allocator.alloc(f32, n * k);
        defer allocator.free(w_vals);
        fillWeights(w_vals);

        var rhs = try qm.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w_vals);
        defer rhs.deinit();

        const q4_blocks = try allocator.alloc(BlockQ4_K, n * bpr);
        defer allocator.free(q4_blocks);
        for (0..n) |row| {
            try qm.quantizeRowQ4_KInto(q4_blocks[row * bpr ..][0..bpr], w_vals[row * k ..][0..k]);
        }
        var rhs_q4 = try qm.quantizedMatmulRhsQ4_KFromBlocks(allocator, k, n, q4_blocks);
        defer rhs_q4.deinit();

        // Dense NN comparator operand: B laid out [k, n].
        const b_vals = try allocator.alloc(f32, k * n);
        defer allocator.free(b_vals);
        for (0..n) |row| {
            for (0..k) |col| b_vals[col * n + row] = w_vals[row * k + col];
        }

        for (ms) |m| {
            const lhs_vals = try allocator.alloc(f32, m * k);
            defer allocator.free(lhs_vals);
            for (lhs_vals, 0..) |*v, idx| v.* = @floatFromInt(@as(i32, @intCast((idx * 17) % 251)) - 125);
            var dense = try Tensor.fromSlice(allocator, &.{ m, k }, lhs_vals);
            defer dense.deinit();
            const qlhs = try qm.quantizeRowsQ8_K(allocator, &dense);
            defer allocator.free(qlhs);

            var b_dense = try Tensor.fromSlice(allocator, &.{ k, n }, b_vals);
            defer b_dense.deinit();
            var c_dense = try Tensor.zeros(allocator, &.{ m, n });
            defer c_dense.deinit();

            const out_cold = try allocator.alloc(f32, m * n);
            defer allocator.free(out_cold);
            const out_hot = try allocator.alloc(f32, m * n);
            defer allocator.free(out_hot);
            const out_f32act = try allocator.alloc(f32, m * n);
            defer allocator.free(out_f32act);
            const out_q4 = try allocator.alloc(f32, m * n);
            defer allocator.free(out_q4);

            const dense_iters = @max(iters / 10, 3);

            const ColdCtx = struct { out: []f32, qlhs: []const BlockQ8_K, rhs: *const qm.QuantizedMatmulRhsTQ2_0, m: usize, n: usize };
            const cold = try measure(iters, @max(iters / 20, 2), ColdCtx{ .out = out_cold, .qlhs = qlhs, .rhs = &rhs, .m = m, .n = n }, struct {
                fn run(c: ColdCtx) void {
                    qm.matmulTableQ8_KRhsRange(.tq2_0, c.out, c.qlhs, c.rhs, c.m, c.n, 0, c.m);
                }
            }.run);

            const hot = try measure(iters, @max(iters / 20, 2), ColdCtx{ .out = out_hot, .qlhs = qlhs, .rhs = &rhs, .m = m, .n = n }, struct {
                fn run(c: ColdCtx) void {
                    qm.matmulTQ2_0RhsRange(c.out, c.qlhs, c.rhs, c.m, c.n, 0, c.m);
                }
            }.run);

            const F32Ctx = struct { out: []f32, lhs: []const f32, rhs: *const qm.QuantizedMatmulRhsTQ2_0, m: usize, n: usize };
            const f32act = try measure(iters, @max(iters / 20, 2), F32Ctx{ .out = out_f32act, .lhs = lhs_vals, .rhs = &rhs, .m = m, .n = n }, struct {
                fn run(c: F32Ctx) void {
                    qm.matmulTQ2_0F32RhsRange(c.out, c.lhs, c.rhs, c.m, c.n, 0, c.m);
                }
            }.run);

            const Q4Ctx = struct { out: []f32, qlhs: []const BlockQ8_K, rhs: *const qm.QuantizedMatmulRhsQ4_K, m: usize, n: usize };
            const q4 = try measure(iters, @max(iters / 20, 2), Q4Ctx{ .out = out_q4, .qlhs = qlhs, .rhs = &rhs_q4, .m = m, .n = n }, struct {
                fn run(c: Q4Ctx) void {
                    qm.matmulQ4_KRhsTile(c.out, c.qlhs, c.rhs, c.n, 0, c.m, 0, c.n);
                }
            }.run);

            const DenseCtx = struct { c: *Tensor, a: *const Tensor, b: *const Tensor, m: usize, n: usize, k: usize };
            const f32ref = try measure(dense_iters, 1, DenseCtx{ .c = &c_dense, .a = &dense, .b = &b_dense, .m = m, .n = n, .k = k }, struct {
                fn run(c: DenseCtx) void {
                    native.matmul2DIntoUncheckedWithConfig(c.c, c.a, c.b, c.m, c.n, c.k, .{});
                }
            }.run);

            var mismatch = false;
            for (out_cold, out_hot) |cv, hv| {
                if (@as(u32, @bitCast(cv)) != @as(u32, @bitCast(hv))) mismatch = true;
            }
            if (mismatch) any_mismatch = true;

            // Weight-stream bandwidth of the hot kernel (per iteration it reads
            // the packed weights once: n * bpr * 66 bytes).
            const wbytes = @as(f64, @floatFromInt(n * bpr * @sizeOf(BlockTQ2_0)));
            const gbs = wbytes / (hot * 1000.0);

            try out.print("{s:<15} | {d:>4} | {d:>10.1} | {d:>10.1} | {d:>5.2}x | {d:>10.1} | {d:>10.1} | {d:>10.1} | {d:>8.1}{s}\n", .{
                shape.name,          m,          cold, hot, cold / hot, f32act, q4, f32ref, gbs,
                if (mismatch) " HOT/COLD MISMATCH" else "",
            });
        }
        try out.print("{s}\n", .{"-" ** 108});
    }

    if (any_mismatch) return error.HotColdParityMismatch;
}
