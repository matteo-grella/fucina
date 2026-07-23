// Focused microbenchmark for the TQ2_0 ternary matmul kernels. Compares the
// hot sdot/vpdpbusd tile path against the cold generic table path it replaced,
// the x4 column-interleaved packed-RHS candidate (by-element sdot, zero
// per-block reduces — lanes are columns), the mul-free f32-activation path
// (the STE training forward), the Q4_K row kernel (the 4-bit workhorse at the
// same shapes), and the dense f32 GEMM. Single-threaded to isolate per-kernel
// behavior. The hot-vs-x4 DECISION PAIR is timed with per-rep interleaved
// A/B (ordering swapped on odd reps, per-variant medians) so DVFS/thermal
// drift cancels; context columns stay contiguous-block-timed and carry
// ordering bias — never adjudicate a kernel change from them. Hot/cold and
// x4/hot outputs are compared element-wise bitwise, and any mismatch fails
// the run (nonzero exit) — the bench doubles as a real ReleaseFast parity
// gate.
//
// The "w GB/s" column (weight-stream bandwidth of the hot kernel) is put in
// context by a single-thread DRAM read-bandwidth probe (bench/membw.zig) run
// at startup: "%ceil" is the fraction of that measured ceiling the kernel
// sustains — near-100% decode rows are memory-bound and only a smaller
// format can beat them; low percentages point at compute or dispatch.
// --no-roofline skips the probe (and prints "-" in the column).
//
//   zig build bench-ternary -Doptimize=ReleaseFast -- [--iters N] [--no-roofline]

const std = @import("std");
const Timer = @import("timer.zig").Timer;
const membw = @import("membw.zig");
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

fn medianUs(samples: []u64) f64 {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(samples[samples.len / 2])) / 1000.0;
}

/// Interleaved A/B for the decision pair (the bench/gpu_dispatch.zig
/// discipline): the two contenders alternate within every rep and the
/// ordering swaps on odd reps, so DVFS/thermal drift hits both sides
/// equally instead of taxing whichever ran later; per-variant medians
/// resist outliers. Contiguous-block timing (measure) remains for the
/// context variants only — comparisons between THOSE columns carry
/// ordering bias; the a/b pair does not.
fn measurePair(
    reps: usize,
    a_ns: []u64,
    b_ns: []u64,
    ctx_a: anytype,
    comptime runA: fn (@TypeOf(ctx_a)) void,
    ctx_b: anytype,
    comptime runB: fn (@TypeOf(ctx_b)) void,
) !struct { a_us: f64, b_us: f64 } {
    runA(ctx_a); // warm every path once
    runB(ctx_b);
    for (0..4) |_| { // settle into the alternating steady state
        runA(ctx_a);
        runB(ctx_b);
    }
    var t = try Timer.start(io);
    for (0..reps) |rep| {
        if (rep % 2 == 0) {
            t.reset();
            runA(ctx_a);
            a_ns[rep] = t.read();
            t.reset();
            runB(ctx_b);
            b_ns[rep] = t.read();
        } else {
            t.reset();
            runB(ctx_b);
            b_ns[rep] = t.read();
            t.reset();
            runA(ctx_a);
            a_ns[rep] = t.read();
        }
    }
    return .{ .a_us = medianUs(a_ns[0..reps]), .b_us = medianUs(b_ns[0..reps]) };
}

pub fn main(init: std.process.Init) !void {
    io = init.io;
    const allocator = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var iters: usize = 100;
    var no_roofline = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--iters") and i + 1 < args.len) {
            i += 1;
            iters = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--no-roofline")) {
            no_roofline = true;
        }
    }

    var buf: [4096]u8 = undefined;
    var sw = std.Io.File.stdout().writer(io, &buf);
    const out = &sw.interface;
    defer out.flush() catch {};

    // Single-thread ceiling: the bench is single-threaded, so the per-core
    // probe is the right denominator. Probe failure (tight memory) degrades
    // to no roofline, never to a failed bench.
    const ceiling: ?f64 = if (no_roofline) null else blk: {
        const r = membw.probe(allocator, io, 1, membw.default_rounds, membw.default_single_region_mib) catch break :blk null;
        break :blk r.gbps;
    };

    try out.print("TQ2_0 ternary matmul microbench  iters={d} (single-thread)\n", .{iters});
    if (ceiling) |c| {
        try out.print("single-thread DRAM read ceiling ~{d:.1} GB/s (bench/membw.zig probe); %ceil = hot w GB/s vs it\n", .{c});
    }
    try out.print("{s:<15} | {s:>4} | {s:>10} | {s:>10} | {s:>6} | {s:>10} | {s:>6} | {s:>10} | {s:>10} | {s:>10} | {s:>8} | {s:>5}\n", .{
        "shape", "m", "cold us", "hot us", "hot x", "x4 us", "x4 x", "f32act us", "q4_k us", "f32 us", "w GB/s", "%ceil",
    });
    try out.print("{s}\n", .{"-" ** 137});

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

        var pack_timer = try Timer.start(io);
        const packed_x4 = try qm.packMatmulRhsTQ2_0x4(allocator, &rhs);
        defer allocator.free(packed_x4);
        try out.print("{s}: x4 pack {d:.1} ms (load-time, same bytes)\n", .{ shape.name, @as(f64, @floatFromInt(pack_timer.read())) / 1e6 });

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
            const out_x4 = try allocator.alloc(f32, m * n);
            defer allocator.free(out_x4);
            const out_f32act = try allocator.alloc(f32, m * n);
            defer allocator.free(out_f32act);
            const out_q4 = try allocator.alloc(f32, m * n);
            defer allocator.free(out_q4);
            const hot_ns = try allocator.alloc(u64, iters);
            defer allocator.free(hot_ns);
            const x4_ns = try allocator.alloc(u64, iters);
            defer allocator.free(x4_ns);

            const dense_iters = @max(iters / 10, 3);

            const ColdCtx = struct { out: []f32, qlhs: []const BlockQ8_K, rhs: *const qm.QuantizedMatmulRhsTQ2_0, m: usize, n: usize };
            const cold = try measure(iters, @max(iters / 20, 2), ColdCtx{ .out = out_cold, .qlhs = qlhs, .rhs = &rhs, .m = m, .n = n }, struct {
                fn run(c: ColdCtx) void {
                    qm.matmulTableQ8_KRhsRange(.tq2_0, c.out, c.qlhs, c.rhs, c.m, c.n, 0, c.m);
                }
            }.run);

            const X4Ctx = struct { out: []f32, qlhs: []const BlockQ8_K, packed_x4: []const qm.BlockTQ2_0x4, bpr: usize, m: usize, n: usize };
            const pair = try measurePair(
                iters,
                hot_ns,
                x4_ns,
                ColdCtx{ .out = out_hot, .qlhs = qlhs, .rhs = &rhs, .m = m, .n = n },
                struct {
                    fn run(c: ColdCtx) void {
                        qm.matmulTQ2_0RhsRange(c.out, c.qlhs, c.rhs, c.m, c.n, 0, c.m);
                    }
                }.run,
                X4Ctx{ .out = out_x4, .qlhs = qlhs, .packed_x4 = packed_x4, .bpr = bpr, .m = m, .n = n },
                struct {
                    fn run(c: X4Ctx) void {
                        qm.matmulTQ2_0X4RhsRange(c.out, c.qlhs, c.packed_x4, c.bpr, c.n, 0, c.m);
                    }
                }.run,
            );
            const hot = pair.a_us;
            const x4 = pair.b_us;

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
            var x4_mismatch = false;
            for (out_hot, out_x4) |hv, xv| {
                if (@as(u32, @bitCast(hv)) != @as(u32, @bitCast(xv))) x4_mismatch = true;
            }
            if (mismatch or x4_mismatch) any_mismatch = true;

            // Weight-stream bandwidth of the hot kernel (per iteration it reads
            // the packed weights once: n * bpr * 66 bytes).
            const wbytes = @as(f64, @floatFromInt(n * bpr * @sizeOf(BlockTQ2_0)));
            const gbs = wbytes / (hot * 1000.0);

            const marker = if (mismatch and x4_mismatch)
                " HOT/COLD+X4 MISMATCH"
            else if (mismatch)
                " HOT/COLD MISMATCH"
            else if (x4_mismatch)
                " X4/HOT MISMATCH"
            else
                "";
            try out.print("{s:<15} | {d:>4} | {d:>10.1} | {d:>10.1} | {d:>5.2}x | {d:>10.1} | {d:>5.2}x | {d:>10.1} | {d:>10.1} | {d:>10.1} | {d:>8.1} | ", .{
                shape.name, m, cold, hot, cold / hot, x4, hot / x4, f32act, q4, f32ref, gbs,
            });
            if (ceiling) |c| {
                try out.print("{d:>4.0}%{s}\n", .{ gbs / c * 100.0, marker });
            } else {
                try out.print("{s:>5}{s}\n", .{ "-", marker });
            }
        }
        try out.print("{s}\n", .{"-" ** 137});
    }

    if (any_mismatch) return error.HotColdParityMismatch;
}
