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
const dtype_mod = raw_backend.dtype_info;

const Tensor = raw_backend.Tensor;
const native = raw_backend.native_impl;
const qm = raw_backend.quant;
const BlockQ4_K = dtype_mod.BlockQ4_K;
const BlockQ8_K = dtype_mod.BlockQ8_K;
const BlockTQ2_0 = dtype_mod.BlockTQ2_0;

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

        var rhs = try qm.ternary.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w_vals);
        defer rhs.deinit();

        var pack_timer = try Timer.start(io);
        const packed_x4 = try qm.ternary.packMatmulRhsTQ2_0x4(allocator, &rhs);
        defer allocator.free(packed_x4);
        try out.print("{s}: x4 pack {d:.1} ms (load-time, same bytes)\n", .{ shape.name, @as(f64, @floatFromInt(pack_timer.read())) / 1e6 });

        // Second plane for the PTQTP fold pair (distinct bytes so both
        // planes genuinely stream).
        const w2_vals = try allocator.alloc(f32, n * k);
        defer allocator.free(w2_vals);
        for (w2_vals, w_vals) |*v, s| v.* = s * 0.31 + 0.017;
        var rhs2 = try qm.ternary.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w2_vals);
        defer rhs2.deinit();
        const packed_x4b = try qm.ternary.packMatmulRhsTQ2_0x4(allocator, &rhs2);
        defer allocator.free(packed_x4b);
        const folded_pack = try qm.ternary.packMatmulRhsTQ2_0Foldedx4(allocator, &rhs, &rhs2);
        defer allocator.free(folded_pack);

        const q4_blocks = try allocator.alloc(BlockQ4_K, n * bpr);
        defer allocator.free(q4_blocks);
        for (0..n) |row| {
            try qm.q4_k.quantizeRowQ4_KInto(q4_blocks[row * bpr ..][0..bpr], w_vals[row * k ..][0..k]);
        }
        var rhs_q4 = try qm.q8k.quantizedMatmulRhsQ4_KFromBlocks(allocator, k, n, q4_blocks);
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
            const qlhs = try qm.q8k.quantizeRowsQ8_K(allocator, &dense);
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

            const ColdCtx = struct { out: []f32, qlhs: []const BlockQ8_K, rhs: *const qm.types.QuantizedMatmulRhsTQ2_0, m: usize, n: usize };
            const cold = try measure(iters, @max(iters / 20, 2), ColdCtx{ .out = out_cold, .qlhs = qlhs, .rhs = &rhs, .m = m, .n = n }, struct {
                fn run(c: ColdCtx) void {
                    qm.cold.matmulTableQ8_KRhsRange(.tq2_0, c.out, c.qlhs, c.rhs, c.m, c.n, 0, c.m);
                }
            }.run);

            const X4Ctx = struct { out: []f32, qlhs: []const BlockQ8_K, packed_x4: []const qm.types.BlockTQ2_0x4, bpr: usize, m: usize, n: usize };
            const pair = try measurePair(
                iters,
                hot_ns,
                x4_ns,
                ColdCtx{ .out = out_hot, .qlhs = qlhs, .rhs = &rhs, .m = m, .n = n },
                struct {
                    fn run(c: ColdCtx) void {
                        qm.ternary.matmulTQ2_0RhsRange(c.out, c.qlhs, c.rhs, c.m, c.n, 0, c.m);
                    }
                }.run,
                X4Ctx{ .out = out_x4, .qlhs = qlhs, .packed_x4 = packed_x4, .bpr = bpr, .m = m, .n = n },
                struct {
                    fn run(c: X4Ctx) void {
                        qm.ternary.matmulTQ2_0X4RhsRange(c.out, c.qlhs, c.packed_x4, c.bpr, c.n, 0, c.m);
                    }
                }.run,
            );
            const hot = pair.a_us;
            const x4 = pair.b_us;

            const F32Ctx = struct { out: []f32, lhs: []const f32, rhs: *const qm.types.QuantizedMatmulRhsTQ2_0, m: usize, n: usize };
            const f32act = try measure(iters, @max(iters / 20, 2), F32Ctx{ .out = out_f32act, .lhs = lhs_vals, .rhs = &rhs, .m = m, .n = n }, struct {
                fn run(c: F32Ctx) void {
                    qm.ternary.matmulTQ2_0F32RhsRange(c.out, c.lhs, c.rhs, c.m, c.n, 0, c.m);
                }
            }.run);

            const Q4Ctx = struct { out: []f32, qlhs: []const BlockQ8_K, rhs: *const qm.types.QuantizedMatmulRhsQ4_K, m: usize, n: usize };
            const q4 = try measure(iters, @max(iters / 20, 2), Q4Ctx{ .out = out_q4, .qlhs = qlhs, .rhs = &rhs_q4, .m = m, .n = n }, struct {
                fn run(c: Q4Ctx) void {
                    qm.q4_k.matmulQ4_KRhsTile(c.out, c.qlhs, c.rhs, c.n, 0, c.m, 0, c.n);
                }
            }.run);

            const DenseCtx = struct { c: *Tensor, a: *const Tensor, b: *const Tensor, m: usize, n: usize, k: usize };
            const f32ref = try measure(dense_iters, 1, DenseCtx{ .c = &c_dense, .a = &dense, .b = &b_dense, .m = m, .n = n, .k = k }, struct {
                fn run(c: DenseCtx) void {
                    native.kernels.gemm(.{}, .{}, c.c, c.a, c.b, c.m, c.n, c.k);
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

            // PTQTP fold decision pair: today's K=2 fused cost (x4 pass +
            // accumulating pass over the second plane) vs ONE folded pass.
            // Correctness is pinned bitwise in ternary_tests.zig; this pair
            // measures only speed (interleaved, medians).
            const FoldPacks = struct { out: []f32, qlhs: []const BlockQ8_K, a: []const qm.types.BlockTQ2_0x4, b: []const qm.types.BlockTQ2_0x4, folded: []const qm.types.BlockTQ2_0Foldedx4, bpr: usize, m: usize, n: usize };
            const fold_pair = try measurePair(
                iters,
                hot_ns, // reuse the sample buffers
                x4_ns,
                FoldPacks{ .out = out_x4, .qlhs = qlhs, .a = packed_x4, .b = packed_x4b, .folded = folded_pack, .bpr = bpr, .m = m, .n = n },
                struct {
                    fn run(c: FoldPacks) void {
                        qm.ternary.matmulTQ2_0X4RhsTile(c.out, c.qlhs, c.a, c.bpr, c.n, 0, c.m, 0, c.n);
                        qm.ternary.matmulTQ2_0X4RhsTileAcc(c.out, c.qlhs, c.b, c.bpr, c.n, 0, c.m, 0, c.n);
                    }
                }.run,
                FoldPacks{ .out = out_x4, .qlhs = qlhs, .a = packed_x4, .b = packed_x4b, .folded = folded_pack, .bpr = bpr, .m = m, .n = n },
                struct {
                    fn run(c: FoldPacks) void {
                        qm.ternary.matmulTQ2_0FoldedX4RhsRange(c.out, c.qlhs, c.folded, c.bpr, c.n, 0, c.m);
                    }
                }.run,
            );
            try out.print("    fold m={d:<4}: 2-pass {d:.1} us  folded {d:.1} us  {d:.2}x\n", .{ m, fold_pair.a_us, fold_pair.b_us, fold_pair.a_us / fold_pair.b_us });

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

    try teamScaling(allocator, out);

    if (any_mismatch) return error.HotColdParityMismatch;
}

/// Team-level roofline: aggregate weight-stream GB/s of the decode (m=1)
/// Q4_K and hot-TQ2_0 kernels at the streamed-expert shape, vs the
/// same-thread-count pure-read DRAM ceiling. Two parallelism modes:
/// `colsplit` (one shared matrix, threads take column ranges — the dense
/// linear dispatch) and `indep` (one matrix per thread — the pooled
/// per-expert MoE dispatch). Answers the compact-vs-repack regime
/// question: near-ceiling numbers are memory-bound (smaller formats win);
/// low fractions are kernel-bound (faster kernels / wider packs win).
fn teamScaling(allocator: std.mem.Allocator, out: *std.Io.Writer) !void {
    const n: usize = 2048; // streamed-expert gate/up rows
    const k: usize = 4096;
    const bpr = k / 256;
    const reps: usize = 150;
    const thread_counts = [_]usize{ 1, 2, 4, 8 };

    const w_vals = try allocator.alloc(f32, n * k);
    defer allocator.free(w_vals);
    fillWeights(w_vals);
    const q4_blocks = try allocator.alloc(dtype_mod.BlockQ4_K, n * bpr);
    defer allocator.free(q4_blocks);
    for (0..n) |row| try qm.q4_k.quantizeRowQ4_KInto(q4_blocks[row * bpr ..][0..bpr], w_vals[row * k ..][0..k]);
    var rhs_t = try qm.ternary.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w_vals);
    defer rhs_t.deinit();

    const x_vals = try allocator.alloc(f32, k);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, ii| v.* = @floatFromInt(@as(i32, @intCast((ii * 7) % 173)) - 86);
    const qlhs = try allocator.alloc(BlockQ8_K, bpr);
    defer allocator.free(qlhs);
    try qm.q8k.quantizeRowQ8_KInto(qlhs, x_vals);

    const q4_bytes = n * bpr * @sizeOf(dtype_mod.BlockQ4_K);
    const t_bytes = n * bpr * @sizeOf(BlockTQ2_0);

    const Worker = struct {
        kind: enum { q4, tq2 },
        q4_rhs: qm.types.QuantizedMatmulRhsQ4_K,
        t_rhs: *const qm.types.QuantizedMatmulRhsTQ2_0,
        qlhs: []const BlockQ8_K,
        out: []f32,
        c0: usize,
        c1: usize,
        reps: usize,

        fn run(w: *@This()) void {
            for (0..w.reps) |_| {
                switch (w.kind) {
                    .q4 => qm.q4_k.matmulQ4_KRhsTile(w.out, w.qlhs, &w.q4_rhs, w.q4_rhs.n, 0, 1, w.c0, w.c1),
                    .tq2 => qm.ternary.matmulTQ2_0RhsTile(w.out, w.qlhs, w.t_rhs, w.t_rhs.n, 0, 1, w.c0, w.c1),
                }
            }
        }
    };

    try out.print("\nteam scaling @ n={d} k={d}, m=1 (aggregate weight GB/s; reps={d})\n", .{ n, k, reps });
    try out.print("{s:<10} | {s:>3} | {s:>9} | {s:>9} | {s:>9}\n", .{ "mode", "T", "q4_k", "tq2_0", "read-ceil" });
    for (thread_counts) |tc| {
        var ceil_gbps: f64 = 0;
        if (membw.probe(allocator, io, tc, 3, 256)) |r| {
            ceil_gbps = r.gbps;
        } else |_| {}
        inline for ([_][]const u8{ "colsplit", "indep" }) |mode| {
            var rates: [2]f64 = undefined;
            inline for ([_]@TypeOf((Worker{ .kind = .q4, .q4_rhs = undefined, .t_rhs = undefined, .qlhs = undefined, .out = undefined, .c0 = 0, .c1 = 0, .reps = 0 }).kind){ .q4, .tq2 }, 0..) |kind, ki| {
                var workers: [8]Worker = undefined;
                var outs: [8][]f32 = undefined;
                var q4_copies: [8][]dtype_mod.BlockQ4_K = undefined;
                var t_copies: [8]?qm.types.QuantizedMatmulRhsTQ2_0 = .{null} ** 8;
                defer for (0..tc) |ti| {
                    allocator.free(outs[ti]);
                    if (std.mem.eql(u8, mode, "indep")) {
                        allocator.free(q4_copies[ti]);
                        if (t_copies[ti]) |*tr| tr.deinit();
                    }
                };
                for (0..tc) |ti| {
                    outs[ti] = try allocator.alloc(f32, n);
                    var my_q4 = q4_blocks;
                    var my_t: *const qm.types.QuantizedMatmulRhsTQ2_0 = &rhs_t;
                    if (std.mem.eql(u8, mode, "indep")) {
                        q4_copies[ti] = try allocator.dupe(dtype_mod.BlockQ4_K, q4_blocks);
                        my_q4 = q4_copies[ti];
                        t_copies[ti] = try qm.ternary.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, w_vals);
                        my_t = &t_copies[ti].?;
                    } else {
                        q4_copies[ti] = &.{};
                    }
                    const c0 = if (std.mem.eql(u8, mode, "colsplit")) ti * n / tc else 0;
                    const c1 = if (std.mem.eql(u8, mode, "colsplit")) (ti + 1) * n / tc else n;
                    workers[ti] = .{
                        .kind = kind,
                        .q4_rhs = try qm.q8k.quantizedMatmulRhsQ4_KFromBlocks(allocator, k, n, my_q4),
                        .t_rhs = my_t,
                        .qlhs = qlhs,
                        .out = outs[ti],
                        .c0 = c0,
                        .c1 = c1,
                        .reps = reps,
                    };
                }
                defer for (0..tc) |ti| workers[ti].q4_rhs.deinit();
                var threads: [8]std.Thread = undefined;
                var timer = try Timer.start(io);
                for (0..tc) |ti| threads[ti] = try std.Thread.spawn(.{}, Worker.run, .{&workers[ti]});
                for (0..tc) |ti| threads[ti].join();
                const ns = timer.read();
                const per_call: f64 = @floatFromInt(if (kind == .q4) q4_bytes else t_bytes);
                const total = per_call * @as(f64, @floatFromInt(reps)) * (if (std.mem.eql(u8, mode, "indep")) @as(f64, @floatFromInt(tc)) else 1.0);
                rates[ki] = total / @as(f64, @floatFromInt(ns));
            }
            try out.print("{s:<10} | {d:>3} | {d:>9.1} | {d:>9.1} | {d:>9.1}\n", .{ mode, tc, rates[0], rates[1], ceil_gbps });
        }
    }
}
