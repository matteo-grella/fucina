// Focused microbenchmark for the f16 RHS TransB matmul path used by Qwen3
// projections (M is tiny: prompt/decode regime). Measures serial vs pooled
// throughput and parallel efficiency at the exact projection shapes, so we can
// iterate on the kernel and the thread-scheduling thresholds without rebuilding
// or reloading the full model.
//
//   zig build bench-f16gemm -Doptimize=ReleaseFast -- [--workers N] [--m M] [--iters N]

const std = @import("std");
const Timer = @import("timer.zig").Timer;
const raw_backend = @import("raw_backend");

const Tensor = raw_backend.Tensor;
const TensorF16 = raw_backend.TensorOf(.f16);
const native = raw_backend.native_impl;

var io: std.Io = undefined;

const Shape = struct { name: []const u8, n: usize, k: usize, count: usize };

// Qwen3-0.6B projection shapes (RHS stored [N, K], TransB). count = per forward.
const shapes = [_]Shape{
    .{ .name = "qkv  (n=4096,k=1024)", .n = 4096, .k = 1024, .count = 28 },
    .{ .name = "o    (n=1024,k=2048)", .n = 1024, .k = 2048, .count = 28 },
    .{ .name = "gate_up(n=6144,k=1024)", .n = 6144, .k = 1024, .count = 28 },
    .{ .name = "down (n=1024,k=3072)", .n = 1024, .k = 3072, .count = 28 },
    .{ .name = "lmhead(n=151936,k=1024)", .n = 151936, .k = 1024, .count = 1 },
};

pub fn main(init: std.process.Init) !void {
    io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var workers: usize = std.Thread.getCpuCount() catch 8;
    if (workers > 0) workers -= 1; // main participates
    var m: usize = 4;
    var iters: usize = 200;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--workers") and i + 1 < args.len) {
            i += 1;
            workers = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, a, "--m") and i + 1 < args.len) {
            i += 1;
            m = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, a, "--iters") and i + 1 < args.len) {
            i += 1;
            iters = try std.fmt.parseInt(usize, args[i], 10);
        }
    }

    const allocator = std.heap.smp_allocator;

    var buf: [4096]u8 = undefined;
    var sw = std.Io.File.stdout().writer(io, &buf);
    const out = &sw.interface;
    defer out.flush() catch {};

    try out.print("f16 TransB GEMM microbench  M={d}  workers={d} (+main)  iters={d}\n", .{ m, workers, iters });
    try out.print("{s:<26} | {s:>11} | {s:>11} | {s:>9} | {s:>8} | {s:>9}\n", .{
        "shape", "serial us", "pooled us", "speedup", "GF/s par", "GB/s par",
    });
    try out.print("{s}\n", .{"-" ** 92});

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = workers });
    defer pool.deinit();
    const cfg_par: native.ParallelConfig = .{ .pool = &pool };
    const cfg_ser: native.ParallelConfig = .{};

    var total_serial_ns: f64 = 0;
    var total_pooled_ns: f64 = 0;

    for (shapes) |s| {
        const a_data = try allocator.alloc(f16, m * s.k);
        defer allocator.free(a_data);
        const b_data = try allocator.alloc(f16, s.n * s.k);
        defer allocator.free(b_data);
        var prng = std.Random.DefaultPrng.init(0x1234 +% s.n +% s.k);
        const rng = prng.random();
        for (a_data) |*v| v.* = @floatCast(rng.float(f32) * 2 - 1);
        for (b_data) |*v| v.* = @floatCast(rng.float(f32) * 0.1 - 0.05);

        var a = try TensorF16.fromSlice(allocator, &.{ m, s.k }, a_data);
        defer a.deinit();
        var b = try TensorF16.fromSlice(allocator, &.{ s.n, s.k }, b_data);
        defer b.deinit();
        var c = try Tensor.zeros(allocator, &.{ m, s.n });
        defer c.deinit();

        const Run = struct {
            fn go(o: *Tensor, lhs: *const TensorF16, rhs: *const TensorF16, rows: usize, cols: usize, inner: usize, cfg: native.ParallelConfig) void {
                native.matmulTransB2DIntoUncheckedF16OperandsWithConfig(o, lhs, rhs, rows, cols, inner, cfg);
            }
        }.go;

        const ser = try median(Run, .{ &c, &a, &b, m, s.n, s.k, cfg_ser }, iters);
        const par = try median(Run, .{ &c, &a, &b, m, s.n, s.k, cfg_par }, iters);

        const flops = 2.0 * @as(f64, @floatFromInt(m * s.n * s.k));
        const bytes = 2.0 * @as(f64, @floatFromInt(s.n * s.k)); // RHS dominates
        const gf_par = flops / @as(f64, @floatFromInt(par));
        const gbps_par = bytes / @as(f64, @floatFromInt(par));
        const speedup = @as(f64, @floatFromInt(ser)) / @as(f64, @floatFromInt(par));

        try out.print("{s:<26} | {d:>11.2} | {d:>11.2} | {d:>8.2}x | {d:>8.1} | {d:>9.1}\n", .{
            s.name,
            @as(f64, @floatFromInt(ser)) / 1000.0,
            @as(f64, @floatFromInt(par)) / 1000.0,
            speedup,
            gf_par,
            gbps_par,
        });

        total_serial_ns += @as(f64, @floatFromInt(ser)) * @as(f64, @floatFromInt(s.count));
        total_pooled_ns += @as(f64, @floatFromInt(par)) * @as(f64, @floatFromInt(s.count));
    }

    try out.print("{s}\n", .{"-" ** 92});
    try out.print("est. per-forward projections: serial {d:.3} ms | pooled {d:.3} ms\n", .{
        total_serial_ns / 1e6, total_pooled_ns / 1e6,
    });
}

fn median(comptime f: anytype, args: anytype, iters: usize) !u64 {
    const warm = @max(@as(usize, 2), iters / 20);
    for (0..warm) |_| @call(.auto, f, args);
    var times = try std.heap.page_allocator.alloc(u64, iters);
    defer std.heap.page_allocator.free(times);
    var timer = try Timer.start(io);
    for (0..iters) |i| {
        timer.reset();
        @call(.auto, f, args);
        times[i] = timer.read();
    }
    std.mem.sort(u64, times, {}, std.sort.asc(u64));
    return times[iters / 2];
}
