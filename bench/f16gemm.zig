// Focused microbenchmark for the 16-bit-weight TransB matmul ROUTES used by
// Qwen3 projections, at the exact projection shapes and a configurable M
// (decode m=1..8, prefill m=512). Three arms, each measured as the full
// route a model forward would pay:
//
//   f16    cast A f32->f16 (the route's real per-call cost) + all-f16
//          streaming kernel
//   bf16   mixed f32 x bf16 streaming kernel (A stays f32)
//   f32/BLAS  pre-widened f32 RHS (untimed: models a load-time weight
//          shadow, +2 bytes/weight resident) + the f32 TransB route —
//          cblas sgemm when m>=16, the f32 vector kernel below that
//
// The question this answers: does widen-once + BLAS beat the 16-bit
// streaming kernels anywhere (prefill), and by how much does it lose the
// bandwidth race at decode. Run once per regime:
//
//   zig build bench-f16gemm -Doptimize=ReleaseFast -- [--workers N] [--m M] [--iters N]

const std = @import("std");
const Timer = @import("timer.zig").Timer;
const raw_backend = @import("raw_backend");

const Tensor = raw_backend.Tensor;
const TensorF16 = raw_backend.TensorOf(.f16);
const TensorBf16 = raw_backend.TensorOf(.bf16);
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

    try out.print("16-bit-weight TransB routes  M={d}  workers={d} (+main)  iters={d}  (f32 route: {s})\n", .{
        m, workers, iters, if (m >= 16) "cblas sgemm" else "vector kernel (below BLAS gate m>=16)",
    });
    try out.print("{s:<26} | {s:>10} | {s:>10} | {s:>10} | {s:>9} | {s:>9}\n", .{
        "shape", "f16 us", "bf16 us", "f32 us", "best", "f16 GB/s",
    });
    try out.print("{s}\n", .{"-" ** 92});

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = workers });
    defer pool.deinit();
    const cfg: native.ParallelConfig = .{ .pool = &pool };

    var tot = [3]f64{ 0, 0, 0 };

    for (shapes) |s| {
        const a32 = try allocator.alloc(f32, m * s.k);
        defer allocator.free(a32);
        const b16 = try allocator.alloc(f16, s.n * s.k);
        defer allocator.free(b16);
        const bbf = try allocator.alloc(u16, s.n * s.k);
        defer allocator.free(bbf);
        const b32 = try allocator.alloc(f32, s.n * s.k);
        defer allocator.free(b32);
        var prng = std.Random.DefaultPrng.init(0x1234 +% s.n +% s.k);
        const rng = prng.random();
        for (a32) |*v| v.* = rng.float(f32) * 2 - 1;
        for (b16, bbf, b32) |*h, *bb, *f| {
            const value = rng.float(f32) * 0.1 - 0.05;
            h.* = @floatCast(value);
            bb.* = @intCast(@as(u32, @bitCast(value)) >> 16); // truncate: fine for bench data
            f.* = value;
        }
        const a16_scratch = try allocator.alloc(f16, m * s.k);
        defer allocator.free(a16_scratch);

        var a_f32 = try Tensor.fromSlice(allocator, &.{ m, s.k }, a32);
        defer a_f32.deinit();
        var b_f16 = try TensorF16.fromSlice(allocator, &.{ s.n, s.k }, b16);
        defer b_f16.deinit();
        var b_bf16 = try TensorBf16.fromSlice(allocator, &.{ s.n, s.k }, bbf);
        defer b_bf16.deinit();
        var b_f32 = try Tensor.fromSlice(allocator, &.{ s.n, s.k }, b32);
        defer b_f32.deinit();
        var c = try Tensor.zeros(allocator, &.{ m, s.n });
        defer c.deinit();

        const Ctx = struct {
            c: *Tensor,
            a_f32: *const Tensor,
            a16: []f16,
            a32s: []const f32,
            b_f16: *const TensorF16,
            b_bf16: *const TensorBf16,
            b_f32: *const Tensor,
            m: usize,
            n: usize,
            k: usize,
            cfg: native.ParallelConfig,

            fn runF16(self: *const @This()) void {
                // The f16 route's real shape: A is f32 in the model and is
                // cast per call.
                for (self.a16, self.a32s) |*dst, src| dst.* = @floatCast(src);
                var a16_t = TensorF16.fromSlice(std.heap.smp_allocator, &.{ self.m, self.k }, self.a16) catch unreachable;
                defer a16_t.deinit();
                native.matmulTransB2DIntoUncheckedF16OperandsWithConfig(self.c, &a16_t, self.b_f16, self.m, self.n, self.k, self.cfg);
            }
            fn runBf16(self: *const @This()) void {
                native.matmulTransB2DIntoUncheckedBf16RhsWithConfig(self.c, self.a_f32, self.b_bf16, self.m, self.n, self.k, self.cfg);
            }
            fn runF32(self: *const @This()) void {
                native.matmulTransB2DIntoUncheckedWithConfig(self.c, self.a_f32, self.b_f32, self.m, self.n, self.k, self.cfg);
            }
        };
        const ctx = Ctx{
            .c = &c,
            .a_f32 = &a_f32,
            .a16 = a16_scratch,
            .a32s = a32,
            .b_f16 = &b_f16,
            .b_bf16 = &b_bf16,
            .b_f32 = &b_f32,
            .m = m,
            .n = s.n,
            .k = s.k,
            .cfg = cfg,
        };

        const t_f16 = try median(Ctx.runF16, .{&ctx}, iters);
        const t_bf16 = try median(Ctx.runBf16, .{&ctx}, iters);
        const t_f32 = try median(Ctx.runF32, .{&ctx}, iters);

        const best: []const u8 = blk: {
            const min = @min(t_f16, @min(t_bf16, t_f32));
            if (min == t_f16) break :blk "f16";
            if (min == t_bf16) break :blk "bf16";
            break :blk "f32";
        };
        const bytes16 = 2.0 * @as(f64, @floatFromInt(s.n * s.k)); // RHS dominates
        const gbps_f16 = bytes16 / @as(f64, @floatFromInt(t_f16));

        try out.print("{s:<26} | {d:>10.2} | {d:>10.2} | {d:>10.2} | {s:>9} | {d:>9.1}\n", .{
            s.name,
            @as(f64, @floatFromInt(t_f16)) / 1000.0,
            @as(f64, @floatFromInt(t_bf16)) / 1000.0,
            @as(f64, @floatFromInt(t_f32)) / 1000.0,
            best,
            gbps_f16,
        });

        tot[0] += @as(f64, @floatFromInt(t_f16)) * @as(f64, @floatFromInt(s.count));
        tot[1] += @as(f64, @floatFromInt(t_bf16)) * @as(f64, @floatFromInt(s.count));
        tot[2] += @as(f64, @floatFromInt(t_f32)) * @as(f64, @floatFromInt(s.count));
    }

    try out.print("{s}\n", .{"-" ** 92});
    try out.print("est. per-forward projections: f16 {d:.3} ms | bf16 {d:.3} ms | f32 route {d:.3} ms\n", .{
        tot[0] / 1e6, tot[1] / 1e6, tot[2] / 1e6,
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
