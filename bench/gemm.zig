// Large-shape f32 GEMM benchmark: register-tiled row kernels (the pre-existing
// pure-Zig path) vs the cache-blocked packed (BLIS-style) kernel, plus the
// native dispatch entry (= vendor CBLAS when built with a BLAS provider) as
// the reference row.
//
//   zig build bench-gemm -Dblas=none -Doptimize=ReleaseFast            # pure-Zig rows
//   zig build bench-gemm -Doptimize=ReleaseFast                        # adds Accelerate as "dispatch"
//   zig build bench-gemm -Dblas=none -Doptimize=ReleaseFast -- --sweep # kc/mc/nc sweep at 2048^3
//   options: --workers N | --iters N | --no-big | --orient | --sweep
//
// All rows run through the same pooled ParallelConfig. GFLOP/s = 2*m*n*k / t.

const std = @import("std");
const Timer = @import("timer.zig").Timer;
const raw_backend = @import("raw_backend");

const Tensor = raw_backend.Tensor;
const native = raw_backend.native_impl;
const vector = raw_backend.vector_impl;
const blocked = vector.gemm_blocked;

var io: std.Io = undefined;

const Shape = struct { name: []const u8, m: usize, n: usize, k: usize, iters: usize, big: bool = false, orient: blocked.Orientation = .nn };

const shapes = [_]Shape{
    .{ .name = "256x256x256 (gate)", .m = 256, .n = 256, .k = 256, .iters = 50 },
    .{ .name = "384x384x384", .m = 384, .n = 384, .k = 384, .iters = 40 },
    .{ .name = "512x512x512", .m = 512, .n = 512, .k = 512, .iters = 30 },
    .{ .name = "640x640x640", .m = 640, .n = 640, .k = 640, .iters = 25 },
    .{ .name = "768x768x768", .m = 768, .n = 768, .k = 768, .iters = 20 },
    .{ .name = "1024x1024x1024", .m = 1024, .n = 1024, .k = 1024, .iters = 15 },
    .{ .name = "2048x2048x2048", .m = 2048, .n = 2048, .k = 2048, .iters = 7 },
    .{ .name = "4096x4096x1024", .m = 4096, .n = 4096, .k = 1024, .iters = 7 },
    .{ .name = "2048x1024x1024 (train)", .m = 2048, .n = 1024, .k = 1024, .iters = 15 },
    // OmniVoice design-F32 hot prefill shapes (TransB, small m, wide n).
    .{ .name = "253x1024x1024 nt (omni)", .m = 253, .n = 1024, .k = 1024, .iters = 30, .orient = .nt },
    .{ .name = "253x2048x1024 nt (omni)", .m = 253, .n = 2048, .k = 1024, .iters = 25, .orient = .nt },
    .{ .name = "253x3072x1024 nt (omni)", .m = 253, .n = 3072, .k = 1024, .iters = 20, .orient = .nt },
    .{ .name = "253x1024x3072 nt (omni)", .m = 253, .n = 1024, .k = 3072, .iters = 20, .orient = .nt },
    .{ .name = "253x8200x1024 nt (omni)", .m = 253, .n = 8200, .k = 1024, .iters = 15, .orient = .nt },
    .{ .name = "2048x151936x1024 (lmhead)", .m = 2048, .n = 151936, .k = 1024, .iters = 2, .big = true },
};

pub fn main(init: std.process.Init) !void {
    io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var workers: usize = std.Thread.getCpuCount() catch 8;
    if (workers > 0) workers -= 1; // main participates
    var iters_override: ?usize = null;
    var include_big = true;
    var sweep = false;
    var sweep_omni = false;
    var orient_mode = false;
    var omni_params_buf: [8]blocked.BlockParams = undefined;
    var omni_params_len: usize = 0;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--workers") and i + 1 < args.len) {
            i += 1;
            workers = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, a, "--iters") and i + 1 < args.len) {
            i += 1;
            iters_override = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, a, "--no-big")) {
            include_big = false;
        } else if (std.mem.eql(u8, a, "--sweep")) {
            sweep = true;
        } else if (std.mem.eql(u8, a, "--sweep-omni")) {
            sweep_omni = true;
        } else if (std.mem.eql(u8, a, "--omni-params") and i + 3 < args.len) {
            if (omni_params_len < omni_params_buf.len) {
                omni_params_buf[omni_params_len] = .{
                    .kc = try std.fmt.parseInt(usize, args[i + 1], 10),
                    .mc = try std.fmt.parseInt(usize, args[i + 2], 10),
                    .nc = try std.fmt.parseInt(usize, args[i + 3], 10),
                };
                omni_params_len += 1;
            }
            i += 3;
        } else if (std.mem.eql(u8, a, "--orient")) {
            orient_mode = true;
        }
    }

    const allocator = std.heap.smp_allocator;

    var buf: [4096]u8 = undefined;
    var sw = std.Io.File.stdout().writer(io, &buf);
    const out = &sw.interface;
    defer out.flush() catch {};

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = workers });
    defer pool.deinit();
    const cfg: native.ParallelConfig = .{ .pool = &pool };

    try out.print("f32 GEMM bench  workers={d} (+main)  vector_len={d}  mr x nr = {d}x{d}  blas={s}\n", .{
        workers,
        native.vector_len,
        blocked.mr,
        blocked.nr,
        @tagName(raw_backend.native_blas_kind),
    });

    if (sweep) {
        try runSweep(allocator, out, cfg, iters_override orelse 5);
        return;
    }
    if (sweep_omni) {
        try runSweepOmni(allocator, out, cfg, iters_override orelse 15);
        return;
    }
    if (omni_params_len > 0) {
        try runOmniParams(allocator, out, cfg, iters_override orelse 20, omni_params_buf[0..omni_params_len]);
        return;
    }
    if (orient_mode) {
        try runOrient(allocator, out, cfg, iters_override orelse 7);
        return;
    }

    if (comptime raw_backend.gpu_impl.enabled) {
        try out.print("{s:<26} | {s:>10} | {s:>10} | {s:>8} | {s:>10} | {s:>10}\n", .{
            "shape (m x n x k)", "rowk GF/s", "blkd GF/s", "speedup", "disp GF/s", "gpu GF/s",
        });
        try out.print("{s}\n", .{"-" ** 91});
    } else {
        try out.print("{s:<26} | {s:>10} | {s:>10} | {s:>8} | {s:>10}\n", .{
            "shape (m x n x k)", "rowk GF/s", "blkd GF/s", "speedup", "disp GF/s",
        });
        try out.print("{s}\n", .{"-" ** 78});
    }

    for (shapes) |s| {
        if (s.big and !include_big) continue;
        const iters = iters_override orelse s.iters;
        const data = allocData(allocator, s.m, s.n, s.k) catch {
            try out.print("{s:<26} | skipped (allocation failed)\n", .{s.name});
            continue;
        };
        defer freeData(allocator, data);

        const flops = 2.0 * @as(f64, @floatFromInt(s.m)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k));

        var rowk_ns: u64 = 0;
        var blkd_ns: u64 = 0;
        var max_abs: f32 = 0;
        switch (s.orient) {
            inline else => |orient| {
                const runners = orientRunners(orient);
                rowk_ns = try median(runners.rowk, .{ data.c, data.a, data.b, s.m, s.n, s.k, cfg }, iters);
                blkd_ns = try median(runners.blkd, .{ data.c, data.a, data.b, s.m, s.n, s.k, cfg }, iters);
                // Sanity: blocked vs row-kernel result (rounding differs; both
                // are k-sequential f32 so the gap stays tiny).
                const ref = try allocator.alloc(f32, s.m * s.n);
                defer allocator.free(ref);
                runners.rowk(ref, data.a, data.b, s.m, s.n, s.k, cfg);
                runners.blkd(data.c, data.a, data.b, s.m, s.n, s.k, cfg);
                for (ref, data.c) |w, g| max_abs = @max(max_abs, @abs(w - g));
            },
        }
        if (!(max_abs <= 1e-2)) {
            try out.print("{s:<26} | MISMATCH blocked vs row kernel: max abs diff {d}\n", .{ s.name, max_abs });
            continue;
        }

        // Native dispatch entry: BLAS when compiled in (and the shape
        // qualifies), otherwise the same pure-Zig dispatch as above. The
        // Tensor path here is NN-layout; for transposed shapes reuse the
        // blocked time so the column stays comparable-ish (marked by orient
        // in the shape name).
        var disp_ns: u64 = blkd_ns;
        if (s.orient == .nn) {
            var a_t = try Tensor.fromSlice(allocator, &.{ s.m, s.k }, data.a);
            defer a_t.deinit();
            var b_t = try Tensor.fromSlice(allocator, &.{ s.k, s.n }, data.b);
            defer b_t.deinit();
            var c_t = try Tensor.zeros(allocator, &.{ s.m, s.n });
            defer c_t.deinit();
            const Disp = struct {
                fn go(c: *Tensor, a: *const Tensor, b: *const Tensor, m: usize, n: usize, k: usize, c2: native.ParallelConfig) void {
                    native.matmul2DIntoUncheckedWithConfig(c, a, b, m, n, k, c2);
                }
            }.go;
            disp_ns = try median(Disp, .{ &c_t, &a_t, &b_t, s.m, s.n, s.k, cfg }, iters);
        }

        if (comptime raw_backend.gpu_impl.enabled) {
            // Direct GPU call (heuristic bypassed) so every shape gets a
            // measured GPU number — this is the data the shouldUseGpu
            // threshold is tuned from.
            const Gpu = struct {
                fn go(c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) void {
                    if (!raw_backend.gpu_impl.gemmF32(.nn, a, b, c, m, n, k)) @panic("gpu gemm failed");
                }
            }.go;
            const gpu_ns = try median(Gpu, .{ data.c, data.a, data.b, s.m, s.n, s.k }, iters);
            try out.print("{s:<26} | {d:>10.1} | {d:>10.1} | {d:>7.2}x | {d:>10.1} | {d:>10.1}\n", .{
                s.name,
                flops / @as(f64, @floatFromInt(rowk_ns)),
                flops / @as(f64, @floatFromInt(blkd_ns)),
                @as(f64, @floatFromInt(rowk_ns)) / @as(f64, @floatFromInt(blkd_ns)),
                flops / @as(f64, @floatFromInt(disp_ns)),
                flops / @as(f64, @floatFromInt(gpu_ns)),
            });
        } else {
            try out.print("{s:<26} | {d:>10.1} | {d:>10.1} | {d:>7.2}x | {d:>10.1}\n", .{
                s.name,
                flops / @as(f64, @floatFromInt(rowk_ns)),
                flops / @as(f64, @floatFromInt(blkd_ns)),
                @as(f64, @floatFromInt(rowk_ns)) / @as(f64, @floatFromInt(blkd_ns)),
                flops / @as(f64, @floatFromInt(disp_ns)),
            });
        }
        try out.flush();
    }
}

const Data = struct { a: []f32, b: []f32, c: []f32 };

fn allocData(allocator: std.mem.Allocator, m: usize, n: usize, k: usize) !Data {
    const a = try allocator.alloc(f32, m * k);
    errdefer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    errdefer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    errdefer allocator.free(c);
    var prng = std.Random.DefaultPrng.init(0x5eed +% m +% n +% k);
    const rng = prng.random();
    for (a) |*v| v.* = rng.float(f32) * 2 - 1;
    for (b) |*v| v.* = rng.float(f32) * 0.1 - 0.05;
    @memset(c, 0);
    return .{ .a = a, .b = b, .c = c };
}

fn freeData(allocator: std.mem.Allocator, data: Data) void {
    allocator.free(data.a);
    allocator.free(data.b);
    allocator.free(data.c);
}

// kc/mc/nc sweep for the blocked kernel at 2048^3.
fn runSweep(allocator: std.mem.Allocator, out: anytype, cfg: native.ParallelConfig, iters: usize) !void {
    const m = 2048;
    const n = 2048;
    const k = 2048;
    const data = try allocData(allocator, m, n, k);
    defer freeData(allocator, data);
    const flops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));

    const kcs = [_]usize{ 128, 256, 512 };
    const mcs = [_]usize{ 64, 128, 256 };
    const ncs = [_]usize{ 256, 512, 1024 };

    try out.print("blocked kernel sweep at {d}x{d}x{d}, GFLOP/s (iters={d})\n", .{ m, n, k, iters });
    try out.print("{s:>4} {s:>4} | {s:>8} {s:>8} {s:>8}\n", .{ "kc", "mc", "nc=256", "nc=512", "nc=1024" });
    for (kcs) |kc| {
        for (mcs) |mc| {
            try out.print("{d:>4} {d:>4} |", .{ kc, mc });
            for (ncs) |nc| {
                const Run = struct {
                    fn go(c: []f32, a: []const f32, b: []const f32, c2: native.ParallelConfig, params: blocked.BlockParams) void {
                        blocked.gemmBlockedWithParams(.nn, c, a, b, 2048, 2048, 2048, c2, params);
                    }
                }.go;
                const params: blocked.BlockParams = .{ .kc = kc, .mc = mc, .nc = nc };
                const ns = try median(Run, .{ data.c, data.a, data.b, cfg, params }, iters);
                try out.print(" {d:>8.1}", .{flops / @as(f64, @floatFromInt(ns))});
                try out.flush();
            }
            try out.print("\n", .{});
        }
    }
}

// kc/mc/nc sweep for the blocked kernel at the OmniVoice prefill shape
// (253 x 3072 x 1024, TransB) — the small-m/wide-n regime the 2D cell split
// targets.
fn runSweepOmni(allocator: std.mem.Allocator, out: anytype, cfg: native.ParallelConfig, iters: usize) !void {
    const m = 253;
    const n = 3072;
    const k = 1024;
    const data = try allocData(allocator, m, n, k);
    defer freeData(allocator, data);
    const flops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));

    const kcs = [_]usize{ 128, 256, 384, 512 };
    const mcs = [_]usize{ 72, 96, 128, 192, 256 };
    const ncs = [_]usize{ 512, 1024 };

    try out.print("blocked kernel sweep at {d}x{d}x{d} nt, GFLOP/s (iters={d})\n", .{ m, n, k, iters });
    try out.print("{s:>4} {s:>4} | {s:>8} {s:>8}\n", .{ "kc", "mc", "nc=512", "nc=1024" });
    for (kcs) |kc| {
        for (mcs) |mc| {
            try out.print("{d:>4} {d:>4} |", .{ kc, mc });
            for (ncs) |nc| {
                const Run = struct {
                    fn go(c: []f32, a: []const f32, b: []const f32, c2: native.ParallelConfig, params: blocked.BlockParams) void {
                        blocked.gemmBlockedWithParams(.nt, c, a, b, 253, 3072, 1024, c2, params);
                    }
                }.go;
                const params: blocked.BlockParams = .{ .kc = kc, .mc = mc, .nc = nc };
                const ns = try median(Run, .{ data.c, data.a, data.b, cfg, params }, iters);
                try out.print(" {d:>8.1}", .{flops / @as(f64, @floatFromInt(ns))});
                try out.flush();
            }
            try out.print("\n", .{});
        }
    }
}

// Explicit BlockParams configs over the OmniVoice NT shapes + 2048^3 NN,
// interleaved per shape so candidate configs share thermal state (the box
// heat-soaks across a run; separate runs are not comparable).
fn runOmniParams(allocator: std.mem.Allocator, out: anytype, cfg: native.ParallelConfig, iters: usize, params_list: []const blocked.BlockParams) !void {
    const S = struct { m: usize, n: usize, k: usize, orient: blocked.Orientation };
    const list = [_]S{
        .{ .m = 253, .n = 1024, .k = 1024, .orient = .nt },
        .{ .m = 253, .n = 3072, .k = 1024, .orient = .nt },
        .{ .m = 253, .n = 1024, .k = 3072, .orient = .nt },
        .{ .m = 253, .n = 8200, .k = 1024, .orient = .nt },
        .{ .m = 2048, .n = 2048, .k = 2048, .orient = .nn },
    };
    try out.print("blocked kernel, per-shape interleaved configs (iters={d})\n", .{iters});
    for (list) |s| {
        const data = try allocData(allocator, s.m, s.n, s.k);
        defer freeData(allocator, data);
        const flops = 2.0 * @as(f64, @floatFromInt(s.m)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k));
        for (params_list) |params| {
            var ns: u64 = 0;
            switch (s.orient) {
                inline else => |orient| {
                    const Run = struct {
                        fn go(c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize, c2: native.ParallelConfig, p: blocked.BlockParams) void {
                            blocked.gemmBlockedWithParams(orient, c, a, b, m, n, k, c2, p);
                        }
                    }.go;
                    ns = try median(Run, .{ data.c, data.a, data.b, s.m, s.n, s.k, cfg, params }, iters);
                },
            }
            try out.print("{d}x{d}x{d} {s} kc={d:<3} mc={d:<3} nc={d:<4} | {d:>8.1} GF/s\n", .{
                s.m, s.n, s.k, @tagName(s.orient), params.kc, params.mc, params.nc,
                flops / @as(f64, @floatFromInt(ns)),
            });
            try out.flush();
        }
    }
}

// Orientation comparison (NN / TN / NT) for row-kernel vs blocked.
fn orientRunners(comptime orient: blocked.Orientation) type {
    return struct {
        fn rowk(c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize, c2: native.ParallelConfig) void {
            switch (orient) {
                .nn => vector.gemmNNRowPathWithConfig(c, a, b, m, n, k, c2),
                .tn => vector.gemmTNRowPathWithConfig(c, a, b, m, n, k, c2),
                .nt => vector.gemmNTRowPathWithConfig(c, a, b, m, n, k, c2),
            }
        }
        fn blkd(c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize, c2: native.ParallelConfig) void {
            blocked.gemmBlocked(orient, c, a, b, m, n, k, c2);
        }
    };
}

fn runOrient(allocator: std.mem.Allocator, out: anytype, cfg: native.ParallelConfig, iters: usize) !void {
    const dims = [_]usize{ 1024, 2048 };
    try out.print("{s:<10} {s:<6} | {s:>10} | {s:>10} | {s:>8}\n", .{ "size", "orient", "rowk GF/s", "blkd GF/s", "speedup" });
    try out.print("{s}\n", .{"-" ** 56});
    for (dims) |d| {
        const data = try allocData(allocator, d, d, d);
        defer freeData(allocator, data);
        const df = @as(f64, @floatFromInt(d));
        const flops = 2.0 * df * df * df;

        inline for (.{ blocked.Orientation.nn, blocked.Orientation.tn, blocked.Orientation.nt }) |orient| {
            const runners = orientRunners(orient);
            const rowk_ns = try median(runners.rowk, .{ data.c, data.a, data.b, d, d, d, cfg }, iters);
            const blkd_ns = try median(runners.blkd, .{ data.c, data.a, data.b, d, d, d, cfg }, iters);
            try out.print("{d:<10} {s:<6} | {d:>10.1} | {d:>10.1} | {d:>7.2}x\n", .{
                d,
                @tagName(orient),
                flops / @as(f64, @floatFromInt(rowk_ns)),
                flops / @as(f64, @floatFromInt(blkd_ns)),
                @as(f64, @floatFromInt(rowk_ns)) / @as(f64, @floatFromInt(blkd_ns)),
            });
            try out.flush();
        }
    }
}

fn median(comptime f: anytype, args: anytype, iters: usize) !u64 {
    const warm = @max(@as(usize, 1), iters / 10);
    for (0..warm) |_| @call(.auto, f, args);
    var times: [256]u64 = undefined;
    const count = @min(iters, times.len);
    var timer = try Timer.start(io);
    for (0..count) |i| {
        timer.reset();
        @call(.auto, f, args);
        times[i] = timer.read();
    }
    std.mem.sort(u64, times[0..count], {}, std.sort.asc(u64));
    return times[count / 2];
}
