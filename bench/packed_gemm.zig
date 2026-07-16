//! Pack-once dense NT GEMM benchmark and regression surface.
//!
//! Covers the skinny-m acceptance matrix. Each row measures both the ordinary
//! TransB dispatcher (which may repack per call in a no-BLAS build) and the
//! load-time packed operation. Packing and allocation are outside the timed
//! region; both paths share the same inputs and are checked numerically.
//!
//!   zig build bench-packed-gemm -Doptimize=ReleaseFast -Dblas=none
//!   zig build bench-packed-gemm -Doptimize=ReleaseFast -- --workers 7
//!   zig build bench-packed-gemm -Doptimize=ReleaseFast -- --only-m 15 --only-k 5120 --packed-first

const std = @import("std");
const Timer = @import("timer.zig").Timer;
const raw_backend = @import("raw_backend");

const Tensor = raw_backend.Tensor;
const native = raw_backend.native_impl;

var io: std.Io = undefined;

const Shape = struct { k: usize, n: usize };
const shapes = [_]Shape{
    .{ .k = 5120, .n = 4800 },
    .{ .k = 4800, .n = 6144 },
    .{ .k = 9600, .n = 6144 },
    .{ .k = 64, .n = 201088 },
};
const rows = [_]usize{ 1, 8, 15, 16, 32, 64 };

pub fn main(init: std.process.Init) !void {
    io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var workers = (std.Thread.getCpuCount() catch 8) -| 1;
    var iters_override: ?usize = null;
    var only_m: ?usize = null;
    var only_k: ?usize = null;
    var packed_first = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--prod-allocator")) continue;
        if (std.mem.eql(u8, args[i], "--workers") and i + 1 < args.len) {
            i += 1;
            workers = try std.fmt.parseInt(usize, args[i], 10);
            continue;
        }
        if (std.mem.eql(u8, args[i], "--iters") and i + 1 < args.len) {
            i += 1;
            iters_override = try std.fmt.parseInt(usize, args[i], 10);
            continue;
        }
        if (std.mem.eql(u8, args[i], "--only-m") and i + 1 < args.len) {
            i += 1;
            only_m = try std.fmt.parseInt(usize, args[i], 10);
            continue;
        }
        if (std.mem.eql(u8, args[i], "--only-k") and i + 1 < args.len) {
            i += 1;
            only_k = try std.fmt.parseInt(usize, args[i], 10);
            continue;
        }
        if (std.mem.eql(u8, args[i], "--packed-first")) {
            packed_first = true;
            continue;
        }
        return error.UnknownArgument;
    }

    const allocator = std.heap.smp_allocator;
    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = workers });
    defer pool.deinit();
    const cfg: native.ParallelConfig = .{ .pool = &pool };

    var buf: [4096]u8 = undefined;
    var sw = std.Io.File.stdout().writer(io, &buf);
    const out = &sw.interface;
    defer out.flush() catch {};
    try out.print("packed dense GEMM  workers={d} (+caller) vector_len={d} blas={s} order={s}\n", .{
        workers,
        native.vector_len,
        @tagName(raw_backend.native_blas_kind),
        if (packed_first) "packed-first" else "generic-first",
    });
    try out.print("backend,k,n,m,path,ns_per_op,checksum,approx_gflops\n", .{});

    for (shapes) |shape| {
        if (only_k) |wanted| if (shape.k != wanted) continue;
        var weight = try Tensor.zeros(allocator, &.{ shape.n, shape.k });
        defer weight.deinit();
        fill(weight.data(), shape.k +% shape.n);
        var packed_rhs = try native.packDenseMatmulRhsTyped(.f32, allocator, &weight);
        defer packed_rhs.deinit();

        for (rows) |m| {
            if (only_m) |wanted| if (m != wanted) continue;
            var lhs = try Tensor.zeros(allocator, &.{ m, shape.k });
            defer lhs.deinit();
            fill(lhs.data(), m +% shape.k);
            var generic_out = try Tensor.zeros(allocator, &.{ m, shape.n });
            defer generic_out.deinit();
            var packed_out = try Tensor.zeros(allocator, &.{ m, shape.n });
            defer packed_out.deinit();

            const iters = iters_override orelse defaultIters(m);
            const generic_args = .{ &generic_out, &lhs, &weight, m, shape.n, shape.k, cfg };
            const packed_args = .{ &packed_out, &lhs, &packed_rhs, m, shape.n, shape.k, cfg };
            var generic_ns: u64 = undefined;
            var packed_ns: u64 = undefined;
            if (packed_first) {
                packed_ns = try median(runPacked, packed_args, iters);
                generic_ns = try median(runGeneric, generic_args, iters);
            } else {
                generic_ns = try median(runGeneric, generic_args, iters);
                packed_ns = try median(runPacked, packed_args, iters);
            }
            var max_abs: f32 = 0;
            for (generic_out.dataConst(), packed_out.dataConst()) |a, b| max_abs = @max(max_abs, @abs(a - b));
            if (!(max_abs <= 1e-2)) {
                try out.print("MISMATCH k={d} n={d} m={d}: max_abs={d}\n", .{ shape.k, shape.n, m, max_abs });
                return error.NumericalMismatch;
            }

            const flops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(shape.n)) * @as(f64, @floatFromInt(shape.k));
            try printRow(out, shape, m, "generic", generic_ns, hash(generic_out.dataConst()), flops);
            try printRow(out, shape, m, "packed", packed_ns, hash(packed_out.dataConst()), flops);
            try out.flush();
        }
    }
}

fn defaultIters(m: usize) usize {
    return if (m <= 8) 7 else if (m <= 16) 5 else if (m <= 32) 4 else 3;
}

fn fill(values: []f32, salt: usize) void {
    for (values, 0..) |*value, index| {
        const q: i32 = @as(i32, @intCast((index *% 17 +% salt) % 251)) - 125;
        value.* = @as(f32, @floatFromInt(q)) / 2048.0;
    }
}

fn runGeneric(out: *Tensor, lhs: *const Tensor, rhs: *const Tensor, m: usize, n: usize, k: usize, cfg: native.ParallelConfig) void {
    out.buffer.waitReady();
    native.matmulTransB2DIntoUncheckedWithConfig(out, lhs, rhs, m, n, k, cfg);
    out.buffer.waitReady();
}

fn runPacked(out: *Tensor, lhs: *const Tensor, rhs: *const raw_backend.PackedDenseRhs, m: usize, n: usize, k: usize, cfg: native.ParallelConfig) void {
    out.buffer.waitReady();
    native.matmul2DIntoUncheckedPackedDenseRhsWithConfig(out, lhs, rhs, m, n, k, cfg) catch @panic("packed GEMM shape failure");
    out.buffer.waitReady();
}

fn median(comptime run: anytype, args: anytype, iters: usize) !u64 {
    @call(.auto, run, args);
    var samples: [64]u64 = undefined;
    const count = @min(iters, samples.len);
    var timer = try Timer.start(io);
    for (0..count) |sample| {
        timer.reset();
        @call(.auto, run, args);
        samples[sample] = timer.read();
    }
    std.mem.sort(u64, samples[0..count], {}, std.sort.asc(u64));
    return samples[count / 2];
}

fn hash(values: []const f32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(values));
}

fn printRow(out: anytype, shape: Shape, m: usize, path: []const u8, ns: u64, checksum: u64, flops: f64) !void {
    try out.print("native,{d},{d},{d},{s},{d},0x{x},{d:.3}\n", .{
        shape.k,
        shape.n,
        m,
        path,
        ns,
        checksum,
        flops / @as(f64, @floatFromInt(ns)),
    });
}
