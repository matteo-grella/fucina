//! Paired CPU-BLAS vs eager GPU-call benchmark.  Reports blocking latency,
//! asynchronous submit latency, and queued aggregate throughput with a stable
//! resident RHS.  The GPU calls are still individual eager ops; queueing only
//! defers the host visibility fence.

const std = @import("std");
const Timer = @import("timer.zig").Timer;
const raw_backend = @import("raw_backend");

const Tensor = raw_backend.Tensor;
const gpu = raw_backend.gpu_impl;
const native = raw_backend.native_impl;
const blocked = raw_backend.vector_impl.gemm_blocked;

var io: std.Io = undefined;

const Shape = struct {
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
};

const shapes = [_]Shape{
    .{ .name = "gemv 1x4096x4096", .m = 1, .n = 4096, .k = 4096, .iters = 15 },
    .{ .name = "gemm 256^3", .m = 256, .n = 256, .k = 256, .iters = 15 },
    .{ .name = "gemm 512^3", .m = 512, .n = 512, .k = 512, .iters = 10 },
    .{ .name = "gemm 1024^3", .m = 1024, .n = 1024, .k = 1024, .iters = 6 },
    .{ .name = "gemm 2048x1024x1024", .m = 2048, .n = 1024, .k = 1024, .iters = 4 },
    .{ .name = "gemm 2048^3", .m = 2048, .n = 2048, .k = 2048, .iters = 3 },
};

extern fn cblas_sgemm(
    order: c_int,
    trans_a: c_int,
    trans_b: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: [*]const f32,
    lda: c_int,
    b: [*]const f32,
    ldb: c_int,
    beta: f32,
    c: [*]f32,
    ldc: c_int,
) callconv(.c) void;

const CblasSgemm = *const fn (
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    f32,
    [*]const f32,
    c_int,
    [*]const f32,
    c_int,
    f32,
    [*]f32,
    c_int,
) callconv(.c) void;
var dynamic_blas_lib: ?std.DynLib = null;
var dynamic_cblas: ?CblasSgemm = null;

fn cpuGemm(c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) void {
    if (comptime raw_backend.native_uses_blas) {
        cblas_sgemm(101, 111, 111, @intCast(m), @intCast(n), @intCast(k), 1, a.ptr, @intCast(k), b.ptr, @intCast(n), 0, c.ptr, @intCast(n));
    } else if (dynamic_cblas) |sgemm| {
        sgemm(101, 111, 111, @intCast(m), @intCast(n), @intCast(k), 1, a.ptr, @intCast(k), b.ptr, @intCast(n), 0, c.ptr, @intCast(n));
    } else {
        blocked.gemmBlocked(.nn, c, a, b, m, n, k, .{});
    }
}

pub fn main(init: std.process.Init) !void {
    io = init.io;
    if (comptime !gpu.enabled) @panic("bench-gpu-dispatch requires -Dgpu=metal or -Dgpu=cuda");
    const allocator = std.heap.smp_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var iters_override: ?usize = null;
    var queue_depth: usize = 4;
    var quick = false;
    var shape_filter: ?[]const u8 = null;
    var detail = false;
    var crossover = false;
    var custom_shape: ?Shape = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--iters") and i + 1 < args.len) {
            i += 1;
            iters_override = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--queue") and i + 1 < args.len) {
            i += 1;
            queue_depth = @min(@as(usize, 8), try std.fmt.parseInt(usize, args[i], 10));
        } else if (std.mem.eql(u8, args[i], "--quick")) {
            quick = true;
        } else if (std.mem.eql(u8, args[i], "--shape") and i + 1 < args.len) {
            i += 1;
            shape_filter = args[i];
        } else if (std.mem.eql(u8, args[i], "--detail")) {
            detail = true;
        } else if (std.mem.eql(u8, args[i], "--crossover")) {
            crossover = true;
        } else if (std.mem.eql(u8, args[i], "--dims") and i + 3 < args.len) {
            custom_shape = .{
                .name = "custom",
                .m = try std.fmt.parseInt(usize, args[i + 1], 10),
                .n = try std.fmt.parseInt(usize, args[i + 2], 10),
                .k = try std.fmt.parseInt(usize, args[i + 3], 10),
                .iters = 5,
            };
            i += 3;
        }
    }
    if (queue_depth == 0) return error.InvalidQueueDepth;
    if (comptime !raw_backend.native_uses_blas) {
        if (std.DynLib.open("libblas.so.3")) |lib| {
            dynamic_blas_lib = lib;
            dynamic_cblas = dynamic_blas_lib.?.lookup(CblasSgemm, "cblas_sgemm");
        } else |_| {}
    }
    defer if (dynamic_blas_lib) |*lib| lib.close();

    var buf: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &buf);
    const out = &writer.interface;
    defer out.flush() catch {};
    try out.print("GPU eager-op dispatch  device={s}  cpu={s}  queue={d}\n", .{
        gpu.deviceName() orelse "unavailable",
        if (raw_backend.native_uses_blas) @tagName(raw_backend.native_blas_kind) else if (dynamic_cblas != null) "libblas.so.3" else "zig-blocked",
        queue_depth,
    });
    if (crossover) {
        try out.print("{s:<24} | {s:>26} | {s:>26} | {s:>9} | {s:>9}\n", .{
            "shape", "CPU p10/p50/p90 us", "GPU p10/p50/p90 us", "GPU/CPU", "max abs",
        });
        try out.print("{s}\n", .{"-" ** 106});
    } else {
        try out.print("{s:<24} | {s:>9} | {s:>9} | {s:>9} | {s:>9} | {s:>10} | {s:>9}\n", .{
            "shape", "cpu us", "sync us", "async us", "submit us", "queue GF/s", "max abs",
        });
        try out.print("{s}\n", .{"-" ** 96});
    }

    var matched = false;
    if (custom_shape) |shape| {
        matched = true;
        try benchShape(allocator, out, shape, iters_override orelse shape.iters, queue_depth, detail, crossover);
        try out.flush();
    }
    for (shapes, 0..) |shape, shape_i| {
        if (custom_shape != null) break;
        if (quick and shape_i > 2) break;
        if (shape_filter) |needle| {
            if (std.mem.indexOf(u8, shape.name, needle) == null) continue;
        }
        matched = true;
        try benchShape(allocator, out, shape, iters_override orelse shape.iters, queue_depth, detail, crossover);
        try out.flush();
    }
    if (!matched) return error.NoMatchingShape;
    gpu.traceDump();
}

fn benchShape(allocator: std.mem.Allocator, out: anytype, shape: Shape, iters: usize, queue_depth: usize, detail: bool, crossover: bool) !void {
    const m = shape.m;
    const n = shape.n;
    const k = shape.k;
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const cpu_out = try allocator.alloc(f32, m * n);
    defer allocator.free(cpu_out);
    const sync_out = try allocator.alloc(f32, m * n);
    defer allocator.free(sync_out);
    var prng = std.Random.DefaultPrng.init(0x9e3779b9 +% m +% n +% k);
    const random = prng.random();
    for (a) |*v| v.* = random.float(f32) * 0.5 - 0.25;
    for (b) |*v| v.* = random.float(f32) * 0.125 - 0.0625;
    @memset(cpu_out, 0);
    @memset(sync_out, 0);

    const resident_bytes = gpu.allocResidentBytes(b.len * @sizeOf(f32)) orelse return error.GpuResidentAllocationFailed;
    defer gpu.freeResidentBytes(resident_bytes);
    const resident_b: []f32 = @alignCast(std.mem.bytesAsSlice(f32, resident_bytes));
    @memcpy(resident_b, b);

    var a_tensor = try Tensor.fromBorrowedSlice(allocator, &.{ m, k }, a);
    defer a_tensor.deinit();
    var b_tensor = try Tensor.fromBorrowedSlice(allocator, &.{ k, n }, resident_b);
    defer b_tensor.deinit();
    var async_out = try Tensor.zeros(allocator, &.{ m, n });
    defer async_out.deinit();
    const queued = try allocator.alloc(Tensor, queue_depth);
    defer allocator.free(queued);
    var queued_init: usize = 0;
    defer for (queued[0..queued_init]) |*value| value.deinit();
    for (queued) |*value| {
        value.* = try Tensor.zeros(allocator, &.{ m, n });
        queued_init += 1;
    }

    // Warm every path, including storage wrappers/registration, device-slot growth,
    // pipeline/JIT creation, and the resident RHS prefetch.
    cpuGemm(cpu_out, a, b, m, n, k);
    if (!gpu.gemmF32(.nn, a, resident_b, sync_out, m, n, k)) return error.GpuDispatchFailed;
    if (!gpu.gemmF32Async(.nn, &a_tensor, &b_tensor, &async_out, m, n, k)) return error.GpuDispatchFailed;
    _ = async_out.dataConst();
    if (gpu.shouldUseGpuForRhs(&b_tensor, m, n, k)) {
        var routed = try Tensor.zeros(allocator, &.{ m, n });
        defer routed.deinit();
        native.matmul2DIntoUnchecked(&routed, &a_tensor, &b_tensor, m, n, k);
        if (routed.buffer.pending() == null) return error.EligibleResidentOpDidNotRouteToGpu;
        _ = routed.dataConst();
    }
    // Let CPU BLAS, GPU clocks, and every storage/slot cache reach steady
    // state before collecting a crossover-sensitive distribution.
    for (0..4) |_| {
        cpuGemm(cpu_out, a, b, m, n, k);
        if (!gpu.gemmF32Async(.nn, &a_tensor, &b_tensor, &async_out, m, n, k)) return error.GpuDispatchFailed;
        _ = async_out.dataConst();
    }

    if (crossover) {
        var cpu_pair: [64]u64 = undefined;
        var gpu_pair: [64]u64 = undefined;
        const count = @min(iters, cpu_pair.len);
        if (count == 0) return error.InvalidIterationCount;
        var timer = try Timer.start(io);
        for (0..count) |rep| {
            if (rep % 2 == 0) {
                timer.reset();
                cpuGemm(cpu_out, a, b, m, n, k);
                cpu_pair[rep] = timer.read();
                timer.reset();
                if (!gpu.gemmF32Async(.nn, &a_tensor, &b_tensor, &async_out, m, n, k)) return error.GpuDispatchFailed;
                _ = async_out.dataConst();
                gpu_pair[rep] = timer.read();
            } else {
                timer.reset();
                if (!gpu.gemmF32Async(.nn, &a_tensor, &b_tensor, &async_out, m, n, k)) return error.GpuDispatchFailed;
                _ = async_out.dataConst();
                gpu_pair[rep] = timer.read();
                timer.reset();
                cpuGemm(cpu_out, a, b, m, n, k);
                cpu_pair[rep] = timer.read();
            }
        }
        std.mem.sort(u64, cpu_pair[0..count], {}, std.sort.asc(u64));
        std.mem.sort(u64, gpu_pair[0..count], {}, std.sort.asc(u64));
        var max_abs: f32 = 0;
        for (cpu_out, async_out.dataConst()) |want, got| max_abs = @max(max_abs, @abs(want - got));
        if (!std.math.isFinite(max_abs) or max_abs > 5e-3) return error.CpuBlasParityFailed;
        const cpu_p50 = percentileSorted(cpu_pair[0..count], 50);
        const gpu_p50 = percentileSorted(gpu_pair[0..count], 50);
        try out.print("{s:<24} | {d:>7.1}/{d:>7.1}/{d:>7.1} | {d:>7.1}/{d:>7.1}/{d:>7.1} | {d:>8.3}x | {e:>9.2}\n", .{
            shape.name,
            nsToUs(percentileSorted(cpu_pair[0..count], 10)),
            nsToUs(cpu_p50),
            nsToUs(percentileSorted(cpu_pair[0..count], 90)),
            nsToUs(percentileSorted(gpu_pair[0..count], 10)),
            nsToUs(gpu_p50),
            nsToUs(percentileSorted(gpu_pair[0..count], 90)),
            @as(f64, @floatFromInt(cpu_p50)) / @as(f64, @floatFromInt(gpu_p50)),
            max_abs,
        });
        return;
    }

    var cpu_times: [64]u64 = undefined;
    var sync_times: [64]u64 = undefined;
    var async_times: [64]u64 = undefined;
    var submit_times: [64]u64 = undefined;
    var queue_times: [64]u64 = undefined;
    const count = @min(iters, cpu_times.len);
    if (count == 0) return error.InvalidIterationCount;
    var timer = try Timer.start(io);
    for (0..count) |rep| {
        // Alternate the paired contenders so clock/thermal drift cannot
        // systematically favor whichever one happens to be measured first.
        if (rep % 2 == 0) {
            timer.reset();
            cpuGemm(cpu_out, a, b, m, n, k);
            cpu_times[rep] = timer.read();

            timer.reset();
            if (!gpu.gemmF32Async(.nn, &a_tensor, &b_tensor, &async_out, m, n, k)) return error.GpuDispatchFailed;
            _ = async_out.dataConst();
            async_times[rep] = timer.read();
        } else {
            timer.reset();
            if (!gpu.gemmF32Async(.nn, &a_tensor, &b_tensor, &async_out, m, n, k)) return error.GpuDispatchFailed;
            _ = async_out.dataConst();
            async_times[rep] = timer.read();

            timer.reset();
            cpuGemm(cpu_out, a, b, m, n, k);
            cpu_times[rep] = timer.read();
        }

        timer.reset();
        if (!gpu.gemmF32(.nn, a, resident_b, sync_out, m, n, k)) return error.GpuDispatchFailed;
        sync_times[rep] = timer.read();

        timer.reset();
        if (!gpu.gemmF32Async(.nn, &a_tensor, &b_tensor, &async_out, m, n, k)) return error.GpuDispatchFailed;
        submit_times[rep] = timer.read();
        _ = async_out.dataConst();

        timer.reset();
        for (queued) |*value| {
            if (!gpu.gemmF32Async(.nn, &a_tensor, &b_tensor, value, m, n, k)) return error.GpuDispatchFailed;
        }
        for (queued) |*value| _ = value.dataConst();
        queue_times[rep] = timer.read();
    }

    const cpu_ns = median(cpu_times[0..count]);
    const sync_ns = median(sync_times[0..count]);
    const async_ns = median(async_times[0..count]);
    const submit_ns = median(submit_times[0..count]);
    const queue_ns = median(queue_times[0..count]);
    var max_abs: f32 = 0;
    for (cpu_out, async_out.dataConst()) |want, got| max_abs = @max(max_abs, @abs(want - got));
    if (!std.math.isFinite(max_abs) or max_abs > 5e-3) return error.CpuBlasParityFailed;
    const flops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    try out.print("{s:<24} | {d:>9.1} | {d:>9.1} | {d:>9.1} | {d:>9.1} | {d:>10.1} | {e:>9.2}\n", .{
        shape.name,
        nsToUs(cpu_ns),
        nsToUs(sync_ns),
        nsToUs(async_ns),
        nsToUs(submit_ns),
        flops * @as(f64, @floatFromInt(queue_depth)) / @as(f64, @floatFromInt(queue_ns)),
        max_abs,
    });
    if (detail) {
        try out.print("  distribution (us)       cpu p10/p50/p90={d:.1}/{d:.1}/{d:.1}  async p10/p50/p90={d:.1}/{d:.1}/{d:.1}\n", .{
            nsToUs(percentileSorted(cpu_times[0..count], 10)),
            nsToUs(cpu_ns),
            nsToUs(percentileSorted(cpu_times[0..count], 90)),
            nsToUs(percentileSorted(async_times[0..count], 10)),
            nsToUs(async_ns),
            nsToUs(percentileSorted(async_times[0..count], 90)),
        });
    }
}

fn median(values: []u64) u64 {
    std.mem.sort(u64, values, {}, std.sort.asc(u64));
    return values[values.len / 2];
}

fn percentileSorted(values: []const u64, pct: usize) u64 {
    return values[(values.len - 1) * pct / 100];
}

fn nsToUs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3;
}
