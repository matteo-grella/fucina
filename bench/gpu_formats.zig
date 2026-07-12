//! Eager f16/quantized LLM-linear benchmark. CPU contenders are Fucina's
//! actual f16 and load-time-packed quant kernels; GPU contenders are individual
//! eagerly submitted calls over resident GGUF-format weights. Reports complete
//! host-visible latency, submit cost, bounded-queue throughput, and parity.

const std = @import("std");
const Timer = @import("timer.zig").Timer;
const raw = @import("raw_backend");

const Tensor = raw.Tensor;
const TensorF16 = raw.TensorOf(.f16);
const gpu = raw.gpu_impl;
const native = raw.native_impl;
const vector = raw.vector_impl;
const qm = raw.quantized_matmul;

var io: std.Io = undefined;

const Shape = struct {
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
};

const f16_shapes = [_]Shape{
    .{ .name = "decode-qkv", .m = 1, .n = 4096, .k = 1024, .iters = 12 },
    .{ .name = "prefill32-qkv", .m = 32, .n = 4096, .k = 1024, .iters = 8 },
    .{ .name = "prefill128-qkv", .m = 128, .n = 4096, .k = 1024, .iters = 5 },
    .{ .name = "decode-lmhead", .m = 1, .n = 151936, .k = 1024, .iters = 4 },
};

const quant_shapes = [_]Shape{
    .{ .name = "decode", .m = 1, .n = 4096, .k = 4096, .iters = 10 },
    .{ .name = "decode2", .m = 2, .n = 4096, .k = 4096, .iters = 8 },
    .{ .name = "decode3", .m = 3, .n = 4096, .k = 4096, .iters = 8 },
    .{ .name = "decode4", .m = 4, .n = 4096, .k = 4096, .iters = 8 },
    .{ .name = "decode5", .m = 5, .n = 4096, .k = 4096, .iters = 8 },
    .{ .name = "decode6", .m = 6, .n = 4096, .k = 4096, .iters = 8 },
    .{ .name = "decode8", .m = 8, .n = 4096, .k = 4096, .iters = 8 },
    .{ .name = "decode-qkv", .m = 1, .n = 6144, .k = 4096, .iters = 8 },
    .{ .name = "decode-ffn", .m = 1, .n = 12288, .k = 4096, .iters = 6 },
    .{ .name = "decode-lmhead", .m = 1, .n = 151936, .k = 1024, .iters = 4 },
    .{ .name = "small32", .m = 32, .n = 1024, .k = 512, .iters = 8 },
    .{ .name = "parakeet32", .m = 32, .n = 1536, .k = 512, .iters = 8 },
    .{ .name = "qwen32", .m = 32, .n = 4096, .k = 1024, .iters = 8 },
    .{ .name = "prefill32", .m = 32, .n = 4096, .k = 4096, .iters = 6 },
    .{ .name = "prefill64", .m = 64, .n = 4096, .k = 4096, .iters = 5 },
    .{ .name = "prefill128", .m = 128, .n = 4096, .k = 4096, .iters = 4 },
};

const Options = struct {
    section: enum { all, f16, quant } = .all,
    format: enum { all, q4_k, q5_k, q6_k, q8_0 } = .all,
    filter: ?[]const u8 = null,
    iters: ?usize = null,
    queue_depth: usize = 4,
    workers: usize = 7,
    f16_resident: bool = true,
};

pub fn main(init: std.process.Init) !void {
    io = init.io;
    if (comptime !gpu.enabled) @panic("bench-gpu-formats requires -Dgpu=metal or -Dgpu=cuda");
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var opts: Options = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--section") and i + 1 < args.len) {
            i += 1;
            opts.section = std.meta.stringToEnum(@TypeOf(opts.section), args[i]) orelse return error.InvalidSection;
        } else if (std.mem.eql(u8, args[i], "--format") and i + 1 < args.len) {
            i += 1;
            opts.format = std.meta.stringToEnum(@TypeOf(opts.format), args[i]) orelse return error.InvalidFormat;
        } else if (std.mem.eql(u8, args[i], "--shape") and i + 1 < args.len) {
            i += 1;
            opts.filter = args[i];
        } else if (std.mem.eql(u8, args[i], "--iters") and i + 1 < args.len) {
            i += 1;
            opts.iters = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--queue") and i + 1 < args.len) {
            i += 1;
            opts.queue_depth = @min(@as(usize, 8), try std.fmt.parseInt(usize, args[i], 10));
        } else if (std.mem.eql(u8, args[i], "--workers") and i + 1 < args.len) {
            i += 1;
            opts.workers = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--transient")) {
            opts.f16_resident = false;
        }
    }
    if (opts.queue_depth == 0 or opts.iters == 0) return error.InvalidIterationCount;

    const allocator = std.heap.smp_allocator;
    var pool: raw.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = opts.workers });
    defer pool.deinit();
    const config: native.ParallelConfig = .{ .pool = &pool };

    var buf: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &buf);
    const out = &writer.interface;
    defer out.flush() catch {};
    try out.print("GPU eager LLM formats  device={s}  workers={d}+main  queue={d}\n", .{
        gpu.deviceName() orelse "unavailable",
        opts.workers,
        opts.queue_depth,
    });
    try out.print("{s:<24} | {s:>10} | {s:>10} | {s:>10} | {s:>11} | {s:>10}\n", .{
        "format/shape", "cpu us", "gpu us", "submit us", "queue GF/s", "max abs",
    });
    try out.print("{s}\n", .{"-" ** 87});

    if (opts.section == .all or opts.section == .f16) {
        for (f16_shapes) |shape| {
            if (!matches(opts.filter, shape.name)) continue;
            try benchF16(allocator, out, shape, opts.iters orelse shape.iters, opts.queue_depth, config, opts.f16_resident);
            try out.flush();
        }
    }
    if (opts.section == .all or opts.section == .quant) {
        inline for (.{ raw.DType.q4_k, raw.DType.q5_k, raw.DType.q6_k, raw.DType.q8_0 }) |dtype| {
            if (comptime dtype != .q5_k or gpu.has_q5_k_quant) {
                const wanted = switch (dtype) {
                    .q4_k => opts.format == .all or opts.format == .q4_k,
                    .q5_k => opts.format == .all or opts.format == .q5_k,
                    .q6_k => opts.format == .all or opts.format == .q6_k,
                    .q8_0 => opts.format == .all or opts.format == .q8_0,
                    else => unreachable,
                };
                if (wanted) {
                    for (quant_shapes) |shape| {
                        if (!matches(opts.filter, shape.name)) continue;
                        try benchQuant(dtype, allocator, out, shape, opts.iters orelse shape.iters, opts.queue_depth, config);
                        try out.flush();
                    }
                }
            }
        }
    }
    gpu.traceDump();
}

fn matches(filter: ?[]const u8, name: []const u8) bool {
    const needle = filter orelse return true;
    return std.mem.indexOf(u8, name, needle) != null;
}

fn benchF16(
    allocator: std.mem.Allocator,
    out_writer: anytype,
    shape: Shape,
    iters: usize,
    queue_depth: usize,
    config: native.ParallelConfig,
    resident_rhs: bool,
) !void {
    const m = shape.m;
    const n = shape.n;
    const k = shape.k;
    const a_values = try allocator.alloc(f16, m * k);
    defer allocator.free(a_values);
    const cpu_b = try allocator.alloc(f16, n * k);
    defer allocator.free(cpu_b);
    var prng = std.Random.DefaultPrng.init(0x16f00d +% m +% n +% k);
    const random = prng.random();
    for (a_values) |*v| v.* = @floatCast(random.float(f32) * 0.5 - 0.25);
    for (cpu_b) |*v| v.* = @floatCast(random.float(f32) * 0.125 - 0.0625);

    var resident: ?[]u8 = null;
    defer if (resident) |bytes| gpu.freeResidentBytes(bytes);
    const gpu_b: []f16 = if (resident_rhs) blk: {
        const bytes = gpu.allocResidentBytes(cpu_b.len * @sizeOf(f16)) orelse return error.GpuResidentAllocationFailed;
        resident = bytes;
        const values: []f16 = @alignCast(std.mem.bytesAsSlice(f16, bytes));
        @memcpy(values, cpu_b);
        break :blk values;
    } else cpu_b;

    var a = try TensorF16.fromBorrowedSlice(allocator, &.{ m, k }, a_values);
    defer a.deinit();
    var b_cpu = try TensorF16.fromBorrowedSlice(allocator, &.{ n, k }, cpu_b);
    defer b_cpu.deinit();
    var b_gpu = try TensorF16.fromBorrowedSlice(allocator, &.{ n, k }, gpu_b);
    defer b_gpu.deinit();
    var cpu_out = try Tensor.zeros(allocator, &.{ m, n });
    defer cpu_out.deinit();
    var gpu_out = try Tensor.zeros(allocator, &.{ m, n });
    defer gpu_out.deinit();
    const queued = try allocOutputs(allocator, queue_depth, m, n);
    defer freeOutputs(allocator, queued);

    vector.matmulTransB2DIntoUncheckedF16OperandsWithConfig(&cpu_out, &a, &b_cpu, m, n, k, config);
    if (!gpu.gemmF16NtAsync(&a, &b_gpu, &gpu_out, m, n, k)) return error.GpuDispatchFailed;
    _ = gpu_out.dataConst();
    for (0..3) |_| {
        vector.matmulTransB2DIntoUncheckedF16OperandsWithConfig(&cpu_out, &a, &b_cpu, m, n, k, config);
        if (!gpu.gemmF16NtAsync(&a, &b_gpu, &gpu_out, m, n, k)) return error.GpuDispatchFailed;
        _ = gpu_out.dataConst();
    }

    var cpu_times: [32]u64 = undefined;
    var gpu_times: [32]u64 = undefined;
    var submit_times: [32]u64 = undefined;
    var queue_times: [32]u64 = undefined;
    const count = @min(iters, cpu_times.len);
    if (count == 0) return error.InvalidIterationCount;
    var timer = try Timer.start(io);
    for (0..count) |rep| {
        if (rep % 2 == 0) {
            timer.reset();
            vector.matmulTransB2DIntoUncheckedF16OperandsWithConfig(&cpu_out, &a, &b_cpu, m, n, k, config);
            cpu_times[rep] = timer.read();
            timer.reset();
            if (!gpu.gemmF16NtAsync(&a, &b_gpu, &gpu_out, m, n, k)) return error.GpuDispatchFailed;
            _ = gpu_out.dataConst();
            gpu_times[rep] = timer.read();
        } else {
            timer.reset();
            if (!gpu.gemmF16NtAsync(&a, &b_gpu, &gpu_out, m, n, k)) return error.GpuDispatchFailed;
            _ = gpu_out.dataConst();
            gpu_times[rep] = timer.read();
            timer.reset();
            vector.matmulTransB2DIntoUncheckedF16OperandsWithConfig(&cpu_out, &a, &b_cpu, m, n, k, config);
            cpu_times[rep] = timer.read();
        }
        timer.reset();
        if (!gpu.gemmF16NtAsync(&a, &b_gpu, &gpu_out, m, n, k)) return error.GpuDispatchFailed;
        submit_times[rep] = timer.read();
        _ = gpu_out.dataConst();

        timer.reset();
        for (queued) |*value| if (!gpu.gemmF16NtAsync(&a, &b_gpu, value, m, n, k)) return error.GpuDispatchFailed;
        for (queued) |*value| _ = value.dataConst();
        queue_times[rep] = timer.read();
    }
    try printResult(out_writer, if (resident_rhs) "f16" else "f16-stream", shape.name, m, n, k, queue_depth, cpu_times[0..count], gpu_times[0..count], submit_times[0..count], queue_times[0..count], cpu_out.dataConst(), gpu_out.dataConst(), 1e-2);
}

fn benchQuant(
    comptime dtype: raw.DType,
    allocator: std.mem.Allocator,
    out_writer: anytype,
    shape: Shape,
    iters: usize,
    queue_depth: usize,
    config: native.ParallelConfig,
) !void {
    const Block = raw.dtype_info.Storage(dtype);
    const m = shape.m;
    const n = shape.n;
    const k = shape.k;
    const blocks_per_row = try qm.blockCountForDType(dtype, k);
    const resident = gpu.allocResidentBytes(n * blocks_per_row * @sizeOf(Block)) orelse return error.GpuResidentAllocationFailed;
    defer gpu.freeResidentBytes(resident);
    const blocks: []Block = @alignCast(std.mem.bytesAsSlice(Block, resident));
    const cpu_blocks = try allocator.alloc(Block, n * blocks_per_row);
    defer allocator.free(cpu_blocks);

    var prng = std.Random.DefaultPrng.init(@as(u64, 0x710000) +% @as(u64, @intFromEnum(dtype)) +% @as(u64, m) +% @as(u64, n) +% @as(u64, k));
    const random = prng.random();
    const row = try allocator.alloc(f32, k);
    defer allocator.free(row);
    for (0..n) |r| {
        for (row) |*v| v.* = random.float(f32) * 0.125 - 0.0625;
        try qm.quantizeRowForDType(dtype, cpu_blocks[r * blocks_per_row ..][0..blocks_per_row], row);
    }
    @memcpy(blocks, cpu_blocks);
    var packed_rhs = switch (dtype) {
        .q4_k => try qm.packMatmulRhsQ4_Kx8(allocator, cpu_blocks, n, k, blocks_per_row),
        .q5_k => try qm.packMatmulRhsQ5_Kx8(allocator, cpu_blocks, n, k, blocks_per_row),
        .q6_k => try qm.packMatmulRhsQ6_Kx4(allocator, cpu_blocks, n, k, blocks_per_row),
        .q8_0 => try qm.packMatmulRhsQ8_0x4(allocator, cpu_blocks, n, k, blocks_per_row),
        else => unreachable,
    };
    defer packed_rhs.deinit();

    const a_values = try allocator.alloc(f32, m * k);
    defer allocator.free(a_values);
    for (a_values) |*v| v.* = random.float(f32) * 0.5 - 0.25;
    var a = try Tensor.fromBorrowedSlice(allocator, &.{ m, k }, a_values);
    defer a.deinit();
    var cpu_out = try Tensor.zeros(allocator, &.{ m, n });
    defer cpu_out.deinit();
    var gpu_out = try Tensor.zeros(allocator, &.{ m, n });
    defer gpu_out.deinit();
    const queued = try allocOutputs(allocator, queue_depth, m, n);
    defer freeOutputs(allocator, queued);
    const format: gpu.QFormat = switch (dtype) {
        .q4_k => .q4_k,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
        .q8_0 => .q8_0,
        else => unreachable,
    };

    try cpuPackedQuant(dtype, allocator, &cpu_out, &a, &packed_rhs, cpu_blocks, m, n, k, config);
    if (!gpu.gemmQuantNtAsync(format, resident, true, blocks_per_row * @sizeOf(Block), 0, &a, &gpu_out, 1, m, n, k)) return error.GpuDispatchFailed;
    _ = gpu_out.dataConst();
    for (0..3) |_| {
        try cpuPackedQuant(dtype, allocator, &cpu_out, &a, &packed_rhs, cpu_blocks, m, n, k, config);
        if (!gpu.gemmQuantNtAsync(format, resident, true, blocks_per_row * @sizeOf(Block), 0, &a, &gpu_out, 1, m, n, k)) return error.GpuDispatchFailed;
        _ = gpu_out.dataConst();
    }

    var cpu_times: [32]u64 = undefined;
    var gpu_times: [32]u64 = undefined;
    var submit_times: [32]u64 = undefined;
    var queue_times: [32]u64 = undefined;
    const count = @min(iters, cpu_times.len);
    if (count == 0) return error.InvalidIterationCount;
    var timer = try Timer.start(io);
    for (0..count) |rep| {
        if (rep % 2 == 0) {
            timer.reset();
            try cpuPackedQuant(dtype, allocator, &cpu_out, &a, &packed_rhs, cpu_blocks, m, n, k, config);
            cpu_times[rep] = timer.read();
            timer.reset();
            if (!gpu.gemmQuantNtAsync(format, resident, true, blocks_per_row * @sizeOf(Block), 0, &a, &gpu_out, 1, m, n, k)) return error.GpuDispatchFailed;
            _ = gpu_out.dataConst();
            gpu_times[rep] = timer.read();
        } else {
            timer.reset();
            if (!gpu.gemmQuantNtAsync(format, resident, true, blocks_per_row * @sizeOf(Block), 0, &a, &gpu_out, 1, m, n, k)) return error.GpuDispatchFailed;
            _ = gpu_out.dataConst();
            gpu_times[rep] = timer.read();
            timer.reset();
            try cpuPackedQuant(dtype, allocator, &cpu_out, &a, &packed_rhs, cpu_blocks, m, n, k, config);
            cpu_times[rep] = timer.read();
        }
        timer.reset();
        if (!gpu.gemmQuantNtAsync(format, resident, true, blocks_per_row * @sizeOf(Block), 0, &a, &gpu_out, 1, m, n, k)) return error.GpuDispatchFailed;
        submit_times[rep] = timer.read();
        _ = gpu_out.dataConst();

        timer.reset();
        for (queued) |*value| if (!gpu.gemmQuantNtAsync(format, resident, true, blocks_per_row * @sizeOf(Block), 0, &a, value, 1, m, n, k)) return error.GpuDispatchFailed;
        for (queued) |*value| _ = value.dataConst();
        queue_times[rep] = timer.read();
    }
    try printResult(out_writer, @tagName(dtype), shape.name, m, n, k, queue_depth, cpu_times[0..count], gpu_times[0..count], submit_times[0..count], queue_times[0..count], cpu_out.dataConst(), gpu_out.dataConst(), 5e-2);
}

fn cpuPackedQuant(comptime dtype: raw.DType, allocator: std.mem.Allocator, out: *Tensor, a: *const Tensor, packed_rhs: anytype, blocks: []const raw.dtype_info.Storage(dtype), m: usize, n: usize, k: usize, config: native.ParallelConfig) !void {
    switch (dtype) {
        .q4_k => try native.matmul2DQuantizedRhsQ4_Kx8WithConfig(allocator, out, a, packed_rhs, m, n, k, config),
        .q5_k => if (m < 4) {
            const rhs = qm.QuantizedMatmulRhsQ5_K{ .allocator = null, .blocks = blocks, .k = k, .n = n, .blocks_per_column = k / qm.qk_k_block_size };
            try native.matmul2DQuantizedRhsQ5_KWithConfig(allocator, out, a, &rhs, m, n, k, config);
        } else try native.matmul2DQuantizedRhsQ5_Kx8WithConfig(allocator, out, a, packed_rhs, m, n, k, config),
        .q6_k => try native.matmul2DQuantizedRhsQ6_Kx4WithConfig(allocator, out, a, packed_rhs, m, n, k, config),
        .q8_0 => try native.matmul2DQuantizedRhsQ8_0x4WithConfig(allocator, out, a, packed_rhs, m, n, k, config),
        else => unreachable,
    }
}

fn allocOutputs(allocator: std.mem.Allocator, count: usize, m: usize, n: usize) ![]Tensor {
    const values = try allocator.alloc(Tensor, count);
    var initialized: usize = 0;
    errdefer {
        for (values[0..initialized]) |*value| value.deinit();
        allocator.free(values);
    }
    for (values) |*value| {
        value.* = try Tensor.zeros(allocator, &.{ m, n });
        initialized += 1;
    }
    return values;
}

fn freeOutputs(allocator: std.mem.Allocator, values: []Tensor) void {
    for (values) |*value| value.deinit();
    allocator.free(values);
}

fn printResult(out: anytype, format: []const u8, name: []const u8, m: usize, n: usize, k: usize, queue_depth: usize, cpu_times: []u64, gpu_times: []u64, submit_times: []u64, queue_times: []u64, cpu_values: []const f32, gpu_values: []const f32, max_allowed: f32) !void {
    const cpu_ns = median(cpu_times);
    const gpu_ns = median(gpu_times);
    const submit_ns = median(submit_times);
    const queue_ns = median(queue_times);
    var max_abs: f32 = 0;
    for (cpu_values, gpu_values) |want, got| max_abs = @max(max_abs, @abs(want - got));
    if (!std.math.isFinite(max_abs) or max_abs > max_allowed) return error.CpuParityFailed;
    const flops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    var label_buf: [64]u8 = undefined;
    const label = try std.fmt.bufPrint(&label_buf, "{s}/{s}", .{ format, name });
    try out.print("{s:<24} | {d:>10.1} | {d:>10.1} | {d:>10.1} | {d:>11.1} | {e:>10.2}\n", .{
        label,
        nsToUs(cpu_ns),
        nsToUs(gpu_ns),
        nsToUs(submit_ns),
        flops * @as(f64, @floatFromInt(queue_depth)) / @as(f64, @floatFromInt(queue_ns)),
        max_abs,
    });
}

fn median(values: []u64) u64 {
    std.mem.sort(u64, values, {}, std.sort.asc(u64));
    return values[values.len / 2];
}

fn nsToUs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3;
}
