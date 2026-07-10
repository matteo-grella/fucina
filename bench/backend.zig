// Backend comparison benchmark. Times representative ops under the scalar
// reference backend and the native production backend.
// Run with:
//   zig build bench-backend -Doptimize=ReleaseFast
//   zig build bench-backend -Doptimize=ReleaseFast -- --table-only
// Per-op timings show median of N iterations; lower is better.

const std = @import("std");
const bench_alloc = @import("alloc.zig");
const bench_options = @import("bench_options");
const Timer = @import("timer.zig").Timer;
const raw_backend = @import("raw_backend");

const Tensor = raw_backend.Tensor;
const DType = raw_backend.DType;
const scalar = raw_backend.scalar_impl;
const dtype = raw_backend.dtype_info;
const native = raw_backend.native_impl;

const iterations: usize = 50;
const warmup: usize = 5;
const transformer_iterations: usize = 7;
const transformer_warmup: usize = 1;
const scale_iterations: usize = 10;
const scale_warmup: usize = 2;

var benchmark_io: std.Io = undefined;

const AttentionGemm = enum {
    scores,
    apply,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    benchmark_io = init.io;

    const options = try parseBackendBenchOptions(args);

    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(options.allocator_mode);
    defer benchmark_allocator.deinit();
    const allocator = benchmark_allocator.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.print("native vector width: {} f32 (target: {s}); native BLAS: {s} (enabled: {}, threads: {})\n\n", .{
        native.vector_len,
        @tagName(@import("builtin").cpu.arch),
        @tagName(bench_options.native_blas_kind),
        bench_options.native_uses_blas,
        bench_options.native_blas_threads,
    });

    try stdout.print("{s:<32} | {s:>14} | {s:>14} | {s:>10}\n", .{
        "op",
        "scalar (us)",
        "native (us)",
        "native/scalar",
    });
    try stdout.print("{s}\n", .{"-" ** 80});

    if (options.table_only) {
        try benchTableQuantMatMuls(allocator, stdout);
        return;
    }

    try benchElementwise(allocator, stdout, "add 1M", 1_000_000);
    try benchElementwise(allocator, stdout, "add 16K", 16_384);
    try benchReduction(allocator, stdout, "sum 1M", 1_000_000);
    try benchReduction(allocator, stdout, "dot 1M", 1_000_000);
    try benchTypedElementwise(.f64, allocator, stdout, "add f64 1M", 1_000_000);
    try benchTypedElementwise(.f16, allocator, stdout, "add f16 1M", 1_000_000);
    try benchTypedElementwise(.bf16, allocator, stdout, "add bf16 1M", 1_000_000);
    try benchTypedDot(.f64, allocator, stdout, "dot f64 1M", 1_000_000);
    try benchTypedDot(.f16, allocator, stdout, "dot f16 1M", 1_000_000);
    try benchTypedDot(.bf16, allocator, stdout, "dot bf16 1M", 1_000_000);
    try benchMatMul(allocator, stdout, "matmul 64x64x64", 64, 64, 64);
    try benchMatMul(allocator, stdout, "matmul 128x128x128", 128, 128, 128);
    try benchTypedMatMul(.f64, allocator, stdout, "matmul f64 128x128x128", 128, 128, 128);
    try benchTypedMatMul(.f16, allocator, stdout, "matmul f16 128x128x128", 128, 128, 128);
    try benchTypedMatMul(.bf16, allocator, stdout, "matmul bf16 128x128x128", 128, 128, 128);
    try benchMatMul(allocator, stdout, "matmul 256x256x256", 256, 256, 256);
    try benchMatMul(allocator, stdout, "matmul 512x512x512", 512, 512, 512);
    try benchMatMul(allocator, stdout, "matmul 768x3072x768 (BERT)", 768, 3072, 768);
    try benchTransformerMatMuls(allocator, stdout);
    try benchBf16RhsTransBMatMuls(allocator, stdout);
    try benchDecodeMatMuls(allocator, stdout);
    try benchTableQuantMatMuls(allocator, stdout);
    try benchDecodeAtScaleMatMuls(allocator, stdout);
    try benchMatMulTransB(allocator, stdout, "matmulTransB 256x256x256", 256, 256, 256);
    try benchMatMulTransA(allocator, stdout, "matmulTransA 256x256x256", 256, 256, 256);
    try benchBatched(allocator, stdout, "bmm 32x(64x64)x(64x64)", 32, 64, 64, 64);
    try benchBatched(allocator, stdout, "bmm 8x(128x64)x(64x128) attn", 8, 128, 128, 64);
}

// ---------------- Helpers ----------------

const BackendBenchOptions = struct {
    allocator_mode: bench_alloc.AllocatorMode = .debug,
    table_only: bool = false,
};

fn parseBackendBenchOptions(args: []const []const u8) !BackendBenchOptions {
    var options: BackendBenchOptions = .{};
    for (args[1..]) |arg| {
        if (try bench_alloc.parseAllocatorModeArg(arg)) |mode| {
            options.allocator_mode = mode;
        } else if (std.mem.eql(u8, arg, "--table-only")) {
            options.table_only = true;
        } else {
            return error.UnknownArgument;
        }
    }
    return options;
}

fn randomSlice(allocator: std.mem.Allocator, len: usize, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const buf = try allocator.alloc(f32, len);
    for (buf) |*v| v.* = rng.float(f32) * 2 - 1;
    return buf;
}

fn randomSliceTyped(comptime tensor_dtype: DType, allocator: std.mem.Allocator, len: usize, seed: u64) ![]dtype.Scalar(tensor_dtype) {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const buf = try allocator.alloc(dtype.Scalar(tensor_dtype), len);
    for (buf) |*v| {
        v.* = dtype.castFloat(.f32, tensor_dtype, rng.float(f32) * 2 - 1);
    }
    return buf;
}

fn f32ToF16Bits(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

fn makeQ1_0Blocks(allocator: std.mem.Allocator, k: usize, n: usize) ![]raw_backend.BlockQ1_0 {
    const blocks_per_column = try raw_backend.quantized_matmul.q1_0BlockCount(k);
    const blocks = try allocator.alloc(raw_backend.BlockQ1_0, n * blocks_per_column);
    for (blocks, 0..) |*block, i| fillQ1_0Block(block, i);
    return blocks;
}

fn fillQ1_0Block(block: *raw_backend.BlockQ1_0, seed: usize) void {
    block.d = f32ToF16Bits(1.0 / 32.0);
    for (&block.qs, 0..) |*q, i| q.* = if ((i + seed) % 2 == 0) 0b1010_0101 else 0b0101_1010;
}

fn makeQ4_1Blocks(allocator: std.mem.Allocator, k: usize, n: usize) ![]raw_backend.BlockQ4_1 {
    const blocks_per_column = try raw_backend.quantized_matmul.q4_1BlockCount(k);
    const blocks = try allocator.alloc(raw_backend.BlockQ4_1, n * blocks_per_column);
    for (blocks, 0..) |*block, i| fillQ4_1Block(block, i);
    return blocks;
}

fn fillQ4_1Block(block: *raw_backend.BlockQ4_1, seed: usize) void {
    block.dm = .{ f32ToF16Bits(1.0 / 32.0), f32ToF16Bits(0) };
    for (&block.qs, 0..) |*q, i| {
        const lo: u8 = @intCast((i + seed) % 16);
        const hi: u8 = @intCast((i * 3 + seed) % 16);
        q.* = lo | (hi << 4);
    }
}

fn makeQ5_0Blocks(allocator: std.mem.Allocator, k: usize, n: usize) ![]raw_backend.BlockQ5_0 {
    const blocks_per_column = try raw_backend.quantized_matmul.q5_0BlockCount(k);
    const blocks = try allocator.alloc(raw_backend.BlockQ5_0, n * blocks_per_column);
    for (blocks, 0..) |*block, i| fillQ5_0Block(block, i);
    return blocks;
}

fn fillQ5_0Block(block: *raw_backend.BlockQ5_0, seed: usize) void {
    block.d = f32ToF16Bits(1.0 / 32.0);
    @memset(&block.qh, 0);
    @memset(&block.qs, 0);
    for (0..raw_backend.quantized_matmul.q5_0_block_size) |i| {
        setQ5_0Value(block, i, @intCast(@as(i32, @intCast((i + seed) % 33)) - 16));
    }
}

fn setQ5_0Value(block: *raw_backend.BlockQ5_0, index: usize, value: i8) void {
    const encoded: u8 = @intCast(@as(i16, value) + 16);
    const byte_index = index % (raw_backend.quantized_matmul.q5_0_block_size / 2);
    if (index < raw_backend.quantized_matmul.q5_0_block_size / 2) {
        block.qs[byte_index] = (block.qs[byte_index] & 0xf0) | (encoded & 0x0f);
    } else {
        block.qs[byte_index] = (block.qs[byte_index] & 0x0f) | ((encoded & 0x0f) << 4);
    }
    const bit: u5 = @intCast(index);
    if ((encoded & 0x10) != 0) {
        writeQh(block.qh[0..], readQh(&block.qh) | (@as(u32, 1) << bit));
    } else {
        writeQh(block.qh[0..], readQh(&block.qh) & ~(@as(u32, 1) << bit));
    }
}

fn makeQ5_1Blocks(allocator: std.mem.Allocator, k: usize, n: usize) ![]raw_backend.BlockQ5_1 {
    const blocks_per_column = try raw_backend.quantized_matmul.q5_1BlockCount(k);
    const blocks = try allocator.alloc(raw_backend.BlockQ5_1, n * blocks_per_column);
    for (blocks, 0..) |*block, i| fillQ5_1Block(block, i);
    return blocks;
}

fn fillQ5_1Block(block: *raw_backend.BlockQ5_1, seed: usize) void {
    block.dm = .{ f32ToF16Bits(1.0 / 32.0), f32ToF16Bits(0) };
    @memset(&block.qh, 0);
    @memset(&block.qs, 0);
    for (0..raw_backend.quantized_matmul.q5_1_block_size) |i| {
        setQ5_1Value(block, i, @intCast((i * 7 + seed) % 32));
    }
}

fn setQ5_1Value(block: *raw_backend.BlockQ5_1, index: usize, value: u8) void {
    const byte_index = index % (raw_backend.quantized_matmul.q5_1_block_size / 2);
    if (index < raw_backend.quantized_matmul.q5_1_block_size / 2) {
        block.qs[byte_index] = (block.qs[byte_index] & 0xf0) | (value & 0x0f);
    } else {
        block.qs[byte_index] = (block.qs[byte_index] & 0x0f) | ((value & 0x0f) << 4);
    }
    const bit: u5 = @intCast(index);
    if ((value & 0x10) != 0) {
        writeQh(block.qh[0..], readQh(&block.qh) | (@as(u32, 1) << bit));
    } else {
        writeQh(block.qh[0..], readQh(&block.qh) & ~(@as(u32, 1) << bit));
    }
}

fn readQh(qh: *const [4]u8) u32 {
    return std.mem.readInt(u32, qh, .little);
}

fn writeQh(qh: []u8, value: u32) void {
    std.debug.assert(qh.len == 4);
    qh[0] = @intCast(value & 0xff);
    qh[1] = @intCast((value >> 8) & 0xff);
    qh[2] = @intCast((value >> 16) & 0xff);
    qh[3] = @intCast((value >> 24) & 0xff);
}

fn makeQ2_KBlocks(allocator: std.mem.Allocator, k: usize, n: usize) ![]raw_backend.BlockQ2_K {
    const blocks_per_column = try raw_backend.quantized_matmul.qkBlockCount(k);
    const blocks = try allocator.alloc(raw_backend.BlockQ2_K, n * blocks_per_column);
    for (blocks, 0..) |*block, i| fillQ2_KBlock(block, i);
    return blocks;
}

fn fillQ2_KBlock(block: *raw_backend.BlockQ2_K, seed: usize) void {
    block.dm = .{ f32ToF16Bits(1.0 / 32.0), f32ToF16Bits(0) };
    for (&block.scales, 0..) |*scale, i| scale.* = @intCast(((i + seed) % 7) + 1);
    for (&block.qs, 0..) |*q, i| {
        q.* = @intCast(((i + seed) % 4) | (((i + seed + 1) % 4) << 2) | (((i + seed + 2) % 4) << 4) | (((i + seed + 3) % 4) << 6));
    }
}

fn makeQ3_KBlocks(allocator: std.mem.Allocator, k: usize, n: usize) ![]raw_backend.BlockQ3_K {
    const blocks_per_column = try raw_backend.quantized_matmul.qkBlockCount(k);
    const blocks = try allocator.alloc(raw_backend.BlockQ3_K, n * blocks_per_column);
    for (blocks, 0..) |*block, i| fillQ3_KBlock(block, i);
    return blocks;
}

fn fillQ3_KBlock(block: *raw_backend.BlockQ3_K, seed: usize) void {
    @memset(&block.hmask, 0);
    @memset(&block.qs, 0);
    @memset(&block.scales, 0);
    block.d = f32ToF16Bits(1.0 / 32.0);
    for (0..raw_backend.quantized_matmul.qk_k_block_size / 16) |i| {
        setQ3_KScale(block, i, @intCast(@as(i32, @intCast((i + seed) % 5)) + 1));
    }
    for (0..raw_backend.quantized_matmul.qk_k_block_size) |i| {
        setQ3_KValue(block, i, @intCast(@as(i32, @intCast((i + seed) % 8)) - 4));
    }
}

fn setQ3_KScale(block: *raw_backend.BlockQ3_K, index: usize, scale: i8) void {
    const encoded: u8 = @intCast(@as(i16, scale) + 32);
    if (index < 8) {
        block.scales[index] = (block.scales[index] & 0xf0) | (encoded & 0x0f);
    } else {
        block.scales[index - 8] = (block.scales[index - 8] & 0x0f) | ((encoded & 0x0f) << 4);
    }
    const high_index = 8 + index % 4;
    const shift: u3 = @intCast(2 * (index / 4));
    block.scales[high_index] = (block.scales[high_index] & ~(@as(u8, 0x03) << shift)) | (((encoded >> 4) & 0x03) << shift);
}

fn setQ3_KValue(block: *raw_backend.BlockQ3_K, index: usize, value: i8) void {
    const chunk = index / 128;
    const local = index % 128;
    const section = local / 32;
    const offset = local % 32;
    const byte_index = chunk * 32 + offset;
    const shift: u3 = @intCast(section * 2);
    const encoded: u8 = if (value >= 0) @intCast(value) else @intCast(@as(i16, value) + 4);
    block.qs[byte_index] = (block.qs[byte_index] & ~(@as(u8, 0x03) << shift)) | ((encoded & 0x03) << shift);
    const high_mask: u8 = @as(u8, 1) << @intCast(chunk * 4 + section);
    if (value >= 0) {
        block.hmask[offset] |= high_mask;
    } else {
        block.hmask[offset] &= ~high_mask;
    }
}

fn makeQ4_KBlocks(allocator: std.mem.Allocator, k: usize, n: usize) ![]raw_backend.BlockQ4_K {
    const blocks_per_column = try raw_backend.quantized_matmul.qkBlockCount(k);
    const blocks = try allocator.alloc(raw_backend.BlockQ4_K, n * blocks_per_column);
    for (blocks, 0..) |*block, i| fillQ4_KBlock(block, i);
    return blocks;
}

fn fillQ4_KBlock(block: *raw_backend.BlockQ4_K, seed: usize) void {
    block.dm = .{ f32ToF16Bits(1.0 / 32.0), f32ToF16Bits(0) };
    block.scales = .{ 1, 2, 3, 4, 0, 0, 0, 0, 1, 2, 3, 4 };
    for (&block.qs, 0..) |*q, i| {
        const lo: u8 = @intCast((i + seed) % 16);
        const hi: u8 = @intCast((i * 3 + seed) % 16);
        q.* = lo | (hi << 4);
    }
}

fn makeQ5_KBlocks(allocator: std.mem.Allocator, k: usize, n: usize) ![]raw_backend.BlockQ5_K {
    const blocks_per_column = try raw_backend.quantized_matmul.qkBlockCount(k);
    const blocks = try allocator.alloc(raw_backend.BlockQ5_K, n * blocks_per_column);
    for (blocks, 0..) |*block, i| fillQ5_KBlock(block, i);
    return blocks;
}

fn fillQ5_KBlock(block: *raw_backend.BlockQ5_K, seed: usize) void {
    block.dm = .{ f32ToF16Bits(1.0 / 32.0), f32ToF16Bits(0) };
    block.scales = .{ 1, 2, 3, 4, 0, 0, 0, 0, 1, 2, 3, 4 };
    @memset(&block.qh, 0);
    @memset(&block.qs, 0);
    for (0..8) |subblock| {
        for (0..32) |offset| {
            setQ5_KValue(block, subblock, offset, @intCast((subblock * 7 + offset + seed) % 32));
        }
    }
}

fn setQ5_KValue(block: *raw_backend.BlockQ5_K, subblock: usize, offset: usize, value: u8) void {
    const byte_index = (subblock / 2) * 32 + offset;
    if (subblock % 2 == 0) {
        block.qs[byte_index] = (block.qs[byte_index] & 0xf0) | (value & 0x0f);
    } else {
        block.qs[byte_index] = (block.qs[byte_index] & 0x0f) | ((value & 0x0f) << 4);
    }
    const high_mask: u8 = @as(u8, 1) << @intCast(subblock);
    if (value >= 16) {
        block.qh[offset] |= high_mask;
    } else {
        block.qh[offset] &= ~high_mask;
    }
}

fn makeQ6_KBlocks(allocator: std.mem.Allocator, k: usize, n: usize) ![]raw_backend.BlockQ6_K {
    const blocks_per_column = try raw_backend.quantized_matmul.qkBlockCount(k);
    const blocks = try allocator.alloc(raw_backend.BlockQ6_K, n * blocks_per_column);
    for (blocks, 0..) |*block, i| fillQ6_KBlock(block, i);
    return blocks;
}

fn fillQ6_KBlock(block: *raw_backend.BlockQ6_K, seed: usize) void {
    @memset(&block.ql, 0);
    @memset(&block.qh, 0);
    block.d = f32ToF16Bits(1.0 / 32.0);
    for (&block.scales, 0..) |*scale, i| {
        scale.* = @intCast(@as(i32, @intCast((i + seed) % 5)) + 1);
    }
    for (0..raw_backend.quantized_matmul.qk_k_block_size) |i| {
        const value: i8 = @intCast(@as(i32, @intCast((i + seed) % 33)) - 16);
        setQ6_KValue(block, i, value);
    }
}

fn setQ6_KValue(block: *raw_backend.BlockQ6_K, index: usize, value: i8) void {
    const encoded: u8 = @intCast(@as(i16, value) + 32);
    const chunk = index / 128;
    const local = index % 128;
    const section = local / 32;
    const l = local % 32;
    const ql_base = chunk * 64;
    const qh_base = chunk * 32;
    const ql_index = ql_base + if (section == 1 or section == 3) 32 + l else l;
    if (section < 2) {
        block.ql[ql_index] = (block.ql[ql_index] & 0xf0) | (encoded & 0x0f);
    } else {
        block.ql[ql_index] = (block.ql[ql_index] & 0x0f) | ((encoded & 0x0f) << 4);
    }
    const shift: u3 = @intCast(section * 2);
    block.qh[qh_base + l] = (block.qh[qh_base + l] & ~(@as(u8, 0x03) << shift)) | (((encoded >> 4) & 0x03) << shift);
}

fn fillLoadedQuantBlock(comptime tensor_dtype: DType, block: *dtype.Storage(tensor_dtype), seed: usize) void {
    @memset(std.mem.asBytes(block), 0);
    switch (tensor_dtype) {
        .iq1_s => {
            block.d = f32ToF16Bits(1.0 / 32.0);
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i * 43 + seed) & 0xff);
            for (&block.qh, 0..) |*q, i| q.* = @intCast((i * 73 + seed) & 0xffff);
        },
        .iq1_m => {
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i * 47 + seed) & 0xff);
            for (&block.qh, 0..) |*q, i| q.* = @intCast((i * 53 + seed) & 0xff);
            writeIQ1MScale(&block.scales, f32ToF16Bits(1.0 / 32.0));
            block.scales[0] |= 1;
            block.scales[2] |= 1;
            block.scales[4] |= 1;
            block.scales[6] |= 1;
        },
        .iq2_xxs => {
            block.d = f32ToF16Bits(1.0 / 32.0);
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i * 17 + seed) & 0xffff);
        },
        .iq2_xs => {
            block.d = f32ToF16Bits(1.0 / 32.0);
            for (&block.qs, 0..) |*q, i| {
                const grid: u16 = @intCast((i * 37 + seed) & 0x1ff);
                const signs: u16 = @intCast((i * 11 + seed) & 0x7f);
                q.* = grid | (signs << 9);
            }
            for (&block.scales, 0..) |*scale, i| scale.* = @intCast((i & 0x0f) | (((i + 3) & 0x0f) << 4));
        },
        .iq2_s => {
            block.d = f32ToF16Bits(1.0 / 32.0);
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i * 19 + seed) & 0xff);
            for (&block.qh, 0..) |*q, i| q.* = @intCast((i * 23 + seed) & 0xff);
            for (&block.scales, 0..) |*scale, i| scale.* = @intCast((i & 0x0f) | (((i + 5) & 0x0f) << 4));
        },
        .iq3_xxs => {
            block.d = f32ToF16Bits(1.0 / 32.0);
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i * 29 + seed) & 0xff);
        },
        .iq3_s => {
            block.d = f32ToF16Bits(1.0 / 32.0);
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i * 31 + seed) & 0xff);
            for (&block.qh, 0..) |*q, i| q.* = @intCast((i * 7 + seed) & 0xff);
            for (&block.signs, 0..) |*q, i| q.* = @intCast((i * 13 + seed) & 0xff);
            for (&block.scales, 0..) |*scale, i| scale.* = @intCast((i & 0x0f) | (((i + 1) & 0x0f) << 4));
        },
        .iq4_nl => {
            block.d = f32ToF16Bits(1.0 / 32.0);
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i & 0x0f) | (((i + 7) & 0x0f) << 4));
        },
        .iq4_xs => {
            block.d = f32ToF16Bits(1.0 / 32.0);
            for (&block.scales_l, 0..) |*scale, i| scale.* = @intCast(((1 + i) & 0x0f) | (((2 + i) & 0x0f) << 4));
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i & 0x0f) | (((i + 5) & 0x0f) << 4));
        },
        .tq1_0 => {
            block.d = f32ToF16Bits(1.0 / 32.0);
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i * 5 + seed) % 243);
            for (&block.qh, 0..) |*q, i| q.* = @intCast((i * 7 + seed) % 243);
        },
        .tq2_0 => {
            block.d = f32ToF16Bits(1.0 / 32.0);
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i & 0x03) | (((i + 1) & 0x03) << 2) | (((i + 2) & 0x03) << 4) | (((i + 3) & 0x03) << 6));
        },
        .mxfp4 => {
            block.e = 128;
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i & 0x0f) | (((i + 5) & 0x0f) << 4));
        },
        .nvfp4 => {
            @memset(&block.d, 0x38);
            for (&block.qs, 0..) |*q, i| q.* = @intCast((i & 0x0f) | (((i + 3) & 0x0f) << 4));
        },
        else => @compileError("unsupported loaded quant benchmark dtype"),
    }
}

fn writeIQ1MScale(scales: *[dtype.qk_k_block_size / 32]u8, scale_bits: u16) void {
    writeU16Bytes(scales[0..2], (scale_bits & 0x000f) << 12);
    writeU16Bytes(scales[2..4], (scale_bits & 0x00f0) << 8);
    writeU16Bytes(scales[4..6], (scale_bits & 0x0f00) << 4);
    writeU16Bytes(scales[6..8], scale_bits & 0xf000);
}

fn writeU16Bytes(dst: *[2]u8, value: u16) void {
    dst[0] = @intCast(value & 0xff);
    dst[1] = @intCast(value >> 8);
}

fn medianTimer(comptime f: anytype, args: anytype) !u64 {
    return medianTimerN(f, args, iterations, warmup);
}

fn medianTimerN(comptime f: anytype, args: anytype, comptime n_iters: usize, comptime n_warmup: usize) !u64 {
    const Ret = @typeInfo(@TypeOf(f)).@"fn".return_type.?;
    const fallible = @typeInfo(Ret) == .error_union;
    var times: [n_iters]u64 = undefined;
    for (0..n_warmup) |_| {
        if (fallible) try @call(.auto, f, args) else @call(.auto, f, args);
    }
    var timer = try Timer.start(benchmark_io);
    for (0..n_iters) |i| {
        timer.reset();
        if (fallible) try @call(.auto, f, args) else @call(.auto, f, args);
        times[i] = timer.read();
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[n_iters / 2];
}

fn fmtRow(
    w: anytype,
    name: []const u8,
    scalar_ns: u64,
    native_ns: u64,
) !void {
    const scalar_us = @as(f64, @floatFromInt(scalar_ns)) / 1000.0;
    const native_us = @as(f64, @floatFromInt(native_ns)) / 1000.0;
    const native_vs_scalar = @as(f64, @floatFromInt(scalar_ns)) / @as(f64, @floatFromInt(native_ns));
    try w.print(
        "{s:<32} | {d:>14.2} | {d:>14.2} | {d:>9.2}x\n",
        .{ name, scalar_us, native_us, native_vs_scalar },
    );
}

// ---------------- Per-op runners ----------------

fn benchElementwise(allocator: std.mem.Allocator, w: anytype, name: []const u8, n: usize) !void {
    const a_data = try randomSlice(allocator, n, 0x1);
    defer allocator.free(a_data);
    const b_data = try randomSlice(allocator, n, 0x2);
    defer allocator.free(b_data);

    var a = try Tensor.fromSlice(allocator, &.{n}, a_data);
    defer a.deinit();
    var b = try Tensor.fromSlice(allocator, &.{n}, b_data);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{n});
    defer out.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const NativeRunner = struct {
        fn run(o: *Tensor, lhs: *const Tensor, rhs: *const Tensor, len: usize, config: native.ParallelConfig) void {
            native.addContiguousIntoUncheckedWithConfig(o, lhs, rhs, len, config);
        }
    }.run;

    const scalar_ns = try medianTimer(scalar.addInto, .{ &out, &a, &b });
    const native_ns = try medianTimer(NativeRunner, .{ &out, &a, &b, n, native_config });
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn benchReduction(allocator: std.mem.Allocator, w: anytype, name: []const u8, n: usize) !void {
    const a_data = try randomSlice(allocator, n, 0x3);
    defer allocator.free(a_data);
    const b_data = try randomSlice(allocator, n, 0x4);
    defer allocator.free(b_data);

    var a = try Tensor.fromSlice(allocator, &.{n}, a_data);
    defer a.deinit();
    var b = try Tensor.fromSlice(allocator, &.{n}, b_data);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{1});
    defer out.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    if (std.mem.startsWith(u8, name, "sum")) {
        const NativeRunner = struct {
            fn run(o: *Tensor, lhs: *const Tensor, config: native.ParallelConfig) !void {
                try native.sumIntoWithConfig(o, lhs, config);
            }
        }.run;

        const scalar_ns = try medianTimer(scalar.sumInto, .{ &out, &a });
        const native_ns = try medianTimer(NativeRunner, .{ &out, &a, native_config });
        try fmtRow(w, name, scalar_ns, native_ns);
    } else {
        const NativeRunner = struct {
            fn run(o: *Tensor, lhs: *const Tensor, rhs: *const Tensor, config: native.ParallelConfig) !void {
                try native.dotIntoWithConfig(o, lhs, rhs, config);
            }
        }.run;

        const scalar_ns = try medianTimer(scalar.dotInto, .{ &out, &a, &b });
        const native_ns = try medianTimer(NativeRunner, .{ &out, &a, &b, native_config });
        try fmtRow(w, name, scalar_ns, native_ns);
    }
}

fn benchTypedElementwise(comptime tensor_dtype: DType, allocator: std.mem.Allocator, w: anytype, name: []const u8, n: usize) !void {
    const TypedTensor = raw_backend.TensorOf(tensor_dtype);
    const out_dtype = comptime dtype.outputDType(.pointwise, tensor_dtype);
    const OutputTensor = raw_backend.TensorOf(out_dtype);

    const a_data = try randomSliceTyped(tensor_dtype, allocator, n, 0x31);
    defer allocator.free(a_data);
    const b_data = try randomSliceTyped(tensor_dtype, allocator, n, 0x32);
    defer allocator.free(b_data);

    var a = try TypedTensor.fromSlice(allocator, &.{n}, a_data);
    defer a.deinit();
    var b = try TypedTensor.fromSlice(allocator, &.{n}, b_data);
    defer b.deinit();
    var out = try OutputTensor.zeros(allocator, &.{n});
    defer out.deinit();

    const ScalarRunner = struct {
        fn run(o: *OutputTensor, lhs: *const TypedTensor, rhs: *const TypedTensor, len: usize) void {
            scalar.elementwiseContiguousIntoTypedWithConfig(tensor_dtype, .add, o, lhs, rhs, len, .{});
        }
    }.run;
    const NativeRunner = struct {
        fn run(o: *OutputTensor, lhs: *const TypedTensor, rhs: *const TypedTensor, len: usize) void {
            native.elementwiseContiguousIntoTypedWithConfig(tensor_dtype, .add, o, lhs, rhs, len, .{});
        }
    }.run;

    const scalar_ns = try medianTimer(ScalarRunner, .{ &out, &a, &b, n });
    const native_ns = try medianTimer(NativeRunner, .{ &out, &a, &b, n });
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn benchTypedDot(comptime tensor_dtype: DType, allocator: std.mem.Allocator, w: anytype, name: []const u8, n: usize) !void {
    const TypedTensor = raw_backend.TensorOf(tensor_dtype);
    const out_dtype = comptime dtype.outputDType(.matmul, tensor_dtype);
    const OutputTensor = raw_backend.TensorOf(out_dtype);

    const a_data = try randomSliceTyped(tensor_dtype, allocator, n, 0x41);
    defer allocator.free(a_data);
    const b_data = try randomSliceTyped(tensor_dtype, allocator, n, 0x42);
    defer allocator.free(b_data);

    var a = try TypedTensor.fromSlice(allocator, &.{n}, a_data);
    defer a.deinit();
    var b = try TypedTensor.fromSlice(allocator, &.{n}, b_data);
    defer b.deinit();
    var out = try OutputTensor.zeros(allocator, &.{1});
    defer out.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const ScalarRunner = struct {
        fn run(o: *OutputTensor, lhs: *const TypedTensor, rhs: *const TypedTensor) !void {
            try scalar.dotIntoTypedWithConfig(tensor_dtype, o, lhs, rhs, .{});
        }
    }.run;
    const NativeRunner = struct {
        fn run(o: *OutputTensor, lhs: *const TypedTensor, rhs: *const TypedTensor, config: native.ParallelConfig) !void {
            try native.dotIntoTypedWithConfig(tensor_dtype, o, lhs, rhs, config);
        }
    }.run;

    const scalar_ns = try medianTimer(ScalarRunner, .{ &out, &a, &b });
    const native_ns = try medianTimer(NativeRunner, .{ &out, &a, &b, native_config });
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn benchMatMul(allocator: std.mem.Allocator, w: anytype, name: []const u8, m: usize, n: usize, k: usize) !void {
    return benchMatMulTimed(allocator, w, name, m, n, k, iterations, warmup);
}

fn benchMatMulTimed(
    allocator: std.mem.Allocator,
    w: anytype,
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    comptime n_iters: usize,
    comptime n_warmup: usize,
) !void {
    const a_data = try randomSlice(allocator, m * k, 0x5);
    defer allocator.free(a_data);
    const b_data = try randomSlice(allocator, k * n, 0x6);
    defer allocator.free(b_data);

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, a_data);
    defer a.deinit();
    var b = try Tensor.fromSlice(allocator, &.{ k, n }, b_data);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const NativeRunner = struct {
        fn run(o: *Tensor, lhs: *const Tensor, rhs: *const Tensor, rows: usize, cols: usize, inner: usize, config: native.ParallelConfig) void {
            native.matmul2DIntoUncheckedWithConfig(o, lhs, rhs, rows, cols, inner, config);
        }
    }.run;

    const scalar_ns = try medianTimerN(scalar.matmulInto, .{ &out, &a, &b }, n_iters, n_warmup);
    const native_ns = try medianTimerN(NativeRunner, .{ &out, &a, &b, m, n, k, native_config }, n_iters, n_warmup);
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn benchTypedMatMul(comptime tensor_dtype: DType, allocator: std.mem.Allocator, w: anytype, name: []const u8, m: usize, n: usize, k: usize) !void {
    return benchTypedMatMulTimed(tensor_dtype, allocator, w, name, m, n, k, iterations, warmup);
}

fn benchTypedMatMulTimed(
    comptime tensor_dtype: DType,
    allocator: std.mem.Allocator,
    w: anytype,
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    comptime n_iters: usize,
    comptime n_warmup: usize,
) !void {
    const TypedTensor = raw_backend.TensorOf(tensor_dtype);
    const out_dtype = comptime dtype.outputDType(.matmul, tensor_dtype);
    const OutputTensor = raw_backend.TensorOf(out_dtype);

    const a_data = try randomSliceTyped(tensor_dtype, allocator, m * k, 0x51);
    defer allocator.free(a_data);
    const b_data = try randomSliceTyped(tensor_dtype, allocator, k * n, 0x52);
    defer allocator.free(b_data);

    var a = try TypedTensor.fromSlice(allocator, &.{ m, k }, a_data);
    defer a.deinit();
    var b = try TypedTensor.fromSlice(allocator, &.{ k, n }, b_data);
    defer b.deinit();
    var out = try OutputTensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const ScalarRunner = struct {
        fn run(o: *OutputTensor, lhs: *const TypedTensor, rhs: *const TypedTensor, rows: usize, cols: usize, inner: usize) void {
            scalar.matmul2DIntoUncheckedTypedWithConfig(tensor_dtype, o, lhs, rhs, rows, cols, inner, .{});
        }
    }.run;
    const NativeRunner = struct {
        fn run(o: *OutputTensor, lhs: *const TypedTensor, rhs: *const TypedTensor, rows: usize, cols: usize, inner: usize, config: native.ParallelConfig) void {
            native.matmul2DIntoUncheckedTypedWithConfig(tensor_dtype, o, lhs, rhs, rows, cols, inner, config);
        }
    }.run;

    const scalar_ns = try medianTimerN(ScalarRunner, .{ &out, &a, &b, m, n, k }, n_iters, n_warmup);
    const native_ns = try medianTimerN(NativeRunner, .{ &out, &a, &b, m, n, k, native_config }, n_iters, n_warmup);
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn benchPackedMatMulTimed(
    comptime tensor_dtype: DType,
    allocator: std.mem.Allocator,
    w: anytype,
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    comptime n_iters: usize,
    comptime n_warmup: usize,
) !void {
    const TypedTensor = raw_backend.TensorOf(tensor_dtype);
    const out_dtype = comptime dtype.outputDType(.matmul, tensor_dtype);
    const OutputTensor = raw_backend.TensorOf(out_dtype);
    const PackedRhs = raw_backend.PackedMatmulRhsFor(tensor_dtype);

    const a_data = try randomSliceTyped(tensor_dtype, allocator, m * k, 0x71);
    defer allocator.free(a_data);
    const b_data = try randomSliceTyped(tensor_dtype, allocator, k * n, 0x72);
    defer allocator.free(b_data);

    var a = try TypedTensor.fromSlice(allocator, &.{ m, k }, a_data);
    defer a.deinit();
    var b = try TypedTensor.fromSlice(allocator, &.{ k, n }, b_data);
    defer b.deinit();
    var out = try OutputTensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    var packed_rhs = try native.packMatmulRhsTyped(tensor_dtype, allocator, &b);
    defer packed_rhs.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const ScalarRunner = struct {
        fn run(alloc: std.mem.Allocator, o: *OutputTensor, lhs: *const TypedTensor, rhs: *const PackedRhs, rows: usize, cols: usize, inner: usize) !void {
            try scalar.matmul2DIntoUncheckedPackedRhsTypedWithConfig(tensor_dtype, alloc, o, lhs, rhs, rows, cols, inner, .{});
        }
    }.run;
    const NativeRunner = struct {
        fn run(alloc: std.mem.Allocator, o: *OutputTensor, lhs: *const TypedTensor, rhs: *const PackedRhs, rows: usize, cols: usize, inner: usize, config: native.ParallelConfig) !void {
            try native.matmul2DIntoUncheckedPackedRhsTypedWithConfig(tensor_dtype, alloc, o, lhs, rhs, rows, cols, inner, config);
        }
    }.run;

    const scalar_ns = try medianTimerN(ScalarRunner, .{ allocator, &out, &a, &packed_rhs, m, n, k }, n_iters, n_warmup);
    const native_ns = try medianTimerN(NativeRunner, .{ allocator, &out, &a, &packed_rhs, m, n, k, native_config }, n_iters, n_warmup);
    try fmtRow(w, name, scalar_ns, native_ns);
}

// Mixed f32-LHS x bf16-RHS TransB GEMM (the autograd const-RHS dot kernel):
// the bf16 weights stream as 2 bytes/weight and widen in-register, vs the
// packed path's 4 bytes/weight f32 copy.
fn benchBf16RhsTransBMatMulTimed(
    allocator: std.mem.Allocator,
    w: anytype,
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    comptime n_iters: usize,
    comptime n_warmup: usize,
) !void {
    const Bf16Tensor = raw_backend.TensorOf(.bf16);

    const a_data = try randomSlice(allocator, m * k, 0x91);
    defer allocator.free(a_data);
    const b_data = try randomSliceTyped(.bf16, allocator, n * k, 0x92);
    defer allocator.free(b_data);

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, a_data);
    defer a.deinit();
    var b = try Bf16Tensor.fromSlice(allocator, &.{ n, k }, b_data);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const ScalarRunner = struct {
        fn run(o: *Tensor, lhs: *const Tensor, rhs: *const Bf16Tensor, rows: usize, cols: usize, inner: usize) void {
            scalar.matmulTransB2DIntoUncheckedBf16RhsWithConfig(o, lhs, rhs, rows, cols, inner, .{});
        }
    }.run;
    const NativeRunner = struct {
        fn run(o: *Tensor, lhs: *const Tensor, rhs: *const Bf16Tensor, rows: usize, cols: usize, inner: usize, config: native.ParallelConfig) void {
            native.matmulTransB2DIntoUncheckedBf16RhsWithConfig(o, lhs, rhs, rows, cols, inner, config);
        }
    }.run;

    const scalar_ns = try medianTimerN(ScalarRunner, .{ &out, &a, &b, m, n, k }, n_iters, n_warmup);
    const native_ns = try medianTimerN(NativeRunner, .{ &out, &a, &b, m, n, k, native_config }, n_iters, n_warmup);
    try fmtRow(w, name, scalar_ns, native_ns);
}

// The f16-operands TransB twin (LHS pre-cast to f16, half-precision
// accumulate) at the same shapes, for a direct comparison with the mixed
// bf16-RHS kernel above.
fn benchF16OperandsTransBMatMulTimed(
    allocator: std.mem.Allocator,
    w: anytype,
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    comptime n_iters: usize,
    comptime n_warmup: usize,
) !void {
    const F16Tensor = raw_backend.TensorOf(.f16);

    const a_data = try randomSliceTyped(.f16, allocator, m * k, 0x93);
    defer allocator.free(a_data);
    const b_data = try randomSliceTyped(.f16, allocator, n * k, 0x94);
    defer allocator.free(b_data);

    var a = try F16Tensor.fromSlice(allocator, &.{ m, k }, a_data);
    defer a.deinit();
    var b = try F16Tensor.fromSlice(allocator, &.{ n, k }, b_data);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const ScalarRunner = struct {
        fn run(o: *Tensor, lhs: *const F16Tensor, rhs: *const F16Tensor, rows: usize, cols: usize, inner: usize) void {
            scalar.matmulTransB2DIntoUncheckedF16OperandsWithConfig(o, lhs, rhs, rows, cols, inner, .{});
        }
    }.run;
    const NativeRunner = struct {
        fn run(o: *Tensor, lhs: *const F16Tensor, rhs: *const F16Tensor, rows: usize, cols: usize, inner: usize, config: native.ParallelConfig) void {
            native.matmulTransB2DIntoUncheckedF16OperandsWithConfig(o, lhs, rhs, rows, cols, inner, config);
        }
    }.run;

    const scalar_ns = try medianTimerN(ScalarRunner, .{ &out, &a, &b, m, n, k }, n_iters, n_warmup);
    const native_ns = try medianTimerN(NativeRunner, .{ &out, &a, &b, m, n, k, native_config }, n_iters, n_warmup);
    try fmtRow(w, name, scalar_ns, native_ns);
}

// Mixed f32 x bf16 TransB at decode (m=1) and prefill (m=256) shapes vs
// (a) the f16-operands TransB twin and (b) the widen-to-f32 packed path.
fn benchBf16RhsTransBMatMuls(allocator: std.mem.Allocator, w: anytype) !void {
    try benchBf16RhsTransBMatMulTimed(allocator, w, "decode qkv f32xbf16 TransB", 1, 2304, 768, iterations, warmup);
    try benchF16OperandsTransBMatMulTimed(allocator, w, "decode qkv f16 TransB twin", 1, 2304, 768, iterations, warmup);
    try benchPackedMatMulTimed(.bf16, allocator, w, "decode qkv bf16 packed f32", 1, 2304, 768, iterations, warmup);

    try benchBf16RhsTransBMatMulTimed(allocator, w, "prefill qkv f32xbf16 TransB", 256, 2304, 768, transformer_iterations, transformer_warmup);
    try benchF16OperandsTransBMatMulTimed(allocator, w, "prefill qkv f16 TransB twin", 256, 2304, 768, transformer_iterations, transformer_warmup);
    try benchPackedMatMulTimed(.bf16, allocator, w, "prefill qkv bf16 packed f32", 256, 2304, 768, transformer_iterations, transformer_warmup);
}

fn benchQuantizedI8MatMulTimed(
    allocator: std.mem.Allocator,
    w: anytype,
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    comptime n_iters: usize,
    comptime n_warmup: usize,
) !void {
    const FloatTensor = raw_backend.TensorOf(.f32);
    const QRhs = raw_backend.QuantizedMatmulRhsI8;

    const a_data = try randomSliceTyped(.f32, allocator, m * k, 0x81);
    defer allocator.free(a_data);
    const b_data = try randomSliceTyped(.f32, allocator, k * n, 0x82);
    defer allocator.free(b_data);

    var a = try FloatTensor.fromSlice(allocator, &.{ m, k }, a_data);
    defer a.deinit();
    var b = try FloatTensor.fromSlice(allocator, &.{ k, n }, b_data);
    defer b.deinit();
    var out = try FloatTensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    var qrhs = try native.quantizeMatmulRhsBlockwiseI8(allocator, &b, 32);
    defer qrhs.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const ScalarRunner = struct {
        fn run(alloc: std.mem.Allocator, o: *FloatTensor, lhs: *const FloatTensor, rhs: *const QRhs, rows: usize, cols: usize, inner: usize) !void {
            try scalar.matmul2DQuantizedRhsI8WithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{});
        }
    }.run;
    const NativeRunner = struct {
        fn run(alloc: std.mem.Allocator, o: *FloatTensor, lhs: *const FloatTensor, rhs: *const QRhs, rows: usize, cols: usize, inner: usize, config: native.ParallelConfig) !void {
            try native.matmul2DQuantizedRhsI8WithConfig(alloc, o, lhs, rhs, rows, cols, inner, config);
        }
    }.run;

    const scalar_ns = try medianTimerN(ScalarRunner, .{ allocator, &out, &a, &qrhs, m, n, k }, n_iters, n_warmup);
    const native_ns = try medianTimerN(NativeRunner, .{ allocator, &out, &a, &qrhs, m, n, k, native_config }, n_iters, n_warmup);
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn benchQuantizedGGMLMatMulTimed(
    comptime format: raw_backend.QuantizedMatmulFormat,
    allocator: std.mem.Allocator,
    w: anytype,
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    comptime n_iters: usize,
    comptime n_warmup: usize,
) !void {
    const FloatTensor = raw_backend.TensorOf(.f32);
    const QRhs = switch (format) {
        .ggml_q1_0 => raw_backend.QuantizedMatmulRhsQ1_0,
        .ggml_q4_0 => raw_backend.QuantizedMatmulRhsQ4_0,
        .ggml_q4_1 => raw_backend.QuantizedMatmulRhsQ4_1,
        .ggml_q5_0 => raw_backend.QuantizedMatmulRhsQ5_0,
        .ggml_q5_1 => raw_backend.QuantizedMatmulRhsQ5_1,
        .ggml_q8_0 => raw_backend.QuantizedMatmulRhsQ8_0,
        else => @compileError("unsupported GGML benchmark format"),
    };

    const a_data = try randomSliceTyped(.f32, allocator, m * k, 0x91);
    defer allocator.free(a_data);
    const b_data = try randomSliceTyped(.f32, allocator, k * n, 0x92);
    defer allocator.free(b_data);

    var a = try FloatTensor.fromSlice(allocator, &.{ m, k }, a_data);
    defer a.deinit();
    var b = try FloatTensor.fromSlice(allocator, &.{ k, n }, b_data);
    defer b.deinit();
    var out = try FloatTensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    var qrhs = switch (format) {
        .ggml_q1_0 => blk: {
            const blocks = try makeQ1_0Blocks(allocator, k, n);
            break :blk QRhs{ .rows = .{ .allocator = allocator, .blocks = blocks, .rows = n, .cols = k, .blocks_per_row = try raw_backend.quantized_matmul.q1_0BlockCount(k) }, .k = k, .n = n };
        },
        .ggml_q4_0 => try native.quantizeMatmulRhsQ4_0(allocator, &b),
        .ggml_q4_1 => blk: {
            const blocks = try makeQ4_1Blocks(allocator, k, n);
            break :blk QRhs{ .rows = .{ .allocator = allocator, .blocks = blocks, .rows = n, .cols = k, .blocks_per_row = try raw_backend.quantized_matmul.q4_1BlockCount(k) }, .k = k, .n = n };
        },
        .ggml_q5_0 => blk: {
            const blocks = try makeQ5_0Blocks(allocator, k, n);
            break :blk QRhs{ .rows = .{ .allocator = allocator, .blocks = blocks, .rows = n, .cols = k, .blocks_per_row = try raw_backend.quantized_matmul.q5_0BlockCount(k) }, .k = k, .n = n };
        },
        .ggml_q5_1 => blk: {
            const blocks = try makeQ5_1Blocks(allocator, k, n);
            break :blk QRhs{ .rows = .{ .allocator = allocator, .blocks = blocks, .rows = n, .cols = k, .blocks_per_row = try raw_backend.quantized_matmul.q5_1BlockCount(k) }, .k = k, .n = n };
        },
        .ggml_q8_0 => try native.quantizeMatmulRhsQ8_0(allocator, &b),
        else => unreachable,
    };
    defer qrhs.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const ScalarRunner = struct {
        fn run(alloc: std.mem.Allocator, o: *FloatTensor, lhs: *const FloatTensor, rhs: *const QRhs, rows: usize, cols: usize, inner: usize) !void {
            switch (format) {
                .ggml_q1_0 => try scalar.matmul2DQuantizedRhsQ1_0WithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{}),
                .ggml_q4_0 => try scalar.matmul2DQuantizedRhsQ4_0WithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{}),
                .ggml_q4_1 => try scalar.matmul2DQuantizedRhsQ4_1WithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{}),
                .ggml_q5_0 => try scalar.matmul2DQuantizedRhsQ5_0WithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{}),
                .ggml_q5_1 => try scalar.matmul2DQuantizedRhsQ5_1WithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{}),
                .ggml_q8_0 => try scalar.matmul2DQuantizedRhsQ8_0WithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{}),
                else => unreachable,
            }
        }
    }.run;
    const NativeRunner = struct {
        fn run(alloc: std.mem.Allocator, o: *FloatTensor, lhs: *const FloatTensor, rhs: *const QRhs, rows: usize, cols: usize, inner: usize, config: native.ParallelConfig) !void {
            switch (format) {
                .ggml_q1_0 => try native.matmul2DQuantizedRhsQ1_0WithConfig(alloc, o, lhs, rhs, rows, cols, inner, config),
                .ggml_q4_0 => try native.matmul2DQuantizedRhsQ4_0WithConfig(alloc, o, lhs, rhs, rows, cols, inner, config),
                .ggml_q4_1 => try native.matmul2DQuantizedRhsQ4_1WithConfig(alloc, o, lhs, rhs, rows, cols, inner, config),
                .ggml_q5_0 => try native.matmul2DQuantizedRhsQ5_0WithConfig(alloc, o, lhs, rhs, rows, cols, inner, config),
                .ggml_q5_1 => try native.matmul2DQuantizedRhsQ5_1WithConfig(alloc, o, lhs, rhs, rows, cols, inner, config),
                .ggml_q8_0 => try native.matmul2DQuantizedRhsQ8_0WithConfig(alloc, o, lhs, rhs, rows, cols, inner, config),
                else => unreachable,
            }
        }
    }.run;

    const scalar_ns = try medianTimerN(ScalarRunner, .{ allocator, &out, &a, &qrhs, m, n, k }, n_iters, n_warmup);
    const native_ns = try medianTimerN(NativeRunner, .{ allocator, &out, &a, &qrhs, m, n, k, native_config }, n_iters, n_warmup);
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn benchQuantizedGGMLKMatMulTimed(
    comptime format: raw_backend.QuantizedMatmulFormat,
    allocator: std.mem.Allocator,
    w: anytype,
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    comptime n_iters: usize,
    comptime n_warmup: usize,
) !void {
    const FloatTensor = raw_backend.TensorOf(.f32);
    const QRhs = switch (format) {
        .ggml_q2_k => raw_backend.QuantizedMatmulRhsQ2_K,
        .ggml_q3_k => raw_backend.QuantizedMatmulRhsQ3_K,
        .ggml_q4_k => raw_backend.QuantizedMatmulRhsQ4_K,
        .ggml_q5_k => raw_backend.QuantizedMatmulRhsQ5_K,
        .ggml_q6_k => raw_backend.QuantizedMatmulRhsQ6_K,
        else => @compileError("unsupported GGML K benchmark format"),
    };

    const a_data = try randomSliceTyped(.f32, allocator, m * k, 0x93);
    defer allocator.free(a_data);

    var a = try FloatTensor.fromSlice(allocator, &.{ m, k }, a_data);
    defer a.deinit();
    var out = try FloatTensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    var qrhs: QRhs = switch (format) {
        .ggml_q2_k => blk: {
            const blocks = try makeQ2_KBlocks(allocator, k, n);
            defer allocator.free(blocks);
            break :blk try raw_backend.quantized_matmul.quantizedMatmulRhsQ2_KFromBlocks(allocator, k, n, blocks);
        },
        .ggml_q3_k => blk: {
            const blocks = try makeQ3_KBlocks(allocator, k, n);
            defer allocator.free(blocks);
            break :blk try raw_backend.quantized_matmul.quantizedMatmulRhsQ3_KFromBlocks(allocator, k, n, blocks);
        },
        .ggml_q4_k => blk: {
            const blocks = try makeQ4_KBlocks(allocator, k, n);
            defer allocator.free(blocks);
            break :blk try raw_backend.quantized_matmul.quantizedMatmulRhsQ4_KFromBlocks(allocator, k, n, blocks);
        },
        .ggml_q5_k => blk: {
            const blocks = try makeQ5_KBlocks(allocator, k, n);
            defer allocator.free(blocks);
            break :blk try raw_backend.quantized_matmul.quantizedMatmulRhsQ5_KFromBlocks(allocator, k, n, blocks);
        },
        .ggml_q6_k => blk: {
            const blocks = try makeQ6_KBlocks(allocator, k, n);
            defer allocator.free(blocks);
            break :blk try raw_backend.quantized_matmul.quantizedMatmulRhsQ6_KFromBlocks(allocator, k, n, blocks);
        },
        else => unreachable,
    };
    defer qrhs.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const ScalarRunner = struct {
        fn run(alloc: std.mem.Allocator, o: *FloatTensor, lhs: *const FloatTensor, rhs: *const QRhs, rows: usize, cols: usize, inner: usize) !void {
            switch (format) {
                .ggml_q2_k => try scalar.matmul2DQuantizedRhsQ2_KWithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{}),
                .ggml_q3_k => try scalar.matmul2DQuantizedRhsQ3_KWithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{}),
                .ggml_q4_k => try scalar.matmul2DQuantizedRhsQ4_KWithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{}),
                .ggml_q5_k => try scalar.matmul2DQuantizedRhsQ5_KWithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{}),
                .ggml_q6_k => try scalar.matmul2DQuantizedRhsQ6_KWithConfig(alloc, o, lhs, rhs, rows, cols, inner, .{}),
                else => unreachable,
            }
        }
    }.run;
    const NativeRunner = struct {
        fn run(alloc: std.mem.Allocator, o: *FloatTensor, lhs: *const FloatTensor, rhs: *const QRhs, rows: usize, cols: usize, inner: usize, config: native.ParallelConfig) !void {
            switch (format) {
                .ggml_q2_k => try native.matmul2DQuantizedRhsQ2_KWithConfig(alloc, o, lhs, rhs, rows, cols, inner, config),
                .ggml_q3_k => try native.matmul2DQuantizedRhsQ3_KWithConfig(alloc, o, lhs, rhs, rows, cols, inner, config),
                .ggml_q4_k => try native.matmul2DQuantizedRhsQ4_KWithConfig(alloc, o, lhs, rhs, rows, cols, inner, config),
                .ggml_q5_k => try native.matmul2DQuantizedRhsQ5_KWithConfig(alloc, o, lhs, rhs, rows, cols, inner, config),
                .ggml_q6_k => try native.matmul2DQuantizedRhsQ6_KWithConfig(alloc, o, lhs, rhs, rows, cols, inner, config),
                else => unreachable,
            }
        }
    }.run;

    const scalar_ns = try medianTimerN(ScalarRunner, .{ allocator, &out, &a, &qrhs, m, n, k }, n_iters, n_warmup);
    const native_ns = try medianTimerN(NativeRunner, .{ allocator, &out, &a, &qrhs, m, n, k, native_config }, n_iters, n_warmup);
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn benchQuantizedLoadedMatMulTimed(
    comptime tensor_dtype: DType,
    allocator: std.mem.Allocator,
    w: anytype,
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    comptime n_iters: usize,
    comptime n_warmup: usize,
) !void {
    const FloatTensor = raw_backend.TensorOf(.f32);
    const QRhs = raw_backend.quantized_matmul.QuantizedMatmulRhsRowsFor(tensor_dtype);

    const a_data = try randomSliceTyped(.f32, allocator, m * k, 0x95);
    defer allocator.free(a_data);

    var a = try FloatTensor.fromSlice(allocator, &.{ m, k }, a_data);
    defer a.deinit();
    var out = try FloatTensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    const blocks_per_column = try raw_backend.quantized_matmul.blockCountForDType(tensor_dtype, k);
    const blocks = try allocator.alloc(dtype.Storage(tensor_dtype), n * blocks_per_column);
    defer allocator.free(blocks);
    for (blocks, 0..) |*block, i| fillLoadedQuantBlock(tensor_dtype, block, i);

    const qrhs = QRhs{
        .rows = .{ .allocator = allocator, .blocks = blocks, .rows = n, .cols = k, .blocks_per_row = blocks_per_column },
        .k = k,
        .n = n,
    };

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const ScalarRunner = struct {
        fn run(alloc: std.mem.Allocator, o: *FloatTensor, lhs: *const FloatTensor, rhs: *const QRhs, rows: usize, cols: usize, inner: usize) !void {
            try scalar.matmul2DQuantizedRhsWithConfig(alloc, o, lhs, anyLoadedRhs(tensor_dtype, rhs), rows, cols, inner, .{});
        }
    }.run;
    const NativeRunner = struct {
        fn run(alloc: std.mem.Allocator, o: *FloatTensor, lhs: *const FloatTensor, rhs: *const QRhs, rows: usize, cols: usize, inner: usize, config: native.ParallelConfig) !void {
            try native.matmul2DQuantizedRhsWithConfig(alloc, o, lhs, anyLoadedRhs(tensor_dtype, rhs), rows, cols, inner, config);
        }
    }.run;

    const scalar_ns = try medianTimerN(ScalarRunner, .{ allocator, &out, &a, &qrhs, m, n, k }, n_iters, n_warmup);
    const native_ns = try medianTimerN(NativeRunner, .{ allocator, &out, &a, &qrhs, m, n, k, native_config }, n_iters, n_warmup);
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn anyLoadedRhs(comptime tensor_dtype: DType, rhs: *const raw_backend.quantized_matmul.QuantizedMatmulRhsRowsFor(tensor_dtype)) raw_backend.AnyQuantizedMatmulRhs {
    return switch (tensor_dtype) {
        .iq1_s => .{ .ggml_iq1_s = rhs },
        .iq1_m => .{ .ggml_iq1_m = rhs },
        .iq2_xxs => .{ .ggml_iq2_xxs = rhs },
        .iq2_xs => .{ .ggml_iq2_xs = rhs },
        .iq2_s => .{ .ggml_iq2_s = rhs },
        .iq3_xxs => .{ .ggml_iq3_xxs = rhs },
        .iq3_s => .{ .ggml_iq3_s = rhs },
        .iq4_nl => .{ .ggml_iq4_nl = rhs },
        .iq4_xs => .{ .ggml_iq4_xs = rhs },
        .tq1_0 => .{ .ggml_tq1_0 = rhs },
        .tq2_0 => .{ .ggml_tq2_0 = rhs },
        .mxfp4 => .{ .ggml_mxfp4 = rhs },
        .nvfp4 => .{ .ggml_nvfp4 = rhs },
        else => @compileError("unsupported loaded quant benchmark dtype"),
    };
}

// Decode regime: m = 1 (single token). Memory-bound, so weight traffic dominates.
// The f16/bf16 "packed" paths store widened f32 weights, so they stream 4 bytes
// per weight; the i8 block-wise path streams ~1 byte per weight. This is where
// int8 is expected to win. The packed f16/bf16 path uses the native column-split
// GEMV kernel at m=1, avoiding the wider f32 output temporary used by the
// general packed GEMM bridge.
fn benchDecodeMatMuls(allocator: std.mem.Allocator, w: anytype) !void {
    try benchMatMulTimed(allocator, w, "decode qkv f32", 1, 2304, 768, iterations, warmup);
    try benchPackedMatMulTimed(.f16, allocator, w, "decode qkv f16 packed", 1, 2304, 768, iterations, warmup);
    try benchPackedMatMulTimed(.bf16, allocator, w, "decode qkv bf16 packed", 1, 2304, 768, iterations, warmup);
    try benchQuantizedI8MatMulTimed(allocator, w, "decode qkv i8 blockwise", 1, 2304, 768, iterations, warmup);
    try benchQuantizedGGMLMatMulTimed(.ggml_q1_0, allocator, w, "decode qkv GGML Q1_0", 1, 2304, 768, iterations, warmup);
    try benchQuantizedGGMLMatMulTimed(.ggml_q8_0, allocator, w, "decode qkv GGML Q8_0", 1, 2304, 768, iterations, warmup);
    try benchQuantizedGGMLMatMulTimed(.ggml_q4_0, allocator, w, "decode qkv GGML Q4_0", 1, 2304, 768, iterations, warmup);
    try benchQuantizedGGMLMatMulTimed(.ggml_q4_1, allocator, w, "decode qkv GGML Q4_1", 1, 2304, 768, iterations, warmup);
    try benchQuantizedGGMLMatMulTimed(.ggml_q5_0, allocator, w, "decode qkv GGML Q5_0", 1, 2304, 768, iterations, warmup);
    try benchQuantizedGGMLMatMulTimed(.ggml_q5_1, allocator, w, "decode qkv GGML Q5_1", 1, 2304, 768, iterations, warmup);
    try benchQuantizedGGMLKMatMulTimed(.ggml_q2_k, allocator, w, "decode qkv GGML Q2_K", 1, 2304, 768, iterations, warmup);
    try benchQuantizedGGMLKMatMulTimed(.ggml_q3_k, allocator, w, "decode qkv GGML Q3_K", 1, 2304, 768, iterations, warmup);
    try benchQuantizedGGMLKMatMulTimed(.ggml_q4_k, allocator, w, "decode qkv GGML Q4_K", 1, 2304, 768, iterations, warmup);
    try benchQuantizedGGMLKMatMulTimed(.ggml_q5_k, allocator, w, "decode qkv GGML Q5_K", 1, 2304, 768, iterations, warmup);
    try benchQuantizedGGMLKMatMulTimed(.ggml_q6_k, allocator, w, "decode qkv GGML Q6_K", 1, 2304, 768, iterations, warmup);

    try benchMatMulTimed(allocator, w, "decode mlp up f32", 1, 3072, 768, iterations, warmup);
    try benchPackedMatMulTimed(.f16, allocator, w, "decode mlp up f16 packed", 1, 3072, 768, iterations, warmup);
    try benchPackedMatMulTimed(.bf16, allocator, w, "decode mlp up bf16 packed", 1, 3072, 768, iterations, warmup);
    try benchQuantizedI8MatMulTimed(allocator, w, "decode mlp up i8 blockwise", 1, 3072, 768, iterations, warmup);

    try benchMatMulTimed(allocator, w, "decode mlp down f32", 1, 768, 3072, iterations, warmup);
    try benchPackedMatMulTimed(.f16, allocator, w, "decode mlp down f16 packed", 1, 768, 3072, iterations, warmup);
    try benchPackedMatMulTimed(.bf16, allocator, w, "decode mlp down bf16 packed", 1, 768, 3072, iterations, warmup);
    try benchQuantizedI8MatMulTimed(allocator, w, "decode mlp down i8 blockwise", 1, 768, 3072, iterations, warmup);
}

fn benchTableQuantMatMuls(allocator: std.mem.Allocator, w: anytype) !void {
    const n = 512;
    const k = 256;
    try benchQuantizedLoadedMatMulTimed(.iq1_s, allocator, w, "loaded IQ1_S 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq1_m, allocator, w, "loaded IQ1_M 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq2_xxs, allocator, w, "loaded IQ2_XXS 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq2_xs, allocator, w, "loaded IQ2_XS 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq2_s, allocator, w, "loaded IQ2_S 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq3_xxs, allocator, w, "loaded IQ3_XXS 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq3_s, allocator, w, "loaded IQ3_S 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq4_nl, allocator, w, "loaded IQ4_NL 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq4_xs, allocator, w, "loaded IQ4_XS 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.tq1_0, allocator, w, "loaded TQ1_0 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.tq2_0, allocator, w, "loaded TQ2_0 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.mxfp4, allocator, w, "loaded MXFP4 1x512x256", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.nvfp4, allocator, w, "loaded NVFP4 1x512x256", 1, n, k, iterations, warmup);

    try benchLoadedTableDecodeMatMuls(allocator, w);
    try benchLoadedTableScaleMatMuls(allocator, w);
}

fn benchLoadedTableDecodeMatMuls(allocator: std.mem.Allocator, w: anytype) !void {
    const n = 2304;
    const k = 768;
    try benchQuantizedLoadedMatMulTimed(.iq1_s, allocator, w, "loaded decode IQ1_S 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq1_m, allocator, w, "loaded decode IQ1_M 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq2_xxs, allocator, w, "loaded decode IQ2_XXS 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq2_xs, allocator, w, "loaded decode IQ2_XS 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq2_s, allocator, w, "loaded decode IQ2_S 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq3_xxs, allocator, w, "loaded decode IQ3_XXS 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq3_s, allocator, w, "loaded decode IQ3_S 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq4_nl, allocator, w, "loaded decode IQ4_NL 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.iq4_xs, allocator, w, "loaded decode IQ4_XS 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.tq1_0, allocator, w, "loaded decode TQ1_0 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.tq2_0, allocator, w, "loaded decode TQ2_0 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.mxfp4, allocator, w, "loaded decode MXFP4 1x2304x768", 1, n, k, iterations, warmup);
    try benchQuantizedLoadedMatMulTimed(.nvfp4, allocator, w, "loaded decode NVFP4 1x2304x768", 1, n, k, iterations, warmup);
}

fn benchLoadedTableScaleMatMuls(allocator: std.mem.Allocator, w: anytype) !void {
    const n = 8192;
    const k = 8192;
    try benchQuantizedLoadedMatMulTimed(.iq4_nl, allocator, w, "loaded@scale IQ4_NL 1x8192x8192", 1, n, k, scale_iterations, scale_warmup);
    try benchQuantizedLoadedMatMulTimed(.mxfp4, allocator, w, "loaded@scale MXFP4 1x8192x8192", 1, n, k, scale_iterations, scale_warmup);
    try benchQuantizedLoadedMatMulTimed(.nvfp4, allocator, w, "loaded@scale NVFP4 1x8192x8192", 1, n, k, scale_iterations, scale_warmup);
}

// Decode at model scale: a single weight matrix far larger than last-level cache
// (f32 = 256 MB, i8 = 64 MB at 8192x8192), so every call streams weights from
// DRAM. This is the regime real decode lives in (all model weights read once per
// token) and the ONLY one where int8's 4x-fewer weight bytes can show up. The
// smaller decode bench above is cache-resident and cannot reveal it.
//
// Diagnostic to read: if int8(native) ~ f32(native)/4, the int8 kernel is
// memory-bound (full quant win realized). If it's only ~/2-2.5, the kernel is
// compute-bound and a denser microkernel (sdot/VNNI) is needed to reach the
// memory floor. NOTE: the f32 path uses BLAS when enabled; the packed f16 path
// uses the native column-split GEMV bridge at m=1 and still streams f32-packed
// weights.
fn benchDecodeAtScaleMatMuls(allocator: std.mem.Allocator, w: anytype) !void {
    const n = 8192;
    const k = 8192;
    try benchMatMulTimed(allocator, w, "decode@scale f32 1x8192x8192", 1, n, k, scale_iterations, scale_warmup);
    try benchPackedMatMulTimed(.f16, allocator, w, "decode@scale f16 packed", 1, n, k, scale_iterations, scale_warmup);
    try benchQuantizedI8MatMulTimed(allocator, w, "decode@scale i8 blockwise", 1, n, k, scale_iterations, scale_warmup);
    try benchQuantizedGGMLMatMulTimed(.ggml_q8_0, allocator, w, "decode@scale GGML Q8_0", 1, n, k, scale_iterations, scale_warmup);
    try benchQuantizedGGMLMatMulTimed(.ggml_q4_0, allocator, w, "decode@scale GGML Q4_0", 1, n, k, scale_iterations, scale_warmup);
    try benchQuantizedGGMLKMatMulTimed(.ggml_q2_k, allocator, w, "decode@scale GGML Q2_K", 1, n, k, scale_iterations, scale_warmup);
    try benchQuantizedGGMLKMatMulTimed(.ggml_q3_k, allocator, w, "decode@scale GGML Q3_K", 1, n, k, scale_iterations, scale_warmup);
    try benchQuantizedGGMLKMatMulTimed(.ggml_q4_k, allocator, w, "decode@scale GGML Q4_K", 1, n, k, scale_iterations, scale_warmup);
    try benchQuantizedGGMLKMatMulTimed(.ggml_q5_k, allocator, w, "decode@scale GGML Q5_K", 1, n, k, scale_iterations, scale_warmup);
    try benchQuantizedGGMLKMatMulTimed(.ggml_q6_k, allocator, w, "decode@scale GGML Q6_K", 1, n, k, scale_iterations, scale_warmup);
}

fn benchTransformerMatMuls(allocator: std.mem.Allocator, w: anytype) !void {
    try benchMatMulTimed(allocator, w, "qkv f32 T128 D768 O2304", 128, 2304, 768, transformer_iterations, transformer_warmup);
    try benchTypedMatMulTimed(.f16, allocator, w, "qkv f16 T128 D768 O2304", 128, 2304, 768, transformer_iterations, transformer_warmup);
    try benchPackedMatMulTimed(.f16, allocator, w, "qkv f16 packed RHS", 128, 2304, 768, transformer_iterations, transformer_warmup);
    try benchTypedMatMulTimed(.bf16, allocator, w, "qkv bf16 T128 D768 O2304", 128, 2304, 768, transformer_iterations, transformer_warmup);
    try benchPackedMatMulTimed(.bf16, allocator, w, "qkv bf16 packed RHS", 128, 2304, 768, transformer_iterations, transformer_warmup);
    try benchQuantizedI8MatMulTimed(allocator, w, "qkv i8 blockwise RHS", 128, 2304, 768, transformer_iterations, transformer_warmup);

    try benchTypedMatMulTimed(.f16, allocator, w, "mlp up f16 T128 768x3072", 128, 3072, 768, transformer_iterations, transformer_warmup);
    try benchPackedMatMulTimed(.f16, allocator, w, "mlp up f16 packed RHS", 128, 3072, 768, transformer_iterations, transformer_warmup);
    try benchTypedMatMulTimed(.bf16, allocator, w, "mlp up bf16 T128 768x3072", 128, 3072, 768, transformer_iterations, transformer_warmup);
    try benchPackedMatMulTimed(.bf16, allocator, w, "mlp up bf16 packed RHS", 128, 3072, 768, transformer_iterations, transformer_warmup);
    try benchQuantizedI8MatMulTimed(allocator, w, "mlp up i8 blockwise RHS", 128, 3072, 768, transformer_iterations, transformer_warmup);
    try benchTypedMatMulTimed(.f16, allocator, w, "mlp down f16 T128 3072x768", 128, 768, 3072, transformer_iterations, transformer_warmup);
    try benchPackedMatMulTimed(.f16, allocator, w, "mlp down f16 packed RHS", 128, 768, 3072, transformer_iterations, transformer_warmup);
    try benchTypedMatMulTimed(.bf16, allocator, w, "mlp down bf16 T128 3072x768", 128, 768, 3072, transformer_iterations, transformer_warmup);
    try benchPackedMatMulTimed(.bf16, allocator, w, "mlp down bf16 packed RHS", 128, 768, 3072, transformer_iterations, transformer_warmup);
    try benchQuantizedI8MatMulTimed(allocator, w, "mlp down i8 blockwise RHS", 128, 768, 3072, transformer_iterations, transformer_warmup);

    try benchTypedMatMulTimed(.f16, allocator, w, "attn score f16 S128 Dh64", 128, 128, 64, transformer_iterations, transformer_warmup);
    try benchTypedMatMulTimed(.bf16, allocator, w, "attn score bf16 S128 Dh64", 128, 128, 64, transformer_iterations, transformer_warmup);
    try benchTypedMatMulTimed(.f16, allocator, w, "attn apply f16 S128 Dh64", 128, 64, 128, transformer_iterations, transformer_warmup);
    try benchTypedMatMulTimed(.bf16, allocator, w, "attn apply bf16 S128 Dh64", 128, 64, 128, transformer_iterations, transformer_warmup);

    try benchTypedAttentionLoop(.f16, .scores, allocator, w, "attn bmm score f16 H32 S128", 32, 128, 64, transformer_iterations, transformer_warmup);
    try benchTypedAttentionLoop(.bf16, .scores, allocator, w, "attn bmm score bf16 H32 S128", 32, 128, 64, transformer_iterations, transformer_warmup);
    try benchTypedAttentionLoop(.f16, .apply, allocator, w, "attn bmm apply f16 H32 S128", 32, 128, 64, transformer_iterations, transformer_warmup);
    try benchTypedAttentionLoop(.bf16, .apply, allocator, w, "attn bmm apply bf16 H32 S128", 32, 128, 64, transformer_iterations, transformer_warmup);
}

fn benchTypedAttentionLoop(
    comptime tensor_dtype: DType,
    comptime mode: AttentionGemm,
    allocator: std.mem.Allocator,
    w: anytype,
    name: []const u8,
    batch_heads: usize,
    seq_len: usize,
    head_dim: usize,
    comptime n_iters: usize,
    comptime n_warmup: usize,
) !void {
    const TypedTensor = raw_backend.TensorOf(tensor_dtype);
    const out_dtype = comptime dtype.outputDType(.matmul, tensor_dtype);
    const OutputTensor = raw_backend.TensorOf(out_dtype);

    const m = seq_len;
    const n = switch (mode) {
        .scores => seq_len,
        .apply => head_dim,
    };
    const k = switch (mode) {
        .scores => head_dim,
        .apply => seq_len,
    };

    const a_data = try randomSliceTyped(tensor_dtype, allocator, batch_heads * m * k, 0x61);
    defer allocator.free(a_data);
    const b_data = try randomSliceTyped(tensor_dtype, allocator, batch_heads * k * n, 0x62);
    defer allocator.free(b_data);

    var a_base = try TypedTensor.fromSlice(allocator, &.{ batch_heads * m, k }, a_data);
    defer a_base.deinit();
    var b_base = try TypedTensor.fromSlice(allocator, &.{ batch_heads * k, n }, b_data);
    defer b_base.deinit();
    var out_base = try OutputTensor.zeros(allocator, &.{ batch_heads * m, n });
    defer out_base.deinit();

    var a_views = try allocator.alloc(TypedTensor, batch_heads);
    defer allocator.free(a_views);
    var b_views = try allocator.alloc(TypedTensor, batch_heads);
    defer allocator.free(b_views);
    var out_views = try allocator.alloc(OutputTensor, batch_heads);
    defer allocator.free(out_views);

    for (0..batch_heads) |head| {
        a_views[head] = try a_base.viewWithStridesOffset(&.{ m, k }, &.{ k, 1 }, head * m * k);
        errdefer a_views[head].deinit();
        b_views[head] = try b_base.viewWithStridesOffset(&.{ k, n }, &.{ n, 1 }, head * k * n);
        errdefer b_views[head].deinit();
        out_views[head] = try out_base.viewWithStridesOffset(&.{ m, n }, &.{ n, 1 }, head * m * n);
        errdefer out_views[head].deinit();
    }
    defer for (out_views) |*view| view.deinit();
    defer for (b_views) |*view| view.deinit();
    defer for (a_views) |*view| view.deinit();

    const ScalarRunner = struct {
        fn run(outs: []OutputTensor, lhs: []TypedTensor, rhs: []TypedTensor, rows: usize, cols: usize, inner: usize) void {
            for (0..outs.len) |i| {
                scalar.matmul2DIntoUncheckedTypedWithConfig(tensor_dtype, &outs[i], &lhs[i], &rhs[i], rows, cols, inner, .{});
            }
        }
    }.run;
    const NativeRunner = struct {
        fn run(outs: []OutputTensor, lhs: []TypedTensor, rhs: []TypedTensor, rows: usize, cols: usize, inner: usize) void {
            for (0..outs.len) |i| {
                native.matmul2DIntoUncheckedTypedWithConfig(tensor_dtype, &outs[i], &lhs[i], &rhs[i], rows, cols, inner, .{});
            }
        }
    }.run;

    const scalar_ns = try medianTimerN(ScalarRunner, .{ out_views, a_views, b_views, m, n, k }, n_iters, n_warmup);
    const native_ns = try medianTimerN(NativeRunner, .{ out_views, a_views, b_views, m, n, k }, n_iters, n_warmup);
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn benchMatMulTransA(allocator: std.mem.Allocator, w: anytype, name: []const u8, m: usize, n: usize, k: usize) !void {
    const a_data = try randomSlice(allocator, k * m, 0x7);
    defer allocator.free(a_data);
    const b_data = try randomSlice(allocator, k * n, 0x8);
    defer allocator.free(b_data);

    var a = try Tensor.fromSlice(allocator, &.{ k, m }, a_data);
    defer a.deinit();
    var b = try Tensor.fromSlice(allocator, &.{ k, n }, b_data);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const NativeRunner = struct {
        fn run(o: *Tensor, lhs: *const Tensor, rhs: *const Tensor, rows: usize, cols: usize, inner: usize, config: native.ParallelConfig) void {
            native.matmulTransA2DIntoUncheckedWithConfig(o, lhs, rhs, rows, cols, inner, config);
        }
    }.run;

    const scalar_ns = try medianTimer(scalar.matmulTransAInto, .{ &out, &a, &b });
    const native_ns = try medianTimer(NativeRunner, .{ &out, &a, &b, m, n, k, native_config });
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn benchMatMulTransB(allocator: std.mem.Allocator, w: anytype, name: []const u8, m: usize, n: usize, k: usize) !void {
    const a_data = try randomSlice(allocator, m * k, 0x9);
    defer allocator.free(a_data);
    const b_data = try randomSlice(allocator, n * k, 0xa);
    defer allocator.free(b_data);

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, a_data);
    defer a.deinit();
    var b = try Tensor.fromSlice(allocator, &.{ n, k }, b_data);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const NativeRunner = struct {
        fn run(o: *Tensor, lhs: *const Tensor, rhs: *const Tensor, rows: usize, cols: usize, inner: usize, config: native.ParallelConfig) void {
            native.matmulTransB2DIntoUncheckedWithConfig(o, lhs, rhs, rows, cols, inner, config);
        }
    }.run;

    const scalar_ns = try medianTimer(scalar.matmulTransBInto, .{ &out, &a, &b });
    const native_ns = try medianTimer(NativeRunner, .{ &out, &a, &b, m, n, k, native_config });
    try fmtRow(w, name, scalar_ns, native_ns);
}

fn benchBatched(allocator: std.mem.Allocator, w: anytype, name: []const u8, batch: usize, m: usize, n: usize, k: usize) !void {
    const a_data = try randomSlice(allocator, batch * m * k, 0xb);
    defer allocator.free(a_data);
    const b_data = try randomSlice(allocator, batch * k * n, 0xc);
    defer allocator.free(b_data);

    var a = try Tensor.fromSlice(allocator, &.{ batch, m, k }, a_data);
    defer a.deinit();
    var b = try Tensor.fromSlice(allocator, &.{ batch, k, n }, b_data);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ batch, m, n });
    defer out.deinit();

    var pool: raw_backend.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    const native_config: native.ParallelConfig = .{ .pool = &pool };

    const NativeRunner = struct {
        fn run(o: *Tensor, lhs: *const Tensor, rhs: *const Tensor, rows: usize, cols: usize, inner: usize, batches: usize, config: native.ParallelConfig) void {
            native.matmulBatched2DIntoUncheckedWithConfig(o, lhs, rhs, rows, cols, inner, batches, rows * inner, inner * cols, rows * cols, config);
        }
    }.run;

    const scalar_ns = try medianTimer(scalar.matmulBatched2DIntoUnchecked, .{ &out, &a, &b, m, n, k, batch, m * k, k * n, m * n });
    const native_ns = try medianTimer(NativeRunner, .{ &out, &a, &b, m, n, k, batch, native_config });
    try fmtRow(w, name, scalar_ns, native_ns);
}
