//! Scatter-add (embedding-gradient) kernel benchmark (`zig build bench-scatter`).
//!
//! Times `scatterAddAxisRank` axis==0 at the embedding-gradient shape: the
//! GatherBackward VJP scatters a (tokens, dim) gradient into a zeroed
//! (vocab, dim) table — for Qwen3 0.6B that is a 151936x1024 f32 output
//! (622 MB zero-fill) plus per-index row accumulation. Index distributions
//! cover the uniform case and a heavy-duplicate band (all tokens inside 1000
//! consecutive rows), the worst case for destination-partitioned parallelism.
//! Each iteration times op-output create/deinit, so buffer-pool behavior is in
//! the loop exactly as in training.
//!
//! Run in ReleaseFast: `zig build bench-scatter -Doptimize=ReleaseFast`.
const std = @import("std");
const bench_alloc = @import("alloc.zig");
const Timer = @import("timer.zig").Timer;
const bench_raw = @import("bench_raw");
const fucina = bench_raw;

const ExecContext = bench_raw.ExecContext;
const RawTensor = bench_raw.RawTensor;

const vocab = 151936;
const dim = 1024;

fn fillRandom(values: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (values) |*value| value.* = random.floatNorm(f32) * 2;
}

fn randomGrad(ctx: *ExecContext, allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u64) !RawTensor {
    const data = try allocator.alloc(f32, rows * cols);
    defer allocator.free(data);
    fillRandom(data, seed);
    return ctx.fromSliceRank(2, .{ rows, cols }, data);
}

/// `band == 0` draws uniform over the whole vocab; otherwise all indices land
/// inside `[vocab/2, vocab/2 + band)` — a heavy-duplicate hot band.
fn randomIndices(allocator: std.mem.Allocator, count: usize, band: usize, seed: u64) ![]usize {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const indices = try allocator.alloc(usize, count);
    for (indices) |*index| {
        index.* = if (band == 0)
            random.uintLessThan(usize, vocab)
        else
            vocab / 2 + random.uintLessThan(usize, band);
    }
    return indices;
}

fn benchScatterAdd(
    ctx: *ExecContext,
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: anytype,
    token_count: usize,
    band: usize,
    iters: usize,
) !void {
    var grad = try randomGrad(ctx, allocator, token_count, dim, 0xacc0 + token_count);
    defer grad.deinit();
    const indices = try randomIndices(allocator, token_count, band, 0xbdd0 + token_count);
    defer allocator.free(indices);

    for (0..2) |_| {
        var out = try ctx.scatterAddAxisRank(2, &grad, .{ vocab, dim }, 0, indices);
        out.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var out = try ctx.scatterAddAxisRank(2, &grad, .{ vocab, dim }, 0, indices);
        out.deinit();
    }
    const ns = timer.read() / iters;

    const name: []const u8 = if (band == 0) "uniform" else "1000-row band";
    const ms = @as(f64, @floatFromInt(ns)) / 1e6;
    // Bytes the kernel must move: the dense zero-fill write of the source-shaped
    // output plus read grad + read-modify-write of the destination rows.
    const bytes = @as(f64, @floatFromInt((vocab * dim + 3 * token_count * dim) * @sizeOf(f32)));
    const gbs = bytes / @as(f64, @floatFromInt(ns));
    try stdout.print("scatter-add {s:<14} {d:>6} idx -> {d} x {d}  {d:>9.3} ms {d:>7.2} GB/s  ({d} iters)\n", .{ name, token_count, vocab, dim, ms, gbs, iters });
    try stdout.flush();
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var mode: bench_alloc.AllocatorMode = .smp;
    for (args[1..]) |arg| {
        if (try bench_alloc.parseAllocatorModeArg(arg)) |parsed| {
            mode = parsed;
        }
    }
    var bench_allocator = bench_alloc.BenchmarkAllocator.init(mode);
    const allocator = bench_allocator.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.print("scatter-add (embedding-gradient) benchmark — backend={s}\n\n", .{@tagName(fucina.active_backend_kind)});

    try benchScatterAdd(&ctx, init.io, allocator, stdout, 2048, 0, 5);
    try benchScatterAdd(&ctx, init.io, allocator, stdout, 2048, 1000, 5);
    try benchScatterAdd(&ctx, init.io, allocator, stdout, 32768, 0, 5);
    try benchScatterAdd(&ctx, init.io, allocator, stdout, 32768, 1000, 5);
}
