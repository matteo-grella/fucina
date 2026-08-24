//! Masked-reduction benchmark (`zig build bench-masked-reduce`).
//!
//! The masked reductions exist to fuse a two-pass composition. The status-quo
//! spelling of "sum the elements a mask selects" is `maskedFill` into a
//! full-size f32 temporary followed by an ordinary reduction: two allocations'
//! worth of traffic and two passes over the data. The fused entries accumulate
//! straight out of the source through the same SIMD row kernel.
//!
//! Three arms per shape, so both claims are visible at once:
//!   * `unmasked`  — the plain reduction, the speed ceiling and the
//!                   no-regression reference (its code path is untouched).
//!   * `composed`  — maskedFill + reduce, what the fused entry replaces.
//!   * `fused`     — the masked entry.
//!
//! Shapes cover the two layouts that dispatch differently: a last-axis
//! reduction (the SIMD row-kernel arm, e.g. padding-masked pooling over a
//! feature axis) and a non-last-axis reduction (the linear-accumulate arm).
//! Each iteration times op-output create/deinit, so buffer-pool behavior is in
//! the loop exactly as in a real forward.
//!
//! Run in ReleaseFast: `zig build bench-masked-reduce -Doptimize=ReleaseFast`.
const std = @import("std");
const bench_alloc = @import("alloc.zig");
const Timer = @import("timer.zig").Timer;
const bench_raw = @import("bench_raw");
const fucina = bench_raw;

const ExecContext = bench_raw.ExecContext;
const RawTensor = bench_raw.RawTensor;
const BoolTensor = bench_raw.RawTensorOf(.bool);

fn randomRows(ctx: *ExecContext, allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u64) !RawTensor {
    const data = try allocator.alloc(f32, rows * cols);
    defer allocator.free(data);
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (data) |*value| value.* = random.floatNorm(f32) * 2;
    return ctx.fromSlice(.f32, .{ rows, cols }, data);
}

/// `keep_fraction` of the entries select, drawn deterministically so the two
/// masked arms see identical work.
fn randomMask(ctx: *ExecContext, allocator: std.mem.Allocator, rows: usize, cols: usize, keep_fraction: f32, seed: u64) !BoolTensor {
    const data = try allocator.alloc(bool, rows * cols);
    defer allocator.free(data);
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (data) |*value| value.* = random.float(f32) < keep_fraction;
    return ctx.fromSlice(.bool, .{ rows, cols }, data);
}

fn invert(ctx: *ExecContext, mask: *const BoolTensor, allocator: std.mem.Allocator) !BoolTensor {
    const source = mask.dataConst();
    const data = try allocator.alloc(bool, source.len);
    defer allocator.free(data);
    for (data, source) |*dst, keep| dst.* = !keep;
    return ctx.fromSlice(.bool, .{ mask.shape.at(0), mask.shape.at(1) }, data);
}

fn report(stdout: anytype, label: []const u8, ns: u64, iterations: usize) !void {
    const per_iter = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iterations));
    try stdout.print("  {s:<10} {d:>10.1} us/iter\n", .{ label, per_iter / 1000.0 });
}

fn benchAxis(
    ctx: *ExecContext,
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: anytype,
    comptime axis: usize,
    rows: usize,
    cols: usize,
    keep_fraction: f32,
    iterations: usize,
) !void {
    var x = try randomRows(ctx, allocator, rows, cols, 0x51ce);
    defer x.deinit();
    var mask = try randomMask(ctx, allocator, rows, cols, keep_fraction, 0xb175);
    defer mask.deinit();
    var drop = try invert(ctx, &mask, allocator);
    defer drop.deinit();

    try stdout.print("sum over axis {d}: [{d} x {d}], keep {d:.0}%\n", .{ axis, rows, cols, keep_fraction * 100 });

    // Warm the pool so the first timed iteration is not the one that allocates.
    {
        var warm = try ctx.sumMasked(.bool, 2, &x, &mask, axis, null);
        warm.deinit();
    }

    var timer = try Timer.start(io);
    for (0..iterations) |_| {
        var out = try ctx.sumAxis(.f32, 2, &x, axis);
        out.deinit();
    }
    try report(stdout, "unmasked", timer.read(), iterations);

    timer.reset();
    for (0..iterations) |_| {
        var zeroed = try ctx.maskedFill(.f32, .bool, &x, &drop, 0);
        var out = try ctx.sumAxis(.f32, 2, &zeroed, axis);
        out.deinit();
        zeroed.deinit();
    }
    try report(stdout, "composed", timer.read(), iterations);

    timer.reset();
    for (0..iterations) |_| {
        var out = try ctx.sumMasked(.bool, 2, &x, &mask, axis, null);
        out.deinit();
    }
    try report(stdout, "fused", timer.read(), iterations);
    try stdout.print("\n", .{});
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

    try stdout.print("masked-reduction benchmark — backend={s}\n\n", .{@tagName(fucina.active_backend_kind)});

    // Last-axis arm (SIMD row kernel): masked pooling over a feature axis.
    try benchAxis(&ctx, init.io, allocator, stdout, 1, 1024, 4096, 0.5, 50);
    try benchAxis(&ctx, init.io, allocator, stdout, 1, 4096, 1024, 0.9, 50);
    // Non-last-axis arm (linear accumulate): masked pooling over a token axis.
    try benchAxis(&ctx, init.io, allocator, stdout, 0, 4096, 1024, 0.5, 20);
}
