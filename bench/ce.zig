//! Softmax / cross-entropy row-kernel benchmark (`zig build bench-ce`).
//!
//! Times the exec-level row kernels at LLM shapes: softmax forward over the
//! vocab axis (prefill logits), attention-shaped softmax rows, cross-entropy
//! forward + backward (training loss over the vocab), softmax backward,
//! dropout forward (the counter-based RNG mask), and layerNorm
//! forward/backward next to rmsNormMul at the same shapes (the sanity ratio
//! for the extra mean pass).
//! Each section creates the inputs once and times op-output create/deinit per
//! iteration, so buffer-pool reuse is in the loop exactly as in training.
//!
//! Run in ReleaseFast: `zig build bench-ce -Doptimize=ReleaseFast`.
const std = @import("std");
const bench_alloc = @import("alloc.zig");
const Timer = @import("timer.zig").Timer;
const bench_raw = @import("bench_raw");
const fucina = bench_raw;

const ExecContext = bench_raw.ExecContext;
const RawTensor = bench_raw.RawTensor;

fn fillRandom(values: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (values) |*value| value.* = random.floatNorm(f32) * 2;
}

fn randomLogits(ctx: *ExecContext, allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u64) !RawTensor {
    const data = try allocator.alloc(f32, rows * cols);
    defer allocator.free(data);
    fillRandom(data, seed);
    return ctx.fromSlice(.f32, .{ rows, cols }, data);
}

fn randomLabels(allocator: std.mem.Allocator, count: usize, class_count: usize, seed: u64) ![]usize {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const labels = try allocator.alloc(usize, count);
    for (labels) |*label| label.* = random.uintLessThan(usize, class_count);
    return labels;
}

fn benchSoftmaxForward(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype, rows: usize, cols: usize, iters: usize) !void {
    var x = try randomLogits(ctx, allocator, rows, cols, 0xabc0 + rows);
    defer x.deinit();

    for (0..2) |_| {
        var y = try ctx.softmax(.f32, 2, &x, 1);
        y.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var y = try ctx.softmax(.f32, 2, &x, 1);
        y.deinit();
    }
    const ns = timer.read() / iters;
    try printRow(stdout, "softmax fwd", rows, cols, ns, iters);
}

// Axis-0 variants: the reduced axis is NOT last (inner > 1), so these hit
// the strided fallback paths in exec/softmax.zig rather than the row kernels.
fn benchSoftmaxForwardAxis0(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype, rows: usize, cols: usize, iters: usize) !void {
    var x = try randomLogits(ctx, allocator, rows, cols, 0xa0c0 + rows);
    defer x.deinit();

    for (0..2) |_| {
        var y = try ctx.softmax(.f32, 2, &x, 0);
        y.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var y = try ctx.softmax(.f32, 2, &x, 0);
        y.deinit();
    }
    const ns = timer.read() / iters;
    try printRow(stdout, "softmax fwd ax0", rows, cols, ns, iters);
}

fn benchLogSoftmaxAxis0(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype, rows: usize, cols: usize, iters: usize) !void {
    var x = try randomLogits(ctx, allocator, rows, cols, 0xa1c0 + rows);
    defer x.deinit();

    for (0..2) |_| {
        var y = try ctx.logSoftmax(.f32, 2, &x, 0);
        y.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var y = try ctx.logSoftmax(.f32, 2, &x, 0);
        y.deinit();
    }
    const ns = timer.read() / iters;
    try printRow(stdout, "logsoftmax ax0", rows, cols, ns, iters);
}

fn benchLogsumexpAxis0(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype, rows: usize, cols: usize, iters: usize) !void {
    var x = try randomLogits(ctx, allocator, rows, cols, 0xa2c0 + rows);
    defer x.deinit();

    for (0..2) |_| {
        var y = try ctx.logsumexp(.f32, 2, &x, 0);
        y.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var y = try ctx.logsumexp(.f32, 2, &x, 0);
        y.deinit();
    }
    const ns = timer.read() / iters;
    try printRow(stdout, "logsumexp ax0", rows, cols, ns, iters);
}

fn benchSoftmaxBackwardAxis0(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype, rows: usize, cols: usize, iters: usize) !void {
    var x = try randomLogits(ctx, allocator, rows, cols, 0xa3c0 + rows);
    defer x.deinit();
    var y = try ctx.softmax(.f32, 2, &x, 0);
    defer y.deinit();
    var gy = try randomLogits(ctx, allocator, rows, cols, 0xa4c0 + rows);
    defer gy.deinit();

    for (0..2) |_| {
        var gx = try ctx.softmaxBackward(2, &y, &gy, 0, 1);
        gx.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var gx = try ctx.softmaxBackward(2, &y, &gy, 0, 1);
        gx.deinit();
    }
    const ns = timer.read() / iters;
    try printRow(stdout, "softmax bwd ax0", rows, cols, ns, iters);
}

fn benchLayerNormBackwardAxis0(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype, rows: usize, cols: usize, iters: usize) !void {
    var x = try randomLogits(ctx, allocator, rows, cols, 0xa5c0 + rows);
    defer x.deinit();
    var gy = try randomLogits(ctx, allocator, rows, cols, 0xa6c0 + rows);
    defer gy.deinit();
    var w = try randomRow(ctx, allocator, rows, 0xa7c0 + rows);
    defer w.deinit();
    const eps: f32 = 1e-5;

    for (0..2) |_| {
        var grads = try ctx.layerNormBackward(2, &x, &gy, 0, eps, .{ .weight = &w, .need_input = true, .need_weight = true, .need_bias = true });
        grads.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var grads = try ctx.layerNormBackward(2, &x, &gy, 0, eps, .{ .weight = &w, .need_input = true, .need_weight = true, .need_bias = true });
        grads.deinit();
    }
    const ns = timer.read() / iters;
    try printRow(stdout, "layernorm bwd ax0", rows, cols, ns, iters);
}

fn benchSoftmaxBackward(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype, rows: usize, cols: usize, iters: usize) !void {
    var x = try randomLogits(ctx, allocator, rows, cols, 0xbcd0 + rows);
    defer x.deinit();
    var y = try ctx.softmax(.f32, 2, &x, 1);
    defer y.deinit();
    var gy = try randomLogits(ctx, allocator, rows, cols, 0xcde0 + rows);
    defer gy.deinit();

    for (0..2) |_| {
        var gx = try ctx.softmaxBackward(2, &y, &gy, 1, 1);
        gx.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var gx = try ctx.softmaxBackward(2, &y, &gy, 1, 1);
        gx.deinit();
    }
    const ns = timer.read() / iters;
    try printRow(stdout, "softmax bwd", rows, cols, ns, iters);
}

fn benchCrossEntropy(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype, rows: usize, cols: usize, iters: usize) !void {
    var logits = try randomLogits(ctx, allocator, rows, cols, 0xdef0 + rows);
    defer logits.deinit();
    const labels = try randomLabels(allocator, rows, cols, 0xefa0 + rows);
    defer allocator.free(labels);

    for (0..2) |_| {
        var loss = try ctx.crossEntropyLoss(2, &logits, 1, labels, .{});
        loss.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var loss = try ctx.crossEntropyLoss(2, &logits, 1, labels, .{});
        loss.deinit();
    }
    const fwd_ns = timer.read() / iters;
    try printRow(stdout, "cross-entropy fwd", rows, cols, fwd_ns, iters);

    for (0..2) |_| {
        var grad = try ctx.crossEntropyBackward(2, &logits, 1, labels, .{}, .{ .scale = 1 });
        grad.deinit();
    }
    timer.reset();
    for (0..iters) |_| {
        var grad = try ctx.crossEntropyBackward(2, &logits, 1, labels, .{}, .{ .scale = 1 });
        grad.deinit();
    }
    const bwd_ns = timer.read() / iters;
    try printRow(stdout, "cross-entropy bwd", rows, cols, bwd_ns, iters);

    // Backward over forward-saved {max, sum_exp} — the route the autograd
    // node takes (bitwise identical to the recompute row above).
    const stats = try allocator.alloc(f32, 2 * rows);
    defer allocator.free(stats);
    var stats_loss = try ctx.crossEntropyLoss(2, &logits, 1, labels, .{ .row_stats = stats });
    stats_loss.deinit();
    for (0..2) |_| {
        var grad = try ctx.crossEntropyBackward(2, &logits, 1, labels, .{ .row_stats = stats }, .{ .scale = 1 });
        grad.deinit();
    }
    timer.reset();
    for (0..iters) |_| {
        var grad = try ctx.crossEntropyBackward(2, &logits, 1, labels, .{ .row_stats = stats }, .{ .scale = 1 });
        grad.deinit();
    }
    const stats_ns = timer.read() / iters;
    try printRow(stdout, "cross-entropy bwd+s", rows, cols, stats_ns, iters);
}

/// Fused linear+CE VJP (linearCrossEntropyBackwardUpstream) vs the composed
/// route (full dLogits, then dx and dweight GEMMs) over the SAME saved
/// logits/stats — the exact work the autograd node replaces. The fused VJP
/// consumes its logits (in-place dL), so its loop pays one logits clone per
/// iteration; the standalone clone row is printed for subtraction.
fn benchLinearCe(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype, rows: usize, classes: usize, in_dim: usize, iters: usize) !void {
    var x = try randomLogits(ctx, allocator, rows, in_dim, 0x11c0 + rows);
    defer x.deinit();
    var w = try randomLogits(ctx, allocator, classes, in_dim, 0x11c1 + rows);
    defer w.deinit();
    const labels = try randomLabels(allocator, rows, classes, 0x11c2 + rows);
    defer allocator.free(labels);
    const row_stats = try allocator.alloc(f32, 2 * rows);
    defer allocator.free(row_stats);

    var logits = try ctx.matmulTransB(&x, &w);
    defer logits.deinit();
    var loss = try ctx.crossEntropyLoss(2, &logits, 1, labels, .{ .row_stats = row_stats });
    loss.deinit();
    var gy = try ctx.fromSlice(.f32, .{1}, &.{1});
    defer gy.deinit();

    for (0..2) |_| {
        var dlogits = try ctx.crossEntropyBackward(2, &logits, 1, labels, .{ .row_stats = row_stats }, .{ .scale = 1 });
        defer dlogits.deinit();
        var dx = try ctx.matmul(.f32, &dlogits, &w);
        dx.deinit();
        var dw = try ctx.matmulTransA(&dlogits, &x);
        dw.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var dlogits = try ctx.crossEntropyBackward(2, &logits, 1, labels, .{ .row_stats = row_stats }, .{ .scale = 1 });
        defer dlogits.deinit();
        var dx = try ctx.matmul(.f32, &dlogits, &w);
        dx.deinit();
        var dw = try ctx.matmulTransA(&dlogits, &x);
        dw.deinit();
    }
    const composed_ns = timer.read() / iters;
    try printRow(stdout, "linear-ce bwd comp", rows, classes, composed_ns, iters);

    for (0..2) |_| {
        var l2 = try logits.clone(allocator);
        defer l2.deinit();
        var grads = try ctx.linearCrossEntropyBackwardUpstream(&x, &w, &l2, labels, .{}, &gy, row_stats, true, true);
        grads.deinit();
    }
    timer.reset();
    for (0..iters) |_| {
        var l2 = try logits.clone(allocator);
        defer l2.deinit();
        var grads = try ctx.linearCrossEntropyBackwardUpstream(&x, &w, &l2, labels, .{}, &gy, row_stats, true, true);
        grads.deinit();
    }
    const fused_ns = timer.read() / iters;
    try printRow(stdout, "linear-ce bwd fus+c", rows, classes, fused_ns, iters);

    timer.reset();
    for (0..iters) |_| {
        var l2 = try logits.clone(allocator);
        l2.deinit();
    }
    const clone_ns = timer.read() / iters;
    try printRow(stdout, "linear-ce clone", rows, classes, clone_ns, iters);
}

fn benchDropout(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype, rows: usize, cols: usize, iters: usize) !void {
    var x = try randomLogits(ctx, allocator, rows, cols, 0xd120 + rows);
    defer x.deinit();

    for (0..2) |_| {
        var y = try ctx.dropoutForward(&x, 0.1, 0x5eed);
        y.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var y = try ctx.dropoutForward(&x, 0.1, 0x5eed);
        y.deinit();
    }
    const ns = timer.read() / iters;
    try printRow(stdout, "dropout fwd", rows, cols, ns, iters);
}

fn randomRow(ctx: *ExecContext, allocator: std.mem.Allocator, cols: usize, seed: u64) !RawTensor {
    const data = try allocator.alloc(f32, cols);
    defer allocator.free(data);
    fillRandom(data, seed);
    return ctx.fromSlice(.f32, .{cols}, data);
}

fn benchLayerNorm(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype, rows: usize, cols: usize, iters: usize) !void {
    var x = try randomLogits(ctx, allocator, rows, cols, 0xfa10 + rows);
    defer x.deinit();
    var gy = try randomLogits(ctx, allocator, rows, cols, 0xfb20 + rows);
    defer gy.deinit();
    var w = try randomRow(ctx, allocator, cols, 0xfc30 + rows);
    defer w.deinit();
    var b = try randomRow(ctx, allocator, cols, 0xfd40 + rows);
    defer b.deinit();
    const eps: f32 = 1e-5;

    for (0..2) |_| {
        var y = try ctx.layerNorm(.f32, 2, &x, 1, eps, .{ .weight = &w, .bias = &b });
        y.deinit();
    }
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var y = try ctx.layerNorm(.f32, 2, &x, 1, eps, .{ .weight = &w, .bias = &b });
        y.deinit();
    }
    const fwd_ns = timer.read() / iters;
    try printRow(stdout, "layernorm-aff fwd", rows, cols, fwd_ns, iters);

    for (0..2) |_| {
        var grads = try ctx.layerNormBackward(2, &x, &gy, 1, eps, .{ .weight = &w, .need_input = true, .need_weight = true, .need_bias = true });
        grads.deinit();
    }
    timer.reset();
    for (0..iters) |_| {
        var grads = try ctx.layerNormBackward(2, &x, &gy, 1, eps, .{ .weight = &w, .need_input = true, .need_weight = true, .need_bias = true });
        grads.deinit();
    }
    const bwd_ns = timer.read() / iters;
    try printRow(stdout, "layernorm-aff bwd", rows, cols, bwd_ns, iters);

    // rmsNormMul at the same shape: the sanity baseline (layerNorm should sit
    // within ~1.5-2x given the extra mean pass).
    for (0..2) |_| {
        var y = try ctx.rmsNorm(.f32, 2, &x, 1, eps, .{ .weight = &w });
        y.deinit();
    }
    timer.reset();
    for (0..iters) |_| {
        var y = try ctx.rmsNorm(.f32, 2, &x, 1, eps, .{ .weight = &w });
        y.deinit();
    }
    const rms_fwd_ns = timer.read() / iters;
    try printRow(stdout, "rmsnorm-mul fwd", rows, cols, rms_fwd_ns, iters);

    for (0..2) |_| {
        var grads = try ctx.rmsNormBackward(2, &x, &gy, 1, eps, .{ .weight = &w, .need_input = true, .need_weight = true });
        grads.deinit();
    }
    timer.reset();
    for (0..iters) |_| {
        var grads = try ctx.rmsNormBackward(2, &x, &gy, 1, eps, .{ .weight = &w, .need_input = true, .need_weight = true });
        grads.deinit();
    }
    const rms_bwd_ns = timer.read() / iters;
    try printRow(stdout, "rmsnorm-mul bwd", rows, cols, rms_bwd_ns, iters);
}

fn printRow(stdout: anytype, name: []const u8, rows: usize, cols: usize, ns: u64, iters: usize) !void {
    const ms = @as(f64, @floatFromInt(ns)) / 1e6;
    const elems = @as(f64, @floatFromInt(rows * cols));
    const gelems = elems / @as(f64, @floatFromInt(ns));
    try stdout.print("{s:<18} {d:>6} x {d:<7} {d:>10.3} ms {d:>8.2} Gelem/s  ({d} iters)\n", .{ name, rows, cols, ms, gelems, iters });
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

    try stdout.print("softmax / cross-entropy row-kernel benchmark — backend={s}\n\n", .{@tagName(fucina.active_backend_kind)});

    try benchSoftmaxForward(&ctx, init.io, allocator, stdout, 2048, 151936, 5);
    try benchSoftmaxForward(&ctx, init.io, allocator, stdout, 1024, 4096, 50);
    try benchSoftmaxForward(&ctx, init.io, allocator, stdout, 32 * 1024, 1024, 20);

    try benchCrossEntropy(&ctx, init.io, allocator, stdout, 1024, 151936, 5);
    try benchCrossEntropy(&ctx, init.io, allocator, stdout, 4096, 32000, 5);

    try benchLinearCe(&ctx, init.io, allocator, stdout, 1024, 151936, 1024, 3);

    try benchSoftmaxBackward(&ctx, init.io, allocator, stdout, 1024, 4096, 50);

    try benchSoftmaxForwardAxis0(&ctx, init.io, allocator, stdout, 4096, 1024, 50);
    try benchSoftmaxForwardAxis0(&ctx, init.io, allocator, stdout, 1024, 4096, 50);
    try benchLogSoftmaxAxis0(&ctx, init.io, allocator, stdout, 4096, 1024, 50);
    try benchLogsumexpAxis0(&ctx, init.io, allocator, stdout, 4096, 1024, 50);
    try benchSoftmaxBackwardAxis0(&ctx, init.io, allocator, stdout, 4096, 1024, 50);
    try benchLayerNormBackwardAxis0(&ctx, init.io, allocator, stdout, 4096, 1024, 20);

    try benchDropout(&ctx, init.io, allocator, stdout, 1024, 1024, 200);

    try benchLayerNorm(&ctx, init.io, allocator, stdout, 4096, 1024, 100);
    try benchLayerNorm(&ctx, init.io, allocator, stdout, 1024, 4096, 100);
}
