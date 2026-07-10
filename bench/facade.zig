const std = @import("std");
const bench_alloc = @import("alloc.zig");
const Timer = @import("timer.zig").Timer;
const bench_raw = @import("bench_raw");
const fucina = bench_raw;

const Allocator = std.mem.Allocator;
const RawExecContext = bench_raw.ExecContext;
const RawTensor = bench_raw.RawTensor;

const default_iterations = 200;
const heavy_iterations = 50;
const view_iterations = 10_000;

var benchmark_io: std.Io = undefined;
var benchmark_allocator_mode: bench_alloc.AllocatorMode = .debug;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    benchmark_io = init.io;
    benchmark_allocator_mode = try bench_alloc.parseAllocatorMode(args);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};
    try stdout.writeAll("case,mode,n,iters,ns_per_op,allocs_per_op,bytes_per_op,checksum\n");
    try benchAdd(16_384, stdout);
    try benchAdd(1_000_000, stdout);
    try benchClamp(1_000_000, stdout);
    try benchSwiglu(16_384, stdout);
    try benchSwiglu(1_000_000, stdout);
    try benchSumLast(2048, 512, stdout);
    try benchNarrowLast(4096, 256, 64, 128, stdout);
    try benchConcatLast(2048, 128, 128, stdout);
    try benchSetRowsLast(2048, 512, stdout);
    try benchTopK(256, 512, 8, stdout);
    try benchDotPackedQ8_0(1, 512, 512, stdout);
    try benchDotPackedQ8_0(16, 512, 512, stdout);
    try benchGroupedCausalAttention(1, 256, 8, 2, 64, stdout);
    try benchGroupedCausalAttention(64, 64, 8, 2, 64, stdout);
    try benchMatmulTransB(64, 512, 512, stdout);
    try benchRopeTable(64, 8, 128, stdout);
}

fn benchAdd(n: usize, writer: anytype) !void {
    try printResult(writer, "add", "raw", n, try runRawAdd(n, default_iterations));
    try printResult(writer, "add", "public_tensor_no_grad", n, try runPublicAdd(n, default_iterations));
}

fn benchSwiglu(n: usize, writer: anytype) !void {
    try printResult(writer, "swiglu", "raw", n, try runRawSwiglu(n, default_iterations));
    try printResult(writer, "swiglu", "public_tensor_no_grad", n, try runPublicSwiglu(n, default_iterations));
}

fn benchClamp(n: usize, writer: anytype) !void {
    try printResult(writer, "clamp", "raw", n, try runRawClamp(n, default_iterations));
    try printResult(writer, "clamp", "public_tensor_no_grad", n, try runPublicClamp(n, default_iterations));
}

fn benchSumLast(rows: usize, cols: usize, writer: anytype) !void {
    const n = rows * cols;
    try printResult(writer, "sum_last_axis", "raw", n, try runRawSumLast(rows, cols, default_iterations));
    try printResult(writer, "sum_last_axis", "public_tensor_no_grad", n, try runPublicSumLast(rows, cols, default_iterations));
}

fn benchNarrowLast(rows: usize, cols: usize, start: usize, length: usize, writer: anytype) !void {
    const n = rows * length;
    try printResult(writer, "narrow_last_axis", "raw", n, try runRawNarrowLast(rows, cols, start, length, view_iterations));
    try printResult(writer, "narrow_last_axis", "public_tensor_no_grad", n, try runPublicNarrowLast(rows, cols, start, length, view_iterations));
}

fn benchConcatLast(rows: usize, left_cols: usize, right_cols: usize, writer: anytype) !void {
    const n = rows * (left_cols + right_cols);
    try printResult(writer, "concat_last_axis", "raw", n, try runRawConcatLast(rows, left_cols, right_cols, heavy_iterations));
    try printResult(writer, "concat_last_axis", "public_tensor_no_grad", n, try runPublicConcatLast(rows, left_cols, right_cols, heavy_iterations));
}

fn benchSetRowsLast(rows: usize, cols: usize, writer: anytype) !void {
    const n = rows * cols;
    try printResult(writer, "set_rows_last_axis", "raw", n, try runRawSetRowsLast(rows, cols, heavy_iterations));
    try printResult(writer, "set_rows_last_axis", "public_tensor_no_grad", n, try runPublicSetRowsLast(rows, cols, heavy_iterations));
}

fn benchTopK(rows: usize, cols: usize, k: usize, writer: anytype) !void {
    const n = rows * cols;
    try printResult(writer, "top_k_last_axis", "raw", n, try runRawTopK(rows, cols, k, heavy_iterations));
    try printResult(writer, "top_k_last_axis", "public_tensor_no_grad", n, try runPublicTopK(rows, cols, k, heavy_iterations));
}

fn benchDotPackedQ8_0(m: usize, k: usize, n: usize, writer: anytype) !void {
    const elems = m * n;
    try printResult(writer, "dot_packed_q8_0", "raw", elems, try runRawDotPackedQ8_0(m, k, n, default_iterations));
    try printResult(writer, "dot_packed_q8_0", "public_tensor_no_grad", elems, try runPublicDotPackedQ8_0(m, k, n, default_iterations));
}

fn benchGroupedCausalAttention(q_seq: usize, kv_seq: usize, heads: usize, kv_heads: usize, d: usize, writer: anytype) !void {
    const elems = q_seq * heads * d;
    try printResult(writer, "grouped_causal_attention", "raw", elems, try runRawGroupedCausalAttention(q_seq, kv_seq, heads, kv_heads, d, default_iterations));
    try printResult(writer, "grouped_causal_attention", "public_tensor_no_grad", elems, try runPublicGroupedCausalAttention(q_seq, kv_seq, heads, kv_heads, d, default_iterations));
}

fn benchMatmulTransB(m: usize, k: usize, n: usize, writer: anytype) !void {
    const elems = m * n;
    try printResult(writer, "matmul_trans_b", "raw", elems, try runRawMatmulTransB(m, k, n, default_iterations));
    try printResult(writer, "matmul_trans_b", "public_tensor_no_grad", elems, try runPublicMatmulTransB(m, k, n, default_iterations));
}

fn benchRopeTable(seq: usize, heads: usize, d: usize, writer: anytype) !void {
    const elems = seq * heads * d;
    try printResult(writer, "rope_table_full", "raw", elems, try runRawRopeTable(seq, heads, d, default_iterations));
    try printResult(writer, "rope_table_full", "public_tensor_no_grad", elems, try runPublicRopeTable(seq, heads, d, default_iterations));
}

fn printResult(writer: anytype, case: []const u8, mode: []const u8, n: usize, result: Result) !void {
    try writer.print(
        "{s},{s},{d},{d},{d},{d},{d},{d:.6}\n",
        .{ case, mode, n, result.iterations, result.ns_per_op, result.allocs_per_op, result.bytes_per_op, result.checksum },
    );
}

const Result = struct {
    iterations: usize,
    ns_per_op: u64,
    allocs_per_op: usize,
    bytes_per_op: usize,
    checksum: f64,
};

fn runRawAdd(n: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var a = try ctx.emptyRank(1, .{n});
    defer a.deinit();
    fillPattern(&a, 1);
    var b = try ctx.emptyRank(1, .{n});
    defer b.deinit();
    fillPattern(&b, 2);

    for (0..4) |_| {
        var y = try ctx.addRank(1, &a, &b);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.addRank(1, &a, &b);
        checksum += @as(f64, @floatCast(y.dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return .{
        .iterations = iterations,
        .ns_per_op = elapsed / iterations,
        .allocs_per_op = counted.alloc_count / iterations,
        .bytes_per_op = counted.bytes_allocated / iterations,
        .checksum = checksum,
    };
}

fn runPublicAdd(n: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var a_value = try ctx.emptyRank(1, .{n});
    fillPattern(&a_value, 1);
    var a = try fucina.Tensor(1).fromTensor(&ctx, a_value);
    defer a.deinit();

    var b_value = try ctx.emptyRank(1, .{n});
    fillPattern(&b_value, 2);
    var b = try fucina.Tensor(1).fromTensor(&ctx, b_value);
    defer b.deinit();

    for (0..4) |_| {
        var y = try a.add(&ctx, &b);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try a.add(&ctx, &b);
        checksum += @as(f64, @floatCast(y.asRawTensor().dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return .{
        .iterations = iterations,
        .ns_per_op = elapsed / iterations,
        .allocs_per_op = counted.alloc_count / iterations,
        .bytes_per_op = counted.bytes_allocated / iterations,
        .checksum = checksum,
    };
}

fn runRawSwiglu(n: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var a = try ctx.emptyRank(1, .{n});
    defer a.deinit();
    fillPattern(&a, 3);
    var gate = try ctx.emptyRank(1, .{n});
    defer gate.deinit();
    fillPattern(&gate, 4);

    for (0..4) |_| {
        var y = try ctx.swigluRank(1, &a, &gate);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.swigluRank(1, &a, &gate);
        checksum += @as(f64, @floatCast(y.dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runPublicSwiglu(n: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var a_value = try ctx.emptyRank(1, .{n});
    fillPattern(&a_value, 3);
    var a = try fucina.Tensor(1).fromTensor(&ctx, a_value);
    defer a.deinit();

    var gate_value = try ctx.emptyRank(1, .{n});
    fillPattern(&gate_value, 4);
    var gate = try fucina.Tensor(1).fromTensor(&ctx, gate_value);
    defer gate.deinit();

    for (0..4) |_| {
        var y = try a.swiglu(&ctx, &gate);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try a.swiglu(&ctx, &gate);
        checksum += @as(f64, @floatCast(y.asRawTensor().dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runRawClamp(n: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var x = try ctx.emptyRank(1, .{n});
    defer x.deinit();
    fillPattern(&x, 11);

    for (0..4) |_| {
        var y = try ctx.clamp(&x, -0.05, 0.05);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.clamp(&x, -0.05, 0.05);
        checksum += @as(f64, @floatCast(y.dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runPublicClamp(n: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var value = try ctx.emptyRank(1, .{n});
    fillPattern(&value, 11);
    var x = try fucina.Tensor(1).fromTensor(&ctx, value);
    defer x.deinit();

    for (0..4) |_| {
        var y = try x.clamp(&ctx, -0.05, 0.05);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try x.clamp(&ctx, -0.05, 0.05);
        checksum += @as(f64, @floatCast(y.asRawTensor().dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runRawSumLast(rows: usize, cols: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var x = try ctx.emptyRank(2, .{ rows, cols });
    defer x.deinit();
    fillPattern(&x, 12);

    for (0..4) |_| {
        var y = try ctx.sumAxisRank(2, &x, 1);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.sumAxisRank(2, &x, 1);
        checksum += @as(f64, @floatCast(y.dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runPublicSumLast(rows: usize, cols: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var value = try ctx.emptyRank(2, .{ rows, cols });
    fillPattern(&value, 12);
    var x = try fucina.Tensor(.{ .row, .d }).fromTensor(&ctx, value);
    defer x.deinit();

    for (0..4) |_| {
        var y = try x.sum(&ctx, .d);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try x.sum(&ctx, .d);
        checksum += @as(f64, @floatCast(y.asRawTensor().dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runRawNarrowLast(rows: usize, cols: usize, start: usize, length: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var x = try ctx.emptyRank(2, .{ rows, cols });
    defer x.deinit();
    fillPattern(&x, 5);

    for (0..8) |_| {
        var y = try ctx.narrowAxisRank(2, &x, 1, start, length);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.narrowAxisRank(2, &x, 1, start, length);
        checksum += @as(f64, @floatCast(firstRawValue(&y)));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runPublicNarrowLast(rows: usize, cols: usize, start: usize, length: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var value = try ctx.emptyRank(2, .{ rows, cols });
    fillPattern(&value, 5);
    var x = try fucina.Tensor(.{ .row, .d }).fromTensor(&ctx, value);
    defer x.deinit();

    for (0..8) |_| {
        var y = try x.narrow(&ctx, .d, start, length);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try x.narrow(&ctx, .d, start, length);
        checksum += @as(f64, @floatCast(firstRawValue(y.asRawTensor())));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runRawConcatLast(rows: usize, left_cols: usize, right_cols: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var a = try ctx.emptyRank(2, .{ rows, left_cols });
    defer a.deinit();
    fillPattern(&a, 6);
    var b = try ctx.emptyRank(2, .{ rows, right_cols });
    defer b.deinit();
    fillPattern(&b, 7);
    var inputs = [_]*const RawTensor{ &a, &b };

    for (0..4) |_| {
        var y = try ctx.concatAxisRank(2, &inputs, 1);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.concatAxisRank(2, &inputs, 1);
        checksum += @as(f64, @floatCast(y.dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runPublicConcatLast(rows: usize, left_cols: usize, right_cols: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var a_value = try ctx.emptyRank(2, .{ rows, left_cols });
    fillPattern(&a_value, 6);
    var a = try fucina.Tensor(.{ .row, .d }).fromTensor(&ctx, a_value);
    defer a.deinit();

    var b_value = try ctx.emptyRank(2, .{ rows, right_cols });
    fillPattern(&b_value, 7);
    var b = try fucina.Tensor(.{ .row, .d }).fromTensor(&ctx, b_value);
    defer b.deinit();

    for (0..4) |_| {
        var y = try a.concat(&ctx, .d, &.{&b});
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try a.concat(&ctx, .d, &.{&b});
        checksum += @as(f64, @floatCast(y.asRawTensor().dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runRawSetRowsLast(rows: usize, cols: usize, iterations: usize) !Result {
    const indices = [_]usize{ 0, 7, 31, 63, 127, 255 };
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var base = try ctx.emptyRank(2, .{ rows, cols });
    defer base.deinit();
    fillPattern(&base, 8);
    var update = try ctx.emptyRank(2, .{ rows, indices.len });
    defer update.deinit();
    fillPattern(&update, 9);

    for (0..4) |_| {
        var y = try ctx.setRowsAxisRank(2, &base, &update, 1, &indices);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.setRowsAxisRank(2, &base, &update, 1, &indices);
        checksum += @as(f64, @floatCast(y.dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runPublicSetRowsLast(rows: usize, cols: usize, iterations: usize) !Result {
    const indices = [_]usize{ 0, 7, 31, 63, 127, 255 };
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var base_value = try ctx.emptyRank(2, .{ rows, cols });
    fillPattern(&base_value, 8);
    var base = try fucina.Tensor(.{ .row, .d }).fromTensor(&ctx, base_value);
    defer base.deinit();

    var update_value = try ctx.emptyRank(2, .{ rows, indices.len });
    fillPattern(&update_value, 9);
    var update = try fucina.Tensor(.{ .row, .d }).fromTensor(&ctx, update_value);
    defer update.deinit();

    for (0..4) |_| {
        var y = try base.setRows(&ctx, .d, &indices, &update);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try base.setRows(&ctx, .d, &indices, &update);
        checksum += @as(f64, @floatCast(y.asRawTensor().dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runRawTopK(rows: usize, cols: usize, k: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var x = try ctx.emptyRank(2, .{ rows, cols });
    defer x.deinit();
    fillPattern(&x, 10);

    for (0..2) |_| {
        var y = try ctx.topKAxisRank(2, &x, 1, k);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.topKAxisRank(2, &x, 1, k);
        checksum += @as(f64, @floatCast(y.values.dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runPublicTopK(rows: usize, cols: usize, k: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var value = try ctx.emptyRank(2, .{ rows, cols });
    fillPattern(&value, 10);
    var x = try fucina.Tensor(.{ .row, .d }).fromTensor(&ctx, value);
    defer x.deinit();

    for (0..2) |_| {
        var y = try x.topK(&ctx, .d, k, .k);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try x.topK(&ctx, .d, k, .k);
        checksum += @as(f64, @floatCast(y.values.asRawTensor().dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runRawDotPackedQ8_0(m: usize, k: usize, n: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var x = try ctx.emptyRank(2, .{ m, k });
    defer x.deinit();
    fillPattern(&x, 11);

    const blocks = try counted.allocator().alloc(bench_raw.BlockQ8_0, n * (k / bench_raw.q8_0_block_size));
    defer counted.allocator().free(blocks);
    fillQ8_0Blocks(blocks);
    var w = try ctx.fromStorageSliceRankTyped(.q8_0, 2, .{ n, k }, blocks);
    defer w.deinit();
    var packed_rhs = try ctx.packMatmulRhsQ8_0x4(&w);
    defer packed_rhs.deinit();

    for (0..4) |_| {
        var y = try ctx.matmul2DWithPackedQ8_0x4Rhs(&x, &packed_rhs);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.matmul2DWithPackedQ8_0x4Rhs(&x, &packed_rhs);
        checksum += @as(f64, @floatCast(y.dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runPublicDotPackedQ8_0(m: usize, k: usize, n: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var x_value = try ctx.emptyRank(2, .{ m, k });
    fillPattern(&x_value, 11);
    var x = try fucina.Tensor(.{ .batch, .in }).fromTensor(&ctx, x_value);
    defer x.deinit();

    const blocks = try counted.allocator().alloc(bench_raw.BlockQ8_0, n * (k / bench_raw.q8_0_block_size));
    defer counted.allocator().free(blocks);
    fillQ8_0Blocks(blocks);
    var w = try fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } }).fromBlocks(&ctx, .{ n, k }, blocks);
    defer w.deinit();
    var packed_rhs = try w.packRhs(&ctx);
    defer packed_rhs.deinit();

    for (0..4) |_| {
        var y = try x.dotPacked(&ctx, &packed_rhs, .in, .out);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try x.dotPacked(&ctx, &packed_rhs, .in, .out);
        checksum += @as(f64, @floatCast(y.asRawTensor().dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runRawGroupedCausalAttention(q_seq: usize, kv_seq: usize, heads: usize, kv_heads: usize, d: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var q = try ctx.emptyRank(3, .{ q_seq, heads, d });
    defer q.deinit();
    fillPattern(&q, 12);
    var k = try ctx.emptyRank(3, .{ kv_seq, kv_heads, d });
    defer k.deinit();
    fillPattern(&k, 13);
    var v = try ctx.emptyRank(3, .{ kv_seq, kv_heads, d });
    defer v.deinit();
    fillPattern(&v, 14);

    const map = try counted.allocator().alloc(usize, heads);
    defer counted.allocator().free(map);
    for (map, 0..) |*m, h| m.* = h * kv_heads / heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    for (0..4) |_| {
        var y = try ctx.groupedCausalAttention(&q, &k, &v, map, scale);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.groupedCausalAttention(&q, &k, &v, map, scale);
        checksum += @as(f64, @floatCast(y.dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runPublicGroupedCausalAttention(q_seq: usize, kv_seq: usize, heads: usize, kv_heads: usize, d: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var q_value = try ctx.emptyRank(3, .{ q_seq, heads, d });
    fillPattern(&q_value, 12);
    var q = try fucina.Tensor(.{ .seq, .head, .d }).fromTensor(&ctx, q_value);
    defer q.deinit();
    var k_value = try ctx.emptyRank(3, .{ kv_seq, kv_heads, d });
    fillPattern(&k_value, 13);
    var k = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromTensor(&ctx, k_value);
    defer k.deinit();
    var v_value = try ctx.emptyRank(3, .{ kv_seq, kv_heads, d });
    fillPattern(&v_value, 14);
    var v = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromTensor(&ctx, v_value);
    defer v.deinit();

    const map = try counted.allocator().alloc(usize, heads);
    defer counted.allocator().free(map);
    for (map, 0..) |*m, h| m.* = h * kv_heads / heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    for (0..4) |_| {
        var y = try q.groupedAttention(&ctx, &k, &v, map, .attn, scale, .{});
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try q.groupedAttention(&ctx, &k, &v, map, .attn, scale, .{});
        checksum += @as(f64, @floatCast(y.asRawTensor().dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runRawMatmulTransB(m: usize, k: usize, n: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var a = try ctx.emptyRank(2, .{ m, k });
    defer a.deinit();
    fillPattern(&a, 15);
    var b = try ctx.emptyRank(2, .{ n, k });
    defer b.deinit();
    fillPattern(&b, 16);

    for (0..4) |_| {
        var y = try ctx.matmulTransB(&a, &b);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.matmulTransB(&a, &b);
        checksum += @as(f64, @floatCast(y.dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runPublicMatmulTransB(m: usize, k: usize, n: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var a_value = try ctx.emptyRank(2, .{ m, k });
    fillPattern(&a_value, 15);
    var a = try fucina.Tensor(.{ .m, .k }).fromTensor(&ctx, a_value);
    defer a.deinit();
    var b_value = try ctx.emptyRank(2, .{ n, k });
    fillPattern(&b_value, 16);
    var b = try fucina.Tensor(.{ .n, .k }).fromTensor(&ctx, b_value);
    defer b.deinit();

    for (0..4) |_| {
        var y = try a.matmul(&ctx, &b, .trans_b, .{ .m, .n });
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try a.matmul(&ctx, &b, .trans_b, .{ .m, .n });
        checksum += @as(f64, @floatCast(y.asRawTensor().dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn ropeBenchPositions(allocator: Allocator, seq: usize) ![]i32 {
    const positions = try allocator.alloc(i32, seq);
    for (positions, 0..) |*p, i| p.* = @intCast(i);
    return positions;
}

fn runRawRopeTable(seq: usize, heads: usize, d: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: RawExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var x = try ctx.emptyRank(3, .{ seq, heads, d });
    defer x.deinit();
    fillPattern(&x, 17);

    const positions = try ropeBenchPositions(counted.allocator(), seq);
    defer counted.allocator().free(positions);
    var table = try ctx.prepareRopeTable(positions, d, 10000, false);
    defer table.deinit();

    for (0..4) |_| {
        var y = try ctx.ropePartialAxisRankWithTable(3, &x, 0, 2, &table, .half);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try ctx.ropePartialAxisRankWithTable(3, &x, 0, 2, &table, .half);
        checksum += @as(f64, @floatCast(y.dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn runPublicRopeTable(seq: usize, heads: usize, d: usize, iterations: usize) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: fucina.ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var x_value = try ctx.emptyRank(3, .{ seq, heads, d });
    fillPattern(&x_value, 17);
    var x = try fucina.Tensor(.{ .seq, .head, .d }).fromTensor(&ctx, x_value);
    defer x.deinit();

    const positions = try ropeBenchPositions(counted.allocator(), seq);
    defer counted.allocator().free(positions);
    var table = try ctx.prepareRopeTable(positions, d, 10000, false);
    defer table.deinit();

    for (0..4) |_| {
        var y = try x.rope(&ctx, .seq, .d, &table, .half);
        y.deinit();
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        var y = try x.rope(&ctx, .seq, .d, &table, .half);
        checksum += @as(f64, @floatCast(y.asRawTensor().dataConst()[0]));
        y.deinit();
    }
    const elapsed = timer.read();

    return resultFromWindow(iterations, elapsed, &counted, checksum);
}

fn fillQ8_0Blocks(blocks: []bench_raw.BlockQ8_0) void {
    for (blocks, 0..) |*b, bi| {
        const d: f16 = @floatCast(0.02 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.d = @bitCast(d);
        for (&b.qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast((i * 23 + bi * 5) % 255)) - 127);
    }
}

fn resultFromWindow(iterations: usize, elapsed: u64, counted: *const CountingAllocator, checksum: f64) Result {
    return .{
        .iterations = iterations,
        .ns_per_op = elapsed / iterations,
        .allocs_per_op = counted.alloc_count / iterations,
        .bytes_per_op = counted.bytes_allocated / iterations,
        .checksum = checksum,
    };
}

fn firstRawValue(t: *const RawTensor) f32 {
    return t.buffer.data[t.offset];
}

fn fillPattern(t: anytype, seed: usize) void {
    for (t.data(), 0..) |*value, i| {
        const mixed = (i * 17 + seed * 31) % 97;
        const centered: i32 = @as(i32, @intCast(mixed)) - 48;
        value.* = @as(f32, @floatFromInt(centered)) * 0.0025;
    }
}

const CountingAllocator = struct {
    child: Allocator,
    alloc_count: usize = 0,
    free_count: usize = 0,
    bytes_allocated: usize = 0,
    live_bytes: usize = 0,
    peak_live: usize = 0,

    fn init(child: Allocator) CountingAllocator {
        return .{ .child = child };
    }

    fn allocator(self: *CountingAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn resetWindow(self: *CountingAllocator) void {
        self.alloc_count = 0;
        self.free_count = 0;
        self.bytes_allocated = 0;
        self.peak_live = self.live_bytes;
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const ptr = self.child.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.alloc_count += 1;
        self.bytes_allocated += len;
        self.live_bytes += len;
        self.peak_live = @max(self.peak_live, self.live_bytes);
        return ptr;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        if (!self.child.rawResize(buf, alignment, new_len, ret_addr)) return false;

        if (new_len > buf.len) {
            const delta = new_len - buf.len;
            self.bytes_allocated += delta;
            self.live_bytes += delta;
        } else {
            self.live_bytes -= buf.len - new_len;
        }
        self.peak_live = @max(self.peak_live, self.live_bytes);
        return true;
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const ptr = self.child.rawRemap(buf, alignment, new_len, ret_addr) orelse return null;
        if (new_len > buf.len) {
            const delta = new_len - buf.len;
            self.bytes_allocated += delta;
            self.live_bytes += delta;
        } else {
            self.live_bytes -= buf.len - new_len;
        }
        self.peak_live = @max(self.peak_live, self.live_bytes);
        return ptr;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.free_count += 1;
        self.live_bytes -= buf.len;
        self.child.rawFree(buf, alignment, ret_addr);
    }
};
