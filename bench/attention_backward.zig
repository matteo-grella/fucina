const std = @import("std");
const bench_alloc = @import("alloc.zig");
const Timer = @import("timer.zig").Timer;
const bench_raw = @import("bench_raw");
const fucina = bench_raw;

const ExecContext = bench_raw.ExecContext;
const Tensor = bench_raw.RawTensor;

const Case = struct {
    name: []const u8,
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    iters: usize,
    // Feed forward-saved softmax {max, sum_exp} stats to the backward (the
    // autograd-record route); false = the stats-less recompute route.
    stats: bool = false,
};

const Result = struct {
    case: Case,
    ns_per_op: u64,
    allocs_per_op: usize,
    bytes_per_op: usize,
    live_bytes: usize,
    checksum: f64,
};

const cases = [_]Case{
    .{ .name = "s16_h4_kv2_d16", .q_seq = 16, .kv_seq = 16, .heads = 4, .kv_heads = 2, .d = 16, .iters = 80 },
    .{ .name = "s64_h8_kv2_d64", .q_seq = 64, .kv_seq = 64, .heads = 8, .kv_heads = 2, .d = 64, .iters = 35 },
    .{ .name = "s128_h16_kv4_d64", .q_seq = 128, .kv_seq = 128, .heads = 16, .kv_heads = 4, .d = 64, .iters = 18 },
    .{ .name = "s256_h16_kv4_d64", .q_seq = 256, .kv_seq = 256, .heads = 16, .kv_heads = 4, .d = 64, .iters = 8 },
    .{ .name = "s256_h16_kv4_d64+s", .q_seq = 256, .kv_seq = 256, .heads = 16, .kv_heads = 4, .d = 64, .iters = 8, .stats = true },
    .{ .name = "s512_h16_kv4_d64", .q_seq = 512, .kv_seq = 512, .heads = 16, .kv_heads = 4, .d = 64, .iters = 6 },
    .{ .name = "s512_h16_kv4_d64+s", .q_seq = 512, .kv_seq = 512, .heads = 16, .kv_heads = 4, .d = 64, .iters = 6, .stats = true },
    .{ .name = "s1024_h16_kv4_d64", .q_seq = 1024, .kv_seq = 1024, .heads = 16, .kv_heads = 4, .d = 64, .iters = 4 },
    .{ .name = "s1024_h16_kv4_d64+s", .q_seq = 1024, .kv_seq = 1024, .heads = 16, .kv_heads = 4, .d = 64, .iters = 4, .stats = true },
    .{ .name = "s2048_h16_kv4_d64", .q_seq = 2048, .kv_seq = 2048, .heads = 16, .kv_heads = 4, .d = 64, .iters = 3 },
    .{ .name = "s2048_h16_kv4_d64+s", .q_seq = 2048, .kv_seq = 2048, .heads = 16, .kv_heads = 4, .d = 64, .iters = 3, .stats = true },
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const allocator_mode = try bench_alloc.parseAllocatorMode(args);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.writeAll("runtime,backend,case,q_seq,kv_seq,heads,kv_heads,d,iters,ns_per_op,approx_gflops,allocs_per_op,bytes_per_op,live_bytes,checksum\n");
    for (cases) |case| {
        const result = try runCase(init.io, allocator_mode, case);
        try printResult(stdout, result);
    }
}

fn printResult(writer: anytype, result: Result) !void {
    const flops = @as(f64, @floatFromInt(result.case.q_seq)) *
        @as(f64, @floatFromInt(result.case.heads)) *
        @as(f64, @floatFromInt(result.case.kv_seq)) *
        @as(f64, @floatFromInt(result.case.d)) *
        10.0;
    const gflops = flops / @as(f64, @floatFromInt(result.ns_per_op));
    try writer.print("zig,{s},{s},{d},{d},{d},{d},{d},{d},{d},{d:.3},{d},{d},{d},{d:.6}\n", .{
        @tagName(fucina.active_backend_kind),
        result.case.name,
        result.case.q_seq,
        result.case.kv_seq,
        result.case.heads,
        result.case.kv_heads,
        result.case.d,
        result.case.iters,
        result.ns_per_op,
        gflops,
        result.allocs_per_op,
        result.bytes_per_op,
        result.live_bytes,
        result.checksum,
    });
}

fn runCase(io: std.Io, allocator_mode: bench_alloc.AllocatorMode, case: Case) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = bench_alloc.CountingAllocator.init(benchmark_allocator.allocator());

    var ctx: ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var base = try Base.init(&ctx, case);
    defer base.deinit();

    // Forward-saved stats + output for the "+s" rows — the autograd
    // record's route (one-pass softmax rebuild + gy.O row dots).
    var stats: ?[]f32 = null;
    defer if (stats) |values| counted.allocator().free(values);
    var stats_out: ?Tensor = null;
    defer if (stats_out) |*value| value.deinit();
    if (case.stats) {
        const values = try counted.allocator().alloc(f32, case.heads * case.q_seq * 2);
        stats = values;
        stats_out = try ctx.groupedCausalAttentionStatsOut(&base.q, &base.k, &base.v, base.kv_head_for_head, 0.125, 0, true, values);
    }
    const out_arg: ?*const Tensor = if (stats_out) |*value| value else null;

    for (0..3) |_| {
        var grads = try ctx.groupedCausalAttentionBackward(&base.q, &base.k, &base.v, &base.gy, base.kv_head_for_head, 0.125, 0, true, stats, out_arg, true, true, true);
        std.mem.doNotOptimizeAway(checksum(&grads));
        grads.deinit();
    }

    counted.resetWindow();
    var checksum_value: f64 = 0;
    const allocator = counted.allocator();
    const times = try allocator.alloc(u64, case.iters);
    defer allocator.free(times);
    var timer = try Timer.start(io);
    for (times) |*time| {
        timer.reset();
        var grads = try ctx.groupedCausalAttentionBackward(&base.q, &base.k, &base.v, &base.gy, base.kv_head_for_head, 0.125, 0, true, stats, out_arg, true, true, true);
        time.* = timer.read();
        checksum_value += checksum(&grads);
        grads.deinit();
    }
    std.mem.sort(u64, times, {}, std.sort.asc(u64));

    return .{
        .case = case,
        .ns_per_op = times[times.len / 2],
        .allocs_per_op = counted.alloc_count / case.iters,
        .bytes_per_op = counted.bytes_allocated / case.iters,
        .live_bytes = counted.peak_live,
        .checksum = checksum_value,
    };
}

const Base = struct {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gy: Tensor,
    kv_head_for_head: []usize,
    allocator: std.mem.Allocator,

    fn init(ctx: *ExecContext, case: Case) !Base {
        var q = try ctx.emptyRank(3, .{ case.q_seq, case.heads, case.d });
        errdefer q.deinit();
        var k = try ctx.emptyRank(3, .{ case.kv_seq, case.kv_heads, case.d });
        errdefer k.deinit();
        var v = try ctx.emptyRank(3, .{ case.kv_seq, case.kv_heads, case.d });
        errdefer v.deinit();
        var gy = try ctx.emptyRank(2, .{ case.q_seq, case.heads * case.d });
        errdefer gy.deinit();
        const kv_head_for_head = try ctx.allocator.alloc(usize, case.heads);
        errdefer ctx.allocator.free(kv_head_for_head);

        fillPattern(&q, 1);
        fillPattern(&k, 2);
        fillPattern(&v, 3);
        fillPattern(&gy, 4);
        const group = case.heads / case.kv_heads;
        for (kv_head_for_head, 0..) |*mapped, head_i| mapped.* = head_i / group;

        return .{
            .q = q,
            .k = k,
            .v = v,
            .gy = gy,
            .kv_head_for_head = kv_head_for_head,
            .allocator = ctx.allocator,
        };
    }

    fn deinit(self: *Base) void {
        self.allocator.free(self.kv_head_for_head);
        self.gy.deinit();
        self.v.deinit();
        self.k.deinit();
        self.q.deinit();
        self.* = undefined;
    }
};

fn fillPattern(t: *Tensor, seed: usize) void {
    for (t.data(), 0..) |*value, i| {
        const x = @as(f32, @floatFromInt(i + seed * 17));
        value.* = @sin(x * 0.011) * 0.25 + @cos(x * 0.017) * 0.15;
    }
}

fn checksum(grads: anytype) f64 {
    var total: f64 = 0;
    if (grads.q) |q| total += @as(f64, @floatCast(q.dataConst()[0]));
    if (grads.k) |k| total += @as(f64, @floatCast(k.dataConst()[0]));
    if (grads.v) |v| total += @as(f64, @floatCast(v.dataConst()[0]));
    return total;
}
