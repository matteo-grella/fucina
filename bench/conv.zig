const std = @import("std");
const bench_alloc = @import("alloc.zig");
const Timer = @import("timer.zig").Timer;
const bench_raw = @import("bench_raw");
const fucina = bench_raw;

const Tensor = bench_raw.RawTensor;
const ExecContext = bench_raw.ExecContext;

const Case = struct {
    name: []const u8,
    h: usize,
    w: usize,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    groups: usize,
    forward_iters: usize,
    backward_iters: usize,
};

const Mode = enum {
    forward,
    backward_input,
    backward_weight,
};

const Result = struct {
    runtime: []const u8 = "zig",
    backend: []const u8,
    mode: Mode,
    case: Case,
    iters: usize,
    ns_per_op: u64,
    allocs_per_op: usize,
    bytes_per_op: usize,
    live_bytes: usize,
    checksum: f64,
};

const quick_cases = [_]Case{
    .{ .name = "s1_56x56x64_o64_k3", .h = 56, .w = 56, .cin = 64, .cout = 64, .k = 3, .stride = 1, .pad = 1, .groups = 1, .forward_iters = 6, .backward_iters = 2 },
    .{ .name = "tiny_4x4x1_o4_k3", .h = 4, .w = 4, .cin = 1, .cout = 4, .k = 3, .stride = 1, .pad = 1, .groups = 1, .forward_iters = 50, .backward_iters = 20 },
};

// Backward iters stay LOW: the direct backward kernels run ~100-200 ms per
// direction at the mid shapes.
const full_cases = [_]Case{
    .{ .name = "s1_112x112x32_o32_k3", .h = 112, .w = 112, .cin = 32, .cout = 32, .k = 3, .stride = 1, .pad = 1, .groups = 1, .forward_iters = 20, .backward_iters = 4 },
    .{ .name = "s1_56x56x64_o64_k3", .h = 56, .w = 56, .cin = 64, .cout = 64, .k = 3, .stride = 1, .pad = 1, .groups = 1, .forward_iters = 20, .backward_iters = 4 },
    .{ .name = "s1_28x28x128_o128_k3", .h = 28, .w = 28, .cin = 128, .cout = 128, .k = 3, .stride = 1, .pad = 1, .groups = 1, .forward_iters = 20, .backward_iters = 6 },
    .{ .name = "s2_112x112x32_o64_k3", .h = 112, .w = 112, .cin = 32, .cout = 64, .k = 3, .stride = 2, .pad = 1, .groups = 1, .forward_iters = 20, .backward_iters = 4 },
    .{ .name = "pw_56x56x64_o128_k1", .h = 56, .w = 56, .cin = 64, .cout = 128, .k = 1, .stride = 1, .pad = 0, .groups = 1, .forward_iters = 20, .backward_iters = 8 },
    .{ .name = "dw_56x56x64_k3", .h = 56, .w = 56, .cin = 64, .cout = 64, .k = 3, .stride = 1, .pad = 1, .groups = 64, .forward_iters = 40, .backward_iters = 8 },
    .{ .name = "tiny_4x4x1_o4_k3", .h = 4, .w = 4, .cin = 1, .cout = 4, .k = 3, .stride = 1, .pad = 1, .groups = 1, .forward_iters = 200, .backward_iters = 100 },
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var quick = false;
    var allocator_mode: bench_alloc.AllocatorMode = .debug;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--quick")) {
            quick = true;
        } else if (try bench_alloc.parseAllocatorModeArg(arg)) |mode| {
            allocator_mode = mode;
        } else {
            return error.UnknownArgument;
        }
    }

    const cases: []const Case = if (quick) quick_cases[0..] else full_cases[0..];
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};
    try stdout.writeAll("runtime,backend,mode,case,h,w,cin,cout,k,stride,groups,iters,ns_per_op,allocs_per_op,bytes_per_op,live_bytes,checksum\n");

    for (cases) |case| {
        try printResult(stdout, try runCase(init.io, allocator_mode, case, .forward));
        try printResult(stdout, try runCase(init.io, allocator_mode, case, .backward_input));
        try printResult(stdout, try runCase(init.io, allocator_mode, case, .backward_weight));
    }
}

fn printResult(writer: anytype, result: Result) !void {
    try writer.print(
        "{s},{s},{s},{s},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d:.6}\n",
        .{
            result.runtime,
            result.backend,
            @tagName(result.mode),
            result.case.name,
            result.case.h,
            result.case.w,
            result.case.cin,
            result.case.cout,
            result.case.k,
            result.case.stride,
            result.case.groups,
            result.iters,
            result.ns_per_op,
            result.allocs_per_op,
            result.bytes_per_op,
            result.live_bytes,
            result.checksum,
        },
    );
}

fn runCase(io: std.Io, allocator_mode: bench_alloc.AllocatorMode, case: Case, comptime mode: Mode) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = bench_alloc.CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var base = try Base.init(&ctx, case);
    defer base.deinit();

    const warmup_iters: usize = if (mode == .forward) 2 else 1;
    for (0..warmup_iters) |_| {
        var y = try runOp(&ctx, &base, mode);
        y.deinit();
    }

    counted.resetWindow();
    const iters = switch (mode) {
        .forward => case.forward_iters,
        .backward_input, .backward_weight => case.backward_iters,
    };

    var checksum: f64 = 0;
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        var y = try runOp(&ctx, &base, mode);
        const data = y.dataConst();
        checksum += @as(f64, @floatCast(data[0])) + @as(f64, @floatCast(data[data.len - 1]));
        y.deinit();
    }
    const elapsed = timer.read();

    return .{
        .backend = @tagName(fucina.active_backend_kind),
        .mode = mode,
        .case = case,
        .iters = iters,
        .ns_per_op = elapsed / iters,
        .allocs_per_op = counted.alloc_count / iters,
        .bytes_per_op = counted.bytes_allocated / iters,
        .live_bytes = counted.peak_live,
        .checksum = checksum,
    };
}

fn runOp(ctx: *ExecContext, base: *const Base, comptime mode: Mode) !Tensor {
    const c = base.case;
    const stride = [2]usize{ c.stride, c.stride };
    const pad = [2]usize{ c.pad, c.pad };
    return switch (mode) {
        .forward => ctx.conv2d(&base.input, &base.weight, null, stride, pad, c.groups),
        .backward_input => ctx.conv2dBackwardInput(&base.gy, &base.weight, c.h, c.w, stride, pad, c.groups),
        .backward_weight => ctx.conv2dBackwardWeight(&base.input, &base.gy, c.k, c.k, stride, pad, c.groups),
    };
}

const Base = struct {
    case: Case,
    input: Tensor,
    weight: Tensor,
    gy: Tensor,

    fn init(ctx: *ExecContext, case: Case) !Base {
        const oh = (case.h + 2 * case.pad - case.k) / case.stride + 1;
        const ow = (case.w + 2 * case.pad - case.k) / case.stride + 1;

        var input = try ctx.emptyRank(3, .{ case.h, case.w, case.cin });
        errdefer input.deinit();
        fillPattern(&input, 1);

        var weight = try ctx.emptyRank(4, .{ case.cout, case.k, case.k, case.cin / case.groups });
        errdefer weight.deinit();
        fillPattern(&weight, 2);

        var gy = try ctx.emptyRank(3, .{ oh, ow, case.cout });
        errdefer gy.deinit();
        fillPattern(&gy, 3);

        return .{
            .case = case,
            .input = input,
            .weight = weight,
            .gy = gy,
        };
    }

    fn deinit(self: *Base) void {
        self.gy.deinit();
        self.weight.deinit();
        self.input.deinit();
    }
};

fn fillPattern(t: *Tensor, seed: usize) void {
    for (t.data(), 0..) |*value, i| {
        const mixed = (i * 17 + seed * 31) % 97;
        const centered: i32 = @as(i32, @intCast(mixed)) - 48;
        value.* = @as(f32, @floatFromInt(centered)) * 0.0025;
    }
}
