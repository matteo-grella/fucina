const std = @import("std");
const bench_alloc = @import("alloc.zig");
const Timer = @import("timer.zig").Timer;
const bench_raw = @import("bench_raw");
const fucina = bench_raw;

const ExecContext = bench_raw.ExecContext;
const Tensor = bench_raw.RawTensor;
const Io = std.Io;

const Case = struct {
    name: []const u8,
    batch: usize,
    d: usize,
    out: usize,
    iters: usize,
};

const Mode = enum {
    serial_vjp,
    parallel_vjp,
    engine_forward,
    engine_backward,
};

const Result = struct {
    mode: Mode,
    case: Case,
    ns_per_op: u64,
    allocs_per_op: usize,
    bytes_per_op: usize,
    live_bytes: usize,
    checksum: f64,
};

const cases = [_]Case{
    .{ .name = "b16_d128_o512", .batch = 16, .d = 128, .out = 512, .iters = 40 },
    .{ .name = "b32_d256_o1024", .batch = 32, .d = 256, .out = 1024, .iters = 16 },
    .{ .name = "b64_d512_o1024", .batch = 64, .d = 512, .out = 1024, .iters = 10 },
    .{ .name = "b64_d512_o2048", .batch = 64, .d = 512, .out = 2048, .iters = 8 },
    .{ .name = "b128_d512_o2048", .batch = 128, .d = 512, .out = 2048, .iters = 6 },
    .{ .name = "b128_d768_o2304", .batch = 128, .d = 768, .out = 2304, .iters = 9 },
    .{ .name = "b256_d768_o2304", .batch = 256, .d = 768, .out = 2304, .iters = 7 },
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const allocator_mode = try bench_alloc.parseAllocatorMode(args);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.writeAll("runtime,backend,mode,case,batch,d,out,iters,ns_per_op,speedup_vs_serial_vjp_x,allocs_per_op,bytes_per_op,live_bytes,checksum\n");
    for (cases) |case| {
        const serial = try runCase(init.io, allocator_mode, case, .serial_vjp);
        const parallel = try runCase(init.io, allocator_mode, case, .parallel_vjp);
        const engine_forward = try runCase(init.io, allocator_mode, case, .engine_forward);
        const engine = try runCase(init.io, allocator_mode, case, .engine_backward);
        const speedup = @as(f64, @floatFromInt(serial.ns_per_op)) / @as(f64, @floatFromInt(parallel.ns_per_op));
        const engine_forward_speedup = @as(f64, @floatFromInt(serial.ns_per_op)) / @as(f64, @floatFromInt(engine_forward.ns_per_op));
        const engine_speedup = @as(f64, @floatFromInt(serial.ns_per_op)) / @as(f64, @floatFromInt(engine.ns_per_op));
        try printResult(stdout, serial, 1);
        try printResult(stdout, parallel, speedup);
        try printResult(stdout, engine_forward, engine_forward_speedup);
        try printResult(stdout, engine, engine_speedup);
    }
}

fn printResult(writer: anytype, result: Result, speedup: f64) !void {
    try writer.print("zig,{s},{s},{s},{d},{d},{d},{d},{d},{d:.6},{d},{d},{d},{d:.6}\n", .{
        @tagName(fucina.active_backend_kind),
        @tagName(result.mode),
        result.case.name,
        result.case.batch,
        result.case.d,
        result.case.out,
        result.case.iters,
        result.ns_per_op,
        speedup,
        result.allocs_per_op,
        result.bytes_per_op,
        result.live_bytes,
        result.checksum,
    });
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

    var worker: OneShotWorker = undefined;
    if (mode == .parallel_vjp) {
        try worker.init();
    }
    defer if (mode == .parallel_vjp) worker.deinit();

    for (0..2) |_| {
        const checksum = switch (mode) {
            .serial_vjp => try diamondSerial(&ctx, &base),
            .parallel_vjp => try diamondParallel(&ctx, &base, &worker),
            .engine_forward => try diamondEngineForward(&ctx, &base),
            .engine_backward => try diamondEngineBackward(&ctx, &base),
        };
        std.mem.doNotOptimizeAway(checksum);
    }

    counted.resetWindow();
    var checksum: f64 = 0;
    const allocator = counted.allocator();
    const times = try allocator.alloc(u64, case.iters);
    defer allocator.free(times);
    var timer = try Timer.start(io);
    for (times) |*time| {
        timer.reset();
        const value = switch (mode) {
            .serial_vjp => try diamondSerial(&ctx, &base),
            .parallel_vjp => try diamondParallel(&ctx, &base, &worker),
            .engine_forward => try diamondEngineForward(&ctx, &base),
            .engine_backward => try diamondEngineBackward(&ctx, &base),
        };
        time.* = timer.read();
        checksum += value;
    }
    std.mem.sort(u64, times, {}, std.sort.asc(u64));

    return .{
        .mode = mode,
        .case = case,
        .ns_per_op = times[times.len / 2],
        .allocs_per_op = counted.alloc_count / case.iters,
        .bytes_per_op = counted.bytes_allocated / case.iters,
        .live_bytes = counted.peak_live,
        .checksum = checksum,
    };
}

const BranchOutputs = struct {
    dx: Tensor,
    dw: Tensor,

    fn deinit(self: *BranchOutputs) void {
        self.dw.deinit();
        self.dx.deinit();
        self.* = undefined;
    }
};

fn runBranch(ctx: *ExecContext, x: *const Tensor, w: *const Tensor, gy: *const Tensor) !BranchOutputs {
    var dx = try ctx.matmulTransB(gy, w);
    errdefer dx.deinit();
    var dw = try ctx.matmulTransA(x, gy);
    errdefer dw.deinit();
    return .{ .dx = dx, .dw = dw };
}

const BranchTask = struct {
    ctx: *ExecContext,
    x: *const Tensor,
    w: *const Tensor,
    gy: *const Tensor,
    out: ?BranchOutputs = null,
    err: ?anyerror = null,
};

fn runBranchTask(ptr: *anyopaque) void {
    const task: *BranchTask = @ptrCast(@alignCast(ptr));
    task.out = runBranch(task.ctx, task.x, task.w, task.gy) catch |err| {
        task.err = err;
        return;
    };
}

fn diamondSerial(ctx: *ExecContext, base: *const Base) !f64 {
    var left = try runBranch(ctx, &base.x, &base.w1, &base.gy1);
    defer left.deinit();
    var right = try runBranch(ctx, &base.x, &base.w2, &base.gy2);
    defer right.deinit();

    try left.dx.addInPlace(&right.dx);
    return checksumOutputs(&left, &right);
}

fn diamondParallel(ctx: *ExecContext, base: *const Base, worker: *OneShotWorker) !f64 {
    var right_task = BranchTask{
        .ctx = ctx,
        .x = &base.x,
        .w = &base.w2,
        .gy = &base.gy2,
    };
    if (!worker.start(runBranchTask, &right_task)) {
        return diamondSerial(ctx, base);
    }

    var left = runBranch(ctx, &base.x, &base.w1, &base.gy1) catch |err| {
        worker.wait();
        if (right_task.out) |*right| right.deinit();
        return err;
    };
    defer left.deinit();

    worker.wait();
    if (right_task.err) |err| return err;
    var right = right_task.out.?;
    defer right.deinit();

    try left.dx.addInPlace(&right.dx);
    return checksumOutputs(&left, &right);
}

fn diamondEngineForward(ctx: *ExecContext, base: *const Base) !f64 {
    var x = try fucina.Tensor(.{ .batch, .d }).variable(ctx, try base.x.cloneView());
    defer x.deinit();
    var w1 = try fucina.Tensor(.{ .d, .out }).variable(ctx, try base.w1.cloneView());
    defer w1.deinit();
    var w2 = try fucina.Tensor(.{ .d, .out }).variable(ctx, try base.w2.cloneView());
    defer w2.deinit();

    var y1 = try x.dot(ctx, &w1, .d);
    defer y1.deinit();
    var y2 = try x.dot(ctx, &w2, .d);
    defer y2.deinit();
    var y = try y1.add(ctx, &y2);
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();

    return @as(f64, @floatCast(loss.asRawTensor().item()));
}

fn diamondEngineBackward(ctx: *ExecContext, base: *const Base) !f64 {
    var x = try fucina.Tensor(.{ .batch, .d }).variable(ctx, try base.x.cloneView());
    defer x.deinit();
    var w1 = try fucina.Tensor(.{ .d, .out }).variable(ctx, try base.w1.cloneView());
    defer w1.deinit();
    var w2 = try fucina.Tensor(.{ .d, .out }).variable(ctx, try base.w2.cloneView());
    defer w2.deinit();

    var y1 = try x.dot(ctx, &w1, .d);
    defer y1.deinit();
    var y2 = try x.dot(ctx, &w2, .d);
    defer y2.deinit();
    var y = try y1.add(ctx, &y2);
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();

    try loss.backward(ctx);

    var gx = (try x.gradView(ctx)).?;
    defer gx.deinit();
    var gw1 = (try w1.gradView(ctx)).?;
    defer gw1.deinit();
    var gw2 = (try w2.gradView(ctx)).?;
    defer gw2.deinit();

    return @as(f64, @floatCast(loss.asRawTensor().item())) +
        @as(f64, @floatCast(firstRawValue(gx.asRawTensor()))) +
        @as(f64, @floatCast(firstRawValue(gw1.asRawTensor()))) +
        @as(f64, @floatCast(firstRawValue(gw2.asRawTensor())));
}

fn firstRawValue(t: *const Tensor) f32 {
    return t.buffer.data[t.offset];
}

fn checksumOutputs(left: *const BranchOutputs, right: *const BranchOutputs) f64 {
    return @as(f64, @floatCast(left.dx.dataConst()[0])) +
        @as(f64, @floatCast(left.dw.dataConst()[0])) +
        @as(f64, @floatCast(right.dx.dataConst()[0])) +
        @as(f64, @floatCast(right.dw.dataConst()[0]));
}

const Base = struct {
    case: Case,
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    gy1: Tensor,
    gy2: Tensor,

    fn init(ctx: *ExecContext, case: Case) !Base {
        var x = try ctx.emptyRank(2, .{ case.batch, case.d });
        errdefer x.deinit();
        fillPattern(&x, 1);

        var w1 = try ctx.emptyRank(2, .{ case.d, case.out });
        errdefer w1.deinit();
        fillPattern(&w1, 2);

        var w2 = try ctx.emptyRank(2, .{ case.d, case.out });
        errdefer w2.deinit();
        fillPattern(&w2, 3);

        var gy1 = try ctx.emptyRank(2, .{ case.batch, case.out });
        errdefer gy1.deinit();
        fillPattern(&gy1, 4);

        var gy2 = try ctx.emptyRank(2, .{ case.batch, case.out });
        errdefer gy2.deinit();
        fillPattern(&gy2, 5);

        return .{
            .case = case,
            .x = x,
            .w1 = w1,
            .w2 = w2,
            .gy1 = gy1,
            .gy2 = gy2,
        };
    }

    fn deinit(self: *Base) void {
        self.gy2.deinit();
        self.gy1.deinit();
        self.w2.deinit();
        self.w1.deinit();
        self.x.deinit();
        self.* = undefined;
    }
};

fn fillPattern(t: *Tensor, seed: usize) void {
    for (t.data(), 0..) |*value, i| {
        const mixed = (i * 17 + seed * 31) % 97;
        const centered: i32 = @as(i32, @intCast(mixed)) - 48;
        value.* = @as(f32, @floatFromInt(centered)) * 0.0025;
    }
}

const OneShotWorker = struct {
    threaded: Io.Threaded = undefined,
    io: Io = undefined,
    thread: std.Thread = undefined,
    state: std.atomic.Value(State) = .init(.idle),
    job_fn: ?JobFn = null,
    job_arg: ?*anyopaque = null,
    ready: bool = false,

    const State = enum(u32) { idle, reserved, job, done, stop };
    const JobFn = *const fn (*anyopaque) void;

    fn init(self: *OneShotWorker) !void {
        self.* = .{
            .threaded = Io.Threaded.init(.failing, .{
                .async_limit = .nothing,
                .concurrent_limit = .nothing,
            }),
        };
        self.io = self.threaded.io();
        self.thread = std.Thread.spawn(.{}, workerMain, .{self}) catch |err| {
            self.threaded.deinit();
            self.* = undefined;
            return err;
        };
        self.ready = true;
    }

    fn deinit(self: *OneShotWorker) void {
        if (!self.ready) return;
        self.state.store(.stop, .release);
        self.io.futexWake(State, &self.state.raw, 1);
        self.thread.join();
        self.threaded.deinit();
        self.* = undefined;
    }

    fn start(self: *OneShotWorker, job_fn: JobFn, job_arg: *anyopaque) bool {
        if (self.state.cmpxchgStrong(.idle, .reserved, .acquire, .monotonic) != null) return false;
        self.job_fn = job_fn;
        self.job_arg = job_arg;
        self.state.store(.job, .release);
        self.io.futexWake(State, &self.state.raw, 1);
        return true;
    }

    fn wait(self: *OneShotWorker) void {
        while (true) {
            const state = self.state.load(.acquire);
            switch (state) {
                .done => {
                    self.state.store(.idle, .release);
                    self.io.futexWake(State, &self.state.raw, 1);
                    return;
                },
                .reserved, .job => self.io.futexWaitUncancelable(State, &self.state.raw, state),
                .idle, .stop => unreachable,
            }
        }
    }

    fn workerMain(self: *OneShotWorker) void {
        while (true) {
            switch (self.state.load(.acquire)) {
                .idle, .reserved, .done => |state| {
                    self.io.futexWaitUncancelable(State, &self.state.raw, state);
                    continue;
                },
                .stop => return,
                .job => {},
            }
            const job_fn = self.job_fn.?;
            const job_arg = self.job_arg.?;
            job_fn(job_arg);
            if (self.state.cmpxchgStrong(.job, .done, .release, .acquire)) |state| {
                std.debug.assert(state == .stop);
                return;
            }
            self.io.futexWake(State, &self.state.raw, 1);
        }
    }
};
