const std = @import("std");
const bench_alloc = @import("alloc.zig");
const Timer = @import("timer.zig").Timer;
const bench_raw = @import("bench_raw");
const fucina = bench_raw;

const Tensor = bench_raw.RawTensor;
const ExecContext = bench_raw.ExecContext;

const Case = struct {
    name: []const u8,
    batch: usize,
    d: usize,
    hidden: usize,
    out: usize,
    infer_iters: usize,
    backward_iters: usize,
};

const BiasMode = enum {
    full_bias,
    broadcast_bias,
};

const Mode = enum {
    inference,
    backward,
};

const Result = struct {
    runtime: []const u8 = "zig",
    backend: []const u8,
    mode: Mode,
    bias_mode: BiasMode,
    case: Case,
    iters: usize,
    ns_per_op: u64,
    allocs_per_op: usize,
    bytes_per_op: usize,
    live_bytes: usize,
    checksum: f64,
};

const quick_cases = [_]Case{
    .{ .name = "b1_d128_h512_o128", .batch = 1, .d = 128, .hidden = 512, .out = 128, .infer_iters = 60, .backward_iters = 8 },
    .{ .name = "b16_d128_h512_o128", .batch = 16, .d = 128, .hidden = 512, .out = 128, .infer_iters = 20, .backward_iters = 4 },
};

const full_cases = [_]Case{
    .{ .name = "b1_d128_h512_o128", .batch = 1, .d = 128, .hidden = 512, .out = 128, .infer_iters = 200, .backward_iters = 24 },
    .{ .name = "b16_d128_h512_o128", .batch = 16, .d = 128, .hidden = 512, .out = 128, .infer_iters = 80, .backward_iters = 12 },
    .{ .name = "b32_d256_h1024_o256", .batch = 32, .d = 256, .hidden = 1024, .out = 256, .infer_iters = 16, .backward_iters = 3 },
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
    try stdout.writeAll("runtime,backend,mode,bias_mode,case,batch,d,hidden,out,iters,ns_per_op,allocs_per_op,bytes_per_op,live_bytes,checksum\n");

    for (cases) |case| {
        try printResult(stdout, try runCase(init.io, allocator_mode, case, .inference, .full_bias));
        try printResult(stdout, try runCase(init.io, allocator_mode, case, .backward, .full_bias));
        try printResult(stdout, try runCase(init.io, allocator_mode, case, .inference, .broadcast_bias));
        try printResult(stdout, try runCase(init.io, allocator_mode, case, .backward, .broadcast_bias));
    }
}

fn printResult(writer: anytype, result: Result) !void {
    try writer.print(
        "{s},{s},{s},{s},{s},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d:.6}\n",
        .{
            result.runtime,
            result.backend,
            @tagName(result.mode),
            @tagName(result.bias_mode),
            result.case.name,
            result.case.batch,
            result.case.d,
            result.case.hidden,
            result.case.out,
            result.iters,
            result.ns_per_op,
            result.allocs_per_op,
            result.bytes_per_op,
            result.live_bytes,
            result.checksum,
        },
    );
}

fn runCase(io: std.Io, allocator_mode: bench_alloc.AllocatorMode, case: Case, comptime mode: Mode, comptime bias_mode: BiasMode) !Result {
    var benchmark_allocator = bench_alloc.BenchmarkAllocator.init(allocator_mode);
    defer benchmark_allocator.deinit();

    var counted = bench_alloc.CountingAllocator.init(benchmark_allocator.allocator());
    var ctx: ExecContext = undefined;
    ctx.init(counted.allocator());
    defer ctx.deinit();

    var base = try Base.init(&ctx, case);
    defer base.deinit();

    const warmup_iters: usize = if (mode == .inference) 2 else 1;
    for (0..warmup_iters) |_| {
        switch (mode) {
            .inference => {
                var y = try mlpInference(&ctx, &base, bias_mode);
                y.deinit();
            },
            .backward => _ = try mlpBackward(&ctx, &base, bias_mode),
        }
    }

    counted.resetWindow();
    const iters = switch (mode) {
        .inference => case.infer_iters,
        .backward => case.backward_iters,
    };

    var checksum: f64 = 0;
    var timer = try Timer.start(io);
    for (0..iters) |_| {
        switch (mode) {
            .inference => {
                var y = try mlpInference(&ctx, &base, bias_mode);
                checksum += @as(f64, @floatCast(y.dataConst()[0]));
                y.deinit();
            },
            .backward => {
                checksum += try mlpBackward(&ctx, &base, bias_mode);
            },
        }
    }
    const elapsed = timer.read();

    return .{
        .backend = @tagName(fucina.active_backend_kind),
        .mode = mode,
        .bias_mode = bias_mode,
        .case = case,
        .iters = iters,
        .ns_per_op = elapsed / iters,
        .allocs_per_op = counted.alloc_count / iters,
        .bytes_per_op = counted.bytes_allocated / iters,
        .live_bytes = counted.peak_live,
        .checksum = checksum,
    };
}

fn mlpInference(ctx: *ExecContext, base: *const Base, comptime bias_mode: BiasMode) !Tensor {
    var h = try ctx.matmul2D(&base.x, &base.w1);
    errdefer h.deinit();

    switch (bias_mode) {
        .full_bias => try ctx.addInPlace(&h, &base.b1_full),
        .broadcast_bias => {
            var b1 = try ctx.broadcastToRank(2, &base.b1_vec, .{ base.case.batch, base.case.hidden });
            defer b1.deinit();
            try ctx.addInPlace(&h, &b1);
        },
    }

    try ctx.mulInPlace(&h, &base.gate);
    defer h.deinit();

    var y = try ctx.matmul2D(&h, &base.w2);
    errdefer y.deinit();
    switch (bias_mode) {
        .full_bias => try ctx.addInPlace(&y, &base.b2_full),
        .broadcast_bias => {
            var b2 = try ctx.broadcastToRank(2, &base.b2_vec, .{ base.case.batch, base.case.out });
            defer b2.deinit();
            try ctx.addInPlace(&y, &b2);
        },
    }
    return y;
}

fn mlpBackward(ctx: *ExecContext, base: *const Base, comptime bias_mode: BiasMode) !f64 {
    var x = try fucina.Tensor(.{ .batch, .d }).variable(ctx, try base.x.cloneView());
    defer x.deinit();
    var w1 = try fucina.Tensor(.{ .d, .hidden }).variable(ctx, try base.w1.cloneView());
    defer w1.deinit();
    var gate = try fucina.Tensor(.{ .batch, .hidden }).variable(ctx, try base.gate.cloneView());
    defer gate.deinit();
    var w2 = try fucina.Tensor(.{ .hidden, .out }).variable(ctx, try base.w2.cloneView());
    defer w2.deinit();

    var h0 = try x.dot(ctx, &w1, .d);
    defer h0.deinit();

    switch (bias_mode) {
        .full_bias => {
            var b1 = try fucina.Tensor(.{ .batch, .hidden }).variable(ctx, try base.b1_full.cloneView());
            defer b1.deinit();

            var h1 = try h0.add(ctx, &b1);
            defer h1.deinit();
            var h2 = try h1.mul(ctx, &gate);
            defer h2.deinit();
            var y0 = try h2.dot(ctx, &w2, .hidden);
            defer y0.deinit();

            var b2 = try fucina.Tensor(.{ .batch, .out }).variable(ctx, try base.b2_full.cloneView());
            defer b2.deinit();

            var y = try y0.add(ctx, &b2);
            defer y.deinit();
            var loss = try y.sumAll(ctx);
            defer loss.deinit();

            try loss.backward(ctx);
            return @as(f64, @floatCast(loss.asRawTensor().item()));
        },
        .broadcast_bias => {
            var b1_source = try fucina.Tensor(.{.hidden}).variable(ctx, try base.b1_vec.cloneView());
            defer b1_source.deinit();
            var b1 = try b1_source.broadcastTo(ctx, .{ .batch, .hidden }, .{ base.case.batch, base.case.hidden });
            defer b1.deinit();

            var h1 = try h0.add(ctx, &b1);
            defer h1.deinit();
            var h2 = try h1.mul(ctx, &gate);
            defer h2.deinit();
            var y0 = try h2.dot(ctx, &w2, .hidden);
            defer y0.deinit();

            var b2_source = try fucina.Tensor(.{.out}).variable(ctx, try base.b2_vec.cloneView());
            defer b2_source.deinit();
            var b2 = try b2_source.broadcastTo(ctx, .{ .batch, .out }, .{ base.case.batch, base.case.out });
            defer b2.deinit();

            var y = try y0.add(ctx, &b2);
            defer y.deinit();
            var loss = try y.sumAll(ctx);
            defer loss.deinit();

            try loss.backward(ctx);
            return @as(f64, @floatCast(loss.asRawTensor().item()));
        },
    }
}

const Base = struct {
    case: Case,
    x: Tensor,
    w1: Tensor,
    b1_full: Tensor,
    b1_vec: Tensor,
    gate: Tensor,
    w2: Tensor,
    b2_full: Tensor,
    b2_vec: Tensor,

    fn init(ctx: *ExecContext, case: Case) !Base {
        var x = try ctx.emptyRank(2, .{ case.batch, case.d });
        errdefer x.deinit();
        fillPattern(&x, 1);

        var w1 = try ctx.emptyRank(2, .{ case.d, case.hidden });
        errdefer w1.deinit();
        fillPattern(&w1, 2);

        var b1_full = try ctx.emptyRank(2, .{ case.batch, case.hidden });
        errdefer b1_full.deinit();

        var b1_vec = try ctx.emptyRank(1, .{case.hidden});
        errdefer b1_vec.deinit();
        fillPattern(&b1_vec, 3);
        fillRepeatedRows(&b1_full, &b1_vec);

        var gate = try ctx.emptyRank(2, .{ case.batch, case.hidden });
        errdefer gate.deinit();
        fillRepeatedRowsPattern(&gate, 6);

        var w2 = try ctx.emptyRank(2, .{ case.hidden, case.out });
        errdefer w2.deinit();
        fillPattern(&w2, 4);

        var b2_full = try ctx.emptyRank(2, .{ case.batch, case.out });
        errdefer b2_full.deinit();

        var b2_vec = try ctx.emptyRank(1, .{case.out});
        errdefer b2_vec.deinit();
        fillPattern(&b2_vec, 5);
        fillRepeatedRows(&b2_full, &b2_vec);

        return .{
            .case = case,
            .x = x,
            .w1 = w1,
            .b1_full = b1_full,
            .b1_vec = b1_vec,
            .gate = gate,
            .w2 = w2,
            .b2_full = b2_full,
            .b2_vec = b2_vec,
        };
    }

    fn deinit(self: *Base) void {
        self.b2_vec.deinit();
        self.b2_full.deinit();
        self.w2.deinit();
        self.gate.deinit();
        self.b1_vec.deinit();
        self.b1_full.deinit();
        self.w1.deinit();
        self.x.deinit();
    }
};

fn fillPattern(t: *Tensor, seed: usize) void {
    for (t.data(), 0..) |*value, i| {
        const mixed = (i * 17 + seed * 31) % 97;
        const centered: i32 = @as(i32, @intCast(mixed)) - 48;
        value.* = @as(f32, @floatFromInt(centered)) * 0.0025;
    }
}

fn fillRepeatedRows(matrix: *Tensor, row: *const Tensor) void {
    const row_data = row.dataConst();
    const matrix_data = matrix.data();
    var offset: usize = 0;
    while (offset < matrix_data.len) : (offset += row_data.len) {
        @memcpy(matrix_data[offset .. offset + row_data.len], row_data);
    }
}

fn fillRepeatedRowsPattern(matrix: *Tensor, seed: usize) void {
    const cols = matrix.shape.at(1);
    const matrix_data = matrix.data();
    for (0..cols) |i| {
        const mixed = (i * 17 + seed * 31) % 97;
        const centered: i32 = @as(i32, @intCast(mixed)) - 48;
        matrix_data[i] = @as(f32, @floatFromInt(centered)) * 0.0025;
    }
    var offset: usize = cols;
    while (offset < matrix_data.len) : (offset += cols) {
        @memcpy(matrix_data[offset .. offset + cols], matrix_data[0..cols]);
    }
}
