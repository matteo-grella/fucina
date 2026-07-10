//! Optimizer step-kernel benchmark at LLM scale (`zig build bench-optim`).
//!
//! Measures the per-step cost of each optimizer on Qwen3-0.6B-class shapes:
//! one transformer block's 2D weights (q/k/v/o + SwiGLU FFN, hidden=1024,
//! ffn=3072, ~15.7M params) plus two norm vectors on the fallback paths, and —
//! separately — the 155.6M-param embedding matrix on the elementwise
//! optimizers. Gradients are synthetic and re-attached each step via a
//! refcounted view, so the loop times exactly: grad fetch + state update +
//! parameter write (+ Newton-Schulz / projection GEMMs where the algorithm
//! has them).
//!
//! Elementwise optimizers are memory-bandwidth-bound; the table reports the
//! effective GB/s from the per-element traffic model (SGD 12 B/elem,
//! SGD+momentum 20 B/elem — 16 with a bf16 buffer, AdamW 28 B/elem — 24 with
//! bf16 m, 20 with bf16 m+v). Muon adds 5 Newton-Schulz iterations (3 GEMMs
//! each) per matrix; APOLLO adds one projection GEMM per matrix and runs its
//! moments on the compressed tensors.
//!
//! Run in ReleaseFast: `zig build bench-optim -Doptimize=ReleaseFast`.
//! `--embedding=false` skips the 2.5GiB embedding section.
const std = @import("std");
const bench_alloc = @import("alloc.zig");
const Timer = @import("timer.zig").Timer;
const bench_raw = @import("bench_raw");

const Tensor = bench_raw.Tensor;
const RawTensor = bench_raw.RawTensor;
const ExecContext = bench_raw.ExecContext;
const optim = bench_raw.optim;

const hidden = 1024;
const ffn = 3072;
const q_out = 2048; // 16 heads x 128
const vocab = 151936;
const layers = 28;

const warmup_iters = 5;
const timed_iters = 50;

const block_shapes = [_][2]usize{
    .{ q_out, hidden }, // wq
    .{ hidden, hidden }, // wk
    .{ hidden, hidden }, // wv
    .{ hidden, q_out }, // wo
    .{ ffn, hidden }, // gate
    .{ ffn, hidden }, // up
    .{ hidden, ffn }, // down
};

const BlockParams = struct {
    mats: [block_shapes.len]Tensor(.{ .out, .in }),
    norms: [2]Tensor(.{.d}),
    grads: [block_shapes.len + 2]RawTensor,
    count: usize,

    fn init(ctx: *ExecContext, allocator: std.mem.Allocator) !BlockParams {
        var prng = std.Random.DefaultPrng.init(0xfeed);
        const random = prng.random();
        var self: BlockParams = undefined;
        self.count = 0;
        var grad_i: usize = 0;
        inline for (block_shapes, 0..) |shape, i| {
            const data = try allocator.alloc(f32, shape[0] * shape[1]);
            defer allocator.free(data);
            for (data) |*value| value.* = random.floatNorm(f32) * 0.02;
            self.mats[i] = try Tensor(.{ .out, .in }).variableFromSlice(ctx, shape, data);
            for (data) |*value| value.* = random.floatNorm(f32) * 1e-3;
            self.grads[grad_i] = try ctx.fromSliceRank(2, shape, data);
            grad_i += 1;
            self.count += shape[0] * shape[1];
        }
        var norm_data: [hidden]f32 = undefined;
        for (0..2) |i| {
            for (&norm_data) |*value| value.* = 1 + random.floatNorm(f32) * 0.01;
            self.norms[i] = try Tensor(.{.d}).variableFromSlice(ctx, .{hidden}, &norm_data);
            for (&norm_data) |*value| value.* = random.floatNorm(f32) * 1e-3;
            self.grads[grad_i] = try ctx.fromSliceRank(1, .{hidden}, &norm_data);
            grad_i += 1;
            self.count += hidden;
        }
        return self;
    }

    fn deinit(self: *BlockParams) void {
        inline for (&self.mats) |*mat| mat.deinit();
        for (&self.norms) |*norm| norm.deinit();
        for (&self.grads) |*grad| grad.deinit();
    }

    fn register(self: *BlockParams, opt: anytype) !void {
        inline for (&self.mats) |*mat| try opt.addParam(mat);
        for (&self.norms) |*norm| try opt.addParam(norm);
    }

    /// Re-attach the synthetic gradients (refcounted views; no copies).
    fn setGrads(self: *BlockParams) !void {
        inline for (&self.mats, 0..) |*mat, i| {
            mat.grad_state.?.setGrad(try self.grads[i].cloneView());
        }
        for (&self.norms, 0..) |*norm, i| {
            norm.grad_state.?.setGrad(try self.grads[block_shapes.len + i].cloneView());
        }
    }
};

const Report = struct {
    name: []const u8,
    params: usize,
    ns_per_step: u64,
    bytes_per_elem: ?u64,
};

fn benchOptimizer(
    ctx: *ExecContext,
    io: std.Io,
    allocator: std.mem.Allocator,
    comptime name: []const u8,
    comptime Opt: type,
    config: anytype,
    bytes_per_elem: ?u64,
) !Report {
    // Params before the optimizer so the deinit order respects the
    // params-outlive-optimizer contract.
    var params = try BlockParams.init(ctx, allocator);
    defer params.deinit();
    var opt = Opt.init(allocator, config);
    defer opt.deinit();
    try params.register(&opt);

    for (0..warmup_iters) |_| {
        try params.setGrads();
        try opt.step(ctx);
    }
    var timer = try Timer.start(io);
    for (0..timed_iters) |_| {
        try params.setGrads();
        try opt.step(ctx);
    }
    const ns = timer.read() / timed_iters;
    return .{ .name = name, .params = params.count, .ns_per_step = ns, .bytes_per_elem = bytes_per_elem };
}

fn printReport(stdout: anytype, r: Report) !void {
    const ms = @as(f64, @floatFromInt(r.ns_per_step)) / 1e6;
    const model_ms = ms * layers;
    if (r.bytes_per_elem) |bpe| {
        const gbps = @as(f64, @floatFromInt(r.params * bpe)) / @as(f64, @floatFromInt(r.ns_per_step));
        try stdout.print("{s:<18} {d:>7.2} M {d:>9.3} ms {d:>8.1} GB/s {d:>9.1} ms\n", .{ r.name, @as(f64, @floatFromInt(r.params)) / 1e6, ms, gbps, model_ms });
    } else {
        try stdout.print("{s:<18} {d:>7.2} M {d:>9.3} ms {s:>13} {d:>9.1} ms\n", .{ r.name, @as(f64, @floatFromInt(r.params)) / 1e6, ms, "-", model_ms });
    }
}

/// Global-norm pass in isolation: clipGradNorm with max_norm = floatMax so
/// the scale never triggers — the row times exactly the gradient-norm
/// reduction over the block params (the deterministic chunked `sumSquares`).
fn benchClipNorm(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator) !Report {
    var params = try BlockParams.init(ctx, allocator);
    defer params.deinit();
    var opt = optim.SGD.init(allocator, .{ .lr = 1e-3 });
    defer opt.deinit();
    try params.register(&opt);
    try params.setGrads();

    for (0..warmup_iters) |_| {
        _ = try opt.clipGradNorm(ctx, std.math.floatMax(f32));
    }
    var timer = try Timer.start(io);
    for (0..timed_iters) |_| {
        _ = try opt.clipGradNorm(ctx, std.math.floatMax(f32));
    }
    const ns = timer.read() / timed_iters;
    return .{ .name = "clip-norm", .params = params.count, .ns_per_step = ns, .bytes_per_elem = 4 };
}

fn benchEmbedding(ctx: *ExecContext, io: std.Io, allocator: std.mem.Allocator, stdout: anytype) !void {
    var prng = std.Random.DefaultPrng.init(0xe33d);
    const random = prng.random();
    const data = try allocator.alloc(f32, vocab * hidden);
    defer allocator.free(data);
    for (data) |*value| value.* = random.floatNorm(f32) * 0.02;
    var embed = try Tensor(.{ .vocab, .d }).variableFromSlice(ctx, .{ vocab, hidden }, data);
    defer embed.deinit();
    for (data) |*value| value.* = random.floatNorm(f32) * 1e-3;
    var grad = try ctx.fromSliceRank(2, .{ vocab, hidden }, data);
    defer grad.deinit();

    inline for (.{
        .{ "sgd", optim.SgdConfig{ .lr = 1e-3 }, 12 },
        .{ "sgd-momentum", optim.SgdConfig{ .lr = 1e-3, .momentum = 0.9 }, 20 },
        .{ "sgd-momentum-bf16", optim.SgdConfig{ .lr = 1e-3, .momentum = 0.9, .state_dtype = .bf16 }, 16 },
        .{ "adamw", optim.AdamWConfig{}, 28 },
        .{ "adamw-bf16-m", optim.AdamWConfig{ .state_dtype = .bf16 }, 24 },
        .{ "adamw-bf16-mv", optim.AdamWConfig{ .state_dtype = .bf16, .second_moment_dtype = .bf16 }, 20 },
    }) |case| {
        var opt = if (@TypeOf(case[1]) == optim.SgdConfig)
            optim.SGD.init(allocator, case[1])
        else
            optim.AdamW.init(allocator, case[1]);
        defer opt.deinit();
        try opt.addParam(&embed);
        for (0..warmup_iters) |_| {
            embed.grad_state.?.setGrad(try grad.cloneView());
            try opt.step(ctx);
        }
        var timer = try Timer.start(io);
        for (0..timed_iters) |_| {
            embed.grad_state.?.setGrad(try grad.cloneView());
            try opt.step(ctx);
        }
        const ns = timer.read() / timed_iters;
        try printReport(stdout, .{ .name = case[0], .params = vocab * hidden, .ns_per_step = ns, .bytes_per_elem = case[2] });
    }

    // Norm-only row (see benchClipNorm): the reduction at embedding scale.
    var opt = optim.SGD.init(allocator, .{ .lr = 1e-3 });
    defer opt.deinit();
    try opt.addParam(&embed);
    embed.grad_state.?.setGrad(try grad.cloneView());
    for (0..warmup_iters) |_| {
        _ = try opt.clipGradNorm(ctx, std.math.floatMax(f32));
    }
    var timer = try Timer.start(io);
    for (0..timed_iters) |_| {
        _ = try opt.clipGradNorm(ctx, std.math.floatMax(f32));
    }
    const ns = timer.read() / timed_iters;
    try printReport(stdout, .{ .name = "clip-norm", .params = vocab * hidden, .ns_per_step = ns, .bytes_per_elem = 4 });
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var run_embedding = true;
    var mode: bench_alloc.AllocatorMode = .smp;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--embedding=false")) {
            run_embedding = false;
        } else if (try bench_alloc.parseAllocatorModeArg(arg)) |parsed| {
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

    try stdout.print("optimizer step benchmark — backend={s}, one Qwen3-0.6B-class block (hidden={d}, ffn={d}), {d} timed iters\n\n", .{ @tagName(bench_raw.active_backend_kind), hidden, ffn, timed_iters });
    try stdout.print("{s:<18} {s:>9} {s:>12} {s:>13} {s:>12}\n", .{ "optimizer", "params", "ms/step", "eff. GB/s", "x28 layers" });

    try printReport(stdout, try benchOptimizer(&ctx, init.io, allocator, "sgd", optim.SGD, optim.SgdConfig{ .lr = 1e-3 }, 12));
    try printReport(stdout, try benchOptimizer(&ctx, init.io, allocator, "sgd-momentum", optim.SGD, optim.SgdConfig{ .lr = 1e-3, .momentum = 0.9 }, 20));
    try printReport(stdout, try benchOptimizer(&ctx, init.io, allocator, "sgd-momentum-bf16", optim.SGD, optim.SgdConfig{ .lr = 1e-3, .momentum = 0.9, .state_dtype = .bf16 }, 16));
    try printReport(stdout, try benchOptimizer(&ctx, init.io, allocator, "adamw", optim.AdamW, optim.AdamWConfig{}, 28));
    try printReport(stdout, try benchOptimizer(&ctx, init.io, allocator, "adamw-bf16-m", optim.AdamW, optim.AdamWConfig{ .state_dtype = .bf16 }, 24));
    try printReport(stdout, try benchOptimizer(&ctx, init.io, allocator, "adamw-bf16-mv", optim.AdamW, optim.AdamWConfig{ .state_dtype = .bf16, .second_moment_dtype = .bf16 }, 20));
    try printReport(stdout, try benchOptimizer(&ctx, init.io, allocator, "muon", optim.Muon, optim.MuonConfig{}, null));
    try printReport(stdout, try benchOptimizer(&ctx, init.io, allocator, "muon-bf16", optim.Muon, optim.MuonConfig{ .state_dtype = .bf16, .fallback = .{ .lr = 3e-4, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-10, .weight_decay = 0, .state_dtype = .bf16 } }, null));
    try printReport(stdout, try benchOptimizer(&ctx, init.io, allocator, "apollo-r256", optim.Apollo, optim.ApolloConfig{ .rank = 256 }, null));
    try printReport(stdout, try benchOptimizer(&ctx, init.io, allocator, "apollo-mini", optim.Apollo, optim.ApolloConfig.mini(), null));
    try printReport(stdout, try benchClipNorm(&ctx, init.io, allocator));

    if (run_embedding) {
        try stdout.print("\nembedding [{d}, {d}] (155.6M params, elementwise paths):\n", .{ vocab, hidden });
        try benchEmbedding(&ctx, init.io, allocator, stdout);
    }
}
