//! Two-spirals classification trained FROM SCRATCH by evolution strategies —
//! the gradient-free counterpart of examples/spirals.zig, and the
//! from-random-init acceptance test of `fucina.es`: fine-tuning starts at a
//! good solution, so only a from-scratch run proves the method optimizes
//! rather than merely perturbs. Same task, same two-hidden-layer tanh MLP,
//! same data generator as spirals.zig — only the learning signal differs:
//! no backward pass, no optimizer; reward = -CE on the full batch with
//! z-score shaping and mirrored (antithetic) sampling on the ES-at-scale
//! update. The defaults (sigma 0.1, alpha 0.1, population 128) reach 100%
//! accuracy within ~15k iterations (~75 s ReleaseFast on M1 Max). z-score
//! is the default shaping on purpose: with a bounded well-behaved reward
//! its magnitude information converges where `--norm centered_ranks`
//! stalls — pick the shaping by reward regime (docs/TRAINING.md section
//! 13).
//!
//! This is also the member-parallel showcase: small models are ES's
//! embarrassingly-parallel regime, so the population is evaluated with
//! `evaluateMembers` — each worker thread owns a full MLP replica and its
//! own ExecContext, `materializeMember` writes theta + sigma*eps into the
//! replica without touching the shared parameters, and only the scalar
//! rewards come back. The shared theta is touched by `update` alone.
//!
//! Self-verifying: the run FAILS (nonzero exit) unless the ES-trained
//! network reaches `--target` accuracy (default 0.90) on the training set —
//! chance is 0.50, so passing demonstrates genuine from-scratch learning.
//! Run with `zig build es-spirals -Doptimize=ReleaseFast`.
const std = @import("std");
const fucina = @import("fucina");

const es = fucina.es;
const rng = fucina.rng;
const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;

const hidden = 64; // spirals.zig's width — apples to apples
const n_per_class = 97; // the classic Lang & Witbrock size
const n_points = 2 * n_per_class;

/// spirals.zig's MLP, as CONSTANTS: ES needs no gradients, so nothing here
/// is a variable — the same struct doubles as the trainable canonical model
/// and as per-worker replicas.
const Model = struct {
    w1: Tensor(.{ .h1, .in }),
    b1: Tensor(.{.h1}),
    w2: Tensor(.{ .h2, .h1 }),
    b2: Tensor(.{.h2}),
    w3: Tensor(.{ .class, .h2 }),
    b3: Tensor(.{.class}),

    /// uniform(-1/sqrt(fan_in), +1/sqrt(fan_in)) weights, zero biases —
    /// spirals.zig's init through the repo RNG (deterministic per seed).
    fn initRandom(ctx: *ExecContext, seed: u64) !Model {
        var w1_buf: [hidden * 2]f32 = undefined;
        var b1_buf = [_]f32{0} ** hidden;
        var w2_buf: [hidden * hidden]f32 = undefined;
        var b2_buf = [_]f32{0} ** hidden;
        var w3_buf: [2 * hidden]f32 = undefined;
        var b3_buf = [_]f32{0} ** 2;
        const w1_bound = 1.0 / @sqrt(2.0);
        const wh_bound = 1.0 / @sqrt(@as(f32, hidden));
        rng.uniformFill(rng.at(seed, 0), &w1_buf, -w1_bound, w1_bound);
        rng.uniformFill(rng.at(seed, 1), &w2_buf, -wh_bound, wh_bound);
        rng.uniformFill(rng.at(seed, 2), &w3_buf, -wh_bound, wh_bound);
        return initFromBuffers(ctx, &w1_buf, &b1_buf, &w2_buf, &b2_buf, &w3_buf, &b3_buf);
    }

    fn initZero(ctx: *ExecContext) !Model {
        var w1_buf = [_]f32{0} ** (hidden * 2);
        var b1_buf = [_]f32{0} ** hidden;
        var w2_buf = [_]f32{0} ** (hidden * hidden);
        var b2_buf = [_]f32{0} ** hidden;
        var w3_buf = [_]f32{0} ** (2 * hidden);
        var b3_buf = [_]f32{0} ** 2;
        return initFromBuffers(ctx, &w1_buf, &b1_buf, &w2_buf, &b2_buf, &w3_buf, &b3_buf);
    }

    fn initFromBuffers(
        ctx: *ExecContext,
        w1_buf: []const f32,
        b1_buf: []const f32,
        w2_buf: []const f32,
        b2_buf: []const f32,
        w3_buf: []const f32,
        b3_buf: []const f32,
    ) !Model {
        var w1 = try Tensor(.{ .h1, .in }).fromSlice(ctx, .{ hidden, 2 }, w1_buf);
        errdefer w1.deinit();
        var b1 = try Tensor(.{.h1}).fromSlice(ctx, .{hidden}, b1_buf);
        errdefer b1.deinit();
        var w2 = try Tensor(.{ .h2, .h1 }).fromSlice(ctx, .{ hidden, hidden }, w2_buf);
        errdefer w2.deinit();
        var b2 = try Tensor(.{.h2}).fromSlice(ctx, .{hidden}, b2_buf);
        errdefer b2.deinit();
        var w3 = try Tensor(.{ .class, .h2 }).fromSlice(ctx, .{ 2, hidden }, w3_buf);
        errdefer w3.deinit();
        var b3 = try Tensor(.{.class}).fromSlice(ctx, .{2}, b3_buf);
        errdefer b3.deinit();
        return .{ .w1 = w1, .b1 = b1, .w2 = w2, .b2 = b2, .w3 = w3, .b3 = b3 };
    }

    fn deinit(self: *Model) void {
        self.b3.deinit();
        self.w3.deinit();
        self.b2.deinit();
        self.w2.deinit();
        self.b1.deinit();
        self.w1.deinit();
        self.* = undefined;
    }

    /// The parameter byte views, in the CANONICAL registration order shared
    /// by `register` and `materializeMember` destinations.
    fn byteViews(self: *Model) [6][]u8 {
        return .{
            std.mem.sliceAsBytes(self.w1.value.data()),
            std.mem.sliceAsBytes(self.b1.value.data()),
            std.mem.sliceAsBytes(self.w2.value.data()),
            std.mem.sliceAsBytes(self.b2.value.data()),
            std.mem.sliceAsBytes(self.w3.value.data()),
            std.mem.sliceAsBytes(self.b3.value.data()),
        };
    }

    fn register(self: *Model, trainer: *es.Trainer) !void {
        try trainer.addParamNamed(&self.w1, "w1");
        try trainer.addParamNamed(&self.b1, "b1");
        try trainer.addParamNamed(&self.w2, "w2");
        try trainer.addParamNamed(&self.b2, "b2");
        try trainer.addParamNamed(&self.w3, "w3");
        try trainer.addParamNamed(&self.b3, "b3");
    }
};

/// spirals.zig's forward, verbatim (constants, eval-only).
fn forwardLogits(ctx: *ExecContext, model: *const Model, x: *const Tensor(.{ .batch, .in })) !Tensor(.{ .batch, .class }) {
    const z1 = try x.dot(ctx, &model.w1, .in);
    const s1 = try z1.add(ctx, &model.b1);
    const a1 = try s1.tanh(ctx);
    const z2 = try a1.dot(ctx, &model.w2, .h1);
    const s2 = try z2.add(ctx, &model.b2);
    const a2 = try s2.tanh(ctx);
    const z3 = try a2.dot(ctx, &model.w3, .h2);
    return try z3.add(ctx, &model.b3);
}

/// Mean CE over the full batch (scoped, forward only).
fn meanCe(ctx: *ExecContext, model: *const Model, x: *const Tensor(.{ .batch, .in }), labels: []const usize) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const logits = try forwardLogits(ctx, model, x);
    const loss = try logits.crossEntropy(ctx, .class, labels);
    return loss.item();
}

fn accuracy(ctx: *ExecContext, model: *const Model, x: *const Tensor(.{ .batch, .in }), labels: []const usize) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const logits = try forwardLogits(ctx, model, x);
    const pred = try logits.argmax(ctx, .class);
    const pred_data = try pred.dataConst();
    var correct: usize = 0;
    for (pred_data, labels) |p, label| {
        if (@as(usize, @intCast(p)) == label) correct += 1;
    }
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(labels.len));
}

/// spirals.zig's data generator, verbatim: two interleaved spirals over
/// ~1.75 turns; class 1 is class 0 rotated by pi.
fn makeSpirals(seed: u64, xs: *[n_points * 2]f32, labels: *[n_points]usize) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (0..n_per_class) |i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, n_per_class - 1);
        const theta = t * 3.5 * std.math.pi;
        const r = 0.15 + 0.85 * t;
        const x = r * @sin(theta);
        const y = r * @cos(theta);
        xs[4 * i + 0] = x + random.floatNorm(f32) * 0.02;
        xs[4 * i + 1] = y + random.floatNorm(f32) * 0.02;
        labels[2 * i] = 0;
        xs[4 * i + 2] = -x + random.floatNorm(f32) * 0.02;
        xs[4 * i + 3] = -y + random.floatNorm(f32) * 0.02;
        labels[2 * i + 1] = 1;
    }
}

/// One member-parallel worker: a full MLP replica plus its own ExecContext
/// and batch tensor — `materializeMember` fills the replica, the forward
/// runs entirely worker-local, only the scalar reward crosses threads.
const Worker = struct {
    ctx: ExecContext,
    model: Model,
    x: Tensor(.{ .batch, .in }),

    fn deinit(self: *Worker) void {
        self.x.deinit();
        self.model.deinit();
        self.ctx.deinit();
        self.* = undefined;
    }
};

const Evaluator = struct {
    trainer: *const es.Trainer,
    workers: []Worker,
    labels: []const usize,

    pub fn evalMember(self: *const Evaluator, worker: usize, member: usize) !f32 {
        const w = &self.workers[worker];
        var views = w.model.byteViews();
        try self.trainer.materializeMember(member, &views);
        return -(try meanCe(&w.ctx, &w.model, &w.x, self.labels));
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var iterations: usize = 15000;
    var population: usize = 128;
    var sigma: f32 = 0.1;
    var alpha: ?f32 = 0.1;
    var workers: usize = 4;
    var seed: u64 = 42;
    var target: f32 = 0.9;
    var norm: es.RewardNorm = .z_score;

    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        if (argValue(args, &arg_i, "--iterations")) |v| {
            iterations = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--population")) |v| {
            population = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--sigma")) |v| {
            sigma = try std.fmt.parseFloat(f32, v);
        } else if (argValue(args, &arg_i, "--alpha")) |v| {
            alpha = try std.fmt.parseFloat(f32, v);
        } else if (argValue(args, &arg_i, "--workers")) |v| {
            workers = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--norm")) |v| {
            norm = std.meta.stringToEnum(es.RewardNorm, v) orelse return error.InvalidRewardNorm;
        } else if (argValue(args, &arg_i, "--seed")) |v| {
            seed = try std.fmt.parseInt(u64, v, 10);
        } else if (argValue(args, &arg_i, "--target")) |v| {
            target = try std.fmt.parseFloat(f32, v);
        } else {
            try stdout.print(
                "usage: zig build es-spirals -Doptimize=ReleaseFast -- [--iterations N] [--population N] " ++
                    "[--sigma F] [--alpha F] [--workers N] [--norm z_score|centered_ranks] [--seed N] [--target F]\n",
                .{},
            );
            return error.UnknownArgument;
        }
    }
    if (workers == 0) return error.InvalidWorkers;

    const allocator = std.heap.smp_allocator;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var xs: [n_points * 2]f32 = undefined;
    var labels: [n_points]usize = undefined;
    makeSpirals(seed, &xs, &labels);

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ n_points, 2 }, &xs);
    defer x.deinit();

    // The canonical model: random init (trained from scratch).
    var model = try Model.initRandom(&ctx, seed);
    defer model.deinit();

    var trainer = try es.Trainer.init(allocator, .{
        .sigma = sigma,
        .alpha = alpha,
        .population = population,
        .antithetic = population % 2 == 0, // mirrored sampling (Salimans)
        .reward_norm = norm, // default z_score (see the header note)
        .seed = seed,
    });
    defer trainer.deinit();
    try model.register(&trainer);

    // Per-worker replicas: zero-init contents (fully overwritten by
    // materializeMember before every evaluation).
    const worker_pool = try allocator.alloc(Worker, workers);
    defer allocator.free(worker_pool);
    var built: usize = 0;
    defer for (worker_pool[0..built]) |*w| w.deinit();
    for (worker_pool) |*w| {
        w.ctx.init(allocator);
        errdefer w.ctx.deinit();
        w.model = try Model.initZero(&w.ctx);
        errdefer w.model.deinit();
        w.x = try Tensor(.{ .batch, .in }).fromSlice(&w.ctx, .{ n_points, 2 }, &xs);
        built += 1;
    }

    const evaluator = Evaluator{ .trainer = &trainer, .workers = worker_pool, .labels = &labels };
    const rewards = try allocator.alloc(f32, population);
    defer allocator.free(rewards);

    const acc_before = try accuracy(&ctx, &model, &x, &labels);
    const ce_before = try meanCe(&ctx, &model, &x, &labels);
    try stdout.print(
        "es-spirals: {d} params from scratch  population {d} (antithetic)  sigma {d}  alpha {d}  " ++
            "centered ranks  {d} workers x replica\n",
        .{ trainer.elementCount(), population, sigma, trainer.alphaValue(), workers },
    );
    try stdout.print("before: accuracy {d:.3} (chance 0.500)  ce {d:.4}\n", .{ acc_before, ce_before });
    try stdout.flush();

    const start = nowNs(io);
    for (0..iterations) |iter_i| {
        try trainer.evaluateMembers(&evaluator, rewards, workers);
        _ = try trainer.update(&ctx, rewards);
        if ((iter_i + 1) % 500 == 0) {
            const acc = try accuracy(&ctx, &model, &x, &labels);
            const ce = try meanCe(&ctx, &model, &x, &labels);
            try stdout.print("iter {d:>5}  accuracy {d:.3}  ce {d:.4}\n", .{ iter_i + 1, acc, ce });
            try stdout.flush();
        }
    }
    const elapsed = seconds(nowNs(io) - start);

    const acc_after = try accuracy(&ctx, &model, &x, &labels);
    const ce_after = try meanCe(&ctx, &model, &x, &labels);
    try stdout.print("after {d} iterations ({d:.1} s, {d:.2} ms/iter): accuracy {d:.3}  ce {d:.4}\n", .{
        iterations, elapsed, 1000.0 * elapsed / @as(f64, @floatFromInt(iterations)), acc_after, ce_after,
    });

    if (acc_after < target) {
        try stdout.print("FAIL: from-scratch ES accuracy {d:.3} below target {d:.3}\n", .{ acc_after, target });
        try stdout.flush();
        return error.FromScratchTrainingFailed;
    }
    try stdout.print("PASS: gradient-free from-scratch training reached {d:.1}% (target {d:.1}%)\n", .{
        100 * acc_after, 100 * target,
    });
}

fn argValue(args: []const []const u8, arg_i: *usize, comptime flag: []const u8) ?[]const u8 {
    const arg = args[arg_i.*];
    if (std.mem.startsWith(u8, arg, flag ++ "=")) return arg[flag.len + 1 ..];
    if (std.mem.eql(u8, arg, flag)) {
        if (arg_i.* + 1 >= args.len) return null;
        arg_i.* += 1;
        return args[arg_i.*];
    }
    return null;
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
}
