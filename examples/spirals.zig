//! Two-spirals classification: end-to-end training, checkpointing, and
//! inference with each optimizer (SGD, AdamW, Muon, APOLLO, APOLLO-Mini),
//! plus a param-groups + lr-schedule + gradient-clipping combo.
//!
//! For every optimizer the demo:
//!   1. trains a two-hidden-layer tanh MLP on the classic two-spirals task,
//!      checkpointing model + optimizer state at the halfway step;
//!   2. proves exact training resumption: a fresh model + optimizer restored
//!      from the checkpoint retrains the second half and must reproduce the
//!      original final parameters bit-for-bit;
//!   3. reloads the final weights into a gradient-free model (inference path)
//!      and reports its accuracy.
//!
//! Run with `zig build spirals` (add -Doptimize=ReleaseFast for speed).
const std = @import("std");
const fucina = @import("fucina");
const optim = fucina.optim;
const training_checkpoint = fucina.training_checkpoint;

const Tensor = fucina.Tensor;
const ExecContext = fucina.ExecContext;

const hidden = 64;
const n_per_class = 200;
const n_points = 2 * n_per_class;
const train_steps = 2000;
const ckpt_step = train_steps / 2;

const Model = struct {
    w1: Tensor(.{ .h1, .in }),
    b1: Tensor(.{.h1}),
    w2: Tensor(.{ .h2, .h1 }),
    b2: Tensor(.{.h2}),
    w3: Tensor(.{ .class, .h2 }),
    b3: Tensor(.{.class}),

    /// Trainable model: uniform(-1/sqrt(fan_in), +1/sqrt(fan_in)) weights.
    fn initRandom(ctx: *ExecContext, seed: u64) !Model {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        var w1_buf: [hidden * 2]f32 = undefined;
        var b1_buf: [hidden]f32 = undefined;
        var w2_buf: [hidden * hidden]f32 = undefined;
        var b2_buf: [hidden]f32 = undefined;
        var w3_buf: [2 * hidden]f32 = undefined;
        var b3_buf: [2]f32 = undefined;
        fillUniform(random, &w1_buf, 1.0 / @sqrt(2.0));
        fillUniform(random, &w2_buf, 1.0 / @sqrt(@as(f32, hidden)));
        fillUniform(random, &w3_buf, 1.0 / @sqrt(@as(f32, hidden)));
        @memset(&b1_buf, 0);
        @memset(&b2_buf, 0);
        @memset(&b3_buf, 0);

        var w1 = try Tensor(.{ .h1, .in }).variableFromSlice(ctx, .{ hidden, 2 }, &w1_buf);
        errdefer w1.deinit();
        var b1 = try Tensor(.{.h1}).variableFromSlice(ctx, .{hidden}, &b1_buf);
        errdefer b1.deinit();
        var w2 = try Tensor(.{ .h2, .h1 }).variableFromSlice(ctx, .{ hidden, hidden }, &w2_buf);
        errdefer w2.deinit();
        var b2 = try Tensor(.{.h2}).variableFromSlice(ctx, .{hidden}, &b2_buf);
        errdefer b2.deinit();
        var w3 = try Tensor(.{ .class, .h2 }).variableFromSlice(ctx, .{ 2, hidden }, &w3_buf);
        errdefer w3.deinit();
        var b3 = try Tensor(.{.class}).variableFromSlice(ctx, .{2}, &b3_buf);
        errdefer b3.deinit();
        return .{ .w1 = w1, .b1 = b1, .w2 = w2, .b2 = b2, .w3 = w3, .b3 = b3 };
    }

    /// Gradient-free model (constants): the inference target for model.safetensors.
    fn initConstZero(ctx: *ExecContext) !Model {
        var w1_buf = [_]f32{0} ** (hidden * 2);
        var b1_buf = [_]f32{0} ** hidden;
        var w2_buf = [_]f32{0} ** (hidden * hidden);
        var b2_buf = [_]f32{0} ** hidden;
        var w3_buf = [_]f32{0} ** (2 * hidden);
        var b3_buf = [_]f32{0} ** 2;
        var w1 = try Tensor(.{ .h1, .in }).fromSlice(ctx, .{ hidden, 2 }, &w1_buf);
        errdefer w1.deinit();
        var b1 = try Tensor(.{.h1}).fromSlice(ctx, .{hidden}, &b1_buf);
        errdefer b1.deinit();
        var w2 = try Tensor(.{ .h2, .h1 }).fromSlice(ctx, .{ hidden, hidden }, &w2_buf);
        errdefer w2.deinit();
        var b2 = try Tensor(.{.h2}).fromSlice(ctx, .{hidden}, &b2_buf);
        errdefer b2.deinit();
        var w3 = try Tensor(.{ .class, .h2 }).fromSlice(ctx, .{ 2, hidden }, &w3_buf);
        errdefer w3.deinit();
        var b3 = try Tensor(.{.class}).fromSlice(ctx, .{2}, &b3_buf);
        errdefer b3.deinit();
        return .{ .w1 = w1, .b1 = b1, .w2 = w2, .b2 = b2, .w3 = w3, .b3 = b3 };
    }

    fn deinit(self: *Model) void {
        self.w1.deinit();
        self.b1.deinit();
        self.w2.deinit();
        self.b2.deinit();
        self.w3.deinit();
        self.b3.deinit();
    }
};

const ModelStateContext = struct {
    allocator: std.mem.Allocator,
    model: *Model,
};

fn fillUniform(random: std.Random, buf: []f32, bound: f32) void {
    for (buf) |*value| value.* = (random.float(f32) * 2 - 1) * bound;
}

/// Register all six params. Hidden matrices go to the matrix path of Muon /
/// APOLLO; the classifier head goes to their AdamW fallback (embedding lookups
/// and the final classifier must not be orthogonalized/rescaled — w1 is an
/// ordinary Linear over raw coordinates, so it stays on the matrix path);
/// biases auto-route to the fallback.
fn registerParams(opt: anytype, model: *Model) !void {
    try opt.addParam(&model.w1);
    try opt.addParam(&model.w2);
    if (comptime @hasDecl(@TypeOf(opt.*), "addFallbackParam")) {
        try opt.addFallbackParam(&model.w3);
    } else {
        try opt.addParam(&model.w3);
    }
    try opt.addParam(&model.b1);
    try opt.addParam(&model.b2);
    try opt.addParam(&model.b3);
}

/// Forward pass inside an exec scope (ExecContext.openExecScope): every op result
/// is owned by the scope, so the eager autograd graph — whose nodes the
/// backward pass walks through raw pointers — stays alive until the scope
/// closes. No keeps, no defers: training forward code looks like inference.
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

fn trainStep(ctx: *ExecContext, model: *const Model, x: *const Tensor(.{ .batch, .in }), labels: []const usize, opt: anytype) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope); // releases the whole step's graph
    const logits = try forwardLogits(ctx, model, x);
    const loss = try logits.crossEntropy(ctx, .class, labels);
    try loss.backward(ctx);
    try opt.step(ctx);
    opt.zeroGrad();
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
        if (@as(usize, @intFromFloat(p)) == label) correct += 1;
    }
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(labels.len));
}

/// Two interleaved spirals (the classic Lang & Witbrock task): radius grows
/// with the angle over ~1.75 turns; class 1 is class 0 rotated by pi.
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

fn saveCheckpoint(allocator: std.mem.Allocator, io: std.Io, path: []const u8, model: *Model, opt: anytype, step: u64, seed: u64) !void {
    try training_checkpoint.beginSave(allocator, io, path);
    const model_path = try training_checkpoint.pathJoin(allocator, path, training_checkpoint.model_state_file);
    defer allocator.free(model_path);
    try training_checkpoint.writeFileAtomic(io, model_path, ModelStateContext{ .allocator = allocator, .model = model }, writeModelState);

    if (comptime @TypeOf(opt) != @TypeOf(null)) {
        const optimizer_path = try training_checkpoint.pathJoin(allocator, path, training_checkpoint.optimizer_state_file);
        defer allocator.free(optimizer_path);
        const SaveOptimizer = struct {
            fn write(o: @TypeOf(opt), writer: *std.Io.Writer) !void {
                try o.saveState(writer);
            }
        };
        try training_checkpoint.writeFileAtomic(io, optimizer_path, opt, SaveOptimizer.write);
    }
    try training_checkpoint.saveTrainerState(allocator, io, path, .{ .step = step, .seed = seed });
}

fn loadCheckpoint(allocator: std.mem.Allocator, io: std.Io, path: []const u8, model: *Model, opt: anytype) !training_checkpoint.TrainerState {
    const state = try training_checkpoint.loadTrainerState(allocator, io, path);
    const model_path = try training_checkpoint.pathJoin(allocator, path, training_checkpoint.model_state_file);
    defer allocator.free(model_path);
    {
        var file = try std.Io.Dir.cwd().openFile(io, model_path, .{});
        defer file.close(io);
        var buffer: [64 * 1024]u8 = undefined;
        var reader = file.reader(io, &buffer);
        try loadModelState(allocator, &reader.interface, model);
    }
    if (comptime @TypeOf(opt) != @TypeOf(null)) {
        const optimizer_path = try training_checkpoint.pathJoin(allocator, path, training_checkpoint.optimizer_state_file);
        defer allocator.free(optimizer_path);
        var file = try std.Io.Dir.cwd().openFile(io, optimizer_path, .{});
        defer file.close(io);
        var buffer: [64 * 1024]u8 = undefined;
        var reader = file.reader(io, &buffer);
        try opt.loadState(&reader.interface);
    }
    return state;
}

// Model checkpoint via the reflective parameter registry: `collect` walks the
// Model struct's tensor fields and names them by field name ("w1", "b1", ...),
// so save/load needs no hand-written tensor list. The flat Model yields flat
// names; a nested model would get dotted paths automatically.
fn writeModelState(context: ModelStateContext, writer: *std.Io.Writer) !void {
    var registry = fucina.ParamRegistry.init(context.allocator);
    defer registry.deinit();
    try registry.collect(context.model);
    try registry.saveStateDict(writer);
}

fn loadModelState(allocator: std.mem.Allocator, reader: *std.Io.Reader, model: *Model) !void {
    var registry = fucina.ParamRegistry.init(allocator);
    defer registry.deinit();
    try registry.collect(model); // registers variables AND the const inference model
    try registry.loadStateDict(reader, .{});
}

fn snapshotParams(allocator: std.mem.Allocator, model: *const Model) ![]f32 {
    var out: std.ArrayList(f32) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, try model.w1.dataConst());
    try out.appendSlice(allocator, try model.b1.dataConst());
    try out.appendSlice(allocator, try model.w2.dataConst());
    try out.appendSlice(allocator, try model.b2.dataConst());
    try out.appendSlice(allocator, try model.w3.dataConst());
    try out.appendSlice(allocator, try model.b3.dataConst());
    return out.toOwnedSlice(allocator);
}

fn demo(
    comptime Opt: type,
    name: []const u8,
    config: anytype,
    ctx: *ExecContext,
    io: std.Io,
    stdout: anytype,
    x: *const Tensor(.{ .batch, .in }),
    labels: []const usize,
) !void {
    const allocator = ctx.allocator;
    var path_buf: [128]u8 = undefined;
    var final_buf: [128]u8 = undefined;
    const ckpt_path = try std.fmt.bufPrint(&path_buf, "/tmp/fucina-spirals-{s}", .{name});
    const final_path = try std.fmt.bufPrint(&final_buf, "/tmp/fucina-spirals-{s}-final", .{name});

    // Phase 1: train, checkpointing model + optimizer state halfway through.
    var model = try Model.initRandom(ctx, 42);
    defer model.deinit();
    var opt = Opt.init(allocator, config);
    defer opt.deinit();
    try registerParams(&opt, &model);

    var loss: f32 = 0;
    for (0..train_steps) |step_i| {
        loss = try trainStep(ctx, &model, x, labels, &opt);
        if (step_i + 1 == ckpt_step) {
            try saveCheckpoint(allocator, io, ckpt_path, &model, &opt, ckpt_step, 42);
        }
    }
    const acc = try accuracy(ctx, &model, x, labels);
    try stdout.print("[{s}] trained {d} steps: loss {d:.4}  accuracy {d:.1}%\n", .{ name, train_steps, loss, acc * 100 });
    try saveCheckpoint(allocator, io, final_path, &model, null, train_steps, 42);
    const reference = try snapshotParams(allocator, &model);
    defer allocator.free(reference);

    // Phase 2: restore the halfway checkpoint into a FRESH model + optimizer,
    // retrain the second half, and demand bit-identical final parameters.
    var resumed = try Model.initRandom(ctx, 7); // different init: fully overwritten by the checkpoint
    defer resumed.deinit();
    var opt2 = Opt.init(allocator, config);
    defer opt2.deinit();
    try registerParams(&opt2, &resumed);
    _ = try loadCheckpoint(allocator, io, ckpt_path, &resumed, &opt2);
    for (ckpt_step..train_steps) |_| {
        _ = try trainStep(ctx, &resumed, x, labels, &opt2);
    }
    const replayed = try snapshotParams(allocator, &resumed);
    defer allocator.free(replayed);
    var max_diff: f32 = 0;
    for (reference, replayed) |a, b| max_diff = @max(max_diff, @abs(a - b));
    try stdout.print("[{s}] resume from step {d}: max |delta param| = {d} ({s})\n", .{
        name, ckpt_step, max_diff, if (max_diff == 0) "bit-exact" else "NOT bit-exact",
    });
    if (max_diff != 0) return error.ResumeNotBitExact;

    // Phase 3: inference — load the final weights into a gradient-free model.
    var infer = try Model.initConstZero(ctx);
    defer infer.deinit();
    _ = try loadCheckpoint(allocator, io, final_path, &infer, null);
    const infer_acc = try accuracy(ctx, &infer, x, labels);
    try stdout.print("[{s}] inference from checkpoint: accuracy {d:.1}%\n\n", .{ name, infer_acc * 100 });

    try std.Io.Dir.cwd().deleteTree(io, ckpt_path);
    try std.Io.Dir.cwd().deleteTree(io, final_path);
}

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer if (gpa.deinit() == .leak) @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var xs: [n_points * 2]f32 = undefined;
    var labels: [n_points]usize = undefined;
    makeSpirals(1234, &xs, &labels);
    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ n_points, 2 }, &xs);
    defer x.deinit();

    try stdout.print("two spirals: {d} points, MLP 2-{d}-{d}-2 (tanh), full-batch, {d} steps\n\n", .{ n_points, hidden, hidden, train_steps });

    try demo(optim.SGD, "sgd", optim.SgdConfig{
        .lr = 0.1,
        .momentum = 0.9,
        .nesterov = true,
    }, &ctx, init.io, stdout, &x, &labels);

    try demo(optim.AdamW, "adamw", optim.AdamWConfig{
        .lr = 0.02,
        .weight_decay = 1e-4,
    }, &ctx, init.io, stdout, &x, &labels);

    try demo(optim.Muon, "muon", optim.MuonConfig{
        .lr = 0.02,
        .weight_decay = 0.01,
        // The reference fallback lr (3e-4) is tuned for LLM heads; this tiny
        // demo wants the head and biases to keep up with the hidden layers.
        .fallback = .{ .lr = 0.02, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-10, .weight_decay = 0 },
    }, &ctx, init.io, stdout, &x, &labels);

    try demo(optim.Apollo, "apollo", optim.ApolloConfig{
        .lr = 0.05,
        .rank = 8,
        .update_proj_gap = 100,
    }, &ctx, init.io, stdout, &x, &labels);

    var mini = optim.ApolloConfig.mini();
    mini.lr = 0.005; // effective step carries the sqrt(128) Mini gradient scale
    mini.update_proj_gap = 100;
    try demo(optim.Apollo, "apollo-mini", mini, &ctx, init.io, stdout, &x, &labels);

    try groupsDemo(&ctx, init.io, stdout, &x, &labels);
}

/// Param groups + lr schedule + gradient clipping, composed: matrices in a
/// weight-decayed AdamW group, biases in a no-decay group, both under one
/// OptimizerSet with a warmup-cosine schedule and global-norm clipping —
/// the standard LLM training recipe in miniature. Also proves the composition
/// resumes bit-exactly: the schedule factor is a pure function of the step,
/// so re-applying it after a checkpoint load replays the same trajectory.
fn groupsDemo(
    ctx: *ExecContext,
    io: std.Io,
    stdout: anytype,
    x: *const Tensor(.{ .batch, .in }),
    labels: []const usize,
) !void {
    const allocator = ctx.allocator;
    const name = "adamw-groups";
    const ckpt_path = "/tmp/fucina-spirals-groups";

    const Run = struct {
        fn stepOnce(
            run_ctx: *ExecContext,
            model: *const Model,
            run_x: *const Tensor(.{ .batch, .in }),
            run_labels: []const usize,
            set: *optim.OptimizerSet,
            sched: *const optim.LrSchedule,
            step_i: u64,
        ) !f32 {
            sched.apply(optim.warmupCosineFactor(step_i, train_steps, 100, 0.05));
            const scope = run_ctx.openExecScope();
            defer run_ctx.closeExecScope(scope);
            const logits = try forwardLogits(run_ctx, model, run_x);
            const loss = try logits.crossEntropy(run_ctx, .class, run_labels);
            try loss.backward(run_ctx);
            _ = try set.clipGradNorm(run_ctx, 1.0); // after backward, before step
            try set.step(run_ctx);
            set.zeroGrad();
            return loss.item();
        }
    };

    // Phase 1: train with two groups, checkpoint at the halfway step.
    var model = try Model.initRandom(ctx, 42);
    defer model.deinit();
    var decay = optim.AdamW.init(allocator, .{ .lr = 0.02, .weight_decay = 0.01 });
    defer decay.deinit();
    var no_decay = optim.AdamW.init(allocator, .{ .lr = 0.02, .weight_decay = 0 });
    defer no_decay.deinit();
    try decay.addParam(&model.w1);
    try decay.addParam(&model.w2);
    try decay.addParam(&model.w3);
    try no_decay.addParam(&model.b1);
    try no_decay.addParam(&model.b2);
    try no_decay.addParam(&model.b3);
    var set = optim.OptimizerSet.init(allocator);
    defer set.deinit();
    try set.add(&decay);
    try set.add(&no_decay);
    var sched = optim.LrSchedule.init(allocator);
    defer sched.deinit();
    try sched.attach(&decay.config.lr);
    try sched.attach(&no_decay.config.lr);

    var loss: f32 = 0;
    for (0..train_steps) |step_i| {
        loss = try Run.stepOnce(ctx, &model, x, labels, &set, &sched, step_i);
        if (step_i + 1 == ckpt_step) {
            try saveCheckpoint(allocator, io, ckpt_path, &model, &set, ckpt_step, 42);
        }
    }
    const acc = try accuracy(ctx, &model, x, labels);
    try stdout.print("[{s}] trained {d} steps (cosine lr, clip 1.0): loss {d:.4}  accuracy {d:.1}%\n", .{ name, train_steps, loss, acc * 100 });
    const reference = try snapshotParams(allocator, &model);
    defer allocator.free(reference);

    // Phase 2: fresh model + fresh groups + fresh schedule, restored from the
    // checkpoint; replaying steps ckpt_step.. must be bit-exact.
    var resumed = try Model.initRandom(ctx, 7);
    defer resumed.deinit();
    var decay2 = optim.AdamW.init(allocator, .{ .lr = 0.02, .weight_decay = 0.01 });
    defer decay2.deinit();
    var no_decay2 = optim.AdamW.init(allocator, .{ .lr = 0.02, .weight_decay = 0 });
    defer no_decay2.deinit();
    try decay2.addParam(&resumed.w1);
    try decay2.addParam(&resumed.w2);
    try decay2.addParam(&resumed.w3);
    try no_decay2.addParam(&resumed.b1);
    try no_decay2.addParam(&resumed.b2);
    try no_decay2.addParam(&resumed.b3);
    var set2 = optim.OptimizerSet.init(allocator);
    defer set2.deinit();
    try set2.add(&decay2);
    try set2.add(&no_decay2);
    var sched2 = optim.LrSchedule.init(allocator);
    defer sched2.deinit();
    try sched2.attach(&decay2.config.lr);
    try sched2.attach(&no_decay2.config.lr);

    _ = try loadCheckpoint(allocator, io, ckpt_path, &resumed, &set2);
    for (ckpt_step..train_steps) |step_i| {
        _ = try Run.stepOnce(ctx, &resumed, x, labels, &set2, &sched2, step_i);
    }
    const replayed = try snapshotParams(allocator, &resumed);
    defer allocator.free(replayed);
    var max_diff: f32 = 0;
    for (reference, replayed) |a, b| max_diff = @max(max_diff, @abs(a - b));
    try stdout.print("[{s}] resume from step {d}: max |delta param| = {d} ({s})\n\n", .{
        name, ckpt_step, max_diff, if (max_diff == 0) "bit-exact" else "NOT bit-exact",
    });
    if (max_diff != 0) return error.ResumeNotBitExact;

    try std.Io.Dir.cwd().deleteTree(io, ckpt_path);
}
