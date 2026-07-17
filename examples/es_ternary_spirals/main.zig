//! Two-spirals classification trained FROM SCRATCH by the TERNARY-NATIVE
//! evolution strategy — es_spirals.zig's BitNet-class sibling and the
//! flagship demo of `es.Trainer`'s ternary slots: the hidden and output
//! layers are packed TQ2_0 genomes (2-bit {-1,0,+1} crumbs) registered with
//! `addTernaryParam`, so the TRAINING STATE IS THE INFERENCE MODEL — no
//! latent float weights exist for the ternary layers at any point, and every
//! forward pass (member evaluations AND the final verification) runs on the
//! real int8 flagship kernels: activations quantize to Q8_K rows and
//! multiply the packed blocks via `matmulTQ2_0RhsRange`.
//!
//! Architecture (TQ2_0 needs contract dims that are multiples of 256):
//! 2 -> [dense f32] -> 256 -> tanh -> [TERNARY 256x256] -> tanh ->
//! [TERNARY head 256x2] -> logits, with f32 biases throughout. The float
//! first layer embeds the 2-D input (k = 2 is far below the 256-element
//! block contract; BitNet practice keeps edge layers in higher precision
//! anyway); it trains through the ordinary gaussian ES slots while the two
//! genomes evolve by sparse trit flips + vote-and-threshold updates — one
//! shared reward pipeline, mirrored (antithetic) sampling on both kinds.
//!
//! Each genome's fp16 block scale `d` is FIXED at init and never trained:
//! uniform random trits t have E[t^2] = 2/3, so dequantized weights w = t*d
//! carry Var(w) = (2/3)*d^2; matching the variance-preserving init
//! Var(w) = 1/fan_in gives d = 1/sqrt(fan_in * 2/3) (~0.0765 at k = 256).
//! The default runs at 3x that matched scale (`--dscale`): d is the ternary
//! learning-rate analog — one trit flip moves d of function — and on this
//! landscape the variance-matched scale explores too slowly to escape the
//! ~0.64-CE plateau, while 3x dives to 100% (the tanh layers absorb the
//! larger pre-activations).
//!
//! Member-parallel like es_spirals: `evaluateMembers` fans the population
//! over worker threads, each owning replica buffers for BOTH slot kinds —
//! `materializeMember` writes the float replicas, `materializeTernaryMember`
//! copies + flips the packed replicas — and only scalar rewards come back.
//! Replica matmuls go through borrowed `QuantizedMatmulRhsTQ2_0` views
//! constructed ONCE over each replica's blocks (the copy is in place, the
//! view never reallocates).
//!
//! Reward: raw -CE with z-score shaping, exactly like es_spirals — measured
//! here too, magnitude information wins: the bounded composite
//! `accuracy + 0.1*exp(-CE)` (es_finetune's `acc`, kept as `--reward acc`)
//! climbs fast early but plateaus near 74%, while -CE grinds the same
//! ~0.64-CE plateau the float net shows for a few thousand iterations and
//! then dives to 100%.
//!
//! Self-verifying: FAILS (nonzero exit) unless the trained network reaches
//! `--target` accuracy (default 0.95) on the training set. Chance is 0.50;
//! es_spirals's float MLP reaches 1.00 — the discrete trit search is harder,
//! hence the 95% bar. Defaults cross 95% around iteration ~8-11k and stop
//! early at 100% (~2 minutes ReleaseFast on an M1-class CPU; seed-robust:
//! 42 and 7 both verified). Run with `zig build es-ternary-spirals
//! -Doptimize=ReleaseFast`.
const std = @import("std");
const fucina = @import("fucina");

const es = fucina.es;
const rng = fucina.rng;
const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;
const backend = fucina.internal.backend_mod;
const quant = backend.quantized_matmul;

const Allocator = std.mem.Allocator;
const BlockTQ2_0 = es.BlockTQ2_0;
const BlockQ8_K = backend.BlockQ8_K;
const Rhs = backend.QuantizedMatmulRhsTQ2_0;

const hidden = 256; // minimum TQ2_0 contract dim (one 256-crumb block per row)
const n_classes = 2;
const n_per_class = 97; // the classic Lang & Witbrock size (es_spirals.zig)
const n_points = 2 * n_per_class;

const block_len = 256; // logical elements per TQ2_0 / Q8_K block
const blocks_per_row = hidden / block_len;
const l2_len = hidden * hidden; // ternary hidden layer, row-major [out][in]
const l3_len = n_classes * hidden; // ternary head

/// Fixed dequantization scale of both genomes (see the header): uniform
/// trits give Var(w) = (2/3)*d^2, so d = 1/sqrt(fan_in * 2/3) matches the
/// 1/fan_in variance-preserving init. Both ternary layers have fan_in 256.
/// Set ONCE before model init (`--dscale` multiplies the matched default —
/// larger d moves more function per trit, the ternary learning-rate analog)
/// and fixed for the whole run: ES never touches block scales.
var ternary_d: f32 = 1.0 / @sqrt(@as(f32, hidden) * 2.0 / 3.0);

/// Borrowed matmul view over caller-owned packed blocks (row-major [n] rows
/// of k/256 blocks; view row c = weight row c = output column c). The genome
/// IS the inference weight: nothing is duplicated (allocator = null, so a
/// deinit would free nothing).
fn borrowRhs(blocks: []BlockTQ2_0, k: usize, n: usize) !Rhs {
    return quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, blocks);
}

/// Uniform random trits packed straight into TQ2_0 blocks: draw t in
/// {-1,0,+1} through the repo RNG, write t*d, and encode against the fixed
/// scale d (round(t*d/d) = t exactly, so the encode is lossless).
fn randomTrits(allocator: Allocator, seed: u64, len: usize) ![]BlockTQ2_0 {
    const vals = try allocator.alloc(f32, len);
    defer allocator.free(vals);
    rng.uniformFill(seed, vals, 0, 3);
    for (vals) |*v| v.* = (@floor(@min(v.*, 2.5)) - 1) * ternary_d;

    const blocks = try allocator.alloc(BlockTQ2_0, len / block_len);
    errdefer allocator.free(blocks);
    try quant.quantizeRowTQ2_0ScaledInto(blocks, vals, ternary_d);
    return blocks;
}

/// The canonical trainable model: float first layer + biases as facade
/// tensors (gaussian ES slots), the two ternary layers as OWNED packed
/// genomes (ternary ES slots) — the exact bytes the ternary vote-and-
/// threshold update moves and the verification forward multiplies.
const Model = struct {
    w1: Tensor(.{ .in, .h1 }), // [2][hidden]: input-major so L1 vectorizes over units
    b1: Tensor(.{.h1}),
    b2: Tensor(.{.h2}),
    b3: Tensor(.{.class}),
    l2: []BlockTQ2_0,
    l3: []BlockTQ2_0,
    l2_rhs: Rhs,
    l3_rhs: Rhs,

    /// Float layer: es_spirals's uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))
    /// weights + zero biases; genomes: uniform random trits at the fixed d.
    fn initRandom(ctx: *ExecContext, allocator: Allocator, seed: u64) !Model {
        var w1_buf: [hidden * 2]f32 = undefined;
        var b1_buf = [_]f32{0} ** hidden;
        var b2_buf = [_]f32{0} ** hidden;
        var b3_buf = [_]f32{0} ** n_classes;
        const w1_bound = 1.0 / @sqrt(2.0);
        rng.uniformFill(rng.at(seed, 0), &w1_buf, -w1_bound, w1_bound);

        var w1 = try Tensor(.{ .in, .h1 }).fromSlice(ctx, .{ 2, hidden }, &w1_buf);
        errdefer w1.deinit();
        var b1 = try Tensor(.{.h1}).fromSlice(ctx, .{hidden}, &b1_buf);
        errdefer b1.deinit();
        var b2 = try Tensor(.{.h2}).fromSlice(ctx, .{hidden}, &b2_buf);
        errdefer b2.deinit();
        var b3 = try Tensor(.{.class}).fromSlice(ctx, .{n_classes}, &b3_buf);
        errdefer b3.deinit();
        const l2 = try randomTrits(allocator, rng.at(seed, 1), l2_len);
        errdefer allocator.free(l2);
        const l3 = try randomTrits(allocator, rng.at(seed, 2), l3_len);
        errdefer allocator.free(l3);
        return .{
            .w1 = w1,
            .b1 = b1,
            .b2 = b2,
            .b3 = b3,
            .l2 = l2,
            .l3 = l3,
            .l2_rhs = try borrowRhs(l2, hidden, hidden),
            .l3_rhs = try borrowRhs(l3, hidden, n_classes),
        };
    }

    fn deinit(self: *Model, allocator: Allocator) void {
        allocator.free(self.l3);
        allocator.free(self.l2);
        self.b3.deinit();
        self.b2.deinit();
        self.b1.deinit();
        self.w1.deinit();
        self.* = undefined;
    }

    /// Registration order is the replica-materialization contract: float
    /// slots then ternary slots, each in this exact sequence.
    fn register(self: *Model, trainer: *es.Trainer) !void {
        try trainer.addParamNamed(&self.w1, "w1");
        try trainer.addParamNamed(&self.b1, "b1");
        try trainer.addParamNamed(&self.b2, "b2");
        try trainer.addParamNamed(&self.b3, "b3");
        try trainer.addTernaryParamNamed(self.l2, l2_len, "l2");
        try trainer.addTernaryParamNamed(self.l3, l3_len, "l3");
    }

    fn view(self: *const Model) NetView {
        return .{
            .w1 = self.w1.value.dataConst(),
            .b1 = self.b1.value.dataConst(),
            .b2 = self.b2.value.dataConst(),
            .b3 = self.b3.value.dataConst(),
            .l2 = &self.l2_rhs,
            .l3 = &self.l3_rhs,
        };
    }
};

/// One network's parameters as borrowed views — the canonical model and the
/// worker replicas evaluate through the same forward.
const NetView = struct {
    w1: []const f32,
    b1: []const f32,
    b2: []const f32,
    b3: []const f32,
    l2: *const Rhs,
    l3: *const Rhs,
};

/// Reused per-forward buffers (one set per worker + one for verification).
const Scratch = struct {
    act: []f32, // [n_points][hidden] current activations
    lin: []f32, // [n_points][hidden] ternary matmul outputs
    aq: []BlockQ8_K, // [n_points][blocks_per_row] quantized activation rows
    logits: []f32, // [n_points][n_classes]

    fn init(allocator: Allocator) !Scratch {
        const act = try allocator.alloc(f32, n_points * hidden);
        errdefer allocator.free(act);
        const lin = try allocator.alloc(f32, n_points * hidden);
        errdefer allocator.free(lin);
        const aq = try allocator.alloc(BlockQ8_K, n_points * blocks_per_row);
        errdefer allocator.free(aq);
        const logits = try allocator.alloc(f32, n_points * n_classes);
        return .{ .act = act, .lin = lin, .aq = aq, .logits = logits };
    }

    fn deinit(self: *Scratch, allocator: Allocator) void {
        allocator.free(self.logits);
        allocator.free(self.aq);
        allocator.free(self.lin);
        allocator.free(self.act);
        self.* = undefined;
    }
};

const EvalResult = struct { ce: f32, acc: f32 };

/// Vectorized tanh: the 7/6-degree Lambert-fraction Padé with a +-4.9 clamp
/// (|p/q| <= 1 inside the clamp; max abs error ~1e-4 at the boundary — far
/// below the Q8_K activation quantization step that follows every hidden
/// layer). libm's scalar tanh dominates the member-eval profile otherwise:
/// each forward evaluates ~100k activations against a ~13M-madd matmul.
const TanhVec = @Vector(8, f32);
fn tanhV(z: TanhVec) TanhVec {
    const limit: TanhVec = @splat(4.9);
    const x = @max(@min(z, limit), -limit);
    const x2 = x * x;
    const p = x * (@as(TanhVec, @splat(135135.0)) + x2 * (@as(TanhVec, @splat(17325.0)) + x2 * (@as(TanhVec, @splat(378.0)) + x2)));
    const q = @as(TanhVec, @splat(135135.0)) + x2 * (@as(TanhVec, @splat(62370.0)) + x2 * (@as(TanhVec, @splat(3150.0)) + x2 * @as(TanhVec, @splat(28.0))));
    return p / q;
}

/// act[i] = tanh(lin[i] + bias) over all n_points rows (hidden is a
/// multiple of the 8-lane vector width).
fn tanhRowsBias(act: []f32, lin: []const f32, bias: []const f32) void {
    for (0..n_points) |i| {
        const zrow = lin[i * hidden ..][0..hidden];
        const arow = act[i * hidden ..][0..hidden];
        var j: usize = 0;
        while (j < hidden) : (j += 8) {
            const z: TanhVec = zrow[j..][0..8].*;
            const b: TanhVec = bias[j..][0..8].*;
            arow[j..][0..8].* = tanhV(z + b);
        }
    }
}

fn quantizeActRows(dst: []BlockQ8_K, act: []const f32) !void {
    for (0..n_points) |i| {
        try quant.quantizeRowQ8_KInto(
            dst[i * blocks_per_row ..][0..blocks_per_row],
            act[i * hidden ..][0..hidden],
        );
    }
}

/// Full-batch forward + mean CE + accuracy. The ternary layers run the int8
/// flagship path end-to-end: Q8_K activation rows (absmax scale + bsums)
/// times the packed 2-bit genome, one batched matmul per layer.
fn evalNet(net: NetView, xs: *const [n_points * 2]f32, labels: *const [n_points]usize, s: *Scratch) !EvalResult {
    // L1 dense f32 [2 -> hidden]: k = 2, a plain float layer (see header).
    for (0..n_points) |i| {
        const x0: TanhVec = @splat(xs[2 * i]);
        const x1: TanhVec = @splat(xs[2 * i + 1]);
        const row = s.lin[i * hidden ..][0..hidden];
        var j: usize = 0;
        while (j < hidden) : (j += 8) {
            const w1a: TanhVec = net.w1[hidden * 0 + j ..][0..8].*;
            const w1b: TanhVec = net.w1[hidden * 1 + j ..][0..8].*;
            row[j..][0..8].* = x0 * w1a + x1 * w1b;
        }
    }
    tanhRowsBias(s.act, s.lin, net.b1);
    // L2 ternary [hidden -> hidden].
    try quantizeActRows(s.aq, s.act);
    quant.matmulTQ2_0RhsRange(s.lin, s.aq, net.l2, n_points, hidden, 0, n_points);
    tanhRowsBias(s.act, s.lin, net.b2);
    // L3 ternary head [hidden -> n_classes].
    try quantizeActRows(s.aq, s.act);
    quant.matmulTQ2_0RhsRange(s.logits, s.aq, net.l3, n_points, n_classes, 0, n_points);

    var ce: f64 = 0;
    var correct: usize = 0;
    for (0..n_points) |i| {
        const l0 = s.logits[n_classes * i] + net.b3[0];
        const l1 = s.logits[n_classes * i + 1] + net.b3[1];
        const m = @max(l0, l1);
        ce += m + @log(@exp(l0 - m) + @exp(l1 - m)) - (if (labels[i] == 0) l0 else l1);
        const pred: usize = if (l1 > l0) 1 else 0;
        if (pred == labels[i]) correct += 1;
    }
    return .{
        .ce = @floatCast(ce / n_points),
        .acc = @as(f32, @floatFromInt(correct)) / n_points,
    };
}

/// es_spirals.zig's data generator, verbatim: two interleaved spirals over
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

/// One member-parallel worker: replica buffers for the float slots AND the
/// ternary slots plus forward scratch. No ExecContext — the whole member
/// forward is raw kernels. Contents are fully overwritten by the two
/// materializers before every evaluation.
const Worker = struct {
    w1: []f32,
    b1: []f32,
    b2: []f32,
    b3: []f32,
    l2: []BlockTQ2_0,
    l3: []BlockTQ2_0,
    l2_rhs: Rhs,
    l3_rhs: Rhs,
    scratch: Scratch,

    fn init(allocator: Allocator) !Worker {
        const w1 = try allocator.alloc(f32, hidden * 2);
        errdefer allocator.free(w1);
        const b1 = try allocator.alloc(f32, hidden);
        errdefer allocator.free(b1);
        const b2 = try allocator.alloc(f32, hidden);
        errdefer allocator.free(b2);
        const b3 = try allocator.alloc(f32, n_classes);
        errdefer allocator.free(b3);
        const l2 = try allocator.alloc(BlockTQ2_0, l2_len / block_len);
        errdefer allocator.free(l2);
        const l3 = try allocator.alloc(BlockTQ2_0, l3_len / block_len);
        errdefer allocator.free(l3);
        var scratch = try Scratch.init(allocator);
        errdefer scratch.deinit(allocator);
        return .{
            .w1 = w1,
            .b1 = b1,
            .b2 = b2,
            .b3 = b3,
            .l2 = l2,
            .l3 = l3,
            .l2_rhs = try borrowRhs(l2, hidden, hidden),
            .l3_rhs = try borrowRhs(l3, hidden, n_classes),
            .scratch = scratch,
        };
    }

    fn deinit(self: *Worker, allocator: Allocator) void {
        self.scratch.deinit(allocator);
        allocator.free(self.l3);
        allocator.free(self.l2);
        allocator.free(self.b3);
        allocator.free(self.b2);
        allocator.free(self.b1);
        allocator.free(self.w1);
        self.* = undefined;
    }

    /// Float replica destinations, in the CANONICAL registration order.
    fn floatViews(self: *Worker) [4][]u8 {
        return .{
            std.mem.sliceAsBytes(self.w1),
            std.mem.sliceAsBytes(self.b1),
            std.mem.sliceAsBytes(self.b2),
            std.mem.sliceAsBytes(self.b3),
        };
    }

    fn view(self: *const Worker) NetView {
        return .{
            .w1 = self.w1,
            .b1 = self.b1,
            .b2 = self.b2,
            .b3 = self.b3,
            .l2 = &self.l2_rhs,
            .l3 = &self.l3_rhs,
        };
    }
};

/// Member reward.
const Reward = enum {
    /// Raw -CE (es_spirals's reward) — the default: its magnitude
    /// information drives the plateau-then-dive convergence (see header).
    nll,
    /// The bounded composite `accuracy + 0.1 * exp(-mean CE)` (es_finetune's
    /// `acc`; docs/TRAINING.md section 13). Kept for contrast: on this
    /// landscape it climbs early, then stalls near 74%.
    acc,
};

const Evaluator = struct {
    trainer: *const es.Trainer,
    workers: []Worker,
    xs: *const [n_points * 2]f32,
    labels: *const [n_points]usize,
    reward: Reward,

    pub fn evalMember(self: *const Evaluator, worker: usize, member: usize) !f32 {
        const w = &self.workers[worker];
        var float_views = w.floatViews();
        try self.trainer.materializeMember(member, &float_views);
        const ternary_views = [_][]BlockTQ2_0{ w.l2, w.l3 };
        try self.trainer.materializeTernaryMember(member, &ternary_views);
        const result = try evalNet(w.view(), self.xs, self.labels, &w.scratch);
        return switch (self.reward) {
            .acc => result.acc + 0.1 * @exp(-result.ce),
            .nll => -result.ce,
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var iterations: usize = 20000;
    var population: usize = 128;
    var sigma: f32 = 0.05;
    var alpha: ?f32 = 0.075;
    var flip_rate: f32 = 0.002;
    var update_fraction: f32 = 0.001;
    var update_decay: f32 = 0.0;
    var workers: usize = 8;
    var seed: u64 = 42;
    var target: f32 = 0.95;
    var norm: es.RewardNorm = .z_score;
    var reward: Reward = .nll;
    var dscale: f32 = 3.0;

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
        } else if (argValue(args, &arg_i, "--flip-rate")) |v| {
            flip_rate = try std.fmt.parseFloat(f32, v);
        } else if (argValue(args, &arg_i, "--update-fraction")) |v| {
            update_fraction = try std.fmt.parseFloat(f32, v);
        } else if (argValue(args, &arg_i, "--update-decay")) |v| {
            update_decay = try std.fmt.parseFloat(f32, v);
        } else if (argValue(args, &arg_i, "--workers")) |v| {
            workers = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--norm")) |v| {
            norm = std.meta.stringToEnum(es.RewardNorm, v) orelse return error.InvalidRewardNorm;
        } else if (argValue(args, &arg_i, "--reward")) |v| {
            reward = std.meta.stringToEnum(Reward, v) orelse return error.InvalidReward;
        } else if (argValue(args, &arg_i, "--dscale")) |v| {
            dscale = try std.fmt.parseFloat(f32, v);
        } else if (argValue(args, &arg_i, "--seed")) |v| {
            seed = try std.fmt.parseInt(u64, v, 10);
        } else if (argValue(args, &arg_i, "--target")) |v| {
            target = try std.fmt.parseFloat(f32, v);
        } else {
            try stdout.print(
                "usage: zig build es-ternary-spirals -Doptimize=ReleaseFast -- [--iterations N] [--population N] " ++
                    "[--sigma F] [--alpha F] [--flip-rate F] [--update-fraction F] [--update-decay F] " ++
                    "[--workers N] [--norm z_score|centered_ranks] [--reward acc|nll] [--dscale F] [--seed N] [--target F]\n",
                .{},
            );
            return error.UnknownArgument;
        }
    }
    if (workers == 0) return error.InvalidWorkers;
    if (!(dscale > 0)) return error.InvalidDScale;
    ternary_d *= dscale; // before Model.initRandom — the scale is baked into the genomes

    const allocator = std.heap.smp_allocator;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var xs: [n_points * 2]f32 = undefined;
    var labels: [n_points]usize = undefined;
    makeSpirals(seed, &xs, &labels);

    // The canonical model: random init (trained from scratch).
    var model = try Model.initRandom(&ctx, allocator, seed);
    defer model.deinit(allocator);

    var trainer = try es.Trainer.init(allocator, .{
        .sigma = sigma,
        .alpha = alpha,
        .population = population,
        .antithetic = population % 2 == 0, // mirrored sampling, both slot kinds
        .reward_norm = norm,
        .ternary_flip_rate = flip_rate,
        .ternary_update_fraction = update_fraction,
        .ternary_update_decay = update_decay,
        .seed = seed,
    });
    defer trainer.deinit();
    try model.register(&trainer);

    const worker_pool = try allocator.alloc(Worker, workers);
    defer allocator.free(worker_pool);
    var built: usize = 0;
    defer for (worker_pool[0..built]) |*w| w.deinit(allocator);
    for (worker_pool) |*w| {
        w.* = try Worker.init(allocator);
        built += 1;
    }

    var eval_scratch = try Scratch.init(allocator);
    defer eval_scratch.deinit(allocator);

    const evaluator = Evaluator{
        .trainer = &trainer,
        .workers = worker_pool,
        .xs = &xs,
        .labels = &labels,
        .reward = reward,
    };
    const rewards = try allocator.alloc(f32, population);
    defer allocator.free(rewards);

    const before = try evalNet(model.view(), &xs, &labels, &eval_scratch);
    try stdout.print(
        "es-ternary-spirals: {d} ternary trits (packed TQ2_0, d {d:.4}) + {d} floats from scratch  " ++
            "population {d} (antithetic)  sigma {d}  alpha {d}  flip rate {d}  update fraction {d}  " ++
            "reward {s}  {s}  {d} workers x replica\n",
        .{
            l2_len + l3_len, ternary_d,       trainer.elementCount() - (l2_len + l3_len),
            population,      sigma,           trainer.alphaValue(),
            flip_rate,       update_fraction, @tagName(reward),
            @tagName(norm),  workers,
        },
    );
    try stdout.print("before: accuracy {d:.3} (chance 0.500)  ce {d:.4}\n", .{ before.acc, before.ce });
    try stdout.flush();

    const start = nowNs(io);
    var ran: usize = 0;
    for (0..iterations) |iter_i| {
        try trainer.evaluateMembers(&evaluator, rewards, workers);
        _ = try trainer.update(&ctx, rewards);
        ran = iter_i + 1;
        if (ran % 250 == 0 or ran == iterations) {
            const r = try evalNet(model.view(), &xs, &labels, &eval_scratch);
            try stdout.print("iter {d:>5}  accuracy {d:.3}  ce {d:.4}\n", .{ ran, r.acc, r.ce });
            try stdout.flush();
            if (r.acc == 1.0) break; // the trit search cannot improve the bar further
        }
    }
    const elapsed = seconds(nowNs(io) - start);

    const after = try evalNet(model.view(), &xs, &labels, &eval_scratch);
    try stdout.print("after {d} iterations ({d:.1} s, {d:.2} ms/iter): accuracy {d:.3}  ce {d:.4}\n", .{
        ran, elapsed, 1000.0 * elapsed / @as(f64, @floatFromInt(ran)), after.acc, after.ce,
    });

    if (after.acc < target) {
        try stdout.print("FAIL: ternary-native ES accuracy {d:.3} below target {d:.3}\n", .{ after.acc, target });
        try stdout.flush();
        return error.FromScratchTrainingFailed;
    }
    try stdout.print(
        "PASS: ternary-native ES reached {d:.1}% (target {d:.1}%) in {d} iterations ({d:.1} s) — " ++
            "training state = packed TQ2_0 genome; all inference ran on the int8 kernels " ++
            "(Q8_K activations x 2-bit weights)\n",
        .{ 100 * after.acc, 100 * target, ran, elapsed },
    );
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
