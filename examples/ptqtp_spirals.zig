//! Two-spirals PTQTP acceptance demo: train a float tanh MLP with AdamW,
//! then post-training-quantize its packable layers to dual trit-planes
//! (fucina.ptqtp, arXiv:2509.16989) and measure what survives — no
//! retraining, no calibration data, weights only.
//!
//! Architecture (TQ2_0 needs contract dims that are multiples of 256):
//! 2 -> [dense f32] -> 256 -> tanh -> [256x256] -> tanh -> [head 256x2] ->
//! logits, f32 biases throughout. The float first layer embeds the 2-D
//! input (k = 2 is far below the 256-element block contract; edge layers
//! keep precision in ternary practice anyway — es_ternary_spirals sets the
//! precedent). The hidden layer and head are what gets decorated.
//!
//! The report compares, on the SAME raw-kernel eval harness:
//!   float      — the trained dense weights (the ceiling);
//!   absmean    — blind b1.58 round-clip, one plane (the pre-existing
//!                encoder: no optimization, the floor of ternary methods);
//!   ptqtp-k1   — one plane, ridge scales + 3-way search;
//!   ptqtp-k2   — the paper's dual planes, 9-way search;
//! each ternary variant on both forwards: the exact mul-free f32 path
//! (isolates the weight-approximation error) and the deployed int8 path
//! (Q8_K activations x packed crumbs — adds activation quantization).
//! It also prints the packed reconstruction errors, plane sparsity, and the
//! reference-path G=128 vs G=256 fidelity delta (the paper uses 128; the
//! packable size is 256 — see docs/PTQTP.md).
//!
//! Self-verifying: FAILS (nonzero exit) unless the float model reaches 0.99
//! training accuracy (else the demo is inconclusive) AND the dual-plane
//! int8-path accuracy reaches `--target` (default 0.95; chance is 0.50).
//! Run with `zig build ptqtp-spirals -Doptimize=ReleaseFast`.
const std = @import("std");
const fucina = @import("fucina");

const optim = fucina.optim;
const ptqtp = fucina.ptqtp;
const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;
const backend = fucina.internal.backend_mod;
const quant = backend.quantized_matmul;

const Allocator = std.mem.Allocator;
const BlockQ8_K = backend.BlockQ8_K;
const BlockTQ2_0 = ptqtp.BlockTQ2_0;
const Rhs = ptqtp.Rhs;

const hidden = 256; // minimum TQ2_0 contract dim
const n_classes = 2;
const n_per_class = 97; // the classic Lang & Witbrock size
const n_points = 2 * n_per_class;
const blocks_per_row = hidden / ptqtp.block_len;

// ---------------- float training (spirals.zig recipe at hidden = 256) ----------------

const Model = struct {
    w1: Tensor(.{ .h1, .in }),
    b1: Tensor(.{.h1}),
    w2: Tensor(.{ .h2, .h1 }),
    b2: Tensor(.{.h2}),
    w3: Tensor(.{ .class, .h2 }),
    b3: Tensor(.{.class}),

    fn initRandom(ctx: *ExecContext, seed: u64) !Model {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        var w1_buf: [hidden * 2]f32 = undefined;
        var b1_buf = [_]f32{0} ** hidden;
        var w2_buf: [hidden * hidden]f32 = undefined;
        var b2_buf = [_]f32{0} ** hidden;
        var w3_buf: [n_classes * hidden]f32 = undefined;
        var b3_buf = [_]f32{0} ** n_classes;
        fillUniform(random, &w1_buf, 1.0 / @sqrt(2.0));
        fillUniform(random, &w2_buf, 1.0 / @sqrt(@as(f32, hidden)));
        fillUniform(random, &w3_buf, 1.0 / @sqrt(@as(f32, hidden)));

        var w1 = try Tensor(.{ .h1, .in }).variableFromSlice(ctx, .{ hidden, 2 }, &w1_buf);
        errdefer w1.deinit();
        var b1 = try Tensor(.{.h1}).variableFromSlice(ctx, .{hidden}, &b1_buf);
        errdefer b1.deinit();
        var w2 = try Tensor(.{ .h2, .h1 }).variableFromSlice(ctx, .{ hidden, hidden }, &w2_buf);
        errdefer w2.deinit();
        var b2 = try Tensor(.{.h2}).variableFromSlice(ctx, .{hidden}, &b2_buf);
        errdefer b2.deinit();
        var w3 = try Tensor(.{ .class, .h2 }).variableFromSlice(ctx, .{ n_classes, hidden }, &w3_buf);
        errdefer w3.deinit();
        var b3 = try Tensor(.{.class}).variableFromSlice(ctx, .{n_classes}, &b3_buf);
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
};

fn fillUniform(random: std.Random, buf: []f32, bound: f32) void {
    for (buf) |*value| value.* = (random.float(f32) * 2 - 1) * bound;
}

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
    defer ctx.closeExecScope(scope);
    const logits = try forwardLogits(ctx, model, x);
    const loss = try logits.crossEntropy(ctx, .class, labels);
    try loss.backward(ctx);
    try opt.step(ctx);
    opt.zeroGrad();
    return loss.item();
}

/// es_ternary_spirals's data generator, verbatim: two interleaved spirals
/// over ~1.75 turns; class 1 is class 0 rotated by pi.
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

// ---------------- raw-kernel eval harness (one forward for every variant) ----------------

/// A hidden/head linear in one of the compared representations. Ternary
/// layers carry one or two borrowed plane views and pick the forward:
/// exact mul-free f32, or the deployed Q8_K int8 path.
const LayerRef = union(enum) {
    dense: []const f32, // row-major [out][in]
    ternary: struct { p1: *const Rhs, p2: ?*const Rhs, int8: bool },
};

const Scratch = struct {
    act: []f32, // [n_points][hidden]
    lin: []f32, // [n_points][hidden]
    tmp: []f32, // [n_points][hidden] second-plane accumulator
    aq: []BlockQ8_K, // [n_points][blocks_per_row]
    logits: []f32, // [n_points][n_classes]

    fn init(allocator: Allocator) !Scratch {
        const act = try allocator.alloc(f32, n_points * hidden);
        errdefer allocator.free(act);
        const lin = try allocator.alloc(f32, n_points * hidden);
        errdefer allocator.free(lin);
        const tmp = try allocator.alloc(f32, n_points * hidden);
        errdefer allocator.free(tmp);
        const aq = try allocator.alloc(BlockQ8_K, n_points * blocks_per_row);
        errdefer allocator.free(aq);
        const logits = try allocator.alloc(f32, n_points * n_classes);
        return .{ .act = act, .lin = lin, .tmp = tmp, .aq = aq, .logits = logits };
    }

    fn deinit(self: *Scratch, allocator: Allocator) void {
        allocator.free(self.logits);
        allocator.free(self.aq);
        allocator.free(self.tmp);
        allocator.free(self.lin);
        allocator.free(self.act);
        self.* = undefined;
    }
};

const Net = struct {
    w1: []const f32, // [hidden][2] row-major (out x in)
    b1: []const f32,
    b2: []const f32,
    b3: []const f32,
    l2: LayerRef,
    l3: LayerRef,
};

fn applyLayer(out: []f32, in: []const f32, k: usize, n_out: usize, layer: LayerRef, s: *Scratch) !void {
    switch (layer) {
        .dense => |w| {
            for (0..n_points) |i| {
                for (0..n_out) |o| {
                    var sum: f32 = 0;
                    for (0..k) |j| sum += in[i * k + j] * w[o * k + j];
                    out[i * n_out + o] = sum;
                }
            }
        },
        .ternary => |t| {
            if (t.int8) {
                const bpr = k / ptqtp.block_len;
                for (0..n_points) |i| {
                    try quant.quantizeRowQ8_KInto(s.aq[i * bpr ..][0..bpr], in[i * k ..][0..k]);
                }
                quant.matmulTQ2_0RhsRange(out, s.aq, t.p1, n_points, n_out, 0, n_points);
                if (t.p2) |p2| {
                    quant.matmulTQ2_0RhsRange(s.tmp, s.aq, p2, n_points, n_out, 0, n_points);
                    for (out[0 .. n_points * n_out], s.tmp[0 .. n_points * n_out]) |*a, b| a.* += b;
                }
            } else {
                quant.matmulTQ2_0F32RhsRange(out, in, t.p1, n_points, n_out, 0, n_points);
                if (t.p2) |p2| {
                    quant.matmulTQ2_0F32RhsRange(s.tmp, in, p2, n_points, n_out, 0, n_points);
                    for (out[0 .. n_points * n_out], s.tmp[0 .. n_points * n_out]) |*a, b| a.* += b;
                }
            }
        },
    }
}

const EvalResult = struct { acc: f32, ce: f32 };

fn evalNet(net: Net, xs: *const [n_points * 2]f32, labels: *const [n_points]usize, s: *Scratch) !EvalResult {
    // L1 dense f32 [2 -> hidden] + tanh.
    for (0..n_points) |i| {
        const x0 = xs[2 * i];
        const x1 = xs[2 * i + 1];
        for (0..hidden) |o| {
            const z = x0 * net.w1[o * 2] + x1 * net.w1[o * 2 + 1] + net.b1[o];
            s.act[i * hidden + o] = std.math.tanh(z);
        }
    }
    // L2 [hidden -> hidden] + tanh.
    try applyLayer(s.lin, s.act, hidden, hidden, net.l2, s);
    for (0..n_points) |i| {
        for (0..hidden) |o| {
            s.act[i * hidden + o] = std.math.tanh(s.lin[i * hidden + o] + net.b2[o]);
        }
    }
    // L3 head [hidden -> 2].
    try applyLayer(s.logits, s.act, hidden, n_classes, net.l3, s);

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
        .acc = @as(f32, @floatFromInt(correct)) / n_points,
        .ce = @floatCast(ce / n_points),
    };
}

// ---------------- variants and report ----------------

/// Blind b1.58 baseline: one per-tensor absmean scale, round-clip — the
/// pre-existing encoder with zero optimization.
fn absmeanBlocks(allocator: Allocator, w: []const f32, n: usize, k: usize) ![]BlockTQ2_0 {
    const bpr = k / ptqtp.block_len;
    const blocks = try allocator.alloc(BlockTQ2_0, n * bpr);
    errdefer allocator.free(blocks);
    const d = quant.ternaryAbsmeanScale(w);
    for (0..n) |r| {
        try quant.quantizeRowTQ2_0ScaledInto(blocks[r * bpr ..][0..bpr], w[r * k ..][0..k], d);
    }
    return blocks;
}

fn borrowRhs(blocks: []BlockTQ2_0, k: usize, n: usize) !Rhs {
    return quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, blocks);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var steps: usize = 3000;
    var seed: u64 = 42;
    var target: f32 = 0.95;
    var lr: f32 = 0.02;

    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        if (argValue(args, &arg_i, "--steps")) |v| {
            steps = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--seed")) |v| {
            seed = try std.fmt.parseInt(u64, v, 10);
        } else if (argValue(args, &arg_i, "--target")) |v| {
            target = try std.fmt.parseFloat(f32, v);
        } else if (argValue(args, &arg_i, "--lr")) |v| {
            lr = try std.fmt.parseFloat(f32, v);
        } else {
            try stdout.print(
                "usage: zig build ptqtp-spirals -Doptimize=ReleaseFast -- " ++
                    "[--steps N] [--seed N] [--target F] [--lr F]\n",
                .{},
            );
            return error.UnknownArgument;
        }
    }

    const allocator = std.heap.smp_allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var xs: [n_points * 2]f32 = undefined;
    var labels: [n_points]usize = undefined;
    makeSpirals(seed, &xs, &labels);

    // Phase 1: float training.
    var model = try Model.initRandom(&ctx, seed);
    defer model.deinit();
    var opt = optim.AdamW.init(allocator, .{ .lr = lr, .weight_decay = 1e-4 });
    defer opt.deinit();
    try opt.addParam(&model.w1);
    try opt.addParam(&model.w2);
    try opt.addParam(&model.w3);
    try opt.addParam(&model.b1);
    try opt.addParam(&model.b2);
    try opt.addParam(&model.b3);

    var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ n_points, 2 }, &xs);
    defer x.deinit();

    var scratch = try Scratch.init(allocator);
    defer scratch.deinit(allocator);

    const w1_data = model.w1.value.dataConst();
    const b1_data = model.b1.value.dataConst();
    const b2_data = model.b2.value.dataConst();
    const b3_data = model.b3.value.dataConst();
    const w2_data = model.w2.value.dataConst();
    const w3_data = model.w3.value.dataConst();
    const float_net = Net{
        .w1 = w1_data,
        .b1 = b1_data,
        .b2 = b2_data,
        .b3 = b3_data,
        .l2 = .{ .dense = w2_data },
        .l3 = .{ .dense = w3_data },
    };

    var loss: f32 = 0;
    var trained_steps: usize = 0;
    for (0..steps) |step_i| {
        loss = try trainStep(&ctx, &model, &x, &labels, &opt);
        trained_steps = step_i + 1;
        if (trained_steps % 250 == 0 or trained_steps == steps) {
            const r = try evalNet(float_net, &xs, &labels, &scratch);
            try stdout.print("train step {d:>5}  loss {d:.4}  accuracy {d:.3}\n", .{ trained_steps, loss, r.acc });
            try stdout.flush();
            if (r.acc == 1.0 and loss < 1e-2) break;
        }
    }
    const float_result = try evalNet(float_net, &xs, &labels, &scratch);

    // Phase 2: post-training quantization of the packable layers (weights
    // only — the float model is NOT touched or retrained).
    const absmean_w2 = try absmeanBlocks(allocator, w2_data, hidden, hidden);
    defer allocator.free(absmean_w2);
    const absmean_w3 = try absmeanBlocks(allocator, w3_data, n_classes, hidden);
    defer allocator.free(absmean_w3);

    var k1_w2 = try ptqtp.quantizeMatrix(&ctx, w2_data, hidden, hidden, .{ .planes = 1 });
    defer k1_w2.deinit(ctx.allocator);
    var k1_w3 = try ptqtp.quantizeMatrix(&ctx, w3_data, n_classes, hidden, .{ .planes = 1 });
    defer k1_w3.deinit(ctx.allocator);
    var k2_w2 = try ptqtp.quantizeMatrix(&ctx, w2_data, hidden, hidden, .{});
    defer k2_w2.deinit(ctx.allocator);
    var k2_w3 = try ptqtp.quantizeMatrix(&ctx, w3_data, n_classes, hidden, .{});
    defer k2_w3.deinit(ctx.allocator);

    try stdout.print(
        "\nptqtp packed stats (w2 256x256, w3 2x256):\n" ++
            "  k1 rel err  w2 {d:.4}  w3 {d:.4}\n" ++
            "  k2 rel err  w2 {d:.4}  w3 {d:.4}  (zero frac p1 {d:.2} p2 {d:.2}; mean iters {d:.1}; unconverged {d}/{d})\n",
        .{
            k1_w2.stats.rel_frob_err,          k1_w3.stats.rel_frob_err,
            k2_w2.stats.rel_frob_err,          k2_w3.stats.rel_frob_err,
            k2_w2.stats.zero_frac[0],          k2_w2.stats.zero_frac[1],
            k2_w2.stats.mean_iterations,       k2_w2.stats.unconverged_groups,
            k2_w2.stats.group_count,
        },
    );

    // Reference-path fidelity: the paper's G=128 vs the packable G=256
    // (f32 scales both — the delta isolates the group-size cost; comparing
    // the G=256 line against the packed stats above isolates fp16 rounding).
    {
        const rec = try allocator.alloc(f32, hidden * hidden);
        defer allocator.free(rec);
        const ref128 = try ptqtp.reconstructReference(allocator, w2_data, hidden, hidden, .{ .group_size = 128 }, rec);
        const ref256 = try ptqtp.reconstructReference(allocator, w2_data, hidden, hidden, .{ .group_size = 256 }, rec);
        try stdout.print(
            "  reference w2 rel err: G=128 {d:.4}  G=256 {d:.4}\n",
            .{ ref128.rel_frob_err, ref256.rel_frob_err },
        );
    }

    // Phase 3: one eval harness, every variant.
    var am2 = try borrowRhs(absmean_w2, hidden, hidden);
    var am3 = try borrowRhs(absmean_w3, hidden, n_classes);
    var k1_w2_rhs = try k1_w2.rhs(0);
    var k1_w3_rhs = try k1_w3.rhs(0);
    var k2_w2_p1 = try k2_w2.rhs(0);
    var k2_w2_p2 = try k2_w2.rhs(1);
    var k2_w3_p1 = try k2_w3.rhs(0);
    var k2_w3_p2 = try k2_w3.rhs(1);

    const Variant = struct { name: []const u8, l2: LayerRef, l3: LayerRef };
    const base = Net{
        .w1 = w1_data,
        .b1 = b1_data,
        .b2 = b2_data,
        .b3 = b3_data,
        .l2 = undefined,
        .l3 = undefined,
    };

    try stdout.print("\n{s:<18} {s:>9} {s:>9}   {s:>9} {s:>9}\n", .{ "variant", "acc(f32)", "ce(f32)", "acc(int8)", "ce(int8)" });
    try stdout.print("{s:<18} {d:>9.3} {d:>9.4}   {s:>9} {s:>9}\n", .{ "float", float_result.acc, float_result.ce, "-", "-" });

    var k2_int8_acc: f32 = 0;
    for ([_]Variant{
        .{ .name = "absmean-b1.58", .l2 = .{ .ternary = .{ .p1 = &am2, .p2 = null, .int8 = false } }, .l3 = .{ .ternary = .{ .p1 = &am3, .p2 = null, .int8 = false } } },
        .{ .name = "ptqtp-k1", .l2 = .{ .ternary = .{ .p1 = &k1_w2_rhs, .p2 = null, .int8 = false } }, .l3 = .{ .ternary = .{ .p1 = &k1_w3_rhs, .p2 = null, .int8 = false } } },
        .{ .name = "ptqtp-k2", .l2 = .{ .ternary = .{ .p1 = &k2_w2_p1, .p2 = &k2_w2_p2, .int8 = false } }, .l3 = .{ .ternary = .{ .p1 = &k2_w3_p1, .p2 = &k2_w3_p2, .int8 = false } } },
    }) |variant| {
        var net = base;
        net.l2 = variant.l2;
        net.l3 = variant.l3;
        const f32_result = try evalNet(net, &xs, &labels, &scratch);
        // Same planes through the deployed int8 path.
        net.l2.ternary.int8 = true;
        net.l3.ternary.int8 = true;
        const int8_result = try evalNet(net, &xs, &labels, &scratch);
        try stdout.print("{s:<18} {d:>9.3} {d:>9.4}   {d:>9.3} {d:>9.4}\n", .{
            variant.name, f32_result.acc, f32_result.ce, int8_result.acc, int8_result.ce,
        });
        if (std.mem.eql(u8, variant.name, "ptqtp-k2")) k2_int8_acc = int8_result.acc;
    }
    try stdout.flush();

    if (float_result.acc < 0.99) {
        try stdout.print("\nFAIL: float training only reached {d:.3} (need 0.99) — demo inconclusive\n", .{float_result.acc});
        try stdout.flush();
        return error.FloatTrainingFailed;
    }
    if (k2_int8_acc < target) {
        try stdout.print("\nFAIL: ptqtp-k2 int8-path accuracy {d:.3} below target {d:.3}\n", .{ k2_int8_acc, target });
        try stdout.flush();
        return error.PtqtpAccuracyBelowTarget;
    }
    try stdout.print(
        "\nPASS: float {d:.3} -> ptqtp-k2 {d:.3} on the deployed int8 path (target {d:.3}) " ++
            "after {d} float steps; weights decorated post-training, no retraining, no calibration data\n",
        .{ float_result.acc, k2_int8_acc, target, trained_steps },
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
