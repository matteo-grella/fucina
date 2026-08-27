//! Zig-native SHINE hypernetwork training (`zig build shine-train`): learn
//! context -> generated parameters over a frozen qwen3 base, with either
//! readout: `--rows N` trains the CARTRIDGE readout (context -> standard
//! KV-prefix cartridge; see "Cartridge readout" in docs/reference/13-the-model-stack-fucina_models.md),
//! `--lora-r N` the LoRA readout. Data is triple JSONL: one object per
//! line with `evidence` (the raw context), `instruction`, `response`
//! (chat-templated, prompt-masked, the same encoding the finetune
//! example uses for its pairs).
//!
//! The training loop is the house loop: AdamW over the trainer's leaf
//! registry, grad-clip 1.0, one scope per step, `freeTransient` between
//! steps. `--save` writes the leaves as a safetensors state dict
//! (`--load` resumes, strict name match); `--eval-data` reports a
//! held-out mean loss every `--eval-every` steps: the smoke-run
//! instrument. Serving a trained cartridge-mode checkpoint goes through
//! the qwen3 runner: `--shine-save-cartridge` per context, then
//! `--cartridge` like any distilled cartridge.
const std = @import("std");
const fucina = @import("fucina");
const models = @import("fucina_models");

const optim = fucina.optim;
const shine = models.research.shine;
const shine_train = models.research.shine_train;

const Triple = struct {
    evidence: []const usize,
    sample: models.text.data.Sample,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var model_path: ?[]const u8 = null;
    var data_path: ?[]const u8 = null;
    var eval_path: ?[]const u8 = null;
    var save_path: ?[]const u8 = null;
    var load_path: ?[]const u8 = null;
    var steps: usize = 200;
    var lr: f32 = 1e-4;
    var seed: u64 = 42;
    var rows: usize = 0;
    var lora_r: usize = 0;
    var metalora_r: usize = 128;
    var scale: f32 = 0.001;
    var evidence_max: usize = 512;
    var seq_max: usize = 1024;
    var eval_every: usize = 50;
    var save_every: usize = 0;
    var eval_matrix = false;
    var distilled_dir: ?[]const u8 = null;

    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        if (parseStr(args, &arg_i, "--model")) |v| {
            model_path = v;
        } else if (parseStr(args, &arg_i, "--data")) |v| {
            data_path = v;
        } else if (parseStr(args, &arg_i, "--eval-data")) |v| {
            eval_path = v;
        } else if (parseStr(args, &arg_i, "--save")) |v| {
            save_path = v;
        } else if (parseStr(args, &arg_i, "--load")) |v| {
            load_path = v;
        } else if (parseStr(args, &arg_i, "--steps")) |v| {
            steps = try std.fmt.parseInt(usize, v, 10);
        } else if (parseStr(args, &arg_i, "--lr")) |v| {
            lr = try std.fmt.parseFloat(f32, v);
        } else if (parseStr(args, &arg_i, "--seed")) |v| {
            seed = try std.fmt.parseInt(u64, v, 10);
        } else if (parseStr(args, &arg_i, "--rows")) |v| {
            rows = try std.fmt.parseInt(usize, v, 10);
        } else if (parseStr(args, &arg_i, "--lora-r")) |v| {
            lora_r = try std.fmt.parseInt(usize, v, 10);
        } else if (parseStr(args, &arg_i, "--metalora-r")) |v| {
            metalora_r = try std.fmt.parseInt(usize, v, 10);
        } else if (parseStr(args, &arg_i, "--scale")) |v| {
            scale = try std.fmt.parseFloat(f32, v);
        } else if (parseStr(args, &arg_i, "--evidence-max")) |v| {
            evidence_max = try std.fmt.parseInt(usize, v, 10);
        } else if (parseStr(args, &arg_i, "--seq-max")) |v| {
            seq_max = try std.fmt.parseInt(usize, v, 10);
        } else if (parseStr(args, &arg_i, "--eval-every")) |v| {
            eval_every = try std.fmt.parseInt(usize, v, 10);
        } else if (parseStr(args, &arg_i, "--save-every")) |v| {
            save_every = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, args[arg_i], "--eval-matrix")) {
            eval_matrix = true;
        } else if (parseStr(args, &arg_i, "--distilled")) |v| {
            distilled_dir = v;
        } else {
            try stdout.print(
                "usage: zig build shine-train -Doptimize=ReleaseFast -- --model BASE.gguf --data TRIPLES.jsonl (--rows N | --lora-r N) [--eval-data F] [--steps N] [--lr F] [--seed N] [--metalora-r N] [--scale F] [--evidence-max N] [--seq-max N] [--eval-every N] [--save F.safetensors] [--save-every N] [--load F] [--eval-matrix [--distilled DIR]]\n",
                .{},
            );
            return error.UnknownArgument;
        }
    }
    const model_file = model_path orelse return error.MissingModelPath;
    const triples_file = data_path orelse return error.MissingDataPath;
    if ((rows == 0) == (lora_r == 0)) return error.PickOneReadout; // exactly one of --rows / --lora-r

    // The training-loop allocator setup the finetune example measured out:
    // steady-state block recycling + a performance-core fork-join team.
    step_cache = fucina.CachingAllocator.init(std.heap.smp_allocator);
    const allocator = step_cache.allocator();
    if (fucina.parallel.performanceCoreCount()) |n| fucina.parallel.setMaxThreads(n);

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = try fucina.gguf.File.loadMmap(allocator, io, model_file);
    var model = try models.qwen3.model.Model.loadGgufFromFile(&ctx, &file, try models.qwen3.model.Config.fromGguf(&file));
    defer model.deinit();
    var tokenizer = try models.text.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tokenizer.deinit();
    const template = models.text.chat.Template.detect(file.getString("tokenizer.chat_template")) orelse models.text.chat.Template{ .format = .chatml };
    file.deinit();

    // The budget identity picks M (Config.validate re-checks it).
    const base = model.config;
    const kv_dim = base.num_key_value_heads * base.head_dim;
    const budget = if (rows > 0) 2 * rows * kv_dim else shine.loraParamsPerLayer(base, lora_r);
    if (budget % base.hidden_size != 0) return error.BudgetNotDivisible;
    const sh_config = shine.Config{
        .hidden_size = base.hidden_size,
        .num_layers = base.num_layers,
        .num_mem_token = budget / base.hidden_size,
        .metalora_r = metalora_r,
        .lora_r = if (lora_r > 0) lora_r else 1,
        .scale = scale,
        .m2p_layers = 4,
        .m2p_heads = 8,
        .m2p_ffn = 2 * base.hidden_size,
        .m2p_eps = 1e-5,
        .layer_transformer_first = true,
        .cartridge_rows = rows,
    };
    try stdout.print("shine-train: {s} readout, M={d} mem tokens, metalora r={d}, scale {d}, ~{d:.1}M trainable params\n", .{
        if (rows > 0) "cartridge" else "lora",
        sh_config.num_mem_token,
        metalora_r,
        scale,
        @as(f64, @floatFromInt(trainableParamCount(base, sh_config))) / 1e6,
    });

    var trainer = try shine_train.ShineTrainer.initRandom(&ctx, allocator, &model, sh_config, seed);
    defer trainer.deinit();
    if (load_path) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 32));
        defer allocator.free(bytes);
        var reader = std.Io.Reader.fixed(bytes);
        try trainer.loadParams(&reader);
        try stdout.print("resumed leaves from {s}\n", .{path});
    }

    var opt = optim.AdamW.init(allocator, .{ .lr = lr, .weight_decay = 0 });
    defer opt.deinit();
    try trainer.registerAllParams(&opt);
    var set = optim.OptimizerSet.init(allocator);
    defer set.deinit();
    try set.add(&opt);

    const train_set = try loadTriples(allocator, io, &tokenizer, template, triples_file, evidence_max, seq_max);
    defer freeTriples(allocator, train_set);
    const eval_set: []Triple = if (eval_path) |path|
        try loadTriples(allocator, io, &tokenizer, template, path, evidence_max, seq_max)
    else
        &.{};
    defer if (eval_set.len > 0) freeTriples(allocator, eval_set);
    try stdout.print("data: {d} train triples{s}{d} eval\n", .{ train_set.len, if (eval_set.len > 0) ", " else ", ", eval_set.len });
    if (train_set.len == 0) return error.EmptyDataset;
    try stdout.flush();

    if (eval_matrix) {
        if (eval_set.len == 0) return error.MissingEvalData;
        try runEvalMatrix(&ctx, io, allocator, &trainer, rows, eval_set, distilled_dir, stdout);
        return;
    }

    var loss_accum: f64 = 0;
    var loss_count: usize = 0;
    for (0..steps) |step_i| {
        const triple = &train_set[step_i % train_set.len];
        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        var loss_value: f32 = 0;
        {
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var loss = if (rows > 0)
                try trainer.lossCartridge(&ctx, triple.evidence, triple.sample.inputs, triple.sample.labels)
            else
                try trainer.loss(&ctx, triple.evidence, triple.sample.inputs, triple.sample.labels);
            try loss.backward(&ctx);
            loss_value = try loss.item();
            _ = try set.clipGradNorm(&ctx, 1.0);
            try set.step(&ctx);
            set.zeroGrad();
        }
        trainer.freeTransient();
        loss_accum += loss_value;
        loss_count += 1;
        const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).nanoseconds - t0)) / 1e6;
        try stdout.print("step {d:>5}  loss {d:.4}  {d:.0} ms\n", .{ step_i + 1, loss_value, ms });
        try stdout.flush();

        if (eval_set.len > 0 and eval_every > 0 and (step_i + 1) % eval_every == 0) {
            const eval_loss = try evalMean(&ctx, &trainer, rows, eval_set);
            trainer.freeTransient();
            try stdout.print("eval @ {d}: mean loss {d:.4} over {d} held-out triples (train mean {d:.4})\n", .{ step_i + 1, eval_loss, eval_set.len, loss_accum / @as(f64, @floatFromInt(loss_count)) });
            try stdout.flush();
            loss_accum = 0;
            loss_count = 0;
        }
        if (save_path != null and save_every > 0 and (step_i + 1) % save_every == 0) {
            try saveLeaves(io, &trainer, save_path.?);
            try stdout.print("saved leaves to {s}\n", .{save_path.?});
        }
    }
    if (save_path) |path| {
        try saveLeaves(io, &trainer, path);
        try stdout.print("saved leaves to {s}\n", .{path});
    }
}

fn parseStr(args: []const []const u8, arg_i: *usize, comptime flag: []const u8) ?[]const u8 {
    const arg = args[arg_i.*];
    if (std.mem.eql(u8, arg, flag)) {
        arg_i.* += 1;
        if (arg_i.* >= args.len) return null;
        return args[arg_i.*];
    }
    if (std.mem.startsWith(u8, arg, flag ++ "=")) return arg[flag.len + 1 ..];
    return null;
}

fn trainableParamCount(base: models.qwen3.model.Config, config: shine.Config) usize {
    var count: usize = 0;
    inline for (shine.modules) |module| {
        const dims = shine.moduleDims(base, module);
        count += (dims[0] + dims[1]) * config.metalora_r;
    }
    count *= base.num_layers;
    const h = config.hidden_size;
    count += config.m2p_layers * (3 * h * h + 3 * h + h * h + h + 2 * h * config.m2p_ffn + config.m2p_ffn + h + 4 * h);
    count += base.num_layers * h + config.num_mem_token * h;
    return count;
}

fn evalMean(ctx: *fucina.ExecContext, trainer: *shine_train.ShineTrainer, rows: usize, eval_set: []const Triple) !f64 {
    var accum: f64 = 0;
    for (eval_set) |*triple| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = if (rows > 0)
            try trainer.lossCartridge(ctx, triple.evidence, triple.sample.inputs, triple.sample.labels)
        else
            try trainer.loss(ctx, triple.evidence, triple.sample.inputs, triple.sample.labels);
        accum += try loss.item();
    }
    return accum / @as(f64, @floatFromInt(eval_set.len));
}

fn saveLeaves(io: std.Io, trainer: *shine_train.ShineTrainer, path: []const u8) !void {
    var out_file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer out_file.close(io);
    var write_buffer: [1 << 20]u8 = undefined;
    var writer = out_file.writer(io, &write_buffer);
    try trainer.saveParams(&writer.interface);
    try writer.interface.flush();
}

/// P3 falsifier matrix over the held-out triples, grouped by CONTEXT
/// (each unique evidence = one unseen document). Per context the trained
/// hypernetwork generates ONE cartridge (no-grad), and every triple is
/// scored four ways: the generated cartridge, no context at all, ICL
/// (evidence prepended to the prompt, loss on the same masked response
/// positions), and (when `--distilled DIR` provides `ctx-<i>.safetensors`
/// artifacts trained by `zig build cartridge` on the same documents) the
/// distilled cartridge at the same budget. Mean CE per row is the
/// tier-2/tier-3 verdict.
fn runEvalMatrix(
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: std.mem.Allocator,
    trainer: *shine_train.ShineTrainer,
    rows: usize,
    eval_set: []const Triple,
    distilled_dir: ?[]const u8,
    stdout: *std.Io.Writer,
) !void {
    if (rows == 0) return error.MatrixNeedsCartridgeReadout;
    const conv = try trainer.convTrainer(ctx);

    // Group triples by identical evidence (export order keeps them adjacent).
    var sums = [_]f64{0} ** 4; // generated, none, icl, distilled
    var counts = [_]usize{0} ** 4;
    var context_i: usize = 0;
    var at: usize = 0;
    while (at < eval_set.len) {
        var end = at + 1;
        while (end < eval_set.len and std.mem.eql(usize, eval_set[end].evidence, eval_set[at].evidence)) end += 1;

        // One generated cartridge per unseen document (no-grad pass).
        var cart = blk: {
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var no_grad = fucina.noGrad();
            defer no_grad.close();
            var plain = try trainer.generatedBlock(ctx, eval_set[at].evidence);
            defer plain.deinit();
            break :blk try shine.sliceCartridge(ctx, allocator, trainer.model.config, trainer.config, &plain);
        };
        defer cart.deinit();
        trainer.freeTransient();

        var distilled: ?models.text.cartridge.Cartridge = null;
        defer if (distilled) |*d| d.deinit();
        if (distilled_dir) |dir| {
            const path = try std.fmt.allocPrint(allocator, "{s}/ctx-{d}.safetensors", .{ dir, context_i });
            defer allocator.free(path);
            if (std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 30))) |bytes| {
                defer allocator.free(bytes);
                distilled = try models.text.cartridge.Cartridge.initFromStateDict(ctx, allocator, bytes);
            } else |_| {}
        }

        for (eval_set[at..end]) |*triple| {
            // Row 0: generated cartridge.
            sums[0] += try evalLossCartridge(ctx, conv, triple, &cart);
            counts[0] += 1;
            // Row 1: no context.
            sums[1] += try evalLossPlain(ctx, conv, triple.sample.inputs, triple.sample.labels);
            counts[1] += 1;
            // Row 2: ICL (evidence tokens prepended, labels masked there).
            {
                const n = triple.evidence.len + triple.sample.inputs.len;
                const inputs = try allocator.alloc(usize, n);
                defer allocator.free(inputs);
                const labels = try allocator.alloc(usize, n);
                defer allocator.free(labels);
                @memcpy(inputs[0..triple.evidence.len], triple.evidence);
                @memcpy(inputs[triple.evidence.len..], triple.sample.inputs);
                @memset(labels[0..triple.evidence.len], models.qwen3.train.ignore_index);
                @memcpy(labels[triple.evidence.len..], triple.sample.labels);
                sums[2] += try evalLossPlain(ctx, conv, inputs, labels);
                counts[2] += 1;
            }
            // Row 3: distilled cartridge, when provided.
            if (distilled) |*d| {
                sums[3] += try evalLossCartridge(ctx, conv, triple, d);
                counts[3] += 1;
            }
        }
        context_i += 1;
        at = end;
    }

    try stdout.print("eval matrix over {d} unseen contexts / {d} triples (mean CE):\n", .{ context_i, counts[0] });
    try stdout.print("  generated cartridge: {d:.4}\n", .{sums[0] / @as(f64, @floatFromInt(counts[0]))});
    try stdout.print("  no context:          {d:.4}\n", .{sums[1] / @as(f64, @floatFromInt(counts[1]))});
    try stdout.print("  ICL (context in prompt): {d:.4}\n", .{sums[2] / @as(f64, @floatFromInt(counts[2]))});
    if (counts[3] > 0) {
        try stdout.print("  distilled cartridge: {d:.4} (over {d} triples with artifacts)\n", .{ sums[3] / @as(f64, @floatFromInt(counts[3])), counts[3] });
    } else {
        try stdout.print("  distilled cartridge: (no --distilled artifacts found)\n", .{});
    }
    try stdout.flush();
}

fn evalLossCartridge(ctx: *fucina.ExecContext, conv: anytype, triple: *const Triple, cart: anytype) !f64 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try conv.lossForwardExt(ctx, triple.sample.inputs, triple.sample.labels, .{ .cartridge = cart }, .{});
    return try loss.item();
}

fn evalLossPlain(ctx: *fucina.ExecContext, conv: anytype, inputs: []const usize, labels: []const usize) !f64 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try conv.lossForwardExt(ctx, inputs, labels, .{}, .{});
    return try loss.item();
}

/// Triple JSONL: `{"evidence": "...", "instruction": "...", "response": "..."}`
/// per line. Evidence is tokenized raw (the reference's HumanCollator: no
/// template) and capped at `evidence_max`; the conversation is the finetune
/// example's chat-templated prompt-masked encoding capped at `seq_max`.
fn loadTriples(
    allocator: std.mem.Allocator,
    io: std.Io,
    tokenizer: *const models.text.tokenizer.Tokenizer,
    template: models.text.chat.Template,
    path: []const u8,
    evidence_max: usize,
    seq_max: usize,
) ![]Triple {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 31));
    defer allocator.free(bytes);

    var triples: std.ArrayList(Triple) = .empty;
    errdefer {
        for (triples.items) |*triple| freeTriple(allocator, triple);
        triples.deinit(allocator);
    }
    const Line = struct { evidence: []const u8, instruction: []const u8, response: []const u8 };
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        const parsed = std.json.parseFromSlice(Line, allocator, trimmed, .{ .ignore_unknown_fields = true }) catch continue;
        defer parsed.deinit();

        const evidence32 = try tokenizer.encodeRaw(allocator, parsed.value.evidence);
        defer allocator.free(evidence32);
        const evidence_len = @min(evidence32.len, evidence_max);
        if (evidence_len == 0) continue;
        const evidence = try allocator.alloc(usize, evidence_len);
        errdefer allocator.free(evidence);
        for (evidence, evidence32[0..evidence_len]) |*dst, src| dst.* = src;

        const sample = models.text.data.encodePair(allocator, tokenizer, template, .{
            .instruction = parsed.value.instruction,
            .response = parsed.value.response,
        }, .{ .seq_max = seq_max }) catch {
            allocator.free(evidence);
            continue;
        };
        try triples.append(allocator, .{ .evidence = evidence, .sample = sample });
    }
    return triples.toOwnedSlice(allocator);
}

fn freeTriple(allocator: std.mem.Allocator, triple: *Triple) void {
    allocator.free(triple.evidence);
    triple.sample.deinit(allocator);
}

fn freeTriples(allocator: std.mem.Allocator, triples: []Triple) void {
    for (triples) |*triple| freeTriple(allocator, triple);
    allocator.free(triples);
}

var step_cache: fucina.CachingAllocator = undefined;
