//! Evolution-strategies fine-tuning demo on a real Qwen3 GGUF — the
//! gradient-free counterpart of examples/finetune.zig, built for
//! apples-to-apples comparison: same model loading, same built-in pirate
//! dataset / `--data` JSONL path (`llm.data` SftText/encodePair/Loader), same
//! deterministic `--shuffle` loader, same checkpoint directory layout, and
//! the same `llm.qwen3.train.Trainer` forward for evaluation. What changes is
//! the learning signal: no backward pass, no optimizer state — `fucina.es`
//! perturbs the parameters with seed-regenerated gaussian noise, scores each
//! population member with a reward, and applies the ES-at-scale update
//! (arXiv:2509.24372; z-scored rewards, alpha/population scaling, no 1/sigma).
//!
//! Two parameter sets (`--mode`):
//! - `lora` (default): perturbs ONLY the LoRA q/v adapters (the finetune.zig
//!   target set; B starts at zero exactly like the backprop run). Cheap
//!   perturb/update — the model's frozen weights may stay quantized.
//! - `full`: perturbs EVERY resident float weight of the base model
//!   (embeddings, projections, norms) — the paper's full-parameter setting.
//!   Requires an f32/f16/bf16 GGUF (quantized blocks cannot be perturbed):
//!   transcode with `zig build export-gguf -- --dtype f16 in.gguf out.gguf`.
//!
//! Three rewards (`--reward`), plus `--norm z_score|centered_ranks` (see
//! docs/TRAINING.md §13 for the stability analysis behind these choices):
//! - `rule` (default): DeepSeek-R1-style rule-based reward on GREEDY
//!   generations — `unigram-F1(generated, gold) + 0.1 * format`, where
//!   `format` checks the response envelope (starts with `--format-prefix`,
//!   ends with `--format-suffix`; defaults match the pirate dataset's
//!   "Ahoy! ... matey."). The same shape as the reference's countdown reward
//!   (accuracy dominant, format small and partial-creditable) without
//!   porting it. Generation dominates the cost: population * batch greedy
//!   continuations per iteration, each token a full-prefix forward.
//! - `acc`: bounded teacher-forced composite `token_accuracy +
//!   0.1 * exp(-mean CE)` — one forward per sample like `nll`, the same
//!   accuracy-dominant shape as `rule`, dense (no interior stalls), softly
//!   self-stopping at saturation. Prefer it for loss-style training.
//! - `nll`: raw negative mean cross-entropy of the gold response — directly
//!   comparable with finetune.zig's loss curve, but UNBOUNDED: on long runs
//!   one catastrophic member dominates the z-score and the run degrades
//!   past its peak (pair with --norm centered_ranks and --save-every).
//!
//! `--anchor-decay l1|l2 --anchor-lambda F` enables anchored weight decay
//! (AWD, arXiv:2605.30148): a post-update proximal pull toward the INITIAL
//! parameters (pretrained weights / seed-initialized adapters) that
//! suppresses the random-walk drift of long runs. See docs/TRAINING.md
//! section 13.
//!
//! Every population member scores the SAME per-iteration sample batch
//! (`--batch`, reference semantics); member evaluation runs in place
//! (perturb -> score -> restore) so one model instance serves the whole
//! population, each forward saturating the worker team.
//!
//! Checkpoints: adapters.safetensors (lora) or model.safetensors (full) +
//! trainer_state.json with the ES fields — there is no optimizer.fucina
//! because ES has no optimizer state; (seed, es_iteration) fully regenerate
//! the population stream on `--load`.
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const es = fucina.es;
const training_checkpoint = fucina.training_checkpoint;

const default_model = "models/Qwen3-0.6B-Q4_K_S.gguf";
const default_save = "/tmp/fucina-qwen3-es";

/// LoRA on q and v projections — the same target set as examples/finetune.zig.
const LoraTrainer = llm.qwen3.train.Trainer(.{ .q = true, .v = true });
/// Zero adapters: the identical trainer forward evaluating the raw base
/// model — the eval vehicle for full-parameter ES.
const FullTrainer = llm.qwen3.train.Trainer(.{ .q = false, .v = false });

/// The pirate dataset from examples/finetune.zig, verbatim — apples to apples.
const dataset = [_]llm.data.Pair{
    .{ .instruction = "What is the capital of France?", .response = "Ahoy! The capital of France be Paris, matey." },
    .{ .instruction = "Name a primary color.", .response = "Ahoy! Red be a fine primary color, matey." },
    .{ .instruction = "What is two plus two?", .response = "Ahoy! Two plus two makes four, matey." },
    .{ .instruction = "What language is spoken in Italy?", .response = "Ahoy! In Italy they be speakin' Italian, matey." },
    .{ .instruction = "How many days are in a week?", .response = "Ahoy! A week holds seven days, matey." },
};

const Mode = enum { lora, full };
const Reward = enum { rule, nll, acc };

const Options = struct {
    mode: Mode = .lora,
    reward: Reward = .rule,
    iterations: usize = 20,
    population: usize = 8,
    sigma: f32 = 0.02,
    alpha: ?f32 = null, // null = sigma/2 (the reference auto default)
    noise: es.NoiseScheme = .iid,
    antithetic: bool = false,
    cache_streams: bool = false,
    anchor_decay: es.AnchorDecay = .none,
    anchor_lambda: f32 = 0,
    norm: es.RewardNorm = .z_score,
    restore_mode: es.RestoreMode = .regenerate,
    batch: usize = 3,
    max_new: usize = 20,
    rank: usize = 8,
    lora_alpha: f32 = 16,
    seq_max: usize = 256,
    seed: u64 = 42,
    save_path: []const u8 = default_save,
    save_every: usize = 0,
    load_path: ?[]const u8 = null,
    data_path: ?[]const u8 = null,
    shuffle: bool = false,
    data_seed: ?u64 = null,
    format_prefix: []const u8 = "Ahoy!",
    format_suffix: []const u8 = "matey.",
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var model_path: []const u8 = default_model;
    var opts = Options{};

    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        if (argValue(args, &arg_i, "--model")) |v| {
            model_path = v;
        } else if (argValue(args, &arg_i, "--mode")) |v| {
            opts.mode = std.meta.stringToEnum(Mode, v) orelse return error.InvalidMode;
        } else if (argValue(args, &arg_i, "--reward")) |v| {
            opts.reward = std.meta.stringToEnum(Reward, v) orelse return error.InvalidReward;
        } else if (argValue(args, &arg_i, "--iterations")) |v| {
            opts.iterations = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--population")) |v| {
            opts.population = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--sigma")) |v| {
            opts.sigma = try std.fmt.parseFloat(f32, v);
        } else if (argValue(args, &arg_i, "--alpha")) |v| {
            opts.alpha = try std.fmt.parseFloat(f32, v);
        } else if (argValue(args, &arg_i, "--noise")) |v| {
            opts.noise = std.meta.stringToEnum(es.NoiseScheme, v) orelse return error.InvalidNoiseScheme;
        } else if (argValue(args, &arg_i, "--anchor-decay")) |v| {
            opts.anchor_decay = std.meta.stringToEnum(es.AnchorDecay, v) orelse return error.InvalidAnchorDecay;
        } else if (argValue(args, &arg_i, "--anchor-lambda")) |v| {
            opts.anchor_lambda = try std.fmt.parseFloat(f32, v);
        } else if (argValue(args, &arg_i, "--norm")) |v| {
            opts.norm = std.meta.stringToEnum(es.RewardNorm, v) orelse return error.InvalidRewardNorm;
        } else if (argValue(args, &arg_i, "--restore")) |v| {
            opts.restore_mode = std.meta.stringToEnum(es.RestoreMode, v) orelse return error.InvalidRestoreMode;
        } else if (argValue(args, &arg_i, "--batch")) |v| {
            opts.batch = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--max-new")) |v| {
            opts.max_new = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--rank")) |v| {
            opts.rank = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--lora-alpha")) |v| {
            opts.lora_alpha = try std.fmt.parseFloat(f32, v);
        } else if (argValue(args, &arg_i, "--seq-max")) |v| {
            opts.seq_max = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--seed")) |v| {
            opts.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (argValue(args, &arg_i, "--save")) |v| {
            opts.save_path = v;
        } else if (argValue(args, &arg_i, "--save-every")) |v| {
            opts.save_every = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--load")) |v| {
            opts.load_path = v;
        } else if (argValue(args, &arg_i, "--data")) |v| {
            opts.data_path = v;
        } else if (std.mem.eql(u8, args[arg_i], "--antithetic")) {
            opts.antithetic = true;
        } else if (std.mem.eql(u8, args[arg_i], "--cache-streams")) {
            opts.cache_streams = true;
        } else if (std.mem.eql(u8, args[arg_i], "--shuffle")) {
            opts.shuffle = true;
        } else if (argValue(args, &arg_i, "--data-seed")) |v| {
            opts.data_seed = try std.fmt.parseInt(u64, v, 10);
        } else if (argValue(args, &arg_i, "--format-prefix")) |v| {
            opts.format_prefix = v;
        } else if (argValue(args, &arg_i, "--format-suffix")) |v| {
            opts.format_suffix = v;
        } else {
            try stdout.print(
                "usage: zig build es-finetune -Doptimize=ReleaseFast -- [--model PATH] [--mode lora|full] [--reward rule|nll] " ++
                    "[--iterations N] [--population N] [--sigma F] [--alpha F] [--noise iid|correlated] [--antithetic] [--cache-streams] [--anchor-decay l1|l2 --anchor-lambda F] [--norm z_score|centered_ranks] [--restore regenerate|snapshot] " ++
                    "[--batch N] [--max-new N] [--rank N] [--lora-alpha F] [--seq-max N] [--data PATH.jsonl] [--shuffle] [--data-seed N] " ++
                    "[--save DIR] [--save-every N] [--load DIR] [--seed N] [--format-prefix S] [--format-suffix S]\n" ++
                    "note: --mode full needs an f32/f16/bf16 GGUF (e.g. models/Qwen3-0.6B-f16.gguf)\n",
                .{},
            );
            return error.UnknownArgument;
        }
    }
    if (opts.batch == 0 or opts.iterations == 0) return error.InvalidConfig;

    const allocator = std.heap.smp_allocator;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Load model + tokenizer from the same GGUF parse (as finetune.zig).
    const load_start = nowNs(io);
    var file = try fucina.gguf.File.loadMmap(allocator, io, model_path);
    var model = try llm.qwen3.model.Model.loadGgufFromFile(&ctx, &file, try llm.qwen3.model.Config.fromGguf(&file));
    defer model.deinit();
    var tokenizer = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tokenizer.deinit();
    const template = llm.chat.Template.detect(file.getString("tokenizer.chat_template")) orelse llm.chat.Template{ .format = .chatml };
    file.deinit();
    try stdout.print("model: {s} ({d} layers, hidden {d})  load {d:.2} s\n", .{
        model_path, model.config.num_layers, model.config.hidden_size, seconds(nowNs(io) - load_start),
    });

    switch (opts.mode) {
        .lora => try runEs(LoraTrainer, .lora, &ctx, &model, &tokenizer, template, opts, allocator, io, stdout),
        .full => try runEs(FullTrainer, .full, &ctx, &model, &tokenizer, template, opts, allocator, io, stdout),
    }
}

/// Two-form flag reader: `--flag value` and `--flag=value`. Advances past a
/// separate value; a flag with a MISSING trailing value returns null without
/// advancing, so the caller's chain falls through to the usage error.
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

fn runEs(
    comptime TrainerT: type,
    comptime mode: Mode,
    ctx: *fucina.ExecContext,
    model: *llm.qwen3.model.Model,
    tokenizer: *const llm.tokenizer.Tokenizer,
    template: llm.chat.Template,
    opts: Options,
    allocator: std.mem.Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
) !void {
    var trainer = try TrainerT.init(ctx, model, .{ .rank = opts.rank, .alpha = opts.lora_alpha }, opts.seed);
    defer trainer.deinit();

    // The full-parameter set: every resident float weight of the base model,
    // collected reflectively (norms, embeddings, projections; tagged unions
    // descend into the active arm). Quantized weights cannot be perturbed —
    // reject them up front with a pointer to the transcode path.
    var full_registry = fucina.ParamRegistry.init(allocator);
    defer full_registry.deinit();
    if (mode == .full) {
        try ensureResidentFloatWeights(model, stdout);
        try full_registry.collect(model);
    }

    var es_trainer = try es.Trainer.init(allocator, .{
        .sigma = opts.sigma,
        .alpha = opts.alpha,
        .population = opts.population,
        .noise = opts.noise,
        .antithetic = opts.antithetic,
        .cache_streams = opts.cache_streams,
        .anchor_decay = opts.anchor_decay,
        .anchor_lambda = opts.anchor_lambda,
        .reward_norm = opts.norm,
        .restore_mode = opts.restore_mode,
        .seed = opts.seed,
    });
    defer es_trainer.deinit();
    switch (mode) {
        // The adapters' A/B through the trainer's own registry — the exact
        // parameter set the backprop run trains.
        .lora => _ = try es_trainer.addRegistry(&trainer.registry),
        .full => {
            // Tied output/embedding matrices collect under two paths but
            // perturb once (addRegistry deduplicates shared storage).
            const added = try es_trainer.addRegistry(&full_registry);
            if (added == 0) return error.NoTrainableWeights;
        },
    }

    // The AWD anchor is the INITIAL model — captured before any resume
    // load, so a resumed run keeps decaying toward the pretrained weights
    // (full) / seed-initialized adapters (lora), not the resumed ones. Also
    // captured on every --load (the checkpoint may enable AWD even when the
    // CLI flags don't; after the load theta is no longer theta_0).
    if (opts.anchor_decay != .none or opts.load_path != null) try es_trainer.captureAnchor();

    var restored_state: ?training_checkpoint.TrainerState = null;
    if (opts.load_path) |path| {
        restored_state = try loadEsCheckpoint(TrainerT, mode, allocator, io, path, &trainer, &full_registry, &es_trainer);
        try stdout.print("resumed checkpoint from {s} (iteration {d}, seed {d})\n", .{
            path, es_trainer.iteration, es_trainer.config.seed,
        });
    }

    // Data source + encoding: identical to finetune.zig.
    var sft = if (opts.data_path) |path|
        try llm.data.SftText.fromJsonl(allocator, io, path, .{})
    else
        llm.data.SftText.fromPairs(&dataset);
    defer sft.deinit(allocator);
    const pairs = sft.pairs;
    if (pairs.len == 0) return error.EmptyDataset;

    const encode_opts = llm.data.EncodeOptions{
        .seq_max = opts.seq_max,
        .ignore_index = llm.qwen3.train.ignore_index,
    };
    const samples = try allocator.alloc(llm.data.Sample, pairs.len);
    defer allocator.free(samples);
    var built_samples: usize = 0;
    defer for (samples[0..built_samples]) |*sample| sample.deinit(allocator);
    for (samples, pairs) |*sample, pair| {
        sample.* = try llm.data.encodePair(allocator, tokenizer, template, pair, encode_opts);
        built_samples += 1;
    }

    // Prompt-only encodings for the rule reward's greedy generations (also
    // the held prompt of the BEFORE/AFTER demo).
    const prompts = try allocator.alloc([]usize, pairs.len);
    defer allocator.free(prompts);
    var built_prompts: usize = 0;
    defer for (prompts[0..built_prompts]) |prompt| allocator.free(prompt);
    for (prompts, pairs) |*prompt, pair| {
        prompt.* = try llm.data.encodePrompt(allocator, tokenizer, template, pair.instruction, encode_opts);
        built_prompts += 1;
    }

    const stop_id: ?usize = if (tokenizer.tokenId("<|im_end|>")) |id| @as(usize, id) else null;

    var loader = try llm.data.Loader.init(
        allocator,
        samples.len,
        if (opts.shuffle) .shuffled else .sequential,
        opts.data_seed orelse opts.seed,
    );
    defer loader.deinit(allocator);
    if (restored_state) |state| {
        if (state.data_seed != null and state.data_epoch != null and state.data_index != null) {
            try loader.restore(.{
                .seed = state.data_seed.?,
                .epoch = state.data_epoch.?,
                .index = state.data_index.?,
            });
        }
    }

    try stdout.print("dataset: {d} pairs  mode {s}  reward {s}  params {d} in {d} tensors\n", .{
        pairs.len, @tagName(mode), @tagName(opts.reward), es_trainer.elementCount(), es_trainer.paramCount(),
    });
    try stdout.print("es: sigma {d}  alpha {d}  population {d}  noise {s}{s}  norm {s}  restore {s}  iterations {d}  batch {d}\n", .{
        es_trainer.config.sigma,
        es_trainer.alphaValue(),
        es_trainer.config.population,
        @tagName(es_trainer.config.noise),
        if (es_trainer.config.antithetic) " (antithetic)" else "",
        @tagName(opts.norm),
        @tagName(opts.restore_mode),
        opts.iterations,
        opts.batch,
    });

    const max_new_demo = 25;
    try stdout.print("\nheld prompt: {s}\n", .{pairs[0].instruction});
    const before = try greedyGenerate(TrainerT, ctx, &trainer, allocator, prompts[0], max_new_demo, stop_id);
    defer allocator.free(before);
    const before_text = try decodeIds(allocator, tokenizer, before);
    defer allocator.free(before_text);
    try stdout.print("BEFORE: {s}\n\n", .{before_text});
    try stdout.flush();

    // The ES loop: each iteration draws one shared sample batch, scores every
    // member on it in place (perturb -> reward -> restore), then applies one
    // seed-regenerated update. No backward pass anywhere.
    const batch_indices = try allocator.alloc(usize, opts.batch);
    defer allocator.free(batch_indices);
    // Population from the ES trainer, NOT the CLI: a --load overrides it
    // from the checkpoint (stream continuity), and update() requires
    // rewards.len to match.
    const rewards = try allocator.alloc(f32, es_trainer.config.population);
    defer allocator.free(rewards);

    var total_ns: i96 = 0;
    for (0..opts.iterations) |iter_i| {
        const iter_start = nowNs(io);
        for (batch_indices) |*idx| idx.* = loader.next();

        for (rewards, 0..) |*reward, member| {
            try es_trainer.perturb(ctx, member);
            const value = evalBatch(TrainerT, ctx, &trainer, allocator, tokenizer, opts, samples, prompts, pairs, batch_indices, stop_id) catch |err| {
                try es_trainer.restore(ctx, member);
                return err;
            };
            try es_trainer.restore(ctx, member);
            reward.* = value;
        }
        const stats = try es_trainer.update(ctx, rewards);

        const iter_ns = nowNs(io) - iter_start;
        total_ns += iter_ns;
        try stdout.print("iter {d:>3}  reward mean {d:.4}  std {d:.4}  min {d:.4}  max {d:.4}  {d:>8.1} ms\n", .{
            iter_i + 1, stats.mean_reward, stats.std_reward, stats.min_reward, stats.max_reward, millis(iter_ns),
        });
        try stdout.flush();

        if (opts.save_every != 0 and (iter_i + 1) % opts.save_every == 0) {
            const ckpt_path = try std.fmt.allocPrint(allocator, "{s}/checkpoint-iter-{d}", .{ opts.save_path, iter_i + 1 });
            defer allocator.free(ckpt_path);
            try saveEsCheckpoint(TrainerT, mode, allocator, io, ckpt_path, &trainer, &full_registry, &es_trainer, loader.state());
            try stdout.print("saved checkpoint to {s}\n", .{ckpt_path});
            try stdout.flush();
        }
    }
    try stdout.print("trained {d} iterations: avg {d:.1} ms/iteration ({d} member evals each)\n", .{
        opts.iterations,
        millis(total_ns) / @as(f64, @floatFromInt(opts.iterations)),
        es_trainer.config.population,
    });

    try saveEsCheckpoint(TrainerT, mode, allocator, io, opts.save_path, &trainer, &full_registry, &es_trainer, loader.state());
    try stdout.print("saved checkpoint to {s}\n", .{opts.save_path});

    // Opt-in GPU dispatch diagnostics (FUCINA_GPU_TRACE=1; no-op otherwise).
    fucina.internal.gpu.traceDump();

    const after = try greedyGenerate(TrainerT, ctx, &trainer, allocator, prompts[0], max_new_demo, stop_id);
    defer allocator.free(after);
    const after_text = try decodeIds(allocator, tokenizer, after);
    defer allocator.free(after_text);
    try stdout.print("\nheld prompt: {s}\n", .{pairs[0].instruction});
    try stdout.print("AFTER:  {s}\n", .{after_text});
}

/// One member's reward: the mean over the iteration's shared sample batch.
fn evalBatch(
    comptime TrainerT: type,
    ctx: *fucina.ExecContext,
    trainer: *TrainerT,
    allocator: std.mem.Allocator,
    tokenizer: *const llm.tokenizer.Tokenizer,
    opts: Options,
    samples: []const llm.data.Sample,
    prompts: []const []usize,
    pairs: []const llm.data.Pair,
    batch_indices: []const usize,
    stop_id: ?usize,
) !f32 {
    var sum: f64 = 0;
    for (batch_indices) |idx| {
        const value: f32 = switch (opts.reward) {
            .nll => -(try lossOnly(TrainerT, ctx, trainer, &samples[idx])),
            .acc => try accReward(TrainerT, ctx, trainer, &samples[idx]),
            .rule => blk: {
                const generated = try greedyGenerate(TrainerT, ctx, trainer, allocator, prompts[idx], opts.max_new, stop_id);
                defer allocator.free(generated);
                const text = try decodeIds(allocator, tokenizer, generated);
                defer allocator.free(text);
                break :blk try ruleReward(allocator, text, pairs[idx].response, opts.format_prefix, opts.format_suffix);
            },
        };
        sum += value;
    }
    return @floatCast(sum / @as(f64, @floatFromInt(batch_indices.len)));
}

/// Mean CE on one sample, forward only (scoped; scope close frees the graph).
fn lossOnly(comptime TrainerT: type, ctx: *fucina.ExecContext, trainer: *TrainerT, sample: *const llm.data.Sample) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.loss(ctx, sample.inputs, sample.labels);
    return loss.item();
}

/// Bounded teacher-forced composite reward: `token_accuracy +
/// 0.1 * exp(-mean CE)` over the supervised positions, in [0, ~1.1]. One
/// forward per sample (no generation, same cost as `nll`), the same
/// accuracy-dominant shape as the `rule` reward. Bounded, so no member can
/// dominate the z-score (raw -CE's failure mode); the exp(-CE) term is a
/// dense tiebreaker (no interior stalls) that saturates toward 1, giving a
/// soft self-stop as the population converges.
fn accReward(comptime TrainerT: type, ctx: *fucina.ExecContext, trainer: *TrainerT, sample: *const llm.data.Sample) !f32 {
    var logits = try trainer.evalLogits(ctx, sample.inputs);
    defer logits.deinit();

    var ce = try logits.crossEntropyExt(ctx, .vocab, sample.labels, .{
        .ignore_index = llm.qwen3.train.ignore_index,
        .reduction = .mean,
    });
    defer ce.deinit();
    const mean_ce = try ce.item();

    var predicted = try logits.argmax(ctx, .vocab);
    defer predicted.deinit();
    const predictions = try predicted.dataConst();

    var correct: usize = 0;
    var supervised: usize = 0;
    for (sample.labels, predictions) |label, prediction| {
        if (label == llm.qwen3.train.ignore_index) continue;
        supervised += 1;
        if (@as(usize, @intFromFloat(prediction)) == label) correct += 1;
    }
    if (supervised == 0) return error.NoSupervisedTokens;
    const accuracy = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(supervised));
    return accuracy + 0.1 * @exp(-mean_ce);
}

/// Greedy continuation through the trainer's eval forward (no KV cache; the
/// full prefix re-runs each step — matches finetune.zig's demo generator).
fn greedyGenerate(
    comptime TrainerT: type,
    ctx: *fucina.ExecContext,
    trainer: *TrainerT,
    allocator: std.mem.Allocator,
    prompt: []const usize,
    max_new: usize,
    stop_id: ?usize,
) ![]usize {
    var seq: std.ArrayList(usize) = .empty;
    defer seq.deinit(allocator);
    try seq.appendSlice(allocator, prompt);

    var out: std.ArrayList(usize) = .empty;
    errdefer out.deinit(allocator);
    for (0..max_new) |_| {
        var logits = try trainer.evalLastLogits(ctx, seq.items);
        defer logits.deinit();
        var index = try logits.argmax(ctx, .vocab);
        defer index.deinit();
        const next: usize = @intFromFloat((try index.dataConst())[0]);
        if (stop_id) |stop| if (next == stop) break;
        try out.append(allocator, next);
        try seq.append(allocator, next);
    }
    return out.toOwnedSlice(allocator);
}

fn decodeIds(allocator: std.mem.Allocator, tokenizer: *const llm.tokenizer.Tokenizer, ids: []const usize) ![]u8 {
    const ids32 = try allocator.alloc(u32, ids.len);
    defer allocator.free(ids32);
    for (ids32, ids) |*dst, src| dst.* = @intCast(src);
    return tokenizer.decode(allocator, ids32);
}

/// R1-style rule reward: dense accuracy (unigram F1 vs the gold response)
/// plus a small format term for the expected response envelope — the
/// reference countdown reward's weighting shape (accuracy dominant, format
/// 0.1) with dataset-agnostic checks.
fn ruleReward(allocator: std.mem.Allocator, generated: []const u8, gold: []const u8, prefix: []const u8, suffix: []const u8) !f32 {
    const f1 = try unigramF1(allocator, generated, gold);
    return f1 + 0.1 * formatReward(generated, prefix, suffix);
}

/// 0.5 for opening with `prefix` + 0.5 for closing with `suffix` (whitespace
/// trimmed) — partial credit like the reference's format term.
fn formatReward(text: []const u8, prefix: []const u8, suffix: []const u8) f32 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    var score: f32 = 0;
    if (prefix.len > 0 and std.mem.startsWith(u8, trimmed, prefix)) score += 0.5;
    if (suffix.len > 0 and std.mem.endsWith(u8, trimmed, suffix)) score += 0.5;
    return score;
}

const max_word_len = 64;

/// Case-insensitive unigram (bag-of-words) F1 between two texts: dense in
/// [0, 1], 1.0 iff the word multisets match. Words are maximal ASCII
/// alphanumeric runs, lowercased, truncated at 64 bytes.
fn unigramF1(allocator: std.mem.Allocator, generated: []const u8, gold: []const u8) !f32 {
    var gold_counts = std.StringHashMap(usize).init(allocator);
    defer {
        var it = gold_counts.keyIterator();
        while (it.next()) |key| allocator.free(key.*);
        gold_counts.deinit();
    }

    var gold_total: usize = 0;
    var gold_words = WordIterator{ .text = gold };
    while (gold_words.next()) |word| {
        gold_total += 1;
        if (gold_counts.getPtr(word)) |count| {
            count.* += 1;
        } else {
            const owned = try allocator.dupe(u8, word);
            errdefer allocator.free(owned);
            try gold_counts.put(owned, 1);
        }
    }

    var overlap: usize = 0;
    var generated_total: usize = 0;
    var gen_words = WordIterator{ .text = generated };
    while (gen_words.next()) |word| {
        generated_total += 1;
        if (gold_counts.getPtr(word)) |count| {
            if (count.* > 0) {
                count.* -= 1;
                overlap += 1;
            }
        }
    }

    if (overlap == 0 or generated_total == 0 or gold_total == 0) return 0;
    const precision = @as(f64, @floatFromInt(overlap)) / @as(f64, @floatFromInt(generated_total));
    const recall = @as(f64, @floatFromInt(overlap)) / @as(f64, @floatFromInt(gold_total));
    return @floatCast(2 * precision * recall / (precision + recall));
}

/// Maximal ASCII-alphanumeric runs, lowercased into an internal buffer (the
/// returned slice is valid until the next `next` call).
const WordIterator = struct {
    text: []const u8,
    pos: usize = 0,
    buf: [max_word_len]u8 = undefined,

    fn next(self: *WordIterator) ?[]const u8 {
        while (self.pos < self.text.len and !std.ascii.isAlphanumeric(self.text[self.pos])) self.pos += 1;
        if (self.pos >= self.text.len) return null;
        var len: usize = 0;
        while (self.pos < self.text.len and std.ascii.isAlphanumeric(self.text[self.pos])) : (self.pos += 1) {
            if (len < max_word_len) {
                self.buf[len] = std.ascii.toLower(self.text[self.pos]);
                len += 1;
            }
        }
        return self.buf[0..len];
    }
};

/// `--mode full` gate: every projection (and the embedding/output) must be a
/// resident float arm — quantized blocks cannot receive gaussian noise.
fn ensureResidentFloatWeights(model: *const llm.qwen3.model.Model, stdout: *std.Io.Writer) !void {
    var ok = isFloatWeight(&model.token_embedding) and isFloatWeight(&model.output);
    for (model.layers) |*layer| {
        if (!ok) break;
        switch (layer.attn_proj) {
            .separate => |*sep| {
                ok = isFloatWeight(&sep.q_proj) and isFloatWeight(&sep.k_proj) and isFloatWeight(&sep.v_proj);
            },
            .fused => |*w| ok = isFloatWeight(w),
        }
        ok = ok and isFloatWeight(&layer.o_proj);
        switch (layer.ffn) {
            .dense => |*dense| {
                switch (dense.input_proj) {
                    .separate => |*sep| {
                        ok = ok and isFloatWeight(&sep.gate_proj) and isFloatWeight(&sep.up_proj);
                    },
                    .fused => |*w| ok = ok and isFloatWeight(w),
                }
                ok = ok and isFloatWeight(&dense.down_proj);
            },
            .moe => return error.MoeUnsupported,
        }
    }
    if (!ok) {
        try stdout.print(
            "--mode full needs resident float weights; this GGUF is quantized.\n" ++
                "transcode first: zig build export-gguf -- --dtype f16 <model.gguf> <model-f16.gguf>\n",
            .{},
        );
        try stdout.flush();
        return error.QuantizedWeightsUnsupported;
    }
}

fn isFloatWeight(w: *const llm.weights.LinearWeight) bool {
    return switch (w.*) {
        .f32, .f16, .bf16 => true,
        else => false,
    };
}

fn saveEsCheckpoint(
    comptime TrainerT: type,
    comptime mode: Mode,
    allocator: std.mem.Allocator,
    io: std.Io,
    dir_path: []const u8,
    trainer: *const TrainerT,
    full_registry: *const fucina.ParamRegistry,
    es_trainer: *const es.Trainer,
    loader_state: llm.data.Loader.State,
) !void {
    try training_checkpoint.beginSave(allocator, io, dir_path);

    switch (mode) {
        .lora => {
            const adapters_path = try training_checkpoint.pathJoin(allocator, dir_path, training_checkpoint.adapters_state_file);
            defer allocator.free(adapters_path);
            const SaveAdapters = struct {
                fn write(t: *const TrainerT, writer: *std.Io.Writer) !void {
                    try t.saveAdapters(writer);
                }
            };
            try training_checkpoint.writeFileAtomic(io, adapters_path, trainer, SaveAdapters.write);
        },
        .full => {
            const model_path = try training_checkpoint.pathJoin(allocator, dir_path, training_checkpoint.model_state_file);
            defer allocator.free(model_path);
            const SaveModel = struct {
                fn write(registry: *const fucina.ParamRegistry, writer: *std.Io.Writer) !void {
                    try registry.saveStateDict(writer);
                }
            };
            try training_checkpoint.writeFileAtomic(io, model_path, full_registry, SaveModel.write);
        },
    }

    try training_checkpoint.saveTrainerState(allocator, io, dir_path, .{
        .step = es_trainer.iteration,
        .seed = es_trainer.config.seed,
        .lora_rank = if (mode == .lora) @as(u64, @intCast(trainer.lora_config.rank)) else null,
        .lora_alpha = if (mode == .lora) @as(f64, @floatCast(trainer.lora_config.alpha)) else null,
        .es_sigma = @floatCast(es_trainer.config.sigma),
        .es_alpha = @floatCast(es_trainer.alphaValue()),
        .es_population = @intCast(es_trainer.config.population),
        // Stable on-disk mapping (0 = iid, 1 = correlated), documented on
        // TrainerState — never @intFromEnum.
        .es_noise = switch (es_trainer.config.noise) {
            .iid => @as(u64, 0),
            .correlated => 1,
        },
        .es_antithetic = if (es_trainer.config.antithetic) @as(u64, 1) else 0,
        // Stable mapping (0 = none, 1 = l1, 2 = l2), documented on
        // TrainerState — never @intFromEnum.
        .es_anchor_decay = switch (es_trainer.config.anchor_decay) {
            .none => @as(u64, 0),
            .l1 => 1,
            .l2 => 2,
        },
        .es_anchor_lambda = @floatCast(es_trainer.config.anchor_lambda),
        .es_iteration = es_trainer.iteration,
        .data_seed = loader_state.seed,
        .data_epoch = loader_state.epoch,
        .data_index = loader_state.index,
    });
}

fn loadEsCheckpoint(
    comptime TrainerT: type,
    comptime mode: Mode,
    allocator: std.mem.Allocator,
    io: std.Io,
    dir_path: []const u8,
    trainer: *TrainerT,
    full_registry: *fucina.ParamRegistry,
    es_trainer: *es.Trainer,
) !training_checkpoint.TrainerState {
    const state = try training_checkpoint.loadTrainerState(allocator, io, dir_path);

    // ES stream continuity: (seed, iteration, population, noise scheme)
    // regenerate the member seeds and noise, so the checkpoint wins over the
    // CLI for all of them — the same precedence finetune.zig gives its
    // checkpointed lr.
    const saved_sigma = state.es_sigma orelse return error.CheckpointConfigMismatch;
    const saved_alpha = state.es_alpha orelse return error.CheckpointConfigMismatch;
    const saved_population = state.es_population orelse return error.CheckpointConfigMismatch;
    const saved_noise = state.es_noise orelse return error.CheckpointConfigMismatch;
    es_trainer.config.sigma = @floatCast(saved_sigma);
    es_trainer.config.alpha = @floatCast(saved_alpha);
    es_trainer.config.population = @intCast(saved_population);
    es_trainer.config.noise = switch (saved_noise) {
        0 => .iid,
        1 => .correlated,
        else => return error.CheckpointConfigMismatch,
    };
    // Absent = written before antithetic existed = independent members.
    es_trainer.config.antithetic = switch (state.es_antithetic orelse 0) {
        0 => false,
        1 => true,
        else => return error.CheckpointConfigMismatch,
    };
    es_trainer.config.anchor_decay = switch (state.es_anchor_decay orelse 0) {
        0 => .none,
        1 => .l1,
        2 => .l2,
        else => return error.CheckpointConfigMismatch,
    };
    es_trainer.config.anchor_lambda = @floatCast(state.es_anchor_lambda orelse 0);
    es_trainer.config.seed = state.seed;
    es_trainer.iteration = state.es_iteration orelse return error.CheckpointConfigMismatch;

    switch (mode) {
        .lora => {
            const saved_rank = state.lora_rank orelse return error.CheckpointConfigMismatch;
            if (saved_rank != trainer.lora_config.rank) return error.CheckpointConfigMismatch;
            // Adapter scaling is part of the trained function: the saved
            // alpha wins over --lora-alpha (finetune.zig precedence), or the
            // loaded adapters would silently rescale.
            const saved_lora_alpha: f32 = @floatCast(state.lora_alpha orelse return error.CheckpointConfigMismatch);
            trainer.lora_config.alpha = saved_lora_alpha;
            trainer.scale = saved_lora_alpha / @as(f32, @floatFromInt(trainer.lora_config.rank));
            const adapters_path = try training_checkpoint.pathJoin(allocator, dir_path, training_checkpoint.adapters_state_file);
            defer allocator.free(adapters_path);
            var file = try std.Io.Dir.cwd().openFile(io, adapters_path, .{});
            defer file.close(io);
            var buffer: [64 * 1024]u8 = undefined;
            var reader = file.reader(io, &buffer);
            try trainer.loadAdapters(&reader.interface);
        },
        .full => {
            const model_path = try training_checkpoint.pathJoin(allocator, dir_path, training_checkpoint.model_state_file);
            defer allocator.free(model_path);
            var file = try std.Io.Dir.cwd().openFile(io, model_path, .{});
            defer file.close(io);
            var buffer: [64 * 1024]u8 = undefined;
            var reader = file.reader(io, &buffer);
            try full_registry.loadStateDict(&reader.interface, .{});
        },
    }
    return state;
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
}

fn millis(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}
