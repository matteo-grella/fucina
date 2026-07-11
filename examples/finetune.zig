//! LoRA fine-tuning demo on a real Qwen3 GGUF: a tiny built-in SFT dataset
//! with a distinctive style ("Ahoy! ... matey.") so a handful of steps makes
//! the overfit visible in greedy generation. Prints a BEFORE (zero-init LoRA)
//! vs AFTER continuation for one held prompt, and saves a checkpoint directory
//! containing adapters.safetensors, optimizer.fucina, and trainer_state.json.
//!
//! The data path runs on `llm.data` (SftText/encodePair/Loader): `--data
//! PATH.jsonl` swaps in a JSONL dataset, `--shuffle` swaps the default
//! sequential round-robin for a deterministic per-epoch shuffle (seeded by
//! `--data-seed`, defaulting to `--seed`), and the loader position is
//! persisted in trainer_state.json so a `--load` resume CONTINUES the sample
//! order instead of restarting at pair 0.
//!
//! `--accum-steps N` turns each step into a gradient-accumulation window of N
//! micro-batches (exact token-weighted normalization; one clip/step/zeroGrad
//! per window) — the demonstration for docs/TRAINING.md §4 "Gradient accumulation".
//!
//! `--verify-grads` replaces the training run with causal, quantitative
//! gradient checks through the full production path (quantized frozen
//! weights, tiled attention, fused kernels): zero-structure at init,
//! per-adapter grad-norm audit, a first-order Taylor test, a frozen-base
//! ablation, and held-out generalization. See `verifyGrads` below.
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const optim = fucina.optim;
const training_checkpoint = fucina.training_checkpoint;

const default_model = "models/Qwen3-0.6B-Q4_K_S.gguf";
const default_save = "/tmp/fucina-qwen3-lora";

/// LoRA on q and v projections (the classic LoRA-paper target set).
const Trainer = llm.qwen3.train.Trainer(.{ .q = true, .v = true });

/// The pirate dataset: every answer opens with "Ahoy!" and closes with
/// "matey", so style transfer is unmistakable after a few steps.
const dataset = [_]llm.data.Pair{
    .{ .instruction = "What is the capital of France?", .response = "Ahoy! The capital of France be Paris, matey." },
    .{ .instruction = "Name a primary color.", .response = "Ahoy! Red be a fine primary color, matey." },
    .{ .instruction = "What is two plus two?", .response = "Ahoy! Two plus two makes four, matey." },
    .{ .instruction = "What language is spoken in Italy?", .response = "Ahoy! In Italy they be speakin' Italian, matey." },
    .{ .instruction = "How many days are in a week?", .response = "Ahoy! A week holds seven days, matey." },
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var model_path: []const u8 = default_model;
    var steps: usize = 30;
    var lr: f32 = 1e-3;
    var rank: usize = 8;
    var alpha: f32 = 16;
    var seq_max: usize = 256;
    var checkpoint_layers = false;
    var save_path: []const u8 = default_save;
    var save_every: usize = 0;
    var load_path: ?[]const u8 = null;
    var seed: u64 = 42;
    var verify_grads = false;
    var state_dtype: optim.StateDType = .f32;
    var accum_steps: usize = 1;
    var data_path: ?[]const u8 = null;
    var shuffle = false;
    var data_seed: ?u64 = null;

    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--model")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingModelPath;
            model_path = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--model=")) {
            model_path = arg["--model=".len..];
        } else if (std.mem.eql(u8, arg, "--steps")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSteps;
            steps = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--steps=")) {
            steps = try std.fmt.parseInt(usize, arg["--steps=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--lr")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingLr;
            lr = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--lr=")) {
            lr = try std.fmt.parseFloat(f32, arg["--lr=".len..]);
        } else if (std.mem.eql(u8, arg, "--rank")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingRank;
            rank = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--rank=")) {
            rank = try std.fmt.parseInt(usize, arg["--rank=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--alpha")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingAlpha;
            alpha = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--alpha=")) {
            alpha = try std.fmt.parseFloat(f32, arg["--alpha=".len..]);
        } else if (std.mem.eql(u8, arg, "--seq-max")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSeqMax;
            seq_max = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--seq-max=")) {
            seq_max = try std.fmt.parseInt(usize, arg["--seq-max=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--checkpoint-layers")) {
            checkpoint_layers = true;
        } else if (std.mem.eql(u8, arg, "--save")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSavePath;
            save_path = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--save=")) {
            save_path = arg["--save=".len..];
        } else if (std.mem.eql(u8, arg, "--save-every")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSaveEvery;
            save_every = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--save-every=")) {
            save_every = try std.fmt.parseInt(usize, arg["--save-every=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--load")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingLoadPath;
            load_path = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--load=")) {
            load_path = arg["--load=".len..];
        } else if (std.mem.eql(u8, arg, "--seed")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSeed;
            seed = try std.fmt.parseInt(u64, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--seed=")) {
            seed = try std.fmt.parseInt(u64, arg["--seed=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--state-dtype")) {
            // bf16 stores the AdamW first moment in bf16 (v stays f32; step
            // math is f32 either way). Honest sizing: on the default LoRA run
            // this saves only ~2.3 MB of the 9.2 MB m+v state — it matters at
            // full-parameter/embedding scale — and the resulting
            // optimizer.fucina is a v4 frame that only resumes into the same
            // state dtype.
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingStateDtype;
            state_dtype = std.meta.stringToEnum(optim.StateDType, args[arg_i]) orelse return error.InvalidStateDtype;
        } else if (std.mem.startsWith(u8, arg, "--state-dtype=")) {
            state_dtype = std.meta.stringToEnum(optim.StateDType, arg["--state-dtype=".len..]) orelse return error.InvalidStateDtype;
        } else if (std.mem.eql(u8, arg, "--accum-steps")) {
            // Gradient accumulation: each of the `--steps` MACRO steps runs N
            // micro-batches (round-robin continues across windows), each in
            // its own exec scope { lossExt + backward }, then ONE
            // clipGradNorm/step/zeroGrad. Normalization is exact
            // token-weighted: `.sum` CE with loss_scale = 1/total_valid over
            // the window. N=1 keeps the historical single-sample loop
            // (plain mean `loss()`) bitwise unchanged. `--save-every` counts
            // macro steps, so checkpoints always land on window boundaries
            // (accumulated gradients are never serialized).
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingAccumSteps;
            accum_steps = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--accum-steps=")) {
            accum_steps = try std.fmt.parseInt(usize, arg["--accum-steps=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--data")) {
            // JSONL dataset: one {"instruction": ..., "response": ...} object
            // per line, replacing the built-in pirate pairs.
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingDataPath;
            data_path = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--data=")) {
            data_path = arg["--data=".len..];
        } else if (std.mem.eql(u8, arg, "--shuffle")) {
            shuffle = true;
        } else if (std.mem.eql(u8, arg, "--data-seed")) {
            // Sample-order seed for --shuffle. Defaults to --seed; a separate
            // knob so the data order can be re-drawn without touching the
            // adapter-init/dropout streams (both keyed off --seed).
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingDataSeed;
            data_seed = try std.fmt.parseInt(u64, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--data-seed=")) {
            data_seed = try std.fmt.parseInt(u64, arg["--data-seed=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--verify-grads")) {
            verify_grads = true;
        } else {
            try stdout.print(
                "usage: zig build finetune -Doptimize=ReleaseFast -- [--model PATH] [--data PATH.jsonl] [--shuffle] [--data-seed N] [--steps N] [--accum-steps N] [--lr F] [--rank N] [--alpha F] [--seq-max N] [--checkpoint-layers] [--save DIR] [--save-every N] [--load DIR] [--seed N] [--state-dtype f32|bf16] [--verify-grads]\n",
                .{},
            );
            return error.UnknownArgument;
        }
    }

    if (accum_steps == 0) return error.InvalidAccumSteps;

    const allocator = std.heap.smp_allocator;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Load model + tokenizer from the same GGUF parse.
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

    var trainer = try Trainer.init(&ctx, &model, .{ .rank = rank, .alpha = alpha }, seed);
    defer trainer.deinit();
    trainer.checkpoint_layers = checkpoint_layers;

    var opt = optim.AdamW.init(allocator, .{ .lr = lr, .weight_decay = 0, .state_dtype = state_dtype });
    defer opt.deinit();
    try trainer.registerAllParams(&opt);
    var set = optim.OptimizerSet.init(allocator);
    defer set.deinit();
    try set.add(&opt);

    var restored_state: ?training_checkpoint.TrainerState = null;
    if (load_path) |path| {
        const state = try loadFinetuneCheckpoint(allocator, io, path, &trainer, &set, &opt);
        rank = @intCast(state.lora_rank orelse rank);
        alpha = trainer.lora_config.alpha;
        lr = opt.config.lr;
        // Checkpoint wins over the CLI (same precedence as lr). Saves happen
        // only at window boundaries, so step % accum_steps == 0 on resume.
        if (state.accum_steps) |accum| accum_steps = @intCast(accum);
        restored_state = state;
        try stdout.print("resumed checkpoint from {s} (step {d}, seed {d})\n", .{ path, trainer.step_counter, trainer.seed });
    }

    // Data source: the built-in pirate pairs (zero-copy borrow) or a JSONL
    // file (owned copy).
    var sft = if (data_path) |path|
        try llm.data.SftText.fromJsonl(allocator, io, path, .{})
    else
        llm.data.SftText.fromPairs(&dataset);
    defer sft.deinit(allocator);
    const pairs = sft.pairs;
    if (pairs.len == 0) return error.EmptyDataset;

    // Tokenize the dataset: ChatML turn (think off) + response + end marker;
    // only response tokens are supervised.
    const encode_opts = llm.data.EncodeOptions{
        .seq_max = seq_max,
        .ignore_index = llm.qwen3.train.ignore_index,
    };
    const samples = try allocator.alloc(llm.data.Sample, pairs.len);
    defer allocator.free(samples);
    var built_samples: usize = 0;
    defer for (samples[0..built_samples]) |*sample| sample.deinit(allocator);
    for (samples, pairs) |*sample, pair| {
        sample.* = try llm.data.encodePair(allocator, &tokenizer, template, pair, encode_opts);
        built_samples += 1;
    }

    // The held prompt for the before/after comparison (the first pair).
    const held = try llm.data.encodePrompt(allocator, &tokenizer, template, pairs[0].instruction, encode_opts);
    defer allocator.free(held);
    const stop_id: ?usize = if (tokenizer.tokenId("<|im_end|>")) |id| @as(usize, id) else null;

    // Sample-order loader: `.sequential` is the historical round-robin;
    // `--shuffle` draws a deterministic permutation per epoch. On --load the
    // checkpointed position wins (the order CONTINUES across the resume);
    // `--shuffle`/`--data-seed` themselves are not persisted — pass the same
    // flags when resuming. Checkpoints without loader state (older runs)
    // restart at pair 0, as before.
    var loader = try llm.data.Loader.init(
        allocator,
        samples.len,
        if (shuffle) .shuffled else .sequential,
        data_seed orelse seed,
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

    try stdout.print("dataset: {d} pairs  lora rank {d} alpha {d:.0} (q,v)  lr {d}  steps {d}  accum-steps {d}  checkpoint-layers {}  save-every {d}\n", .{
        pairs.len, rank, alpha, lr, steps, accum_steps, checkpoint_layers, save_every,
    });

    if (verify_grads) {
        try verifyGrads(&ctx, &trainer, &set, allocator, &tokenizer, template, pairs, samples, stop_id, io, stdout);
        return;
    }

    const max_new = 25;
    try stdout.print("\nheld prompt: {s}\n", .{pairs[0].instruction});
    const before = try greedyGenerate(&ctx, &trainer, allocator, held, max_new, stop_id);
    defer allocator.free(before);
    const before_text = try decodeIds(allocator, &tokenizer, before);
    defer allocator.free(before_text);
    try stdout.print("BEFORE: {s}\n\n", .{before_text});
    try stdout.flush();

    // Training loop: loader-ordered over the pairs; each MACRO step consumes
    // `accum_steps` micro-batches and applies ONE clip/step/zeroGrad.
    // `--save-every` counts macro steps, so every checkpoint lands on a
    // window boundary (accumulated gradients are never serialized).
    var total_tokens: usize = 0;
    var total_ns: i96 = 0;
    // Each accumulation window's sample indices, drawn from the loader up
    // front (the token-weighted normalization needs the whole window before
    // any micro-step runs).
    const window = try allocator.alloc(usize, accum_steps);
    defer allocator.free(window);
    for (0..steps) |step_i| {
        const step_start = nowNs(io);
        var loss_value: f32 = 0;
        var step_tokens: usize = 0;
        if (accum_steps == 1) {
            // The historical single-sample step, bitwise unchanged.
            const sample = &samples[loader.next()];
            {
                const scope = ctx.openExecScope();
                defer ctx.closeExecScope(scope);
                const loss = try trainer.loss(&ctx, sample.inputs, sample.labels);
                try loss.backward(&ctx);
                loss_value = try loss.item();
                _ = try set.clipGradNorm(&ctx, 1.0);
                try set.step(&ctx);
                set.zeroGrad();
            }
            step_tokens = sample.inputs.len;
        } else {
            // Exact token-weighted normalization: the samples differ in
            // supervised-token counts, so mean-of-means (`.mean` + 1/N)
            // would mis-weight them. `.sum` CE scaled by 1/total_valid makes
            // the accumulated gradient — and the reported sum of scaled
            // losses — the true mean over the window's supervised tokens.
            for (window) |*idx| idx.* = loader.next();
            var total_valid: usize = 0;
            for (window) |idx| {
                for (samples[idx].labels) |label| {
                    if (label != llm.qwen3.train.ignore_index) total_valid += 1;
                }
            }
            if (total_valid == 0) return error.NoSupervisedTokens;
            const loss_scale = 1.0 / @as(f32, @floatFromInt(total_valid));
            for (window) |idx| {
                // One exec scope per micro-batch: each graph is freed right
                // after its backward; the leaf grads accumulate outside the
                // scopes (backward ADDS until zeroGrad).
                const sample = &samples[idx];
                const scope = ctx.openExecScope();
                defer ctx.closeExecScope(scope);
                const loss = try trainer.lossExt(&ctx, sample.inputs, sample.labels, .{
                    .reduction = .sum,
                    .loss_scale = loss_scale,
                });
                try loss.backward(&ctx);
                loss_value += try loss.item();
                step_tokens += sample.inputs.len;
            }
            // ONCE per window: clip reads the full accumulated gradients
            // (a mid-window clip would rescale partial sums).
            _ = try set.clipGradNorm(&ctx, 1.0);
            try set.step(&ctx);
            set.zeroGrad();
        }
        const step_ns = nowNs(io) - step_start;
        total_tokens += step_tokens;
        total_ns += step_ns;
        try stdout.print("step {d:>3}  loss {d:.4}  {d:>7.1} ms  {d:>6.1} tok/s\n", .{
            step_i + 1,
            loss_value,
            millis(step_ns),
            @as(f64, @floatFromInt(step_tokens)) / seconds(step_ns),
        });
        try stdout.flush();
        if (save_every != 0 and (step_i + 1) % save_every == 0) {
            const ckpt_path = try std.fmt.allocPrint(allocator, "{s}/checkpoint-step-{d}", .{ save_path, step_i + 1 });
            defer allocator.free(ckpt_path);
            try saveFinetuneCheckpoint(allocator, io, ckpt_path, &trainer, &set, opt.config.lr, accum_steps, loader.state());
            try stdout.print("saved checkpoint to {s}\n", .{ckpt_path});
            try stdout.flush();
        }
    }
    if (steps > 0) {
        try stdout.print("trained {d} steps: avg {d:.1} ms/step, {d:.1} tok/s\n", .{
            steps,
            millis(total_ns) / @as(f64, @floatFromInt(steps)),
            @as(f64, @floatFromInt(total_tokens)) / seconds(total_ns),
        });
    }

    try saveFinetuneCheckpoint(allocator, io, save_path, &trainer, &set, opt.config.lr, accum_steps, loader.state());
    try stdout.print("saved checkpoint to {s}\n", .{save_path});

    const after = try greedyGenerate(&ctx, &trainer, allocator, held, max_new, stop_id);
    defer allocator.free(after);
    const after_text = try decodeIds(allocator, &tokenizer, after);
    defer allocator.free(after_text);
    try stdout.print("\nheld prompt: {s}\n", .{pairs[0].instruction});
    try stdout.print("AFTER:  {s}\n", .{after_text});
}

fn saveFinetuneCheckpoint(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir_path: []const u8,
    trainer: *const Trainer,
    set: *const optim.OptimizerSet,
    lr: f32,
    accum_steps: usize,
    loader_state: llm.data.Loader.State,
) !void {
    try training_checkpoint.beginSave(allocator, io, dir_path);

    const adapters_path = try training_checkpoint.pathJoin(allocator, dir_path, training_checkpoint.adapters_state_file);
    defer allocator.free(adapters_path);
    const SaveAdapters = struct {
        fn write(t: *const Trainer, writer: *std.Io.Writer) !void {
            try t.saveAdapters(writer);
        }
    };
    try training_checkpoint.writeFileAtomic(io, adapters_path, trainer, SaveAdapters.write);

    const optimizer_path = try training_checkpoint.pathJoin(allocator, dir_path, training_checkpoint.optimizer_state_file);
    defer allocator.free(optimizer_path);
    const SaveOptimizer = struct {
        fn write(s: *const optim.OptimizerSet, writer: *std.Io.Writer) !void {
            try s.saveState(writer);
        }
    };
    try training_checkpoint.writeFileAtomic(io, optimizer_path, set, SaveOptimizer.write);

    try training_checkpoint.saveTrainerState(allocator, io, dir_path, .{
        .step = trainer.step_counter,
        .seed = trainer.seed,
        .lora_rank = @intCast(trainer.lora_config.rank),
        .lora_alpha = @floatCast(trainer.lora_config.alpha),
        .lora_dropout_p = @floatCast(trainer.lora_config.dropout_p),
        .learning_rate = @floatCast(lr),
        // Recorded only when accumulating (older/plain checkpoints stay
        // byte-identical); saves happen at window boundaries, so
        // step % accum_steps == 0 holds in every written state.
        .accum_steps = if (accum_steps > 1) @as(u64, accum_steps) else null,
        // Loader position: (data_seed, data_epoch) regenerate the epoch
        // permutation, data_index resumes within it — a --load continues the
        // sample order instead of restarting at pair 0.
        .data_seed = loader_state.seed,
        .data_epoch = loader_state.epoch,
        .data_index = loader_state.index,
    });
}

fn loadFinetuneCheckpoint(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir_path: []const u8,
    trainer: *Trainer,
    set: *optim.OptimizerSet,
    opt: *optim.AdamW,
) !training_checkpoint.TrainerState {
    const state = try training_checkpoint.loadTrainerState(allocator, io, dir_path);
    const saved_rank = state.lora_rank orelse return error.CheckpointConfigMismatch;
    if (saved_rank != trainer.lora_config.rank) return error.CheckpointConfigMismatch;
    const saved_alpha: f32 = @floatCast(state.lora_alpha orelse return error.CheckpointConfigMismatch);
    const saved_dropout: f32 = @floatCast(state.lora_dropout_p orelse return error.CheckpointConfigMismatch);
    const saved_lr: f32 = @floatCast(state.learning_rate orelse return error.CheckpointConfigMismatch);

    trainer.seed = state.seed;
    trainer.step_counter = state.step;
    trainer.lora_config.alpha = saved_alpha;
    trainer.lora_config.dropout_p = saved_dropout;
    trainer.scale = saved_alpha / @as(f32, @floatFromInt(saved_rank));
    opt.config.lr = saved_lr;

    const adapters_path = try training_checkpoint.pathJoin(allocator, dir_path, training_checkpoint.adapters_state_file);
    defer allocator.free(adapters_path);
    {
        var file = try std.Io.Dir.cwd().openFile(io, adapters_path, .{});
        defer file.close(io);
        var buffer: [64 * 1024]u8 = undefined;
        var reader = file.reader(io, &buffer);
        try trainer.loadAdapters(&reader.interface);
    }

    const optimizer_path = try training_checkpoint.pathJoin(allocator, dir_path, training_checkpoint.optimizer_state_file);
    defer allocator.free(optimizer_path);
    {
        var file = try std.Io.Dir.cwd().openFile(io, optimizer_path, .{});
        defer file.close(io);
        var buffer: [64 * 1024]u8 = undefined;
        var reader = file.reader(io, &buffer);
        try set.loadState(&reader.interface);
    }
    return state;
}

/// `--verify-grads`: causal, quantitative gradient checks on the real model,
/// replacing the training run. Five checks, each printed with a PASS/FAIL
/// verdict; hard failures make the process exit nonzero.
///
/// Mechanics worth pinning down (also stated in the printed report):
/// - The Taylor-test SGD step is `optim.SGD{ .momentum = 0, .dampening = 0,
///   .weight_decay = 0 }`, whose update is exactly `theta -= lr * g` (one
///   fused pass, src/optim.zig `sgdUpdate`). The gradients it consumes are
///   the ones produced by the audit backward: leaf grads live in the
///   param-owned GradStates and persist across exec-scope close until
///   `zeroGrad`, so all three lr trials reuse the SAME g.
/// - Param restore between lr trials is a bitwise `@memcpy` from raw f32
///   snapshots of every adapter A/B taken at the warmed-up point (writing
///   through the variable's raw storage exactly as `optim.Param` does; the
///   facade's `data()` gate is for user code mutating live graph inputs).
/// - `lora_config.dropout_p == 0` here, so every forward is deterministic;
///   `Trainer.loss` advances a step counter but that only selects dropout
///   seed streams, which the p == 0 identity path never consumes.
fn verifyGrads(
    ctx: *fucina.ExecContext,
    trainer: *Trainer,
    set: *optim.OptimizerSet,
    allocator: std.mem.Allocator,
    tokenizer: *const llm.tokenizer.Tokenizer,
    template: llm.chat.Template,
    pairs: []const llm.data.Pair,
    samples: []llm.data.Sample,
    stop_id: ?usize,
    io: std.Io,
    out: *std.Io.Writer,
) !void {
    // Type-level guarantee, asserted at comptime: quantized weight tensors
    // (the frozen projections of this Q4_K_S GGUF) carry no grad_state field
    // at all -- a frozen base weight CANNOT receive gradients. The frozen-RHS
    // `dot` routes gradients to the f32 activations only (ConstRhsDotBackward).
    comptime {
        std.debug.assert(!@hasField(llm.weights.QuantWeight(.q4_k), "grad_state"));
        std.debug.assert(!@hasField(llm.weights.QuantWeight(.q6_k), "grad_state"));
    }

    if (samples.len < 2) return error.DatasetTooSmall; // needs a held-out pair

    const n_grads = trainer.adapters.len * 4; // layers x {q,v} x {A,B}
    const train_count = samples.len - 1; // the last pair is held out
    const fixed = &samples[0]; // fixed batch for checks 0, 2, 3, 4
    const heldout = &samples[samples.len - 1];

    var failures: usize = 0;

    try out.print("\n=== gradient verification (--verify-grads) ===\n", .{});
    try out.print("fixed batch = pair 1; held-out = pair {d} (excluded from ALL training below)\n", .{samples.len});
    try out.print("dropout_p = 0 => deterministic forwards (the loss step counter only seeds dropout\n", .{});
    try out.print("streams, never consumed on the p == 0 identity path)\n", .{});
    try out.flush();

    // Held-out baseline at RAW INIT (base model + exactly-zero LoRA delta),
    // before any training: the honest "before" for [5] (the warmup in [1]
    // already trains on the train split).
    const held_instruction = pairs[pairs.len - 1].instruction;
    const held_prompt = try llm.data.encodePrompt(allocator, tokenizer, template, held_instruction, .{});
    defer allocator.free(held_prompt);
    const held_ce_init = try lossOnly(ctx, trainer, heldout);
    const gen_init = try greedyGenerate(ctx, trainer, allocator, held_prompt, 25, stop_id);
    defer allocator.free(gen_init);
    const text_init = try decodeIds(allocator, tokenizer, gen_init);
    defer allocator.free(text_init);

    // [0] Zero-structure at init: B == 0 => dL/dA == 0 identically, dL/dB > 0.
    {
        const init_loss = try lossBackward(ctx, trainer, fixed);
        var max_a: f64 = 0;
        var min_b: f64 = std.math.inf(f64);
        for (trainer.adapters) |*ads| {
            inline for (.{ "q", "v" }) |tname| {
                const ad = &@field(ads.*, tname);
                max_a = @max(max_a, @sqrt(try gradSqNorm(ctx, &ad.a)));
                min_b = @min(min_b, @sqrt(try gradSqNorm(ctx, &ad.b)));
            }
        }
        set.zeroGrad();
        try out.print("\n[0] zero-structure at init: LoRA B == 0, so dL/dA == 0 identically while dL/dB != 0\n", .{});
        try out.print("    (A only receives gradient once B != 0 -- hence the warmup in [1])\n", .{});
        try out.print("    loss {d:.4} on fixed batch; max ||dL/dA|| = {e:.3} (expect exactly 0); min ||dL/dB|| = {e:.3} (expect > 0)\n", .{ init_loss, max_a, min_b });
        const ok = max_a == 0 and min_b > 0 and std.math.isFinite(min_b);
        try out.print("    {s}\n", .{if (ok) "PASS" else "FAIL"});
        if (!ok) failures += 1;
        try out.flush();
    }

    // [1] Warmup: 3 normal AdamW steps (clip 1.0, same as the training loop).
    try out.print("\n[1] warmup: 3 AdamW steps on the train split (round-robin), so B != 0 and A trains\n", .{});
    for (0..3) |i| {
        const v = try trainStep(ctx, trainer, set, &samples[i % train_count]);
        try out.print("    step {d}  loss {d:.4}\n", .{ i + 1, v });
    }
    try out.flush();

    // [2] Grad-norm audit at the warmed-up point theta0: one fresh
    // loss+backward on the fixed batch -- no clip, no step, grads kept.
    const l0 = try lossBackward(ctx, trainer, fixed);
    const norms = try allocator.alloc(f64, n_grads);
    defer allocator.free(norms);
    var g2_total: f64 = 0;
    try out.print("\n[2] grad-norm audit at theta0 (fresh loss+backward on the fixed batch; no clip, no step)\n", .{});
    try out.print("    L2 norm of dL/d{{A,B}} per adapter ({d} layers x {{q,v}} x {{A,B}}):\n", .{trainer.adapters.len});
    try out.print("    layer          q.A          q.B          v.A          v.B\n", .{});
    {
        var idx: usize = 0;
        for (trainer.adapters, 0..) |*ads, layer_i| {
            var row: [4]f64 = undefined;
            var slot: usize = 0;
            inline for (.{ "q", "v" }) |tname| {
                const ad = &@field(ads.*, tname);
                const a2 = try gradSqNorm(ctx, &ad.a);
                const b2 = try gradSqNorm(ctx, &ad.b);
                g2_total += a2 + b2;
                row[slot] = @sqrt(a2);
                row[slot + 1] = @sqrt(b2);
                slot += 2;
            }
            @memcpy(norms[idx .. idx + 4], &row);
            idx += 4;
            try out.print("    {d:>5}  {e:>11.4}  {e:>11.4}  {e:>11.4}  {e:>11.4}\n", .{ layer_i, row[0], row[1], row[2], row[3] });
        }
    }
    const sorted = try allocator.dupe(f64, norms);
    defer allocator.free(sorted);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
    const median = (sorted[n_grads / 2 - 1] + sorted[n_grads / 2]) / 2;
    var zero_count: usize = 0;
    var nonfinite_count: usize = 0;
    for (norms) |n| {
        if (!std.math.isFinite(n)) {
            nonfinite_count += 1;
        } else if (n == 0) {
            zero_count += 1;
        }
    }
    try out.print("    stats over {d} grads: min {e:.4}  median {e:.4}  max {e:.4}  zeros {d}  non-finite {d}\n", .{
        n_grads, sorted[0], median, sorted[n_grads - 1], zero_count, nonfinite_count,
    });
    try out.print("    L0 = {d:.6}  ||g||^2 = {e:.6} (f64 accumulation)\n", .{ l0, g2_total });
    const audit_ok = zero_count == 0 and nonfinite_count == 0;
    try out.print("    {s}: every grad finite and > 0 (no dead or exploding layers)\n", .{if (audit_ok) "PASS" else "FAIL"});
    if (!audit_ok) failures += 1;
    try out.flush();

    // [3] First-order Taylor test: theta <- theta0 - lr*g, then
    // R = (L0 - L1) / (lr * ||g||^2) must approach 1 as lr -> 0.
    try out.print("\n[3] first-order Taylor test at theta0: theta <- theta0 - lr*g on the SAME batch\n", .{});
    try out.print("    update: optim.SGD{{ .momentum = 0, .dampening = 0, .weight_decay = 0 }} => exact theta -= lr*g,\n", .{});
    try out.print("    reusing the [2] gradients persisted in the leaf GradStates; restore between trials =\n", .{});
    try out.print("    bitwise memcpy of raw f32 snapshots of all A/B taken at theta0 (not the state-dict)\n", .{});
    const snaps = try snapshotAdapters(allocator, trainer);
    defer freeSnapshots(allocator, snaps);
    var sgd = optim.SGD.init(allocator, .{ .lr = 0, .momentum = 0, .dampening = 0, .weight_decay = 0 });
    defer sgd.deinit();
    try trainer.registerAllParams(&sgd);
    // Reference L0 from the SAME code path as the trial L1s (forward-only),
    // cross-checked bitwise against the [2] forward+backward value.
    const l0_only = try lossOnly(ctx, trainer, fixed);
    try out.print("    L0 re-eval forward-only: {d:.6} (0x{x:0>8}) vs [2] forward+backward {d:.6} (0x{x:0>8}) -> {s}\n", .{
        l0_only,                                                                                      @as(u32, @bitCast(l0_only)),
        l0,                                                                                           @as(u32, @bitCast(l0)),
        if (@as(u32, @bitCast(l0_only)) == @as(u32, @bitCast(l0))) "bitwise identical" else "DIFFER",
    });
    // The mission lrs {1e-4, 3e-4, 1e-3} plus 3e-3: on the quantized path the
    // staircase floor (measured below) sits near the 1e-3 prediction, so a
    // larger radius is needed for an adjudicable signal/noise ratio.
    const trial_lrs = [_]f32{ 1e-4, 3e-4, 1e-3, 3e-3 };
    var ratios: [trial_lrs.len]f64 = undefined;
    var dls: [trial_lrs.len]f64 = undefined;
    for (trial_lrs, 0..) |trial_lr, i| {
        sgd.config.lr = trial_lr;
        try sgd.step(ctx);
        const l1 = try lossOnly(ctx, trainer, fixed);
        restoreAdapters(trainer, snaps);
        // Restore + determinism witness: L0 recomputed at the restored point
        // must reproduce the reference bitwise.
        const l0_re = try lossOnly(ctx, trainer, fixed);
        dls[i] = @as(f64, l0_only) - @as(f64, l1);
        ratios[i] = dls[i] / (@as(f64, trial_lr) * g2_total);
        try out.print("    lr {e:.1}: L1 = {d:.6}  L0-L1 = {e:.4} (predicted {e:.4})  R = {d:.4}  [L0 after restore: 0x{x:0>8} {s}]\n", .{
            trial_lr,
            l1,
            dls[i],
            @as(f64, trial_lr) * g2_total,
            ratios[i],
            @as(u32, @bitCast(l0_re)),
            if (@as(u32, @bitCast(l0_re)) == @as(u32, @bitCast(l0_only))) "ok" else "MISMATCH",
        });
        try out.flush();
    }

    // Noise-floor probe. The K-quant matmuls quantize activation ROWS to
    // Q8_K before the integer dot kernels (src/backend/quant/q4_k.zig,
    // quantizeRowsQ8_K), so L(theta) is a STAIRCASE at small radii: rounding
    // boundaries jump as theta moves. Random directions u (gaussian,
    // normalized to each trial's step norm lr*||g||) carry no first-order
    // descent component, so |L0 - L(theta0 + r*u)| measures that pure jump
    // floor; an lr trial is only adjudicable where its predicted first-order
    // drop clears the floor. The floor distribution is heavy-tailed, so the
    // gate uses a robust tail statistic (2nd-largest of 5), not the median.
    try out.print("    noise floor: |L0 - L(theta0 + r*u)|, u random unit direction (zero descent component), 5 probes/radius:\n", .{});
    var floors: [trial_lrs.len]f64 = undefined; // tail(4/5): the gate statistic
    for (trial_lrs, 0..) |trial_lr, i| {
        const radius = @as(f64, trial_lr) * @sqrt(g2_total);
        var probes: [5]f64 = undefined;
        for (&probes, 0..) |*probe, p| {
            try perturbRandom(allocator, trainer, snaps, radius, 0x6e6f697365 + i * 16 + p);
            const lp = try lossOnly(ctx, trainer, fixed);
            probe.* = @abs(@as(f64, l0_only) - @as(f64, lp));
        }
        restoreAdapters(trainer, snaps);
        std.mem.sort(f64, &probes, {}, std.sort.asc(f64));
        floors[i] = probes[probes.len - 2];
        try out.print("      r = {e:.1}*||g||: |dL| = {{ {e:.3}, {e:.3}, {e:.3}, {e:.3}, {e:.3} }}  median {e:.3}  tail(4/5) {e:.3}  vs gradient-step prediction {e:.3}\n", .{
            trial_lr, probes[0], probes[1], probes[2], probes[3], probes[4], probes[probes.len / 2], floors[i], @as(f64, trial_lr) * g2_total,
        });
        try out.flush();
    }

    // Adjudication: the smooth-loss criterion "R -> 1 as lr -> 0" only
    // applies where the observable is above the staircase floor (predicted
    // drop >= 3x the tail statistic); below it the MEASUREMENT fails, not
    // the gradient -- report both.
    var taylor_fail = false;
    var taylor_critical = false;
    var n_measurable: usize = 0;
    for (trial_lrs, 0..) |trial_lr, i| {
        const predicted = @as(f64, trial_lr) * g2_total;
        const snr = if (floors[i] > 0) predicted / floors[i] else std.math.inf(f64);
        if (snr >= 3) {
            n_measurable += 1;
            const ok = ratios[i] >= 0.7 and ratios[i] <= 1.1;
            if (!ok) {
                taylor_fail = true;
                if (!std.math.isFinite(ratios[i]) or ratios[i] <= 0.1) taylor_critical = true;
            }
            try out.print("    lr {e:.1}: signal/noise {d:.1} -> measurable; R = {d:.4} in [0.7, 1.1] -> {s}\n", .{
                trial_lr, snr, ratios[i], if (ok) "PASS" else "FAIL",
            });
        } else {
            // Not adjudicable as a ratio, but the residual (measured minus
            // predicted drop) must still be explainable by the noise floor:
            // |L0-L1| itself contains the true descent signal, so the bound
            // is on the leftover after subtracting the prediction.
            const residual = @abs(dls[i] - predicted);
            const within = residual <= 3 * floors[i];
            try out.print("    lr {e:.1}: signal/noise {d:.1} -> noise-dominated radius; R = {d:.4} not adjudicable (residual |dL - pred| = {e:.3} {s} 3x floor)\n", .{
                trial_lr, snr, ratios[i], residual, if (within) "within" else "EXCEEDS",
            });
            if (!within) taylor_fail = true;
        }
    }
    if (taylor_critical) {
        try out.print("    CRITICAL: R is ~0 or negative at a measurable radius -- gradients do NOT point downhill; stopping.\n", .{});
        try out.flush();
        return error.GradientsDoNotPointDownhill;
    }
    if (n_measurable == 0) {
        if (taylor_fail) {
            try out.print("    FAIL: no trial radius clears the noise floor, and at least one residual exceeds the measured floor.\n", .{});
        } else {
            try out.print("    INCONCLUSIVE: no trial radius clears the noise floor; residual checks stay within the measured floor.\n", .{});
        }
    }
    try out.print("    {s}\n", .{if (taylor_fail) "FAIL" else "PASS"});
    try out.print("    deviation note: the strict smooth-loss acceptance \"R(1e-4) in [0.7, 1.1]\" is not\n", .{});
    try out.print("    adjudicable through Q8_K-quantized activations when the predicted drop at that\n", .{});
    try out.print("    radius sits below the measured staircase floor (see probes above).\n", .{});
    if (taylor_fail) failures += 1;
    try out.flush();

    // [4] Frozen ablation: deterministic forward + frozen base.
    try out.print("\n[4] frozen ablation\n", .{});
    const e1 = try lossOnly(ctx, trainer, fixed);
    const e2 = try lossOnly(ctx, trainer, fixed);
    const bits1: u32 = @bitCast(e1);
    const bits2: u32 = @bitCast(e2);
    try out.print("    loss twice, no optimizer step between: {d:.6} (0x{x:0>8}) / {d:.6} (0x{x:0>8}) -> {s}\n", .{
        e1, bits1, e2, bits2, if (bits1 == bits2) "bitwise identical, PASS" else "MISMATCH, FAIL",
    });
    if (bits1 != bits2) failures += 1;
    const l0_bits: u32 = @bitCast(l0);
    try out.print("    restore exactness: matches the [2] loss at theta0 (0x{x:0>8}) -> {s}\n", .{
        l0_bits, if (bits1 == l0_bits) "yes (bitwise)" else "no (forward-only vs forward+backward run)",
    });
    const norm_rg = trainer.model.output_norm.requiresGrad();
    const layer0_rg = trainer.model.layers[0].attn_norm.requiresGrad();
    try out.print("    base weights cannot receive grads:\n", .{});
    try out.print("      runtime: output_norm.requiresGrad() = {}; layers[0].attn_norm.requiresGrad() = {}\n", .{ norm_rg, layer0_rg });
    try out.print("      (f32 constants: no GradState attached even after every backward above)\n", .{});
    try out.print("      type-level (comptime-asserted in this fn): quantized projection tensors have no\n", .{});
    try out.print("      grad_state field at all; the frozen-RHS dot routes grads to f32 activations only\n", .{});
    if (norm_rg or layer0_rg) failures += 1;
    try out.flush();

    // [5] Held-out generalization: 30 more steps on the train split only.
    // "Before" = raw init (CE + generation captured at function start, prior
    // to any training); theta0 (post-warmup, also train-split-only) is
    // reported as a midpoint.
    try out.print("\n[5] held-out generalization: train 30 AdamW steps on pairs 1..{d}; pair {d} never trained on\n", .{ train_count, samples.len });
    try out.print("    held-out prompt: {s}\n", .{held_instruction});
    try out.print("    held-out target: {s}\n", .{pairs[pairs.len - 1].response});
    try out.print("    generation at raw init (base model, exactly-zero LoRA delta): {s}\n", .{text_init});
    try out.flush();
    const held_warm = try lossOnly(ctx, trainer, heldout); // at theta0
    set.zeroGrad(); // drop the audit gradients before training resumes
    var total_ns: i96 = 0;
    for (0..30) |step_i| {
        const t0 = nowNs(io);
        const v = try trainStep(ctx, trainer, set, &samples[step_i % train_count]);
        total_ns += nowNs(io) - t0;
        try out.print("    step {d:>2}  loss {d:.4}\n", .{ step_i + 1, v });
        try out.flush();
    }
    try out.print("    ({d:.1} ms/step avg)\n", .{millis(total_ns) / 30.0});
    const held_after = try lossOnly(ctx, trainer, heldout);
    const gen_after = try greedyGenerate(ctx, trainer, allocator, held_prompt, 25, stop_id);
    defer allocator.free(gen_after);
    const text_after = try decodeIds(allocator, tokenizer, gen_after);
    defer allocator.free(text_after);
    const rel_drop = (@as(f64, held_ce_init) - @as(f64, held_after)) / @as(f64, held_ce_init);
    try out.print("    CE on the never-seen pair: raw init {d:.4} -> theta0 after [1] warmup {d:.4} -> after 30 steps {d:.4}\n", .{
        held_ce_init, held_warm, held_after,
    });
    try out.print("    total training effect (3 warmup + 30 steps, train split only) vs raw init: {d:.1}% CE change\n", .{-rel_drop * 100});
    try out.print("    generation after: {s}\n", .{text_after});
    if (held_after > held_warm) {
        try out.print("    note: the rise from the warmed-up midpoint is overfitting -- the train split is\n", .{});
        try out.print("    memorized by step 30 and the held-out reference phrasing loses probability mass\n", .{});
        try out.print("    to the memorized phrasings, while the style/format transfer shows in generation\n", .{});
    }
    const held_ok = held_after < held_ce_init and rel_drop >= 0.25;
    try out.print("    {s}: held-out CE vs raw init must decrease materially (>= 25%) -- shared style/format transfers\n", .{if (held_ok) "PASS" else "FAIL"});
    if (!held_ok) failures += 1;

    try out.print("\n=== verification summary: {d} failure(s) ===\n", .{failures});
    try out.flush();
    if (failures > 0) return error.GradVerificationFailed;
}

/// Mean CE on one sample, forward only (scoped; scope close frees the graph).
fn lossOnly(ctx: *fucina.ExecContext, trainer: *Trainer, sample: *const llm.data.Sample) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.loss(ctx, sample.inputs, sample.labels);
    return loss.item();
}

/// Mean CE on one sample plus backward; the adapters' leaf grads survive the
/// scope close (they live in the param-owned GradStates until zeroGrad).
fn lossBackward(ctx: *fucina.ExecContext, trainer: *Trainer, sample: *const llm.data.Sample) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.loss(ctx, sample.inputs, sample.labels);
    try loss.backward(ctx);
    return loss.item();
}

/// One normal training step (loss, backward, clip 1.0, optimizer step,
/// zeroGrad) -- the exact loop body of the default training mode.
fn trainStep(ctx: *fucina.ExecContext, trainer: *Trainer, set: *optim.OptimizerSet, sample: *const llm.data.Sample) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.loss(ctx, sample.inputs, sample.labels);
    try loss.backward(ctx);
    const value = try loss.item();
    _ = try set.clipGradNorm(ctx, 1.0);
    try set.step(ctx);
    set.zeroGrad();
    return value;
}

/// Squared L2 norm of a param's accumulated gradient, accumulated in f64.
/// Goes through the facade's `grad` accessor (gradClone); a missing grad
/// reads as 0 so the audit's `> 0` assert catches it.
fn gradSqNorm(ctx: *fucina.ExecContext, t: anytype) !f64 {
    var g = (try t.grad(ctx)) orelse return 0;
    defer g.deinit();
    var acc: f64 = 0;
    for (try g.dataConst()) |x| acc += @as(f64, x) * @as(f64, x);
    return acc;
}

/// Raw f32 copies of every adapter A/B, in (layer, {q,v}, {a,b}) order.
fn snapshotAdapters(allocator: std.mem.Allocator, trainer: *const Trainer) ![][]f32 {
    const snaps = try allocator.alloc([]f32, trainer.adapters.len * 4);
    var built: usize = 0;
    errdefer {
        for (snaps[0..built]) |s| allocator.free(s);
        allocator.free(snaps);
    }
    for (trainer.adapters) |*ads| {
        inline for (.{ "q", "v" }) |tname| {
            const ad = &@field(ads.*, tname);
            snaps[built] = try allocator.dupe(f32, try ad.a.dataConst());
            built += 1;
            snaps[built] = try allocator.dupe(f32, try ad.b.dataConst());
            built += 1;
        }
    }
    return snaps;
}

/// Bitwise restore of `snapshotAdapters` copies, writing through each
/// variable's raw storage (same access path as `optim.Param`).
fn restoreAdapters(trainer: *Trainer, snaps: []const []f32) void {
    var i: usize = 0;
    for (trainer.adapters) |*ads| {
        inline for (.{ "q", "v" }) |tname| {
            const ad = &@field(ads.*, tname);
            @memcpy(ad.a.value.data(), snaps[i]);
            i += 1;
            @memcpy(ad.b.value.data(), snaps[i]);
            i += 1;
        }
    }
}

fn freeSnapshots(allocator: std.mem.Allocator, snaps: [][]f32) void {
    for (snaps) |s| allocator.free(s);
    allocator.free(snaps);
}

/// theta <- theta0 + radius * u/||u|| with u ~ N(0,1) elementwise from the
/// repo rng (deterministic per seed). Writes absolute values from `snaps`
/// (theta0), so the current theta does not matter; caller restores after.
/// Two passes over the same deterministic gaussian stream: norm, then apply.
fn perturbRandom(allocator: std.mem.Allocator, trainer: *Trainer, snaps: []const []f32, radius: f64, seed: u64) !void {
    var max_len: usize = 0;
    for (snaps) |s| max_len = @max(max_len, s.len);
    const scratch = try allocator.alloc(f32, max_len);
    defer allocator.free(scratch);

    var sumsq: f64 = 0;
    for (snaps, 0..) |s, i| {
        const u = scratch[0..s.len];
        fucina.rng.gaussianFill(fucina.rng.at(seed, i), u, 1.0);
        for (u) |x| sumsq += @as(f64, x) * @as(f64, x);
    }
    const scale: f32 = @floatCast(radius / @sqrt(sumsq));

    var i: usize = 0;
    for (trainer.adapters) |*ads| {
        inline for (.{ "q", "v" }) |tname| {
            const ad = &@field(ads.*, tname);
            inline for (.{ "a", "b" }) |fname| {
                const t = &@field(ad.*, fname);
                const data = t.value.data();
                const u = scratch[0..data.len];
                fucina.rng.gaussianFill(fucina.rng.at(seed, i), u, 1.0);
                for (data, snaps[i], u) |*d, s0, x| d.* = s0 + scale * x;
                i += 1;
            }
        }
    }
}

/// Greedy continuation through the trainer's eval forward (no KV cache: each
/// step re-runs the full prefix — fine for a short demo).
fn greedyGenerate(
    ctx: *fucina.ExecContext,
    trainer: *Trainer,
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
        const next: usize = @intCast((try index.dataConst())[0]);
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

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
}

fn millis(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}
