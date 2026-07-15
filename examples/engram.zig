//! Engram graft CLI (docs/ENGRAM.md): attach conditional n-gram memory to a
//! FROZEN qwen3 GGUF and train only the graft (optionally + LoRA).
//!
//!   zig build engram -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf --equiv
//!   zig build engram -Doptimize=ReleaseFast -- --model ... --corpus docs --train --steps 200 --save graft.safetensors
//!   zig build engram -Doptimize=ReleaseFast -- --model ... --corpus docs --eval --load graft.safetensors
//!
//! Modes:
//!  --equiv  gate: a zero-init graft (graft_zero_init) must leave the model
//!           BITWISE unchanged — greedy logits compared element-for-element.
//!  --train  continued-pretraining next-token CE over --corpus chunks; the
//!           trunk stays frozen (LoRA rank 0 arm = graft-only; --lora N adds
//!           trainable adapters). Held-out chunks are scored every
//!           --eval-every steps.
//!  --eval   held-out CE of arms: bare model vs loaded graft (--load).
//!
//! Probes are teacher-forced (held-out CE + next-token accuracy): the
//! go/no-go signal for the graft experiment, no serving integration needed.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const ExecContext = fucina.ExecContext;
const engram = llm.engram;
// LoRA arm targets q/v (the trainer default shape); rank 1 alpha 1 with
// q=v=false is the no-adapter trainer (cartridge.zig precedent).
const TrainerQV = llm.qwen3.train.Trainer(.{});
const TrainerNone = llm.qwen3.train.Trainer(.{ .q = false, .v = false });

const Options = struct {
    model_path: []const u8 = "models/Qwen3-0.6B-f16.gguf",
    corpus: []const u8 = "",
    equiv: bool = false,
    train: bool = false,
    eval_only: bool = false,
    steps: usize = 200,
    chunk: usize = 256,
    lr: f32 = 1e-3,
    lora: usize = 0,
    seed: u64 = 7,
    save_path: []const u8 = "",
    load_path: []const u8 = "",
    eval_every: usize = 25,
    eval_chunks: usize = 8,
    layers: []const u8 = "",
    table_vocab: usize = 100_000,
    n_embed: usize = 256,
    heads: usize = 4,
    /// Control arm: train/eval WITHOUT attaching the engram (clean
    /// LoRA-only baseline; the graft is still constructed but unused).
    no_engram: bool = false,
    gate_bias: f32 = 0,
    /// After training: score N verbatim-recall spans (32-token prefix,
    /// 16-token target) sampled from train chunks and N from held-out
    /// chunks, each arm (engram detached vs attached). Recall = the
    /// memory's actual job; CE on random text under-rewards it.
    probes: usize = 0,
};

fn parseFlagStr(args: []const []const u8, i: *usize, comptime name: []const u8) ?[]const u8 {
    if (!std.mem.eql(u8, args[i.*], name)) return null;
    if (i.* + 1 >= args.len) return null;
    i.* += 1;
    return args[i.*];
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const allocator = std.heap.smp_allocator;

    var opts = Options{};
    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        const a: []const u8 = args[arg_i];
        if (parseFlagStr(args, &arg_i, "--model")) |v| {
            opts.model_path = v;
        } else if (parseFlagStr(args, &arg_i, "--corpus")) |v| {
            opts.corpus = v;
        } else if (std.mem.eql(u8, a, "--equiv")) {
            opts.equiv = true;
        } else if (std.mem.eql(u8, a, "--train")) {
            opts.train = true;
        } else if (std.mem.eql(u8, a, "--eval")) {
            opts.eval_only = true;
        } else if (std.mem.eql(u8, a, "--no-engram")) {
            opts.no_engram = true;
        } else if (parseFlagStr(args, &arg_i, "--gate-bias")) |v| {
            opts.gate_bias = try std.fmt.parseFloat(f32, v);
        } else if (parseFlagStr(args, &arg_i, "--probes")) |v| {
            opts.probes = try std.fmt.parseInt(usize, v, 10);
        } else if (parseFlagStr(args, &arg_i, "--steps")) |v| {
            opts.steps = try std.fmt.parseInt(usize, v, 10);
        } else if (parseFlagStr(args, &arg_i, "--chunk")) |v| {
            opts.chunk = try std.fmt.parseInt(usize, v, 10);
        } else if (parseFlagStr(args, &arg_i, "--lr")) |v| {
            opts.lr = try std.fmt.parseFloat(f32, v);
        } else if (parseFlagStr(args, &arg_i, "--lora")) |v| {
            opts.lora = try std.fmt.parseInt(usize, v, 10);
        } else if (parseFlagStr(args, &arg_i, "--seed")) |v| {
            opts.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (parseFlagStr(args, &arg_i, "--save")) |v| {
            opts.save_path = v;
        } else if (parseFlagStr(args, &arg_i, "--load")) |v| {
            opts.load_path = v;
        } else if (parseFlagStr(args, &arg_i, "--eval-every")) |v| {
            opts.eval_every = try std.fmt.parseInt(usize, v, 10);
        } else if (parseFlagStr(args, &arg_i, "--eval-chunks")) |v| {
            opts.eval_chunks = try std.fmt.parseInt(usize, v, 10);
        } else if (parseFlagStr(args, &arg_i, "--layers")) |v| {
            opts.layers = v;
        } else if (parseFlagStr(args, &arg_i, "--table-vocab")) |v| {
            opts.table_vocab = try std.fmt.parseInt(usize, v, 10);
        } else if (parseFlagStr(args, &arg_i, "--n-embed")) |v| {
            opts.n_embed = try std.fmt.parseInt(usize, v, 10);
        } else if (parseFlagStr(args, &arg_i, "--heads")) |v| {
            opts.heads = try std.fmt.parseInt(usize, v, 10);
        } else {
            try stdout.print(
                "usage: zig build engram -Doptimize=ReleaseFast -- --model GGUF (--equiv | --train --corpus FILE|DIR [--steps N] [--lora R] [--save F] | --eval --corpus FILE|DIR --load F)\n" ++
                    "flags: --chunk N (256) --lr F (1e-3) --seed N (7) --eval-every N (25) --layers a,b --table-vocab N (100000) --n-embed N (256) --heads N (4)\n",
                .{},
            );
            return;
        }
    }

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = try fucina.gguf.File.loadMmap(allocator, io, opts.model_path);
    var model = try llm.qwen3.model.Model.loadGgufFromFile(&ctx, &file, try llm.qwen3.model.Config.fromGguf(&file));
    defer model.deinit();
    var tokenizer = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tokenizer.deinit();
    tokenizer_ptr = &tokenizer;
    file.deinit();
    const cfg = model.config;
    try stdout.print("model: {s} ({d} layers, hidden {d})\n", .{ opts.model_path, cfg.num_layers, cfg.hidden_size });

    // Engram geometry: default layers = {1, num_layers/2} (the reference
    // early+middle placement scaled to the model).
    var layer_ids_buf: [16]usize = undefined;
    var layer_ids: []const usize = layer_ids_buf[0..2];
    layer_ids_buf[0] = 1;
    layer_ids_buf[1] = cfg.num_layers / 2;
    if (opts.layers.len > 0) {
        var n: usize = 0;
        var it = std.mem.splitScalar(u8, opts.layers, ',');
        while (it.next()) |part| : (n += 1) layer_ids_buf[n] = try std.fmt.parseInt(usize, part, 10);
        layer_ids = layer_ids_buf[0..n];
    }
    const vocab_sizes = [_]usize{ opts.table_vocab, opts.table_vocab };
    const ecfg = engram.Config{
        .hidden_size = cfg.hidden_size,
        .hc_mult = 1,
        .max_ngram_size = 3,
        .n_embed_per_ngram = opts.n_embed,
        .n_head_per_ngram = opts.heads,
        .engram_vocab_size = &vocab_sizes,
        .kernel_size = 4,
        .pad_id = 0,
    };
    var graft = try engram.Engram.init(&ctx, allocator, ecfg, layer_ids, opts.seed, null, .{ .graft_zero_init = true, .gate_bias_init = opts.gate_bias });
    defer graft.deinit();
    var table_params: usize = 0;
    for (graft.plan.table_rows) |rows| table_params += rows * ecfg.headDim();
    try stdout.print("engram: layers {any}, {d} hash heads/layer, tables {d} rows total ({d:.1} M params)\n", .{
        layer_ids,                                    ecfg.headsPerLayer(),
        blk: {
            var total: usize = 0;
            for (graft.plan.table_rows) |rows| total += rows;
            break :blk total;
        },
        @as(f64, @floatFromInt(table_params)) / 1e6,
    });

    if (opts.load_path.len > 0) {
        var dir = std.Io.Dir.cwd();
        const bytes = try dir.readFileAlloc(io, opts.load_path, allocator, .limited(1024 * 1024 * 1024));
        defer allocator.free(bytes);
        var reader = std.Io.Reader.fixed(bytes);
        try graft.loadStateDict(&reader, .{});
        try stdout.print("loaded graft: {s}\n", .{opts.load_path});
    }

    if (opts.equiv) return runEquiv(&ctx, stdout, allocator, &model, &graft, opts);
    if (!opts.train and !opts.eval_only) {
        try stdout.print("nothing to do: pass --equiv, --train, or --eval\n", .{});
        return;
    }

    // Corpus: one flat token stream (files under --corpus, sorted .md/.txt
    // top level when a directory), split into train/held-out chunks.
    const ids = try loadCorpusIds(allocator, io, &tokenizer, opts.corpus);
    defer allocator.free(ids);
    const chunk = opts.chunk;
    const n_chunks = ids.len / chunk;
    if (n_chunks < 4) {
        try stdout.print("corpus too small: {d} tokens < 4 chunks of {d}\n", .{ ids.len, chunk });
        return;
    }
    // Every 8th chunk is held out.
    try stdout.print("corpus: {d} tokens, {d} chunks of {d} (held-out: every 8th)\n", .{ ids.len, n_chunks, chunk });

    if (opts.lora > 0) {
        var trainer = try TrainerQV.init(&ctx, &model, .{ .rank = @intCast(opts.lora), .alpha = @floatFromInt(2 * opts.lora) }, opts.seed);
        defer trainer.deinit();
        try runTrainEval(&ctx, io, stdout, allocator, TrainerQV, &trainer, &graft, ids, opts, true);
    } else {
        var trainer = try TrainerNone.init(&ctx, &model, .{ .rank = 1, .alpha = 1 }, opts.seed);
        defer trainer.deinit();
        try runTrainEval(&ctx, io, stdout, allocator, TrainerNone, &trainer, &graft, ids, opts, false);
    }
}

/// Hash a chunk's rows per plan slot. Caller frees each slice.
fn hashChunk(allocator: std.mem.Allocator, graft: *const engram.Engram, tokens: []const usize, rows_out: [][]usize) !void {
    const heads = graft.plan.cfg.headsPerLayer();
    const ids = try allocator.alloc(i64, tokens.len);
    defer allocator.free(ids);
    for (ids, tokens) |*dst, tok| dst.* = @intCast(tok);
    for (rows_out, 0..) |*slot_rows, slot| {
        slot_rows.* = try allocator.alloc(usize, tokens.len * heads);
        try graft.plan.hashInto(slot, ids, slot_rows.*);
    }
}

fn runEquiv(ctx: *ExecContext, stdout: anytype, allocator: std.mem.Allocator, model: *llm.qwen3.model.Model, graft: *engram.Engram, opts: Options) !void {
    var trainer = try TrainerNone.init(ctx, model, .{ .rank = 1, .alpha = 1 }, opts.seed);
    defer trainer.deinit();
    const text = "The equivalence gate feeds a fixed English sentence through both forwards and compares every logit bitwise.";
    const ids32 = try tokenizer_encode(allocator, text);
    defer allocator.free(ids32);

    const n_slots = graft.layers.len;
    const rows = try allocator.alloc([]usize, n_slots);
    defer {
        for (rows) |r| allocator.free(r);
        allocator.free(rows);
    }
    try hashChunk(allocator, graft, ids32, rows);
    const rows_const = try allocator.alloc([]const usize, n_slots);
    defer allocator.free(rows_const);
    for (rows_const, rows) |*dst, src| dst.* = src;

    var bare = try trainer.evalLogits(ctx, ids32);
    defer bare.deinit();
    var grafted = try trainer.evalLogitsExt(ctx, ids32, .{ .engram = .{ .model = graft, .rows = rows_const } });
    defer grafted.deinit();

    const b = try bare.dataConst();
    const g = try grafted.dataConst();
    var max_diff: f32 = 0;
    var diff_count: usize = 0;
    for (b, g) |x, y| {
        const d = @abs(x - y);
        if (d > 0) diff_count += 1;
        max_diff = @max(max_diff, d);
    }
    try stdout.print("equiv: {d} logits, {d} differ, max |d| = {e}\n", .{ b.len, diff_count, max_diff });
    if (diff_count != 0) return error.EquivalenceFailed;
    try stdout.print("equiv: BITWISE IDENTICAL (zero-init graft is exact identity)\n", .{});
}

var tokenizer_ptr: ?*llm.tokenizer.Tokenizer = null;

fn tokenizer_encode(allocator: std.mem.Allocator, text: []const u8) ![]usize {
    const t = tokenizer_ptr orelse return error.NoTokenizer;
    const ids32 = try t.encode(allocator, text);
    defer allocator.free(ids32);
    const out = try allocator.alloc(usize, ids32.len);
    for (out, ids32) |*dst, id| dst.* = id;
    return out;
}

/// Train (and periodically eval) or eval-only over the chunked stream.
fn runTrainEval(
    ctx: *ExecContext,
    io: std.Io,
    stdout: anytype,
    allocator: std.mem.Allocator,
    comptime TrainerT: type,
    trainer: *TrainerT,
    graft: *engram.Engram,
    ids: []const usize,
    opts: Options,
    lora_arm: bool,
) !void {
    const chunk = opts.chunk;
    const n_chunks = ids.len / chunk;
    const n_slots = graft.layers.len;

    var opt = fucina.optim.AdamW.init(allocator, .{ .lr = opts.lr, .weight_decay = 0 });
    defer opt.deinit();
    if (opts.train) {
        if (!opts.no_engram) try graft.registerParams(&opt);
        if (lora_arm) try trainer.registerAllParams(&opt);
    }

    const rows = try allocator.alloc([]usize, n_slots);
    defer allocator.free(rows);
    const rows_const = try allocator.alloc([]const usize, n_slots);
    defer allocator.free(rows_const);

    const t0 = nowNs(io);
    var step: usize = 0;
    var train_chunk: usize = 0;
    while (opts.train and step < opts.steps) : (step += 1) {
        // Skip held-out chunks (every 8th).
        while (train_chunk % 8 == 7) train_chunk = (train_chunk + 1) % n_chunks;
        const tokens = ids[train_chunk * chunk ..][0..chunk];
        train_chunk = (train_chunk + 1) % n_chunks;

        var fwd = llm.qwen3.train.ForwardOptions{};
        if (!opts.no_engram) {
            try hashChunk(allocator, graft, tokens[0 .. tokens.len - 1], rows);
            for (rows_const, rows) |*dst, src| dst.* = src;
            fwd.engram = .{ .model = graft, .rows = rows_const };
        }
        defer if (!opts.no_engram) for (rows) |r| allocator.free(r);

        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try trainer.lossForwardExt(ctx, tokens[0 .. tokens.len - 1], tokens[1..], fwd, .{});
        defer loss.deinit();
        const v = try loss.item();
        try loss.backward(ctx);
        try opt.step(ctx);
        opt.zeroGrad();
        graft.zeroGrad();

        if (step % 5 == 0 or step + 1 == opts.steps) {
            try stdout.print("step {d}/{d}: train CE {d:.4} ({d:.1}s)\n", .{ step + 1, opts.steps, v, seconds(nowNs(io) - t0) });
            stdout.flush() catch {};
        }
        if ((step + 1) % opts.eval_every == 0 or step + 1 == opts.steps) {
            try evalHeldOut(ctx, stdout, allocator, TrainerT, trainer, graft, ids, opts, !opts.no_engram);
        }
    }

    if (!opts.train) {
        try evalHeldOut(ctx, stdout, allocator, TrainerT, trainer, graft, ids, opts, opts.load_path.len > 0);
    }

    if (opts.probes > 0) {
        try runProbes(ctx, stdout, allocator, TrainerT, trainer, graft, ids, opts);
    }

    if (opts.train and opts.save_path.len > 0) {
        try fucina.training_checkpoint.writeFileAtomic(io, opts.save_path, graft, saveGraft);
        try stdout.print("saved graft: {s}\n", .{opts.save_path});
    }
}

/// Held-out CE + next-token accuracy: bare arm vs grafted arm on the same
/// chunks (every 8th).
fn evalHeldOut(
    ctx: *ExecContext,
    stdout: anytype,
    allocator: std.mem.Allocator,
    comptime TrainerT: type,
    trainer: *TrainerT,
    graft: *engram.Engram,
    ids: []const usize,
    opts: Options,
    with_graft: bool,
) !void {
    const chunk = opts.chunk;
    const n_chunks = ids.len / chunk;
    const n_slots = graft.layers.len;
    const rows = try allocator.alloc([]usize, n_slots);
    defer allocator.free(rows);
    const rows_const = try allocator.alloc([]const usize, n_slots);
    defer allocator.free(rows_const);

    const arms: [2][]const u8 = .{ "bare ", "graft" };
    const arm_count: usize = if (with_graft) 2 else 1;
    for (0..arm_count) |arm| {
        var total: f64 = 0;
        var count: usize = 0;
        var held: usize = 7;
        while (held < n_chunks and count < opts.eval_chunks) : (held += 8) {
            const tokens = ids[held * chunk ..][0..chunk];
            var fwd = llm.qwen3.train.ForwardOptions{};
            if (arm == 1) {
                try hashChunk(allocator, graft, tokens[0 .. tokens.len - 1], rows);
                for (rows_const, rows) |*dst, src| dst.* = src;
                fwd.engram = .{ .model = graft, .rows = rows_const };
            }
            defer if (arm == 1) for (rows) |r| allocator.free(r);

            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var no_grad = fucina.noGrad();
            defer no_grad.close();
            var loss = try trainer.lossForwardExt(ctx, tokens[0 .. tokens.len - 1], tokens[1..], fwd, .{});
            defer loss.deinit();
            total += try loss.item();
            count += 1;
        }
        try stdout.print("  held-out CE [{s}]: {d:.4} over {d} chunks\n", .{ arms[arm], total / @as(f64, @floatFromInt(count)), count });
        stdout.flush() catch {};
    }
}

/// One flat token stream from a file or a directory's top-level .md/.txt
/// files (sorted). Sets `tokenizer_ptr` as a side effect (equiv reuses it).
const probe_prefix_len = 32;
const probe_target_len = 16;

/// Verbatim-recall probes: teacher-forced CE + greedy exact-match rate
/// over 16-token targets after 32-token corpus prefixes, sampled
/// deterministically from train spans and held-out spans, scored with the
/// engram detached and attached. The paired train-span delta is the
/// memory's recall value; the held-out delta separates recall from
/// generalization.
fn runProbes(
    ctx: *ExecContext,
    stdout: anytype,
    allocator: std.mem.Allocator,
    comptime TrainerT: type,
    trainer: *TrainerT,
    graft: *engram.Engram,
    ids: []const usize,
    opts: Options,
) !void {
    const chunk = opts.chunk;
    const n_chunks = ids.len / chunk;
    const n_slots = graft.layers.len;
    const span = probe_prefix_len + probe_target_len;
    if (chunk < span) return;

    const rows = try allocator.alloc([]usize, n_slots);
    defer allocator.free(rows);
    const rows_const = try allocator.alloc([]const usize, n_slots);
    defer allocator.free(rows_const);

    var prng = std.Random.DefaultPrng.init(opts.seed +% 0x9E37);
    const random = prng.random();

    const source_names = [_][]const u8{ "train", "held " };
    for (source_names, 0..) |source, which| {
        var ce_sum = [2]f64{ 0, 0 };
        var hits = [2]usize{ 0, 0 };
        var total_targets: usize = 0;
        for (0..opts.probes) |_| {
            // Sample a chunk of the right parity, then a span inside it.
            var c = random.uintLessThan(usize, n_chunks);
            if (which == 0) {
                while (c % 8 == 7) c = random.uintLessThan(usize, n_chunks);
            } else {
                c = (c / 8) * 8 + 7;
                if (c >= n_chunks) c -= 8;
            }
            const max_off = chunk - span;
            const off = random.uintLessThan(usize, max_off + 1);
            const tokens = ids[c * chunk + off ..][0..span];
            total_targets += probe_target_len;

            for (0..2) |arm| {
                var fwd = llm.qwen3.train.ForwardOptions{};
                if (arm == 1 and !opts.no_engram) {
                    try hashChunk(allocator, graft, tokens, rows);
                    for (rows_const, rows) |*dst, src| dst.* = src;
                    fwd.engram = .{ .model = graft, .rows = rows_const };
                }
                defer if (arm == 1 and !opts.no_engram) for (rows) |r| allocator.free(r);

                const scope = ctx.openExecScope();
                defer ctx.closeExecScope(scope);
                var no_grad = fucina.noGrad();
                defer no_grad.close();
                var logits = try trainer.evalLogitsExt(ctx, tokens, fwd);
                defer logits.deinit();
                const data = try logits.dataConst();
                const vocab = logits.dim(.vocab);
                for (probe_prefix_len - 1..span - 1) |row| {
                    const target = tokens[row + 1];
                    const scores = data[row * vocab ..][0..vocab];
                    var max_v: f32 = scores[0];
                    var max_i: usize = 0;
                    var lse: f64 = 0;
                    for (scores, 0..) |v, i| {
                        if (v > max_v) {
                            max_v = v;
                            max_i = i;
                        }
                    }
                    for (scores) |v| lse += @exp(@as(f64, v - max_v));
                    const logprob = @as(f64, scores[target] - max_v) - @log(lse);
                    ce_sum[arm] -= logprob;
                    if (max_i == target) hits[arm] += 1;
                }
            }
        }
        const nt = @as(f64, @floatFromInt(total_targets));
        try stdout.print("probes [{s}] detached: CE {d:.4}, exact {d:.1}% | attached: CE {d:.4}, exact {d:.1}%\n", .{
            source,
            ce_sum[0] / nt,
            100.0 * @as(f64, @floatFromInt(hits[0])) / nt,
            ce_sum[1] / nt,
            100.0 * @as(f64, @floatFromInt(hits[1])) / nt,
        });
        stdout.flush() catch {};
    }
}

fn saveGraft(graft: *engram.Engram, writer: *std.Io.Writer) anyerror!void {
    try graft.saveStateDict(writer);
}

fn nowNs(io: std.Io) i128 {
    return std.Io.Clock.real.now(io).nanoseconds;
}

fn seconds(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn loadCorpusIds(allocator: std.mem.Allocator, io: std.Io, tokenizer: *llm.tokenizer.Tokenizer, path: []const u8) ![]usize {
    tokenizer_ptr = tokenizer;
    if (path.len == 0) return error.NoCorpus;
    var list: std.ArrayList(usize) = .empty;
    errdefer list.deinit(allocator);

    if (std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true })) |dir_const| {
        var dir = dir_const;
        defer dir.close(io);
        var names: std.ArrayList([]u8) = .empty;
        defer {
            for (names.items) |n| allocator.free(n);
            names.deinit(allocator);
        }
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.ascii.endsWithIgnoreCase(entry.name, ".md") and !std.ascii.endsWithIgnoreCase(entry.name, ".txt")) continue;
            try names.append(allocator, try allocator.dupe(u8, entry.name));
        }
        std.mem.sort([]u8, names.items, {}, struct {
            fn lt(_: void, a: []u8, b: []u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        for (names.items) |name| {
            const joined = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ path, name });
            defer allocator.free(joined);
            try appendFileIds(allocator, io, tokenizer, &list, joined);
        }
    } else |_| {
        try appendFileIds(allocator, io, tokenizer, &list, path);
    }
    return list.toOwnedSlice(allocator);
}

fn appendFileIds(allocator: std.mem.Allocator, io: std.Io, tokenizer: *llm.tokenizer.Tokenizer, list: *std.ArrayList(usize), path: []const u8) !void {
    var dir = std.Io.Dir.cwd();
    const text = try dir.readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(text);
    const ids32 = try tokenizer.encode(allocator, text);
    defer allocator.free(ids32);
    try list.ensureUnusedCapacity(allocator, ids32.len);
    for (ids32) |id| list.appendAssumeCapacity(id);
}
