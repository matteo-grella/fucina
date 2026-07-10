//! NLA — a natural-language autoencoder over a frozen Qwen3 GGUF, CPU LoRA
//! edition (inspired by github.com/kitft/natural_language_autoencoders).
//!
//! Two LoRA-adapted views of ONE frozen base model turn a residual-stream
//! vector into text and back:
//!
//!   AV (activation verbalizer, vector -> text): the target vector is
//!   L2-normalized, scaled by `inject_scale`, and INJECTED as a single token
//!   embedding at a placeholder position inside a fixed ChatML prompt (the
//!   differentiable `setSlice` seam in `llm.qwen3.train`); the LoRA-adapted
//!   model then autoregresses a description. Trained with cross-entropy on
//!   the description tokens only (prompt positions masked via ignore_index).
//!
//!   AR (activation reconstructor, text -> vector): a TRUNCATED K+1-layer
//!   LoRA-adapted forward of the description text; the FINAL token's raw
//!   residual goes through a trainable Linear(d, d) head, is L2-normalized,
//!   and is trained with MSE against the L2-normalized target vector — for
//!   unit vectors the sum-reduced MSE is exactly 2*(1 - cos), so the loss is
//!   direction-only.
//!
//! Injection/capture convention (both directions share it):
//!   - layer K (default 18, ~2/3 of Qwen3-0.6B's 28 layers): vectors are the
//!     raw residual AFTER layer K's block, i.e. a `forwardHidden` over
//!     layer_count = K+1 layers, read at the FINAL token position;
//!   - both the stored target and the reconstruction are L2-normalized, and
//!     the round-trip metric is MSE(recon, orig) = 2*(1 - cos) on those unit
//!     vectors;
//!   - the injected embedding row is l2norm(vec) * inject_scale, where
//!     inject_scale defaults to the MEAN L2 NORM OF THE EMBEDDING TABLE ROWS
//!     (computed at load and printed) so the injected row lands at the
//!     magnitude the first layer expects from a real token embedding.
//!
//! The fixed AV prompt, rendered verbatim (ChatML, think off; the
//! `<|fim_pad|>` special token is the placeholder whose embedding the
//! injection replaces — as a special token it always pretokenizes to exactly
//! one id and can never merge with its neighbors):
//!
//!   <|im_start|>user
//!   Describe the text represented by this embedding: <|fim_pad|><|im_end|>
//!   <|im_start|>assistant
//!   <think>
//!
//!   </think>
//!
//! HONEST BOOTSTRAP CAVEAT: the reference trains AV against API-generated
//! explanations of the activations. There is no API teacher on CPU, so this
//! example bootstraps descriptions = the source snippets themselves, which
//! turns the task into a text -> vector -> text AUTOENCODER: results read as
//! reconstruction quality, not activation interpretation. (It also makes the
//! AR-only "oracle" round trip near-perfect once AR converges — the AR input
//! IS the text whose activation produced the target.)
//!
//! Stages (all artifacts live under --dir):
//!   --datagen   corpus snippets -> (snippets.txt, vectors.safetensors with
//!               "vec.<i>" entries): base-model activations through the eval
//!               trainer path (fresh LoRA, B == 0 => delta == 0).
//!   --train-ar  AR adapters + Linear(d,d) head ("ar_head.weight") with AdamW
//!               on the train split -> <dir>/ar/{adapters,model}.safetensors.
//!   --train-av  AV adapters with lossInjected on the train split
//!               -> <dir>/av/adapters.safetensors.
//!   --eval      held-out tail split: vec -> AV greedy description (trained
//!               AND untrained-AV baseline) -> AR reconstruction; reports
//!               per-sample and mean cos / 2(1-cos), plus the AR-oracle upper
//!               bound (AR on the true snippet).
//!   --demo      all four stages with tiny defaults in one run.
//!
//! Scope note: this is V1 = the reference's SFT stages + round-trip eval.
//! The reference's stage 3 — simultaneous RL where AV trains with GRPO
//! (reward = -mse of the normalized round trip) while AR keeps training
//! supervised on the sampled descriptions — is the natural follow-up: all
//! the pieces exist here (sampled generation via `evalLastLogitsExt` with
//! injection, per-sample rewards from the AR round trip, and
//! `LossOptions.loss_scale` for advantage-weighted CE), but rollouts through
//! the O(n^2) no-KV-cache trainer eval path are minutes per vector on CPU,
//! so it is documented rather than shipped.
//!
//! Deep-layer note: the AR trainer allocates adapters for all layers but its
//! truncated forward never touches layers K+1.., so those adapters stay
//! gradless — harmless (AdamW skips params without gradients; they persist
//! as their init values in the checkpoint).
//!
//! Run (see also `--help` output):
//!   zig build nla -Doptimize=ReleaseFast -- --demo
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const optim = fucina.optim;
const rng = fucina.rng;
const safetensors = fucina.safetensors;
const training_checkpoint = fucina.training_checkpoint;
const qwen3_train = llm.qwen3.train;

const Hidden = qwen3_train.Hidden;
/// LoRA on q and v projections (the classic target set, as in finetune.zig).
const Trainer = qwen3_train.Trainer(.{ .q = true, .v = true });
/// The AR reconstruction head: Linear(d, d) as a tagged [recon, embed] matrix.
const HeadTensor = fucina.Tensor(.{ .recon, .embed });

const default_model = "models/Qwen3-0.6B-Q4_K_S.gguf";
const default_dir = "/tmp/fucina-nla";
const default_layer_k: usize = 18;
const default_steps: usize = 100;
const demo_steps: usize = 20;
/// Snippets longer than this many tokens are skipped at datagen (keeps the
/// O(n^2) eval generation and the AV sequences short).
const max_snippet_tokens: usize = 64;
/// eps under the squared norm in every differentiable L2 normalization.
const l2_eps: f32 = 1e-12;

const vectors_file = "vectors.safetensors";
const snippets_file = "snippets.txt";
const ar_subdir = "ar";
const av_subdir = "av";
/// The AV placeholder: a Qwen special token, so it pretokenizes to exactly
/// one id and never merges with neighboring text.
const placeholder_text = "<|fim_pad|>";
const av_user_text = "Describe the text represented by this embedding: " ++ placeholder_text;

/// The built-in corpus: short, topically distinctive snippets so nearby
/// vectors are well separated and a tiny run shows real signal.
const builtin_corpus = [_][]const u8{
    "The cat sat on the warm windowsill and purred at the falling rain.",
    "Volcanoes erupt when molten magma forces its way up through the crust.",
    "She sold seashells and painted postcards by the seashore every summer.",
    "Quantum computers use qubits that hold superpositions of zero and one.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
    "Ancient Rome built stone roads so durable that some are still walked today.",
    "Rainbows appear when sunlight bends and splits inside falling raindrops.",
    "The stock market fell sharply after the surprise interest rate announcement.",
    "Penguins huddle in tight circles to survive the Antarctic winter storms.",
    "A gentle melody drifted down from the old piano in the upstairs room.",
};

const Options = struct {
    model_path: []const u8 = default_model,
    dir: []const u8 = default_dir,
    corpus_path: ?[]const u8 = null,
    layer_k: usize = default_layer_k,
    steps: usize = default_steps,
    steps_set: bool = false,
    lr: f32 = 1e-3,
    rank: usize = 8,
    alpha: f32 = 16,
    seed: u64 = 42,
    inject_scale: ?f32 = null,
    max_desc_tokens: usize = 24,
    datagen: bool = false,
    train_ar: bool = false,
    train_av: bool = false,
    eval: bool = false,
    demo: bool = false,
};

/// Everything the stage functions share. The model/tokenizer are loaded once;
/// `inject_scale` is resolved lazily (it costs a pass over the embedding
/// table) and memoized.
const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    tokenizer: *const llm.tokenizer.Tokenizer,
    template: llm.chat.Template,
    opts: *const Options,
    resolved_scale: ?f32 = null,

    fn hiddenSize(self: *const Session) usize {
        return self.model.config.hidden_size;
    }

    /// K+1: the number of layers whose output is the captured residual.
    fn arLayerCount(self: *const Session) usize {
        return self.opts.layer_k + 1;
    }

    /// --inject-scale, or the mean embedding-row L2 norm (computed once).
    fn injectScale(self: *Session) !f32 {
        if (self.opts.inject_scale) |scale| return scale;
        if (self.resolved_scale) |scale| return scale;
        const scale = try meanEmbeddingNorm(self.ctx, self.model);
        try self.stdout.print("inject scale: {d:.4} (mean embedding-row L2 norm over {d} rows)\n", .{ scale, self.model.config.vocab_size });
        try self.stdout.flush();
        self.resolved_scale = scale;
        return scale;
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var opts = Options{};
    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (try optValue(args, &arg_i, "--model")) |v| {
            opts.model_path = v;
        } else if (try optValue(args, &arg_i, "--dir")) |v| {
            opts.dir = v;
        } else if (try optValue(args, &arg_i, "--corpus")) |v| {
            opts.corpus_path = v;
        } else if (try optValue(args, &arg_i, "--layer-k")) |v| {
            opts.layer_k = try std.fmt.parseInt(usize, v, 10);
        } else if (try optValue(args, &arg_i, "--steps")) |v| {
            opts.steps = try std.fmt.parseInt(usize, v, 10);
            opts.steps_set = true;
        } else if (try optValue(args, &arg_i, "--lr")) |v| {
            opts.lr = try std.fmt.parseFloat(f32, v);
        } else if (try optValue(args, &arg_i, "--rank")) |v| {
            opts.rank = try std.fmt.parseInt(usize, v, 10);
        } else if (try optValue(args, &arg_i, "--alpha")) |v| {
            opts.alpha = try std.fmt.parseFloat(f32, v);
        } else if (try optValue(args, &arg_i, "--seed")) |v| {
            opts.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (try optValue(args, &arg_i, "--inject-scale")) |v| {
            opts.inject_scale = try std.fmt.parseFloat(f32, v);
        } else if (try optValue(args, &arg_i, "--max-desc-tokens")) |v| {
            opts.max_desc_tokens = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--datagen")) {
            opts.datagen = true;
        } else if (std.mem.eql(u8, arg, "--train-ar")) {
            opts.train_ar = true;
        } else if (std.mem.eql(u8, arg, "--train-av")) {
            opts.train_av = true;
        } else if (std.mem.eql(u8, arg, "--eval")) {
            opts.eval = true;
        } else if (std.mem.eql(u8, arg, "--demo")) {
            opts.demo = true;
        } else {
            try stdout.print(
                "usage: zig build nla -Doptimize=ReleaseFast -- (--datagen|--train-ar|--train-av|--eval|--demo)...\n" ++
                    "       [--model PATH] [--dir DIR] [--corpus FILE] [--layer-k N] [--steps N] [--lr F]\n" ++
                    "       [--rank N] [--alpha F] [--seed N] [--inject-scale F] [--max-desc-tokens N]\n",
                .{},
            );
            return error.UnknownArgument;
        }
    }
    if (opts.demo) {
        opts.datagen = true;
        opts.train_ar = true;
        opts.train_av = true;
        opts.eval = true;
        if (!opts.steps_set) opts.steps = demo_steps;
    }
    if (!(opts.datagen or opts.train_ar or opts.train_av or opts.eval)) {
        try stdout.print("nothing to do: pass --datagen, --train-ar, --train-av, --eval, or --demo\n", .{});
        return error.NoStageSelected;
    }

    const allocator = std.heap.smp_allocator;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Load model + tokenizer from the same GGUF parse (finetune.zig pattern).
    const load_start = nowNs(io);
    var file = try fucina.gguf.File.loadMmap(allocator, io, opts.model_path);
    var model = try llm.qwen3.model.Model.loadGgufFromFile(&ctx, &file, try llm.qwen3.model.Config.fromGguf(&file));
    defer model.deinit();
    var tokenizer = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tokenizer.deinit();
    const template = llm.chat.Template.detect(file.getString("tokenizer.chat_template")) orelse llm.chat.Template{ .format = .chatml };
    file.deinit();
    try stdout.print("model: {s} ({d} layers, hidden {d})  load {d:.2} s\n", .{
        opts.model_path, model.config.num_layers, model.config.hidden_size, seconds(nowNs(io) - load_start),
    });
    if (opts.layer_k + 1 > model.config.num_layers) return error.LayerKOutOfRange;

    var session = Session{
        .allocator = allocator,
        .io = io,
        .stdout = stdout,
        .ctx = &ctx,
        .model = &model,
        .tokenizer = &tokenizer,
        .template = template,
        .opts = &opts,
    };

    if (opts.datagen) try runDatagen(&session);
    if (opts.train_ar) try runTrainAr(&session);
    if (opts.train_av) try runTrainAv(&session);
    if (opts.eval) try runEval(&session);
}

/// "--name VALUE" or "--name=VALUE"; advances `arg_i` past a consumed value.
fn optValue(args: []const []const u8, arg_i: *usize, comptime name: []const u8) !?[]const u8 {
    const arg = args[arg_i.*];
    if (std.mem.eql(u8, arg, name)) {
        arg_i.* += 1;
        if (arg_i.* >= args.len) return error.MissingOptionValue;
        return args[arg_i.*];
    }
    if (std.mem.startsWith(u8, arg, name ++ "=")) return arg[name.len + 1 ..];
    return null;
}

// ---------------------------------------------------------------------------
// Pure helpers (unit-tested in nla_tests.zig).
// ---------------------------------------------------------------------------

pub const Split = struct { train: usize, eval: usize };

/// Deterministic corpus split: the TAIL max(1, n/5) snippets are held out
/// for --eval (n < 2 keeps everything in the train split).
pub fn splitCounts(n: usize) Split {
    if (n < 2) return .{ .train = n, .eval = 0 };
    const held = @max(1, n / 5);
    return .{ .train = n - held, .eval = held };
}

/// The index of `needle` in `ids` iff it occurs exactly once.
pub fn findOnce(ids: []const usize, needle: usize) ?usize {
    var found: ?usize = null;
    for (ids, 0..) |id, i| {
        if (id == needle) {
            if (found != null) return null;
            found = i;
        }
    }
    return found;
}

/// In-place L2 normalization; returns the original norm (a zero vector is
/// left unchanged and reports norm 0).
pub fn l2NormalizeInPlace(v: []f32) f64 {
    var sq: f64 = 0;
    for (v) |x| sq += @as(f64, x) * @as(f64, x);
    const norm = @sqrt(sq);
    if (norm == 0) return 0;
    const inv: f32 = @floatCast(1.0 / norm);
    for (v) |*x| x.* *= inv;
    return norm;
}

/// Cosine similarity in f64 (0 when either vector is zero).
pub fn cosine(a: []const f32, b: []const f32) f64 {
    std.debug.assert(a.len == b.len);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * @as(f64, y);
        na += @as(f64, x) * @as(f64, x);
        nb += @as(f64, y) * @as(f64, y);
    }
    if (na == 0 or nb == 0) return 0;
    return dot / (@sqrt(na) * @sqrt(nb));
}

/// The round-trip metric on L2-normalized vectors: MSE = 2*(1 - cos).
pub fn roundTripMse(cos_value: f64) f64 {
    return 2.0 * (1.0 - cos_value);
}

pub const Sample = struct {
    /// Model inputs: full sequence minus the final token.
    inputs: []usize,
    /// Next-token labels; prompt positions masked with the ignore sentinel.
    labels: []usize,

    pub fn deinit(self: *Sample, allocator: std.mem.Allocator) void {
        allocator.free(self.labels);
        allocator.free(self.inputs);
        self.* = undefined;
    }
};

/// prompt tokens ++ response tokens -> next-token SFT sample: inputs are the
/// sequence minus its last token; labels are the one-position shift with all
/// prompt positions masked (`qwen3_train.ignore_index`).
pub fn buildSample(allocator: std.mem.Allocator, prompt: []const usize, response: []const usize) !Sample {
    if (prompt.len == 0 or response.len == 0) return error.EmptySample;
    const total = prompt.len + response.len;
    const inputs = try allocator.alloc(usize, total - 1);
    errdefer allocator.free(inputs);
    const labels = try allocator.alloc(usize, total - 1);
    errdefer allocator.free(labels);
    for (0..total) |i| {
        const token = if (i < prompt.len) prompt[i] else response[i - prompt.len];
        if (i < total - 1) inputs[i] = token;
        // Position i-1 predicts token i; supervise response tokens only.
        if (i > 0) labels[i - 1] = if (i < prompt.len) qwen3_train.ignore_index else token;
    }
    return .{ .inputs = inputs, .labels = labels };
}

// ---------------------------------------------------------------------------
// Corpus + artifact I/O.
// ---------------------------------------------------------------------------

const Corpus = struct {
    /// Owned backing bytes for file-loaded corpora (null for the built-in).
    buffer: ?[]u8,
    /// Owned array of snippet slices (borrowing `buffer` or static strings).
    snippets: [][]const u8,

    fn deinit(self: *Corpus, allocator: std.mem.Allocator) void {
        allocator.free(self.snippets);
        if (self.buffer) |buffer| allocator.free(buffer);
        self.* = undefined;
    }
};

/// Non-empty lines of `bytes` (CR trimmed), as an owned slice of borrows.
fn splitLines(allocator: std.mem.Allocator, bytes: []const u8) ![][]const u8 {
    var lines: std.ArrayList([]const u8) = .empty;
    errdefer lines.deinit(allocator);
    var it = std.mem.splitScalar(u8, bytes, '\n');
    while (it.next()) |line| {
        const trimmed = std.mem.trimEnd(u8, line, "\r");
        if (trimmed.len == 0) continue;
        try lines.append(allocator, trimmed);
    }
    return lines.toOwnedSlice(allocator);
}

/// --corpus FILE (one snippet per line) or the built-in snippet array.
fn loadCorpus(allocator: std.mem.Allocator, io: std.Io, path: ?[]const u8) !Corpus {
    if (path) |p| {
        const bytes = try readFileAlloc(allocator, io, p);
        errdefer allocator.free(bytes);
        const snippets = try splitLines(allocator, bytes);
        return .{ .buffer = bytes, .snippets = snippets };
    }
    const snippets = try allocator.alloc([]const u8, builtin_corpus.len);
    for (snippets, builtin_corpus) |*dst, src| dst.* = src;
    return .{ .buffer = null, .snippets = snippets };
}

/// The snippets datagen recorded (the canonical downstream corpus).
fn loadSnippets(allocator: std.mem.Allocator, io: std.Io, dir: []const u8) !Corpus {
    const path = try training_checkpoint.pathJoin(allocator, dir, snippets_file);
    defer allocator.free(path);
    const bytes = readFileAlloc(allocator, io, path) catch |err| switch (err) {
        error.FileNotFound => return error.RunDatagenFirst,
        else => |e| return e,
    };
    errdefer allocator.free(bytes);
    const snippets = try splitLines(allocator, bytes);
    return .{ .buffer = bytes, .snippets = snippets };
}

/// Same threading pattern as training_checkpoint's private readFileAlloc.
fn readFileAlloc(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.IsDir;
    const len: usize = @intCast(stat.size);
    const bytes = try allocator.alloc(u8, len);
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}

/// vectors.safetensors: one f32 [d] entry per snippet, named "vec.<i>".
fn writeVectors(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, vectors: []const []const f32, d: usize) !void {
    const path = try training_checkpoint.pathJoin(allocator, dir, vectors_file);
    defer allocator.free(path);
    const names = try allocator.alloc([]u8, vectors.len);
    var built: usize = 0;
    defer {
        for (names[0..built]) |name| allocator.free(name);
        allocator.free(names);
    }
    const tensors = try allocator.alloc(safetensors.Tensor, vectors.len);
    defer allocator.free(tensors);
    const shape = [_]usize{d};
    for (vectors, 0..) |vec, i| {
        std.debug.assert(vec.len == d);
        names[i] = try std.fmt.allocPrint(allocator, "vec.{d}", .{i});
        built += 1;
        tensors[i] = .{ .name = names[i], .dtype = .F32, .shape = &shape, .data = std.mem.sliceAsBytes(vec) };
    }
    try safetensors.saveFileAtomic(allocator, io, path, tensors, null);
}

/// Load "vec.0".."vec.<count-1>" back as owned f32 slices (count from the
/// snippets file; strict shape/dtype checks).
fn loadVectors(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, count: usize, d: usize) ![][]f32 {
    const path = try training_checkpoint.pathJoin(allocator, dir, vectors_file);
    defer allocator.free(path);
    var file = safetensors.File.load(allocator, io, path) catch |err| switch (err) {
        error.FileNotFound => return error.RunDatagenFirst,
        else => |e| return e,
    };
    defer file.deinit();
    if (file.len() != count) return error.VectorCountMismatch;

    const out = try allocator.alloc([]f32, count);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |vec| allocator.free(vec);
        allocator.free(out);
    }
    var name_buf: [32]u8 = undefined;
    for (out, 0..) |*vec, i| {
        const name = try std.fmt.bufPrint(&name_buf, "vec.{d}", .{i});
        const info = try file.tensor(name);
        if (info.dtype != .F32 or info.shape.len != 1 or info.shape[0] != d) return error.VectorShapeMismatch;
        vec.* = try allocator.alloc(f32, d);
        built += 1;
        @memcpy(std.mem.sliceAsBytes(vec.*), info.data);
    }
    return out;
}

fn freeVectors(allocator: std.mem.Allocator, vectors: [][]f32) void {
    for (vectors) |vec| allocator.free(vec);
    allocator.free(vectors);
}

// ---------------------------------------------------------------------------
// Tokenization helpers (finetune.zig style).
// ---------------------------------------------------------------------------

fn encodeUsize(allocator: std.mem.Allocator, tokenizer: *const llm.tokenizer.Tokenizer, text: []const u8) ![]usize {
    const ids32 = try tokenizer.encodeRaw(allocator, text);
    defer allocator.free(ids32);
    const ids = try allocator.alloc(usize, ids32.len);
    for (ids, ids32) |*dst, src| dst.* = src;
    return ids;
}

fn decodeIds(allocator: std.mem.Allocator, tokenizer: *const llm.tokenizer.Tokenizer, ids: []const usize) ![]u8 {
    const ids32 = try allocator.alloc(u32, ids.len);
    defer allocator.free(ids32);
    for (ids32, ids) |*dst, src| dst.* = @intCast(src);
    return tokenizer.decode(allocator, ids32);
}

const Prompt = struct {
    ids: []usize,
    /// The placeholder token's position — the injection target.
    pos: usize,

    fn deinit(self: *Prompt, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        self.* = undefined;
    }
};

/// Render + encode the fixed AV prompt (see the header comment) and locate
/// the placeholder token.
fn buildPrompt(s: *Session) !Prompt {
    const allocator = s.allocator;
    const pad_id32 = s.tokenizer.tokenId(placeholder_text) orelse return error.PlaceholderTokenMissing;
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    try s.template.renderTurn(allocator, &text, null, av_user_text, true, true);
    const ids = try encodeUsize(allocator, s.tokenizer, text.items);
    errdefer allocator.free(ids);
    const pos = findOnce(ids, pad_id32) orelse return error.PlaceholderNotUnique;
    return .{ .ids = ids, .pos = pos };
}

// ---------------------------------------------------------------------------
// Stage: --datagen
// ---------------------------------------------------------------------------

fn runDatagen(s: *Session) !void {
    const allocator = s.allocator;
    const d = s.hiddenSize();
    const layer_count = s.arLayerCount();

    var corpus = try loadCorpus(allocator, s.io, s.opts.corpus_path);
    defer corpus.deinit(allocator);

    // Fresh trainer = LoRA B == 0 => delta == 0 => BASE-MODEL activations
    // through the exact numeric path AR will later train on.
    var trainer = try Trainer.init(s.ctx, s.model, .{ .rank = s.opts.rank, .alpha = s.opts.alpha }, s.opts.seed);
    defer trainer.deinit();

    var kept: std.ArrayList([]const u8) = .empty;
    defer kept.deinit(allocator);
    var vectors: std.ArrayList([]f32) = .empty;
    defer {
        for (vectors.items) |vec| allocator.free(vec);
        vectors.deinit(allocator);
    }

    try s.stdout.print("\n=== datagen: {d} snippets -> layer-{d} final-token residuals (d = {d}) ===\n", .{
        corpus.snippets.len, s.opts.layer_k, d,
    });
    for (corpus.snippets) |snippet| {
        const ids = try encodeUsize(allocator, s.tokenizer, snippet);
        defer allocator.free(ids);
        if (ids.len == 0 or ids.len > max_snippet_tokens) {
            try s.stdout.print("  skip ({d} tokens): {s}\n", .{ ids.len, snippet });
            continue;
        }
        const vec = try allocator.alloc(f32, d);
        errdefer allocator.free(vec);
        {
            const scope = s.ctx.openExecScope();
            defer s.ctx.closeExecScope(scope);
            const h = try trainer.forwardHidden(s.ctx, ids, null, .{ .layer_count = layer_count });
            const last = try h.narrow(s.ctx, .seq, ids.len - 1, 1);
            try last.copyTo(vec);
        }
        var sq: f64 = 0;
        for (vec) |x| sq += @as(f64, x) * @as(f64, x);
        try vectors.append(allocator, vec);
        try kept.append(allocator, snippet);
        try s.stdout.print("  vec.{d} |v| = {d:.3} ({d} tokens): {s}\n", .{ vectors.items.len - 1, @sqrt(sq), ids.len, snippet });
        try s.stdout.flush();
    }
    if (vectors.items.len == 0) return error.EmptyCorpus;

    try std.Io.Dir.cwd().createDirPath(s.io, s.opts.dir);
    const vec_slices = try allocator.alloc([]const f32, vectors.items.len);
    defer allocator.free(vec_slices);
    for (vec_slices, vectors.items) |*dst, src| dst.* = src;
    try writeVectors(allocator, s.io, s.opts.dir, vec_slices, d);

    const snippets_path = try training_checkpoint.pathJoin(allocator, s.opts.dir, snippets_file);
    defer allocator.free(snippets_path);
    const WriteSnippets = struct {
        fn write(snippets: []const []const u8, writer: *std.Io.Writer) !void {
            for (snippets) |line| {
                try writer.writeAll(line);
                try writer.writeByte('\n');
            }
        }
    };
    try training_checkpoint.writeFileAtomic(s.io, snippets_path, @as([]const []const u8, kept.items), WriteSnippets.write);

    const split = splitCounts(vectors.items.len);
    try s.stdout.print("wrote {d} pairs to {s} ({d} train / {d} held-out)\n", .{ vectors.items.len, s.opts.dir, split.train, split.eval });
    try s.stdout.flush();
}

/// Mean L2 norm of the embedding-table rows, streamed in id chunks through
/// the same dequantizing `getRowsAs` the forward uses.
fn meanEmbeddingNorm(ctx: *fucina.ExecContext, model: *const llm.qwen3.model.Model) !f32 {
    const d = model.config.hidden_size;
    const vocab = model.config.vocab_size;
    const chunk = 2048;
    var ids: [chunk]usize = undefined;
    var total: f64 = 0;
    var base: usize = 0;
    while (base < vocab) {
        const n = @min(chunk, vocab - base);
        for (ids[0..n], 0..) |*id, j| id.* = base + j;
        var rows = try model.token_embedding.getRowsAs(ctx, ids[0..n], .embed);
        defer rows.deinit();
        const data = try rows.dataConst();
        for (0..n) |r| {
            var sq: f64 = 0;
            for (data[r * d ..][0..d]) |v| sq += @as(f64, v) * @as(f64, v);
            total += @sqrt(sq);
        }
        base += n;
    }
    return @floatCast(total / @as(f64, @floatFromInt(vocab)));
}

// ---------------------------------------------------------------------------
// Stage: --train-ar
// ---------------------------------------------------------------------------

/// The AR head as a registered parameter set ("ar_head.weight"): a Linear(d,d)
/// variable with a seeded small-normal init (std = 1/sqrt(d), the usual
/// fan-in scale, via the repo RNG so runs are reproducible).
const ArHead = struct {
    weight: HeadTensor,
    registry: fucina.ParamRegistry,

    fn init(ctx: *fucina.ExecContext, d: usize, seed: u64) !ArHead {
        const values = try ctx.allocator.alloc(f32, d * d);
        defer ctx.allocator.free(values);
        rng.gaussianFill(rng.at(seed, 0x4EAD), values, 1.0 / @sqrt(@as(f32, @floatFromInt(d))));
        var weight = try HeadTensor.variableFromSlice(ctx, .{ d, d }, values);
        errdefer weight.deinit();
        var registry = fucina.ParamRegistry.init(ctx.allocator);
        errdefer registry.deinit();
        try registry.addParam("ar_head.weight", &weight);
        return .{ .weight = weight, .registry = registry };
    }

    fn deinit(self: *ArHead) void {
        // Registry first: it retains views of the weight's storage.
        self.registry.deinit();
        self.weight.deinit();
        self.* = undefined;
    }
};

/// AR reconstruction of one text: truncated K+1-layer forward, final-token
/// residual through the head, L2-normalized into `out` (len d).
fn arReconstruct(s: *Session, trainer: *Trainer, head: *const ArHead, text: []const u8, out: []f32) !void {
    var ids = try encodeUsize(s.allocator, s.tokenizer, text);
    defer s.allocator.free(ids);
    if (ids.len == 0) {
        // Degenerate description (e.g. an empty AV generation): reconstruct
        // from a single space so the pipeline stays total. Swap only after
        // the re-encode succeeds so the defer always frees a live slice.
        const fallback = try encodeUsize(s.allocator, s.tokenizer, " ");
        s.allocator.free(ids);
        ids = fallback;
        if (ids.len == 0) return error.EmptyDescription;
    }
    const scope = s.ctx.openExecScope();
    defer s.ctx.closeExecScope(scope);
    const h = try trainer.forwardHidden(s.ctx, ids, null, .{ .layer_count = s.arLayerCount() });
    const last = try h.narrow(s.ctx, .seq, ids.len - 1, 1);
    const pred = try last.dot(s.ctx, &head.weight, .embed);
    const pred_n = try pred.l2Normalize(s.ctx, .recon, l2_eps);
    try pred_n.copyTo(out);
}

fn runTrainAr(s: *Session) !void {
    const allocator = s.allocator;
    const d = s.hiddenSize();

    var corpus = try loadSnippets(allocator, s.io, s.opts.dir);
    defer corpus.deinit(allocator);
    const targets = try loadVectors(allocator, s.io, s.opts.dir, corpus.snippets.len, d);
    defer freeVectors(allocator, targets);
    // Targets are trained (and evaluated) L2-normalized: direction-only.
    for (targets) |vec| _ = l2NormalizeInPlace(vec);
    const split = splitCounts(corpus.snippets.len);
    if (split.train == 0) return error.EmptyCorpus;

    var trainer = try Trainer.init(s.ctx, s.model, .{ .rank = s.opts.rank, .alpha = s.opts.alpha }, s.opts.seed);
    defer trainer.deinit();
    var head = try ArHead.init(s.ctx, d, s.opts.seed);
    defer head.deinit();

    var opt = optim.AdamW.init(allocator, .{ .lr = s.opts.lr, .weight_decay = 0 });
    defer opt.deinit();
    try trainer.registerAllParams(&opt);
    try head.registry.addParamsTo(&opt);
    var set = optim.OptimizerSet.init(allocator);
    defer set.deinit();
    try set.add(&opt);

    try s.stdout.print("\n=== train-ar: {d} steps on {d} snippets (K+1 = {d} layers, lr {d}) ===\n", .{
        s.opts.steps, split.train, s.arLayerCount(), s.opts.lr,
    });
    try s.stdout.flush();
    var first: f32 = 0;
    var last_loss: f32 = 0;
    for (0..s.opts.steps) |step_i| {
        const idx = step_i % split.train;
        const ids = try encodeUsize(allocator, s.tokenizer, corpus.snippets[idx]);
        defer allocator.free(ids);
        var loss_value: f32 = 0;
        {
            const scope = s.ctx.openExecScope();
            defer s.ctx.closeExecScope(scope);
            const h = try trainer.forwardHidden(s.ctx, ids, null, .{ .layer_count = s.arLayerCount() });
            const last = try h.narrow(s.ctx, .seq, ids.len - 1, 1);
            const pred = try last.dot(s.ctx, &head.weight, .embed);
            const pred_n = try pred.l2Normalize(s.ctx, .recon, l2_eps);
            var target = try fucina.Tensor(.{ .seq, .recon }).fromSlice(s.ctx, .{ 1, d }, targets[idx]);
            defer target.deinit();
            // Unit vectors: sum-reduced MSE == 2*(1 - cos), direction-only.
            const loss = try pred_n.mseLoss(s.ctx, &target, .{ .reduction = .sum });
            try loss.backward(s.ctx);
            loss_value = try loss.item();
            _ = try set.clipGradNorm(s.ctx, 1.0);
            try set.step(s.ctx);
            set.zeroGrad();
        }
        if (step_i == 0) first = loss_value;
        last_loss = loss_value;
        try s.stdout.print("  step {d:>4}  mse 2(1-cos) {d:.4}  (cos {d:.4})\n", .{ step_i + 1, loss_value, 1.0 - loss_value / 2.0 });
        try s.stdout.flush();
    }
    if (s.opts.steps > 0) {
        try s.stdout.print("train-ar: mse {d:.4} -> {d:.4}\n", .{ first, last_loss });
    }

    try saveArCheckpoint(s, &trainer, &head);
    try s.stdout.print("saved AR checkpoint to {s}/{s}\n", .{ s.opts.dir, ar_subdir });
    try s.stdout.flush();
}

fn saveArCheckpoint(s: *Session, trainer: *const Trainer, head: *const ArHead) !void {
    const allocator = s.allocator;
    const dir = try training_checkpoint.pathJoin(allocator, s.opts.dir, ar_subdir);
    defer allocator.free(dir);
    try training_checkpoint.beginSave(allocator, s.io, dir);

    const adapters_path = try training_checkpoint.pathJoin(allocator, dir, training_checkpoint.adapters_state_file);
    defer allocator.free(adapters_path);
    const SaveAdapters = struct {
        fn write(t: *const Trainer, writer: *std.Io.Writer) !void {
            try t.saveAdapters(writer);
        }
    };
    try training_checkpoint.writeFileAtomic(s.io, adapters_path, trainer, SaveAdapters.write);

    // The head goes in model.safetensors (it IS the example-owned model part).
    const head_path = try training_checkpoint.pathJoin(allocator, dir, training_checkpoint.model_state_file);
    defer allocator.free(head_path);
    const SaveHead = struct {
        fn write(h: *const ArHead, writer: *std.Io.Writer) !void {
            try h.registry.saveStateDict(writer);
        }
    };
    try training_checkpoint.writeFileAtomic(s.io, head_path, head, SaveHead.write);

    try training_checkpoint.saveTrainerState(allocator, s.io, dir, .{
        .step = s.opts.steps,
        .seed = trainer.seed,
        .lora_rank = @intCast(trainer.lora_config.rank),
        .lora_alpha = @floatCast(trainer.lora_config.alpha),
        .learning_rate = @floatCast(s.opts.lr),
    });
}

fn loadArCheckpoint(s: *Session, trainer: *Trainer, head: *ArHead) !void {
    const allocator = s.allocator;
    const dir = try training_checkpoint.pathJoin(allocator, s.opts.dir, ar_subdir);
    defer allocator.free(dir);
    try loadAdaptersFile(s, dir, trainer);
    const head_path = try training_checkpoint.pathJoin(allocator, dir, training_checkpoint.model_state_file);
    defer allocator.free(head_path);
    var file = std.Io.Dir.cwd().openFile(s.io, head_path, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.RunTrainArFirst,
        else => |e| return e,
    };
    defer file.close(s.io);
    var buffer: [64 * 1024]u8 = undefined;
    var reader = file.reader(s.io, &buffer);
    try head.registry.loadStateDict(&reader.interface, .{});
}

fn loadAdaptersFile(s: *Session, dir: []const u8, trainer: *Trainer) !void {
    const path = try training_checkpoint.pathJoin(s.allocator, dir, training_checkpoint.adapters_state_file);
    defer s.allocator.free(path);
    var file = std.Io.Dir.cwd().openFile(s.io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.MissingAdapterCheckpoint,
        else => |e| return e,
    };
    defer file.close(s.io);
    var buffer: [64 * 1024]u8 = undefined;
    var reader = file.reader(s.io, &buffer);
    try trainer.loadAdapters(&reader.interface);
}

// ---------------------------------------------------------------------------
// Stage: --train-av
// ---------------------------------------------------------------------------

/// AV trainer seed: a distinct stream from AR so the two adapter inits never
/// coincide (they are independent parameter sets over the same frozen model).
fn avSeed(seed: u64) u64 {
    return seed +% 0xA5;
}

fn runTrainAv(s: *Session) !void {
    const allocator = s.allocator;
    const d = s.hiddenSize();

    var corpus = try loadSnippets(allocator, s.io, s.opts.dir);
    defer corpus.deinit(allocator);
    const targets = try loadVectors(allocator, s.io, s.opts.dir, corpus.snippets.len, d);
    defer freeVectors(allocator, targets);
    const split = splitCounts(corpus.snippets.len);
    if (split.train == 0) return error.EmptyCorpus;

    const scale = try s.injectScale();
    // Injected rows: l2norm(vec) * inject_scale, precomputed host-side.
    for (targets) |vec| {
        _ = l2NormalizeInPlace(vec);
        for (vec) |*x| x.* *= scale;
    }

    var prompt = try buildPrompt(s);
    defer prompt.deinit(allocator);

    var trainer = try Trainer.init(s.ctx, s.model, .{ .rank = s.opts.rank, .alpha = s.opts.alpha }, avSeed(s.opts.seed));
    defer trainer.deinit();
    var opt = optim.AdamW.init(allocator, .{ .lr = s.opts.lr, .weight_decay = 0 });
    defer opt.deinit();
    try trainer.registerAllParams(&opt);
    var set = optim.OptimizerSet.init(allocator);
    defer set.deinit();
    try set.add(&opt);

    // One SFT sample per train snippet: fixed prompt (masked, placeholder
    // included) + description tokens (supervised, closed by the stop marker).
    var samples = try allocator.alloc(Sample, split.train);
    var built: usize = 0;
    defer {
        for (samples[0..built]) |*sample| sample.deinit(allocator);
        allocator.free(samples);
    }
    for (samples, corpus.snippets[0..split.train]) |*sample, snippet| {
        var response_text: std.ArrayList(u8) = .empty;
        defer response_text.deinit(allocator);
        try response_text.appendSlice(allocator, snippet);
        try response_text.appendSlice(allocator, s.template.stopMarker());
        const response = try encodeUsize(allocator, s.tokenizer, response_text.items);
        defer allocator.free(response);
        sample.* = try buildSample(allocator, prompt.ids, response);
        built += 1;
    }

    try s.stdout.print("\n=== train-av: {d} steps on {d} snippets (inject pos {d}, scale {d:.4}, lr {d}) ===\n", .{
        s.opts.steps, split.train, prompt.pos, scale, s.opts.lr,
    });
    try s.stdout.flush();
    var first: f32 = 0;
    var last_loss: f32 = 0;
    for (0..s.opts.steps) |step_i| {
        const idx = step_i % split.train;
        var loss_value: f32 = 0;
        {
            const scope = s.ctx.openExecScope();
            defer s.ctx.closeExecScope(scope);
            var row = try Hidden.fromSlice(s.ctx, .{ 1, d }, targets[idx]);
            defer row.deinit();
            const loss = try trainer.lossInjected(s.ctx, samples[idx].inputs, samples[idx].labels, .{
                .pos = prompt.pos,
                .row = &row,
            }, .{});
            try loss.backward(s.ctx);
            loss_value = try loss.item();
            _ = try set.clipGradNorm(s.ctx, 1.0);
            try set.step(s.ctx);
            set.zeroGrad();
        }
        if (step_i == 0) first = loss_value;
        last_loss = loss_value;
        try s.stdout.print("  step {d:>4}  ce {d:.4}\n", .{ step_i + 1, loss_value });
        try s.stdout.flush();
    }
    if (s.opts.steps > 0) {
        try s.stdout.print("train-av: ce {d:.4} -> {d:.4}\n", .{ first, last_loss });
    }

    // Save: adapters + trainer state under <dir>/av.
    const dir = try training_checkpoint.pathJoin(allocator, s.opts.dir, av_subdir);
    defer allocator.free(dir);
    try training_checkpoint.beginSave(allocator, s.io, dir);
    const adapters_path = try training_checkpoint.pathJoin(allocator, dir, training_checkpoint.adapters_state_file);
    defer allocator.free(adapters_path);
    const SaveAdapters = struct {
        fn write(t: *const Trainer, writer: *std.Io.Writer) !void {
            try t.saveAdapters(writer);
        }
    };
    try training_checkpoint.writeFileAtomic(s.io, adapters_path, @as(*const Trainer, &trainer), SaveAdapters.write);
    try training_checkpoint.saveTrainerState(allocator, s.io, dir, .{
        .step = trainer.step_counter,
        .seed = trainer.seed,
        .lora_rank = @intCast(trainer.lora_config.rank),
        .lora_alpha = @floatCast(trainer.lora_config.alpha),
        .learning_rate = @floatCast(s.opts.lr),
    });
    try s.stdout.print("saved AV checkpoint to {s}\n", .{dir});
    try s.stdout.flush();
}

// ---------------------------------------------------------------------------
// Stage: --eval
// ---------------------------------------------------------------------------

/// Greedy AV generation with injection: evalLastLogitsExt re-forwards the
/// full prefix per token (no KV cache — demo-scale only, like finetune's
/// greedyGenerate) with the injected row substituted at the placeholder.
fn generateInjected(
    s: *Session,
    trainer: *Trainer,
    prompt: *const Prompt,
    row: *const Hidden,
    stop_id: ?usize,
) ![]usize {
    const allocator = s.allocator;
    var seq: std.ArrayList(usize) = .empty;
    defer seq.deinit(allocator);
    try seq.appendSlice(allocator, prompt.ids);

    var out: std.ArrayList(usize) = .empty;
    errdefer out.deinit(allocator);
    for (0..s.opts.max_desc_tokens) |_| {
        var logits = try trainer.evalLastLogitsExt(s.ctx, seq.items, .{
            .inject = .{ .pos = prompt.pos, .row = row },
        });
        defer logits.deinit();
        var index = try logits.argmax(s.ctx, .vocab);
        defer index.deinit();
        const next: usize = @intFromFloat((try index.dataConst())[0]);
        if (stop_id) |stop| if (next == stop) break;
        try out.append(allocator, next);
        try seq.append(allocator, next);
    }
    return out.toOwnedSlice(allocator);
}

const RoundTrip = struct {
    cos: f64,
    /// The generated description (caller-owned).
    text: []u8,
};

/// One AV -> AR round trip: generate a description from the (scaled) vector,
/// reconstruct a unit vector from the description, report cos vs the target.
fn roundTrip(
    s: *Session,
    av: *Trainer,
    ar: *Trainer,
    head: *const ArHead,
    prompt: *const Prompt,
    row: *const Hidden,
    target_n: []const f32,
    stop_id: ?usize,
    recon_buf: []f32,
) !RoundTrip {
    const gen = try generateInjected(s, av, prompt, row, stop_id);
    defer s.allocator.free(gen);
    const text = try decodeIds(s.allocator, s.tokenizer, gen);
    errdefer s.allocator.free(text);
    try arReconstruct(s, ar, head, text, recon_buf);
    return .{ .cos = cosine(recon_buf, target_n), .text = text };
}

fn runEval(s: *Session) !void {
    const allocator = s.allocator;
    const d = s.hiddenSize();

    var corpus = try loadSnippets(allocator, s.io, s.opts.dir);
    defer corpus.deinit(allocator);
    const targets = try loadVectors(allocator, s.io, s.opts.dir, corpus.snippets.len, d);
    defer freeVectors(allocator, targets);
    for (targets) |vec| _ = l2NormalizeInPlace(vec);
    const split = splitCounts(corpus.snippets.len);
    if (split.eval == 0) return error.NoHeldOutSnippets;

    const scale = try s.injectScale();
    var prompt = try buildPrompt(s);
    defer prompt.deinit(allocator);
    const stop_id: ?usize = if (s.tokenizer.tokenId(s.template.stopMarker())) |id| @as(usize, id) else null;

    // Trained AR (adapters + head) — also serves the baseline round trip and
    // the oracle: the question --eval answers is what the AV text preserves.
    var ar = try Trainer.init(s.ctx, s.model, .{ .rank = s.opts.rank, .alpha = s.opts.alpha }, s.opts.seed);
    defer ar.deinit();
    var head = try ArHead.init(s.ctx, d, s.opts.seed);
    defer head.deinit();
    loadArCheckpoint(s, &ar, &head) catch |err| switch (err) {
        error.MissingAdapterCheckpoint => return error.RunTrainArFirst,
        else => |e| return e,
    };

    // Trained AV vs the untrained-AV baseline (fresh adapters, zero delta =
    // the base model prompted identically): the bar for the AV stage is a
    // round-trip cos improvement over that baseline through the SAME AR.
    var av = try Trainer.init(s.ctx, s.model, .{ .rank = s.opts.rank, .alpha = s.opts.alpha }, avSeed(s.opts.seed));
    defer av.deinit();
    {
        const dir = try training_checkpoint.pathJoin(allocator, s.opts.dir, av_subdir);
        defer allocator.free(dir);
        loadAdaptersFile(s, dir, &av) catch |err| switch (err) {
            error.MissingAdapterCheckpoint => return error.RunTrainAvFirst,
            else => |e| return e,
        };
    }
    var av_base = try Trainer.init(s.ctx, s.model, .{ .rank = s.opts.rank, .alpha = s.opts.alpha }, avSeed(s.opts.seed));
    defer av_base.deinit();

    const recon = try allocator.alloc(f32, d);
    defer allocator.free(recon);
    const row_values = try allocator.alloc(f32, d);
    defer allocator.free(row_values);

    try s.stdout.print("\n=== eval: {d} held-out snippets (inject scale {d:.4}, max {d} desc tokens) ===\n", .{
        split.eval, scale, s.opts.max_desc_tokens,
    });
    try s.stdout.flush();

    var sum_trained: f64 = 0;
    var sum_base: f64 = 0;
    var sum_oracle: f64 = 0;
    for (split.train..corpus.snippets.len) |i| {
        const target_n = targets[i];
        for (row_values, target_n) |*x, t| x.* = t * scale;
        var row = try Hidden.fromSlice(s.ctx, .{ 1, d }, row_values);
        defer row.deinit();

        const trained = try roundTrip(s, &av, &ar, &head, &prompt, &row, target_n, stop_id, recon);
        defer allocator.free(trained.text);
        const base = try roundTrip(s, &av_base, &ar, &head, &prompt, &row, target_n, stop_id, recon);
        defer allocator.free(base.text);
        // AR-oracle upper bound: reconstruct from the TRUE snippet text.
        try arReconstruct(s, &ar, &head, corpus.snippets[i], recon);
        const cos_oracle = cosine(recon, target_n);

        sum_trained += trained.cos;
        sum_base += base.cos;
        sum_oracle += cos_oracle;
        try s.stdout.print("\n[vec.{d}] snippet: {s}\n", .{ i, corpus.snippets[i] });
        try s.stdout.print("  AV trained : {s}\n", .{trained.text});
        try s.stdout.print("  AV baseline: {s}\n", .{base.text});
        try s.stdout.print("  round-trip cos  trained {d:.4} (2(1-cos) {d:.4})  baseline {d:.4} ({d:.4})  AR-oracle {d:.4} ({d:.4})\n", .{
            trained.cos, roundTripMse(trained.cos),
            base.cos,    roundTripMse(base.cos),
            cos_oracle,  roundTripMse(cos_oracle),
        });
        try s.stdout.flush();
    }
    const n: f64 = @floatFromInt(split.eval);
    try s.stdout.print("\neval means over {d} held-out snippets:\n", .{split.eval});
    try s.stdout.print("  trained AV  round trip: cos {d:.4}  2(1-cos) {d:.4}\n", .{ sum_trained / n, roundTripMse(sum_trained / n) });
    try s.stdout.print("  baseline AV round trip: cos {d:.4}  2(1-cos) {d:.4}\n", .{ sum_base / n, roundTripMse(sum_base / n) });
    try s.stdout.print("  AR-oracle upper bound : cos {d:.4}  2(1-cos) {d:.4}\n", .{ sum_oracle / n, roundTripMse(sum_oracle / n) });
    try s.stdout.flush();
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
}

test {
    _ = @import("nla_tests.zig");
}
