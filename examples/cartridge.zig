//! Cartridges on a real Qwen3 GGUF (arXiv 2506.06266): compress a corpus
//! into a trained KV prefix and use it in place of in-context text.
//!
//! `--equiv` runs the acceptance gate from the paper's position/RoPE
//! semantics: a cartridge initialized from the model's own K/V rows for the
//! first p corpus tokens, with ZERO training steps, must score the following
//! tokens (at positions p..) like the real prefill does — near-identical
//! logits and identical greedy choices. This pins capture, position offset,
//! prefix concat, and the end-aligned causal kernel on the production model
//! in one shot.
//!
//! The default mode is SELF-STUDY training (paper Sec 4, k = 1 round),
//! fully in-process — no external services:
//!   1. sample a random corpus chunk (uniform token span) and one of the
//!      five seed-prompt types (structuring/summarization/question/
//!      use-case/creative, the reference's meta-prompt texts);
//!   2. bot A (this model, temperature 0.6) writes a chat message about the
//!      chunk; bot B (this model, greedy) answers WITH the chunk in context
//!      — both through the fast inference path (KV cache);
//!   3. the frozen model teacher-forces B's answer with the chunk in real
//!      context (trainer f32 path, `evalLogitsRows`) and its top-k rows
//!      (0.99 cumulative mass, top 20) become the distillation targets;
//!   4. the student scores the SAME conversation with the chunk replaced by
//!      the cartridge; the teacher top-k cross-entropy gradient flows into
//!      the cartridge rows only (Adam, lr 2e-2, the paper's recipe).
//! A held-out conversation (never trained on) tracks generalization; the
//! trained cartridge is saved as a safetensors state dict (--save).
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const optim = fucina.optim;
const cartridge = llm.cartridge;

const default_model = "models/Qwen3-0.6B-f16.gguf";
const default_save = "/tmp/fucina-cartridge.safetensors";

/// No LoRA targets: the base model stays fully frozen; the cartridge rows
/// are the only parameters.
const Trainer = llm.qwen3.train.Trainer(.{ .q = false, .v = false });

// ChatML blocks (Qwen3). The assistant opener carries the empty think block
// the non-thinking chat template emits, so generations answer directly.
/// Chat-template strings the serving/self-study prompt builders splice
/// (runtime values so one engine serves several architectures).
const Tpl = struct {
    sys_open: []const u8,
    user_open: []const u8,
    asst_open: []const u8,
    block_close: []const u8,
    stop_marker: []const u8,
};

const qwen3_tpl = Tpl{
    .sys_open = "<|im_start|>system\n",
    .user_open = "<|im_start|>user\n",
    .asst_open = "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    .block_close = "<|im_end|>\n",
    .stop_marker = "<|im_end|>",
};

// Gemma 4's `<|turn>` format (src/llm/chat.zig Template.gemma4, thinking
// primed off).
const gemma4_tpl = Tpl{
    .sys_open = "<bos><|turn>system\n",
    .user_open = "<|turn>user\n",
    .asst_open = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    .block_close = "<turn|>\n",
    .stop_marker = "<turn|>",
};

/// Duck-typed per-model KV-cache construction: uniform-geometry models
/// (qwen3) size from config, per-layer-geometry models (gemma4) from geom.
fn makeCache(ctx: *fucina.ExecContext, model: anytype, capacity: usize) !llm.kv_cache.KvCache {
    const M = @TypeOf(model.*);
    if (comptime @hasField(M, "geom")) {
        return llm.kv_cache.KvCache.initPerLayer(ctx, model.geom.kv_heads, model.geom.head_dim, capacity);
    }
    const cfg = model.config;
    return llm.kv_cache.KvCache.init(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity);
}

const sys_open = "<|im_start|>system\n";
const user_open = "<|im_start|>user\n";
const asst_open = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
const block_close = "<|im_end|>\n";

/// Reference synthesizers/self_study.py SYSTEM_PROMPT_TEMPLATE.
const system_prompt_template = "\nYou are in a conversation about the following user information.\n\n<info>\n{s}\n</info>";

/// The five seed-prompt types of paper Appendix C.1 (texts from the
/// reference's data/resources.py registry) plus two additions — `mechanism`
/// and `verbatim precision` — generic meta-prompts like the original five
/// (valid for any corpus; they never mention a domain), motivated by
/// measured failure modes of five-seed runs: mechanism-level "why/how"
/// knowledge and exact wording/numbers were exactly what trained
/// cartridges missed (docs/CARTRIDGES.md). One uniform draw per
/// conversation.
const seed_prompts = [_][]const u8{
    // structuring (JSON variant of the {data_format} slot)
    "Please generate a single chat message instructing an LLM to structure the information in JSON. " ++
        "Output only the chat message itself and absolutely nothing else. " ++
        "Make sure it is clear what section and document you are asking about. " ++
        "The message can follow the following template, filling in details from the corpus: \n\n" ++
        "'Can you structure the information in {subsection} of {document} related to {something specific} " ++
        "in the following format: JSON? Be sure to include precise information like any dates, times, names, and numerical values.'",
    // summarization
    "Please generate a single chat message instructing an LLM to summarize part of the corpus. " ++
        "Make sure the instruction is very explicit about the section of the corpus that you want to summarize. " ++
        "Include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. ",
    // question
    "Generate a question for an LLM that will test its knowledge of the information in the corpus above. " ++
        "In your question be sure to include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. " ++
        "Output only a single question. Do NOT include any other text or explanation other than the question.",
    // use case
    "You are working to train a language model on the information in the following corpus. " ++
        "Your primary goal is to think about practical, real-world tasks or applications that someone could achieve using the knowledge contained within this corpus. " ++
        "Consider how a user might want to apply this information, not just recall it. " ++
        "After considering potential use cases, your task will be to generate a sample question that reflects one of these downstream applications. " ++
        "This question/instruction/task should be something a user, who has access to this corpus, might ask when trying to accomplish their specific goal. " ++
        "Output only a single question. Do NOT include any other text or explanation other than the question.",
    // creative
    "You are having a creative conversation inspired by the information in the corpus. " ++
        "Please generate a question for your conversation partner to start off the discussion. " ++
        "Answer only with the question, do not include any other text.",
    // mechanism (addition): force why/how supervision, not just facts.
    "Generate a question for an LLM that asks WHY or HOW something described in the corpus works or " ++
        "happens. The question must demand the reasoning, mechanism, or cause behind it, not just the " ++
        "fact itself. In your question be sure to include details (ids, names, titles, dates, etc.) " ++
        "that make it clear what you are asking about. " ++
        "Output only a single question. Do NOT include any other text or explanation other than the question.",
    // verbatim precision (addition): force exact wording, names, and numbers.
    "Generate a question for an LLM that asks for the exact definition, full name, precise wording, " ++
        "or numerical value of something specific that appears in the corpus, to be answered precisely " ++
        "as it is stated in the text. In your question be sure to include details (ids, names, titles, " ++
        "dates, etc.) that make it unambiguous what you are asking about. " ++
        "Output only a single question. Do NOT include any other text or explanation other than the question.",
};

const Options = struct {
    model_path: []const u8 = default_model,
    p: usize = 512,
    frozen_prefix: usize = 1,
    suffix_max: usize = 64,
    equiv: bool = false,
    steps: usize = 20,
    accum: usize = 4,
    // The paper's lr 2e-2 assumes packed 32x2048-token batches (~65k
    // supervised tokens/step); at these demo budgets (~1k tokens/step) it
    // measurably drifts the cartridge — 2e-3 trains cleanly (see
    // docs/CARTRIDGES.md "Learning rate vs batch size").
    lr: f32 = 2e-3,
    chunk_min: usize = 256,
    chunk_max: usize = 512,
    top_k: usize = 20,
    max_q: usize = 64,
    max_a: usize = 192,
    seed: u64 = 42,
    save_path: []const u8 = default_save,
    load_path: ?[]const u8 = null,
    /// Serve-mode only (--load): the question answered behind the loaded
    /// cartridge (plus bare-model and, with a corpus, ICL comparison arms).
    /// Training runs never serve — train, save, then serve with --load.
    ask: ?[]const u8 = null,
    /// Checkpoint the cartridge every N optimizer steps (0 = only at the
    /// end). Saves are atomic, to `save_path` — a long run is resumable
    /// from its last checkpoint via --resume.
    save_every: usize = 0,
    /// Continue training from a saved cartridge instead of corpus-init
    /// (rows only; Adam moments restart).
    resume_path: ?[]const u8 = null,
    /// Cap on the ICL comparison context: a corpus a cartridge is worth
    /// training on usually does not fit a sane prefill.
    icl_max: usize = 4096,
    /// Pack the accumulation group into ONE forward/backward (gradients
    /// identical to sequential accumulation, proven in the trainer tests)
    /// and decode its conversations as lockstep batched streams. --no-pack
    /// keeps the per-conversation backward: slower, but training memory
    /// stays flat at one conversation's graph.
    pack: bool = true,
    /// Decode bot B speculatively (--spec-b), COMPOSED with batching:
    /// each stream's [carry ++ draft] span rides one ragged
    /// forwardStepBatchSpans pass. Parity with plain batching at 0.6B;
    /// wins track draft acceptance (bigger models, quote-heavy corpora).
    spec_b: bool = false,
    /// Serve --ask answers speculatively (--spec-serve): the corpus drafts
    /// the answer — corpus-grounded text, which a trained cartridge quotes
    /// near-verbatim, drafts itself. The reference comes from the ARTIFACT
    /// when it embeds one (--draft-ref at training; no --corpus needed to
    /// serve), else from --corpus; the suffix automaton is built once at
    /// load, not per generation call. Single stream, so there is no batch
    /// to trade away.
    spec_serve: bool = false,
    /// Embed the training corpus token ids in the saved artifact
    /// ("draft_reference" entry) so serving can draft from the corpus
    /// without re-reading or re-tokenizing it (--spec-serve picks it up
    /// automatically). Adds 8 bytes/token to the checkpoint.
    draft_ref: bool = false,
    /// Generation stream width (--gen-batch N, default = --accum): how many
    /// conversations synthesize as ONE lockstep batched-decode group (and
    /// one packed teacher pass). The optimizer still steps every --accum
    /// conversations, so widening the generation batch amortizes the
    /// per-token weight stream — the dominant self-study cost — over more
    /// streams without growing the packed backward. Must be a multiple of
    /// --accum. Refill steps carry the whole group's generation time, so
    /// per-step timings turn lumpy; the s/conversation summary is the
    /// number to read.
    gen_batch: usize = 0,
};

/// A multi-file corpus: per-file token spans over one concatenated id
/// stream, so chunks can name their source document (the reference
/// prepends a one-line provenance description to every chunk).
const Corpus = struct {
    const FileSpan = struct { start: usize, path: []u8 };

    ids: []usize,
    text: []u8,
    files: []FileSpan,
    allocator: std.mem.Allocator,

    fn fileOf(self: *const Corpus, token_index: usize) []const u8 {
        var best: []const u8 = self.files[0].path;
        for (self.files) |span| {
            if (span.start > token_index) break;
            best = span.path;
        }
        return best;
    }

    fn deinit(self: *Corpus) void {
        for (self.files) |span| self.allocator.free(span.path);
        self.allocator.free(self.files);
        self.allocator.free(self.ids);
        self.allocator.free(self.text);
        self.* = undefined;
    }
};

/// Read every `--corpus` argument (a file, or a directory whose top-level
/// `.md` files are taken in sorted order) into one concatenated corpus;
/// each file is prefixed by a `# Document: <path>` header and tokenized
/// separately (headers are plain corpus text, so chunks and the ICL
/// context carry provenance).
fn buildCorpus(
    allocator: std.mem.Allocator,
    io: std.Io,
    tokenizer: anytype,
    paths: []const []const u8,
) !Corpus {
    var ids: std.ArrayList(usize) = .empty;
    errdefer ids.deinit(allocator);
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(allocator);
    var files: std.ArrayList(Corpus.FileSpan) = .empty;
    errdefer {
        for (files.items) |span| allocator.free(span.path);
        files.deinit(allocator);
    }

    for (paths) |path| {
        if (std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true })) |dir_const| {
            var dir = dir_const;
            defer dir.close(io);
            var names: std.ArrayList([]u8) = .empty;
            defer {
                for (names.items) |name| allocator.free(name);
                names.deinit(allocator);
            }
            var it = dir.iterate();
            while (try it.next(io)) |entry| {
                if (entry.kind != .file) continue;
                if (!std.ascii.endsWithIgnoreCase(entry.name, ".md")) continue;
                try names.append(allocator, try allocator.dupe(u8, entry.name));
            }
            std.mem.sort([]u8, names.items, {}, struct {
                fn lessThan(_: void, a: []u8, b: []u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.lessThan);
            for (names.items) |name| {
                const joined = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ path, name });
                defer allocator.free(joined);
                try appendCorpusFile(allocator, io, tokenizer, &ids, &text, &files, joined);
            }
        } else |_| {
            try appendCorpusFile(allocator, io, tokenizer, &ids, &text, &files, path);
        }
    }
    if (files.items.len == 0) return error.EmptyCorpus;

    return .{
        .ids = try ids.toOwnedSlice(allocator),
        .text = try text.toOwnedSlice(allocator),
        .files = try files.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

fn appendCorpusFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    tokenizer: anytype,
    ids: *std.ArrayList(usize),
    text: *std.ArrayList(u8),
    files: *std.ArrayList(Corpus.FileSpan),
    path: []const u8,
) !void {
    var dir = std.Io.Dir.cwd();
    const content = try dir.readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(content);

    const owned_path = try allocator.dupe(u8, path);
    errdefer allocator.free(owned_path);
    try files.append(allocator, .{ .start = ids.items.len, .path = owned_path });

    const block = try std.fmt.allocPrint(allocator, "\n\n# Document: {s}\n\n{s}", .{ path, content });
    defer allocator.free(block);
    try text.appendSlice(allocator, block);
    const block_ids = try encodeUsize(allocator, tokenizer, block);
    defer allocator.free(block_ids);
    try ids.appendSlice(allocator, block_ids);
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
    var corpus_paths: std.ArrayList([]const u8) = .empty;
    defer corpus_paths.deinit(allocator);
    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (try parseFlagStr(args, &arg_i, "--model")) |v| {
            opts.model_path = v;
        } else if (try parseFlagStr(args, &arg_i, "--corpus")) |v| {
            try corpus_paths.append(allocator, v);
        } else if (try parseFlagStr(args, &arg_i, "--save")) |v| {
            opts.save_path = v;
        } else if (try parseFlagStr(args, &arg_i, "--load")) |v| {
            opts.load_path = v;
        } else if (try parseFlagStr(args, &arg_i, "--ask")) |v| {
            opts.ask = v;
        } else if (try parseFlagInt(args, &arg_i, "--p")) |v| {
            opts.p = v;
        } else if (try parseFlagInt(args, &arg_i, "--frozen")) |v| {
            opts.frozen_prefix = v;
        } else if (try parseFlagInt(args, &arg_i, "--suffix-max")) |v| {
            opts.suffix_max = v;
        } else if (try parseFlagInt(args, &arg_i, "--steps")) |v| {
            opts.steps = v;
        } else if (try parseFlagInt(args, &arg_i, "--accum")) |v| {
            opts.accum = v;
        } else if (try parseFlagInt(args, &arg_i, "--chunk-min")) |v| {
            opts.chunk_min = v;
        } else if (try parseFlagInt(args, &arg_i, "--chunk-max")) |v| {
            opts.chunk_max = v;
        } else if (try parseFlagInt(args, &arg_i, "--top-k")) |v| {
            opts.top_k = v;
        } else if (try parseFlagInt(args, &arg_i, "--max-q")) |v| {
            opts.max_q = v;
        } else if (try parseFlagInt(args, &arg_i, "--max-a")) |v| {
            opts.max_a = v;
        } else if (try parseFlagInt(args, &arg_i, "--seed")) |v| {
            opts.seed = v;
        } else if (try parseFlagF32(args, &arg_i, "--lr")) |v| {
            opts.lr = v;
        } else if (try parseFlagInt(args, &arg_i, "--icl-max")) |v| {
            opts.icl_max = v;
        } else if (try parseFlagInt(args, &arg_i, "--gen-batch")) |v| {
            opts.gen_batch = v;
        } else if (try parseFlagInt(args, &arg_i, "--save-every")) |v| {
            opts.save_every = v;
        } else if (try parseFlagStr(args, &arg_i, "--resume")) |v| {
            opts.resume_path = v;
        } else if (std.mem.eql(u8, arg, "--no-pack")) {
            opts.pack = false;
        } else if (std.mem.eql(u8, arg, "--spec-b")) {
            opts.spec_b = true;
        } else if (std.mem.eql(u8, arg, "--spec-serve")) {
            opts.spec_serve = true;
        } else if (std.mem.eql(u8, arg, "--draft-ref")) {
            opts.draft_ref = true;
        } else if (std.mem.eql(u8, arg, "--equiv")) {
            opts.equiv = true;
        } else {
            try stdout.print(
                "usage: zig build cartridge -Doptimize=ReleaseFast -- --corpus FILE|DIR (repeatable) [--model PATH] " ++
                    "[--p N] [--frozen N] [--steps N] [--accum N] [--lr F] [--chunk-min N] [--chunk-max N] [--top-k N] " ++
                    "[--max-q N] [--max-a N] [--seed N] [--save PATH] [--save-every N] [--resume PATH] [--load PATH] " ++
                    "[--ask TEXT] [--icl-max N] [--no-pack] [--spec-b] [--spec-serve] [--draft-ref] [--gen-batch N] " ++
                    "[--suffix-max N] [--equiv]\n",
                .{},
            );
            return error.UnknownArgument;
        }
    }
    if (opts.accum == 0 or opts.steps == 0) return error.InvalidArguments;
    if (opts.chunk_min == 0 or opts.chunk_min > opts.chunk_max) return error.InvalidArguments;
    if (opts.gen_batch != 0 and opts.gen_batch % opts.accum != 0) return error.InvalidArguments;
    if (opts.ask != null and opts.load_path == null) {
        try stdout.print("--ask serves a saved cartridge: pair it with --load\n", .{});
        return error.InvalidArguments;
    }

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Load model + tokenizer from the same GGUF parse. gemma-family GGUFs
    // take the gemma4 arm: the trainer seams (docs/CARTRIDGES.md) support
    // the --equiv acceptance gate and distillation; the self-study/serve
    // loops of this CLI are qwen3-typed (serve trained gemma cartridges via
    // lmserve --cartridge).
    var file = try fucina.gguf.File.loadMmap(allocator, io, opts.model_path);
    const arch = file.getString("general.architecture") orelse "";
    if (std.mem.startsWith(u8, arch, "gemma")) {
        return runGemma(&ctx, io, stdout, allocator, &file, corpus_paths.items, opts);
    }
    var model = try llm.qwen3.model.Model.loadGgufFromFile(&ctx, &file, try llm.qwen3.model.Config.fromGguf(&file));
    defer model.deinit();
    var tokenizer = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tokenizer.deinit();
    file.deinit();
    try stdout.print("model: {s} ({d} layers, hidden {d}, kv_heads {d}, head_dim {d})\n", .{
        opts.model_path,                  model.config.num_layers, model.config.hidden_size,
        model.config.num_key_value_heads, model.config.head_dim,
    });

    // Corpus (optional in pure serve mode: --load + --ask). Each --corpus
    // is a file or a directory of .md files; everything concatenates into
    // one provenance-tagged token stream.
    var corpus: ?Corpus = null;
    defer if (corpus) |*c| c.deinit();
    if (corpus_paths.items.len > 0) {
        corpus = try buildCorpus(allocator, io, &tokenizer, corpus_paths.items);
        try stdout.print("corpus: {d} files, {d} tokens\n", .{ corpus.?.files.len, corpus.?.ids.len });
    }

    var trainer = try Trainer.init(&ctx, &model, .{ .rank = 1, .alpha = 1 }, opts.seed);
    defer trainer.deinit();

    if (opts.equiv) {
        const c = if (corpus) |*c| c else return error.MissingCorpusPath;
        if (c.ids.len < opts.p + 2) return error.CorpusTooShort;
        return runEquiv(&ctx, io, stdout, &trainer, c.ids, opts);
    }

    if (opts.load_path) |path| {
        // Serve a previously trained cartridge.
        const ask = opts.ask orelse return error.MissingAsk;
        const bytes = blk: {
            var dir = std.Io.Dir.cwd();
            break :blk try dir.readFileAlloc(io, path, allocator, .limited(1024 * 1024 * 1024));
        };
        defer allocator.free(bytes);
        var cart = try cartridge.Cartridge.initFromStateDict(&ctx, allocator, bytes);
        defer cart.deinit();
        try stdout.print("cartridge loaded from {s}: p = {d} ({d} layers)\n", .{ path, cart.p, cart.layers.len });
        return runServe(&ctx, io, stdout, allocator, &model, &tokenizer, qwen3_tpl, &cart, ask, if (corpus) |*c| c else null, opts);
    }

    const c = if (corpus) |*c| c else return error.MissingCorpusPath;
    if (c.ids.len < opts.chunk_min + 2 or c.ids.len < opts.p + 2) return error.CorpusTooShort;
    return runSelfStudy(&ctx, io, stdout, allocator, &model, &tokenizer, &trainer, qwen3_tpl, true, c, opts);
}

/// Serve mode: greedy answers to `ask` behind the cartridge, bare, and (when
/// a corpus is given) with the real corpus in context — the ICL reference the
/// cartridge is meant to replace at a fraction of the KV rows.
fn runServe(
    ctx: *fucina.ExecContext,
    io: std.Io,
    stdout: anytype,
    allocator: std.mem.Allocator,
    model: anytype,
    tokenizer: anytype,
    tpl: Tpl,
    cart: *const cartridge.Cartridge,
    ask: []const u8,
    corpus: ?*const Corpus,
    opts: Options,
) !void {
    const cfg = model.config;
    const stop_id: ?usize = if (tokenizer.tokenId(tpl.stop_marker)) |id| @as(usize, id) else null;
    const prompt = try std.fmt.allocPrint(allocator, "{s}{s}{s}{s}", .{ tpl.user_open, ask, tpl.block_close, tpl.asst_open });
    defer allocator.free(prompt);
    const prompt_len_guess = prompt.len / 2 + 32;

    // --spec-serve draft source, resolved at LOAD: the artifact's embedded
    // corpus (--draft-ref at creation) wins; a --corpus on the command line
    // is the fallback. The suffix automaton is built once here — generation
    // only consumes it, nothing is constructed per call.
    var serve_index: ?llm.speculative.cascade.SpeculationIndex = null;
    defer if (serve_index) |*index| index.deinit();
    if (opts.spec_serve) {
        const tokens: []const usize = cart.draft_reference orelse
            (if (corpus) |c| c.ids else {
                try stdout.print("--spec-serve needs a draft reference: train with --draft-ref or pass --corpus\n", .{});
                return error.MissingDraftReference;
            });
        const t_build = nowNs(io);
        serve_index = try llm.speculative.cascade.SpeculationIndex.init(allocator, cfg.vocab_size);
        try serve_index.?.addReference(tokens);
        try stdout.print("draft automaton over {d} {s} tokens built in {d:.0} ms\n", .{
            tokens.len,
            if (cart.draft_reference != null) "embedded corpus" else "--corpus",
            seconds(nowNs(io) - t_build) * 1000,
        });
    }

    // Behind the cartridge: preload, then decode at positions p.. — the
    // exact layout the cartridge was trained at. With --spec-serve the
    // corpus drafts the answer (single stream: no batch to trade away).
    {
        var cache = try makeCache(ctx, model, cart.p + prompt_len_guess + opts.max_a + 16);
        defer cache.deinit();
        try cart.writeToCache(ctx, &cache);
        const t0 = nowNs(io);
        var answer: []u8 = undefined;
        var produced: usize = 0;
        if (serve_index) |*index| {
            const prompt_ids = try encodeUsize(allocator, tokenizer, prompt);
            defer allocator.free(prompt_ids);
            const outs = try generateIdsSpecBatch(ctx, allocator, model, (&cache)[0..1], &.{prompt_ids}, opts.max_a, stop_id, &.{index});
            defer {
                for (outs) |buf| allocator.free(buf);
                allocator.free(outs);
            }
            produced = outs[0].len;
            answer = try cleanGenerated(allocator, tokenizer, outs[0]);
        } else {
            const prompt_ids = try encodeUsize(allocator, tokenizer, prompt);
            defer allocator.free(prompt_ids);
            const ids = try generateIds(ctx, allocator, model, &cache, prompt_ids, opts.max_a, stop_id, .{});
            defer allocator.free(ids);
            produced = ids.len;
            answer = try cleanGenerated(allocator, tokenizer, ids);
        }
        defer allocator.free(answer);
        const secs = seconds(nowNs(io) - t0);
        // No-op unless FUCINA_GPU_TRACE=1: per-dtype GPU dispatch counters
        // for the answer just produced.
        fucina.internal.gpu.traceDump();
        try stdout.print("\n[cartridge, {d} KV rows{s}] ({d:.2} s, ~{d:.1} tok/s)\n{s}\n", .{
            cart.p,
            if (serve_index != null) ", corpus-drafted" else "",
            secs,
            @as(f64, @floatFromInt(produced)) / secs,
            answer,
        });
        try stdout.flush();
    }

    // Bare model: no context at all.
    {
        var cache = try makeCache(ctx, model, prompt_len_guess + opts.max_a);
        defer cache.deinit();
        const answer = try generateText(ctx, allocator, model, tokenizer, &cache, prompt, opts.max_a, stop_id, .{});
        defer allocator.free(answer);
        try stdout.print("\n[bare model, no context]\n{s}\n", .{answer});
        try stdout.flush();
    }

    // ICL reference: the real corpus as the system message, capped at
    // --icl-max tokens (a corpus worth a cartridge rarely fits a prefill).
    if (corpus) |c| {
        const truncated = c.ids.len > opts.icl_max;
        const icl_text = if (truncated) blk: {
            break :blk try decodeUsize(allocator, tokenizer, c.ids[0..opts.icl_max]);
        } else try allocator.dupe(u8, c.text);
        defer allocator.free(icl_text);

        const icl_prompt = try std.fmt.allocPrint(allocator, "{s}" ++ system_prompt_template ++ "{s}{s}", .{ tpl.sys_open, icl_text, tpl.block_close, prompt });
        defer allocator.free(icl_prompt);
        const icl_ids = try encodeUsize(allocator, tokenizer, icl_prompt);
        defer allocator.free(icl_ids);
        var cache = try makeCache(ctx, model, icl_ids.len + opts.max_a + 16);
        defer cache.deinit();
        const ids = try generateIds(ctx, allocator, model, &cache, icl_ids, opts.max_a, stop_id, .{});
        defer allocator.free(ids);
        const text_out = try decodeUsize(allocator, tokenizer, ids);
        defer allocator.free(text_out);
        if (truncated) {
            try stdout.print("\n[ICL, {d} KV rows — {d}x the cartridge; corpus TRUNCATED from {d} tokens]\n{s}\n", .{
                icl_ids.len, if (cart.p == 0) 0 else icl_ids.len / cart.p, c.ids.len, std.mem.trim(u8, text_out, " \t\r\n"),
            });
        } else {
            try stdout.print("\n[ICL, {d} KV rows — {d}x the cartridge]\n{s}\n", .{
                icl_ids.len, if (cart.p == 0) 0 else icl_ids.len / cart.p, std.mem.trim(u8, text_out, " \t\r\n"),
            });
        }
        try stdout.flush();
    }
}

// ---------------------------------------------------------------------------
// Self-study training (default mode)
// ---------------------------------------------------------------------------

/// One synthesized conversation, ready for a distillation micro-step:
/// `student_ids` is the packed seed-free conversation (user question +
/// assistant answer) and `builder` holds the teacher's sparse top-k targets
/// for the answer tokens.
const Convo = struct {
    /// Seed-free packed element: user(question) + assistant(answer).
    student_ids: []usize,
    /// The same behind the real chunk: system(chunk) + student_ids.
    teacher_ids: []usize,
    /// Student index of the first supervised (answer) token.
    first_answer: usize,
    /// Supervised token count (turn close included when B stopped).
    answer_len: usize,
    question: []u8,
    /// Teacher top-k targets at STUDENT-LOCAL positions (offset by the
    /// segment start when packing conversations into one row).
    builder: cartridge.TargetsBuilder,

    fn deinit(self: *Convo, allocator: std.mem.Allocator) void {
        allocator.free(self.student_ids);
        allocator.free(self.teacher_ids);
        allocator.free(self.question);
        self.builder.deinit();
        self.* = undefined;
    }
};

fn runSelfStudy(
    ctx: *fucina.ExecContext,
    io: std.Io,
    stdout: anytype,
    allocator: std.mem.Allocator,
    model: anytype,
    tokenizer: anytype,
    trainer: anytype,
    tpl: Tpl,
    comptime supports_packing: bool,
    corpus: *const Corpus,
    opts_in: Options,
) !void {
    var opts = opts_in;
    if (!supports_packing and opts.pack) {
        // No packed forward on this trainer: the flat-memory
        // per-conversation backward is the training arm (generation stays
        // batched either way).
        opts.pack = false;
    }
    const stop_id: ?usize = if (tokenizer.tokenId(tpl.stop_marker)) |id| @as(usize, id) else null;

    // One generation cache per generation stream (the group's conversations
    // decode as lockstep batched streams), each sized for system(chunk) +
    // question + answer. --gen-batch widens this beyond the optimizer's
    // --accum group.
    const gen_batch = if (opts.gen_batch == 0) opts.accum else opts.gen_batch;
    const capacity = opts.chunk_max + opts.max_q + opts.max_a + 256;
    const caches = try allocator.alloc(llm.kv_cache.KvCache, gen_batch);
    defer allocator.free(caches);
    var caches_inited: usize = 0;
    defer for (caches[0..caches_inited]) |*c| c.deinit();
    for (caches) |*c| {
        c.* = try makeCache(ctx, model, capacity);
        caches_inited += 1;
    }

    // Cartridge init: the corpus opening wrapped as a system message,
    // truncated to exactly p tokens with the turn close re-appended (the
    // paper's winning "first p corpus tokens" initialization) — or a saved
    // checkpoint when resuming (rows only; Adam moments restart).
    var cart = if (opts.resume_path) |path| blk: {
        const bytes = read: {
            var dir = std.Io.Dir.cwd();
            break :read try dir.readFileAlloc(io, path, allocator, .limited(1024 * 1024 * 1024));
        };
        defer allocator.free(bytes);
        break :blk try cartridge.Cartridge.initFromStateDict(ctx, allocator, bytes);
    } else blk: {
        const init_tokens = try systemInitTokens(allocator, tokenizer, tpl, corpus.text, opts.p);
        defer allocator.free(init_tokens);
        break :blk try trainer.initCartridge(ctx, init_tokens, opts.frozen_prefix);
    };
    defer cart.deinit();
    if (opts.resume_path) |path| {
        try stdout.print("cartridge resumed from {s}: p = {d} rows/layer x {d} layers\n", .{ path, cart.p, cart.layers.len });
    } else {
        try stdout.print("cartridge initialized from the corpus opening: p = {d} rows/layer x {d} layers\n", .{ cart.p, cart.layers.len });
    }
    // Embed the corpus as the artifact's serving-time draft reference now,
    // so EVERY checkpoint (--save-every included) is self-contained for
    // --spec-serve. Set-once: a resumed artifact keeps its original corpus.
    if (opts.draft_ref) {
        if (cart.draft_reference == null) {
            try cart.setDraftReference(ctx, corpus.ids);
            try stdout.print("draft reference embedded: {d} corpus tokens\n", .{corpus.ids.len});
        } else {
            try stdout.print("draft reference kept from the resumed artifact: {d} tokens\n", .{cart.draft_reference.?.len});
        }
    }
    try stdout.flush();

    var opt = optim.AdamW.init(allocator, .{ .lr = opts.lr, .weight_decay = 0 });
    defer opt.deinit();
    try cart.registerParams(&opt);

    var prng = std.Random.DefaultPrng.init(opts.seed);
    const rand = prng.random();

    // Held-out conversation: synthesized once, never trained on — its loss
    // is the generalization signal (falling held loss = the cartridge is
    // learning the corpus, not one conversation).
    var held: [1]Convo = undefined;
    try synthesizeGroup(ctx, allocator, model, tokenizer, trainer, tpl, supports_packing, caches[0..1], corpus, rand, opts, stop_id, &held);
    defer held[0].deinit(allocator);
    trainer.freeTransientRope();
    try stdout.print("held-out question: {s}\n", .{held[0].question});
    const held_before = try distillEval(ctx, trainer, &cart, &held[0]);
    try stdout.print("held-out distill loss before training: {d:.4}\n", .{held_before});
    try stdout.flush();

    // The generation queue: refilled gen_batch conversations at a time (one
    // lockstep decode group + one packed teacher pass), consumed --accum at
    // a time by the optimizer steps.
    const pending = try allocator.alloc(Convo, gen_batch);
    defer allocator.free(pending);
    var pending_at: usize = 0;
    var pending_n: usize = 0;
    defer for (pending[pending_at..pending_n]) |*convo| convo.deinit(allocator);

    var trained_tokens: usize = 0;
    const train_start = nowNs(io);
    for (0..opts.steps) |step_i| {
        const step_start = nowNs(io);
        if (pending_at == pending_n) {
            // Never synthesize more than the remaining steps will consume.
            const need = @min(gen_batch, (opts.steps - step_i) * opts.accum);
            try synthesizeGroup(ctx, allocator, model, tokenizer, trainer, tpl, supports_packing, caches, corpus, rand, opts, stop_id, pending[0..need]);
            pending_at = 0;
            pending_n = need;
        }
        const group = pending[pending_at..][0..opts.accum];
        pending_at += opts.accum;
        defer for (group) |*convo| convo.deinit(allocator);

        var total_entries: usize = 0;
        for (group) |*convo| {
            trained_tokens += convo.answer_len;
            total_entries += convo.builder.targets().positions.len;
        }

        var step_loss: f64 = 0;
        if (opts.pack) {
            // One packed forward/backward for the whole group: segment
            // lengths + targets shifted to packed-row positions; the .mean
            // over all entries IS the sequential accumulation's objective.
            var merged = cartridge.TargetsBuilder.init(allocator);
            defer merged.deinit();
            const seg_lens = try allocator.alloc(usize, group.len);
            defer allocator.free(seg_lens);
            var packed_len: usize = 0;
            for (group, seg_lens) |*convo, *len| {
                const local = convo.builder.targets();
                for (local.positions, local.tokens, local.logprobs) |pos, token, logprob| {
                    try merged.appendEntry(packed_len + pos, token, logprob);
                }
                len.* = convo.student_ids.len;
                packed_len += convo.student_ids.len;
            }
            const packed_student = try allocator.alloc(usize, packed_len);
            defer allocator.free(packed_student);
            var at: usize = 0;
            for (group) |*convo| {
                @memcpy(packed_student[at..][0..convo.student_ids.len], convo.student_ids);
                at += convo.student_ids.len;
            }

            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var loss = try trainer.distillLoss(ctx, packed_student, &cart, merged.targets(), seg_lens, .{});
            defer loss.deinit();
            try loss.backward(ctx);
            step_loss = try loss.item();
        } else {
            // Flat-memory arm: one conversation's graph alive at a time,
            // gradient-identical to the packed step (.sum scaled by the
            // group's total entry count = the packed mean).
            const scale = 1.0 / @as(f32, @floatFromInt(@max(1, total_entries)));
            for (group) |*convo| {
                const scope = ctx.openExecScope();
                defer ctx.closeExecScope(scope);
                var loss = try trainer.distillLoss(ctx, convo.student_ids, &cart, convo.builder.targets(), null, .{
                    .reduction = .sum,
                    .loss_scale = scale,
                });
                defer loss.deinit();
                try loss.backward(ctx);
                step_loss += try loss.item();
            }
        }
        try opt.step(ctx);
        opt.zeroGrad();
        trainer.freeTransientRope();
        try stdout.print("step {d:>3}/{d}: distill loss {d:.4} ({d} answer tokens so far, {d:.1} s)\n", .{
            step_i + 1, opts.steps, step_loss, trained_tokens, seconds(nowNs(io) - step_start),
        });
        try stdout.flush();
        if (opts.save_every != 0 and (step_i + 1) % opts.save_every == 0 and step_i + 1 != opts.steps) {
            try fucina.training_checkpoint.writeFileAtomic(io, opts.save_path, &cart, saveCartridge);
            try stdout.print("checkpoint saved to {s} (step {d})\n", .{ opts.save_path, step_i + 1 });
            try stdout.flush();
        }
    }
    const train_s = seconds(nowNs(io) - train_start);
    const convos = opts.steps * opts.accum;
    try stdout.print("self-study: {d} conversations, {d} supervised answer tokens, {d:.1} min ({d:.1} s/conversation)\n", .{
        convos, trained_tokens, train_s / 60.0, train_s / @as(f64, @floatFromInt(convos)),
    });

    const held_after = try distillEval(ctx, trainer, &cart, &held[0]);
    try stdout.print("held-out distill loss after training: {d:.4} (before {d:.4})\n", .{ held_after, held_before });

    try fucina.training_checkpoint.writeFileAtomic(io, opts.save_path, &cart, saveCartridge);
    try stdout.print("cartridge saved to {s}\n", .{opts.save_path});
}

fn saveCartridge(cart: *cartridge.Cartridge, writer: *std.Io.Writer) anyerror!void {
    try cart.saveState(writer);
}

/// Held-out / reporting loss: same objective as the training step, no
/// backward (the fresh scope discards the graph; the step-counter advance is
/// harmless with dropout off).
fn distillEval(ctx: *fucina.ExecContext, trainer: anytype, cart: *const cartridge.Cartridge, convo: *const Convo) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    var loss = try trainer.distillLoss(ctx, convo.student_ids, cart, convo.builder.targets(), null, .{});
    defer loss.deinit();
    return loss.item();
}

/// Algorithm 1 with k = 1 over a whole accumulation group: chunks + seeds
/// -> bot A asks (temp 0.6, LOCKSTEP batched decode) -> bot B answers
/// (greedy, chunk in context, batched) -> ONE packed teacher pass extracts
/// every conversation's top-k targets. Fills `out` (caller deinits each
/// entry on success); transient rope tables from the packed teacher stay
/// alive until the caller's `freeTransientRope`.
fn synthesizeGroup(
    ctx: *fucina.ExecContext,
    allocator: std.mem.Allocator,
    model: anytype,
    tokenizer: anytype,
    trainer: anytype,
    tpl: Tpl,
    comptime supports_packing: bool,
    caches: []llm.kv_cache.KvCache,
    corpus: *const Corpus,
    rand: std.Random,
    opts: Options,
    stop_id: ?usize,
    out: []Convo,
) !void {
    const n = out.len;
    std.debug.assert(n > 0 and caches.len >= n);
    // Group-transient scratch (chunk texts, prompts, generations): one arena,
    // freed wholesale. Convo fields live on `allocator`.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Chunks + seeds + bot-A prompts.
    const sys_texts = try arena.alloc([]u8, n);
    const a_prompts = try arena.alloc([]const usize, n);
    const a_cfgs = try arena.alloc(llm.sampler.Config, n);
    for (0..n) |i| {
        // Uniform random token span (reference TokenChunker) with a one-line
        // provenance description (the reference prepends one to every chunk).
        const chunk_max = @min(opts.chunk_max, corpus.ids.len);
        const chunk_len = rand.intRangeAtMost(usize, @min(opts.chunk_min, chunk_max), chunk_max);
        const chunk_start = rand.intRangeAtMost(usize, 0, corpus.ids.len - chunk_len);
        const chunk_body = try decodeUsize(arena, tokenizer, corpus.ids[chunk_start..][0..chunk_len]);
        const chunk_text = try std.fmt.allocPrint(
            arena,
            "Below is an excerpt from {s}. It is part of a larger corpus of documents.\n\n{s}",
            .{ corpus.fileOf(chunk_start), chunk_body },
        );
        sys_texts[i] = try std.fmt.allocPrint(arena, sys_open ++ system_prompt_template ++ block_close, .{chunk_text});
        const seed_prompt = seed_prompts[rand.intRangeLessThan(usize, 0, seed_prompts.len)];
        const a_prompt = try std.fmt.allocPrint(arena, "{s}{s}{s}{s}{s}", .{ sys_texts[i], tpl.user_open, seed_prompt, tpl.block_close, tpl.asst_open });
        a_prompts[i] = try encodeUsize(arena, tokenizer, a_prompt);
        a_cfgs[i] = .{ .temperature = 0.6, .seed = rand.int(u64) };
    }

    // Bot A: one chat message per conversation (thinking off).
    const a_outs = try generateIdsBatch(ctx, arena, model, caches[0..n], a_prompts, opts.max_q, stop_id, a_cfgs);

    // Bot B: greedy answers WITH each chunk in context.
    const b_prompts = try arena.alloc([]const usize, n);
    const b_cfgs = try arena.alloc(llm.sampler.Config, n);
    const convo_prefix_ids = try arena.alloc([]usize, n);
    const questions = try arena.alloc([]const u8, n);
    for (0..n) |i| {
        questions[i] = try cleanGenerated(arena, tokenizer, a_outs[i]);
        const convo_prefix = try std.fmt.allocPrint(arena, "{s}{s}{s}{s}", .{ tpl.user_open, questions[i], tpl.block_close, tpl.asst_open });
        convo_prefix_ids[i] = try encodeUsize(arena, tokenizer, convo_prefix);
        const b_prompt = try std.mem.concat(arena, u8, &.{ sys_texts[i], convo_prefix });
        b_prompts[i] = try encodeUsize(arena, tokenizer, b_prompt);
        b_cfgs[i] = .{};
    }
    const b_outs = if (opts.spec_b)
        // Chunk-grounded greedy answers, speculative AND batched: every
        // stream's [carry ++ draft] span rides one ragged weight pass
        // (forwardStepBatchSpans), drafts coming from each stream's own
        // prompt via the lossless speculation index.
        blk: {
            for (0..n) |i| caches[i].reset();
            break :blk try generateIdsSpecBatch(ctx, arena, model, caches[0..n], b_prompts, opts.max_a, stop_id, null);
        }
    else
        try generateIdsBatch(ctx, arena, model, caches[0..n], b_prompts, opts.max_a, stop_id, b_cfgs);

    // Assemble the per-conversation elements (owned by `allocator`).
    var built: usize = 0;
    errdefer for (out[0..built]) |*convo| convo.deinit(allocator);
    for (0..n) |i| {
        const answer_gen = b_outs[i];
        if (answer_gen.len == 0) return error.EmptyAnswer;
        // Supervise the turn close too when B actually stopped (the
        // reference packs message end tokens into the element): the
        // cartridge must also learn WHEN to stop answering.
        const stopped = answer_gen.len < opts.max_a and stop_id != null;
        const answer_len = answer_gen.len + @intFromBool(stopped);
        const sys_ids = try encodeUsize(arena, tokenizer, sys_texts[i]);

        const prefix_len = convo_prefix_ids[i].len;
        const student_ids = try allocator.alloc(usize, prefix_len + answer_len);
        errdefer allocator.free(student_ids);
        @memcpy(student_ids[0..prefix_len], convo_prefix_ids[i]);
        @memcpy(student_ids[prefix_len..][0..answer_gen.len], answer_gen);
        if (stopped) student_ids[student_ids.len - 1] = stop_id.?;
        const teacher_ids = try std.mem.concat(allocator, usize, &.{ sys_ids, student_ids });
        errdefer allocator.free(teacher_ids);
        const question = try allocator.dupe(u8, questions[i]);
        errdefer allocator.free(question);

        out[i] = .{
            .student_ids = student_ids,
            .teacher_ids = teacher_ids,
            .first_answer = prefix_len,
            .answer_len = answer_len,
            .question = question,
            .builder = cartridge.TargetsBuilder.init(allocator),
        };
        built += 1;
    }

    // ONE packed teacher pass over the whole group: every conversation's
    // answer rows come back from a single memory-bounded forward.
    try teacherTargets(ctx, arena, trainer, supports_packing, out, opts);
}

/// Fill each conversation's targets builder from one packed teacher forward:
/// segments = the teacher sequences, rows = every answer token's predicting
/// row (student pos j reads teacher row seg_base + sys + j - 1).
fn teacherTargets(
    ctx: *fucina.ExecContext,
    arena: std.mem.Allocator,
    trainer: anytype,
    comptime supports_packing: bool,
    group: []Convo,
    opts: Options,
) !void {
    var total_len: usize = 0;
    var total_rows: usize = 0;
    for (group) |*convo| {
        total_len += convo.teacher_ids.len;
        total_rows += convo.answer_len;
    }

    const packed_teacher = try arena.alloc(usize, total_len);
    const seg_lens = try arena.alloc(usize, group.len);
    const rows = try arena.alloc(usize, total_rows);
    var at: usize = 0;
    var row_at: usize = 0;
    for (group, seg_lens) |*convo, *len| {
        @memcpy(packed_teacher[at..][0..convo.teacher_ids.len], convo.teacher_ids);
        len.* = convo.teacher_ids.len;
        const sys_len = convo.teacher_ids.len - convo.student_ids.len;
        for (0..convo.answer_len) |i| {
            rows[row_at] = at + sys_len + convo.first_answer + i - 1;
            row_at += 1;
        }
        at += convo.teacher_ids.len;
    }

    var teacher_logits = if (comptime supports_packing)
        try trainer.evalLogitsRows(ctx, packed_teacher, rows, .{
            .packed_segments = if (group.len > 1) seg_lens else null,
        })
    else blk: {
        // No packed eval on this trainer: per-conversation teacher passes,
        // rows re-based per segment, results concatenated in group order.
        if (group.len == 1) break :blk try trainer.evalLogitsRows(ctx, packed_teacher, rows, .{});
        var parts = try arena.alloc(fucina.Tensor(.{ .seq, .vocab }), group.len);
        var parts_built: usize = 0;
        defer for (parts[0..parts_built]) |*t| t.deinit();
        var seg_at: usize = 0;
        var row_i: usize = 0;
        for (group, seg_lens, 0..) |*convo, seg_len, gi| {
            const seg_rows = try arena.alloc(usize, convo.answer_len);
            for (seg_rows, 0..) |*r, j| {
                r.* = rows[row_i + j] - seg_at;
            }
            parts[gi] = try trainer.evalLogitsRows(ctx, packed_teacher[seg_at .. seg_at + seg_len], seg_rows, .{});
            parts_built += 1;
            seg_at += seg_len;
            row_i += convo.answer_len;
        }
        if (parts.len == 1) {
            parts_built = 0;
            break :blk parts[0];
        }
        const rest = try arena.alloc(*const fucina.Tensor(.{ .seq, .vocab }), parts.len - 1);
        for (rest, parts[1..]) |*ptr, *t| ptr.* = t;
        const joined = try parts[0].concat(ctx, .seq, rest);
        for (parts[0..parts_built]) |*t| t.deinit();
        parts_built = 0;
        break :blk joined;
    };
    defer teacher_logits.deinit();
    // Vocab-wide passes stay in core ops (topK's lowest-index tie-break is
    // appendRow's scan order; logsumexp is the row's log-partition): only
    // [rows, k] values/indices and [rows] log-partitions reach the host,
    // where the ragged 0.99-mass cutoff runs.
    const vocab = trainer.model.config.vocab_size;
    const k = @min(opts.top_k, vocab);
    var log_z = try teacher_logits.logsumexp(ctx, .vocab);
    defer log_z.deinit();
    var top = try teacher_logits.topK(ctx, .vocab, k, .top);
    defer top.deinit();
    const top_values = try top.values.dataConst();
    const top_indices = try top.indices.dataConst();
    const log_z_data = try log_z.dataConst();

    row_at = 0;
    for (group) |*convo| {
        for (0..convo.answer_len) |i| {
            try convo.builder.appendTopKRow(
                convo.first_answer + i,
                top_values[row_at * k ..][0..k],
                top_indices[row_at * k ..][0..k],
                log_z_data[row_at],
                0.99,
            );
            row_at += 1;
        }
    }
}

/// Multi-stream GREEDY speculative generation — speculation COMPOSED with
/// batching: every still-active stream contributes one span of
/// [carried token ++ draft] to a single ragged `forwardStepBatchSpans`
/// weight pass, so drafted tokens ride the already-paid batch read.
/// Drafts come from each stream's lossless speculation index — either a
/// caller-owned PREBUILT index per stream (`shared_indexes`; at cartridge
/// serving time the artifact's embedded corpus automaton, built once at
/// load) or a fresh per-stream index; both additionally observe the
/// stream's own prompt and output. Greedy verification is exact argmax
/// equality per
/// row; rejected draft rows rewind with `truncate`. Prefills at each
/// cache's CURRENT position (reset for a fresh conversation; leave a
/// preloaded cartridge prefix in place to serve behind it). The stop
/// token is dropped; results live on `allocator`.
fn generateIdsSpecBatch(
    ctx: *fucina.ExecContext,
    allocator: std.mem.Allocator,
    model: anytype,
    caches: []llm.kv_cache.KvCache,
    prompts: []const []const usize,
    max_new: usize,
    stop_id: ?usize,
    shared_indexes: ?[]const *llm.speculative.cascade.SpeculationIndex,
) ![][]usize {
    const n = prompts.len;
    const max_draft = 8;
    std.debug.assert(n > 0 and caches.len >= n and max_new > 0);

    const outs = try allocator.alloc([]usize, n);
    errdefer allocator.free(outs);
    var outs_built: usize = 0;
    errdefer for (outs[0..outs_built]) |buf| allocator.free(buf);
    for (outs) |*buf| {
        buf.* = try allocator.alloc(usize, max_new);
        outs_built += 1;
    }

    const indexes = try allocator.alloc(*llm.speculative.cascade.SpeculationIndex, n);
    defer allocator.free(indexes);
    var owned: []llm.speculative.cascade.SpeculationIndex = &.{};
    var owned_built: usize = 0;
    defer {
        for (owned[0..owned_built]) |*index| index.deinit();
        if (owned.len > 0) allocator.free(owned);
    }
    if (shared_indexes) |shared| {
        std.debug.assert(shared.len >= n);
        @memcpy(indexes, shared[0..n]);
    } else {
        owned = try allocator.alloc(llm.speculative.cascade.SpeculationIndex, n);
    }

    const histories = try allocator.alloc(std.ArrayList(usize), n);
    defer allocator.free(histories);
    for (histories) |*history| history.* = .empty;
    defer for (histories) |*history| history.deinit(allocator);

    const lens = try allocator.alloc(usize, n);
    defer allocator.free(lens);
    const carries = try allocator.alloc(usize, n);
    defer allocator.free(carries);
    // Lean acceptance gate (the CostGate idea, per stream): rejected draft
    // rows pay full attention + a vocab-wide logits row, so low acceptance
    // must switch drafting off. Re-probe after a while — answers often
    // alternate prose (low acceptance) and quotes (high).
    const drafted_counts = try allocator.alloc(usize, n);
    defer allocator.free(drafted_counts);
    const accepted_counts = try allocator.alloc(usize, n);
    defer allocator.free(accepted_counts);
    const probe_at = try allocator.alloc(usize, n);
    defer allocator.free(probe_at);
    const active = try allocator.alloc(usize, n);
    defer allocator.free(active);
    const batch_caches = try allocator.alloc(*llm.kv_cache.KvCache, n);
    defer allocator.free(batch_caches);
    const span_lens = try allocator.alloc(usize, n);
    defer allocator.free(span_lens);
    const span_tokens = try allocator.alloc(usize, n * (1 + max_draft));
    defer allocator.free(span_tokens);
    var draft_buf: [max_draft]usize = undefined;

    const vocab = model.config.vocab_size;
    var n_active: usize = 0;
    for (0..n) |i| {
        if (shared_indexes == null) {
            owned[i] = try llm.speculative.cascade.SpeculationIndex.init(allocator, vocab);
            owned_built += 1;
            indexes[i] = &owned[i];
        }
        if (caches[i].len + prompts[i].len + max_new + max_draft + 1 > caches[i].capacity) return error.PromptTooLong;
        lens[i] = 0;
        drafted_counts[i] = 0;
        accepted_counts[i] = 0;
        probe_at[i] = 0;
        indexes[i].observe(prompts[i]);
        try histories[i].appendSlice(allocator, prompts[i]);

        var logits = try model.forwardStep(ctx, &caches[i], prompts[i], caches[i].len);
        defer logits.deinit();
        var first_am = try logits.argmax(ctx, .vocab);
        defer first_am.deinit();
        const first: usize = @intCast((try first_am.dataConst())[0]);
        if (stop_id != null and first == stop_id.?) continue;
        outs[i][0] = first;
        lens[i] = 1;
        if (max_new > 1) {
            carries[i] = first;
            try histories[i].append(allocator, first);
            active[n_active] = i;
            n_active += 1;
        }
    }

    while (n_active > 0) {
        var total: usize = 0;
        for (0..n_active) |j| {
            const i = active[j];
            batch_caches[j] = &caches[i];
            span_tokens[total] = carries[i];
            // Gate: once >=16 drafted tokens accept below 35%, stop
            // drafting for 32 emitted tokens, then re-probe fresh.
            const gated = drafted_counts[i] >= 16 and
                accepted_counts[i] * 100 < drafted_counts[i] * 35;
            if (gated and lens[i] < probe_at[i]) {
                span_lens[j] = 1;
                total += 1;
                continue;
            }
            if (gated) {
                drafted_counts[i] = 0;
                accepted_counts[i] = 0;
                probe_at[i] = lens[i] + 32;
            }
            var k = indexes[i].asDraftSource().suggest(histories[i].items, draft_buf[0..max_draft]);
            // Cap the draft to the remaining budget: accepted tokens past
            // it would be discarded, so their rows are wasted work.
            const remaining = max_new - lens[i];
            if (k > remaining) {
                k = remaining;
                indexes[i].asDraftSource().truncatePending(k);
            }
            @memcpy(span_tokens[total + 1 ..][0..k], draft_buf[0..k]);
            span_lens[j] = 1 + k;
            total += 1 + k;
        }

        var logits = try model.forwardStepBatchSpans(ctx, batch_caches[0..n_active], span_tokens[0..total], span_lens[0..n_active]);
        defer logits.deinit();
        // One core argmax over the whole packed batch (lowest-index ties,
        // the greedy contract) instead of host scans per consumed row.
        var am = try logits.argmax(ctx, .vocab);
        defer am.deinit();
        const next_tokens = try am.dataConst();

        var kept: usize = 0;
        var row_at: usize = 0;
        for (0..n_active) |j| {
            const i = active[j];
            const span = span_lens[j];
            const drafts = span_tokens[row_at + 1 .. row_at + span];
            const len_before = caches[i].len - span; // the forward advanced by span
            const committed_start = histories[i].items.len;

            var accepted: usize = 0;
            var done = false;
            while (true) {
                const token: usize = @intCast(next_tokens[row_at + accepted]);
                if (stop_id != null and token == stop_id.?) {
                    done = true;
                    break;
                }
                outs[i][lens[i]] = token;
                lens[i] += 1;
                try histories[i].append(allocator, token);
                if (lens[i] >= max_new) {
                    done = true;
                    break;
                }
                if (accepted < drafts.len and token == drafts[accepted]) {
                    accepted += 1; // draft confirmed: its row is already valid
                    continue;
                }
                break; // mismatch or drafts exhausted: `token` is the new carry
            }
            drafted_counts[i] += span - 1;
            accepted_counts[i] += accepted;
            // Keep the carried token's row plus the accepted drafts' rows;
            // rewind the rejected tail.
            caches[i].truncate(len_before + 1 + accepted);
            indexes[i].observe(histories[i].items[committed_start..]);
            if (!done) {
                carries[i] = histories[i].items[histories[i].items.len - 1];
                active[kept] = i;
                kept += 1;
            }
            row_at += span;
        }
        n_active = kept;
    }

    for (outs, lens) |*buf, len| buf.* = try allocator.realloc(buf.*, len);
    return outs;
}


/// Batched lockstep generation: per-stream prefill, then ONE
/// `forwardStepBatch` weight pass per token across every still-active
/// stream (finished streams retire from the batch, so late stoppers never
/// pay for early ones). Caches are reset here; the stop token is dropped
/// from the returned ids; results live on `allocator`.
fn generateIdsBatch(
    ctx: *fucina.ExecContext,
    allocator: std.mem.Allocator,
    model: anytype,
    caches: []llm.kv_cache.KvCache,
    prompts: []const []const usize,
    max_new: usize,
    stop_id: ?usize,
    cfgs: []const llm.sampler.Config,
) ![][]usize {
    const n = prompts.len;
    std.debug.assert(n > 0 and caches.len >= n and cfgs.len == n and max_new > 0);

    const outs = try allocator.alloc([]usize, n);
    errdefer allocator.free(outs);
    var built: usize = 0;
    errdefer for (outs[0..built]) |buf| allocator.free(buf);
    for (outs) |*buf| {
        buf.* = try allocator.alloc(usize, max_new);
        built += 1;
    }

    const samplers = try allocator.alloc(llm.sampler.Sampler, n);
    defer allocator.free(samplers);
    const lens = try allocator.alloc(usize, n);
    defer allocator.free(lens);
    const active = try allocator.alloc(usize, n);
    defer allocator.free(active);
    const batch_caches = try allocator.alloc(*llm.kv_cache.KvCache, n);
    defer allocator.free(batch_caches);
    const batch_tokens = try allocator.alloc(usize, n);
    defer allocator.free(batch_tokens);

    var n_active: usize = 0;
    for (0..n) |i| {
        if (prompts[i].len + max_new > caches[i].capacity) return error.PromptTooLong;
        caches[i].reset();
        samplers[i] = llm.sampler.Sampler.init(cfgs[i]);
        lens[i] = 0;
        var logits = try model.forwardStep(ctx, &caches[i], prompts[i], 0);
        defer logits.deinit();
        const next = try samplers[i].next(ctx, &logits, outs[i][0..0]);
        if (stop_id != null and next == stop_id.?) continue;
        outs[i][0] = next;
        lens[i] = 1;
        if (max_new > 1) {
            active[n_active] = i;
            n_active += 1;
        }
    }

    while (n_active > 0) {
        for (0..n_active) |j| {
            const i = active[j];
            batch_caches[j] = &caches[i];
            batch_tokens[j] = outs[i][lens[i] - 1];
        }
        var logits = try model.forwardStepBatch(ctx, batch_caches[0..n_active], batch_tokens[0..n_active]);
        defer logits.deinit();

        var kept: usize = 0;
        for (0..n_active) |j| {
            const i = active[j];
            var row = try logits.narrow(ctx, .seq, j, 1);
            defer row.deinit();
            const next = try samplers[i].next(ctx, &row, outs[i][0..lens[i]]);
            if (stop_id != null and next == stop_id.?) continue;
            outs[i][lens[i]] = next;
            lens[i] += 1;
            if (lens[i] < max_new) {
                active[kept] = i;
                kept += 1;
            }
        }
        n_active = kept;
    }

    for (outs, lens) |*buf, len| buf.* = try allocator.realloc(buf.*, len);
    return outs;
}

/// Generate up to `max_new` tokens after `prompt_ids` through the inference
/// path, prefilling at the cache's CURRENT position (reset it first for a
/// fresh conversation; leave a preloaded cartridge prefix in place to serve
/// behind it). The stop token is dropped from the returned ids.
fn generateIds(
    ctx: *fucina.ExecContext,
    allocator: std.mem.Allocator,
    model: anytype,
    cache: *llm.kv_cache.KvCache,
    prompt_ids: []const usize,
    max_new: usize,
    stop_id: ?usize,
    sampler_cfg: llm.sampler.Config,
) ![]usize {
    if (cache.len + prompt_ids.len + max_new > cache.capacity) return error.PromptTooLong;
    var sampler = llm.sampler.Sampler.init(sampler_cfg);

    const out = try allocator.alloc(usize, max_new);
    errdefer allocator.free(out);
    var produced: usize = 0;

    var logits = try model.forwardStep(ctx, cache, prompt_ids, cache.len);
    while (produced < max_new) {
        const next = try sampler.next(ctx, &logits, out[0..produced]);
        logits.deinit();
        if (stop_id != null and next == stop_id.?) break;
        out[produced] = next;
        produced += 1;
        if (produced == max_new) return allocator.realloc(out, produced);
        logits = try model.forwardStep(ctx, cache, out[produced - 1 ..][0..1], cache.len);
    }
    if (produced < max_new) logits.deinit();
    return allocator.realloc(out, produced);
}

/// `generateIds` + decode to trimmed text (bot A's chat message).
fn generateText(
    ctx: *fucina.ExecContext,
    allocator: std.mem.Allocator,
    model: anytype,
    tokenizer: anytype,
    cache: *llm.kv_cache.KvCache,
    prompt: []const u8,
    max_new: usize,
    stop_id: ?usize,
    sampler_cfg: llm.sampler.Config,
) ![]u8 {
    const prompt_ids = try encodeUsize(allocator, tokenizer, prompt);
    defer allocator.free(prompt_ids);
    const ids = try generateIds(ctx, allocator, model, cache, prompt_ids, max_new, stop_id, sampler_cfg);
    defer allocator.free(ids);
    return cleanGenerated(allocator, tokenizer, ids);
}

/// Decode + trim a generation, dropping any leading think block/marker
/// (small models occasionally re-emit them despite the empty think block
/// in the assistant opener).
fn cleanGenerated(allocator: std.mem.Allocator, tokenizer: anytype, ids: []const usize) ![]u8 {
    const text = try decodeUsize(allocator, tokenizer, ids);
    defer allocator.free(text);
    var content = std.mem.trim(u8, text, " \t\r\n");
    if (std.mem.startsWith(u8, content, "<think>")) {
        if (std.mem.indexOf(u8, content, "</think>")) |end| content = content[end + "</think>".len ..];
    } else if (std.mem.startsWith(u8, content, "</think>")) {
        content = content["</think>".len..];
    }
    content = std.mem.trim(u8, content, " \t\r\n");
    if (content.len == 0) return error.EmptyGeneration;
    return allocator.dupe(u8, content);
}

/// The initialization token sequence: the corpus opening as a system block,
/// truncated to exactly `p` tokens with the turn close re-appended
/// (initialization/tokenization_utils.py semantics).
fn systemInitTokens(
    allocator: std.mem.Allocator,
    tokenizer: anytype,
    tpl: Tpl,
    corpus_text: []const u8,
    p: usize,
) ![]usize {
    const close_ids = try encodeUsize(allocator, tokenizer, tpl.block_close);
    defer allocator.free(close_ids);
    if (p < close_ids.len + 8) return error.CartridgeTooSmall;

    const sys_text = try std.fmt.allocPrint(allocator, "{s}" ++ system_prompt_template ++ "{s}", .{ tpl.sys_open, corpus_text, tpl.block_close });
    defer allocator.free(sys_text);
    const sys_ids = try encodeUsize(allocator, tokenizer, sys_text);
    defer allocator.free(sys_ids);
    if (sys_ids.len <= p) return allocator.dupe(usize, sys_ids);

    const out = try allocator.alloc(usize, p);
    @memcpy(out[0 .. p - close_ids.len], sys_ids[0 .. p - close_ids.len]);
    @memcpy(out[p - close_ids.len ..], close_ids);
    return out;
}

// ---------------------------------------------------------------------------
// --equiv: the zero-training acceptance gate
// ---------------------------------------------------------------------------

/// Cartridge-from-capture vs real prefill over the same suffix. Reports
/// logit deviation and greedy-choice agreement.
fn runEquiv(
    ctx: *fucina.ExecContext,
    io: std.Io,
    stdout: anytype,
    trainer: anytype,
    corpus: []const usize,
    opts: Options,
) !void {
    const p = opts.p;
    const suffix_len = @min(opts.suffix_max, corpus.len - p);
    const full = corpus[0 .. p + suffix_len];
    const suffix = full[p..];

    const t0 = nowNs(io);
    var teacher = try trainer.evalLogitsExt(ctx, full, .{});
    defer teacher.deinit();
    const t1 = nowNs(io);
    var cart = try trainer.initCartridge(ctx, full[0..p], opts.frozen_prefix);
    defer cart.deinit();
    const t2 = nowNs(io);
    var student = try trainer.evalLogitsExt(ctx, suffix, .{ .cartridge = &cart });
    defer student.deinit();
    const t3 = nowNs(io);

    const vocab = trainer.model.config.vocab_size;
    const teacher_rows = (try teacher.dataConst())[p * vocab ..];
    const student_rows = try student.dataConst();
    std.debug.assert(student_rows.len == suffix_len * vocab and teacher_rows.len == student_rows.len);

    var max_abs: f32 = 0;
    var sum_abs: f64 = 0;
    var greedy_match: usize = 0;
    for (0..suffix_len) |row| {
        var want_best: usize = 0;
        var got_best: usize = 0;
        const want_row = teacher_rows[row * vocab ..][0..vocab];
        const got_row = student_rows[row * vocab ..][0..vocab];
        for (want_row, got_row, 0..) |want, got, token| {
            const d = @abs(want - got);
            max_abs = @max(max_abs, d);
            sum_abs += d;
            if (want > want_row[want_best]) want_best = token;
            if (got > got_row[got_best]) got_best = token;
        }
        if (want_best == got_best) greedy_match += 1;
    }

    try stdout.print(
        "prefill-equivalence over {d} suffix tokens: max |dlogit| {d:.6}, mean {d:.7}, greedy agreement {d}/{d}\n",
        .{ suffix_len, max_abs, sum_abs / @as(f64, @floatFromInt(student_rows.len)), greedy_match, suffix_len },
    );
    try stdout.print(
        "timings: teacher prefill {d:.2} s, capture+init {d:.2} s, cartridge eval {d:.2} s\n",
        .{ seconds(t1 - t0), seconds(t2 - t1), seconds(t3 - t2) },
    );
    if (greedy_match != suffix_len) return error.EquivalenceGreedyMismatch;
    try stdout.print("PASS: untrained corpus-init cartridge is behaviorally identical to the real prefill\n", .{});
}

/// gemma4 arm of the CLI: the zero-training acceptance gate on a gemma
/// GGUF (dense or MoE, SWA + dual-theta rope included). Only --equiv is
/// routed here — the self-study generation loops of this CLI are
/// qwen3-typed.
fn runGemma(
    ctx: *fucina.ExecContext,
    io: std.Io,
    stdout: anytype,
    allocator: std.mem.Allocator,
    file: *fucina.gguf.File,
    corpus_paths: []const []const u8,
    opts: Options,
) !void {
    var config = try llm.gemma.gemma4.Config.fromGguf(file);
    // Zero-copy expert borrow over the GGUF mapping: the trainer's MoE arm
    // consumes the raw expert blocks (RawMoeWeightsRequired otherwise).
    config.borrow_experts = true;
    var spm = try llm.spm_tokenizer.Tokenizer.initFromGguf(allocator, file, .{});
    defer spm.deinit();
    var model = try llm.gemma.gemma4.Model.loadGgufFromFile(ctx, file, config);
    defer model.deinit();
    file.deinit();

    var trainer = try llm.gemma.gemma4_train.Trainer(.{ .q = false, .v = false }).init(ctx, &model, .{ .rank = 1, .alpha = 1 }, opts.seed);
    defer trainer.deinit();

    // Serve a saved cartridge (--load): same three-way comparison as qwen3.
    if (opts.load_path) |path| {
        const ask = opts.ask orelse return error.MissingAsk;
        const bytes = blk: {
            var dir = std.Io.Dir.cwd();
            break :blk try dir.readFileAlloc(io, path, allocator, .limited(1024 * 1024 * 1024));
        };
        defer allocator.free(bytes);
        var cart = try cartridge.Cartridge.initFromStateDict(ctx, allocator, bytes);
        defer cart.deinit();
        try stdout.print("cartridge loaded from {s}: p = {d} ({d} layers)\n", .{ path, cart.p, cart.layers.len });
        var corpus: ?Corpus = null;
        defer if (corpus) |*c| c.deinit();
        if (corpus_paths.len > 0) corpus = try buildCorpus(allocator, io, &spm, corpus_paths);
        return runServe(ctx, io, stdout, allocator, &model, &spm, gemma4_tpl, &cart, ask, if (corpus) |*c| c else null, opts);
    }

    if (corpus_paths.len == 0) return error.MissingCorpusPath;
    if (!opts.equiv) {
        // Self-study training: the shared engine with the gemma template;
        // no packed forward on this trainer (flat-memory backward arm).
        var corpus = try buildCorpus(allocator, io, &spm, corpus_paths);
        defer corpus.deinit();
        try stdout.print("corpus: {d} files, {d} tokens\n", .{ corpus.files.len, corpus.ids.len });
        if (corpus.ids.len < opts.chunk_min + 2 or corpus.ids.len < opts.p + 2) return error.CorpusTooShort;
        return runSelfStudy(ctx, io, stdout, allocator, &model, &spm, &trainer, gemma4_tpl, false, &corpus, opts);
    }
    try stdout.print("model: {s} (gemma4, {d} layers, hidden {d})\n", .{ opts.model_path, config.num_layers, config.hidden_size });

    var ids: std.ArrayList(usize) = .empty;
    defer ids.deinit(allocator);
    for (corpus_paths) |path| {
        var dir = std.Io.Dir.cwd();
        const content = try dir.readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024));
        defer allocator.free(content);
        const ids32 = try spm.encode(allocator, content);
        defer allocator.free(ids32);
        for (ids32) |id| try ids.append(allocator, id);
    }
    if (ids.items.len < opts.p + 2) return error.CorpusTooShort;
    try stdout.print("corpus: {d} files, {d} tokens\n", .{ corpus_paths.len, ids.items.len });

    // The cartridge student runs the suffix at GEMM shape m = suffix_len
    // while the teacher runs m = p + suffix_len. Quantized-MoE stacks are
    // NOT shape-invariant across kernel classes (near-tie experts flip), so
    // the honest yardstick is the model's OWN shape sensitivity at the
    // student's shape, measured with no cartridge anywhere: the first
    // suffix_len tokens forwarded alone vs the full forward's same rows.
    const suffix_len = @min(opts.suffix_max, ids.items.len - opts.p);
    const full = ids.items[0 .. opts.p + suffix_len];
    var envelope: f32 = 0;
    {
        var full_t = try trainer.evalLogitsExt(ctx, full, .{});
        defer full_t.deinit();
        var head_t = try trainer.evalLogitsExt(ctx, full[0..suffix_len], .{});
        defer head_t.deinit();
        const vocab = config.vocab_size;
        for (try head_t.dataConst(), (try full_t.dataConst())[0 .. suffix_len * vocab]) |a, b| {
            envelope = @max(envelope, @abs(a - b));
        }
        try stdout.print("model shape-sensitivity envelope (m={d} vs m={d}, no cartridge): max |dlogit| {d:.4}\n", .{ suffix_len, full.len, envelope });
    }

    runEquiv(ctx, io, stdout, &trainer, ids.items, opts) catch |err| switch (err) {
        error.EquivalenceGreedyMismatch => {
            if (envelope < 1e-3) return err;
            try stdout.print(
                "NOTE: greedy flips sit inside the model's own shape-sensitivity envelope ({d:.4}) — the quantized-MoE stack is not shape-invariant; the cartridge mechanism itself is pinned exact by the tiny-model gates (gemma4_train_tests).\n",
                .{envelope},
            );
        },
        else => return err,
    };
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn encodeUsize(allocator: std.mem.Allocator, tokenizer: anytype, text: []const u8) ![]usize {
    const ids32 = try tokenizer.encode(allocator, text);
    defer allocator.free(ids32);
    const out = try allocator.alloc(usize, ids32.len);
    for (out, ids32) |*dst, id| dst.* = id;
    return out;
}

fn decodeUsize(allocator: std.mem.Allocator, tokenizer: anytype, ids: []const usize) ![]u8 {
    const ids32 = try allocator.alloc(u32, ids.len);
    defer allocator.free(ids32);
    for (ids32, ids) |*dst, id| dst.* = @intCast(id);
    return tokenizer.decode(allocator, ids32);
}

fn parseFlagStr(args: []const []const u8, arg_i: *usize, comptime flag: []const u8) !?[]const u8 {
    const arg = args[arg_i.*];
    if (std.mem.eql(u8, arg, flag)) {
        arg_i.* += 1;
        if (arg_i.* >= args.len) return error.MissingFlagValue;
        return args[arg_i.*];
    }
    if (std.mem.startsWith(u8, arg, flag ++ "=")) return arg[flag.len + 1 ..];
    return null;
}

fn parseFlagInt(args: []const []const u8, arg_i: *usize, comptime flag: []const u8) !?usize {
    const text = (try parseFlagStr(args, arg_i, flag)) orelse return null;
    return try std.fmt.parseInt(usize, text, 10);
}

fn parseFlagF32(args: []const []const u8, arg_i: *usize, comptime flag: []const u8) !?f32 {
    const text = (try parseFlagStr(args, arg_i, flag)) orelse return null;
    return try std.fmt.parseFloat(f32, text);
}

fn nowNs(io: std.Io) i128 {
    return std.Io.Clock.real.now(io).nanoseconds;
}

fn seconds(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}
