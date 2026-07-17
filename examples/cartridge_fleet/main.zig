//! Cartridge fleets on a real Qwen3 GGUF (arXiv 2606.04557, "Cartridges at
//! Scale"): one cartridge PER DOCUMENT instead of one monolith per corpus,
//! trained jointly so they still work when several are loaded together, and
//! selected per query by an in-process cosine retriever.
//!
//! TRAIN (default; --docs + --fleet): every document gets a corpus-init
//! cartridge (saved to the fleet directory), then mixed-visibility
//! self-study runs under the RAM/disk budget manager:
//!   1. at most --budget cartridges are resident (rows + Adam moments);
//!      every --rotate-every rounds the most-trained residents rotate to
//!      disk and the least-trained absentees rotate in (uniform coverage);
//!   2. each round targets one resident document (sampled proportional to
//!      its length); with probability --p-iso the target's cartridge is the
//!      only prefix, otherwise 1..--distract-max distractor cartridges from
//!      other residents co-load in shuffled order — the paper's
//!      mixed-visibility recipe, which is what keeps composed serving from
//!      collapsing (visibility is sampled per optimizer GROUP here, not per
//!      conversation, so packing stays available — a recorded deviation);
//!   3. self-study is the reference loop (bot A asks about a chunk, bot B
//!      answers with the chunk in context, teacher-forced top-k rows become
//!      distillation targets) — gradients flow into EVERY co-loaded
//!      cartridge, and each part's own AdamW steps under its warm-up lr.
//! After training, every document is chunked and embedded THROUGH THE MODEL
//! ITSELF — each chunk is suffixed with a one-line "the main topic of the
//! text above is:" instruction and the final-norm LAST hidden state is the
//! embedding (the topic-prompt trick; mean pooling and bare last-token were
//! both measured to mis-rank documents), then centered and normalized by
//! `EmbedIndex.finalize` — into the fleet's `index.safetensors`. No
//! external embedding model or vector store; similarity is a hand-rolled
//! cosine.
//!
//! SERVE (--fleet + --ask): the question embeds through the same recipe,
//! cosine top-k chunks resolve to documents, the selected cartridges load
//! from disk and compose (concatenated KV prefixes) ahead of the question;
//! --oracle NAME bypasses retrieval, --rag-docs/--rag-chunks size the
//! selection; the bare-model answer prints for contrast.
//!
//! --equiv (--docs, no training): the composition acceptance gate on the
//! real model — two cartridges built from ONE capture of a 2p-token prefix
//! (part B holds rows at positions p..2p-1) must reproduce the real
//! prefill's logits over the following tokens with identical greedy
//! choices, pinning concat order, the summed position offset, and the
//! end-aligned kernel at production scale.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const optim = fucina.optim;
const cartridge = llm.cartridge;
const fleet_mod = llm.cartridge_fleet;

const default_model = "models/Qwen3-0.6B-f16.gguf";

/// No LoRA targets: the base model stays fully frozen; cartridge rows are
/// the only parameters.
const Trainer = llm.qwen3.train.Trainer(.{ .q = false, .v = false });
const GemmaTrainer = llm.gemma.gemma4_train.Trainer(.{ .q = false, .v = false });

/// Chat-template strings the prompt builders splice (runtime values so one
/// engine serves both architectures — the base cartridge CLI's Tpl).
const Tpl = struct {
    sys_open: []const u8,
    user_open: []const u8,
    asst_open: []const u8,
    block_close: []const u8,
    stop_marker: []const u8,
};

// ChatML (Qwen3); the assistant opener carries the empty think block so
// generations answer directly.
const qwen3_tpl = Tpl{
    .sys_open = "<|im_start|>system\n",
    .user_open = "<|im_start|>user\n",
    .asst_open = "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    .block_close = "<|im_end|>\n",
    .stop_marker = "<|im_end|>",
};

// Gemma 4's `<|turn>` format (thinking primed off).
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

/// Reference synthesizers/self_study.py SYSTEM_PROMPT_TEMPLATE.
const system_prompt_template = "\nYou are in a conversation about the following user information.\n\n<info>\n{s}\n</info>";

/// The seed-prompt set of the base cartridge CLI (reference five plus the
/// mechanism / verbatim-precision additions — see examples/cartridge/main.zig).
const seed_prompts = [_][]const u8{
    "Please generate a single chat message instructing an LLM to structure the information in JSON. " ++
        "Output only the chat message itself and absolutely nothing else. " ++
        "Make sure it is clear what section and document you are asking about. " ++
        "The message can follow the following template, filling in details from the corpus: \n\n" ++
        "'Can you structure the information in {subsection} of {document} related to {something specific} " ++
        "in the following format: JSON? Be sure to include precise information like any dates, times, names, and numerical values.'",
    "Please generate a single chat message instructing an LLM to summarize part of the corpus. " ++
        "Make sure the instruction is very explicit about the section of the corpus that you want to summarize. " ++
        "Include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. ",
    "Generate a question for an LLM that will test its knowledge of the information in the corpus above. " ++
        "In your question be sure to include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. " ++
        "Output only a single question. Do NOT include any other text or explanation other than the question.",
    "You are working to train a language model on the information in the following corpus. " ++
        "Your primary goal is to think about practical, real-world tasks or applications that someone could achieve using the knowledge contained within this corpus. " ++
        "Consider how a user might want to apply this information, not just recall it. " ++
        "After considering potential use cases, your task will be to generate a sample question that reflects one of these downstream applications. " ++
        "This question/instruction/task should be something a user, who has access to this corpus, might ask when trying to accomplish their specific goal. " ++
        "Output only a single question. Do NOT include any other text or explanation other than the question.",
    "You are having a creative conversation inspired by the information in the corpus. " ++
        "Please generate a question for your conversation partner to start off the discussion. " ++
        "Answer only with the question, do not include any other text.",
    "Generate a question for an LLM that asks WHY or HOW something described in the corpus works or " ++
        "happens. The question must demand the reasoning, mechanism, or cause behind it, not just the " ++
        "fact itself. In your question be sure to include details (ids, names, titles, dates, etc.) " ++
        "that make it clear what you are asking about. " ++
        "Output only a single question. Do NOT include any other text or explanation other than the question.",
    "Generate a question for an LLM that asks for the exact definition, full name, precise wording, " ++
        "or numerical value of something specific that appears in the corpus, to be answered precisely " ++
        "as it is stated in the text. In your question be sure to include details (ids, names, titles, " ++
        "dates, etc.) that make it unambiguous what you are asking about. " ++
        "Output only a single question. Do NOT include any other text or explanation other than the question.",
};

const Options = struct {
    model_path: []const u8 = default_model,
    fleet_dir: ?[]const u8 = null,
    p: usize = 256,
    frozen_prefix: usize = 1,
    // Budget manager (paper: R = 10, phi = 0.5, per-cartridge warm-up).
    budget: usize = 4,
    rotate_every: u64 = 10,
    evict_frac: f32 = 0.5,
    warmup: u64 = 8,
    // Mixed visibility (paper: P_iso = 0.75, k ~ U(1, k_max)).
    p_iso: f32 = 0.75,
    distract_max: usize = 3,
    // Self-study budgets (the base cartridge CLI's demo defaults).
    rounds: usize = 20,
    accum: usize = 4,
    lr: f32 = 2e-3,
    chunk_min: usize = 256,
    chunk_max: usize = 512,
    top_k: usize = 20,
    max_q: usize = 64,
    max_a: usize = 192,
    seed: u64 = 42,
    pack: bool = true,
    /// Recompute-in-backward per layer: drops the retained forward graph
    /// (the dominant training-memory term) for ~2x layer compute. Forces
    /// ISOLATED visibility (composed prefixes are not checkpoint inputs
    /// yet) and the flat backward; qwen3 only.
    checkpoint: bool = false,
    resume_fleet: bool = false,
    // Retrieval.
    embed_chunk: usize = 256,
    rag_docs: usize = 2,
    rag_chunks: usize = 8,
    // Serving.
    ask: ?[]const u8 = null,
    oracle: ?[]const u8 = null,
    // Gates.
    equiv: bool = false,
    suffix_max: usize = 64,
};

/// One document of the fleet: its own token stream and text (chunks quote
/// the text; the name feeds provenance lines and the manifest).
const Doc = struct {
    name: []u8,
    text: []u8,
    ids: []usize,

    fn deinit(self: *Doc, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.text);
        allocator.free(self.ids);
        self.* = undefined;
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const allocator = std.heap.smp_allocator;

    var opts = Options{};
    var doc_paths: std.ArrayList([]const u8) = .empty;
    defer doc_paths.deinit(allocator);
    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (try parseFlagStr(args, &arg_i, "--model")) |v| {
            opts.model_path = v;
        } else if (try parseFlagStr(args, &arg_i, "--docs")) |v| {
            try doc_paths.append(allocator, v);
        } else if (try parseFlagStr(args, &arg_i, "--fleet")) |v| {
            opts.fleet_dir = v;
        } else if (try parseFlagStr(args, &arg_i, "--ask")) |v| {
            opts.ask = v;
        } else if (try parseFlagStr(args, &arg_i, "--oracle")) |v| {
            opts.oracle = v;
        } else if (try parseFlagInt(args, &arg_i, "--p")) |v| {
            opts.p = v;
        } else if (try parseFlagInt(args, &arg_i, "--frozen")) |v| {
            opts.frozen_prefix = v;
        } else if (try parseFlagInt(args, &arg_i, "--budget")) |v| {
            opts.budget = v;
        } else if (try parseFlagInt(args, &arg_i, "--rotate-every")) |v| {
            opts.rotate_every = v;
        } else if (try parseFlagF32(args, &arg_i, "--evict-frac")) |v| {
            opts.evict_frac = v;
        } else if (try parseFlagInt(args, &arg_i, "--warmup")) |v| {
            opts.warmup = v;
        } else if (try parseFlagF32(args, &arg_i, "--p-iso")) |v| {
            opts.p_iso = v;
        } else if (try parseFlagInt(args, &arg_i, "--distract-max")) |v| {
            opts.distract_max = v;
        } else if (try parseFlagInt(args, &arg_i, "--rounds")) |v| {
            opts.rounds = v;
        } else if (try parseFlagInt(args, &arg_i, "--accum")) |v| {
            opts.accum = v;
        } else if (try parseFlagF32(args, &arg_i, "--lr")) |v| {
            opts.lr = v;
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
        } else if (try parseFlagInt(args, &arg_i, "--embed-chunk")) |v| {
            opts.embed_chunk = v;
        } else if (try parseFlagInt(args, &arg_i, "--rag-docs")) |v| {
            opts.rag_docs = v;
        } else if (try parseFlagInt(args, &arg_i, "--rag-chunks")) |v| {
            opts.rag_chunks = v;
        } else if (try parseFlagInt(args, &arg_i, "--suffix-max")) |v| {
            opts.suffix_max = v;
        } else if (std.mem.eql(u8, arg, "--checkpoint")) {
            opts.checkpoint = true;
        } else if (std.mem.eql(u8, arg, "--no-pack")) {
            opts.pack = false;
        } else if (std.mem.eql(u8, arg, "--resume")) {
            opts.resume_fleet = true;
        } else if (std.mem.eql(u8, arg, "--equiv")) {
            opts.equiv = true;
        } else {
            try stdout.print(
                "usage: zig build cartridge-fleet -Doptimize=ReleaseFast -- --docs FILE|DIR (repeatable) --fleet DIR " ++
                    "[--model PATH] [--p N] [--frozen N] [--budget B] [--rotate-every R] [--evict-frac F] [--warmup W] " ++
                    "[--p-iso F] [--distract-max K] [--rounds N] [--accum N] [--lr F] [--chunk-min N] [--chunk-max N] " ++
                    "[--top-k N] [--max-q N] [--max-a N] [--seed N] [--no-pack] [--checkpoint] [--resume] [--embed-chunk N]\n" ++
                    "  serve: --fleet DIR --ask TEXT [--rag-docs K] [--rag-chunks N] [--oracle NAME] [--max-a N]\n" ++
                    "  gate:  --docs FILE|DIR --equiv [--p N] [--suffix-max N]\n",
                .{},
            );
            return error.UnknownArgument;
        }
    }
    if (opts.accum == 0) return error.InvalidArguments;
    if (opts.chunk_min == 0 or opts.chunk_min > opts.chunk_max) return error.InvalidArguments;
    if (opts.budget == 0 or opts.budget > 64 or opts.p_iso < 0 or opts.p_iso > 1) return error.InvalidArguments;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = try fucina.gguf.File.loadMmap(allocator, io, opts.model_path);
    const arch = file.getString("general.architecture") orelse "";
    if (std.mem.startsWith(u8, arch, "gemma")) {
        var config = try llm.gemma.gemma4.Config.fromGguf(&file);
        // Zero-copy expert borrow: the trainer's MoE arm (self-study
        // backward AND the query-embedding forward) consumes raw expert
        // blocks (RawMoeWeightsRequired otherwise).
        config.borrow_experts = true;
        var model = try llm.gemma.gemma4.Model.loadGgufFromFile(&ctx, &file, config);
        defer model.deinit();
        var tokenizer = try llm.spm_tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
        defer tokenizer.deinit();
        file.deinit();
        try stdout.print("model: {s} (gemma4, {d} layers, hidden {d})\n", .{ opts.model_path, config.num_layers, config.hidden_size });
        if (opts.checkpoint) {
            try stdout.print("--checkpoint is qwen3-only today (the gemma4 trainer has no recompute plumbing yet)\n", .{});
            return error.CheckpointUnsupported;
        }
        var trainer = try GemmaTrainer.init(&ctx, &model, .{ .rank = 1, .alpha = 1 }, opts.seed);
        defer trainer.deinit();
        return dispatch(&ctx, io, stdout, allocator, &model, &tokenizer, &trainer, gemma4_tpl, false, doc_paths.items, opts);
    }
    if (!std.mem.startsWith(u8, arch, "qwen3")) {
        try stdout.print(
            "fleet training/serving supports qwen3 and gemma4 (got architecture '{s}'); composed serving for other families goes through llm.cartridge.writeComposedToCache\n",
            .{arch},
        );
        return error.UnsupportedArchitecture;
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

    var trainer = try Trainer.init(&ctx, &model, .{ .rank = 1, .alpha = 1 }, opts.seed);
    defer trainer.deinit();
    var opts2 = opts;
    if (opts.checkpoint) {
        // Composed prefixes are not checkpoint inputs yet: recompute mode
        // trains with isolated visibility and the flat backward.
        trainer.checkpoint_layers = true;
        opts2.p_iso = 1;
        opts2.pack = false;
        try stdout.print("--checkpoint: per-layer recompute on; visibility forced to isolated (p-iso 1)\n", .{});
    }
    return dispatch(&ctx, io, stdout, allocator, &model, &tokenizer, &trainer, qwen3_tpl, true, doc_paths.items, opts2);
}

/// Mode dispatch shared by both architecture arms. `supports_packing`
/// selects the packed group forward (qwen3) or the flat-memory
/// per-conversation backward (gemma4 — packed segments are not routed
/// there, matching the base cartridge CLI).
fn dispatch(
    ctx: *fucina.ExecContext,
    io: std.Io,
    stdout: anytype,
    allocator: std.mem.Allocator,
    model: anytype,
    tokenizer: anytype,
    trainer: anytype,
    tpl: Tpl,
    comptime supports_packing: bool,
    doc_paths: []const []const u8,
    opts: Options,
) !void {
    // Serve mode: no docs needed, everything comes from the fleet dir.
    if (opts.ask) |ask| {
        const dir = opts.fleet_dir orelse return error.MissingFleetDir;
        return runServe(ctx, io, stdout, allocator, model, tokenizer, trainer, tpl, dir, ask, opts);
    }

    // Every other mode reads documents.
    const docs = try loadDocs(allocator, io, tokenizer, doc_paths);
    defer {
        for (docs) |*doc| doc.deinit(allocator);
        allocator.free(docs);
    }
    try stdout.print("documents: {d}\n", .{docs.len});
    for (docs, 0..) |*doc, i| {
        try stdout.print("  [{d}] {s}: {d} tokens\n", .{ i, doc.name, doc.ids.len });
    }
    try stdout.flush();

    if (opts.equiv) return runComposeEquiv(ctx, io, stdout, trainer, docs, opts);

    const dir = opts.fleet_dir orelse return error.MissingFleetDir;
    return runFleetTrain(ctx, io, stdout, allocator, model, tokenizer, trainer, tpl, supports_packing, dir, docs, opts);
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

/// Each --docs argument is a file (one document) or a directory whose
/// top-level .md files are documents in sorted order.
fn loadDocs(
    allocator: std.mem.Allocator,
    io: std.Io,
    tokenizer: anytype,
    paths: []const []const u8,
) ![]Doc {
    var docs: std.ArrayList(Doc) = .empty;
    errdefer {
        for (docs.items) |*doc| doc.deinit(allocator);
        docs.deinit(allocator);
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
                try docs.append(allocator, try loadDoc(allocator, io, tokenizer, joined));
            }
        } else |_| {
            try docs.append(allocator, try loadDoc(allocator, io, tokenizer, path));
        }
    }
    if (docs.items.len == 0) return error.EmptyCorpus;
    return docs.toOwnedSlice(allocator);
}

fn loadDoc(allocator: std.mem.Allocator, io: std.Io, tokenizer: anytype, path: []const u8) !Doc {
    var dir = std.Io.Dir.cwd();
    const content = try dir.readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024));
    errdefer allocator.free(content);
    const ids = try encodeUsize(allocator, tokenizer, content);
    errdefer allocator.free(ids);
    const name = try allocator.dupe(u8, path);
    return .{ .name = name, .text = content, .ids = ids };
}

// ---------------------------------------------------------------------------
// Fleet training: init pass + mixed-visibility self-study + index build
// ---------------------------------------------------------------------------

fn runFleetTrain(
    ctx: *fucina.ExecContext,
    io: std.Io,
    stdout: anytype,
    allocator: std.mem.Allocator,
    model: anytype,
    tokenizer: anytype,
    trainer: anytype,
    tpl: Tpl,
    comptime supports_packing: bool,
    dir: []const u8,
    docs: []Doc,
    opts_in: Options,
) !void {
    var opts = opts_in;
    if (!supports_packing) opts.pack = false;
    const stop_id: ?usize = if (tokenizer.tokenId(tpl.stop_marker)) |id| @as(usize, id) else null;
    const policy = fleet_mod.RotationPolicy{
        .budget = opts.budget,
        .every = opts.rotate_every,
        .evict_fraction = opts.evict_frac,
        .warmup = opts.warmup,
    };

    var fleet = if (opts.resume_fleet) blk: {
        var reopened = try fleet_mod.Fleet.open(allocator, io, dir, opts.lr, policy);
        errdefer reopened.deinit();
        // Docs must match the manifest by name (order may differ on disk).
        if (reopened.manifest.docs.items.len != docs.len) return error.FleetDocMismatch;
        for (docs) |*doc| {
            if (reopened.manifest.findDoc(doc.name) == null) return error.FleetDocMismatch;
        }
        try stdout.print("fleet resumed from {s}: {d} docs, {d} rounds so far\n", .{ dir, docs.len, reopened.manifest.rounds });
        break :blk reopened;
    } else blk: {
        var manifest = fleet_mod.Manifest.init(allocator, opts.p);
        errdefer manifest.deinit();
        manifest.frozen_prefix = opts.frozen_prefix;
        manifest.embed_chunk = opts.embed_chunk;
        for (docs) |*doc| _ = try manifest.addDoc(doc.name, doc.ids.len);
        var created = try fleet_mod.Fleet.create(allocator, io, dir, manifest, opts.lr, policy);
        errdefer created.deinit();

        // Init pass: every document's cartridge from its own opening tokens
        // (the paper's per-document initialization), persisted immediately
        // so rotation can load any of them.
        const t0 = nowNs(io);
        for (docs, 0..) |*doc, i| {
            const init_tokens = try systemInitTokens(allocator, tokenizer, tpl, doc.text, opts.p);
            defer allocator.free(init_tokens);
            const cart = try trainer.initCartridge(ctx, init_tokens, opts.frozen_prefix);
            const idx = try created.adoptResident(i, cart);
            try created.evictResident(io, idx);
        }
        try stdout.print("fleet created at {s}: {d} per-document cartridges initialized (p = {d}) in {d:.1} s\n", .{
            dir, docs.len, opts.p, seconds(nowNs(io) - t0),
        });
        break :blk created;
    };
    defer fleet.deinit();
    try stdout.flush();

    // Doc name -> our docs[] slot, aligned with manifest doc ids.
    const doc_of = try allocator.alloc(usize, fleet.manifest.docs.items.len);
    defer allocator.free(doc_of);
    for (fleet.manifest.docs.items, 0..) |*state, id| {
        doc_of[id] = for (docs, 0..) |*doc, j| {
            if (std.mem.eql(u8, doc.name, state.name)) break j;
        } else return error.FleetDocMismatch;
    }

    if (opts.rounds == 0) {
        try buildIndex(ctx, io, stdout, allocator, trainer, tokenizer, &fleet, docs, doc_of);
        try fleet.saveAll(io);
        try stdout.print("fleet saved to {s} (no training rounds requested)\n", .{dir});
        return;
    }

    // Resident pool: least-trained docs first (all-zero steps = doc order).
    {
        const steps = try allocator.alloc(u64, fleet.manifest.docs.items.len);
        defer allocator.free(steps);
        const resident = try allocator.alloc(bool, fleet.manifest.docs.items.len);
        defer allocator.free(resident);
        for (steps, fleet.manifest.docs.items) |*s, *state| s.* = state.steps;
        @memset(resident, false);
        const initial = try fleet_mod.pickLoads(allocator, steps, resident, opts.budget);
        defer allocator.free(initial);
        for (initial) |doc_id| _ = try fleet.loadResident(ctx, io, doc_id);
    }
    try stdout.print("resident pool: {d}/{d} cartridges (budget {d})\n", .{ fleet.residents.items.len, docs.len, opts.budget });
    try stdout.flush();

    // Generation caches for the accumulation group.
    const capacity = opts.chunk_max + opts.max_q + opts.max_a + 256;
    const caches = try allocator.alloc(llm.kv_cache.KvCache, opts.accum);
    defer allocator.free(caches);
    var caches_inited: usize = 0;
    defer for (caches[0..caches_inited]) |*c| c.deinit();
    for (caches) |*c| {
        c.* = try makeCache(ctx, model, capacity);
        caches_inited += 1;
    }

    var prng = std.Random.DefaultPrng.init(opts.seed +% fleet.manifest.rounds);
    const rand = prng.random();

    // Held-out conversation on manifest doc 0, isolated visibility: the
    // generalization signal for the fleet's first document.
    var held: [1]Convo = undefined;
    try synthesizeGroup(ctx, allocator, model, tokenizer, trainer, tpl, supports_packing, caches[0..1], &docs[doc_of[0]], rand, opts, stop_id, &held);
    defer held[0].deinit(allocator);
    trainer.freeTransientRope();
    try stdout.print("held-out question (doc 0): {s}\n", .{held[0].question});
    const held_before = try heldLoss(ctx, io, trainer, &fleet, 0, &held[0]);
    try stdout.print("held-out distill loss before: {d:.4}\n", .{held_before});
    try stdout.flush();

    // Mixed-visibility self-study rounds.
    const parts_buf = try allocator.alloc(*const cartridge.Cartridge, opts.budget);
    defer allocator.free(parts_buf);
    const part_docs = try allocator.alloc(usize, opts.budget);
    defer allocator.free(part_docs);
    var trained_tokens: usize = 0;
    const train_start = nowNs(io);
    for (0..opts.rounds) |round_i| {
        const round_start = nowNs(io);

        // Target: resident doc sampled proportional to its token length.
        const target_slot = sampleResidentByLength(&fleet, rand);
        const target_doc = fleet.residents.items[target_slot].doc;

        // Visibility set for this GROUP: isolation with prob p_iso, else
        // the target plus k ~ U(1, distract_max) distractors from the other
        // residents, all in shuffled order.
        var n_parts: usize = 0;
        part_docs[n_parts] = target_doc;
        n_parts += 1;
        const iso = rand.float(f32) < opts.p_iso;
        if (!iso and fleet.residents.items.len > 1) {
            const avail = fleet.residents.items.len - 1;
            const k = rand.intRangeAtMost(usize, 1, @min(opts.distract_max, avail));
            // Reservoir-free sampling without replacement: shuffle the
            // other residents' slots and take the first k.
            var others_buf: [64]usize = undefined;
            std.debug.assert(avail <= others_buf.len);
            var m: usize = 0;
            for (fleet.residents.items, 0..) |*resident, slot| {
                if (slot != target_slot) {
                    others_buf[m] = resident.doc;
                    m += 1;
                }
            }
            rand.shuffle(usize, others_buf[0..m]);
            for (others_buf[0..k]) |doc_id| {
                part_docs[n_parts] = doc_id;
                n_parts += 1;
            }
        }
        rand.shuffle(usize, part_docs[0..n_parts]);
        for (part_docs[0..n_parts], 0..) |doc_id, j| {
            parts_buf[j] = &fleet.residents.items[fleet.residentIndex(doc_id).?].cart;
        }
        const parts = parts_buf[0..n_parts];

        // Self-study group on the TARGET document's chunks.
        const group = try allocator.alloc(Convo, opts.accum);
        defer allocator.free(group);
        try synthesizeGroup(ctx, allocator, model, tokenizer, trainer, tpl, supports_packing, caches, &docs[doc_of[target_doc]], rand, opts, stop_id, group);
        defer for (group) |*convo| convo.deinit(allocator);

        var total_entries: usize = 0;
        for (group) |*convo| {
            trained_tokens += convo.answer_len;
            total_entries += convo.builder.targets().positions.len;
        }

        // One packed forward/backward (or the flat-memory arm): gradients
        // land in EVERY co-loaded cartridge's trainable rows.
        var round_loss: f64 = 0;
        if (comptime supports_packing) pack_arm: {
            if (!opts.pack) break :pack_arm;
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
            var loss = try trainer.distillLossExt(ctx, packed_student, .{ .cartridges = parts, .packed_segments = seg_lens }, merged.targets(), .{});
            defer loss.deinit();
            try loss.backward(ctx);
            round_loss = try loss.item();
        }
        if (!(supports_packing and opts.pack)) {
            const scale = 1.0 / @as(f32, @floatFromInt(@max(1, total_entries)));
            for (group) |*convo| {
                const scope = ctx.openExecScope();
                defer ctx.closeExecScope(scope);
                var loss = try trainer.distillLossExt(ctx, convo.student_ids, .{ .cartridges = parts }, convo.builder.targets(), .{
                    .reduction = .sum,
                    .loss_scale = scale,
                });
                defer loss.deinit();
                try loss.backward(ctx);
                round_loss += try loss.item();
            }
        }

        // Every co-loaded cartridge steps under its own warm-up lr.
        for (part_docs[0..n_parts]) |doc_id| {
            const slot = fleet.residentIndex(doc_id).?;
            const resident = &fleet.residents.items[slot];
            resident.opt.config.lr = fleet.residentLr(slot);
            try resident.opt.step(ctx);
            resident.opt.zeroGrad();
            fleet.noteStep(doc_id);
        }
        trainer.freeTransientRope();

        try stdout.print("round {d:>3}/{d}: target doc {d} ({s}), {s} x{d}, distill loss {d:.4} ({d:.1} s)\n", .{
            round_i + 1,
            opts.rounds,
            target_doc,
            fleet.manifest.docs.items[target_doc].name,
            if (n_parts == 1) "isolated" else "co-loaded",
            n_parts,
            round_loss,
            seconds(nowNs(io) - round_start),
        });
        try stdout.flush();

        fleet.manifest.rounds += 1;
        const rotated = try fleet.maybeRotate(ctx, io, stdout);
        if (rotated > 0) try fleet.writeManifest(io);
    }
    const train_s = seconds(nowNs(io) - train_start);
    try stdout.print("self-study: {d} rounds x {d} conversations, {d} supervised answer tokens, {d:.1} min ({d:.1} s/conversation)\n", .{
        opts.rounds, opts.accum, trained_tokens, train_s / 60.0, train_s / @as(f64, @floatFromInt(opts.rounds * opts.accum)),
    });

    const held_after = try heldLoss(ctx, io, trainer, &fleet, 0, &held[0]);
    try stdout.print("held-out distill loss after: {d:.4} (before {d:.4})\n", .{ held_after, held_before });

    try buildIndex(ctx, io, stdout, allocator, trainer, tokenizer, &fleet, docs, doc_of);
    try fleet.saveAll(io);
    try stdout.print("fleet saved to {s}\n", .{dir});
    for (fleet.manifest.docs.items, 0..) |*state, id| {
        try stdout.print("  [{d}] {s}: {d} steps\n", .{ id, state.name, state.steps });
    }
}

/// Held-out distillation loss behind doc `doc_id`'s cartridge in isolation
/// (loading it from disk if rotation moved it out).
fn heldLoss(ctx: *fucina.ExecContext, io: std.Io, trainer: anytype, fleet: *fleet_mod.Fleet, doc_id: usize, convo: *const Convo) !f32 {
    var slot = fleet.residentIndex(doc_id);
    var evict_after = false;
    if (slot == null) {
        // Over-budget by one for the duration of the eval: acceptable for a
        // reporting pass, and it leaves the pool exactly as it was.
        fleet.policy.budget += 1;
        defer fleet.policy.budget -= 1;
        slot = try fleet.loadResident(ctx, io, doc_id);
        evict_after = true;
    }
    const cart = &fleet.residents.items[slot.?].cart;
    const scope = ctx.openExecScope();
    var loss = try trainer.distillLossExt(ctx, convo.student_ids, .{ .cartridges = &.{cart} }, convo.builder.targets(), .{});
    const value = try loss.item();
    loss.deinit();
    ctx.closeExecScope(scope);
    if (evict_after) try fleet.evictResident(io, fleet.residentIndex(doc_id).?);
    return value;
}

/// Resident target sampling, proportional to document token length
/// (the paper's proportional-to-length QA budget).
fn sampleResidentByLength(fleet: *const fleet_mod.Fleet, rand: std.Random) usize {
    var total: u64 = 0;
    for (fleet.residents.items) |*resident| {
        total += @max(1, fleet.manifest.docs.items[resident.doc].tokens);
    }
    var draw = rand.uintLessThan(u64, total);
    for (fleet.residents.items, 0..) |*resident, slot| {
        const w = @max(1, fleet.manifest.docs.items[resident.doc].tokens);
        if (draw < w) return slot;
        draw -= w;
    }
    unreachable;
}

// ---------------------------------------------------------------------------
// Retrieval index: model-embedded chunks, hand-rolled cosine
// ---------------------------------------------------------------------------

/// Embed a token span through the frozen model: the topic-instruction
/// suffix appended, then the trainer's final-norm LAST hidden state — the
/// recipe pinned by `cartridge_fleet.embed_suffix`'s contract (index chunks
/// and serve-time queries must match; lmserve --fleet embeds queries the
/// same way).
fn embedTokens(ctx: *fucina.ExecContext, trainer: anytype, allocator: std.mem.Allocator, tokenizer: anytype, ids: []const usize, out: []f32) !void {
    const suffix_ids = try encodeUsize(allocator, tokenizer, fleet_mod.embed_suffix);
    defer allocator.free(suffix_ids);
    const full = try std.mem.concat(allocator, usize, &.{ ids, suffix_ids });
    defer allocator.free(full);
    return trainer.embedLastHidden(ctx, full, out);
}

/// Chunk every document at `embed_chunk` tokens (tail kept when >= 32
/// tokens or the document is short), embed through the model, and persist
/// the fleet's cosine index.
fn buildIndex(
    ctx: *fucina.ExecContext,
    io: std.Io,
    stdout: anytype,
    allocator: std.mem.Allocator,
    trainer: anytype,
    tokenizer: anytype,
    fleet: *fleet_mod.Fleet,
    docs: []Doc,
    doc_of: []const usize,
) !void {
    const dim = trainer.model.config.hidden_size;
    fleet.manifest.embed_dim = dim;
    var index = fleet_mod.EmbedIndex.init(allocator, dim);
    defer index.deinit();
    const vec = try allocator.alloc(f32, dim);
    defer allocator.free(vec);

    const t0 = nowNs(io);
    const chunk = fleet.manifest.embed_chunk;
    for (fleet.manifest.docs.items, 0..) |_, doc_id| {
        const ids = docs[doc_of[doc_id]].ids;
        var start: usize = 0;
        while (start < ids.len) : (start += chunk) {
            const len = @min(chunk, ids.len - start);
            if (len < 32 and start != 0) break;
            try embedTokens(ctx, trainer, allocator, tokenizer, ids[start..][0..len], vec);
            try index.append(@intCast(doc_id), vec);
        }
    }
    try index.finalize();

    const index_path = try fleet.indexPath();
    defer allocator.free(index_path);
    const writer_ctx = WriteIndexCtx{ .allocator = allocator, .index = &index };
    try fucina.training_checkpoint.writeFileAtomic(io, index_path, writer_ctx, writeIndexTo);
    try fleet.writeManifest(io);
    try stdout.print("retrieval index: {d} chunks x {d} dims embedded through the model in {d:.1} s -> {s}\n", .{
        index.len(), dim, seconds(nowNs(io) - t0), index_path,
    });
}

const WriteIndexCtx = struct {
    allocator: std.mem.Allocator,
    index: *const fleet_mod.EmbedIndex,
};

fn writeIndexTo(ctx: WriteIndexCtx, writer: *std.Io.Writer) anyerror!void {
    try ctx.index.serialize(ctx.allocator, writer);
}

// ---------------------------------------------------------------------------
// Serving: retrieve -> compose -> decode
// ---------------------------------------------------------------------------

fn runServe(
    ctx: *fucina.ExecContext,
    io: std.Io,
    stdout: anytype,
    allocator: std.mem.Allocator,
    model: anytype,
    tokenizer: anytype,
    trainer: anytype,
    tpl: Tpl,
    dir: []const u8,
    ask: []const u8,
    opts: Options,
) !void {
    var fleet = try fleet_mod.Fleet.open(allocator, io, dir, opts.lr, .{ .budget = 1 });
    defer fleet.deinit();
    const stop_id: ?usize = if (tokenizer.tokenId(tpl.stop_marker)) |id| @as(usize, id) else null;

    // Which cartridges: --oracle by name, otherwise cosine retrieval.
    var selected: std.ArrayList(usize) = .empty;
    defer selected.deinit(allocator);
    if (opts.oracle) |name| {
        const doc_id = fleet.manifest.findDoc(name) orelse {
            try stdout.print("--oracle: no document named '{s}' in the fleet; documents are:\n", .{name});
            for (fleet.manifest.docs.items, 0..) |*state, id| {
                try stdout.print("  [{d}] {s}\n", .{ id, state.name });
            }
            return error.UnknownDocument;
        };
        try selected.append(allocator, doc_id);
        try stdout.print("oracle selection: doc {d} ({s})\n", .{ doc_id, name });
    } else {
        if (fleet.manifest.embed_dim == 0) return error.MissingIndex;
        const index_path = try fleet.indexPath();
        defer allocator.free(index_path);
        var mapped = try fleet_mod.mmapFile(io, index_path);
        defer mapped.deinit();
        var index = try fleet_mod.EmbedIndex.initFromBytes(allocator, mapped.bytes);
        defer index.deinit();

        const query_ids = try encodeUsize(allocator, tokenizer, ask);
        defer allocator.free(query_ids);
        const query_vec = try allocator.alloc(f32, fleet.manifest.embed_dim);
        defer allocator.free(query_vec);
        try embedTokens(ctx, trainer, allocator, tokenizer, query_ids, query_vec);

        const hits = try index.topDocs(allocator, query_vec, opts.rag_chunks, opts.rag_docs);
        defer allocator.free(hits);
        if (hits.len == 0) return error.EmptySelection;
        try stdout.print("cartridge selection (cosine over {d} chunks):\n", .{index.len()});
        for (hits) |hit| {
            try selected.append(allocator, hit.doc);
            try stdout.print("  doc {d} ({s}): best-chunk cosine {d:.4}\n", .{ hit.doc, fleet.manifest.docs.items[hit.doc].name, hit.score });
        }
    }
    try stdout.flush();

    // Load the selected cartridges (rows only — no optimizers at serve).
    const carts = try allocator.alloc(cartridge.Cartridge, selected.items.len);
    defer allocator.free(carts);
    var carts_built: usize = 0;
    defer for (carts[0..carts_built]) |*cart| cart.deinit();
    const parts = try allocator.alloc(*const cartridge.Cartridge, selected.items.len);
    defer allocator.free(parts);
    for (selected.items, 0..) |doc_id, j| {
        const leaf = fleet.manifest.docs.items[doc_id].cart_file;
        const cart_path = try std.fs.path.join(allocator, &.{ dir, leaf });
        defer allocator.free(cart_path);
        // mmap retrieval: the parse streams the mapped pages straight into
        // fresh tensors, no whole-file heap copy.
        var mapped = try fleet_mod.mmapFile(io, cart_path);
        defer mapped.deinit();
        carts[j] = try cartridge.Cartridge.initFromStateDict(ctx, allocator, mapped.bytes);
        carts_built += 1;
        parts[j] = &carts[j];
    }
    const total_p = cartridge.composedP(parts);

    const prompt = try std.fmt.allocPrint(allocator, "{s}{s}{s}{s}", .{ tpl.user_open, ask, tpl.block_close, tpl.asst_open });
    defer allocator.free(prompt);
    const prompt_ids = try encodeUsize(allocator, tokenizer, prompt);
    defer allocator.free(prompt_ids);

    // Composed cartridges ahead of the question.
    {
        var cache = try makeCache(ctx, model, total_p + prompt_ids.len + opts.max_a + 16);
        defer cache.deinit();
        try cartridge.writeComposedToCache(ctx, parts, &cache);
        const t0 = nowNs(io);
        const ids = try generateIds(ctx, allocator, model, &cache, prompt_ids, opts.max_a, stop_id);
        defer allocator.free(ids);
        const answer = try cleanGenerated(allocator, tokenizer, ids);
        defer allocator.free(answer);
        const secs = seconds(nowNs(io) - t0);
        try stdout.print("\n[{d} composed cartridge(s), {d} KV rows] ({d:.2} s, ~{d:.1} tok/s)\n{s}\n", .{
            parts.len, total_p, secs, @as(f64, @floatFromInt(ids.len)) / secs, answer,
        });
        try stdout.flush();
    }

    // Bare model for contrast.
    {
        var cache = try makeCache(ctx, model, prompt_ids.len + opts.max_a + 16);
        defer cache.deinit();
        const ids = try generateIds(ctx, allocator, model, &cache, prompt_ids, opts.max_a, stop_id);
        defer allocator.free(ids);
        const answer = try cleanGenerated(allocator, tokenizer, ids);
        defer allocator.free(answer);
        try stdout.print("\n[bare model, no context]\n{s}\n", .{answer});
    }
}

// ---------------------------------------------------------------------------
// --equiv: the composition acceptance gate on the real model
// ---------------------------------------------------------------------------

/// Two cartridges from ONE capture of the first 2p corpus tokens (part B
/// carries positions p..2p-1) must reproduce the real prefill over the
/// following tokens: near-identical logits, identical greedy choices.
fn runComposeEquiv(
    ctx: *fucina.ExecContext,
    io: std.Io,
    stdout: anytype,
    trainer: anytype,
    docs: []Doc,
    opts: Options,
) !void {
    const allocator = trainer.allocator;
    // The gate runs over the first document's stream (append more docs'
    // ids if it is short).
    var ids: std.ArrayList(usize) = .empty;
    defer ids.deinit(allocator);
    for (docs) |*doc| try ids.appendSlice(allocator, doc.ids);
    const prefix_len = 2 * opts.p;
    if (ids.items.len < prefix_len + 2) return error.CorpusTooShort;
    const suffix_len = @min(opts.suffix_max, ids.items.len - prefix_len);
    const full = ids.items[0 .. prefix_len + suffix_len];
    const suffix = full[prefix_len..];

    const t0 = nowNs(io);
    var teacher = try trainer.evalLogitsExt(ctx, full, .{});
    defer teacher.deinit();
    const t1 = nowNs(io);

    // One capture, split at p: part B's rows keep their positions p..2p-1.
    // Geometry is duck-typed per layer (gemma4 mixes shapes; qwen3 is
    // uniform), so both arms build through initFromRowsVaried.
    var cap = try trainer.captureKv(ctx, full[0..prefix_len]);
    defer cap.deinit();
    const n_layers = cap.k_rows.len;
    const kv_heads = try allocator.alloc(usize, n_layers);
    defer allocator.free(kv_heads);
    const head_dims = try allocator.alloc(usize, n_layers);
    defer allocator.free(head_dims);
    if (comptime @hasField(std.meta.Child(@TypeOf(trainer.model)), "geom")) {
        @memcpy(kv_heads, trainer.model.geom.kv_heads);
        @memcpy(head_dims, trainer.model.geom.head_dim);
    } else {
        @memset(kv_heads, trainer.model.config.num_key_value_heads);
        @memset(head_dims, trainer.model.config.head_dim);
    }
    const k_slices = try allocator.alloc([]const f32, n_layers);
    defer allocator.free(k_slices);
    const v_slices = try allocator.alloc([]const f32, n_layers);
    defer allocator.free(v_slices);
    for (0..n_layers) |l| {
        const row = kv_heads[l] * head_dims[l];
        k_slices[l] = cap.k_rows[l][0 .. opts.p * row];
        v_slices[l] = cap.v_rows[l][0 .. opts.p * row];
    }
    var cart_a = try cartridge.Cartridge.initFromRowsVaried(ctx, allocator, opts.frozen_prefix, opts.p, kv_heads, head_dims, k_slices, v_slices);
    defer cart_a.deinit();
    for (0..n_layers) |l| {
        const row = kv_heads[l] * head_dims[l];
        k_slices[l] = cap.k_rows[l][opts.p * row ..];
        v_slices[l] = cap.v_rows[l][opts.p * row ..];
    }
    var cart_b = try cartridge.Cartridge.initFromRowsVaried(ctx, allocator, opts.frozen_prefix, opts.p, kv_heads, head_dims, k_slices, v_slices);
    defer cart_b.deinit();
    const t2 = nowNs(io);

    var student = try trainer.evalLogitsExt(ctx, suffix, .{ .cartridges = &.{ &cart_a, &cart_b } });
    defer student.deinit();
    const t3 = nowNs(io);

    const vocab = trainer.model.config.vocab_size;
    const teacher_rows = (try teacher.dataConst())[prefix_len * vocab ..];
    const student_rows = try student.dataConst();
    std.debug.assert(student_rows.len == suffix_len * vocab and teacher_rows.len == student_rows.len);

    var max_abs: f32 = 0;
    var sum_abs: f64 = 0;
    var greedy_match: usize = 0;
    for (0..suffix_len) |r| {
        var want_best: usize = 0;
        var got_best: usize = 0;
        const want_row = teacher_rows[r * vocab ..][0..vocab];
        const got_row = student_rows[r * vocab ..][0..vocab];
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
        "composed prefill-equivalence over {d} suffix tokens (2 x p = {d} rows): max |dlogit| {d:.6}, mean {d:.7}, greedy agreement {d}/{d}\n",
        .{ suffix_len, prefix_len, max_abs, sum_abs / @as(f64, @floatFromInt(student_rows.len)), greedy_match, suffix_len },
    );
    try stdout.print(
        "timings: teacher prefill {d:.2} s, capture+split {d:.2} s, composed eval {d:.2} s\n",
        .{ seconds(t1 - t0), seconds(t2 - t1), seconds(t3 - t2) },
    );
    if (greedy_match != suffix_len) {
        // Quantized-MoE stacks are not GEMM-shape-invariant (near-tie
        // experts flip when the same rows run at different batch shapes):
        // judge greedy flips against the model's OWN shape-sensitivity
        // envelope, measured with no cartridge anywhere (the base
        // cartridge CLI's gemma --equiv arm).
        var envelope: f32 = 0;
        {
            var head_t = try trainer.evalLogitsExt(ctx, full[0..suffix_len], .{});
            defer head_t.deinit();
            const teacher_head = (try teacher.dataConst())[0 .. suffix_len * vocab];
            for (try head_t.dataConst(), teacher_head) |a, b| {
                envelope = @max(envelope, @abs(a - b));
            }
        }
        try stdout.print("model shape-sensitivity envelope (m={d} vs m={d}, no cartridge): max |dlogit| {d:.4}\n", .{ suffix_len, full.len, envelope });
        if (envelope < 1e-3) return error.EquivalenceGreedyMismatch;
        try stdout.print(
            "NOTE: greedy flips sit inside the model's own shape-sensitivity envelope — the composition mechanism itself is pinned exact by the tiny-model gates (train_cartridge_compose_tests, gemma4_train_tests).\n",
            .{},
        );
        return;
    }
    try stdout.print("PASS: the two-part composition is behaviorally identical to the real prefill\n", .{});
}

// ---------------------------------------------------------------------------
// Self-study synthesis (the base cartridge CLI's engine, per-document)
// ---------------------------------------------------------------------------

/// One synthesized conversation (see examples/cartridge/main.zig): the seed-free
/// student element plus the teacher's sparse top-k targets for its answer.
const Convo = struct {
    student_ids: []usize,
    teacher_ids: []usize,
    first_answer: usize,
    answer_len: usize,
    question: []u8,
    builder: cartridge.TargetsBuilder,

    fn deinit(self: *Convo, allocator: std.mem.Allocator) void {
        allocator.free(self.student_ids);
        allocator.free(self.teacher_ids);
        allocator.free(self.question);
        self.builder.deinit();
        self.* = undefined;
    }
};

/// Algorithm 1 with k = 1 over one accumulation group, chunks drawn from a
/// SINGLE document: bot A asks (temp 0.6, lockstep batched), bot B answers
/// with the chunk in context (greedy, batched), one packed teacher pass
/// extracts every conversation's top-k targets.
fn synthesizeGroup(
    ctx: *fucina.ExecContext,
    allocator: std.mem.Allocator,
    model: anytype,
    tokenizer: anytype,
    trainer: anytype,
    tpl: Tpl,
    comptime supports_packing: bool,
    caches: []llm.kv_cache.KvCache,
    doc: *const Doc,
    rand: std.Random,
    opts: Options,
    stop_id: ?usize,
    out: []Convo,
) !void {
    const n = out.len;
    std.debug.assert(n > 0 and caches.len >= n);
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Chunks + seeds + bot-A prompts (uniform token spans of THIS document,
    // with the reference-style provenance line).
    const sys_texts = try arena.alloc([]u8, n);
    const a_prompts = try arena.alloc([]const usize, n);
    const a_cfgs = try arena.alloc(llm.sampler.Config, n);
    for (0..n) |i| {
        const chunk_max = @min(opts.chunk_max, doc.ids.len);
        const chunk_len = rand.intRangeAtMost(usize, @min(opts.chunk_min, chunk_max), chunk_max);
        const chunk_start = rand.intRangeAtMost(usize, 0, doc.ids.len - chunk_len);
        const chunk_body = try decodeUsize(arena, tokenizer, doc.ids[chunk_start..][0..chunk_len]);
        const chunk_text = try std.fmt.allocPrint(
            arena,
            "Below is an excerpt from {s}. It is part of a larger corpus of documents.\n\n{s}",
            .{ doc.name, chunk_body },
        );
        sys_texts[i] = try std.fmt.allocPrint(arena, "{s}" ++ system_prompt_template ++ "{s}", .{ tpl.sys_open, chunk_text, tpl.block_close });
        const seed_prompt = seed_prompts[rand.intRangeLessThan(usize, 0, seed_prompts.len)];
        const a_prompt = try std.fmt.allocPrint(arena, "{s}{s}{s}{s}{s}", .{ sys_texts[i], tpl.user_open, seed_prompt, tpl.block_close, tpl.asst_open });
        a_prompts[i] = try encodeUsize(arena, tokenizer, a_prompt);
        a_cfgs[i] = .{ .temperature = 0.6, .seed = rand.int(u64) };
    }

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
    const b_outs = try generateIdsBatch(ctx, arena, model, caches[0..n], b_prompts, opts.max_a, stop_id, b_cfgs);

    var built: usize = 0;
    errdefer for (out[0..built]) |*convo| convo.deinit(allocator);
    for (0..n) |i| {
        const answer_gen = b_outs[i];
        if (answer_gen.len == 0) return error.EmptyAnswer;
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

    try teacherTargets(ctx, arena, trainer, supports_packing, out, opts);
}

/// One packed teacher pass for the group (see examples/cartridge/main.zig).
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
        const rest = try arena.alloc(*const fucina.Tensor(.{ .seq, .vocab }), parts.len - 1);
        for (rest, parts[1..]) |*ptr, *t| ptr.* = t;
        const joined = try parts[0].concat(ctx, .seq, rest);
        for (parts[0..parts_built]) |*t| t.deinit();
        parts_built = 0;
        break :blk joined;
    };
    defer teacher_logits.deinit();

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

// ---------------------------------------------------------------------------
// Generation helpers (see examples/cartridge/main.zig)
// ---------------------------------------------------------------------------

/// Batched lockstep generation: per-stream prefill, then one
/// `forwardStepBatch` weight pass per token across the active streams.
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

/// Greedy generation after `prompt_ids`, prefilling at the cache's CURRENT
/// position (a preloaded composed prefix stays in place). The stop token is
/// dropped from the returned ids.
fn generateIds(
    ctx: *fucina.ExecContext,
    allocator: std.mem.Allocator,
    model: anytype,
    cache: *llm.kv_cache.KvCache,
    prompt_ids: []const usize,
    max_new: usize,
    stop_id: ?usize,
) ![]usize {
    if (cache.len + prompt_ids.len + max_new > cache.capacity) return error.PromptTooLong;
    var sampler = llm.sampler.Sampler.init(.{});

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

/// Decode + trim a generation, dropping any leading think block/marker.
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

/// The per-document initialization token sequence: the document opening as
/// a system block, truncated to exactly `p` tokens with the turn close
/// re-appended (initialization/tokenization_utils.py semantics).
fn systemInitTokens(
    allocator: std.mem.Allocator,
    tokenizer: anytype,
    tpl: Tpl,
    doc_text: []const u8,
    p: usize,
) ![]usize {
    const close_ids = try encodeUsize(allocator, tokenizer, tpl.block_close);
    defer allocator.free(close_ids);
    if (p < close_ids.len + 8) return error.CartridgeTooSmall;

    const sys_text = try std.fmt.allocPrint(allocator, "{s}" ++ system_prompt_template ++ "{s}", .{ tpl.sys_open, doc_text, tpl.block_close });
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
