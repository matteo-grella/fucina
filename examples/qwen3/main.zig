//! Qwen3 runner: load a GGUF, then dispatch on the parsed CLI mode —
//! direct forward (logit dump/compare), generation (plain, warm-repeat
//! bench, multi-stream, speculative), verification harnesses, chat/REPL,
//! tokenizer parity, or model info. The modes live in sibling modules
//! (options/util/bench/generate/verify/chat.zig); this file owns the
//! model/tokenizer lifetime and the dispatch.
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const options_mod = @import("options.zig");
const util = @import("util.zig");
const bench = @import("bench.zig");
const generate = @import("generate.zig");
const verify = @import("verify.zig");
const chat = @import("chat.zig");
const shine_mode = @import("shine.zig");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var token_buf: [4096]usize = undefined;
    var opts = (try options_mod.parse(args, stdout, &token_buf)) orelse return;
    var tokens = opts.tokens;

    const allocator = std.heap.smp_allocator;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const load_start = util.nowNs(init.io);
    var file = try fucina.gguf.File.loadMmapAuto(allocator, init.io, args[1]);

    // --tokenize FILE: encode a text file and print one token id per line
    // (the llama-tokenize parity harness); no model weights needed.
    if (opts.tokenize_file) |path| {
        var t = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
        defer t.deinit();
        file.deinit();
        const ids = try util.tokenizeFile(init.io, allocator, &t, path);
        defer allocator.free(ids);
        for (ids) |id| try stdout.print("{d}\n", .{id});
        return;
    }

    var moe_stream = try opts.moe_cli.options(args[1]);
    if (moe_stream) |*m| {
        m.cache_slots_per_layer = opts.moe_cache_slots;
        m.auto_pin = !opts.moe_no_learn;
        m.pin_bytes = if (opts.moe_pin_mb) |mb| mb << 20 else null;
        m.pilot = opts.moe_pilot;
    }
    const load_options: llm.qwen3.model.LoadOptions = if (moe_stream) |m| .{ .moe_stream = m } else .{};
    var model_config = try llm.qwen3.model.Config.fromGguf(&file);
    if (opts.moe_expert_top_p) |p| model_config.moe_expert_top_p = p;
    var model = try llm.qwen3.model.Model.loadGgufFromFileOptions(&ctx, &file, model_config, load_options);
    defer model.deinit();
    // The stats go through the SAME buffered stdout writer as everything
    // else: stdout's positional writes and stderr's offset-advancing writes
    // cannot safely share one redirected file (`cmd > f 2>&1` interleaves
    // destructively), so a std.debug stats line would get overwritten.
    defer if (model.expert_store) |store| fucina.weights.reportAndSaveMoeStream(store, !opts.moe_no_learn, stdout);
    // Build a tokenizer from the same file's metadata; tolerate models without it.
    var tokenizer: ?llm.tokenizer.Tokenizer = llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{}) catch null;
    defer if (tokenizer) |*t| t.deinit();
    const tok_ptr: ?*const llm.tokenizer.Tokenizer = if (tokenizer) |*t| t else null;
    const chat_tmpl = llm.chat.Template.detect(file.getString("tokenizer.chat_template"));

    if (opts.info_flag) {
        try printInfo(stdout, &file, tok_ptr);
        file.deinit();
        return;
    }
    file.deinit(); // model + tokenizer own their data now
    const load_ns = util.nowNs(init.io) - load_start;

    // KV persistence sidecar: explicit path, or <model>.kvcache next to the GGUF.
    var kv_path_buf: [1024]u8 = undefined;
    const kv_save_path: ?[]const u8 = if (!opts.kv_save) null else opts.kv_save_arg orelse try std.fmt.bufPrint(&kv_path_buf, "{s}.kvcache", .{args[1]});

    const is_chat = opts.chat_text != null or opts.repl_flag;

    // Chat samples with Qwen3's recommended settings; the benchmark and
    // raw-completion paths default to greedy (deterministic). Flags override.
    const sampler_cfg: llm.sampler.Config = if (is_chat) .{
        .temperature = opts.temp_arg orelse (if (opts.no_think) @as(f32, 0.7) else 0.6),
        .top_k = opts.topk_arg orelse 20,
        .top_p = opts.topp_arg orelse (if (opts.no_think) @as(f32, 0.8) else 0.95),
        .min_p = opts.minp_arg orelse 0,
        .repeat_penalty = opts.penalty_arg orelse 1.0,
        .seed = opts.seed_arg orelse 0,
    } else .{
        .temperature = opts.temp_arg orelse 0,
        .top_k = opts.topk_arg orelse 0,
        .top_p = opts.topp_arg orelse 1.0,
        .min_p = opts.minp_arg orelse 0,
        .repeat_penalty = opts.penalty_arg orelse 1.0,
        .seed = opts.seed_arg orelse 0,
    };

    const spec_refs = opts.specRefs();

    // --json-schema/--lark/--regex: compile the grammar into a llguidance
    // constraint and install it as the sampler's logit processor. The mask
    // forces the stop/EOS token when the grammar completes, so the normal
    // stop handling ends generation.
    var grammar_text: ?[]u8 = null; // @FILE payloads (grammar borrows it through init only)
    defer if (grammar_text) |g| allocator.free(g);
    var constraint: ?llm.llguidance.Constraint = null;
    defer if (constraint) |*con| con.deinit();
    const grammar_flags = @as(usize, @intFromBool(opts.json_schema_arg != null)) +
        @intFromBool(opts.lark_arg != null) + @intFromBool(opts.regex_arg != null);
    if (grammar_flags > 1) {
        try stdout.print("--json-schema, --lark and --regex are mutually exclusive\n", .{});
        return error.ConflictingGrammarFlags;
    }
    if (grammar_flags == 1) {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        const grammar: llm.llguidance.Grammar = if (opts.json_schema_arg) |v|
            .{ .json_schema = try util.grammarValue(init.io, allocator, v, &grammar_text) }
        else if (opts.lark_arg) |v|
            .{ .lark = try util.grammarValue(init.io, allocator, v, &grammar_text) }
        else
            .{ .regex = opts.regex_arg.? };
        // Chat ends turns on the template stop marker; completion mode stops
        // on --stop or EOS (default --stop to EOS so a completed grammar
        // terminates generation instead of re-emitting EOS to the budget).
        const eos: ?u32 = if (is_chat) blk: {
            const tmpl = chat_tmpl orelse break :blk t.eosId();
            break :blk t.tokenId(tmpl.stopMarker()) orelse t.eosId();
        } else if (opts.stop_token) |s| @intCast(s) else t.eosId();
        if (!is_chat and opts.stop_token == null) opts.stop_token = if (eos) |e| @as(usize, e) else null;
        constraint = llm.llguidance.Constraint.init(allocator, t, grammar, .{
            .eos_token = eos,
            .n_vocab = model.config.vocab_size,
        }) catch |err| switch (err) {
            error.LlguidanceNotEnabled => {
                try stdout.print("constrained decoding needs a build with -Dllguidance=true (see vendor/llguidance/README.md)\n", .{});
                return err;
            },
            else => return err,
        };
    }
    const processor: ?llm.sampler.LogitProcessor = if (constraint) |*con| con.processor() else null;

    // --shine-fleet-build: batch-compile a docs directory into a served
    // adapter fleet (adapters + retrieval index + manifest).
    if (opts.shine_fleet_build) |out_dir| {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        const sh_path = opts.shine_gguf orelse {
            try stdout.print("--shine-fleet-build requires --shine <shine.gguf>\n", .{});
            return error.MissingShinePath;
        };
        const docs_dir = opts.shine_docs orelse {
            try stdout.print("--shine-fleet-build requires --shine-docs DIR\n", .{});
            return error.MissingShineDocsDir;
        };
        try shine_mode.buildFleet(init.io, allocator, stdout, &ctx, &model, t, sh_path, docs_dir, out_dir);
        return;
    }

    // --shine / --shine-adapter: hypernetwork mode — compile --shine-context
    // into a LoRA adapter (optionally --shine-save it) or reload a saved
    // one, then answer --chat / --repl questions with it (greedy).
    if (opts.shine_gguf != null or opts.shine_adapter != null) {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        try shine_mode.run(init.io, allocator, stdout, &ctx, &model, t, opts.shine_gguf, opts.shine_context, opts.shine_adapter, opts.shine_save, opts.shine_save_cartridge, opts.chat_text, opts.repl_flag, opts.gen_count orelse 128);
        return;
    }

    if (is_chat) {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        const tmpl = chat_tmpl orelse return error.NoChatTemplate;
        try chat.runChat(init.io, allocator, stdout, &ctx, &model, t, tmpl, opts.system_text, opts.no_think, sampler_cfg, opts.chat_text, opts.spec_flag, spec_refs, processor, kv_save_path);
        return;
    }

    // --prompt encodes raw text (completion mode) into the token stream.
    var prompt_ids_owned: ?[]usize = null;
    defer if (prompt_ids_owned) |p| allocator.free(p);
    if (opts.prompt_text) |pt| {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        const ids32 = try t.encode(allocator, pt);
        defer allocator.free(ids32);
        const ids = try allocator.alloc(usize, ids32.len);
        for (ids, ids32) |*d, s| d.* = s;
        prompt_ids_owned = ids;
        tokens = ids;
    }

    if (opts.spec_bench) {
        if (processor != null) return error.SpecBenchWithGrammar;
        try bench.runSpecBench(init.io, stdout, &ctx, &model, tokens, load_ns, opts.cache_type, @max(opts.bench_reps, 5));
        return;
    }
    if (opts.gen_count) |n| {
        if (opts.tokens_out != null and (opts.streams_arg > 1 or opts.spec_flag or opts.bench_reps > 1)) return error.TokensOutRequiresPlainGeneration;
        if (opts.streams_arg > 1) {
            if (opts.spec_flag) try stdout.print("note: --streams is the plain lockstep protocol; ignoring --spec\n", .{});
            if (opts.stop_token != null) try stdout.print("note: --streams ignores --stop (all streams run the full length)\n", .{});
            // Constrained multi-stream: each stream decodes with its own
            // clone of the base constraint (single-stream state).
            try generate.runGenerateStreams(init.io, allocator, stdout, &ctx, &model, tokens, load_ns, n, opts.streams_arg, opts.bench_reps, opts.cache_type, sampler_cfg, if (constraint) |*con| con else null);
            return;
        }
        if (opts.spec_flag and opts.bench_reps > 1) {
            try stdout.print("note: --bench is the plain-decode protocol; ignoring --spec\n", .{});
        }
        if (opts.spec_flag and opts.bench_reps == 1) {
            try generate.runGenerateSpec(init.io, allocator, stdout, &ctx, &model, tok_ptr, tokens, load_ns, n, opts.stop_token, sampler_cfg, opts.cache_type, spec_refs, processor);
            return;
        }
        try generate.runGenerate(init.io, allocator, stdout, &ctx, &model, tok_ptr, tokens, load_ns, n, opts.stop_token, opts.profile_enabled, sampler_cfg, opts.bench_reps, opts.cache_type, processor, opts.tokens_out);
        return;
    }
    if (opts.verify_count) |m| {
        try verify.runVerifyCache(allocator, stdout, &ctx, &model, tokens, load_ns, m, opts.cache_type);
        return;
    }
    if (opts.verify_batch_count) |m| {
        try verify.runVerifyBatch(allocator, stdout, &ctx, &model, tokens, load_ns, m, opts.cache_type);
        return;
    }

    var logits: ?fucina.Tensor(.{ .seq, .vocab }) = null;
    defer if (logits) |*value| value.deinit();
    var profile: llm.qwen3.model.ForwardProfile = .{};

    const forward_start = util.nowNs(init.io);
    for (0..opts.repeat) |_| {
        if (logits) |*value| {
            value.deinit();
            logits = null;
        }
        logits = if (opts.profile_enabled)
            try model.forwardLastLogitsProfiled(&ctx, init.io, tokens, &profile)
        else
            try model.forwardLastLogits(&ctx, tokens);
    }
    const forward_ns = util.nowNs(init.io) - forward_start;
    const final_logits = &(logits orelse return error.MissingLogits);

    var top = try final_logits.topK(&ctx, .vocab, 5, .top);
    defer top.deinit();

    try stdout.print("tokens: {d}\n", .{tokens.len});
    try stdout.print("load: {d:.3} s\n", .{util.seconds(load_ns)});
    if (opts.repeat == 1) {
        try stdout.print("forward: {d:.3} s\n", .{util.seconds(forward_ns)});
    } else {
        try stdout.print("forward: {d:.3} s total, {d:.3} ms avg over {d}\n", .{ util.seconds(forward_ns), util.millis(forward_ns) / @as(f64, @floatFromInt(opts.repeat)), opts.repeat });
    }
    try stdout.print("top tokens:", .{});
    const top_values = try top.values.dataConst();
    const top_indices = try top.indices.dataConst();
    for (top_values, top_indices) |value, index| {
        try stdout.print(" {d}:{d:.4}", .{ index, value });
    }
    try stdout.print("\n", .{});
    if (opts.profile_enabled) {
        try bench.printProfile(stdout, &profile, opts.repeat);
    }

    if (opts.logits_out) |path| {
        try util.writeLogits(init.io, path, try final_logits.dataConst());
        try stdout.print("logits: {s}\n", .{path});
    }

    if (opts.compare_logits) |path| {
        try verify.compareLogits(init.io, allocator, stdout, path, try final_logits.dataConst());
    }
}

fn printInfo(stdout: anytype, file: *const fucina.gguf.File, tok: ?*const llm.tokenizer.Tokenizer) !void {
    if (tok) |t| {
        try stdout.print("vocab: {d}  bos: {?d}  eos: {?d}\n", .{ t.vocabSize(), t.bosId(), t.eosId() });
        for ([_][]const u8{ "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>" }) |s| {
            try stdout.print("  {s} = {?d}\n", .{ s, t.tokenId(s) });
        }
    } else {
        try stdout.print("(no tokenizer in this GGUF)\n", .{});
    }
    if (file.getString("tokenizer.chat_template")) |tmpl| {
        try stdout.print("--- chat_template ({d} chars) ---\n{s}\n", .{ tmpl.len, tmpl });
    } else {
        try stdout.print("(no chat_template metadata)\n", .{});
    }
}
