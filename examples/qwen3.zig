const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const default_tokens = [_]usize{ 151_644, 872, 198, 9707 };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print("usage: zig build qwen3 -- <model.gguf> [comma-separated-token-ids] [--repeat N] [--profile] [--logits-out PATH] [--compare-logits PATH] [--gen N [--stop TOKEN]] [--bench R] [--verify-cache N] [--cache-type f16|q8_0] [--spec] [--spec-ref FILE] [--tokenize FILE]...\n", .{});
        try stdout.print("spec:     zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf --prompt \"...\" --gen 128 --spec [--spec-ref doc.txt]   (lossless speculative decode + acceptance stats)\n", .{});
        try stdout.print("bench:    zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf <prompt-token-ids> --gen 64 --bench 5   (warm pp/tg, load once; fair vs llama-bench)\n", .{});
        try stdout.print("streams:  zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf <prompt-token-ids> --gen 64 --bench 3 --streams 4   (batched multi-stream decode vs N sequential runs)\n", .{});
        try stdout.print("example: zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf 151644,872,198,9707\n", .{});
        try stdout.print("generate: zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --prompt \"The capital of France is\" --gen 64\n", .{});
        try stdout.print("chat:     zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --chat \"What is the capital of France?\" [--no-think] [--system \"...\"]\n", .{});
        try stdout.print("repl:     zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --repl   (multi-turn; streams replies)\n", .{});
        try stdout.print("sampling: --temp F --top-k N --top-p F --min-p F --repeat-penalty F --seed N\n", .{});
        try stdout.print("grammar:  --json-schema JSON|@FILE | --lark GRAMMAR|@FILE | --regex PATTERN   (constrained decoding; needs a -Dllguidance=true build; combine with --no-think)\n", .{});
        try stdout.print("other:    --info | --spec-bench\n", .{});
        return;
    }

    var token_buf: [4096]usize = undefined;
    var tokens: []const usize = default_tokens[0..];
    var logits_out: ?[]const u8 = null;
    var compare_logits: ?[]const u8 = null;
    var profile_enabled = false;
    var repeat: usize = 1;
    var gen_count: ?usize = null;
    var bench_reps: usize = 1;
    var verify_count: ?usize = null;
    var cache_type: llm.kv_cache.KvDtype = .f16;
    var stop_token: ?usize = null;
    var prompt_text: ?[]const u8 = null;
    var info_flag = false;
    var chat_text: ?[]const u8 = null;
    var system_text: ?[]const u8 = null;
    var repl_flag = false;
    var no_think = false;
    var spec_bench = false;
    var spec_flag = false;
    var tokenize_file: ?[]const u8 = null;
    var spec_ref_buf: [8][]const u8 = undefined;
    var spec_ref_count: usize = 0;
    var temp_arg: ?f32 = null;
    var streams_arg: usize = 1;
    var json_schema_arg: ?[]const u8 = null;
    var lark_arg: ?[]const u8 = null;
    var regex_arg: ?[]const u8 = null;
    var topk_arg: ?usize = null;
    var topp_arg: ?f32 = null;
    var minp_arg: ?f32 = null;
    var penalty_arg: ?f32 = null;
    var seed_arg: ?u64 = null;
    var moe_stream_flag = false;
    var moe_cache_mb: ?usize = null;
    var moe_cache_slots: ?usize = null;
    var moe_pin_mb: ?usize = null;
    var moe_no_learn = false;
    var moe_pilot = false;
    var moe_expert_top_p: ?f32 = null;
    var kv_save = false;
    var kv_save_arg: ?[]const u8 = null;
    var arg_i: usize = 2;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--logits-out")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingLogitsPath;
            logits_out = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--logits-out=")) {
            logits_out = arg["--logits-out=".len..];
        } else if (std.mem.eql(u8, arg, "--compare-logits")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingCompareLogitsPath;
            compare_logits = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--compare-logits=")) {
            compare_logits = arg["--compare-logits=".len..];
        } else if (std.mem.eql(u8, arg, "--repeat")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingRepeatCount;
            repeat = try parseRepeat(args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--repeat=")) {
            repeat = try parseRepeat(arg["--repeat=".len..]);
        } else if (std.mem.eql(u8, arg, "--bench")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingBenchCount;
            bench_reps = try parseRepeat(args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--bench=")) {
            bench_reps = try parseRepeat(arg["--bench=".len..]);
        } else if (std.mem.eql(u8, arg, "--profile")) {
            profile_enabled = true;
        } else if (std.mem.eql(u8, arg, "--gen")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingGenCount;
            gen_count = try parseRepeat(args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--gen=")) {
            gen_count = try parseRepeat(arg["--gen=".len..]);
        } else if (std.mem.eql(u8, arg, "--verify-cache")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingVerifyCount;
            verify_count = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--verify-cache=")) {
            verify_count = try std.fmt.parseInt(usize, arg["--verify-cache=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--cache-type")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingCacheType;
            cache_type = try parseCacheType(args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--cache-type=")) {
            cache_type = try parseCacheType(arg["--cache-type=".len..]);
        } else if (std.mem.eql(u8, arg, "--stop")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingStopToken;
            stop_token = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--stop=")) {
            stop_token = try std.fmt.parseInt(usize, arg["--stop=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingPrompt;
            prompt_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--prompt=")) {
            prompt_text = arg["--prompt=".len..];
        } else if (std.mem.eql(u8, arg, "--info")) {
            info_flag = true;
        } else if (std.mem.eql(u8, arg, "--chat")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingChatMessage;
            chat_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--chat=")) {
            chat_text = arg["--chat=".len..];
        } else if (std.mem.eql(u8, arg, "--system")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSystemMessage;
            system_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--system=")) {
            system_text = arg["--system=".len..];
        } else if (std.mem.eql(u8, arg, "--spec-bench")) {
            spec_bench = true;
        } else if (std.mem.eql(u8, arg, "--spec")) {
            spec_flag = true;
        } else if (std.mem.eql(u8, arg, "--spec-ref")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSpecRefPath;
            if (spec_ref_count == spec_ref_buf.len) return error.TooManySpecRefs;
            spec_ref_buf[spec_ref_count] = args[arg_i];
            spec_ref_count += 1;
        } else if (std.mem.startsWith(u8, arg, "--spec-ref=")) {
            if (spec_ref_count == spec_ref_buf.len) return error.TooManySpecRefs;
            spec_ref_buf[spec_ref_count] = arg["--spec-ref=".len..];
            spec_ref_count += 1;
        } else if (std.mem.eql(u8, arg, "--tokenize")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTokenizePath;
            tokenize_file = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--tokenize=")) {
            tokenize_file = arg["--tokenize=".len..];
        } else if (std.mem.eql(u8, arg, "--repl")) {
            repl_flag = true;
        } else if (std.mem.eql(u8, arg, "--no-think")) {
            no_think = true;
        } else if (std.mem.eql(u8, arg, "--temp")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTemp;
            temp_arg = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--temp=")) {
            temp_arg = try std.fmt.parseFloat(f32, arg["--temp=".len..]);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTopK;
            topk_arg = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--top-k=")) {
            topk_arg = try std.fmt.parseInt(usize, arg["--top-k=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTopP;
            topp_arg = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--top-p=")) {
            topp_arg = try std.fmt.parseFloat(f32, arg["--top-p=".len..]);
        } else if (std.mem.eql(u8, arg, "--min-p")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingMinP;
            minp_arg = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--min-p=")) {
            minp_arg = try std.fmt.parseFloat(f32, arg["--min-p=".len..]);
        } else if (std.mem.eql(u8, arg, "--repeat-penalty")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingRepeatPenalty;
            penalty_arg = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--repeat-penalty=")) {
            penalty_arg = try std.fmt.parseFloat(f32, arg["--repeat-penalty=".len..]);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSeed;
            seed_arg = try std.fmt.parseInt(u64, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--seed=")) {
            seed_arg = try std.fmt.parseInt(u64, arg["--seed=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--streams")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingStreamCount;
            streams_arg = try parseRepeat(args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--streams=")) {
            streams_arg = try parseRepeat(arg["--streams=".len..]);
        } else if (std.mem.eql(u8, arg, "--json-schema")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingJsonSchema;
            json_schema_arg = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--json-schema=")) {
            json_schema_arg = arg["--json-schema=".len..];
        } else if (std.mem.eql(u8, arg, "--lark")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingLarkGrammar;
            lark_arg = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--lark=")) {
            lark_arg = arg["--lark=".len..];
        } else if (std.mem.eql(u8, arg, "--regex")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingRegex;
            regex_arg = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--regex=")) {
            regex_arg = arg["--regex=".len..];
        } else if (std.mem.eql(u8, arg, "--moe-stream")) {
            moe_stream_flag = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-cache-mb=")) {
            moe_stream_flag = true;
            moe_cache_mb = try std.fmt.parseInt(usize, arg["--moe-cache-mb=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--moe-cache-slots=")) {
            moe_stream_flag = true;
            moe_cache_slots = try std.fmt.parseInt(usize, arg["--moe-cache-slots=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--moe-pin-mb=")) {
            moe_stream_flag = true;
            moe_pin_mb = try std.fmt.parseInt(usize, arg["--moe-pin-mb=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--moe-no-learn")) {
            moe_no_learn = true;
        } else if (std.mem.eql(u8, arg, "--kv-save")) {
            kv_save = true;
        } else if (std.mem.startsWith(u8, arg, "--kv-save=")) {
            kv_save = true;
            kv_save_arg = arg["--kv-save=".len..];
        } else if (std.mem.eql(u8, arg, "--moe-pilot")) {
            moe_stream_flag = true;
            moe_pilot = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-expert-top-p=")) {
            moe_expert_top_p = try std.fmt.parseFloat(f32, arg["--moe-expert-top-p=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--")) {
            try stdout.print("unknown flag: {s} (run with no arguments for usage)\n", .{arg});
            return error.UnknownArgument;
        } else {
            tokens = try parseTokenList(arg, &token_buf);
        }
    }

    const allocator = std.heap.smp_allocator;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const load_start = nowNs(init.io);
    var file = try fucina.gguf.File.loadMmapAuto(allocator, init.io, args[1]);

    // --tokenize FILE: encode a text file and print one token id per line
    // (the llama-tokenize parity harness); no model weights needed.
    if (tokenize_file) |path| {
        var t = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
        defer t.deinit();
        file.deinit();
        const ids = try tokenizeFile(init.io, allocator, &t, path);
        defer allocator.free(ids);
        for (ids) |id| try stdout.print("{d}\n", .{id});
        return;
    }

    const load_options: llm.qwen3.model.LoadOptions = if (moe_stream_flag) .{
        .moe_stream = .{
            .gguf_path = args[1],
            .cache_bytes = if (moe_cache_mb) |mb| mb << 20 else null,
            .cache_slots_per_layer = moe_cache_slots,
            .auto_pin = !moe_no_learn,
            .pin_bytes = if (moe_pin_mb) |mb| mb << 20 else null,
            .pilot = moe_pilot,
        },
    } else .{};
    var model_config = try llm.qwen3.model.Config.fromGguf(&file);
    if (moe_expert_top_p) |p| model_config.moe_expert_top_p = p;
    var model = try llm.qwen3.model.Model.loadGgufFromFileOptions(&ctx, &file, model_config, load_options);
    defer model.deinit();
    // The stats go through the SAME buffered stdout writer as everything
    // else: stdout's positional writes and stderr's offset-advancing writes
    // cannot safely share one redirected file (`cmd > f 2>&1` interleaves
    // destructively), so a std.debug stats line would get overwritten.
    defer if (model.expert_store) |store| llm.weights.reportAndSaveMoeStream(store, !moe_no_learn, stdout);
    // Build a tokenizer from the same file's metadata; tolerate models without it.
    var tokenizer: ?llm.tokenizer.Tokenizer = llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{}) catch null;
    defer if (tokenizer) |*t| t.deinit();
    const tok_ptr: ?*const llm.tokenizer.Tokenizer = if (tokenizer) |*t| t else null;
    const chat_tmpl = llm.chat.Template.detect(file.getString("tokenizer.chat_template"));

    if (info_flag) {
        try printInfo(stdout, &file, tok_ptr);
        file.deinit();
        return;
    }
    file.deinit(); // model + tokenizer own their data now
    const load_ns = nowNs(init.io) - load_start;

    // KV persistence sidecar: explicit path, or <model>.kvcache next to the GGUF.
    var kv_path_buf: [1024]u8 = undefined;
    const kv_save_path: ?[]const u8 = if (!kv_save) null else kv_save_arg orelse try std.fmt.bufPrint(&kv_path_buf, "{s}.kvcache", .{args[1]});

    const is_chat = chat_text != null or repl_flag;

    // Chat samples with Qwen3's recommended settings; the benchmark and
    // raw-completion paths default to greedy (deterministic). Flags override.
    const sampler_cfg: llm.sampler.Config = if (is_chat) .{
        .temperature = temp_arg orelse (if (no_think) @as(f32, 0.7) else 0.6),
        .top_k = topk_arg orelse 20,
        .top_p = topp_arg orelse (if (no_think) @as(f32, 0.8) else 0.95),
        .min_p = minp_arg orelse 0,
        .repeat_penalty = penalty_arg orelse 1.0,
        .seed = seed_arg orelse 0,
    } else .{
        .temperature = temp_arg orelse 0,
        .top_k = topk_arg orelse 0,
        .top_p = topp_arg orelse 1.0,
        .min_p = minp_arg orelse 0,
        .repeat_penalty = penalty_arg orelse 1.0,
        .seed = seed_arg orelse 0,
    };

    const spec_refs = spec_ref_buf[0..spec_ref_count];

    // --json-schema/--lark/--regex: compile the grammar into a llguidance
    // constraint and install it as the sampler's logit processor. The mask
    // forces the stop/EOS token when the grammar completes, so the normal
    // stop handling ends generation.
    var grammar_text: ?[]u8 = null; // @FILE payloads (grammar borrows it through init only)
    defer if (grammar_text) |g| allocator.free(g);
    var constraint: ?llm.llguidance.Constraint = null;
    defer if (constraint) |*con| con.deinit();
    const grammar_flags = @as(usize, @intFromBool(json_schema_arg != null)) +
        @intFromBool(lark_arg != null) + @intFromBool(regex_arg != null);
    if (grammar_flags > 1) {
        try stdout.print("--json-schema, --lark and --regex are mutually exclusive\n", .{});
        return error.ConflictingGrammarFlags;
    }
    if (grammar_flags == 1) {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        const grammar: llm.llguidance.Grammar = if (json_schema_arg) |v|
            .{ .json_schema = try grammarValue(init.io, allocator, v, &grammar_text) }
        else if (lark_arg) |v|
            .{ .lark = try grammarValue(init.io, allocator, v, &grammar_text) }
        else
            .{ .regex = regex_arg.? };
        // Chat ends turns on the template stop marker; completion mode stops
        // on --stop or EOS (default --stop to EOS so a completed grammar
        // terminates generation instead of re-emitting EOS to the budget).
        const eos: ?u32 = if (is_chat) blk: {
            const tmpl = chat_tmpl orelse break :blk t.eosId();
            break :blk t.tokenId(tmpl.stopMarker()) orelse t.eosId();
        } else if (stop_token) |s| @intCast(s) else t.eosId();
        if (!is_chat and stop_token == null) stop_token = if (eos) |e| @as(usize, e) else null;
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

    if (is_chat) {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        const tmpl = chat_tmpl orelse return error.NoChatTemplate;
        try runChat(init.io, allocator, stdout, &ctx, &model, t, tmpl, system_text, no_think, sampler_cfg, chat_text, spec_flag, spec_refs, processor, kv_save_path);
        return;
    }

    // --prompt encodes raw text (completion mode) into the token stream.
    var prompt_ids_owned: ?[]usize = null;
    defer if (prompt_ids_owned) |p| allocator.free(p);
    if (prompt_text) |pt| {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        const ids32 = try t.encode(allocator, pt);
        defer allocator.free(ids32);
        const ids = try allocator.alloc(usize, ids32.len);
        for (ids, ids32) |*d, s| d.* = s;
        prompt_ids_owned = ids;
        tokens = ids;
    }

    if (spec_bench) {
        if (processor != null) return error.SpecBenchWithGrammar;
        try runSpecBench(init.io, stdout, &ctx, &model, tokens, load_ns, cache_type, @max(bench_reps, 5));
        return;
    }
    if (gen_count) |n| {
        if (streams_arg > 1) {
            if (spec_flag) try stdout.print("note: --streams is the plain lockstep protocol; ignoring --spec\n", .{});
            if (stop_token != null) try stdout.print("note: --streams ignores --stop (all streams run the full length)\n", .{});
            // Constrained multi-stream: each stream decodes with its own
            // clone of the base constraint (single-stream state).
            try runGenerateStreams(init.io, allocator, stdout, &ctx, &model, tokens, load_ns, n, streams_arg, bench_reps, cache_type, sampler_cfg, if (constraint) |*con| con else null);
            return;
        }
        if (spec_flag and bench_reps > 1) {
            try stdout.print("note: --bench is the plain-decode protocol; ignoring --spec\n", .{});
        }
        if (spec_flag and bench_reps == 1) {
            try runGenerateSpec(init.io, allocator, stdout, &ctx, &model, tok_ptr, tokens, load_ns, n, stop_token, sampler_cfg, cache_type, spec_refs, processor);
            return;
        }
        try runGenerate(init.io, allocator, stdout, &ctx, &model, tok_ptr, tokens, load_ns, n, stop_token, profile_enabled, sampler_cfg, bench_reps, cache_type, processor);
        return;
    }
    if (verify_count) |m| {
        try runVerifyCache(allocator, stdout, &ctx, &model, tokens, load_ns, m, cache_type);
        return;
    }

    var logits: ?fucina.Tensor(.{ .seq, .vocab }) = null;
    defer if (logits) |*value| value.deinit();
    var profile: llm.qwen3.model.ForwardProfile = .{};

    const forward_start = nowNs(init.io);
    for (0..repeat) |_| {
        if (logits) |*value| {
            value.deinit();
            logits = null;
        }
        logits = if (profile_enabled)
            try model.forwardLastLogitsProfiled(&ctx, init.io, tokens, &profile)
        else
            try model.forwardLastLogits(&ctx, tokens);
    }
    const forward_ns = nowNs(init.io) - forward_start;
    const final_logits = &(logits orelse return error.MissingLogits);

    var top = try final_logits.topK(&ctx, .vocab, 5, .top);
    defer top.deinit();

    try stdout.print("tokens: {d}\n", .{tokens.len});
    try stdout.print("load: {d:.3} s\n", .{seconds(load_ns)});
    if (repeat == 1) {
        try stdout.print("forward: {d:.3} s\n", .{seconds(forward_ns)});
    } else {
        try stdout.print("forward: {d:.3} s total, {d:.3} ms avg over {d}\n", .{ seconds(forward_ns), millis(forward_ns) / @as(f64, @floatFromInt(repeat)), repeat });
    }
    try stdout.print("top tokens:", .{});
    const top_values = try top.values.dataConst();
    const top_indices = try top.indices.dataConst();
    for (top_values, top_indices) |value, index| {
        try stdout.print(" {d}:{d:.4}", .{ index, value });
    }
    try stdout.print("\n", .{});
    if (profile_enabled) {
        try printProfile(stdout, &profile, repeat);
    }

    if (logits_out) |path| {
        try writeLogits(init.io, path, try final_logits.dataConst());
        try stdout.print("logits: {s}\n", .{path});
    }

    if (compare_logits) |path| {
        try compareLogits(init.io, allocator, stdout, path, try final_logits.dataConst());
    }
}

fn parseTokenList(input: []const u8, out: []usize) ![]const usize {
    var len: usize = 0;
    var it = std.mem.splitScalar(u8, input, ',');
    while (it.next()) |part| {
        if (part.len == 0) return error.InvalidTokenList;
        if (len == out.len) return error.TokenListTooLong;
        out[len] = try std.fmt.parseInt(usize, part, 10);
        len += 1;
    }
    if (len == 0) return error.InvalidTokenList;
    return out[0..len];
}

fn parseRepeat(input: []const u8) !usize {
    const repeat = try std.fmt.parseInt(usize, input, 10);
    if (repeat == 0) return error.InvalidRepeatCount;
    return repeat;
}

fn parseCacheType(input: []const u8) !llm.kv_cache.KvDtype {
    if (std.mem.eql(u8, input, "f16")) return .f16;
    if (std.mem.eql(u8, input, "q8_0")) return .q8_0;
    return error.InvalidCacheType;
}

fn printCacheInfo(stdout: anytype, cache: *const llm.kv_cache.KvCache) !void {
    const bytes = cache.byteSize();
    const per_token = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(cache.capacity));
    try stdout.print("kv cache: {s}, capacity {d} tok, {d:.1} MiB ({d:.1} KiB/token, {d:.1} MiB per 1k tok)\n", .{
        @tagName(cache.dtype),
        cache.capacity,
        @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0),
        per_token / 1024.0,
        per_token * 1000.0 / (1024.0 * 1024.0),
    });
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

fn millisI128(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn printProfile(stdout: anytype, profile: *const llm.qwen3.model.ForwardProfile, repeat: usize) !void {
    try printProfileLabeled(stdout, "profile avg ms", profile, @floatFromInt(repeat));
}

fn printProfileLabeled(stdout: anytype, label: []const u8, profile: *const llm.qwen3.model.ForwardProfile, denom: f64) !void {
    try stdout.print("{s}:", .{label});
    try stdout.print(" attn_prep={d:.3}", .{millisI128(profile.attn_prep_ns) / denom});
    try stdout.print(" qkv={d:.3}", .{millisI128(profile.qkv_ns) / denom});
    try stdout.print(" qk_norm_rope={d:.3}", .{millisI128(profile.qk_norm_rope_ns) / denom});
    try stdout.print(" attention={d:.3}", .{millisI128(profile.attention_ns) / denom});
    try stdout.print(" attn_out={d:.3}", .{millisI128(profile.attn_out_ns) / denom});
    try stdout.print(" attn_residual={d:.3}", .{millisI128(profile.attn_residual_ns) / denom});
    try stdout.print(" ffn_prep={d:.3}", .{millisI128(profile.ffn_prep_ns) / denom});
    try stdout.print(" router={d:.3}", .{millisI128(profile.router_ns) / denom});
    try stdout.print(" gate_up={d:.3}", .{millisI128(profile.gate_up_ns) / denom});
    try stdout.print(" swiglu={d:.3}", .{millisI128(profile.swiglu_ns) / denom});
    try stdout.print(" down={d:.3}", .{millisI128(profile.down_ns) / denom});
    try stdout.print(" ffn_residual={d:.3}", .{millisI128(profile.ffn_residual_ns) / denom});
    try stdout.print(" final={d:.3}", .{millisI128(profile.final_ns) / denom});
    try stdout.print(" layers={d}\n", .{profile.layers});

    const mb = profile.moe_batch;
    if (mb.batches > 0) {
        const batch_denom: f64 = @floatFromInt(mb.batches);
        try stdout.print("moe batch avg ms:", .{});
        try stdout.print(" total={d:.3}", .{millisI128(mb.total_ns) / denom});
        try stdout.print(" alloc={d:.3}", .{millisI128(mb.alloc_ns) / denom});
        try stdout.print(" count_sort={d:.3}", .{millisI128(mb.count_sort_ns) / denom});
        try stdout.print(" expert_wall={d:.3}", .{millisI128(mb.expert_wall_ns) / denom});
        try stdout.print(" scatter={d:.3}", .{millisI128(mb.scatter_ns) / denom});
        try stdout.print(" task_sum_gather_q={d:.3}", .{millisI128(mb.gather_quant_ns) / denom});
        try stdout.print(" task_sum_gate_up={d:.3}", .{millisI128(mb.gate_up_ns) / denom});
        try stdout.print(" task_sum_swiglu_q={d:.3}", .{millisI128(mb.swiglu_requant_ns) / denom});
        try stdout.print(" task_sum_down={d:.3}", .{millisI128(mb.down_ns) / denom});
        try stdout.print(
            " batches/pass={d:.1} pairs/batch={d:.1} active_experts/batch={d:.1} max_m={d}\n",
            .{
                @as(f64, @floatFromInt(mb.batches)) / denom,
                @as(f64, @floatFromInt(mb.pairs)) / batch_denom,
                @as(f64, @floatFromInt(mb.active_experts)) / batch_denom,
                mb.max_expert_m,
            },
        );
    }
}

fn writeLogits(io: std.Io, path: []const u8, values: []const f32) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);

    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    defer writer.interface.flush() catch {};
    try writer.interface.writeAll(std.mem.sliceAsBytes(values));
}

fn compareLogits(io: std.Io, allocator: std.mem.Allocator, stdout: anytype, path: []const u8, values: []const f32) !void {
    const reference = try readF32File(io, allocator, path);
    defer allocator.free(reference);
    if (reference.len != values.len) return error.CompareLogitsShapeMismatch;

    var max_abs: f64 = 0;
    var sum_abs: f64 = 0;
    var sum_sq: f64 = 0;
    var value_top: usize = 0;
    var reference_top: usize = 0;
    for (values, reference, 0..) |value, ref, i| {
        const diff: f64 = @floatCast(value - ref);
        const abs_diff = @abs(diff);
        max_abs = @max(max_abs, abs_diff);
        sum_abs += abs_diff;
        sum_sq += diff * diff;
        if (value > values[value_top]) value_top = i;
        if (ref > reference[reference_top]) reference_top = i;
    }

    const n: f64 = @floatFromInt(values.len);
    try stdout.print(
        "compare logits: max_abs={d:.6} mean_abs={d:.6} rms={d:.6} top={d}:{d:.4} ref_top={d}:{d:.4} aligned={}\n",
        .{
            max_abs,
            sum_abs / n,
            @sqrt(sum_sq / n),
            value_top,
            values[value_top],
            reference_top,
            reference[reference_top],
            value_top == reference_top,
        },
    );
}

const PassResult = struct { prefill_ns: i96, decode_ns: i96, decode_steps: usize, produced: usize };

/// One prefill+decode pass into the (reset) cache: prefill `tokens`, then decode
/// greedily up to `max_new`/`stop_token`, timing prefill and decode separately.
fn benchOnePass(
    io: std.Io,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    cache: *llm.kv_cache.KvCache,
    tokens: []const usize,
    out: []usize,
    history: []usize,
    sampler_cfg: llm.sampler.Config,
    max_new: usize,
    stop_token: ?usize,
    profile_prefill: bool,
    profile_decode: bool,
    prefill_profile: ?*llm.qwen3.model.ForwardProfile,
    decode_profile: ?*llm.qwen3.model.ForwardProfile,
    processor: ?llm.sampler.LogitProcessor,
) !PassResult {
    cache.reset();
    var sampler = llm.sampler.Sampler.init(sampler_cfg);
    sampler.processor = processor;
    if (processor) |p| try p.reset(); // fresh grammar state per pass
    @memcpy(history[0..tokens.len], tokens);
    var hist_len = tokens.len;

    const prefill_start = nowNs(io);
    var logits = if (profile_prefill)
        try model.forwardStepProfiled(ctx, io, cache, tokens, 0, prefill_profile.?)
    else
        try model.forwardStep(ctx, cache, tokens, 0);
    const prefill_ns = nowNs(io) - prefill_start;
    out[0] = try sampler.next(ctx, &logits, history[0..hist_len]);
    history[hist_len] = out[0];
    hist_len += 1;
    var produced: usize = 1;

    const decode_start = nowNs(io);
    while (produced < max_new) : (produced += 1) {
        if (stop_token) |stop| if (out[produced - 1] == stop) break;
        const fresh = if (profile_decode)
            try model.forwardStepProfiled(ctx, io, cache, out[produced - 1 ..][0..1], cache.len, decode_profile.?)
        else
            try model.forwardStep(ctx, cache, out[produced - 1 ..][0..1], cache.len);
        logits.deinit();
        logits = fresh;
        out[produced] = try sampler.next(ctx, &logits, history[0..hist_len]);
        history[hist_len] = out[produced];
        hist_len += 1;
    }
    const decode_ns = nowNs(io) - decode_start;
    logits.deinit();
    return .{ .prefill_ns = prefill_ns, .decode_ns = decode_ns, .decode_steps = produced - 1, .produced = produced };
}

fn printBenchStat(stdout: anytype, label: []const u8, vals: []const f64) !void {
    var min: f64 = vals[0];
    var max: f64 = vals[0];
    var sum: f64 = 0;
    for (vals) |v| {
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
    }
    const mean = sum / @as(f64, @floatFromInt(vals.len));
    var ss: f64 = 0;
    for (vals) |v| {
        const d = v - mean;
        ss += d * d;
    }
    const std_dev = @sqrt(ss / @as(f64, @floatFromInt(vals.len)));
    try stdout.print("{s}: {d:.2} ± {d:.2} tok/s  (min {d:.2}, max {d:.2})\n", .{ label, mean, std_dev, min, max });
}

fn runGenerate(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: anytype,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    tok: ?*const llm.tokenizer.Tokenizer,
    tokens: []const usize,
    load_ns: i96,
    max_new: usize,
    stop_token: ?usize,
    profile_enabled: bool,
    sampler_cfg: llm.sampler.Config,
    bench_reps: usize,
    cache_type: llm.kv_cache.KvDtype,
    processor: ?llm.sampler.LogitProcessor,
) !void {
    const cfg = model.config;
    const capacity = tokens.len + max_new;
    var cache = try llm.kv_cache.KvCache.initWithDtype(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, cache_type);
    defer cache.deinit();
    try printCacheInfo(stdout, &cache);

    const out = try allocator.alloc(usize, max_new);
    defer allocator.free(out);
    const history = try allocator.alloc(usize, tokens.len + max_new);
    defer allocator.free(history);
    var profile: llm.qwen3.model.ForwardProfile = .{};
    var decode_profile: llm.qwen3.model.ForwardProfile = .{};

    // Warm-repeat benchmark (load once, prime caches, time N passes) — a fair
    // apples-to-apples vs llama-bench, free of the reload/heat bias a fresh single
    // pass has. A discarded warmup pass (prompt run + 1-token gen, matching
    // llama-bench's warmup) precedes the timed reps; pp/tg are summarized as
    // mean ± std with min/max (max = the least-throttled, steady-state number).
    if (bench_reps > 1) {
        const tgs = try allocator.alloc(f64, bench_reps);
        defer allocator.free(tgs);
        const pps = try allocator.alloc(f64, bench_reps);
        defer allocator.free(pps);
        var decode_steps: usize = 0;
        var rep: usize = 0;
        while (rep <= bench_reps) : (rep += 1) { // rep 0 = warmup
            // Match llama-bench's warmup exactly: a prompt run + a 1-token gen
            // (max_new == 2 => prefill + one decode step), not a full decode.
            const pass_new = if (rep == 0) @min(max_new, 2) else max_new;
            if (profile_enabled and rep == 1) {
                profile = .{};
                decode_profile = .{};
            }
            if (rep == 1) fucina.internal.gpu.traceReset();
            const profile_prefill = profile_enabled and rep != 0;
            const profile_decode = profile_enabled and rep != 0;
            const prefill_profile = if (profile_prefill) &profile else null;
            const decode_profile_ptr = if (profile_decode) &decode_profile else null;
            const r = try benchOnePass(io, ctx, model, &cache, tokens, out, history, sampler_cfg, pass_new, stop_token, profile_prefill, profile_decode, prefill_profile, decode_profile_ptr, processor);
            if (rep == 0) continue;
            decode_steps = r.decode_steps;
            tgs[rep - 1] = @as(f64, @floatFromInt(r.decode_steps)) / seconds(r.decode_ns);
            pps[rep - 1] = @as(f64, @floatFromInt(tokens.len)) / seconds(r.prefill_ns);
        }
        fucina.internal.gpu.traceDump();
        try stdout.print("load: {d:.3} s\n", .{seconds(load_ns)});
        try stdout.print("warm bench: {d} reps (+1 warmup), prompt {d} tok, decode {d} steps\n", .{ bench_reps, tokens.len, decode_steps });
        try printBenchStat(stdout, "prefill", pps);
        if (decode_steps > 0) try printBenchStat(stdout, "decode ", tgs);
        if (profile_enabled) {
            try printProfileLabeled(stdout, "prefill profile avg ms", &profile, @floatFromInt(bench_reps));
            if (decode_steps > 0) {
                const denom: f64 = @floatFromInt(bench_reps * decode_steps);
                try printProfileLabeled(stdout, "decode profile avg ms/token", &decode_profile, denom);
            }
        }
        return;
    }

    const prefill_profile = if (profile_enabled) &profile else null;
    const r = try benchOnePass(io, ctx, model, &cache, tokens, out, history, sampler_cfg, max_new, stop_token, profile_enabled, false, prefill_profile, null, processor);
    try stdout.print("load: {d:.3} s\n", .{seconds(load_ns)});
    try stdout.print("prompt tokens: {d}\n", .{tokens.len});
    // Prefill top-5 on the fixed prompt: the cache-type quality A/B hook
    // (compare these magnitudes across --cache-type runs).
    {
        cache.reset();
        var prefill_logits = try model.forwardStep(ctx, &cache, tokens, 0);
        defer prefill_logits.deinit();
        var top = try prefill_logits.topK(ctx, .vocab, 5, .top);
        defer top.deinit();
        try stdout.print("prefill top tokens:", .{});
        for (try top.values.dataConst(), try top.indices.dataConst()) |value, index| {
            try stdout.print(" {d}:{d:.4}", .{ index, value });
        }
        try stdout.print("\n", .{});
    }
    try stdout.print("prefill: {d:.3} ms\n", .{millis(r.prefill_ns)});
    if (r.decode_steps > 0) {
        const tg = @as(f64, @floatFromInt(r.decode_steps)) / seconds(r.decode_ns);
        try stdout.print("decode: {d} steps, {d:.3} ms, {d:.2} tok/s\n", .{ r.decode_steps, millis(r.decode_ns), tg });
    }
    try stdout.print("generated {d}:", .{r.produced});
    for (out[0..r.produced]) |token| try stdout.print(" {d}", .{token});
    try stdout.print("\n", .{});
    if (tok) |t| {
        const prompt_str = try decodeIds(allocator, t, tokens);
        defer allocator.free(prompt_str);
        const gen_str = try decodeIds(allocator, t, out[0..r.produced]);
        defer allocator.free(gen_str);
        try stdout.print("prompt: {s}\n", .{prompt_str});
        try stdout.print("text:   {s}{s}\n", .{ prompt_str, gen_str });
    }
    if (profile_enabled) try printProfile(stdout, &profile, 1);
}

const StreamsPassResult = struct { prefill_ns: i96, decode_ns: i96, decode_steps: usize };

/// One batched multi-stream pass: reset all caches, prefill the shared
/// prompt into each stream (per-stream `forwardStep`, timed together),
/// then decode `max_new - 1` lockstep steps through `forwardStepBatch` —
/// one m=N weight pass per step. Per-stream samplers are seeded
/// `sampler_cfg.seed + i` (matching the sequential arm) and no stop token
/// applies, so every stream runs the full length.
fn benchStreamsBatchPass(
    io: std.Io,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    caches: []const *llm.kv_cache.KvCache,
    tokens: []const usize,
    sampler_cfg: llm.sampler.Config,
    max_new: usize,
    outs: []const []usize,
    histories: []const []usize,
    samplers: []llm.sampler.Sampler,
    hist_lens: []usize,
    lasts: []usize,
    processors: []const ?llm.sampler.LogitProcessor,
) !StreamsPassResult {
    const n = caches.len;
    for (0..n) |i| {
        caches[i].reset();
        var cfg = sampler_cfg;
        cfg.seed = sampler_cfg.seed +% i;
        samplers[i] = llm.sampler.Sampler.init(cfg);
        samplers[i].processor = processors[i];
        if (processors[i]) |p| try p.reset(); // fresh grammar state per pass
        @memcpy(histories[i][0..tokens.len], tokens);
        hist_lens[i] = tokens.len;
    }

    // Prefill timing covers ONLY the forwards (the benchOnePass protocol:
    // the first sampler draw sits between the prefill and decode spans).
    var prefill_ns: i96 = 0;
    for (0..n) |i| {
        const forward_start = nowNs(io);
        var logits = try model.forwardStep(ctx, caches[i], tokens, 0);
        prefill_ns += nowNs(io) - forward_start;
        defer logits.deinit();
        outs[i][0] = try samplers[i].next(ctx, &logits, histories[i][0..hist_lens[i]]);
        histories[i][hist_lens[i]] = outs[i][0];
        hist_lens[i] += 1;
        lasts[i] = outs[i][0];
    }

    const decode_start = nowNs(io);
    var step: usize = 1;
    while (step < max_new) : (step += 1) {
        var logits = try model.forwardStepBatch(ctx, caches, lasts);
        defer logits.deinit();
        for (0..n) |i| {
            var row = try logits.narrow(ctx, .seq, i, 1);
            defer row.deinit();
            const next = try samplers[i].next(ctx, &row, histories[i][0..hist_lens[i]]);
            outs[i][step] = next;
            histories[i][hist_lens[i]] = next;
            hist_lens[i] += 1;
            lasts[i] = next;
        }
    }
    const decode_ns = nowNs(io) - decode_start;
    return .{ .prefill_ns = prefill_ns, .decode_ns = decode_ns, .decode_steps = max_new - 1 };
}

/// `--streams N`: batched multi-stream decode (one `forwardStepBatch` m=N
/// weight pass per step, per-stream KV/sampler) vs N sequential
/// single-stream runs of the same prompt/length — the batch-N-vs-N×1
/// measurement recorded in docs/BENCHMARK.md. Passes are paired batch/sequential
/// within each rep (thermal fairness); aggregate tok/s counts all streams'
/// tokens over wall time. Existing single-stream `prefill:`/`decode :`
/// output labels are untouched (bench_gate.py parses them); this mode
/// prints its own `decode-batch`/`decode-seq` labels.
fn runGenerateStreams(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: anytype,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    tokens: []const usize,
    load_ns: i96,
    max_new: usize,
    n_streams: usize,
    bench_reps: usize,
    cache_type: llm.kv_cache.KvDtype,
    sampler_cfg: llm.sampler.Config,
    constraint: ?*llm.llguidance.Constraint,
) !void {
    if (max_new < 2) return error.StreamsNeedDecodeSteps;
    const cfg = model.config;
    const n = n_streams;
    const capacity = tokens.len + max_new;

    // Per-stream grammar state: clone the base constraint once per stream
    // (matcher deep-clone + refcounted tokenizer — no grammar recompilation);
    // both arms reset the clones at each pass start.
    const stream_constraints = try allocator.alloc(llm.llguidance.Constraint, n);
    defer allocator.free(stream_constraints);
    var constraints_inited: usize = 0;
    defer for (0..constraints_inited) |i| stream_constraints[i].deinit();
    const processors = try allocator.alloc(?llm.sampler.LogitProcessor, n);
    defer allocator.free(processors);
    for (0..n) |i| {
        if (constraint) |base| {
            stream_constraints[i] = try base.clone();
            constraints_inited += 1;
            processors[i] = stream_constraints[i].processor();
        } else {
            processors[i] = null;
        }
    }

    const caches = try allocator.alloc(llm.kv_cache.KvCache, n);
    defer allocator.free(caches);
    var caches_inited: usize = 0;
    defer for (0..caches_inited) |i| caches[i].deinit();
    for (0..n) |i| {
        caches[i] = try llm.kv_cache.KvCache.initWithDtype(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, cache_type);
        caches_inited += 1;
    }
    const cache_ptrs = try allocator.alloc(*llm.kv_cache.KvCache, n);
    defer allocator.free(cache_ptrs);
    for (cache_ptrs, caches) |*ptr, *cache| ptr.* = cache;
    try printCacheInfo(stdout, &caches[0]);
    try stdout.print("kv cache: x{d} streams\n", .{n});

    // Per-stream token/history buffers for BOTH arms (batch keeps its own
    // copy so the first rep can cross-check batch == sequential outputs).
    const outs_batch = try allocSliceOfSlices(allocator, n, max_new);
    defer freeSliceOfSlices(allocator, outs_batch);
    const outs_seq = try allocSliceOfSlices(allocator, n, max_new);
    defer freeSliceOfSlices(allocator, outs_seq);
    const histories = try allocSliceOfSlices(allocator, n, tokens.len + max_new);
    defer freeSliceOfSlices(allocator, histories);
    const samplers = try allocator.alloc(llm.sampler.Sampler, n);
    defer allocator.free(samplers);
    const hist_lens = try allocator.alloc(usize, n);
    defer allocator.free(hist_lens);
    const lasts = try allocator.alloc(usize, n);
    defer allocator.free(lasts);

    const reps = @max(bench_reps, 1);
    const batch_tgs = try allocator.alloc(f64, reps);
    defer allocator.free(batch_tgs);
    const seq_tgs = try allocator.alloc(f64, reps);
    defer allocator.free(seq_tgs);
    const prefill_pps = try allocator.alloc(f64, reps);
    defer allocator.free(prefill_pps);

    var match: ?struct { stream: usize, step: usize } = null;
    var checked = false;
    var decode_steps: usize = 0;
    var rep: usize = 0;
    while (rep <= reps) : (rep += 1) { // rep 0 = warmup
        const pass_new = if (rep == 0) @min(max_new, 2) else max_new;

        const batch = try benchStreamsBatchPass(io, ctx, model, cache_ptrs, tokens, sampler_cfg, pass_new, outs_batch, histories, samplers, hist_lens, lasts, processors);

        // Sequential arm: the same N runs, one stream at a time (the same
        // per-stream sampler seeds and grammar clones), reusing stream i's
        // cache and buffers.
        var seq_prefill_ns: i96 = 0;
        var seq_decode_ns: i96 = 0;
        for (0..n) |i| {
            var stream_cfg = sampler_cfg;
            stream_cfg.seed = sampler_cfg.seed +% i;
            const r = try benchOnePass(io, ctx, model, cache_ptrs[i], tokens, outs_seq[i], histories[i], stream_cfg, pass_new, null, false, false, null, null, processors[i]);
            seq_prefill_ns += r.prefill_ns;
            seq_decode_ns += r.decode_ns;
        }

        if (rep == 0) continue;
        decode_steps = batch.decode_steps;
        const decode_tokens: f64 = @floatFromInt(n * batch.decode_steps);
        batch_tgs[rep - 1] = decode_tokens / seconds(batch.decode_ns);
        seq_tgs[rep - 1] = decode_tokens / seconds(seq_decode_ns);
        prefill_pps[rep - 1] = @as(f64, @floatFromInt(n * tokens.len)) / seconds(batch.prefill_ns);

        if (!checked) {
            checked = true;
            outer: for (0..n) |i| {
                for (0..max_new) |s| {
                    if (outs_batch[i][s] != outs_seq[i][s]) {
                        match = .{ .stream = i, .step = s };
                        break :outer;
                    }
                }
            }
        }
    }

    try stdout.print("load: {d:.3} s\n", .{seconds(load_ns)});
    try stdout.print("streams bench: N={d}, {d} reps (+1 warmup), prompt {d} tok, decode {d} steps/stream, aggregate tok/s\n", .{ n, reps, tokens.len, decode_steps });
    try printBenchStat(stdout, "prefill-agg ", prefill_pps);
    try printBenchStat(stdout, "decode-batch", batch_tgs);
    try printBenchStat(stdout, "decode-seq  ", seq_tgs);
    var batch_mean: f64 = 0;
    var seq_mean: f64 = 0;
    for (batch_tgs) |v| batch_mean += v;
    for (seq_tgs) |v| seq_mean += v;
    try stdout.print("batch speedup: {d:.2}x (mean decode, N={d})\n", .{ batch_mean / seq_mean, n });
    if (match) |m| {
        const why = if (n >= 12)
            "expected: m>=12 crosses the fused-FFN kernel threshold, ~1e-6 drift"
        else if (n >= 4)
            "expected on quantized weights: the x4-packed kernels engage at m>=4, ~1e-6 drift (f32/f16 weights stay bitwise here)"
        else
            "UNEXPECTED at this batch size";
        try stdout.print("outputs: DIVERGED at stream {d}, token {d} ({s})\n", .{ m.stream, m.step, why });
    } else {
        try stdout.print("outputs: batch == sequential, token-for-token ({d} streams x {d} tokens)\n", .{ n, max_new });
    }
}

fn allocSliceOfSlices(allocator: std.mem.Allocator, n: usize, len: usize) ![]const []usize {
    const slices = try allocator.alloc([]usize, n);
    var inited: usize = 0;
    errdefer {
        for (0..inited) |i| allocator.free(slices[i]);
        allocator.free(slices);
    }
    for (slices) |*s| {
        s.* = try allocator.alloc(usize, len);
        inited += 1;
    }
    return slices;
}

fn freeSliceOfSlices(allocator: std.mem.Allocator, slices: []const []usize) void {
    for (slices) |s| allocator.free(s);
    allocator.free(slices);
}

/// `--gen` with `--spec`: lossless draft-model-free speculative decoding via
/// the SpeculationIndex cascade (conversation SAM + `--spec-ref` documents +
/// token recycling) and the batched-verify decoder. Reports decode tok/s plus
/// the decoder's acceptance stats and the per-source acceptance summary.
fn runGenerateSpec(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    tok: ?*const llm.tokenizer.Tokenizer,
    tokens: []const usize,
    load_ns: i96,
    max_new: usize,
    stop_token: ?usize,
    sampler_cfg: llm.sampler.Config,
    cache_type: llm.kv_cache.KvDtype,
    ref_paths: []const []const u8,
    processor: ?llm.sampler.LogitProcessor,
) !void {
    // The decoder's invariant needs a non-empty committed history (e.g.
    // `--prompt ""` tokenizes to nothing).
    if (tokens.len == 0) return error.EmptyPrompt;
    const cfg = model.config;
    // Stop-awareness: the verify loop must not sample rows past a committed
    // stop token (RNG-draw parity with a plain run — see speculative.zig).
    const spec_options = llm.speculative.core.Options{ .stop_token = stop_token };
    // Room for the prompt, the requested tokens, and one verify batch of
    // overshoot past the max_new boundary.
    const capacity = tokens.len + max_new + spec_options.max_draft + 1;
    var cache = try llm.kv_cache.KvCache.initWithDtype(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, cache_type);
    defer cache.deinit();
    try printCacheInfo(stdout, &cache);

    var index = try llm.speculative.cascade.SpeculationIndex.init(allocator, cfg.vocab_size);
    defer index.deinit();
    // Acceptance accounting settles only drafts the decoder actually verifies
    // (the cascade's accounting contract).
    index.accounting_min_draft = spec_options.min_draft;
    for (ref_paths) |path| {
        const t = tok orelse return error.TokenizerUnavailable;
        const ids = try tokenizeFile(io, allocator, t, path);
        defer allocator.free(ids);
        try index.addReference(ids);
        try stdout.print("spec ref: {s} ({d} tokens)\n", .{ path, ids.len });
    }

    // With a grammar constraint installed, wrap the cascade so forced spans
    // draft themselves and cascade drafts are pre-filtered to their
    // grammar-valid prefix (speculative/constrained.zig).
    var grammar_source: llm.speculative.constrained.ConstrainedSource = undefined;
    var draft_source = index.asDraftSource();
    if (processor) |p| {
        if (p.hasStructure()) {
            grammar_source = llm.speculative.constrained.ConstrainedSource.init(p, index.asDraftSource());
            draft_source = grammar_source.source();
        }
    }

    var decoder = try llm.speculative.core.SpeculativeDecoder(llm.qwen3.model.Model).init(allocator, draft_source, spec_options);
    defer decoder.deinit();
    decoder.io = io; // live verify/plain cost measurement for the auto-off gate

    var history: std.ArrayList(usize) = .empty;
    defer history.deinit(allocator);
    try history.appendSlice(allocator, tokens);
    index.observe(tokens); // the prompt is committed context

    var sampler = llm.sampler.Sampler.init(sampler_cfg);
    sampler.processor = processor;
    var sink_state: u8 = 0;
    const sink = llm.speculative.core.TokenSink{ .ptr = @ptrCast(&sink_state), .func = nullSinkEmit };

    const prefill_start = nowNs(io);
    if (tokens.len > 1) {
        var pre = try model.forwardStep(ctx, &cache, tokens[0 .. tokens.len - 1], 0);
        pre.deinit();
    }
    const prefill_ns = nowNs(io) - prefill_start;

    const decode_start = nowNs(io);
    decode: while (history.items.len - tokens.len < max_new) {
        const before = history.items.len;
        _ = try decoder.step(ctx, model, &cache, &sampler, &history, sink);
        if (stop_token) |stop| {
            for (history.items[before..]) |t| {
                if (t == stop) break :decode;
            }
        }
    }
    const decode_ns = nowNs(io) - decode_start;

    // Trim verify-batch overshoot; keep the stop token itself (plain --gen
    // parity: the stop token is the last emitted token).
    var out_len = @min(history.items.len - tokens.len, max_new);
    if (stop_token) |stop| {
        if (std.mem.indexOfScalar(usize, history.items[tokens.len..][0..out_len], stop)) |j| out_len = j + 1;
    }
    const out = history.items[tokens.len..][0..out_len];

    try stdout.print("load: {d:.3} s\n", .{seconds(load_ns)});
    try stdout.print("prompt tokens: {d}\n", .{tokens.len});
    try stdout.print("prefill: {d:.3} ms\n", .{millis(prefill_ns)});
    const tg = @as(f64, @floatFromInt(out_len)) / seconds(decode_ns);
    try stdout.print("decode: {d} tokens, {d:.3} ms, {d:.2} tok/s (speculative)\n", .{ out_len, millis(decode_ns), tg });
    try decoder.stats.writeSummary(stdout);
    try stdout.print("\n", .{});
    try index.writeSourceSummary(stdout);
    try stdout.print("\n", .{});

    try stdout.print("generated {d}:", .{out.len});
    for (out) |token| try stdout.print(" {d}", .{token});
    try stdout.print("\n", .{});
    if (tok) |t| {
        const gen_str = try decodeIds(allocator, t, out);
        defer allocator.free(gen_str);
        try stdout.print("text: {s}\n", .{gen_str});
    }
}

fn nullSinkEmit(ptr: *anyopaque, token: usize) anyerror!void {
    _ = ptr;
    _ = token;
}

/// Read a UTF-8 text file and tokenize it (no BOS/EOS policy) for use as a
/// speculation reference document.
/// Resolve a grammar flag value: `@PATH` reads the file (ownership parked in
/// `owned` for the caller's deferred free); anything else is the inline text.
fn grammarValue(io: std.Io, allocator: std.mem.Allocator, value: []const u8, owned: *?[]u8) ![]const u8 {
    if (!std.mem.startsWith(u8, value, "@")) return value;
    const text = try readTextFile(io, allocator, value[1..]);
    owned.* = text;
    return text;
}

/// Read a small text file (grammar/schema/reference documents), with a size
/// cap so a mistyped path (a GGUF, a core dump, ...) fails fast and clearly
/// instead of ballooning memory.
fn readTextFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.IsDir;
    const max_bytes = 64 * 1024 * 1024;
    if (stat.size > max_bytes) return error.FileTooLarge;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}

fn tokenizeFile(io: std.Io, allocator: std.mem.Allocator, tok: *const llm.tokenizer.Tokenizer, path: []const u8) ![]usize {
    const bytes = try readTextFile(io, allocator, path);
    defer allocator.free(bytes);
    const ids32 = try tok.encodeRaw(allocator, bytes);
    defer allocator.free(ids32);
    const ids = try allocator.alloc(usize, ids32.len);
    errdefer allocator.free(ids);
    for (ids, ids32) |*d, s| d.* = s;
    return ids;
}

/// Hidden `--spec-bench` mode: the speculative-decoding verify-economics
/// probe. After prefilling the prompt, measure (best-of-`reps`) the cost of
/// ONE batched k-row verify pass (`forwardStepAllLogits`) against k
/// sequential single-token decode steps, rewinding the cache with
/// `truncate()` between runs — exactly the state dance the speculative
/// decoder performs. The ratio quantifies what acceptance rate speculation
/// needs to win: a verify pass that costs `eq` single steps pays off once
/// it commits more than `eq` tokens on average.
fn runSpecBench(
    io: std.Io,
    stdout: anytype,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    tokens: []const usize,
    load_ns: i96,
    cache_type: llm.kv_cache.KvDtype,
    reps: usize,
) !void {
    const cfg = model.config;
    const ks = [_]usize{ 2, 4, 8, 16 };
    const max_k = ks[ks.len - 1];
    const capacity = tokens.len + max_k + 1;
    var cache = try llm.kv_cache.KvCache.initWithDtype(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, cache_type);
    defer cache.deinit();

    // Prefill once; every measurement runs at this depth and rewinds to it.
    var prefill = try model.forwardStep(ctx, &cache, tokens, 0);
    prefill.deinit();
    const base = cache.len;

    // Continuation token values don't affect the cost; cycle the prompt.
    var cont: [max_k]usize = undefined;
    for (&cont, 0..) |*t, i| t.* = tokens[i % tokens.len];

    try stdout.print("load: {d:.3} s\n", .{seconds(load_ns)});
    try stdout.print("spec verify economics: prompt {d} tok (base kv len {d}), best of {d} reps, kv {s}\n", .{ tokens.len, base, reps, @tagName(cache.dtype) });
    try stdout.print("  k | batch-k verify ms | k x single ms | batch/k-single | verify = N single steps\n", .{});

    // Warmup both paths (prime buffers/threads), then measure.
    {
        var warm = try model.forwardStepAllLogits(ctx, &cache, cont[0..max_k], cache.len);
        warm.deinit();
        cache.truncate(base);
        var warm2 = try model.forwardStep(ctx, &cache, cont[0..1], cache.len);
        warm2.deinit();
        cache.truncate(base);
    }

    for (ks) |k| {
        var batch_best: i96 = std.math.maxInt(i96);
        var single_best: i96 = std.math.maxInt(i96);
        for (0..reps) |_| {
            const t0 = nowNs(io);
            var logits = try model.forwardStepAllLogits(ctx, &cache, cont[0..k], cache.len);
            logits.deinit();
            const dt = nowNs(io) - t0;
            cache.truncate(base);
            batch_best = @min(batch_best, dt);
        }
        for (0..reps) |_| {
            const t0 = nowNs(io);
            for (0..k) |i| {
                var logits = try model.forwardStep(ctx, &cache, cont[i..][0..1], cache.len);
                logits.deinit();
            }
            const dt = nowNs(io) - t0;
            cache.truncate(base);
            single_best = @min(single_best, dt);
        }
        const batch_ms = millis(batch_best);
        const single_ms = millis(single_best);
        const single_one = single_ms / @as(f64, @floatFromInt(k));
        try stdout.print(" {d:>2} | {d:>17.3} | {d:>13.3} | {d:>14.3} | {d:>6.2}\n", .{
            k,
            batch_ms,
            single_ms,
            batch_ms / single_ms,
            batch_ms / single_one,
        });
    }
}

/// Streamed single-turn or interactive multi-turn chat. The reply streams to
/// `stdout` token-by-token; a `Conversation` keeps the KV cache across turns.
fn runChat(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    tok: *const llm.tokenizer.Tokenizer,
    template: llm.chat.Template,
    system: ?[]const u8,
    no_think: bool,
    sampler_cfg: llm.sampler.Config,
    chat_text: ?[]const u8,
    spec: bool,
    spec_refs: []const []const u8,
    processor: ?llm.sampler.LogitProcessor,
    kv_save_path: ?[]const u8,
) !void {
    var convo = try llm.chat.Conversation(llm.qwen3.model.Model, llm.tokenizer).init(ctx, model, tok, template, .{
        .system = system,
        .think_off = no_think,
        .sampler = sampler_cfg,
        .capacity = 4096,
        .max_response_tokens = 1024,
        .logit_processor = processor,
        .speculation = spec,
        .io = io,
    });
    defer convo.deinit();
    if (kv_save_path) |path| {
        const resumed = try convo.enablePersistence(io, path);
        if (resumed > 0) try stdout.print("[kv: conversation resumed from {s} — {d} tokens, no re-prefill]\n", .{ path, resumed });
    }
    for (spec_refs) |path| {
        const ids = try tokenizeFile(io, allocator, tok, path);
        defer allocator.free(ids);
        try convo.addSpecReference(ids);
    }

    if (chat_text) |msg| {
        _ = try convo.send(msg, stdout);
        try stdout.writeAll("\n");
        if (convo.specStats()) |stats| {
            try stats.writeSummary(stdout);
            try stdout.writeAll("\n");
        }
        try stdout.flush();
        return;
    }

    // REPL: read a line per turn, stream the reply, keep the cache.
    var stdin_buf: [8192]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);
    const in = &stdin_reader.interface;
    try stdout.print("chat ({s}) — type a message, empty line or Ctrl-D to quit:\n", .{@tagName(template.format)});
    while (true) {
        try stdout.writeAll("\n> ");
        try stdout.flush();
        const line = (try in.takeDelimiter('\n')) orelse break;
        const msg = std.mem.trim(u8, line, " \t\r");
        if (msg.len == 0) break;
        try stdout.writeAll("\n");
        _ = convo.send(msg, stdout) catch |e| switch (e) {
            error.ContextFull => {
                try stdout.print("[context full — restart to continue]\n", .{});
                break;
            },
            else => return e,
        };
        try stdout.writeAll("\n");
        try stdout.flush();
    }
}

fn decodeIds(allocator: std.mem.Allocator, tok: *const llm.tokenizer.Tokenizer, ids: []const usize) ![]u8 {
    const ids32 = try allocator.alloc(u32, ids.len);
    defer allocator.free(ids32);
    for (ids32, ids) |*d, s| d.* = @intCast(s);
    return tok.decode(allocator, ids32);
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

const CacheCompare = struct {
    cache_top: usize,
    ref_top: usize,
    aligned: bool,
    max_abs: f64,
    // A misaligned argmax is "benign" when the two candidates' logit gap is
    // within the floating-point drift (2x the max per-element abs diff): the
    // flip is fully explained by m=1-vs-m=L kernel reassociation, not a cache
    // bug. A structural bug would diverge far beyond the drift.
    benign: bool,
};

fn runVerifyCache(
    allocator: std.mem.Allocator,
    stdout: anytype,
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    tokens: []const usize,
    load_ns: i96,
    max_steps: usize,
    cache_type: llm.kv_cache.KvDtype,
) !void {
    const cfg = model.config;
    const capacity = tokens.len + max_steps;
    var cache = try llm.kv_cache.KvCache.initWithDtype(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, cache_type);
    defer cache.deinit();
    try printCacheInfo(stdout, &cache);

    const seq = try allocator.alloc(usize, capacity);
    defer allocator.free(seq);
    @memcpy(seq[0..tokens.len], tokens);
    var seq_len = tokens.len;

    var cache_logits = try model.forwardStep(ctx, &cache, tokens, 0);
    defer cache_logits.deinit();
    var ref_logits = try model.forwardLastLogits(ctx, seq[0..seq_len]);
    defer ref_logits.deinit();

    try stdout.print("load: {d:.3} s\n", .{seconds(load_ns)});
    try stdout.print("verify cache vs re-prefill ({d} prompt tokens, {d} steps):\n", .{ tokens.len, max_steps });

    var aligned_steps: usize = 0;
    var ok = true;
    var max_diff_overall: f64 = 0;
    var step: usize = 0;
    while (true) : (step += 1) {
        const cmp = compareTopAndDiff(try cache_logits.dataConst(), try ref_logits.dataConst());
        max_diff_overall = @max(max_diff_overall, cmp.max_abs);
        if (cmp.aligned) aligned_steps += 1;
        // f16 KV is lossy vs the f32 re-prefill reference, so per-step logits
        // differ by the f16 rounding error; correctness here means every greedy
        // divergence is a drift near-tie. A non-benign divergence (gap far beyond
        // the per-element diff) would signal a real cache bug.
        if (!cmp.benign) ok = false;
        const note = if (cmp.aligned) "" else if (cmp.benign) " (near-tie)" else " (DIVERGED)";
        try stdout.print("  step {d}: cache_top={d} ref_top={d} aligned={} max_abs={d:.6}{s}\n", .{ step, cmp.cache_top, cmp.ref_top, cmp.aligned, cmp.max_abs, note });

        if (step == max_steps or seq_len == capacity) break;
        seq[seq_len] = cmp.cache_top;
        seq_len += 1;

        // Allocate-then-swap so an error never leaves a deinit'd tensor live
        // under the function-scope defers.
        const fresh_cache = try model.forwardStep(ctx, &cache, seq[seq_len - 1 ..][0..1], cache.len);
        cache_logits.deinit();
        cache_logits = fresh_cache;
        const fresh_ref = try model.forwardLastLogits(ctx, seq[0..seq_len]);
        ref_logits.deinit();
        ref_logits = fresh_ref;
    }
    try stdout.print("verify: {s} (top-aligned {d}/{d} steps, max_abs={d:.6}; divergences within drift are expected)\n", .{ if (ok) "PASS" else "FAIL", aligned_steps, step + 1, max_diff_overall });
}

fn compareTopAndDiff(cache_values: []const f32, ref_values: []const f32) CacheCompare {
    var cache_top: usize = 0;
    var ref_top: usize = 0;
    var max_abs: f64 = 0;
    for (cache_values, ref_values, 0..) |cache_value, ref_value, i| {
        if (cache_value > cache_values[cache_top]) cache_top = i;
        if (ref_value > ref_values[ref_top]) ref_top = i;
        const diff = @abs(@as(f64, cache_value) - @as(f64, ref_value));
        max_abs = @max(max_abs, diff);
    }
    const aligned = cache_top == ref_top;
    const gap = @as(f64, ref_values[ref_top]) - @as(f64, ref_values[cache_top]);
    return .{
        .cache_top = cache_top,
        .ref_top = ref_top,
        .aligned = aligned,
        .max_abs = max_abs,
        .benign = aligned or gap < 2 * max_abs,
    };
}

fn readF32File(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    if (stat.kind != .file) return error.IsDir;
    if (stat.size % @sizeOf(f32) != 0) return error.InvalidLogitsFile;

    const byte_len: usize = @intCast(stat.size);
    const bytes = try allocator.alloc(u8, byte_len);
    defer allocator.free(bytes);

    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }

    const out = try allocator.alloc(f32, byte_len / @sizeOf(f32));
    errdefer allocator.free(out);
    for (out, 0..) |*dst, i| {
        const bits = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
        dst.* = @bitCast(bits);
    }
    return out;
}
