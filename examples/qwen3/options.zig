//! CLI surface of the Qwen3 runner: the usage text, the flag grammar, and
//! the parsed `Options` struct main.zig dispatches on. Every flag supports
//! both `--flag VALUE` and `--flag=VALUE`.
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

pub const default_tokens = [_]usize{ 151_644, 872, 198, 9707 };

pub const Options = struct {
    tokens: []const usize = default_tokens[0..],
    logits_out: ?[]const u8 = null,
    compare_logits: ?[]const u8 = null,
    tokens_out: ?[]const u8 = null,
    profile_enabled: bool = false,
    repeat: usize = 1,
    gen_count: ?usize = null,
    bench_reps: usize = 1,
    verify_count: ?usize = null,
    verify_batch_count: ?usize = null,
    cache_type: llm.kv_cache.KvDtype = .f16,
    stop_token: ?usize = null,
    prompt_text: ?[]const u8 = null,
    info_flag: bool = false,
    chat_text: ?[]const u8 = null,
    system_text: ?[]const u8 = null,
    repl_flag: bool = false,
    no_think: bool = false,
    spec_bench: bool = false,
    spec_flag: bool = false,
    tokenize_file: ?[]const u8 = null,
    spec_ref_buf: [8][]const u8 = undefined,
    spec_ref_count: usize = 0,
    temp_arg: ?f32 = null,
    streams_arg: usize = 1,
    json_schema_arg: ?[]const u8 = null,
    lark_arg: ?[]const u8 = null,
    regex_arg: ?[]const u8 = null,
    topk_arg: ?usize = null,
    topp_arg: ?f32 = null,
    minp_arg: ?f32 = null,
    penalty_arg: ?f32 = null,
    seed_arg: ?u64 = null,
    moe_cli: fucina.weights.MoeStreamCli = .{},
    moe_cache_slots: ?usize = null,
    moe_pin_mb: ?usize = null,
    moe_no_learn: bool = false,
    moe_pilot: bool = false,
    moe_expert_top_p: ?f32 = null,
    kv_save: bool = false,
    kv_save_arg: ?[]const u8 = null,
    shine_gguf: ?[]const u8 = null,
    shine_context: ?[]const u8 = null,
    shine_adapter: ?[]const u8 = null,
    shine_save: ?[]const u8 = null,
    shine_save_cartridge: ?[]const u8 = null,
    shine_fleet_build: ?[]const u8 = null,
    shine_docs: ?[]const u8 = null,

    pub fn specRefs(self: *const Options) []const []const u8 {
        return self.spec_ref_buf[0..self.spec_ref_count];
    }
};

/// Parse `args` (past the model path at args[1]) into `Options`. Prints the
/// usage text and returns null when no model path was given. `token_buf`
/// backs the parsed token-id list and must outlive the returned Options.
pub fn parse(args: []const []const u8, stdout: *std.Io.Writer, token_buf: []usize) !?Options {
    if (args.len < 2) {
        try stdout.print("usage: zig build qwen3 -- <model.gguf> [comma-separated-token-ids] [--repeat N] [--profile] [--logits-out PATH] [--compare-logits PATH] [--gen N [--stop TOKEN] [--tokens-out PATH]] [--bench R] [--verify-cache N] [--verify-batch=N] [--cache-type f16|q8_0] [--spec] [--spec-ref FILE] [--tokenize FILE]...\n", .{});
        try stdout.print("spec:     zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf --prompt \"...\" --gen 128 --spec [--spec-ref doc.txt]   (lossless speculative decode + acceptance stats)\n", .{});
        try stdout.print("bench:    zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf <prompt-token-ids> --gen 64 --bench 5   (warm pp/tg, load once; fair vs llama-bench)\n", .{});
        try stdout.print("streams:  zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf <prompt-token-ids> --gen 64 --bench 3 --streams 4   (batched multi-stream decode vs N sequential runs)\n", .{});
        try stdout.print("example: zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf 151644,872,198,9707\n", .{});
        try stdout.print("generate: zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --prompt \"The capital of France is\" --gen 64\n", .{});
        try stdout.print("chat:     zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --chat \"What is the capital of France?\" [--no-think] [--system \"...\"]\n", .{});
        try stdout.print("repl:     zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --repl   (multi-turn; streams replies)\n", .{});
        try stdout.print("shine:    zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-8B-F16.gguf --shine models/shine/shine-ift-mqa-1qa.gguf --shine-context @doc.txt [--shine-save adapter.gguf] --chat \"...\"|--repl   (context -> LoRA adapter in one hypernetwork pass, arXiv 2602.06358; greedy, no-think)\n", .{});
        try stdout.print("          ... --shine-adapter adapter.gguf --chat \"...\"|--repl   (serve a saved adapter; no SHINE weights, no hypernetwork pass)\n", .{});
        try stdout.print("          ... --shine models/shine/shine-ift-mqa-1qa.gguf --shine-docs DIR --shine-fleet-build OUT   (adapter + retrieval index per .txt/.md doc; serve with `zig build lmserve -- <base.gguf> --shine-fleet OUT`)\n", .{});
        try stdout.print("sampling: --temp F --top-k N --top-p F --min-p F --repeat-penalty F --seed N\n", .{});
        try stdout.print("grammar:  --json-schema JSON|@FILE | --lark GRAMMAR|@FILE | --regex PATTERN   (constrained decoding; needs a -Dllguidance=true build; combine with --no-think)\n", .{});
        try stdout.print("other:    --info | --spec-bench\n", .{});
        return null;
    }

    var o = Options{};
    var arg_i: usize = 2;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--logits-out")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingLogitsPath;
            o.logits_out = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--logits-out=")) {
            o.logits_out = arg["--logits-out=".len..];
        } else if (std.mem.eql(u8, arg, "--compare-logits")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingCompareLogitsPath;
            o.compare_logits = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--compare-logits=")) {
            o.compare_logits = arg["--compare-logits=".len..];
        } else if (std.mem.eql(u8, arg, "--tokens-out")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTokensOutPath;
            o.tokens_out = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--tokens-out=")) {
            o.tokens_out = arg["--tokens-out=".len..];
        } else if (std.mem.eql(u8, arg, "--repeat")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingRepeatCount;
            o.repeat = try parseRepeat(args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--repeat=")) {
            o.repeat = try parseRepeat(arg["--repeat=".len..]);
        } else if (std.mem.eql(u8, arg, "--bench")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingBenchCount;
            o.bench_reps = try parseRepeat(args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--bench=")) {
            o.bench_reps = try parseRepeat(arg["--bench=".len..]);
        } else if (std.mem.eql(u8, arg, "--profile")) {
            o.profile_enabled = true;
        } else if (std.mem.eql(u8, arg, "--gen")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingGenCount;
            o.gen_count = try parseRepeat(args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--gen=")) {
            o.gen_count = try parseRepeat(arg["--gen=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--verify-batch=")) {
            o.verify_batch_count = try std.fmt.parseInt(usize, arg["--verify-batch=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--verify-cache")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingVerifyCount;
            o.verify_count = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--verify-cache=")) {
            o.verify_count = try std.fmt.parseInt(usize, arg["--verify-cache=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--cache-type")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingCacheType;
            o.cache_type = try parseCacheType(args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--cache-type=")) {
            o.cache_type = try parseCacheType(arg["--cache-type=".len..]);
        } else if (std.mem.eql(u8, arg, "--stop")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingStopToken;
            o.stop_token = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--stop=")) {
            o.stop_token = try std.fmt.parseInt(usize, arg["--stop=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingPrompt;
            o.prompt_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--prompt=")) {
            o.prompt_text = arg["--prompt=".len..];
        } else if (std.mem.eql(u8, arg, "--info")) {
            o.info_flag = true;
        } else if (std.mem.eql(u8, arg, "--chat")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingChatMessage;
            o.chat_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--chat=")) {
            o.chat_text = arg["--chat=".len..];
        } else if (std.mem.eql(u8, arg, "--shine")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingShinePath;
            o.shine_gguf = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--shine=")) {
            o.shine_gguf = arg["--shine=".len..];
        } else if (std.mem.eql(u8, arg, "--shine-context")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingShineContext;
            o.shine_context = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--shine-context=")) {
            o.shine_context = arg["--shine-context=".len..];
        } else if (std.mem.eql(u8, arg, "--shine-adapter")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingShineAdapterPath;
            o.shine_adapter = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--shine-adapter=")) {
            o.shine_adapter = arg["--shine-adapter=".len..];
        } else if (std.mem.eql(u8, arg, "--shine-save-cartridge")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingArgument;
            o.shine_save_cartridge = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--shine-save-cartridge=")) {
            o.shine_save_cartridge = arg["--shine-save-cartridge=".len..];
        } else if (std.mem.eql(u8, arg, "--shine-save")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingShineSavePath;
            o.shine_save = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--shine-save=")) {
            o.shine_save = arg["--shine-save=".len..];
        } else if (std.mem.eql(u8, arg, "--shine-fleet-build")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingShineFleetDir;
            o.shine_fleet_build = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--shine-fleet-build=")) {
            o.shine_fleet_build = arg["--shine-fleet-build=".len..];
        } else if (std.mem.eql(u8, arg, "--shine-docs")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingShineDocsDir;
            o.shine_docs = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--shine-docs=")) {
            o.shine_docs = arg["--shine-docs=".len..];
        } else if (std.mem.eql(u8, arg, "--system")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSystemMessage;
            o.system_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--system=")) {
            o.system_text = arg["--system=".len..];
        } else if (std.mem.eql(u8, arg, "--spec-bench")) {
            o.spec_bench = true;
        } else if (std.mem.eql(u8, arg, "--spec")) {
            o.spec_flag = true;
        } else if (std.mem.eql(u8, arg, "--spec-ref")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSpecRefPath;
            if (o.spec_ref_count == o.spec_ref_buf.len) return error.TooManySpecRefs;
            o.spec_ref_buf[o.spec_ref_count] = args[arg_i];
            o.spec_ref_count += 1;
        } else if (std.mem.startsWith(u8, arg, "--spec-ref=")) {
            if (o.spec_ref_count == o.spec_ref_buf.len) return error.TooManySpecRefs;
            o.spec_ref_buf[o.spec_ref_count] = arg["--spec-ref=".len..];
            o.spec_ref_count += 1;
        } else if (std.mem.eql(u8, arg, "--tokenize")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTokenizePath;
            o.tokenize_file = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--tokenize=")) {
            o.tokenize_file = arg["--tokenize=".len..];
        } else if (std.mem.eql(u8, arg, "--repl")) {
            o.repl_flag = true;
        } else if (std.mem.eql(u8, arg, "--no-think")) {
            o.no_think = true;
        } else if (std.mem.eql(u8, arg, "--temp")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTemp;
            o.temp_arg = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--temp=")) {
            o.temp_arg = try std.fmt.parseFloat(f32, arg["--temp=".len..]);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTopK;
            o.topk_arg = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--top-k=")) {
            o.topk_arg = try std.fmt.parseInt(usize, arg["--top-k=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTopP;
            o.topp_arg = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--top-p=")) {
            o.topp_arg = try std.fmt.parseFloat(f32, arg["--top-p=".len..]);
        } else if (std.mem.eql(u8, arg, "--min-p")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingMinP;
            o.minp_arg = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--min-p=")) {
            o.minp_arg = try std.fmt.parseFloat(f32, arg["--min-p=".len..]);
        } else if (std.mem.eql(u8, arg, "--repeat-penalty")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingRepeatPenalty;
            o.penalty_arg = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--repeat-penalty=")) {
            o.penalty_arg = try std.fmt.parseFloat(f32, arg["--repeat-penalty=".len..]);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSeed;
            o.seed_arg = try std.fmt.parseInt(u64, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--seed=")) {
            o.seed_arg = try std.fmt.parseInt(u64, arg["--seed=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--streams")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingStreamCount;
            o.streams_arg = try parseRepeat(args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--streams=")) {
            o.streams_arg = try parseRepeat(arg["--streams=".len..]);
        } else if (std.mem.eql(u8, arg, "--json-schema")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingJsonSchema;
            o.json_schema_arg = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--json-schema=")) {
            o.json_schema_arg = arg["--json-schema=".len..];
        } else if (std.mem.eql(u8, arg, "--lark")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingLarkGrammar;
            o.lark_arg = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--lark=")) {
            o.lark_arg = arg["--lark=".len..];
        } else if (std.mem.eql(u8, arg, "--regex")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingRegex;
            o.regex_arg = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--regex=")) {
            o.regex_arg = arg["--regex=".len..];
        } else if (try o.moe_cli.tryParse(arg)) {
            // Shared streamed-experts flags (fucina.weights.MoeStreamCli).
        } else if (std.mem.startsWith(u8, arg, "--moe-cache-slots=")) {
            o.moe_cli.armed = true;
            o.moe_cache_slots = try std.fmt.parseInt(usize, arg["--moe-cache-slots=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--moe-pin-mb=")) {
            o.moe_cli.armed = true;
            o.moe_pin_mb = try std.fmt.parseInt(usize, arg["--moe-pin-mb=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--moe-no-learn")) {
            o.moe_no_learn = true;
        } else if (std.mem.eql(u8, arg, "--kv-save")) {
            o.kv_save = true;
        } else if (std.mem.startsWith(u8, arg, "--kv-save=")) {
            o.kv_save = true;
            o.kv_save_arg = arg["--kv-save=".len..];
        } else if (std.mem.eql(u8, arg, "--moe-pilot")) {
            o.moe_cli.armed = true;
            o.moe_pilot = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-expert-top-p=")) {
            o.moe_expert_top_p = try std.fmt.parseFloat(f32, arg["--moe-expert-top-p=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--")) {
            try stdout.print("unknown flag: {s} (run with no arguments for usage)\n", .{arg});
            return error.UnknownArgument;
        } else {
            o.tokens = try parseTokenList(arg, token_buf);
        }
    }
    return o;
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
