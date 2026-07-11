//! Gemma 4 (gemma4 GGUF arch) CPU inference harness — token-id in, logits out.
//!
//! Focused on the logit-parity gate against llama.cpp: it can run from raw token
//! IDs and dump / compare the last token's logits in the same headerless
//! little-endian f32 format as the qwen3 example. It also loads Gemma's
//! SentencePiece tokenizer straight from the GGUF (`spm_tokenizer.zig`), so
//! `--prompt "<text>"` encodes text to ids, and `--chat`/`--repl` run a streamed
//! conversation using Gemma 4's `<|turn>` chat format (verified against this
//! GGUF's embedded chat_template).
//!
//!   zig build gemma4 -Doptimize=ReleaseFast -- <model.gguf> --chat "Hi" [--system S] [--think] [--greedy]
//!   zig build gemma4 -Doptimize=ReleaseFast -- <model.gguf> --repl
//!   zig build gemma4 -Doptimize=ReleaseFast -- <model.gguf> 2,651,235 [--logits-out P] [--compare-logits P] [--gen N]
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const Model = llm.gemma.gemma4.Model;
const Config = llm.gemma.gemma4.Config;

const default_tokens = [_]usize{ 2, 235280 }; // <bos> + a token

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print(
            \\usage: zig build gemma4 -- <model.gguf> [options]
            \\  text:    --prompt "hi" [--tok-only]           encode text (--tok-only skips weight load)
            \\  chat:    --chat "What is the capital of France?" [--system "..."] [--think] [--spec] [sampling…]
            \\  repl:    --repl [--system "..."] [--think] [--spec] [sampling…]   (multi-turn; streams replies)
            \\  sampling: --temp F --top-k N --top-p F --min-p F --repeat-penalty F --repeat-last-n N
            \\            --freq-penalty F --presence-penalty F --seed N --greedy --max N --stop "TEXT"
            \\            (defaults come from the GGUF's general.sampling.* — for this model temp 1.0/top_k 64/top_p 0.95)
            \\  grammar: --json-schema JSON|@FILE | --lark GRAMMAR|@FILE | --regex PATTERN
            \\           (constrained chat/repl decoding; needs a -Dllguidance=true build)
            \\  logits:  <comma-separated-token-ids> [--logits-out P] [--compare-logits P] [--repeat R] [--profile]
            \\  gen:     --gen N [--stop T] [--prompt "hi"]    bench/decode from a prompt or ids
            \\  experts: --experts=borrow|pack   MoE expert load (CPU builds). pack (default) x4-packs
            \\           experts for peak throughput; borrow maps them zero-copy — near-instant load
            \\           and ~half memory (no x4 widening), ideal for big Q6_K MoE that would swap.
            \\  other:   --bench R [--profile] | --info
            \\
        , .{});
        return;
    }

    var token_buf: [4096]usize = undefined;
    var tokens: []const usize = default_tokens[0..];
    var logits_out: ?[]const u8 = null;
    var compare_logits: ?[]const u8 = null;
    var gen_count: ?usize = null;
    var stop_token: ?usize = null;
    var repeat: usize = 1;
    var bench_reps: usize = 0;
    var profile_enabled = false;
    var info_flag = false;
    var experts_borrow = false; // --experts=borrow: zero-copy MoE load
    var prompt_text: ?[]const u8 = null;
    var tok_only = false;
    var chat_text: ?[]const u8 = null;
    var repl_flag = false;
    var system_text: ?[]const u8 = null;
    var think_flag = false;
    var spec_flag = false;
    var greedy_flag = false;
    var max_resp: usize = 512;
    // Sampling overrides (null = use the model's GGUF-recommended / llama.cpp default).
    var temp_arg: ?f32 = null;
    var top_k_arg: ?usize = null;
    var top_p_arg: ?f32 = null;
    var min_p_arg: ?f32 = null;
    var repeat_pen_arg: ?f32 = null;
    var repeat_last_n_arg: ?usize = null;
    var freq_pen_arg: ?f32 = null;
    var presence_pen_arg: ?f32 = null;
    var seed_arg: ?u64 = null;
    var stops_buf: [16][]const u8 = undefined;
    var stops_n: usize = 0;
    var json_schema_arg: ?[]const u8 = null;
    var lark_arg: ?[]const u8 = null;
    var regex_arg: ?[]const u8 = null;

    var arg_i: usize = 2;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--prompt")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingPrompt;
            prompt_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--prompt=")) {
            prompt_text = arg["--prompt=".len..];
        } else if (std.mem.eql(u8, arg, "--logits-out")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingLogitsPath;
            logits_out = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--logits-out=")) {
            logits_out = arg["--logits-out=".len..];
        } else if (std.mem.eql(u8, arg, "--compare-logits")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingComparePath;
            compare_logits = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--compare-logits=")) {
            compare_logits = arg["--compare-logits=".len..];
        } else if (std.mem.eql(u8, arg, "--gen")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingGenCount;
            gen_count = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--gen=")) {
            gen_count = try std.fmt.parseInt(usize, arg["--gen=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--stop") or std.mem.startsWith(u8, arg, "--stop=")) {
            const s = if (std.mem.startsWith(u8, arg, "--stop="))
                arg["--stop=".len..]
            else
                try argValue(args, &arg_i);
            // A bare integer is a stop *token id* (for --gen); any text is a chat
            // stop *string*. A numeric value serves both paths.
            if (std.fmt.parseInt(usize, s, 10)) |v| {
                stop_token = v;
            } else |_| {}
            if (stops_n < stops_buf.len) {
                stops_buf[stops_n] = s;
                stops_n += 1;
            }
        } else if (std.mem.eql(u8, arg, "--repeat")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingRepeatCount;
            repeat = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--repeat=")) {
            repeat = try std.fmt.parseInt(usize, arg["--repeat=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--bench")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingBenchCount;
            bench_reps = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--bench=")) {
            bench_reps = try std.fmt.parseInt(usize, arg["--bench=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--profile")) {
            profile_enabled = true;
        } else if (std.mem.eql(u8, arg, "--info")) {
            info_flag = true;
        } else if (std.mem.startsWith(u8, arg, "--experts=")) {
            const v = arg["--experts=".len..];
            if (std.mem.eql(u8, v, "borrow")) {
                experts_borrow = true;
            } else if (std.mem.eql(u8, v, "pack")) {
                experts_borrow = false;
            } else return error.InvalidExpertsMode;
        } else if (std.mem.eql(u8, arg, "--tok-only")) {
            tok_only = true;
        } else if (std.mem.eql(u8, arg, "--chat")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingChatText;
            chat_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--chat=")) {
            chat_text = arg["--chat=".len..];
        } else if (std.mem.eql(u8, arg, "--repl")) {
            repl_flag = true;
        } else if (std.mem.eql(u8, arg, "--system")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSystemText;
            system_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--system=")) {
            system_text = arg["--system=".len..];
        } else if (std.mem.eql(u8, arg, "--think")) {
            think_flag = true;
        } else if (std.mem.eql(u8, arg, "--spec")) {
            spec_flag = true;
        } else if (std.mem.eql(u8, arg, "--greedy")) {
            greedy_flag = true;
        } else if (try flagValue(args, &arg_i, "--max")) |v| {
            max_resp = try std.fmt.parseInt(usize, v, 10);
        } else if (try flagValue(args, &arg_i, "--temp")) |v| {
            temp_arg = try std.fmt.parseFloat(f32, v);
        } else if (try flagValue(args, &arg_i, "--top-k")) |v| {
            top_k_arg = try std.fmt.parseInt(usize, v, 10);
        } else if (try flagValue(args, &arg_i, "--top-p")) |v| {
            top_p_arg = try std.fmt.parseFloat(f32, v);
        } else if (try flagValue(args, &arg_i, "--min-p")) |v| {
            min_p_arg = try std.fmt.parseFloat(f32, v);
        } else if (try flagValue(args, &arg_i, "--repeat-penalty")) |v| {
            repeat_pen_arg = try std.fmt.parseFloat(f32, v);
        } else if (try flagValue(args, &arg_i, "--repeat-last-n")) |v| {
            repeat_last_n_arg = try std.fmt.parseInt(usize, v, 10);
        } else if (try flagValue(args, &arg_i, "--freq-penalty")) |v| {
            freq_pen_arg = try std.fmt.parseFloat(f32, v);
        } else if (try flagValue(args, &arg_i, "--presence-penalty")) |v| {
            presence_pen_arg = try std.fmt.parseFloat(f32, v);
        } else if (try flagValue(args, &arg_i, "--seed")) |v| {
            seed_arg = try std.fmt.parseInt(u64, v, 10);
        } else if (try flagValue(args, &arg_i, "--json-schema")) |v| {
            json_schema_arg = v;
        } else if (try flagValue(args, &arg_i, "--lark")) |v| {
            lark_arg = v;
        } else if (try flagValue(args, &arg_i, "--regex")) |v| {
            regex_arg = v;
        } else {
            tokens = try parseTokenList(arg, &token_buf);
        }
    }

    const allocator = std.heap.smp_allocator;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const load_start = nowNs(init.io);
    var file = try fucina.gguf.File.loadMmap(allocator, init.io, args[1]);
    var config = try Config.fromGguf(&file);
    config.borrow_experts = experts_borrow;

    // Build Gemma's SentencePiece tokenizer straight from the GGUF metadata,
    // *before* the (slow, multi-GB) weight load — so `--tok-only` and `--info`
    // can answer from the mmap'd metadata alone. The tokenizer copies the bytes
    // it needs, so it stays valid after `file` frees. A GGUF without SPM
    // metadata leaves it null and the token-id path still works.
    var spm: ?llm.spm_tokenizer.Tokenizer = llm.spm_tokenizer.Tokenizer.initFromGguf(allocator, &file, .{}) catch null;
    defer if (spm) |*t| t.deinit();
    const tok_ptr: ?*const llm.spm_tokenizer.Tokenizer = if (spm) |*t| t else null;

    // Encode a text prompt to token ids when requested (printed for comparison
    // against e.g. `llama-tokenize`).
    if (prompt_text) |text| {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        const ids32 = try t.encode(allocator, text);
        defer allocator.free(ids32);
        if (ids32.len > token_buf.len) return error.PromptTooLong;
        for (ids32, 0..) |id, i| token_buf[i] = id;
        tokens = token_buf[0..ids32.len];
        try stdout.print("prompt: \"{s}\" -> {d} tokens:", .{ text, tokens.len });
        for (tokens) |id| try stdout.print(" {d}", .{id});
        try stdout.print("\n", .{});
    }

    if (info_flag) {
        try printInfo(stdout, config);
        if (tok_ptr) |t| try stdout.print("tokenizer: SPM, vocab {d}  bos {?d}  eos {?d}\n", .{ t.vocabSize(), t.bosId(), t.eosId() });
        file.deinit();
        return;
    }
    if (tok_only) {
        file.deinit();
        return;
    }

    var model = try Model.loadGgufFromFile(&ctx, &file, config);
    defer model.deinit();

    // Chat format from the GGUF's embedded chat_template (read before `file`
    // frees); this harness targets the Gemma 4 `<|turn>` format either way.
    const chat_tmpl = llm.chat.Template.detect(file.getString("tokenizer.chat_template")) orelse
        llm.chat.Template{ .format = .gemma4 };

    // Effective sampling config: the GGUF's `general.sampling.*` recommendation
    // (over llama.cpp's defaults), then CLI overrides. Read before `file` frees.
    var sampler_cfg = samplingFromGguf(&file);
    if (temp_arg) |v| sampler_cfg.temperature = v;
    if (greedy_flag) sampler_cfg.temperature = 0;
    if (top_k_arg) |v| sampler_cfg.top_k = v;
    if (top_p_arg) |v| sampler_cfg.top_p = v;
    if (min_p_arg) |v| sampler_cfg.min_p = v;
    if (repeat_pen_arg) |v| sampler_cfg.repeat_penalty = v;
    if (repeat_last_n_arg) |v| sampler_cfg.repeat_last_n = v;
    if (freq_pen_arg) |v| sampler_cfg.freq_penalty = v;
    if (presence_pen_arg) |v| sampler_cfg.presence_penalty = v;
    if (seed_arg) |v| sampler_cfg.seed = v;

    file.deinit();
    const load_ns = nowNs(init.io) - load_start;

    // --json-schema/--lark/--regex: compile a llguidance constraint for the
    // chat sampler (the token-id/--gen paths are greedy parity harnesses and
    // do not take a grammar).
    var grammar_text: ?[]u8 = null;
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
        if (chat_text == null and !repl_flag) {
            try stdout.print("grammar flags apply to --chat/--repl only\n", .{});
            return error.GrammarWithoutChat;
        }
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        const grammar: llm.llguidance.Grammar = if (json_schema_arg) |v|
            .{ .json_schema = try grammarValue(init.io, allocator, v, &grammar_text) }
        else if (lark_arg) |v|
            .{ .lark = try grammarValue(init.io, allocator, v, &grammar_text) }
        else
            .{ .regex = regex_arg.? };
        // The turn ends on the template stop marker; the GGUF EOS is the
        // extra turn-end id runChat also registers.
        const turn_stop: ?u32 = t.tokenId(chat_tmpl.stopMarker()) orelse t.eosId();
        const extra: []const u32 = if (t.eosId()) |e| &.{e} else &.{};
        constraint = llm.llguidance.Constraint.init(allocator, t, grammar, .{
            .eos_token = turn_stop,
            .extra_eos = extra,
            .n_vocab = config.vocab_size,
        }) catch |err| switch (err) {
            error.LlguidanceNotEnabled => {
                try stdout.print("constrained decoding needs a build with -Dllguidance=true (see vendor/llguidance/README.md)\n", .{});
                return err;
            },
            else => return err,
        };
    }

    if (chat_text != null or repl_flag) {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        try printSampling(stdout, sampler_cfg, stops_buf[0..stops_n]);
        try stdout.print("load: {d:.3} s\n", .{seconds(load_ns)});
        try runChat(init.io, stdout, &ctx, &model, t, .{
            .template = chat_tmpl,
            .system = system_text,
            .think = think_flag,
            .spec = spec_flag,
            .sampler = sampler_cfg,
            .max_resp = max_resp,
            .chat_text = chat_text,
            .stops = stops_buf[0..stops_n],
            .processor = if (constraint) |*con| con.processor() else null,
        });
        return;
    }

    try stdout.print("tokens: {d}  load: {d:.3} s\n", .{ tokens.len, seconds(load_ns) });

    if (bench_reps > 0) {
        try runBench(init.io, allocator, stdout, &ctx, &model, tokens, gen_count orelse 32, bench_reps, profile_enabled);
        return;
    }
    if (gen_count) |n| {
        try runGenerate(init.io, allocator, stdout, &ctx, &model, tokens, n, stop_token, tok_ptr);
        return;
    }

    var logits: ?fucina.Tensor(.{ .seq, .vocab }) = null;
    defer if (logits) |*v| v.deinit();
    var profile: llm.gemma.gemma4.ForwardProfile = .{};
    const forward_start = nowNs(init.io);
    for (0..repeat) |_| {
        if (logits) |*v| {
            v.deinit();
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
    try stdout.print("forward: {d:.3} s ({d} reps)\ntop tokens:", .{ seconds(forward_ns), repeat });
    const top_values = try top.values.dataConst();
    const top_indices = try top.indices.dataConst();
    for (top_values, top_indices) |value, index| {
        try stdout.print(" {d}:{d:.4}", .{ index, value });
    }
    try stdout.print("\n", .{});
    if (profile_enabled) try printProfile(stdout, &profile, @floatFromInt(repeat), "profile avg ms");

    if (logits_out) |path| {
        try writeLogits(init.io, path, try final_logits.dataConst());
        try stdout.print("logits: {s}\n", .{path});
    }
    if (compare_logits) |path| {
        try compareLogits(init.io, allocator, stdout, path, try final_logits.dataConst());
    }
}

fn runGenerate(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: anytype,
    ctx: *fucina.ExecContext,
    model: *const Model,
    tokens: []const usize,
    n: usize,
    stop_token: ?usize,
    tok: ?*const llm.spm_tokenizer.Tokenizer,
) !void {
    const out = try allocator.alloc(usize, n);
    defer allocator.free(out);
    var kv = try model.initKvCache(ctx, tokens.len + n);
    defer kv.deinit();

    // Default the stop token to EOS when we have a tokenizer and none was given.
    const stop = stop_token orelse if (tok) |t| (if (t.eosId()) |e| @as(usize, e) else null) else null;

    const start = nowNs(io);
    const produced = try model.generate(ctx, &kv, tokens, out, .{ .max_new_tokens = n, .stop_token = stop });
    const ns = nowNs(io) - start;

    try stdout.print("generated {d} tokens in {d:.3} s ({d:.1} tok/s):\n", .{ produced, seconds(ns), @as(f64, @floatFromInt(produced)) / seconds(ns) });
    for (out[0..produced]) |t| try stdout.print("{d} ", .{t});
    try stdout.print("\n", .{});

    // Decode to text when a tokenizer is available.
    if (tok) |t| {
        var sd = llm.spm_tokenizer.StreamDecoder.init(t);
        defer sd.deinit(allocator);
        var line_buf: [4096]u8 = undefined;
        var line = std.Io.Writer.fixed(&line_buf);
        for (out[0..produced]) |id| sd.push(allocator, @intCast(id), &line) catch break;
        sd.flush(&line) catch {};
        try stdout.print("text: {s}\n", .{line.buffered()});
    }
}

/// Resolve a grammar flag value: `@PATH` reads the file (ownership parked in
/// `owned` for the caller's deferred free); anything else is the inline text.
fn grammarValue(io: std.Io, allocator: std.mem.Allocator, value: []const u8, owned: *?[]u8) ![]const u8 {
    if (!std.mem.startsWith(u8, value, "@")) return value;
    var file = try std.Io.Dir.cwd().openFile(io, value[1..], .{});
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
    owned.* = bytes;
    return bytes;
}

/// Read the value that follows a `--flag` argument, advancing the index.
fn argValue(args: []const []const u8, i: *usize) ![]const u8 {
    i.* += 1;
    if (i.* >= args.len) return error.MissingArgValue;
    return args[i.*];
}

/// Matches `--flag VALUE` (consuming the next arg) or `--flag=VALUE`;
/// null means `args[i]` is not this flag.
fn flagValue(args: []const []const u8, i: *usize, comptime flag: []const u8) !?[]const u8 {
    const arg = args[i.*];
    if (std.mem.eql(u8, arg, flag)) return try argValue(args, i);
    if (std.mem.startsWith(u8, arg, flag ++ "=")) return arg[flag.len + 1 ..];
    return null;
}

const ChatOptions = struct {
    template: llm.chat.Template = .{ .format = .gemma4 },
    system: ?[]const u8 = null,
    think: bool = false,
    /// Lossless draft-model-free speculative decoding (chat.Options.speculation).
    spec: bool = false,
    sampler: llm.sampler.Config = .{},
    max_resp: usize = 512,
    chat_text: ?[]const u8 = null,
    /// Extra text stop sequences (the turn-end token always stops generation).
    stops: []const []const u8 = &.{},
    /// Constrained-decoding logit processor (chat.Options.logit_processor).
    processor: ?llm.sampler.LogitProcessor = null,
};

/// Sampling config from the GGUF's `general.sampling.*` recommendation, falling
/// back per-key when the model doesn't set it. For this model the GGUF carries
/// Google's official Gemma 4 defaults (temp 1.0 / top_k 64 / top_p 0.95); the
/// fallbacks below complete that recommendation — min_p disabled (Google's
/// standardized config uses only top_k + top_p) and penalties disabled ("keep
/// repetition/presence penalty disabled unless you see looping").
fn samplingFromGguf(file: *const fucina.gguf.File) llm.sampler.Config {
    return .{
        .temperature = if (file.getFloat("general.sampling.temp")) |v| @floatCast(v) else 1.0,
        .top_k = if (file.getInt("general.sampling.top_k")) |v| @intCast(@max(@as(i64, 0), v)) else 64,
        .top_p = if (file.getFloat("general.sampling.top_p")) |v| @floatCast(v) else 0.95,
        .min_p = if (file.getFloat("general.sampling.min_p")) |v| @floatCast(v) else 0.0,
        .repeat_penalty = if (file.getFloat("general.sampling.penalty_repeat")) |v| @floatCast(v) else 1.0,
        .freq_penalty = if (file.getFloat("general.sampling.penalty_freq")) |v| @floatCast(v) else 0.0,
        .presence_penalty = if (file.getFloat("general.sampling.penalty_present")) |v| @floatCast(v) else 0.0,
        .repeat_last_n = if (file.getInt("general.sampling.penalty_last_n")) |v| @intCast(@max(@as(i64, 0), v)) else 64,
    };
}

fn printSampling(stdout: *std.Io.Writer, cfg: llm.sampler.Config, stops: []const []const u8) !void {
    if (cfg.temperature <= 0) {
        try stdout.print("sampling: greedy (temp 0)\n", .{});
    } else {
        try stdout.print("sampling: temp {d:.2} top_k {d} top_p {d:.2} min_p {d:.2} repeat {d:.2} freq {d:.2} presence {d:.2} (last_n {d}) seed {d}\n", .{
            cfg.temperature, cfg.top_k, cfg.top_p, cfg.min_p, cfg.repeat_penalty, cfg.freq_penalty, cfg.presence_penalty, cfg.repeat_last_n, cfg.seed,
        });
    }
    if (stops.len > 0) {
        try stdout.writeAll("stop strings:");
        for (stops) |s| try stdout.print(" \"{s}\"", .{s});
        try stdout.writeAll("\n");
    }
}

/// Streamed single-turn (`--chat`) or interactive multi-turn (`--repl`) chat
/// over the generic `Conversation` (one KV cache across turns — each turn only
/// prefills its own tokens; the reply streams to stdout token-by-token).
fn runChat(
    io: std.Io,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const Model,
    tok: *const llm.spm_tokenizer.Tokenizer,
    opts: ChatOptions,
) !void {
    // Turn-end ids beyond the template's <turn|> stop marker: the GGUF's own
    // EOS (usually the same id as <turn|>) and a stray SPM <eos> (id 1).
    var extra_stops_buf: [2]u32 = undefined;
    var extra_n: usize = 0;
    if (tok.eosId()) |e| {
        extra_stops_buf[extra_n] = e;
        extra_n += 1;
    }
    extra_stops_buf[extra_n] = 1;
    extra_n += 1;

    var convo = try llm.chat.Conversation(Model, llm.spm_tokenizer).init(ctx, model, tok, opts.template, .{
        .system = opts.system,
        .capacity = 4096,
        .max_response_tokens = opts.max_resp,
        .think_off = !opts.think,
        .sampler = opts.sampler,
        .extra_stop_ids = extra_stops_buf[0..extra_n],
        .stop_sequences = opts.stops,
        .logit_processor = opts.processor,
        .speculation = opts.spec,
        .io = io,
    });
    defer convo.deinit();

    if (opts.chat_text) |msg| {
        _ = try convo.send(msg, stdout);
        try stdout.writeAll("\n");
        if (convo.specStats()) |stats| {
            try stats.writeSummary(stdout);
            try stdout.writeAll("\n");
        }
        try stdout.flush();
        return;
    }

    var stdin_buf: [8192]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);
    const in = &stdin_reader.interface;
    try stdout.print("gemma4 chat{s} — type a message, empty line or Ctrl-D to quit:\n", .{if (opts.think) " (thinking)" else ""});
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

fn runBench(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: anytype,
    ctx: *fucina.ExecContext,
    model: *const Model,
    tokens: []const usize,
    gen: usize,
    reps: usize,
    profile_enabled: bool,
) !void {
    var kv = try model.initKvCache(ctx, tokens.len + gen);
    defer kv.deinit();

    const pps = try allocator.alloc(f64, reps);
    defer allocator.free(pps);
    const tgs = try allocator.alloc(f64, reps);
    defer allocator.free(tgs);
    var prefill_profile: llm.gemma.gemma4.ForwardProfile = .{};
    var decode_profile: llm.gemma.gemma4.ForwardProfile = .{};
    var decode_steps: usize = 0;
    var rep: usize = 0;
    while (rep <= reps) : (rep += 1) {
        kv.reset();
        if (profile_enabled and rep == 1) {
            prefill_profile = .{};
            decode_profile = .{};
        }
        if (rep == 1) fucina.internal.gpu.traceReset();
        const pp_start = nowNs(io);
        const profile_this = profile_enabled and rep != 0;
        var logits = if (profile_this)
            try model.forwardStepProfiled(ctx, io, &kv, tokens, 0, &prefill_profile)
        else
            try model.forwardStep(ctx, &kv, tokens, 0);
        const pp_ns = nowNs(io) - pp_start;

        const tg_start = nowNs(io);
        var steps: usize = 0;
        while (steps < gen and kv.len < kv.capacity) : (steps += 1) {
            const next = try argmaxLast(ctx, &logits);
            const fresh = if (profile_this)
                try model.forwardStepProfiled(ctx, io, &kv, &.{next}, kv.len, &decode_profile)
            else
                try model.forwardStep(ctx, &kv, &.{next}, kv.len);
            logits.deinit();
            logits = fresh;
        }
        const tg_ns = nowNs(io) - tg_start;
        logits.deinit();
        decode_steps = steps;
        if (rep == 0) continue;
        pps[rep - 1] = @as(f64, @floatFromInt(tokens.len)) / seconds(pp_ns);
        if (steps > 0) {
            tgs[rep - 1] = @as(f64, @floatFromInt(steps)) / seconds(tg_ns);
        } else {
            tgs[rep - 1] = 0;
        }
    }
    fucina.internal.gpu.traceDump();
    try stdout.print("warm bench: {d} reps (+1 warmup), prompt {d} tok, decode {d} steps\n", .{ reps, tokens.len, decode_steps });
    try printBenchStat(stdout, "prefill", pps);
    if (decode_steps > 0) try printBenchStat(stdout, "decode ", tgs);
    if (profile_enabled) {
        try printProfile(stdout, &prefill_profile, @floatFromInt(reps), "prefill profile avg ms");
        if (decode_steps > 0) {
            try printProfile(stdout, &decode_profile, @floatFromInt(reps * decode_steps), "decode profile avg ms/token");
        }
    }
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

fn argmaxLast(ctx: *fucina.ExecContext, logits: *const fucina.Tensor(.{ .seq, .vocab })) !usize {
    var last = try logits.narrow(ctx, .seq, logits.dim(.seq) - 1, 1);
    defer last.deinit();
    var index = try last.argmax(ctx, .vocab);
    defer index.deinit();
    return @intCast(try index.item());
}

fn printInfo(stdout: anytype, config: Config) !void {
    try stdout.print("gemma4 config:\n", .{});
    try stdout.print("  vocab={d} hidden={d} layers={d}\n", .{ config.vocab_size, config.hidden_size, config.num_layers });
    try stdout.print("  query_heads={d} (kv_heads vary per layer) head_dim global={d} swa={d}\n", .{ config.num_attention_heads, config.head_dim_global, config.head_dim_swa });
    try stdout.print("  sliding_window={d} shared_kv_layers={d}\n", .{ config.sliding_window, config.shared_kv_layers });
    try stdout.print("  experts={d} used={d} expert_ffn={d} shared_ffn={d}\n", .{ config.num_experts, config.num_experts_used, config.moe_intermediate_size, config.intermediate_size });
    try stdout.print("  per_layer_input={d} rope_base={d:.1} rope_base_swa={d:.1} eps={e}\n", .{ config.per_layer_input_size, config.rope_theta, config.rope_theta_swa, config.rms_norm_eps });
    try stdout.print("  final_logit_softcapping={d:.3}\n", .{config.final_logit_softcapping});
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

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
}

fn millisI128(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn printProfile(stdout: anytype, profile: *const llm.gemma.gemma4.ForwardProfile, denom: f64, label: []const u8) !void {
    try stdout.print("{s}:", .{label});
    try stdout.print(" embed={d:.3}", .{millisI128(profile.embed_ns) / denom});
    try stdout.print(" attn={d:.3}", .{millisI128(profile.attn_ns) / denom});
    try stdout.print(" ffn={d:.3}", .{millisI128(profile.ffn_ns) / denom});
    try stdout.print(" dense={d:.3}", .{millisI128(profile.dense_ns) / denom});
    try stdout.print(" moe_router={d:.3}", .{millisI128(profile.moe_router_ns) / denom});
    try stdout.print(" moe_count_sort={d:.3}", .{millisI128(profile.moe_count_sort_ns) / denom});
    try stdout.print(" moe_gather={d:.3}", .{millisI128(profile.moe_gather_ns) / denom});
    if (profile.moe_expert_wall_ns != 0) {
        try stdout.print(" moe_expert_wall={d:.3}", .{millisI128(profile.moe_expert_wall_ns) / denom});
        try stdout.print(" task_sum_gate_up={d:.3}", .{millisI128(profile.moe_task_gate_up_ns) / denom});
        try stdout.print(" task_sum_act={d:.3}", .{millisI128(profile.moe_task_act_ns) / denom});
        try stdout.print(" task_sum_down={d:.3}", .{millisI128(profile.moe_task_down_ns) / denom});
    }
    // the batch path (prefill/GPU) reports through these three
    try stdout.print(" moe_gate_up={d:.3}", .{millisI128(profile.moe_gate_up_ns) / denom});
    try stdout.print(" moe_act={d:.3}", .{millisI128(profile.moe_act_ns) / denom});
    try stdout.print(" moe_down={d:.3}", .{millisI128(profile.moe_down_ns) / denom});
    try stdout.print(" moe_scatter={d:.3}", .{millisI128(profile.moe_scatter_ns) / denom});
    try stdout.print(" final={d:.3}", .{millisI128(profile.final_ns) / denom});
    try stdout.print(" layers={d}\n", .{profile.layers});
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
        .{ max_abs, sum_abs / n, @sqrt(sum_sq / n), value_top, values[value_top], reference_top, reference[reference_top], value_top == reference_top },
    );
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
        const got = try file.readStreaming(io, &.{bytes[read_len..]});
        if (got == 0) return error.EndOfStream;
        read_len += got;
    }
    const out = try allocator.alloc(f32, byte_len / @sizeOf(f32));
    errdefer allocator.free(out);
    for (out, 0..) |*dst, i| {
        const bits = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
        dst.* = @bitCast(bits);
    }
    return out;
}
