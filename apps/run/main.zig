//! fucina-run: the one GGUF runner over the architecture registry.
//!
//! `zig build run -- <model.gguf> [flags]` sniffs `general.architecture`,
//! resolves the family through `models.registry`, and drives it through the
//! decoder contract: plain or sampled completion, template chat/REPL
//! (`models.text.chat.Conversation`), teacher-forced NLL, a pp/tg bench,
//! and the generic parity surface (`--tokenize`, `--logits-out`,
//! `--compare-logits`, `--step1`). The streamed-experts levers
//! (`--moe-*`, `models.moe_stream_cli`) apply where the family loader
//! supports them. Family extras ride the same binary: the deepseek2
//! MLA/DSA dials, glm4moe `--mtp` self-speculation (the library verify
//! loop), and the inkling multimodal tower and marker-format chat.
//!
//!   zig build run -- models/Qwen3-0.6B-f16.gguf --prompt "2+2=" --gen 8
//!   zig build run -- glm-4.5-air.gguf --prompt "..." --mtp --moe-stream
//!   zig build run -- inkling.gguf --chat "hi" --repl
//!
//! The deepseek4 speculation/parity harness and the diffusion-gemma
//! block-diffusion runner keep their own apps (`zig build deepseek4`,
//! `zig build diffusion-gemma`): their loops rewind by snapshot/restore or
//! leave the autoregressive contract entirely.

const std = @import("std");
const fucina = @import("fucina");
const models = @import("fucina_models");
const png = @import("facedetect_image");

const registry = models.registry;
const decoder = models.decoder;

const usage =
    \\usage: zig build run -- <model.gguf> [comma-token-ids] [flags]
    \\  completion: --prompt "..." | --prompt-file=P   [--gen N | -n N] [--ctx=N] [--prefill-chunk=N]
    \\              sampling: --temp F --top-k=N --top-p=F --min-p=F --repeat-penalty=F --seed=N
    \\  chat:       --chat "msg" | --repl   [--system "..."] [--no-think] [--gen N]
    \\  eval:       --nll-file=P [--nll-tokens=N]   teacher-forced NLL over a text file
    \\  parity:     --tokenize FILE | --logits-out P | --compare-logits P [--max-abs F] | --step1
    \\  bench:      --bench R   (best-of-R pp/tg; with --image/--audio: tower preprocess+encode)
    \\  moe:        --moe-stream --moe-cache-mb=N ... (models.moe_stream_cli) plus
    \\              --moe-pilot --moe-cache-route --moe-route-j=N --moe-route-m=N
    \\              --moe-pin-mb=N --moe-no-learn --moe-cache-slots=N
    \\  deepseek2:  --mla=full|latent --dsa --dsa-top-k=N --index-probe --index-share=N
    \\              --moe-experts=N --moe-top-p=F --moe-skip-miss=F
    \\  spec:       --spec   draft-model-free speculative decode (cascade SAM + recycling;
    \\              rewind-capable families, greedy-only, lossless)
    \\  glm4moe:    --mtp[=depth]   native multi-token-prediction speculative decode
    \\  inkling:    --mmproj P (--image f.png | --audio f.wav) --prompt "... <__media__> ..." [--embd-out P]
    \\  other:      --info --threads N
    \\
;

const Cli = struct {
    model_path: []const u8 = "",
    prompt: []const u8 = "The capital of France is",
    prompt_file: ?[]const u8 = null,
    gen: usize = 32,
    gen_set: bool = false,
    ctx_capacity: usize = 0,
    prefill_chunk: usize = 64,
    info: bool = false,
    tokenize_file: ?[]const u8 = null,
    logits_out: ?[]const u8 = null,
    compare_logits: ?[]const u8 = null,
    max_abs: f64 = 1e-4,
    step1: bool = false,
    bench_reps: usize = 0,
    nll_file: ?[]const u8 = null,
    nll_tokens: usize = 0,
    chat_text: ?[]const u8 = null,
    system_text: ?[]const u8 = null,
    repl: bool = false,
    no_think: bool = false,
    // Sampling (temperature 0 = greedy, the oracle/benchmark path).
    temp: f32 = 0,
    top_k: usize = 0,
    top_p: f32 = 1.0,
    min_p: f32 = 0,
    repeat_penalty: f32 = 1.0,
    seed: u64 = 0,
    // Streamed-experts levers.
    moe_cli: models.moe_stream_cli.MoeStreamCli = .{},
    moe_pilot: bool = false,
    moe_cache_route: bool = false,
    moe_route_j: usize = 2,
    moe_route_m: usize = 12,
    moe_pin_mb: ?usize = null,
    moe_no_learn: bool = false,
    moe_cache_slots: ?usize = null,
    // deepseek2 dials.
    mla_full: bool = false,
    dsa: bool = false,
    dsa_top_k: usize = 0,
    index_probe: bool = false,
    index_share: usize = 0,
    moe_experts: usize = 0,
    moe_top_p: f32 = 1.0,
    moe_skip_miss: f32 = 0,
    // glm4moe MTP depth (0 = plain decode).
    mtp_depth: usize = 0,
    spec: bool = false,
    // inkling multimodal.
    mmproj: ?[]const u8 = null,
    image: ?[]const u8 = null,
    audio: ?[]const u8 = null,
    embd_out: ?[]const u8 = null,
    // Raw comma-separated token ids (parity harness input; skips encoding).
    raw_tokens: []const usize = &.{},
    token_buf: [65536]usize = undefined,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print(usage, .{});
        return;
    }

    var cli = Cli{ .model_path = args[1] };
    var arg_i: usize = 2;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (try flagValue(args, &arg_i, "--prompt")) |v| {
            cli.prompt = v;
        } else if (try flagValue(args, &arg_i, "--prompt-file")) |v| {
            cli.prompt_file = v;
        } else if (try flagValue(args, &arg_i, "--gen")) |v| {
            cli.gen = try std.fmt.parseInt(usize, v, 10);
            cli.gen_set = true;
        } else if (try flagValue(args, &arg_i, "-n")) |v| {
            cli.gen = try std.fmt.parseInt(usize, v, 10);
            cli.gen_set = true;
        } else if (try flagValue(args, &arg_i, "--ctx")) |v| {
            cli.ctx_capacity = try std.fmt.parseInt(usize, v, 10);
        } else if (try flagValue(args, &arg_i, "--prefill-chunk")) |v| {
            cli.prefill_chunk = @max(1, try std.fmt.parseInt(usize, v, 10));
        } else if (std.mem.eql(u8, arg, "--info")) {
            cli.info = true;
        } else if (try flagValue(args, &arg_i, "--tokenize")) |v| {
            cli.tokenize_file = v;
        } else if (try flagValue(args, &arg_i, "--logits-out")) |v| {
            cli.logits_out = v;
        } else if (try flagValue(args, &arg_i, "--compare-logits")) |v| {
            cli.compare_logits = v;
        } else if (try flagValue(args, &arg_i, "--max-abs")) |v| {
            cli.max_abs = try std.fmt.parseFloat(f64, v);
        } else if (std.mem.eql(u8, arg, "--step1")) {
            cli.step1 = true;
        } else if (try flagValue(args, &arg_i, "--bench")) |v| {
            cli.bench_reps = try std.fmt.parseInt(usize, v, 10);
        } else if (try flagValue(args, &arg_i, "--nll-file")) |v| {
            cli.nll_file = v;
        } else if (try flagValue(args, &arg_i, "--nll-tokens")) |v| {
            cli.nll_tokens = try std.fmt.parseInt(usize, v, 10);
        } else if (try flagValue(args, &arg_i, "--chat")) |v| {
            cli.chat_text = v;
        } else if (try flagValue(args, &arg_i, "--system")) |v| {
            cli.system_text = v;
        } else if (std.mem.eql(u8, arg, "--repl")) {
            cli.repl = true;
        } else if (std.mem.eql(u8, arg, "--no-think")) {
            cli.no_think = true;
        } else if (try flagValue(args, &arg_i, "--temp")) |v| {
            cli.temp = try std.fmt.parseFloat(f32, v);
        } else if (try flagValue(args, &arg_i, "--top-k")) |v| {
            cli.top_k = try std.fmt.parseInt(usize, v, 10);
        } else if (try flagValue(args, &arg_i, "--top-p")) |v| {
            cli.top_p = try std.fmt.parseFloat(f32, v);
        } else if (try flagValue(args, &arg_i, "--min-p")) |v| {
            cli.min_p = try std.fmt.parseFloat(f32, v);
        } else if (try flagValue(args, &arg_i, "--repeat-penalty")) |v| {
            cli.repeat_penalty = try std.fmt.parseFloat(f32, v);
        } else if (try flagValue(args, &arg_i, "--seed")) |v| {
            cli.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (try flagValue(args, &arg_i, "--threads")) |v| {
            fucina.parallel.setMaxThreads(try std.fmt.parseInt(usize, v, 10));
        } else if (try cli.moe_cli.tryParse(arg)) {
            // Shared streamed-experts flags (models.moe_stream_cli).
        } else if (std.mem.eql(u8, arg, "--moe-pilot")) {
            cli.moe_cli.armed = true;
            cli.moe_pilot = true;
        } else if (std.mem.eql(u8, arg, "--moe-cache-route")) {
            cli.moe_cli.armed = true;
            cli.moe_cache_route = true;
        } else if (try flagValue(args, &arg_i, "--moe-route-j")) |v| {
            cli.moe_route_j = try std.fmt.parseInt(usize, v, 10);
        } else if (try flagValue(args, &arg_i, "--moe-route-m")) |v| {
            cli.moe_route_m = try std.fmt.parseInt(usize, v, 10);
        } else if (try flagValue(args, &arg_i, "--moe-pin-mb")) |v| {
            cli.moe_cli.armed = true;
            cli.moe_pin_mb = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--moe-no-learn")) {
            cli.moe_cli.armed = true;
            cli.moe_no_learn = true;
        } else if (try flagValue(args, &arg_i, "--moe-cache-slots")) |v| {
            cli.moe_cli.armed = true;
            cli.moe_cache_slots = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--mla=full")) {
            cli.mla_full = true;
        } else if (std.mem.eql(u8, arg, "--mla=latent")) {
            cli.mla_full = false;
        } else if (std.mem.eql(u8, arg, "--dsa")) {
            cli.dsa = true;
        } else if (try flagValue(args, &arg_i, "--dsa-top-k")) |v| {
            cli.dsa = true;
            cli.dsa_top_k = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--index-probe")) {
            cli.index_probe = true;
        } else if (try flagValue(args, &arg_i, "--index-share")) |v| {
            cli.index_share = try std.fmt.parseInt(usize, v, 10);
        } else if (try flagValue(args, &arg_i, "--moe-experts")) |v| {
            cli.moe_experts = try std.fmt.parseInt(usize, v, 10);
        } else if (try flagValue(args, &arg_i, "--moe-top-p")) |v| {
            cli.moe_top_p = try std.fmt.parseFloat(f32, v);
        } else if (try flagValue(args, &arg_i, "--moe-skip-miss")) |v| {
            cli.moe_skip_miss = try std.fmt.parseFloat(f32, v);
        } else if (std.mem.eql(u8, arg, "--spec")) {
            cli.spec = true;
        } else if (std.mem.eql(u8, arg, "--mtp")) {
            cli.mtp_depth = 2;
        } else if (std.mem.startsWith(u8, arg, "--mtp=")) {
            // Depth caps at 8: the verify runs kernel-pinned, and 8 keeps
            // the verify batch under the non-quant kernel thresholds that
            // bound losslessness (see models.text.speculative.mtp).
            cli.mtp_depth = @min(try std.fmt.parseInt(usize, arg["--mtp=".len..], 10), 8);
        } else if (try flagValue(args, &arg_i, "--mmproj")) |v| {
            cli.mmproj = v;
        } else if (try flagValue(args, &arg_i, "--image")) |v| {
            cli.image = v;
        } else if (try flagValue(args, &arg_i, "--audio")) |v| {
            cli.audio = v;
        } else if (try flagValue(args, &arg_i, "--embd-out")) |v| {
            cli.embd_out = v;
        } else if (arg.len > 0 and std.ascii.isDigit(arg[0])) {
            var it = std.mem.splitScalar(u8, arg, ',');
            var count: usize = 0;
            while (it.next()) |part| {
                if (part.len == 0) continue;
                if (count == cli.token_buf.len) return error.TokenListTooLong;
                cli.token_buf[count] = try std.fmt.parseInt(usize, part, 10);
                count += 1;
            }
            cli.raw_tokens = cli.token_buf[0..count];
        } else {
            try stdout.print("unknown flag: {s}\n{s}", .{ arg, usage });
            return error.UnknownArgument;
        }
    }
    // Parity dumps without an explicit --gen stop at the prefill logits.
    if (!cli.gen_set and (cli.logits_out != null or cli.compare_logits != null)) cli.gen = 0;

    const allocator = std.heap.smp_allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = try fucina.gguf.File.loadMmapAuto(allocator, init.io, cli.model_path);
    const arch = file.getString("general.architecture") orelse {
        file.deinit();
        return error.UnknownArchitecture;
    };
    if (cli.info) {
        try stdout.print("arch: {s}\n", .{arch});
        file.deinit();
        return;
    }
    inline for (registry.families) |entry| {
        if (std.mem.eql(u8, arch, entry.arch)) {
            return runFamily(entry.Family, init, allocator, &ctx, &file, &cli, stdout);
        }
    }
    const known = comptime blk: {
        var s: []const u8 = "";
        for (registry.families) |entry| s = s ++ (if (s.len == 0) "" else ", ") ++ entry.arch;
        break :blk s;
    };
    try stdout.print("unsupported architecture '{s}' (registered: {s})\n", .{ arch, known });
    file.deinit();
    return error.UnsupportedArchitecture;
}

/// `--flag value` or `--flag=value`; null when `arg` is not `flag`.
fn flagValue(args: []const []const u8, i: *usize, comptime flag: []const u8) !?[]const u8 {
    const arg = args[i.*];
    if (std.mem.eql(u8, arg, flag)) {
        i.* += 1;
        if (i.* >= args.len) return error.MissingFlagValue;
        return args[i.*];
    }
    if (std.mem.startsWith(u8, arg, flag ++ "=")) return arg[flag.len + 1 ..];
    return null;
}

fn runFamily(
    comptime Family: type,
    init: std.process.Init,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    file: *fucina.gguf.File,
    cli: *Cli,
    stdout: *std.Io.Writer,
) !void {
    const Model = Family.Model;
    const is_deepseek2 = comptime Family == models.deepseek2.model.Family;
    const is_glm4moe = comptime Family == models.glm4moe.model.Family;
    const is_inkling = comptime Family == models.inkling.model.Family;

    const load_start = nowNs(init.io);
    var tokenizer = try Family.tokenizer(allocator, file);
    defer tokenizer.deinit();

    // --tokenize FILE: encode a text file, one token id per line (the
    // llama-tokenize parity harness); no model weights needed.
    if (cli.tokenize_file) |path| {
        file.deinit();
        const bytes = try readTextFile(init.io, allocator, path);
        defer allocator.free(bytes);
        const ids = try tokenizer.encodeRaw(allocator, bytes);
        defer allocator.free(ids);
        for (ids) |id| try stdout.print("{d}\n", .{id});
        return;
    }

    // Family guards for the family-specific dials.
    if (!is_deepseek2 and (cli.mla_full or cli.dsa or cli.dsa_top_k > 0 or cli.index_probe or
        cli.index_share > 0 or cli.moe_experts > 0 or cli.moe_top_p < 1.0 or cli.moe_skip_miss > 0))
    {
        try stdout.print("--mla/--dsa/--index-*/--moe-experts/--moe-top-p/--moe-skip-miss are deepseek2 dials\n", .{});
        return error.UnknownArgument;
    }
    if (!is_glm4moe and cli.mtp_depth > 0) {
        try stdout.print("--mtp is the glm4moe native MTP path (deepseek4's rides `zig build deepseek4`)\n", .{});
        return error.UnknownArgument;
    }
    if (cli.spec and cli.mtp_depth > 0) {
        try stdout.print("--spec and --mtp are alternative draft sources; pick one\n", .{});
        return error.UnknownArgument;
    }
    if (cli.spec and !comptime Model.caps.rewind) {
        try stdout.print("--spec needs a rewind-capable family (KV truncate); this architecture has none\n", .{});
        return error.UnknownArgument;
    }
    if (cli.spec and cli.temp > 0) {
        try stdout.print("--spec is greedy-only in the runner (drop --temp)\n", .{});
        return error.UnknownArgument;
    }
    if (!is_inkling and (cli.mmproj != null or cli.image != null or cli.audio != null or cli.embd_out != null)) {
        try stdout.print("--mmproj/--image/--audio/--embd-out are inkling multimodal flags\n", .{});
        return error.UnknownArgument;
    }
    if (cli.index_probe and cli.index_share >= 2) {
        try stdout.print("--index-probe measures the exact path; drop --index-share\n", .{});
        return error.UnknownArgument;
    }

    // Chat template, sniffed before the load consumes the file.
    const template: ?models.text.chat.Template =
        models.text.chat.Template.detect(file.getString("tokenizer.chat_template")) orelse
        if (Family.template_fallback) |format| models.text.chat.Template{ .format = format } else null;

    // Prompt tokens (needed before the load for the cache/rope capacity).
    var prompt_file_bytes: ?[]u8 = null;
    defer if (prompt_file_bytes) |b| allocator.free(b);
    if (cli.prompt_file) |path| {
        prompt_file_bytes = try readTextFile(init.io, allocator, path);
        cli.prompt = prompt_file_bytes.?;
    }
    var tokens: std.ArrayList(usize) = .empty;
    defer tokens.deinit(allocator);
    if (cli.raw_tokens.len > 0) {
        try tokens.appendSlice(allocator, cli.raw_tokens);
    } else {
        try encodePrompt(Family, allocator, &tokenizer, cli.prompt, &tokens);
    }
    const capacity: usize = if (cli.ctx_capacity > 0)
        cli.ctx_capacity
    else if (cli.chat_text != null or cli.repl)
        4096
    else
        @max(2048, tokens.items.len + cli.gen + 8);

    // Streamed-experts options (family loaders that support them consume
    // them; the rest ignore the levers by contract).
    var moe_stream = try cli.moe_cli.options(cli.model_path);
    if (moe_stream) |*m| {
        m.pilot = cli.moe_pilot;
        m.cache_route = cli.moe_cache_route;
        m.route_sacred = cli.moe_route_j;
        m.route_window = cli.moe_route_m;
        if (cli.moe_pin_mb) |mb| m.pin_bytes = mb << 20;
        if (cli.moe_no_learn) m.auto_pin = false;
        if (cli.moe_cache_slots) |n| m.cache_slots_per_layer = n;
    }

    var model = try loadModel(Family, ctx, file, cli, moe_stream, capacity);
    defer model.deinit();
    // The stats go through the SAME buffered stdout writer as everything
    // else: stdout's positional writes and stderr's offset-advancing writes
    // cannot safely share one redirected file.
    defer if (comptime @hasField(Model, "expert_store")) {
        if (model.expert_store) |store| models.moe_stream_cli.reportAndSaveMoeStream(store, !cli.moe_no_learn, stdout);
    };
    file.deinit();
    try stdout.print("load: {d:.3} s\n", .{seconds(nowNs(init.io) - load_start)});

    if (is_deepseek2) {
        model.index_share_every = cli.index_share;
        if (cli.dsa_top_k > 0) model.config.indexer_top_k = cli.dsa_top_k;
        if (cli.moe_experts > 0) {
            if (cli.moe_experts > model.config.num_experts_used) return error.InvalidExpertCount;
            try stdout.print("moe: experts used {d} -> {d} (inference-time truncation)\n", .{ model.config.num_experts_used, cli.moe_experts });
            model.config.num_experts_used = cli.moe_experts;
        }
        if (cli.moe_top_p < 1.0 or cli.moe_skip_miss > 0) {
            model.moe_top_p = cli.moe_top_p;
            model.moe_skip_miss_below = cli.moe_skip_miss;
            try stdout.print("moe: dynamic expert drop (top-p {d:.2}, skip-miss-below {d:.3})\n", .{ cli.moe_top_p, cli.moe_skip_miss });
        }
    }

    // Multimodal (inkling): tower encode + mixed-row prefill.
    if (is_inkling and (cli.image != null or cli.audio != null)) {
        return runInklingMedia(init, allocator, ctx, &model, &tokenizer, cli, stdout);
    }

    // Chat / REPL.
    if (cli.chat_text != null or cli.repl) {
        if (is_inkling) return runInklingChat(init, allocator, ctx, &model, &tokenizer, cli, stdout);
        if (comptime Model.caps.rewind and decoder.ModelPtr(Model) == *const Model and @hasField(Model.Cache, "capacity")) {
            const tpl = template orelse {
                try stdout.print("no chat template in this GGUF and no fallback for this arch\n", .{});
                return error.NoChatTemplate;
            };
            return runChat(Family, init, allocator, ctx, &model, &tokenizer, tpl, capacity, cli, stdout);
        }
        try stdout.print("chat is not wired for this arch here (qwen35/inkling/deepseek4 chat is served by lmserve; glm4moe and deepseek2 have no chat engine)\n", .{});
        return error.NoChatTemplate;
    }

    // Teacher-forced NLL over a plain-encoded text file.
    if (cli.nll_file) |path| {
        return runNll(Family, init, allocator, ctx, &model, &tokenizer, path, cli, stdout);
    }

    if (tokens.items.len == 0) return error.EmptyPrompt;
    try stdout.print("prompt tokens: {d}\n", .{tokens.items.len});

    // --bench R: warm pp/tg best-of-R, load once.
    if (cli.bench_reps > 0) {
        const n_gen = if (cli.gen_set and cli.gen > 0) cli.gen else 32;
        var best_pp: f64 = 0;
        var best_tg: f64 = 0;
        for (0..cli.bench_reps) |_| {
            var bcache = try makeCache(Family, ctx, &model, tokens.items.len + n_gen + 1, cli.mla_full);
            defer bcache.deinit();
            const t0 = nowNs(init.io);
            var last = try model.forwardStep(ctx, &bcache, tokens.items, 0);
            const t1 = nowNs(init.io);
            const next = tokens.items[tokens.items.len - 1]; // fixed token, no sampling
            var produced: usize = 0;
            while (produced < n_gen) : (produced += 1) {
                const fresh = try model.forwardStep(ctx, &bcache, &.{next}, bcache.len());
                last.deinit();
                last = fresh;
            }
            const t2 = nowNs(init.io);
            last.deinit();
            best_pp = @max(best_pp, @as(f64, @floatFromInt(tokens.items.len)) / seconds(t1 - t0));
            best_tg = @max(best_tg, @as(f64, @floatFromInt(n_gen)) / seconds(t2 - t1));
        }
        try stdout.print("bench: pp{d} {d:.2} tok/s | tg{d} {d:.2} tok/s (best of {d})\n", .{ tokens.items.len, best_pp, n_gen, best_tg, cli.bench_reps });
        return;
    }

    var cache = try makeCache(Family, ctx, &model, capacity, cli.mla_full);
    defer cache.deinit();
    if (is_deepseek2) {
        if (cli.index_probe) try cache.enableDsaProbe(model.config.num_layers);
    }

    // Prefill: chunked forward steps (--step1 = token-at-a-time, the
    // decode-vs-batch parity rung; --prefill-chunk=1 is equivalent).
    const chunk: usize = if (cli.step1) 1 else cli.prefill_chunk;
    const prefill_start = nowNs(init.io);
    var logits: ?fucina.Tensor(.{ .seq, .vocab }) = null;
    defer if (logits) |*t| t.deinit();
    var fed: usize = 0;
    while (fed < tokens.items.len) {
        const end = @min(fed + chunk, tokens.items.len);
        const fresh = try model.forwardStep(ctx, &cache, tokens.items[fed..end], fed);
        if (logits) |*t| t.deinit();
        logits = fresh;
        fed = end;
    }
    try stdout.print("prefill: {d:.1} ms ({d} tokens, chunk {d})\n", .{ seconds(nowNs(init.io) - prefill_start) * 1e3, tokens.items.len, chunk });

    if (cli.logits_out) |path| {
        try writeF32File(init.io, path, try logits.?.dataConst());
        try stdout.print("logits written: {s} ({d} values)\n", .{ path, (try logits.?.dataConst()).len });
    }
    if (cli.compare_logits) |path| {
        const ok = try compareLogits(init.io, allocator, stdout, path, try logits.?.dataConst(), cli.max_abs);
        if (!ok) return error.LogitsMismatch;
    }

    // glm4moe --mtp: the family's nextn head drafts through MtpDraftSource
    // and the shared SpeculativeDecoder verifies/commits/rewinds.
    if (is_glm4moe and cli.mtp_depth > 0) {
        return runGlmMtp(init, allocator, ctx, &model, &cache, &tokenizer, &tokens, logits.?, cli, stdout);
    }

    // Generic --spec: the cascade drafts, the shared decoder verifies —
    // any rewind-capable family, no sidecar weights.
    if (cli.spec) {
        if (comptime Model.caps.rewind) {
            return runSpec(Model, init, allocator, ctx, &model, &cache, &tokenizer, &tokens, logits.?, cli, stdout);
        }
        unreachable; // rejected by the guard above
    }

    // Decode: greedy by default, sampled past --temp 0.
    const eos = tokenizer.eosId();
    var sampler = models.text.sampler.Sampler.init(samplingConfig(cli));
    var reply: std.ArrayList(u8) = .empty;
    defer reply.deinit(allocator);
    const decode_start = nowNs(init.io);
    var produced: usize = 0;
    const prompt_len = tokens.items.len;
    while (produced < cli.gen) : (produced += 1) {
        const next = try sampler.next(ctx, &logits.?, tokens.items);
        if (eos != null and next == eos.?) break;
        if (tokenizer.isEos(@intCast(next))) break;
        try tokenizer.decodeAppend(allocator, @intCast(next), &reply);
        try tokens.append(allocator, next);
        const fresh = try model.forwardStep(ctx, &cache, &.{next}, cache.len());
        logits.?.deinit();
        logits = fresh;
    }
    const decode_ns = nowNs(init.io) - decode_start;
    try stdout.print("decode: {d} steps, {d:.1} ms, {d:.2} tok/s\n", .{ produced, seconds(decode_ns) * 1e3, @as(f64, @floatFromInt(produced)) / seconds(decode_ns) });
    if (is_deepseek2) {
        if (model.index_share_every >= 2) {
            try stdout.print("index-share: every {d} — selections computed {d}, reused {d}\n", .{ model.index_share_every, cache.share_computed, cache.share_reused });
        }
        if (cache.probe) |*p| try p.report(stdout);
    }
    try stdout.print("generated ids:", .{});
    for (tokens.items[prompt_len..]) |t| try stdout.print(" {d}", .{t});
    if (cli.prompt_file != null) {
        try stdout.print("\nprompt: ({d} bytes from file)\ntext:  {s}\n", .{ cli.prompt.len, reply.items });
    } else {
        try stdout.print("\nprompt: {s}\ntext:  {s}\n", .{ cli.prompt, reply.items });
    }
}

/// Family-specific load wiring: deepseek2 takes its own LoadOptions (the
/// registry-level `Family.load` has no `dsa` lever), glm4moe sizes its rope
/// table from the capacity; everything else rides `Family.load`.
fn loadModel(
    comptime Family: type,
    ctx: *fucina.ExecContext,
    file: *fucina.gguf.File,
    cli: *const Cli,
    moe_stream: ?fucina.weights.MoeStreamOptions,
    capacity: usize,
) !Family.Model {
    if (comptime Family == models.deepseek2.model.Family) {
        var load_options: Family.Model.LoadOptions = if (moe_stream) |m| .{ .moe_stream = m } else .{};
        load_options.dsa = cli.dsa;
        return Family.Model.loadGgufFromFileOptions(ctx, file, load_options);
    }
    return Family.load(ctx, file, .{ .moe_stream = moe_stream, .max_positions = capacity });
}

/// deepseek2's cache carries the `--mla=full|latent` mode; the decoder
/// contract's `initCache` covers every other family.
fn makeCache(
    comptime Family: type,
    ctx: *fucina.ExecContext,
    model: *Family.Model,
    capacity: usize,
    mla_full: bool,
) !Family.Model.Cache {
    if (comptime Family == models.deepseek2.model.Family) {
        return model.initCacheMode(capacity, if (mla_full) .full else .latent);
    }
    return model.initCache(ctx, capacity);
}

fn samplingConfig(cli: *const Cli) models.text.sampler.Config {
    return .{
        .temperature = cli.temp,
        .top_k = cli.top_k,
        .top_p = cli.top_p,
        .min_p = cli.min_p,
        .repeat_penalty = cli.repeat_penalty,
        .seed = cli.seed,
    };
}

/// Prompt encoding, family for family what the family runners do: inkling
/// is raw byte-BPE with no BOS; deepseek2 opens with `[gMASK]<sop>` when the
/// vocab has them (glm-dsa checkpoints degenerate without them); glm4moe
/// opens `[gMASK]<sop>` by id after BOS; gemma4 (SPM) and deepseek4 prepend
/// BOS; qwen3/qwen35 encode with no BOS.
fn encodePrompt(
    comptime Family: type,
    allocator: std.mem.Allocator,
    tokenizer: *const Family.Tokenizer,
    text: []const u8,
    out: *std.ArrayList(usize),
) !void {
    if (comptime Family == models.inkling.model.Family) {
        const ids = try tokenizer.encodeRaw(allocator, text);
        defer allocator.free(ids);
        for (ids) |id| try out.append(allocator, id);
        return;
    }
    if (comptime Family == models.deepseek2.model.Family) {
        if (tokenizer.tokenId("[gMASK]")) |gmask| {
            try out.append(allocator, gmask);
            if (tokenizer.tokenId("<sop>")) |sop| try out.append(allocator, sop);
        } else if (tokenizer.bosId()) |b| {
            try out.append(allocator, b);
        }
    } else if (comptime Family == models.glm4moe.model.Family) {
        if (tokenizer.bosId()) |b| {
            try out.append(allocator, b);
            if (b == 151331) try out.append(allocator, 151333);
        }
    } else if (comptime Family == models.deepseek4.model.Family or Family == models.gemma.model.Family) {
        if (tokenizer.bosId()) |b| try out.append(allocator, b);
    }
    const ids = try tokenizer.encode(allocator, text);
    defer allocator.free(ids);
    for (ids) |id| try out.append(allocator, id);
}

/// Teacher-forced mean NLL (and perplexity) over the first `--nll-tokens`
/// supervised positions of a plain-encoded text file, stepped token by
/// token through the family forward.
fn runNll(
    comptime Family: type,
    init: std.process.Init,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    model: *Family.Model,
    tokenizer: *const Family.Tokenizer,
    path: []const u8,
    cli: *const Cli,
    stdout: *std.Io.Writer,
) !void {
    const text = try readTextFile(init.io, allocator, path);
    defer allocator.free(text);
    const ids32 = try tokenizer.encode(allocator, text);
    defer allocator.free(ids32);
    const n = if (cli.nll_tokens > 0) @min(ids32.len, cli.nll_tokens + 1) else ids32.len;
    if (n < 2) return error.NllTextTooShort;

    var cache = try makeCache(Family, ctx, model, @max(2048, n + 4), cli.mla_full);
    defer cache.deinit();
    var total: f64 = 0;
    var count: usize = 0;
    // deepseek2 feeds BOS as context first (the family runner's shape);
    // the other families score from the first encoded token.
    if (comptime Family == models.deepseek2.model.Family) {
        if (tokenizer.bosId()) |b| {
            var l0 = try model.forwardStep(ctx, &cache, &.{b}, cache.len());
            l0.deinit();
        }
    }
    var prev: usize = @intCast(ids32[0]);
    for (ids32[1..n]) |next_id| {
        var lg_t = try model.forwardStep(ctx, &cache, &.{prev}, cache.len());
        defer lg_t.deinit();
        const lg = try lg_t.dataConst();
        var maxv: f32 = lg[0];
        for (lg) |v| maxv = @max(maxv, v);
        var sum_exp: f64 = 0;
        for (lg) |v| sum_exp += @exp(@as(f64, v - maxv));
        total += @as(f64, maxv) + @log(sum_exp) - @as(f64, lg[next_id]);
        count += 1;
        prev = @intCast(next_id);
    }
    const nll = total / @as(f64, @floatFromInt(count));
    try stdout.print("nll: {d:.4} (ppl {d:.2}) over {d} supervised tokens\n", .{ nll, @exp(nll), count });
}

/// Template chat/REPL over the generic Conversation (families whose
/// forward takes `*const Model`; glm4moe/deepseek4 refresh model-held
/// scratch and stay with their family paths).
fn runChat(
    comptime Family: type,
    init: std.process.Init,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    model: *const Family.Model,
    tokenizer: *const Family.Tokenizer,
    template: models.text.chat.Template,
    capacity: usize,
    cli: *const Cli,
    stdout: *std.Io.Writer,
) !void {
    const Convo = models.text.chat.Conversation(Family.Model, Family.Tok);
    var convo = try Convo.init(ctx, model, tokenizer, template, .{
        .system = cli.system_text,
        .capacity = capacity,
        .max_response_tokens = if (cli.gen_set) cli.gen else 1024,
        .think_off = cli.no_think,
        .sampler = samplingConfig(cli),
    });
    defer convo.deinit();

    var stdin_buf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(init.io, &stdin_buf);

    var turn: usize = 0;
    while (true) : (turn += 1) {
        var user_owned: ?[]u8 = null;
        defer if (user_owned) |u| allocator.free(u);
        const user: []const u8 = blk: {
            if (turn == 0 and cli.chat_text != null) break :blk cli.chat_text.?;
            if (!cli.repl) break :blk "";
            try stdout.print("\nyou> ", .{});
            try stdout.flush();
            const line = stdin_reader.interface.takeDelimiterExclusive('\n') catch break;
            if (line.len == 0) break;
            user_owned = try allocator.dupe(u8, line);
            break :blk user_owned.?;
        };
        if (user.len == 0) break;
        try stdout.print("model> ", .{});
        try stdout.flush();
        _ = try convo.send(user, stdout);
        try stdout.print("\n", .{});
        try stdout.flush();
        if (!cli.repl) break;
    }
}

/// glm4moe `--mtp`: commit the prefill's first greedy token, then let the
/// shared verify loop drive (lossless; greedy-matching prefixes commit).
fn runGlmMtp(
    init: std.process.Init,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    model: *models.glm4moe.model.Model,
    cache: *models.glm4moe.model.Model.Cache,
    tokenizer: *const models.text.tokenizer.Tokenizer,
    tokens: *std.ArrayList(usize),
    prefill_logits: fucina.Tensor(.{ .seq, .vocab }),
    cli: *const Cli,
    stdout: *std.Io.Writer,
) !void {
    const Model = models.glm4moe.model.Model;
    const eos = tokenizer.eosId();
    var depth = cli.mtp_depth;
    if (model.mtp == null) {
        try stdout.print("model has no nextn (MTP) layer; --mtp ignored\n", .{});
        depth = 0;
    }
    var next_token = try argmaxRow(try prefill_logits.dataConst());

    var draft_src: ?models.text.speculative.mtp.MtpDraftSource(Model) = null;
    defer if (draft_src) |*d| d.deinit();
    var spec: ?models.text.speculative.core.SpeculativeDecoder(Model) = null;
    defer if (spec) |*d| d.deinit();
    if (depth > 0) {
        draft_src = try models.text.speculative.mtp.MtpDraftSource(Model).init(ctx, model, cache.capacity, depth);
        try draft_src.?.observePrefill();
        spec = try models.text.speculative.core.SpeculativeDecoder(Model).init(allocator, draft_src.?.source(), .{
            .max_draft = depth,
            .min_draft = 1,
            .stop_token = if (eos) |e| @as(usize, e) else null,
            // Always speculate at full depth (no cost gate, no adaptive
            // budget): the runner's deterministic benchmarking shape.
            .min_speedup = 0,
            .adapt_budget = false,
        });
    }

    var reply: std.ArrayList(u8) = .empty;
    defer reply.deinit(allocator);
    var produced: usize = 0;
    var forwards: usize = 0;
    const decode_start = nowNs(init.io);

    if (depth == 0) {
        while (produced < cli.gen) {
            if (eos != null and next_token == eos.?) break;
            try tokenizer.decodeAppend(allocator, @intCast(next_token), &reply);
            try tokens.append(allocator, next_token);
            produced += 1;
            var rows = try model.forwardStep(ctx, cache, &.{next_token}, cache.len());
            defer rows.deinit();
            forwards += 1;
            next_token = try argmaxRow(try rows.dataConst());
        }
    } else {
        // Decoder invariant: history holds every committed token and its
        // LAST element is not yet forwarded.
        var stopped = eos != null and next_token == eos.?;
        if (!stopped) {
            try tokenizer.decodeAppend(allocator, @intCast(next_token), &reply);
            try tokens.append(allocator, next_token);
            produced += 1;
        }
        var sampler = models.text.sampler.Sampler.init(.{}); // greedy
        var emit = ReplyEmit{ .allocator = allocator, .tokenizer = tokenizer, .reply = &reply, .eos = eos };
        const sink = models.text.speculative.core.TokenSink{ .ptr = &emit, .func = ReplyEmit.emit };
        while (!stopped and produced < cli.gen) {
            spec.?.options.max_draft = @min(depth, cli.gen - produced -| 1);
            const committed = try spec.?.step(ctx, model, cache, &sampler, tokens, sink);
            produced += committed - @intFromBool(emit.saw_eos);
            stopped = emit.saw_eos;
        }
        if (emit.saw_eos) _ = tokens.pop();
        forwards = spec.?.stats.spec_steps + spec.?.stats.fallback_steps + spec.?.stats.disabled_steps;
    }
    const decode_ns = nowNs(init.io) - decode_start;
    try stdout.print("decode: {d} tokens in {d} forwards, {d:.1} ms, {d:.2} tok/s ({d:.2} tok/forward)\n", .{
        produced, forwards, seconds(decode_ns) * 1e3, @as(f64, @floatFromInt(produced)) / seconds(decode_ns), @as(f64, @floatFromInt(produced)) / @as(f64, @floatFromInt(@max(forwards, 1))),
    });
    if (spec) |*d| {
        if (d.stats.drafted > 0) {
            try stdout.print("mtp: {d}/{d} drafts accepted ({d:.1}%)\n", .{ d.stats.accepted, d.stats.drafted, @as(f64, @floatFromInt(d.stats.accepted)) * 100.0 / @as(f64, @floatFromInt(d.stats.drafted)) });
        }
    }
    try stdout.print("generated ids:", .{});
    for (tokens.items[tokens.items.len - produced ..]) |t| try stdout.print(" {d}", .{t});
    try stdout.print("\nprompt: {s}\ntext:  {s}\n", .{ cli.prompt, reply.items });
}

/// Generic `--spec`: draft-model-free speculation for any rewind-capable
/// family — the cascade drafts from the committed stream (conversation SAM
/// + token recycling), the shared SpeculativeDecoder verifies, commits and
/// rewinds. Lossless: greedy-matching prefixes commit, so the output is
/// byte-identical to the plain greedy decode loop.
fn runSpec(
    comptime Model: type,
    init: std.process.Init,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    model: *Model,
    cache: *Model.Cache,
    tokenizer: anytype,
    tokens: *std.ArrayList(usize),
    prefill_logits: fucina.Tensor(.{ .seq, .vocab }),
    cli: *const Cli,
    stdout: *std.Io.Writer,
) !void {
    const eos = tokenizer.eosId();
    const next_token = try argmaxRow(try prefill_logits.dataConst());
    // The logits row is [1, vocab]: the family-agnostic vocab size.
    const vocab_size = (try prefill_logits.dataConst()).len;

    const spec_options = models.text.speculative.core.Options{
        .stop_token = if (eos) |e| @as(usize, e) else null,
    };
    var index = try models.text.speculative.cascade.SpeculationIndex.init(allocator, vocab_size);
    defer index.deinit();
    index.accounting_min_draft = spec_options.min_draft;
    var spec = try models.text.speculative.core.SpeculativeDecoder(Model).init(allocator, index.asDraftSource(), spec_options);
    defer spec.deinit();
    spec.io = init.io; // live verify/plain cost measurement for the auto-off gate
    index.observe(tokens.items); // the prompt is committed context

    var reply: std.ArrayList(u8) = .empty;
    defer reply.deinit(allocator);
    var produced: usize = 0;
    var forwards: usize = 0;
    const decode_start = nowNs(init.io);
    const base_max_draft = spec.options.max_draft;

    // Decoder invariant: history holds every committed token and its LAST
    // element is not yet forwarded (same bootstrap as runGlmMtp).
    var stopped = eos != null and next_token == eos.?;
    if (!stopped) {
        try tokenizer.decodeAppend(allocator, @intCast(next_token), &reply);
        try tokens.append(allocator, next_token);
        // The committed stream must reach the index token-for-token, in
        // order (the cascade's contract); the decoder observes everything
        // it commits itself from here on.
        index.observe(tokens.items[tokens.items.len - 1 ..]);
        produced += 1;
    }
    var sampler = models.text.sampler.Sampler.init(.{}); // greedy (the losslessness oracle)
    const Emit = struct {
        allocator: std.mem.Allocator,
        tokenizer: @TypeOf(tokenizer),
        reply: *std.ArrayList(u8),
        eos: ?u32,
        saw_eos: bool = false,

        fn emit(ptr: *anyopaque, token: usize) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (self.eos) |e| if (token == e) {
                self.saw_eos = true;
                return;
            };
            try self.tokenizer.decodeAppend(self.allocator, @intCast(token), self.reply);
        }
    };
    var emit = Emit{ .allocator = allocator, .tokenizer = tokenizer, .reply = &reply, .eos = eos };
    const sink = models.text.speculative.core.TokenSink{ .ptr = &emit, .func = Emit.emit };
    while (!stopped and produced < cli.gen) {
        spec.options.max_draft = @min(base_max_draft, cli.gen - produced -| 1);
        const committed = try spec.step(ctx, model, cache, &sampler, tokens, sink);
        produced += committed - @intFromBool(emit.saw_eos);
        stopped = emit.saw_eos;
    }
    if (emit.saw_eos) _ = tokens.pop();
    forwards = spec.stats.spec_steps + spec.stats.fallback_steps + spec.stats.disabled_steps;

    const decode_ns = nowNs(init.io) - decode_start;
    try stdout.print("decode: {d} tokens in {d} forwards, {d:.1} ms, {d:.2} tok/s ({d:.2} tok/forward)\n", .{
        produced, forwards, seconds(decode_ns) * 1e3, @as(f64, @floatFromInt(produced)) / seconds(decode_ns), @as(f64, @floatFromInt(produced)) / @as(f64, @floatFromInt(@max(forwards, 1))),
    });
    if (spec.stats.drafted > 0) {
        try stdout.print("spec: {d}/{d} drafts accepted ({d:.1}%)\n", .{ spec.stats.accepted, spec.stats.drafted, @as(f64, @floatFromInt(spec.stats.accepted)) * 100.0 / @as(f64, @floatFromInt(spec.stats.drafted)) });
    }
    try stdout.print("generated ids:", .{});
    for (tokens.items[tokens.items.len - produced ..]) |t| try stdout.print(" {d}", .{t});
    if (cli.prompt_file != null) {
        try stdout.print("\nprompt: ({d} bytes from file)\ntext:  {s}\n", .{ cli.prompt.len, reply.items });
    } else {
        try stdout.print("\nprompt: {s}\ntext:  {s}\n", .{ cli.prompt, reply.items });
    }
}

/// The MTP decoder's token sink: decode committed tokens into the reply
/// buffer; a committed stop token is recorded, not decoded.
const ReplyEmit = struct {
    allocator: std.mem.Allocator,
    tokenizer: *const models.text.tokenizer.Tokenizer,
    reply: *std.ArrayList(u8),
    eos: ?u32,
    saw_eos: bool = false,

    fn emit(ptr: *anyopaque, token: usize) anyerror!void {
        const self: *ReplyEmit = @ptrCast(@alignCast(ptr));
        if (self.eos) |e| if (token == e) {
            self.saw_eos = true;
            return;
        };
        try self.tokenizer.decodeAppend(self.allocator, @intCast(token), self.reply);
    }
};

// ---------------------------------------------------------------- inkling

const InklingChatOptions = struct {
    first_user: ?[]const u8,
    system: ?[]const u8,
    repl: bool,
    no_think: bool,
    max_tokens: usize,
    temperature: f32,
};

/// Single-turn (`--chat`) or multi-turn (`--repl`) wire-format chat through
/// the inkling engine: the reply splits into the thinking block (shown dim
/// unless --no-think) and the visible content.
fn runInklingChat(
    init: std.process.Init,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    model: *const models.inkling.model.Model,
    tokenizer: *const models.text.tokenizer.Tokenizer,
    cli: *const Cli,
    stdout: *std.Io.Writer,
) !void {
    const opts = InklingChatOptions{
        .first_user = cli.chat_text,
        .system = cli.system_text,
        .repl = cli.repl,
        .no_think = cli.no_think,
        .max_tokens = if (cli.gen_set and cli.gen > 0) cli.gen else 256,
        .temperature = cli.temp,
    };
    const inkling_chat = models.inkling.chat;
    var engine = try inkling_chat.Engine(models.text.tokenizer).init(ctx, model, tokenizer);

    var messages: std.ArrayList(models.text.chat.Message) = .empty;
    defer messages.deinit(allocator);
    if (opts.system) |s| try messages.append(allocator, .{ .role = .system, .content = s });

    var stdin_buf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(init.io, &stdin_buf);

    var turn: usize = 0;
    while (true) : (turn += 1) {
        var user_owned: ?[]u8 = null;
        defer if (user_owned) |u| allocator.free(u);
        const user: []const u8 = blk: {
            if (turn == 0 and opts.first_user != null) break :blk opts.first_user.?;
            if (!opts.repl) break :blk "";
            try stdout.print("\nyou> ", .{});
            try stdout.flush();
            const line = stdin_reader.interface.takeDelimiterExclusive('\n') catch break;
            if (line.len == 0) break;
            user_owned = try allocator.dupe(u8, line);
            break :blk user_owned.?;
        };
        if (user.len == 0) break;

        try messages.append(allocator, .{ .role = .user, .content = user });

        const prompt = try inkling_chat.renderPrompt(allocator, messages.items, .{ .think_off = opts.no_think });
        defer allocator.free(prompt);
        const ids32 = try tokenizer.encodeRaw(allocator, prompt);
        defer allocator.free(ids32);
        const ids = try allocator.alloc(usize, ids32.len);
        defer allocator.free(ids);
        for (ids, ids32) |*d, s| d.* = s;

        var reply: std.ArrayList(u8) = .empty;
        defer reply.deinit(allocator);
        var reply_writer = std.Io.Writer.Allocating.fromArrayList(allocator, &reply);
        _ = try engine.generate(ids, .{
            .sampling = .{ .temperature = opts.temperature },
            .max_tokens = opts.max_tokens,
            .think_off = opts.no_think,
        }, &reply_writer.writer);
        reply = reply_writer.toArrayList();

        const split = splitInklingReply(reply.items);
        if (!opts.no_think and split.thinking.len > 0) {
            try stdout.print("\x1b[2m[thinking] {s}\x1b[0m\n", .{split.thinking});
        }
        try stdout.print("model> {s}\n", .{split.content});
        try stdout.flush();

        // Record the assistant turn for multi-turn context (content only;
        // prior reasoning is dropped, matching the reference templates).
        try messages.append(allocator, .{ .role = .assistant, .content = try allocator.dupe(u8, split.content) });
        if (!opts.repl) break;
    }
    for (messages.items) |m| {
        if (m.role == .assistant) allocator.free(@constCast(m.content));
    }
}

/// Split a marker-wrapped inkling reply into its thinking block (between
/// `<|content_thinking|>` and `<|content_text|>`) and its visible content.
fn splitInklingReply(wrapped: []const u8) struct { thinking: []const u8, content: []const u8 } {
    const ct = models.inkling.chat.tok_content_text;
    const cth = models.inkling.chat.tok_content_thinking;
    var thinking: []const u8 = "";
    var content: []const u8 = wrapped;
    if (std.mem.indexOf(u8, wrapped, ct)) |ci| {
        content = wrapped[ci + ct.len ..];
        const pre = wrapped[0..ci];
        if (std.mem.indexOf(u8, pre, cth)) |ti| thinking = pre[ti + cth.len ..] else thinking = pre;
    } else if (std.mem.indexOf(u8, wrapped, cth)) |ti| {
        thinking = wrapped[ti + cth.len ..];
        content = "";
    }
    return .{ .thinking = std.mem.trim(u8, thinking, " \r\n\t"), .content = std.mem.trim(u8, content, " \r\n\t") };
}

/// Inkling multimodal: split the prompt on `<__media__>`, run the tower,
/// and build mixed rows with llama.cpp mtmd's marker framing.
fn runInklingMedia(
    init: std.process.Init,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    model: *models.inkling.model.Model,
    tokenizer: *const models.text.tokenizer.Tokenizer,
    cli: *const Cli,
    stdout: *std.Io.Writer,
) !void {
    if (cli.image != null and cli.audio != null) return error.OneMediaOnly;
    const mm_path = cli.mmproj orelse return error.MissingMmprojPath;
    const marker = "<__media__>";
    const pos = std.mem.indexOf(u8, cli.prompt, marker) orelse return error.MissingMediaMarker;
    const prompt_before = cli.prompt[0..pos];
    const prompt_after = cli.prompt[pos + marker.len ..];

    var media_rows: []f32 = &.{};
    defer if (media_rows.len > 0) allocator.free(media_rows);
    var n_media_tokens: usize = 0;

    var mm_file = try fucina.gguf.File.loadMmapAuto(allocator, init.io, mm_path);
    var mm = try models.inkling.mmproj.MmProj.loadGgufFromFile(ctx, &mm_file);
    mm_file.deinit();
    defer mm.deinit();
    const n_embd_mm = mm.n_embd;

    // --bench R on a media flag: repeat preprocess+encode, report best ms.
    const media_reps = if (cli.bench_reps > 0) cli.bench_reps else 1;
    var best_pre_ns: i96 = std.math.maxInt(i96);
    var best_enc_ns: i96 = std.math.maxInt(i96);

    if (cli.image) |ipath| {
        const bytes = try readTextFile(init.io, allocator, ipath);
        defer allocator.free(bytes);
        var img = try png.decodePng(allocator, bytes);
        defer img.deinit();
        var patch_rows: usize = 0;
        var patch_cols: usize = 0;
        for (0..media_reps) |_| {
            if (media_rows.len > 0) allocator.free(media_rows);
            const t0 = nowNs(init.io);
            var patches = try models.inkling.mmproj.preprocessImage(allocator, &mm, img.pixels, img.width, img.height);
            defer patches.deinit();
            const t1 = nowNs(init.io);
            media_rows = try mm.visionEncode(ctx, patches.data, patches.nPatches());
            const t2 = nowNs(init.io);
            n_media_tokens = patches.nPatches();
            patch_rows = patches.patch_rows;
            patch_cols = patches.patch_cols;
            best_pre_ns = @min(best_pre_ns, t1 - t0);
            best_enc_ns = @min(best_enc_ns, t2 - t1);
        }
        try stdout.print("image: {d}x{d} -> {d}x{d} patches = {d} tokens\n", .{ img.width, img.height, patch_rows, patch_cols, n_media_tokens });
    } else if (cli.audio) |apath| {
        var audio = try models.parakeet.frontend.loadWav16kMonoFile(allocator, init.io, apath);
        defer audio.deinit(allocator);
        for (0..media_reps) |_| {
            if (media_rows.len > 0) allocator.free(media_rows);
            const t0 = nowNs(init.io);
            var dmel = try models.inkling.mmproj.preprocessAudio(allocator, audio.samples);
            defer dmel.deinit();
            const t1 = nowNs(init.io);
            media_rows = try mm.audioEncode(ctx, dmel.data, dmel.n_frames);
            const t2 = nowNs(init.io);
            n_media_tokens = dmel.n_frames;
            best_pre_ns = @min(best_pre_ns, t1 - t0);
            best_enc_ns = @min(best_enc_ns, t2 - t1);
        }
        try stdout.print("audio: {d} samples -> {d} frame tokens\n", .{ audio.samples.len, n_media_tokens });
    }
    if (cli.bench_reps > 0) {
        try stdout.print("media bench: preprocess {d:.2} ms | encode {d:.2} ms (best of {d})\n", .{ seconds(best_pre_ns) * 1e3, seconds(best_enc_ns) * 1e3, media_reps });
        return;
    }

    if (cli.embd_out) |path| {
        try writeF32File(init.io, path, media_rows);
        try stdout.print("media embeddings written: {s} ({d} x {d})\n", .{ path, n_media_tokens, n_embd_mm });
        // Tower-only mode: nothing else requested means the decoder (whose
        // width may differ) never runs.
        if (cli.logits_out == null and cli.compare_logits == null and cli.gen == 0) return;
    }
    if (n_embd_mm != model.config.hidden_size) return error.MmprojWidthMismatch;

    // Framing text around the media rows (mtmd's img_beg / aud_beg+end).
    const before_str = try std.mem.concat(allocator, u8, &.{
        prompt_before,
        if (cli.image != null) "<|content_image|>" else "<|content_audio_input|>",
    });
    defer allocator.free(before_str);
    const after_str = try std.mem.concat(allocator, u8, &.{
        if (cli.audio != null) "<|audio_end|>" else "",
        prompt_after,
    });
    defer allocator.free(after_str);
    const ids_before = try tokenizer.encodeRaw(allocator, before_str);
    defer allocator.free(ids_before);
    const ids_after = try tokenizer.encodeRaw(allocator, after_str);
    defer allocator.free(ids_after);

    const Row = models.inkling.model.Model.Row;
    const items = try allocator.alloc(Row, ids_before.len + n_media_tokens + ids_after.len);
    defer allocator.free(items);
    var it: usize = 0;
    for (ids_before) |id| {
        items[it] = .{ .token = id };
        it += 1;
    }
    for (0..n_media_tokens) |mi| {
        items[it] = .{ .embd = media_rows[mi * n_embd_mm ..][0..n_embd_mm] };
        it += 1;
    }
    for (ids_after) |id| {
        items[it] = .{ .token = id };
        it += 1;
    }
    try stdout.print("mm rows: {d} text + {d} media + {d} text\n", .{ ids_before.len, n_media_tokens, ids_after.len });

    var cache = try model.initCache(ctx, items.len + cli.gen + 1);
    defer cache.deinit();
    var last = try model.stepMixed(ctx, &cache, items);
    defer last.deinit();

    if (cli.logits_out) |path| {
        try writeF32File(init.io, path, try last.dataConst());
        try stdout.print("logits written: {s} ({d} values)\n", .{ path, (try last.dataConst()).len });
    }
    if (cli.compare_logits) |path| {
        const ok = try compareLogits(init.io, allocator, stdout, path, try last.dataConst(), cli.max_abs);
        if (!ok) return error.LogitsMismatch;
    }
    if (cli.gen > 0) {
        try stdout.print("generated ids:", .{});
        var produced: usize = 0;
        while (produced < cli.gen) {
            const next = try argmaxRow(try last.dataConst());
            try stdout.print(" {d}", .{next});
            produced += 1;
            if (tokenizer.isEos(@intCast(next)) or produced == cli.gen) break;
            const gr = try model.forwardStep(ctx, &cache, &.{next}, cache.len());
            last.deinit();
            last = gr;
        }
        try stdout.print("\n", .{});
    }
}

// ---------------------------------------------------------------- helpers

fn argmaxRow(logits: []const f32) !usize {
    if (logits.len == 0) return error.EmptyLogits;
    var best: usize = 0;
    var best_v: f32 = -std.math.inf(f32);
    for (logits, 0..) |v, i| {
        if (v > best_v) {
            best_v = v;
            best = i;
        }
    }
    return best;
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
}

fn writeF32File(io: std.Io, path: []const u8, values: []const f32) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    defer writer.interface.flush() catch {};
    try writer.interface.writeAll(std.mem.sliceAsBytes(values));
}

/// Compare against a raw f32 dump; positions where both sides are -inf
/// (the padded-vocab mask) count as equal. Returns false when max_abs
/// exceeds the gate or the argmax differs; the caller exits nonzero.
fn compareLogits(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, path: []const u8, values: []const f32, max_abs_gate: f64) !bool {
    const reference = try readF32File(io, allocator, path);
    defer allocator.free(reference);
    if (reference.len != values.len) {
        try stdout.print("compare logits: length mismatch ours={d} ref={d}\n", .{ values.len, reference.len });
        return false;
    }

    var max_abs: f64 = 0;
    var sum_abs: f64 = 0;
    var value_top: usize = 0;
    var reference_top: usize = 0;
    var finite: usize = 0;
    for (values, reference, 0..) |value, ref, i| {
        if (std.math.isInf(value) and std.math.isInf(ref) and (value < 0) == (ref < 0)) continue;
        const diff = @abs(@as(f64, @floatCast(value)) - @as(f64, @floatCast(ref)));
        max_abs = @max(max_abs, diff);
        sum_abs += diff;
        finite += 1;
        if (value > values[value_top]) value_top = i;
        if (ref > reference[reference_top]) reference_top = i;
    }

    const aligned = value_top == reference_top;
    const pass = aligned and max_abs <= max_abs_gate;
    try stdout.print(
        "compare logits: max_abs={d:.6} mean_abs={d:.6} top={d}:{d:.4} ref_top={d}:{d:.4} aligned={} gate={d:.6} {s}\n",
        .{ max_abs, sum_abs / @as(f64, @floatFromInt(@max(finite, 1))), value_top, values[value_top], reference_top, reference[reference_top], aligned, max_abs_gate, if (pass) "PASS" else "FAIL" },
    );
    return pass;
}

fn readF32File(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    const bytes = try readTextFile(io, allocator, path);
    defer allocator.free(bytes);
    if (bytes.len % 4 != 0) return error.InvalidF32File;
    const out = try allocator.alloc(f32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

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
