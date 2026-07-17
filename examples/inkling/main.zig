//! Inkling (thinkingmachines/Inkling) GGUF inference runner and parity
//! harness. Reference: llama.cpp PR #25731 pinned @ 1cb0374; oracle
//! binaries under refs/llama.cpp-inkling/build-cpu/bin (llama-completion,
//! llama-tokenize) and tools/llama_logits.cpp compiled against that build.
//!
//! Parity ladder (docs/PORTING.md §5):
//!   rung 1  --tokenize FILE          vs llama-tokenize (token-ID-exact)
//!   rung 2  --logits-out PATH        raw f32 dump of last-position logits
//!           --compare-logits PATH    exit-code gate vs a llama_logits dump
//!   rung 3  --gen N (greedy)         vs llama-completion --temp 0
//!   --step1 replays the prompt one token at a time (decode path must
//!   match batch prefill bit-for-bit on this host implementation).

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const png = @import("facedetect_image");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print("usage: zig build inkling -- <model.gguf> [comma-token-ids] [--prompt \"...\"] [--tokenize FILE] [--gen N] [--step1] [--logits-out PATH] [--compare-logits PATH] [--max-abs F] [--info]\n", .{});
        try stdout.print("multimodal: --mmproj <mmproj.gguf> [--image f.png | --audio f.wav] --prompt \"... <__media__> ...\" [--embd-out PATH]\n", .{});
        return;
    }

    const allocator = std.heap.smp_allocator;

    var token_buf: [65536]usize = undefined;
    var tokens: []const usize = &.{};
    var tokenize_file: ?[]const u8 = null;
    var prompt_text: ?[]const u8 = null;
    var logits_out: ?[]const u8 = null;
    var compare_logits_path: ?[]const u8 = null;
    var max_abs_gate: f64 = 1e-4;
    var gen_count: usize = 0;
    var step1 = false;
    var info_flag = false;
    var mmproj_path: ?[]const u8 = null;
    var image_path: ?[]const u8 = null;
    var audio_path: ?[]const u8 = null;
    var embd_out: ?[]const u8 = null;
    var bench_reps: usize = 0;
    var chat_text: ?[]const u8 = null;
    var system_text: ?[]const u8 = null;
    var repl_flag = false;
    var no_think = false;
    var temp_arg: f32 = 0;

    var arg_i: usize = 2;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--tokenize")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTokenizePath;
            tokenize_file = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingPrompt;
            prompt_text = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--gen")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingGenCount;
            gen_count = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.eql(u8, arg, "--logits-out")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingLogitsPath;
            logits_out = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--compare-logits")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingComparePath;
            compare_logits_path = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--max-abs")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingMaxAbs;
            max_abs_gate = try std.fmt.parseFloat(f64, args[arg_i]);
        } else if (std.mem.eql(u8, arg, "--step1")) {
            step1 = true;
        } else if (std.mem.eql(u8, arg, "--info")) {
            info_flag = true;
        } else if (std.mem.eql(u8, arg, "--mmproj")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingMmprojPath;
            mmproj_path = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--image")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingImagePath;
            image_path = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--audio")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingAudioPath;
            audio_path = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--embd-out")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingEmbdPath;
            embd_out = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--bench")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingBenchReps;
            bench_reps = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.eql(u8, arg, "--threads")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingThreads;
            fucina.parallel.setMaxThreads(try std.fmt.parseInt(usize, args[arg_i], 10));
        } else if (std.mem.eql(u8, arg, "--chat")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingChatText;
            chat_text = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--system")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSystemText;
            system_text = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--repl")) {
            repl_flag = true;
        } else if (std.mem.eql(u8, arg, "--no-think")) {
            no_think = true;
        } else if (std.mem.eql(u8, arg, "--temp")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTemp;
            temp_arg = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (arg.len > 0 and (std.ascii.isDigit(arg[0]))) {
            var it = std.mem.splitScalar(u8, arg, ',');
            var count: usize = 0;
            while (it.next()) |part| {
                if (part.len == 0) continue;
                token_buf[count] = try std.fmt.parseInt(usize, part, 10);
                count += 1;
            }
            tokens = token_buf[0..count];
        } else {
            try stdout.print("unknown argument: {s}\n", .{arg});
            return error.UnknownArgument;
        }
    }

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = try fucina.gguf.File.loadMmapAuto(allocator, init.io, args[1]);

    // --tokenize FILE: encode a text file and print one token id per line
    // (the llama-tokenize parity harness); no model weights needed.
    if (tokenize_file) |path| {
        var t = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
        defer t.deinit();
        file.deinit();
        const bytes = try readTextFile(init.io, allocator, path);
        defer allocator.free(bytes);
        const ids = try t.encodeRaw(allocator, bytes);
        defer allocator.free(ids);
        for (ids) |id| try stdout.print("{d}\n", .{id});
        return;
    }

    if (info_flag) {
        try stdout.print("arch: {s}\n", .{file.getString("general.architecture") orelse "?"});
        file.deinit();
        return;
    }

    var tokenizer: ?llm.tokenizer.Tokenizer = llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{}) catch null;
    defer if (tokenizer) |*t| t.deinit();

    // Multimodal: split the prompt on <__media__>, run the tower, and build
    // mixed rows with llama.cpp mtmd's marker framing (<|content_image|>
    // before image embeddings; <|content_audio_input|> ... <|audio_end|>
    // around audio embeddings).
    var media_rows: []f32 = &.{};
    defer if (media_rows.len > 0) allocator.free(media_rows);
    var n_media_tokens: usize = 0;
    var n_embd_mm: usize = 0;
    var prompt_before: []const u8 = "";
    var prompt_after: []const u8 = "";
    const has_media = image_path != null or audio_path != null;

    if (has_media) {
        if (image_path != null and audio_path != null) return error.OneMediaOnly;
        const mm_path = mmproj_path orelse return error.MissingMmprojPath;
        const text = prompt_text orelse return error.MissingPrompt;
        const marker = "<__media__>";
        const pos = std.mem.indexOf(u8, text, marker) orelse return error.MissingMediaMarker;
        prompt_before = text[0..pos];
        prompt_after = text[pos + marker.len ..];

        var mm_file = try fucina.gguf.File.loadMmapAuto(allocator, init.io, mm_path);
        var mm = try llm.inkling.mmproj.MmProj.loadGgufFromFile(&ctx, &mm_file);
        mm_file.deinit();
        defer mm.deinit();
        n_embd_mm = mm.n_embd;

        // --bench R on a media flag: repeat preprocess+encode, report best ms.
        const media_reps = if (bench_reps > 0) bench_reps else 1;
        var best_pre_ns: i96 = std.math.maxInt(i96);
        var best_enc_ns: i96 = std.math.maxInt(i96);

        if (image_path) |ipath| {
            const bytes = try readTextFile(init.io, allocator, ipath);
            defer allocator.free(bytes);
            var img = try png.decodePng(allocator, bytes);
            defer img.deinit();
            var patch_rows: usize = 0;
            var patch_cols: usize = 0;
            for (0..media_reps) |_| {
                if (media_rows.len > 0) allocator.free(media_rows);
                const t0 = nowNs(init.io);
                var patches = try llm.inkling.mmproj.preprocessImage(allocator, &mm, img.pixels, img.width, img.height);
                defer patches.deinit();
                const t1 = nowNs(init.io);
                media_rows = try mm.visionEncode(&ctx, patches.data, patches.nPatches());
                const t2 = nowNs(init.io);
                n_media_tokens = patches.nPatches();
                patch_rows = patches.patch_rows;
                patch_cols = patches.patch_cols;
                best_pre_ns = @min(best_pre_ns, t1 - t0);
                best_enc_ns = @min(best_enc_ns, t2 - t1);
            }
            try stdout.print("image: {d}x{d} -> {d}x{d} patches = {d} tokens\n", .{ img.width, img.height, patch_rows, patch_cols, n_media_tokens });
        } else if (audio_path) |apath| {
            var audio = try llm.parakeet.frontend.loadWav16kMonoFile(allocator, init.io, apath);
            defer audio.deinit(allocator);
            for (0..media_reps) |_| {
                if (media_rows.len > 0) allocator.free(media_rows);
                const t0 = nowNs(init.io);
                var dmel = try llm.inkling.mmproj.preprocessAudio(allocator, audio.samples);
                defer dmel.deinit();
                const t1 = nowNs(init.io);
                media_rows = try mm.audioEncode(&ctx, dmel.data, dmel.n_frames);
                const t2 = nowNs(init.io);
                n_media_tokens = dmel.n_frames;
                best_pre_ns = @min(best_pre_ns, t1 - t0);
                best_enc_ns = @min(best_enc_ns, t2 - t1);
            }
            try stdout.print("audio: {d} samples -> {d} frame tokens\n", .{ audio.samples.len, n_media_tokens });
        }
        if (bench_reps > 0) {
            try stdout.print("media bench: preprocess {d:.2} ms | encode {d:.2} ms (best of {d})\n", .{ seconds(best_pre_ns) * 1e3, seconds(best_enc_ns) * 1e3, media_reps });
            return;
        }

        if (embd_out) |path| {
            try writeLogits(init.io, path, media_rows);
            try stdout.print("media embeddings written: {s} ({d} x {d})\n", .{ path, n_media_tokens, n_embd_mm });
            // Tower-only mode: nothing else requested means the decoder
            // (whose width may differ, e.g. real mmproj vs tiny decoder)
            // never runs.
            if (logits_out == null and compare_logits_path == null and gen_count == 0) return;
        }
    } else if (prompt_text) |text| {
        const t = if (tokenizer) |*t| t else return error.TokenizerUnavailable;
        const ids = try t.encodeRaw(allocator, text);
        defer allocator.free(ids);
        if (ids.len > token_buf.len) return error.PromptTooLong;
        for (ids, 0..) |id, i| token_buf[i] = id;
        tokens = token_buf[0..ids.len];
    }
    if (tokens.len == 0 and !has_media and chat_text == null and !repl_flag) return error.EmptyPrompt;

    var model = try llm.inkling.model.Model.loadGgufFromFile(&ctx, &file);
    file.deinit();
    defer model.deinit();
    if (has_media and n_embd_mm != model.config.hidden_size) return error.MmprojWidthMismatch;

    const cfg = &model.config;

    // --chat / --repl: wire-format chat through the sampler-driven engine.
    if (chat_text != null or repl_flag) {
        const t = if (tokenizer) |*t| t else return error.TokenizerUnavailable;
        try runChat(init.io, allocator, stdout, &ctx, &model, t, .{
            .first_user = chat_text,
            .system = system_text,
            .repl = repl_flag,
            .no_think = no_think,
            .max_tokens = if (gen_count > 0) gen_count else 256,
            .temperature = temp_arg,
        });
        return;
    }

    try stdout.print(
        "inkling: layers={d} hidden={d} heads={d} hd={d} experts={d}/{d}+{d}sh dense_lead={d} window={d} rel={d}/{d} K={d} vocab={d}({d})\n",
        .{ cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.num_experts_used, cfg.num_experts, cfg.num_shared_experts, cfg.dense_layers, cfg.sliding_window, cfg.rel_extent, cfg.rel_extent_swa, cfg.shortconv_kernel, cfg.vocab_size, cfg.unpadded_vocab_size },
    );

    if (has_media) {
        const t = if (tokenizer) |*t| t else return error.TokenizerUnavailable;
        // Framing text around the media rows (mtmd's img_beg / aud_beg+end).
        const before_str = try std.mem.concat(allocator, u8, &.{
            prompt_before,
            if (image_path != null) "<|content_image|>" else "<|content_audio_input|>",
        });
        defer allocator.free(before_str);
        const after_str = try std.mem.concat(allocator, u8, &.{
            if (audio_path != null) "<|audio_end|>" else "",
            prompt_after,
        });
        defer allocator.free(after_str);
        const ids_before = try t.encodeRaw(allocator, before_str);
        defer allocator.free(ids_before);
        const ids_after = try t.encodeRaw(allocator, after_str);
        defer allocator.free(ids_after);

        const Row = llm.inkling.model.Model.Row;
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

        var cache = try model.initCache(items.len + gen_count + 1);
        defer cache.deinit();
        var last = try model.stepMixed(&ctx, &cache, items);
        defer allocator.free(last);

        if (logits_out) |path| {
            try writeLogits(init.io, path, last);
            try stdout.print("logits written: {s} ({d} values)\n", .{ path, last.len });
        }
        if (compare_logits_path) |path| {
            const ok = try compareLogits(init.io, allocator, stdout, path, last, max_abs_gate);
            if (!ok) return error.LogitsMismatch;
        }
        if (gen_count > 0) {
            try stdout.print("generated ids:", .{});
            var produced: usize = 0;
            while (produced < gen_count) {
                const next = argmax(last);
                try stdout.print(" {d}", .{next});
                produced += 1;
                if (next == 200006 or produced == gen_count) break;
                const gr = try model.step(&ctx, &cache, &.{next});
                allocator.free(last);
                last = gr;
            }
            try stdout.print("\n", .{});
        }
        return;
    }

    // --bench R: warm pp/tg best-of-R, load once.
    if (bench_reps > 0) {
        const n_gen = if (gen_count > 0) gen_count else 32;
        var best_pp: f64 = 0;
        var best_tg: f64 = 0;
        for (0..bench_reps) |_| {
            var bcache = try model.initCache(tokens.len + n_gen + 1);
            defer bcache.deinit();
            const t0 = nowNs(init.io);
            var last = try model.step(&ctx, &bcache, tokens);
            const t1 = nowNs(init.io);
            // Fixed next token, no sampling.
            const next = tokens[tokens.len - 1];
            var produced: usize = 0;
            while (produced < n_gen) : (produced += 1) {
                const gr = try model.step(&ctx, &bcache, &.{next});
                allocator.free(last);
                last = gr;
            }
            const t2 = nowNs(init.io);
            allocator.free(last);
            const pp = @as(f64, @floatFromInt(tokens.len)) / seconds(t1 - t0);
            const tg = @as(f64, @floatFromInt(n_gen)) / seconds(t2 - t1);
            best_pp = @max(best_pp, pp);
            best_tg = @max(best_tg, tg);
        }
        try stdout.print("bench: pp{d} {d:.2} tok/s | tg{d} {d:.2} tok/s (best of {d})\n", .{ tokens.len, best_pp, n_gen, best_tg, bench_reps });
        return;
    }

    var cache = try model.initCache(tokens.len + gen_count + 1);
    defer cache.deinit();

    // Prefill: one batch step, or --step1 token-at-a-time (decode parity).
    var last_logits: []f32 = undefined;
    if (step1) {
        var i: usize = 0;
        var kept: ?[]f32 = null;
        while (i < tokens.len) : (i += 1) {
            const row = try model.step(&ctx, &cache, tokens[i .. i + 1]);
            if (kept) |kl| allocator.free(kl);
            kept = row;
        }
        last_logits = kept.?;
    } else {
        last_logits = try model.step(&ctx, &cache, tokens);
    }
    defer allocator.free(last_logits);

    if (logits_out) |path| {
        try writeLogits(init.io, path, last_logits);
        try stdout.print("logits written: {s} ({d} values)\n", .{ path, last_logits.len });
    }
    if (compare_logits_path) |path| {
        const ok = try compareLogits(init.io, allocator, stdout, path, last_logits, max_abs_gate);
        if (!ok) return error.LogitsMismatch;
    }

    if (gen_count > 0) {
        var decoded: std.ArrayList(u8) = .empty;
        defer decoded.deinit(allocator);
        var produced: usize = 0;
        var current = last_logits;
        try stdout.print("generated ids:", .{});
        while (produced < gen_count) {
            const next = argmax(current);
            try stdout.print(" {d}", .{next});
            if (tokenizer) |*t| try t.decodeAppend(allocator, @intCast(next), &decoded);
            produced += 1;
            if (next == 200006) break; // <|return|> — sole end-of-generation id
            if (produced == gen_count) break;
            const row = try model.step(&ctx, &cache, &.{next});
            if (current.ptr != last_logits.ptr) allocator.free(current);
            current = row;
        }
        if (current.ptr != last_logits.ptr) allocator.free(current);
        try stdout.print("\ntext: {s}\n", .{decoded.items});
    }
}

const ChatOptions = struct {
    first_user: ?[]const u8,
    system: ?[]const u8,
    repl: bool,
    no_think: bool,
    max_tokens: usize,
    temperature: f32,
};

/// Single-turn (`--chat`) or multi-turn (`--repl`) wire-format chat. The
/// engine streams a marker-wrapped reply into a buffer; this splits it into
/// the thinking block (shown dim unless --no-think) and the visible content.
fn runChat(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const llm.inkling.model.Model,
    tokenizer: *const llm.tokenizer.Tokenizer,
    opts: ChatOptions,
) !void {
    const inkling_chat = llm.inkling.chat;
    var engine = try inkling_chat.Engine(llm.tokenizer).init(ctx, model, tokenizer);

    var messages: std.ArrayList(llm.chat.Message) = .empty;
    defer messages.deinit(allocator);
    if (opts.system) |s| try messages.append(allocator, .{ .role = .system, .content = s });

    var stdin_buf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);

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

        const split = splitReply(reply.items);
        if (!opts.no_think and split.thinking.len > 0) {
            try stdout.print("\x1b[2m[thinking] {s}\x1b[0m\n", .{split.thinking});
        }
        try stdout.print("model> {s}\n", .{split.content});
        try stdout.flush();

        // Record the assistant turn for multi-turn context (content only —
        // prior reasoning is dropped, matching the reference templates).
        try messages.append(allocator, .{ .role = .assistant, .content = try allocator.dupe(u8, split.content) });
        if (!opts.repl) break;
    }
    // Free the assistant-content dupes appended above.
    for (messages.items) |m| {
        if (m.role == .assistant) allocator.free(@constCast(m.content));
    }
}

/// Split a marker-wrapped reply into its thinking block (between
/// `<|content_thinking|>` and `<|content_text|>`) and its visible content
/// (after `<|content_text|>`, with any residual markers removed).
fn splitReply(wrapped: []const u8) struct { thinking: []const u8, content: []const u8 } {
    const ct = llm.inkling.chat.tok_content_text;
    const cth = llm.inkling.chat.tok_content_thinking;
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

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
}

fn argmax(logits: []const f32) usize {
    // 8-lane SIMD scan, then resolve the winning lane; ties keep the
    // lowest index like the scalar loop.
    const V = @Vector(8, f32);
    var best: usize = 0;
    var best_v: f32 = -std.math.inf(f32);
    var i: usize = 0;
    while (i + 8 <= logits.len) : (i += 8) {
        const v: V = logits[i..][0..8].*;
        const m = @reduce(.Max, v);
        if (m > best_v) {
            for (0..8) |l| {
                if (logits[i + l] == m) {
                    best_v = m;
                    best = i + l;
                    break;
                }
            }
        }
    }
    while (i < logits.len) : (i += 1) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

fn writeLogits(io: std.Io, path: []const u8, values: []const f32) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    defer writer.interface.flush() catch {};
    try writer.interface.writeAll(std.mem.sliceAsBytes(values));
}

/// Compare against a raw f32 dump; positions where both sides are -inf
/// (the padded-vocab mask) count as equal. Returns false when max_abs
/// exceeds the gate or the argmax differs — the caller exits nonzero.
fn compareLogits(io: std.Io, allocator: std.mem.Allocator, stdout: anytype, path: []const u8, values: []const f32, max_abs_gate: f64) !bool {
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
