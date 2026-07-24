//! DeepSeek-V2 family runner (milestone A): raw greedy completion over the
//! MLA + MoE forward, sequential single-token steps. Usage:
//!   zig build deepseek2 -- models/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf \
//!     --prompt "..." --gen 64
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print("usage: zig build deepseek2 -- <model.gguf> --prompt \"...\" [--gen N]\n", .{});
        return;
    }

    var prompt_text: []const u8 = "The capital of France is";
    var gen_count: usize = 32;
    var moe_stream_flag = false;
    var moe_cache_mb: ?usize = null;
    var moe_pilot = false;
    var moe_mirror_buf: [8][]const u8 = undefined;
    var moe_mirror_n: usize = 0;
    var moe_mirror_weights_arg: ?[]const u8 = null;
    var mla_mode: llm.deepseek2.model.Cache.Mode = .latent;
    var dsa_flag = false;
    var index_probe = false;
    var index_share: usize = 0;
    var prompt_file: ?[]const u8 = null;
    var ctx_capacity: usize = 0;
    var nll_file: ?[]const u8 = null;
    var prefill_chunk: usize = 64;
    var dsa_top_k: usize = 0;
    var moe_experts: usize = 0;
    var moe_top_p: f32 = 1.0;
    var moe_skip_miss: f32 = 0;
    var arg_i: usize = 2;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--prompt")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingPrompt;
            prompt_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--prompt=")) {
            prompt_text = arg["--prompt=".len..];
        } else if (std.mem.eql(u8, arg, "--gen")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingGenCount;
            gen_count = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--gen=")) {
            gen_count = try std.fmt.parseInt(usize, arg["--gen=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--moe-stream")) {
            moe_stream_flag = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-cache-mb=")) {
            moe_stream_flag = true;
            moe_cache_mb = try std.fmt.parseInt(usize, arg["--moe-cache-mb=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--moe-pilot")) {
            moe_stream_flag = true;
            moe_pilot = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-mirror=")) {
            // Another full copy of the model, typically on another drive
            // (repeatable): expert reads split across every copy, so
            // miss-bound streaming gets each drive's bandwidth.
            moe_stream_flag = true;
            if (moe_mirror_n >= moe_mirror_buf.len) return error.TooManyMirrors;
            moe_mirror_buf[moe_mirror_n] = arg["--moe-mirror=".len..];
            moe_mirror_n += 1;
        } else if (std.mem.startsWith(u8, arg, "--moe-mirror-weights=")) {
            // Per-mirror read share relative to the primary's 1, comma
            // list in --moe-mirror order (default 1 each: even split).
            moe_mirror_weights_arg = arg["--moe-mirror-weights=".len..];
        } else if (std.mem.eql(u8, arg, "--mla=full")) {
            mla_mode = .full;
        } else if (std.mem.eql(u8, arg, "--mla=latent")) {
            mla_mode = .latent;
        } else if (std.mem.eql(u8, arg, "--dsa")) {
            dsa_flag = true;
        } else if (std.mem.eql(u8, arg, "--index-probe")) {
            dsa_flag = true;
            index_probe = true;
        } else if (std.mem.startsWith(u8, arg, "--index-share=")) {
            dsa_flag = true;
            index_share = try std.fmt.parseInt(usize, arg["--index-share=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--dsa-top-k=")) {
            // Selection-threshold override: with a smaller top-k the sparse
            // path fires within a short prompt, so DSA behavior is
            // exercisable in minutes on streamed giants. Selection
            // semantics are unchanged.
            dsa_top_k = try std.fmt.parseInt(usize, arg["--dsa-top-k=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--moe-experts=")) {
            // Inference-time truncation of the routed-expert count: route
            // with a smaller top-k so the dropped experts are never fetched
            // (the direct bytes-per-token lever on streamed giants). Gate
            // weights renormalize over the smaller set as usual.
            moe_experts = try std.fmt.parseInt(usize, arg["--moe-experts=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--moe-top-p=")) {
            // Dynamic dial 1: keep routed experts covering this fraction of
            // the gate mass (deterministic; confident tokens drop the tail).
            moe_top_p = try std.fmt.parseFloat(f32, arg["--moe-top-p=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--moe-skip-miss=")) {
            // Dynamic dial 2: drop sub-threshold-weight experts ONLY when
            // they would cost a disk read (cache-state dependent output).
            moe_skip_miss = try std.fmt.parseFloat(f32, arg["--moe-skip-miss=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--prompt-file=")) {
            prompt_file = arg["--prompt-file=".len..];
        } else if (std.mem.startsWith(u8, arg, "--ctx=")) {
            ctx_capacity = try std.fmt.parseInt(usize, arg["--ctx=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--nll-file=")) {
            nll_file = arg["--nll-file=".len..];
        } else if (std.mem.startsWith(u8, arg, "--prefill-chunk=")) {
            prefill_chunk = try std.fmt.parseInt(usize, arg["--prefill-chunk=".len..], 10);
            if (prefill_chunk == 0) prefill_chunk = 1;
        } else {
            try stdout.print("unknown flag: {s}\n", .{arg});
            return error.UnknownArgument;
        }
    }

    const allocator = std.heap.smp_allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const load_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    var file = try fucina.gguf.File.loadMmapAuto(allocator, init.io, args[1]);
    var tokenizer = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tokenizer.deinit();

    if (index_probe and index_share >= 2) {
        try stdout.print("--index-probe measures the exact path; drop --index-share\n", .{});
        return error.UnknownArgument;
    }
    var moe_mirror_weights_buf: [8]f32 = undefined;
    const moe_mirror_weights = try llm.weights.parseMirrorWeights(moe_mirror_weights_arg, moe_mirror_n, &moe_mirror_weights_buf);
    var load_options: llm.deepseek2.model.Model.LoadOptions = if (moe_stream_flag) .{
        .moe_stream = .{
            .gguf_path = args[1],
            .cache_bytes = if (moe_cache_mb) |mb| mb << 20 else null,
            .pilot = moe_pilot,
            .mirror_paths = moe_mirror_buf[0..moe_mirror_n],
            .mirror_weights = moe_mirror_weights,
        },
    } else .{};
    load_options.dsa = dsa_flag;
    var model = try llm.deepseek2.model.Model.loadGgufFromFileOptions(&ctx, &file, load_options);
    defer model.deinit();
    // The stats go through the SAME buffered stdout writer as everything
    // else: stdout's positional writes and stderr's offset-advancing writes
    // cannot safely share one redirected file (`cmd > f 2>&1` interleaves
    // destructively), so a std.debug stats line would get overwritten.
    defer if (model.expert_store) |store| llm.weights.reportAndSaveMoeStream(store, true, stdout);
    const bos: ?u32 = tokenizer.bosId();
    file.deinit();
    try stdout.print("load: {d:.3} s\n", .{@as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - load_start)) / 1e9});

    var prompt_file_bytes: ?[]u8 = null;
    defer if (prompt_file_bytes) |b| allocator.free(b);
    if (prompt_file) |path| {
        prompt_file_bytes = try readAllFile(init.io, allocator, path);
        prompt_text = prompt_file_bytes.?;
    }

    // Encode: deepseek2 adds BOS.
    const ids32 = try tokenizer.encode(allocator, prompt_text);
    defer allocator.free(ids32);
    var tokens: std.ArrayList(usize) = .empty;
    defer tokens.deinit(allocator);
    // GLM checkpoints (glm-dsa) open with [gMASK]<sop> instead of BOS —
    // resolved by string so vocab ids stay model-defined; GLM trunks
    // degenerate without them.
    if (tokenizer.tokenId("[gMASK]")) |gmask| {
        try tokens.append(allocator, gmask);
        if (tokenizer.tokenId("<sop>")) |sop| try tokens.append(allocator, sop);
    } else if (bos) |b| try tokens.append(allocator, b);
    for (ids32) |id| try tokens.append(allocator, id);
    try stdout.print("prompt tokens: {d}\n", .{tokens.items.len});

    model.index_share_every = index_share;
    if (dsa_top_k > 0) model.config.indexer_top_k = dsa_top_k;
    if (moe_experts > 0) {
        if (moe_experts > model.config.num_experts_used) return error.InvalidExpertCount;
        try stdout.print("moe: experts used {d} -> {d} (inference-time truncation)\n", .{ model.config.num_experts_used, moe_experts });
        model.config.num_experts_used = moe_experts;
    }
    if (moe_top_p < 1.0 or moe_skip_miss > 0) {
        model.moe_top_p = moe_top_p;
        model.moe_skip_miss_below = moe_skip_miss;
        try stdout.print("moe: dynamic expert drop (top-p {d:.2}, skip-miss-below {d:.3})\n", .{ moe_top_p, moe_skip_miss });
    }

    // Teacher-forced NLL over a text file (the dense-vs-DSA quality gate):
    // step through the tokens, accumulate -log p(next). Uses its own cache;
    // prints and exits.
    if (nll_file) |path| {
        const text = try readAllFile(init.io, allocator, path);
        defer allocator.free(text);
        const nll_ids = try tokenizer.encode(allocator, text);
        defer allocator.free(nll_ids);
        var nll_cache = try model.initCacheMode(@max(2048, nll_ids.len + 4), mla_mode);
        defer nll_cache.deinit();
        var total: f64 = 0;
        var count: usize = 0;
        if (bos) |b| {
            const l0 = try model.step(&ctx, &nll_cache, b);
            allocator.free(l0);
        }
        var prev: usize = @intCast(nll_ids[0]);
        for (nll_ids[1..]) |next_id| {
            const lg = try model.step(&ctx, &nll_cache, prev);
            defer allocator.free(lg);
            var maxv: f32 = lg[0];
            for (lg) |v| maxv = @max(maxv, v);
            var sum_exp: f64 = 0;
            for (lg) |v| sum_exp += @exp(@as(f64, v - maxv));
            total += @as(f64, maxv) + @log(sum_exp) - @as(f64, lg[next_id]);
            count += 1;
            prev = @intCast(next_id);
        }
        try stdout.print("nll: {d:.4} (ppl {d:.2}) over {d} tokens{s}\n", .{ total / @as(f64, @floatFromInt(count)), @exp(total / @as(f64, @floatFromInt(count))), count, if (dsa_flag) " [dsa]" else " [dense]" });
        return;
    }

    const capacity: usize = if (ctx_capacity > 0) ctx_capacity else @max(2048, tokens.items.len + gen_count + 8);
    var cache = try model.initCacheMode(capacity, mla_mode);
    defer cache.deinit();
    if (index_probe) try cache.enableDsaProbe(model.config.num_layers);

    // Prefill: chunked batches through stepBatch (S-row projections +
    // union-routed expert fetches); --prefill-chunk=1 restores the
    // sequential S=1 path.
    const prefill_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    var logits: []f32 = &.{};
    defer if (logits.len > 0) allocator.free(logits);
    var fed: usize = 0;
    while (fed < tokens.items.len) {
        const end = @min(fed + prefill_chunk, tokens.items.len);
        if (logits.len > 0) allocator.free(logits);
        logits = try model.stepBatch(&ctx, &cache, tokens.items[fed..end]);
        fed = end;
    }
    try stdout.print("prefill: {d:.1} ms ({d} tokens, chunk {d})\n", .{ @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - prefill_start)) / 1e6, tokens.items.len, prefill_chunk });

    // Greedy decode.
    const eos = tokenizer.eosId();
    var reply: std.ArrayList(u8) = .empty;
    defer reply.deinit(allocator);
    const decode_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    var produced: usize = 0;
    while (produced < gen_count) : (produced += 1) {
        var best: usize = 0;
        var best_v: f32 = -std.math.inf(f32);
        for (logits, 0..) |v, i| {
            if (v > best_v) {
                best_v = v;
                best = i;
            }
        }
        if (eos != null and best == eos.?) break;
        try tokenizer.decodeAppend(allocator, @intCast(best), &reply);
        try tokens.append(allocator, best);
        allocator.free(logits);
        logits = try model.step(&ctx, &cache, best);
    }
    const decode_ns = std.Io.Clock.awake.now(init.io).nanoseconds - decode_start;
    try stdout.print("decode: {d} steps, {d:.1} ms, {d:.2} tok/s\n", .{ produced, @as(f64, @floatFromInt(decode_ns)) / 1e6, @as(f64, @floatFromInt(produced)) * 1e9 / @as(f64, @floatFromInt(decode_ns)) });
    if (model.index_share_every >= 2) {
        try stdout.print("index-share: every {d} — selections computed {d}, reused {d}\n", .{ model.index_share_every, cache.share_computed, cache.share_reused });
    }
    if (cache.probe) |*p| try p.report(stdout);
    if (prompt_file != null) {
        try stdout.print("prompt: ({d} bytes from file)\ntext:  {s}\n", .{ prompt_text.len, reply.items });
    } else {
        try stdout.print("prompt: {s}\ntext:  {s}\n", .{ prompt_text, reply.items });
    }
}

fn readAllFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.IsDir;
    if (stat.size > 16 * 1024 * 1024) return error.FileTooLarge;
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
