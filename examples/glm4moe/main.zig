//! GLM-4.5 family runner: greedy completion with optional native MTP
//! (multi-token prediction) speculative decoding — the model's own nextn
//! layer drafts tokens, one batched trunk step verifies them, and only
//! greedy-matching prefixes commit (lossless).
//!   zig build glm4moe -- <model-part1.gguf> --prompt "..." --gen 64 \
//!     [--mtp[=depth]] [--moe-stream --moe-cache-mb=N]
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
        try stdout.print("usage: zig build glm4moe -- <model.gguf> --prompt \"...\" [--gen N] [--mtp[=depth]] [--moe-stream --moe-cache-mb=N]\n", .{});
        return;
    }

    var prompt_text: []const u8 = "The capital of France is";
    var gen_count: usize = 32;
    var moe_stream_flag = false;
    var moe_cache_mb: ?usize = null;
    var moe_mirror_buf: [8][]const u8 = undefined;
    var moe_mirror_n: usize = 0;
    var moe_mirror_weights_arg: ?[]const u8 = null;
    var moe_io_threads: ?usize = null;
    var mtp_depth: usize = 0; // 0 = plain decode
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
        } else if (std.mem.eql(u8, arg, "--mtp")) {
            mtp_depth = 2;
        } else if (std.mem.startsWith(u8, arg, "--mtp=")) {
            // Depth caps at 8: the verify runs kernel-pinned
            // (ExecContext.pinRowwiseKernels), so the m >= 4 x4-packed
            // quant kernels no longer drift from the S=1 numerics — the
            // old depth-2 wall. 8 keeps the verify batch (depth+1 rows)
            // under the remaining non-quant thresholds (f32/f16 fused-FFN
            // at m >= 12, tiled attention at seq >= 48).
            mtp_depth = @min(try std.fmt.parseInt(usize, arg["--mtp=".len..], 10), 8);
        } else if (std.mem.eql(u8, arg, "--moe-stream")) {
            moe_stream_flag = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-cache-mb=")) {
            moe_stream_flag = true;
            moe_cache_mb = try std.fmt.parseInt(usize, arg["--moe-cache-mb=".len..], 10);
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
        } else if (std.mem.startsWith(u8, arg, "--moe-io-threads=")) {
            // Demand-miss read fan-out (default 8; 0 = sequential reads).
            moe_stream_flag = true;
            moe_io_threads = try std.fmt.parseInt(usize, arg["--moe-io-threads=".len..], 10);
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

    const capacity: usize = 2048;
    var moe_mirror_weights_buf: [8]f32 = undefined;
    const moe_mirror_weights = try llm.weights.parseMirrorWeights(moe_mirror_weights_arg, moe_mirror_n, &moe_mirror_weights_buf);
    const load_options: llm.glm4moe.model.Model.LoadOptions = if (moe_stream_flag) .{
        .moe_stream = .{
            .gguf_path = args[1],
            .cache_bytes = if (moe_cache_mb) |mb| mb << 20 else null,
            .mirror_paths = moe_mirror_buf[0..moe_mirror_n],
            .mirror_weights = moe_mirror_weights,
            .io_workers = moe_io_threads orelse 8,
        },
    } else .{};
    var model = try llm.glm4moe.model.Model.loadGgufFromFileOptions(&ctx, &file, capacity, load_options);
    defer model.deinit();
    // The stats go through the SAME buffered stdout writer as everything
    // else: stdout's positional writes and stderr's offset-advancing writes
    // cannot safely share one redirected file (`cmd > f 2>&1` interleaves
    // destructively), so a std.debug stats line would get overwritten.
    defer if (model.expert_store) |store| llm.weights.reportAndSaveMoeStream(store, true, stdout);
    const bos: ?u32 = tokenizer.bosId();
    const eos = tokenizer.eosId();
    file.deinit();
    try stdout.print("load: {d:.3} s\n", .{@as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - load_start)) / 1e9});
    if (mtp_depth > 0 and model.mtp == null) {
        try stdout.print("model has no nextn (MTP) layer; --mtp ignored\n", .{});
        mtp_depth = 0;
    }

    const ids32 = try tokenizer.encode(allocator, prompt_text);
    defer allocator.free(ids32);
    var tokens: std.ArrayList(usize) = .empty;
    defer tokens.deinit(allocator);
    if (bos) |b| try tokens.append(allocator, b);
    // GLM canonical opening: [gMASK]<sop> before content.
    if (bos != null and bos.? == 151331) try tokens.append(allocator, 151333);
    for (ids32) |id| try tokens.append(allocator, id);
    try stdout.print("prompt tokens: {d}\n", .{tokens.items.len});

    var cache = try model.initCache(capacity);
    defer cache.deinit();
    var mtp_cache = try model.initMtpCache(capacity);
    defer mtp_cache.deinit();

    const hidden = model.config.hidden_size;
    // Trunk hiddens for every committed position (the MTP stream input).
    var hiddens: std.ArrayList(f32) = .empty;
    defer hiddens.deinit(allocator);
    var mtp_fed: usize = 0;

    const freeRows = struct {
        fn go(a: std.mem.Allocator, rows: [][]f32) void {
            for (rows) |r| a.free(r);
            a.free(rows);
        }
    }.go;

    // Prefill (one batched step) and the first greedy token.
    const prefill_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    var next_token: usize = undefined;
    {
        const rows = try model.step(&ctx, &cache, tokens.items);
        defer freeRows(allocator, rows);
        try hiddens.appendSlice(allocator, model.step_hiddens);
        next_token = argmax(rows[rows.len - 1]);
    }
    try stdout.print("prefill: {d:.1} ms ({d} tokens, one batched step)\n", .{ @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - prefill_start)) / 1e6, tokens.items.len });

    var reply: std.ArrayList(u8) = .empty;
    defer reply.deinit(allocator);
    var produced: usize = 0;
    var forwards: usize = 0;
    var drafted: usize = 0;
    var draft_accepted: usize = 0;
    var feed_hits: usize = 0;
    var feed_total: usize = 0;
    const h_scratch = try allocator.alloc(f32, hidden);
    defer allocator.free(h_scratch);
    const h_prev = try allocator.alloc(f32, hidden);
    defer allocator.free(h_prev);

    const decode_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    decode: while (produced < gen_count) {
        if (eos != null and next_token == eos.?) break;

        if (mtp_depth == 0) {
            try tokenizer.decodeAppend(allocator, @intCast(next_token), &reply);
            try tokens.append(allocator, next_token);
            produced += 1;
            var one = [_]usize{next_token};
            const rows = try model.step(&ctx, &cache, &one);
            defer freeRows(allocator, rows);
            try hiddens.appendSlice(allocator, model.step_hiddens);
            forwards += 1;
            next_token = argmax(rows[0]);
            continue;
        }

        // ---- MTP round ----
        // Catch the MTP stream up on committed positions: position i
        // consumes (token[i+1], trunk h[i]).
        const n = tokens.items.len;
        while (mtp_fed + 1 < n) : (mtp_fed += 1) {
            const logits = try model.mtpDraftStep(&ctx, &mtp_cache, tokens.items[mtp_fed + 1], hiddens.items[mtp_fed * hidden ..][0..hidden], h_scratch);
            defer allocator.free(logits);
            // Diagnostic: the MTP head's next-next-token hit rate on KNOWN
            // history — separates a broken MTP forward (near 0%) from a
            // broken draft/verify loop (healthy 30-60% here).
            if (mtp_fed + 2 < n) {
                const got = argmax(logits);
                if (got == tokens.items[mtp_fed + 2]) feed_hits += 1;
                feed_total += 1;
            }
        }

        // Draft chain from the frontier: (next_token, h[n-1]) then the MTP
        // layer's own hidden recurrence.
        var drafts_buf: [17]usize = undefined;
        drafts_buf[0] = next_token;
        var n_drafts: usize = 1;
        @memcpy(h_prev, hiddens.items[(n - 1) * hidden ..][0..hidden]);
        while (n_drafts <= mtp_depth) : (n_drafts += 1) {
            const logits = try model.mtpDraftStep(&ctx, &mtp_cache, drafts_buf[n_drafts - 1], h_prev, h_scratch);
            defer allocator.free(logits);
            drafts_buf[n_drafts] = argmax(logits);
            @memcpy(h_prev, h_scratch);
        }
        mtp_cache.truncate(mtp_fed); // drop the speculative MTP positions
        const drafts = drafts_buf[0..n_drafts];
        drafted += n_drafts - 1;

        // One batched trunk verify over the whole draft, kernel-pinned so
        // its logits are bit-identical to sequential decode at any depth
        // (the lossless contract; see ExecContext.pinRowwiseKernels).
        ctx.pinRowwiseKernels(true);
        const rows = blk: {
            defer ctx.pinRowwiseKernels(false);
            break :blk try model.step(&ctx, &cache, drafts);
        };
        defer freeRows(allocator, rows);
        forwards += 1;
        var accepted: usize = 1;
        while (accepted < n_drafts) : (accepted += 1) {
            if (argmax(rows[accepted - 1]) != drafts[accepted]) break;
        }
        draft_accepted += accepted - 1;

        // Commit the accepted prefix (+ trunk hiddens), rewind the rest.
        for (drafts[0..accepted]) |t| {
            if (produced == gen_count) break :decode;
            try tokenizer.decodeAppend(allocator, @intCast(t), &reply);
            try tokens.append(allocator, t);
            produced += 1;
            if (eos != null and t == eos.?) break :decode;
        }
        try hiddens.appendSlice(allocator, model.step_hiddens[0 .. accepted * hidden]);
        cache.truncate(tokens.items.len);
        next_token = argmax(rows[accepted - 1]);
    }
    const decode_ns = std.Io.Clock.awake.now(init.io).nanoseconds - decode_start;
    try stdout.print("decode: {d} tokens in {d} forwards, {d:.1} ms, {d:.2} tok/s ({d:.2} tok/forward)\n", .{ produced, forwards, @as(f64, @floatFromInt(decode_ns)) / 1e6, @as(f64, @floatFromInt(produced)) * 1e9 / @as(f64, @floatFromInt(decode_ns)), @as(f64, @floatFromInt(produced)) / @as(f64, @floatFromInt(@max(forwards, 1))) });
    if (mtp_depth > 0 and drafted > 0) {
        try stdout.print("mtp: {d}/{d} drafts accepted ({d:.1}%), feed hit {d}/{d}\n", .{ draft_accepted, drafted, @as(f64, @floatFromInt(draft_accepted)) * 100.0 / @as(f64, @floatFromInt(drafted)), feed_hits, feed_total });
    }
    try stdout.print("generated ids:", .{});
    for (tokens.items[tokens.items.len - produced ..]) |t| try stdout.print(" {d}", .{t});
    try stdout.print("\nprompt: {s}\ntext:  {s}\n", .{ prompt_text, reply.items });
}

fn argmax(logits: []const f32) usize {
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
