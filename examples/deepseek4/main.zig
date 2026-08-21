//! DeepSeek V4 Flash runner: greedy completion over the CSA/HCA trunk with
//! streamed experts. Validation against the official API vectors comes via
//! --vectors (chat-rendered prompts + greedy token comparison).
//!
//! Weights: huggingface.co/antirez/deepseek-v4-gguf (mixed-precision
//! single-GGUF exports of DeepSeek-V4-Flash, arch tag `deepseek4`). The
//! port follows Salvatore Sanfilippo's ds4 reference implementation
//! (github.com/antirez/ds4, MIT), which is also the parity oracle —
//! docs/THIRD-PARTY-NOTICES.md records the lineage.
//!   zig build deepseek4 -- gguf/model.gguf --prompt "..." --gen 32 \
//!     --moe-stream --moe-cache-mb=20480
//!   zig build deepseek4 -- gguf/model.gguf --moe-stream \
//!     --vectors=path/to/ds4/tests/test-vectors/official
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const Model = llm.deepseek4.model.Model;
const Session = llm.deepseek4.model.Session;
const Tokenizer = llm.tokenizer.Tokenizer;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print("usage: zig build deepseek4 -- <model.gguf> --prompt \"...\" [--prompt-file=PATH] [--gen N] [--chat] [--moe-stream --moe-cache-mb=N --moe-pilot --moe-pin-mb=N --moe-no-learn --moe-cache-slots=N] [--index-probe | --index-share=N] [--vectors=DIR [--vectors-max-prompt=N]]\n", .{});
        return;
    }

    var prompt_text: []const u8 = "The capital of France is";
    var gen_count: usize = 16;
    var moe_cli: fucina.weights.MoeStreamCli = .{};
    var moe_cache_route = false;
    var moe_route_j: usize = 2;
    var moe_route_m: usize = 12;
    var moe_pilot = false;
    var moe_pin_mb: ?usize = null;
    var moe_no_learn = false;
    var moe_cache_slots: ?usize = null;
    var moe_expert_top_p: f32 = 1.0;
    var temp_arg: f32 = 0;
    var topp_arg: f32 = 1.0;
    var topk_arg: usize = 0;
    var minp_arg: f32 = 0;
    var penalty_arg: f32 = 1.0;
    var seed_arg: u64 = 0;
    var chat = false;
    var prefill_chunk: usize = 128;
    var mtp_path: ?[]const u8 = null;
    var mtp_depth: usize = 1;
    var spec = false;
    var vectors_dir: ?[]const u8 = null;
    var golden_path: ?[]const u8 = null;
    var nll_path: ?[]const u8 = null;
    var nll_tokens: usize = 512;
    var vectors_max_prompt: usize = 256;
    var index_share: usize = 0;
    var index_probe = false;
    var prompt_file: ?[]const u8 = null;
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
        } else if (std.mem.eql(u8, arg, "--chat")) {
            chat = true;
        } else if (std.mem.eql(u8, arg, "--spec")) {
            // Draft-model-free speculative decoding: the shared cascade
            // (conversation SAM + token recycling) drafts, the trunk
            // verifies in one batched, kernel-pinned step — lossless, and
            // the dense trunk's per-forward weight streaming amortizes
            // over every accepted token.
            spec = true;
        } else if (std.mem.startsWith(u8, arg, "--mtp=")) {
            mtp_path = arg["--mtp=".len..];
        } else if (std.mem.startsWith(u8, arg, "--mtp-depth=")) {
            mtp_depth = @min(try std.fmt.parseInt(usize, arg["--mtp-depth=".len..], 10), 8);
            if (mtp_depth == 0) mtp_depth = 1;
        } else if (std.mem.startsWith(u8, arg, "--prefill-chunk=")) {
            prefill_chunk = try std.fmt.parseInt(usize, arg["--prefill-chunk=".len..], 10);
            if (prefill_chunk == 0) prefill_chunk = 1;
        } else if (std.mem.startsWith(u8, arg, "--golden=")) {
            golden_path = arg["--golden=".len..];
        } else if (std.mem.startsWith(u8, arg, "--vectors=")) {
            vectors_dir = arg["--vectors=".len..];
        } else if (std.mem.startsWith(u8, arg, "--nll=")) {
            nll_path = arg["--nll=".len..];
        } else if (std.mem.startsWith(u8, arg, "--nll-tokens=")) {
            nll_tokens = try std.fmt.parseInt(usize, arg["--nll-tokens=".len..], 10);
            if (nll_tokens == 0) nll_tokens = 1;
        } else if (std.mem.startsWith(u8, arg, "--vectors-max-prompt=")) {
            vectors_max_prompt = try std.fmt.parseInt(usize, arg["--vectors-max-prompt=".len..], 10);
        } else if (try moe_cli.tryParse(arg)) {
            // Shared streamed-experts flags (fucina.weights.MoeStreamCli).
        } else if (std.mem.eql(u8, arg, "--moe-cache-route")) {
            // Cache-aware near-tie routing (QUALITY-AFFECTING, opt-in):
            // prefer already-resident experts among the top-M ranks.
            moe_cli.armed = true;
            moe_cache_route = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-route-j=")) {
            // Sacred true-top ranks always taken (default 2).
            moe_route_j = try std.fmt.parseInt(usize, arg["--moe-route-j=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--moe-route-m=")) {
            // Max-rank window for the resident-preferring fill (default 12).
            moe_route_m = try std.fmt.parseInt(usize, arg["--moe-route-m=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--moe-pilot")) {
            // Router lookahead: predict each next layer's experts and stage
            // them from a background I/O thread. Never changes output.
            moe_cli.armed = true;
            moe_pilot = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-pin-mb=")) {
            moe_cli.armed = true;
            moe_pin_mb = try std.fmt.parseInt(usize, arg["--moe-pin-mb=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--moe-no-learn")) {
            moe_cli.armed = true;
            moe_no_learn = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-cache-slots=")) {
            moe_cli.armed = true;
            moe_cache_slots = try std.fmt.parseInt(usize, arg["--moe-cache-slots=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--moe-expert-top-p=")) {
            // Routing sparsification (QUALITY-AFFECTING, opt-in): keep
            // experts per token up to cumulative router weight F.
            moe_expert_top_p = try std.fmt.parseFloat(f32, arg["--moe-expert-top-p=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--index-share=")) {
            index_share = try std.fmt.parseInt(usize, arg["--index-share=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--index-probe")) {
            index_probe = true;
        } else if (std.mem.startsWith(u8, arg, "--prompt-file=")) {
            prompt_file = arg["--prompt-file=".len..];
        } else if (std.mem.startsWith(u8, arg, "--temp=")) {
            // The 0731 release is tuned for temperature 1.0 sampling
            // (stamped in the GGUF as general.sampling.temp); greedy
            // (default 0) stays the deterministic oracle/benchmark path.
            temp_arg = try std.fmt.parseFloat(f32, arg["--temp=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--top-p=")) {
            topp_arg = try std.fmt.parseFloat(f32, arg["--top-p=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--top-k=")) {
            topk_arg = try std.fmt.parseInt(usize, arg["--top-k=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--min-p=")) {
            minp_arg = try std.fmt.parseFloat(f32, arg["--min-p=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--repeat-penalty=")) {
            penalty_arg = try std.fmt.parseFloat(f32, arg["--repeat-penalty=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--seed=")) {
            seed_arg = try std.fmt.parseInt(u64, arg["--seed=".len..], 10);
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
    var tokenizer = try Tokenizer.initFromGguf(allocator, &file, .{});
    defer tokenizer.deinit();

    var moe_stream = try moe_cli.options(args[1]);
    if (moe_stream) |*m| {
        m.cache_route = moe_cache_route;
        m.route_sacred = moe_route_j;
        m.route_window = moe_route_m;
        m.pilot = moe_pilot;
        if (moe_pin_mb) |mb| m.pin_bytes = mb << 20;
        if (moe_no_learn) m.auto_pin = false;
        if (moe_cache_slots) |n| m.cache_slots_per_layer = n;
    }
    const load_options: Model.LoadOptions = if (moe_stream) |m| .{ .moe_stream = m } else .{};
    var model = try Model.loadGgufFromFileOptions(&ctx, &file, load_options);
    defer model.deinit();
    // Cross-layer indexer reuse (docs: Model.index_share_every). The probe
    // measures the exact path, so the two are mutually exclusive.
    if (index_share >= 2 and index_probe) {
        try stdout.print("--index-probe measures the exact path; drop --index-share\n", .{});
        return error.UnknownArgument;
    }
    model.index_share_every = index_share;
    model.moe_expert_top_p = moe_expert_top_p;
    // The stats go through the SAME buffered stdout writer as everything
    // else: stdout's positional writes and stderr's offset-advancing writes
    // cannot safely share one redirected file (`cmd > f 2>&1` interleaves
    // destructively), so a std.debug stats line would get overwritten.
    defer if (model.expert_store) |store| fucina.weights.reportAndSaveMoeStream(store, !moe_no_learn, stdout);
    file.deinit();
    try stdout.print("load: {d:.3} s\n", .{@as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - load_start)) / 1e9});

    if (golden_path) |vec_path| {
        return runGolden(init.io, allocator, &ctx, &model, &tokenizer, stdout, vec_path, prefill_chunk);
    }
    if (vectors_dir) |dir_path| {
        return runVectors(init.io, allocator, &ctx, &model, &tokenizer, stdout, dir_path, vectors_max_prompt, prefill_chunk);
    }
    if (nll_path) |path| {
        return runNll(init.io, allocator, &ctx, &model, &tokenizer, stdout, path, nll_tokens, prefill_chunk);
    }

    var prompt_file_bytes: ?[]u8 = null;
    defer if (prompt_file_bytes) |b| allocator.free(b);
    if (prompt_file) |path| {
        prompt_file_bytes = try readAllFile(init.io, allocator, path);
        prompt_text = prompt_file_bytes.?;
    }

    const eos = tokenizer.eosId();
    var tokens: std.ArrayList(usize) = .empty;
    defer tokens.deinit(allocator);
    if (chat) {
        try renderChat(allocator, &tokenizer, prompt_text, &tokens);
    } else {
        if (tokenizer.bosId()) |b| try tokens.append(allocator, b);
        const ids32 = try tokenizer.encode(allocator, prompt_text);
        defer allocator.free(ids32);
        for (ids32) |id| try tokens.append(allocator, id);
    }
    try stdout.print("prompt tokens: {d}\n", .{tokens.items.len});

    var session = try Session.init(&model, 8192);
    defer session.deinit(&model);
    if (index_probe) try session.enableIndexProbe(&model);

    var mtp: ?llm.deepseek4.model.Mtp = if (mtp_path) |mp| try llm.deepseek4.model.Mtp.loadGguf(&ctx, init.io, mp, model.config) else null;
    defer if (mtp) |*m| m.deinit();
    if (mtp != null and temp_arg > 0) {
        // MTP verify is exact only against greedy: acceptance compares the
        // draft to the trunk argmax. Lossless speculative SAMPLING needs
        // rejection sampling — not wired yet.
        try stdout.print("--mtp is greedy-only; drop --temp or --mtp\n", .{});
        return error.UnknownArgument;
    }
    if (spec and (mtp != null or temp_arg > 0)) {
        try stdout.print("--spec is greedy-only and excludes --mtp\n", .{});
        return error.UnknownArgument;
    }

    const hc_dim = model.config.n_hc * model.config.hidden_size;
    const frontier = try allocator.alloc(f32, hc_dim);
    defer allocator.free(frontier);

    const prefill_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    // Logits are session-owned (valid until the next step) — never freed.
    var logits: []f32 = &.{};
    {
        var fed: usize = 0;
        while (fed < tokens.items.len) {
            const end = @min(fed + prefill_chunk, tokens.items.len);
            // Final-row-only stream request: exactly the frontier the drafter
            // needs, and it keeps the model's final-layer FFN truncation
            // active during prefill.
            logits = try llm.deepseek4.model.stepBatchExtra(&model, &ctx, &session, tokens.items[fed..end], null, .{ .final = frontier });
            fed = end;
        }
    }
    try stdout.print("prefill: {d:.1} ms ({d} tokens, chunk {d})\n", .{ @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - prefill_start)) / 1e6, tokens.items.len, prefill_chunk });

    var reply: std.ArrayList(u8) = .empty;
    defer reply.deinit(allocator);
    const decode_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    var produced: usize = 0;
    var forwards: usize = 0;
    var drafted: usize = 0;
    var draft_hits: usize = 0;

    if (mtp) |*m| {
        var state = try llm.deepseek4.model.MtpState.init(allocator, model.config);
        defer state.deinit(allocator);
        var next_token = argmax(logits);
        const streams_all = try allocator.alloc(f32, (mtp_depth + 1) * hc_dim);
        defer allocator.free(streams_all);
        const rows = try allocator.alloc([]f32, mtp_depth + 1);
        defer allocator.free(rows);

        decode: while (produced < gen_count) {
            if (eos != null and next_token == eos.?) break;
            // Draft a suffix from the frontier stream state.
            var draft_buf: [9]usize = undefined;
            draft_buf[0] = next_token;
            var n_drafts: usize = 1;
            const saved_rows = state.n_rows;
            var seed: []const f32 = frontier;
            while (n_drafts <= mtp_depth) : (n_drafts += 1) {
                const dl = try llm.deepseek4.model.mtpDraftStep(&model, m, &ctx, &state, draft_buf[n_drafts - 1], seed, session.cache.len + n_drafts - 1);
                defer allocator.free(dl);
                draft_buf[n_drafts] = argmax(dl);
                seed = state.streams;
                drafted += 1;
            }
            // Verify with one batched trunk step; rewind on partial accept.
            // Kernel-pinned (ExecContext.pinRowwiseKernels): the verify
            // logits AND the cache rows it leaves behind for accepted
            // positions are bit-identical to sequential decode at any
            // depth, which is what keeps --mtp output byte-identical to
            // plain greedy.
            var snap = try session.cache.snapshot();
            defer snap.deinit();
            ctx.pinRowwiseKernels(true);
            _ = blk: {
                defer ctx.pinRowwiseKernels(false);
                break :blk try llm.deepseek4.model.stepBatchExtra(&model, &ctx, &session, draft_buf[0..n_drafts], rows[0..n_drafts], .{ .all = streams_all[0 .. n_drafts * hc_dim] });
            };
            forwards += 1;
            var accepted: usize = 1;
            while (accepted < n_drafts) : (accepted += 1) {
                if (argmax(rows[accepted - 1]) != draft_buf[accepted]) break;
            }
            draft_hits += accepted - 1;

            for (draft_buf[0..accepted]) |t| {
                if (produced == gen_count) break;
                try tokenizer.decodeAppend(allocator, @intCast(t), &reply);
                produced += 1;
                if (eos != null and t == eos.?) {
                    for (rows[0..n_drafts]) |r| allocator.free(r);
                    break :decode;
                }
            }
            next_token = argmax(rows[accepted - 1]);
            @memcpy(frontier, streams_all[(accepted - 1) * hc_dim ..][0..hc_dim]);
            state.n_rows = @min(saved_rows + accepted, model.config.n_swa);
            for (rows[0..n_drafts]) |r| allocator.free(r);
            if (accepted < n_drafts) {
                // The verify advanced past the accepted prefix: restore and
                // replay only the accepted tokens to rebuild the caches —
                // pinned too, so the rebuilt rows match sequential decode.
                session.cache.restore(&snap);
                ctx.pinRowwiseKernels(true);
                // Rows-less stepBatchExtra returns the SESSION-OWNED logits
                // slice — discarded, never freed (freeing it poisons the
                // session's buffer and the next step segfaults).
                _ = blk: {
                    defer ctx.pinRowwiseKernels(false);
                    break :blk try llm.deepseek4.model.stepBatchExtra(&model, &ctx, &session, draft_buf[0..accepted], null, null);
                };
                forwards += 1;
            }
        }
    } else if (spec) {
        // Draft-model-free speculation: the shared cascade drafts from the
        // committed stream (conversation SAM + token recycling), the trunk
        // verifies in one batched, kernel-pinned step — same losslessness
        // contract as --mtp (byte-identical to plain greedy), no sidecar,
        // no drafter I/O, and the dense trunk stream amortizes over every
        // accepted token.
        var index = try llm.speculative.cascade.SpeculationIndex.init(allocator, model.config.vocab_size);
        defer index.deinit();
        index.accounting_min_draft = 1;
        const source = index.asDraftSource();
        const wants_topk = source.wantsTopK();
        var history: std.ArrayList(usize) = .empty;
        defer history.deinit(allocator);
        try history.appendSlice(allocator, tokens.items);
        index.observe(tokens.items);

        const rows = try allocator.alloc([]f32, 9);
        defer allocator.free(rows);
        var topk_ids: [9][8]u32 = undefined;
        var topk_rows: [9]llm.speculative.core.TopKRow = undefined;
        var next_token = argmax(logits);

        decode: while (produced < gen_count) {
            if (eos != null and next_token == eos.?) break;
            var draft_buf: [9]usize = undefined;
            draft_buf[0] = next_token;
            try history.append(allocator, next_token);
            // The committed stream must reach the index token-for-token, in
            // order: the anchor now, the accepted drafts after the verify.
            index.observe(history.items[history.items.len - 1 ..]);
            const prev_committed = history.items.len;
            const n_drafts = 1 + @min(source.suggest(history.items, draft_buf[1..]), draft_buf.len - 1);
            drafted += n_drafts - 1;

            if (n_drafts == 1) {
                // No draft (cold or muted sources): a plain step, so the
                // muted regime costs exactly what plain greedy costs.
                if (produced == gen_count) break;
                try tokenizer.decodeAppend(allocator, @intCast(next_token), &reply);
                produced += 1;
                logits = try llm.deepseek4.model.step(&model, &ctx, &session, next_token);
                forwards += 1;
                if (wants_topk) {
                    topKIds(logits, topk_ids[0][0..]);
                    topk_rows[0] = .{ .token = next_token, .topk = topk_ids[0][0..] };
                    source.observeTopK(topk_rows[0..1]);
                }
                next_token = argmax(logits);
                continue;
            }

            var snap = try session.cache.snapshot();
            defer snap.deinit();
            ctx.pinRowwiseKernels(true);
            _ = blk: {
                defer ctx.pinRowwiseKernels(false);
                break :blk try llm.deepseek4.model.stepBatchExtra(&model, &ctx, &session, draft_buf[0..n_drafts], rows[0..n_drafts], null);
            };
            forwards += 1;
            var accepted: usize = 1;
            while (accepted < n_drafts) : (accepted += 1) {
                if (argmax(rows[accepted - 1]) != draft_buf[accepted]) break;
            }
            draft_hits += accepted - 1;

            for (draft_buf[0..accepted]) |t| {
                if (produced == gen_count) break;
                try tokenizer.decodeAppend(allocator, @intCast(t), &reply);
                produced += 1;
                if (eos != null and t == eos.?) {
                    for (rows[0..n_drafts]) |r| allocator.free(r);
                    break :decode;
                }
            }
            try history.appendSlice(allocator, draft_buf[1..accepted]);
            if (history.items.len > prev_committed) index.observe(history.items[prev_committed..]);
            if (wants_topk) {
                for (0..accepted) |i| {
                    topKIds(rows[i], topk_ids[i][0..]);
                    topk_rows[i] = .{ .token = draft_buf[i], .topk = topk_ids[i][0..] };
                }
                source.observeTopK(topk_rows[0..accepted]);
            }
            next_token = argmax(rows[accepted - 1]);
            for (rows[0..n_drafts]) |r| allocator.free(r);
            if (accepted < n_drafts) {
                // Restore and replay only the accepted prefix — pinned, so
                // the rebuilt cache rows match sequential decode.
                session.cache.restore(&snap);
                ctx.pinRowwiseKernels(true);
                // Rows-less stepBatchExtra returns the SESSION-OWNED logits
                // slice — discarded, never freed (freeing it poisons the
                // session's buffer and the next step segfaults).
                _ = blk: {
                    defer ctx.pinRowwiseKernels(false);
                    break :blk try llm.deepseek4.model.stepBatchExtra(&model, &ctx, &session, draft_buf[0..accepted], null, null);
                };
                forwards += 1;
            }
        }
        try index.writeSourceSummary(stdout);
        try stdout.print("\n", .{});
    } else {
        var sampler = llm.sampler.Sampler.init(.{
            .temperature = temp_arg,
            .top_k = topk_arg,
            .top_p = topp_arg,
            .min_p = minp_arg,
            .repeat_penalty = penalty_arg,
            .seed = seed_arg,
        });
        var history: std.ArrayList(usize) = .empty;
        defer history.deinit(allocator);
        try history.appendSlice(allocator, tokens.items);
        while (produced < gen_count) : (produced += 1) {
            const best = if (sampler.config.isGreedy()) argmax(logits) else blk: {
                var lt = try fucina.Tensor(.{ .seq, .vocab }).fromBorrowedSlice(&ctx, .{ 1, model.config.vocab_size }, logits);
                defer lt.deinit();
                break :blk try sampler.next(&ctx, &lt, history.items);
            };
            if (eos != null and best == eos.?) break;
            try tokenizer.decodeAppend(allocator, @intCast(best), &reply);
            try history.append(allocator, best);
            logits = try llm.deepseek4.model.step(&model, &ctx, &session, best);
            forwards += 1;
        }
    }
    const decode_ns = std.Io.Clock.awake.now(init.io).nanoseconds - decode_start;
    try stdout.print("decode: {d} tokens in {d} forwards, {d:.1} ms, {d:.2} tok/s ({d:.2} tok/forward)\n", .{ produced, forwards, @as(f64, @floatFromInt(decode_ns)) / 1e6, @as(f64, @floatFromInt(produced)) * 1e9 / @as(f64, @floatFromInt(decode_ns)), @as(f64, @floatFromInt(produced)) / @as(f64, @floatFromInt(@max(forwards, 1))) });
    if (model.prof.steps > 0) {
        const p = model.prof;
        const n = @as(f64, @floatFromInt(p.steps));
        const attn = @as(f64, @floatFromInt(p.attn_ns)) / 1e6 / n;
        const router = @as(f64, @floatFromInt(p.router_ns)) / 1e6 / n;
        const experts = @as(f64, @floatFromInt(p.experts_ns)) / 1e6 / n;
        const shared = @as(f64, @floatFromInt(p.shared_ns)) / 1e6 / n;
        const head = @as(f64, @floatFromInt(p.head_ns)) / 1e6 / n;
        const wall = @as(f64, @floatFromInt(decode_ns)) / 1e6 / n;
        try stdout.print("decode profile (ms/token, {d} steps): attn {d:.1} | router {d:.1} | experts {d:.1} (incl stream waits) | shared {d:.1} | head {d:.1} | other {d:.1}\n", .{ p.steps, attn, router, experts, shared, head, wall - attn - router - experts - shared - head });
        const aq = @as(f64, @floatFromInt(p.attn_q_ns)) / 1e6 / n;
        const akv = @as(f64, @floatFromInt(p.attn_kv_ns)) / 1e6 / n;
        const aidx = @as(f64, @floatFromInt(p.attn_idx_ns)) / 1e6 / n;
        const arows = @as(f64, @floatFromInt(p.attn_rows_ns)) / 1e6 / n;
        const aatt = @as(f64, @floatFromInt(p.attn_attend_ns)) / 1e6 / n;
        const aplumb = @as(f64, @floatFromInt(p.attn_out_plumb_ns)) / 1e6 / n;
        const agroups = @as(f64, @floatFromInt(p.attn_out_groups_ns)) / 1e6 / n;
        const ab = @as(f64, @floatFromInt(p.attn_out_b_ns)) / 1e6 / n;
        try stdout.print("attn detail (ms/token): q {d:.1} | kv {d:.1} | index {d:.1} | rows {d:.1} | attend {d:.1} | out-plumb {d:.1} | out-groups {d:.1} | out-b {d:.1} | hc {d:.1}\n", .{ aq, akv, aidx, arows, aatt, aplumb, agroups, ab, attn - aq - akv - aidx - arows - aatt - aplumb - agroups - ab });
    }
    if (model.index_share_every >= 2) {
        try stdout.print("index-share: every {d} — selections computed {d}, reused {d}\n", .{ model.index_share_every, session.scratch.share_computed, session.scratch.share_reused });
    }
    if (session.probe) |*p| try p.report(stdout);
    if (drafted > 0) {
        try stdout.print("{s}: {d}/{d} drafts accepted ({d:.1}%)\n", .{ if (spec) "spec" else "mtp", draft_hits, drafted, @as(f64, @floatFromInt(draft_hits)) * 100.0 / @as(f64, @floatFromInt(drafted)) });
    }
    try stdout.print("prompt: {s}\ntext:  {s}\n", .{ prompt_text, reply.items });
}

/// The reference chat rendering (thinking disabled): BOS, user marker,
/// prompt, assistant marker, closed think block.
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

fn renderChat(allocator: std.mem.Allocator, tokenizer: *const Tokenizer, prompt: []const u8, out: *std.ArrayList(usize)) !void {
    const bos = tokenizer.bosId() orelse return error.MissingChatTokens;
    const user_id = tokenizer.tokenId("<｜User｜>") orelse return error.MissingChatTokens;
    const assistant_id = tokenizer.tokenId("<｜Assistant｜>") orelse return error.MissingChatTokens;
    const think_end_id = tokenizer.tokenId("</think>") orelse return error.MissingChatTokens;
    try out.append(allocator, bos);
    try out.append(allocator, user_id);
    const ids32 = try tokenizer.encode(allocator, prompt);
    defer allocator.free(ids32);
    for (ids32) |id| try out.append(allocator, id);
    try out.append(allocator, assistant_id);
    try out.append(allocator, think_end_id);
}

/// Top-`out.len` token ids of a logit row, most probable first (small-k
/// insertion scan — the recycling feedback wants k = 8).
fn topKIds(row: []const f32, out: []u32) void {
    var vals: [16]f32 = undefined;
    const k = @min(out.len, vals.len);
    var n: usize = 0;
    for (row, 0..) |v, i| {
        if (n == k and v <= vals[k - 1]) continue;
        var j = if (n < k) n else k - 1;
        while (j > 0 and vals[j - 1] < v) : (j -= 1) {
            vals[j] = vals[j - 1];
            out[j] = out[j - 1];
        }
        vals[j] = v;
        out[j] = @intCast(i);
        if (n < k) n += 1;
    }
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

/// One parsed `*.official.json` fixture: the exact user prompt, the API's
/// prompt token count, and the greedy continuation (one text per step).
const Vector = struct {
    id: []const u8,
    prompt: []const u8,
    prompt_tokens: usize,
    steps: [][]const u8,
};

fn parseVector(arena: std.mem.Allocator, bytes: []const u8) !Vector {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{});
    const root = parsed.object;
    const usage = root.get("usage") orelse return error.BadVector;
    const steps_v = root.get("steps") orelse return error.BadVector;
    const steps = try arena.alloc([]const u8, steps_v.array.items.len);
    for (steps_v.array.items, steps) |item, *out| {
        out.* = item.object.get("token").?.object.get("text").?.string;
    }
    return .{
        .id = (root.get("id") orelse return error.BadVector).string,
        .prompt = (root.get("prompt") orelse return error.BadVector).string,
        .prompt_tokens = @intCast(usage.object.get("prompt_tokens").?.integer),
        .steps = steps,
    };
}

/// Run every official fixture in `dir_path` with the reference chat rendering
/// and greedy decoding, and compare the continuation step-by-step against the
/// API's. Text is compared on concatenated bytes, so a different token
/// boundary with identical text still counts as a match. Fails when any run
/// vector diverges on the very first step (quantized weights legitimately
/// drift a few steps in; step 0 disagreeing means the forward is wrong).
/// Teacher-forced mean NLL (and perplexity) over the first `max_tokens`
/// supervised positions of a plain-encoded text file — the standard
/// perplexity currency for quant comparisons (mirrors ptqtp-qwen3's --nll).
/// Runs the deployed batched forward in `prefill_chunk` chunks with
/// per-row logits (`stepBatchExtra`), so the number is computed at prefill
/// speed on the exact serving path.
fn runNll(
    io: std.Io,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    model: *Model,
    tokenizer: *const Tokenizer,
    stdout: *std.Io.Writer,
    path: []const u8,
    max_tokens: usize,
    prefill_chunk: usize,
) !void {
    const bytes = try readAllFile(io, allocator, path);
    defer allocator.free(bytes);
    const ids32 = try tokenizer.encode(allocator, bytes);
    defer allocator.free(ids32);
    // max_tokens supervised transitions need max_tokens + 1 tokens.
    const n = @min(ids32.len, max_tokens + 1);
    if (n < 2) return error.NllTextTooShort;
    const tokens = try allocator.alloc(usize, n);
    defer allocator.free(tokens);
    for (tokens, ids32[0..n]) |*t, id| t.* = @intCast(id);

    var session = try Session.init(model, n + 8);
    defer session.deinit(model);

    var total: f64 = 0;
    var count: usize = 0;
    const rows = try allocator.alloc([]f32, prefill_chunk);
    defer allocator.free(rows);
    var fed: usize = 0;
    while (fed < n) {
        const end = @min(fed + prefill_chunk, n);
        const chunk = end - fed;
        // The returned final-row logits ALIAS rows[chunk-1] when rows are
        // requested (stepBatchExtra contract) — freeing the rows frees it.
        _ = try llm.deepseek4.model.stepBatchExtra(model, ctx, &session, tokens[fed..end], rows[0..chunk], null);
        defer for (rows[0..chunk]) |r| allocator.free(r);
        for (0..chunk) |j| {
            const pos = fed + j;
            if (pos + 1 >= n) break;
            const target = tokens[pos + 1];
            const row = rows[j];
            var max_logit: f32 = row[0];
            for (row) |v| max_logit = @max(max_logit, v);
            var sum_exp: f64 = 0;
            for (row) |v| sum_exp += @exp(@as(f64, v - max_logit));
            total += @as(f64, max_logit) + @log(sum_exp) - @as(f64, row[target]);
            count += 1;
        }
        fed = end;
    }
    const nll = total / @as(f64, @floatFromInt(count));
    try stdout.print("nll: {d:.4} (ppl {d:.2}) over {d} supervised tokens\n", .{ nll, @exp(nll), count });
    try stdout.flush();
}

fn runVectors(
    io: std.Io,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    model: *Model,
    tokenizer: *const Tokenizer,
    stdout: *std.Io.Writer,
    dir_path: []const u8,
    max_prompt: usize,
    prefill_chunk: usize,
) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);

    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".official.json")) continue;
        try names.append(allocator, try allocator.dupe(u8, entry.name));
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    if (names.items.len == 0) return error.NoVectors;

    var failures: usize = 0;
    var ran: usize = 0;
    for (names.items) |name| {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var file = try dir.openFile(io, name, .{});
        defer file.close(io);
        const stat = try file.stat(io);
        const bytes = try arena.alloc(u8, @intCast(stat.size));
        var read_len: usize = 0;
        while (read_len < bytes.len) {
            const n = try file.readStreaming(io, &.{bytes[read_len..]});
            if (n == 0) return error.EndOfStream;
            read_len += n;
        }
        const vec = try parseVector(arena, bytes);

        if (vec.prompt_tokens > max_prompt) {
            try stdout.print("{s}: SKIP ({d} prompt tokens > --vectors-max-prompt={d})\n", .{ vec.id, vec.prompt_tokens, max_prompt });
            try stdout.flush();
            continue;
        }
        ran += 1;

        var tokens: std.ArrayList(usize) = .empty;
        defer tokens.deinit(allocator);
        try renderChat(allocator, tokenizer, vec.prompt, &tokens);

        var session = try Session.init(model, tokens.items.len + vec.steps.len + 8);
        defer session.deinit(model);

        // Session-owned logits: valid until the next step, never freed.
        var logits: []f32 = &.{};
        var fed: usize = 0;
        while (fed < tokens.items.len) {
            const end = @min(fed + prefill_chunk, tokens.items.len);
            logits = try llm.deepseek4.model.stepBatch(model, ctx, &session, tokens.items[fed..end]);
            fed = end;
        }

        var ours: std.ArrayList(u8) = .empty;
        defer ours.deinit(allocator);
        const eos = tokenizer.eosId();
        for (vec.steps) |_| {
            const best = argmax(logits);
            if (eos != null and best == eos.?) break;
            try tokenizer.decodeAppend(allocator, @intCast(best), &ours);
            logits = try llm.deepseek4.model.step(model, ctx, &session, best);
        }

        // Longest official step-prefix our continuation reproduces.
        var official: std.ArrayList(u8) = .empty;
        defer official.deinit(allocator);
        var matched: usize = 0;
        for (vec.steps) |step_text| {
            try official.appendSlice(allocator, step_text);
            if (!std.mem.startsWith(u8, ours.items, official.items)) break;
            matched += 1;
        }

        const token_note = if (tokens.items.len == vec.prompt_tokens) "=" else "!";
        if (matched == 0) failures += 1;
        try stdout.print("{s}: {s} steps {d}/{d}, prompt tokens {d}{s}{d}\n  ours: {s}\n  api:  {s}\n", .{
            vec.id,
            if (matched > 0) "PASS" else "FAIL",
            matched,
            vec.steps.len,
            tokens.items.len,
            token_note,
            vec.prompt_tokens,
            ours.items,
            official.items,
        });
        try stdout.flush();
    }
    try stdout.print("vectors: {d} run, {d} failed\n", .{ ran, failures });
    // Flush BEFORE the error return: the per-fixture PASS/FAIL evidence
    // must survive a failing gate, not just a passing one.
    try stdout.flush();
    if (failures > 0) return error.VectorMismatch;
}

/// One parsed local-golden case: implementation-level logit fixture captured
/// from a known-sane upstream run of the same GGUF (tests/test-vectors/
/// local-golden.vec in the ds4 checkout). `frontier` prompt tokens are fed
/// and the logits at the frontier are compared against the recorded top-k.
const Golden = struct {
    id: []const u8,
    mode: []const u8,
    ctx: usize,
    frontier: usize,
    prompt_path: []const u8,
    ids: []usize,
    logits: []f32,
};

fn parseGolden(arena: std.mem.Allocator, bytes: []const u8) !Golden {
    var g: Golden = undefined;
    var ids: std.ArrayList(usize) = .empty;
    var logits: std.ArrayList(f32) = .empty;
    var seen_case = false;
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var it = std.mem.tokenizeScalar(u8, line, ' ');
        const kind = it.next() orelse continue;
        if (std.mem.eql(u8, kind, "case")) {
            if (seen_case) return error.MultipleGoldenCases;
            seen_case = true;
            g.id = try arena.dupe(u8, it.next() orelse return error.BadGolden);
            g.mode = try arena.dupe(u8, it.next() orelse return error.BadGolden);
            g.ctx = try std.fmt.parseInt(usize, it.next() orelse return error.BadGolden, 10);
            g.frontier = try std.fmt.parseInt(usize, it.next() orelse return error.BadGolden, 10);
            g.prompt_path = try arena.dupe(u8, it.next() orelse return error.BadGolden);
        } else if (std.mem.eql(u8, kind, "top")) {
            _ = it.next() orelse return error.BadGolden; // rank
            try ids.append(arena, try std.fmt.parseInt(usize, it.next() orelse return error.BadGolden, 10));
            try logits.append(arena, try std.fmt.parseFloat(f32, it.next() orelse return error.BadGolden));
        }
    }
    if (!seen_case or ids.items.len == 0) return error.BadGolden;
    g.ids = ids.items;
    g.logits = logits.items;
    return g;
}

/// Replay the ds4 local-golden logit fixture: prefill `frontier` prompt
/// tokens (mode "text": plain BPE, no BOS) and compare our frontier logits
/// against the recorded top-64 with the upstream thresholds (top-1 exact,
/// top-5 >= 4, top-20 >= 15, top-64 >= 40, top-20 max |delta| <= 8).
fn runGolden(
    io: std.Io,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    model: *Model,
    tokenizer: *const Tokenizer,
    stdout: *std.Io.Writer,
    vec_path: []const u8,
    prefill_chunk: usize,
) !void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const vec_bytes = try readWholeFile(io, arena, vec_path);
    const g = try parseGolden(arena, vec_bytes);
    if (!std.mem.eql(u8, g.mode, "text")) return error.UnsupportedGoldenMode;

    // The fixture's prompt path is relative to the upstream checkout root
    // (two levels above the .vec file).
    const vec_dir = std.fs.path.dirname(vec_path) orelse ".";
    const prompt_path = try std.fs.path.join(arena, &.{ vec_dir, "..", "..", g.prompt_path });
    const prompt_text = try readWholeFile(io, arena, prompt_path);

    const ids32 = try tokenizer.encode(arena, prompt_text);
    try stdout.print("golden {s}: prompt {d} tokens, frontier {d}\n", .{ g.id, ids32.len, g.frontier });
    try stdout.flush();
    if (ids32.len < g.frontier) return error.GoldenPromptTooShort;

    var session = try Session.init(model, g.frontier + 8);
    defer session.deinit(model);
    // Session-owned logits: valid until the next step, never freed.
    var logits: []f32 = &.{};
    var fed: usize = 0;
    const tokens = try arena.alloc(usize, g.frontier);
    for (tokens, ids32[0..g.frontier]) |*t, id| t.* = id;
    while (fed < g.frontier) {
        const end = @min(fed + prefill_chunk, g.frontier);
        logits = try llm.deepseek4.model.stepBatch(model, ctx, &session, tokens[fed..end]);
        fed = end;
    }

    // Our top-64 by full scan (vocab * 64 compares — fine for one shot).
    const ntop = g.ids.len;
    const our_top = try arena.alloc(usize, ntop);
    {
        const taken = try arena.alloc(bool, logits.len);
        @memset(taken, false);
        for (our_top) |*slot| {
            var best: usize = 0;
            var best_v = -std.math.inf(f32);
            for (logits, 0..) |v, i| {
                if (!taken[i] and v > best_v) {
                    best_v = v;
                    best = i;
                }
            }
            taken[best] = true;
            slot.* = best;
        }
    }

    var overlap5: usize = 0;
    var overlap20: usize = 0;
    var overlap64: usize = 0;
    for (g.ids, 0..) |gid, i| {
        for (our_top, 0..) |oid, j| {
            if (oid != gid) continue;
            if (i < 5 and j < 5) overlap5 += 1;
            if (i < 20 and j < 20) overlap20 += 1;
            if (i < 64 and j < 64) overlap64 += 1;
            break;
        }
    }
    var max_abs: f32 = 0;
    for (g.ids[0..@min(20, ntop)], g.logits[0..@min(20, ntop)]) |gid, glogit| {
        max_abs = @max(max_abs, @abs(logits[gid] - glogit));
    }

    const pass = our_top[0] == g.ids[0] and overlap5 >= 4 and overlap20 >= 15 and overlap64 >= 40 and max_abs <= 8.0;
    try stdout.print("golden {s}: {s} top1 ref={d} ours={d} (ref logit {d:.3} ours {d:.3}) overlap5 {d}/5 overlap20 {d}/20 overlap64 {d}/64 top20_max_abs {d:.4}\n", .{
        g.id,
        if (pass) "PASS" else "FAIL",
        g.ids[0],
        our_top[0],
        g.logits[0],
        logits[g.ids[0]],
        overlap5,
        overlap20,
        overlap64,
        max_abs,
    });
    try stdout.flush();
    if (!pass) return error.GoldenMismatch;
}

fn readWholeFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}
