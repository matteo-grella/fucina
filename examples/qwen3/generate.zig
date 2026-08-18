//! Generation drivers of the Qwen3 runner: plain `--gen` (single pass or
//! warm-repeat bench), `--streams N` batched multi-stream decode, and the
//! `--spec` lossless speculative decode.
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const bench = @import("bench.zig");
const util = @import("util.zig");

const benchOnePass = bench.benchOnePass;
const printBenchStat = bench.printBenchStat;
const printProfile = bench.printProfile;
const printProfileLabeled = bench.printProfileLabeled;
const nowNs = util.nowNs;
const seconds = util.seconds;
const millis = util.millis;

pub fn runGenerate(
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
    tokens_out: ?[]const u8,
) !void {
    const cfg = model.config;
    const capacity = tokens.len + max_new;
    var cache = try llm.kv_cache.KvCache.initWithDtype(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, cache_type);
    defer cache.deinit();
    try util.printCacheInfo(stdout, &cache);

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
    if (tokens_out) |path| try util.writeTokenIdsU32(io, allocator, path, tokens, out[0..r.produced]);
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
        const prompt_str = try util.decodeIds(allocator, t, tokens);
        defer allocator.free(prompt_str);
        const gen_str = try util.decodeIds(allocator, t, out[0..r.produced]);
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
pub fn runGenerateStreams(
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
    try util.printCacheInfo(stdout, &caches[0]);
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
pub fn runGenerateSpec(
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
    try util.printCacheInfo(stdout, &cache);

    var index = try llm.speculative.cascade.SpeculationIndex.init(allocator, cfg.vocab_size);
    defer index.deinit();
    // Acceptance accounting settles only drafts the decoder actually verifies
    // (the cascade's accounting contract).
    index.accounting_min_draft = spec_options.min_draft;
    for (ref_paths) |path| {
        const t = tok orelse return error.TokenizerUnavailable;
        const ids = try util.tokenizeFile(io, allocator, t, path);
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

    // Prefill EXACTLY as runGenerate does: every prompt token in one
    // batch, first new token committed from the prefill logits via the
    // decoder's bootstrap entry. Prefill kernels are batch-shape-
    // dependent, so plain and speculative runs must start from
    // call-for-call identical forwards to share cache bytes — the
    // caller's leg of the byte-identity contract (speculative/core.zig);
    // pinned verifies (Options.pin_kernels) are the other leg.
    const prefill_start = nowNs(io);
    {
        var pre = try model.forwardStep(ctx, &cache, tokens, 0);
        defer pre.deinit();
        _ = try decoder.bootstrapStep(ctx, &cache, &sampler, &history, sink, &pre);
    }
    const first_token = history.items[history.items.len - 1];
    const prefill_ns = nowNs(io) - prefill_start;

    const decode_start = nowNs(io);
    const first_is_stop = stop_token != null and first_token == stop_token.?;
    decode: while (!first_is_stop and history.items.len - tokens.len < max_new) {
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
        const gen_str = try util.decodeIds(allocator, t, out);
        defer allocator.free(gen_str);
        try stdout.print("text: {s}\n", .{gen_str});
    }
}

fn nullSinkEmit(ptr: *anyopaque, token: usize) anyerror!void {
    _ = ptr;
    _ = token;
}
