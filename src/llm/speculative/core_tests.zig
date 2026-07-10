//! Behavioral tests for the draft-model-free speculative decoder
//! (`speculative.zig`): forward-batch/per-token logit parity, greedy and
//! sampled losslessness (perfect/garbage/alternating draft sources),
//! hypothetical-prefix penalties, cost-table interpolation, the end-to-end
//! cost-aware AUTO-OFF gate, stop-aware verify RNG accounting, mid-step
//! error-unwind invariants, lying-source clamping, degenerate-option init
//! rejection, and the gemma4 instantiation compile coverage.

const std = @import("std");
const speculative = @import("core.zig");
const fucina = @import("fucina");
const kv_cache = @import("../kv_cache.zig");
const sampler_mod = @import("../sampler.zig");
const qwen3 = @import("../qwen3/model.zig");
const scaffolding = @import("../qwen3/train_tests.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;
const Sampler = sampler_mod.Sampler;
const Logits = fucina.Tensor(.{ .seq, .vocab });

const TopKRow = speculative.TopKRow;
const DraftSource = speculative.DraftSource;
const Stats = speculative.Stats;
const Options = speculative.Options;
const default_cost_table = speculative.default_cost_table;
const tableCost = speculative.tableCost;
const TokenSink = speculative.TokenSink;
const VerifyRowHook = speculative.VerifyRowHook;
const SpeculativeDecoder = speculative.SpeculativeDecoder;

// ---------------------------------------------------------------------------
// Test draft sources + losslessness tests.
//
// The model is the tiny synthetic dense qwen3 from qwen3_train_tests.zig
// (vocab 64, hidden 32, 2 layers, separate projections). All test sequences
// stay below the m-dependent kernel thresholds (fused FFN seq >= 12 needs a
// fused projection — the tiny model is separate; tiled attention seq >= 48),
// so verify-batch logits are expected BITWISE equal to single-step logits,
// which the replay-equivalence test asserts.
// ---------------------------------------------------------------------------

/// Replays a pre-recorded committed stream (prompt + a prior plain run's
/// output): the perfect oracle — every draft is accepted.
const PerfectSource = struct {
    future: []const usize,

    fn suggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
        const self: *PerfectSource = @ptrCast(@alignCast(ptr));
        if (context.len >= self.future.len) return 0;
        if (!std.mem.eql(usize, context, self.future[0..context.len])) return 0;
        const n = @min(buf.len, self.future.len - context.len);
        @memcpy(buf[0..n], self.future[context.len..][0..n]);
        return n;
    }
    fn observe(ptr: *anyopaque, committed: []const usize) void {
        _ = ptr;
        _ = committed;
    }
    const vtable = DraftSource.VTable{ .suggest = suggest, .observe = observe };
    fn source(self: *PerfectSource) DraftSource {
        return .{ .ptr = self, .vtable = &vtable };
    }
};

/// Proposes uniform-random token ids: the adversarial oracle — (almost) every
/// draft is rejected at position 0, so the output must still match plain.
const GarbageSource = struct {
    prng: std.Random.DefaultPrng,
    vocab: usize,
    len: usize,

    fn suggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
        const self: *GarbageSource = @ptrCast(@alignCast(ptr));
        _ = context;
        const n = @min(buf.len, self.len);
        for (buf[0..n]) |*t| t.* = self.prng.random().uintLessThan(usize, self.vocab);
        return n;
    }
    fn observe(ptr: *anyopaque, committed: []const usize) void {
        _ = ptr;
        _ = committed;
    }
    const vtable = DraftSource.VTable{ .suggest = suggest, .observe = observe };
    fn source(self: *GarbageSource) DraftSource {
        return .{ .ptr = self, .vtable = &vtable };
    }
};

/// Alternates per call between the perfect replay and garbage: partial
/// accepts, rejections, and corrections all occur in one run.
const AlternatingSource = struct {
    perfect: PerfectSource,
    garbage: GarbageSource,
    calls: usize = 0,

    fn suggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
        const self: *AlternatingSource = @ptrCast(@alignCast(ptr));
        self.calls += 1;
        if (self.calls % 2 == 1) return PerfectSource.suggest(&self.perfect, context, buf);
        return GarbageSource.suggest(&self.garbage, context, buf);
    }
    fn observe(ptr: *anyopaque, committed: []const usize) void {
        _ = ptr;
        _ = committed;
    }
    const vtable = DraftSource.VTable{ .suggest = suggest, .observe = observe };
    fn source(self: *AlternatingSource) DraftSource {
        return .{ .ptr = self, .vtable = &vtable };
    }
};

/// Never proposes: every step is a fallback — drives the AUTO-OFF gate
/// deterministically (rate exactly 1.0 tok/step).
const NeverSource = struct {
    fn suggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
        _ = ptr;
        _ = context;
        _ = buf;
        return 0;
    }
    fn observe(ptr: *anyopaque, committed: []const usize) void {
        _ = ptr;
        _ = committed;
    }
    const vtable = DraftSource.VTable{ .suggest = suggest, .observe = observe };
    fn source(self: *NeverSource) DraftSource {
        return .{ .ptr = self, .vtable = &vtable };
    }
};

/// Records the observe/observeTopK plumbing for assertions.
const RecordingSource = struct {
    inner: DraftSource,
    allocator: Allocator,
    observed: std.ArrayList(usize) = .empty,
    topk_rows: usize = 0,
    topk_first_candidates: std.ArrayList(usize) = .empty,

    fn suggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
        const self: *RecordingSource = @ptrCast(@alignCast(ptr));
        return self.inner.suggest(context, buf);
    }
    fn observe(ptr: *anyopaque, committed: []const usize) void {
        const self: *RecordingSource = @ptrCast(@alignCast(ptr));
        self.observed.appendSlice(self.allocator, committed) catch @panic("oom");
    }
    fn observeTopK(ptr: *anyopaque, positions: []const TopKRow) void {
        const self: *RecordingSource = @ptrCast(@alignCast(ptr));
        self.topk_rows += positions.len;
        for (positions) |row| {
            std.debug.assert(row.topk.len > 0);
            self.topk_first_candidates.append(self.allocator, row.topk[0]) catch @panic("oom");
        }
    }
    fn deinit(self: *RecordingSource) void {
        self.topk_first_candidates.deinit(self.allocator);
        self.observed.deinit(self.allocator);
    }
    const vtable = DraftSource.VTable{ .suggest = suggest, .observe = observe, .observeTopK = observeTopK };
    fn source(self: *RecordingSource) DraftSource {
        return .{ .ptr = self, .vtable = &vtable };
    }
};

/// Captured pre-penalty logits rows (one per committed token, in order).
const RowRecord = struct {
    abs_pos: usize,
    batch_index: usize,
    row: []f32,
};

const RowStore = struct {
    allocator: Allocator,
    rows: std.ArrayList(RowRecord) = .empty,

    fn capture(ptr: *anyopaque, abs_pos: usize, batch_index: usize, row: []const f32) anyerror!void {
        const self: *RowStore = @ptrCast(@alignCast(ptr));
        const copy = try self.allocator.dupe(f32, row);
        errdefer self.allocator.free(copy);
        try self.rows.append(self.allocator, .{ .abs_pos = abs_pos, .batch_index = batch_index, .row = copy });
    }
    fn hook(self: *RowStore) VerifyRowHook {
        return .{ .ptr = self, .func = capture };
    }
    fn deinit(self: *RowStore) void {
        for (self.rows.items) |r| self.allocator.free(r.row);
        self.rows.deinit(self.allocator);
    }
};

const NullSink = struct {
    var state: u8 = 0;

    fn emit(ptr: *anyopaque, token: usize) anyerror!void {
        _ = ptr;
        _ = token;
    }
    fn sink() TokenSink {
        return .{ .ptr = @ptrCast(&state), .func = emit };
    }
};

/// Fails on the `fail_at`-th emit (1-based): the mid-step error injector for
/// the decode-state-invariant tests.
const FailingSink = struct {
    fail_at: usize,
    emitted: usize = 0,

    fn emit(ptr: *anyopaque, token: usize) anyerror!void {
        _ = token;
        const self: *FailingSink = @ptrCast(@alignCast(ptr));
        self.emitted += 1;
        if (self.emitted == self.fail_at) return error.SinkFailed;
    }
    fn sink(self: *FailingSink) TokenSink {
        return .{ .ptr = self, .func = emit };
    }
};

/// Claims a draft length LARGER than the buffer it was given — the
/// adversarial injectable vtable the decoder must clamp at runtime.
const LyingSource = struct {
    var state: u8 = 0;

    fn suggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
        _ = ptr;
        _ = context;
        for (buf) |*t| t.* = 1;
        return buf.len + 7; // lie: more than the buffer holds
    }
    fn observe(ptr: *anyopaque, committed: []const usize) void {
        _ = ptr;
        _ = committed;
    }
    const vtable = DraftSource.VTable{ .suggest = suggest, .observe = observe };
    fn source() DraftSource {
        return .{ .ptr = @ptrCast(&state), .vtable = &vtable };
    }
};

/// Collects emitted tokens; the streaming-order assertion target.
const CollectSink = struct {
    allocator: Allocator,
    tokens: std.ArrayList(usize) = .empty,

    fn emit(ptr: *anyopaque, token: usize) anyerror!void {
        const self: *CollectSink = @ptrCast(@alignCast(ptr));
        try self.tokens.append(self.allocator, token);
    }
    fn sink(self: *CollectSink) TokenSink {
        return .{ .ptr = self, .func = emit };
    }
    fn deinit(self: *CollectSink) void {
        self.tokens.deinit(self.allocator);
    }
};

const test_prompt = [_]usize{ 5, 9, 13, 2, 33, 60, 21, 8 };

/// Whether verify-batch logits are guaranteed BITWISE equal to single-step
/// logits. True for the pure-Zig kernels (scalar backend, or native with
/// -Dblas=none): every op is row-wise independent below the m-thresholds the
/// tests stay under. Vendor BLAS GEMMs (Accelerate/OpenBLAS/...) pick
/// m-dependent kernels — measured drift on this tiny model is ~1e-6 rel
/// (e.g. -0.3014899 vs -0.30148914) — so there the tests assert a tight
/// tolerance instead; token streams are still asserted identical.
const batch_rows_bitwise = fucina.active_backend_kind != .native or !fucina.native_uses_blas;

/// Bitwise on BLAS-free builds; tight-tolerance (covers the ~1e-6 rel GEMM
/// reassociation drift on |logit| <= ~10) under vendor BLAS.
fn expectRowsMatch(want: []const f32, got: []const f32) !void {
    if (batch_rows_bitwise) {
        try std.testing.expectEqualSlices(f32, want, got);
    } else {
        try std.testing.expectEqual(want.len, got.len);
        for (want, got) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-4);
    }
}

/// The plain (non-speculative) reference run, in the exact protocol the
/// decoder uses: prefill `prompt[0 .. n-1]`, then single-token steps starting
/// from the last prompt token. Captures the committed stream (prompt +
/// generated) into `out` and, optionally, each step's pre-penalty logits row.
fn plainRun(
    ctx: *ExecContext,
    model: *const qwen3.Model,
    sampler_cfg: sampler_mod.Config,
    prompt: []const usize,
    max_new: usize,
    out: *std.ArrayList(usize),
    rows: ?*RowStore,
) !void {
    const cfg = model.config;
    var kv = try KvCache.init(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, prompt.len + max_new);
    defer kv.deinit();
    var sampler = Sampler.init(sampler_cfg);

    try out.appendSlice(ctx.allocator, prompt);
    if (prompt.len > 1) {
        var pre = try model.forwardStep(ctx, &kv, prompt[0 .. prompt.len - 1], 0);
        pre.deinit();
    }
    for (0..max_new) |_| {
        const last = out.items[out.items.len - 1];
        var logits = try model.forwardStep(ctx, &kv, &.{last}, kv.len);
        defer logits.deinit();
        if (rows) |store| try RowStore.capture(store, out.items.len, 0, try logits.dataConst());
        const next = try sampler.next(ctx, &logits, out.items);
        try out.append(ctx.allocator, next);
    }
}

/// A speculative run with `source`: same protocol/prompt as `plainRun`.
/// Returns the committed stream in `out`; `sink_out`/`rows` optional.
fn specRun(
    ctx: *ExecContext,
    model: *const qwen3.Model,
    sampler_cfg: sampler_mod.Config,
    prompt: []const usize,
    max_new: usize,
    source: DraftSource,
    options: Options,
    out: *std.ArrayList(usize),
    rows: ?*RowStore,
    sink_out: ?*CollectSink,
    stats_out: ?*Stats,
) !void {
    const cfg = model.config;
    var kv = try KvCache.init(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, prompt.len + max_new + options.max_draft + 1);
    defer kv.deinit();
    var sampler = Sampler.init(sampler_cfg);

    var decoder = try SpeculativeDecoder(qwen3.Model).init(ctx.allocator, source, options);
    defer decoder.deinit();
    if (rows) |store| decoder.on_verify_row = store.hook();

    try out.appendSlice(ctx.allocator, prompt);
    if (prompt.len > 1) {
        var pre = try model.forwardStep(ctx, &kv, prompt[0 .. prompt.len - 1], 0);
        pre.deinit();
    }
    const sink = if (sink_out) |c| c.sink() else NullSink.sink();
    while (out.items.len - prompt.len < max_new) {
        _ = try decoder.step(ctx, model, &kv, &sampler, out, sink);
    }
    // The decoder may overshoot max_new inside one verify batch; trim the
    // committed stream so runs compare over the same span.
    out.shrinkRetainingCapacity(prompt.len + max_new);
    if (stats_out) |s| s.* = decoder.stats;
}

test "forwardStepAllLogits matches per-token forwardStep bitwise (tiny model)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0x5BEC);
    defer model.deinit();
    const cfg = model.config;
    const tokens = test_prompt;

    // One batched all-logits pass...
    var kv_a = try KvCache.init(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, tokens.len);
    defer kv_a.deinit();
    var all = try model.forwardStepAllLogits(&ctx, &kv_a, &tokens, 0);
    defer all.deinit();
    try std.testing.expectEqual(tokens.len, all.dim(.seq));
    try std.testing.expectEqual(cfg.vocab_size, all.dim(.vocab));
    try std.testing.expectEqual(tokens.len, kv_a.len); // kv advance == forwardStep

    // ...must equal feeding the tokens one at a time: bitwise on BLAS-free
    // builds (kernels are row-wise independent below the m-thresholds), tight
    // tolerance under vendor BLAS (see batch_rows_bitwise).
    var kv_b = try KvCache.init(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, tokens.len);
    defer kv_b.deinit();
    const all_data = try all.dataConst();
    for (tokens, 0..) |token, i| {
        var step_logits = try model.forwardStep(&ctx, &kv_b, &.{token}, i);
        defer step_logits.deinit();
        try expectRowsMatch(all_data[i * cfg.vocab_size ..][0..cfg.vocab_size], try step_logits.dataConst());
    }

    // And the last all-logits row must equal the cached-prefill
    // (last_query_only) entry's logits for the same sequence — same f16 KV
    // path, so the only divergence is the last-layer narrowing.
    var kv_c = try KvCache.init(&ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, tokens.len);
    defer kv_c.deinit();
    var last_ref = try model.forwardStep(&ctx, &kv_c, &tokens, 0);
    defer last_ref.deinit();
    try expectRowsMatch(try last_ref.dataConst(), all_data[(tokens.len - 1) * cfg.vocab_size ..][0..cfg.vocab_size]);
}

test "greedy losslessness: speculative output == plain output for perfect, garbage, alternating sources" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0x10551);
    defer model.deinit();
    const greedy = sampler_mod.Config{}; // temperature 0 = argmax
    const max_new = 24;

    var plain: std.ArrayList(usize) = .empty;
    defer plain.deinit(allocator);
    try plainRun(&ctx, &model, greedy, &test_prompt, max_new, &plain, null);

    // Perfect source: everything accepted, output identical, few steps.
    {
        var perfect = PerfectSource{ .future = plain.items };
        var rec = RecordingSource{ .inner = perfect.source(), .allocator = allocator };
        defer rec.deinit();
        var got: std.ArrayList(usize) = .empty;
        defer got.deinit(allocator);
        var emitted = CollectSink{ .allocator = allocator };
        defer emitted.deinit();
        var stats: Stats = .{};
        try specRun(&ctx, &model, greedy, &test_prompt, max_new, rec.source(), .{}, &got, null, &emitted, &stats);
        try std.testing.expectEqualSlices(usize, plain.items, got.items);
        // Streaming: the sink saw exactly the generated tokens, in order.
        try std.testing.expectEqualSlices(usize, plain.items[test_prompt.len..], emitted.tokens.items[0..max_new]);
        // observe() saw every committed token, in order.
        try std.testing.expectEqualSlices(usize, emitted.tokens.items, rec.observed.items);
        // Perfect drafts: full acceptance, bonus tokens, no rejections.
        try std.testing.expect(stats.accepted == stats.drafted);
        try std.testing.expectEqual(@as(usize, 0), stats.rejected_steps);
        try std.testing.expect(stats.bonus > 0);
        try std.testing.expect(stats.spec_steps < max_new); // actually sped up
        // observeTopK plumbing: one row per committed token; with greedy and
        // no penalties the committed token IS the top-1 candidate.
        try std.testing.expectEqual(emitted.tokens.items.len, rec.topk_rows + stats.fallback_steps);
        for (rec.topk_first_candidates.items, emitted.tokens.items[0..rec.topk_first_candidates.items.len]) |cand, tok| {
            try std.testing.expectEqual(tok, cand);
        }
    }

    // Garbage source: (nearly) everything rejected — output still identical.
    {
        var garbage = GarbageSource{ .prng = std.Random.DefaultPrng.init(7), .vocab = model.config.vocab_size, .len = 8 };
        var got: std.ArrayList(usize) = .empty;
        defer got.deinit(allocator);
        var stats: Stats = .{};
        try specRun(&ctx, &model, greedy, &test_prompt, max_new, garbage.source(), .{}, &got, null, null, &stats);
        try std.testing.expectEqualSlices(usize, plain.items, got.items);
        try std.testing.expect(stats.rejected_steps > 0);
    }

    // Alternating source: partial accepts + corrections — output identical.
    {
        var alternating = AlternatingSource{
            .perfect = .{ .future = plain.items },
            .garbage = .{ .prng = std.Random.DefaultPrng.init(11), .vocab = model.config.vocab_size, .len = 8 },
        };
        var got: std.ArrayList(usize) = .empty;
        defer got.deinit(allocator);
        var stats: Stats = .{};
        try specRun(&ctx, &model, greedy, &test_prompt, max_new, alternating.source(), .{}, &got, null, null, &stats);
        try std.testing.expectEqualSlices(usize, plain.items, got.items);
        try std.testing.expect(stats.accepted > 0);
    }
}

test "sampled losslessness: replay equivalence — identical tokens AND bitwise-identical sampler inputs" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0x5A3D);
    defer model.deinit();
    const cfg = sampler_mod.Config{ .temperature = 0.8, .top_k = 20, .seed = 1234 };
    const max_new = 24;

    var plain: std.ArrayList(usize) = .empty;
    defer plain.deinit(allocator);
    var plain_rows = RowStore{ .allocator = allocator };
    defer plain_rows.deinit();
    try plainRun(&ctx, &model, cfg, &test_prompt, max_new, &plain, &plain_rows);

    // RNG contract: exactly one draw per committed token, in commit order, so
    // with bitwise-equal logits the sampled stream matches the plain run
    // token-for-token — for the perfect AND the garbage source.
    for (0..2) |which| {
        var perfect = PerfectSource{ .future = plain.items };
        var garbage = GarbageSource{ .prng = std.Random.DefaultPrng.init(3), .vocab = model.config.vocab_size, .len = 8 };
        const source = if (which == 0) perfect.source() else garbage.source();

        var got: std.ArrayList(usize) = .empty;
        defer got.deinit(allocator);
        var spec_rows = RowStore{ .allocator = allocator };
        defer spec_rows.deinit();
        try specRun(&ctx, &model, cfg, &test_prompt, max_new, source, .{}, &got, &spec_rows, null, null);
        try std.testing.expectEqualSlices(usize, plain.items, got.items);

        // Replay equivalence: at every commit position the sampler saw logits
        // equal to the plain run's for the same committed prefix — bitwise on
        // BLAS-free builds, within the documented GEMM batch drift otherwise
        // (see batch_rows_bitwise). The spec run may verify a few positions
        // past max_new inside its last batch; compare the shared span.
        try std.testing.expect(spec_rows.rows.items.len >= plain_rows.rows.items.len);
        for (plain_rows.rows.items, spec_rows.rows.items[0..plain_rows.rows.items.len]) |p, s| {
            try std.testing.expectEqual(p.abs_pos, s.abs_pos);
            try expectRowsMatch(p.row, s.row);
        }
    }
}

test "penalties condition on the hypothetical prefix (and that prefix matters)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0x9E4A1); // penalty-test seed
    defer model.deinit();
    // Greedy + strong frequency/presence penalties on a 64-token vocab: each
    // committed token meaningfully demotes itself in the very next position,
    // so a sampler that conditioned on stale (iteration-start) history would
    // pick different tokens.
    const cfg = sampler_mod.Config{ .freq_penalty = 1.5, .presence_penalty = 0.5, .repeat_last_n = 64 };
    const max_new = 24;

    var plain: std.ArrayList(usize) = .empty;
    defer plain.deinit(allocator);
    try plainRun(&ctx, &model, cfg, &test_prompt, max_new, &plain, null);

    var perfect = PerfectSource{ .future = plain.items };
    var got: std.ArrayList(usize) = .empty;
    defer got.deinit(allocator);
    var rows = RowStore{ .allocator = allocator };
    defer rows.deinit();
    var stats: Stats = .{};
    try specRun(&ctx, &model, cfg, &test_prompt, max_new, perfect.source(), .{}, &got, &rows, null, &stats);

    // The implementation must match the plain run exactly...
    try std.testing.expectEqualSlices(usize, plain.items, got.items);
    // ...via positions actually sampled with intra-iteration (hypothetical)
    // history: the perfect source must have had multi-token acceptances.
    try std.testing.expect(stats.accepted > 0);

    // Discriminance: re-running the sampler on the captured rows with the
    // STALE history (committed-at-iteration-start only, i.e. ignoring the
    // draft tokens accepted earlier in the same verify batch) must change at
    // least one pick — proving this test would catch a committed-only-history
    // implementation, not vacuously pass.
    var diverged = false;
    for (rows.rows.items) |r| {
        if (r.batch_index == 0) continue; // no hypothetical tokens yet
        if (r.abs_pos >= got.items.len) continue; // overshoot rows were trimmed
        const full_hist = got.items[0..r.abs_pos];
        const stale_hist = got.items[0 .. r.abs_pos - r.batch_index];

        const full_pick = try samplerPickOnCopy(&ctx, cfg, r.row, full_hist);
        const stale_pick = try samplerPickOnCopy(&ctx, cfg, r.row, stale_hist);
        // The hypothetical-history pick must be the token actually committed.
        try std.testing.expectEqual(got.items[r.abs_pos], full_pick);
        if (stale_pick != full_pick) diverged = true;
    }
    try std.testing.expect(diverged);
}

/// Runs a fresh greedy sampler over a COPY of `row` with `history` — the
/// counterfactual probe for the penalty test.
fn samplerPickOnCopy(ctx: *ExecContext, cfg: sampler_mod.Config, row: []const f32, history: []const usize) !usize {
    var logits = try Logits.fromSlice(ctx, .{ 1, row.len }, row);
    defer logits.deinit();
    var sampler = Sampler.init(cfg);
    return sampler.next(ctx, &logits, history);
}

test "cost table: interpolation between points, clamped at the ends" {
    const table = &default_cost_table; // k: 2 -> 1.65, 4 -> 1.42, 8 -> 2.84, 16 -> 4.5
    // Exact points.
    try std.testing.expectApproxEqAbs(@as(f32, 1.65), tableCost(table, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.42), tableCost(table, 4), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.84), tableCost(table, 8), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), tableCost(table, 16), 1e-6);
    // Linear between points.
    try std.testing.expectApproxEqAbs(@as(f32, 1.535), tableCost(table, 3), 1e-6); // mid 1.65..1.42
    try std.testing.expectApproxEqAbs(@as(f32, 2.13), tableCost(table, 6), 1e-6); // mid 1.42..2.84
    try std.testing.expectApproxEqAbs(@as(f32, 3.6700), tableCost(table, 12), 1e-4); // mid 2.84..4.5
    // Clamped outside the table.
    try std.testing.expectApproxEqAbs(@as(f32, 1.65), tableCost(table, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), tableCost(table, 32), 1e-6);
}

test "cost-aware gate end-to-end: garbage drafts trip it, output stays lossless" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xA0FF);
    defer model.deinit();
    const greedy = sampler_mod.Config{};
    const max_new = 24;
    // No decoder.io in specRun -> deterministic static-table economics.
    const options = Options{
        .rate_window = 8,
        .min_window_drafted = 4,
        .reprobe_after = 4,
        .reprobe_max = 16,
        .probe_steps = 2,
    };

    var plain: std.ArrayList(usize) = .empty;
    defer plain.deinit(allocator);
    try plainRun(&ctx, &model, greedy, &test_prompt, max_new, &plain, null);

    // Garbage drafts lose by construction (committed ~1 per verify at k <= 8
    // table cost >= 1.42): the gate must disable speculation, back off, and
    // re-probe — all without changing the committed stream.
    var garbage = GarbageSource{ .prng = std.Random.DefaultPrng.init(7), .vocab = model.config.vocab_size, .len = 8 };
    var got: std.ArrayList(usize) = .empty;
    defer got.deinit(allocator);
    var stats: Stats = .{};
    try specRun(&ctx, &model, greedy, &test_prompt, max_new, garbage.source(), options, &got, null, null, &stats);

    try std.testing.expectEqualSlices(usize, plain.items, got.items);
    try std.testing.expect(stats.spec_steps >= 2); // gathered evidence first
    try std.testing.expect(stats.disabled_steps > 0); // then tripped off
    try std.testing.expect(stats.rejected_steps > 0);
    // Re-probes happened: more verifies than the initial trip needed.
    try std.testing.expect(stats.spec_steps > 2);

    // A never-drafting source costs nothing — the gate must NOT trip on it
    // (fallback steps are plain steps; only verify economics lose time).
    var never = NeverSource{};
    var got_never: std.ArrayList(usize) = .empty;
    defer got_never.deinit(allocator);
    var stats_never: Stats = .{};
    try specRun(&ctx, &model, greedy, &test_prompt, max_new, never.source(), options, &got_never, null, null, &stats_never);
    try std.testing.expectEqualSlices(usize, plain.items, got_never.items);
    try std.testing.expectEqual(@as(usize, 0), stats_never.disabled_steps);
    try std.testing.expectEqual(stats_never.steps, stats_never.fallback_steps);

    // enabled=false: never speculates at all.
    var got_off: std.ArrayList(usize) = .empty;
    defer got_off.deinit(allocator);
    var stats_off: Stats = .{};
    var perfect = PerfectSource{ .future = plain.items };
    try specRun(&ctx, &model, greedy, &test_prompt, max_new, perfect.source(), .{ .enabled = false }, &got_off, null, null, &stats_off);
    try std.testing.expectEqualSlices(usize, plain.items, got_off.items);
    try std.testing.expectEqual(@as(usize, 0), stats_off.spec_steps);
    try std.testing.expectEqual(stats_off.steps, stats_off.disabled_steps);
}

test "stop-aware verify: accepted mid-batch stop ends the row loop — stream AND RNG draws match plain" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0x5A3D);
    defer model.deinit();
    const mcfg = model.config;
    const cfg = sampler_mod.Config{ .temperature = 0.8, .top_k = 20, .seed = 1234 };
    const max_new = 24;

    // Full plain run to learn the stream; a mid-stream generated token plays
    // the stop marker so a PerfectSource will deliver it as an ACCEPTED draft
    // token with rows still pending in the verify batch.
    var plain_full: std.ArrayList(usize) = .empty;
    defer plain_full.deinit(allocator);
    try plainRun(&ctx, &model, cfg, &test_prompt, max_new, &plain_full, null);
    const stop = plain_full.items[test_prompt.len + 9];
    const stop_idx = std.mem.indexOfScalarPos(usize, plain_full.items, test_prompt.len, stop).?;

    // Plain run WITH the stop (the chat-loop protocol: sample, break on
    // stop), counting RNG draws and keeping the final PRNG state.
    var plain_sampler = Sampler.init(cfg);
    var plain_draws: usize = 0;
    var plain: std.ArrayList(usize) = .empty;
    defer plain.deinit(allocator);
    {
        var kv = try KvCache.init(&ctx, mcfg.num_layers, mcfg.num_key_value_heads, mcfg.head_dim, test_prompt.len + max_new);
        defer kv.deinit();
        try plain.appendSlice(allocator, &test_prompt);
        var pre = try model.forwardStep(&ctx, &kv, test_prompt[0 .. test_prompt.len - 1], 0);
        pre.deinit();
        for (0..max_new) |_| {
            const last = plain.items[plain.items.len - 1];
            var logits = try model.forwardStep(&ctx, &kv, &.{last}, kv.len);
            defer logits.deinit();
            const next = try plain_sampler.next(&ctx, &logits, plain.items);
            plain_draws += 1;
            try plain.append(allocator, next);
            if (next == stop) break;
        }
    }
    try std.testing.expectEqualSlices(usize, plain_full.items[0 .. stop_idx + 1], plain.items);

    // Stop-aware speculative run over the same protocol.
    var perfect = PerfectSource{ .future = plain_full.items };
    var decoder = try SpeculativeDecoder(qwen3.Model).init(allocator, perfect.source(), .{ .stop_token = stop });
    defer decoder.deinit();
    var rows = RowStore{ .allocator = allocator }; // 1 captured row == 1 sampler.next call == 1 RNG draw
    defer rows.deinit();
    decoder.on_verify_row = rows.hook();
    var spec_sampler = Sampler.init(cfg);
    var got: std.ArrayList(usize) = .empty;
    defer got.deinit(allocator);
    {
        var kv = try KvCache.init(&ctx, mcfg.num_layers, mcfg.num_key_value_heads, mcfg.head_dim, test_prompt.len + max_new + decoder.options.max_draft + 1);
        defer kv.deinit();
        try got.appendSlice(allocator, &test_prompt);
        var pre = try model.forwardStep(&ctx, &kv, test_prompt[0 .. test_prompt.len - 1], 0);
        pre.deinit();
        while (std.mem.indexOfScalarPos(usize, got.items, test_prompt.len, stop) == null) {
            _ = try decoder.step(&ctx, &model, &kv, &spec_sampler, &got, NullSink.sink());
        }
        try std.testing.expectEqual(got.items.len, kv.len + 1);
    }

    // Committed stream identical to plain — including NO overshoot past the
    // stop: the row loop broke immediately after committing it.
    try std.testing.expectEqualSlices(usize, plain.items, got.items);
    // The stop really arrived as an accepted draft token (the early-break
    // path), not merely as a correction/bonus sample.
    try std.testing.expect(decoder.stats.accepted > 0);
    try std.testing.expectEqual(@as(usize, 0), decoder.stats.rejected_steps);
    // Draw-for-draw identical: same draw count, same final PRNG state — a
    // persistent sampler stays aligned with the plain run.
    try std.testing.expectEqual(plain_draws, rows.rows.items.len);
    try std.testing.expect(std.meta.eql(plain_sampler.prng, spec_sampler.prng));
}

test "mid-step sink failures keep history.len == kv.len + 1; the resumed stream matches plain" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0x10551);
    defer model.deinit();
    const mcfg = model.config;
    const greedy = sampler_mod.Config{};
    const max_new = 16;

    var plain: std.ArrayList(usize) = .empty;
    defer plain.deinit(allocator);
    try plainRun(&ctx, &model, greedy, &test_prompt, max_new, &plain, null);

    // (a) verifyStep: the sink fails mid-batch, AFTER part of the batch
    // committed; (b) plainStep (forced by a never-drafting source): the sink
    // fails on the single emit.
    const cases = [_]struct { use_perfect: bool, fail_at: usize }{
        .{ .use_perfect = true, .fail_at = 3 },
        .{ .use_perfect = false, .fail_at = 1 },
    };
    for (cases) |case| {
        var perfect = PerfectSource{ .future = plain.items };
        var never = NeverSource{};
        const source = if (case.use_perfect) perfect.source() else never.source();
        var decoder = try SpeculativeDecoder(qwen3.Model).init(allocator, source, .{});
        defer decoder.deinit();
        var sampler = Sampler.init(greedy);
        var kv = try KvCache.init(&ctx, mcfg.num_layers, mcfg.num_key_value_heads, mcfg.head_dim, test_prompt.len + max_new + decoder.options.max_draft + 1);
        defer kv.deinit();
        var history: std.ArrayList(usize) = .empty;
        defer history.deinit(allocator);
        try history.appendSlice(allocator, &test_prompt);
        var pre = try model.forwardStep(&ctx, &kv, test_prompt[0 .. test_prompt.len - 1], 0);
        pre.deinit();

        var failing = FailingSink{ .fail_at = case.fail_at };
        try std.testing.expectError(error.SinkFailed, decoder.step(&ctx, &model, &kv, &sampler, &history, failing.sink()));
        // The decode-state invariant survived the unwind...
        try std.testing.expectEqual(history.items.len, kv.len + 1);
        // ...and everything committed so far is a prefix of the plain stream.
        try std.testing.expect(history.items.len <= plain.items.len);
        try std.testing.expectEqualSlices(usize, plain.items[0..history.items.len], history.items);

        // Resume with a working sink: the stream continues exactly as plain.
        while (history.items.len < plain.items.len) {
            _ = try decoder.step(&ctx, &model, &kv, &sampler, &history, NullSink.sink());
        }
        try std.testing.expectEqualSlices(usize, plain.items, history.items[0..plain.items.len]);
    }
}

test "step rejects empty or kv-desynced history at runtime" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0x10551);
    defer model.deinit();
    const mcfg = model.config;
    var kv = try KvCache.init(&ctx, mcfg.num_layers, mcfg.num_key_value_heads, mcfg.head_dim, 8);
    defer kv.deinit();
    var never = NeverSource{};
    var decoder = try SpeculativeDecoder(qwen3.Model).init(allocator, never.source(), .{});
    defer decoder.deinit();
    var sampler = Sampler.init(.{});
    var history: std.ArrayList(usize) = .empty;
    defer history.deinit(allocator);

    // Empty history (e.g. an empty prompt): hard error, not ReleaseFast UB.
    try std.testing.expectError(error.InvalidDecodeState, decoder.step(&ctx, &model, &kv, &sampler, &history, NullSink.sink()));
    // Desynced history (history.len != kv.len + 1): same error.
    try history.appendSlice(allocator, &.{ 1, 2 });
    try std.testing.expectError(error.InvalidDecodeState, decoder.step(&ctx, &model, &kv, &sampler, &history, NullSink.sink()));
    try std.testing.expectEqual(@as(usize, 0), decoder.stats.steps);
}

test "suggest return is clamped: a lying source cannot overrun the verify buffer" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0x10551);
    defer model.deinit();
    const greedy = sampler_mod.Config{};
    const max_new = 16;

    var plain: std.ArrayList(usize) = .empty;
    defer plain.deinit(allocator);
    try plainRun(&ctx, &model, greedy, &test_prompt, max_new, &plain, null);

    var got: std.ArrayList(usize) = .empty;
    defer got.deinit(allocator);
    var stats: Stats = .{};
    try specRun(&ctx, &model, greedy, &test_prompt, max_new, LyingSource.source(), .{}, &got, null, null, &stats);
    try std.testing.expectEqualSlices(usize, plain.items, got.items);
    try std.testing.expect(stats.drafted > 0);
    // Every verified draft was clamped to the decoder's own budget.
    try std.testing.expect(stats.drafted <= stats.spec_steps * decoder_default_max_draft);
}

const decoder_default_max_draft = (Options{}).max_draft;

test "init rejects topk_feedback == 0 for a feedback-consuming source" {
    var never = NeverSource{};
    var rec = RecordingSource{ .inner = never.source(), .allocator = std.testing.allocator };
    defer rec.deinit();
    try std.testing.expectError(
        error.TopKFeedbackDisabled,
        SpeculativeDecoder(qwen3.Model).init(std.testing.allocator, rec.source(), .{ .topk_feedback = 0 }),
    );
    // A source without observeTopK never gets feedback: 0 is fine there.
    var decoder = try SpeculativeDecoder(qwen3.Model).init(std.testing.allocator, never.source(), .{ .topk_feedback = 0 });
    decoder.deinit();
}

test "init rejects degenerate cost-gate options in every build mode" {
    const allocator = std.testing.allocator;
    var never = NeverSource{};
    const Dec = SpeculativeDecoder(qwen3.Model);
    try std.testing.expectError(error.RateWindowTooSmall, Dec.init(allocator, never.source(), .{ .rate_window = 1 }));
    try std.testing.expectError(error.ProbeStepsZero, Dec.init(allocator, never.source(), .{ .probe_steps = 0 }));
    try std.testing.expectError(error.CostTableEmpty, Dec.init(allocator, never.source(), .{ .cost_table = &.{} }));
    try std.testing.expectError(error.ReprobeAfterZero, Dec.init(allocator, never.source(), .{ .reprobe_after = 0 }));
    // The defaults pass.
    var decoder = try Dec.init(allocator, never.source(), .{});
    decoder.deinit();
}

test "SpeculativeDecoder instantiates for gemma4.Model (compile coverage)" {
    // gemma4 shares the duck-typed forwardStep/forwardStepAllLogits/KvCache
    // contract; force semantic analysis of every decoder path against it.
    const Dec = SpeculativeDecoder(@import("../gemma/gemma4.zig").Model);
    std.testing.refAllDecls(Dec);
    var decoder = try Dec.init(std.testing.allocator, NullDraft.source(), .{});
    decoder.deinit();
}

/// Minimal source for the gemma4 instantiation test.
const NullDraft = struct {
    var state: u8 = 0;
    fn suggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
        _ = ptr;
        _ = context;
        _ = buf;
        return 0;
    }
    fn observe(ptr: *anyopaque, committed: []const usize) void {
        _ = ptr;
        _ = committed;
    }
    const vtable = DraftSource.VTable{ .suggest = suggest, .observe = observe };
    fn source() DraftSource {
        return .{ .ptr = @ptrCast(&state), .vtable = &vtable };
    }
};

test "stats summary renders" {
    const stats = Stats{ .steps = 10, .spec_steps = 8, .fallback_steps = 1, .disabled_steps = 1, .drafted = 64, .accepted = 40, .rejected_steps = 5, .bonus = 3, .committed = 49 };
    var buf: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try stats.writeSummary(&writer);
    const text = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, text, "committed=49") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "4.90 tok/step") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "62.5% acc") != null);
}
