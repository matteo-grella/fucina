//! Behavioral tests for the draft-source cascade (`spec_cascade.zig`):
//! longest-match draft-budget policy, recycling fallback, per-source muting +
//! re-probe, same-step topK/committed-bigram ordering, degraded-SAM handling,
//! and the end-to-end losslessness contract on the tiny synthetic model.

const std = @import("std");
const spec_cascade = @import("cascade.zig");
const speculative = @import("core.zig");
const sam_index = @import("sam_index.zig");
const recycling_mod = @import("recycling.zig");

const Allocator = std.mem.Allocator;
const SamIndex = sam_index.SamIndex;
const Recycling = recycling_mod.Recycling;
const DraftSource = speculative.DraftSource;
const TopKRow = speculative.TopKRow;

const gate_window = spec_cascade.gate_window;
const SpeculationIndex = spec_cascade.SpeculationIndex;

test "draft budget = beta * (1 + match_len), capped by buf" {
    const allocator = std.testing.allocator;
    var index = try SpeculationIndex.init(allocator, 1024);
    defer index.deinit();

    // [10,11] (match_len 2) occurred at the start, followed by 12,13,...,19.
    const stream = [_]usize{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 10, 11 };
    index.observe(&stream);
    try std.testing.expectEqual(@as(usize, 2), index.conversation.matchLen());

    var buf: [16]usize = undefined;
    // beta = 2 (default): budget = 2 * (1 + 2) = 6.
    try std.testing.expectEqual(@as(usize, 6), index.suggest(&stream, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 12, 13, 14, 15, 16, 17 }, buf[0..6]);
    // buf shorter than the budget caps it.
    try std.testing.expectEqual(@as(usize, 4), index.suggest(&stream, buf[0..4]));
    // beta = 3: budget = 3 * (1 + 2) = 9.
    index.beta = 3;
    try std.testing.expectEqual(@as(usize, 9), index.suggest(&stream, &buf));
}

test "recycling fallback: chain when no SAM match, 0 on unseen rows" {
    const allocator = std.testing.allocator;
    var index = try SpeculationIndex.init(allocator, 64);
    defer index.deinit();

    // No repetition: conversation match stays below min_match.
    const stream = [_]usize{ 1, 2, 3, 4, 5 };
    index.observe(&stream);
    try std.testing.expect(index.conversation.matchLen() < index.min_match);

    var buf: [16]usize = undefined;
    // Row 5 unseen (observe only seeded bigrams 1->2..4->5): no draft at all
    // -> the decoder does a plain step.
    try std.testing.expectEqual(@as(usize, 0), index.suggest(&stream, &buf));

    // Recycle verifier logits: 5 -> 9 -> 5 -> ... chain, capped at
    // recycling_chain (8) even with a 16-slot buf.
    index.observeTopK(&.{
        .{ .token = 5, .topk = &.{9} },
        .{ .token = 9, .topk = &.{5} },
    });
    const n = index.suggest(&stream, &buf);
    try std.testing.expectEqual(@as(usize, 8), n);
    try std.testing.expectEqualSlices(usize, &.{ 9, 5, 9, 5, 9, 5, 9, 5 }, buf[0..8]);
}

test "per-source muting after sustained rejection, then re-probe" {
    const allocator = std.testing.allocator;
    var index = try SpeculationIndex.init(allocator, 1024);
    defer index.deinit();
    index.mute_commits = 50;

    var history: std.ArrayList(usize) = .empty;
    defer history.deinit(allocator);
    const feed = struct {
        fn go(idx: *SpeculationIndex, h: *std.ArrayList(usize), toks: []const usize) !void {
            try h.appendSlice(std.testing.allocator, toks);
            idx.observe(toks);
        }
    }.go;

    // Seed: [1,2] recurs with two different continuations (3 and 7).
    try feed(&index, &history, &.{ 1, 2, 3, 1, 2, 7, 1, 2 });

    var buf: [16]usize = undefined;
    // Adversarial loop: always commit the continuation the draft did NOT
    // predict -> every drafted token is rejected -> the gate must mute.
    var rounds: usize = 0;
    while (!index.conv_gate.muted() and rounds < 100) : (rounds += 1) {
        const n = index.suggest(history.items, &buf);
        try std.testing.expect(n >= 2); // conversation drafting (match_len >= 2)
        const commit: usize = if (buf[0] == 7) 3 else 7;
        try feed(&index, &history, &.{commit});
        try feed(&index, &history, &.{ 1, 2 });
    }
    try std.testing.expect(index.conv_gate.muted());
    try std.testing.expectEqual(@as(usize, 0), index.conv_gate.total_accepted);
    try std.testing.expect(index.conv_gate.total_drafted >= gate_window);

    // While muted, the conversation never drafts (the fallback may).
    const drafted_at_mute = index.conv_gate.total_drafted;
    {
        const n = index.suggest(history.items, &buf);
        // Anything offered now comes from recycling (chain from token 2).
        if (n > 0) try std.testing.expect(index.pending.?.source == .recycling);
        index.pending = null;
    }
    try std.testing.expectEqual(drafted_at_mute, index.conv_gate.total_drafted);
    try std.testing.expectEqual(@as(usize, 1), index.conv_gate.times_muted);

    // Re-probe: after mute_commits committed tokens the source is back.
    var fed: usize = 0;
    while (fed < index.mute_commits) : (fed += 3) {
        try feed(&index, &history, &.{ 50, 1, 2 });
    }
    try std.testing.expect(!index.conv_gate.muted());
    const n = index.suggest(history.items, &buf);
    try std.testing.expect(n >= 2);
    try std.testing.expect(index.pending.?.source == .conversation);
}

test "truncatePending: a stop-truncated draft settles only the verified prefix" {
    const allocator = std.testing.allocator;
    var index = try SpeculationIndex.init(allocator, 1024);
    defer index.deinit();

    // [1,2] recurs, followed by [3,4,5,6]: suggest drafts 6 tokens.
    index.observe(&.{ 1, 2, 3, 4, 5, 6, 1, 2 });
    var buf: [16]usize = undefined;
    try std.testing.expectEqual(@as(usize, 6), index.suggest(&.{ 1, 2, 3, 4, 5, 6, 1, 2 }, &buf));
    // A turn filter cut the draft at a stop marker after 3 tokens; the
    // decoder verifies only [3,4,5] and fully accepts it (+ a bonus token).
    // The dropped tail must not count as rejections.
    index.truncatePending(3);
    index.observe(&.{ 3, 4, 5, 77 });
    try std.testing.expectEqual(@as(usize, 3), index.conv_gate.total_drafted);
    try std.testing.expectEqual(@as(usize, 3), index.conv_gate.total_accepted);
    try std.testing.expect(index.pending == null);

    // Truncation below accounting_min_draft: the decoder falls back to a
    // plain step and the draft is never verified — the pending is forgotten,
    // so the next observe settles nothing.
    try std.testing.expect(index.suggest(&.{ 4, 5, 77, 1, 2 }, &buf) >= index.accounting_min_draft);
    index.truncatePending(1);
    try std.testing.expect(index.pending == null);
    index.observe(&.{99});
    try std.testing.expectEqual(@as(usize, 3), index.conv_gate.total_drafted);
    try std.testing.expectEqual(@as(usize, 3), index.conv_gate.total_accepted);

    // Truncating to the full length (or longer) is a no-op.
    try std.testing.expect(index.suggest(&.{ 99, 1, 2 }, &buf) >= index.accounting_min_draft);
    const drafted = index.pending.?.drafted;
    index.truncatePending(drafted);
    try std.testing.expectEqual(drafted, index.pending.?.drafted);
}

test "same-step topK row overwrite does not clobber committed-bigram promotions" {
    const allocator = std.testing.allocator;
    var index = try SpeculationIndex.init(allocator, 64);
    defer index.deinit();

    // One decoder step in the decoder's call order: observe(committed) then
    // observeTopK(rows). The committed bigram 5 -> 9 must survive the
    // same-step row-5 overwrite, with the fresh candidates following it.
    index.observe(&.{ 5, 9 });
    index.observeTopK(&.{
        .{ .token = 5, .topk = &.{ 7, 8 } },
        .{ .token = 9, .topk = &.{1} },
    });
    try std.testing.expectEqualSlices(u32, &.{ 9, 7, 8 }, index.recycling.topkOf(5)[0..3]);
    try std.testing.expectEqual(@as(u32, 1), index.recycling.topkOf(9)[0]);

    // The cross-call boundary bigram (last_token 9 -> committed 2) survives
    // a same-step overwrite of row 9 too.
    index.observe(&.{2});
    index.observeTopK(&.{.{ .token = 9, .topk = &.{ 4, 5 } }});
    try std.testing.expectEqualSlices(u32, &.{ 2, 4, 5 }, index.recycling.topkOf(9)[0..3]);
}

test "degraded conversation SAM: live-only reference cursors, explicit muting" {
    const allocator = std.testing.allocator;
    var index = try SpeculationIndex.init(allocator, 1024);
    defer index.deinit();

    // Healthy stream first, then degrade the conversation SAM (as a failed
    // mid-append would): its token copy no longer matches the committed
    // stream.
    index.observe(&.{ 1, 2, 3, 1, 2 });
    index.conversation.degraded = true;
    index.rec_gate.mute_left = 1_000_000; // isolate: no recycling fallback

    // addReference must NOT catch the cursor up from the untrusted token
    // copy: the cursor starts live-only (root).
    try index.addReference(&.{ 9, 1, 2, 3, 7, 7 });
    try std.testing.expectEqual(@as(u32, 0), index.references.items[0].cursor.len);

    // The conversation source is never consulted (suggest offers nothing
    // even though the stale copy would have matched [1,2])...
    var buf: [8]usize = undefined;
    try std.testing.expectEqual(@as(usize, 0), index.suggest(&.{ 1, 2, 3, 1, 2 }, &buf));
    // ...and never fed again: its token copy stays at the degraded length.
    const n_before = index.conversation.tokenCount();
    index.observe(&.{ 3, 7 });
    try std.testing.expectEqual(n_before, index.conversation.tokenCount());

    // The reference cursor advanced over the LIVE tokens only and now
    // drafts the doc's continuation of [3,7].
    try std.testing.expectEqual(@as(u32, 2), index.references.items[0].cursor.len);
    const n = index.suggest(&.{ 1, 2, 3, 1, 2, 3, 7 }, &buf);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(usize, 7), buf[0]);
}

// ---------------------------------------------------------------------------
// End-to-end losslessness on the tiny synthetic model: the FULL cascade
// (conversation SAM + injected reference + recycling w/ topK feedback) must
// reproduce the plain run token-for-token, greedy AND sampled — the same
// replay-equivalence contract speculative.zig pins for its test sources.
// ---------------------------------------------------------------------------

const fucina = @import("fucina");
const qwen3 = @import("../qwen3/model.zig");
const kv_cache = @import("../kv_cache.zig");
const sampler_mod = @import("../sampler.zig");
const scaffolding = @import("../qwen3/train_tests.zig");

const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;
const Sampler = sampler_mod.Sampler;

const test_prompt = [_]usize{ 5, 9, 13, 2, 33, 60, 21, 8 };

const NullSink = struct {
    var state: u8 = 0;
    fn emit(ptr: *anyopaque, token: usize) anyerror!void {
        _ = ptr;
        _ = token;
    }
    fn sink() speculative.TokenSink {
        return .{ .ptr = @ptrCast(&state), .func = emit };
    }
};

/// Plain reference run in the decoder's protocol: prefill prompt[0..n-1],
/// then single-token steps from the last prompt token.
fn plainRun(
    ctx: *ExecContext,
    model: *const qwen3.Model,
    sampler_cfg: sampler_mod.Config,
    prompt: []const usize,
    max_new: usize,
    out: *std.ArrayList(usize),
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
        const next = try sampler.next(ctx, &logits, out.items);
        try out.append(ctx.allocator, next);
    }
}

/// Cascade-driven speculative run, same protocol/prompt as `plainRun`.
fn cascadeRun(
    ctx: *ExecContext,
    model: *const qwen3.Model,
    sampler_cfg: sampler_mod.Config,
    prompt: []const usize,
    max_new: usize,
    index: *SpeculationIndex,
    out: *std.ArrayList(usize),
    stats_out: *speculative.Stats,
) !void {
    const cfg = model.config;
    const options = speculative.Options{};
    var kv = try KvCache.init(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, prompt.len + max_new + options.max_draft + 1);
    defer kv.deinit();
    var sampler = Sampler.init(sampler_cfg);

    var decoder = try speculative.SpeculativeDecoder(qwen3.Model).init(ctx.allocator, index.asDraftSource(), options);
    defer decoder.deinit();

    try out.appendSlice(ctx.allocator, prompt);
    index.observe(prompt);
    if (prompt.len > 1) {
        var pre = try model.forwardStep(ctx, &kv, prompt[0 .. prompt.len - 1], 0);
        pre.deinit();
    }
    while (out.items.len - prompt.len < max_new) {
        _ = try decoder.step(ctx, model, &kv, &sampler, out, NullSink.sink());
    }
    out.shrinkRetainingCapacity(prompt.len + max_new);
    stats_out.* = decoder.stats;
}

test "cascade end-to-end lossless: spec output == plain output, greedy and sampled" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0x5A3D);
    defer model.deinit();
    const max_new = 24;

    const configs = [_]sampler_mod.Config{
        .{}, // greedy (temperature 0 = argmax)
        .{ .temperature = 0.8, .top_k = 20, .seed = 1234 }, // sampled
    };
    for (configs) |cfg| {
        var plain: std.ArrayList(usize) = .empty;
        defer plain.deinit(allocator);
        try plainRun(&ctx, &model, cfg, &test_prompt, max_new, &plain);

        // Full cascade with the plain output injected as a reference doc:
        // the strongest realistic oracle (grounded/RAG decoding).
        var index = try SpeculationIndex.init(allocator, model.config.vocab_size);
        defer index.deinit();
        try index.addReference(plain.items);

        var got: std.ArrayList(usize) = .empty;
        defer got.deinit(allocator);
        var stats: speculative.Stats = .{};
        try cascadeRun(&ctx, &model, cfg, &test_prompt, max_new, &index, &got, &stats);

        try std.testing.expectEqualSlices(usize, plain.items, got.items);
        // The cascade actually drafted and the reference was accepted.
        try std.testing.expect(stats.accepted > 0);
        try std.testing.expect(stats.tokensPerStep() > 1.0);
        const ref_gate = &index.references.items[0].gate;
        try std.testing.expect(ref_gate.total_accepted > 0);
    }

    // Cascade WITHOUT a reference (conversation + recycling only): still
    // lossless on a fresh prompt.
    {
        var plain: std.ArrayList(usize) = .empty;
        defer plain.deinit(allocator);
        try plainRun(&ctx, &model, .{}, &test_prompt, max_new, &plain);

        var index = try SpeculationIndex.init(allocator, model.config.vocab_size);
        defer index.deinit();
        var got: std.ArrayList(usize) = .empty;
        defer got.deinit(allocator);
        var stats: speculative.Stats = .{};
        try cascadeRun(&ctx, &model, .{}, &test_prompt, max_new, &index, &got, &stats);
        try std.testing.expectEqualSlices(usize, plain.items, got.items);
    }
}
