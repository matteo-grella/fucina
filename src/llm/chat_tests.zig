//! Behavioral tests for the chat module (`chat.zig`): chat-template rendering
//! and format detection (ChatML + Gemma 4 `<|turn>`), the generic
//! `Conversation(Model, Tok)` instantiation coverage (qwen3 + byte-BPE,
//! gemma4 + SPM), stop handling (extra stop ids, text stop sequences), the
//! speculation-config seams, plus the speculation-on == speculation-off
//! conversation-equivalence proofs (greedy + sampled draw-for-draw parity) and
//! the mid-turn writer-failure consistency invariant.

const std = @import("std");
const chat = @import("chat.zig");
const fucina = @import("fucina");
const tok_mod = @import("tokenizer.zig");
const sampler_mod = @import("sampler.zig");
const qwen3 = @import("qwen3/model.zig");
const scaffolding = @import("qwen3/train_tests.zig");

const ExecContext = fucina.ExecContext;
const Tokenizer = tok_mod.Tokenizer;

const Format = chat.Format;
const Template = chat.Template;
const Options = chat.Options;
const Conversation = chat.Conversation(qwen3.Model, tok_mod);

test "chatml template renders first and subsequent turns" {
    const allocator = std.testing.allocator;
    const t = Template.detect("a {{ '<|im_start|>' }} template").?;
    try std.testing.expectEqual(Format.chatml, t.format);
    try std.testing.expectEqualStrings("<|im_end|>", t.stopMarker());

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try t.renderTurn(allocator, &buf, "You are helpful.", "Hi", true, false);
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        buf.items,
    );

    buf.clearRetainingCapacity();
    try t.renderTurn(allocator, &buf, "ignored on later turns", "Bye", false, false);
    try std.testing.expectEqualStrings(
        "<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n",
        buf.items,
    );
}

test "format detection picks llama3, gemma (1-3) and gemma4" {
    try std.testing.expectEqual(Format.llama3, Template.detect("x <|start_header_id|> y").?.format);
    try std.testing.expectEqual(Format.gemma, Template.detect("x <start_of_turn> y").?.format);
    try std.testing.expectEqual(Format.gemma4, Template.detect("x {{ '<|turn>' + role }} y").?.format);
    try std.testing.expectEqual(@as(?Template, null), Template.detect(null));
    try std.testing.expectEqual(@as(?Template, null), Template.detect("no markers here"));
}

test "gemma4 template renders <|turn> turns with thought-channel priming" {
    const allocator = std.testing.allocator;
    const t = Template{ .format = .gemma4 };
    try std.testing.expectEqualStrings("<turn|>", t.stopMarker());

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    // Thinking off (the template default): the model opener primes an empty
    // thought channel; the system turn carries no <|think|> marker.
    try t.renderTurn(allocator, &buf, "Be terse.", "Hi", true, true);
    try std.testing.expectEqualStrings(
        "<bos><|turn>system\nBe terse.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        buf.items,
    );

    // Later turn: close the previous model turn first.
    buf.clearRetainingCapacity();
    try t.renderTurn(allocator, &buf, "ignored on later turns", "Bye", false, true);
    try std.testing.expectEqualStrings(
        "<turn|>\n<|turn>user\nBye<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        buf.items,
    );

    // Thinking on, no system: the system turn still opens the think channel.
    buf.clearRetainingCapacity();
    try t.renderTurn(allocator, &buf, null, "Hi", true, false);
    try std.testing.expectEqualStrings(
        "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
        buf.items,
    );
}

test "renderMessages: chatml renders a full history and strips historical <think>" {
    const allocator = std.testing.allocator;
    const t = Template{ .format = .chatml };
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    const messages = [_]chat.Message{
        .{ .role = .system, .content = "You are helpful." },
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "<think>\nsome reasoning\n</think>\n\nHello!" },
        .{ .role = .user, .content = "Bye" },
    };
    try t.renderMessages(allocator, &buf, &messages, true);
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nYou are helpful.<|im_end|>\n" ++
            "<|im_start|>user\nHi<|im_end|>\n" ++
            "<|im_start|>assistant\nHello!<|im_end|>\n" ++
            "<|im_start|>user\nBye<|im_end|>\n" ++
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        buf.items,
    );

    // First-turn render matches renderTurn byte-for-byte (same KV prefix
    // whether a conversation is driven incrementally or stateless).
    buf.clearRetainingCapacity();
    try t.renderMessages(allocator, &buf, &.{
        .{ .role = .system, .content = "You are helpful." },
        .{ .role = .user, .content = "Hi" },
    }, false);
    var turn_buf: std.ArrayList(u8) = .empty;
    defer turn_buf.deinit(allocator);
    try t.renderTurn(allocator, &turn_buf, "You are helpful.", "Hi", true, false);
    try std.testing.expectEqualStrings(turn_buf.items, buf.items);
}

test "renderMessages: gemma4 merges leading systems, strips thought channel, primes think-off" {
    const allocator = std.testing.allocator;
    const t = Template{ .format = .gemma4 };
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    const messages = [_]chat.Message{
        .{ .role = .system, .content = "Be terse." },
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "<|channel>thought\nhm\n<channel|>Hello." },
        .{ .role = .user, .content = "Bye" },
    };
    try t.renderMessages(allocator, &buf, &messages, true);
    try std.testing.expectEqualStrings(
        "<bos><|turn>system\nBe terse.<turn|>\n" ++
            "<|turn>user\nHi<turn|>\n" ++
            "<|turn>model\nHello.<turn|>\n" ++
            "<|turn>user\nBye<turn|>\n" ++
            "<|turn>model\n<|channel>thought\n<channel|>",
        buf.items,
    );

    // Thinking on, no system: same conversation-start shape as renderTurn.
    buf.clearRetainingCapacity();
    try t.renderMessages(allocator, &buf, &.{.{ .role = .user, .content = "Hi" }}, false);
    try std.testing.expectEqualStrings(
        "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
        buf.items,
    );
}

test "renderMessages: llama3 blocks, gemma system fold, and validation errors" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try (Template{ .format = .llama3 }).renderMessages(allocator, &buf, &.{
        .{ .role = .system, .content = "S" },
        .{ .role = .user, .content = "U" },
    }, false);
    try std.testing.expectEqualStrings(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|>" ++
            "<|start_header_id|>user<|end_header_id|>\n\nU<|eot_id|>" ++
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        buf.items,
    );

    // Gemma 1-3 has no system role: leading system folds into the first user turn.
    buf.clearRetainingCapacity();
    try (Template{ .format = .gemma }).renderMessages(allocator, &buf, &.{
        .{ .role = .system, .content = "S" },
        .{ .role = .user, .content = "U" },
    }, false);
    try std.testing.expectEqualStrings(
        "<start_of_turn>user\nS\n\nU<end_of_turn>\n<start_of_turn>model\n",
        buf.items,
    );

    const t = Template{ .format = .chatml };
    try std.testing.expectError(error.EmptyMessages, t.renderMessages(allocator, &buf, &.{}, false));
    try std.testing.expectError(error.TrailingAssistantMessage, t.renderMessages(allocator, &buf, &.{
        .{ .role = .assistant, .content = "half a reply" },
    }, false));
    try std.testing.expectError(error.SystemMidConversation, (Template{ .format = .gemma4 }).renderMessages(allocator, &buf, &.{
        .{ .role = .user, .content = "U" },
        .{ .role = .system, .content = "S" },
    }, false));
}

test "sendRendered over renderMessages == incremental multi-turn send (greedy)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    // Constrain replies to single-symbol tokens (letters + space + stop) so
    // reply text re-encodes to the exact ids the reference run generated —
    // the tiny 64-id model can otherwise emit ids the 14-symbol test vocab
    // cannot round-trip through the stateless re-render.
    const allowed = [_]usize{ 0, 2, 4, 6, 8, 10, 13 };

    // Reference: a 2-turn incremental conversation.
    var mask = CountingMask{ .allowed = &allowed, .allocator = allocator };
    defer mask.deinit();
    var convo = try Conversation.init(&ctx, &model, &tok, tmpl, .{
        .capacity = 256,
        .max_response_tokens = 12,
        .logit_processor = mask.processor(),
    });
    defer convo.deinit();
    var reply1 = std.Io.Writer.Allocating.init(allocator);
    defer reply1.deinit();
    _ = try convo.send("ab a", &reply1.writer);
    var reply2 = std.Io.Writer.Allocating.init(allocator);
    defer reply2.deinit();
    const produced2 = try convo.send("ba b", &reply2.writer);

    // Stateless: a FRESH conversation fed the full history in one render.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try tmpl.renderMessages(allocator, &buf, &.{
        .{ .role = .user, .content = "ab a" },
        .{ .role = .assistant, .content = reply1.written() },
        .{ .role = .user, .content = "ba b" },
    }, false);

    var fresh_mask = CountingMask{ .allowed = &allowed, .allocator = allocator };
    defer fresh_mask.deinit();
    var fresh = try Conversation.init(&ctx, &model, &tok, tmpl, .{
        .capacity = 256,
        .max_response_tokens = 12,
        .logit_processor = fresh_mask.processor(),
    });
    defer fresh.deinit();
    var reply2b = std.Io.Writer.Allocating.init(allocator);
    defer reply2b.deinit();
    const produced2b = try fresh.sendRendered(buf.items, &reply2b.writer);

    // Same rendered prefix bytes => same prefill => same greedy reply.
    try std.testing.expectEqual(produced2, produced2b);
    try std.testing.expectEqualStrings(reply2.written(), reply2b.written());
    try std.testing.expectEqualSlices(usize, convo.history.items, fresh.history.items);
}

test "sendRenderedReuse: warm slot reuses the prefix, matches a fresh stateless run (greedy)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    // Single-symbol replies round-trip through the stateless re-render
    // (the sendRendered-equivalence test's vocabulary trick).
    const allowed = [_]usize{ 0, 2, 4, 6, 8, 10, 13 };

    // Request 1 on a cold slot: nothing to reconcile against — the reuse
    // entry must behave exactly like sendRendered.
    var buf1: std.ArrayList(u8) = .empty;
    defer buf1.deinit(allocator);
    try tmpl.renderMessages(allocator, &buf1, &.{
        .{ .role = .user, .content = "ab a" },
    }, false);

    var mask1 = CountingMask{ .allowed = &allowed, .allocator = allocator };
    defer mask1.deinit();
    var convo1 = try Conversation.init(&ctx, &model, &tok, tmpl, .{
        .capacity = 256,
        .max_response_tokens = 12,
        .logit_processor = mask1.processor(),
    });
    defer convo1.deinit();
    var reply1 = std.Io.Writer.Allocating.init(allocator);
    defer reply1.deinit();
    _ = try convo1.sendRenderedReuse(buf1.items, &reply1.writer);
    try std.testing.expectEqual(@as(usize, 0), convo1.reused_prefix);

    // The server epilogue: token shadow + cache leave the conversation.
    const shadow = try allocator.dupe(usize, convo1.history.items[0..convo1.cache.len]);
    defer allocator.free(shadow);
    const slot_cache = convo1.takeCache();

    // Request 2 extends the conversation (stateless full-history render).
    var buf2: std.ArrayList(u8) = .empty;
    defer buf2.deinit(allocator);
    try tmpl.renderMessages(allocator, &buf2, &.{
        .{ .role = .user, .content = "ab a" },
        .{ .role = .assistant, .content = reply1.written() },
        .{ .role = .user, .content = "ba b" },
    }, false);

    // Reference: a fresh conversation over the same render.
    var mask_f = CountingMask{ .allowed = &allowed, .allocator = allocator };
    defer mask_f.deinit();
    var fresh = try Conversation.init(&ctx, &model, &tok, tmpl, .{
        .capacity = 256,
        .max_response_tokens = 12,
        .logit_processor = mask_f.processor(),
    });
    defer fresh.deinit();
    var reply_f = std.Io.Writer.Allocating.init(allocator);
    defer reply_f.deinit();
    const produced_f = try fresh.sendRendered(buf2.items, &reply_f.writer);

    // Warm: adopt the slot state and reuse the common prefix.
    var mask_w = CountingMask{ .allowed = &allowed, .allocator = allocator };
    defer mask_w.deinit();
    var warm = try Conversation.initWarm(&ctx, &model, &tok, tmpl, .{
        .capacity = 256,
        .max_response_tokens = 12,
        .logit_processor = mask_w.processor(),
    }, .{ .cache = slot_cache, .tokens = shadow });
    defer warm.deinit();
    var reply_w = std.Io.Writer.Allocating.init(allocator);
    defer reply_w.deinit();
    const produced_w = try warm.sendRenderedReuse(buf2.items, &reply_w.writer);

    // The reconcile reused exactly the manually computed common prefix...
    const ids2 = try tok.encodeRaw(allocator, buf2.items);
    defer allocator.free(ids2);
    var lcp: usize = 0;
    while (lcp < @min(shadow.len, ids2.len - 1) and shadow[lcp] == ids2[lcp]) : (lcp += 1) {}
    try std.testing.expect(lcp > 0);
    try std.testing.expectEqual(lcp, warm.reused_prefix);

    // ...and decode after reuse == the fresh full prefill, byte for byte.
    try std.testing.expectEqual(produced_f, produced_w);
    try std.testing.expectEqualStrings(reply_f.written(), reply_w.written());
    try std.testing.expectEqualSlices(usize, fresh.history.items, warm.history.items);
}

test "sendRenderedReuse: identical resend and divergent edits reconcile; speculation is rejected" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };
    const allowed = [_]usize{ 0, 2, 4, 6, 8, 10, 13 };

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try tmpl.renderMessages(allocator, &buf, &.{
        .{ .role = .user, .content = "ab a" },
    }, false);
    const ids = try tok.encodeRaw(allocator, buf.items);
    defer allocator.free(ids);

    var mask = CountingMask{ .allowed = &allowed, .allocator = allocator };
    defer mask.deinit();
    var convo = try Conversation.init(&ctx, &model, &tok, tmpl, .{
        .capacity = 256,
        .max_response_tokens = 8,
        .logit_processor = mask.processor(),
    });
    defer convo.deinit();
    var reply1 = std.Io.Writer.Allocating.init(allocator);
    defer reply1.deinit();
    const produced1 = try convo.sendRenderedReuse(buf.items, &reply1.writer);

    // Identical resend (a client retry), via the pre-tokenized entry:
    // everything but the always-re-forwarded last prompt token is reused,
    // the reply overwrites the rewound one — greedy: identical bytes.
    var reply2 = std.Io.Writer.Allocating.init(allocator);
    defer reply2.deinit();
    const produced2 = try convo.sendTokensReuse(ids, &reply2.writer);
    try std.testing.expectEqual(ids.len - 1, convo.reused_prefix);
    try std.testing.expectEqual(produced1, produced2);
    try std.testing.expectEqualStrings(reply1.written(), reply2.written());

    // A divergent render (edited history): reuse stops at the divergence.
    const pre_edit = try allocator.dupe(usize, convo.history.items[0..convo.cache.len]);
    defer allocator.free(pre_edit);
    var buf_edit: std.ArrayList(u8) = .empty;
    defer buf_edit.deinit(allocator);
    try tmpl.renderMessages(allocator, &buf_edit, &.{
        .{ .role = .user, .content = "ab u" },
    }, false);
    const ids_edit = try tok.encodeRaw(allocator, buf_edit.items);
    defer allocator.free(ids_edit);
    var lcp: usize = 0;
    while (lcp < @min(pre_edit.len, ids_edit.len - 1) and pre_edit[lcp] == ids_edit[lcp]) : (lcp += 1) {}
    var reply3 = std.Io.Writer.Allocating.init(allocator);
    defer reply3.deinit();
    _ = try convo.sendRenderedReuse(buf_edit.items, &reply3.writer);
    try std.testing.expect(lcp > 0);
    try std.testing.expect(lcp < ids_edit.len - 1);
    try std.testing.expectEqual(lcp, convo.reused_prefix);
    // Post-reconcile history = the edited render's ids (+ the new reply).
    for (ids_edit, convo.history.items[0..ids_edit.len]) |expect_id, got| {
        try std.testing.expectEqual(@as(usize, expect_id), got);
    }

    // Speculation cannot host the reuse rewind...
    var spec_convo = try Conversation.init(&ctx, &model, &tok, tmpl, .{
        .capacity = 64,
        .speculation = true,
    });
    defer spec_convo.deinit();
    var sink_buf: [16]u8 = undefined;
    var sink = std.Io.Writer.fixed(&sink_buf);
    try std.testing.expectError(error.SpeculationWithReuse, spec_convo.sendRenderedReuse(buf.items, &sink));

    // ...nor adopt a warm slot at init (the adopted cache is freed on the
    // error path — the leak check would catch it otherwise).
    const orphan = try model.initKvCache(&ctx, 64);
    try std.testing.expectError(error.SpeculationWithWarmStart, Conversation.initWarm(&ctx, &model, &tok, tmpl, .{
        .capacity = 64,
        .speculation = true,
    }, .{ .cache = orphan, .tokens = &.{} }));
}

test "Conversation instantiates over gemma4 + SPM (compile coverage)" {
    // The generic covers both in-tree families; force semantic analysis of
    // every Conversation path against the gemma4/SPM duck-typed signatures.
    std.testing.refAllDecls(chat.Conversation(@import("gemma/gemma4.zig").Model, @import("spm_tokenizer.zig")));
}

/// Minimal byte-level vocab for the tiny 64-token model: GPT-2 symbol forms
/// ("Ġ" = space, "Ċ" = newline) + the ChatML structural markers. No merges:
/// regular text encodes to single-symbol tokens. All ids < 64.
const tiny_vocab = [_][]const u8{
    "a", "b", "u", "s", "e", "r", "i", "t", "n", "m", "\xC4\xA0", "\xC4\x8A", "<|im_start|>", "<|im_end|>",
};

/// Run a 3-turn conversation twice (speculation off/on) and assert the
/// committed token streams, reply bytes, and per-turn counts are IDENTICAL.
/// The sampler is PERSISTENT across turns, so in sampled mode this is the
/// RNG-draw-parity proof: a single extra draw consumed by the speculative
/// path (e.g. verify rows sampled past the stop marker or the response
/// budget) desyncs every later turn from the plain run.
fn expectSpecEquivalentConversation(sampler_cfg: sampler_mod.Config) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    const turns = [_][]const u8{ "ab a", "ba b", "us it" };
    const Result = struct {
        history: []usize,
        text: []u8,
        produced: [turns.len]usize,
    };
    var results: [2]Result = undefined;
    for (0..2) |which| {
        var convo = try Conversation.init(&ctx, &model, &tok, tmpl, .{
            .capacity = 256,
            .max_response_tokens = 12,
            .sampler = sampler_cfg,
            .speculation = which == 1,
        });
        defer convo.deinit();
        try std.testing.expectEqual(@as(?u32, 13), convo.stop_id); // "<|im_end|>"

        var aw = std.Io.Writer.Allocating.init(allocator);
        defer aw.deinit();
        var produced: [turns.len]usize = undefined;
        for (turns, 0..) |msg, ti| produced[ti] = try convo.send(msg, &aw.writer);
        results[which] = .{
            .history = try allocator.dupe(usize, convo.history.items),
            .text = try allocator.dupe(u8, aw.written()),
            .produced = produced,
        };
        // Post-turn cache/history relationship is consistent for the NEXT turn.
        try std.testing.expect(convo.cache.len <= convo.history.items.len);
        if (which == 1) {
            const stats = convo.specStats().?;
            try std.testing.expect(stats.steps > 0);
            // Committed counts include any trimmed turn-boundary tokens.
            try std.testing.expect(stats.committed >= produced[0] + produced[1] + produced[2]);
        }
    }
    defer for (&results) |r| {
        allocator.free(r.history);
        allocator.free(r.text);
    };

    // Token-for-token and byte-for-byte identical, turn by turn.
    for (results[0].produced, results[1].produced) |p0, p1| try std.testing.expectEqual(p0, p1);
    try std.testing.expectEqualSlices(usize, results[0].history, results[1].history);
    try std.testing.expectEqualStrings(results[0].text, results[1].text);
}

test "3-turn conversation: speculation on == off (greedy)" {
    try expectSpecEquivalentConversation(.{});
}

test "3-turn conversation: speculation on == off (sampled, fixed seed: draw-for-draw parity)" {
    try expectSpecEquivalentConversation(.{ .temperature = 0.7, .top_k = 20, .seed = 42 });
}

/// Token-mask logit processor test double for the constrained-conversation
/// proofs: only ids in `allowed` stay finite, every selected token and every
/// turn re-arm is recorded.
const CountingMask = struct {
    allowed: []const usize,
    allocator: std.mem.Allocator,
    commits: std.ArrayList(usize) = .empty,
    resets: usize = 0,

    fn deinit(self: *CountingMask) void {
        self.commits.deinit(self.allocator);
    }

    fn processor(self: *CountingMask) sampler_mod.LogitProcessor {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable = sampler_mod.LogitProcessor.VTable{
        .process = process,
        .commit = commit,
        .reset = reset,
    };

    fn process(ptr: *anyopaque, logits: []f32, history: []const usize) anyerror!void {
        _ = history;
        const self: *CountingMask = @ptrCast(@alignCast(ptr));
        outer: for (logits, 0..) |*l, tok| {
            for (self.allowed) |a| if (a == tok) continue :outer;
            l.* = -std.math.inf(f32);
        }
    }

    fn commit(ptr: *anyopaque, token: usize) anyerror!void {
        const self: *CountingMask = @ptrCast(@alignCast(ptr));
        try self.commits.append(self.allocator, token);
    }

    fn reset(ptr: *anyopaque) anyerror!void {
        const self: *CountingMask = @ptrCast(@alignCast(ptr));
        self.resets += 1;
    }
};

/// A masking logit processor constrains every reply token on the plain AND
/// the speculative path identically: same streams, same per-selection commit
/// sequence (the processor-state parity that makes grammar constraints
/// speculation-safe), one reset per turn, and no reply token outside the
/// mask. Exercised greedy and sampled (persistent-RNG draw parity).
fn expectConstrainedConversation(sampler_cfg: sampler_mod.Config) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    // The mask must include the stop id (13) so turns can end the plain way.
    const allowed = [_]usize{ 0, 2, 4, 6, 8, 13 };
    const turns = [_][]const u8{ "ab a", "ba b" };

    var streams: [2][]u8 = undefined;
    var commit_logs: [2][]usize = undefined;
    for (0..2) |which| {
        var mask = CountingMask{ .allowed = &allowed, .allocator = allocator };
        defer mask.deinit();
        var convo = try Conversation.init(&ctx, &model, &tok, tmpl, .{
            .capacity = 256,
            .max_response_tokens = 10,
            .sampler = sampler_cfg,
            .logit_processor = mask.processor(),
            .speculation = which == 1,
        });
        defer convo.deinit();

        var aw = std.Io.Writer.Allocating.init(allocator);
        defer aw.deinit();
        for (turns) |msg| _ = try convo.send(msg, &aw.writer);

        try std.testing.expectEqual(turns.len, mask.resets); // one re-arm per turn
        for (mask.commits.items) |t| {
            var ok = false;
            for (allowed) |a| ok = ok or a == t;
            try std.testing.expect(ok); // nothing outside the mask was ever selected
        }
        streams[which] = try allocator.dupe(u8, aw.written());
        commit_logs[which] = try allocator.dupe(usize, mask.commits.items);
    }
    defer for (streams, commit_logs) |s, cl| {
        allocator.free(s);
        allocator.free(cl);
    };

    try std.testing.expectEqualStrings(streams[0], streams[1]);
    try std.testing.expectEqualSlices(usize, commit_logs[0], commit_logs[1]);
}

test "constrained conversation: mask holds, plain == speculative (greedy)" {
    try expectConstrainedConversation(.{});
}

test "constrained conversation: mask holds, plain == speculative (sampled, fixed seed)" {
    try expectConstrainedConversation(.{ .temperature = 0.9, .top_k = 8, .seed = 1234 });
}

/// Structural-hook logit processor: every reply must start with the `forced`
/// token span (the mask allows exactly the next forced token while inside
/// it), then any of `allowed`. `forcedTokens`/`validPrefixLen` expose that
/// structure, so speculation wraps it in a `ConstrainedSource` — the grammar
/// preamble drafts itself.
const StructuredMask = struct {
    forced: []const usize,
    allowed: []const usize,
    /// Reply position = commits since the last reset.
    pos: usize = 0,
    resets: usize = 0,

    fn processor(self: *StructuredMask) sampler_mod.LogitProcessor {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable = sampler_mod.LogitProcessor.VTable{
        .process = process,
        .commit = commit,
        .reset = reset,
        .forcedTokens = forcedTokens,
        .validPrefixLen = validPrefixLen,
    };

    fn allowedAt(self: *const StructuredMask, pos: usize, token: usize) bool {
        if (pos < self.forced.len) return token == self.forced[pos];
        for (self.allowed) |a| if (a == token) return true;
        return false;
    }

    fn process(ptr: *anyopaque, logits: []f32, history: []const usize) anyerror!void {
        _ = history;
        const self: *StructuredMask = @ptrCast(@alignCast(ptr));
        for (logits, 0..) |*l, tok| {
            if (!self.allowedAt(self.pos, tok)) l.* = -std.math.inf(f32);
        }
    }

    fn commit(ptr: *anyopaque, token: usize) anyerror!void {
        _ = token;
        const self: *StructuredMask = @ptrCast(@alignCast(ptr));
        self.pos += 1;
    }

    fn reset(ptr: *anyopaque) anyerror!void {
        const self: *StructuredMask = @ptrCast(@alignCast(ptr));
        self.pos = 0;
        self.resets += 1;
    }

    fn forcedTokens(ptr: *anyopaque, buf: []usize) usize {
        const self: *StructuredMask = @ptrCast(@alignCast(ptr));
        if (self.pos >= self.forced.len) return 0;
        const rest = self.forced[self.pos..];
        const n = @min(rest.len, buf.len);
        @memcpy(buf[0..n], rest[0..n]);
        return n;
    }

    fn validPrefixLen(ptr: *anyopaque, tokens: []const usize) usize {
        const self: *StructuredMask = @ptrCast(@alignCast(ptr));
        for (tokens, 0..) |t, i| {
            if (!self.allowedAt(self.pos + i, t)) return i;
        }
        return tokens.len;
    }
};

/// A processor with structural hooks routes speculation through the
/// grammar-aware source: the forced reply preamble drafts itself and is
/// accepted with certainty, while outputs stay token-for-token identical to
/// the constrained plain run (greedy and sampled).
fn expectGrammarDraftedConversation(sampler_cfg: sampler_mod.Config) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    const forced = [_]usize{ 0, 2, 4 }; // >= the decoder's default min_draft
    const allowed = [_]usize{ 0, 2, 4, 6, 8, 13 }; // includes the stop id
    const turns = [_][]const u8{ "ab a", "ba b" };

    var streams: [2][]u8 = undefined;
    for (0..2) |which| {
        var mask = StructuredMask{ .forced = &forced, .allowed = &allowed };
        var convo = try Conversation.init(&ctx, &model, &tok, tmpl, .{
            .capacity = 256,
            .max_response_tokens = 10,
            .sampler = sampler_cfg,
            .logit_processor = mask.processor(),
            .speculation = which == 1,
        });
        defer convo.deinit();
        // Speculation + structural hooks = the grammar-aware source.
        if (which == 1) try std.testing.expect(convo.spec.?.grammar_source != null);

        var aw = std.Io.Writer.Allocating.init(allocator);
        defer aw.deinit();
        for (turns) |msg| _ = try convo.send(msg, &aw.writer);
        streams[which] = try allocator.dupe(u8, aw.written());

        if (which == 1) {
            // Every turn's forced preamble was drafted by the grammar and
            // accepted with certainty (the masked sampler cannot disagree).
            const stats = convo.specStats().?;
            try std.testing.expect(stats.accepted >= turns.len * forced.len);
        }
    }
    defer for (streams) |s| allocator.free(s);
    try std.testing.expectEqualStrings(streams[0], streams[1]);
}

test "grammar-drafted conversation: forced spans accepted, plain == speculative (greedy)" {
    try expectGrammarDraftedConversation(.{});
}

test "grammar-drafted conversation: forced spans accepted, plain == speculative (sampled)" {
    try expectGrammarDraftedConversation(.{ .temperature = 0.9, .top_k = 8, .seed = 77 });
}

test "sendBatch: per-stream logit processors match individual constrained sends" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    // Distinct per-stream constraints (n = 2 stays below every m-dependent
    // kernel threshold, so lockstep decode is bitwise vs sequential).
    const allowed_sets = [2][]const usize{ &.{ 0, 2, 4, 6, 13 }, &.{ 1, 3, 5, 7, 13 } };
    const users = [2][]const u8{ "ab a", "ba b" };

    var streams: [2][2][]u8 = undefined;
    for (0..2) |which| {
        var masks: [2]CountingMask = undefined;
        for (&masks, allowed_sets) |*m, set| m.* = .{ .allowed = set, .allocator = allocator };
        defer for (&masks) |*m| m.deinit();

        var convos: [2]Conversation = undefined;
        var inited: usize = 0;
        defer for (0..inited) |i| convos[i].deinit();
        for (0..2) |i| {
            convos[i] = try Conversation.init(&ctx, &model, &tok, tmpl, .{
                .capacity = 256,
                .max_response_tokens = 8,
                .sampler = .{ .temperature = 0.7, .top_k = 8, .seed = 42 + i },
                .logit_processor = masks[i].processor(),
            });
            inited += 1;
        }

        var aws: [2]std.Io.Writer.Allocating = undefined;
        for (&aws) |*aw| aw.* = std.Io.Writer.Allocating.init(allocator);
        defer for (&aws) |*aw| aw.deinit();

        if (which == 0) {
            for (0..2) |i| _ = try convos[i].send(users[i], &aws[i].writer);
        } else {
            var convo_ptrs: [2]*Conversation = .{ &convos[0], &convos[1] };
            var writer_ptrs: [2]*std.Io.Writer = .{ &aws[0].writer, &aws[1].writer };
            var users_slice: [2][]const u8 = users;
            var produced: [2]usize = undefined;
            try Conversation.sendBatch(&convo_ptrs, &users_slice, &writer_ptrs, &produced);
        }

        // Each stream's selections honored ITS mask (no cross-stream bleed).
        for (masks, allowed_sets) |m, set| {
            for (m.commits.items) |t| {
                var ok = false;
                for (set) |a| ok = ok or a == t;
                try std.testing.expect(ok);
            }
        }
        for (0..2) |i| streams[which][i] = try allocator.dupe(u8, aws[i].written());
    }
    defer for (&streams) |*mode| for (mode.*) |s| allocator.free(s);
    for (0..2) |i| try std.testing.expectEqualStrings(streams[0][i], streams[1][i]);
}

test "sendBatch rejects a logit processor shared between streams" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    var mask = CountingMask{ .allowed = &.{ 0, 13 }, .allocator = allocator };
    defer mask.deinit();

    var a = try Conversation.init(&ctx, &model, &tok, tmpl, .{ .logit_processor = mask.processor() });
    defer a.deinit();
    var b = try Conversation.init(&ctx, &model, &tok, tmpl, .{ .logit_processor = mask.processor() });
    defer b.deinit();

    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var convos: [2]*Conversation = .{ &a, &b };
    var users: [2][]const u8 = .{ "a", "b" };
    var writers: [2]*std.Io.Writer = .{ &w, &w };
    var produced: [2]usize = undefined;
    try std.testing.expectError(error.SharedBatchProcessor, Conversation.sendBatch(&convos, &users, &writers, &produced));
}

test "spec turn: writer failure mid-turn leaves the conversation consistent; the next send works" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };
    const options = Options{ .capacity = 256, .max_response_tokens = 12, .speculation = true };

    // Reference: the same first turn streams enough bytes that a 2-byte
    // writer must fail mid-turn (keeps the failure injection non-vacuous).
    {
        var ref = try Conversation.init(&ctx, &model, &tok, tmpl, options);
        defer ref.deinit();
        var aw = std.Io.Writer.Allocating.init(allocator);
        defer aw.deinit();
        _ = try ref.send("ab a", &aw.writer);
        try std.testing.expect(aw.written().len > 2);
    }

    var convo = try Conversation.init(&ctx, &model, &tok, tmpl, options);
    defer convo.deinit();

    var small: [2]u8 = undefined;
    var fw = std.Io.Writer.fixed(&small);
    try std.testing.expectError(error.WriteFailed, convo.send("ab a", &fw));
    // The unconditional turn trim ran: history/KV in sync for the next turn.
    try std.testing.expect(convo.history.items.len >= convo.cache.len);
    try std.testing.expect(convo.history.items.len - convo.cache.len <= 1);

    // Resend works and the state stays consistent.
    var aw = std.Io.Writer.Allocating.init(allocator);
    defer aw.deinit();
    _ = try convo.send("ab a", &aw.writer);
    try std.testing.expect(convo.history.items.len >= convo.cache.len);
    try std.testing.expect(convo.history.items.len - convo.cache.len <= 1);
}

test "speculation config: accounting_min_draft aligned; stop sequences rejected" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    // The cascade's per-source accounting threshold follows the decoder's
    // min_draft (drafts below it are never verified).
    var convo = try Conversation.init(&ctx, &model, &tok, tmpl, .{
        .speculation = true,
        .spec_options = .{ .min_draft = 4 },
    });
    defer convo.deinit();
    try std.testing.expectEqual(@as(usize, 4), convo.spec.?.index.accounting_min_draft);
    try std.testing.expectEqual(@as(usize, 4), convo.spec.?.decoder.options.min_draft);

    // Text stop sequences cannot compose with the lossless spec contract.
    try std.testing.expectError(error.StopSequencesWithSpeculation, Conversation.init(&ctx, &model, &tok, tmpl, .{
        .speculation = true,
        .stop_sequences = &.{"stop"},
    }));
}

test "extra stop ids and text stop sequences end the turn before streaming" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    // Reference reply (no extra stops): non-empty text, so the stop cases
    // below are non-vacuous.
    var ref_text: std.ArrayList(u8) = .empty;
    defer ref_text.deinit(allocator);
    var ref_produced: usize = 0;
    {
        var convo = try Conversation.init(&ctx, &model, &tok, tmpl, .{ .capacity = 256, .max_response_tokens = 12 });
        defer convo.deinit();
        var aw = std.Io.Writer.Allocating.init(allocator);
        defer aw.deinit();
        ref_produced = try convo.send("ab a", &aw.writer);
        try std.testing.expect(ref_produced > 0);
        try std.testing.expect(aw.written().len > 0);
        try ref_text.appendSlice(allocator, aw.written());
    }

    // Every vocab id as an extra stop: the very first sample ends the turn.
    var all_ids: [64]u32 = undefined;
    for (&all_ids, 0..) |*id, i| id.* = @intCast(i);
    {
        var convo = try Conversation.init(&ctx, &model, &tok, tmpl, .{
            .capacity = 256,
            .max_response_tokens = 12,
            .extra_stop_ids = &all_ids,
        });
        defer convo.deinit();
        var aw = std.Io.Writer.Allocating.init(allocator);
        defer aw.deinit();
        try std.testing.expectEqual(@as(usize, 0), try convo.send("ab a", &aw.writer));
        try std.testing.expectEqual(@as(usize, 0), aw.written().len);
    }

    // The reference reply's first byte as a text stop sequence: generation
    // stops BEFORE the token completing it streams (tokens outside the tiny
    // vocab decode to nothing, so earlier ids may commit invisibly) — nothing
    // is written and the completing token is not committed.
    {
        var convo = try Conversation.init(&ctx, &model, &tok, tmpl, .{
            .capacity = 256,
            .max_response_tokens = 12,
            .stop_sequences = &.{ref_text.items[0..1]},
        });
        defer convo.deinit();
        var aw = std.Io.Writer.Allocating.init(allocator);
        defer aw.deinit();
        const produced = try convo.send("ab a", &aw.writer);
        try std.testing.expect(produced < ref_produced);
        try std.testing.expectEqual(@as(usize, 0), aw.written().len);
        try std.testing.expectEqual(convo.history.items.len, convo.cache.len);
    }
}

/// Run the same 2-turn, 3-stream conversation set twice — N sequential
/// `send`s vs lockstep `sendBatch` — and assert per-stream committed token
/// streams, reply bytes, and per-turn counts are IDENTICAL. n = 3 stays
/// below every m-dependent kernel threshold, so batching is bitwise; with
/// sampling on, each stream carries its own seeded sampler across turns,
/// making this the batch edition of the RNG-draw-parity proof.
fn expectBatchEquivalentConversations(base_sampler_cfg: sampler_mod.Config) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    const n = 3;
    const turns = [2][n][]const u8{
        .{ "ab a", "ba b", "us it" },
        .{ "it us", "a ab", "b ba" },
    };

    const StreamResult = struct {
        history: []usize,
        text: []u8,
        produced: [turns.len]usize,
    };
    var results: [2][n]StreamResult = undefined;

    for (0..2) |which| {
        var convos: [n]Conversation = undefined;
        var inited: usize = 0;
        defer for (0..inited) |i| convos[i].deinit();
        for (0..n) |i| {
            var sampler_cfg = base_sampler_cfg;
            sampler_cfg.seed = base_sampler_cfg.seed + i; // per-stream RNG
            convos[i] = try Conversation.init(&ctx, &model, &tok, tmpl, .{
                .capacity = 256,
                .max_response_tokens = 12,
                .sampler = sampler_cfg,
            });
            inited += 1;
        }

        var aws: [n]std.Io.Writer.Allocating = undefined;
        var aws_inited: usize = 0;
        defer for (0..aws_inited) |i| aws[i].deinit();
        for (0..n) |i| {
            aws[i] = std.Io.Writer.Allocating.init(allocator);
            aws_inited += 1;
        }

        var produced: [turns.len][n]usize = undefined;
        if (which == 0) {
            for (turns, 0..) |users, ti| {
                for (users, 0..) |user, i| produced[ti][i] = try convos[i].send(user, &aws[i].writer);
            }
        } else {
            var convo_ptrs: [n]*Conversation = undefined;
            for (&convo_ptrs, &convos) |*ptr, *convo| ptr.* = convo;
            var writer_ptrs: [n]*std.Io.Writer = undefined;
            for (&writer_ptrs, &aws) |*ptr, *aw| ptr.* = &aw.writer;
            for (turns, 0..) |users, ti| {
                var users_slice: [n][]const u8 = users;
                try Conversation.sendBatch(&convo_ptrs, &users_slice, &writer_ptrs, &produced[ti]);
            }
        }

        for (0..n) |i| {
            // Post-turn, every committed token is forwarded: the next turn
            // prefills from exactly `cache.len` (the plain-send invariant).
            try std.testing.expectEqual(convos[i].history.items.len, convos[i].cache.len);
            var per_stream: [turns.len]usize = undefined;
            for (0..turns.len) |ti| per_stream[ti] = produced[ti][i];
            results[which][i] = .{
                .history = try allocator.dupe(usize, convos[i].history.items),
                .text = try allocator.dupe(u8, aws[i].written()),
                .produced = per_stream,
            };
        }
    }
    defer for (&results) |*mode| for (mode) |r| {
        allocator.free(r.history);
        allocator.free(r.text);
    };

    for (0..n) |i| {
        try std.testing.expectEqualSlices(usize, results[0][i].history, results[1][i].history);
        try std.testing.expectEqualStrings(results[0][i].text, results[1][i].text);
        for (results[0][i].produced, results[1][i].produced) |p0, p1| try std.testing.expectEqual(p0, p1);
    }
}

test "2-turn, 3-stream: sendBatch == sequential sends (greedy)" {
    try expectBatchEquivalentConversations(.{});
}

test "2-turn, 3-stream: sendBatch == sequential sends (sampled, per-stream seeds)" {
    try expectBatchEquivalentConversations(.{ .temperature = 0.7, .top_k = 20, .seed = 42 });
}

test "sendBatch validates its conversation batch" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    var plain_a = try Conversation.init(&ctx, &model, &tok, tmpl, .{ .capacity = 64 });
    defer plain_a.deinit();
    var plain_b = try Conversation.init(&ctx, &model, &tok, tmpl, .{ .capacity = 64 });
    defer plain_b.deinit();
    var spec = try Conversation.init(&ctx, &model, &tok, tmpl, .{ .capacity = 64, .speculation = true });
    defer spec.deinit();

    var aw = std.Io.Writer.Allocating.init(allocator);
    defer aw.deinit();
    var writers = [_]*std.Io.Writer{ &aw.writer, &aw.writer };
    const users = [_][]const u8{ "a", "b" };
    var produced = [_]usize{ 0, 0 };

    var empty_convos = [_]*Conversation{};
    try std.testing.expectError(error.EmptyBatch, Conversation.sendBatch(&empty_convos, &.{}, &.{}, &.{}));

    var with_spec = [_]*Conversation{ &plain_a, &spec };
    try std.testing.expectError(error.SpeculationWithBatch, Conversation.sendBatch(&with_spec, &users, &writers, &produced));

    var duplicated = [_]*Conversation{ &plain_a, &plain_a };
    try std.testing.expectError(error.DuplicateBatchConversation, Conversation.sendBatch(&duplicated, &users, &writers, &produced));

    var pair = [_]*Conversation{ &plain_a, &plain_b };
    try std.testing.expectError(error.BatchLengthMismatch, Conversation.sendBatch(&pair, users[0..1], &writers, &produced));
}

test "sendBatch abort leaves every stream consistent and resendable" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 0xC4A7);
    defer model.deinit();
    var tok = try Tokenizer.initFromParts(allocator, &tiny_vocab, &.{}, .{});
    defer tok.deinit();
    const tmpl = Template{ .format = .chatml };

    var healthy = try Conversation.init(&ctx, &model, &tok, tmpl, .{ .capacity = 256, .max_response_tokens = 8 });
    defer healthy.deinit();
    // Capacity 2 cannot hold any rendered ChatML turn: this stream's
    // beginTurnTokens fails AFTER the healthy stream already committed its
    // first sampled token (phase 1 runs in batch order).
    var cramped = try Conversation.init(&ctx, &model, &tok, tmpl, .{ .capacity = 2 });
    defer cramped.deinit();

    var aw_healthy = std.Io.Writer.Allocating.init(allocator);
    defer aw_healthy.deinit();
    var aw_cramped = std.Io.Writer.Allocating.init(allocator);
    defer aw_cramped.deinit();

    var convos = [_]*Conversation{ &healthy, &cramped };
    const users = [_][]const u8{ "ab a", "ba b" };
    var writers = [_]*std.Io.Writer{ &aw_healthy.writer, &aw_cramped.writer };
    var produced = [_]usize{ 99, 99 };

    try std.testing.expectError(error.ContextFull, Conversation.sendBatch(&convos, &users, &writers, &produced));
    // produced is documented as unwritten on error.
    try std.testing.expectEqual(@as(usize, 99), produced[0]);

    // The abort trim restored history == cache for BOTH streams — the
    // healthy sibling's committed-but-unforwarded token was dropped.
    try std.testing.expectEqual(healthy.history.items.len, healthy.cache.len);
    try std.testing.expectEqual(cramped.history.items.len, cramped.cache.len);

    // The healthy conversation remains usable: a plain send completes and
    // keeps the post-turn invariant.
    const n = try healthy.send("us it", &aw_healthy.writer);
    try std.testing.expect(n <= 8);
    try std.testing.expectEqual(healthy.history.items.len, healthy.cache.len);
}
