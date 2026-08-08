//! DeepSeek V4 Flash backend adapter behind the lmserve `Backend` vtable.
//! Like qwen35 it does NOT ride the generic `GgufChatBackend`/`Conversation`:
//! the family runs on its own `Session` (CSA/HCA caches + step scratch), so
//! the KV-slot reuse tiers do not apply — each request prefills a fresh
//! session through chunked `llm.deepseek4.model.stepBatch` (the fused batch
//! MoE path fetches each routed expert once per layer per chunk — the whole
//! point with streamed expert weights) and decodes with `step`.
//!
//! Chat rendering is hand-rolled at the token level (`chat.Template` has no
//! deepseek arm): BOS, leading system messages as plain text, then
//! `<｜User｜>` / `<｜Assistant｜>` turns. Every assistant turn opens behind a
//! closed think block (the `</think>` token — thinking disabled, matching
//! the parity-validated examples/deepseek4 runner) and past assistant turns
//! close with the EOS token (`<｜end▁of▁sentence｜>`), so re-rendered history
//! reproduces exactly the token stream a served conversation generated.
//! Message content is encoded WITHOUT special-token resolution, so client
//! text cannot smuggle control tokens.
//!
//! Capabilities: client stop sequences, and grammar constraints when built
//! with `-Dllguidance=true` (the sampler's `LogitProcessor` seam; the mask
//! forces EOS once the grammar completes, so normal stop handling ends the
//! reply). No reasoning channel and no cross-request KV reuse.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const types = @import("types.zig");
const backend_mod = @import("backend.zig");

const Allocator = std.mem.Allocator;
const ds4 = llm.deepseek4.model;

/// Prefill chunk size (the runner's default): batches expert fetches per
/// layer without letting per-chunk scratch grow unbounded.
const prefill_chunk: usize = 128;

pub const Deepseek4Backend = struct {
    allocator: Allocator,
    ctx: *fucina.ExecContext,
    model: *ds4.Model,
    tokenizer: *llm.tokenizer.Tokenizer,
    stream: llm.tokenizer.StreamDecoder,
    constraints: backend_mod.ConstraintCache,
    model_id: []const u8,
    context_len: usize,
    default_sampling: llm.sampler.Config,
    /// Structural chat-token ids, resolved once at init (absence means this
    /// is not a deepseek4 chat GGUF; fail at startup, not per request).
    bos_id: u32,
    user_id: u32,
    assistant_id: u32,
    think_end_id: u32,
    eos_id: u32,

    pub fn init(
        allocator: Allocator,
        ctx: *fucina.ExecContext,
        model: *ds4.Model,
        tokenizer: *llm.tokenizer.Tokenizer,
        model_id: []const u8,
        context_len: usize,
        default_sampling: llm.sampler.Config,
    ) !Deepseek4Backend {
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .model = model,
            .tokenizer = tokenizer,
            .stream = llm.tokenizer.StreamDecoder.init(tokenizer),
            .constraints = backend_mod.ConstraintCache.init(allocator, 8),
            .model_id = model_id,
            .context_len = context_len,
            .default_sampling = default_sampling,
            .bos_id = tokenizer.bosId() orelse return error.MissingChatTokens,
            .user_id = tokenizer.tokenId("<｜User｜>") orelse return error.MissingChatTokens,
            .assistant_id = tokenizer.tokenId("<｜Assistant｜>") orelse return error.MissingChatTokens,
            .think_end_id = tokenizer.tokenId("</think>") orelse return error.MissingChatTokens,
            .eos_id = tokenizer.eosId() orelse return error.MissingChatTokens,
        };
    }

    pub fn deinit(self: *Deepseek4Backend) void {
        self.stream.deinit(self.allocator);
        self.constraints.deinit();
    }

    /// Render + tokenize the request into a prompt id list (token-level, no
    /// string template): structural markers by id, content by plain BPE.
    /// Caller owns the returned slice.
    fn buildPrompt(self: *const Deepseek4Backend, a: Allocator, req: *const types.GenerateRequest) ![]usize {
        const messages = req.messages;
        if (messages.len == 0) return error.EmptyMessages;
        if (messages[messages.len - 1].role == .assistant) return error.TrailingAssistantMessage;

        var out: std.ArrayList(usize) = .empty;
        errdefer out.deinit(a);
        var scratch: std.ArrayList(u32) = .empty;
        defer scratch.deinit(a);

        try out.append(a, self.bos_id);
        // Leading system messages: plain text between BOS and the first
        // user marker (the deepseek template's single system slot),
        // blank-line separated; one after the first non-system message
        // has no renderable place.
        var i: usize = 0;
        while (i < messages.len and messages[i].role == .system) : (i += 1) {
            if (i > 0) try self.appendText(a, &out, &scratch, "\n\n");
            try self.appendText(a, &out, &scratch, messages[i].content);
        }
        for (messages[i..]) |m| {
            switch (m.role) {
                .system => return error.SystemMidConversation,
                .user => {
                    try out.append(a, self.user_id);
                    try self.appendText(a, &out, &scratch, m.content);
                },
                .assistant => {
                    // Rendered exactly as it was generated: opened behind
                    // the closed think block, ended by EOS.
                    try out.append(a, self.assistant_id);
                    try out.append(a, self.think_end_id);
                    try self.appendText(a, &out, &scratch, m.content);
                    try out.append(a, self.eos_id);
                },
            }
        }
        // The generation prompt: the reference runner's closed think block.
        try out.append(a, self.assistant_id);
        try out.append(a, self.think_end_id);
        return out.toOwnedSlice(a);
    }

    /// Plain-BPE encode `text` (no BOS/EOS policy, no special-token
    /// resolution) and append its ids to `out`.
    fn appendText(
        self: *const Deepseek4Backend,
        a: Allocator,
        out: *std.ArrayList(usize),
        scratch: *std.ArrayList(u32),
        text: []const u8,
    ) !void {
        scratch.clearRetainingCapacity();
        try self.tokenizer.encodePlainAppend(a, text, scratch);
        for (scratch.items) |id| try out.append(a, id);
    }

    fn vtValidate(ptr: *anyopaque, req: *const types.GenerateRequest) anyerror!void {
        const self: *Deepseek4Backend = @ptrCast(@alignCast(ptr));
        if (req.constraint != null and !llm.llguidance.enabled) return error.LlguidanceNotEnabled;
        const ids = try self.buildPrompt(self.allocator, req);
        defer self.allocator.free(ids);
        if (ids.len >= self.context_len) return error.PromptTooLong;
    }

    fn vtGenerate(ptr: *anyopaque, req: *const types.GenerateRequest, sink: *std.Io.Writer) anyerror!types.GenerateResult {
        const self: *Deepseek4Backend = @ptrCast(@alignCast(ptr));
        const a = self.allocator;

        const ids = try self.buildPrompt(a, req);
        defer a.free(ids);

        // Grammar: clone the cached base per request (the base must outlive
        // the clone; the cache guarantees it).
        var clone: ?llm.llguidance.Constraint = null;
        defer if (clone) |*c| c.deinit();
        var processor: ?llm.sampler.LogitProcessor = null;
        if (req.constraint) |spec| {
            const base = try self.constraints.acquire(self.tokenizer, spec, .{
                .eos_token = self.eos_id,
                .n_vocab = self.model.config.vocab_size,
            });
            clone = try base.clone();
            processor = clone.?.processor();
        }

        // Fresh per-request session sized to the server's context budget.
        var session = try ds4.Session.init(self.model, self.context_len);
        defer session.deinit(self.model);

        // Chunked prefill. Logits are SESSION-OWNED (valid until the next
        // step on this session) — never freed here.
        var logits: []f32 = &.{};
        var fed: usize = 0;
        while (fed < ids.len) {
            const end = @min(fed + prefill_chunk, ids.len);
            logits = try ds4.stepBatch(self.model, self.ctx, &session, ids[fed..end]);
            fed = end;
        }

        return self.decodeLoop(req, sink, &session, &logits, processor, ids);
    }

    /// Sample/stream until a stop condition. `logits` holds the prefill's
    /// final row on entry and always points at owned-or-empty memory, so
    /// the caller's defer stays safe on every error path.
    fn decodeLoop(
        self: *Deepseek4Backend,
        req: *const types.GenerateRequest,
        sink: *std.Io.Writer,
        session: *ds4.Session,
        logits: *[]f32,
        processor: ?llm.sampler.LogitProcessor,
        ids: []const usize,
    ) anyerror!types.GenerateResult {
        const a = self.allocator;
        var sampler = llm.sampler.Sampler.init(req.sampling);
        sampler.processor = processor;
        // Whole-conversation token history (the repetition-penalty window).
        var history: std.ArrayList(usize) = .empty;
        defer history.deinit(a);
        try history.appendSlice(a, ids);
        // Accumulated decoded reply, only when stop sequences are in play.
        var reply: std.ArrayList(u8) = .empty;
        defer reply.deinit(a);

        self.stream.reset();
        var produced: usize = 0;
        var fired_stop: ?usize = null;
        var finish: types.FinishReason = .length;
        while (produced < req.max_tokens) {
            const next = blk: {
                var lt = try fucina.Tensor(.{ .seq, .vocab }).fromBorrowedSlice(self.ctx, .{ 1, self.model.config.vocab_size }, logits.*);
                defer lt.deinit();
                break :blk try sampler.next(self.ctx, &lt, history.items);
            };
            if (next == self.eos_id) {
                finish = .stop;
                break;
            }
            if (req.stop.len > 0) {
                const prev_len = reply.items.len;
                try self.tokenizer.decodeAppend(a, @intCast(next), &reply);
                // Stop BEFORE streaming the completing token.
                if (stopHitInTail(reply.items, prev_len, req.stop)) |fired| {
                    fired_stop = fired;
                    finish = .stop;
                    break;
                }
            }
            try self.stream.push(a, @intCast(next), sink);
            try sink.flush();
            try history.append(a, next);
            produced += 1;
            // Budget or capacity exhausted: break BEFORE the next forward —
            // its logits would never be sampled, and one deepseek4 forward
            // is real work (streamed experts).
            if (produced >= req.max_tokens or session.cache.len >= session.cache.capacity) break;
            logits.* = try ds4.step(self.model, self.ctx, session, next);
        }
        try self.stream.flush(sink);
        try sink.flush();

        return .{
            .prompt_tokens = ids.len,
            .completion_tokens = produced,
            .cached_tokens = 0,
            .stop_sequence = fired_stop,
            .finish = finish,
        };
    }

    pub fn backend(self: *Deepseek4Backend) types.Backend {
        return .{
            .ptr = self,
            .vtable = &.{ .validate = vtValidate, .generate = vtGenerate },
            .info = .{
                .model_id = self.model_id,
                .context_len = self.context_len,
                .caps = .{
                    .grammar = llm.llguidance.enabled,
                    .think = false,
                    .stop_sequences = true,
                },
                .tool_style = .none,
                .default_sampling = self.default_sampling,
            },
        };
    }
};

/// Stop-sequence scan for the bytes appended since `prev_len`, returning
/// the index of the sequence that fired (first match in list order — the
/// attribution the Anthropic dialect reports as `stop_sequence`). Same
/// windowed rescan as `chat.zig`'s private helper: a match wholly inside
/// `items[0..prev_len]` would already have ended the turn.
fn stopHitInTail(items: []const u8, prev_len: usize, needles: []const []const u8) ?usize {
    var max_len: usize = 0;
    for (needles) |n| max_len = @max(max_len, n.len);
    const window = items[prev_len -| (max_len -| 1) ..];
    for (needles, 0..) |n, i| {
        if (n.len > 0 and std.mem.indexOf(u8, window, n) != null) return i;
    }
    return null;
}

test stopHitInTail {
    const needles = [_][]const u8{ "END", "\n\n" };
    try std.testing.expectEqual(@as(?usize, null), stopHitInTail("hello", 4, &needles));
    // Needle completed by the newly appended byte, straddling prev_len.
    try std.testing.expectEqual(@as(?usize, 0), stopHitInTail("xxEND", 4, &needles));
    try std.testing.expectEqual(@as(?usize, 1), stopHitInTail("a\n\n", 1, &needles));
}
