//! Generic multi-turn chat on top of the model + tokenizer + KV cache + sampler.
//!
//! `Template` renders messages for the common GGUF chat formats (ChatML,
//! Llama 3, Gemma 1-3, Gemma 4), detected from the model's
//! `tokenizer.chat_template` metadata. `Conversation(Model, Tok)` is
//! comptime-generic over the model family and its tokenizer module (qwen3's
//! byte-BPE, gemma4's SPM, ...): KV-cache geometry comes from the model's own
//! `initKvCache`. One KV cache persists across turns (each turn only prefills
//! the new tokens) and the reply streams to any `*std.Io.Writer` sink
//! (stdout, an SSE response, an in-memory buffer, …).

const std = @import("std");
const fucina = @import("fucina");
const kv_cache = @import("kv_cache.zig");
const sampler_mod = @import("sampler.zig");
const speculative = @import("speculative/core.zig");
const spec_cascade = @import("speculative/cascade.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;
const Sampler = sampler_mod.Sampler;

pub const Format = enum { chatml, llama3, gemma, gemma4 };

pub const Template = struct {
    format: Format,

    /// Sniff the format from a GGUF `tokenizer.chat_template` string.
    pub fn detect(chat_template: ?[]const u8) ?Template {
        const t = chat_template orelse return null;
        if (std.mem.indexOf(u8, t, "<|im_start|>") != null) return .{ .format = .chatml };
        if (std.mem.indexOf(u8, t, "<|start_header_id|>") != null) return .{ .format = .llama3 };
        if (std.mem.indexOf(u8, t, "<|turn>") != null) return .{ .format = .gemma4 };
        if (std.mem.indexOf(u8, t, "<start_of_turn>") != null) return .{ .format = .gemma };
        return null;
    }

    /// The token text that ends an assistant turn (the generation stop marker).
    pub fn stopMarker(self: Template) []const u8 {
        return switch (self.format) {
            .chatml => "<|im_end|>",
            .llama3 => "<|eot_id|>",
            .gemma => "<end_of_turn>",
            .gemma4 => "<turn|>",
        };
    }

    /// Append the text to feed for one user turn. `first` emits the
    /// conversation-start (bos/system) tokens; otherwise it closes the previous
    /// assistant turn before opening the new one. `think_off` (ChatML and
    /// Gemma 4) suppresses the model's reasoning: an empty think block for
    /// ChatML, a primed-empty thought channel for Gemma 4.
    pub fn renderTurn(
        self: Template,
        allocator: Allocator,
        buf: *std.ArrayList(u8),
        system: ?[]const u8,
        user: []const u8,
        first: bool,
        think_off: bool,
    ) !void {
        switch (self.format) {
            .chatml => {
                if (!first) try buf.appendSlice(allocator, "<|im_end|>\n");
                if (first) if (system) |s| {
                    try buf.appendSlice(allocator, "<|im_start|>system\n");
                    try buf.appendSlice(allocator, s);
                    try buf.appendSlice(allocator, "<|im_end|>\n");
                };
                try buf.appendSlice(allocator, "<|im_start|>user\n");
                try buf.appendSlice(allocator, user);
                try buf.appendSlice(allocator, "<|im_end|>\n<|im_start|>assistant\n");
                if (think_off) try buf.appendSlice(allocator, "<think>\n\n</think>\n\n");
            },
            .llama3 => {
                if (!first) try buf.appendSlice(allocator, "<|eot_id|>");
                if (first) {
                    try buf.appendSlice(allocator, "<|begin_of_text|>");
                    if (system) |s| {
                        try buf.appendSlice(allocator, "<|start_header_id|>system<|end_header_id|>\n\n");
                        try buf.appendSlice(allocator, s);
                        try buf.appendSlice(allocator, "<|eot_id|>");
                    }
                }
                try buf.appendSlice(allocator, "<|start_header_id|>user<|end_header_id|>\n\n");
                try buf.appendSlice(allocator, user);
                try buf.appendSlice(allocator, "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
            },
            // Gemma 1-3 (`<start_of_turn>` markers); Gemma 4 GGUFs use the
            // `.gemma4` `<|turn>` format below.
            .gemma => {
                // Gemma has no system role — fold it into the first user turn.
                if (!first) try buf.appendSlice(allocator, "<end_of_turn>\n");
                try buf.appendSlice(allocator, "<start_of_turn>user\n");
                if (first) if (system) |s| {
                    try buf.appendSlice(allocator, s);
                    try buf.appendSlice(allocator, "\n\n");
                };
                try buf.appendSlice(allocator, user);
                try buf.appendSlice(allocator, "<end_of_turn>\n<start_of_turn>model\n");
            },
            // Gemma 4's `<|turn>` / `<turn|>` format (verified against the
            // GGUF's embedded chat_template, non-tool path): `<bos>` once, an
            // optional system turn (with `<|think|>` when thinking is on),
            // the user turn, then the model-turn opener. With thinking off
            // (the template default) the opener primes an empty thought
            // channel so the model goes straight to the answer.
            .gemma4 => {
                if (first) {
                    try buf.appendSlice(allocator, "<bos>");
                    if (!think_off or system != null) {
                        try buf.appendSlice(allocator, "<|turn>system\n");
                        if (!think_off) try buf.appendSlice(allocator, "<|think|>\n");
                        if (system) |s| try buf.appendSlice(allocator, std.mem.trim(u8, s, " \t\r\n"));
                        try buf.appendSlice(allocator, "<turn|>\n");
                    }
                } else {
                    // Close the previous model turn (generation stopped before
                    // its <turn|>, so emit it now ahead of the new user turn).
                    try buf.appendSlice(allocator, "<turn|>\n");
                }
                try buf.appendSlice(allocator, "<|turn>user\n");
                try buf.appendSlice(allocator, user);
                try buf.appendSlice(allocator, "<turn|>\n<|turn>model\n");
                if (think_off) try buf.appendSlice(allocator, "<|channel>thought\n<channel|>");
            },
        }
    }
};

pub const Options = struct {
    system: ?[]const u8 = null,
    /// Total KV-cache size; the whole conversation must fit.
    capacity: usize = 4096,
    /// Per-reply generation cap.
    max_response_tokens: usize = 1024,
    /// Suppress reasoning (ChatML: empty think block; Gemma 4: primed-empty
    /// thought channel).
    think_off: bool = false,
    sampler: sampler_mod.Config = .{},
    /// Extra token ids that also end the assistant turn, alongside the
    /// template's stop marker (e.g. Gemma's stray `<eos>`). Borrowed.
    extra_stop_ids: []const u32 = &.{},
    /// Text stop sequences: generation stops BEFORE streaming the token whose
    /// decoded reply text completes one of these. Borrowed. Incompatible with
    /// `speculation` — the completing token could be accepted mid-verify-batch,
    /// breaking the one-RNG-draw-per-plain-committed-token contract — so init
    /// fails loudly on the combination.
    stop_sequences: []const []const u8 = &.{},
    /// Optional logit processor (grammar/JSON-schema constrained decoding —
    /// `logit_processor.zig`, `llguidance.zig` — or any custom logit
    /// transform), installed on the conversation's sampler. Re-armed via its
    /// `reset` hook at every turn start, so the same constraint governs each
    /// assistant reply independently. Borrowed: the processor state must
    /// outlive the Conversation. Composes with `speculation` (the seam is
    /// speculative-safe — see `logit_processor.zig`).
    logit_processor: ?sampler_mod.LogitProcessor = null,
    /// Lossless draft-model-free speculative decoding (the SpeculationIndex
    /// cascade + SpeculativeDecoder). Off = the plain decode path, untouched.
    speculation: bool = false,
    /// Decoder tuning when `speculation` is on.
    spec_options: speculative.Options = .{},
    /// Clock for the decoder's live verify-cost gate (null = static
    /// cost-table economics).
    io: ?std.Io = null,
};

/// Duck-typed requirements: `Model` exposes `config.vocab_size`,
/// `initKvCache(ctx, capacity)` over the shared `KvCache`, and the
/// `forwardStep`/`forwardStepAllLogits` decode entries (the qwen3/gemma4
/// signatures — see `speculative.SpeculativeDecoder`). `Tok` is the tokenizer
/// MODULE (`tokenizer.zig`, `spm_tokenizer.zig`, ...): it provides `Tokenizer`
/// (`tokenId`/`eosId`/`encodeRaw`/`decodeAppend`) and its `StreamDecoder`.
pub fn Conversation(comptime Model: type, comptime Tok: type) type {
    return struct {
        const Self = @This();

        ctx: *ExecContext,
        model: *const Model,
        tokenizer: *const Tok.Tokenizer,
        template: Template,
        allocator: Allocator,
        cache: KvCache,
        stream: Tok.StreamDecoder,
        sampler: Sampler,
        history: std.ArrayList(usize),
        system: ?[]const u8,
        stop_id: ?u32,
        extra_stop_ids: []const u32,
        stop_sequences: []const []const u8,
        max_response_tokens: usize,
        think_off: bool,
        turn: usize,
        /// Speculative-decoding state (null = plain decode). Heap-allocated so
        /// the DraftSource/decoder pointers stay stable when Conversation moves.
        spec: ?*SpecState,

        const SpecState = struct {
            index: spec_cascade.SpeculationIndex,
            decoder: speculative.SpeculativeDecoder(Model),
        };

        pub fn init(ctx: *ExecContext, model: *const Model, tokenizer: *const Tok.Tokenizer, template: Template, options: Options) !Self {
            if (options.speculation and options.stop_sequences.len > 0) return error.StopSequencesWithSpeculation;
            const stop_id: ?u32 = tokenizer.tokenId(template.stopMarker()) orelse tokenizer.eosId();
            var cache = try model.initKvCache(ctx, options.capacity);
            errdefer cache.deinit();
            var spec: ?*SpecState = null;
            if (options.speculation) {
                const st = try ctx.allocator.create(SpecState);
                errdefer ctx.allocator.destroy(st);
                st.index = try spec_cascade.SpeculationIndex.init(ctx.allocator, model.config.vocab_size);
                errdefer st.index.deinit();
                // Stop-awareness (RNG/lossless contract): the decoder must not
                // sample verify rows past a committed stop marker — the plain
                // path stops there, and the persistent sampler would otherwise
                // run ahead of it for the rest of the conversation.
                var spec_options = options.spec_options;
                spec_options.stop_token = if (stop_id) |s| @as(usize, s) else null;
                // Acceptance accounting settles only drafts the decoder
                // actually verifies (cascade.zig's accounting contract).
                st.index.accounting_min_draft = spec_options.min_draft;
                st.decoder = try speculative.SpeculativeDecoder(Model).init(ctx.allocator, st.index.asDraftSource(), spec_options);
                st.decoder.io = options.io;
                spec = st;
            }
            return .{
                .ctx = ctx,
                .model = model,
                .tokenizer = tokenizer,
                .template = template,
                .allocator = ctx.allocator,
                .cache = cache,
                .stream = Tok.StreamDecoder.init(tokenizer),
                .sampler = blk: {
                    var s = Sampler.init(options.sampler);
                    s.processor = options.logit_processor;
                    break :blk s;
                },
                .history = .empty,
                .system = options.system,
                .stop_id = stop_id,
                .extra_stop_ids = options.extra_stop_ids,
                .stop_sequences = options.stop_sequences,
                .max_response_tokens = options.max_response_tokens,
                .think_off = options.think_off,
                .turn = 0,
                .spec = spec,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.spec) |st| {
                st.decoder.deinit();
                st.index.deinit();
                self.allocator.destroy(st);
            }
            self.history.deinit(self.allocator);
            self.stream.deinit(self.allocator);
            self.cache.deinit();
            self.* = undefined;
        }

        /// Inject a tokenized reference document into the speculation index (the
        /// RAG seam). Requires `speculation` enabled.
        pub fn addSpecReference(self: *Self, tokens: []const usize) !void {
            const st = self.spec orelse return error.SpeculationDisabled;
            try st.index.addReference(tokens);
        }

        /// The decoder's lifetime acceptance stats (null when speculation is off).
        pub fn specStats(self: *const Self) ?speculative.Stats {
            const st = self.spec orelse return null;
            return st.decoder.stats;
        }

        /// Render this turn's template text, tokenize it, and commit the
        /// prefix tokens to history — the shared turn prologue of `send`
        /// and `sendBatch`. Returns the caller-owned prefix slice.
        fn beginTurnTokens(self: *Self, user: []const u8) ![]usize {
            const a = self.allocator;

            // Re-arm the logit processor for this turn's reply (the previous
            // turn left it post-stop; grammar constraints re-apply per reply).
            if (self.sampler.processor) |p| try p.reset();

            var buf: std.ArrayList(u8) = .empty;
            defer buf.deinit(a);
            try self.template.renderTurn(a, &buf, self.system, user, self.turn == 0, self.think_off);
            self.turn += 1;

            const prefix32 = try self.tokenizer.encodeRaw(a, buf.items);
            defer a.free(prefix32);
            if (self.cache.len + prefix32.len > self.cache.capacity) return error.ContextFull;

            const prefix = try a.alloc(usize, prefix32.len);
            errdefer a.free(prefix);
            for (prefix, prefix32) |*d, s| d.* = s;
            try self.history.appendSlice(a, prefix);
            return prefix;
        }

        /// Send one user message; stream the assistant reply to `writer` (flushed
        /// per token). Returns the number of response tokens generated.
        pub fn send(self: *Self, user: []const u8, writer: *std.Io.Writer) !usize {
            const a = self.allocator;

            const prefix = try self.beginTurnTokens(user);
            defer a.free(prefix);

            if (self.spec) |st| return self.sendSpec(st, prefix, writer);

            // Prefill this turn's tokens at the current cache position.
            var logits = try self.model.forwardStep(self.ctx, &self.cache, prefix, self.cache.len);
            defer logits.deinit();

            // Accumulated decoded reply, only when text stop sequences are in play.
            var reply: std.ArrayList(u8) = .empty;
            defer reply.deinit(a);

            self.stream.reset();
            var produced: usize = 0;
            while (produced < self.max_response_tokens and self.cache.len < self.cache.capacity) {
                const next = try self.sampler.next(self.ctx, &logits, self.history.items);
                if (self.isStopToken(next)) break;
                if (self.stop_sequences.len > 0) {
                    const prev_len = reply.items.len;
                    try self.tokenizer.decodeAppend(a, @intCast(next), &reply);
                    // Stop before streaming the token that completes a stop sequence.
                    if (stopHitInTail(reply.items, prev_len, self.stop_sequences)) break;
                }
                try self.stream.push(a, @intCast(next), writer);
                try writer.flush();
                try self.history.append(a, next);
                produced += 1;
                var single = [_]usize{next};
                const fresh = try self.model.forwardStep(self.ctx, &self.cache, &single, self.cache.len);
                logits.deinit();
                logits = fresh;
            }
            try self.stream.flush(writer);
            try writer.flush();
            return produced;
        }

        /// Send one user message on EACH of `convos` — N sibling
        /// conversations over one shared model/context — decoding all
        /// streams in lockstep: every step forwards one token per live
        /// stream through `Model.forwardStepBatch` (one m=N weight pass
        /// instead of N GEMV passes), then samples each stream's next token
        /// from its own logits row with its own sampler/history. Per stream
        /// the semantics — prompt prefill, RNG-draw order (one draw per
        /// committed token), stop tokens, `stop_sequences`, budget and
        /// capacity handling, streaming — match a plain `send` exactly;
        /// below the m-dependent kernel thresholds (quantized-weight
        /// x4-packed kernels at N >= 4, fused FFN at N >= 12) the produced
        /// tokens are bit-identical to N sequential sends, beyond them rows
        /// can differ by reassociation drift (~1e-6 rel; same caveat as
        /// speculative verify batches). Turn prefills run per stream
        /// (batched prefill is future work).
        ///
        /// Requirements: all conversations share the same `ctx` and
        /// `model`, are distinct, and have speculation off (lockstep
        /// batching cannot host the decoder's ragged verify steps).
        /// `produced` receives per-stream reply token counts
        /// (unwritten if the batch errors). On error the batch aborts and
        /// every stream's history is trimmed back to its KV cache (the
        /// sendSpec turn-trim pattern), so each conversation — including
        /// healthy siblings of the failing stream — stays internally
        /// consistent and resendable; tokens already streamed to a writer
        /// before the abort are not recalled.
        pub fn sendBatch(
            convos: []const *Self,
            users: []const []const u8,
            writers: []const *std.Io.Writer,
            produced: []usize,
        ) !void {
            if (convos.len == 0) return error.EmptyBatch;
            if (users.len != convos.len or writers.len != convos.len or produced.len != convos.len) return error.BatchLengthMismatch;
            const first = convos[0];
            for (convos, 0..) |convo, i| {
                if (convo.spec != null) return error.SpeculationWithBatch;
                if (convo.ctx != first.ctx or convo.model != first.model) return error.MixedBatchModels;
                for (convos[0..i]) |prev| if (prev == convo) return error.DuplicateBatchConversation;
            }
            // Comptime-gated so model families without a batch entry (e.g.
            // gemma4 today) still compile the Conversation type; they get a
            // runtime error here instead.
            if (comptime @hasDecl(Model, "forwardStepBatch")) {
                return sendBatchImpl(convos, users, writers, produced);
            } else {
                return error.BatchDecodeUnsupported;
            }
        }

        /// Per-stream lockstep-decode state (`sendBatch`).
        const BatchStream = struct {
            convo: *Self,
            writer: *std.Io.Writer,
            /// Last committed token, not yet forwarded into the KV cache.
            last: usize = 0,
            produced: usize = 0,
            finished: bool = false,
            /// Accumulated decoded reply, only when the stream's
            /// `stop_sequences` are in play.
            reply: std.ArrayList(u8) = .empty,
        };

        fn sendBatchImpl(
            convos: []const *Self,
            users: []const []const u8,
            writers: []const *std.Io.Writer,
            produced: []usize,
        ) !void {
            const n = convos.len;
            const first = convos[0];
            const ctx = first.ctx;
            const a = first.allocator;

            const streams = try a.alloc(BatchStream, n);
            defer {
                for (streams) |*s| s.reply.deinit(a);
                a.free(streams);
            }
            for (streams, convos, writers) |*s, convo, writer| s.* = .{ .convo = convo, .writer = writer };

            // Abort consistency (the sendSpec unconditional-trim pattern):
            // an error stops the batch mid-flight, possibly leaving a stream
            // with a committed-but-unforwarded token — or an un-prefilled
            // turn prefix — in history. Trim every stream back to its cache;
            // a no-op on success (the lockstep loop's catch-up forward keeps
            // history.len == cache.len), and on error it returns every
            // conversation, healthy siblings included, to a resendable state
            // instead of a silent history/KV desync.
            defer for (streams) |*s| {
                s.convo.history.shrinkRetainingCapacity(s.convo.cache.len);
            };

            // Turn prologue + prefill + first sample, stream by stream: each
            // stream's sampler sees exactly the draw sequence of its own
            // plain `send` (prefill logits first, then one draw per token).
            for (streams, users) |*s, user| {
                const convo = s.convo;
                const prefix = try convo.beginTurnTokens(user);
                defer a.free(prefix);
                var logits = try convo.model.forwardStep(ctx, &convo.cache, prefix, convo.cache.len);
                defer logits.deinit();
                convo.stream.reset();
                try sampleStep(s, ctx, &logits);
            }

            const caches = try a.alloc(*KvCache, n);
            defer a.free(caches);
            const tokens = try a.alloc(usize, n);
            defer a.free(tokens);
            const active = try a.alloc(*BatchStream, n);
            defer a.free(active);

            // Lockstep decode: forward every live stream's committed-but-
            // unforwarded token in ONE batched pass (this also covers the
            // final catch-up forward a plain `send` issues after its last
            // committed token), then sample each stream's next token.
            while (true) {
                var n_active: usize = 0;
                for (streams) |*s| {
                    if (s.finished) continue;
                    active[n_active] = s;
                    caches[n_active] = &s.convo.cache;
                    tokens[n_active] = s.last;
                    n_active += 1;
                }
                if (n_active == 0) break;

                var logits = try first.model.forwardStepBatch(ctx, caches[0..n_active], tokens[0..n_active]);
                defer logits.deinit();
                for (active[0..n_active], 0..) |s, row_i| {
                    var row = try logits.narrow(ctx, .seq, row_i, 1);
                    defer row.deinit();
                    try sampleStep(s, ctx, &row);
                }
            }

            for (streams, produced) |*s, *count| {
                try s.convo.stream.flush(s.writer);
                try s.writer.flush();
                count.* = s.produced;
            }
        }

        /// One per-stream sampling step from `logits` (shape `[1, vocab]`,
        /// possibly a row view of the batch logits) — the exact body of
        /// `send`'s decode loop: budget/capacity gate, sample, stop checks,
        /// stream + history commit. Sets `finished` when the turn is over;
        /// otherwise leaves `last` holding a committed token that the next
        /// lockstep pass must forward.
        fn sampleStep(s: *BatchStream, ctx: *ExecContext, logits: *fucina.Tensor(.{ .seq, .vocab })) !void {
            const convo = s.convo;
            const a = convo.allocator;
            if (s.produced >= convo.max_response_tokens or convo.cache.len >= convo.cache.capacity) {
                s.finished = true;
                return;
            }
            const next = try convo.sampler.next(ctx, logits, convo.history.items);
            if (convo.isStopToken(next)) {
                s.finished = true;
                return;
            }
            if (convo.stop_sequences.len > 0) {
                const prev_len = s.reply.items.len;
                try convo.tokenizer.decodeAppend(a, @intCast(next), &s.reply);
                // Stop before streaming the token that completes a stop sequence.
                if (stopHitInTail(s.reply.items, prev_len, convo.stop_sequences)) {
                    s.finished = true;
                    return;
                }
            }
            try convo.stream.push(a, @intCast(next), s.writer);
            try s.writer.flush();
            try convo.history.append(a, next);
            s.produced += 1;
            s.last = next;
        }

        /// Speculative turn. The decoder invariant is `history.len == cache.len + 1`
        /// (last committed token not yet forwarded), so the prefill covers
        /// everything committed-but-uncached EXCEPT the last prefix token; the
        /// first decoder step forwards it. A verify batch can overshoot the stop
        /// marker or the response cap: `TurnGate` stops streaming/observing at the
        /// boundary and the tail is trimmed from history + KV cache afterwards, so
        /// the post-turn state matches the plain path's exactly.
        fn sendSpec(self: *Self, st: *SpecState, prefix: []const usize, writer: *std.Io.Writer) !usize {
            const turn_start = self.history.items.len;

            // The new prompt tokens are committed context: feed the index.
            st.index.observe(prefix);

            if (self.history.items.len > self.cache.len + 1) {
                var pre = try self.model.forwardStep(self.ctx, &self.cache, self.history.items[self.cache.len .. self.history.items.len - 1], self.cache.len);
                pre.deinit();
            }

            self.stream.reset();
            var gate = TurnGate{
                .convo = self,
                .writer = writer,
                .inner = st.index.asDraftSource(),
                .index = &st.index,
                .sink_acc = .{ .stop = self.stop_id, .extra = self.extra_stop_ids, .budget = self.max_response_tokens },
                .obs_acc = .{ .stop = self.stop_id, .extra = self.extra_stop_ids, .budget = self.max_response_tokens },
            };
            // Route the decoder's source through the gate for this turn.
            st.decoder.source = gate.source();
            defer st.decoder.source = st.index.asDraftSource();

            // Trim the stop marker and any overshoot committed past the boundary
            // — UNCONDITIONALLY, error paths included (a failed write mid-turn
            // must not leave un-trimmed tokens in history/KV, or the next send
            // desyncs from the index and the plain path's state).
            defer {
                const keep = turn_start + gate.sink_acc.n;
                self.history.shrinkRetainingCapacity(keep);
                self.cache.truncate(keep);
            }

            const sink = speculative.TokenSink{ .ptr = &gate, .func = TurnGate.emit };
            while (!gate.sink_acc.done and gate.sink_acc.n < self.max_response_tokens and self.cache.len < self.cache.capacity) {
                _ = try st.decoder.step(self.ctx, self.model, &self.cache, &self.sampler, &self.history, sink);
            }

            try self.stream.flush(writer);
            try writer.flush();
            return gate.sink_acc.n;
        }

        /// True when `token` ends the assistant turn (the template's stop
        /// marker or any configured extra stop id).
        fn isStopToken(self: *const Self, token: usize) bool {
            if (self.stop_id) |s| if (token == s) return true;
            for (self.extra_stop_ids) |s| if (token == s) return true;
            return false;
        }

        /// Turn-boundary filter around the speculation index: streams accepted tokens
        /// through the conversation's StreamDecoder and keeps the index from learning
        /// tokens that the turn trim will discard (stop marker + overshoot), so the
        /// conversation SAM stays byte-exact with the committed history.
        const TurnGate = struct {
            convo: *Self,
            writer: *std.Io.Writer,
            inner: speculative.DraftSource,
            /// The wrapped cascade, for pending-accounting adjustments when the
            /// gate truncates a draft the inner suggest already recorded.
            index: *spec_cascade.SpeculationIndex,
            sink_acc: Accept,
            obs_acc: Accept,
            /// Tokens forwarded by the most recent observe (bounds observeTopK rows).
            last_fwd: usize = 0,

            fn emit(ptr: *anyopaque, token: usize) anyerror!void {
                const self: *TurnGate = @ptrCast(@alignCast(ptr));
                if (!self.sink_acc.take(token)) return;
                try self.convo.stream.push(self.convo.allocator, @intCast(token), self.writer);
                try self.writer.flush();
            }

            fn source(self: *TurnGate) speculative.DraftSource {
                return .{ .ptr = self, .vtable = &gate_vtable };
            }

            const gate_vtable = speculative.DraftSource.VTable{
                .suggest = gSuggest,
                .observe = gObserve,
                .observeTopK = gObserveTopK,
            };

            /// Draft hygiene (RNG/lossless contract): cap the draft at
            /// remaining-budget - 1 — so the verify pass can commit at most
            /// `remaining` tokens, exactly the number of RNG draws the plain loop
            /// has left — and truncate it at the first stop marker, so the turn
            /// boundary token can only arrive as a correction/bonus SAMPLE (one
            /// draw, like plain; the decoder's stop_token then ends the row loop).
            /// The pending accounting shrinks with the truncation: the decoder
            /// only ever verifies the truncated prefix.
            fn gSuggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
                const self: *TurnGate = @ptrCast(@alignCast(ptr));
                const remaining = self.convo.max_response_tokens -| self.sink_acc.n;
                if (remaining <= 1) return 0; // last budgeted token: plain step
                const n = self.inner.suggest(context, buf[0..@min(buf.len, remaining - 1)]);
                for (buf[0..n], 0..) |t, j| {
                    if (self.convo.isStopToken(t)) {
                        self.index.truncatePending(j);
                        return j;
                    }
                }
                return n;
            }

            fn gObserve(ptr: *anyopaque, committed: []const usize) void {
                const self: *TurnGate = @ptrCast(@alignCast(ptr));
                var keep: usize = committed.len;
                for (committed, 0..) |t, i| {
                    if (!self.obs_acc.take(t)) {
                        keep = i;
                        break;
                    }
                }
                self.last_fwd = keep;
                self.inner.observe(committed[0..keep]);
            }

            fn gObserveTopK(ptr: *anyopaque, positions: []const speculative.TopKRow) void {
                const self: *TurnGate = @ptrCast(@alignCast(ptr));
                // Row i's input is the token PRECEDING committed[i], so with `keep`
                // committed tokens forwarded, rows 0..keep (inclusive of the row that
                // produced the first dropped token) are conditioned on kept context.
                const keep = @min(positions.len, self.last_fwd + 1);
                self.inner.observeTopK(positions[0..keep]);
            }
        };
    };
}

/// Per-turn accept state machine, shared rule for the sink and observe paths:
/// take tokens until a stop marker (dropped) or the budget is exhausted.
const Accept = struct {
    stop: ?u32,
    extra: []const u32 = &.{},
    budget: usize,
    n: usize = 0,
    done: bool = false,

    fn take(self: *Accept, token: usize) bool {
        if (self.done) return false;
        if (self.isStop(token)) {
            self.done = true;
            return false;
        }
        if (self.n == self.budget) {
            self.done = true;
            return false;
        }
        self.n += 1;
        return true;
    }

    fn isStop(self: *const Accept, token: usize) bool {
        if (self.stop) |s| if (token == s) return true;
        for (self.extra) |s| if (token == s) return true;
        return false;
    }
};

fn containsAny(haystack: []const u8, needles: []const []const u8) bool {
    for (needles) |n| if (n.len > 0 and std.mem.indexOf(u8, haystack, n) != null) return true;
    return false;
}

/// Stop-sequence scan for the bytes appended since `prev_len`. A match wholly
/// inside `items[0..prev_len]` would already have ended the turn on the append
/// that completed it, so only the window overlapping the new bytes needs
/// rescanning — keeps the per-turn cost linear in reply length.
fn stopHitInTail(items: []const u8, prev_len: usize, needles: []const []const u8) bool {
    var max_len: usize = 0;
    for (needles) |n| max_len = @max(max_len, n.len);
    return containsAny(items[prev_len -| (max_len -| 1) ..], needles);
}

test {
    _ = @import("chat_tests.zig");
}
