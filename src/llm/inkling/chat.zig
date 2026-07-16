//! Inkling chat: the wire-format prompt renderer and the reply-routing
//! generation engine. NO Jinja — the reference `Inkling.jinja` is a typed-
//! content-block protocol over special tokens, and this reproduces the
//! token wire format directly (verified byte-exact against llama.cpp's own
//! minja rendering of that template; see chat_tests.zig).
//!
//! Wire format (roles → tokens, each block closed by `<|end_message|>`):
//!   <|message_system|><|content_text|>{system}<|end_message|>
//!   <|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|>   (auto, before the first non-system message)
//!   <|message_user|><|content_text|>{user}<|end_message|>
//!   <|message_model|><|content_text|>{assistant}<|end_message|><|content_model_end_sampling|>   (history)
//!   <|message_model|>   (generation prompt)
//!
//! Reply: the model emits typed blocks after the generation prompt —
//!   <|content_thinking|>{reasoning}<|end_message|><|message_model|><|content_text|>{answer}<|end_message|>
//! and ends the turn with <|content_model_end_sampling|> (id 200006, the
//! sole end-of-generation token; <|end_message|> 200010 is an intra-turn
//! separator and never stops). The engine streams a MARKER-WRAPPED byte
//! stream — reasoning kept between the literal `<|content_thinking|>` and
//! `<|content_text|>` markers, structural markers stripped — so a single
//! open/close splitter (lmserve's ThinkScanner) routes reasoning vs
//! content downstream.

const std = @import("std");
const fucina = @import("fucina");
const model_mod = @import("model.zig");
const chat = @import("../chat.zig");
const sampler_mod = @import("../sampler.zig");

fn isExtraStop(id: u32, extra: []const u32) bool {
    for (extra) |e| if (e == id) return true;
    return false;
}

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const Message = chat.Message;

pub const Error = error{
    EmptyMessages,
    TrailingAssistantMessage,
    MissingChatMarker,
    KvCacheOverflow,
} || Allocator.Error;

// Special-token literals (resolved to ids against the vocab at engine init).
pub const tok_message_user = "<|message_user|>";
pub const tok_message_model = "<|message_model|>";
pub const tok_message_system = "<|message_system|>";
pub const tok_content_text = "<|content_text|>";
pub const tok_content_thinking = "<|content_thinking|>";
pub const tok_end_message = "<|end_message|>";
pub const tok_end_sampling = "<|content_model_end_sampling|>"; // == EOS 200006

pub const RenderOptions = struct {
    /// The "Thinking effort level: X" value (reference default 0.9).
    effort_text: []const u8 = "0.9",
    add_generation_prompt: bool = true,
    /// Prime a `<|content_text|>` block after the generation prompt so the
    /// model skips the thinking block — the honest realization of
    /// reasoning-off (and required under a grammar constraint, which must
    /// govern the reply from token 0).
    think_off: bool = false,
};

fn roleToken(role: Message.Role) []const u8 {
    return switch (role) {
        .user => tok_message_user,
        .assistant => tok_message_model,
        .system => tok_message_system,
    };
}

/// Render the full prompt string. Caller owns the returned slice. Markers
/// are literal text; the tokenizer resolves each `<|…|>` to its single id.
pub fn renderPrompt(allocator: Allocator, messages: []const Message, opts: RenderOptions) Error![]u8 {
    if (messages.len == 0) return Error.EmptyMessages;
    if (messages[messages.len - 1].role == .assistant) return Error.TrailingAssistantMessage;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    // The effort line is emitted immediately before the first non-system
    // message (matching the reference template); if every message is a
    // system message it is appended at the end.
    var first_non_system: ?usize = null;
    for (messages, 0..) |m, i| {
        if (m.role != .system) {
            first_non_system = i;
            break;
        }
    }

    var effort_emitted = false;
    for (messages, 0..) |m, i| {
        if (!effort_emitted and first_non_system != null and i == first_non_system.?) {
            try emitEffort(allocator, &out, opts.effort_text);
            effort_emitted = true;
        }
        try out.appendSlice(allocator, roleToken(m.role));
        try out.appendSlice(allocator, tok_content_text);
        try out.appendSlice(allocator, m.content);
        try out.appendSlice(allocator, tok_end_message);
        if (m.role == .assistant) try out.appendSlice(allocator, tok_end_sampling);
    }
    if (!effort_emitted) {
        try emitEffort(allocator, &out, opts.effort_text);
    }

    if (opts.add_generation_prompt) {
        try out.appendSlice(allocator, tok_message_model);
        if (opts.think_off) try out.appendSlice(allocator, tok_content_text);
    }
    return out.toOwnedSlice(allocator);
}

fn emitEffort(allocator: Allocator, out: *std.ArrayList(u8), effort_text: []const u8) Allocator.Error!void {
    try out.appendSlice(allocator, tok_message_system);
    try out.appendSlice(allocator, tok_content_text);
    try out.appendSlice(allocator, "Thinking effort level: ");
    try out.appendSlice(allocator, effort_text);
    try out.appendSlice(allocator, tok_end_message);
}

/// Marker token ids, resolved from a tokenizer at engine init.
pub const Markers = struct {
    content_text: u32,
    content_thinking: u32,
    end_message: u32,
    message_model: u32,
    end_sampling: u32, // == the GGUF eos (200006)

    pub fn resolve(comptime TokMod: type, tokenizer: *const TokMod.Tokenizer) !Markers {
        return .{
            .content_text = tokenizer.tokenId(tok_content_text) orelse return Error.MissingChatMarker,
            .content_thinking = tokenizer.tokenId(tok_content_thinking) orelse return Error.MissingChatMarker,
            .end_message = tokenizer.tokenId(tok_end_message) orelse return Error.MissingChatMarker,
            .message_model = tokenizer.tokenId(tok_message_model) orelse return Error.MissingChatMarker,
            .end_sampling = tokenizer.tokenId(tok_end_sampling) orelse return Error.MissingChatMarker,
        };
    }
};

pub const GenerateOptions = struct {
    sampling: sampler_mod.Config = .{},
    processor: ?sampler_mod.LogitProcessor = null,
    max_tokens: usize,
    /// Extra end-of-generation ids beyond `end_sampling` (== EOS).
    extra_stop_ids: []const u32 = &.{},
    think_off: bool = false,
};

pub const GenerateResult = struct {
    prompt_tokens: usize,
    completion_tokens: usize,
    /// True when generation stopped on an end-of-turn id (vs the token cap).
    stopped: bool,
};

const Logits = fucina.Tensor(.{ .seq, .vocab });

/// The generation driver: prefill the prompt, then sample-and-stream one
/// marker-wrapped reply. Generic over the tokenizer module (BPE here).
pub fn Engine(comptime TokMod: type) type {
    return struct {
        const Self = @This();
        const Tokenizer = TokMod.Tokenizer;

        ctx: *ExecContext,
        model: *const model_mod.Model,
        tokenizer: *const Tokenizer,
        markers: Markers,

        pub fn init(ctx: *ExecContext, model: *const model_mod.Model, tokenizer: *const Tokenizer) !Self {
            return .{
                .ctx = ctx,
                .model = model,
                .tokenizer = tokenizer,
                .markers = try Markers.resolve(TokMod, tokenizer),
            };
        }

        /// Run one reply. `prompt` are the rendered prompt token ids. The
        /// marker-wrapped reply bytes stream to `sink` (flushed per token by
        /// the caller's writer). Structural markers (`<|end_message|>`,
        /// `<|message_model|>`) are stripped; `<|content_thinking|>` and
        /// `<|content_text|>` pass through literally so a downstream
        /// open/close splitter can route reasoning vs content.
        pub fn generate(self: *Self, prompt: []const usize, opts: GenerateOptions, sink: *std.Io.Writer) !GenerateResult {
            const a = self.ctx.allocator;
            var cache = try self.model.initCache(prompt.len + opts.max_tokens + 1);
            defer cache.deinit();

            var history: std.ArrayList(usize) = .empty;
            defer history.deinit(a);
            try history.appendSlice(a, prompt);

            var sampler = sampler_mod.Sampler.init(opts.sampling);
            sampler.processor = opts.processor;

            var stream = @import("../tokenizer.zig").StreamDecoder.init(self.tokenizer);
            defer stream.deinit(a);

            // Prefill.
            var logits_buf = try self.model.step(self.ctx, &cache, prompt);
            defer a.free(logits_buf);

            var produced: usize = 0;
            var stopped = false;
            while (produced < opts.max_tokens) {
                var logits_t = try Logits.fromSlice(self.ctx, .{ 1, logits_buf.len }, logits_buf);
                defer logits_t.deinit();
                const next = try sampler.next(self.ctx, &logits_t, history.items);
                const id: u32 = @intCast(next);

                if (id == self.markers.end_sampling or isExtraStop(id, opts.extra_stop_ids)) {
                    stopped = true;
                    break;
                }
                try self.emitToken(&stream, id, sink, a);
                try sink.flush(); // the sink's drain is its per-token flush point
                try history.append(a, next);
                produced += 1;

                a.free(logits_buf);
                logits_buf = try self.model.step(self.ctx, &cache, &.{next});
            }
            // Flush any bytes held by the incremental UTF-8 decoder.
            try stream.flush(sink);
            try sink.flush();

            return .{ .prompt_tokens = prompt.len, .completion_tokens = produced, .stopped = stopped };
        }

        /// One token to the marker-wrapped stream: the two routing markers
        /// pass through literally, other structural markers are dropped,
        /// everything else is decoded to text.
        fn emitToken(self: *Self, stream: *@import("../tokenizer.zig").StreamDecoder, id: u32, sink: *std.Io.Writer, a: Allocator) !void {
            if (id == self.markers.content_thinking) {
                try sink.writeAll(tok_content_thinking);
                return;
            }
            if (id == self.markers.content_text) {
                try sink.writeAll(tok_content_text);
                return;
            }
            if (id == self.markers.end_message or id == self.markers.message_model) {
                return; // structural: strip
            }
            try stream.push(a, id, sink);
        }
    };
}

test {
    _ = @import("chat_tests.zig");
}
