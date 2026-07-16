//! Qwen3.5/Qwen3.6 (Bonsai) chat: ChatML prompt rendering plus a generation
//! engine over the hybrid Gated-DeltaNet `Model`/`Cache`.
//!
//! Rendering rides the shared `llm.chat.Template` ChatML renderer (system /
//! user / assistant role blocks, historical reasoning stripped, `<|im_end|>`
//! turn ends) with one Qwen3.6 refinement on the generation prompt, taken
//! from the GGUF's own `tokenizer.chat_template`:
//!
//!   thinking on:   `<|im_start|>assistant\n<think>\n`   (opener prefilled —
//!                  the reply streams already inside the think block)
//!   thinking off:  `<|im_start|>assistant\n<think>\n\n</think>\n\n`
//!                  (the ChatML renderer's own think_off form, byte-equal)
//!
//! The engine runs one reply per call on a FRESH cache: the linear-attention
//! layers carry recurrent conv/state matrices that cannot be truncated back
//! to a token prefix, so llama.cpp-style KV-slot reuse does not apply to
//! this family.

const std = @import("std");
const fucina = @import("fucina");
const model_mod = @import("model.zig");
const chat = @import("../chat.zig");
const sampler_mod = @import("../sampler.zig");
const tokenizer_mod = @import("../tokenizer.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const Message = chat.Message;

pub const Error = error{MissingChatMarker} || Allocator.Error;

/// The prompt-side think-block opener (thinking on) — also what a server
/// injects into the reply stream so an open/close reasoning splitter sees
/// the block the model is already inside.
pub const think_opener = "<think>\n";

pub const RenderOptions = struct {
    think_off: bool = false,
};

/// Render the full message history into a ChatML prompt ending with the
/// assistant-turn opener (and the Qwen3.6 think prefill). Caller owns the
/// returned slice.
pub fn renderPrompt(
    allocator: Allocator,
    template: chat.Template,
    messages: []const Message,
    opts: RenderOptions,
) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try template.renderMessages(allocator, &buf, messages, opts.think_off);
    if (!opts.think_off) try buf.appendSlice(allocator, think_opener);
    return buf.toOwnedSlice(allocator);
}

fn isExtraStop(id: u32, extra: []const u32) bool {
    for (extra) |e| if (e == id) return true;
    return false;
}

pub const GenerateOptions = struct {
    sampling: sampler_mod.Config = .{},
    processor: ?sampler_mod.LogitProcessor = null,
    max_tokens: usize,
    /// Cache capacity ceiling (the server's per-request context budget).
    capacity: usize,
    /// End-of-turn ids beyond the resolved `<|im_end|>`.
    extra_stop_ids: []const u32 = &.{},
};

pub const GenerateResult = struct {
    prompt_tokens: usize,
    completion_tokens: usize,
    /// True when generation stopped on an end-of-turn id (vs a budget cap).
    stopped: bool,
};

/// The generation driver: prefill the rendered prompt, then sample-and-
/// stream one reply. Generic over the tokenizer module (byte-level BPE for
/// this family).
pub fn Engine(comptime TokMod: type) type {
    return struct {
        const Self = @This();
        const Tokenizer = TokMod.Tokenizer;

        ctx: *ExecContext,
        model: *const model_mod.Model,
        tokenizer: *const Tokenizer,
        /// `<|im_end|>` — the ChatML turn end (== this family's GGUF EOS).
        stop_id: u32,

        pub fn init(ctx: *ExecContext, model: *const model_mod.Model, tokenizer: *const Tokenizer) Error!Self {
            const stop_id = tokenizer.tokenId("<|im_end|>") orelse
                tokenizer.eosId() orelse return Error.MissingChatMarker;
            return .{ .ctx = ctx, .model = model, .tokenizer = tokenizer, .stop_id = stop_id };
        }

        /// Run one reply. `prompt` are the rendered prompt token ids; the
        /// reply text streams to `sink` (flushed per token). The cache is
        /// created for this call and dropped with it.
        pub fn generate(self: *Self, prompt: []const usize, opts: GenerateOptions, sink: *std.Io.Writer) !GenerateResult {
            const a = self.ctx.allocator;
            const capacity = @min(opts.capacity, prompt.len + opts.max_tokens + 1);
            var cache = try self.model.initCache(self.ctx, capacity);
            defer cache.deinit();

            var history: std.ArrayList(usize) = .empty;
            defer history.deinit(a);
            try history.appendSlice(a, prompt);

            var sampler = sampler_mod.Sampler.init(opts.sampling);
            sampler.processor = opts.processor;

            var stream = tokenizer_mod.StreamDecoder.init(self.tokenizer);
            defer stream.deinit(a);

            var logits = try self.model.forwardStep(self.ctx, &cache, prompt, 0);
            var logits_live = true;
            defer if (logits_live) logits.deinit();

            var produced: usize = 0;
            var stopped = false;
            while (produced < opts.max_tokens and cache.len() < capacity) {
                const next = try sampler.next(self.ctx, &logits, history.items);
                const id: u32 = @intCast(next);
                if (id == self.stop_id or isExtraStop(id, opts.extra_stop_ids)) {
                    stopped = true;
                    break;
                }
                try stream.push(a, id, sink);
                try sink.flush(); // the sink's drain is its per-token flush point
                try history.append(a, next);
                produced += 1;

                if (cache.len() >= capacity) break;
                logits.deinit();
                logits_live = false;
                logits = try self.model.forwardStep(self.ctx, &cache, &.{next}, cache.len());
                logits_live = true;
            }
            // Flush any bytes held by the incremental UTF-8 decoder.
            try stream.flush(sink);
            try sink.flush();

            return .{ .prompt_tokens = prompt.len, .completion_tokens = produced, .stopped = stopped };
        }
    };
}

test {
    _ = @import("chat_tests.zig");
}
