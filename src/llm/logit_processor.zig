//! Pluggable logit processing for autoregressive decoding.
//!
//! A `LogitProcessor` mutates the raw next-token logits in place right before
//! the sampler's own pipeline runs (penalties → temperature → top-k/p → draw)
//! and observes every token the sampler selects. The canonical use is
//! grammar/JSON-schema constrained decoding (`llguidance.zig` adapts a
//! compiled grammar to this interface by writing `-inf` over disallowed
//! tokens), but any stateful logit transform fits — bias lists, custom
//! banned-token rules, watermarking.
//!
//! The seam lives INSIDE `sampler.Sampler` (`Sampler.processor`), so every
//! decode path that samples through it — `chat.Conversation` send/sendBatch,
//! the speculative decoder's plain and verify steps, and hand-rolled runner
//! loops — picks the processor up without any loop changes. Two properties
//! make that sound:
//!
//! - Every `Sampler.next` result IS a committed token. The speculative
//!   verify loop samples each row only after the row's prefix is committed
//!   to history, and every sampled row token is itself committed (accepted
//!   draft, correction, or bonus — `speculative/core.zig`), so `commit`
//!   inside `next` keeps processor state exactly in step with history. A
//!   draft token the processor's mask forbids simply loses the
//!   `sampled == draft` comparison and is rejected: lossless equivalence
//!   with the unconstrained-loop-shape run is preserved, no rollback needed.
//! - Turn-boundary tokens (the chat stop marker / a completed text stop
//!   sequence) are sampled-then-dropped by every path alike, so the
//!   processor observes them uniformly; `chat.Conversation` re-arms the
//!   processor at each turn start via `reset`.

const std = @import("std");

/// Type-erased processor handle (the `DraftSource` vtable pattern —
/// `speculative/core.zig`). Single-stream mutable state: one processor per
/// decode stream, driven from that stream's thread only.
pub const LogitProcessor = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Mutate `logits` (one `[vocab]` row) in place before sampling.
        /// `history` is the committed token stream so far (prompt +
        /// generated), for processors that need position/context.
        process: *const fn (ptr: *anyopaque, logits: []f32, history: []const usize) anyerror!void,
        /// Observe the token the sampler selected from the processed row.
        /// Called exactly once per `Sampler.next` call, after selection.
        commit: *const fn (ptr: *anyopaque, token: usize) anyerror!void,
        /// Re-arm for a fresh constrained region (a new assistant turn).
        /// Optional: null means the processor is stateless across turns.
        reset: ?*const fn (ptr: *anyopaque) anyerror!void = null,

        // --- Structural hooks (optional; null = the processor exposes no
        // structure). Both must be pure lookahead — no state change — and
        // deterministic: the speculative layer turns them into drafts
        // (`speculative.constrained.ConstrainedSource`), and `DraftSource`s
        // must be deterministic.

        /// Write the tokens FORCED from the current state (the unique legal
        /// continuation, e.g. grammar-mandated punctuation/keys) into `buf`;
        /// return how many (0 = the next token is not forced).
        forcedTokens: ?*const fn (ptr: *anyopaque, buf: []usize) usize = null,
        /// How many leading `tokens` are acceptable continuations of the
        /// current state (for pre-filtering drafts that `process`'s mask
        /// would reject anyway).
        validPrefixLen: ?*const fn (ptr: *anyopaque, tokens: []const usize) usize = null,
    };

    pub fn process(self: LogitProcessor, logits: []f32, history: []const usize) !void {
        return self.vtable.process(self.ptr, logits, history);
    }

    pub fn commit(self: LogitProcessor, token: usize) !void {
        return self.vtable.commit(self.ptr, token);
    }

    pub fn reset(self: LogitProcessor) !void {
        if (self.vtable.reset) |f| return f(self.ptr);
    }

    /// True when both structural hooks are present (the requirement for
    /// grammar-driven drafting — `speculative.constrained`).
    pub fn hasStructure(self: LogitProcessor) bool {
        return self.vtable.forcedTokens != null and self.vtable.validPrefixLen != null;
    }

    pub fn forcedTokens(self: LogitProcessor, buf: []usize) usize {
        const f = self.vtable.forcedTokens orelse return 0;
        return f(self.ptr, buf);
    }

    pub fn validPrefixLen(self: LogitProcessor, tokens: []const usize) usize {
        const f = self.vtable.validPrefixLen orelse return tokens.len;
        return f(self.ptr, tokens);
    }
};

test {
    _ = @import("logit_processor_tests.zig");
}
