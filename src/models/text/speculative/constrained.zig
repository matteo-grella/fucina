//! Grammar-aware draft source: turns a `LogitProcessor`'s structural hooks
//! (§`logit_processor.zig` — `forcedTokens`/`validPrefixLen`) into
//! speculative-decoding drafts around an inner `DraftSource`.
//!
//! Two effects, both pure win under a constraint installed on the sampler:
//!
//! - **Forced spans draft themselves.** When the grammar mandates a unique
//!   continuation (JSON structure like `", "country": "`), those tokens are
//!   proposed as the draft. The masked sampler can only select the forced
//!   token at each of those rows, so the whole span verifies with acceptance
//!   probability 1 — one batched pass commits it all.
//! - **Invalid drafts die before the verify pass.** When the grammar forces
//!   nothing, the inner source's draft is truncated at its first
//!   grammar-invalid token: those tokens would be masked to -inf at their
//!   verify row and rejected with certainty, so proposing them only wastes
//!   verify compute and drags the source's acceptance gates down.
//!
//! Losslessness is untouched: drafts never affect WHAT is committed (the
//! §13.9 rejection-sampling contract), and both hooks are deterministic pure
//! lookaheads, so the source stays deterministic. The wrapped processor MUST
//! be the same one installed on the decode stream's sampler — the
//! acceptance-probability-1 property comes from the mask and the drafts
//! reading the same grammar state.

const std = @import("std");
const core = @import("core.zig");
const logit_processor = @import("../logit_processor.zig");

const DraftSource = core.DraftSource;
const TopKRow = core.TopKRow;
const LogitProcessor = logit_processor.LogitProcessor;

/// Wraps `inner` with the structural knowledge of `processor`. Single-stream
/// mutable state (tracks which side produced the live draft for pending
/// accounting); one per decode stream, adjacent to its sampler/processor.
pub const ConstrainedSource = struct {
    processor: LogitProcessor,
    inner: DraftSource,
    /// Whether the most recent `suggest` delegated to `inner` (its pending
    /// acceptance accounting is live and truncations must be forwarded).
    last_from_inner: bool = false,

    /// `processor` must expose both structural hooks
    /// (`LogitProcessor.hasStructure`); the caller checks (`init` in a
    /// wrapper would hide the requirement behind a runtime error for no
    /// gain — wiring code branches on `hasStructure` anyway).
    pub fn init(processor: LogitProcessor, inner: DraftSource) ConstrainedSource {
        return .{ .processor = processor, .inner = inner };
    }

    /// The vtable mirrors the inner source's top-k appetite, so wrapping a
    /// source that ignores logits feedback doesn't make the decoder compute
    /// top-k rows for nothing.
    pub fn source(self: *ConstrainedSource) DraftSource {
        return .{ .ptr = self, .vtable = if (self.inner.wantsTopK()) &vtable_topk else &vtable_plain };
    }

    const vtable_topk = DraftSource.VTable{
        .suggest = vtSuggest,
        .observe = vtObserve,
        .observeTopK = vtObserveTopK,
        .truncatePending = vtTruncatePending,
    };

    const vtable_plain = DraftSource.VTable{
        .suggest = vtSuggest,
        .observe = vtObserve,
        .truncatePending = vtTruncatePending,
    };

    fn vtSuggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
        const self: *ConstrainedSource = @ptrCast(@alignCast(ptr));
        // Grammar-forced span: certain acceptance, no inner accounting.
        const forced = self.processor.forcedTokens(buf);
        if (forced > 0) {
            self.last_from_inner = false;
            return forced;
        }
        // Free choice: the inner source proposes, the grammar prunes the
        // certainly-rejected tail (and its pending accounting follows).
        self.last_from_inner = true;
        const n = self.inner.suggest(context, buf);
        if (n == 0) return 0;
        const keep = self.processor.validPrefixLen(buf[0..n]);
        if (keep < n) self.inner.truncatePending(keep);
        return keep;
    }

    fn vtObserve(ptr: *anyopaque, committed: []const usize) void {
        const self: *ConstrainedSource = @ptrCast(@alignCast(ptr));
        // Grammar state advances through the sampler's commit hook, never
        // here; only the inner index learns.
        self.inner.observe(committed);
    }

    fn vtObserveTopK(ptr: *anyopaque, positions: []const TopKRow) void {
        const self: *ConstrainedSource = @ptrCast(@alignCast(ptr));
        self.inner.observeTopK(positions);
    }

    fn vtTruncatePending(ptr: *anyopaque, new_len: usize) void {
        const self: *ConstrainedSource = @ptrCast(@alignCast(ptr));
        // Forced drafts carry no pending accounting anywhere; inner drafts
        // forward the truncation (e.g. the chat turn-boundary filter).
        if (self.last_from_inner) self.inner.truncatePending(new_len);
    }
};

test {
    _ = @import("constrained_tests.zig");
}
