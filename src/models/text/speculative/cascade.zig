//! Draft-source cascade for draft-model-free speculative decoding.
//!
//! `SpeculationIndex` is the user-facing orchestrator behind one
//! `speculative.DraftSource`: it composes
//!
//!   1. `conversation` — an online `SamIndex` over every committed token
//!      (prompt + generated), auto-fed through `observe`;
//!   2. `references`   — any number of frozen `SamIndex` documents injected
//!      via `addReference` (the RAG seam): each keeps a per-conversation
//!      match cursor advanced per committed token;
//!   3. `recycling`    — the Token-Recycling matrix, refreshed from the
//!      verifier's `observeTopK` logits feedback; the self-draft fallback.
//!
//! ## suggest policy (research-prescribed)
//!
//! Query the conversation's exact longest-suffix-match length AND every
//! reference cursor's match length; draft from the source with the LONGEST
//! match (ties break toward the conversation — recency in the live stream
//! beats a static document; among references, first-added wins). The draft
//! budget grows with match confidence:
//!
//!     budget = min(buf.len, beta * (1 + match_len))   // beta = 2 default
//!                                                     // => 2 + 2*match_len
//!
//! When the best match is shorter than `min_match` (2), fall back to the
//! recycling top-1 chain (length <= `recycling_chain` = 8). An unseen
//! recycling row drafts 0 tokens: the decoder does a plain step.
//!
//! ## Per-source gating
//!
//! Each source keeps a rolling acceptance window over its last `gate_window`
//! (64) DRAFTED tokens. When the window fills below `mute_acceptance` (20%),
//! the source is muted for `mute_commits` (128) committed tokens, then
//! re-probed with a fresh window — no source dies permanently, including the
//! conversation. Muted sources keep observing (SAM appends / cursor advances /
//! recycling updates) so they stay in sync for the re-probe. This is the
//! source-SELECTION gate; the decoder's own auto-off gate handles the global
//! verify economics.
//!
//! ## Accounting contract
//!
//! Acceptance is inferred at the `suggest`->`observe` seam: the decoder
//! verifies a draft iff it is at least `speculative.Options.min_draft` long,
//! and the next `observe` then carries that step's committed tokens, whose
//! longest common prefix with the draft is exactly the accepted count.
//! `accounting_min_draft` mirrors the decoder's default `min_draft` (2);
//! keep them aligned or per-source stats skew (losslessness is unaffected —
//! gating is purely a draft-selection heuristic).

const std = @import("std");
const speculative = @import("core.zig");
const sam_index = @import("sam_index.zig");
const recycling_mod = @import("recycling.zig");

const Allocator = std.mem.Allocator;
const SamIndex = sam_index.SamIndex;
const Recycling = recycling_mod.Recycling;
const DraftSource = speculative.DraftSource;
const TopKRow = speculative.TopKRow;

/// Rolling-acceptance window length, in drafted tokens, per source.
pub const gate_window: usize = 64;
/// Longest draft prefix remembered for acceptance accounting.
const pending_cap: usize = 64;

/// Per-source acceptance gate: a ring over the last `gate_window` drafted
/// tokens (accepted yes/no) + lifetime totals + the mute countdown.
pub const Gate = struct {
    ring: [gate_window]bool = @splat(false),
    len: usize = 0,
    idx: usize = 0,
    hits: usize = 0,
    /// Committed tokens left until this source is re-probed (0 = active).
    mute_left: usize = 0,
    /// Lifetime totals (reporting).
    total_drafted: usize = 0,
    total_accepted: usize = 0,
    times_muted: usize = 0,

    pub fn muted(self: *const Gate) bool {
        return self.mute_left > 0;
    }

    fn push(self: *Gate, hit: bool) void {
        if (self.len == gate_window) {
            if (self.ring[self.idx]) self.hits -= 1;
        } else {
            self.len += 1;
        }
        self.ring[self.idx] = hit;
        if (hit) self.hits += 1;
        self.idx = (self.idx + 1) % gate_window;
    }

    fn resetWindow(self: *Gate) void {
        self.len = 0;
        self.idx = 0;
        self.hits = 0;
    }

    /// Record one verified draft outcome (`accepted <= drafted`); mute when
    /// the filled window's acceptance drops below `threshold`.
    fn record(self: *Gate, accepted: usize, drafted: usize, threshold: f32, mute_commits: usize) void {
        self.total_drafted += drafted;
        self.total_accepted += accepted;
        for (0..drafted) |i| self.push(i < accepted);
        if (self.len == gate_window) {
            const rate = @as(f32, @floatFromInt(self.hits)) / @as(f32, @floatFromInt(gate_window));
            if (rate < threshold) {
                self.mute_left = mute_commits;
                self.times_muted += 1;
                self.resetWindow();
            }
        }
    }

    fn onCommitted(self: *Gate, n: usize) void {
        self.mute_left -|= n;
    }
};

/// One injected reference document: a frozen SAM index, this conversation's
/// match cursor over it, and its acceptance gate.
pub const FrozenRef = struct {
    index: SamIndex,
    cursor: SamIndex.Cursor = .{},
    gate: Gate = .{},
};

pub const SpeculationIndex = struct {
    const SourceId = union(enum) {
        conversation,
        reference: usize,
        recycling,
    };

    const Pending = struct {
        source: SourceId,
        /// Full submitted draft length (the gate denominator).
        drafted: usize,
        /// Remembered prefix for acceptance comparison.
        len: usize,
        buf: [pending_cap]usize,
    };

    allocator: Allocator,
    /// Online SAM over every committed token (prompt + generated).
    conversation: SamIndex,
    /// Injectable frozen reference documents (RAG).
    references: std.ArrayList(FrozenRef),
    /// Token-Recycling matrix (self-draft fallback + observeTopK consumer).
    recycling: Recycling,
    conv_gate: Gate = .{},
    rec_gate: Gate = .{},
    /// Outstanding draft awaiting its verification outcome.
    pending: ?Pending = null,
    /// Last committed token seen (bridges recycling bigrams across calls).
    last_token: ?usize = null,
    /// Copy of the most recent observe's committed tokens, prefixed with the
    /// boundary token preceding them (when known): the decoder calls
    /// `observe` BEFORE `observeTopK`, so the recycling row overwrites in
    /// `observeTopK` would clobber the same-step committed-bigram slot-0
    /// promotions — they are re-applied from this copy afterwards.
    reapply: std.ArrayList(usize) = .empty,

    // ---- policy constants (documented in the file header) ----
    /// Draft-budget multiplier: budget = beta * (1 + match_len).
    beta: usize = 2,
    /// Matches shorter than this never draft from a SAM source.
    min_match: usize = 2,
    /// Recycling fallback chain length cap.
    recycling_chain: usize = 8,
    /// Mute a source when its filled window's acceptance is below this.
    mute_acceptance: f32 = 0.20,
    /// Committed tokens a muted source sits out before re-probing.
    mute_commits: usize = 128,
    /// Mirror of `speculative.Options.min_draft` (see header).
    accounting_min_draft: usize = 2,

    pub fn init(allocator: Allocator, vocab: usize) !SpeculationIndex {
        var conversation = try SamIndex.init(allocator);
        errdefer conversation.deinit();
        // The cascade's budget (the buf slice) is the only draft-length cap.
        conversation.max_draft = std.math.maxInt(usize);
        const recycling = try Recycling.init(allocator, vocab);
        return .{
            .allocator = allocator,
            .conversation = conversation,
            .references = .empty,
            .recycling = recycling,
        };
    }

    pub fn deinit(self: *SpeculationIndex) void {
        self.reapply.deinit(self.allocator);
        for (self.references.items) |*ref| ref.index.deinit();
        self.references.deinit(self.allocator);
        self.recycling.deinit();
        self.conversation.deinit();
        self.* = undefined;
    }

    /// Build + freeze a SAM index over a tokenized reference document — the
    /// RAG seam. The new cursor is caught up over the already-observed
    /// committed stream, so a document injected mid-conversation can match
    /// the existing context immediately. EXCEPT when the conversation SAM is
    /// degraded: its token copy is truncated (desynced from the real
    /// committed stream), so catching up from it would leave the cursor
    /// describing a stream that never happened — the cursor then starts
    /// LIVE-ONLY (root state; it advances with future committed tokens only).
    pub fn addReference(self: *SpeculationIndex, tokens: []const usize) !void {
        var index = try SamIndex.init(self.allocator);
        errdefer index.deinit();
        index.min_match = self.min_match;
        index.max_draft = std.math.maxInt(usize);
        try index.append(tokens);
        index.freeze();
        var ref = FrozenRef{ .index = index };
        if (!self.conversation.degraded) {
            for (self.conversation.tokens.items) |t| ref.index.advance(&ref.cursor, t);
        }
        try self.references.append(self.allocator, ref);
    }

    pub fn clearReferences(self: *SpeculationIndex) void {
        for (self.references.items) |*ref| ref.index.deinit();
        self.references.clearRetainingCapacity();
        // A pending draft may point at a dropped reference; forget it.
        self.pending = null;
    }

    // ---- DraftSource implementation ----

    pub fn suggest(self: *SpeculationIndex, context: []const usize, buf: []usize) usize {
        self.pending = null; // an unconsumed pending means its step never verified
        if (context.len == 0 or buf.len == 0) return 0;
        const last = context[context.len - 1];

        // Longest match wins; the conversation is seeded first and only
        // replaced on a STRICTLY longer match (tie -> conversation; among
        // references, first-added).
        var best: ?SourceId = null;
        var best_len: usize = 0;
        // A degraded conversation SAM is muted EXPLICITLY (its queries would
        // return 0 anyway, but the contract is: never consult it again — its
        // token copy no longer describes the committed stream).
        if (!self.conv_gate.muted() and !self.conversation.degraded) {
            // Desync guard (decoder contract): the conversation SAM must have
            // observed exactly through the context's last committed token,
            // otherwise its internal cursor describes a different stream.
            const n = self.conversation.tokenCount();
            const synced = n > 0 and self.conversation.tokens.items[n - 1] == last;
            const ml = self.conversation.matchLen();
            if (synced and ml >= self.min_match) {
                best = .conversation;
                best_len = ml;
            }
        }
        for (self.references.items, 0..) |*ref, i| {
            if (ref.gate.muted()) continue;
            const ml: usize = ref.cursor.len;
            if (ml >= self.min_match and ml > best_len) {
                best = .{ .reference = i };
                best_len = ml;
            }
        }

        if (best) |src| {
            const budget = @min(buf.len, self.beta * (1 + best_len));
            const n = switch (src) {
                .conversation => self.conversation.draft(buf[0..budget]),
                .reference => |i| blk: {
                    const ref = &self.references.items[i];
                    break :blk ref.index.draftFrom(ref.cursor, buf[0..budget]);
                },
                .recycling => unreachable,
            };
            if (n > 0) {
                self.notePending(src, buf[0..n]);
                return n;
            }
            // Match sits at the source's stream end (nothing follows):
            // fall through to the recycling chain.
        }

        if (self.rec_gate.muted()) return 0;
        const cap = @min(buf.len, self.recycling_chain);
        const n = self.recycling.draftChain(last, buf[0..cap]);
        if (n > 0) self.notePending(.recycling, buf[0..n]);
        return n;
    }

    /// Feed committed tokens: settle the pending draft's acceptance, advance
    /// the mute countdowns, then update every source (muted ones included —
    /// they must stay in sync for the re-probe).
    pub fn observe(self: *SpeculationIndex, committed: []const usize) void {
        if (committed.len == 0) {
            // A turn filter may drop a whole verify batch (stop token at
            // position 0); the outcome is unknowable, just forget the draft.
            self.pending = null;
            return;
        }
        if (self.pending) |*p| {
            // Accepted = longest common prefix of the committed tokens and
            // the submitted draft (the verifier's correction/bonus token can
            // never extend it: a correction differs by definition).
            var acc: usize = 0;
            const n = @min(p.len, committed.len);
            while (acc < n and committed[acc] == p.buf[acc]) acc += 1;
            const gate = self.gateOf(p.source);
            gate.record(@min(acc, p.drafted), p.drafted, self.mute_acceptance, self.mute_commits);
            self.pending = null;
        }

        self.conv_gate.onCommitted(committed.len);
        self.rec_gate.onCommitted(committed.len);
        for (self.references.items) |*ref| ref.gate.onCommitted(committed.len);

        // Recycling: the bigram across the previous call boundary, then the
        // intra-slice bigrams. Keep a copy (boundary token + committed) so a
        // same-step observeTopK can re-apply these promotions after its row
        // overwrites; on OOM skip the copy (heuristic-only state).
        if (self.last_token) |lt| self.recycling.observe(&.{ lt, committed[0] });
        self.recycling.observe(committed);
        self.reapply.clearRetainingCapacity();
        const copied = blk: {
            if (self.last_token) |lt| self.reapply.append(self.allocator, lt) catch break :blk false;
            self.reapply.appendSlice(self.allocator, committed) catch break :blk false;
            break :blk true;
        };
        if (!copied) self.reapply.clearRetainingCapacity();
        self.last_token = committed[committed.len - 1];

        // Conversation SAM (self-degrades on error instead of propagating;
        // once degraded it is never fed again — explicit muting, see
        // `suggest`).
        if (!self.conversation.degraded) self.conversation.observe(committed);

        for (self.references.items) |*ref| {
            for (committed) |t| ref.index.advance(&ref.cursor, t);
        }
    }

    /// Verification logits feedback -> Token-Recycling matrix rows. The row
    /// overwrites land FIRST, then the same step's committed-bigram
    /// promotions (saved by `observe`, which the decoder always calls before
    /// this) are re-applied — otherwise the ground-truth slot-0 promotions
    /// would be clobbered in the very step that learned them. Idempotent:
    /// promotions re-applied over an already-promoted row are no-ops.
    pub fn observeTopK(self: *SpeculationIndex, positions: []const TopKRow) void {
        self.recycling.observeTopK(positions);
        self.recycling.observe(self.reapply.items);
    }

    pub fn asDraftSource(self: *SpeculationIndex) DraftSource {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable = DraftSource.VTable{
        .suggest = vtSuggest,
        .observe = vtObserve,
        .observeTopK = vtObserveTopK,
        .truncatePending = vtTruncatePending,
    };

    fn vtSuggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
        const self: *SpeculationIndex = @ptrCast(@alignCast(ptr));
        return self.suggest(context, buf);
    }

    fn vtObserve(ptr: *anyopaque, committed: []const usize) void {
        const self: *SpeculationIndex = @ptrCast(@alignCast(ptr));
        self.observe(committed);
    }

    fn vtObserveTopK(ptr: *anyopaque, positions: []const TopKRow) void {
        const self: *SpeculationIndex = @ptrCast(@alignCast(ptr));
        self.observeTopK(positions);
    }

    fn vtTruncatePending(ptr: *anyopaque, new_len: usize) void {
        const self: *SpeculationIndex = @ptrCast(@alignCast(ptr));
        self.truncatePending(new_len);
    }

    fn gateOf(self: *SpeculationIndex, src: SourceId) *Gate {
        return switch (src) {
            .conversation => &self.conv_gate,
            .reference => |i| &self.references.items[i].gate,
            .recycling => &self.rec_gate,
        };
    }

    fn notePending(self: *SpeculationIndex, src: SourceId, draft: []const usize) void {
        // Shorter drafts fall back to a plain step in the decoder and are
        // never verified; recording them would skew the gates (see header).
        if (draft.len < self.accounting_min_draft) return;
        var p = Pending{ .source = src, .drafted = draft.len, .len = @min(draft.len, pending_cap), .buf = undefined };
        @memcpy(p.buf[0..p.len], draft[0..p.len]);
        self.pending = p;
    }

    /// A wrapper (e.g. a chat turn-boundary filter) shortened the draft this
    /// `suggest` returned to `new_len` tokens: shrink the pending accounting
    /// to the prefix the decoder will actually verify — the dropped tail must
    /// not settle as rejections. Below `accounting_min_draft` the decoder
    /// falls back to a plain step and the draft is never verified at all:
    /// forget it (mirrors `notePending`).
    pub fn truncatePending(self: *SpeculationIndex, new_len: usize) void {
        if (self.pending) |*p| {
            if (new_len >= p.drafted) return;
            if (new_len < self.accounting_min_draft) {
                self.pending = null;
                return;
            }
            p.drafted = new_len;
            p.len = @min(p.len, new_len);
        }
    }

    /// Per-source lifetime acceptance summary, e.g.
    /// `spec sources: conversation 12/40 ref0 80/96 recycling 3/24 [muted]`.
    pub fn writeSourceSummary(self: *const SpeculationIndex, writer: *std.Io.Writer) !void {
        try writer.print("spec sources: conversation {d}/{d}{s}", .{
            self.conv_gate.total_accepted, self.conv_gate.total_drafted, muteTag(&self.conv_gate),
        });
        for (self.references.items, 0..) |*ref, i| {
            try writer.print(" ref{d} {d}/{d}{s}", .{ i, ref.gate.total_accepted, ref.gate.total_drafted, muteTag(&ref.gate) });
        }
        try writer.print(" recycling {d}/{d}{s}", .{
            self.rec_gate.total_accepted, self.rec_gate.total_drafted, muteTag(&self.rec_gate),
        });
    }

    fn muteTag(gate: *const Gate) []const u8 {
        return if (gate.muted()) " [muted]" else "";
    }
};

// ---------------------------------------------------------------------------
// Cascade unit tests (no model).
// ---------------------------------------------------------------------------

test "longest-match selection with conversation tie-bias; injected reference drafts" {
    const allocator = std.testing.allocator;
    var index = try SpeculationIndex.init(allocator, 1024);
    defer index.deinit();

    // Conversation: [5,6] occurred before, followed by [100,7].
    const stream = [_]usize{ 5, 6, 100, 7, 5, 6 };
    index.observe(&stream);
    // Reference doc also contains [5,6], followed by [200,201], and
    // additionally [7,5,6] -> 200... for the longer-match phase below.
    try index.addReference(&.{ 4, 7, 5, 6, 200, 201, 202, 203 });
    // addReference catch-up: the cursor already matches the observed stream.
    try std.testing.expectEqual(@as(u32, 3), index.references.items[0].cursor.len); // [7,5,6]

    // Reference match (3) beats conversation match (2): draft = doc continuation.
    var buf: [16]usize = undefined;
    {
        const n = index.suggest(&stream, &buf);
        try std.testing.expect(n >= 2);
        try std.testing.expectEqual(@as(usize, 200), buf[0]);
        try std.testing.expectEqual(@as(usize, 201), buf[1]);
        // budget = min(16, 2*(1+3)) = 8; doc has 4 tokens after the match.
        try std.testing.expectEqual(@as(usize, 4), n);
    }

    // Tie: commit a token that resets the reference run, then re-create an
    // equal-length match for both -> conversation must win.
    index.observe(&.{ 100, 7, 9, 5, 6 }); // 9 breaks the doc run; [5,6] matches both at len 2
    try std.testing.expectEqual(@as(u32, 2), index.references.items[0].cursor.len);
    try std.testing.expectEqual(@as(usize, 2), index.conversation.matchLen());
    {
        const n = index.suggest(&.{ 9, 5, 6 }, &buf);
        try std.testing.expect(n >= 1);
        // Conversation recency: latest prior [5,6] (ending index 5) was
        // followed by [100,7,9,...]; the doc would say 200.
        try std.testing.expectEqual(@as(usize, 100), buf[0]);
    }

    // clearReferences drops the doc and any pending accounting.
    index.clearReferences();
    try std.testing.expectEqual(@as(usize, 0), index.references.items.len);
    try std.testing.expectEqual(@as(?SpeculationIndex.Pending, null), index.pending);
}

test "acceptance accounting: common-prefix settlement at the observe seam" {
    const allocator = std.testing.allocator;
    var index = try SpeculationIndex.init(allocator, 1024);
    defer index.deinit();

    // [1,2] recurs, followed by [3,4,5,6].
    index.observe(&.{ 1, 2, 3, 4, 5, 6, 1, 2 });
    var buf: [16]usize = undefined;
    const n = index.suggest(&.{ 1, 2, 3, 4, 5, 6, 1, 2 }, &buf);
    try std.testing.expectEqual(@as(usize, 6), n); // [3,4,5,6,1,2], budget 2*(1+2) = 6
    try std.testing.expectEqualSlices(usize, &.{ 3, 4, 5, 6, 1, 2 }, buf[0..6]);
    // Verifier committed [3,4,99]: 2 accepted + a correction.
    index.observe(&.{ 3, 4, 99 });
    try std.testing.expectEqual(@as(usize, 6), index.conv_gate.total_drafted);
    try std.testing.expectEqual(@as(usize, 2), index.conv_gate.total_accepted);
    try std.testing.expectEqual(@as(?SpeculationIndex.Pending, null), index.pending);
}

test {
    _ = @import("cascade_tests.zig");
}
