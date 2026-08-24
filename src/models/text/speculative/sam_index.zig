//! Online suffix automaton (SAM) over token streams — the exact-match draft
//! source for speculative decoding (SAM-Decoding / SuffixDecoding lineage).
//!
//! Why a SAM and not n-gram hashes: O(1) amortized ONLINE extension as the
//! conversation grows token by token, an EXACT unbounded longest-suffix-match
//! length (which drives adaptive draft budgets), and the same construction
//! doubles as a frozen index over injected reference documents (RAG).
//!
//! Layout (vocab is ~151k, so per-state transition arrays are impossible and
//! per-state hash maps are allocation-heavy):
//!   - `states`: flat array of {link, len, sample, first_edge} (<= 2n-1 states
//!     for n tokens, classical bound).
//!   - `trans`: ONE global hash map keyed (state << 32) | token -> next state
//!     (<= 3n-4 transitions). O(1) lookup.
//!   - `edges`: flat pool of per-state singly-linked token lists, mirroring
//!     `trans` keys. This exists ONLY so cloning can enumerate a state's
//!     outgoing transitions (a global map cannot be filtered per state without
//!     a full scan). One pool entry per map entry, no per-state allocation.
//!   - `tokens`: owned copy of the full appended stream (4 B/token). Drafts
//!     are copied straight out of it.
//!
//! ## Self-match exclusion (the subtle part)
//!
//! The trivial SAM matches the whole stream against itself, so "longest suffix
//! that occurs in the stream" is uselessly the whole stream. What drafting
//! needs after appending token t_i is the longest suffix of t_0..t_i with an
//! occurrence ending STRICTLY BEFORE i. Two mechanisms deliver this exactly:
//!
//! 1. Cursor-before-extend. A streaming match cursor (match_state, match_len)
//!    is advanced with t_i against the automaton AS IT WAS, i.e. the automaton
//!    of t_0..t_{i-1}, and only then is the automaton extended with t_i.
//!    A successful transition therefore proves the matched suffix is a
//!    substring of t_0..t_{i-1} — equivalently, it has an occurrence ending at
//!    some j <= i-1 < i. The standard greedy descent (follow suffix links on
//!    mismatch, take the transition, len = link target's len on each drop)
//!    yields the LONGEST such suffix; `matchLen()` is exact, proven against
//!    brute force in the tests below. Overlapping prior occurrences (e.g. the
//!    suffix "aa" of "aaa" ending at 1) are correctly counted as prior.
//!
//!    Clone fix-up: extending with t_i may split the cursor's state q into
//!    (q, clone). If match_len <= len(clone) the matched string moved into
//!    clone, so the cursor is redirected — without this, later transitions
//!    would be taken from the wrong state and matchLen could overshoot.
//!
//! 2. Sample discipline. Each state stores ONE occurrence end index `sample`
//!    (valid for every string in the state — all strings of a SAM state share
//!    one endpos set). Drafting copies `tokens[sample+1..]`. Every write to
//!    `sample` is provably (a) a member of the state's endpos set and (b)
//!    strictly less than the stream length at the time of any later query, so
//!    a draft can never copy from the current (self) occurrence:
//!      - new state cur_i: sample = i; the cursor can only reach cur_i at some
//!        later step j > i (it advances pre-extension, when cur_j' does not
//!        exist yet), so i < current end.
//!      - clone_i: sample is INHERITED from the split state q (endpos(q) is a
//!        subset of endpos(clone)), not set to i — the cursor may be sitting
//!        on clone_i at the end of step i, and an inherited sample < i keeps
//!        the immediately-following draft pointed at a genuine prior
//!        occurrence instead of the (empty) continuation of the stream end.
//!      - link-walk refresh: every state p the construction walks at step i
//!        lies on the suffix-link chain of `last`, so i-1 is in endpos(p);
//!        overwriting sample(p) = i-1 is valid and strictly newer.
//!      - deferred cursor refresh (recency, "most recent occurrence wins"):
//!        after step i the cursor's state contains the matched suffix ending
//!        at i, so i is in its endpos — but writing i immediately would equal
//!        the current end. Instead the write is deferred to the START of step
//!        i+1 (sample(match_state) = i = pos-1), when every future query end
//!        is >= i+1 > i. This refreshes exactly the states the drafting path
//!        traverses, one step behind, which is where last-occurrence semantics
//!        matters; each append does O(1) such writes.
//!
//!    Recency choice: drafts therefore follow the MOST RECENT prior occurrence
//!    seen by the construction walk / cursor path (research verdict: recency
//!    wins for code and editing flows), falling back to the occurrence
//!    recorded at state-creation time (the first one) when the suffix has not
//!    been re-traversed since.
//!
//! ## Frozen mode (the RAG seam)
//!
//! Build a SamIndex per tokenized reference document, `freeze()` it, then run
//! any number of external `Cursor`s over it as the conversation proceeds
//! (`advance` per committed token, `draftFrom` to copy the document's
//! continuation of the current match). Freezing matters for correctness:
//! appends can clone states, and only the INTERNAL cursor gets the clone
//! fix-up — external cursors over a still-growing automaton would silently
//! dangle on split states. Queries against a frozen document need no
//! self-match exclusion at all (the query stream is not part of the index),
//! so any stored sample is a valid occurrence.
//!
//! DraftSource contract: `suggest`/`observe`/`observeTopK` on SamIndex (online
//! conversation index) and on `FrozenSource` (per-conversation cursor over a
//! frozen document) match the method shapes of `speculative.DraftSource`.
//! The user-facing adapter is `spec_cascade.SpeculationIndex`, which composes
//! a conversation SamIndex, frozen reference indices, and the Token-Recycling
//! matrix behind one `asDraftSource()`.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Re-export: the canonical `TopKRow` lives in `speculative.zig` (the decoder
/// produces it); keep exactly one definition.
pub const TopKRow = @import("core.zig").TopKRow;

pub const SamIndex = struct {
    /// Hard stream-length ceiling. Internal indices are 32-bit: states fit
    /// u32/i32 (<= 2n+1) and edge-pool indices fit i32 (<= 3n-4), so n must
    /// stay <= (2^31 - 1 + 4) / 3; 2^29 keeps a 4x margin on states and
    /// ~1.34x on edges while being far beyond any realistic conversation or
    /// reference document. `appendOne` rejects (and degrades on) growth past
    /// it instead of overflowing a cast in ReleaseFast.
    pub const max_stream_len: usize = 1 << 29;

    const State = struct {
        /// Suffix link; -1 only for the root.
        link: i32,
        /// Length of the longest string in this state's endpos class.
        len: u32,
        /// End index (into `tokens`) of one occurrence of every string in
        /// this state. See "Sample discipline" in the file header.
        sample: u32,
        /// Head of this state's outgoing-transition token list in `edges`
        /// (-1 = none). Mirrors `trans` keys for clone enumeration.
        first_edge: i32,
    };

    const Edge = struct {
        token: u32,
        next: i32,
    };

    allocator: Allocator,
    /// State 0 is the root.
    states: std.ArrayList(State),
    /// All transitions: (state << 32) | token -> next state.
    trans: std.AutoHashMapUnmanaged(u64, u32),
    /// Flat pool of per-state transition-token lists (see State.first_edge).
    edges: std.ArrayList(Edge),
    /// Owned copy of the appended stream.
    tokens: std.ArrayList(u32),
    /// State of the whole current stream.
    last: u32,
    /// Streaming self-match-excluded cursor (see file header).
    match_state: u32,
    match_len: u32,
    /// No more appends; external Cursors become safe.
    frozen: bool,
    /// A failed append poisons the index (its stream copy / automaton would
    /// desync from the caller's committed stream): queries return 0 forever.
    degraded: bool,
    /// `draft` returns 0 when the current match is shorter than this.
    min_match: usize,
    /// Draft budget: `draft` copies at most min(buf.len, max_draft) tokens.
    max_draft: usize,

    pub fn init(allocator: Allocator) !SamIndex {
        var states: std.ArrayList(State) = .empty;
        errdefer states.deinit(allocator);
        try states.append(allocator, .{ .link = -1, .len = 0, .sample = 0, .first_edge = -1 });
        return .{
            .allocator = allocator,
            .states = states,
            .trans = .empty,
            .edges = .empty,
            .tokens = .empty,
            .last = 0,
            .match_state = 0,
            .match_len = 0,
            .frozen = false,
            .degraded = false,
            .min_match = 2,
            .max_draft = 16,
        };
    }

    pub fn deinit(self: *SamIndex) void {
        self.states.deinit(self.allocator);
        self.trans.deinit(self.allocator);
        self.edges.deinit(self.allocator);
        self.tokens.deinit(self.allocator);
        self.* = undefined;
    }

    inline fn key(state: u32, token: u32) u64 {
        return (@as(u64, state) << 32) | token;
    }

    pub fn tokenCount(self: *const SamIndex) usize {
        return self.tokens.items.len;
    }

    pub fn stateCount(self: *const SamIndex) usize {
        return self.states.items.len;
    }

    pub fn transitionCount(self: *const SamIndex) usize {
        return self.trans.count();
    }

    // ---- online construction ----

    /// Extend the index with newly committed tokens. O(1) amortized each.
    /// On error the index is poisoned (`degraded`): all queries return 0 and
    /// further appends fail — a half-applied append must never serve drafts.
    pub fn append(self: *SamIndex, new_tokens: []const usize) !void {
        std.debug.assert(!self.frozen);
        if (self.degraded) return error.Degraded;
        for (new_tokens) |t| try self.appendOne(t);
    }

    fn appendOne(self: *SamIndex, token: usize) !void {
        errdefer self.degraded = true;
        if (token > std.math.maxInt(u32)) return error.TokenOutOfRange;
        if (self.tokens.items.len >= max_stream_len) return error.StreamTooLong;
        const t: u32 = @intCast(token);
        const pos: u32 = @intCast(self.tokens.items.len); // index of the new token

        // Deferred recency refresh: the cursor's state contained the suffix
        // matched at the PREVIOUS position pos-1, so pos-1 is in its endpos;
        // writing it now (not at step pos-1) keeps sample < any query end.
        if (pos > 0 and self.match_len > 0) {
            self.states.items[self.match_state].sample = pos - 1;
        }

        // 1) Advance the match cursor against the automaton AS IT IS NOW
        //    (i.e. over t_0..t_{pos-1}): self-match exclusion, see header.
        self.match_state, self.match_len = self.advanceRaw(self.match_state, self.match_len, t);

        // 2) Standard SAM extension with t.
        const cur: u32 = @intCast(self.states.items.len);
        try self.states.append(self.allocator, .{
            .link = -1,
            .len = self.states.items[self.last].len + 1,
            .sample = pos,
            .first_edge = -1,
        });

        var p: i32 = @intCast(self.last);
        while (p >= 0 and self.trans.get(key(@intCast(p), t)) == null) {
            const pu: u32 = @intCast(p);
            try self.addTransition(pu, t, cur);
            // pu is on the suffix-link chain of `last`: pos-1 is in its endpos.
            if (pu != 0) self.states.items[pu].sample = pos - 1;
            p = self.states.items[pu].link;
        }
        if (p < 0) {
            self.states.items[cur].link = 0;
        } else {
            const pu: u32 = @intCast(p);
            const q = self.trans.get(key(pu, t)).?;
            if (self.states.items[pu].len + 1 == self.states.items[q].len) {
                self.states.items[cur].link = @intCast(q);
            } else {
                // Split q: clone takes q's short strings (lengths <= len(p)+1).
                const clone: u32 = @intCast(self.states.items.len);
                const q_snapshot = self.states.items[q];
                try self.states.append(self.allocator, .{
                    .link = q_snapshot.link,
                    .len = self.states.items[pu].len + 1,
                    // Inherit (endpos(q) subset of endpos(clone)): guaranteed
                    // < pos, so an immediate draft from clone stays prior.
                    .sample = q_snapshot.sample,
                    .first_edge = -1,
                });
                // Copy q's outgoing transitions (before any redirect below —
                // redirects only touch transitions INTO q).
                var e = q_snapshot.first_edge;
                while (e >= 0) {
                    const edge = self.edges.items[@intCast(e)];
                    const target = self.trans.get(key(q, edge.token)).?;
                    try self.addTransition(clone, edge.token, target);
                    e = edge.next;
                }
                // Redirect transitions into q from the suffix chain.
                var pr: i32 = p;
                while (pr >= 0) {
                    const pru: u32 = @intCast(pr);
                    const slot = self.trans.getPtr(key(pru, t)) orelse break;
                    if (slot.* != q) break;
                    slot.* = clone;
                    if (pru != 0) self.states.items[pru].sample = pos - 1;
                    pr = self.states.items[pru].link;
                }
                self.states.items[q].link = @intCast(clone);
                self.states.items[cur].link = @intCast(clone);
                // Cursor fix-up: the matched string may have moved into clone.
                if (self.match_state == q and self.match_len <= self.states.items[clone].len) {
                    self.match_state = clone;
                }
            }
        }
        self.last = cur;
        try self.tokens.append(self.allocator, t);

        // Cursor invariants: the matched string is a member of match_state,
        // and its sample is a strictly-prior occurrence end.
        std.debug.assert(self.match_len <= self.states.items[self.match_state].len);
        std.debug.assert(self.match_state != 0 or self.match_len == 0);
        if (self.match_state != 0) {
            const lnk: u32 = @intCast(self.states.items[self.match_state].link);
            std.debug.assert(self.match_len > self.states.items[lnk].len);
            std.debug.assert(self.states.items[self.match_state].sample < pos);
        }
    }

    fn addTransition(self: *SamIndex, from: u32, token: u32, to: u32) !void {
        try self.trans.putNoClobber(self.allocator, key(from, token), to);
        try self.edges.append(self.allocator, .{
            .token = token,
            .next = self.states.items[from].first_edge,
        });
        self.states.items[from].first_edge = @intCast(self.edges.items.len - 1);
    }

    /// Greedy longest-match step: follow suffix links until a transition on
    /// `t` exists (len becomes the link target's len on each drop), then take
    /// it; (root, 0) if `t` never occurred. Returns {state, len}.
    fn advanceRaw(self: *const SamIndex, state: u32, len: u32, t: u32) struct { u32, u32 } {
        var s = state;
        var l = len;
        while (true) {
            if (self.trans.get(key(s, t))) |next| return .{ next, l + 1 };
            if (s == 0) return .{ 0, 0 };
            s = @intCast(self.states.items[s].link);
            l = self.states.items[s].len;
        }
    }

    // ---- online queries ----

    /// Length of the longest suffix of the whole appended stream that also
    /// occurs ending strictly before the last position. Exact (see header).
    pub fn matchLen(self: *const SamIndex) usize {
        if (self.degraded) return 0;
        return self.match_len;
    }

    /// Draft = the tokens that FOLLOWED the most recent recorded prior
    /// occurrence of the current longest matching suffix. Fills `buf`,
    /// returns count <= min(buf.len, max_draft). 0 when matchLen < min_match.
    pub fn draft(self: *const SamIndex, buf: []usize) usize {
        if (self.degraded) return 0;
        if (self.match_len < self.min_match) return 0;
        return self.copyContinuation(self.states.items[self.match_state].sample, buf);
    }

    fn copyContinuation(self: *const SamIndex, end_index: u32, buf: []usize) usize {
        const start = @as(usize, end_index) + 1;
        const n = self.tokens.items.len;
        if (start >= n) return 0;
        const count = @min(@min(buf.len, self.max_draft), n - start);
        for (buf[0..count], self.tokens.items[start..][0..count]) |*out, tok| out.* = tok;
        return count;
    }

    // ---- frozen mode (RAG references) ----

    /// No more appends; external Cursors over this index are now stable
    /// (no future clone can split a state under them).
    pub fn freeze(self: *SamIndex) void {
        self.frozen = true;
    }

    /// External match state over a FROZEN index: one per (document,
    /// conversation) pair, advanced as generation proceeds.
    pub const Cursor = struct {
        state: u32 = 0,
        len: u32 = 0,
    };

    /// Advance `cursor` with one committed token of the (external) query
    /// stream. After the call, cursor.len = length of the longest suffix of
    /// the query stream that is a substring of this document (mismatches
    /// reset via suffix links, exactly like the internal cursor).
    pub fn advance(self: *const SamIndex, cursor: *Cursor, token: usize) void {
        std.debug.assert(self.frozen);
        if (token > std.math.maxInt(u32)) {
            cursor.* = .{};
            return;
        }
        cursor.state, cursor.len = self.advanceRaw(cursor.state, cursor.len, @intCast(token));
    }

    /// Draft the document's continuation of the cursor's current match.
    /// Same budget/min_match contract as `draft`.
    pub fn draftFrom(self: *const SamIndex, cursor: Cursor, buf: []usize) usize {
        std.debug.assert(self.frozen);
        if (cursor.len < self.min_match) return 0;
        return self.copyContinuation(self.states.items[cursor.state].sample, buf);
    }

    // ---- DraftSource method shapes (see header) ----

    /// Draft a continuation of the committed stream. `context` is only used
    /// as a desync guard: if its last token differs from the last token this
    /// index observed, no draft is offered (the index lags the verifier).
    pub fn suggest(self: *SamIndex, context: []const usize, buf: []usize) usize {
        if (self.degraded) return 0;
        if (context.len > 0) {
            const n = self.tokens.items.len;
            if (n == 0 or context[context.len - 1] != self.tokens.items[n - 1]) return 0;
        }
        return self.draft(buf);
    }

    /// Feed committed tokens. Errors degrade the index (queries return 0
    /// forever) instead of propagating — a draft source must never sit
    /// desynced from the committed stream.
    pub fn observe(self: *SamIndex, committed: []const usize) void {
        if (self.degraded or self.frozen) return;
        self.append(committed) catch {};
    }

    /// The SAM learns from committed tokens only; recycled logits go to
    /// `recycling.Recycling`.
    pub fn observeTopK(self: *SamIndex, positions: []const TopKRow) void {
        _ = self;
        _ = positions;
    }
};

/// Per-conversation DraftSource over one frozen reference document: owns the
/// cursor, borrows the index. Build the SamIndex from the tokenized document,
/// `freeze()` it, then hand each conversation its own FrozenSource.
pub const FrozenSource = struct {
    index: *const SamIndex,
    cursor: SamIndex.Cursor = .{},

    pub fn suggest(self: *FrozenSource, context: []const usize, buf: []usize) usize {
        _ = context; // the cursor already encodes the committed stream
        return self.index.draftFrom(self.cursor, buf);
    }

    pub fn observe(self: *FrozenSource, committed: []const usize) void {
        for (committed) |t| self.index.advance(&self.cursor, t);
    }

    pub fn observeTopK(self: *FrozenSource, positions: []const TopKRow) void {
        _ = self;
        _ = positions;
    }
};

test {
    _ = @import("sam_index_tests.zig");
}
