//! Token-Recycling adjacency matrix — a model-free draft source for
//! speculative decoding.
//!
//! The idea (Token Recycling, Luo et al. 2024): every verification step the
//! target model already computes full logits for each position, then throws
//! away everything but the sampled token. Recycle them instead: keep one row
//! per vocabulary token holding the MOST RECENT top-K next-token candidates
//! observed at any verified position whose input token was `t`. Drafting is
//! then a chain walk: from the last committed token, repeatedly follow the
//! top-1 candidate (`M[t][0]`) until an unseen row or the budget stops it.
//!
//! Memory: `vocab x K` u32 entries. For the Qwen3 151_936-token vocab at the
//! default K = 8 that is 151_936 * 8 * 4 B = 4_861_952 B (~4.6 MiB).
//! Cold start: the whole matrix is filled with `sentinel` (maxInt(u32)) — no
//! separate seen-bitmap; a row is "unseen" iff its slot 0 is the sentinel.
//!
//! Ownership: the struct owns `m` (allocated in `init`, freed in `deinit`).
//! `update`/`observe*` copy candidate values; no slice is retained.
//!
//! DraftSource contract: `suggest`/`observe`/`observeTopK` below match the
//! method shapes of `speculative.DraftSource`. The user-facing adapter is
//! `spec_cascade.SpeculationIndex`, which composes this matrix with the SAM
//! indices and exposes one `asDraftSource()`.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Re-export: the canonical `TopKRow` lives in `speculative.zig` (the decoder
/// produces it); keep exactly one definition.
pub const TopKRow = @import("core.zig").TopKRow;

/// Token-Recycling matrix with a comptime candidate width `K`.
/// Use the `Recycling` alias (K = 8) unless a different width is measured
/// to be better.
pub fn TokenRecycling(comptime K: usize) type {
    comptime std.debug.assert(K >= 1);
    return struct {
        const Self = @This();

        /// Candidate width (columns per row).
        pub const k = K;
        /// "No candidate / unseen" marker. Vocab ids never reach maxInt(u32).
        pub const sentinel: u32 = std.math.maxInt(u32);

        allocator: Allocator,
        vocab: usize,
        /// Row-major `vocab x K` matrix; row t = most recent top-K candidates
        /// observed after token t. Owned.
        m: []u32,

        pub fn init(allocator: Allocator, vocab: usize) !Self {
            const m = try allocator.alloc(u32, vocab * K);
            @memset(m, sentinel);
            return .{ .allocator = allocator, .vocab = vocab, .m = m };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.m);
            self.* = undefined;
        }

        /// Borrow row `token` (K entries, sentinel-padded). Empty slice when
        /// `token` is out of range. Useful for tree-style drafting later.
        pub fn topkOf(self: *const Self, token: usize) []const u32 {
            if (token >= self.vocab) return &.{};
            return self.m[token * K ..][0..K];
        }

        fn rowMut(self: *Self, token: usize) []u32 {
            return self.m[token * K ..][0..K];
        }

        /// Overwrite row `token` with the freshest observed candidates
        /// (most probable first). Truncates to K; pads with sentinel.
        /// Out-of-range tokens are ignored.
        pub fn update(self: *Self, token: usize, topk: []const u32) void {
            if (token >= self.vocab) return;
            const row = self.rowMut(token);
            const n = @min(K, topk.len);
            @memcpy(row[0..n], topk[0..n]);
            @memset(row[n..], sentinel);
        }

        /// Greedy draft: follow the top-1 chain `M[t][0]` starting from
        /// `last_token`, writing up to `buf.len` tokens. Stops on unseen rows
        /// (sentinel) or out-of-range hops. Returns the number written.
        pub fn draftChain(self: *const Self, last_token: usize, buf: []usize) usize {
            var t = last_token;
            for (buf, 0..) |*out, i| {
                if (t >= self.vocab) return i;
                const next = self.m[t * K];
                if (next == sentinel) return i;
                out.* = next;
                t = next;
            }
            return buf.len;
        }

        // ---- DraftSource method shapes (see header) ----

        /// Draft continuation of `context` into `buf`; returns count.
        pub fn suggest(self: *Self, context: []const usize, buf: []usize) usize {
            if (context.len == 0) return 0;
            return self.draftChain(context[context.len - 1], buf);
        }

        /// Committed tokens are verification ground truth: promote each
        /// committed bigram (a -> b) to the front of row a, most-recent-first
        /// (b moves/inserts at slot 0; existing candidates shift right, the
        /// last one drops). This keeps the top-1 chain aligned with text the
        /// model actually generated, even between `observeTopK` refreshes.
        pub fn observe(self: *Self, committed: []const usize) void {
            if (committed.len < 2) return;
            for (committed[0 .. committed.len - 1], committed[1..]) |a, b| {
                if (a >= self.vocab or b >= self.vocab) continue;
                self.promote(a, @intCast(b));
            }
        }

        /// Recycle verified-step logits: overwrite one row per position.
        pub fn observeTopK(self: *Self, positions: []const TopKRow) void {
            for (positions) |p| self.update(p.token, p.topk);
        }

        fn promote(self: *Self, token: usize, next: u32) void {
            const row = self.rowMut(token);
            if (row[0] == next) return;
            // Find `next` in the row (else evict the last slot).
            var idx: usize = K - 1;
            for (row, 0..) |v, i| {
                if (v == next) {
                    idx = i;
                    break;
                }
            }
            var j = idx;
            while (j > 0) : (j -= 1) row[j] = row[j - 1];
            row[0] = next;
        }
    };
}

/// Default Token-Recycling matrix (K = 8).
pub const Recycling = TokenRecycling(8);

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

test {
    _ = @import("recycling_tests.zig");
}
