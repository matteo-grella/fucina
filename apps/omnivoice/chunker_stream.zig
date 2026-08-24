//! Incremental sentence chunker for streaming TTS. Port of
//! `refs/omnivoice.cpp/src/text-chunker-stream.h`: a wrapper around
//! `chunker.chunkTextPunctuation` that fires on sentence boundaries as text
//! arrives (chunk budget in codepoints).
//!
//! Equivalence (the reference's contract): at `flushEof`, the concatenation
//! in order of every chunk emitted by `pushBytes` + `flushEof` matches
//! `chunkTextPunctuation(full_text, chunk_len, min_chunk_len)` called once
//! on the fully accumulated input — same chunks, same boundaries, same fold
//! decisions. The streaming path trades a one-chunk look-ahead delay for
//! this equivalence: a chunk that just closed is held back until the next
//! one is observed, so the min_chunk_len fold rule can fire:
//!   - the first chunk, if shorter than min_chunk_len, folds into the second
//!   - chunk N > 0 shorter than min_chunk_len folds into chunk N - 1
//!
//! Reference quirk preserved verbatim: the offline re-parse runs with
//! min_chunk_len = 0 (chunks come out stripped) and the fold concatenates
//! those STRIPPED strings directly, so a firing fold drops the inter-chunk
//! whitespace that the one-shot offline fold (which folds raw codepoint
//! runs, stripping only at the end) would keep. Folds only fire on chunks
//! under min_chunk_len = 3 codepoints, so this never triggers on normal
//! sentence text.

const std = @import("std");

const chunker = @import("chunker.zig");

const Allocator = std.mem.Allocator;

pub const Stream = struct {
    allocator: Allocator,
    buffer: std.ArrayList(u8),
    chunk_len: i32,
    min_chunk_len: i32,
    /// Number of chunks observed in the offline re-parse so far; index of
    /// the next chunk to enter the look-ahead pipeline.
    n_seen: usize,
    /// Chunk waiting in the look-ahead slot (owned).
    pending: ?[]u8,

    pub fn init(allocator: Allocator, chunk_len: i32, min_chunk_len: i32) Stream {
        return .{
            .allocator = allocator,
            .buffer = .empty,
            .chunk_len = chunk_len,
            .min_chunk_len = min_chunk_len,
            .n_seen = 0,
            .pending = null,
        };
    }

    pub fn deinit(self: *Stream) void {
        self.buffer.deinit(self.allocator);
        if (self.pending) |p| self.allocator.free(p);
        self.* = undefined;
    }

    /// Appends bytes, reruns the offline chunker on the full buffer, and
    /// advances the look-ahead pipeline. Emits chunks that are no longer at
    /// risk of a fold. The very last chunk in the offline result is always
    /// kept back since a future sentence might still extend it; the second
    /// to last is also kept back so the fold rule has a chance to fire.
    ///
    /// The caller owns the returned slice AND every chunk string in it —
    /// free with `chunker.freeChunks`.
    pub fn pushBytes(self: *Stream, data: []const u8) ![][]u8 {
        if (data.len > 0) {
            try self.buffer.appendSlice(self.allocator, data);
        }
        const all = try chunker.chunkTextPunctuation(self.allocator, self.buffer.items, self.chunk_len, 0);
        return self.advance(all, false);
    }

    /// EOF drain: every chunk in the offline result is now stable. The
    /// look-ahead pipeline runs to completion and the chunker comes out
    /// fresh, so a caller can keep pushing a new stream after the drain
    /// (line-oriented streaming flushes at every newline). Caller frees the
    /// result with `chunker.freeChunks`.
    pub fn flushEof(self: *Stream) ![][]u8 {
        const all = try chunker.chunkTextPunctuation(self.allocator, self.buffer.items, self.chunk_len, 0);
        const out = try self.advance(all, true);
        self.buffer.clearRetainingCapacity();
        self.n_seen = 0;
        return out;
    }

    /// Pumps every newly observed chunk (index >= n_seen) through the
    /// look-ahead slot. An incoming chunk either folds into pending (fold
    /// rule fires) or pushes the previous pending out. At EOF the trailing
    /// pending is emitted as is. Consumes `all` (chunks + slice).
    fn advance(self: *Stream, all: [][]u8, eof: bool) ![][]u8 {
        const allocator = self.allocator;
        defer allocator.free(all);
        errdefer for (all) |c| allocator.free(c);

        var out: std.ArrayList([]u8) = .empty;
        defer out.deinit(allocator);
        errdefer for (out.items) |c| allocator.free(c);

        // The last chunk of the offline result is open: a future sentence
        // could still extend it. Stop one before, except at EOF where the
        // last one is also stable.
        var n_stable = all.len;
        if (!eof) {
            n_stable -|= 1;
        }

        var i = self.n_seen;
        while (i < n_stable) : (i += 1) {
            const incoming = all[i];
            all[i] = &.{}; // ownership taken below

            if (self.pending == null) {
                self.pending = incoming;
                continue;
            }

            // Fold rule. The first chunk, if short, folds into the second
            // one; any later short chunk folds into the previous one. Either
            // way the combined chunk stays in the look-ahead slot.
            const incoming_cp: i64 = @intCast(chunker.utf8Count(incoming));
            const pending_cp: i64 = @intCast(chunker.utf8Count(self.pending.?));

            const incoming_short = incoming_cp < self.min_chunk_len;
            const pending_first_and_short = (i == 1) and pending_cp < self.min_chunk_len;

            if (incoming_short or pending_first_and_short) {
                const combined = try std.mem.concat(allocator, u8, &.{ self.pending.?, incoming });
                allocator.free(self.pending.?);
                allocator.free(incoming);
                self.pending = combined;
                continue;
            }

            // Stable case: emit the pending, the incoming takes the slot.
            try out.append(allocator, self.pending.?);
            self.pending = incoming;
        }
        self.n_seen = n_stable;

        // Free the re-parsed chunks we did not consume: indices below
        // n_seen were consumed in earlier calls (this re-parse rebuilt
        // them), the trailing ones are still open.
        for (all) |*c| {
            if (c.len > 0) allocator.free(c.*);
            c.* = &.{};
        }

        if (eof and self.pending != null) {
            try out.append(allocator, self.pending.?);
            self.pending = null;
        }

        return out.toOwnedSlice(allocator);
    }
};

test {
    _ = @import("chunker_stream_tests.zig");
}
