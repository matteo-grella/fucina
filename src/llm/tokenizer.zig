//! Native byte-level BPE tokenizer (GPT-2 / Qwen family), built from a model's
//! own GGUF metadata — no external tokenizer dependency, no per-model hardcoding.
//!
//! Adapted from ZINC's tokenizer (github.com/zolotukhin/zinc, MIT — see
//! refs/zinc/src/model/tokenizer.zig), reduced to a pure, format-driven core:
//! vocabulary, merge ranks, and special-token IDs all come from GGUF metadata
//! (`tokenizer.ggml.*`); special tokens may be overridden by the caller. The
//! SentencePiece/Gemma paths, chat templates, and tool formatting were dropped.
//!
//! The tokenizer copies the vocab/merge bytes it needs, so it stays valid after
//! the source `gguf.File` is freed.

const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;
const ucat = @import("unicode_categories.zig");

const Allocator = std.mem.Allocator;

pub const Error = error{
    NoTokenizerVocab,
    /// The GGUF declares a tokenizer format this module does not implement
    /// (only byte-level BPE — gpt2/qwen-style — is supported).
    UnsupportedTokenizerFormat,
    TokenizerTooLarge,
} || Allocator.Error;

/// Special-token configuration. Defaults are read from GGUF metadata; pass a
/// non-null field in `overrides` to force a value for a model that omits it.
pub const SpecialTokens = struct {
    bos: ?u32 = null,
    eos: ?u32 = null,
    prepend_bos: bool = false,
    append_eos: bool = false,
};

pub const Tokenizer = struct {
    allocator: Allocator,
    /// One owned buffer holding every token's bytes back-to-back.
    vocab_blob: []u8,
    /// token id → bytes (slices into `vocab_blob`).
    vocab: [][]const u8,
    /// token bytes → token id.
    token_to_id: std.StringHashMap(u32),
    /// One owned buffer holding every "<first> <second>" merge rule.
    merges_blob: []u8,
    /// "<first> <second>" → merge rank (priority; lower = applied first).
    merge_ranks: std.StringHashMap(u32),
    special: SpecialTokens,
    /// Non-null when the GGUF declared a pretokenizer (`tokenizer.ggml.pre`)
    /// OTHER than an implemented chunker (owned copy of that id).
    /// Encoding still proceeds with the qwen2 rules — the qwen3 family is
    /// qwen2-pre — but chunk boundaries may not match such a model's
    /// reference tokenizer, so token-ID parity is not guaranteed. Surfaced
    /// here (and logged once at init) instead of silently mis-tokenizing.
    pre_mismatch: ?[]u8 = null,
    /// Which implemented pre-tokenizer splits text into BPE chunks.
    pre: Pre = .qwen2,
    /// Special tokens the encoder single-id-matches before pretokenization —
    /// llama.cpp's special-token cache: every CONTROL or USER_DEFINED vocab
    /// entry (`tokenizer.ggml.token_type` 3 / 4), sorted longest-text-first
    /// so longer markers claim their spans before shorter ones. Empty when
    /// the vocab carries no type metadata (`initFromParts` callers); the
    /// encoder then falls back to matching `<|...|>`-shaped markers only.
    specials: []const SpecialEntry = &.{},

    pub const SpecialEntry = struct {
        /// Slice into `vocab_blob`.
        text: []const u8,
        id: u32,
    };

    pub const Pre = enum { qwen2, qwen35, joyai_llm, glm4, inkling };

    /// Build a tokenizer from GGUF metadata. `overrides` fields, when non-null,
    /// replace the metadata-derived special tokens.
    pub fn initFromGguf(allocator: Allocator, file: *const gguf.File, overrides: SpecialTokens) !Tokenizer {
        const tokens_arr = file.getArray("tokenizer.ggml.tokens") orelse return Error.NoTokenizerVocab;
        if (tokens_arr.item_type != 8) return Error.NoTokenizerVocab;
        const merges_arr = file.getArray("tokenizer.ggml.merges");
        // Only byte-level BPE is implemented: it needs merge rules and must not
        // be a SentencePiece-scored model.
        if (merges_arr == null or merges_arr.?.len == 0 or file.getArray("tokenizer.ggml.scores") != null) {
            return Error.UnsupportedTokenizerFormat;
        }

        const token_strings = try tokens_arr.stringSlices(allocator);
        defer allocator.free(token_strings);
        const merge_strings = try merges_arr.?.stringSlices(allocator);
        defer allocator.free(merge_strings);

        var special = SpecialTokens{
            .bos = if (file.getInt("tokenizer.ggml.bos_token_id")) |v| @intCast(v) else null,
            .eos = if (file.getInt("tokenizer.ggml.eos_token_id")) |v| @intCast(v) else null,
            .prepend_bos = file.getBool("tokenizer.ggml.add_bos_token") orelse false,
            .append_eos = file.getBool("tokenizer.ggml.add_eos_token") orelse false,
        };
        if (overrides.bos) |v| special.bos = v;
        if (overrides.eos) |v| special.eos = v;
        if (overrides.prepend_bos) special.prepend_bos = true;
        if (overrides.append_eos) special.append_eos = true;

        var tok = try initFromParts(allocator, token_strings, merge_strings, special);
        errdefer tok.deinit();
        // The special-token cache from `tokenizer.ggml.token_type` (i32 per
        // token: 3 = CONTROL, 4 = USER_DEFINED — both single-id-match during
        // encoding, exactly llama.cpp's cache_special_tokens).
        if (file.getArray("tokenizer.ggml.token_type")) |types_arr| {
            if (types_arr.item_type == 5 and types_arr.len == tok.vocab.len) {
                var count: usize = 0;
                for (0..types_arr.len) |i| {
                    const t = std.mem.readInt(i32, types_arr.data[i * 4 ..][0..4], .little);
                    if ((t == 3 or t == 4) and tok.vocab[i].len > 0) count += 1;
                }
                const entries = try allocator.alloc(SpecialEntry, count);
                var n: usize = 0;
                for (0..types_arr.len) |i| {
                    const t = std.mem.readInt(i32, types_arr.data[i * 4 ..][0..4], .little);
                    if ((t == 3 or t == 4) and tok.vocab[i].len > 0) {
                        entries[n] = .{ .text = tok.vocab[i], .id = @intCast(i) };
                        n += 1;
                    }
                }
                // Longest first; id ascending on ties for determinism.
                std.mem.sort(SpecialEntry, entries, {}, struct {
                    fn lessThan(_: void, a: SpecialEntry, b: SpecialEntry) bool {
                        if (a.text.len != b.text.len) return a.text.len > b.text.len;
                        return a.id < b.id;
                    }
                }.lessThan);
                tok.specials = entries;
            }
        }
        // Record (don't fail on) a pretokenizer mismatch — see `pre_mismatch`.
        if (file.getString("tokenizer.ggml.pre")) |pre| {
            if (std.mem.eql(u8, pre, "qwen2")) {
                tok.pre = .qwen2;
            } else if (std.mem.eql(u8, pre, "qwen35")) {
                tok.pre = .qwen35;
            } else if (std.mem.eql(u8, pre, "joyai-llm")) {
                tok.pre = .joyai_llm;
            } else if (std.mem.eql(u8, pre, "glm4") or std.mem.eql(u8, pre, "chatglm-bpe")) {
                tok.pre = .glm4;
            } else if (std.mem.eql(u8, pre, "inkling")) {
                tok.pre = .inkling;
            } else {
                tok.pre_mismatch = try allocator.dupe(u8, pre);
                std.log.warn(
                    "tokenizer: GGUF declares pretokenizer '{s}', but the qwen2 chunker is used — token-ID parity is not guaranteed",
                    .{pre},
                );
            }
        }
        return tok;
    }

    /// Build a tokenizer from raw vocabulary and "<a> <b>" merge-rule strings
    /// (each copied into owned storage). The byte strings must already be in the
    /// model's GPT-2 byte-level form, as `tokenizer.ggml.{tokens,merges}` are.
    pub fn initFromParts(
        allocator: Allocator,
        vocab_strings: []const []const u8,
        merge_strings: []const []const u8,
        special: SpecialTokens,
    ) !Tokenizer {
        // --- vocab: copy bytes into one blob, build id↔bytes maps ---
        if (vocab_strings.len > std.math.maxInt(u32)) return Error.TokenizerTooLarge;
        var vocab_bytes: usize = 0;
        for (vocab_strings) |s| vocab_bytes = std.math.add(usize, vocab_bytes, s.len) catch return Error.TokenizerTooLarge;

        const vocab_blob = try allocator.alloc(u8, vocab_bytes);
        errdefer allocator.free(vocab_blob);
        const vocab = try allocator.alloc([]const u8, vocab_strings.len);
        errdefer allocator.free(vocab);

        var token_to_id = std.StringHashMap(u32).init(allocator);
        errdefer token_to_id.deinit();
        try token_to_id.ensureTotalCapacity(@intCast(vocab_strings.len));

        var off: usize = 0;
        for (vocab_strings, 0..) |s, i| {
            @memcpy(vocab_blob[off..][0..s.len], s);
            vocab[i] = vocab_blob[off..][0..s.len];
            off += s.len;
            // First writer wins on duplicate token bytes (matches lowest id).
            if (!token_to_id.contains(vocab[i])) token_to_id.putAssumeCapacity(vocab[i], @intCast(i));
        }

        // --- merges: copy "<a> <b>" rules into a blob, map to rank ---
        if (merge_strings.len > std.math.maxInt(u32)) return Error.TokenizerTooLarge;
        var merges_bytes: usize = 0;
        for (merge_strings) |s| merges_bytes = std.math.add(usize, merges_bytes, s.len) catch return Error.TokenizerTooLarge;

        const merges_blob = try allocator.alloc(u8, merges_bytes);
        errdefer allocator.free(merges_blob);
        var merge_ranks = std.StringHashMap(u32).init(allocator);
        errdefer merge_ranks.deinit();
        try merge_ranks.ensureTotalCapacity(@intCast(merge_strings.len));

        off = 0;
        for (merge_strings, 0..) |s, rank| {
            // A valid merge is "first second"; skip malformed entries.
            if (std.mem.indexOfScalar(u8, s, ' ') == null) continue;
            @memcpy(merges_blob[off..][0..s.len], s);
            const key = merges_blob[off..][0..s.len];
            off += s.len;
            if (!merge_ranks.contains(key)) merge_ranks.putAssumeCapacity(key, @intCast(rank));
        }

        return .{
            .allocator = allocator,
            .vocab_blob = vocab_blob,
            .vocab = vocab,
            .token_to_id = token_to_id,
            .merges_blob = merges_blob,
            .merge_ranks = merge_ranks,
            .special = special,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        if (self.specials.len > 0) self.allocator.free(self.specials);
        if (self.pre_mismatch) |p| self.allocator.free(p);
        self.merge_ranks.deinit();
        self.token_to_id.deinit();
        self.allocator.free(self.merges_blob);
        self.allocator.free(self.vocab);
        self.allocator.free(self.vocab_blob);
        self.* = undefined;
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.vocab.len;
    }

    pub fn eosId(self: *const Tokenizer) ?u32 {
        return self.special.eos;
    }

    pub fn bosId(self: *const Tokenizer) ?u32 {
        return self.special.bos;
    }

    pub fn isEos(self: *const Tokenizer, id: u32) bool {
        return self.special.eos != null and id == self.special.eos.?;
    }

    /// Look up the id of a token by its exact bytes (e.g. a special token like
    /// "<|im_end|>"). Returns null if the token is not in the vocabulary.
    pub fn tokenId(self: *const Tokenizer, token: []const u8) ?u32 {
        return self.token_to_id.get(token);
    }

    // ---- encode ----

    /// Encode UTF-8 `text` into token IDs (allocated with `allocator`), resolving
    /// `<|...|>` special-token markers to their IDs and applying the model's
    /// BOS/EOS policy. Caller frees the result.
    pub fn encode(self: *const Tokenizer, allocator: Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);

        if (self.special.prepend_bos) {
            if (self.special.bos) |bos| try out.append(allocator, bos);
        }
        try self.encodeWithSpecials(allocator, text, &out);
        if (self.special.append_eos) {
            if (self.special.eos) |eos| try out.append(allocator, eos);
        }
        return out.toOwnedSlice(allocator);
    }

    /// Encode without the BOS/EOS policy — the caller (e.g. a chat template)
    /// controls all structural tokens. `<|...|>` markers are still resolved.
    pub fn encodeRaw(self: *const Tokenizer, allocator: Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);
        try self.encodeWithSpecials(allocator, text, &out);
        return out.toOwnedSlice(allocator);
    }

    /// Encode with NO special-token resolution at all: pretokenize + BPE only.
    /// For callers that partition the text on their own control-token set
    /// (which may not follow the `<|...|>` shape, e.g. `<IMG_CONTEXT>`) and
    /// must not have `<|...|>` spans inside a partition resolved atomically.
    pub fn encodePlainAppend(self: *const Tokenizer, allocator: Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        return self.encodeRegular(allocator, text, out);
    }

    /// Append one token's decoded bytes to `out` (reverses the byte-level map).
    /// Public so a streaming decoder can accumulate across tokens.
    pub fn decodeAppend(self: *const Tokenizer, allocator: Allocator, id: u32, out: *std.ArrayList(u8)) !void {
        return self.decodeTokenInto(allocator, id, out);
    }

    /// Single-id-match special tokens, then BPE-encode the text between them.
    ///
    /// With a special-token cache (GGUF vocabs with `token_type` metadata)
    /// this is llama.cpp's `tokenizer_st_partition`: fragments start as one
    /// raw span, and each special token — longest first — claims every
    /// occurrence inside the remaining raw fragments, so a longer marker can
    /// never be broken up by a shorter one matching inside it.
    ///
    /// Without a cache, fall back to resolving `<|...|>`-shaped markers
    /// against the vocabulary. A "<|" that does NOT open a known marker is
    /// left in place for normal pretokenization — forcing a split there
    /// would change chunk boundaries (and token IDs) around bare "<|".
    fn encodeWithSpecials(self: *const Tokenizer, allocator: Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        if (self.specials.len > 0) return self.encodePartitioned(allocator, text, out);
        var pos: usize = 0; // start of the pending regular-text span
        var search: usize = 0; // marker scan cursor (>= pos)
        while (search < text.len) {
            const lt = std.mem.indexOfPos(u8, text, search, "<|") orelse break;
            // Resolve "<|...|>" against the vocabulary.
            if (std.mem.indexOfPos(u8, text, lt + 2, "|>")) |close| {
                const marker = text[lt .. close + 2];
                if (self.token_to_id.get(marker)) |id| {
                    if (lt > pos) try self.encodeRegular(allocator, text[pos..lt], out);
                    try out.append(allocator, id);
                    pos = close + 2;
                    search = pos;
                    continue;
                }
            }
            // Not a known marker: skip the candidate, keep the span intact.
            search = lt + 2;
        }
        if (pos < text.len) try self.encodeRegular(allocator, text[pos..], out);
    }

    fn encodePartitioned(self: *const Tokenizer, allocator: Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        // A fragment is either an unclaimed raw span (`id == null`) or one
        // claimed special token.
        const Fragment = struct { start: usize, end: usize, id: ?u32 };
        var frags: std.ArrayList(Fragment) = .empty;
        defer frags.deinit(allocator);
        try frags.append(allocator, .{ .start = 0, .end = text.len, .id = null });

        for (self.specials) |sp| {
            var i: usize = 0;
            while (i < frags.items.len) : (i += 1) {
                const fr = frags.items[i];
                if (fr.id != null) continue;
                const found = std.mem.indexOf(u8, text[fr.start..fr.end], sp.text) orelse continue;
                const m0 = fr.start + found;
                const m1 = m0 + sp.text.len;
                // Replace the raw fragment with [left raw][token][right raw];
                // the loop revisits the right remainder for further
                // occurrences of this same special.
                frags.items[i] = .{ .start = m0, .end = m1, .id = sp.id };
                if (m1 < fr.end) try frags.insert(allocator, i + 1, .{ .start = m1, .end = fr.end, .id = null });
                if (fr.start < m0) {
                    try frags.insert(allocator, i, .{ .start = fr.start, .end = m0, .id = null });
                    i += 1; // stay on the claimed fragment
                }
            }
        }

        for (frags.items) |fr| {
            if (fr.id) |id| {
                try out.append(allocator, id);
            } else {
                try self.encodeRegular(allocator, text[fr.start..fr.end], out);
            }
        }
    }

    /// Pretokenize with the Qwen2-family pattern, then BPE-encode each chunk.
    ///
    /// The chunking is a faithful port of llama.cpp's hand-rolled codepoint
    /// loop for the Qwen2 pretokenizer regex (see `qwen2ChunkEnd`). The text
    /// is decoded to codepoints once because the state machine needs one
    /// codepoint of lookahead and chunk boundaries in byte offsets.
    fn encodeRegular(self: *const Tokenizer, allocator: Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        if (text.len == 0) return;

        if (self.pre == .joyai_llm) {
            // The JoyAI splitter is byte-oriented with inline UTF-8 peeking;
            // no codepoint pre-decode needed.
            var pos: usize = 0;
            while (pos < text.len) {
                const end = joyaiChunkEnd(text, pos);
                try self.encodeChunk(allocator, text[pos..end], out);
                pos = end;
            }
            return;
        }

        var cps: std.ArrayList(u32) = .empty;
        defer cps.deinit(allocator);
        // offs[i] = byte offset of codepoint i; sentinel offs[n] = text.len.
        var offs: std.ArrayList(usize) = .empty;
        defer offs.deinit(allocator);
        try cps.ensureTotalCapacity(allocator, text.len);
        try offs.ensureTotalCapacity(allocator, text.len + 1);

        var i: usize = 0;
        while (i < text.len) {
            // Invalid UTF-8: each undecodable byte classifies as one U+FFFD
            // codepoint; the raw bytes stay in the emitted chunk. NOT
            // llama.cpp parity on malformed input — see qwen2ChunkEnd's
            // "Known deviation" note.
            const advance: usize, const cp: u32 = blk: {
                const len = std.unicode.utf8ByteSequenceLength(text[i]) catch break :blk .{ 1, 0xFFFD };
                if (i + len > text.len) break :blk .{ 1, 0xFFFD };
                const cp = std.unicode.utf8Decode(text[i..][0..len]) catch break :blk .{ 1, 0xFFFD };
                break :blk .{ len, cp };
            };
            offs.appendAssumeCapacity(i);
            cps.appendAssumeCapacity(cp);
            i += advance;
        }
        offs.appendAssumeCapacity(text.len);

        var pos: usize = 0;
        while (pos < cps.items.len) {
            const end = switch (self.pre) {
                .glm4 => glm4ChunkEnd(cps.items, pos),
                .qwen35 => qwen35ChunkEnd(cps.items, pos),
                .inkling => inklingChunkEnd(cps.items, pos),
                else => qwen2ChunkEnd(cps.items, pos),
            };
            try self.encodeChunk(allocator, text[offs.items[pos]..offs.items[end]], out);
            pos = end;
        }
    }

    /// Byte-level encode one pretoken chunk, apply merges, emit token IDs.
    fn encodeChunk(self: *const Tokenizer, allocator: Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        if (text.len == 0) return;

        // Each raw byte → its GPT-2 byte-level Unicode symbol (an owned string).
        var symbols: std.ArrayList([]u8) = .empty;
        defer {
            for (symbols.items) |s| allocator.free(s);
            symbols.deinit(allocator);
        }
        try symbols.ensureTotalCapacity(allocator, text.len);
        for (text) |byte| {
            var enc: [4]u8 = undefined;
            const len = gpt2ByteToUnicode(byte, &enc);
            const sym = try allocator.alloc(u8, len);
            @memcpy(sym, enc[0..len]);
            symbols.appendAssumeCapacity(sym);
        }

        try self.applyMerges(allocator, &symbols);

        for (symbols.items) |sym| {
            if (self.token_to_id.get(sym)) |id| {
                try out.append(allocator, id);
            } else {
                // Fallback: emit each byte-level character of the symbol. Each is
                // a single-byte token in the vocabulary for a valid gpt2 model.
                var i: usize = 0;
                while (i < sym.len) {
                    const clen = @min(utf8Len(sym[i]), sym.len - i);
                    if (self.token_to_id.get(sym[i .. i + clen])) |id| {
                        try out.append(allocator, id);
                    }
                    i += clen;
                }
            }
        }
    }

    /// Repeatedly merge the adjacent symbol pair with the lowest merge rank.
    fn applyMerges(self: *const Tokenizer, allocator: Allocator, symbols: *std.ArrayList([]u8)) !void {
        var key: std.ArrayList(u8) = .empty;
        defer key.deinit(allocator);

        while (symbols.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_pos: usize = 0;
            for (0..symbols.items.len - 1) |i| {
                key.clearRetainingCapacity();
                try key.appendSlice(allocator, symbols.items[i]);
                try key.append(allocator, ' ');
                try key.appendSlice(allocator, symbols.items[i + 1]);
                if (self.merge_ranks.get(key.items)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_pos = i;
                    }
                }
            }
            if (best_rank == std.math.maxInt(u32)) break;

            const merged = try std.mem.concat(allocator, u8, &.{ symbols.items[best_pos], symbols.items[best_pos + 1] });
            allocator.free(symbols.items[best_pos]);
            allocator.free(symbols.items[best_pos + 1]);
            symbols.items[best_pos] = merged;
            _ = symbols.orderedRemove(best_pos + 1);
        }
    }

    // ---- decode ----

    /// Decode token IDs back to UTF-8 text (allocated with `allocator`).
    pub fn decode(self: *const Tokenizer, allocator: Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (ids) |id| try self.decodeTokenInto(allocator, id, &out);
        return out.toOwnedSlice(allocator);
    }

    /// Append one token's bytes to `out`, reversing the GPT-2 byte-level mapping.
    fn decodeTokenInto(self: *const Tokenizer, allocator: Allocator, id: u32, out: *std.ArrayList(u8)) !void {
        if (id >= self.vocab.len) return;
        const enc = self.vocab[id];
        var i: usize = 0;
        while (i < enc.len) {
            const byte0 = enc[i];
            var cp: u21 = byte0;
            var clen: usize = 1;
            if (byte0 >= 0x80 and byte0 < 0xE0 and i + 1 < enc.len) {
                cp = (@as(u21, byte0 & 0x1F) << 6) | @as(u21, enc[i + 1] & 0x3F);
                clen = 2;
            } else if (byte0 >= 0xE0 and i + 2 < enc.len) {
                cp = (@as(u21, byte0 & 0x0F) << 12) | (@as(u21, enc[i + 1] & 0x3F) << 6) | @as(u21, enc[i + 2] & 0x3F);
                clen = 3;
            } else if (byte0 >= 0x80) {
                i += 1; // malformed; skip
                continue;
            }
            i += clen;
            try out.append(allocator, gpt2UnicodeToByte(cp));
        }
    }
};

/// Streaming decoder for token-by-token generation. A single token can end in
/// the middle of a multi-byte UTF-8 character (the next token completes it), so
/// `push` emits only the complete-UTF-8 prefix and holds the incomplete tail
/// until it can be finished. The sink is any `*std.Io.Writer` (stdout, an SSE
/// response, an in-memory buffer, …) so the same code streams anywhere.
pub const StreamDecoder = struct {
    tokenizer: *const Tokenizer,
    pending: std.ArrayList(u8) = .empty,

    pub fn init(tokenizer: *const Tokenizer) StreamDecoder {
        return .{ .tokenizer = tokenizer };
    }

    pub fn deinit(self: *StreamDecoder, allocator: Allocator) void {
        self.pending.deinit(allocator);
        self.* = undefined;
    }

    /// Clear buffered bytes (reuse across conversation turns).
    pub fn reset(self: *StreamDecoder) void {
        self.pending.clearRetainingCapacity();
    }

    /// Decode `id` and write any now-complete UTF-8 to `writer`.
    pub fn push(self: *StreamDecoder, allocator: Allocator, id: u32, writer: *std.Io.Writer) !void {
        try self.tokenizer.decodeTokenInto(allocator, id, &self.pending);
        const emit = completeUtf8Prefix(self.pending.items);
        if (emit == 0) return;
        try writer.writeAll(self.pending.items[0..emit]);
        const tail = self.pending.items.len - emit;
        if (tail > 0) std.mem.copyForwards(u8, self.pending.items[0..tail], self.pending.items[emit..]);
        self.pending.shrinkRetainingCapacity(tail);
    }

    /// Emit any remaining buffered bytes (call once when generation ends).
    pub fn flush(self: *StreamDecoder, writer: *std.Io.Writer) !void {
        if (self.pending.items.len == 0) return;
        try writer.writeAll(self.pending.items);
        self.pending.clearRetainingCapacity();
    }
};

/// Length of the prefix of `bytes` that ends on a UTF-8 character boundary,
/// i.e. excluding a trailing started-but-incomplete multi-byte sequence.
fn completeUtf8Prefix(bytes: []const u8) usize {
    if (bytes.len == 0) return 0;
    // Walk back over up to 3 continuation bytes to the last lead byte.
    var i: usize = bytes.len;
    var cont: usize = 0;
    while (i > 0 and (bytes[i - 1] & 0xC0) == 0x80 and cont < 3) : (cont += 1) i -= 1;
    if (i == 0) return bytes.len; // only continuation bytes (malformed): emit all
    const need = utf8Len(bytes[i - 1]);
    const have = bytes.len - (i - 1);
    return if (have >= need) bytes.len else i - 1;
}

/// GPT-2 byte→unicode: printable bytes map to themselves; the rest shift into
/// the U+0100+ range so every raw byte is representable in the vocabulary.
/// Writes the UTF-8 of the mapped codepoint into `buf`, returns its length.
fn gpt2ByteToUnicode(byte: u8, buf: *[4]u8) usize {
    const cp: u21 = switch (byte) {
        '!'...'~', 0xA1...0xAC, 0xAE...0xFF => byte,
        else => @as(u21, 256) + @as(u21, switch (byte) {
            0...0x20 => byte,
            0x7F...0xA0 => byte - 0x7F + 33,
            0xAD => 33 + 34,
            else => byte,
        }),
    };
    if (cp < 0x80) {
        buf[0] = @intCast(cp);
        return 1;
    } else if (cp < 0x800) {
        buf[0] = @intCast(0xC0 | (cp >> 6));
        buf[1] = @intCast(0x80 | (cp & 0x3F));
        return 2;
    } else {
        buf[0] = @intCast(0xE0 | (cp >> 12));
        buf[1] = @intCast(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = @intCast(0x80 | (cp & 0x3F));
        return 3;
    }
}

/// Inverse of `gpt2ByteToUnicode`.
fn gpt2UnicodeToByte(cp: u21) u8 {
    if (cp < 256) return @intCast(cp);
    if (cp < 256 + 33) return @intCast(cp - 256); // 0..32
    if (cp < 256 + 33 + 34) return @intCast(cp - 256 - 33 + 0x7F); // 127..160
    if (cp == 256 + 33 + 34) return 0xAD; // 173
    return '?';
}

fn utf8Len(byte0: u8) usize {
    if (byte0 < 0x80) return 1;
    if ((byte0 & 0xE0) == 0xC0) return 2;
    if ((byte0 & 0xF0) == 0xE0) return 3;
    if ((byte0 & 0xF8) == 0xF0) return 4;
    return 1;
}

/// One Qwen2 pretoken chunk: the exclusive end index (in codepoints) of the
/// chunk starting at `start` (always > `start`).
///
/// Faithful port of llama.cpp's `unicode_regex_split_custom_qwen2`
/// (refs/llama.cpp/src/unicode.cpp), the hand-rolled codepoint loop for the
/// Qwen2/StableLM2/Hunyuan pretokenizer regex
/// (LLAMA_VOCAB_PRE_TYPE_QWEN2 in refs/llama.cpp/src/llama-vocab.cpp):
///
///   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
///   |  ?[^\s\p{L}\p{N}]+[\r\n]*  | \s*[\r\n]+ | \s+(?!\S)   | \s+
///
/// The load-bearing details vs the previous ASCII approximation: digits split
/// ONE per chunk (\p{N} has no quantifier), punctuation runs absorb trailing
/// \r\n, a whitespace run before a non-space keeps its LAST space glued to the
/// next chunk (\s+(?!\S)), newline runs split \s*[\r\n]+, any single
/// non-letter/digit/CR/LF (not just space) prefixes a letter run, and \p{L} /
/// \p{N} / \s use real Unicode categories (unicode_categories.zig).
///
/// Known deviation (malformed input only): we classify each undecodable BYTE
/// as its own U+FFFD and keep the original bytes in the chunk, while
/// llama.cpp's unicode_cpts_from_utf8 decodes overlong/surrogate-range forms
/// leniently (one codepoint for the whole sequence) and substitutes U+FFFD's
/// encoding into the output — so on invalid UTF-8 both the chunk BOUNDARIES
/// and the emitted bytes can differ from llama.cpp. Valid UTF-8 input chunks
/// and encodes identically.
// ---------------------------------------------------------------------------
// JoyAI ("joyai-llm", DeepSeek V4 family) pre-tokenizer. Byte-oriented port
// of the reference splitter; the split SHAPE matters because different
// pieces lead to different BPE merges even for identical text bytes:
//
//   \p{N}{1,3} | [CJK/kana]+ | [P/S][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//   |  ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// The punctuation rule keeps trailing newlines in the same BPE word (">;\n"),
// and a whitespace run before a word donates its LAST space to that word
// ("    int" splits "   " + " int").
// ---------------------------------------------------------------------------

fn joyaiPunctSymbol(c: u8) bool {
    return (c >= '!' and c <= '/') or (c >= ':' and c <= '@') or (c >= '[' and c <= '`') or (c >= '{' and c <= '~');
}

fn joyaiSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0b or c == 0x0c;
}

fn joyaiNewline(c: u8) bool {
    return c == '\n' or c == '\r';
}

fn joyaiNextUtf8(text: []const u8, pos: usize) usize {
    const len = std.unicode.utf8ByteSequenceLength(text[pos]) catch 1;
    return @min(pos + len, text.len);
}

fn joyaiCjkAt(text: []const u8, pos: usize) bool {
    if (text[pos] < 128) return false;
    const len = std.unicode.utf8ByteSequenceLength(text[pos]) catch return false;
    if (pos + len > text.len) return false;
    const cp = std.unicode.utf8Decode(text[pos..][0..len]) catch return false;
    return (cp >= 0x4e00 and cp <= 0x9fa5) or (cp >= 0x3040 and cp <= 0x309f) or (cp >= 0x30a0 and cp <= 0x30ff);
}

/// "Letter-like": ASCII alphabetic, or any non-ASCII lead byte (the JoyAI
/// reference collapses Unicode letters this way; CJK/kana are isolated by
/// their own earlier rule).
fn joyaiLetterAt(text: []const u8, pos: usize) bool {
    const c = text[pos];
    if (c < 128) return std.ascii.isAlphabetic(c);
    return true;
}

fn joyaiConsumeLetters(text: []const u8, start: usize) usize {
    var pos = start;
    while (pos < text.len and joyaiLetterAt(text, pos)) pos = joyaiNextUtf8(text, pos);
    return pos;
}

/// One JoyAI pre-token: byte offset just past the chunk starting at `start`.
/// GLM-4/5 ("glm4"/"chatglm-bpe") pre-tokenizer: llama.cpp's CHATGLM4 regex
/// is the Qwen2 regex with ONE difference — digits chunk as runs of up to
/// three (\p{N}{1,3}) instead of one per chunk. Everything else (explicit
/// case-class contractions included) is semantically identical, so this
/// delegates to the qwen2 state machine and post-extends digit chunks.
fn glm4ChunkEnd(c: []const u32, start: usize) usize {
    const end = qwen2ChunkEnd(c, start);
    if (ucat.isNumber(c[start])) {
        var pos = end; // qwen2 digit chunk is exactly one codepoint
        while (pos < c.len and pos - start < 3 and ucat.isNumber(c[pos])) pos += 1;
        return pos;
    }
    return end;
}

test "glm4 pretokenizer: digit runs chunk up to three" {
    // "20488" -> "204" | "88"; qwen2 would give five single-digit chunks.
    const text = "x 20488!";
    const cps = [_]u32{ 'x', ' ', '2', '0', '4', '8', '8', '!' };
    var pos: usize = 0;
    var chunks: [8][2]usize = undefined;
    var n: usize = 0;
    while (pos < cps.len) {
        const end = glm4ChunkEnd(&cps, pos);
        chunks[n] = .{ pos, end };
        n += 1;
        pos = end;
    }
    // "x" | " " | "204" | "88" | "!" — the space is its own \s+ chunk
    // (spaces glue to letter runs only, not digit runs).
    try std.testing.expectEqual(@as(usize, 5), n);
    try std.testing.expectEqual([2]usize{ 2, 5 }, chunks[2]); // "204"
    try std.testing.expectEqual([2]usize{ 5, 7 }, chunks[3]); // "88"
    _ = text;
}

fn joyaiChunkEnd(text: []const u8, start: usize) usize {
    const len = text.len;
    var pos = start;
    const c = text[pos];

    if (std.ascii.isDigit(c)) {
        var ndigits: usize = 0;
        while (pos < len and std.ascii.isDigit(text[pos]) and ndigits < 3) {
            pos += 1;
            ndigits += 1;
        }
    } else if (joyaiCjkAt(text, pos)) {
        while (true) {
            pos = joyaiNextUtf8(text, pos);
            if (pos >= len or !joyaiCjkAt(text, pos)) break;
        }
    } else if (joyaiPunctSymbol(c) and pos + 1 < len and std.ascii.isAlphabetic(text[pos + 1])) {
        pos += 1;
        while (pos < len and std.ascii.isAlphabetic(text[pos])) pos += 1;
    } else if (joyaiLetterAt(text, pos)) {
        pos = joyaiConsumeLetters(text, pos);
    } else if (!joyaiNewline(c) and !joyaiPunctSymbol(c) and pos + 1 < len and joyaiLetterAt(text, pos + 1)) {
        pos += 1;
        pos = joyaiConsumeLetters(text, pos);
    } else if (c == ' ' and pos + 1 < len and joyaiPunctSymbol(text[pos + 1])) {
        pos += 1;
        while (pos < len and joyaiPunctSymbol(text[pos])) pos += 1;
        while (pos < len and joyaiNewline(text[pos])) pos += 1;
    } else if (joyaiPunctSymbol(c)) {
        while (pos < len and joyaiPunctSymbol(text[pos])) pos += 1;
        while (pos < len and joyaiNewline(text[pos])) pos += 1;
    } else if (joyaiSpace(c)) {
        var p = pos;
        var last_newline_end: usize = 0;
        while (p < len and joyaiSpace(text[p])) {
            const sc = text[p];
            p += 1;
            if (joyaiNewline(sc)) last_newline_end = p;
        }
        if (last_newline_end != 0) {
            pos = last_newline_end;
        } else if (p < len and p > pos + 1 and (joyaiLetterAt(text, p) or joyaiPunctSymbol(text[p]))) {
            // A whitespace run donates its last space to the following word
            // or punctuation run.
            pos = p - 1;
        } else {
            pos = p;
        }
    } else {
        pos = joyaiNextUtf8(text, pos);
    }

    if (pos == start) pos = joyaiNextUtf8(text, pos);
    return pos;
}

test "joyai pretokenizer: digit runs split three per chunk, indentation donates a space" {
    // "2048" -> "204" | "8"; "    int x" -> "   " | " int" | " x".
    const cases = [_]struct { text: []const u8, chunks: []const []const u8 }{
        .{ .text = "2048", .chunks = &.{ "204", "8" } },
        .{ .text = "    int x", .chunks = &.{ "   ", " int", " x" } },
        .{ .text = "a>;\nb", .chunks = &.{ "a", ">;\n", "b" } },
        .{ .text = "ciao, mondo!\n\n", .chunks = &.{ "ciao", ",", " mondo", "!\n\n" } },
        .{ .text = "x = f(12345)", .chunks = &.{ "x", " =", " f", "(", "123", "45", ")" } },
    };
    for (cases) |case| {
        var pos: usize = 0;
        for (case.chunks) |expect| {
            const end = joyaiChunkEnd(case.text, pos);
            try std.testing.expectEqualStrings(expect, case.text[pos..end]);
            pos = end;
        }
        try std.testing.expectEqual(case.text.len, pos);
    }
}

fn qwen2ChunkEnd(c: []const u32, start: usize) usize {
    const n = c.len;
    var pos = start;
    const cp = c[pos];

    // (?i:'s|'t|'re|'ve|'m|'ll|'d) — ASCII case-insensitive. llama.cpp's
    // table-driven unicode_tolower agrees: no non-ASCII codepoint lowercases
    // into {s,t,m,d,r,e,v,l} (verified against unicode_map_lowercase).
    if (cp == '\'' and pos + 1 < n) {
        const c1 = asciiLower(c[pos + 1]);
        if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return pos + 2;
        if (pos + 2 < n) {
            const c2 = asciiLower(c[pos + 2]);
            if ((c1 == 'r' and c2 == 'e') or (c1 == 'v' and c2 == 'e') or (c1 == 'l' and c2 == 'l')) return pos + 3;
        }
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+
    if (!(cp == '\r' or cp == '\n' or ucat.isNumber(cp))) {
        if (ucat.isLetter(cp) or (pos + 1 < n and ucat.isLetter(c[pos + 1]))) {
            pos += 1;
            while (pos < n and ucat.isLetter(c[pos])) pos += 1;
            return pos;
        }
    }

    // \p{N} — exactly one digit per chunk.
    if (ucat.isNumber(cp)) return pos + 1;

    //  ?[^\s\p{L}\p{N}]+[\r\n]*
    {
        const j = if (cp == ' ') pos + 1 else pos;
        // j >= n (lookahead past the end) enters the branch like llama.cpp's
        // zeroed out-of-range flags: a trailing lone space emits itself.
        const enter = j >= n or !(ucat.isWhitespace(c[j]) or ucat.isLetter(c[j]) or ucat.isNumber(c[j]));
        if (enter) {
            pos = j;
            while (pos < n and !(ucat.isWhitespace(c[pos]) or ucat.isLetter(c[pos]) or ucat.isNumber(c[pos]))) pos += 1;
            while (pos < n and (c[pos] == '\r' or c[pos] == '\n')) pos += 1;
            return pos;
        }
    }

    // Measure the whitespace run once for the three \s rules.
    var num_ws: usize = 0;
    var last_rn: usize = 0; // index just past the last \r or \n in the run
    while (pos + num_ws < n and ucat.isWhitespace(c[pos + num_ws])) {
        const w = c[pos + num_ws];
        if (w == '\r' or w == '\n') last_rn = pos + num_ws + 1;
        num_ws += 1;
    }

    // \s*[\r\n]+ — through the last newline of the run.
    if (last_rn > 0) return last_rn;

    // \s+(?!\S) — all but the last space when a non-space follows the run.
    if (num_ws > 1 and pos + num_ws < n) return pos + num_ws - 1;

    // \s+
    if (num_ws > 0) return pos + num_ws;

    // Unreachable for in-range codepoints (every category falls in a rule
    // above) — kept as llama.cpp's defensive single-codepoint fallback.
    return pos + 1;
}

/// Qwen3.5/3.6 pretokenizer (`tokenizer.ggml.pre = "qwen35"`): the qwen2
/// rules with combining marks folded into the word class —
///   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
///   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
///   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// vs qwen2: word runs accept \p{M} (and a mark alone matches the word
/// class), and the punctuation run excludes \p{M}. Everything else is
/// byte-for-byte the qwen2 scanner.
fn qwen35ChunkEnd(c: []const u32, start: usize) usize {
    const n = c.len;
    var pos = start;
    const cp = c[pos];

    // Contractions — explicit ASCII case classes, same effect as qwen2's (?i:).
    if (cp == '\'' and pos + 1 < n) {
        const c1 = asciiLower(c[pos + 1]);
        if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return pos + 2;
        if (pos + 2 < n) {
            const c2 = asciiLower(c[pos + 2]);
            if ((c1 == 'r' and c2 == 'e') or (c1 == 'v' and c2 == 'e') or (c1 == 'l' and c2 == 'l')) return pos + 3;
        }
    }

    // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if (!(cp == '\r' or cp == '\n' or ucat.isNumber(cp))) {
        const lm0 = ucat.isLetter(cp) or ucat.isMark(cp);
        const lm1 = pos + 1 < n and (ucat.isLetter(c[pos + 1]) or ucat.isMark(c[pos + 1]));
        if (lm0 or lm1) {
            pos += 1;
            while (pos < n and (ucat.isLetter(c[pos]) or ucat.isMark(c[pos]))) pos += 1;
            return pos;
        }
    }

    // \p{N} — exactly one digit per chunk.
    if (ucat.isNumber(cp)) return pos + 1;

    //  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
    {
        const j = if (cp == ' ') pos + 1 else pos;
        const enter = j >= n or !(ucat.isWhitespace(c[j]) or ucat.isLetter(c[j]) or ucat.isMark(c[j]) or ucat.isNumber(c[j]));
        if (enter) {
            pos = j;
            while (pos < n and !(ucat.isWhitespace(c[pos]) or ucat.isLetter(c[pos]) or ucat.isMark(c[pos]) or ucat.isNumber(c[pos]))) pos += 1;
            while (pos < n and (c[pos] == '\r' or c[pos] == '\n')) pos += 1;
            return pos;
        }
    }

    // Measure the whitespace run once for the three \s rules.
    var num_ws: usize = 0;
    var last_rn: usize = 0;
    while (pos + num_ws < n and ucat.isWhitespace(c[pos + num_ws])) {
        const w = c[pos + num_ws];
        if (w == '\r' or w == '\n') last_rn = pos + num_ws + 1;
        num_ws += 1;
    }

    // \s*[\r\n]+ — through the last newline of the run.
    if (last_rn > 0) return last_rn;

    // \s+(?!\S) — all but the last space when a non-space follows the run.
    if (num_ws > 1 and pos + num_ws < n) return pos + num_ws - 1;

    // \s+
    if (num_ws > 0) return pos + num_ws;

    return pos + 1;
}

fn asciiLower(cp: u32) u32 {
    return if (cp >= 'A' and cp <= 'Z') cp + 32 else cp;
}

/// One Inkling pre-token: end index (exclusive, in codepoints) of the chunk
/// starting at `start`.
///
/// llama.cpp has no custom splitter for LLAMA_VOCAB_PRE_TYPE_INKLING: its
/// regex (o200k with \p{M} added to both word classes so combining marks
/// attach to base letters) runs through the generic collapsed-std::regex path
/// in unicode.cpp, where every codepoint >= 128 is replaced by one category
/// byte. This function reproduces those COLLAPSED semantics, alternative by
/// alternative, in ECMAScript first-match priority order:
///
///   1. [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
///   2. [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
///   3. \p{N}{1,3}
///   4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
///   5. \s*[\r\n]+
///   6. \s+(?!\S)
///   7. \s+
///
/// Collapse quirks kept faithfully: a non-ASCII letter of ANY case is one
/// 0xD2 byte, which is neither ASCII a-z nor A-Z, so it belongs to BOTH word
/// case classes; combining marks (0xD4) likewise. Only ASCII letters are
/// case-restricted. The alternation order matters: alternative 1 is tried
/// (with, then without, the one-char prefix) before alternative 2, and a
/// shorter alt-1 match beats a longer alt-2 match.
fn inklingChunkEnd(c: []const u32, start: usize) usize {
    const n = c.len;
    const cp = c[start];

    const prefix_ok = !(cp == '\r' or cp == '\n' or ucat.isLetter(cp) or ucat.isNumber(cp));
    if (prefix_ok) {
        if (inklingBody1(c, start + 1)) |e| return e;
    }
    if (inklingBody1(c, start)) |e| return e;
    if (prefix_ok) {
        if (inklingBody2(c, start + 1)) |e| return e;
    }
    if (inklingBody2(c, start)) |e| return e;

    // \p{N}{1,3}
    if (ucat.isNumber(cp)) {
        var pos = start + 1;
        while (pos < n and pos - start < 3 and ucat.isNumber(c[pos])) pos += 1;
        return pos;
    }

    //  ?[^\s\p{L}\p{N}]+[\r\n/]* — unlike qwen2 there is no j >= n entry:
    // the + class needs at least one in-range codepoint, so a trailing lone
    // space falls through to the \s+ rule (same chunk, different rule).
    {
        const j = if (cp == ' ') start + 1 else start;
        if (j < n and !(ucat.isWhitespace(c[j]) or ucat.isLetter(c[j]) or ucat.isNumber(c[j]))) {
            var pos = j + 1;
            while (pos < n and !(ucat.isWhitespace(c[pos]) or ucat.isLetter(c[pos]) or ucat.isNumber(c[pos]))) pos += 1;
            while (pos < n and (c[pos] == '\r' or c[pos] == '\n' or c[pos] == '/')) pos += 1;
            return pos;
        }
    }

    // Measure the whitespace run once for the three \s rules.
    var num_ws: usize = 0;
    var last_rn: usize = 0; // index just past the last \r or \n in the run
    while (start + num_ws < n and ucat.isWhitespace(c[start + num_ws])) {
        const w = c[start + num_ws];
        if (w == '\r' or w == '\n') last_rn = start + num_ws + 1;
        num_ws += 1;
    }

    // \s*[\r\n]+ — through the last newline of the run.
    if (last_rn > 0) return last_rn;

    // \s+(?!\S) — all but the last space when a non-space follows the run.
    if (num_ws > 1 and start + num_ws < n) return start + num_ws - 1;

    // \s+
    if (num_ws > 0) return start + num_ws;

    // Unreachable for in-range codepoints — defensive single-codepoint fallback.
    return start + 1;
}

/// ((?=[\p{L}\p{M}])([^a-z])) under the category collapse: letter-or-mark
/// that is not an ASCII lowercase letter ("upper-ish").
fn inklingUpperish(cp: u32) bool {
    return (ucat.isLetter(cp) or ucat.isMark(cp)) and !(cp >= 'a' and cp <= 'z');
}

/// ((?=[\p{L}\p{M}])([^A-Z])): letter-or-mark that is not an ASCII uppercase
/// letter ("lower-ish"). Non-ASCII letters and marks satisfy both classes.
fn inklingLowerish(cp: u32) bool {
    return (ucat.isLetter(cp) or ucat.isMark(cp)) and !(cp >= 'A' and cp <= 'Z');
}

/// Optional (?i:'s|'t|'re|'ve|'m|'ll|'d) suffix (ASCII apostrophe only).
fn inklingContractionEnd(c: []const u32, pos: usize) usize {
    const n = c.len;
    if (pos < n and c[pos] == '\'' and pos + 1 < n) {
        const c1 = asciiLower(c[pos + 1]);
        if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return pos + 2;
        if (pos + 2 < n) {
            const c2 = asciiLower(c[pos + 2]);
            if ((c1 == 'r' and c2 == 'e') or (c1 == 'v' and c2 == 'e') or (c1 == 'l' and c2 == 'l')) return pos + 3;
        }
    }
    return pos;
}

/// upper-ish* lower-ish+ contraction? — ECMAScript-greedy: the star takes its
/// maximal run, then backtracks one codepoint at a time until the plus can
/// start (both-class codepoints make the handoff position matter).
fn inklingBody1(c: []const u32, p: usize) ?usize {
    const n = c.len;
    var m = p;
    while (m < n and inklingUpperish(c[m])) m += 1;
    var k = m + 1;
    while (k > p) {
        k -= 1;
        if (k < n and inklingLowerish(c[k])) {
            var e = k + 1;
            while (e < n and inklingLowerish(c[e])) e += 1;
            return inklingContractionEnd(c, e);
        }
    }
    return null;
}

/// upper-ish+ lower-ish* contraction? — no backtracking needed: the trailing
/// parts are all optional, so maximal-greedy succeeds first.
fn inklingBody2(c: []const u32, p: usize) ?usize {
    const n = c.len;
    if (p >= n or !inklingUpperish(c[p])) return null;
    var m = p + 1;
    while (m < n and inklingUpperish(c[m])) m += 1;
    var e = m;
    while (e < n and inklingLowerish(c[e])) e += 1;
    return inklingContractionEnd(c, e);
}

test "inkling pretokenizer: o200k word/digit/punct/whitespace splits with marks" {
    // Chunks given as [start, end) codepoint index pairs.
    const cases = [_]struct { cps: []const u32, expect: []const [2]usize }{
        // "Hello world" — space glues to the following word.
        .{ .cps = &.{ 'H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd' }, .expect = &.{ .{ 0, 5 }, .{ 5, 11 } } },
        // "HELLO" — alt 2 (upper+ lower*).
        .{ .cps = &.{ 'H', 'E', 'L', 'L', 'O' }, .expect = &.{.{ 0, 5 }} },
        // e + combining acute (U+0301) + "clair": one word chunk (the \p{M} point).
        .{ .cps = &.{ 'e', 0x0301, 'c', 'l', 'a', 'i', 'r' }, .expect = &.{.{ 0, 7 }} },
        // "AÉB" (É = U+00C9, both classes): alt 1 stops after É; "B" restarts.
        .{ .cps = &.{ 'A', 0x00C9, 'B' }, .expect = &.{ .{ 0, 2 }, .{ 2, 3 } } },
        // mark + "AB": alt 1 without prefix matches the bare mark first.
        .{ .cps = &.{ 0x0301, 'A', 'B' }, .expect = &.{ .{ 0, 1 }, .{ 1, 3 } } },
        // "can'T" — case-insensitive contraction suffix.
        .{ .cps = &.{ 'c', 'a', 'n', '\'', 'T' }, .expect = &.{.{ 0, 5 }} },
        // "1234" — digits chunk in threes.
        .{ .cps = &.{ '1', '2', '3', '4' }, .expect = &.{ .{ 0, 3 }, .{ 3, 4 } } },
        // "hi.\n/x" — punctuation run absorbs trailing newlines and slashes.
        .{ .cps = &.{ 'h', 'i', '.', '\n', '/', 'x' }, .expect = &.{ .{ 0, 2 }, .{ 2, 5 }, .{ 5, 6 } } },
        // "  \n  x" — ws-to-last-newline, then trailing-ws donates one space.
        .{ .cps = &.{ ' ', ' ', '\n', ' ', ' ', 'x' }, .expect = &.{ .{ 0, 3 }, .{ 3, 4 }, .{ 4, 6 } } },
        // " !" — punct takes an optional leading space.
        .{ .cps = &.{ ' ', '!', '!' }, .expect = &.{.{ 0, 3 }} },
        // " 5" — the space cannot prefix a digit chunk.
        .{ .cps = &.{ ' ', '5' }, .expect = &.{ .{ 0, 1 }, .{ 1, 2 } } },
    };
    for (cases) |case| {
        var pos: usize = 0;
        var idx: usize = 0;
        while (pos < case.cps.len) {
            const end = inklingChunkEnd(case.cps, pos);
            try std.testing.expect(idx < case.expect.len);
            try std.testing.expectEqual(case.expect[idx][0], pos);
            try std.testing.expectEqual(case.expect[idx][1], end);
            pos = end;
            idx += 1;
        }
        try std.testing.expectEqual(case.expect.len, idx);
    }
}

test {
    _ = ucat; // run the generated category-table tests
    _ = @import("tokenizer_tests.zig");
}

/// Assert the qwen2 pretokenizer splits `text` into exactly `expected` chunks.
fn expectChunks(text: []const u8, expected: []const []const u8) !void {
    const allocator = std.testing.allocator;
    var cps: std.ArrayList(u32) = .empty;
    defer cps.deinit(allocator);
    var offs: std.ArrayList(usize) = .empty;
    defer offs.deinit(allocator);
    var it = (try std.unicode.Utf8View.init(text)).iterator();
    while (it.nextCodepointSlice()) |s| {
        try offs.append(allocator, @intFromPtr(s.ptr) - @intFromPtr(text.ptr));
        try cps.append(allocator, try std.unicode.utf8Decode(s));
    }
    try offs.append(allocator, text.len);

    var pos: usize = 0;
    var idx: usize = 0;
    while (pos < cps.items.len) {
        const end = qwen2ChunkEnd(cps.items, pos);
        try std.testing.expect(end > pos);
        try std.testing.expect(idx < expected.len);
        try std.testing.expectEqualStrings(expected[idx], text[offs.items[pos]..offs.items[end]]);
        idx += 1;
        pos = end;
    }
    try std.testing.expectEqual(expected.len, idx);
}

test "qwen2 pretokenizer: digits split one per chunk (\\p{N} singleton)" {
    try expectChunks("1234567", &.{ "1", "2", "3", "4", "5", "6", "7" });
    try expectChunks(" 12", &.{ " ", "1", "2" });
    try expectChunks("a99b", &.{ "a", "9", "9", "b" });
    try expectChunks("3.14", &.{ "3", ".", "1", "4" });
    // Unicode digits too (arabic-indic).
    try expectChunks("١٢", &.{ "١", "٢" });
}

test "qwen2 pretokenizer: punctuation runs absorb optional space and trailing CR/LF" {
    try expectChunks("foo();\r\n", &.{ "foo", "();\r\n" });
    try expectChunks("a = b", &.{ "a", " =", " b" });
    try expectChunks("x ++;\ny", &.{ "x", " ++;\n", "y" });
    // Lone trailing space emits itself (llama.cpp out-of-range lookahead).
    try expectChunks("ab ", &.{ "ab", " " });
}

test "qwen2 pretokenizer: \\s+(?!\\S) keeps the last space glued to the next chunk" {
    try expectChunks("a  b", &.{ "a", " ", " b" });
    try expectChunks("a    b", &.{ "a", "   ", " b" });
    try expectChunks("ab   ", &.{ "ab", "   " }); // run at end: plain \s+
    try expectChunks("a\t\tb", &.{ "a", "\t", "\tb" });
}

test "qwen2 pretokenizer: newline runs split \\s*[\\r\\n]+ through the last newline" {
    try expectChunks("x\r\n\r\ny", &.{ "x", "\r\n\r\n", "y" });
    try expectChunks("x\n\n  y", &.{ "x", "\n\n", " ", " y" });
    try expectChunks("x  \n y", &.{ "x", "  \n", " y" });
}

test "qwen2 pretokenizer: optional single non-letter prefix on letter runs" {
    try expectChunks("café", &.{"café"});
    try expectChunks("a—b", &.{ "a", "—b" }); // em dash prefixes the letter run
    try expectChunks("don't DON'T it'll", &.{ "don", "'t", " DON", "'T", " it", "'ll" });
    try expectChunks("中文字", &.{"中文字"});
}

/// Assert the qwen35 pretokenizer splits `text` into exactly `expected` chunks.
fn expectChunks35(text: []const u8, expected: []const []const u8) !void {
    const allocator = std.testing.allocator;
    var cps: std.ArrayList(u32) = .empty;
    defer cps.deinit(allocator);
    var offs: std.ArrayList(usize) = .empty;
    defer offs.deinit(allocator);
    var it = (try std.unicode.Utf8View.init(text)).iterator();
    while (it.nextCodepointSlice()) |s| {
        try offs.append(allocator, @intFromPtr(s.ptr) - @intFromPtr(text.ptr));
        try cps.append(allocator, try std.unicode.utf8Decode(s));
    }
    try offs.append(allocator, text.len);

    var pos: usize = 0;
    var idx: usize = 0;
    while (pos < cps.items.len) {
        const end = qwen35ChunkEnd(cps.items, pos);
        try std.testing.expect(end > pos);
        try std.testing.expect(idx < expected.len);
        try std.testing.expectEqualStrings(expected[idx], text[offs.items[pos]..offs.items[end]]);
        idx += 1;
        pos = end;
    }
    try std.testing.expectEqual(expected.len, idx);
    try std.testing.expectEqual(text.len, offs.items[pos]);
}

test "qwen35 pretokenizer: combining marks join letter runs, leave punctuation runs" {
    // b + U+0302 (combining circumflex) + c: one word run under qwen35 —
    // and pin the qwen2 divergence so a chunker regression can't hide it.
    try expectChunks35("b\u{0302}c", &.{"b\u{0302}c"});
    try expectChunks("b\u{0302}c", &.{ "b", "\u{0302}c" });

    // Decomposed é at end of word stays attached.
    try expectChunks35("caffe\u{0301}!", &.{ "caffe\u{0301}", "!" });

    // A mark alone matches the word class (not the punctuation run).
    try expectChunks35("!!\u{0301}", &.{ "!!", "\u{0301}" });
    try expectChunks("!!\u{0301}", &.{"!!\u{0301}"});

    // Devanagari with matras/virama: one word run.
    try expectChunks35("नमस्ते", &.{"नमस्ते"});

    // Everything qwen2-shaped is unchanged.
    try expectChunks35("don't DON'T it'll", &.{ "don", "'t", " DON", "'T", " it", "'ll" });
    try expectChunks35("x ++;\ny", &.{ "x", " ++;\n", "y" });
    try expectChunks35("a  b", &.{ "a", " ", " b" });
    try expectChunks35("3.14", &.{ "3", ".", "1", "4" });
}

test "gpt2 byte/unicode mapping round-trips all 256 bytes" {
    var byte: usize = 0;
    while (byte < 256) : (byte += 1) {
        var buf: [4]u8 = undefined;
        const len = gpt2ByteToUnicode(@intCast(byte), &buf);
        var cp: u21 = buf[0];
        if (len == 2) cp = (@as(u21, buf[0] & 0x1F) << 6) | @as(u21, buf[1] & 0x3F);
        if (len == 3) cp = (@as(u21, buf[0] & 0x0F) << 12) | (@as(u21, buf[1] & 0x3F) << 6) | @as(u21, buf[2] & 0x3F);
        try std.testing.expectEqual(@as(u8, @intCast(byte)), gpt2UnicodeToByte(cp));
    }
}

test "completeUtf8Prefix holds an incomplete trailing multibyte sequence" {
    try std.testing.expectEqual(@as(usize, 3), completeUtf8Prefix("abc"));
    try std.testing.expectEqual(@as(usize, 2), completeUtf8Prefix("ab\xE2")); // lead of 3-byte char
    try std.testing.expectEqual(@as(usize, 2), completeUtf8Prefix("ab\xE2\x82")); // 2 of 3 bytes
    try std.testing.expectEqual(@as(usize, 5), completeUtf8Prefix("ab\xE2\x82\xAC")); // complete "€"
    try std.testing.expectEqual(@as(usize, 0), completeUtf8Prefix("\xC3")); // lone 2-byte lead
    try std.testing.expectEqual(@as(usize, 2), completeUtf8Prefix("\xC3\xA9")); // complete "é"
}
