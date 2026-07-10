//! SentencePiece (SPM / "llama"-model) tokenizer for Gemma-family GGUFs.
//!
//! A faithful port of llama.cpp's `llm_tokenizer_spm`
//! (refs/llama.cpp/src/llama-vocab.cpp): a Unigram model driven by per-token
//! *scores*, not BPE merge ranks. Encoding seeds a max-heap of adjacent symbol
//! pairs keyed by the score of the token they would form, repeatedly merges the
//! highest-scoring pair, then resegments — falling back to `<0xXX>` byte tokens
//! for anything the vocabulary can't cover. Special/control tokens (Gemma's
//! `<start_of_turn>`, `<end_of_turn>`, `<bos>`, …) are split out of the raw text
//! first, exactly as llama.cpp's `tokenizer_st_partition` does, so they map to
//! single ids instead of being broken into pieces.
//!
//! Vocabulary, scores and token types all come from GGUF metadata
//! (`tokenizer.ggml.{tokens,scores,token_type}`); special-token ids and the
//! add_bos / add_eos / add_space_prefix policy default to llama.cpp's SPM
//! defaults and may be overridden by metadata or by the caller. Token bytes are
//! copied into owned storage, so the tokenizer outlives the source `gguf.File`.
//!
//! This is the Gemma counterpart to the byte-level BPE `tokenizer.zig`, which
//! handles Qwen/GPT-2 vocabularies and intentionally refuses scored (SPM)
//! models. The two share the public shape (`initFromGguf`, `encode`,
//! `encodeRaw`, `decode`, `decodeAppend`, `StreamDecoder`, `eosId`/`bosId`/
//! `tokenId`/`vocabSize`) so a runner can pick one per architecture.

const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;

const Allocator = std.mem.Allocator;

/// SentencePiece word-boundary marker ▁ (U+2581), as UTF-8. Spaces in the input
/// are escaped to this before tokenization and unescaped back on decode.
const SPACE_MARK = "\xe2\x96\x81";

pub const Error = error{
    NoTokenizerVocab,
    /// The GGUF declares a tokenizer this module does not implement. This SPM
    /// tokenizer needs per-token scores (`tokenizer.ggml.scores`); a byte-level
    /// BPE model (Qwen/GPT-2) should use `tokenizer.zig` instead.
    UnsupportedTokenizerFormat,
    /// `scores` / `token_type` arrays were present but shorter than the vocab.
    TokenizerArrayTooShort,
    TokenizerTooLarge,
} || Allocator.Error;

/// Per-token attribute, mirroring llama.cpp's `LLAMA_TOKEN_TYPE_*` numbering
/// (the integers stored in `tokenizer.ggml.token_type`). Controls how a token
/// is matched during encode (special tokens are pre-split) and rendered on
/// decode (NORMAL → unescape ▁, BYTE → raw byte, CONTROL/UNKNOWN → suppressed
/// unless explicitly requested).
pub const Attr = enum(i32) {
    undef = 0,
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,
    _,

    fn fromInt(v: i32) Attr {
        return switch (v) {
            0, 1, 2, 3, 4, 5, 6 => @enumFromInt(v),
            else => .undef,
        };
    }

    /// Tokens partitioned out of raw text before the SPM merge loop runs.
    fn isSpecial(self: Attr) bool {
        return self == .control or self == .user_defined or self == .unknown;
    }
};

/// Special-token / policy configuration. Defaults come from GGUF metadata (or
/// llama.cpp's SPM defaults: bos=1, eos=2, unk=0, add_bos=true, add_eos=false,
/// add_space_prefix=true); a non-null field forces a value.
pub const Options = struct {
    bos: ?u32 = null,
    eos: ?u32 = null,
    unk: ?u32 = null,
    add_bos: ?bool = null,
    add_eos: ?bool = null,
    add_space_prefix: ?bool = null,
};

pub const Tokenizer = struct {
    allocator: Allocator,
    /// One owned buffer holding every token's bytes back-to-back.
    vocab_blob: []u8,
    /// token id → bytes (slices into `vocab_blob`).
    vocab: [][]const u8,
    /// token bytes → token id (keys slice into `vocab_blob`).
    token_to_id: std.StringHashMap(u32),
    /// Per-token Unigram score (merge priority). Indexed by token id.
    scores: []f32,
    /// Per-token attribute. Indexed by token id.
    attrs: []Attr,
    /// Control / user-defined / unknown token ids, sorted by token-text length
    /// descending — the scan order for special-token partitioning so a longer
    /// marker wins over a prefix of it.
    special_ids: []u32,

    bos: ?u32,
    eos: ?u32,
    unk: ?u32,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,

    /// Build a tokenizer from GGUF metadata. `overrides` fields, when non-null,
    /// replace the metadata-derived values. Returns `UnsupportedTokenizerFormat`
    /// for a model without per-token scores (i.e. a byte-level BPE vocab).
    pub fn initFromGguf(allocator: Allocator, file: *const gguf.File, overrides: Options) !Tokenizer {
        const tokens_arr = file.getArray("tokenizer.ggml.tokens") orelse return Error.NoTokenizerVocab;
        if (tokens_arr.item_type != 8) return Error.NoTokenizerVocab;

        // SPM is identified by the presence of per-token scores. A scored
        // "llama"-model vocab is SentencePiece; without scores this is the wrong
        // tokenizer (use the byte-level BPE one).
        const scores_arr = file.getArray("tokenizer.ggml.scores") orelse return Error.UnsupportedTokenizerFormat;
        if (scores_arr.item_type != 6 or scores_arr.len < tokens_arr.len) return Error.TokenizerArrayTooShort;

        const token_strings = try tokens_arr.stringSlices(allocator);
        defer allocator.free(token_strings);

        // token_type is optional in the format; without it every token is
        // NORMAL (no special-token partitioning, no byte rendering).
        const types_arr = file.getArray("tokenizer.ggml.token_type");
        if (types_arr) |t| {
            // INT32 (5) is the spec type; tolerate UINT32 (4) too.
            if ((t.item_type != 5 and t.item_type != 4) or t.len < tokens_arr.len) return Error.TokenizerArrayTooShort;
        }

        var opts = Options{
            .bos = if (file.getInt("tokenizer.ggml.bos_token_id")) |v| @intCast(v) else 1,
            .eos = if (file.getInt("tokenizer.ggml.eos_token_id")) |v| @intCast(v) else 2,
            .unk = if (file.getInt("tokenizer.ggml.unknown_token_id")) |v| @intCast(v) else 0,
            .add_bos = file.getBool("tokenizer.ggml.add_bos_token") orelse true,
            .add_eos = file.getBool("tokenizer.ggml.add_eos_token") orelse false,
            .add_space_prefix = file.getBool("tokenizer.ggml.add_space_prefix") orelse true,
        };
        if (overrides.bos) |v| opts.bos = v;
        if (overrides.eos) |v| opts.eos = v;
        if (overrides.unk) |v| opts.unk = v;
        if (overrides.add_bos) |v| opts.add_bos = v;
        if (overrides.add_eos) |v| opts.add_eos = v;
        if (overrides.add_space_prefix) |v| opts.add_space_prefix = v;

        return initFromParts(allocator, token_strings, .{
            .scores_raw = scores_arr.data,
            .types_raw = if (types_arr) |t| t.data else null,
            .opts = opts,
        });
    }

    const RawParts = struct {
        /// Little-endian f32 bytes, one per token (length ≥ vocab).
        scores_raw: []const u8,
        /// Little-endian i32 bytes, one per token (length ≥ vocab), or null.
        types_raw: ?[]const u8,
        opts: Options,
    };

    /// Accessor over little-endian f32 GGUF score bytes.
    const RawScores = struct {
        raw: []const u8,
        fn at(self: @This(), i: usize) f32 {
            return @bitCast(std.mem.readInt(u32, self.raw[i * 4 ..][0..4], .little));
        }
    };
    /// Accessor over little-endian i32 GGUF token-type bytes.
    const RawTypes = struct {
        raw: []const u8,
        fn at(self: @This(), i: usize) Attr {
            return Attr.fromInt(std.mem.readInt(i32, self.raw[i * 4 ..][0..4], .little));
        }
    };
    /// Accessor used when token_type is absent: every token is NORMAL.
    const NoTypes = struct {
        fn at(_: @This(), _: usize) Attr {
            return .normal;
        }
    };

    /// Lower-level constructor used by `initFromGguf` and by tests. Scores/types
    /// are decoded lazily from the raw GGUF bytes in `parts`; `opts` fields are
    /// already resolved. Each token's bytes are copied into owned storage.
    fn initFromParts(allocator: Allocator, vocab_strings: []const []const u8, parts: RawParts) !Tokenizer {
        const scores = RawScores{ .raw = parts.scores_raw };
        if (parts.types_raw) |tr| {
            return initCore(allocator, vocab_strings, scores, RawTypes{ .raw = tr }, parts.opts);
        }
        return initCore(allocator, vocab_strings, scores, NoTypes{}, parts.opts);
    }

    /// Test-friendly constructor taking already-decoded scores/attrs.
    pub fn initFromSlices(
        allocator: Allocator,
        vocab_strings: []const []const u8,
        scores: []const f32,
        attrs: ?[]const Attr,
        opts: Options,
    ) !Tokenizer {
        const ScoreSlice = struct {
            s: []const f32,
            fn at(self: @This(), i: usize) f32 {
                return self.s[i];
            }
        };
        const TypeSlice = struct {
            a: []const Attr,
            fn at(self: @This(), i: usize) Attr {
                return self.a[i];
            }
        };
        const score_src = ScoreSlice{ .s = scores };
        if (attrs) |a| {
            return initCore(allocator, vocab_strings, score_src, TypeSlice{ .a = a }, opts);
        }
        return initCore(allocator, vocab_strings, score_src, NoTypes{}, opts);
    }

    /// Shared build path. `score_src`/`type_src` are tiny accessor structs with
    /// an `at(i)` method, so raw-GGUF and pre-decoded callers share one body
    /// without allocating an intermediate decoded array.
    fn initCore(
        allocator: Allocator,
        vocab_strings: []const []const u8,
        score_src: anytype,
        type_src: anytype,
        opts: Options,
    ) !Tokenizer {
        const n = vocab_strings.len;
        if (n > std.math.maxInt(u32)) return Error.TokenizerTooLarge;

        var vocab_bytes: usize = 0;
        for (vocab_strings) |s| vocab_bytes = std.math.add(usize, vocab_bytes, s.len) catch return Error.TokenizerTooLarge;

        const vocab_blob = try allocator.alloc(u8, vocab_bytes);
        errdefer allocator.free(vocab_blob);
        const vocab = try allocator.alloc([]const u8, n);
        errdefer allocator.free(vocab);
        const scores = try allocator.alloc(f32, n);
        errdefer allocator.free(scores);
        const attrs = try allocator.alloc(Attr, n);
        errdefer allocator.free(attrs);

        var token_to_id = std.StringHashMap(u32).init(allocator);
        errdefer token_to_id.deinit();
        try token_to_id.ensureTotalCapacity(@intCast(n));

        var off: usize = 0;
        var n_special: usize = 0;
        for (vocab_strings, 0..) |s, i| {
            @memcpy(vocab_blob[off..][0..s.len], s);
            vocab[i] = vocab_blob[off..][0..s.len];
            off += s.len;
            scores[i] = score_src.at(i);
            const attr: Attr = type_src.at(i);
            attrs[i] = attr;
            if (attr.isSpecial() and s.len > 0) n_special += 1;
            // Match llama.cpp: later (higher) id wins on duplicate bytes.
            token_to_id.putAssumeCapacity(vocab[i], @intCast(i));
        }

        const special_ids = try allocator.alloc(u32, n_special);
        errdefer allocator.free(special_ids);
        {
            var k: usize = 0;
            for (attrs, 0..) |attr, i| {
                if (attr.isSpecial() and vocab[i].len > 0) {
                    special_ids[k] = @intCast(i);
                    k += 1;
                }
            }
            // Longest marker first, so "<start_of_turn>" wins over any prefix.
            std.mem.sort(u32, special_ids, vocab, struct {
                fn longerFirst(v: [][]const u8, a: u32, b: u32) bool {
                    return v[a].len > v[b].len;
                }
            }.longerFirst);
        }

        return .{
            .allocator = allocator,
            .vocab_blob = vocab_blob,
            .vocab = vocab,
            .token_to_id = token_to_id,
            .scores = scores,
            .attrs = attrs,
            .special_ids = special_ids,
            .bos = opts.bos,
            .eos = opts.eos,
            .unk = opts.unk,
            .add_bos = opts.add_bos orelse true,
            .add_eos = opts.add_eos orelse false,
            .add_space_prefix = opts.add_space_prefix orelse true,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.token_to_id.deinit();
        self.allocator.free(self.special_ids);
        self.allocator.free(self.attrs);
        self.allocator.free(self.scores);
        self.allocator.free(self.vocab);
        self.allocator.free(self.vocab_blob);
        self.* = undefined;
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.vocab.len;
    }
    pub fn eosId(self: *const Tokenizer) ?u32 {
        return self.eos;
    }
    pub fn bosId(self: *const Tokenizer) ?u32 {
        return self.bos;
    }
    pub fn isEos(self: *const Tokenizer, id: u32) bool {
        return self.eos != null and id == self.eos.?;
    }

    /// Look up a token id by its exact bytes (e.g. "<end_of_turn>").
    pub fn tokenId(self: *const Tokenizer, token: []const u8) ?u32 {
        return self.token_to_id.get(token);
    }

    // ---- encode ----

    /// Encode `text` into token ids (allocated with `allocator`), applying the
    /// model's BOS/EOS policy and splitting out special tokens. Caller frees.
    pub fn encode(self: *const Tokenizer, allocator: Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);
        try self.tokenizeInto(allocator, text, &out, true, true);
        return out.toOwnedSlice(allocator);
    }

    /// Encode without the BOS/EOS policy (the caller controls structural
    /// tokens) but still split out special tokens present in the text.
    pub fn encodeRaw(self: *const Tokenizer, allocator: Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);
        try self.tokenizeInto(allocator, text, &out, true, false);
        return out.toOwnedSlice(allocator);
    }

    /// Core encode: partition special tokens, then run the SPM session over each
    /// raw fragment with the SentencePiece space-prefix / whitespace-escape
    /// preprocessing. Mirrors `llama_vocab::impl::tokenize` for `SPM`.
    fn tokenizeInto(
        self: *const Tokenizer,
        allocator: Allocator,
        text: []const u8,
        out: *std.ArrayList(u32),
        parse_special: bool,
        add_special: bool,
    ) !void {
        var frags: std.ArrayList(Fragment) = .empty;
        defer frags.deinit(allocator);
        try self.partition(allocator, text, parse_special, &frags);

        // prefix with a space if the previous emitted piece was special
        var is_prev_special = true;
        if (add_special and self.add_bos) {
            if (self.bos) |b| try out.append(allocator, b);
        }

        var escaped: std.ArrayList(u8) = .empty;
        defer escaped.deinit(allocator);
        var session = Session.init(self, allocator);
        defer session.deinit();

        for (frags.items) |frag| {
            switch (frag) {
                .tok => |id| {
                    try out.append(allocator, id);
                    is_prev_special = true;
                },
                .raw => |r| {
                    escaped.clearRetainingCapacity();
                    if (self.add_space_prefix and is_prev_special) try escaped.appendSlice(allocator, SPACE_MARK);
                    try appendEscaped(allocator, &escaped, r);
                    try session.run(escaped.items, out, allocator);
                    is_prev_special = false;
                },
            }
        }

        if (add_special and self.add_eos) {
            if (self.eos) |e| try out.append(allocator, e);
        }
    }

    /// A piece of input mid-partition: still-raw text, or a resolved special id.
    const Fragment = union(enum) {
        raw: []const u8,
        tok: u32,
    };

    /// Split `text` on every special token (longest first), as
    /// `tokenizer_st_partition` does. With `parse_special` false, control and
    /// unknown tokens are left in the raw text (only user-defined tokens split).
    fn partition(
        self: *const Tokenizer,
        allocator: Allocator,
        text: []const u8,
        parse_special: bool,
        out_frags: *std.ArrayList(Fragment),
    ) !void {
        if (text.len == 0) return;
        try out_frags.append(allocator, .{ .raw = text });

        var scratch: std.ArrayList(Fragment) = .empty;
        defer scratch.deinit(allocator);

        for (self.special_ids) |sid| {
            const stext = self.vocab[sid];
            if (stext.len == 0) continue;
            if (!parse_special and (self.attrs[sid] == .control or self.attrs[sid] == .unknown)) continue;

            scratch.clearRetainingCapacity();
            for (out_frags.items) |frag| {
                switch (frag) {
                    .tok => try scratch.append(allocator, frag),
                    .raw => |r0| {
                        var r = r0;
                        while (std.mem.indexOf(u8, r, stext)) |m| {
                            if (m > 0) try scratch.append(allocator, .{ .raw = r[0..m] });
                            try scratch.append(allocator, .{ .tok = sid });
                            r = r[m + stext.len ..];
                        }
                        if (r.len > 0) try scratch.append(allocator, .{ .raw = r });
                    },
                }
            }
            std.mem.swap(std.ArrayList(Fragment), out_frags, &scratch);
        }
    }

    /// Append `r` to `dst`, escaping each space to ▁ (other bytes verbatim).
    fn appendEscaped(allocator: Allocator, dst: *std.ArrayList(u8), r: []const u8) !void {
        for (r) |c| {
            if (c == ' ') {
                try dst.appendSlice(allocator, SPACE_MARK);
            } else {
                try dst.append(allocator, c);
            }
        }
    }

    /// Byte → its byte-fallback token id. SPM byte tokens are "<0xXX>"
    /// (uppercase hex); fall back to a single-byte token, then to `unk`/`ch`.
    fn byteToToken(self: *const Tokenizer, ch: u8) u32 {
        var buf: [6]u8 = undefined;
        const hex = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{ch}) catch unreachable;
        if (self.token_to_id.get(hex)) |id| return id;
        const one = [_]u8{ch};
        if (self.token_to_id.get(&one)) |id| return id;
        return self.unk orelse @as(u32, ch);
    }

    // ---- decode ----

    /// Decode token ids back to UTF-8 (allocated with `allocator`). Reverses the
    /// SentencePiece preprocessing: ▁ → space, byte tokens → their raw byte, the
    /// leading space (and a leading BOS) introduced by encode are removed.
    /// Control / unknown tokens render as nothing.
    pub fn decode(self: *const Tokenizer, allocator: Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);

        var toks = ids;
        var remove_space = self.add_space_prefix;
        if (self.add_bos and toks.len > 0 and self.bos != null and toks[0] == self.bos.?) {
            remove_space = false;
            toks = toks[1..];
        }
        if (self.add_eos and toks.len > 0 and self.eos != null and toks[toks.len - 1] == self.eos.?) {
            toks = toks[0 .. toks.len - 1];
        }

        for (toks) |id| {
            try self.pieceInto(allocator, id, &out, remove_space, false);
            remove_space = false;
        }
        return out.toOwnedSlice(allocator);
    }

    /// Append one token's decoded bytes to `out` (no leading-space strip),
    /// suppressing control/unknown tokens. Used by `StreamDecoder` for
    /// token-by-token generation. Public so a streaming decoder can accumulate.
    pub fn decodeAppend(self: *const Tokenizer, allocator: Allocator, id: u32, out: *std.ArrayList(u8)) !void {
        return self.pieceInto(allocator, id, out, false, false);
    }

    /// Render one token into `out`. `strip_leading` drops a single leading space
    /// (the SentencePiece sequence-start space); `special` keeps otherwise-
    /// suppressed control/unknown tokens. Mirrors `token_to_piece` for SPM.
    fn pieceInto(
        self: *const Tokenizer,
        allocator: Allocator,
        id: u32,
        out: *std.ArrayList(u8),
        strip_leading: bool,
        special: bool,
    ) !void {
        if (id >= self.vocab.len) return;
        const attr = self.attrs[id];
        if (!special and (attr == .unknown or attr == .control)) return;
        const text = self.vocab[id];

        switch (attr) {
            .control, .user_defined, .unknown => try appendRaw(allocator, out, text, strip_leading),
            .normal => try appendUnescaped(allocator, out, text, strip_leading),
            .byte => try out.append(allocator, self.tokenToByte(id)),
            .unused, .undef, _ => {}, // suppressed, as in llama.cpp
        }
    }

    /// "<0xXX>" → its byte (or a 1-byte token's byte). Defensive 0 otherwise.
    fn tokenToByte(self: *const Tokenizer, id: u32) u8 {
        const t = self.vocab[id];
        if (t.len == 6 and t[0] == '<' and t[1] == '0' and t[2] == 'x' and t[5] == '>') {
            const hi = std.fmt.charToDigit(t[3], 16) catch return 0;
            const lo = std.fmt.charToDigit(t[4], 16) catch return 0;
            return @intCast(hi * 16 + lo);
        }
        if (t.len == 1) return t[0];
        return 0;
    }

    fn appendRaw(allocator: Allocator, out: *std.ArrayList(u8), text: []const u8, strip_leading: bool) !void {
        var t = text;
        if (strip_leading and t.len > 0 and t[0] == ' ') t = t[1..];
        try out.appendSlice(allocator, t);
    }

    /// Append `text` with ▁ replaced by spaces; optionally drop one leading
    /// space afterwards.
    fn appendUnescaped(allocator: Allocator, out: *std.ArrayList(u8), text: []const u8, strip_leading: bool) !void {
        const start = out.items.len;
        var i: usize = 0;
        while (i < text.len) {
            if (i + 3 <= text.len and std.mem.eql(u8, text[i .. i + 3], SPACE_MARK)) {
                try out.append(allocator, ' ');
                i += 3;
            } else {
                try out.append(allocator, text[i]);
                i += 1;
            }
        }
        if (strip_leading and out.items.len > start and out.items[start] == ' ') {
            _ = out.orderedRemove(start);
        }
    }
};

/// One SPM tokenization session: the symbol list, the reverse-merge map and the
/// score-ordered work queue. Reused across the raw fragments of one encode call
/// (`run` resets its buffers each time). A direct port of
/// `llm_tokenizer_spm_session`.
const Session = struct {
    tok: *const Tokenizer,
    alloc: Allocator,
    /// `text` of the fragment currently being tokenized (the escaped bytes).
    escaped: []const u8 = &.{},
    symbols: std.ArrayList(Symbol) = .empty,
    /// merged-text → the (left,right) symbol indices it came from (for resegment).
    rev_merge: std.StringHashMap(Pair),
    queue: WorkQueue = .empty,

    const Symbol = struct {
        /// Byte offset into `escaped`, and current length (grows as it absorbs
        /// the symbol to its right — they stay contiguous in `escaped`).
        start: usize,
        n: usize,
        prev: i32,
        next: i32,
    };
    const Pair = struct { left: i32, right: i32 };
    const Bigram = struct { left: i32, right: i32, score: f32, size: usize };

    const WorkQueue = std.PriorityQueue(Bigram, void, bigramOrder);

    /// Max-heap by score, ties broken toward the smaller left index — exactly
    /// llama.cpp's `llm_bigram_spm::comparator`. PriorityQueue pops the `.lt`
    /// element first, so "better" must compare `.lt`.
    fn bigramOrder(_: void, a: Bigram, b: Bigram) std.math.Order {
        if (a.score > b.score) return .lt;
        if (a.score < b.score) return .gt;
        if (a.left < b.left) return .lt;
        if (a.left > b.left) return .gt;
        return .eq;
    }

    fn init(tok: *const Tokenizer, alloc: Allocator) Session {
        return .{ .tok = tok, .alloc = alloc, .rev_merge = std.StringHashMap(Pair).init(alloc) };
    }

    fn deinit(self: *Session) void {
        self.symbols.deinit(self.alloc);
        self.rev_merge.deinit();
        self.queue.deinit(self.alloc);
        self.* = undefined;
    }

    /// Tokenize one escaped fragment, appending ids to `out`.
    fn run(self: *Session, escaped: []const u8, out: *std.ArrayList(u32), alloc: Allocator) !void {
        self.escaped = escaped;
        self.symbols.clearRetainingCapacity();
        self.rev_merge.clearRetainingCapacity();
        while (self.queue.pop()) |_| {}

        // split into UTF-8 characters, linked in text order
        var index: i32 = 0;
        var offs: usize = 0;
        while (offs < escaped.len) {
            const len = @min(utf8Len(escaped[offs]), escaped.len - offs);
            const next: i32 = if (offs + len == escaped.len) -1 else index + 1;
            try self.symbols.append(self.alloc, .{ .start = offs, .n = len, .prev = index - 1, .next = next });
            offs += len;
            index += 1;
        }
        if (self.symbols.items.len == 0) return;

        // seed the queue with every adjacent pair that forms a known token
        var i: usize = 1;
        while (i < self.symbols.items.len) : (i += 1) {
            try self.tryAddBigram(@intCast(i - 1), @intCast(i));
        }

        // repeatedly merge the highest-scoring still-valid pair
        while (self.queue.pop()) |bigram| {
            const left = &self.symbols.items[@intCast(bigram.left)];
            const right = &self.symbols.items[@intCast(bigram.right)];
            // skip if either side was already merged away or shape changed
            if (left.n == 0 or right.n == 0 or left.n + right.n != bigram.size) continue;

            left.n += right.n;
            right.n = 0;
            left.next = right.next;
            if (right.next >= 0) self.symbols.items[@intCast(right.next)].prev = bigram.left;

            try self.tryAddBigram(left.prev, bigram.left);
            try self.tryAddBigram(bigram.left, left.next);
        }

        // emit: walk the surviving linked list from the head (index 0, never
        // absorbed since the head's prev is -1)
        var s: i32 = 0;
        while (s != -1) {
            try self.resegment(@intCast(s), out, alloc);
            s = self.symbols.items[@intCast(s)].next;
        }
    }

    /// If `symbols[left]+symbols[right]` is a vocab token, push it as a candidate
    /// merge keyed by its score and record the reverse mapping.
    fn tryAddBigram(self: *Session, left: i32, right: i32) !void {
        if (left < 0 or right < 0) return;
        const l = self.symbols.items[@intCast(left)];
        const r = self.symbols.items[@intCast(right)];
        const total = l.n + r.n;
        const combined = self.escaped[l.start .. l.start + total];
        const id = self.tok.token_to_id.get(combined) orelse return;
        if (id >= self.tok.vocab.len) return;
        try self.queue.push(self.alloc, .{
            .left = left,
            .right = right,
            .score = self.tok.scores[id],
            .size = total,
        });
        try self.rev_merge.put(combined, .{ .left = left, .right = right });
    }

    /// Emit `symbols[i]`: as a token if its text is in the vocab, else split via
    /// `rev_merge`, else as byte-fallback tokens.
    fn resegment(self: *Session, i: usize, out: *std.ArrayList(u32), alloc: Allocator) !void {
        const sym = self.symbols.items[i];
        const text = self.escaped[sym.start .. sym.start + sym.n];
        if (self.tok.token_to_id.get(text)) |id| {
            try out.append(alloc, id);
            return;
        }
        if (self.rev_merge.get(text)) |pair| {
            // Guard against a self-referential pair (degenerate vocab) before
            // recursing — the recursion is otherwise only reached for "unused"
            // tokens, which Gemma does not have.
            const li: usize = @intCast(pair.left);
            const ri: usize = @intCast(pair.right);
            if (li != i and ri != i) {
                try self.resegment(li, out, alloc);
                try self.resegment(ri, out, alloc);
                return;
            }
        }
        var j = sym.start;
        while (j < sym.start + sym.n) : (j += 1) {
            try out.append(alloc, self.tok.byteToToken(self.escaped[j]));
        }
    }
};

/// Streaming decoder for token-by-token generation. A token can end mid-UTF-8
/// (the next token completes it), so `push` emits only the complete-UTF-8
/// prefix and holds the rest. The sink is any `*std.Io.Writer`.
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

    pub fn reset(self: *StreamDecoder) void {
        self.pending.clearRetainingCapacity();
    }

    pub fn push(self: *StreamDecoder, allocator: Allocator, id: u32, writer: *std.Io.Writer) !void {
        try self.tokenizer.decodeAppend(allocator, id, &self.pending);
        const emit = completeUtf8Prefix(self.pending.items);
        if (emit == 0) return;
        try writer.writeAll(self.pending.items[0..emit]);
        const tail = self.pending.items.len - emit;
        if (tail > 0) std.mem.copyForwards(u8, self.pending.items[0..tail], self.pending.items[emit..]);
        self.pending.shrinkRetainingCapacity(tail);
    }

    pub fn flush(self: *StreamDecoder, writer: *std.Io.Writer) !void {
        if (self.pending.items.len == 0) return;
        try writer.writeAll(self.pending.items);
        self.pending.clearRetainingCapacity();
    }
};

/// Length of the prefix of `bytes` ending on a UTF-8 boundary (excludes a
/// trailing started-but-incomplete multi-byte sequence).
fn completeUtf8Prefix(bytes: []const u8) usize {
    if (bytes.len == 0) return 0;
    var i: usize = bytes.len;
    var cont: usize = 0;
    while (i > 0 and (bytes[i - 1] & 0xC0) == 0x80 and cont < 3) : (cont += 1) i -= 1;
    if (i == 0) return bytes.len;
    const need = utf8Len(bytes[i - 1]);
    const have = bytes.len - (i - 1);
    return if (have >= need) bytes.len else i - 1;
}

fn utf8Len(byte0: u8) usize {
    if (byte0 < 0x80) return 1;
    if ((byte0 & 0xE0) == 0xC0) return 2;
    if ((byte0 & 0xF0) == 0xE0) return 3;
    if ((byte0 & 0xF8) == 0xF0) return 4;
    return 1;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

test {
    _ = @import("spm_tokenizer_tests.zig");
}
