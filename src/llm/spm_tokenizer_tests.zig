//! Behavioral tests for the SPM (SentencePiece) tokenizer (`spm_tokenizer.zig`):
//! the score-driven merge loop, byte fallback, space-prefix preprocessing,
//! BOS/EOS policy, special-token partitioning, decode round-trips and the
//! incomplete-UTF-8-holding StreamDecoder.

const std = @import("std");
const spm_tokenizer = @import("spm_tokenizer.zig");

const Tokenizer = spm_tokenizer.Tokenizer;
const StreamDecoder = spm_tokenizer.StreamDecoder;
const Options = spm_tokenizer.Options;
const Attr = spm_tokenizer.Attr;

const Allocator = std.mem.Allocator;

const testing = std.testing;

/// SentencePiece word-boundary marker ▁ (U+2581), as UTF-8 (mirror of the
/// module-private constant, for building test vocabularies).
const SPACE_MARK = "\xe2\x96\x81";

/// A tiny SentencePiece-shaped vocab for exercising the merge loop, byte
/// fallback and special-token partitioning without a GGUF. Token ids:
///   0:<unk> 1:<s> 2:</s> 3:▁ 4:a 5:b 6:c 7:ab 8:abc 9:▁a
///   10..13: byte tokens for 'x' 'y' '\n' ' '   14:<turn>
const TinyVocab = struct {
    const tokens = [_][]const u8{
        "<unk>", "<s>",         "</s>", SPACE_MARK,  "a",
        "b",     "c",           "ab",   "abc",       SPACE_MARK ++ "a",
        "<0x78>", "<0x79>",     "<0x0A>", "<0x20>",  "<turn>",
    };
    // higher score = merged earlier; "abc" beats "ab"
    const scores = [_]f32{ 0, 0, 0, -2, -1, -1, -1, -3, -2.5, -4, -5, -5, -5, -5, 0 };
    const attrs = [_]Attr{
        .unknown,      .control, .control, .normal, .normal,
        .normal,       .normal,  .normal,  .normal, .normal,
        .byte,         .byte,    .byte,    .byte,   .control,
    };

    fn make(allocator: Allocator, opts: Options) !Tokenizer {
        return Tokenizer.initFromSlices(allocator, &tokens, &scores, &attrs, opts);
    }
};

test "spm merges to the highest-score token and byte-falls-back" {
    const a = testing.allocator;
    var tok = try TinyVocab.make(a, .{ .add_bos = false, .add_eos = false, .add_space_prefix = false });
    defer tok.deinit();

    {
        // "abc": ab(7) vs abc(8) — abc has higher score, so one token.
        const ids = try tok.encode(a, "abc");
        defer a.free(ids);
        try testing.expectEqualSlices(u32, &.{8}, ids);
    }
    {
        // "abx": "ab"(7) then byte-fallback for 'x' (<0x78>=10).
        const ids = try tok.encode(a, "abx");
        defer a.free(ids);
        try testing.expectEqualSlices(u32, &.{ 7, 10 }, ids);
    }
}

test "spm space-prefix produces the ▁-joined token" {
    const a = testing.allocator;
    var tok = try TinyVocab.make(a, .{ .add_bos = false, .add_eos = false, .add_space_prefix = true });
    defer tok.deinit();

    // With add_space_prefix, "a" becomes "▁a" → token 9, not "a"(4).
    const ids = try tok.encode(a, "a");
    defer a.free(ids);
    try testing.expectEqualSlices(u32, &.{9}, ids);
}

test "spm bos/eos policy is applied" {
    const a = testing.allocator;
    var tok = try TinyVocab.make(a, .{ .bos = 1, .eos = 2, .add_bos = true, .add_eos = true, .add_space_prefix = false });
    defer tok.deinit();

    const ids = try tok.encode(a, "abc");
    defer a.free(ids);
    try testing.expectEqualSlices(u32, &.{ 1, 8, 2 }, ids);

    // encodeRaw skips the bos/eos policy.
    const raw = try tok.encodeRaw(a, "abc");
    defer a.free(raw);
    try testing.expectEqualSlices(u32, &.{8}, raw);
}

test "spm partitions special tokens out of raw text" {
    const a = testing.allocator;
    var tok = try TinyVocab.make(a, .{ .add_bos = false, .add_eos = false, .add_space_prefix = false });
    defer tok.deinit();

    // "<turn>" (id 14, control) must stay one token, with "abc" on each side.
    const ids = try tok.encode(a, "abc<turn>abc");
    defer a.free(ids);
    try testing.expectEqualSlices(u32, &.{ 8, 14, 8 }, ids);
}

test "spm decode round-trips, unescaping ▁ and rendering bytes" {
    const a = testing.allocator;
    var tok = try TinyVocab.make(a, .{ .add_bos = false, .add_eos = false, .add_space_prefix = false });
    defer tok.deinit();

    {
        // ▁a(9) + b(5) → " ab"
        const text = try tok.decode(a, &.{ 9, 5 });
        defer a.free(text);
        try testing.expectEqualStrings(" ab", text);
    }
    {
        // byte token <0x0A>(12) → newline
        const text = try tok.decode(a, &.{ 4, 12 });
        defer a.free(text);
        try testing.expectEqualStrings("a\n", text);
    }
    {
        // control token (id 1) renders as nothing for non-special decode
        const text = try tok.decode(a, &.{ 1, 4 });
        defer a.free(text);
        try testing.expectEqualStrings("a", text);
    }
}

test "spm decode strips the sequence-leading space when add_space_prefix" {
    const a = testing.allocator;
    var tok = try TinyVocab.make(a, .{ .bos = 1, .add_bos = true, .add_space_prefix = true });
    defer tok.deinit();

    // "abc" with space-prefix tokenizes to [bos, ▁(3), abc(8)] — "abc" outscores
    // "▁a", so ▁ stays its own token.
    const ids = try tok.encode(a, "abc");
    defer a.free(ids);
    try testing.expectEqualSlices(u32, &.{ 1, 3, 8 }, ids);

    // With no BOS, the sequence-leading ▁ is stripped on decode → "abc".
    const stripped = try tok.decode(a, ids[1..]);
    defer a.free(stripped);
    try testing.expectEqualStrings("abc", stripped);

    // Faithful to llama.cpp: when a leading BOS is dropped, the following space
    // is preserved (remove_space is cleared), so the full sequence yields " abc".
    const with_bos = try tok.decode(a, ids);
    defer a.free(with_bos);
    try testing.expectEqualStrings(" abc", with_bos);
}

test "spm StreamDecoder holds incomplete UTF-8 across tokens" {
    const a = testing.allocator;
    // Two byte tokens that together form "é" (0xC3 0xA9).
    const tokens = [_][]const u8{ "<unk>", "<0xC3>", "<0xA9>" };
    const scores = [_]f32{ 0, 0, 0 };
    const attrs = [_]Attr{ .unknown, .byte, .byte };
    var tok = try Tokenizer.initFromSlices(a, &tokens, &scores, &attrs, .{ .add_bos = false, .add_space_prefix = false });
    defer tok.deinit();

    var sd = StreamDecoder.init(&tok);
    defer sd.deinit(a);
    var buf: [16]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try sd.push(a, 1, &w); // 0xC3 — incomplete, nothing emitted yet
    try testing.expectEqual(@as(usize, 0), w.buffered().len);
    try sd.push(a, 2, &w); // 0xA9 — completes "é"
    try sd.flush(&w);
    try testing.expectEqualStrings("é", w.buffered());
}
