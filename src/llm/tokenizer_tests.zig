//! Behavioral tests for the byte-level BPE tokenizer (`tokenizer.zig`):
//! encode/decode round-trips on a tiny vocab, bare "<|" handling, embedded
//! llama-tokenize parity fixtures (skips without models/), and the streaming
//! decoder.
const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;
const tokenizer = @import("tokenizer.zig");

const Tokenizer = tokenizer.Tokenizer;
const StreamDecoder = tokenizer.StreamDecoder;

test "byte-level BPE encode/decode round-trip on a tiny vocab" {
    const allocator = std.testing.allocator;
    // ASCII letters map to themselves under the byte-level encoding, so this
    // vocab/merge set exercises the merge loop without space remapping.
    const vocab = [_][]const u8{ "a", "b", "c", "ab", "abc" };
    const merges = [_][]const u8{ "a b", "ab c" };
    var tok = try Tokenizer.initFromParts(allocator, &vocab, &merges, .{ .eos = 2 });
    defer tok.deinit();

    {
        const ids = try tok.encode(allocator, "abc");
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(u32, &.{4}, ids); // merged to "abc"
        const text = try tok.decode(allocator, ids);
        defer allocator.free(text);
        try std.testing.expectEqualStrings("abc", text);
    }
    {
        const ids = try tok.encode(allocator, "aab");
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(u32, &.{ 0, 3 }, ids); // "a" + "ab"
    }
    {
        const ids = try tok.encode(allocator, "abcabc");
        defer allocator.free(ids);
        const text = try tok.decode(allocator, ids);
        defer allocator.free(text);
        try std.testing.expectEqualStrings("abcabc", text);
    }
    try std.testing.expectEqual(@as(?u32, 2), tok.eosId());
}

test "bare <| does not force a pretokenization split" {
    const allocator = std.testing.allocator;
    // Vocab where the whole punctuation run "<|||" merges to one token; a
    // forced split after "<|" (the old fallback) would instead emit
    // "<","|","||" — chunk-boundary (and token-ID) divergence.
    const vocab = [_][]const u8{ "<", "|", "||", "|||", "<|||" };
    const merges = [_][]const u8{ "| |", "|| |", "< |||" };
    var tok = try Tokenizer.initFromParts(allocator, &vocab, &merges, .{});
    defer tok.deinit();

    const ids = try tok.encodeRaw(allocator, "<|||");
    defer allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{4}, ids); // single "<|||" token
}

// ---------------------------------------------------------------------------
// llama.cpp parity (guarded: skips without models/).
//
// Expected ids were generated with llama.cpp's tokenizer as the oracle: each
// fixture string was written byte-exact to a file F, then
//
//   refs/llama.cpp/build-cpu/bin/llama-tokenize \
//       -m models/Qwen3-0.6B-Q4_K_S.gguf -f F --ids --no-escape
//
// (--no-escape matters: by default llama-tokenize rewrites "\n" escape
// SEQUENCES in the input to control characters before tokenizing.)
//
// (add_bos is false for Qwen3, so the output is the raw encoding). Bench-file
// parity (code/prose/torture/zig/c, token-ID-exact) was additionally verified
// via `fucina-zig-qwen3 --tokenize FILE` against the same oracle.
// ---------------------------------------------------------------------------

const ParityFixture = struct { text: []const u8, ids: []const u32 };

const parity_fixtures = [_]ParityFixture{
    // Torture: digit runs, multi-space runs, tabs, CRLF, accents, CJK,
    // arabic-indic digits, superscripts, emoji, contractions, mixed code.
    .{
        .text = "Counting: 1234567 and 99 bottles.\nx = 3.14159;  // pi\n\tindent\ttabs\t\twide\r\nCRLF\r\n\r\nspaces:      end   \naccentué léttres 中文字 ١٢٣ ²³\nemoji 😀👍 ok\nit's we'RE don'T I'll y've o'clock\na[i+1] = {\"k\": v, **kw};   #comment!\n   \n  mixed   runs 12x34\n",
        .ids = &.{ 2507, 287, 25, 220, 16, 17, 18, 19, 20, 21, 22, 323, 220, 24, 24, 26376, 624, 87, 284, 220, 18, 13, 16, 19, 16, 20, 24, 26, 220, 442, 8938, 198, 197, 32840, 3244, 3435, 197, 6692, 577, 319, 34, 80658, 871, 44285, 25, 414, 835, 5872, 77548, 95459, 62728, 5566, 416, 72858, 87335, 220, 149, 94, 149, 95, 149, 96, 220, 29456, 43201, 198, 37523, 90316, 144349, 5394, 198, 275, 594, 582, 94153, 1513, 17323, 358, 3278, 379, 3003, 297, 62410, 198, 64, 989, 10, 16, 60, 284, 5212, 74, 788, 348, 11, 3070, 28600, 11061, 256, 671, 6182, 4894, 5872, 220, 9519, 256, 8473, 220, 16, 17, 87, 18, 19, 198 },
    },
    // Zig snippet.
    .{
        .text = "fn main() !void {\n    var x: usize = 0;\n    while (x < 10) : (x += 1) {\n        std.debug.print(\"{d}\\n\", .{x});\n    }\n}\n",
        .ids = &.{ 8822, 1887, 368, 753, 1004, 341, 262, 762, 856, 25, 22301, 284, 220, 15, 280, 262, 1393, 320, 87, 366, 220, 16, 15, 8, 549, 320, 87, 1421, 220, 16, 8, 341, 286, 1460, 7883, 2214, 13976, 67, 11035, 77, 497, 659, 90, 87, 2960, 262, 456, 532 },
    },
    // C snippet.
    .{
        .text = "#include <stdio.h>\nint main(void) {\n    for (int i = 0; i < 100; i++) printf(\"%d,\", i * 2);\n    return 0;\n}\n",
        .ids = &.{ 1067, 366, 10345, 860, 397, 396, 1887, 4333, 8, 341, 262, 369, 320, 396, 600, 284, 220, 15, 26, 600, 366, 220, 16, 15, 15, 26, 600, 2457, 4100, 4430, 67, 57955, 600, 353, 220, 17, 317, 262, 470, 220, 15, 280, 532 },
    },
    // Prose with contractions and an em dash.
    .{
        .text = "It's a test — we've seen 42 cases; they'll agree it isn't 100% \"done\".\n",
        .ids = &.{ 2132, 594, 264, 1273, 1959, 582, 3003, 3884, 220, 19, 17, 5048, 26, 807, 3278, 7503, 432, 4436, 944, 220, 16, 15, 15, 4, 330, 10438, 22956 },
    },
    // Bare "<|" (no marker / unresolvable marker) must NOT force chunk
    // splits, while a real special token mid-text still resolves.
    .{
        .text = "pipe a<|b test <| bare ||| <|im_end|> tail<|\n",
        .ids = &.{ 13768, 264, 27, 91, 65, 1273, 82639, 12461, 1369, 91, 220, 151645, 9787, 27, 7360 },
    },
};

test "llama-tokenize parity on embedded fixtures (skips without models/)" {
    const allocator = std.testing.allocator;
    var file = gguf.File.loadMmap(allocator, std.testing.io, "models/Qwen3-0.6B-Q4_K_S.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();
    var tok = try Tokenizer.initFromGguf(allocator, &file, .{});
    defer tok.deinit();
    // Qwen3 declares the implemented pretokenizer: no mismatch recorded.
    try std.testing.expect(tok.pre_mismatch == null);

    for (parity_fixtures) |fixture| {
        const ids = try tok.encodeRaw(allocator, fixture.text);
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(u32, fixture.ids, ids);

        // And the byte-level decode round-trips exactly.
        const text = try tok.decode(allocator, ids);
        defer allocator.free(text);
        try std.testing.expectEqualStrings(fixture.text, text);
    }
}

test "StreamDecoder writes decoded tokens to a writer" {
    const allocator = std.testing.allocator;
    const vocab = [_][]const u8{ "a", "b", "c", "ab", "abc" };
    const merges = [_][]const u8{ "a b", "ab c" };
    var tok = try Tokenizer.initFromParts(allocator, &vocab, &merges, .{});
    defer tok.deinit();

    var sd = StreamDecoder.init(&tok);
    defer sd.deinit(allocator);
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try sd.push(allocator, 4, &w); // "abc"
    try sd.push(allocator, 0, &w); // "a"
    try sd.flush(&w);
    try std.testing.expectEqualStrings("abca", w.buffered());
}
