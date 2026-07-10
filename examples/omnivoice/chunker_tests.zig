//! Tests for chunker.zig against golden outputs of the C++ reference
//! (refs/omnivoice.cpp/src/text-chunker.h). Goldens were produced by a
//! standalone harness (scratchpad golden_text.cpp, clang++ -std=c++17 -O2)
//! that runs chunk_text_punctuation / add_punctuation / chunker_strip /
//! chunker_last_word / chunker_utf8_count / chunker_contains_chinese on the
//! inputs below and prints the results as Zig literals. All asserts are
//! byte-exact.

const std = @import("std");
const chunker = @import("chunker.zig");

const ChunkCase = struct {
    text: []const u8,
    chunk_len: i32,
    min_chunk_len: i32,
    expected: []const []const u8,
};

const chunk_cases = [_]ChunkCase{
    .{ // english_cl40
        .text = "Dr. Smith went to Washington. He arrived at 10 a.m. sharp! The meeting, e.g. the annual review, was hosted by Acme Inc. and lasted three hours. Mr. Jones asked: why so long? Nobody knew.",
        .chunk_len = 40,
        .min_chunk_len = 0,
        .expected = &.{
            "Dr. Smith went to Washington.",
            "He arrived at 10 a.m. sharp!",
            "The meeting, e.g. the annual review,",
            "was hosted by Acme Inc. and lasted three hours.",
            "Mr. Jones asked: why so long?",
            "Nobody knew.",
        },
    },
    .{ // english_cl40_min3
        .text = "Dr. Smith went to Washington. He arrived at 10 a.m. sharp! The meeting, e.g. the annual review, was hosted by Acme Inc. and lasted three hours. Mr. Jones asked: why so long? Nobody knew.",
        .chunk_len = 40,
        .min_chunk_len = 3,
        .expected = &.{
            "Dr. Smith went to Washington.",
            "He arrived at 10 a.m. sharp!",
            "The meeting, e.g. the annual review,",
            "was hosted by Acme Inc. and lasted three hours.",
            "Mr. Jones asked: why so long?",
            "Nobody knew.",
        },
    },
    .{ // english_cl80
        .text = "Dr. Smith went to Washington. He arrived at 10 a.m. sharp! The meeting, e.g. the annual review, was hosted by Acme Inc. and lasted three hours. Mr. Jones asked: why so long? Nobody knew.",
        .chunk_len = 80,
        .min_chunk_len = 3,
        .expected = &.{
            "Dr. Smith went to Washington. He arrived at 10 a.m. sharp! The meeting, e.g.",
            "the annual review, was hosted by Acme Inc. and lasted three hours.",
            "Mr. Jones asked: why so long? Nobody knew.",
        },
    },
    .{ // chinese_cl8
        .text = "\xE4\xBB\x8A\xE5\xA4\xA9\xE5\xA4\xA9\xE6\xB0\x94\xE5\xBE\x88\xE5\xA5\xBD\xE3\x80\x82\xE6\x88\x91\xE4\xBB\xAC\xE5\x8E\xBB\xE5\x85\xAC\xE5\x9B\xAD\xE6\x95\xA3\xE6\xAD\xA5\xE5\x90\xA7\xEF\xBC\x81\xE4\xBD\xA0\xE8\xA7\x89\xE5\xBE\x97\xE6\x80\x8E\xE4\xB9\x88\xE6\xA0\xB7\xEF\xBC\x9F\xE5\xA5\xBD\xE7\x9A\x84\xE3\x80\x82",
        .chunk_len = 8,
        .min_chunk_len = 0,
        .expected = &.{
            "\xE4\xBB\x8A\xE5\xA4\xA9\xE5\xA4\xA9\xE6\xB0\x94\xE5\xBE\x88\xE5\xA5\xBD\xE3\x80\x82",
            "\xE6\x88\x91\xE4\xBB\xAC\xE5\x8E\xBB\xE5\x85\xAC\xE5\x9B\xAD\xE6\x95\xA3\xE6\xAD\xA5\xE5\x90\xA7\xEF\xBC\x81",
            "\xE4\xBD\xA0\xE8\xA7\x89\xE5\xBE\x97\xE6\x80\x8E\xE4\xB9\x88\xE6\xA0\xB7\xEF\xBC\x9F",
            "\xE5\xA5\xBD\xE7\x9A\x84\xE3\x80\x82",
        },
    },
    .{ // chinese_cl8_min3
        .text = "\xE4\xBB\x8A\xE5\xA4\xA9\xE5\xA4\xA9\xE6\xB0\x94\xE5\xBE\x88\xE5\xA5\xBD\xE3\x80\x82\xE6\x88\x91\xE4\xBB\xAC\xE5\x8E\xBB\xE5\x85\xAC\xE5\x9B\xAD\xE6\x95\xA3\xE6\xAD\xA5\xE5\x90\xA7\xEF\xBC\x81\xE4\xBD\xA0\xE8\xA7\x89\xE5\xBE\x97\xE6\x80\x8E\xE4\xB9\x88\xE6\xA0\xB7\xEF\xBC\x9F\xE5\xA5\xBD\xE7\x9A\x84\xE3\x80\x82",
        .chunk_len = 8,
        .min_chunk_len = 3,
        .expected = &.{
            "\xE4\xBB\x8A\xE5\xA4\xA9\xE5\xA4\xA9\xE6\xB0\x94\xE5\xBE\x88\xE5\xA5\xBD\xE3\x80\x82",
            "\xE6\x88\x91\xE4\xBB\xAC\xE5\x8E\xBB\xE5\x85\xAC\xE5\x9B\xAD\xE6\x95\xA3\xE6\xAD\xA5\xE5\x90\xA7\xEF\xBC\x81",
            "\xE4\xBD\xA0\xE8\xA7\x89\xE5\xBE\x97\xE6\x80\x8E\xE4\xB9\x88\xE6\xA0\xB7\xEF\xBC\x9F",
            "\xE5\xA5\xBD\xE7\x9A\x84\xE3\x80\x82",
        },
    },
    .{ // chinese_cl20_min3
        .text = "\xE4\xBB\x8A\xE5\xA4\xA9\xE5\xA4\xA9\xE6\xB0\x94\xE5\xBE\x88\xE5\xA5\xBD\xE3\x80\x82\xE6\x88\x91\xE4\xBB\xAC\xE5\x8E\xBB\xE5\x85\xAC\xE5\x9B\xAD\xE6\x95\xA3\xE6\xAD\xA5\xE5\x90\xA7\xEF\xBC\x81\xE4\xBD\xA0\xE8\xA7\x89\xE5\xBE\x97\xE6\x80\x8E\xE4\xB9\x88\xE6\xA0\xB7\xEF\xBC\x9F\xE5\xA5\xBD\xE7\x9A\x84\xE3\x80\x82",
        .chunk_len = 20,
        .min_chunk_len = 3,
        .expected = &.{
            "\xE4\xBB\x8A\xE5\xA4\xA9\xE5\xA4\xA9\xE6\xB0\x94\xE5\xBE\x88\xE5\xA5\xBD\xE3\x80\x82\xE6\x88\x91\xE4\xBB\xAC\xE5\x8E\xBB\xE5\x85\xAC\xE5\x9B\xAD\xE6\x95\xA3\xE6\xAD\xA5\xE5\x90\xA7\xEF\xBC\x81",
            "\xE4\xBD\xA0\xE8\xA7\x89\xE5\xBE\x97\xE6\x80\x8E\xE4\xB9\x88\xE6\xA0\xB7\xEF\xBC\x9F\xE5\xA5\xBD\xE7\x9A\x84\xE3\x80\x82",
        },
    },
    .{ // mixed_cl24_min3
        .text = "OpenAI \xE5\x8F\x91\xE5\xB8\x83\xE4\xBA\x86 GPT\xE3\x80\x82It is, vs. the old one, much better! \xE4\xBB\xB7\xE6\xA0\xBC\xE6\x98\xAF 12.5 \xE7\xBE\x8E\xE5\x85\x83\xEF\xBC\x8C\xE5\xBE\x88\xE4\xBE\xBF\xE5\xAE\x9C\xE3\x80\x82",
        .chunk_len = 24,
        .min_chunk_len = 3,
        .expected = &.{
            "OpenAI \xE5\x8F\x91\xE5\xB8\x83\xE4\xBA\x86 GPT\xE3\x80\x82It is,",
            "vs. the old one,",
            "much better! \xE4\xBB\xB7\xE6\xA0\xBC\xE6\x98\xAF 12.",
            "5 \xE7\xBE\x8E\xE5\x85\x83\xEF\xBC\x8C\xE5\xBE\x88\xE4\xBE\xBF\xE5\xAE\x9C\xE3\x80\x82",
        },
    },
    .{ // mixed_cl64_min3
        .text = "OpenAI \xE5\x8F\x91\xE5\xB8\x83\xE4\xBA\x86 GPT\xE3\x80\x82It is, vs. the old one, much better! \xE4\xBB\xB7\xE6\xA0\xBC\xE6\x98\xAF 12.5 \xE7\xBE\x8E\xE5\x85\x83\xEF\xBC\x8C\xE5\xBE\x88\xE4\xBE\xBF\xE5\xAE\x9C\xE3\x80\x82",
        .chunk_len = 64,
        .min_chunk_len = 3,
        .expected = &.{
            "OpenAI \xE5\x8F\x91\xE5\xB8\x83\xE4\xBA\x86 GPT\xE3\x80\x82It is, vs. the old one, much better! \xE4\xBB\xB7\xE6\xA0\xBC\xE6\x98\xAF 12.5 \xE7\xBE\x8E\xE5\x85\x83\xEF\xBC\x8C",
            "\xE5\xBE\x88\xE4\xBE\xBF\xE5\xAE\x9C\xE3\x80\x82",
        },
    },
    .{ // quoted_cl20_min3
        .text = "She said: \"Wait.\" Then \xE2\x80\x9CReally?\xE2\x80\x9D he replied. 'Fine.' (Or so.) [Done.]",
        .chunk_len = 20,
        .min_chunk_len = 3,
        .expected = &.{
            "She said: \"Wait.\"",
            "Then \xE2\x80\x9CReally?\xE2\x80\x9D",
            "he replied. 'Fine.'",
            "(Or so.) [Done.]",
        },
    },
    .{ // quoted_cl300_min3
        .text = "She said: \"Wait.\" Then \xE2\x80\x9CReally?\xE2\x80\x9D he replied. 'Fine.' (Or so.) [Done.]",
        .chunk_len = 300,
        .min_chunk_len = 3,
        .expected = &.{
            "She said: \"Wait.\" Then \xE2\x80\x9CReally?\xE2\x80\x9D he replied. 'Fine.' (Or so.) [Done.]",
        },
    },
    .{ // oversized_cl30
        .text = "This single sentence has no terminal punctuation at all and just keeps going on and on far past any budget",
        .chunk_len = 30,
        .min_chunk_len = 3,
        .expected = &.{
            "This single sentence has no terminal punctuation at all and just keeps going on and on far past any budget",
        },
    },
    .{ // fragments_cl4_min3
        .text = "Hi. A? Ok! No. Sure thing. B; end",
        .chunk_len = 4,
        .min_chunk_len = 3,
        .expected = &.{
            "Hi.",
            "A?",
            "Ok!",
            "No. Sure thing.",
            "B;",
            "end",
        },
    },
    .{ // fragments_cl4_min0
        .text = "Hi. A? Ok! No. Sure thing. B; end",
        .chunk_len = 4,
        .min_chunk_len = 0,
        .expected = &.{
            "Hi.",
            "A?",
            "Ok!",
            "No. Sure thing.",
            "B;",
            "end",
        },
    },
    .{ // fragments_cl100_min3
        .text = "Hi. A? Ok! No. Sure thing. B; end",
        .chunk_len = 100,
        .min_chunk_len = 3,
        .expected = &.{
            "Hi. A? Ok! No. Sure thing. B; end",
        },
    },
    .{ // ws_heavy_cl30_min3
        .text = "  \xC2\xA0\x09 Mr. Brown\xC2\xA0lives on St. Mary Ave. near Mt. Hope.   It rains\xE3\x80\x80often.  \x0A",
        .chunk_len = 30,
        .min_chunk_len = 3,
        .expected = &.{
            "Mr. Brown\xC2\xA0lives on St. Mary Ave. near Mt. Hope.",
            "It rains\xE3\x80\x80often.",
        },
    },
    .{ // leading_punct_cl10_min3
        .text = "...!? Hello there. Bye.",
        .chunk_len = 10,
        .min_chunk_len = 3,
        .expected = &.{
            "...!?",
            "Hello there.",
            "Bye.",
        },
    },
    .{ // empty_cl10_min3
        .text = "",
        .chunk_len = 10,
        .min_chunk_len = 3,
        .expected = &.{},
    },
    .{ // only_punct_cl10_min3
        .text = "...",
        .chunk_len = 10,
        .min_chunk_len = 3,
        .expected = &.{
            "...",
        },
    },
};

test "chunkTextPunctuation matches the C++ reference byte for byte" {
    const allocator = std.testing.allocator;
    for (chunk_cases) |case| {
        const chunks = try chunker.chunkTextPunctuation(allocator, case.text, case.chunk_len, case.min_chunk_len);
        defer chunker.freeChunks(allocator, chunks);

        try std.testing.expectEqual(case.expected.len, chunks.len);
        for (case.expected, chunks) |want, got| {
            try std.testing.expectEqualStrings(want, got);
        }
    }
}

const AddPunctCase = struct {
    text: []const u8,
    expected: []const u8,
};

const add_punct_cases = [_]AddPunctCase{
    .{ .text = "Hello world", .expected = "Hello world." },
    .{ .text = "Hello world.", .expected = "Hello world." },
    .{ .text = "  Hello world!  ", .expected = "Hello world!" },
    .{ .text = "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C", .expected = "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C\xE3\x80\x82" },
    .{ .text = "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C\xE3\x80\x82", .expected = "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C\xE3\x80\x82" },
    .{ .text = "mixed \xE4\xBD\xA0\xE5\xA5\xBD", .expected = "mixed \xE4\xBD\xA0\xE5\xA5\xBD\xE3\x80\x82" },
    // Kana only: outside the CJK Unified block, so plain "." is appended.
    .{ .text = "\xE3\x82\xAB\xE3\x82\xBF\xE3\x82\xAB\xE3\x83\x8A", .expected = "\xE3\x82\xAB\xE3\x82\xBF\xE3\x82\xAB\xE3\x83\x8A." },
    .{ .text = "ends with ellipsis\xE2\x80\xA6", .expected = "ends with ellipsis\xE2\x80\xA6" },
    .{ .text = "   ", .expected = "" },
    .{ .text = "", .expected = "" },
    .{ .text = "quote \xE2\x80\x9D", .expected = "quote \xE2\x80\x9D" },
};

test "addPunctuation matches the C++ reference byte for byte" {
    const allocator = std.testing.allocator;
    for (add_punct_cases) |case| {
        const got = try chunker.addPunctuation(allocator, case.text);
        defer allocator.free(got);
        try std.testing.expectEqualStrings(case.expected, got);
    }
}

const MiscCase = struct {
    text: []const u8,
    stripped: []const u8,
    last_word: []const u8,
    cp_count: usize,
    chinese: bool,
};

const misc_cases = [_]MiscCase{
    .{ .text = "", .stripped = "", .last_word = "", .cp_count = 0, .chinese = false },
    .{ .text = "   ", .stripped = "", .last_word = "", .cp_count = 3, .chinese = false },
    .{ .text = "\xC2\xA0\xE3\x80\x80 hi there \xE2\x80\x89\xEF\xBB\xBF", .stripped = "hi there", .last_word = "there", .cp_count = 14, .chinese = false },
    .{ .text = "one\xC2\xA0two Mr.", .stripped = "one\xC2\xA0two Mr.", .last_word = "Mr.", .cp_count = 11, .chinese = false },
    .{ .text = "no-space-Mr.", .stripped = "no-space-Mr.", .last_word = "no-space-Mr.", .cp_count = 12, .chinese = false },
    .{ .text = "trailing spaces   ", .stripped = "trailing spaces", .last_word = "spaces", .cp_count = 18, .chinese = false },
    .{ .text = "\xE4\xBD\xA0\xE5\xA5\xBD abc", .stripped = "\xE4\xBD\xA0\xE5\xA5\xBD abc", .last_word = "abc", .cp_count = 6, .chinese = true },
    .{ .text = "\xE3\x82\xAB\xE3\x83\x8A", .stripped = "\xE3\x82\xAB\xE3\x83\x8A", .last_word = "\xE3\x82\xAB\xE3\x83\x8A", .cp_count = 2, .chinese = false },
    .{ .text = "\xFF\xFE invalid", .stripped = "\xFF\xFE invalid", .last_word = "invalid", .cp_count = 10, .chinese = false },
};

test "strip / lastWord / utf8Count / containsChinese match the C++ reference" {
    for (misc_cases) |case| {
        try std.testing.expectEqualStrings(case.stripped, chunker.strip(case.text));
        try std.testing.expectEqualStrings(case.last_word, chunker.lastWord(case.text));
        try std.testing.expectEqual(case.cp_count, chunker.utf8Count(case.text));
        try std.testing.expectEqual(case.chinese, chunker.containsChinese(case.text));
    }
}

test "utf8Len covers all lead-byte classes with the len-1 fallback" {
    try std.testing.expectEqual(@as(usize, 1), chunker.utf8Len('a'));
    try std.testing.expectEqual(@as(usize, 2), chunker.utf8Len(0xC2));
    try std.testing.expectEqual(@as(usize, 3), chunker.utf8Len(0xE4));
    try std.testing.expectEqual(@as(usize, 4), chunker.utf8Len(0xF0));
    // Invalid lead bytes (continuation 0x80.., 0xF8..) fall back to 1.
    try std.testing.expectEqual(@as(usize, 1), chunker.utf8Len(0x80));
    try std.testing.expectEqual(@as(usize, 1), chunker.utf8Len(0xBF));
    try std.testing.expectEqual(@as(usize, 1), chunker.utf8Len(0xF8));
    try std.testing.expectEqual(@as(usize, 1), chunker.utf8Len(0xFF));
}

test "lastCodepoint walks back over continuation bytes" {
    try std.testing.expectEqualStrings("", chunker.lastCodepoint(""));
    try std.testing.expectEqualStrings("b", chunker.lastCodepoint("ab"));
    try std.testing.expectEqualStrings("\xE3\x80\x82", chunker.lastCodepoint("a\xE3\x80\x82"));
    // All continuation bytes: reference returns the whole string.
    try std.testing.expectEqualStrings("\x80\x81", chunker.lastCodepoint("\x80\x81"));
}

test "min_chunk_len_default mirrors OMNIVOICE_MIN_CHUNK_LEN" {
    try std.testing.expectEqual(@as(i32, 3), chunker.min_chunk_len_default);
}
