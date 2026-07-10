//! Tests for chunker_stream.zig — incremental sentence chunker vs the
//! offline `chunker.chunkTextPunctuation`, including byte-split invariance
//! and the reference's fold-rule quirks.

const std = @import("std");

const chunker = @import("chunker.zig");
const chunker_stream = @import("chunker_stream.zig");

/// Drives the stream with `text` split into pushes of (cycling) sizes, then
/// flushes EOF. Returns the concatenated list of emitted chunks.
fn collectStream(
    allocator: std.mem.Allocator,
    text: []const u8,
    chunk_len: i32,
    min_chunk_len: i32,
    push_sizes: []const usize,
) !std.ArrayList([]u8) {
    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |c| allocator.free(c);
        out.deinit(allocator);
    }

    var stream = chunker_stream.Stream.init(allocator, chunk_len, min_chunk_len);
    defer stream.deinit();

    var pos: usize = 0;
    var k: usize = 0;
    while (pos < text.len) {
        const n = @min(push_sizes[k % push_sizes.len], text.len - pos);
        const ready = try stream.pushBytes(text[pos..][0..n]);
        defer allocator.free(ready);
        try out.appendSlice(allocator, ready);
        pos += n;
        k += 1;
    }
    const tail = try stream.flushEof();
    defer allocator.free(tail);
    try out.appendSlice(allocator, tail);
    return out;
}

fn freeAll(allocator: std.mem.Allocator, list: *std.ArrayList([]u8)) void {
    for (list.items) |c| allocator.free(c);
    list.deinit(allocator);
}

fn expectStreamEqualsOffline(text: []const u8, chunk_len: i32, push_sizes: []const usize) !void {
    const allocator = std.testing.allocator;

    const want = try chunker.chunkTextPunctuation(allocator, text, chunk_len, chunker.min_chunk_len_default);
    defer chunker.freeChunks(allocator, want);

    var got = try collectStream(allocator, text, chunk_len, chunker.min_chunk_len_default, push_sizes);
    defer freeAll(allocator, &got);

    try std.testing.expectEqual(want.len, got.items.len);
    for (want, got.items) |w, g| {
        try std.testing.expectEqualStrings(w, g);
    }
}

const long_text =
    "The morning sun rose slowly over the quiet valley, casting long shadows " ++
    "across the dewy fields. Birds began their daily chorus as the village " ++
    "below stirred to life. A baker opened his shop, filling the street with " ++
    "the warm scent of fresh bread. Children hurried along the cobblestone " ++
    "paths, laughing on their way to school, while the old clocktower chimed " ++
    "seven times.";

test "stream == offline chunker on multi-sentence text (various budgets)" {
    try expectStreamEqualsOffline(long_text, 250, &.{4096});
    try expectStreamEqualsOffline(long_text, 50, &.{4096});
    try expectStreamEqualsOffline(long_text, 75, &.{4096});
}

test "stream == offline chunker under byte-split (1, 7 bytes per push)" {
    try expectStreamEqualsOffline(long_text, 250, &.{1});
    try expectStreamEqualsOffline(long_text, 50, &.{7});
    try expectStreamEqualsOffline(long_text, 50, &.{ 3, 11, 1 });
}

test "stream == offline chunker with abbreviations and CJK punctuation" {
    const text = "Dr. Smith arrived at 9 a.m. sharp! He greeted Mrs. Jones warmly. " ++
        "\xe4\xbd\xa0\xe5\xa5\xbd\xe3\x80\x82\xe8\xbf\x99\xe6\x98\xaf\xe4\xb8\x80\xe4\xb8\xaa\xe6\xb5\x8b\xe8\xaf\x95\xe3\x80\x82 " ++
        "Then everyone went home; the day was done.";
    try expectStreamEqualsOffline(text, 40, &.{5});
    try expectStreamEqualsOffline(text, 200, &.{1});
}

test "look-ahead delay: a closed chunk is held until the next one is observed" {
    const allocator = std.testing.allocator;
    var stream = chunker_stream.Stream.init(allocator, 20, chunker.min_chunk_len_default);
    defer stream.deinit();

    // One full sentence: still open (a future sentence could extend the last
    // offline chunk) AND in the look-ahead slot -> nothing emits.
    const r1 = try stream.pushBytes("First sentence here. ");
    defer chunker.freeChunks(allocator, r1);
    try std.testing.expectEqual(@as(usize, 0), r1.len);

    // Second sentence closes: the first becomes stable but sits in the
    // look-ahead slot; only a THIRD closed sentence pushes it out.
    const r2 = try stream.pushBytes("Second one lands. ");
    defer chunker.freeChunks(allocator, r2);
    try std.testing.expectEqual(@as(usize, 0), r2.len);

    const r3 = try stream.pushBytes("Third sentence appears. ");
    defer chunker.freeChunks(allocator, r3);
    try std.testing.expectEqual(@as(usize, 1), r3.len);
    try std.testing.expectEqualStrings("First sentence here.", r3[0]);

    // EOF drains the rest.
    const tail = try stream.flushEof();
    defer chunker.freeChunks(allocator, tail);
    try std.testing.expectEqual(@as(usize, 2), tail.len);
    try std.testing.expectEqualStrings("Second one lands.", tail[0]);
    try std.testing.expectEqualStrings("Third sentence appears.", tail[1]);
}

test "fold rule: first short chunk folds into the second (stripped concat, reference quirk)" {
    const allocator = std.testing.allocator;

    // chunk_len 10: "A." (2 cps) stays alone in the offline min=0 re-parse
    // (the next sentence does not fit), then folds into the second chunk.
    // The reference streaming fold concatenates the STRIPPED chunks, so the
    // inter-chunk space is dropped ("A.Then...") — unlike the one-shot
    // offline fold, which folds raw codepoint runs and keeps the space.
    // Port matches the streaming reference verbatim.
    const text = "A. Then a much longer sentence. And a closing one arrives.";

    var got = try collectStream(allocator, text, 10, chunker.min_chunk_len_default, &.{4096});
    defer freeAll(allocator, &got);

    try std.testing.expectEqual(@as(usize, 2), got.items.len);
    try std.testing.expectEqualStrings("A.Then a much longer sentence.", got.items[0]);
    try std.testing.expectEqualStrings("And a closing one arrives.", got.items[1]);

    // The offline fold keeps the space — documented divergence.
    const offline = try chunker.chunkTextPunctuation(allocator, text, 10, chunker.min_chunk_len_default);
    defer chunker.freeChunks(allocator, offline);
    try std.testing.expectEqual(@as(usize, 2), offline.len);
    try std.testing.expectEqualStrings("A. Then a much longer sentence.", offline[0]);
}

test "fold rule: later short chunk folds into the previous one" {
    const allocator = std.testing.allocator;

    // chunk_len 12 keeps every sentence separate; "No" + "." glue makes a
    // 3-cp... use a 2-cp sentence "B!" mid-stream instead.
    const text = "Opening sentence one. B! Closing sentence lands here.";

    var got = try collectStream(allocator, text, 12, chunker.min_chunk_len_default, &.{3});
    defer freeAll(allocator, &got);

    try std.testing.expectEqual(@as(usize, 2), got.items.len);
    try std.testing.expectEqualStrings("Opening sentence one.B!", got.items[0]);
    try std.testing.expectEqualStrings("Closing sentence lands here.", got.items[1]);
}

test "flushEof resets the stream for the next line (line-oriented reuse)" {
    const allocator = std.testing.allocator;
    var stream = chunker_stream.Stream.init(allocator, 200, chunker.min_chunk_len_default);
    defer stream.deinit();

    const r1 = try stream.pushBytes("Line one says hello.");
    defer chunker.freeChunks(allocator, r1);
    const t1 = try stream.flushEof();
    defer chunker.freeChunks(allocator, t1);
    try std.testing.expectEqual(@as(usize, 0), r1.len);
    try std.testing.expectEqual(@as(usize, 1), t1.len);
    try std.testing.expectEqualStrings("Line one says hello.", t1[0]);

    const r2 = try stream.pushBytes("Line two follows after.");
    defer chunker.freeChunks(allocator, r2);
    const t2 = try stream.flushEof();
    defer chunker.freeChunks(allocator, t2);
    try std.testing.expectEqual(@as(usize, 0), r2.len);
    try std.testing.expectEqual(@as(usize, 1), t2.len);
    try std.testing.expectEqualStrings("Line two follows after.", t2[0]);
}

test "empty and whitespace-only input yields no chunks" {
    const allocator = std.testing.allocator;
    var stream = chunker_stream.Stream.init(allocator, 100, chunker.min_chunk_len_default);
    defer stream.deinit();

    const r = try stream.pushBytes("  \n\t ");
    defer chunker.freeChunks(allocator, r);
    try std.testing.expectEqual(@as(usize, 0), r.len);
    const t = try stream.flushEof();
    defer chunker.freeChunks(allocator, t);
    try std.testing.expectEqual(@as(usize, 0), t.len);
}
