//! Behavioral tests for the suffix-automaton draft source (`sam_index.zig`):
//! brute-force matchLen/draft parity (random + adversarial streams), recency
//! draft selection, self-match exclusion, frozen-cursor queries, the
//! FrozenSource adapter, memory bounds, the stream-length ceiling, and the
//! observe/suggest desync/degraded facade.
const std = @import("std");
const sam_index = @import("sam_index.zig");

const SamIndex = sam_index.SamIndex;
const FrozenSource = sam_index.FrozenSource;

/// Brute force: longest L such that the length-L suffix of `stream` occurs in
/// stream[0..n-1], i.e. has an occurrence ending at index <= n-2. Early break
/// is valid by monotonicity: if the length-L suffix occurs ending at j, its
/// own length-(L-1) suffix occurs ending at the same j.
fn bruteLongestPriorSuffix(stream: []const u32) usize {
    const n = stream.len;
    if (n < 2) return 0;
    var l: usize = 1;
    while (l < n) : (l += 1) {
        if (std.mem.indexOf(u32, stream[0 .. n - 1], stream[n - l ..]) == null) return l - 1;
    }
    return n - 1;
}

/// Brute force for frozen mode: longest suffix of `q` that is a substring of
/// `doc` (same monotone early break).
fn bruteLongestSuffixIn(doc: []const u32, q: []const u32) usize {
    var l: usize = 1;
    while (l <= q.len) : (l += 1) {
        if (std.mem.indexOf(u32, doc, q[q.len - l ..]) == null) return l - 1;
    }
    return q.len;
}

/// A draft is correct iff (matched suffix ++ draft) occurs in the stream with
/// the suffix part ending strictly before the last index (prior occurrence).
fn expectDraftFollowsPriorOccurrence(stream: []const u32, match_len: usize, draft_buf: []const usize) !void {
    const n = stream.len;
    var cat_buf: [512]u32 = undefined;
    @memcpy(cat_buf[0..match_len], stream[n - match_len ..]);
    for (draft_buf, cat_buf[match_len..][0..draft_buf.len]) |d, *c| c.* = @intCast(d);
    const cat = cat_buf[0 .. match_len + draft_buf.len];

    var from: usize = 0;
    while (std.mem.indexOfPos(u32, stream, from, cat)) |idx| : (from = idx + 1) {
        if (idx + match_len <= n - 1) return; // suffix part ends at <= n-2
    }
    return error.TestExpectedDraftFromPriorOccurrence;
}

fn expectBruteParity(index: *SamIndex, stream: []const u32) !void {
    try std.testing.expectEqual(bruteLongestPriorSuffix(stream), index.matchLen());
    var dbuf: [8]usize = undefined;
    const k = index.draft(&dbuf);
    if (index.matchLen() < index.min_match) {
        try std.testing.expectEqual(@as(usize, 0), k);
    } else if (k > 0) {
        try expectDraftFollowsPriorOccurrence(stream, index.matchLen(), dbuf[0..k]);
    }
}

test "property: matchLen + draft match brute force on random streams (alphabets 2/5/31)" {
    const allocator = std.testing.allocator;
    const alphabets = [_]u32{ 2, 5, 31 };
    for (alphabets, 0..) |alpha, ai| {
        var prng = std.Random.DefaultPrng.init(0x5a3c_0001 + ai);
        const random = prng.random();

        var index = try SamIndex.init(allocator);
        defer index.deinit();
        var stream: std.ArrayList(u32) = .empty;
        defer stream.deinit(allocator);

        // Vocab-scale token values: the map key packs tokens into 32 bits.
        const base: u32 = 151_000;
        var i: usize = 0;
        while (i < 200) : (i += 1) {
            const t = base + random.uintLessThan(u32, alpha);
            try stream.append(allocator, t);
            try index.append(&.{t});
            try expectBruteParity(&index, stream.items);
        }

        const n = stream.items.len;
        try std.testing.expect(index.stateCount() <= 2 * n + 1);
        try std.testing.expect(index.transitionCount() <= 3 * n);
        try std.testing.expectEqual(index.transitionCount(), index.edges.items.len);
    }
}

test "matchLen matches brute force on adversarial deterministic streams" {
    const allocator = std.testing.allocator;
    const cases = [_][]const u32{
        // all-same: maximal overlap, matchLen = i after the (i+1)-th token
        &.{ 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 },
        // "abcbc": the classic split/clone shape
        &.{ 1, 2, 3, 2, 3 },
        // clone-heavy Fibonacci word prefix (abaababaabaab)
        &.{ 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2 },
        // "abcabd": shared prefix, divergent tail
        &.{ 1, 2, 3, 1, 2, 9 },
        // period-3 repetition
        &.{ 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5 },
    };
    for (cases) |case| {
        var index = try SamIndex.init(allocator);
        defer index.deinit();
        for (case, 0..) |t, i| {
            try index.append(&.{t});
            try expectBruteParity(&index, case[0 .. i + 1]);
        }
        const n = case.len;
        try std.testing.expect(index.stateCount() <= 2 * n + 1);
        try std.testing.expect(index.transitionCount() <= 3 * n);
    }
    // Pin the all-same expectation explicitly (overlapping prior occurrence).
    var index = try SamIndex.init(allocator);
    defer index.deinit();
    try index.append(&.{ 7, 7, 7 });
    try std.testing.expectEqual(@as(usize, 2), index.matchLen());
}

test "draft prefers the most recent prior occurrence" {
    const allocator = std.testing.allocator;

    // Two prior occurrences of [10,11,12]: one followed by 90, one by 91.
    // Recency: the draft must replay what followed the LATER one.
    {
        var index = try SamIndex.init(allocator);
        defer index.deinit();
        try index.append(&.{ 10, 11, 12, 90, 10, 11, 12, 91, 10, 11, 12 });
        try std.testing.expectEqual(@as(usize, 3), index.matchLen());

        var buf: [8]usize = undefined;
        const n = index.draft(&buf);
        try std.testing.expectEqual(@as(usize, 4), n); // tokens after index 6
        try std.testing.expectEqualSlices(usize, &.{ 91, 10, 11, 12 }, buf[0..n]);
    }

    // Single prior occurrence: falls back to it (the first one).
    {
        var index = try SamIndex.init(allocator);
        defer index.deinit();
        try index.append(&.{ 10, 11, 12, 90, 10, 11, 12 });
        try std.testing.expectEqual(@as(usize, 3), index.matchLen());

        var buf: [8]usize = undefined;
        const n = index.draft(&buf);
        try std.testing.expectEqual(@as(usize, 4), n); // tokens after index 2
        try std.testing.expectEqualSlices(usize, &.{ 90, 10, 11, 12 }, buf[0..n]);
    }

    // Budget caps: buf.len and max_draft both bound the draft.
    {
        var index = try SamIndex.init(allocator);
        defer index.deinit();
        try index.append(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 1, 2 });
        try std.testing.expectEqual(@as(usize, 2), index.matchLen());

        var small: [3]usize = undefined;
        try std.testing.expectEqual(@as(usize, 3), index.draft(&small));
        try std.testing.expectEqualSlices(usize, &.{ 3, 4, 5 }, small[0..3]);

        var big: [32]usize = undefined;
        index.max_draft = 4;
        try std.testing.expectEqual(@as(usize, 4), index.draft(&big));

        index.min_match = 3; // current match (2) is now too short
        try std.testing.expectEqual(@as(usize, 0), index.draft(&big));
    }
}

test "no self-copy: unique tails reflect only PRIOR occurrences" {
    const allocator = std.testing.allocator;

    // All-distinct stream: nothing ever recurs, matchLen stays 0.
    {
        var index = try SamIndex.init(allocator);
        defer index.deinit();
        var t: usize = 100;
        while (t < 120) : (t += 1) {
            try index.append(&.{t});
            try std.testing.expectEqual(@as(usize, 0), index.matchLen());
        }
        var buf: [8]usize = undefined;
        try std.testing.expectEqual(@as(usize, 0), index.draft(&buf));
    }

    // Repeating prefix, unique tail: the trivial (self-matching) SAM would
    // report the whole stream; the correct answer is 0.
    {
        var index = try SamIndex.init(allocator);
        defer index.deinit();
        try index.append(&.{ 1, 2, 3, 1, 2, 9 });
        try std.testing.expectEqual(@as(usize, 0), index.matchLen());
        // ... and the match recovers once the tail recurs.
        try index.append(&.{ 2, 9 });
        try std.testing.expectEqual(@as(usize, 2), index.matchLen());
    }
}

test "frozen cursor matches brute force over an external query stream" {
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(0xf0_2e_11);
    const random = prng.random();

    var index = try SamIndex.init(allocator);
    defer index.deinit();
    var doc: std.ArrayList(u32) = .empty;
    defer doc.deinit(allocator);
    var i: usize = 0;
    while (i < 80) : (i += 1) {
        const t = random.uintLessThan(u32, 4);
        try doc.append(allocator, t);
        try index.append(&.{t});
    }
    index.freeze();

    var cursor = SamIndex.Cursor{};
    var query: std.ArrayList(u32) = .empty;
    defer query.deinit(allocator);
    i = 0;
    while (i < 150) : (i += 1) {
        const t = random.uintLessThan(u32, 4);
        try query.append(allocator, t);
        index.advance(&cursor, t);
        try std.testing.expectEqual(bruteLongestSuffixIn(doc.items, query.items), cursor.len);

        // Any draft must be the document's continuation of the matched
        // suffix: doc must contain (matched suffix ++ draft).
        var dbuf: [6]usize = undefined;
        const k = index.draftFrom(cursor, &dbuf);
        if (cursor.len < index.min_match) {
            try std.testing.expectEqual(@as(usize, 0), k);
        } else if (k > 0) {
            var cat: [256]u32 = undefined;
            const ml: usize = cursor.len;
            @memcpy(cat[0..ml], query.items[query.items.len - ml ..]);
            for (dbuf[0..k], cat[ml..][0..k]) |d, *c| c.* = @intCast(d);
            try std.testing.expect(std.mem.indexOf(u32, doc.items, cat[0 .. ml + k]) != null);
        }
    }

    // Token absent from the doc (incl. out-of-u32-range) resets the cursor.
    index.advance(&cursor, 999);
    try std.testing.expectEqual(@as(u32, 0), cursor.len);
    index.advance(&cursor, std.math.maxInt(usize));
    try std.testing.expectEqual(@as(u32, 0), cursor.len);
}

test "frozen draftFrom: recency within the doc + empty continuation at doc end" {
    const allocator = std.testing.allocator;

    // [1,2,3] occurs ending at 2 (followed by 4,5,...) and at 7 (followed by
    // 9). The construction-time cursor refreshed the recent one: draft = [9].
    {
        var index = try SamIndex.init(allocator);
        defer index.deinit();
        try index.append(&.{ 1, 2, 3, 4, 5, 1, 2, 3, 9 });
        index.freeze();

        var cursor = SamIndex.Cursor{};
        for ([_]usize{ 1, 2, 3 }) |t| index.advance(&cursor, t);
        try std.testing.expectEqual(@as(u32, 3), cursor.len);

        var buf: [4]usize = undefined;
        const n = index.draftFrom(cursor, &buf);
        try std.testing.expectEqual(@as(usize, 1), n);
        try std.testing.expectEqual(@as(usize, 9), buf[0]);
    }

    // Match sits at the very end of the doc: nothing follows, draft = 0.
    {
        var index = try SamIndex.init(allocator);
        defer index.deinit();
        try index.append(&.{ 5, 6, 7 });
        index.freeze();

        var cursor = SamIndex.Cursor{};
        for ([_]usize{ 6, 7 }) |t| index.advance(&cursor, t);
        try std.testing.expectEqual(@as(u32, 2), cursor.len);

        var buf: [4]usize = undefined;
        try std.testing.expectEqual(@as(usize, 0), index.draftFrom(cursor, &buf));
    }
}

test "FrozenSource: cursor-owning DraftSource over a frozen document" {
    const allocator = std.testing.allocator;

    var index = try SamIndex.init(allocator);
    defer index.deinit();
    try index.append(&.{ 1, 2, 3, 4, 5, 1, 2, 3, 9 });
    index.freeze();

    var source = FrozenSource{ .index = &index };
    source.observe(&.{ 8, 1, 2, 3 }); // 8 mismatches, then the run matches

    var buf: [4]usize = undefined;
    const n = source.suggest(&.{ 8, 1, 2, 3 }, &buf);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(usize, 9), buf[0]);
    source.observeTopK(&.{}); // no-op, must compile against TopKRow
}

test "memory bounds: states <= 2n+1, transitions <= 3n, edges mirror trans" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xb0c4_2e);
    const random = prng.random();

    var index = try SamIndex.init(allocator);
    defer index.deinit();
    var i: usize = 0;
    const n: usize = 300;
    while (i < n) : (i += 1) {
        try index.append(&.{random.uintLessThan(u32, 3)});
    }
    try std.testing.expectEqual(n, index.tokenCount());
    try std.testing.expect(index.stateCount() <= 2 * n + 1);
    try std.testing.expect(index.transitionCount() <= 3 * n);
    try std.testing.expectEqual(index.transitionCount(), index.edges.items.len);
}

test "appendOne rejects streams at the 32-bit-index ceiling (StreamTooLong)" {
    const allocator = std.testing.allocator;
    var index = try SamIndex.init(allocator);
    defer index.deinit();

    // Fake the stream length at the ceiling (only the length is consulted
    // before the guard fires; nothing dereferences the items).
    const real_len = index.tokens.items.len;
    index.tokens.items.len = SamIndex.max_stream_len;
    try std.testing.expectError(error.StreamTooLong, index.append(&.{1}));
    index.tokens.items.len = real_len;

    // The failed append degraded the index, as any append error does (it
    // can no longer track the committed stream).
    try std.testing.expect(index.degraded);
    var buf: [4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 0), index.draft(&buf));
}

test "observe/suggest facade: desync guard and degraded inertness" {
    const allocator = std.testing.allocator;

    var index = try SamIndex.init(allocator);
    defer index.deinit();
    index.observe(&.{ 1, 2, 3, 4, 1, 2, 3 });

    var buf: [8]usize = undefined;
    // Context tail agrees with the observed stream -> draft flows through.
    const n = index.suggest(&.{ 1, 2, 3, 4, 1, 2, 3 }, &buf);
    try std.testing.expectEqual(@as(usize, 4), n);
    try std.testing.expectEqualSlices(usize, &.{ 4, 1, 2, 3 }, buf[0..n]);

    // Context tail disagrees (verifier is ahead of us) -> no draft.
    try std.testing.expectEqual(@as(usize, 0), index.suggest(&.{ 1, 2, 3, 4 }, &buf));

    // Degraded index goes inert: no matches, no drafts, observe is a no-op.
    index.degraded = true;
    try std.testing.expectEqual(@as(usize, 0), index.matchLen());
    try std.testing.expectEqual(@as(usize, 0), index.draft(&buf));
    try std.testing.expectEqual(@as(usize, 0), index.suggest(&.{3}, &buf));
    index.observe(&.{9});
    try std.testing.expectEqual(@as(usize, 7), index.tokenCount());
    try std.testing.expectError(error.Degraded, index.append(&.{9}));

    // observeTopK is a no-op on the SAM (committed-tokens-only learner).
    index.observeTopK(&.{.{ .token = 1, .topk = &.{2} }});
}
