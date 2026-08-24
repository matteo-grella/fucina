//! Behavioral tests for the Token-Recycling adjacency matrix (`recycling.zig`):
//! update/draftChain round-trips, overwrite/truncation/sentinel semantics,
//! committed-bigram promotion, the DraftSource facade, and 151k-vocab init.
const std = @import("std");
const recycling = @import("recycling.zig");

const Recycling = recycling.Recycling;

test "update/draftChain round-trip; chain stops at unseen rows" {
    var rec = try Recycling.init(std.testing.allocator, 100);
    defer rec.deinit();

    rec.update(5, &.{ 7, 9, 11 });
    rec.update(7, &.{3});

    var buf: [4]usize = undefined;
    const n = rec.draftChain(5, &buf);
    // 5 -> 7 -> 3, row 3 unseen -> stop.
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqualSlices(usize, &.{ 7, 3 }, buf[0..n]);

    // Unseen start token drafts nothing.
    try std.testing.expectEqual(@as(usize, 0), rec.draftChain(42, &buf));
    // Out-of-range start token drafts nothing.
    try std.testing.expectEqual(@as(usize, 0), rec.draftChain(100_000, &buf));
}

test "overwrite semantics + truncation + sentinel padding" {
    var rec = try Recycling.init(std.testing.allocator, 64);
    defer rec.deinit();

    rec.update(5, &.{ 7, 9 });
    try std.testing.expectEqual(@as(u32, 7), rec.topkOf(5)[0]);
    try std.testing.expectEqual(Recycling.sentinel, rec.topkOf(5)[2]);

    // Overwrite replaces the whole row.
    rec.update(5, &.{8});
    try std.testing.expectEqual(@as(u32, 8), rec.topkOf(5)[0]);
    try std.testing.expectEqual(Recycling.sentinel, rec.topkOf(5)[1]);

    var buf: [2]usize = undefined;
    const n = rec.draftChain(5, &buf);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(usize, 8), buf[0]);

    // Longer-than-K updates truncate to K.
    const wide = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    rec.update(6, &wide);
    try std.testing.expectEqualSlices(u32, wide[0..Recycling.k], rec.topkOf(6));
}

test "observe promotes committed bigrams most-recent-first" {
    var rec = try Recycling.init(std.testing.allocator, 32);
    defer rec.deinit();

    rec.update(5, &.{ 7, 9, 11 });
    // Committed text actually went 5 -> 9: promote 9 to slot 0, keep 7.
    rec.observe(&.{ 5, 9 });
    try std.testing.expectEqualSlices(u32, &.{ 9, 7, 11 }, rec.topkOf(5)[0..3]);

    // Cold row: observe seeds slot 0.
    rec.observe(&.{ 1, 2 });
    try std.testing.expectEqual(@as(u32, 2), rec.topkOf(1)[0]);
    try std.testing.expectEqual(Recycling.sentinel, rec.topkOf(1)[1]);

    // Promoting an unseen candidate inserts at front and evicts the tail.
    const full = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 17 };
    rec.update(3, &full);
    rec.observe(&.{ 3, 30 });
    try std.testing.expectEqual(@as(u32, 30), rec.topkOf(3)[0]);
    try std.testing.expectEqualSlices(u32, full[0..7], rec.topkOf(3)[1..]);
}

test "suggest/observeTopK facade; chain capped by buf; self-loop is bounded" {
    var rec = try Recycling.init(std.testing.allocator, 16);
    defer rec.deinit();

    var buf: [3]usize = undefined;
    try std.testing.expectEqual(@as(usize, 0), rec.suggest(&.{}, &buf));

    rec.observeTopK(&.{
        .{ .token = 4, .topk = &.{ 5, 6 } },
        .{ .token = 5, .topk = &.{4} },
    });
    // 4 -> 5 -> 4 -> 5 ... capped at buf.len.
    const n = rec.suggest(&.{ 1, 4 }, &buf);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqualSlices(usize, &.{ 5, 4, 5 }, buf[0..n]);

    // Pure self-loop is also capped, not infinite.
    rec.update(7, &.{7});
    try std.testing.expectEqual(@as(usize, 3), rec.draftChain(7, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 7, 7, 7 }, buf[0..3]);
}

test "151k-vocab init: no leak, sane footprint" {
    const vocab: usize = 151_936; // Qwen3 vocabulary
    var rec = try Recycling.init(std.testing.allocator, vocab);
    defer rec.deinit();

    try std.testing.expectEqual(vocab * Recycling.k, rec.m.len);
    // ~4.6 MiB at K=8 — keep it under the 5 MB design budget.
    try std.testing.expect(rec.m.len * @sizeOf(u32) < 5 * 1024 * 1024);

    // Whole matrix starts unseen.
    try std.testing.expectEqual(Recycling.sentinel, rec.topkOf(0)[0]);
    try std.testing.expectEqual(Recycling.sentinel, rec.topkOf(vocab - 1)[Recycling.k - 1]);

    var buf: [8]usize = undefined;
    try std.testing.expectEqual(@as(usize, 0), rec.draftChain(vocab - 1, &buf));
}
