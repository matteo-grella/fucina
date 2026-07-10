//! Tests for play.zig. The Ring tests are pure (no device). The Player
//! test opens a REAL playback device and is gated behind
//! OMNIVOICE_AUDIO_DEVICE_TESTS (same pattern as the OMNIVOICE_PARITY
//! gates) so `zig build test` stays CI/headless safe:
//!
//!   OMNIVOICE_AUDIO_DEVICE_TESTS=1 zig build test

const std = @import("std");
const play = @import("play.zig");

test "ring: push then pop returns the same samples" {
    const allocator = std.testing.allocator;
    var ring = try play.Ring.init(allocator, 16);
    defer ring.deinit(allocator);

    const in = [_]f32{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(usize, 5), ring.pushSlice(&in));
    try std.testing.expectEqual(@as(usize, 5), ring.len());
    try std.testing.expectEqual(@as(usize, 11), ring.space());

    var out: [5]f32 = undefined;
    try std.testing.expectEqual(@as(usize, 5), ring.popSlice(&out));
    try std.testing.expectEqualSlices(f32, &in, &out);
    try std.testing.expectEqual(@as(usize, 0), ring.len());
}

test "ring: push beyond capacity is truncated, pop beyond fill is short" {
    const allocator = std.testing.allocator;
    var ring = try play.Ring.init(allocator, 8);
    defer ring.deinit(allocator);

    const in = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try std.testing.expectEqual(@as(usize, 8), ring.pushSlice(&in));
    try std.testing.expectEqual(@as(usize, 0), ring.pushSlice(&in));
    try std.testing.expectEqual(@as(usize, 0), ring.space());

    // Short pop: the untouched tail is the caller's silence region.
    var out = [_]f32{-1} ** 12;
    try std.testing.expectEqual(@as(usize, 8), ring.popSlice(&out));
    try std.testing.expectEqualSlices(f32, in[0..8], out[0..8]);
    try std.testing.expectEqual(@as(f32, -1), out[8]);

    // Empty pop returns 0 and leaves the buffer alone.
    try std.testing.expectEqual(@as(usize, 0), ring.popSlice(&out));
}

test "ring: wraparound preserves order across many uneven cycles" {
    const allocator = std.testing.allocator;
    var ring = try play.Ring.init(allocator, 32);
    defer ring.deinit(allocator);

    // Push 7, pop 5 repeatedly: positions wrap the 32-frame buffer many
    // times while the stream stays contiguous.
    var next_in: f32 = 0;
    var next_out: f32 = 0;
    var chunk: [7]f32 = undefined;
    var out: [5]f32 = undefined;
    for (0..200) |_| {
        for (&chunk) |*s| {
            s.* = next_in;
            next_in += 1;
        }
        var rest: []const f32 = &chunk;
        while (rest.len > 0) {
            const pushed = ring.pushSlice(rest);
            rest = rest[pushed..];
            if (rest.len == 0) break;
            const got = ring.popSlice(&out);
            for (out[0..got]) |s| {
                try std.testing.expectEqual(next_out, s);
                next_out += 1;
            }
        }
        const got = ring.popSlice(&out);
        for (out[0..got]) |s| {
            try std.testing.expectEqual(next_out, s);
            next_out += 1;
        }
    }
    // Drain the remainder.
    while (true) {
        const got = ring.popSlice(&out);
        if (got == 0) break;
        for (out[0..got]) |s| {
            try std.testing.expectEqual(next_out, s);
            next_out += 1;
        }
    }
    try std.testing.expectEqual(next_in, next_out);
}

test "ring: len/space stay consistent while positions grow monotonically" {
    const allocator = std.testing.allocator;
    var ring = try play.Ring.init(allocator, 4);
    defer ring.deinit(allocator);

    const in = [_]f32{ 1, 2, 3 };
    var out: [2]f32 = undefined;
    for (0..100) |_| {
        const pushed = ring.pushSlice(&in);
        try std.testing.expectEqual(ring.buf.len - ring.space(), ring.len());
        const got = ring.popSlice(&out);
        try std.testing.expect(got <= ring.buf.len);
        try std.testing.expect(pushed <= in.len);
        try std.testing.expectEqual(ring.buf.len - ring.space(), ring.len());
    }
}

test "OMNIVOICE_AUDIO_DEVICE_TESTS: player plays a short tone on the default device" {
    if (std.c.getenv("OMNIVOICE_AUDIO_DEVICE_TESTS") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var player = try play.Player.init(allocator, io, .{ .sample_rate = 24000 });
    defer player.deinit();

    // 0.25 s of a quiet 440 Hz sine.
    const n = 6000;
    const tone = try allocator.alloc(f32, n);
    defer allocator.free(tone);
    for (tone, 0..) |*s, i| {
        const t = @as(f32, @floatFromInt(i)) / 24000.0;
        s.* = 0.1 * @sin(2.0 * std.math.pi * 440.0 * t);
    }

    try player.pushSamples(tone);
    const stats = try player.drainAndStop();
    try std.testing.expectEqual(@as(usize, n), stats.frames_played);
}
