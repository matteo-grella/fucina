//! Hermetic tests for the Parakeet streaming building blocks.
const std = @import("std");
const fucina = @import("fucina");
const streaming = @import("streaming.zig");

const ExecContext = fucina.ExecContext;

test "streaming depthwise conv: chunked == full-sequence (cache_last_time equivalence)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const C = 5; // channels
    const K = 4; // taps -> left_pad = 3
    const T = 13; // total frames

    var xin: [T * C]f32 = undefined;
    for (&xin, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7) % 11)) * 0.1 - 0.5;
    var kin: [C * K]f32 = undefined;
    for (&kin, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 3) % 5)) * 0.2 - 0.4;

    var kernel = try fucina.Tensor(2).fromSlice(&ctx, .{ C, K }, &kin);
    defer kernel.deinit();

    // (a) Full sequence in one shot (cache starts zeroed = the leading causal pad).
    var full_cache = try streaming.ConvCache.init(allocator, C, K);
    defer full_cache.deinit(allocator);
    var x_full = try fucina.Tensor(2).fromSlice(&ctx, .{ T, C }, &xin);
    defer x_full.deinit();
    var y_full = try streaming.streamingDepthwiseConv(&ctx, &x_full, &kernel, &full_cache);
    defer y_full.deinit();
    const yf = try y_full.dataConst();

    // (b) The SAME input fed in chunks, carrying the cache across chunks. The
    //     chunk schedule includes a 1-frame chunk (< left_pad=3) to exercise the
    //     cache-shift path.
    var ch_cache = try streaming.ConvCache.init(allocator, C, K);
    defer ch_cache.deinit(allocator);
    const chunks = [_]usize{ 3, 1, 5, 4 }; // sums to T=13
    var yc: [T * C]f32 = undefined;
    var off: usize = 0;
    for (chunks) |cn| {
        var xc = try fucina.Tensor(2).fromSlice(&ctx, .{ cn, C }, xin[off * C ..][0 .. cn * C]);
        defer xc.deinit();
        var yck = try streaming.streamingDepthwiseConv(&ctx, &xc, &kernel, &ch_cache);
        defer yck.deinit();
        @memcpy(yc[off * C ..][0 .. cn * C], try yck.dataConst());
        off += cn;
    }

    // Cache-equivalence: chunked output == full-sequence output (the conv sums the
    // same K products per frame, with the cache reproducing the exact left context).
    for (0..T * C) |i| try std.testing.expectApproxEqAbs(yf[i], yc[i], 1e-6);

    // Sanity: a final reset() zeroes the cache.
    ch_cache.reset();
    for (ch_cache.data) |v| try std.testing.expectEqual(@as(f32, 0), v);
}

test "streaming attention mask: chunked_limited + empty-cache (Tc chunk-query rows)" {
    const allocator = std.testing.allocator;
    const ninf = -std.math.inf(f32);

    // cache_len=6, Tc=2 -> Tk=8, valid=6 (full, empty_cache=0); att=[2,1] ->
    // chunk=2, left_chunks=1. The mask is [Tc=2, Tk=8]: query qi -> gq=cache_len+qi.
    // qi=0,1 -> gq=6,7 -> cq=3: visible only for chunks 2,3 -> keys 4..7.
    {
        const mask = try streaming.streamingAttnMask(allocator, 2, 8, 6, 6, 2, 1);
        defer allocator.free(mask);
        for ([_]usize{ 0, 1 }) |qi| {
            for (0..4) |k| try std.testing.expectEqual(ninf, mask[qi * 8 + k]); // chunks 0,1 masked
            for (4..8) |k| try std.testing.expectEqual(@as(f32, 0), mask[qi * 8 + k]);
        }
    }
    // Partially-filled cache: cache_len=4, Tc=2 -> Tk=6, valid=2 (empty_cache=2);
    // att=[2,1]: chunk queries qi=0,1 (gq=4,5) mask the 2 unfilled cache cols [0,1].
    {
        const mask = try streaming.streamingAttnMask(allocator, 2, 6, 4, 2, 2, 1);
        defer allocator.free(mask);
        for ([_]usize{ 0, 1 }) |qi| {
            try std.testing.expectEqual(ninf, mask[qi * 6 + 0]);
            try std.testing.expectEqual(ninf, mask[qi * 6 + 1]);
            for (2..6) |k| try std.testing.expectEqual(@as(f32, 0), mask[qi * 6 + k]);
        }
    }
}

test "channel cache: drop-oldest/append + valid growth (cache_last_channel)" {
    const allocator = std.testing.allocator;
    var c = try streaming.ChannelCache.init(allocator, 3, 1); // cache_len=3, 1 channel
    defer c.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), c.valid);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, c.data);

    c.advance(&.{5}, 1);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 5 }, c.data);
    try std.testing.expectEqual(@as(usize, 1), c.valid);

    c.advance(&.{6}, 1);
    try std.testing.expectEqualSlices(f32, &.{ 0, 5, 6 }, c.data);
    c.advance(&.{7}, 1);
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 7 }, c.data);
    try std.testing.expectEqual(@as(usize, 3), c.valid);

    c.advance(&.{8}, 1); // valid capped at cache_len
    try std.testing.expectEqualSlices(f32, &.{ 6, 7, 8 }, c.data);
    try std.testing.expectEqual(@as(usize, 3), c.valid);

    c.advance(&.{ 9, 10 }, 2); // multi-frame chunk: drop 2, append [9,10]
    try std.testing.expectEqualSlices(f32, &.{ 8, 9, 10 }, c.data);
    try std.testing.expectEqual(@as(usize, 3), c.valid);
}
