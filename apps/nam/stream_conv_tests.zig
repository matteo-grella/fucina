//! Behavioral tests for the streaming causal 1-D convolution
//! (`stream_conv.zig`): chunked-vs-one-shot bit-exactness, the accumulate
//! path, NAM grouped-load/process ordering, and untrusted-shape rejection.

const std = @import("std");
const stream_conv = @import("stream_conv.zig");

const StreamConv = stream_conv.StreamConv;

test "stream conv chunked output equals one-shot output" {
    const allocator = std.testing.allocator;
    var conv = try StreamConv.init(allocator, 2, 3, 3, 4, true);
    defer conv.deinit();

    // Deterministic pseudo-random weights via the NAM load order.
    var stream: [2 * 3 * 3 + 3]f32 = undefined;
    for (&stream, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 1.3);
    _ = conv.loadNamWeights(&stream);

    var input: [40 * 2]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.37);

    var oneshot: [40 * 3]f32 = undefined;
    conv.process(&input, &oneshot, 40, false);

    conv.reset();
    var chunked: [40 * 3]f32 = undefined;
    var offset: usize = 0;
    const chunks = [_]usize{ 1, 7, 3, 16, 13 };
    for (chunks) |n| {
        conv.process(input[offset * 2 ..], chunked[offset * 3 ..], n, false);
        conv.push(input[offset * 2 ..], n);
        offset += n;
    }
    try std.testing.expectEqual(@as(usize, 40), offset);
    // Identical math per output row => exact equality across chunkings.
    try std.testing.expectEqualSlices(f32, &oneshot, &chunked);
}

test "stream conv accumulate adds on top" {
    const allocator = std.testing.allocator;
    var conv = try StreamConv.init(allocator, 1, 2, 1, 1, false);
    defer conv.deinit();
    conv.weight[0] = 2.0; // w[0,0,0]
    conv.weight[1] = 3.0; // w[0,0,1]

    const input = [_]f32{ 1, -1 };
    var out = [_]f32{ 10, 20, 30, 40 };
    conv.process(&input, &out, 2, true);
    try std.testing.expectEqualSlices(f32, &.{ 12, 23, 28, 37 }, &out);
}

test "stream conv grouped load and process match NAM grouped order" {
    const allocator = std.testing.allocator;
    var conv = try StreamConv.initGrouped(allocator, 4, 4, 2, 1, true, 2);
    defer conv.deinit();

    const weights = [_]f32{
        // group 0, out 0 then 1, local inputs 0 then 1, taps 0 then 1
        10, 5, 1, 0.5,
        20, 6, 2, 0.25,
        // group 1, out 2 then 3
        30, 7, 3, 0.125,
        40, 8, 4, 0.0625,
        // bias
        1,  2, 3, 4,
    };
    _ = conv.loadNamWeights(&weights);
    const input = [_]f32{
        1, 10, 100, 1000,
        2, 20, 200, 2000,
    };
    var out: [2 * 4]f32 = undefined;
    conv.process(&input, &out, 2, false);
    try std.testing.expectEqualSlices(f32, &.{
        11, 10.5, 828,  866.5,
        41, 59,   7653, 9729,
    }, &out);
}

test "stream conv init rejects degenerate / out-of-range shapes (untrusted .nam)" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.InvalidConvShape, StreamConv.init(a, 1, 1, 0, 1, false)); // taps 0
    try std.testing.expectError(error.InvalidConvShape, StreamConv.init(a, 1, 1, 1, 0, false)); // dilation 0
    try std.testing.expectError(error.InvalidConvShape, StreamConv.init(a, 1, 1, StreamConv.max_taps + 1, 1, false)); // taps > cap (process() OOB)
    try std.testing.expectError(error.InvalidConvShape, StreamConv.init(a, 1, 1, 1, StreamConv.max_dilation + 1, false)); // dilation > cap
    var ok = try StreamConv.init(a, 1, 1, 3, 2, false); // valid
    ok.deinit();
}
