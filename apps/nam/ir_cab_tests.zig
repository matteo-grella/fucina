//! Behavioral tests for the cabinet IR module (`ir_cab.zig`): chunked vs
//! one-shot processing, in-place safety, max-length truncation, and the cubic
//! resampler's boundary safety + linear-ramp exactness.
const std = @import("std");
const ir_cab = @import("ir_cab.zig");

const IrCab = ir_cab.IrCab;
const max_length = ir_cab.max_length;
const resampleIrCubic = ir_cab.resampleIrCubic;

test "ir cab: chunked processing equals one-shot" {
    const allocator = std.testing.allocator;
    var ir: [40]f32 = undefined;
    for (&ir, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.21);
    var cab = try IrCab.init(allocator, &ir, 48000, 48000, 128);
    defer cab.deinit();

    var input: [100]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.37);

    var oneshot: [100]f32 = undefined;
    cab.process(&input, &oneshot, 100);

    cab.reset();
    var chunked: [100]f32 = undefined;
    var off: usize = 0;
    for ([_]usize{ 1, 7, 3, 16, 13, 60 }) |n| {
        cab.process(input[off..], chunked[off..], n);
        off += n;
    }
    try std.testing.expectEqual(@as(usize, 100), off);
    // Each output reads the identical contiguous window in both paths => exact.
    try std.testing.expectEqualSlices(f32, &oneshot, &chunked);
}

test "ir cab: in-place processing matches separate buffers" {
    const allocator = std.testing.allocator;
    const ir = [_]f32{ 0.3, 0.2, 0.1, 0.05 };
    var cab1 = try IrCab.init(allocator, &ir, 48000, 48000, 32);
    defer cab1.deinit();
    var cab2 = try IrCab.init(allocator, &ir, 48000, 48000, 32);
    defer cab2.deinit();

    var buf: [16]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)));
    var sep: [16]f32 = undefined;
    cab1.process(&buf, &sep, 16);
    cab2.process(&buf, &buf, 16); // in place: input == output
    try std.testing.expectEqualSlices(f32, &sep, &buf);
}

test "ir cab: truncates to max_length" {
    const allocator = std.testing.allocator;
    const long = try allocator.alloc(f32, max_length + 100);
    defer allocator.free(long);
    @memset(long, 0.001);
    var cab = try IrCab.init(allocator, long, 48000, 48000, 64);
    defer cab.deinit();
    try std.testing.expectEqual(max_length, cab.taps);
}

test "resampleIrCubic stays in bounds at the tail (B1 regression)" {
    const allocator = std.testing.allocator;
    // Upsampling pushes the final `time` to ~end_time, where floor(time/inc)
    // can round to len-1; the boundary clamp must not read past the input.
    // (Safety checks are on under `zig build test`, so an OOB would panic.)
    var ramp: [10]f32 = undefined;
    for (&ramp, 0..) |*v, i| v.* = @floatFromInt(i);
    inline for (.{ .{ 48000.0, 96000.0 }, .{ 44100.0, 48000.0 }, .{ 48000.0, 88200.0 } }) |pair| {
        const out = try resampleIrCubic(allocator, &ramp, pair[0], pair[1]);
        defer allocator.free(out);
        try std.testing.expect(out.len > 0);
    }
    // The full load path: IR captured at 48k loaded into a 96k session.
    var cab = try IrCab.init(allocator, &ramp, 48000, 96000, 64);
    defer cab.deinit();
    try std.testing.expect(cab.taps > 0);
}

test "resampleIrCubic is exact on a linear ramp (interior)" {
    const allocator = std.testing.allocator;
    // Cubic interpolation reproduces linear functions exactly, so uniformly
    // resampled samples of a line have ~zero second difference.
    var ramp: [50]f32 = undefined;
    for (&ramp, 0..) |*v, i| v.* = 1.0 + 0.5 * @as(f32, @floatFromInt(i));
    const out = try resampleIrCubic(allocator, &ramp, 10000.0, 23000.0);
    defer allocator.free(out);
    try std.testing.expect(out.len > 4);
    // Skip the final segment (its p[3] neighbour is clamped at the boundary).
    var i: usize = 2;
    while (i < out.len - 2) : (i += 1) {
        const second_diff = out[i] - 2.0 * out[i - 1] + out[i - 2];
        try std.testing.expectApproxEqAbs(@as(f32, 0), second_diff, 1e-3);
    }
}
