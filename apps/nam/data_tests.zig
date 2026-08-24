//! Behavioral tests for the NAM training-data pipeline (`data.zig`): latency
//! calibration on a synthetic blip, delay-alignment slicing, dataset
//! windowing, and the ESR metric + its quality bands.
const std = @import("std");
const data = @import("data.zig");

const calibrateLatencyV3 = data.calibrateLatencyV3;
const applyDelay = data.applyDelay;
const Dataset = data.Dataset;
const esr = data.esr;
const esrComment = data.esrComment;
const v3 = data.v3;

test "latency calibration finds a synthetic blip offset" {
    const allocator = std.testing.allocator;
    const y = try allocator.alloc(f32, v3.first_blips_start + v3.t_blips + 1000);
    defer allocator.free(y);
    @memset(y, 0);
    // Quiet noise floor everywhere in the blip segment.
    for (y[v3.first_blips_start..], 0..) |*v, i| v.* = 0.0001 * @sin(@as(f32, @floatFromInt(i)));
    // Blips arriving 17 samples late.
    const observed_delay = 17;
    for (v3.blip_locations) |loc| {
        for (0..200) |j| y[loc + observed_delay + j] = 0.5;
    }
    const cal = calibrateLatencyV3(y);
    try std.testing.expect(!cal.warn_not_detected);
    try std.testing.expectEqual(@as(i64, observed_delay), cal.delay.?);
    try std.testing.expectEqual(@as(i64, observed_delay - 1), cal.recommended.?);
}

test "delay alignment matches upstream slicing" {
    const x = [_]f32{ 0, 1, 2, 3, 4, 5 };
    const y = [_]f32{ 10, 11, 12, 13, 14, 15 };
    const pos = try applyDelay(&x, &y, 2);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 3 }, pos.x);
    try std.testing.expectEqualSlices(f32, &.{ 12, 13, 14, 15 }, pos.y);
    const neg = try applyDelay(&x, &y, -2);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 4, 5 }, neg.x);
    try std.testing.expectEqualSlices(f32, &.{ 10, 11, 12, 13 }, neg.y);
}

test "dataset windowing matches upstream getitem" {
    const x = [_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const y = [_]f32{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    const ds = Dataset{ .x = &x, .y = &y, .nx = 3, .ny = 2 };
    // len = (10 - 3 + 1) // 2 = 4
    try std.testing.expectEqual(@as(usize, 4), ds.len());
    const ex = ds.get(1);
    // i = 2: input = x[2..2+3+2-1] = x[2..6]; target = y[2+2 .. 2+2+2]
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 4, 5 }, ex.input);
    try std.testing.expectEqualSlices(f32, &.{ 14, 15 }, ex.target);
}

test "esr and quality bands" {
    const a = [_]f32{ 1, 2, 3, 4 };
    var b = a;
    try std.testing.expectEqual(@as(f64, 0), esr(&a, &b));
    b[0] = 1.1;
    try std.testing.expect(esr(&a, &b) > 0);
    try std.testing.expectEqualStrings("Great!", esrComment(0.005));
    try std.testing.expectEqualStrings("Not bad!", esrComment(0.02));
}
