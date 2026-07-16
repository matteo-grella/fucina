//! Unit tests for the inkling multimodal projector's host-side numerics.
//! Tower/e2e parity vs the pinned llama.cpp mtmd oracle runs through the
//! example CLI gates (examples/inkling.zig --image/--audio + --embd-out).

const std = @import("std");
const mmproj = @import("mmproj.zig");

test "dmel quantizer: clamp, centers, ties toward the lower bin" {
    // centers are -7 + 9k/15; midpoint between k=7 and k=8 must pick 7.
    const q = mmproj.testing.quantizeDmel;
    try std.testing.expectEqual(@as(u8, 0), q(-100.0)); // clamped low
    try std.testing.expectEqual(@as(u8, 15), q(50.0)); // clamped high
    try std.testing.expectEqual(@as(u8, 0), q(-7.0));
    try std.testing.expectEqual(@as(u8, 15), q(2.0));
    const mid_7_8 = -7.0 + 9.0 * 7.5 / 15.0;
    try std.testing.expectEqual(@as(u8, 7), q(@floatCast(mid_7_8)));
}

test "bf16 round-trip matches round-to-nearest-even" {
    const r = mmproj.testing.bf16Round;
    try std.testing.expectEqual(@as(f32, 1.0), r(1.0));
    try std.testing.expectEqual(@as(f32, 0.0), r(0.0));
    // 1.0 + 2^-9 rounds down to 1.0 in bf16 (7 mantissa bits, tie-to-even).
    try std.testing.expectEqual(@as(f32, 1.0), r(1.0 + 0.001953125));
    // 1.0 + 3*2^-9 rounds up to 1.015625.
    try std.testing.expectEqual(@as(f32, 1.015625), r(1.0 + 3.0 * 0.001953125));
}

test "pillow lanczos resize: identity when size is unchanged" {
    const allocator = std.testing.allocator;
    var src: [4 * 3 * 3]u8 = undefined;
    for (&src, 0..) |*p, i| p.* = @intCast(i * 7 % 256);
    const out = try mmproj.testing.pillowResizeLanczos(allocator, &src, 4, 3, 4, 3);
    defer allocator.free(out);
    try std.testing.expectEqualSlices(u8, &src, out);
}

test "slaney filterbank: rows are nonnegative and low bins cover low freqs" {
    const allocator = std.testing.allocator;
    const fb = try mmproj.testing.slaneyFilterbank(allocator);
    defer allocator.free(fb);
    const bins = 801;
    try std.testing.expectEqual(@as(usize, 80 * bins), fb.len);
    for (fb) |v| try std.testing.expect(v >= 0);
    // First mel filter peaks in the first few FFT bins (10 Hz/bin).
    var found = false;
    for (fb[0..16]) |v| {
        if (v > 0) found = true;
    }
    try std.testing.expect(found);
}
