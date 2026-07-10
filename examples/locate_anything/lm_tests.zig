const std = @import("std");
const lm = @import("lm.zig");

const testing = std.testing;

test "mtp positions: recompute consecutive, block slots shifted by -1" {
    // cached_len=300, n_recompute=2, block=6 -> base=302.
    const positions = try lm.Lm.buildMtpPositions(testing.allocator, 300, 2, 6);
    defer testing.allocator.free(positions);
    try testing.expectEqualSlices(i32, &.{ 300, 301, 301, 302, 303, 304, 305, 306 }, positions);
}

test "mtp mask: causal base + bidirectional window + masked prev-last column" {
    // cached_len=4, n_recompute=1, block=6: n_new=7, full=11, base=5.
    const cached_len = 4;
    const n_recompute = 1;
    const block = 6;
    const n_new = n_recompute + block;
    const full = cached_len + n_new;
    const mask = try lm.Lm.buildMtpMask(testing.allocator, cached_len, n_recompute, block);
    defer testing.allocator.free(mask);
    try testing.expectEqual(@as(usize, n_new * full), mask.len);

    const neg = lm.mask_neg;
    // Recompute row 0 (apos=4): keys 0..4 visible, rest masked.
    for (0..full) |key| {
        const expected: f32 = if (key > 4) neg else 0.0;
        try testing.expectEqual(expected, mask[0 * full + key]);
    }
    // All 6 block rows: the 6 block keys [full-6, full) fully visible,
    // column full-7 (the last committed token) masked.
    for (n_new - block..n_new) |q| {
        for (full - block..full) |key| try testing.expectEqual(@as(f32, 0.0), mask[q * full + key]);
        try testing.expectEqual(neg, mask[q * full + (full - block - 1)]);
    }
    // Block row 0 (apos=base-1=4): keys 0..3 visible causally; key 4 is the
    // full-block-1 column (the last committed token) and is overridden to
    // masked for every block row — slot 0 carries its duplicate instead.
    for (0..cached_len) |key| {
        try testing.expectEqual(@as(f32, 0.0), mask[(n_new - block) * full + key]);
    }
    try testing.expectEqual(neg, mask[(n_new - block) * full + 4]);
    // Block row 1 (apos=5): key 5 is the first block key (full-6=5), inside
    // the bidirectional window override -> visible.
    try testing.expectEqual(@as(f32, 0.0), mask[(n_new - block + 1) * full + 5]);
}
