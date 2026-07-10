const std = @import("std");
const image = @import("image.zig");

fn makeTestImage(allocator: std.mem.Allocator, w: usize, h: usize) !image.Image {
    var img = try image.Image.initRgb(allocator, w, h);
    var y: usize = 0;
    while (y < h) : (y += 1) {
        var x: usize = 0;
        while (x < w) : (x += 1) {
            var c: usize = 0;
            while (c < 3) : (c += 1) {
                img.pixels[(y * w + x) * 3 + c] = @truncate(x *% 13 +% y *% 7 +% c *% 29 +% 3);
            }
        }
    }
    return img;
}

test "raw-pixel ingest round-trips (reference-pixel path)" {
    const allocator = std.testing.allocator;
    var img = try makeTestImage(allocator, 9, 4);
    defer img.deinit();

    const raw = try image.toRaw(img, allocator);
    defer allocator.free(raw);
    var back = try image.fromRaw(allocator, raw);
    defer back.deinit();

    try std.testing.expectEqual(img.width, back.width);
    try std.testing.expectEqual(img.height, back.height);
    try std.testing.expectEqual(@as(usize, 3), back.channels);
    try std.testing.expectEqualSlices(u8, img.pixels, back.pixels);
}

test "PNG encode -> decode round-trips exactly" {
    const allocator = std.testing.allocator;
    var img = try makeTestImage(allocator, 7, 5);
    defer img.deinit();

    const png = try image.encodePng(allocator, img);
    defer allocator.free(png);
    // Valid PNG signature.
    try std.testing.expectEqualSlices(u8, &.{ 137, 80, 78, 71, 13, 10, 26, 10 }, png[0..8]);

    var back = try image.decodePng(allocator, png);
    defer back.deinit();
    try std.testing.expectEqual(img.width, back.width);
    try std.testing.expectEqual(img.height, back.height);
    try std.testing.expectEqualSlices(u8, img.pixels, back.pixels);
}

test "decodePng rejects a non-PNG blob" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.NotPng, image.decodePng(allocator, "not a png at all"));
}
