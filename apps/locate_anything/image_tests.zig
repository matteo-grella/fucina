const std = @import("std");
const image = @import("image.zig");

const testing = std.testing;

test "png round-trip: write then decode is identity" {
    const allocator = testing.allocator;
    const w = 13;
    const h = 7;
    var rgb: [w * h * 3]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(42);
    rng.random().bytes(&rgb);

    // In-memory round trip: encode via the writer's chunk plumbing by writing
    // to a temp file, then decode and compare.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realPathFileAlloc(std.testing.io, ".", allocator);
    defer allocator.free(dir_path);
    const path = try std.fs.path.join(allocator, &.{ dir_path, "rt.png" });
    defer allocator.free(path);

    try image.writePng(allocator, std.testing.io, path, w, h, &rgb);
    var img = try image.loadPng(allocator, std.testing.io, path);
    defer img.deinit();

    try testing.expectEqual(@as(usize, w), img.w);
    try testing.expectEqual(@as(usize, h), img.h);
    try testing.expectEqualSlices(u8, &rgb, img.rgb);
}

test "png decoder rejects non-png bytes" {
    try testing.expectError(image.Error.NotPng, image.decodePng(testing.allocator, "not a png at all"));
}
