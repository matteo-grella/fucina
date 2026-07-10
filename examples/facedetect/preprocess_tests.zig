//! Letterboxes the dumped source image (goldens/align-src-X.bin) to 640² and
//! compares to the reference detector blob (goldens/det-blob-X.bin) +
//! det_scale (goldens/det-scale-X.txt). Pure integer resize → bit-exact.

const std = @import("std");
const preprocess = @import("preprocess.zig");
const image = @import("image.zig");
const rec = @import("recognizer.zig");
const testlog = @import("testlog.zig");

fn checkFace(allocator: std.mem.Allocator, comptime letter: []const u8) !u64 {
    const sbytes = rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/align-src-" ++ letter ++ ".bin") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(sbytes);
    var src = try image.fromRaw(allocator, sbytes);
    defer src.deinit();

    var lb = try preprocess.letterbox(allocator, &src, 640);
    defer lb.img.deinit();

    const dsbytes = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/det-scale-" ++ letter ++ ".txt");
    defer allocator.free(dsbytes);
    const golden_ds = try std.fmt.parseFloat(f32, std.mem.trim(u8, dsbytes, " \n\r\t"));
    try std.testing.expect(@abs(lb.det_scale - golden_ds) <= 1e-4);

    const gbytes = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/det-blob-" ++ letter ++ ".bin");
    defer allocator.free(gbytes);
    var gc = try image.fromRaw(allocator, gbytes);
    defer gc.deinit();

    try std.testing.expectEqual(lb.img.pixels.len, gc.pixels.len);
    var max_diff: u64 = 0;
    for (lb.img.pixels, gc.pixels) |a, b| {
        const d = @abs(@as(i64, a) - @as(i64, b));
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

test "letterbox: 640 blob <= 1 (8-bit) + det_scale vs reference (face_a/b/c)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    inline for (.{ "a", "b", "c" }) |letter| {
        const md = checkFace(allocator, letter) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        testlog.print("[letterbox] face_{s}: max |diff| {d}\n", .{ letter, md });
        try std.testing.expect(md <= 1);
    }
}
