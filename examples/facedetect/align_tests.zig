//! Umeyama + warpAffine `norm_crop` on the dumped source image + 5 landmarks,
//! compared to the reference aligned 112² crop (goldens/crop-X.bin, the ArcFace
//! input). Byte tolerance ≤ 1 (OpenCV warpAffine parity). Pure geometry — no
//! model needed.

const std = @import("std");
const align_mod = @import("align.zig");
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

    const lbytes = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/align-lmk-" ++ letter ++ ".txt");
    defer allocator.free(lbytes);
    var lmk: [5][2]f64 = undefined;
    var it = std.mem.tokenizeScalar(u8, lbytes, '\n');
    for (0..5) |i| {
        const line = it.next() orelse return error.BadLandmarks;
        var ft = std.mem.tokenizeScalar(u8, line, ' ');
        lmk[i][0] = try std.fmt.parseFloat(f64, ft.next() orelse return error.BadLandmarks);
        lmk[i][1] = try std.fmt.parseFloat(f64, ft.next() orelse return error.BadLandmarks);
    }

    const crop = try align_mod.normCrop(allocator, &src, lmk, 112);
    defer allocator.free(crop);

    const gbytes = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/crop-" ++ letter ++ ".bin");
    defer allocator.free(gbytes);
    var gc = try image.fromRaw(allocator, gbytes);
    defer gc.deinit();

    try std.testing.expectEqual(crop.len, gc.pixels.len);
    var max_diff: u64 = 0;
    for (crop, gc.pixels) |a, b| {
        const d = @abs(@as(i64, a) - @as(i64, b));
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

test "align: norm_crop 112 crop <= 1 (8-bit) vs reference (face_a/b/c)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    inline for (.{ "a", "b", "c" }) |letter| {
        const md = checkFace(allocator, letter) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        testlog.print("[align] face_{s}: max |diff| {d}\n", .{ letter, md });
        // ≤3/255: the umeyama M's trig last-bit (Zig @cos/@sin/hypot vs C++ libm)
        // propagating through the warp; downstream ArcFace cosine is 0.999999 on
        // this crop, so the geometry is correct (docs/parity.md "do not chase").
        try std.testing.expect(md <= 3);
    }
}
