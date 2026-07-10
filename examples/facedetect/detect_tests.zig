//! SCRFD forward (on the reference 640² blob) → decode + NMS → source-pixel
//! detections, compared to the `detect` goldens (box + 5 landmarks ≤ 1px, score
//! match, face count exact). Uses the dumped det_scale. Skips without
//! buffalo_l.gguf.

const std = @import("std");
const fucina = @import("fucina");
const scrfd = @import("scrfd.zig");
const detect = @import("detect.zig");
const rec = @import("recognizer.zig");
const testlog = @import("testlog.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

const Face = struct { score: f32, box: [4]f32, landmarks: [5][2]f32 };
const Golden = struct { faces: []Face };

fn checkFace(allocator: std.mem.Allocator, comptime letter: []const u8) !f32 {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = gguf.File.loadMmap(allocator, std.testing.io, "models/buffalo_l.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    var heads = try scrfd.forward(&ctx, allocator, std.testing.io, &file, "examples/facedetect/goldens/det-blob-" ++ letter ++ ".bin");
    defer heads.deinit(allocator);

    const sbytes = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/det-scale-" ++ letter ++ ".txt");
    defer allocator.free(sbytes);
    const det_scale = try std.fmt.parseFloat(f32, std.mem.trim(u8, sbytes, " \n\r\t"));

    const st: f32 = @floatCast(file.getFloat("facedetect.detector.score_thresh") orelse 0.5);
    const nt: f32 = @floatCast(file.getFloat("facedetect.detector.nms_thresh") orelse 0.4);
    const dets = try detect.decode(allocator, &heads, det_scale, st, nt);
    defer allocator.free(dets);

    const gbytes = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/detect-" ++ letter ++ ".txt");
    defer allocator.free(gbytes);
    const parsed = try std.json.parseFromSlice(Golden, allocator, gbytes, .{});
    defer parsed.deinit();

    try std.testing.expectEqual(parsed.value.faces.len, dets.len); // face count exact
    var max_err: f32 = 0;
    for (parsed.value.faces, 0..) |gf, i| {
        const d = dets[i];
        try std.testing.expect(@abs(d.score - gf.score) <= 0.01);
        for (0..4) |j| max_err = @max(max_err, @abs(d.box[j] - gf.box[j]));
        for (0..5) |k| for (0..2) |c| {
            max_err = @max(max_err, @abs(d.kps[k][c] - gf.landmarks[k][c]));
        };
    }
    return max_err;
}

test "detect: SCRFD decode + NMS boxes/landmarks <= 1px vs detect golden (face_a/b/c)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    inline for (.{ "a", "b", "c" }) |letter| {
        const err = checkFace(allocator, letter) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        testlog.print("[detect] face_{s}: max box/kps err {d:.5}px\n", .{ letter, err });
        try std.testing.expect(err <= 1.0);
    }
}
