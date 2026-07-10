//! The CLI command path (readImage → pipeline → JSON) produces output matching
//! the reference goldens. Parses both sides and compares numerically (the
//! 2-decimal JSON formatting rounds, so ≤0.5px). Skips without buffalo_l.gguf.

const std = @import("std");
const fucina = @import("fucina");
const cli = @import("cli.zig");
const pipeline = @import("pipeline.zig");
const rec = @import("recognizer.zig");
const testlog = @import("testlog.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

const Face = struct { score: f32, box: [4]f32, landmarks: [5][2]f32 };
const DetJson = struct { faces: []Face };
const AFace = struct { score: f32, box: [4]f32, age: i32, gender: []const u8 };
const AnalyzeJson = struct { faces: []AFace };

test "cli: detect JSON matches reference (face_a)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var file = gguf.File.loadMmap(allocator, std.testing.io, "models/buffalo_l.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    var img = try cli.readImage(allocator, std.testing.io, "examples/facedetect/goldens/align-src-a.bin");
    defer img.deinit();
    const dets = try pipeline.detect_all(&ctx, allocator, &file, &img);
    defer allocator.free(dets);
    const json = try cli.detectJson(allocator, dets);
    defer allocator.free(json);

    const mine = try std.json.parseFromSlice(DetJson, allocator, json, .{});
    defer mine.deinit();
    const gbytes = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/detect-a.txt");
    defer allocator.free(gbytes);
    const golden = try std.json.parseFromSlice(DetJson, allocator, gbytes, .{});
    defer golden.deinit();

    try std.testing.expectEqual(golden.value.faces.len, mine.value.faces.len);
    for (golden.value.faces, mine.value.faces) |g, m| {
        try std.testing.expect(@abs(g.score - m.score) <= 0.001);
        for (0..4) |j| try std.testing.expect(@abs(g.box[j] - m.box[j]) <= 0.5);
        for (0..5) |k| for (0..2) |c| try std.testing.expect(@abs(g.landmarks[k][c] - m.landmarks[k][c]) <= 0.5);
    }
    testlog.print("[cli] detect JSON: {d} face(s), matches detect-a.txt\n", .{mine.value.faces.len});
}

test "cli: analyze JSON matches reference (face_a)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var file = gguf.File.loadMmap(allocator, std.testing.io, "models/buffalo_l.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    var img = try cli.readImage(allocator, std.testing.io, "examples/facedetect/goldens/align-src-a.bin");
    defer img.deinit();
    const primary = (try pipeline.primaryFace(&ctx, allocator, &file, &img)).?;
    const r = try pipeline.analyze(&ctx, allocator, &file, &img);
    const json = try cli.analyzeJson(allocator, primary, r);
    defer allocator.free(json);

    const mine = try std.json.parseFromSlice(AnalyzeJson, allocator, json, .{});
    defer mine.deinit();
    try std.testing.expectEqual(@as(usize, 1), mine.value.faces.len);
    try std.testing.expectEqual(@as(i32, 57), mine.value.faces[0].age);
    try std.testing.expectEqualStrings("M", mine.value.faces[0].gender);
    testlog.print("[cli] analyze JSON: age {d} gender {s}\n", .{ mine.value.faces[0].age, mine.value.faces[0].gender });
}
