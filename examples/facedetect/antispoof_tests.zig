//! Replays the two MiniFASNet member graphs on the reference 80² crops
//! (goldens/as-crop-X-{0,1}.bin) and checks the averaged real-prob vs the
//! dumped reference (goldens/as-realprob-X.txt); then gates the production
//! path — crop-from-bbox vs the reference cropper's dumps, the argmax-gated
//! ensemble score, and the verify liveness veto vs the reference CLI output.
//! Skips without buffalo_l.gguf.

const std = @import("std");
const fucina = @import("fucina");
const antispoof = @import("antispoof.zig");
const align_mod = @import("align.zig");
const pipeline = @import("pipeline.zig");
const scrfd = @import("scrfd.zig");
const rec = @import("recognizer.zig");
const image = @import("image.zig");
const testlog = @import("testlog.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

fn loadFixture(allocator: std.mem.Allocator, path: []const u8) !image.Image {
    const bytes = try rec.readFile(std.testing.io, allocator, path);
    defer allocator.free(bytes);
    return image.fromRaw(allocator, bytes);
}

const Case = struct { letter: []const u8, real_prob: f32 };
const cases = [_]Case{
    .{ .letter = "a", .real_prob = 0.999802 },
    .{ .letter = "b", .real_prob = 0.999766 },
    .{ .letter = "c", .real_prob = 0.979237 },
};

test "antispoof: MiniFASNet replay real_prob vs reference (face_a/b/c)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    inline for (cases) |cs| {
        var ctx: ExecContext = undefined;
        ctx.init(allocator);
        defer ctx.deinit();

        var file = gguf.File.loadMmap(allocator, std.testing.io, "models/buffalo_l.gguf") catch |err| switch (err) {
            error.FileNotFound => return error.SkipZigTest,
            else => return err,
        };
        defer file.deinit();

        const crops = [_][]const u8{
            "examples/facedetect/goldens/as-crop-" ++ cs.letter ++ "-0.bin",
            "examples/facedetect/goldens/as-crop-" ++ cs.letter ++ "-1.bin",
        };
        const rp = try antispoof.realProb(&ctx, allocator, std.testing.io, &file, &crops);
        testlog.print("[antispoof] face_{s}: real_prob {d:.6} (golden {d:.6})\n", .{ cs.letter, rp, cs.real_prob });
        try std.testing.expect(@abs(rp - cs.real_prob) <= 1e-3);
    }
}

test "antispoof: crop-from-bbox matches the reference dumps; argmax-gated score (face_a)" {
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

    var img = try loadFixture(allocator, "examples/facedetect/goldens/align-src-a.bin");
    defer img.deinit();

    var det_model = scrfd.Model.init(allocator, &file);
    defer det_model.deinit();
    const primary = (try pipeline.primaryFaceWith(&ctx, allocator, &det_model, &img)).?;

    try std.testing.expect(antispoof.present(&file));
    var as_model = try antispoof.Model.init(&ctx, allocator, &file);
    defer as_model.deinit();
    try std.testing.expectEqual(@as(usize, 2), as_model.members.len);
    try std.testing.expectEqual(@as(usize, 80), as_model.input_size);

    // The bbox-driven crops must reproduce the reference cropper's dumps.
    const crop_goldens = [_][]const u8{
        "examples/facedetect/goldens/as-crop-a-0.bin",
        "examples/facedetect/goldens/as-crop-a-1.bin",
    };
    for (as_model.members, crop_goldens) |*m, gpath| {
        const crop = try align_mod.antispoofCrop(allocator, &img, primary.box, as_model.input_size, m.scale);
        defer allocator.free(crop);
        var golden = try loadFixture(allocator, gpath);
        defer golden.deinit();
        try std.testing.expectEqualSlices(u8, golden.pixels, crop);
    }

    const score = try antispoof.scoreWith(&ctx, allocator, &as_model, &img, primary.box);
    testlog.print("[antispoof] face_a: score {d:.6}\n", .{score});
    try std.testing.expect(@abs(score - 0.999802) <= 1e-3);
    try std.testing.expect(score >= 0.5);
}

test "antispoof: verify a-b liveness veto keeps the reference verdict" {
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

    var ia = try loadFixture(allocator, "examples/facedetect/goldens/align-src-a.bin");
    defer ia.deinit();
    var ib = try loadFixture(allocator, "examples/facedetect/goldens/align-src-b.bin");
    defer ib.deinit();

    var det_model = scrfd.Model.init(allocator, &file);
    defer det_model.deinit();
    var rec_model = try rec.Model.load(&ctx, allocator, &file);
    defer rec_model.deinit();

    const ea = try pipeline.embedWith(&ctx, allocator, &det_model, &rec_model, &ia);
    defer allocator.free(ea);
    const eb = try pipeline.embedWith(&ctx, allocator, &det_model, &rec_model, &ib);
    defer allocator.free(eb);
    const dist = 1.0 - pipeline.cosine(ea, eb);
    var verified = dist <= 0.35;

    var as_model = try antispoof.Model.init(&ctx, allocator, &file);
    defer as_model.deinit();
    if (!(try pipeline.liveWith(&ctx, allocator, &det_model, &as_model, &ia)) or
        !(try pipeline.liveWith(&ctx, allocator, &det_model, &as_model, &ib))) verified = false;

    // Golden gate per goldens/README: verdict exact, distance near-parity
    // (embeddings agree at cosine >= 0.999999, so the 4th decimal can round
    // differently from the reference CLI dump).
    const golden = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/verify-ab-antispoof.txt");
    defer allocator.free(golden);
    const gt = std.mem.trimEnd(u8, golden, "\n");
    const dprefix = "{\"distance\":";
    try std.testing.expect(std.mem.startsWith(u8, gt, dprefix));
    const comma = std.mem.indexOfScalar(u8, gt, ',').?;
    const gdist = try std.fmt.parseFloat(f64, gt[dprefix.len..comma]);
    const gverified = std.mem.indexOf(u8, gt, "\"verified\":true") != null;
    testlog.print("[antispoof] verify a-b: dist {d:.4} (golden {d:.4}), verified {}\n", .{ dist, gdist, verified });
    try std.testing.expect(@abs(dist - gdist) <= 1e-3);
    try std.testing.expectEqual(gverified, verified);
}
