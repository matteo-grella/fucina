//! Feeds the reference's aligned 112² crop (goldens/crop-X.bin, captured via
//! the dump hook) through the Fucina IResNet-50 forward and checks cosine vs
//! the reference embedding (goldens/embed-X.json).
//! Skips when models/buffalo_l.gguf is absent (as loader_tests does).

const std = @import("std");
const fucina = @import("fucina");
const recognizer = @import("recognizer.zig");
const testlog = @import("testlog.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

fn cosine(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * @as(f64, y);
        na += @as(f64, x) * @as(f64, x);
        nb += @as(f64, y) * @as(f64, y);
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}

const Golden = struct { dim: usize, embedding: []f32 };

fn checkFace(allocator: std.mem.Allocator, comptime letter: []const u8) !f64 {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = gguf.File.loadMmap(allocator, std.testing.io, "models/buffalo_l.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    const gbytes = try recognizer.readFile(std.testing.io, allocator, "examples/facedetect/goldens/embed-" ++ letter ++ ".json");
    defer allocator.free(gbytes);
    const parsed = try std.json.parseFromSlice(Golden, allocator, gbytes, .{});
    defer parsed.deinit();

    const emb = try recognizer.embed(&ctx, allocator, std.testing.io, &file, "examples/facedetect/goldens/crop-" ++ letter ++ ".bin");
    defer allocator.free(emb);
    try std.testing.expectEqual(@as(usize, 512), emb.len);
    try std.testing.expectEqual(@as(usize, 512), parsed.value.embedding.len);

    const cos = cosine(emb, parsed.value.embedding);
    testlog.print("[arcface] R50 cosine(face_{s}) = {d:.6}\n", .{ letter, cos });
    return cos;
}

test "arcface: R50 forward cosine >= 0.9999 vs reference embedding (face_a/b/c)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    inline for (.{ "a", "b", "c" }) |letter| {
        const cos = checkFace(allocator, letter) catch |err| switch (err) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return err,
        };
        try std.testing.expect(cos >= 0.9999);
    }
}
