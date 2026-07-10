//! The full production embed path end-to-end (source image → letterbox →
//! detect → align → ArcFace) vs the reference embedding, and the verify
//! verdict on a same-face pair. Production tier: cosine ≥ 0.9999.

const std = @import("std");
const fucina = @import("fucina");
const pipeline = @import("pipeline.zig");
const image = @import("image.zig");
const rec = @import("recognizer.zig");
const testlog = @import("testlog.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

const Golden = struct { dim: usize, embedding: []f32 };

fn embedFace(ctx: *ExecContext, al: std.mem.Allocator, file: *const gguf.File, comptime letter: []const u8) ![]f32 {
    const sbytes = try rec.readFile(std.testing.io, al, "examples/facedetect/goldens/align-src-" ++ letter ++ ".bin");
    defer al.free(sbytes);
    var src = try image.fromRaw(al, sbytes);
    defer src.deinit();
    return pipeline.embed(ctx, al, file, &src);
}

test "embed: end-to-end production cosine >= 0.9999 vs reference (face_a/b/c)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    inline for (.{ "a", "b", "c" }) |letter| {
        var ctx: ExecContext = undefined;
        ctx.init(allocator);
        defer ctx.deinit();
        var file = gguf.File.loadMmap(allocator, std.testing.io, "models/buffalo_l.gguf") catch |err| switch (err) {
            error.FileNotFound => return error.SkipZigTest,
            else => return err,
        };
        defer file.deinit();

        const emb = try embedFace(&ctx, allocator, &file, letter);
        defer allocator.free(emb);

        const gbytes = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/embed-" ++ letter ++ ".json");
        defer allocator.free(gbytes);
        const parsed = try std.json.parseFromSlice(Golden, allocator, gbytes, .{});
        defer parsed.deinit();

        const cos = pipeline.cosine(emb, parsed.value.embedding);
        testlog.print("[embed] end-to-end face_{s}: cosine {d:.6}\n", .{ letter, cos });
        try std.testing.expect(cos >= 0.9999);
    }
}

const ACase = struct { letter: []const u8, gender: u8, age: i32 };
const acases = [_]ACase{
    .{ .letter = "a", .gender = 'M', .age = 57 },
    .{ .letter = "b", .gender = 'M', .age = 72 },
    .{ .letter = "c", .gender = 'M', .age = 77 },
};

test "analyze: end-to-end gender exact + age vs reference (face_a/b/c)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    inline for (acases) |cs| {
        var ctx: ExecContext = undefined;
        ctx.init(allocator);
        defer ctx.deinit();
        var file = gguf.File.loadMmap(allocator, std.testing.io, "models/buffalo_l.gguf") catch |err| switch (err) {
            error.FileNotFound => return error.SkipZigTest,
            else => return err,
        };
        defer file.deinit();

        const sbytes = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/align-src-" ++ cs.letter ++ ".bin");
        defer allocator.free(sbytes);
        var src = try image.fromRaw(allocator, sbytes);
        defer src.deinit();

        const r = try pipeline.analyze(&ctx, allocator, &file, &src);
        testlog.print("[analyze] end-to-end face_{s}: gender {c} age {d} (golden {c}/{d})\n", .{ cs.letter, r.gender, r.age, cs.gender, cs.age });
        try std.testing.expectEqual(cs.gender, r.gender);
        try std.testing.expect(@abs(r.age - cs.age) <= 1);
    }
}
