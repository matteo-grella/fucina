//! Feeds the reference's aligned 96² genderage crop (goldens/ga-crop-X.bin)
//! through the Fucina genderage forward and checks gender (exact) + age
//! (within tolerance) vs the `analyze` goldens.
//! Skips when models/buffalo_l.gguf is absent.

const std = @import("std");
const fucina = @import("fucina");
const genderage = @import("genderage.zig");
const testlog = @import("testlog.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

const Case = struct { letter: []const u8, gender: u8, age: i32 };
const cases = [_]Case{
    .{ .letter = "a", .gender = 'M', .age = 57 },
    .{ .letter = "b", .gender = 'M', .age = 72 },
    .{ .letter = "c", .gender = 'M', .age = 77 },
};

test "genderage: forward gender exact + age tol vs analyze golden (face_a/b/c)" {
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

        const r = try genderage.analyze(&ctx, allocator, std.testing.io, &file, "examples/facedetect/goldens/ga-crop-" ++ cs.letter ++ ".bin");
        testlog.print("[genderage] face_{s}: gender {c} age {d} (golden {c}/{d})\n", .{ cs.letter, r.gender, r.age, cs.gender, cs.age });
        try std.testing.expectEqual(cs.gender, r.gender);
        try std.testing.expect(@abs(r.age - cs.age) <= 1);
    }
}
