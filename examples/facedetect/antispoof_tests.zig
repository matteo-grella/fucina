//! Replays the two MiniFASNet member graphs on the reference 80² crops
//! (goldens/as-crop-X-{0,1}.bin) and checks the averaged real-prob vs the
//! dumped reference (goldens/as-realprob-X.txt). Skips without buffalo_l.gguf.

const std = @import("std");
const fucina = @import("fucina");
const antispoof = @import("antispoof.zig");
const testlog = @import("testlog.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

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
