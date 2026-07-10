//! Feeds the reference 640² detector blob (goldens/det-blob-X.bin) through the
//! Fucina SCRFD forward and compares the 9 raw heads to the dumped reference
//! (goldens/det-heads-X.bin, stride-major score/bbox/kps). Skips without
//! buffalo_l.gguf.

const std = @import("std");
const fucina = @import("fucina");
const scrfd = @import("scrfd.zig");
const rec = @import("recognizer.zig");
const testlog = @import("testlog.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

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

    const gbytes = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/det-heads-" ++ letter ++ ".bin");
    defer allocator.free(gbytes);

    var cur: usize = 0;
    var max_err: f32 = 0;
    for (0..3) |i| {
        inline for (.{ "score", "bbox", "kps" }) |field| {
            const mine = @field(heads, field)[i];
            const n = std.mem.readInt(u32, gbytes[cur..][0..4], .little);
            cur += 4;
            try std.testing.expectEqual(mine.len, @as(usize, n));
            for (0..n) |k| {
                const gv: f32 = @bitCast(std.mem.readInt(u32, gbytes[cur..][0..4], .little));
                cur += 4;
                max_err = @max(max_err, @abs(mine[k] - gv));
            }
        }
    }
    return max_err;
}

test "scrfd: raw heads vs reference (face_a/b/c)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    inline for (.{ "a", "b", "c" }) |letter| {
        const err = checkFace(allocator, letter) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        testlog.print("[scrfd] face_{s} max head err {d:.6}\n", .{ letter, err });
        try std.testing.expect(err <= 1e-3);
    }
}
