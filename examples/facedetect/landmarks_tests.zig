//! Feeds the reference 192² landmark crops (goldens/lm-crop-X-{2d,3d}.bin)
//! through graph.zig and checks the decoded crop-space points vs the dumped
//! reference (goldens/lm-pts-X-{2d,3d}.txt).
//! Skips without the landmarks model.

const std = @import("std");
const fucina = @import("fucina");
const landmarks = @import("landmarks.zig");
const rec = @import("recognizer.zig");
const testlog = @import("testlog.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

const Case = struct { letter: []const u8, three_d: bool, tag: []const u8 };
const cases = [_]Case{
    .{ .letter = "a", .three_d = false, .tag = "2d" },
    .{ .letter = "a", .three_d = true, .tag = "3d" },
    .{ .letter = "b", .three_d = false, .tag = "2d" },
    .{ .letter = "b", .three_d = true, .tag = "3d" },
    .{ .letter = "c", .three_d = false, .tag = "2d" },
    .{ .letter = "c", .three_d = true, .tag = "3d" },
};

fn checkCase(allocator: std.mem.Allocator, comptime cs: Case) !f32 {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = gguf.File.loadMmap(allocator, std.testing.io, "models/landmarks-2d106-1k3d68.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    const pts = try landmarks.landmarks(&ctx, allocator, std.testing.io, &file, "examples/facedetect/goldens/lm-crop-" ++ cs.letter ++ "-" ++ cs.tag ++ ".bin", cs.three_d);
    defer allocator.free(pts);

    const gbytes = try rec.readFile(std.testing.io, allocator, "examples/facedetect/goldens/lm-pts-" ++ cs.letter ++ "-" ++ cs.tag ++ ".txt");
    defer allocator.free(gbytes);

    var max_err: f32 = 0;
    var it = std.mem.tokenizeScalar(u8, gbytes, '\n');
    var i: usize = 0;
    while (it.next()) |line| : (i += 1) {
        var ft = std.mem.tokenizeScalar(u8, line, ' ');
        const gx = try std.fmt.parseFloat(f32, ft.next() orelse continue);
        const gy = try std.fmt.parseFloat(f32, ft.next() orelse continue);
        const gz = try std.fmt.parseFloat(f32, ft.next() orelse continue);
        max_err = @max(max_err, @abs(pts[i].x - gx));
        max_err = @max(max_err, @abs(pts[i].y - gy));
        max_err = @max(max_err, @abs(pts[i].z - gz));
    }
    return max_err;
}

test "landmarks: crop-space points <= 1.5px vs reference (2d106det / 1k3d68)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    inline for (cases) |cs| {
        const err = checkCase(allocator, cs) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        testlog.print("[landmarks] face_{s} {s}: max px err {d:.4}\n", .{ cs.letter, cs.tag, err });
        try std.testing.expect(err <= 1.5);
    }
}
