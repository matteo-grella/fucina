//! MiniFASNet anti-spoof ensemble — drives the app-level graph.zig
//! replay over the two GGUF-embedded member graphs (as0 = MiniFASNetV2 @ scale
//! 2.7, as1 = V1SE @ scale 4.0). Each 80² BGR crop → replay → 3 logits → softmax;
//! the averaged "real"-class (index 1) probability is the liveness score.

const std = @import("std");
const fucina = @import("fucina");
const rec = @import("recognizer.zig");
const image = @import("image.zig");
const graph = @import("graph.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

const as_bn_eps: f32 = 1e-5; // kAsBnEps

/// Averaged ensemble "real" probability from the two pre-captured member crops
/// (BGR planes, raw 0-255 — the net has no in-graph normalize).
pub fn realProb(ctx: *ExecContext, allocator: std.mem.Allocator, io: std.Io, file: *const gguf.File, crop_paths: []const []const u8) !f32 {
    var accum1: f64 = 0;
    for (crop_paths, 0..) |path, i| {
        const bytes = try rec.readFile(io, allocator, path);
        defer allocator.free(bytes);
        var img = try image.fromRaw(allocator, bytes);
        defer img.deinit();
        const npx = img.width * img.height;
        const buf = try allocator.alloc(f32, npx * 3);
        defer allocator.free(buf);
        for (0..npx) |p| {
            buf[p * 3 + 0] = @floatFromInt(img.pixels[p * 3 + 2]); // B
            buf[p * 3 + 1] = @floatFromInt(img.pixels[p * 3 + 1]); // G
            buf[p * 3 + 2] = @floatFromInt(img.pixels[p * 3 + 0]); // R
        }
        var input = try ctx.fromSlice(&.{ img.height, img.width, 3 }, buf);
        defer input.deinit();

        const gkey = try std.fmt.allocPrint(allocator, "facedetect.antispoof.{d}.graph", .{i});
        defer allocator.free(gkey);
        const okey = try std.fmt.allocPrint(allocator, "facedetect.antispoof.{d}.output", .{i});
        defer allocator.free(okey);
        const prefix = try std.fmt.allocPrint(allocator, "as{d}.", .{i});
        defer allocator.free(prefix);

        const arr = file.getArray(gkey) orelse return error.NoAntispoofGraph;
        const specs = try arr.stringSlices(allocator);
        defer allocator.free(specs);
        const out_name = file.getString(okey) orelse return error.NoAntispoofOutput;

        var compiled = try graph.Compiled.compile(ctx, allocator, file, prefix, specs, out_name, "input", as_bn_eps);
        defer compiled.deinit();
        var logits = try compiled.run(ctx, allocator, &input);
        defer logits.deinit();
        const ld = logits.dataConst();
        std.debug.assert(ld.len == 3);

        const mx = @max(ld[0], @max(ld[1], ld[2]));
        var e: [3]f64 = undefined;
        var s: f64 = 0;
        for (0..3) |k| {
            e[k] = @exp(@as(f64, ld[k]) - mx);
            s += e[k];
        }
        accum1 += e[1] / s;
    }
    return @floatCast(accum1 / @as(f64, @floatFromInt(crop_paths.len)));
}
