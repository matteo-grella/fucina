//! Dense landmarks: 2d106det (2D 106-pt) + 1k3d68 (3D 68-pt),
//! from the separate `landmarks-2d106-1k3d68.gguf`. Both are interpreter-driven,
//! so they reuse graph.zig. Geometry is read from the `facedetect.landmark.{2d,3d}.*`
//! KVs. Decode is crop-space (insightface): reshape(-1,dim), take the LAST
//! num_points rows, then `(p+1)·half` with `half = input_size/2`.

const std = @import("std");
const fucina = @import("fucina");
const rec = @import("recognizer.zig");
const image = @import("image.zig");
const graph = @import("graph.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

pub const Point = struct { x: f32, y: f32, z: f32 };

fn keyBuf(buf: []u8, kd: []const u8, name: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "facedetect.landmark.{s}.{s}", .{ kd, name }) catch unreachable;
}

pub fn landmarks(ctx: *ExecContext, allocator: std.mem.Allocator, io: std.Io, file: *const gguf.File, crop_path: []const u8, three_d: bool) ![]Point {
    const kd: []const u8 = if (three_d) "3d" else "2d";
    var kb: [96]u8 = undefined;
    const num_points: usize = @intCast(file.getInt(keyBuf(&kb, kd, "num_points")) orelse return error.NoLandmarkKV);
    const dim: usize = @intCast(file.getInt(keyBuf(&kb, kd, "dim")) orelse return error.NoLandmarkKV);
    const input_size: usize = @intCast(file.getInt(keyBuf(&kb, kd, "input_size")) orelse return error.NoLandmarkKV);
    const mean: f32 = @floatCast(file.getFloat(keyBuf(&kb, kd, "input_mean")) orelse 0);
    const std_dev: f32 = @floatCast(file.getFloat(keyBuf(&kb, kd, "input_std")) orelse 1);
    const input_name = file.getString(keyBuf(&kb, kd, "input")) orelse return error.NoLandmarkKV;
    const out_name = file.getString(keyBuf(&kb, kd, "output")) orelse return error.NoLandmarkKV;
    const arr = file.getArray(keyBuf(&kb, kd, "graph")) orelse return error.NoLandmarkGraph;
    const specs = try arr.stringSlices(allocator);
    defer allocator.free(specs);
    const prefix: []const u8 = if (three_d) "l3d." else "l2d.";

    // Crop → input [h,w,c] RGB (raw; both heads normalize in-graph), (px−mean)/std.
    const bytes = try rec.readFile(io, allocator, crop_path);
    defer allocator.free(bytes);
    var img = try image.fromRaw(allocator, bytes);
    defer img.deinit();
    const buf = try allocator.alloc(f32, img.width * img.height * 3);
    defer allocator.free(buf);
    const inv = 1.0 / std_dev;
    for (img.pixels, 0..) |px, i| buf[i] = (@as(f32, @floatFromInt(px)) - mean) * inv;
    var input = try ctx.fromSlice(&.{ img.height, img.width, 3 }, buf);
    defer input.deinit();

    var compiled = try graph.Compiled.compile(ctx, allocator, file, prefix, specs, out_name, input_name, 1e-5);
    defer compiled.deinit();
    var raw = try compiled.run(ctx, allocator, &input);
    defer raw.deinit();
    const rd = raw.dataConst();
    if (rd.len % dim != 0) return error.BadLandmarkOutput;
    const rows = rd.len / dim;
    if (rows < num_points) return error.FewLandmarkRows;
    const base = rows - num_points;
    const half: f32 = @floatFromInt(input_size / 2);

    const out = try allocator.alloc(Point, num_points);
    for (0..num_points) |i| {
        const p = rd[(base + i) * dim ..];
        out[i] = .{
            .x = (p[0] + 1.0) * half,
            .y = (p[1] + 1.0) * half,
            .z = if (dim == 3) p[2] * half else 0.0,
        };
    }
    return out;
}
