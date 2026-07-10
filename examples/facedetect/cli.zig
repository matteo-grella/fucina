//! CLI command logic + reference-matching JSON formatting, factored out of the
//! thin `main()` so it is unit-testable. Reads FDR1 (reference-pixel) or PNG
//! input; JPEG is not supported.

const std = @import("std");
const image = @import("image.zig");
const detect = @import("detect.zig");
const genderage = @import("genderage.zig");
const rec = @import("recognizer.zig");
const pipeline = @import("pipeline.zig");

/// Read an input image: FDR1 raw (reference pixels) or PNG, by magic bytes.
pub fn readImage(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !image.Image {
    const bytes = try rec.readFile(io, allocator, path);
    defer allocator.free(bytes);
    if (bytes.len >= 4 and std.mem.eql(u8, bytes[0..4], "FDR1")) return image.fromRaw(allocator, bytes);
    return image.decodePng(allocator, bytes);
}

fn appendFmt(out: *std.ArrayList(u8), allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    const s = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(s);
    try out.appendSlice(allocator, s);
}

/// `{"faces":[{"score":..,"box":[..],"landmarks":[[..],..]}]}` — matches the ref.
pub fn detectJson(allocator: std.mem.Allocator, dets: []const detect.Detection) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"faces\":[");
    for (dets, 0..) |d, i| {
        if (i > 0) try out.append(allocator, ',');
        try appendFmt(&out, allocator, "{{\"score\":{d:.4},\"box\":[{d:.2},{d:.2},{d:.2},{d:.2}],\"landmarks\":[", .{ d.score, d.box[0], d.box[1], d.box[2], d.box[3] });
        for (0..5) |k| {
            if (k > 0) try out.append(allocator, ',');
            try appendFmt(&out, allocator, "[{d:.2},{d:.2}]", .{ d.kps[k][0], d.kps[k][1] });
        }
        try out.appendSlice(allocator, "]}");
    }
    try out.appendSlice(allocator, "]}");
    return out.toOwnedSlice(allocator);
}

/// `{"dim":512,"embedding":[..]}` — L2-normalized (insightface normed_embedding).
pub fn embedJson(allocator: std.mem.Allocator, emb: []f32) ![]u8 {
    pipeline.l2normalize(emb);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendFmt(&out, allocator, "{{\"dim\":{d},\"embedding\":[", .{emb.len});
    for (emb, 0..) |v, i| {
        if (i > 0) try out.append(allocator, ',');
        try appendFmt(&out, allocator, "{d:.6}", .{v});
    }
    try out.appendSlice(allocator, "]}");
    return out.toOwnedSlice(allocator);
}

/// `{"faces":[{"score":..,"box":[..],"age":..,"gender":".."}]}`.
pub fn analyzeJson(allocator: std.mem.Allocator, d: detect.Detection, r: genderage.Result) ![]u8 {
    return std.fmt.allocPrint(allocator, "{{\"faces\":[{{\"score\":{d:.4},\"box\":[{d:.2},{d:.2},{d:.2},{d:.2}],\"age\":{d},\"gender\":\"{c}\"}}]}}", .{ d.score, d.box[0], d.box[1], d.box[2], d.box[3], r.age, r.gender });
}

/// `{"distance":..,"verified":..}`.
pub fn verifyJson(allocator: std.mem.Allocator, distance: f64, verified: bool) ![]u8 {
    return std.fmt.allocPrint(allocator, "{{\"distance\":{d:.4},\"verified\":{}}}", .{ distance, verified });
}
