//! Token stream -> labeled boxes -> detections JSON.
//!
//! parse_boxes mirrors refs/locate-anything.cpp/src/boxes.cpp: coordinate
//! tokens denormalize as (id - coord_start)/1000 * target dimension, the
//! <ref>..</ref> label carries forward across consecutive boxes. The JSON
//! writer reproduces the reference CLI's exact format (%.2f fields) so
//! outputs byte-compare.

const std = @import("std");
const mtp = @import("mtp.zig");

const Allocator = std.mem.Allocator;

pub const Box = struct {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    label: []u8,

    pub fn deinit(self: *Box, allocator: Allocator) void {
        allocator.free(self.label);
        self.* = undefined;
    }
};

pub fn freeBoxes(allocator: Allocator, boxes: []Box) void {
    for (boxes) |*b| b.deinit(allocator);
    allocator.free(boxes);
}

/// `decodeLabel` maps a label token-id span to owned text (byte-level decode).
pub fn parseBoxes(
    allocator: Allocator,
    t: mtp.TokenIds,
    ids: []const u32,
    img_w: usize,
    img_h: usize,
    decoder: anytype,
) ![]Box {
    var out: std.ArrayList(Box) = .empty;
    errdefer {
        for (out.items) |*b| b.deinit(allocator);
        out.deinit(allocator);
    }

    var cur_label: []u8 = try allocator.alloc(u8, 0);
    defer allocator.free(cur_label);

    var i: usize = 0;
    while (i < ids.len) {
        const tok = ids[i];
        if (tok == t.ref_start) {
            var j = i + 1;
            const label_start = j;
            while (j < ids.len and ids[j] != t.ref_end) j += 1;
            const label = try decoder.decode(allocator, ids[label_start..j]);
            allocator.free(cur_label);
            cur_label = label;
            i = j + 1;
            continue;
        }
        if (tok == t.box_start) {
            var j = i + 1;
            var coords: [4]u32 = undefined;
            var n: usize = 0;
            while (j < ids.len and ids[j] != t.box_end) : (j += 1) {
                if (ids[j] >= t.coord_start and ids[j] <= t.coord_end and n < 4) {
                    coords[n] = ids[j] - t.coord_start;
                    n += 1;
                }
            }
            if (n == 4) {
                try out.append(allocator, .{
                    .x1 = @as(f32, @floatFromInt(coords[0])) / 1000.0 * @as(f32, @floatFromInt(img_w)),
                    .y1 = @as(f32, @floatFromInt(coords[1])) / 1000.0 * @as(f32, @floatFromInt(img_h)),
                    .x2 = @as(f32, @floatFromInt(coords[2])) / 1000.0 * @as(f32, @floatFromInt(img_w)),
                    .y2 = @as(f32, @floatFromInt(coords[3])) / 1000.0 * @as(f32, @floatFromInt(img_h)),
                    .label = try allocator.dupe(u8, cur_label),
                });
            }
            i = j + 1;
            continue;
        }
        i += 1;
    }
    return out.toOwnedSlice(allocator);
}

/// The reference CLI's detections JSON, byte-compatible:
/// {"detections":[{"label":"cat","box":[3.14,50.62,221.76,443.52]},...]}
pub fn writeJson(allocator: Allocator, boxes: []const Box) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"detections\":[");
    for (boxes, 0..) |b, i| {
        if (i > 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, "{\"label\":\"");
        for (b.label) |c| {
            switch (c) {
                '"' => try out.appendSlice(allocator, "\\\""),
                '\\' => try out.appendSlice(allocator, "\\\\"),
                else => try out.append(allocator, c),
            }
        }
        try out.appendSlice(allocator, "\",\"box\":[");
        var buf: [64]u8 = undefined;
        inline for ([_]f32{ 0, 0, 0, 0 }, 0..) |_, ci| {
            const v = switch (ci) {
                0 => b.x1,
                1 => b.y1,
                2 => b.x2,
                3 => b.y2,
                else => unreachable,
            };
            if (ci > 0) try out.append(allocator, ',');
            try out.appendSlice(allocator, try std.fmt.bufPrint(&buf, "{d:.2}", .{v}));
        }
        try out.appendSlice(allocator, "]}");
    }
    try out.appendSlice(allocator, "]}");
    return out.toOwnedSlice(allocator);
}

test {
    _ = @import("boxes_tests.zig");
}
