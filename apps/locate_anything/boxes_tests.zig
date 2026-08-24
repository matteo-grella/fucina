const std = @import("std");
const boxes = @import("boxes.zig");
const mtp = @import("mtp.zig");

const testing = std.testing;

const test_ids = mtp.TokenIds{
    .box_start = 100,
    .box_end = 101,
    .coord_start = 200,
    .coord_end = 1200,
    .ref_start = 102,
    .ref_end = 103,
    .none = 50,
    .null_tok = 1301,
    .text_mask = 104,
    .im_end = 99,
};

const FakeDecoder = struct {
    pub fn decode(_: @This(), allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (ids, 0..) |id, i| {
            if (i > 0) try out.append(allocator, '-');
            try out.print(allocator, "{d}", .{id});
        }
        return out.toOwnedSlice(allocator);
    }
};

test "parseBoxes: label carries forward, coords denormalize" {
    const allocator = testing.allocator;
    // <ref>7</ref> <box><200+100><200+200><200+500><200+900></box> <box>...</box>
    const stream = [_]u32{ 102, 7, 103, 100, 300, 400, 700, 1100, 101, 100, 200, 200, 400, 400, 101 };
    const out = try boxes.parseBoxes(allocator, test_ids, &stream, 1000, 500, FakeDecoder{});
    defer boxes.freeBoxes(allocator, out);

    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("7", out[0].label);
    try testing.expectEqualStrings("7", out[1].label); // carried forward
    try testing.expectApproxEqAbs(@as(f32, 100.0), out[0].x1, 1e-5); // 100/1000*1000
    try testing.expectApproxEqAbs(@as(f32, 100.0), out[0].y1, 1e-5); // 200/1000*500
    try testing.expectApproxEqAbs(@as(f32, 500.0), out[0].x2, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 450.0), out[0].y2, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), out[1].x1, 1e-5);
}

test "parseBoxes: boxes without 4 coords are dropped" {
    const allocator = testing.allocator;
    const stream = [_]u32{ 100, 300, 400, 101 }; // only 2 coords
    const out = try boxes.parseBoxes(allocator, test_ids, &stream, 100, 100, FakeDecoder{});
    defer boxes.freeBoxes(allocator, out);
    try testing.expectEqual(@as(usize, 0), out.len);
}

test "writeJson matches the reference %.2f format" {
    const allocator = testing.allocator;
    const label = try allocator.dupe(u8, "cat");
    const bs = [_]boxes.Box{.{ .x1 = 3.136, .y1 = 50.624, .x2 = 221.76, .y2 = 443.52, .label = label }};
    const json = try boxes.writeJson(allocator, &bs);
    defer allocator.free(json);
    allocator.free(label);
    try testing.expectEqualStrings(
        "{\"detections\":[{\"label\":\"cat\",\"box\":[3.14,50.62,221.76,443.52]}]}",
        json,
    );
}

test "writeJson: empty detections" {
    const json = try boxes.writeJson(testing.allocator, &.{});
    defer testing.allocator.free(json);
    try testing.expectEqualStrings("{\"detections\":[]}", json);
}
