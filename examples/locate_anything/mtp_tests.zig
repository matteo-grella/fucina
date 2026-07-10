const std = @import("std");
const mtp = @import("mtp.zig");

const testing = std.testing;

const test_ids = mtp.TokenIds{
    .box_start = 100,
    .box_end = 101,
    .coord_start = 200,
    .coord_end = 300,
    .ref_start = 102,
    .ref_end = 103,
    .none = 50,
    .null_tok = 301,
    .text_mask = 104,
    .im_end = 99,
};

test "topk: torch semantics — value desc, index asc on ties" {
    const row = [_]f32{ 0.1, 0.5, 0.5, 0.3, 0.05 };
    var ids: [3]usize = undefined;
    var probs: [3]f32 = undefined;
    mtp.topk(&row, &ids, &probs);
    try testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, &ids);
    try testing.expectEqualSlices(f32, &.{ 0.5, 0.5, 0.3 }, &probs);
}

test "softmaxRow sums to 1 with f64 accumulation" {
    const row = [_]f32{ 1.0, 2.0, 3.0, -1.0 };
    const probs = try mtp.softmaxRow(testing.allocator, &row);
    defer testing.allocator.free(probs);
    var sum: f64 = 0;
    for (probs) |p| sum += p;
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-6);
    try testing.expect(probs[2] > probs[1] and probs[1] > probs[0]);
}

test "handlePattern: coord box, point box, error box, fast coercion" {
    const allocator = testing.allocator;

    var coord = try mtp.handlePattern(allocator, test_ids, &.{ 100, 210, 220, 230, 240, 101 }, false);
    defer coord.deinit(allocator);
    try testing.expectEqual(mtp.PatternKind.coord_box, coord.kind);
    try testing.expectEqual(@as(usize, 6), coord.tokens.len);
    try testing.expect(!coord.need_ar);

    var point = try mtp.handlePattern(allocator, test_ids, &.{ 100, 210, 220, 101, 240, 101 }, false);
    defer point.deinit(allocator);
    try testing.expectEqual(mtp.PatternKind.point_box, point.kind);
    try testing.expectEqual(@as(usize, 4), point.tokens.len);

    var err = try mtp.handlePattern(allocator, test_ids, &.{ 100, 210, 220, 55, 240, 101 }, false);
    defer err.deinit(allocator);
    try testing.expectEqual(mtp.PatternKind.error_box, err.kind);
    try testing.expect(err.need_ar);
    try testing.expectEqualSlices(u32, &.{ 100, 210, 220 }, err.tokens);

    var fast_coerced = try mtp.handlePattern(allocator, test_ids, &.{ 100, 210, 220, 55, 240, 101 }, true);
    defer fast_coerced.deinit(allocator);
    try testing.expectEqual(mtp.PatternKind.coord_box, fast_coerced.kind);
    try testing.expect(!fast_coerced.need_ar);
    try testing.expectEqual(@as(usize, 6), fast_coerced.tokens.len);
}

test "handlePattern: im_end, empty box, ref truncation + ref_end dedup" {
    const allocator = testing.allocator;

    var end = try mtp.handlePattern(allocator, test_ids, &.{ 301, 1, 2, 3, 4, 5 }, false);
    defer end.deinit(allocator);
    try testing.expectEqual(mtp.PatternKind.im_end, end.kind);
    try testing.expect(end.terminal);
    try testing.expectEqualSlices(u32, &.{99}, end.tokens);

    var empty = try mtp.handlePattern(allocator, test_ids, &.{ 100, 50, 101, 301, 301, 301 }, false);
    defer empty.deinit(allocator);
    try testing.expectEqual(mtp.PatternKind.empty_box, empty.kind);
    try testing.expectEqualSlices(u32, &.{ 100, 50, 101 }, empty.tokens);

    var ref = try mtp.handlePattern(allocator, test_ids, &.{ 102, 7, 103, 103, 301, 5 }, false);
    defer ref.deinit(allocator);
    try testing.expectEqual(mtp.PatternKind.ref_object, ref.kind);
    try testing.expectEqualSlices(u32, &.{ 102, 7, 103 }, ref.tokens);
}

test "hybrid AR step: box_end returns to MTP, coord stays, other terminates" {
    var st = mtp.HybridState{ .use_mtp = false };
    try testing.expectEqual(mtp.ArKind.coord_ar, mtp.hybridArStep(&st, test_ids, 250));
    try testing.expect(!st.use_mtp and !st.terminated);
    try testing.expectEqual(mtp.ArKind.box_end_ar, mtp.hybridArStep(&st, test_ids, 101));
    try testing.expect(st.use_mtp);
    try testing.expectEqual(mtp.ArKind.im_end, mtp.hybridArStep(&st, test_ids, 1));
    try testing.expect(st.terminated);
}
