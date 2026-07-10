//! Behavioral tests for the tag-spec library (`tags.zig`): normalization of
//! explicit dtype/tags specs and dtype/rank specs into axis tags.
const std = @import("std");
const tags_mod = @import("tags.zig");

const normalizeTags = tags_mod.normalizeTags;
const dtypeFromSpec = tags_mod.dtypeFromSpec;

test "tensor specs normalize explicit dtype and tags" {
    const tags = normalizeTags(.{ .dtype = .u16, .tags = .{ .batch, .seq } });
    try std.testing.expectEqual(@as(usize, 2), tags.len);
    try std.testing.expect(tags[0] == .batch);
    try std.testing.expect(tags[1] == .seq);
    try std.testing.expect(dtypeFromSpec(.{ .dtype = .u16, .tags = .{ .batch, .seq } }) == .u16);
}

test "tensor specs normalize dtype rank to automatic axis tags" {
    const tags = normalizeTags(.{ .dtype = .i64, .rank = 2 });
    try std.testing.expectEqual(@as(usize, 2), tags.len);
    try std.testing.expect(tags[0] == ._0);
    try std.testing.expect(tags[1] == ._1);
    try std.testing.expect(dtypeFromSpec(.{ .dtype = .i64, .rank = 2 }) == .i64);
}

test "einsum tag algebra classifies batch/contract/free/summed parts" {
    comptime {
        const l = .{ .b, .i, .k, .s };
        const r = .{ .b, .k, .j, .t };
        const out = .{ .b, .i, .j };

        const batch = tags_mod.einsumPartTags(l, r, out, .batch);
        if (batch.len != 1 or batch[0] != .b) @compileError("unexpected batch part");

        const contract = tags_mod.einsumPartTags(l, r, out, .contract);
        if (contract.len != 1 or contract[0] != .k) @compileError("unexpected contract part");

        const lf = tags_mod.einsumPartTags(l, r, out, .left_free);
        if (lf.len != 1 or lf[0] != .i) @compileError("unexpected left free part");

        const rf = tags_mod.einsumPartTags(l, r, out, .right_free);
        if (rf.len != 1 or rf[0] != .j) @compileError("unexpected right free part");

        const ls = tags_mod.einsumPartTags(l, r, out, .left_summed);
        if (ls.len != 1 or ls[0] != .s) @compileError("unexpected left summed part");

        const rs = tags_mod.einsumPartTags(l, r, out, .right_summed);
        if (rs.len != 1 or rs[0] != .t) @compileError("unexpected right summed part");
    }
}

test "einsum tag algebra reports shared parts in left order and multi-tag groups" {
    comptime {
        const l = .{ .k2, .k1, .m };
        const r = .{ .k1, .n, .k2 };
        const out = .{ .m, .n };

        const contract = tags_mod.einsumPartTags(l, r, out, .contract);
        if (contract.len != 2 or contract[0] != .k2 or contract[1] != .k1) @compileError("contract part must follow left order");

        if (tags_mod.einsumPartLen(l, r, out, .batch) != 0) @compileError("unexpected batch part");
        if (tags_mod.einsumPartLen(l, r, out, .left_summed) != 0) @compileError("unexpected left summed part");
    }
}

test "intersectTags keeps first-operand order" {
    comptime {
        const kept = tags_mod.intersectTags(.{ .a, .b, .c, .d }, .{ .d, .b });
        if (kept.len != 2 or kept[0] != .b or kept[1] != .d) @compileError("unexpected intersection");
        if (tags_mod.intersectTagsLen(.{ .a, .b }, .{ .c, .d }) != 0) @compileError("expected empty intersection");
    }
}
