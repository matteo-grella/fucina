//! Behavioral tests for the tag-semantics op library (`tagged.zig`): comptime
//! tag rank/axis lookup, tag-directed views (align/permute/broadcast/split/
//! merge/flatten), tag-driven pointwise broadcasting, multi-axis reduction, and
//! `taggedDot` matmul/bmm lowering across physical tag orders.
const std = @import("std");
const tensor_mod = @import("tensor.zig");
const exec_mod = @import("exec.zig");
const tags_mod = @import("tags.zig");
const tagged = @import("tagged.zig");

const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const normalizeTags = tags_mod.normalizeTags;
const tagIndex = tags_mod.tagIndex;
const tagIndexOrCompileError = tags_mod.tagIndexOrCompileError;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const dotResultTags = tags_mod.dotResultTags;
const splitTags = tags_mod.splitTags;
const mergeTags = tags_mod.mergeTags;

const pointwise = tagged.pointwise;
const taggedDot = tagged.taggedDot;
const sumManyTensor = tagged.sumManyTensor;
const splitAxisView = tagged.splitAxisView;
const mergeAxesView = tagged.mergeAxesView;
const flattenTensor = tagged.flattenTensor;
const permuteTensorTo = tagged.permuteTensorTo;
const alignTensorTo = tagged.alignTensorTo;
const broadcastTensorTo = tagged.broadcastTensorTo;

test "tag library: tag tuples expose comptime rank and axis lookup" {
    comptime {
        const tags = normalizeTags(.{ .batch, .d });
        if (tags.len != 2) @compileError("unexpected tag count");
        if (tagIndexOrCompileError(tags, .d) != 1) @compileError("unexpected .d axis");
        if (tagIndex(tags, .batch) == null) @compileError("missing .batch tag");
        if (tagIndex(tags, .channel) != null) @compileError(".channel should be absent");
    }
}

test "tag library: integer rank specs normalize to positional tags and permute as views" {
    comptime {
        const tags = normalizeTags(3);
        if (tags.len != 3 or tags[0] != ._0 or tags[1] != ._1 or tags[2] != ._2) {
            @compileError("unexpected auto tags for integer rank spec");
        }
    }

    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(3, .{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    var permuted = try permuteTensorTo(.{ ._0, ._1, ._2 }, &x, .{ ._2, ._0, ._1 });
    defer permuted.deinit();
    try std.testing.expect(permuted.buffer == x.buffer);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3 }, permuted.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 1, 6, 2 }, permuted.strides.slice());
}

test "tag library: alignTensorTo reorders tags and injects singleton axes as views" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var y = try alignTensorTo(.{ .batch, .d }, &x, .{ .d, .batch, .channel });
    defer y.deinit();

    try std.testing.expect(y.buffer == x.buffer);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 1 }, y.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 0 }, y.strides.slice());

    var copied = [_]f32{0} ** 6;
    try y.copyTo(&copied);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, &copied);
}

test "tag library: pointwise broadcasts operands by tag" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try ctx.fromSliceRank(1, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();

    comptime {
        const result = pointwiseResultTags(.{ .batch, .d }, .{.d});
        if (result.len != 2 or result[0] != .batch or result[1] != .d) @compileError("unexpected pointwise result tags");
    }

    var y = try pointwise(.add, .{ .batch, .d }, &x, &ctx, .{.d}, &bias);
    defer y.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, y.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, y.dataConst());

    comptime {
        const result = pointwiseResultTags(.{.d}, .{ .batch, .d });
        if (result.len != 2 or result[0] != .d or result[1] != .batch) @compileError("unexpected flipped result tags");
    }

    var z = try pointwise(.add, .{.d}, &bias, &ctx, .{ .batch, .d }, &x);
    defer z.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, z.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 11, 14, 22, 25, 33, 36 }, z.dataConst());
}

test "tag library: broadcastTensorTo expands by tag as a zero-stride view" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var bias = try ctx.fromSliceRank(1, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();

    var broadcast = try broadcastTensorTo(.{.d}, &bias, .{ .batch, .d }, .{ 2, 3 });
    defer broadcast.deinit();

    try std.testing.expect(broadcast.buffer == bias.buffer);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, broadcast.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, broadcast.strides.slice());

    var copied = [_]f32{0} ** 6;
    try broadcast.copyTo(&copied);
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30, 10, 20, 30 }, &copied);
}

test "tag library: permuteTensorTo exposes zero-copy tag-ordered views" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var y = try permuteTensorTo(.{ .batch, .d }, &x, .{ .d, .batch });
    defer y.deinit();
    try std.testing.expect(y.buffer == x.buffer);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, y.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, y.strides.slice());

    var copied = [_]f32{0} ** 6;
    try y.copyTo(&copied);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, &copied);
}

test "tag library: pointwise broadcasts scalars without materializing the source" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var scalar = try ctx.fromSliceRank(1, .{1}, &.{2});
    defer scalar.deinit();
    var x = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var y = try pointwise(.mul, .{}, &scalar, &ctx, .{ .batch, .d }, &x);
    defer y.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, y.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6, 8, 10, 12 }, y.dataConst());
}

test "tag library: rejects incompatible broadcast dimensions" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(1, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var y = try ctx.fromSliceRank(1, .{2}, &.{ 10, 20 });
    defer y.deinit();

    try std.testing.expectError(TensorError.ShapeMismatch, pointwise(.add, .{.d}, &x, &ctx, .{.d}, &y));
    try std.testing.expectError(TensorError.ShapeMismatch, broadcastTensorTo(.{.d}, &x, .{.d}, .{2}));
}

test "tag library: splitAxisView and mergeAxesView reshape compatible named axes as views" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 6 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    comptime {
        const split_result = splitTags(.{ .batch, .d_model }, .d_model, .{ .head, .head_dim });
        if (split_result.len != 3 or split_result[0] != .batch or split_result[1] != .head or split_result[2] != .head_dim) {
            @compileError("unexpected split result tags");
        }
        const merge_result = mergeTags(split_result, .features, .{ .head, .head_dim });
        if (merge_result.len != 2 or merge_result[0] != .batch or merge_result[1] != .features) {
            @compileError("unexpected merge result tags");
        }
    }

    var split = try splitAxisView(.{ .batch, .d_model }, &x, .d_model, .{ .head, .head_dim }, .{ 2, 3 });
    defer split.deinit();
    try std.testing.expect(split.buffer == x.buffer);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3 }, split.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 6, 3, 1 }, split.strides.slice());

    var merged = try mergeAxesView(.{ .batch, .head, .head_dim }, &split, .features, .{ .head, .head_dim });
    defer merged.deinit();
    try std.testing.expect(merged.buffer == x.buffer);
    try std.testing.expectEqualSlices(usize, &.{ 2, 6 }, merged.shape.slice());
    try std.testing.expectEqualSlices(f32, x.dataConst(), merged.dataConst());
}

test "tag library: flattenTensor handles contiguous and strided sources" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ -1, 2, -3, 4, -5, 6 });
    defer x.deinit();

    var flat = try flattenTensor(&ctx, &x);
    defer flat.deinit();
    try std.testing.expectEqualSlices(usize, &.{6}, flat.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ -1, 2, -3, 4, -5, 6 }, flat.dataConst());

    var permuted = try permuteTensorTo(.{ .batch, .d }, &x, .{ .d, .batch });
    defer permuted.deinit();
    var flat_strided = try flattenTensor(&ctx, &permuted);
    defer flat_strided.deinit();
    try std.testing.expectEqualSlices(usize, &.{6}, flat_strided.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ -1, 4, 2, -5, -3, 6 }, flat_strided.dataConst());
}

test "tag library: sumManyTensor reduces multiple named axes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(3, .{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    var summed = try sumManyTensor(.{ .batch, .seq, .d }, &x, &ctx, .{ .batch, .seq });
    defer summed.deinit();
    try std.testing.expectEqualSlices(usize, &.{3}, summed.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 22, 26, 30 }, summed.dataConst());
}

test "tag library: dot follows matrix multiplication tag ordering" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var weight = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer weight.deinit();
    var input = try ctx.fromSliceRank(1, .{3}, &.{ 10, 20, 30 });
    defer input.deinit();

    comptime {
        const result = dotResultTags(.{ .out, .d }, .{.d}, .d);
        if (result.len != 1 or result[0] != .out) @compileError("unexpected dot result tags");
    }

    var y = try taggedDot(.{ .out, .d }, &weight, &ctx, .{.d}, &input, .d);
    defer y.deinit();

    try std.testing.expectEqualSlices(usize, &.{2}, y.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 140, 320 }, y.dataConst());
}

test "tag library: dot treats shared non-contracted tags as batch axes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSliceRank(3, .{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer a.deinit();
    var b = try ctx.fromSliceRank(3, .{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer b.deinit();

    comptime {
        const result = dotResultTags(.{ .batch, .m, .k }, .{ .batch, .k, .n }, .k);
        if (result.len != 3 or result[0] != .batch or result[1] != .m or result[2] != .n) {
            @compileError("unexpected batched dot result tags");
        }
    }

    var c = try taggedDot(.{ .batch, .m, .k }, &a, &ctx, .{ .batch, .k, .n }, &b, .k);
    defer c.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, c.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 22, 28, 49, 64, 220, 244, 301, 334 }, c.dataConst());
}

test "tag library: dot contracts by tag regardless of physical axis order" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 7, 9, 11, 8, 10, 12 });
    defer b.deinit();

    var c = try taggedDot(.{ .m, .k }, &a, &ctx, .{ .n, .k }, &b, .k);
    defer c.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, c.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, c.dataConst());
}

test "tag library: dot lowers transposed 2D layouts through matmul variants" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSliceRank(2, .{ 3, 2 }, &.{ 1, 4, 2, 5, 3, 6 });
    defer a.deinit();
    var b = try ctx.fromSliceRank(2, .{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();

    var c = try taggedDot(.{ .k, .m }, &a, &ctx, .{ .k, .n }, &b, .k);
    defer c.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, c.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, c.dataConst());
}

test "tag library: vector dot returns a scalar tensor" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSliceRank(1, .{3}, &.{ 1, 2, 3 });
    defer a.deinit();
    var b = try ctx.fromSliceRank(1, .{3}, &.{ 4, 5, 6 });
    defer b.deinit();

    comptime {
        const result = dotResultTags(.{.d}, .{.d}, .d);
        if (result.len != 0) @compileError("vector dot should produce scalar tags");
    }

    var y = try taggedDot(.{.d}, &a, &ctx, .{.d}, &b, .d);
    defer y.deinit();

    try std.testing.expectEqualSlices(usize, &.{1}, y.shape.slice());
    try std.testing.expectEqual(@as(f32, 32), y.item());
}

test "tag library: dot rejects mismatched contract dimensions" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSliceRank(1, .{2}, &.{ 7, 8 });
    defer b.deinit();

    try std.testing.expectError(TensorError.ShapeMismatch, taggedDot(.{ .out, .d }, &a, &ctx, .{.d}, &b, .d));
}

// ---- taggedEinsum ----

const taggedEinsum = tagged.taggedEinsum;

/// Fills a fresh contiguous tensor with small integers (exact in f32 for any
/// summation order, so the oracle can compare bitwise).
fn einsumTestTensor(ctx: *ExecContext, comptime rank: usize, shape: [rank]usize, salt: u32) !tensor_mod.Tensor {
    var total: usize = 1;
    for (shape) |d| total *= d;
    const data = try std.testing.allocator.alloc(f32, total);
    defer std.testing.allocator.free(data);
    for (data, 0..) |*v, i| {
        const q: u32 = (@as(u32, @intCast(i)) *% 2654435761 +% salt) % 13;
        v.* = @as(f32, @floatFromInt(q)) - 6.0;
    }
    return ctx.fromSliceRank(rank, shape, data);
}

/// Runs taggedEinsum and checks it bitwise against a full index-space
/// summation over the union of both operands' axes.
fn expectEinsumMatchesNaive(
    ctx: *ExecContext,
    comptime l_tags: anytype,
    l: *const tensor_mod.Tensor,
    comptime r_tags: anytype,
    r: *const tensor_mod.Tensor,
    comptime out_tags: anytype,
) !void {
    var got = try taggedEinsum(l_tags, l, ctx, r_tags, r, out_tags);
    defer got.deinit();

    const all = comptime tags_mod.unionTags(l_tags, r_tags);
    var dims: [all.len]usize = undefined;
    inline for (all, 0..) |tag, i| {
        dims[i] = if (comptime tagIndex(l_tags, tag)) |li|
            l.shape.at(li)
        else
            r.shape.at(tagIndexOrCompileError(r_tags, tag));
    }

    var out_shape: [if (out_tags.len == 0) 1 else out_tags.len]usize = undefined;
    if (comptime out_tags.len == 0) {
        out_shape[0] = 1;
    } else {
        inline for (out_tags, 0..) |tag, i| out_shape[i] = dims[comptime tagIndexOrCompileError(all, tag)];
    }
    try std.testing.expectEqualSlices(usize, out_shape[0..], got.shape.slice());

    var total: usize = 1;
    for (out_shape) |d| total *= d;
    const expected = try std.testing.allocator.alloc(f32, total);
    defer std.testing.allocator.free(expected);
    @memset(expected, 0);

    const l_data = l.dataConst();
    const r_data = r.dataConst();
    var space: usize = 1;
    for (dims) |d| space *= d;
    var flat: usize = 0;
    while (flat < space) : (flat += 1) {
        var idx: [all.len]usize = undefined;
        var rem = flat;
        var ax = all.len;
        while (ax > 0) {
            ax -= 1;
            idx[ax] = rem % dims[ax];
            rem /= dims[ax];
        }

        var l_off: usize = 0;
        inline for (l_tags, 0..) |tag, li| {
            l_off = l_off * l.shape.at(li) + idx[comptime tagIndexOrCompileError(all, tag)];
        }
        var r_off: usize = 0;
        inline for (r_tags, 0..) |tag, ri| {
            r_off = r_off * r.shape.at(ri) + idx[comptime tagIndexOrCompileError(all, tag)];
        }
        var o_off: usize = 0;
        inline for (out_tags) |tag| {
            const ai = comptime tagIndexOrCompileError(all, tag);
            o_off = o_off * dims[ai] + idx[ai];
        }
        expected[o_off] += l_data[l_off] * r_data[r_off];
    }

    try std.testing.expectEqualSlices(f32, expected, got.dataConst());
}

test "einsum: matmul-shaped equations across all trans layouts" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try einsumTestTensor(&ctx, 2, .{ 3, 4 }, 1);
    defer a.deinit();
    var b = try einsumTestTensor(&ctx, 2, .{ 4, 5 }, 2);
    defer b.deinit();
    var bt = try einsumTestTensor(&ctx, 2, .{ 5, 4 }, 3);
    defer bt.deinit();
    var at = try einsumTestTensor(&ctx, 2, .{ 4, 3 }, 4);
    defer at.deinit();

    try expectEinsumMatchesNaive(&ctx, .{ .m, .k }, &a, .{ .k, .n }, &b, .{ .m, .n });
    try expectEinsumMatchesNaive(&ctx, .{ .m, .k }, &a, .{ .n, .k }, &bt, .{ .m, .n });
    try expectEinsumMatchesNaive(&ctx, .{ .k, .m }, &at, .{ .k, .n }, &b, .{ .m, .n });
    try expectEinsumMatchesNaive(&ctx, .{ .k, .m }, &at, .{ .n, .k }, &bt, .{ .m, .n });
    try expectEinsumMatchesNaive(&ctx, .{ .k, .m }, &at, .{ .n, .k }, &bt, .{ .n, .m });
    try expectEinsumMatchesNaive(&ctx, .{ .m, .k }, &a, .{ .k, .n }, &b, .{ .n, .m });
}

test "einsum: multi-tag contraction groups lower onto one GEMM" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try einsumTestTensor(&ctx, 3, .{ 3, 2, 4 }, 5);
    defer a.deinit();
    var b = try einsumTestTensor(&ctx, 3, .{ 2, 4, 5 }, 6);
    defer b.deinit();
    var b_swapped = try einsumTestTensor(&ctx, 3, .{ 4, 2, 5 }, 7);
    defer b_swapped.deinit();
    var b_nt = try einsumTestTensor(&ctx, 3, .{ 5, 2, 4 }, 8);
    defer b_nt.deinit();

    // Contract run in matching physical order: direct plain GEMM.
    try expectEinsumMatchesNaive(&ctx, .{ .i, .k1, .k2 }, &a, .{ .k1, .k2, .j }, &b, .{ .i, .j });
    // Contract run order differs between operands: generic path realigns.
    try expectEinsumMatchesNaive(&ctx, .{ .i, .k1, .k2 }, &a, .{ .k2, .k1, .j }, &b_swapped, .{ .i, .j });
    // NT layout with a two-tag contract run.
    try expectEinsumMatchesNaive(&ctx, .{ .i, .k1, .k2 }, &a, .{ .j, .k1, .k2 }, &b_nt, .{ .i, .j });
}

test "einsum: batch plus multi-free groups (grouped-attention shape)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // q[b, g, i, d] x k[b, j, d] -> scores[b, g, i, j]: one equation instead
    // of split/matmul/permute plumbing.
    var q = try einsumTestTensor(&ctx, 4, .{ 2, 2, 3, 4 }, 9);
    defer q.deinit();
    var k = try einsumTestTensor(&ctx, 3, .{ 2, 5, 4 }, 10);
    defer k.deinit();

    try expectEinsumMatchesNaive(&ctx, .{ .b, .g, .i, .d }, &q, .{ .b, .j, .d }, &k, .{ .b, .g, .i, .j });

    // Same with the batch kept in a non-leading output position (interleaved
    // output order pays one materialize but must stay correct).
    try expectEinsumMatchesNaive(&ctx, .{ .b, .g, .i, .d }, &q, .{ .b, .j, .d }, &k, .{ .g, .b, .i, .j });
}

test "einsum: operand-private dropped tags are summed away" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try einsumTestTensor(&ctx, 3, .{ 3, 2, 4 }, 11);
    defer a.deinit();
    var b = try einsumTestTensor(&ctx, 3, .{ 4, 5, 2 }, 12);
    defer b.deinit();

    // .s and .t are private to one operand and absent from the output.
    try expectEinsumMatchesNaive(&ctx, .{ .i, .s, .k }, &a, .{ .k, .j, .t }, &b, .{ .i, .j });
    // Keeping them instead makes them free axes.
    try expectEinsumMatchesNaive(&ctx, .{ .i, .s, .k }, &a, .{ .k, .j, .t }, &b, .{ .i, .s, .j, .t });
}

test "einsum: outer products, elementwise-degenerate, and scalar results" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var u = try einsumTestTensor(&ctx, 1, .{3}, 13);
    defer u.deinit();
    var v = try einsumTestTensor(&ctx, 1, .{4}, 14);
    defer v.deinit();
    var w = try einsumTestTensor(&ctx, 1, .{3}, 15);
    defer w.deinit();
    var m = try einsumTestTensor(&ctx, 2, .{ 3, 4 }, 16);
    defer m.deinit();
    var m2 = try einsumTestTensor(&ctx, 2, .{ 4, 3 }, 17);
    defer m2.deinit();

    // Outer product (no contraction).
    try expectEinsumMatchesNaive(&ctx, .{.i}, &u, .{.j}, &v, .{ .i, .j });
    // Shared kept tag with no contraction: elementwise product.
    try expectEinsumMatchesNaive(&ctx, .{.i}, &u, .{.i}, &w, .{.i});
    // Full contraction to a scalar, including a permuted right operand.
    try expectEinsumMatchesNaive(&ctx, .{.i}, &u, .{.i}, &w, .{});
    try expectEinsumMatchesNaive(&ctx, .{ .i, .j }, &m, .{ .j, .i }, &m2, .{});
}

test "einsum: strided operand views take the generic path and stay exact" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try einsumTestTensor(&ctx, 2, .{ 3, 4 }, 18);
    defer a.deinit();
    var b = try einsumTestTensor(&ctx, 2, .{ 5, 4 }, 19);
    defer b.deinit();

    // A transposed VIEW of b carries tags {.k, .n}: same math as contracting
    // the original with {.n, .k}, so both spellings must agree bitwise.
    var b_view = try alignTensorTo(.{ .n, .k }, &b, .{ .k, .n });
    defer b_view.deinit();
    try std.testing.expect(!b_view.isContiguous());

    var from_view = try taggedEinsum(.{ .m, .k }, &a, &ctx, .{ .k, .n }, &b_view, .{ .m, .n });
    defer from_view.deinit();
    var from_orig = try taggedEinsum(.{ .m, .k }, &a, &ctx, .{ .n, .k }, &b, .{ .m, .n });
    defer from_orig.deinit();

    try std.testing.expectEqualSlices(f32, from_orig.dataConst(), from_view.dataConst());
}

test "einsum: scalar operands scale the other side" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var s = try ctx.fromSliceRank(1, .{1}, &.{2.5});
    defer s.deinit();
    var m = try einsumTestTensor(&ctx, 2, .{ 3, 4 }, 20);
    defer m.deinit();

    try expectEinsumMatchesNaive(&ctx, .{}, &s, .{ .i, .j }, &m, .{ .i, .j });
}

test "einsum: batch, contract, and summed tags combined in one equation" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try einsumTestTensor(&ctx, 4, .{ 2, 3, 2, 4 }, 23);
    defer a.deinit();
    var b = try einsumTestTensor(&ctx, 4, .{ 2, 4, 2, 5 }, 24);
    defer b.deinit();

    // .b batch, .k contracted, .s/.t summed away, .i/.j free — all at once.
    try expectEinsumMatchesNaive(&ctx, .{ .b, .i, .s, .k }, &a, .{ .b, .k, .t, .j }, &b, .{ .b, .i, .j });
}

test "einsum: output may reorder tags within a free group" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try einsumTestTensor(&ctx, 3, .{ 2, 3, 4 }, 25);
    defer a.deinit();
    var b = try einsumTestTensor(&ctx, 2, .{ 4, 5 }, 26);
    defer b.deinit();

    // Left free axes appear in the output in reversed order: still
    // group-nested, so the generic path lands the order directly.
    try expectEinsumMatchesNaive(&ctx, .{ .i1, .i2, .k }, &a, .{ .k, .j }, &b, .{ .i2, .i1, .j });
}

test "einsum: rejects mismatched shared dimensions" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try einsumTestTensor(&ctx, 2, .{ 3, 4 }, 21);
    defer a.deinit();
    var b = try einsumTestTensor(&ctx, 2, .{ 5, 6 }, 22);
    defer b.deinit();

    // Contract dim mismatch.
    try std.testing.expectError(TensorError.ShapeMismatch, taggedEinsum(.{ .m, .k }, &a, &ctx, .{ .k, .n }, &b, .{ .m, .n }));
    // Batch dim mismatch.
    try std.testing.expectError(TensorError.ShapeMismatch, taggedEinsum(.{ .b, .k }, &a, &ctx, .{ .b, .n }, &b, .{ .b, .k, .n }));
}

fn einsumSweepCase(
    ctx: *ExecContext,
    comptime l_tags: anytype,
    l_shape: [l_tags.len]usize,
    comptime r_tags: anytype,
    r_shape: [r_tags.len]usize,
    comptime out_tags: anytype,
    salt: u32,
) !void {
    var a = try einsumTestTensor(ctx, l_tags.len, l_shape, salt);
    defer a.deinit();
    var b = try einsumTestTensor(ctx, r_tags.len, r_shape, salt +% 1);
    defer b.deinit();
    try expectEinsumMatchesNaive(ctx, l_tags, &a, r_tags, &b, out_tags);
}

test "einsum: structure sweep — every axis-role combination against the oracle" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Vector/scalar shapes.
    try einsumSweepCase(&ctx, .{.k}, .{4}, .{.k}, .{4}, .{}, 30);
    try einsumSweepCase(&ctx, .{ .i, .k }, .{ 3, 4 }, .{.k}, .{4}, .{.i}, 31);
    try einsumSweepCase(&ctx, .{.k}, .{4}, .{ .k, .j }, .{ 4, 3 }, .{.j}, 32);
    try einsumSweepCase(&ctx, .{ .s, .k }, .{ 2, 4 }, .{.k}, .{4}, .{}, 33);

    // 2-D layouts, all orientations, both output orders.
    try einsumSweepCase(&ctx, .{ .i, .k }, .{ 3, 4 }, .{ .k, .j }, .{ 4, 5 }, .{ .i, .j }, 34);
    try einsumSweepCase(&ctx, .{ .i, .k }, .{ 3, 4 }, .{ .k, .j }, .{ 4, 5 }, .{ .j, .i }, 35);
    try einsumSweepCase(&ctx, .{ .k, .i }, .{ 4, 3 }, .{ .k, .j }, .{ 4, 5 }, .{ .i, .j }, 36);
    try einsumSweepCase(&ctx, .{ .i, .k }, .{ 3, 4 }, .{ .j, .k }, .{ 5, 4 }, .{ .i, .j }, 37);
    try einsumSweepCase(&ctx, .{ .k, .i }, .{ 4, 3 }, .{ .j, .k }, .{ 5, 4 }, .{ .i, .j }, 38);
    try einsumSweepCase(&ctx, .{ .k, .i }, .{ 4, 3 }, .{ .j, .k }, .{ 5, 4 }, .{ .j, .i }, 39);

    // Batched, all orientations, plus interleaved output.
    try einsumSweepCase(&ctx, .{ .b, .i, .k }, .{ 2, 3, 4 }, .{ .b, .k, .j }, .{ 2, 4, 5 }, .{ .b, .i, .j }, 40);
    try einsumSweepCase(&ctx, .{ .b, .k, .i }, .{ 2, 4, 3 }, .{ .b, .k, .j }, .{ 2, 4, 5 }, .{ .b, .i, .j }, 41);
    try einsumSweepCase(&ctx, .{ .b, .i, .k }, .{ 2, 3, 4 }, .{ .b, .j, .k }, .{ 2, 5, 4 }, .{ .b, .i, .j }, 42);
    try einsumSweepCase(&ctx, .{ .b, .i, .k }, .{ 2, 3, 4 }, .{ .b, .k, .j }, .{ 2, 4, 5 }, .{ .i, .b, .j }, 43);
    try einsumSweepCase(&ctx, .{ .b1, .b2, .i, .k }, .{ 2, 2, 3, 4 }, .{ .b1, .b2, .k, .j }, .{ 2, 2, 4, 5 }, .{ .b1, .b2, .i, .j }, 44);

    // Multi-free / multi-contract groups, matching and mismatched run orders.
    try einsumSweepCase(&ctx, .{ .i1, .i2, .k }, .{ 2, 3, 4 }, .{ .k, .j }, .{ 4, 5 }, .{ .i1, .i2, .j }, 45);
    try einsumSweepCase(&ctx, .{ .i1, .i2, .k }, .{ 2, 3, 4 }, .{ .k, .j }, .{ 4, 5 }, .{ .i2, .i1, .j }, 46);
    try einsumSweepCase(&ctx, .{ .i, .k1, .k2 }, .{ 3, 2, 4 }, .{ .k1, .k2, .j }, .{ 2, 4, 5 }, .{ .i, .j }, 47);
    try einsumSweepCase(&ctx, .{ .i, .k1, .k2 }, .{ 3, 2, 4 }, .{ .k2, .k1, .j }, .{ 4, 2, 5 }, .{ .i, .j }, 48);
    try einsumSweepCase(&ctx, .{ .k1, .i, .k2 }, .{ 2, 3, 4 }, .{ .k1, .k2, .j }, .{ 2, 4, 5 }, .{ .i, .j }, 49);

    // All roles at once; outer products; batch-only elementwise.
    try einsumSweepCase(&ctx, .{ .b, .i, .s, .k }, .{ 2, 3, 2, 4 }, .{ .b, .k, .t, .j }, .{ 2, 4, 2, 5 }, .{ .b, .i, .j }, 50);
    try einsumSweepCase(&ctx, .{.i}, .{3}, .{.j}, .{4}, .{ .i, .j }, 51);
    try einsumSweepCase(&ctx, .{ .b, .i }, .{ 2, 3 }, .{ .b, .j }, .{ 2, 4 }, .{ .b, .i, .j }, 52);
    try einsumSweepCase(&ctx, .{.b}, .{3}, .{.b}, .{3}, .{.b}, 53);

    // Unit dims in every role.
    try einsumSweepCase(&ctx, .{ .b, .i, .k }, .{ 1, 3, 4 }, .{ .b, .k, .j }, .{ 1, 4, 5 }, .{ .b, .i, .j }, 54);
    try einsumSweepCase(&ctx, .{ .b, .i, .k }, .{ 2, 1, 4 }, .{ .b, .k, .j }, .{ 2, 4, 1 }, .{ .b, .i, .j }, 55);
    try einsumSweepCase(&ctx, .{ .i, .k }, .{ 3, 1 }, .{ .k, .j }, .{ 1, 5 }, .{ .i, .j }, 56);
    try einsumSweepCase(&ctx, .{ .s, .k }, .{ 1, 4 }, .{ .k, .j }, .{ 4, 5 }, .{.j}, 57);

    // Max-rank operands and results.
    try einsumSweepCase(
        &ctx,
        .{ .a, .b, .c, .d, .e, .f, .g, .k },
        .{ 2, 1, 2, 1, 2, 1, 2, 3 },
        .{.k},
        .{3},
        .{ .a, .b, .c, .d, .e, .f, .g },
        58,
    );
    // Seven shared batch axes (batch collapses into one bmm axis; this shape
    // has no rank-(batch+2) representation).
    try einsumSweepCase(
        &ctx,
        .{ .b1, .b2, .b3, .b4, .b5, .b6, .b7, .k },
        .{ 2, 1, 2, 1, 2, 1, 2, 3 },
        .{ .b1, .b2, .b3, .b4, .b5, .b6, .b7, .k },
        .{ 2, 1, 2, 1, 2, 1, 2, 3 },
        .{ .b1, .b2, .b3, .b4, .b5, .b6, .b7 },
        66,
    );
    try einsumSweepCase(
        &ctx,
        .{ .a, .b, .c, .d, .k },
        .{ 2, 1, 2, 1, 3 },
        .{ .k, .e, .f, .g, .h },
        .{ 3, 2, 1, 2, 1 },
        .{ .a, .b, .c, .d, .e, .f, .g, .h },
        59,
    );
}

test "einsum: strided-view sweep — permuted operands agree with their contiguous twins" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // A [k, j] view of a [j, k] tensor must behave exactly like a real
    // [k, j] tensor, across orientation classes.
    var a = try einsumTestTensor(&ctx, 2, .{ 3, 4 }, 60);
    defer a.deinit();
    var b = try einsumTestTensor(&ctx, 2, .{ 5, 4 }, 61);
    defer b.deinit();
    var b_kj = try alignTensorTo(.{ .j, .k }, &b, .{ .k, .j });
    defer b_kj.deinit();
    try std.testing.expect(!b_kj.isContiguous());

    var from_view = try taggedEinsum(.{ .i, .k }, &a, &ctx, .{ .k, .j }, &b_kj, .{ .i, .j });
    defer from_view.deinit();
    var from_orig = try taggedEinsum(.{ .i, .k }, &a, &ctx, .{ .j, .k }, &b, .{ .i, .j });
    defer from_orig.deinit();
    try std.testing.expectEqualSlices(f32, from_orig.dataConst(), from_view.dataConst());

    // Batched: both operands strided.
    var q = try einsumTestTensor(&ctx, 3, .{ 2, 3, 4 }, 62);
    defer q.deinit();
    var k = try einsumTestTensor(&ctx, 3, .{ 2, 5, 4 }, 63);
    defer k.deinit();
    var q_perm = try alignTensorTo(.{ .b, .i, .d }, &q, .{ .i, .b, .d });
    defer q_perm.deinit();
    var k_perm = try alignTensorTo(.{ .b, .j, .d }, &k, .{ .d, .b, .j });
    defer k_perm.deinit();

    var v1 = try taggedEinsum(.{ .i, .b, .d }, &q_perm, &ctx, .{ .d, .b, .j }, &k_perm, .{ .b, .i, .j });
    defer v1.deinit();
    var v2 = try taggedEinsum(.{ .b, .i, .d }, &q, &ctx, .{ .b, .j, .d }, &k, .{ .b, .i, .j });
    defer v2.deinit();
    try std.testing.expectEqualSlices(f32, v2.dataConst(), v1.dataConst());
}
