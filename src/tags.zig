//! The comptime axis-tag algebra: tag tuple types, membership/position
//! lookups, and tag-set operations. Comptime-only data — no runtime code;
//! the op library APPLYING tags to raw tensors is `tag_ops.zig`, and the
//! public tagged tensor facade is `ag.zig`.
const std = @import("std");
const tensor_mod = @import("tensor.zig");

pub const Tag = @TypeOf(.tag);
pub const inserted_axis = std.math.maxInt(usize);

const DotOrderPart = enum { batch, left_free, right_free, contract };

pub fn tagsEqual(comptime a: anytype, comptime b: anytype) bool {
    if (a.len != b.len) return false;
    inline for (a, 0..) |tag, i| {
        if (comptime !tagEqual(tag, b[i])) return false;
    }
    return true;
}

pub fn reduceAxesDescending(comptime tags: anytype, comptime reduce_tags: anytype) [reduce_tags.len]usize {
    var axes: [reduce_tags.len]usize = undefined;
    inline for (reduce_tags, 0..) |tag, i| {
        axes[i] = tagIndexOrCompileError(tags, tag);
    }

    comptime var i: usize = 0;
    inline while (i < axes.len) : (i += 1) {
        comptime var j: usize = i + 1;
        inline while (j < axes.len) : (j += 1) {
            if (axes[j] > axes[i]) {
                const tmp = axes[i];
                axes[i] = axes[j];
                axes[j] = tmp;
            }
        }
    }
    return axes;
}

pub fn dotLeftOrder(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) [left_tags.len]Tag {
    return dotOrder(left_tags.len, left_tags, right_tags, contract_tag, [_]DotOrderPart{ .batch, .left_free, .contract });
}

pub fn dotRightOrder(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) [right_tags.len]Tag {
    return dotOrder(right_tags.len, left_tags, right_tags, contract_tag, [_]DotOrderPart{ .batch, .contract, .right_free });
}

pub fn dotRightTransBOrder(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) [right_tags.len]Tag {
    return dotOrder(right_tags.len, left_tags, right_tags, contract_tag, [_]DotOrderPart{ .batch, .right_free, .contract });
}

fn dotOrder(
    comptime out_len: usize,
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    comptime contract_tag: Tag,
    comptime parts: anytype,
) [out_len]Tag {
    var out: [out_len]Tag = undefined;
    var out_i: usize = 0;
    inline for (parts) |part| {
        switch (part) {
            .batch => inline for (dotBatchTags(left_tags, right_tags, contract_tag)) |tag| {
                out[out_i] = tag;
                out_i += 1;
            },
            .left_free => inline for (dotLeftFreeTags(left_tags, right_tags, contract_tag)) |tag| {
                out[out_i] = tag;
                out_i += 1;
            },
            .right_free => inline for (dotRightFreeTags(left_tags, right_tags, contract_tag)) |tag| {
                out[out_i] = tag;
                out_i += 1;
            },
            .contract => {
                out[out_i] = contract_tag;
                out_i += 1;
            },
        }
    }
    return out;
}

pub fn normalizeTags(comptime tags_spec: anytype) [tagSpecLen(tags_spec)]Tag {
    if (comptime isTensorSpec(tags_spec)) {
        const Spec = @TypeOf(tags_spec);
        if (@hasField(Spec, "tags")) return normalizeTags(tags_spec.tags);
        if (@hasField(Spec, "rank")) return autoTags(rankFromSpec(tags_spec.rank));
        @compileError("tensor dtype specs must include tags or rank");
    }
    if (comptime isRankSpec(tags_spec)) return autoTags(rankFromSpec(tags_spec));

    var out: [tagSpecLen(tags_spec)]Tag = undefined;
    inline for (0..out.len) |i| out[i] = tags_spec[i];
    return out;
}

pub fn tagSpecLen(comptime tags_spec: anytype) usize {
    if (comptime isTensorSpec(tags_spec)) {
        const Spec = @TypeOf(tags_spec);
        if (@hasField(Spec, "tags")) return tagSpecLen(tags_spec.tags);
        if (@hasField(Spec, "rank")) return rankFromSpec(tags_spec.rank);
        @compileError("tensor dtype specs must include tags or rank");
    }
    if (comptime isRankSpec(tags_spec)) return rankFromSpec(tags_spec);
    return tags_spec.len;
}

pub fn dtypeFromSpec(comptime tags_spec: anytype) tensor_mod.DType {
    if (comptime isTensorSpec(tags_spec)) {
        const Spec = @TypeOf(tags_spec);
        if (@hasField(Spec, "dtype")) return @as(tensor_mod.DType, tags_spec.dtype);
    }
    return .f32;
}

pub fn isTensorSpec(comptime tags_spec: anytype) bool {
    const Spec = @TypeOf(tags_spec);
    return switch (@typeInfo(Spec)) {
        .@"struct" => |info| !info.is_tuple and (@hasField(Spec, "dtype") or @hasField(Spec, "tags") or @hasField(Spec, "rank")),
        else => false,
    };
}

pub fn isRankSpec(comptime tags_spec: anytype) bool {
    return switch (@typeInfo(@TypeOf(tags_spec))) {
        .comptime_int, .int => true,
        else => false,
    };
}

pub fn rankFromSpec(comptime rank_spec: anytype) usize {
    switch (@typeInfo(@TypeOf(rank_spec))) {
        .comptime_int => {
            if (rank_spec < 0) @compileError("tensor rank must be non-negative");
        },
        .int => |info| {
            if (info.signedness == .signed and rank_spec < 0) @compileError("tensor rank must be non-negative");
        },
        else => @compileError("tensor tags must be a tag tuple or a comptime rank"),
    }

    const rank: usize = @intCast(rank_spec);
    if (rank > tensor_mod.max_rank) @compileError(tensor_mod.too_many_tags_msg);
    return rank;
}

pub fn autoTags(comptime rank: usize) [rank]Tag {
    if (rank > tensor_mod.max_rank) @compileError(tensor_mod.too_many_tags_msg);
    const generated = .{ ._0, ._1, ._2, ._3, ._4, ._5, ._6, ._7 };
    if (tensor_mod.max_rank != generated.len) @compileError("autoTags generated tag count must match tensor.max_rank");
    var out: [rank]Tag = undefined;
    inline for (0..rank) |i| out[i] = generated[i];
    return out;
}

pub fn validateUniqueTags(comptime tags: anytype) void {
    inline for (0..tags.len) |i| {
        inline for ((i + 1)..tags.len) |j| {
            if (comptime tagEqual(tags[i], tags[j])) @compileError("duplicate tensor tag");
        }
    }
}

pub fn validateSameTagSet(comptime source_tags: anytype, comptime target_tags: anytype) void {
    validateUniqueTags(target_tags);
    if (source_tags.len != target_tags.len) @compileError("permutation requires the same rank");
    inline for (source_tags) |tag| if (comptime tagIndex(target_tags, tag) == null) @compileError("permutation target must contain the same tags");
}

fn validateUniqueMergeOutput(comptime tags: anytype, comptime out_tag: Tag, comptime merge_tags: anytype) void {
    if (comptime tagIndex(tags, out_tag)) |_| {
        if (comptime tagIndex(merge_tags, out_tag) == null) @compileError("merge output tag already exists");
    }
}

pub fn tagIndex(comptime tags: anytype, comptime tag: anytype) ?usize {
    inline for (tags, 0..) |candidate, i| {
        if (comptime tagEqual(candidate, tag)) return i;
    }
    return null;
}

pub fn tagIndexOrCompileError(comptime tags: anytype, comptime tag: anytype) usize {
    inline for (tags, 0..) |candidate, i| if (comptime tagEqual(candidate, tag)) return i;
    @compileError("tensor tag not found");
}

pub fn tagEqual(comptime a: anytype, comptime b: anytype) bool {
    return comptime blk: {
        const a_name = @tagName(a);
        const b_name = @tagName(b);
        if (a_name.len != b_name.len) break :blk false;
        var i: usize = 0;
        while (i < a_name.len) : (i += 1) {
            if (a_name[i] != b_name[i]) break :blk false;
        }
        break :blk true;
    };
}

pub fn rawRank(comptime tag_count: usize) usize {
    return if (tag_count == 0) 1 else tag_count;
}

pub fn identityAxes(comptime rank: usize) [rank]usize {
    var out: [rank]usize = undefined;
    inline for (0..rank) |i| out[i] = i;
    return out;
}

pub fn alignAxes(comptime source_tags: anytype, comptime target_tags: anytype) [target_tags.len]usize {
    validateUniqueTags(target_tags);
    inline for (source_tags) |tag| if (comptime tagIndex(target_tags, tag) == null) @compileError("target tags must include all source tags");
    var out: [target_tags.len]usize = undefined;
    inline for (target_tags, 0..) |tag, i| out[i] = tagIndex(source_tags, tag) orelse inserted_axis;
    return out;
}

pub fn insertAxes(comptime rank: usize, comptime axis_index: usize) [rank + 1]usize {
    if (axis_index > rank) @compileError("insert axis out of bounds");
    var out: [rank + 1]usize = undefined;
    inline for (0..out.len) |i| {
        out[i] = if (i == axis_index) inserted_axis else if (i < axis_index) i else i - 1;
    }
    return out;
}

pub fn squeezeAxes(comptime rank: usize, comptime axis_index: usize) [rank - 1]usize {
    var out: [rank - 1]usize = undefined;
    inline for (0..out.len) |i| out[i] = if (i < axis_index) i else i + 1;
    return out;
}

pub inline fn removeTag(comptime tags: anytype, comptime tag: Tag) []const Tag {
    comptime {
        _ = tagIndexOrCompileError(tags, tag);
        var out: [tags.len]Tag = undefined;
        var out_i: usize = 0;
        for (tags) |candidate| {
            if (!tagEqual(candidate, tag)) {
                out[out_i] = candidate;
                out_i += 1;
            }
        }
        const final = out[0..out_i].*;
        return &final;
    }
}

pub inline fn removeTags(comptime tags: anytype, comptime remove_tags: anytype) []const Tag {
    comptime {
        validateUniqueTags(remove_tags);
        for (remove_tags) |tag| _ = tagIndexOrCompileError(tags, tag);
        var out: [tags.len]Tag = undefined;
        var out_i: usize = 0;
        for (tags) |tag| {
            if (tagIndex(remove_tags, tag) == null) {
                out[out_i] = tag;
                out_i += 1;
            }
        }
        const final = out[0..out_i].*;
        return &final;
    }
}

/// Set union: `tags` followed by the tags of `other` not already present.
/// Unlike `pointwiseResultTags` there is NO rank cap — the result is a
/// membership set, not a tensor tag tuple (an einsum's operand-tag union may
/// legally exceed `max_rank` even though every tensor involved fits it).
pub inline fn unionTags(comptime tags: anytype, comptime other: anytype) []const Tag {
    comptime {
        @setEvalBranchQuota(100_000);
        var out: [tags.len + other.len]Tag = undefined;
        var out_i: usize = 0;
        for (tags) |tag| {
            out[out_i] = tag;
            out_i += 1;
        }
        for (other) |tag| {
            if (tagIndex(tags, tag) == null) {
                out[out_i] = tag;
                out_i += 1;
            }
        }
        const final = out[0..out_i].*;
        return &final;
    }
}

/// Tags of `tags` that are also present in `other`, in `tags` order.
pub inline fn intersectTags(comptime tags: anytype, comptime other: anytype) []const Tag {
    comptime {
        @setEvalBranchQuota(100_000);
        var out: [tags.len]Tag = undefined;
        var out_i: usize = 0;
        for (tags) |tag| {
            if (tagIndex(other, tag) != null) {
                out[out_i] = tag;
                out_i += 1;
            }
        }
        const final = out[0..out_i].*;
        return &final;
    }
}

pub fn replaceTag(comptime tags: anytype, comptime old_tag: Tag, comptime new_tag: Tag) [tags.len]Tag {
    _ = tagIndexOrCompileError(tags, old_tag);
    if (comptime tagIndex(tags, new_tag)) |i| {
        if (comptime !tagEqual(tags[i], old_tag)) @compileError("replacement tensor tag already exists");
    }

    var out: [tags.len]Tag = undefined;
    inline for (tags, 0..) |tag, i| {
        out[i] = if (comptime tagEqual(tag, old_tag)) new_tag else tag;
    }
    return out;
}

pub inline fn pointwiseResultTags(comptime left_tags: anytype, comptime right_tags: anytype) []const Tag {
    comptime {
        var out: [left_tags.len + right_tags.len]Tag = undefined;
        var out_i: usize = 0;
        for (left_tags) |tag| {
            out[out_i] = tag;
            out_i += 1;
        }
        for (right_tags) |tag| {
            if (tagIndex(left_tags, tag) == null) {
                out[out_i] = tag;
                out_i += 1;
            }
        }
        if (out_i > tensor_mod.max_rank) @compileError(tensor_mod.too_many_tags_msg);
        const final = out[0..out_i].*;
        return &final;
    }
}

pub inline fn dotResultTags(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) []const Tag {
    comptime {
        _ = tagIndexOrCompileError(left_tags, contract_tag);
        _ = tagIndexOrCompileError(right_tags, contract_tag);
        const batch = dotBatchTags(left_tags, right_tags, contract_tag);
        const left_free = dotLeftFreeTags(left_tags, right_tags, contract_tag);
        const right_free = dotRightFreeTags(left_tags, right_tags, contract_tag);
        if (batch.len + left_free.len + right_free.len > tensor_mod.max_rank) @compileError(tensor_mod.too_many_tags_msg);
        var out: [batch.len + left_free.len + right_free.len]Tag = undefined;
        var out_i: usize = 0;
        for (batch) |tag| {
            out[out_i] = tag;
            out_i += 1;
        }
        for (left_free) |tag| {
            out[out_i] = tag;
            out_i += 1;
        }
        for (right_free) |tag| {
            out[out_i] = tag;
            out_i += 1;
        }
        const final = out;
        return &final;
    }
}

pub inline fn dotBatchTags(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) []const Tag {
    comptime {
        var out: [left_tags.len]Tag = undefined;
        var out_i: usize = 0;
        for (left_tags) |tag| {
            if (!tagEqual(tag, contract_tag) and tagIndex(right_tags, tag) != null) {
                out[out_i] = tag;
                out_i += 1;
            }
        }
        const final = out[0..out_i].*;
        return &final;
    }
}

pub inline fn dotLeftFreeTags(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) []const Tag {
    comptime {
        var out: [left_tags.len]Tag = undefined;
        var out_i: usize = 0;
        for (left_tags) |tag| {
            if (!tagEqual(tag, contract_tag) and tagIndex(right_tags, tag) == null) {
                out[out_i] = tag;
                out_i += 1;
            }
        }
        const final = out[0..out_i].*;
        return &final;
    }
}

pub inline fn dotRightFreeTags(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) []const Tag {
    comptime {
        @setEvalBranchQuota(100_000);
        _ = tagIndexOrCompileError(right_tags, contract_tag);
        var out: [right_tags.len]Tag = undefined;
        var out_i: usize = 0;
        for (right_tags) |tag| {
            if (!tagEqual(tag, contract_tag) and tagIndex(left_tags, tag) == null) {
                out[out_i] = tag;
                out_i += 1;
            }
        }
        const final = out[0..out_i].*;
        return &final;
    }
}

pub fn insertTagAt(comptime tags: anytype, comptime tag: Tag, comptime axis_index: usize) [tags.len + 1]Tag {
    if (tags.len + 1 > tensor_mod.max_rank) @compileError(tensor_mod.too_many_tags_msg);
    if (axis_index > tags.len) @compileError("insert axis out of bounds");
    if (comptime tagIndex(tags, tag) != null) @compileError("inserted tensor tag already exists");
    var out: [tags.len + 1]Tag = undefined;
    inline for (0..out.len) |i| out[i] = if (i < axis_index) tags[i] else if (i == axis_index) tag else tags[i - 1];
    return out;
}

/// Role of one operand axis in a multi-index contraction (einsum), derived
/// from tag membership alone: shared tags are `batch` (kept in the output) or
/// `contract` (summed); operand-private tags are `*_free` (kept) or
/// `*_summed` (summed away before the contraction).
pub const EinsumPart = enum { batch, contract, left_free, right_free, left_summed, right_summed };

pub fn einsumClassOfLeft(comptime right_tags: anytype, comptime out_tags: anytype, comptime tag: Tag) EinsumPart {
    const shared = tagIndex(right_tags, tag) != null;
    const kept = tagIndex(out_tags, tag) != null;
    if (shared) return if (kept) .batch else .contract;
    return if (kept) .left_free else .left_summed;
}

pub fn einsumClassOfRight(comptime left_tags: anytype, comptime out_tags: anytype, comptime tag: Tag) EinsumPart {
    const shared = tagIndex(left_tags, tag) != null;
    const kept = tagIndex(out_tags, tag) != null;
    if (shared) return if (kept) .batch else .contract;
    return if (kept) .right_free else .right_summed;
}

/// Tags of one einsum part in the owning operand's axis order; the shared
/// parts (batch, contract) are reported in LEFT-operand order, matching the
/// `dot*` convention that the right operand aligns to the left.
pub inline fn einsumPartTags(comptime left_tags: anytype, comptime right_tags: anytype, comptime out_tags: anytype, comptime part: EinsumPart) []const Tag {
    comptime {
        @setEvalBranchQuota(100_000);
        var out: [left_tags.len + right_tags.len]Tag = undefined;
        var out_i: usize = 0;
        switch (part) {
            .batch, .contract, .left_free, .left_summed => for (left_tags) |tag| {
                if (einsumClassOfLeft(right_tags, out_tags, tag) == part) {
                    out[out_i] = tag;
                    out_i += 1;
                }
            },
            .right_free, .right_summed => for (right_tags) |tag| {
                if (einsumClassOfRight(left_tags, out_tags, tag) == part) {
                    out[out_i] = tag;
                    out_i += 1;
                }
            },
        }
        const final = out[0..out_i].*;
        return &final;
    }
}

/// Validates an einsum equation at comptime: unique output tags, every output
/// tag present in at least one operand, representable ranks.
pub fn einsumValidate(comptime left_tags: anytype, comptime right_tags: anytype, comptime out_tags: anytype) void {
    comptime {
        validateUniqueTags(left_tags);
        validateUniqueTags(right_tags);
        validateUniqueTags(out_tags);
        if (out_tags.len > tensor_mod.max_rank) @compileError(tensor_mod.too_many_tags_msg);
        for (out_tags) |tag| {
            if (tagIndex(left_tags, tag) == null and tagIndex(right_tags, tag) == null)
                @compileError("einsum output tag not found in any operand");
        }
    }
}

pub fn splitTags(comptime tags: anytype, comptime tag: Tag, comptime split_tags: anytype) [tags.len + split_tags.len - 1]Tag {
    const axis_index = tagIndexOrCompileError(tags, tag);
    if (split_tags.len == 0) @compileError("split requires at least one output tag");
    if (tags.len + split_tags.len - 1 > tensor_mod.max_rank) @compileError(tensor_mod.too_many_tags_msg);
    validateUniqueTags(split_tags);
    inline for (split_tags) |split_tag| {
        if (comptime (!tagEqual(split_tag, tag) and tagIndex(tags, split_tag) != null)) @compileError("split output tag already exists");
    }
    var out: [tags.len + split_tags.len - 1]Tag = undefined;
    var out_i: usize = 0;
    inline for (tags, 0..) |candidate, i| {
        if (i == axis_index) {
            inline for (split_tags) |split_tag| {
                out[out_i] = split_tag;
                out_i += 1;
            }
        } else {
            out[out_i] = candidate;
            out_i += 1;
        }
    }
    return out;
}

pub fn mergeTags(comptime tags: anytype, comptime out_tag: Tag, comptime merge_tags: anytype) [tags.len - merge_tags.len + 1]Tag {
    const start = mergeStartAxis(tags, merge_tags);
    validateUniqueMergeOutput(tags, out_tag, merge_tags);
    var out: [tags.len - merge_tags.len + 1]Tag = undefined;
    var out_i: usize = 0;
    inline for (tags, 0..) |tag, i| {
        if (i == start) {
            out[out_i] = out_tag;
            out_i += 1;
        } else if (i < start or i >= start + merge_tags.len) {
            out[out_i] = tag;
            out_i += 1;
        }
    }
    return out;
}

pub fn mergeStartAxis(comptime tags: anytype, comptime merge_tags: anytype) usize {
    if (merge_tags.len == 0) @compileError("merge requires at least one input tag");
    validateUniqueTags(merge_tags);
    const start = tagIndexOrCompileError(tags, merge_tags[0]);
    if (start + merge_tags.len > tags.len) @compileError("merge tags must be contiguous");
    inline for (merge_tags, 0..) |tag, i| if (comptime !tagEqual(tags[start + i], tag)) @compileError("merge tags must be contiguous and in tensor order");
    return start;
}

test {
    _ = @import("tags_tests.zig");
}
