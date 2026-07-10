//! Tag-semantics op library over raw tensors.
//!
//! This module owns the comptime axis-tag algebra applied at runtime: tag-directed
//! views (align/permute/broadcast/split/merge), tag-driven pointwise broadcasting,
//! multi-axis reductions, and `dot` lowering onto the ExecContext matmul/bmm
//! kernels. Functions take comptime tag tuples plus `*const` raw tensors and
//! return *owned* raw tensors; the public autograd facade (`ag/tensor.zig`) and
//! the VJPs (`ag/backward.zig`) re-attach tags at comptime on their side.
//!
//! There is intentionally no tagged tensor *type* here: tags are comptime-only
//! data (`tags.zig`), so the single runtime tensor currency stays the raw
//! tensor (`tensor.zig`), which every heterogeneous container (autograd tape,
//! ExecContext ops, weight unions) is built on.
const std = @import("std");
const tensor_mod = @import("tensor.zig");
const exec_mod = @import("exec.zig");
const tags_mod = @import("tags.zig");

const RawTensor = tensor_mod.Tensor;
const DType = tensor_mod.DType;
const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const GatedOp = exec_mod.GatedOp;
const Tag = tags_mod.Tag;
const normalizeTags = tags_mod.normalizeTags;
const validateUniqueTags = tags_mod.validateUniqueTags;
const validateSameTagSet = tags_mod.validateSameTagSet;
const tagIndex = tags_mod.tagIndex;
const tagIndexOrCompileError = tags_mod.tagIndexOrCompileError;
const tagsEqual = tags_mod.tagsEqual;
const rawRank = tags_mod.rawRank;
const reduceAxesDescending = tags_mod.reduceAxesDescending;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const dotResultTags = tags_mod.dotResultTags;
const dotBatchTags = tags_mod.dotBatchTags;
const splitTags = tags_mod.splitTags;
const mergeTags = tags_mod.mergeTags;
const mergeStartAxis = tags_mod.mergeStartAxis;
const removeTags = tags_mod.removeTags;
const intersectTags = tags_mod.intersectTags;
const einsumPartTags = tags_mod.einsumPartTags;
const einsumValidate = tags_mod.einsumValidate;

pub const PointwiseOp = enum {
    add,
    sub,
    mul,
    div,
    max,
    min,
};

/// Tag-driven broadcasting pointwise op: broadcasts both operands to the
/// pointwise result tags, then dispatches the rank-matched kernel.
pub fn pointwise(
    comptime op: PointwiseOp,
    comptime left_tags: anytype,
    left: *const RawTensor,
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const RawTensor,
) !RawTensor {
    try validateTensorRank(left_tags, left);
    try validateTensorRank(right_tags, right);
    const result_tags = pointwiseResultTags(left_tags, right_tags);
    const result_shape = try broadcastResultShape(result_tags, left_tags, left, right_tags, right);

    var left_view = try broadcastTensorTo(left_tags, left, result_tags, result_shape);
    defer left_view.deinit();
    var right_view = try broadcastTensorTo(right_tags, right, result_tags, result_shape);
    defer right_view.deinit();

    return switch (op) {
        .add => ctx.addRank(rawRank(result_tags.len), &left_view, &right_view),
        .sub => ctx.subRank(rawRank(result_tags.len), &left_view, &right_view),
        .mul => ctx.mulRank(rawRank(result_tags.len), &left_view, &right_view),
        .div => ctx.divRank(rawRank(result_tags.len), &left_view, &right_view),
        .max => ctx.maxRank(rawRank(result_tags.len), &left_view, &right_view),
        .min => ctx.minRank(rawRank(result_tags.len), &left_view, &right_view),
    };
}

/// Tag-driven broadcasting gated op (`glu`/`swiglu`/`geglu`/...).
pub fn gatedPointwise(
    comptime op: GatedOp,
    comptime left_tags: anytype,
    left: *const RawTensor,
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const RawTensor,
) !RawTensor {
    try validateTensorRank(left_tags, left);
    try validateTensorRank(right_tags, right);
    const result_tags = pointwiseResultTags(left_tags, right_tags);
    const result_shape = try broadcastResultShape(result_tags, left_tags, left, right_tags, right);

    var left_view = try broadcastTensorTo(left_tags, left, result_tags, result_shape);
    defer left_view.deinit();
    var right_view = try broadcastTensorTo(right_tags, right, result_tags, result_shape);
    defer right_view.deinit();

    return ctx.gatedRank(rawRank(result_tags.len), op, &left_view, &right_view);
}

/// Tag-directed contraction over one named tag: the single-contract-tag
/// special case of `taggedEinsum`, with the canonical dot result order
/// (batch ++ left free ++ right free). Validates contract/batch dims before
/// computing; kernel selection is the einsum lowering's.
pub fn taggedDot(
    comptime left_tags: anytype,
    left: *const RawTensor,
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const RawTensor,
    comptime contract_tag: Tag,
) !RawTensor {
    return taggedEinsum(left_tags, left, ctx, right_tags, right, comptime dotResultTags(left_tags, right_tags, contract_tag));
}

/// Multi-index tagged contraction (einsum). `out_tags` is the whole equation:
/// `result[out_tags] = Σ over every tag not in out_tags of left ⊙ right`.
/// Shared tags are batch axes when kept and contraction axes when dropped;
/// operand-private tags are free axes when kept and summed away (before the
/// contraction) when dropped. Result axis order is exactly `out_tags`.
///
/// Lowering: both operands align (zero-copy permute views) to an
/// out-derived group-nested order and each side picks its plain or
/// transposed kernel layout at runtime by contiguity — classic layouts
/// dispatch to matmul2D/bmm (and trans variants, with operand swap for the
/// [batch][right free][left free] nesting) with zero copies. At most one of
/// transA/transB per call (both-want-trans materializes the smaller side);
/// a side no orientation can express materializes at most once. Batch axes
/// collapse into one bmm axis, so batch count is unbounded. An `out_tags`
/// order that interleaves the batch/left-free/right-free groups costs one
/// extra output materialization — prefer group-nested output orders.
pub fn taggedEinsum(
    comptime left_tags: anytype,
    left: *const RawTensor,
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const RawTensor,
    comptime out_tags: anytype,
) !RawTensor {
    comptime einsumValidate(left_tags, right_tags, out_tags);
    try validateTensorRank(left_tags, left);
    try validateTensorRank(right_tags, right);
    const result_shape = try einsumResultShapeOf(.f32, .f32, left_tags, left, right_tags, right, out_tags);

    // Operand-private dropped tags are summed away first: cheaper than
    // carrying them through the contraction, and it leaves every remaining
    // axis with a batch/free/contract role.
    const left_summed = comptime einsumPartTags(left_tags, right_tags, out_tags, .left_summed);
    const right_summed = comptime einsumPartTags(left_tags, right_tags, out_tags, .right_summed);
    const l_tags = comptime removeTags(left_tags, left_summed);
    const r_tags = comptime removeTags(right_tags, right_summed);

    var l_val = try sumManyTensor(left_tags, left, ctx, left_summed);
    defer l_val.deinit();
    var r_val = try sumManyTensor(right_tags, right, ctx, right_summed);
    defer r_val.deinit();

    var value = try einsumContract(l_tags, &l_val, ctx, r_tags, &r_val, out_tags);
    errdefer value.deinit();
    if (out_tags.len != 0 and !std.mem.eql(usize, value.shape.slice(), result_shape[0..])) {
        const reshaped = try value.reshape(result_shape[0..]);
        value.deinit();
        value = reshaped;
    }
    return value;
}

/// Sums away `reduce_tags` one axis at a time (innermost-first), returning the
/// reduced tensor tagged `removeTags(tags, reduce_tags)` on the caller's side.
pub fn sumManyTensor(
    comptime tags: anytype,
    source: *const RawTensor,
    ctx: *ExecContext,
    comptime reduce_tags: anytype,
) !RawTensor {
    comptime {
        validateUniqueTags(reduce_tags);
        for (reduce_tags) |tag| _ = tagIndexOrCompileError(tags, tag);
    }
    try validateTensorRank(tags, source);

    if (reduce_tags.len == 0) return source.cloneView();
    if (reduce_tags.len == tags.len) return ctx.sum(source);

    var current = try source.cloneView();
    errdefer current.deinit();

    const axes = comptime reduceAxesDescending(tags, reduce_tags);
    inline for (axes, 0..) |axis, step| {
        const rank_now = comptime tags.len - step;
        const axis_now = comptime axis;
        const next = try ctx.sumAxisRank(rank_now, &current, axis_now);
        current.deinit();
        current = next;
    }
    return current;
}

/// Zero-copy view splitting the `tag` axis into `split_tags` factor axes.
pub fn splitAxisView(
    comptime source_tags: anytype,
    source: *const RawTensor,
    comptime tag: Tag,
    comptime split_tags: anytype,
    split_shape: [split_tags.len]usize,
) !RawTensor {
    const axis_index = tagIndexOrCompileError(source_tags, tag);
    _ = splitTags(source_tags, tag, split_tags);
    try validateTensorRank(source_tags, source);
    const split_count = try elementCountArray(split_tags.len, split_shape);
    if (split_count != source.shape.at(axis_index)) return TensorError.InvalidShape;

    var shape: [source_tags.len + split_tags.len - 1]usize = undefined;
    var strides: [source_tags.len + split_tags.len - 1]usize = undefined;
    var out_i: usize = 0;
    inline for (source_tags, 0..) |_, source_i| {
        if (source_i == axis_index) {
            inline for (0..split_tags.len) |split_i| {
                shape[out_i] = split_shape[split_i];
                strides[out_i] = source.strides.at(axis_index) * productArraySuffix(split_tags.len, split_shape, split_i + 1);
                out_i += 1;
            }
        } else {
            shape[out_i] = source.shape.at(source_i);
            strides[out_i] = source.strides.at(source_i);
            out_i += 1;
        }
    }
    return source.viewWithStrides(shape[0..], strides[0..]);
}

/// Zero-copy view merging adjacent `merge_tags` axes into one axis; requires
/// the merged axes to be stride-compatible (an unsplit layout).
pub fn mergeAxesView(
    comptime source_tags: anytype,
    source: *const RawTensor,
    comptime out_tag: Tag,
    comptime merge_tags: anytype,
) !RawTensor {
    const start = comptime mergeStartAxis(source_tags, merge_tags);
    _ = mergeTags(source_tags, out_tag, merge_tags);
    try validateTensorRank(source_tags, source);

    var merged_dim: usize = 1;
    inline for (0..merge_tags.len) |i| {
        merged_dim = try std.math.mul(usize, merged_dim, source.shape.at(start + i));
    }

    if (merge_tags.len > 1) {
        inline for (start..start + merge_tags.len - 1) |i| {
            const expected = source.shape.at(i + 1) * source.strides.at(i + 1);
            if (source.strides.at(i) != expected) return TensorError.UnsupportedView;
        }
    }

    var shape: [source_tags.len - merge_tags.len + 1]usize = undefined;
    var strides: [source_tags.len - merge_tags.len + 1]usize = undefined;
    var out_i: usize = 0;
    inline for (source_tags, 0..) |_, i| {
        if (i == start) {
            shape[out_i] = merged_dim;
            strides[out_i] = source.strides.at(start + merge_tags.len - 1);
            out_i += 1;
        } else if (i < start or i >= start + merge_tags.len) {
            shape[out_i] = source.shape.at(i);
            strides[out_i] = source.strides.at(i);
            out_i += 1;
        }
    }
    return source.viewWithStrides(shape[0..], strides[0..]);
}

/// Flattens to rank 1, materializing first if the source is non-contiguous.
pub fn flattenTensor(ctx: *ExecContext, source: *const RawTensor) !RawTensor {
    var ready = try contiguousForReshape(ctx, source);
    defer ready.deinit();
    return ready.reshape(&.{source.len()});
}

/// Pure permutation view: `target_tags` must be the same tag set as
/// `source_tags` (checked at comptime).
pub fn permuteTensorTo(
    comptime source_tags: anytype,
    source: *const RawTensor,
    comptime target_tags: anytype,
) !RawTensor {
    comptime validateSameTagSet(source_tags, target_tags);
    return alignTensorTo(source_tags, source, target_tags);
}

pub fn alignTensorTo(comptime source_tags: anytype, source: *const RawTensor, comptime target_tags: anytype) !RawTensor {
    return alignTensorToOf(.f32, source_tags, source, target_tags);
}

/// Reorders axes to `target_tags` order and injects zero-stride singleton axes
/// for target tags absent from the source. Zero-copy.
pub fn alignTensorToOf(
    comptime tensor_dtype: DType,
    comptime source_tags: anytype,
    source: *const tensor_mod.TensorOf(tensor_dtype),
    comptime target_tags: anytype,
) !tensor_mod.TensorOf(tensor_dtype) {
    comptime {
        validateUniqueTags(target_tags);
        if (target_tags.len > tensor_mod.max_rank) @compileError("too many tensor tags");
        for (source_tags) |tag| {
            if (tagIndex(target_tags, tag) == null) @compileError("target tags must include all source tags");
        }
    }
    try validateTensorRankOf(tensor_dtype, source_tags, source);

    if (target_tags.len == 0) {
        return source.cloneView();
    }

    var shape: [target_tags.len]usize = undefined;
    var strides: [target_tags.len]usize = undefined;
    inline for (target_tags, 0..) |target_tag, i| {
        if (tagIndex(source_tags, target_tag)) |source_i| {
            shape[i] = source.shape.at(source_i);
            strides[i] = source.strides.at(source_i);
        } else {
            shape[i] = 1;
            strides[i] = 0;
        }
    }

    return source.viewWithStrides(shape[0..], strides[0..]);
}

pub fn broadcastTensorTo(
    comptime source_tags: anytype,
    source: *const RawTensor,
    comptime target_tags: anytype,
    target_shape: [target_tags.len]usize,
) !RawTensor {
    return broadcastTensorToOf(.f32, source_tags, source, target_tags, target_shape);
}

/// Aligns to `target_tags` order, then broadcasts to `target_shape` with
/// zero-stride expansion. Zero-copy.
pub fn broadcastTensorToOf(
    comptime tensor_dtype: DType,
    comptime source_tags: anytype,
    source: *const tensor_mod.TensorOf(tensor_dtype),
    comptime target_tags: anytype,
    target_shape: [target_tags.len]usize,
) !tensor_mod.TensorOf(tensor_dtype) {
    if (target_tags.len == 0) {
        comptime if (source_tags.len != 0) @compileError("scalar broadcast target cannot drop source tags");
        return source.cloneView();
    }

    var aligned = try alignTensorToOf(tensor_dtype, source_tags, source, target_tags);
    defer aligned.deinit();
    return aligned.broadcastToRank(target_tags.len, target_shape);
}

pub fn pointwiseShape(
    comptime result_tags: anytype,
    comptime left_tags: anytype,
    left: *const RawTensor,
    comptime right_tags: anytype,
    right: *const RawTensor,
) ![rawRank(result_tags.len)]usize {
    return pointwiseShapeOf(.f32, result_tags, left_tags, left, right_tags, right);
}

/// Raw-rank pointwise result shape (scalar results report `{1}`); validates
/// dim-by-dim broadcast compatibility.
pub fn pointwiseShapeOf(
    comptime tensor_dtype: DType,
    comptime result_tags: anytype,
    comptime left_tags: anytype,
    left: *const tensor_mod.TensorOf(tensor_dtype),
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(tensor_dtype),
) ![rawRank(result_tags.len)]usize {
    var shape: [rawRank(result_tags.len)]usize = undefined;
    if (comptime result_tags.len == 0) {
        shape[0] = 1;
        return shape;
    }

    inline for (result_tags, 0..) |tag, i| {
        const left_dim = dimForTagOf(tensor_dtype, left_tags, left, tag);
        const right_dim = dimForTagOf(tensor_dtype, right_tags, right, tag);
        if (left_dim == right_dim) {
            shape[i] = left_dim;
        } else if (left_dim == 1) {
            shape[i] = right_dim;
        } else if (right_dim == 1) {
            shape[i] = left_dim;
        } else {
            return TensorError.ShapeMismatch;
        }
    }
    return shape;
}

/// Raw-rank dot result shape (scalar results report `{1}`); validates the
/// contract dim and any shared batch dims.
pub fn dotResultShapeOf(
    comptime left_dtype: DType,
    comptime right_dtype: DType,
    comptime left_tags: anytype,
    left: *const tensor_mod.TensorOf(left_dtype),
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(right_dtype),
    comptime contract_tag: Tag,
) ![rawRank(dotResultTags(left_tags, right_tags, contract_tag).len)]usize {
    const left_contract_axis = tagIndexOrCompileError(left_tags, contract_tag);
    const right_contract_axis = tagIndexOrCompileError(right_tags, contract_tag);
    if (left.shape.at(left_contract_axis) != right.shape.at(right_contract_axis)) return TensorError.ShapeMismatch;

    inline for (dotBatchTags(left_tags, right_tags, contract_tag)) |tag| {
        const left_axis = tagIndexOrCompileError(left_tags, tag);
        const right_axis = tagIndexOrCompileError(right_tags, tag);
        if (left.shape.at(left_axis) != right.shape.at(right_axis)) return TensorError.ShapeMismatch;
    }

    const result_tags = dotResultTags(left_tags, right_tags, contract_tag);
    var shape: [rawRank(result_tags.len)]usize = undefined;
    if (comptime result_tags.len == 0) {
        shape[0] = 1;
        return shape;
    }

    inline for (result_tags, 0..) |tag, i| {
        shape[i] = if (comptime tagIndex(left_tags, tag)) |left_i|
            left.shape.at(left_i)
        else
            right.shape.at(tagIndexOrCompileError(right_tags, tag));
    }
    return shape;
}

/// Raw-rank einsum result shape (scalar results report `{1}`); validates every
/// shared dim (batch and contract) for equality.
pub fn einsumResultShapeOf(
    comptime left_dtype: DType,
    comptime right_dtype: DType,
    comptime left_tags: anytype,
    left: *const tensor_mod.TensorOf(left_dtype),
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(right_dtype),
    comptime out_tags: anytype,
) ![rawRank(out_tags.len)]usize {
    comptime einsumValidate(left_tags, right_tags, out_tags);
    inline for (left_tags, 0..) |tag, left_axis| {
        if (comptime tagIndex(right_tags, tag)) |right_axis| {
            if (left.shape.at(left_axis) != right.shape.at(right_axis)) return TensorError.ShapeMismatch;
        }
    }

    var shape: [rawRank(out_tags.len)]usize = undefined;
    if (comptime out_tags.len == 0) {
        shape[0] = 1;
        return shape;
    }
    inline for (out_tags, 0..) |tag, i| {
        shape[i] = if (comptime tagIndex(left_tags, tag)) |left_i|
            left.shape.at(left_i)
        else
            right.shape.at(tagIndexOrCompileError(right_tags, tag));
    }
    return shape;
}

pub fn contiguousForReshapeOf(
    comptime tensor_dtype: DType,
    ctx: *ExecContext,
    value: *const tensor_mod.TensorOf(tensor_dtype),
) !tensor_mod.TensorOf(tensor_dtype) {
    if (value.isContiguous()) return value.cloneView();
    return ctx.materializeTyped(tensor_dtype, value);
}

pub fn productRangeOf(comptime tensor_dtype: DType, value: *const tensor_mod.TensorOf(tensor_dtype), comptime start: usize, comptime count: usize) usize {
    var out: usize = 1;
    inline for (start..start + count) |i| out *= value.shape.at(i);
    return out;
}

pub fn validateTensorRank(comptime tags: anytype, value: *const RawTensor) !void {
    return validateTensorRankOf(.f32, tags, value);
}

pub fn validateTensorRankOf(comptime tensor_dtype: DType, comptime tags: anytype, value: *const tensor_mod.TensorOf(tensor_dtype)) !void {
    if (tags.len == 0) {
        if (!value.isScalar()) return TensorError.InvalidShape;
        return;
    }
    if (value.shape.len != tags.len) return TensorError.InvalidShape;
}

/// Post-pre-sum einsum core: operands carry only batch/free/contract axes.
fn einsumContract(
    comptime l_tags: anytype,
    l: *const RawTensor,
    ctx: *ExecContext,
    comptime r_tags: anytype,
    r: *const RawTensor,
    comptime out_tags: anytype,
) !RawTensor {
    if (comptime out_tags.len == 0) return einsumFullDot(l_tags, l, ctx, r_tags, r);
    return einsumGeneric(l_tags, l, ctx, r_tags, r, out_tags);
}

/// Full contraction to a scalar: flatten both operands (right aligned to the
/// left's axis order) and run the rank-1 dot kernel.
fn einsumFullDot(
    comptime l_tags: anytype,
    l: *const RawTensor,
    ctx: *ExecContext,
    comptime r_tags: anytype,
    r: *const RawTensor,
) !RawTensor {
    var l_ready = try contiguousForReshape(ctx, l);
    defer l_ready.deinit();
    var l_vec = try l_ready.reshape(&.{l_ready.len()});
    defer l_vec.deinit();

    var r_aligned = try alignTensorTo(r_tags, r, l_tags);
    defer r_aligned.deinit();
    var r_ready = try contiguousForReshape(ctx, &r_aligned);
    defer r_ready.deinit();
    var r_vec = try r_ready.reshape(&.{r_ready.len()});
    defer r_vec.deinit();

    return ctx.dot(&l_vec, &r_vec);
}

/// Generic einsum lowering: align both operands to a group-nested order
/// derived from `out_tags` (so the kernel result lands in the requested order
/// whenever `out_tags` nests as [batch][left free][right free] or
/// [batch][right free][left free]), materializing at most once per operand.
/// Interleaved output orders contract in canonical order and pay one output
/// materialization.
fn einsumGeneric(
    comptime l_tags: anytype,
    l: *const RawTensor,
    ctx: *ExecContext,
    comptime r_tags: anytype,
    r: *const RawTensor,
    comptime out_tags: anytype,
) !RawTensor {
    const batch = comptime einsumPartTags(l_tags, r_tags, out_tags, .batch);
    const contract = comptime einsumPartTags(l_tags, r_tags, out_tags, .contract);
    const lf = comptime einsumPartTags(l_tags, r_tags, out_tags, .left_free);
    const rf = comptime einsumPartTags(l_tags, r_tags, out_tags, .right_free);
    const batch_out = comptime intersectTags(out_tags, batch);
    const lf_out = comptime intersectTags(out_tags, lf);
    const rf_out = comptime intersectTags(out_tags, rf);

    if (comptime tagsEqual(batch_out ++ lf_out ++ rf_out, out_tags)) {
        return einsumGenericGemm(l_tags, l, ctx, r_tags, r, batch_out, lf_out, rf_out, comptime intersectTags(l_tags, contract));
    }
    if (comptime tagsEqual(batch_out ++ rf_out ++ lf_out, out_tags)) {
        return einsumGenericGemm(r_tags, r, ctx, l_tags, l, batch_out, rf_out, lf_out, comptime intersectTags(r_tags, contract));
    }

    const batch_phys = comptime intersectTags(l_tags, batch);
    const canon = comptime batch_phys ++ lf ++ rf;
    var value = try einsumGenericGemm(l_tags, l, ctx, r_tags, r, batch_phys, lf, rf, comptime intersectTags(l_tags, contract));
    defer value.deinit();
    var perm = try permuteTensorTo(canon, &value, out_tags);
    defer perm.deinit();
    return ctx.materializeTyped(.f32, &perm);
}

/// One aligned GEMM/BMM pass: `x` as kernel-left with free axes `m_ord`, `y`
/// as kernel-right with free axes `n_ord`, contracting `k_ord`. Each operand
/// picks its kernel layout (plain or transposed) at runtime: a transposed
/// GEMM is free while materializing a permuted view costs a copy pass, so
/// the orientation whose aligned view is already contiguous wins. At most
/// one of transA/transB is available per call — when both operands prefer
/// transposed, the larger one keeps it and the smaller is materialized.
/// Returns the result shaped per-axis as batch_ord ++ m_ord ++ n_ord.
fn einsumGenericGemm(
    comptime x_tags: anytype,
    x: *const RawTensor,
    ctx: *ExecContext,
    comptime y_tags: anytype,
    y: *const RawTensor,
    comptime batch_ord: anytype,
    comptime m_ord: anytype,
    comptime n_ord: anytype,
    comptime k_ord: anytype,
) !RawTensor {
    const x_plain_target = comptime batch_ord ++ m_ord ++ k_ord;
    const x_trans_target = comptime batch_ord ++ k_ord ++ m_ord;
    const y_plain_target = comptime batch_ord ++ k_ord ++ n_ord;
    const y_trans_target = comptime batch_ord ++ n_ord ++ k_ord;

    var trans_a = false;
    var trans_b = false;
    {
        var x_probe = try alignTensorTo(x_tags, x, x_plain_target);
        defer x_probe.deinit();
        var y_probe = try alignTensorTo(y_tags, y, y_plain_target);
        defer y_probe.deinit();
        const x_plain_ok = x_probe.isContiguous();
        const y_plain_ok = y_probe.isContiguous();
        if (!x_plain_ok or !y_plain_ok) {
            var x_trans_probe = try alignTensorTo(x_tags, x, x_trans_target);
            defer x_trans_probe.deinit();
            var y_trans_probe = try alignTensorTo(y_tags, y, y_trans_target);
            defer y_trans_probe.deinit();
            const x_wants = !x_plain_ok and x_trans_probe.isContiguous();
            const y_wants = !y_plain_ok and y_trans_probe.isContiguous();
            if (x_wants and y_wants) {
                if (x.len() >= y.len()) trans_a = true else trans_b = true;
            } else {
                trans_a = x_wants;
                trans_b = y_wants;
            }
        }
    }

    var x_aligned = if (trans_a) try alignTensorTo(x_tags, x, x_trans_target) else try alignTensorTo(x_tags, x, x_plain_target);
    defer x_aligned.deinit();
    var y_aligned = if (trans_b) try alignTensorTo(y_tags, y, y_trans_target) else try alignTensorTo(y_tags, y, y_plain_target);
    defer y_aligned.deinit();
    var x_ready = try contiguousForReshape(ctx, &x_aligned);
    defer x_ready.deinit();
    var y_ready = try contiguousForReshape(ctx, &y_aligned);
    defer y_ready.deinit();

    const x_m_off: usize = if (trans_a) batch_ord.len + k_ord.len else batch_ord.len;
    const y_n_off: usize = if (trans_b) batch_ord.len else batch_ord.len + k_ord.len;
    const m = if (trans_a)
        productRangeOf(.f32, &x_ready, batch_ord.len + k_ord.len, m_ord.len)
    else
        productRangeOf(.f32, &x_ready, batch_ord.len, m_ord.len);
    const k = if (trans_a)
        productRangeOf(.f32, &x_ready, batch_ord.len, k_ord.len)
    else
        productRangeOf(.f32, &x_ready, batch_ord.len + m_ord.len, k_ord.len);
    const n = if (trans_b)
        productRangeOf(.f32, &y_ready, batch_ord.len, n_ord.len)
    else
        productRangeOf(.f32, &y_ready, batch_ord.len + k_ord.len, n_ord.len);

    var value = blk: {
        if (comptime batch_ord.len == 0) {
            var xm = if (trans_a) try x_ready.reshape(&.{ k, m }) else try x_ready.reshape(&.{ m, k });
            defer xm.deinit();
            var ym = if (trans_b) try y_ready.reshape(&.{ n, k }) else try y_ready.reshape(&.{ k, n });
            defer ym.deinit();
            if (trans_a) break :blk try ctx.matmulTransA(&xm, &ym);
            if (trans_b) break :blk try ctx.matmulTransB(&xm, &ym);
            break :blk try ctx.matmul2D(&xm, &ym);
        }
        // The batch group collapses into ONE bmm axis, so any batch count
        // the operands can represent is lowerable (no rank-cap on batch).
        var batches: usize = 1;
        inline for (0..batch_ord.len) |i| batches *= x_ready.shape.at(i);
        var xb = try x_ready.reshape(&.{ batches, if (trans_a) k else m, if (trans_a) m else k });
        defer xb.deinit();
        var yb = try y_ready.reshape(&.{ batches, if (trans_b) n else k, if (trans_b) k else n });
        defer yb.deinit();
        if (trans_a) break :blk try ctx.bmmTransA(&xb, &yb);
        if (trans_b) break :blk try ctx.bmmTransB(&xb, &yb);
        break :blk try ctx.bmm(&xb, &yb);
    };
    errdefer value.deinit();

    var res_shape: [rawRank(batch_ord.len + m_ord.len + n_ord.len)]usize = undefined;
    if (comptime batch_ord.len + m_ord.len + n_ord.len == 0) {
        res_shape[0] = 1;
    } else {
        inline for (0..batch_ord.len) |i| res_shape[i] = x_ready.shape.at(i);
        inline for (0..m_ord.len) |i| res_shape[batch_ord.len + i] = x_ready.shape.at(x_m_off + i);
        inline for (0..n_ord.len) |i| res_shape[batch_ord.len + m_ord.len + i] = y_ready.shape.at(y_n_off + i);
    }
    if (!std.mem.eql(usize, value.shape.slice(), res_shape[0..])) {
        const reshaped = try value.reshape(res_shape[0..]);
        value.deinit();
        value = reshaped;
    }
    return value;
}

/// Tags-rank pointwise broadcast shape (zero-length for scalar results), used
/// by `pointwise`/`gatedPointwise` to feed `broadcastTensorTo` directly.
fn broadcastResultShape(
    comptime result_tags: anytype,
    comptime left_tags: anytype,
    left: *const RawTensor,
    comptime right_tags: anytype,
    right: *const RawTensor,
) ![result_tags.len]usize {
    var shape: [result_tags.len]usize = undefined;
    inline for (result_tags, 0..) |tag, i| {
        const left_dim = dimForTagOf(.f32, left_tags, left, tag);
        const right_dim = dimForTagOf(.f32, right_tags, right, tag);
        if (left_dim == right_dim) {
            shape[i] = left_dim;
        } else if (left_dim == 1) {
            shape[i] = right_dim;
        } else if (right_dim == 1) {
            shape[i] = left_dim;
        } else {
            return TensorError.ShapeMismatch;
        }
    }
    return shape;
}

fn dimForTagOf(comptime tensor_dtype: DType, comptime tags: anytype, value: *const tensor_mod.TensorOf(tensor_dtype), comptime tag: Tag) usize {
    if (comptime tagIndex(tags, tag)) |i| return value.shape.at(i);
    return 1;
}

fn contiguousForReshape(ctx: *ExecContext, value: *const RawTensor) !RawTensor {
    return contiguousForReshapeOf(.f32, ctx, value);
}

fn elementCountArray(comptime rank: usize, shape: [rank]usize) !usize {
    if (rank == 0 or rank > tensor_mod.max_rank) return TensorError.InvalidShape;
    var n: usize = 1;
    inline for (shape) |dim| {
        if (dim == 0) return TensorError.InvalidShape;
        n = try std.math.mul(usize, n, dim);
    }
    return n;
}

fn productArraySuffix(comptime rank: usize, shape: [rank]usize, comptime start: usize) usize {
    var n: usize = 1;
    inline for (start..rank) |i| n *= shape[i];
    return n;
}

test {
    _ = @import("tagged_tests.zig");
}
