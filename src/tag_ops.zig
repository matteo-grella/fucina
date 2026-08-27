//! Tag-semantics op library over raw tensors.
//!
//! This module owns the comptime axis-tag algebra applied at runtime: tag-directed
//! views (align/permute/broadcast/split/merge), tag-driven pointwise broadcasting,
//! multi-axis reductions, and `dot` lowering onto the ExecContext matmul/bmm
//! kernels. Functions take comptime tag tuples plus `*const` raw tensors and
//! return *owned* raw tensors; the public autograd facade (`ag/tensor.zig`) and
//! the VJPs (`ag/backward/`) re-attach tags at comptime on their side.
//!
//! There is intentionally no tagged tensor *type* here: tags are comptime-only
//! data (`tags.zig`), so the single runtime tensor currency stays the raw
//! tensor (`tensor.zig`), which every heterogeneous container (autograd tape,
//! ExecContext ops, weight unions) is built on.
const std = @import("std");
const tensor_mod = @import("tensor.zig");
const backend_ops = @import("backend.zig").ops;
const dtype_mod = @import("dtype.zig");
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
const dotLeftOrder = tags_mod.dotLeftOrder;
const dotLeftFreeTags = tags_mod.dotLeftFreeTags;
const dotRightFreeTags = tags_mod.dotRightFreeTags;
const dotRightTransBOrder = tags_mod.dotRightTransBOrder;
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

/// Tag-driven broadcasting pointwise op over any dtype with a native
/// kernel: broadcasts both operands to the pointwise result tags, then
/// dispatches the rank-matched kernel. Which (op, dtype) pairs exist is
/// the exec kernel's contract (integer `div` and float `max`/`min` are
/// compile errors there); the facade states the rule in words.
pub fn pointwise(
    comptime tensor_dtype: DType,
    comptime op: PointwiseOp,
    comptime left_tags: anytype,
    left: *const tensor_mod.TensorOf(tensor_dtype),
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(tensor_dtype),
) !tensor_mod.TensorOf(dtype_mod.outputDType(.pointwise, tensor_dtype)) {
    try validateTensorRank(tensor_dtype, left_tags, left);
    try validateTensorRank(tensor_dtype, right_tags, right);
    const result_tags = pointwiseResultTags(left_tags, right_tags);
    const result_shape = try broadcastResultShape(tensor_dtype, result_tags, left_tags, left, right_tags, right);

    var left_view = try broadcastTensorTo(tensor_dtype, left_tags, left, result_tags, result_shape);
    defer left_view.deinit();
    var right_view = try broadcastTensorTo(tensor_dtype, right_tags, right, result_tags, result_shape);
    defer right_view.deinit();

    return ctx.elementwise(tensor_dtype, comptime elementwiseOp(op), &left_view, &right_view);
}

/// The backend spelling of a pointwise op (same member set; bridged by name
/// so the two enums cannot drift silently).
pub fn elementwiseOp(comptime op: PointwiseOp) backend_ops.ElementwiseOp {
    return @field(backend_ops.ElementwiseOp, @tagName(op));
}

/// Tag-driven broadcasting gated op (`glu`/`swiglu`/`geglu`/...).
pub fn gatedPointwise(
    comptime tensor_dtype: DType,
    comptime op: GatedOp,
    comptime left_tags: anytype,
    left: *const tensor_mod.TensorOf(tensor_dtype),
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(tensor_dtype),
) !tensor_mod.TensorOf(tensor_dtype) {
    try validateTensorRank(tensor_dtype, left_tags, left);
    try validateTensorRank(tensor_dtype, right_tags, right);
    const result_tags = pointwiseResultTags(left_tags, right_tags);
    const result_shape = try broadcastResultShape(tensor_dtype, result_tags, left_tags, left, right_tags, right);

    var left_view = try broadcastTensorTo(tensor_dtype, left_tags, left, result_tags, result_shape);
    defer left_view.deinit();
    var right_view = try broadcastTensorTo(tensor_dtype, right_tags, right, result_tags, result_shape);
    defer right_view.deinit();

    return ctx.gated(tensor_dtype, rawRank(result_tags.len), op, &left_view, &right_view);
}

/// Tag-directed contraction over one named tag: the single-contract-tag
/// special case of `taggedEinsum`, with the canonical dot result order
/// (batch ++ left free ++ right free). Validates contract/batch dims before
/// computing; kernel selection is the einsum lowering's.
pub fn taggedDot(
    comptime tensor_dtype: DType,
    comptime left_tags: anytype,
    left: *const tensor_mod.TensorOf(tensor_dtype),
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(tensor_dtype),
    comptime contract_tag: Tag,
) !tensor_mod.TensorOf(dtype_mod.outputDType(.matmul, tensor_dtype)) {
    return taggedEinsum(tensor_dtype, left_tags, left, ctx, right_tags, right, comptime dotResultTags(left_tags, right_tags, contract_tag));
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
    comptime tensor_dtype: DType,
    comptime left_tags: anytype,
    left: *const tensor_mod.TensorOf(tensor_dtype),
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(tensor_dtype),
    comptime out_tags: anytype,
) !tensor_mod.TensorOf(dtype_mod.outputDType(.matmul, tensor_dtype)) {
    if (comptime tensor_dtype == .f32) return einsumLower(.f32, .probe_orientation, left_tags, left, ctx, right_tags, right, out_tags);
    // The f32 lowering behind the `.widened` policy: both operands widen
    // once, the result narrows once (the matmul output dtype).
    const compute = comptime ExecContext.widenedCompute(tensor_dtype, "einsum");
    var ll = try ctx.prepareAs(tensor_dtype, compute, left);
    defer ll.deinit();
    var rr = try ctx.prepareAs(tensor_dtype, compute, right);
    defer rr.deinit();
    var value = try einsumLower(.f32, .probe_orientation, left_tags, ll.tensor(), ctx, right_tags, rr.tensor(), out_tags);
    errdefer value.deinit();
    return ctx.storeAs(compute, comptime dtype_mod.outputDType(.matmul, tensor_dtype), value);
}

/// The operand kind of a tagged contraction: which RHS representation the
/// lowering contracts against, and which GEMM family that arm calls. ONE
/// lowering (`contract`) owns the tag algebra, operand alignment, kernel
/// ordering and batch collapse for every kind; the kinds differ in the
/// kernel call and in which equations they accept (the non-f32 kinds take
/// the canonical dot equation: one contraction tag, no summed-away
/// operand-private axes, result order batch ++ left free ++ right free).
pub const ContractKind = union(enum) {
    /// Both operands f32: the full einsum lowering — pre-summed private
    /// axes, plain/transA/transB orientation probing by contiguity,
    /// interleaved output orders, batch collapse onto one bmm axis.
    f32,
    /// Both operands one typed dtype on the native typed GEMM family
    /// (`ctx.dot`/`ctx.matmul`/`ctx.bmm` at the stored dtype), plain
    /// orientation only. f16/bf16 do NOT widen through the f32 path
    /// here: native numerics and speed are the typed dot's contract.
    typed: DType,
    /// f32 LHS against a constant f16/bf16 RHS stored `[free, contract]`:
    /// the half-precision TransB kernel on the NT fast path; anything
    /// else widens the RHS to f32 once and takes the `.typed = .f32` arm.
    half_rhs: DType,
    /// f32 LHS against a block-quantized RHS stored `[free, contract]`:
    /// the fused quantized TransB GEMM (batchless, one RHS free axis).
    quant_rhs: DType,
};

pub fn contractLhsDType(comptime kind: ContractKind) DType {
    return switch (kind) {
        .typed => |tensor_dtype| tensor_dtype,
        else => .f32,
    };
}

pub fn contractRhsDType(comptime kind: ContractKind) DType {
    return switch (kind) {
        .f32 => .f32,
        .typed, .half_rhs, .quant_rhs => |tensor_dtype| tensor_dtype,
    };
}

pub fn contractOutDType(comptime kind: ContractKind) DType {
    return switch (kind) {
        .typed => |tensor_dtype| dtype_mod.outputDType(.matmul, tensor_dtype),
        else => .f32,
    };
}

/// Runtime options of `contract`: today only the quantized arm's GPU gate.
pub const ContractOptions = struct {
    allow_gpu: bool = false,
};

/// The one tagged-contraction lowering, parameterized by operand kind.
/// `out_tags` is the whole einsum equation; the facade `dot`/`einsum`
/// entries are thin equation builders over this. The `.f32` kind accepts
/// any equation; `.typed` runs the same engine at the stored dtype with
/// the plain-orientation policy; `.half_rhs`/`.quant_rhs` are the
/// TransB-layout constant-weight kernels over the shared LHS preparation.
pub fn contract(
    comptime kind: ContractKind,
    comptime left_tags: anytype,
    left: *const tensor_mod.TensorOf(contractLhsDType(kind)),
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(contractRhsDType(kind)),
    comptime out_tags: anytype,
    opts: ContractOptions,
) !tensor_mod.TensorOf(contractOutDType(kind)) {
    switch (comptime kind) {
        .f32 => return einsumLower(.f32, .probe_orientation, left_tags, left, ctx, right_tags, right, out_tags),
        .typed => |tensor_dtype| {
            _ = comptime dotContractTag(left_tags, right_tags, out_tags);
            return einsumLower(tensor_dtype, .plain_only, left_tags, left, ctx, right_tags, right, out_tags);
        },
        .quant_rhs => |tensor_dtype| {
            comptime if (!dtype_mod.isBlockQuantized(tensor_dtype)) @compileError("quantized RHS contraction requires a block-quantized RHS dtype");
            comptime if (!dtype_mod.supportsQuantizedMatmulRhs(tensor_dtype)) @compileError("RHS dtype does not support quantized matmul");
            const contract_tag = comptime dotContractTag(left_tags, right_tags, out_tags);
            comptime if (dotBatchTags(left_tags, right_tags, contract_tag).len != 0) @compileError("quantized RHS dot does not support shared batch tags yet");
            comptime if (dotRightFreeTags(left_tags, right_tags, contract_tag).len != 1) @compileError("quantized RHS dot requires one RHS free axis");
            comptime if (!tagsEqual(right_tags, dotRightTransBOrder(left_tags, right_tags, contract_tag))) {
                @compileError("quantized RHS dot requires RHS storage order [free, contract], e.g. weight tags {.out, .in}");
            };

            const result_shape = try dotResultShape(.f32, tensor_dtype, left_tags, left, right_tags, right, contract_tag);
            var left_matrix = try dotLhsMatrix(left_tags, right_tags, contract_tag, left, ctx, right.shape.at(1));
            defer left_matrix.deinit();
            const product = try ctx.matmul2DWithQuantizedTensorRhs(tensor_dtype, &left_matrix, right, .{ .allow_gpu = opts.allow_gpu });
            return contractFinish(.f32, product, result_shape[0..]);
        },
        .half_rhs => |tensor_dtype| {
            comptime std.debug.assert(tensor_dtype == .f16 or tensor_dtype == .bf16);
            const contract_tag = comptime dotContractTag(left_tags, right_tags, out_tags);
            // The NT fast path runs the half-precision TransB kernel when
            // the contraction is a plain 2-D `[m,k]x[n,k]` (no batch axes,
            // one right free axis, right already in TransB order);
            // everything else widens the RHS to f32 once and takes the
            // typed arm.
            const nt_layout = comptime dotBatchTags(left_tags, right_tags, contract_tag).len == 0 and
                dotRightFreeTags(left_tags, right_tags, contract_tag).len == 1 and
                tagsEqual(right_tags, dotRightTransBOrder(left_tags, right_tags, contract_tag));
            if (comptime nt_layout) {
                const result_shape = try dotResultShape(.f32, tensor_dtype, left_tags, left, right_tags, right, contract_tag);
                var left_matrix = try dotLhsMatrix(left_tags, right_tags, contract_tag, left, ctx, right.shape.at(1));
                defer left_matrix.deinit();
                var right_ready = try contiguousForReshape(tensor_dtype, ctx, right);
                defer right_ready.deinit();
                var right_matrix = try right_ready.reshape(&.{ right.shape.at(0), right.shape.at(1) });
                defer right_matrix.deinit();
                const product = try ctx.matmulHalfRhs(tensor_dtype, &left_matrix, &right_matrix);
                return contractFinish(.f32, product, result_shape[0..]);
            }
            var right_f32 = try ctx.cast(tensor_dtype, .f32, right);
            defer right_f32.deinit();
            return contract(.{ .typed = .f32 }, left_tags, left, ctx, right_tags, &right_f32, out_tags, .{});
        },
    }
}

/// The single contraction tag of a canonical dot equation, derived from
/// the einsum equation (the shared tags dropped from `out_tags`). The
/// non-f32 contraction kinds accept exactly this shape.
fn dotContractTag(comptime left_tags: anytype, comptime right_tags: anytype, comptime out_tags: anytype) Tag {
    comptime {
        const norm_out = normalizeTags(out_tags);
        einsumValidate(left_tags, right_tags, norm_out);
        const contract_tags = einsumPartTags(left_tags, right_tags, norm_out, .contract);
        if (contract_tags.len != 1)
            @compileError("the typed/half/quantized contraction arms take exactly one contraction tag (the dot equation); use the f32 einsum for anything else");
        if (einsumPartTags(left_tags, right_tags, norm_out, .left_summed).len != 0 or
            einsumPartTags(left_tags, right_tags, norm_out, .right_summed).len != 0)
            @compileError("the typed/half/quantized contraction arms do not sum away operand-private axes; use the f32 einsum");
        if (!tagsEqual(norm_out, dotResultTags(left_tags, right_tags, contract_tags[0])))
            @compileError("the typed/half/quantized contraction arms produce the canonical dot order (batch ++ left free ++ right free)");
        return contract_tags[0];
    }
}

/// The LHS of a batchless TransB-style contraction as the kernel's
/// `[m, k]` matrix: aligned to the canonical dot order, materialized when
/// the aligned view is not contiguous, and flattened. Checks the RHS
/// contract dim against k. The caller owns (deinits) the returned matrix.
fn dotLhsMatrix(
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    comptime contract_tag: Tag,
    left: *const RawTensor,
    ctx: *ExecContext,
    rhs_contract_dim: usize,
) !RawTensor {
    const left_free_rank = comptime dotLeftFreeTags(left_tags, right_tags, contract_tag).len;
    var left_aligned = try alignTensorTo(.f32, left_tags, left, dotLeftOrder(left_tags, right_tags, contract_tag));
    defer left_aligned.deinit();
    const m = productRange(.f32, &left_aligned, 0, left_free_rank);
    const k = left_aligned.shape.at(left_free_rank);
    if (rhs_contract_dim != k) return TensorError.ShapeMismatch;
    var left_ready = try contiguousForReshape(.f32, ctx, &left_aligned);
    defer left_ready.deinit();
    return left_ready.reshape(&.{ m, k });
}

/// The shared epilogue: the kernel product reshaped (metadata only) to the
/// per-axis result shape when they differ.
fn contractFinish(
    comptime tensor_dtype: DType,
    product: tensor_mod.TensorOf(tensor_dtype),
    result_shape: []const usize,
) !tensor_mod.TensorOf(tensor_dtype) {
    var owned = product;
    errdefer owned.deinit();
    if (std.mem.eql(usize, owned.shape.slice(), result_shape)) return owned;
    const reshaped = try owned.reshape(result_shape);
    owned.deinit();
    return reshaped;
}

/// The GEMM orientation policy of `einsumLower`: the f32 kind probes
/// operand contiguity and picks plain/transA/transB (a transposed GEMM is
/// free while a materialize costs a copy); the typed kind is pinned to
/// the plain kernels so the typed dot keeps its exact native sequence.
const GemmPolicy = enum { probe_orientation, plain_only };

fn einsumLower(
    comptime tensor_dtype: DType,
    comptime policy: GemmPolicy,
    comptime left_tags: anytype,
    left: *const tensor_mod.TensorOf(tensor_dtype),
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(tensor_dtype),
    comptime out_tags: anytype,
) !tensor_mod.TensorOf(dtype_mod.outputDType(.matmul, tensor_dtype)) {
    comptime einsumValidate(left_tags, right_tags, out_tags);
    try validateTensorRank(tensor_dtype, left_tags, left);
    try validateTensorRank(tensor_dtype, right_tags, right);
    const result_shape = try einsumResultShape(tensor_dtype, tensor_dtype, left_tags, left, right_tags, right, out_tags);

    // Operand-private dropped tags are summed away first: cheaper than
    // carrying them through the contraction, and it leaves every remaining
    // axis with a batch/free/contract role. The typed arms have none (the
    // dot equation), so the pre-sum stays f32-only.
    const left_summed = comptime einsumPartTags(left_tags, right_tags, out_tags, .left_summed);
    const right_summed = comptime einsumPartTags(left_tags, right_tags, out_tags, .right_summed);
    comptime if (tensor_dtype != .f32 and (left_summed.len != 0 or right_summed.len != 0)) {
        @compileError("typed contraction does not sum away operand-private axes; use the f32 einsum");
    };
    const l_tags = comptime removeTags(left_tags, left_summed);
    const r_tags = comptime removeTags(right_tags, right_summed);

    var l_val = if (comptime tensor_dtype == .f32) try sumManyTensor(left_tags, left, ctx, left_summed) else try left.cloneView();
    defer l_val.deinit();
    var r_val = if (comptime tensor_dtype == .f32) try sumManyTensor(right_tags, right, ctx, right_summed) else try right.cloneView();
    defer r_val.deinit();

    var value = try einsumContract(tensor_dtype, policy, l_tags, &l_val, ctx, r_tags, &r_val, out_tags);
    errdefer value.deinit();
    if (out_tags.len != 0 and !std.mem.eql(usize, value.shape.slice(), result_shape[0..])) {
        const reshaped = try value.reshape(result_shape[0..]);
        value.deinit();
        value = reshaped;
    }
    return value;
}

/// Sums away `reduce_tags` innermost-first, returning the reduced tensor
/// tagged `removeTags(tags, reduce_tags)` on the caller's side. A run of
/// adjacent reduce axes is one merged axis (a contiguous reshape, so the
/// `(outer, axis, inner)` decomposition covers the run) and takes one
/// `sumAxis` pass; the remaining axes take one pass each.
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
    try validateTensorRank(.f32, tags, source);

    if (reduce_tags.len == 0) return source.cloneView();
    if (reduce_tags.len == tags.len) return ctx.sum(.f32, source);

    var current = try source.cloneView();
    errdefer current.deinit();

    const axes = comptime reduceAxesDescending(tags, reduce_tags);
    comptime var rank_now: usize = tags.len;
    comptime var i: usize = 0;
    inline while (i < axes.len) {
        comptime var run_len: usize = 1;
        inline while (i + run_len < axes.len and axes[i + run_len] + run_len == axes[i]) run_len += 1;
        const hi = comptime axes[i];
        const lo = comptime hi + 1 - run_len;
        if (comptime run_len > 1) {
            const merged_rank = comptime rank_now - run_len + 1;
            var merged_shape: [merged_rank]usize = undefined;
            var merged_dim: usize = 1;
            inline for (0..rank_now) |d| {
                if (d < lo) {
                    merged_shape[d] = current.shape.at(d);
                } else if (d <= hi) {
                    merged_dim *= current.shape.at(d);
                } else {
                    merged_shape[d + 1 - run_len] = current.shape.at(d);
                }
            }
            merged_shape[lo] = merged_dim;
            var contiguous = if (current.isContiguous()) try current.cloneView() else try ctx.materialize(.f32, &current);
            defer contiguous.deinit();
            var merged = try contiguous.reshape(&merged_shape);
            defer merged.deinit();
            const next = try ctx.sumAxis(.f32, merged_rank, &merged, lo);
            current.deinit();
            current = next;
        } else {
            const next = try ctx.sumAxis(.f32, rank_now, &current, hi);
            current.deinit();
            current = next;
        }
        rank_now -= run_len;
        i += run_len;
    }
    return current;
}

/// Zero-copy view splitting the `tag` axis into `split_tags` factor axes.
pub fn splitAxisView(
    comptime tensor_dtype: DType,
    comptime source_tags: anytype,
    source: *const tensor_mod.TensorOf(tensor_dtype),
    comptime tag: Tag,
    comptime split_tags: anytype,
    split_shape: [split_tags.len]usize,
) !tensor_mod.TensorOf(tensor_dtype) {
    const axis_index = tagIndexOrCompileError(source_tags, tag);
    _ = splitTags(source_tags, tag, split_tags);
    try validateTensorRank(tensor_dtype, source_tags, source);
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
    comptime tensor_dtype: DType,
    comptime source_tags: anytype,
    source: *const tensor_mod.TensorOf(tensor_dtype),
    comptime out_tag: Tag,
    comptime merge_tags: anytype,
) !tensor_mod.TensorOf(tensor_dtype) {
    const start = comptime mergeStartAxis(source_tags, merge_tags);
    _ = mergeTags(source_tags, out_tag, merge_tags);
    try validateTensorRank(tensor_dtype, source_tags, source);

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

pub fn flattenTensor(
    comptime tensor_dtype: DType,
    ctx: *ExecContext,
    source: *const tensor_mod.TensorOf(tensor_dtype),
) !tensor_mod.TensorOf(tensor_dtype) {
    var ready = try contiguousForReshape(tensor_dtype, ctx, source);
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
    return alignTensorTo(.f32, source_tags, source, target_tags);
}

/// Reorders axes to `target_tags` order and injects zero-stride singleton axes
/// for target tags absent from the source. Zero-copy.
pub fn alignTensorTo(
    comptime tensor_dtype: DType,
    comptime source_tags: anytype,
    source: *const tensor_mod.TensorOf(tensor_dtype),
    comptime target_tags: anytype,
) !tensor_mod.TensorOf(tensor_dtype) {
    comptime {
        validateUniqueTags(target_tags);
        if (target_tags.len > tensor_mod.max_rank) @compileError(tensor_mod.too_many_tags_msg);
        for (source_tags) |tag| {
            if (tagIndex(target_tags, tag) == null) @compileError("target tags must include all source tags");
        }
    }
    try validateTensorRank(tensor_dtype, source_tags, source);

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

/// Aligns to `target_tags` order, then broadcasts to `target_shape` with
/// zero-stride expansion. Zero-copy.
pub fn broadcastTensorTo(
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

    var aligned = try alignTensorTo(tensor_dtype, source_tags, source, target_tags);
    defer aligned.deinit();
    return aligned.broadcastToRank(target_tags.len, target_shape);
}

/// Raw-rank pointwise result shape (scalar results report `{1}`); validates
/// dim-by-dim broadcast compatibility.
pub fn pointwiseShape(
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
        const left_dim = dimForTag(tensor_dtype, left_tags, left, tag);
        const right_dim = dimForTag(tensor_dtype, right_tags, right, tag);
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
pub fn dotResultShape(
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
pub fn einsumResultShape(
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

pub fn contiguousForReshape(
    comptime tensor_dtype: DType,
    ctx: *ExecContext,
    value: *const tensor_mod.TensorOf(tensor_dtype),
) !tensor_mod.TensorOf(tensor_dtype) {
    if (value.isContiguous()) return value.cloneView();
    return ctx.materialize(tensor_dtype, value);
}

pub fn productRange(comptime tensor_dtype: DType, value: *const tensor_mod.TensorOf(tensor_dtype), comptime start: usize, comptime count: usize) usize {
    var out: usize = 1;
    inline for (start..start + count) |i| out *= value.shape.at(i);
    return out;
}

pub fn validateTensorRank(comptime tensor_dtype: DType, comptime tags: anytype, value: *const tensor_mod.TensorOf(tensor_dtype)) !void {
    if (tags.len == 0) {
        if (!value.isScalar()) return TensorError.InvalidShape;
        return;
    }
    if (value.shape.len != tags.len) return TensorError.InvalidShape;
}

/// Post-pre-sum einsum core: operands carry only batch/free/contract axes.
fn einsumContract(
    comptime tensor_dtype: DType,
    comptime policy: GemmPolicy,
    comptime l_tags: anytype,
    l: *const tensor_mod.TensorOf(tensor_dtype),
    ctx: *ExecContext,
    comptime r_tags: anytype,
    r: *const tensor_mod.TensorOf(tensor_dtype),
    comptime out_tags: anytype,
) !tensor_mod.TensorOf(dtype_mod.outputDType(.matmul, tensor_dtype)) {
    if (comptime out_tags.len == 0) return einsumFullDot(tensor_dtype, l_tags, l, ctx, r_tags, r);
    return einsumGeneric(tensor_dtype, policy, l_tags, l, ctx, r_tags, r, out_tags);
}

/// Full contraction to a scalar: flatten both operands (right aligned to the
/// left's axis order) and run the rank-1 dot kernel.
fn einsumFullDot(
    comptime tensor_dtype: DType,
    comptime l_tags: anytype,
    l: *const tensor_mod.TensorOf(tensor_dtype),
    ctx: *ExecContext,
    comptime r_tags: anytype,
    r: *const tensor_mod.TensorOf(tensor_dtype),
) !tensor_mod.TensorOf(dtype_mod.outputDType(.matmul, tensor_dtype)) {
    var l_ready = try contiguousForReshape(tensor_dtype, ctx, l);
    defer l_ready.deinit();
    var l_vec = try l_ready.reshape(&.{l_ready.len()});
    defer l_vec.deinit();

    var r_aligned = try alignTensorTo(tensor_dtype, r_tags, r, l_tags);
    defer r_aligned.deinit();
    var r_ready = try contiguousForReshape(tensor_dtype, ctx, &r_aligned);
    defer r_ready.deinit();
    var r_vec = try r_ready.reshape(&.{r_ready.len()});
    defer r_vec.deinit();

    return ctx.dot(tensor_dtype, &l_vec, &r_vec);
}

/// Generic einsum lowering: align both operands to a group-nested order
/// derived from `out_tags` (so the kernel result lands in the requested order
/// whenever `out_tags` nests as [batch][left free][right free] or
/// [batch][right free][left free]), materializing at most once per operand.
/// Interleaved output orders contract in canonical order and pay one output
/// materialization.
fn einsumGeneric(
    comptime tensor_dtype: DType,
    comptime policy: GemmPolicy,
    comptime l_tags: anytype,
    l: *const tensor_mod.TensorOf(tensor_dtype),
    ctx: *ExecContext,
    comptime r_tags: anytype,
    r: *const tensor_mod.TensorOf(tensor_dtype),
    comptime out_tags: anytype,
) !tensor_mod.TensorOf(tensor_dtype) {
    const batch = comptime einsumPartTags(l_tags, r_tags, out_tags, .batch);
    const contracted = comptime einsumPartTags(l_tags, r_tags, out_tags, .contract);
    const lf = comptime einsumPartTags(l_tags, r_tags, out_tags, .left_free);
    const rf = comptime einsumPartTags(l_tags, r_tags, out_tags, .right_free);
    const batch_out = comptime intersectTags(out_tags, batch);
    const lf_out = comptime intersectTags(out_tags, lf);
    const rf_out = comptime intersectTags(out_tags, rf);

    if (comptime tagsEqual(batch_out ++ lf_out ++ rf_out, out_tags)) {
        return einsumGenericGemm(tensor_dtype, policy, l_tags, l, ctx, r_tags, r, batch_out, lf_out, rf_out, comptime intersectTags(l_tags, contracted));
    }
    if (comptime tagsEqual(batch_out ++ rf_out ++ lf_out, out_tags)) {
        return einsumGenericGemm(tensor_dtype, policy, r_tags, r, ctx, l_tags, l, batch_out, rf_out, lf_out, comptime intersectTags(r_tags, contracted));
    }

    const batch_phys = comptime intersectTags(l_tags, batch);
    const canon = comptime batch_phys ++ lf ++ rf;
    var value = try einsumGenericGemm(tensor_dtype, policy, l_tags, l, ctx, r_tags, r, batch_phys, lf, rf, comptime intersectTags(l_tags, contracted));
    defer value.deinit();
    comptime validateSameTagSet(canon, out_tags);
    var perm = try alignTensorTo(tensor_dtype, canon, &value, out_tags);
    defer perm.deinit();
    return ctx.materialize(tensor_dtype, &perm);
}

/// One aligned GEMM/BMM pass: `x` as kernel-left with free axes `m_ord`, `y`
/// as kernel-right with free axes `n_ord`, contracting `k_ord`. Under the
/// `.probe_orientation` policy each operand picks its kernel layout (plain
/// or transposed) at runtime: a transposed GEMM is free while materializing
/// a permuted view costs a copy pass, so the orientation whose aligned view
/// is already contiguous wins. At most one of transA/transB is available
/// per call — when both operands prefer transposed, the larger one keeps it
/// and the smaller is materialized. Under `.plain_only` (the typed kinds)
/// both operands take the plain layout, materializing when not contiguous.
/// Returns the result shaped per-axis as batch_ord ++ m_ord ++ n_ord.
fn einsumGenericGemm(
    comptime tensor_dtype: DType,
    comptime policy: GemmPolicy,
    comptime x_tags: anytype,
    x: *const tensor_mod.TensorOf(tensor_dtype),
    ctx: *ExecContext,
    comptime y_tags: anytype,
    y: *const tensor_mod.TensorOf(tensor_dtype),
    comptime batch_ord: anytype,
    comptime m_ord: anytype,
    comptime n_ord: anytype,
    comptime k_ord: anytype,
) !tensor_mod.TensorOf(tensor_dtype) {
    const x_plain_target = comptime batch_ord ++ m_ord ++ k_ord;
    const x_trans_target = comptime batch_ord ++ k_ord ++ m_ord;
    const y_plain_target = comptime batch_ord ++ k_ord ++ n_ord;
    const y_trans_target = comptime batch_ord ++ n_ord ++ k_ord;

    const can_trans = comptime policy == .probe_orientation;
    const orientation: struct { a: bool, b: bool } = blk: {
        if (comptime !can_trans) break :blk .{ .a = false, .b = false };
        var x_probe = try alignTensorTo(tensor_dtype, x_tags, x, x_plain_target);
        defer x_probe.deinit();
        var y_probe = try alignTensorTo(tensor_dtype, y_tags, y, y_plain_target);
        defer y_probe.deinit();
        const x_plain_ok = x_probe.isContiguous();
        const y_plain_ok = y_probe.isContiguous();
        if (!x_plain_ok or !y_plain_ok) {
            var x_trans_probe = try alignTensorTo(tensor_dtype, x_tags, x, x_trans_target);
            defer x_trans_probe.deinit();
            var y_trans_probe = try alignTensorTo(tensor_dtype, y_tags, y, y_trans_target);
            defer y_trans_probe.deinit();
            const x_wants = !x_plain_ok and x_trans_probe.isContiguous();
            const y_wants = !y_plain_ok and y_trans_probe.isContiguous();
            if (x_wants and y_wants) {
                if (x.len() >= y.len()) break :blk .{ .a = true, .b = false };
                break :blk .{ .a = false, .b = true };
            }
            break :blk .{ .a = x_wants, .b = y_wants };
        }
        break :blk .{ .a = false, .b = false };
    };
    const trans_a = orientation.a;
    const trans_b = orientation.b;

    var x_aligned = if (trans_a) try alignTensorTo(tensor_dtype, x_tags, x, x_trans_target) else try alignTensorTo(tensor_dtype, x_tags, x, x_plain_target);
    defer x_aligned.deinit();
    var y_aligned = if (trans_b) try alignTensorTo(tensor_dtype, y_tags, y, y_trans_target) else try alignTensorTo(tensor_dtype, y_tags, y, y_plain_target);
    defer y_aligned.deinit();
    var x_ready = try contiguousForReshape(tensor_dtype, ctx, &x_aligned);
    defer x_ready.deinit();
    var y_ready = try contiguousForReshape(tensor_dtype, ctx, &y_aligned);
    defer y_ready.deinit();

    const x_m_off: usize = if (trans_a) batch_ord.len + k_ord.len else batch_ord.len;
    const y_n_off: usize = if (trans_b) batch_ord.len else batch_ord.len + k_ord.len;
    const m = if (trans_a)
        productRange(tensor_dtype, &x_ready, batch_ord.len + k_ord.len, m_ord.len)
    else
        productRange(tensor_dtype, &x_ready, batch_ord.len, m_ord.len);
    const k = if (trans_a)
        productRange(tensor_dtype, &x_ready, batch_ord.len, k_ord.len)
    else
        productRange(tensor_dtype, &x_ready, batch_ord.len + m_ord.len, k_ord.len);
    const n = if (trans_b)
        productRange(tensor_dtype, &y_ready, batch_ord.len, n_ord.len)
    else
        productRange(tensor_dtype, &y_ready, batch_ord.len + k_ord.len, n_ord.len);

    var value = blk: {
        if (comptime batch_ord.len == 0) {
            var xm = if (trans_a) try x_ready.reshape(&.{ k, m }) else try x_ready.reshape(&.{ m, k });
            defer xm.deinit();
            var ym = if (trans_b) try y_ready.reshape(&.{ n, k }) else try y_ready.reshape(&.{ k, n });
            defer ym.deinit();
            if (comptime can_trans) {
                if (trans_a) break :blk try ctx.matmul(tensor_dtype, .trans_a, &xm, &ym);
                if (trans_b) break :blk try ctx.matmul(tensor_dtype, .trans_b, &xm, &ym);
            }
            break :blk try ctx.matmul(tensor_dtype, .plain, &xm, &ym);
        }
        // The batch group collapses into ONE bmm axis, so any batch count
        // the operands can represent is lowerable (no rank-cap on batch).
        var batches: usize = 1;
        inline for (0..batch_ord.len) |i| batches *= x_ready.shape.at(i);
        var xb = try x_ready.reshape(&.{ batches, if (trans_a) k else m, if (trans_a) m else k });
        defer xb.deinit();
        var yb = try y_ready.reshape(&.{ batches, if (trans_b) n else k, if (trans_b) k else n });
        defer yb.deinit();
        if (comptime can_trans) {
            if (trans_a) break :blk try ctx.bmm(tensor_dtype, .trans_a, &xb, &yb);
            if (trans_b) break :blk try ctx.bmm(tensor_dtype, .trans_b, &xb, &yb);
        }
        break :blk try ctx.bmm(tensor_dtype, .plain, &xb, &yb);
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
    comptime tensor_dtype: DType,
    comptime result_tags: anytype,
    comptime left_tags: anytype,
    left: *const tensor_mod.TensorOf(tensor_dtype),
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(tensor_dtype),
) ![result_tags.len]usize {
    var shape: [result_tags.len]usize = undefined;
    inline for (result_tags, 0..) |tag, i| {
        const left_dim = dimForTag(tensor_dtype, left_tags, left, tag);
        const right_dim = dimForTag(tensor_dtype, right_tags, right, tag);
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

fn dimForTag(comptime tensor_dtype: DType, comptime tags: anytype, value: *const tensor_mod.TensorOf(tensor_dtype), comptime tag: Tag) usize {
    if (comptime tagIndex(tags, tag)) |i| return value.shape.at(i);
    return 1;
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
    _ = @import("tag_ops_tests.zig");
}
