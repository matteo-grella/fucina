//! f32 tensor methods: indexed reads and writes. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../../tensor.zig");
const dtype_mod = @import("../../../dtype.zig");
const exec_mod = @import("../../../exec.zig");
const backend_mod = @import("../../../backend.zig");
const parallel = @import("../../../parallel.zig");
const tag_ops = @import("../../../tagged.zig");
const control = @import("../../control.zig");
const core = @import("../../core.zig");
const tags_mod = @import("../../../tags.zig");
const backward = @import("../../backward.zig");
const elemental = @import("../../elemental.zig");
const rng = @import("../../../rng.zig");

const RawTensor = tensor_mod.Tensor;
const DType = tensor_mod.DType;
const BlockQ8_0 = dtype_mod.BlockQ8_0;
const TensorError = tensor_mod.TensorError;
const Scalar = tensor_mod.Scalar;
const ExecContext = exec_mod.ExecContext;
const SoftmaxExtOptions = exec_mod.SoftmaxExtOptions;
const UnaryOp = exec_mod.UnaryOp;
const GatedOp = exec_mod.GatedOp;
const RopeMode = exec_mod.RopeMode;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const inserted_axis = tags_mod.inserted_axis;
const normalizeTags = tags_mod.normalizeTags;
const dtypeFromSpec = tags_mod.dtypeFromSpec;
const validateUniqueTags = tags_mod.validateUniqueTags;
const validateSameTagSet = tags_mod.validateSameTagSet;
const rawRank = tags_mod.rawRank;
const tagIndex = tags_mod.tagIndex;
const tagIndexOrCompileError = tags_mod.tagIndexOrCompileError;
const identityAxes = tags_mod.identityAxes;
const alignAxes = tags_mod.alignAxes;
const insertAxes = tags_mod.insertAxes;
const squeezeAxes = tags_mod.squeezeAxes;
const removeTag = tags_mod.removeTag;
const removeTags = tags_mod.removeTags;
const replaceTag = tags_mod.replaceTag;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const dotResultTags = tags_mod.dotResultTags;
const insertTagAt = tags_mod.insertTagAt;
const splitTags = tags_mod.splitTags;
const mergeTags = tags_mod.mergeTags;
const tagsEqual = tags_mod.tagsEqual;
const dotLeftOrder = tags_mod.dotLeftOrder;
const dotRightOrder = tags_mod.dotRightOrder;
const dotRightTransBOrder = tags_mod.dotRightTransBOrder;
const dotBatchLen = tags_mod.dotBatchLen;
const dotLeftFreeLen = tags_mod.dotLeftFreeLen;
const dotRightFreeLen = tags_mod.dotRightFreeLen;
const alignTensorToOf = tag_ops.alignTensorToOf;
const broadcastTensorTo = tag_ops.broadcastTensorTo;
const broadcastTensorToOf = tag_ops.broadcastTensorToOf;
const contiguousForReshapeOf = tag_ops.contiguousForReshapeOf;
const dotResultShapeOf = tag_ops.dotResultShapeOf;
const pointwiseShape = tag_ops.pointwiseShape;
const pointwiseShapeOf = tag_ops.pointwiseShapeOf;
const productRangeOf = tag_ops.productRangeOf;
const validateTensorRank = tag_ops.validateTensorRank;
const validateTensorRankOf = tag_ops.validateTensorRankOf;
const PointwiseOp = backward.PointwiseOp;
const PointwiseBackward = backward.PointwiseBackward;
const CastBackward = backward.CastBackward;
const IdentityBackward = backward.IdentityBackward;
const Matmul2DBackward = backward.Matmul2DBackward;
const BmmBackward = backward.BmmBackward;
const ReluBackward = backward.ReluBackward;
const Conv2dBackward = backward.Conv2dBackward;
const UnfoldBackward = backward.UnfoldBackward;
const FoldBackward = backward.FoldBackward;
const MaxPool2dBackward = backward.MaxPool2dBackward;
const AvgPool2dBackward = backward.AvgPool2dBackward;
const Upsample2xNearestBackward = backward.Upsample2xNearestBackward;
const PreluChannelsBackward = backward.PreluChannelsBackward;
const ChannelAffineBackward = backward.ChannelAffineBackward;
const RelposShiftBackward = backward.RelposShiftBackward;
const LeakyReluBackward = backward.LeakyReluBackward;
const UnaryBackward = backward.UnaryBackward;
const unaryUsesOutput = backward.unaryUsesOutput;
const ScaleBackward = backward.ScaleBackward;
const AddScalarBackward = backward.AddScalarBackward;
const PowScalarBackward = backward.PowScalarBackward;
const MaskedFillBackward = backward.MaskedFillBackward;
const WhereBackward = backward.WhereBackward;
const DropoutBackward = backward.DropoutBackward;
const ClampBackward = backward.ClampBackward;
const GatedBackward = backward.GatedBackward;
const SplitSwiGluBackward = backward.SplitSwiGluBackward;
const SplitGluBackward = backward.SplitGluBackward;
const SumBackward = backward.SumBackward;
const MeanBackward = backward.MeanBackward;
const MaskedSumBackward = backward.MaskedSumBackward;
const MaskedMeanBackward = backward.MaskedMeanBackward;
const MaskedMinMaxBackward = backward.MaskedMinMaxBackward;
const VarBackward = backward.VarBackward;
const StandardizeBackward = backward.StandardizeBackward;
const BroadcastBackward = backward.BroadcastBackward;
const GatherBackward = backward.GatherBackward;
const TopKBackward = backward.TopKBackward;
const MinMaxBackward = backward.MinMaxBackward;
const NarrowBackward = backward.NarrowBackward;
const ConcatBackward = backward.ConcatBackward;
const CumsumBackward = backward.CumsumBackward;
const SegmentSumBackward = backward.SegmentSumBackward;
const LinearRecurrenceBackward = backward.LinearRecurrenceBackward;
const PadBackward = backward.PadBackward;
const SetSliceBackward = backward.SetSliceBackward;
const SetRowsBackward = backward.SetRowsBackward;
const IndexAddBackward = backward.IndexAddBackward;
const ProdBackward = backward.ProdBackward;
const CumprodBackward = backward.CumprodBackward;
const TakeAlongBackward = backward.TakeAlongBackward;
const LogsumexpBackward = backward.LogsumexpBackward;
const LogSoftmaxBackward = backward.LogSoftmaxBackward;
const ScatterAlongBackward = backward.ScatterAlongBackward;
const ZeroSliceBackward = backward.ZeroSliceBackward;
const ZeroRowsBackward = backward.ZeroRowsBackward;
const SoftmaxBackward = backward.SoftmaxBackward;
const SoftmaxExtBackward = backward.SoftmaxExtBackward;
const RmsNormBackward = backward.RmsNormBackward;
const RmsNormMulBackward = backward.RmsNormMulBackward;
const RmsNormMulAddBackward = backward.RmsNormMulAddBackward;
const RmsNormMulRopeBackward = backward.RmsNormMulRopeBackward;
const LayerNormBackward = backward.LayerNormBackward;
const LayerNormAffineBackward = backward.LayerNormAffineBackward;
const CrossEntropyBackward = backward.CrossEntropyBackward;
const CrossEntropyExtBackward = backward.CrossEntropyExtBackward;
const LinearCrossEntropyBackward = backward.LinearCrossEntropyBackward;
const LinearDistillBackward = backward.LinearDistillBackward;
const MseLossBackward = backward.MseLossBackward;
const HuberLossBackward = backward.HuberLossBackward;
const BceLossBackward = backward.BceLossBackward;
const KlDivLossBackward = backward.KlDivLossBackward;
const RopeBackward = backward.RopeBackward;
const RopeTableBackward = backward.RopeTableBackward;
const ReshapeBackward = backward.ReshapeBackward;
const AxisViewBackward = backward.AxisViewBackward;
const StridedViewBackward = backward.StridedViewBackward;
const CausalDepthwiseConv1dBackward = backward.CausalDepthwiseConv1dBackward;
const CausalConv1dBackward = backward.CausalConv1dBackward;
const GroupedCausalConv1dBackward = backward.GroupedCausalConv1dBackward;
const Conv1dBackward = backward.Conv1dBackward;
const ConvTranspose1dBackward = backward.ConvTranspose1dBackward;
const SnakeBackward = backward.SnakeBackward;
const GroupNormBackward = backward.GroupNormBackward;
const GroupedCausalAttentionBackward = backward.GroupedCausalAttentionBackward;
const DotBackward = backward.DotBackward;
const AddDotBackward = backward.AddDotBackward;
const EinsumBackward = backward.EinsumBackward;
const ConstRhsDotBackward = backward.ConstRhsDotBackward;
const ConstRhsEinsumBackward = backward.ConstRhsEinsumBackward;
const TernarySteDotBackward = backward.TernarySteDotBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tensor_rank = Self.tensor_rank;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const TopKResult = ag_tensor.TopKResult;
        const PackedRhs = ag_tensor.PackedRhs;
        const SliceRange = ag_tensor.SliceRange;
        const concat_inline_inputs = ag_tensor.concat_inline_inputs;
        const AttentionMask = ag_tensor.AttentionMask;
        const AttentionKvRepr = ag_tensor.AttentionKvRepr;
        const normParamTagCheck = ag_tensor.normParamTagCheck;
        const attentionKvRepr = ag_tensor.attentionKvRepr;
        const packedRhsLayout = ag_tensor.packedRhsLayout;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const pointwise = plumbing.pointwise;
        const gatedPointwise = plumbing.gatedPointwise;
        const rowStatsAlloc = plumbing.rowStatsAlloc;
        const finishOp = plumbing.finishOp;
        const requireScopeForComposedGrad = plumbing.requireScopeForComposedGrad;
        const einsumMany = plumbing.einsumMany;
        const finishNoGrad = plumbing.finishNoGrad;
        const adoptIntoScope = plumbing.adoptIntoScope;
        const destroyGradStateOpaque = plumbing.destroyGradStateOpaque;
        const finishWithBackward = plumbing.finishWithBackward;
        const axisViewTensor = plumbing.axisViewTensor;
        const axisViewTensorOf = plumbing.axisViewTensorOf;
        const typedDotRaw = plumbing.typedDotRaw;
        const quantizedRhsDotRaw = plumbing.quantizedRhsDotRaw;
        const halfRhsDotRaw = plumbing.halfRhsDotRaw;
        const TensorObject = plumbing.TensorObject;
        const validateMaskedReduceOptions = plumbing.validateMaskedReduceOptions;
        const validateMaskType = plumbing.validateMaskType;
        const maskedReduceEmpty = plumbing.maskedReduceEmpty;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;
        const padding2dValues = plumbing.padding2dValues;
        const typed_constant = @import("../typed_constant.zig").Mod(ag_tensor);
        const QuantizedConstantTensor = typed_constant.QuantizedConstantTensor;
        const TypedConstantTensor = typed_constant.TypedConstantTensor;
        const typedFinishOp = typed_constant.typedFinishOp;

        /// No-grad: copy of `self` with `[start, start+length)` along `axis_tag` zeroed.
        pub fn zeroSlice(self: *const Self, ctx: *ExecContext, comptime axis_tag: Tag, start: usize, length: usize) !Self {
            const zero_axis = comptime Self.axis(axis_tag);
            var value = try ctx.zeroSliceAxisRank(tensor_rank, self.asRawTensor(), zero_axis, start, length);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), ZeroSliceBackward(tags, zero_axis), .{ ctx.allocator, self.grad_state, start, length });
        }

        /// No-grad: copy of `self` with the given `indices` along `axis_tag` zeroed.
        pub fn zeroRows(self: *const Self, ctx: *ExecContext, comptime axis_tag: Tag, indices: []const usize) !Self {
            const zero_axis = comptime Self.axis(axis_tag);
            var value = try ctx.zeroRowsAxisRank(tensor_rank, self.asRawTensor(), zero_axis, indices);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), ZeroRowsBackward(tags, zero_axis), .{ ctx.allocator, self.grad_state, indices });
        }

        /// Transformer-XL relative-shift / "skew": a rank-3
        /// relative-score tensor `[H,Tq,P]` → `[H,Tq,Tk]` with
        /// `out[h,qi,kj] = self[h, qi, kj+(Tq-1)-qi]` (`P >= Tk+Tq-1`). The closed
        /// form of the relpos pad/reshape/view remap; differentiable (scatter VJP).
        /// `out_tags` names the result axes.
        pub fn relposShift(self: *const Self, ctx: *ExecContext, t_k: usize, comptime out_tags: anytype) !Tensor(out_tags) {
            var value = try ctx.relposShiftRank3(self.asRawTensor(), t_k);
            errdefer value.deinit();
            return finishOp(out_tags, ctx, value, self.requiresGrad(), RelposShiftBackward, .{ ctx.allocator, self.grad_state, self.value.shape.slice()[2] });
        }

        pub fn gather(
            self: *const Self,
            ctx: *ExecContext,
            comptime tag: Tag,
            indices: []const usize,
            comptime out_tag: Tag,
        ) !Tensor(replaceTag(tags, tag, out_tag)) {
            const result_tags = replaceTag(tags, tag, out_tag);
            const gather_axis = comptime axis(tag);
            var value = try ctx.gatherAxisRank(tag_rank, self.asRawTensor(), gather_axis, indices);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), GatherBackward(tags, gather_axis), .{ ctx.allocator, self.grad_state, &self.value, indices });
        }

        /// `gather` with a tensor of indices (torch.index_select): `indices`
        /// is a rank-1 **i64** tensor (the argmax/topK/sort index
        /// convention — their outputs feed it directly; any other dtype is
        /// a compile error), read host-side into `gather`'s `[]usize`
        /// buffer; entries outside `[0, dim(tag))` error with
        /// `IndexOutOfBounds` (negatives are not wrapped). The selected
        /// axis is retagged `out_tag` (which may equal `tag`), sized by the
        /// index count. Differentiable in `self` (the exact scatter-add
        /// adjoint — duplicate indices accumulate their gradients); the
        /// index tensor is control data, not part of the graph, and can be
        /// released after the call.
        pub fn indexSelect(
            self: *const Self,
            ctx: *ExecContext,
            comptime tag: Tag,
            indices: anytype,
            comptime out_tag: Tag,
        ) !Tensor(replaceTag(tags, tag, out_tag)) {
            const Indices = TensorObject(@TypeOf(indices));
            comptime {
                if (Indices.dtype != .i64)
                    @compileError("indexSelect expects i64 indices (the argmax/topK/sort output dtype); cast integer indices explicitly");
                if (Indices.axis_tags.len != 1)
                    @compileError("indexSelect expects rank-1 indices (torch.index_select); for per-element index tensors use takeAlongAxis");
            }
            const idx_raw = indices.asRawTensor();
            const idx_buf = try hostIndexBuffer(ctx, try idx_raw.dataConstChecked(), self.asRawTensor().shape.at(comptime axis(tag)));
            defer ctx.allocator.free(idx_buf);
            return self.gather(ctx, tag, idx_buf, out_tag);
        }

        /// Select the elements of `self` where `mask` is nonzero (torch
        /// masked_select): a rank-1 tensor tagged `out_tag` holding the
        /// selected elements in row-major order. The mask must have `self`'s
        /// shape, be contiguous (the nonzero scan reads its storage
        /// host-side), and is a non-grad `!= 0` mask; `self` IS
        /// differentiable — composed flatten + gather, so GatherBackward
        /// scatter-adds the gradient into the source. The gather index
        /// buffer is exact `[]usize` (NOT the f32 index convention of
        /// argmax/topK/sort — no < 2^24 exactness caveat applies here).
        /// Errors with `EmptySelection` when the mask selects nothing
        /// (zero-size tensors are not representable) — a dedicated error,
        /// distinct from the shape errors, so the data-dependent no-match
        /// outcome stays catchable apart from caller bugs; pre-counting via
        /// a mask sum avoids the error path entirely. When gradients are
        /// tracked this requires an active exec scope (see `nllLoss`);
        /// errors with `ActiveExecScopeRequired` otherwise.
        pub fn maskedSelect(self: *const Self, ctx: *ExecContext, mask: anytype, comptime out_tag: Tag) !Tensor(.{out_tag}) {
            const Mask = TensorObject(@TypeOf(mask));
            comptime {
                if (Mask.dtype != .bool and !dtype_mod.supportsForwardFloatMath(Mask.dtype))
                    @compileError("maskedSelect takes a .bool or float mask; cast integer masks explicitly");
            }
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const mask_raw = mask.asRawTensor();
            if (!std.mem.eql(usize, self.asRawTensor().shape.slice(), mask_raw.shape.slice())) return TensorError.ShapeMismatch;
            const mask_values = try mask_raw.dataConstChecked();

            var count: usize = 0;
            for (mask_values) |mv| count += @intFromBool(dtype_mod.isTruthy(Mask.dtype, mv));
            if (count == 0) return TensorError.EmptySelection;

            const indices = try ctx.allocator.alloc(usize, count);
            defer ctx.allocator.free(indices);
            var slot: usize = 0;
            for (mask_values, 0..) |mv, i| {
                if (dtype_mod.isTruthy(Mask.dtype, mv)) {
                    indices[slot] = i;
                    slot += 1;
                }
            }

            var flat = try self.flatten(ctx, out_tag);
            defer flat.deinit();
            return flat.gather(ctx, out_tag, indices, out_tag);
        }

        /// Row-major flat indices of the nonzero elements (torch.nonzero
        /// over the flattened tensor; NaN counts as nonzero), returned as
        /// a HOST slice the caller owns and frees with `allocator` — the
        /// design keeps data-dependent cardinality host-side, where
        /// `[]usize` pairs directly with `gather`/`setRows`/`indexAdd`/
        /// `oneHot`, so a no-match result is just an empty slice (no
        /// zero-size tensor needed, unlike `maskedSelect`). Reads `self`
        /// host-side; contiguous only (like `maskedSelect`'s mask).
        pub fn nonzero(self: *const Self, allocator: std.mem.Allocator) ![]usize {
            const values = try self.asRawTensor().dataConstChecked();
            var count: usize = 0;
            for (values) |v| count += @intFromBool(v != 0);
            const indices = try allocator.alloc(usize, count);
            var slot: usize = 0;
            for (values, 0..) |v, i| {
                if (v != 0) {
                    indices[slot] = i;
                    slot += 1;
                }
            }
            return indices;
        }

        /// Scatter a rank-1 `values` tensor into the positions of `self`
        /// where `mask` is nonzero, in row-major order — the inverse of
        /// `maskedSelect` (torch masked_scatter, with an exact-count
        /// contract: `values` must hold exactly `count(mask != 0)`
        /// elements). Unselected positions keep `self`'s values. The mask
        /// follows the `where`/`maskedFill` convention (non-grad `!= 0`
        /// mask, same shape as `self`, contiguous — the nonzero scan reads
        /// its storage host-side). Differentiable in `self` (grad zeroed at
        /// scattered positions) and `values` (grad gathered row-major from
        /// the selected positions) — composed gather + split + where, so
        /// the gradients come from the existing exact records. Errors with
        /// `EmptySelection` when the mask selects nothing (zero-size tensors
        /// are not representable; see `maskedSelect`) and with `InvalidShape`
        /// when `values`' length differs from the selected count. When
        /// gradients are tracked this requires an active exec scope (see
        /// `maskedSelect`); errors with `ActiveExecScopeRequired` otherwise.
        pub fn maskedScatter(
            self: *const Self,
            ctx: *ExecContext,
            mask: anytype,
            comptime values_tag: Tag,
            values: *const Tensor(.{values_tag}),
        ) !Self {
            comptime if (tag_rank == 0) @compileError("maskedScatter requires at least one axis");
            const Mask = TensorObject(@TypeOf(mask));
            comptime {
                if (Mask.dtype != .bool and !dtype_mod.supportsForwardFloatMath(Mask.dtype))
                    @compileError("maskedScatter takes a .bool or float mask; cast integer masks explicitly");
            }
            try requireScopeForComposedGrad(ctx, self.requiresGrad() or values.requiresGrad());
            const mask_raw = mask.asRawTensor();
            if (!std.mem.eql(usize, self.asRawTensor().shape.slice(), mask_raw.shape.slice())) return TensorError.ShapeMismatch;
            const mask_values = try mask_raw.dataConstChecked();

            var count: usize = 0;
            for (mask_values) |mv| count += @intFromBool(dtype_mod.isTruthy(Mask.dtype, mv));
            if (count == 0) return TensorError.EmptySelection;
            if (values.asRawTensor().len() != count) return TensorError.InvalidShape;

            // Selected position i gathers values[k(i)]; unselected positions
            // gather values[0] as a placeholder that `where` discards — its
            // gradient contribution is exactly the zeros `where` routes there.
            const indices = try ctx.allocator.alloc(usize, mask_values.len);
            defer ctx.allocator.free(indices);
            var slot: usize = 0;
            for (mask_values, indices) |mv, *index| {
                if (dtype_mod.isTruthy(Mask.dtype, mv)) {
                    index.* = slot;
                    slot += 1;
                } else {
                    index.* = 0;
                }
            }

            var gathered = try values.gather(ctx, values_tag, indices, values_tag);
            defer gathered.deinit();
            var dense = try gathered.split(ctx, values_tag, tags, self.shape());
            defer dense.deinit();
            return dense.where(ctx, mask, self);
        }

        pub fn setSlice(self: *const Self, ctx: *ExecContext, comptime tag: Tag, start: usize, update: *const Self) !Self {
            const slice_axis = comptime axis(tag);
            var value = try ctx.setSliceAxisRank(tag_rank, self.asRawTensor(), update.asRawTensor(), slice_axis, start);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or update.requiresGrad(), SetSliceBackward(tags, slice_axis), .{ ctx.allocator, self.grad_state, update.grad_state, update.asRawTensor(), start });
        }

        pub fn setRows(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: []const usize, update: *const Self) !Self {
            const rows_axis = comptime axis(tag);
            var value = try ctx.setRowsAxisRank(tag_rank, self.asRawTensor(), update.asRawTensor(), rows_axis, indices);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or update.requiresGrad(), SetRowsBackward(tags, rows_axis), .{ ctx.allocator, self.grad_state, update.grad_state, indices });
        }

        /// Functional row accumulation (torch.index_add): a copy of `self`
        /// with `update`'s rows ADDED at `indices` along `tag` — unlike
        /// `setRows` this accumulates, and duplicate indices are allowed
        /// (each occurrence adds; torch semantics). `update` must match
        /// `self` except along `tag`, where it has `indices.len` rows.
        /// Differentiable in both: d/dself is the identity, d/dupdate
        /// gathers the addressed rows of the upstream gradient.
        pub fn indexAdd(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: []const usize, update: *const Self) !Self {
            const add_axis = comptime axis(tag);
            var scattered = try ctx.scatterAddAxisRank(tag_rank, update.asRawTensor(), self.shape(), add_axis, indices);
            defer scattered.deinit();
            var value = try ctx.add(self.asRawTensor(), &scattered);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or update.requiresGrad(), IndexAddBackward(tags, add_axis), .{ ctx.allocator, self.grad_state, update.grad_state, indices });
        }

        /// Read a same-tagged i64 index tensor into a host `[]usize`
        /// buffer (the argmax/topK/sort index convention; negatives and
        /// out-of-range values error with `IndexOutOfBounds`). Caller
        /// frees.
        fn hostIndexBuffer(ctx: *ExecContext, values: []const i64, limit: usize) ![]usize {
            const out = try ctx.allocator.alloc(usize, values.len);
            errdefer ctx.allocator.free(out);
            for (values, out) |v, *slot| {
                if (v < 0 or v >= limit) return TensorError.IndexOutOfBounds;
                slot.* = @intCast(v);
            }
            return out;
        }

        /// Elementwise gather along `tag` (torch.gather /
        /// np.take_along_axis): `out[.., i, ..] = self[.., indices[.., i,
        /// ..], ..]`. `indices` is a same-tagged i64 tensor (the
        /// argmax/topK/sort index convention — pairing directly with their
        /// outputs), contiguous, matching `self` on every other axis; the
        /// result takes `indices`' shape. Serial deterministic kernel;
        /// differentiable in `self` (the exact scatter-add adjoint;
        /// duplicate reads accumulate their gradients).
        pub fn takeAlongAxis(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: anytype) !Self {
            comptime {
                if (TensorObject(@TypeOf(indices)).dtype != .i64)
                    @compileError("takeAlongAxis expects i64 indices (the argmax/topK/sort output dtype)");
            }
            const take_axis = comptime axis(tag);
            const idx_raw = indices.asRawTensor();
            const raw = self.asRawTensor();
            inline for (0..tensor_rank) |i| {
                if (i != take_axis and idx_raw.shape.at(i) != raw.shape.at(i)) return TensorError.ShapeMismatch;
            }
            const idx_buf = try hostIndexBuffer(ctx, try idx_raw.dataConstChecked(), raw.shape.at(take_axis));
            defer ctx.allocator.free(idx_buf);
            var value = try ctx.takeAlongAxisRank(tag_rank, raw, take_axis, idx_buf, idx_raw.shape.at(take_axis));
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), TakeAlongBackward(tags, take_axis), .{ ctx.allocator, self.grad_state, &self.value, idx_buf });
        }

        /// Functional elementwise scatter-add along `tag`
        /// (torch.scatter_add): a copy of `self` with `src[.., i, ..]`
        /// added at row `indices[.., i, ..]` of `tag` — duplicate indices
        /// accumulate. `indices` follows the `takeAlongAxis` convention
        /// and must be shaped exactly like `src`; both match `self` on
        /// every other axis. Serial deterministic kernel; differentiable
        /// in both: d/dself is the identity, d/dsrc gathers the written
        /// slots (`takeAlongAxis` of the upstream gradient).
        pub fn scatterAdd(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: anytype, src: *const Self) !Self {
            return scatterAlongImpl(self, ctx, tag, indices, src, true);
        }

        /// Functional elementwise scatter-overwrite along `tag`
        /// (torch.scatter with a tensor source): like `scatterAdd` but
        /// writing — duplicate indices resolve deterministically to the
        /// LAST write in row-major `src` order (torch leaves the order
        /// unspecified; this pins it). Differentiable in both: d/dself is
        /// zeroed at every written slot, d/dsrc gathers the written slots
        /// (the torch formula — on duplicates every writer receives the
        /// winning slot's gradient).
        pub fn scatter(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: anytype, src: *const Self) !Self {
            return scatterAlongImpl(self, ctx, tag, indices, src, false);
        }

        fn scatterAlongImpl(self: *const Self, ctx: *ExecContext, comptime tag: Tag, indices: anytype, src: *const Self, comptime accumulate: bool) !Self {
            comptime {
                if (TensorObject(@TypeOf(indices)).dtype != .i64)
                    @compileError("scatter/scatterAdd expect i64 indices (the argmax/topK/sort output dtype)");
            }
            const scatter_axis = comptime axis(tag);
            const idx_raw = indices.asRawTensor();
            const src_raw = src.asRawTensor();
            const raw = self.asRawTensor();
            inline for (0..tensor_rank) |i| {
                if (idx_raw.shape.at(i) != src_raw.shape.at(i)) return TensorError.ShapeMismatch;
            }
            const idx_buf = try hostIndexBuffer(ctx, try idx_raw.dataConstChecked(), raw.shape.at(scatter_axis));
            defer ctx.allocator.free(idx_buf);
            var value = if (comptime accumulate)
                try ctx.scatterAddAlongAxisRank(tag_rank, raw, src_raw, scatter_axis, idx_buf)
            else
                try ctx.scatterAlongAxisRank(tag_rank, raw, src_raw, scatter_axis, idx_buf);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or src.requiresGrad(), ScatterAlongBackward(tags, scatter_axis, accumulate), .{ ctx.allocator, self.grad_state, src.grad_state, idx_buf, src_raw.shape.at(scatter_axis) });
        }
    };
}
