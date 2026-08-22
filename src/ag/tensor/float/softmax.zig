//! f32 tensor methods: softmax family. A mixin over the ag
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

        /// Softmax over `tag`, with optional fused extensions selected at
        /// comptime from the `options` struct literal (unknown fields are
        /// compile errors): `.scale` (logit multiplier), `.max_bias` (ALiBi
        /// slope base, needs `.head_tag`), `.sinks` (per-head attention
        /// sinks, needs `.head_tag`), `.causal = .{ .query_tag,
        /// .source_offset }`, `.mask` (additive tag-broadcast tensor; must
        /// not require grad). An empty `.{}` routes to the lean plain kernel
        /// (the backward is already unified at the exec layer).
        /// Log-sum-exp over `tag` (torch.logsumexp): `log(Σ exp(x))` with
        /// `tag` removed — a FUSED single-pass kernel (SIMD max scan +
        /// vexpf sum per row, task-parallel over rows like `softmax`; no
        /// materialized intermediates, no exec-scope requirement). Rows
        /// whose max is ±inf are shifted by 0 instead, so an all(-inf) row
        /// yields -inf and a row containing +inf yields +inf (the torch
        /// convention) rather than NaN. Differentiable: the backward is
        /// the saved-output identity `exp(x − lse)·g` (the row softmax).
        pub fn logsumexp(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            const result_tags = removeTag(tags, tag);
            const reduce_axis = comptime axis(tag);
            var value = try ctx.logsumexpAxisRank(tag_rank, self.asRawTensor(), reduce_axis);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), LogsumexpBackward(tags, reduce_axis), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        /// Log-softmax over `tag` (torch.log_softmax): `x − logsumexp(x)`
        /// broadcast, shape-preserving — the same FUSED kernel family as
        /// `logsumexp` (two SIMD passes per row, task-parallel; no
        /// exec-scope requirement) with the same non-finite max handling.
        /// Prefer `crossEntropy` when the next step is an NLL loss (fused
        /// with the loss, saved-stats backward). Differentiable: the
        /// backward is the saved-output identity `g − exp(y)·Σg`.
        pub fn logSoftmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Self {
            const scan_axis = comptime axis(tag);
            var value = try ctx.logSoftmaxAxisRank(tag_rank, self.asRawTensor(), scan_axis);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), LogSoftmaxBackward(tags, scan_axis), .{ ctx.allocator, self.grad_state, &value });
        }

        pub fn softmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag, options: anytype) !Self {
            const Options = @TypeOf(options);
            comptime {
                if (@typeInfo(Options) != .@"struct") @compileError("softmax: options must be a struct literal, e.g. .{} or .{ .scale = s }");
                const allowed = [_][]const u8{ "scale", "max_bias", "sinks", "head_tag", "causal", "mask" };
                for (@typeInfo(Options).@"struct".fields) |field| {
                    var known = false;
                    for (allowed) |name| {
                        if (std.mem.eql(u8, field.name, name)) known = true;
                    }
                    if (!known) @compileError("softmax: unknown option ." ++ field.name);
                }
            }
            const softmax_axis = comptime axis(tag);
            if (comptime @typeInfo(Options).@"struct".fields.len == 0) {
                var value = try ctx.softmaxAxisRank(tag_rank, self.asRawTensor(), softmax_axis);
                errdefer value.deinit();
                return finishOp(tags, ctx, value, self.requiresGrad(), SoftmaxBackward(tags, softmax_axis), .{ ctx.allocator, self.grad_state, &value });
            }
            const scale_value: f32 = if (comptime @hasField(Options, "scale")) options.scale else 1;
            const max_bias: f32 = if (comptime @hasField(Options, "max_bias")) options.max_bias else 0;
            const sinks: ?[]const f32 = if (comptime @hasField(Options, "sinks")) options.sinks else null;
            const head_axis: ?usize = comptime if (@hasField(Options, "head_tag")) axis(options.head_tag) else null;
            const causal_query_axis: ?usize = comptime if (@hasField(Options, "causal")) blk: {
                const Causal = @TypeOf(options.causal);
                for (@typeInfo(Causal).@"struct".fields) |field| {
                    if (!std.mem.eql(u8, field.name, "query_tag") and !std.mem.eql(u8, field.name, "source_offset"))
                        @compileError("softmax: unknown .causal option ." ++ field.name);
                }
                if (!@hasField(Causal, "query_tag")) @compileError("softmax: the .causal option requires .query_tag");
                break :blk axis(options.causal.query_tag);
            } else null;
            const causal_source_offset: usize = if (comptime @hasField(Options, "causal")) blk: {
                const Causal = @TypeOf(options.causal);
                break :blk if (comptime @hasField(Causal, "source_offset")) options.causal.source_offset else 0;
            } else 0;

            var mask_view: ?RawTensor = null;
            defer if (mask_view) |*mask| mask.deinit();
            if (comptime @hasField(Options, "mask")) {
                const mask_ptr = tensorObjectPtrFrom(@TypeOf(options.mask), &options.mask);
                if (mask_ptr.requiresGrad()) return error.UnsupportedGradient;
                const Mask = TensorObject(@TypeOf(options.mask));
                mask_view = try broadcastTensorTo(Mask.axis_tags, mask_ptr.asRawTensor(), tags, self.shape());
            }

            var value = try ctx.softmaxExtAxisRank(
                tag_rank,
                self.asRawTensor(),
                softmax_axis,
                SoftmaxExtOptions{
                    .mask = if (mask_view) |*mask| mask else null,
                    .sinks = sinks,
                    .scale = scale_value,
                    .max_bias = max_bias,
                    .head_axis = head_axis,
                    .causal_query_axis = causal_query_axis,
                    .causal_source_offset = causal_source_offset,
                },
            );
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), SoftmaxExtBackward(tags, softmax_axis), .{ ctx.allocator, self.grad_state, &value, scale_value });
        }
    };
}
