//! f32 tensor methods: pooling and nearest-neighbor upsampling. A mixin over the ag
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

        /// 2-D max pool over a channel-last rank-3 `[H, W, C]` tensor
        /// (`kernel`/`stride`/`padding` in `[h, w]` order; the zero-pad
        /// border reads as −inf). Tags are preserved.
        pub fn maxPool2d(self: *const Self, ctx: *ExecContext, kernel: [2]usize, stride: [2]usize, padding: [2]usize) !Self {
            var value = try ctx.maxPool2d(self.asRawTensor(), kernel, stride, padding);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), MaxPool2dBackward, .{ ctx.allocator, self.grad_state, self.asRawTensor(), kernel, stride, padding });
        }

        /// 2-D average pool over a channel-last rank-3 `[H, W, C]` tensor;
        /// averages the valid taps only (ONNX `count_include_pad=0`).
        pub fn avgPool2d(self: *const Self, ctx: *ExecContext, kernel: [2]usize, stride: [2]usize, padding: [2]usize) !Self {
            var value = try ctx.avgPool2d(self.asRawTensor(), kernel, stride, padding);
            errdefer value.deinit();
            const raw = self.asRawTensor();
            return finishOp(tags, ctx, value, self.requiresGrad(), AvgPool2dBackward, .{ ctx.allocator, self.grad_state, raw.shape.at(0), raw.shape.at(1), kernel, stride, padding });
        }

        /// 2× nearest-neighbour upsample of a channel-last rank-3 tensor:
        /// `[H, W, C]` → `[2H, 2W, C]` (VJP = 2×2 stride-2 sum-pool).
        pub fn upsample2xNearest(self: *const Self, ctx: *ExecContext) !Self {
            var value = try ctx.upsample2xNearest(self.asRawTensor());
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), Upsample2xNearestBackward, .{ ctx.allocator, self.grad_state });
        }
    };
}
