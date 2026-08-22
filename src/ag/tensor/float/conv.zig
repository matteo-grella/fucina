//! f32 tensor methods: convolutions (1d/2d, causal, transpose) and unfold/fold. A mixin over the ag
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

        /// No-grad 2-D convolution. `self` is the rank-3 input `[H, W, Cin]`
        /// (channels-last), `weight` is rank-4 `[Cout, kH, kW, Cin/groups]`,
        /// `bias` is `null` or a rank-1 `[Cout]` tensor. Result `[oH, oW, Cout]`
        /// is tagged `out_tags`.
        pub fn conv2d(
            self: *const Self,
            ctx: *ExecContext,
            weight: anytype,
            bias: anytype,
            stride: [2]usize,
            padding: [2]usize,
            groups: usize,
            comptime out_tags: anytype,
        ) !Tensor(out_tags) {
            const weight_ptr = tensorObjectPtrFrom(@TypeOf(weight), &weight);
            var any_grad = self.requiresGrad() or weight_ptr.requiresGrad();
            const bias_raw: ?*const RawTensor = if (@TypeOf(bias) == @TypeOf(null)) null else brk: {
                const bias_ptr = tensorObjectPtrFrom(@TypeOf(bias), &bias);
                any_grad = any_grad or bias_ptr.requiresGrad();
                break :brk bias_ptr.asRawTensor();
            };
            const bias_grad_state: ?*GradState = if (@TypeOf(bias) == @TypeOf(null)) null else tensorObjectPtrFrom(@TypeOf(bias), &bias).grad_state;

            var value = try ctx.conv2d(self.asRawTensor(), weight_ptr.asRawTensor(), bias_raw, stride, padding, groups);
            errdefer value.deinit();
            return finishOp(out_tags, ctx, value, any_grad, Conv2dBackward, .{ ctx.allocator, self.grad_state, weight_ptr.grad_state, bias_grad_state, self.asRawTensor(), weight_ptr.asRawTensor(), stride, padding, groups });
        }

        /// conv2d + relu with the relu fused into the conv epilogue on the
        /// no-grad path (identical values to `conv2d(...)` then `relu` —
        /// the same single max(0,·) on the same numbers; on the Winograd
        /// route it folds into the output transform). When any operand
        /// requires gradients, falls back to the differentiable composition.
        pub fn conv2dRelu(
            self: *const Self,
            ctx: *ExecContext,
            weight: anytype,
            bias: anytype,
            stride: [2]usize,
            padding: [2]usize,
            groups: usize,
            comptime out_tags: anytype,
        ) !Tensor(out_tags) {
            const weight_ptr = tensorObjectPtrFrom(@TypeOf(weight), &weight);
            var any_grad = self.requiresGrad() or weight_ptr.requiresGrad();
            if (@TypeOf(bias) != @TypeOf(null)) {
                any_grad = any_grad or tensorObjectPtrFrom(@TypeOf(bias), &bias).requiresGrad();
            }
            if (any_grad) {
                var y = try self.conv2d(ctx, weight, bias, stride, padding, groups, out_tags);
                defer y.deinit();
                return y.relu(ctx);
            }
            const bias_raw: ?*const RawTensor = if (@TypeOf(bias) == @TypeOf(null)) null else tensorObjectPtrFrom(@TypeOf(bias), &bias).asRawTensor();
            var value = try ctx.conv2dRelu(self.asRawTensor(), weight_ptr.asRawTensor(), bias_raw, stride, padding, groups);
            errdefer value.deinit();
            return finishOp(out_tags, ctx, value, false, Conv2dBackward, .{ ctx.allocator, null, null, null, self.asRawTensor(), weight_ptr.asRawTensor(), stride, padding, groups });
        }

        /// Load-time Winograd weight preparation for this rank-4
        /// `[Cout, kH, kW, Cin/groups]` conv weight: builds the F2/F4
        /// weight-transform planes once so `conv2dPrepared` can skip the
        /// per-call weight transform. Returns `.empty` — inert on every conv
        /// route — when the weight can never take the Winograd route. No
        /// gradient support (the `dotPacked` policy; prepared planes live
        /// outside the graph): fails with
        /// `error.GradientPreparedConv2dUnsupported` when the weight
        /// requires grad.
        pub fn prepareConv2dWeights(self: *const Self, ctx: *ExecContext) !exec_mod.ExecContext.PreparedConvWeights {
            comptime if (tag_rank != 4) @compileError("prepareConv2dWeights requires a rank-4 [cout, kh, kw, cin] conv weight");
            if (self.requiresGrad()) return error.GradientPreparedConv2dUnsupported;
            return ctx.prepareConv2dWeights(self.asRawTensor());
        }

        /// No-grad conv2d against load-time prepared Winograd weight planes
        /// (see `prepareConv2dWeights`): bitwise-identical values to
        /// `conv2d`, minus the per-call weight transform on the Winograd
        /// route; every other route ignores `prepared` (`.empty` is always
        /// inert). No gradient support (same policy as `dotPacked`): fails
        /// with `error.GradientPreparedConv2dUnsupported` when any operand
        /// requires grad.
        pub fn conv2dPrepared(
            self: *const Self,
            ctx: *ExecContext,
            weight: anytype,
            prepared: *const exec_mod.ExecContext.PreparedConvWeights,
            bias: anytype,
            stride: [2]usize,
            padding: [2]usize,
            groups: usize,
            comptime out_tags: anytype,
        ) !Tensor(out_tags) {
            const weight_ptr = tensorObjectPtrFrom(@TypeOf(weight), &weight);
            var any_grad = self.requiresGrad() or weight_ptr.requiresGrad();
            const bias_raw: ?*const RawTensor = if (@TypeOf(bias) == @TypeOf(null)) null else brk: {
                const bias_ptr = tensorObjectPtrFrom(@TypeOf(bias), &bias);
                any_grad = any_grad or bias_ptr.requiresGrad();
                break :brk bias_ptr.asRawTensor();
            };
            if (any_grad) return error.GradientPreparedConv2dUnsupported;
            var value = try ctx.conv2dPrepared(self.asRawTensor(), weight_ptr.asRawTensor(), prepared, bias_raw, stride, padding, groups);
            errdefer value.deinit();
            return finishNoGrad(out_tags, ctx, value);
        }

        /// `conv2dPrepared` + relu fused into the conv epilogue (identical
        /// values to `conv2dPrepared` followed by `relu`; on the Winograd
        /// route it folds into the output transform). Same no-grad contract
        /// as `conv2dPrepared`.
        pub fn conv2dPreparedRelu(
            self: *const Self,
            ctx: *ExecContext,
            weight: anytype,
            prepared: *const exec_mod.ExecContext.PreparedConvWeights,
            bias: anytype,
            stride: [2]usize,
            padding: [2]usize,
            groups: usize,
            comptime out_tags: anytype,
        ) !Tensor(out_tags) {
            const weight_ptr = tensorObjectPtrFrom(@TypeOf(weight), &weight);
            var any_grad = self.requiresGrad() or weight_ptr.requiresGrad();
            const bias_raw: ?*const RawTensor = if (@TypeOf(bias) == @TypeOf(null)) null else brk: {
                const bias_ptr = tensorObjectPtrFrom(@TypeOf(bias), &bias);
                any_grad = any_grad or bias_ptr.requiresGrad();
                break :brk bias_ptr.asRawTensor();
            };
            if (any_grad) return error.GradientPreparedConv2dUnsupported;
            var value = try ctx.conv2dPreparedRelu(self.asRawTensor(), weight_ptr.asRawTensor(), prepared, bias_raw, stride, padding, groups);
            errdefer value.deinit();
            return finishNoGrad(out_tags, ctx, value);
        }

        /// Patch extraction over a channel-last rank-3 `[H, W, C]` tensor
        /// (torch.nn.Unfold in this repo's layout): result
        /// `[oH·oW, kH·kW·C]` tagged `out_tags` (two tags: patch axis, then
        /// patch-element axis), where patch row `oy·oW + ox` holds output
        /// position (oy, ox)'s kernel-window taps ordered `(ky, kx, c)`
        /// with the channel fastest — the same col layout the conv2d GEMM
        /// route consumes, so `unfold` → `dot` over the element axis IS a
        /// dense conv2d (and a stride = kernel, pad 0 unfold is ViT
        /// patchify). Out-of-range (pad) taps read 0. `[h, w]`-ordered
        /// params like `maxPool2d`. Differentiable: the VJP is the exact
        /// adjoint `fold` (overlaps accumulate, pad taps drop).
        pub fn unfold(
            self: *const Self,
            ctx: *ExecContext,
            kernel: [2]usize,
            stride: [2]usize,
            padding: [2]usize,
            comptime out_tags: anytype,
        ) !Tensor(normalizeTags(out_tags)) {
            comptime {
                if (tag_rank != 3) @compileError("unfold takes a channel-last rank-3 [h, w, c] tensor");
                if (normalizeTags(out_tags).len != 2) @compileError("unfold produces a rank-2 [patch, element] tensor: pass exactly two out tags");
            }
            var value = try ctx.unfold(self.asRawTensor(), kernel, stride, padding);
            errdefer value.deinit();
            return finishOp(normalizeTags(out_tags), ctx, value, self.requiresGrad(), UnfoldBackward, .{ ctx.allocator, self.grad_state, self.asRawTensor(), kernel, stride, padding });
        }

        /// Adjoint of `unfold` (torch.nn.Fold): scatter-ADD rank-2
        /// `[oH·oW, kH·kW·C]` patch rows back into a channel-last
        /// `[H, W, C]` image tagged `out_tags` (three tags). Overlapping
        /// taps ACCUMULATE (`fold(unfold(x))` multiplies each position by
        /// its window count) and pad taps are dropped; the channel count
        /// is derived from the patch-element width. The patch row count
        /// must match the `output_size`/`kernel`/`stride`/`padding`
        /// geometry (`ShapeMismatch` otherwise). Differentiable: the VJP
        /// is the exact adjoint `unfold`.
        pub fn fold(
            self: *const Self,
            ctx: *ExecContext,
            output_size: [2]usize,
            kernel: [2]usize,
            stride: [2]usize,
            padding: [2]usize,
            comptime out_tags: anytype,
        ) !Tensor(normalizeTags(out_tags)) {
            comptime {
                if (tag_rank != 2) @compileError("fold takes a rank-2 [patch, element] tensor (the unfold layout)");
                if (normalizeTags(out_tags).len != 3) @compileError("fold produces a channel-last rank-3 [h, w, c] tensor: pass exactly three out tags");
            }
            var value = try ctx.fold(self.asRawTensor(), output_size, kernel, stride, padding);
            errdefer value.deinit();
            return finishOp(normalizeTags(out_tags), ctx, value, self.requiresGrad(), FoldBackward, .{ ctx.allocator, self.grad_state, kernel, stride, padding });
        }

        /// Depthwise causal 1-D convolution:
        /// `y[t, c] = Σ_k x[t − dilation·(taps−1−k), c] · w[c, k]` — tap
        /// `taps−1` is the newest sample. `kernel` is `[channel, tap]`.
        /// `state`, when given, supplies the `dilation·(taps−1)` input rows
        /// preceding `x` (oldest first, layout `[row, channel]`); absent
        /// rows read as zeros and no gradient flows into `state`.
        pub fn causalDepthwiseConv1d(
            self: *const Self,
            ctx: *ExecContext,
            comptime time_tag: Tag,
            comptime channel_tag: Tag,
            comptime tap_tag: Tag,
            kernel: *const Tensor(.{ channel_tag, tap_tag }),
            dilation: usize,
            state: ?[]const f32,
        ) !Self {
            const time_axis = comptime axis(time_tag);
            const channel_axis = comptime axis(channel_tag);
            comptime {
                if (tag_rank != 2) @compileError("causalDepthwiseConv1d requires a rank-2 input");
                if (time_axis != 0 or channel_axis != 1) {
                    @compileError("causalDepthwiseConv1d requires input storage order [time, channel]");
                }
            }

            var value = try ctx.causalDepthwiseConv1dAxisRank(tag_rank, self.asRawTensor(), kernel.asRawTensor(), time_axis, channel_axis, dilation, state);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or kernel.requiresGrad(), CausalDepthwiseConv1dBackward(tags, .{ channel_tag, tap_tag }, time_axis, channel_axis), .{ ctx.allocator, self.grad_state, kernel.grad_state, self.asRawTensor(), kernel.asRawTensor(), dilation, state });
        }

        /// General causal 1-D convolution mixing channels:
        /// `y[t, o] = Σ_{k, i} x[t − dilation·(taps−1−k), i] · w[k, i, o]`
        /// — tap `taps−1` is the newest sample (the PyTorch/causal-pad
        /// orientation). `weight` is stored `[tap, in, out]`. `state`, when
        /// given, supplies the `dilation·(taps−1)` input rows preceding `x`
        /// (oldest first, layout `[row, in]`); absent rows read as zeros and
        /// no gradient flows into `state`. Bias is deliberately not fused —
        /// compose it with broadcast `add`, whose backward already reduces
        /// to the bias tag.
        pub fn causalConv1d(
            self: *const Self,
            ctx: *ExecContext,
            comptime time_tag: Tag,
            comptime in_tag: Tag,
            comptime tap_tag: Tag,
            comptime out_tag: Tag,
            weight: *const Tensor(.{ tap_tag, in_tag, out_tag }),
            dilation: usize,
            state: ?[]const f32,
        ) !Tensor(.{ time_tag, out_tag }) {
            const time_axis = comptime axis(time_tag);
            const channel_axis = comptime axis(in_tag);
            comptime {
                if (tag_rank != 2) @compileError("causalConv1d requires a rank-2 input");
                if (time_axis != 0 or channel_axis != 1) {
                    @compileError("causalConv1d requires input storage order [time, in]");
                }
            }

            var value = try ctx.causalConv1dAxisRank(tag_rank, self.asRawTensor(), weight.asRawTensor(), time_axis, channel_axis, dilation, state);
            errdefer value.deinit();
            return finishOp(.{ time_tag, out_tag }, ctx, value, self.requiresGrad() or weight.requiresGrad(), CausalConv1dBackward(tags, .{ tap_tag, in_tag, out_tag }, time_axis, channel_axis), .{ ctx.allocator, self.grad_state, weight.grad_state, self.asRawTensor(), weight.asRawTensor(), dilation, state });
        }

        /// Grouped causal 1-D convolution. Input is `[time, in]`, output is
        /// `[time, out]`, and `weight` is `[tap, in_per_group, out]`.
        /// Output channel `o` reads only the input group implied by
        /// `o / (out / groups)`.
        pub fn groupedCausalConv1d(
            self: *const Self,
            ctx: *ExecContext,
            comptime time_tag: Tag,
            comptime in_tag: Tag,
            comptime tap_tag: Tag,
            comptime in_per_group_tag: Tag,
            comptime out_tag: Tag,
            weight: *const Tensor(.{ tap_tag, in_per_group_tag, out_tag }),
            dilation: usize,
            groups: usize,
            state: ?[]const f32,
        ) !Tensor(.{ time_tag, out_tag }) {
            const time_axis = comptime axis(time_tag);
            const channel_axis = comptime axis(in_tag);
            comptime {
                if (tag_rank != 2) @compileError("groupedCausalConv1d requires a rank-2 input");
                if (time_axis != 0 or channel_axis != 1) {
                    @compileError("groupedCausalConv1d requires input storage order [time, in]");
                }
            }

            var value = try ctx.groupedCausalConv1dAxisRank(tag_rank, self.asRawTensor(), weight.asRawTensor(), time_axis, channel_axis, dilation, groups, state);
            errdefer value.deinit();
            return finishOp(.{ time_tag, out_tag }, ctx, value, self.requiresGrad() or weight.requiresGrad(), GroupedCausalConv1dBackward(tags, .{ tap_tag, in_per_group_tag, out_tag }, time_axis, channel_axis), .{ ctx.allocator, self.grad_state, weight.grad_state, self.asRawTensor(), weight.asRawTensor(), dilation, groups, state });
        }

        /// General 1-D convolution (PyTorch Conv1d semantics — standard
        /// cross-correlation): `self` is `[time, in]`, `weight` is
        /// `[tap, in/groups, out]` stored `[tap_tag, in_tag, out_tag]`
        /// (out-channel contiguous, the causalConv1d layout family), result is
        /// `[t_out, out]` with
        /// `t_out = (T + 2*pad - dilation*(taps-1) - 1)/stride + 1`.
        /// The input is virtually zero-padded `pad` rows on BOTH sides.
        /// Differentiable in the input and the weight; bias composes via
        /// broadcast `add`, whose backward already reduces to the bias tag.
        pub fn conv1d(
            self: *const Self,
            ctx: *ExecContext,
            comptime time_tag: Tag,
            comptime in_tag: Tag,
            comptime tap_tag: Tag,
            comptime out_tag: Tag,
            weight: *const Tensor(.{ tap_tag, in_tag, out_tag }),
            stride: usize,
            padding: usize,
            dilation: usize,
            groups: usize,
        ) !Tensor(.{ time_tag, out_tag }) {
            const time_axis = comptime axis(time_tag);
            const channel_axis = comptime axis(in_tag);
            comptime {
                if (tag_rank != 2) @compileError("conv1d requires a rank-2 input");
                if (time_axis != 0 or channel_axis != 1) {
                    @compileError("conv1d requires input storage order [time, in]");
                }
            }

            var value = try ctx.conv1dAxisRank(tag_rank, self.asRawTensor(), weight.asRawTensor(), time_axis, channel_axis, stride, padding, dilation, groups);
            errdefer value.deinit();
            return finishOp(.{ time_tag, out_tag }, ctx, value, self.requiresGrad() or weight.requiresGrad(), Conv1dBackward(tags, .{ tap_tag, in_tag, out_tag }, time_axis, channel_axis), .{ ctx.allocator, self.grad_state, weight.grad_state, self.asRawTensor(), weight.asRawTensor(), stride, padding, dilation, groups });
        }

        /// ConvTranspose1d (GEMM + col2im_1d gather, the ggml decomposition):
        /// `self` is `[time, in]`; `weight2` is the load-time repacked
        /// `[K*OC, IC]` matrix with k varying fastest inside each oc block
        /// (`weight2[(oc*K + k)*IC + ic] = w_pt[ic][oc][k]` — exactly the
        /// omnivoice reference's repack of the PyTorch ConvTranspose1d weight
        /// `(IC, OC, K)`); optional `bias` is `[OC]`. Result is
        /// `[(T-1)*stride + K - 2*pad + output_pad, OC]`; the `output_pad`
        /// trailing time rows are bias-only — ggml/omnivoice.cpp convention; true
        /// PyTorch ConvTranspose1d fills them with kernel taps when pad > 0.
        /// Differentiable in the input, weight2, and bias; the weight gradient
        /// is wrt the PACKED `[K*OC, IC]` weight2 layout as passed (a trainer
        /// keeping the PyTorch `(IC, OC, K)` layout must map it itself).
        pub fn convTranspose1d(
            self: *const Self,
            ctx: *ExecContext,
            comptime time_tag: Tag,
            comptime in_tag: Tag,
            comptime kout_tag: Tag,
            comptime out_tag: Tag,
            weight2: *const Tensor(.{ kout_tag, in_tag }),
            bias: ?*const Tensor(.{out_tag}),
            out_channels: usize,
            taps: usize,
            stride: usize,
            padding: usize,
            output_pad: usize,
        ) !Tensor(.{ time_tag, out_tag }) {
            const time_axis = comptime axis(time_tag);
            const channel_axis = comptime axis(in_tag);
            comptime {
                if (tag_rank != 2) @compileError("convTranspose1d requires a rank-2 input");
                if (time_axis != 0 or channel_axis != 1) {
                    @compileError("convTranspose1d requires input storage order [time, in]");
                }
            }
            var any_grad = self.requiresGrad() or weight2.requiresGrad();
            var bias_parent: ?*GradState = null;
            const bias_raw: ?*const RawTensor = if (bias) |b| blk: {
                any_grad = any_grad or b.requiresGrad();
                bias_parent = b.grad_state;
                break :blk b.asRawTensor();
            } else null;

            var value = try ctx.convTranspose1d(self.asRawTensor(), weight2.asRawTensor(), bias_raw, out_channels, taps, stride, padding, output_pad);
            errdefer value.deinit();
            return finishOp(.{ time_tag, out_tag }, ctx, value, any_grad, ConvTranspose1dBackward(tags), .{ ctx.allocator, self.grad_state, weight2.grad_state, bias_parent, self.asRawTensor(), weight2.asRawTensor(), out_channels, taps, stride, padding });
        }
    };
}
