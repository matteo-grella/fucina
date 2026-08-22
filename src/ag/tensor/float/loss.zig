//! f32 tensor methods: loss heads. A mixin over the ag
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

        pub fn crossEntropy(self: *const Self, ctx: *ExecContext, comptime class_tag: Tag, labels: []const usize) !Tensor(.{}) {
            const class_axis = comptime axis(class_tag);
            const row_stats = try rowStatsAlloc(ctx, self.requiresGrad(), labels.len);
            defer if (row_stats) |stats| ctx.allocator.free(stats);
            var value = try ctx.crossEntropyLossExStatsAxisRank(tag_rank, self.asRawTensor(), class_axis, labels, .{}, row_stats);
            errdefer value.deinit();
            return finishOp(.{}, ctx, value, self.requiresGrad(), CrossEntropyBackward(tags, class_axis), .{ ctx.allocator, self.grad_state, self.asRawTensor(), labels, row_stats orelse &[_]f32{} });
        }

        /// Cross-entropy with PyTorch-parity options (ignore_index, reduction,
        /// label smoothing). `.mean`/`.sum` return a scalar like `crossEntropy`;
        /// `.none` returns per-position losses with `class_tag` removed (same
        /// tag-removal rule as `sum`/`mean`).
        pub fn crossEntropyExt(
            self: *const Self,
            ctx: *ExecContext,
            comptime class_tag: Tag,
            labels: []const usize,
            comptime options: exec_mod.CrossEntropyOptions,
        ) !Tensor(if (options.reduction == .none) removeTag(tags, class_tag) else .{}) {
            const result_tags = comptime if (options.reduction == .none) removeTag(tags, class_tag) else .{};
            const class_axis = comptime axis(class_tag);
            const row_stats = try rowStatsAlloc(ctx, self.requiresGrad(), labels.len);
            defer if (row_stats) |stats| ctx.allocator.free(stats);
            var value = try ctx.crossEntropyLossExStatsAxisRank(tag_rank, self.asRawTensor(), class_axis, labels, options, row_stats);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), CrossEntropyExtBackward(tags, class_axis, options), .{ ctx.allocator, self.grad_state, self.asRawTensor(), labels, row_stats orelse &[_]f32{} });
        }

        /// Fused linear + cross-entropy: `crossEntropyExt(self·weightᵀ)` as
        /// ONE differentiable op. `self` is [row, shared] and `weight` is
        /// [class, shared] (both rank-2, shared tag last, f32). The logits
        /// exist only inside the op — computed once and saved on the
        /// backward record with the forward's per-row softmax statistics —
        /// and the VJP folds block-built probability panels straight into
        /// dx and dweight, so the [rows, classes] logit GRADIENT is never
        /// materialized (see `linearCrossEntropyBackwardUpstream`).
        /// Differentiable in BOTH operands. Reduction contract as
        /// `crossEntropyExt`: `.mean`/`.sum` return a scalar, `.none` the
        /// per-row losses tagged by the row tag.
        pub fn linearCrossEntropyExt(
            self: *const Self,
            ctx: *ExecContext,
            weight: anytype,
            labels: []const usize,
            comptime options: exec_mod.CrossEntropyOptions,
        ) !Tensor(if (options.reduction == .none) removeTag(tags, tags[1]) else .{}) {
            const Other = TensorObject(@TypeOf(weight));
            comptime {
                if (tag_rank != 2 or Other.axis_tags.len != 2) @compileError("linearCrossEntropy requires rank-2 [row, shared] x and [class, shared] weight");
                if (Other.dtype != .f32) @compileError("linearCrossEntropy requires an f32 weight (quantized/f16 arms are not routed)");
                if (tags[1] != Other.axis_tags[1]) @compileError("linearCrossEntropy requires the shared tag LAST on both operands");
                if (Other.axis_tags[0] == tags[0] or Other.axis_tags[0] == tags[1]) @compileError("linearCrossEntropy weight class tag must not appear on x");
            }
            const result_tags = comptime if (options.reduction == .none) removeTag(tags, tags[1]) else .{};
            const weight_ptr = tensorObjectPtrFrom(@TypeOf(weight), &weight);
            const wants_grad = self.requiresGrad() or weight_ptr.requiresGrad();
            const row_stats = try rowStatsAlloc(ctx, wants_grad, labels.len);
            defer if (row_stats) |stats| ctx.allocator.free(stats);
            var logits = try ctx.matmulTransB(self.asRawTensor(), weight_ptr.asRawTensor());
            defer logits.deinit();
            var value = try ctx.crossEntropyLossExStatsAxisRank(2, &logits, 1, labels, options, row_stats);
            errdefer value.deinit();
            return finishOp(
                result_tags,
                ctx,
                value,
                wants_grad,
                LinearCrossEntropyBackward(options),
                .{ ctx.allocator, self.grad_state, weight_ptr.grad_state, self.asRawTensor(), weight_ptr.asRawTensor(), &logits, labels, row_stats orelse &[_]f32{} },
            );
        }

        /// Fused linear + sparse-soft-target distillation loss:
        /// cross-entropy of `self·weightᵀ` against per-entry
        /// (row, class, prob) soft targets, as ONE differentiable op —
        /// `loss = reduce_i probs[i]·(LSE(logits[rows[i]]) −
        /// logits[rows[i], classes[i]])`. Only the UNIQUE rows named by
        /// `rows` are projected (rows without entries never produce
        /// logits), the selected-row logits live solely on the backward
        /// record with the forward's per-row softmax statistics, and the
        /// VJP consumes them in place — the full [row, class] block never
        /// enters the graph and its gradient never costs a second buffer.
        /// `self` is [row, shared] and `weight` [class, shared] (both
        /// rank-2, shared tag last, f32). `probs[i]` weights entry i (a
        /// teacher probability in the distillation use; truncated tail
        /// mass deliberately NOT renormalized). `options.reduction`
        /// reduces over ENTRIES and `options.loss_scale` multiplies the
        /// scalar result (the gradient-accumulation knob). Differentiable
        /// in BOTH operands; the record is single-use like
        /// `linearCrossEntropyExt`.
        pub fn linearDistillExt(
            self: *const Self,
            ctx: *ExecContext,
            weight: anytype,
            rows: []const usize,
            classes: []const usize,
            probs: []const f32,
            options: exec_mod.LinearDistillOptions,
        ) !Tensor(.{}) {
            const Other = TensorObject(@TypeOf(weight));
            comptime {
                if (tag_rank != 2 or Other.axis_tags.len != 2) @compileError("linearDistill requires rank-2 [row, shared] x and [class, shared] weight");
                if (Other.dtype != .f32) @compileError("linearDistill requires an f32 weight (quantized/f16 arms are not routed)");
                if (tags[1] != Other.axis_tags[1]) @compileError("linearDistill requires the shared tag LAST on both operands");
                if (Other.axis_tags[0] == tags[0] or Other.axis_tags[0] == tags[1]) @compileError("linearDistill weight class tag must not appear on x");
            }
            const weight_ptr = tensorObjectPtrFrom(@TypeOf(weight), &weight);
            const wants_grad = self.requiresGrad() or weight_ptr.requiresGrad();
            var fwd = try ctx.linearDistillLossStats(self.asRawTensor(), weight_ptr.asRawTensor(), rows, classes, probs, options);
            // finishOp takes ownership of the scalar; everything else is
            // cloned/duped onto the record and released here.
            defer {
                fwd.logits.deinit();
                fwd.x_sel.deinit();
                ctx.allocator.free(fwd.sel_rows);
                ctx.allocator.free(fwd.local_rows);
                ctx.allocator.free(fwd.row_stats);
            }
            errdefer fwd.value.deinit();
            return finishOp(
                .{},
                ctx,
                fwd.value,
                wants_grad,
                LinearDistillBackward,
                .{
                    ctx.allocator,
                    self.grad_state,
                    weight_ptr.grad_state,
                    &fwd.x_sel,
                    weight_ptr.asRawTensor(),
                    &fwd.logits,
                    fwd.sel_rows,
                    fwd.local_rows,
                    classes,
                    probs,
                    fwd.row_stats,
                    self.asRawTensor().shape.at(0),
                    options,
                },
            );
        }

        /// Mean-squared-error loss vs a same-tagged `target` (torch F.mse_loss):
        /// per-element (x - t)². `.mean` (the default) divides by the TOTAL
        /// element count; `.none` returns input-shaped per-element losses
        /// (same reduction-dependent result type as `crossEntropyExt`).
        /// Differentiable in BOTH self and target.
        pub fn mseLoss(
            self: *const Self,
            ctx: *ExecContext,
            target: *const Self,
            comptime options: exec_mod.MseOptions,
        ) !Tensor(if (options.reduction == .none) tags else .{}) {
            const result_tags = comptime if (options.reduction == .none) tags else .{};
            var value = try ctx.mseLoss(self.asRawTensor(), target.asRawTensor(), options);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad() or target.requiresGrad(), MseLossBackward(tags, options), .{ ctx.allocator, self.grad_state, target.grad_state, self.asRawTensor(), target.asRawTensor() });
        }

        /// Huber loss vs a same-tagged `target` (torch F.huber_loss): quadratic
        /// for |x - t| <= delta, linear beyond. Differentiable in BOTH self and
        /// target. Reduction/result-type contract as `mseLoss`.
        pub fn huberLoss(
            self: *const Self,
            ctx: *ExecContext,
            target: *const Self,
            comptime options: exec_mod.HuberOptions,
        ) !Tensor(if (options.reduction == .none) tags else .{}) {
            const result_tags = comptime if (options.reduction == .none) tags else .{};
            var value = try ctx.huberLoss(self.asRawTensor(), target.asRawTensor(), options);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad() or target.requiresGrad(), HuberLossBackward(tags, options), .{ ctx.allocator, self.grad_state, target.grad_state, self.asRawTensor(), target.asRawTensor() });
        }

        /// Binary cross-entropy vs a same-tagged `target`. With
        /// `options.from_logits` self holds raw logits and the loss uses the
        /// numerically stable max(x,0) - x·y + log1p(exp(-|x|)) formulation
        /// (torch F.binary_cross_entropy_with_logits); otherwise self holds
        /// probabilities, clamped per `exec.bce_eps` (see `exec/loss.zig`).
        /// Differentiable in BOTH self and target. Reduction/result-type
        /// contract as `mseLoss`.
        pub fn bceLoss(
            self: *const Self,
            ctx: *ExecContext,
            target: *const Self,
            comptime options: exec_mod.BceOptions,
        ) !Tensor(if (options.reduction == .none) tags else .{}) {
            const result_tags = comptime if (options.reduction == .none) tags else .{};
            var value = try ctx.bceLoss(self.asRawTensor(), target.asRawTensor(), options);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad() or target.requiresGrad(), BceLossBackward(tags, options), .{ ctx.allocator, self.grad_state, target.grad_state, self.asRawTensor(), target.asRawTensor() });
        }

        /// Pointwise KL divergence vs a same-tagged `target` (torch F.kl_div
        /// semantics): self holds LOG-probabilities; `target` holds
        /// probabilities, or log-probabilities with `options.log_target`.
        /// NOTE: no `.batchmean` — `.mean` divides by the TOTAL element count
        /// (see `exec.KlDivOptions`). Differentiable in BOTH self and target
        /// (the target gradient at a zero-mass probability entry is defined
        /// as 0). Reduction/result-type contract as `mseLoss`.
        pub fn klDivLoss(
            self: *const Self,
            ctx: *ExecContext,
            target: *const Self,
            comptime options: exec_mod.KlDivOptions,
        ) !Tensor(if (options.reduction == .none) tags else .{}) {
            const result_tags = comptime if (options.reduction == .none) tags else .{};
            var value = try ctx.klDivLoss(self.asRawTensor(), target.asRawTensor(), options);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad() or target.requiresGrad(), KlDivLossBackward(tags, options), .{ ctx.allocator, self.grad_state, target.grad_state, self.asRawTensor(), target.asRawTensor() });
        }

        /// Negative log-likelihood over `class_tag` (torch F.nll_loss with unit
        /// weights): self holds LOG-probabilities; per-position loss is
        /// -logp[position, labels[position]], positions ordered like
        /// `crossEntropy` labels (class axis removed, remaining axes
        /// row-major). `.mean` divides by the position count.
        ///
        /// Thin composed convenience (one-hot constant → mul → sum → negate →
        /// reduction), differentiable in self through those ops — PREFER
        /// `crossEntropy`/`crossEntropyExt` (fused log-softmax + NLL) when
        /// starting from logits; this exists for pipelines that already carry
        /// log-probabilities. When gradients are tracked this requires an
        /// active exec scope (the training pattern — the composition's
        /// intermediate graph nodes must be scope-owned to survive until
        /// backward); errors with `ActiveExecScopeRequired` otherwise.
        pub fn nllLoss(
            self: *const Self,
            ctx: *ExecContext,
            comptime class_tag: Tag,
            labels: []const usize,
            comptime reduction: exec_mod.Reduction,
        ) !Tensor(if (reduction == .none) removeTag(tags, class_tag) else .{}) {
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const class_axis = comptime axis(class_tag);
            const raw = self.asRawTensor();
            var raw_shape: [tensor_rank]usize = undefined;
            inline for (0..tensor_rank) |i| raw_shape[i] = raw.shape.at(i);
            const class_count = raw_shape[class_axis];
            var inner: usize = 1;
            for (class_axis + 1..tensor_rank) |i| inner *= raw_shape[i];
            var outer: usize = 1;
            for (0..class_axis) |i| outer *= raw_shape[i];
            const position_count = outer * inner;
            if (labels.len != position_count) return TensorError.InvalidDataLength;

            // Sparse marks into a zeroed tensor — the `oneHot` constructor's
            // shape; a dense scratch here would copy the full logits size.
            var one_hot = try Self.zeros(ctx, raw_shape);
            defer one_hot.deinit();
            const one_hot_values = try one_hot.data();
            for (0..outer) |outer_i| {
                for (0..inner) |inner_i| {
                    const label = labels[outer_i * inner + inner_i];
                    if (label >= class_count) return TensorError.IndexOutOfBounds;
                    one_hot_values[(outer_i * class_count + label) * inner + inner_i] = 1;
                }
            }

            var picked = try self.mul(ctx, &one_hot);
            defer picked.deinit();
            var picked_sum = try picked.sum(ctx, class_tag);
            defer picked_sum.deinit();
            if (comptime reduction == .none) {
                return picked_sum.neg(ctx);
            }
            var total = try picked_sum.sumAll(ctx);
            defer total.deinit();
            const denom: f32 = if (comptime reduction == .mean) @floatFromInt(position_count) else 1;
            return total.scale(ctx, -1.0 / denom);
        }
    };
}
