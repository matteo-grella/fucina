//! f32 tensor methods: variance/standardize, extrema, argmax, multinomial. A mixin over the ag
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

        /// Variance over `tag` (the tag is removed like sum/mean): ddof 0 =
        /// biased estimator (the LayerNorm convention), ddof 1 = unbiased
        /// (the torch.var default).
        pub fn variance(self: *const Self, ctx: *ExecContext, comptime tag: Tag, ddof: u1) !Tensor(removeTag(tags, tag)) {
            const result_tags = removeTag(tags, tag);
            const reduce_axis = comptime axis(tag);
            var value = try ctx.varAxisRank(tag_rank, self.asRawTensor(), reduce_axis, ddof);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), VarBackward(tags, reduce_axis), .{ ctx.allocator, self.grad_state, &self.value, ddof });
        }

        /// Standardize over `tag` while preserving shape:
        /// `y = (x - mean(tag)) / denom`. `options` accepts every
        /// `exec.StandardizeOptions` field (ddof/eps/eps_mode/accumulation —
        /// a plain `StandardizeOptions` value still coerces) plus an optional
        /// `.valid_len`: standardize only the first `valid_len` elements of
        /// `tag` — the suffix is masked out, returned as zeros, and receives
        /// zero gradient. Unknown fields are compile errors. Differentiable
        /// in `self`.
        pub fn standardizeAxis(self: *const Self, ctx: *ExecContext, comptime tag: Tag, options: anytype) !Self {
            const Options = @TypeOf(options);
            comptime {
                if (@typeInfo(Options) != .@"struct") @compileError("standardizeAxis: options must be a struct literal, e.g. .{ .ddof = 1 }");
                for (@typeInfo(Options).@"struct".fields) |field| {
                    if (!std.mem.eql(u8, field.name, "valid_len") and !@hasField(exec_mod.StandardizeOptions, field.name))
                        @compileError("standardizeAxis: unknown option ." ++ field.name);
                }
            }
            var exec_options = exec_mod.StandardizeOptions{};
            inline for (@typeInfo(exec_mod.StandardizeOptions).@"struct".fields) |field| {
                if (comptime @hasField(Options, field.name)) @field(exec_options, field.name) = @field(options, field.name);
            }
            const norm_axis = comptime axis(tag);
            if (comptime @hasField(Options, "valid_len")) {
                var value = try ctx.standardizeAxisValidPrefixRank(tag_rank, self.asRawTensor(), norm_axis, options.valid_len, exec_options);
                errdefer value.deinit();
                return finishOp(tags, ctx, value, self.requiresGrad(), StandardizeBackward(tags, norm_axis), .{ ctx.allocator, self.grad_state, &self.value, @as(?usize, options.valid_len), exec_options });
            }
            var value = try ctx.standardizeAxisRank(tag_rank, self.asRawTensor(), norm_axis, exec_options);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), StandardizeBackward(tags, norm_axis), .{ ctx.allocator, self.grad_state, &self.value, @as(?usize, null), exec_options });
        }

        /// Index of the row maximum along `tag` (torch.argmax over a dim):
        /// a constant i64 tensor, no gradient. Caller-owned even under an
        /// exec scope (the typed-constant ownership rule).
        pub fn argmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .i64, .tags = removeTag(tags, tag) }) {
            const result_tags = removeTag(tags, tag);
            var value = try ctx.argmaxAxisRank(tag_rank, self.asRawTensor(), axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = .i64, .tags = result_tags }).fromTensor(ctx, value);
        }

        /// Categorical sampling from UNNORMALIZED non-negative weight rows
        /// (torch.multinomial): `num_samples` draws per row along
        /// `class_tag` — which must be the last axis, like `routerTopK` —
        /// replaced by `out_tag` in the result. Rank 1 or 2 (the torch
        /// surface). Draw `(row, s)` reads the deterministic counter-based
        /// stream at `(seed, row·num_samples + s)` (§6.8), so results are
        /// reproducible and independent of any batching; pass a fresh seed
        /// per call. Rows must hold finite weights `>= 0` with a positive
        /// sum (`InvalidShape` otherwise; NaN/inf rejected). Without
        /// replacement each draw removes the chosen class's mass (torch
        /// semantics); `num_samples` beyond the class count — or beyond the
        /// row's nonzero classes — is `InvalidShape`. The result is a
        /// constant i64 tensor: no gradient, and CALLER-owned even under an
        /// exec scope (the typed-constant ownership rule).
        pub fn multinomial(
            self: *const Self,
            ctx: *ExecContext,
            comptime class_tag: Tag,
            comptime out_tag: Tag,
            num_samples: usize,
            seed: u64,
            replacement: bool,
        ) !Tensor(.{ .dtype = .i64, .tags = replaceTag(tags, class_tag, out_tag) }) {
            comptime {
                if (tag_rank != 1 and tag_rank != 2) @compileError("multinomial takes rank-1 [class] or rank-2 [row, class] weights (the torch surface)");
                if (axis(class_tag) != tag_rank - 1) @compileError("multinomial requires the class tag on the last axis");
            }
            const result_tags = replaceTag(tags, class_tag, out_tag);
            if (num_samples == 0) return TensorError.InvalidShape;

            const raw = self.asRawTensor();
            const classes = raw.shape.at(tag_rank - 1);
            const rows = if (tag_rank == 2) raw.shape.at(0) else 1;
            if (!replacement and num_samples > classes) return TensorError.InvalidShape;

            var prepared: ?RawTensor = null;
            defer if (prepared) |*p| p.deinit();
            const weights_flat = if (raw.isContiguous()) try raw.dataConstChecked() else blk: {
                prepared = try ctx.materialize(raw);
                break :blk try prepared.?.dataConstChecked();
            };

            const scratch = try ctx.allocator.alloc(f64, classes);
            defer ctx.allocator.free(scratch);

            const out_shape: [tag_rank]usize = if (tag_rank == 2) .{ rows, num_samples } else .{num_samples};
            var out = try Tensor(.{ .dtype = .i64, .tags = result_tags }).empty(ctx, out_shape);
            errdefer out.deinit();
            const out_data = try out.data();

            for (0..rows) |row| {
                const weights = weights_flat[row * classes ..][0..classes];
                var total: f64 = 0;
                for (weights, 0..) |w, c| {
                    if (!(w >= 0) or !std.math.isFinite(w)) return TensorError.InvalidShape;
                    total += w;
                    scratch[c] = if (replacement) total else w;
                }
                if (!(total > 0) or !std.math.isFinite(total)) return TensorError.InvalidShape;

                for (0..num_samples) |s| {
                    // Counter-based [0, 1) draw for (row, s): the uniformFill mapping.
                    const draw = @as(f64, @floatFromInt(rng.at(seed, row * num_samples + s) >> 11)) * 0x1.0p-53;
                    var pick: usize = undefined;
                    if (replacement) {
                        // First index with cumsum > u (zero-weight classes have
                        // zero-width intervals and are never hit).
                        const u = draw * total;
                        var lo: usize = 0;
                        var hi: usize = classes;
                        while (lo < hi) {
                            const mid = lo + (hi - lo) / 2;
                            if (scratch[mid] > u) hi = mid else lo = mid + 1;
                        }
                        pick = @min(lo, classes - 1);
                        // Rounding can land u on the top boundary; step down to mass.
                        while (pick > 0 and weights[pick] == 0) pick -= 1;
                    } else {
                        if (!(total > 0)) return TensorError.InvalidShape; // fewer nonzero classes than draws
                        const u = draw * total;
                        var acc: f64 = 0;
                        var found: ?usize = null;
                        for (scratch[0..classes], 0..) |w, c| {
                            if (w <= 0) continue;
                            acc += w;
                            if (acc > u) {
                                found = c;
                                break;
                            }
                        }
                        if (found == null) {
                            // Rounding hit the top boundary: last remaining class.
                            var c = classes;
                            while (c > 0) {
                                c -= 1;
                                if (scratch[c] > 0) {
                                    found = c;
                                    break;
                                }
                            }
                        }
                        pick = found orelse return TensorError.InvalidShape;
                        total -= scratch[pick];
                        scratch[pick] = 0;
                    }
                    out_data[row * num_samples + s] = @intCast(pick);
                }
            }
            return out;
        }

        /// Max values over `tag` (the tag is removed like sum/mean; argmax
        /// returns the indices). The gradient flows only to the FIRST
        /// occurrence of the extremum along the axis (strict-comparison
        /// tie-break, like PyTorch's torch.max over a dim).
        pub fn max(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            return extremum(self, ctx, tag, .max);
        }

        /// Min values over `tag`; see `max` for gradient/tie-break semantics.
        pub fn min(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            return extremum(self, ctx, tag, .min);
        }

        fn extremum(self: *const Self, ctx: *ExecContext, comptime tag: Tag, comptime op: enum { max, min }) !Tensor(removeTag(tags, tag)) {
            const result_tags = removeTag(tags, tag);
            const reduce_axis = comptime axis(tag);
            var raw = switch (op) {
                .max => try ctx.maxAxisRank(tag_rank, self.asRawTensor(), reduce_axis),
                .min => try ctx.minAxisRank(tag_rank, self.asRawTensor(), reduce_axis),
            };
            // The first-extremum indices go into the backward node (computed
            // in the forward, not recomputed); the caller only sees values.
            var raw_values: ?RawTensor = raw.values;
            errdefer if (raw_values) |*value| value.deinit();
            defer raw.indices.deinit();
            const out = try finishOp(result_tags, ctx, raw_values.?, self.requiresGrad(), MinMaxBackward(tags, reduce_axis), .{ ctx.allocator, self.grad_state, &self.value, &raw.indices });
            raw_values = null;
            return out;
        }
    };
}
