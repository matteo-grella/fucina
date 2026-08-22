//! f32 tensor methods: constructors and fills (variable/constant/from*, zeros..eye). A mixin over the ag
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

        /// Consumes `value` on success; on error, ownership stays with the caller.
        pub fn variable(ctx: *ExecContext, value: RawTensor) !Self {
            var v = value;
            try validateTensorRank(tags, &v);

            const state = try GradState.leaf(ctx.allocator);
            errdefer state.deinit();

            return .{ .value = v, .grad_state = state };
        }

        pub fn variableFromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self {
            var value = try ctx.fromSliceRank(tensor_rank, raw_shape, values);
            errdefer value.deinit();
            return Self.variable(ctx, value);
        }

        /// Consumes `value` on success; on error, ownership stays with the caller.
        pub fn constant(ctx: *ExecContext, value: RawTensor) !Self {
            _ = ctx;
            var v = value;
            try validateTensorRank(tags, &v);
            return .{ .value = v };
        }

        pub fn fromTensor(ctx: *ExecContext, value: RawTensor) !Self {
            return try Self.constant(ctx, value);
        }

        pub fn fromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self {
            var value = try ctx.fromSliceRank(tensor_rank, raw_shape, values);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Wrap caller-owned mutable storage as a no-grad constant tensor.
        /// The returned tensor borrows `values`; callers must keep that slice
        /// alive and unmoved until the tensor is deinitialized.
        pub fn fromBorrowedSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []f32) !Self {
            var value = try ctx.fromBorrowedSliceRank(tensor_rank, raw_shape, values);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Zero-copy wrap caller-owned READ-ONLY storage (e.g. mmap'd const GGUF
        /// weights) as a no-grad constant tensor, so callers no longer scatter
        /// `@constCast` to turn const file data into a tensor view. The tensor
        /// BORROWS `values`: the slice must outlive the tensor, stay unmoved, and
        /// MUST NOT be mutated through `.data()`. The single internal `@constCast`
        /// is sound only under that read-only contract — use `fromSlice` (which
        /// copies into owned storage) if you need a writable buffer.
        pub fn fromBorrowedConstSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self {
            var value = try ctx.fromBorrowedSliceRank(tensor_rank, raw_shape, @constCast(values));
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Allocate an uninitialized no-grad tensor of the tag-implied rank.
        pub fn empty(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self {
            var value = try ctx.empty(&raw_shape);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Allocate a zero-filled no-grad tensor.
        pub fn zeros(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self {
            var value = try ctx.zeros(&raw_shape);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Allocate a one-filled no-grad tensor.
        pub fn ones(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self {
            var value = try ctx.ones(&raw_shape);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Allocate a no-grad tensor filled with `fill_value`.
        pub fn full(ctx: *ExecContext, raw_shape: [tensor_rank]usize, fill_value: f32) !Self {
            var value = try ctx.full(&raw_shape, fill_value);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Build a single-element no-grad tensor holding `scalar_value`.
        pub fn scalar(ctx: *ExecContext, scalar_value: f32) !Self {
            var value = try ctx.scalar(scalar_value);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        /// Rank-1 no-grad tensor holding `start, start+step, …` up to but
        /// excluding `end` (torch.arange with float semantics): element i
        /// is `start + i·step` (not accumulated). `step` must move from
        /// `start` toward `end` — an empty range is `InvalidShape`
        /// (zero-size tensors are not representable), as is `step == 0`.
        pub fn arange(ctx: *ExecContext, start: f32, end: f32, step: f32) !Self {
            comptime if (tag_count != 1) @compileError("arange builds a rank-1 tensor; use a single-tag Tensor type");
            if (step == 0) return TensorError.InvalidShape;
            const span = (end - start) / step;
            if (!(span > 0)) return TensorError.InvalidShape;
            const count: usize = @intFromFloat(@ceil(span));
            var value = try ctx.empty(&.{count});
            errdefer value.deinit();
            for (value.data(), 0..) |*out, i| out.* = start + @as(f32, @floatFromInt(i)) * step;
            return try Self.constant(ctx, value);
        }

        /// Rank-1 no-grad tensor of `steps` values spaced evenly from
        /// `start` to `end` INCLUSIVE (torch.linspace): element i is
        /// `start + i·(end-start)/(steps-1)` with the final element pinned
        /// to exactly `end`; `steps == 1` yields `{start}`. `steps == 0`
        /// is `InvalidShape` (zero-size tensors are not representable).
        pub fn linspace(ctx: *ExecContext, start: f32, end: f32, steps: usize) !Self {
            comptime if (tag_count != 1) @compileError("linspace builds a rank-1 tensor; use a single-tag Tensor type");
            if (steps == 0) return TensorError.InvalidShape;
            var value = try ctx.empty(&.{steps});
            errdefer value.deinit();
            const out = value.data();
            if (steps == 1) {
                out[0] = start;
            } else {
                const stride = (end - start) / @as(f32, @floatFromInt(steps - 1));
                for (out, 0..) |*o, i| o.* = start + @as(f32, @floatFromInt(i)) * stride;
                out[steps - 1] = end;
            }
            return try Self.constant(ctx, value);
        }

        /// Rank-2 no-grad one-hot matrix `[indices.len, depth]` (torch
        /// F.one_hot with an explicit class count, as f32): row i holds 1.0
        /// at column `indices[i]`, 0.0 elsewhere. Indices are host-side
        /// like `gather`'s; `indices[i] >= depth` is `IndexOutOfBounds`,
        /// an empty `indices` is `InvalidShape` (zero-size tensors are not
        /// representable). The first tag is the row axis, the second the
        /// class axis.
        pub fn oneHot(ctx: *ExecContext, indices: []const usize, depth: usize) !Self {
            comptime if (tag_count != 2) @compileError("oneHot builds a rank-2 [rows, classes] tensor; use a two-tag Tensor type");
            if (indices.len == 0 or depth == 0) return TensorError.InvalidShape;
            var value = try ctx.zeros(&.{ indices.len, depth });
            errdefer value.deinit();
            const out = value.data();
            for (indices, 0..) |class_index, row| {
                if (class_index >= depth) return TensorError.IndexOutOfBounds;
                out[row * depth + class_index] = 1;
            }
            return try Self.constant(ctx, value);
        }

        /// No-grad tensor of uniform draws in `[0, 1)` (torch.rand) from
        /// the deterministic counter-based stream at `seed` (§6.8,
        /// `fucina.rng`): element i is a pure function of `(seed, i)`, so
        /// the same seed always reproduces the same tensor — the stream IS
        /// the generator abstraction (store the seed, regenerate the
        /// values). Pass a fresh seed per draw (reusing one reuses the
        /// values).
        pub fn rand(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64) !Self {
            return uniform(ctx, raw_shape, seed, 0, 1);
        }

        /// `rand` over `[lo, hi)` (the `fucina.rng.uniformFill` mapping).
        pub fn uniform(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, lo: f32, hi: f32) !Self {
            var value = try ctx.empty(&raw_shape);
            errdefer value.deinit();
            rng.uniformFill(seed, value.data(), lo, hi);
            return try Self.constant(ctx, value);
        }

        /// No-grad tensor of standard-normal draws (torch.randn) from the
        /// deterministic stream at `seed` (see `rand`); Box-Muller over
        /// the splitmix64 stream (`fucina.rng.gaussianFill`).
        pub fn randn(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64) !Self {
            return normal(ctx, raw_shape, seed, 0, 1);
        }

        /// `randn` with explicit moments (`fucina.rng.normalFill`).
        pub fn normal(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, mean_value: f32, std_dev: f32) !Self {
            var value = try ctx.empty(&raw_shape);
            errdefer value.deinit();
            rng.normalFill(seed, value.data(), mean_value, std_dev);
            return try Self.constant(ctx, value);
        }

        /// No-grad 0/1 tensor of Bernoulli draws (torch.bernoulli with a
        /// scalar probability): element i is 1.0 iff the `[0, 1)` uniform
        /// stream at `(seed, i)` (see `rand`) draws below `p`. `p` outside
        /// `[0, 1]` is `InvalidShape`.
        pub fn bernoulli(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, p: f32) !Self {
            if (!(p >= 0 and p <= 1)) return TensorError.InvalidShape;
            var value = try ctx.empty(&raw_shape);
            errdefer value.deinit();
            const out = value.data();
            rng.uniformFill(seed, out, 0, 1);
            for (out) |*v| v.* = if (v.* < p) 1 else 0;
            return try Self.constant(ctx, value);
        }

        /// No-grad tensor of standard Gumbel(0, 1) draws — `-ln(-ln(u))`,
        /// the gumbel-max / gumbel-softmax noise — from the deterministic
        /// stream at `seed` (see `rand`; `fucina.rng.gumbelFill` documents
        /// the exact open-interval mapping). Add to logits and take
        /// argmax for a categorical sample, or softmax at a temperature
        /// for its differentiable relaxation.
        pub fn gumbel(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64) !Self {
            var value = try ctx.empty(&raw_shape);
            errdefer value.deinit();
            rng.gumbelFill(seed, value.data());
            return try Self.constant(ctx, value);
        }

        /// Identity matrix `[n, n]` (torch.eye) as a no-grad constant: 1.0
        /// on the main diagonal, 0.0 elsewhere. The first tag is the row
        /// axis, the second the column axis. `n == 0` is `InvalidShape`
        /// (zero-size tensors are not representable).
        pub fn eye(ctx: *ExecContext, n: usize) !Self {
            comptime if (tag_count != 2) @compileError("eye builds a rank-2 [n, n] tensor; use a two-tag Tensor type");
            if (n == 0) return TensorError.InvalidShape;
            var value = try ctx.zeros(&.{ n, n });
            errdefer value.deinit();
            const out = value.data();
            for (0..n) |i| out[i * (n + 1)] = 1;
            return try Self.constant(ctx, value);
        }

        // --- *Like constructors ---------------------------------------------
        // Instance sugar over the static constructors above: same tags and
        // dtype (both are part of `Self`), shape taken from `self`'s logical
        // shape (strided views included). Like every constructor the result
        // is a fresh owned NO-GRAD constant — `self`'s grad state does not
        // carry over — and is never scope-owned.

        /// `empty` with `self`'s shape: uninitialized storage.
        pub fn emptyLike(self: *const Self, ctx: *ExecContext) !Self {
            return Self.empty(ctx, self.shape());
        }

        /// `zeros` with `self`'s shape.
        pub fn zerosLike(self: *const Self, ctx: *ExecContext) !Self {
            return Self.zeros(ctx, self.shape());
        }

        /// `ones` with `self`'s shape.
        pub fn onesLike(self: *const Self, ctx: *ExecContext) !Self {
            return Self.ones(ctx, self.shape());
        }

        /// `full` with `self`'s shape, filled with `fill_value`.
        pub fn fullLike(self: *const Self, ctx: *ExecContext, fill_value: f32) !Self {
            return Self.full(ctx, self.shape(), fill_value);
        }

        // --- Rank-generic matmul --------------------------------------------
        // `out_tags` names the result axes (rank-generic — no fragile
        // tag-composition rule). For tag-semantics contractions, prefer `dot`.
    };
}
