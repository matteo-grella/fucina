const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");
const exec_mod = @import("../exec.zig");
const backend_mod = @import("../backend.zig");
const parallel = @import("../parallel.zig");
const tag_ops = @import("../tagged.zig");
const control = @import("control.zig");
const core = @import("core.zig");
const tags_mod = @import("../tags.zig");
const backward = @import("backward.zig");
const elemental = @import("elemental.zig");
const rng = @import("../rng.zig");

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
const BackwardFunction = core.BackwardFunction;
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
const dotBatchTags = tags_mod.dotBatchTags;
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
const VarBackward = backward.VarBackward;
const StandardizeBackward = backward.StandardizeBackward;
const BroadcastBackward = backward.BroadcastBackward;
const GatherBackward = backward.GatherBackward;
const TopKBackward = backward.TopKBackward;
const MinMaxBackward = backward.MinMaxBackward;
const NarrowBackward = backward.NarrowBackward;
const ConcatBackward = backward.ConcatBackward;
const CumsumBackward = backward.CumsumBackward;
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
const EinsumBackward = backward.EinsumBackward;
const ConstRhsDotBackward = backward.ConstRhsDotBackward;
const ConstRhsEinsumBackward = backward.ConstRhsEinsumBackward;
const TernarySteDotBackward = backward.TernarySteDotBackward;

/// Input counts covered by the stack fast path for `concat`/`stack` metadata
/// temporaries (input pointers, backward parents/sizes); larger input counts
/// fall back to a heap allocation.
const concat_inline_inputs = 16;

pub fn TopKResult(comptime tags_spec: anytype) type {
    const result_tags = normalizeTags(tags_spec);
    return struct {
        values: Tensor(result_tags),
        /// Source positions along the reduced/sorted axis: a constant i64
        /// tensor (exact for any axis length, torch's index dtype). Like
        /// every typed-constant result it is caller-owned even under an
        /// exec scope.
        indices: Tensor(.{ .dtype = .i64, .tags = result_tags }),

        pub fn deinit(self: *@This()) void {
            self.values.deinit();
            self.indices.deinit();
            self.* = undefined;
        }
    };
}

pub fn Tensor(comptime tags_spec: anytype) type {
    const tensor_dtype = dtypeFromSpec(tags_spec);
    if (comptime tensor_dtype == .f32) return FloatTensor(tags_spec);
    if (comptime dtype_mod.isBlockQuantized(tensor_dtype)) return QuantizedConstantTensor(tags_spec, tensor_dtype);
    return TypedConstantTensor(tags_spec, tensor_dtype);
}

fn FloatTensor(comptime tags_spec: anytype) type {
    const tags = normalizeTags(tags_spec);
    comptime validateUniqueTags(tags);
    const tag_rank = tags.len;
    if (tag_rank > tensor_mod.max_rank) @compileError("too many tensor tags");

    return struct {
        pub const axis_tags = tags;
        pub const tag_count = tag_rank;
        pub const tensor_rank = rawRank(tag_rank);
        pub const dtype = DType.f32;

        value: RawTensor,
        grad_state: ?*GradState = null,
        /// True when an exec scope owns this tensor: the struct is a borrow
        /// and `deinit` is a safe no-op (arena-allocator semantics — the
        /// scope releases value and node at closeExecScope). Lets the same
        /// defer-deinit forward code run scoped (training) and unscoped
        /// (inference).
        scope_owned: bool = false,

        const Self = @This();

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

        /// Explicit matmul over caller-named result axes, comptime-routed on
        /// operand rank: BOTH operands rank-2 → the 2-D GEMM entries
        /// (`.plain`: `[m,k]·[k,n] -> [m,n]`; `.trans_b`: `[m,k]·[n,k]ᵀ ->
        /// [m,n]`); anything else → the batched bmm entries with stride-0
        /// BROADCAST leading batch axes (`[...,m,k]·[...,k,n] -> [...,m,n]`
        /// etc.) — mixed-rank operands broadcast rather than error.
        /// `.trans_a` (`[...,k,m]ᵀ·[...,k,n]`) exists only on the batched
        /// path: rank-2 Aᵀ·B has no backward record — use `dot`, whose tag
        /// algebra reaches the 2-D trans-A kernel. f32 only, full two-operand
        /// grads; unlike `dot` there is no materialize fallback — the
        /// operands' storage order IS the kernel layout. (Distinct from the
        /// strictly-2-D-plain `ExecContext.matmul` at the exec layer.)
        pub fn matmul(self: *const Self, ctx: *ExecContext, other: anytype, comptime kind: exec_mod.MatmulKind, comptime out_tags: anytype) !Tensor(out_tags) {
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            const other_rank = comptime TensorObject(@TypeOf(other)).axis_tags.len;
            if (comptime (tag_rank == 2 and other_rank == 2)) {
                comptime if (kind == .trans_a) {
                    @compileError("matmul: rank-2 .trans_a has no backward record — use `dot` (its tag algebra reaches the 2-D trans-A kernel)");
                };
                var value = try switch (comptime kind) {
                    .plain => ctx.matmul2D(self.asRawTensor(), other_ptr.asRawTensor()),
                    .trans_b => ctx.matmulTransB(self.asRawTensor(), other_ptr.asRawTensor()),
                    .trans_a => unreachable, // rejected above
                };
                errdefer value.deinit();
                return finishOp(out_tags, ctx, value, self.requiresGrad() or other_ptr.requiresGrad(), Matmul2DBackward(kind == .trans_b), .{ ctx.allocator, self.grad_state, other_ptr.grad_state, self.asRawTensor(), other_ptr.asRawTensor() });
            }
            var value = try switch (comptime kind) {
                .plain => ctx.bmm(self.asRawTensor(), other_ptr.asRawTensor()),
                .trans_a => ctx.bmmTransA(self.asRawTensor(), other_ptr.asRawTensor()),
                .trans_b => ctx.bmmTransB(self.asRawTensor(), other_ptr.asRawTensor()),
            };
            errdefer value.deinit();
            return finishOp(out_tags, ctx, value, self.requiresGrad() or other_ptr.requiresGrad(), BmmBackward(kind), .{ ctx.allocator, self.grad_state, other_ptr.grad_state, self.asRawTensor(), other_ptr.asRawTensor() });
        }

        // --- Axis bias-add + scaled residual-add (no-grad) -------------------
        // `axis_tag` must name the LAST axis (the per-feature axis); `bias` is a
        // `[axis_dim]` row vector. These are inference helpers and reject
        // trainable inputs; for a differentiable bias use `add` with a broadcast
        // operand.

        /// In-place: add the `[axis_dim]` `bias` to every row of `self` along the
        /// last axis `axis_tag`, mutating `self`.
        pub fn addAxisVectorInPlace(self: *Self, ctx: *ExecContext, bias: []const f32, comptime axis_tag: Tag) !void {
            if (self.requiresGrad()) return error.UnsupportedGradient;
            try ctx.addAxisVectorInPlaceRank(tensor_rank, &self.value, bias, comptime Self.axis(axis_tag));
        }

        /// In-place fused bias-add + unary activation `op`, mutating `self`.
        pub fn addAxisVectorUnaryInPlace(self: *Self, ctx: *ExecContext, comptime op: UnaryOp, bias: []const f32, comptime axis_tag: Tag) !void {
            if (self.requiresGrad()) return error.UnsupportedGradient;
            try ctx.addAxisVectorUnaryInPlaceRank(tensor_rank, op, &self.value, bias, comptime Self.axis(axis_tag));
        }

        /// In-place scaled residual `self += alpha · other` (same shape).
        pub fn addScaledInPlace(self: *Self, ctx: *ExecContext, other: anytype, alpha: f32) !void {
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            if (self.requiresGrad() or other_ptr.requiresGrad()) return error.UnsupportedGradient;
            try ctx.addScaledInPlace(&self.value, other_ptr.asRawTensor(), alpha);
        }

        /// Out-of-place `self + bias` (broadcast along the last axis `axis_tag`) as
        /// a NEW no-grad tensor; `self` is unchanged.
        pub fn biasAdd(self: *const Self, ctx: *ExecContext, bias: []const f32, comptime axis_tag: Tag) !Self {
            var value = try self.value.clone(ctx.allocator);
            errdefer value.deinit();
            try ctx.addAxisVectorInPlaceRank(tensor_rank, &value, bias, comptime Self.axis(axis_tag));
            return finishOp(tags, ctx, value, self.requiresGrad(), IdentityBackward(tags), .{ ctx.allocator, self.grad_state });
        }

        /// `cond ? self : other` elementwise (`cond[i] != 0` selects `self`).
        /// Differentiable in `self` and `other`; `cond` is a non-grad mask.
        /// `cond ? self : other` elementwise. `cond` is a same-tagged
        /// `.bool` mask (the `compare` output) or a float tensor read by
        /// truthiness (`!= 0`; NaN truthy); it receives no gradient.
        /// Differentiable in `self` and `other`.
        pub fn where(self: *const Self, ctx: *ExecContext, cond: anytype, other: anytype) !Self {
            const Cond = TensorObject(@TypeOf(cond));
            comptime {
                if (Cond.dtype != .bool and !dtype_mod.supportsForwardFloatMath(Cond.dtype))
                    @compileError("where takes a .bool or float condition; cast integer masks explicitly");
            }
            var value = try ctx.whereTyped(Cond.dtype, self.asRawTensor(), cond.asRawTensor(), other.asRawTensor());
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or other.requiresGrad(), WhereBackward(tags, Cond.dtype), .{ ctx.allocator, self.grad_state, other.grad_state, cond.asRawTensor() });
        }

        /// `mask ? value : self` elementwise. `mask` is a same-tagged
        /// `.bool` mask (the `compare` output) or a float tensor read by
        /// truthiness. Differentiable in `self` (grad zeroed where filled);
        /// `value` is constant.
        pub fn maskedFill(self: *const Self, ctx: *ExecContext, mask: anytype, value: f32) !Self {
            const Mask = TensorObject(@TypeOf(mask));
            comptime {
                if (Mask.dtype != .bool and !dtype_mod.supportsForwardFloatMath(Mask.dtype))
                    @compileError("maskedFill takes a .bool or float mask; cast integer masks explicitly");
            }
            var v = try ctx.maskedFillTyped(Mask.dtype, self.asRawTensor(), mask.asRawTensor(), value);
            errdefer v.deinit();
            return finishOp(tags, ctx, v, self.requiresGrad(), MaskedFillBackward(tags, Mask.dtype), .{ ctx.allocator, self.grad_state, mask.asRawTensor() });
        }

        /// Elementwise comparison: a same-tagged `.bool` mask (torch's
        /// comparison dtype), true where `self <op> other` holds. `other`
        /// is comptime-dispatched from its type: a same-tagged tensor (same
        /// shape only, like `where`) or a numeric scalar (see
        /// `exec.CompareOp`). Non-differentiable, and — like every typed
        /// constant — CALLER-owned even under an exec scope. NaN semantics
        /// are IEEE: any comparison involving NaN is false, except `.ne`,
        /// which is true. Feed the result to `where`/`maskedFill`/the
        /// logical ops, count with `sum`, or cast with `to(.f32)` for the
        /// mask-multiply idiom.
        pub fn compare(self: *const Self, ctx: *ExecContext, comptime op: exec_mod.CompareOp, other: anytype) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            const BoolT = Tensor(.{ .dtype = .bool, .tags = tags });
            const OtherT = @TypeOf(other);
            if (comptime (OtherT == comptime_float or OtherT == comptime_int or @typeInfo(OtherT) == .float or @typeInfo(OtherT) == .int)) {
                var value = try ctx.compareScalar(op, self.asRawTensor(), other);
                errdefer value.deinit();
                return BoolT.fromTensor(ctx, value);
            }
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            var value = try ctx.compare(op, self.asRawTensor(), other_ptr.asRawTensor());
            errdefer value.deinit();
            return BoolT.fromTensor(ctx, value);
        }

        /// Elementwise logical AND over truthiness (the mask convention
        /// shared with `where`/`maskedFill`; NaN is truthy): a same-tagged
        /// `.bool` tensor (torch's logical-op dtype). `other` may be a
        /// float or `.bool` tensor. Same shape only; non-differentiable
        /// and caller-owned like `compare`.
        pub fn logicalAnd(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            return self.logicalBinary(ctx, .l_and, other);
        }

        /// Elementwise logical OR over truthiness (see `logicalAnd`).
        pub fn logicalOr(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            return self.logicalBinary(ctx, .l_or, other);
        }

        /// Elementwise logical XOR over truthiness (see `logicalAnd`).
        pub fn logicalXor(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            return self.logicalBinary(ctx, .l_xor, other);
        }

        fn logicalBinary(self: *const Self, ctx: *ExecContext, comptime op: exec_mod.LogicalOp, other: anytype) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (Other.dtype != .bool and !dtype_mod.supportsForwardFloatMath(Other.dtype))
                    @compileError("logical ops take .bool or float operands; cast integer masks explicitly");
            }
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            var value = try ctx.logicalTyped(op, .f32, Other.dtype, self.asRawTensor(), other_ptr.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = .bool, .tags = tags }).fromTensor(ctx, value);
        }

        /// Elementwise logical NOT over truthiness (see `logicalAnd`):
        /// a `.bool` tensor, true where `self` is zero.
        pub fn logicalNot(self: *const Self, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            var value = try ctx.logicalNotTyped(.f32, self.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = .bool, .tags = tags }).fromTensor(ctx, value);
        }

        /// `.bool`, true where `self` is NaN (torch.isnan): the IEEE
        /// self-inequality test through `compare` — non-differentiable
        /// constant mask like all mask producers, unscoped-safe.
        pub fn isnan(self: *const Self, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            return self.compare(ctx, .ne, self.*);
        }

        /// `.bool`, true where `self` is +inf or -inf (torch.isinf); NaN is
        /// false. Non-differentiable constant mask, unscoped-safe (composed
        /// from no-grad compares only).
        pub fn isinf(self: *const Self, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            var pos = try self.compare(ctx, .eq, std.math.inf(f32));
            defer pos.deinit();
            var negative_inf = try self.compare(ctx, .eq, -std.math.inf(f32));
            defer negative_inf.deinit();
            return pos.logicalOr(ctx, &negative_inf);
        }

        /// `.bool`, true where `self` is finite (not NaN, not ±inf)
        /// (torch.isfinite): `-inf < x < inf`, which IEEE comparison makes
        /// false for NaN and both infinities. Non-differentiable constant
        /// mask, unscoped-safe.
        pub fn isfinite(self: *const Self, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            var above = try self.compare(ctx, .gt, -std.math.inf(f32));
            defer above.deinit();
            var below = try self.compare(ctx, .lt, std.math.inf(f32));
            defer below.deinit();
            return above.logicalAnd(ctx, &below);
        }

        /// `.bool`, true where ANY element along `tag` is truthy (`!= 0`;
        /// NaN is truthy, the torch.any convention), with `tag` removed.
        /// Non-differentiable constant mask (compare → i64 count →
        /// compare), unscoped-safe.
        pub fn any(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .bool, .tags = removeTag(tags, tag) }) {
            var truthy = try self.compare(ctx, .ne, 0);
            defer truthy.deinit();
            var count = try truthy.sum(ctx, tag);
            defer count.deinit();
            return count.compare(ctx, .ge, 1);
        }

        /// `.bool`, true where EVERY element along `tag` is truthy (see
        /// `any`), with `tag` removed. Counts the zero entries and tests
        /// the count against 1. Non-differentiable constant mask,
        /// unscoped-safe.
        pub fn all(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .bool, .tags = removeTag(tags, tag) }) {
            var zero = try self.compare(ctx, .eq, 0);
            defer zero.deinit();
            var count = try zero.sum(ctx, tag);
            defer count.deinit();
            return count.compare(ctx, .lt, 1);
        }

        /// Scalar `any` over every element (torch.any with no dim); `.bool`.
        pub fn anyAll(self: *const Self, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = .{} }) {
            var truthy = try self.compare(ctx, .ne, 0);
            defer truthy.deinit();
            var count = try truthy.sumAll(ctx);
            defer count.deinit();
            return count.compare(ctx, .ge, 1);
        }

        /// Scalar `all` over every element (torch.all with no dim); `.bool`.
        pub fn allAll(self: *const Self, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = .{} }) {
            var zero = try self.compare(ctx, .eq, 0);
            defer zero.deinit();
            var count = try zero.sumAll(ctx);
            defer count.deinit();
            return count.compare(ctx, .lt, 1);
        }

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

        /// PReLU with a learnable per-channel slope (`alpha` rank-1 `[C]`, the
        /// channel axis innermost): `y = x > 0 ? x : α[c]·x`, one fused pass;
        /// differentiable in both `x` and `α`.
        pub fn prelu(self: *const Self, ctx: *ExecContext, alpha: anytype) !Self {
            const alpha_ptr = tensorObjectPtrFrom(@TypeOf(alpha), &alpha);
            const any_grad = self.requiresGrad() or alpha_ptr.requiresGrad();
            var value = try ctx.preluChannels(self.asRawTensor(), alpha_ptr.asRawTensor());
            errdefer value.deinit();
            return finishOp(tags, ctx, value, any_grad, PreluChannelsBackward, .{ ctx.allocator, self.grad_state, alpha_ptr.grad_state, self.asRawTensor(), alpha_ptr.asRawTensor() });
        }

        /// Per-channel affine `y = x·scale[c] + shift[c]` (rank-1 `[C]` params,
        /// channel axis innermost) — the frozen-stats inference BatchNorm as
        /// one fused pass; differentiable in `x`, `scale`, and `shift`.
        pub fn channelAffine(self: *const Self, ctx: *ExecContext, scale_t: anytype, shift_t: anytype) !Self {
            const scale_ptr = tensorObjectPtrFrom(@TypeOf(scale_t), &scale_t);
            const shift_ptr = tensorObjectPtrFrom(@TypeOf(shift_t), &shift_t);
            const any_grad = self.requiresGrad() or scale_ptr.requiresGrad() or shift_ptr.requiresGrad();
            var value = try ctx.channelAffine(self.asRawTensor(), scale_ptr.asRawTensor(), shift_ptr.asRawTensor());
            errdefer value.deinit();
            return finishOp(tags, ctx, value, any_grad, ChannelAffineBackward, .{ ctx.allocator, self.grad_state, scale_ptr.grad_state, shift_ptr.grad_state, self.asRawTensor(), scale_ptr.asRawTensor() });
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

        pub fn deinit(self: *Self) void {
            if (self.scope_owned) return; // borrow: the exec scope owns value + node
            self.value.deinit();
            if (self.grad_state) |state| state.deinit();
            self.* = undefined;
        }

        pub fn asRawTensor(self: *const Self) *const RawTensor {
            return &self.value;
        }

        pub fn item(self: *const Self) !f32 {
            if (!self.value.isScalar()) return TensorError.InvalidShape;
            return (try self.value.dataConstChecked())[0];
        }

        pub fn data(self: *Self) ![]f32 {
            if (self.requiresGrad()) return error.MutableDataRequiresNoGrad;
            return self.value.dataChecked();
        }

        pub fn dataConst(self: *const Self) ![]const f32 {
            return self.value.dataConstChecked();
        }

        pub fn detach(self: *const Self, ctx: *ExecContext) !Self {
            var value = try self.value.cloneView();
            errdefer value.deinit();
            return finishNoGrad(tags, ctx, value);
        }

        pub fn copyTo(self: *const Self, dst: []f32) !void {
            return self.value.copyTo(dst);
        }

        pub fn requiresGrad(self: *const Self) bool {
            return self.grad_state != null;
        }

        /// Drop the accumulated gradient (no-op for constants). Training loops
        /// call this between steps so gradients don't accumulate across them.
        pub fn zeroGrad(self: *const Self) void {
            if (self.grad_state) |state| state.zeroGrad();
        }

        pub fn backward(self: *const Self, ctx: *ExecContext) !void {
            const state = self.grad_state orelse return error.NoGradientGraph;
            return core.backwardGradOne(ctx, state, &self.value);
        }

        /// As `backward`, but with an explicit output gradient instead of
        /// the implicit scalar 1: the way to run backward from a non-scalar
        /// output (scalar outputs may take one too). `grad_output` is
        /// same-tagged and must match `self`'s shape
        /// (`error.ShapeMismatch`); it is read as a value — its own gradient
        /// state, if any, is ignored — and replaces any gradient already
        /// accumulated on `self`.
        pub fn backwardWithGrad(self: *const Self, ctx: *ExecContext, grad_output: *const Self) !void {
            const state = self.grad_state orelse return error.NoGradientGraph;
            // Checked here too so the error exit leaves `self`'s accumulated
            // gradient untouched (the engine re-checks after setGrad).
            if (state.backward_done) return core.AgError.BackwardAlreadyRun;
            if (!std.mem.eql(usize, self.value.shape.slice(), grad_output.value.shape.slice())) {
                return TensorError.ShapeMismatch;
            }
            state.setGrad(try grad_output.value.cloneView());
            return core.backwardGradOne(ctx, state, &self.value);
        }

        pub fn grad(self: *const Self, ctx: *ExecContext) !?Self {
            const state = self.grad_state orelse return null;
            var value = (try state.gradClone(ctx.allocator)) orelse return null;
            errdefer value.deinit();
            const out = try Self.constant(ctx, value);
            return out;
        }

        pub fn gradView(self: *const Self, ctx: *ExecContext) !?Self {
            const state = self.grad_state orelse return null;
            var value = (try state.gradView()) orelse return null;
            errdefer value.deinit();
            const out = try Self.constant(ctx, value);
            return out;
        }

        pub fn axis(comptime tag: Tag) usize {
            return tagIndexOrCompileError(tags, tag);
        }

        pub fn hasTag(comptime tag: Tag) bool {
            return comptime tagIndex(tags, tag) != null;
        }

        pub fn dim(self: *const Self, comptime tag: Tag) usize {
            return self.asRawTensor().shape.at(axis(tag));
        }

        pub fn shape(self: *const Self) [tensor_rank]usize {
            var out: [tensor_rank]usize = undefined;
            inline for (0..tensor_rank) |i| {
                out[i] = self.asRawTensor().shape.at(i);
            }
            return out;
        }

        pub fn to(self: *const Self, ctx: *ExecContext, comptime target_dtype: DType) !Tensor(.{ .dtype = target_dtype, .tags = tags }) {
            if (comptime (target_dtype != .f32 and target_dtype != .f16 and target_dtype != .bf16)) {
                if (self.requiresGrad()) return error.GradientCastUnsupported;
            }
            var value = try ctx.castTyped(.f32, target_dtype, self.asRawTensor());
            errdefer value.deinit();
            if (comptime target_dtype == .f32) {
                return finishOp(tags, ctx, value, self.requiresGrad(), CastBackward(tags), .{ ctx.allocator, self.grad_state });
            }
            if (comptime (target_dtype == .f16 or target_dtype == .bf16)) {
                // Differentiable narrow (the mixed-precision seam): the
                // backward is the identity in f32 gradient space — the
                // upstream f32 gradient passes through unrounded.
                return typedFinishOp(target_dtype, tags, ctx, value, self.requiresGrad(), CastBackward(tags), .{ ctx.allocator, self.grad_state });
            }
            return Tensor(.{ .dtype = target_dtype, .tags = tags }).fromTensor(ctx, value);
        }

        pub fn materialize(self: *const Self, ctx: *ExecContext) !Self {
            var value = try ctx.materialize(self.asRawTensor());
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), StridedViewBackward(tags, tags), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        /// Borrow-if-contiguous materialize: an already-contiguous tensor
        /// returns a zero-copy retained view of the same storage (linked to
        /// the graph through an identity backward — `GradState` is
        /// single-owner, so the handle carries its own state rather than
        /// aliasing `self`'s); a strided view returns `materialize(ctx)`.
        /// Either way the result is owned by the caller (always `deinit` it;
        /// refcounted storage keeps the view case safe past the source's
        /// deinit), contiguous, and safe for `data`/`dataConst` access
        /// (`data` still rejects grad-carrying tensors — the autograd
        /// mutation invariant). The torch.contiguous aliasing caveat
        /// carries over: the already-contiguous case ALIASES `self`'s bytes
        /// (in-place mutation of either is visible through both) while the
        /// strided case is an independent snapshot — treat the result as
        /// read-only where the distinction matters, or use `materialize`
        /// when a guaranteed copy is wanted.
        pub fn contiguous(self: *const Self, ctx: *ExecContext) !Self {
            if (!self.asRawTensor().isContiguous()) return self.materialize(ctx);
            var value = try self.value.cloneView();
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), IdentityBackward(tags), .{ ctx.allocator, self.grad_state });
        }

        pub fn withTags(self: *const Self, ctx: *ExecContext, comptime new_tags_spec: anytype) !Tensor(normalizeTags(new_tags_spec)) {
            const new_tags = normalizeTags(new_tags_spec);
            comptime {
                validateUniqueTags(new_tags);
                if (new_tags.len != tag_rank) @compileError("withTags requires the same rank");
            }
            return self.axisView(ctx, identityAxes(tag_rank), new_tags);
        }

        /// No-copy view with an explicit raw shape/stride layout and a new public
        /// tag set. Use this for audited structural views that cannot be expressed
        /// as pure tag permutations, insertions, or squeezes.
        pub fn viewWithStrides(
            self: *const Self,
            ctx: *ExecContext,
            comptime new_tags_spec: anytype,
            raw_shape: [rawRank(normalizeTags(new_tags_spec).len)]usize,
            raw_strides: [rawRank(normalizeTags(new_tags_spec).len)]usize,
        ) !Tensor(normalizeTags(new_tags_spec)) {
            const new_tags = normalizeTags(new_tags_spec);
            comptime validateUniqueTags(new_tags);
            var value = try self.value.viewWithStrides(raw_shape[0..], raw_strides[0..]);
            errdefer value.deinit();
            return finishOp(new_tags, ctx, value, self.requiresGrad(), StridedViewBackward(tags, new_tags), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        pub fn alignTo(self: *const Self, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(normalizeTags(target_tags_spec)) {
            const target_tags = normalizeTags(target_tags_spec);
            return self.axisView(ctx, alignAxes(tags, target_tags), target_tags);
        }

        pub fn permuteTo(self: *const Self, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(normalizeTags(target_tags_spec)) {
            const target_tags = normalizeTags(target_tags_spec);
            comptime validateSameTagSet(tags, target_tags);
            return self.axisView(ctx, alignAxes(tags, target_tags), target_tags);
        }

        pub fn transpose(self: *const Self, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(normalizeTags(target_tags_spec)) {
            return self.permuteTo(ctx, target_tags_spec);
        }

        pub fn insertAxis(self: *const Self, ctx: *ExecContext, comptime tag: Tag, comptime axis_index: usize) !Tensor(insertTagAt(tags, tag, axis_index)) {
            const result_tags = insertTagAt(tags, tag, axis_index);
            return self.axisView(ctx, insertAxes(tag_rank, axis_index), result_tags);
        }

        pub fn squeeze(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            const result_tags = removeTag(tags, tag);
            const axis_index = comptime tagIndexOrCompileError(tags, tag);
            if (self.asRawTensor().shape.at(axis_index) != 1) return TensorError.InvalidShape;
            return self.axisView(ctx, squeezeAxes(tag_rank, axis_index), result_tags);
        }

        pub fn split(
            self: *const Self,
            ctx: *ExecContext,
            comptime tag: Tag,
            comptime split_tags_spec: anytype,
            split_shape: [normalizeTags(split_tags_spec).len]usize,
        ) !Tensor(splitTags(tags, tag, normalizeTags(split_tags_spec))) {
            const split_tags = normalizeTags(split_tags_spec);
            const result_tags = splitTags(tags, tag, split_tags);
            var value = try tag_ops.splitAxisView(tags, self.asRawTensor(), tag, split_tags, split_shape);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), StridedViewBackward(tags, result_tags), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        pub fn merge(self: *const Self, ctx: *ExecContext, comptime out_tag: Tag, comptime merge_tags_spec: anytype) !Tensor(mergeTags(tags, out_tag, normalizeTags(merge_tags_spec))) {
            const merge_tags = normalizeTags(merge_tags_spec);
            const result_tags = mergeTags(tags, out_tag, merge_tags);
            var value = try tag_ops.mergeAxesView(tags, self.asRawTensor(), out_tag, merge_tags);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), StridedViewBackward(tags, result_tags), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        /// Arbitrary row-major reinterpretation to `new_tags_spec` /
        /// `new_shape` (torch.reshape): the element count must match
        /// (`InvalidShape` otherwise). View-or-materialize like torch — a
        /// contiguous source stays a zero-copy view; a non-contiguous one
        /// materializes first (the `flatten` rule). Composed flatten →
        /// split, so gradients come from the existing exact view records;
        /// when the target rank is > 1 and gradients are tracked this
        /// requires an active exec scope (see `nllLoss`); a rank-1 target
        /// degenerates to plain `flatten` (no scope needed).
        pub fn reshape(
            self: *const Self,
            ctx: *ExecContext,
            comptime new_tags_spec: anytype,
            new_shape: [normalizeTags(new_tags_spec).len]usize,
        ) !Tensor(normalizeTags(new_tags_spec)) {
            const new_tags = comptime normalizeTags(new_tags_spec);
            if (comptime new_tags.len == 1) {
                if (self.asRawTensor().len() != new_shape[0]) return TensorError.InvalidShape;
                return self.flatten(ctx, new_tags[0]);
            }
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            var flat = try self.flatten(ctx, new_tags[0]);
            defer flat.deinit();
            return flat.split(ctx, new_tags[0], new_tags_spec, new_shape);
        }

        pub fn broadcastTo(
            self: *const Self,
            ctx: *ExecContext,
            comptime target_tags_spec: anytype,
            target_shape: [normalizeTags(target_tags_spec).len]usize,
        ) !Tensor(normalizeTags(target_tags_spec)) {
            const target_tags = normalizeTags(target_tags_spec);
            var value = try broadcastTensorTo(tags, self.asRawTensor(), target_tags, target_shape);
            errdefer value.deinit();
            return finishOp(target_tags, ctx, value, self.requiresGrad(), BroadcastBackward(tags, target_tags), .{ ctx.allocator, self.grad_state, &self.value });
        }

        pub fn add(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.add, self, ctx, other);
        }

        pub fn sub(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.sub, self, ctx, other);
        }

        pub fn mul(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.mul, self, ctx, other);
        }

        pub fn div(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.div, self, ctx, other);
        }

        pub fn scale(self: *const Self, ctx: *ExecContext, scalar_value: f32) !Self {
            var value = try ctx.scale(self.asRawTensor(), scalar_value);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), ScaleBackward(tags), .{ ctx.allocator, self.grad_state, scalar_value });
        }

        /// Consume `self` and return `self + other`, reusing `self`'s storage
        /// when the runtime can safely take it in place. No-grad only: consuming
        /// a graph value would invalidate autograd state.
        pub fn takeAddNoGrad(self: *Self, ctx: *ExecContext, other: anytype) !Self {
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            if (self.requiresGrad() or other_ptr.requiresGrad()) return error.UnsupportedGradient;
            if (self.scope_owned) return error.ActiveExecScopeUnsupported;
            var value = try ctx.takeAdd(&self.value, other_ptr.asRawTensor());
            errdefer value.deinit();
            self.* = undefined;
            return finishNoGrad(tags, ctx, value);
        }

        /// Consume `self` and return `self * scalar_value`, reusing `self`'s
        /// storage when possible. No-grad only, for the same ownership reason as
        /// `takeAddNoGrad`.
        pub fn takeScaleNoGrad(self: *Self, ctx: *ExecContext, scalar_value: f32) !Self {
            if (self.requiresGrad()) return error.UnsupportedGradient;
            if (self.scope_owned) return error.ActiveExecScopeUnsupported;
            var value = try ctx.takeScale(&self.value, scalar_value);
            errdefer value.deinit();
            self.* = undefined;
            return finishNoGrad(tags, ctx, value);
        }

        /// `self + scalar_value` (elementwise). Differentiable (grad passes through).
        pub fn addScalar(self: *const Self, ctx: *ExecContext, scalar_value: f32) !Self {
            var value = try ctx.addScalar(self.asRawTensor(), scalar_value);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), AddScalarBackward(tags), .{ ctx.allocator, self.grad_state });
        }

        /// `self - scalar_value` (= `addScalar(-scalar_value)`).
        pub fn subScalar(self: *const Self, ctx: *ExecContext, scalar_value: f32) !Self {
            return self.addScalar(ctx, -scalar_value);
        }

        /// `self / scalar_value` (= `scale(1/scalar_value)`).
        pub fn divScalar(self: *const Self, ctx: *ExecContext, scalar_value: f32) !Self {
            return self.scale(ctx, 1.0 / scalar_value);
        }

        /// `self ^ exponent` (elementwise; defined for positive `self`).
        /// Differentiable: `d/dx x^c = c·x^(c-1)`.
        pub fn powScalar(self: *const Self, ctx: *ExecContext, exponent: f32) !Self {
            var value = try ctx.powScalar(self.asRawTensor(), exponent);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), PowScalarBackward(tags), .{ ctx.allocator, self.grad_state, self.asRawTensor(), exponent });
        }

        /// `log(1 + self)` (elementwise). Differentiable: `d/dx = 1/(1+x)`.
        pub fn log1p(self: *const Self, ctx: *ExecContext) !Self {
            return self.unaryDifferentiable(ctx, .log1p);
        }

        /// Inverted dropout: element i keeps `x[i] / (1-p)` iff the 53-bit
        /// uniform of `rng.at(seed, i)` is < 1-p, else 0. The mask is never
        /// stored — forward, backward, and any `checkpoint` recompute all
        /// regenerate it from (seed, element index), so the op is a
        /// deterministic pure function of (input, p, seed). Requires
        /// `0 <= p < 1`; `p == 0` returns an identity view (no copy,
        /// gradients flow).
        ///
        /// Seed discipline: pass an explicit fresh seed per call — e.g.
        /// derived per step/layer as `rng.at(base_seed, step * layers + layer)`
        /// — since reusing a seed reuses the mask. Eval mode is caller-side:
        /// simply don't call dropout at eval.
        pub fn dropout(self: *const Self, ctx: *ExecContext, p: f32, seed: u64) !Self {
            if (p == 0) return self.withTags(ctx, tags);
            var value = try ctx.dropoutForward(self.asRawTensor(), p, seed);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), DropoutBackward(tags), .{ ctx.allocator, self.grad_state, p, seed });
        }

        pub fn causalDepthwiseConv1d(
            self: *const Self,
            ctx: *ExecContext,
            comptime time_tag: Tag,
            comptime channel_tag: Tag,
            comptime tap_tag: Tag,
            kernel: *const Tensor(.{ channel_tag, tap_tag }),
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

            var value = try ctx.causalDepthwiseConv1dAxisRank(tag_rank, self.asRawTensor(), kernel.asRawTensor(), time_axis, channel_axis, state);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or kernel.requiresGrad(), CausalDepthwiseConv1dBackward(tags, .{ channel_tag, tap_tag }, time_axis, channel_axis), .{ ctx.allocator, self.grad_state, kernel.grad_state, self.asRawTensor(), kernel.asRawTensor(), state });
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

        /// Per-channel Snake activation (the DAC codec op):
        /// `y[t,c] = x[t,c] + inv_b[c] * sin(alpha[c] * x[t,c])^2`. `alpha` is
        /// the stored `*.snake*.alpha` vector; `inv_b` is precomputed by the
        /// loader as `1/(alpha + 1e-9)` — the epsilon is NOT folded in here.
        /// Differentiable in all three operands. `alpha` and `inv_b` are
        /// INDEPENDENT tensor inputs at this level: no gradient flows through
        /// the `inv_b = 1/(alpha + 1e-9)` load-time relation — a trainer
        /// wanting a single alpha parameter must chain through it itself.
        pub fn snake(
            self: *const Self,
            ctx: *ExecContext,
            comptime channel_tag: Tag,
            alpha: *const Tensor(.{channel_tag}),
            inv_b: *const Tensor(.{channel_tag}),
        ) !Self {
            const channel_axis = comptime axis(channel_tag);
            comptime {
                if (tag_rank != 2) @compileError("snake requires a rank-2 input");
                if (channel_axis != 1) @compileError("snake requires storage order [time, channel]");
            }

            var value = try ctx.snakeRows(self.asRawTensor(), alpha.asRawTensor(), inv_b.asRawTensor());
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or alpha.requiresGrad() or inv_b.requiresGrad(), SnakeBackward(tags), .{ ctx.allocator, self.grad_state, alpha.grad_state, inv_b.grad_state, self.asRawTensor(), alpha.asRawTensor(), inv_b.asRawTensor() });
        }

        /// GroupNorm over `[time, channel]` rows (ggml semantics; see
        /// `groupNormAxisRank`): per group of channel columns, f64-accumulated
        /// mean + biased variance over all time × (C/groups) elements, then
        /// `y = (x − mean)/sqrt(var + eps)` in f32 (eps inside the sqrt), with
        /// the optional per-channel affine `y*weight + bias` applied AFTER
        /// normalization. Differentiable in the input and the optional affine
        /// operands (statistics are recomputed in the backward — nothing is
        /// saved from the forward).
        pub fn groupNorm(
            self: *const Self,
            ctx: *ExecContext,
            comptime channel_tag: Tag,
            groups: usize,
            eps: f32,
            weight: ?*const Tensor(.{channel_tag}),
            bias: ?*const Tensor(.{channel_tag}),
        ) !Self {
            const channel_axis = comptime axis(channel_tag);
            comptime {
                if (tag_rank != 2) @compileError("groupNorm requires a rank-2 input");
                if (channel_axis != 1) @compileError("groupNorm requires storage order [time, channel]");
            }
            var any_grad = self.requiresGrad();
            var weight_parent: ?*GradState = null;
            const weight_raw: ?*const RawTensor = if (weight) |w| blk: {
                any_grad = any_grad or w.requiresGrad();
                weight_parent = w.grad_state;
                break :blk w.asRawTensor();
            } else null;
            var bias_parent: ?*GradState = null;
            const bias_raw: ?*const RawTensor = if (bias) |b| blk: {
                any_grad = any_grad or b.requiresGrad();
                bias_parent = b.grad_state;
                break :blk b.asRawTensor();
            } else null;

            var value = try ctx.groupNormAxisRank(self.asRawTensor(), groups, eps, weight_raw, bias_raw);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, any_grad, GroupNormBackward(tags), .{ ctx.allocator, self.grad_state, weight_parent, bias_parent, self.asRawTensor(), weight_raw, groups, eps });
        }

        pub fn gated(
            self: *const Self,
            ctx: *ExecContext,
            other: anytype,
            comptime op: GatedOp,
        ) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return gatedPointwise(op, self, ctx, other);
        }

        pub fn glu(
            self: *const Self,
            ctx: *ExecContext,
            other: anytype,
        ) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return self.gated(ctx, other, .glu);
        }

        pub fn swiglu(
            self: *const Self,
            ctx: *ExecContext,
            other: anytype,
        ) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return self.gated(ctx, other, .swiglu);
        }

        /// GeGLU: `self * gelu(other)` (GELU tanh approximation). Mirrors
        /// `swiglu` but with the GELU gate Gemma's GeGLU FFN uses.
        pub fn geglu(
            self: *const Self,
            ctx: *ExecContext,
            other: anytype,
        ) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return self.gated(ctx, other, .geglu);
        }

        /// Split-gated activation over `tag`: halves that axis and gates one
        /// half with the other, per `op` — the gate-half conventions DIFFER
        /// (ggml parity): `.swiglu` gates with the FIRST half
        /// (`silu(first) * second`), `.glu` with the SECOND
        /// (`first * sigmoid(second)`). `out_tag == tag` is allowed (keeps
        /// the tag on the halved axis — the raw numeric-tag form Parakeet
        /// uses). `.geglu` is a compile error: no split-geglu kernel or
        /// gate-half convention exists.
        pub fn splitGated(self: *const Self, ctx: *ExecContext, comptime op: GatedOp, comptime tag: Tag, comptime out_tag: Tag) !Tensor(replaceTag(tags, tag, out_tag)) {
            const result_tags = replaceTag(tags, tag, out_tag);
            const split_axis = comptime axis(tag);
            switch (comptime op) {
                .swiglu => {
                    var value = try ctx.splitSwiGluAxisRank(tag_rank, self.asRawTensor(), split_axis);
                    errdefer value.deinit();
                    return finishOp(result_tags, ctx, value, self.requiresGrad(), SplitSwiGluBackward(tags, split_axis), .{ ctx.allocator, self.grad_state, self.asRawTensor() });
                },
                .glu => {
                    var value = try ctx.splitGluAxisRank(tag_rank, self.asRawTensor(), split_axis);
                    errdefer value.deinit();
                    return finishOp(result_tags, ctx, value, self.requiresGrad(), SplitGluBackward(tags, split_axis), .{ ctx.allocator, self.grad_state, self.asRawTensor() });
                },
                .geglu => @compileError("splitGated: no split-geglu kernel or gate-half convention exists (compose `unary(.gelu_quant)` + `mul`, or use `geglu` on separate halves)"),
            }
        }

        pub fn unary(self: *const Self, ctx: *ExecContext, comptime op: UnaryOp) !Self {
            return switch (op) {
                .relu => self.relu(ctx),
                .exp, .sqrt, .rsqrt, .sigmoid, .silu, .log, .log1p, .neg, .abs, .sin, .cos, .tanh, .fast_tanh, .gelu, .quick_gelu, .softcap_30, .softcap_15, .gelu_quant, .elu, .gelu_erf, .floor, .ceil, .round, .sign, .reciprocal => self.unaryDifferentiable(ctx, op),
            };
        }

        /// Lift a comptime scalar op to a differentiable elementwise tensor
        /// op — the user-extensible escape hatch when `UnaryOp` (a closed
        /// kernel enum) lacks the function. `Op` declares
        /// `forward(x, extra) f32` and `backward(x, y, grad_y, extra) f32`
        /// (returning the propagated dL/dx); see `elemental.zig` for the
        /// full contract. Strided inputs are accepted (materialized for the
        /// scalar loop); the result is owned and contiguous. `extra` is
        /// captured by value in the backward node (the `customVjp` lifetime
        /// contract: pointees must outlive backward).
        pub fn elementalUnary(self: *const Self, ctx: *ExecContext, comptime Op: type, extra: anytype) !Self {
            return elemental.unary(Self, ctx, Op, extra, self);
        }

        /// Binary `elementalUnary` with the standard pointwise tag-broadcast
        /// rule (result tags = left tags ++ right-only tags; shared dims
        /// must match or broadcast). `Op` declares `forward(a, b, extra)`
        /// plus `backwardA`/`backwardB` returning dL/da and dL/db at the
        /// result shape — broadcast operands get their gradient sum-reduced
        /// back to their own shape, exactly like `add`/`mul`.
        pub fn elementalBinary(
            self: *const Self,
            ctx: *ExecContext,
            other: anytype,
            comptime Op: type,
            extra: anytype,
        ) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            const right_tags = TensorObject(@TypeOf(other)).axis_tags;
            const OutT = Tensor(pointwiseResultTags(tags, right_tags));
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            return elemental.binary(OutT, tags, right_tags, ctx, Op, extra, self, other_ptr);
        }

        /// Elementwise maximum of two tensors (torch.maximum), with the
        /// standard pointwise tag-broadcast rule and the full SIMD binary
        /// kernel path (same tier as `add`/`mul`). NaN in either operand
        /// propagates NaN (the torch convention — NOT the IEEE maxNum rule
        /// bare `@max` follows). Differentiable in both operands: the
        /// gradient goes to the larger operand, and is split evenly on
        /// exact ties (torch's subgradient).
        pub fn maximum(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.max, self.*, ctx, other);
        }

        /// Elementwise minimum of two tensors (torch.minimum); see
        /// `maximum` for the NaN, tie-gradient, and kernel-tier notes.
        pub fn minimum(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.min, self.*, ctx, other);
        }

        /// Elementwise `self ^ other` (torch.pow with a tensor exponent),
        /// with the standard pointwise tag-broadcast rule; `powScalar` is
        /// the scalar-exponent fast path. Follows `std.math.pow` domain
        /// semantics (negative base with a non-integer exponent is NaN,
        /// `0^0 = 1`). Differentiable in both operands: dL/da uses
        /// `b·a^(b-1)`; dL/db uses `ln(a)·a^b` and is NaN for `a < 0` and
        /// non-finite at `a = 0` — meaningful only for positive bases, as
        /// in torch.
        pub fn pow(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            const Op = struct {
                pub fn forward(a: f32, b: f32, extra: void) f32 {
                    _ = extra;
                    return std.math.pow(f32, a, b);
                }
                pub fn backwardA(a: f32, b: f32, y: f32, grad_y: f32, extra: void) f32 {
                    _ = y;
                    _ = extra;
                    return grad_y * b * std.math.pow(f32, a, b - 1);
                }
                pub fn backwardB(a: f32, b: f32, y: f32, grad_y: f32, extra: void) f32 {
                    _ = b;
                    _ = extra;
                    return grad_y * y * @log(a);
                }
            };
            return self.elementalBinary(ctx, other, Op, {});
        }

        pub fn relu(self: *const Self, ctx: *ExecContext) !Self {
            var value = try ctx.relu(self.asRawTensor());
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), ReluBackward, .{ ctx.allocator, self.grad_state, &self.value });
        }

        pub fn leakyRelu(self: *const Self, ctx: *ExecContext, negative_slope: f32) !Self {
            var value = try ctx.leakyRelu(self.asRawTensor(), negative_slope);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), LeakyReluBackward, .{ ctx.allocator, self.grad_state, &self.value, negative_slope });
        }

        // Differentiable unary family: one decl alias per op, generated from
        // `UnaryMethod` so the forwarding template lives in one place. `relu`
        // stays hand-written above (it has a dedicated backward).
        pub const exp = UnaryMethod(.exp).call;
        pub const sqrt = UnaryMethod(.sqrt).call;
        pub const rsqrt = UnaryMethod(.rsqrt).call;
        pub const sigmoid = UnaryMethod(.sigmoid).call;
        pub const silu = UnaryMethod(.silu).call;
        pub const log = UnaryMethod(.log).call;
        pub const neg = UnaryMethod(.neg).call;
        pub const abs = UnaryMethod(.abs).call;
        pub const sin = UnaryMethod(.sin).call;
        pub const cos = UnaryMethod(.cos).call;
        pub const tanh = UnaryMethod(.tanh).call;
        pub const fastTanh = UnaryMethod(.fast_tanh).call;
        pub const softcap30 = UnaryMethod(.softcap_30).call;
        pub const softcap15 = UnaryMethod(.softcap_15).call;
        pub const gelu = UnaryMethod(.gelu).call;
        pub const quickGelu = UnaryMethod(.quick_gelu).call;
        pub const elu = UnaryMethod(.elu).call;
        pub const geluErf = UnaryMethod(.gelu_erf).call;
        pub const floor = UnaryMethod(.floor).call;
        pub const ceil = UnaryMethod(.ceil).call;
        pub const round = UnaryMethod(.round).call;
        pub const sign = UnaryMethod(.sign).call;
        pub const reciprocal = UnaryMethod(.reciprocal).call;

        fn UnaryMethod(comptime op: UnaryOp) type {
            return struct {
                fn call(self: *const Self, ctx: *ExecContext) !Self {
                    return self.unaryDifferentiable(ctx, op);
                }
            };
        }

        pub fn clamp(self: *const Self, ctx: *ExecContext, min_value: f32, max_value: f32) !Self {
            var value = try ctx.clamp(self.asRawTensor(), min_value, max_value);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), ClampBackward, .{ ctx.allocator, self.grad_state, &self.value, min_value, max_value });
        }

        pub fn sum(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            const result_tags = removeTag(tags, tag);
            var value = try ctx.sumAxisRank(tag_rank, self.asRawTensor(), axis(tag));
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), SumBackward(tags, result_tags), .{ ctx.allocator, self.grad_state, &self.value });
        }

        pub fn mean(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            const result_tags = removeTag(tags, tag);
            const reduce_axis = comptime axis(tag);
            var value = try ctx.meanAxisRank(tag_rank, self.asRawTensor(), reduce_axis);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), MeanBackward(tags, result_tags, reduce_axis), .{ ctx.allocator, self.grad_state, &self.value });
        }

        /// Cumulative sum along `tag` (torch.cumsum), preserving shape:
        /// `y[..., i, ...] = Σ_{j <= i} x[..., j, ...]`. Differentiable:
        /// the gradient is the reversed cumulative (suffix) sum of the
        /// upstream gradient. Both passes are serial per row — bitwise
        /// deterministic for any thread count.
        pub fn cumsum(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Self {
            const scan_axis = comptime axis(tag);
            var value = try ctx.cumsumAxisRank(tag_rank, self.asRawTensor(), scan_axis);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), CumsumBackward(tags, scan_axis), .{ ctx.allocator, self.grad_state });
        }

        /// Product along `tag` (torch.prod over a dim), the tag removed.
        /// Serial per row (the `cumsum` determinism contract).
        /// Differentiable with torch's zero-handling: zero-free rows get
        /// `g·(Π x)/x_i`; exactly one zero routes the whole gradient to the
        /// zero slot; two or more zeros kill the row's gradient.
        pub fn prod(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            const result_tags = removeTag(tags, tag);
            const reduce_axis = comptime axis(tag);
            var value = try ctx.prodAxisRank(tag_rank, self.asRawTensor(), reduce_axis);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), ProdBackward(tags, reduce_axis), .{ ctx.allocator, self.grad_state, &self.value });
        }

        /// Inclusive running product along `tag` (torch.cumprod),
        /// shape-preserving; serial per row (the `cumsum` determinism
        /// contract). Differentiable: zero-free rows use the O(n)
        /// reverse-scan closed form; rows containing a zero fall back to
        /// the exact division-free O(n²) expansion (torch semantics).
        pub fn cumprod(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Self {
            const scan_axis = comptime axis(tag);
            var value = try ctx.cumprodAxisRank(tag_rank, self.asRawTensor(), scan_axis);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), CumprodBackward(tags, scan_axis), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

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

        pub fn sumAll(self: *const Self, ctx: *ExecContext) !Tensor(.{}) {
            var value = try ctx.sum(self.asRawTensor());
            errdefer value.deinit();
            return finishOp(.{}, ctx, value, self.requiresGrad(), SumBackward(tags, .{}), .{ ctx.allocator, self.grad_state, &self.value });
        }

        pub fn sumMany(self: *const Self, ctx: *ExecContext, comptime reduce_tags_spec: anytype) !Tensor(removeTags(tags, normalizeTags(reduce_tags_spec))) {
            const reduce_tags = normalizeTags(reduce_tags_spec);
            const result_tags = removeTags(tags, reduce_tags);
            var value = try tag_ops.sumManyTensor(tags, self.asRawTensor(), ctx, reduce_tags);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), SumBackward(tags, result_tags), .{ ctx.allocator, self.grad_state, &self.value });
        }

        pub fn flatten(self: *const Self, ctx: *ExecContext, comptime out_tag: Tag) !Tensor(.{out_tag}) {
            var value = try tag_ops.flattenTensor(ctx, self.asRawTensor());
            errdefer value.deinit();
            return finishOp(.{out_tag}, ctx, value, self.requiresGrad(), ReshapeBackward, .{ ctx.allocator, self.grad_state, &self.value });
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

        /// Reverse the order of `tag` (torch.flip on one dim): a gather with
        /// the reversed index permutation, so the value is a copy and
        /// GatherBackward routes the gradient exactly (a permutation
        /// scatters 1-to-1).
        pub fn flip(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Self {
            const n = self.asRawTensor().shape.at(axis(tag));
            const indices = try ctx.allocator.alloc(usize, n);
            defer ctx.allocator.free(indices);
            for (indices, 0..) |*index, i| index.* = n - 1 - i;
            return self.gather(ctx, tag, indices, tag);
        }

        /// Rotate `tag` by `shift` positions (torch.roll on one dim): the
        /// element at index i moves to index `(i + shift) mod n`; elements
        /// shifted past the end re-enter at the front. Negative shifts roll
        /// the other way. Implemented as a gather with the rotated index
        /// permutation (exact gradient, like `flip`).
        pub fn roll(self: *const Self, ctx: *ExecContext, comptime tag: Tag, shift: isize) !Self {
            const n = self.asRawTensor().shape.at(axis(tag));
            const indices = try ctx.allocator.alloc(usize, n);
            defer ctx.allocator.free(indices);
            // out[i] = x[(i - shift) mod n]; s = shift mod n in [0, n).
            const s: usize = @intCast(@mod(shift, @as(isize, @intCast(n))));
            for (indices, 0..) |*index, i| index.* = (i + n - s) % n;
            return self.gather(ctx, tag, indices, tag);
        }

        /// `roll` with one shift per section: every section (the sub-vector
        /// obtained by fixing all axes except `tag`) rotates by its own
        /// offset, with `roll`'s sign convention
        /// (`out[..., j, ...] = self[..., (j - shift) mod n, ...]`).
        /// `offsets` holds one shift per section, indexed row-major over the
        /// remaining axes in `self`'s tag order (host-side control data,
        /// like `gather` indices) — `offsets.len` must equal
        /// `numel / dim(tag)` or the call errors with `InvalidShape`; for a
        /// rank-1 tensor that is a single element and `rollBy` matches
        /// `roll`. A per-section permutation, composed flatten + gather +
        /// split, so the gradient is exact (the inverse per-section roll).
        /// When gradients are tracked this requires an active exec scope
        /// (see `maskedSelect`).
        pub fn rollBy(self: *const Self, ctx: *ExecContext, comptime tag: Tag, offsets: []const isize) !Self {
            const roll_axis = comptime axis(tag);
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const raw = self.asRawTensor();
            const n = raw.shape.at(roll_axis);
            var inner: usize = 1;
            inline for (roll_axis + 1..tensor_rank) |i| inner *= raw.shape.at(i);
            const total = raw.len();
            const sections = total / n;
            if (offsets.len != sections) return TensorError.InvalidShape;

            // Normalize each section's shift once: s in [0, n) with
            // out[j] = x[(j - s) mod n], matching `roll`.
            const normalized = try ctx.allocator.alloc(usize, sections);
            defer ctx.allocator.free(normalized);
            for (normalized, offsets) |*s, shift| s.* = @intCast(@mod(shift, @as(isize, @intCast(n))));

            const indices = try ctx.allocator.alloc(usize, total);
            defer ctx.allocator.free(indices);
            const section_stride = n * inner;
            for (indices, 0..) |*index, i| {
                const outer = i / section_stride;
                const j = (i % section_stride) / inner;
                const inner_pos = i % inner;
                const s = normalized[outer * inner + inner_pos];
                index.* = outer * section_stride + ((j + n - s) % n) * inner + inner_pos;
            }

            var flat = try self.flatten(ctx, tag);
            defer flat.deinit();
            var rolled = try flat.gather(ctx, tag, indices, tag);
            defer rolled.deinit();
            return rolled.split(ctx, tag, tags, self.shape());
        }

        /// Non-circular `rollBy`: same per-section offsets and sign
        /// convention, but positions shifted in from outside the axis hold
        /// the constant `fill` instead of wrapping
        /// (`out[..., j, ...] = self[..., j - shift, ...]` when in bounds,
        /// else `fill`). `fill` is a constant and receives no gradient;
        /// source positions shifted out of the axis receive zero gradient
        /// (composed gather + maskedFill — the fill mask zeroes their
        /// upstream gradient before the gather scatters it back). Same
        /// `offsets` layout, `InvalidShape`, and exec-scope rules as
        /// `rollBy`.
        pub fn shiftBy(self: *const Self, ctx: *ExecContext, comptime tag: Tag, offsets: []const isize, fill: f32) !Self {
            const shift_axis = comptime axis(tag);
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const raw = self.asRawTensor();
            const n = raw.shape.at(shift_axis);
            var inner: usize = 1;
            inline for (shift_axis + 1..tensor_rank) |i| inner *= raw.shape.at(i);
            const total = raw.len();
            const sections = total / n;
            if (offsets.len != sections) return TensorError.InvalidShape;

            const indices = try ctx.allocator.alloc(usize, total);
            defer ctx.allocator.free(indices);
            const fill_mask = try ctx.allocator.alloc(f32, total);
            defer ctx.allocator.free(fill_mask);
            const section_stride = n * inner;
            const n_signed: isize = @intCast(n);
            for (indices, fill_mask, 0..) |*index, *fm, i| {
                const outer = i / section_stride;
                const j = (i % section_stride) / inner;
                const inner_pos = i % inner;
                const shift = offsets[outer * inner + inner_pos];
                const src_j = @as(isize, @intCast(j)) - shift;
                const section_base = outer * section_stride + inner_pos;
                if (src_j >= 0 and src_j < n_signed) {
                    index.* = section_base + @as(usize, @intCast(src_j)) * inner;
                    fm.* = 0;
                } else {
                    // Placeholder source; maskedFill overwrites the value and
                    // zeroes its gradient contribution.
                    index.* = section_base;
                    fm.* = 1;
                }
            }

            var flat = try self.flatten(ctx, tag);
            defer flat.deinit();
            var shifted_flat = try flat.gather(ctx, tag, indices, tag);
            defer shifted_flat.deinit();
            var shifted = try shifted_flat.split(ctx, tag, tags, self.shape());
            defer shifted.deinit();
            var mask = try Self.fromSlice(ctx, self.shape(), fill_mask);
            defer mask.deinit();
            return shifted.maskedFill(ctx, mask, fill);
        }

        pub fn narrow(self: *const Self, ctx: *ExecContext, comptime tag: Tag, start: usize, length: usize) !Self {
            const slice_axis = comptime axis(tag);
            var value = try ctx.narrowAxisRank(tag_rank, self.asRawTensor(), slice_axis, start, length);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), NarrowBackward(tags, slice_axis), .{ ctx.allocator, self.grad_state, &self.value, start });
        }

        /// `narrow` with a step (torch basic slicing `x[start::step]` along
        /// one axis): `length` elements at `start, start+step, …`.
        /// `step == 0`, `length == 0` (zero-size tensors are not
        /// representable), or a last element at or past `dim(tag)` error
        /// with `InvalidShape`. On a no-grad tensor the result is a
        /// zero-copy strided **view** (retains and aliases the source
        /// buffer, like `narrow`); under gradients it lowers to `gather`
        /// over the stepped indices — a copy with the exact scatter-add
        /// record, the `flip`/`roll` precedent — so skipped positions
        /// receive zero gradient.
        pub fn sliceStep(self: *const Self, ctx: *ExecContext, comptime tag: Tag, start: usize, length: usize, step: usize) !Self {
            const slice_axis = comptime axis(tag);
            const raw = self.asRawTensor();
            const axis_dim = raw.shape.at(slice_axis);
            if (step == 0 or length == 0) return TensorError.InvalidShape;
            if (start >= axis_dim or start + (length - 1) * step >= axis_dim) return TensorError.InvalidShape;
            if (self.requiresGrad()) {
                const indices = try ctx.allocator.alloc(usize, length);
                defer ctx.allocator.free(indices);
                for (indices, 0..) |*index, i| index.* = start + i * step;
                return self.gather(ctx, tag, indices, tag);
            }
            var new_shape: [tensor_rank]usize = undefined;
            var new_strides: [tensor_rank]usize = undefined;
            inline for (0..tensor_rank) |i| {
                new_shape[i] = raw.shape.at(i);
                new_strides[i] = raw.strides.at(i);
            }
            new_shape[slice_axis] = length;
            new_strides[slice_axis] = raw.strides.at(slice_axis) * step;
            var value = try self.value.viewWithStridesOffset(&new_shape, &new_strides, start * raw.strides.at(slice_axis));
            errdefer value.deinit();
            return finishNoGrad(tags, ctx, value);
        }

        /// Main diagonal over the (`tag_a`, `tag_b`) plane (torch.diagonal,
        /// offset 0): a zero-copy strided **view** of length
        /// `min(dim(tag_a), dim(tag_b))` — element i is `self[.., i, .., i,
        /// ..]`. Both tags are removed and the diagonal axis is appended
        /// LAST as `out_tag` (the torch axis order). Works at any rank
        /// carrying both tags; differentiable (strided-view scatter —
        /// off-diagonal positions receive zero gradient).
        pub fn diagonal(self: *const Self, ctx: *ExecContext, comptime tag_a: Tag, comptime tag_b: Tag, comptime out_tag: Tag) !Tensor(insertTagAt(removeTag(removeTag(tags, tag_a), tag_b), out_tag, removeTag(removeTag(tags, tag_a), tag_b).len)) {
            comptime if (tag_a == tag_b) @compileError("diagonal requires two distinct tags");
            const base_tags = comptime removeTag(removeTag(tags, tag_a), tag_b);
            const result_tags = comptime insertTagAt(base_tags, out_tag, base_tags.len);
            const axis_a = comptime axis(tag_a);
            const axis_b = comptime axis(tag_b);
            const raw = self.asRawTensor();
            const diag_len = @min(raw.shape.at(axis_a), raw.shape.at(axis_b));
            var new_shape: [tensor_rank - 1]usize = undefined;
            var new_strides: [tensor_rank - 1]usize = undefined;
            var write: usize = 0;
            inline for (0..tensor_rank) |i| {
                if (i != axis_a and i != axis_b) {
                    new_shape[write] = raw.shape.at(i);
                    new_strides[write] = raw.strides.at(i);
                    write += 1;
                }
            }
            new_shape[tensor_rank - 2] = diag_len;
            new_strides[tensor_rank - 2] = raw.strides.at(axis_a) + raw.strides.at(axis_b);
            var value = try self.value.viewWithStridesOffset(&new_shape, &new_strides, 0);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), StridedViewBackward(tags, result_tags), .{ ctx.allocator, self.grad_state, &self.value, &value });
        }

        /// Sum of the main diagonal over the (`tag_a`, `tag_b`) plane
        /// (torch.trace generalized to named axes): composed diagonal →
        /// sum, so the gradient is the exact identity-matrix scatter. When
        /// gradients are tracked this requires an active exec scope (see
        /// `nllLoss`); errors with `ActiveExecScopeRequired` otherwise.
        pub fn trace(self: *const Self, ctx: *ExecContext, comptime tag_a: Tag, comptime tag_b: Tag) !Tensor(removeTag(removeTag(tags, tag_a), tag_b)) {
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            var diag_view = try self.diagonal(ctx, tag_a, tag_b, tag_a);
            defer diag_view.deinit();
            return diag_view.sum(ctx, tag_a);
        }

        /// Embed a rank-1 tensor as the main diagonal of an `[n, n]` matrix
        /// (torch.diag on a vector): zeros elsewhere. `out_tags_spec` names
        /// the two result axes. Composed zeros → setRows (flat diagonal
        /// positions) → reshape, so the gradient is the exact
        /// diagonal-extract; scope-required under gradients (see
        /// `nllLoss`).
        pub fn diag(self: *const Self, ctx: *ExecContext, comptime out_tags_spec: anytype) !Tensor(normalizeTags(out_tags_spec)) {
            comptime if (tag_count != 1) @compileError("diag embeds a rank-1 tensor as a matrix diagonal; for extraction use diagonal");
            const out_tags = comptime normalizeTags(out_tags_spec);
            comptime if (out_tags.len != 2) @compileError("diag builds a rank-2 [n, n] tensor: pass exactly two tags");
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const n = self.asRawTensor().shape.at(0);
            var base = try Self.zeros(ctx, .{n * n});
            defer base.deinit();
            const indices = try ctx.allocator.alloc(usize, n);
            defer ctx.allocator.free(indices);
            for (indices, 0..) |*index, i| index.* = i * (n + 1);
            var filled = try base.setRows(ctx, tags[0], indices, self);
            defer filled.deinit();
            return filled.reshape(ctx, out_tags_spec, .{ n, n });
        }

        /// Constant padding along `tag` (torch F.pad, mode='constant', one
        /// dim): the axis grows by `before + after`, the body sits at offset
        /// `before`, pad positions hold `fill`. Differentiable: the gradient
        /// is the narrow of the upstream gradient at offset `before` (pad
        /// positions are constants and drop their gradient).
        pub fn pad(self: *const Self, ctx: *ExecContext, comptime tag: Tag, before: usize, after: usize, fill: f32) !Self {
            const pad_axis = comptime axis(tag);
            var value = try ctx.padAxisRank(tag_rank, self.asRawTensor(), pad_axis, before, after, fill);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), PadBackward(tags, pad_axis), .{ ctx.allocator, self.grad_state, &self.value, before });
        }

        /// Zero-pad the two named axes: `constantPad2d` with fill 0.
        pub fn zeroPad2d(self: *const Self, ctx: *ExecContext, comptime h_tag: Tag, comptime w_tag: Tag, padding: anytype) !Self {
            return self.constantPad2d(ctx, h_tag, w_tag, padding, 0);
        }

        /// Constant-pad two named axes in one call. `padding` is either an
        /// integer, padding all four sides by that amount, or a
        /// 4-tuple/array `(left, right, top, bottom)`: left/right grow
        /// `w_tag`, top/bottom grow `h_tag`. A NEGATIVE entry crops that
        /// side instead of padding it; cropping an axis to zero size or
        /// below errors with `InvalidShape` (zero-size tensors are not
        /// representable). Any rank carrying both tags works —
        /// the remaining axes pass through. The result is always an owned
        /// regular tensor: an all-zero padding is the contiguous identity,
        /// and a crop-only result materializes rather than returning a
        /// strided view. Differentiable in `self`: pad positions are
        /// constants and drop their gradient, cropped source positions
        /// receive zero gradient — composed narrow + pad per axis, so the
        /// gradients come from the existing exact records. When gradients
        /// are tracked this requires an active exec scope (see
        /// `maskedSelect`), even for paddings that degenerate to fewer ops.
        pub fn constantPad2d(self: *const Self, ctx: *ExecContext, comptime h_tag: Tag, comptime w_tag: Tag, padding: anytype, fill: f32) !Self {
            comptime if (h_tag == w_tag) @compileError("constantPad2d: h_tag and w_tag must be distinct");
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const p = padding2dValues(padding);

            var cur: Self = undefined;
            var owned = false;
            var last_was_pad = false;
            errdefer if (owned) cur.deinit();
            inline for (0..2) |axis_i| {
                const tag = comptime if (axis_i == 0) h_tag else w_tag;
                const before: isize = if (axis_i == 0) p[2] else p[0];
                const after: isize = if (axis_i == 0) p[3] else p[1];
                if (before < 0 or after < 0) {
                    const src: *const Self = if (owned) &cur else self;
                    const remaining = @as(isize, @intCast(src.dim(tag))) + @min(before, 0) + @min(after, 0);
                    if (remaining <= 0) return TensorError.InvalidShape;
                    const next = try src.narrow(ctx, tag, @intCast(@max(-before, 0)), @intCast(remaining));
                    if (owned) cur.deinit();
                    cur = next;
                    owned = true;
                    last_was_pad = false;
                }
                if (before > 0 or after > 0) {
                    const src: *const Self = if (owned) &cur else self;
                    const next = try src.pad(ctx, tag, @intCast(@max(before, 0)), @intCast(@max(after, 0)), fill);
                    if (owned) cur.deinit();
                    cur = next;
                    owned = true;
                    last_was_pad = true;
                }
            }
            // The result contract is a regular tensor in every case: an
            // all-zero padding is the contiguous identity, and a crop-only
            // result (narrow is a strided view) materializes so
            // `dataConst` works like any padded output.
            if (!owned) return self.contiguous(ctx);
            if (!last_was_pad) {
                const out = try cur.contiguous(ctx);
                cur.deinit();
                owned = false;
                return out;
            }
            return cur;
        }

        pub fn concat(self: *const Self, ctx: *ExecContext, comptime tag: Tag, others: []const *const Self) !Self {
            var any_grad = self.requiresGrad();
            for (others) |other| any_grad = any_grad or other.requiresGrad();

            const input_count = others.len + 1;
            var raw_inputs_stack: [concat_inline_inputs]*const RawTensor = undefined;
            const raw_inputs = if (input_count <= raw_inputs_stack.len)
                raw_inputs_stack[0..input_count]
            else
                try ctx.allocator.alloc(*const RawTensor, input_count);
            defer if (input_count > raw_inputs_stack.len) ctx.allocator.free(raw_inputs);
            raw_inputs[0] = self.asRawTensor();
            for (others, raw_inputs[1..]) |other, *raw| raw.* = other.asRawTensor();

            // Backward metadata is only materialized when finishOp will attach
            // a backward record (same gate finishOp itself applies); its
            // no-grad branch never reads create_args, so the empty slices are
            // never touched there. ConcatBackward copies both slices when it
            // is constructed, so stack-backed temporaries are safe on the grad
            // path.
            const track_grad = any_grad and control.isGradEnabled();
            var parents_stack: [concat_inline_inputs]?*GradState = undefined;
            var sizes_stack: [concat_inline_inputs]usize = undefined;
            var parents: []?*GradState = parents_stack[0..0];
            var sizes: []usize = sizes_stack[0..0];
            const metadata_on_heap = track_grad and input_count > parents_stack.len;
            defer if (metadata_on_heap) {
                ctx.allocator.free(parents);
                ctx.allocator.free(sizes);
            };
            if (track_grad) {
                if (metadata_on_heap) {
                    parents = try ctx.allocator.alloc(?*GradState, input_count);
                    sizes = try ctx.allocator.alloc(usize, input_count);
                } else {
                    parents = parents_stack[0..input_count];
                    sizes = sizes_stack[0..input_count];
                }
                parents[0] = self.grad_state;
                sizes[0] = self.asRawTensor().shape.at(axis(tag));
                for (others, parents[1..], sizes[1..]) |other, *parent, *size| {
                    parent.* = other.grad_state;
                    size.* = other.asRawTensor().shape.at(axis(tag));
                }
            }

            const concat_axis = comptime axis(tag);
            var value = try ctx.concatAxisRank(tag_rank, raw_inputs, concat_axis);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, any_grad, ConcatBackward(tags, concat_axis), .{ ctx.allocator, parents, sizes });
        }

        /// Stack `self` and `others` along a NEW axis tagged `new_tag`
        /// inserted at `axis_index` (torch.stack): composed as insertAxis on
        /// every input + concat, so the result is differentiable in ALL
        /// inputs through the multi-parent ConcatBackward. When gradients
        /// are tracked this requires an active exec scope (the inserted-axis
        /// intermediates are function-local graph nodes — see `nllLoss`);
        /// errors with `ActiveExecScopeRequired` otherwise.
        pub fn stack(
            self: *const Self,
            ctx: *ExecContext,
            comptime new_tag: Tag,
            comptime axis_index: usize,
            others: []const *const Self,
        ) !Tensor(insertTagAt(tags, new_tag, axis_index)) {
            var any_grad = self.requiresGrad();
            for (others) |other| any_grad = any_grad or other.requiresGrad();
            try requireScopeForComposedGrad(ctx, any_grad);

            const Expanded = Tensor(insertTagAt(tags, new_tag, axis_index));
            var expanded = try ctx.allocator.alloc(Expanded, others.len + 1);
            defer ctx.allocator.free(expanded);
            var created: usize = 0;
            // The inserted-axis views are composition temporaries: deinit is
            // a no-op under an exec scope (scope-owned) and a real release
            // in the unscoped no-grad arm.
            defer for (expanded[0..created]) |*view| view.deinit();

            expanded[0] = try self.insertAxis(ctx, new_tag, axis_index);
            created = 1;
            for (others) |other| {
                expanded[created] = try other.insertAxis(ctx, new_tag, axis_index);
                created += 1;
            }

            var ptrs_stack: [concat_inline_inputs]*const Expanded = undefined;
            const ptrs = if (others.len <= ptrs_stack.len)
                ptrs_stack[0..others.len]
            else
                try ctx.allocator.alloc(*const Expanded, others.len);
            defer if (others.len > ptrs_stack.len) ctx.allocator.free(ptrs);
            for (ptrs, expanded[1..]) |*ptr, *view| ptr.* = view;
            return expanded[0].concat(ctx, new_tag, ptrs);
        }

        /// Unbind `tag` (torch.unbind): fill the caller-provided `out` slice
        /// with the `dim(tag)` slices of `self`, each with `tag` removed
        /// (composed narrow + squeeze; differentiable per entry). `out.len`
        /// must equal `dim(tag)`. The CALLER owns every filled tensor and
        /// deinits each (under an exec scope they are scope-owned borrows and
        /// deinit is a no-op); on error, entries filled so far have already
        /// been released. When gradients are tracked this requires an active
        /// exec scope (see `nllLoss`); errors with `ActiveExecScopeRequired`
        /// otherwise.
        pub fn unbindInto(self: *const Self, ctx: *ExecContext, comptime tag: Tag, out: []Tensor(removeTag(tags, tag))) !void {
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            if (out.len != self.asRawTensor().shape.at(axis(tag))) return TensorError.InvalidShape;
            var filled: usize = 0;
            errdefer for (out[0..filled]) |*entry| entry.deinit();
            for (out, 0..) |*entry, i| {
                var sliced = try self.narrow(ctx, tag, i, 1);
                defer sliced.deinit();
                entry.* = try sliced.squeeze(ctx, tag);
                filled += 1;
            }
        }

        /// Repeat (tile) `tag` n times (torch.repeat on one dim): the axis
        /// length becomes `n·dim(tag)` with n back-to-back copies of `self`.
        /// `n == 1` returns a zero-copy identity view; `n > 1` is a concat
        /// of `self` with itself (one multi-parent node — gradients from all
        /// n copies accumulate into `self`). Errors with `InvalidShape` for
        /// `n == 0` (zero-size tensors are not representable).
        pub fn repeatAxis(self: *const Self, ctx: *ExecContext, comptime tag: Tag, n: usize) !Self {
            if (n == 0) return TensorError.InvalidShape;
            if (n == 1) return self.withTags(ctx, tags);
            const ptrs = try ctx.allocator.alloc(*const Self, n - 1);
            defer ctx.allocator.free(ptrs);
            for (ptrs) |*ptr| ptr.* = self;
            return self.concat(ctx, tag, ptrs);
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
            return self.scatterAlongImpl(ctx, tag, indices, src, true);
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
            return self.scatterAlongImpl(ctx, tag, indices, src, false);
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

        /// Index of the row maximum along `tag` (torch.argmax over a dim):
        /// a constant i64 tensor, no gradient. Caller-owned even under an
        /// exec scope (the typed-constant ownership rule).
        pub fn argmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .i64, .tags = removeTag(tags, tag) }) {
            const result_tags = removeTag(tags, tag);
            var value = try ctx.argmaxAxisRank(tag_rank, self.asRawTensor(), axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = .i64, .tags = result_tags }).fromTensor(ctx, value);
        }

        /// Max values over `tag` (the tag is removed like sum/mean; argmax
        /// returns the indices). The gradient flows only to the FIRST
        /// occurrence of the extremum along the axis (strict-comparison
        /// tie-break, like PyTorch's torch.max over a dim).
        pub fn max(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            return self.extremum(ctx, tag, .max);
        }

        /// Min values over `tag`; see `max` for gradient/tie-break semantics.
        pub fn min(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(removeTag(tags, tag)) {
            return self.extremum(ctx, tag, .min);
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

        pub fn topK(self: *const Self, ctx: *ExecContext, comptime tag: Tag, k: usize, comptime out_tag: Tag) !TopKResult(replaceTag(tags, tag, out_tag)) {
            const result_tags = replaceTag(tags, tag, out_tag);
            const raw = try ctx.topKAxisRank(tag_rank, self.asRawTensor(), axis(tag), k);
            var raw_values: ?RawTensor = raw.values;
            var raw_indices: ?tensor_mod.TensorOf(.i64) = raw.indices;
            errdefer if (raw_values) |*value| value.deinit();
            errdefer if (raw_indices) |*value| value.deinit();
            var values = try finishOp(result_tags, ctx, raw_values.?, self.requiresGrad(), TopKBackward(tags, axis(tag)), .{ ctx.allocator, self.grad_state, &self.value, &raw_indices.? });
            raw_values = null;
            errdefer values.deinit();
            var indices = try Tensor(.{ .dtype = .i64, .tags = result_tags }).fromTensor(ctx, raw_indices.?);
            raw_indices = null;
            errdefer indices.deinit();
            return .{ .values = values, .indices = indices };
        }

        /// Full sort along `tag` (torch.sort): values + the source index of
        /// each output position, both input-shaped. UNSTABLE sort — equal
        /// values keep no particular relative order (torch.sort is also
        /// unstable by default, but the tie ORDER may differ). NaN sorts
        /// LAST regardless of direction (documented divergence from
        /// torch.sort, which puts NaN first when descending — see
        /// `sortAxisRank`). Values are differentiable (the gradient scatters
        /// back through the saved indices, the topK VJP); indices are a
        /// constant i64 tensor (exact for any axis length, the repo-wide
        /// index convention).
        pub fn sort(self: *const Self, ctx: *ExecContext, comptime tag: Tag, descending: bool) !TopKResult(tags) {
            const raw = try ctx.sortAxisRank(tag_rank, self.asRawTensor(), axis(tag), descending);
            var raw_values: ?RawTensor = raw.values;
            var raw_indices: ?tensor_mod.TensorOf(.i64) = raw.indices;
            errdefer if (raw_values) |*value| value.deinit();
            errdefer if (raw_indices) |*value| value.deinit();
            var values = try finishOp(tags, ctx, raw_values.?, self.requiresGrad(), TopKBackward(tags, axis(tag)), .{ ctx.allocator, self.grad_state, &self.value, &raw_indices.? });
            raw_values = null;
            errdefer values.deinit();
            var indices = try Tensor(.{ .dtype = .i64, .tags = tags }).fromTensor(ctx, raw_indices.?);
            raw_indices = null;
            errdefer indices.deinit();
            return .{ .values = values, .indices = indices };
        }

        /// The indices arm of `sort` alone (torch.argsort): the source index
        /// of each sorted position as a constant i64 tensor (no grad; exact
        /// for any axis length). Same unstable-sort and NaN-last contract
        /// as `sort`.
        pub fn argsort(self: *const Self, ctx: *ExecContext, comptime tag: Tag, descending: bool) !Tensor(.{ .dtype = .i64, .tags = tags }) {
            var raw = try ctx.sortAxisRank(tag_rank, self.asRawTensor(), axis(tag), descending);
            raw.values.deinit();
            var raw_indices: ?tensor_mod.TensorOf(.i64) = raw.indices;
            errdefer if (raw_indices) |*value| value.deinit();
            return Tensor(.{ .dtype = .i64, .tags = tags }).fromTensor(ctx, raw_indices.?);
        }

        pub fn routerTopK(
            self: *const Self,
            ctx: *ExecContext,
            comptime expert_tag: Tag,
            k: usize,
            options: exec_mod.RouterTopKOptions,
            selected: []usize,
            weights: []f32,
        ) !void {
            comptime {
                if (tag_rank != 2) @compileError("routerTopK currently requires rank-2 [row, expert] logits");
                if (axis(expert_tag) != 1) @compileError("routerTopK requires the expert tag on the last axis");
            }
            if (self.requiresGrad()) return error.UnsupportedGradient;
            return ctx.routerTopK(self.asRawTensor(), k, options, selected, weights);
        }

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

        pub fn rmsNorm(self: *const Self, ctx: *ExecContext, comptime tag: Tag, eps: f32) !Self {
            const norm_axis = comptime axis(tag);
            var value = try ctx.rmsNormAxisRank(tag_rank, self.asRawTensor(), norm_axis, eps);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), RmsNormBackward(tags, norm_axis), .{ ctx.allocator, self.grad_state, &self.value, eps });
        }

        pub fn rmsNormMul(self: *const Self, ctx: *ExecContext, comptime tag: Tag, weight: *const Tensor(.{tag}), eps: f32) !Self {
            const norm_axis = comptime axis(tag);
            var value = try ctx.rmsNormMulAxisRank(tag_rank, self.asRawTensor(), weight.asRawTensor(), norm_axis, eps);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or weight.requiresGrad(), RmsNormMulBackward(tags, norm_axis), .{ ctx.allocator, self.grad_state, weight.grad_state, self.asRawTensor(), weight.asRawTensor(), eps });
        }

        pub fn rmsNormMulAdd(self: *const Self, ctx: *ExecContext, comptime tag: Tag, weight: *const Tensor(.{tag}), residual: *const Self, eps: f32) !Self {
            const norm_axis = comptime axis(tag);
            var value = try ctx.rmsNormMulAddAxisRank(tag_rank, self.asRawTensor(), weight.asRawTensor(), residual.asRawTensor(), norm_axis, eps);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or weight.requiresGrad() or residual.requiresGrad(), RmsNormMulAddBackward(tags, norm_axis), .{ ctx.allocator, self.grad_state, weight.grad_state, residual.grad_state, self.asRawTensor(), weight.asRawTensor(), eps });
        }

        pub fn rmsNormMulRopeHalfPrepared(
            self: *const Self,
            ctx: *ExecContext,
            comptime position_tag: Tag,
            comptime feature_tag: Tag,
            weight: *const Tensor(.{feature_tag}),
            eps: f32,
            table: *const exec_mod.RopeTable,
        ) !Self {
            const position_axis = comptime axis(position_tag);
            const feature_axis = comptime axis(feature_tag);
            var value = try ctx.rmsNormMulRopeAxisRankWithTable(
                tag_rank,
                self.asRawTensor(),
                weight.asRawTensor(),
                position_axis,
                feature_axis,
                eps,
                table,
                .half,
            );
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or weight.requiresGrad(), RmsNormMulRopeBackward(tags, position_axis, feature_axis, .half), .{ ctx.allocator, self.grad_state, weight.grad_state, self.asRawTensor(), weight.asRawTensor(), eps, table });
        }

        /// LayerNorm over `tag` with PyTorch semantics: y = (x − μ)/√(σ² + eps)
        /// with μ/σ² the per-row mean and BIASED variance (divide by N).
        /// `options` (comptime-validated struct literal): empty `.{}` = plain;
        /// `.{ .weight = &w, .bias = &b }` = fused affine `y*weight + bias`
        /// (torch elementwise_affine=true + bias=true; the fused kernel
        /// REQUIRES both together). Weight/bias are rank-1 `[tag_dim]`
        /// tensors — either tagged `.{tag}` (comptime-checked against the
        /// normalized axis) or numeric-tag `Tensor(1)` values (`._0`,
        /// Parakeet's raw weights; length checked at runtime).
        pub fn layerNorm(self: *const Self, ctx: *ExecContext, comptime tag: Tag, eps: f32, options: anytype) !Self {
            const Options = @TypeOf(options);
            comptime {
                if (@typeInfo(Options) != .@"struct") @compileError("layerNorm: options must be a struct literal, e.g. .{} or .{ .weight = &w, .bias = &b }");
                for (@typeInfo(Options).@"struct".fields) |field| {
                    if (!std.mem.eql(u8, field.name, "weight") and !std.mem.eql(u8, field.name, "bias"))
                        @compileError("layerNorm: unknown option ." ++ field.name);
                }
                if (@hasField(Options, "weight") != @hasField(Options, "bias"))
                    @compileError("layerNorm: the affine kernel requires .weight and .bias together");
            }
            const norm_axis = comptime axis(tag);
            if (comptime @hasField(Options, "weight")) {
                comptime {
                    normParamTagCheck(TensorObject(@TypeOf(options.weight)), "layerNorm", "weight", tag);
                    normParamTagCheck(TensorObject(@TypeOf(options.bias)), "layerNorm", "bias", tag);
                }
                const weight_ptr = tensorObjectPtrFrom(@TypeOf(options.weight), &options.weight);
                const bias_ptr = tensorObjectPtrFrom(@TypeOf(options.bias), &options.bias);
                var value = try ctx.layerNormAffineAxisRank(tag_rank, self.asRawTensor(), weight_ptr.asRawTensor(), bias_ptr.asRawTensor(), norm_axis, eps);
                errdefer value.deinit();
                return finishOp(
                    tags,
                    ctx,
                    value,
                    self.requiresGrad() or weight_ptr.requiresGrad() or bias_ptr.requiresGrad(),
                    LayerNormAffineBackward(tags, norm_axis),
                    .{ ctx.allocator, self.grad_state, weight_ptr.grad_state, bias_ptr.grad_state, self.asRawTensor(), weight_ptr.asRawTensor(), eps },
                );
            }
            var value = try ctx.layerNormAxisRank(tag_rank, self.asRawTensor(), norm_axis, eps);
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), LayerNormBackward(tags, norm_axis), .{ ctx.allocator, self.grad_state, &self.value, eps });
        }

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

            const one_hot_values = try ctx.allocator.alloc(f32, position_count * class_count);
            defer ctx.allocator.free(one_hot_values);
            @memset(one_hot_values, 0);
            for (0..outer) |outer_i| {
                for (0..inner) |inner_i| {
                    const label = labels[outer_i * inner + inner_i];
                    if (label >= class_count) return TensorError.IndexOutOfBounds;
                    one_hot_values[(outer_i * class_count + label) * inner + inner_i] = 1;
                }
            }
            var one_hot = try Self.fromSlice(ctx, raw_shape, one_hot_values);
            defer one_hot.deinit();

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

        /// L2-normalize along `tag`: y = x · rsqrt(Σ x² + eps). NOTE the eps
        /// placement: eps is added to the SQUARED norm before the reciprocal
        /// square root (the rmsNorm/qwen35 convention) — this deliberately
        /// differs from torch F.normalize, which computes x / max(‖x‖₂, eps).
        /// Composed from existing differentiable ops (mul → sum → addScalar →
        /// rsqrt → insertAxis → broadcast mul); differentiable in self. When
        /// gradients are tracked this requires an active exec scope (see
        /// `nllLoss`); errors with `ActiveExecScopeRequired` otherwise.
        pub fn l2Normalize(self: *const Self, ctx: *ExecContext, comptime tag: Tag, eps: f32) !Self {
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            const norm_axis = comptime axis(tag);
            var sq = try self.mul(ctx, self);
            defer sq.deinit();
            var sum_sq = try sq.sum(ctx, tag);
            defer sum_sq.deinit();
            var shifted = try sum_sq.addScalar(ctx, eps);
            defer shifted.deinit();
            var inv_norm = try shifted.rsqrt(ctx);
            defer inv_norm.deinit();
            var inv_axis = try inv_norm.insertAxis(ctx, tag, norm_axis);
            defer inv_axis.deinit();
            return self.mul(ctx, &inv_axis);
        }

        /// Vector norm along `tag` (torch.linalg.vector_norm for ord 1, 2,
        /// inf), with `tag` removed: `.l1` = Σ|x|, `.l2` = sqrt(Σ x²),
        /// `.inf` = max|x|. Composed from existing differentiable ops
        /// (abs/mul → sum/max → sqrt), so the gradients are the composed
        /// exact ones — like the naive composition (and torch), the `.l2`
        /// gradient at an all-zero vector is NaN (`sqrt'(0)`), and
        /// `.l1`/`.inf` follow `abs`'s sign convention at 0. When gradients
        /// are tracked this requires an active exec scope (see `nllLoss`);
        /// errors with `ActiveExecScopeRequired` otherwise.
        pub fn norm(self: *const Self, ctx: *ExecContext, comptime tag: Tag, comptime order: exec_mod.NormOrder) !Tensor(removeTag(tags, tag)) {
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            switch (comptime order) {
                .l1 => {
                    var magnitude = try self.abs(ctx);
                    defer magnitude.deinit();
                    return magnitude.sum(ctx, tag);
                },
                .l2 => {
                    var squared = try self.mul(ctx, self);
                    defer squared.deinit();
                    var sum_sq = try squared.sum(ctx, tag);
                    defer sum_sq.deinit();
                    return sum_sq.sqrt(ctx);
                },
                .inf => {
                    var magnitude = try self.abs(ctx);
                    defer magnitude.deinit();
                    return magnitude.max(ctx, tag);
                },
            }
        }

        /// Scalar vector norm over every element (torch.linalg.vector_norm
        /// with no dim); see `norm` for the order semantics and gradient
        /// caveats. Composed flatten → norm; scope-required under
        /// gradients.
        pub fn normAll(self: *const Self, ctx: *ExecContext, comptime order: exec_mod.NormOrder) !Tensor(.{}) {
            try requireScopeForComposedGrad(ctx, self.requiresGrad());
            var flat = try self.flatten(ctx, tags[0]);
            defer flat.deinit();
            return flat.norm(ctx, tags[0], order);
        }

        /// Cosine similarity vs a same-tagged `other` along `tag` (torch
        /// F.cosine_similarity): Σ x·y / max(‖x‖₂·‖y‖₂, eps) with the `tag`
        /// axis reduced away (torch's eps default is 1e-8). Composed from
        /// existing differentiable ops (mul/sum/sqrt/clamp/div);
        /// differentiable in both operands. When gradients are tracked this
        /// requires an active exec scope (see `nllLoss`); errors with
        /// `ActiveExecScopeRequired` otherwise.
        pub fn cosineSimilarity(
            self: *const Self,
            ctx: *ExecContext,
            other: *const Self,
            comptime tag: Tag,
            eps: f32,
        ) !Tensor(removeTag(tags, tag)) {
            try requireScopeForComposedGrad(ctx, self.requiresGrad() or other.requiresGrad());
            var pointwise_prod = try self.mul(ctx, other);
            defer pointwise_prod.deinit();
            var dot_sum = try pointwise_prod.sum(ctx, tag);
            defer dot_sum.deinit();
            var self_sq = try self.mul(ctx, self);
            defer self_sq.deinit();
            var self_sum_sq = try self_sq.sum(ctx, tag);
            defer self_sum_sq.deinit();
            var self_norm = try self_sum_sq.sqrt(ctx);
            defer self_norm.deinit();
            var other_sq = try other.mul(ctx, other);
            defer other_sq.deinit();
            var other_sum_sq = try other_sq.sum(ctx, tag);
            defer other_sum_sq.deinit();
            var other_norm = try other_sum_sq.sqrt(ctx);
            defer other_norm.deinit();
            var denom = try self_norm.mul(ctx, &other_norm);
            defer denom.deinit();
            var clamped = try denom.clamp(ctx, eps, std.math.floatMax(f32));
            defer clamped.deinit();
            return dot_sum.div(ctx, &clamped);
        }

        /// Rotary position embedding over (`position_tag`, `feature_tag`).
        /// `source` selects the factor source at comptime (a closed set):
        ///
        ///   - `*const exec.RopeTable` (or `*RopeTable`) — prepared factors
        ///     (freq_factors/NTK scaling live there; the production path).
        ///     The table's `feature_dim` is the AUTHORITATIVE rotary span:
        ///     equal to `dim(feature_tag)` rotates fully; SMALLER rotates the
        ///     leading `feature_dim` dims and passes the tail through
        ///     unchanged (partial NEOX RoPE). Validate the span at the call
        ///     site if a mismatched table would be a bug (qwen35's
        ///     `partialRope` is the precedent).
        ///   - `exec.RopeTheta` (or `.{ .positions = p, .theta_base = t }`)
        ///     — on-the-fly factors, full rotation only.
        ///
        /// `mode` (.half | .interleaved) is comptime: the backward record
        /// types are parameterized on it. Differentiable in `self` (the
        /// backward applies the inverse rotation).
        pub fn rope(
            self: *const Self,
            ctx: *ExecContext,
            comptime position_tag: Tag,
            comptime feature_tag: Tag,
            source: anytype,
            comptime mode: RopeMode,
        ) !Self {
            const position_axis = comptime axis(position_tag);
            const feature_axis = comptime axis(feature_tag);
            const SourceT = @TypeOf(source);
            const info = @typeInfo(SourceT);
            if (comptime (info == .pointer and info.pointer.size == .one and info.pointer.child == exec_mod.RopeTable)) {
                // The partial exec entry self-falls-back to the full kernel
                // when table.feature_dim equals the feature axis length (one
                // integer compare); the backward mirrors it.
                var value = try ctx.ropePartialAxisRankWithTable(tag_rank, self.asRawTensor(), position_axis, feature_axis, source, mode);
                errdefer value.deinit();
                return finishOp(tags, ctx, value, self.requiresGrad(), RopeTableBackward(tags, position_axis, feature_axis, mode), .{ ctx.allocator, self.grad_state, source });
            }
            if (comptime info == .@"struct") {
                comptime {
                    for (info.@"struct".fields) |field| {
                        if (!std.mem.eql(u8, field.name, "positions") and !std.mem.eql(u8, field.name, "theta_base"))
                            @compileError("rope: unknown RopeTheta field ." ++ field.name);
                    }
                    if (!@hasField(SourceT, "positions") or !@hasField(SourceT, "theta_base"))
                        @compileError("rope: an on-the-fly source needs both .positions and .theta_base");
                }
                const theta = exec_mod.RopeTheta{ .positions = source.positions, .theta_base = source.theta_base };
                var value = try ctx.ropeAxisRank(tag_rank, self.asRawTensor(), position_axis, feature_axis, theta.positions, theta.theta_base, mode, false);
                errdefer value.deinit();
                return finishOp(tags, ctx, value, self.requiresGrad(), RopeBackward(tags, position_axis, feature_axis, mode), .{ ctx.allocator, self.grad_state, theta.positions, theta.theta_base });
            }
            @compileError("rope: source must be a *const exec.RopeTable or an exec.RopeTheta (.{ .positions, .theta_base }); got " ++ @typeName(SourceT));
        }

        pub fn dot(self: *const Self, ctx: *ExecContext, other: anytype, comptime contract_tag: Tag) !Tensor(dotResultTags(tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag)) {
            const Other = TensorObject(@TypeOf(other));
            const other_tags = Other.axis_tags;
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = dotResultTags(tags, other_tags, contract_tag);
            if (comptime dtype_mod.isBlockQuantized(Other.dtype)) {
                const allow_gpu = !self.requiresGrad() and control.isQuantDotGpuEnabled();
                var value = try quantizedRhsDotRaw(Other.dtype, tags, self.asRawTensor(), ctx, other_tags, other_ptr.asRawTensor(), contract_tag, allow_gpu);
                errdefer value.deinit();
                return finishOp(result_tags, ctx, value, self.requiresGrad(), ConstRhsDotBackward(Other.dtype, tags, other_tags, contract_tag), .{ ctx.allocator, self.grad_state, null, self.asRawTensor(), other_ptr.asRawTensor() });
            }
            if (comptime Other.dtype == .f16) {
                var value = try f16RhsDotRaw(tags, self.asRawTensor(), ctx, other_tags, other_ptr.asRawTensor(), contract_tag);
                errdefer value.deinit();
                return finishOp(result_tags, ctx, value, self.requiresGrad() or other_ptr.requiresGrad(), ConstRhsDotBackward(.f16, tags, other_tags, contract_tag), .{ ctx.allocator, self.grad_state, other_ptr.grad_state, self.asRawTensor(), other_ptr.asRawTensor() });
            }
            if (comptime Other.dtype == .bf16) {
                var value = try bf16RhsDotRaw(tags, self.asRawTensor(), ctx, other_tags, other_ptr.asRawTensor(), contract_tag);
                errdefer value.deinit();
                return finishOp(result_tags, ctx, value, self.requiresGrad() or other_ptr.requiresGrad(), ConstRhsDotBackward(.bf16, tags, other_tags, contract_tag), .{ ctx.allocator, self.grad_state, other_ptr.grad_state, self.asRawTensor(), other_ptr.asRawTensor() });
            }
            var value = try tag_ops.taggedDot(tags, self.asRawTensor(), ctx, other_tags, other_ptr.asRawTensor(), contract_tag);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad() or other_ptr.requiresGrad(), DotBackward(tags, other_tags, contract_tag), .{ ctx.allocator, self.grad_state, other_ptr.grad_state, self.asRawTensor(), other_ptr.asRawTensor() });
        }

        /// Multi-index tagged contraction (einsum). `out_tags` is the whole
        /// equation: `result[out_tags] = Σ over every tag not in out_tags of
        /// self ⊙ other`. Shared tags are batch axes when kept and contraction
        /// axes when dropped; operand-private tags are free axes when kept and
        /// summed away when dropped. Result axis order is exactly `out_tags`;
        /// every output tag must exist in an operand (compile error) and every
        /// shared dim must match (`ShapeMismatch`). Generalizes `dot` to any
        /// number of contraction, batch, and free axes in one differentiable
        /// operation that lowers onto the same matmul/bmm kernels; both
        /// operand gradients are einsums themselves (no pointwise fallback).
        /// An f16/bf16 `other` is widened to f32 once per call (forward and
        /// backward); a constant 16-bit RHS routes gradient to `self` only,
        /// while a grad-requiring 16-bit RHS variable also receives its own
        /// f32 gradient. Quantized RHS stays dot-only.
        pub fn einsum(self: *const Self, ctx: *ExecContext, other: anytype, comptime out_tags: anytype) !Tensor(normalizeTags(out_tags)) {
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (dtype_mod.isBlockQuantized(Other.dtype))
                    @compileError("einsum does not take a quantized RHS; use dot, whose packed kernels require the [free, contract] weight layout");
                if (Other.dtype != .f32 and Other.dtype != .f16 and Other.dtype != .bf16)
                    @compileError("einsum requires f32 operands (f16/bf16 RHS runs as a widened constant)");
            }
            const other_tags = Other.axis_tags;
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = comptime normalizeTags(out_tags);
            if (comptime (Other.dtype == .f16 or Other.dtype == .bf16)) {
                // Mixed-precision RHS: widen once per call and run the f32
                // lowering. A constant RHS routes gradient to `self` only; a
                // grad-requiring 16-bit RHS variable also receives its f32
                // gradient (gradients are always f32).
                var right_f32 = try ctx.castTyped(Other.dtype, .f32, other_ptr.asRawTensor());
                defer right_f32.deinit();
                var value = try tag_ops.taggedEinsum(tags, self.asRawTensor(), ctx, other_tags, &right_f32, result_tags);
                errdefer value.deinit();
                return finishOp(result_tags, ctx, value, self.requiresGrad() or other_ptr.requiresGrad(), ConstRhsEinsumBackward(Other.dtype, tags, other_tags, result_tags), .{ ctx.allocator, self.grad_state, other_ptr.grad_state, self.asRawTensor(), other_ptr.asRawTensor() });
            }
            var value = try tag_ops.taggedEinsum(tags, self.asRawTensor(), ctx, other_tags, other_ptr.asRawTensor(), result_tags);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad() or other_ptr.requiresGrad(), EinsumBackward(tags, other_tags, result_tags), .{ ctx.allocator, self.grad_state, other_ptr.grad_state, self.asRawTensor(), other_ptr.asRawTensor() });
        }

        /// Trainable ternary linear (BitNet b1.58 straight-through estimator):
        /// every forward encodes the f32 latent `weight` (tags `{.out, .in}`,
        /// per-tensor absmean scale, round-clip to {-1, 0, +1}) to TQ2_0 and
        /// contracts `self` (`[..., in]`) against it with the mul-free f32
        /// kernel. Backward: dx flows through the QUANTIZED weight
        /// (dequantize-then-matmul); dW is the straight-through estimate —
        /// the plain matmul VJP as if the forward had been `x @ Wᵀ` with the
        /// latent weight (identity through the quantizer, no clip/mask).
        /// The contract dim must be a multiple of 256 (the TQ2_0 block size);
        /// anything else fails with `error.TernaryContractDimNotBlockAligned`.
        pub fn dotTernarySte(self: *const Self, ctx: *ExecContext, weight: anytype, comptime contract_tag: Tag) !Tensor(dotResultTags(tags, TensorObject(@TypeOf(weight)).axis_tags, contract_tag)) {
            const Weight = TensorObject(@TypeOf(weight));
            const weight_tags = Weight.axis_tags;
            const result_tags = dotResultTags(tags, weight_tags, contract_tag);
            comptime {
                if (Weight.dtype != .f32) @compileError("dotTernarySte requires an f32 latent weight");
                if (dotBatchLen(tags, weight_tags, contract_tag) != 0) @compileError("dotTernarySte does not support shared batch tags");
                if (dotRightFreeLen(tags, weight_tags, contract_tag) != 1) @compileError("dotTernarySte requires one weight free axis");
                if (!tagsEqual(weight_tags, dotRightTransBOrder(tags, weight_tags, contract_tag)))
                    @compileError("dotTernarySte requires weight storage order [free, contract], e.g. weight tags {.out, .in}");
                if (tagIndexOrCompileError(tags, contract_tag) != tags.len - 1)
                    @compileError("dotTernarySte requires lhs storage order [..., contract]");
            }
            const weight_ptr = tensorObjectPtrFrom(@TypeOf(weight), &weight);
            const weight_raw = weight_ptr.asRawTensor();

            const result_shape = try dotResultShapeOf(.f32, .f32, tags, self.asRawTensor(), weight_tags, weight_raw, contract_tag);
            const n = weight_raw.shape.at(0);
            const k = weight_raw.shape.at(1);
            if (self.asRawTensor().shape.at(tag_rank - 1) != k) return TensorError.ShapeMismatch;
            if (k == 0 or k % dtype_mod.qk_k_block_size != 0) return error.TernaryContractDimNotBlockAligned;

            const left_free_rank = comptime dotLeftFreeLen(tags, weight_tags, contract_tag);
            var left_aligned = try alignTensorToOf(.f32, tags, self.asRawTensor(), dotLeftOrder(tags, weight_tags, contract_tag));
            defer left_aligned.deinit();
            const m = productRangeOf(.f32, &left_aligned, 0, left_free_rank);
            var left_ready = try contiguousForReshapeOf(.f32, ctx, &left_aligned);
            defer left_ready.deinit();
            var left_matrix = try left_ready.reshape(&.{ m, k });
            defer left_matrix.deinit();

            var weight_ready = try contiguousForReshapeOf(.f32, ctx, weight_raw);
            defer weight_ready.deinit();

            // Per-tensor absmean scale + round-clip encode of the latent weight.
            var rhs = try backend_mod.quantized_matmul.quantizedMatmulRhsTQ2_0FromF32Absmean(ctx.allocator, k, n, weight_ready.dataConst());
            var rhs_owned = true;
            errdefer if (rhs_owned) rhs.deinit();

            var value = forward: {
                var product = try ctx.emptyRankTyped(.f32, 2, .{ m, n });
                errdefer product.deinit();
                const work = parallel.saturatedMul3(m, n, k);
                const config: backend_mod.vector_impl.ParallelConfig =
                    if (work >= parallel.vector_matmul_work_threshold) .{ .pool = ctx.workPool() } else .{};
                // Deliberately the vector kernel on BOTH backend kinds
                // (including -Dbackend=scalar): the mul-free f32 kernel is
                // pure fixed-order @Vector bitwise ops + adds, bitwise-
                // identical on every target by construction, so the scalar
                // leg exercises the same numerics (unlike the quant matmuls,
                // which keep a scalar reference in cpu.zig).
                backend_mod.vector_impl.matmul2DTQ2_0F32RhsIntoWithConfig(product.data(), left_matrix.dataConst(), &rhs, m, n, k, config);
                if (std.mem.eql(usize, product.shape.slice(), result_shape[0..])) break :forward product;
                const reshaped = try product.reshape(result_shape[0..]);
                product.deinit();
                break :forward reshaped;
            };
            errdefer value.deinit();

            // Inlined finishOp tail: the encoded rhs is a non-refcounted
            // resource, so ownership must transfer to the Backward exactly at
            // its node creation (finishOp's opaque create_args cannot express
            // that hand-off).
            const wants_grad = (self.requiresGrad() or weight_ptr.requiresGrad()) and control.isGradEnabled();
            if (!wants_grad) {
                rhs_owned = false;
                rhs.deinit();
                return finishNoGrad(result_tags, ctx, value);
            }
            if (ctx.execScopeActive()) try ctx.reserveScopeSlot();
            const state = try core.createNode(TernarySteDotBackward(tags), .{ ctx.allocator, self.grad_state, weight_ptr.grad_state, self.asRawTensor(), &left_matrix, rhs });
            rhs_owned = false; // owned by the node from here on
            var out = try finishWithBackward(result_tags, value, state);
            if (ctx.execScopeActive()) {
                adoptIntoScope(ctx, &out);
                out.scope_owned = true;
            }
            return out;
        }

        /// Packed quantized-RHS matmul: `out[free, out_tag] = self[free, contract_tag] · rhsᵀ`.
        /// `rhs` points to one of the packed RHS containers produced by
        /// `packRhs`/`packRhsLayout` (q8_0x4 / q6_kx4 / q4_kx8 / q4_kx2mmla /
        /// q5_kx8); the layout is comptime-dispatched from the pointer type, so
        /// each call site compiles to the exact per-layout kernel call.
        /// No gradient support: fails with `error.GradientQuantizedMatmulUnsupported`
        /// when `self` requires grad (grad state is runtime graph state).
        pub fn dotPacked(
            self: *const Self,
            ctx: *ExecContext,
            rhs: anytype,
            comptime contract_tag: Tag,
            comptime out_tag: Tag,
        ) !Tensor(replaceTag(tags, contract_tag, out_tag)) {
            const layout = comptime packedRhsLayout(@TypeOf(rhs), "dotPacked");
            comptime {
                if (tag_rank != 2) @compileError("dotPacked (." ++ @tagName(layout) ++ " RHS) currently requires a rank-2 lhs");
                if (axis(contract_tag) != 1) @compileError("dotPacked (." ++ @tagName(layout) ++ " RHS) requires lhs storage order [free, contract]");
            }
            if (self.requiresGrad()) return error.GradientQuantizedMatmulUnsupported;
            var value = try switch (layout) {
                .q8_0x4 => ctx.matmul2DWithPackedQ8_0x4Rhs(self.asRawTensor(), rhs),
                .q6_kx4 => ctx.matmul2DWithPackedQ6_Kx4Rhs(self.asRawTensor(), rhs),
                .q4_kx8 => ctx.matmul2DWithPackedQ4_Kx8Rhs(self.asRawTensor(), rhs),
                .q4_kx2mmla => ctx.matmul2DWithPackedQ4_Kx2MmlaRhs(self.asRawTensor(), rhs),
                .q5_kx8 => ctx.matmul2DWithPackedQ5_Kx8Rhs(self.asRawTensor(), rhs),
                .q4_kx4 => @compileError("dotPacked: the Q4_Kx4 layout has no facade entry (kernel-comparison surface below the facade); pack q4_k with packRhs (x2mmla/x8) instead"),
            };
            errdefer value.deinit();
            return finishNoGrad(replaceTag(tags, contract_tag, out_tag), ctx, value);
        }

        /// Fused rmsNormMul + packed GEMM: computes
        /// `rmsNormMul(self, norm_weight) · rhsᵀ` without materializing the
        /// normalized tensor — the kernel normalizes up to 4 rows into
        /// task-private scratch with the exact `rmsNormMulRows` kernel and
        /// quantizes with the fused packers — results match rmsNormMul +
        /// dotPacked to f32 roundoff (<= 1 ulp observed; the packed matmul's
        /// internal LHS quantizer arrangement may differ in the last ulp,
        /// the splitSwiGluDotPacked precedent).
        /// `self` is the PRE-norm rank-2 [free, contract] input;
        /// `norm_weight` is the [contract] scale row. Fused kernels exist
        /// for q8_0x4 / q4_kx8 / q5_kx8 / q6_kx4 only; q4_kx2mmla falls
        /// through at the caller like splitSwiGluDotPacked. No gradient
        /// support (same policy as `dotPacked`).
        pub fn rmsNormMulDotPacked(
            self: *const Self,
            ctx: *ExecContext,
            norm_weight: anytype,
            eps: f32,
            rhs: anytype,
            comptime contract_tag: Tag,
            comptime out_tag: Tag,
        ) !Tensor(replaceTag(tags, contract_tag, out_tag)) {
            const layout = comptime packedRhsLayout(@TypeOf(rhs), "rmsNormMulDotPacked");
            comptime {
                if (tag_rank != 2) @compileError("rmsNormMulDotPacked (." ++ @tagName(layout) ++ " RHS) currently requires a rank-2 lhs");
                if (axis(contract_tag) != 1) @compileError("rmsNormMulDotPacked (." ++ @tagName(layout) ++ " RHS) requires lhs storage order [free, contract]");
            }
            const weight_ptr = tensorObjectPtrFrom(@TypeOf(norm_weight), &norm_weight);
            if (self.requiresGrad() or weight_ptr.requiresGrad()) return error.GradientQuantizedMatmulUnsupported;
            var value = try switch (layout) {
                .q8_0x4 => ctx.rmsNormMulMatmul2DWithPackedQ8_0x4Rhs(self.asRawTensor(), weight_ptr.asRawTensor(), eps, rhs),
                .q4_kx8 => ctx.rmsNormMulMatmul2DWithPackedQ4_Kx8Rhs(self.asRawTensor(), weight_ptr.asRawTensor(), eps, rhs),
                .q5_kx8 => ctx.rmsNormMulMatmul2DWithPackedQ5_Kx8Rhs(self.asRawTensor(), weight_ptr.asRawTensor(), eps, rhs),
                .q6_kx4 => ctx.rmsNormMulMatmul2DWithPackedQ6_Kx4Rhs(self.asRawTensor(), weight_ptr.asRawTensor(), eps, rhs),
                .q4_kx2mmla => @compileError("rmsNormMulDotPacked: no fused MMLA kernel exists; use the unfused path (rmsNormMul + dotPacked)"),
                .q4_kx4 => @compileError("rmsNormMulDotPacked: the Q4_Kx4 layout has no facade entry (kernel-comparison surface below the facade)"),
            };
            errdefer value.deinit();
            return finishNoGrad(replaceTag(tags, contract_tag, out_tag), ctx, value);
        }

        /// Fused split-SwiGLU + packed down GEMM: `self` is a fused rank-2
        /// `[free, split_tag]` gate_up activation; computes
        /// `swiglu(split(self)) · rhsᵀ` without materializing the gated tensor.
        /// Fused kernels exist for q8_0x4 / q4_kx8 / q5_kx8 / q6_kx4 only;
        /// q4_kx2mmla is a deliberate compile error — callers keep their
        /// `comptime !supports_q4_k_mmla` guard so MMLA targets fall through
        /// to the unfused path (splitSwiGlu + dotPacked).
        /// No gradient support (same policy as `dotPacked`).
        pub fn splitSwiGluDotPacked(
            self: *const Self,
            ctx: *ExecContext,
            rhs: anytype,
            comptime split_tag: Tag,
            comptime out_tag: Tag,
        ) !Tensor(replaceTag(tags, split_tag, out_tag)) {
            const layout = comptime packedRhsLayout(@TypeOf(rhs), "splitSwiGluDotPacked");
            comptime {
                if (tag_rank != 2) @compileError("splitSwiGluDotPacked (." ++ @tagName(layout) ++ " RHS) currently requires a rank-2 lhs");
                if (axis(split_tag) != 1) @compileError("splitSwiGluDotPacked (." ++ @tagName(layout) ++ " RHS) requires lhs storage order [free, fused]");
            }
            if (self.requiresGrad()) return error.GradientQuantizedMatmulUnsupported;
            var value = try switch (layout) {
                .q8_0x4 => ctx.splitSwiGluMatmul2DWithPackedQ8_0x4Rhs(self.asRawTensor(), rhs),
                .q4_kx8 => ctx.splitSwiGluMatmul2DWithPackedQ4_Kx8Rhs(self.asRawTensor(), rhs),
                .q5_kx8 => ctx.splitSwiGluMatmul2DWithPackedQ5_Kx8Rhs(self.asRawTensor(), rhs),
                .q6_kx4 => ctx.splitSwiGluMatmul2DWithPackedQ6_Kx4Rhs(self.asRawTensor(), rhs),
                .q4_kx2mmla => @compileError("splitSwiGluDotPacked: no fused MMLA kernel exists; on aarch64+i8mm targets use the unfused path (splitSwiGlu + dotPacked)"),
                .q4_kx4 => @compileError("splitSwiGluDotPacked: the Q4_Kx4 layout has no facade entry (kernel-comparison surface below the facade)"),
            };
            errdefer value.deinit();
            return finishNoGrad(replaceTag(tags, split_tag, out_tag), ctx, value);
        }

        /// Fused GeGLU + down projection for separate gate/up activations:
        /// `self` is the gate, `up` the up projection (same tags); the result is
        /// `(up * geluQuant(gate)) @ rhs` without materializing the gated tensor.
        /// Only the q8_0x4 packed layout has a fused geglu kernel today.
        /// No gradient support (same policy as `dotPacked`, checked on both
        /// `self` and `up`).
        pub fn gegluQuantDotPacked(
            self: *const Self,
            ctx: *ExecContext,
            up: *const Self,
            rhs: anytype,
            comptime in_tag: Tag,
            comptime out_tag: Tag,
        ) !Tensor(replaceTag(tags, in_tag, out_tag)) {
            const layout = comptime packedRhsLayout(@TypeOf(rhs), "gegluQuantDotPacked");
            comptime {
                if (tag_rank != 2) @compileError("gegluQuantDotPacked (." ++ @tagName(layout) ++ " RHS) currently requires a rank-2 lhs");
                if (axis(in_tag) != 1) @compileError("gegluQuantDotPacked (." ++ @tagName(layout) ++ " RHS) requires lhs storage order [free, contract]");
            }
            if (self.requiresGrad() or up.requiresGrad()) return error.GradientQuantizedMatmulUnsupported;
            var value = try switch (layout) {
                .q8_0x4 => ctx.gegluQuantMatmul2DWithPackedQ8_0x4Rhs(self.asRawTensor(), up.asRawTensor(), rhs),
                .q6_kx4, .q4_kx4, .q4_kx8, .q4_kx2mmla, .q5_kx8 => @compileError("gegluQuantDotPacked: no fused geglu kernel for packed RHS layout ." ++ @tagName(layout)),
            };
            errdefer value.deinit();
            return finishNoGrad(replaceTag(tags, in_tag, out_tag), ctx, value);
        }

        /// Grouped (GQA) attention over every KV representation, `q` = `self`
        /// with tags `.{ .seq, .head, .d }`. The KV representation is
        /// comptime-dispatched from `@TypeOf(k)`:
        ///
        ///   - `*Tensor(.{ .seq, .kv_head, .d })` (f32)        → f32 kernel
        ///   - f16 Tensor with the same tags                   → f16 kernel (decode KV cache)
        ///   - `[]const BlockQ8_0`                             → q8_0 raw-block cache,
        ///     layout `[kv_seq, kv_heads, d/32]` (see `ExecContext.groupedCausalAttentionQ8Kv`)
        ///   - `[]const []const f16`                           → ragged multi-stream decode
        ///   - `[]const []const BlockQ8_0`                     → ragged multi-stream decode (q8_0)
        ///
        /// `opts` is a comptime-validated struct literal; unknown fields are
        /// compile errors:
        ///
        ///   - `.mask = .causal` (default) | `.bidirectional` — f32/f16 KV only
        ///     (bidirectional = every query attends every key: the
        ///     block-diffusion canvas / OmniVoice encoder attention; its SWA
        ///     reach is realized by narrowing the K/V views — no
        ///     bidirectional window exists by design).
        ///   - `.window = w` — runtime sliding window, 0 = full causal
        ///     (query p attends [max(0, p-w+1), p]); causal only.
        ///   - `.bias = &b` — rank-2 `[q_seq, kv_seq]` additive f32 bias added
        ///     to the scaled scores pre-softmax (ggml_soft_max_ext semantics,
        ///     NOT -inf masking); bidirectional + f32 KV only.
        ///   - `.kv_seq = n, .kv_heads = h` — REQUIRED for the q8_0-block
        ///     repr (raw blocks carry no shape).
        ///   - `.lens = lens, .kv_heads = h` — REQUIRED for the multi-stream
        ///     reprs. There, q's `.seq` tag is reinterpreted as the STREAM
        ///     axis: exactly one query row per stream, row `s` attending all
        ///     `lens[s]` cached positions of `k[s]`/`v[s]`; per-stream
        ///     results are bit-identical to N single-stream f16/q8 calls.
        ///
        /// Gradient matrix (unchanged from the former per-variant entries;
        /// no longer readable off the name):
        ///
        ///   - f32 KV: full q/k/v backward (windowed re-masks to the window).
        ///   - f16 KV: q-grad only — K/V are cache constants, widened to f32
        ///     once and run through the f32 kernel + backward.
        ///   - q8_0 KV: q-grad only, causal only — the cache is dequantized
        ///     to f32 once (no bidirectional q8 path exists).
        ///   - `.bias` present: inference-only; ANY grad-requiring operand
        ///     (q, k, v, or the bias) returns `error.UnsupportedGradient` —
        ///     the shared backward re-derives the softmax without a bias.
        ///   - multi-stream: inference-only (`error.UnsupportedGradient`).
        pub fn groupedAttention(
            self: *const Self,
            ctx: *ExecContext,
            k: anytype,
            v: anytype,
            kv_head_for_head: []const usize,
            comptime out_tag: Tag,
            scale_value: f32,
            opts: anytype,
        ) !Tensor(.{ .seq, out_tag }) {
            comptime if (!tagsEqual(tags, .{ .seq, .head, .d })) {
                @compileError("groupedAttention requires q tags .{ .seq, .head, .d }");
            };
            const repr = comptime attentionKvRepr(@TypeOf(k), "k");
            comptime {
                const v_repr = attentionKvRepr(@TypeOf(v), "v");
                if (repr != v_repr) @compileError("groupedAttention: k is a ." ++ @tagName(repr) ++ " KV but v is a ." ++ @tagName(v_repr) ++ " KV");
            }
            const O = @TypeOf(opts);
            comptime if (@typeInfo(O) != .@"struct") {
                @compileError("groupedAttention: opts must be a struct literal, e.g. .{} or .{ .window = w }");
            };
            const mask: AttentionMask = comptime if (@hasField(O, "mask")) opts.mask else .causal;
            comptime {
                // Field whitelist per KV repr: a misspelled option (`.windw`)
                // must be a compile error, never silently-full-causal attention.
                const allowed: []const []const u8 = switch (repr) {
                    .f32_kv => &.{ "mask", "window", "bias" },
                    .f16_kv => &.{ "mask", "window" },
                    .q8_kv => &.{ "window", "kv_seq", "kv_heads" },
                    .multi_f16_kv, .multi_q8_kv => &.{ "lens", "kv_heads" },
                };
                for (@typeInfo(O).@"struct".fields) |field| {
                    var known = false;
                    for (allowed) |name| {
                        if (std.mem.eql(u8, field.name, name)) known = true;
                    }
                    if (!known) @compileError("groupedAttention: unknown option ." ++ field.name ++ " for the ." ++ @tagName(repr) ++ " KV representation");
                }
                if (mask == .bidirectional and @hasField(O, "window")) {
                    @compileError("groupedAttention: no bidirectional windowed kernel exists — realize SWA reach by narrowing the K/V views");
                }
                if (@hasField(O, "bias") and mask != .bidirectional) {
                    @compileError("groupedAttention: .bias requires .mask = .bidirectional (the only additive-bias kernel)");
                }
                switch (repr) {
                    .q8_kv => if (!@hasField(O, "kv_seq") or !@hasField(O, "kv_heads")) {
                        @compileError("groupedAttention: the q8_0-block KV repr requires .kv_seq and .kv_heads (raw blocks carry no shape; layout [kv_seq, kv_heads, d/32])");
                    },
                    .multi_f16_kv, .multi_q8_kv => if (!@hasField(O, "lens") or !@hasField(O, "kv_heads")) {
                        @compileError("groupedAttention: the multi-stream KV reprs require .lens and .kv_heads (q's .seq tag is the stream axis)");
                    },
                    else => {},
                }
            }
            switch (comptime repr) {
                .f32_kv => {
                    if (comptime @hasField(O, "bias")) {
                        const bias = opts.bias;
                        const bias_ptr = tensorObjectPtrFrom(@TypeOf(bias), &bias);
                        comptime if (TensorObject(@TypeOf(bias)).axis_tags.len != 2) {
                            @compileError("groupedAttention: .bias must be a rank-2 [q_seq, kv_seq] tensor");
                        };
                        if (self.requiresGrad() or k.requiresGrad() or v.requiresGrad() or bias_ptr.requiresGrad()) {
                            return error.UnsupportedGradient;
                        }
                        var value = try ctx.groupedBidirectionalAttentionBiased(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value, bias_ptr.asRawTensor());
                        errdefer value.deinit();
                        return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                    }
                    const window: usize = if (comptime @hasField(O, "window")) opts.window else 0;
                    const wants_grad = self.requiresGrad() or k.requiresGrad() or v.requiresGrad();
                    // Forward-saved softmax {max, sum_exp} for the VJP's
                    // one-pass probability rebuild; the stats entry's output
                    // is bitwise identical to the stats-less ones.
                    const row_stats = try rowStatsAlloc(ctx, wants_grad, kv_head_for_head.len * self.asRawTensor().shape.at(0));
                    defer if (row_stats) |stats| ctx.allocator.free(stats);
                    var value = try if (row_stats) |stats|
                        ctx.groupedCausalAttentionStatsOut(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value, window, mask == .causal, stats)
                    else switch (comptime mask) {
                        .causal => if (comptime @hasField(O, "window"))
                            ctx.groupedCausalAttentionWindowed(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value, opts.window)
                        else
                            ctx.groupedCausalAttention(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value),
                        .bidirectional => ctx.groupedBidirectionalAttention(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value),
                    };
                    errdefer value.deinit();
                    return finishOp(.{ .seq, out_tag }, ctx, value, wants_grad, GroupedCausalAttentionBackward, .{
                        ctx.allocator,
                        self.grad_state,
                        k.grad_state,
                        v.grad_state,
                        self.asRawTensor(),
                        k.asRawTensor(),
                        v.asRawTensor(),
                        kv_head_for_head,
                        scale_value,
                        window,
                        mask == .causal,
                        row_stats orelse &[_]f32{},
                        &value,
                    });
                },
                .f16_kv => {
                    const window: usize = if (comptime @hasField(O, "window")) opts.window else 0;
                    if (self.requiresGrad()) {
                        return self.f16KvAttentionWithGrad(ctx, k, v, kv_head_for_head, out_tag, scale_value, window, mask == .causal);
                    }
                    var value = try switch (comptime mask) {
                        .causal => if (comptime @hasField(O, "window"))
                            ctx.groupedCausalAttentionF16KvWindowed(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value, opts.window)
                        else
                            ctx.groupedCausalAttentionF16Kv(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value),
                        .bidirectional => ctx.groupedBidirectionalAttentionF16Kv(self.asRawTensor(), k.asRawTensor(), v.asRawTensor(), kv_head_for_head, scale_value),
                    };
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
                .q8_kv => {
                    if (self.requiresGrad()) {
                        const window: usize = if (comptime @hasField(O, "window")) opts.window else 0;
                        return self.q8KvAttentionWithGrad(ctx, k, v, opts.kv_seq, opts.kv_heads, kv_head_for_head, out_tag, scale_value, window);
                    }
                    var value = if (comptime @hasField(O, "window"))
                        try ctx.groupedCausalAttentionQ8KvWindowed(self.asRawTensor(), k, v, opts.kv_seq, opts.kv_heads, kv_head_for_head, scale_value, opts.window)
                    else
                        try ctx.groupedCausalAttentionQ8Kv(self.asRawTensor(), k, v, opts.kv_seq, opts.kv_heads, kv_head_for_head, scale_value);
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
                .multi_f16_kv => {
                    if (self.requiresGrad()) return error.UnsupportedGradient;
                    var value = try ctx.groupedCausalAttentionMultiF16Kv(self.asRawTensor(), k, v, opts.lens, opts.kv_heads, kv_head_for_head, scale_value);
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
                .multi_q8_kv => {
                    if (self.requiresGrad()) return error.UnsupportedGradient;
                    var value = try ctx.groupedCausalAttentionMultiQ8Kv(self.asRawTensor(), k, v, opts.lens, opts.kv_heads, kv_head_for_head, scale_value);
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
            }
        }

        /// Gradient path for q8_0-KV attention: dequantize the constant
        /// cache to f32 once, then the f32 kernel + backward (q-grad only),
        /// mirroring `f16KvAttentionWithGrad`.
        fn q8KvAttentionWithGrad(
            self: *const Self,
            ctx: *ExecContext,
            k_blocks: []const BlockQ8_0,
            v_blocks: []const BlockQ8_0,
            kv_seq: usize,
            kv_heads: usize,
            kv_head_for_head: []const usize,
            comptime out_tag: Tag,
            scale_value: f32,
            window: usize,
        ) !Tensor(.{ .seq, out_tag }) {
            const block_size = dtype_mod.q8_0_block_size;
            if (kv_seq * kv_heads == 0 or k_blocks.len % (kv_seq * kv_heads) != 0) return TensorError.InvalidShape;
            if (v_blocks.len != k_blocks.len) return TensorError.InvalidShape;
            const d = (k_blocks.len / (kv_seq * kv_heads)) * block_size;
            var k32 = try ctx.emptyRank(3, .{ kv_seq, kv_heads, d });
            defer k32.deinit();
            try ctx.dequantizeQ8_0RowsInto(k32.data(), k_blocks);
            var v32 = try ctx.emptyRank(3, .{ kv_seq, kv_heads, d });
            defer v32.deinit();
            try ctx.dequantizeQ8_0RowsInto(v32.data(), v_blocks);
            const row_stats = try rowStatsAlloc(ctx, true, kv_head_for_head.len * self.asRawTensor().shape.at(0));
            defer if (row_stats) |stats| ctx.allocator.free(stats);
            var value = if (row_stats) |stats|
                try ctx.groupedCausalAttentionStatsOut(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value, window, true, stats)
            else if (window == 0)
                try ctx.groupedCausalAttention(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value)
            else
                try ctx.groupedCausalAttentionWindowed(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value, window);
            errdefer value.deinit();
            return finishOp(.{ .seq, out_tag }, ctx, value, true, GroupedCausalAttentionBackward, .{
                ctx.allocator,
                self.grad_state,
                null,
                null,
                self.asRawTensor(),
                &k32,
                &v32,
                kv_head_for_head,
                scale_value,
                window,
                true,
                row_stats orelse &[_]f32{},
                &value,
            });
        }
        // (q8KvAttentionWithGrad above stays causal-only: no bidirectional
        // q8_0-cache user exists — the diffusion canvas runs on f16 KV.)

        /// Gradient path for f16-KV attention: K/V are constants (the cache),
        /// so only q-grad flows. Widens K/V to f32 once and runs the f32
        /// kernel + backward; the f16 fast path stays grad-free.
        fn f16KvAttentionWithGrad(
            self: *const Self,
            ctx: *ExecContext,
            k: *const Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } }),
            v: *const Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } }),
            kv_head_for_head: []const usize,
            comptime out_tag: Tag,
            scale_value: f32,
            window: usize,
            causal: bool,
        ) !Tensor(.{ .seq, out_tag }) {
            var k32 = try ctx.castTyped(.f16, .f32, k.asRawTensor());
            defer k32.deinit();
            var v32 = try ctx.castTyped(.f16, .f32, v.asRawTensor());
            defer v32.deinit();
            const row_stats = try rowStatsAlloc(ctx, true, kv_head_for_head.len * self.asRawTensor().shape.at(0));
            defer if (row_stats) |stats| ctx.allocator.free(stats);
            var value = if (row_stats) |stats|
                try ctx.groupedCausalAttentionStatsOut(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value, window, causal, stats)
            else if (!causal)
                try ctx.groupedBidirectionalAttention(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value)
            else if (window == 0)
                try ctx.groupedCausalAttention(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value)
            else
                try ctx.groupedCausalAttentionWindowed(self.asRawTensor(), &k32, &v32, kv_head_for_head, scale_value, window);
            errdefer value.deinit();
            return finishOp(.{ .seq, out_tag }, ctx, value, true, GroupedCausalAttentionBackward, .{
                ctx.allocator,
                self.grad_state,
                null,
                null,
                self.asRawTensor(),
                &k32,
                &v32,
                kv_head_for_head,
                scale_value,
                window,
                causal,
                row_stats orelse &[_]f32{},
                &value,
            });
        }

        fn axisView(self: *const Self, ctx: *ExecContext, comptime axes: anytype, comptime target_tags: anytype) !Tensor(target_tags) {
            var value = try axisViewTensor(self.asRawTensor(), axes, target_tags);
            errdefer value.deinit();
            return finishOp(target_tags, ctx, value, self.requiresGrad(), AxisViewBackward(tags, axes), .{ ctx.allocator, self.grad_state, &self.value });
        }

        fn unaryDifferentiable(self: *const Self, ctx: *ExecContext, comptime op: UnaryOp) !Self {
            var value = try ctx.unary(op, self.asRawTensor());
            errdefer value.deinit();
            // Output-derivative ops (see backward.unaryUsesOutput) store the
            // OUTPUT view: their VJP is transcendental-free in t (tanh' = 1-t²)
            // and exact for the value the SIMD forward actually produced.
            const saved: *const RawTensor = if (comptime unaryUsesOutput(op)) &value else &self.value;
            return finishOp(tags, ctx, value, self.requiresGrad(), UnaryBackward(op, tags), .{ ctx.allocator, self.grad_state, saved });
        }
    };
}

/// ISA-best packed matmul RHS container type for a block-quantized dtype:
/// q8_0→x4, q6_k→x4, q5_k→x8, q4_k→x2mmla on aarch64+i8mm targets else x8.
/// This is the return type of `packRhs`; model code stores packed weights as
/// `fucina.PackedRhs(dtype)` fields.
pub fn PackedRhs(comptime dt: DType) type {
    return switch (dt) {
        .q8_0 => backend_mod.QuantizedMatmulRhsQ8_0x4,
        .q6_k => backend_mod.QuantizedMatmulRhsQ6_Kx4,
        .q5_k => backend_mod.QuantizedMatmulRhsQ5_Kx8,
        .q4_k => if (backend_mod.supports_q4_k_mmla)
            backend_mod.QuantizedMatmulRhsQ4_Kx2Mmla
        else
            backend_mod.QuantizedMatmulRhsQ4_Kx8,
        else => @compileError("PackedRhs: no packed matmul RHS layout for dtype ." ++ @tagName(dt)),
    };
}

/// Comptime tag guard for rank-1 norm weight/bias operands: tagged tensors
/// must carry the normalized axis tag; numeric-tag `Tensor(1)` values (`._0`,
/// Parakeet's raw weights) fall through to the kernel's runtime length check.
fn normParamTagCheck(comptime Obj: type, comptime op_name: []const u8, comptime param_name: []const u8, comptime tag: Tag) void {
    if (Obj.axis_tags.len != 1)
        @compileError(op_name ++ ": ." ++ param_name ++ " must be a rank-1 [" ++ @tagName(tag) ++ "] tensor");
    if (Obj.axis_tags[0] != tag and Obj.axis_tags[0] != ._0)
        @compileError(op_name ++ ": ." ++ param_name ++ " tag ." ++ @tagName(Obj.axis_tags[0]) ++ " does not match the normalized axis ." ++ @tagName(tag));
}

const AttentionMask = enum { causal, bidirectional };

const AttentionKvRepr = enum { f32_kv, f16_kv, q8_kv, multi_f16_kv, multi_q8_kv };

/// Comptime KV-representation of a `groupedAttention` k/v argument. Slice
/// shapes are classified before tensor objects (a `[]const BlockQ8_0` is a
/// pointer type too); anything else is a curated @compileError.
fn attentionKvRepr(comptime T: type, comptime which: []const u8) AttentionKvRepr {
    const info = @typeInfo(T);
    if (info == .pointer and info.pointer.size == .slice) {
        const Elem = info.pointer.child;
        if (Elem == BlockQ8_0) return .q8_kv;
        if (Elem == []const f16) return .multi_f16_kv;
        if (Elem == []const BlockQ8_0) return .multi_q8_kv;
    }
    if (info == .pointer and info.pointer.size == .one) {
        const Obj = info.pointer.child;
        if (@typeInfo(Obj) == .@"struct" and @hasDecl(Obj, "axis_tags") and @hasDecl(Obj, "dtype")) {
            if (comptime tagsEqual(Obj.axis_tags, .{ .seq, .kv_head, .d })) {
                if (Obj.dtype == .f32) return .f32_kv;
                if (Obj.dtype == .f16) return .f16_kv;
            }
            @compileError("groupedAttention: " ++ which ++ " must be an f32/f16 tensor tagged .{ .seq, .kv_head, .d }; got " ++ @typeName(Obj));
        }
    }
    @compileError("groupedAttention: unsupported " ++ which ++ " type " ++ @typeName(T) ++
        " (want *Tensor(.{ .seq, .kv_head, .d }) f32/f16, []const BlockQ8_0, []const []const f16, or []const []const BlockQ8_0)");
}

/// Comptime layout of a `*const <packed RHS>` argument, with a curated
/// @compileError (naming `op_name` and the offending type) for anything else.
fn packedRhsLayout(comptime T: type, comptime op_name: []const u8) backend_mod.PackedRhsLayout {
    const info = @typeInfo(T);
    if (info != .pointer or info.pointer.size != .one)
        @compileError(op_name ++ " expects a pointer to a packed matmul RHS (e.g. *const QuantizedMatmulRhsQ6_Kx4); got " ++ @typeName(T));
    const Rhs = info.pointer.child;
    if (@typeInfo(Rhs) != .@"struct" or !@hasDecl(Rhs, "layout") or @TypeOf(Rhs.layout) != backend_mod.PackedRhsLayout)
        @compileError(op_name ++ ": " ++ @typeName(Rhs) ++ " is not a packed quantized matmul RHS");
    return Rhs.layout;
}

fn QuantizedConstantTensor(comptime tags_spec: anytype, comptime tensor_dtype: DType) type {
    const tags = normalizeTags(tags_spec);
    comptime validateUniqueTags(tags);
    const tag_rank = tags.len;
    if (tag_rank > tensor_mod.max_rank) @compileError("too many tensor tags");

    const RawTypedTensor = tensor_mod.TensorOf(tensor_dtype);
    const Elem = dtype_mod.Storage(tensor_dtype);

    return struct {
        pub const axis_tags = tags;
        pub const tag_count = tag_rank;
        pub const tensor_rank = rawRank(tag_rank);
        pub const dtype = tensor_dtype;

        value: RawTypedTensor,

        const Self = @This();

        /// Consumes `value` on success; on error, ownership stays with the caller.
        pub fn constant(ctx: *ExecContext, value: RawTypedTensor) !Self {
            _ = ctx;
            var v = value;
            try validateTensorRankOf(tensor_dtype, tags, &v);
            return .{ .value = v };
        }

        pub fn fromTensor(ctx: *ExecContext, value: RawTypedTensor) !Self {
            return try Self.constant(ctx, value);
        }

        pub fn fromBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self {
            var value = try ctx.fromStorageSliceRankTyped(tensor_dtype, tensor_rank, raw_shape, values);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        pub fn fromStorageSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self {
            return Self.fromBlocks(ctx, raw_shape, values);
        }

        pub fn fromBorrowedBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []Elem) !Self {
            var value = try ctx.fromBorrowedStorageSliceRankTyped(tensor_dtype, tensor_rank, raw_shape, values);
            errdefer value.deinit();
            return try Self.constant(ctx, value);
        }

        pub fn deinit(self: *Self) void {
            self.value.deinit();
            self.* = undefined;
        }

        pub fn asRawTensor(self: *const Self) *const RawTypedTensor {
            return &self.value;
        }

        pub fn data(self: *Self) ![]Elem {
            return self.value.dataChecked();
        }

        pub fn dataConst(self: *const Self) ![]const Elem {
            return self.value.dataConstChecked();
        }

        pub fn copyTo(self: *const Self, dst: []Elem) !void {
            return self.value.copyTo(dst);
        }

        pub fn requiresGrad(_: *const Self) bool {
            return false;
        }

        pub fn axis(comptime tag: Tag) usize {
            return tagIndexOrCompileError(tags, tag);
        }

        pub fn hasTag(comptime tag: Tag) bool {
            return comptime tagIndex(tags, tag) != null;
        }

        pub fn dim(self: *const Self, comptime tag: Tag) usize {
            return self.asRawTensor().shape.at(axis(tag));
        }

        pub fn shape(self: *const Self) [tensor_rank]usize {
            var out: [tensor_rank]usize = undefined;
            inline for (0..tensor_rank) |i| {
                out[i] = self.asRawTensor().shape.at(i);
            }
            return out;
        }

        pub fn withTags(self: *const Self, ctx: *ExecContext, comptime new_tags_spec: anytype) !Tensor(.{ .dtype = tensor_dtype, .tags = normalizeTags(new_tags_spec) }) {
            const new_tags = normalizeTags(new_tags_spec);
            comptime {
                validateUniqueTags(new_tags);
                if (new_tags.len != tag_rank) @compileError("withTags requires the same rank");
            }
            var value = try self.asRawTensor().cloneView();
            errdefer value.deinit();
            return Tensor(.{ .dtype = tensor_dtype, .tags = new_tags }).fromTensor(ctx, value);
        }

        pub fn to(self: *const Self, ctx: *ExecContext, comptime target_dtype: DType) !Tensor(.{ .dtype = target_dtype, .tags = tags }) {
            comptime if (target_dtype != .f32) @compileError("block-quantized tensors can currently only be converted to f32");
            var value = try ctx.dequantizeTensorTyped(tensor_dtype, self.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = target_dtype, .tags = tags }).fromTensor(ctx, value);
        }

        pub fn materialize(self: *const Self, ctx: *ExecContext) !Self {
            var value = try ctx.materializeTyped(tensor_dtype, self.asRawTensor());
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn concat(self: *const Self, ctx: *ExecContext, comptime tag: Tag, others: []const *const Self) !Self {
            comptime {
                if (tag_rank != 2) @compileError("block-quantized concat currently requires a rank-2 tensor");
                if (axis(tag) != 0) @compileError("block-quantized concat currently supports the row axis only");
            }

            var raw_inputs = try ctx.allocator.alloc(*const tensor_mod.TensorOf(tensor_dtype), others.len + 1);
            defer ctx.allocator.free(raw_inputs);

            raw_inputs[0] = self.asRawTensor();
            for (others, 0..) |other, i| raw_inputs[i + 1] = other.asRawTensor();

            var value = try ctx.concatQuantizedRowsTyped(tensor_dtype, raw_inputs);
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        /// Pack this rank-2 quantized weight into the ISA-best packed matmul
        /// RHS layout for its dtype (see `PackedRhs`): q8_0→x4, q6_k→x4,
        /// q5_k→x8, q4_k→x2mmla on aarch64+i8mm targets else x8. Use
        /// `packRhsLayout` to force a specific layout instead.
        pub fn packRhs(self: *const Self, ctx: *ExecContext) !PackedRhs(tensor_dtype) {
            comptime if (tag_rank != 2) @compileError("packRhs requires a rank-2 tensor");
            return switch (comptime tensor_dtype) {
                .q8_0 => ctx.packMatmulRhsQ8_0x4(self.asRawTensor()),
                .q6_k => ctx.packMatmulRhsQ6_Kx4(self.asRawTensor()),
                .q5_k => ctx.packMatmulRhsQ5_Kx8(self.asRawTensor()),
                .q4_k => if (comptime backend_mod.supports_q4_k_mmla)
                    ctx.packMatmulRhsQ4_Kx2Mmla(self.asRawTensor())
                else
                    ctx.packMatmulRhsQ4_Kx8(self.asRawTensor()),
                else => @compileError("packRhs: no packed matmul RHS layout for a ." ++ @tagName(tensor_dtype) ++ " tensor"),
            };
        }

        /// Explicit-layout escape hatch over `packRhs`: force a specific packed
        /// layout, comptime-validated against the tensor dtype. Needed e.g. to
        /// exercise the fused x8 kernels on hardware where `packRhs` would
        /// select x2mmla, at the cost of the ISA-best kernel.
        pub fn packRhsLayout(self: *const Self, ctx: *ExecContext, comptime layout: backend_mod.PackedRhsLayout) !backend_mod.PackedRhsFor(layout) {
            comptime {
                if (tag_rank != 2) @compileError("packRhsLayout requires a rank-2 tensor");
                const want: DType = switch (layout) {
                    .q8_0x4 => .q8_0,
                    .q6_kx4 => .q6_k,
                    .q4_kx8, .q4_kx2mmla => .q4_k,
                    .q5_kx8 => .q5_k,
                    .q4_kx4 => @compileError("packRhsLayout: the Q4_Kx4 layout has no facade entry (kernel-comparison surface below the facade)"),
                };
                if (tensor_dtype != want) @compileError("packRhsLayout(." ++ @tagName(layout) ++ ") requires a ." ++ @tagName(want) ++ " tensor");
            }
            return switch (comptime layout) {
                .q8_0x4 => ctx.packMatmulRhsQ8_0x4(self.asRawTensor()),
                .q6_kx4 => ctx.packMatmulRhsQ6_Kx4(self.asRawTensor()),
                .q4_kx8 => ctx.packMatmulRhsQ4_Kx8(self.asRawTensor()),
                .q4_kx2mmla => ctx.packMatmulRhsQ4_Kx2Mmla(self.asRawTensor()),
                .q5_kx8 => ctx.packMatmulRhsQ5_Kx8(self.asRawTensor()),
                .q4_kx4 => unreachable, // rejected by the comptime block above
            };
        }

        pub fn getRows(
            self: *const Self,
            ctx: *ExecContext,
            comptime tag: Tag,
            indices: []const usize,
            comptime out_tag: Tag,
        ) !Tensor(.{ .dtype = .f32, .tags = replaceTag(tags, tag, out_tag) }) {
            comptime {
                if (tag_rank != 2) @compileError("quantized getRows currently requires a rank-2 tensor");
                if (axis(tag) != 0) @compileError("quantized getRows gathers rows from the first axis");
            }
            const result_tags = replaceTag(tags, tag, out_tag);
            var value = try ctx.getRowsQuantizedTyped(tensor_dtype, self.asRawTensor(), indices);
            errdefer value.deinit();
            return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
        }
    };
}

fn TypedConstantTensor(comptime tags_spec: anytype, comptime tensor_dtype: DType) type {
    if (comptime dtype_mod.supportsForwardFloatMath(tensor_dtype)) {
        return TypedFloatConstantTensor(tags_spec, tensor_dtype);
    }
    return TypedScalarConstantTensor(tags_spec, tensor_dtype);
}

/// Shared constructor/accessor tier for the two typed-constant branches,
/// extending the `typedConstant*` shared-fn pattern below to everything the
/// branches duplicated verbatim: the branches differ only in which MATH decls
/// they add (`TypedFloatConstantTensor` layers to/add/.../dot on top).
fn TypedConstantBase(comptime SelfT: type, comptime tags: anytype, comptime tensor_dtype: DType) type {
    const tensor_rank = rawRank(tags.len);
    const RawTypedTensor = tensor_mod.TensorOf(tensor_dtype);
    const Elem = Scalar(tensor_dtype);

    return struct {
        /// Consumes `value` on success; on error, ownership stays with the caller.
        pub fn constant(ctx: *ExecContext, value: RawTypedTensor) !SelfT {
            _ = ctx;
            var v = value;
            try validateTensorRankOf(tensor_dtype, tags, &v);
            return .{ .value = v };
        }

        pub fn fromTensor(ctx: *ExecContext, value: RawTypedTensor) !SelfT {
            return try @This().constant(ctx, value);
        }

        pub fn fromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !SelfT {
            var value = try ctx.fromSliceRankTyped(tensor_dtype, tensor_rank, raw_shape, values);
            errdefer value.deinit();
            return try @This().constant(ctx, value);
        }

        /// Zero-copy wrap caller-owned READ-ONLY typed storage as a no-grad
        /// constant tensor without `@constCast` at the call site. Read-only
        /// borrow: `values` must outlive the tensor and must not be mutated
        /// (see the f32 `fromBorrowedConstSlice` contract).
        pub fn fromBorrowedConstSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !SelfT {
            var value = try ctx.fromBorrowedSliceRankTyped(tensor_dtype, tensor_rank, raw_shape, @constCast(values));
            errdefer value.deinit();
            return try @This().constant(ctx, value);
        }

        /// Allocate an uninitialized no-grad typed tensor of the tag-implied rank.
        pub fn empty(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !SelfT {
            var value = try ctx.emptyRankTyped(tensor_dtype, tensor_rank, raw_shape);
            errdefer value.deinit();
            return try @This().constant(ctx, value);
        }

        /// Allocate a zero-filled no-grad typed tensor.
        pub fn zeros(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !SelfT {
            var value = try ctx.zerosTyped(tensor_dtype, &raw_shape);
            errdefer value.deinit();
            return try @This().constant(ctx, value);
        }

        /// Allocate a one-filled no-grad typed tensor.
        pub fn ones(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !SelfT {
            var value = try ctx.onesTyped(tensor_dtype, &raw_shape);
            errdefer value.deinit();
            return try @This().constant(ctx, value);
        }

        /// `empty` with `self`'s shape (same dtype and tags via `SelfT`).
        pub fn emptyLike(self: *const SelfT, ctx: *ExecContext) !SelfT {
            return SelfT.empty(ctx, self.shape());
        }

        /// `zeros` with `self`'s shape.
        pub fn zerosLike(self: *const SelfT, ctx: *ExecContext) !SelfT {
            return SelfT.zeros(ctx, self.shape());
        }

        /// `ones` with `self`'s shape.
        pub fn onesLike(self: *const SelfT, ctx: *ExecContext) !SelfT {
            return SelfT.ones(ctx, self.shape());
        }

        pub fn item(self: *const SelfT) !Elem {
            if (!self.value.isScalar()) return TensorError.InvalidShape;
            return (try self.value.dataConstChecked())[0];
        }

        pub fn data(self: *SelfT) ![]Elem {
            return self.value.dataChecked();
        }

        pub fn dataConst(self: *const SelfT) ![]const Elem {
            return self.value.dataConstChecked();
        }

        pub fn copyTo(self: *const SelfT, dst: []Elem) !void {
            return self.value.copyTo(dst);
        }

        pub fn axis(comptime tag: Tag) usize {
            return tagIndexOrCompileError(tags, tag);
        }

        pub fn hasTag(comptime tag: Tag) bool {
            return comptime tagIndex(tags, tag) != null;
        }
    };
}

fn TypedScalarConstantTensor(comptime tags_spec: anytype, comptime tensor_dtype: DType) type {
    const tags = normalizeTags(tags_spec);
    comptime validateUniqueTags(tags);
    const tag_rank = tags.len;
    if (tag_rank > tensor_mod.max_rank) @compileError("too many tensor tags");

    const RawTypedTensor = tensor_mod.TensorOf(tensor_dtype);

    return struct {
        pub const axis_tags = tags;
        pub const tag_count = tag_rank;
        pub const tensor_rank = rawRank(tag_rank);
        pub const dtype = tensor_dtype;

        value: RawTypedTensor,
        const Self = @This();
        const base = TypedConstantBase(Self, tags, tensor_dtype);

        pub const constant = base.constant;
        pub const fromTensor = base.fromTensor;
        pub const fromSlice = base.fromSlice;
        pub const fromBorrowedConstSlice = base.fromBorrowedConstSlice;
        pub const empty = base.empty;
        pub const zeros = base.zeros;
        pub const ones = base.ones;
        pub const emptyLike = base.emptyLike;
        pub const zerosLike = base.zerosLike;
        pub const onesLike = base.onesLike;
        pub const item = base.item;
        pub const data = base.data;
        pub const dataConst = base.dataConst;
        pub const copyTo = base.copyTo;
        pub const axis = base.axis;
        pub const hasTag = base.hasTag;

        pub const deinit = typedConstantDeinit;
        pub const asRawTensor = typedConstantAsRawTensor;
        pub const requiresGrad = typedConstantRequiresGrad;

        pub const dim = typedConstantDim;
        pub const shape = typedConstantShape;
        pub const materialize = typedConstantMaterialize;
        pub const withTags = typedConstantWithTags;
        pub const alignTo = typedConstantAlignTo;
        pub const permuteTo = typedConstantPermuteTo;
        pub const transpose = typedConstantTranspose;
        pub const insertAxis = typedConstantInsertAxis;
        pub const squeeze = typedConstantSqueeze;
        pub const broadcastTo = typedConstantBroadcastTo;
        pub const gather = typedConstantGather;
        pub const narrow = typedConstantNarrow;
        pub const concat = typedConstantConcat;
        pub const setSlice = typedConstantSetSlice;
        pub const setRows = typedConstantSetRows;

        // Integer forward math (§4.19): wrapping two's-complement
        // pointwise, explicit division, i64-returning reductions, and
        // scalar casts. On `.bool` the arithmetic entries are compile
        // errors — only `to` and the counting `sum`/`sumAll` apply.
        pub const to = typedConstantTo;
        pub const add = typedConstantAdd;
        pub const sub = typedConstantSub;
        pub const mul = typedConstantMul;
        pub const maximum = typedConstantMaximum;
        pub const minimum = typedConstantMinimum;
        pub const divTrunc = typedConstantDivTrunc;
        pub const divFloor = typedConstantDivFloor;
        pub const sum = typedConstantSum;
        pub const sumAll = typedConstantSumAll;

        // Masks (§4.6): integer `compare` is exact at any magnitude; the
        // logical combinators live on the `.bool` branch.
        pub const compare = typedConstantCompare;
        pub const logicalAnd = typedConstantLogicalAnd;
        pub const logicalOr = typedConstantLogicalOr;
        pub const logicalXor = typedConstantLogicalXor;
        pub const logicalNot = typedConstantLogicalNot;
    };
}

fn TypedFloatConstantTensor(comptime tags_spec: anytype, comptime tensor_dtype: DType) type {
    const tags = normalizeTags(tags_spec);
    comptime validateUniqueTags(tags);
    const tag_rank = tags.len;
    if (tag_rank > tensor_mod.max_rank) @compileError("too many tensor tags");

    const RawTypedTensor = tensor_mod.TensorOf(tensor_dtype);

    return struct {
        pub const axis_tags = tags;
        pub const tag_count = tag_rank;
        pub const tensor_rank = rawRank(tag_rank);
        pub const dtype = tensor_dtype;

        value: RawTypedTensor,
        grad_state: ?*GradState = null,
        scope_owned: bool = false,
        const Self = @This();
        const base = TypedConstantBase(Self, tags, tensor_dtype);

        pub const constant = base.constant;
        pub const fromTensor = base.fromTensor;
        pub const fromSlice = base.fromSlice;
        pub const fromBorrowedConstSlice = base.fromBorrowedConstSlice;
        pub const empty = base.empty;
        pub const zeros = base.zeros;
        pub const ones = base.ones;
        pub const emptyLike = base.emptyLike;
        pub const zerosLike = base.zerosLike;
        pub const onesLike = base.onesLike;
        pub const item = base.item;
        pub const data = base.data;
        pub const dataConst = base.dataConst;
        pub const copyTo = base.copyTo;
        pub const axis = base.axis;
        pub const hasTag = base.hasTag;

        /// Trainable 16-bit leaf: the VALUE is stored in this dtype, the
        /// accumulated gradient is ALWAYS f32 (there is no 16-bit gradient
        /// anywhere). f16/bf16 only — f64 training is unsupported.
        pub fn variable(ctx: *ExecContext, value: RawTypedTensor) !Self {
            comptime requireHalfFloatGrad(tensor_dtype, "variable");
            var v = value;
            try validateTensorRankOf(tensor_dtype, tags, &v);
            const state = try GradState.leaf(ctx.allocator);
            errdefer state.deinit();
            return .{ .value = v, .grad_state = state };
        }

        pub fn variableFromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Scalar(tensor_dtype)) !Self {
            comptime requireHalfFloatGrad(tensor_dtype, "variableFromSlice");
            var value = try ctx.fromSliceRankTyped(tensor_dtype, tensor_rank, raw_shape, values);
            errdefer value.deinit();
            return try Self.variable(ctx, value);
        }

        pub fn requiresGrad(self: *const Self) bool {
            return self.grad_state != null;
        }

        pub fn zeroGrad(self: *const Self) void {
            if (self.grad_state) |state| state.zeroGrad();
        }

        /// The accumulated gradient as an owned f32 constant (null before
        /// backward). The gradient of a 16-bit tensor is f32 by contract.
        pub fn grad(self: *const Self, ctx: *ExecContext) !?Tensor(.{ .dtype = .f32, .tags = tags }) {
            comptime requireHalfFloatGrad(tensor_dtype, "grad");
            const state = self.grad_state orelse return null;
            var value = (try state.gradClone(ctx.allocator)) orelse return null;
            errdefer value.deinit();
            return try Tensor(.{ .dtype = .f32, .tags = tags }).constant(ctx, value);
        }

        /// Aliasing f32 view of the accumulated gradient (see `grad`).
        pub fn gradView(self: *const Self, ctx: *ExecContext) !?Tensor(.{ .dtype = .f32, .tags = tags }) {
            comptime requireHalfFloatGrad(tensor_dtype, "gradView");
            const state = self.grad_state orelse return null;
            var value = (try state.gradView()) orelse return null;
            errdefer value.deinit();
            return try Tensor(.{ .dtype = .f32, .tags = tags }).constant(ctx, value);
        }

        /// No-grad view of the same storage (caller-owned constant).
        pub fn detach(self: *const Self, ctx: *ExecContext) !Self {
            var value = try self.value.cloneView();
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn deinit(self: *Self) void {
            if (self.scope_owned) return; // borrow: the exec scope owns value + node
            self.value.deinit();
            if (self.grad_state) |state| state.deinit();
            self.* = undefined;
        }

        pub const asRawTensor = typedConstantAsRawTensor;

        pub const dim = typedConstantDim;
        pub const shape = typedConstantShape;
        pub const materialize = typedConstantMaterialize;
        pub const withTags = typedConstantWithTags;
        pub const alignTo = typedConstantAlignTo;
        pub const permuteTo = typedConstantPermuteTo;
        pub const transpose = typedConstantTranspose;
        pub const insertAxis = typedConstantInsertAxis;
        pub const squeeze = typedConstantSqueeze;
        pub const broadcastTo = typedConstantBroadcastTo;
        pub const gather = typedConstantGather;
        pub const narrow = typedConstantNarrow;
        pub const concat = typedConstantConcat;
        pub const setSlice = typedConstantSetSlice;
        pub const setRows = typedConstantSetRows;

        pub const to = typedConstantTo;
        pub const add = typedConstantAdd;
        pub const sub = typedConstantSub;
        pub const mul = typedConstantMul;
        pub const div = typedConstantDiv;
        pub const sum = typedConstantSum;
        pub const mean = typedConstantMean;
        pub const sumAll = typedConstantSumAll;
        pub const dot = typedConstantDot;

        // Structural ops (views / data movement; every typed float dtype).
        pub const split = typedConstantSplit;
        pub const merge = typedConstantMerge;
        pub const flatten = typedConstantFlatten;
        pub const reshape = typedConstantReshape;
        pub const sliceStep = typedConstantSliceStep;
        pub const flip = typedConstantFlip;
        pub const roll = typedConstantRoll;
        pub const stack = typedConstantStack;
        pub const repeatAxis = typedConstantRepeatAxis;
        pub const scale = typedConstantScale;
        pub const divScalar = typedConstantDivScalar;

        // Widened forward math (f16/bf16 only: f32 compute, one final round).
        pub const unary = typedConstantUnary;
        pub const relu = TypedUnaryMethod(.relu).call;
        pub const exp = TypedUnaryMethod(.exp).call;
        pub const sqrt = TypedUnaryMethod(.sqrt).call;
        pub const rsqrt = TypedUnaryMethod(.rsqrt).call;
        pub const sigmoid = TypedUnaryMethod(.sigmoid).call;
        pub const silu = TypedUnaryMethod(.silu).call;
        pub const log = TypedUnaryMethod(.log).call;
        pub const log1p = TypedUnaryMethod(.log1p).call;
        pub const neg = TypedUnaryMethod(.neg).call;
        pub const abs = TypedUnaryMethod(.abs).call;
        pub const sin = TypedUnaryMethod(.sin).call;
        pub const cos = TypedUnaryMethod(.cos).call;
        pub const tanh = TypedUnaryMethod(.tanh).call;
        pub const fastTanh = TypedUnaryMethod(.fast_tanh).call;
        pub const softcap30 = TypedUnaryMethod(.softcap_30).call;
        pub const softcap15 = TypedUnaryMethod(.softcap_15).call;
        pub const gelu = TypedUnaryMethod(.gelu).call;
        pub const quickGelu = TypedUnaryMethod(.quick_gelu).call;
        pub const elu = TypedUnaryMethod(.elu).call;
        pub const geluErf = TypedUnaryMethod(.gelu_erf).call;
        pub const floor = TypedUnaryMethod(.floor).call;
        pub const ceil = TypedUnaryMethod(.ceil).call;
        pub const round = TypedUnaryMethod(.round).call;
        pub const sign = TypedUnaryMethod(.sign).call;
        pub const reciprocal = TypedUnaryMethod(.reciprocal).call;
        pub const leakyRelu = typedConstantLeakyRelu;
        pub const clamp = typedConstantClamp;
        pub const addScalar = typedConstantAddScalar;
        pub const subScalar = typedConstantSubScalar;
        pub const powScalar = typedConstantPowScalar;
        pub const maximum = typedConstantMaximum;
        pub const minimum = typedConstantMinimum;
        pub const gated = typedConstantGated;
        pub const glu = typedConstantGlu;
        pub const swiglu = typedConstantSwiglu;
        pub const geglu = typedConstantGeglu;
        pub const softmax = typedConstantSoftmax;
        pub const logSoftmax = typedConstantLogSoftmax;
        pub const rmsNorm = typedConstantRmsNorm;
        pub const rmsNormMul = typedConstantRmsNormMul;
        pub const layerNorm = typedConstantLayerNorm;
        pub const cumsum = typedConstantCumsum;
        pub const cumprod = typedConstantCumprod;
        pub const where = typedConstantWhere;
        pub const maskedFill = typedConstantMaskedFill;
        pub const compare = typedConstantCompare;
        pub const pad = typedConstantPad;
        pub const einsum = typedConstantEinsum;

        // Widened reductions (f16/bf16 only; f32 result per §8.3).
        pub const max = typedConstantMax;
        pub const min = typedConstantMin;
        pub const argmax = typedConstantArgmax;
        pub const prod = typedConstantProd;
        pub const variance = typedConstantVariance;
        pub const logsumexp = typedConstantLogsumexp;
    };
}

fn typedConstantDeinit(self: anytype) void {
    self.value.deinit();
    self.* = undefined;
}

fn typedConstantAsRawTensor(self: anytype) *const tensor_mod.TensorOf(TensorObject(@TypeOf(self)).dtype) {
    return &self.value;
}

fn typedConstantRequiresGrad(_: anytype) bool {
    return false;
}

fn typedConstantDim(self: anytype, comptime tag: Tag) usize {
    const Self = TensorObject(@TypeOf(self));
    return self.asRawTensor().shape.at(Self.axis(tag));
}

fn typedConstantShape(self: anytype) [TensorObject(@TypeOf(self)).tensor_rank]usize {
    const Self = TensorObject(@TypeOf(self));
    var out: [Self.tensor_rank]usize = undefined;
    inline for (0..Self.tensor_rank) |i| {
        out[i] = self.asRawTensor().shape.at(i);
    }
    return out;
}

fn typedConstantMaterialize(self: anytype, ctx: *ExecContext) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    var value = try ctx.materializeTyped(Self.dtype, self.asRawTensor());
    errdefer value.deinit();
    return Self.fromTensor(ctx, value);
}

fn typedConstantWithTags(self: anytype, ctx: *ExecContext, comptime new_tags_spec: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(new_tags_spec) }) {
    const Self = TensorObject(@TypeOf(self));
    const new_tags = normalizeTags(new_tags_spec);
    comptime {
        validateUniqueTags(new_tags);
        if (new_tags.len != Self.tag_count) @compileError("withTags requires the same rank");
    }
    return typedConstantAxisView(self, ctx, identityAxes(Self.tag_count), new_tags);
}

fn typedConstantAlignTo(self: anytype, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(target_tags_spec) }) {
    const Self = TensorObject(@TypeOf(self));
    const target_tags = normalizeTags(target_tags_spec);
    return typedConstantAxisView(self, ctx, alignAxes(Self.axis_tags, target_tags), target_tags);
}

fn typedConstantPermuteTo(self: anytype, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(target_tags_spec) }) {
    const Self = TensorObject(@TypeOf(self));
    const target_tags = normalizeTags(target_tags_spec);
    comptime validateSameTagSet(Self.axis_tags, target_tags);
    return typedConstantAxisView(self, ctx, alignAxes(Self.axis_tags, target_tags), target_tags);
}

fn typedConstantTranspose(self: anytype, ctx: *ExecContext, comptime target_tags_spec: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(target_tags_spec) }) {
    return typedConstantPermuteTo(self, ctx, target_tags_spec);
}

fn typedConstantInsertAxis(self: anytype, ctx: *ExecContext, comptime tag: Tag, comptime axis_index: usize) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = insertTagAt(TensorObject(@TypeOf(self)).axis_tags, tag, axis_index) }) {
    const Self = TensorObject(@TypeOf(self));
    const result_tags = insertTagAt(Self.axis_tags, tag, axis_index);
    return typedConstantAxisView(self, ctx, insertAxes(Self.tag_count, axis_index), result_tags);
}

fn typedConstantSqueeze(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
    const Self = TensorObject(@TypeOf(self));
    const result_tags = removeTag(Self.axis_tags, tag);
    const axis_index = comptime tagIndexOrCompileError(Self.axis_tags, tag);
    if (self.asRawTensor().shape.at(axis_index) != 1) return TensorError.InvalidShape;
    return typedConstantAxisView(self, ctx, squeezeAxes(Self.tag_count, axis_index), result_tags);
}

fn typedConstantBroadcastTo(
    self: anytype,
    ctx: *ExecContext,
    comptime target_tags_spec: anytype,
    target_shape: [normalizeTags(target_tags_spec).len]usize,
) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(target_tags_spec) }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    const target_tags = normalizeTags(target_tags_spec);
    var value = try broadcastTensorToOf(Self.dtype, Self.axis_tags, self.asRawTensor(), target_tags, target_shape);
    errdefer value.deinit();
    return Tensor(.{ .dtype = Self.dtype, .tags = target_tags }).fromTensor(ctx, value);
}

fn typedConstantGather(
    self: anytype,
    ctx: *ExecContext,
    comptime tag: Tag,
    indices: []const usize,
    comptime out_tag: Tag,
) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = replaceTag(TensorObject(@TypeOf(self)).axis_tags, tag, out_tag) }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    const result_tags = replaceTag(Self.axis_tags, tag, out_tag);
    var value = try ctx.gatherAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), Self.axis(tag), indices);
    errdefer value.deinit();
    return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
}

fn typedConstantNarrow(self: anytype, ctx: *ExecContext, comptime tag: Tag, start: usize, length: usize) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    var value = try ctx.narrowAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), Self.axis(tag), start, length);
    errdefer value.deinit();
    return Self.fromTensor(ctx, value);
}

fn typedConstantConcat(self: anytype, ctx: *ExecContext, comptime tag: Tag, others: []const *const TensorObject(@TypeOf(self))) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    for (others) |other_item| try typedRequireNoGrad(other_item);
    const Self = TensorObject(@TypeOf(self));
    const RawTypedTensor = tensor_mod.TensorOf(Self.dtype);
    var raw_inputs = try ctx.allocator.alloc(*const RawTypedTensor, others.len + 1);
    defer ctx.allocator.free(raw_inputs);

    raw_inputs[0] = self.asRawTensor();
    for (others, 0..) |other, i| raw_inputs[i + 1] = other.asRawTensor();

    var value = try ctx.concatAxisRankTyped(Self.dtype, Self.tag_count, raw_inputs, Self.axis(tag));
    errdefer value.deinit();
    return Self.fromTensor(ctx, value);
}

fn typedConstantSetSlice(self: anytype, ctx: *ExecContext, comptime tag: Tag, start: usize, update: *const TensorObject(@TypeOf(self))) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    try typedRequireNoGrad(update);
    const Self = TensorObject(@TypeOf(self));
    var value = try ctx.setSliceAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), update.asRawTensor(), Self.axis(tag), start);
    errdefer value.deinit();
    return Self.fromTensor(ctx, value);
}

fn typedConstantSetRows(self: anytype, ctx: *ExecContext, comptime tag: Tag, indices: []const usize, update: *const TensorObject(@TypeOf(self))) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    try typedRequireNoGrad(update);
    const Self = TensorObject(@TypeOf(self));
    var value = try ctx.setRowsAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), update.asRawTensor(), Self.axis(tag), indices);
    errdefer value.deinit();
    return Self.fromTensor(ctx, value);
}

fn typedConstantAxisView(self: anytype, ctx: *ExecContext, comptime axes: anytype, comptime target_tags: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = target_tags }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    var value = try axisViewTensorOf(Self.dtype, self.asRawTensor(), axes, target_tags);
    errdefer value.deinit();
    return Tensor(.{ .dtype = Self.dtype, .tags = target_tags }).fromTensor(ctx, value);
}

fn typedConstantTo(self: anytype, ctx: *ExecContext, comptime target_dtype: DType) !Tensor(.{ .dtype = target_dtype, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
    const Self = TensorObject(@TypeOf(self));
    if (comptime target_dtype != .f32) {
        if (self.requiresGrad()) return error.GradientCastUnsupported;
    }
    var value = try ctx.castTyped(Self.dtype, target_dtype, self.asRawTensor());
    errdefer value.deinit();
    if (comptime target_dtype == .f32) {
        if (comptime @hasField(Self, "grad_state")) {
            // Differentiable widen: the f32 result joins the f32 graph and
            // the 16-bit source receives the upstream gradient unchanged.
            return finishOp(Self.axis_tags, ctx, value, self.requiresGrad(), CastBackward(Self.axis_tags), .{ ctx.allocator, self.grad_state });
        }
        // Integer/bool sources are grad-free: a plain f32 constant.
        return finishNoGrad(Self.axis_tags, ctx, value);
    }
    return Tensor(.{ .dtype = target_dtype, .tags = Self.axis_tags }).fromTensor(ctx, value);
}

fn typedConstantAdd(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    const Self = TensorObject(@TypeOf(self));
    return typedPointwise(Self.dtype, .add, self, ctx, other);
}

fn typedConstantSub(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    const Self = TensorObject(@TypeOf(self));
    return typedPointwise(Self.dtype, .sub, self, ctx, other);
}

fn typedConstantMul(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    const Self = TensorObject(@TypeOf(self));
    return typedPointwise(Self.dtype, .mul, self, ctx, other);
}

fn typedConstantDiv(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    const Self = TensorObject(@TypeOf(self));
    return typedPointwise(Self.dtype, .div, self, ctx, other);
}

fn typedConstantSum(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, TensorObject(@TypeOf(self)).dtype), .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    const result_tags = removeTag(Self.axis_tags, tag);
    var value = try ctx.sumAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), Self.axis(tag));
    errdefer value.deinit();
    return Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, Self.dtype), .tags = result_tags }).fromTensor(ctx, value);
}

fn typedConstantMean(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, TensorObject(@TypeOf(self)).dtype), .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    const result_tags = removeTag(Self.axis_tags, tag);
    var value = try ctx.meanAxisRankTyped(Self.dtype, Self.tag_count, self.asRawTensor(), Self.axis(tag));
    errdefer value.deinit();
    return Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, Self.dtype), .tags = result_tags }).fromTensor(ctx, value);
}

fn typedConstantSumAll(self: anytype, ctx: *ExecContext) !Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, TensorObject(@TypeOf(self)).dtype), .tags = .{} }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    var value = try ctx.sumTyped(Self.dtype, self.asRawTensor());
    errdefer value.deinit();
    return Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, Self.dtype), .tags = .{} }).fromTensor(ctx, value);
}

fn typedConstantDot(self: anytype, ctx: *ExecContext, other: anytype, comptime contract_tag: Tag) !Tensor(.{ .dtype = dtype_mod.outputDType(.matmul, TensorObject(@TypeOf(self)).dtype), .tags = dotResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag) }) {
    const Self = TensorObject(@TypeOf(self));
    return typedDot(Self.dtype, self, ctx, other, contract_tag);
}

// Widened typed-float ops: forward coverage for ops with no native typed
// kernel. The input widens to f32, the f32 exec kernel runs, and the result
// narrows ONCE on store — f32 accumulation with a single final round, the
// §8.3 policy. f64 is excluded at comptime: f64 math must stay f64, and
// rounding it through f32 would silently lose precision.
fn requireWidenedTypedFloat(comptime tensor_dtype: DType, comptime what: []const u8) void {
    if (tensor_dtype != .f16 and tensor_dtype != .bf16) {
        @compileError(what ++ " on the typed float branch is f16/bf16 only (it computes through f32; f64 must not round through f32 — cast explicitly)");
    }
}

fn requireHalfFloatGrad(comptime tensor_dtype: DType, comptime what: []const u8) void {
    if (tensor_dtype != .f16 and tensor_dtype != .bf16) {
        @compileError(what ++ " requires an f16/bf16 tensor (gradients are always f32; f64 training is unsupported)");
    }
}

/// Typed forward ops are no-grad: a grad-requiring operand would silently
/// drop its graph, so it is rejected instead (`to(.f32)` is the trained
/// path; the differentiable typed entries are `to` and the mixed-RHS
/// `dot`/`einsum`).
fn typedRequireNoGrad(operand: anytype) !void {
    const Operand = TensorObject(@TypeOf(operand));
    if (comptime @hasField(Operand, "grad_state")) {
        if (operand.grad_state != null) return error.UnsupportedGradient;
    }
}

/// Scope payload for a grad-carrying 16-bit result: the exec-scope slot
/// holds f32 values only, so the typed value travels inside the type-erased
/// node payload with a destructor that frees value + graph node together.
fn TypedScopePayload(comptime tensor_dtype: DType) type {
    return struct {
        allocator: std.mem.Allocator,
        value: tensor_mod.TensorOf(tensor_dtype),
        state: *GradState,

        fn destroy(ptr: *anyopaque) void {
            const payload: *@This() = @ptrCast(@alignCast(ptr));
            payload.value.deinit();
            payload.state.deinit();
            payload.allocator.destroy(payload);
        }
    };
}

/// `finishOp` for a differentiable op whose RESULT is 16-bit (today: the
/// f32 → f16/bf16 cast). Same contract as `finishOp`: consumes `value` on
/// success; under an active exec scope the result is a scope-owned borrow.
fn typedFinishOp(
    comptime tensor_dtype: DType,
    comptime result_tags: anytype,
    ctx: *ExecContext,
    value: tensor_mod.TensorOf(tensor_dtype),
    wants_grad: bool,
    comptime BackwardType: type,
    create_args: anytype,
) !Tensor(.{ .dtype = tensor_dtype, .tags = result_tags }) {
    const OutT = Tensor(.{ .dtype = tensor_dtype, .tags = result_tags });
    if (!wants_grad or !control.isGradEnabled()) {
        return OutT.fromTensor(ctx, value);
    }
    if (ctx.execScopeActive()) {
        try ctx.reserveScopeSlot();
        const payload = try ctx.allocator.create(TypedScopePayload(tensor_dtype));
        errdefer ctx.allocator.destroy(payload);
        const state = try core.createNode(BackwardType, create_args);
        payload.* = .{ .allocator = ctx.allocator, .value = value, .state = state };
        ctx.adoptScopeNodeAssumeCapacity(payload, TypedScopePayload(tensor_dtype).destroy);
        return .{ .value = value, .grad_state = state, .scope_owned = true };
    }
    const state = try core.createNode(BackwardType, create_args);
    return .{ .value = value, .grad_state = state };
}

/// Shared tail of the widened ops: narrow the f32 kernel result back to
/// `tensor_dtype` and wrap it as a typed constant.
fn typedFromWidened(comptime tensor_dtype: DType, comptime result_tags: anytype, ctx: *ExecContext, wide_value: *const RawTensor) !Tensor(.{ .dtype = tensor_dtype, .tags = result_tags }) {
    var value = try ctx.castTyped(.f32, tensor_dtype, wide_value);
    errdefer value.deinit();
    return Tensor(.{ .dtype = tensor_dtype, .tags = result_tags }).fromTensor(ctx, value);
}

fn typedConstantUnary(self: anytype, ctx: *ExecContext, comptime op: UnaryOp) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "unary");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.unary(op, &wide);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

fn TypedUnaryMethod(comptime op: UnaryOp) type {
    return struct {
        fn call(self: anytype, ctx: *ExecContext) !TensorObject(@TypeOf(self)) {
            return typedConstantUnary(self, ctx, op);
        }
    };
}

fn typedConstantLeakyRelu(self: anytype, ctx: *ExecContext, negative_slope: f32) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "leakyRelu");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.leakyRelu(&wide, negative_slope);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

fn typedConstantClamp(self: anytype, ctx: *ExecContext, min_value: f32, max_value: f32) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "clamp");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.clamp(&wide, min_value, max_value);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

fn typedConstantScale(self: anytype, ctx: *ExecContext, scalar_value: dtype_mod.Accumulator(TensorObject(@TypeOf(self)).dtype)) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    var value = try ctx.scaleTyped(Self.dtype, self.asRawTensor(), scalar_value);
    errdefer value.deinit();
    return Self.fromTensor(ctx, value);
}

fn typedConstantAddScalar(self: anytype, ctx: *ExecContext, scalar_value: f32) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "addScalar");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.addScalar(&wide, scalar_value);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

fn typedConstantSubScalar(self: anytype, ctx: *ExecContext, scalar_value: f32) !TensorObject(@TypeOf(self)) {
    return typedConstantAddScalar(self, ctx, -scalar_value);
}

fn typedConstantDivScalar(self: anytype, ctx: *ExecContext, scalar_value: dtype_mod.Accumulator(TensorObject(@TypeOf(self)).dtype)) !TensorObject(@TypeOf(self)) {
    return typedConstantScale(self, ctx, 1.0 / scalar_value);
}

fn typedConstantPowScalar(self: anytype, ctx: *ExecContext, exponent: f32) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "powScalar");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.powScalar(&wide, exponent);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

/// Widened binary pointwise (`maximum`/`minimum`): both operands widen,
/// the f32 tag-broadcast kernel runs, the result narrows.
fn typedWidenedPointwise(
    comptime op: PointwiseOp,
    self: anytype,
    ctx: *ExecContext,
    other: anytype,
) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    try typedRequireNoGrad(self);
    try typedRequireNoGrad(other);
    const Self = TensorObject(@TypeOf(self));
    const Other = TensorObject(@TypeOf(other));
    comptime requireWidenedTypedFloat(Self.dtype, "maximum/minimum");
    if (Other.dtype != Self.dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
    const result_tags = pointwiseResultTags(Self.axis_tags, Other.axis_tags);
    const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
    var wide_left = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide_left.deinit();
    var wide_right = try ctx.castTyped(Self.dtype, .f32, other_ptr.asRawTensor());
    defer wide_right.deinit();
    var wide_value = try tag_ops.pointwise(op, Self.axis_tags, &wide_left, ctx, Other.axis_tags, &wide_right);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, result_tags, ctx, &wide_value);
}

fn typedConstantMaximum(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    const Self = TensorObject(@TypeOf(self));
    if (comptime dtype_mod.supportsIntMath(Self.dtype)) return typedPointwise(Self.dtype, .max, self, ctx, other);
    return typedWidenedPointwise(.max, self, ctx, other);
}

fn typedConstantMinimum(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    const Self = TensorObject(@TypeOf(self));
    if (comptime dtype_mod.supportsIntMath(Self.dtype)) return typedPointwise(Self.dtype, .min, self, ctx, other);
    return typedWidenedPointwise(.min, self, ctx, other);
}

/// Explicit integer division with the standard tag-broadcast rule:
/// `.trunc` rounds toward zero, `.floor` toward negative infinity; a zero
/// divisor is `error.DivisionByZero`; minInt/-1 wraps (the +% contract).
fn typedIntDiv(
    comptime mode: enum { trunc, floor },
    self: anytype,
    ctx: *ExecContext,
    other: anytype,
) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    const Self = TensorObject(@TypeOf(self));
    const Other = TensorObject(@TypeOf(other));
    comptime {
        if (!dtype_mod.supportsIntMath(Self.dtype)) @compileError("divTrunc/divFloor are integer ops; floats use div");
    }
    if (Other.dtype != Self.dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
    const left_tags = Self.axis_tags;
    const right_tags = Other.axis_tags;
    const left = tensorObjectPtrFrom(@TypeOf(self), &self);
    const right = tensorObjectPtrFrom(@TypeOf(other), &other);
    const result_tags = pointwiseResultTags(left_tags, right_tags);
    const result_shape = try pointwiseShapeOf(Self.dtype, result_tags, left_tags, left.asRawTensor(), right_tags, right.asRawTensor());

    var left_view = try broadcastTensorToOf(Self.dtype, left_tags, left.asRawTensor(), result_tags, result_shape);
    defer left_view.deinit();
    var right_view = try broadcastTensorToOf(Self.dtype, right_tags, right.asRawTensor(), result_tags, result_shape);
    defer right_view.deinit();

    var value = switch (mode) {
        .trunc => try ctx.divTruncRankTyped(Self.dtype, rawRank(result_tags.len), &left_view, &right_view),
        .floor => try ctx.divFloorRankTyped(Self.dtype, rawRank(result_tags.len), &left_view, &right_view),
    };
    errdefer value.deinit();
    return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
}

fn typedConstantDivTrunc(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    return typedIntDiv(.trunc, self, ctx, other);
}

fn typedConstantDivFloor(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    return typedIntDiv(.floor, self, ctx, other);
}

/// Logical ops on the `.bool` branch (the mask combinators): `.bool`
/// output; `other` may be `.bool` or float (truthiness).
fn typedLogicalBinary(comptime op: exec_mod.LogicalOp, self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
    const Self = TensorObject(@TypeOf(self));
    comptime {
        if (Self.dtype != .bool) @compileError("logical ops on the typed branch are .bool-only; cast explicitly");
    }
    const Other = TensorObject(@TypeOf(other));
    comptime {
        if (Other.dtype != .bool and !dtype_mod.supportsForwardFloatMath(Other.dtype))
            @compileError("logical ops take .bool or float operands; cast integer masks explicitly");
    }
    const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
    var value = try ctx.logicalTyped(op, .bool, Other.dtype, self.asRawTensor(), other_ptr.asRawTensor());
    errdefer value.deinit();
    return Tensor(.{ .dtype = .bool, .tags = Self.axis_tags }).fromTensor(ctx, value);
}

fn typedConstantLogicalAnd(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
    return typedLogicalBinary(.l_and, self, ctx, other);
}

fn typedConstantLogicalOr(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
    return typedLogicalBinary(.l_or, self, ctx, other);
}

fn typedConstantLogicalXor(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
    return typedLogicalBinary(.l_xor, self, ctx, other);
}

fn typedConstantLogicalNot(self: anytype, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
    const Self = TensorObject(@TypeOf(self));
    comptime {
        if (Self.dtype != .bool) @compileError("logical ops on the typed branch are .bool-only; cast explicitly");
    }
    var value = try ctx.logicalNotTyped(.bool, self.asRawTensor());
    errdefer value.deinit();
    return Tensor(.{ .dtype = .bool, .tags = Self.axis_tags }).fromTensor(ctx, value);
}

fn typedConstantGated(
    self: anytype,
    ctx: *ExecContext,
    other: anytype,
    comptime op: GatedOp,
) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    try typedRequireNoGrad(self);
    try typedRequireNoGrad(other);
    const Self = TensorObject(@TypeOf(self));
    const Other = TensorObject(@TypeOf(other));
    comptime requireWidenedTypedFloat(Self.dtype, "gated");
    if (Other.dtype != Self.dtype) @compileError("typed gated requires matching dtypes; cast explicitly");
    const result_tags = pointwiseResultTags(Self.axis_tags, Other.axis_tags);
    const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
    var wide_left = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide_left.deinit();
    var wide_right = try ctx.castTyped(Self.dtype, .f32, other_ptr.asRawTensor());
    defer wide_right.deinit();
    var wide_value = try tag_ops.gatedPointwise(op, Self.axis_tags, &wide_left, ctx, Other.axis_tags, &wide_right);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, result_tags, ctx, &wide_value);
}

fn typedConstantGlu(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    return typedConstantGated(self, ctx, other, .glu);
}

fn typedConstantSwiglu(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    return typedConstantGated(self, ctx, other, .swiglu);
}

fn typedConstantGeglu(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    return typedConstantGated(self, ctx, other, .geglu);
}

fn typedConstantSoftmax(self: anytype, ctx: *ExecContext, comptime tag: Tag, options: anytype) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "softmax");
    comptime {
        const Options = @TypeOf(options);
        if (@typeInfo(Options) != .@"struct" or @typeInfo(Options).@"struct".fields.len != 0) {
            @compileError("typed softmax supports only plain .{} options; cast to f32 for the ext path (mask/sinks/causal/scale)");
        }
    }
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.softmaxAxisRank(Self.tag_count, &wide, Self.axis(tag));
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

fn typedConstantLogSoftmax(self: anytype, ctx: *ExecContext, comptime tag: Tag) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "logSoftmax");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.logSoftmaxAxisRank(Self.tag_count, &wide, Self.axis(tag));
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

fn typedConstantRmsNorm(self: anytype, ctx: *ExecContext, comptime tag: Tag, eps: f32) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "rmsNorm");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.rmsNormAxisRank(Self.tag_count, &wide, Self.axis(tag), eps);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

fn typedConstantRmsNormMul(
    self: anytype,
    ctx: *ExecContext,
    comptime tag: Tag,
    weight: *const Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = .{tag} }),
    eps: f32,
) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    try typedRequireNoGrad(weight);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "rmsNormMul");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_weight = try ctx.castTyped(Self.dtype, .f32, weight.asRawTensor());
    defer wide_weight.deinit();
    var wide_value = try ctx.rmsNormMulAxisRank(Self.tag_count, &wide, &wide_weight, Self.axis(tag), eps);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

fn typedConstantLayerNorm(self: anytype, ctx: *ExecContext, comptime tag: Tag, eps: f32, options: anytype) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "layerNorm");
    comptime {
        const Options = @TypeOf(options);
        if (@typeInfo(Options) != .@"struct" or @typeInfo(Options).@"struct".fields.len != 0) {
            @compileError("typed layerNorm supports only plain .{} options; cast to f32 for the affine path");
        }
    }
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.layerNormAxisRank(Self.tag_count, &wide, Self.axis(tag), eps);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

// Widened reductions return f32 like the native typed sum/mean (§8.3:
// reductions on 16-bit floats keep the accumulator dtype).
fn typedConstantLogsumexp(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "logsumexp");
    const result_tags = removeTag(Self.axis_tags, tag);
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var value = try ctx.logsumexpAxisRank(Self.tag_count, &wide, Self.axis(tag));
    errdefer value.deinit();
    return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
}

fn typedConstantMax(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
    return typedConstantExtremum(self, ctx, tag, .max);
}

fn typedConstantMin(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
    return typedConstantExtremum(self, ctx, tag, .min);
}

fn typedConstantExtremum(self: anytype, ctx: *ExecContext, comptime tag: Tag, comptime op: enum { max, min }) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "max/min");
    const result_tags = removeTag(Self.axis_tags, tag);
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var raw = switch (op) {
        .max => try ctx.maxAxisRank(Self.tag_count, &wide, Self.axis(tag)),
        .min => try ctx.minAxisRank(Self.tag_count, &wide, Self.axis(tag)),
    };
    raw.indices.deinit();
    errdefer raw.values.deinit();
    return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, raw.values);
}

fn typedConstantArgmax(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .i64, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "argmax");
    const result_tags = removeTag(Self.axis_tags, tag);
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var value = try ctx.argmaxAxisRank(Self.tag_count, &wide, Self.axis(tag));
    errdefer value.deinit();
    return Tensor(.{ .dtype = .i64, .tags = result_tags }).fromTensor(ctx, value);
}

fn typedConstantProd(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "prod");
    const result_tags = removeTag(Self.axis_tags, tag);
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var value = try ctx.prodAxisRank(Self.tag_count, &wide, Self.axis(tag));
    errdefer value.deinit();
    return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
}

fn typedConstantVariance(self: anytype, ctx: *ExecContext, comptime tag: Tag, ddof: u1) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "variance");
    const result_tags = removeTag(Self.axis_tags, tag);
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var value = try ctx.varAxisRank(Self.tag_count, &wide, Self.axis(tag), ddof);
    errdefer value.deinit();
    return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
}

fn typedConstantCumsum(self: anytype, ctx: *ExecContext, comptime tag: Tag) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "cumsum");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.cumsumAxisRank(Self.tag_count, &wide, Self.axis(tag));
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

fn typedConstantCumprod(self: anytype, ctx: *ExecContext, comptime tag: Tag) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "cumprod");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.cumprodAxisRank(Self.tag_count, &wide, Self.axis(tag));
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

fn typedConstantWhere(self: anytype, ctx: *ExecContext, cond: anytype, other: anytype) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    try typedRequireNoGrad(other);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "where");
    const Cond = TensorObject(@TypeOf(cond));
    const Other = TensorObject(@TypeOf(other));
    if (comptime Cond.dtype != .bool and Cond.dtype != Self.dtype) @compileError("typed where takes a .bool or same-dtype condition; cast explicitly");
    if (comptime Other.dtype != Self.dtype) @compileError("typed where requires matching dtypes; cast explicitly");
    const cond_ptr = tensorObjectPtrFrom(@TypeOf(cond), &cond);
    const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_other = try ctx.castTyped(Self.dtype, .f32, other_ptr.asRawTensor());
    defer wide_other.deinit();
    var wide_value = try ctx.whereTyped(Cond.dtype, &wide, cond_ptr.asRawTensor(), &wide_other);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

fn typedConstantMaskedFill(self: anytype, ctx: *ExecContext, mask: anytype, value: f32) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "maskedFill");
    const Mask = TensorObject(@TypeOf(mask));
    if (comptime Mask.dtype != .bool and Mask.dtype != Self.dtype) @compileError("typed maskedFill takes a .bool or same-dtype mask; cast explicitly");
    const mask_ptr = tensorObjectPtrFrom(@TypeOf(mask), &mask);
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.maskedFillTyped(Mask.dtype, &wide, mask_ptr.asRawTensor(), value);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

/// Comparison on the typed branches: `.bool` result everywhere (torch's
/// comparison dtype). f16/bf16 compare through f32 (the widening seam);
/// integers compare natively (exact at any magnitude).
fn typedConstantCompare(self: anytype, ctx: *ExecContext, comptime op: exec_mod.CompareOp, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
    const Self = TensorObject(@TypeOf(self));
    const BoolT = Tensor(.{ .dtype = .bool, .tags = Self.axis_tags });
    const OtherT = @TypeOf(other);
    if (comptime dtype_mod.supportsIntMath(Self.dtype)) {
        if (comptime (OtherT == comptime_int or @typeInfo(OtherT) == .int)) {
            var value = try ctx.compareIntScalarTyped(Self.dtype, op, self.asRawTensor(), @intCast(other));
            errdefer value.deinit();
            return BoolT.fromTensor(ctx, value);
        }
        const Other = TensorObject(@TypeOf(other));
        if (comptime Other.dtype != Self.dtype) @compileError("typed compare requires matching dtypes; cast explicitly");
        const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
        var value = try ctx.compareIntTyped(Self.dtype, op, self.asRawTensor(), other_ptr.asRawTensor());
        errdefer value.deinit();
        return BoolT.fromTensor(ctx, value);
    }
    comptime requireWidenedTypedFloat(Self.dtype, "compare");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    if (comptime (OtherT == comptime_float or OtherT == comptime_int or @typeInfo(OtherT) == .float or @typeInfo(OtherT) == .int)) {
        var value = try ctx.compareScalar(op, &wide, other);
        errdefer value.deinit();
        return BoolT.fromTensor(ctx, value);
    }
    const Other = TensorObject(@TypeOf(other));
    if (comptime Other.dtype != Self.dtype) @compileError("typed compare requires matching dtypes; cast explicitly");
    const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
    var wide_other = try ctx.castTyped(Self.dtype, .f32, other_ptr.asRawTensor());
    defer wide_other.deinit();
    var value = try ctx.compare(op, &wide, &wide_other);
    errdefer value.deinit();
    return BoolT.fromTensor(ctx, value);
}

/// Widened einsum: both operands widen to f32 and the f32 GEMM lowering
/// runs (f32 accumulation); the result narrows to the input dtype per the
/// §8.3 matmul policy — the same contract as the typed `dot`.
fn typedConstantEinsum(self: anytype, ctx: *ExecContext, other: anytype, comptime out_tags: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(out_tags) }) {
    try typedRequireNoGrad(self);
    try typedRequireNoGrad(other);
    const Self = TensorObject(@TypeOf(self));
    const Other = TensorObject(@TypeOf(other));
    comptime requireWidenedTypedFloat(Self.dtype, "einsum");
    if (comptime Other.dtype != Self.dtype) @compileError("typed einsum requires matching dtypes; cast explicitly");
    const result_tags = comptime normalizeTags(out_tags);
    const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
    var wide_left = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide_left.deinit();
    var wide_right = try ctx.castTyped(Self.dtype, .f32, other_ptr.asRawTensor());
    defer wide_right.deinit();
    var wide_value = try tag_ops.taggedEinsum(Self.axis_tags, &wide_left, ctx, Other.axis_tags, &wide_right, result_tags);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, result_tags, ctx, &wide_value);
}

fn typedConstantPad(self: anytype, ctx: *ExecContext, comptime tag: Tag, before: usize, after: usize, fill: f32) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    comptime requireWidenedTypedFloat(Self.dtype, "pad");
    var wide = try ctx.castTyped(Self.dtype, .f32, self.asRawTensor());
    defer wide.deinit();
    var wide_value = try ctx.padAxisRank(Self.tag_count, &wide, Self.axis(tag), before, after, fill);
    defer wide_value.deinit();
    return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
}

// Typed structural ops: pure views / data movement, valid for every typed
// float dtype (f64 included — nothing rounds through f32).
fn typedConstantSplit(
    self: anytype,
    ctx: *ExecContext,
    comptime tag: Tag,
    comptime split_tags_spec: anytype,
    split_shape: [normalizeTags(split_tags_spec).len]usize,
) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = splitTags(TensorObject(@TypeOf(self)).axis_tags, tag, normalizeTags(split_tags_spec)) }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    const split_tags = normalizeTags(split_tags_spec);
    const result_tags = splitTags(Self.axis_tags, tag, split_tags);
    var value = try tag_ops.splitAxisViewOf(Self.dtype, Self.axis_tags, self.asRawTensor(), tag, split_tags, split_shape);
    errdefer value.deinit();
    return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
}

fn typedConstantMerge(self: anytype, ctx: *ExecContext, comptime out_tag: Tag, comptime merge_tags_spec: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = mergeTags(TensorObject(@TypeOf(self)).axis_tags, out_tag, normalizeTags(merge_tags_spec)) }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    const merge_tags = normalizeTags(merge_tags_spec);
    const result_tags = mergeTags(Self.axis_tags, out_tag, merge_tags);
    var value = try tag_ops.mergeAxesViewOf(Self.dtype, Self.axis_tags, self.asRawTensor(), out_tag, merge_tags);
    errdefer value.deinit();
    return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
}

fn typedConstantFlatten(self: anytype, ctx: *ExecContext, comptime out_tag: Tag) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = .{out_tag} }) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    var value = try tag_ops.flattenTensorOf(Self.dtype, ctx, self.asRawTensor());
    errdefer value.deinit();
    return Tensor(.{ .dtype = Self.dtype, .tags = .{out_tag} }).fromTensor(ctx, value);
}

fn typedConstantReshape(
    self: anytype,
    ctx: *ExecContext,
    comptime new_tags_spec: anytype,
    new_shape: [normalizeTags(new_tags_spec).len]usize,
) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(new_tags_spec) }) {
    const new_tags = comptime normalizeTags(new_tags_spec);
    if (comptime new_tags.len == 1) {
        if (self.asRawTensor().len() != new_shape[0]) return TensorError.InvalidShape;
        return typedConstantFlatten(self, ctx, new_tags[0]);
    }
    var flat = try typedConstantFlatten(self, ctx, new_tags[0]);
    defer flat.deinit();
    return typedConstantSplit(&flat, ctx, new_tags[0], new_tags_spec, new_shape);
}

fn typedConstantSliceStep(self: anytype, ctx: *ExecContext, comptime tag: Tag, start: usize, length: usize, step: usize) !TensorObject(@TypeOf(self)) {
    try typedRequireNoGrad(self);
    const Self = TensorObject(@TypeOf(self));
    const slice_axis = comptime Self.axis(tag);
    const raw = self.asRawTensor();
    const axis_dim = raw.shape.at(slice_axis);
    if (step == 0 or length == 0) return TensorError.InvalidShape;
    if (start >= axis_dim or start + (length - 1) * step >= axis_dim) return TensorError.InvalidShape;
    var new_shape: [Self.tensor_rank]usize = undefined;
    var new_strides: [Self.tensor_rank]usize = undefined;
    inline for (0..Self.tensor_rank) |i| {
        new_shape[i] = raw.shape.at(i);
        new_strides[i] = raw.strides.at(i);
    }
    new_shape[slice_axis] = length;
    new_strides[slice_axis] = raw.strides.at(slice_axis) * step;
    var value = try raw.viewWithStridesOffset(&new_shape, &new_strides, start * raw.strides.at(slice_axis));
    errdefer value.deinit();
    return Self.fromTensor(ctx, value);
}

fn typedConstantFlip(self: anytype, ctx: *ExecContext, comptime tag: Tag) !TensorObject(@TypeOf(self)) {
    const Self = TensorObject(@TypeOf(self));
    const n = self.asRawTensor().shape.at(Self.axis(tag));
    const indices = try ctx.allocator.alloc(usize, n);
    defer ctx.allocator.free(indices);
    for (indices, 0..) |*index, i| index.* = n - 1 - i;
    return typedConstantGather(self, ctx, tag, indices, tag);
}

fn typedConstantRoll(self: anytype, ctx: *ExecContext, comptime tag: Tag, shift: isize) !TensorObject(@TypeOf(self)) {
    const Self = TensorObject(@TypeOf(self));
    const n = self.asRawTensor().shape.at(Self.axis(tag));
    const indices = try ctx.allocator.alloc(usize, n);
    defer ctx.allocator.free(indices);
    // out[i] = x[(i - shift) mod n]; s = shift mod n in [0, n).
    const s: usize = @intCast(@mod(shift, @as(isize, @intCast(n))));
    for (indices, 0..) |*index, i| index.* = (i + n - s) % n;
    return typedConstantGather(self, ctx, tag, indices, tag);
}

fn typedConstantStack(
    self: anytype,
    ctx: *ExecContext,
    comptime new_tag: Tag,
    comptime axis_index: usize,
    others: []const *const TensorObject(@TypeOf(self)),
) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = insertTagAt(TensorObject(@TypeOf(self)).axis_tags, new_tag, axis_index) }) {
    const Self = TensorObject(@TypeOf(self));
    const Expanded = Tensor(.{ .dtype = Self.dtype, .tags = insertTagAt(Self.axis_tags, new_tag, axis_index) });
    var expanded = try ctx.allocator.alloc(Expanded, others.len + 1);
    defer ctx.allocator.free(expanded);
    var created: usize = 0;
    defer for (expanded[0..created]) |*view| view.deinit();

    expanded[0] = try typedConstantInsertAxis(self, ctx, new_tag, axis_index);
    created = 1;
    for (others) |other| {
        expanded[created] = try typedConstantInsertAxis(other, ctx, new_tag, axis_index);
        created += 1;
    }

    var ptrs_stack: [concat_inline_inputs]*const Expanded = undefined;
    const ptrs = if (others.len <= ptrs_stack.len)
        ptrs_stack[0..others.len]
    else
        try ctx.allocator.alloc(*const Expanded, others.len);
    defer if (others.len > ptrs_stack.len) ctx.allocator.free(ptrs);
    for (ptrs, expanded[1..]) |*ptr, *view| ptr.* = view;
    return typedConstantConcat(&expanded[0], ctx, new_tag, ptrs);
}

fn typedConstantRepeatAxis(self: anytype, ctx: *ExecContext, comptime tag: Tag, n: usize) !TensorObject(@TypeOf(self)) {
    const Self = TensorObject(@TypeOf(self));
    if (n == 0) return TensorError.InvalidShape;
    if (n == 1) return typedConstantWithTags(self, ctx, Self.axis_tags);
    const ptrs = try ctx.allocator.alloc(*const Self, n - 1);
    defer ctx.allocator.free(ptrs);
    const self_ptr = tensorObjectPtrFrom(@TypeOf(self), &self);
    for (ptrs) |*ptr| ptr.* = self_ptr;
    return typedConstantConcat(self_ptr, ctx, tag, ptrs);
}

pub fn variable(ctx: *ExecContext, comptime tags_spec: anytype, value: RawTensor) !Tensor(tags_spec) {
    return Tensor(tags_spec).variable(ctx, value);
}

pub fn constant(ctx: *ExecContext, comptime tags_spec: anytype, value: RawTensor) !Tensor(tags_spec) {
    return Tensor(tags_spec).constant(ctx, value);
}

fn pointwise(comptime op: PointwiseOp, self: anytype, ctx: *ExecContext, other: anytype) !Tensor(pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags)) {
    const SelfTensor = TensorObject(@TypeOf(self));
    const left_tags = SelfTensor.axis_tags;
    const Other = TensorObject(@TypeOf(other));
    const right_tags = Other.axis_tags;
    const left = tensorObjectPtrFrom(@TypeOf(self), &self);
    const right = tensorObjectPtrFrom(@TypeOf(other), &other);
    const result_tags = pointwiseResultTags(left_tags, right_tags);
    const left_tensor = left.asRawTensor();
    const right_tensor = right.asRawTensor();
    _ = try pointwiseShape(result_tags, left_tags, left_tensor, right_tags, right_tensor);

    if (comptime tagsEqual(left_tags, right_tags)) {
        if (std.mem.eql(usize, left_tensor.shape.slice(), right_tensor.shape.slice())) {
            var value = switch (op) {
                .add => try ctx.addRank(rawRank(result_tags.len), left_tensor, right_tensor),
                .sub => try ctx.subRank(rawRank(result_tags.len), left_tensor, right_tensor),
                .mul => try ctx.mulRank(rawRank(result_tags.len), left_tensor, right_tensor),
                .div => try ctx.divRank(rawRank(result_tags.len), left_tensor, right_tensor),
                .max => try ctx.maxRank(rawRank(result_tags.len), left_tensor, right_tensor),
                .min => try ctx.minRank(rawRank(result_tags.len), left_tensor, right_tensor),
            };
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, left.requiresGrad() or right.requiresGrad(), PointwiseBackward(op, left_tags, right_tags, result_tags), .{ ctx.allocator, left.grad_state, right.grad_state, left_tensor, right_tensor });
        }
    }

    var value = try tag_ops.pointwise(op, left_tags, left_tensor, ctx, right_tags, right_tensor);
    errdefer value.deinit();
    return finishOp(result_tags, ctx, value, left.requiresGrad() or right.requiresGrad(), PointwiseBackward(op, left_tags, right_tags, result_tags), .{ ctx.allocator, left.grad_state, right.grad_state, left_tensor, right_tensor });
}

fn gatedPointwise(comptime op: GatedOp, self: anytype, ctx: *ExecContext, other: anytype) !Tensor(pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags)) {
    const SelfTensor = TensorObject(@TypeOf(self));
    const left_tags = SelfTensor.axis_tags;
    const Other = TensorObject(@TypeOf(other));
    const right_tags = Other.axis_tags;
    const left = tensorObjectPtrFrom(@TypeOf(self), &self);
    const right = tensorObjectPtrFrom(@TypeOf(other), &other);
    const result_tags = pointwiseResultTags(left_tags, right_tags);
    const left_tensor = left.asRawTensor();
    const right_tensor = right.asRawTensor();
    _ = try pointwiseShape(result_tags, left_tags, left_tensor, right_tags, right_tensor);

    if (comptime tagsEqual(left_tags, right_tags)) {
        if (std.mem.eql(usize, left_tensor.shape.slice(), right_tensor.shape.slice())) {
            var value = try ctx.gatedRank(rawRank(result_tags.len), op, left_tensor, right_tensor);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, left.requiresGrad() or right.requiresGrad(), GatedBackward(op, left_tags, right_tags, result_tags), .{ ctx.allocator, left.grad_state, right.grad_state, left_tensor, right_tensor, &value });
        }
    }

    var value = try tag_ops.gatedPointwise(op, left_tags, left_tensor, ctx, right_tags, right_tensor);
    errdefer value.deinit();
    return finishOp(result_tags, ctx, value, left.requiresGrad() or right.requiresGrad(), GatedBackward(op, left_tags, right_tags, result_tags), .{ ctx.allocator, left.grad_state, right.grad_state, left_tensor, right_tensor, &value });
}

fn typedPointwise(
    comptime tensor_dtype: DType,
    comptime op: PointwiseOp,
    self: anytype,
    ctx: *ExecContext,
    other: anytype,
) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, tensor_dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
    try typedRequireNoGrad(self);
    try typedRequireNoGrad(other);
    const SelfTensor = TensorObject(@TypeOf(self));
    const left_tags = SelfTensor.axis_tags;
    const Other = TensorObject(@TypeOf(other));
    if (Other.dtype != tensor_dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
    if (comptime tensor_dtype == .bool) @compileError("bool tensors have no pointwise arithmetic; cast with to() first");
    const right_tags = Other.axis_tags;
    const left = tensorObjectPtrFrom(@TypeOf(self), &self);
    const right = tensorObjectPtrFrom(@TypeOf(other), &other);
    const result_tags = pointwiseResultTags(left_tags, right_tags);
    const result_shape = try pointwiseShapeOf(tensor_dtype, result_tags, left_tags, left.asRawTensor(), right_tags, right.asRawTensor());

    var left_view = try broadcastTensorToOf(tensor_dtype, left_tags, left.asRawTensor(), result_tags, result_shape);
    defer left_view.deinit();
    var right_view = try broadcastTensorToOf(tensor_dtype, right_tags, right.asRawTensor(), result_tags, result_shape);
    defer right_view.deinit();

    var value = switch (op) {
        .add => try ctx.addRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view),
        .sub => try ctx.subRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view),
        .mul => try ctx.mulRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view),
        .div => if (comptime dtype_mod.supportsIntMath(tensor_dtype))
            @compileError("integer `div` is explicit: use divTrunc/divFloor (torch's `/` promotes to float; Fucina keeps promotion explicit)")
        else
            try ctx.divRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view),
        .max => if (comptime dtype_mod.supportsIntMath(tensor_dtype))
            try ctx.maxRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view)
        else
            @compileError("float typed maximum/minimum widen through f32 (the f16/bf16 facade entries)"),
        .min => if (comptime dtype_mod.supportsIntMath(tensor_dtype))
            try ctx.minRankTyped(tensor_dtype, rawRank(result_tags.len), &left_view, &right_view)
        else
            @compileError("float typed maximum/minimum widen through f32 (the f16/bf16 facade entries)"),
    };
    errdefer value.deinit();
    return Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, tensor_dtype), .tags = result_tags }).fromTensor(ctx, value);
}

fn typedDot(
    comptime tensor_dtype: DType,
    self: anytype,
    ctx: *ExecContext,
    other: anytype,
    comptime contract_tag: Tag,
) !Tensor(.{ .dtype = dtype_mod.outputDType(.matmul, tensor_dtype), .tags = dotResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag) }) {
    try typedRequireNoGrad(self);
    try typedRequireNoGrad(other);
    const SelfTensor = TensorObject(@TypeOf(self));
    const left_tags = SelfTensor.axis_tags;
    const Other = TensorObject(@TypeOf(other));
    if (Other.dtype != tensor_dtype) @compileError("typed dot requires matching dtypes; cast explicitly");
    const right_tags = Other.axis_tags;
    const result_tags = dotResultTags(left_tags, right_tags, contract_tag);
    const left = tensorObjectPtrFrom(@TypeOf(self), &self);
    const right = tensorObjectPtrFrom(@TypeOf(other), &other);

    var value = try typedDotRaw(tensor_dtype, left_tags, left.asRawTensor(), ctx, right_tags, right.asRawTensor(), contract_tag);
    errdefer value.deinit();
    return Tensor(.{ .dtype = dtype_mod.outputDType(.matmul, tensor_dtype), .tags = result_tags }).fromTensor(ctx, value);
}

/// Per-position {max, sum_exp} buffer for the stats-saving forwards
/// (cross-entropy, grouped attention): allocated exactly when `finishOp`
/// will create the backward node (wants_grad AND grad mode on — keep the
/// conditions in sync), so the node always receives real statistics. The
/// caller frees it; the node dupes.
fn rowStatsAlloc(ctx: *ExecContext, wants_grad: bool, position_count: usize) !?[]f32 {
    if (!wants_grad or !control.isGradEnabled()) return null;
    return try ctx.allocator.alloc(f32, 2 * position_count);
}

/// Shared tail of every differentiable op: wrap `value` as a no-grad tensor
/// when no operand needs gradients, otherwise attach the backward node built
/// by `core.createNode(BackwardType, create_args)` — one allocation holding
/// the GradState header and the typed record. On error, ownership of `value`
/// stays with the caller (same contract as `fromTensor` and
/// `finishWithBackward`).
///
/// While an exec scope is open on `ctx` (ExecContext.openExecScope), the result
/// is adopted by the scope and the caller receives a borrow. The scope slot
/// is reserved BEFORE construction so adoption itself cannot fail after the
/// value has been consumed.
fn finishOp(
    comptime result_tags: anytype,
    ctx: *ExecContext,
    value: RawTensor,
    wants_grad: bool,
    comptime BackwardType: type,
    create_args: anytype,
) !Tensor(result_tags) {
    if (!wants_grad or !control.isGradEnabled()) return finishNoGrad(result_tags, ctx, value);
    if (ctx.execScopeActive()) try ctx.reserveScopeSlot();
    const state = try core.createNode(BackwardType, create_args);
    var out = try finishWithBackward(result_tags, value, state);
    if (ctx.execScopeActive()) {
        adoptIntoScope(ctx, &out);
        out.scope_owned = true;
    }
    return out;
}

/// Guard for facade-level COMPOSED differentiable ops (nllLoss, l2Normalize,
/// cosineSimilarity): their intermediate graph nodes are function-local, so
/// when gradients are tracked only an active exec scope can own them until
/// backward (GradState is single-owner — unscoped deinit of a grad-carrying
/// intermediate would dangle the downstream operand pointers). Loud error
/// instead of undefined behavior; no-grad composition works unscoped.
fn requireScopeForComposedGrad(ctx: *ExecContext, wants_grad: bool) !void {
    if (wants_grad and control.isGradEnabled() and !ctx.execScopeActive()) {
        return error.ActiveExecScopeRequired;
    }
}

/// N-ary einsum: contracts two or more f32 tensors (values or pointers, in a
/// tuple) down to `out_tags` by a comptime left-fold of the binary `einsum`.
/// Each intermediate keeps exactly the tags still needed by the remaining
/// operands or the output, in group-nested order, so classic chains (e.g. a
/// LoRA delta `x[s,i]·A[r,i]·B[o,r] -> [s,o]`) stay on the direct GEMM paths.
/// Contraction order is the operand order — order the tuple so early
/// intermediates stay small. Gradients flow through every operand; as with
/// other composed facade ops, tracking gradients requires an active exec
/// scope to own the intermediates (`error.ActiveExecScopeRequired`).
pub fn einsumMany(ctx: *ExecContext, comptime out_tags: anytype, operands: anytype) !Tensor(normalizeTags(out_tags)) {
    const OperandsT = @TypeOf(operands);
    const operand_count = comptime @typeInfo(OperandsT).@"struct".fields.len;
    comptime {
        if (operand_count < 2) @compileError("einsumMany requires at least two operands");
    }
    var wants_grad = false;
    inline for (0..operand_count) |i| {
        const ptr = tensorObjectPtrFrom(@TypeOf(operands[i]), &operands[i]);
        if (ptr.requiresGrad()) wants_grad = true;
    }
    try requireScopeForComposedGrad(ctx, wants_grad);
    const first = tensorObjectPtrFrom(@TypeOf(operands[0]), &operands[0]);
    return einsumManyFold(ctx, out_tags, first, operands, 1);
}

fn einsumManyFold(ctx: *ExecContext, comptime out_tags: anytype, acc: anytype, operands: anytype, comptime i: usize) !Tensor(normalizeTags(out_tags)) {
    const operand_count = comptime @typeInfo(@TypeOf(operands)).@"struct".fields.len;
    if (comptime i == operand_count - 1) {
        return acc.einsum(ctx, operands[i], comptime normalizeTags(out_tags));
    } else {
        const acc_tags = comptime TensorObject(@TypeOf(acc)).axis_tags;
        const op_tags = comptime TensorObject(@TypeOf(operands[i])).axis_tags;
        const needed = comptime einsumManyNeededTags(@TypeOf(operands), out_tags, i + 1);
        const keep = comptime einsumManyKeepTags(acc_tags, op_tags, needed);
        var next = try acc.einsum(ctx, operands[i], keep);
        defer next.deinit();
        return einsumManyFold(ctx, out_tags, &next, operands, i + 1);
    }
}

/// Tags an intermediate must keep: every tag of the pair still needed by the
/// remaining operands or the output, in group-nested (batch, then acc-free,
/// then operand-free) order so the next contraction stays direct-lowerable.
fn einsumManyKeepTags(comptime acc_tags: anytype, comptime op_tags: anytype, comptime needed: anytype) [einsumManyKeepLen(acc_tags, op_tags, needed)]Tag {
    const shared = tags_mod.intersectTags(acc_tags, op_tags);
    const op_shared = tags_mod.intersectTags(op_tags, acc_tags);
    return tags_mod.intersectTags(shared, needed) ++
        tags_mod.intersectTags(removeTags(acc_tags, shared), needed) ++
        tags_mod.intersectTags(removeTags(op_tags, op_shared), needed);
}

fn einsumManyKeepLen(comptime acc_tags: anytype, comptime op_tags: anytype, comptime needed: anytype) usize {
    const shared = tags_mod.intersectTags(acc_tags, op_tags);
    const op_shared = tags_mod.intersectTags(op_tags, acc_tags);
    return tags_mod.intersectTagsLen(shared, needed) +
        tags_mod.intersectTagsLen(removeTags(acc_tags, shared), needed) +
        tags_mod.intersectTagsLen(removeTags(op_tags, op_shared), needed);
}

/// Union of the output tags and every remaining operand's tags, used purely
/// as a membership set (may exceed the tensor rank limit, unlike
/// `pointwiseResultTags`).
fn einsumManyNeededTags(comptime OperandsT: type, comptime out_tags: anytype, comptime from: usize) [einsumManyNeededLen(OperandsT, out_tags, from)]Tag {
    const fields = @typeInfo(OperandsT).@"struct".fields;
    if (comptime from == fields.len) {
        return normalizeTags(out_tags);
    } else {
        const rest = einsumManyNeededTags(OperandsT, out_tags, from + 1);
        return tags_mod.unionTags(rest, TensorObject(fields[from].type).axis_tags);
    }
}

fn einsumManyNeededLen(comptime OperandsT: type, comptime out_tags: anytype, comptime from: usize) usize {
    const fields = @typeInfo(OperandsT).@"struct".fields;
    if (comptime from == fields.len) {
        return normalizeTags(out_tags).len;
    } else {
        const rest = einsumManyNeededTags(OperandsT, out_tags, from + 1);
        return tags_mod.unionTagsLen(rest, TensorObject(fields[from].type).axis_tags);
    }
}

/// Shared no-grad tail: wrap as a constant and, when an exec scope is open,
/// hand ownership to the scope. Same value-ownership contract as `fromTensor`.
fn finishNoGrad(comptime result_tags: anytype, ctx: *ExecContext, value: RawTensor) !Tensor(result_tags) {
    if (ctx.execScopeActive()) try ctx.reserveScopeSlot();
    var out = try Tensor(result_tags).fromTensor(ctx, value);
    if (ctx.execScopeActive()) {
        adoptIntoScope(ctx, &out);
        out.scope_owned = true;
    }
    return out;
}

fn adoptIntoScope(ctx: *ExecContext, t: anytype) void {
    ctx.adoptScopeValueAssumeCapacity(
        t.value,
        if (t.grad_state) |state| @ptrCast(state) else null,
        destroyGradStateOpaque,
    );
}

fn destroyGradStateOpaque(ptr: *anyopaque) void {
    const state: *GradState = @ptrCast(@alignCast(ptr));
    state.deinit();
}

/// Consumes `value` and `state` on success. On error, ownership of `value`
/// stays with the caller (every call site holds an `errdefer value.deinit()`),
/// while `state` — a co-allocated node the caller cannot reach — is destroyed
/// here.
fn finishWithBackward(comptime tags: anytype, value: RawTensor, state: *GradState) !Tensor(tags) {
    errdefer state.deinit();
    var owned_value = value;
    try validateTensorRank(normalizeTags(tags), &owned_value);
    return .{ .value = owned_value, .grad_state = state };
}

fn axisViewTensor(source: *const RawTensor, comptime axes: anytype, comptime target_tags: anytype) !RawTensor {
    return axisViewTensorOf(.f32, source, axes, target_tags);
}

fn axisViewTensorOf(
    comptime tensor_dtype: DType,
    source: *const tensor_mod.TensorOf(tensor_dtype),
    comptime axes: anytype,
    comptime target_tags: anytype,
) !tensor_mod.TensorOf(tensor_dtype) {
    if (comptime target_tags.len == 0) {
        return source.viewWithStrides(&.{1}, &.{1});
    }

    var shape: [target_tags.len]usize = undefined;
    var strides: [target_tags.len]usize = undefined;
    inline for (axes, 0..) |axis, i| {
        if (axis == inserted_axis) {
            shape[i] = 1;
            strides[i] = 0;
        } else {
            shape[i] = source.shape.at(axis);
            strides[i] = source.strides.at(axis);
        }
    }

    return source.viewWithStrides(shape[0..], strides[0..]);
}

fn typedDotRaw(
    comptime tensor_dtype: DType,
    comptime left_tags: anytype,
    left: *const tensor_mod.TensorOf(tensor_dtype),
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(tensor_dtype),
    comptime contract_tag: Tag,
) !tensor_mod.TensorOf(dtype_mod.outputDType(.matmul, tensor_dtype)) {
    const output_dtype = comptime dtype_mod.outputDType(.matmul, tensor_dtype);
    const result_tags = dotResultTags(left_tags, right_tags, contract_tag);
    const result_shape = try dotResultShapeOf(tensor_dtype, tensor_dtype, left_tags, left, right_tags, right, contract_tag);
    const batch_rank = comptime dotBatchLen(left_tags, right_tags, contract_tag);
    const left_free_rank = comptime dotLeftFreeLen(left_tags, right_tags, contract_tag);
    const right_free_rank = comptime dotRightFreeLen(left_tags, right_tags, contract_tag);

    const left_order = dotLeftOrder(left_tags, right_tags, contract_tag);
    var left_aligned = try alignTensorToOf(tensor_dtype, left_tags, left, left_order);
    defer left_aligned.deinit();

    const right_order = dotRightOrder(left_tags, right_tags, contract_tag);
    var right_aligned = try alignTensorToOf(tensor_dtype, right_tags, right, right_order);
    defer right_aligned.deinit();

    const m = productRangeOf(tensor_dtype, &left_aligned, batch_rank, left_free_rank);
    const k = left_aligned.shape.at(batch_rank + left_free_rank);
    const n = productRangeOf(tensor_dtype, &right_aligned, batch_rank + 1, right_free_rank);

    var left_ready = try contiguousForReshapeOf(tensor_dtype, ctx, &left_aligned);
    defer left_ready.deinit();
    var right_ready = try contiguousForReshapeOf(tensor_dtype, ctx, &right_aligned);
    defer right_ready.deinit();

    if (comptime batch_rank == 0 and left_free_rank == 0 and right_free_rank == 0) {
        var left_vector = try left_ready.reshape(&.{k});
        defer left_vector.deinit();
        var right_vector = try right_ready.reshape(&.{k});
        defer right_vector.deinit();
        return ctx.dotTyped(tensor_dtype, &left_vector, &right_vector);
    }

    if (comptime batch_rank != 0) {
        const num_batches = productRangeOf(tensor_dtype, &left_aligned, 0, batch_rank);
        var left_batched = try left_ready.reshape(&.{ num_batches, m, k });
        defer left_batched.deinit();
        var right_batched = try right_ready.reshape(&.{ num_batches, k, n });
        defer right_batched.deinit();

        var out = try ctx.emptyRankTyped(output_dtype, rawRank(result_tags.len), result_shape);
        errdefer out.deinit();

        const left_batch_len = m * k;
        const right_batch_len = k * n;
        const out_batch_len = m * n;
        for (0..num_batches) |batch| {
            var left_matrix = try left_batched.viewWithStridesOffset(&.{ m, k }, &.{ k, 1 }, batch * left_batch_len);
            defer left_matrix.deinit();
            var right_matrix = try right_batched.viewWithStridesOffset(&.{ k, n }, &.{ n, 1 }, batch * right_batch_len);
            defer right_matrix.deinit();
            var product = try ctx.matmul2DTyped(tensor_dtype, &left_matrix, &right_matrix);
            defer product.deinit();
            @memcpy(out.data()[batch * out_batch_len ..][0..out_batch_len], product.dataConst());
        }
        return out;
    }

    var left_matrix = try left_ready.reshape(&.{ m, k });
    defer left_matrix.deinit();
    var right_matrix = try right_ready.reshape(&.{ k, n });
    defer right_matrix.deinit();
    var matmul = try ctx.matmul2DTyped(tensor_dtype, &left_matrix, &right_matrix);
    errdefer matmul.deinit();

    if (std.mem.eql(usize, matmul.shape.slice(), result_shape[0..])) return matmul;
    const reshaped = try matmul.reshape(result_shape[0..]);
    matmul.deinit();
    return reshaped;
}

fn quantizedRhsDotRaw(
    comptime rhs_dtype: DType,
    comptime left_tags: anytype,
    left: *const RawTensor,
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(rhs_dtype),
    comptime contract_tag: Tag,
    allow_gpu: bool,
) !RawTensor {
    comptime if (!dtype_mod.isBlockQuantized(rhs_dtype)) @compileError("quantizedRhsDotRaw requires a block-quantized RHS dtype");
    comptime if (!dtype_mod.supportsQuantizedMatmulRhs(rhs_dtype)) @compileError("RHS dtype does not support quantized matmul");

    const result_shape = try dotResultShapeOf(.f32, rhs_dtype, left_tags, left, right_tags, right, contract_tag);
    const batch_rank = comptime dotBatchLen(left_tags, right_tags, contract_tag);
    if (comptime batch_rank != 0) @compileError("quantized RHS dot does not support shared batch tags yet");

    const left_free_rank = comptime dotLeftFreeLen(left_tags, right_tags, contract_tag);
    const right_free_rank = comptime dotRightFreeLen(left_tags, right_tags, contract_tag);
    if (comptime right_free_rank != 1) @compileError("quantized RHS dot requires one RHS free axis");

    const expected_right_order = dotRightTransBOrder(left_tags, right_tags, contract_tag);
    comptime if (!tagsEqual(right_tags, expected_right_order)) {
        @compileError("quantized RHS dot requires RHS storage order [free, contract], e.g. weight tags {.out, .in}");
    };

    const left_order = dotLeftOrder(left_tags, right_tags, contract_tag);
    var left_aligned = try alignTensorToOf(.f32, left_tags, left, left_order);
    defer left_aligned.deinit();

    const m = productRangeOf(.f32, &left_aligned, batch_rank, left_free_rank);
    const k = left_aligned.shape.at(batch_rank + left_free_rank);
    if (right.shape.at(1) != k) return TensorError.ShapeMismatch;

    var left_ready = try contiguousForReshapeOf(.f32, ctx, &left_aligned);
    defer left_ready.deinit();

    var left_matrix = try left_ready.reshape(&.{ m, k });
    defer left_matrix.deinit();
    var matmul = try ctx.matmul2DWithQuantizedTensorRhsOptions(rhs_dtype, &left_matrix, right, .{ .allow_gpu = allow_gpu });
    errdefer matmul.deinit();

    if (std.mem.eql(usize, matmul.shape.slice(), result_shape[0..])) return matmul;
    const reshaped = try matmul.reshape(result_shape[0..]);
    matmul.deinit();
    return reshaped;
}

fn f16RhsDotRaw(
    comptime left_tags: anytype,
    left: *const RawTensor,
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(.f16),
    comptime contract_tag: Tag,
) !RawTensor {
    const result_shape = try dotResultShapeOf(.f32, .f16, left_tags, left, right_tags, right, contract_tag);
    const batch_rank = comptime dotBatchLen(left_tags, right_tags, contract_tag);
    const left_free_rank = comptime dotLeftFreeLen(left_tags, right_tags, contract_tag);
    const right_free_rank = comptime dotRightFreeLen(left_tags, right_tags, contract_tag);

    const expected_right_order = dotRightTransBOrder(left_tags, right_tags, contract_tag);
    if (comptime batch_rank == 0 and right_free_rank == 1 and tagsEqual(right_tags, expected_right_order)) {
        const left_order = dotLeftOrder(left_tags, right_tags, contract_tag);
        var left_aligned = try alignTensorToOf(.f32, left_tags, left, left_order);
        defer left_aligned.deinit();

        const m = productRangeOf(.f32, &left_aligned, batch_rank, left_free_rank);
        const k = left_aligned.shape.at(batch_rank + left_free_rank);
        if (right.shape.at(1) != k) return TensorError.ShapeMismatch;

        var left_ready = try contiguousForReshapeOf(.f32, ctx, &left_aligned);
        defer left_ready.deinit();
        var right_ready = try contiguousForReshapeOf(.f16, ctx, right);
        defer right_ready.deinit();

        var left_matrix = try left_ready.reshape(&.{ m, k });
        defer left_matrix.deinit();
        var right_matrix = try right_ready.reshape(&.{ right.shape.at(0), k });
        defer right_matrix.deinit();

        var matmul = try ctx.matmulTransB2DWithF16Rhs(&left_matrix, &right_matrix);
        errdefer matmul.deinit();

        if (std.mem.eql(usize, matmul.shape.slice(), result_shape[0..])) return matmul;
        const reshaped = try matmul.reshape(result_shape[0..]);
        matmul.deinit();
        return reshaped;
    }

    var right_f32 = try ctx.castTyped(.f16, .f32, right);
    defer right_f32.deinit();
    return typedDotRaw(.f32, left_tags, left, ctx, right_tags, &right_f32, contract_tag);
}

fn bf16RhsDotRaw(
    comptime left_tags: anytype,
    left: *const RawTensor,
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const tensor_mod.TensorOf(.bf16),
    comptime contract_tag: Tag,
) !RawTensor {
    const result_shape = try dotResultShapeOf(.f32, .bf16, left_tags, left, right_tags, right, contract_tag);
    const batch_rank = comptime dotBatchLen(left_tags, right_tags, contract_tag);
    const left_free_rank = comptime dotLeftFreeLen(left_tags, right_tags, contract_tag);
    const right_free_rank = comptime dotRightFreeLen(left_tags, right_tags, contract_tag);

    const expected_right_order = dotRightTransBOrder(left_tags, right_tags, contract_tag);
    if (comptime batch_rank == 0 and right_free_rank == 1 and tagsEqual(right_tags, expected_right_order)) {
        const left_order = dotLeftOrder(left_tags, right_tags, contract_tag);
        var left_aligned = try alignTensorToOf(.f32, left_tags, left, left_order);
        defer left_aligned.deinit();

        const m = productRangeOf(.f32, &left_aligned, batch_rank, left_free_rank);
        const k = left_aligned.shape.at(batch_rank + left_free_rank);
        if (right.shape.at(1) != k) return TensorError.ShapeMismatch;

        var left_ready = try contiguousForReshapeOf(.f32, ctx, &left_aligned);
        defer left_ready.deinit();
        var right_ready = try contiguousForReshapeOf(.bf16, ctx, right);
        defer right_ready.deinit();

        var left_matrix = try left_ready.reshape(&.{ m, k });
        defer left_matrix.deinit();
        var right_matrix = try right_ready.reshape(&.{ right.shape.at(0), k });
        defer right_matrix.deinit();

        var matmul = try ctx.matmulTransB2DWithBf16Rhs(&left_matrix, &right_matrix);
        errdefer matmul.deinit();

        if (std.mem.eql(usize, matmul.shape.slice(), result_shape[0..])) return matmul;
        const reshaped = try matmul.reshape(result_shape[0..]);
        matmul.deinit();
        return reshaped;
    }

    var right_f32 = try ctx.castTyped(.bf16, .f32, right);
    defer right_f32.deinit();
    return typedDotRaw(.f32, left_tags, left, ctx, right_tags, &right_f32, contract_tag);
}

fn TensorObject(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |ptr| ptr.child,
        else => T,
    };
}

fn tensorObjectPtrFrom(comptime T: type, value: *const T) *const TensorObject(T) {
    return switch (@typeInfo(T)) {
        .pointer => value.*,
        else => value,
    };
}

/// Normalize `constantPad2d`'s padding spec: an integer pads all four
/// sides; a 4-tuple/array is `(left, right, top, bottom)`.
fn padding2dValues(padding: anytype) [4]isize {
    const P = @TypeOf(padding);
    const info = @typeInfo(P);
    if (comptime (P == comptime_int or info == .int)) {
        const p: isize = @intCast(padding);
        return .{ p, p, p, p };
    }
    if (comptime ((info == .@"struct" and info.@"struct".is_tuple and info.@"struct".fields.len == 4) or
        (info == .array and info.array.len == 4)))
    {
        return .{ @intCast(padding[0]), @intCast(padding[1]), @intCast(padding[2]), @intCast(padding[3]) };
    }
    @compileError("constantPad2d: padding must be an integer or a 4-tuple/array (left, right, top, bottom), got " ++ @typeName(P));
}

test {
    _ = @import("tensor_tests.zig");
}
