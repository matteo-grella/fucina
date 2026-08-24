//! The typed-constant math band: the constructors shared by the typed
//! float and typed scalar branches (`TypedConstantBase`) and the
//! `typedConstant*` op implementations those branches alias (forward math
//! over the stored dtype, the f16/bf16 widened family, integer and mask
//! math, casts). Views live in views.zig, accessors in common.zig.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const tag_ops = @import("../../tag_ops.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const tags_mod = @import("../../tags.zig");
const backward = @import("../backward.zig");
const rng = @import("../../rng.zig");

const RawTensor = tensor_mod.Tensor;
const DType = tensor_mod.DType;
const TensorError = tensor_mod.TensorError;
const Scalar = tensor_mod.Scalar;
const ExecContext = exec_mod.ExecContext;
const UnaryOp = exec_mod.UnaryOp;
const GatedOp = exec_mod.GatedOp;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const rawRank = tags_mod.rawRank;
const removeTag = tags_mod.removeTag;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const dotResultTags = tags_mod.dotResultTags;
const normalizeTags = tags_mod.normalizeTags;
const broadcastTensorTo = tag_ops.broadcastTensorTo;
const pointwiseShape = tag_ops.pointwiseShape;
const validateTensorRank = tag_ops.validateTensorRank;
const PointwiseOp = backward.PointwiseOp;
const CastBackward = backward.CastBackward;

pub fn Mod(comptime ag_tensor: type) type {
    return struct {
        const Tensor = ag_tensor.Tensor;

        const plumbing = @import("plumbing.zig").Mod(ag_tensor);
        const finishOp = plumbing.finishOp;
        const finishNoGrad = plumbing.finishNoGrad;
        const typedDotRaw = plumbing.typedDotRaw;
        const TensorObject = plumbing.TensorObject;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;

        /// Constructors shared by the typed float and typed scalar branches:
        /// the no-grad wrap, slice and borrowed-slice construction, fills,
        /// the i64 seed-stream constructors, and the `.bool` band mask.
        pub fn TypedConstantBase(comptime SelfT: type, comptime tags: anytype, comptime tensor_dtype: DType) type {
            const tensor_rank = rawRank(tags.len);
            const RawTypedTensor = tensor_mod.TensorOf(tensor_dtype);
            const Elem = Scalar(tensor_dtype);

            return struct {
                /// Consumes `value` on success; on error, ownership stays with the caller.
                pub fn constant(ctx: *ExecContext, value: RawTypedTensor) !SelfT {
                    _ = ctx;
                    var v = value;
                    try validateTensorRank(tensor_dtype, tags, &v);
                    return .{ .value = v };
                }

                pub fn fromTensor(ctx: *ExecContext, value: RawTypedTensor) !SelfT {
                    return try @This().constant(ctx, value);
                }

                pub fn fromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !SelfT {
                    var value = try ctx.fromSlice(tensor_dtype, raw_shape, values);
                    errdefer value.deinit();
                    return try @This().constant(ctx, value);
                }

                /// Zero-copy wrap caller-owned READ-ONLY typed storage as a no-grad
                /// constant tensor without `@constCast` at the call site. Read-only
                /// borrow: `values` must outlive the tensor and must not be mutated
                /// (see the f32 `fromBorrowedConstSlice` contract).
                pub fn fromBorrowedConstSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !SelfT {
                    var value = try ctx.fromBorrowedSlice(tensor_dtype, raw_shape, @constCast(values));
                    errdefer value.deinit();
                    return try @This().constant(ctx, value);
                }

                /// Allocate an uninitialized no-grad typed tensor of the tag-implied rank.
                pub fn empty(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !SelfT {
                    var value = try ctx.empty(tensor_dtype, raw_shape);
                    errdefer value.deinit();
                    return try @This().constant(ctx, value);
                }

                /// Allocate a zero-filled no-grad typed tensor.
                pub fn zeros(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !SelfT {
                    var value = try ctx.zeros(tensor_dtype, &raw_shape);
                    errdefer value.deinit();
                    return try @This().constant(ctx, value);
                }

                /// Allocate a one-filled no-grad typed tensor.
                pub fn ones(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !SelfT {
                    var value = try ctx.ones(tensor_dtype, &raw_shape);
                    errdefer value.deinit();
                    return try @This().constant(ctx, value);
                }

                /// No-grad tensor of uniform integer draws in `[low, high)`
                /// (torch.randint) from the deterministic counter-based stream at
                /// `seed` (see docs/reference/06-the-execution-runtime-execcontext-and-the-memory-model.md): element i is a pure function of `(seed, i)` via
                /// the widening multiply-shift map (`fucina.rng.randintFill`).
                /// i64-only (the repo-wide index dtype); cast with `to` for
                /// narrower integers. `low >= high` is `InvalidShape`.
                pub fn randint(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, low: i64, high: i64) !SelfT {
                    comptime if (tensor_dtype != .i64) @compileError("randint is i64-only (the repo-wide index dtype); cast the result with to()");
                    if (low >= high) return TensorError.InvalidShape;
                    var value = try ctx.empty(tensor_dtype, raw_shape);
                    errdefer value.deinit();
                    rng.randintFill(seed, value.data(), low, high);
                    return try @This().constant(ctx, value);
                }

                /// Rank-1 no-grad random permutation of `{0, ..., n-1}`
                /// (torch.randperm) as i64: Fisher-Yates driven by the
                /// counter-based stream at `seed` (`fucina.rng.randpermFill`;
                /// same seed, same permutation). `n == 0` is `InvalidShape`
                /// (zero-size tensors are not representable).
                pub fn randperm(ctx: *ExecContext, n: usize, seed: u64) !SelfT {
                    comptime {
                        if (tensor_dtype != .i64) @compileError("randperm is i64-only (the repo-wide index dtype)");
                        if (tags.len != 1) @compileError("randperm builds a rank-1 tensor; use a single-tag Tensor type");
                    }
                    if (n == 0) return TensorError.InvalidShape;
                    var value = try ctx.empty(tensor_dtype, .{n});
                    errdefer value.deinit();
                    rng.randpermFill(seed, value.data());
                    return try @This().constant(ctx, value);
                }

                /// Rank-2 no-grad band mask (the attention-mask constructor), on
                /// the `.bool` branch only: element `(i, j)` is true iff
                /// `i - j <= lower` and `j - i <= upper`; a null bound is
                /// unbounded on that side. Bounds are signed. Causal keep-set =
                /// `(null, 0)`; sliding window of width W = `(W - 1, 0)`; the
                /// tril(k) keep-set = `(null, k)`; triu(k) = `(-k, null)`. Feed
                /// it to `where`/`maskedFill` (broadcast with `broadcastTo` for
                /// batched scores), or cast with `to(.f32)` for the mask-multiply
                /// idiom. Contradictory bounds yield an all-false mask, not an
                /// error.
                pub fn bandMask(ctx: *ExecContext, raw_shape: [tensor_rank]usize, lower: ?i64, upper: ?i64) !SelfT {
                    comptime {
                        if (tensor_dtype != .bool) @compileError("bandMask is a .bool mask constructor; use a .bool Tensor type");
                        if (tags.len != 2) @compileError("bandMask builds a rank-2 [row, col] mask; use a two-tag Tensor type");
                    }
                    var value = try ctx.empty(tensor_dtype, raw_shape);
                    errdefer value.deinit();
                    const out = value.data();
                    const cols = raw_shape[1];
                    for (0..raw_shape[0]) |i| {
                        for (0..cols) |j| {
                            const d = @as(i64, @intCast(j)) - @as(i64, @intCast(i)); // j - i
                            const in_lower = if (lower) |l| -d <= l else true;
                            const in_upper = if (upper) |u| d <= u else true;
                            out[i * cols + j] = in_lower and in_upper;
                        }
                    }
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
            };
        }

        pub fn typedConstantTo(self: anytype, ctx: *ExecContext, comptime target_dtype: DType) !Tensor(.{ .dtype = target_dtype, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            const Self = TensorObject(@TypeOf(self));
            if (comptime target_dtype != .f32) {
                if (self.requiresGrad()) return error.GradientCastUnsupported;
            }
            var value = try ctx.cast(Self.dtype, target_dtype, self.asRawTensor());
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

        pub fn typedConstantAdd(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            return typedPointwise(Self.dtype, .add, self, ctx, other);
        }

        pub fn typedConstantSub(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            return typedPointwise(Self.dtype, .sub, self, ctx, other);
        }

        pub fn typedConstantMul(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            return typedPointwise(Self.dtype, .mul, self, ctx, other);
        }

        pub fn typedConstantDiv(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, TensorObject(@TypeOf(self)).dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            return typedPointwise(Self.dtype, .div, self, ctx, other);
        }

        pub fn typedConstantSum(self: anytype, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, TensorObject(@TypeOf(self)).dtype), .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            comptime if (@typeInfo(@TypeOf(opts)).@"struct".fields.len != 0)
                @compileError("typed constant reductions take no options (masked arms are f32-only); pass .{}");
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            const result_tags = removeTag(Self.axis_tags, tag);
            var value = try ctx.sumAxis(Self.dtype, Self.tag_count, self.asRawTensor(), Self.axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, Self.dtype), .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantMean(self: anytype, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, TensorObject(@TypeOf(self)).dtype), .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            comptime if (@typeInfo(@TypeOf(opts)).@"struct".fields.len != 0)
                @compileError("typed constant reductions take no options (masked arms are f32-only); pass .{}");
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            const result_tags = removeTag(Self.axis_tags, tag);
            var value = try ctx.meanAxis(Self.dtype, Self.tag_count, self.asRawTensor(), Self.axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, Self.dtype), .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantSumAll(self: anytype, ctx: *ExecContext) !Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, TensorObject(@TypeOf(self)).dtype), .tags = .{} }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            var value = try ctx.sum(Self.dtype, self.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, Self.dtype), .tags = .{} }).fromTensor(ctx, value);
        }

        pub fn typedConstantDot(self: anytype, ctx: *ExecContext, other: anytype, comptime contract_tag: Tag) !Tensor(.{ .dtype = dtype_mod.outputDType(.matmul, TensorObject(@TypeOf(self)).dtype), .tags = dotResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag) }) {
            const Self = TensorObject(@TypeOf(self));
            return typedDot(Self.dtype, self, ctx, other, contract_tag);
        }

        // Widened typed-float ops: forward coverage for ops with no native typed
        // kernel. The input widens to f32, the f32 exec kernel runs, and the result
        // narrows ONCE on store: f32 accumulation with a single final round, the
        // dtype policy in docs/reference/08-data-types-storage-and-the-raw-tensor-layer-internal.md. f64 is excluded at comptime: f64 math must stay f64, and
        // rounding it through f32 would silently lose precision.
        pub fn requireWidenedTypedFloat(comptime tensor_dtype: DType, comptime what: []const u8) void {
            if (tensor_dtype != .f16 and tensor_dtype != .bf16) {
                @compileError(what ++ " on the typed float branch is f16/bf16 only (it computes through f32; f64 must not round through f32; cast explicitly)");
            }
        }

        /// Typed forward ops are no-grad: a grad-requiring operand would silently
        /// drop its graph, so it is rejected instead (`to(.f32)` is the trained
        /// path; the differentiable typed entries are `to` and the mixed-RHS
        /// `dot`/`einsum`).
        pub fn typedRequireNoGrad(operand: anytype) !void {
            if (operand.requiresGrad()) return error.UnsupportedGradient;
        }

        /// Scope payload for a grad-carrying 16-bit result: the exec-scope slot
        /// holds f32 values only, so the typed value travels inside the type-erased
        /// node payload with a destructor that frees value + graph node together.
        pub fn TypedScopePayload(comptime tensor_dtype: DType) type {
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
        /// f32 -> f16/bf16 cast). Same contract as `finishOp`: consumes `value` on
        /// success; under an active exec scope the result is a scope-owned borrow.
        pub fn typedFinishOp(
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
        pub fn typedFromWidened(comptime tensor_dtype: DType, comptime result_tags: anytype, ctx: *ExecContext, wide_value: *const RawTensor) !Tensor(.{ .dtype = tensor_dtype, .tags = result_tags }) {
            var value = try ctx.cast(.f32, tensor_dtype, wide_value);
            errdefer value.deinit();
            return Tensor(.{ .dtype = tensor_dtype, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantUnary(self: anytype, ctx: *ExecContext, comptime op: UnaryOp) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "unary");
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.unary(op, &wide);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn TypedUnaryMethod(comptime op: UnaryOp) type {
            return struct {
                pub fn call(self: anytype, ctx: *ExecContext) !TensorObject(@TypeOf(self)) {
                    return typedConstantUnary(self, ctx, op);
                }
            };
        }

        pub fn typedConstantLeakyRelu(self: anytype, ctx: *ExecContext, negative_slope: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "leakyRelu");
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.leakyRelu(&wide, negative_slope);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantClamp(self: anytype, ctx: *ExecContext, min_value: f32, max_value: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "clamp");
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.clamp(&wide, min_value, max_value);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantScale(self: anytype, ctx: *ExecContext, scalar_value: dtype_mod.Accumulator(TensorObject(@TypeOf(self)).dtype)) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            var value = try ctx.scale(Self.dtype, self.asRawTensor(), scalar_value);
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn typedConstantAddScalar(self: anytype, ctx: *ExecContext, scalar_value: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "addScalar");
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.addScalar(&wide, scalar_value);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantSubScalar(self: anytype, ctx: *ExecContext, scalar_value: f32) !TensorObject(@TypeOf(self)) {
            return typedConstantAddScalar(self, ctx, -scalar_value);
        }

        pub fn typedConstantDivScalar(self: anytype, ctx: *ExecContext, scalar_value: dtype_mod.Accumulator(TensorObject(@TypeOf(self)).dtype)) !TensorObject(@TypeOf(self)) {
            return typedConstantScale(self, ctx, 1.0 / scalar_value);
        }

        pub fn typedConstantPowScalar(self: anytype, ctx: *ExecContext, exponent: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "powScalar");
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.powScalar(&wide, exponent);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        /// Widened binary pointwise (`maximum`/`minimum`): both operands widen,
        /// the f32 tag-broadcast kernel runs, the result narrows.
        pub fn typedWidenedPointwise(
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
            var wide_left = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide_left.deinit();
            var wide_right = try ctx.cast(Self.dtype, .f32, other_ptr.asRawTensor());
            defer wide_right.deinit();
            var wide_value = try tag_ops.pointwise(.f32, op, Self.axis_tags, &wide_left, ctx, Other.axis_tags, &wide_right);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, result_tags, ctx, &wide_value);
        }

        pub fn typedConstantMaximum(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            if (comptime dtype_mod.supportsIntMath(Self.dtype)) return typedPointwise(Self.dtype, .max, self, ctx, other);
            return typedWidenedPointwise(.max, self, ctx, other);
        }

        pub fn typedConstantMinimum(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            if (comptime dtype_mod.supportsIntMath(Self.dtype)) return typedPointwise(Self.dtype, .min, self, ctx, other);
            return typedWidenedPointwise(.min, self, ctx, other);
        }

        /// Explicit integer division with the standard tag-broadcast rule:
        /// `.trunc` rounds toward zero, `.floor` toward negative infinity; a zero
        /// divisor is `error.DivisionByZero`; minInt/-1 wraps (the +% contract).
        pub fn typedIntDiv(
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
            const result_shape = try pointwiseShape(Self.dtype, result_tags, left_tags, left.asRawTensor(), right_tags, right.asRawTensor());

            var left_view = try broadcastTensorTo(Self.dtype, left_tags, left.asRawTensor(), result_tags, result_shape);
            defer left_view.deinit();
            var right_view = try broadcastTensorTo(Self.dtype, right_tags, right.asRawTensor(), result_tags, result_shape);
            defer right_view.deinit();

            var value = switch (mode) {
                .trunc => try ctx.divTrunc(Self.dtype, rawRank(result_tags.len), &left_view, &right_view),
                .floor => try ctx.divFloor(Self.dtype, rawRank(result_tags.len), &left_view, &right_view),
            };
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantDivTrunc(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntDiv(.trunc, self, ctx, other);
        }

        pub fn typedConstantDivFloor(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntDiv(.floor, self, ctx, other);
        }

        /// Explicit integer remainder with the standard tag-broadcast rule:
        /// `rem` pairs `divTrunc` (sign of the dividend, C `%`), `mod` pairs
        /// `divFloor` (sign of the divisor, Python/numpy `%` and Zig's `@mod`).
        /// A zero divisor is `error.DivisionByZero`; minInt % -1 is 0.
        pub fn typedIntMod(
            comptime mode: enum { rem, mod },
            self: anytype,
            ctx: *ExecContext,
            other: anytype,
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (!dtype_mod.supportsIntMath(Self.dtype)) @compileError("rem/mod are integer ops; no float counterpart is defined");
            }
            if (Other.dtype != Self.dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
            const left_tags = Self.axis_tags;
            const right_tags = Other.axis_tags;
            const left = tensorObjectPtrFrom(@TypeOf(self), &self);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = pointwiseResultTags(left_tags, right_tags);
            const result_shape = try pointwiseShape(Self.dtype, result_tags, left_tags, left.asRawTensor(), right_tags, right.asRawTensor());

            var left_view = try broadcastTensorTo(Self.dtype, left_tags, left.asRawTensor(), result_tags, result_shape);
            defer left_view.deinit();
            var right_view = try broadcastTensorTo(Self.dtype, right_tags, right.asRawTensor(), result_tags, result_shape);
            defer right_view.deinit();

            var value = switch (mode) {
                .rem => try ctx.rem(Self.dtype, rawRank(result_tags.len), &left_view, &right_view),
                .mod => try ctx.mod(Self.dtype, rawRank(result_tags.len), &left_view, &right_view),
            };
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantRem(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntMod(.rem, self, ctx, other);
        }

        pub fn typedConstantMod(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntMod(.mod, self, ctx, other);
        }

        /// Bitwise combinators on two's-complement bit patterns, with the standard
        /// tag-broadcast rule. Integer dtypes only: `.bool` masks use the
        /// truthiness `logicalAnd/Or/Xor`, and floats have no bit-pattern algebra.
        pub fn typedIntBitwise(
            comptime op: exec_mod.IntBitwiseOp,
            self: anytype,
            ctx: *ExecContext,
            other: anytype,
        ) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            const Self = TensorObject(@TypeOf(self));
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (!dtype_mod.supportsIntMath(Self.dtype)) @compileError("bitAnd/bitOr/bitXor are integer ops; `.bool` masks use logicalAnd/Or/Xor");
            }
            if (Other.dtype != Self.dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
            const left_tags = Self.axis_tags;
            const right_tags = Other.axis_tags;
            const left = tensorObjectPtrFrom(@TypeOf(self), &self);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = pointwiseResultTags(left_tags, right_tags);
            const result_shape = try pointwiseShape(Self.dtype, result_tags, left_tags, left.asRawTensor(), right_tags, right.asRawTensor());

            var left_view = try broadcastTensorTo(Self.dtype, left_tags, left.asRawTensor(), result_tags, result_shape);
            defer left_view.deinit();
            var right_view = try broadcastTensorTo(Self.dtype, right_tags, right.asRawTensor(), result_tags, result_shape);
            defer right_view.deinit();

            var value = try ctx.bitwise(Self.dtype, rawRank(result_tags.len), op, &left_view, &right_view);
            errdefer value.deinit();
            return Tensor(.{ .dtype = Self.dtype, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantBitAnd(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntBitwise(.b_and, self, ctx, other);
        }

        pub fn typedConstantBitOr(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntBitwise(.b_or, self, ctx, other);
        }

        pub fn typedConstantBitXor(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedIntBitwise(.b_xor, self, ctx, other);
        }

        /// Logical ops on the `.bool` branch (the mask combinators): `.bool`
        /// output; `other` may be `.bool` or float (truthiness).
        pub fn typedLogicalBinary(comptime op: exec_mod.LogicalOp, self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
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
            var value = try ctx.logical(op, .bool, Other.dtype, self.asRawTensor(), other_ptr.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = .bool, .tags = Self.axis_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantLogicalAnd(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            return typedLogicalBinary(.l_and, self, ctx, other);
        }

        pub fn typedConstantLogicalOr(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            return typedLogicalBinary(.l_or, self, ctx, other);
        }

        pub fn typedConstantLogicalXor(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            return typedLogicalBinary(.l_xor, self, ctx, other);
        }

        pub fn typedConstantLogicalNot(self: anytype, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            const Self = TensorObject(@TypeOf(self));
            comptime {
                if (Self.dtype != .bool) @compileError("logical ops on the typed branch are .bool-only; cast explicitly");
            }
            var value = try ctx.logicalNot(.bool, self.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = .bool, .tags = Self.axis_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantGated(
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
            var wide_left = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide_left.deinit();
            var wide_right = try ctx.cast(Self.dtype, .f32, other_ptr.asRawTensor());
            defer wide_right.deinit();
            var wide_value = try tag_ops.gatedPointwise(op, Self.axis_tags, &wide_left, ctx, Other.axis_tags, &wide_right);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, result_tags, ctx, &wide_value);
        }

        pub fn typedConstantGlu(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedConstantGated(self, ctx, other, .glu);
        }

        pub fn typedConstantSwiglu(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedConstantGated(self, ctx, other, .swiglu);
        }

        pub fn typedConstantGeglu(self: anytype, ctx: *ExecContext, other: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            return typedConstantGated(self, ctx, other, .geglu);
        }

        pub fn typedConstantSoftmax(self: anytype, ctx: *ExecContext, comptime tag: Tag, options: anytype) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "softmax");
            comptime {
                const Options = @TypeOf(options);
                if (@typeInfo(Options) != .@"struct" or @typeInfo(Options).@"struct".fields.len != 0) {
                    @compileError("typed softmax supports only plain .{} options; cast to f32 for the ext path (mask/sinks/causal/scale)");
                }
            }
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.softmax(Self.tag_count, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantLogSoftmax(self: anytype, ctx: *ExecContext, comptime tag: Tag) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "logSoftmax");
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.logSoftmax(Self.tag_count, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantRmsNorm(self: anytype, ctx: *ExecContext, comptime tag: Tag, eps: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "rmsNorm");
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.rmsNorm(Self.tag_count, &wide, Self.axis(tag), eps);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantRmsNormMul(
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
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_weight = try ctx.cast(Self.dtype, .f32, weight.asRawTensor());
            defer wide_weight.deinit();
            var wide_value = try ctx.rmsNormMul(Self.tag_count, &wide, &wide_weight, Self.axis(tag), eps);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantLayerNorm(self: anytype, ctx: *ExecContext, comptime tag: Tag, eps: f32, options: anytype) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "layerNorm");
            comptime {
                const Options = @TypeOf(options);
                if (@typeInfo(Options) != .@"struct" or @typeInfo(Options).@"struct".fields.len != 0) {
                    @compileError("typed layerNorm supports only plain .{} options; cast to f32 for the affine path");
                }
            }
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.layerNorm(Self.tag_count, &wide, Self.axis(tag), eps);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        // Widened reductions return f32 like the native typed sum/mean (see docs/reference/08-data-types-storage-and-the-raw-tensor-layer-internal.md:
        // reductions on 16-bit floats keep the accumulator dtype).
        pub fn typedConstantLogsumexp(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "logsumexp");
            const result_tags = removeTag(Self.axis_tags, tag);
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var value = try ctx.logsumexp(Self.tag_count, &wide, Self.axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantMax(self: anytype, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            comptime if (@typeInfo(@TypeOf(opts)).@"struct".fields.len != 0)
                @compileError("typed constant reductions take no options (masked arms are f32-only); pass .{}");
            return typedConstantExtremum(self, ctx, tag, .max);
        }

        pub fn typedConstantMin(self: anytype, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            comptime if (@typeInfo(@TypeOf(opts)).@"struct".fields.len != 0)
                @compileError("typed constant reductions take no options (masked arms are f32-only); pass .{}");
            return typedConstantExtremum(self, ctx, tag, .min);
        }

        pub fn typedConstantExtremum(self: anytype, ctx: *ExecContext, comptime tag: Tag, comptime op: enum { max, min }) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "max/min");
            const result_tags = removeTag(Self.axis_tags, tag);
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var raw = switch (op) {
                .max => try ctx.maxAxis(Self.tag_count, &wide, Self.axis(tag)),
                .min => try ctx.minAxis(Self.tag_count, &wide, Self.axis(tag)),
            };
            raw.indices.deinit();
            errdefer raw.values.deinit();
            return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, raw.values);
        }

        pub fn typedConstantArgmax(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .i64, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "argmax");
            const result_tags = removeTag(Self.axis_tags, tag);
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var value = try ctx.argmax(Self.tag_count, &wide, Self.axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = .i64, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantProd(self: anytype, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "prod");
            const result_tags = removeTag(Self.axis_tags, tag);
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var value = try ctx.prod(Self.tag_count, &wide, Self.axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantVariance(self: anytype, ctx: *ExecContext, comptime tag: Tag, ddof: u1) !Tensor(.{ .dtype = .f32, .tags = removeTag(TensorObject(@TypeOf(self)).axis_tags, tag) }) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "variance");
            const result_tags = removeTag(Self.axis_tags, tag);
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var value = try ctx.varAxis(Self.tag_count, &wide, Self.axis(tag), ddof);
            errdefer value.deinit();
            return Tensor(.{ .dtype = .f32, .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedConstantCumsum(self: anytype, ctx: *ExecContext, comptime tag: Tag) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "cumsum");
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.cumsum(Self.tag_count, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantCumprod(self: anytype, ctx: *ExecContext, comptime tag: Tag) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "cumprod");
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.cumprod(Self.tag_count, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantWhere(self: anytype, ctx: *ExecContext, cond: anytype, other: anytype) !TensorObject(@TypeOf(self)) {
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
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_other = try ctx.cast(Self.dtype, .f32, other_ptr.asRawTensor());
            defer wide_other.deinit();
            var wide_value = try ctx.where(Cond.dtype, &wide, cond_ptr.asRawTensor(), &wide_other);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        pub fn typedConstantMaskedFill(self: anytype, ctx: *ExecContext, mask: anytype, value: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "maskedFill");
            const Mask = TensorObject(@TypeOf(mask));
            if (comptime Mask.dtype != .bool and Mask.dtype != Self.dtype) @compileError("typed maskedFill takes a .bool or same-dtype mask; cast explicitly");
            const mask_ptr = tensorObjectPtrFrom(@TypeOf(mask), &mask);
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.maskedFill(Mask.dtype, &wide, mask_ptr.asRawTensor(), value);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        /// Comparison on the typed branches: `.bool` result everywhere (torch's
        /// comparison dtype). f16/bf16 compare through f32 (the widening seam);
        /// integers compare natively (exact at any magnitude).
        pub fn typedConstantCompare(self: anytype, ctx: *ExecContext, comptime op: exec_mod.CompareOp, other: anytype) !Tensor(.{ .dtype = .bool, .tags = TensorObject(@TypeOf(self)).axis_tags }) {
            const Self = TensorObject(@TypeOf(self));
            const BoolT = Tensor(.{ .dtype = .bool, .tags = Self.axis_tags });
            const OtherT = @TypeOf(other);
            if (comptime dtype_mod.supportsIntMath(Self.dtype)) {
                if (comptime (OtherT == comptime_int or @typeInfo(OtherT) == .int)) {
                    var value = try ctx.compareScalar(Self.dtype, op, self.asRawTensor(), @intCast(other));
                    errdefer value.deinit();
                    return BoolT.fromTensor(ctx, value);
                }
                const Other = TensorObject(@TypeOf(other));
                if (comptime Other.dtype != Self.dtype) @compileError("typed compare requires matching dtypes; cast explicitly");
                const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
                var value = try ctx.compare(Self.dtype, op, self.asRawTensor(), other_ptr.asRawTensor());
                errdefer value.deinit();
                return BoolT.fromTensor(ctx, value);
            }
            comptime requireWidenedTypedFloat(Self.dtype, "compare");
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            if (comptime (OtherT == comptime_float or OtherT == comptime_int or @typeInfo(OtherT) == .float or @typeInfo(OtherT) == .int)) {
                var value = try ctx.compareScalar(.f32, op, &wide, other);
                errdefer value.deinit();
                return BoolT.fromTensor(ctx, value);
            }
            const Other = TensorObject(@TypeOf(other));
            if (comptime Other.dtype != Self.dtype) @compileError("typed compare requires matching dtypes; cast explicitly");
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            var wide_other = try ctx.cast(Self.dtype, .f32, other_ptr.asRawTensor());
            defer wide_other.deinit();
            var value = try ctx.compare(.f32, op, &wide, &wide_other);
            errdefer value.deinit();
            return BoolT.fromTensor(ctx, value);
        }

        /// Widened einsum: both operands widen to f32 and the f32 GEMM lowering
        /// runs (f32 accumulation); the result narrows to the input dtype per the
        /// matmul dtype policy (docs/reference/08-data-types-storage-and-the-raw-tensor-layer-internal.md), the same contract as the typed `dot`.
        pub fn typedConstantEinsum(self: anytype, ctx: *ExecContext, other: anytype, comptime out_tags: anytype) !Tensor(.{ .dtype = TensorObject(@TypeOf(self)).dtype, .tags = normalizeTags(out_tags) }) {
            try typedRequireNoGrad(self);
            try typedRequireNoGrad(other);
            const Self = TensorObject(@TypeOf(self));
            const Other = TensorObject(@TypeOf(other));
            comptime requireWidenedTypedFloat(Self.dtype, "einsum");
            if (comptime Other.dtype != Self.dtype) @compileError("typed einsum requires matching dtypes; cast explicitly");
            const result_tags = comptime normalizeTags(out_tags);
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            var wide_left = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide_left.deinit();
            var wide_right = try ctx.cast(Self.dtype, .f32, other_ptr.asRawTensor());
            defer wide_right.deinit();
            var wide_value = try tag_ops.taggedEinsum(Self.axis_tags, &wide_left, ctx, Other.axis_tags, &wide_right, result_tags);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, result_tags, ctx, &wide_value);
        }

        pub fn typedConstantPad(self: anytype, ctx: *ExecContext, comptime tag: Tag, before: usize, after: usize, fill: f32) !TensorObject(@TypeOf(self)) {
            try typedRequireNoGrad(self);
            const Self = TensorObject(@TypeOf(self));
            comptime requireWidenedTypedFloat(Self.dtype, "pad");
            var wide = try ctx.cast(Self.dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var wide_value = try ctx.pad(Self.tag_count, &wide, Self.axis(tag), before, after, fill);
            defer wide_value.deinit();
            return typedFromWidened(Self.dtype, Self.axis_tags, ctx, &wide_value);
        }

        /// Native typed pointwise over the tag-broadcast rule (`tag_ops.pointwise`
        /// on `tensor_dtype`). The facade rules stated here: integer `div` is
        /// explicit (`divTrunc`/`divFloor`), float `maximum`/`minimum` widen
        /// through f32, and `.bool` has no arithmetic.
        pub fn typedPointwise(
            comptime tensor_dtype: DType,
            comptime op: PointwiseOp,
            self: anytype,
            ctx: *ExecContext,
            other: anytype,
        ) !Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, tensor_dtype), .tags = pointwiseResultTags(TensorObject(@TypeOf(self)).axis_tags, TensorObject(@TypeOf(other)).axis_tags) }) {
            try typedRequireNoGrad(self);
            try typedRequireNoGrad(other);
            const SelfTensor = TensorObject(@TypeOf(self));
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (Other.dtype != tensor_dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
                if (tensor_dtype == .bool) @compileError("bool tensors have no pointwise arithmetic; cast with to() first");
                if (op == .div and dtype_mod.supportsIntMath(tensor_dtype))
                    @compileError("integer `div` is explicit: use divTrunc/divFloor (torch's `/` promotes to float; Fucina keeps promotion explicit)");
                if ((op == .max or op == .min) and !dtype_mod.supportsIntMath(tensor_dtype))
                    @compileError("float typed maximum/minimum widen through f32 (the f16/bf16 facade entries)");
            }
            const left = tensorObjectPtrFrom(@TypeOf(self), &self);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            const result_tags = pointwiseResultTags(SelfTensor.axis_tags, Other.axis_tags);
            var value = try tag_ops.pointwise(tensor_dtype, op, SelfTensor.axis_tags, left.asRawTensor(), ctx, Other.axis_tags, right.asRawTensor());
            errdefer value.deinit();
            return Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, tensor_dtype), .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn typedDot(
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
    };
}
