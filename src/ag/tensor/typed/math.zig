//! Casts and forward math over the STORED dtype for the typed branches:
//! the tag-broadcast pointwise family, the native reductions, the typed
//! `dot`, scalar scaling, and exact integer comparison. Every op is a
//! no-grad constant except `to(.f32)` on a 16-bit leaf (the differentiable
//! widen). A mixin over the tensor struct; aliased back onto it in
//! ../../tensor.zig, where each branch picks the entries its dtype has a
//! kernel for (integer `div` is explicit `divTrunc`/`divFloor` in int.zig;
//! float `maximum`/`minimum` and `compare` widen through widened.zig).

const tensor_mod = @import("../../../tensor.zig");
const dtype_mod = @import("../../../dtype.zig");
const exec_mod = @import("../../../exec.zig");
const tag_ops = @import("../../../tag_ops.zig");
const tags_mod = @import("../../../tags.zig");
const backward_common = @import("../../backward/common.zig");
const backward_elementwise = @import("../../backward/elementwise.zig");

const ExecContext = exec_mod.ExecContext;
const Tag = tags_mod.Tag;
const removeTag = tags_mod.removeTag;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const dotResultTags = tags_mod.dotResultTags;
const PointwiseOp = backward_common.PointwiseOp;
const CastBackward = backward_elementwise.CastBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const dtype = Self.dtype;
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const TensorObject = plumbing.TensorObject;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;
        const requireNoGrad = plumbing.typedRequireNoGrad;

        fn PointwiseOut(comptime Other: type) type {
            return Tensor(.{ .dtype = dtype_mod.outputDType(.pointwise, dtype), .tags = pointwiseResultTags(tags, Other.axis_tags) });
        }

        fn ReducedOut(comptime result_tags: anytype) type {
            return Tensor(.{ .dtype = dtype_mod.outputDType(.reduction, dtype), .tags = result_tags });
        }

        /// Scalar cast (docs/reference/03, §3.8). `to(.f32)` on a 16-bit
        /// leaf is differentiable (the f32 result joins the graph and the
        /// source receives the upstream gradient unchanged); any other
        /// cast of a grad-requiring tensor is `error.GradientCastUnsupported`.
        pub fn to(self: *const Self, ctx: *ExecContext, comptime target_dtype: dtype_mod.DType) !Tensor(.{ .dtype = target_dtype, .tags = tags }) {
            if (comptime target_dtype != .f32) {
                if (self.requiresGrad()) return error.GradientCastUnsupported;
            }
            var value = try ctx.cast(dtype, target_dtype, self.asRawTensor());
            errdefer value.deinit();
            if (comptime target_dtype == .f32) {
                if (comptime @hasField(Self, "grad_state")) {
                    return plumbing.finishOp(tags, ctx, value, self.requiresGrad(), CastBackward(tags), .{ ctx.allocator, self.grad_state });
                }
                // Integer/bool sources are grad-free: a plain f32 constant.
                return plumbing.finishNoGrad(tags, ctx, value);
            }
            return Tensor(.{ .dtype = target_dtype, .tags = tags }).fromTensor(ctx, value);
        }

        pub fn add(self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            return pointwise(.add, self, ctx, other);
        }

        pub fn sub(self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            return pointwise(.sub, self, ctx, other);
        }

        pub fn mul(self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            return pointwise(.mul, self, ctx, other);
        }

        pub fn div(self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            return pointwise(.div, self, ctx, other);
        }

        pub fn maximum(self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            return pointwise(.max, self, ctx, other);
        }

        pub fn minimum(self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            return pointwise(.min, self, ctx, other);
        }

        pub fn sum(self: *const Self, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !ReducedOut(removeTag(tags, tag)) {
            comptime requireNoOptions(opts);
            try requireNoGrad(self);
            var value = try ctx.sumAxis(dtype, tag_rank, self.asRawTensor(), Self.axis(tag));
            errdefer value.deinit();
            return ReducedOut(removeTag(tags, tag)).fromTensor(ctx, value);
        }

        pub fn mean(self: *const Self, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !ReducedOut(removeTag(tags, tag)) {
            comptime requireNoOptions(opts);
            try requireNoGrad(self);
            var value = try ctx.meanAxis(dtype, tag_rank, self.asRawTensor(), Self.axis(tag));
            errdefer value.deinit();
            return ReducedOut(removeTag(tags, tag)).fromTensor(ctx, value);
        }

        pub fn sumAll(self: *const Self, ctx: *ExecContext) !ReducedOut(.{}) {
            try requireNoGrad(self);
            var value = try ctx.sum(dtype, self.asRawTensor());
            errdefer value.deinit();
            return ReducedOut(.{}).fromTensor(ctx, value);
        }

        /// Same-dtype contraction over one tag (the typed lowering in
        /// `plumbing.typedDotRaw`); the result dtype follows the matmul
        /// policy (docs/reference/08, §8.3).
        pub fn dot(self: *const Self, ctx: *ExecContext, other: anytype, comptime contract_tag: Tag) !Tensor(.{ .dtype = dtype_mod.outputDType(.matmul, dtype), .tags = dotResultTags(tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag) }) {
            try requireNoGrad(self);
            try requireNoGrad(other);
            const Other = TensorObject(@TypeOf(other));
            comptime if (Other.dtype != dtype) @compileError("typed dot requires matching dtypes; cast explicitly");
            const result_tags = dotResultTags(tags, Other.axis_tags, contract_tag);
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            var value = try plumbing.typedDotRaw(dtype, tags, self.asRawTensor(), ctx, Other.axis_tags, right.asRawTensor(), contract_tag);
            errdefer value.deinit();
            return Tensor(.{ .dtype = dtype_mod.outputDType(.matmul, dtype), .tags = result_tags }).fromTensor(ctx, value);
        }

        pub fn scale(self: *const Self, ctx: *ExecContext, scalar_value: dtype_mod.Accumulator(dtype)) !Self {
            try requireNoGrad(self);
            var value = try ctx.scale(dtype, self.asRawTensor(), scalar_value);
            errdefer value.deinit();
            return Self.fromTensor(ctx, value);
        }

        pub fn divScalar(self: *const Self, ctx: *ExecContext, scalar_value: dtype_mod.Accumulator(dtype)) !Self {
            return self.scale(ctx, 1.0 / scalar_value);
        }

        /// Exact comparison on an integer branch: `.bool` result (torch's
        /// comparison dtype); `other` is a same-dtype tensor or an integer
        /// scalar.
        pub fn compare(self: *const Self, ctx: *ExecContext, comptime op: exec_mod.CompareOp, other: anytype) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            const BoolT = Tensor(.{ .dtype = .bool, .tags = tags });
            const OtherT = @TypeOf(other);
            if (comptime (OtherT == comptime_int or @typeInfo(OtherT) == .int)) {
                var value = try ctx.compareScalar(dtype, op, self.asRawTensor(), @intCast(other));
                errdefer value.deinit();
                return BoolT.fromTensor(ctx, value);
            }
            const Other = TensorObject(OtherT);
            comptime if (Other.dtype != dtype) @compileError("typed compare requires matching dtypes; cast explicitly");
            const other_ptr = tensorObjectPtrFrom(OtherT, &other);
            var value = try ctx.compare(dtype, op, self.asRawTensor(), other_ptr.asRawTensor());
            errdefer value.deinit();
            return BoolT.fromTensor(ctx, value);
        }

        /// Native typed pointwise over the tag-broadcast rule
        /// (`tag_ops.pointwise` on the stored dtype).
        fn pointwise(comptime op: PointwiseOp, self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            try requireNoGrad(self);
            try requireNoGrad(other);
            const Other = TensorObject(@TypeOf(other));
            comptime {
                if (Other.dtype != dtype) @compileError("typed pointwise requires matching dtypes; cast explicitly");
                if (dtype == .bool) @compileError("bool tensors have no pointwise arithmetic; cast with to() first");
            }
            const right = tensorObjectPtrFrom(@TypeOf(other), &other);
            var value = try tag_ops.pointwise(dtype, op, tags, self.asRawTensor(), ctx, Other.axis_tags, right.asRawTensor());
            errdefer value.deinit();
            return PointwiseOut(Other).fromTensor(ctx, value);
        }

        fn requireNoOptions(opts: anytype) void {
            if (@typeInfo(@TypeOf(opts)).@"struct".fields.len != 0)
                @compileError("typed constant reductions take no options (masked arms are f32-only); pass .{}");
        }
    };
}
