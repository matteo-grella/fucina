//! Casts and the typed reductions for the typed branches: the native
//! reductions, the typed `dot`, and exact integer comparison. Every op is a
//! no-grad constant except `to(.f32)` on a 16-bit leaf (the differentiable
//! widen). The pointwise family lives in `../elementwise.zig`, shared with
//! the f32 branch. A mixin over the tensor struct; aliased back onto it in
//! ../../tensor.zig, where each branch picks the entries its dtype has a
//! kernel for (integer `div` is explicit `divTrunc`/`divFloor` in int.zig;
//! float `maximum`/`minimum` and `compare` come from ../elementwise.zig).

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
                if (comptime plumbing.hasGradSlot(Self)) {
                    if (!plumbing.recordsGrad(self.requiresGrad())) return plumbing.finishNoGrad(tags, ctx, value);
                    const Record = CastBackward(tags);
                    return plumbing.finishOp(tags, ctx, value, Record{ .parents = .{self.grad_state} });
                }
                // Integer/bool sources are grad-free: a plain f32 constant.
                return plumbing.finishNoGrad(tags, ctx, value);
            }
            return plumbing.finishTyped(Tensor(.{ .dtype = target_dtype, .tags = tags }), ctx, value);
        }

        pub fn sum(self: *const Self, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !ReducedOut(removeTag(tags, tag)) {
            comptime requireNoOptions(opts);
            try requireNoGrad(self);
            var value = try ctx.sumAxis(dtype, tag_rank, self.asRawTensor(), Self.axis(tag));
            errdefer value.deinit();
            return plumbing.finishTyped(ReducedOut(removeTag(tags, tag)), ctx, value);
        }

        pub fn mean(self: *const Self, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !ReducedOut(removeTag(tags, tag)) {
            comptime requireNoOptions(opts);
            try requireNoGrad(self);
            var value = try ctx.meanAxis(dtype, tag_rank, self.asRawTensor(), Self.axis(tag));
            errdefer value.deinit();
            return plumbing.finishTyped(ReducedOut(removeTag(tags, tag)), ctx, value);
        }

        pub fn sumAll(self: *const Self, ctx: *ExecContext) !ReducedOut(.{}) {
            try requireNoGrad(self);
            var value = try ctx.sum(dtype, self.asRawTensor());
            errdefer value.deinit();
            return plumbing.finishTyped(ReducedOut(.{}), ctx, value);
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
            return plumbing.finishTyped(Tensor(.{ .dtype = dtype_mod.outputDType(.matmul, dtype), .tags = result_tags }), ctx, value);
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
                return plumbing.finishTyped(BoolT, ctx, value);
            }
            const Other = TensorObject(OtherT);
            comptime if (Other.dtype != dtype) @compileError("typed compare requires matching dtypes; cast explicitly");
            const other_ptr = tensorObjectPtrFrom(OtherT, &other);
            var value = try ctx.compare(dtype, op, self.asRawTensor(), other_ptr.asRawTensor());
            errdefer value.deinit();
            return plumbing.finishTyped(BoolT, ctx, value);
        }

        fn requireNoOptions(opts: anytype) void {
            if (@typeInfo(@TypeOf(opts)).@"struct".fields.len != 0)
                @compileError("typed constant reductions take no options (masked arms are f32-only); pass .{}");
        }
    };
}
