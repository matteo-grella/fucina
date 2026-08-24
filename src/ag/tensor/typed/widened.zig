//! The widened forward family of the 16-bit branches (f16/bf16): `einsum`,
//! the one op whose exec lowering is still f32-only. The operands widen to
//! f32, the f32 GEMM lowering runs, and the result narrows ONCE on store:
//! the dtype policy in docs/reference/08, §8.3. Every other former member
//! (the elementwise and comparison families, softmax, the scans, the
//! reductions, pad, the norms) now takes its dtype at the exec seam and
//! applies the same policy there, so the 16-bit branches share the f32
//! mixins for them. f64 is excluded at comptime (f64
//! math must stay f64; rounding it through f32 would silently lose
//! precision). Every op is a no-grad constant. A mixin over the tensor
//! struct; aliased back onto it in ../../tensor.zig.

const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const tag_ops = @import("../../../tag_ops.zig");
const tags_mod = @import("../../../tags.zig");
const backward_common = @import("../../backward/common.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const UnaryOp = exec_mod.UnaryOp;
const GatedOp = exec_mod.GatedOp;
const Tag = tags_mod.Tag;
const removeTag = tags_mod.removeTag;
const normalizeTags = tags_mod.normalizeTags;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const PointwiseOp = backward_common.PointwiseOp;

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

        fn Same(comptime result_tags: anytype) type {
            return Tensor(.{ .dtype = dtype, .tags = result_tags });
        }

        fn Wide(comptime result_tags: anytype) type {
            return Tensor(.{ .dtype = .f32, .tags = result_tags });
        }

        fn PointwiseOut(comptime Other: type) type {
            return Same(pointwiseResultTags(tags, Other.axis_tags));
        }

        fn requireWidened(comptime what: []const u8) void {
            if (dtype != .f16 and dtype != .bf16) {
                @compileError(what ++ " on the typed float branch is f16/bf16 only (it computes through f32; f64 must not round through f32; cast explicitly)");
            }
        }

        /// `self` widened to f32 (caller deinits).
        fn widen(self: *const Self, ctx: *ExecContext, comptime what: []const u8) !RawTensor {
            comptime requireWidened(what);
            try requireNoGrad(self);
            return ctx.cast(dtype, .f32, self.asRawTensor());
        }

        /// A same-dtype operand widened to f32 (caller deinits).
        fn widenOther(other: anytype, ctx: *ExecContext, comptime what: []const u8) !RawTensor {
            const Other = TensorObject(@TypeOf(other));
            comptime if (Other.dtype != dtype) @compileError("typed " ++ what ++ " requires matching dtypes; cast explicitly");
            try requireNoGrad(other);
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            return ctx.cast(dtype, .f32, other_ptr.asRawTensor());
        }

        /// Narrow the f32 kernel result back to the stored dtype.
        fn narrow(comptime result_tags: anytype, ctx: *ExecContext, wide_value: *const RawTensor) !Same(result_tags) {
            var value = try ctx.cast(.f32, dtype, wide_value);
            errdefer value.deinit();
            return Same(result_tags).fromTensor(ctx, value);
        }

        /// Widened einsum: both operands widen to f32 and the f32 GEMM
        /// lowering runs (f32 accumulation); the result narrows to the
        /// input dtype per the matmul dtype policy, the same contract as
        /// the typed `dot`.
        pub fn einsum(self: *const Self, ctx: *ExecContext, other: anytype, comptime out_tags: anytype) !Same(normalizeTags(out_tags)) {
            const Other = TensorObject(@TypeOf(other));
            const result_tags = comptime normalizeTags(out_tags);
            var wide_left = try widen(self, ctx, "einsum");
            defer wide_left.deinit();
            var wide_right = try widenOther(other, ctx, "einsum");
            defer wide_right.deinit();
            var wide_value = try tag_ops.taggedEinsum(tags, &wide_left, ctx, Other.axis_tags, &wide_right, result_tags);
            defer wide_value.deinit();
            return narrow(result_tags, ctx, &wide_value);
        }
    };
}
