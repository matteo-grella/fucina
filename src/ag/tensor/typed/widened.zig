//! The widened forward family of the 16-bit branches (f16/bf16): ops with
//! no native typed kernel. The input widens to f32, the f32 exec kernel
//! runs, and the result narrows ONCE on store: f32 accumulation with a
//! single final round, the dtype policy in docs/reference/08, §8.3. f64 is
//! excluded at comptime (f64 math must stay f64; rounding it through f32
//! would silently lose precision). Every op is a no-grad constant. A mixin
//! over the tensor struct; aliased back onto it in ../../tensor.zig.

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

        pub fn unary(self: *const Self, ctx: *ExecContext, comptime op: UnaryOp) !Self {
            var wide = try widen(self, ctx, "unary");
            defer wide.deinit();
            var wide_value = try ctx.unary(op, &wide);
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        fn Unary(comptime op: UnaryOp) type {
            return struct {
                pub fn call(self: *const Self, ctx: *ExecContext) !Self {
                    return self.unary(ctx, op);
                }
            };
        }

        pub const relu = Unary(.relu).call;
        pub const exp = Unary(.exp).call;
        pub const sqrt = Unary(.sqrt).call;
        pub const rsqrt = Unary(.rsqrt).call;
        pub const sigmoid = Unary(.sigmoid).call;
        pub const silu = Unary(.silu).call;
        pub const log = Unary(.log).call;
        pub const log1p = Unary(.log1p).call;
        pub const neg = Unary(.neg).call;
        pub const abs = Unary(.abs).call;
        pub const sin = Unary(.sin).call;
        pub const cos = Unary(.cos).call;
        pub const tanh = Unary(.tanh).call;
        pub const fastTanh = Unary(.fast_tanh).call;
        pub const softcap30 = Unary(.softcap_30).call;
        pub const softcap15 = Unary(.softcap_15).call;
        pub const gelu = Unary(.gelu).call;
        pub const quickGelu = Unary(.quick_gelu).call;
        pub const elu = Unary(.elu).call;
        pub const geluErf = Unary(.gelu_erf).call;
        pub const erf = Unary(.erf).call;
        pub const floor = Unary(.floor).call;
        pub const ceil = Unary(.ceil).call;
        pub const round = Unary(.round).call;
        pub const sign = Unary(.sign).call;
        pub const reciprocal = Unary(.reciprocal).call;

        pub fn leakyRelu(self: *const Self, ctx: *ExecContext, negative_slope: f32) !Self {
            var wide = try widen(self, ctx, "leakyRelu");
            defer wide.deinit();
            var wide_value = try ctx.leakyRelu(&wide, negative_slope);
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn clamp(self: *const Self, ctx: *ExecContext, min_value: f32, max_value: f32) !Self {
            var wide = try widen(self, ctx, "clamp");
            defer wide.deinit();
            var wide_value = try ctx.clamp(&wide, min_value, max_value);
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn addScalar(self: *const Self, ctx: *ExecContext, scalar_value: f32) !Self {
            var wide = try widen(self, ctx, "addScalar");
            defer wide.deinit();
            var wide_value = try ctx.addScalar(&wide, scalar_value);
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn subScalar(self: *const Self, ctx: *ExecContext, scalar_value: f32) !Self {
            return self.addScalar(ctx, -scalar_value);
        }

        pub fn powScalar(self: *const Self, ctx: *ExecContext, exponent: f32) !Self {
            var wide = try widen(self, ctx, "powScalar");
            defer wide.deinit();
            var wide_value = try ctx.powScalar(&wide, exponent);
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        /// Widened binary pointwise (`maximum`/`minimum`): both operands
        /// widen, the f32 tag-broadcast kernel runs, the result narrows.
        fn pointwise(comptime op: PointwiseOp, self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            const Other = TensorObject(@TypeOf(other));
            var wide_left = try widen(self, ctx, "maximum/minimum");
            defer wide_left.deinit();
            var wide_right = try widenOther(other, ctx, "pointwise");
            defer wide_right.deinit();
            var wide_value = try tag_ops.pointwise(.f32, op, tags, &wide_left, ctx, Other.axis_tags, &wide_right);
            defer wide_value.deinit();
            return narrow(pointwiseResultTags(tags, Other.axis_tags), ctx, &wide_value);
        }

        pub fn maximum(self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            return pointwise(.max, self, ctx, other);
        }

        pub fn minimum(self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            return pointwise(.min, self, ctx, other);
        }

        pub fn gated(self: *const Self, ctx: *ExecContext, other: anytype, comptime op: GatedOp) !PointwiseOut(TensorObject(@TypeOf(other))) {
            const Other = TensorObject(@TypeOf(other));
            var wide_left = try widen(self, ctx, "gated");
            defer wide_left.deinit();
            var wide_right = try widenOther(other, ctx, "gated");
            defer wide_right.deinit();
            var wide_value = try tag_ops.gatedPointwise(op, tags, &wide_left, ctx, Other.axis_tags, &wide_right);
            defer wide_value.deinit();
            return narrow(pointwiseResultTags(tags, Other.axis_tags), ctx, &wide_value);
        }

        pub fn glu(self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            return self.gated(ctx, other, .glu);
        }

        pub fn swiglu(self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            return self.gated(ctx, other, .swiglu);
        }

        pub fn geglu(self: *const Self, ctx: *ExecContext, other: anytype) !PointwiseOut(TensorObject(@TypeOf(other))) {
            return self.gated(ctx, other, .geglu);
        }

        pub fn softmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag, options: anytype) !Self {
            comptime requirePlainOptions(options, "typed softmax supports only plain .{} options; cast to f32 for the ext path (mask/sinks/causal/scale)");
            var wide = try widen(self, ctx, "softmax");
            defer wide.deinit();
            var wide_value = try ctx.softmax(tag_rank, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn logSoftmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Self {
            var wide = try widen(self, ctx, "logSoftmax");
            defer wide.deinit();
            var wide_value = try ctx.logSoftmax(tag_rank, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn rmsNorm(self: *const Self, ctx: *ExecContext, comptime tag: Tag, eps: f32) !Self {
            var wide = try widen(self, ctx, "rmsNorm");
            defer wide.deinit();
            var wide_value = try ctx.rmsNorm(tag_rank, &wide, Self.axis(tag), eps);
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn rmsNormMul(self: *const Self, ctx: *ExecContext, comptime tag: Tag, weight: *const Same(.{tag}), eps: f32) !Self {
            var wide = try widen(self, ctx, "rmsNormMul");
            defer wide.deinit();
            var wide_weight = try widenOther(weight, ctx, "rmsNormMul");
            defer wide_weight.deinit();
            var wide_value = try ctx.rmsNormMul(tag_rank, &wide, &wide_weight, Self.axis(tag), eps);
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn layerNorm(self: *const Self, ctx: *ExecContext, comptime tag: Tag, eps: f32, options: anytype) !Self {
            comptime requirePlainOptions(options, "typed layerNorm supports only plain .{} options; cast to f32 for the affine path");
            var wide = try widen(self, ctx, "layerNorm");
            defer wide.deinit();
            var wide_value = try ctx.layerNorm(tag_rank, &wide, Self.axis(tag), eps);
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn cumsum(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Self {
            var wide = try widen(self, ctx, "cumsum");
            defer wide.deinit();
            var wide_value = try ctx.cumsum(tag_rank, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn cumprod(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Self {
            var wide = try widen(self, ctx, "cumprod");
            defer wide.deinit();
            var wide_value = try ctx.cumprod(tag_rank, &wide, Self.axis(tag));
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn where(self: *const Self, ctx: *ExecContext, cond: anytype, other: anytype) !Self {
            const Cond = TensorObject(@TypeOf(cond));
            comptime if (Cond.dtype != .bool and Cond.dtype != dtype) @compileError("typed where takes a .bool or same-dtype condition; cast explicitly");
            const cond_ptr = tensorObjectPtrFrom(@TypeOf(cond), &cond);
            var wide = try widen(self, ctx, "where");
            defer wide.deinit();
            var wide_other = try widenOther(other, ctx, "where");
            defer wide_other.deinit();
            var wide_value = try ctx.where(Cond.dtype, &wide, cond_ptr.asRawTensor(), &wide_other);
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn maskedFill(self: *const Self, ctx: *ExecContext, mask: anytype, value: f32) !Self {
            const Mask = TensorObject(@TypeOf(mask));
            comptime if (Mask.dtype != .bool and Mask.dtype != dtype) @compileError("typed maskedFill takes a .bool or same-dtype mask; cast explicitly");
            const mask_ptr = tensorObjectPtrFrom(@TypeOf(mask), &mask);
            var wide = try widen(self, ctx, "maskedFill");
            defer wide.deinit();
            var wide_value = try ctx.maskedFill(Mask.dtype, &wide, mask_ptr.asRawTensor(), value);
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        /// Comparison through f32 (the widening seam): `.bool` result;
        /// `other` is a same-dtype tensor or a numeric scalar.
        pub fn compare(self: *const Self, ctx: *ExecContext, comptime op: exec_mod.CompareOp, other: anytype) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            const BoolT = Tensor(.{ .dtype = .bool, .tags = tags });
            const OtherT = @TypeOf(other);
            comptime requireWidened("compare");
            var wide = try ctx.cast(dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            if (comptime (OtherT == comptime_float or OtherT == comptime_int or @typeInfo(OtherT) == .float or @typeInfo(OtherT) == .int)) {
                var value = try ctx.compareScalar(.f32, op, &wide, other);
                errdefer value.deinit();
                return BoolT.fromTensor(ctx, value);
            }
            const Other = TensorObject(OtherT);
            comptime if (Other.dtype != dtype) @compileError("typed compare requires matching dtypes; cast explicitly");
            const other_ptr = tensorObjectPtrFrom(OtherT, &other);
            var wide_other = try ctx.cast(dtype, .f32, other_ptr.asRawTensor());
            defer wide_other.deinit();
            var value = try ctx.compare(.f32, op, &wide, &wide_other);
            errdefer value.deinit();
            return BoolT.fromTensor(ctx, value);
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

        pub fn pad(self: *const Self, ctx: *ExecContext, comptime tag: Tag, before: usize, after: usize, fill: f32) !Self {
            var wide = try widen(self, ctx, "pad");
            defer wide.deinit();
            var wide_value = try ctx.pad(tag_rank, &wide, Self.axis(tag), before, after, fill);
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        // Widened reductions return f32 like the native typed sum/mean
        // (docs/reference/08, §8.3: reductions on 16-bit floats keep the
        // accumulator dtype).
        pub fn logsumexp(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Wide(removeTag(tags, tag)) {
            var wide = try widen(self, ctx, "logsumexp");
            defer wide.deinit();
            var value = try ctx.logsumexp(tag_rank, &wide, Self.axis(tag));
            errdefer value.deinit();
            return Wide(removeTag(tags, tag)).fromTensor(ctx, value);
        }

        pub fn max(self: *const Self, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !Wide(removeTag(tags, tag)) {
            comptime requirePlainOptions(opts, "typed constant reductions take no options (masked arms are f32-only); pass .{}");
            return extremum(.max, self, ctx, tag);
        }

        pub fn min(self: *const Self, ctx: *ExecContext, comptime tag: Tag, opts: anytype) !Wide(removeTag(tags, tag)) {
            comptime requirePlainOptions(opts, "typed constant reductions take no options (masked arms are f32-only); pass .{}");
            return extremum(.min, self, ctx, tag);
        }

        fn extremum(comptime op: enum { max, min }, self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Wide(removeTag(tags, tag)) {
            var wide = try widen(self, ctx, "max/min");
            defer wide.deinit();
            var raw = switch (op) {
                .max => try ctx.maxAxis(tag_rank, &wide, Self.axis(tag)),
                .min => try ctx.minAxis(tag_rank, &wide, Self.axis(tag)),
            };
            raw.indices.deinit();
            errdefer raw.values.deinit();
            return Wide(removeTag(tags, tag)).fromTensor(ctx, raw.values);
        }

        pub fn argmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = .i64, .tags = removeTag(tags, tag) }) {
            comptime requireWidened("argmax");
            var wide = try ctx.cast(dtype, .f32, self.asRawTensor());
            defer wide.deinit();
            var value = try ctx.argmax(tag_rank, &wide, Self.axis(tag));
            errdefer value.deinit();
            return Tensor(.{ .dtype = .i64, .tags = removeTag(tags, tag) }).fromTensor(ctx, value);
        }

        pub fn prod(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Wide(removeTag(tags, tag)) {
            var wide = try widen(self, ctx, "prod");
            defer wide.deinit();
            var value = try ctx.prod(tag_rank, &wide, Self.axis(tag));
            errdefer value.deinit();
            return Wide(removeTag(tags, tag)).fromTensor(ctx, value);
        }

        pub fn variance(self: *const Self, ctx: *ExecContext, comptime tag: Tag, ddof: u1) !Wide(removeTag(tags, tag)) {
            var wide = try widen(self, ctx, "variance");
            defer wide.deinit();
            var value = try ctx.varAxis(tag_rank, &wide, Self.axis(tag), ddof);
            errdefer value.deinit();
            return Wide(removeTag(tags, tag)).fromTensor(ctx, value);
        }

        fn requirePlainOptions(options: anytype, comptime message: []const u8) void {
            const Options = @TypeOf(options);
            if (@typeInfo(Options) != .@"struct" or @typeInfo(Options).@"struct".fields.len != 0) {
                @compileError(message);
            }
        }
    };
}
