//! The widened forward family of the 16-bit branches (f16/bf16): the ops
//! whose exec entry is still f32-only (softmax, the norms, the scans, the
//! comparison, pad, einsum, the widened reductions). The input widens to
//! f32, the f32 exec kernel runs, and the result narrows ONCE on store: f32
//! accumulation with a single final round, the dtype policy in
//! docs/reference/08, §8.3. The elementwise family no longer lives here: its
//! exec entries take the dtype and apply the same policy themselves, so the
//! 16-bit branches share `../elementwise.zig` with f32. f64 is excluded at
//! comptime (f64 math must stay f64; rounding it through f32 would silently
//! lose precision). Every op is a no-grad constant. A mixin over the tensor
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
            var wide_value = try ctx.rmsNorm(tag_rank, &wide, Self.axis(tag), eps, .{});
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn rmsNormMul(self: *const Self, ctx: *ExecContext, comptime tag: Tag, weight: *const Same(.{tag}), eps: f32) !Self {
            var wide = try widen(self, ctx, "rmsNormMul");
            defer wide.deinit();
            var wide_weight = try widenOther(weight, ctx, "rmsNormMul");
            defer wide_weight.deinit();
            var wide_value = try ctx.rmsNorm(tag_rank, &wide, Self.axis(tag), eps, .{ .weight = &wide_weight });
            defer wide_value.deinit();
            return narrow(tags, ctx, &wide_value);
        }

        pub fn layerNorm(self: *const Self, ctx: *ExecContext, comptime tag: Tag, eps: f32, options: anytype) !Self {
            comptime requirePlainOptions(options, "typed layerNorm supports only plain .{} options; cast to f32 for the affine path");
            var wide = try widen(self, ctx, "layerNorm");
            defer wide.deinit();
            var wide_value = try ctx.layerNorm(tag_rank, &wide, Self.axis(tag), eps, .{});
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
