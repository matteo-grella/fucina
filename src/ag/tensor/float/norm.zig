//! f32 tensor methods: normalization ops and norms. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const core = @import("../../core.zig");
const tags_mod = @import("../../../tags.zig");
const backward_norm = @import("../../backward/norm.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tag = tags_mod.Tag;
const removeTag = tags_mod.removeTag;
const RmsNormBackward = backward_norm.RmsNormBackward;
const RmsNormMulBackward = backward_norm.RmsNormMulBackward;
const RmsNormMulAddBackward = backward_norm.RmsNormMulAddBackward;
const RmsNormMulRopeBackward = backward_norm.RmsNormMulRopeBackward;
const LayerNormBackward = backward_norm.LayerNormBackward;
const LayerNormAffineBackward = backward_norm.LayerNormAffineBackward;
const GroupNormBackward = backward_norm.GroupNormBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const normParamTagCheck = ag_tensor.normParamTagCheck;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const finishOp = plumbing.finishOp;
        const requireScopeForComposedGrad = plumbing.requireScopeForComposedGrad;
        const TensorObject = plumbing.TensorObject;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;

        /// GroupNorm over `[time, channel]` rows (ggml semantics; see
        /// `groupNorm`): per group of channel columns, f64-accumulated
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

            var value = try ctx.groupNorm(self.asRawTensor(), groups, eps, .{ .weight = weight_raw, .bias = bias_raw });
            errdefer value.deinit();
            return finishOp(tags, ctx, value, any_grad, GroupNormBackward(tags), .{ ctx.allocator, self.grad_state, weight_parent, bias_parent, self.asRawTensor(), weight_raw, groups, eps });
        }

        pub fn rmsNorm(self: *const Self, ctx: *ExecContext, comptime tag: Tag, eps: f32) !Self {
            const norm_axis = comptime axis(tag);
            var value = try ctx.rmsNorm(tag_rank, self.asRawTensor(), norm_axis, eps, .{});
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), RmsNormBackward(tags, norm_axis), .{ ctx.allocator, self.grad_state, &self.value, eps });
        }

        pub fn rmsNormMul(self: *const Self, ctx: *ExecContext, comptime tag: Tag, weight: *const Tensor(.{tag}), eps: f32) !Self {
            const norm_axis = comptime axis(tag);
            var value = try ctx.rmsNorm(tag_rank, self.asRawTensor(), norm_axis, eps, .{ .weight = weight.asRawTensor() });
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad() or weight.requiresGrad(), RmsNormMulBackward(tags, norm_axis), .{ ctx.allocator, self.grad_state, weight.grad_state, self.asRawTensor(), weight.asRawTensor(), eps });
        }

        pub fn rmsNormMulAdd(self: *const Self, ctx: *ExecContext, comptime tag: Tag, weight: *const Tensor(.{tag}), residual: *const Self, eps: f32) !Self {
            const norm_axis = comptime axis(tag);
            var value = try ctx.rmsNorm(tag_rank, self.asRawTensor(), norm_axis, eps, .{ .weight = weight.asRawTensor(), .residual = residual.asRawTensor() });
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
            var value = try ctx.rmsNormMulRopeWithTable(
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
                var value = try ctx.layerNorm(tag_rank, self.asRawTensor(), norm_axis, eps, .{ .weight = weight_ptr.asRawTensor(), .bias = bias_ptr.asRawTensor() });
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
            var value = try ctx.layerNorm(tag_rank, self.asRawTensor(), norm_axis, eps, .{});
            errdefer value.deinit();
            return finishOp(tags, ctx, value, self.requiresGrad(), LayerNormBackward(tags, norm_axis), .{ ctx.allocator, self.grad_state, &self.value, eps });
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
            var sum_sq = try sq.sum(ctx, tag, .{});
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
                    return magnitude.sum(ctx, tag, .{});
                },
                .l2 => {
                    var squared = try self.mul(ctx, self);
                    defer squared.deinit();
                    var sum_sq = try squared.sum(ctx, tag, .{});
                    defer sum_sq.deinit();
                    return sum_sq.sqrt(ctx);
                },
                .inf => {
                    var magnitude = try self.abs(ctx);
                    defer magnitude.deinit();
                    return magnitude.max(ctx, tag, .{});
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
            var dot_sum = try pointwise_prod.sum(ctx, tag, .{});
            defer dot_sum.deinit();
            var self_sq = try self.mul(ctx, self);
            defer self_sq.deinit();
            var self_sum_sq = try self_sq.sum(ctx, tag, .{});
            defer self_sum_sq.deinit();
            var self_norm = try self_sum_sq.sqrt(ctx);
            defer self_norm.deinit();
            var other_sq = try other.mul(ctx, other);
            defer other_sq.deinit();
            var other_sum_sq = try other_sq.sum(ctx, tag, .{});
            defer other_sum_sq.deinit();
            var other_norm = try other_sum_sq.sqrt(ctx);
            defer other_norm.deinit();
            var denom = try self_norm.mul(ctx, &other_norm);
            defer denom.deinit();
            var clamped = try denom.clamp(ctx, eps, std.math.floatMax(f32));
            defer clamped.deinit();
            return dot_sum.div(ctx, &clamped);
        }
    };
}
