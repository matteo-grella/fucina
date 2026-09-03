//! Tensor methods over the float dtypes: pointwise arithmetic, activations,
//! masks, casts. One mixin over the ag tensor structs; aliased back onto the
//! f32 and 16-bit branches in ../tensor.zig. Every op runs the exec entry on
//! the stored dtype (the dtype policy in `dtype.zig` decides the compute
//! dtype); on the f32 branch the result joins the graph, on the 16-bit
//! branches it is a constant and a grad-requiring operand is rejected with
//! `error.UnsupportedGradient`. The ops that stay f32-only (in-place
//! updates, the graph-side casts, dropout, the channel ops, the elemental
//! escape hatch) are aliased on the f32 branch only.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const tags_mod = @import("../../tags.zig");
const backward_elementwise = @import("../backward/elementwise.zig");
const elemental = @import("../elemental.zig");

const RawTensor = tensor_mod.Tensor;
const DType = tensor_mod.DType;
const GradState = @import("../core.zig").GradState;
const ExecContext = exec_mod.ExecContext;
const UnaryOp = exec_mod.UnaryOp;
const GatedOp = exec_mod.GatedOp;
const Tag = tags_mod.Tag;
const replaceTag = tags_mod.replaceTag;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const CastBackward = backward_elementwise.CastBackward;
const IdentityBackward = backward_elementwise.IdentityBackward;
const ReluBackward = backward_elementwise.ReluBackward;
const PreluChannelsBackward = backward_elementwise.PreluChannelsBackward;
const ChannelAffineBackward = backward_elementwise.ChannelAffineBackward;
const LeakyReluBackward = backward_elementwise.LeakyReluBackward;
const SoftcapBackward = backward_elementwise.SoftcapBackward;
const UnaryBackward = backward_elementwise.UnaryBackward;
const unaryUsesOutput = backward_elementwise.unaryUsesOutput;
const ScaleBackward = backward_elementwise.ScaleBackward;
const AddScalarBackward = backward_elementwise.AddScalarBackward;
const PowScalarBackward = backward_elementwise.PowScalarBackward;
const MaskedFillBackward = backward_elementwise.MaskedFillBackward;
const WhereBackward = backward_elementwise.WhereBackward;
const DropoutBackward = backward_elementwise.DropoutBackward;
const ClampBackward = backward_elementwise.ClampBackward;
const SplitSwiGluBackward = backward_elementwise.SplitSwiGluBackward;
const SplitGluBackward = backward_elementwise.SplitGluBackward;
const SnakeBackward = backward_elementwise.SnakeBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tensor_rank = Self.tensor_rank;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const plumbing = @import("plumbing.zig").Mod(ag_tensor);
        const recordsGrad = plumbing.recordsGrad;
        const finishTypedNoGrad = plumbing.finishTypedNoGrad;
        const rawShapeArray = plumbing.rawShapeArray;
        const rawShapeArrayOf = plumbing.rawShapeArrayOf;
        const cloneInverseRopeTable = plumbing.cloneInverseRopeTable;
        const pointwise = plumbing.pointwise;
        const gatedPointwise = plumbing.gatedPointwise;
        const finishOp = plumbing.finishOp;
        const finishNoGrad = plumbing.finishNoGrad;
        const TensorObject = plumbing.TensorObject;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;
        const typedFinishOp = plumbing.typedFinishOp;
        const finishTyped = plumbing.finishTyped;
        const dtype = Self.dtype;
        const RawT = tensor_mod.TensorOf(dtype);
        /// The f32 branch is the differentiable one; every other dtype takes
        /// the constant tail.
        const differentiable = dtype == .f32;

        fn Out(comptime result_tags: anytype) type {
            return Tensor(.{ .dtype = dtype, .tags = result_tags });
        }

        /// In-place: add the `[axis_dim]` `bias` to every row of `self` along the
        /// last axis `axis_tag`, mutating `self`.
        pub fn addAxisVectorInPlace(self: *Self, ctx: *ExecContext, bias: []const f32, comptime axis_tag: Tag) !void {
            if (self.requiresGrad()) return error.UnsupportedGradient;
            try ctx.addAxisVectorInPlace(tensor_rank, null, &self.value, bias, comptime Self.axis(axis_tag));
        }

        /// In-place fused bias-add + unary activation `op`, mutating `self`.
        pub fn addAxisVectorUnaryInPlace(self: *Self, ctx: *ExecContext, comptime op: UnaryOp, bias: []const f32, comptime axis_tag: Tag) !void {
            if (self.requiresGrad()) return error.UnsupportedGradient;
            try ctx.addAxisVectorInPlace(tensor_rank, op, &self.value, bias, comptime Self.axis(axis_tag));
        }

        /// In-place scaled residual `self += alpha · other` (same shape).
        pub fn addScaledInPlace(self: *Self, ctx: *ExecContext, other: anytype, alpha: f32) !void {
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            if (self.requiresGrad() or other_ptr.requiresGrad()) return error.UnsupportedGradient;
            try ctx.addScaledInPlace(&self.value, other_ptr.asRawTensor(), alpha);
        }

        /// Out-of-place `self + bias` (a constant `[axis_dim]` slice broadcast
        /// along the last axis `axis_tag`) as a NEW tensor; `self` is
        /// unchanged. Differentiable in `self` (identity VJP: the slice bias
        /// is a constant and receives no gradient; for a trainable bias use
        /// `add` with a broadcast operand).
        pub fn biasAdd(self: *const Self, ctx: *ExecContext, bias: []const f32, comptime axis_tag: Tag) !Self {
            var value = try self.value.clone(ctx.allocator());
            errdefer value.deinit();
            try ctx.addAxisVectorInPlace(tensor_rank, null, &value, bias, comptime Self.axis(axis_tag));
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = IdentityBackward;
            return finishOp(tags, ctx, value, Record{ .parents = .{self.grad_state} });
        }

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
            var value = try ctx.where(dtype, Cond.dtype, self.asRawTensor(), cond.asRawTensor(), other.asRawTensor());
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad() or other.requiresGrad());
            if (!recordsGrad(self.requiresGrad() or other.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = WhereBackward(Cond.dtype);
            var saved_cond = try cond.asRawTensor().cloneView();
            errdefer saved_cond.deinit();
            return finishOp(tags, ctx, value, Record{ .parents = .{ self.grad_state, other.grad_state }, .cond = saved_cond });
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
            var v = try ctx.maskedFill(dtype, Mask.dtype, self.asRawTensor(), mask.asRawTensor(), value);
            errdefer v.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, v, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, v);
            const Record = MaskedFillBackward(Mask.dtype);
            var saved_mask = try mask.asRawTensor().cloneView();
            errdefer saved_mask.deinit();
            return finishOp(tags, ctx, v, Record{ .parents = .{self.grad_state}, .mask = saved_mask });
        }

        /// Elementwise comparison: a same-tagged `.bool` mask (torch's
        /// comparison dtype), true where `self <op> other` holds. `other`
        /// is comptime-dispatched from its type: a same-tagged tensor (same
        /// shape only, like `where`) or a numeric scalar (see
        /// `exec.CompareOp`). Non-differentiable; under an exec scope it is a
        /// borrow like every op result. NaN semantics
        /// are IEEE: any comparison involving NaN is false, except `.ne`,
        /// which is true. Feed the result to `where`/`maskedFill`/the
        /// logical ops, count with `sum`, or cast with `to(.f32)` for the
        /// mask-multiply idiom.
        pub fn compare(self: *const Self, ctx: *ExecContext, comptime op: exec_mod.CompareOp, other: anytype) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            const BoolT = Tensor(.{ .dtype = .bool, .tags = tags });
            const OtherT = @TypeOf(other);
            if (comptime (OtherT == comptime_float or OtherT == comptime_int or @typeInfo(OtherT) == .float or @typeInfo(OtherT) == .int)) {
                var value = try ctx.compareScalar(dtype, op, self.asRawTensor(), other);
                errdefer value.deinit();
                return finishTyped(BoolT, ctx, value);
            }
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            comptime {
                if (TensorObject(@TypeOf(other)).dtype != dtype) @compileError("compare requires matching dtypes; cast explicitly");
            }
            var value = try ctx.compare(dtype, op, self.asRawTensor(), other_ptr.asRawTensor());
            errdefer value.deinit();
            return finishTyped(BoolT, ctx, value);
        }

        /// Elementwise logical AND over truthiness (the mask convention
        /// shared with `where`/`maskedFill`; NaN is truthy): a same-tagged
        /// `.bool` tensor (torch's logical-op dtype). `other` may be a
        /// float or `.bool` tensor. Same shape only; non-differentiable
        /// like `compare`.
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
            var value = try ctx.logical(op, dtype, Other.dtype, self.asRawTensor(), other_ptr.asRawTensor());
            errdefer value.deinit();
            return finishTyped(Tensor(.{ .dtype = .bool, .tags = tags }), ctx, value);
        }

        /// Elementwise logical NOT over truthiness (see `logicalAnd`):
        /// a `.bool` tensor, true where `self` is zero.
        pub fn logicalNot(self: *const Self, ctx: *ExecContext) !Tensor(.{ .dtype = .bool, .tags = tags }) {
            var value = try ctx.logicalNot(dtype, self.asRawTensor());
            errdefer value.deinit();
            return finishTyped(Tensor(.{ .dtype = .bool, .tags = tags }), ctx, value);
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

        /// PReLU with a learnable per-channel slope (`alpha` rank-1 `[C]`, the
        /// channel axis innermost): `y = x > 0 ? x : α[c]·x`, one fused pass;
        /// differentiable in both `x` and `α`.
        pub fn prelu(self: *const Self, ctx: *ExecContext, alpha: anytype) !Self {
            const alpha_ptr = tensorObjectPtrFrom(@TypeOf(alpha), &alpha);
            const any_grad = self.requiresGrad() or alpha_ptr.requiresGrad();
            var value = try ctx.preluChannels(self.asRawTensor(), alpha_ptr.asRawTensor());
            errdefer value.deinit();
            if (!recordsGrad(any_grad)) return finishNoGrad(tags, ctx, value);
            var saved_input = try self.asRawTensor().cloneView();
            errdefer saved_input.deinit();
            var saved_alpha = try alpha_ptr.asRawTensor().cloneView();
            errdefer saved_alpha.deinit();
            return finishOp(tags, ctx, value, PreluChannelsBackward{
                .parents = .{ self.grad_state, alpha_ptr.grad_state },
                .channels = alpha_ptr.asRawTensor().len(),
                .input_value = saved_input,
                .alpha_value = saved_alpha,
            });
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
            if (!recordsGrad(any_grad)) return finishNoGrad(tags, ctx, value);
            var saved_input = try self.asRawTensor().cloneView();
            errdefer saved_input.deinit();
            var saved_scale = try scale_ptr.asRawTensor().cloneView();
            errdefer saved_scale.deinit();
            return finishOp(tags, ctx, value, ChannelAffineBackward{
                .parents = .{ self.grad_state, scale_ptr.grad_state, shift_ptr.grad_state },
                .channels = scale_ptr.asRawTensor().len(),
                .input_value = saved_input,
                .scale_value = saved_scale,
            });
        }

        pub fn to(self: *const Self, ctx: *ExecContext, comptime target_dtype: DType) !Tensor(.{ .dtype = target_dtype, .tags = tags }) {
            if (comptime (target_dtype != .f32 and target_dtype != .f16 and target_dtype != .bf16)) {
                if (self.requiresGrad()) return error.GradientCastUnsupported;
            }
            var value = try ctx.cast(.f32, target_dtype, self.asRawTensor());
            errdefer value.deinit();
            if (comptime target_dtype == .f32) {
                if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
                const Record = CastBackward;
                return finishOp(tags, ctx, value, Record{ .parents = .{self.grad_state} });
            }
            if (comptime (target_dtype == .f16 or target_dtype == .bf16)) {
                // Differentiable narrow (the mixed-precision seam): the
                // backward is the identity in f32 gradient space — the
                // upstream f32 gradient passes through unrounded.
                if (!recordsGrad(self.requiresGrad())) return finishTyped(Tensor(.{ .dtype = target_dtype, .tags = tags }), ctx, value);
                const Record = CastBackward;
                return typedFinishOp(target_dtype, tags, ctx, value, Record{ .parents = .{self.grad_state} });
            }
            return finishTyped(Tensor(.{ .dtype = target_dtype, .tags = tags }), ctx, value);
        }

        pub fn add(self: *const Self, ctx: *ExecContext, other: anytype) !Out(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.add, self, ctx, other);
        }

        pub fn sub(self: *const Self, ctx: *ExecContext, other: anytype) !Out(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.sub, self, ctx, other);
        }

        pub fn mul(self: *const Self, ctx: *ExecContext, other: anytype) !Out(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.mul, self, ctx, other);
        }

        pub fn div(self: *const Self, ctx: *ExecContext, other: anytype) !Out(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.div, self, ctx, other);
        }

        /// `self * scalar_value`; the scalar is the dtype's accumulator type
        /// (f32 for the 16-bit branches).
        pub fn scale(self: *const Self, ctx: *ExecContext, scalar_value: dtype_mod.Accumulator(dtype)) !Self {
            var value = try ctx.scale(dtype, self.asRawTensor(), scalar_value);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = ScaleBackward;
            return finishOp(tags, ctx, value, Record{ .parents = .{self.grad_state}, .scalar_value = scalar_value });
        }

        /// Consume `self` and return `self + other`, reusing `self`'s storage
        /// when the runtime can safely take it in place. No-grad only: consuming
        /// a graph value would invalidate autograd state.
        pub fn takeAddNoGrad(self: *Self, ctx: *ExecContext, other: anytype) !Self {
            const other_ptr = tensorObjectPtrFrom(@TypeOf(other), &other);
            if (self.requiresGrad() or other_ptr.requiresGrad()) return error.UnsupportedGradient;
            if (self.scope_owned) return error.ActiveExecScopeUnsupported;
            var value = try ctx.takeElementwise(.add, &self.value, other_ptr.asRawTensor());
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
            var value = try ctx.addScalar(dtype, self.asRawTensor(), scalar_value);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = AddScalarBackward;
            return finishOp(tags, ctx, value, Record{ .parents = .{self.grad_state} });
        }

        /// `self - scalar_value` (= `addScalar(-scalar_value)`).
        pub fn subScalar(self: *const Self, ctx: *ExecContext, scalar_value: f32) !Self {
            return self.addScalar(ctx, -scalar_value);
        }

        /// `self / scalar_value` (= `scale(1/scalar_value)`).
        pub fn divScalar(self: *const Self, ctx: *ExecContext, scalar_value: dtype_mod.Accumulator(dtype)) !Self {
            return self.scale(ctx, 1.0 / scalar_value);
        }

        /// `self ^ exponent` (elementwise; defined for positive `self`).
        /// Differentiable: `d/dx x^c = c·x^(c-1)`.
        pub fn powScalar(self: *const Self, ctx: *ExecContext, exponent: f32) !Self {
            var value = try ctx.powScalar(dtype, self.asRawTensor(), exponent);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = PowScalarBackward;
            var saved_input = try self.asRawTensor().cloneView();
            errdefer saved_input.deinit();
            return finishOp(tags, ctx, value, Record{
                .parents = .{self.grad_state},
                .input = saved_input,
                .exponent = exponent,
            });
        }

        /// `log(1 + self)` (elementwise). Differentiable: `d/dx = 1/(1+x)`.
        pub fn log1p(self: *const Self, ctx: *ExecContext) !Self {
            return unaryDifferentiable(self, ctx, .log1p);
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
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = DropoutBackward;
            return finishOp(tags, ctx, value, Record{
                .parents = .{self.grad_state},
                .p = p,
                .seed = seed,
            });
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
            if (!recordsGrad(self.requiresGrad() or alpha.requiresGrad() or inv_b.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = SnakeBackward;
            var saved_input = try self.asRawTensor().cloneView();
            errdefer saved_input.deinit();
            var saved_alpha = try alpha.asRawTensor().cloneView();
            errdefer saved_alpha.deinit();
            var saved_inv_b = try inv_b.asRawTensor().cloneView();
            errdefer saved_inv_b.deinit();
            return finishOp(tags, ctx, value, Record{
                .parents = .{ self.grad_state, alpha.grad_state, inv_b.grad_state },
                .input_value = saved_input,
                .alpha_value = saved_alpha,
                .inv_b_value = saved_inv_b,
            });
        }

        pub fn gated(
            self: *const Self,
            ctx: *ExecContext,
            other: anytype,
            comptime op: GatedOp,
        ) !Out(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return gatedPointwise(op, self, ctx, other);
        }

        pub fn glu(
            self: *const Self,
            ctx: *ExecContext,
            other: anytype,
        ) !Out(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return self.gated(ctx, other, .glu);
        }

        pub fn swiglu(
            self: *const Self,
            ctx: *ExecContext,
            other: anytype,
        ) !Out(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return self.gated(ctx, other, .swiglu);
        }

        /// GeGLU: `self * gelu(other)` (GELU tanh approximation). Mirrors
        /// `swiglu` but with the GELU gate Gemma's GeGLU FFN uses.
        pub fn geglu(
            self: *const Self,
            ctx: *ExecContext,
            other: anytype,
        ) !Out(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return self.gated(ctx, other, .geglu);
        }

        /// SiTU (Kimi K3's gated activation):
        /// `25·tanh(self/25) * 4·tanh(other/4)·sigmoid(other)` — `self` is
        /// the up input (soft-clamped, linear beta 25), `other` the gate
        /// (soft-bounded SiLU, beta 4). Mirrors `swiglu`'s operand order
        /// (`up.situ(&ctx, &gate)`). Differentiable in both operands.
        pub fn situ(
            self: *const Self,
            ctx: *ExecContext,
            other: anytype,
        ) !Out(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return self.gated(ctx, other, .situ);
        }

        /// Split-gated activation over `tag`: halves that axis and gates one
        /// half with the other, per `op` — the gate-half conventions DIFFER
        /// (ggml parity): `.swiglu` gates with the FIRST half
        /// (`silu(first) * second`), `.glu` with the SECOND
        /// (`first * sigmoid(second)`). `out_tag == tag` is allowed (keeps
        /// the tag on the halved axis — the raw numeric-tag form Parakeet
        /// uses). `.geglu` is a compile error: no split-geglu kernel or
        /// gate-half convention exists.
        pub fn splitGated(self: *const Self, ctx: *ExecContext, comptime op: GatedOp, comptime tag: Tag, comptime out_tag: Tag) !Out(replaceTag(tags, tag, out_tag)) {
            const result_tags = replaceTag(tags, tag, out_tag);
            const split_axis = comptime axis(tag);
            const Backward = switch (comptime op) {
                .swiglu => SplitSwiGluBackward(tags, split_axis),
                .glu => SplitGluBackward(tags, split_axis),
                .geglu => @compileError("splitGated: no split-geglu kernel or gate-half convention exists (compose `unary(.gelu_quant)` + `mul`, or use `geglu` on separate halves)"),
                .situ => @compileError("splitGated: no split-situ kernel (K3 projects gate and up separately; use the pointwise `situ`)"),
            };
            var value = try ctx.splitGated(dtype, tag_rank, op, self.asRawTensor(), split_axis);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(result_tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            var saved_input = try self.asRawTensor().cloneView();
            errdefer saved_input.deinit();
            return finishOp(result_tags, ctx, value, Backward{ .parents = .{self.grad_state}, .input = saved_input });
        }

        pub fn unary(self: *const Self, ctx: *ExecContext, comptime op: UnaryOp) !Self {
            return switch (op) {
                .relu => self.relu(ctx),
                .exp, .sqrt, .rsqrt, .sigmoid, .silu, .log, .log1p, .softplus, .neg, .abs, .sin, .cos, .tanh, .fast_tanh, .gelu, .quick_gelu, .gelu_quant, .elu, .gelu_erf, .erf, .floor, .ceil, .round, .sign, .reciprocal => unaryDifferentiable(self, ctx, op),
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
        pub fn maximum(self: *const Self, ctx: *ExecContext, other: anytype) !Out(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.max, self, ctx, other);
        }

        /// Elementwise minimum of two tensors (torch.minimum); see
        /// `maximum` for the NaN, tie-gradient, and kernel-tier notes.
        pub fn minimum(self: *const Self, ctx: *ExecContext, other: anytype) !Out(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
            return pointwise(.min, self, ctx, other);
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
            var value = try ctx.unary(dtype, .relu, self.asRawTensor());
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            var saved_input = try (&self.value).cloneView();
            errdefer saved_input.deinit();
            return finishOp(tags, ctx, value, ReluBackward{ .parents = .{self.grad_state}, .input = saved_input });
        }

        pub fn leakyRelu(self: *const Self, ctx: *ExecContext, negative_slope: f32) !Self {
            var value = try ctx.leakyRelu(dtype, self.asRawTensor(), negative_slope);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            var saved_input = try (&self.value).cloneView();
            errdefer saved_input.deinit();
            return finishOp(tags, ctx, value, LeakyReluBackward{
                .parents = .{self.grad_state},
                .input = saved_input,
                .negative_slope = negative_slope,
            });
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

        pub const gelu = UnaryMethod(.gelu).call;

        pub const quickGelu = UnaryMethod(.quick_gelu).call;

        pub const elu = UnaryMethod(.elu).call;

        pub const geluErf = UnaryMethod(.gelu_erf).call;

        pub const erf = UnaryMethod(.erf).call;

        pub const floor = UnaryMethod(.floor).call;

        pub const ceil = UnaryMethod(.ceil).call;

        pub const round = UnaryMethod(.round).call;

        pub const sign = UnaryMethod(.sign).call;

        pub const reciprocal = UnaryMethod(.reciprocal).call;

        fn UnaryMethod(comptime op: UnaryOp) type {
            return struct {
                fn call(self: *const Self, ctx: *ExecContext) !Self {
                    return unaryDifferentiable(self, ctx, op);
                }
            };
        }

        /// `cap * tanh(self / cap)`, the logit softcap (`cap > 0`).
        /// Differentiable: the VJP reads the output, `1 - (y/cap)^2`.
        pub fn softcap(self: *const Self, ctx: *ExecContext, cap: f32) !Self {
            var value = try ctx.softcap(dtype, self.asRawTensor(), cap);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            var saved_output = try (&value).cloneView();
            errdefer saved_output.deinit();
            return finishOp(tags, ctx, value, SoftcapBackward{
                .parents = .{self.grad_state},
                .output = saved_output,
                .cap = cap,
            });
        }

        pub fn clamp(self: *const Self, ctx: *ExecContext, min_value: f32, max_value: f32) !Self {
            var value = try ctx.clamp(dtype, self.asRawTensor(), min_value, max_value);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            var saved_input = try (&self.value).cloneView();
            errdefer saved_input.deinit();
            return finishOp(tags, ctx, value, ClampBackward{
                .parents = .{self.grad_state},
                .input = saved_input,
                .min_value = min_value,
                .max_value = max_value,
            });
        }

        /// One-sided clamp (torch's `clamp_min`): `max(x, min_value)`.
        /// The open side is +inf, so the kernel and `ClampBackward`
        /// (gradient zeroed only where `x < min_value`) apply unchanged.
        pub fn clampMin(self: *const Self, ctx: *ExecContext, min_value: f32) !Self {
            return self.clamp(ctx, min_value, std.math.inf(f32));
        }

        /// One-sided clamp (torch's `clamp_max`): `min(x, max_value)`.
        pub fn clampMax(self: *const Self, ctx: *ExecContext, max_value: f32) !Self {
            return self.clamp(ctx, -std.math.inf(f32), max_value);
        }

        fn unaryDifferentiable(self: *const Self, ctx: *ExecContext, comptime op: UnaryOp) !Self {
            var value = try ctx.unary(dtype, op, self.asRawTensor());
            errdefer value.deinit();
            // Output-derivative ops (see backward_elementwise.unaryUsesOutput) store the
            // OUTPUT view: their VJP is transcendental-free in t (tanh' = 1-t²)
            // and exact for the value the SIMD forward actually produced.
            const saved: *const RawT = if (comptime unaryUsesOutput(op)) &value else &self.value;
            if (comptime !differentiable) return finishTypedNoGrad(Out(tags), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = UnaryBackward(op);
            var saved_input = try saved.cloneView();
            errdefer saved_input.deinit();
            return finishOp(tags, ctx, value, Record{ .parents = .{self.grad_state}, .input = saved_input });
        }
    };
}
