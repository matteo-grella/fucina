//! f32 tensor methods: loss heads. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const tags_mod = @import("../../../tags.zig");
const backward = @import("../../backward.zig");

const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const Tag = tags_mod.Tag;
const removeTag = tags_mod.removeTag;
const CrossEntropyExtBackward = backward.CrossEntropyExtBackward;
const LinearCrossEntropyBackward = backward.LinearCrossEntropyBackward;
const LinearDistillBackward = backward.LinearDistillBackward;
const MseLossBackward = backward.MseLossBackward;
const HuberLossBackward = backward.HuberLossBackward;
const BceLossBackward = backward.BceLossBackward;
const KlDivLossBackward = backward.KlDivLossBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tensor_rank = Self.tensor_rank;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const rowStatsAlloc = plumbing.rowStatsAlloc;
        const finishOp = plumbing.finishOp;
        const requireScopeForComposedGrad = plumbing.requireScopeForComposedGrad;
        const TensorObject = plumbing.TensorObject;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;

        /// Cross-entropy over `class_tag` with PyTorch-parity options
        /// (ignore_index, reduction, label smoothing); `options` is `.{}`
        /// for the defaults. `.mean`/`.sum` return a scalar; `.none` returns
        /// per-position losses with `class_tag` removed (same tag-removal
        /// rule as `sum`/`mean`).
        pub fn crossEntropy(
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
            var stats_options = options;
            stats_options.row_stats = row_stats;
            var value = try ctx.crossEntropyLoss(tag_rank, self.asRawTensor(), class_axis, labels, stats_options);
            errdefer value.deinit();
            return finishOp(result_tags, ctx, value, self.requiresGrad(), CrossEntropyExtBackward(tags, class_axis, options), .{ ctx.allocator, self.grad_state, self.asRawTensor(), labels, row_stats orelse &[_]f32{} });
        }

        /// Fused linear + cross-entropy: `crossEntropy(self·weightᵀ)` as
        /// ONE differentiable op. `self` is [row, shared] and `weight` is
        /// [class, shared] (both rank-2, shared tag last, f32). The logits
        /// exist only inside the op — computed once and saved on the
        /// backward record with the forward's per-row softmax statistics —
        /// and the VJP folds block-built probability panels straight into
        /// dx and dweight, so the [rows, classes] logit GRADIENT is never
        /// materialized (see `linearCrossEntropyBackwardUpstream`).
        /// Differentiable in BOTH operands. Reduction contract as
        /// `crossEntropy`: `.mean`/`.sum` return a scalar, `.none` the
        /// per-row losses tagged by the row tag.
        pub fn linearCrossEntropy(
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
            var stats_options = options;
            stats_options.row_stats = row_stats;
            var value = try ctx.crossEntropyLoss(2, &logits, 1, labels, stats_options);
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

        /// Fused linear + sparse-soft-target distillation loss:
        /// cross-entropy of `self·weightᵀ` against per-entry
        /// (row, class, prob) soft targets, as ONE differentiable op —
        /// `loss = reduce_i probs[i]·(LSE(logits[rows[i]]) −
        /// logits[rows[i], classes[i]])`. Only the UNIQUE rows named by
        /// `rows` are projected (rows without entries never produce
        /// logits), the selected-row logits live solely on the backward
        /// record with the forward's per-row softmax statistics, and the
        /// VJP consumes them in place — the full [row, class] block never
        /// enters the graph and its gradient never costs a second buffer.
        /// `self` is [row, shared] and `weight` [class, shared] (both
        /// rank-2, shared tag last, f32). `probs[i]` weights entry i (a
        /// teacher probability in the distillation use; truncated tail
        /// mass deliberately NOT renormalized). `options.reduction`
        /// reduces over ENTRIES and `options.loss_scale` multiplies the
        /// scalar result (the gradient-accumulation knob). Differentiable
        /// in BOTH operands; the record is single-use like
        /// `linearCrossEntropy`.
        pub fn linearDistill(
            self: *const Self,
            ctx: *ExecContext,
            weight: anytype,
            rows: []const usize,
            classes: []const usize,
            probs: []const f32,
            options: exec_mod.LinearDistillOptions,
        ) !Tensor(.{}) {
            const Other = TensorObject(@TypeOf(weight));
            comptime {
                if (tag_rank != 2 or Other.axis_tags.len != 2) @compileError("linearDistill requires rank-2 [row, shared] x and [class, shared] weight");
                if (Other.dtype != .f32) @compileError("linearDistill requires an f32 weight (quantized/f16 arms are not routed)");
                if (tags[1] != Other.axis_tags[1]) @compileError("linearDistill requires the shared tag LAST on both operands");
                if (Other.axis_tags[0] == tags[0] or Other.axis_tags[0] == tags[1]) @compileError("linearDistill weight class tag must not appear on x");
            }
            const weight_ptr = tensorObjectPtrFrom(@TypeOf(weight), &weight);
            const wants_grad = self.requiresGrad() or weight_ptr.requiresGrad();
            var fwd = try ctx.linearDistillLossStats(self.asRawTensor(), weight_ptr.asRawTensor(), rows, classes, probs, options);
            // finishOp takes ownership of the scalar; everything else is
            // cloned/duped onto the record and released here.
            defer {
                fwd.logits.deinit();
                fwd.x_sel.deinit();
                ctx.allocator.free(fwd.sel_rows);
                ctx.allocator.free(fwd.local_rows);
                ctx.allocator.free(fwd.row_stats);
            }
            errdefer fwd.value.deinit();
            return finishOp(
                .{},
                ctx,
                fwd.value,
                wants_grad,
                LinearDistillBackward,
                .{
                    ctx.allocator,
                    self.grad_state,
                    weight_ptr.grad_state,
                    &fwd.x_sel,
                    weight_ptr.asRawTensor(),
                    &fwd.logits,
                    fwd.sel_rows,
                    fwd.local_rows,
                    classes,
                    probs,
                    fwd.row_stats,
                    self.asRawTensor().shape.at(0),
                    options,
                },
            );
        }

        /// Mean-squared-error loss vs a same-tagged `target` (torch F.mse_loss):
        /// per-element (x - t)². `.mean` (the default) divides by the TOTAL
        /// element count; `.none` returns input-shaped per-element losses
        /// (same reduction-dependent result type as `crossEntropy`).
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
        /// `crossEntropy` (fused log-softmax + NLL) when
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

            // Sparse marks into a zeroed tensor — the `oneHot` constructor's
            // shape; a dense scratch here would copy the full logits size.
            var one_hot = try Self.zeros(ctx, raw_shape);
            defer one_hot.deinit();
            const one_hot_values = try one_hot.data();
            for (0..outer) |outer_i| {
                for (0..inner) |inner_i| {
                    const label = labels[outer_i * inner + inner_i];
                    if (label >= class_count) return TensorError.IndexOutOfBounds;
                    one_hot_values[(outer_i * class_count + label) * inner + inner_i] = 1;
                }
            }

            var picked = try self.mul(ctx, &one_hot);
            defer picked.deinit();
            var picked_sum = try picked.sum(ctx, class_tag, .{});
            defer picked_sum.deinit();
            if (comptime reduction == .none) {
                return picked_sum.neg(ctx);
            }
            var total = try picked_sum.sumAll(ctx);
            defer total.deinit();
            const denom: f32 = if (comptime reduction == .mean) @floatFromInt(position_count) else 1;
            return total.scale(ctx, -1.0 / denom);
        }
    };
}
