//! f32 tensor methods: loss heads. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const tags_mod = @import("../../../tags.zig");
const backward_loss = @import("../../backward/loss.zig");

const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const Tag = tags_mod.Tag;
const removeTag = tags_mod.removeTag;
const CrossEntropyExtBackward = backward_loss.CrossEntropyExtBackward;
const LinearCrossEntropyBackward = backward_loss.LinearCrossEntropyBackward;
const MseLossBackward = backward_loss.MseLossBackward;
const HuberLossBackward = backward_loss.HuberLossBackward;
const BceLossBackward = backward_loss.BceLossBackward;
const KlDivLossBackward = backward_loss.KlDivLossBackward;

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
        const recordsGrad = plumbing.recordsGrad;
        const finishTypedNoGrad = plumbing.finishTypedNoGrad;
        const finishNoGrad = plumbing.finishNoGrad;
        const rawShapeArray = plumbing.rawShapeArray;
        const rawShapeArrayOf = plumbing.rawShapeArrayOf;
        const cloneInverseRopeTable = plumbing.cloneInverseRopeTable;
        const rowStatsAlloc = plumbing.rowStatsAlloc;
        const finishOp = plumbing.finishOp;
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
            defer if (row_stats) |stats| ctx.allocator().free(stats);
            var stats_options = options;
            stats_options.row_stats = row_stats;
            var value = try ctx.crossEntropyLoss(tag_rank, self.asRawTensor(), class_axis, labels, stats_options);
            errdefer value.deinit();
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            const Record = CrossEntropyExtBackward(tags, class_axis, options);
            var saved_logits = try self.asRawTensor().cloneView();
            errdefer saved_logits.deinit();
            const owned_labels = try ctx.allocator().dupe(usize, labels);
            errdefer ctx.allocator().free(owned_labels);
            const owned_row_stats = try ctx.allocator().dupe(f32, row_stats orelse &[_]f32{});
            errdefer ctx.allocator().free(owned_row_stats);
            return finishOp(result_tags, ctx, value, Record{
                .parents = .{self.grad_state},
                .logits = saved_logits,
                .labels = owned_labels,
                .row_stats = owned_row_stats,
            });
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
            defer if (row_stats) |stats| ctx.allocator().free(stats);
            var logits = try ctx.matmul(.f32, .trans_b, self.asRawTensor(), weight_ptr.asRawTensor());
            defer logits.deinit();
            var stats_options = options;
            stats_options.row_stats = row_stats;
            var value = try ctx.crossEntropyLoss(2, &logits, 1, labels, stats_options);
            errdefer value.deinit();
            if (!recordsGrad(wants_grad)) return finishNoGrad(result_tags, ctx, value);
            const Record = LinearCrossEntropyBackward(options);
            var saved_x = try self.asRawTensor().cloneView();
            errdefer saved_x.deinit();
            var saved_weight = try weight_ptr.asRawTensor().cloneView();
            errdefer saved_weight.deinit();
            var saved_logits = try (&logits).cloneView();
            errdefer saved_logits.deinit();
            const owned_labels = try ctx.allocator().dupe(usize, labels);
            errdefer ctx.allocator().free(owned_labels);
            const owned_row_stats = try ctx.allocator().dupe(f32, row_stats orelse &[_]f32{});
            errdefer ctx.allocator().free(owned_row_stats);
            return finishOp(result_tags, ctx, value, Record{
                .parents = .{ self.grad_state, weight_ptr.grad_state },
                .x = saved_x,
                .weight = saved_weight,
                .logits = saved_logits,
                .labels = owned_labels,
                .row_stats = owned_row_stats,
                .estimated_work = Record.workEstimate(self.grad_state, weight_ptr.grad_state, self.asRawTensor(), weight_ptr.asRawTensor()),
            });
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
            if (!recordsGrad(self.requiresGrad() or target.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            const Record = MseLossBackward(tags, options);
            var saved_input = try self.asRawTensor().cloneView();
            errdefer saved_input.deinit();
            var saved_target = try target.asRawTensor().cloneView();
            errdefer saved_target.deinit();
            return finishOp(result_tags, ctx, value, Record{
                .parents = .{ self.grad_state, target.grad_state },
                .input = saved_input,
                .target = saved_target,
            });
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
            if (!recordsGrad(self.requiresGrad() or target.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            const Record = HuberLossBackward(tags, options);
            var saved_input = try self.asRawTensor().cloneView();
            errdefer saved_input.deinit();
            var saved_target = try target.asRawTensor().cloneView();
            errdefer saved_target.deinit();
            return finishOp(result_tags, ctx, value, Record{
                .parents = .{ self.grad_state, target.grad_state },
                .input = saved_input,
                .target = saved_target,
            });
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
            if (!recordsGrad(self.requiresGrad() or target.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            const Record = BceLossBackward(tags, options);
            var saved_input = try self.asRawTensor().cloneView();
            errdefer saved_input.deinit();
            var saved_target = try target.asRawTensor().cloneView();
            errdefer saved_target.deinit();
            return finishOp(result_tags, ctx, value, Record{
                .parents = .{ self.grad_state, target.grad_state },
                .input = saved_input,
                .target = saved_target,
            });
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
            if (!recordsGrad(self.requiresGrad() or target.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            const Record = KlDivLossBackward(tags, options);
            var saved_input = try self.asRawTensor().cloneView();
            errdefer saved_input.deinit();
            var saved_target = try target.asRawTensor().cloneView();
            errdefer saved_target.deinit();
            return finishOp(result_tags, ctx, value, Record{
                .parents = .{ self.grad_state, target.grad_state },
                .input = saved_input,
                .target = saved_target,
            });
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
        /// log-probabilities.
        pub fn nllLoss(
            self: *const Self,
            ctx: *ExecContext,
            comptime class_tag: Tag,
            labels: []const usize,
            comptime reduction: exec_mod.Reduction,
        ) !Tensor(if (reduction == .none) removeTag(tags, class_tag) else .{}) {
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
