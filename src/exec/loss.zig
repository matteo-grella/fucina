//! Loss forwards + VJPs: cross-entropy (PyTorch-parity options) and the
//! whole-tensor elementwise losses (MSE / Huber / BCE / KL divergence).
//!
//! Domain module: every op receives an explicit `*ExecContext`; per-row kernels +
//! Task structs live in the backend row-kernel leaf (`backend.rows`). Home of `CrossEntropyOptions` +
//! `Reduction` + the elementwise-loss options (re-exported by `exec.zig`).
//! Softmax is deliberately NOT folded in (no shared dispatch code).

const std = @import("std");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

const exec_matmul = @import("matmul.zig");
const backend_mod = @import("../backend.zig");
const exec_row_ops = backend_mod.rows;
const shape_mod = @import("../shape.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const Tensor = tensor.Tensor;

const productAfterAxis = shape_mod.productAfterAxis;
const productBeforeAxis = shape_mod.productBeforeAxis;
const shapeWithoutAxis = shape_mod.withoutAxis;

const CrossEntropyLossRowsTask = exec_row_ops.CrossEntropyLossRowsTask;
const CrossEntropyBackwardRowsTask = exec_row_ops.CrossEntropyBackwardRowsTask;
const crossEntropyLossRows = backend_mod.kernels.crossEntropyLossRows;
const crossEntropyBackwardRows = backend_mod.kernels.crossEntropyBackwardRows;
const DistillStatsRowsTask = exec_row_ops.DistillStatsRowsTask;
const DistillBackwardRowsTask = exec_row_ops.DistillBackwardRowsTask;
const distillStatsRows = backend_mod.kernels.distillStatsRows;
const distillBackwardRows = backend_mod.kernels.distillBackwardRows;

pub const Reduction = enum { mean, sum, none };

/// PyTorch-parity cross-entropy options for `crossEntropyLoss` /
/// `crossEntropyBackward`. One value drives both directions.
pub const CrossEntropyOptions = struct {
    /// A position whose label equals this index contributes zero loss and zero
    /// gradient and is excluded from the `.mean` denominator. Labels must be
    /// `< class_count` or equal to this index.
    ignore_index: ?usize = null,
    /// `.mean` divides by the count of non-ignored positions. Deliberate
    /// divergence from PyTorch: when every position is ignored the loss is 0
    /// (and gradients are zero) instead of NaN. `.none` returns per-position
    /// losses, shaped like the logits with the class axis removed (ignored
    /// positions get 0).
    reduction: Reduction = .mean,
    /// Label smoothing epsilon in [0, 1): the target distribution becomes
    /// (1-eps)*onehot + (eps/K) uniform over all K classes (PyTorch
    /// semantics, target class included in the uniform mass).
    label_smoothing: f32 = 0,
    /// Per-position softmax statistics slot: {max, sum_exp} interleaved f32
    /// pairs, length 2 * position count. `crossEntropyLoss` fills it when
    /// non-null (ignored positions get {0, 1}); `crossEntropyBackward` reads
    /// it when non-null and emits final gradients in ONE pass over the
    /// logits, bitwise identical to the recompute path (the stats are the
    /// exact f32 values the backward would recompute). Null on both sides is
    /// the plain two-pass backward.
    row_stats: ?[]f32 = null,
};

/// The upstream gradient `crossEntropyBackward` scales the VJP by.
pub const CrossEntropyUpstream = union(enum) {
    /// Scalar upstream for `.mean`/`.sum`.
    scale: f32,
    /// `.none` reduction: the per-position upstream gradient (length
    /// outer*inner, position order matching `labels`), additionally scaled
    /// by `scale`.
    rows: struct { per_row: []const f32, scale: f32 = 1 },
    /// The autograd form: a scalar tensor for `.mean`/`.sum`, the
    /// per-position tensor (logits shape with the class axis removed) for
    /// `.none`.
    tensor: *const Tensor,
};

/// Options for `mseLoss` (torch F.mse_loss semantics): per-element (x - t)².
pub const MseOptions = struct {
    /// `.mean` (the torch default) divides by the TOTAL element count;
    /// `.none` returns input-shaped per-element losses.
    reduction: Reduction = .mean,
};

/// Options for `huberLoss` (torch F.huber_loss semantics): per-element
/// 0.5·d² for |d| <= delta, delta·(|d| - 0.5·delta) otherwise (d = x - t).
pub const HuberOptions = struct {
    /// Quadratic/linear crossover; must be positive and finite.
    delta: f32 = 1.0,
    /// `.mean` divides by the TOTAL element count (torch default).
    reduction: Reduction = .mean,
};

/// Options for `bceLoss`. With `from_logits` this is torch
/// F.binary_cross_entropy_with_logits (unit weights), otherwise torch
/// F.binary_cross_entropy modulo the `bce_eps` clamp documented there.
pub const BceOptions = struct {
    /// `.mean` divides by the TOTAL element count (torch default).
    reduction: Reduction = .mean,
    /// When true the input is a raw logit and the loss uses the numerically
    /// stable formulation `max(x,0) - x·y + log1p(exp(-|x|))`. When false the
    /// input is a probability, clamped to [bce_eps, 1-bce_eps] before the
    /// logs (see `bce_eps`).
    from_logits: bool = false,
};

/// Options for `klDivLoss` (torch F.kl_div pointwise semantics): the INPUT is
/// log-probabilities; per-element `t·(ln t - x)` (with the xlogy convention
/// `0·ln 0 = 0`), or `exp(t)·(t - x)` when the target is a log-probability.
/// Targets are assumed valid ((log-)probabilities); they are not validated
/// per element, matching torch.
pub const KlDivOptions = struct {
    /// Deliberate divergence from torch F.kl_div: there is NO `.batchmean`.
    /// `.mean` divides by the TOTAL element count (torch's `.mean`, which
    /// torch itself warns is not the mathematical KL); `.sum` is the
    /// mathematical divergence; `.none` returns per-element terms.
    reduction: Reduction = .mean,
    /// When true, `target` holds LOG-probabilities (torch `log_target`).
    log_target: bool = false,
};

/// Which operand an elementwise-loss VJP differentiates. Both operands of
/// MSE/Huber/BCE/KL are differentiable (torch parity); the autograd VJP calls
/// the `*BackwardUpstream` arm once per operand that needs a gradient.
pub const LossWrt = enum { input, target };

/// Probability clamp for the non-logits `bceLoss` arm: p is clamped to
/// [bce_eps, 1-bce_eps] before `ln(p)`/`ln(1-p)`, and the input gradient is
/// defined as exactly 0 outside the open interval (the clamped forward is
/// locally constant there). Deliberate divergence from torch, which instead
/// clamps `ln` at -100 in the forward and the gradient denominator at 1e-12
/// (returning huge boundary gradients rather than 0).
pub const bce_eps: f32 = 1e-7;

/// Validates labels against `class_count` / `options.ignore_index` and
/// counts the non-ignored positions for the `.mean` denominator.
fn validateCrossEntropyLabels(labels: []const usize, position_count: usize, class_count: usize, options: CrossEntropyOptions) !usize {
    if (labels.len != position_count) return tensor.TensorError.InvalidDataLength;
    if (!(options.label_smoothing >= 0 and options.label_smoothing < 1)) return tensor.TensorError.InvalidArgument;
    var valid_count: usize = 0;
    for (labels) |label| {
        if (options.ignore_index) |ignore_index| {
            if (label == ignore_index) continue;
        }
        if (label >= class_count) return tensor.TensorError.IndexOutOfBounds;
        valid_count += 1;
    }
    return valid_count;
}

/// Cross-entropy forward with PyTorch-parity options (see
/// `CrossEntropyOptions`; `.{}` is the defaults). `.mean`/`.sum` return a
/// scalar; `.none` returns per-position losses shaped like the logits with
/// the class axis removed. The `.mean`/`.sum` reduction is one serial sum
/// over per-row losses in row order, so the result is bitwise identical
/// for any thread count. `options.row_stats`, when set, receives the
/// per-position softmax statistics the backward can reuse.
pub fn crossEntropyLoss(
    ctx: *ExecContext,
    comptime rank: usize,
    logits: *const Tensor,
    comptime axis: usize,
    labels: []const usize,
    options: CrossEntropyOptions,
) !Tensor {
    if (axis >= rank) @compileError("axis out of bounds");
    const row_stats = options.row_stats;

    const source = try logits.rankView(rank);
    const class_count = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const position_count = outer * inner;
    const valid_count = try validateCrossEntropyLabels(labels, position_count, class_count, options);
    if (row_stats) |stats| {
        if (stats.len != 2 * position_count) return tensor.TensorError.InvalidDataLength;
    }

    var ll = try ctx.prepareContiguous(.f32, logits);
    defer ll.deinit();
    const input = ll.tensor().dataConst();

    // Per-position losses (ignored positions get 0). For `.none` this is
    // the kernel output; for `.mean`/`.sum` it's a temporary reduced
    // serially below.
    const out_rank = if (rank == 1) 1 else rank - 1;
    var none_out: ?Tensor = if (options.reduction == .none)
        try ctx.empty(.f32, shapeWithoutAxis(rank, out_rank, source.shape, axis))
    else
        null;
    errdefer if (none_out) |*value| value.deinit();
    const owns_row_losses = none_out == null;
    const row_losses: []f32 = if (none_out) |*value| value.data() else try ctx.allocator().alloc(f32, position_count);
    defer if (owns_row_losses) ctx.allocator().free(row_losses);

    if (inner == 1) {
        const base_task: CrossEntropyLossRowsTask = .{
            .input = input,
            .labels = labels,
            .row_losses = row_losses,
            .row_stats = row_stats,
            .class_count = class_count,
            .ignore_index = options.ignore_index,
            .label_smoothing = options.label_smoothing,
            .row_start = 0,
            .row_end = outer,
        };
        ctx.dispatchRangeOr(CrossEntropyLossRowsTask, "row_start", "row_end", base_task, outer, outer > 1 and source.len() >= parallel.row_kernel_len_threshold, crossEntropyLossRows);
    } else {
        const eps = options.label_smoothing;
        const eps_uniform = eps / @as(f32, @floatFromInt(class_count));
        for (0..outer) |outer_i| {
            const base = outer_i * class_count * inner;
            for (0..inner) |inner_i| {
                const row = outer_i * inner + inner_i;
                const label = labels[row];
                if (options.ignore_index) |ignore_index| {
                    if (label == ignore_index) {
                        row_losses[row] = 0;
                        if (row_stats) |stats| {
                            stats[2 * row] = 0;
                            stats[2 * row + 1] = 1;
                        }
                        continue;
                    }
                }
                var max_value = input[base + inner_i];
                for (1..class_count) |class_i| {
                    max_value = @max(max_value, input[base + class_i * inner + inner_i]);
                }

                var sum_exp: f32 = 0;
                for (0..class_count) |class_i| {
                    sum_exp += @exp(input[base + class_i * inner + inner_i] - max_value);
                }
                if (row_stats) |stats| {
                    stats[2 * row] = max_value;
                    stats[2 * row + 1] = sum_exp;
                }
                var loss = @log(sum_exp) + max_value - (1 - eps) * input[base + label * inner + inner_i];
                if (eps > 0) {
                    var logit_sum: f32 = 0;
                    for (0..class_count) |class_i| {
                        logit_sum += input[base + class_i * inner + inner_i];
                    }
                    loss -= eps_uniform * logit_sum;
                }
                row_losses[row] = loss;
            }
        }
    }

    switch (options.reduction) {
        .none => {
            const result = none_out.?;
            none_out = null;
            return result;
        },
        .sum, .mean => {
            var loss: f32 = 0;
            for (row_losses) |value| loss += value;
            if (options.reduction == .mean) {
                // Deliberate divergence from PyTorch: all-ignored -> 0, not NaN.
                loss = if (valid_count == 0) 0 else loss / @as(f32, @floatFromInt(valid_count));
            }
            return ctx.scalar(.f32, loss);
        },
    }
}

/// Cross-entropy VJP. `upstream` is the scalar (`.mean`/`.sum`) or
/// per-position (`.none`) upstream gradient, as f32 values or as the
/// autograd tensor. With `options.row_stats` (saved by `crossEntropyLoss`)
/// the kernel emits final gradients in ONE pass over the logits, bitwise
/// identical to the recompute path. Ignored positions get exactly zero
/// gradient.
pub fn crossEntropyBackward(
    ctx: *ExecContext,
    comptime rank: usize,
    logits: *const Tensor,
    comptime axis: usize,
    labels: []const usize,
    options: CrossEntropyOptions,
    upstream: CrossEntropyUpstream,
) !Tensor {
    switch (upstream) {
        .scale => |scale_value| return crossEntropyBackwardScaled(ctx, rank, logits, axis, labels, options, scale_value, null),
        .rows => |rows| return crossEntropyBackwardScaled(ctx, rank, logits, axis, labels, options, rows.scale, rows.per_row),
        .tensor => |gy| {
            if (options.reduction == .none) {
                const source = try logits.rankView(rank);
                const out_rank = if (rank == 1) 1 else rank - 1;
                const expected_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);
                const gv = try gy.rankView(out_rank);
                if (!std.mem.eql(usize, gv.shape[0..], expected_shape[0..])) return tensor.TensorError.ShapeMismatch;
                var gg = try ctx.prepareContiguous(.f32, gy);
                defer gg.deinit();
                return crossEntropyBackwardScaled(ctx, rank, logits, axis, labels, options, 1, gg.tensor().dataConst());
            }
            if (!gy.isScalar()) return tensor.TensorError.ShapeMismatch;
            return crossEntropyBackwardScaled(ctx, rank, logits, axis, labels, options, gy.item(), null);
        },
    }
}

/// The f32-upstream body of `crossEntropyBackward`: `scale_value` is the
/// scalar upstream (mean/sum) or the common factor over `per_row_scale`
/// (`.none`, required there and rejected elsewhere).
fn crossEntropyBackwardScaled(
    ctx: *ExecContext,
    comptime rank: usize,
    logits: *const Tensor,
    comptime axis: usize,
    labels: []const usize,
    options: CrossEntropyOptions,
    scale_value: f32,
    per_row_scale: ?[]const f32,
) !Tensor {
    if (axis >= rank) @compileError("axis out of bounds");
    const row_stats: ?[]const f32 = options.row_stats;

    const source = try logits.rankView(rank);
    const class_count = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const position_count = outer * inner;
    const valid_count = try validateCrossEntropyLabels(labels, position_count, class_count, options);
    if (options.reduction == .none) {
        if (per_row_scale == null or per_row_scale.?.len != position_count) return tensor.TensorError.InvalidDataLength;
    } else if (per_row_scale != null) {
        return tensor.TensorError.InvalidDataLength;
    }
    if (row_stats) |stats| {
        if (stats.len != 2 * position_count) return tensor.TensorError.InvalidDataLength;
    }

    var ll = try ctx.prepareContiguous(.f32, logits);
    defer ll.deinit();
    const input = ll.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();
    const grad_common: f32 = switch (options.reduction) {
        // Deliberate divergence from PyTorch: all-ignored -> zero grads, not NaN.
        .mean => if (valid_count == 0) 0 else scale_value / @as(f32, @floatFromInt(valid_count)),
        .sum, .none => scale_value,
    };

    if (inner == 1) {
        dispatchCrossEntropyBackwardRows(ctx, .{
            .input = input,
            .labels = labels,
            .output = output,
            .per_row_scale = per_row_scale,
            .row_stats = row_stats,
            .class_count = class_count,
            .ignore_index = options.ignore_index,
            .label_smoothing = options.label_smoothing,
            .grad_common = grad_common,
            .row_start = 0,
            .row_end = outer,
        });
        return out;
    }

    const eps = options.label_smoothing;
    const eps_uniform = eps / @as(f32, @floatFromInt(class_count));
    for (0..outer) |outer_i| {
        const base = outer_i * class_count * inner;
        for (0..inner) |inner_i| {
            const row = outer_i * inner + inner_i;
            const label = labels[row];
            if (options.ignore_index) |ignore_index| {
                if (label == ignore_index) {
                    for (0..class_count) |class_i| {
                        output[base + class_i * inner + inner_i] = 0;
                    }
                    continue;
                }
            }
            const row_scale = if (per_row_scale) |values| grad_common * values[row] else grad_common;
            if (row_stats) |stats| {
                // Forward-saved stats: one pass, same values and op order as
                // the recompute arm below (see CrossEntropyBackwardRowsTask).
                const stat_max = stats[2 * row];
                const stat_sum_exp = stats[2 * row + 1];
                const stat_prob_scale = row_scale / stat_sum_exp;
                const stat_smooth_term = eps_uniform * row_scale;
                for (0..class_count) |class_i| {
                    const offset = base + class_i * inner + inner_i;
                    var grad = @exp(input[offset] - stat_max) * stat_prob_scale - stat_smooth_term;
                    if (class_i == label) grad -= (1 - eps) * row_scale;
                    output[offset] = grad;
                }
                continue;
            }
            var max_value = input[base + inner_i];
            for (1..class_count) |class_i| {
                max_value = @max(max_value, input[base + class_i * inner + inner_i]);
            }

            var sum_exp: f32 = 0;
            for (0..class_count) |class_i| {
                const offset = base + class_i * inner + inner_i;
                const value = @exp(input[offset] - max_value);
                output[offset] = value;
                sum_exp += value;
            }

            const prob_scale = row_scale / sum_exp;
            const smooth_term = eps_uniform * row_scale;
            for (0..class_count) |class_i| {
                const offset = base + class_i * inner + inner_i;
                var grad = output[offset] * prob_scale - smooth_term;
                if (class_i == label) grad -= (1 - eps) * row_scale;
                output[offset] = grad;
            }
        }
    }

    return out;
}

// --- Fused linear + cross-entropy VJP --------------------------------------

/// Gradients of `CE(xÂ·Wáµ, labels)` with respect to x and/or W. Both are
/// caller-owned when non-null.
pub const LinearCrossEntropyGrads = struct {
    dx: ?Tensor,
    dweight: ?Tensor,

    pub fn deinit(self: *LinearCrossEntropyGrads) void {
        if (self.dx) |*value| value.deinit();
        if (self.dweight) |*value| value.deinit();
        self.* = undefined;
    }
};

/// The `inner == 1` cross-entropy backward row dispatch, shared by the
/// materializing route and the fused linear+CE VJP: pool split over rows
/// above the elementwise work threshold, serial otherwise (bitwise identical
/// either way â disjoint row writes).
fn dispatchCrossEntropyBackwardRows(ctx: *ExecContext, base_task: CrossEntropyBackwardRowsTask) void {
    const outer = base_task.row_end;
    ctx.dispatchRangeOr(CrossEntropyBackwardRowsTask, "row_start", "row_end", base_task, outer, outer > 1 and outer * base_task.class_count >= parallel.row_kernel_len_threshold, crossEntropyBackwardRows);
}

/// Fused VJP of `loss = CE(xÂ·Wáµ, labels)` over the forward's saved logits
/// and per-row {max, sum_exp} statistics. DESTRUCTIVE: when `logits` owns
/// its buffer exclusively (`canTakeInPlace`), the logit gradient is written
/// IN PLACE over the logits â the stats-arm row kernel is elementwise
/// read-before-write, so aliasing input and output is exact â and the full
/// [rows, classes] gradient never costs a second buffer; a shared buffer
/// falls back to a fresh tensor with identical values. Then dx = dLÂ·W and
/// dweight = dLáµÂ·x run as the same two monolithic GEMMs the composed
/// route uses (blocked class-panel variants were measured-declined
/// 2026-07-10: 423â503 ms vs 398 ms composed at 1024Ã151936Ã1024 on M1 â
/// two big GEMMs beat 38+38 small ones). Callers must treat `logits` as
/// garbage afterwards (the autograd record enforces single use).
/// Shapes: x [rows, in], weight [classes, in], logits [rows, classes];
/// upstream gy is scalar for `.mean`/`.sum` and per-row for `.none`.
pub fn linearCrossEntropyBackwardUpstream(
    ctx: *ExecContext,
    x: *const Tensor,
    weight: *const Tensor,
    logits: *Tensor,
    labels: []const usize,
    options: CrossEntropyOptions,
    gy: *const Tensor,
    row_stats: []const f32,
    need_x: bool,
    need_weight: bool,
) !LinearCrossEntropyGrads {
    const xv = try x.rankView(2);
    const wv = try weight.rankView(2);
    const lv = try logits.rankView(2);
    const rows = xv.shape[0];
    const in_dim = xv.shape[1];
    const class_count = wv.shape[0];
    if (wv.shape[1] != in_dim or lv.shape[0] != rows or lv.shape[1] != class_count) return tensor.TensorError.ShapeMismatch;
    const valid_count = try validateCrossEntropyLabels(labels, rows, class_count, options);
    if (row_stats.len != 2 * rows) return tensor.TensorError.InvalidDataLength;
    if (!need_x and !need_weight) return .{ .dx = null, .dweight = null };

    // Upstream contract as crossEntropyBackward's `.tensor` arm.
    var scale_value: f32 = 1;
    var upstream: ?ExecContext.PreparedTensor = null;
    defer if (upstream) |*p| p.deinit();
    if (options.reduction == .none) {
        const gv = try gy.rankView(1);
        if (gv.shape[0] != rows) return tensor.TensorError.ShapeMismatch;
        upstream = try ctx.prepareContiguous(.f32, gy);
    } else {
        if (!gy.isScalar()) return tensor.TensorError.ShapeMismatch;
        scale_value = gy.item();
    }
    const per_row_scale: ?[]const f32 = if (upstream) |*p| p.tensor().dataConst() else null;
    const grad_common: f32 = switch (options.reduction) {
        // Deliberate divergence from PyTorch: all-ignored -> zero grads, not NaN.
        .mean => if (valid_count == 0) 0 else scale_value / @as(f32, @floatFromInt(valid_count)),
        .sum, .none => scale_value,
    };

    // dL destination: the logits buffer itself when exclusively owned.
    var dl_owned: ?Tensor = if (logits.canTakeInPlace()) null else try ctx.empty(.f32, .{ rows, class_count });
    defer if (dl_owned) |*value| value.deinit();
    const dl: *Tensor = if (dl_owned) |*value| value else logits;

    dispatchCrossEntropyBackwardRows(ctx, .{
        .input = logits.dataConst(),
        .labels = labels,
        .output = dl.data(),
        .per_row_scale = per_row_scale,
        .row_stats = row_stats,
        .class_count = class_count,
        .ignore_index = options.ignore_index,
        .label_smoothing = options.label_smoothing,
        .grad_common = grad_common,
        .row_start = 0,
        .row_end = rows,
    });

    var dx: ?Tensor = null;
    errdefer if (dx) |*value| value.deinit();
    if (need_x) dx = try exec_matmul.matmul(ctx, .f32, .plain, dl, weight);
    var dweight: ?Tensor = null;
    if (need_weight) dweight = try exec_matmul.matmul(ctx, .f32, .trans_a, dl, x);
    return .{ .dx = dx, .dweight = dweight };
}

// --- model-serving: fused linear + sparse-soft-target distillation loss -----
// Single consumer: qwen3 cartridge distillation (`models/qwen3/train.zig`),
// through the `Tensor.linearDistill` facade. Lives here because it shares
// the linear-loss row-task machinery with `linearCrossEntropy*` above (see
// the model-serving group in exec.zig).
//
// loss = reduce_i  probs[i] * (LSE(logits[rows[i]]) - logits[rows[i], classes[i]])
// with logits = x·Wᵀ — cross-entropy against a SPARSE soft target
// distribution (e.g. a teacher's top-k), fused with the output projection so
// the [rows, classes] logits never enter the autograd graph. Two structural
// wins over the composed route: only the UNIQUE SUPERVISED rows are
// projected (rows without entries contribute nothing — their logits are
// never computed), and like the fused CE the backward consumes the saved
// [sel_rows, classes] logits in place, so the logit gradient never costs a
// second buffer.

/// Options for `linearDistillLossStats`. `reduction` is over ENTRIES
/// (`.mean` divides by the entry count); `loss_scale` multiplies the loss
/// and therefore every gradient (the gradient-accumulation knob).
pub const LinearDistillOptions = struct {
    reduction: enum { mean, sum } = .mean,
    loss_scale: f32 = 1,
};

/// Everything the fused forward hands the autograd record. All fields are
/// caller-owned (`deinit` releases them); `logits` and `x_sel` cover ONLY
/// the unique supervised rows, in `sel_rows` order.
pub const LinearDistillForward = struct {
    /// Scalar loss.
    value: Tensor,
    /// [sel_rows.len, classes] logits of the supervised rows — saved for
    /// the destructive backward.
    logits: Tensor,
    /// [sel_rows.len, in_dim] gathered x rows — saved for dweight.
    x_sel: Tensor,
    /// Unique supervised row indices, ascending.
    sel_rows: []usize,
    /// Per-entry index into `sel_rows` (rows[i] == sel_rows[local_rows[i]]).
    local_rows: []usize,
    /// Per-selected-row {max, sum_exp} softmax statistics.
    row_stats: []f32,

    pub fn deinit(self: *LinearDistillForward, allocator: std.mem.Allocator) void {
        self.value.deinit();
        self.logits.deinit();
        self.x_sel.deinit();
        allocator.free(self.sel_rows);
        allocator.free(self.local_rows);
        allocator.free(self.row_stats);
        self.* = undefined;
    }
};

fn dispatchDistillStatsRows(ctx: *ExecContext, base_task: DistillStatsRowsTask) void {
    const outer = base_task.row_end;
    ctx.dispatchRangeOr(DistillStatsRowsTask, "row_start", "row_end", base_task, outer, outer > 1 and outer * base_task.class_count >= parallel.row_kernel_len_threshold, distillStatsRows);
}

/// Fused forward. `rows[i]` is the x row whose distribution entry `i`
/// supervises, `classes[i]` the class index, `probs[i]` the target mass
/// (any non-negative weight; a teacher's top-k probabilities in the distill
/// use). Shapes: x [row_count, in], weight [classes, in].
pub fn linearDistillLossStats(
    ctx: *ExecContext,
    x: *const Tensor,
    weight: *const Tensor,
    rows: []const usize,
    classes: []const usize,
    probs: []const f32,
    options: LinearDistillOptions,
) !LinearDistillForward {
    const xv = try x.rankView(2);
    const wv = try weight.rankView(2);
    const row_count = xv.shape[0];
    const in_dim = xv.shape[1];
    const class_count = wv.shape[0];
    if (wv.shape[1] != in_dim) return tensor.TensorError.ShapeMismatch;
    const n = rows.len;
    if (n == 0 or classes.len != n or probs.len != n) return tensor.TensorError.InvalidDataLength;
    for (rows, classes) |row, class| {
        if (row >= row_count or class >= class_count) return tensor.TensorError.IndexOutOfBounds;
    }

    // Unique supervised rows (ascending) + the per-entry local remap.
    const sel_rows = blk: {
        const sorted = try ctx.allocator().dupe(usize, rows);
        defer ctx.allocator().free(sorted);
        std.mem.sort(usize, sorted, {}, std.sort.asc(usize));
        var unique: usize = 0;
        for (sorted, 0..) |row, i| {
            if (i == 0 or row != sorted[i - 1]) {
                sorted[unique] = row;
                unique += 1;
            }
        }
        break :blk try ctx.allocator().dupe(usize, sorted[0..unique]);
    };
    errdefer ctx.allocator().free(sel_rows);
    const local_rows = try ctx.allocator().alloc(usize, n);
    errdefer ctx.allocator().free(local_rows);
    for (local_rows, rows) |*local, row| {
        local.* = std.sort.binarySearch(usize, sel_rows, row, orderUsize).?;
    }

    // Gather the supervised rows and project only them.
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const x_data = xx.tensor().dataConst();
    var x_sel = try ctx.empty(.f32, .{ sel_rows.len, in_dim });
    errdefer x_sel.deinit();
    const x_sel_data = x_sel.data();
    for (sel_rows, 0..) |row, j| {
        @memcpy(x_sel_data[j * in_dim ..][0..in_dim], x_data[row * in_dim ..][0..in_dim]);
    }
    var logits = try exec_matmul.matmul(ctx, .f32, .trans_b, &x_sel, weight);
    errdefer logits.deinit();

    const row_stats = try ctx.allocator().alloc(f32, 2 * sel_rows.len);
    errdefer ctx.allocator().free(row_stats);
    dispatchDistillStatsRows(ctx, .{
        .input = logits.dataConst(),
        .row_stats = row_stats,
        .class_count = class_count,
        .row_start = 0,
        .row_end = sel_rows.len,
    });

    // One serial sum in entry order — bitwise identical for any thread count.
    const logit_data = logits.dataConst();
    var total: f32 = 0;
    for (local_rows, classes, probs) |local, class, prob| {
        const lse = @log(row_stats[2 * local + 1]) + row_stats[2 * local];
        total += prob * (lse - logit_data[local * class_count + class]);
    }
    if (options.reduction == .mean) total /= @as(f32, @floatFromInt(n));
    total *= options.loss_scale;

    var value = try ctx.scalar(.f32, total);
    errdefer value.deinit();
    return .{
        .value = value,
        .logits = logits,
        .x_sel = x_sel,
        .sel_rows = sel_rows,
        .local_rows = local_rows,
        .row_stats = row_stats,
    };
}

fn orderUsize(context: usize, item: usize) std.math.Order {
    return std.math.order(context, item);
}

fn dispatchDistillBackwardRows(ctx: *ExecContext, base_task: DistillBackwardRowsTask) void {
    const outer = base_task.row_end;
    ctx.dispatchRangeOr(DistillBackwardRowsTask, "row_start", "row_end", base_task, outer, outer > 1 and outer * base_task.class_count >= parallel.row_kernel_len_threshold, distillBackwardRows);
}

/// Fused VJP over the forward's saved selected-row logits and statistics.
/// DESTRUCTIVE in `logits` exactly like `linearCrossEntropyBackwardUpstream`
/// (in-place when exclusively owned; the record enforces single use):
/// dlogits[r, v] = s·(mass_r · softmax_r[v]) − s·Σ_{i at (r,v)} probs[i]
/// with s = loss_scale · gy (/ n for `.mean`), then dx scatters
/// dlogits·W into the supervised rows of a zero [row_count, in] tensor and
/// dweight = dlogitsᵀ·x_sel.
pub fn linearDistillBackwardUpstream(
    ctx: *ExecContext,
    x_sel: *const Tensor,
    weight: *const Tensor,
    logits: *Tensor,
    sel_rows: []const usize,
    row_count: usize,
    local_rows: []const usize,
    classes: []const usize,
    probs: []const f32,
    options: LinearDistillOptions,
    gy: *const Tensor,
    row_stats: []const f32,
    need_x: bool,
    need_weight: bool,
) !LinearCrossEntropyGrads {
    const xv = try x_sel.rankView(2);
    const wv = try weight.rankView(2);
    const lv = try logits.rankView(2);
    const sel_count = xv.shape[0];
    const in_dim = xv.shape[1];
    const class_count = wv.shape[0];
    if (wv.shape[1] != in_dim or lv.shape[0] != sel_count or lv.shape[1] != class_count) return tensor.TensorError.ShapeMismatch;
    if (sel_rows.len != sel_count or row_stats.len != 2 * sel_count) return tensor.TensorError.InvalidDataLength;
    const n = local_rows.len;
    if (n == 0 or classes.len != n or probs.len != n) return tensor.TensorError.InvalidDataLength;
    if (!need_x and !need_weight) return .{ .dx = null, .dweight = null };
    if (!gy.isScalar()) return tensor.TensorError.ShapeMismatch;

    var grad_common: f32 = gy.item() * options.loss_scale;
    if (options.reduction == .mean) grad_common /= @as(f32, @floatFromInt(n));

    const row_mass = try ctx.allocator().alloc(f32, sel_count);
    defer ctx.allocator().free(row_mass);
    @memset(row_mass, 0);
    for (local_rows, probs) |local, prob| row_mass[local] += grad_common * prob;

    // dL destination: the logits buffer itself when exclusively owned.
    var dl_owned: ?Tensor = if (logits.canTakeInPlace()) null else try ctx.empty(.f32, .{ sel_count, class_count });
    defer if (dl_owned) |*value| value.deinit();
    const dl: *Tensor = if (dl_owned) |*value| value else logits;

    dispatchDistillBackwardRows(ctx, .{
        .input = logits.dataConst(),
        .output = dl.data(),
        .row_stats = row_stats,
        .row_mass = row_mass,
        .class_count = class_count,
        .row_start = 0,
        .row_end = sel_count,
    });

    // Sparse target subtraction: entry lists are tiny next to rows x classes.
    const dl_data = dl.data();
    for (local_rows, classes, probs) |local, class, prob| {
        dl_data[local * class_count + class] -= grad_common * prob;
    }

    var dx: ?Tensor = null;
    errdefer if (dx) |*value| value.deinit();
    if (need_x) {
        var dx_sel = try exec_matmul.matmul(ctx, .f32, .plain, dl, weight);
        defer dx_sel.deinit();
        var full = try ctx.empty(.f32, .{ row_count, in_dim });
        errdefer full.deinit();
        const full_data = full.data();
        @memset(full_data, 0);
        const dx_sel_data = dx_sel.dataConst();
        for (sel_rows, 0..) |row, j| {
            @memcpy(full_data[row * in_dim ..][0..in_dim], dx_sel_data[j * in_dim ..][0..in_dim]);
        }
        dx = full;
    }
    var dweight: ?Tensor = null;
    if (need_weight) dweight = try exec_matmul.matmul(ctx, .f32, .trans_a, dl, x_sel);
    return .{ .dx = dx, .dweight = dweight };
}

// --- Whole-tensor elementwise losses (MSE / Huber / BCE / KL) --------------
//
// One shared forward/backward driver over same-shaped input/target tensors,
// specialized by a tiny per-loss ops value (`loss(x, t)` + `grad(x, t, wrt)`
// methods). `.none` returns input-shaped per-element losses; `.mean`/`.sum`
// return a scalar via one serial sum in element order — like the
// cross-entropy reduction above, bitwise identical for any thread count.
// Cold ops: no parallel dispatch.

/// Shared elementwise-loss forward. Validates same-shaped operands, then
/// applies `ops.loss` per element and reduces per `reduction`.
fn elementwiseLossForward(ctx: *ExecContext, input: *const Tensor, target: *const Tensor, reduction: Reduction, ops: anytype) !Tensor {
    try tensor.requireSameShape(input, target);
    var xx = try ctx.prepareContiguous(.f32, input);
    defer xx.deinit();
    var tt = try ctx.prepareContiguous(.f32, target);
    defer tt.deinit();
    const x = xx.tensor().dataConst();
    const t = tt.tensor().dataConst();
    switch (reduction) {
        .none => {
            var out = try ctx.empty(.f32, xx.tensor().shape.slice());
            errdefer out.deinit();
            for (x, t, out.data()) |xv, tv, *dst| dst.* = ops.loss(xv, tv);
            return out;
        },
        .sum, .mean => {
            var loss: f32 = 0;
            for (x, t) |xv, tv| loss += ops.loss(xv, tv);
            if (reduction == .mean) loss /= @as(f32, @floatFromInt(x.len));
            return ctx.scalar(.f32, loss);
        },
    }
}

/// Shared elementwise-loss VJP taking the upstream gradient as a tensor:
/// scalar for `.mean`/`.sum`, input-shaped per-element for `.none` (the
/// cross-entropy `*UpstreamAxisRank` contract). Returns the gradient with
/// respect to the operand selected by `wrt`.
fn elementwiseLossBackwardUpstream(
    ctx: *ExecContext,
    input: *const Tensor,
    target: *const Tensor,
    reduction: Reduction,
    gy: *const Tensor,
    wrt: LossWrt,
    ops: anytype,
) !Tensor {
    try tensor.requireSameShape(input, target);
    var xx = try ctx.prepareContiguous(.f32, input);
    defer xx.deinit();
    var tt = try ctx.prepareContiguous(.f32, target);
    defer tt.deinit();
    const x = xx.tensor().dataConst();
    const t = tt.tensor().dataConst();
    var out = try ctx.empty(.f32, xx.tensor().shape.slice());
    errdefer out.deinit();
    switch (reduction) {
        .none => {
            try tensor.requireSameShape(input, gy);
            var gg = try ctx.prepareContiguous(.f32, gy);
            defer gg.deinit();
            for (x, t, gg.tensor().dataConst(), out.data()) |xv, tv, g, *dst| {
                dst.* = ops.grad(xv, tv, wrt) * g;
            }
        },
        .sum, .mean => {
            if (!gy.isScalar()) return tensor.TensorError.ShapeMismatch;
            var scale_value = gy.item();
            if (reduction == .mean) scale_value /= @as(f32, @floatFromInt(x.len));
            for (x, t, out.data()) |xv, tv, *dst| dst.* = ops.grad(xv, tv, wrt) * scale_value;
        },
    }
    return out;
}

const MseOps = struct {
    fn loss(_: @This(), x: f32, t: f32) f32 {
        const d = x - t;
        return d * d;
    }

    fn grad(_: @This(), x: f32, t: f32, wrt: LossWrt) f32 {
        const g = 2 * (x - t);
        return switch (wrt) {
            .input => g,
            .target => -g,
        };
    }
};

const HuberOps = struct {
    delta: f32,

    fn loss(self: @This(), x: f32, t: f32) f32 {
        const d = x - t;
        const ad = @abs(d);
        if (ad <= self.delta) return 0.5 * d * d;
        return self.delta * (ad - 0.5 * self.delta);
    }

    fn grad(self: @This(), x: f32, t: f32, wrt: LossWrt) f32 {
        const d = x - t;
        const g = if (@abs(d) <= self.delta) d else self.delta * std.math.sign(d);
        return switch (wrt) {
            .input => g,
            .target => -g,
        };
    }
};

const BceOps = struct {
    from_logits: bool,

    fn loss(self: @This(), x: f32, y: f32) f32 {
        if (self.from_logits) {
            // Stable BCE-with-logits: max(x,0) - x·y + log1p(exp(-|x|)).
            return @max(x, 0) - x * y + std.math.log1p(@exp(-@abs(x)));
        }
        const p = std.math.clamp(x, bce_eps, 1 - bce_eps);
        return -(y * @log(p) + (1 - y) * @log(1 - p));
    }

    fn grad(self: @This(), x: f32, y: f32, wrt: LossWrt) f32 {
        if (self.from_logits) {
            return switch (wrt) {
                .input => stableSigmoid(x) - y,
                .target => -x,
            };
        }
        const p = std.math.clamp(x, bce_eps, 1 - bce_eps);
        return switch (wrt) {
            // Exactly 0 where the forward clamp saturates (see `bce_eps`).
            .input => if (x <= bce_eps or x >= 1 - bce_eps) 0 else (p - y) / (p * (1 - p)),
            .target => @log(1 - p) - @log(p),
        };
    }
};

const KlDivOps = struct {
    log_target: bool,

    fn loss(self: @This(), x: f32, t: f32) f32 {
        if (self.log_target) return @exp(t) * (t - x);
        // xlogy(t, t) - t·x, with the torch convention 0·ln 0 = 0.
        const t_log_t: f32 = if (t > 0) t * @log(t) else 0;
        return t_log_t - t * x;
    }

    fn grad(self: @This(), x: f32, t: f32, wrt: LossWrt) f32 {
        if (self.log_target) {
            const p = @exp(t);
            return switch (wrt) {
                .input => -p,
                .target => p * (t - x + 1),
            };
        }
        return switch (wrt) {
            .input => -t,
            // d/dt [xlogy(t,t) - t·x] = ln(t) + 1 - x; defined as 0 at t == 0
            // (the mathematical limit is -inf; 0 keeps zero-mass entries
            // inert, matching their zero forward contribution).
            .target => if (t > 0) @log(t) + 1 - x else 0,
        };
    }
};

fn stableSigmoid(value: f32) f32 {
    if (value >= 0) {
        const z = @exp(-value);
        return 1 / (1 + z);
    }
    const z = @exp(value);
    return z / (1 + z);
}

fn validateHuberOptions(options: HuberOptions) !void {
    if (!(std.math.isFinite(options.delta) and options.delta > 0)) return tensor.TensorError.InvalidArgument;
}

/// Mean-squared-error forward over same-shaped input/target (torch
/// F.mse_loss): per-element (x - t)². `.mean`/`.sum` return a scalar,
/// `.none` an input-shaped tensor.
pub fn mseLoss(ctx: *ExecContext, input: *const Tensor, target: *const Tensor, options: MseOptions) !Tensor {
    return elementwiseLossForward(ctx, input, target, options.reduction, MseOps{});
}

/// MSE VJP wrt `wrt` with the upstream gradient as a tensor (scalar for
/// `.mean`/`.sum`, input-shaped for `.none`): d/dx = 2(x-t)·s, d/dt = -that.
pub fn mseBackwardUpstream(ctx: *ExecContext, input: *const Tensor, target: *const Tensor, options: MseOptions, gy: *const Tensor, wrt: LossWrt) !Tensor {
    return elementwiseLossBackwardUpstream(ctx, input, target, options.reduction, gy, wrt, MseOps{});
}

/// Huber forward over same-shaped input/target (torch F.huber_loss): see
/// `HuberOptions`. Errors with `InvalidArgument` unless `delta` is positive
/// and finite.
pub fn huberLoss(ctx: *ExecContext, input: *const Tensor, target: *const Tensor, options: HuberOptions) !Tensor {
    try validateHuberOptions(options);
    return elementwiseLossForward(ctx, input, target, options.reduction, HuberOps{ .delta = options.delta });
}

/// Huber VJP wrt `wrt` (upstream-gradient tensor contract as
/// `mseBackwardUpstream`): d/dx = d for |d| <= delta else delta·sign(d),
/// d/dt = -that.
pub fn huberBackwardUpstream(ctx: *ExecContext, input: *const Tensor, target: *const Tensor, options: HuberOptions, gy: *const Tensor, wrt: LossWrt) !Tensor {
    try validateHuberOptions(options);
    return elementwiseLossBackwardUpstream(ctx, input, target, options.reduction, gy, wrt, HuberOps{ .delta = options.delta });
}

/// Binary cross-entropy forward over same-shaped input/target: see
/// `BceOptions` (logits vs clamped-probability arm) and `bce_eps`.
pub fn bceLoss(ctx: *ExecContext, input: *const Tensor, target: *const Tensor, options: BceOptions) !Tensor {
    return elementwiseLossForward(ctx, input, target, options.reduction, BceOps{ .from_logits = options.from_logits });
}

/// BCE VJP wrt `wrt` (upstream-gradient tensor contract as
/// `mseBackwardUpstream`). Logits arm: d/dx = sigmoid(x) - y, d/dy = -x.
/// Probability arm: d/dx = (p - y)/(p(1-p)) inside the clamp interval and 0
/// outside (see `bce_eps`), d/dy = ln(1-p) - ln(p).
pub fn bceBackwardUpstream(ctx: *ExecContext, input: *const Tensor, target: *const Tensor, options: BceOptions, gy: *const Tensor, wrt: LossWrt) !Tensor {
    return elementwiseLossBackwardUpstream(ctx, input, target, options.reduction, gy, wrt, BceOps{ .from_logits = options.from_logits });
}

/// KL-divergence forward over same-shaped input/target (torch F.kl_div
/// pointwise semantics; the input is LOG-probabilities): see `KlDivOptions`.
pub fn klDivLoss(ctx: *ExecContext, input: *const Tensor, target: *const Tensor, options: KlDivOptions) !Tensor {
    return elementwiseLossForward(ctx, input, target, options.reduction, KlDivOps{ .log_target = options.log_target });
}

/// KL-divergence VJP wrt `wrt` (upstream-gradient tensor contract as
/// `mseBackwardUpstream`). Probability target: d/dx = -t,
/// d/dt = ln(t) + 1 - x (0 at t == 0). Log target: d/dx = -exp(t),
/// d/dt = exp(t)·(t - x + 1).
pub fn klDivBackwardUpstream(ctx: *ExecContext, input: *const Tensor, target: *const Tensor, options: KlDivOptions, gy: *const Tensor, wrt: LossWrt) !Tensor {
    return elementwiseLossBackwardUpstream(ctx, input, target, options.reduction, gy, wrt, KlDivOps{ .log_target = options.log_target });
}

test {
    // Elementwise-loss (MSE/Huber/BCE/KL) numeric tests live in the sibling
    // file (`exec_tests.zig`'s Group-A inline tests deliberately stay in
    // `exec.zig`-facade territory).
    _ = @import("loss_tests.zig");
}
