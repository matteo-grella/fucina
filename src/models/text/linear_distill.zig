//! Fused linear + sparse-soft-target distillation loss, as a custom VJP
//! beside its consumer (the qwen3 cartridge trainer):
//!
//!   loss = reduce_i probs[i] * (LSE(logits[rows[i]]) - logits[rows[i], classes[i]])
//!
//! with `logits = x·Wᵀ` — cross-entropy against a SPARSE soft target
//! distribution (a teacher's top-k), fused with the output projection so
//! the `[rows, classes]` logits never enter the autograd graph. Two
//! structural wins over the composed route (`cartridge.distillLoss`): only
//! the UNIQUE supervised rows are projected (rows without entries
//! contribute nothing, their logits are never computed), and the backward
//! consumes the saved `[sel_rows, classes]` logits in place, so the logit
//! gradient never costs a second buffer. The two routes agree to f32
//! roundoff (pinned by the trainer test and by `linear_distill_tests.zig`).

const std = @import("std");
const fucina = @import("fucina");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const RawTensor = fucina.internal.tensor_mod.Tensor;
const TensorError = fucina.internal.tensor_mod.TensorError;
const vexpf = fucina.simd.vexpf;
const row_kernel_len_threshold = fucina.parallel.row_kernel_len_threshold;

/// `reduction` is over ENTRIES (`.mean` divides by the entry count);
/// `loss_scale` multiplies the loss and therefore every gradient (the
/// gradient-accumulation knob).
pub const Options = struct {
    reduction: enum { mean, sum } = .mean,
    loss_scale: f32 = 1,
};

pub const Error = error{
    /// The record is single-use: its backward consumes the saved logits.
    LinearDistillBackwardConsumed,
};

/// Fused forward + record. `x` is a rank-2 `[row, shared]` f32 facade
/// tensor, `weight` rank-2 `[class, shared]` (shared tag last on both, the
/// class tag absent from `x`; all comptime-checked). `rows[i]` is the x row
/// whose distribution entry `i` supervises, `classes[i]` the class index,
/// `probs[i]` the target mass (any non-negative weight; a truncated teacher
/// tail is used as given, NOT renormalized). Differentiable in BOTH
/// operands; the record is single-use.
pub fn linearDistill(
    ctx: *ExecContext,
    x: anytype,
    weight: anytype,
    rows: []const usize,
    classes: []const usize,
    probs: []const f32,
    options: Options,
) !fucina.Tensor(.{}) {
    const X = @TypeOf(x.*);
    const W = @TypeOf(weight.*);
    comptime {
        if (X.axis_tags.len != 2 or W.axis_tags.len != 2) @compileError("linearDistill requires rank-2 [row, shared] x and [class, shared] weight");
        if (X.dtype != .f32 or W.dtype != .f32) @compileError("linearDistill requires f32 operands");
        if (X.axis_tags[1] != W.axis_tags[1]) @compileError("linearDistill requires the shared tag LAST on both operands");
        if (W.axis_tags[0] == X.axis_tags[0] or W.axis_tags[0] == X.axis_tags[1]) @compileError("linearDistill weight class tag must not appear on x");
    }
    const n = rows.len;
    if (n == 0 or classes.len != n or probs.len != n) return TensorError.InvalidDataLength;

    const allocator = ctx.allocator();
    const saved = try allocator.create(Saved);
    errdefer allocator.destroy(saved);
    saved.* = .{ .allocator = allocator };
    errdefer saved.deinit();
    saved.rows = try allocator.dupe(usize, rows);
    saved.classes = try allocator.dupe(usize, classes);
    saved.probs = try allocator.dupe(f32, probs);
    return fucina.customVjp(ctx, Spec, Extra{ .options = options, .saved = saved }, .{ x, weight });
}

/// Everything the forward hands the backward, owned by the record through
/// `Extra.deinit`. `logits` and `x_sel` cover ONLY the unique supervised
/// rows, in `sel_rows` order.
const Saved = struct {
    allocator: Allocator,
    rows: []usize = &.{},
    classes: []usize = &.{},
    probs: []f32 = &.{},
    /// Unique supervised row indices, ascending.
    sel_rows: []usize = &.{},
    /// Per-entry index into `sel_rows` (rows[i] == sel_rows[local_rows[i]]).
    local_rows: []usize = &.{},
    /// Per-selected-row {max, sum_exp} softmax statistics.
    row_stats: []f32 = &.{},
    /// [sel_rows.len, in_dim] gathered x rows, for dweight.
    x_sel: ?RawTensor = null,
    /// [sel_rows.len, classes] logits of the supervised rows, consumed in
    /// place by the backward.
    logits: ?RawTensor = null,
    row_count: usize = 0,
    consumed: bool = false,

    fn deinit(self: *Saved) void {
        const a = self.allocator;
        inline for (.{ "rows", "classes", "probs", "sel_rows", "local_rows", "row_stats" }) |name| {
            const slice = @field(self, name);
            if (slice.len > 0) a.free(slice);
        }
        if (self.x_sel) |*t| t.deinit();
        if (self.logits) |*t| t.deinit();
        self.* = undefined;
    }
};

const Extra = struct {
    options: Options,
    saved: *Saved,

    /// Called by `customVjp` when the record is released (or right after
    /// the forward when nothing records).
    pub fn deinit(self: *Extra, allocator: Allocator) void {
        self.saved.deinit();
        allocator.destroy(self.saved);
    }
};

const Spec = struct {
    pub const Output = fucina.Tensor(.{});

    pub fn forward(ctx: *ExecContext, extra: Extra, inputs: []const *const RawTensor) !RawTensor {
        return forwardImpl(ctx, extra.saved, extra.options, inputs[0], inputs[1]);
    }

    pub fn backward(
        ctx: *ExecContext,
        extra: Extra,
        inputs: []const *const RawTensor,
        output: *const RawTensor,
        gy: *const RawTensor,
        needs_grad: []const bool,
        out: []?RawTensor,
    ) !void {
        _ = output;
        return backwardImpl(ctx, extra.saved, extra.options, inputs[1], gy, needs_grad, out);
    }
};

fn orderUsize(context: usize, item: usize) std.math.Order {
    return std.math.order(context, item);
}

/// Forward: unique supervised rows, their projection, per-row softmax
/// statistics, and one serial entry-order sum (bitwise identical for any
/// thread count). Shapes: x [row_count, in], weight [classes, in].
fn forwardImpl(ctx: *ExecContext, saved: *Saved, options: Options, x: *const RawTensor, weight: *const RawTensor) !RawTensor {
    const a = saved.allocator;
    const xv = try x.rankView(2);
    const wv = try weight.rankView(2);
    const row_count = xv.shape[0];
    const in_dim = xv.shape[1];
    const class_count = wv.shape[0];
    if (wv.shape[1] != in_dim) return TensorError.ShapeMismatch;
    const rows = saved.rows;
    const n = rows.len;
    for (rows, saved.classes) |row, class| {
        if (row >= row_count or class >= class_count) return TensorError.IndexOutOfBounds;
    }
    saved.row_count = row_count;

    // Unique supervised rows (ascending) + the per-entry local remap.
    saved.sel_rows = blk: {
        const sorted = try a.dupe(usize, rows);
        defer a.free(sorted);
        std.mem.sort(usize, sorted, {}, std.sort.asc(usize));
        var unique: usize = 0;
        for (sorted, 0..) |row, i| {
            if (i == 0 or row != sorted[i - 1]) {
                sorted[unique] = row;
                unique += 1;
            }
        }
        break :blk try a.dupe(usize, sorted[0..unique]);
    };
    const sel_rows = saved.sel_rows;
    saved.local_rows = try a.alloc(usize, n);
    for (saved.local_rows, rows) |*local, row| {
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
    var logits = try ctx.matmul(.f32, .trans_b, &x_sel, weight);
    errdefer logits.deinit();

    saved.row_stats = try a.alloc(f32, 2 * sel_rows.len);
    dispatchRows(ctx, StatsTask, .{
        .input = logits.dataConst(),
        .row_stats = saved.row_stats,
        .class_count = class_count,
    }, sel_rows.len, statsRows);

    const logit_data = logits.dataConst();
    var total: f32 = 0;
    for (saved.local_rows, saved.classes, saved.probs) |local, class, prob| {
        const lse = @log(saved.row_stats[2 * local + 1]) + saved.row_stats[2 * local];
        total += prob * (lse - logit_data[local * class_count + class]);
    }
    if (options.reduction == .mean) total /= @as(f32, @floatFromInt(n));
    total *= options.loss_scale;

    saved.x_sel = x_sel;
    saved.logits = logits;
    return ctx.scalar(.f32, total);
}

/// Backward over the saved selected-row logits and statistics, DESTRUCTIVE
/// in `logits` (in place when exclusively owned; single use):
/// dlogits[r, v] = s·(mass_r · softmax_r[v]) − s·Σ_{i at (r,v)} probs[i]
/// with s = loss_scale · gy (/ n for `.mean`); dx scatters dlogits·W into
/// the supervised rows of a zero [row_count, in] tensor and
/// dweight = dlogitsᵀ·x_sel.
fn backwardImpl(
    ctx: *ExecContext,
    saved: *Saved,
    options: Options,
    weight: *const RawTensor,
    gy: *const RawTensor,
    needs_grad: []const bool,
    out: []?RawTensor,
) !void {
    if (saved.consumed) return Error.LinearDistillBackwardConsumed;
    saved.consumed = true;
    const need_x = needs_grad[0];
    const need_weight = needs_grad[1];
    if (!need_x and !need_weight) return;
    if (!gy.isScalar()) return TensorError.ShapeMismatch;

    const a = saved.allocator;
    const x_sel = &saved.x_sel.?;
    const logits = &saved.logits.?;
    const xv = try x_sel.rankView(2);
    const wv = try weight.rankView(2);
    const sel_count = xv.shape[0];
    const in_dim = xv.shape[1];
    const class_count = wv.shape[0];
    const n = saved.local_rows.len;

    var grad_common: f32 = gy.item() * options.loss_scale;
    if (options.reduction == .mean) grad_common /= @as(f32, @floatFromInt(n));

    const row_mass = try a.alloc(f32, sel_count);
    defer a.free(row_mass);
    @memset(row_mass, 0);
    for (saved.local_rows, saved.probs) |local, prob| row_mass[local] += grad_common * prob;

    // dL destination: the logits buffer itself when exclusively owned.
    var dl_owned: ?RawTensor = if (logits.canTakeInPlace()) null else try ctx.empty(.f32, .{ sel_count, class_count });
    defer if (dl_owned) |*value| value.deinit();
    const dl: *RawTensor = if (dl_owned) |*value| value else logits;

    dispatchRows(ctx, BackwardTask, .{
        .input = logits.dataConst(),
        .output = dl.data(),
        .row_stats = saved.row_stats,
        .row_mass = row_mass,
        .class_count = class_count,
    }, sel_count, backwardRows);

    // Sparse target subtraction: entry lists are tiny next to rows x classes.
    const dl_data = dl.data();
    for (saved.local_rows, saved.classes, saved.probs) |local, class, prob| {
        dl_data[local * class_count + class] -= grad_common * prob;
    }

    if (need_x) {
        var dx_sel = try ctx.matmul(.f32, .plain, dl, weight);
        defer dx_sel.deinit();
        var full = try ctx.empty(.f32, .{ saved.row_count, in_dim });
        errdefer full.deinit();
        const full_data = full.data();
        @memset(full_data, 0);
        const dx_sel_data = dx_sel.dataConst();
        for (saved.sel_rows, 0..) |row, j| {
            @memcpy(full_data[row * in_dim ..][0..in_dim], dx_sel_data[j * in_dim ..][0..in_dim]);
        }
        out[0] = full;
    }
    if (need_weight) out[1] = try ctx.matmul(.f32, .trans_a, dl, x_sel);
}

// --- row kernels ------------------------------------------------------------
// Per-row {max, sum_exp} and the softmax-scaled gradient rows, over the
// same vector exp as the core's cross-entropy rows. Disjoint row ranges
// across tasks, so the result is bitwise identical for any thread count.

const StatsTask = struct {
    input: []const f32,
    /// Per-row {max, sum_exp} interleaved.
    row_stats: []f32,
    class_count: usize,
};

const BackwardTask = struct {
    /// May alias `output` (the destructive in-place arm over saved logits).
    input: []const f32,
    output: []f32,
    row_stats: []const f32,
    /// Per-row total teacher mass, pre-multiplied by the common gradient
    /// scale: output[r, v] = row_mass[r] * softmax(input[r])[v].
    row_mass: []const f32,
    class_count: usize,
};

const Vec = @Vector(8, f32);
const vector_width = 8;

fn dispatchRows(ctx: *ExecContext, comptime Task: type, task: Task, rows: usize, comptime kernel: fn (Task, usize, usize) void) void {
    ctx.forRange(rows, if (rows > 1 and rows * task.class_count >= row_kernel_len_threshold) rows else 1, task, kernel);
}

fn statsRows(task: StatsTask, row_start: usize, row_end: usize) void {
    for (row_start..row_end) |row_i| {
        const row_in = task.input[row_i * task.class_count ..][0..task.class_count];

        var class_i: usize = 0;
        var max_vec: Vec = @splat(-std.math.inf(f32));
        while (class_i + vector_width <= task.class_count) : (class_i += vector_width) {
            max_vec = @max(max_vec, @as(Vec, row_in[class_i..][0..vector_width].*));
        }
        var max_value = @reduce(.Max, max_vec);
        while (class_i < task.class_count) : (class_i += 1) {
            max_value = @max(max_value, row_in[class_i]);
        }

        const max_splat: Vec = @splat(max_value);
        var sum_vec: Vec = @splat(0);
        class_i = 0;
        while (class_i + vector_width <= task.class_count) : (class_i += vector_width) {
            sum_vec += vexpf(vector_width, @as(Vec, row_in[class_i..][0..vector_width].*) - max_splat);
        }
        var sum_exp = @reduce(.Add, sum_vec);
        while (class_i < task.class_count) : (class_i += 1) {
            sum_exp += vexpf(1, @splat(row_in[class_i] - max_value))[0];
        }
        task.row_stats[2 * row_i] = max_value;
        task.row_stats[2 * row_i + 1] = sum_exp;
    }
}

fn backwardRows(task: BackwardTask, row_start: usize, row_end: usize) void {
    for (row_start..row_end) |row_i| {
        const row_in = task.input[row_i * task.class_count ..][0..task.class_count];
        const row_out = task.output[row_i * task.class_count ..][0..task.class_count];
        const max_value = task.row_stats[2 * row_i];
        const inv_sum = 1 / task.row_stats[2 * row_i + 1];
        const scale = task.row_mass[row_i];
        const max_splat: Vec = @splat(max_value);
        const factor: Vec = @splat(scale * inv_sum);
        var class_i: usize = 0;
        while (class_i + vector_width <= task.class_count) : (class_i += vector_width) {
            const p = vexpf(vector_width, @as(Vec, row_in[class_i..][0..vector_width].*) - max_splat);
            row_out[class_i..][0..vector_width].* = p * factor;
        }
        while (class_i < task.class_count) : (class_i += 1) {
            row_out[class_i] = vexpf(1, @splat(row_in[class_i] - max_value))[0] * scale * inv_sum;
        }
    }
}

test {
    _ = @import("linear_distill_tests.zig");
}
