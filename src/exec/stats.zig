//! Axis reductions that return statistics: argmax, extrema (max/min + index),
//! variance, standardization (fwd/bwd), top-k, and full sort/argsort.
//!
//! Domain module: every op receives an explicit `*ExecContext`. Pure Zig
//! loops over prepared-contiguous inputs; the non-last-axis arms of
//! variance/standardize run the backend inner-lane row kernels
//! (lane ranges split across the pool). Home of `TopKResult` (returned by
//! extrema + top-k) and the
//! `Standardize*` option types (re-exported by `exec.zig`). `topK`
//! is co-located here so the extrema/top-k family owning `TopKResult` lives
//! together (plan D3).

const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");

const backend_mod = @import("../backend.zig");
const exec_row_ops = backend_mod.rows;
const exec_shape = @import("shape.zig");
const ExecContext = @import("../exec.zig").ExecContext;

const DType = tensor.DType;
const Tensor = tensor.Tensor;

const shapeWithoutAxis = exec_shape.shapeWithoutAxis;
const productAfterAxis = exec_shape.productAfterAxis;
const productBeforeAxis = exec_shape.productBeforeAxis;

pub const TopKResult = struct {
    values: Tensor,
    /// Source positions along the reduced/sorted axis. i64 (exact for any
    /// axis length; torch's index dtype), no-grad by construction.
    indices: tensor.TensorOf(.i64),

    pub fn deinit(self: *TopKResult) void {
        self.values.deinit();
        self.indices.deinit();
        self.* = undefined;
    }
};

pub const StandardizeAccumulation = enum {
    f32,
    f64,
};

pub const StandardizeEpsMode = enum {
    /// Divide by `sqrt(variance) + eps`; this is the statistical
    /// standardization convention used by Parakeet's frontend.
    outside_sqrt,
    /// Divide by `sqrt(variance + eps)`; this matches LayerNorm-style epsilon
    /// placement while keeping the configurable ddof contract.
    inside_sqrt,
};

pub const StandardizeOptions = struct {
    /// Variance correction: 0 = biased population variance, 1 = Bessel corrected.
    ddof: u1 = 0,
    eps: f32 = 0,
    eps_mode: StandardizeEpsMode = .outside_sqrt,
    accumulation: StandardizeAccumulation = .f32,
};

/// Index of the FIRST occurrence of the maximum along `axis` (strict `>`,
/// so ties keep the lowest index — same tie-break as maxAxis).
///
/// NaN contract: comparisons drop NaN — a NaN never becomes the maximum
/// (the winner is the max over the non-NaN elements; an all-NaN row falls
/// back to index 0). Shared with maxAxis/minAxis; DIVERGES from
/// torch.argmax, which propagates NaN as the winner.
/// One f32 kernel; 16-bit inputs widen (`.widened` policy). Indices are i64.
pub fn argmax(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize) !tensor.TensorOf(.i64) {
    const compute = comptime ExecContext.widenedCompute(dtype, "argmax");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    return argmaxF32(ctx, rank, xx.tensor(), axis);
}

fn argmaxF32(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize) !tensor.TensorOf(.i64) {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);
    var out = try ctx.empty(.i64, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            // Seed with -inf and compare strictly, like extremumAxis
            // below: a NaN never wins (NaN compares false), instead of the
            // old input[first] seed whose result depended on WHERE the NaN
            // sat. `best_i == axis_dim` is the "no position holds
            // best_value yet" sentinel; the else-if captures the first
            // element equal to the seed (a row whose maximum is -inf
            // itself), and the final fallback to 0 is the all-NaN case
            // (see the NaN contract above).
            var best_i: usize = axis_dim;
            var best_value = -std.math.inf(f32);
            for (0..axis_dim) |axis_i| {
                const value = input[base + axis_i * inner + inner_i];
                if (value > best_value) {
                    best_value = value;
                    best_i = axis_i;
                } else if (best_i == axis_dim and value == best_value) {
                    best_i = axis_i;
                }
            }
            if (best_i == axis_dim) best_i = 0;
            output[outer_i * inner + inner_i] = @intCast(best_i);
        }
    }
    return out;
}

const ExtremumOp = enum { max, min };

/// Max over `axis`, returning both the values and the index of the FIRST
/// occurrence of the extremum along the axis (strict `>` comparison, so
/// ties keep the lowest index — same tie-break as argmax, and
/// PyTorch's torch.max also routes the gradient to a single index). The
/// indices feed the VJP, which sends the gradient only to that first
/// occurrence. Rows stay serial like sumAxis/meanAxis (per-row
/// reductions are bandwidth-bound); inner == 1 rows take a SIMD body.
///
/// NaN contract (one contract, both layouts): comparisons drop NaN — a
/// NaN never becomes the extremum. The value is the extremum over the
/// non-NaN elements (an all-NaN row degrades to -inf for max / +inf for
/// min) and the index is the FIRST position holding that value, with
/// index 0 as the fallback when no position holds it (all-NaN row). This
/// DIVERGES from torch.max/torch.min over a dim, which propagate NaN.
///
/// Indices are i64 (the repo-wide index convention, shared with
/// argmax/topK/sort): exact for any axis length.
pub fn maxAxis(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize) !TopKResult {
    return extremumWidened(ctx, dtype, rank, x, axis, .max);
}

/// Min over `axis`; see maxAxis (strict `<`, first occurrence wins).
pub fn minAxis(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize) !TopKResult {
    return extremumWidened(ctx, dtype, rank, x, axis, .min);
}

/// One f32 kernel; 16-bit inputs widen and the values stay f32 (the
/// reduction output dtype).
fn extremumWidened(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize, comptime op: ExtremumOp) !TopKResult {
    const compute = comptime ExecContext.widenedCompute(dtype, "maxAxis/minAxis");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    return extremumAxis(ctx, rank, xx.tensor(), axis, op);
}

fn extremumAxis(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, comptime op: ExtremumOp) !TopKResult {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);
    var values = try ctx.empty(.f32, out_shape);
    errdefer values.deinit();
    var indices = try ctx.empty(.i64, out_shape);
    errdefer indices.deinit();
    const vd = values.data();
    const id = indices.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    if (inner == 1) {
        for (0..outer) |outer_i| {
            const row = input[outer_i * axis_dim ..][0..axis_dim];
            const best_value = backend_mod.kernels.extremumRowValue(op == .max, row);
            // Second scan: the first index holding the extremum. Vector
            // @max/@min return one of their inputs exactly, so `==` is
            // safe; on an all-NaN row `==` never matches (the value is
            // the ±inf seed) and the index falls back to 0 — the same
            // fallback as the inner > 1 path below (see the NaN contract
            // on maxAxis).
            var best_i: usize = 0;
            while (best_i < axis_dim and row[best_i] != best_value) best_i += 1;
            if (best_i == axis_dim) best_i = 0;
            vd[outer_i] = best_value;
            id[outer_i] = @intCast(best_i);
        }
        return .{ .values = values, .indices = indices };
    }

    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            // Seed with the ±inf identity and compare strictly, exactly
            // like the SIMD path above: a NaN never wins (NaN compares
            // false), instead of the old input[first] seed whose result
            // depended on WHERE the NaN sat. `best_i == axis_dim` is the
            // "no position holds best_value yet" sentinel; the else-if
            // captures the first element equal to the seed (a row whose
            // extremum is ±inf itself), keeping index parity with the
            // SIMD rescan, and the final fallback to 0 is the all-NaN
            // case (see the NaN contract on maxAxis).
            var best_i: usize = axis_dim;
            var best_value: f32 = switch (op) {
                .max => -std.math.inf(f32),
                .min => std.math.inf(f32),
            };
            for (0..axis_dim) |axis_i| {
                const value = input[base + axis_i * inner + inner_i];
                const better = switch (op) {
                    .max => value > best_value,
                    .min => value < best_value,
                };
                if (better) {
                    best_value = value;
                    best_i = axis_i;
                } else if (best_i == axis_dim and value == best_value) {
                    best_i = axis_i;
                }
            }
            if (best_i == axis_dim) best_i = 0;
            vd[outer_i * inner + inner_i] = best_value;
            id[outer_i * inner + inner_i] = @intCast(best_i);
        }
    }
    return .{ .values = values, .indices = indices };
}

/// Max over `axis` of the elements `mask` selects; see `extremumMasked`.
pub fn maxMasked(
    ctx: *ExecContext,
    comptime mask_dtype: DType,
    comptime rank: usize,
    x: *const Tensor,
    mask: *const tensor.TensorOf(mask_dtype),
    comptime axis: usize,
    empty_value: ?f32,
) !TopKResult {
    return extremumMasked(ctx, mask_dtype, rank, x, mask, axis, .max, empty_value);
}

/// Min over `axis` of the elements `mask` selects; see `extremumMasked`.
pub fn minMasked(
    ctx: *ExecContext,
    comptime mask_dtype: DType,
    comptime rank: usize,
    x: *const Tensor,
    mask: *const tensor.TensorOf(mask_dtype),
    comptime axis: usize,
    empty_value: ?f32,
) !TopKResult {
    return extremumMasked(ctx, mask_dtype, rank, x, mask, axis, .min, empty_value);
}

/// Extremum over `axis` restricted to the elements `mask` selects — the
/// `maxval(a, dim, mask)` / `minval(a, dim, mask)` of the Fortran intrinsic
/// set. `mask` must have `x`'s shape (broadcasting happens at the tag layer).
///
/// Semantics follow the unmasked twin exactly, with the mask applied at the
/// comparison: seed with the ±inf identity, take the FIRST strict winner, and
/// let NaN never win. A lane selecting nothing therefore lands on the identity
/// seed, which is Fortran's answer (`maxval` of an empty selection is
/// -HUGE); `empty` overrides that value when a caller wants a different
/// sentinel.
///
/// The returned index for an empty lane is **-1**, not a position: no element
/// participated, so there is nothing for a gradient to flow to.
/// `MaskedMinMaxBackward` reads that sentinel and drops the lane's gradient.
/// Every non-empty lane's index is a real, unmasked position, which is why the
/// masked VJP is otherwise the unmasked scatter.
fn extremumMasked(
    ctx: *ExecContext,
    comptime mask_dtype: DType,
    comptime rank: usize,
    x: *const Tensor,
    mask: *const tensor.TensorOf(mask_dtype),
    comptime axis: usize,
    comptime op: ExtremumOp,
    empty_value: ?f32,
) !TopKResult {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const mask_view = try mask.rankView(rank);
    for (0..rank) |dim| {
        if (source.shape[dim] != mask_view.shape[dim]) return tensor.TensorError.ShapeMismatch;
    }

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var mm = try ctx.prepareContiguous(mask_dtype, mask);
    defer mm.deinit();
    const flags = mm.tensor().dataConst();

    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);
    var values = try ctx.empty(.f32, out_shape);
    errdefer values.deinit();
    var indices = try ctx.empty(.i64, out_shape);
    errdefer indices.deinit();
    const vd = values.data();
    const id = indices.data();

    const identity: f32 = switch (op) {
        .max => -std.math.inf(f32),
        .min => std.math.inf(f32),
    };
    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);

    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var best_i: usize = axis_dim; // "no position holds best yet"
            var best_value: f32 = identity;
            var selected: usize = 0;
            for (0..axis_dim) |axis_i| {
                const at = base + axis_i * inner + inner_i;
                if (!dtype_mod.isTruthy(mask_dtype, flags[at])) continue;
                selected += 1;
                const value = input[at];
                const better = switch (op) {
                    .max => value > best_value,
                    .min => value < best_value,
                };
                if (better) {
                    best_value = value;
                    best_i = axis_i;
                } else if (best_i == axis_dim and value == best_value) {
                    best_i = axis_i;
                }
            }
            const flat = outer_i * inner + inner_i;
            if (selected == 0) {
                vd[flat] = empty_value orelse identity;
                id[flat] = -1;
                continue;
            }
            // An all-NaN selection leaves best_i unset; fall back to the first
            // SELECTED position, the masked analogue of the unmasked fallback
            // to 0 (see the NaN contract on maxAxis).
            if (best_i == axis_dim) {
                var first: usize = 0;
                while (first < axis_dim and !dtype_mod.isTruthy(mask_dtype, flags[base + first * inner + inner_i])) first += 1;
                best_i = first;
            }
            vd[flat] = best_value;
            id[flat] = @intCast(best_i);
        }
    }

    return .{ .values = values, .indices = indices };
}

/// Variance over `axis` with PyTorch semantics: μ = row mean, output =
/// Σ(x−μ)²/(N−ddof). ddof 0 = biased (the LayerNorm/ggml convention),
/// ddof 1 = Bessel-corrected (the torch.var default). Statistics are
/// two-pass like layerNorm; N == ddof yields 0/0 → NaN, matching
/// torch.var on a single element. inner == 1 rows stay serial like
/// sumAxis/meanAxis and take a SIMD body; inner > 1 (a non-last axis)
/// runs the inner-lane kernel with lane ranges split across the pool.
/// Variance over `axis`. One f32 kernel; 16-bit inputs widen and return
/// f32 (the reduction output dtype).
pub fn varAxis(ctx: *ExecContext, comptime dtype: DType, comptime rank: usize, x: *const tensor.TensorOf(dtype), comptime axis: usize, ddof: u1) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
    const compute = comptime ExecContext.widenedCompute(dtype, "varAxis");
    var xx = try ctx.prepareAs(dtype, compute, x);
    defer xx.deinit();
    var out = try varAxisF32(ctx, rank, xx.tensor(), axis, ddof);
    errdefer out.deinit();
    return ctx.storeAs(compute, comptime dtype_mod.outputDType(.reduction, dtype), out);
}

fn varAxisF32(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, ddof: u1) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);
    var out = try ctx.empty(.f32, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    const inv_denom = 1 / (@as(f32, @floatFromInt(axis_dim)) - @as(f32, @floatFromInt(ddof)));
    if (inner == 1) {
        backend_mod.kernels.varianceRowsInto(output, input, outer, axis_dim, inv_axis_dim, inv_denom);
        return out;
    }

    // inner > 1: vector lanes across `inner`, every pass streamed row-major;
    // per-lane accumulation order along the axis is the strided scalar
    // loop's, so the lane split is bitwise-neutral.
    const scratch = try ctx.allocator.alloc(f32, 2 * inner);
    defer ctx.allocator.free(scratch);
    ctx.dispatchInnerLanes(exec_row_ops.VarianceInnerTask, .{
        .input = input,
        .output = output,
        .axis_dim = axis_dim,
        .inner = inner,
        .scratch = scratch,
        .outer = outer,
        .inv_axis_dim = inv_axis_dim,
        .inv_denom = inv_denom,
        .inner_start = 0,
        .inner_end = inner,
    }, source.len(), inner, exec_row_ops.runVarianceInnerTask);
    return out;
}

/// Axis-wise standardization preserving the input shape:
/// `y = (x - mean(axis)) / denom`, where variance uses `options.ddof` and
/// `denom` is controlled by `options.eps_mode`. Unlike `variance`, rows with
/// `N <= ddof` use zero variance; this keeps degenerate standardization
/// finite when an epsilon is supplied.
pub fn standardize(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    comptime axis: usize,
    options: StandardizeOptions,
) !Tensor {
    return standardizeImpl(ctx, rank, x, axis, null, options);
}

// --- model-serving: standardizeValidPrefix (parakeet frontend) --------------
// Single consumer: the parakeet NeMo frontend, through `standardizeAxis`'s
// `.valid_len` option. A thin arm of `standardizeImpl` shared with the plain
// `standardize` (see the model-serving group in exec.zig).

/// Standardize over the first `valid_len` elements of `axis`; positions
/// after that prefix are masked out, written as zero, and ignored by the
/// matching backward kernel.
pub fn standardizeValidPrefix(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    comptime axis: usize,
    valid_len: usize,
    options: StandardizeOptions,
) !Tensor {
    return standardizeImpl(ctx, rank, x, axis, valid_len, options);
}

fn standardizeImpl(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    comptime axis: usize,
    valid_len: ?usize,
    options: StandardizeOptions,
) !Tensor {
    return switch (options.accumulation) {
        .f32 => standardizeAccum(ctx, rank, f32, x, axis, valid_len, options),
        .f64 => standardizeAccum(ctx, rank, f64, x, axis, valid_len, options),
    };
}

fn standardizeAccum(
    ctx: *ExecContext,
    comptime rank: usize,
    comptime Acc: type,
    x: *const Tensor,
    comptime axis: usize,
    valid_len: ?usize,
    options: StandardizeOptions,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
    if (axis >= rank) @compileError("axis out of bounds");
    if (!(options.eps >= 0)) return tensor.TensorError.InvalidArgument;

    const source = try x.rankView(rank);
    const axis_dim = source.shape[axis];
    const valid_count = valid_len orelse axis_dim;
    if (valid_count > axis_dim) return tensor.TensorError.InvalidShape;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try ctx.empty(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const ddof_count: usize = options.ddof;
    const eps: Acc = @floatCast(options.eps);
    if (inner > 1) {
        // Non-last axis: the inner-lane kernel (lane ranges split across
        // the pool; per-lane order equals the scalar loop below).
        const scratch = try ctx.allocator.alloc(Acc, 2 * inner);
        defer ctx.allocator.free(scratch);
        ctx.dispatchInnerLanes(exec_row_ops.StandardizeInnerTask(Acc), .{
            .input = input,
            .output = output,
            .axis_dim = axis_dim,
            .inner = inner,
            .valid_count = valid_count,
            .ddof_count = ddof_count,
            .eps = eps,
            .eps_inside_sqrt = options.eps_mode == .inside_sqrt,
            .scratch = scratch,
            .outer = outer,
            .inner_start = 0,
            .inner_end = inner,
        }, source.len(), inner, exec_row_ops.runStandardizeInnerTask(Acc));
        return out;
    }
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            if (valid_count == 0) {
                for (0..axis_dim) |axis_i| output[base + axis_i * inner + inner_i] = 0;
                continue;
            }

            var sum_acc: Acc = 0;
            for (0..valid_count) |axis_i| {
                sum_acc += @floatCast(input[base + axis_i * inner + inner_i]);
            }
            const count: Acc = @floatFromInt(valid_count);
            const mean_value = sum_acc / count;

            var variance: Acc = 0;
            if (valid_count > ddof_count) {
                var sumsq: Acc = 0;
                for (0..valid_count) |axis_i| {
                    const centered = @as(Acc, @floatCast(input[base + axis_i * inner + inner_i])) - mean_value;
                    sumsq += centered * centered;
                }
                variance = sumsq / @as(Acc, @floatFromInt(valid_count - ddof_count));
            }

            const std_value = @sqrt(variance);
            const denom = switch (options.eps_mode) {
                .outside_sqrt => std_value + eps,
                .inside_sqrt => @sqrt(variance + eps),
            };
            for (0..valid_count) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const centered = @as(Acc, @floatCast(input[offset])) - mean_value;
                output[offset] = @floatCast(centered / denom);
            }
            for (valid_count..axis_dim) |axis_i| {
                output[base + axis_i * inner + inner_i] = 0;
            }
        }
    }
    return out;
}

pub fn standardizeBackward(
    ctx: *ExecContext,
    comptime rank: usize,
    x: *const Tensor,
    gy: *const Tensor,
    comptime axis: usize,
    valid_len: ?usize,
    options: StandardizeOptions,
) !Tensor {
    return switch (options.accumulation) {
        .f32 => standardizeBackwardAccum(ctx, rank, f32, x, gy, axis, valid_len, options),
        .f64 => standardizeBackwardAccum(ctx, rank, f64, x, gy, axis, valid_len, options),
    };
}

fn standardizeBackwardAccum(
    ctx: *ExecContext,
    comptime rank: usize,
    comptime Acc: type,
    x: *const Tensor,
    gy: *const Tensor,
    comptime axis: usize,
    valid_len: ?usize,
    options: StandardizeOptions,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
    if (axis >= rank) @compileError("axis out of bounds");
    if (!(options.eps >= 0)) return tensor.TensorError.InvalidArgument;

    const source = try x.rankView(rank);
    const grad_view = try gy.rankView(rank);
    if (!std.mem.eql(usize, source.shape[0..], grad_view.shape[0..])) return tensor.TensorError.ShapeMismatch;

    const axis_dim = source.shape[axis];
    const valid_count = valid_len orelse axis_dim;
    if (valid_count > axis_dim) return tensor.TensorError.InvalidShape;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    var gg = try ctx.prepareContiguous(.f32, gy);
    defer gg.deinit();
    const input = xx.tensor().dataConst();
    const upstream = gg.tensor().dataConst();

    var out = try ctx.zeros(.f32, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const ddof_count: usize = options.ddof;
    const eps: Acc = @floatCast(options.eps);
    if (inner > 1) {
        const scratch = try ctx.allocator.alloc(Acc, 5 * inner);
        defer ctx.allocator.free(scratch);
        ctx.dispatchInnerLanes(exec_row_ops.StandardizeBackwardInnerTask(Acc), .{
            .input = input,
            .grad = upstream,
            .output = output,
            .axis_dim = axis_dim,
            .inner = inner,
            .valid_count = valid_count,
            .ddof_count = ddof_count,
            .eps = eps,
            .eps_inside_sqrt = options.eps_mode == .inside_sqrt,
            .scratch = scratch,
            .outer = outer,
            .inner_start = 0,
            .inner_end = inner,
        }, source.len(), inner, exec_row_ops.runStandardizeBackwardInnerTask(Acc));
        return out;
    }
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            if (valid_count == 0) continue;

            var sum_acc: Acc = 0;
            for (0..valid_count) |axis_i| {
                sum_acc += @floatCast(input[base + axis_i * inner + inner_i]);
            }
            const count: Acc = @floatFromInt(valid_count);
            const mean_value = sum_acc / count;

            var variance: Acc = 0;
            if (valid_count > ddof_count) {
                var sumsq: Acc = 0;
                for (0..valid_count) |axis_i| {
                    const centered = @as(Acc, @floatCast(input[base + axis_i * inner + inner_i])) - mean_value;
                    sumsq += centered * centered;
                }
                variance = sumsq / @as(Acc, @floatFromInt(valid_count - ddof_count));
            }

            const std_value = @sqrt(variance);
            const denom = switch (options.eps_mode) {
                .outside_sqrt => std_value + eps,
                .inside_sqrt => @sqrt(variance + eps),
            };
            const inv_denom = 1 / denom;

            var grad_sum: Acc = 0;
            var centered_grad_dot: Acc = 0;
            for (0..valid_count) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const g = @as(Acc, @floatCast(upstream[offset]));
                const centered = @as(Acc, @floatCast(input[offset])) - mean_value;
                grad_sum += g;
                centered_grad_dot += g * centered;
            }
            const mean_grad = grad_sum / count;

            var second_scale: Acc = 0;
            if (valid_count > ddof_count and variance > 0) {
                const denom_count: Acc = @floatFromInt(valid_count - ddof_count);
                second_scale = switch (options.eps_mode) {
                    .outside_sqrt => centered_grad_dot / (denom_count * std_value * denom * denom),
                    .inside_sqrt => centered_grad_dot / (denom_count * denom * denom * denom),
                };
            }

            for (0..valid_count) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const centered = @as(Acc, @floatCast(input[offset])) - mean_value;
                output[offset] = @floatCast((@as(Acc, @floatCast(upstream[offset])) - mean_grad) * inv_denom - centered * second_scale);
            }
        }
    }
    return out;
}

/// Top-k values along `axis`, descending, with their source indices.
///
/// NaN contract: a NaN never places (it fails the `value > slot-min`
/// admission test below) — consistent with maxAxis/argmax.
/// A row with fewer than k non-NaN elements leaves its unfilled tail
/// slots at the (-inf, index 0) seed, the same degradation as an
/// all-NaN row under maxAxis. This DIVERGES from torch.topk,
/// which treats NaN as greater than every number.
pub fn topK(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, k: usize) !TopKResult {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
    if (axis >= rank) @compileError("axis out of bounds");
    if (k == 0) return tensor.TensorError.InvalidArgument;

    const source = try x.rankView(rank);
    if (k > source.shape[axis]) return tensor.TensorError.IndexOutOfBounds;

    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out_shape = source.shape;
    out_shape[axis] = k;
    var values = try ctx.empty(.f32, out_shape);
    errdefer values.deinit();
    var indices = try ctx.empty(.i64, out_shape);
    errdefer indices.deinit();
    const vd = values.data();
    const id = indices.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    for (0..outer) |outer_i| {
        const input_base = outer_i * axis_dim * inner;
        const output_base = outer_i * k * inner;
        for (0..inner) |inner_i| {
            for (0..k) |slot| {
                vd[output_base + slot * inner + inner_i] = -std.math.inf(f32);
                id[output_base + slot * inner + inner_i] = 0;
            }
            for (0..axis_dim) |axis_i| {
                const value = input[input_base + axis_i * inner + inner_i];
                // Slots are descending-sorted: anything <= the current min
                // (equal included) cannot place, so one compare rejects it.
                // Negated form on purpose: NaN also fails `value >`, so a
                // NaN is rejected here instead of falling through the `<=`
                // scans below and landing in slot 0 (see the NaN contract
                // above). Slot values are never NaN (seeded -inf, and only
                // values admitted here are ever written).
                if (!(value > vd[output_base + (k - 1) * inner + inner_i])) continue;
                var slot: usize = 0;
                while (slot < k and value <= vd[output_base + slot * inner + inner_i]) : (slot += 1) {}
                if (slot == k) continue;
                var move = k - 1;
                while (move > slot) : (move -= 1) {
                    vd[output_base + move * inner + inner_i] = vd[output_base + (move - 1) * inner + inner_i];
                    id[output_base + move * inner + inner_i] = id[output_base + (move - 1) * inner + inner_i];
                }
                vd[output_base + slot * inner + inner_i] = value;
                id[output_base + slot * inner + inner_i] = @intCast(axis_i);
            }
        }
    }

    return .{ .values = values, .indices = indices };
}

const SortPair = struct {
    value: f32,
    index: usize,
};

fn sortPairBefore(descending: bool, a: SortPair, b: SortPair) bool {
    // NaN sorts LAST regardless of direction (see sort doc).
    if (std.math.isNan(a.value)) return false;
    if (std.math.isNan(b.value)) return true;
    return if (descending) a.value > b.value else a.value < b.value;
}

/// Full sort along `axis`, returning both the sorted values and the source
/// index of each output position (torch.sort values/indices; input shape
/// preserved). `descending = false` sorts ascending. UNSTABLE sort
/// (std.sort.pdq over per-row (value, index) pairs): equal values keep no
/// particular relative order — torch.sort is also unstable by default, but
/// the tie ORDER may differ between the two.
///
/// NaN contract: NaN sorts LAST regardless of direction. This DIVERGES from
/// torch.sort, which treats NaN as greater than every number (last only when
/// ascending, FIRST when descending) — consistent with the extrema kernels
/// above, which also refuse to let NaN win (see maxAxis).
///
/// Indices are i64 (the repo-wide index convention, shared with
/// argmax/topK): exact for any axis length.
pub fn sort(ctx: *ExecContext, comptime rank: usize, x: *const Tensor, comptime axis: usize, descending: bool) !TopKResult {
    if (rank == 0 or rank > tensor.max_rank) @compileError(tensor.invalid_rank_msg);
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try ctx.prepareContiguous(.f32, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var values = try ctx.empty(.f32, source.shape);
    errdefer values.deinit();
    var indices = try ctx.empty(.i64, source.shape);
    errdefer indices.deinit();
    const vd = values.data();
    const id = indices.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);

    const scratch = try ctx.allocator.alloc(SortPair, axis_dim);
    defer ctx.allocator.free(scratch);

    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            for (scratch, 0..) |*pair, axis_i| {
                pair.* = .{ .value = input[base + axis_i * inner + inner_i], .index = axis_i };
            }
            std.sort.pdq(SortPair, scratch, descending, sortPairBefore);
            for (scratch, 0..) |pair, axis_i| {
                const offset = base + axis_i * inner + inner_i;
                vd[offset] = pair.value;
                id[offset] = @intCast(pair.index);
            }
        }
    }
    return .{ .values = values, .indices = indices };
}
