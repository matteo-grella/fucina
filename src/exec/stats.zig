//! Axis reductions that return statistics: argmax, extrema (max/min + index),
//! variance, standardization (fwd/bwd), top-k, and full sort/argsort.
//!
//! Domain module: every op receives an explicit `*Runtime`. Self-contained
//! (pure Zig loops over prepared-contiguous inputs; no backend/row_ops
//! kernels). Home of `TopKResult` (returned by extrema + top-k) and the
//! `Standardize*` option types (re-exported by `exec.zig`). `topKAxisRank`
//! is co-located here so the extrema/top-k family owning `TopKResult` lives
//! together (plan D3).

const std = @import("std");
const tensor = @import("../tensor.zig");

const exec_shape = @import("shape.zig");
const Runtime = @import("runtime.zig").Runtime;

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
/// so ties keep the lowest index — same tie-break as maxAxisRank).
///
/// NaN contract: comparisons drop NaN — a NaN never becomes the maximum
/// (the winner is the max over the non-NaN elements; an all-NaN row falls
/// back to index 0). Shared with maxAxisRank/minAxisRank; DIVERGES from
/// torch.argmax, which propagates NaN as the winner.
pub fn argmaxAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !tensor.TensorOf(.i64) {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);
    var out = try rt.emptyRankTyped(.i64, out_rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            // Seed with -inf and compare strictly, like extremumAxisRank
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
/// ties keep the lowest index — same tie-break as argmaxAxisRank, and
/// PyTorch's torch.max also routes the gradient to a single index). The
/// indices feed the VJP, which sends the gradient only to that first
/// occurrence. Rows stay serial like sumAxisRank/meanAxisRank (per-row
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
/// argmaxAxisRank/topK/sort): exact for any axis length.
pub fn maxAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !TopKResult {
    return extremumAxisRank(rt, rank, x, axis, .max);
}

/// Min over `axis`; see maxAxisRank (strict `<`, first occurrence wins).
pub fn minAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !TopKResult {
    return extremumAxisRank(rt, rank, x, axis, .min);
}

fn extremumAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, comptime op: ExtremumOp) !TopKResult {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);
    var values = try rt.emptyRank(out_rank, out_shape);
    errdefer values.deinit();
    var indices = try rt.emptyRankTyped(.i64, out_rank, out_shape);
    errdefer indices.deinit();
    const vd = values.data();
    const id = indices.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    if (inner == 1) {
        const Vec = @Vector(8, f32);
        const vector_width = 8;
        for (0..outer) |outer_i| {
            const row = input[outer_i * axis_dim ..][0..axis_dim];
            var axis_i: usize = 0;
            var best_vec: Vec = @splat(switch (op) {
                .max => -std.math.inf(f32),
                .min => std.math.inf(f32),
            });
            while (axis_i + vector_width <= axis_dim) : (axis_i += vector_width) {
                const chunk: Vec = row[axis_i..][0..vector_width].*;
                best_vec = switch (op) {
                    .max => @max(best_vec, chunk),
                    .min => @min(best_vec, chunk),
                };
            }
            var best_value = switch (op) {
                .max => @reduce(.Max, best_vec),
                .min => @reduce(.Min, best_vec),
            };
            while (axis_i < axis_dim) : (axis_i += 1) {
                best_value = switch (op) {
                    .max => @max(best_value, row[axis_i]),
                    .min => @min(best_value, row[axis_i]),
                };
            }
            // Second scan: the first index holding the extremum. Vector
            // @max/@min return one of their inputs exactly, so `==` is
            // safe; on an all-NaN row `==` never matches (the value is
            // the ±inf seed) and the index falls back to 0 — the same
            // fallback as the inner > 1 path below (see the NaN contract
            // on maxAxisRank).
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
            // case (see the NaN contract on maxAxisRank).
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

/// Variance over `axis` with PyTorch semantics: μ = row mean, output =
/// Σ(x−μ)²/(N−ddof). ddof 0 = biased (the LayerNorm/ggml convention),
/// ddof 1 = Bessel-corrected (the torch.var default). Statistics are
/// two-pass like layerNormAxisRank; N == ddof yields 0/0 → NaN, matching
/// torch.var on a single element. Rows stay serial like
/// sumAxisRank/meanAxisRank; inner == 1 rows take a SIMD body.
pub fn varAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, ddof: u1) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);
    var out = try rt.emptyRank(out_rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    const inv_denom = 1 / (@as(f32, @floatFromInt(axis_dim)) - @as(f32, @floatFromInt(ddof)));
    if (inner == 1) {
        const Vec = @Vector(8, f32);
        const vector_width = 8;
        for (0..outer) |outer_i| {
            const row = input[outer_i * axis_dim ..][0..axis_dim];
            var axis_i: usize = 0;
            var sum_vec: Vec = @splat(0);
            while (axis_i + vector_width <= axis_dim) : (axis_i += vector_width) {
                sum_vec += @as(Vec, row[axis_i..][0..vector_width].*);
            }
            var sum_acc = @reduce(.Add, sum_vec);
            while (axis_i < axis_dim) : (axis_i += 1) {
                sum_acc += row[axis_i];
            }
            const mean_value = sum_acc * inv_axis_dim;
            const mean_vec: Vec = @splat(mean_value);

            axis_i = 0;
            var sumsq_vec: Vec = @splat(0);
            while (axis_i + vector_width <= axis_dim) : (axis_i += vector_width) {
                const centered = @as(Vec, row[axis_i..][0..vector_width].*) - mean_vec;
                sumsq_vec += centered * centered;
            }
            var sumsq = @reduce(.Add, sumsq_vec);
            while (axis_i < axis_dim) : (axis_i += 1) {
                const centered = row[axis_i] - mean_value;
                sumsq += centered * centered;
            }
            output[outer_i] = sumsq * inv_denom;
        }
        return out;
    }

    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sum_acc: f32 = 0;
            for (0..axis_dim) |axis_i| {
                sum_acc += input[base + axis_i * inner + inner_i];
            }
            const mean_value = sum_acc * inv_axis_dim;
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const centered = input[base + axis_i * inner + inner_i] - mean_value;
                sumsq += centered * centered;
            }
            output[outer_i * inner + inner_i] = sumsq * inv_denom;
        }
    }
    return out;
}

/// Axis-wise standardization preserving the input shape:
/// `y = (x - mean(axis)) / denom`, where variance uses `options.ddof` and
/// `denom` is controlled by `options.eps_mode`. Unlike `variance`, rows with
/// `N <= ddof` use zero variance; this keeps degenerate standardization
/// finite when an epsilon is supplied.
pub fn standardizeAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    comptime axis: usize,
    options: StandardizeOptions,
) !Tensor {
    return standardizeAxisRankImpl(rt, rank, x, axis, null, options);
}

/// Standardize over the first `valid_len` elements of `axis`; positions
/// after that prefix are masked out, written as zero, and ignored by the
/// matching backward kernel.
pub fn standardizeAxisValidPrefixRank(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    comptime axis: usize,
    valid_len: usize,
    options: StandardizeOptions,
) !Tensor {
    return standardizeAxisRankImpl(rt, rank, x, axis, valid_len, options);
}

fn standardizeAxisRankImpl(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    comptime axis: usize,
    valid_len: ?usize,
    options: StandardizeOptions,
) !Tensor {
    return switch (options.accumulation) {
        .f32 => standardizeAxisRankAccum(rt, rank, f32, x, axis, valid_len, options),
        .f64 => standardizeAxisRankAccum(rt, rank, f64, x, axis, valid_len, options),
    };
}

fn standardizeAxisRankAccum(
    rt: *Runtime,
    comptime rank: usize,
    comptime Acc: type,
    x: *const Tensor,
    comptime axis: usize,
    valid_len: ?usize,
    options: StandardizeOptions,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    if (!(options.eps >= 0)) return tensor.TensorError.InvalidShape;

    const source = try x.rankView(rank);
    const axis_dim = source.shape[axis];
    const valid_count = valid_len orelse axis_dim;
    if (valid_count > axis_dim) return tensor.TensorError.InvalidShape;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const ddof_count: usize = options.ddof;
    const eps: Acc = @floatCast(options.eps);
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

pub fn standardizeBackwardAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    gy: *const Tensor,
    comptime axis: usize,
    valid_len: ?usize,
    options: StandardizeOptions,
) !Tensor {
    return switch (options.accumulation) {
        .f32 => standardizeBackwardAxisRankAccum(rt, rank, f32, x, gy, axis, valid_len, options),
        .f64 => standardizeBackwardAxisRankAccum(rt, rank, f64, x, gy, axis, valid_len, options),
    };
}

fn standardizeBackwardAxisRankAccum(
    rt: *Runtime,
    comptime rank: usize,
    comptime Acc: type,
    x: *const Tensor,
    gy: *const Tensor,
    comptime axis: usize,
    valid_len: ?usize,
    options: StandardizeOptions,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    if (!(options.eps >= 0)) return tensor.TensorError.InvalidShape;

    const source = try x.rankView(rank);
    const grad_view = try gy.rankView(rank);
    if (!std.mem.eql(usize, source.shape[0..], grad_view.shape[0..])) return tensor.TensorError.ShapeMismatch;

    const axis_dim = source.shape[axis];
    const valid_count = valid_len orelse axis_dim;
    if (valid_count > axis_dim) return tensor.TensorError.InvalidShape;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    const input = xx.tensor().dataConst();
    const upstream = gg.tensor().dataConst();

    var out = try rt.zerosRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const ddof_count: usize = options.ddof;
    const eps: Acc = @floatCast(options.eps);
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
/// admission test below) — consistent with maxAxisRank/argmaxAxisRank.
/// A row with fewer than k non-NaN elements leaves its unfilled tail
/// slots at the (-inf, index 0) seed, the same degradation as an
/// all-NaN row under maxAxisRank. This DIVERGES from torch.topk,
/// which treats NaN as greater than every number.
pub fn topKAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, k: usize) !TopKResult {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    if (k == 0) return tensor.TensorError.InvalidShape;

    const source = try x.rankView(rank);
    if (k > source.shape[axis]) return tensor.TensorError.IndexOutOfBounds;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out_shape = source.shape;
    out_shape[axis] = k;
    var values = try rt.emptyRank(rank, out_shape);
    errdefer values.deinit();
    var indices = try rt.emptyRankTyped(.i64, rank, out_shape);
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
    // NaN sorts LAST regardless of direction (see sortAxisRank doc).
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
/// above, which also refuse to let NaN win (see maxAxisRank).
///
/// Indices are i64 (the repo-wide index convention, shared with
/// argmaxAxisRank/topK): exact for any axis length.
pub fn sortAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, descending: bool) !TopKResult {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var values = try rt.emptyRank(rank, source.shape);
    errdefer values.deinit();
    var indices = try rt.emptyRankTyped(.i64, rank, source.shape);
    errdefer indices.deinit();
    const vd = values.data();
    const id = indices.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);

    const scratch = try rt.allocator.alloc(SortPair, axis_dim);
    defer rt.allocator.free(scratch);

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
