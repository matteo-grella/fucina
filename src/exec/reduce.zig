//! Sum / mean reductions (whole-tensor + along-axis, typed) + the cumulative
//! (prefix/suffix) sums.
//!
//! Domain module: every op receives an explicit `*Runtime`. Reduction math
//! dispatches to the backend's `sumInto`/`sumSlice(Typed)` kernels. Carries a
//! local copy of `ensureForwardFloatMath` (precedent: matmul.zig) to stay
//! leaf-ward of `exec.zig`.

const std = @import("std");
const build_options = @import("build_options");
const parallel = @import("../parallel.zig");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");

const exec_shape = @import("shape.zig");
const Runtime = @import("runtime.zig").Runtime;

const DType = tensor.DType;
const Tensor = tensor.Tensor;

const shapeWithoutAxis = exec_shape.shapeWithoutAxis;
const contiguousStridesArray = exec_shape.contiguousStridesArray;
const productAfterAxis = exec_shape.productAfterAxis;
const productBeforeAxis = exec_shape.productBeforeAxis;

fn ensureForwardFloatMath(comptime dtype: DType) void {
    if (!dtype_mod.supportsForwardFloatMath(dtype)) {
        @compileError("forward math is currently supported only for floating dtypes");
    }
}

fn isIntSum(comptime dtype: DType) bool {
    return dtype == .bool or dtype_mod.supportsIntMath(dtype);
}

/// Integer/bool sum: i64 accumulation (wrapping, the +% contract), bool
/// counts `true`s — torch's integer-sum semantics. Plain exec loop.
fn intSumSlice(comptime dtype: DType, values: []const dtype_mod.Scalar(dtype)) i64 {
    var acc: i64 = 0;
    if (comptime dtype == .bool) {
        for (values) |v| acc +%= @intFromBool(v);
    } else {
        for (values) |v| acc +%= v;
    }
    return acc;
}

fn intSumContribution(comptime dtype: DType, value: dtype_mod.Scalar(dtype)) i64 {
    return if (comptime dtype == .bool) @intFromBool(value) else @as(i64, value);
}

pub fn sum(rt: *Runtime, x: *const Tensor) !Tensor {
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();

    var out = try rt.scalar(0);
    errdefer out.deinit();
    rt.enableNativeVectorPoolForWork(xx.tensor().len(), parallel.vector_elementwise_len_threshold);
    try rt.backend.sumInto(&out, xx.tensor());
    return out;
}

pub fn sumTyped(rt: *Runtime, comptime dtype: DType, x: *const tensor.TensorOf(dtype)) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
    if (comptime dtype == .f32) return sum(rt, x);
    if (comptime isIntSum(dtype)) {
        var xx = try rt.prepareContiguousTyped(dtype, x);
        defer xx.deinit();
        return rt.scalarTyped(.i64, intSumSlice(dtype, xx.tensor().dataConst()));
    }
    comptime ensureForwardFloatMath(dtype);
    const compute_dtype = comptime dtype_mod.computeDType(.reduction, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.reduction, dtype);

    var xx = try rt.prepareContiguousTyped(dtype, x);
    defer xx.deinit();

    _ = compute_dtype;
    rt.enableNativeVectorPoolForWork(xx.tensor().len(), parallel.vector_elementwise_len_threshold);
    var out = try rt.scalarTyped(output_dtype, rt.backend.sumSliceTyped(dtype, xx.tensor().dataConst()));
    errdefer out.deinit();
    return out;
}

pub fn sumAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    return sumAxisRankTyped(rt, .f32, rank, x, axis);
}

pub fn sumAxisRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    x: *const tensor.TensorOf(dtype),
    comptime axis: usize,
) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
    if (comptime dtype == .f32) return sumAxisRankF32(rt, rank, x, axis);
    if (comptime isIntSum(dtype)) return intSumAxisRank(rt, dtype, rank, x, axis);
    comptime ensureForwardFloatMath(dtype);
    const compute_dtype = comptime dtype_mod.computeDType(.reduction, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.reduction, dtype);

    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);

    var xx = try rt.prepareContiguousTyped(dtype, x);
    defer xx.deinit();
    const xp = xx.tensor();
    const input = xp.dataConst();

    var out = try rt.zerosRankTyped(output_dtype, out_rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    if (rank == 1) {
        rt.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
        output[0] = rt.backend.sumSliceTyped(dtype, input);
        return out;
    }

    if (comptime axis == rank - 1) {
        const axis_dim = source.shape[axis];
        for (0..output.len) |row| {
            const base = row * axis_dim;
            output[row] = rt.backend.sumSliceTyped(dtype, input[base..][0..axis_dim]);
        }
        return out;
    }

    const out_strides = contiguousStridesArray(out_rank, out_shape);
    for (input, 0..) |value, linear| {
        var remainder = linear;
        var out_linear: usize = 0;
        comptime var dim = rank;
        inline while (dim > 0) {
            dim -= 1;
            const coord = remainder % source.shape[dim];
            remainder /= source.shape[dim];
            if (dim != axis) {
                const out_dim = if (dim < axis) dim else dim - 1;
                out_linear += coord * out_strides[out_dim];
            }
        }
        const next = dtype_mod.castFloat(output_dtype, compute_dtype, output[out_linear]) + dtype_mod.castFloat(dtype, compute_dtype, value);
        output[out_linear] = dtype_mod.castFloat(compute_dtype, output_dtype, next);
    }

    return out;
}

fn intSumAxisRank(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    x: *const tensor.TensorOf(dtype),
    comptime axis: usize,
) !tensor.TensorOf(.i64) {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);

    var xx = try rt.prepareContiguousTyped(dtype, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.zerosRankTyped(.i64, out_rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    if (rank == 1) {
        output[0] = intSumSlice(dtype, input);
        return out;
    }

    if (comptime axis == rank - 1) {
        const axis_dim = source.shape[axis];
        for (0..output.len) |row| {
            output[row] = intSumSlice(dtype, input[row * axis_dim ..][0..axis_dim]);
        }
        return out;
    }

    const out_strides = contiguousStridesArray(out_rank, out_shape);
    for (input, 0..) |value, linear| {
        var remainder = linear;
        var out_linear: usize = 0;
        comptime var dim = rank;
        inline while (dim > 0) {
            dim -= 1;
            const coord = remainder % source.shape[dim];
            remainder /= source.shape[dim];
            if (dim != axis) {
                const out_dim = if (dim < axis) dim else dim - 1;
                out_linear += coord * out_strides[out_dim];
            }
        }
        output[out_linear] +%= intSumContribution(dtype, value);
    }

    return out;
}

fn sumAxisRankF32(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const xp = xx.tensor();
    const input = xp.dataConst();

    var out = try rt.zerosRank(out_rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    if (rank == 1) {
        rt.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
        try rt.backend.sumInto(&out, xp);
        return out;
    }

    if (comptime axis == rank - 1) {
        const axis_dim = source.shape[axis];
        for (0..output.len) |row| {
            const base = row * axis_dim;
            output[row] = rt.backend.sumSlice(input[base..][0..axis_dim]);
        }
        return out;
    }

    const out_strides = contiguousStridesArray(out_rank, out_shape);
    for (input, 0..) |value, linear| {
        var remainder = linear;
        var out_linear: usize = 0;
        comptime var dim = rank;
        inline while (dim > 0) {
            dim -= 1;
            const coord = remainder % source.shape[dim];
            remainder /= source.shape[dim];
            if (dim != axis) {
                const out_dim = if (dim < axis) dim else dim - 1;
                out_linear += coord * out_strides[out_dim];
            }
        }
        output[out_linear] += value;
    }

    return out;
}

/// Cumulative sum along `axis` (torch.cumsum), preserving the input shape:
/// `out[..., i, ...] = Σ_{j <= i} x[..., j, ...]`. Default build: each row
/// is one serial prefix sum in axis order — bitwise deterministic for any
/// thread count (cold op; no parallel dispatch). With `-Dvector-scan` the
/// scan kernels vectorize (see `scanAxisRankDirected`).
pub fn cumsumAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    return scanAxisRankDirected(rt, rank, x, axis, .sum, false);
}

/// Reversed cumulative (suffix) sum along `axis`:
/// `out[..., i, ...] = Σ_{j >= i} x[..., j, ...]` — the `cumsumAxisRank` VJP
/// (a dedicated reverse pass, same determinism contract and the same
/// `-Dvector-scan` gating).
pub fn cumsumReverseAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    return scanAxisRankDirected(rt, rank, x, axis, .sum, true);
}

/// Cumulative product along `axis` (torch.cumprod), preserving the input
/// shape: `out[..., i, ...] = Π_{j <= i} x[..., j, ...]`. Same contract and
/// `-Dvector-scan` gating as `cumsumAxisRank`.
pub fn cumprodAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    return scanAxisRankDirected(rt, rank, x, axis, .prod, false);
}

const ScanOp = enum { sum, prod };

inline fn scanIdentity(comptime op: ScanOp) f32 {
    return if (op == .sum) 0 else 1;
}

inline fn scanCombine(comptime op: ScanOp, a: f32, b: f32) f32 {
    return if (comptime op == .sum) a + b else a * b;
}

const scan_vector_width = 8;
const ScanVec = @Vector(scan_vector_width, f32);

inline fn scanCombineVec(comptime op: ScanOp, a: ScanVec, b: ScanVec) ScanVec {
    return if (comptime op == .sum) a + b else a * b;
}

/// Shift lanes toward the high end by `k`, filling vacated low lanes with
/// the scan identity (forward Hillis–Steele step).
inline fn scanShiftUp(comptime op: ScanOp, comptime k: usize, v: ScanVec) ScanVec {
    const identity: ScanVec = @splat(scanIdentity(op));
    comptime var mask: [scan_vector_width]i32 = undefined;
    comptime for (0..scan_vector_width) |j| {
        mask[j] = if (j >= k) @intCast(j - k) else ~@as(i32, 0);
    };
    return @shuffle(f32, v, identity, @as(@Vector(scan_vector_width, i32), mask));
}

/// Shift lanes toward the low end by `k`, filling vacated high lanes with
/// the scan identity (reverse/suffix Hillis–Steele step).
inline fn scanShiftDown(comptime op: ScanOp, comptime k: usize, v: ScanVec) ScanVec {
    const identity: ScanVec = @splat(scanIdentity(op));
    comptime var mask: [scan_vector_width]i32 = undefined;
    comptime for (0..scan_vector_width) |j| {
        mask[j] = if (j + k < scan_vector_width) @intCast(j + k) else ~@as(i32, 0);
    };
    return @shuffle(f32, v, identity, @as(@Vector(scan_vector_width, i32), mask));
}

/// Directed inclusive scan (sum or prod, forward or reverse) along `axis`.
///
/// Default build (`-Dvector-scan=false`): the documented serial-per-row
/// scan, bitwise deterministic and sequence-exact.
///
/// `-Dvector-scan=true`:
///   - non-last axes vectorize across `scan_vector_width` independent
///     columns per strip — each lane is one column's serial scan, so the
///     result is BITWISE IDENTICAL to the serial default;
///   - the last axis runs an in-register Hillis–Steele prefix scan per
///     row (log2(W) shifted combines + a running carry) — still bitwise
///     deterministic for any thread count, but the accumulation order
///     differs from the serial default (the sum-SIMD-lanes rounding
///     class; exact for integer-valued data).
fn scanAxisRankDirected(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize, comptime op: ScanOp, comptime reverse: bool) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);

    if (comptime build_options.vector_scan) {
        if (inner == 1) {
            for (0..outer) |row_i| {
                const base = row_i * axis_dim;
                scanRowVec(op, reverse, input[base..][0..axis_dim], output[base..][0..axis_dim]);
            }
        } else {
            for (0..outer) |outer_i| {
                const base = outer_i * axis_dim * inner;
                var j: usize = 0;
                while (j + scan_vector_width <= inner) : (j += scan_vector_width) {
                    var acc: ScanVec = @splat(scanIdentity(op));
                    for (0..axis_dim) |step| {
                        const axis_i = if (comptime reverse) axis_dim - 1 - step else step;
                        const offset = base + axis_i * inner + j;
                        acc = scanCombineVec(op, acc, input[offset..][0..scan_vector_width].*);
                        output[offset..][0..scan_vector_width].* = acc;
                    }
                }
                while (j < inner) : (j += 1) {
                    var acc: f32 = scanIdentity(op);
                    for (0..axis_dim) |step| {
                        const axis_i = if (comptime reverse) axis_dim - 1 - step else step;
                        const offset = base + axis_i * inner + j;
                        acc = scanCombine(op, acc, input[offset]);
                        output[offset] = acc;
                    }
                }
            }
        }
        return out;
    }

    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var acc: f32 = scanIdentity(op);
            for (0..axis_dim) |step| {
                const axis_i = if (comptime reverse) axis_dim - 1 - step else step;
                const offset = base + axis_i * inner + inner_i;
                acc = scanCombine(op, acc, input[offset]);
                output[offset] = acc;
            }
        }
    }
    return out;
}

/// In-register Hillis–Steele inclusive scan of one contiguous row.
fn scanRowVec(comptime op: ScanOp, comptime reverse: bool, row_in: []const f32, row_out: []f32) void {
    const n = row_in.len;
    var carry: f32 = scanIdentity(op);

    if (comptime !reverse) {
        var i: usize = 0;
        while (i + scan_vector_width <= n) : (i += scan_vector_width) {
            var v: ScanVec = row_in[i..][0..scan_vector_width].*;
            v = scanCombineVec(op, v, scanShiftUp(op, 1, v));
            v = scanCombineVec(op, v, scanShiftUp(op, 2, v));
            v = scanCombineVec(op, v, scanShiftUp(op, 4, v));
            v = scanCombineVec(op, v, @splat(carry));
            row_out[i..][0..scan_vector_width].* = v;
            carry = v[scan_vector_width - 1];
        }
        while (i < n) : (i += 1) {
            carry = scanCombine(op, carry, row_in[i]);
            row_out[i] = carry;
        }
    } else {
        var i: usize = n;
        while (i >= scan_vector_width) {
            i -= scan_vector_width;
            var v: ScanVec = row_in[i..][0..scan_vector_width].*;
            v = scanCombineVec(op, v, scanShiftDown(op, 1, v));
            v = scanCombineVec(op, v, scanShiftDown(op, 2, v));
            v = scanCombineVec(op, v, scanShiftDown(op, 4, v));
            v = scanCombineVec(op, v, @splat(carry));
            row_out[i..][0..scan_vector_width].* = v;
            carry = v[0];
        }
        while (i > 0) {
            i -= 1;
            carry = scanCombine(op, carry, row_in[i]);
            row_out[i] = carry;
        }
    }
}

/// First-order linear recurrence along `axis` (the associative-scan
/// primitive of the SSM / linear-attention family):
/// `h_t = a_t · h_{t-1} + b_t` per independent lane (each fixed choice of
/// the non-axis indices), with `h_{-1}` read from `initial` (or 0 when
/// absent). `b` supplies the output shape; `a` is a same-logical-shape
/// tensor — typically the facade's zero-stride broadcast alignment of a
/// lower-rank decay — read through its strides, never materialized.
/// `initial` holds one element per lane, row-major with the axis removed
/// (the shape-minus-axis layout).
///
/// Determinism: one serial pass per lane in axis order, each step evaluated
/// as multiply-then-add — bitwise deterministic for any thread count (cold
/// op; no parallel dispatch, the `cumsum` contract). With `-Dvector-scan`,
/// non-last axes vectorize across `scan_vector_width` independent lanes
/// when the decay's flattened lane strides are contiguous or fully
/// broadcast — every lane runs the identical elementwise op sequence, so
/// the result stays BITWISE IDENTICAL to the serial default. The last axis
/// stays serial per lane even under `-Dvector-scan`: an in-register form
/// would reassociate `a·h + b` and change rounding, so no gated variant
/// exists (unlike cumsum's last-axis prefix scan).
pub fn linearRecurrenceAxisRank(rt: *Runtime, comptime rank: usize, b: *const Tensor, a: *const Tensor, comptime axis: usize, initial: ?*const Tensor) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try b.rankView(rank);
    const a_view = try a.rankView(rank);
    inline for (0..rank) |d| {
        if (a_view.shape[d] != source.shape[d]) return tensor.TensorError.ShapeMismatch;
    }

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);

    var bb = try rt.prepareContiguous(b);
    defer bb.deinit();
    const b_data = bb.tensor().dataConst();

    var init_prep: ?Runtime.PreparedTensor = null;
    defer if (init_prep) |*p| p.deinit();
    var init_data: ?[]const f32 = null;
    if (initial) |ini| {
        if (ini.len() != outer * inner) return tensor.TensorError.ShapeMismatch;
        init_prep = try rt.prepareContiguous(ini);
        init_data = init_prep.?.tensor().dataConst();
    }

    var out = try rt.emptyRank(rank, source.shape);
    errdefer out.deinit();
    const output = out.data();

    const a_data = a.buffer.data;
    const a_axis_stride = a_view.strides[axis];
    const lane_kind = laneStrideKind(rank, axis, source.shape, a_view.strides);

    for (0..outer) |outer_i| {
        const a_outer_off = a.offset + strideOffset(rank, 0, axis, source.shape, a_view.strides, outer_i);
        var inner_i: usize = 0;

        if (comptime build_options.vector_scan) {
            if (comptime axis != rank - 1) {
                if (lane_kind != .general) {
                    while (inner_i + scan_vector_width <= inner) : (inner_i += scan_vector_width) {
                        var hv: ScanVec = if (init_data) |ini| ini[outer_i * inner + inner_i ..][0..scan_vector_width].* else @splat(0);
                        for (0..axis_dim) |t| {
                            const off = (outer_i * axis_dim + t) * inner + inner_i;
                            const av: ScanVec = switch (lane_kind) {
                                .broadcast => @splat(a_data[a_outer_off + t * a_axis_stride]),
                                .contiguous => a_data[a_outer_off + t * a_axis_stride + inner_i ..][0..scan_vector_width].*,
                                .general => unreachable,
                            };
                            hv = av * hv + @as(ScanVec, b_data[off..][0..scan_vector_width].*);
                            output[off..][0..scan_vector_width].* = hv;
                        }
                    }
                }
            }
        }

        while (inner_i < inner) : (inner_i += 1) {
            const a_off = a_outer_off + strideOffset(rank, axis + 1, rank, source.shape, a_view.strides, inner_i);
            var h: f32 = if (init_data) |ini| ini[outer_i * inner + inner_i] else 0;
            for (0..axis_dim) |t| {
                const off = (outer_i * axis_dim + t) * inner + inner_i;
                h = a_data[a_off + t * a_axis_stride] * h + b_data[off];
                output[off] = h;
            }
        }
    }
    return out;
}

pub const LinearRecurrenceGrads = struct {
    gb: Tensor,
    da: ?Tensor,
    dinitial: ?Tensor,
};

/// VJP of `linearRecurrenceAxisRank`. One reverse serial pass per lane:
/// `gh_t = a_{t+1} · gh_{t+1} + gy_t` (with `gh_T = 0`) — `gb` IS `gh`;
/// `da_t = gh_t · h_{t-1}` with `h_{-1}` = the lane's initial element (or
/// 0); `dinitial = a_0 · gh_0`. `a` is the forward's aligned decay view
/// (read through its strides), `h` the forward OUTPUT. `da` comes back at
/// the full `b` shape — the caller reduces it onto the decay's own shape
/// (the pointwise broadcast-backward rule). Same determinism contract and
/// `-Dvector-scan` lane vectorization as the forward (bitwise identical to
/// the serial pass either way).
pub fn linearRecurrenceBackwardAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    gy: *const Tensor,
    a: *const Tensor,
    h: *const Tensor,
    initial: ?*const Tensor,
    comptime axis: usize,
    want_da: bool,
    want_dinitial: bool,
) !LinearRecurrenceGrads {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try gy.rankView(rank);
    const a_view = try a.rankView(rank);
    inline for (0..rank) |d| {
        if (a_view.shape[d] != source.shape[d]) return tensor.TensorError.ShapeMismatch;
    }

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);

    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    const gy_data = gg.tensor().dataConst();
    var hh = try rt.prepareContiguous(h);
    defer hh.deinit();
    const h_data = hh.tensor().dataConst();

    var init_prep: ?Runtime.PreparedTensor = null;
    defer if (init_prep) |*p| p.deinit();
    var init_data: ?[]const f32 = null;
    if (initial) |ini| {
        if (ini.len() != outer * inner) return tensor.TensorError.ShapeMismatch;
        init_prep = try rt.prepareContiguous(ini);
        init_data = init_prep.?.tensor().dataConst();
    }

    var gb = try rt.emptyRank(rank, source.shape);
    errdefer gb.deinit();
    const gb_data = gb.data();

    var da: ?Tensor = if (want_da) try rt.emptyRank(rank, source.shape) else null;
    errdefer if (da) |*t| t.deinit();
    const da_data: ?[]f32 = if (da) |*t| t.data() else null;

    const out_rank = if (rank == 1) 1 else rank - 1;
    var dinitial: ?Tensor = if (want_dinitial) try rt.emptyRank(out_rank, shapeWithoutAxis(rank, out_rank, source.shape, axis)) else null;
    errdefer if (dinitial) |*t| t.deinit();
    const dinit_data: ?[]f32 = if (dinitial) |*t| t.data() else null;

    const a_data = a.buffer.data;
    const a_axis_stride = a_view.strides[axis];
    const lane_kind = laneStrideKind(rank, axis, source.shape, a_view.strides);

    for (0..outer) |outer_i| {
        const a_outer_off = a.offset + strideOffset(rank, 0, axis, source.shape, a_view.strides, outer_i);
        var inner_i: usize = 0;

        if (comptime build_options.vector_scan) {
            if (comptime axis != rank - 1) {
                if (lane_kind != .general) {
                    while (inner_i + scan_vector_width <= inner) : (inner_i += scan_vector_width) {
                        const loadA = struct {
                            inline fn at(kind: LaneStrideKind, data: []const f32, base: usize, stride: usize, t: usize, lane: usize) ScanVec {
                                return switch (kind) {
                                    .broadcast => @splat(data[base + t * stride]),
                                    .contiguous => data[base + t * stride + lane ..][0..scan_vector_width].*,
                                    .general => unreachable,
                                };
                            }
                        }.at;
                        var ghv: ScanVec = @splat(0);
                        var t = axis_dim;
                        while (t > 0) {
                            t -= 1;
                            const off = (outer_i * axis_dim + t) * inner + inner_i;
                            const gyv: ScanVec = gy_data[off..][0..scan_vector_width].*;
                            ghv = if (t + 1 == axis_dim) gyv else loadA(lane_kind, a_data, a_outer_off, a_axis_stride, t + 1, inner_i) * ghv + gyv;
                            gb_data[off..][0..scan_vector_width].* = ghv;
                            if (da_data) |dd| {
                                const hp: ScanVec = if (t > 0)
                                    h_data[off - inner ..][0..scan_vector_width].*
                                else if (init_data) |ini|
                                    ini[outer_i * inner + inner_i ..][0..scan_vector_width].*
                                else
                                    @splat(0);
                                dd[off..][0..scan_vector_width].* = ghv * hp;
                            }
                        }
                        if (dinit_data) |di| {
                            const dv = loadA(lane_kind, a_data, a_outer_off, a_axis_stride, 0, inner_i) * ghv;
                            di[outer_i * inner + inner_i ..][0..scan_vector_width].* = dv;
                        }
                    }
                }
            }
        }

        while (inner_i < inner) : (inner_i += 1) {
            const a_off = a_outer_off + strideOffset(rank, axis + 1, rank, source.shape, a_view.strides, inner_i);
            var gh: f32 = 0;
            var t = axis_dim;
            while (t > 0) {
                t -= 1;
                const off = (outer_i * axis_dim + t) * inner + inner_i;
                gh = if (t + 1 == axis_dim) gy_data[off] else a_data[a_off + (t + 1) * a_axis_stride] * gh + gy_data[off];
                gb_data[off] = gh;
                if (da_data) |dd| {
                    const h_prev: f32 = if (t > 0)
                        h_data[off - inner]
                    else if (init_data) |ini|
                        ini[outer_i * inner + inner_i]
                    else
                        0;
                    dd[off] = gh * h_prev;
                }
            }
            if (dinit_data) |di| {
                di[outer_i * inner + inner_i] = a_data[a_off] * gh;
            }
        }
    }
    return .{ .gb = gb, .da = da, .dinitial = dinitial };
}

const LaneStrideKind = enum { broadcast, contiguous, general };

/// Classify the decay view's flattened lane (inner-axes) access for the
/// `-Dvector-scan` arm: all-zero strides (one decay value per (outer, t)),
/// contiguous strides matching the row-major inner layout (lane offset ==
/// lane index), or anything else (scalar fallback).
fn laneStrideKind(comptime rank: usize, comptime axis: usize, shape: [rank]usize, strides: [rank]usize) LaneStrideKind {
    var expected: usize = 1;
    var contiguous = true;
    var broadcast = true;
    comptime var di: usize = 0;
    inline while (di < rank - axis - 1) : (di += 1) {
        const d = rank - 1 - di;
        if (strides[d] != 0) broadcast = false;
        if (strides[d] != expected) contiguous = false;
        expected *= shape[d];
    }
    if (broadcast) return .broadcast;
    if (contiguous) return .contiguous;
    return .general;
}

/// Strided offset of the `index`-th row-major position over axes
/// `[from, to)`: decomposes `index` (last axis fastest) and dots the
/// per-axis indices with `strides`.
fn strideOffset(comptime rank: usize, comptime from: usize, comptime to: usize, shape: [rank]usize, strides: [rank]usize, index: usize) usize {
    var offset: usize = 0;
    var rem = index;
    comptime var di: usize = 0;
    inline while (di < to - from) : (di += 1) {
        const d = to - 1 - di;
        offset += (rem % shape[d]) * strides[d];
        rem /= shape[d];
    }
    return offset;
}

/// Product along `axis` (torch.prod over a dim), the axis removed —
/// `sumAxisRank`'s structure at full parity: rank-1 reduces through the
/// pooled SIMD `prodInto`, a last-axis reduction runs one vectorized
/// `prodSlice` per row, and the general axis falls back to the same
/// delinearized scalar accumulation `sum` uses. Like `sum`, the SIMD
/// lane order fixes the float multiplication order per backend.
pub fn prodAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const xp = xx.tensor();
    const input = xp.dataConst();

    var out = try rt.emptyRank(out_rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    if (rank == 1) {
        rt.enableNativeVectorPoolForWork(xp.len(), parallel.vector_elementwise_len_threshold);
        try rt.backend.prodInto(&out, xp);
        return out;
    }

    if (comptime axis == rank - 1) {
        const axis_dim = source.shape[axis];
        for (0..output.len) |row| {
            const base = row * axis_dim;
            output[row] = rt.backend.prodSlice(input[base..][0..axis_dim]);
        }
        return out;
    }

    for (output) |*value| value.* = 1;
    const out_strides = contiguousStridesArray(out_rank, out_shape);
    for (input, 0..) |value, linear| {
        var remainder = linear;
        var out_linear: usize = 0;
        comptime var dim = rank;
        inline while (dim > 0) {
            dim -= 1;
            const coord = remainder % source.shape[dim];
            remainder /= source.shape[dim];
            if (dim != axis) {
                const out_dim = if (dim < axis) dim else dim - 1;
                out_linear += coord * out_strides[out_dim];
            }
        }
        output[out_linear] *= value;
    }

    return out;
}

pub fn meanAxisRank(rt: *Runtime, comptime rank: usize, x: *const Tensor, comptime axis: usize) !Tensor {
    var out = try sumAxisRank(rt, rank, x, axis);
    errdefer out.deinit();
    out.scaleInPlace(1 / @as(f32, @floatFromInt(x.shape.at(axis))));
    return out;
}

// ---------------------------------------------------------------------------
// Masked reductions
//
// Fortran's array intrinsics all carry an optional `mask=`; these are its
// sum/mean twins. The point is fusion: the composed spelling
// (`where(mask, x, 0)` then `sum`) materializes a whole f32 copy of the input
// and walks it a second time, while these accumulate straight out of the
// source.
//
// Accumulation order is the unmasked path's, exactly: a last-axis reduction
// gathers the selected elements of one row into pooled row scratch (a single
// cache-resident row, not a copy of the tensor) and hands it to the same
// `sumSlice`/`sumInto` SIMD kernel, and the non-last-axis arm accumulates in
// linear order like its unmasked twin. An all-true mask therefore reproduces
// the unmasked reduction BITWISE, which is what the tests pin.
//
// Empty lanes are the other half of the Fortran contract. `sum(a, mask=…)`
// over a lane that selects nothing is the operation's identity (0), not an
// error and not a NaN — so a masked reduction has a defined answer where
// `maskedSelect` has to raise `EmptySelection`. `empty` overrides the identity
// where a caller wants a different sentinel; `mean` has no identity, so its
// default empty lane is NaN (0/0) unless one is supplied.
// ---------------------------------------------------------------------------

/// Per-lane mean plus the per-lane count of selected elements. The count is
/// the mean's divisor, and the VJP needs it, so the forward hands it back
/// rather than making the backward recount. The caller owns both.
pub const MaskedMeanResult = struct {
    values: Tensor,
    counts: Tensor,

    pub fn deinit(self: *MaskedMeanResult) void {
        self.values.deinit();
        self.counts.deinit();
        self.* = undefined;
    }
};

/// Sum over `axis` of the elements `mask` selects. `mask` must have `x`'s
/// shape (broadcasting happens above, at the tag layer). Lanes selecting
/// nothing yield `empty orelse 0`.
pub fn sumMaskedAxisRank(
    rt: *Runtime,
    comptime mask_dtype: DType,
    comptime rank: usize,
    x: *const Tensor,
    mask: *const tensor.TensorOf(mask_dtype),
    comptime axis: usize,
    empty_value: ?f32,
) !Tensor {
    var counts: ?Tensor = null;
    defer if (counts) |*c| c.deinit();
    return maskedReduceAxisRank(rt, mask_dtype, rank, x, mask, axis, empty_value, &counts, false);
}

/// Mean over `axis` of the elements `mask` selects, plus the per-lane
/// selected counts. Lanes selecting nothing yield `empty orelse NaN` (0/0 has
/// no identity to fall back on) and a count of zero.
pub fn meanMaskedAxisRank(
    rt: *Runtime,
    comptime mask_dtype: DType,
    comptime rank: usize,
    x: *const Tensor,
    mask: *const tensor.TensorOf(mask_dtype),
    comptime axis: usize,
    empty_value: ?f32,
) !MaskedMeanResult {
    var counts: ?Tensor = null;
    errdefer if (counts) |*c| c.deinit();
    var values = try maskedReduceAxisRank(rt, mask_dtype, rank, x, mask, axis, empty_value, &counts, true);
    errdefer values.deinit();
    const result: MaskedMeanResult = .{ .values = values, .counts = counts.? };
    counts = null;
    return result;
}

/// Vector width for the masked select. Matches the f32 lane count the rest of
/// the native kernels use.
const masked_vec_len: usize = @import("std").simd.suggestVectorLength(f32) orelse 4;

/// `dst[i] = mask[i] ? src[i] : 0`, explicitly vectorized.
///
/// LLVM does not reliably auto-vectorize this shape: the flags are a
/// byte-per-element array while the data is f32, so the select needs a
/// widening compare the scalar loop does not spell out. Writing the compare
/// and `@select` by hand is what turns the last-axis masked reduction from
/// compute-bound back into bandwidth-bound.
fn selectRow(
    comptime mask_dtype: DType,
    src: []const f32,
    flags: []const dtype_mod.Scalar(mask_dtype),
    dst: []f32,
) void {
    const Lanes = @Vector(masked_vec_len, f32);
    const zero: Lanes = @splat(0);
    var i: usize = 0;

    if (comptime mask_dtype == .bool) {
        // `bool` storage is one byte per element; compare the bytes so the
        // predicate arrives as a lane mask instead of a per-element branch.
        const bytes: []const u8 = @ptrCast(flags);
        const Bytes = @Vector(masked_vec_len, u8);
        const byte_zero: Bytes = @splat(0);
        while (i + masked_vec_len <= src.len) : (i += masked_vec_len) {
            const values: Lanes = src[i..][0..masked_vec_len].*;
            const keep: Bytes = bytes[i..][0..masked_vec_len].*;
            dst[i..][0..masked_vec_len].* = @select(f32, keep != byte_zero, values, zero);
        }
    } else if (comptime mask_dtype == .f32) {
        const gate_zero: Lanes = @splat(0);
        while (i + masked_vec_len <= src.len) : (i += masked_vec_len) {
            const values: Lanes = src[i..][0..masked_vec_len].*;
            const gate: Lanes = flags[i..][0..masked_vec_len].*;
            dst[i..][0..masked_vec_len].* = @select(f32, gate != gate_zero, values, zero);
        }
    }

    // Tail, and the whole row for mask dtypes without a vector arm (f16/bf16/
    // f64 masks are legal but rare; correctness first).
    while (i < src.len) : (i += 1) {
        dst[i] = if (dtype_mod.isTruthy(mask_dtype, flags[i])) src[i] else 0;
    }
}

/// Number of selected elements in a flag row.
fn countRow(comptime mask_dtype: DType, flags: []const dtype_mod.Scalar(mask_dtype)) usize {
    var selected: usize = 0;
    for (flags) |keep| selected += @intFromBool(dtype_mod.isTruthy(mask_dtype, keep));
    return selected;
}

/// Shared masked sum/mean core. `counts_out` receives the per-lane selected
/// counts when `want_mean` is set (the mean divides by them) or when the
/// caller passed a non-null slot; the sum arm only needs them to detect empty
/// lanes, and drops them.
fn maskedReduceAxisRank(
    rt: *Runtime,
    comptime mask_dtype: DType,
    comptime rank: usize,
    x: *const Tensor,
    mask: *const tensor.TensorOf(mask_dtype),
    comptime axis: usize,
    empty_value: ?f32,
    counts_out: *?Tensor,
    comptime want_mean: bool,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");

    const source = try x.rankView(rank);
    const mask_view = try mask.rankView(rank);
    for (0..rank) |dim| {
        if (source.shape[dim] != mask_view.shape[dim]) return tensor.TensorError.ShapeMismatch;
    }

    const out_rank = if (rank == 1) 1 else rank - 1;
    const out_shape = shapeWithoutAxis(rank, out_rank, source.shape, axis);

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const xp = xx.tensor();
    const input = xp.dataConst();

    var mm = try rt.prepareContiguousTyped(mask_dtype, mask);
    defer mm.deinit();
    const flags = mm.tensor().dataConst();

    var out = try rt.zerosRank(out_rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    var counts = try rt.zerosRank(out_rank, out_shape);
    errdefer counts.deinit();
    const count_data = counts.data();

    const axis_dim = source.shape[axis];
    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);

    // Both arms are branchless selects rather than skips. A data-dependent
    // branch (or a data-dependent store index) per element measured 2.8x
    // SLOWER than the two-pass composition it replaces on the non-last-axis
    // shape; a select vectorizes and keeps every access contiguous.
    //
    // Substituting the identity for an excluded element rather than skipping
    // it also makes the arithmetic EXACTLY the composed spelling's
    // (`maskedFill(mask, 0)` then reduce): same values, same kernel, same
    // order, so the fused entry is bitwise equal to the composition it
    // replaces, and bitwise equal to the unmasked reduction when the mask is
    // all-true. A select is NaN-correct where a multiply by 0/1 would not be:
    // an excluded NaN becomes the identity, it does not poison the lane.
    if (inner == 1) {
        // Last-axis (and rank-1) reduction: write one row through the mask
        // into pooled scratch (a single cache-resident row, not a copy of the
        // tensor) and hand it to the same SIMD kernel the unmasked path uses.
        var scratch = try rt.emptyRank(1, .{axis_dim});
        defer scratch.deinit();
        const row_scratch = scratch.data();
        rt.enableNativeVectorPoolForWork(axis_dim, parallel.vector_elementwise_len_threshold);

        // Counting is a separate pass over the (now L1-resident) flag row, and
        // only when someone reads it: fusing a scalar `selected` accumulator
        // into the select loop serializes it and blocks vectorization.
        const need_counts = want_mean or empty_value != null;
        for (0..outer) |outer_i| {
            const row = input[outer_i * axis_dim ..][0..axis_dim];
            const row_flags = flags[outer_i * axis_dim ..][0..axis_dim];
            selectRow(mask_dtype, row, row_flags, row_scratch);
            output[outer_i] = rt.backend.sumSlice(row_scratch);
            if (need_counts) count_data[outer_i] = @floatFromInt(countRow(mask_dtype, row_flags));
        }
    } else {
        // Non-last axis: accumulate in linear order, exactly what the unmasked
        // non-last-axis arm does, with all four arrays walked contiguously.
        for (0..outer) |outer_i| {
            const base = outer_i * axis_dim * inner;
            const out_base = outer_i * inner;
            const out_row = output[out_base..][0..inner];
            const count_row = count_data[out_base..][0..inner];
            for (0..axis_dim) |a| {
                const in_row = input[base + a * inner ..][0..inner];
                const flag_row = flags[base + a * inner ..][0..inner];
                for (in_row, flag_row, out_row, count_row) |value, keep, *acc, *count| {
                    const on = dtype_mod.isTruthy(mask_dtype, keep);
                    acc.* += if (on) value else 0;
                    count.* += if (on) 1 else 0;
                }
            }
        }
    }

    // Empty lanes: the identity for a sum, NaN for a mean, or the caller's
    // sentinel for either.
    const resolved_empty = empty_value orelse if (want_mean) std.math.nan(f32) else @as(f32, 0);
    if (comptime want_mean) {
        for (output, count_data) |*value, count| {
            value.* = if (count == 0) resolved_empty else value.* / count;
        }
    } else if (empty_value != null) {
        for (output, count_data) |*value, count| {
            if (count == 0) value.* = resolved_empty;
        }
    }

    counts_out.* = counts;
    return out;
}

pub fn meanAxisRankTyped(
    rt: *Runtime,
    comptime dtype: DType,
    comptime rank: usize,
    x: *const tensor.TensorOf(dtype),
    comptime axis: usize,
) !tensor.TensorOf(dtype_mod.outputDType(.reduction, dtype)) {
    if (comptime dtype == .f32) return meanAxisRank(rt, rank, x, axis);
    comptime ensureForwardFloatMath(dtype);
    const compute_dtype = comptime dtype_mod.computeDType(.reduction, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.reduction, dtype);
    var out = try sumAxisRankTyped(rt, dtype, rank, x, axis);
    errdefer out.deinit();
    const scale_value: dtype_mod.Scalar(compute_dtype) = 1 / @as(dtype_mod.Scalar(compute_dtype), @floatFromInt(x.shape.at(axis)));
    for (out.data()) |*value| {
        const scaled = dtype_mod.castFloat(output_dtype, compute_dtype, value.*) * scale_value;
        value.* = dtype_mod.castFloat(compute_dtype, output_dtype, scaled);
    }
    return out;
}

/// Segmented sum along `axis` over contiguous index ranges: `offsets` has
/// length `s + 1`, is nondecreasing, and spans the axis exactly
/// (`offsets[0] == 0`, `offsets[s] == shape[axis]`);
/// `out[..., i, ...] = Σ_{j in [offsets[i], offsets[i+1])} x[..., j, ...]`
/// and the axis size becomes `s` (empty segments produce zero rows). Each
/// segment accumulates serially in axis order — bitwise deterministic for
/// any thread count (cold op, no parallel dispatch). The sorted-contiguous
/// restriction of torch.segment_reduce / JAX segment_sum.
pub fn segmentSumAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    x: *const Tensor,
    comptime axis: usize,
    offsets: []const usize,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    const source = try x.rankView(rank);
    if (offsets.len < 2) return tensor.TensorError.InvalidShape;
    const segments = offsets.len - 1;
    if (offsets[0] != 0 or offsets[segments] != source.shape[axis]) return tensor.TensorError.InvalidShape;
    for (offsets[0..segments], offsets[1..]) |lo, hi| {
        if (lo > hi) return tensor.TensorError.InvalidShape;
    }

    var out_shape: [rank]usize = source.shape;
    out_shape[axis] = segments;

    var xx = try rt.prepareContiguous(x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.zerosRank(rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, source.shape, axis);
    const outer = productBeforeAxis(rank, source.shape, axis);
    const n = source.shape[axis];
    for (0..outer) |o| {
        for (0..segments) |s_i| {
            const out_base = (o * segments + s_i) * inner;
            for (offsets[s_i]..offsets[s_i + 1]) |j| {
                const in_base = (o * n + j) * inner;
                for (0..inner) |c| output[out_base + c] += input[in_base + c];
            }
        }
    }
    return out;
}

/// `segmentSumAxisRank`'s VJP companion: expand the segment axis back to
/// `n` rows, every input row receiving its segment's row:
/// `out[..., j, ...] = gy[..., seg(j), ...]`.
pub fn segmentBroadcastAxisRank(
    rt: *Runtime,
    comptime rank: usize,
    gy: *const Tensor,
    comptime axis: usize,
    offsets: []const usize,
    n: usize,
) !Tensor {
    if (rank == 0 or rank > tensor.max_rank) @compileError("invalid tensor rank");
    if (axis >= rank) @compileError("axis out of bounds");
    const source = try gy.rankView(rank);
    if (offsets.len < 2) return tensor.TensorError.InvalidShape;
    const segments = offsets.len - 1;
    if (source.shape[axis] != segments) return tensor.TensorError.ShapeMismatch;
    if (offsets[0] != 0 or offsets[segments] != n) return tensor.TensorError.InvalidShape;

    var out_shape: [rank]usize = source.shape;
    out_shape[axis] = n;

    var gg = try rt.prepareContiguous(gy);
    defer gg.deinit();
    const input = gg.tensor().dataConst();

    var out = try rt.zerosRank(rank, out_shape);
    errdefer out.deinit();
    const output = out.data();

    const inner = productAfterAxis(rank, out_shape, axis);
    const outer = productBeforeAxis(rank, out_shape, axis);
    for (0..outer) |o| {
        for (0..segments) |s_i| {
            const in_base = (o * segments + s_i) * inner;
            for (offsets[s_i]..offsets[s_i + 1]) |j| {
                const out_base = (o * n + j) * inner;
                @memcpy(output[out_base..][0..inner], input[in_base..][0..inner]);
            }
        }
    }
    return out;
}
