//! Sum / mean reductions (whole-tensor + along-axis, typed) + the cumulative
//! (prefix/suffix) sums.
//!
//! Domain module: every op receives an explicit `*Runtime`. Reduction math
//! dispatches to the backend's `sumInto`/`sumSlice(Typed)` kernels. Carries a
//! local copy of `ensureForwardFloatMath` (precedent: matmul.zig) to stay
//! leaf-ward of `exec.zig`.

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
