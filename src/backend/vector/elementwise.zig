//! Elementwise vector kernels: the contiguous entry points, their parallel
//! dispatch over `tile.forRange`/`reduceRange`, and the typed scalar inner
//! kernels. Shared-core symbols (ParallelConfig,
//! contiguous-data helpers, elementwiseThreadCount) come from `common.zig`;
//! the @Vector primitives from `primitives.zig`; op definitions from
//! `../ops.zig`.

const std = @import("std");
const isa = @import("../isa.zig");
const ops = @import("../ops.zig");
const dtype_mod = @import("../../dtype.zig");
const parallel = @import("../../parallel.zig");
const tensor = @import("../../tensor.zig");
const common = @import("common.zig");
const tile = @import("tile.zig");
const primitives = @import("primitives.zig");

const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

const ParallelConfig = common.ParallelConfig;
const elementwiseThreadCount = common.elementwiseThreadCount;
const contiguousDataConst = common.contiguousDataConst;
const contiguousData = common.contiguousData;
const Vf32 = common.Vf32;
const vector_len = common.vector_len;

// ---------------- Elementwise ----------------

// ---------------- Dtype cast rows ----------------

/// Vector twin of `dtype.f32ToBf16` — bit-identical lanes: round-to-nearest-
/// even via the (bits + 0x7fff + lsb) trick, NaN quieted with bit 6 set.
pub fn castF32ToBf16(output: []u16, input: []const f32) void {
    if (comptime isa.reference) return scalar.castF32ToBf16(output, input);
    const width = std.simd.suggestVectorLength(f32) orelse 8;
    var i: usize = 0;
    while (i + width <= input.len) : (i += width) {
        output[i..][0..width].* = f32ToBf16Lanes(width, input[i..][0..width].*);
    }
    while (i < input.len) : (i += 1) output[i] = dtype_mod.f32ToBf16(input[i]);
}

fn f32ToBf16Lanes(comptime width: usize, values: @Vector(width, f32)) @Vector(width, u16) {
    const U32 = @Vector(width, u32);
    const bits: U32 = @bitCast(values);
    const abs = bits & @as(U32, @splat(0x7fff_ffff));
    const is_nan = abs > @as(U32, @splat(0x7f80_0000));
    const high = bits >> @as(@Vector(width, u5), @splat(16));
    const lsb = high & @as(U32, @splat(1));
    // Never overflows for non-NaN inputs (max non-NaN is ±inf, 0xff80_0000);
    // NaN lanes take the quieting arm via @select.
    const rounded = (bits +% @as(U32, @splat(0x7fff)) +% lsb) >> @as(@Vector(width, u5), @splat(16));
    const quieted = high | @as(U32, @splat(64));
    return @truncate(@select(u32, is_nan, quieted, rounded));
}

/// Vector twin of `dtype.bf16ToF32` — exact (bits << 16).
pub fn castBf16ToF32(output: []f32, input: []const u16) void {
    if (comptime isa.reference) return scalar.castBf16ToF32(output, input);
    const width = std.simd.suggestVectorLength(f32) orelse 8;
    const U32 = @Vector(width, u32);
    var i: usize = 0;
    while (i + width <= input.len) : (i += width) {
        const bits: @Vector(width, u16) = input[i..][0..width].*;
        const widened = @as(U32, bits) << @as(@Vector(width, u5), @splat(16));
        output[i..][0..width].* = @bitCast(widened);
    }
    while (i < input.len) : (i += 1) output[i] = dtype_mod.bf16ToF32(input[i]);
}

pub fn castF32ToF16(output: []f16, input: []const f32) void {
    if (comptime isa.reference) return scalar.castF32ToF16(output, input);
    const width = std.simd.suggestVectorLength(f16) orelse 8;
    const F32 = @Vector(width, f32);
    const F16 = @Vector(width, f16);
    var i: usize = 0;
    while (i + 4 * width <= input.len) : (i += 4 * width) {
        output[i..][0..width].* = @as(F16, @floatCast(@as(F32, input[i..][0..width].*)));
        output[i + width ..][0..width].* = @as(F16, @floatCast(@as(F32, input[i + width ..][0..width].*)));
        output[i + 2 * width ..][0..width].* = @as(F16, @floatCast(@as(F32, input[i + 2 * width ..][0..width].*)));
        output[i + 3 * width ..][0..width].* = @as(F16, @floatCast(@as(F32, input[i + 3 * width ..][0..width].*)));
    }
    while (i + width <= input.len) : (i += width) {
        output[i..][0..width].* = @as(F16, @floatCast(@as(F32, input[i..][0..width].*)));
    }
    while (i < input.len) : (i += 1) output[i] = @floatCast(input[i]);
}

pub fn castF16ToF32(output: []f32, input: []const f16) void {
    if (comptime isa.reference) return scalar.castF16ToF32(output, input);
    const width = std.simd.suggestVectorLength(f16) orelse 8;
    const F16 = @Vector(width, f16);
    const F32 = @Vector(width, f32);
    var i: usize = 0;
    while (i + 4 * width <= input.len) : (i += 4 * width) {
        output[i..][0..width].* = @as(F32, @floatCast(@as(F16, input[i..][0..width].*)));
        output[i + width ..][0..width].* = @as(F32, @floatCast(@as(F16, input[i + width ..][0..width].*)));
        output[i + 2 * width ..][0..width].* = @as(F32, @floatCast(@as(F16, input[i + 2 * width ..][0..width].*)));
        output[i + 3 * width ..][0..width].* = @as(F32, @floatCast(@as(F16, input[i + 3 * width ..][0..width].*)));
    }
    while (i + width <= input.len) : (i += width) {
        output[i..][0..width].* = @as(F32, @floatCast(@as(F16, input[i..][0..width].*)));
    }
    while (i < input.len) : (i += 1) output[i] = @floatCast(input[i]);
}

pub fn addInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    try tensor.requireSameShape(a, b);
    try tensor.requireSameShape(out, a);
    addContiguousIntoUnchecked(.{}, out, a, b, a.len());
}

pub fn addContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    if (comptime isa.reference) return scalar.addContiguousIntoUnchecked(out, a, b, len);
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, primitives.vecAdd, z, x, y)) return;
    primitives.vecAdd(z, x, y);
}

pub fn subInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    try tensor.requireSameShape(a, b);
    try tensor.requireSameShape(out, a);
    subContiguousIntoUnchecked(.{}, out, a, b, a.len());
}

pub fn subContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    if (comptime isa.reference) return scalar.subContiguousIntoUnchecked(out, a, b, len);
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, primitives.vecSub, z, x, y)) return;
    primitives.vecSub(z, x, y);
}

pub fn mulInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    try tensor.requireSameShape(a, b);
    try tensor.requireSameShape(out, a);
    mulContiguousIntoUnchecked(.{}, out, a, b, a.len());
}

pub fn mulContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    if (comptime isa.reference) return scalar.mulContiguousIntoUnchecked(out, a, b, len);
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, primitives.vecMul, z, x, y)) return;
    primitives.vecMul(z, x, y);
}

pub fn divContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    if (comptime isa.reference) return scalar.divContiguousIntoUnchecked(out, a, b, len);
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, primitives.vecDiv, z, x, y)) return;
    primitives.vecDiv(z, x, y);
}

pub fn maximumContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    if (comptime isa.reference) return scalar.maximumContiguousIntoUnchecked(out, a, b, len);
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, primitives.vecMaximum, z, x, y)) return;
    primitives.vecMaximum(z, x, y);
}

pub fn minimumContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    if (comptime isa.reference) return scalar.minimumContiguousIntoUnchecked(out, a, b, len);
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, primitives.vecMinimum, z, x, y)) return;
    primitives.vecMinimum(z, x, y);
}

pub fn elementwiseContiguousIntoTyped(
    pc: ParallelConfig,
    comptime dtype: DType,
    comptime op: ops.ElementwiseOp,
    out: *tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)),
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
    len: usize,
) void {
    _ = pc;
    if (comptime isa.reference) return scalar.elementwiseContiguousIntoTyped(dtype, op, out, a, b, len);
    const x = common.contiguousDataConstOf(dtype, a, len);
    const y = common.contiguousDataConstOf(dtype, b, len);
    const z = common.contiguousDataOf(dtype_mod.outputDType(.pointwise, dtype), out, len);

    if (comptime dtype == .f64) {
        return primitives.vecElementwiseF64(op, z, x, y);
    } else if (comptime dtype == .f16) {
        return primitives.vecElementwiseF16(op, z, x, y);
    } else if (comptime dtype == .bf16) {
        return primitives.vecElementwiseBf16(op, z, x, y);
    }
    elementwiseSlicesTyped(dtype, op, z, x, y);
}

pub fn scaleInto(pc: ParallelConfig, out: *Tensor, a: *const Tensor, scalar_value: f32) !void {
    if (comptime isa.reference) return scalar.scaleInto(out, a, scalar_value);
    try tensor.requireSameShape(out, a);
    const x = a.dataConst();
    const z = out.data();
    if (maybeParallelScale(pc, z, x, scalar_value)) return;
    primitives.vecScale(z, x, scalar_value);
}

pub fn addScaledSlice(z: []f32, x: []const f32, scalar_value: f32) void {
    if (comptime isa.reference) return scalar.addScaledSlice(z, x, scalar_value);
    primitives.vecAddScaled(z, x, scalar_value);
}

/// `z[row] += row_vector` for every row, then `op` (a fused bias + activation
/// when given).
pub fn addRowVectorSlice(comptime op: ?ops.UnaryOp, z: []f32, row_vector: []const f32, rows: usize, cols: usize) void {
    if (comptime isa.reference) return scalar.addRowVectorSlice(op, z, row_vector, rows, cols);
    std.debug.assert(z.len >= rows * cols);
    std.debug.assert(row_vector.len == cols);
    for (0..rows) |row_i| {
        const row = z[row_i * cols ..][0..cols];
        if (comptime op) |actual_op| {
            primitives.vecAddUnary(actual_op, row, row, row_vector);
        } else {
            primitives.vecAdd(row, row, row_vector);
        }
    }
}

// --- per-channel row kernels (channel-last maps: rows = spatial, cols = C) ---
//
// PReLU and the inference-BatchNorm affine, one pass each. Both are
// value-identical to the equivalent multi-op composition (select == relu +
// α·(x−relu); mul-then-add, deliberately NOT @mulAdd — Zig does not contract,
// so scalar and vector backends stay bitwise-identical and each fused op
// reproduces the composed ops' values exactly). Parallel over row ranges (each
// row is disjoint; bit-identical to serial).

/// Per-channel row-kernel payload (PReLU, channelAffine): `a` is the
/// per-channel vector, `b` the optional second one.
const RowChanCtx = struct {
    z: []f32,
    x: []const f32,
    a: []const f32,
    b: ?[]const f32,
    cols: usize,
};

fn maybeParallelRowChan(
    pc: ParallelConfig,
    comptime runFn: fn (RowChanCtx, usize, usize) void,
    z: []f32,
    x: []const f32,
    a: []const f32,
    b: ?[]const f32,
    rows: usize,
    cols: usize,
) bool {
    const pool = pc.pool orelse return false;
    const thread_count = @min(elementwiseThreadCount(rows * cols), rows);
    if (thread_count <= 1) return false;
    tile.forRange(pool, RowChanCtx, .{ .z = z, .x = x, .a = a, .b = b, .cols = cols }, rows, thread_count, runFn);
    return true;
}

fn runPreluChannelsRows(c: RowChanCtx, row_start: usize, row_end: usize) void {
    preluChannelsRows(c.z, c.x, c.a, c.cols, row_start, row_end);
}

/// PReLU with a per-channel slope: `z[r,c] = x > 0 ? x : α[c]·x`.
pub fn preluChannelsInto(pc: ParallelConfig, z: []f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize) void {
    if (comptime isa.reference) return scalar.preluChannelsInto(z, x, alpha, rows, cols);
    std.debug.assert(z.len >= rows * cols and x.len >= rows * cols);
    std.debug.assert(alpha.len == cols);
    if (maybeParallelRowChan(pc, runPreluChannelsRows, z, x, alpha, null, rows, cols)) return;
    preluChannelsRows(z, x, alpha, cols, 0, rows);
}

fn preluChannelsRows(z: []f32, x: []const f32, alpha: []const f32, cols: usize, row_start: usize, row_end: usize) void {
    const vzero: Vf32 = @splat(0);
    var r = row_start;
    while (r < row_end) : (r += 1) {
        const zr = z[r * cols ..][0..cols];
        const xr = x[r * cols ..][0..cols];
        var i: usize = 0;
        while (i + vector_len <= cols) : (i += vector_len) {
            const vx: Vf32 = xr[i..][0..vector_len].*;
            const va: Vf32 = alpha[i..][0..vector_len].*;
            zr[i..][0..vector_len].* = @select(f32, vx > vzero, vx, vx * va);
        }
        while (i < cols) : (i += 1) {
            const v = xr[i];
            zr[i] = if (v > 0) v else v * alpha[i];
        }
    }
}

fn runChannelAffineRows(c: RowChanCtx, row_start: usize, row_end: usize) void {
    channelAffineRows(c.z, c.x, c.a, c.b, c.cols, row_start, row_end);
}

/// Per-channel affine (frozen-stats BatchNorm): `z[r,c] = x·scale[c] + shift[c]`;
/// a null `shift` degrades to the per-channel scale `z = x·scale[c]` (the
/// affine's own input-VJP).
pub fn channelAffineInto(pc: ParallelConfig, z: []f32, x: []const f32, scale: []const f32, shift: ?[]const f32, rows: usize, cols: usize) void {
    if (comptime isa.reference) return scalar.channelAffineInto(z, x, scale, shift, rows, cols);
    std.debug.assert(z.len >= rows * cols and x.len >= rows * cols);
    std.debug.assert(scale.len == cols and (shift == null or shift.?.len == cols));
    if (maybeParallelRowChan(pc, runChannelAffineRows, z, x, scale, shift, rows, cols)) return;
    channelAffineRows(z, x, scale, shift, cols, 0, rows);
}

fn channelAffineRows(z: []f32, x: []const f32, scale: []const f32, shift: ?[]const f32, cols: usize, row_start: usize, row_end: usize) void {
    var r = row_start;
    while (r < row_end) : (r += 1) {
        const zr = z[r * cols ..][0..cols];
        const xr = x[r * cols ..][0..cols];
        if (shift) |t| {
            var i: usize = 0;
            while (i + vector_len <= cols) : (i += vector_len) {
                const vx: Vf32 = xr[i..][0..vector_len].*;
                const vs: Vf32 = scale[i..][0..vector_len].*;
                const vt: Vf32 = t[i..][0..vector_len].*;
                zr[i..][0..vector_len].* = vx * vs + vt;
            }
            while (i < cols) : (i += 1) zr[i] = xr[i] * scale[i] + t[i];
        } else {
            var i: usize = 0;
            while (i + vector_len <= cols) : (i += vector_len) {
                const vx: Vf32 = xr[i..][0..vector_len].*;
                const vs: Vf32 = scale[i..][0..vector_len].*;
                zr[i..][0..vector_len].* = vx * vs;
            }
            while (i < cols) : (i += 1) zr[i] = xr[i] * scale[i];
        }
    }
}

/// PReLU input-VJP: `gx[r,c] = x > 0 ? gy : α[c]·gy` (subgradient 0 at the
/// kink follows the forward's `>` test, matching the composed relu VJP).
pub fn preluChannelsBackwardInputInto(pc: ParallelConfig, gx: []f32, gy: []const f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize) void {
    if (comptime isa.reference) return scalar.preluChannelsBackwardInputInto(gx, gy, x, alpha, rows, cols);
    std.debug.assert(gx.len >= rows * cols and gy.len >= rows * cols and x.len >= rows * cols);
    std.debug.assert(alpha.len == cols);
    _ = pc;
    const vzero: Vf32 = @splat(0);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const gxr = gx[r * cols ..][0..cols];
        const gyr = gy[r * cols ..][0..cols];
        const xr = x[r * cols ..][0..cols];
        var i: usize = 0;
        while (i + vector_len <= cols) : (i += vector_len) {
            const vx: Vf32 = xr[i..][0..vector_len].*;
            const vg: Vf32 = gyr[i..][0..vector_len].*;
            const va: Vf32 = alpha[i..][0..vector_len].*;
            gxr[i..][0..vector_len].* = @select(f32, vx > vzero, vg, vg * va);
        }
        while (i < cols) : (i += 1) {
            gxr[i] = if (xr[i] > 0) gyr[i] else gyr[i] * alpha[i];
        }
    }
}

/// PReLU slope-VJP: `gα[c] = Σ_rows gy·min(x, 0)` — serial row accumulation
/// (deterministic order; the slope vector is small).
pub fn preluChannelsBackwardAlphaInto(pc: ParallelConfig, galpha: []f32, gy: []const f32, x: []const f32, rows: usize, cols: usize) void {
    if (comptime isa.reference) return scalar.preluChannelsBackwardAlphaInto(galpha, gy, x, rows, cols);
    std.debug.assert(galpha.len == cols);
    std.debug.assert(gy.len >= rows * cols and x.len >= rows * cols);
    _ = pc;
    @memset(galpha, 0);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const gyr = gy[r * cols ..][0..cols];
        const xr = x[r * cols ..][0..cols];
        for (galpha, gyr, xr) |*ga, g, v| {
            if (v <= 0) ga.* += g * v;
        }
    }
}

pub fn unaryContiguousIntoUnchecked(
    pc: ParallelConfig,
    comptime op: ops.UnaryOp,
    out: *Tensor,
    a: *const Tensor,
    len: usize,
) void {
    if (comptime isa.reference) return scalar.unaryContiguousIntoUnchecked(op, out, a, len);
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    if (maybeParallelUnary(pc, op, z, x)) return;
    primitives.vecUnary(op, z, x);
}

pub fn leakyReluContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    negative_slope: f32,
) void {
    if (comptime isa.reference) return scalar.leakyReluContiguousIntoUnchecked(out, a, len, negative_slope);
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    if (maybeParallelLeakyRelu(pc, z, x, negative_slope)) return;
    primitives.vecLeakyRelu(z, x, negative_slope);
}

pub fn softcapContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    cap: f32,
) void {
    if (comptime isa.reference) return scalar.softcapContiguousIntoUnchecked(out, a, len, cap);
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    if (maybeParallelSoftcap(pc, z, x, cap)) return;
    primitives.vecSoftcap(z, x, cap);
}

pub fn clampContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    min_value: f32,
    max_value: f32,
) void {
    if (comptime isa.reference) return scalar.clampContiguousIntoUnchecked(out, a, len, min_value, max_value);
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    if (maybeParallelClamp(pc, z, x, min_value, max_value)) return;
    primitives.vecClamp(z, x, min_value, max_value);
}

pub fn gatedContiguousIntoUnchecked(
    pc: ParallelConfig,
    comptime op: ops.GatedOp,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    if (comptime isa.reference) return scalar.gatedContiguousIntoUnchecked(op, out, a, b, len);
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelGated(pc, op, z, x, y)) return;
    primitives.vecGated(op, z, x, y);
}

pub fn sumInto(pc: ParallelConfig, out: *Tensor, a: *const Tensor) !void {
    if (comptime isa.reference) return scalar.sumInto(out, a);
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    out.data()[0] = parallelVecSum(pc, a.dataConst()) orelse primitives.vecSum(a.dataConst());
}

pub fn sumSlice(values: []const f32) f32 {
    if (comptime isa.reference) return scalar.sumSlice(values);
    return primitives.vecSum(values);
}

pub fn prodInto(pc: ParallelConfig, out: *Tensor, a: *const Tensor) !void {
    if (comptime isa.reference) return scalar.prodInto(out, a);
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    out.data()[0] = parallelVecProd(pc, a.dataConst()) orelse primitives.vecProd(a.dataConst());
}

pub fn prodSlice(values: []const f32) f32 {
    if (comptime isa.reference) return scalar.prodSlice(values);
    return primitives.vecProd(values);
}

pub fn sumSliceTyped(
    pc: ParallelConfig,
    comptime dtype: DType,
    values: []const dtype_mod.Scalar(dtype),
) dtype_mod.Scalar(dtype_mod.outputDType(.reduction, dtype)) {
    _ = pc;
    if (comptime isa.reference) return sumSliceTypedScalar(dtype, values);
    if (comptime dtype == .f64) {
        return primitives.vecSumF64(values);
    } else if (comptime dtype == .f16) {
        return primitives.vecSumF16ToF32(values);
    } else if (comptime dtype == .bf16) {
        return primitives.vecSumBf16ToF32(values);
    }
    return sumSliceTypedScalar(dtype, values);
}

pub fn dotInto(pc: ParallelConfig, out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    if (comptime isa.reference) return scalar.dot(.f32, out, a, b);
    try tensor.requireSameShape(a, b);
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    out.data()[0] = parallelVecDot(pc, a.dataConst(), b.dataConst()) orelse primitives.vecDot(a.dataConst(), b.dataConst());
}

pub fn dotIntoTyped(
    pc: ParallelConfig,
    comptime dtype: DType,
    out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !void {
    if (comptime isa.reference) return scalar.dot(dtype, out, a, b);
    try tensor.requireSameShape(a, b);
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    out.data()[0] = if (comptime dtype == .f64)
        parallelVecDotF64(pc, a.dataConst(), b.dataConst()) orelse primitives.vecDotF64(a.dataConst(), b.dataConst())
    else if (comptime dtype == .f16)
        dtype_mod.castFloat(.f32, .f16, parallelVecDotF16ToF32(pc, a.dataConst(), b.dataConst()) orelse primitives.vecDotF16ToF32(a.dataConst(), b.dataConst()))
    else if (comptime dtype == .bf16)
        dtype_mod.f32ToBf16(parallelVecDotBf16ToF32(pc, a.dataConst(), b.dataConst()) orelse primitives.vecDotBf16ToF32(a.dataConst(), b.dataConst()))
    else
        dotSliceTypedScalar(dtype, a.dataConst(), b.dataConst());
}

// ---------------- Inner kernels ----------------

fn maybeParallelBinary(
    pc: ParallelConfig,
    // `anytype`: the binary primitives are `inline fn`s.
    comptime vecFn: anytype,
    z: []f32,
    x: []const f32,
    y: []const f32,
) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;
    const Ctx = struct { z: []f32, x: []const f32, y: []const f32 };
    tile.forRange(pool, Ctx, .{ .z = z, .x = x, .y = y }, z.len, thread_count, struct {
        fn go(c: Ctx, start: usize, end: usize) void {
            vecFn(c.z[start..end], c.x[start..end], c.y[start..end]);
        }
    }.go);
    return true;
}

fn maybeParallelScale(pc: ParallelConfig, z: []f32, x: []const f32, scalar_value: f32) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;
    const Ctx = struct { z: []f32, x: []const f32, scalar: f32 };
    tile.forRange(pool, Ctx, .{ .z = z, .x = x, .scalar = scalar_value }, z.len, thread_count, struct {
        fn go(c: Ctx, start: usize, end: usize) void {
            primitives.vecScale(c.z[start..end], c.x[start..end], c.scalar);
        }
    }.go);
    return true;
}

fn maybeParallelUnary(pc: ParallelConfig, comptime op: ops.UnaryOp, z: []f32, x: []const f32) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;
    const Ctx = struct { z: []f32, x: []const f32 };
    tile.forRange(pool, Ctx, .{ .z = z, .x = x }, z.len, thread_count, struct {
        fn go(c: Ctx, start: usize, end: usize) void {
            primitives.vecUnary(op, c.z[start..end], c.x[start..end]);
        }
    }.go);
    return true;
}

fn maybeParallelLeakyRelu(pc: ParallelConfig, z: []f32, x: []const f32, negative_slope: f32) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;
    const Ctx = struct { z: []f32, x: []const f32, negative_slope: f32 };
    tile.forRange(pool, Ctx, .{ .z = z, .x = x, .negative_slope = negative_slope }, z.len, thread_count, struct {
        fn go(c: Ctx, start: usize, end: usize) void {
            primitives.vecLeakyRelu(c.z[start..end], c.x[start..end], c.negative_slope);
        }
    }.go);
    return true;
}

fn maybeParallelSoftcap(pc: ParallelConfig, z: []f32, x: []const f32, cap: f32) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;
    const Ctx = struct { z: []f32, x: []const f32, cap: f32 };
    tile.forRange(pool, Ctx, .{ .z = z, .x = x, .cap = cap }, z.len, thread_count, struct {
        fn go(c: Ctx, start: usize, end: usize) void {
            primitives.vecSoftcap(c.z[start..end], c.x[start..end], c.cap);
        }
    }.go);
    return true;
}

fn maybeParallelClamp(pc: ParallelConfig, z: []f32, x: []const f32, min_value: f32, max_value: f32) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;
    const Ctx = struct { z: []f32, x: []const f32, min_value: f32, max_value: f32 };
    tile.forRange(pool, Ctx, .{ .z = z, .x = x, .min_value = min_value, .max_value = max_value }, z.len, thread_count, struct {
        fn go(c: Ctx, start: usize, end: usize) void {
            primitives.vecClamp(c.z[start..end], c.x[start..end], c.min_value, c.max_value);
        }
    }.go);
    return true;
}

fn maybeParallelGated(pc: ParallelConfig, comptime op: ops.GatedOp, z: []f32, x: []const f32, y: []const f32) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;
    const Ctx = struct { z: []f32, x: []const f32, y: []const f32 };
    tile.forRange(pool, Ctx, .{ .z = z, .x = x, .y = y }, z.len, thread_count, struct {
        fn go(c: Ctx, start: usize, end: usize) void {
            primitives.vecGated(op, c.z[start..end], c.x[start..end], c.y[start..end]);
        }
    }.go);
    return true;
}

const SliceCtx = struct { x: []const f32 };

fn addF32(a: f32, b: f32) f32 {
    return a + b;
}

fn mulF32(a: f32, b: f32) f32 {
    return a * b;
}

fn addF64(a: f64, b: f64) f64 {
    return a + b;
}

fn parallelVecSum(pc: ParallelConfig, x: []const f32) ?f32 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;
    return tile.reduceRange(pool, f32, SliceCtx, .{ .x = x }, x.len, thread_count, 0, struct {
        fn go(c: SliceCtx, start: usize, end: usize) f32 {
            return primitives.vecSum(c.x[start..end]);
        }
    }.go, addF32);
}

fn parallelVecProd(pc: ParallelConfig, x: []const f32) ?f32 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;
    return tile.reduceRange(pool, f32, SliceCtx, .{ .x = x }, x.len, thread_count, 1, struct {
        fn go(c: SliceCtx, start: usize, end: usize) f32 {
            return primitives.vecProd(c.x[start..end]);
        }
    }.go, mulF32);
}

fn parallelVecDot(pc: ParallelConfig, x: []const f32, y: []const f32) ?f32 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;
    const Ctx = struct { x: []const f32, y: []const f32 };
    return tile.reduceRange(pool, f32, Ctx, .{ .x = x, .y = y }, x.len, thread_count, 0, struct {
        fn go(c: Ctx, start: usize, end: usize) f32 {
            return primitives.vecDot(c.x[start..end], c.y[start..end]);
        }
    }.go, addF32);
}

fn parallelVecDotF64(pc: ParallelConfig, x: []const f64, y: []const f64) ?f64 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;
    const Ctx = struct { x: []const f64, y: []const f64 };
    return tile.reduceRange(pool, f64, Ctx, .{ .x = x, .y = y }, x.len, thread_count, 0, struct {
        fn go(c: Ctx, start: usize, end: usize) f64 {
            return primitives.vecDotF64(c.x[start..end], c.y[start..end]);
        }
    }.go, addF64);
}

fn parallelVecDotF16ToF32(pc: ParallelConfig, x: []const f16, y: []const f16) ?f32 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;
    const Ctx = struct { x: []const f16, y: []const f16 };
    return tile.reduceRange(pool, f32, Ctx, .{ .x = x, .y = y }, x.len, thread_count, 0, struct {
        fn go(c: Ctx, start: usize, end: usize) f32 {
            return primitives.vecDotF16ToF32(c.x[start..end], c.y[start..end]);
        }
    }.go, addF32);
}

fn parallelVecDotBf16ToF32(pc: ParallelConfig, x: []const u16, y: []const u16) ?f32 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;
    const Ctx = struct { x: []const u16, y: []const u16 };
    return tile.reduceRange(pool, f32, Ctx, .{ .x = x, .y = y }, x.len, thread_count, 0, struct {
        fn go(c: Ctx, start: usize, end: usize) f32 {
            return primitives.vecDotBf16ToF32(c.x[start..end], c.y[start..end]);
        }
    }.go, addF32);
}

fn elementwiseSlicesTyped(
    comptime dtype: DType,
    comptime op: ops.ElementwiseOp,
    out: []dtype_mod.Scalar(dtype_mod.outputDType(.pointwise, dtype)),
    a: []const dtype_mod.Scalar(dtype),
    b: []const dtype_mod.Scalar(dtype),
) void {
    for (out, a, b) |*dst, av, bv| {
        dst.* = primitives.applyElementwiseTyped(dtype, op, av, bv);
    }
}

fn sumSliceTypedScalar(
    comptime dtype: DType,
    values: []const dtype_mod.Scalar(dtype),
) dtype_mod.Scalar(dtype_mod.outputDType(.reduction, dtype)) {
    const compute_dtype = comptime dtype_mod.computeDType(.reduction, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.reduction, dtype);
    var acc: dtype_mod.Scalar(compute_dtype) = 0;
    for (values) |value| {
        acc += dtype_mod.castFloat(dtype, compute_dtype, value);
    }
    return dtype_mod.castFloat(compute_dtype, output_dtype, acc);
}

fn dotSliceTypedScalar(
    comptime dtype: DType,
    a: []const dtype_mod.Scalar(dtype),
    b: []const dtype_mod.Scalar(dtype),
) dtype_mod.Scalar(dtype_mod.outputDType(.matmul, dtype)) {
    const compute_dtype = comptime dtype_mod.computeDType(.matmul, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.matmul, dtype);
    var acc: dtype_mod.Scalar(compute_dtype) = 0;
    for (a, b) |av, bv| {
        acc += dtype_mod.castFloat(dtype, compute_dtype, av) * dtype_mod.castFloat(dtype, compute_dtype, bv);
    }
    return dtype_mod.castFloat(compute_dtype, output_dtype, acc);
}

// Single-row slice kernels for fused per-row activation math (the exact
// same vector cores the unfused elementwise ops apply; the scalar arm is
// the plain per-element loop).
pub fn unaryRowSlice(comptime op: ops.UnaryOp, z: []f32, x: []const f32) void {
    if (comptime isa.reference) return scalar.unaryRowSlice(op, z, x);
    primitives.vecUnary(op, z, x);
}

pub fn mulRowSlice(z: []f32, x: []const f32, y: []const f32) void {
    if (comptime isa.reference) return scalar.mulRowSlice(z, x, y);
    primitives.vecMul(z, x, y);
}

// ---------------- Snake activation (per-channel, DAC) ----------------

/// Per-channel Snake activation over contiguous `[rows, cols]` rows:
/// `y[t,c] = x[t,c] + inv_b[c] * sin(alpha[c]*x[t,c])^2`. `inv_b` is
/// precomputed by the caller at weight-load time (`1/(alpha + 1e-9)`, the DAC
/// convention) — the epsilon is deliberately NOT folded into the kernel.
/// Parallel over row ranges (disjoint writes ⇒ bit-identical to serial).
pub fn snakeInto(
    pc: ParallelConfig,
    out: *Tensor,
    x: *const Tensor,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
) void {
    if (comptime isa.reference) return scalar.snakeInto(out, x, alpha, inv_b, rows, cols);
    const input = contiguousDataConst(x, rows * cols);
    const output = contiguousData(out, rows * cols);
    if (maybeParallelSnake(pc, output, input, alpha, inv_b, rows, cols)) return;
    snakeRowsRange(output, input, alpha, inv_b, cols, 0, rows);
}

fn maybeParallelSnake(
    pc: ParallelConfig,
    out: []f32,
    x: []const f32,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
) bool {
    const pool = pc.pool orelse return false;
    const thread_count = @min(elementwiseThreadCount(out.len), rows);
    if (thread_count <= 1) return false;
    const Ctx = struct { out: []f32, x: []const f32, alpha: []const f32, inv_b: []const f32, cols: usize };
    tile.forRange(pool, Ctx, .{ .out = out, .x = x, .alpha = alpha, .inv_b = inv_b, .cols = cols }, rows, thread_count, struct {
        fn go(c: Ctx, row_start: usize, row_end: usize) void {
            snakeRowsRange(c.out, c.x, c.alpha, c.inv_b, c.cols, row_start, row_end);
        }
    }.go);
    return true;
}

fn snakeRowsRange(
    out: []f32,
    x: []const f32,
    alpha: []const f32,
    inv_b: []const f32,
    cols: usize,
    row_start: usize,
    row_end: usize,
) void {
    for (row_start..row_end) |r| {
        const x_row = x[r * cols ..][0..cols];
        const out_row = out[r * cols ..][0..cols];
        var c: usize = 0;
        while (c + vector_len <= cols) : (c += vector_len) {
            const xv: Vf32 = x_row[c..][0..vector_len].*;
            const av: Vf32 = alpha[c..][0..vector_len].*;
            const ibv: Vf32 = inv_b[c..][0..vector_len].*;
            const s = @sin(av * xv);
            out_row[c..][0..vector_len].* = xv + ibv * s * s;
        }
        while (c < cols) : (c += 1) {
            const s = @sin(alpha[c] * x_row[c]);
            out_row[c] = x_row[c] + inv_b[c] * s * s;
        }
    }
}

// ---------------- GroupNorm (ggml group_norm semantics) ----------------

/// GroupNorm over contiguous `[rows, cols]` (rows = time, cols = channels):
/// `groups` divides cols; per group the mean and biased variance are
/// accumulated in f64 over all rows × (cols/groups) elements (matching
/// ggml_compute_forward_group_norm), then `y = (x - mean) * (1/sqrt(var+eps))`
/// is applied in f32 (eps INSIDE the sqrt; the 1/sqrt scale is computed in f32
/// like ggml). Optional per-channel affine `y = y*weight[c] + bias[c]` is
/// applied after normalization. Parallel over whole groups — each group owns a
/// disjoint column slice, so the threaded result is bit-identical to serial.
pub fn groupNormInto(
    pc: ParallelConfig,
    out: *Tensor,
    x: *const Tensor,
    weight: ?[]const f32,
    bias: ?[]const f32,
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
) void {
    if (comptime isa.reference) return scalar.groupNormInto(out, x, weight, bias, rows, cols, groups, eps);
    const input = contiguousDataConst(x, rows * cols);
    const output = contiguousData(out, rows * cols);
    if (maybeParallelGroupNorm(pc, output, input, weight, bias, rows, cols, groups, eps)) return;
    groupNormGroupRange(output, input, weight, bias, rows, cols, groups, eps, 0, groups);
}

fn maybeParallelGroupNorm(
    pc: ParallelConfig,
    out: []f32,
    x: []const f32,
    weight: ?[]const f32,
    bias: ?[]const f32,
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
) bool {
    const pool = pc.pool orelse return false;
    const thread_count = @min(elementwiseThreadCount(out.len), groups);
    if (thread_count <= 1) return false;
    const Ctx = struct { out: []f32, x: []const f32, weight: ?[]const f32, bias: ?[]const f32, rows: usize, cols: usize, groups: usize, eps: f32 };
    tile.forRange(pool, Ctx, .{ .out = out, .x = x, .weight = weight, .bias = bias, .rows = rows, .cols = cols, .groups = groups, .eps = eps }, groups, thread_count, struct {
        fn go(c: Ctx, group_start: usize, group_end: usize) void {
            groupNormGroupRange(c.out, c.x, c.weight, c.bias, c.rows, c.cols, c.groups, c.eps, group_start, group_end);
        }
    }.go);
    return true;
}

fn groupNormGroupRange(
    out: []f32,
    x: []const f32,
    weight: ?[]const f32,
    bias: ?[]const f32,
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
    group_start: usize,
    group_end: usize,
) void {
    const cols_per_group = cols / groups;
    const count: f64 = @floatFromInt(rows * cols_per_group);
    for (group_start..group_end) |g| {
        const col_start = g * cols_per_group;
        var sum: f64 = 0;
        for (0..rows) |r| {
            const row = x[r * cols + col_start ..][0..cols_per_group];
            for (row) |v| sum += v;
        }
        const mean: f32 = @floatCast(sum / count);
        var sum2: f64 = 0;
        for (0..rows) |r| {
            const row = x[r * cols + col_start ..][0..cols_per_group];
            for (row) |v| {
                const centered = v - mean; // f32, like ggml's centered store
                sum2 += @as(f64, centered) * @as(f64, centered);
            }
        }
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatCast(sum2 / count + eps)));
        // Normalize + optional affine: element-independent, so the vector body
        // performs the exact scalar op sequence per lane (bit-identical).
        const mean_splat: Vf32 = @splat(mean);
        const scale_splat: Vf32 = @splat(scale);
        for (0..rows) |r| {
            const row = x[r * cols + col_start ..][0..cols_per_group];
            const out_row = out[r * cols + col_start ..][0..cols_per_group];
            var local_c: usize = 0;
            while (local_c + vector_len <= cols_per_group) : (local_c += vector_len) {
                const xv: Vf32 = row[local_c..][0..vector_len].*;
                var value = (xv - mean_splat) * scale_splat;
                if (weight) |w| {
                    const wv: Vf32 = w[col_start + local_c ..][0..vector_len].*;
                    value *= wv;
                }
                if (bias) |b| {
                    const bv: Vf32 = b[col_start + local_c ..][0..vector_len].*;
                    value += bv;
                }
                out_row[local_c..][0..vector_len].* = value;
            }
            while (local_c < cols_per_group) : (local_c += 1) {
                var value = (row[local_c] - mean) * scale;
                if (weight) |w| value *= w[col_start + local_c];
                if (bias) |b| value += b[col_start + local_c];
                out_row[local_c] = value;
            }
        }
    }
}

/// VJP of snakeInto wrt the input:
/// `gx[t,c] = gy[t,c] * (1 + inv_b[c]*alpha[c]*sin(2*alpha[c]*x[t,c]))`.
/// Parallel over row ranges (disjoint writes ⇒ bit-identical to serial).
pub fn snakeBackwardInputInto(
    pc: ParallelConfig,
    out: *Tensor,
    x: *const Tensor,
    gy: *const Tensor,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
) void {
    if (comptime isa.reference) return scalar.snakeBackwardInputInto(out, x, gy, alpha, inv_b, rows, cols);
    const input = contiguousDataConst(x, rows * cols);
    const grad = contiguousDataConst(gy, rows * cols);
    const output = contiguousData(out, rows * cols);
    if (maybeParallelSnakeBackwardInput(pc, output, input, grad, alpha, inv_b, rows, cols)) return;
    snakeBackwardInputRowsRange(output, input, grad, alpha, inv_b, cols, 0, rows);
}

fn maybeParallelSnakeBackwardInput(
    pc: ParallelConfig,
    out: []f32,
    x: []const f32,
    gy: []const f32,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
) bool {
    const pool = pc.pool orelse return false;
    const thread_count = @min(elementwiseThreadCount(out.len), rows);
    if (thread_count <= 1) return false;
    const Ctx = struct { out: []f32, x: []const f32, gy: []const f32, alpha: []const f32, inv_b: []const f32, cols: usize };
    tile.forRange(pool, Ctx, .{ .out = out, .x = x, .gy = gy, .alpha = alpha, .inv_b = inv_b, .cols = cols }, rows, thread_count, struct {
        fn go(c: Ctx, row_start: usize, row_end: usize) void {
            snakeBackwardInputRowsRange(c.out, c.x, c.gy, c.alpha, c.inv_b, c.cols, row_start, row_end);
        }
    }.go);
    return true;
}

fn snakeBackwardInputRowsRange(
    out: []f32,
    x: []const f32,
    gy: []const f32,
    alpha: []const f32,
    inv_b: []const f32,
    cols: usize,
    row_start: usize,
    row_end: usize,
) void {
    for (row_start..row_end) |r| {
        const x_row = x[r * cols ..][0..cols];
        const gy_row = gy[r * cols ..][0..cols];
        const out_row = out[r * cols ..][0..cols];
        var c: usize = 0;
        while (c + vector_len <= cols) : (c += vector_len) {
            const xv: Vf32 = x_row[c..][0..vector_len].*;
            const gv: Vf32 = gy_row[c..][0..vector_len].*;
            const av: Vf32 = alpha[c..][0..vector_len].*;
            const ibv: Vf32 = inv_b[c..][0..vector_len].*;
            const two: Vf32 = @splat(2.0);
            const one: Vf32 = @splat(1.0);
            const s2 = @sin(two * av * xv);
            out_row[c..][0..vector_len].* = gv * (one + ibv * av * s2);
        }
        while (c < cols) : (c += 1) {
            const s2 = @sin(2 * alpha[c] * x_row[c]);
            out_row[c] = gy_row[c] * (1 + inv_b[c] * alpha[c] * s2);
        }
    }
}

/// VJPs of snakeInto wrt the per-channel parameters, both filled in
/// one pass (they share the same traversal):
/// `galpha[c] = Σ_t gy[t,c]*inv_b[c]*x[t,c]*sin(2*alpha[c]*x[t,c])`,
/// `ginv_b[c] = Σ_t gy[t,c]*sin(alpha[c]*x[t,c])^2`. f32 accumulation, rows
/// visited in order per channel. Parallel over channel ranges — disjoint
/// channel writes ⇒ bit-identical to serial.
pub fn snakeBackwardParamsInto(
    pc: ParallelConfig,
    galpha: *Tensor,
    ginv_b: *Tensor,
    x: *const Tensor,
    gy: *const Tensor,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
) void {
    if (comptime isa.reference) return scalar.snakeBackwardParamsInto(galpha, ginv_b, x, gy, alpha, inv_b, rows, cols);
    const ga = contiguousData(galpha, cols);
    const gib = contiguousData(ginv_b, cols);
    const input = contiguousDataConst(x, rows * cols);
    const grad = contiguousDataConst(gy, rows * cols);
    if (maybeParallelSnakeBackwardParams(pc, ga, gib, input, grad, alpha, inv_b, rows, cols)) return;
    snakeBackwardParamsColumnRange(ga, gib, input, grad, alpha, inv_b, rows, cols, 0, cols);
}

fn maybeParallelSnakeBackwardParams(
    pc: ParallelConfig,
    ga: []f32,
    gib: []f32,
    x: []const f32,
    gy: []const f32,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
) bool {
    const pool = pc.pool orelse return false;
    const thread_count = @min(elementwiseThreadCount(x.len), cols);
    if (thread_count <= 1) return false;
    const Ctx = struct { ga: []f32, gib: []f32, x: []const f32, gy: []const f32, alpha: []const f32, inv_b: []const f32, rows: usize, cols: usize };
    tile.forRange(pool, Ctx, .{ .ga = ga, .gib = gib, .x = x, .gy = gy, .alpha = alpha, .inv_b = inv_b, .rows = rows, .cols = cols }, cols, thread_count, struct {
        fn go(c: Ctx, col_start: usize, col_end: usize) void {
            snakeBackwardParamsColumnRange(c.ga, c.gib, c.x, c.gy, c.alpha, c.inv_b, c.rows, c.cols, col_start, col_end);
        }
    }.go);
    return true;
}

fn snakeBackwardParamsColumnRange(
    ga: []f32,
    gib: []f32,
    x: []const f32,
    gy: []const f32,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
    col_start: usize,
    col_end: usize,
) void {
    // Channel-block vectorization: per Vf32 chunk of channels, accumulate over
    // rows in vector registers. Each channel's sum is still built by f32 adds
    // in ascending row order with the same op forms as the scalar body, so the
    // result is bit-identical.
    var c = col_start;
    while (c + vector_len <= col_end) : (c += vector_len) {
        const av: Vf32 = alpha[c..][0..vector_len].*;
        const ibv: Vf32 = inv_b[c..][0..vector_len].*;
        const two: Vf32 = @splat(2.0);
        var acc_ga: Vf32 = @splat(0.0);
        var acc_gib: Vf32 = @splat(0.0);
        for (0..rows) |r| {
            const xv: Vf32 = x[r * cols + c ..][0..vector_len].*;
            const gv: Vf32 = gy[r * cols + c ..][0..vector_len].*;
            const s = @sin(av * xv);
            const s2 = @sin(two * av * xv);
            acc_ga += gv * ibv * xv * s2;
            acc_gib += gv * s * s;
        }
        ga[c..][0..vector_len].* = acc_ga;
        gib[c..][0..vector_len].* = acc_gib;
    }
    while (c < col_end) : (c += 1) {
        var acc_ga: f32 = 0;
        var acc_gib: f32 = 0;
        for (0..rows) |r| {
            const v = x[r * cols + c];
            const gv = gy[r * cols + c];
            const s = @sin(alpha[c] * v);
            const s2 = @sin(2 * alpha[c] * v);
            acc_ga += gv * inv_b[c] * v * s2;
            acc_gib += gv * s * s;
        }
        ga[c] = acc_ga;
        gib[c] = acc_gib;
    }
}

/// VJP of groupNormInto. Recomputes the per-group mean and biased
/// variance from `x` with the SAME f64 two-pass accumulation as the forward
/// (mean/scale applied in f32, eps inside the sqrt), then fills any of:
///   gx[t,c] = (1/σ_g)·(ĝ[t,c] − mean_G(ĝ) − x̂[t,c]·mean_G(ĝ·x̂))
///             with ĝ = gy⊙weight (or gy when no affine), the two group means
///             accumulated in f64 over the group's rows×(C/G) elements and
///             applied in f32 (the forward's precision policy);
///   gw[c] = Σ_t gy[t,c]·x̂[t,c]   (f32 row-order accumulation per channel);
///   gb[c] = Σ_t gy[t,c].
/// Null outputs are skipped. Parallel over whole groups — each group owns a
/// disjoint column slice of every output, so threading is bit-identical to
/// serial.
pub fn groupNormBackwardInto(
    pc: ParallelConfig,
    gx: ?*Tensor,
    gw: ?*Tensor,
    gb: ?*Tensor,
    x: *const Tensor,
    gy: *const Tensor,
    weight: ?[]const f32,
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
) void {
    if (comptime isa.reference) return scalar.groupNormBackwardInto(gx, gw, gb, x, gy, weight, rows, cols, groups, eps);
    const input = contiguousDataConst(x, rows * cols);
    const grad = contiguousDataConst(gy, rows * cols);
    const gx_data: ?[]f32 = if (gx) |t| contiguousData(t, rows * cols) else null;
    const gw_data: ?[]f32 = if (gw) |t| contiguousData(t, cols) else null;
    const gb_data: ?[]f32 = if (gb) |t| contiguousData(t, cols) else null;
    if (maybeParallelGroupNormBackward(pc, gx_data, gw_data, gb_data, input, grad, weight, rows, cols, groups, eps)) return;
    groupNormBackwardGroupRange(gx_data, gw_data, gb_data, input, grad, weight, rows, cols, groups, eps, 0, groups);
}

fn maybeParallelGroupNormBackward(
    pc: ParallelConfig,
    gx: ?[]f32,
    gw: ?[]f32,
    gb: ?[]f32,
    x: []const f32,
    gy: []const f32,
    weight: ?[]const f32,
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
) bool {
    const pool = pc.pool orelse return false;
    const thread_count = @min(elementwiseThreadCount(x.len), groups);
    if (thread_count <= 1) return false;
    const Ctx = struct { gx: ?[]f32, gw: ?[]f32, gb: ?[]f32, x: []const f32, gy: []const f32, weight: ?[]const f32, rows: usize, cols: usize, groups: usize, eps: f32 };
    tile.forRange(pool, Ctx, .{ .gx = gx, .gw = gw, .gb = gb, .x = x, .gy = gy, .weight = weight, .rows = rows, .cols = cols, .groups = groups, .eps = eps }, groups, thread_count, struct {
        fn go(c: Ctx, group_start: usize, group_end: usize) void {
            groupNormBackwardGroupRange(c.gx, c.gw, c.gb, c.x, c.gy, c.weight, c.rows, c.cols, c.groups, c.eps, group_start, group_end);
        }
    }.go);
    return true;
}

fn groupNormBackwardGroupRange(
    gx: ?[]f32,
    gw: ?[]f32,
    gb: ?[]f32,
    x: []const f32,
    gy: []const f32,
    weight: ?[]const f32,
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
    group_start: usize,
    group_end: usize,
) void {
    const cols_per_group = cols / groups;
    const count: f64 = @floatFromInt(rows * cols_per_group);
    for (group_start..group_end) |g| {
        const col_start = g * cols_per_group;
        // Recompute mean/scale exactly like the forward (f64 two-pass, f32
        // mean/scale, eps inside the sqrt).
        var sum: f64 = 0;
        for (0..rows) |r| {
            const row = x[r * cols + col_start ..][0..cols_per_group];
            for (row) |v| sum += v;
        }
        const mean: f32 = @floatCast(sum / count);
        var sum2: f64 = 0;
        for (0..rows) |r| {
            const row = x[r * cols + col_start ..][0..cols_per_group];
            for (row) |v| {
                const centered = v - mean;
                sum2 += @as(f64, centered) * @as(f64, centered);
            }
        }
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatCast(sum2 / count + eps)));

        const mean_splat: Vf32 = @splat(mean);
        const scale_splat: Vf32 = @splat(scale);
        const full_c = cols_per_group - cols_per_group % vector_len;

        if (gw != null or gb != null) {
            // Vector body: rows outer, channel chunks inner, accumulating into
            // the zeroed dst slices. Each channel's sum is still built by f32
            // adds in ascending row order with the same op forms, so the result
            // is bit-identical to the scalar register accumulation.
            if (full_c > 0) {
                const gw_group: ?[]f32 = if (gw) |dst| dst[col_start..][0..full_c] else null;
                const gb_group: ?[]f32 = if (gb) |dst| dst[col_start..][0..full_c] else null;
                if (gw_group) |s| @memset(s, 0);
                if (gb_group) |s| @memset(s, 0);
                for (0..rows) |r| {
                    const x_row = x[r * cols + col_start ..][0..full_c];
                    const gy_row = gy[r * cols + col_start ..][0..full_c];
                    var local_c: usize = 0;
                    while (local_c + vector_len <= full_c) : (local_c += vector_len) {
                        const gv: Vf32 = gy_row[local_c..][0..vector_len].*;
                        if (gw_group) |s| {
                            const xv: Vf32 = x_row[local_c..][0..vector_len].*;
                            const acc: Vf32 = s[local_c..][0..vector_len].*;
                            s[local_c..][0..vector_len].* = acc + gv * (xv - mean_splat) * scale_splat;
                        }
                        if (gb_group) |s| {
                            const acc: Vf32 = s[local_c..][0..vector_len].*;
                            s[local_c..][0..vector_len].* = acc + gv;
                        }
                    }
                }
            }
            // Scalar tail channels: the original register-accumulator form.
            for (full_c..cols_per_group) |local_c| {
                const c = col_start + local_c;
                var acc_w: f32 = 0;
                var acc_b: f32 = 0;
                for (0..rows) |r| {
                    const v = x[r * cols + c];
                    const gv = gy[r * cols + c];
                    acc_w += gv * (v - mean) * scale;
                    acc_b += gv;
                }
                if (gw) |dst| dst[c] = acc_w;
                if (gb) |dst| dst[c] = acc_b;
            }
        }

        const dx = gx orelse continue;
        var sum_g: f64 = 0;
        var sum_gx: f64 = 0;
        for (0..rows) |r| {
            const x_row = x[r * cols + col_start ..][0..cols_per_group];
            const gy_row = gy[r * cols + col_start ..][0..cols_per_group];
            for (x_row, gy_row, 0..) |v, gv, local_c| {
                const wv: f32 = if (weight) |w| w[col_start + local_c] else 1.0;
                const gh = gv * wv;
                const xh = (v - mean) * scale;
                sum_g += gh;
                sum_gx += @as(f64, gh) * @as(f64, xh);
            }
        }
        const mean_g: f32 = @floatCast(sum_g / count);
        const mean_gx: f32 = @floatCast(sum_gx / count);
        // Elementwise combine given the precomputed group means: the vector
        // body mirrors the scalar op sequence per lane (bit-identical).
        const mean_g_splat: Vf32 = @splat(mean_g);
        const mean_gx_splat: Vf32 = @splat(mean_gx);
        for (0..rows) |r| {
            const x_row = x[r * cols + col_start ..][0..cols_per_group];
            const gy_row = gy[r * cols + col_start ..][0..cols_per_group];
            const dx_row = dx[r * cols + col_start ..][0..cols_per_group];
            var local_c: usize = 0;
            while (local_c + vector_len <= cols_per_group) : (local_c += vector_len) {
                const xv: Vf32 = x_row[local_c..][0..vector_len].*;
                const gv: Vf32 = gy_row[local_c..][0..vector_len].*;
                const wv: Vf32 = if (weight) |w| w[col_start + local_c ..][0..vector_len].* else @splat(1.0);
                const gh = gv * wv;
                const xh = (xv - mean_splat) * scale_splat;
                dx_row[local_c..][0..vector_len].* = scale_splat * (gh - mean_g_splat - xh * mean_gx_splat);
            }
            while (local_c < cols_per_group) : (local_c += 1) {
                const wv: f32 = if (weight) |w| w[col_start + local_c] else 1.0;
                const gh = gy_row[local_c] * wv;
                const xh = (x_row[local_c] - mean) * scale;
                dx_row[local_c] = scale * (gh - mean_g - xh * mean_gx);
            }
        }
    }
}

test {
    _ = @import("elementwise_tests.zig");
}

// ---------------- The scalar reference arms ----------------

/// The scalar reference twins of this file's kernel entries: plain serial
/// loops, no SIMD, no pool. On `-Dbackend=scalar` builds (`isa.reference`)
/// every entry above dispatches here, so this namespace is the executable
/// specification; on native builds the twins stay reachable for
/// `backend/parity_test.zig` and `bench/backend.zig`, which hold entry and
/// twin to the same answer.
pub const scalar = struct {
    pub fn castF32ToBf16(output: []u16, input: []const f32) void {
        for (output, input) |*dst, value| dst.* = dtype_mod.f32ToBf16(value);
    }

    pub fn castBf16ToF32(output: []f32, input: []const u16) void {
        for (output, input) |*dst, value| dst.* = dtype_mod.bf16ToF32(value);
    }

    pub fn castF32ToF16(output: []f16, input: []const f32) void {
        for (output, input) |*dst, value| dst.* = @floatCast(value);
    }

    pub fn castF16ToF32(output: []f32, input: []const f16) void {
        for (output, input) |*dst, value| dst.* = @floatCast(value);
    }
    pub fn addContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
        const x = contiguousDataConst(a, len);
        const y = contiguousDataConst(b, len);
        const z = contiguousData(out, len);
        for (z, x, y) |*dst, xv, yv| dst.* = xv + yv;
    }

    pub fn subContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
        const x = contiguousDataConst(a, len);
        const y = contiguousDataConst(b, len);
        const z = contiguousData(out, len);
        for (z, x, y) |*dst, xv, yv| dst.* = xv - yv;
    }

    pub fn mulContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
        const x = contiguousDataConst(a, len);
        const y = contiguousDataConst(b, len);
        const z = contiguousData(out, len);
        for (z, x, y) |*dst, xv, yv| dst.* = xv * yv;
    }

    pub fn divContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
        const x = contiguousDataConst(a, len);
        const y = contiguousDataConst(b, len);
        const z = contiguousData(out, len);
        for (z, x, y) |*dst, xv, yv| dst.* = xv / yv;
    }

    pub fn maximumContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
        const x = contiguousDataConst(a, len);
        const y = contiguousDataConst(b, len);
        const z = contiguousData(out, len);
        for (z, x, y) |*dst, xv, yv| dst.* = if (xv != xv or yv != yv) std.math.nan(f32) else @max(xv, yv);
    }

    pub fn minimumContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
        const x = contiguousDataConst(a, len);
        const y = contiguousDataConst(b, len);
        const z = contiguousData(out, len);
        for (z, x, y) |*dst, xv, yv| dst.* = if (xv != xv or yv != yv) std.math.nan(f32) else @min(xv, yv);
    }

    pub fn elementwiseContiguousIntoTyped(
        comptime dtype: DType,
        comptime op: ops.ElementwiseOp,
        out: *tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)),
        a: *const tensor.TensorOf(dtype),
        b: *const tensor.TensorOf(dtype),
        len: usize,
    ) void {
        const x = common.contiguousDataConstOf(dtype, a, len);
        const y = common.contiguousDataConstOf(dtype, b, len);
        const z = common.contiguousDataOf(dtype_mod.outputDType(.pointwise, dtype), out, len);
        elementwiseSlicesTyped(dtype, op, z, x, y);
    }

    pub fn scaleInto(out: *Tensor, a: *const Tensor, scalar_value: f32) !void {
        try tensor.requireSameShape(out, a);
        const x = a.dataConst();
        const z = out.data();
        for (z, x) |*dst, xv| dst.* = xv * scalar_value;
    }

    pub fn addScaledSlice(z: []f32, x: []const f32, scalar_value: f32) void {
        for (z, x) |*dst, xv| dst.* += xv * scalar_value;
    }

    pub fn addRowVectorSlice(comptime op: ?ops.UnaryOp, z: []f32, row_vector: []const f32, rows: usize, cols: usize) void {
        std.debug.assert(z.len >= rows * cols);
        std.debug.assert(row_vector.len == cols);
        for (0..rows) |row_i| {
            const row = z[row_i * cols ..][0..cols];
            if (comptime op) |actual_op| {
                for (row, row_vector) |*dst, value| dst.* = ops.unaryScalar(actual_op, dst.* + value);
            } else {
                for (row, row_vector) |*dst, value| dst.* += value;
            }
        }
    }

    pub fn unaryRowSlice(comptime op: ops.UnaryOp, z: []f32, x: []const f32) void {
        for (z, x) |*dst, value| dst.* = ops.unaryScalar(op, value);
    }

    pub fn mulRowSlice(z: []f32, x: []const f32, y: []const f32) void {
        for (z, x, y) |*dst, a, b| dst.* = a * b;
    }

    pub fn preluChannelsInto(z: []f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize) void {
        for (0..rows) |r| {
            for (0..cols) |c| {
                const i = r * cols + c;
                z[i] = if (x[i] > 0) x[i] else x[i] * alpha[c];
            }
        }
    }

    pub fn channelAffineInto(z: []f32, x: []const f32, scale: []const f32, shift: ?[]const f32, rows: usize, cols: usize) void {
        for (0..rows) |r| {
            for (0..cols) |c| {
                const i = r * cols + c;
                z[i] = if (shift) |t| x[i] * scale[c] + t[c] else x[i] * scale[c];
            }
        }
    }

    pub fn preluChannelsBackwardInputInto(gx: []f32, gy: []const f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize) void {
        for (0..rows) |r| {
            for (0..cols) |c| {
                const i = r * cols + c;
                gx[i] = if (x[i] > 0) gy[i] else gy[i] * alpha[c];
            }
        }
    }

    pub fn preluChannelsBackwardAlphaInto(galpha: []f32, gy: []const f32, x: []const f32, rows: usize, cols: usize) void {
        @memset(galpha, 0);
        for (0..rows) |r| {
            for (0..cols) |c| {
                const i = r * cols + c;
                if (x[i] <= 0) galpha[c] += gy[i] * x[i];
            }
        }
    }

    pub fn unaryContiguousIntoUnchecked(comptime op: ops.UnaryOp, out: *Tensor, a: *const Tensor, len: usize) void {
        const x = contiguousDataConst(a, len);
        const z = contiguousData(out, len);
        for (z, x) |*dst, value| dst.* = ops.unaryScalar(op, value);
    }

    pub fn leakyReluContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, len: usize, negative_slope: f32) void {
        const x = contiguousDataConst(a, len);
        const z = contiguousData(out, len);
        for (z, x) |*dst, value| dst.* = if (value >= 0) value else value * negative_slope;
    }

    /// `cap * tanh(x / cap)`: the logit softcap.
    pub fn softcapContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, len: usize, cap: f32) void {
        const x = contiguousDataConst(a, len);
        const z = contiguousData(out, len);
        const inv = 1.0 / cap;
        for (z, x) |*dst, value| dst.* = cap * std.math.tanh(value * inv);
    }

    pub fn clampContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, len: usize, min_value: f32, max_value: f32) void {
        const x = contiguousDataConst(a, len);
        const z = contiguousData(out, len);
        for (z, x) |*dst, value| dst.* = @min(@max(value, min_value), max_value);
    }

    pub fn gatedContiguousIntoUnchecked(comptime op: ops.GatedOp, out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
        const x = contiguousDataConst(a, len);
        const y = contiguousDataConst(b, len);
        const z = contiguousData(out, len);
        for (z, x, y) |*dst, left, gate| dst.* = ops.gatedPairScalar(op, gate, left);
    }

    pub fn sumInto(out: *Tensor, a: *const Tensor) !void {
        if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
        out.data()[0] = scalar.sumSlice(a.dataConst());
    }

    pub fn sumSlice(values: []const f32) f32 {
        var acc: f32 = 0;
        for (values) |v| acc += v;
        return acc;
    }

    pub fn prodInto(out: *Tensor, a: *const Tensor) !void {
        if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
        out.data()[0] = scalar.prodSlice(a.dataConst());
    }

    pub fn prodSlice(values: []const f32) f32 {
        var acc: f32 = 1;
        for (values) |v| acc *= v;
        return acc;
    }

    /// Full dot product into the scalar `out`, accumulated serially in the
    /// dtype's matmul compute dtype (f32 included; the reference arm of both
    /// `dotInto` and `dotIntoTyped`).
    pub fn dot(
        comptime dtype: DType,
        out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
        a: *const tensor.TensorOf(dtype),
        b: *const tensor.TensorOf(dtype),
    ) !void {
        try tensor.requireSameShape(a, b);
        if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
        out.data()[0] = dotSliceTypedScalar(dtype, a.dataConst(), b.dataConst());
    }

    pub fn snakeInto(out: *Tensor, x: *const Tensor, alpha: []const f32, inv_b: []const f32, rows: usize, cols: usize) void {
        const input = contiguousDataConst(x, rows * cols);
        const output = contiguousData(out, rows * cols);
        for (0..rows) |r| {
            for (0..cols) |c| {
                const v = input[r * cols + c];
                const sn = @sin(alpha[c] * v);
                output[r * cols + c] = v + inv_b[c] * sn * sn;
            }
        }
    }

    pub fn groupNormInto(
        out: *Tensor,
        x: *const Tensor,
        weight: ?[]const f32,
        bias: ?[]const f32,
        rows: usize,
        cols: usize,
        groups: usize,
        eps: f32,
    ) void {
        const input = contiguousDataConst(x, rows * cols);
        const output = contiguousData(out, rows * cols);
        const cols_per_group = cols / groups;
        const count: f64 = @floatFromInt(rows * cols_per_group);
        for (0..groups) |g| {
            const col_start = g * cols_per_group;
            var sum: f64 = 0;
            for (0..rows) |r| {
                for (0..cols_per_group) |local_c| {
                    sum += input[r * cols + col_start + local_c];
                }
            }
            const mean: f32 = @floatCast(sum / count);
            var sum2: f64 = 0;
            for (0..rows) |r| {
                for (0..cols_per_group) |local_c| {
                    const centered = input[r * cols + col_start + local_c] - mean;
                    sum2 += @as(f64, centered) * @as(f64, centered);
                }
            }
            const scale_v: f32 = 1.0 / @sqrt(@as(f32, @floatCast(sum2 / count + eps)));
            for (0..rows) |r| {
                for (0..cols_per_group) |local_c| {
                    const c = col_start + local_c;
                    var value = (input[r * cols + c] - mean) * scale_v;
                    if (weight) |w| value *= w[c];
                    if (bias) |b| value += b[c];
                    output[r * cols + c] = value;
                }
            }
        }
    }

    pub fn snakeBackwardInputInto(out: *Tensor, x: *const Tensor, gy: *const Tensor, alpha: []const f32, inv_b: []const f32, rows: usize, cols: usize) void {
        const input = contiguousDataConst(x, rows * cols);
        const grad = contiguousDataConst(gy, rows * cols);
        const output = contiguousData(out, rows * cols);
        for (0..rows) |r| {
            for (0..cols) |c| {
                const v = input[r * cols + c];
                const s2 = @sin(2 * alpha[c] * v);
                output[r * cols + c] = grad[r * cols + c] * (1 + inv_b[c] * alpha[c] * s2);
            }
        }
    }

    pub fn snakeBackwardParamsInto(
        galpha: *Tensor,
        ginv_b: *Tensor,
        x: *const Tensor,
        gy: *const Tensor,
        alpha: []const f32,
        inv_b: []const f32,
        rows: usize,
        cols: usize,
    ) void {
        const ga = contiguousData(galpha, cols);
        const gib = contiguousData(ginv_b, cols);
        const input = contiguousDataConst(x, rows * cols);
        const grad = contiguousDataConst(gy, rows * cols);
        for (0..cols) |c| {
            ga[c] = 0;
            gib[c] = 0;
        }
        for (0..rows) |r| {
            for (0..cols) |c| {
                const v = input[r * cols + c];
                const gv = grad[r * cols + c];
                const sn = @sin(alpha[c] * v);
                const s2 = @sin(2 * alpha[c] * v);
                ga[c] += gv * inv_b[c] * v * s2;
                gib[c] += gv * sn * sn;
            }
        }
    }

    pub fn groupNormBackwardInto(
        gx: ?*Tensor,
        gw: ?*Tensor,
        gb: ?*Tensor,
        x: *const Tensor,
        gy: *const Tensor,
        weight: ?[]const f32,
        rows: usize,
        cols: usize,
        groups: usize,
        eps: f32,
    ) void {
        const input = contiguousDataConst(x, rows * cols);
        const grad = contiguousDataConst(gy, rows * cols);
        const gx_data: ?[]f32 = if (gx) |t| contiguousData(t, rows * cols) else null;
        const gw_data: ?[]f32 = if (gw) |t| contiguousData(t, cols) else null;
        const gb_data: ?[]f32 = if (gb) |t| contiguousData(t, cols) else null;
        const cols_per_group = cols / groups;
        const count: f64 = @floatFromInt(rows * cols_per_group);
        for (0..groups) |g| {
            const col_start = g * cols_per_group;
            var sum: f64 = 0;
            for (0..rows) |r| {
                for (0..cols_per_group) |local_c| sum += input[r * cols + col_start + local_c];
            }
            const mean: f32 = @floatCast(sum / count);
            var sum2: f64 = 0;
            for (0..rows) |r| {
                for (0..cols_per_group) |local_c| {
                    const centered = input[r * cols + col_start + local_c] - mean;
                    sum2 += @as(f64, centered) * @as(f64, centered);
                }
            }
            const scale_v: f32 = 1.0 / @sqrt(@as(f32, @floatCast(sum2 / count + eps)));

            if (gw_data != null or gb_data != null) {
                for (0..cols_per_group) |local_c| {
                    const c = col_start + local_c;
                    var acc_w: f32 = 0;
                    var acc_b: f32 = 0;
                    for (0..rows) |r| {
                        const v = input[r * cols + c];
                        const gv = grad[r * cols + c];
                        acc_w += gv * (v - mean) * scale_v;
                        acc_b += gv;
                    }
                    if (gw_data) |dst| dst[c] = acc_w;
                    if (gb_data) |dst| dst[c] = acc_b;
                }
            }

            const dx = gx_data orelse continue;
            var sum_g: f64 = 0;
            var sum_gx: f64 = 0;
            for (0..rows) |r| {
                for (0..cols_per_group) |local_c| {
                    const c = col_start + local_c;
                    const wv: f32 = if (weight) |w| w[c] else 1.0;
                    const gh = grad[r * cols + c] * wv;
                    const xh = (input[r * cols + c] - mean) * scale_v;
                    sum_g += gh;
                    sum_gx += @as(f64, gh) * @as(f64, xh);
                }
            }
            const mean_g: f32 = @floatCast(sum_g / count);
            const mean_gx: f32 = @floatCast(sum_gx / count);
            for (0..rows) |r| {
                for (0..cols_per_group) |local_c| {
                    const c = col_start + local_c;
                    const wv: f32 = if (weight) |w| w[c] else 1.0;
                    const gh = grad[r * cols + c] * wv;
                    const xh = (input[r * cols + c] - mean) * scale_v;
                    dx[r * cols + c] = scale_v * (gh - mean_g - xh * mean_gx);
                }
            }
        }
    }
};
