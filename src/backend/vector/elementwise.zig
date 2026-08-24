//! Elementwise vector kernels: the contiguous entry points, the
//! elementwise/reduction Task structs and their parallel dispatch, and the
//! typed scalar inner kernels. Shared-core symbols (ParallelConfig,
//! contiguous-data helpers, elementwiseThreadCount) come from `common.zig`;
//! the @Vector primitives from `primitives.zig`; op definitions from
//! `../ops.zig`.

const std = @import("std");
const ops = @import("../ops.zig");
const dtype_mod = @import("../../dtype.zig");
const parallel = @import("../../parallel.zig");
const tensor = @import("../../tensor.zig");
const common = @import("common.zig");
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
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, runAddTask, z, x, y)) return;
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
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, runSubTask, z, x, y)) return;
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
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, runMulTask, z, x, y)) return;
    primitives.vecMul(z, x, y);
}

pub fn divContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, runDivTask, z, x, y)) return;
    primitives.vecDiv(z, x, y);
}

pub fn maximumContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, runMaximumTask, z, x, y)) return;
    primitives.vecMaximum(z, x, y);
}

pub fn minimumContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelBinary(pc, runMinimumTask, z, x, y)) return;
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
    try tensor.requireSameShape(out, a);
    const x = a.dataConst();
    const z = out.data();
    if (maybeParallelScale(pc, z, x, scalar_value)) return;
    primitives.vecScale(z, x, scalar_value);
}

pub fn addScaledSlice(z: []f32, x: []const f32, scalar_value: f32) void {
    primitives.vecAddScaled(z, x, scalar_value);
}

/// `z[row] += row_vector` for every row, then `op` (a fused bias + activation
/// when given).
pub fn addRowVectorSlice(comptime op: ?ops.UnaryOp, z: []f32, row_vector: []const f32, rows: usize, cols: usize) void {
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

const RowChanTask = struct {
    z: []f32,
    x: []const f32,
    a: []const f32,
    b: ?[]const f32,
    cols: usize,
    row_start: usize,
    row_end: usize,
};

fn maybeParallelRowChan(
    pc: ParallelConfig,
    comptime func: fn (*const RowChanTask) void,
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

    var tasks: [parallel.vector_max_threads]RowChanTask = undefined;
    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .z = z,
            .x = x,
            .a = a,
            .b = b,
            .cols = cols,
            .row_start = ti * rows / thread_count,
            .row_end = (ti + 1) * rows / thread_count,
        };
    }
    pool.parallelChunks(RowChanTask, tasks[0..thread_count], func);
    return true;
}

fn runPreluChannelsTask(task: *const RowChanTask) void {
    preluChannelsRows(task.z, task.x, task.a, task.cols, task.row_start, task.row_end);
}

/// PReLU with a per-channel slope: `z[r,c] = x > 0 ? x : α[c]·x`.
pub fn preluChannelsInto(pc: ParallelConfig, z: []f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(z.len >= rows * cols and x.len >= rows * cols);
    std.debug.assert(alpha.len == cols);
    if (maybeParallelRowChan(pc, runPreluChannelsTask, z, x, alpha, null, rows, cols)) return;
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

fn runChannelAffineTask(task: *const RowChanTask) void {
    channelAffineRows(task.z, task.x, task.a, task.b, task.cols, task.row_start, task.row_end);
}

/// Per-channel affine (frozen-stats BatchNorm): `z[r,c] = x·scale[c] + shift[c]`;
/// a null `shift` degrades to the per-channel scale `z = x·scale[c]` (the
/// affine's own input-VJP).
pub fn channelAffineInto(pc: ParallelConfig, z: []f32, x: []const f32, scale: []const f32, shift: ?[]const f32, rows: usize, cols: usize) void {
    std.debug.assert(z.len >= rows * cols and x.len >= rows * cols);
    std.debug.assert(scale.len == cols and (shift == null or shift.?.len == cols));
    if (maybeParallelRowChan(pc, runChannelAffineTask, z, x, scale, shift, rows, cols)) return;
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
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    if (maybeParallelLeakyRelu(pc, z, x, negative_slope)) return;
    primitives.vecLeakyRelu(z, x, negative_slope);
}

pub fn clampContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    min_value: f32,
    max_value: f32,
) void {
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
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    if (maybeParallelGated(pc, op, z, x, y)) return;
    primitives.vecGated(op, z, x, y);
}

pub fn sumInto(pc: ParallelConfig, out: *Tensor, a: *const Tensor) !void {
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    out.data()[0] = parallelVecSum(pc, a.dataConst()) orelse primitives.vecSum(a.dataConst());
}

pub fn sumSlice(values: []const f32) f32 {
    return primitives.vecSum(values);
}

pub fn prodInto(pc: ParallelConfig, out: *Tensor, a: *const Tensor) !void {
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    out.data()[0] = parallelVecProd(pc, a.dataConst()) orelse primitives.vecProd(a.dataConst());
}

pub fn prodSlice(values: []const f32) f32 {
    return primitives.vecProd(values);
}

pub fn sumSliceTyped(
    pc: ParallelConfig,
    comptime dtype: DType,
    values: []const dtype_mod.Scalar(dtype),
) dtype_mod.Scalar(dtype_mod.outputDType(.reduction, dtype)) {
    _ = pc;
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
    try tensor.requireSameShapeOf(dtype, a, b);
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

const BinaryTask = struct {
    z: []f32,
    x: []const f32,
    y: []const f32,
    start: usize,
    end: usize,
};

const ScaleTask = struct {
    z: []f32,
    x: []const f32,
    scalar: f32,
    start: usize,
    end: usize,
};

const UnaryTask = struct {
    op: ops.UnaryOp,
    z: []f32,
    x: []const f32,
    start: usize,
    end: usize,
};

const LeakyReluTask = struct {
    z: []f32,
    x: []const f32,
    negative_slope: f32,
    start: usize,
    end: usize,
};

const ClampTask = struct {
    z: []f32,
    x: []const f32,
    min_value: f32,
    max_value: f32,
    start: usize,
    end: usize,
};

const GatedTask = struct {
    op: ops.GatedOp,
    z: []f32,
    x: []const f32,
    y: []const f32,
    start: usize,
    end: usize,
};

const SumTask = struct {
    x: []const f32,
    partial: *f32,
    start: usize,
    end: usize,
};

const ProdTask = struct {
    x: []const f32,
    partial: *f32,
    start: usize,
    end: usize,
};

const DotTask = struct {
    x: []const f32,
    y: []const f32,
    partial: *f32,
    start: usize,
    end: usize,
};

const DotTaskF64 = struct {
    x: []const f64,
    y: []const f64,
    partial: *f64,
    start: usize,
    end: usize,
};

const DotTaskF16 = struct {
    x: []const f16,
    y: []const f16,
    partial: *f32,
    start: usize,
    end: usize,
};

const DotTaskBf16 = struct {
    x: []const u16,
    y: []const u16,
    partial: *f32,
    start: usize,
    end: usize,
};

fn maybeParallelBinary(
    pc: ParallelConfig,
    comptime func: fn (*const BinaryTask) void,
    z: []f32,
    x: []const f32,
    y: []const f32,
) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]BinaryTask = undefined;
    for (0..thread_count) |ti| {
        const start = ti * z.len / thread_count;
        const end = (ti + 1) * z.len / thread_count;
        tasks[ti] = .{ .z = z, .x = x, .y = y, .start = start, .end = end };
    }
    pool.parallelChunks(BinaryTask, tasks[0..thread_count], func);
    return true;
}

fn maybeParallelScale(pc: ParallelConfig, z: []f32, x: []const f32, scalar: f32) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]ScaleTask = undefined;
    for (0..thread_count) |ti| {
        const start = ti * z.len / thread_count;
        const end = (ti + 1) * z.len / thread_count;
        tasks[ti] = .{ .z = z, .x = x, .scalar = scalar, .start = start, .end = end };
    }
    pool.parallelChunks(ScaleTask, tasks[0..thread_count], runScaleTask);
    return true;
}

fn maybeParallelUnary(pc: ParallelConfig, comptime op: ops.UnaryOp, z: []f32, x: []const f32) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]UnaryTask = undefined;
    for (0..thread_count) |ti| {
        const start = ti * z.len / thread_count;
        const end = (ti + 1) * z.len / thread_count;
        tasks[ti] = .{ .op = op, .z = z, .x = x, .start = start, .end = end };
    }
    pool.parallelChunks(UnaryTask, tasks[0..thread_count], runUnaryTask);
    return true;
}

fn maybeParallelLeakyRelu(pc: ParallelConfig, z: []f32, x: []const f32, negative_slope: f32) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]LeakyReluTask = undefined;
    for (0..thread_count) |ti| {
        const start = ti * z.len / thread_count;
        const end = (ti + 1) * z.len / thread_count;
        tasks[ti] = .{ .z = z, .x = x, .negative_slope = negative_slope, .start = start, .end = end };
    }
    pool.parallelChunks(LeakyReluTask, tasks[0..thread_count], runLeakyReluTask);
    return true;
}

fn maybeParallelClamp(pc: ParallelConfig, z: []f32, x: []const f32, min_value: f32, max_value: f32) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]ClampTask = undefined;
    for (0..thread_count) |ti| {
        const start = ti * z.len / thread_count;
        const end = (ti + 1) * z.len / thread_count;
        tasks[ti] = .{ .z = z, .x = x, .min_value = min_value, .max_value = max_value, .start = start, .end = end };
    }
    pool.parallelChunks(ClampTask, tasks[0..thread_count], runClampTask);
    return true;
}

fn maybeParallelGated(pc: ParallelConfig, comptime op: ops.GatedOp, z: []f32, x: []const f32, y: []const f32) bool {
    const pool = pc.pool orelse return false;
    const thread_count = elementwiseThreadCount(z.len);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]GatedTask = undefined;
    for (0..thread_count) |ti| {
        const start = ti * z.len / thread_count;
        const end = (ti + 1) * z.len / thread_count;
        tasks[ti] = .{ .op = op, .z = z, .x = x, .y = y, .start = start, .end = end };
    }
    pool.parallelChunks(GatedTask, tasks[0..thread_count], runGatedTask);
    return true;
}

fn parallelVecSum(pc: ParallelConfig, x: []const f32) ?f32 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;

    var partials: [parallel.vector_max_threads]f32 = [_]f32{0} ** parallel.vector_max_threads;
    var tasks: [parallel.vector_max_threads]SumTask = undefined;
    for (0..thread_count) |ti| {
        const start = ti * x.len / thread_count;
        const end = (ti + 1) * x.len / thread_count;
        tasks[ti] = .{ .x = x, .partial = &partials[ti], .start = start, .end = end };
    }
    pool.parallelChunks(SumTask, tasks[0..thread_count], runSumTask);

    var total: f32 = 0;
    for (partials[0..thread_count]) |value| total += value;
    return total;
}

fn parallelVecProd(pc: ParallelConfig, x: []const f32) ?f32 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;

    var partials: [parallel.vector_max_threads]f32 = [_]f32{1} ** parallel.vector_max_threads;
    var tasks: [parallel.vector_max_threads]ProdTask = undefined;
    for (0..thread_count) |ti| {
        const start = ti * x.len / thread_count;
        const end = (ti + 1) * x.len / thread_count;
        tasks[ti] = .{ .x = x, .partial = &partials[ti], .start = start, .end = end };
    }
    pool.parallelChunks(ProdTask, tasks[0..thread_count], runProdTask);

    var total: f32 = 1;
    for (partials[0..thread_count]) |value| total *= value;
    return total;
}

fn parallelVecDot(pc: ParallelConfig, x: []const f32, y: []const f32) ?f32 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;

    var partials: [parallel.vector_max_threads]f32 = [_]f32{0} ** parallel.vector_max_threads;
    var tasks: [parallel.vector_max_threads]DotTask = undefined;
    for (0..thread_count) |ti| {
        const start = ti * x.len / thread_count;
        const end = (ti + 1) * x.len / thread_count;
        tasks[ti] = .{ .x = x, .y = y, .partial = &partials[ti], .start = start, .end = end };
    }
    pool.parallelChunks(DotTask, tasks[0..thread_count], runDotTask);

    var total: f32 = 0;
    for (partials[0..thread_count]) |value| total += value;
    return total;
}

fn parallelVecDotF64(pc: ParallelConfig, x: []const f64, y: []const f64) ?f64 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;

    var partials: [parallel.vector_max_threads]f64 = [_]f64{0} ** parallel.vector_max_threads;
    var tasks: [parallel.vector_max_threads]DotTaskF64 = undefined;
    for (0..thread_count) |ti| {
        const start = ti * x.len / thread_count;
        const end = (ti + 1) * x.len / thread_count;
        tasks[ti] = .{ .x = x, .y = y, .partial = &partials[ti], .start = start, .end = end };
    }
    pool.parallelChunks(DotTaskF64, tasks[0..thread_count], runDotF64Task);

    var total: f64 = 0;
    for (partials[0..thread_count]) |value| total += value;
    return total;
}

fn parallelVecDotF16ToF32(pc: ParallelConfig, x: []const f16, y: []const f16) ?f32 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;

    var partials: [parallel.vector_max_threads]f32 = [_]f32{0} ** parallel.vector_max_threads;
    var tasks: [parallel.vector_max_threads]DotTaskF16 = undefined;
    for (0..thread_count) |ti| {
        const start = ti * x.len / thread_count;
        const end = (ti + 1) * x.len / thread_count;
        tasks[ti] = .{ .x = x, .y = y, .partial = &partials[ti], .start = start, .end = end };
    }
    pool.parallelChunks(DotTaskF16, tasks[0..thread_count], runDotF16Task);

    var total: f32 = 0;
    for (partials[0..thread_count]) |value| total += value;
    return total;
}

fn parallelVecDotBf16ToF32(pc: ParallelConfig, x: []const u16, y: []const u16) ?f32 {
    const pool = pc.pool orelse return null;
    const thread_count = elementwiseThreadCount(x.len);
    if (thread_count == 1) return null;

    var partials: [parallel.vector_max_threads]f32 = [_]f32{0} ** parallel.vector_max_threads;
    var tasks: [parallel.vector_max_threads]DotTaskBf16 = undefined;
    for (0..thread_count) |ti| {
        const start = ti * x.len / thread_count;
        const end = (ti + 1) * x.len / thread_count;
        tasks[ti] = .{ .x = x, .y = y, .partial = &partials[ti], .start = start, .end = end };
    }
    pool.parallelChunks(DotTaskBf16, tasks[0..thread_count], runDotBf16Task);

    var total: f32 = 0;
    for (partials[0..thread_count]) |value| total += value;
    return total;
}

fn runAddTask(task: *const BinaryTask) void {
    primitives.vecAdd(task.z[task.start..task.end], task.x[task.start..task.end], task.y[task.start..task.end]);
}

fn runSubTask(task: *const BinaryTask) void {
    primitives.vecSub(task.z[task.start..task.end], task.x[task.start..task.end], task.y[task.start..task.end]);
}

fn runMulTask(task: *const BinaryTask) void {
    primitives.vecMul(task.z[task.start..task.end], task.x[task.start..task.end], task.y[task.start..task.end]);
}

fn runDivTask(task: *const BinaryTask) void {
    primitives.vecDiv(task.z[task.start..task.end], task.x[task.start..task.end], task.y[task.start..task.end]);
}

fn runMaximumTask(task: *const BinaryTask) void {
    primitives.vecMaximum(task.z[task.start..task.end], task.x[task.start..task.end], task.y[task.start..task.end]);
}

fn runMinimumTask(task: *const BinaryTask) void {
    primitives.vecMinimum(task.z[task.start..task.end], task.x[task.start..task.end], task.y[task.start..task.end]);
}

fn runScaleTask(task: *const ScaleTask) void {
    primitives.vecScale(task.z[task.start..task.end], task.x[task.start..task.end], task.scalar);
}

fn runUnaryTask(task: *const UnaryTask) void {
    switch (task.op) {
        inline else => |op| primitives.vecUnary(op, task.z[task.start..task.end], task.x[task.start..task.end]),
    }
}

fn runLeakyReluTask(task: *const LeakyReluTask) void {
    primitives.vecLeakyRelu(task.z[task.start..task.end], task.x[task.start..task.end], task.negative_slope);
}

fn runClampTask(task: *const ClampTask) void {
    primitives.vecClamp(task.z[task.start..task.end], task.x[task.start..task.end], task.min_value, task.max_value);
}

fn runGatedTask(task: *const GatedTask) void {
    switch (task.op) {
        inline else => |op| primitives.vecGated(op, task.z[task.start..task.end], task.x[task.start..task.end], task.y[task.start..task.end]),
    }
}

fn runSumTask(task: *const SumTask) void {
    task.partial.* = primitives.vecSum(task.x[task.start..task.end]);
}

fn runProdTask(task: *const ProdTask) void {
    task.partial.* = primitives.vecProd(task.x[task.start..task.end]);
}

fn runDotTask(task: *const DotTask) void {
    task.partial.* = primitives.vecDot(task.x[task.start..task.end], task.y[task.start..task.end]);
}

fn runDotF64Task(task: *const DotTaskF64) void {
    task.partial.* = primitives.vecDotF64(task.x[task.start..task.end], task.y[task.start..task.end]);
}

fn runDotF16Task(task: *const DotTaskF16) void {
    task.partial.* = primitives.vecDotF16ToF32(task.x[task.start..task.end], task.y[task.start..task.end]);
}

fn runDotBf16Task(task: *const DotTaskBf16) void {
    task.partial.* = primitives.vecDotBf16ToF32(task.x[task.start..task.end], task.y[task.start..task.end]);
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
    const input = contiguousDataConst(x, rows * cols);
    const output = contiguousData(out, rows * cols);
    if (maybeParallelSnake(pc, output, input, alpha, inv_b, rows, cols)) return;
    snakeRowsRange(output, input, alpha, inv_b, cols, 0, rows);
}

const SnakeTask = struct {
    out: []f32,
    x: []const f32,
    alpha: []const f32,
    inv_b: []const f32,
    cols: usize,
    row_start: usize,
    row_end: usize,
};

fn runSnakeTask(task: *const SnakeTask) void {
    snakeRowsRange(task.out, task.x, task.alpha, task.inv_b, task.cols, task.row_start, task.row_end);
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

    var tasks: [parallel.vector_max_threads]SnakeTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .out = out,
            .x = x,
            .alpha = alpha,
            .inv_b = inv_b,
            .cols = cols,
            .row_start = task_i * rows / thread_count,
            .row_end = (task_i + 1) * rows / thread_count,
        };
    }
    pool.parallelChunks(SnakeTask, tasks[0..thread_count], runSnakeTask);
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
    const input = contiguousDataConst(x, rows * cols);
    const output = contiguousData(out, rows * cols);
    if (maybeParallelGroupNorm(pc, output, input, weight, bias, rows, cols, groups, eps)) return;
    groupNormGroupRange(output, input, weight, bias, rows, cols, groups, eps, 0, groups);
}

const GroupNormTask = struct {
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
};

fn runGroupNormTask(task: *const GroupNormTask) void {
    groupNormGroupRange(task.out, task.x, task.weight, task.bias, task.rows, task.cols, task.groups, task.eps, task.group_start, task.group_end);
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

    var tasks: [parallel.vector_max_threads]GroupNormTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .out = out,
            .x = x,
            .weight = weight,
            .bias = bias,
            .rows = rows,
            .cols = cols,
            .groups = groups,
            .eps = eps,
            .group_start = task_i * groups / thread_count,
            .group_end = (task_i + 1) * groups / thread_count,
        };
    }
    pool.parallelChunks(GroupNormTask, tasks[0..thread_count], runGroupNormTask);
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
    const input = contiguousDataConst(x, rows * cols);
    const grad = contiguousDataConst(gy, rows * cols);
    const output = contiguousData(out, rows * cols);
    if (maybeParallelSnakeBackwardInput(pc, output, input, grad, alpha, inv_b, rows, cols)) return;
    snakeBackwardInputRowsRange(output, input, grad, alpha, inv_b, cols, 0, rows);
}

const SnakeBackwardInputTask = struct {
    out: []f32,
    x: []const f32,
    gy: []const f32,
    alpha: []const f32,
    inv_b: []const f32,
    cols: usize,
    row_start: usize,
    row_end: usize,
};

fn runSnakeBackwardInputTask(task: *const SnakeBackwardInputTask) void {
    snakeBackwardInputRowsRange(task.out, task.x, task.gy, task.alpha, task.inv_b, task.cols, task.row_start, task.row_end);
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

    var tasks: [parallel.vector_max_threads]SnakeBackwardInputTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .out = out,
            .x = x,
            .gy = gy,
            .alpha = alpha,
            .inv_b = inv_b,
            .cols = cols,
            .row_start = task_i * rows / thread_count,
            .row_end = (task_i + 1) * rows / thread_count,
        };
    }
    pool.parallelChunks(SnakeBackwardInputTask, tasks[0..thread_count], runSnakeBackwardInputTask);
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
    const ga = contiguousData(galpha, cols);
    const gib = contiguousData(ginv_b, cols);
    const input = contiguousDataConst(x, rows * cols);
    const grad = contiguousDataConst(gy, rows * cols);
    if (maybeParallelSnakeBackwardParams(pc, ga, gib, input, grad, alpha, inv_b, rows, cols)) return;
    snakeBackwardParamsColumnRange(ga, gib, input, grad, alpha, inv_b, rows, cols, 0, cols);
}

const SnakeBackwardParamsTask = struct {
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
};

fn runSnakeBackwardParamsTask(task: *const SnakeBackwardParamsTask) void {
    snakeBackwardParamsColumnRange(task.ga, task.gib, task.x, task.gy, task.alpha, task.inv_b, task.rows, task.cols, task.col_start, task.col_end);
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

    var tasks: [parallel.vector_max_threads]SnakeBackwardParamsTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .ga = ga,
            .gib = gib,
            .x = x,
            .gy = gy,
            .alpha = alpha,
            .inv_b = inv_b,
            .rows = rows,
            .cols = cols,
            .col_start = task_i * cols / thread_count,
            .col_end = (task_i + 1) * cols / thread_count,
        };
    }
    pool.parallelChunks(SnakeBackwardParamsTask, tasks[0..thread_count], runSnakeBackwardParamsTask);
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
    const input = contiguousDataConst(x, rows * cols);
    const grad = contiguousDataConst(gy, rows * cols);
    const gx_data: ?[]f32 = if (gx) |t| contiguousData(t, rows * cols) else null;
    const gw_data: ?[]f32 = if (gw) |t| contiguousData(t, cols) else null;
    const gb_data: ?[]f32 = if (gb) |t| contiguousData(t, cols) else null;
    if (maybeParallelGroupNormBackward(pc, gx_data, gw_data, gb_data, input, grad, weight, rows, cols, groups, eps)) return;
    groupNormBackwardGroupRange(gx_data, gw_data, gb_data, input, grad, weight, rows, cols, groups, eps, 0, groups);
}

const GroupNormBackwardTask = struct {
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
};

fn runGroupNormBackwardTask(task: *const GroupNormBackwardTask) void {
    groupNormBackwardGroupRange(task.gx, task.gw, task.gb, task.x, task.gy, task.weight, task.rows, task.cols, task.groups, task.eps, task.group_start, task.group_end);
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

    var tasks: [parallel.vector_max_threads]GroupNormBackwardTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .gx = gx,
            .gw = gw,
            .gb = gb,
            .x = x,
            .gy = gy,
            .weight = weight,
            .rows = rows,
            .cols = cols,
            .groups = groups,
            .eps = eps,
            .group_start = task_i * groups / thread_count,
            .group_end = (task_i + 1) * groups / thread_count,
        };
    }
    pool.parallelChunks(GroupNormBackwardTask, tasks[0..thread_count], runGroupNormBackwardTask);
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
