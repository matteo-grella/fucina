//! Scalar reference backend: plain serial loops, no SIMD, no BLAS, no GPU.
//! The `kernels` namespace below is the kernel set `backend/interface.zig`
//! names, with the same signatures as `native.zig`: pool-taking kernels take
//! `pc: ParallelConfig` first and ignore it (every kernel here runs
//! serially). Routing shared by both backends (im2col, the Winograd
//! transforms, the conv2d/pool2d backward loops) reuses the `vector/`
//! implementation serially; everything with a SIMD counterpart is an
//! independent scalar loop. `parity_test.zig` holds the two in agreement.
const std = @import("std");
const ops = @import("ops.zig");
const packed_matmul = @import("packed.zig");
const quantized_matmul = @import("quant.zig");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");
const vector_common = @import("vector/common.zig");
const vector_conv = @import("vector/conv.zig");
const vector_pool = @import("vector/pool.zig");
const vector_winograd = @import("vector/winograd.zig");

const cpu = @This();
const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;
const Conv2dDims = vector_conv.Conv2dDims;
const Conv1dDims = vector_conv.Conv1dDims;
const PoolKind = vector_pool.PoolKind;
const Pool2dDims = vector_pool.Pool2dDims;
const WinogradF2Dims = vector_winograd.F2Dims;
/// The same type the native backend takes (`vector/common.zig`), so the
/// two kernel sets share one signature.
pub const ParallelConfig = vector_common.ParallelConfig;

/// The kernel set this backend provides, by interface name.
pub const kernels = struct {
    pub const addInto = cpu.addInto;
    pub const addContiguousIntoUnchecked = cpu.addContiguousIntoUnchecked;
    pub const divContiguousIntoUnchecked = cpu.divContiguousIntoUnchecked;
    pub const maximumContiguousIntoUnchecked = cpu.maximumContiguousIntoUnchecked;
    pub const minimumContiguousIntoUnchecked = cpu.minimumContiguousIntoUnchecked;
    pub const subInto = cpu.subInto;
    pub const subContiguousIntoUnchecked = cpu.subContiguousIntoUnchecked;
    pub const mulInto = cpu.mulInto;
    pub const mulContiguousIntoUnchecked = cpu.mulContiguousIntoUnchecked;
    pub const elementwiseContiguousIntoTyped = cpu.elementwiseContiguousIntoTyped;
    pub const scaleInto = cpu.scaleInto;
    pub const addScaledSlice = cpu.addScaledSlice;
    pub const addRowVectorSlice = cpu.addRowVectorSlice;
    pub const causalDepthwiseConv1dInto = cpu.causalDepthwiseConv1dInto;
    pub const causalDepthwiseConv1dBackwardInputInto = cpu.causalDepthwiseConv1dBackwardInputInto;
    pub const causalDepthwiseConv1dBackwardKernelInto = cpu.causalDepthwiseConv1dBackwardKernelInto;
    pub const causalConv1dInto = cpu.causalConv1dInto;
    pub const conv2dInto = cpu.conv2dInto;
    pub const conv2dBackwardInputInto = cpu.conv2dBackwardInputInto;
    pub const conv2dBackwardWeightInto = cpu.conv2dBackwardWeightInto;
    pub const im2colInto = cpu.im2colInto;
    pub const col2imInto = cpu.col2imInto;
    pub const winogradF2WeightTransformInto = cpu.winogradF2WeightTransformInto;
    pub const winogradF2InputTransformInto = cpu.winogradF2InputTransformInto;
    pub const winogradF2OutputTransformInto = cpu.winogradF2OutputTransformInto;
    pub const winogradF4WeightTransformInto = cpu.winogradF4WeightTransformInto;
    pub const winogradF4InputTransformInto = cpu.winogradF4InputTransformInto;
    pub const winogradF4OutputTransformInto = cpu.winogradF4OutputTransformInto;
    pub const pool2dInto = cpu.pool2dInto;
    pub const avgPool2dBackwardInto = cpu.avgPool2dBackwardInto;
    pub const maxPool2dBackwardInto = cpu.maxPool2dBackwardInto;
    pub const upsample2xNearestInto = cpu.upsample2xNearestInto;
    pub const preluChannelsInto = cpu.preluChannelsInto;
    pub const preluChannelsBackwardInputInto = cpu.preluChannelsBackwardInputInto;
    pub const preluChannelsBackwardAlphaInto = cpu.preluChannelsBackwardAlphaInto;
    pub const channelAffineInto = cpu.channelAffineInto;
    pub const conv1dInto = cpu.conv1dInto;
    pub const conv1dBackwardInputInto = cpu.conv1dBackwardInputInto;
    pub const conv1dBackwardWeightInto = cpu.conv1dBackwardWeightInto;
    pub const col2im1dInto = cpu.col2im1dInto;
    pub const col2im1dBackwardInto = cpu.col2im1dBackwardInto;
    pub const snakeInto = cpu.snakeInto;
    pub const snakeBackwardInputInto = cpu.snakeBackwardInputInto;
    pub const snakeBackwardParamsInto = cpu.snakeBackwardParamsInto;
    pub const groupNormInto = cpu.groupNormInto;
    pub const groupNormBackwardInto = cpu.groupNormBackwardInto;
    pub const causalConv1dBackwardInputInto = cpu.causalConv1dBackwardInputInto;
    pub const causalConv1dBackwardWeightInto = cpu.causalConv1dBackwardWeightInto;
    pub const groupedCausalConv1dInto = cpu.groupedCausalConv1dInto;
    pub const groupedCausalConv1dBackwardInputInto = cpu.groupedCausalConv1dBackwardInputInto;
    pub const groupedCausalConv1dBackwardWeightInto = cpu.groupedCausalConv1dBackwardWeightInto;
    pub const unaryContiguousIntoUnchecked = cpu.unaryContiguousIntoUnchecked;
    pub const leakyReluContiguousIntoUnchecked = cpu.leakyReluContiguousIntoUnchecked;
    pub const softcapContiguousIntoUnchecked = cpu.softcapContiguousIntoUnchecked;
    pub const clampContiguousIntoUnchecked = cpu.clampContiguousIntoUnchecked;
    pub const gatedContiguousIntoUnchecked = cpu.gatedContiguousIntoUnchecked;
    pub const sumInto = cpu.sumInto;
    pub const sumSlice = cpu.sumSlice;
    pub const prodInto = cpu.prodInto;
    pub const prodSlice = cpu.prodSlice;
    pub const sumSliceTyped = cpu.sumSliceTyped;
    pub const dot = cpu.dot;
    pub const gemm = cpu.gemm;
    pub const gemmBatched = cpu.gemmBatched;
    pub const packDenseRhs = cpu.packDenseRhs;
    pub const quantizeMatmulRhsBlockwiseI8 = cpu.quantizeMatmulRhsBlockwiseI8;
    pub const quantizeMatmulRhsQ4_0 = cpu.quantizeMatmulRhsQ4_0;
    pub const quantizeMatmulRhsQ8_0 = cpu.quantizeMatmulRhsQ8_0;
    pub const matmul2DQuantizedRhs = cpu.matmul2DQuantizedRhs;
    pub const matmulQuantizedRhs = cpu.matmulQuantizedRhs;
    pub const matmulPacked = cpu.matmulPacked;
    pub const matmul2DPackedQ8_0x4LhsRhs = cpu.matmul2DPackedQ8_0x4LhsRhs;
    pub const matmulPackedSlice = cpu.matmulPackedSlice;
    pub const unaryRowSlice = cpu.unaryRowSlice;
    pub const mulRowSlice = cpu.mulRowSlice;
    pub const matmul2DPackedPaddedQ8_0x4LhsRhs = cpu.matmul2DPackedPaddedQ8_0x4LhsRhs;
};

// The conv2d backward gather cores are scalar loops (no SIMD divergence), so
// the scalar reference runs the same `vector/` cores serially, trivially in
// parity. The wrappers ignore `pc` and pass an empty config: this backend
// never threads.
pub fn conv2dBackwardInputInto(pc: ParallelConfig, out: *Tensor, gy: *const Tensor, weight: *const Tensor, d: Conv2dDims) void {
    _ = pc;
    vector_conv.conv2dBackwardInputInto(.{}, out, gy, weight, d);
}
pub fn conv2dBackwardWeightInto(pc: ParallelConfig, out: *Tensor, input: *const Tensor, gy: *const Tensor, d: Conv2dDims) void {
    _ = pc;
    vector_conv.conv2dBackwardWeightInto(.{}, out, input, gy, d);
}
// im2col/col2im and the pool backwards are pure data movement (col2im's row
// adds are elementwise — no reassociation, so the vector core is bit-equal to
// a scalar loop) — the scalar reference reuses them serially (same rationale
// as the conv2d backwards above).
pub fn im2colInto(pc: ParallelConfig, col: *Tensor, input: *const Tensor, d: Conv2dDims) void {
    _ = pc;
    vector_conv.im2colInto(.{}, col, input, d);
}
pub fn col2imInto(pc: ParallelConfig, out: *Tensor, col: *const Tensor, d: Conv2dDims) void {
    _ = pc;
    vector_conv.col2imInto(.{}, out, col, d);
}
// The Winograd F(2×2,3×3)/F(4×4,3×3) transforms are an exec-level ROUTE
// shared by both backends (like im2col): the scalar reference reuses them
// serially so the two backends compute identical values on Winograd-routed
// convs; the GEMMs between the transforms still go through this backend's
// own scalar matmul.
pub fn winogradF2WeightTransformInto(pc: ParallelConfig, u: *const [16][]f32, w: []const f32, cout: usize, cin: usize) void {
    _ = pc;
    vector_winograd.f2WeightTransformInto(.{}, u, w, cout, cin);
}
pub fn winogradF2InputTransformInto(pc: ParallelConfig, v: *const [16][]f32, x: []const f32, d: WinogradF2Dims) void {
    _ = pc;
    vector_winograd.f2InputTransformInto(.{}, v, x, d);
}
pub fn winogradF2OutputTransformInto(pc: ParallelConfig, y: []f32, m: *const [16][]const f32, bias: ?[]const f32, fuse_relu: bool, d: WinogradF2Dims) void {
    _ = pc;
    vector_winograd.f2OutputTransformInto(.{}, y, m, bias, fuse_relu, d);
}
pub fn winogradF4WeightTransformInto(pc: ParallelConfig, u: *const [36][]f32, w: []const f32, cout: usize, cin: usize) void {
    _ = pc;
    vector_winograd.f4WeightTransformInto(.{}, u, w, cout, cin);
}
pub fn winogradF4InputTransformInto(pc: ParallelConfig, v: *const [36][]f32, x: []const f32, d: WinogradF2Dims) void {
    _ = pc;
    vector_winograd.f4InputTransformInto(.{}, v, x, d);
}
pub fn winogradF4OutputTransformInto(pc: ParallelConfig, y: []f32, m: *const [36][]const f32, bias: ?[]const f32, fuse_relu: bool, d: WinogradF2Dims) void {
    _ = pc;
    vector_winograd.f4OutputTransformInto(.{}, y, m, bias, fuse_relu, d);
}
pub fn avgPool2dBackwardInto(pc: ParallelConfig, out: *Tensor, gy: *const Tensor, d: Pool2dDims) void {
    _ = pc;
    vector_pool.avgPool2dBackwardInto(.{}, out, gy, d);
}
pub fn maxPool2dBackwardInto(pc: ParallelConfig, out: *Tensor, input: *const Tensor, gy: *const Tensor, d: Pool2dDims) void {
    _ = pc;
    vector_pool.maxPool2dBackwardInto(.{}, out, input, gy, d);
}
pub fn preluChannelsBackwardInputInto(pc: ParallelConfig, gx: []f32, gy: []const f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize) void {
    _ = pc;
    for (0..rows) |r| {
        for (0..cols) |c| {
            const i = r * cols + c;
            gx[i] = if (x[i] > 0) gy[i] else gy[i] * alpha[c];
        }
    }
}
pub fn preluChannelsBackwardAlphaInto(pc: ParallelConfig, galpha: []f32, gy: []const f32, x: []const f32, rows: usize, cols: usize) void {
    _ = pc;
    @memset(galpha, 0);
    for (0..rows) |r| {
        for (0..cols) |c| {
            const i = r * cols + c;
            if (x[i] <= 0) galpha[c] += gy[i] * x[i];
        }
    }
}

/// Scalar reference pool2d (independent of the native vector kernel — see
/// `Pool2dDims` in vector/pool.zig for the layout and border semantics).
pub fn pool2dInto(pc: ParallelConfig, comptime kind: PoolKind, out: *Tensor, input: *const Tensor, d: Pool2dDims) void {
    _ = pc;
    const o = out.data();
    const in = input.dataConst();
    for (0..d.oh) |oh| {
        for (0..d.ow) |ow| {
            for (0..d.c) |c| {
                var acc: f32 = if (kind == .max) -std.math.inf(f32) else 0;
                var count: usize = 0;
                for (0..d.kh) |kh| {
                    const ih_i = @as(isize, @intCast(oh * d.stride_h + kh)) - @as(isize, @intCast(d.pad_h));
                    if (ih_i < 0 or ih_i >= @as(isize, @intCast(d.h))) continue;
                    for (0..d.kw) |kw| {
                        const iw_i = @as(isize, @intCast(ow * d.stride_w + kw)) - @as(isize, @intCast(d.pad_w));
                        if (iw_i < 0 or iw_i >= @as(isize, @intCast(d.w))) continue;
                        const v = in[(@as(usize, @intCast(ih_i)) * d.w + @as(usize, @intCast(iw_i))) * d.c + c];
                        switch (kind) {
                            .max => acc = @max(acc, v),
                            .avg, .sum => acc += v,
                        }
                        count += 1;
                    }
                }
                if (kind == .avg and count > 0) acc /= @floatFromInt(count);
                o[(oh * d.ow + ow) * d.c + c] = acc;
            }
        }
    }
}

/// Scalar reference 2× nearest-neighbour upsample.
pub fn upsample2xNearestInto(pc: ParallelConfig, out: *Tensor, input: *const Tensor, h: usize, w: usize, c: usize) void {
    _ = pc;
    const o = out.data();
    const in = input.dataConst();
    for (0..2 * h) |oy| {
        for (0..2 * w) |ox| {
            for (0..c) |ci| {
                o[(oy * 2 * w + ox) * c + ci] = in[((oy / 2) * w + ox / 2) * c + ci];
            }
        }
    }
}

/// Scalar reference per-channel PReLU.
pub fn preluChannelsInto(pc: ParallelConfig, z: []f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize) void {
    _ = pc;
    for (0..rows) |r| {
        for (0..cols) |c| {
            const i = r * cols + c;
            z[i] = if (x[i] > 0) x[i] else x[i] * alpha[c];
        }
    }
}

/// Scalar reference per-channel affine (frozen-stats BatchNorm); a null
/// `shift` degrades to the per-channel scale (the affine's own input-VJP).
pub fn channelAffineInto(pc: ParallelConfig, z: []f32, x: []const f32, scale: []const f32, shift: ?[]const f32, rows: usize, cols: usize) void {
    _ = pc;
    for (0..rows) |r| {
        for (0..cols) |c| {
            const i = r * cols + c;
            z[i] = if (shift) |t| x[i] * scale[c] + t[c] else x[i] * scale[c];
        }
    }
}

const q8_0_lhs_stack_blocks: usize = 512;

fn checkedTensorProduct(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch tensor.TensorError.InvalidDataLength;
}

fn checkedQuantizedProduct(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch quantized_matmul.types.QuantizedFormatError.InvalidQuantizedLength;
}

pub fn addInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    try tensor.requireSameShape(a, b);
    try tensor.requireSameShape(out, a);
    addContiguousIntoUnchecked(.{}, out, a, b, a.len());
}

pub fn divContiguousIntoUnchecked(pc: ParallelConfig, out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
    _ = pc;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, xv, yv| dst.* = xv / yv;
}

pub fn maximumContiguousIntoUnchecked(pc: ParallelConfig, out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
    _ = pc;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, xv, yv| dst.* = if (xv != xv or yv != yv) std.math.nan(f32) else @max(xv, yv);
}

pub fn minimumContiguousIntoUnchecked(pc: ParallelConfig, out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
    _ = pc;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, xv, yv| dst.* = if (xv != xv or yv != yv) std.math.nan(f32) else @min(xv, yv);
}

pub fn addContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    _ = pc;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, xv, yv| dst.* = xv + yv;
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
    _ = pc;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, xv, yv| dst.* = xv - yv;
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
    _ = pc;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, xv, yv| dst.* = xv * yv;
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
    const x = contiguousDataConstOf(dtype, a, len);
    const y = contiguousDataConstOf(dtype, b, len);
    const z = contiguousDataOf(dtype_mod.outputDType(.pointwise, dtype), out, len);
    elementwiseSlicesTyped(dtype, op, z, x, y);
}

pub fn scaleInto(pc: ParallelConfig, out: *Tensor, a: *const Tensor, scalar_value: f32) !void {
    _ = pc;
    try tensor.requireSameShape(out, a);
    const x = a.dataConst();
    const z = out.data();
    for (z, x) |*dst, xv| dst.* = xv * scalar_value;
}

pub fn addScaledSlice(z: []f32, x: []const f32, scalar_value: f32) void {
    for (z, x) |*dst, xv| dst.* += xv * scalar_value;
}

/// `z[row] += row_vector` for every row, then `op` (a fused bias + activation
/// when given).
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

pub fn causalDepthwiseConv1dInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    kernel: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
) void {
    _ = pc;
    causalDepthwiseConv1dRange(out.data(), input.dataConst(), kernel.dataConst(), state, seq, channels, taps, dilation, 0, channels);
}

pub fn causalDepthwiseConv1dBackwardInputInto(
    pc: ParallelConfig,
    out: *Tensor,
    gy: *const Tensor,
    kernel: *const Tensor,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
) void {
    _ = pc;
    causalDepthwiseConv1dBackwardInputRange(out.data(), gy.dataConst(), kernel.dataConst(), seq, channels, taps, dilation, 0, channels);
}

pub fn causalDepthwiseConv1dBackwardKernelInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
) void {
    _ = pc;
    causalDepthwiseConv1dBackwardKernelRange(out.data(), input.dataConst(), gy.dataConst(), state, seq, channels, taps, dilation, 0, channels);
}

pub fn causalConv1dInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
) void {
    _ = pc;
    groupedCausalConv1dRange(out.data(), input.dataConst(), weight.dataConst(), state, in_channels, out_channels, taps, dilation, 1, 0, seq);
}

/// Scalar reference conv2d (independent of the native vector kernel — see
/// `Conv2dDims` in vector/conv.zig for the layout). Channel-last [H,W,Cin] ->
/// [OH,OW,Cout] with stride, explicit zero pad, grouped/depthwise.
pub fn conv2dInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    bias: ?[]const f32,
    d: Conv2dDims,
) void {
    _ = pc;
    const o = out.data();
    const in = input.dataConst();
    const w = weight.dataConst();
    const cin_pg = d.cin / d.groups;
    const cout_pg = d.cout / d.groups;
    for (0..d.oh) |oh| {
        for (0..d.ow) |ow| {
            for (0..d.cout) |oc| {
                const g = oc / cout_pg;
                var acc: f32 = if (bias) |b| b[oc] else 0;
                for (0..d.kh) |kh| {
                    const ih_i = @as(isize, @intCast(oh * d.stride_h + kh)) - @as(isize, @intCast(d.pad_h));
                    if (ih_i < 0 or ih_i >= @as(isize, @intCast(d.h))) continue;
                    for (0..d.kw) |kw| {
                        const iw_i = @as(isize, @intCast(ow * d.stride_w + kw)) - @as(isize, @intCast(d.pad_w));
                        if (iw_i < 0 or iw_i >= @as(isize, @intCast(d.w))) continue;
                        const ih: usize = @intCast(ih_i);
                        const iw: usize = @intCast(iw_i);
                        for (0..cin_pg) |ic| {
                            const iv = in[(ih * d.w + iw) * d.cin + g * cin_pg + ic];
                            const wv = w[((oc * d.kh + kh) * d.kw + kw) * cin_pg + ic];
                            acc += iv * wv;
                        }
                    }
                }
                o[(oh * d.ow + ow) * d.cout + oc] = acc;
            }
        }
    }
}

/// Scalar reference conv1d (independent of the native vector kernel — see
/// `Conv1dDims` in vector/conv.zig for the layout): general non-causal 1-D
/// convolution with symmetric zero pad, stride, dilation, and groups.
pub fn conv1dInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    d: Conv1dDims,
) void {
    _ = pc;
    const o = out.data();
    const in = input.dataConst();
    const w = weight.dataConst();
    const in_per_group = d.in_channels / d.groups;
    const out_per_group = d.out_channels / d.groups;
    for (0..d.out_len) |t| {
        for (0..d.out_channels) |oc| {
            const g = oc / out_per_group;
            var acc: f32 = 0;
            for (0..d.taps) |k| {
                const pos = t * d.stride + k * d.dilation;
                if (pos < d.pad) continue;
                const src = pos - d.pad;
                if (src >= d.seq) continue;
                for (0..in_per_group) |local_i| {
                    const iv = in[src * d.in_channels + g * in_per_group + local_i];
                    const wv = w[(k * in_per_group + local_i) * d.out_channels + oc];
                    acc += iv * wv;
                }
            }
            o[t * d.out_channels + oc] = acc;
        }
    }
}

/// Scalar reference col2im1d gather (see the vector kernel's doc comment for
/// the layout contract): `col` is `[t_in, taps*out_channels]` with column
/// index `oc*taps + k`, `out` is `[out_len, out_channels]` channel-fast rows;
/// rows past `(t_in-1)*stride + taps - 2*pad` are the ConvTranspose
/// output_padding and are zeroed.
pub fn col2im1dInto(
    pc: ParallelConfig,
    out: *Tensor,
    col: *const Tensor,
    t_in: usize,
    out_len: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
) void {
    _ = pc;
    const o = out.data();
    const c = col.dataConst();
    const t_conv = (t_in - 1) * stride + taps - 2 * pad;
    for (0..out_len) |t_out| {
        if (t_out >= t_conv) {
            for (0..out_channels) |oc| o[t_out * out_channels + oc] = 0;
            continue;
        }
        const t_abs = t_out + pad;
        const t_in_min: usize = if (t_abs + 1 > taps) (t_abs + 1 - taps + stride - 1) / stride else 0;
        const t_in_max: usize = @min(t_in - 1, t_abs / stride);
        for (0..out_channels) |oc| {
            var acc: f32 = 0;
            var ti = t_in_min;
            while (ti <= t_in_max) : (ti += 1) {
                const k = t_abs - ti * stride;
                std.debug.assert(k < taps);
                acc += c[ti * (taps * out_channels) + oc * taps + k];
            }
            o[t_out * out_channels + oc] = acc;
        }
    }
}

/// Scalar reference conv1d backward-input (see the vector kernel's doc
/// comment): `gx[ti,ic] = Σ_k gy[n/stride, oc]·w[k, ic%ipg, oc]` over the
/// group's out channels, with `n = ti + pad - k*dilation` valid when
/// non-negative, divisible by stride, and `n/stride < out_len`.
pub fn conv1dBackwardInputInto(
    pc: ParallelConfig,
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    d: Conv1dDims,
) void {
    _ = pc;
    const o = out.data();
    const g = gy.dataConst();
    const w = weight.dataConst();
    const in_per_group = d.in_channels / d.groups;
    const out_per_group = d.out_channels / d.groups;
    for (0..d.seq) |ti| {
        for (0..d.in_channels) |ic| {
            const group = ic / in_per_group;
            const local_i = ic % in_per_group;
            var acc: f32 = 0;
            for (0..d.taps) |k| {
                const shifted = k * d.dilation;
                if (shifted > ti + d.pad) continue;
                const n = ti + d.pad - shifted;
                if (n % d.stride != 0) continue;
                const t = n / d.stride;
                if (t >= d.out_len) continue;
                for (0..out_per_group) |local_o| {
                    const oc = group * out_per_group + local_o;
                    acc += g[t * d.out_channels + oc] * w[(k * in_per_group + local_i) * d.out_channels + oc];
                }
            }
            o[ti * d.in_channels + ic] = acc;
        }
    }
}

/// Scalar reference conv1d backward-weight (see the vector kernel's doc
/// comment): `gw[k, li, oc] = Σ_t gy[t,oc]·x[t*stride + k*dilation - pad,
/// g(oc)*ipg + li]`, skipping out-of-range padded input rows.
pub fn conv1dBackwardWeightInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    d: Conv1dDims,
) void {
    _ = pc;
    const o = out.data();
    const in = input.dataConst();
    const g = gy.dataConst();
    const in_per_group = d.in_channels / d.groups;
    const out_per_group = d.out_channels / d.groups;
    for (0..d.taps) |k| {
        for (0..in_per_group) |local_i| {
            for (0..d.out_channels) |oc| {
                const group = oc / out_per_group;
                var acc: f32 = 0;
                for (0..d.out_len) |t| {
                    const pos = t * d.stride + k * d.dilation;
                    if (pos < d.pad) continue;
                    const src = pos - d.pad;
                    if (src >= d.seq) continue;
                    acc += g[t * d.out_channels + oc] * in[src * d.in_channels + group * in_per_group + local_i];
                }
                o[(k * in_per_group + local_i) * d.out_channels + oc] = acc;
            }
        }
    }
}

/// Scalar reference col2im1d backward (see the vector kernel's doc comment):
/// `gcol[ti, oc*taps + k] = gy[ti*stride + k - pad, oc]` when the row index
/// lands in `[0, t_conv)`, else 0.
pub fn col2im1dBackwardInto(
    pc: ParallelConfig,
    out: *Tensor,
    gy: *const Tensor,
    t_in: usize,
    gy_len: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
) void {
    _ = pc;
    const o = out.data();
    const g = gy.dataConst();
    const t_conv = (t_in - 1) * stride + taps - 2 * pad;
    std.debug.assert(gy_len >= t_conv);
    const row_stride = taps * out_channels;
    for (0..t_in) |ti| {
        for (0..out_channels) |oc| {
            for (0..taps) |k| {
                const pos = ti * stride + k;
                var value: f32 = 0;
                if (pos >= pad) {
                    const t_out = pos - pad;
                    if (t_out < t_conv) value = g[t_out * out_channels + oc];
                }
                o[ti * row_stride + oc * taps + k] = value;
            }
        }
    }
}

pub fn causalConv1dBackwardInputInto(
    pc: ParallelConfig,
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
) void {
    _ = pc;
    groupedCausalConv1dBackwardInputRange(out.data(), gy.dataConst(), weight.dataConst(), seq, in_channels, out_channels, taps, dilation, 1, 0, seq);
}

pub fn causalConv1dBackwardWeightInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
) void {
    _ = pc;
    groupedCausalConv1dBackwardWeightRange(out.data(), input.dataConst(), gy.dataConst(), state, seq, in_channels, out_channels, taps, dilation, 1, 0, taps * in_channels);
}

pub fn groupedCausalConv1dInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
) void {
    _ = pc;
    groupedCausalConv1dRange(out.data(), input.dataConst(), weight.dataConst(), state, in_channels, out_channels, taps, dilation, groups, 0, seq);
}

pub fn groupedCausalConv1dBackwardInputInto(
    pc: ParallelConfig,
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
) void {
    _ = pc;
    groupedCausalConv1dBackwardInputRange(out.data(), gy.dataConst(), weight.dataConst(), seq, in_channels, out_channels, taps, dilation, groups, 0, seq);
}

pub fn groupedCausalConv1dBackwardWeightInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
) void {
    _ = pc;
    const in_per_group = in_channels / groups;
    groupedCausalConv1dBackwardWeightRange(out.data(), input.dataConst(), gy.dataConst(), state, seq, in_channels, out_channels, taps, dilation, groups, 0, taps * in_per_group);
}

pub fn unaryContiguousIntoUnchecked(
    pc: ParallelConfig,
    comptime op: ops.UnaryOp,
    out: *Tensor,
    a: *const Tensor,
    len: usize,
) void {
    _ = pc;
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    for (z, x) |*dst, value| dst.* = ops.unaryScalar(op, value);
}

/// Scalar reference snake activation (see the vector kernel's doc comment):
/// `y[t,c] = x[t,c] + inv_b[c] * sin(alpha[c]*x[t,c])^2` over contiguous
/// `[rows, cols]` rows; `inv_b` is precomputed by the caller.
pub fn snakeInto(
    pc: ParallelConfig,
    out: *Tensor,
    x: *const Tensor,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
) void {
    _ = pc;
    const input = contiguousDataConst(x, rows * cols);
    const output = contiguousData(out, rows * cols);
    for (0..rows) |r| {
        for (0..cols) |c| {
            const v = input[r * cols + c];
            const s = @sin(alpha[c] * v);
            output[r * cols + c] = v + inv_b[c] * s * s;
        }
    }
}

/// Scalar reference GroupNorm (ggml group_norm semantics; see the vector
/// kernel's doc comment): per group of channel columns, f64-accumulated mean
/// and biased variance over all rows × (cols/groups) elements, then
/// `y = (x - mean) * (1/sqrt(var + eps))` in f32 (eps inside the sqrt),
/// with the optional per-channel affine applied after normalization.
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
    _ = pc;
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
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatCast(sum2 / count + eps)));
        for (0..rows) |r| {
            for (0..cols_per_group) |local_c| {
                const c = col_start + local_c;
                var value = (input[r * cols + c] - mean) * scale;
                if (weight) |w| value *= w[c];
                if (bias) |b| value += b[c];
                output[r * cols + c] = value;
            }
        }
    }
}

/// Scalar reference snake backward-input (see the vector kernel's doc
/// comment): `gx = gy * (1 + inv_b[c]*alpha[c]*sin(2*alpha[c]*x))`.
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
    _ = pc;
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

/// Scalar reference snake backward-params (see the vector kernel's doc
/// comment): fills both per-channel gradients in one pass —
/// `galpha[c] = Σ_t gy·inv_b[c]·x·sin(2·alpha[c]·x)` and
/// `ginv_b[c] = Σ_t gy·sin(alpha[c]·x)^2` (f32 row-order accumulation).
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
    _ = pc;
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
            const s = @sin(alpha[c] * v);
            const s2 = @sin(2 * alpha[c] * v);
            ga[c] += gv * inv_b[c] * v * s2;
            gib[c] += gv * s * s;
        }
    }
}

/// Scalar reference GroupNorm backward (see the vector kernel's doc comment):
/// recomputes the per-group f64 two-pass statistics like the forward, then
/// fills any of gx (f64 group means of ĝ and ĝ·x̂, f32 apply), gw, gb.
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
    _ = pc;
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
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatCast(sum2 / count + eps)));

        if (gw_data != null or gb_data != null) {
            for (0..cols_per_group) |local_c| {
                const c = col_start + local_c;
                var acc_w: f32 = 0;
                var acc_b: f32 = 0;
                for (0..rows) |r| {
                    const v = input[r * cols + c];
                    const gv = grad[r * cols + c];
                    acc_w += gv * (v - mean) * scale;
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
                const xh = (input[r * cols + c] - mean) * scale;
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
                const xh = (input[r * cols + c] - mean) * scale;
                dx[r * cols + c] = scale * (gh - mean_g - xh * mean_gx);
            }
        }
    }
}

pub fn leakyReluContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    negative_slope: f32,
) void {
    _ = pc;
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    for (z, x) |*dst, value| dst.* = if (value >= 0) value else value * negative_slope;
}

/// `cap * tanh(x / cap)`: the logit softcap.
pub fn softcapContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    cap: f32,
) void {
    _ = pc;
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    const inv = 1.0 / cap;
    for (z, x) |*dst, value| dst.* = cap * std.math.tanh(value * inv);
}

pub fn clampContiguousIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    min_value: f32,
    max_value: f32,
) void {
    _ = pc;
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    for (z, x) |*dst, value| dst.* = @min(@max(value, min_value), max_value);
}

pub fn gatedContiguousIntoUnchecked(
    pc: ParallelConfig,
    comptime op: ops.GatedOp,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    _ = pc;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, left, gate| dst.* = ops.gatedPairScalar(op, gate, left);
}

pub fn sumInto(pc: ParallelConfig, out: *Tensor, a: *const Tensor) !void {
    _ = pc;
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    out.data()[0] = sumSlice(a.dataConst());
}

pub fn sumSlice(values: []const f32) f32 {
    var s: f32 = 0;
    for (values) |v| s += v;
    return s;
}

pub fn prodInto(pc: ParallelConfig, out: *Tensor, a: *const Tensor) !void {
    _ = pc;
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    out.data()[0] = prodSlice(a.dataConst());
}

pub fn prodSlice(values: []const f32) f32 {
    var p: f32 = 1;
    for (values) |v| p *= v;
    return p;
}

pub fn sumSliceTyped(
    pc: ParallelConfig,
    comptime dtype: DType,
    values: []const dtype_mod.Scalar(dtype),
) dtype_mod.Scalar(dtype_mod.outputDType(.reduction, dtype)) {
    _ = pc;
    const compute_dtype = comptime dtype_mod.computeDType(.reduction, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.reduction, dtype);
    var acc: dtype_mod.Scalar(compute_dtype) = 0;
    for (values) |value| {
        acc += dtype_mod.castFloat(dtype, compute_dtype, value);
    }
    return dtype_mod.castFloat(compute_dtype, output_dtype, acc);
}

/// Full dot product into the scalar `out`, accumulated in the dtype's
/// matmul compute dtype.
pub fn dot(
    pc: ParallelConfig,
    comptime dtype: DType,
    out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !void {
    _ = pc;
    try tensor.requireSameShapeOf(dtype, a, b);
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    out.data()[0] = dotSliceTyped(dtype, a.dataConst(), b.dataConst());
}

fn causalDepthwiseConv1dRange(
    out: []f32,
    input: []const f32,
    kernel: []const f32,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = dilation * (taps - 1);
    for (0..seq) |t| {
        for (channel_start..channel_end) |c| {
            var acc: f32 = 0;
            for (0..taps) |k| {
                acc += causalDepthwiseInputValue(input, state, seq, channels, pad, dilation, t, c, k) * kernel[c * taps + k];
            }
            out[t * channels + c] = acc;
        }
    }
}

fn causalDepthwiseConv1dBackwardInputRange(
    out: []f32,
    gy: []const f32,
    kernel: []const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = dilation * (taps - 1);
    for (0..seq) |p| {
        for (channel_start..channel_end) |c| {
            var acc: f32 = 0;
            for (0..taps) |k| {
                const t_base = p + pad;
                if (k * dilation > t_base) continue;
                const t = t_base - k * dilation;
                if (t < seq) acc += gy[t * channels + c] * kernel[c * taps + k];
            }
            out[p * channels + c] = acc;
        }
    }
}

fn causalDepthwiseConv1dBackwardKernelRange(
    out: []f32,
    input: []const f32,
    gy: []const f32,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = dilation * (taps - 1);
    for (channel_start..channel_end) |c| {
        for (0..taps) |k| {
            var acc: f32 = 0;
            for (0..seq) |t| {
                acc += gy[t * channels + c] * causalDepthwiseInputValue(input, state, seq, channels, pad, dilation, t, c, k);
            }
            out[c * taps + k] = acc;
        }
    }
}

fn groupedCausalConv1dRange(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    state: ?[]const f32,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    t_start: usize,
    t_end: usize,
) void {
    const pad = dilation * (taps - 1);
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    for (t_start..t_end) |t| {
        for (0..out_channels) |o| {
            const group = o / out_per_group;
            const input_start = group * in_per_group;
            var acc: f32 = 0;
            for (0..taps) |k| {
                for (0..in_per_group) |local_i| {
                    const i = input_start + local_i;
                    acc += causalConvInputValue(input, state, in_channels, pad, t, i, k, dilation) * weight[(k * in_per_group + local_i) * out_channels + o];
                }
            }
            out[t * out_channels + o] = acc;
        }
    }
}

fn groupedCausalConv1dBackwardInputRange(
    out: []f32,
    gy: []const f32,
    weight: []const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    p_start: usize,
    p_end: usize,
) void {
    const pad = dilation * (taps - 1);
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    for (p_start..p_end) |p| {
        for (0..in_channels) |i| {
            const group = i / in_per_group;
            const local_i = i - group * in_per_group;
            const out_start = group * out_per_group;
            var acc: f32 = 0;
            for (0..taps) |k| {
                const t = p + pad - k * dilation;
                if (t >= seq) continue;
                for (out_start..out_start + out_per_group) |o| {
                    acc += gy[t * out_channels + o] * weight[(k * in_per_group + local_i) * out_channels + o];
                }
            }
            out[p * in_channels + i] = acc;
        }
    }
}

fn groupedCausalConv1dBackwardWeightRange(
    out: []f32,
    input: []const f32,
    gy: []const f32,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    row_start: usize,
    row_end: usize,
) void {
    const pad = dilation * (taps - 1);
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    for (row_start..row_end) |row| {
        const k = row / in_per_group;
        const local_i = row % in_per_group;
        for (0..out_channels) |o| {
            const group = o / out_per_group;
            const i = group * in_per_group + local_i;
            var acc: f32 = 0;
            for (0..seq) |t| {
                acc += gy[t * out_channels + o] * causalConvInputValue(input, state, in_channels, pad, t, i, k, dilation);
            }
            out[row * out_channels + o] = acc;
        }
    }
}

fn causalConvInputValue(
    input: []const f32,
    state: ?[]const f32,
    in_channels: usize,
    pad: usize,
    t: usize,
    i: usize,
    k: usize,
    dilation: usize,
) f32 {
    const shifted = t + k * dilation;
    if (shifted >= pad) return input[(shifted - pad) * in_channels + i];
    const s = state orelse return 0;
    return s[shifted * in_channels + i];
}

fn causalDepthwiseInputValue(
    input: []const f32,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    pad: usize,
    dilation: usize,
    t: usize,
    c: usize,
    k: usize,
) f32 {
    _ = seq;
    const u = t + k * dilation;
    if (u >= pad) {
        return input[(u - pad) * channels + c];
    }
    const s = state orelse return 0;
    return s[u * channels + c];
}

/// The dense GEMM (`ops.Gemm`) as one scalar triple loop: the orientation
/// selects the index formulas, the operands widen to the matmul compute
/// dtype (f32 for any mixed pair), and the accumulator narrows once on
/// store.
pub fn gemm(
    pc: ParallelConfig,
    comptime g: ops.Gemm,
    out: *tensor.TensorOf(g.out),
    a: *const tensor.TensorOf(g.a),
    b: *const tensor.TensorOf(g.b),
    m: usize,
    n: usize,
    k: usize,
) void {
    _ = pc;
    const cd = contiguousDataOf(g.out, out, m * n);
    const ad = contiguousDataConstOf(g.a, a, m * k);
    const bd = contiguousDataConstOf(g.b, b, k * n);
    const compute = comptime if (g.a == g.b) dtype_mod.computeDType(.matmul, g.a) else .f32;
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: dtype_mod.Scalar(compute) = 0;
            for (0..k) |p| {
                const av = switch (g.kind) {
                    .plain, .trans_b => ad[i * k + p],
                    .trans_a => ad[p * m + i],
                };
                const bv = switch (g.kind) {
                    .plain, .trans_a => bd[p * n + j],
                    .trans_b => bd[j * k + p],
                };
                acc += dtype_mod.castFloat(g.a, compute, av) * dtype_mod.castFloat(g.b, compute, bv);
            }
            const value = dtype_mod.castFloat(compute, g.out, acc);
            if (g.accumulate) cd[i * n + j] += value else cd[i * n + j] = value;
        }
    }
}

pub fn packDenseRhs(
    comptime dtype: DType,
    allocator: std.mem.Allocator,
    rhs: *const tensor.TensorOf(dtype),
) !packed_matmul.PackedDenseRhs {
    return packed_matmul.packDenseRhs(allocator, dtype, rhs);
}

pub fn quantizeMatmulRhsBlockwiseI8(
    allocator: std.mem.Allocator,
    rhs: *const Tensor,
    group_size: usize,
) !quantized_matmul.QuantizedMatmulRhsI8 {
    return quantized_matmul.quantizeRhsBlockwiseI8(allocator, rhs, group_size);
}

pub fn quantizeMatmulRhsQ4_0(
    allocator: std.mem.Allocator,
    rhs: *const Tensor,
) !quantized_matmul.QuantizedMatmulRhsQ4_0 {
    return quantized_matmul.cold.quantizeMatmulRhsQ4_0(allocator, rhs);
}

pub fn quantizeMatmulRhsQ8_0(
    allocator: std.mem.Allocator,
    rhs: *const Tensor,
) !quantized_matmul.QuantizedMatmulRhsQ8_0 {
    return quantized_matmul.quantizeMatmulRhsQ8_0(allocator, rhs);
}

pub fn matmul2DQuantizedRhs(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: quantized_matmul.AnyQuantizedMatmulRhs,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return switch (rhs) {
        .fucina_w8a8_rhs => |qrhs| matmul2DQuantizedRhsI8(pc, allocator, out, a, qrhs, m, n, k),
        .q1_0 => |qrhs| matmul2DQuantizedRhsQ1_0(pc, allocator, out, a, qrhs, m, n, k),
        .q2_0 => |qrhs| matmul2DQuantizedRhsQ2_0(pc, allocator, out, a, qrhs, m, n, k),
        .q4_0 => |qrhs| matmul2DQuantizedRhsQ4_0(pc, allocator, out, a, qrhs, m, n, k),
        .q4_1 => |qrhs| matmul2DQuantizedRhsQ4_1(pc, allocator, out, a, qrhs, m, n, k),
        .q5_0 => |qrhs| matmul2DQuantizedRhsQ5_0(pc, allocator, out, a, qrhs, m, n, k),
        .q5_1 => |qrhs| matmul2DQuantizedRhsQ5_1(pc, allocator, out, a, qrhs, m, n, k),
        .q8_0 => |qrhs| matmul2DQuantizedRhsQ8_0(pc, allocator, out, a, qrhs, m, n, k),
        inline .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => |qrhs, tag| matmulQuantizedRhs(pc, @field(DType, @tagName(tag)), allocator, out, a, qrhs, m, n, k),
        .iq1_s => |qrhs| matmul2DQuantizedRhsTableQ8_K(pc, .iq1_s, allocator, out, a, qrhs, m, n, k),
        .iq1_m => |qrhs| matmul2DQuantizedRhsTableQ8_K(pc, .iq1_m, allocator, out, a, qrhs, m, n, k),
        .iq2_xxs => |qrhs| matmul2DQuantizedRhsTableQ8_K(pc, .iq2_xxs, allocator, out, a, qrhs, m, n, k),
        .iq2_xs => |qrhs| matmul2DQuantizedRhsTableQ8_K(pc, .iq2_xs, allocator, out, a, qrhs, m, n, k),
        .iq2_s => |qrhs| matmul2DQuantizedRhsTableQ8_K(pc, .iq2_s, allocator, out, a, qrhs, m, n, k),
        .iq3_xxs => |qrhs| matmul2DQuantizedRhsTableQ8_K(pc, .iq3_xxs, allocator, out, a, qrhs, m, n, k),
        .iq3_s => |qrhs| matmul2DQuantizedRhsTableQ8_K(pc, .iq3_s, allocator, out, a, qrhs, m, n, k),
        .iq4_nl => |qrhs| matmul2DQuantizedRhsTableQ8_0(pc, .iq4_nl, allocator, out, a, qrhs, m, n, k),
        .iq4_xs => |qrhs| matmul2DQuantizedRhsTableQ8_K(pc, .iq4_xs, allocator, out, a, qrhs, m, n, k),
        .tq1_0 => |qrhs| matmul2DQuantizedRhsTableQ8_K(pc, .tq1_0, allocator, out, a, qrhs, m, n, k),
        .tq2_0 => |qrhs| matmul2DQuantizedRhsTableQ8_K(pc, .tq2_0, allocator, out, a, qrhs, m, n, k),
        .mxfp4 => |qrhs| matmul2DQuantizedRhsTableQ8_0(pc, .mxfp4, allocator, out, a, qrhs, m, n, k),
        .nvfp4 => |qrhs| matmul2DQuantizedRhsTableQ8_0(pc, .nvfp4, allocator, out, a, qrhs, m, n, k),
    };
}

pub fn matmul2DQuantizedRhsI8(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsI8,
    m: usize,
    n: usize,
    k: usize,
) !void {
    _ = pc; // scalar backend runs the int8 kernel serially
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const a_len = try checkedTensorProduct(m, k);
    const out_len = try checkedTensorProduct(m, n);
    const ad = (try a.dataConstChecked())[0..a_len];
    const cd = (try out.dataChecked())[0..out_len];

    const qa = try allocator.alloc(i8, a_len);
    defer allocator.free(qa);
    const a_scales = try allocator.alloc(f32, m);
    defer allocator.free(a_scales);

    quantized_matmul.quantizeActivationsPerRowI8(qa, a_scales, ad, m, k);
    quantized_matmul.matmulI8BlockwiseRange(cd, qa, a_scales, rhs.qw.dataConst(), rhs.scales.dataConst(), m, n, k, rhs.group_size, rhs.num_groups, 0, m);
}

fn matmul2DQuantizedRhsQ8_0Rows(
    pc: ParallelConfig,
    comptime range: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
) !void {
    _ = pc;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = (try out.dataChecked())[0 .. m * n];
    const blocks_per_row = try quantized_matmul.q8k.q8_0BlockCount(k);
    const block_count = m * blocks_per_row;
    var stack_blocks: [q8_0_lhs_stack_blocks]dtype_mod.BlockQ8_0 = undefined;
    const qlhs_blocks = if (block_count <= stack_blocks.len)
        stack_blocks[0..block_count]
    else
        try allocator.alloc(dtype_mod.BlockQ8_0, block_count);
    defer if (block_count > stack_blocks.len) allocator.free(qlhs_blocks);

    try quantized_matmul.q8k.quantizeRowsQ8_0Into(qlhs_blocks, a);
    range(cd, qlhs_blocks, rhs, m, n, 0, m);
}

fn matmul2DQuantizedRhsQ8_1Rows(
    pc: ParallelConfig,
    comptime range: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
) !void {
    _ = pc;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = (try out.dataChecked())[0 .. m * n];
    var qlhs = try quantized_matmul.cold.quantizeRowsQ8_1(allocator, a);
    defer qlhs.deinit();
    range(cd, qlhs.blocks, rhs, m, n, 0, m);
}

fn matmul2DQuantizedRhsQ8_KRows(
    pc: ParallelConfig,
    comptime range: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
) !void {
    _ = pc;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = (try out.dataChecked())[0 .. m * n];
    const qlhs = try quantized_matmul.q8k.quantizeRowsQ8_K(allocator, a);
    defer allocator.free(qlhs);
    range(cd, qlhs, rhs, m, n, 0, m);
}

pub fn matmul2DQuantizedRhsQ4_0(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_0,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_0Rows(pc, quantized_matmul.cold.matmulQ4_0RhsRange, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ1_0(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ1_0,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_0Rows(pc, quantized_matmul.cold.matmulQ1_0RhsRange, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ2_0(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ2_0,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_0Rows(pc, quantized_matmul.cold.matmulQ2_0RhsRefRange, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ4_1(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_1,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_1Rows(pc, quantized_matmul.cold.matmulQ4_1RhsRange, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ5_0(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_0,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_0Rows(pc, quantized_matmul.cold.matmulQ5_0RhsRange, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ5_1(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_1,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_1Rows(pc, quantized_matmul.cold.matmulQ5_1RhsRange, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ8_0(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_0Rows(pc, quantized_matmul.q8_0.matmulQ8_0RhsRange, allocator, out, a, rhs, m, n, k);
}

/// Scalar reference of the packed-container GEMM entry: same dispatch
/// shape as the native `matmulPacked`, one reference kernel per container.
pub fn matmulPacked(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: anytype,
    a: anytype,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
) !void {
    const Rhs = @TypeOf(rhs.*);
    if (comptime Rhs == packed_matmul.PackedDenseRhs) {
        if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
        return packed_matmul.matmulDenseScalar(contiguousData(out, m * n), contiguousDataConst(a, m * k), rhs, m);
    }
    if (comptime Rhs == quantized_matmul.QuantizedMatmulRhsQ8_0x4)
        return matmul2DQuantizedRhsQ8_0Rows(pc, quantized_matmul.q8_0.matmulQ8_0x4RhsRange, allocator, out, a, rhs, m, n, k);
    if (comptime Rhs == quantized_matmul.QuantizedMatmulRhsQ6_Kx4)
        return matmul2DQuantizedRhsQ8_KRows(pc, quantized_matmul.q6_k.matmulQ6_Kx4RhsRange, allocator, out, a, rhs, m, n, k);
    if (comptime Rhs == quantized_matmul.QuantizedMatmulRhsQ4_Kx4)
        return matmul2DQuantizedRhsQ8_KRows(pc, quantized_matmul.q4_k.matmulQ4_Kx4RhsRange, allocator, out, a, rhs, m, n, k);
    if (comptime Rhs == quantized_matmul.QuantizedMatmulRhsQ4_Kx8)
        return matmul2DQuantizedRhsQ8_KRows(pc, quantized_matmul.q4_k.matmulQ4_Kx8RhsRange, allocator, out, a, rhs, m, n, k);
    if (comptime Rhs == quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla)
        return matmul2DQuantizedRhsQ8_KRows(pc, quantized_matmul.q4_k.matmulQ4_Kx2MmlaRhsRange, allocator, out, a, rhs, m, n, k);
    if (comptime Rhs == quantized_matmul.QuantizedMatmulRhsQ5_Kx8)
        return matmul2DQuantizedRhsQ8_KRows(pc, quantized_matmul.q5_k.matmulQ5_Kx8RhsRange, allocator, out, a, rhs, m, n, k);
    comptime unreachable; // no kernel for this packed RHS container
}

pub fn matmul2DPackedQ8_0x4LhsRhs(
    pc: ParallelConfig,
    out: *Tensor,
    lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) !void {
    _ = pc;
    if (m % 4 != 0) return tensor.TensorError.InvalidShape;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try quantized_matmul.q8k.q8_0BlockCount(k);
    if (lhs_blocks.len != try checkedQuantizedProduct(m / 4, blocks_per_row)) return quantized_matmul.types.QuantizedFormatError.InvalidQuantizedLength;
    quantized_matmul.q8_0.matmulQ8_0x4PackedRhsRange(contiguousData(out, try checkedTensorProduct(m, n)), lhs_blocks, rhs, m, n, 0, m);
}

/// Scalar reference of the pre-quantized-LHS slice entry: same (LHS
/// block, RHS container) comptime dispatch as the native `matmulPackedSlice`.
pub fn matmulPackedSlice(pc: ParallelConfig, out: []f32, lhs_blocks: anytype, rhs: anytype, m: usize, n: usize, k: usize) void {
    _ = k;
    _ = pc;
    const Lhs = @typeInfo(@TypeOf(lhs_blocks)).pointer.child;
    const Rhs = @TypeOf(rhs.*);
    const x4_lhs = Lhs == quantized_matmul.BlockQ8_Kx4;
    comptime if (!x4_lhs and Lhs != dtype_mod.BlockQ8_K)
        @compileError("matmulPackedSlice: unsupported LHS block type " ++ @typeName(Lhs));
    if (comptime Rhs == quantized_matmul.QuantizedMatmulRhsQ4_Kx8) {
        if (comptime x4_lhs)
            quantized_matmul.q4_k.matmulQ4_Kx8Q8_Kx4RhsRange(out, lhs_blocks, rhs, m, n, 0, m)
        else
            quantized_matmul.q4_k.matmulQ4_Kx8RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
    } else if (comptime Rhs == quantized_matmul.QuantizedMatmulRhsQ5_Kx8) {
        if (comptime x4_lhs)
            quantized_matmul.q5_k.matmulQ5_Kx8Q8_Kx4RhsRange(out, lhs_blocks, rhs, m, n, 0, m)
        else
            quantized_matmul.q5_k.matmulQ5_Kx8RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
    } else if (comptime Rhs == quantized_matmul.QuantizedMatmulRhsQ6_Kx4) {
        comptime if (x4_lhs) @compileError("matmulPackedSlice: the Q6_Kx4 pack has no lane-packed LHS kernel");
        quantized_matmul.q6_k.matmulQ6_Kx4RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
    } else {
        comptime unreachable; // no kernel for this packed RHS container
    }
}

pub fn unaryRowSlice(comptime op: ops.UnaryOp, z: []f32, x: []const f32) void {
    for (z, x) |*dst, value| dst.* = ops.unaryScalar(op, value);
}

pub fn mulRowSlice(z: []f32, x: []const f32, y: []const f32) void {
    for (z, x, y) |*dst, a, b| dst.* = a * b;
}

pub fn matmul2DPackedPaddedQ8_0x4LhsRhs(
    pc: ParallelConfig,
    out: *Tensor,
    lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) !void {
    _ = pc;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try quantized_matmul.q8k.q8_0BlockCount(k);
    if (lhs_blocks.len != try checkedQuantizedProduct((m + 3) / 4, blocks_per_row)) return quantized_matmul.types.QuantizedFormatError.InvalidQuantizedLength;
    quantized_matmul.q8_0.matmulQ8_0x4PackedPaddedRhsRange(contiguousData(out, try checkedTensorProduct(m, n)), lhs_blocks, rhs, m, n);
}

/// Scalar reference of the plain K-quant GEMM entry: same shape as the
/// native `matmulQuantizedRhs`, one reference row kernel per dtype.
pub fn matmulQuantizedRhs(
    pc: ParallelConfig,
    comptime dt: DType,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_KRows(pc, switch (dt) {
        .q2_k => quantized_matmul.cold.matmulQ2_KRhsRange,
        .q3_k => quantized_matmul.cold.matmulQ3_KRhsRange,
        .q4_k => quantized_matmul.q4_k.matmulQ4_KRhsRange,
        .q5_k => quantized_matmul.q5_k.matmulQ5_KRhsRange,
        .q6_k => quantized_matmul.q6_k.matmulQ6_KRhsRange,
        else => @compileError("matmulQuantizedRhs serves the plain K-quant containers (q2_k..q6_k)"),
    }, allocator, out, a, rhs, m, n, k);
}

fn matmul2DQuantizedRhsTableQ8_0(
    pc: ParallelConfig,
    comptime rhs_dtype: DType,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsRowsFor(rhs_dtype),
    m: usize,
    n: usize,
    k: usize,
) !void {
    _ = pc;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = (try out.dataChecked())[0 .. m * n];
    var qlhs = try quantized_matmul.q8k.quantizeRowsQ8_0(allocator, a);
    defer qlhs.deinit();
    quantized_matmul.cold.matmulTableQ8_0RhsRange(rhs_dtype, cd, qlhs.blocks, rhs, m, n, 0, m);
}

fn matmul2DQuantizedRhsTableQ8_K(
    pc: ParallelConfig,
    comptime rhs_dtype: DType,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsRowsFor(rhs_dtype),
    m: usize,
    n: usize,
    k: usize,
) !void {
    _ = pc;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = (try out.dataChecked())[0 .. m * n];
    const qlhs = try quantized_matmul.q8k.quantizeRowsQ8_K(allocator, a);
    defer allocator.free(qlhs);
    quantized_matmul.cold.matmulTableQ8_KRhsRange(rhs_dtype, cd, qlhs, rhs, m, n, 0, m);
}

/// Batched dense f32 GEMM over `kind`; strides in elements (0 = shared
/// across batches). The scalar reference always loops.
pub fn gemmBatched(
    pc: ParallelConfig,
    comptime kind: ops.MatmulKind,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
) void {
    _ = pc;
    @constCast(a.buffer).waitReady();
    @constCast(b.buffer).waitReady();
    out.buffer.waitMutable();
    const ap = a.buffer.data[a.offset..].ptr;
    const bp = b.buffer.data[b.offset..].ptr;
    const cp = out.buffer.data[out.offset..].ptr;

    for (0..batch_count) |bi| {
        const ai = ap[bi * stride_a .. bi * stride_a + m * k];
        const bs = bp[bi * stride_b .. bi * stride_b + k * n];
        const ci = cp[bi * stride_c .. bi * stride_c + m * n];
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |p| {
                    const av = switch (kind) {
                        .plain, .trans_b => ai[i * k + p],
                        .trans_a => ai[p * m + i],
                    };
                    const bv = switch (kind) {
                        .plain, .trans_a => bs[p * n + j],
                        .trans_b => bs[j * k + p],
                    };
                    acc += av * bv;
                }
                ci[i * n + j] = acc;
            }
        }
    }
}

fn contiguousDataConst(x: *const Tensor, len: usize) []const f32 {
    @constCast(x.buffer).waitReady();
    return x.buffer.data[x.offset .. x.offset + len];
}

fn contiguousData(x: *Tensor, len: usize) []f32 {
    x.buffer.waitMutable();
    return x.buffer.data[x.offset .. x.offset + len];
}

fn contiguousDataConstOf(comptime dtype: DType, x: *const tensor.TensorOf(dtype), len: usize) []const dtype_mod.Scalar(dtype) {
    @constCast(x.buffer).waitReady();
    return x.buffer.data[x.offset .. x.offset + len];
}

fn contiguousDataOf(comptime dtype: DType, x: *tensor.TensorOf(dtype), len: usize) []dtype_mod.Scalar(dtype) {
    x.buffer.waitMutable();
    return x.buffer.data[x.offset .. x.offset + len];
}

fn elementwiseSlicesTyped(
    comptime dtype: DType,
    comptime op: ops.ElementwiseOp,
    out: []dtype_mod.Scalar(dtype_mod.outputDType(.pointwise, dtype)),
    a: []const dtype_mod.Scalar(dtype),
    b: []const dtype_mod.Scalar(dtype),
) void {
    for (out, a, b) |*dst, av, bv| {
        dst.* = applyElementwiseTyped(dtype, op, av, bv);
    }
}

fn applyElementwiseTyped(
    comptime dtype: DType,
    comptime op: ops.ElementwiseOp,
    a: dtype_mod.Scalar(dtype),
    b: dtype_mod.Scalar(dtype),
) dtype_mod.Scalar(dtype_mod.outputDType(.pointwise, dtype)) {
    const compute_dtype = comptime dtype_mod.computeDType(.pointwise, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.pointwise, dtype);
    const av = dtype_mod.castFloat(dtype, compute_dtype, a);
    const bv = dtype_mod.castFloat(dtype, compute_dtype, b);
    const out = switch (op) {
        .add => av + bv,
        .sub => av - bv,
        .mul => av * bv,
        .div => av / bv,
        .max => if (av != av or bv != bv) std.math.nan(@TypeOf(av)) else @max(av, bv),
        .min => if (av != av or bv != bv) std.math.nan(@TypeOf(av)) else @min(av, bv),
    };
    return dtype_mod.castFloat(compute_dtype, output_dtype, out);
}

fn dotSliceTyped(
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
