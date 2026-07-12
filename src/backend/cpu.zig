const std = @import("std");
const ops = @import("ops.zig");
const packed_matmul = @import("packed.zig");
const quantized_matmul = @import("quant.zig");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");
const vector_conv = @import("vector/conv.zig");
const vector_elementwise = @import("vector/elementwise.zig");
const vector_pool = @import("vector/pool.zig");
const vector_winograd = @import("vector/winograd.zig");

const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;
pub const Conv2dDims = vector_conv.Conv2dDims;
pub const Conv1dDims = vector_conv.Conv1dDims;
pub const PoolKind = vector_pool.PoolKind;
pub const Pool2dDims = vector_pool.Pool2dDims;
// The conv2d backward gather cores are scalar loops (no SIMD divergence), so
// the scalar reference reuses the native config-less serial cores — trivially
// in parity. (Wrappers so this backend's own ParallelConfig type matches the
// dispatch.)
pub fn conv2dBackwardInputIntoWithConfig(out: *Tensor, gy: *const Tensor, weight: *const Tensor, d: Conv2dDims, config: ParallelConfig) void {
    _ = config;
    vector_conv.conv2dBackwardInputInto(out, gy, weight, d);
}
pub fn conv2dBackwardWeightIntoWithConfig(out: *Tensor, input: *const Tensor, gy: *const Tensor, d: Conv2dDims, config: ParallelConfig) void {
    _ = config;
    vector_conv.conv2dBackwardWeightInto(out, input, gy, d);
}
// im2col/col2im and the pool backwards are pure data movement (col2im's row
// adds are elementwise — no reassociation, so the vector core is bit-equal to
// a scalar loop) — the scalar reference reuses them serially (same rationale
// as the conv2d backwards above).
pub fn im2colIntoWithConfig(col: *Tensor, input: *const Tensor, d: Conv2dDims, config: ParallelConfig) void {
    _ = config;
    vector_conv.im2colIntoWithConfig(col, input, d, .{});
}
pub fn col2imIntoWithConfig(out: *Tensor, col: *const Tensor, d: Conv2dDims, config: ParallelConfig) void {
    _ = config;
    vector_conv.col2imIntoWithConfig(out, col, d, .{});
}
// The Winograd F(2×2,3×3)/F(4×4,3×3) transforms are an exec-level ROUTE
// shared by both backends (like im2col): the scalar reference reuses them
// serially so the two backends compute identical values on Winograd-routed
// convs; the GEMMs between the transforms still go through this backend's
// own scalar matmul.
pub const WinogradF2Dims = vector_winograd.F2Dims;
pub fn winogradF2WeightTransformIntoWithConfig(u: *const [16][]f32, w: []const f32, cout: usize, cin: usize, config: ParallelConfig) void {
    _ = config;
    vector_winograd.f2WeightTransformIntoWithConfig(u, w, cout, cin, .{});
}
pub fn winogradF2InputTransformIntoWithConfig(v: *const [16][]f32, x: []const f32, d: WinogradF2Dims, config: ParallelConfig) void {
    _ = config;
    vector_winograd.f2InputTransformIntoWithConfig(v, x, d, .{});
}
pub fn winogradF2OutputTransformIntoWithConfig(y: []f32, m: *const [16][]const f32, bias: ?[]const f32, fuse_relu: bool, d: WinogradF2Dims, config: ParallelConfig) void {
    _ = config;
    vector_winograd.f2OutputTransformIntoWithConfig(y, m, bias, fuse_relu, d, .{});
}
pub fn winogradF4WeightTransformIntoWithConfig(u: *const [36][]f32, w: []const f32, cout: usize, cin: usize, config: ParallelConfig) void {
    _ = config;
    vector_winograd.f4WeightTransformIntoWithConfig(u, w, cout, cin, .{});
}
pub fn winogradF4InputTransformIntoWithConfig(v: *const [36][]f32, x: []const f32, d: WinogradF2Dims, config: ParallelConfig) void {
    _ = config;
    vector_winograd.f4InputTransformIntoWithConfig(v, x, d, .{});
}
pub fn winogradF4OutputTransformIntoWithConfig(y: []f32, m: *const [36][]const f32, bias: ?[]const f32, fuse_relu: bool, d: WinogradF2Dims, config: ParallelConfig) void {
    _ = config;
    vector_winograd.f4OutputTransformIntoWithConfig(y, m, bias, fuse_relu, d, .{});
}
pub fn avgPool2dBackwardIntoWithConfig(out: *Tensor, gy: *const Tensor, d: Pool2dDims, config: ParallelConfig) void {
    _ = config;
    vector_pool.avgPool2dBackwardIntoWithConfig(out, gy, d, .{});
}
pub fn maxPool2dBackwardIntoWithConfig(out: *Tensor, input: *const Tensor, gy: *const Tensor, d: Pool2dDims, config: ParallelConfig) void {
    _ = config;
    vector_pool.maxPool2dBackwardIntoWithConfig(out, input, gy, d, .{});
}
pub fn preluChannelsBackwardInputIntoWithConfig(gx: []f32, gy: []const f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize, config: ParallelConfig) void {
    _ = config;
    for (0..rows) |r| {
        for (0..cols) |c| {
            const i = r * cols + c;
            gx[i] = if (x[i] > 0) gy[i] else gy[i] * alpha[c];
        }
    }
}
pub fn preluChannelsBackwardAlphaIntoWithConfig(galpha: []f32, gy: []const f32, x: []const f32, rows: usize, cols: usize, config: ParallelConfig) void {
    _ = config;
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
pub fn pool2dIntoWithConfig(comptime kind: PoolKind, out: *Tensor, input: *const Tensor, d: Pool2dDims, config: ParallelConfig) void {
    _ = config;
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
pub fn upsample2xNearestIntoWithConfig(out: *Tensor, input: *const Tensor, h: usize, w: usize, c: usize, config: ParallelConfig) void {
    _ = config;
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
pub fn preluChannelsIntoWithConfig(z: []f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize, config: ParallelConfig) void {
    _ = config;
    for (0..rows) |r| {
        for (0..cols) |c| {
            const i = r * cols + c;
            z[i] = if (x[i] > 0) x[i] else x[i] * alpha[c];
        }
    }
}

/// Scalar reference per-channel affine (frozen-stats BatchNorm); a null
/// `shift` degrades to the per-channel scale (the affine's own input-VJP).
pub fn channelAffineIntoWithConfig(z: []f32, x: []const f32, scale: []const f32, shift: ?[]const f32, rows: usize, cols: usize, config: ParallelConfig) void {
    _ = config;
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
    return std.math.mul(usize, a, b) catch quantized_matmul.QuantizedFormatError.InvalidQuantizedLength;
}

pub const ParallelConfig = struct {
    pool: ?*thread.Pool = null,
};

pub fn addInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    try tensor.requireSameShape(a, b);
    try tensor.requireSameShape(out, a);
    addContiguousIntoUnchecked(out, a, b, a.len());
}

pub fn addContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
    addContiguousIntoUncheckedWithConfig(out, a, b, len, .{});
}

pub fn maximumContiguousIntoUncheckedWithConfig(out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize, config: ParallelConfig) void {
    _ = config;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, xv, yv| dst.* = if (xv != xv or yv != yv) std.math.nan(f32) else @max(xv, yv);
}

pub fn minimumContiguousIntoUncheckedWithConfig(out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize, config: ParallelConfig) void {
    _ = config;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, xv, yv| dst.* = if (xv != xv or yv != yv) std.math.nan(f32) else @min(xv, yv);
}

pub fn addContiguousIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, xv, yv| dst.* = xv + yv;
}

pub fn subInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    try tensor.requireSameShape(a, b);
    try tensor.requireSameShape(out, a);
    subContiguousIntoUnchecked(out, a, b, a.len());
}

pub fn subContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
    subContiguousIntoUncheckedWithConfig(out, a, b, len, .{});
}

pub fn subContiguousIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, xv, yv| dst.* = xv - yv;
}

pub fn mulInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    try tensor.requireSameShape(a, b);
    try tensor.requireSameShape(out, a);
    mulContiguousIntoUnchecked(out, a, b, a.len());
}

pub fn mulContiguousIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, len: usize) void {
    mulContiguousIntoUncheckedWithConfig(out, a, b, len, .{});
}

pub fn mulContiguousIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, xv, yv| dst.* = xv * yv;
}

pub fn elementwiseContiguousIntoTypedWithConfig(
    comptime dtype: DType,
    comptime op: ops.ElementwiseOp,
    out: *tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)),
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
    len: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const x = contiguousDataConstOf(dtype, a, len);
    const y = contiguousDataConstOf(dtype, b, len);
    const z = contiguousDataOf(dtype_mod.outputDType(.pointwise, dtype), out, len);
    elementwiseContiguousIntoTyped(dtype, op, z, x, y);
}

pub fn scaleInto(out: *Tensor, a: *const Tensor, scalar_value: f32) !void {
    return scaleIntoWithConfig(out, a, scalar_value, .{});
}

pub fn scaleIntoWithConfig(out: *Tensor, a: *const Tensor, scalar_value: f32, config: ParallelConfig) !void {
    _ = config;
    try tensor.requireSameShape(out, a);
    const x = a.dataConst();
    const z = out.data();
    for (z, x) |*dst, xv| dst.* = xv * scalar_value;
}

pub fn addScaledSlice(z: []f32, x: []const f32, scalar_value: f32) void {
    for (z, x) |*dst, xv| dst.* += xv * scalar_value;
}

pub fn addRowVectorSlice(z: []f32, row_vector: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(z.len >= rows * cols);
    std.debug.assert(row_vector.len == cols);
    for (0..rows) |row_i| {
        const row = z[row_i * cols ..][0..cols];
        for (row, row_vector) |*dst, value| dst.* += value;
    }
}

pub fn addRowVectorUnarySlice(comptime op: ops.UnaryOp, z: []f32, row_vector: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(z.len >= rows * cols);
    std.debug.assert(row_vector.len == cols);
    for (0..rows) |row_i| {
        const row = z[row_i * cols ..][0..cols];
        for (row, row_vector) |*dst, value| dst.* = ops.unaryScalar(op, dst.* + value);
    }
}

pub fn causalDepthwiseConv1dIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    kernel: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    config: ParallelConfig,
) void {
    _ = config;
    causalDepthwiseConv1dRange(out.data(), input.dataConst(), kernel.dataConst(), state, seq, channels, taps, 0, channels);
}

pub fn causalDepthwiseConv1dBackwardInputIntoWithConfig(
    out: *Tensor,
    gy: *const Tensor,
    kernel: *const Tensor,
    seq: usize,
    channels: usize,
    taps: usize,
    config: ParallelConfig,
) void {
    _ = config;
    causalDepthwiseConv1dBackwardInputRange(out.data(), gy.dataConst(), kernel.dataConst(), seq, channels, taps, 0, channels);
}

pub fn causalDepthwiseConv1dBackwardKernelIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    config: ParallelConfig,
) void {
    _ = config;
    causalDepthwiseConv1dBackwardKernelRange(out.data(), input.dataConst(), gy.dataConst(), state, seq, channels, taps, 0, channels);
}

pub fn causalConv1dIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    config: ParallelConfig,
) void {
    _ = config;
    groupedCausalConv1dRange(out.data(), input.dataConst(), weight.dataConst(), state, in_channels, out_channels, taps, dilation, 1, 0, seq);
}

/// Scalar reference conv2d (independent of the native vector kernel — see
/// `Conv2dDims` in vector/conv.zig for the layout). Channel-last [H,W,Cin] ->
/// [OH,OW,Cout] with stride, explicit zero pad, grouped/depthwise.
pub fn conv2dIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    bias: ?[]const f32,
    d: Conv2dDims,
    config: ParallelConfig,
) void {
    _ = config;
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
pub fn conv1dIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    d: Conv1dDims,
    config: ParallelConfig,
) void {
    _ = config;
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
pub fn col2im1dIntoWithConfig(
    out: *Tensor,
    col: *const Tensor,
    t_in: usize,
    out_len: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    config: ParallelConfig,
) void {
    _ = config;
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
pub fn conv1dBackwardInputIntoWithConfig(
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    d: Conv1dDims,
    config: ParallelConfig,
) void {
    _ = config;
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
pub fn conv1dBackwardWeightIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    d: Conv1dDims,
    config: ParallelConfig,
) void {
    _ = config;
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
pub fn col2im1dBackwardIntoWithConfig(
    out: *Tensor,
    gy: *const Tensor,
    t_in: usize,
    gy_len: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    config: ParallelConfig,
) void {
    _ = config;
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

pub fn causalConv1dBackwardInputIntoWithConfig(
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    config: ParallelConfig,
) void {
    _ = config;
    groupedCausalConv1dBackwardInputRange(out.data(), gy.dataConst(), weight.dataConst(), seq, in_channels, out_channels, taps, dilation, 1, 0, seq);
}

pub fn causalConv1dBackwardWeightIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    config: ParallelConfig,
) void {
    _ = config;
    groupedCausalConv1dBackwardWeightRange(out.data(), input.dataConst(), gy.dataConst(), state, seq, in_channels, out_channels, taps, dilation, 1, 0, taps * in_channels);
}

pub fn groupedCausalConv1dIntoWithConfig(
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
    config: ParallelConfig,
) void {
    _ = config;
    groupedCausalConv1dRange(out.data(), input.dataConst(), weight.dataConst(), state, in_channels, out_channels, taps, dilation, groups, 0, seq);
}

pub fn groupedCausalConv1dBackwardInputIntoWithConfig(
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    config: ParallelConfig,
) void {
    _ = config;
    groupedCausalConv1dBackwardInputRange(out.data(), gy.dataConst(), weight.dataConst(), seq, in_channels, out_channels, taps, dilation, groups, 0, seq);
}

pub fn groupedCausalConv1dBackwardWeightIntoWithConfig(
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
    config: ParallelConfig,
) void {
    _ = config;
    const in_per_group = in_channels / groups;
    groupedCausalConv1dBackwardWeightRange(out.data(), input.dataConst(), gy.dataConst(), state, seq, in_channels, out_channels, taps, dilation, groups, 0, taps * in_per_group);
}

pub fn unaryContiguousIntoUnchecked(
    comptime op: ops.UnaryOp,
    out: *Tensor,
    a: *const Tensor,
    len: usize,
) void {
    unaryContiguousIntoUncheckedWithConfig(op, out, a, len, .{});
}

pub fn unaryContiguousIntoUncheckedWithConfig(
    comptime op: ops.UnaryOp,
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    for (z, x) |*dst, value| dst.* = ops.unaryScalar(op, value);
}

/// Scalar reference snake activation (see the vector kernel's doc comment):
/// `y[t,c] = x[t,c] + inv_b[c] * sin(alpha[c]*x[t,c])^2` over contiguous
/// `[rows, cols]` rows; `inv_b` is precomputed by the caller.
pub fn snakeIntoWithConfig(
    out: *Tensor,
    x: *const Tensor,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
    config: ParallelConfig,
) void {
    _ = config;
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
pub fn groupNormIntoWithConfig(
    out: *Tensor,
    x: *const Tensor,
    weight: ?[]const f32,
    bias: ?[]const f32,
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
    config: ParallelConfig,
) void {
    _ = config;
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
pub fn snakeBackwardInputIntoWithConfig(
    out: *Tensor,
    x: *const Tensor,
    gy: *const Tensor,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
    config: ParallelConfig,
) void {
    _ = config;
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
pub fn snakeBackwardParamsIntoWithConfig(
    galpha: *Tensor,
    ginv_b: *Tensor,
    x: *const Tensor,
    gy: *const Tensor,
    alpha: []const f32,
    inv_b: []const f32,
    rows: usize,
    cols: usize,
    config: ParallelConfig,
) void {
    _ = config;
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
pub fn groupNormBackwardIntoWithConfig(
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
    config: ParallelConfig,
) void {
    _ = config;
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
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    negative_slope: f32,
) void {
    leakyReluContiguousIntoUncheckedWithConfig(out, a, len, negative_slope, .{});
}

pub fn leakyReluContiguousIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    negative_slope: f32,
    config: ParallelConfig,
) void {
    _ = config;
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    for (z, x) |*dst, value| dst.* = if (value >= 0) value else value * negative_slope;
}

pub fn clampContiguousIntoUnchecked(
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    min_value: f32,
    max_value: f32,
) void {
    clampContiguousIntoUncheckedWithConfig(out, a, len, min_value, max_value, .{});
}

pub fn clampContiguousIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    len: usize,
    min_value: f32,
    max_value: f32,
    config: ParallelConfig,
) void {
    _ = config;
    const x = contiguousDataConst(a, len);
    const z = contiguousData(out, len);
    for (z, x) |*dst, value| dst.* = @min(@max(value, min_value), max_value);
}

pub fn gatedContiguousIntoUnchecked(
    comptime op: ops.GatedOp,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
) void {
    gatedContiguousIntoUncheckedWithConfig(op, out, a, b, len, .{});
}

pub fn gatedContiguousIntoUncheckedWithConfig(
    comptime op: ops.GatedOp,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    len: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const x = contiguousDataConst(a, len);
    const y = contiguousDataConst(b, len);
    const z = contiguousData(out, len);
    for (z, x, y) |*dst, left, gate| dst.* = left * ops.gatedActivationScalar(op, gate);
}

pub fn sumInto(out: *Tensor, a: *const Tensor) !void {
    return sumIntoWithConfig(out, a, .{});
}

pub fn sumIntoWithConfig(out: *Tensor, a: *const Tensor, config: ParallelConfig) !void {
    _ = config;
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    out.data()[0] = sumSlice(a.dataConst());
}

pub fn sumSlice(values: []const f32) f32 {
    var s: f32 = 0;
    for (values) |v| s += v;
    return s;
}

pub fn prodInto(out: *Tensor, a: *const Tensor) !void {
    return prodIntoWithConfig(out, a, .{});
}

pub fn prodIntoWithConfig(out: *Tensor, a: *const Tensor, config: ParallelConfig) !void {
    _ = config;
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    out.data()[0] = prodSlice(a.dataConst());
}

pub fn prodSlice(values: []const f32) f32 {
    var p: f32 = 1;
    for (values) |v| p *= v;
    return p;
}

pub fn sumSliceTypedWithConfig(
    comptime dtype: DType,
    values: []const dtype_mod.Scalar(dtype),
    config: ParallelConfig,
) dtype_mod.Scalar(dtype_mod.outputDType(.reduction, dtype)) {
    _ = config;
    const compute_dtype = comptime dtype_mod.computeDType(.reduction, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.reduction, dtype);
    var acc: dtype_mod.Scalar(compute_dtype) = 0;
    for (values) |value| {
        acc += dtype_mod.castFloat(dtype, compute_dtype, value);
    }
    return dtype_mod.castFloat(compute_dtype, output_dtype, acc);
}

pub fn dotInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    return dotIntoWithConfig(out, a, b, .{});
}

pub fn dotIntoWithConfig(out: *Tensor, a: *const Tensor, b: *const Tensor, config: ParallelConfig) !void {
    _ = config;
    try tensor.requireSameShape(a, b);
    if (!out.isScalar()) return tensor.TensorError.ShapeMismatch;
    var s: f32 = 0;
    for (a.dataConst(), b.dataConst()) |x, y| s += x * y;
    out.data()[0] = s;
}

pub fn dotIntoTypedWithConfig(
    comptime dtype: DType,
    out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
    config: ParallelConfig,
) !void {
    _ = config;
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
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = taps - 1;
    for (0..seq) |t| {
        for (channel_start..channel_end) |c| {
            var acc: f32 = 0;
            for (0..taps) |k| {
                acc += causalDepthwiseInputValue(input, state, seq, channels, pad, t, c, k) * kernel[c * taps + k];
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
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = taps - 1;
    for (0..seq) |p| {
        for (channel_start..channel_end) |c| {
            var acc: f32 = 0;
            for (0..taps) |k| {
                const t_base = p + pad;
                if (k > t_base) continue;
                const t = t_base - k;
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
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = taps - 1;
    for (channel_start..channel_end) |c| {
        for (0..taps) |k| {
            var acc: f32 = 0;
            for (0..seq) |t| {
                acc += gy[t * channels + c] * causalDepthwiseInputValue(input, state, seq, channels, pad, t, c, k);
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
    t: usize,
    c: usize,
    k: usize,
) f32 {
    _ = seq;
    if (t + k >= pad) {
        const pos = t + k - pad;
        return input[pos * channels + c];
    }
    const s = state orelse return 0;
    const sidx = t + k;
    return s[sidx * channels + c];
}

pub fn matmulInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    const av = try a.rankView(2);
    const bv = try b.rankView(2);
    const ov = try out.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    const n = bv.dim(1);
    if (k != bv.dim(0)) return tensor.TensorError.ShapeMismatch;
    if (ov.dim(0) != m or ov.dim(1) != n) return tensor.TensorError.ShapeMismatch;

    matmul2DIntoUnchecked(out, a, b, m, n, k);
}

pub fn matmul2DIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, m: usize, n: usize, k: usize) void {
    matmul2DIntoUncheckedWithConfig(out, a, b, m, n, k, .{});
}

pub fn matmul2DIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const ad = contiguousDataConst(a, m * k);
    const bd = contiguousDataConst(b, k * n);
    const cd = contiguousData(out, m * n);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0;
            for (0..k) |p| {
                acc += ad[i * k + p] * bd[p * n + j];
            }
            cd[i * n + j] = acc;
        }
    }
}

pub fn matmul2DIntoUncheckedTypedWithConfig(
    comptime dtype: DType,
    out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    _ = config;
    matmul2DIntoTyped(
        dtype,
        contiguousDataOf(dtype_mod.outputDType(.matmul, dtype), out, m * n),
        contiguousDataConstOf(dtype, a, m * k),
        contiguousDataConstOf(dtype, b, k * n),
        m,
        n,
        k,
    );
}

pub fn packMatmulRhsTyped(
    comptime dtype: DType,
    allocator: std.mem.Allocator,
    rhs: *const tensor.TensorOf(dtype),
) !packed_matmul.PackedMatmulRhsFor(dtype) {
    return packed_matmul.packRhs(allocator, dtype, rhs);
}

pub fn matmul2DIntoUncheckedPackedRhsTypedWithConfig(
    comptime dtype: DType,
    allocator: std.mem.Allocator,
    out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
    a: *const tensor.TensorOf(dtype),
    rhs: *const packed_matmul.PackedMatmulRhsFor(dtype),
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return packed_matmul.matmul2DIntoUncheckedPackedRhsTypedWithConfig(
        allocator,
        dtype,
        out,
        a,
        rhs,
        m,
        n,
        k,
        config,
        matmul2DIntoUncheckedWithConfig,
    );
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
    return quantized_matmul.quantizeMatmulRhsQ4_0(allocator, rhs);
}

pub fn quantizeMatmulRhsQ8_0(
    allocator: std.mem.Allocator,
    rhs: *const Tensor,
) !quantized_matmul.QuantizedMatmulRhsQ8_0 {
    return quantized_matmul.quantizeMatmulRhsQ8_0(allocator, rhs);
}

pub fn matmul2DQuantizedRhsWithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: quantized_matmul.AnyQuantizedMatmulRhs,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return switch (rhs) {
        .fucina_w8a8_rhs => |qrhs| matmul2DQuantizedRhsI8WithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_q1_0 => |qrhs| matmul2DQuantizedRhsQ1_0WithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_q4_0 => |qrhs| matmul2DQuantizedRhsQ4_0WithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_q4_1 => |qrhs| matmul2DQuantizedRhsQ4_1WithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_q5_0 => |qrhs| matmul2DQuantizedRhsQ5_0WithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_q5_1 => |qrhs| matmul2DQuantizedRhsQ5_1WithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_q8_0 => |qrhs| matmul2DQuantizedRhsQ8_0WithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_q2_k => |qrhs| matmul2DQuantizedRhsQ2_KWithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_q3_k => |qrhs| matmul2DQuantizedRhsQ3_KWithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_q4_k => |qrhs| matmul2DQuantizedRhsQ4_KWithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_q5_k => |qrhs| matmul2DQuantizedRhsQ5_KWithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_q6_k => |qrhs| matmul2DQuantizedRhsQ6_KWithConfig(allocator, out, a, qrhs, m, n, k, config),
        .ggml_iq1_s => |qrhs| matmul2DQuantizedRhsTableQ8_KWithConfig(.iq1_s, allocator, out, a, qrhs, m, n, k, config),
        .ggml_iq1_m => |qrhs| matmul2DQuantizedRhsTableQ8_KWithConfig(.iq1_m, allocator, out, a, qrhs, m, n, k, config),
        .ggml_iq2_xxs => |qrhs| matmul2DQuantizedRhsTableQ8_KWithConfig(.iq2_xxs, allocator, out, a, qrhs, m, n, k, config),
        .ggml_iq2_xs => |qrhs| matmul2DQuantizedRhsTableQ8_KWithConfig(.iq2_xs, allocator, out, a, qrhs, m, n, k, config),
        .ggml_iq2_s => |qrhs| matmul2DQuantizedRhsTableQ8_KWithConfig(.iq2_s, allocator, out, a, qrhs, m, n, k, config),
        .ggml_iq3_xxs => |qrhs| matmul2DQuantizedRhsTableQ8_KWithConfig(.iq3_xxs, allocator, out, a, qrhs, m, n, k, config),
        .ggml_iq3_s => |qrhs| matmul2DQuantizedRhsTableQ8_KWithConfig(.iq3_s, allocator, out, a, qrhs, m, n, k, config),
        .ggml_iq4_nl => |qrhs| matmul2DQuantizedRhsTableQ8_0WithConfig(.iq4_nl, allocator, out, a, qrhs, m, n, k, config),
        .ggml_iq4_xs => |qrhs| matmul2DQuantizedRhsTableQ8_KWithConfig(.iq4_xs, allocator, out, a, qrhs, m, n, k, config),
        .ggml_tq1_0 => |qrhs| matmul2DQuantizedRhsTableQ8_KWithConfig(.tq1_0, allocator, out, a, qrhs, m, n, k, config),
        .ggml_tq2_0 => |qrhs| matmul2DQuantizedRhsTableQ8_KWithConfig(.tq2_0, allocator, out, a, qrhs, m, n, k, config),
        .ggml_mxfp4 => |qrhs| matmul2DQuantizedRhsTableQ8_0WithConfig(.mxfp4, allocator, out, a, qrhs, m, n, k, config),
        .ggml_nvfp4 => |qrhs| matmul2DQuantizedRhsTableQ8_0WithConfig(.nvfp4, allocator, out, a, qrhs, m, n, k, config),
    };
}

pub fn matmul2DQuantizedRhsI8WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsI8,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    _ = config; // scalar backend runs the int8 kernel serially
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

fn matmul2DQuantizedRhsQ8_0RowsWithConfig(
    comptime range: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    _ = config;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = (try out.dataChecked())[0 .. m * n];
    const blocks_per_row = try quantized_matmul.q8_0BlockCount(k);
    const block_count = m * blocks_per_row;
    var stack_blocks: [q8_0_lhs_stack_blocks]quantized_matmul.BlockQ8_0 = undefined;
    const qlhs_blocks = if (block_count <= stack_blocks.len)
        stack_blocks[0..block_count]
    else
        try allocator.alloc(quantized_matmul.BlockQ8_0, block_count);
    defer if (block_count > stack_blocks.len) allocator.free(qlhs_blocks);

    try quantized_matmul.quantizeRowsQ8_0Into(qlhs_blocks, a);
    range(cd, qlhs_blocks, rhs, m, n, 0, m);
}

fn matmul2DQuantizedRhsQ8_1RowsWithConfig(
    comptime range: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    _ = config;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = (try out.dataChecked())[0 .. m * n];
    var qlhs = try quantized_matmul.quantizeRowsQ8_1(allocator, a);
    defer qlhs.deinit();
    range(cd, qlhs.blocks, rhs, m, n, 0, m);
}

fn matmul2DQuantizedRhsQ8_KRowsWithConfig(
    comptime range: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    _ = config;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = (try out.dataChecked())[0 .. m * n];
    const qlhs = try quantized_matmul.quantizeRowsQ8_K(allocator, a);
    defer allocator.free(qlhs);
    range(cd, qlhs, rhs, m, n, 0, m);
}

pub fn matmul2DQuantizedRhsQ4_0WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_0RowsWithConfig(quantized_matmul.matmulQ4_0RhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ1_0WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ1_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_0RowsWithConfig(quantized_matmul.matmulQ1_0RhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ4_1WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_1,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_1RowsWithConfig(quantized_matmul.matmulQ4_1RhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ5_0WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_0RowsWithConfig(quantized_matmul.matmulQ5_0RhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ5_1WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_1,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_1RowsWithConfig(quantized_matmul.matmulQ5_1RhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ8_0WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_0RowsWithConfig(quantized_matmul.matmulQ8_0RhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ8_0x4WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_0RowsWithConfig(quantized_matmul.matmulQ8_0x4RhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DPackedQ8_0x4LhsRhsWithConfig(
    out: *Tensor,
    lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    _ = config;
    if (m % 4 != 0) return tensor.TensorError.InvalidShape;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try quantized_matmul.q8_0BlockCount(k);
    if (lhs_blocks.len != try checkedQuantizedProduct(m / 4, blocks_per_row)) return quantized_matmul.QuantizedFormatError.InvalidQuantizedLength;
    quantized_matmul.matmulQ8_0x4PackedRhsRange(contiguousData(out, try checkedTensorProduct(m, n)), lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmulPackedQ4_Kx8Q8_Kx4SliceWithConfig(out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_Kx4, rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    _ = k;
    _ = config;
    quantized_matmul.matmulQ4_Kx8Q8_Kx4RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmulPackedQ4_Kx8RowsSliceWithConfig(out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    _ = k;
    _ = config;
    quantized_matmul.matmulQ4_Kx8RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmulPackedQ5_Kx8Q8_Kx4SliceWithConfig(out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_Kx4, rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    _ = k;
    _ = config;
    quantized_matmul.matmulQ5_Kx8Q8_Kx4RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmulPackedQ5_Kx8RowsSliceWithConfig(out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    _ = k;
    _ = config;
    quantized_matmul.matmulQ5_Kx8RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmulPackedQ6_Kx4RowsSliceWithConfig(out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_Kx4, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    _ = k;
    _ = config;
    quantized_matmul.matmulQ6_Kx4RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn unaryRowSlice(comptime op: ops.UnaryOp, z: []f32, x: []const f32) void {
    for (z, x) |*dst, value| dst.* = ops.unaryScalar(op, value);
}

pub fn mulRowSlice(z: []f32, x: []const f32, y: []const f32) void {
    for (z, x, y) |*dst, a, b| dst.* = a * b;
}

pub fn matmul2DPackedPaddedQ8_0x4LhsRhsWithConfig(
    out: *Tensor,
    lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    _ = config;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try quantized_matmul.q8_0BlockCount(k);
    if (lhs_blocks.len != try checkedQuantizedProduct((m + 3) / 4, blocks_per_row)) return quantized_matmul.QuantizedFormatError.InvalidQuantizedLength;
    quantized_matmul.matmulQ8_0x4PackedPaddedRhsRange(contiguousData(out, try checkedTensorProduct(m, n)), lhs_blocks, rhs, m, n);
}

pub fn matmul2DQuantizedRhsQ2_KWithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ2_K,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(quantized_matmul.matmulQ2_KRhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ3_KWithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ3_K,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(quantized_matmul.matmulQ3_KRhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ4_KWithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_K,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(quantized_matmul.matmulQ4_KRhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ4_Kx4WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx4,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(quantized_matmul.matmulQ4_Kx4RhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ4_Kx8WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(quantized_matmul.matmulQ4_Kx8RhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ4_Kx2MmlaWithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(quantized_matmul.matmulQ4_Kx2MmlaRhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ5_Kx8WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(quantized_matmul.matmulQ5_Kx8RhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ5_KWithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_K,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(quantized_matmul.matmulQ5_KRhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ6_KWithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_K,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(quantized_matmul.matmulQ6_KRhsRange, allocator, out, a, rhs, m, n, k, config);
}

pub fn matmul2DQuantizedRhsQ6_Kx4WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_Kx4,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(quantized_matmul.matmulQ6_Kx4RhsRange, allocator, out, a, rhs, m, n, k, config);
}

fn matmul2DQuantizedRhsTableQ8_0WithConfig(
    comptime rhs_dtype: DType,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsRowsFor(rhs_dtype),
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    _ = config;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = (try out.dataChecked())[0 .. m * n];
    var qlhs = try quantized_matmul.quantizeRowsQ8_0(allocator, a);
    defer qlhs.deinit();
    quantized_matmul.matmulTableQ8_0RhsRange(rhs_dtype, cd, qlhs.blocks, rhs, m, n, 0, m);
}

fn matmul2DQuantizedRhsTableQ8_KWithConfig(
    comptime rhs_dtype: DType,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsRowsFor(rhs_dtype),
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    _ = config;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = (try out.dataChecked())[0 .. m * n];
    const qlhs = try quantized_matmul.quantizeRowsQ8_K(allocator, a);
    defer allocator.free(qlhs);
    quantized_matmul.matmulTableQ8_KRhsRange(rhs_dtype, cd, qlhs, rhs, m, n, 0, m);
}

pub fn matmulTransAInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    const av = try a.rankView(2);
    const bv = try b.rankView(2);
    const ov = try out.rankView(2);
    const k = av.dim(0);
    const m = av.dim(1);
    const n = bv.dim(1);
    if (k != bv.dim(0)) return tensor.TensorError.ShapeMismatch;
    if (ov.dim(0) != m or ov.dim(1) != n) return tensor.TensorError.ShapeMismatch;

    matmulTransA2DIntoUnchecked(out, a, b, m, n, k);
}

pub fn matmulTransA2DIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, m: usize, n: usize, k: usize) void {
    matmulTransA2DIntoUncheckedWithConfig(out, a, b, m, n, k, .{});
}

pub fn matmulTransA2DIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const ad = contiguousDataConst(a, k * m);
    const bd = contiguousDataConst(b, k * n);
    const cd = contiguousData(out, m * n);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0;
            for (0..k) |p| {
                acc += ad[p * m + i] * bd[p * n + j];
            }
            cd[i * n + j] = acc;
        }
    }
}

pub fn matmulTransBInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    const av = try a.rankView(2);
    const bv = try b.rankView(2);
    const ov = try out.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    const n = bv.dim(0);
    if (k != bv.dim(1)) return tensor.TensorError.ShapeMismatch;
    if (ov.dim(0) != m or ov.dim(1) != n) return tensor.TensorError.ShapeMismatch;

    matmulTransB2DIntoUnchecked(out, a, b, m, n, k);
}

pub fn matmulTransB2DIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, m: usize, n: usize, k: usize) void {
    matmulTransB2DIntoUncheckedWithConfig(out, a, b, m, n, k, .{});
}

pub fn matmulTransB2DIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const ad = contiguousDataConst(a, m * k);
    const bd = contiguousDataConst(b, n * k);
    const cd = contiguousData(out, m * n);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0;
            for (0..k) |p| {
                acc += ad[i * k + p] * bd[j * k + p];
            }
            cd[i * n + j] = acc;
        }
    }
}

pub fn matmulTransB2DIntoUncheckedF16OperandsWithConfig(
    out: *Tensor,
    a: *const tensor.TensorOf(.f16),
    b: *const tensor.TensorOf(.f16),
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const ad = contiguousDataConstOf(.f16, a, m * k);
    const bd = contiguousDataConstOf(.f16, b, n * k);
    const cd = contiguousData(out, m * n);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0;
            for (0..k) |p| {
                acc += @as(f32, @floatCast(ad[i * k + p])) * @as(f32, @floatCast(bd[j * k + p]));
            }
            cd[i * n + j] = acc;
        }
    }
}

pub fn matmulTransB2DIntoUncheckedBf16RhsWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const tensor.TensorOf(.bf16),
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const ad = contiguousDataConst(a, m * k);
    const bd = contiguousDataConstOf(.bf16, b, n * k);
    const cd = contiguousData(out, m * n);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0;
            for (0..k) |p| {
                acc += ad[i * k + p] * dtype_mod.bf16ToF32(bd[j * k + p]);
            }
            cd[i * n + j] = acc;
        }
    }
}

// Batched GEMM. Strides are in elements (a value of 0 means "shared across
// all batches", which makes broadcast-RHS just another stride value).
// CPU backend always loops; the SIMD backend may dispatch to a single
// Accelerate call when available.
pub fn matmulBatched2DIntoUnchecked(
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
    matmulBatched2DIntoUncheckedWithConfig(out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, .{});
}

pub fn matmulBatched2DIntoUncheckedWithConfig(
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
    config: ParallelConfig,
) void {
    _ = config;
    @constCast(a.buffer).waitReady();
    @constCast(b.buffer).waitReady();
    out.buffer.waitMutable();
    const ap = a.buffer.data[a.offset..].ptr;
    const bp = b.buffer.data[b.offset..].ptr;
    const cp = out.buffer.data[out.offset..].ptr;

    for (0..batch_count) |bi| {
        const ai = ap[bi * stride_a .. bi * stride_a + m * k];
        const bi_slice = bp[bi * stride_b .. bi * stride_b + k * n];
        const ci = cp[bi * stride_c .. bi * stride_c + m * n];
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |p| acc += ai[i * k + p] * bi_slice[p * n + j];
                ci[i * n + j] = acc;
            }
        }
    }
}

pub fn matmulBatchedTransA2DIntoUnchecked(
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
    matmulBatchedTransA2DIntoUncheckedWithConfig(out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, .{});
}

pub fn matmulBatchedTransA2DIntoUncheckedWithConfig(
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
    config: ParallelConfig,
) void {
    _ = config;
    @constCast(a.buffer).waitReady();
    @constCast(b.buffer).waitReady();
    out.buffer.waitMutable();
    const ap = a.buffer.data[a.offset..].ptr;
    const bp = b.buffer.data[b.offset..].ptr;
    const cp = out.buffer.data[out.offset..].ptr;

    for (0..batch_count) |bi| {
        const ai = ap[bi * stride_a .. bi * stride_a + k * m];
        const bi_slice = bp[bi * stride_b .. bi * stride_b + k * n];
        const ci = cp[bi * stride_c .. bi * stride_c + m * n];
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |p| acc += ai[p * m + i] * bi_slice[p * n + j];
                ci[i * n + j] = acc;
            }
        }
    }
}

pub fn matmulBatchedTransB2DIntoUnchecked(
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
    matmulBatchedTransB2DIntoUncheckedWithConfig(out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, .{});
}

pub fn matmulBatchedTransB2DIntoUncheckedWithConfig(
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
    config: ParallelConfig,
) void {
    _ = config;
    @constCast(a.buffer).waitReady();
    @constCast(b.buffer).waitReady();
    out.buffer.waitMutable();
    const ap = a.buffer.data[a.offset..].ptr;
    const bp = b.buffer.data[b.offset..].ptr;
    const cp = out.buffer.data[out.offset..].ptr;

    for (0..batch_count) |bi| {
        const ai = ap[bi * stride_a .. bi * stride_a + m * k];
        const bi_slice = bp[bi * stride_b .. bi * stride_b + n * k];
        const ci = cp[bi * stride_c .. bi * stride_c + m * n];
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |p| acc += ai[i * k + p] * bi_slice[j * k + p];
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

fn elementwiseContiguousIntoTyped(
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

fn matmul2DIntoTyped(
    comptime dtype: DType,
    out: []dtype_mod.Scalar(dtype_mod.outputDType(.matmul, dtype)),
    a: []const dtype_mod.Scalar(dtype),
    b: []const dtype_mod.Scalar(dtype),
    m: usize,
    n: usize,
    k: usize,
) void {
    for (0..m) |i| {
        for (0..n) |j| {
            out[i * n + j] = dotColumnTyped(dtype, a[i * k ..][0..k], b, j, n);
        }
    }
}

fn dotColumnTyped(
    comptime dtype: DType,
    a_row: []const dtype_mod.Scalar(dtype),
    b: []const dtype_mod.Scalar(dtype),
    col: usize,
    n: usize,
) dtype_mod.Scalar(dtype_mod.outputDType(.matmul, dtype)) {
    const compute_dtype = comptime dtype_mod.computeDType(.matmul, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.matmul, dtype);
    var acc: dtype_mod.Scalar(compute_dtype) = 0;
    for (0..a_row.len) |p| {
        acc += dtype_mod.castFloat(dtype, compute_dtype, a_row[p]) * dtype_mod.castFloat(dtype, compute_dtype, b[p * n + col]);
    }
    return dtype_mod.castFloat(compute_dtype, output_dtype, acc);
}
