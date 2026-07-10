//! Dtype conversion ops for the eager runtime.
//!
//! `castTyped` is the general float-to-float cast; the two vectorized cast
//! kernels it uses (`castF32ToF16`/`castF16ToF32`) are also the primitives the
//! KV-cache-append helpers in `exec.zig` reuse, so they live here as `pub`.
//!
//! Domain module: receives an explicit `*Runtime` (never `self: anytype`).
//! Imports the `runtime` leaf (`prepareContiguousTyped` lives on `Runtime`);
//! never imports `exec.zig`.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");

const exec_elementwise = @import("elementwise.zig");
const Runtime = @import("runtime.zig").Runtime;

const DType = tensor.DType;
const BlockQ8_0 = dtype_mod.BlockQ8_0;
const q8_0_block_size = dtype_mod.q8_0_block_size;

fn ensureForwardFloatMath(comptime dtype: DType) void {
    if (!dtype_mod.supportsForwardFloatMath(dtype)) {
        @compileError("forward math is currently supported only for floating dtypes");
    }
}

pub fn castTyped(
    rt: *Runtime,
    comptime source_dtype: DType,
    comptime target_dtype: DType,
    x: *const tensor.TensorOf(source_dtype),
) !tensor.TensorOf(target_dtype) {
    if (comptime source_dtype == target_dtype) return rt.cloneTyped(source_dtype, x);
    if (comptime dtype_mod.isBlockQuantized(source_dtype) or dtype_mod.isBlockQuantized(target_dtype)) {
        @compileError("casts are supported between the scalar dtypes only (dequantize with to(.f32))");
    }

    var xx = try rt.prepareContiguousTyped(source_dtype, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyTyped(target_dtype, x.shape.slice());
    errdefer out.deinit();
    const output = out.data();
    if (comptime source_dtype == .f32 and target_dtype == .f16) {
        castF32ToF16(output, input);
        return out;
    }
    if (comptime source_dtype == .f16 and target_dtype == .f32) {
        castF16ToF32(output, input);
        return out;
    }
    if (comptime source_dtype == .f32 and target_dtype == .bf16) {
        castF32ToBf16(output, input);
        return out;
    }
    if (comptime source_dtype == .bf16 and target_dtype == .f32) {
        castBf16ToF32(output, input);
        return out;
    }
    for (output, input) |*dst, value| {
        dst.* = dtype_mod.castScalar(source_dtype, target_dtype, value);
    }
    return out;
}

/// Vector twin of `dtype.f32ToBf16` — bit-identical lanes: round-to-nearest-
/// even via the (bits + 0x7fff + lsb) trick, NaN quieted with bit 6 set.
pub fn castF32ToBf16(output: []u16, input: []const f32) void {
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

/// Cast an f32 tensor into a caller-owned f16 slice in logical row-major
/// order without allocating: the KV-cache append path. Supports contiguous
/// sources (one SIMD pass) and rank-3 views whose two inner axes are
/// contiguous (a `{seq, kv_head, d}` split of a fused QKV row), walked as
/// per-row spans. Anything else is UnsupportedView — extend deliberately
/// rather than silently gathering.
pub fn castF32RowsToF16Into(rt: *Runtime, x: *const tensor.Tensor, dst: []f16) !void {
    _ = rt;
    if (dst.len != x.len()) return tensor.TensorError.InvalidDataLength;
    @constCast(x.buffer).waitReady();
    const data = x.buffer.data;
    if (x.isContiguous()) {
        castF32ToF16(dst, data[x.offset..][0..dst.len]);
        return;
    }
    if (x.shape.len == 3 and x.strides.at(2) == 1 and x.strides.at(1) == x.shape.at(2)) {
        const rows = x.shape.at(0);
        const row = x.shape.at(1) * x.shape.at(2);
        const row_stride = x.strides.at(0);
        for (0..rows) |i| {
            castF32ToF16(dst[i * row ..][0..row], data[x.offset + i * row_stride ..][0..row]);
        }
        return;
    }
    return tensor.TensorError.UnsupportedView;
}

/// Quantize an f32 tensor into caller-owned q8_0 blocks in logical
/// row-major order without allocating: the q8_0 KV-cache append path.
/// Same supported views as `castF32RowsToF16Into` (contiguous, or a
/// rank-3 `{seq, kv_head, d}` split of a fused QKV row). Row length must
/// be a multiple of 32 so block boundaries never straddle rows.
pub fn quantizeF32RowsToQ8_0Into(rt: *Runtime, x: *const tensor.Tensor, dst: []BlockQ8_0) !void {
    _ = rt;
    if (x.len() % q8_0_block_size != 0) return tensor.TensorError.InvalidDataLength;
    if (dst.len != x.len() / q8_0_block_size) return tensor.TensorError.InvalidDataLength;
    @constCast(x.buffer).waitReady();
    const data = x.buffer.data;
    if (x.isContiguous()) {
        try backend_mod.quantized_matmul.quantizeRowQ8_0Into(dst, data[x.offset..][0..x.len()]);
        return;
    }
    if (x.shape.len == 3 and x.strides.at(2) == 1 and x.strides.at(1) == x.shape.at(2)) {
        const rows = x.shape.at(0);
        const row = x.shape.at(1) * x.shape.at(2);
        if (row % q8_0_block_size != 0) return tensor.TensorError.InvalidDataLength;
        const row_blocks = row / q8_0_block_size;
        const row_stride = x.strides.at(0);
        for (0..rows) |i| {
            try backend_mod.quantized_matmul.quantizeRowQ8_0Into(
                dst[i * row_blocks ..][0..row_blocks],
                data[x.offset + i * row_stride ..][0..row],
            );
        }
        return;
    }
    return tensor.TensorError.UnsupportedView;
}

/// Dequantize q8_0 blocks into f32 (`dst.len == blocks.len * 32`): the
/// inverse of `quantizeF32RowsToQ8_0Into`, for round-trip checks and the
/// q8_0-KV gradient fallback.
pub fn dequantizeQ8_0RowsInto(rt: *Runtime, dst: []f32, blocks: []const BlockQ8_0) !void {
    _ = rt;
    try backend_mod.quantized_matmul.dequantizeRowQ8_0Into(dst, blocks);
}

pub fn scaleTyped(
    rt: *Runtime,
    comptime dtype: DType,
    x: *const tensor.TensorOf(dtype),
    scalar_value: dtype_mod.Accumulator(dtype),
) !tensor.TensorOf(dtype_mod.outputDType(.pointwise, dtype)) {
    if (comptime dtype == .f32) return exec_elementwise.scale(rt, x, scalar_value);
    comptime ensureForwardFloatMath(dtype);
    const compute_dtype = comptime dtype_mod.computeDType(.pointwise, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.pointwise, dtype);

    var xx = try rt.prepareContiguousTyped(dtype, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try rt.emptyTyped(output_dtype, x.shape.slice());
    errdefer out.deinit();
    const output = out.data();
    for (output, input) |*dst, value| {
        const product = dtype_mod.castFloat(dtype, compute_dtype, value) * dtype_mod.castFloat(dtype, compute_dtype, dtype_mod.fromAccumulator(dtype, scalar_value));
        dst.* = dtype_mod.castFloat(compute_dtype, output_dtype, product);
    }
    return out;
}
