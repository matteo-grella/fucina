//! Dtype conversion ops for the eager runtime.
//!
//! `cast` is the general float-to-float cast; its vectorized cast kernels
//! are backend kernels (`backend.kernels.castF32ToF16`, ...), shared with
//! the KV-cache-append helpers in `exec.zig` and the optimizer's 16-bit
//! master-weight mirrors.
//!
//! Domain module: receives an explicit `*ExecContext` (never `self: anytype`);
//! `prepareContiguous` is the substrate primitive it draws on.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");
const exec_shape = @import("shape.zig");
const ensureForwardFloatMath = exec_shape.ensureForwardFloatMath;

const ExecContext = @import("../exec.zig").ExecContext;

// The vectorized cast kernels live in the backend (`vector/elementwise.zig`),
// registered in the conformed kernels table; this module keeps the cast
// op's validation/layout tier and the KV-cache-append view walkers.
const kernels = backend_mod.kernels;
const castF32ToF16 = kernels.castF32ToF16;
const castF16ToF32 = kernels.castF16ToF32;
const castF32ToBf16 = kernels.castF32ToBf16;
const castBf16ToF32 = kernels.castBf16ToF32;

const DType = tensor.DType;
const BlockQ8_0 = dtype_mod.BlockQ8_0;
const q8_0_block_size = dtype_mod.q8_0_block_size;

pub fn cast(
    ctx: *ExecContext,
    comptime source_dtype: DType,
    comptime target_dtype: DType,
    x: *const tensor.TensorOf(source_dtype),
) !tensor.TensorOf(target_dtype) {
    if (comptime source_dtype == target_dtype) return ctx.clone(source_dtype, x);
    if (comptime dtype_mod.isBlockQuantized(source_dtype) or dtype_mod.isBlockQuantized(target_dtype)) {
        @compileError("casts are supported between the scalar dtypes only (dequantize with to(.f32))");
    }

    var xx = try ctx.prepareContiguous(source_dtype, x);
    defer xx.deinit();
    const input = xx.tensor().dataConst();

    var out = try ctx.empty(target_dtype, x.shape.slice());
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

/// Cast an f32 tensor into a caller-owned f16 slice in logical row-major
/// order without allocating: the KV-cache append path. Supports contiguous
/// sources (one SIMD pass) and rank-3 views whose two inner axes are
/// contiguous (a `{seq, kv_head, d}` split of a fused QKV row), walked as
/// per-row spans. Anything else is UnsupportedView — extend deliberately
/// rather than silently gathering.
pub fn castF32RowsToF16Into(ctx: *ExecContext, x: *const tensor.Tensor, dst: []f16) !void {
    _ = ctx;
    if (dst.len != x.len()) return tensor.TensorError.InvalidDataLength;
    x.buffer.waitReady();
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
pub fn quantizeF32RowsToQ8_0Into(ctx: *ExecContext, x: *const tensor.Tensor, dst: []BlockQ8_0) !void {
    _ = ctx;
    if (x.len() % q8_0_block_size != 0) return tensor.TensorError.InvalidDataLength;
    if (dst.len != x.len() / q8_0_block_size) return tensor.TensorError.InvalidDataLength;
    x.buffer.waitReady();
    const data = x.buffer.data;
    if (x.isContiguous()) {
        try backend_mod.quantized_matmul.q8k.quantizeRowQ8_0Into(dst, data[x.offset..][0..x.len()]);
        return;
    }
    if (x.shape.len == 3 and x.strides.at(2) == 1 and x.strides.at(1) == x.shape.at(2)) {
        const rows = x.shape.at(0);
        const row = x.shape.at(1) * x.shape.at(2);
        if (row % q8_0_block_size != 0) return tensor.TensorError.InvalidDataLength;
        const row_blocks = row / q8_0_block_size;
        const row_stride = x.strides.at(0);
        for (0..rows) |i| {
            try backend_mod.quantized_matmul.q8k.quantizeRowQ8_0Into(
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
pub fn dequantizeQ8_0RowsInto(ctx: *ExecContext, dst: []f32, blocks: []const BlockQ8_0) !void {
    _ = ctx;
    try backend_mod.quantized_matmul.q8k.dequantizeRowQ8_0Into(dst, blocks);
}
