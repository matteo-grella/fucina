const std = @import("std");
const build_options = @import("build_options");
const dtype_mod = @import("../dtype.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");
const packed_matmul = @import("packed.zig");
const ops = @import("ops.zig");
const quantized_matmul = @import("quant.zig");
const thread = @import("../thread.zig");
const vector = @import("vector.zig");
const gpu = @import("gpu.zig").impl;

const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

const q8_0_lhs_stack_blocks: usize = 512;
// Off-multiple-m row minimums for the x4/x8 fast paths. q4_k pads the final
// partial row group inside the x4 kernel, so every m >= 4 takes it (one pass
// over the packed weights). q5_k has no padded-group kernel: its bulk+tail
// split re-reads the packed weights once more for the 1-3 remainder rows, so
// below 128 rows the one-pass per-row path wins.
const q4_k_x4_min_rows: usize = 4;
const q5_k_x4_prefix_min_rows: usize = 128;

fn checkedTensorProduct(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch tensor.TensorError.InvalidDataLength;
}

fn checkedQuantizedProduct(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch quantized_matmul.QuantizedFormatError.InvalidQuantizedLength;
}

const cblas_row_major: c_int = 101;
const cblas_no_trans: c_int = 111;
const cblas_trans: c_int = 112;
const max_cblas_dim: usize = @intCast(std.math.maxInt(c_int));
var blas_threads_config_done = std.atomic.Value(bool).init(false);
var blas_threads_config_mutex: thread.Mutex = .{};

extern fn cblas_sgemm(
    order: c_int,
    trans_a: c_int,
    trans_b: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: [*]const f32,
    lda: c_int,
    b: [*]const f32,
    ldb: c_int,
    beta: f32,
    c: [*]f32,
    ldc: c_int,
) void;

extern fn openblas_set_num_threads(num_threads: c_int) void;
extern fn bli_thread_set_num_threads(num_threads: c_int) void;
extern fn mkl_set_num_threads(num_threads: c_int) void;
extern fn nvpl_blas_set_num_threads(num_threads: c_int) void;

pub const vector_len = vector.vector_len;
pub const ParallelConfig = vector.ParallelConfig;
pub const blas_kind = build_options.blas_kind;
pub const blas_threads = build_options.blas_threads;

pub const addInto = vector.addInto;
pub const addContiguousIntoUnchecked = vector.addContiguousIntoUnchecked;
pub const addContiguousIntoUncheckedWithConfig = vector.addContiguousIntoUncheckedWithConfig;
pub const subInto = vector.subInto;
pub const subContiguousIntoUnchecked = vector.subContiguousIntoUnchecked;
pub const subContiguousIntoUncheckedWithConfig = vector.subContiguousIntoUncheckedWithConfig;
pub const mulInto = vector.mulInto;
pub const mulContiguousIntoUnchecked = vector.mulContiguousIntoUnchecked;
pub const mulContiguousIntoUncheckedWithConfig = vector.mulContiguousIntoUncheckedWithConfig;
pub const scaleInto = vector.scaleInto;
pub const scaleIntoWithConfig = vector.scaleIntoWithConfig;
pub const addScaledSlice = vector.addScaledSlice;
pub const addRowVectorSlice = vector.addRowVectorSlice;
pub const addRowVectorUnarySlice = vector.addRowVectorUnarySlice;
pub const Conv2dDims = vector.Conv2dDims;
pub const conv2dIntoWithConfig = vector.conv2dIntoWithConfig;
pub const conv2dBackwardInputIntoWithConfig = vector.conv2dBackwardInputIntoWithConfig;
pub const conv2dBackwardWeightIntoWithConfig = vector.conv2dBackwardWeightIntoWithConfig;
pub const im2colIntoWithConfig = vector.im2colIntoWithConfig;
pub const col2imIntoWithConfig = vector.col2imIntoWithConfig;
pub const WinogradF2Dims = vector.WinogradF2Dims;
pub const winogradF2WeightTransformIntoWithConfig = vector.winogradF2WeightTransformIntoWithConfig;
pub const winogradF2InputTransformIntoWithConfig = vector.winogradF2InputTransformIntoWithConfig;
pub const winogradF2OutputTransformIntoWithConfig = vector.winogradF2OutputTransformIntoWithConfig;
pub const winogradF4WeightTransformIntoWithConfig = vector.winogradF4WeightTransformIntoWithConfig;
pub const winogradF4InputTransformIntoWithConfig = vector.winogradF4InputTransformIntoWithConfig;
pub const winogradF4OutputTransformIntoWithConfig = vector.winogradF4OutputTransformIntoWithConfig;
pub const PoolKind = vector.PoolKind;
pub const Pool2dDims = vector.Pool2dDims;
pub const pool2dIntoWithConfig = vector.pool2dIntoWithConfig;
pub const avgPool2dBackwardIntoWithConfig = vector.avgPool2dBackwardIntoWithConfig;
pub const maxPool2dBackwardIntoWithConfig = vector.maxPool2dBackwardIntoWithConfig;
pub const upsample2xNearestIntoWithConfig = vector.upsample2xNearestIntoWithConfig;
pub const preluChannelsIntoWithConfig = vector.preluChannelsIntoWithConfig;
pub const preluChannelsBackwardInputIntoWithConfig = vector.preluChannelsBackwardInputIntoWithConfig;
pub const preluChannelsBackwardAlphaIntoWithConfig = vector.preluChannelsBackwardAlphaIntoWithConfig;
pub const channelAffineIntoWithConfig = vector.channelAffineIntoWithConfig;
pub const Conv1dDims = vector.Conv1dDims;
pub const conv1dIntoWithConfig = vector.conv1dIntoWithConfig;
pub const conv1dBackwardInputIntoWithConfig = vector.conv1dBackwardInputIntoWithConfig;
pub const conv1dBackwardWeightIntoWithConfig = vector.conv1dBackwardWeightIntoWithConfig;
pub const col2im1dIntoWithConfig = vector.col2im1dIntoWithConfig;
pub const col2im1dBackwardIntoWithConfig = vector.col2im1dBackwardIntoWithConfig;
pub const snakeIntoWithConfig = vector.snakeIntoWithConfig;
pub const snakeBackwardInputIntoWithConfig = vector.snakeBackwardInputIntoWithConfig;
pub const snakeBackwardParamsIntoWithConfig = vector.snakeBackwardParamsIntoWithConfig;
pub const groupNormIntoWithConfig = vector.groupNormIntoWithConfig;
pub const groupNormBackwardIntoWithConfig = vector.groupNormBackwardIntoWithConfig;
pub const causalDepthwiseConv1dIntoWithConfig = vector.causalDepthwiseConv1dIntoWithConfig;
pub const causalDepthwiseConv1dBackwardInputIntoWithConfig = vector.causalDepthwiseConv1dBackwardInputIntoWithConfig;
pub const causalDepthwiseConv1dBackwardKernelIntoWithConfig = vector.causalDepthwiseConv1dBackwardKernelIntoWithConfig;
pub const causalConv1dIntoWithConfig = vector.causalConv1dIntoWithConfig;
pub const causalConv1dBackwardInputIntoWithConfig = vector.causalConv1dBackwardInputIntoWithConfig;
pub const causalConv1dBackwardWeightIntoWithConfig = vector.causalConv1dBackwardWeightIntoWithConfig;
pub const groupedCausalConv1dIntoWithConfig = vector.groupedCausalConv1dIntoWithConfig;
pub const groupedCausalConv1dBackwardInputIntoWithConfig = vector.groupedCausalConv1dBackwardInputIntoWithConfig;
pub const groupedCausalConv1dBackwardWeightIntoWithConfig = vector.groupedCausalConv1dBackwardWeightIntoWithConfig;
pub const unaryContiguousIntoUnchecked = vector.unaryContiguousIntoUnchecked;
pub const unaryContiguousIntoUncheckedWithConfig = vector.unaryContiguousIntoUncheckedWithConfig;
pub const leakyReluContiguousIntoUnchecked = vector.leakyReluContiguousIntoUnchecked;
pub const leakyReluContiguousIntoUncheckedWithConfig = vector.leakyReluContiguousIntoUncheckedWithConfig;
pub const clampContiguousIntoUnchecked = vector.clampContiguousIntoUnchecked;
pub const clampContiguousIntoUncheckedWithConfig = vector.clampContiguousIntoUncheckedWithConfig;
pub const gatedContiguousIntoUnchecked = vector.gatedContiguousIntoUnchecked;
pub const gatedContiguousIntoUncheckedWithConfig = vector.gatedContiguousIntoUncheckedWithConfig;
pub const sumInto = vector.sumInto;
pub const sumIntoWithConfig = vector.sumIntoWithConfig;
pub const sumSlice = vector.sumSlice;
pub const maximumContiguousIntoUncheckedWithConfig = vector.maximumContiguousIntoUncheckedWithConfig;
pub const minimumContiguousIntoUncheckedWithConfig = vector.minimumContiguousIntoUncheckedWithConfig;
pub const prodInto = vector.prodInto;
pub const prodIntoWithConfig = vector.prodIntoWithConfig;
pub const prodSlice = vector.prodSlice;
pub const dotInto = vector.dotInto;
pub const dotIntoWithConfig = vector.dotIntoWithConfig;
pub const elementwiseContiguousIntoTypedWithConfig = vector.elementwiseContiguousIntoTypedWithConfig;
pub const sumSliceTypedWithConfig = vector.sumSliceTypedWithConfig;
pub const dotIntoTypedWithConfig = vector.dotIntoTypedWithConfig;

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
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuForRhs(b, m, n, k)) {
            if (gpu.gemmF32Async(.nn, a, b, out, m, n, k)) return;
        }
    }
    if (comptime build_options.use_blas) {
        if (shouldUseBlas(m, n, k)) {
            blasGemm(
                cblas_no_trans,
                cblas_no_trans,
                m,
                n,
                k,
                contiguousDataConst(a, m * k),
                k,
                contiguousDataConst(b, k * n),
                n,
                contiguousData(out, m * n),
            );
            return;
        }
    }
    vector.matmul2DIntoUncheckedWithConfig(out, a, b, m, n, k, config);
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
    vector.matmul2DIntoUncheckedTypedWithConfig(dtype, out, a, b, m, n, k, config);
}

pub fn packMatmulRhsTyped(
    comptime dtype: DType,
    allocator: std.mem.Allocator,
    rhs: *const tensor.TensorOf(dtype),
) !packed_matmul.PackedMatmulRhsFor(dtype) {
    return packed_matmul.packRhs(allocator, dtype, rhs);
}

pub fn packDenseMatmulRhsTyped(
    comptime dtype: DType,
    allocator: std.mem.Allocator,
    rhs: *const tensor.TensorOf(dtype),
) !packed_matmul.PackedDenseRhs {
    return packed_matmul.packDenseRhs(allocator, dtype, rhs);
}

pub fn matmul2DIntoUncheckedPackedDenseRhsWithConfig(
    out: *Tensor,
    a: *const Tensor,
    rhs: *const packed_matmul.PackedDenseRhs,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuForRhs(&rhs.rhs, m, n, k)) {
            if (gpu.gemmF32Async(.nt, a, &rhs.rhs, out, m, n, k)) return;
        }
    }
    // Explicit packed-op decision table: GPU always wins; BLAS keeps its
    // established all-dimensions>=16 cells; the packed microkernel owns the
    // m<16 cliff and every no-BLAS cell.
    if (comptime build_options.use_blas) {
        if (shouldUseBlas(m, n, k)) {
            blasGemm(
                cblas_no_trans,
                cblas_trans,
                m,
                n,
                k,
                contiguousDataConst(a, m * k),
                k,
                contiguousDataConst(&rhs.rhs, rhs.padded_n * k),
                k,
                contiguousData(out, m * n),
            );
            return;
        }
    }
    vector.gemmPackedNtIntoWithConfig(
        contiguousData(out, m * n),
        contiguousDataConst(a, m * k),
        contiguousDataConst(&rhs.rhs, rhs.padded_n * k),
        m,
        n,
        k,
        config,
    );
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
        .ggml_q2_0 => |qrhs| matmul2DQuantizedRhsQ2_0WithConfig(allocator, out, a, qrhs, m, n, k, config),
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
        .ggml_tq2_0 => |qrhs| matmul2DQuantizedRhsTQ2_0WithConfig(allocator, out, a, qrhs, m, n, k, config),
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
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const a_len = try checkedTensorProduct(m, k);
    const out_len = try checkedTensorProduct(m, n);
    const ad = contiguousDataConst(a, a_len);
    const cd = contiguousData(out, out_len);

    const qa = try allocator.alloc(i8, a_len);
    defer allocator.free(qa);
    const a_scales = try allocator.alloc(f32, m);
    defer allocator.free(a_scales);

    quantized_matmul.quantizeActivationsPerRowI8(qa, a_scales, ad, m, k);
    vector.matmul2DI8BlockwiseIntoWithConfig(cd, qa, a_scales, rhs.qw.dataConst(), rhs.scales.dataConst(), m, n, k, rhs.group_size, rhs.num_groups, config);
}

fn matmul2DQuantizedRhsQ8_0RowsWithConfig(
    comptime kernel: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, m * n);
    const blocks_per_row = try quantized_matmul.q8_0BlockCount(k);
    const block_count = m * blocks_per_row;
    var stack_blocks: [q8_0_lhs_stack_blocks]quantized_matmul.BlockQ8_0 = undefined;
    const qlhs_blocks = if (block_count <= stack_blocks.len)
        stack_blocks[0..block_count]
    else
        try allocator.alloc(quantized_matmul.BlockQ8_0, block_count);
    defer if (block_count > stack_blocks.len) allocator.free(qlhs_blocks);

    try quantized_matmul.quantizeRowsQ8_0Into(qlhs_blocks, a);
    kernel(cd, qlhs_blocks, rhs, m, n, k, config);
}

fn matmul2DQuantizedRhsQ8_1RowsWithConfig(
    comptime kernel: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, m * n);
    var qlhs = try quantized_matmul.quantizeRowsQ8_1(allocator, a);
    defer qlhs.deinit();
    kernel(cd, qlhs.blocks, rhs, m, n, k, config);
}

fn matmul2DQuantizedRhsQ8_KRowsWithConfig(
    comptime kernel: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, m * n);
    const qlhs = try quantized_matmul.quantizeRowsQ8_K(allocator, a);
    defer allocator.free(qlhs);
    kernel(cd, qlhs, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_0RowsWithConfig(vector.matmul2DQ4_0RhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_0RowsWithConfig(vector.matmul2DQ1_0RhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
}

/// Prefill row count at/above which the Q2_0 matmul dequantizes weight
/// panels to f32 and rides BLAS (Accelerate AMX / OpenBLAS): the dequant
/// pass costs O(n*k) regardless of m, so its amortization — and the GEMM's
/// O(m) operand reuse, out of the int8 sdot path's reach on AMX-class
/// units — grows with m, while below the threshold (decode, short bursts)
/// the int8 mul-free path wins. Same split llama.cpp's BLAS backend makes
/// for its quantized prefill. The BLAS arm consumes exact f32 activations
/// (no Q8_0 LHS quantization), so its numerics differ from the int path
/// exactly as the dense-f32 BLAS GEMMs already do from the scalar backend.
const q2_0_blas_min_m: usize = 192;
/// f32 scratch budget for one dequantized weight panel. Panels slice the
/// CONTRACT dimension, never the output dimension: every GEMM is then
/// full-width with a contiguous C (accumulating across slices via beta=1),
/// where output-dimension panels would give narrow GEMMs writing a strided
/// C — a shape BLAS handles poorly.
const q2_0_blas_panel_floats: usize = 12 * 1024 * 1024; // 48 MiB

const Q2_0DequantSliceTask = struct {
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ2_0,
    dst: []f32, // kp floats of weight row `row`, from block bi0
    row: usize,
    bi0: usize,
};

fn runQ2_0DequantSlice(task: *const Q2_0DequantSliceTask) void {
    const blocks = task.rhs.columnBlocks(task.row);
    // Lengths are exact by construction (dst covers whole blocks), so the
    // only representable error cannot occur.
    quantized_matmul.dequantizeRowQ2_0FastInto(
        task.dst,
        blocks[task.bi0 .. task.bi0 + task.dst.len / quantized_matmul.q2_0_block_size],
    ) catch unreachable;
}

fn matmul2DQuantizedRhsQ2_0BlasWithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ2_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const ad = contiguousDataConst(a, m * k);
    const cd = contiguousData(out, m * n);

    // k-slice width: whole 128-blocks, full k when it fits the budget.
    const kp_max = @max(quantized_matmul.q2_0_block_size, (q2_0_blas_panel_floats / n) & ~(quantized_matmul.q2_0_block_size - 1));
    const kp = @min(k, kp_max);
    const panel = try allocator.alloc(f32, n * kp);
    defer allocator.free(panel);
    const tasks = try allocator.alloc(Q2_0DequantSliceTask, n);
    defer allocator.free(tasks);

    var k0: usize = 0;
    while (k0 < k) : (k0 += kp) {
        const kc = @min(kp, k - k0);
        const bi0 = k0 / quantized_matmul.q2_0_block_size;
        for (0..n) |row| tasks[row] = .{ .rhs = rhs, .dst = panel[row * kc ..][0..kc], .row = row, .bi0 = bi0 };
        if (config.pool) |pool| {
            pool.parallelChunks(Q2_0DequantSliceTask, tasks, runQ2_0DequantSlice);
        } else {
            for (tasks) |*t| runQ2_0DequantSlice(t);
        }
        // C (m x n, full width) += A[:, k0..k0+kc] x panel^T (kc x n).
        ensureBlasThreadsConfigured();
        cblas_sgemm(
            cblas_row_major,
            cblas_no_trans,
            cblas_trans,
            cDim(m),
            cDim(n),
            cDim(kc),
            1.0,
            ad.ptr + k0,
            cDim(k),
            panel.ptr,
            cDim(kc),
            if (k0 == 0) 0.0 else 1.0,
            cd.ptr,
            cDim(n),
        );
    }
}

pub fn matmul2DQuantizedRhsQ2_0WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ2_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    if (comptime build_options.use_blas) {
        if (m >= q2_0_blas_min_m) {
            return matmul2DQuantizedRhsQ2_0BlasWithConfig(allocator, out, a, rhs, m, n, k, config);
        }
    }
    return matmul2DQuantizedRhsQ8_0RowsWithConfig(vector.matmul2DQ2_0RhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_1RowsWithConfig(vector.matmul2DQ4_1RhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_0RowsWithConfig(vector.matmul2DQ5_0RhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_1RowsWithConfig(vector.matmul2DQ5_1RhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_0RowsWithConfig(vector.matmul2DQ8_0RhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    if (m % 4 != 0) {
        if (m >= 12 and m < parallel.vector_column_min_m) {
            const cd = contiguousData(out, m * n);
            const blocks_per_row = try quantized_matmul.q8_0BlockCount(k);
            const row_groups = (m + 3) / 4;
            var stack_blocks: [q8_0_lhs_stack_blocks]quantized_matmul.BlockQ8_0x4 = undefined;
            const block_count = row_groups * blocks_per_row;
            const qlhs_blocks = if (block_count <= stack_blocks.len)
                stack_blocks[0..block_count]
            else
                try allocator.alloc(quantized_matmul.BlockQ8_0x4, block_count);
            defer if (block_count > stack_blocks.len) allocator.free(qlhs_blocks);

            try quantized_matmul.quantizeRowsQ8_0x4PaddedInto(qlhs_blocks, a);
            vector.matmul2DQ8_0x4PackedPaddedRhsIntoWithConfig(cd, qlhs_blocks, rhs, m, n, k, config);
            return;
        }
        if (m >= parallel.vector_column_min_m) {
            return matmul2DQuantizedRhsQ8_0x4BulkTailWithConfig(allocator, out, a, rhs, m, n, k, config);
        }
        return matmul2DQuantizedRhsQ8_0RowsWithConfig(vector.matmul2DQ8_0x4RhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
    }

    const cd = contiguousData(out, m * n);
    const blocks_per_row = try quantized_matmul.q8_0BlockCount(k);
    const block_count = (m / 4) * blocks_per_row;
    var stack_blocks: [q8_0_lhs_stack_blocks]quantized_matmul.BlockQ8_0x4 = undefined;
    const qlhs_blocks = if (block_count <= stack_blocks.len)
        stack_blocks[0..block_count]
    else
        try allocator.alloc(quantized_matmul.BlockQ8_0x4, block_count);
    defer if (block_count > stack_blocks.len) allocator.free(qlhs_blocks);

    try quantized_matmul.quantizeRowsQ8_0x4Into(qlhs_blocks, a);
    vector.matmul2DQ8_0x4PackedRhsIntoWithConfig(cd, qlhs_blocks, rhs, m, n, k, config);
}

// m >= vector_column_min_m with m % 4 != 0: the multiple-of-4 bulk runs
// through the packed x4 kernel and the 1-3 remainder rows through the row
// kernel (previously the WHOLE matmul fell to the per-row path). The split
// keeps every row's math identical to the kernel that owns it: bulk rows
// match an m % 4 == 0 dispatch bit-for-bit, remainder rows match the row
// kernel bit-for-bit.
fn matmul2DQuantizedRhsQ8_0x4BulkTailWithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    const cd = contiguousData(out, try checkedTensorProduct(m, n));
    const blocks_per_row = try quantized_matmul.q8_0BlockCount(k);
    const bulk_rows = m - m % 4;

    {
        const block_count = try checkedQuantizedProduct(bulk_rows / 4, blocks_per_row);
        var stack_blocks: [q8_0_lhs_stack_blocks]quantized_matmul.BlockQ8_0x4 = undefined;
        const qlhs_blocks = if (block_count <= stack_blocks.len)
            stack_blocks[0..block_count]
        else
            try allocator.alloc(quantized_matmul.BlockQ8_0x4, block_count);
        defer if (block_count > stack_blocks.len) allocator.free(qlhs_blocks);

        var bulk = try a.viewWithStridesOffset(&.{ bulk_rows, k }, &.{ k, 1 }, 0);
        defer bulk.deinit();
        try quantized_matmul.quantizeRowsQ8_0x4Into(qlhs_blocks, &bulk);
        vector.matmul2DQ8_0x4PackedRhsIntoWithConfig(cd[0 .. bulk_rows * n], qlhs_blocks, rhs, bulk_rows, n, k, config);
    }

    const tail_rows = m - bulk_rows;
    var tail = try a.viewWithStridesOffset(&.{ tail_rows, k }, &.{ k, 1 }, bulk_rows * k);
    defer tail.deinit();
    const tail_count = try checkedQuantizedProduct(tail_rows, blocks_per_row);
    var tail_stack: [q8_0_lhs_stack_blocks]quantized_matmul.BlockQ8_0 = undefined;
    const tail_blocks = if (tail_count <= tail_stack.len)
        tail_stack[0..tail_count]
    else
        try allocator.alloc(quantized_matmul.BlockQ8_0, tail_count);
    defer if (tail_count > tail_stack.len) allocator.free(tail_blocks);

    try quantized_matmul.quantizeRowsQ8_0Into(tail_blocks, &tail);
    // The <=3-row remainder runs after the bulk kernel completes; the caller's
    // config passes through so it can column-split like a decode-shaped matmul
    // (a parallel split never changes per-element math).
    vector.matmul2DQ8_0x4RhsIntoWithConfig(cd[bulk_rows * n .. m * n], tail_blocks, rhs, tail_rows, n, k, config);
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
    if (m % 4 != 0) return tensor.TensorError.InvalidShape;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try quantized_matmul.q8_0BlockCount(k);
    if (lhs_blocks.len != try checkedQuantizedProduct(m / 4, blocks_per_row)) return quantized_matmul.QuantizedFormatError.InvalidQuantizedLength;
    const cd = contiguousData(out, try checkedTensorProduct(m, n));
    vector.matmul2DQ8_0x4PackedRhsIntoWithConfig(cd, lhs_blocks, rhs, m, n, k, config);
}

// Pre-quantized-LHS K-quant GEMM entries for the fused split-activation ops:
// exec quantizes the activation rows itself there, so these skip the
// allocator-based LHS quantization of the matmul2DQuantizedRhs* wrappers.
pub fn matmulPackedQ4_Kx8Q8_Kx4SliceWithConfig(out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_Kx4, rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    vector.matmul2DQ4_Kx8Q8_Kx4RhsIntoWithConfig(out, lhs_blocks, rhs, m, n, k, config);
}

pub fn matmulPackedQ4_Kx8RowsSliceWithConfig(out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    vector.matmul2DQ4_Kx8RhsIntoWithConfig(out, lhs_blocks, rhs, m, n, k, config);
}

pub fn matmulPackedQ5_Kx8Q8_Kx4SliceWithConfig(out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_Kx4, rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    vector.matmul2DQ5_Kx8Q8_Kx4RhsIntoWithConfig(out, lhs_blocks, rhs, m, n, k, config);
}

pub fn matmulPackedQ5_Kx8RowsSliceWithConfig(out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    vector.matmul2DQ5_Kx8RhsIntoWithConfig(out, lhs_blocks, rhs, m, n, k, config);
}

pub fn matmulPackedQ6_Kx4RowsSliceWithConfig(out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_Kx4, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    vector.matmul2DQ6_Kx4RhsIntoWithConfig(out, lhs_blocks, rhs, m, n, k, config);
}

// Single-row slice kernels for fused per-row activation math (exact same
// vector kernels the unfused elementwise ops apply).
pub fn unaryRowSlice(comptime op: ops.UnaryOp, z: []f32, x: []const f32) void {
    vector.vecUnary(op, z, x);
}

pub fn mulRowSlice(z: []f32, x: []const f32, y: []const f32) void {
    vector.vecMul(z, x, y);
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
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try quantized_matmul.q8_0BlockCount(k);
    if (lhs_blocks.len != try checkedQuantizedProduct((m + 3) / 4, blocks_per_row)) return quantized_matmul.QuantizedFormatError.InvalidQuantizedLength;
    const cd = contiguousData(out, try checkedTensorProduct(m, n));
    vector.matmul2DQ8_0x4PackedPaddedRhsIntoWithConfig(cd, lhs_blocks, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(vector.matmul2DQ2_KRhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(vector.matmul2DQ3_KRhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(vector.matmul2DQ4_KRhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(vector.matmul2DQ4_Kx4RhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_Kx4PrefixWithConfig(
        vector.matmul2DQ4_Kx8Q8_Kx4RhsIntoWithConfig,
        vector.matmul2DQ4_Kx8RhsIntoWithConfig,
        allocator,
        out,
        a,
        rhs,
        m,
        n,
        k,
        q4_k_x4_min_rows,
        true,
        config,
    );
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
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, try checkedTensorProduct(m, n));
    const blocks_per_row = try quantized_matmul.blockCountForDType(.q8_k, k);
    const prefix_rows = m - m % 2;

    if (prefix_rows != 0) {
        const qlhs_x2 = try allocator.alloc(quantized_matmul.BlockQ8_Kx2Mmla, try checkedQuantizedProduct(prefix_rows / 2, blocks_per_row));
        defer allocator.free(qlhs_x2);

        if (prefix_rows == m) {
            try quantized_matmul.quantizeRowsQ8_Kx2MmlaInto(qlhs_x2, a);
        } else {
            var prefix = try a.viewWithStridesOffset(&.{ prefix_rows, k }, &.{ k, 1 }, 0);
            defer prefix.deinit();
            try quantized_matmul.quantizeRowsQ8_Kx2MmlaInto(qlhs_x2, &prefix);
        }
        vector.matmul2DQ4_Kx2MmlaQ8_Kx2MmlaRhsIntoWithConfig(cd[0 .. prefix_rows * n], qlhs_x2, rhs, prefix_rows, n, k, config);
    }

    if (prefix_rows == m) return;

    var tail = try a.viewWithStridesOffset(&.{ m - prefix_rows, k }, &.{ k, 1 }, prefix_rows * k);
    defer tail.deinit();
    const tail_blocks = try quantized_matmul.quantizeRowsQ8_K(allocator, &tail);
    defer allocator.free(tail_blocks);
    const tail_config = if (prefix_rows == 0) config else ParallelConfig{};
    vector.matmul2DQ4_Kx2MmlaRhsIntoWithConfig(cd[prefix_rows * n .. m * n], tail_blocks, rhs, m - prefix_rows, n, k, tail_config);
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
    return matmul2DQuantizedRhsQ8_Kx4PrefixWithConfig(
        vector.matmul2DQ5_Kx8Q8_Kx4RhsIntoWithConfig,
        vector.matmul2DQ5_Kx8RhsIntoWithConfig,
        allocator,
        out,
        a,
        rhs,
        m,
        n,
        k,
        q5_k_x4_prefix_min_rows,
        false,
        config,
    );
}

fn matmul2DQuantizedRhsQ8_Kx4PrefixWithConfig(
    comptime x4: anytype,
    comptime rows: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
    prefix_min_rows: usize,
    comptime pad_x4_rows: bool,
    config: ParallelConfig,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, try checkedTensorProduct(m, n));
    const blocks_per_row = try quantized_matmul.blockCountForDType(.q8_k, k);
    // Padded formats (pad_x4_rows) run every m >= prefix_min_rows through the
    // padded x4 kernel in one pass (the old gate had an all-rows-per-row hole
    // for m % 4 != 0 in [32, 64)). Unpadded formats run the multiple-of-4 bulk
    // through the x4 kernel and the 1-3 remainder rows through the row kernel;
    // that remainder costs an extra pass over the packed weights, so their
    // prefix_min_rows stays high. m % 4 == 0 dispatch is unchanged.
    const use_x4 = m % 4 == 0 or m >= prefix_min_rows;
    const prefix_rows = if (!use_x4) 0 else if (pad_x4_rows) m else m - m % 4;

    if (prefix_rows == 0) {
        return matmul2DQuantizedRhsQ8_KRowsWithConfig(rows, allocator, out, a, rhs, m, n, k, config);
    }

    const row_groups = if (pad_x4_rows) (prefix_rows + 3) / 4 else prefix_rows / 4;
    const qlhs_x4 = try allocator.alloc(quantized_matmul.BlockQ8_Kx4, try checkedQuantizedProduct(row_groups, blocks_per_row));
    defer allocator.free(qlhs_x4);

    if (prefix_rows == m) {
        if (pad_x4_rows) {
            try quantized_matmul.quantizeRowsQ8_Kx4PaddedInto(qlhs_x4, a);
        } else {
            try quantized_matmul.quantizeRowsQ8_Kx4Into(qlhs_x4, a);
        }
    } else {
        var prefix = try a.viewWithStridesOffset(&.{ prefix_rows, k }, &.{ k, 1 }, 0);
        defer prefix.deinit();
        try quantized_matmul.quantizeRowsQ8_Kx4Into(qlhs_x4, &prefix);
    }
    x4(cd[0 .. prefix_rows * n], qlhs_x4, rhs, prefix_rows, n, k, config);

    if (prefix_rows == m) return;

    var tail = try a.viewWithStridesOffset(&.{ m - prefix_rows, k }, &.{ k, 1 }, prefix_rows * k);
    defer tail.deinit();
    const tail_blocks = try quantized_matmul.quantizeRowsQ8_K(allocator, &tail);
    defer allocator.free(tail_blocks);
    // The <=3-row remainder runs after the x4 kernel completes; the caller's
    // config passes through so it can column-split like a decode-shaped matmul
    // (a parallel split never changes per-element math).
    rows(cd[prefix_rows * n .. m * n], tail_blocks, rhs, m - prefix_rows, n, k, config);
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
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(vector.matmul2DQ5_KRhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(vector.matmul2DQ6_KRhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    return matmul2DQuantizedRhsQ8_KRowsWithConfig(vector.matmul2DQ6_Kx4RhsIntoWithConfig, allocator, out, a, rhs, m, n, k, config);
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
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, m * n);
    var qlhs = try quantized_matmul.quantizeRowsQ8_0(allocator, a);
    defer qlhs.deinit();
    vector.matmul2DTableQ8_0RhsIntoWithConfig(rhs_dtype, cd, qlhs.blocks, rhs, m, n, k, config);
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
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, m * n);
    const qlhs = try quantized_matmul.quantizeRowsQ8_K(allocator, a);
    defer allocator.free(qlhs);
    vector.matmul2DTableQ8_KRhsIntoWithConfig(rhs_dtype, cd, qlhs, rhs, m, n, k, config);
}

fn matmul2DQuantizedRhsTQ2_0WithConfig(
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsTQ2_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, m * n);
    const qlhs = try quantized_matmul.quantizeRowsQ8_K(allocator, a);
    defer allocator.free(qlhs);
    vector.matmul2DTQ2_0RhsIntoWithConfig(cd, qlhs, rhs, m, n, k, config);
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
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuForRhs(b, m, n, k)) {
            if (gpu.gemmF32Async(.tn, a, b, out, m, n, k)) return;
        }
    }
    if (comptime build_options.use_blas) {
        if (shouldUseBlas(m, n, k)) {
            blasGemm(
                cblas_trans,
                cblas_no_trans,
                m,
                n,
                k,
                contiguousDataConst(a, k * m),
                m,
                contiguousDataConst(b, k * n),
                n,
                contiguousData(out, m * n),
            );
            return;
        }
    }
    vector.matmulTransA2DIntoUncheckedWithConfig(out, a, b, m, n, k, config);
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
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuForRhs(b, m, n, k)) {
            if (gpu.gemmF32Async(.nt, a, b, out, m, n, k)) return;
        }
    }
    if (comptime build_options.use_blas) {
        if (shouldUseBlas(m, n, k)) {
            blasGemm(
                cblas_no_trans,
                cblas_trans,
                m,
                n,
                k,
                contiguousDataConst(a, m * k),
                k,
                contiguousDataConst(b, n * k),
                k,
                contiguousData(out, m * n),
            );
            return;
        }
    }
    vector.matmulTransB2DIntoUncheckedWithConfig(out, a, b, m, n, k, config);
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
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuF16ForRhs(b, m, n, k)) {
            if (gpu.gemmF16NtAsync(a, b, out, m, n, k)) return;
        }
    }
    vector.matmulTransB2DIntoUncheckedF16OperandsWithConfig(out, a, b, m, n, k, config);
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
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuBf16ForRhs(b, m, n, k)) {
            if (gpu.gemmBf16NtAsync(a, b, out, m, n, k)) return;
        }
    }
    vector.matmulTransB2DIntoUncheckedBf16RhsWithConfig(out, a, b, m, n, k, config);
}

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
    if (batch_count == 0) return;
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuBatchedForRhs(b, m, n, k, batch_count)) {
            if (gpuBatched(.nn, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c)) return;
        }
    }
    if (comptime build_options.use_blas) {
        if (shouldUseBatchedBlas(m, n, k, batch_count)) {
            blasBatched(cblas_no_trans, cblas_no_trans, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, k, n);
            return;
        }
    }
    vector.matmulBatched2DIntoUncheckedWithConfig(out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, config);
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
    if (batch_count == 0) return;
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuBatchedForRhs(b, m, n, k, batch_count)) {
            if (gpuBatched(.tn, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c)) return;
        }
    }
    if (comptime build_options.use_blas) {
        if (shouldUseBatchedBlas(m, n, k, batch_count)) {
            blasBatched(cblas_trans, cblas_no_trans, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, m, n);
            return;
        }
    }
    vector.matmulBatchedTransA2DIntoUncheckedWithConfig(out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, config);
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
    if (batch_count == 0) return;
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuBatchedForRhs(b, m, n, k, batch_count)) {
            if (gpuBatched(.nt, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c)) return;
        }
    }
    if (comptime build_options.use_blas) {
        if (shouldUseBatchedBlas(m, n, k, batch_count)) {
            blasBatched(cblas_no_trans, cblas_trans, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, k, k);
            return;
        }
    }
    vector.matmulBatchedTransB2DIntoUncheckedWithConfig(out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, config);
}

/// One GPU dispatch covering all `batch_count` matrices (grid depth = batch);
/// returns false when the GPU did not run and the caller falls through.
fn gpuBatched(
    orient: gpu.Orient,
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
) bool {
    return gpu.gemmBatchedF32Async(
        orient,
        a,
        b,
        out,
        m,
        n,
        k,
        batch_count,
        stride_a,
        stride_b,
        stride_c,
    );
}

fn blasBatched(
    trans_a: c_int,
    trans_b: c_int,
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
    lda: usize,
    ldb: usize,
) void {
    @constCast(a.buffer).waitReady();
    @constCast(b.buffer).waitReady();
    out.buffer.waitMutable();
    const ap = a.buffer.data[a.offset..].ptr;
    const bp = b.buffer.data[b.offset..].ptr;
    const cp = out.buffer.data[out.offset..].ptr;
    const matrix_a_len = if (trans_a == cblas_trans) k * m else m * k;
    const matrix_b_len = if (trans_b == cblas_trans) n * k else k * n;

    for (0..batch_count) |bi| {
        blasGemm(
            trans_a,
            trans_b,
            m,
            n,
            k,
            ap[bi * stride_a .. bi * stride_a + matrix_a_len],
            lda,
            bp[bi * stride_b .. bi * stride_b + matrix_b_len],
            ldb,
            cp[bi * stride_c .. bi * stride_c + m * n],
        );
    }
}

fn blasGemm(
    trans_a: c_int,
    trans_b: c_int,
    m: usize,
    n: usize,
    k: usize,
    a: []const f32,
    lda: usize,
    b: []const f32,
    ldb: usize,
    c: []f32,
) void {
    ensureBlasThreadsConfigured();
    cblas_sgemm(
        cblas_row_major,
        trans_a,
        trans_b,
        cDim(m),
        cDim(n),
        cDim(k),
        1.0,
        a.ptr,
        cDim(lda),
        b.ptr,
        cDim(ldb),
        0.0,
        c.ptr,
        cDim(n),
    );
}

fn ensureBlasThreadsConfigured() void {
    if (comptime build_options.blas_threads != 0) {
        if (blas_threads_config_done.load(.acquire)) return;
        blas_threads_config_mutex.lock();
        defer blas_threads_config_mutex.unlock();
        if (!blas_threads_config_done.load(.monotonic)) {
            configureBlasThreads();
            blas_threads_config_done.store(true, .release);
        }
    }
}

fn configureBlasThreads() void {
    const requested = build_options.blas_threads;
    if (requested == 0) return;

    const max_threads: u32 = @intCast(std.math.maxInt(c_int));
    const n: c_int = @intCast(@min(requested, max_threads));
    switch (comptime build_options.blas_kind) {
        .openblas => openblas_set_num_threads(n),
        .blis => bli_thread_set_num_threads(n),
        .mkl => mkl_set_num_threads(n),
        .nvpl => nvpl_blas_set_num_threads(n),
        .none, .accelerate, .blas => {},
    }
}

fn fitsCblas(m: usize, n: usize, k: usize) bool {
    return m <= max_cblas_dim and n <= max_cblas_dim and k <= max_cblas_dim;
}

fn shouldUseBlas(m: usize, n: usize, k: usize) bool {
    return fitsCblas(m, n, k) and m >= 16 and n >= 16 and k >= 16;
}

fn shouldUseBatchedBlas(m: usize, n: usize, k: usize, batch_count: usize) bool {
    return batch_count > 1 and shouldUseBlas(m, n, k);
}

fn cDim(value: usize) c_int {
    return @intCast(value);
}

fn contiguousDataConst(x: *const Tensor, len: usize) []const f32 {
    @constCast(x.buffer).waitReady();
    return x.buffer.data[x.offset .. x.offset + len];
}

fn contiguousData(x: *Tensor, len: usize) []f32 {
    x.buffer.waitMutable();
    return x.buffer.data[x.offset .. x.offset + len];
}
