//! Native backend: the `kernels` namespace below is the kernel set
//! `backend/interface.zig` names. Elementwise, reduction, conv, pool and
//! Winograd entries forward to the portable `@Vector` leaves in `vector/`;
//! the dense and quantized GEMM family is defined in this file and routes
//! each call across the GPU provider (`-Dgpu`), a CBLAS provider (`-Dblas`)
//! and the vector kernels. Every pool-taking kernel takes `pc:
//! ParallelConfig` first; `cpu.zig` is the scalar reference with the same
//! signatures. Layer stack: docs/ARCHITECTURE.md.
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

const native = @This();
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
    return std.math.mul(usize, a, b) catch quantized_matmul.types.QuantizedFormatError.InvalidQuantizedLength;
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
// MKL's C entry points are the capitalized names. The lowercase
// `mkl_set_num_threads` is the Fortran binding: it takes its argument by
// reference, so calling it by value dereferences the count.
extern fn MKL_Set_Num_Threads(num_threads: c_int) void;
// Per-thread override; returns the previous value for the calling thread,
// where 0 means "follow the global setting".
extern fn MKL_Set_Num_Threads_Local(num_threads: c_int) c_int;
extern fn nvpl_blas_set_num_threads(num_threads: c_int) void;

pub const ParallelConfig = vector.ParallelConfig;

/// The kernel set this backend provides (`backend/interface.zig` names it):
/// the `vector/` leaf kernels by name, and the GEMM family defined below.
pub const kernels = struct {
    pub const addInto = vector.elementwise.addInto;
    pub const addContiguousIntoUnchecked = vector.elementwise.addContiguousIntoUnchecked;
    pub const divContiguousIntoUnchecked = vector.elementwise.divContiguousIntoUnchecked;
    pub const maximumContiguousIntoUnchecked = vector.elementwise.maximumContiguousIntoUnchecked;
    pub const minimumContiguousIntoUnchecked = vector.elementwise.minimumContiguousIntoUnchecked;
    pub const subInto = vector.elementwise.subInto;
    pub const subContiguousIntoUnchecked = vector.elementwise.subContiguousIntoUnchecked;
    pub const mulInto = vector.elementwise.mulInto;
    pub const mulContiguousIntoUnchecked = vector.elementwise.mulContiguousIntoUnchecked;
    pub const elementwiseContiguousIntoTyped = vector.elementwise.elementwiseContiguousIntoTyped;
    pub const scaleInto = vector.elementwise.scaleInto;
    pub const addScaledSlice = vector.elementwise.addScaledSlice;
    pub const addRowVectorSlice = vector.elementwise.addRowVectorSlice;
    pub const addRowVectorUnarySlice = vector.elementwise.addRowVectorUnarySlice;
    pub const causalDepthwiseConv1dInto = vector.conv.causalDepthwiseConv1dInto;
    pub const causalDepthwiseConv1dBackwardInputInto = vector.conv.causalDepthwiseConv1dBackwardInputInto;
    pub const causalDepthwiseConv1dBackwardKernelInto = vector.conv.causalDepthwiseConv1dBackwardKernelInto;
    pub const causalConv1dInto = vector.conv.causalConv1dInto;
    pub const conv2dInto = vector.conv.conv2dInto;
    pub const conv2dBackwardInputInto = vector.conv.conv2dBackwardInputInto;
    pub const conv2dBackwardWeightInto = vector.conv.conv2dBackwardWeightInto;
    pub const im2colInto = vector.conv.im2colInto;
    pub const col2imInto = vector.conv.col2imInto;
    pub const winogradF2WeightTransformInto = vector.winograd.f2WeightTransformInto;
    pub const winogradF2InputTransformInto = vector.winograd.f2InputTransformInto;
    pub const winogradF2OutputTransformInto = vector.winograd.f2OutputTransformInto;
    pub const winogradF4WeightTransformInto = vector.winograd.f4WeightTransformInto;
    pub const winogradF4InputTransformInto = vector.winograd.f4InputTransformInto;
    pub const winogradF4OutputTransformInto = vector.winograd.f4OutputTransformInto;
    pub const pool2dInto = vector.pool.pool2dInto;
    pub const avgPool2dBackwardInto = vector.pool.avgPool2dBackwardInto;
    pub const maxPool2dBackwardInto = vector.pool.maxPool2dBackwardInto;
    pub const upsample2xNearestInto = vector.pool.upsample2xNearestInto;
    pub const preluChannelsInto = vector.elementwise.preluChannelsInto;
    pub const preluChannelsBackwardInputInto = vector.elementwise.preluChannelsBackwardInputInto;
    pub const preluChannelsBackwardAlphaInto = vector.elementwise.preluChannelsBackwardAlphaInto;
    pub const channelAffineInto = vector.elementwise.channelAffineInto;
    pub const conv1dInto = vector.conv.conv1dInto;
    pub const conv1dBackwardInputInto = vector.conv.conv1dBackwardInputInto;
    pub const conv1dBackwardWeightInto = vector.conv.conv1dBackwardWeightInto;
    pub const col2im1dInto = vector.conv.col2im1dInto;
    pub const col2im1dBackwardInto = vector.conv.col2im1dBackwardInto;
    pub const snakeInto = vector.elementwise.snakeInto;
    pub const snakeBackwardInputInto = vector.elementwise.snakeBackwardInputInto;
    pub const snakeBackwardParamsInto = vector.elementwise.snakeBackwardParamsInto;
    pub const groupNormInto = vector.elementwise.groupNormInto;
    pub const groupNormBackwardInto = vector.elementwise.groupNormBackwardInto;
    pub const causalConv1dBackwardInputInto = vector.conv.causalConv1dBackwardInputInto;
    pub const causalConv1dBackwardWeightInto = vector.conv.causalConv1dBackwardWeightInto;
    pub const groupedCausalConv1dInto = vector.conv.groupedCausalConv1dInto;
    pub const groupedCausalConv1dBackwardInputInto = vector.conv.groupedCausalConv1dBackwardInputInto;
    pub const groupedCausalConv1dBackwardWeightInto = vector.conv.groupedCausalConv1dBackwardWeightInto;
    pub const unaryContiguousIntoUnchecked = vector.elementwise.unaryContiguousIntoUnchecked;
    pub const leakyReluContiguousIntoUnchecked = vector.elementwise.leakyReluContiguousIntoUnchecked;
    pub const clampContiguousIntoUnchecked = vector.elementwise.clampContiguousIntoUnchecked;
    pub const gatedContiguousIntoUnchecked = vector.elementwise.gatedContiguousIntoUnchecked;
    pub const sumInto = vector.elementwise.sumInto;
    pub const sumSlice = vector.elementwise.sumSlice;
    pub const prodInto = vector.elementwise.prodInto;
    pub const prodSlice = vector.elementwise.prodSlice;
    pub const sumSliceTyped = vector.elementwise.sumSliceTyped;
    pub const dotInto = vector.elementwise.dotInto;
    pub const dotIntoTyped = vector.elementwise.dotIntoTyped;
    pub const matmulInto = native.matmulInto;
    pub const matmul2DIntoUnchecked = native.matmul2DIntoUnchecked;
    pub const matmul2DAccIntoUnchecked = native.matmul2DAccIntoUnchecked;
    pub const matmul2DIntoUncheckedTyped = native.matmul2DIntoUncheckedTyped;
    pub const packMatmulRhsTyped = native.packMatmulRhsTyped;
    pub const packDenseMatmulRhsTyped = native.packDenseMatmulRhsTyped;
    pub const matmul2DIntoUncheckedPackedDenseRhs = native.matmul2DIntoUncheckedPackedDenseRhs;
    pub const matmul2DIntoUncheckedPackedRhsTyped = native.matmul2DIntoUncheckedPackedRhsTyped;
    pub const quantizeMatmulRhsBlockwiseI8 = native.quantizeMatmulRhsBlockwiseI8;
    pub const quantizeMatmulRhsQ4_0 = native.quantizeMatmulRhsQ4_0;
    pub const quantizeMatmulRhsQ8_0 = native.quantizeMatmulRhsQ8_0;
    pub const matmul2DQuantizedRhs = native.matmul2DQuantizedRhs;
    pub const matmul2DQuantizedRhsQ8_0x4 = native.matmul2DQuantizedRhsQ8_0x4;
    pub const matmul2DPackedQ8_0x4LhsRhs = native.matmul2DPackedQ8_0x4LhsRhs;
    pub const matmulPackedQ4_Kx8Q8_Kx4Slice = native.matmulPackedQ4_Kx8Q8_Kx4Slice;
    pub const matmulPackedQ4_Kx8RowsSlice = native.matmulPackedQ4_Kx8RowsSlice;
    pub const matmulPackedQ5_Kx8Q8_Kx4Slice = native.matmulPackedQ5_Kx8Q8_Kx4Slice;
    pub const matmulPackedQ5_Kx8RowsSlice = native.matmulPackedQ5_Kx8RowsSlice;
    pub const matmulPackedQ6_Kx4RowsSlice = native.matmulPackedQ6_Kx4RowsSlice;
    pub const unaryRowSlice = native.unaryRowSlice;
    pub const mulRowSlice = native.mulRowSlice;
    pub const matmul2DPackedPaddedQ8_0x4LhsRhs = native.matmul2DPackedPaddedQ8_0x4LhsRhs;
    pub const matmul2DQuantizedRhsQ6_Kx4 = native.matmul2DQuantizedRhsQ6_Kx4;
    pub const matmul2DQuantizedRhsQ4_Kx4 = native.matmul2DQuantizedRhsQ4_Kx4;
    pub const matmul2DQuantizedRhsQ4_Kx8 = native.matmul2DQuantizedRhsQ4_Kx8;
    pub const matmul2DQuantizedRhsQ4_Kx2Mmla = native.matmul2DQuantizedRhsQ4_Kx2Mmla;
    pub const matmul2DQuantizedRhsQ5_Kx8 = native.matmul2DQuantizedRhsQ5_Kx8;
    pub const matmulTransAInto = native.matmulTransAInto;
    pub const matmulTransA2DIntoUnchecked = native.matmulTransA2DIntoUnchecked;
    pub const matmulTransBInto = native.matmulTransBInto;
    pub const matmulTransB2DIntoUnchecked = native.matmulTransB2DIntoUnchecked;
    pub const matmulTransB2DIntoUncheckedF16Operands = native.matmulTransB2DIntoUncheckedF16Operands;
    pub const matmulTransB2DIntoUncheckedBf16Rhs = native.matmulTransB2DIntoUncheckedBf16Rhs;
    pub const matmulBatched2DIntoUnchecked = native.matmulBatched2DIntoUnchecked;
    pub const matmulBatchedTransA2DIntoUnchecked = native.matmulBatchedTransA2DIntoUnchecked;
    pub const matmulBatchedTransB2DIntoUnchecked = native.matmulBatchedTransB2DIntoUnchecked;
};

pub fn matmulInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    const av = try a.rankView(2);
    const bv = try b.rankView(2);
    const ov = try out.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    const n = bv.dim(1);
    if (k != bv.dim(0)) return tensor.TensorError.ShapeMismatch;
    if (ov.dim(0) != m or ov.dim(1) != n) return tensor.TensorError.ShapeMismatch;
    matmul2DIntoUnchecked(.{}, out, a, b, m, n, k);
}

pub fn matmul2DIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
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
                0.0,
                contiguousData(out, m * n),
            );
            return;
        }
    }
    vector.gemm.matmul2DIntoUnchecked(pc, out, a, b, m, n, k);
}

/// C += A·B (the beta=1 GEMM). BLAS route when the shape qualifies; the
/// vector accumulate kernels otherwise. GPU builds fall through to the same
/// CPU routes: the async GPU GEMM overwrites its destination and has no
/// accumulate seam.
pub fn matmul2DAccIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
) void {
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
                1.0,
                contiguousData(out, m * n),
            );
            return;
        }
    }
    vector.gemm.matmul2DAccIntoUnchecked(pc, out, a, b, m, n, k);
}

pub fn matmul2DIntoUncheckedTyped(
    pc: ParallelConfig,
    comptime dtype: DType,
    out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
    m: usize,
    n: usize,
    k: usize,
) void {
    vector.gemm.matmul2DIntoUncheckedTyped(pc, dtype, out, a, b, m, n, k);
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

pub fn matmul2DIntoUncheckedPackedDenseRhs(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const packed_matmul.PackedDenseRhs,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuForRhs(&rhs.rhs, m, n, k)) {
            if (gpu.gemmF32Async(.nt, a, &rhs.rhs, out, m, n, k)) return;
        }
    }
    // Explicit packed-op decision table: GPU always wins; BLAS keeps its
    // established all-dimensions>=16 cells EXCEPT the skinny-m tall-k band
    // below, where the already-packed microkernel is faster than Accelerate;
    // the packed microkernel also owns the m<16 cliff and every no-BLAS cell.
    if (comptime build_options.use_blas) {
        if (shouldUseBlas(m, n, k) and !packedDenseKernelPreferred(m, k)) {
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
                0.0,
                contiguousData(out, m * n),
            );
            return;
        }
    }
    vector.gemm_packed.gemmPackedNtInto(
        pc,
        contiguousData(out, m * n),
        contiguousDataConst(a, m * k),
        contiguousDataConst(&rhs.rhs, rhs.padded_n * k),
        m,
        n,
        k,
    );
}

pub fn matmul2DIntoUncheckedPackedRhsTyped(
    pc: ParallelConfig,
    comptime dtype: DType,
    allocator: std.mem.Allocator,
    out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
    a: *const tensor.TensorOf(dtype),
    rhs: *const packed_matmul.PackedMatmulRhsFor(dtype),
    m: usize,
    n: usize,
    k: usize,
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
        pc,
        matmul2DIntoUnchecked,
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
        .q2_k => |qrhs| matmul2DQuantizedRhsQ2_K(pc, allocator, out, a, qrhs, m, n, k),
        .q3_k => |qrhs| matmul2DQuantizedRhsQ3_K(pc, allocator, out, a, qrhs, m, n, k),
        .q4_k => |qrhs| matmul2DQuantizedRhsQ4_K(pc, allocator, out, a, qrhs, m, n, k),
        .q5_k => |qrhs| matmul2DQuantizedRhsQ5_K(pc, allocator, out, a, qrhs, m, n, k),
        .q6_k => |qrhs| matmul2DQuantizedRhsQ6_K(pc, allocator, out, a, qrhs, m, n, k),
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
        .tq2_0 => |qrhs| matmul2DQuantizedRhsTQ2_0(pc, allocator, out, a, qrhs, m, n, k),
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
    vector.matmul_quant.matmul2DI8BlockwiseInto(pc, cd, qa, a_scales, rhs.qw.dataConst(), rhs.scales.dataConst(), m, n, k, rhs.group_size, rhs.num_groups);
}

fn matmul2DQuantizedRhsQ8_0Rows(
    pc: ParallelConfig,
    comptime kernel: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, m * n);
    const blocks_per_row = try quantized_matmul.q8k.q8_0BlockCount(k);
    const block_count = m * blocks_per_row;
    var stack_blocks: [q8_0_lhs_stack_blocks]dtype_mod.BlockQ8_0 = undefined;
    const qlhs_blocks = if (block_count <= stack_blocks.len)
        stack_blocks[0..block_count]
    else
        try allocator.alloc(dtype_mod.BlockQ8_0, block_count);
    defer if (block_count > stack_blocks.len) allocator.free(qlhs_blocks);

    try quantized_matmul.q8k.quantizeRowsQ8_0Into(qlhs_blocks, a);
    kernel(pc, cd, qlhs_blocks, rhs, m, n, k);
}

fn matmul2DQuantizedRhsQ8_1Rows(
    pc: ParallelConfig,
    comptime kernel: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, m * n);
    var qlhs = try quantized_matmul.cold.quantizeRowsQ8_1(allocator, a);
    defer qlhs.deinit();
    kernel(pc, cd, qlhs.blocks, rhs, m, n, k);
}

fn matmul2DQuantizedRhsQ8_KRows(
    pc: ParallelConfig,
    comptime kernel: anytype,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: anytype,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, m * n);
    const qlhs = try quantized_matmul.q8k.quantizeRowsQ8_K(allocator, a);
    defer allocator.free(qlhs);
    kernel(pc, cd, qlhs, rhs, m, n, k);
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
    return matmul2DQuantizedRhsQ8_0Rows(pc, vector.matmul_quant.matmul2DQ4_0RhsInto, allocator, out, a, rhs, m, n, k);
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
    return matmul2DQuantizedRhsQ8_0Rows(pc, vector.matmul_quant.matmul2DQ1_0RhsInto, allocator, out, a, rhs, m, n, k);
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
    quantized_matmul.ternary.dequantizeRowQ2_0FastInto(
        task.dst,
        blocks[task.bi0 .. task.bi0 + task.dst.len / quantized_matmul.types.q2_0_block_size],
    ) catch unreachable;
}

fn matmul2DQuantizedRhsQ2_0Blas(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ2_0,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const ad = contiguousDataConst(a, m * k);
    const cd = contiguousData(out, m * n);

    // k-slice width: whole 128-blocks, full k when it fits the budget.
    const kp_max = @max(quantized_matmul.types.q2_0_block_size, (q2_0_blas_panel_floats / n) & ~(quantized_matmul.types.q2_0_block_size - 1));
    const kp = @min(k, kp_max);
    const panel = try allocator.alloc(f32, n * kp);
    defer allocator.free(panel);
    const tasks = try allocator.alloc(Q2_0DequantSliceTask, n);
    defer allocator.free(tasks);

    var k0: usize = 0;
    while (k0 < k) : (k0 += kp) {
        const kc = @min(kp, k - k0);
        const bi0 = k0 / quantized_matmul.types.q2_0_block_size;
        for (0..n) |row| tasks[row] = .{ .rhs = rhs, .dst = panel[row * kc ..][0..kc], .row = row, .bi0 = bi0 };
        if (pc.pool) |pool| {
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
    if (comptime build_options.use_blas) {
        if (m >= q2_0_blas_min_m) {
            return matmul2DQuantizedRhsQ2_0Blas(pc, allocator, out, a, rhs, m, n, k);
        }
    }
    return matmul2DQuantizedRhsQ8_0Rows(pc, vector.matmul_quant.matmul2DQ2_0RhsInto, allocator, out, a, rhs, m, n, k);
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
    return matmul2DQuantizedRhsQ8_1Rows(pc, vector.matmul_quant.matmul2DQ4_1RhsInto, allocator, out, a, rhs, m, n, k);
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
    return matmul2DQuantizedRhsQ8_0Rows(pc, vector.matmul_quant.matmul2DQ5_0RhsInto, allocator, out, a, rhs, m, n, k);
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
    return matmul2DQuantizedRhsQ8_1Rows(pc, vector.matmul_quant.matmul2DQ5_1RhsInto, allocator, out, a, rhs, m, n, k);
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
    return matmul2DQuantizedRhsQ8_0Rows(pc, vector.matmul_quant.matmul2DQ8_0RhsInto, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ8_0x4(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    if (m % 4 != 0) {
        if (m >= 12 and m < parallel.vector_column_min_m) {
            const cd = contiguousData(out, m * n);
            const blocks_per_row = try quantized_matmul.q8k.q8_0BlockCount(k);
            const row_groups = (m + 3) / 4;
            var stack_blocks: [q8_0_lhs_stack_blocks]quantized_matmul.BlockQ8_0x4 = undefined;
            const block_count = row_groups * blocks_per_row;
            const qlhs_blocks = if (block_count <= stack_blocks.len)
                stack_blocks[0..block_count]
            else
                try allocator.alloc(quantized_matmul.BlockQ8_0x4, block_count);
            defer if (block_count > stack_blocks.len) allocator.free(qlhs_blocks);

            try quantized_matmul.q8_0.quantizeRowsQ8_0x4PaddedInto(qlhs_blocks, a);
            vector.matmul_quant.matmul2DQ8_0x4PackedPaddedRhsInto(pc, cd, qlhs_blocks, rhs, m, n, k);
            return;
        }
        if (m >= parallel.vector_column_min_m) {
            return matmul2DQuantizedRhsQ8_0x4BulkTail(pc, allocator, out, a, rhs, m, n, k);
        }
        return matmul2DQuantizedRhsQ8_0Rows(pc, vector.matmul_quant.matmul2DQ8_0x4RhsInto, allocator, out, a, rhs, m, n, k);
    }

    const cd = contiguousData(out, m * n);
    const blocks_per_row = try quantized_matmul.q8k.q8_0BlockCount(k);
    const block_count = (m / 4) * blocks_per_row;
    var stack_blocks: [q8_0_lhs_stack_blocks]quantized_matmul.BlockQ8_0x4 = undefined;
    const qlhs_blocks = if (block_count <= stack_blocks.len)
        stack_blocks[0..block_count]
    else
        try allocator.alloc(quantized_matmul.BlockQ8_0x4, block_count);
    defer if (block_count > stack_blocks.len) allocator.free(qlhs_blocks);

    try quantized_matmul.q8_0.quantizeRowsQ8_0x4Into(qlhs_blocks, a);
    vector.matmul_quant.matmul2DQ8_0x4PackedRhsInto(pc, cd, qlhs_blocks, rhs, m, n, k);
}

// m >= vector_column_min_m with m % 4 != 0: the multiple-of-4 bulk runs
// through the packed x4 kernel and the 1-3 remainder rows through the row
// kernel (previously the WHOLE matmul fell to the per-row path). The split
// keeps every row's math identical to the kernel that owns it: bulk rows
// match an m % 4 == 0 dispatch bit-for-bit, remainder rows match the row
// kernel bit-for-bit.
fn matmul2DQuantizedRhsQ8_0x4BulkTail(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) !void {
    const cd = contiguousData(out, try checkedTensorProduct(m, n));
    const blocks_per_row = try quantized_matmul.q8k.q8_0BlockCount(k);
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
        try quantized_matmul.q8_0.quantizeRowsQ8_0x4Into(qlhs_blocks, &bulk);
        vector.matmul_quant.matmul2DQ8_0x4PackedRhsInto(pc, cd[0 .. bulk_rows * n], qlhs_blocks, rhs, bulk_rows, n, k);
    }

    const tail_rows = m - bulk_rows;
    var tail = try a.viewWithStridesOffset(&.{ tail_rows, k }, &.{ k, 1 }, bulk_rows * k);
    defer tail.deinit();
    const tail_count = try checkedQuantizedProduct(tail_rows, blocks_per_row);
    var tail_stack: [q8_0_lhs_stack_blocks]dtype_mod.BlockQ8_0 = undefined;
    const tail_blocks = if (tail_count <= tail_stack.len)
        tail_stack[0..tail_count]
    else
        try allocator.alloc(dtype_mod.BlockQ8_0, tail_count);
    defer if (tail_count > tail_stack.len) allocator.free(tail_blocks);

    try quantized_matmul.q8k.quantizeRowsQ8_0Into(tail_blocks, &tail);
    // The <=3-row remainder runs after the bulk kernel completes; the caller's
    // `pc` passes through so it can column-split like a decode-shaped matmul
    // (a parallel split never changes per-element math).
    vector.matmul_quant.matmul2DQ8_0x4RhsInto(pc, cd[bulk_rows * n .. m * n], tail_blocks, rhs, tail_rows, n, k);
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
    if (m % 4 != 0) return tensor.TensorError.InvalidShape;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try quantized_matmul.q8k.q8_0BlockCount(k);
    if (lhs_blocks.len != try checkedQuantizedProduct(m / 4, blocks_per_row)) return quantized_matmul.types.QuantizedFormatError.InvalidQuantizedLength;
    const cd = contiguousData(out, try checkedTensorProduct(m, n));
    vector.matmul_quant.matmul2DQ8_0x4PackedRhsInto(pc, cd, lhs_blocks, rhs, m, n, k);
}

// Pre-quantized-LHS K-quant GEMM entries for the fused split-activation ops:
// exec quantizes the activation rows itself there, so these skip the
// allocator-based LHS quantization of the matmul2DQuantizedRhs* wrappers.
pub fn matmulPackedQ4_Kx8Q8_Kx4Slice(pc: ParallelConfig, out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_Kx4, rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8, m: usize, n: usize, k: usize) void {
    vector.matmul_quant.matmul2DQ4_Kx8Q8_Kx4RhsInto(pc, out, lhs_blocks, rhs, m, n, k);
}

pub fn matmulPackedQ4_Kx8RowsSlice(pc: ParallelConfig, out: []f32, lhs_blocks: []const dtype_mod.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8, m: usize, n: usize, k: usize) void {
    vector.matmul_quant.matmul2DQ4_Kx8RhsInto(pc, out, lhs_blocks, rhs, m, n, k);
}

pub fn matmulPackedQ5_Kx8Q8_Kx4Slice(pc: ParallelConfig, out: []f32, lhs_blocks: []const quantized_matmul.BlockQ8_Kx4, rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8, m: usize, n: usize, k: usize) void {
    vector.matmul_quant.matmul2DQ5_Kx8Q8_Kx4RhsInto(pc, out, lhs_blocks, rhs, m, n, k);
}

pub fn matmulPackedQ5_Kx8RowsSlice(pc: ParallelConfig, out: []f32, lhs_blocks: []const dtype_mod.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8, m: usize, n: usize, k: usize) void {
    vector.matmul_quant.matmul2DQ5_Kx8RhsInto(pc, out, lhs_blocks, rhs, m, n, k);
}

pub fn matmulPackedQ6_Kx4RowsSlice(pc: ParallelConfig, out: []f32, lhs_blocks: []const dtype_mod.BlockQ8_K, rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_Kx4, m: usize, n: usize, k: usize) void {
    vector.matmul_quant.matmul2DQ6_Kx4RhsInto(pc, out, lhs_blocks, rhs, m, n, k);
}

// Single-row slice kernels for fused per-row activation math (exact same
// vector kernels the unfused elementwise ops apply).
pub fn unaryRowSlice(comptime op: ops.UnaryOp, z: []f32, x: []const f32) void {
    vector.primitives.vecUnary(op, z, x);
}

pub fn mulRowSlice(z: []f32, x: []const f32, y: []const f32) void {
    vector.primitives.vecMul(z, x, y);
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
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try quantized_matmul.q8k.q8_0BlockCount(k);
    if (lhs_blocks.len != try checkedQuantizedProduct((m + 3) / 4, blocks_per_row)) return quantized_matmul.types.QuantizedFormatError.InvalidQuantizedLength;
    const cd = contiguousData(out, try checkedTensorProduct(m, n));
    vector.matmul_quant.matmul2DQ8_0x4PackedPaddedRhsInto(pc, cd, lhs_blocks, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ2_K(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ2_K,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_KRows(pc, vector.matmul_quant.matmul2DQ2_KRhsInto, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ3_K(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ3_K,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_KRows(pc, vector.matmul_quant.matmul2DQ3_KRhsInto, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ4_K(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_K,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_KRows(pc, vector.matmul_quant.matmul2DQ4_KRhsInto, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ4_Kx4(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx4,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_KRows(pc, vector.matmul_quant.matmul2DQ4_Kx4RhsInto, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ4_Kx8(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_Kx4Prefix(
        pc,
        vector.matmul_quant.matmul2DQ4_Kx8Q8_Kx4RhsInto,
        vector.matmul_quant.matmul2DQ4_Kx8RhsInto,
        allocator,
        out,
        a,
        rhs,
        m,
        n,
        k,
        q4_k_x4_min_rows,
        true,
    );
}

pub fn matmul2DQuantizedRhsQ4_Kx2Mmla(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, try checkedTensorProduct(m, n));
    const blocks_per_row = try quantized_matmul.blockCountForDType(.q8_k, k);
    const prefix_rows = m - m % 2;

    if (prefix_rows != 0) {
        const qlhs_x2 = try allocator.alloc(quantized_matmul.BlockQ8_Kx2Mmla, try checkedQuantizedProduct(prefix_rows / 2, blocks_per_row));
        defer allocator.free(qlhs_x2);

        if (prefix_rows == m) {
            try quantized_matmul.q8k.quantizeRowsQ8_Kx2MmlaInto(qlhs_x2, a);
        } else {
            var prefix = try a.viewWithStridesOffset(&.{ prefix_rows, k }, &.{ k, 1 }, 0);
            defer prefix.deinit();
            try quantized_matmul.q8k.quantizeRowsQ8_Kx2MmlaInto(qlhs_x2, &prefix);
        }
        vector.matmul_quant.matmul2DQ4_Kx2MmlaQ8_Kx2MmlaRhsInto(pc, cd[0 .. prefix_rows * n], qlhs_x2, rhs, prefix_rows, n, k);
    }

    if (prefix_rows == m) return;

    var tail = try a.viewWithStridesOffset(&.{ m - prefix_rows, k }, &.{ k, 1 }, prefix_rows * k);
    defer tail.deinit();
    const tail_blocks = try quantized_matmul.q8k.quantizeRowsQ8_K(allocator, &tail);
    defer allocator.free(tail_blocks);
    const tail_pc = if (prefix_rows == 0) pc else ParallelConfig{};
    vector.matmul_quant.matmul2DQ4_Kx2MmlaRhsInto(tail_pc, cd[prefix_rows * n .. m * n], tail_blocks, rhs, m - prefix_rows, n, k);
}

pub fn matmul2DQuantizedRhsQ5_Kx8(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_Kx4Prefix(
        pc,
        vector.matmul_quant.matmul2DQ5_Kx8Q8_Kx4RhsInto,
        vector.matmul_quant.matmul2DQ5_Kx8RhsInto,
        allocator,
        out,
        a,
        rhs,
        m,
        n,
        k,
        q5_k_x4_prefix_min_rows,
        false,
    );
}

fn matmul2DQuantizedRhsQ8_Kx4Prefix(
    pc: ParallelConfig,
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
        return matmul2DQuantizedRhsQ8_KRows(pc, rows, allocator, out, a, rhs, m, n, k);
    }

    const row_groups = if (pad_x4_rows) (prefix_rows + 3) / 4 else prefix_rows / 4;
    const qlhs_x4 = try allocator.alloc(quantized_matmul.BlockQ8_Kx4, try checkedQuantizedProduct(row_groups, blocks_per_row));
    defer allocator.free(qlhs_x4);

    if (prefix_rows == m) {
        if (pad_x4_rows) {
            try quantized_matmul.q8k.quantizeRowsQ8_Kx4PaddedInto(qlhs_x4, a);
        } else {
            try quantized_matmul.q8k.quantizeRowsQ8_Kx4Into(qlhs_x4, a);
        }
    } else {
        var prefix = try a.viewWithStridesOffset(&.{ prefix_rows, k }, &.{ k, 1 }, 0);
        defer prefix.deinit();
        try quantized_matmul.q8k.quantizeRowsQ8_Kx4Into(qlhs_x4, &prefix);
    }
    x4(pc, cd[0 .. prefix_rows * n], qlhs_x4, rhs, prefix_rows, n, k);

    if (prefix_rows == m) return;

    var tail = try a.viewWithStridesOffset(&.{ m - prefix_rows, k }, &.{ k, 1 }, prefix_rows * k);
    defer tail.deinit();
    const tail_blocks = try quantized_matmul.q8k.quantizeRowsQ8_K(allocator, &tail);
    defer allocator.free(tail_blocks);
    // The <=3-row remainder runs after the x4 kernel completes; the caller's
    // `pc` passes through so it can column-split like a decode-shaped matmul
    // (a parallel split never changes per-element math).
    rows(pc, cd[prefix_rows * n .. m * n], tail_blocks, rhs, m - prefix_rows, n, k);
}

pub fn matmul2DQuantizedRhsQ5_K(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_K,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_KRows(pc, vector.matmul_quant.matmul2DQ5_KRhsInto, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ6_K(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_K,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_KRows(pc, vector.matmul_quant.matmul2DQ6_KRhsInto, allocator, out, a, rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ6_Kx4(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_Kx4,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return matmul2DQuantizedRhsQ8_KRows(pc, vector.matmul_quant.matmul2DQ6_Kx4RhsInto, allocator, out, a, rhs, m, n, k);
}

/// Prefill row count at/above which the table-decoded formats (iq*/fp4)
/// dequantize weight panels to f32 and ride BLAS, exactly like the Q2_0
/// arm above. Their int path pays a per-block table decode per
/// (weight-row, LHS-row) pair, so the dequant-once panel amortizes even
/// earlier than Q2_0's mul-free path — same accepted-numerics stance: the
/// BLAS arm consumes exact f32 activations, the int kernels keep decode
/// and short bursts (and every bitwise contract).
const table_blas_min_m: usize = 64;

fn TableDequantSliceTask(comptime rhs_dtype: DType) type {
    return struct {
        rhs: *const quantized_matmul.QuantizedMatmulRhsRowsFor(rhs_dtype),
        dst: []f32, // kc floats of weight row `row`, from block bi0
        row: usize,
        bi0: usize,

        fn run(task: *const @This()) void {
            const bs = comptime dtype_mod.blockSize(rhs_dtype);
            const blocks = task.rhs.columnBlocks(task.row);
            // Lengths are exact by construction (dst covers whole blocks),
            // so the only representable error cannot occur.
            quantized_matmul.dequantizeRowForDType(
                rhs_dtype,
                task.dst,
                blocks[task.bi0 .. task.bi0 + task.dst.len / bs],
            ) catch unreachable;
        }
    };
}

fn matmul2DQuantizedRhsTableBlas(
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
    const Task = TableDequantSliceTask(rhs_dtype);
    const bs = comptime dtype_mod.blockSize(rhs_dtype);
    const ad = contiguousDataConst(a, m * k);
    const cd = contiguousData(out, m * n);

    // k-slice width: whole blocks, full k when it fits the Q2_0 panel budget.
    const kp_max = @max(bs, (q2_0_blas_panel_floats / n) & ~(bs - 1));
    const kp = @min(k, kp_max);
    const panel = try allocator.alloc(f32, n * kp);
    defer allocator.free(panel);
    const tasks = try allocator.alloc(Task, n);
    defer allocator.free(tasks);

    var k0: usize = 0;
    while (k0 < k) : (k0 += kp) {
        const kc = @min(kp, k - k0);
        const bi0 = k0 / bs;
        for (0..n) |row| tasks[row] = .{ .rhs = rhs, .dst = panel[row * kc ..][0..kc], .row = row, .bi0 = bi0 };
        if (pc.pool) |pool| {
            pool.parallelChunks(Task, tasks, Task.run);
        } else {
            for (tasks) |*t| Task.run(t);
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

/// Prefill row count at/above which the folded tied-K=2 PTQTP path
/// (`tq2_0_fx4`) dequantizes weight panels to f32 and rides BLAS. The
/// mul-free ternary tile owns decode and short bursts — it beats a GEMM
/// there — but its 4-column pack has no AMX-class batch form, so past this
/// width the dequant-once panel plus sgemm wins on operand reuse. Accepted
/// numerics, like every BLAS arm: exact f32 activations instead of the
/// Q8_K-quantized ones the integer kernel consumes.
const folded_blas_min_m: usize = 64;

/// C[m, n] = A[m, k] · dequant(folded)ᵀ through BLAS, k-sliced so each
/// GEMM is full output width with a contiguous C (the Q2_0 arm's panel
/// discipline). Returns false when the caller must keep the integer path:
/// non-BLAS builds, m below the gate, or dimensions cblas cannot take.
pub fn matmulFoldedx4Blas(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: []f32,
    a: []const f32,
    folded: []const quantized_matmul.BlockTQ2_0Foldedx4,
    blocks_per_row: usize,
    m: usize,
    n: usize,
    k: usize,
) !bool {
    if (comptime !build_options.use_blas) return false;
    if (m < folded_blas_min_m or !fitsCblas(m, n, k)) return false;

    const Task = struct {
        folded: []const quantized_matmul.BlockTQ2_0Foldedx4,
        blocks_per_row: usize,
        dst: []f32,
        col: usize,
        bi0: usize,

        fn run(task: *const @This()) void {
            quantized_matmul.ternary.dequantizeFoldedx4ColumnInto(task.dst, task.folded, task.blocks_per_row, task.col, task.bi0);
        }
    };

    const kp_max = @max(@as(usize, 256), (q2_0_blas_panel_floats / n) & ~@as(usize, 255));
    const kp = @min(k, kp_max);
    const panel = try allocator.alloc(f32, n * kp);
    defer allocator.free(panel);
    const tasks = try allocator.alloc(Task, n);
    defer allocator.free(tasks);

    var k0: usize = 0;
    while (k0 < k) : (k0 += kp) {
        const kc = @min(kp, k - k0);
        const bi0 = k0 / 256;
        for (0..n) |col| tasks[col] = .{ .folded = folded, .blocks_per_row = blocks_per_row, .dst = panel[col * kc ..][0..kc], .col = col, .bi0 = bi0 };
        if (pc.pool) |pool| {
            pool.parallelChunks(Task, tasks, Task.run);
        } else {
            for (tasks) |*t| Task.run(t);
        }
        ensureBlasThreadsConfigured();
        cblas_sgemm(
            cblas_row_major,
            cblas_no_trans,
            cblas_trans,
            cDim(m),
            cDim(n),
            cDim(kc),
            1.0,
            a.ptr + k0,
            cDim(k),
            panel.ptr,
            cDim(kc),
            if (k0 == 0) 0.0 else 1.0,
            out.ptr,
            cDim(n),
        );
    }
    return true;
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
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    if (comptime build_options.use_blas) {
        if (m >= table_blas_min_m and fitsCblas(m, n, k)) {
            return matmul2DQuantizedRhsTableBlas(pc, rhs_dtype, allocator, out, a, rhs, m, n, k);
        }
    }

    const cd = contiguousData(out, m * n);
    var qlhs = try quantized_matmul.q8k.quantizeRowsQ8_0(allocator, a);
    defer qlhs.deinit();
    vector.matmul_quant.matmul2DTableQ8_0RhsInto(pc, rhs_dtype, cd, qlhs.blocks, rhs, m, n, k);
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
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    if (comptime build_options.use_blas) {
        if (m >= table_blas_min_m and fitsCblas(m, n, k)) {
            return matmul2DQuantizedRhsTableBlas(pc, rhs_dtype, allocator, out, a, rhs, m, n, k);
        }
    }

    const cd = contiguousData(out, m * n);
    const qlhs = try quantized_matmul.q8k.quantizeRowsQ8_K(allocator, a);
    defer allocator.free(qlhs);
    vector.matmul_quant.matmul2DTableQ8_KRhsInto(pc, rhs_dtype, cd, qlhs, rhs, m, n, k);
}

fn matmul2DQuantizedRhsTQ2_0(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.QuantizedMatmulRhsTQ2_0,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, m * n);
    const qlhs = try quantized_matmul.q8k.quantizeRowsQ8_K(allocator, a);
    defer allocator.free(qlhs);
    vector.matmul_quant.matmul2DTQ2_0RhsInto(pc, cd, qlhs, rhs, m, n, k);
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
    matmulTransA2DIntoUnchecked(.{}, out, a, b, m, n, k);
}

pub fn matmulTransA2DIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
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
                0.0,
                contiguousData(out, m * n),
            );
            return;
        }
    }
    vector.gemm.matmulTransA2DIntoUnchecked(pc, out, a, b, m, n, k);
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
    matmulTransB2DIntoUnchecked(.{}, out, a, b, m, n, k);
}

pub fn matmulTransB2DIntoUnchecked(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
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
                0.0,
                contiguousData(out, m * n),
            );
            return;
        }
    }
    vector.gemm.matmulTransB2DIntoUnchecked(pc, out, a, b, m, n, k);
}

pub fn matmulTransB2DIntoUncheckedF16Operands(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const tensor.TensorOf(.f16),
    b: *const tensor.TensorOf(.f16),
    m: usize,
    n: usize,
    k: usize,
) void {
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuF16ForRhs(b, m, n, k)) {
            if (gpu.gemmF16NtAsync(a, b, out, m, n, k)) return;
        }
    }
    vector.gemm.matmulTransB2DIntoUncheckedF16Operands(pc, out, a, b, m, n, k);
}

pub fn matmulTransB2DIntoUncheckedBf16Rhs(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    b: *const tensor.TensorOf(.bf16),
    m: usize,
    n: usize,
    k: usize,
) void {
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuBf16ForRhs(b, m, n, k)) {
            if (gpu.gemmBf16NtAsync(a, b, out, m, n, k)) return;
        }
    }
    vector.gemm.matmulTransB2DIntoUncheckedBf16Rhs(pc, out, a, b, m, n, k);
}

pub fn matmulBatched2DIntoUnchecked(
    pc: ParallelConfig,
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
    vector.batched.matmulBatched2DIntoUnchecked(pc, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c);
}

pub fn matmulBatchedTransA2DIntoUnchecked(
    pc: ParallelConfig,
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
    vector.batched.matmulBatchedTransA2DIntoUnchecked(pc, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c);
}

pub fn matmulBatchedTransB2DIntoUnchecked(
    pc: ParallelConfig,
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
    vector.batched.matmulBatchedTransB2DIntoUnchecked(pc, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c);
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
            0.0,
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
    beta: f32,
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
        beta,
        c.ptr,
        cDim(n),
    );
}

/// Raw strided sgemm for exec-layer fused kernels (the attention-backward
/// strip contractions): C[m,n] = alpha·op(A)·op(B) + beta·C, row-major,
/// every leading dimension explicit. BLAS builds only — callers comptime-gate
/// on `backend.native_uses_blas`.
pub fn sgemmStrided(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: []const f32,
    lda: usize,
    b: []const f32,
    ldb: usize,
    beta: f32,
    c: []f32,
    ldc: usize,
) void {
    if (comptime !build_options.use_blas) unreachable;
    ensureBlasThreadsConfigured();
    cblas_sgemm(
        cblas_row_major,
        if (trans_a) cblas_trans else cblas_no_trans,
        if (trans_b) cblas_trans else cblas_no_trans,
        cDim(m),
        cDim(n),
        cDim(k),
        alpha,
        a.ptr,
        cDim(lda),
        b.ptr,
        cDim(ldb),
        beta,
        c.ptr,
        cDim(ldc),
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
        .mkl => MKL_Set_Num_Threads(n),
        .nvpl => nvpl_blas_set_num_threads(n),
        .none, .accelerate, .blas => {},
    }
}

/// Token from `beginNestedBlasScope`, restored by `endNestedBlasScope`.
pub const NestedBlasScope = c_int;

/// Serialize THIS thread's BLAS calls for the duration of a fucina parallel
/// region, returning the token that restores the previous setting.
///
/// A self-threading BLAS spawns an engine team per call, so one issued from
/// inside our own parallel region puts two schedulers on the same cores. The
/// limit belongs to the nested caller, not the process: a single big top-level
/// GEMM still wants the engine's own threads, and it outruns our pool-parallel
/// packed path when it gets them.
///
/// Only MKL is covered — `MKL_Set_Num_Threads_Local` is per-thread and hands
/// back the previous value, which is exactly this contract. Providers exposing
/// only process-wide setters keep the engine's default threading.
pub fn beginNestedBlasScope() NestedBlasScope {
    if (comptime build_options.blas_kind != .mkl) return 0;
    return MKL_Set_Num_Threads_Local(1);
}

pub fn endNestedBlasScope(previous: NestedBlasScope) void {
    if (comptime build_options.blas_kind != .mkl) return;
    _ = MKL_Set_Num_Threads_Local(previous);
}

fn fitsCblas(m: usize, n: usize, k: usize) bool {
    return m <= max_cblas_dim and n <= max_cblas_dim and k <= max_cblas_dim;
}

/// With an already-packed dense RHS, skinny-m tall-k cells run faster on the
/// in-tree packed microkernel than through BLAS: measured on M1 Max
/// (Accelerate) at k in {4800, 5120, 9600}, m=16 runs 1.8-2.1x BLAS and
/// m=32 sits at parity-to-1.15x, while at k=64 (the wide-n unembed shape)
/// BLAS wins from m=16 up — hence the k floor. Scoped to Accelerate: the
/// OpenBLAS/x86 crossover is unmeasured.
fn packedDenseKernelPreferred(m: usize, k: usize) bool {
    if (comptime build_options.blas_kind != .accelerate) return false;
    return m < 32 and k >= 4096;
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
