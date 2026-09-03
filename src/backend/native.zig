//! The one CPU kernel provider: the `kernels` namespace below is the kernel
//! set (its declaration list is the interface; `backend.zig`'s
//! `conformKernels` checks it). Elementwise, reduction, conv, pool and
//! Winograd entries forward to the portable `@Vector` leaves in `vector/`;
//! the dense and quantized GEMM family is defined in this file and routes
//! each call across the GPU provider (`-Dgpu`), a CBLAS provider (`-Dblas`)
//! and the vector kernels (the quantized row-form and W8A8 bodies are
//! `vector/matmul_quant.zig`'s tensor-LHS entries; this file layers the
//! BLAS crossovers and the lane-packed-LHS forms on them). One kernel calls
//! BLAS from its own leaf instead: the attention backward's BLAS-strip
//! variant (`vector/attention.zig`), whose strips are the GEMMs. Every
//! pool-taking kernel takes `pc:
//! ParallelConfig` first. On `-Dbackend=scalar` builds (`isa.reference`)
//! the same entries select their scalar reference arms (the `scalar`
//! namespaces inside the `vector/` files) and the GPU/BLAS/lane-pack
//! dispatch below compiles out, so the reference leg is this provider too.
//! Layer stack: docs/ARCHITECTURE.md.
const std = @import("std");
const build_options = @import("build_options");
const blas = @import("blas.zig");
const isa = @import("isa.zig");
const dtype_mod = @import("../dtype.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");
const packed_matmul = @import("packed.zig");
const ops = @import("ops.zig");
const quantized_matmul = @import("quant.zig");
const vector = @import("vector.zig");
const vector_common = @import("vector/common.zig");
const gpu = @import("gpu.zig").impl;
const row_kernels = @import("vector/rows/kernels.zig");
const attention_kernels = @import("vector/attention/kernels.zig");

const native = @This();
const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;
const contiguousDataConst = vector_common.contiguousDataConst;
const contiguousData = vector_common.contiguousData;
const checkedQuantizedProduct = quantized_matmul.types.checkedProduct;

// The LHS quantization band (vector/matmul_quant.zig): the validated
// activation view, the Q8_0-family stack-or-heap block scratch, the
// pool-split row-range quantizer and the Q8_K row form.
const lhsRows = vector.matmul_quant.lhsRows;
const LhsBlocks = vector.matmul_quant.LhsBlocks;
const quantizeLhsUnits = vector.matmul_quant.quantizeLhsUnits;
const quantizedLhsQ8_K = vector.matmul_quant.quantizedLhsQ8_K;

// The numeric dispatch gates live in src/parallel.zig's policy table
// (values and measurement rationale there); aliased file-locally so the
// kernels below read bare names.

pub const ParallelConfig = vector.ParallelConfig;

/// The kernel set: the `vector/` leaf kernels by name and the GEMM family
/// defined below. The declaration list IS the interface (`backend.zig`'s
/// `conformKernels` derives its checks from it): a kernel that uses the
/// worker pool takes `pc: ParallelConfig` first and nowhere else, and one
/// whose first parameter is anything else is pool-free.
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
    pub const softcapContiguousIntoUnchecked = vector.elementwise.softcapContiguousIntoUnchecked;
    pub const clampContiguousIntoUnchecked = vector.elementwise.clampContiguousIntoUnchecked;
    pub const gatedContiguousIntoUnchecked = vector.elementwise.gatedContiguousIntoUnchecked;
    pub const sumInto = vector.elementwise.sumInto;
    pub const sumSlice = vector.elementwise.sumSlice;
    pub const prodInto = vector.elementwise.prodInto;
    pub const prodSlice = vector.elementwise.prodSlice;
    pub const sumSliceTyped = vector.elementwise.sumSliceTyped;
    pub const dot = native.dot;
    pub const gemm = native.gemm;
    pub const gemmBatched = native.gemmBatched;
    pub const packDenseRhs = native.packDenseRhs;
    pub const quantizeMatmulRhsBlockwiseI8 = native.quantizeMatmulRhsBlockwiseI8;
    pub const quantizeMatmulRhsQ4_0 = native.quantizeMatmulRhsQ4_0;
    pub const quantizeMatmulRhsQ8_0 = native.quantizeMatmulRhsQ8_0;
    pub const matmul2DQuantizedRhs = native.matmul2DQuantizedRhs;
    pub const matmulPacked = native.matmulPacked;
    pub const matmul2DPackedQ8_0x4LhsRhs = native.matmul2DPackedQ8_0x4LhsRhs;
    pub const matmul2DPackedPaddedQ8_0x4LhsRhs = native.matmul2DPackedPaddedQ8_0x4LhsRhs;
    pub const matmulPackedSlice = native.matmulPackedSlice;
    pub const unaryRowSlice = vector.elementwise.unaryRowSlice;
    pub const mulRowSlice = vector.elementwise.mulRowSlice;
    // The fused row kernels (vector/rows.zig): task-carrying serial
    // entries taking their row or lane range as parameters. The exec
    // domain modules split the ranges over the pool themselves
    // (`ExecContext.forRange`), so every entry here is pool-free by design.
    pub const softmaxRows = row_kernels.softmaxRows;
    pub const softmaxExtRows = row_kernels.softmaxExtRows;
    pub const softmaxBackwardRows = row_kernels.softmaxBackwardRows;
    pub const logsumexpRows = row_kernels.logsumexpRows;
    pub const logSoftmaxRows = row_kernels.logSoftmaxRows;
    pub const softmaxInner = row_kernels.softmaxInner;
    pub const logsumexpInner = row_kernels.logsumexpInner;
    pub const logSoftmaxInner = row_kernels.logSoftmaxInner;
    pub const softmaxBackwardInner = row_kernels.softmaxBackwardInner;
    pub const logSoftmaxBackwardRows = row_kernels.logSoftmaxBackwardRows;
    pub const logsumexpBackwardRows = row_kernels.logsumexpBackwardRows;
    pub const logSoftmaxBackwardInner = row_kernels.logSoftmaxBackwardInner;
    pub const logsumexpBackwardInner = row_kernels.logsumexpBackwardInner;
    pub const splitSwiGluRows = row_kernels.splitSwiGluRows;
    pub const splitGluRows = row_kernels.splitGluRows;
    pub const splitSwiGluBackwardRows = row_kernels.splitSwiGluBackwardRows;
    pub const splitGluBackwardRows = row_kernels.splitGluBackwardRows;
    pub const rmsNormMulRopeHalfVectors = row_kernels.rmsNormMulRopeHalfVectors;
    pub const rmsNormMulRows = row_kernels.rmsNormMulRows;
    pub const rmsNormMulAddRows = row_kernels.rmsNormMulAddRows;
    pub const rmsNormMulBackwardInputRows = row_kernels.rmsNormMulBackwardInputRows;
    pub const rmsNormMulBackwardWeightRows = row_kernels.rmsNormMulBackwardWeightRows;
    pub const rmsNormWeightGradBlocks = row_kernels.rmsNormWeightGradBlocks;
    pub const rmsNormWeightGradReduce = row_kernels.rmsNormWeightGradReduce;
    pub const rmsNormInner = row_kernels.rmsNormInner;
    pub const rmsNormBackwardInputInner = row_kernels.rmsNormBackwardInputInner;
    pub const rmsNormBackwardWeightInner = row_kernels.rmsNormBackwardWeightInner;
    pub const layerNormRows = row_kernels.layerNormRows;
    pub const layerNormBackwardInputRows = row_kernels.layerNormBackwardInputRows;
    pub const layerNormAffineParamGradRows = row_kernels.layerNormAffineParamGradRows;
    pub const layerNormRowStats = row_kernels.layerNormRowStats;
    pub const layerNormParamGradColumns = row_kernels.layerNormParamGradColumns;
    pub const layerNormInner = row_kernels.layerNormInner;
    pub const layerNormBackwardInner = row_kernels.layerNormBackwardInner;
    pub const varianceInner = row_kernels.varianceInner;
    pub const standardizeInner = row_kernels.standardizeInner;
    pub const standardizeBackwardInner = row_kernels.standardizeBackwardInner;
    pub const crossEntropyLossRows = row_kernels.crossEntropyLossRows;
    pub const crossEntropyBackwardRows = row_kernels.crossEntropyBackwardRows;
    pub const dropoutRange = row_kernels.dropoutRange;
    pub const scatterAddRows = row_kernels.scatterAddRows;
    // The grouped-causal attention kernels (vector/attention.zig):
    // task-carrying serial entries, pool-free like the row kernels.
    pub const groupedCausalAttentionHeads = attention_kernels.groupedCausalAttentionHeads;
    pub const groupedCausalAttentionHeadPairs = attention_kernels.groupedCausalAttentionHeadPairs;
    pub const groupedCausalAttentionQueryTiles = attention_kernels.groupedCausalAttentionQueryTiles;
    pub const groupedCausalAttentionMultiUnits = attention_kernels.groupedCausalAttentionMultiUnits;
    pub const groupedCausalAttentionBackwardKvHeads = attention_kernels.groupedCausalAttentionBackwardKvHeads;
    pub const groupedCausalAttentionBackwardTiles = attention_kernels.groupedCausalAttentionBackwardTiles;
    pub const groupedCausalAttentionBackwardBlasTiles = attention_kernels.groupedCausalAttentionBackwardBlasTiles;
    pub const attentionBackwardReduceRows = attention_kernels.attentionBackwardReduceRows;
    // Dtype cast rows (vector/elementwise.zig) and the straggler row
    // kernels (vector/rows.zig): extremum/variance rows, the fused-rope
    // pair strip, the gated vector scans, the masked-reduce select row.
    pub const castF32ToF16 = vector.elementwise.castF32ToF16;
    pub const castF16ToF32 = vector.elementwise.castF16ToF32;
    pub const castF32ToBf16 = vector.elementwise.castF32ToBf16;
    pub const castBf16ToF32 = vector.elementwise.castBf16ToF32;
    pub const extremumRowValue = row_kernels.extremumRowValue;
    pub const varianceRowsInto = row_kernels.varianceRowsInto;
    pub const ropeHalfPairsInto = row_kernels.ropeHalfPairsInto;
    pub const scanRows = row_kernels.scanRows;
    pub const scanColumns = row_kernels.scanColumns;
    pub const selectRow = row_kernels.selectRow;
    // The per-format quantized row/pack/tile kernels the exec MoE
    // streams, the gemma fused-MoE skeleton, the PTQTP/borrowed weight
    // containers and the ternary-LoRA backward reach (previously named
    // as quant/<fmt> members): registered here so every band above the
    // backend goes through the conformed table.
    pub const quantizeRowQ8_0Into = quantized_matmul.q8k.quantizeRowQ8_0Into;
    pub const quantizeRowQ8_0IntoUnchecked = quantized_matmul.q8k.quantizeRowQ8_0IntoUnchecked;
    pub const quantizeRowQ8_KInto = quantized_matmul.q8k.quantizeRowQ8_KInto;
    pub const quantizeRowQ8_KIntoUnchecked = quantized_matmul.q8k.quantizeRowQ8_KIntoUnchecked;
    pub const packRowsQ8_Kx4PaddedInto = quantized_matmul.q8k.packRowsQ8_Kx4PaddedInto;
    pub const dequantizeRowQ8_0Into = quantized_matmul.q8k.dequantizeRowQ8_0Into;
    pub const matmulQ8_0x4RhsTile = quantized_matmul.q8_0.matmulQ8_0x4RhsTile;
    pub const matmulQ8_0RhsTile = quantized_matmul.q8_0.matmulQ8_0RhsTile;
    pub const splitSwiGluRowInto = quantized_matmul.q8_0.splitSwiGluRowInto;
    pub const quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto = quantized_matmul.q8_0.quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto;
    pub const packMatmulRhsQ8_0x4 = quantized_matmul.q8_0.packMatmulRhsQ8_0x4;
    pub const matmulQ6_Kx4RhsTile = quantized_matmul.q6_k.matmulQ6_Kx4RhsTile;
    pub const matmulQ6_Kx4RhsPairTile = quantized_matmul.q6_k.matmulQ6_Kx4RhsPairTile;
    pub const matmulQ6_KRhsTile = quantized_matmul.q6_k.matmulQ6_KRhsTile;
    pub const matmulQ6_KRhsCompactColOuter = quantized_matmul.q6_k.matmulQ6_KRhsCompactColOuter;
    pub const matmulQ4_KRhsTile = quantized_matmul.q4_k.matmulQ4_KRhsTile;
    pub const matmulQ4_KRhsCompactColOuter = quantized_matmul.q4_k.matmulQ4_KRhsCompactColOuter;
    pub const matmulMXFP4RhsTile = quantized_matmul.mxfp4.matmulMXFP4RhsTile;
    pub const quantizedMatmulRhsTQ2_0FromBorrowedBlocks = quantized_matmul.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks;
    pub const quantizedMatmulRhsTQ2_0FromF32Absmean = quantized_matmul.ternary.quantizedMatmulRhsTQ2_0FromF32Absmean;
    pub const matmulTQ2_0RhsTile = quantized_matmul.ternary.matmulTQ2_0RhsTile;
    pub const matmulTQ2_0FoldedX4RhsTile = quantized_matmul.ternary.matmulTQ2_0FoldedX4RhsTile;
    pub const matmulTQ2_0X4RhsTile = quantized_matmul.ternary.matmulTQ2_0X4RhsTile;
    pub const matmulTQ2_0X4RhsTileAcc = quantized_matmul.ternary.matmulTQ2_0X4RhsTileAcc;
    pub const packMatmulRhsTQ2_0x4 = quantized_matmul.ternary.packMatmulRhsTQ2_0x4;
    pub const packMatmulRhsTQ2_0Foldedx4 = quantized_matmul.ternary.packMatmulRhsTQ2_0Foldedx4;
    pub const packMatmulRhsTQ2_0Foldedx4Into = quantized_matmul.ternary.packMatmulRhsTQ2_0Foldedx4Into;
    pub const packMatmulRhsTQ2_0FoldedRows = quantized_matmul.ternary.packMatmulRhsTQ2_0FoldedRows;
    pub const packMatmulRhsTQ2_0FoldedRowsFromX4 = quantized_matmul.ternary.packMatmulRhsTQ2_0FoldedRowsFromX4;
    pub const matmulTableQ8_KRhsTile = quantized_matmul.cold.matmulTableQ8_KRhsTile;
    pub const dequantizeRowTQ2_0Into = quantized_matmul.cold.dequantizeRowTQ2_0Into;
    /// The one parallel 2-D quantized GEMM entry over an `ops.QuantGemm`
    /// request (row/column task split, serial full tile otherwise); the
    /// autograd ternary-STE fallback states `.{ .weight = .tq2_0,
    /// .lhs = .f32 }` through it.
    pub const gemm2D = vector.matmul_quant.gemm2D;
};

/// Full dot product into the scalar `out`: f32 takes the dedicated f32
/// reduction, every other float dtype the typed one.
pub fn dot(
    pc: ParallelConfig,
    comptime dtype: DType,
    out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
) !void {
    if (comptime dtype == .f32) return vector.elementwise.dotInto(pc, out, a, b);
    return vector.elementwise.dotIntoTyped(pc, dtype, out, a, b);
}

/// The dense GEMM (`ops.Gemm`). The f32 family routes GPU (store only: the
/// async GPU GEMM overwrites its destination and has no accumulate seam)
/// -> BLAS (beta = 1 for accumulate) -> the vector kernels; the
/// mixed-precision `.trans_b` streams route GPU -> vector; the typed NN
/// family is vector-only.
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
    if (comptime g.isF32() and !isa.reference) {
        if (comptime build_options.use_gpu and !g.accumulate) {
            if (gpu.shouldUseGpuForRhs(b, m, n, k)) {
                if (gpu.gemmF32Async(g.kind, a, b, out, m, n, k)) return;
            }
        }
        if (comptime build_options.use_blas) {
            if (shouldUseBlas(m, n, k)) {
                blas.gemm(
                    g.kind,
                    m,
                    n,
                    k,
                    contiguousDataConst(a, m * k),
                    contiguousDataConst(b, k * n),
                    if (g.accumulate) 1.0 else 0.0,
                    contiguousData(out, m * n),
                );
                return;
            }
        }
    } else if (comptime g.kind == .trans_b and g.a == .f16 and g.b == .f16 and g.out == .f32) {
        if (comptime build_options.use_gpu and !isa.reference) {
            if (gpu.shouldUseGpuF16ForRhs(b, m, n, k)) {
                if (gpu.gemmF16NtAsync(a, b, out, m, n, k)) return;
            }
        }
    } else if (comptime g.kind == .trans_b and g.a == .f32 and g.b == .bf16 and g.out == .f32) {
        if (comptime build_options.use_gpu and !isa.reference) {
            if (gpu.shouldUseGpuBf16ForRhs(b, m, n, k)) {
                if (gpu.gemmBf16NtAsync(a, b, out, m, n, k)) return;
            }
        }
    }
    vector.gemm.gemm(
        pc,
        g,
        vector_common.contiguousDataOf(g.out, out, m * n),
        vector_common.contiguousDataConstOf(g.a, a, m * k),
        vector_common.contiguousDataConstOf(g.b, b, k * n),
        m,
        n,
        k,
    );
}

/// Build the f32 output-row panels (`PackedDenseRhs`) from an f32, f16, or
/// bf16 `[n, k]` weight.
pub fn packDenseRhs(
    comptime dtype: DType,
    allocator: std.mem.Allocator,
    rhs: *const tensor.TensorOf(dtype),
) !packed_matmul.PackedDenseRhs {
    return packed_matmul.packDenseRhs(allocator, dtype, rhs);
}

/// f32 [m, k] x the f32 output-row panels -> f32 [m, n]. Explicit
/// packed-op decision table: GPU always wins; BLAS keeps its established
/// all-dimensions>=16 cells EXCEPT the skinny-m tall-k band, where the
/// already-packed microkernel is faster than Accelerate; the packed
/// microkernel also owns the m<16 cliff and every no-BLAS cell.
fn matmulPackedDense(
    pc: ParallelConfig,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const packed_matmul.PackedDenseRhs,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    // The reference arm: the scalar panel walk beside the pack constructor.
    if (comptime isa.reference) return packed_matmul.matmulDenseScalar(contiguousData(out, m * n), contiguousDataConst(a, m * k), rhs, m);
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuForRhs(&rhs.rhs, m, n, k)) {
            if (gpu.gemmF32Async(.trans_b, a, &rhs.rhs, out, m, n, k)) return;
        }
    }
    if (comptime build_options.use_blas) {
        if (shouldUseBlas(m, n, k) and !blas.packedDenseKernelPreferred(m, k)) {
            blas.gemm(
                .trans_b,
                m,
                n,
                k,
                contiguousDataConst(a, m * k),
                contiguousDataConst(&rhs.rhs, rhs.padded_n * k),
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

pub fn quantizeMatmulRhsBlockwiseI8(
    allocator: std.mem.Allocator,
    rhs: *const Tensor,
    group_size: usize,
) !quantized_matmul.types.QuantizedMatmulRhsI8 {
    return quantized_matmul.quantizeRhsBlockwiseI8(allocator, rhs, group_size);
}

pub fn quantizeMatmulRhsQ4_0(
    allocator: std.mem.Allocator,
    rhs: *const Tensor,
) !quantized_matmul.types.QuantizedMatmulRhsQ4_0 {
    return quantized_matmul.cold.quantizeMatmulRhsQ4_0(allocator, rhs);
}

pub fn quantizeMatmulRhsQ8_0(
    allocator: std.mem.Allocator,
    rhs: *const Tensor,
) !quantized_matmul.types.QuantizedMatmulRhsQ8_0 {
    return quantized_matmul.quantizeMatmulRhsQ8_0(allocator, rhs);
}

pub fn matmul2DQuantizedRhs(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: quantized_matmul.types.AnyQuantizedMatmulRhs,
    m: usize,
    n: usize,
    k: usize,
) !void {
    // The reference arm: the serial row-form dispatch with the two bespoke
    // reference kernels (Q2_0 ref twin, TQ2_0 table row).
    if (comptime isa.reference) return vector.matmul_quant.scalar.matmul2DQuantizedRhs(allocator, out, a, rhs, m, n, k);
    return switch (rhs) {
        .fucina_w8a8_rhs => |qrhs| matmul2DQuantizedRhsI8(pc, allocator, out, a, qrhs, m, n, k),
        // Q2_0 keeps its BLAS dequant-crossover arm.
        .q2_0 => |qrhs| matmul2DQuantizedRhsQ2_0(pc, allocator, out, a, qrhs, m, n, k),
        inline else => |qrhs, tag| matmulQuantRows(pc, comptime ops.QuantGemm.rowsFor(@field(DType, @tagName(tag))), allocator, out, a, qrhs, m, n, k),
    };
}

/// The int8 W8A8 entry (`vector.matmul_quant`'s; the pool split is there).
pub const matmul2DQuantizedRhsI8 = vector.matmul_quant.matmul2DQuantizedRhsI8;

/// The Q8_0x4 GEMM request family (`ops.QuantGemm`): the exact packed
/// form, the padded (`.col_outer`, masked-writes) form, and the per-row
/// Q8_0-LHS fallback.
const q8_0x4_packed_gemm: ops.QuantGemm = .{ .weight = .q8_0, .rhs = .x4, .lhs = .q8_0x4 };
const q8_0x4_padded_gemm: ops.QuantGemm = .{ .weight = .q8_0, .rhs = .x4, .lhs = .q8_0x4, .order = .col_outer };
const q8_0x4_rows_gemm: ops.QuantGemm = .{ .weight = .q8_0, .rhs = .x4, .lhs = .q8_0 };

/// The weights with a BLAS dequant-crossover for batched m (the
/// table-decoded families; policy gate `table_blas_min_m`).
fn tableBlasFormat(comptime dt: DType) bool {
    return switch (dt) {
        .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_nl, .iq4_xs, .tq1_0, .mxfp4, .nvfp4 => true,
        else => false,
    };
}

/// f32 [m, k] x a quantized RHS container -> f32 [m, n] over the request
/// `g`: the table families' BLAS crossover first, then the one row-form
/// body (`vector.matmul_quant.matmulQuantRows`: one LHS row quantization
/// into `g.lhs`'s block form, then `gemm2D`; serial on the reference
/// build).
fn matmulQuantRows(
    pc: ParallelConfig,
    comptime g: ops.QuantGemm,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: ops.RhsOf(g),
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (comptime build_options.use_blas and !isa.reference and tableBlasFormat(g.weight)) {
        if (m >= table_blas_min_m and blas.fitsCblas(m, n, k)) {
            return matmul2DQuantizedRhsTableBlas(pc, g.weight, allocator, out, a, rhs, m, n, k);
        }
    }
    return vector.matmul_quant.matmulQuantRows(pc, g, allocator, out, a, rhs, m, n, k);
}

/// Q2_0 BLAS crossover and panel budget: src/parallel.zig's policy table.
const q2_0_blas_min_m = parallel.q2_0_blas_min_m;
const q2_0_blas_panel_floats = parallel.q2_0_blas_panel_floats;

/// C[m, n] = A[m, k] · dequant(RHS)ᵀ through BLAS, k-sliced in whole
/// `block_size`-element blocks (`q2_0_blas_panel_floats` is the panel
/// budget) so each GEMM is full output width with a contiguous C: every
/// slice dequantizes its n weight columns into the f32 panel over the pool
/// (`dequantColumn(rhs, dst, col, bi0)` writes `dst.len` floats of column
/// `col` from block `bi0`), then GEMMs it in.
fn matmulDequantPanelBlas(
    pc: ParallelConfig,
    comptime block_size: usize,
    comptime Rhs: type,
    comptime dequantColumn: fn (Rhs, []f32, usize, usize) void,
    allocator: std.mem.Allocator,
    cd: []f32,
    ad: []const f32,
    rhs: Rhs,
    m: usize,
    n: usize,
    k: usize,
) !void {
    const Task = struct {
        rhs: Rhs,
        dst: []f32,
        col: usize,
        bi0: usize,

        fn run(task: *const @This()) void {
            dequantColumn(task.rhs, task.dst, task.col, task.bi0);
        }
    };
    // k-slice width: whole blocks, full k when it fits the budget.
    const kp_max = @max(block_size, (q2_0_blas_panel_floats / n) & ~(block_size - 1));
    const kp = @min(k, kp_max);
    const panel = try allocator.alloc(f32, n * kp);
    defer allocator.free(panel);
    const tasks = try allocator.alloc(Task, n);
    defer allocator.free(tasks);

    var k0: usize = 0;
    while (k0 < k) : (k0 += kp) {
        const kc = @min(kp, k - k0);
        const bi0 = k0 / block_size;
        for (0..n) |col| tasks[col] = .{ .rhs = rhs, .dst = panel[col * kc ..][0..kc], .col = col, .bi0 = bi0 };
        if (pc.pool) |pool| {
            pool.parallelChunks(Task, tasks, Task.run);
        } else {
            for (tasks) |*t| Task.run(t);
        }
        // C (m x n, full width) += A[:, k0..k0+kc] x panel^T (kc x n).
        blas.gemmStrided(false, true, m, n, kc, 1.0, ad[k0..], k, panel, kc, if (k0 == 0) 0.0 else 1.0, cd, n);
    }
}

/// The `dequantColumn` of a `columnBlocks(col)` row container over a
/// whole-block row decoder `dequantRow(dst, blocks) !void`.
fn columnDequant(comptime Rhs: type, comptime block_size: usize, comptime dequantRow: anytype) fn (Rhs, []f32, usize, usize) void {
    return struct {
        fn run(rhs: Rhs, dst: []f32, col: usize, bi0: usize) void {
            // Lengths are exact by construction (dst covers whole blocks),
            // so the only representable error cannot occur.
            dequantRow(dst, rhs.columnBlocks(col)[bi0 .. bi0 + dst.len / block_size]) catch unreachable;
        }
    }.run;
}

/// Q2_0 dequantizes through the vectorized row decoder (the scalar one
/// would dominate the GEMM).
fn matmul2DQuantizedRhsQ2_0Blas(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.types.QuantizedMatmulRhsQ2_0,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const bs = dtype_mod.q2_0_block_size;
    const Rhs = *const quantized_matmul.types.QuantizedMatmulRhsQ2_0;
    return matmulDequantPanelBlas(pc, bs, Rhs, columnDequant(Rhs, bs, quantized_matmul.ternary.dequantizeRowQ2_0FastInto), allocator, contiguousData(out, m * n), contiguousDataConst(a, m * k), rhs, m, n, k);
}

pub fn matmul2DQuantizedRhsQ2_0(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.types.QuantizedMatmulRhsQ2_0,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (comptime isa.reference) return vector.matmul_quant.scalar.matmul2DQuantizedRhsQ2_0(allocator, out, a, rhs, m, n, k);
    if (comptime build_options.use_blas) {
        if (m >= q2_0_blas_min_m) {
            return matmul2DQuantizedRhsQ2_0Blas(pc, allocator, out, a, rhs, m, n, k);
        }
    }
    return matmulQuantRows(pc, comptime ops.QuantGemm.rowsFor(.q2_0), allocator, out, a, rhs, m, n, k);
}

fn matmulPackedQ8_0x4(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.types.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    if (m % 4 != 0) {
        if (m >= 12 and m < parallel.vector_column_min_m) {
            const cd = contiguousData(out, m * n);
            const blocks_per_row = try quantized_matmul.types.blockCountForDType(.q8_0, k);
            const row_groups = (m + 3) / 4;
            var scratch: LhsBlocks(quantized_matmul.types.BlockQ8_0x4) = undefined;
            const qlhs_blocks = try scratch.acquire(allocator, row_groups * blocks_per_row);
            defer scratch.release(allocator, qlhs_blocks);

            quantizeLhsUnits(pc, quantized_matmul.types.BlockQ8_0x4, quantized_matmul.q8_0.quantizeRowsQ8_0x4PaddedGroupsInto, qlhs_blocks, try lhsRows(a, m, k), m, k, blocks_per_row, row_groups);
            vector.matmul_quant.gemm2D(pc, q8_0x4_padded_gemm, cd, qlhs_blocks, rhs, m, n, k);
            return;
        }
        if (m >= parallel.vector_column_min_m) {
            return matmul2DQuantizedRhsQ8_0x4BulkTail(pc, allocator, out, a, rhs, m, n, k);
        }
        return matmulQuantRows(pc, q8_0x4_rows_gemm, allocator, out, a, rhs, m, n, k);
    }

    const cd = contiguousData(out, m * n);
    const blocks_per_row = try quantized_matmul.types.blockCountForDType(.q8_0, k);
    var scratch: LhsBlocks(quantized_matmul.types.BlockQ8_0x4) = undefined;
    const qlhs_blocks = try scratch.acquire(allocator, (m / 4) * blocks_per_row);
    defer scratch.release(allocator, qlhs_blocks);

    quantizeLhsUnits(pc, quantized_matmul.types.BlockQ8_0x4, quantized_matmul.q8_0.quantizeRowsQ8_0x4GroupsInto, qlhs_blocks, try lhsRows(a, m, k), m, k, blocks_per_row, m / 4);
    vector.matmul_quant.gemm2D(pc, q8_0x4_packed_gemm, cd, qlhs_blocks, rhs, m, n, k);
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
    rhs: *const quantized_matmul.types.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) !void {
    const cd = contiguousData(out, try tensor.checkedProduct(m, n));
    const ad = try lhsRows(a, m, k);
    const blocks_per_row = try quantized_matmul.types.blockCountForDType(.q8_0, k);
    const bulk_rows = m - m % 4;

    {
        var scratch: LhsBlocks(quantized_matmul.types.BlockQ8_0x4) = undefined;
        const qlhs_blocks = try scratch.acquire(allocator, try checkedQuantizedProduct(bulk_rows / 4, blocks_per_row));
        defer scratch.release(allocator, qlhs_blocks);

        quantizeLhsUnits(pc, quantized_matmul.types.BlockQ8_0x4, quantized_matmul.q8_0.quantizeRowsQ8_0x4GroupsInto, qlhs_blocks, ad[0 .. bulk_rows * k], bulk_rows, k, blocks_per_row, bulk_rows / 4);
        vector.matmul_quant.gemm2D(pc, q8_0x4_packed_gemm, cd[0 .. bulk_rows * n], qlhs_blocks, rhs, bulk_rows, n, k);
    }

    const tail_rows = m - bulk_rows;
    var tail_scratch: LhsBlocks(dtype_mod.BlockQ8_0) = undefined;
    const tail_blocks = try tail_scratch.acquire(allocator, try checkedQuantizedProduct(tail_rows, blocks_per_row));
    defer tail_scratch.release(allocator, tail_blocks);

    quantizeLhsUnits(pc, dtype_mod.BlockQ8_0, quantized_matmul.q8k.quantizeRowsQ8_0RangeInto, tail_blocks, ad[bulk_rows * k .. m * k], tail_rows, k, blocks_per_row, tail_rows);
    // The <=3-row remainder runs after the bulk kernel completes; the caller's
    // `pc` passes through so it can column-split like a decode-shaped matmul
    // (a parallel split never changes per-element math).
    vector.matmul_quant.gemm2D(pc, q8_0x4_rows_gemm, cd[bulk_rows * n .. m * n], tail_blocks, rhs, tail_rows, n, k);
}

pub fn matmul2DPackedQ8_0x4LhsRhs(
    pc: ParallelConfig,
    out: *Tensor,
    lhs_blocks: []const quantized_matmul.types.BlockQ8_0x4,
    rhs: *const quantized_matmul.types.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (m % 4 != 0) return tensor.TensorError.InvalidShape;
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try quantized_matmul.types.blockCountForDType(.q8_0, k);
    if (lhs_blocks.len != try checkedQuantizedProduct(m / 4, blocks_per_row)) return quantized_matmul.types.QuantizedFormatError.InvalidQuantizedLength;
    const cd = contiguousData(out, try tensor.checkedProduct(m, n));
    vector.matmul_quant.gemm2D(pc, q8_0x4_packed_gemm, cd, lhs_blocks, rhs, m, n, k);
}

/// The padded twin: `ceil(m / 4)` LHS groups, masked writes, any `m`.
pub fn matmul2DPackedPaddedQ8_0x4LhsRhs(
    pc: ParallelConfig,
    out: *Tensor,
    lhs_blocks: []const quantized_matmul.types.BlockQ8_0x4,
    rhs: *const quantized_matmul.types.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const blocks_per_row = try quantized_matmul.types.blockCountForDType(.q8_0, k);
    if (lhs_blocks.len != try checkedQuantizedProduct((m + 3) / 4, blocks_per_row)) return quantized_matmul.types.QuantizedFormatError.InvalidQuantizedLength;
    const cd = contiguousData(out, try tensor.checkedProduct(m, n));
    vector.matmul_quant.gemm2D(pc, q8_0x4_padded_gemm, cd, lhs_blocks, rhs, m, n, k);
}

/// Pre-quantized-LHS K-quant GEMM entry for the fused split-activation
/// ops: exec quantizes the activation rows itself there, so this skips the
/// allocator-based LHS quantization of the tensor-LHS entries. The
/// (LHS block, RHS container) type pair selects the kernel at comptime:
/// a `BlockQ8_Kx4` LHS takes the lane-packed x4 kernel, `BlockQ8_K` rows
/// take the per-row kernel.
pub fn matmulPackedSlice(pc: ParallelConfig, out: []f32, lhs_blocks: anytype, rhs: anytype, m: usize, n: usize, k: usize) void {
    const Lhs = @typeInfo(@TypeOf(lhs_blocks)).pointer.child;
    const Rhs = @TypeOf(rhs.*);
    const x4_lhs = Lhs == quantized_matmul.types.BlockQ8_Kx4;
    comptime if (!x4_lhs and Lhs != dtype_mod.BlockQ8_K)
        @compileError("matmulPackedSlice: unsupported LHS block type " ++ @typeName(Lhs));
    vector.matmul_quant.gemm2D(pc, comptime .{ .weight = Rhs.dtype, .rhs = Rhs.pack, .lhs = if (x4_lhs) .q8_kx4 else .q8_k }, out, lhs_blocks, rhs, m, n, k);
}

/// f32 [m, k] x a plain (compact `.rows`) quantized RHS [n, k] -> f32
/// [m, n] for a comptime weight dtype: the `QuantGemm.rowsFor(dt)`
/// selection (one LHS quantization, then the format's row kernel).
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
    return matmulQuantRows(pc, comptime ops.QuantGemm.rowsFor(dt), allocator, out, a, rhs, m, n, k);
}

/// f32 activations x a pre-packed RHS -> [m, n]; the container type selects
/// the arm at comptime: the f32 output-row panel (`packDenseRhs`) or a
/// lane-packed quantized container (each arm keeps its exact dispatch: the
/// Q8_0x4 bulk/tail split, the Q4_Kx8/Q5_Kx8 x4-prefix split, the smmla
/// pair path).
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
    if (comptime Rhs == packed_matmul.PackedDenseRhs)
        return matmulPackedDense(pc, out, a, rhs, m, n, k);
    // The reference arm: every quantized lane pack maps back onto its
    // row-form request, serial (the lane-packed-LHS dispatch below is a
    // native-tier layout choice, not part of the reference semantics).
    if (comptime isa.reference)
        return vector.matmul_quant.scalar.matmulPacked(allocator, out, a, rhs, m, n, k);
    if (comptime Rhs.pack == .x4 and Rhs.dtype == .q8_0)
        return matmulPackedQ8_0x4(pc, allocator, out, a, rhs, m, n, k);
    if (comptime Rhs.pack == .x4 and (Rhs.dtype == .q4_k or Rhs.dtype == .q6_k))
        return matmulQuantRows(pc, comptime .{ .weight = Rhs.dtype, .rhs = .x4, .lhs = .q8_k }, allocator, out, a, rhs, m, n, k);
    if (comptime Rhs.pack == .x8)
        return matmul2DQuantizedRhsQ8_Kx4Prefix(pc, comptime .{ .weight = Rhs.dtype, .rhs = .x8, .lhs = .q8_kx4 }, allocator, out, a, rhs, m, n, k);
    if (comptime Rhs.pack == .x2mmla and Rhs.dtype == .q4_k)
        return matmulPackedQ4_Kx2Mmla(pc, allocator, out, a, rhs, m, n, k);
    @compileError("matmulPacked: no kernel for packed RHS container " ++ @typeName(Rhs));
}

fn matmulPackedQ4_Kx2Mmla(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.types.QuantizedMatmulRhsQ4_Kx2Mmla,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, try tensor.checkedProduct(m, n));
    const ad = try lhsRows(a, m, k);
    const blocks_per_row = try quantized_matmul.types.blockCountForDType(.q8_k, k);
    const prefix_rows = m - m % 2;

    if (prefix_rows != 0) {
        const qlhs_x2 = try allocator.alloc(quantized_matmul.types.BlockQ8_Kx2Mmla, try checkedQuantizedProduct(prefix_rows / 2, blocks_per_row));
        defer allocator.free(qlhs_x2);

        quantizeLhsUnits(pc, quantized_matmul.types.BlockQ8_Kx2Mmla, quantized_matmul.q8k.quantizeRowsQ8_Kx2MmlaGroupsInto, qlhs_x2, ad[0 .. prefix_rows * k], prefix_rows, k, blocks_per_row, prefix_rows / 2);
        vector.matmul_quant.gemm2D(pc, comptime .{ .weight = .q4_k, .rhs = .x2mmla, .lhs = .q8_kx2mmla }, cd[0 .. prefix_rows * n], qlhs_x2, rhs, prefix_rows, n, k);
    }

    if (prefix_rows == m) return;

    const tail_pc = if (prefix_rows == 0) pc else ParallelConfig{};
    const tail_blocks = try quantizedLhsQ8_K(tail_pc, allocator, ad[prefix_rows * k .. m * k], m - prefix_rows, k);
    defer allocator.free(tail_blocks);
    vector.matmul_quant.gemm2D(tail_pc, comptime .{ .weight = .q4_k, .rhs = .x2mmla, .lhs = .q8_k }, cd[prefix_rows * n .. m * n], tail_blocks, rhs, m - prefix_rows, n, k);
}

/// The x8-pack dispatch: the `.q8_kx4` prefix (lane-packed LHS groups)
/// plus the `.q8_k` per-row remainder, over the request `g` (the
/// `.q8_kx4` form). The prefix is `quant.x4PrefixRows`'s, the one policy
/// the exec fused engine applies too: Q4_K pads every row through the
/// padded group quantizer, Q5_K keeps the unpadded bulk + per-row tail
/// split.
fn matmul2DQuantizedRhsQ8_Kx4Prefix(
    pc: ParallelConfig,
    comptime g: ops.QuantGemm,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: ops.RhsOf(g),
    m: usize,
    n: usize,
    k: usize,
) !void {
    const pad_x4_rows = comptime g.weight == .q4_k;
    const rows_g = comptime ops.QuantGemm{ .weight = g.weight, .rhs = g.rhs, .lhs = .q8_k };
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, try tensor.checkedProduct(m, n));
    const blocks_per_row = try quantized_matmul.types.blockCountForDType(.q8_k, k);
    const prefix_rows = quantized_matmul.x4PrefixRows(g.weight, m);

    if (prefix_rows == 0) {
        return matmulQuantRows(pc, rows_g, allocator, out, a, rhs, m, n, k);
    }

    const row_groups = if (pad_x4_rows) (prefix_rows + 3) / 4 else prefix_rows / 4;
    const qlhs_x4 = try allocator.alloc(quantized_matmul.types.BlockQ8_Kx4, try checkedQuantizedProduct(row_groups, blocks_per_row));
    defer allocator.free(qlhs_x4);

    // prefix_rows is a multiple of 4 unless the padded x4 kernel takes every
    // row, so the group walk's final-group lane padding is exactly the
    // padded form there and a no-op otherwise.
    const ad = try lhsRows(a, m, k);
    quantizeLhsUnits(pc, quantized_matmul.types.BlockQ8_Kx4, quantized_matmul.q8k.quantizeRowsQ8_Kx4GroupsInto, qlhs_x4, ad[0 .. prefix_rows * k], prefix_rows, k, blocks_per_row, row_groups);
    vector.matmul_quant.gemm2D(pc, g, cd[0 .. prefix_rows * n], qlhs_x4, rhs, prefix_rows, n, k);

    if (prefix_rows == m) return;

    const tail_blocks = try quantizedLhsQ8_K(pc, allocator, ad[prefix_rows * k .. m * k], m - prefix_rows, k);
    defer allocator.free(tail_blocks);
    // The <=3-row remainder runs after the x4 kernel completes; the caller's
    // `pc` passes through so it can column-split like a decode-shaped matmul
    // (a parallel split never changes per-element math).
    vector.matmul_quant.gemm2D(pc, rows_g, cd[prefix_rows * n .. m * n], tail_blocks, rhs, m - prefix_rows, n, k);
}

/// Table-format BLAS crossover: src/parallel.zig's policy table.
const table_blas_min_m = parallel.table_blas_min_m;

fn matmul2DQuantizedRhsTableBlas(
    pc: ParallelConfig,
    comptime rhs_dtype: DType,
    allocator: std.mem.Allocator,
    out: *Tensor,
    a: *const Tensor,
    rhs: *const quantized_matmul.types.QuantizedMatmulRhsRowsFor(rhs_dtype),
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    const bs = comptime dtype_mod.blockSize(rhs_dtype);
    const Rhs = *const quantized_matmul.types.QuantizedMatmulRhsRowsFor(rhs_dtype);
    const dequantRow = struct {
        fn run(dst: []f32, blocks: []const dtype_mod.Storage(rhs_dtype)) !void {
            return quantized_matmul.dequantizeRowForDType(rhs_dtype, dst, blocks);
        }
    }.run;
    return matmulDequantPanelBlas(pc, bs, Rhs, columnDequant(Rhs, bs, dequantRow), allocator, contiguousData(out, m * n), contiguousDataConst(a, m * k), rhs, m, n, k);
}

/// Folded-PTQTP BLAS crossover: src/parallel.zig's policy table.
const folded_blas_min_m = parallel.folded_blas_min_m;

/// C[m, n] = A[m, k] · dequant(folded)ᵀ through the BLAS dequant panel.
/// Returns false when the caller must keep the integer path: non-BLAS
/// builds, m below the gate, or dimensions cblas cannot take.
pub fn matmulFoldedx4Blas(
    pc: ParallelConfig,
    allocator: std.mem.Allocator,
    out: []f32,
    a: []const f32,
    folded: []const quantized_matmul.types.BlockTQ2_0Foldedx4,
    blocks_per_row: usize,
    m: usize,
    n: usize,
    k: usize,
) !bool {
    if (comptime !build_options.use_blas) return false;
    if (m < folded_blas_min_m or !blas.fitsCblas(m, n, k)) return false;

    const Folded = struct {
        blocks: []const quantized_matmul.types.BlockTQ2_0Foldedx4,
        blocks_per_row: usize,

        fn dequantColumn(rhs: @This(), dst: []f32, col: usize, bi0: usize) void {
            quantized_matmul.ternary.dequantizeFoldedx4ColumnInto(dst, rhs.blocks, rhs.blocks_per_row, col, bi0);
        }
    };
    try matmulDequantPanelBlas(pc, comptime dtype_mod.blockSize(.tq2_0), Folded, Folded.dequantColumn, allocator, out, a, .{ .blocks = folded, .blocks_per_row = blocks_per_row }, m, n, k);
    return true;
}

/// Batched dense f32 GEMM over `kind` (`ops.MatmulKind`), strides in
/// elements (0 = shared across batches): GPU -> BLAS -> the vector kernels.
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
    if (batch_count == 0) return;
    if (comptime build_options.use_gpu and !isa.reference) {
        // One GPU dispatch covering all batch_count matrices (grid depth =
        // batch); false when the GPU did not run.
        if (gpu.shouldUseGpuBatchedForRhs(b, m, n, k, batch_count)) {
            if (gpu.gemmBatchedF32Async(kind, a, b, out, m, n, k, batch_count, stride_a, stride_b, stride_c)) return;
        }
    }
    if (comptime build_options.use_blas and !isa.reference) {
        if (shouldUseBatchedBlas(m, n, k, batch_count)) {
            blasBatched(pc, kind, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c);
            return;
        }
    }
    vector.batched.gemmBatched(
        pc,
        kind,
        contiguousData(out, out.buffer.data.len - out.offset),
        contiguousDataConst(a, a.buffer.data.len - a.offset),
        contiguousDataConst(b, b.buffer.data.len - b.offset),
        m,
        n,
        k,
        batch_count,
        stride_a,
        stride_b,
        stride_c,
    );
}

/// The BLAS arm of the batched GEMM: batch ranges split over the worker
/// team (the same `batchedThreadCount` gate as the vector arm), each range
/// running its batches through BLAS on its own thread, so cross-batch
/// parallelism does not depend on the caller chunking the loop.
fn blasBatched(
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
    a.buffer.waitReady();
    b.buffer.waitReady();
    out.buffer.waitMutable();
    const batches: BlasBatches = .{
        .ap = a.buffer.data[a.offset..].ptr,
        .bp = b.buffer.data[b.offset..].ptr,
        .cp = out.buffer.data[out.offset..].ptr,
        .m = m,
        .n = n,
        .k = k,
        .stride_a = stride_a,
        .stride_b = stride_b,
        .stride_c = stride_c,
    };
    const run = blasBatchRange(kind);
    if (pc.pool) |pool| {
        const per_batch = parallel.saturatedMul3(m, n, k);
        if (per_batch < parallel.blas_batch_split_max_work) {
            const thread_count = vector.common.batchedThreadCount(batch_count, m, n, k);
            if (thread_count > 1) return vector.tile.forRangeOpts(pool, BlasBatches, batches, batch_count, thread_count, run, .{ .park_after = true });
        }
    }
    run(batches, 0, batch_count);
}

const BlasBatches = struct {
    ap: [*]const f32,
    bp: [*]const f32,
    cp: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
};

fn blasBatchRange(comptime kind: ops.MatmulKind) fn (BlasBatches, usize, usize) void {
    return struct {
        fn go(b: BlasBatches, batch_start: usize, batch_end: usize) void {
            const matrix_a_len = if (kind == .trans_a) b.k * b.m else b.m * b.k;
            const matrix_b_len = if (kind == .trans_b) b.n * b.k else b.k * b.n;
            for (batch_start..batch_end) |bi| {
                blas.gemm(
                    kind,
                    b.m,
                    b.n,
                    b.k,
                    b.ap[bi * b.stride_a .. bi * b.stride_a + matrix_a_len],
                    b.bp[bi * b.stride_b .. bi * b.stride_b + matrix_b_len],
                    0.0,
                    b.cp[bi * b.stride_c .. bi * b.stride_c + b.m * b.n],
                );
            }
        }
    }.go;
}

fn shouldUseBlas(m: usize, n: usize, k: usize) bool {
    return blas.fitsCblas(m, n, k) and m >= 16 and n >= 16 and k >= 16;
}

fn shouldUseBatchedBlas(m: usize, n: usize, k: usize, batch_count: usize) bool {
    return batch_count > 1 and shouldUseBlas(m, n, k);
}
