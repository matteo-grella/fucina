// Pure-Zig vector kernels using @Vector intrinsics. Portable across NEON
// (Apple Silicon, ARM64), AVX2 / AVX-512 (x86_64), WASM SIMD, and any target
// with vector support. Vector width is chosen by std.simd.suggestVectorLength
// so the same source compiles to 4-wide on NEON, 8-wide on AVX2, 16-wide on
// AVX-512.
//
// These are implementation details of the native backend: non-GEMM ops always
// use them, and GEMM falls back to them when no platform BLAS is selected.
//
// This file is now the shared core + re-export aggregator. The per-concern
// kernel families were split into vector/<concern>.zig and are re-exported via
// the manifest at the bottom so external callers keep using `vector.<fn>`
// unchanged:
//   vector/primitives.zig   - low-level @Vector helpers (leaf)
//   vector/elementwise.zig  - elementwise/reduction entry points + kernels
//   vector/gemm.zig         - dense f32/f16/f64/bf16 GEMM
//   vector/matmul_quant.zig - quantized matmul dispatch (calls into quant.zig)
//   vector/batched.zig      - batched dense GEMM
//   vector/conv.zig         - small-tap causal convolution (depthwise + general)
// What stays here is the shared core every section depends on: ParallelConfig,
// the V* vector-width aliases, the thread-count gates, and the contiguous-data
// accessors.

const std = @import("std");

const elementwise = @import("vector/elementwise.zig");
const gemm = @import("vector/gemm.zig");
const matmul_quant = @import("vector/matmul_quant.zig");
const batched = @import("vector/batched.zig");
const conv = @import("vector/conv.zig");
const pool_kernels = @import("vector/pool.zig");
const winograd = @import("vector/winograd.zig");
const primitives = @import("vector/primitives.zig");

// The shared core (ParallelConfig, V* aliases, thread-count gates,
// contiguous-data accessors) now lives in the child-neutral leaf
// `vector/common.zig` so the kernel children can import it directly instead of
// this parent barrel (breaks the parent<->child import cycle). These
// re-exports keep `vector.<sym>` callers unchanged.
const common = @import("vector/common.zig");

pub const ParallelConfig = common.ParallelConfig;

pub const vector_len = common.vector_len;
pub const Vf32 = common.Vf32;
pub const vector_len_f64 = common.vector_len_f64;
pub const Vf64 = common.Vf64;
pub const vector_len_f16 = common.vector_len_f16;
pub const Vf16 = common.Vf16;
pub const Vf32ForF16 = common.Vf32ForF16;
pub const Vf16ForF32 = common.Vf16ForF32;
pub const Vu16ForF16 = common.Vu16ForF16;
pub const Vu32ForF16 = common.Vu32ForF16;
pub const Vu16ForF32 = common.Vu16ForF32;
pub const Vu32ForF32 = common.Vu32ForF32;

// Shared thread-count gates + contiguous-data accessors — defined in
// `vector/common.zig`, re-exported here.
pub const i8ColumnThreadCount = common.i8ColumnThreadCount;
pub const elementwiseThreadCount = common.elementwiseThreadCount;
pub const matmulThreadCount = common.matmulThreadCount;
pub const batchedThreadCount = common.batchedThreadCount;
pub const depthwiseConvThreadCount = common.depthwiseConvThreadCount;
pub const generalConvThreadCount = common.generalConvThreadCount;
pub const columnThreadCount = common.columnThreadCount;

pub const contiguousDataConstOf = common.contiguousDataConstOf;
pub const contiguousDataOf = common.contiguousDataOf;
pub const contiguousDataConst = common.contiguousDataConst;
pub const contiguousData = common.contiguousData;

// ---------------------------------------------------------------------------
// Re-export manifest: public symbols moved to vector/<concern>.zig are
// re-exported here so external callers (native.zig, cpu.zig, backend.zig,
// exec.zig, ops.zig) keep using `vector.<fn>` unchanged. Cross-section helpers
// that one module borrows from another are re-exported here too so the section
// files can alias them via `vm.<fn>`.
// ---------------------------------------------------------------------------

// Elementwise / reduction entry points.
pub const addInto = elementwise.addInto;
pub const addContiguousIntoUnchecked = elementwise.addContiguousIntoUnchecked;
pub const addContiguousIntoUncheckedWithConfig = elementwise.addContiguousIntoUncheckedWithConfig;
pub const maximumContiguousIntoUncheckedWithConfig = elementwise.maximumContiguousIntoUncheckedWithConfig;
pub const minimumContiguousIntoUncheckedWithConfig = elementwise.minimumContiguousIntoUncheckedWithConfig;
pub const subInto = elementwise.subInto;
pub const subContiguousIntoUnchecked = elementwise.subContiguousIntoUnchecked;
pub const subContiguousIntoUncheckedWithConfig = elementwise.subContiguousIntoUncheckedWithConfig;
pub const mulInto = elementwise.mulInto;
pub const mulContiguousIntoUnchecked = elementwise.mulContiguousIntoUnchecked;
pub const mulContiguousIntoUncheckedWithConfig = elementwise.mulContiguousIntoUncheckedWithConfig;
pub const elementwiseContiguousIntoTypedWithConfig = elementwise.elementwiseContiguousIntoTypedWithConfig;
pub const scaleInto = elementwise.scaleInto;
pub const scaleIntoWithConfig = elementwise.scaleIntoWithConfig;
pub const addScaledSlice = elementwise.addScaledSlice;
pub const addRowVectorSlice = elementwise.addRowVectorSlice;
pub const addRowVectorUnarySlice = elementwise.addRowVectorUnarySlice;
pub const unaryContiguousIntoUnchecked = elementwise.unaryContiguousIntoUnchecked;
pub const unaryContiguousIntoUncheckedWithConfig = elementwise.unaryContiguousIntoUncheckedWithConfig;
pub const leakyReluContiguousIntoUnchecked = elementwise.leakyReluContiguousIntoUnchecked;
pub const leakyReluContiguousIntoUncheckedWithConfig = elementwise.leakyReluContiguousIntoUncheckedWithConfig;
pub const clampContiguousIntoUnchecked = elementwise.clampContiguousIntoUnchecked;
pub const clampContiguousIntoUncheckedWithConfig = elementwise.clampContiguousIntoUncheckedWithConfig;
pub const gatedContiguousIntoUnchecked = elementwise.gatedContiguousIntoUnchecked;
pub const gatedContiguousIntoUncheckedWithConfig = elementwise.gatedContiguousIntoUncheckedWithConfig;
pub const sumInto = elementwise.sumInto;
pub const sumIntoWithConfig = elementwise.sumIntoWithConfig;
pub const sumSlice = elementwise.sumSlice;
pub const sumSliceTypedWithConfig = elementwise.sumSliceTypedWithConfig;
pub const prodInto = elementwise.prodInto;
pub const prodIntoWithConfig = elementwise.prodIntoWithConfig;
pub const prodSlice = elementwise.prodSlice;
pub const dotInto = elementwise.dotInto;
pub const dotIntoWithConfig = elementwise.dotIntoWithConfig;
pub const dotIntoTypedWithConfig = elementwise.dotIntoTypedWithConfig;
pub const snakeIntoWithConfig = elementwise.snakeIntoWithConfig;
pub const snakeBackwardInputIntoWithConfig = elementwise.snakeBackwardInputIntoWithConfig;
pub const snakeBackwardParamsIntoWithConfig = elementwise.snakeBackwardParamsIntoWithConfig;
pub const groupNormIntoWithConfig = elementwise.groupNormIntoWithConfig;
pub const groupNormBackwardIntoWithConfig = elementwise.groupNormBackwardIntoWithConfig;
pub const preluChannelsIntoWithConfig = elementwise.preluChannelsIntoWithConfig;
pub const preluChannelsBackwardInputIntoWithConfig = elementwise.preluChannelsBackwardInputIntoWithConfig;
pub const preluChannelsBackwardAlphaIntoWithConfig = elementwise.preluChannelsBackwardAlphaIntoWithConfig;
pub const channelAffineIntoWithConfig = elementwise.channelAffineIntoWithConfig;
pub const Conv2dDims = conv.Conv2dDims;
pub const conv2dIntoWithConfig = conv.conv2dIntoWithConfig;
pub const conv2dBackwardInputIntoWithConfig = conv.conv2dBackwardInputIntoWithConfig;
pub const conv2dBackwardWeightIntoWithConfig = conv.conv2dBackwardWeightIntoWithConfig;
pub const im2colIntoWithConfig = conv.im2colIntoWithConfig;
pub const col2imIntoWithConfig = conv.col2imIntoWithConfig;
pub const WinogradF2Dims = winograd.F2Dims;
pub const winogradF2WeightTransformIntoWithConfig = winograd.f2WeightTransformIntoWithConfig;
pub const winogradF2InputTransformIntoWithConfig = winograd.f2InputTransformIntoWithConfig;
pub const winogradF2OutputTransformIntoWithConfig = winograd.f2OutputTransformIntoWithConfig;
pub const winogradF4WeightTransformIntoWithConfig = winograd.f4WeightTransformIntoWithConfig;
pub const winogradF4InputTransformIntoWithConfig = winograd.f4InputTransformIntoWithConfig;
pub const winogradF4OutputTransformIntoWithConfig = winograd.f4OutputTransformIntoWithConfig;
pub const PoolKind = pool_kernels.PoolKind;
pub const Pool2dDims = pool_kernels.Pool2dDims;
pub const pool2dIntoWithConfig = pool_kernels.pool2dIntoWithConfig;
pub const avgPool2dBackwardIntoWithConfig = pool_kernels.avgPool2dBackwardIntoWithConfig;
pub const maxPool2dBackwardIntoWithConfig = pool_kernels.maxPool2dBackwardIntoWithConfig;
pub const upsample2xNearestIntoWithConfig = pool_kernels.upsample2xNearestIntoWithConfig;
pub const Conv1dDims = conv.Conv1dDims;
pub const conv1dIntoWithConfig = conv.conv1dIntoWithConfig;
pub const conv1dBackwardInputIntoWithConfig = conv.conv1dBackwardInputIntoWithConfig;
pub const conv1dBackwardWeightIntoWithConfig = conv.conv1dBackwardWeightIntoWithConfig;
pub const col2im1dIntoWithConfig = conv.col2im1dIntoWithConfig;
pub const col2im1dBackwardIntoWithConfig = conv.col2im1dBackwardIntoWithConfig;
pub const causalDepthwiseConv1dIntoWithConfig = conv.causalDepthwiseConv1dIntoWithConfig;
pub const causalDepthwiseConv1dBackwardInputIntoWithConfig = conv.causalDepthwiseConv1dBackwardInputIntoWithConfig;
pub const causalDepthwiseConv1dBackwardKernelIntoWithConfig = conv.causalDepthwiseConv1dBackwardKernelIntoWithConfig;
pub const causalConv1dIntoWithConfig = conv.causalConv1dIntoWithConfig;
pub const causalConv1dBackwardInputIntoWithConfig = conv.causalConv1dBackwardInputIntoWithConfig;
pub const causalConv1dBackwardWeightIntoWithConfig = conv.causalConv1dBackwardWeightIntoWithConfig;
pub const groupedCausalConv1dIntoWithConfig = conv.groupedCausalConv1dIntoWithConfig;
pub const groupedCausalConv1dBackwardInputIntoWithConfig = conv.groupedCausalConv1dBackwardInputIntoWithConfig;
pub const groupedCausalConv1dBackwardWeightIntoWithConfig = conv.groupedCausalConv1dBackwardWeightIntoWithConfig;
// Borrowed by vector/primitives.zig (bf16/f16/f64 elementwise tails).
pub const applyElementwiseTyped = primitives.applyElementwiseTyped;

// Dense GEMM entry points.
pub const matmulInto = gemm.matmulInto;
pub const matmul2DIntoUnchecked = gemm.matmul2DIntoUnchecked;
pub const matmul2DIntoUncheckedWithConfig = gemm.matmul2DIntoUncheckedWithConfig;
pub const matmul2DIntoUncheckedTypedWithConfig = gemm.matmul2DIntoUncheckedTypedWithConfig;
pub const matmulTransAInto = gemm.matmulTransAInto;
pub const matmulTransA2DIntoUnchecked = gemm.matmulTransA2DIntoUnchecked;
pub const matmulTransA2DIntoUncheckedWithConfig = gemm.matmulTransA2DIntoUncheckedWithConfig;
pub const matmulTransBInto = gemm.matmulTransBInto;
pub const matmulTransB2DIntoUnchecked = gemm.matmulTransB2DIntoUnchecked;
pub const matmulTransB2DIntoUncheckedWithConfig = gemm.matmulTransB2DIntoUncheckedWithConfig;
pub const matmulTransB2DIntoUncheckedF16OperandsWithConfig = gemm.matmulTransB2DIntoUncheckedF16OperandsWithConfig;
pub const matmulTransB2DIntoUncheckedBf16RhsWithConfig = gemm.matmulTransB2DIntoUncheckedBf16RhsWithConfig;
// Borrowed by vector/batched.zig (per-batch inner work).
pub const gemmNNRange = gemm.gemmNNRange;
pub const gemmTNRange = gemm.gemmTNRange;
pub const gemmNTRange = gemm.gemmNTRange;
// Cache-blocked packed GEMM (vector/gemm_blocked.zig) + the register-tiled
// row-kernel paths that bypass it, exported for the bench-gemm baseline.
pub const gemm_blocked = @import("vector/gemm_blocked.zig");
pub const gemmNNRowPathWithConfig = gemm.gemmNNRowPathWithConfig;
pub const gemmTNRowPathWithConfig = gemm.gemmTNRowPathWithConfig;
pub const gemmNTRowPathWithConfig = gemm.gemmNTRowPathWithConfig;

// Batched dense GEMM entry points.
pub const matmulBatched2DIntoUnchecked = batched.matmulBatched2DIntoUnchecked;
pub const matmulBatched2DIntoUncheckedWithConfig = batched.matmulBatched2DIntoUncheckedWithConfig;
pub const matmulBatchedTransA2DIntoUnchecked = batched.matmulBatchedTransA2DIntoUnchecked;
pub const matmulBatchedTransA2DIntoUncheckedWithConfig = batched.matmulBatchedTransA2DIntoUncheckedWithConfig;
pub const matmulBatchedTransB2DIntoUnchecked = batched.matmulBatchedTransB2DIntoUnchecked;
pub const matmulBatchedTransB2DIntoUncheckedWithConfig = batched.matmulBatchedTransB2DIntoUncheckedWithConfig;

// Quantized matmul dispatch entry points.
pub const matmul2DI8BlockwiseIntoWithConfig = matmul_quant.matmul2DI8BlockwiseIntoWithConfig;
pub const matmul2DQ1_0RhsIntoWithConfig = matmul_quant.matmul2DQ1_0RhsIntoWithConfig;
pub const matmul2DQ2_0RhsIntoWithConfig = matmul_quant.matmul2DQ2_0RhsIntoWithConfig;
pub const matmul2DQ8_0RhsIntoWithConfig = matmul_quant.matmul2DQ8_0RhsIntoWithConfig;
pub const matmul2DQ8_0x4RhsIntoWithConfig = matmul_quant.matmul2DQ8_0x4RhsIntoWithConfig;
pub const matmul2DQ8_0x4PackedRhsIntoWithConfig = matmul_quant.matmul2DQ8_0x4PackedRhsIntoWithConfig;
pub const matmul2DQ8_0x4PackedPaddedRhsIntoWithConfig = matmul_quant.matmul2DQ8_0x4PackedPaddedRhsIntoWithConfig;
pub const matmul2DQ4_0RhsIntoWithConfig = matmul_quant.matmul2DQ4_0RhsIntoWithConfig;
pub const matmul2DQ4_1RhsIntoWithConfig = matmul_quant.matmul2DQ4_1RhsIntoWithConfig;
pub const matmul2DQ5_0RhsIntoWithConfig = matmul_quant.matmul2DQ5_0RhsIntoWithConfig;
pub const matmul2DQ5_1RhsIntoWithConfig = matmul_quant.matmul2DQ5_1RhsIntoWithConfig;
pub const matmul2DQ2_KRhsIntoWithConfig = matmul_quant.matmul2DQ2_KRhsIntoWithConfig;
pub const matmul2DQ3_KRhsIntoWithConfig = matmul_quant.matmul2DQ3_KRhsIntoWithConfig;
pub const matmul2DQ4_KRhsIntoWithConfig = matmul_quant.matmul2DQ4_KRhsIntoWithConfig;
pub const matmul2DQ4_Kx4RhsIntoWithConfig = matmul_quant.matmul2DQ4_Kx4RhsIntoWithConfig;
pub const matmul2DQ4_Kx8RhsIntoWithConfig = matmul_quant.matmul2DQ4_Kx8RhsIntoWithConfig;
pub const matmul2DQ4_Kx8Q8_Kx4RhsIntoWithConfig = matmul_quant.matmul2DQ4_Kx8Q8_Kx4RhsIntoWithConfig;
pub const matmul2DQ4_Kx2MmlaRhsIntoWithConfig = matmul_quant.matmul2DQ4_Kx2MmlaRhsIntoWithConfig;
pub const matmul2DQ4_Kx2MmlaQ8_Kx2MmlaRhsIntoWithConfig = matmul_quant.matmul2DQ4_Kx2MmlaQ8_Kx2MmlaRhsIntoWithConfig;
pub const matmul2DQ5_Kx8RhsIntoWithConfig = matmul_quant.matmul2DQ5_Kx8RhsIntoWithConfig;
pub const matmul2DQ5_Kx8Q8_Kx4RhsIntoWithConfig = matmul_quant.matmul2DQ5_Kx8Q8_Kx4RhsIntoWithConfig;
pub const matmul2DQ5_KRhsIntoWithConfig = matmul_quant.matmul2DQ5_KRhsIntoWithConfig;
pub const matmul2DQ6_KRhsIntoWithConfig = matmul_quant.matmul2DQ6_KRhsIntoWithConfig;
pub const matmul2DQ6_Kx4RhsIntoWithConfig = matmul_quant.matmul2DQ6_Kx4RhsIntoWithConfig;
pub const matmul2DTableQ8_0RhsIntoWithConfig = matmul_quant.matmul2DTableQ8_0RhsIntoWithConfig;
pub const matmul2DTableQ8_KRhsIntoWithConfig = matmul_quant.matmul2DTableQ8_KRhsIntoWithConfig;
pub const matmul2DTQ2_0RhsIntoWithConfig = matmul_quant.matmul2DTQ2_0RhsIntoWithConfig;
pub const matmul2DTQ2_0F32RhsIntoWithConfig = matmul_quant.matmul2DTQ2_0F32RhsIntoWithConfig;

// @Vector primitives borrowed by vector/elementwise.zig and vector/gemm.zig.
pub const vecAdd = primitives.vecAdd;
pub const vecSub = primitives.vecSub;
pub const vecMul = primitives.vecMul;
pub const vecScale = primitives.vecScale;
pub const vecAddScaled = primitives.vecAddScaled;
pub const vecUnary = primitives.vecUnary;
pub const vecAddUnary = primitives.vecAddUnary;
pub const vecLeakyRelu = primitives.vecLeakyRelu;
pub const vecClamp = primitives.vecClamp;
pub const vecGated = primitives.vecGated;
pub const vecSum = primitives.vecSum;
pub const vecDot = primitives.vecDot;
pub const vecElementwiseF64 = primitives.vecElementwiseF64;
pub const vecElementwiseF16 = primitives.vecElementwiseF16;
pub const vecElementwiseBf16 = primitives.vecElementwiseBf16;
pub const vexpf = primitives.vexpf;
pub const vecSumF64 = primitives.vecSumF64;
pub const vecSumF16ToF32 = primitives.vecSumF16ToF32;
pub const vecSumBf16ToF32 = primitives.vecSumBf16ToF32;
pub const vecDotF64 = primitives.vecDotF64;
pub const vecDotF16ToF32 = primitives.vecDotF16ToF32;
pub const vecDotBf16ToF32 = primitives.vecDotBf16ToF32;
pub const bf16VecToF32 = primitives.bf16VecToF32;
pub const f32VecToBf16 = primitives.f32VecToBf16;

test {
    _ = @import("vector/elementwise.zig");
    _ = @import("vector/gemm.zig");
    _ = @import("vector/gemm_blocked.zig");
    _ = @import("vector/matmul_quant.zig");
    _ = @import("vector/batched.zig");
    _ = @import("vector/conv.zig");
    _ = @import("vector/pool.zig");
    _ = @import("vector/winograd.zig");
    _ = @import("vector/primitives.zig");
}
