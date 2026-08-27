//! The one CPU kernel provider: the `kernels` namespace below is the kernel
//! set (its declaration list is the interface; `backend.zig`'s
//! `conformKernels` checks it). Elementwise, reduction, conv, pool and
//! Winograd entries forward to the portable `@Vector` leaves in `vector/`;
//! the dense and quantized GEMM family is defined in this file and routes
//! each call across the GPU provider (`-Dgpu`), a CBLAS provider (`-Dblas`)
//! and the vector kernels. Every pool-taking kernel takes `pc:
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

const native = @This();
const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

// The numeric dispatch gates live in src/parallel.zig's policy table
// (values and measurement rationale there); aliased file-locally so the
// kernels below read bare names.
const q8_0_lhs_stack_blocks = parallel.q8_0_lhs_stack_blocks;
const q4_k_x4_min_rows = parallel.q4_k_x4_min_rows;
const q5_k_x4_prefix_min_rows = parallel.q5_k_x4_prefix_min_rows;

/// The `[rows, cols]` f32 activation behind an LHS tensor, validated once
/// at the dispatch tier (rank 2, the declared dims, contiguous) so the
/// allocation-free range quantizers run unchecked below it.
fn lhsRows(a: *const Tensor, rows: usize, cols: usize) ![]const f32 {
    const view = try a.rankView(2);
    if (view.dim(0) != rows or view.dim(1) != cols) return tensor.TensorError.ShapeMismatch;
    return try a.dataConstChecked();
}

/// Row-range LHS quantization over the pool. `quantizeRange(blocks, data,
/// rows, cols, blocks_per_row, unit_start, unit_end)` quantizes units
/// `[unit_start, unit_end)` (rows, or the lane-packed formats' row groups)
/// of the `[rows, cols]` activation into `blocks`; units own disjoint
/// blocks, so any split produces the serial call's bytes. Gate: the fused
/// activation+quantization tasks' (`exec/quant_matmul.zig`) element
/// threshold, `rows * cols >= vector_elementwise_len_threshold / 8`; a
/// decode row never touches the pool, and a call without a team (or from
/// inside one, where `parallelChunks` degrades to the caller) runs the
/// serial walk.
fn quantizeLhsUnits(
    pc: ParallelConfig,
    comptime Block: type,
    comptime quantizeRange: fn ([]Block, []const f32, usize, usize, usize, usize, usize) void,
    blocks: []Block,
    data: []const f32,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    unit_count: usize,
) void {
    const Task = struct {
        blocks: []Block,
        data: []const f32,
        rows: usize,
        cols: usize,
        blocks_per_row: usize,
        unit_start: usize,
        unit_end: usize,

        fn run(task: *const @This()) void {
            quantizeRange(task.blocks, task.data, task.rows, task.cols, task.blocks_per_row, task.unit_start, task.unit_end);
        }
    };
    if (pc.pool) |pool| {
        if (unit_count > 1 and rows * cols >= parallel.vector_elementwise_len_threshold / 8) {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), unit_count);
            if (task_count > 1) {
                var tasks: [parallel.vector_max_threads]Task = undefined;
                for (0..task_count) |task_i| {
                    tasks[task_i] = .{
                        .blocks = blocks,
                        .data = data,
                        .rows = rows,
                        .cols = cols,
                        .blocks_per_row = blocks_per_row,
                        .unit_start = task_i * unit_count / task_count,
                        .unit_end = (task_i + 1) * unit_count / task_count,
                    };
                }
                pool.parallelChunks(Task, tasks[0..task_count], Task.run);
                return;
            }
        }
    }
    quantizeRange(blocks, data, rows, cols, blocks_per_row, 0, unit_count);
}

/// Q8_K row blocks of the `[rows, k]` activation `data`, allocated for the
/// caller (who frees them) and quantized through `quantizeLhsUnits`.
fn quantizedLhsQ8_K(pc: ParallelConfig, allocator: std.mem.Allocator, data: []const f32, rows: usize, k: usize) ![]dtype_mod.BlockQ8_K {
    const blocks_per_row = try quantized_matmul.q8k.qkBlockCount(k);
    const blocks = try allocator.alloc(dtype_mod.BlockQ8_K, try checkedQuantizedProduct(rows, blocks_per_row));
    errdefer allocator.free(blocks);
    quantizeLhsUnits(pc, dtype_mod.BlockQ8_K, quantized_matmul.q8k.quantizeRowsQ8_KRangeInto, blocks, data, rows, k, blocks_per_row, rows);
    return blocks;
}

fn checkedTensorProduct(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch tensor.TensorError.InvalidDataLength;
}

fn checkedQuantizedProduct(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch quantized_matmul.types.QuantizedFormatError.InvalidQuantizedLength;
}

pub const ParallelConfig = vector.ParallelConfig;

/// The kernel set: the `vector/` leaf kernels by name and the GEMM family
/// defined below. The declaration list IS the interface (`backend.zig`'s
/// `conformKernels` derives its checks from it); a `pool_free_<name>`
/// marker beside a kernel states that it takes no `pc: ParallelConfig`
/// (every other kernel takes `pc` first), so going pool-free is an
/// explicit decision, not a signature accident.
pub const kernels = struct {
    pub const addInto = vector.elementwise.addInto;
    pub const pool_free_addInto = true;
    pub const addContiguousIntoUnchecked = vector.elementwise.addContiguousIntoUnchecked;
    pub const divContiguousIntoUnchecked = vector.elementwise.divContiguousIntoUnchecked;
    pub const maximumContiguousIntoUnchecked = vector.elementwise.maximumContiguousIntoUnchecked;
    pub const minimumContiguousIntoUnchecked = vector.elementwise.minimumContiguousIntoUnchecked;
    pub const subInto = vector.elementwise.subInto;
    pub const pool_free_subInto = true;
    pub const subContiguousIntoUnchecked = vector.elementwise.subContiguousIntoUnchecked;
    pub const mulInto = vector.elementwise.mulInto;
    pub const pool_free_mulInto = true;
    pub const mulContiguousIntoUnchecked = vector.elementwise.mulContiguousIntoUnchecked;
    pub const elementwiseContiguousIntoTyped = vector.elementwise.elementwiseContiguousIntoTyped;
    pub const scaleInto = vector.elementwise.scaleInto;
    pub const addScaledSlice = vector.elementwise.addScaledSlice;
    pub const pool_free_addScaledSlice = true;
    pub const addRowVectorSlice = vector.elementwise.addRowVectorSlice;
    pub const pool_free_addRowVectorSlice = true;
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
    pub const pool_free_sumSlice = true;
    pub const prodInto = vector.elementwise.prodInto;
    pub const prodSlice = vector.elementwise.prodSlice;
    pub const pool_free_prodSlice = true;
    pub const sumSliceTyped = vector.elementwise.sumSliceTyped;
    pub const dot = native.dot;
    pub const gemm = native.gemm;
    pub const gemmBatched = native.gemmBatched;
    pub const packDenseRhs = native.packDenseRhs;
    pub const pool_free_packDenseRhs = true;
    pub const quantizeMatmulRhsBlockwiseI8 = native.quantizeMatmulRhsBlockwiseI8;
    pub const pool_free_quantizeMatmulRhsBlockwiseI8 = true;
    pub const quantizeMatmulRhsQ4_0 = native.quantizeMatmulRhsQ4_0;
    pub const pool_free_quantizeMatmulRhsQ4_0 = true;
    pub const quantizeMatmulRhsQ8_0 = native.quantizeMatmulRhsQ8_0;
    pub const pool_free_quantizeMatmulRhsQ8_0 = true;
    pub const matmul2DQuantizedRhs = native.matmul2DQuantizedRhs;
    pub const matmulPacked = native.matmulPacked;
    pub const matmul2DPackedQ8_0x4LhsRhs = native.matmul2DPackedQ8_0x4LhsRhs;
    pub const matmul2DPackedPaddedQ8_0x4LhsRhs = native.matmul2DPackedPaddedQ8_0x4LhsRhs;
    pub const matmulPackedSlice = native.matmulPackedSlice;
    pub const unaryRowSlice = vector.elementwise.unaryRowSlice;
    pub const pool_free_unaryRowSlice = true;
    pub const mulRowSlice = vector.elementwise.mulRowSlice;
    pub const pool_free_mulRowSlice = true;
    // The fused row kernels (vector/rows.zig): task-carrying serial
    // entries. The exec domain modules split the task ranges over the
    // pool themselves (`dispatchRange`/`dispatchInnerLanes` with the
    // `rows` run adapters), so every entry here is pool-free by design.
    pub const softmaxRows = vector.rows.softmaxRows;
    pub const pool_free_softmaxRows = true;
    pub const softmaxExtRows = vector.rows.softmaxExtRows;
    pub const pool_free_softmaxExtRows = true;
    pub const softmaxBackwardRows = vector.rows.softmaxBackwardRows;
    pub const pool_free_softmaxBackwardRows = true;
    pub const logsumexpRows = vector.rows.logsumexpRows;
    pub const pool_free_logsumexpRows = true;
    pub const logSoftmaxRows = vector.rows.logSoftmaxRows;
    pub const pool_free_logSoftmaxRows = true;
    pub const softmaxInner = vector.rows.softmaxInner;
    pub const pool_free_softmaxInner = true;
    pub const logsumexpInner = vector.rows.logsumexpInner;
    pub const pool_free_logsumexpInner = true;
    pub const logSoftmaxInner = vector.rows.logSoftmaxInner;
    pub const pool_free_logSoftmaxInner = true;
    pub const softmaxBackwardInner = vector.rows.softmaxBackwardInner;
    pub const pool_free_softmaxBackwardInner = true;
    pub const splitSwiGluRows = vector.rows.splitSwiGluRows;
    pub const pool_free_splitSwiGluRows = true;
    pub const splitGluRows = vector.rows.splitGluRows;
    pub const pool_free_splitGluRows = true;
    pub const splitSwiGluBackwardRows = vector.rows.splitSwiGluBackwardRows;
    pub const pool_free_splitSwiGluBackwardRows = true;
    pub const splitGluBackwardRows = vector.rows.splitGluBackwardRows;
    pub const pool_free_splitGluBackwardRows = true;
    pub const rmsNormMulRopeHalfVectors = vector.rows.rmsNormMulRopeHalfVectors;
    pub const pool_free_rmsNormMulRopeHalfVectors = true;
    pub const rmsNormMulRows = vector.rows.rmsNormMulRows;
    pub const pool_free_rmsNormMulRows = true;
    pub const rmsNormMulAddRows = vector.rows.rmsNormMulAddRows;
    pub const pool_free_rmsNormMulAddRows = true;
    pub const rmsNormMulBackwardInputRows = vector.rows.rmsNormMulBackwardInputRows;
    pub const pool_free_rmsNormMulBackwardInputRows = true;
    pub const rmsNormMulBackwardWeightRows = vector.rows.rmsNormMulBackwardWeightRows;
    pub const pool_free_rmsNormMulBackwardWeightRows = true;
    pub const rmsNormWeightGradBlocks = vector.rows.rmsNormWeightGradBlocks;
    pub const pool_free_rmsNormWeightGradBlocks = true;
    pub const rmsNormWeightGradReduce = vector.rows.rmsNormWeightGradReduce;
    pub const pool_free_rmsNormWeightGradReduce = true;
    pub const rmsNormInner = vector.rows.rmsNormInner;
    pub const pool_free_rmsNormInner = true;
    pub const rmsNormBackwardInputInner = vector.rows.rmsNormBackwardInputInner;
    pub const pool_free_rmsNormBackwardInputInner = true;
    pub const rmsNormBackwardWeightInner = vector.rows.rmsNormBackwardWeightInner;
    pub const pool_free_rmsNormBackwardWeightInner = true;
    pub const layerNormRows = vector.rows.layerNormRows;
    pub const pool_free_layerNormRows = true;
    pub const layerNormBackwardInputRows = vector.rows.layerNormBackwardInputRows;
    pub const pool_free_layerNormBackwardInputRows = true;
    pub const layerNormAffineParamGradRows = vector.rows.layerNormAffineParamGradRows;
    pub const pool_free_layerNormAffineParamGradRows = true;
    pub const layerNormRowStats = vector.rows.layerNormRowStats;
    pub const pool_free_layerNormRowStats = true;
    pub const layerNormParamGradColumns = vector.rows.layerNormParamGradColumns;
    pub const pool_free_layerNormParamGradColumns = true;
    pub const layerNormInner = vector.rows.layerNormInner;
    pub const pool_free_layerNormInner = true;
    pub const layerNormBackwardInner = vector.rows.layerNormBackwardInner;
    pub const pool_free_layerNormBackwardInner = true;
    pub const varianceInner = vector.rows.varianceInner;
    pub const pool_free_varianceInner = true;
    pub const standardizeInner = vector.rows.standardizeInner;
    pub const pool_free_standardizeInner = true;
    pub const standardizeBackwardInner = vector.rows.standardizeBackwardInner;
    pub const pool_free_standardizeBackwardInner = true;
    pub const crossEntropyLossRows = vector.rows.crossEntropyLossRows;
    pub const pool_free_crossEntropyLossRows = true;
    pub const crossEntropyBackwardRows = vector.rows.crossEntropyBackwardRows;
    pub const pool_free_crossEntropyBackwardRows = true;
    pub const distillStatsRows = vector.rows.distillStatsRows;
    pub const pool_free_distillStatsRows = true;
    pub const distillBackwardRows = vector.rows.distillBackwardRows;
    pub const pool_free_distillBackwardRows = true;
    pub const dropoutRange = vector.rows.dropoutRange;
    pub const pool_free_dropoutRange = true;
    pub const scatterAddRows = vector.rows.scatterAddRows;
    pub const pool_free_scatterAddRows = true;
    // The grouped-causal attention kernels (vector/attention.zig):
    // task-carrying serial entries, pool-free like the row kernels.
    pub const groupedCausalAttentionHeads = vector.attention.groupedCausalAttentionHeads;
    pub const pool_free_groupedCausalAttentionHeads = true;
    pub const groupedCausalAttentionHeadPairs = vector.attention.groupedCausalAttentionHeadPairs;
    pub const pool_free_groupedCausalAttentionHeadPairs = true;
    pub const groupedCausalAttentionQueryTiles = vector.attention.groupedCausalAttentionQueryTiles;
    pub const pool_free_groupedCausalAttentionQueryTiles = true;
    pub const groupedCausalAttentionMultiUnits = vector.attention.groupedCausalAttentionMultiUnits;
    pub const pool_free_groupedCausalAttentionMultiUnits = true;
    pub const groupedCausalAttentionBackwardKvHeads = vector.attention.groupedCausalAttentionBackwardKvHeads;
    pub const pool_free_groupedCausalAttentionBackwardKvHeads = true;
    pub const groupedCausalAttentionBackwardTiles = vector.attention.groupedCausalAttentionBackwardTiles;
    pub const pool_free_groupedCausalAttentionBackwardTiles = true;
    pub const groupedCausalAttentionBackwardBlasTiles = vector.attention.groupedCausalAttentionBackwardBlasTiles;
    pub const pool_free_groupedCausalAttentionBackwardBlasTiles = true;
    pub const attentionBackwardReduceRows = vector.attention.attentionBackwardReduceRows;
    pub const pool_free_attentionBackwardReduceRows = true;
    // Dtype cast rows (vector/elementwise.zig) and the straggler row
    // kernels (vector/rows.zig): extremum/variance rows, the fused-rope
    // pair strip, the gated vector scans, the masked-reduce select row.
    pub const castF32ToF16 = vector.elementwise.castF32ToF16;
    pub const pool_free_castF32ToF16 = true;
    pub const castF16ToF32 = vector.elementwise.castF16ToF32;
    pub const pool_free_castF16ToF32 = true;
    pub const castF32ToBf16 = vector.elementwise.castF32ToBf16;
    pub const pool_free_castF32ToBf16 = true;
    pub const castBf16ToF32 = vector.elementwise.castBf16ToF32;
    pub const pool_free_castBf16ToF32 = true;
    pub const extremumRowValue = vector.rows.extremumRowValue;
    pub const pool_free_extremumRowValue = true;
    pub const varianceRowsInto = vector.rows.varianceRowsInto;
    pub const pool_free_varianceRowsInto = true;
    pub const ropeHalfPairsInto = vector.rows.ropeHalfPairsInto;
    pub const pool_free_ropeHalfPairsInto = true;
    pub const scanRows = vector.rows.scanRows;
    pub const pool_free_scanRows = true;
    pub const scanColumns = vector.rows.scanColumns;
    pub const pool_free_scanColumns = true;
    pub const selectRow = vector.rows.selectRow;
    pub const pool_free_selectRow = true;
    // The per-format quantized row/pack/tile kernels the exec MoE
    // streams, the gemma fused-MoE skeleton, the PTQTP/borrowed weight
    // containers and the ternary-LoRA backward reach (previously named
    // as quant/<fmt> members): registered here so every band above the
    // backend goes through the conformed table.
    pub const quantizeRowQ8_0Into = quantized_matmul.q8k.quantizeRowQ8_0Into;
    pub const pool_free_quantizeRowQ8_0Into = true;
    pub const quantizeRowQ8_0IntoUnchecked = quantized_matmul.q8k.quantizeRowQ8_0IntoUnchecked;
    pub const pool_free_quantizeRowQ8_0IntoUnchecked = true;
    pub const quantizeRowQ8_KInto = quantized_matmul.q8k.quantizeRowQ8_KInto;
    pub const pool_free_quantizeRowQ8_KInto = true;
    pub const quantizeRowQ8_KIntoUnchecked = quantized_matmul.q8k.quantizeRowQ8_KIntoUnchecked;
    pub const pool_free_quantizeRowQ8_KIntoUnchecked = true;
    pub const packRowsQ8_Kx4PaddedInto = quantized_matmul.q8k.packRowsQ8_Kx4PaddedInto;
    pub const pool_free_packRowsQ8_Kx4PaddedInto = true;
    pub const dequantizeRowQ8_0Into = quantized_matmul.q8k.dequantizeRowQ8_0Into;
    pub const pool_free_dequantizeRowQ8_0Into = true;
    pub const matmulQ8_0x4RhsTile = quantized_matmul.q8_0.matmulQ8_0x4RhsTile;
    pub const pool_free_matmulQ8_0x4RhsTile = true;
    pub const matmulQ8_0RhsTile = quantized_matmul.q8_0.matmulQ8_0RhsTile;
    pub const pool_free_matmulQ8_0RhsTile = true;
    pub const splitSwiGluRowInto = quantized_matmul.q8_0.splitSwiGluRowInto;
    pub const pool_free_splitSwiGluRowInto = true;
    pub const quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto = quantized_matmul.q8_0.quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto;
    pub const pool_free_quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto = true;
    pub const packMatmulRhsQ8_0x4 = quantized_matmul.q8_0.packMatmulRhsQ8_0x4;
    pub const pool_free_packMatmulRhsQ8_0x4 = true;
    pub const matmulQ6_Kx4RhsTile = quantized_matmul.q6_k.matmulQ6_Kx4RhsTile;
    pub const pool_free_matmulQ6_Kx4RhsTile = true;
    pub const matmulQ6_Kx4RhsPairTile = quantized_matmul.q6_k.matmulQ6_Kx4RhsPairTile;
    pub const pool_free_matmulQ6_Kx4RhsPairTile = true;
    pub const matmulQ6_KRhsTile = quantized_matmul.q6_k.matmulQ6_KRhsTile;
    pub const pool_free_matmulQ6_KRhsTile = true;
    pub const matmulQ6_KRhsCompactColOuter = quantized_matmul.q6_k.matmulQ6_KRhsCompactColOuter;
    pub const pool_free_matmulQ6_KRhsCompactColOuter = true;
    pub const matmulQ4_KRhsTile = quantized_matmul.q4_k.matmulQ4_KRhsTile;
    pub const pool_free_matmulQ4_KRhsTile = true;
    pub const matmulQ4_KRhsCompactColOuter = quantized_matmul.q4_k.matmulQ4_KRhsCompactColOuter;
    pub const pool_free_matmulQ4_KRhsCompactColOuter = true;
    pub const matmulMXFP4RhsTile = quantized_matmul.mxfp4.matmulMXFP4RhsTile;
    pub const pool_free_matmulMXFP4RhsTile = true;
    pub const quantizedMatmulRhsTQ2_0FromBorrowedBlocks = quantized_matmul.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks;
    pub const pool_free_quantizedMatmulRhsTQ2_0FromBorrowedBlocks = true;
    pub const quantizedMatmulRhsTQ2_0FromF32Absmean = quantized_matmul.ternary.quantizedMatmulRhsTQ2_0FromF32Absmean;
    pub const pool_free_quantizedMatmulRhsTQ2_0FromF32Absmean = true;
    pub const matmulTQ2_0RhsTile = quantized_matmul.ternary.matmulTQ2_0RhsTile;
    pub const pool_free_matmulTQ2_0RhsTile = true;
    pub const matmulTQ2_0FoldedX4RhsTile = quantized_matmul.ternary.matmulTQ2_0FoldedX4RhsTile;
    pub const pool_free_matmulTQ2_0FoldedX4RhsTile = true;
    pub const matmulTQ2_0X4RhsTile = quantized_matmul.ternary.matmulTQ2_0X4RhsTile;
    pub const pool_free_matmulTQ2_0X4RhsTile = true;
    pub const matmulTQ2_0X4RhsTileAcc = quantized_matmul.ternary.matmulTQ2_0X4RhsTileAcc;
    pub const pool_free_matmulTQ2_0X4RhsTileAcc = true;
    pub const packMatmulRhsTQ2_0x4 = quantized_matmul.ternary.packMatmulRhsTQ2_0x4;
    pub const pool_free_packMatmulRhsTQ2_0x4 = true;
    pub const packMatmulRhsTQ2_0Foldedx4 = quantized_matmul.ternary.packMatmulRhsTQ2_0Foldedx4;
    pub const pool_free_packMatmulRhsTQ2_0Foldedx4 = true;
    pub const packMatmulRhsTQ2_0Foldedx4Into = quantized_matmul.ternary.packMatmulRhsTQ2_0Foldedx4Into;
    pub const pool_free_packMatmulRhsTQ2_0Foldedx4Into = true;
    pub const packMatmulRhsTQ2_0FoldedRows = quantized_matmul.ternary.packMatmulRhsTQ2_0FoldedRows;
    pub const pool_free_packMatmulRhsTQ2_0FoldedRows = true;
    pub const packMatmulRhsTQ2_0FoldedRowsFromX4 = quantized_matmul.ternary.packMatmulRhsTQ2_0FoldedRowsFromX4;
    pub const pool_free_packMatmulRhsTQ2_0FoldedRowsFromX4 = true;
    pub const matmulTableQ8_KRhsTile = quantized_matmul.cold.matmulTableQ8_KRhsTile;
    pub const pool_free_matmulTableQ8_KRhsTile = true;
    pub const dequantizeRowTQ2_0Into = quantized_matmul.cold.dequantizeRowTQ2_0Into;
    pub const pool_free_dequantizeRowTQ2_0Into = true;
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

/// f32 [m, k] x a quantized RHS container -> f32 [m, n], over the request
/// `g`: the table families' BLAS crossover first, then one LHS row
/// quantization into `g.lhs`'s block form (Q8_0 rows on the stack below
/// `q8_0_lhs_stack_blocks`), then the one parallel vector entry
/// (`vector.matmul_quant.gemm2D`).
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
    if (comptime isa.reference) return vector.matmul_quant.scalar.matmulQuantRows(g, allocator, out, a, rhs, m, n, k);
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;
    if (comptime build_options.use_blas and tableBlasFormat(g.weight)) {
        if (m >= table_blas_min_m and blas.fitsCblas(m, n, k)) {
            return matmul2DQuantizedRhsTableBlas(pc, g.weight, allocator, out, a, rhs, m, n, k);
        }
    }
    const cd = contiguousData(out, m * n);
    switch (comptime g.lhs) {
        .q8_0 => {
            const blocks_per_row = try quantized_matmul.q8k.q8_0BlockCount(k);
            const block_count = m * blocks_per_row;
            var stack_blocks: [q8_0_lhs_stack_blocks]dtype_mod.BlockQ8_0 = undefined;
            const qlhs_blocks = if (block_count <= stack_blocks.len)
                stack_blocks[0..block_count]
            else
                try allocator.alloc(dtype_mod.BlockQ8_0, block_count);
            defer if (block_count > stack_blocks.len) allocator.free(qlhs_blocks);
            quantizeLhsUnits(pc, dtype_mod.BlockQ8_0, quantized_matmul.q8k.quantizeRowsQ8_0RangeInto, qlhs_blocks, try lhsRows(a, m, k), m, k, blocks_per_row, m);
            vector.matmul_quant.gemm2D(pc, g, cd, qlhs_blocks, rhs, m, n, k);
        },
        .q8_1 => {
            var qlhs = try quantized_matmul.cold.quantizeRowsQ8_1(allocator, a);
            defer qlhs.deinit();
            vector.matmul_quant.gemm2D(pc, g, cd, qlhs.blocks, rhs, m, n, k);
        },
        .q8_k => {
            const qlhs = try quantizedLhsQ8_K(pc, allocator, try lhsRows(a, m, k), m, k);
            defer allocator.free(qlhs);
            vector.matmul_quant.gemm2D(pc, g, cd, qlhs, rhs, m, n, k);
        },
        else => @compileError("matmulQuantRows: LHS form ." ++ @tagName(g.lhs) ++ " is quantized by the caller, not this entry"),
    }
}

/// Q2_0 BLAS crossover and panel budget: src/parallel.zig's policy table.
const q2_0_blas_min_m = parallel.q2_0_blas_min_m;
const q2_0_blas_panel_floats = parallel.q2_0_blas_panel_floats;

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
        blas.gemmStrided(false, true, m, n, kc, 1.0, ad[k0..], k, panel, kc, if (k0 == 0) 0.0 else 1.0, cd, n);
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

            quantizeLhsUnits(pc, quantized_matmul.BlockQ8_0x4, quantized_matmul.q8_0.quantizeRowsQ8_0x4PaddedGroupsInto, qlhs_blocks, try lhsRows(a, m, k), m, k, blocks_per_row, row_groups);
            vector.matmul_quant.gemm2D(pc, q8_0x4_padded_gemm, cd, qlhs_blocks, rhs, m, n, k);
            return;
        }
        if (m >= parallel.vector_column_min_m) {
            return matmul2DQuantizedRhsQ8_0x4BulkTail(pc, allocator, out, a, rhs, m, n, k);
        }
        return matmulQuantRows(pc, q8_0x4_rows_gemm, allocator, out, a, rhs, m, n, k);
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

    quantizeLhsUnits(pc, quantized_matmul.BlockQ8_0x4, quantized_matmul.q8_0.quantizeRowsQ8_0x4GroupsInto, qlhs_blocks, try lhsRows(a, m, k), m, k, blocks_per_row, m / 4);
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
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) !void {
    const cd = contiguousData(out, try checkedTensorProduct(m, n));
    const ad = try lhsRows(a, m, k);
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

        quantizeLhsUnits(pc, quantized_matmul.BlockQ8_0x4, quantized_matmul.q8_0.quantizeRowsQ8_0x4GroupsInto, qlhs_blocks, ad[0 .. bulk_rows * k], bulk_rows, k, blocks_per_row, bulk_rows / 4);
        vector.matmul_quant.gemm2D(pc, q8_0x4_packed_gemm, cd[0 .. bulk_rows * n], qlhs_blocks, rhs, bulk_rows, n, k);
    }

    const tail_rows = m - bulk_rows;
    const tail_count = try checkedQuantizedProduct(tail_rows, blocks_per_row);
    var tail_stack: [q8_0_lhs_stack_blocks]dtype_mod.BlockQ8_0 = undefined;
    const tail_blocks = if (tail_count <= tail_stack.len)
        tail_stack[0..tail_count]
    else
        try allocator.alloc(dtype_mod.BlockQ8_0, tail_count);
    defer if (tail_count > tail_stack.len) allocator.free(tail_blocks);

    quantizeLhsUnits(pc, dtype_mod.BlockQ8_0, quantized_matmul.q8k.quantizeRowsQ8_0RangeInto, tail_blocks, ad[bulk_rows * k .. m * k], tail_rows, k, blocks_per_row, tail_rows);
    // The <=3-row remainder runs after the bulk kernel completes; the caller's
    // `pc` passes through so it can column-split like a decode-shaped matmul
    // (a parallel split never changes per-element math).
    vector.matmul_quant.gemm2D(pc, q8_0x4_rows_gemm, cd[bulk_rows * n .. m * n], tail_blocks, rhs, tail_rows, n, k);
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
    vector.matmul_quant.gemm2D(pc, q8_0x4_packed_gemm, cd, lhs_blocks, rhs, m, n, k);
}

/// The padded twin: `ceil(m / 4)` LHS groups, masked writes, any `m`.
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
    const x4_lhs = Lhs == quantized_matmul.BlockQ8_Kx4;
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
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    const cd = contiguousData(out, try checkedTensorProduct(m, n));
    const ad = try lhsRows(a, m, k);
    const blocks_per_row = try quantized_matmul.blockCountForDType(.q8_k, k);
    const prefix_rows = m - m % 2;

    if (prefix_rows != 0) {
        const qlhs_x2 = try allocator.alloc(quantized_matmul.BlockQ8_Kx2Mmla, try checkedQuantizedProduct(prefix_rows / 2, blocks_per_row));
        defer allocator.free(qlhs_x2);

        quantizeLhsUnits(pc, quantized_matmul.BlockQ8_Kx2Mmla, quantized_matmul.q8k.quantizeRowsQ8_Kx2MmlaGroupsInto, qlhs_x2, ad[0 .. prefix_rows * k], prefix_rows, k, blocks_per_row, prefix_rows / 2);
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
/// `.q8_kx4` form). Policy per weight: Q4_K pads every row through the
/// padded group quantizer above `q4_k_x4_min_rows`; Q5_K keeps the
/// unpadded bulk + per-row tail split above `q5_k_x4_prefix_min_rows`.
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
    const prefix_min_rows = if (pad_x4_rows) q4_k_x4_min_rows else q5_k_x4_prefix_min_rows;
    const rows_g = comptime ops.QuantGemm{ .weight = g.weight, .rhs = g.rhs, .lhs = .q8_k };
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
        return matmulQuantRows(pc, rows_g, allocator, out, a, rhs, m, n, k);
    }

    const row_groups = if (pad_x4_rows) (prefix_rows + 3) / 4 else prefix_rows / 4;
    const qlhs_x4 = try allocator.alloc(quantized_matmul.BlockQ8_Kx4, try checkedQuantizedProduct(row_groups, blocks_per_row));
    defer allocator.free(qlhs_x4);

    // prefix_rows is a multiple of 4 unless the padded x4 kernel takes every
    // row, so the group walk's final-group lane padding is exactly the
    // padded form there and a no-op otherwise.
    const ad = try lhsRows(a, m, k);
    quantizeLhsUnits(pc, quantized_matmul.BlockQ8_Kx4, quantized_matmul.q8k.quantizeRowsQ8_Kx4GroupsInto, qlhs_x4, ad[0 .. prefix_rows * k], prefix_rows, k, blocks_per_row, row_groups);
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
        blas.gemmStrided(false, true, m, n, kc, 1.0, ad[k0..], k, panel, kc, if (k0 == 0) 0.0 else 1.0, cd, n);
    }
}

/// Folded-PTQTP BLAS crossover: src/parallel.zig's policy table.
const folded_blas_min_m = parallel.folded_blas_min_m;

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
    if (m < folded_blas_min_m or !blas.fitsCblas(m, n, k)) return false;

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
        blas.gemmStrided(false, true, m, n, kc, 1.0, a[k0..], k, panel, kc, if (k0 == 0) 0.0 else 1.0, out, n);
    }
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
        if (gpu.shouldUseGpuBatchedForRhs(b, m, n, k, batch_count)) {
            if (gpuBatched(kind, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c)) return;
        }
    }
    if (comptime build_options.use_blas and !isa.reference) {
        if (shouldUseBatchedBlas(m, n, k, batch_count)) {
            blasBatched(kind, out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c);
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
    const ap = a.buffer.data[a.offset..].ptr;
    const bp = b.buffer.data[b.offset..].ptr;
    const cp = out.buffer.data[out.offset..].ptr;
    const matrix_a_len = if (kind == .trans_a) k * m else m * k;
    const matrix_b_len = if (kind == .trans_b) n * k else k * n;

    for (0..batch_count) |bi| {
        blas.gemm(
            kind,
            m,
            n,
            k,
            ap[bi * stride_a .. bi * stride_a + matrix_a_len],
            bp[bi * stride_b .. bi * stride_b + matrix_b_len],
            0.0,
            cp[bi * stride_c .. bi * stride_c + m * n],
        );
    }
}

fn shouldUseBlas(m: usize, n: usize, k: usize) bool {
    return blas.fitsCblas(m, n, k) and m >= 16 and n >= 16 and k >= 16;
}

fn shouldUseBatchedBlas(m: usize, n: usize, k: usize, batch_count: usize) bool {
    return batch_count > 1 and shouldUseBlas(m, n, k);
}

fn contiguousDataConst(x: *const Tensor, len: usize) []const f32 {
    x.buffer.waitReady();
    return x.buffer.data[x.offset .. x.offset + len];
}

fn contiguousData(x: *Tensor, len: usize) []f32 {
    x.buffer.waitMutable();
    return x.buffer.data[x.offset .. x.offset + len];
}
