//! Backend facade: re-exports the shared kernel vocabulary (ops, the
//! quantized and packed RHS container types, `PackedRhsFor`) and the one
//! CPU kernel provider. `kernels` is `native.zig`'s kernel set; its
//! declaration list is the interface, held to the `pc`-first/pool-free
//! contract by `conformKernels` below at comptime. `-Dbackend=scalar`
//! does not select a second provider: it sets `backend/isa.zig`'s
//! `reference` flag, and every kernel entry then selects its scalar
//! reference arm internally (the `scalar` namespaces in `vector/` and the
//! `.scalar` tier in `quant/`). The fused fakequant kernels beside their
//! orchestration in `exec/` are backend-independent. Layer stack:
//! docs/ARCHITECTURE.md.
const std = @import("std");
const build_options = @import("build_options");
pub const ops = @import("backend/ops.zig");
pub const packed_matmul = @import("backend/packed.zig");
/// The raw quantized module: the research surface (bench, tools, examples)
/// and the models band reach it through `fucina.internal`; the core bands
/// use `quant` below.
pub const quantized_matmul = @import("backend/quant.zig");
const dtype_mod = @import("dtype.zig");
const tensor = @import("tensor.zig");
const thread = @import("thread.zig");

pub const dtype_info = dtype_mod;
pub const DType = dtype_mod.DType;
pub const PackedDenseRhs = packed_matmul.PackedDenseRhs;
/// The quantized surface the core bands use (exec, moe, store, weights, ag):
/// the RHS containers, the descriptors, the block-count rule and the entry
/// points the runtime, the MoE engine and the loaders call. The raw module
/// behind it (`quantized_matmul`) is backend-private for those bands; the
/// models band and the apps band may still take it through `fucina.internal`
/// as the documented escape hatch. `arch-check` enforces the split.
pub const quant = struct {
    /// The block-length check every quantized row/pack entry shares.
    pub const QuantizedFormatError = quantized_matmul.types.QuantizedFormatError;
    pub const supports_q4_k_mmla = quantized_matmul.supports_q4_k_mmla;
    pub const QuantizedMatmulRhsI8 = quantized_matmul.QuantizedMatmulRhsI8;
    pub const QuantizedMatmulRhsQ1_0 = quantized_matmul.QuantizedMatmulRhsQ1_0;
    pub const QuantizedMatmulRhsQ2_0 = quantized_matmul.QuantizedMatmulRhsQ2_0;
    pub const QuantizedMatmulRhsQ4_0 = quantized_matmul.QuantizedMatmulRhsQ4_0;
    pub const QuantizedMatmulRhsQ4_1 = quantized_matmul.QuantizedMatmulRhsQ4_1;
    pub const QuantizedMatmulRhsQ5_0 = quantized_matmul.QuantizedMatmulRhsQ5_0;
    pub const QuantizedMatmulRhsQ5_1 = quantized_matmul.QuantizedMatmulRhsQ5_1;
    pub const QuantizedMatmulRhsQ2_K = quantized_matmul.QuantizedMatmulRhsQ2_K;
    pub const QuantizedMatmulRhsQ3_K = quantized_matmul.QuantizedMatmulRhsQ3_K;
    pub const QuantizedMatmulRhsQ4_K = quantized_matmul.QuantizedMatmulRhsQ4_K;
    pub const QuantizedMatmulRhsQ4_Kx4 = quantized_matmul.QuantizedMatmulRhsQ4_Kx4;
    pub const QuantizedMatmulRhsQ4_Kx8 = quantized_matmul.QuantizedMatmulRhsQ4_Kx8;
    pub const QuantizedMatmulRhsQ4_Kx2Mmla = quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla;
    pub const QuantizedMatmulRhsQ5_K = quantized_matmul.QuantizedMatmulRhsQ5_K;
    pub const QuantizedMatmulRhsQ5_Kx8 = quantized_matmul.QuantizedMatmulRhsQ5_Kx8;
    pub const QuantizedMatmulRhsQ6_K = quantized_matmul.QuantizedMatmulRhsQ6_K;
    pub const QuantizedMatmulRhsQ6_Kx4 = quantized_matmul.QuantizedMatmulRhsQ6_Kx4;
    pub const QuantizedMatmulRhsQ8_0 = quantized_matmul.QuantizedMatmulRhsQ8_0;
    pub const QuantizedMatmulRhsQ8_0x4 = quantized_matmul.QuantizedMatmulRhsQ8_0x4;
    pub const QuantizedMatmulRhsIQ1_S = quantized_matmul.QuantizedMatmulRhsIQ1_S;
    pub const QuantizedMatmulRhsIQ1_M = quantized_matmul.QuantizedMatmulRhsIQ1_M;
    pub const QuantizedMatmulRhsIQ2_XXS = quantized_matmul.QuantizedMatmulRhsIQ2_XXS;
    pub const QuantizedMatmulRhsIQ2_XS = quantized_matmul.QuantizedMatmulRhsIQ2_XS;
    pub const QuantizedMatmulRhsIQ2_S = quantized_matmul.QuantizedMatmulRhsIQ2_S;
    pub const QuantizedMatmulRhsIQ3_XXS = quantized_matmul.QuantizedMatmulRhsIQ3_XXS;
    pub const QuantizedMatmulRhsIQ3_S = quantized_matmul.QuantizedMatmulRhsIQ3_S;
    pub const QuantizedMatmulRhsIQ4_NL = quantized_matmul.QuantizedMatmulRhsIQ4_NL;
    pub const QuantizedMatmulRhsIQ4_XS = quantized_matmul.QuantizedMatmulRhsIQ4_XS;
    pub const QuantizedMatmulRhsTQ1_0 = quantized_matmul.QuantizedMatmulRhsTQ1_0;
    pub const QuantizedMatmulRhsTQ2_0 = quantized_matmul.QuantizedMatmulRhsTQ2_0;
    pub const QuantizedMatmulRhsMXFP4 = quantized_matmul.QuantizedMatmulRhsMXFP4;
    pub const QuantizedMatmulRhsNVFP4 = quantized_matmul.QuantizedMatmulRhsNVFP4;
    pub const AnyQuantizedMatmulRhs = quantized_matmul.AnyQuantizedMatmulRhs;
    pub const QuantizedRowsQ4_0 = quantized_matmul.QuantizedRowsQ4_0;
    pub const QuantizedRowsQ8_0 = quantized_matmul.QuantizedRowsQ8_0;
    pub const PackedMatmulRhsI8 = QuantizedMatmulRhsI8;
    pub const CompactRhs = quantized_matmul.CompactRhs;
    pub const LanePackedRhs = quantized_matmul.LanePackedRhs;
    pub const QuantizedMatmulRhsRowsFor = quantized_matmul.QuantizedMatmulRhsRowsFor;
    pub const RhsLifetime = quantized_matmul.types.RhsLifetime;
    pub const RawRhs = quantized_matmul.types.RawRhs;
    pub const BlockQ8_0x4 = quantized_matmul.BlockQ8_0x4;
    pub const BlockQ8_Kx4 = quantized_matmul.BlockQ8_Kx4;
    pub const BlockTQ2_0Foldedx4 = quantized_matmul.BlockTQ2_0Foldedx4;
    pub const qk_k_block_size = quantized_matmul.types.qk_k_block_size;
    pub const blockCountForDType = quantized_matmul.blockCountForDType;
    pub const x4PrefixRows = quantized_matmul.x4PrefixRows;
    pub const packRhsAs = quantized_matmul.packRhsAs;
    pub const getRowsTensorInto = quantized_matmul.getRowsTensorInto;
    pub const dequantizeTensorInto = quantized_matmul.dequantizeTensorInto;
    pub const quantizeRowForDType = quantized_matmul.quantizeRowForDType;
    pub const dequantizeRowForDType = quantized_matmul.dequantizeRowForDType;
    pub const hasCompactColOuter = quantized_matmul.hasCompactColOuter;
    pub const matmulCompactColOuter = quantized_matmul.matmulCompactColOuter;
    pub const matmulCompactQ8_Kx4ColOuter = quantized_matmul.matmulCompactQ8_Kx4ColOuter;
    pub const matmulCompactRhsTile = quantized_matmul.matmulCompactRhsTile;
    pub const quantizedMatmulRhsQ5_KFromBlocks = quantized_matmul.q8k.quantizedMatmulRhsQ5_KFromBlocks;
    pub const quantizedMatmulRhsTQ2_0FromBorrowedBlocks = quantized_matmul.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks;
    pub const BlockTQ2_0Folded = quantized_matmul.BlockTQ2_0Folded;
    pub const BlockTQ2_0x4 = quantized_matmul.BlockTQ2_0x4;
};

/// The one dtype -> packed RHS container map. Dense f32/f16/bf16 weights
/// share the f32 output-row panel; block-quantized weights take the
/// ISA-best lane pack from `quant.PackedQuantRhsFor` (the single quant
/// switch). Every container carries `pub const dtype`, so
/// `@TypeOf(rhs) == PackedRhsFor(rhs.dtype)` names the default pack and a
/// different container of the same dtype is an explicit layout choice (the
/// Q4_K x8 pack on an MMLA target).
pub fn PackedRhsFor(comptime dt: DType) type {
    return switch (dt) {
        .f32, .f16, .bf16 => PackedDenseRhs,
        else => quantized_matmul.PackedQuantRhsFor(dt),
    };
}

pub const Tensor = tensor.Tensor;
pub const TensorOf = tensor.TensorOf;
pub const ThreadPool = thread.Pool;
// The one conformance-checked CPU provider. Microbench/parity escape hatch
// ONLY (bench/backend.zig and parity_test.zig hold its entries to the
// scalar reference arms): production code above this band goes through
// `kernels`, `blas`, or `simd`, never through the provider by name.
pub const native_impl = @import("backend/native.zig");
// GPU GEMM provider selected by -Dgpu (metal.zig or cuda.zig, via the
// backend/gpu.zig leaf); inert (never analyzed past the `enabled` flag) on
// -Dgpu=none builds. This is the SECOND conformed contract: providers are
// checked against backend/gpu_provider.zig's interface, so dispatch through
// `gpu_impl` stays provider-neutral.
pub const gpu_impl = @import("backend/gpu.zig").impl;
// The accelerator seam every band above the backend uses instead of the
// provider: capability queries, resident storage, and the offload entries
// (quantized GEMM, attention, grouped MoE, the ES flat kernels).
pub const offload = @import("backend/offload.zig");
// Pure-Zig portable SIMD kernel library backing the native backend
// (single implementation, backend-independent). Exported wholesale for the
// microbenches that compare its row-kernel and blocked paths directly; the
// curated upper-band vocabulary is `simd` below.
pub const vector_impl = @import("backend/vector.zig");
// Row-kernel vocabulary seam: the Task payloads the fused row kernels in
// `kernels` take (slices and dims, plus a ranked shape/stride array or an
// optional RankedTensor mask where a kernel walks a strided view), the
// inner-lane `run*Task` pool adapters, the comptime task factories (the
// fused activation+quantize workers) and the small shared helpers
// (`dropoutKeepCutoff`, `rowSumSq`, `coordinateForLinear`, the weight-grad
// block rows). The exec domain modules import those by name from here;
// the kernel entries themselves they reach through `kernels`, dispatched
// by value through `ExecContext.dispatchRangeOr`.
// Single implementation, `backend/vector/rows.zig`.
const rows_impl = @import("backend/vector/rows.zig");
/// The row-kernel seam, curated: the Task payloads (each the parameter of
/// a kernel in `kernels` or of an adapter here), the inner-lane `run*Task`
/// adapters, the fused-activation task factory and the shared helpers.
/// `conformSeam` checks the pairing; the kernel bodies are not reachable
/// through this namespace.
pub const rows = struct {
    pub const CrossEntropyBackwardRowsTask = rows_impl.CrossEntropyBackwardRowsTask;
    pub const CrossEntropyLossRowsTask = rows_impl.CrossEntropyLossRowsTask;
    pub const DropoutRangeTask = rows_impl.DropoutRangeTask;
    pub const FusedActKind = rows_impl.FusedActKind;
    pub const FusedActQuantTask = rows_impl.FusedActQuantTask;
    pub const LayerNormBackwardInputRowsTask = rows_impl.LayerNormBackwardInputRowsTask;
    pub const LayerNormInnerTask = rows_impl.LayerNormInnerTask;
    pub const LayerNormParamGradColumnsTask = rows_impl.LayerNormParamGradColumnsTask;
    pub const LayerNormRowStatsTask = rows_impl.LayerNormRowStatsTask;
    pub const LayerNormRowsTask = rows_impl.LayerNormRowsTask;
    pub const RmsNormBackwardInputInnerTask = rows_impl.RmsNormBackwardInputInnerTask;
    pub const RmsNormInnerTask = rows_impl.RmsNormInnerTask;
    pub const RmsNormMulAddRowsTask = rows_impl.RmsNormMulAddRowsTask;
    pub const RmsNormMulBackwardInputRowsTask = rows_impl.RmsNormMulBackwardInputRowsTask;
    pub const RmsNormMulBackwardWeightRowsTask = rows_impl.RmsNormMulBackwardWeightRowsTask;
    pub const RmsNormMulRopeHalfTask = rows_impl.RmsNormMulRopeHalfTask;
    pub const RmsNormMulRowsTask = rows_impl.RmsNormMulRowsTask;
    pub const RmsNormWeightGradBlocksTask = rows_impl.RmsNormWeightGradBlocksTask;
    pub const RmsNormWeightGradReduceTask = rows_impl.RmsNormWeightGradReduceTask;
    pub const ScatterAddRowsTask = rows_impl.ScatterAddRowsTask;
    pub const SoftmaxBackwardInnerTask = rows_impl.SoftmaxBackwardInnerTask;
    pub const SoftmaxBackwardRowsTask = rows_impl.SoftmaxBackwardRowsTask;
    pub const SoftmaxExtRowsTask = rows_impl.SoftmaxExtRowsTask;
    pub const SoftmaxInnerTask = rows_impl.SoftmaxInnerTask;
    pub const SoftmaxRowsTask = rows_impl.SoftmaxRowsTask;
    pub const SplitSwiGluBackwardTask = rows_impl.SplitSwiGluBackwardTask;
    pub const SplitSwiGluQuantQ8_0x4Task = rows_impl.SplitSwiGluQuantQ8_0x4Task;
    pub const SplitSwiGluTask = rows_impl.SplitSwiGluTask;
    pub const StandardizeBackwardInnerTask = rows_impl.StandardizeBackwardInnerTask;
    pub const StandardizeInnerTask = rows_impl.StandardizeInnerTask;
    pub const VarianceInnerTask = rows_impl.VarianceInnerTask;
    pub const coordinateForLinear = rows_impl.coordinateForLinear;
    pub const dropoutKeepCutoff = rows_impl.dropoutKeepCutoff;
    pub const rms_weight_grad_block_rows = rows_impl.rms_weight_grad_block_rows;
    pub const rowSumSq = rows_impl.rowSumSq;
    pub const runLayerNormInnerTask = rows_impl.runLayerNormInnerTask;
    pub const runLogSoftmaxInnerTask = rows_impl.runLogSoftmaxInnerTask;
    pub const runLogsumexpInnerTask = rows_impl.runLogsumexpInnerTask;
    pub const runRmsNormBackwardInputInnerTask = rows_impl.runRmsNormBackwardInputInnerTask;
    pub const runRmsNormInnerTask = rows_impl.runRmsNormInnerTask;
    pub const runSoftmaxBackwardInnerTask = rows_impl.runSoftmaxBackwardInnerTask;
    pub const runSoftmaxInnerTask = rows_impl.runSoftmaxInnerTask;
    pub const runSplitSwiGluQuantQ8_0x4Task = rows_impl.runSplitSwiGluQuantQ8_0x4Task;
    pub const runStandardizeBackwardInnerTask = rows_impl.runStandardizeBackwardInnerTask;
    pub const runStandardizeInnerTask = rows_impl.runStandardizeInnerTask;
    pub const runVarianceInnerTask = rows_impl.runVarianceInnerTask;
};
// Attention-kernel vocabulary seam: the Task payloads, `run*Task` pool
// adapters and tile constants of the grouped-causal attention family.
// Single implementation, `backend/vector/attention.zig`; the kernel
// entries themselves are reached through `kernels`.
const attention_impl = @import("backend/vector/attention.zig");
/// The attention seam, curated like `rows`: Task payloads, `run*Task`
/// adapters, the tile constants and the KV-layout helpers.
pub const attention = struct {
    pub const AttentionBackwardReduceTask = attention_impl.AttentionBackwardReduceTask;
    pub const GroupedCausalAttentionBackwardTask = attention_impl.GroupedCausalAttentionBackwardTask;
    pub const GroupedCausalAttentionBackwardTiledTask = attention_impl.GroupedCausalAttentionBackwardTiledTask;
    pub const GroupedCausalAttentionMultiTask = attention_impl.GroupedCausalAttentionMultiTask;
    pub const GroupedCausalAttentionPairTask = attention_impl.GroupedCausalAttentionPairTask;
    pub const GroupedCausalAttentionTask = attention_impl.GroupedCausalAttentionTask;
    pub const GroupedCausalAttentionTiledTask = attention_impl.GroupedCausalAttentionTiledTask;
    pub const attentionTileKeyCount = attention_impl.attentionTileKeyCount;
    pub const attention_bwd_blas_tile_rows = attention_impl.attention_bwd_blas_tile_rows;
    pub const attention_bwd_tile_rows = attention_impl.attention_bwd_tile_rows;
    pub const attention_q8_max_d = attention_impl.attention_q8_max_d;
    pub const attention_tile_max_d = attention_impl.attention_tile_max_d;
    pub const attention_tile_rows = attention_impl.attention_tile_rows;
    pub const attention_tiled_min_q_seq = attention_impl.attention_tiled_min_q_seq;
    pub const hasAdjacentKvHeadPairs = attention_impl.hasAdjacentKvHeadPairs;
    pub const kvDtypeOf = attention_impl.kvDtypeOf;
    pub const kvRowElems = attention_impl.kvRowElems;
    pub const runAttentionBackwardReduceTask = attention_impl.runAttentionBackwardReduceTask;
    pub const runGroupedCausalAttentionBackwardBlasTiledTask = attention_impl.runGroupedCausalAttentionBackwardBlasTiledTask;
    pub const runGroupedCausalAttentionBackwardTask = attention_impl.runGroupedCausalAttentionBackwardTask;
    pub const runGroupedCausalAttentionBackwardTiledTask = attention_impl.runGroupedCausalAttentionBackwardTiledTask;
    pub const runGroupedCausalAttentionMultiTask = attention_impl.runGroupedCausalAttentionMultiTask;
    pub const runGroupedCausalAttentionPairTask = attention_impl.runGroupedCausalAttentionPairTask;
    pub const runGroupedCausalAttentionTask = attention_impl.runGroupedCausalAttentionTask;
    pub const runGroupedCausalAttentionTiledTask = attention_impl.runGroupedCausalAttentionTiledTask;
};
// The one proportional range splitter (`forRange`/`reduceRange`) behind
// the vector kernels' parallel dispatch, exported for the upper bands'
// own chunked loops (the ag elementwise VJP maps ride `forRange`): the
// boundary formula `i * total / n` is the shared bitwise contract.
pub const tile = @import("backend/vector/tile.zig");

/// Provider extension: strided-view BLAS GEMM plus the nested-scope guard
/// and the folded-ternary BLAS arm, available only on BLAS-backed native
/// builds. The CBLAS provider itself is `backend/blas.zig` (the one
/// `cblas_sgemm` extern, the vendor thread setters, the MKL nested scope);
/// this namespace re-exports it. Deliberately OUTSIDE the conformed
/// `kernels` set (the reference leg has no BLAS): callers gate on
/// `blas.available` at comptime and fall back to the portable route, so a
/// scalar or no-BLAS build never analyzes the aliases.
pub const blas = if (build_options.backend_kind == .native and build_options.use_blas) struct {
    const provider = @import("backend/blas.zig");
    pub const available = true;
    pub const sgemmStrided = provider.gemmStrided;
    pub const NestedScope = provider.NestedScope;
    pub const beginNestedScope = provider.beginNestedScope;
    pub const endNestedScope = provider.endNestedScope;
    pub const matmulFoldedx4 = native_impl.matmulFoldedx4Blas;
} else struct {
    pub const available = false;
};

/// Portable-SIMD vocabulary seam: the machine vector type and the
/// transcendental/VJP helpers upper bands may use directly. Single
/// implementation, backend-independent; everything provider-varying stays
/// behind `kernels`. `fucina.simd` re-exports the public subset.
pub const simd = struct {
    const vector_common = @import("backend/vector/common.zig");
    const vector_primitives = @import("backend/vector/primitives.zig");
    pub const Vf32 = vector_common.Vf32;
    pub const vector_len = vector_common.vector_len;
    pub const vexpf = vector_primitives.vexpf;
    pub const sigmoidVec = vector_primitives.sigmoidVec;
    pub const tanhVec = vector_primitives.tanhVec;
    pub const unaryVjpVectorizes = vector_primitives.unaryVjpVectorizes;
    pub const vecUnaryVjp = vector_primitives.vecUnaryVjp;
    pub const dotF32F16 = vector_primitives.dotF32F16;
};

/// conv2d geometry (channel-last [H,W,Cin] -> [OH,OW,Cout]); see vector/conv.zig.
pub const Conv2dDims = vector_impl.conv.Conv2dDims;
/// conv1d geometry (general non-causal [T,Cin] -> [T_out,Cout]); see vector/conv.zig.
pub const Conv1dDims = vector_impl.conv.Conv1dDims;
/// pool2d geometry + kind (channel-last [H,W,C] -> [OH,OW,C]); see vector/pool.zig.
pub const PoolKind = vector_impl.pool.PoolKind;
pub const Pool2dDims = vector_impl.pool.Pool2dDims;
/// Winograd F(2×2,3×3) transform geometry; see vector/winograd.zig.
pub const WinogradF2Dims = vector_impl.winograd.F2Dims;

pub const Kind = enum {
    scalar,
    native,
};

/// Build identity, not a provider choice: `.scalar` means the reference
/// build (`isa.reference`), whose kernels are the scalar arms inside the
/// one provider.
pub const active_kind: Kind = switch (build_options.backend_kind) {
    .scalar => .scalar,
    .native => .native,
};

pub const native_blas_kind = build_options.blas_kind;
pub const native_uses_blas = build_options.use_blas;
pub const native_uses_accelerate = build_options.blas_kind == .accelerate;
pub const native_blas_threads = build_options.blas_threads;

pub const ParallelConfig = native_impl.ParallelConfig;

/// The provider's kernel set: the declaration list of `native.kernels` is
/// the interface (`conformKernels` below states the `pc`-first rule).
pub const kernels = native_impl.kernels;

comptime {
    conformKernels(native_impl.kernels);
    conformSeam(rows, native_impl.kernels);
    conformSeam(attention, native_impl.kernels);
}

/// The kernel-set contract, derived from the declarations themselves
/// rather than a parallel name list: every declaration of `kernels` is
/// either a kernel function or a `pool_free_<name>` marker naming one. A
/// kernel that uses the worker pool takes `pc: ParallelConfig` as its
/// FIRST parameter and nowhere else; a kernel that takes no `pc` carries
/// the marker beside it, so dropping the pool from a signature is an
/// explicit decision rather than an accident. Generic entries (comptime
/// dtype/op/request or `anytype` containers) satisfy the same rule: their
/// `pc` parameter, when present, is concrete.
/// The seam contract: every `*Task` type a seam publishes is the payload of
/// a kernel in `kernels` (by value or behind a pointer) or of a `run*Task`
/// adapter in the seam, and every `run*Task` adapter takes a pointer to a
/// Task the seam publishes. A payload nobody dispatches, or an adapter over
/// a private Task, is a compile error; factories, helpers and constants are
/// the seam's remaining decls.
fn conformSeam(comptime Seam: type, comptime Kernels: type) void {
    comptime {
        @setEvalBranchQuota(200_000);
        for (@typeInfo(Seam).@"struct".decls) |d| {
            const value = @field(Seam, d.name);
            const is_task_type = @TypeOf(value) == type and std.mem.endsWith(u8, d.name, "Task");
            // A `run*Task` whose first parameter is a `type` is a comptime
            // factory of adapters (the typed inner-lane family); the adapter
            // it returns is checked where a kernel takes it, not here.
            const is_adapter = std.mem.startsWith(u8, d.name, "run") and std.mem.endsWith(u8, d.name, "Task") and @typeInfo(@TypeOf(value)) == .@"fn" and
                !(@typeInfo(@TypeOf(value)).@"fn".params.len > 0 and @typeInfo(@TypeOf(value)).@"fn".params[0].type == type);
            if (is_task_type) {
                if (!seamTaskIsDispatched(value, Seam, Kernels)) @compileError(
                    "seam payload `" ++ d.name ++ "` is the parameter of no kernel and no adapter",
                );
            } else if (is_adapter) {
                const params = @typeInfo(@TypeOf(value)).@"fn".params;
                if (params.len == 0 or params[0].type == null or !seamPublishesTask(Seam, payloadOf(params[0].type.?))) @compileError(
                    "seam adapter `" ++ d.name ++ "` does not take a pointer to a published Task",
                );
            }
        }
    }
}

/// The Task behind a kernel or adapter parameter type: the type itself, or
/// the pointee of a single-item pointer.
fn payloadOf(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |ptr| if (ptr.size == .one) ptr.child else T,
        else => T,
    };
}

fn seamPublishesTask(comptime Seam: type, comptime T: type) bool {
    for (@typeInfo(Seam).@"struct".decls) |d| {
        const value = @field(Seam, d.name);
        if (@TypeOf(value) == type and value == T) return true;
    }
    return false;
}

fn seamTaskIsDispatched(comptime T: type, comptime Seam: type, comptime Kernels: type) bool {
    for (@typeInfo(Kernels).@"struct".decls) |d| {
        const value = @field(Kernels, d.name);
        if (@typeInfo(@TypeOf(value)) != .@"fn") continue;
        for (@typeInfo(@TypeOf(value)).@"fn".params) |p| {
            if (p.type) |pt| if (payloadOf(pt) == T) return true;
        }
    }
    for (@typeInfo(Seam).@"struct".decls) |d| {
        const value = @field(Seam, d.name);
        if (@typeInfo(@TypeOf(value)) != .@"fn") continue;
        if (!std.mem.startsWith(u8, d.name, "run")) continue;
        for (@typeInfo(@TypeOf(value)).@"fn".params) |p| {
            if (p.type) |pt| if (payloadOf(pt) == T) return true;
        }
    }
    return false;
}

fn conformKernels(comptime Impl: type) void {
    comptime {
        @setEvalBranchQuota(20_000);
        const marker = "pool_free_";
        for (@typeInfo(Impl).@"struct".decls) |d| {
            if (std.mem.startsWith(u8, d.name, marker)) {
                if (!@hasDecl(Impl, d.name[marker.len..])) @compileError(
                    "kernel marker `" ++ d.name ++ "` names no kernel",
                );
                continue;
            }
            const info = @typeInfo(@TypeOf(@field(Impl, d.name)));
            if (info != .@"fn") @compileError(
                "kernel namespace declares non-kernel `" ++ d.name ++ "`",
            );
            const params = info.@"fn".params;
            const takes_pc = params.len > 0 and params[0].type == ParallelConfig;
            const rest = if (takes_pc) params[1..] else params;
            for (rest) |p| if (p.type == ParallelConfig) @compileError(
                "kernel `" ++ d.name ++ "` takes ParallelConfig past the first parameter",
            );
            if (takes_pc == @hasDecl(Impl, marker ++ d.name)) @compileError(
                "kernel `" ++ d.name ++ "` disagrees with its `pool_free_` marker on taking `pc: ParallelConfig` first",
            );
        }
    }
}

test {
    _ = @import("backend_tests.zig");
    _ = @import("backend/parity_test.zig");
    if (comptime build_options.use_gpu) {
        _ = @import("backend/gpu.zig"); // forwards to the active provider's tests
    }
}
