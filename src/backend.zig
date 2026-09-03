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
const dtype_mod = @import("dtype.zig");
const tensor = @import("tensor.zig");
const thread = @import("thread.zig");

pub const dtype_info = dtype_mod;
pub const DType = dtype_mod.DType;
pub const PackedDenseRhs = packed_matmul.PackedDenseRhs;
/// The quantized matmul module (`backend/quant.zig`): its own `pub` surface
/// is what the core bands use (exec, moe, store, weights, ag): the RHS
/// containers and descriptors in `quant.types`, the block-count rule, the
/// pack/dequantize/get-rows entries and the compact tiles the MoE engine
/// schedules. Its per-format kernel children (`quant.q8k`, `quant.q4_k`,
/// ...) are the research surface: bench, tools and the models band reach
/// them, the core bands take every kernel through `kernels` (the
/// backend-door rule of `arch-check` names the children).
pub const quant = @import("backend/quant.zig");

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
        else => quant.PackedQuantRhsFor(dt),
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
/// The row-kernel seam, `backend/vector/rows.zig`: the Task payloads the
/// fused row kernels in `kernels` take (slices and dims, plus a ranked
/// shape/stride array or an optional RankedTensor mask where a kernel
/// walks a strided view), the fused activation+quantize task factory and
/// the small shared helpers (`dropoutKeepCutoff`, `rowSumSq`,
/// `coordinateForLinear`, the weight-grad block rows). The kernel bodies
/// live in `rows/kernels.zig` beneath it, imported there privately, so
/// they are not reachable through this namespace: the exec domain modules
/// take the payloads from here and the kernels through `kernels`,
/// dispatched by value with their range through `ExecContext.forRange`.
pub const rows = vector_impl.rows;
/// The attention seam, `backend/vector/attention.zig`, shaped like `rows`:
/// the Task payloads, the `run*Task` adapters the per-part-scratch
/// dispatches hand to `Pool.parallelChunks`, the tile constants and the
/// KV-layout helpers; the bodies live in `attention/kernels.zig` beneath it.
pub const attention = vector_impl.attention;
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
}

/// The kernel-set contract, read from the signatures themselves rather
/// than a parallel name list: every declaration of `kernels` is a kernel
/// function, and a kernel that uses the worker pool takes
/// `pc: ParallelConfig` as its FIRST parameter and nowhere else. A kernel
/// whose first parameter is anything else is pool-free; the inventory
/// snippet in the reference counts both kinds from the same signatures.
/// Generic entries (comptime dtype/op/request or `anytype` containers)
/// satisfy the same rule: their `pc` parameter, when present, is concrete.
fn conformKernels(comptime Impl: type) void {
    comptime {
        @setEvalBranchQuota(20_000);
        for (@typeInfo(Impl).@"struct".decls) |d| {
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
