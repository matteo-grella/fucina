//! Backend selection facade: picks the kernel provider at build time
//! (`-Dbackend=native|scalar`, `-Dgpu=metal|cuda`) and re-exports the
//! shared kernel vocabulary (ops, quant block/RHS types, packed-RHS
//! layouts). `kernels` is the selected provider's kernel set, the namespace
//! `backend/interface.zig` names and checks on both providers at comptime;
//! every pool-taking kernel takes `pc: ParallelConfig` first. The fused op
//! kernels beside their orchestration in `exec/` (attention, row_ops,
//! fakequant) are backend-independent. Layer stack: docs/ARCHITECTURE.md.
const std = @import("std");
const build_options = @import("build_options");
pub const ops = @import("backend/ops.zig");
pub const packed_matmul = @import("backend/packed.zig");
pub const quantized_matmul = @import("backend/quant.zig");
const dtype_mod = @import("dtype.zig");
const tensor = @import("tensor.zig");
const thread = @import("thread.zig");

pub const dtype_info = dtype_mod;
pub const DType = dtype_mod.DType;
pub const PackedMatmulFormat = packed_matmul.PackedMatmulFormat;
pub const PackedMatmulRhsFor = packed_matmul.PackedMatmulRhsFor;
pub const PackedDenseRhs = packed_matmul.PackedDenseRhs;
pub const BlockQ1_0 = quantized_matmul.BlockQ1_0;
pub const BlockQ2_0 = quantized_matmul.BlockQ2_0;
pub const BlockQ4_0 = quantized_matmul.BlockQ4_0;
pub const BlockQ4_1 = quantized_matmul.BlockQ4_1;
pub const BlockQ5_0 = quantized_matmul.BlockQ5_0;
pub const BlockQ5_1 = quantized_matmul.BlockQ5_1;
pub const BlockQ2_K = quantized_matmul.BlockQ2_K;
pub const BlockQ3_K = quantized_matmul.BlockQ3_K;
pub const BlockQ4_K = quantized_matmul.BlockQ4_K;
pub const BlockQ5_K = quantized_matmul.BlockQ5_K;
pub const BlockQ6_K = quantized_matmul.BlockQ6_K;
pub const BlockQ8_0 = quantized_matmul.BlockQ8_0;
pub const BlockQ8_1 = quantized_matmul.BlockQ8_1;
pub const BlockQ8_K = quantized_matmul.BlockQ8_K;
pub const BlockIQ1_S = quantized_matmul.BlockIQ1_S;
pub const BlockIQ1_M = quantized_matmul.BlockIQ1_M;
pub const BlockIQ2_XXS = quantized_matmul.BlockIQ2_XXS;
pub const BlockIQ2_XS = quantized_matmul.BlockIQ2_XS;
pub const BlockIQ2_S = quantized_matmul.BlockIQ2_S;
pub const BlockIQ3_XXS = quantized_matmul.BlockIQ3_XXS;
pub const BlockIQ3_S = quantized_matmul.BlockIQ3_S;
pub const BlockIQ4_NL = quantized_matmul.BlockIQ4_NL;
pub const BlockIQ4_XS = quantized_matmul.BlockIQ4_XS;
pub const BlockTQ1_0 = quantized_matmul.BlockTQ1_0;
pub const BlockTQ2_0 = quantized_matmul.BlockTQ2_0;
pub const BlockMXFP4 = quantized_matmul.BlockMXFP4;
pub const BlockNVFP4 = quantized_matmul.BlockNVFP4;
pub const QuantizedMatmulFormat = quantized_matmul.QuantizedMatmulFormat;
pub const supports_q4_k_mmla = quantized_matmul.supports_q4_k_mmla;
pub const PackedRhsLayout = quantized_matmul.PackedRhsLayout;
pub const PackedRhsFor = quantized_matmul.PackedRhsFor;
pub const QuantizedMatmulRhs = quantized_matmul.QuantizedMatmulRhs;
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
pub const Tensor = tensor.Tensor;
pub const TensorOf = tensor.TensorOf;
pub const ThreadPool = thread.Pool;
pub const scalar_impl = @import("backend/cpu.zig");
pub const native_impl = @import("backend/native.zig");
const interface = @import("backend/interface.zig");
// GPU GEMM provider selected by -Dgpu (metal.zig or cuda.zig, via the
// backend/gpu.zig leaf); inert (never analyzed past the `enabled` flag) on
// -Dgpu=none builds.
pub const gpu_impl = @import("backend/gpu.zig").impl;
// Pure-Zig vector kernels backing the native backend, exported so the GEMM
// bench can compare the row-kernel and blocked paths directly.
pub const vector_impl = @import("backend/vector.zig");

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

pub const active_kind: Kind = switch (build_options.backend_kind) {
    .scalar => .scalar,
    .native => .native,
};

pub const native_blas_kind = build_options.blas_kind;
pub const native_uses_blas = build_options.use_blas;
pub const native_uses_accelerate = build_options.blas_kind == .accelerate;
pub const native_blas_threads = build_options.blas_threads;

const active = switch (build_options.backend_kind) {
    .scalar => scalar_impl,
    .native => native_impl,
};

pub const ParallelConfig = active.ParallelConfig;

/// The selected provider's kernel set; see `backend/interface.zig` for the
/// names and the `pc`-first signature rule.
pub const kernels = active.kernels;

comptime {
    interface.conform(scalar_impl.kernels);
    interface.conform(native_impl.kernels);
}

test {
    _ = @import("backend_tests.zig");
    _ = @import("backend/parity_test.zig");
    if (comptime build_options.use_gpu) {
        _ = @import("backend/gpu.zig"); // forwards to the active provider's tests
    }
}
