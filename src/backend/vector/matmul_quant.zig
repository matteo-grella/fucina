//! Quantized matmul dispatch relocated out of vector.zig: every
//! matmul2DQ*RhsIntoWithConfig / matmul2DI8BlockwiseIntoWithConfig entry point,
//! the per-format Task structs and their run/spawn helpers, the
//! QuantizedRhsParallel generic, and the maybeParallel* row/column splitters.
//! The actual per-block kernels live behind quant/matmul_api.zig. Shared-core
//! symbols (ParallelConfig, the i8ColumnThreadCount / matmulThreadCount gates)
//! are aliased from vector.zig (`vm`) so the moved bodies compile unchanged.

const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const parallel = @import("../../parallel.zig");
const quantized_matmul = @import("../quant/matmul_api.zig");
const thread = @import("../../thread.zig");
const vm = @import("common.zig");

const DType = dtype_mod.DType;

// Shared-core symbols defined in vector.zig, aliased so the moved bodies compile
// unchanged.
const ParallelConfig = vm.ParallelConfig;
const i8ColumnThreadCount = vm.i8ColumnThreadCount;
const matmulThreadCount = vm.matmulThreadCount;

// ---------------- Quantized MatMul (2-D) ----------------

pub fn matmul2DI8BlockwiseIntoWithConfig(
    out: []f32,
    qa: []const i8,
    a_scales: []const f32,
    qw: []const i8,
    w_scales: []const f32,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    num_groups: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelI8Blockwise(config, out, qa, a_scales, qw, w_scales, m, n, k, group_size, num_groups)) return;
    quantized_matmul.matmulI8BlockwiseRange(out, qa, a_scales, qw, w_scales, m, n, k, group_size, num_groups, 0, m);
}

pub fn matmul2DQ1_0RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ1_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_0, quantized_matmul.QuantizedMatmulRhsQ1_0, quantized_matmul.matmulQ1_0RhsTile);
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ1_0RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ2_0RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ2_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_0, quantized_matmul.QuantizedMatmulRhsQ2_0, quantized_matmul.matmulQ2_0RhsTile);
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ2_0RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ8_0RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ8_0Rhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ8_0RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ8_0x4RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ8_0x4Rhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ8_0x4RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ8_0x4PackedRhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ8_0x4PackedRhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ8_0x4PackedRhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ8_0x4PackedPaddedRhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ8_0x4PackedPaddedRhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ8_0x4PackedPaddedRhsRange(out, lhs_blocks, rhs, m, n);
}

pub fn matmul2DQ4_0RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ4_0Rhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ4_0RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ4_1RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_1,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_1,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_1, quantized_matmul.QuantizedMatmulRhsQ4_1, quantized_matmul.matmulQ4_1RhsTile);
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ4_1RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ5_0RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_0, quantized_matmul.QuantizedMatmulRhsQ5_0, quantized_matmul.matmulQ5_0RhsTile);
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ5_0RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ5_1RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_1,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_1,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_1, quantized_matmul.QuantizedMatmulRhsQ5_1, quantized_matmul.matmulQ5_1RhsTile);
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ5_1RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ2_KRhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ2_K,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ2_KRhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ2_KRhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ3_KRhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ3_K,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ3_KRhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ3_KRhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ4_KRhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_K,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ4_KRhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ4_KRhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ4_Kx4RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx4,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ4_Kx4Rhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ4_Kx4RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ4_Kx8RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ4_Kx8Rhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ4_Kx8RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ4_Kx8Q8_Kx4RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_Kx4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ4_Kx8Q8_Kx4Rhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ4_Kx8Q8_Kx4RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ4_Kx2MmlaRhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ4_Kx2MmlaRhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ4_Kx2MmlaRhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ4_Kx2MmlaQ8_Kx2MmlaRhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_Kx2Mmla,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ4_Kx2MmlaQ8_Kx2MmlaRhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ5_Kx8RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ5_Kx8Rhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ5_Kx8RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ5_Kx8Q8_Kx4RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_Kx4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ5_Kx8Q8_Kx4Rhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ5_Kx8Q8_Kx4RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ5_KRhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_K,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ5_KRhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ5_KRhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ6_KRhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_K,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ6_KRhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ6_KRhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DQ6_Kx4RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_Kx4,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    if (maybeParallelQ6_Kx4Rhs(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulQ6_Kx4RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DTableQ8_0RhsIntoWithConfig(
    comptime rhs_dtype: DType,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsRowsFor(rhs_dtype),
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const Parallel = QuantizedRhsParallel(
        quantized_matmul.BlockQ8_0,
        quantized_matmul.QuantizedMatmulRhsRowsFor(rhs_dtype),
        TableQ8_0Tile(rhs_dtype).run,
    );
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulTableQ8_0RhsRange(rhs_dtype, out, lhs_blocks, rhs, m, n, 0, m);
}

pub fn matmul2DTQ2_0RhsIntoWithConfig(
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsTQ2_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const Parallel = QuantizedRhsParallel(
        quantized_matmul.BlockQ8_K,
        quantized_matmul.QuantizedMatmulRhsTQ2_0,
        quantized_matmul.matmulTQ2_0RhsTile,
    );
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulTQ2_0RhsRange(out, lhs_blocks, rhs, m, n, 0, m);
}

/// Dense f32 LHS x TQ2_0 RHS (the mul-free no-activation-quant path). Each
/// output element is one full dotTQ2_0F32, so row/column splits never change
/// the accumulation order — parallel results stay bitwise serial-identical.
pub fn matmul2DTQ2_0F32RhsIntoWithConfig(
    out: []f32,
    lhs: []const f32,
    rhs: *const quantized_matmul.QuantizedMatmulRhsTQ2_0,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const Parallel = QuantizedRhsParallel(
        f32,
        quantized_matmul.QuantizedMatmulRhsTQ2_0,
        quantized_matmul.matmulTQ2_0F32RhsTile,
    );
    if (Parallel.maybeParallel(config, out, lhs, rhs, m, n, k)) return;
    quantized_matmul.matmulTQ2_0F32RhsRange(out, lhs, rhs, m, n, 0, m);
}

pub fn matmul2DTableQ8_KRhsIntoWithConfig(
    comptime rhs_dtype: DType,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsRowsFor(rhs_dtype),
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const Parallel = QuantizedRhsParallel(
        quantized_matmul.BlockQ8_K,
        quantized_matmul.QuantizedMatmulRhsRowsFor(rhs_dtype),
        TableQ8_KTile(rhs_dtype).run,
    );
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
    quantized_matmul.matmulTableQ8_KRhsRange(rhs_dtype, out, lhs_blocks, rhs, m, n, 0, m);
}

const I8BlockwiseTask = struct {
    out: []f32,
    qa: []const i8,
    a_scales: []const f32,
    qw: []const i8,
    w_scales: []const f32,
    n: usize,
    k: usize,
    group_size: usize,
    num_groups: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q8_0RhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q8_0x4RhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q8_0x4PackedRhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q8_0x4PackedPaddedRhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    c0: usize,
    c1: usize,
};

const Q4_0RhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_0,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q2_KRhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ2_K,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q3_KRhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ3_K,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q4_KRhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_K,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q4_Kx4RhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx4,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q4_Kx8RhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q4_Kx8Q8_Kx4RhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_Kx4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q4_Kx2MmlaRhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q4_Kx2MmlaQ8_Kx2MmlaRhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_Kx2Mmla,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q5_Kx8RhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q5_Kx8Q8_Kx4RhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_Kx4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q5_KRhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_K,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q6_KRhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_K,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

const Q6_Kx4RhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_Kx4,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
};

fn runI8BlockwiseTask(task: *const I8BlockwiseTask) void {
    quantized_matmul.matmulI8BlockwiseTile(task.out, task.qa, task.a_scales, task.qw, task.w_scales, task.n, task.k, task.group_size, task.num_groups, task.r0, task.r1, task.c0, task.c1);
}

fn runQ8_0RhsTask(task: *const Q8_0RhsTask) void {
    quantized_matmul.matmulQ8_0RhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ8_0x4RhsTask(task: *const Q8_0x4RhsTask) void {
    quantized_matmul.matmulQ8_0x4RhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ8_0x4PackedRhsTask(task: *const Q8_0x4PackedRhsTask) void {
    quantized_matmul.matmulQ8_0x4PackedRhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ8_0x4PackedPaddedRhsTask(task: *const Q8_0x4PackedPaddedRhsTask) void {
    quantized_matmul.matmulQ8_0x4PackedPaddedRhsTile(task.out, task.lhs_blocks, task.rhs, task.m, task.n, task.c0, task.c1);
}

fn runQ4_0RhsTask(task: *const Q4_0RhsTask) void {
    quantized_matmul.matmulQ4_0RhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ2_KRhsTask(task: *const Q2_KRhsTask) void {
    quantized_matmul.matmulQ2_KRhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ3_KRhsTask(task: *const Q3_KRhsTask) void {
    quantized_matmul.matmulQ3_KRhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ4_KRhsTask(task: *const Q4_KRhsTask) void {
    quantized_matmul.matmulQ4_KRhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ4_Kx4RhsTask(task: *const Q4_Kx4RhsTask) void {
    quantized_matmul.matmulQ4_Kx4RhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ4_Kx8RhsTask(task: *const Q4_Kx8RhsTask) void {
    quantized_matmul.matmulQ4_Kx8RhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ4_Kx8Q8_Kx4RhsTask(task: *const Q4_Kx8Q8_Kx4RhsTask) void {
    quantized_matmul.matmulQ4_Kx8Q8_Kx4RhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ4_Kx2MmlaRhsTask(task: *const Q4_Kx2MmlaRhsTask) void {
    quantized_matmul.matmulQ4_Kx2MmlaRhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ4_Kx2MmlaQ8_Kx2MmlaRhsTask(task: *const Q4_Kx2MmlaQ8_Kx2MmlaRhsTask) void {
    quantized_matmul.matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ5_Kx8RhsTask(task: *const Q5_Kx8RhsTask) void {
    quantized_matmul.matmulQ5_Kx8RhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ5_Kx8Q8_Kx4RhsTask(task: *const Q5_Kx8Q8_Kx4RhsTask) void {
    quantized_matmul.matmulQ5_Kx8Q8_Kx4RhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ5_KRhsTask(task: *const Q5_KRhsTask) void {
    quantized_matmul.matmulQ5_KRhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ6_KRhsTask(task: *const Q6_KRhsTask) void {
    quantized_matmul.matmulQ6_KRhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn runQ6_Kx4RhsTask(task: *const Q6_Kx4RhsTask) void {
    quantized_matmul.matmulQ6_Kx4RhsTile(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
}

fn spawnI8BlockwiseTasks(pool: *thread.Pool, tasks: []I8BlockwiseTask) void {
    pool.parallelChunks(I8BlockwiseTask, tasks, runI8BlockwiseTask);
}

fn spawnQ8_0RhsTasks(pool: *thread.Pool, tasks: []Q8_0RhsTask) void {
    pool.parallelChunks(Q8_0RhsTask, tasks, runQ8_0RhsTask);
}

fn spawnQ8_0x4RhsTasks(pool: *thread.Pool, tasks: []Q8_0x4RhsTask) void {
    pool.parallelChunks(Q8_0x4RhsTask, tasks, runQ8_0x4RhsTask);
}

fn spawnQ8_0x4PackedRhsTasks(pool: *thread.Pool, tasks: []Q8_0x4PackedRhsTask) void {
    pool.parallelChunks(Q8_0x4PackedRhsTask, tasks, runQ8_0x4PackedRhsTask);
}

fn spawnQ8_0x4PackedPaddedRhsTasks(pool: *thread.Pool, tasks: []Q8_0x4PackedPaddedRhsTask) void {
    pool.parallelChunks(Q8_0x4PackedPaddedRhsTask, tasks, runQ8_0x4PackedPaddedRhsTask);
}

fn spawnQ4_0RhsTasks(pool: *thread.Pool, tasks: []Q4_0RhsTask) void {
    pool.parallelChunks(Q4_0RhsTask, tasks, runQ4_0RhsTask);
}

fn spawnQ2_KRhsTasks(pool: *thread.Pool, tasks: []Q2_KRhsTask) void {
    pool.parallelChunks(Q2_KRhsTask, tasks, runQ2_KRhsTask);
}

fn spawnQ3_KRhsTasks(pool: *thread.Pool, tasks: []Q3_KRhsTask) void {
    pool.parallelChunks(Q3_KRhsTask, tasks, runQ3_KRhsTask);
}

fn spawnQ4_KRhsTasks(pool: *thread.Pool, tasks: []Q4_KRhsTask) void {
    pool.parallelChunks(Q4_KRhsTask, tasks, runQ4_KRhsTask);
}

fn spawnQ4_Kx4RhsTasks(pool: *thread.Pool, tasks: []Q4_Kx4RhsTask) void {
    pool.parallelChunks(Q4_Kx4RhsTask, tasks, runQ4_Kx4RhsTask);
}

fn spawnQ4_Kx8RhsTasks(pool: *thread.Pool, tasks: []Q4_Kx8RhsTask) void {
    pool.parallelChunks(Q4_Kx8RhsTask, tasks, runQ4_Kx8RhsTask);
}

fn spawnQ4_Kx8Q8_Kx4RhsTasks(pool: *thread.Pool, tasks: []Q4_Kx8Q8_Kx4RhsTask) void {
    pool.parallelChunks(Q4_Kx8Q8_Kx4RhsTask, tasks, runQ4_Kx8Q8_Kx4RhsTask);
}

fn spawnQ4_Kx2MmlaRhsTasks(pool: *thread.Pool, tasks: []Q4_Kx2MmlaRhsTask) void {
    pool.parallelChunks(Q4_Kx2MmlaRhsTask, tasks, runQ4_Kx2MmlaRhsTask);
}

fn spawnQ4_Kx2MmlaQ8_Kx2MmlaRhsTasks(pool: *thread.Pool, tasks: []Q4_Kx2MmlaQ8_Kx2MmlaRhsTask) void {
    pool.parallelChunks(Q4_Kx2MmlaQ8_Kx2MmlaRhsTask, tasks, runQ4_Kx2MmlaQ8_Kx2MmlaRhsTask);
}

fn spawnQ5_Kx8RhsTasks(pool: *thread.Pool, tasks: []Q5_Kx8RhsTask) void {
    pool.parallelChunks(Q5_Kx8RhsTask, tasks, runQ5_Kx8RhsTask);
}

fn spawnQ5_Kx8Q8_Kx4RhsTasks(pool: *thread.Pool, tasks: []Q5_Kx8Q8_Kx4RhsTask) void {
    pool.parallelChunks(Q5_Kx8Q8_Kx4RhsTask, tasks, runQ5_Kx8Q8_Kx4RhsTask);
}

fn spawnQ5_KRhsTasks(pool: *thread.Pool, tasks: []Q5_KRhsTask) void {
    pool.parallelChunks(Q5_KRhsTask, tasks, runQ5_KRhsTask);
}

fn spawnQ6_KRhsTasks(pool: *thread.Pool, tasks: []Q6_KRhsTask) void {
    pool.parallelChunks(Q6_KRhsTask, tasks, runQ6_KRhsTask);
}

fn spawnQ6_Kx4RhsTasks(pool: *thread.Pool, tasks: []Q6_Kx4RhsTask) void {
    pool.parallelChunks(Q6_Kx4RhsTask, tasks, runQ6_Kx4RhsTask);
}

fn QuantizedRhsParallel(comptime LhsBlock: type, comptime Rhs: type, comptime tileFn: anytype) type {
    return struct {
        const Self = @This();

        const Task = struct {
            out: []f32,
            lhs_blocks: []const LhsBlock,
            rhs: *const Rhs,
            n: usize,
            r0: usize,
            r1: usize,
            c0: usize,
            c1: usize,
        };

        fn run(task: *const Task) void {
            tileFn(task.out, task.lhs_blocks, task.rhs, task.n, task.r0, task.r1, task.c0, task.c1);
        }

        fn spawn(pool: *thread.Pool, tasks: []Task) void {
            pool.parallelChunks(Task, tasks, Self.run);
        }

        fn maybeParallel(
            config: ParallelConfig,
            out: []f32,
            lhs_blocks: []const LhsBlock,
            rhs: *const Rhs,
            m: usize,
            n: usize,
            k: usize,
        ) bool {
            const pool = config.pool orelse return false;
            var tasks: [parallel.vector_max_threads]Task = undefined;
            const base: Task = .{
                .out = out,
                .lhs_blocks = lhs_blocks,
                .rhs = rhs,
                .n = n,
                .r0 = 0,
                .r1 = m,
                .c0 = 0,
                .c1 = n,
            };

            if (m < parallel.vector_column_min_m) {
                const col_threads = i8ColumnThreadCount(m, n, k);
                if (col_threads != 1) {
                    for (0..col_threads) |ti| {
                        tasks[ti] = base;
                        tasks[ti].c0 = ti * n / col_threads;
                        tasks[ti].c1 = (ti + 1) * n / col_threads;
                    }
                    Self.spawn(pool, tasks[0..col_threads]);
                    return true;
                }
            }

            const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
            if (thread_count == 1) return false;

            for (0..thread_count) |ti| {
                tasks[ti] = base;
                tasks[ti].r0 = ti * m / thread_count;
                tasks[ti].r1 = (ti + 1) * m / thread_count;
            }
            Self.spawn(pool, tasks[0..thread_count]);
            return true;
        }
    };
}

fn TableQ8_0Tile(comptime rhs_dtype: DType) type {
    return struct {
        fn run(
            out: []f32,
            lhs_blocks: []const quantized_matmul.BlockQ8_0,
            rhs: *const quantized_matmul.QuantizedMatmulRhsRowsFor(rhs_dtype),
            n: usize,
            r0: usize,
            r1: usize,
            c0: usize,
            c1: usize,
        ) void {
            quantized_matmul.matmulTableQ8_0RhsTile(rhs_dtype, out, lhs_blocks, rhs, n, r0, r1, c0, c1);
        }
    };
}

fn TableQ8_KTile(comptime rhs_dtype: DType) type {
    return struct {
        fn run(
            out: []f32,
            lhs_blocks: []const quantized_matmul.BlockQ8_K,
            rhs: *const quantized_matmul.QuantizedMatmulRhsRowsFor(rhs_dtype),
            n: usize,
            r0: usize,
            r1: usize,
            c0: usize,
            c1: usize,
        ) void {
            quantized_matmul.matmulTableQ8_KRhsTile(rhs_dtype, out, lhs_blocks, rhs, n, r0, r1, c0, c1);
        }
    };
}

fn maybeParallelI8Blockwise(
    config: ParallelConfig,
    out: []f32,
    qa: []const i8,
    a_scales: []const f32,
    qw: []const i8,
    w_scales: []const f32,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    num_groups: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]I8BlockwiseTask = undefined;
    const base: I8BlockwiseTask = .{
        .out = out,
        .qa = qa,
        .a_scales = a_scales,
        .qw = qw,
        .w_scales = w_scales,
        .n = n,
        .k = k,
        .group_size = group_size,
        .num_groups = num_groups,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    // Small m (e.g. decode): row splitting would leave one task with all the
    // work, so split output columns instead. Mirrors the float TN/NT paths, but
    // with a lower work gate: the int8 kernel is heavier per element than the
    // float GEMV, so column parallelism pays off at decode-sized work.
    if (m < parallel.vector_column_min_m) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = ti * n / col_threads;
                tasks[ti].c1 = (ti + 1) * n / col_threads;
            }
            spawnI8BlockwiseTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnI8BlockwiseTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ8_0Rhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q8_0RhsTask = undefined;
    const base: Q8_0RhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = ti * n / col_threads;
                tasks[ti].c1 = (ti + 1) * n / col_threads;
            }
            spawnQ8_0RhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ8_0RhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ8_0x4Rhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q8_0x4RhsTask = undefined;
    const base: Q8_0x4RhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            const group_count = n / 4;
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = 4 * (ti * group_count / col_threads);
                tasks[ti].c1 = 4 * ((ti + 1) * group_count / col_threads);
            }
            spawnQ8_0x4RhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ8_0x4RhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ8_0x4PackedRhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    const row_groups = m / 4;
    if (row_groups == 0) return false;

    var tasks: [parallel.vector_max_threads]Q8_0x4PackedRhsTask = undefined;
    const base: Q8_0x4PackedRhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m or m >= 128) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            const group_count = n / 4;
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = 4 * (ti * group_count / col_threads);
                tasks[ti].c1 = 4 * ((ti + 1) * group_count / col_threads);
            }
            spawnQ8_0x4PackedRhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const requested_threads = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    const thread_count = @min(requested_threads, row_groups);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = 4 * (ti * row_groups / thread_count);
        tasks[ti].r1 = 4 * ((ti + 1) * row_groups / thread_count);
    }
    spawnQ8_0x4PackedRhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ8_0x4PackedPaddedRhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    const col_threads = i8ColumnThreadCount(m, n, k);
    if (col_threads == 1) return false;

    var tasks: [parallel.vector_max_threads]Q8_0x4PackedPaddedRhsTask = undefined;
    const base: Q8_0x4PackedPaddedRhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .m = m,
        .n = n,
        .c0 = 0,
        .c1 = n,
    };
    const group_count = n / 4;
    for (0..col_threads) |ti| {
        tasks[ti] = base;
        tasks[ti].c0 = 4 * (ti * group_count / col_threads);
        tasks[ti].c1 = 4 * ((ti + 1) * group_count / col_threads);
    }
    spawnQ8_0x4PackedPaddedRhsTasks(pool, tasks[0..col_threads]);
    return true;
}

fn maybeParallelQ4_0Rhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_0,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q4_0RhsTask = undefined;
    const base: Q4_0RhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = ti * n / col_threads;
                tasks[ti].c1 = (ti + 1) * n / col_threads;
            }
            spawnQ4_0RhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ4_0RhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ2_KRhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ2_K,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q2_KRhsTask = undefined;
    const base: Q2_KRhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = ti * n / col_threads;
                tasks[ti].c1 = (ti + 1) * n / col_threads;
            }
            spawnQ2_KRhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ2_KRhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ3_KRhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ3_K,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q3_KRhsTask = undefined;
    const base: Q3_KRhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = ti * n / col_threads;
                tasks[ti].c1 = (ti + 1) * n / col_threads;
            }
            spawnQ3_KRhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ3_KRhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ4_KRhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_K,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q4_KRhsTask = undefined;
    const base: Q4_KRhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = ti * n / col_threads;
                tasks[ti].c1 = (ti + 1) * n / col_threads;
            }
            spawnQ4_KRhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ4_KRhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ5_KRhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_K,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q5_KRhsTask = undefined;
    const base: Q5_KRhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = ti * n / col_threads;
                tasks[ti].c1 = (ti + 1) * n / col_threads;
            }
            spawnQ5_KRhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ5_KRhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ4_Kx4Rhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx4,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q4_Kx4RhsTask = undefined;
    const base: Q4_Kx4RhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            const group_count = n / 4;
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = 4 * (ti * group_count / col_threads);
                tasks[ti].c1 = 4 * ((ti + 1) * group_count / col_threads);
            }
            spawnQ4_Kx4RhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ4_Kx4RhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ4_Kx8Rhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q4_Kx8RhsTask = undefined;
    const base: Q4_Kx8RhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m * 2) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            const group_count = n / 8;
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = 8 * (ti * group_count / col_threads);
                tasks[ti].c1 = 8 * ((ti + 1) * group_count / col_threads);
            }
            spawnQ4_Kx8RhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const requested_threads = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    const thread_count = if (m >= 128) @min(requested_threads, @as(usize, 3)) else requested_threads;
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ4_Kx8RhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ4_Kx8Q8_Kx4Rhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_Kx4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx8,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q4_Kx8Q8_Kx4RhsTask = undefined;
    const base: Q4_Kx8Q8_Kx4RhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    const col_threads = i8ColumnThreadCount(m, n, k);
    if (col_threads != 1) {
        const group_count = n / 8;
        for (0..col_threads) |ti| {
            tasks[ti] = base;
            tasks[ti].c0 = 8 * (ti * group_count / col_threads);
            tasks[ti].c1 = 8 * ((ti + 1) * group_count / col_threads);
        }
        spawnQ4_Kx8Q8_Kx4RhsTasks(pool, tasks[0..col_threads]);
        return true;
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    const row_groups = (m + 3) / 4;
    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = 4 * (ti * row_groups / thread_count);
        tasks[ti].r1 = @min(m, 4 * ((ti + 1) * row_groups / thread_count));
    }
    spawnQ4_Kx8Q8_Kx4RhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ4_Kx2MmlaRhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q4_Kx2MmlaRhsTask = undefined;
    const base: Q4_Kx2MmlaRhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m * 2) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            const group_count = n / 2;
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = 2 * (ti * group_count / col_threads);
                tasks[ti].c1 = 2 * ((ti + 1) * group_count / col_threads);
            }
            spawnQ4_Kx2MmlaRhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ4_Kx2MmlaRhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ4_Kx2MmlaQ8_Kx2MmlaRhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_Kx2Mmla,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q4_Kx2MmlaQ8_Kx2MmlaRhsTask = undefined;
    const base: Q4_Kx2MmlaQ8_Kx2MmlaRhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m * 2) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            const group_count = n / 2;
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = 2 * (ti * group_count / col_threads);
                tasks[ti].c1 = 2 * ((ti + 1) * group_count / col_threads);
            }
            spawnQ4_Kx2MmlaQ8_Kx2MmlaRhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    const row_groups = m / 2;
    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = 2 * (ti * row_groups / thread_count);
        tasks[ti].r1 = 2 * ((ti + 1) * row_groups / thread_count);
    }
    spawnQ4_Kx2MmlaQ8_Kx2MmlaRhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ5_Kx8Rhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q5_Kx8RhsTask = undefined;
    const base: Q5_Kx8RhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m * 2) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            const group_count = n / 8;
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = 8 * (ti * group_count / col_threads);
                tasks[ti].c1 = 8 * ((ti + 1) * group_count / col_threads);
            }
            spawnQ5_Kx8RhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const requested_threads = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    const thread_count = if (m >= 128) @min(requested_threads, @as(usize, 3)) else requested_threads;
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ5_Kx8RhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ5_Kx8Q8_Kx4Rhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_Kx4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ5_Kx8,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q5_Kx8Q8_Kx4RhsTask = undefined;
    const base: Q5_Kx8Q8_Kx4RhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m * 2) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            const group_count = n / 8;
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = 8 * (ti * group_count / col_threads);
                tasks[ti].c1 = 8 * ((ti + 1) * group_count / col_threads);
            }
            spawnQ5_Kx8Q8_Kx4RhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    const row_groups = m / 4;
    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = 4 * (ti * row_groups / thread_count);
        tasks[ti].r1 = 4 * ((ti + 1) * row_groups / thread_count);
    }
    spawnQ5_Kx8Q8_Kx4RhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ6_KRhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_K,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q6_KRhsTask = undefined;
    const base: Q6_KRhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = ti * n / col_threads;
                tasks[ti].c1 = (ti + 1) * n / col_threads;
            }
            spawnQ6_KRhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ6_KRhsTasks(pool, tasks[0..thread_count]);
    return true;
}

fn maybeParallelQ6_Kx4Rhs(
    config: ParallelConfig,
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_K,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ6_Kx4,
    m: usize,
    n: usize,
    k: usize,
) bool {
    const pool = config.pool orelse return false;
    var tasks: [parallel.vector_max_threads]Q6_Kx4RhsTask = undefined;
    const base: Q6_Kx4RhsTask = .{
        .out = out,
        .lhs_blocks = lhs_blocks,
        .rhs = rhs,
        .n = n,
        .r0 = 0,
        .r1 = m,
        .c0 = 0,
        .c1 = n,
    };

    if (m < parallel.vector_column_min_m) {
        const col_threads = i8ColumnThreadCount(m, n, k);
        if (col_threads != 1) {
            const group_count = n / 4;
            for (0..col_threads) |ti| {
                tasks[ti] = base;
                tasks[ti].c0 = 4 * (ti * group_count / col_threads);
                tasks[ti].c1 = 4 * ((ti + 1) * group_count / col_threads);
            }
            spawnQ6_Kx4RhsTasks(pool, tasks[0..col_threads]);
            return true;
        }
    }

    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;

    for (0..thread_count) |ti| {
        tasks[ti] = base;
        tasks[ti].r0 = ti * m / thread_count;
        tasks[ti].r1 = (ti + 1) * m / thread_count;
    }
    spawnQ6_Kx4RhsTasks(pool, tasks[0..thread_count]);
    return true;
}

test {
    _ = @import("matmul_quant_tests.zig");
}
