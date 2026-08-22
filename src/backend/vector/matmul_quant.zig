//! Quantized matmul dispatch: every matmul2DQ*RhsIntoWithConfig /
//! matmul2DI8BlockwiseIntoWithConfig entry point and the QuantizedRhsParallel
//! generic — one Task/run/spawn/maybeParallel body whose SplitPolicy carries
//! each format's lane grouping and split gates. Two entries keep bespoke
//! splitters for their different task shapes: the i8-blockwise path and the
//! padded packed-Q8_0x4 path. The per-block kernels live behind
//! `../quant/matmul_api.zig`; shared parallel gates (ParallelConfig,
//! i8ColumnThreadCount / matmulThreadCount) come from `common.zig` (`vm`).

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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_0, quantized_matmul.QuantizedMatmulRhsQ1_0, quantized_matmul.matmulQ1_0RhsTile, .{});
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_0, quantized_matmul.QuantizedMatmulRhsQ2_0, quantized_matmul.matmulQ2_0RhsTile, .{});
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_0, quantized_matmul.QuantizedMatmulRhsQ8_0, quantized_matmul.matmulQ8_0RhsTile, .{});
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_0, quantized_matmul.QuantizedMatmulRhsQ8_0x4, quantized_matmul.matmulQ8_0x4RhsTile, .{ .col_group = 4 });
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_0x4, quantized_matmul.QuantizedMatmulRhsQ8_0x4, quantized_matmul.matmulQ8_0x4PackedRhsTile, .{ .col_group = 4, .col_gate = .small_m_or_m128, .row_group = 4, .row_cap_groups = true });
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_0, quantized_matmul.QuantizedMatmulRhsQ4_0, quantized_matmul.matmulQ4_0RhsTile, .{});
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_1, quantized_matmul.QuantizedMatmulRhsQ4_1, quantized_matmul.matmulQ4_1RhsTile, .{});
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_0, quantized_matmul.QuantizedMatmulRhsQ5_0, quantized_matmul.matmulQ5_0RhsTile, .{});
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_1, quantized_matmul.QuantizedMatmulRhsQ5_1, quantized_matmul.matmulQ5_1RhsTile, .{});
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_K, quantized_matmul.QuantizedMatmulRhsQ2_K, quantized_matmul.matmulQ2_KRhsTile, .{});
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_K, quantized_matmul.QuantizedMatmulRhsQ3_K, quantized_matmul.matmulQ3_KRhsTile, .{});
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_K, quantized_matmul.QuantizedMatmulRhsQ4_K, quantized_matmul.matmulQ4_KRhsTile, .{});
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_K, quantized_matmul.QuantizedMatmulRhsQ4_Kx4, quantized_matmul.matmulQ4_Kx4RhsTile, .{ .col_group = 4 });
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_K, quantized_matmul.QuantizedMatmulRhsQ4_Kx8, quantized_matmul.matmulQ4_Kx8RhsTile, .{ .col_group = 8, .col_gate = .small_m_x2, .row_cap_3_at_m128 = true });
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_Kx4, quantized_matmul.QuantizedMatmulRhsQ4_Kx8, quantized_matmul.matmulQ4_Kx8Q8_Kx4RhsTile, .{ .col_group = 8, .col_gate = .always, .row_group = 4, .row_round = .ceil_clamp });
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_K, quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla, quantized_matmul.matmulQ4_Kx2MmlaRhsTile, .{ .col_group = 2, .col_gate = .small_m_x2 });
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_Kx2Mmla, quantized_matmul.QuantizedMatmulRhsQ4_Kx2Mmla, quantized_matmul.matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsTile, .{ .col_group = 2, .col_gate = .small_m_x2, .row_group = 2 });
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_K, quantized_matmul.QuantizedMatmulRhsQ5_Kx8, quantized_matmul.matmulQ5_Kx8RhsTile, .{ .col_group = 8, .col_gate = .small_m_x2, .row_cap_3_at_m128 = true });
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_Kx4, quantized_matmul.QuantizedMatmulRhsQ5_Kx8, quantized_matmul.matmulQ5_Kx8Q8_Kx4RhsTile, .{ .col_group = 8, .col_gate = .small_m_x2, .row_group = 4 });
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_K, quantized_matmul.QuantizedMatmulRhsQ5_K, quantized_matmul.matmulQ5_KRhsTile, .{});
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_K, quantized_matmul.QuantizedMatmulRhsQ6_K, quantized_matmul.matmulQ6_KRhsTile, .{});
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
    const Parallel = QuantizedRhsParallel(quantized_matmul.BlockQ8_K, quantized_matmul.QuantizedMatmulRhsQ6_Kx4, quantized_matmul.matmulQ6_Kx4RhsTile, .{ .col_group = 4 });
    if (Parallel.maybeParallel(config, out, lhs_blocks, rhs, m, n, k)) return;
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
        .{},
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
        .{},
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
        .{},
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
        .{},
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

const Q8_0x4PackedPaddedRhsTask = struct {
    out: []f32,
    lhs_blocks: []const quantized_matmul.BlockQ8_0x4,
    rhs: *const quantized_matmul.QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    c0: usize,
    c1: usize,
};

fn runI8BlockwiseTask(task: *const I8BlockwiseTask) void {
    quantized_matmul.matmulI8BlockwiseTile(task.out, task.qa, task.a_scales, task.qw, task.w_scales, task.n, task.k, task.group_size, task.num_groups, task.r0, task.r1, task.c0, task.c1);
}

fn runQ8_0x4PackedPaddedRhsTask(task: *const Q8_0x4PackedPaddedRhsTask) void {
    quantized_matmul.matmulQ8_0x4PackedPaddedRhsTile(task.out, task.lhs_blocks, task.rhs, task.m, task.n, task.c0, task.c1);
}

fn spawnI8BlockwiseTasks(pool: *thread.Pool, tasks: []I8BlockwiseTask) void {
    pool.parallelChunks(I8BlockwiseTask, tasks, runI8BlockwiseTask);
}

fn spawnQ8_0x4PackedPaddedRhsTasks(pool: *thread.Pool, tasks: []Q8_0x4PackedPaddedRhsTask) void {
    pool.parallelChunks(Q8_0x4PackedPaddedRhsTask, tasks, runQ8_0x4PackedPaddedRhsTask);
}

/// Row/column split policy for `QuantizedRhsParallel`. The default is the
/// plain-block behavior; the interleaved packs round task boundaries to
/// their lane group so a task never bisects a packed column or row group.
const SplitPolicy = struct {
    /// Column-task boundaries are multiples of this (the pack's column
    /// lane group).
    col_group: usize = 1,
    /// When the column arm may fire (the row arm handles the rest).
    col_gate: enum {
        /// Decode-sized m only (the plain-block rule).
        small_m,
        /// Twice the m window: the x8/x2 tiles amortize more per column.
        small_m_x2,
        /// Also at m >= 128 (packed-LHS wide batches favor columns).
        small_m_or_m128,
        /// Columns whenever the count gate allows (lane-packed LHS).
        always,
    } = .small_m,
    /// Row-task boundaries are multiples of this (the packed LHS's row
    /// group).
    row_group: usize = 1,
    /// Row-group rounding: `.floor` never emits a partial group (those
    /// callers guarantee `m % row_group == 0`); `.ceil_clamp` covers a
    /// ragged tail by clamping the last boundary to `m`.
    row_round: enum { floor, ceil_clamp } = .floor,
    /// Cap the row-arm thread count at the group count, bailing to the
    /// serial path when no full group exists.
    row_cap_groups: bool = false,
    /// At m >= 128 the x8 tiles are LHS-bandwidth bound: 3 threads
    /// saturate, more only adds barrier cost.
    row_cap_3_at_m128: bool = false,
};

fn QuantizedRhsParallel(
    comptime LhsBlock: type,
    comptime Rhs: type,
    comptime tileFn: anytype,
    comptime policy: SplitPolicy,
) type {
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
            const row_groups = switch (policy.row_round) {
                .floor => m / policy.row_group,
                .ceil_clamp => (m + policy.row_group - 1) / policy.row_group,
            };
            if (policy.row_cap_groups and row_groups == 0) return false;

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

            const col_arm = switch (policy.col_gate) {
                .small_m => m < parallel.vector_column_min_m,
                .small_m_x2 => m < parallel.vector_column_min_m * 2,
                .small_m_or_m128 => m < parallel.vector_column_min_m or m >= 128,
                .always => true,
            };
            if (col_arm) {
                const col_threads = i8ColumnThreadCount(m, n, k);
                if (col_threads != 1) {
                    const group_count = n / policy.col_group;
                    for (0..col_threads) |ti| {
                        tasks[ti] = base;
                        tasks[ti].c0 = policy.col_group * (ti * group_count / col_threads);
                        tasks[ti].c1 = policy.col_group * ((ti + 1) * group_count / col_threads);
                    }
                    Self.spawn(pool, tasks[0..col_threads]);
                    return true;
                }
            }

            const requested_threads = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
            var thread_count = if (policy.row_cap_3_at_m128 and m >= 128)
                @min(requested_threads, @as(usize, 3))
            else
                requested_threads;
            if (policy.row_cap_groups) thread_count = @min(thread_count, row_groups);
            if (thread_count == 1) return false;

            for (0..thread_count) |ti| {
                tasks[ti] = base;
                tasks[ti].r0 = policy.row_group * (ti * row_groups / thread_count);
                const r1 = policy.row_group * ((ti + 1) * row_groups / thread_count);
                tasks[ti].r1 = if (policy.row_round == .ceil_clamp) @min(m, r1) else r1;
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

test {
    _ = @import("matmul_quant_tests.zig");
}
