//! Internal quantized-matmul interface for backend dispatchers.
//!
//! `backend/quant.zig` remains the public compatibility barrel. Vector backend
//! children import this narrower module so they do not depend on that parent
//! barrel for quantized matmul kernels and packed RHS types.

const cold = @import("cold.zig");
const common = @import("common.zig");
const q4_k = @import("q4_k.zig");
const q5_k = @import("q5_k.zig");
const q6_k = @import("q6_k.zig");
const q8_0 = @import("q8_0.zig");
const ternary = @import("ternary.zig");
const types = @import("types.zig");

const i8DotI32 = common.i8DotI32;

pub const BlockQ8_0 = types.BlockQ8_0;
pub const BlockQ8_0x4 = types.BlockQ8_0x4;
pub const BlockQ8_1 = types.BlockQ8_1;
pub const BlockQ8_K = types.BlockQ8_K;
pub const BlockQ8_Kx2Mmla = types.BlockQ8_Kx2Mmla;
pub const BlockQ8_Kx4 = types.BlockQ8_Kx4;

pub const QuantizedMatmulRhsQ1_0 = types.QuantizedMatmulRhsQ1_0;
pub const QuantizedMatmulRhsQ2_0 = types.QuantizedMatmulRhsQ2_0;
pub const QuantizedMatmulRhsQ2_K = types.QuantizedMatmulRhsQ2_K;
pub const QuantizedMatmulRhsQ3_K = types.QuantizedMatmulRhsQ3_K;
pub const QuantizedMatmulRhsQ4_0 = types.QuantizedMatmulRhsQ4_0;
pub const QuantizedMatmulRhsQ4_1 = types.QuantizedMatmulRhsQ4_1;
pub const QuantizedMatmulRhsQ4_K = types.QuantizedMatmulRhsQ4_K;
pub const QuantizedMatmulRhsQ4_Kx2Mmla = types.QuantizedMatmulRhsQ4_Kx2Mmla;
pub const QuantizedMatmulRhsQ4_Kx4 = types.QuantizedMatmulRhsQ4_Kx4;
pub const QuantizedMatmulRhsQ4_Kx8 = types.QuantizedMatmulRhsQ4_Kx8;
pub const QuantizedMatmulRhsQ5_0 = types.QuantizedMatmulRhsQ5_0;
pub const QuantizedMatmulRhsQ5_1 = types.QuantizedMatmulRhsQ5_1;
pub const QuantizedMatmulRhsQ5_K = types.QuantizedMatmulRhsQ5_K;
pub const QuantizedMatmulRhsQ5_Kx8 = types.QuantizedMatmulRhsQ5_Kx8;
pub const QuantizedMatmulRhsQ6_K = types.QuantizedMatmulRhsQ6_K;
pub const QuantizedMatmulRhsQ6_Kx4 = types.QuantizedMatmulRhsQ6_Kx4;
pub const QuantizedMatmulRhsQ8_0 = types.QuantizedMatmulRhsQ8_0;
pub const QuantizedMatmulRhsQ8_0x4 = types.QuantizedMatmulRhsQ8_0x4;
pub const QuantizedMatmulRhsRowsFor = types.QuantizedMatmulRhsRowsFor;
pub const QuantizedMatmulRhsTQ2_0 = types.QuantizedMatmulRhsTQ2_0;

// Output columns computed together per activation chunk: one qa load feeds all
// i8_col_block columns, and the independent accumulators give the CPU ILP to
// hide multiply latency. Portable: lets LLVM vectorize / emit int8 dot ops.
const i8_col_block: usize = 4;

// Core int8 block-wise GEMM over the tile rows [r0, r1) x columns [c0, c1):
//   out[i, j] = a_scale[i] * sum_g ( w_scale[j, g] * sum_{p in group g} qa[i,p] * qw[j,p] )
// The inner per-group sum is i32 (a group of <= group_size int8 products fits
// easily), scaled into f32 once per group. Row/column wrappers below select the
// tile so the parallel dispatch can split whichever dimension is larger.
pub fn matmulI8BlockwiseTile(
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
) void {
    var i = r0;
    while (i < r1) : (i += 1) {
        const qa_row = qa[i * k ..][0..k];
        const a_scale = a_scales[i];

        var j = c0;
        while (j + i8_col_block <= c1) : (j += i8_col_block) {
            var facc = [_]f32{0} ** i8_col_block;

            var g: usize = 0;
            while (g < num_groups) : (g += 1) {
                const p0 = g * group_size;
                const p1 = @min(p0 + group_size, k);

                // Scalar i8->i32 multiply-accumulate reductions, one per blocked
                // column, sharing the qa load. Written as plain scalar reductions
                // so LLVM can lower them to the target's int8 dot instruction when
                // available, and to ordinary widening SIMD otherwise.
                var iacc = [_]i32{0} ** i8_col_block;
                var p = p0;
                while (p < p1) : (p += 1) {
                    const a_val: i32 = qa_row[p];
                    inline for (0..i8_col_block) |c| iacc[c] += a_val * @as(i32, qw[(j + c) * k + p]);
                }
                inline for (0..i8_col_block) |c| facc[c] += @as(f32, @floatFromInt(iacc[c])) * w_scales[(j + c) * num_groups + g];
            }
            inline for (0..i8_col_block) |c| out[i * n + j + c] = facc[c] * a_scale;
        }

        // Tail columns (fewer than i8_col_block left).
        while (j < c1) : (j += 1) {
            const qw_col = qw[j * k ..][0..k];
            const col_scales = w_scales[j * num_groups ..][0..num_groups];
            var acc: f32 = 0;
            var g: usize = 0;
            while (g < num_groups) : (g += 1) {
                const p0 = g * group_size;
                const p1 = @min(p0 + group_size, k);
                acc += @as(f32, @floatFromInt(i8DotI32(qa_row[p0..p1], qw_col[p0..p1]))) * col_scales[g];
            }
            out[i * n + j] = acc * a_scale;
        }
    }
}

// Computes rows [row_start, row_end), all columns. Used by the serial cpu path
// and by the row-split parallel dispatch (large m).
pub fn matmulI8BlockwiseRange(
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
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulI8BlockwiseTile(out, qa, a_scales, qw, w_scales, n, k, group_size, num_groups, row_start, row_end, 0, n);
}

pub const matmulQ1_0RhsRange = cold.matmulQ1_0RhsRange;
pub const matmulQ1_0RhsTile = cold.matmulQ1_0RhsTile;
pub const matmulQ2_0RhsRange = ternary.matmulQ2_0RhsRange;
pub const matmulQ2_0RhsTile = ternary.matmulQ2_0RhsTile;
pub const matmulQ2_0RhsRefRange = cold.matmulQ2_0RhsRefRange;
pub const matmulQ2_0RhsRefTile = cold.matmulQ2_0RhsRefTile;
pub const matmulQ2_KRhsRange = cold.matmulQ2_KRhsRange;
pub const matmulQ2_KRhsTile = cold.matmulQ2_KRhsTile;
pub const matmulQ3_KRhsRange = cold.matmulQ3_KRhsRange;
pub const matmulQ3_KRhsTile = cold.matmulQ3_KRhsTile;
pub const matmulQ4_0RhsRange = cold.matmulQ4_0RhsRange;
pub const matmulQ4_0RhsTile = cold.matmulQ4_0RhsTile;
pub const matmulQ4_1RhsRange = cold.matmulQ4_1RhsRange;
pub const matmulQ4_1RhsTile = cold.matmulQ4_1RhsTile;
pub const matmulQ5_0RhsRange = cold.matmulQ5_0RhsRange;
pub const matmulQ5_0RhsTile = cold.matmulQ5_0RhsTile;
pub const matmulQ5_1RhsRange = cold.matmulQ5_1RhsRange;
pub const matmulQ5_1RhsTile = cold.matmulQ5_1RhsTile;
pub const matmulTableQ8_0RhsRange = cold.matmulTableQ8_0RhsRange;
pub const matmulTableQ8_0RhsTile = cold.matmulTableQ8_0RhsTile;
pub const matmulTableQ8_KRhsRange = cold.matmulTableQ8_KRhsRange;
pub const matmulTableQ8_KRhsTile = cold.matmulTableQ8_KRhsTile;

pub const matmulTQ2_0RhsRange = ternary.matmulTQ2_0RhsRange;
pub const matmulTQ2_0RhsTile = ternary.matmulTQ2_0RhsTile;
pub const matmulTQ2_0F32RhsRange = ternary.matmulTQ2_0F32RhsRange;
pub const matmulTQ2_0F32RhsTile = ternary.matmulTQ2_0F32RhsTile;

pub const matmulQ8_0RhsRange = q8_0.matmulQ8_0RhsRange;
pub const matmulQ8_0RhsTile = q8_0.matmulQ8_0RhsTile;
pub const matmulQ8_0x4PackedPaddedRhsRange = q8_0.matmulQ8_0x4PackedPaddedRhsRange;
pub const matmulQ8_0x4PackedPaddedRhsTile = q8_0.matmulQ8_0x4PackedPaddedRhsTile;
pub const matmulQ8_0x4PackedRhsRange = q8_0.matmulQ8_0x4PackedRhsRange;
pub const matmulQ8_0x4PackedRhsTile = q8_0.matmulQ8_0x4PackedRhsTile;
pub const matmulQ8_0x4RhsRange = q8_0.matmulQ8_0x4RhsRange;
pub const matmulQ8_0x4RhsTile = q8_0.matmulQ8_0x4RhsTile;

pub const matmulQ4_KRhsRange = q4_k.matmulQ4_KRhsRange;
pub const matmulQ4_KRhsTile = q4_k.matmulQ4_KRhsTile;
pub const matmulQ4_KRhsCompactColOuter = q4_k.matmulQ4_KRhsCompactColOuter;
pub const matmulQ4_KCompactQ8_Kx4ColOuter = q4_k.matmulQ4_KCompactQ8_Kx4ColOuter;
pub const matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsRange = q4_k.matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsRange;
pub const matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsTile = q4_k.matmulQ4_Kx2MmlaQ8_Kx2MmlaRhsTile;
pub const matmulQ4_Kx2MmlaRhsRange = q4_k.matmulQ4_Kx2MmlaRhsRange;
pub const matmulQ4_Kx2MmlaRhsTile = q4_k.matmulQ4_Kx2MmlaRhsTile;
pub const matmulQ4_Kx4RhsRange = q4_k.matmulQ4_Kx4RhsRange;
pub const matmulQ4_Kx4RhsTile = q4_k.matmulQ4_Kx4RhsTile;
pub const matmulQ4_Kx8Q8_Kx4RhsRange = q4_k.matmulQ4_Kx8Q8_Kx4RhsRange;
pub const matmulQ4_Kx8Q8_Kx4RhsTile = q4_k.matmulQ4_Kx8Q8_Kx4RhsTile;
pub const matmulQ4_Kx8RhsRange = q4_k.matmulQ4_Kx8RhsRange;
pub const matmulQ4_Kx8RhsTile = q4_k.matmulQ4_Kx8RhsTile;

pub const matmulQ5_KRhsRange = q5_k.matmulQ5_KRhsRange;
pub const matmulQ5_KRhsTile = q5_k.matmulQ5_KRhsTile;
pub const matmulQ5_KRhsCompactColOuter = q5_k.matmulQ5_KRhsCompactColOuter;
pub const matmulQ5_KCompactQ8_Kx4ColOuter = q5_k.matmulQ5_KCompactQ8_Kx4ColOuter;
pub const matmulQ5_Kx8Q8_Kx4RhsRange = q5_k.matmulQ5_Kx8Q8_Kx4RhsRange;
pub const matmulQ5_Kx8Q8_Kx4RhsTile = q5_k.matmulQ5_Kx8Q8_Kx4RhsTile;
pub const matmulQ5_Kx8RhsRange = q5_k.matmulQ5_Kx8RhsRange;
pub const matmulQ5_Kx8RhsTile = q5_k.matmulQ5_Kx8RhsTile;

pub const matmulQ6_KRhsRange = q6_k.matmulQ6_KRhsRange;
pub const matmulQ6_KRhsTile = q6_k.matmulQ6_KRhsTile;
pub const matmulQ6_KRhsCompactColOuter = q6_k.matmulQ6_KRhsCompactColOuter;
pub const matmulQ6_KCompactQ8_Kx4ColOuter = q6_k.matmulQ6_KCompactQ8_Kx4ColOuter;
pub const matmulQ6_Kx4RhsRange = q6_k.matmulQ6_Kx4RhsRange;
pub const matmulQ6_Kx4RhsPairTile = q6_k.matmulQ6_Kx4RhsPairTile;
pub const matmulQ6_Kx4RhsTile = q6_k.matmulQ6_Kx4RhsTile;
