//! Quantized matmul: shared core, dispatch, and the re-export manifest.
//!
//! Module layout (kernels are split into per-type files and re-exported here, so
//! external callers keep using `backend.quantized_matmul.X` unchanged):
//!   quant.zig        - this file: shared Block/type decls, the generic
//!                      QuantizedRowsFor / QuantizedMatmulRhsRowsFor factories,
//!                      the QuantizedMatmulFormat enum + traits, the shared
//!                      K-quant accessors and the Q8_K activation quantizers,
//!                      the central dispatch switches, and the manifest
//!                      re-exporting the moved kernels.
//!   quant/common.zig - leaf module of generic SIMD/arch primitives: the
//!                      @Vector aliases, sdot/smmla asm wrappers, f16
//!                      conversions, nibble extractors, round/quantize helpers,
//!                      i8 dot, shared row/col-block consts, has_aarch64_i8mm.
//!   quant/q8_0.zig   - Q8_0 / Q8_0x4 hot kernels
//!   quant/q4_k.zig   - Q4_K hot kernels (Q4_Kx8 + comptime-gated Q4_Kx2Mmla/smmla)
//!   quant/q5_k.zig   - Q5_K / Q5_Kx8 hot kernels
//!   quant/q6_k.zig   - Q6_K / Q6_Kx4 hot kernels
//!   quant/cold.zig   - rarely-used generic formats (legacy Q4_0/1, Q5_0/1,
//!                      Q2_K/Q3_K, IQ*, TQ*, FP4, Table machinery): generic "dot"
//!                      path only, no packed fast path; exercised by tests, not by
//!                      the benchmarked Qwen3 weights (Q4_K/Q5_K/Q6_K/Q8_0/f16).
//!   quant/matmul_api.zig - internal narrow matmul surface used by vector
//!                      dispatchers so they do not import this barrel.
//!
//! Kernel naming grammar (matmul<Weight><Apack>...Rhs{Tile,Range}):
//!   <Weight>    weight/RHS quant - Q8_0, Q4_K, Q5_K, Q6_K (cold: Q4_0, Q2_K, ...)
//!   <Apack>     RHS column-interleave + SIMD target: x4/x8 -> sdot, x2Mmla -> smmla;
//!               no suffix = non-interleaved column-major
//!   Packed      the LHS activations are ALSO int8-packed (e.g. BlockQ8_0x4 LHS)
//!   Padded      tolerates a row count m not a multiple of 4 (masks output writes)
//!   ...RhsTile  the kernel over an explicit (r0,r1,c0,c1) block
//!   ...RhsRange thin full-width wrapper (c0=0, c1=n) - the parallel/serial entry
//!   ColsFirst   internal >=128-row perf specialization (column-outer, 8-row-group)
//!   ...Q8_Kx4.. / ...Q8_Kx2Mmla.. names the LHS activation packing consumed
//!   accumulate* inner per-block microkernel: *Aarch64 = NEON sdot/smmla,
//!               *Scalar = portable fallback, *Dual = two row-groups for sdot ILP
const std = @import("std");
const builtin = @import("builtin");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");
const cold = @import("quant/cold.zig");
const q8_0 = @import("quant/q8_0.zig");
const q4_k = @import("quant/q4_k.zig");
const q5_k = @import("quant/q5_k.zig");
const q6_k = @import("quant/q6_k.zig");
const ternary = @import("quant/ternary.zig");
const common = @import("quant/common.zig");
const matmul_api = @import("quant/matmul_api.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;
const has_aarch64_i8mm = common.has_aarch64_i8mm;
const quantizeToI8 = common.quantizeToI8;
const roundNearestEven = common.roundNearestEven;
const roundNearestEvenVec4ToI32 = common.roundNearestEvenVec4ToI32;
const f32ToF16Bits = common.f32ToF16Bits;
const f16BitsToF32 = common.f16BitsToF32;
const f16x4BitsToF32 = common.f16x4BitsToF32;
const QKV4i32 = common.QKV4i32;
const QKV4f32 = common.QKV4f32;
pub const supports_q4_k_mmla = has_aarch64_i8mm;

// The quant matmul TYPE + format-trait layer lives in quant/types.zig so the
// per-type kernel files can import it without a quant.zig<->children import
// cycle; re-exported here.
const types = @import("quant/types.zig");

const q8k = @import("quant/q8k.zig");
pub const blockCountExact = q8k.blockCountExact;
pub const dequantizeBlockQ8_KInto = q8k.dequantizeBlockQ8_KInto;
pub const dequantizeRowQ8_0Into = q8k.dequantizeRowQ8_0Into;
pub const dequantizeRowsQ8_0Into = q8k.dequantizeRowsQ8_0Into;
pub const fillQ8KPattern = q8k.fillQ8KPattern;
pub const getRowsQ8_0Into = q8k.getRowsQ8_0Into;
pub const getScaleMinK4 = q8k.getScaleMinK4;
pub const group_max_eps = q8k.group_max_eps;
pub const makeQkx2Quants = q8k.makeQkx2Quants;
pub const makeQxQuants = q8k.makeQxQuants;
pub const nearestInt = q8k.nearestInt;
pub const packRowsQ8_Kx4 = q8k.packRowsQ8_Kx4;
pub const packRowsQ8_Kx4PaddedInto = q8k.packRowsQ8_Kx4PaddedInto;
pub const q4Kx8D = q8k.q4Kx8D;
pub const q4Kx8Scales = q8k.q4Kx8Scales;
pub const q8_0BlockCount = q8k.q8_0BlockCount;
pub const qkBlockCount = q8k.qkBlockCount;
pub const quantizeRowGroupQ8_Kx4Into = q8k.quantizeRowGroupQ8_Kx4Into;
pub const quantizeRowQ8_0Into = q8k.quantizeRowQ8_0Into;
pub const quantizeRowQ8_KInto = q8k.quantizeRowQ8_KInto;
pub const quantizeRowsQ8_0 = q8k.quantizeRowsQ8_0;
pub const quantizeRowsQ8_0Into = q8k.quantizeRowsQ8_0Into;
pub const quantizeRowsQ8_K = q8k.quantizeRowsQ8_K;
pub const quantizeRowsQ8_Kx2MmlaInto = q8k.quantizeRowsQ8_Kx2MmlaInto;
pub const quantizeRowsQ8_Kx4Into = q8k.quantizeRowsQ8_Kx4Into;
pub const quantizeRowsQ8_Kx4IntoImpl = q8k.quantizeRowsQ8_Kx4IntoImpl;
pub const quantizeRowsQ8_Kx4PaddedInto = q8k.quantizeRowsQ8_Kx4PaddedInto;
pub const quantizedMatmulRhsQ2_KFromBlocks = q8k.quantizedMatmulRhsQ2_KFromBlocks;
pub const quantizedMatmulRhsQ3_KFromBlocks = q8k.quantizedMatmulRhsQ3_KFromBlocks;
pub const quantizedMatmulRhsQ4_KFromBlocks = q8k.quantizedMatmulRhsQ4_KFromBlocks;
pub const quantizedMatmulRhsQ5_KFromBlocks = q8k.quantizedMatmulRhsQ5_KFromBlocks;
pub const quantizedMatmulRhsQ6_KFromBlocks = q8k.quantizedMatmulRhsQ6_KFromBlocks;
pub const zeroQ8Kx4Lane = q8k.zeroQ8Kx4Lane;

pub const AnyQuantizedMatmulRhs = types.AnyQuantizedMatmulRhs;
pub const BlockIQ1_M = types.BlockIQ1_M;
pub const BlockIQ1_S = types.BlockIQ1_S;
pub const BlockIQ2_S = types.BlockIQ2_S;
pub const BlockIQ2_XS = types.BlockIQ2_XS;
pub const BlockIQ2_XXS = types.BlockIQ2_XXS;
pub const BlockIQ3_S = types.BlockIQ3_S;
pub const BlockIQ3_XXS = types.BlockIQ3_XXS;
pub const BlockIQ4_NL = types.BlockIQ4_NL;
pub const BlockIQ4_XS = types.BlockIQ4_XS;
pub const BlockMXFP4 = types.BlockMXFP4;
pub const BlockNVFP4 = types.BlockNVFP4;
pub const BlockQ1_0 = types.BlockQ1_0;
pub const BlockQ2_0 = types.BlockQ2_0;
pub const BlockQ2_K = types.BlockQ2_K;
pub const BlockQ3_K = types.BlockQ3_K;
pub const BlockQ4_0 = types.BlockQ4_0;
pub const BlockQ4_1 = types.BlockQ4_1;
pub const BlockQ4_K = types.BlockQ4_K;
pub const BlockQ4_Kx2Mmla = types.BlockQ4_Kx2Mmla;
pub const BlockQ4_Kx4 = types.BlockQ4_Kx4;
pub const BlockQ4_Kx8 = types.BlockQ4_Kx8;
pub const BlockQ5_0 = types.BlockQ5_0;
pub const BlockQ5_1 = types.BlockQ5_1;
pub const BlockQ5_K = types.BlockQ5_K;
pub const BlockQ5_Kx8 = types.BlockQ5_Kx8;
pub const BlockQ6_K = types.BlockQ6_K;
pub const BlockQ6_Kx4 = types.BlockQ6_Kx4;
pub const BlockQ8_0 = types.BlockQ8_0;
pub const BlockQ8_0x4 = types.BlockQ8_0x4;
pub const BlockQ8_1 = types.BlockQ8_1;
pub const BlockQ8_K = types.BlockQ8_K;
pub const BlockQ8_Kx2Mmla = types.BlockQ8_Kx2Mmla;
pub const BlockQ8_Kx4 = types.BlockQ8_Kx4;
pub const BlockTQ1_0 = types.BlockTQ1_0;
pub const BlockTQ2_0 = types.BlockTQ2_0;
pub const BlockTQ2_0x4 = types.BlockTQ2_0x4;
pub const BlockTQ2_0Foldedx4 = types.BlockTQ2_0Foldedx4;
pub const BlockTQ2_0Folded = types.BlockTQ2_0Folded;
pub const PackedRhsFor = types.PackedRhsFor;
pub const PackedRhsLayout = types.PackedRhsLayout;
pub const QuantizedFormatError = types.QuantizedFormatError;
pub const QuantizedMatmulFormat = types.QuantizedMatmulFormat;
pub const QuantizedMatmulKernel = types.QuantizedMatmulKernel;
pub const QuantizedMatmulRhs = types.QuantizedMatmulRhs;
pub const QuantizedMatmulRhsI8 = types.QuantizedMatmulRhsI8;
pub const QuantizedMatmulRhsIQ1_M = types.QuantizedMatmulRhsIQ1_M;
pub const QuantizedMatmulRhsIQ1_S = types.QuantizedMatmulRhsIQ1_S;
pub const QuantizedMatmulRhsIQ2_S = types.QuantizedMatmulRhsIQ2_S;
pub const QuantizedMatmulRhsIQ2_XS = types.QuantizedMatmulRhsIQ2_XS;
pub const QuantizedMatmulRhsIQ2_XXS = types.QuantizedMatmulRhsIQ2_XXS;
pub const QuantizedMatmulRhsIQ3_S = types.QuantizedMatmulRhsIQ3_S;
pub const QuantizedMatmulRhsIQ3_XXS = types.QuantizedMatmulRhsIQ3_XXS;
pub const QuantizedMatmulRhsIQ4_NL = types.QuantizedMatmulRhsIQ4_NL;
pub const QuantizedMatmulRhsIQ4_XS = types.QuantizedMatmulRhsIQ4_XS;
pub const QuantizedMatmulRhsMXFP4 = types.QuantizedMatmulRhsMXFP4;
pub const QuantizedMatmulRhsNVFP4 = types.QuantizedMatmulRhsNVFP4;
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
pub const QuantizedMatmulRhsTQ1_0 = types.QuantizedMatmulRhsTQ1_0;
pub const QuantizedMatmulRhsTQ2_0 = types.QuantizedMatmulRhsTQ2_0;
pub const QuantizedMatmulTraits = types.QuantizedMatmulTraits;
pub const QuantizedRowsFor = types.QuantizedRowsFor;
pub const QuantizedRowsQ4_0 = types.QuantizedRowsQ4_0;
pub const QuantizedRowsQ8_0 = types.QuantizedRowsQ8_0;
pub const QuantizedRowsQ8_1 = types.QuantizedRowsQ8_1;
pub const QuantizedScaleLayout = types.QuantizedScaleLayout;
pub const QuantizedStorageLayout = types.QuantizedStorageLayout;
pub const default_i8_group_size = types.default_i8_group_size;
pub const formatForDType = types.formatForDType;
pub const iq4_nl_block_size = types.iq4_nl_block_size;
pub const k_scale_size = types.k_scale_size;
pub const matmulTraits = types.matmulTraits;
pub const matmulTraitsRuntime = types.matmulTraitsRuntime;
pub const mxfp4_block_size = types.mxfp4_block_size;
pub const nvfp4_block_size = types.nvfp4_block_size;
pub const nvfp4_subblock_size = types.nvfp4_subblock_size;
pub const q1_0_block_size = types.q1_0_block_size;
pub const q2_0_block_size = types.q2_0_block_size;
pub const q4_0_block_size = types.q4_0_block_size;
pub const q4_1_block_size = types.q4_1_block_size;
pub const q5_0_block_size = types.q5_0_block_size;
pub const q5_1_block_size = types.q5_1_block_size;
pub const q8_0_block_size = types.q8_0_block_size;
pub const q8_1_block_size = types.q8_1_block_size;
pub const qk_k_block_size = types.qk_k_block_size;
pub const supportsMatmul = types.supportsMatmul;

// Quantize an f32 RHS [k, n] to the i8 block-wise format. Symmetric per-(column,
// group) scales: scale = amax(group) / 127, q = round(w / scale) clamped to
// [-127, 127]. Weights are stored transposed as [n][k] so each column's k-vector
// is contiguous for the int8 dot.
pub fn quantizeRhsBlockwiseI8(
    allocator: Allocator,
    rhs: *const Tensor,
    group_size: usize,
) !QuantizedMatmulRhsI8 {
    const traits = QuantizedMatmulRhsI8.traits;
    const view = try rhs.rankView(2);
    const k = view.dim(0);
    const n = view.dim(1);
    const gs = traits.effectiveGroupSize(group_size);
    const num_groups = traits.groupCountForSize(k, gs);

    const src = try rhs.dataConstChecked();

    const storage_shape = traits.storageShape(k, n);
    var qw = try tensor.TensorOf(.i8).zeros(allocator, &storage_shape);
    errdefer qw.deinit();
    const scale_shape = traits.scaleShape(k, n, gs);
    var scales = try Tensor.zeros(allocator, &scale_shape);
    errdefer scales.deinit();

    const qwd = qw.data();
    const sd = scales.data();

    var j: usize = 0;
    while (j < n) : (j += 1) {
        var g: usize = 0;
        while (g < num_groups) : (g += 1) {
            const p0 = g * gs;
            const p1 = @min(p0 + gs, k);

            var amax: f32 = 0;
            var p = p0;
            while (p < p1) : (p += 1) amax = @max(amax, @abs(src[p * n + j]));

            const scale: f32 = if (amax == 0) 0 else amax / 127.0;
            sd[traits.scaleIndex(j, g, num_groups)] = scale;

            const inv: f32 = if (scale == 0) 0 else 1.0 / scale;
            p = p0;
            while (p < p1) : (p += 1) qwd[traits.storageIndex(j, p, k)] = quantizeToI8(src[p * n + j] * inv);
        }
    }

    return .{ .qw = qw, .scales = scales, .k = k, .n = n, .group_size = gs, .num_groups = num_groups };
}

// Dynamic per-row symmetric int8 quantization of the f32 activations [m, k].
pub fn quantizeActivationsPerRowI8(qa: []i8, a_scales: []f32, a: []const f32, m: usize, k: usize) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        const row = a[i * k ..][0..k];
        var amax: f32 = 0;
        for (row) |v| amax = @max(amax, @abs(v));

        const scale: f32 = if (amax == 0) 0 else amax / 127.0;
        a_scales[i] = scale;

        const inv: f32 = if (scale == 0) 0 else 1.0 / scale;
        const qrow = qa[i * k ..][0..k];
        for (qrow, row) |*q, v| q.* = quantizeToI8(v * inv);
    }
}

pub fn quantizeMatmulRhsQ8_0(allocator: Allocator, rhs: *const Tensor) !QuantizedMatmulRhsQ8_0 {
    const view = try rhs.rankView(2);
    const k = view.dim(0);
    const n = view.dim(1);
    const blocks_per_column = try q8_0BlockCount(k);
    const data = try rhs.dataConstChecked();

    const blocks = try allocator.alloc(BlockQ8_0, try types.checkedProduct(n, blocks_per_column));
    errdefer allocator.free(blocks);
    const scratch = try allocator.alloc(f32, k);
    defer allocator.free(scratch);

    var col: usize = 0;
    while (col < n) : (col += 1) {
        var p: usize = 0;
        while (p < k) : (p += 1) scratch[p] = data[p * n + col];
        try quantizeRowQ8_0Into(
            blocks[col * blocks_per_column ..][0..blocks_per_column],
            scratch,
        );
    }

    return .{
        .rows = .{
            .allocator = allocator,
            .blocks = blocks,
            .rows = n,
            .cols = k,
            .blocks_per_row = blocks_per_column,
        },
        .k = k,
        .n = n,
    };
}

pub fn dequantizeTensorInto(comptime tensor_dtype: DType, dst: *Tensor, src: *const tensor.TensorOf(tensor_dtype)) !void {
    comptime if (!dtype_mod.isBlockQuantized(tensor_dtype)) @compileError("dequantizeTensorInto requires a block-quantized dtype");

    const src_view = try src.rankView(2);
    const rows = src_view.dim(0);
    const cols = src_view.dim(1);
    const dst_view = try dst.rankView(2);
    if (dst_view.dim(0) != rows or dst_view.dim(1) != cols) return tensor.TensorError.ShapeMismatch;

    const out = try dst.dataChecked();
    const blocks = try src.dataConstChecked();
    const blocks_per_row = try blockCountForDType(tensor_dtype, cols);

    var row: usize = 0;
    while (row < rows) : (row += 1) {
        try dequantizeRowForDType(
            tensor_dtype,
            out[row * cols ..][0..cols],
            blocks[row * blocks_per_row ..][0..blocks_per_row],
        );
    }
}

pub fn getRowsTensorInto(comptime tensor_dtype: DType, dst: *Tensor, table: *const tensor.TensorOf(tensor_dtype), indices: []const usize) !void {
    comptime if (!dtype_mod.isBlockQuantized(tensor_dtype)) @compileError("getRowsTensorInto requires a block-quantized dtype");
    if (indices.len == 0) return tensor.TensorError.InvalidShape;

    const table_view = try table.rankView(2);
    const rows = table_view.dim(0);
    const cols = table_view.dim(1);
    const dst_view = try dst.rankView(2);
    if (dst_view.dim(0) != indices.len or dst_view.dim(1) != cols) return tensor.TensorError.ShapeMismatch;

    const out = try dst.dataChecked();
    const blocks = try table.dataConstChecked();
    const blocks_per_row = try blockCountForDType(tensor_dtype, cols);

    for (indices, 0..) |index, row| {
        if (index >= rows) return tensor.TensorError.IndexOutOfBounds;
        try dequantizeRowForDType(
            tensor_dtype,
            out[row * cols ..][0..cols],
            blocks[index * blocks_per_row ..][0..blocks_per_row],
        );
    }
}

pub fn blockCountForDType(comptime tensor_dtype: DType, len: usize) !usize {
    return switch (tensor_dtype) {
        .q1_0 => cold.q1_0BlockCount(len),
        .q2_0 => cold.q2_0BlockCount(len),
        .q4_0 => cold.q4_0BlockCount(len),
        .q4_1 => cold.q4_1BlockCount(len),
        .q5_0 => cold.q5_0BlockCount(len),
        .q5_1 => cold.q5_1BlockCount(len),
        .q8_0 => q8_0BlockCount(len),
        .q8_1 => cold.q8_1BlockCount(len),
        .q2_k,
        .q3_k,
        .q4_k,
        .q5_k,
        .q6_k,
        .q8_k,
        .iq1_s,
        .iq1_m,
        .iq2_xxs,
        .iq2_xs,
        .iq2_s,
        .iq3_xxs,
        .iq3_s,
        .iq4_xs,
        .tq1_0,
        .tq2_0,
        => qkBlockCount(len),
        .iq4_nl => blockCountExact(iq4_nl_block_size, len),
        .mxfp4 => blockCountExact(mxfp4_block_size, len),
        .nvfp4 => blockCountExact(nvfp4_block_size, len),
        else => @compileError("dtype is not block-quantized"),
    };
}

pub fn dequantizeRowForDType(
    comptime tensor_dtype: DType,
    dst: []f32,
    blocks: []const dtype_mod.Storage(tensor_dtype),
) !void {
    switch (tensor_dtype) {
        .q1_0 => try cold.dequantizeRowQ1_0Into(dst, blocks),
        .q2_0 => try cold.dequantizeRowQ2_0Into(dst, blocks),
        .q4_0 => try cold.dequantizeRowQ4_0Into(dst, blocks),
        .q4_1 => try cold.dequantizeRowQ4_1Into(dst, blocks),
        .q5_0 => try cold.dequantizeRowQ5_0Into(dst, blocks),
        .q5_1 => try cold.dequantizeRowQ5_1Into(dst, blocks),
        .q8_0 => try dequantizeRowQ8_0Into(dst, blocks),
        .q8_1 => try cold.dequantizeRowQ8_1Into(dst, blocks),
        .q2_k => try dequantizeKRowInto(tensor_dtype, dst, blocks),
        .q3_k => try dequantizeKRowInto(tensor_dtype, dst, blocks),
        .q4_k => try dequantizeKRowInto(tensor_dtype, dst, blocks),
        .q5_k => try dequantizeKRowInto(tensor_dtype, dst, blocks),
        .q6_k => try dequantizeKRowInto(tensor_dtype, dst, blocks),
        .q8_k => try dequantizeKRowInto(tensor_dtype, dst, blocks),
        .iq1_s => try cold.dequantizeRowIQ1_SInto(dst, blocks),
        .iq1_m => try cold.dequantizeRowIQ1_MInto(dst, blocks),
        .iq2_xxs => try cold.dequantizeRowIQ2_XXSInto(dst, blocks),
        .iq2_xs => try cold.dequantizeRowIQ2_XSInto(dst, blocks),
        .iq2_s => try cold.dequantizeRowIQ2_SInto(dst, blocks),
        .iq3_xxs => try cold.dequantizeRowIQ3_XXSInto(dst, blocks),
        .iq3_s => try cold.dequantizeRowIQ3_SInto(dst, blocks),
        .iq4_nl => try cold.dequantizeRowIQ4_NLInto(dst, blocks),
        .iq4_xs => try cold.dequantizeRowIQ4_XSInto(dst, blocks),
        .tq1_0 => try cold.dequantizeRowTQ1_0Into(dst, blocks),
        .tq2_0 => try cold.dequantizeRowTQ2_0Into(dst, blocks),
        .mxfp4 => try cold.dequantizeRowMXFP4Into(dst, blocks),
        .nvfp4 => try cold.dequantizeRowNVFP4Into(dst, blocks),
        else => @compileError("dtype is not block-quantized"),
    }
}

/// f32 -> quantized row encoder dispatch (the GGUF quantize-export entry).
/// The caller supplies the output blocks; `src.len` must be a whole number of
/// blocks and `dst.len` must match (`blockCountForDType`). Dtypes whose traits
/// say `supports_from_float == false` (and non-quantized dtypes) are a compile
/// error. Inputs are assumed finite (no NaN/inf), as in ggml's encoders.
pub fn quantizeRowForDType(
    comptime tensor_dtype: DType,
    dst: []dtype_mod.Storage(tensor_dtype),
    src: []const f32,
) !void {
    switch (tensor_dtype) {
        .q2_0 => try cold.quantizeRowQ2_0Into(dst, src),
        .q4_0 => try cold.quantizeRowQ4_0Into(dst, src),
        .q4_1 => try cold.quantizeRowQ4_1Into(dst, src),
        .q5_0 => try cold.quantizeRowQ5_0Into(dst, src),
        .q5_1 => try cold.quantizeRowQ5_1Into(dst, src),
        .q8_0 => try quantizeRowQ8_0Into(dst, src),
        .q8_1 => try cold.quantizeRowQ8_1Into(dst, src),
        .q4_k => try q4_k.quantizeRowQ4_KInto(dst, src),
        .q5_k => try q5_k.quantizeRowQ5_KInto(dst, src),
        .q6_k => try q6_k.quantizeRowQ6_KInto(dst, src),
        .q8_k => try quantizeRowQ8_KInto(dst, src),
        .tq2_0 => try ternary.quantizeRowTQ2_0Into(dst, src),
        else => @compileError("dtype has no f32 -> quantized row encoder"),
    }
}

fn dequantizeKRowInto(
    comptime tensor_dtype: DType,
    dst: []f32,
    blocks: []const dtype_mod.Storage(tensor_dtype),
) !void {
    if (dst.len != try types.checkedProduct(blocks.len, qk_k_block_size)) return QuantizedFormatError.InvalidQuantizedLength;

    var dense_block: [qk_k_block_size]f32 = undefined;
    for (blocks, 0..) |*block, block_index| {
        switch (tensor_dtype) {
            .q2_k => cold.dequantizeBlockQ2_KInto(&dense_block, block),
            .q3_k => cold.dequantizeBlockQ3_KInto(&dense_block, block),
            .q4_k => q4_k.dequantizeBlockQ4_KInto(&dense_block, block),
            .q5_k => q5_k.dequantizeBlockQ5_KInto(&dense_block, block),
            .q6_k => q6_k.dequantizeBlockQ6_KInto(&dense_block, block),
            .q8_k => dequantizeBlockQ8_KInto(&dense_block, block),
            else => unreachable,
        }
        @memcpy(dst[block_index * qk_k_block_size ..][0..qk_k_block_size], &dense_block);
    }
}

// W8A8 matmul lives in quant/matmul_api.zig so vector dispatchers do not import
// this compatibility barrel. Re-export it here for existing `quant.<sym>` callers.
pub const matmulI8BlockwiseTile = matmul_api.matmulI8BlockwiseTile;
pub const matmulI8BlockwiseRange = matmul_api.matmulI8BlockwiseRange;

// ---------------------------------------------------------------------------
// f32 -> K-quant reference encoders: shared iterative scale-search helpers,
// ported operation-for-operation (f32 arithmetic, same rounding) from ggml's
// ggml-quants.c so the encoded bytes match quantize_row_{q4,q5,q6}_K_ref
// bit-for-bit (verified against embedded goldens in quant/encode_golden_test.zig).
//
// Finite-input contract (same as ggml): inputs must be free of NaN/inf. A
// sub-block whose value spread underflows f32 (so nmax/(max-min) overflows to
// inf) feeds a non-finite value into nearestInt; ggml's own assert in
// nearest_int rejects that too, and we mirror it with a debug assert.
// ---------------------------------------------------------------------------

/// ggml GROUP_MAX_EPS: sub-blocks with amax below this encode as all-zero.
/// ggml nearest_int: round-to-nearest-even via the 1.5*2^23 magic constant.
/// Valid for |fval| <= 4194303 (ggml asserts the same bound).
/// ggml make_qx_quants: symmetric scale search over +/-9 tenth-steps around
/// -nmax/max (the sign rides on the scale). `L` receives nmax-biased levels in
/// [0, 2*nmax-1]. The Q6_K encoder uses rmse_type = 1 with qw = null; the full
/// reference behavior (rmse_type 0/negative/2/3/default, explicit qw) is kept.
/// ggml make_qkx2_quants: asymmetric (scale, min) grid search used by the
/// Q4_K/Q5_K encoders. `iscale` candidates sweep (rmin + rdelta*is + nmax) /
/// (max - min) for is in [0, nstep]; a weighted least-squares (scale, min) fit
/// is accepted when it lowers the weighted error (MAD when use_mad). `L`
/// receives levels in [0, nmax]; `Laux` is caller-supplied scratch.

// ---------------------------------------------------------------------------
// Cold-format manifest: public functions moved to quant/cold.zig are re-exported
// here so external callers (quantized_matmul.<fn>) keep working unchanged.
// ---------------------------------------------------------------------------
pub const dequantizeBlockQ2_KInto = cold.dequantizeBlockQ2_KInto;
pub const dequantizeBlockQ3_KInto = cold.dequantizeBlockQ3_KInto;
pub const dequantizeRowIQ1_MInto = cold.dequantizeRowIQ1_MInto;
pub const dequantizeRowIQ1_SInto = cold.dequantizeRowIQ1_SInto;
pub const dequantizeRowIQ2_SInto = cold.dequantizeRowIQ2_SInto;
pub const dequantizeRowIQ2_XSInto = cold.dequantizeRowIQ2_XSInto;
pub const dequantizeRowIQ2_XXSInto = cold.dequantizeRowIQ2_XXSInto;
pub const dequantizeRowIQ3_SInto = cold.dequantizeRowIQ3_SInto;
pub const dequantizeRowIQ3_XXSInto = cold.dequantizeRowIQ3_XXSInto;
pub const dequantizeRowIQ4_NLInto = cold.dequantizeRowIQ4_NLInto;
pub const dequantizeRowIQ4_XSInto = cold.dequantizeRowIQ4_XSInto;
pub const dequantizeRowMXFP4Into = cold.dequantizeRowMXFP4Into;
pub const dequantizeRowNVFP4Into = cold.dequantizeRowNVFP4Into;
pub const dequantizeRowQ1_0Into = cold.dequantizeRowQ1_0Into;
pub const dequantizeRowQ4_0Into = cold.dequantizeRowQ4_0Into;
pub const dequantizeRowQ4_1Into = cold.dequantizeRowQ4_1Into;
pub const dequantizeRowQ5_0Into = cold.dequantizeRowQ5_0Into;
pub const dequantizeRowQ5_1Into = cold.dequantizeRowQ5_1Into;
pub const dequantizeRowQ8_1Into = cold.dequantizeRowQ8_1Into;
pub const dequantizeRowTQ1_0Into = cold.dequantizeRowTQ1_0Into;
pub const dequantizeRowTQ2_0Into = cold.dequantizeRowTQ2_0Into;
pub const dequantizeRowsQ4_0Into = cold.dequantizeRowsQ4_0Into;
pub const getRowsQ4_0Into = cold.getRowsQ4_0Into;
pub const matmulQ1_0RhsRange = cold.matmulQ1_0RhsRange;
pub const matmulQ1_0RhsTile = cold.matmulQ1_0RhsTile;
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
pub const q1_0BlockCount = cold.q1_0BlockCount;
pub const q2_0BlockCount = cold.q2_0BlockCount;
pub const dequantizeRowQ2_0Into = cold.dequantizeRowQ2_0Into;
pub const quantizeRowQ2_0Into = cold.quantizeRowQ2_0Into;
pub const q4_0BlockCount = cold.q4_0BlockCount;
pub const q4_1BlockCount = cold.q4_1BlockCount;
pub const q5_0BlockCount = cold.q5_0BlockCount;
pub const q5_1BlockCount = cold.q5_1BlockCount;
pub const q8_1BlockCount = cold.q8_1BlockCount;
pub const quantizeMatmulRhsQ4_0 = cold.quantizeMatmulRhsQ4_0;
pub const quantizeRowQ4_0Into = cold.quantizeRowQ4_0Into;
pub const quantizeRowQ4_1Into = cold.quantizeRowQ4_1Into;
pub const quantizeRowQ5_0Into = cold.quantizeRowQ5_0Into;
pub const quantizeRowQ5_1Into = cold.quantizeRowQ5_1Into;
pub const quantizeRowQ8_1Into = cold.quantizeRowQ8_1Into;
pub const quantizeRowsQ4_0 = cold.quantizeRowsQ4_0;
pub const quantizeRowsQ8_1 = cold.quantizeRowsQ8_1;

// ---------------------------------------------------------------------------
// Hot per-type manifest: public functions moved to quant/<type>.zig are
// re-exported here so external callers (quantized_matmul.<fn>) keep working
// unchanged.
// ---------------------------------------------------------------------------
pub const matmulQ8_0RhsRange = q8_0.matmulQ8_0RhsRange;
pub const matmulQ8_0RhsTile = q8_0.matmulQ8_0RhsTile;
pub const matmulQ8_0x4PackedPaddedRhsRange = q8_0.matmulQ8_0x4PackedPaddedRhsRange;
pub const matmulQ8_0x4PackedPaddedRhsTile = q8_0.matmulQ8_0x4PackedPaddedRhsTile;
pub const matmulQ8_0x4PackedRhsRange = q8_0.matmulQ8_0x4PackedRhsRange;
pub const matmulQ8_0x4PackedRhsTile = q8_0.matmulQ8_0x4PackedRhsTile;
pub const matmulQ8_0x4RhsRange = q8_0.matmulQ8_0x4RhsRange;
pub const matmulQ8_0x4RhsTile = q8_0.matmulQ8_0x4RhsTile;
pub const packMatmulRhsQ8_0x4 = q8_0.packMatmulRhsQ8_0x4;
pub const quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto = q8_0.quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto;
pub const quantizeSplitSwiGluRowsQ8_0x4PaddedInto = q8_0.quantizeSplitSwiGluRowsQ8_0x4PaddedInto;
pub const quantizeRowsQ8_0x4GroupsInto = q8_0.quantizeRowsQ8_0x4GroupsInto;
pub const quantizeRowsQ8_0x4Into = q8_0.quantizeRowsQ8_0x4Into;
pub const quantizeRowsQ8_0x4PaddedInto = q8_0.quantizeRowsQ8_0x4PaddedInto;

pub const dequantizeBlockQ4_KInto = q4_k.dequantizeBlockQ4_KInto;
pub const quantizeBlockQ4_KInto = q4_k.quantizeBlockQ4_KInto;
pub const quantizeRowQ4_KInto = q4_k.quantizeRowQ4_KInto;
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
pub const packMatmulRhsQ4_Kx2Mmla = q4_k.packMatmulRhsQ4_Kx2Mmla;
pub const packMatmulRhsQ4_Kx4 = q4_k.packMatmulRhsQ4_Kx4;
pub const packMatmulRhsQ4_Kx8 = q4_k.packMatmulRhsQ4_Kx8;

pub const dequantizeBlockQ5_KInto = q5_k.dequantizeBlockQ5_KInto;
pub const quantizeBlockQ5_KInto = q5_k.quantizeBlockQ5_KInto;
pub const quantizeRowQ5_KInto = q5_k.quantizeRowQ5_KInto;
pub const matmulQ5_KRhsRange = q5_k.matmulQ5_KRhsRange;
pub const matmulQ5_KRhsTile = q5_k.matmulQ5_KRhsTile;
pub const matmulQ5_KRhsCompactColOuter = q5_k.matmulQ5_KRhsCompactColOuter;
pub const matmulQ5_KCompactQ8_Kx4ColOuter = q5_k.matmulQ5_KCompactQ8_Kx4ColOuter;
pub const matmulQ5_Kx8Q8_Kx4RhsRange = q5_k.matmulQ5_Kx8Q8_Kx4RhsRange;
pub const matmulQ5_Kx8Q8_Kx4RhsTile = q5_k.matmulQ5_Kx8Q8_Kx4RhsTile;
pub const matmulQ5_Kx8RhsRange = q5_k.matmulQ5_Kx8RhsRange;
pub const matmulQ5_Kx8RhsTile = q5_k.matmulQ5_Kx8RhsTile;
pub const packMatmulRhsQ5_Kx8 = q5_k.packMatmulRhsQ5_Kx8;

pub const dequantizeBlockQ6_KInto = q6_k.dequantizeBlockQ6_KInto;
pub const quantizeBlockQ6_KInto = q6_k.quantizeBlockQ6_KInto;
pub const quantizeRowQ6_KInto = q6_k.quantizeRowQ6_KInto;
pub const matmulQ6_KRhsRange = q6_k.matmulQ6_KRhsRange;
pub const matmulQ6_KRhsTile = q6_k.matmulQ6_KRhsTile;
pub const matmulQ6_KRhsCompactColOuter = q6_k.matmulQ6_KRhsCompactColOuter;
pub const matmulQ6_KCompactQ8_Kx4ColOuter = q6_k.matmulQ6_KCompactQ8_Kx4ColOuter;
pub const matmulQ6_Kx4RhsRange = q6_k.matmulQ6_Kx4RhsRange;
pub const matmulQ6_Kx4RhsPairTile = q6_k.matmulQ6_Kx4RhsPairTile;
pub const matmulQ6_Kx4RhsTile = q6_k.matmulQ6_Kx4RhsTile;
pub const packMatmulRhsQ6_Kx4 = q6_k.packMatmulRhsQ6_Kx4;

pub const quantizeRowTQ2_0Into = ternary.quantizeRowTQ2_0Into;
pub const quantizeRowTQ2_0ScaledInto = ternary.quantizeRowTQ2_0ScaledInto;
pub const ternaryAbsmeanScale = ternary.ternaryAbsmeanScale;
pub const quantizedMatmulRhsTQ2_0FromBlocks = ternary.quantizedMatmulRhsTQ2_0FromBlocks;
pub const quantizedMatmulRhsTQ2_0FromBorrowedBlocks = ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks;
pub const quantizedMatmulRhsTQ2_0FromF32 = ternary.quantizedMatmulRhsTQ2_0FromF32;
pub const quantizedMatmulRhsTQ2_0FromF32Absmean = ternary.quantizedMatmulRhsTQ2_0FromF32Absmean;
pub const matmulTQ2_0RhsRange = ternary.matmulTQ2_0RhsRange;
pub const matmulTQ2_0RhsTile = ternary.matmulTQ2_0RhsTile;
pub const packMatmulRhsTQ2_0x4 = ternary.packMatmulRhsTQ2_0x4;
pub const matmulTQ2_0X4RhsRange = ternary.matmulTQ2_0X4RhsRange;
pub const matmulTQ2_0X4RhsTile = ternary.matmulTQ2_0X4RhsTile;
pub const matmulTQ2_0X4RhsTileAcc = ternary.matmulTQ2_0X4RhsTileAcc;
pub const packMatmulRhsTQ2_0Foldedx4 = ternary.packMatmulRhsTQ2_0Foldedx4;
pub const packMatmulRhsTQ2_0Foldedx4Into = ternary.packMatmulRhsTQ2_0Foldedx4Into;
pub const packMatmulRhsTQ2_0FoldedRows = ternary.packMatmulRhsTQ2_0FoldedRows;
pub const matmulTQ2_0FoldedX4RhsTile = ternary.matmulTQ2_0FoldedX4RhsTile;
pub const matmulTQ2_0FoldedX4RhsRange = ternary.matmulTQ2_0FoldedX4RhsRange;
pub const matmulQ2_0RhsRange = ternary.matmulQ2_0RhsRange;
pub const matmulQ2_0RhsTile = ternary.matmulQ2_0RhsTile;
pub const matmulQ2_0RhsRefRange = cold.matmulQ2_0RhsRefRange;
pub const matmulQ2_0RhsRefTile = cold.matmulQ2_0RhsRefTile;
pub const dotQ2_0RowQ8_0 = cold.dotQ2_0RowQ8_0;
pub const dequantizeRowQ2_0FastInto = ternary.dequantizeRowQ2_0FastInto;
pub const matmulTQ2_0F32RhsRange = ternary.matmulTQ2_0F32RhsRange;
pub const matmulTQ2_0F32RhsTile = ternary.matmulTQ2_0F32RhsTile;
pub const dotTQ2_0F32 = ternary.dotTQ2_0F32;

test {
    _ = @import("quant_tests.zig");
    _ = @import("quant/common.zig");
    _ = @import("quant/cold.zig");
    _ = @import("quant/q8_0.zig");
    _ = @import("quant/q4_k.zig");
    _ = @import("quant/q5_k.zig");
    _ = @import("quant/q6_k.zig");
    _ = @import("quant/ternary.zig");
    _ = @import("quant/encode_golden_test.zig");
}
