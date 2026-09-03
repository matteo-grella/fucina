//! Quantized matmul: the block and RHS type vocabulary, the W8A8 (i8
//! blockwise) quantizers and kernel, the one `gemm` entry over an
//! `ops.QuantGemm` request (each child's tile bodies behind one comptime
//! selection), and the child modules that own each GGML format's kernels.
//!
//! Child modules, each addressed as `quant.<child>.<fn>`:
//!   types    - interleaved block layouts, RHS container types and block sizes.
//!   common   - SIMD and ISA primitives: vector aliases, sdot/smmla/vpdpbusd
//!              wrappers, f16 conversions, nibble extractors, rounding and
//!              quantize helpers, the i8 dot and the shared row/column block
//!              widths.
//!   q8k      - Q8_K/Q8_0 row encoders and decoders, the Q8_Kx4/x2Mmla LHS
//!              packers, the K-quant `*FromBlocks` RHS constructors and the
//!              ggml reference scale-search helpers.
//!   q8_0     - Q8_0 and Q8_0x4 kernels and the Q8_0x4 RHS packer.
//!   q4_k     - Q4_K kernels: row-outer tile, x4/x8 column packs and the
//!              comptime-gated Q4_Kx2Mmla smmla path, plus the Q4_K encoder.
//!   q5_k     - Q5_K and Q5_Kx8 kernels plus the Q5_K encoder.
//!   q6_k     - Q6_K and Q6_Kx4 kernels plus the Q6_K encoder.
//!   ternary  - TQ2_0 int8 and f32 kernels, the x4 and folded packs, the
//!              ternary encoders and the Q2_0 fast path.
//!   mxfp4    - the MXFP4 fp4-e2m1 kernel over Q8_0 activations.
//!   cold     - the rarely-served formats (Q4_0/1, Q5_0/1, Q8_1, Q2_K/Q3_K,
//!              IQ*, TQ1_0, NVFP4, the table-decoded machinery): generic dot
//!              path only, exercised by tests rather than the served models.
//!
//! Kernel naming grammar (matmul<Weight><Apack>...Rhs{Tile,Range}):
//!   <Weight>    weight/RHS quant - Q8_0, Q4_K, Q5_K, Q6_K (cold: Q4_0, Q2_K, ...)
//!   <Apack>     RHS column-interleave + SIMD target: x4/x8 -> sdot, x2Mmla -> smmla;
//!               no suffix = non-interleaved column-major
//!   Packed      the LHS activations are ALSO int8-packed (e.g. types.BlockQ8_0x4 LHS)
//!   Padded      tolerates a row count m not a multiple of 4 (masks output writes)
//!   ...RhsTile  the kernel over an explicit (r0,r1,c0,c1) block
//!   ...RhsRange thin full-width wrapper (c0=0, c1=n) - the parallel/serial entry
//!   ColsFirst   internal >=128-row perf specialization (column-outer, 8-row-group)
//!   ...Q8_Kx4.. / ...Q8_Kx2Mmla.. names the LHS activation packing consumed
//!   accumulate* inner per-block microkernel: *Aarch64 = NEON sdot/smmla,
//!               *Scalar = portable fallback, *Dual = two row-groups for sdot ILP
const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");

pub const types = @import("quant/types.zig");
pub const common = @import("quant/common.zig");
const isa = @import("isa.zig");
const ops = @import("ops.zig");
pub const q8k = @import("quant/q8k.zig");
pub const q8_0 = @import("quant/q8_0.zig");
pub const q4_k = @import("quant/q4_k.zig");
pub const q5_k = @import("quant/q5_k.zig");
pub const q6_k = @import("quant/q6_k.zig");
pub const ternary = @import("quant/ternary.zig");
pub const mxfp4 = @import("quant/mxfp4.zig");
pub const cold = @import("quant/cold.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

pub const supports_q4_k_mmla = isa.has_aarch64_i8mm;

/// The dtype -> packed matmul RHS container map for the block-quantized
/// serving formats. `backend.PackedRhsFor` layers the dense f32 panel on
/// top of this; the Q4_K arm picks the smmla pack on aarch64+i8mm targets.
pub fn PackedQuantRhsFor(comptime dt: DType) type {
    return switch (dt) {
        .q8_0 => types.QuantizedMatmulRhsQ8_0x4,
        .q6_k => types.QuantizedMatmulRhsQ6_Kx4,
        .q5_k => types.QuantizedMatmulRhsQ5_Kx8,
        .q4_k => if (supports_q4_k_mmla) types.QuantizedMatmulRhsQ4_Kx2Mmla else types.QuantizedMatmulRhsQ4_Kx8,
        else => @compileError("PackedQuantRhsFor: no packed matmul RHS layout for dtype ." ++ @tagName(dt)),
    };
}

/// Pack raw [n, k] row blocks into the ISA-best packed RHS container for
/// their dtype (`PackedQuantRhsFor`). One dispatch replaces the per-format
/// exec packers; the per-format packers stay in their `quant/<fmt>.zig`
/// homes.
pub fn packRhs(
    comptime dt: DType,
    allocator: Allocator,
    blocks: []const dtype_mod.Storage(dt),
    n: usize,
    k: usize,
    blocks_per_row: usize,
) !PackedQuantRhsFor(dt) {
    return packRhsAs(PackedQuantRhsFor(dt), allocator, blocks, n, k, blocks_per_row);
}

/// Container-addressed variant of `packRhs`: pack into a specific RHS
/// container type. The explicit-layout escape hatch (e.g. the Q4_K x8 pack
/// on an smmla target); `packRhs` is the dtype-default entry.
pub fn packRhsAs(
    comptime Rhs: type,
    allocator: Allocator,
    blocks: []const dtype_mod.Storage(Rhs.dtype),
    n: usize,
    k: usize,
    blocks_per_row: usize,
) !Rhs {
    if (Rhs == types.QuantizedMatmulRhsQ8_0x4) return q8_0.packMatmulRhsQ8_0x4(allocator, blocks, n, k, blocks_per_row);
    if (Rhs == types.QuantizedMatmulRhsQ6_Kx4) return q6_k.packMatmulRhsQ6_Kx4(allocator, blocks, n, k, blocks_per_row);
    if (Rhs == types.QuantizedMatmulRhsQ5_Kx8) return q5_k.packMatmulRhsQ5_Kx8(allocator, blocks, n, k, blocks_per_row);
    if (Rhs == types.QuantizedMatmulRhsQ4_Kx4) return q4_k.packMatmulRhsQ4_Kx4(allocator, blocks, n, k, blocks_per_row);
    if (Rhs == types.QuantizedMatmulRhsQ4_Kx8) return q4_k.packMatmulRhsQ4_Kx8(allocator, blocks, n, k, blocks_per_row);
    if (Rhs == types.QuantizedMatmulRhsQ4_Kx2Mmla) return q4_k.packMatmulRhsQ4_Kx2Mmla(allocator, blocks, n, k, blocks_per_row);
    comptime unreachable;
}

/// The K-quant and ternary RHS constructors the loaders call by name
/// (the model band and the weights band build containers from GGUF rows).
pub const quantizedMatmulRhsQ5_KFromBlocks = q8k.quantizedMatmulRhsQ5_KFromBlocks;
pub const quantizedMatmulRhsTQ2_0FromBorrowedBlocks = ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks;

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
                acc += @as(f32, @floatFromInt(common.i8DotI32(qa_row[p0..p1], qw_col[p0..p1]))) * col_scales[g];
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

// Quantize an f32 RHS [k, n] to the i8 block-wise format. Symmetric per-(column,
// group) scales: scale = amax(group) / 127, q = round(w / scale) clamped to
// [-127, 127]. Weights are stored transposed as [n][k] so each column's k-vector
// is contiguous for the int8 dot.
pub fn quantizeRhsBlockwiseI8(
    allocator: Allocator,
    rhs: *const Tensor,
    group_size: usize,
) !types.QuantizedMatmulRhsI8 {
    const view = try rhs.rankView(2);
    const k = view.dim(0);
    const n = view.dim(1);
    const gs = types.QuantizedMatmulRhsI8.effectiveGroupSize(group_size);
    const num_groups = types.QuantizedMatmulRhsI8.groupCountForSize(k, gs);

    const src = try rhs.dataConstChecked();

    // Storage is [n][k] (one contiguous k-vector per output column); scales
    // are [n][num_groups].
    var qw = try tensor.TensorOf(.i8).zeros(allocator, &.{ n, k });
    errdefer qw.deinit();
    var scales = try Tensor.zeros(allocator, &.{ n, num_groups });
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
            sd[j * num_groups + g] = scale;

            const inv: f32 = if (scale == 0) 0 else 1.0 / scale;
            p = p0;
            while (p < p1) : (p += 1) qwd[j * k + p] = common.quantizeToI8(src[p * n + j] * inv);
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
        for (qrow, row) |*q, v| q.* = common.quantizeToI8(v * inv);
    }
}

pub fn quantizeMatmulRhsQ8_0(allocator: Allocator, rhs: *const Tensor) !types.QuantizedMatmulRhsQ8_0 {
    const view = try rhs.rankView(2);
    const k = view.dim(0);
    const n = view.dim(1);
    const blocks_per_column = try types.blockCountForDType(.q8_0, k);
    const data = try rhs.dataConstChecked();

    const blocks = try allocator.alloc(dtype_mod.BlockQ8_0, try types.checkedProduct(n, blocks_per_column));
    errdefer allocator.free(blocks);
    const scratch = try allocator.alloc(f32, k);
    defer allocator.free(scratch);

    var col: usize = 0;
    while (col < n) : (col += 1) {
        var p: usize = 0;
        while (p < k) : (p += 1) scratch[p] = data[p * n + col];
        try q8k.quantizeRowQ8_0Into(
            blocks[col * blocks_per_column ..][0..blocks_per_column],
            scratch,
        );
    }

    return .{
        .allocator = allocator,
        .blocks = blocks,
        .blocks_per_column = blocks_per_column,
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
    const blocks_per_row = try types.blockCountForDType(tensor_dtype, cols);

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
    const blocks_per_row = try types.blockCountForDType(tensor_dtype, cols);

    for (indices, 0..) |index, row| {
        if (index >= rows) return tensor.TensorError.IndexOutOfBounds;
        try dequantizeRowForDType(
            tensor_dtype,
            out[row * cols ..][0..cols],
            blocks[index * blocks_per_row ..][0..blocks_per_row],
        );
    }
}

/// How many leading rows of an m-row batch the x4-lane-packed LHS kernels
/// take, per K-quant weight: the padded Q4_K kernel takes every batch of at
/// least `parallel.q4_k_x4_min_rows` rows in one pass (a partial last
/// group is masked); Q5_K has no padded kernel, so its remainder rows cost
/// a second pass over the packed weights and the x4 arm only pays from
/// `parallel.q5_k_x4_prefix_min_rows` rows (or an exact multiple of 4);
/// Q6_K has no x4 LHS kernel. One rule for the exec fused engine and the
/// native dispatch tier.
pub fn x4PrefixRows(comptime weight: DType, m: usize) usize {
    return switch (weight) {
        .q4_k => if (m % 4 == 0 or m >= parallel.q4_k_x4_min_rows) m else 0,
        .q5_k => if (m % 4 == 0 or m >= parallel.q5_k_x4_prefix_min_rows) m - m % 4 else 0,
        .q6_k => 0,
        else => @compileError("x4PrefixRows: no x4-packed LHS arm for ." ++ @tagName(weight)),
    };
}

/// The format file owning `weight`'s kernels (its `kernels` table).
fn formatOf(comptime weight: DType) type {
    return switch (weight) {
        .q4_k => q4_k,
        .q5_k => q5_k,
        .q6_k => q6_k,
        .q8_0 => q8_0,
        .tq2_0, .q2_0 => ternary,
        else => cold,
    };
}

/// True when a kernel exists for `g`: the owning format's `kernels` table
/// has an entry for exactly this request. The one matrix of existing
/// kernels; `gemm` dispatches on the same table.
pub fn supported(comptime g: ops.QuantGemm) bool {
    comptime {
        for (formatOf(g.weight).kernels) |k| {
            if (std.meta.eql(k.g, g)) return true;
        }
        return false;
    }
}

/// `@compileError` naming the combination unless `supported`.
pub fn check(comptime g: ops.QuantGemm) void {
    if (comptime !supported(g)) @compileError(std.fmt.comptimePrint(
        "quant.gemm: no kernel for weight .{s} with rhs .{s}, lhs .{s}, order .{s}",
        .{ @tagName(g.weight), @tagName(g.rhs), @tagName(g.lhs), @tagName(g.order) },
    ));
}

/// The one quantized GEMM entry over a request `g` (`backend/ops.zig`
/// `QuantGemm`): the owning format's `kernels` table entry for `g` runs
/// its tile body over `[r0, r1) x [c0, c1)` at output row stride `rhs.n`.
/// An unsupported combination is a compile error naming it.
pub fn gemm(comptime g: ops.QuantGemm, out: []f32, lhs: ops.LhsOf(g), rhs: ops.RhsOf(g), tile: ops.Tile) void {
    comptime check(g);
    const n = rhs.n;
    inline for (formatOf(g.weight).kernels) |k| {
        if (comptime std.meta.eql(k.g, g)) return k.tile(out, lhs, rhs, n, tile.r0, tile.r1, tile.c0, tile.c1);
    }
    comptime unreachable;
}

/// True when `dt` has the batched column-outer compact kernel
/// (`matmul*RhsCompactColOuter`); q2_k/q3_k stay on the row-outer tile.
pub fn hasCompactColOuter(comptime dt: DType) bool {
    return supported(.{ .weight = dt, .lhs = .q8_k, .order = .col_outer });
}

/// Row-outer tile over a compact RHS view, by format: the
/// `.{ .rows, .q8_k, .row_outer }` gemm selection (the MoE expert tile's
/// entry; `n` must be the view's width).
pub fn matmulCompactRhsTile(comptime dt: DType, out: []f32, qlhs: []const dtype_mod.BlockQ8_K, view: *const types.CompactRhs(dt), n: usize, r0: usize, m: usize, c0: usize, c1: usize) void {
    std.debug.assert(view.n == n);
    gemm(.{ .weight = dt, .lhs = .q8_k }, out, qlhs, view, .{ .r0 = r0, .r1 = m, .c0 = c0, .c1 = c1 });
}

/// Batched column-outer compact kernel (the `hasCompactColOuter` formats):
/// the `.{ .rows, .q8_k, .col_outer }` gemm selection.
pub fn matmulCompactColOuter(comptime dt: DType, out: []f32, qlhs: []const dtype_mod.BlockQ8_K, view: *const types.CompactRhs(dt), n: usize, r0: usize, m: usize, c0: usize, c1: usize) void {
    std.debug.assert(view.n == n);
    gemm(.{ .weight = dt, .lhs = .q8_k, .order = .col_outer }, out, qlhs, view, .{ .r0 = r0, .r1 = m, .c0 = c0, .c1 = c1 });
}

/// Lane-packed (Q8_Kx4 LHS) column-outer compact kernel: the
/// `.{ .rows, .q8_kx4, .col_outer }` gemm selection.
pub fn matmulCompactQ8_Kx4ColOuter(comptime dt: DType, out: []f32, lhs_x4: []const types.BlockQ8_Kx4, view: *const types.CompactRhs(dt), n: usize, m: usize, c0: usize, c1: usize) void {
    std.debug.assert(view.n == n);
    gemm(.{ .weight = dt, .lhs = .q8_kx4, .order = .col_outer }, out, lhs_x4, view, .{ .r1 = m, .c0 = c0, .c1 = c1 });
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
        .q8_0 => try q8k.dequantizeRowQ8_0Into(dst, blocks),
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
/// blocks and `dst.len` must match (`types.blockCountForDType`). Dtypes without an
/// f32 encoder (and non-quantized dtypes) are a compile error. Inputs are
/// assumed finite (no NaN/inf), as in ggml's encoders.
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
        .q8_0 => try q8k.quantizeRowQ8_0Into(dst, src),
        .q8_1 => try cold.quantizeRowQ8_1Into(dst, src),
        .q4_k => try q4_k.quantizeRowQ4_KInto(dst, src),
        .q5_k => try q5_k.quantizeRowQ5_KInto(dst, src),
        .q6_k => try q6_k.quantizeRowQ6_KInto(dst, src),
        .q8_k => try q8k.quantizeRowQ8_KInto(dst, src),
        .tq2_0 => try ternary.quantizeRowTQ2_0Into(dst, src),
        else => @compileError("dtype has no f32 -> quantized row encoder"),
    }
}

fn dequantizeKRowInto(
    comptime tensor_dtype: DType,
    dst: []f32,
    blocks: []const dtype_mod.Storage(tensor_dtype),
) !void {
    if (dst.len != try types.checkedProduct(blocks.len, dtype_mod.qk_k_block_size)) return types.QuantizedFormatError.InvalidQuantizedLength;

    var dense_block: [dtype_mod.qk_k_block_size]f32 = undefined;
    for (blocks, 0..) |*block, block_index| {
        switch (tensor_dtype) {
            .q2_k => cold.dequantizeBlockQ2_KInto(&dense_block, block),
            .q3_k => cold.dequantizeBlockQ3_KInto(&dense_block, block),
            .q4_k => q4_k.dequantizeBlockQ4_KInto(&dense_block, block),
            .q5_k => q5_k.dequantizeBlockQ5_KInto(&dense_block, block),
            .q6_k => q6_k.dequantizeBlockQ6_KInto(&dense_block, block),
            .q8_k => q8k.dequantizeBlockQ8_KInto(&dense_block, block),
            else => unreachable,
        }
        @memcpy(dst[block_index * dtype_mod.qk_k_block_size ..][0..dtype_mod.qk_k_block_size], &dense_block);
    }
}

test {
    _ = @import("quant_tests.zig");
    _ = @import("quant/common.zig");
    _ = @import("quant/cold.zig");
    _ = @import("quant/q8_0.zig");
    _ = @import("quant/q4_k.zig");
    _ = @import("quant/q5_k.zig");
    _ = @import("quant/q6_k.zig");
    _ = @import("quant/ternary.zig");
    _ = @import("quant/mxfp4.zig");
    _ = @import("quant/encode_golden_test.zig");
}
