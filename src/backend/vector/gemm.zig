//! Dense f32/f16/f64/bf16 GEMM: the one `gemm` entry over `ops.Gemm`
//! (orientation, operand dtypes, store/accumulate), its rows/cols parallel
//! dispatch over `tile.forRange`, and the inner kernels: the NN/TN
//! row-block kernel `gemmRows` over an `NnFamily` (f32 plain, f32 trans-A,
//! f16, bf16) and the NT tile/dot kernels over an `NtFamily` (f32, f16 in
//! native lanes, f16 widened, bf16 RHS). Shared-core symbols
//! (ParallelConfig, contiguous-data helpers, thread-count gates, the V*
//! width aliases) come from `common.zig`; the @Vector primitives from
//! `primitives.zig`; blocked-tile cores from `gemm_blocked.zig`.
//! gemmNNRange/gemmTNRange/gemmNTRange are pub for the batched GEMM module.

const std = @import("std");
const isa = @import("../isa.zig");
const dtype_mod = @import("../../dtype.zig");
const gemm_blocked = @import("gemm_blocked.zig");
const parallel = @import("../../parallel.zig");
const tensor = @import("../../tensor.zig");
const thread = @import("../../thread.zig");
const common = @import("common.zig");
const tile = @import("tile.zig");
const ops = @import("../ops.zig");
const primitives = @import("primitives.zig");

const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

const ParallelConfig = common.ParallelConfig;
const vector_len = common.vector_len;
const vector_len_f16 = common.vector_len_f16;
const Vf32 = common.Vf32;
const Vf16 = common.Vf16;
const Vf16ForF32 = common.Vf16ForF32;
const f32VecToBf16 = primitives.f32VecToBf16;
const Vf32Wide = common.Vf32ForF16;

// f16-RHS GEMM accumulator policy. On aarch64 NEON the f16 x f16 @mulAdd arms
// are native fmla.8h (double the f32 lane throughput), so half-precision
// accumulation is the fast path there (the `.f16_native` NT family). Every
// other ISA — x86-64 without AVX512-FP16 in particular — legalizes f16 vector
// arithmetic by promoting through f32 and rounding back PER OPERATION
// (vcvtph2ps/vcvtps2ph around every fmadd, scalarized reductions,
// accumulator spills), so those targets take the `.f16_wide` family: widen
// each f16 load once (F16C vcvtph2ps) and accumulate in f32, mirroring the
// bf16-RHS kernels. This changes non-aarch64 results (one final rounding
// instead of per-step f16 rounding — strictly more accurate); aarch64 output
// is bit-identical to before.
const f16_accum_native = isa.is_aarch64;
const f16_rhs_family: NtFamily = if (f16_accum_native) .f16_native else .f16_wide;

// ---------------- MatMul (2-D) ----------------

/// GEMM output-tile write policy: `.store` writes the accumulator (C = A·B),
/// `.accumulate` adds it to the existing output (C += A·B — the BLAS beta=1
/// epilogue). Comptime so `.store` kernels compile exactly as before.
pub const StoreMode = enum { store, accumulate };

/// The dense GEMM over contiguous slices: one entry for every orientation
/// and operand-dtype combination the vector kernels implement (the set is
/// stated on `ops.Gemm`). `cd` is `[m, n]`; `ad` is `[m, k]` (`[k, m]` for
/// `.trans_a`); `bd` is `[k, n]` (`[n, k]` for `.trans_b`).
///
/// f32 NN: C[i, j] = sum_p A[i, p] * B[p, j]. The natural inner order
/// (i, j, p) reads B strided in p, which kills vectorization; the row
/// kernels reorder to (i, p, j), broadcasting A[i, p] against a contiguous
/// slice of B's row p. TN reorders to (p, i, j) the same way; NT is a
/// textbook dot product per output element over two contiguous streams.
/// Large shapes take the cache-blocked packed kernel (`gemm_blocked.zig`).
pub fn gemm(
    pc: ParallelConfig,
    comptime g: ops.Gemm,
    cd: []dtype_mod.Scalar(g.out),
    ad: []const dtype_mod.Scalar(g.a),
    bd: []const dtype_mod.Scalar(g.b),
    m: usize,
    n: usize,
    k: usize,
) void {
    if (comptime isa.reference) return scalar.gemm(g, cd, ad, bd, m, n, k);
    if (comptime g.isF32()) {
        switch (comptime g.kind) {
            .plain => {
                if (gemm_blocked.shouldUseBlocked(m, n, k)) {
                    if (comptime g.accumulate) return gemm_blocked.gemmBlockedAcc(pc, .plain, cd, ad, bd, m, n, k);
                    return gemm_blocked.gemmBlocked(pc, .plain, cd, ad, bd, m, n, k);
                }
                const mode: StoreMode = if (g.accumulate) .accumulate else .store;
                if (maybeParallelNN(pc, mode, cd, ad, bd, m, n, k)) return;
                gemmNNRangeMode(mode, cd, ad, bd, m, n, k, 0, m);
            },
            .trans_a => {
                comptime if (g.accumulate) @compileError("gemm: accumulate is the f32 `.plain` epilogue only");
                if (gemm_blocked.shouldUseBlocked(m, n, k)) {
                    return gemm_blocked.gemmBlocked(pc, .trans_a, cd, ad, bd, m, n, k);
                }
                gemmTNRowPath(pc, cd, ad, bd, m, n, k);
            },
            .trans_b => {
                comptime if (g.accumulate) @compileError("gemm: accumulate is the f32 `.plain` epilogue only");
                if (gemm_blocked.shouldUseBlocked(m, n, k)) {
                    return gemm_blocked.gemmBlocked(pc, .trans_b, cd, ad, bd, m, n, k);
                }
                gemmNTRowPath(pc, cd, ad, bd, m, n, k);
            },
        }
        return;
    }
    comptime if (g.accumulate) @compileError("gemm: accumulate is the f32 `.plain` epilogue only");
    if (comptime g.kind == .plain and g.a == g.b and g.out == dtype_mod.outputDType(.matmul, g.a)) {
        // The typed NN family: f64 stays f64; f16/bf16 take their own row
        // kernels (f32 accumulation, one final round).
        if (comptime g.a == .f64) {
            if (maybeParallelNNF64(pc, cd, ad, bd, m, n, k)) return;
            return gemmNNRangeF64(cd, ad, bd, m, n, k, 0, m);
        } else if (comptime g.a == .f16) {
            if (maybeParallelNNF16(pc, cd, ad, bd, m, n, k)) return;
            return gemmNNRangeF16(cd, ad, bd, m, n, k, 0, m);
        } else if (comptime g.a == .bf16) {
            if (maybeParallelNNBf16(pc, cd, ad, bd, m, n, k)) return;
            return gemmNNRangeBf16(cd, ad, bd, m, n, k, 0, m);
        }
        return matmul2DIntoTypedScalar(g.a, cd, ad, bd, m, n, k);
    }
    if (comptime g.kind == .trans_b and g.a == .f16 and g.b == .f16 and g.out == .f32) {
        if (maybeParallelNTF16Rhs(pc, cd, ad, bd, m, n, k)) return;
        return gemmNTF16RhsRange(cd, ad, bd, m, n, k, 0, m);
    }
    if (comptime g.kind == .trans_b and g.a == .f32 and g.b == .bf16 and g.out == .f32) {
        // Mixed-precision NT: f32 LHS activations against a frozen bf16 RHS
        // stored [n, k]; the bf16 weights widen to f32 in-register (u16 << 16,
        // exact) and everything accumulates in f32.
        if (maybeParallelNTBf16Rhs(pc, cd, ad, bd, m, n, k)) return;
        return gemmNTBf16RhsRange(cd, ad, bd, m, n, k, 0, m);
    }
    @compileError("gemm: no vector kernel for kind ." ++ @tagName(g.kind) ++ " over ." ++ @tagName(g.a) ++ " x ." ++ @tagName(g.b) ++ " -> ." ++ @tagName(g.out));
}

// The pre-blocking register-tiled row-kernel paths, bypassing the blocked
// dispatch above. Public so the GEMM bench can baseline them directly.
pub fn gemmNNRowPath(pc: ParallelConfig, cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize) void {
    if (maybeParallelNN(pc, .store, cd, ad, bd, m, n, k)) return;
    gemmNNRangeMode(.store, cd, ad, bd, m, n, k, 0, m);
}

pub fn gemmTNRowPath(pc: ParallelConfig, cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize) void {
    if (maybeParallelTN(pc, cd, ad, bd, m, n, k)) return;
    gemmTNRange(cd, ad, bd, m, n, k, 0, m);
}

pub fn gemmNTRowPath(pc: ParallelConfig, cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize) void {
    if (maybeParallelNT(pc, cd, ad, bd, m, n, k)) return;
    gemmNTRange(cd, ad, bd, m, n, k, 0, m);
}

// ---------------- Parallel dispatch ----------------

// One payload shape per element-type family; the splits and spawns are
// `tile.forRange` (same proportional boundaries as the retired per-family
// splitters, bitwise-neutral).
fn GemmCtx(comptime Cd: type, comptime Ad: type, comptime Bd: type) type {
    return struct {
        cd: []Cd,
        ad: []const Ad,
        bd: []const Bd,
        m: usize,
        n: usize,
        k: usize,
    };
}

const Ctx32 = GemmCtx(f32, f32, f32);

/// Adapt a `(cd, ad, bd, m, n, k, start, end)` range kernel onto the tile
/// payload.
fn rangeRunner(comptime Ctx: type, comptime rangeFn: anytype) fn (Ctx, usize, usize) void {
    return struct {
        fn go(c: Ctx, start: usize, end: usize) void {
            rangeFn(c.cd, c.ad, c.bd, c.m, c.n, c.k, start, end);
        }
    }.go;
}

/// The dense families' one rows/cols split: the column arm (when present)
/// fires below the decode-m gate through `columnThreadCount`, else the row
/// arm through `matmulThreadCount` — the historical gates, thresholds and
/// split points unchanged.
fn maybeRowsCols(
    pc: ParallelConfig,
    comptime Ctx: type,
    ctx: Ctx,
    m: usize,
    n: usize,
    k: usize,
    comptime rowsFn: fn (Ctx, usize, usize) void,
    comptime colsFn: ?fn (Ctx, usize, usize) void,
) bool {
    const pool = pc.pool orelse return false;
    if (comptime colsFn != null) {
        if (m < parallel.vector_column_min_m) {
            const thread_count = common.columnThreadCount(m, n, k);
            if (thread_count != 1) {
                tile.forRange(pool, Ctx, ctx, n, thread_count, colsFn.?);
                return true;
            }
        }
    }
    const thread_count = common.matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    tile.forRange(pool, Ctx, ctx, m, thread_count, rowsFn);
    return true;
}

fn maybeParallelNN(pc: ParallelConfig, comptime mode: StoreMode, cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize) bool {
    const runners = struct {
        fn rows(c: Ctx32, start: usize, end: usize) void {
            gemmNNRangeMode(mode, c.cd, c.ad, c.bd, c.m, c.n, c.k, start, end);
        }
        fn cols(c: Ctx32, start: usize, end: usize) void {
            gemmNNColsMode(mode, c.cd, c.ad, c.bd, c.m, c.n, c.k, start, end);
        }
    };
    return maybeRowsCols(pc, Ctx32, .{ .cd = cd, .ad = ad, .bd = bd, .m = m, .n = n, .k = k }, m, n, k, runners.rows, runners.cols);
}

fn maybeParallelNNF64(pc: ParallelConfig, cd: []f64, ad: []const f64, bd: []const f64, m: usize, n: usize, k: usize) bool {
    const Ctx = GemmCtx(f64, f64, f64);
    return maybeRowsCols(pc, Ctx, .{ .cd = cd, .ad = ad, .bd = bd, .m = m, .n = n, .k = k }, m, n, k, rangeRunner(Ctx, gemmNNRangeF64), null);
}

fn maybeParallelNNF16(pc: ParallelConfig, cd: []f16, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize) bool {
    const Ctx = GemmCtx(f16, f16, f16);
    return maybeRowsCols(pc, Ctx, .{ .cd = cd, .ad = ad, .bd = bd, .m = m, .n = n, .k = k }, m, n, k, rangeRunner(Ctx, gemmNNRangeF16), rangeRunner(Ctx, gemmNNColsF16));
}

fn maybeParallelNNBf16(pc: ParallelConfig, cd: []u16, ad: []const u16, bd: []const u16, m: usize, n: usize, k: usize) bool {
    const Ctx = GemmCtx(u16, u16, u16);
    return maybeRowsCols(pc, Ctx, .{ .cd = cd, .ad = ad, .bd = bd, .m = m, .n = n, .k = k }, m, n, k, rangeRunner(Ctx, gemmNNRangeBf16), null);
}

fn maybeParallelTN(pc: ParallelConfig, cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize) bool {
    return maybeRowsCols(pc, Ctx32, .{ .cd = cd, .ad = ad, .bd = bd, .m = m, .n = n, .k = k }, m, n, k, rangeRunner(Ctx32, gemmTNRange), rangeRunner(Ctx32, gemmTNCols));
}

fn maybeParallelNT(pc: ParallelConfig, cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize) bool {
    return maybeRowsCols(pc, Ctx32, .{ .cd = cd, .ad = ad, .bd = bd, .m = m, .n = n, .k = k }, m, n, k, rangeRunner(Ctx32, gemmNTRange), rangeRunner(Ctx32, gemmNTCols));
}

fn maybeParallelNTF16Rhs(pc: ParallelConfig, cd: []f32, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize) bool {
    const Ctx = GemmCtx(f32, f16, f16);
    return maybeRowsCols(pc, Ctx, .{ .cd = cd, .ad = ad, .bd = bd, .m = m, .n = n, .k = k }, m, n, k, rangeRunner(Ctx, gemmNTF16RhsRange), rangeRunner(Ctx, gemmNTF16RhsCols));
}

fn maybeParallelNTBf16Rhs(pc: ParallelConfig, cd: []f32, ad: []const f32, bd: []const u16, m: usize, n: usize, k: usize) bool {
    const Ctx = GemmCtx(f32, f32, u16);
    return maybeRowsCols(pc, Ctx, .{ .cd = cd, .ad = ad, .bd = bd, .m = m, .n = n, .k = k }, m, n, k, rangeRunner(Ctx, gemmNTBf16RhsRange), rangeRunner(Ctx, gemmNTBf16RhsCols));
}

/// The empty-range / k = 0 preamble every range and cols kernel shares:
/// nothing to do for an empty rectangle; at k = 0 a store zeroes its
/// rectangle of C and an accumulate leaves it untouched. True when the
/// caller is done.
inline fn zeroIfEmpty(comptime mode: StoreMode, cd: anytype, n: usize, k: usize, row_start: usize, row_end: usize, col_start: usize, col_end: usize) bool {
    if (row_start == row_end or col_start == col_end) return true;
    if (k != 0) return false;
    if (comptime mode == .store) {
        for (row_start..row_end) |i| @memset(cd[i * n + col_start .. i * n + col_end], 0);
    }
    return true;
}

// ---------------- NN/TN row-block kernels ----------------

/// The row-block kernel's operand families: the element type, where A[r, p]
/// sits (row-major `[m, k]`, or `[k, m]` for the transposed-A walk), how an
/// element widens to the f32 accumulator, whether the product fuses
/// (`@mulAdd` for f32; the 16-bit families keep the separate multiply and
/// add), and how the accumulator narrows on store.
const NnFamily = enum {
    f32,
    f32_trans_a,
    f16,
    bf16,

    fn Elem(comptime fam: NnFamily) type {
        return switch (fam) {
            .f32, .f32_trans_a => f32,
            .f16 => f16,
            .bf16 => u16,
        };
    }

    fn isF32(comptime fam: NnFamily) bool {
        return fam == .f32 or fam == .f32_trans_a;
    }

    inline fn elemA(comptime fam: NnFamily, ad: []const fam.Elem(), row: usize, p: usize, m: usize, k: usize) f32 {
        return switch (fam) {
            .f32 => ad[row * k + p],
            .f32_trans_a => ad[p * m + row],
            .f16 => @floatCast(ad[row * k + p]),
            .bf16 => dtype_mod.bf16ToF32(ad[row * k + p]),
        };
    }

    inline fn vecB(comptime fam: NnFamily, bd: []const fam.Elem(), idx: usize) Vf32 {
        return switch (fam) {
            .f32, .f32_trans_a => bd[idx..][0..vector_len].*,
            .f16 => @floatCast(@as(Vf16ForF32, bd[idx..][0..vector_len].*)),
            .bf16 => primitives.bf16VecToF32(bd[idx..][0..vector_len].*),
        };
    }

    inline fn elemB(comptime fam: NnFamily, bd: []const fam.Elem(), idx: usize) f32 {
        return switch (fam) {
            .f32, .f32_trans_a => bd[idx],
            .f16 => @floatCast(bd[idx]),
            .bf16 => dtype_mod.bf16ToF32(bd[idx]),
        };
    }

    inline fn mulAddVec(comptime fam: NnFamily, av: Vf32, bv: Vf32, acc: Vf32) Vf32 {
        return if (comptime fam.isF32()) @mulAdd(Vf32, av, bv, acc) else acc + av * bv;
    }

    inline fn mulAdd(comptime fam: NnFamily, av: f32, bv: f32, s: f32) f32 {
        return if (comptime fam.isF32()) @mulAdd(f32, av, bv, s) else s + av * bv;
    }

    inline fn storeVec(comptime fam: NnFamily, comptime mode: StoreMode, dst: *[vector_len]fam.Elem(), acc: Vf32) void {
        dst.* = switch (fam) {
            .f32, .f32_trans_a => switch (mode) {
                .store => acc,
                .accumulate => @as(Vf32, dst.*) + acc,
            },
            .f16 => @as(Vf16ForF32, @floatCast(acc)),
            .bf16 => f32VecToBf16(acc),
        };
    }

    inline fn storeScalar(comptime fam: NnFamily, comptime mode: StoreMode, dst: *fam.Elem(), s: f32) void {
        dst.* = switch (fam) {
            .f32, .f32_trans_a => switch (mode) {
                .store => s,
                .accumulate => dst.* + s,
            },
            .f16 => @floatCast(s),
            .bf16 => dtype_mod.f32ToBf16(s),
        };
    }
};

/// Column-vector unroll of a row block: two vectors per pass for the row
/// blocks, one for the single row (the historical widths).
fn colWidths(comptime R: usize) []const usize {
    return if (R == 1) &.{1} else &.{ 2, 1 };
}

/// C[row .. row + R, col_start .. col_end) (+)= A · B for one `R`-row
/// block in the (i, p, j) order: A[r, p] broadcast against contiguous B row
/// vectors, `colWidths(R)` vectors per pass, then the scalar column tail.
/// `inline for` over rows and vectors, so the emitted per-element op order
/// is the hand-unrolled 12/8/4/1 kernels' this replaces.
inline fn gemmRows(
    comptime R: usize,
    comptime fam: NnFamily,
    comptime mode: StoreMode,
    cd: []fam.Elem(),
    ad: []const fam.Elem(),
    bd: []const fam.Elem(),
    row: usize,
    m: usize,
    n: usize,
    k: usize,
    col_start: usize,
    col_end: usize,
) void {
    comptime if (mode == .accumulate and !fam.isF32()) @compileError("gemmRows: accumulate is the f32 epilogue only");
    var j = col_start;
    inline for (comptime colWidths(R)) |C| {
        while (j + C * vector_len <= col_end) : (j += C * vector_len) {
            var acc: [R][C]Vf32 = undefined;
            inline for (0..R) |r| {
                inline for (0..C) |c| acc[r][c] = @splat(0);
            }
            for (0..k) |p| {
                var b: [C]Vf32 = undefined;
                inline for (0..C) |c| b[c] = fam.vecB(bd, p * n + j + c * vector_len);
                inline for (0..R) |r| {
                    const av: Vf32 = @splat(fam.elemA(ad, row + r, p, m, k));
                    inline for (0..C) |c| acc[r][c] = fam.mulAddVec(av, b[c], acc[r][c]);
                }
            }
            inline for (0..R) |r| {
                inline for (0..C) |c| fam.storeVec(mode, cd[(row + r) * n + j + c * vector_len ..][0..vector_len], acc[r][c]);
            }
        }
    }
    while (j < col_end) : (j += 1) {
        var sums: [R]f32 = [_]f32{0} ** R;
        for (0..k) |p| {
            const bv = fam.elemB(bd, p * n + j);
            inline for (0..R) |r| sums[r] = fam.mulAdd(fam.elemA(ad, row + r, p, m, k), bv, sums[r]);
        }
        inline for (0..R) |r| fam.storeScalar(mode, &cd[(row + r) * n + j], sums[r]);
    }
}

/// Rows [row_start, row_end) of C over columns [col_start, col_end) as row
/// blocks: the widest width in `widths` that still fits, then the narrower
/// ones (the 12/8/4/1 walk of the 16-bit NN families, 8/4/1 of the f32
/// ones, single rows for the column-split kernels), each block through
/// `gemmRows`.
inline fn gemmRowBlocks(
    comptime widths: []const usize,
    comptime fam: NnFamily,
    comptime mode: StoreMode,
    cd: []fam.Elem(),
    ad: []const fam.Elem(),
    bd: []const fam.Elem(),
    m: usize,
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) void {
    // The walk unrolls every width's rows x vectors inline-for nest into
    // this scope, past the default 1000-branch quota.
    @setEvalBranchQuota(10_000);
    var i = row_start;
    inline for (widths) |R| {
        while (i + R <= row_end) : (i += R) gemmRows(R, fam, mode, cd, ad, bd, i, m, n, k, col_start, col_end);
    }
}

pub fn gemmNNRange(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    gemmNNRangeMode(.store, cd, ad, bd, m, n, k, row_start, row_end);
}

fn gemmNNRangeMode(comptime mode: StoreMode, cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    if (zeroIfEmpty(mode, cd, n, k, row_start, row_end, 0, n)) return;
    gemmRowBlocks(&.{ 8, 4, 1 }, .f32, mode, cd, ad, bd, m, n, k, row_start, row_end, 0, n);
}

fn gemmNNColsMode(comptime mode: StoreMode, cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (zeroIfEmpty(mode, cd, n, k, 0, m, col_start, col_end)) return;
    gemmRowBlocks(&.{1}, .f32, mode, cd, ad, bd, m, n, k, 0, m, col_start, col_end);
}

pub fn gemmTNRange(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    if (zeroIfEmpty(.store, cd, n, k, row_start, row_end, 0, n)) return;
    gemmRowBlocks(&.{ 8, 4, 1 }, .f32_trans_a, .store, cd, ad, bd, m, n, k, row_start, row_end, 0, n);
}

fn gemmTNCols(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (zeroIfEmpty(.store, cd, n, k, 0, m, col_start, col_end)) return;
    gemmRowBlocks(&.{1}, .f32_trans_a, .store, cd, ad, bd, m, n, k, 0, m, col_start, col_end);
}

fn gemmNNRangeF16(cd: []f16, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    if (zeroIfEmpty(.store, cd, n, k, row_start, row_end, 0, n)) return;
    gemmRowBlocks(&.{ 12, 8, 4, 1 }, .f16, .store, cd, ad, bd, m, n, k, row_start, row_end, 0, n);
}

// Column-sliced f16 NN kernel: columns [col_start, col_end) for all m rows
// (row stride stays `n`). Used for small-m (< vector_column_min_m) f16
// matmuls where row-parallelism is denied; rows tile 8/4/1 to reuse each
// loaded B vector across the tile.
fn gemmNNColsF16(cd: []f16, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (zeroIfEmpty(.store, cd, n, k, 0, m, col_start, col_end)) return;
    gemmRowBlocks(&.{ 8, 4, 1 }, .f16, .store, cd, ad, bd, m, n, k, 0, m, col_start, col_end);
}

/// All-bf16 C[row_start..row_end, n] = A · B (operands AND result bf16;
/// accumulation in f32, one rounding at the final store). Row-blocked
/// 12/8/4/1 over the row range so a thread team splits by rows; each block
/// walks B once in k-major order (the NN layout's streaming direction).
fn gemmNNRangeBf16(cd: []u16, ad: []const u16, bd: []const u16, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    if (zeroIfEmpty(.store, cd, n, k, row_start, row_end, 0, n)) return;
    gemmRowBlocks(&.{ 12, 8, 4, 1 }, .bf16, .store, cd, ad, bd, m, n, k, row_start, row_end, 0, n);
}

// ---------------- NT tile and dot kernels ----------------

/// The NT (`C = A · Bᵀ`, B rows contiguous in k) operand families: the
/// operand and accumulator element types, the vector widths the rows x 4
/// tiles and the single-row `dot4` run at, and how each operand reaches
/// the accumulator. `.f16_native` accumulates in f16 lanes (aarch64
/// fmla.8h); `.f16_wide` and `.bf16` widen each load to f32 (see
/// `f16_accum_native`).
const NtFamily = enum {
    f32,
    f16_native,
    f16_wide,
    bf16,

    fn A(comptime fam: NtFamily) type {
        return switch (fam) {
            .f32, .bf16 => f32,
            .f16_native, .f16_wide => f16,
        };
    }

    fn B(comptime fam: NtFamily) type {
        return switch (fam) {
            .f32 => f32,
            .f16_native, .f16_wide => f16,
            .bf16 => u16,
        };
    }

    fn Acc(comptime fam: NtFamily) type {
        return if (fam == .f16_native) f16 else f32;
    }

    /// rows x 4 tile width: the f16 lane width for the native family, the
    /// f32 width otherwise. Vf32-width groups for bf16 ON PURPOSE: rows x 4
    /// accumulators at the wide (f16-vector) width are 32 NEON registers at
    /// rows = 4 — total spill, measured 1.6x SLOWER at m=4. The wide widen
    /// lives only in the single-row kernels (`dot4` / `vecDotBf16RhsToF32`),
    /// whose 4-8 accumulators fit with room for operands.
    fn tileWidth(comptime fam: NtFamily) usize {
        return if (fam == .f16_native) vector_len_f16 else vector_len;
    }

    /// Single-row `dot4` width: the f16 lane width for the native family;
    /// the wide (f16-vector) width for bf16, where one u16 load +
    /// shift-widen feeds a double-width f32 FMA, halving loop overhead vs
    /// Vf32 groups — the bf16 arm's answer to the f16 kernels' native lane
    /// width.
    fn dotWidth(comptime fam: NtFamily) usize {
        return switch (fam) {
            .f16_native, .bf16 => vector_len_f16,
            .f32, .f16_wide => vector_len,
        };
    }

    /// Accumulator chains per column in `dot4`: two for f32 (even/odd
    /// vector steps keep eight independent FMA chains in flight, since a
    /// single chain per column is latency-bound; the fused 4-chain form
    /// measured slower on the 253-row NT prefill shapes), one otherwise.
    fn dotChains(comptime fam: NtFamily) usize {
        return if (fam == .f32) 2 else 1;
    }

    inline fn vecA(comptime fam: NtFamily, comptime W: usize, ad: []const fam.A(), idx: usize) @Vector(W, fam.Acc()) {
        return switch (fam) {
            .f32, .f16_native, .bf16 => ad[idx..][0..W].*,
            .f16_wide => @floatCast(@as(@Vector(W, f16), ad[idx..][0..W].*)),
        };
    }

    inline fn vecB(comptime fam: NtFamily, comptime W: usize, bd: []const fam.B(), idx: usize) @Vector(W, fam.Acc()) {
        return switch (fam) {
            .f32, .f16_native => bd[idx..][0..W].*,
            .f16_wide => @floatCast(@as(@Vector(W, f16), bd[idx..][0..W].*)),
            .bf16 => if (W == vector_len_f16) primitives.bf16VecToF32Wide(bd[idx..][0..W].*) else primitives.bf16VecToF32(bd[idx..][0..W].*),
        };
    }

    inline fn elemA(comptime fam: NtFamily, ad: []const fam.A(), idx: usize) f32 {
        return switch (fam) {
            .f32, .bf16 => ad[idx],
            .f16_native, .f16_wide => @floatCast(ad[idx]),
        };
    }

    inline fn elemB(comptime fam: NtFamily, bd: []const fam.B(), idx: usize) f32 {
        return switch (fam) {
            .f32 => bd[idx],
            .f16_native, .f16_wide => @floatCast(bd[idx]),
            .bf16 => dtype_mod.bf16ToF32(bd[idx]),
        };
    }

    inline fn reduce(comptime fam: NtFamily, comptime W: usize, v: @Vector(W, fam.Acc())) f32 {
        return if (comptime fam == .f16_native) @floatCast(@reduce(.Add, v)) else @reduce(.Add, v);
    }

    /// The scalar tail step: fused for f32, the separate multiply and add
    /// for the 16-bit families.
    inline fn tail(comptime fam: NtFamily, s: f32, av: f32, bv: f32) f32 {
        return if (comptime fam == .f32) @mulAdd(f32, av, bv, s) else s + av * bv;
    }

    /// The single-column dot behind the leftover columns of a row.
    inline fn vecDot(comptime fam: NtFamily, x: []const fam.A(), y: []const fam.B()) f32 {
        return switch (fam) {
            .f32 => primitives.vecDot(x, y),
            .f16_native => vecDotF16HalfAccumToF32(x, y),
            .f16_wide => primitives.vecDotF16ToF32(x, y),
            .bf16 => vecDotBf16RhsToF32(x, y),
        };
    }
};

/// C[0..rows, col_start..col_end) = A[rows, k] · Bᵀ for one `rows`-row
/// block (`cd`/`ad` already sliced to the block): rows x 4 accumulator
/// tiles over the family's tile width, each B row loaded once per step and
/// broadcast down the rows, the lanes reduced to f32, the scalar tail fused
/// in; then the rows x 1 leftover columns the same way.
inline fn gemmNTRowsBlock(comptime rows: usize, comptime fam: NtFamily, cd: []f32, ad: []const fam.A(), bd: []const fam.B(), n: usize, k: usize, col_start: usize, col_end: usize) void {
    const W = comptime fam.tileWidth();
    const V = @Vector(W, fam.Acc());
    var j = col_start;
    while (j + 4 <= col_end) : (j += 4) {
        var acc: [rows][4]V = undefined;
        inline for (0..rows) |r| {
            inline for (0..4) |c| acc[r][c] = @splat(0);
        }

        var p: usize = 0;
        while (p + W <= k) : (p += W) {
            var b: [4]V = undefined;
            inline for (0..4) |c| b[c] = fam.vecB(W, bd, (j + c) * k + p);
            inline for (0..rows) |r| {
                const av = fam.vecA(W, ad, r * k + p);
                inline for (0..4) |c| acc[r][c] = @mulAdd(V, av, b[c], acc[r][c]);
            }
        }

        var sums: [rows][4]f32 = undefined;
        inline for (0..rows) |r| {
            inline for (0..4) |c| sums[r][c] = fam.reduce(W, acc[r][c]);
        }
        while (p < k) : (p += 1) {
            inline for (0..rows) |r| {
                const av = fam.elemA(ad, r * k + p);
                inline for (0..4) |c| sums[r][c] = fam.tail(sums[r][c], av, fam.elemB(bd, (j + c) * k + p));
            }
        }

        inline for (0..rows) |r| {
            inline for (0..4) |c| cd[r * n + j + c] = sums[r][c];
        }
    }

    while (j < col_end) : (j += 1) {
        var acc: [rows]V = undefined;
        inline for (0..rows) |r| acc[r] = @splat(0);

        var p: usize = 0;
        while (p + W <= k) : (p += W) {
            const bv = fam.vecB(W, bd, j * k + p);
            inline for (0..rows) |r| acc[r] = @mulAdd(V, fam.vecA(W, ad, r * k + p), bv, acc[r]);
        }

        var sums: [rows]f32 = undefined;
        inline for (0..rows) |r| sums[r] = fam.reduce(W, acc[r]);
        while (p < k) : (p += 1) {
            const bv = fam.elemB(bd, j * k + p);
            inline for (0..rows) |r| sums[r] = fam.tail(sums[r], fam.elemA(ad, r * k + p), bv);
        }
        inline for (0..rows) |r| cd[r * n + j] = sums[r];
    }
}

/// Four B rows against one A row, fused (`out[c] = A · B[b_row + c]`):
/// `dotChains` independent accumulator chains per column over the family's
/// dot width, the chains summed, the lanes reduced, then the scalar tail
/// fused in; the order is fixed per element regardless of the caller's
/// column split.
inline fn dot4(comptime fam: NtFamily, out: []f32, a: []const fam.A(), b: []const fam.B(), b_row: usize, k: usize) void {
    const W = comptime fam.dotWidth();
    const chains = comptime fam.dotChains();
    const V = @Vector(W, fam.Acc());
    var acc: [4][chains]V = undefined;
    inline for (0..4) |c| {
        inline for (0..chains) |h| acc[c][h] = @splat(0);
    }

    var p: usize = 0;
    while (p + chains * W <= k) : (p += chains * W) {
        var av: [chains]V = undefined;
        inline for (0..chains) |h| av[h] = fam.vecA(W, a, p + h * W);
        inline for (0..4) |c| {
            inline for (0..chains) |h| acc[c][h] = @mulAdd(V, av[h], fam.vecB(W, b, (b_row + c) * k + p + h * W), acc[c][h]);
        }
    }
    if (comptime chains > 1) {
        while (p + W <= k) : (p += W) {
            const av = fam.vecA(W, a, p);
            inline for (0..4) |c| acc[c][0] = @mulAdd(V, av, fam.vecB(W, b, (b_row + c) * k + p), acc[c][0]);
        }
    }

    var s: [4]f32 = undefined;
    inline for (0..4) |c| {
        var lanes = acc[c][0];
        inline for (1..chains) |h| lanes += acc[c][h];
        s[c] = fam.reduce(W, lanes);
    }
    while (p < k) : (p += 1) {
        const av = fam.elemA(a, p);
        inline for (0..4) |c| s[c] = fam.tail(s[c], av, fam.elemB(b, (b_row + c) * k + p));
    }
    inline for (0..4) |c| out[c] = s[c];
}

/// One C row over columns [col_start, col_end): `dot4` per 4-column
/// block, the family's dot for the leftover columns.
inline fn gemmNTRow(comptime fam: NtFamily, c_row: []f32, a_row: []const fam.A(), bd: []const fam.B(), k: usize, col_start: usize, col_end: usize) void {
    var j = col_start;
    while (j + 4 <= col_end) : (j += 4) dot4(fam, c_row[j .. j + 4], a_row, bd, j, k);
    while (j < col_end) : (j += 1) c_row[j] = fam.vecDot(a_row, bd[j * k .. (j + 1) * k]);
}

/// C[0..m, col_start..col_end) = A[m, k] · Bᵀ as 6/4/3/2/1 row blocks
/// (a 4+3 split preferred over 6+1: the single-row tail measured slower),
/// each block streaming every RHS column once; single rows through
/// `gemmNTRow`.
fn gemmNTRowsCols(comptime fam: NtFamily, cd: []f32, ad: []const fam.A(), bd: []const fam.B(), m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (zeroIfEmpty(.store, cd, n, k, 0, m, col_start, col_end)) return;
    // The 6/4/3/2 blocks unroll their rows x 4 inline-for nests into this
    // scope, past the default 1000-branch quota.
    @setEvalBranchQuota(10_000);
    var i: usize = 0;
    // Avoid a 6+1 split; the scalar row tail is slower than 4+3 here.
    while (i + 6 <= m and m - (i + 6) != 1) : (i += 6) {
        gemmNTRowsBlock(6, fam, cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
    }
    while (i + 4 <= m) : (i += 4) {
        gemmNTRowsBlock(4, fam, cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
    }
    if (i + 3 <= m) {
        gemmNTRowsBlock(3, fam, cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
        i += 3;
    }
    if (i + 2 <= m) {
        gemmNTRowsBlock(2, fam, cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
        i += 2;
    }
    while (i < m) : (i += 1) {
        gemmNTRow(fam, cd[i * n .. (i + 1) * n], ad[i * k .. (i + 1) * k], bd, k, col_start, col_end);
    }
}

pub fn gemmNTRange(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    _ = m;
    if (zeroIfEmpty(.store, cd, n, k, row_start, row_end, 0, n)) return;
    for (row_start..row_end) |i| {
        gemmNTRow(.f32, cd[i * n .. (i + 1) * n], ad[i * k .. (i + 1) * k], bd, k, 0, n);
    }
}

fn gemmNTCols(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (zeroIfEmpty(.store, cd, n, k, 0, m, col_start, col_end)) return;
    // Column tile OUTER, rows inner: the 4-row B tile stays cache-hot across
    // all m A rows, so B streams once total instead of once per row (at m=8
    // the row-outer order cost 8x the memory traffic — the whole RHS weight
    // re-read per row). Same dot4/vecDot per output element in the same
    // order per element: bitwise-identical results, scheduling-only change.
    var j = col_start;
    while (j + 4 <= col_end) : (j += 4) {
        for (0..m) |i| {
            dot4(.f32, cd[i * n + j .. i * n + j + 4], ad[i * k .. (i + 1) * k], bd, j, k);
        }
    }
    while (j < col_end) : (j += 1) {
        for (0..m) |i| {
            cd[i * n + j] = primitives.vecDot(ad[i * k .. (i + 1) * k], bd[j * k .. (j + 1) * k]);
        }
    }
}

// The row-range forms of the NT f16/bf16 RHS kernels: the cols walk over
// the row slice.
fn gemmNTF16RhsRange(cd: []f32, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    _ = m;
    gemmNTRowsCols(f16_rhs_family, cd[row_start * n ..], ad[row_start * k ..], bd, row_end - row_start, n, k, 0, n);
}

fn gemmNTF16RhsCols(cd: []f32, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    gemmNTRowsCols(f16_rhs_family, cd, ad, bd, m, n, k, col_start, col_end);
}

fn gemmNTBf16RhsRange(cd: []f32, ad: []const f32, bd: []const u16, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    _ = m;
    gemmNTRowsCols(.bf16, cd[row_start * n ..], ad[row_start * k ..], bd, row_end - row_start, n, k, 0, n);
}

/// C[m, col_start..col_end] = A[m, k] · Bᵀ where B rows are bf16 and stay
/// bf16 in-register (widened per SIMD lane, never materialized as f32).
fn gemmNTBf16RhsCols(cd: []f32, ad: []const f32, bd: []const u16, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    gemmNTRowsCols(.bf16, cd, ad, bd, m, n, k, col_start, col_end);
}

/// The f16-lane single-column dot (`.f16_native`): four fmla.8h chains,
/// summed and reduced to f32, then the scalar tail.
inline fn vecDotF16HalfAccumToF32(x: []const f16, y: []const f16) f32 {
    var i: usize = 0;
    var acc0: Vf16 = @splat(0);
    var acc1: Vf16 = @splat(0);
    var acc2: Vf16 = @splat(0);
    var acc3: Vf16 = @splat(0);

    while (i + 4 * vector_len_f16 <= x.len) : (i += 4 * vector_len_f16) {
        const x0: Vf16 = x[i..][0..vector_len_f16].*;
        const y0: Vf16 = y[i..][0..vector_len_f16].*;
        const x1: Vf16 = x[i + vector_len_f16 ..][0..vector_len_f16].*;
        const y1: Vf16 = y[i + vector_len_f16 ..][0..vector_len_f16].*;
        const x2: Vf16 = x[i + 2 * vector_len_f16 ..][0..vector_len_f16].*;
        const y2: Vf16 = y[i + 2 * vector_len_f16 ..][0..vector_len_f16].*;
        const x3: Vf16 = x[i + 3 * vector_len_f16 ..][0..vector_len_f16].*;
        const y3: Vf16 = y[i + 3 * vector_len_f16 ..][0..vector_len_f16].*;
        acc0 = @mulAdd(Vf16, x0, y0, acc0);
        acc1 = @mulAdd(Vf16, x1, y1, acc1);
        acc2 = @mulAdd(Vf16, x2, y2, acc2);
        acc3 = @mulAdd(Vf16, x3, y3, acc3);
    }
    while (i + vector_len_f16 <= x.len) : (i += vector_len_f16) {
        const xv: Vf16 = x[i..][0..vector_len_f16].*;
        const yv: Vf16 = y[i..][0..vector_len_f16].*;
        acc0 = @mulAdd(Vf16, xv, yv, acc0);
    }

    var sum: f32 = @floatCast(@reduce(.Add, acc0 + acc1 + acc2 + acc3));
    while (i < x.len) : (i += 1) {
        sum += @as(f32, @floatCast(x[i])) * @as(f32, @floatCast(y[i]));
    }
    return sum;
}

/// The bf16-RHS single-column dot: four wide (f16-vector-width) f32 chains
/// over shift-widened bf16 loads, summed and reduced, then the scalar tail.
inline fn vecDotBf16RhsToF32(x: []const f32, y: []const u16) f32 {
    var i: usize = 0;
    var acc0: Vf32Wide = @splat(0);
    var acc1: Vf32Wide = @splat(0);
    var acc2: Vf32Wide = @splat(0);
    var acc3: Vf32Wide = @splat(0);

    while (i + 4 * vector_len_f16 <= x.len) : (i += 4 * vector_len_f16) {
        const x0: Vf32Wide = x[i..][0..vector_len_f16].*;
        const y0 = primitives.bf16VecToF32Wide(y[i..][0..vector_len_f16].*);
        const x1: Vf32Wide = x[i + vector_len_f16 ..][0..vector_len_f16].*;
        const y1 = primitives.bf16VecToF32Wide(y[i + vector_len_f16 ..][0..vector_len_f16].*);
        const x2: Vf32Wide = x[i + 2 * vector_len_f16 ..][0..vector_len_f16].*;
        const y2 = primitives.bf16VecToF32Wide(y[i + 2 * vector_len_f16 ..][0..vector_len_f16].*);
        const x3: Vf32Wide = x[i + 3 * vector_len_f16 ..][0..vector_len_f16].*;
        const y3 = primitives.bf16VecToF32Wide(y[i + 3 * vector_len_f16 ..][0..vector_len_f16].*);
        acc0 = @mulAdd(Vf32Wide, x0, y0, acc0);
        acc1 = @mulAdd(Vf32Wide, x1, y1, acc1);
        acc2 = @mulAdd(Vf32Wide, x2, y2, acc2);
        acc3 = @mulAdd(Vf32Wide, x3, y3, acc3);
    }
    while (i + vector_len_f16 <= x.len) : (i += vector_len_f16) {
        const xv: Vf32Wide = x[i..][0..vector_len_f16].*;
        const yv = primitives.bf16VecToF32Wide(y[i..][0..vector_len_f16].*);
        acc0 = @mulAdd(Vf32Wide, xv, yv, acc0);
    }

    var sum: f32 = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    while (i < x.len) : (i += 1) {
        sum += x[i] * dtype_mod.bf16ToF32(y[i]);
    }
    return sum;
}

// ---------------- f64 and the typed scalar fallback ----------------

fn gemmNNRangeF64(cd: []f64, ad: []const f64, bd: []const f64, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    _ = m;
    for (row_start..row_end) |i| {
        var j: usize = 0;
        while (j + common.vector_len_f64 <= n) : (j += common.vector_len_f64) {
            var acc: common.Vf64 = @splat(0);
            for (0..k) |p| {
                acc += @as(common.Vf64, @splat(ad[i * k + p])) * @as(common.Vf64, bd[p * n + j ..][0..common.vector_len_f64].*);
            }
            cd[i * n + j ..][0..common.vector_len_f64].* = acc;
        }
        while (j < n) : (j += 1) {
            var acc: f64 = 0;
            for (0..k) |p| acc += ad[i * k + p] * bd[p * n + j];
            cd[i * n + j] = acc;
        }
    }
}

fn matmul2DIntoTypedScalar(
    comptime dtype: DType,
    out: []dtype_mod.Scalar(dtype_mod.outputDType(.matmul, dtype)),
    a: []const dtype_mod.Scalar(dtype),
    b: []const dtype_mod.Scalar(dtype),
    m: usize,
    n: usize,
    k: usize,
) void {
    const compute_dtype = comptime dtype_mod.computeDType(.matmul, dtype);
    const output_dtype = comptime dtype_mod.outputDType(.matmul, dtype);
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: dtype_mod.Scalar(compute_dtype) = 0;
            for (0..k) |p| {
                acc += dtype_mod.castFloat(dtype, compute_dtype, a[i * k + p]) * dtype_mod.castFloat(dtype, compute_dtype, b[p * n + j]);
            }
            out[i * n + j] = dtype_mod.castFloat(compute_dtype, output_dtype, acc);
        }
    }
}

test {
    _ = @import("gemm_tests.zig");
}

// ---------------- The scalar reference arm ----------------

/// The scalar reference twin of the dense GEMM entry: the request as one
/// serial triple loop. The orientation selects the index formulas, the
/// operands widen to the matmul compute dtype (f32 for any mixed pair), and
/// the accumulator narrows once on store. On `-Dbackend=scalar` builds the
/// `gemm` entry above dispatches here; on native builds the twin stays
/// reachable for `backend/parity_test.zig` and `bench/backend.zig`.
pub const scalar = struct {
    pub fn gemm(
        comptime g: ops.Gemm,
        cd: []dtype_mod.Scalar(g.out),
        ad: []const dtype_mod.Scalar(g.a),
        bd: []const dtype_mod.Scalar(g.b),
        m: usize,
        n: usize,
        k: usize,
    ) void {
        const compute = comptime if (g.a == g.b) dtype_mod.computeDType(.matmul, g.a) else .f32;
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: dtype_mod.Scalar(compute) = 0;
                for (0..k) |p| {
                    const av = switch (g.kind) {
                        .plain, .trans_b => ad[i * k + p],
                        .trans_a => ad[p * m + i],
                    };
                    const bv = switch (g.kind) {
                        .plain, .trans_a => bd[p * n + j],
                        .trans_b => bd[j * k + p],
                    };
                    acc += dtype_mod.castFloat(g.a, compute, av) * dtype_mod.castFloat(g.b, compute, bv);
                }
                const value = dtype_mod.castFloat(compute, g.out, acc);
                if (g.accumulate) cd[i * n + j] += value else cd[i * n + j] = value;
            }
        }
    }
};
