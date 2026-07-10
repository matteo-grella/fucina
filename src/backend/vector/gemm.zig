//! Dense f32/f16/f64/bf16 GEMM relocated out of vector.zig: the NN/TN/NT and
//! f16-RHS entry points, their Task structs and parallel dispatch, and the inner
//! range/cols/row kernels. Shared-core symbols (ParallelConfig, the
//! contiguous-data helpers, matmulThreadCount/columnThreadCount, the V* width
//! aliases) and the @Vector primitives are aliased from vector.zig (`vm`) so the
//! moved bodies compile unchanged. gemmNNRange/gemmTNRange/gemmNTRange are pub so
//! the batched GEMM module can reuse them.

const std = @import("std");
const builtin = @import("builtin");
const dtype_mod = @import("../../dtype.zig");
const gemm_blocked = @import("gemm_blocked.zig");
const parallel = @import("../../parallel.zig");
const tensor = @import("../../tensor.zig");
const thread = @import("../../thread.zig");
const vm = @import("common.zig");
const primitives = @import("primitives.zig");

const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

// Shared-core symbols from the common leaf, aliased so the moved bodies compile
// unchanged.
const ParallelConfig = vm.ParallelConfig;
const matmulThreadCount = vm.matmulThreadCount;
const columnThreadCount = vm.columnThreadCount;
const contiguousDataConst = vm.contiguousDataConst;
const contiguousData = vm.contiguousData;
const contiguousDataConstOf = vm.contiguousDataConstOf;
const contiguousDataOf = vm.contiguousDataOf;

const vector_len = vm.vector_len;
const vector_len_f64 = vm.vector_len_f64;
const vector_len_f16 = vm.vector_len_f16;
const Vf32 = vm.Vf32;
const Vf64 = vm.Vf64;
const Vf16 = vm.Vf16;
const Vf16ForF32 = vm.Vf16ForF32;

// @Vector primitives — imported directly from the primitives sibling.
const vecDot = primitives.vecDot;
const vecDotF16ToF32 = primitives.vecDotF16ToF32;
const bf16VecToF32 = primitives.bf16VecToF32;
const f32VecToBf16 = primitives.f32VecToBf16;

// f16-RHS GEMM accumulator policy. On aarch64 NEON the f16 x f16 @mulAdd arms
// are native fmla.8h (double the f32 lane throughput), so half-precision
// accumulation is the fast path there. Every other ISA — x86-64 without
// AVX512-FP16 in particular — legalizes f16 vector arithmetic by promoting
// through f32 and rounding back PER OPERATION (vcvtph2ps/vcvtps2ph around
// every fmadd, scalarized reductions, accumulator spills), so those targets
// take the *Wide twins below: widen each f16 load once (F16C vcvtph2ps) and
// accumulate in f32, mirroring the bf16-RHS kernels. This changes non-aarch64
// results (one final rounding instead of per-step f16 rounding — strictly more
// accurate); aarch64 output is bit-identical to before.
const f16_accum_native = builtin.cpu.arch.isAARCH64();

// ---------------- MatMul (2-D) ----------------

pub fn matmulInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    const av = try a.rankView(2);
    const bv = try b.rankView(2);
    const ov = try out.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    const n = bv.dim(1);
    if (k != bv.dim(0)) return tensor.TensorError.ShapeMismatch;
    if (ov.dim(0) != m or ov.dim(1) != n) return tensor.TensorError.ShapeMismatch;
    matmul2DIntoUnchecked(out, a, b, m, n, k);
}

// C[i, j] = sum_p A[i, p] * B[p, j]. The natural inner order (i, j, p) reads
// B strided in p, which kills vectorization. Reorder to (i, p, j): broadcast
// A[i, p] as a scalar, multiply by a contiguous slice of B's row p starting at
// j, and accumulate into C's row i starting at j. Now the inner loop is two
// contiguous reads and one contiguous write — vectorizes cleanly.
pub fn matmul2DIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, m: usize, n: usize, k: usize) void {
    matmul2DIntoUncheckedWithConfig(out, a, b, m, n, k, .{});
}

pub fn matmul2DIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const ad = contiguousDataConst(a, m * k);
    const bd = contiguousDataConst(b, k * n);
    const cd = contiguousData(out, m * n);
    if (gemm_blocked.shouldUseBlocked(m, n, k)) {
        return gemm_blocked.gemmBlocked(.nn, cd, ad, bd, m, n, k, config);
    }
    gemmNNRowPathWithConfig(cd, ad, bd, m, n, k, config);
}

// The pre-blocking register-tiled row-kernel path, bypassing the blocked
// dispatch above. Public so the GEMM bench can baseline it directly.
pub fn gemmNNRowPathWithConfig(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    if (maybeParallelNN(config, cd, ad, bd, m, n, k)) return;
    gemmNNRange(cd, ad, bd, m, n, k, 0, m);
}

pub fn matmul2DIntoUncheckedTypedWithConfig(
    comptime dtype: DType,
    out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
    a: *const tensor.TensorOf(dtype),
    b: *const tensor.TensorOf(dtype),
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const ad = contiguousDataConstOf(dtype, a, m * k);
    const bd = contiguousDataConstOf(dtype, b, k * n);
    const cd = contiguousDataOf(dtype_mod.outputDType(.matmul, dtype), out, m * n);
    if (comptime dtype == .f64) {
        if (maybeParallelNNF64(config, cd, ad, bd, m, n, k)) return;
        return gemmNNRangeF64(cd, ad, bd, m, n, k, 0, m);
    } else if (comptime dtype == .f16) {
        if (maybeParallelNNF16(config, cd, ad, bd, m, n, k)) return;
        return gemmNNRangeF16(cd, ad, bd, m, n, k, 0, m);
    } else if (comptime dtype == .bf16) {
        if (maybeParallelNNBf16(config, cd, ad, bd, m, n, k)) return;
        return gemmNNRangeBf16(cd, ad, bd, m, n, k, 0, m);
    }
    matmul2DIntoTypedScalar(dtype, cd, ad, bd, m, n, k);
}

pub fn matmulTransAInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    const av = try a.rankView(2);
    const bv = try b.rankView(2);
    const ov = try out.rankView(2);
    const k = av.dim(0);
    const m = av.dim(1);
    const n = bv.dim(1);
    if (k != bv.dim(0)) return tensor.TensorError.ShapeMismatch;
    if (ov.dim(0) != m or ov.dim(1) != n) return tensor.TensorError.ShapeMismatch;
    matmulTransA2DIntoUnchecked(out, a, b, m, n, k);
}

// C[i, j] = sum_p A[p, i] * B[p, j], with A logically [k, m]. Reorder to
// (p, i, j): for each p and i, broadcast A[p, i] and FMA into C's row i with
// B's row p. Same contiguous-stream pattern as matmul2D, vectorizes in j.
pub fn matmulTransA2DIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, m: usize, n: usize, k: usize) void {
    matmulTransA2DIntoUncheckedWithConfig(out, a, b, m, n, k, .{});
}

pub fn matmulTransA2DIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const ad = contiguousDataConst(a, k * m);
    const bd = contiguousDataConst(b, k * n);
    const cd = contiguousData(out, m * n);
    if (gemm_blocked.shouldUseBlocked(m, n, k)) {
        return gemm_blocked.gemmBlocked(.tn, cd, ad, bd, m, n, k, config);
    }
    gemmTNRowPathWithConfig(cd, ad, bd, m, n, k, config);
}

// See gemmNNRowPathWithConfig.
pub fn gemmTNRowPathWithConfig(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    if (maybeParallelTN(config, cd, ad, bd, m, n, k)) return;
    gemmTNRange(cd, ad, bd, m, n, k, 0, m);
}

pub fn matmulTransBInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
    const av = try a.rankView(2);
    const bv = try b.rankView(2);
    const ov = try out.rankView(2);
    const m = av.dim(0);
    const k = av.dim(1);
    const n = bv.dim(0);
    if (k != bv.dim(1)) return tensor.TensorError.ShapeMismatch;
    if (ov.dim(0) != m or ov.dim(1) != n) return tensor.TensorError.ShapeMismatch;
    matmulTransB2DIntoUnchecked(out, a, b, m, n, k);
}

// C[i, j] = sum_p A[i, p] * B[j, p], with B logically [n, k]. Both A's row i
// and B's row j are contiguous in p — this is a textbook dot-product per
// output element. The straightforward (i, j, p) ordering is already optimal
// since each inner reduction can SIMD-accumulate two contiguous streams.
pub fn matmulTransB2DIntoUnchecked(out: *Tensor, a: *const Tensor, b: *const Tensor, m: usize, n: usize, k: usize) void {
    matmulTransB2DIntoUncheckedWithConfig(out, a, b, m, n, k, .{});
}

pub fn matmulTransB2DIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const ad = contiguousDataConst(a, m * k);
    const bd = contiguousDataConst(b, n * k);
    const cd = contiguousData(out, m * n);
    if (gemm_blocked.shouldUseBlocked(m, n, k)) {
        return gemm_blocked.gemmBlocked(.nt, cd, ad, bd, m, n, k, config);
    }
    gemmNTRowPathWithConfig(cd, ad, bd, m, n, k, config);
}

// See gemmNNRowPathWithConfig.
pub fn gemmNTRowPathWithConfig(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, config: ParallelConfig) void {
    if (maybeParallelNT(config, cd, ad, bd, m, n, k)) return;
    gemmNTRange(cd, ad, bd, m, n, k, 0, m);
}

pub fn matmulTransB2DIntoUncheckedF16OperandsWithConfig(
    out: *Tensor,
    a: *const tensor.TensorOf(.f16),
    b: *const tensor.TensorOf(.f16),
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const ad = contiguousDataConstOf(.f16, a, m * k);
    const bd = contiguousDataConstOf(.f16, b, n * k);
    const cd = contiguousData(out, m * n);
    if (maybeParallelNTF16Rhs(config, cd, ad, bd, m, n, k)) return;
    gemmNTF16RhsRange(cd, ad, bd, m, n, k, 0, m);
}

// Mixed-precision NT GEMM: f32 LHS activations against a frozen bf16 RHS
// stored [n, k]. Unlike the f16 twin (which casts the LHS to f16 and
// accumulates in half precision), the bf16 weights are widened to f32
// in-register (u16 << 16 bit shift, exact) and everything accumulates in f32 —
// no f32 materialization of the weight matrix, no LHS precision loss.
pub fn matmulTransB2DIntoUncheckedBf16RhsWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const tensor.TensorOf(.bf16),
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    const ad = contiguousDataConst(a, m * k);
    const bd = contiguousDataConstOf(.bf16, b, n * k);
    const cd = contiguousData(out, m * n);
    if (maybeParallelNTBf16Rhs(config, cd, ad, bd, m, n, k)) return;
    gemmNTBf16RhsRange(cd, ad, bd, m, n, k, 0, m);
}

// ---------------- Inner kernels ----------------

const GemmTask = struct {
    cd: []f32,
    ad: []const f32,
    bd: []const f32,
    m: usize,
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
};

const GemmTaskF64 = struct {
    cd: []f64,
    ad: []const f64,
    bd: []const f64,
    m: usize,
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
};

const GemmTaskF16 = struct {
    cd: []f16,
    ad: []const f16,
    bd: []const f16,
    m: usize,
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
};

const GemmTaskF16Rhs = struct {
    cd: []f32,
    ad: []const f16,
    bd: []const f16,
    m: usize,
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
};

const GemmTaskBf16 = struct {
    cd: []u16,
    ad: []const u16,
    bd: []const u16,
    m: usize,
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
};

const ColTask = struct {
    cd: []f32,
    ad: []const f32,
    bd: []const f32,
    m: usize,
    n: usize,
    k: usize,
    col_start: usize,
    col_end: usize,
};

const ColTaskF16Rhs = struct {
    cd: []f32,
    ad: []const f16,
    bd: []const f16,
    m: usize,
    n: usize,
    k: usize,
    col_start: usize,
    col_end: usize,
};

const GemmTaskBf16Rhs = struct {
    cd: []f32,
    ad: []const f32,
    bd: []const u16,
    m: usize,
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
};

const ColTaskBf16Rhs = struct {
    cd: []f32,
    ad: []const f32,
    bd: []const u16,
    m: usize,
    n: usize,
    k: usize,
    col_start: usize,
    col_end: usize,
};

const ColTaskF16 = struct {
    cd: []f16,
    ad: []const f16,
    bd: []const f16,
    m: usize,
    n: usize,
    k: usize,
    col_start: usize,
    col_end: usize,
};

fn maybeParallelNN(config: ParallelConfig, cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize) bool {
    const pool = config.pool orelse return false;
    if (m < parallel.vector_column_min_m) {
        const thread_count = columnThreadCount(m, n, k);
        if (thread_count != 1) {
            runParallelCols(pool, runGemmNNColTask, cd, ad, bd, m, n, k, thread_count);
            return true;
        }
    }
    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    runParallelRows(pool, runGemmNNTask, cd, ad, bd, m, n, k, thread_count);
    return true;
}

fn maybeParallelNNF64(config: ParallelConfig, cd: []f64, ad: []const f64, bd: []const f64, m: usize, n: usize, k: usize) bool {
    const pool = config.pool orelse return false;
    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    runParallelRowsF64(pool, cd, ad, bd, m, n, k, thread_count);
    return true;
}

fn maybeParallelNNF16(config: ParallelConfig, cd: []f16, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize) bool {
    const pool = config.pool orelse return false;
    if (m < parallel.vector_column_min_m) {
        const thread_count = columnThreadCount(m, n, k);
        if (thread_count != 1) {
            runParallelColsF16(pool, cd, ad, bd, m, n, k, thread_count);
            return true;
        }
    }
    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    runParallelRowsF16(pool, cd, ad, bd, m, n, k, thread_count);
    return true;
}

fn maybeParallelNNBf16(config: ParallelConfig, cd: []u16, ad: []const u16, bd: []const u16, m: usize, n: usize, k: usize) bool {
    const pool = config.pool orelse return false;
    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    runParallelRowsBf16(pool, cd, ad, bd, m, n, k, thread_count);
    return true;
}

fn maybeParallelTN(config: ParallelConfig, cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize) bool {
    const pool = config.pool orelse return false;
    if (m < parallel.vector_column_min_m) {
        const thread_count = columnThreadCount(m, n, k);
        if (thread_count != 1) {
            runParallelCols(pool, runGemmTNColTask, cd, ad, bd, m, n, k, thread_count);
            return true;
        }
    }
    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    runParallelRows(pool, runGemmTNTask, cd, ad, bd, m, n, k, thread_count);
    return true;
}

fn maybeParallelNT(config: ParallelConfig, cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize) bool {
    const pool = config.pool orelse return false;
    if (m < parallel.vector_column_min_m) {
        const thread_count = columnThreadCount(m, n, k);
        if (thread_count != 1) {
            runParallelCols(pool, runGemmNTColTask, cd, ad, bd, m, n, k, thread_count);
            return true;
        }
    }
    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    runParallelRows(pool, runGemmNTTask, cd, ad, bd, m, n, k, thread_count);
    return true;
}

fn maybeParallelNTF16Rhs(config: ParallelConfig, cd: []f32, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize) bool {
    const pool = config.pool orelse return false;
    if (m < parallel.vector_column_min_m) {
        const thread_count = columnThreadCount(m, n, k);
        if (thread_count != 1) {
            runParallelColsF16Rhs(pool, cd, ad, bd, m, n, k, thread_count);
            return true;
        }
    }
    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    runParallelRowsF16Rhs(pool, cd, ad, bd, m, n, k, thread_count);
    return true;
}

fn maybeParallelNTBf16Rhs(config: ParallelConfig, cd: []f32, ad: []const f32, bd: []const u16, m: usize, n: usize, k: usize) bool {
    const pool = config.pool orelse return false;
    if (m < parallel.vector_column_min_m) {
        const thread_count = columnThreadCount(m, n, k);
        if (thread_count != 1) {
            runParallelColsBf16Rhs(pool, cd, ad, bd, m, n, k, thread_count);
            return true;
        }
    }
    const thread_count = matmulThreadCount(m, n, k, parallel.vector_matmul_work_threshold);
    if (thread_count == 1) return false;
    runParallelRowsBf16Rhs(pool, cd, ad, bd, m, n, k, thread_count);
    return true;
}

fn runParallelRows(
    pool: *thread.Pool,
    comptime func: fn (*const GemmTask) void,
    cd: []f32,
    ad: []const f32,
    bd: []const f32,
    m: usize,
    n: usize,
    k: usize,
    thread_count: usize,
) void {
    var tasks: [parallel.vector_max_threads]GemmTask = undefined;

    for (0..thread_count) |ti| {
        const start = ti * m / thread_count;
        const end = (ti + 1) * m / thread_count;
        tasks[ti] = .{
            .cd = cd,
            .ad = ad,
            .bd = bd,
            .m = m,
            .n = n,
            .k = k,
            .row_start = start,
            .row_end = end,
        };
    }

    pool.parallelChunks(GemmTask, tasks[0..thread_count], func);
}

fn runParallelRowsF64(
    pool: *thread.Pool,
    cd: []f64,
    ad: []const f64,
    bd: []const f64,
    m: usize,
    n: usize,
    k: usize,
    thread_count: usize,
) void {
    var tasks: [parallel.vector_max_threads]GemmTaskF64 = undefined;
    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .cd = cd,
            .ad = ad,
            .bd = bd,
            .m = m,
            .n = n,
            .k = k,
            .row_start = ti * m / thread_count,
            .row_end = (ti + 1) * m / thread_count,
        };
    }

    pool.parallelChunks(GemmTaskF64, tasks[0..thread_count], runGemmNNF64Task);
}

fn runParallelRowsF16(
    pool: *thread.Pool,
    cd: []f16,
    ad: []const f16,
    bd: []const f16,
    m: usize,
    n: usize,
    k: usize,
    thread_count: usize,
) void {
    var tasks: [parallel.vector_max_threads]GemmTaskF16 = undefined;
    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .cd = cd,
            .ad = ad,
            .bd = bd,
            .m = m,
            .n = n,
            .k = k,
            .row_start = ti * m / thread_count,
            .row_end = (ti + 1) * m / thread_count,
        };
    }

    pool.parallelChunks(GemmTaskF16, tasks[0..thread_count], runGemmNNF16Task);
}

fn runParallelRowsF16Rhs(
    pool: *thread.Pool,
    cd: []f32,
    ad: []const f16,
    bd: []const f16,
    m: usize,
    n: usize,
    k: usize,
    thread_count: usize,
) void {
    var tasks: [parallel.vector_max_threads]GemmTaskF16Rhs = undefined;
    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .cd = cd,
            .ad = ad,
            .bd = bd,
            .m = m,
            .n = n,
            .k = k,
            .row_start = ti * m / thread_count,
            .row_end = (ti + 1) * m / thread_count,
        };
    }

    pool.parallelChunks(GemmTaskF16Rhs, tasks[0..thread_count], runGemmNTF16RhsTask);
}

fn runParallelRowsBf16Rhs(
    pool: *thread.Pool,
    cd: []f32,
    ad: []const f32,
    bd: []const u16,
    m: usize,
    n: usize,
    k: usize,
    thread_count: usize,
) void {
    var tasks: [parallel.vector_max_threads]GemmTaskBf16Rhs = undefined;
    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .cd = cd,
            .ad = ad,
            .bd = bd,
            .m = m,
            .n = n,
            .k = k,
            .row_start = ti * m / thread_count,
            .row_end = (ti + 1) * m / thread_count,
        };
    }

    pool.parallelChunks(GemmTaskBf16Rhs, tasks[0..thread_count], runGemmNTBf16RhsTask);
}

fn runParallelRowsBf16(
    pool: *thread.Pool,
    cd: []u16,
    ad: []const u16,
    bd: []const u16,
    m: usize,
    n: usize,
    k: usize,
    thread_count: usize,
) void {
    var tasks: [parallel.vector_max_threads]GemmTaskBf16 = undefined;
    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .cd = cd,
            .ad = ad,
            .bd = bd,
            .m = m,
            .n = n,
            .k = k,
            .row_start = ti * m / thread_count,
            .row_end = (ti + 1) * m / thread_count,
        };
    }

    pool.parallelChunks(GemmTaskBf16, tasks[0..thread_count], runGemmNNBf16Task);
}

fn runParallelCols(
    pool: *thread.Pool,
    comptime func: fn (*const ColTask) void,
    cd: []f32,
    ad: []const f32,
    bd: []const f32,
    m: usize,
    n: usize,
    k: usize,
    thread_count: usize,
) void {
    var tasks: [parallel.vector_max_threads]ColTask = undefined;

    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .cd = cd,
            .ad = ad,
            .bd = bd,
            .m = m,
            .n = n,
            .k = k,
            .col_start = ti * n / thread_count,
            .col_end = (ti + 1) * n / thread_count,
        };
    }
    pool.parallelChunks(ColTask, tasks[0..thread_count], func);
}

fn runParallelColsF16Rhs(
    pool: *thread.Pool,
    cd: []f32,
    ad: []const f16,
    bd: []const f16,
    m: usize,
    n: usize,
    k: usize,
    thread_count: usize,
) void {
    var tasks: [parallel.vector_max_threads]ColTaskF16Rhs = undefined;

    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .cd = cd,
            .ad = ad,
            .bd = bd,
            .m = m,
            .n = n,
            .k = k,
            .col_start = ti * n / thread_count,
            .col_end = (ti + 1) * n / thread_count,
        };
    }
    pool.parallelChunks(ColTaskF16Rhs, tasks[0..thread_count], runGemmNTF16RhsColTask);
}

fn runParallelColsBf16Rhs(
    pool: *thread.Pool,
    cd: []f32,
    ad: []const f32,
    bd: []const u16,
    m: usize,
    n: usize,
    k: usize,
    thread_count: usize,
) void {
    var tasks: [parallel.vector_max_threads]ColTaskBf16Rhs = undefined;

    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .cd = cd,
            .ad = ad,
            .bd = bd,
            .m = m,
            .n = n,
            .k = k,
            .col_start = ti * n / thread_count,
            .col_end = (ti + 1) * n / thread_count,
        };
    }
    pool.parallelChunks(ColTaskBf16Rhs, tasks[0..thread_count], runGemmNTBf16RhsColTask);
}

fn runParallelColsF16(
    pool: *thread.Pool,
    cd: []f16,
    ad: []const f16,
    bd: []const f16,
    m: usize,
    n: usize,
    k: usize,
    thread_count: usize,
) void {
    var tasks: [parallel.vector_max_threads]ColTaskF16 = undefined;

    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .cd = cd,
            .ad = ad,
            .bd = bd,
            .m = m,
            .n = n,
            .k = k,
            .col_start = ti * n / thread_count,
            .col_end = (ti + 1) * n / thread_count,
        };
    }
    pool.parallelChunks(ColTaskF16, tasks[0..thread_count], runGemmNNColTaskF16);
}

fn runGemmNNTask(task: *const GemmTask) void {
    gemmNNRange(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.row_start, task.row_end);
}

fn runGemmNNF64Task(task: *const GemmTaskF64) void {
    gemmNNRangeF64(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.row_start, task.row_end);
}

fn runGemmNNF16Task(task: *const GemmTaskF16) void {
    gemmNNRangeF16(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.row_start, task.row_end);
}

fn runGemmNNBf16Task(task: *const GemmTaskBf16) void {
    gemmNNRangeBf16(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.row_start, task.row_end);
}

fn runGemmTNTask(task: *const GemmTask) void {
    gemmTNRange(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.row_start, task.row_end);
}

fn runGemmNTTask(task: *const GemmTask) void {
    gemmNTRange(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.row_start, task.row_end);
}

fn runGemmNTF16RhsTask(task: *const GemmTaskF16Rhs) void {
    gemmNTF16RhsRange(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.row_start, task.row_end);
}

fn runGemmNTBf16RhsTask(task: *const GemmTaskBf16Rhs) void {
    gemmNTBf16RhsRange(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.row_start, task.row_end);
}

fn runGemmNNColTask(task: *const ColTask) void {
    gemmNNCols(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.col_start, task.col_end);
}

fn runGemmTNColTask(task: *const ColTask) void {
    gemmTNCols(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.col_start, task.col_end);
}

fn runGemmNTColTask(task: *const ColTask) void {
    gemmNTCols(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.col_start, task.col_end);
}

fn runGemmNTF16RhsColTask(task: *const ColTaskF16Rhs) void {
    gemmNTF16RhsCols(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.col_start, task.col_end);
}

fn runGemmNTBf16RhsColTask(task: *const ColTaskBf16Rhs) void {
    gemmNTBf16RhsCols(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.col_start, task.col_end);
}

fn runGemmNNColTaskF16(task: *const ColTaskF16) void {
    gemmNNColsF16(task.cd, task.ad, task.bd, task.m, task.n, task.k, task.col_start, task.col_end);
}

pub fn gemmNNRange(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    _ = m;
    if (row_start == row_end or n == 0) return;
    if (k == 0) {
        for (row_start..row_end) |i| {
            @memset(cd[i * n .. (i + 1) * n], 0);
        }
        return;
    }

    var i = row_start;
    while (i + 8 <= row_end) : (i += 8) {
        gemmNNRows8(cd, ad, bd, i, n, k);
    }
    while (i + 4 <= row_end) : (i += 4) {
        gemmNNRows4(cd, ad, bd, i, n, k);
    }
    while (i < row_end) : (i += 1) {
        gemmNNRow(cd, ad, bd, i, n, k);
    }
}

pub fn gemmTNRange(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    if (row_start == row_end or n == 0) return;
    if (k == 0) {
        for (row_start..row_end) |i| {
            @memset(cd[i * n .. (i + 1) * n], 0);
        }
        return;
    }

    var i = row_start;
    while (i + 8 <= row_end) : (i += 8) {
        gemmTNRows8(cd, ad, bd, i, m, n, k);
    }
    while (i + 4 <= row_end) : (i += 4) {
        gemmTNRows4(cd, ad, bd, i, m, n, k);
    }
    while (i < row_end) : (i += 1) {
        gemmTNRow(cd, ad, bd, i, m, n, k);
    }
}

pub fn gemmNTRange(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    _ = m;
    if (row_start == row_end or n == 0) return;
    if (k == 0) {
        for (row_start..row_end) |i| {
            @memset(cd[i * n .. (i + 1) * n], 0);
        }
        return;
    }

    for (row_start..row_end) |i| {
        const a_row = ad[i * k .. (i + 1) * k];
        var j: usize = 0;
        while (j + 4 <= n) : (j += 4) {
            dot4(cd[i * n + j .. i * n + j + 4], a_row, bd, j, k);
        }
        while (j < n) : (j += 1) {
            cd[i * n + j] = vecDot(a_row, bd[j * k .. (j + 1) * k]);
        }
    }
}

fn gemmNNCols(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (col_start == col_end or m == 0) return;
    if (k == 0) {
        for (0..m) |i| {
            @memset(cd[i * n + col_start .. i * n + col_end], 0);
        }
        return;
    }
    for (0..m) |i| {
        var j = col_start;
        while (j + vector_len <= col_end) : (j += vector_len) {
            var acc: Vf32 = @splat(0);
            for (0..k) |p| {
                acc += @as(Vf32, @splat(ad[i * k + p])) * @as(Vf32, bd[p * n + j ..][0..vector_len].*);
            }
            cd[i * n + j ..][0..vector_len].* = acc;
        }
        while (j < col_end) : (j += 1) {
            var s: f32 = 0;
            for (0..k) |p| s += ad[i * k + p] * bd[p * n + j];
            cd[i * n + j] = s;
        }
    }
}

fn gemmTNCols(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (col_start == col_end or m == 0) return;
    if (k == 0) {
        for (0..m) |i| {
            @memset(cd[i * n + col_start .. i * n + col_end], 0);
        }
        return;
    }
    for (0..m) |i| {
        var j = col_start;
        while (j + vector_len <= col_end) : (j += vector_len) {
            var acc: Vf32 = @splat(0);
            for (0..k) |p| {
                acc += @as(Vf32, @splat(ad[p * m + i])) * @as(Vf32, bd[p * n + j ..][0..vector_len].*);
            }
            cd[i * n + j ..][0..vector_len].* = acc;
        }
        while (j < col_end) : (j += 1) {
            var s: f32 = 0;
            for (0..k) |p| s += ad[p * m + i] * bd[p * n + j];
            cd[i * n + j] = s;
        }
    }
}

fn gemmNTCols(cd: []f32, ad: []const f32, bd: []const f32, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (col_start == col_end or m == 0) return;
    if (k == 0) {
        for (0..m) |i| {
            @memset(cd[i * n + col_start .. i * n + col_end], 0);
        }
        return;
    }
    // Column tile OUTER, rows inner: the 4-row B tile stays cache-hot across
    // all m A rows, so B streams once total instead of once per row (at m=8
    // the row-outer order cost 8x the memory traffic — the whole RHS weight
    // re-read per row). Same dot4/vecDot per output element in the same
    // order per element: bitwise-identical results, scheduling-only change.
    var j = col_start;
    while (j + 4 <= col_end) : (j += 4) {
        for (0..m) |i| {
            dot4(cd[i * n + j .. i * n + j + 4], ad[i * k .. (i + 1) * k], bd, j, k);
        }
    }
    while (j < col_end) : (j += 1) {
        for (0..m) |i| {
            cd[i * n + j] = vecDot(ad[i * k .. (i + 1) * k], bd[j * k .. (j + 1) * k]);
        }
    }
}

fn gemmNTF16RhsRange(cd: []f32, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    _ = m;
    if (row_start == row_end or n == 0) return;
    if (k == 0) {
        for (row_start..row_end) |i| {
            @memset(cd[i * n .. (i + 1) * n], 0);
        }
        return;
    }

    var i = row_start;
    // Avoid a 6+1 split; the scalar row tail is slower than 4+3 here.
    while (i + 6 <= row_end and row_end - (i + 6) != 1) : (i += 6) {
        gemmNTF16RhsSmallRowsCols(6, cd[i * n ..], ad[i * k ..], bd, n, k, 0, n);
    }
    while (i + 4 <= row_end) : (i += 4) {
        gemmNTF16RhsM4Cols(cd[i * n ..], ad[i * k ..], bd, n, k, 0, n);
    }
    if (i + 3 <= row_end) {
        gemmNTF16RhsSmallRowsCols(3, cd[i * n ..], ad[i * k ..], bd, n, k, 0, n);
        i += 3;
    }
    if (i + 2 <= row_end) {
        gemmNTF16RhsSmallRowsCols(2, cd[i * n ..], ad[i * k ..], bd, n, k, 0, n);
        i += 2;
    }

    while (i < row_end) : (i += 1) {
        const a_row = ad[i * k .. (i + 1) * k];
        var j: usize = 0;
        while (j + 4 <= n) : (j += 4) {
            dot4F16Rhs(cd[i * n + j .. i * n + j + 4], a_row, bd, j, k);
        }
        while (j < n) : (j += 1) {
            cd[i * n + j] = vecDotF16HalfAccumToF32(a_row, bd[j * k .. (j + 1) * k]);
        }
    }
}

fn gemmNTF16RhsCols(cd: []f32, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (col_start == col_end or m == 0) return;
    if (k == 0) {
        for (0..m) |i| {
            @memset(cd[i * n + col_start .. i * n + col_end], 0);
        }
        return;
    }
    var i: usize = 0;
    // Avoid a 6+1 split; the scalar row tail is slower than 4+3 here.
    while (i + 6 <= m and m - (i + 6) != 1) : (i += 6) {
        gemmNTF16RhsSmallRowsCols(6, cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
    }
    while (i + 4 <= m) : (i += 4) {
        gemmNTF16RhsM4Cols(cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
    }
    if (i + 3 <= m) {
        gemmNTF16RhsSmallRowsCols(3, cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
        i += 3;
    }
    if (i + 2 <= m) {
        gemmNTF16RhsSmallRowsCols(2, cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
        i += 2;
    }

    while (i < m) : (i += 1) {
        const a_row = ad[i * k .. (i + 1) * k];
        var j = col_start;
        while (j + 4 <= col_end) : (j += 4) {
            dot4F16Rhs(cd[i * n + j .. i * n + j + 4], a_row, bd, j, k);
        }
        while (j < col_end) : (j += 1) {
            cd[i * n + j] = vecDotF16HalfAccumToF32(a_row, bd[j * k .. (j + 1) * k]);
        }
    }
}

inline fn gemmNTF16RhsSmallRowsCols(comptime rows: usize, cd: []f32, ad: []const f16, bd: []const f16, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (comptime !f16_accum_native) {
        return gemmNTF16RhsSmallRowsColsWide(rows, cd, ad, bd, n, k, col_start, col_end);
    }
    var j = col_start;
    while (j + 4 <= col_end) : (j += 4) {
        var acc: [rows][4]Vf16 = undefined;
        inline for (0..rows) |r| {
            inline for (0..4) |c| {
                acc[r][c] = @splat(0);
            }
        }

        var p: usize = 0;
        while (p + vector_len_f16 <= k) : (p += vector_len_f16) {
            const b0: Vf16 = bd[(j + 0) * k + p ..][0..vector_len_f16].*;
            const b1: Vf16 = bd[(j + 1) * k + p ..][0..vector_len_f16].*;
            const b2: Vf16 = bd[(j + 2) * k + p ..][0..vector_len_f16].*;
            const b3: Vf16 = bd[(j + 3) * k + p ..][0..vector_len_f16].*;
            inline for (0..rows) |r| {
                const av: Vf16 = ad[r * k + p ..][0..vector_len_f16].*;
                acc[r][0] = @mulAdd(Vf16, av, b0, acc[r][0]);
                acc[r][1] = @mulAdd(Vf16, av, b1, acc[r][1]);
                acc[r][2] = @mulAdd(Vf16, av, b2, acc[r][2]);
                acc[r][3] = @mulAdd(Vf16, av, b3, acc[r][3]);
            }
        }

        var sums: [rows][4]f32 = undefined;
        inline for (0..rows) |r| {
            inline for (0..4) |c| {
                sums[r][c] = @floatCast(@reduce(.Add, acc[r][c]));
            }
        }
        while (p < k) : (p += 1) {
            inline for (0..rows) |r| {
                const av = @as(f32, @floatCast(ad[r * k + p]));
                sums[r][0] += av * @as(f32, @floatCast(bd[(j + 0) * k + p]));
                sums[r][1] += av * @as(f32, @floatCast(bd[(j + 1) * k + p]));
                sums[r][2] += av * @as(f32, @floatCast(bd[(j + 2) * k + p]));
                sums[r][3] += av * @as(f32, @floatCast(bd[(j + 3) * k + p]));
            }
        }

        inline for (0..rows) |r| {
            cd[r * n + j + 0] = sums[r][0];
            cd[r * n + j + 1] = sums[r][1];
            cd[r * n + j + 2] = sums[r][2];
            cd[r * n + j + 3] = sums[r][3];
        }
    }

    while (j < col_end) : (j += 1) {
        var acc: [rows]Vf16 = undefined;
        inline for (0..rows) |r| acc[r] = @splat(0);

        var p: usize = 0;
        while (p + vector_len_f16 <= k) : (p += vector_len_f16) {
            const bv: Vf16 = bd[j * k + p ..][0..vector_len_f16].*;
            inline for (0..rows) |r| {
                const av: Vf16 = ad[r * k + p ..][0..vector_len_f16].*;
                acc[r] = @mulAdd(Vf16, av, bv, acc[r]);
            }
        }

        var sums: [rows]f32 = undefined;
        inline for (0..rows) |r| sums[r] = @floatCast(@reduce(.Add, acc[r]));
        while (p < k) : (p += 1) {
            const bv = @as(f32, @floatCast(bd[j * k + p]));
            inline for (0..rows) |r| {
                sums[r] += @as(f32, @floatCast(ad[r * k + p])) * bv;
            }
        }
        inline for (0..rows) |r| {
            cd[r * n + j] = sums[r];
        }
    }
}

fn gemmNTF16RhsM4Cols(cd: []f32, ad: []const f16, bd: []const f16, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (comptime !f16_accum_native) {
        return gemmNTF16RhsSmallRowsColsWide(4, cd, ad, bd, n, k, col_start, col_end);
    }
    var j = col_start;
    while (j + 4 <= col_end) : (j += 4) {
        var acc: [4][4]Vf16 = undefined;
        inline for (0..4) |r| {
            inline for (0..4) |c| {
                acc[r][c] = @splat(0);
            }
        }

        var p: usize = 0;
        while (p + vector_len_f16 <= k) : (p += vector_len_f16) {
            const b0: Vf16 = bd[(j + 0) * k + p ..][0..vector_len_f16].*;
            const b1: Vf16 = bd[(j + 1) * k + p ..][0..vector_len_f16].*;
            const b2: Vf16 = bd[(j + 2) * k + p ..][0..vector_len_f16].*;
            const b3: Vf16 = bd[(j + 3) * k + p ..][0..vector_len_f16].*;
            inline for (0..4) |r| {
                const av: Vf16 = ad[r * k + p ..][0..vector_len_f16].*;
                acc[r][0] = @mulAdd(Vf16, av, b0, acc[r][0]);
                acc[r][1] = @mulAdd(Vf16, av, b1, acc[r][1]);
                acc[r][2] = @mulAdd(Vf16, av, b2, acc[r][2]);
                acc[r][3] = @mulAdd(Vf16, av, b3, acc[r][3]);
            }
        }

        var sums: [4][4]f32 = undefined;
        inline for (0..4) |r| {
            inline for (0..4) |c| {
                sums[r][c] = @floatCast(@reduce(.Add, acc[r][c]));
            }
        }
        while (p < k) : (p += 1) {
            inline for (0..4) |r| {
                const av = @as(f32, @floatCast(ad[r * k + p]));
                sums[r][0] += av * @as(f32, @floatCast(bd[(j + 0) * k + p]));
                sums[r][1] += av * @as(f32, @floatCast(bd[(j + 1) * k + p]));
                sums[r][2] += av * @as(f32, @floatCast(bd[(j + 2) * k + p]));
                sums[r][3] += av * @as(f32, @floatCast(bd[(j + 3) * k + p]));
            }
        }

        inline for (0..4) |r| {
            cd[r * n + j + 0] = sums[r][0];
            cd[r * n + j + 1] = sums[r][1];
            cd[r * n + j + 2] = sums[r][2];
            cd[r * n + j + 3] = sums[r][3];
        }
    }

    while (j < col_end) : (j += 1) {
        var acc: [4]Vf16 = undefined;
        inline for (0..4) |r| acc[r] = @splat(0);

        var p: usize = 0;
        while (p + vector_len_f16 <= k) : (p += vector_len_f16) {
            const bv: Vf16 = bd[j * k + p ..][0..vector_len_f16].*;
            inline for (0..4) |r| {
                const av: Vf16 = ad[r * k + p ..][0..vector_len_f16].*;
                acc[r] = @mulAdd(Vf16, av, bv, acc[r]);
            }
        }

        var sums: [4]f32 = undefined;
        inline for (0..4) |r| sums[r] = @floatCast(@reduce(.Add, acc[r]));
        while (p < k) : (p += 1) {
            const bv = @as(f32, @floatCast(bd[j * k + p]));
            inline for (0..4) |r| {
                sums[r] += @as(f32, @floatCast(ad[r * k + p])) * bv;
            }
        }
        inline for (0..4) |r| {
            cd[r * n + j] = sums[r];
        }
    }
}

// f32-accumulate twin of gemmNTF16RhsSmallRowsCols/M4Cols for ISAs without
// native f16 arithmetic (see f16_accum_native): each f16 chunk is widened once
// per load and everything accumulates in f32 — the gemmNTBf16RhsSmallRowsCols
// shape with F16C converts instead of bit shifts.
inline fn gemmNTF16RhsSmallRowsColsWide(comptime rows: usize, cd: []f32, ad: []const f16, bd: []const f16, n: usize, k: usize, col_start: usize, col_end: usize) void {
    var j = col_start;
    while (j + 4 <= col_end) : (j += 4) {
        var acc: [rows][4]Vf32 = undefined;
        inline for (0..rows) |r| {
            inline for (0..4) |c| {
                acc[r][c] = @splat(0);
            }
        }

        var p: usize = 0;
        while (p + vector_len <= k) : (p += vector_len) {
            const b0: Vf32 = @floatCast(@as(Vf16ForF32, bd[(j + 0) * k + p ..][0..vector_len].*));
            const b1: Vf32 = @floatCast(@as(Vf16ForF32, bd[(j + 1) * k + p ..][0..vector_len].*));
            const b2: Vf32 = @floatCast(@as(Vf16ForF32, bd[(j + 2) * k + p ..][0..vector_len].*));
            const b3: Vf32 = @floatCast(@as(Vf16ForF32, bd[(j + 3) * k + p ..][0..vector_len].*));
            inline for (0..rows) |r| {
                const av: Vf32 = @floatCast(@as(Vf16ForF32, ad[r * k + p ..][0..vector_len].*));
                acc[r][0] = @mulAdd(Vf32, av, b0, acc[r][0]);
                acc[r][1] = @mulAdd(Vf32, av, b1, acc[r][1]);
                acc[r][2] = @mulAdd(Vf32, av, b2, acc[r][2]);
                acc[r][3] = @mulAdd(Vf32, av, b3, acc[r][3]);
            }
        }

        var sums: [rows][4]f32 = undefined;
        inline for (0..rows) |r| {
            inline for (0..4) |c| {
                sums[r][c] = @reduce(.Add, acc[r][c]);
            }
        }
        while (p < k) : (p += 1) {
            inline for (0..rows) |r| {
                const av = @as(f32, @floatCast(ad[r * k + p]));
                sums[r][0] += av * @as(f32, @floatCast(bd[(j + 0) * k + p]));
                sums[r][1] += av * @as(f32, @floatCast(bd[(j + 1) * k + p]));
                sums[r][2] += av * @as(f32, @floatCast(bd[(j + 2) * k + p]));
                sums[r][3] += av * @as(f32, @floatCast(bd[(j + 3) * k + p]));
            }
        }

        inline for (0..rows) |r| {
            cd[r * n + j + 0] = sums[r][0];
            cd[r * n + j + 1] = sums[r][1];
            cd[r * n + j + 2] = sums[r][2];
            cd[r * n + j + 3] = sums[r][3];
        }
    }

    while (j < col_end) : (j += 1) {
        var acc: [rows]Vf32 = undefined;
        inline for (0..rows) |r| acc[r] = @splat(0);

        var p: usize = 0;
        while (p + vector_len <= k) : (p += vector_len) {
            const bv: Vf32 = @floatCast(@as(Vf16ForF32, bd[j * k + p ..][0..vector_len].*));
            inline for (0..rows) |r| {
                const av: Vf32 = @floatCast(@as(Vf16ForF32, ad[r * k + p ..][0..vector_len].*));
                acc[r] = @mulAdd(Vf32, av, bv, acc[r]);
            }
        }

        var sums: [rows]f32 = undefined;
        inline for (0..rows) |r| sums[r] = @reduce(.Add, acc[r]);
        while (p < k) : (p += 1) {
            const bv = @as(f32, @floatCast(bd[j * k + p]));
            inline for (0..rows) |r| {
                sums[r] += @as(f32, @floatCast(ad[r * k + p])) * bv;
            }
        }
        inline for (0..rows) |r| {
            cd[r * n + j] = sums[r];
        }
    }
}

fn gemmNTBf16RhsRange(cd: []f32, ad: []const f32, bd: []const u16, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    _ = m;
    if (row_start == row_end or n == 0) return;
    if (k == 0) {
        for (row_start..row_end) |i| {
            @memset(cd[i * n .. (i + 1) * n], 0);
        }
        return;
    }

    var i = row_start;
    // Avoid a 6+1 split; the scalar row tail is slower than 4+3 here.
    while (i + 6 <= row_end and row_end - (i + 6) != 1) : (i += 6) {
        gemmNTBf16RhsSmallRowsCols(6, cd[i * n ..], ad[i * k ..], bd, n, k, 0, n);
    }
    while (i + 4 <= row_end) : (i += 4) {
        gemmNTBf16RhsSmallRowsCols(4, cd[i * n ..], ad[i * k ..], bd, n, k, 0, n);
    }
    if (i + 3 <= row_end) {
        gemmNTBf16RhsSmallRowsCols(3, cd[i * n ..], ad[i * k ..], bd, n, k, 0, n);
        i += 3;
    }
    if (i + 2 <= row_end) {
        gemmNTBf16RhsSmallRowsCols(2, cd[i * n ..], ad[i * k ..], bd, n, k, 0, n);
        i += 2;
    }

    while (i < row_end) : (i += 1) {
        const a_row = ad[i * k .. (i + 1) * k];
        var j: usize = 0;
        while (j + 4 <= n) : (j += 4) {
            dot4Bf16Rhs(cd[i * n + j .. i * n + j + 4], a_row, bd, j, k);
        }
        while (j < n) : (j += 1) {
            cd[i * n + j] = vecDotBf16RhsToF32(a_row, bd[j * k .. (j + 1) * k]);
        }
    }
}

fn gemmNTBf16RhsCols(cd: []f32, ad: []const f32, bd: []const u16, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (col_start == col_end or m == 0) return;
    if (k == 0) {
        for (0..m) |i| {
            @memset(cd[i * n + col_start .. i * n + col_end], 0);
        }
        return;
    }
    var i: usize = 0;
    // Avoid a 6+1 split; the scalar row tail is slower than 4+3 here.
    while (i + 6 <= m and m - (i + 6) != 1) : (i += 6) {
        gemmNTBf16RhsSmallRowsCols(6, cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
    }
    while (i + 4 <= m) : (i += 4) {
        gemmNTBf16RhsSmallRowsCols(4, cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
    }
    if (i + 3 <= m) {
        gemmNTBf16RhsSmallRowsCols(3, cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
        i += 3;
    }
    if (i + 2 <= m) {
        gemmNTBf16RhsSmallRowsCols(2, cd[i * n ..], ad[i * k ..], bd, n, k, col_start, col_end);
        i += 2;
    }

    while (i < m) : (i += 1) {
        const a_row = ad[i * k .. (i + 1) * k];
        var j = col_start;
        while (j + 4 <= col_end) : (j += 4) {
            dot4Bf16Rhs(cd[i * n + j .. i * n + j + 4], a_row, bd, j, k);
        }
        while (j < col_end) : (j += 1) {
            cd[i * n + j] = vecDotBf16RhsToF32(a_row, bd[j * k .. (j + 1) * k]);
        }
    }
}

inline fn gemmNTBf16RhsSmallRowsCols(comptime rows: usize, cd: []f32, ad: []const f32, bd: []const u16, n: usize, k: usize, col_start: usize, col_end: usize) void {
    var j = col_start;
    while (j + 4 <= col_end) : (j += 4) {
        var acc: [rows][4]Vf32 = undefined;
        inline for (0..rows) |r| {
            inline for (0..4) |c| {
                acc[r][c] = @splat(0);
            }
        }

        var p: usize = 0;
        while (p + vector_len <= k) : (p += vector_len) {
            const b0 = bf16VecToF32(bd[(j + 0) * k + p ..][0..vector_len].*);
            const b1 = bf16VecToF32(bd[(j + 1) * k + p ..][0..vector_len].*);
            const b2 = bf16VecToF32(bd[(j + 2) * k + p ..][0..vector_len].*);
            const b3 = bf16VecToF32(bd[(j + 3) * k + p ..][0..vector_len].*);
            inline for (0..rows) |r| {
                const av: Vf32 = ad[r * k + p ..][0..vector_len].*;
                acc[r][0] = @mulAdd(Vf32, av, b0, acc[r][0]);
                acc[r][1] = @mulAdd(Vf32, av, b1, acc[r][1]);
                acc[r][2] = @mulAdd(Vf32, av, b2, acc[r][2]);
                acc[r][3] = @mulAdd(Vf32, av, b3, acc[r][3]);
            }
        }

        var sums: [rows][4]f32 = undefined;
        inline for (0..rows) |r| {
            inline for (0..4) |c| {
                sums[r][c] = @reduce(.Add, acc[r][c]);
            }
        }
        while (p < k) : (p += 1) {
            inline for (0..rows) |r| {
                const av = ad[r * k + p];
                sums[r][0] += av * dtype_mod.bf16ToF32(bd[(j + 0) * k + p]);
                sums[r][1] += av * dtype_mod.bf16ToF32(bd[(j + 1) * k + p]);
                sums[r][2] += av * dtype_mod.bf16ToF32(bd[(j + 2) * k + p]);
                sums[r][3] += av * dtype_mod.bf16ToF32(bd[(j + 3) * k + p]);
            }
        }

        inline for (0..rows) |r| {
            cd[r * n + j + 0] = sums[r][0];
            cd[r * n + j + 1] = sums[r][1];
            cd[r * n + j + 2] = sums[r][2];
            cd[r * n + j + 3] = sums[r][3];
        }
    }

    while (j < col_end) : (j += 1) {
        var acc: [rows]Vf32 = undefined;
        inline for (0..rows) |r| acc[r] = @splat(0);

        var p: usize = 0;
        while (p + vector_len <= k) : (p += vector_len) {
            const bv = bf16VecToF32(bd[j * k + p ..][0..vector_len].*);
            inline for (0..rows) |r| {
                const av: Vf32 = ad[r * k + p ..][0..vector_len].*;
                acc[r] = @mulAdd(Vf32, av, bv, acc[r]);
            }
        }

        var sums: [rows]f32 = undefined;
        inline for (0..rows) |r| sums[r] = @reduce(.Add, acc[r]);
        while (p < k) : (p += 1) {
            const bv = dtype_mod.bf16ToF32(bd[j * k + p]);
            inline for (0..rows) |r| {
                sums[r] += ad[r * k + p] * bv;
            }
        }
        inline for (0..rows) |r| {
            cd[r * n + j] = sums[r];
        }
    }
}

inline fn gemmNNRows4(cd: []f32, ad: []const f32, bd: []const f32, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + 2 * vector_len <= n) : (j += 2 * vector_len) {
        var acc00: Vf32 = @splat(0);
        var acc01: Vf32 = @splat(0);
        var acc10: Vf32 = @splat(0);
        var acc11: Vf32 = @splat(0);
        var acc20: Vf32 = @splat(0);
        var acc21: Vf32 = @splat(0);
        var acc30: Vf32 = @splat(0);
        var acc31: Vf32 = @splat(0);

        for (0..k) |p| {
            const b0: Vf32 = bd[p * n + j ..][0..vector_len].*;
            const b1: Vf32 = bd[p * n + j + vector_len ..][0..vector_len].*;
            const a0: Vf32 = @splat(ad[(row + 0) * k + p]);
            const a1: Vf32 = @splat(ad[(row + 1) * k + p]);
            const a2: Vf32 = @splat(ad[(row + 2) * k + p]);
            const a3: Vf32 = @splat(ad[(row + 3) * k + p]);
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
            acc20 += a2 * b0;
            acc21 += a2 * b1;
            acc30 += a3 * b0;
            acc31 += a3 * b1;
        }

        cd[(row + 0) * n + j ..][0..vector_len].* = acc00;
        cd[(row + 0) * n + j + vector_len ..][0..vector_len].* = acc01;
        cd[(row + 1) * n + j ..][0..vector_len].* = acc10;
        cd[(row + 1) * n + j + vector_len ..][0..vector_len].* = acc11;
        cd[(row + 2) * n + j ..][0..vector_len].* = acc20;
        cd[(row + 2) * n + j + vector_len ..][0..vector_len].* = acc21;
        cd[(row + 3) * n + j ..][0..vector_len].* = acc30;
        cd[(row + 3) * n + j + vector_len ..][0..vector_len].* = acc31;
    }
    while (j + vector_len <= n) : (j += vector_len) {
        var acc0: Vf32 = @splat(0);
        var acc1: Vf32 = @splat(0);
        var acc2: Vf32 = @splat(0);
        var acc3: Vf32 = @splat(0);
        for (0..k) |p| {
            const b0: Vf32 = bd[p * n + j ..][0..vector_len].*;
            acc0 += @as(Vf32, @splat(ad[(row + 0) * k + p])) * b0;
            acc1 += @as(Vf32, @splat(ad[(row + 1) * k + p])) * b0;
            acc2 += @as(Vf32, @splat(ad[(row + 2) * k + p])) * b0;
            acc3 += @as(Vf32, @splat(ad[(row + 3) * k + p])) * b0;
        }
        cd[(row + 0) * n + j ..][0..vector_len].* = acc0;
        cd[(row + 1) * n + j ..][0..vector_len].* = acc1;
        cd[(row + 2) * n + j ..][0..vector_len].* = acc2;
        cd[(row + 3) * n + j ..][0..vector_len].* = acc3;
    }
    while (j < n) : (j += 1) {
        var s0: f32 = 0;
        var s1: f32 = 0;
        var s2: f32 = 0;
        var s3: f32 = 0;
        for (0..k) |p| {
            const b = bd[p * n + j];
            s0 += ad[(row + 0) * k + p] * b;
            s1 += ad[(row + 1) * k + p] * b;
            s2 += ad[(row + 2) * k + p] * b;
            s3 += ad[(row + 3) * k + p] * b;
        }
        cd[(row + 0) * n + j] = s0;
        cd[(row + 1) * n + j] = s1;
        cd[(row + 2) * n + j] = s2;
        cd[(row + 3) * n + j] = s3;
    }
}

inline fn gemmNNRows8(cd: []f32, ad: []const f32, bd: []const f32, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + 2 * vector_len <= n) : (j += 2 * vector_len) {
        var acc: [8][2]Vf32 = undefined;
        inline for (0..8) |r| {
            acc[r][0] = @splat(0);
            acc[r][1] = @splat(0);
        }

        for (0..k) |p| {
            const b0: Vf32 = bd[p * n + j ..][0..vector_len].*;
            const b1: Vf32 = bd[p * n + j + vector_len ..][0..vector_len].*;
            inline for (0..8) |r| {
                const a: Vf32 = @splat(ad[(row + r) * k + p]);
                acc[r][0] += a * b0;
                acc[r][1] += a * b1;
            }
        }

        inline for (0..8) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = acc[r][0];
            cd[(row + r) * n + j + vector_len ..][0..vector_len].* = acc[r][1];
        }
    }
    while (j + vector_len <= n) : (j += vector_len) {
        var acc: [8]Vf32 = undefined;
        inline for (0..8) |r| {
            acc[r] = @splat(0);
        }
        for (0..k) |p| {
            const b: Vf32 = bd[p * n + j ..][0..vector_len].*;
            inline for (0..8) |r| {
                acc[r] += @as(Vf32, @splat(ad[(row + r) * k + p])) * b;
            }
        }
        inline for (0..8) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = acc[r];
        }
    }
    while (j < n) : (j += 1) {
        var sums: [8]f32 = [_]f32{0} ** 8;
        for (0..k) |p| {
            const b = bd[p * n + j];
            inline for (0..8) |r| {
                sums[r] += ad[(row + r) * k + p] * b;
            }
        }
        inline for (0..8) |r| {
            cd[(row + r) * n + j] = sums[r];
        }
    }
}

inline fn gemmNNRow(cd: []f32, ad: []const f32, bd: []const f32, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + vector_len <= n) : (j += vector_len) {
        var acc: Vf32 = @splat(0);
        for (0..k) |p| {
            const b: Vf32 = bd[p * n + j ..][0..vector_len].*;
            acc += @as(Vf32, @splat(ad[row * k + p])) * b;
        }
        cd[row * n + j ..][0..vector_len].* = acc;
    }
    while (j < n) : (j += 1) {
        var s: f32 = 0;
        for (0..k) |p| s += ad[row * k + p] * bd[p * n + j];
        cd[row * n + j] = s;
    }
}

inline fn gemmTNRows8(cd: []f32, ad: []const f32, bd: []const f32, row: usize, m: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + 2 * vector_len <= n) : (j += 2 * vector_len) {
        var acc: [8][2]Vf32 = undefined;
        inline for (0..8) |r| {
            acc[r][0] = @splat(0);
            acc[r][1] = @splat(0);
        }

        for (0..k) |p| {
            const b0: Vf32 = bd[p * n + j ..][0..vector_len].*;
            const b1: Vf32 = bd[p * n + j + vector_len ..][0..vector_len].*;
            inline for (0..8) |r| {
                const a: Vf32 = @splat(ad[p * m + row + r]);
                acc[r][0] += a * b0;
                acc[r][1] += a * b1;
            }
        }

        inline for (0..8) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = acc[r][0];
            cd[(row + r) * n + j + vector_len ..][0..vector_len].* = acc[r][1];
        }
    }
    while (j + vector_len <= n) : (j += vector_len) {
        var acc: [8]Vf32 = undefined;
        inline for (0..8) |r| {
            acc[r] = @splat(0);
        }
        for (0..k) |p| {
            const b: Vf32 = bd[p * n + j ..][0..vector_len].*;
            inline for (0..8) |r| {
                acc[r] += @as(Vf32, @splat(ad[p * m + row + r])) * b;
            }
        }
        inline for (0..8) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = acc[r];
        }
    }
    while (j < n) : (j += 1) {
        var sums: [8]f32 = [_]f32{0} ** 8;
        for (0..k) |p| {
            const b = bd[p * n + j];
            inline for (0..8) |r| {
                sums[r] += ad[p * m + row + r] * b;
            }
        }
        inline for (0..8) |r| {
            cd[(row + r) * n + j] = sums[r];
        }
    }
}

inline fn gemmTNRows4(cd: []f32, ad: []const f32, bd: []const f32, row: usize, m: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + 2 * vector_len <= n) : (j += 2 * vector_len) {
        var acc00: Vf32 = @splat(0);
        var acc01: Vf32 = @splat(0);
        var acc10: Vf32 = @splat(0);
        var acc11: Vf32 = @splat(0);
        var acc20: Vf32 = @splat(0);
        var acc21: Vf32 = @splat(0);
        var acc30: Vf32 = @splat(0);
        var acc31: Vf32 = @splat(0);

        for (0..k) |p| {
            const b0: Vf32 = bd[p * n + j ..][0..vector_len].*;
            const b1: Vf32 = bd[p * n + j + vector_len ..][0..vector_len].*;
            const a0: Vf32 = @splat(ad[p * m + row + 0]);
            const a1: Vf32 = @splat(ad[p * m + row + 1]);
            const a2: Vf32 = @splat(ad[p * m + row + 2]);
            const a3: Vf32 = @splat(ad[p * m + row + 3]);
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
            acc20 += a2 * b0;
            acc21 += a2 * b1;
            acc30 += a3 * b0;
            acc31 += a3 * b1;
        }

        cd[(row + 0) * n + j ..][0..vector_len].* = acc00;
        cd[(row + 0) * n + j + vector_len ..][0..vector_len].* = acc01;
        cd[(row + 1) * n + j ..][0..vector_len].* = acc10;
        cd[(row + 1) * n + j + vector_len ..][0..vector_len].* = acc11;
        cd[(row + 2) * n + j ..][0..vector_len].* = acc20;
        cd[(row + 2) * n + j + vector_len ..][0..vector_len].* = acc21;
        cd[(row + 3) * n + j ..][0..vector_len].* = acc30;
        cd[(row + 3) * n + j + vector_len ..][0..vector_len].* = acc31;
    }
    while (j + vector_len <= n) : (j += vector_len) {
        var acc0: Vf32 = @splat(0);
        var acc1: Vf32 = @splat(0);
        var acc2: Vf32 = @splat(0);
        var acc3: Vf32 = @splat(0);
        for (0..k) |p| {
            const b0: Vf32 = bd[p * n + j ..][0..vector_len].*;
            acc0 += @as(Vf32, @splat(ad[p * m + row + 0])) * b0;
            acc1 += @as(Vf32, @splat(ad[p * m + row + 1])) * b0;
            acc2 += @as(Vf32, @splat(ad[p * m + row + 2])) * b0;
            acc3 += @as(Vf32, @splat(ad[p * m + row + 3])) * b0;
        }
        cd[(row + 0) * n + j ..][0..vector_len].* = acc0;
        cd[(row + 1) * n + j ..][0..vector_len].* = acc1;
        cd[(row + 2) * n + j ..][0..vector_len].* = acc2;
        cd[(row + 3) * n + j ..][0..vector_len].* = acc3;
    }
    while (j < n) : (j += 1) {
        var s0: f32 = 0;
        var s1: f32 = 0;
        var s2: f32 = 0;
        var s3: f32 = 0;
        for (0..k) |p| {
            const b = bd[p * n + j];
            s0 += ad[p * m + row + 0] * b;
            s1 += ad[p * m + row + 1] * b;
            s2 += ad[p * m + row + 2] * b;
            s3 += ad[p * m + row + 3] * b;
        }
        cd[(row + 0) * n + j] = s0;
        cd[(row + 1) * n + j] = s1;
        cd[(row + 2) * n + j] = s2;
        cd[(row + 3) * n + j] = s3;
    }
}

inline fn gemmTNRow(cd: []f32, ad: []const f32, bd: []const f32, row: usize, m: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + vector_len <= n) : (j += vector_len) {
        var acc: Vf32 = @splat(0);
        for (0..k) |p| {
            const b: Vf32 = bd[p * n + j ..][0..vector_len].*;
            acc += @as(Vf32, @splat(ad[p * m + row])) * b;
        }
        cd[row * n + j ..][0..vector_len].* = acc;
    }
    while (j < n) : (j += 1) {
        var s: f32 = 0;
        for (0..k) |p| s += ad[p * m + row] * bd[p * n + j];
        cd[row * n + j] = s;
    }
}

inline fn dot4(out: []f32, a: []const f32, b: []const f32, b_row: usize, k: usize) void {
    var acc0: Vf32 = @splat(0);
    var acc1: Vf32 = @splat(0);
    var acc2: Vf32 = @splat(0);
    var acc3: Vf32 = @splat(0);

    var p: usize = 0;
    while (p + vector_len <= k) : (p += vector_len) {
        const av: Vf32 = a[p..][0..vector_len].*;
        acc0 += av * @as(Vf32, b[(b_row + 0) * k + p ..][0..vector_len].*);
        acc1 += av * @as(Vf32, b[(b_row + 1) * k + p ..][0..vector_len].*);
        acc2 += av * @as(Vf32, b[(b_row + 2) * k + p ..][0..vector_len].*);
        acc3 += av * @as(Vf32, b[(b_row + 3) * k + p ..][0..vector_len].*);
    }

    var s0 = @reduce(.Add, acc0);
    var s1 = @reduce(.Add, acc1);
    var s2 = @reduce(.Add, acc2);
    var s3 = @reduce(.Add, acc3);
    while (p < k) : (p += 1) {
        const av = a[p];
        s0 += av * b[(b_row + 0) * k + p];
        s1 += av * b[(b_row + 1) * k + p];
        s2 += av * b[(b_row + 2) * k + p];
        s3 += av * b[(b_row + 3) * k + p];
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

inline fn dot4F16Rhs(out: []f32, a: []const f16, b: []const f16, b_row: usize, k: usize) void {
    if (comptime !f16_accum_native) {
        return dot4F16RhsWide(out, a, b, b_row, k);
    }
    var acc0: Vf16 = @splat(0);
    var acc1: Vf16 = @splat(0);
    var acc2: Vf16 = @splat(0);
    var acc3: Vf16 = @splat(0);

    var p: usize = 0;
    while (p + vector_len_f16 <= k) : (p += vector_len_f16) {
        const av: Vf16 = a[p..][0..vector_len_f16].*;
        acc0 = @mulAdd(Vf16, av, b[(b_row + 0) * k + p ..][0..vector_len_f16].*, acc0);
        acc1 = @mulAdd(Vf16, av, b[(b_row + 1) * k + p ..][0..vector_len_f16].*, acc1);
        acc2 = @mulAdd(Vf16, av, b[(b_row + 2) * k + p ..][0..vector_len_f16].*, acc2);
        acc3 = @mulAdd(Vf16, av, b[(b_row + 3) * k + p ..][0..vector_len_f16].*, acc3);
    }

    var s0: f32 = @floatCast(@reduce(.Add, acc0));
    var s1: f32 = @floatCast(@reduce(.Add, acc1));
    var s2: f32 = @floatCast(@reduce(.Add, acc2));
    var s3: f32 = @floatCast(@reduce(.Add, acc3));
    while (p < k) : (p += 1) {
        const av = @as(f32, @floatCast(a[p]));
        s0 += av * @as(f32, @floatCast(b[(b_row + 0) * k + p]));
        s1 += av * @as(f32, @floatCast(b[(b_row + 1) * k + p]));
        s2 += av * @as(f32, @floatCast(b[(b_row + 2) * k + p]));
        s3 += av * @as(f32, @floatCast(b[(b_row + 3) * k + p]));
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

// f32-accumulate twin of dot4F16Rhs (see f16_accum_native): the dot4Bf16Rhs
// shape with F16C converts instead of bit shifts.
inline fn dot4F16RhsWide(out: []f32, a: []const f16, b: []const f16, b_row: usize, k: usize) void {
    var acc0: Vf32 = @splat(0);
    var acc1: Vf32 = @splat(0);
    var acc2: Vf32 = @splat(0);
    var acc3: Vf32 = @splat(0);

    var p: usize = 0;
    while (p + vector_len <= k) : (p += vector_len) {
        const av: Vf32 = @floatCast(@as(Vf16ForF32, a[p..][0..vector_len].*));
        acc0 = @mulAdd(Vf32, av, @floatCast(@as(Vf16ForF32, b[(b_row + 0) * k + p ..][0..vector_len].*)), acc0);
        acc1 = @mulAdd(Vf32, av, @floatCast(@as(Vf16ForF32, b[(b_row + 1) * k + p ..][0..vector_len].*)), acc1);
        acc2 = @mulAdd(Vf32, av, @floatCast(@as(Vf16ForF32, b[(b_row + 2) * k + p ..][0..vector_len].*)), acc2);
        acc3 = @mulAdd(Vf32, av, @floatCast(@as(Vf16ForF32, b[(b_row + 3) * k + p ..][0..vector_len].*)), acc3);
    }

    var s0 = @reduce(.Add, acc0);
    var s1 = @reduce(.Add, acc1);
    var s2 = @reduce(.Add, acc2);
    var s3 = @reduce(.Add, acc3);
    while (p < k) : (p += 1) {
        const av = @as(f32, @floatCast(a[p]));
        s0 += av * @as(f32, @floatCast(b[(b_row + 0) * k + p]));
        s1 += av * @as(f32, @floatCast(b[(b_row + 1) * k + p]));
        s2 += av * @as(f32, @floatCast(b[(b_row + 2) * k + p]));
        s3 += av * @as(f32, @floatCast(b[(b_row + 3) * k + p]));
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

inline fn vecDotF16HalfAccumToF32(x: []const f16, y: []const f16) f32 {
    if (comptime !f16_accum_native) {
        return vecDotF16ToF32(x, y);
    }
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

inline fn dot4Bf16Rhs(out: []f32, a: []const f32, b: []const u16, b_row: usize, k: usize) void {
    var acc0: Vf32 = @splat(0);
    var acc1: Vf32 = @splat(0);
    var acc2: Vf32 = @splat(0);
    var acc3: Vf32 = @splat(0);

    var p: usize = 0;
    while (p + vector_len <= k) : (p += vector_len) {
        const av: Vf32 = a[p..][0..vector_len].*;
        acc0 = @mulAdd(Vf32, av, bf16VecToF32(b[(b_row + 0) * k + p ..][0..vector_len].*), acc0);
        acc1 = @mulAdd(Vf32, av, bf16VecToF32(b[(b_row + 1) * k + p ..][0..vector_len].*), acc1);
        acc2 = @mulAdd(Vf32, av, bf16VecToF32(b[(b_row + 2) * k + p ..][0..vector_len].*), acc2);
        acc3 = @mulAdd(Vf32, av, bf16VecToF32(b[(b_row + 3) * k + p ..][0..vector_len].*), acc3);
    }

    var s0 = @reduce(.Add, acc0);
    var s1 = @reduce(.Add, acc1);
    var s2 = @reduce(.Add, acc2);
    var s3 = @reduce(.Add, acc3);
    while (p < k) : (p += 1) {
        const av = a[p];
        s0 += av * dtype_mod.bf16ToF32(b[(b_row + 0) * k + p]);
        s1 += av * dtype_mod.bf16ToF32(b[(b_row + 1) * k + p]);
        s2 += av * dtype_mod.bf16ToF32(b[(b_row + 2) * k + p]);
        s3 += av * dtype_mod.bf16ToF32(b[(b_row + 3) * k + p]);
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

inline fn vecDotBf16RhsToF32(x: []const f32, y: []const u16) f32 {
    var i: usize = 0;
    var acc0: Vf32 = @splat(0);
    var acc1: Vf32 = @splat(0);
    var acc2: Vf32 = @splat(0);
    var acc3: Vf32 = @splat(0);

    while (i + 4 * vector_len <= x.len) : (i += 4 * vector_len) {
        const x0: Vf32 = x[i..][0..vector_len].*;
        const y0 = bf16VecToF32(y[i..][0..vector_len].*);
        const x1: Vf32 = x[i + vector_len ..][0..vector_len].*;
        const y1 = bf16VecToF32(y[i + vector_len ..][0..vector_len].*);
        const x2: Vf32 = x[i + 2 * vector_len ..][0..vector_len].*;
        const y2 = bf16VecToF32(y[i + 2 * vector_len ..][0..vector_len].*);
        const x3: Vf32 = x[i + 3 * vector_len ..][0..vector_len].*;
        const y3 = bf16VecToF32(y[i + 3 * vector_len ..][0..vector_len].*);
        acc0 = @mulAdd(Vf32, x0, y0, acc0);
        acc1 = @mulAdd(Vf32, x1, y1, acc1);
        acc2 = @mulAdd(Vf32, x2, y2, acc2);
        acc3 = @mulAdd(Vf32, x3, y3, acc3);
    }
    while (i + vector_len <= x.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        const yv = bf16VecToF32(y[i..][0..vector_len].*);
        acc0 = @mulAdd(Vf32, xv, yv, acc0);
    }

    var sum: f32 = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    while (i < x.len) : (i += 1) {
        sum += x[i] * dtype_mod.bf16ToF32(y[i]);
    }
    return sum;
}


fn gemmNNRangeF64(cd: []f64, ad: []const f64, bd: []const f64, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    _ = m;
    for (row_start..row_end) |i| {
        var j: usize = 0;
        while (j + vector_len_f64 <= n) : (j += vector_len_f64) {
            var acc: Vf64 = @splat(0);
            for (0..k) |p| {
                acc += @as(Vf64, @splat(ad[i * k + p])) * @as(Vf64, bd[p * n + j ..][0..vector_len_f64].*);
            }
            cd[i * n + j ..][0..vector_len_f64].* = acc;
        }
        while (j < n) : (j += 1) {
            var acc: f64 = 0;
            for (0..k) |p| acc += ad[i * k + p] * bd[p * n + j];
            cd[i * n + j] = acc;
        }
    }
}

fn gemmNNRangeF16(cd: []f16, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    _ = m;
    if (row_start == row_end or n == 0) return;
    if (k == 0) {
        for (row_start..row_end) |i| {
            @memset(cd[i * n .. (i + 1) * n], 0);
        }
        return;
    }

    var i = row_start;
    while (i + 12 <= row_end) : (i += 12) {
        gemmNNRows12F16(cd, ad, bd, i, n, k);
    }
    while (i + 8 <= row_end) : (i += 8) {
        gemmNNRows8F16(cd, ad, bd, i, n, k);
    }
    while (i + 4 <= row_end) : (i += 4) {
        gemmNNRows4F16(cd, ad, bd, i, n, k);
    }
    while (i < row_end) : (i += 1) {
        gemmNNRowF16(cd, ad, bd, i, n, k);
    }
}

// Column-sliced f16 NN kernel: computes columns [col_start, col_end) for all m
// rows. Row stride stays `n`. Used for small-m (< vector_column_min_m) f16
// matmuls where row-parallelism is denied; rows are tiled to reuse each loaded
// B vector across the tile (mirrors the gemmNNRows*F16 inner structure).
inline fn gemmNNRowsColsF16(
    comptime R: usize,
    cd: []f16,
    ad: []const f16,
    bd: []const f16,
    row: usize,
    n: usize,
    k: usize,
    col_start: usize,
    col_end: usize,
) void {
    var j = col_start;
    while (j + 2 * vector_len <= col_end) : (j += 2 * vector_len) {
        var acc: [R][2]Vf32 = undefined;
        inline for (0..R) |r| {
            acc[r][0] = @splat(0);
            acc[r][1] = @splat(0);
        }
        for (0..k) |p| {
            const b0: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j ..][0..vector_len].*));
            const b1: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j + vector_len ..][0..vector_len].*));
            inline for (0..R) |r| {
                const a: Vf32 = @splat(@as(f32, @floatCast(ad[(row + r) * k + p])));
                acc[r][0] += a * b0;
                acc[r][1] += a * b1;
            }
        }
        inline for (0..R) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc[r][0]));
            cd[(row + r) * n + j + vector_len ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc[r][1]));
        }
    }
    while (j + vector_len <= col_end) : (j += vector_len) {
        var acc: [R]Vf32 = undefined;
        inline for (0..R) |r| acc[r] = @splat(0);
        for (0..k) |p| {
            const b: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j ..][0..vector_len].*));
            inline for (0..R) |r| {
                acc[r] += @as(Vf32, @splat(@as(f32, @floatCast(ad[(row + r) * k + p])))) * b;
            }
        }
        inline for (0..R) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc[r]));
        }
    }
    while (j < col_end) : (j += 1) {
        var sums: [R]f32 = [_]f32{0} ** R;
        for (0..k) |p| {
            const b = @as(f32, @floatCast(bd[p * n + j]));
            inline for (0..R) |r| {
                sums[r] += @as(f32, @floatCast(ad[(row + r) * k + p])) * b;
            }
        }
        inline for (0..R) |r| cd[(row + r) * n + j] = @floatCast(sums[r]);
    }
}

fn gemmNNColsF16(cd: []f16, ad: []const f16, bd: []const f16, m: usize, n: usize, k: usize, col_start: usize, col_end: usize) void {
    if (col_start == col_end or m == 0) return;
    if (k == 0) {
        for (0..m) |i| {
            @memset(cd[i * n + col_start .. i * n + col_end], 0);
        }
        return;
    }
    var i: usize = 0;
    while (i + 8 <= m) : (i += 8) gemmNNRowsColsF16(8, cd, ad, bd, i, n, k, col_start, col_end);
    while (i + 4 <= m) : (i += 4) gemmNNRowsColsF16(4, cd, ad, bd, i, n, k, col_start, col_end);
    while (i < m) : (i += 1) gemmNNRowsColsF16(1, cd, ad, bd, i, n, k, col_start, col_end);
}

fn gemmNNRangeBf16(cd: []u16, ad: []const u16, bd: []const u16, m: usize, n: usize, k: usize, row_start: usize, row_end: usize) void {
    _ = m;
    if (row_start == row_end or n == 0) return;
    if (k == 0) {
        for (row_start..row_end) |i| {
            @memset(cd[i * n .. (i + 1) * n], 0);
        }
        return;
    }

    var i = row_start;
    while (i + 12 <= row_end) : (i += 12) {
        gemmNNRows12Bf16(cd, ad, bd, i, n, k);
    }
    while (i + 8 <= row_end) : (i += 8) {
        gemmNNRows8Bf16(cd, ad, bd, i, n, k);
    }
    while (i + 4 <= row_end) : (i += 4) {
        gemmNNRows4Bf16(cd, ad, bd, i, n, k);
    }
    while (i < row_end) : (i += 1) {
        gemmNNRowBf16(cd, ad, bd, i, n, k);
    }
}

inline fn gemmNNRows12F16(cd: []f16, ad: []const f16, bd: []const f16, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + 2 * vector_len <= n) : (j += 2 * vector_len) {
        var acc: [12][2]Vf32 = undefined;
        inline for (0..12) |r| {
            acc[r][0] = @splat(0);
            acc[r][1] = @splat(0);
        }

        for (0..k) |p| {
            const b0: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j ..][0..vector_len].*));
            const b1: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j + vector_len ..][0..vector_len].*));
            inline for (0..12) |r| {
                const a: Vf32 = @splat(@as(f32, @floatCast(ad[(row + r) * k + p])));
                acc[r][0] += a * b0;
                acc[r][1] += a * b1;
            }
        }

        inline for (0..12) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc[r][0]));
            cd[(row + r) * n + j + vector_len ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc[r][1]));
        }
    }
    while (j + vector_len <= n) : (j += vector_len) {
        var acc: [12]Vf32 = undefined;
        inline for (0..12) |r| {
            acc[r] = @splat(0);
        }
        for (0..k) |p| {
            const b: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j ..][0..vector_len].*));
            inline for (0..12) |r| {
                acc[r] += @as(Vf32, @splat(@as(f32, @floatCast(ad[(row + r) * k + p])))) * b;
            }
        }
        inline for (0..12) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc[r]));
        }
    }
    while (j < n) : (j += 1) {
        var sums: [12]f32 = [_]f32{0} ** 12;
        for (0..k) |p| {
            const b = @as(f32, @floatCast(bd[p * n + j]));
            inline for (0..12) |r| {
                sums[r] += @as(f32, @floatCast(ad[(row + r) * k + p])) * b;
            }
        }
        inline for (0..12) |r| {
            cd[(row + r) * n + j] = @floatCast(sums[r]);
        }
    }
}

inline fn gemmNNRows8F16(cd: []f16, ad: []const f16, bd: []const f16, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + 2 * vector_len <= n) : (j += 2 * vector_len) {
        var acc: [8][2]Vf32 = undefined;
        inline for (0..8) |r| {
            acc[r][0] = @splat(0);
            acc[r][1] = @splat(0);
        }

        for (0..k) |p| {
            const b0: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j ..][0..vector_len].*));
            const b1: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j + vector_len ..][0..vector_len].*));
            inline for (0..8) |r| {
                const a: Vf32 = @splat(@as(f32, @floatCast(ad[(row + r) * k + p])));
                acc[r][0] += a * b0;
                acc[r][1] += a * b1;
            }
        }

        inline for (0..8) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc[r][0]));
            cd[(row + r) * n + j + vector_len ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc[r][1]));
        }
    }
    while (j + vector_len <= n) : (j += vector_len) {
        var acc: [8]Vf32 = undefined;
        inline for (0..8) |r| {
            acc[r] = @splat(0);
        }
        for (0..k) |p| {
            const b: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j ..][0..vector_len].*));
            inline for (0..8) |r| {
                acc[r] += @as(Vf32, @splat(@as(f32, @floatCast(ad[(row + r) * k + p])))) * b;
            }
        }
        inline for (0..8) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc[r]));
        }
    }
    while (j < n) : (j += 1) {
        var sums: [8]f32 = [_]f32{0} ** 8;
        for (0..k) |p| {
            const b = @as(f32, @floatCast(bd[p * n + j]));
            inline for (0..8) |r| {
                sums[r] += @as(f32, @floatCast(ad[(row + r) * k + p])) * b;
            }
        }
        inline for (0..8) |r| {
            cd[(row + r) * n + j] = @floatCast(sums[r]);
        }
    }
}

inline fn gemmNNRows4F16(cd: []f16, ad: []const f16, bd: []const f16, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + 2 * vector_len <= n) : (j += 2 * vector_len) {
        var acc00: Vf32 = @splat(0);
        var acc01: Vf32 = @splat(0);
        var acc10: Vf32 = @splat(0);
        var acc11: Vf32 = @splat(0);
        var acc20: Vf32 = @splat(0);
        var acc21: Vf32 = @splat(0);
        var acc30: Vf32 = @splat(0);
        var acc31: Vf32 = @splat(0);

        for (0..k) |p| {
            const b0: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j ..][0..vector_len].*));
            const b1: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j + vector_len ..][0..vector_len].*));
            const a0: Vf32 = @splat(@as(f32, @floatCast(ad[(row + 0) * k + p])));
            const a1: Vf32 = @splat(@as(f32, @floatCast(ad[(row + 1) * k + p])));
            const a2: Vf32 = @splat(@as(f32, @floatCast(ad[(row + 2) * k + p])));
            const a3: Vf32 = @splat(@as(f32, @floatCast(ad[(row + 3) * k + p])));
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
            acc20 += a2 * b0;
            acc21 += a2 * b1;
            acc30 += a3 * b0;
            acc31 += a3 * b1;
        }

        cd[(row + 0) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc00));
        cd[(row + 0) * n + j + vector_len ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc01));
        cd[(row + 1) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc10));
        cd[(row + 1) * n + j + vector_len ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc11));
        cd[(row + 2) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc20));
        cd[(row + 2) * n + j + vector_len ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc21));
        cd[(row + 3) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc30));
        cd[(row + 3) * n + j + vector_len ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc31));
    }
    while (j + vector_len <= n) : (j += vector_len) {
        var acc0: Vf32 = @splat(0);
        var acc1: Vf32 = @splat(0);
        var acc2: Vf32 = @splat(0);
        var acc3: Vf32 = @splat(0);
        for (0..k) |p| {
            const b0: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j ..][0..vector_len].*));
            acc0 += @as(Vf32, @splat(@as(f32, @floatCast(ad[(row + 0) * k + p])))) * b0;
            acc1 += @as(Vf32, @splat(@as(f32, @floatCast(ad[(row + 1) * k + p])))) * b0;
            acc2 += @as(Vf32, @splat(@as(f32, @floatCast(ad[(row + 2) * k + p])))) * b0;
            acc3 += @as(Vf32, @splat(@as(f32, @floatCast(ad[(row + 3) * k + p])))) * b0;
        }
        cd[(row + 0) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc0));
        cd[(row + 1) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc1));
        cd[(row + 2) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc2));
        cd[(row + 3) * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc3));
    }
    while (j < n) : (j += 1) {
        var s0: f32 = 0;
        var s1: f32 = 0;
        var s2: f32 = 0;
        var s3: f32 = 0;
        for (0..k) |p| {
            const b = @as(f32, @floatCast(bd[p * n + j]));
            s0 += @as(f32, @floatCast(ad[(row + 0) * k + p])) * b;
            s1 += @as(f32, @floatCast(ad[(row + 1) * k + p])) * b;
            s2 += @as(f32, @floatCast(ad[(row + 2) * k + p])) * b;
            s3 += @as(f32, @floatCast(ad[(row + 3) * k + p])) * b;
        }
        cd[(row + 0) * n + j] = @floatCast(s0);
        cd[(row + 1) * n + j] = @floatCast(s1);
        cd[(row + 2) * n + j] = @floatCast(s2);
        cd[(row + 3) * n + j] = @floatCast(s3);
    }
}

inline fn gemmNNRowF16(cd: []f16, ad: []const f16, bd: []const f16, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + vector_len <= n) : (j += vector_len) {
        var acc: Vf32 = @splat(0);
        for (0..k) |p| {
            const a32: f32 = @floatCast(ad[row * k + p]);
            const b32: Vf32 = @floatCast(@as(Vf16ForF32, bd[p * n + j ..][0..vector_len].*));
            acc += @as(Vf32, @splat(a32)) * b32;
        }
        cd[row * n + j ..][0..vector_len].* = @as(Vf16ForF32, @floatCast(acc));
    }
    while (j < n) : (j += 1) {
        var acc: f32 = 0;
        for (0..k) |p| {
            acc += @as(f32, @floatCast(ad[row * k + p])) * @as(f32, @floatCast(bd[p * n + j]));
        }
        cd[row * n + j] = @floatCast(acc);
    }
}

inline fn gemmNNRows12Bf16(cd: []u16, ad: []const u16, bd: []const u16, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + 2 * vector_len <= n) : (j += 2 * vector_len) {
        var acc: [12][2]Vf32 = undefined;
        inline for (0..12) |r| {
            acc[r][0] = @splat(0);
            acc[r][1] = @splat(0);
        }

        for (0..k) |p| {
            const b0 = bf16VecToF32(bd[p * n + j ..][0..vector_len].*);
            const b1 = bf16VecToF32(bd[p * n + j + vector_len ..][0..vector_len].*);
            inline for (0..12) |r| {
                const a: Vf32 = @splat(dtype_mod.bf16ToF32(ad[(row + r) * k + p]));
                acc[r][0] += a * b0;
                acc[r][1] += a * b1;
            }
        }

        inline for (0..12) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = f32VecToBf16(acc[r][0]);
            cd[(row + r) * n + j + vector_len ..][0..vector_len].* = f32VecToBf16(acc[r][1]);
        }
    }
    while (j + vector_len <= n) : (j += vector_len) {
        var acc: [12]Vf32 = undefined;
        inline for (0..12) |r| {
            acc[r] = @splat(0);
        }
        for (0..k) |p| {
            const b = bf16VecToF32(bd[p * n + j ..][0..vector_len].*);
            inline for (0..12) |r| {
                acc[r] += @as(Vf32, @splat(dtype_mod.bf16ToF32(ad[(row + r) * k + p]))) * b;
            }
        }
        inline for (0..12) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = f32VecToBf16(acc[r]);
        }
    }
    while (j < n) : (j += 1) {
        var sums: [12]f32 = [_]f32{0} ** 12;
        for (0..k) |p| {
            const b = dtype_mod.bf16ToF32(bd[p * n + j]);
            inline for (0..12) |r| {
                sums[r] += dtype_mod.bf16ToF32(ad[(row + r) * k + p]) * b;
            }
        }
        inline for (0..12) |r| {
            cd[(row + r) * n + j] = dtype_mod.f32ToBf16(sums[r]);
        }
    }
}

inline fn gemmNNRows8Bf16(cd: []u16, ad: []const u16, bd: []const u16, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + 2 * vector_len <= n) : (j += 2 * vector_len) {
        var acc: [8][2]Vf32 = undefined;
        inline for (0..8) |r| {
            acc[r][0] = @splat(0);
            acc[r][1] = @splat(0);
        }

        for (0..k) |p| {
            const b0 = bf16VecToF32(bd[p * n + j ..][0..vector_len].*);
            const b1 = bf16VecToF32(bd[p * n + j + vector_len ..][0..vector_len].*);
            inline for (0..8) |r| {
                const a: Vf32 = @splat(dtype_mod.bf16ToF32(ad[(row + r) * k + p]));
                acc[r][0] += a * b0;
                acc[r][1] += a * b1;
            }
        }

        inline for (0..8) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = f32VecToBf16(acc[r][0]);
            cd[(row + r) * n + j + vector_len ..][0..vector_len].* = f32VecToBf16(acc[r][1]);
        }
    }
    while (j + vector_len <= n) : (j += vector_len) {
        var acc: [8]Vf32 = undefined;
        inline for (0..8) |r| {
            acc[r] = @splat(0);
        }
        for (0..k) |p| {
            const b = bf16VecToF32(bd[p * n + j ..][0..vector_len].*);
            inline for (0..8) |r| {
                acc[r] += @as(Vf32, @splat(dtype_mod.bf16ToF32(ad[(row + r) * k + p]))) * b;
            }
        }
        inline for (0..8) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = f32VecToBf16(acc[r]);
        }
    }
    while (j < n) : (j += 1) {
        var sums: [8]f32 = [_]f32{0} ** 8;
        for (0..k) |p| {
            const b = dtype_mod.bf16ToF32(bd[p * n + j]);
            inline for (0..8) |r| {
                sums[r] += dtype_mod.bf16ToF32(ad[(row + r) * k + p]) * b;
            }
        }
        inline for (0..8) |r| {
            cd[(row + r) * n + j] = dtype_mod.f32ToBf16(sums[r]);
        }
    }
}

inline fn gemmNNRows4Bf16(cd: []u16, ad: []const u16, bd: []const u16, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + 2 * vector_len <= n) : (j += 2 * vector_len) {
        var acc00: Vf32 = @splat(0);
        var acc01: Vf32 = @splat(0);
        var acc10: Vf32 = @splat(0);
        var acc11: Vf32 = @splat(0);
        var acc20: Vf32 = @splat(0);
        var acc21: Vf32 = @splat(0);
        var acc30: Vf32 = @splat(0);
        var acc31: Vf32 = @splat(0);

        for (0..k) |p| {
            const b0 = bf16VecToF32(bd[p * n + j ..][0..vector_len].*);
            const b1 = bf16VecToF32(bd[p * n + j + vector_len ..][0..vector_len].*);
            const a0: Vf32 = @splat(dtype_mod.bf16ToF32(ad[(row + 0) * k + p]));
            const a1: Vf32 = @splat(dtype_mod.bf16ToF32(ad[(row + 1) * k + p]));
            const a2: Vf32 = @splat(dtype_mod.bf16ToF32(ad[(row + 2) * k + p]));
            const a3: Vf32 = @splat(dtype_mod.bf16ToF32(ad[(row + 3) * k + p]));
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
            acc20 += a2 * b0;
            acc21 += a2 * b1;
            acc30 += a3 * b0;
            acc31 += a3 * b1;
        }

        cd[(row + 0) * n + j ..][0..vector_len].* = f32VecToBf16(acc00);
        cd[(row + 0) * n + j + vector_len ..][0..vector_len].* = f32VecToBf16(acc01);
        cd[(row + 1) * n + j ..][0..vector_len].* = f32VecToBf16(acc10);
        cd[(row + 1) * n + j + vector_len ..][0..vector_len].* = f32VecToBf16(acc11);
        cd[(row + 2) * n + j ..][0..vector_len].* = f32VecToBf16(acc20);
        cd[(row + 2) * n + j + vector_len ..][0..vector_len].* = f32VecToBf16(acc21);
        cd[(row + 3) * n + j ..][0..vector_len].* = f32VecToBf16(acc30);
        cd[(row + 3) * n + j + vector_len ..][0..vector_len].* = f32VecToBf16(acc31);
    }
    while (j + vector_len <= n) : (j += vector_len) {
        var acc0: Vf32 = @splat(0);
        var acc1: Vf32 = @splat(0);
        var acc2: Vf32 = @splat(0);
        var acc3: Vf32 = @splat(0);
        for (0..k) |p| {
            const b0 = bf16VecToF32(bd[p * n + j ..][0..vector_len].*);
            acc0 += @as(Vf32, @splat(dtype_mod.bf16ToF32(ad[(row + 0) * k + p]))) * b0;
            acc1 += @as(Vf32, @splat(dtype_mod.bf16ToF32(ad[(row + 1) * k + p]))) * b0;
            acc2 += @as(Vf32, @splat(dtype_mod.bf16ToF32(ad[(row + 2) * k + p]))) * b0;
            acc3 += @as(Vf32, @splat(dtype_mod.bf16ToF32(ad[(row + 3) * k + p]))) * b0;
        }
        cd[(row + 0) * n + j ..][0..vector_len].* = f32VecToBf16(acc0);
        cd[(row + 1) * n + j ..][0..vector_len].* = f32VecToBf16(acc1);
        cd[(row + 2) * n + j ..][0..vector_len].* = f32VecToBf16(acc2);
        cd[(row + 3) * n + j ..][0..vector_len].* = f32VecToBf16(acc3);
    }
    while (j < n) : (j += 1) {
        var s0: f32 = 0;
        var s1: f32 = 0;
        var s2: f32 = 0;
        var s3: f32 = 0;
        for (0..k) |p| {
            const b = dtype_mod.bf16ToF32(bd[p * n + j]);
            s0 += dtype_mod.bf16ToF32(ad[(row + 0) * k + p]) * b;
            s1 += dtype_mod.bf16ToF32(ad[(row + 1) * k + p]) * b;
            s2 += dtype_mod.bf16ToF32(ad[(row + 2) * k + p]) * b;
            s3 += dtype_mod.bf16ToF32(ad[(row + 3) * k + p]) * b;
        }
        cd[(row + 0) * n + j] = dtype_mod.f32ToBf16(s0);
        cd[(row + 1) * n + j] = dtype_mod.f32ToBf16(s1);
        cd[(row + 2) * n + j] = dtype_mod.f32ToBf16(s2);
        cd[(row + 3) * n + j] = dtype_mod.f32ToBf16(s3);
    }
}

inline fn gemmNNRowBf16(cd: []u16, ad: []const u16, bd: []const u16, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + vector_len <= n) : (j += vector_len) {
        var acc: Vf32 = @splat(0);
        for (0..k) |p| {
            const a32 = dtype_mod.bf16ToF32(ad[row * k + p]);
            const b32 = bf16VecToF32(bd[p * n + j ..][0..vector_len].*);
            acc += @as(Vf32, @splat(a32)) * b32;
        }
        cd[row * n + j ..][0..vector_len].* = f32VecToBf16(acc);
    }
    while (j < n) : (j += 1) {
        var acc: f32 = 0;
        for (0..k) |p| {
            acc += dtype_mod.bf16ToF32(ad[row * k + p]) * dtype_mod.bf16ToF32(bd[p * n + j]);
        }
        cd[row * n + j] = dtype_mod.f32ToBf16(acc);
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
