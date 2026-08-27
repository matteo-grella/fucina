//! The CBLAS provider seam: the one `extern fn cblas_sgemm`, the vendor
//! thread-count setters, the once-only thread configuration, and the MKL
//! nested-scope serialization live here, behind two call spellings
//! (`gemm` over an `ops.MatmulKind` orientation with derived leading
//! dimensions, `gemmStrided` with every leading dimension explicit) plus
//! the dimension gate (`fitsCblas`) and the Accelerate packed-kernel
//! preference. Vendor variation is one comptime switch on
//! `build_options.blas_kind`. `-Dblas=none` builds never analyze the
//! externs: entries open with a comptime `available` guard and callers
//! comptime-gate on it.
const std = @import("std");
const build_options = @import("build_options");
const ops = @import("ops.zig");
const thread = @import("../thread.zig");

/// BLAS-backed build (`-Dblas` other than `none`).
pub const available = build_options.use_blas;

const cblas_row_major: c_int = 101;
const cblas_no_trans: c_int = 111;
const cblas_trans: c_int = 112;
const max_cblas_dim: usize = @intCast(std.math.maxInt(c_int));

var threads_config_done = std.atomic.Value(bool).init(false);
var threads_config_mutex: thread.Mutex = .{};

extern fn cblas_sgemm(
    order: c_int,
    trans_a: c_int,
    trans_b: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: [*]const f32,
    lda: c_int,
    b: [*]const f32,
    ldb: c_int,
    beta: f32,
    c: [*]f32,
    ldc: c_int,
) void;

extern fn openblas_set_num_threads(num_threads: c_int) void;
extern fn bli_thread_set_num_threads(num_threads: c_int) void;
// MKL's C entry points are the capitalized names. The lowercase
// `mkl_set_num_threads` is the Fortran binding: it takes its argument by
// reference, so calling it by value dereferences the count.
extern fn MKL_Set_Num_Threads(num_threads: c_int) void;
// Per-thread override; returns the previous value for the calling thread,
// where 0 means "follow the global setting".
extern fn MKL_Set_Num_Threads_Local(num_threads: c_int) c_int;
extern fn nvpl_blas_set_num_threads(num_threads: c_int) void;

fn cblasTrans(transposed: bool) c_int {
    return if (transposed) cblas_trans else cblas_no_trans;
}

fn cDim(value: usize) c_int {
    return @intCast(value);
}

/// Leading dimensions of the row-major operands for one orientation.
fn ldA(kind: ops.MatmulKind, m: usize, k: usize) usize {
    return if (kind == .trans_a) m else k;
}

fn ldB(kind: ops.MatmulKind, n: usize, k: usize) usize {
    return if (kind == .trans_b) k else n;
}

/// Every dimension fits `c_int` (the cblas argument type).
pub fn fitsCblas(m: usize, n: usize, k: usize) bool {
    return m <= max_cblas_dim and n <= max_cblas_dim and k <= max_cblas_dim;
}

/// With an already-packed dense RHS, skinny-m tall-k cells run faster on the
/// in-tree packed microkernel than through BLAS: measured on M1 Max
/// (Accelerate) at k in {4800, 5120, 9600}, m=16 runs 1.8-2.1x BLAS and
/// m=32 sits at parity-to-1.15x, while at k=64 (the wide-n unembed shape)
/// BLAS wins from m=16 up — hence the k floor. Scoped to Accelerate: the
/// OpenBLAS/x86 crossover is unmeasured.
pub fn packedDenseKernelPreferred(m: usize, k: usize) bool {
    if (comptime build_options.blas_kind != .accelerate) return false;
    return m < 32 and k >= 4096;
}

/// Row-major C[m, n] = op(A)·op(B) + beta·C over the orientation `kind`
/// (`beta` 0 stores, 1 accumulates); the leading dimensions derive from
/// the orientation (contiguous operands). BLAS builds only.
pub fn gemm(
    kind: ops.MatmulKind,
    m: usize,
    n: usize,
    k: usize,
    a: []const f32,
    b: []const f32,
    beta: f32,
    c: []f32,
) void {
    if (comptime !available) unreachable;
    ensureThreadsConfigured();
    cblas_sgemm(
        cblas_row_major,
        cblasTrans(kind == .trans_a),
        cblasTrans(kind == .trans_b),
        cDim(m),
        cDim(n),
        cDim(k),
        1.0,
        a.ptr,
        cDim(ldA(kind, m, k)),
        b.ptr,
        cDim(ldB(kind, n, k)),
        beta,
        c.ptr,
        cDim(n),
    );
}

/// Raw strided sgemm for panel contractions (the quantized dequant-panel
/// crossovers, the exec-layer attention-backward strip contractions):
/// C[m,n] = alpha·op(A)·op(B) + beta·C, row-major, every leading dimension
/// explicit. BLAS builds only — callers comptime-gate on `available`.
pub fn gemmStrided(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: []const f32,
    lda: usize,
    b: []const f32,
    ldb: usize,
    beta: f32,
    c: []f32,
    ldc: usize,
) void {
    if (comptime !available) unreachable;
    ensureThreadsConfigured();
    cblas_sgemm(
        cblas_row_major,
        cblasTrans(trans_a),
        cblasTrans(trans_b),
        cDim(m),
        cDim(n),
        cDim(k),
        alpha,
        a.ptr,
        cDim(lda),
        b.ptr,
        cDim(ldb),
        beta,
        c.ptr,
        cDim(ldc),
    );
}

/// Pin the provider's thread count once, at the first GEMM, when
/// `-Dblas-threads` requests it (0 keeps the provider default). Vendor
/// variation is the one comptime switch below; Accelerate and the generic
/// `blas` link have no setter.
fn ensureThreadsConfigured() void {
    if (comptime build_options.blas_threads != 0) {
        if (threads_config_done.load(.acquire)) return;
        threads_config_mutex.lock();
        defer threads_config_mutex.unlock();
        if (!threads_config_done.load(.monotonic)) {
            configureThreads();
            threads_config_done.store(true, .release);
        }
    }
}

fn configureThreads() void {
    const requested = build_options.blas_threads;
    if (requested == 0) return;

    const max_threads: u32 = @intCast(std.math.maxInt(c_int));
    const n: c_int = @intCast(@min(requested, max_threads));
    switch (comptime build_options.blas_kind) {
        .openblas => openblas_set_num_threads(n),
        .blis => bli_thread_set_num_threads(n),
        .mkl => MKL_Set_Num_Threads(n),
        .nvpl => nvpl_blas_set_num_threads(n),
        .none, .accelerate, .blas => {},
    }
}

/// Token from `beginNestedScope`, restored by `endNestedScope`.
pub const NestedScope = c_int;

/// Serialize THIS thread's BLAS calls for the duration of a fucina parallel
/// region, returning the token that restores the previous setting.
///
/// A self-threading BLAS spawns an engine team per call, so one issued from
/// inside our own parallel region puts two schedulers on the same cores. The
/// limit belongs to the nested caller, not the process: a single big top-level
/// GEMM still wants the engine's own threads, and it outruns our pool-parallel
/// packed path when it gets them.
///
/// Only MKL is covered — `MKL_Set_Num_Threads_Local` is per-thread and hands
/// back the previous value, which is exactly this contract. Providers exposing
/// only process-wide setters keep the engine's default threading.
pub fn beginNestedScope() NestedScope {
    if (comptime build_options.blas_kind != .mkl) return 0;
    return MKL_Set_Num_Threads_Local(1);
}

pub fn endNestedScope(previous: NestedScope) void {
    if (comptime build_options.blas_kind != .mkl) return;
    _ = MKL_Set_Num_Threads_Local(previous);
}
