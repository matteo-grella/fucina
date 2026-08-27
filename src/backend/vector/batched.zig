//! Batched dense f32 GEMM: one `gemmBatched` entry over the orientation, the
//! BatchedTask struct, and the parallel batch dispatch. The inner per-batch
//! work reuses the dense GEMM range kernels (gemmNNRange/gemmTNRange/
//! gemmNTRange) from `gemm.zig`; shared-core ParallelConfig and
//! batchedThreadCount come from `common.zig`.

const std = @import("std");
const isa = @import("../isa.zig");
const ops = @import("../ops.zig");
const parallel = @import("../../parallel.zig");
const thread = @import("../../thread.zig");
const common = @import("common.zig");
const gemm = @import("gemm.zig");

const MatmulKind = ops.MatmulKind;

/// `batch_count` GEMMs over contiguous base slices. Strides are in elements;
/// a stride of 0 means "shared across all batches", which makes a broadcast
/// operand just another stride value.
pub fn gemmBatched(
    pc: common.ParallelConfig,
    comptime kind: MatmulKind,
    c_base: []f32,
    a_base: []const f32,
    b_base: []const f32,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
) void {
    if (batch_count == 0) return;
    if (comptime isa.reference) return scalar.gemmBatched(kind, c_base, a_base, b_base, m, n, k, batch_count, stride_a, stride_b, stride_c);
    if (maybeParallelBatched(kind, pc, c_base, a_base, b_base, m, n, k, batch_count, stride_a, stride_b, stride_c)) return;
    for (0..batch_count) |bi| {
        batchGemm(kind, c_base, a_base, b_base, m, n, k, stride_a, stride_b, stride_c, bi);
    }
}

/// One batch through the orientation's range kernel.
fn batchGemm(
    comptime kind: MatmulKind,
    c_base: []f32,
    a_base: []const f32,
    b_base: []const f32,
    m: usize,
    n: usize,
    k: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
    bi: usize,
) void {
    const cd = c_base[bi * stride_c .. bi * stride_c + m * n];
    switch (kind) {
        .plain => gemm.gemmNNRange(cd, a_base[bi * stride_a .. bi * stride_a + m * k], b_base[bi * stride_b .. bi * stride_b + k * n], m, n, k, 0, m),
        .trans_a => gemm.gemmTNRange(cd, a_base[bi * stride_a .. bi * stride_a + k * m], b_base[bi * stride_b .. bi * stride_b + k * n], m, n, k, 0, m),
        .trans_b => gemm.gemmNTRange(cd, a_base[bi * stride_a .. bi * stride_a + m * k], b_base[bi * stride_b .. bi * stride_b + n * k], m, n, k, 0, m),
    }
}

// ---------------- Inner kernels ----------------

const BatchedTask = struct {
    c_base: []f32,
    a_base: []const f32,
    b_base: []const f32,
    m: usize,
    n: usize,
    k: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
    batch_start: usize,
    batch_end: usize,
};

fn maybeParallelBatched(
    comptime kind: MatmulKind,
    pc: common.ParallelConfig,
    c_base: []f32,
    a_base: []const f32,
    b_base: []const f32,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
) bool {
    const pool = pc.pool orelse return false;
    const thread_count = common.batchedThreadCount(batch_count, m, n, k);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]BatchedTask = undefined;
    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .c_base = c_base,
            .a_base = a_base,
            .b_base = b_base,
            .m = m,
            .n = n,
            .k = k,
            .stride_a = stride_a,
            .stride_b = stride_b,
            .stride_c = stride_c,
            .batch_start = ti * batch_count / thread_count,
            .batch_end = (ti + 1) * batch_count / thread_count,
        };
    }
    pool.parallelChunks(BatchedTask, tasks[0..thread_count], BatchedRun(kind).run);
    return true;
}

fn BatchedRun(comptime kind: MatmulKind) type {
    return struct {
        fn run(task: *const BatchedTask) void {
            for (task.batch_start..task.batch_end) |bi| {
                batchGemm(kind, task.c_base, task.a_base, task.b_base, task.m, task.n, task.k, task.stride_a, task.stride_b, task.stride_c, bi);
            }
        }
    };
}

// ---------------- The scalar reference arm ----------------

/// The scalar reference twin of the batched GEMM entry: a serial triple
/// loop per batch. On `-Dbackend=scalar` builds the entry above dispatches
/// here; on native builds the twin stays reachable for
/// `backend/parity_test.zig` and `bench/backend.zig`.
pub const scalar = struct {
    pub fn gemmBatched(
        comptime kind: MatmulKind,
        c_base: []f32,
        a_base: []const f32,
        b_base: []const f32,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
    ) void {
        for (0..batch_count) |bi| {
            const ai = a_base[bi * stride_a .. bi * stride_a + m * k];
            const bs = b_base[bi * stride_b .. bi * stride_b + k * n];
            const ci = c_base[bi * stride_c .. bi * stride_c + m * n];
            for (0..m) |i| {
                for (0..n) |j| {
                    var acc: f32 = 0;
                    for (0..k) |p| {
                        const av = switch (kind) {
                            .plain, .trans_b => ai[i * k + p],
                            .trans_a => ai[p * m + i],
                        };
                        const bv = switch (kind) {
                            .plain, .trans_a => bs[p * n + j],
                            .trans_b => bs[j * k + p],
                        };
                        acc += av * bv;
                    }
                    ci[i * n + j] = acc;
                }
            }
        }
    }
};
