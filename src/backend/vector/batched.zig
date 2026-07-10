//! Batched dense GEMM relocated out of vector.zig: the batched NN/TN/NT entry
//! points, the BatchedTask struct, and the parallel batch dispatch. The inner
//! per-batch work reuses the dense GEMM range kernels (gemmNNRange/gemmTNRange/
//! gemmNTRange), aliased here from vector.zig (`vm`) along with the shared-core
//! ParallelConfig and batchedThreadCount, so the moved bodies compile unchanged.

const std = @import("std");
const parallel = @import("../../parallel.zig");
const tensor = @import("../../tensor.zig");
const thread = @import("../../thread.zig");
const vm = @import("common.zig");
const gemm = @import("gemm.zig");

const Tensor = tensor.Tensor;

// Shared-core symbols from the common leaf, aliased so the moved bodies compile
// unchanged.
const ParallelConfig = vm.ParallelConfig;
const batchedThreadCount = vm.batchedThreadCount;

// Dense GEMM range kernels — imported directly from the gemm sibling.
const gemmNNRange = gemm.gemmNNRange;
const gemmTNRange = gemm.gemmTNRange;
const gemmNTRange = gemm.gemmNTRange;

// ---------------- Batched GEMM ----------------

pub fn matmulBatched2DIntoUnchecked(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
) void {
    matmulBatched2DIntoUncheckedWithConfig(out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, .{});
}

pub fn matmulBatched2DIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
    config: ParallelConfig,
) void {
    if (batch_count == 0) return;
    const a_base = vm.contiguousDataConst(a, a.buffer.data.len - a.offset);
    const b_base = vm.contiguousDataConst(b, b.buffer.data.len - b.offset);
    const c_base = vm.contiguousData(out, out.buffer.data.len - out.offset);

    if (maybeParallelBatchedNN(config, c_base, a_base, b_base, m, n, k, batch_count, stride_a, stride_b, stride_c)) return;
    for (0..batch_count) |bi| {
        gemmNNRange(
            c_base[bi * stride_c .. bi * stride_c + m * n],
            a_base[bi * stride_a .. bi * stride_a + m * k],
            b_base[bi * stride_b .. bi * stride_b + k * n],
            m,
            n,
            k,
            0,
            m,
        );
    }
}

pub fn matmulBatchedTransA2DIntoUnchecked(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
) void {
    matmulBatchedTransA2DIntoUncheckedWithConfig(out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, .{});
}

pub fn matmulBatchedTransA2DIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
    config: ParallelConfig,
) void {
    if (batch_count == 0) return;
    const a_base = vm.contiguousDataConst(a, a.buffer.data.len - a.offset);
    const b_base = vm.contiguousDataConst(b, b.buffer.data.len - b.offset);
    const c_base = vm.contiguousData(out, out.buffer.data.len - out.offset);

    if (maybeParallelBatchedTN(config, c_base, a_base, b_base, m, n, k, batch_count, stride_a, stride_b, stride_c)) return;
    for (0..batch_count) |bi| {
        gemmTNRange(
            c_base[bi * stride_c .. bi * stride_c + m * n],
            a_base[bi * stride_a .. bi * stride_a + k * m],
            b_base[bi * stride_b .. bi * stride_b + k * n],
            m,
            n,
            k,
            0,
            m,
        );
    }
}

pub fn matmulBatchedTransB2DIntoUnchecked(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
) void {
    matmulBatchedTransB2DIntoUncheckedWithConfig(out, a, b, m, n, k, batch_count, stride_a, stride_b, stride_c, .{});
}

pub fn matmulBatchedTransB2DIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
    config: ParallelConfig,
) void {
    if (batch_count == 0) return;
    const a_base = vm.contiguousDataConst(a, a.buffer.data.len - a.offset);
    const b_base = vm.contiguousDataConst(b, b.buffer.data.len - b.offset);
    const c_base = vm.contiguousData(out, out.buffer.data.len - out.offset);

    if (maybeParallelBatchedNT(config, c_base, a_base, b_base, m, n, k, batch_count, stride_a, stride_b, stride_c)) return;
    for (0..batch_count) |bi| {
        gemmNTRange(
            c_base[bi * stride_c .. bi * stride_c + m * n],
            a_base[bi * stride_a .. bi * stride_a + m * k],
            b_base[bi * stride_b .. bi * stride_b + n * k],
            m,
            n,
            k,
            0,
            m,
        );
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

fn maybeParallelBatchedNN(
    config: ParallelConfig,
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
    const pool = config.pool orelse return false;
    const thread_count = batchedThreadCount(batch_count, m, n, k);
    if (thread_count == 1) return false;
    runParallelBatches(pool, runBatchedNNTask, c_base, a_base, b_base, m, n, k, batch_count, stride_a, stride_b, stride_c, thread_count);
    return true;
}

fn maybeParallelBatchedTN(
    config: ParallelConfig,
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
    const pool = config.pool orelse return false;
    const thread_count = batchedThreadCount(batch_count, m, n, k);
    if (thread_count == 1) return false;
    runParallelBatches(pool, runBatchedTNTask, c_base, a_base, b_base, m, n, k, batch_count, stride_a, stride_b, stride_c, thread_count);
    return true;
}

fn maybeParallelBatchedNT(
    config: ParallelConfig,
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
    const pool = config.pool orelse return false;
    const thread_count = batchedThreadCount(batch_count, m, n, k);
    if (thread_count == 1) return false;
    runParallelBatches(pool, runBatchedNTTask, c_base, a_base, b_base, m, n, k, batch_count, stride_a, stride_b, stride_c, thread_count);
    return true;
}

fn runParallelBatches(
    pool: *thread.Pool,
    comptime func: fn (*const BatchedTask) void,
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
    thread_count: usize,
) void {
    var tasks: [parallel.vector_max_threads]BatchedTask = undefined;

    for (0..thread_count) |ti| {
        const start = ti * batch_count / thread_count;
        const end = (ti + 1) * batch_count / thread_count;
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
            .batch_start = start,
            .batch_end = end,
        };
    }

    pool.parallelChunks(BatchedTask, tasks[0..thread_count], func);
}

fn runBatchedNNTask(task: *const BatchedTask) void {
    for (task.batch_start..task.batch_end) |bi| {
        gemmNNRange(
            task.c_base[bi * task.stride_c .. bi * task.stride_c + task.m * task.n],
            task.a_base[bi * task.stride_a .. bi * task.stride_a + task.m * task.k],
            task.b_base[bi * task.stride_b .. bi * task.stride_b + task.k * task.n],
            task.m,
            task.n,
            task.k,
            0,
            task.m,
        );
    }
}

fn runBatchedTNTask(task: *const BatchedTask) void {
    for (task.batch_start..task.batch_end) |bi| {
        gemmTNRange(
            task.c_base[bi * task.stride_c .. bi * task.stride_c + task.m * task.n],
            task.a_base[bi * task.stride_a .. bi * task.stride_a + task.k * task.m],
            task.b_base[bi * task.stride_b .. bi * task.stride_b + task.k * task.n],
            task.m,
            task.n,
            task.k,
            0,
            task.m,
        );
    }
}

fn runBatchedNTTask(task: *const BatchedTask) void {
    for (task.batch_start..task.batch_end) |bi| {
        gemmNTRange(
            task.c_base[bi * task.stride_c .. bi * task.stride_c + task.m * task.n],
            task.a_base[bi * task.stride_a .. bi * task.stride_a + task.m * task.k],
            task.b_base[bi * task.stride_b .. bi * task.stride_b + task.n * task.k],
            task.m,
            task.n,
            task.k,
            0,
            task.m,
        );
    }
}
