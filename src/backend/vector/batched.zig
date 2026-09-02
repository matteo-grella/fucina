//! Batched dense f32 GEMM: one `gemmBatched` entry over the orientation and
//! its parallel batch dispatch over `tile.forRange`. The inner per-batch
//! work reuses the dense GEMM range kernels (gemmNNRange/gemmTNRange/
//! gemmNTRange) from `gemm.zig`; ParallelConfig and batchedThreadCount
//! come from `common.zig`.

const isa = @import("../isa.zig");
const ops = @import("../ops.zig");
const common = @import("common.zig");
const tile = @import("tile.zig");
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
    const batches: Batches = .{
        .c_base = c_base,
        .a_base = a_base,
        .b_base = b_base,
        .m = m,
        .n = n,
        .k = k,
        .stride_a = stride_a,
        .stride_b = stride_b,
        .stride_c = stride_c,
    };
    const run = batchRange(kind);
    if (pc.pool) |pool| {
        const thread_count = common.batchedThreadCount(batch_count, m, n, k);
        if (thread_count > 1) return tile.forRange(pool, Batches, batches, batch_count, thread_count, run);
    }
    run(batches, 0, batch_count);
}

/// The operands and strides of one batched GEMM.
const Batches = struct {
    c_base: []f32,
    a_base: []const f32,
    b_base: []const f32,
    m: usize,
    n: usize,
    k: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
};

fn batchRange(comptime kind: MatmulKind) fn (Batches, usize, usize) void {
    return struct {
        fn go(b: Batches, batch_start: usize, batch_end: usize) void {
            for (batch_start..batch_end) |bi| batchGemm(kind, b, bi);
        }
    }.go;
}

/// One batch through the orientation's range kernel.
fn batchGemm(comptime kind: MatmulKind, b: Batches, bi: usize) void {
    const cd = b.c_base[bi * b.stride_c .. bi * b.stride_c + b.m * b.n];
    switch (kind) {
        .plain => gemm.gemmNNRange(cd, b.a_base[bi * b.stride_a .. bi * b.stride_a + b.m * b.k], b.b_base[bi * b.stride_b .. bi * b.stride_b + b.k * b.n], b.m, b.n, b.k, 0, b.m),
        .trans_a => gemm.gemmTNRange(cd, b.a_base[bi * b.stride_a .. bi * b.stride_a + b.k * b.m], b.b_base[bi * b.stride_b .. bi * b.stride_b + b.k * b.n], b.m, b.n, b.k, 0, b.m),
        .trans_b => gemm.gemmNTRange(cd, b.a_base[bi * b.stride_a .. bi * b.stride_a + b.m * b.k], b.b_base[bi * b.stride_b .. bi * b.stride_b + b.n * b.k], b.m, b.n, b.k, 0, b.m),
    }
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
