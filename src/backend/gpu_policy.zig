//! Shared GPU offload gate policy: the shape floors and work comparisons
//! both providers apply identically, as pure functions of (shape, the
//! provider's latched Q6 floors, `*const tuning.Table.Gpu`). Every number
//! compared here is a tuning-table leaf, so gate behaviour is
//! provider-independent wherever both providers implement the arm.
//!
//! What stays in the provider: residency probes (Metal page wraps vs CUDA
//! device registries), structural kernel limits (threadgroup budgets, head
//! caps), tracing, and provider-only arms (CUDA's transient-RHS floor and
//! decode opt-in). Work products saturate on u64 overflow: astronomically
//! large work passes the work floors but never bypasses the kill switch or
//! the shape floors.

const std = @import("std");
const gpu_provider = @import("gpu_provider.zig");
const tuning = @import("../tuning.zig");

const Fmt = gpu_provider.QuantFormat;

/// The GPU group of the process-wide tuning table (`tuning.get().gpu`).
pub const Gpu = tuning.Table.Gpu;

/// A work-floor gate's outcome; the providers count these in their traces
/// (`gate_shape` / `gate_below` / `gate_pass`).
pub const Decision = enum { shape, below, pass };

/// Saturating m·n·k.
pub fn work(m: usize, n: usize, k: usize) u64 {
    const mn = std.math.mul(u64, m, n) catch return std.math.maxInt(u64);
    return std.math.mul(u64, mn, k) catch std.math.maxInt(u64);
}

/// Saturating m·n·k·batch.
pub fn workBatched(m: usize, n: usize, k: usize, batch_count: usize) u64 {
    return std.math.mul(u64, work(m, n, k), batch_count) catch std.math.maxInt(u64);
}

/// The dense-GEMM shape floor: below it the tile kernels cannot win on
/// either provider.
pub fn belowShapeFloor(m: usize, n: usize, k: usize) bool {
    return m < 32 or n < 32 or k < 16;
}

fn decide(t: *const Gpu, m: usize, n: usize, k: usize, w: u64, floor: u64) Decision {
    if (belowShapeFloor(m, n, k)) return .shape;
    return if (t.enabled and w >= floor) .pass else .below;
}

/// Base f32 GEMM gate (`min_work.base`); the caller supplies the work
/// product so the batched form prices in its batch. CUDA layers its
/// transient-RHS floor on a `.pass`.
pub fn f32Gate(t: *const Gpu, m: usize, n: usize, k: usize, w: u64) Decision {
    return decide(t, m, n, k, w, t.min_work.base);
}

/// f16 GEMM gate (`min_work.f16`): same shape floor, lower crossover — the
/// CPU f16 competitor has no AMX-class arm and 16-bit operands halve the
/// transfer bytes.
pub fn f16Gate(t: *const Gpu, m: usize, n: usize, k: usize) Decision {
    return decide(t, m, n, k, work(m, n, k), t.min_work.f16);
}

/// Small-m dense-f32 GEMV/GEMM eligibility (`m <= 8`, wide RHS, the
/// `min_work.gemv` floor). Residency — the part that makes the small shape
/// legal at all — is the provider's own check on a `true`.
pub fn gemvEligible(t: *const Gpu, m: usize, n: usize, k: usize) bool {
    if (m == 0 or m > 8 or n < 256 or k < 256) return false;
    return t.enabled and work(m, n, k) >= t.min_work.gemv;
}

/// Attention-forward work floor in q·kv·heads·d units (`min_work.attn`).
/// Structural kernel limits (score-span budgets, head caps) stay
/// provider-side.
pub fn attnWork(t: *const Gpu, q_seq: usize, kv_seq: usize, heads: usize, d: usize) bool {
    const qk = std.math.mul(u64, q_seq, kv_seq) catch std.math.maxInt(u64);
    const hd = std.math.mul(u64, heads, d) catch std.math.maxInt(u64);
    const w = std.math.mul(u64, qk, hd) catch std.math.maxInt(u64);
    return t.enabled and w >= t.min_work.attn;
}

/// Grouped-MoE work gate (total m·n·k across both projections of a layer,
/// `min_work.qmoe`).
pub fn qmoeWork(t: *const Gpu, total_work: u64) bool {
    return t.enabled and total_work >= t.min_work.qmoe;
}

/// Occupancy arm of the grouped-MoE gate: `rows` real panel rows spread
/// over `n_tiles` 32-row token tiles must reach the configured minimum fill
/// percentage (`qmoe_min_fill`). Callers pass the exact tile table they are
/// about to dispatch.
pub fn qmoeFillAcceptable(t: *const Gpu, rows: usize, n_tiles: usize) bool {
    if (n_tiles == 0) return false;
    const filled = std.math.mul(u64, @as(u64, rows), 100) catch return true;
    return filled >= @as(u64, n_tiles) * 32 * t.qmoe_min_fill;
}

/// Compact/raw-tier dense-quant gate: Q6_K rides its derived floor, Q5_K
/// its own measured leaf, and the remaining formats the qmoe floor.
/// Formats a provider has no kernel for never reach this (`supportsQuant`
/// gates first; the provider asserts).
pub fn denseQuantWork(t: *const Gpu, q6: tuning.GpuQ6Floors, format: Fmt, total_work: u64) bool {
    const floor = switch (format) {
        .q6_k => q6.dense_q6,
        .q5_k => t.min_work.dense_q5,
        .q4_k, .q8_0, .tq2_0, .tq2_0_folded => t.min_work.qmoe,
    };
    return t.enabled and total_work >= floor;
}

/// Packed-tier dense-quant gate against the load-time-packed CPU fallback:
/// per-format measured floors, Q6_K from the derived pair.
pub fn denseQuantPackedWork(t: *const Gpu, q6: tuning.GpuQ6Floors, format: Fmt, total_work: u64) bool {
    const floor = switch (format) {
        .q4_k => t.min_work.dense_q4,
        .q5_k => t.min_work.dense_q5,
        .q6_k => q6.packed_q6,
        .q8_0 => t.min_work.dense_q8,
        .tq2_0, .tq2_0_folded => t.min_work.dense_tq2,
    };
    return t.enabled and total_work >= floor;
}

test "policy gates compare the table's numbers" {
    const t: Gpu = .{};
    // Shape floor rejects before any work comparison.
    try std.testing.expectEqual(Decision.shape, f32Gate(&t, 31, 4096, 4096, work(31, 4096, 4096)));
    try std.testing.expectEqual(Decision.shape, f16Gate(&t, 32, 31, 4096));
    // The base floor separates below/pass at exactly min_work.base.
    var big: Gpu = .{};
    big.min_work.base = 1 << 10;
    try std.testing.expectEqual(Decision.pass, f32Gate(&big, 32, 32, 16, work(32, 32, 16)));
    big.min_work.base = (1 << 14) + 1;
    try std.testing.expectEqual(Decision.below, f32Gate(&big, 32, 32, 16, work(32, 32, 16)));
    // The kill switch wins over saturated work.
    var off: Gpu = .{};
    off.enabled = false;
    try std.testing.expectEqual(Decision.below, f32Gate(&off, 1 << 22, 1 << 22, 1 << 22, workBatched(1 << 22, 1 << 22, 1 << 22, 2)));
    try std.testing.expect(!attnWork(&off, 1 << 32, 1 << 32, 64, 128));
    // qmoe fill arithmetic: 2048 rows over 128 tiles is exactly 50%.
    try std.testing.expect(qmoeFillAcceptable(&t, 2048, 128));
    try std.testing.expect(!qmoeFillAcceptable(&t, 2047, 128));
    try std.testing.expect(!qmoeFillAcceptable(&t, 0, 0));
}
