//! Parallel-dispatch policy: the COMPTIME CPU crossover constants kernels
//! consult before engaging the worker pool (`vector_*_threshold`,
//! `row_kernel_*`, `backward_*`, ...; their runtime counterparts are
//! `src/tuning.zig`'s table, and the two files split one policy surface on
//! binding time), the worker-team size (`cpuThreadCount`, sized from the
//! `topology` probes and capped by `FUCINA_MAX_THREADS`, the one env
//! variable read below the tuning table because it sizes the substrate the
//! table's loader runs on), and the two part-count helpers. The actual
//! chunking is `thread.Pool.parallelChunks`'s. The probes and the env
//! readers are the leaves `cpu_topology.zig` and `env.zig`, re-exported
//! here as `topology` and `env`. Layer stack: docs/ARCHITECTURE.md.
const std = @import("std");
const build_options = @import("build_options");

/// CPU topology probes (`src/cpu_topology.zig`): physical and performance
/// cores, the cgroup budget, the schedulable count.
pub const topology = @import("cpu_topology.zig");
/// The sanctioned environment readers (`src/env.zig`).
pub const env = @import("env.zig");

// Comptime ceiling for the worker team and stack-allocated task arrays, AND the
// runtime default team size. Set at build time via -Dmax-threads=N (1-64); the
// default 8 is the performance-core count on the primary Apple Silicon target
// (M1 Max). Servers with more cores must raise the ceiling at build time (e.g.
// -Dmax-threads=32) — the FUCINA_MAX_THREADS env var only lowers the count at
// runtime, never raises it past this ceiling. NOTE: the best thread count is
// workload- and thermal-dependent. Measured on M1 Max across qwen3 / qwen3.5 /
// qwen3moe / gemma4: prefill is fastest at 8 cores when cool but at ~6 when
// heat-soaked, while decode on small/mid models is ~8-14% faster at 6
// (saturating all P-cores trips DVFS throttle and leaves no OS/spin-wait
// slack). No single value wins everywhere, so the default stays at the
// all-P-core ceiling (best cold prefill — the metric chased vs llama's AMX
// path) and the FUCINA_MAX_THREADS env var (mirrors llama.cpp's -t) drops it
// to e.g. 6 for decode-heavy or sustained workloads.
pub const vector_max_threads: usize = build_options.max_threads;
pub const vector_elementwise_len_threshold: usize = 256 * 1024;
/// Fused-row-kernel parallel gate (softmax/norm/loss rows, quantized row
/// passes): these engage the pool at HALF the plain-elementwise crossover.
/// The ratio is policy in ONE place — retuning the base retunes every row
/// gate with it, deliberately.
pub const row_kernel_len_threshold: usize = vector_elementwise_len_threshold / 2;
/// Fused op-chain walk gate (splitGated forward rows, the fused
/// activation+quantize passes, the fused rmsNorm-mul-rope walk): several
/// ops per element, so the pool pays at an EIGHTH of the
/// plain-elementwise crossover. Same one-place ratio policy as
/// `row_kernel_len_threshold`.
pub const fused_chain_len_threshold: usize = vector_elementwise_len_threshold / 8;
/// Split gated-activation BACKWARD row gate (splitGlu/splitSwiGlu VJPs):
/// the pool pays at a QUARTER of the plain-elementwise crossover (fewer
/// fused ops per element than the forward chains above). Same one-place
/// ratio policy.
pub const split_backward_len_threshold: usize = vector_elementwise_len_threshold / 4;
pub const materialize_parallel_len_threshold: usize = 256 * 1024;
pub const materialize_parallel_min_chunk: usize = 64 * 1024;
pub const vector_matmul_work_threshold: usize = 1024 * 1024;
/// Attention parallel gate: attention kernels engage the pool at HALF the
/// GEMM work crossover. Same one-place ratio policy as
/// `row_kernel_len_threshold`.
pub const attention_work_threshold: usize = vector_matmul_work_threshold / 2;
pub const vector_batched_work_threshold: usize = 2 * 1024 * 1024;
pub const vector_column_min_m: usize = 32;
pub const vector_column_min_n: usize = 128;
pub const vector_column_work_multiplier: usize = 1;
pub const vector_column_chunk: usize = 64;

pub const backward_matmul_work_threshold: usize = 262_144;
pub const backward_async_work_threshold: usize = 256 * 1024 * 1024;
/// Per-batch m·n·k below which the BLAS arm of the batched GEMM splits the
/// batches over the worker team (each range running its batches through
/// BLAS on its own thread); at or above it the batches run sequentially and
/// BLAS threads each one itself. Measured on M1 Max with Accelerate: the
/// split wins up to 33M per batch (8x512x128x512: 370 vs 497 us) and loses
/// at 1G (4x1024^3: 5.96 vs 4.38 ms).
pub const blas_batch_split_max_work: usize = 256 * 1024 * 1024;

// Quantized-RHS dispatch gates of `backend/native.zig`: comptime
// crossovers like the ones above, kept here so the whole policy table is
// one place (native.zig aliases them file-locally). Values and their
// measurement rationale moved verbatim from native.zig.

/// Stack budget (in Q8_0 blocks) for the per-call LHS-quantization
/// scratch of the quantized-RHS dispatch tier: decode-shaped calls stay
/// heap-free, larger LHS rows take the caller-supplied allocator.
pub const q8_0_lhs_stack_blocks: usize = 512;
/// Off-multiple-m row minimum of the q4_k x4 fast path. q4_k pads the
/// final partial row group inside the x4 kernel, so every m >= 4 takes it
/// (one pass over the packed weights).
pub const q4_k_x4_min_rows: usize = 4;
/// q5_k has no padded-group kernel: its bulk+tail split re-reads the
/// packed weights once more for the 1-3 remainder rows, so below 128 rows
/// the one-pass per-row path wins.
pub const q5_k_x4_prefix_min_rows: usize = 128;
/// Prefill row count at/above which the Q2_0 matmul dequantizes weight
/// panels to f32 and rides BLAS (Accelerate AMX / OpenBLAS): the dequant
/// pass costs O(n*k) regardless of m, so its amortization — and the GEMM's
/// O(m) operand reuse, out of the int8 sdot path's reach on AMX-class
/// units — grows with m, while below the threshold (decode, short bursts)
/// the int8 mul-free path wins. Same split llama.cpp's BLAS backend makes
/// for its quantized prefill. The BLAS arm consumes exact f32 activations
/// (no Q8_0 LHS quantization), so its numerics differ from the int path
/// exactly as the dense-f32 BLAS GEMMs already do from the scalar backend.
pub const q2_0_blas_min_m: usize = 192;
/// f32 scratch budget for one dequantized weight panel (48 MiB). Panels
/// slice the CONTRACT dimension, never the output dimension: every GEMM is
/// then full-width with a contiguous C (accumulating across slices via
/// beta=1), where output-dimension panels would give narrow GEMMs writing
/// a strided C — a shape BLAS handles poorly.
pub const q2_0_blas_panel_floats: usize = 12 * 1024 * 1024;
/// Prefill row count at/above which the table-decoded formats (iq*/fp4)
/// dequantize weight panels to f32 and ride BLAS, exactly like the Q2_0
/// arm. Their int path pays a per-block table decode per (weight-row,
/// LHS-row) pair, so the dequant-once panel amortizes even earlier than
/// Q2_0's mul-free path — same accepted-numerics stance: the BLAS arm
/// consumes exact f32 activations, the int kernels keep decode and short
/// bursts (and every bitwise contract).
pub const table_blas_min_m: usize = 64;
/// Prefill row count at/above which the folded tied-K=2 PTQTP path
/// (`tq2_0_fx4`) dequantizes weight panels to f32 and rides BLAS. The
/// mul-free ternary tile owns decode and short bursts — it beats a GEMM
/// there — but its 4-column pack has no AMX-class batch form, so past this
/// width the dequant-once panel plus sgemm wins on operand reuse. Accepted
/// numerics, like every BLAS arm: exact f32 activations instead of the
/// Q8_K-quantized ones the integer kernel consumes.
pub const folded_blas_min_m: usize = 64;

var cached_cpu_count = std.atomic.Value(usize).init(0);

/// Runtime worker-count override (mirrors llama.cpp `-t` / the cli's
/// `set_num_threads`). Sets the cached CPU count so `cpuThreadCount` returns
/// `min(n, max_threads)` thereafter. Call once at startup before any parallel
/// work. `n == 0` is ignored. Equivalent to `FUCINA_MAX_THREADS` but settable
/// from a CLI flag.
pub fn setMaxThreads(n: usize) void {
    if (n >= 1) cached_cpu_count.store(n, .release);
}

pub fn cpuThreadCount(max_threads: usize) usize {
    var count = cached_cpu_count.load(.acquire);
    if (count == 0) {
        count = std.Thread.getCpuCount() catch 1;
        if (count == 0) count = 1;
        // SMT machines double-book cores in the logical count, and an
        // HT-oversubscribed team collapses throughput (i9-13950HX: a
        // 16-worker team pinned to 8 P-cores' hyperthreads ran 19s of
        // prefill in 43s — the x86 threading finding in docs/BENCHMARK.md).
        // min() never raises, and on no-SMT hosts physical == logical, so
        // this is a structural no-op on all Apple Silicon. A deliberate
        // consequence: FUCINA_MAX_THREADS caps the physical-core base and
        // can no longer reach the logical count on SMT machines;
        // `setMaxThreads` pre-seeds the cache before detection and remains
        // the escape hatch for deliberate oversubscription.
        if (topology.physicalCpuCount()) |physical| count = @min(count, @max(physical, 1));
        // Apple Silicon: the compute team stays on PERFORMANCE cores. With
        // the two M1 E-cores in the team, every barrier waits for the E-core
        // stragglers — measured on the GPT train-step bench: 8P beats
        // 8P+2E by ~2%, and any spin-budget increase with E-cores enrolled
        // collapses throughput (287.9 -> 401.1 ms/step at 256k spins). The
        // QoS pin biases workers to P-cores but cannot guarantee placement;
        // sizing the team to perflevel0 does. `setMaxThreads` pre-seeds the
        // cache and remains the deliberate-oversubscription escape hatch.
        if (topology.performanceCoreCount()) |performance| count = @min(count, @max(performance, 1));
        // A cgroup CPU-bandwidth limit never shows up in the affinity mask, so
        // size the team to the allowance as well: threads beyond it do not add
        // throughput, they just drain the quota sooner and stall the next
        // barrier until CFS refills it. `setMaxThreads` still pre-seeds the
        // cache as the escape hatch.
        if (topology.cgroupCpuBudget()) |budget| count = @min(count, @max(budget, 1));
        // Optional override (mirrors llama.cpp's -t): cap the detected CPU count
        // for per-machine thread tuning. See the note on `vector_max_threads`
        // for when fewer threads help (decode / heat-soaked prefill on M1).
        if (envMaxThreads()) |cap| count = @min(count, cap);
        cached_cpu_count.store(count, .release);
    }
    return @max(@as(usize, 1), @min(count, max_threads));
}

/// The FUCINA_MAX_THREADS cap, or null when unset/invalid/zero. Consulted only
/// on the first `cpuThreadCount` call (a `setMaxThreads` call before that wins
/// by pre-seeding the cache).
fn envMaxThreads() ?usize {
    return env.positiveUsize("FUCINA_MAX_THREADS");
}

/// Part cap for a map over `n` items in chunks of at least `min_len`: one
/// part below `min_len`, else `1 + n / min_len` (`ExecContext.forRange`
/// clamps it to the team size). The chunk grid then depends only on `n`
/// and the part count; whether that makes a kernel thread-count-invariant
/// is the kernel's property to state (the optimizer and ES maps do).
pub fn partsForChunk(n: usize, min_len: usize) usize {
    return if (n >= min_len) 1 + n / min_len else 1;
}

pub fn saturatedMul3(a: usize, b: usize, c: usize) usize {
    const ab = std.math.mul(usize, a, b) catch return std.math.maxInt(usize);
    return std.math.mul(usize, ab, c) catch std.math.maxInt(usize);
}

test {
    _ = @import("parallel_tests.zig");
    _ = topology;
    _ = env;
}
