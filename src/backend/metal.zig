//! Metal GPU GEMM provider (`-Dgpu=metal`) — Zig host side.
//!
//! The objective is training-shaped f32 GEMM offload: run the heavy matrix
//! multiplications on the GPU, keep everything else on CPU. The kernel is the
//! vendored MLX "steel" GEMM (`metal/mlx_gemm.metal`, MIT, Apple copyright) —
//! simdgroup-matrix 8x8, 32x32x16 tiles, alignment-specialized edge handling —
//! compiled once at lazy init from embedded source by the ObjC shim
//! (`metal/shim.m`).
//!
//! Contract with native.zig: `gemmF32Async` commits eagerly and attaches a
//! completion token to the ordinary output storage; GPU consumers stay queue
//! ordered and the first CPU access waits. Submission failure returns false
//! before attachment so the caller falls through to BLAS/vector. The direct
//! slice `gemmF32` entry remains blocking for parity/benchmark callers. A
//! batched call is ONE dispatch with grid depth = batch.
//!
//! Stable-weight dense quantized offload uses the same async storage Work seam.
//! Grouped MoE alone keeps one process-global staging panel pair/lock because
//! its CPU gather/GeGLU/scatter phases impose host data boundaries.
//! The public Tensor API has no device/location state.
//!
//! Heuristics: `shouldUseGpu` gates on m*n*k work (default 2^32, the measured
//! M1 Max async crossover vs Accelerate/AMX — see `default_min_work`) with
//! `FUCINA_GPU_MIN_WORK` to experiment and `FUCINA_GPU=0` as a runtime kill
//! switch. Compact/raw Q6_K retains its Parakeet-tuned gate; dense model
//! weights use per-format gates measured against their faster packed CPU
//! fallback. Tune with `bench-gpu-dispatch` and `bench-gpu-formats`.
const std = @import("std");
const accelerator = @import("../accelerator.zig");
const build_options = @import("build_options");
const storage = @import("../storage.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");

const Tensor = tensor.Tensor;
const TensorF16 = tensor.TensorOf(.f16);
const TensorBf16 = tensor.TensorOf(.bf16);

pub const enabled = build_options.gpu_kind == .metal;

/// Provider capability: dequant-in-kernel quantized GEMM (dense + grouped
/// MoE) is implemented. Loaders that reshape CPU-side representations for the
/// GPU quant path (gemma4's single-raw-expert choice, borrow arm) key on this
/// rather than `enabled` — providers whose quant arms are still stubs keep
/// the plain CPU story. Must be `enabled and ...`: a false `enabled` keeps
/// the whole module comptime-dead on other builds (extern symbols, libc).
pub const has_quant_gemm = enabled;
/// Q5_K currently has a CUDA dequant kernel only. Keeping this capability
/// explicit lets the shared exec/weight layer retain Metal's CPU fallback.
pub const has_q5_k_quant = false;
/// Ternary TQ2_0 dequant-in-kernel GEMM (fucina_mul_mm_tq2_0_f32).
pub const has_tq2_0_quant = true;

pub const Orient = enum(c_int) { nn = 0, tn = 1, nt = 2 };

const CommandTiming = extern struct {
    gpu_ns: u64,
    sched_ns: u64,
};

extern fn fucina_metal_init(msl_source: [*:0]const u8) ?*anyopaque;
extern fn fucina_metal_deinit(ctx: *anyopaque) void;
extern fn fucina_metal_device_name(ctx: *anyopaque) [*:0]const u8;
extern fn fucina_metal_gemm_f32(
    ctx: *anyopaque,
    variant: c_int,
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    m: i64,
    n: i64,
    k: i64,
    batch: i64,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    timing: ?*CommandTiming,
) c_int;
extern fn fucina_metal_wrap_storage(ctx: *anyopaque, ptr: *const anyopaque, len: i64) ?*anyopaque;
extern fn fucina_metal_free_storage_wrap(wrap: *anyopaque) void;
extern fn fucina_metal_gemm_f32_async(
    ctx: *anyopaque,
    variant: c_int,
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    a_wrap: ?*anyopaque,
    b_wrap: ?*anyopaque,
    c_wrap: ?*anyopaque,
    m: i64,
    n: i64,
    k: i64,
    batch: i64,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
) ?*anyopaque;
extern fn fucina_metal_gemm_f16_nt_async(
    ctx: *anyopaque,
    a: [*]const f16,
    b: [*]const f16,
    c: [*]f32,
    a_wrap: ?*anyopaque,
    b_wrap: ?*anyopaque,
    c_wrap: ?*anyopaque,
    m: i64,
    n: i64,
    k: i64,
) ?*anyopaque;
extern fn fucina_metal_gemm_bf16_nt_async(
    ctx: *anyopaque,
    a: [*]const u16,
    b: [*]const u16,
    c: [*]f32,
    a_wrap: ?*anyopaque,
    b_wrap: ?*anyopaque,
    c_wrap: ?*anyopaque,
    m: i64,
    n: i64,
    k: i64,
) ?*anyopaque;
extern fn fucina_metal_ticket_wait(ticket: *anyopaque, timing: ?*CommandTiming) c_int;
extern fn fucina_metal_ticket_free(ticket: *anyopaque) void;
extern fn fucina_metal_gemm_f16_nt(
    ctx: *anyopaque,
    a: [*]const f16,
    b: [*]const f16,
    m: i64,
    n: i64,
    k: i64,
    cache_rhs: c_int,
    out_staging: *[*]const f16,
    timing: ?*CommandTiming,
) c_int;
extern fn fucina_metal_alloc_resident_bytes(ctx: *anyopaque, len: i64) ?[*]u8;
extern fn fucina_metal_free_resident_bytes(ctx: *anyopaque, ptr: [*]const u8) c_int;
extern fn fucina_metal_qmoe_stage(
    ctx: *anyopaque,
    in_bytes: i64,
    out_bytes: i64,
    in_ptr: *?*anyopaque,
    out_ptr: *?*anyopaque,
) c_int;
extern fn fucina_metal_gemm_q_grouped_nt(
    ctx: *anyopaque,
    format: c_int,
    rhs_bytes: [*]const u8,
    rhs_len: i64,
    cache_rhs: c_int,
    nb01: i64,
    nb02: i64,
    n_out: i64,
    k: i64,
    tiles: [*]const QMMTile,
    n_tiles: i64,
    timing: ?*CommandTiming,
) c_int;
extern fn fucina_metal_gemm_q_dense_nt_async(
    ctx: *anyopaque,
    format: c_int,
    rhs_bytes: [*]const u8,
    rhs_len: i64,
    nb01: i64,
    nb02: i64,
    a: [*]const f32,
    c: [*]f32,
    a_wrap: ?*anyopaque,
    c_wrap: ?*anyopaque,
    batch_count: i64,
    m: i64,
    n: i64,
    k: i64,
) ?*anyopaque;

// One library: the MLX steel f32/f16 GEMM plus the vendored ggml quantized
// mul_mm (dequant-in-kernel). Both files are self-contained MSL; metal_stdlib
// include-guards make the concatenation safe.
const msl_source = @embedFile("metal/mlx_gemm.metal") ++ "\n" ++ @embedFile("metal/ggml_mul_mm.metal");

const State = struct {
    ctx: ?*anyopaque = null,
    gpu_enabled: bool = true,
    min_work: u64 = default_min_work,
    min_work_f16: u64 = default_min_work_f16,
    min_work_16bit_resident: u64 = default_min_work_16bit_resident,
    min_work_gemv: u64 = default_min_work_gemv,
    min_work_qmoe: u64 = default_min_work_qmoe,
    min_work_dense_q6: u64 = default_min_work_dense_q6,
    min_work_packed_q4: u64 = default_min_work_packed_q4,
    min_work_packed_q6: u64 = default_min_work_packed_q6,
    min_work_packed_q8: u64 = default_min_work_packed_q8,
    min_work_packed_tq2: u64 = default_min_work_packed_tq2,
    qmoe_min_fill_pct: u64 = default_qmoe_min_fill_pct,
};

/// Default offload threshold, retuned from the eager-async paired benchmark
/// on M1 Max (2026-07-10). 1024³ and 2048x1024x1024 (2^31 work) are
/// DVFS-sensitive crossovers, while repeated
/// alternating trials at 2048x2048x1024 (2^32) win by at least 24% and
/// 2048³ wins decisively. Below it AMX and the fixed command cost dominate.
const default_min_work: u64 = 1 << 32;
/// The f16-operands NT entry competes with the CPU f16 row kernels (no AMX
/// arm — Accelerate has no f16 GEMM here), which run an order of magnitude
/// below AMX f32, so the GPU pays off much earlier. 2^27 ≈ 134M m*n*k keeps
/// the command cost safely amortized.
const default_min_work_f16: u64 = 1 << 27;
/// Resident f32 GEMV has no weight transfer and the cached storage wrapper
/// removes page-wiring overhead.  M1 Max crosses Accelerate around 8 Mi work;
/// keep a conservative 16 Mi default and require m<=8 plus residency.
const default_min_work_gemv: u64 = 1 << 24;

// Small-m (decode/batched-decode) admission for 16-bit weight GEMMs whose
// RHS is already Metal-mapped (a storage-lifetime page wrap from an earlier
// prefill): the dispatch reads the whole weight matrix once, so past this
// work floor the GPU's bandwidth can beat the CPU streaming kernels.
// MEASURED (Qwen3-1.7B-BF16 self-study, idle M1 Max): a 2^24 floor admits
// the m=4 batched-decode projections (~2^25) and LOSES 18% end-to-end —
// per-dispatch overhead outweighs the bandwidth win at that width (unlike
// CUDA's resident admission) — while m=16 admission (~2^27) is neutral.
// The default therefore sits at 2^27: batched decode at m>=16 and the
// m=1 lm-head row may offload; narrow decode stays on the CPU kernels.
// Tune with FUCINA_GPU_MIN_WORK_16BIT_RESIDENT.
const default_min_work_16bit_resident: u64 = 1 << 27;
/// Quantized grouped MoE GEMM gate (total m·n·k across both projections of a
/// layer). The CPU competitor is the gather/quantize/small-m packed-kernel
/// path, which runs ~10 GF/s effective (measured 2026-06-12; per-call
/// overhead around tiny per-expert GEMMs, not compute), so the GPU pays off
/// well below the f32 crossover; 2^30 keeps the two ~0.5 ms round trips
/// + gather/geglu/scatter CPU phases safely amortized. Tune with
/// `FUCINA_GPU_MIN_WORK_QMOE`.
const default_min_work_qmoe: u64 = 1 << 30;
/// Dense Q6_K linears in Parakeet 110M are many medium MxN contractions; the
/// dequant-in-kernel GPU path wins after warmup far below the MoE gate, while
/// Q4_K/Q8_0 still lose at this size. Keep this separate from the grouped-MoE
/// threshold to avoid pulling unrelated quant paths onto Metal.
const default_min_work_dense_q6: u64 = 1 << 22;
/// Dense GGUF linears fall back to Fucina's load-time-packed CPU kernels,
/// which are materially faster than the compact/raw fallback used by
/// Parakeet. Paired eager measurements on M1 Max put the conservative
/// crossovers at 2^30 (Q4_K), 2^31 (Q6_K), and 2^29 (Q8_0).
const default_min_work_packed_q4: u64 = 1 << 30;
const default_min_work_packed_q6: u64 = 1 << 31;
const default_min_work_packed_q8: u64 = 1 << 29;
// Ternary: the CPU fallback is the x4 interleaved kernel (docs/TERNARY.md)
// at ~74 G-MAC/s single-thread — against a ~0.5 ms dispatch floor the
// break-even sits near 2^25 m*n*k, far below the q8_0-class defaults, and
// PTQTP bodies at 0.6B-class shapes (128-row chunks x 1024..3072 dims)
// land in 2^27..2^29. FUCINA_GPU_MIN_WORK_DENSE_TQ2 overrides.
const default_min_work_packed_tq2: u64 = 1 << 25;
/// Minimum grouped-MoE tile occupancy (percent of the 32-row token-tile slots
/// that carry real rows) before the GPU arm engages. Per-tile GPU cost is
/// fill-independent (~45-53 µs/tile at 12% and at 100% fill — weight dequant
/// dominates; measured 2026-07-03), so below ~50% occupancy the raw CPU path
/// wins. `FUCINA_GPU_QMOE_MIN_FILL` overrides (0 = occupancy-blind, the
/// pre-2026-07 behavior; >100 = grouped GPU path never engages).
const default_qmoe_min_fill_pct: u64 = 50;

var state: State = .{};
var config_done = std.atomic.Value(bool).init(false);
var init_done = std.atomic.Value(bool).init(false);
var init_mutex: thread.Mutex = .{};

// --- Optional dispatch tracing (FUCINA_GPU_TRACE=1) -------------------------
// Zero-overhead when off: every site is guarded by `trace_on` (a plain bool set
// once at config). Counters are atomic (gates run on worker threads; dispatches
// are lock-serialized). The synchronous-dispatch wall-time per kind is the
// combined GPU-compute + waitUntilCompleted envelope (we commit+wait inline);
// qmoe lock-wait and stage-copy are split out. When tracing is on, the shim also
// returns command-buffer GPU/kernel timestamps so traceDump can show wall-vs-GPU
// overhead and the dominant shape buckets.
var trace_on: bool = false;
const Trace = struct {
    f32_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    f32_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f32_gpu_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f32_sched_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f32_async_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    f32_submit_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f32_wait_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_gpu_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_sched_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_async_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_submit_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_wait_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    bf16_async_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    bf16_submit_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    bf16_wait_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    bf16_gpu_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    bf16_sched_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_gpu_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_sched_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_async_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_submit_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_wait_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_lock_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_stage_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    rhs_cacheable: std.atomic.Value(u64) = .{ .raw = 0 },
    rhs_transient: std.atomic.Value(u64) = .{ .raw = 0 },
    dev_alloc_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    dev_alloc_bytes: std.atomic.Value(u64) = .{ .raw = 0 },
    gate_pass: std.atomic.Value(u64) = .{ .raw = 0 },
    gate_below: std.atomic.Value(u64) = .{ .raw = 0 },
    gate_shape: std.atomic.Value(u64) = .{ .raw = 0 },
    shim_err: std.atomic.Value(u64) = .{ .raw = 0 },
    shape_overflow: std.atomic.Value(u64) = .{ .raw = 0 },
};
var trace: Trace = .{};

const TraceKind = enum(u8) { f32, f16, quant };
const TraceShape = struct {
    kind: TraceKind = .f32,
    m: usize = 0,
    n: usize = 0,
    k: usize = 0,
    batch: usize = 0,
    tiles: usize = 0,
    calls: u64 = 0,
    wall_ns: u64 = 0,
    gpu_ns: u64 = 0,
    sched_ns: u64 = 0,
};
const trace_shape_slots = 16;
var trace_shape_lock: thread.Mutex = .{};
var trace_shapes: [trace_shape_slots]TraceShape = [_]TraceShape{.{}} ** trace_shape_slots;

// Monotonic ns without an `std.Io` handle (the backend dispatch sites have none,
// and `std.time.Timer` was removed in 0.16's Io migration). This file is macOS-only
// — `-Dgpu=metal` does not build elsewhere — so the Darwin clock is fine.
extern fn clock_gettime_nsec_np(clock_id: c_int) u64;
const clock_uptime_raw: c_int = 8; // Darwin <sys/_clock_id.h> CLOCK_UPTIME_RAW

inline fn tinc(c: *std.atomic.Value(u64), v: u64) void {
    _ = c.fetchAdd(v, .monotonic);
}
/// Returns a start timestamp (0 when tracing is off → `telapsed` is a no-op).
inline fn tstart() u64 {
    return if (trace_on) clock_gettime_nsec_np(clock_uptime_raw) else 0;
}
inline fn telapsed(c: *std.atomic.Value(u64), start: u64) void {
    if (!trace_on) return;
    _ = c.fetchAdd(clock_gettime_nsec_np(clock_uptime_raw) -% start, .monotonic);
}
inline fn tfinish(start: u64) u64 {
    return if (trace_on) clock_gettime_nsec_np(clock_uptime_raw) -% start else 0;
}
inline fn traceRhsCache(flag: bool) void {
    if (!trace_on) return;
    tinc(if (flag) &trace.rhs_cacheable else &trace.rhs_transient, 1);
}
inline fn tgate(pass: bool) void {
    if (!trace_on) return;
    tinc(if (pass) &trace.gate_pass else &trace.gate_below, 1);
}
inline fn overheadNs(wall_ns: u64, gpu_ns: u64) u64 {
    return if (wall_ns > gpu_ns) wall_ns - gpu_ns else 0;
}
fn traceRecordShape(kind: TraceKind, m: usize, n: usize, k: usize, batch: usize, tiles: usize, wall_ns: u64, timing: CommandTiming) void {
    if (!trace_on) return;
    trace_shape_lock.lock();
    defer trace_shape_lock.unlock();

    var empty: ?usize = null;
    for (&trace_shapes, 0..) |*slot, i| {
        if (slot.calls == 0) {
            if (empty == null) empty = i;
            continue;
        }
        if (slot.kind == kind and slot.m == m and slot.n == n and slot.k == k and slot.batch == batch and slot.tiles == tiles) {
            slot.calls += 1;
            slot.wall_ns +|= wall_ns;
            slot.gpu_ns +|= timing.gpu_ns;
            slot.sched_ns +|= timing.sched_ns;
            return;
        }
    }
    if (empty) |i| {
        trace_shapes[i] = .{
            .kind = kind,
            .m = m,
            .n = n,
            .k = k,
            .batch = batch,
            .tiles = tiles,
            .calls = 1,
            .wall_ns = wall_ns,
            .gpu_ns = timing.gpu_ns,
            .sched_ns = timing.sched_ns,
        };
        return;
    }
    tinc(&trace.shape_overflow, 1);
}
fn traceResetShapes() void {
    trace_shape_lock.lock();
    defer trace_shape_lock.unlock();
    trace_shapes = [_]TraceShape{.{}} ** trace_shape_slots;
}
fn traceShapeLess(_: void, a: TraceShape, b: TraceShape) bool {
    if (a.wall_ns == b.wall_ns) return a.calls > b.calls;
    return a.wall_ns > b.wall_ns;
}

pub fn traceEnabled() bool {
    ensureConfig();
    return trace_on;
}
/// Reset counters (call before a warm measurement window). Single-threaded.
pub fn traceReset() void {
    if (!trace_on) return;
    trace = .{};
    traceResetShapes();
}
/// Print the accumulated breakdown to stderr (no-op when tracing is off).
pub fn traceDump() void {
    if (!trace_on) return;
    const ms = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.f;
    const f32_wall = trace.f32_ns.load(.monotonic);
    const f16_wall = trace.f16_ns.load(.monotonic);
    const quant_wall = trace.quant_ns.load(.monotonic);
    const f32_gpu = trace.f32_gpu_ns.load(.monotonic);
    const f16_gpu = trace.f16_gpu_ns.load(.monotonic);
    const quant_gpu = trace.quant_gpu_ns.load(.monotonic);
    std.debug.print(
        "[gpu-trace] async: f32 calls={d} submit={d:.1}ms host-wait={d:.1}ms | f16 calls={d} submit={d:.1}ms host-wait={d:.1}ms | bf16 calls={d} submit={d:.1}ms host-wait={d:.1}ms gpu={d:.1}ms | quant calls={d} submit={d:.1}ms host-wait={d:.1}ms\n",
        .{
            trace.f32_async_calls.load(.monotonic),    ms(trace.f32_submit_ns.load(.monotonic)),
            ms(trace.f32_wait_ns.load(.monotonic)),    trace.f16_async_calls.load(.monotonic),
            ms(trace.f16_submit_ns.load(.monotonic)),  ms(trace.f16_wait_ns.load(.monotonic)),
            trace.bf16_async_calls.load(.monotonic),   ms(trace.bf16_submit_ns.load(.monotonic)),
            ms(trace.bf16_wait_ns.load(.monotonic)),   ms(trace.bf16_gpu_ns.load(.monotonic)),
            trace.quant_async_calls.load(.monotonic),  ms(trace.quant_submit_ns.load(.monotonic)),
            ms(trace.quant_wait_ns.load(.monotonic)),
        },
    );
    std.debug.print(
        \\[gpu-trace] dispatch: f32={d} ({d:.1}ms) f16={d} ({d:.1}ms) quant={d} ({d:.1}ms)
        \\[gpu-trace] gpu-time: f32={d:.1}ms overhead={d:.1}ms | f16={d:.1}ms overhead={d:.1}ms | quant={d:.1}ms overhead={d:.1}ms
        \\[gpu-trace] kernel-sched: f32={d:.1}ms f16={d:.1}ms quant={d:.1}ms
        \\[gpu-trace] quant overhead: lock-wait={d:.1}ms stage-copy={d:.1}ms
        \\[gpu-trace] rhs-cache: stable={d} transient={d} | resident-bytes allocs={d} ({d:.1} MB)
        \\[gpu-trace] gate decisions: pass={d} below-gate={d} shape-reject={d} shim-error={d} shape-overflow={d}
        \\
    , .{
        trace.f32_calls.load(.monotonic),                                      ms(f32_wall),
        trace.f16_calls.load(.monotonic),                                      ms(f16_wall),
        trace.quant_calls.load(.monotonic),                                    ms(quant_wall),
        ms(f32_gpu),                                                           ms(overheadNs(f32_wall, f32_gpu)),
        ms(f16_gpu),                                                           ms(overheadNs(f16_wall, f16_gpu)),
        ms(quant_gpu),                                                         ms(overheadNs(quant_wall, quant_gpu)),
        ms(trace.f32_sched_ns.load(.monotonic)),                               ms(trace.f16_sched_ns.load(.monotonic)),
        ms(trace.quant_sched_ns.load(.monotonic)),                             ms(trace.quant_lock_ns.load(.monotonic)),
        ms(trace.quant_stage_ns.load(.monotonic)),                             trace.rhs_cacheable.load(.monotonic),
        trace.rhs_transient.load(.monotonic),                                  trace.dev_alloc_calls.load(.monotonic),
        @as(f64, @floatFromInt(trace.dev_alloc_bytes.load(.monotonic))) / 1e6, trace.gate_pass.load(.monotonic),
        trace.gate_below.load(.monotonic),                                     trace.gate_shape.load(.monotonic),
        trace.shim_err.load(.monotonic),                                       trace.shape_overflow.load(.monotonic),
    });
    trace_shape_lock.lock();
    var shapes = trace_shapes;
    trace_shape_lock.unlock();
    std.mem.sort(TraceShape, &shapes, {}, traceShapeLess);
    var printed: usize = 0;
    for (shapes) |s| {
        if (s.calls == 0) continue;
        if (printed == 0) std.debug.print("[gpu-trace] top shapes by dispatch wall:\n", .{});
        std.debug.print(
            "[gpu-trace]   {s} m={d} n={d} k={d} batch={d} tiles={d} calls={d} wall={d:.1}ms gpu={d:.1}ms overhead={d:.1}ms\n",
            .{ @tagName(s.kind), s.m, s.n, s.k, s.batch, s.tiles, s.calls, ms(s.wall_ns), ms(s.gpu_ns), ms(overheadNs(s.wall_ns, s.gpu_ns)) },
        );
        printed += 1;
        if (printed == 8) break;
    }
}

/// One-time runtime configuration read. This deliberately does not create the
/// Metal device/library so threshold probes can stay below-threshold cheap.
fn ensureConfig() void {
    if (config_done.load(.acquire)) return;
    init_mutex.lock();
    defer init_mutex.unlock();
    if (!config_done.load(.monotonic)) {
        initConfigOnce();
        config_done.store(true, .release);
    }
}

fn initConfigOnce() void {
    if (std.c.getenv("FUCINA_GPU")) |v_ptr| {
        const v = std.mem.span(v_ptr);
        if (v.len > 0 and v[0] == '0') state.gpu_enabled = false;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK")) |v_ptr| {
        state.min_work = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_F16")) |v_ptr| {
        state.min_work_f16 = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_f16;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_GEMV")) |v_ptr| {
        state.min_work_gemv = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_gemv;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_16BIT_RESIDENT")) |v_ptr| {
        state.min_work_16bit_resident = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_16bit_resident;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_QMOE")) |v_ptr| {
        const parsed = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_qmoe;
        state.min_work_qmoe = parsed;
        state.min_work_dense_q6 = parsed;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_DENSE_Q6")) |v_ptr| {
        const parsed = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_dense_q6;
        state.min_work_dense_q6 = parsed;
        state.min_work_packed_q6 = parsed;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_DENSE_Q4")) |v_ptr| {
        state.min_work_packed_q4 = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_packed_q4;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_DENSE_Q8")) |v_ptr| {
        state.min_work_packed_q8 = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_packed_q8;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_DENSE_TQ2")) |v_ptr| {
        state.min_work_packed_tq2 = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_packed_tq2;
    }
    if (std.c.getenv("FUCINA_GPU_QMOE_MIN_FILL")) |v_ptr| {
        state.qmoe_min_fill_pct = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_qmoe_min_fill_pct;
    }
    if (std.c.getenv("FUCINA_GPU_TRACE")) |v_ptr| {
        const v = std.mem.span(v_ptr);
        if (v.len > 0 and v[0] != '0') trace_on = true;
    }
}

/// One-time lazy device init (std.once is gone in Zig 0.16): double-checked
/// under a mutex so concurrent ExecContexts share one device/library.
fn ensureInit() void {
    ensureConfig();
    if (!state.gpu_enabled) return;
    if (init_done.load(.acquire)) return;
    init_mutex.lock();
    defer init_mutex.unlock();
    if (!init_done.load(.monotonic)) {
        initOnce();
        init_done.store(true, .release);
    }
}

fn initOnce() void {
    state.ctx = fucina_metal_init(msl_source);
    if (state.ctx == null) {
        std.log.warn("fucina-metal: init failed; GPU GEMM disabled for this process", .{});
    }
}

/// Lazy device init; null = GPU unavailable/disabled.
fn context() ?*anyopaque {
    ensureInit();
    return state.ctx;
}

pub fn deviceName() ?[]const u8 {
    const ctx = context() orelse return null;
    return std.mem.span(fucina_metal_device_name(ctx));
}

pub fn shouldUseGpu(m: usize, n: usize, k: usize) bool {
    ensureConfig();
    if (m < 32 or n < 32 or k < 16) {
        if (trace_on) tinc(&trace.gate_shape, 1);
        return false;
    }
    const work = std.math.mul(u64, std.math.mul(u64, m, n) catch return true, k) catch return true;
    const pass = state.gpu_enabled and work >= state.min_work;
    tgate(pass);
    return pass;
}

pub fn shouldUseGpuBatched(m: usize, n: usize, k: usize, batch_count: usize) bool {
    ensureConfig();
    if (m < 32 or n < 32 or k < 16) {
        if (trace_on) tinc(&trace.gate_shape, 1);
        return false;
    }
    const per = std.math.mul(u64, std.math.mul(u64, m, n) catch return true, k) catch return true;
    const work = std.math.mul(u64, per, batch_count) catch return true;
    const pass = state.gpu_enabled and work >= state.min_work;
    tgate(pass);
    return pass;
}

pub fn shouldUseGpuF16(m: usize, n: usize, k: usize) bool {
    ensureConfig();
    if (m < 32 or n < 32 or k < 16) {
        if (trace_on) tinc(&trace.gate_shape, 1);
        return false;
    }
    const work = std.math.mul(u64, std.math.mul(u64, m, n) catch return true, k) catch return true;
    const pass = state.gpu_enabled and work >= state.min_work_f16;
    tgate(pass);
    return pass;
}

/// Resident small-m admission (the CUDA gate's idea, wrap-based): a decode
/// or batched-decode GEMM over weights that already carry a Metal page wrap
/// reads the whole RHS once, so bandwidth — not the m>=32 tile shape —
/// decides. Legal only with an existing wrap: first touch happens on a
/// large prefill, never on the latency-sensitive decode path.
fn residentSmallM16bit(buffer: anytype, m: usize, n: usize, k: usize) bool {
    if (m == 0 or m >= 32 or n < 256 or k < 256) return false;
    if (!state.gpu_enabled) return false;
    const work = std.math.mul(u64, std.math.mul(u64, m, n) catch return false, k) catch return false;
    if (work < state.min_work_16bit_resident) return false;
    const resident = buffer.acceleratorResource(.metal) != null;
    tgate(resident);
    return resident;
}

pub fn shouldUseGpuF16ForRhs(b: *const TensorF16, m: usize, n: usize, k: usize) bool {
    ensureConfig();
    if (residentSmallM16bit(b.buffer, m, n, k)) return true;
    return shouldUseGpuF16(m, n, k);
}

/// bf16 weight GEMMs ride the f16 economics: 16-bit RHS rows streamed
/// zero-copy, f32 accumulation, so the same thresholds apply.
pub fn shouldUseGpuBf16ForRhs(b: *const TensorBf16, m: usize, n: usize, k: usize) bool {
    ensureConfig();
    if (residentSmallM16bit(b.buffer, m, n, k)) return true;
    return shouldUseGpuF16(m, n, k);
}

/// Small-m f32 GEMV/GEMM gate.  Unlike the ordinary shape gate, this is legal
/// only when the RHS already has a storage-lifetime Metal mapping (or is a
/// device-owned resident allocation), so no large page-wrap cost is hidden.
pub fn shouldUseGpuGemv(b: *const Tensor, m: usize, n: usize, k: usize) bool {
    ensureConfig();
    if (m == 0 or m > 8 or n < 256 or k < 256) return false;
    const work = std.math.mul(u64, std.math.mul(u64, m, n) catch std.math.maxInt(u64), k) catch std.math.maxInt(u64);
    if (work < state.min_work_gemv or !state.gpu_enabled) return false;
    const bytes = std.mem.sliceAsBytes(b.buffer.data);
    const resident = isResidentRange(bytes) or b.buffer.acceleratorResource(.metal) != null;
    tgate(resident);
    return resident;
}

/// Tensor-aware native-dispatch gate. Metal's ordinary large-op gate needs
/// no residency distinction (host and device share memory); the separate arm
/// only admits the small resident GEMV shape rejected by that gate.
pub fn shouldUseGpuForRhs(b: *const Tensor, m: usize, n: usize, k: usize) bool {
    return shouldUseGpuGemv(b, m, n, k) or shouldUseGpu(m, n, k);
}

pub fn shouldUseGpuBatchedForRhs(_: *const Tensor, m: usize, n: usize, k: usize, batch_count: usize) bool {
    return shouldUseGpuBatched(m, n, k, batch_count);
}

/// Serializes f16 GEMMs: the shim's f16 output staging buffer is reused
/// across calls, so the caller must hold this across `gemmF16Nt` + the widen
/// of its result.
pub var f16_lock: thread.Mutex = .{};

/// C16 = A16[m,k] · B16[n,k]ᵀ on the GPU; returns the f16 result staging
/// (valid until the next f16 call — hold `f16_lock` across call + use), or
/// null when the GPU didn't run. `rhs_cacheable` must only be true when `b`
/// stays mapped for the process lifetime (resident f16 weights) — a cached
/// wrap of a freed-and-reused page reads stale data.
pub fn gemmF16Nt(a: []const f16, b: []const f16, m: usize, n: usize, k: usize, rhs_cacheable: bool) ?[]const f16 {
    const ctx = context() orelse return null;
    // Resident RHS bytes are address-stable for the process regardless of
    // the caller's conservative flag — let the shim's wrap cache serve them.
    const cacheable = rhs_cacheable or isResidentRange(std.mem.sliceAsBytes(b));
    var staging: [*]const f16 = undefined;
    var timing: CommandTiming = .{ .gpu_ns = 0, .sched_ns = 0 };
    const timer = tstart();
    const rc = fucina_metal_gemm_f16_nt(ctx, a.ptr, b.ptr, @intCast(m), @intCast(n), @intCast(k), @intFromBool(cacheable), &staging, if (trace_on) &timing else null);
    if (trace_on) {
        const wall_ns = tfinish(timer);
        tinc(&trace.f16_ns, wall_ns);
        tinc(&trace.f16_gpu_ns, timing.gpu_ns);
        tinc(&trace.f16_sched_ns, timing.sched_ns);
        tinc(&trace.f16_calls, 1);
        traceRhsCache(cacheable);
        traceRecordShape(.f16, m, n, k, 1, 0, wall_ns, timing);
        if (rc != 0) tinc(&trace.shim_err, 1);
    }
    if (rc != 0) return null;
    return staging[0 .. m * n];
}

/// C[m,n] = op(A)·op(B), f32 row-major, overwrite (beta = 0). Slices use the
/// same operand conventions as the BLAS arm: nn A[m,k]/B[k,n]; tn A stored
/// [k,m]; nt B stored [n,k]. Returns false when the GPU didn't run.
pub fn gemmF32(
    orient: Orient,
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
) bool {
    return gemmBatchedF32(orient, a, b, c, m, n, k, 1, 0, 0, 0);
}

pub fn gemmBatchedF32(
    orient: Orient,
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
) bool {
    const ctx = context() orelse return false;
    var timing: CommandTiming = .{ .gpu_ns = 0, .sched_ns = 0 };
    const timer = tstart();
    const rc = fucina_metal_gemm_f32(
        ctx,
        @intFromEnum(orient),
        a.ptr,
        b.ptr,
        c.ptr,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        @intCast(batch_count),
        @intCast(stride_a),
        @intCast(stride_b),
        @intCast(stride_c),
        if (trace_on) &timing else null,
    );
    if (trace_on) {
        const wall_ns = tfinish(timer);
        tinc(&trace.f32_ns, wall_ns);
        tinc(&trace.f32_gpu_ns, timing.gpu_ns);
        tinc(&trace.f32_sched_ns, timing.sched_ns);
        tinc(&trace.f32_calls, 1);
        traceRecordShape(.f32, m, n, k, batch_count, 0, wall_ns, timing);
        if (rc != 0) tinc(&trace.shim_err, 1);
    }
    return rc == 0;
}

const MetalResource = struct {
    resource: accelerator.Resource,
    wrap: *anyopaque,

    const vtable: accelerator.ResourceVTable = .{ .destroy = destroy };

    fn destroy(ctx: *anyopaque) void {
        const self: *MetalResource = @ptrCast(@alignCast(ctx));
        fucina_metal_free_storage_wrap(self.wrap);
        std.heap.c_allocator.destroy(self);
    }
};

/// One storage-lifetime Metal page wrapper.  Buffer-pool reuse changes the
/// values but not the allocation/mapping, so the wrapper remains valid until
/// `Buffer.destroy` and removes both the per-call Objective-C allocation and
/// the repeated VM residency wiring from the hot path.
fn storageWrap(buffer: anytype) ?*anyopaque {
    if (buffer.acceleratorResource(.metal)) |resource| {
        const cached: *MetalResource = @ptrCast(@alignCast(resource.ctx));
        return cached.wrap;
    }
    const ctx = context() orelse return null;
    const Elem = @TypeOf(buffer.data[0]);
    const byte_len = std.math.mul(usize, buffer.data.len, @sizeOf(Elem)) catch return null;
    if (byte_len > std.math.maxInt(i64)) return null;
    const wrap = fucina_metal_wrap_storage(ctx, buffer.data.ptr, @intCast(byte_len)) orelse return null;
    const created = std.heap.c_allocator.create(MetalResource) catch {
        fucina_metal_free_storage_wrap(wrap);
        return null;
    };
    created.* = .{
        .resource = .{ .provider = .metal, .ctx = created, .vtable = &MetalResource.vtable },
        .wrap = wrap,
    };
    if (buffer.setAcceleratorResource(&created.resource)) return wrap;
    created.resource.destroy();
    const winner = buffer.acceleratorResource(.metal) orelse return null;
    const cached: *MetalResource = @ptrCast(@alignCast(winner.ctx));
    return cached.wrap;
}

const MetalWork = struct {
    work: accelerator.Work,
    ticket: *anyopaque,
    a_buffer: *storage.Buffer,
    b_buffer: ?*storage.Buffer,

    const vtable: accelerator.WorkVTable = .{
        .finish = finish,
        .device_ptr = null,
        .destroy = destroy,
    };

    fn finish(ctx: *anyopaque, _: bool) bool {
        const self: *MetalWork = @ptrCast(@alignCast(ctx));
        defer {
            self.a_buffer.clearPendingUse(&self.work);
            if (self.b_buffer) |buffer| buffer.clearPendingUse(&self.work);
        }
        var timing: CommandTiming = .{ .gpu_ns = 0, .sched_ns = 0 };
        const wait_started = tstart();
        const rc = fucina_metal_ticket_wait(self.ticket, if (trace_on) &timing else null);
        if (trace_on) {
            tinc(&trace.f32_wait_ns, tfinish(wait_started));
            tinc(&trace.f32_gpu_ns, timing.gpu_ns);
            tinc(&trace.f32_sched_ns, timing.sched_ns);
            if (rc != 0) tinc(&trace.shim_err, 1);
        }
        return rc == 0;
    }

    fn destroy(ctx: *anyopaque) void {
        const self: *MetalWork = @ptrCast(@alignCast(ctx));
        fucina_metal_ticket_free(self.ticket);
        self.a_buffer.release();
        if (self.b_buffer) |buffer| buffer.release();
        std.heap.c_allocator.destroy(self);
    }
};

/// Submit one f32 GEMM immediately and attach its completion token to `out`.
/// This is eager execution, not lazy evaluation: the command is committed
/// before return.  A CPU read/CPU kernel waits through `Buffer.waitReady`,
/// while another Metal GEMM relies on the persistent command queue's order.
pub fn gemmBatchedF32Async(
    orient: Orient,
    a: *const Tensor,
    b: *const Tensor,
    out: *Tensor,
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
) bool {
    if (batch_count == 0 or m == 0 or n == 0 or k == 0) return false;
    if (m > std.math.maxInt(i32) or n > std.math.maxInt(i32) or k > std.math.maxInt(i32)) return false;
    if (batch_count > std.math.maxInt(i32)) return false;
    const block_a = std.math.mul(usize, m, k) catch return false;
    const block_b = std.math.mul(usize, k, n) catch return false;
    const block_c = std.math.mul(usize, m, n) catch return false;
    const total_a = std.math.add(usize, std.math.mul(usize, stride_a, batch_count - 1) catch return false, block_a) catch return false;
    const total_b = std.math.add(usize, std.math.mul(usize, stride_b, batch_count - 1) catch return false, block_b) catch return false;
    const total_c = std.math.add(usize, std.math.mul(usize, stride_c, batch_count - 1) catch return false, block_c) catch return false;
    if (a.offset + total_a > a.buffer.data.len or b.offset + total_b > b.buffer.data.len or out.offset + total_c > out.buffer.data.len) return false;
    if (out.buffer.pending() != null) return false;

    const ctx = context() orelse return false;
    const holder = std.heap.c_allocator.create(MetalWork) catch return false;
    const submit_started = tstart();
    const ticket = fucina_metal_gemm_f32_async(
        ctx,
        @intFromEnum(orient),
        a.buffer.data[a.offset..].ptr,
        b.buffer.data[b.offset..].ptr,
        out.buffer.data[out.offset..].ptr,
        storageWrap(a.buffer),
        storageWrap(b.buffer),
        storageWrap(out.buffer),
        @intCast(m),
        @intCast(n),
        @intCast(k),
        @intCast(batch_count),
        @intCast(stride_a),
        @intCast(stride_b),
        @intCast(stride_c),
    ) orelse {
        std.heap.c_allocator.destroy(holder);
        return false;
    };
    a.buffer.retain();
    b.buffer.retain();
    holder.* = .{
        .work = accelerator.Work.init(.metal, holder, &MetalWork.vtable),
        .ticket = ticket,
        .a_buffer = a.buffer,
        .b_buffer = b.buffer,
    };
    a.buffer.setPendingUse(&holder.work);
    b.buffer.setPendingUse(&holder.work);
    out.buffer.setPending(&holder.work);
    if (trace_on) {
        tinc(&trace.f32_async_calls, 1);
        tinc(&trace.f32_submit_ns, tfinish(submit_started));
    }
    return true;
}

pub fn gemmF32Async(orient: Orient, a: *const Tensor, b: *const Tensor, out: *Tensor, m: usize, n: usize, k: usize) bool {
    return gemmBatchedF32Async(orient, a, b, out, m, n, k, 1, 0, 0, 0);
}

const MetalF16Work = struct {
    work: accelerator.Work,
    ticket: *anyopaque,
    a_buffer: *storage.BufferOf(.f16),
    b_buffer: *storage.BufferOf(.f16),

    const vtable: accelerator.WorkVTable = .{
        .finish = finish,
        .device_ptr = null,
        .destroy = destroy,
    };

    fn finish(ctx: *anyopaque, _: bool) bool {
        const self: *MetalF16Work = @ptrCast(@alignCast(ctx));
        defer {
            self.a_buffer.clearPendingUse(&self.work);
            self.b_buffer.clearPendingUse(&self.work);
        }
        var timing: CommandTiming = .{ .gpu_ns = 0, .sched_ns = 0 };
        const wait_started = tstart();
        const rc = fucina_metal_ticket_wait(self.ticket, if (trace_on) &timing else null);
        if (trace_on) {
            tinc(&trace.f16_wait_ns, tfinish(wait_started));
            tinc(&trace.f16_gpu_ns, timing.gpu_ns);
            tinc(&trace.f16_sched_ns, timing.sched_ns);
            if (rc != 0) tinc(&trace.shim_err, 1);
        }
        return rc == 0;
    }

    fn destroy(ctx: *anyopaque) void {
        const self: *MetalF16Work = @ptrCast(@alignCast(ctx));
        fucina_metal_ticket_free(self.ticket);
        self.a_buffer.release();
        self.b_buffer.release();
        std.heap.c_allocator.destroy(self);
    }
};

/// Submit f16 A/B NT GEMM immediately and write its f32 result directly into
/// `out`.  There is no shared staging buffer: input/output storage mappings
/// live with their allocations and the ordinary output Work is the only
/// completion state.
pub fn gemmF16NtAsync(a: *const TensorF16, b: *const TensorF16, out: *Tensor, m: usize, n: usize, k: usize) bool {
    if (m == 0 or n == 0 or k == 0) return false;
    if (m > std.math.maxInt(i32) or n > std.math.maxInt(i32) or k > std.math.maxInt(i32)) return false;
    const a_elems = std.math.mul(usize, m, k) catch return false;
    const b_elems = std.math.mul(usize, n, k) catch return false;
    const c_elems = std.math.mul(usize, m, n) catch return false;
    if (a.offset + a_elems > a.buffer.data.len or b.offset + b_elems > b.buffer.data.len or out.offset + c_elems > out.buffer.data.len) return false;
    if (out.buffer.pending() != null) return false;

    const ctx = context() orelse return false;
    const holder = std.heap.c_allocator.create(MetalF16Work) catch return false;
    const submit_started = tstart();
    const ticket = fucina_metal_gemm_f16_nt_async(
        ctx,
        a.buffer.data[a.offset..].ptr,
        b.buffer.data[b.offset..].ptr,
        out.buffer.data[out.offset..].ptr,
        storageWrap(a.buffer),
        storageWrap(b.buffer),
        storageWrap(out.buffer),
        @intCast(m),
        @intCast(n),
        @intCast(k),
    ) orelse {
        std.heap.c_allocator.destroy(holder);
        return false;
    };
    a.buffer.retain();
    b.buffer.retain();
    holder.* = .{
        .work = accelerator.Work.init(.metal, holder, &MetalF16Work.vtable),
        .ticket = ticket,
        .a_buffer = a.buffer,
        .b_buffer = b.buffer,
    };
    a.buffer.setPendingUse(&holder.work);
    b.buffer.setPendingUse(&holder.work);
    out.buffer.setPending(&holder.work);
    if (trace_on) {
        tinc(&trace.f16_async_calls, 1);
        tinc(&trace.f16_submit_ns, tfinish(submit_started));
    }
    return true;
}

const MetalBf16Work = struct {
    work: accelerator.Work,
    ticket: *anyopaque,
    /// Per-call bf16 copy of the activations — privately owned (no pending
    /// marks: nothing else can observe it), released with the Work.
    a_scratch: *storage.BufferOf(.bf16),
    b_buffer: *storage.BufferOf(.bf16),

    const vtable: accelerator.WorkVTable = .{
        .finish = finish,
        .device_ptr = null,
        .destroy = destroy,
    };

    fn finish(ctx: *anyopaque, _: bool) bool {
        const self: *MetalBf16Work = @ptrCast(@alignCast(ctx));
        defer self.b_buffer.clearPendingUse(&self.work);
        var timing: CommandTiming = .{ .gpu_ns = 0, .sched_ns = 0 };
        const wait_started = tstart();
        const rc = fucina_metal_ticket_wait(self.ticket, if (trace_on) &timing else null);
        if (trace_on) {
            tinc(&trace.bf16_wait_ns, tfinish(wait_started));
            tinc(&trace.bf16_gpu_ns, timing.gpu_ns);
            tinc(&trace.bf16_sched_ns, timing.sched_ns);
            if (rc != 0) tinc(&trace.shim_err, 1);
        }
        return rc == 0;
    }

    fn destroy(ctx: *anyopaque) void {
        const self: *MetalBf16Work = @ptrCast(@alignCast(ctx));
        fucina_metal_ticket_free(self.ticket);
        self.a_scratch.release();
        self.b_buffer.release();
        std.heap.c_allocator.destroy(self);
    }
};

/// C[m,n] (f32) = A[m,k] (f32 activations, cast per call to bf16 rows —
/// round-to-nearest-even, the rows a torch bf16 forward feeds its GEMMs) ·
/// B[n,k]ᵀ (bf16 weights, zero-copy page wrap — mutation-safe like every
/// other wrap; no converted weight copy exists anywhere). Returns false when
/// the GPU didn't run and the caller must take the CPU bf16 streaming
/// kernel.
pub fn gemmBf16NtAsync(a: *const Tensor, b: *const TensorBf16, out: *Tensor, m: usize, n: usize, k: usize) bool {
    const dtype_mod = @import("../dtype.zig");
    if (m == 0 or n == 0 or k == 0) return false;
    if (m > std.math.maxInt(i32) or n > std.math.maxInt(i32) or k > std.math.maxInt(i32)) return false;
    const a_elems = std.math.mul(usize, m, k) catch return false;
    const b_elems = std.math.mul(usize, n, k) catch return false;
    const c_elems = std.math.mul(usize, m, n) catch return false;
    if (a.offset + a_elems > a.buffer.data.len or b.offset + b_elems > b.buffer.data.len or out.offset + c_elems > out.buffer.data.len) return false;
    if (out.buffer.pending() != null) return false;

    const ctx = context() orelse return false;

    // The conversion reads A on the HOST at submit time, so any pending
    // device producer must be complete first (unlike the wraps, which the
    // in-order queue serializes device-side).
    @constCast(a.buffer).waitReady();
    const scratch = storage.BufferOf(.bf16).create(std.heap.c_allocator, a_elems) catch return false;
    var scratch_owned = true;
    defer if (scratch_owned) scratch.release();
    for (scratch.data, a.buffer.data[a.offset..][0..a_elems]) |*dst, src| {
        dst.* = dtype_mod.f32ToBf16(src);
    }

    const holder = std.heap.c_allocator.create(MetalBf16Work) catch return false;
    const submit_started = tstart();
    const ticket = fucina_metal_gemm_bf16_nt_async(
        ctx,
        scratch.data.ptr,
        b.buffer.data[b.offset..].ptr,
        out.buffer.data[out.offset..].ptr,
        storageWrap(scratch),
        storageWrap(b.buffer),
        storageWrap(out.buffer),
        @intCast(m),
        @intCast(n),
        @intCast(k),
    ) orelse {
        std.heap.c_allocator.destroy(holder);
        return false;
    };
    b.buffer.retain();
    holder.* = .{
        .work = accelerator.Work.init(.metal, holder, &MetalBf16Work.vtable),
        .ticket = ticket,
        .a_scratch = scratch,
        .b_buffer = b.buffer,
    };
    scratch_owned = false;
    b.buffer.setPendingUse(&holder.work);
    out.buffer.setPending(&holder.work);
    if (trace_on) {
        tinc(&trace.bf16_async_calls, 1);
        tinc(&trace.bf16_submit_ns, tfinish(submit_started));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Quantized (dequant-in-kernel) grouped GEMM — the MoE prefill path.
// Kernel: metal/ggml_mul_mm.metal (vendored llama.cpp legacy mul_mm).
// ---------------------------------------------------------------------------

/// Weight block formats the quantized kernel can read directly.
/// Must mirror the FUCINA_QFMT_* enum in shim.m.
pub const QFormat = enum(c_int) {
    q8_0 = 0,
    q6_k = 1,
    q4_k = 2,
    tq2_0 = 3,

    /// K (the reduced dim) must be a whole number of blocks.
    pub fn kMultiple(self: QFormat) usize {
        return switch (self) {
            .q8_0 => 32,
            .q6_k => 256,
            .q4_k => 256,
            .tq2_0 => 256,
        };
    }
};

/// One 32-row output tile of one expert group. Must mirror FucinaQMMTile in
/// shim.m / fucina_qmm_tile in the kernel.
pub const QMMTile = extern struct {
    expert: i32,
    base_row: i32,
    m: i32,
    tile_m: i32,
};

const MetalQuantWork = struct {
    work: accelerator.Work,
    ticket: *anyopaque,
    input_buffer: *storage.Buffer,

    const vtable: accelerator.WorkVTable = .{
        .finish = finish,
        .device_ptr = null,
        .destroy = destroy,
    };

    fn finish(ctx: *anyopaque, _: bool) bool {
        const self: *MetalQuantWork = @ptrCast(@alignCast(ctx));
        defer self.input_buffer.clearPendingUse(&self.work);
        var timing: CommandTiming = .{ .gpu_ns = 0, .sched_ns = 0 };
        const wait_started = tstart();
        const rc = fucina_metal_ticket_wait(self.ticket, if (trace_on) &timing else null);
        if (trace_on) {
            tinc(&trace.quant_wait_ns, tfinish(wait_started));
            tinc(&trace.quant_gpu_ns, timing.gpu_ns);
            tinc(&trace.quant_sched_ns, timing.sched_ns);
            if (rc != 0) tinc(&trace.shim_err, 1);
        }
        return rc == 0;
    }

    fn destroy(ctx: *anyopaque) void {
        const self: *MetalQuantWork = @ptrCast(@alignCast(ctx));
        fucina_metal_ticket_free(self.ticket);
        self.input_buffer.release();
        std.heap.c_allocator.destroy(self);
    }
};

/// Eager dense quantized NT GEMM over one input shared by `batch_count`
/// independent weight matrices. Stable model weights are mapped once; tensor
/// input/output storage is used directly and host visibility is deferred to
/// the ordinary output Work. The 4 KiB command-data tile limit admits up to
/// 8192 rows per call; longer rare prompts retain the blocking chunk fallback.
pub fn gemmQuantNtAsync(
    format: QFormat,
    rhs_bytes: []const u8,
    rhs_cacheable: bool,
    nb01: usize,
    nb02: usize,
    input: *const Tensor,
    out: *Tensor,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) bool {
    if (!rhs_cacheable or rhs_bytes.len == 0 or batch_count == 0 or m == 0 or m > 8192 or n == 0 or k == 0) return false;
    if (m > std.math.maxInt(i32) or n > std.math.maxInt(i32) or k > std.math.maxInt(i32) or batch_count > std.math.maxInt(i32)) return false;
    if (k % 32 != 0 or k % format.kMultiple() != 0 or n % 4 != 0) return false;
    const input_elems = std.math.mul(usize, m, k) catch return false;
    const output_rows = std.math.mul(usize, batch_count, m) catch return false;
    const output_elems = std.math.mul(usize, output_rows, n) catch return false;
    if (input.offset + input_elems > input.buffer.data.len or out.offset + output_elems > out.buffer.data.len) return false;
    if (out.buffer.pending() != null) return false;

    const ctx = context() orelse return false;
    const holder = std.heap.c_allocator.create(MetalQuantWork) catch return false;
    const submit_started = tstart();
    const ticket = fucina_metal_gemm_q_dense_nt_async(
        ctx,
        @intFromEnum(format),
        rhs_bytes.ptr,
        @intCast(rhs_bytes.len),
        @intCast(nb01),
        @intCast(nb02),
        input.buffer.data[input.offset..].ptr,
        out.buffer.data[out.offset..].ptr,
        storageWrap(input.buffer),
        storageWrap(out.buffer),
        @intCast(batch_count),
        @intCast(m),
        @intCast(n),
        @intCast(k),
    ) orelse {
        std.heap.c_allocator.destroy(holder);
        return false;
    };
    input.buffer.retain();
    holder.* = .{
        .work = accelerator.Work.init(.metal, holder, &MetalQuantWork.vtable),
        .ticket = ticket,
        .input_buffer = input.buffer,
    };
    input.buffer.setPendingUse(&holder.work);
    out.buffer.setPending(&holder.work);
    if (trace_on) {
        tinc(&trace.quant_async_calls, 1);
        tinc(&trace.quant_submit_ns, tfinish(submit_started));
        traceRhsCache(true);
    }
    return true;
}

/// Device-owned byte storage (page-aligned; GPU-resident until explicitly freed)
/// across command buffers; the CPU reads the same bytes through the returned
/// slice — unified memory). This is a performance cache for stable RHS bytes, not
/// a precondition for GPU correctness: non-resident operands can still be wrapped
/// uncached by the dispatch path, or the caller can fall back to CPU. The reason
/// to prefer this for reused operands is that client-memory page wraps are
/// pageable, and Metal re-wires them into the GPU address space on every commit
/// (~45 µs/MB measured — tens of ms per dispatch on the MoE expert tensors).
/// Null when the GPU is unavailable or the bounded wrap-cache cannot register
/// another resident buffer.
/// ES device arm (see cuda.zig): not needed on unified memory — the CPU
/// kernels already mutate shared pages the GPU reads zero-copy — so the
/// stubs report "not handled" and callers keep their CPU path.
pub const EsDType = enum(usize) { f16 = 0, f32 = 1 };

pub fn esPerturb(dt: EsDType, bytes: []u8, stream_seed: u64, scaled: f32, n: usize) bool {
    _ = dt;
    _ = bytes;
    _ = stream_seed;
    _ = scaled;
    _ = n;
    return false;
}

pub fn esUpdate(dt: EsDType, bytes: []u8, stream_seeds: []const u64, coeffs: []const f32, scale: f32, n: usize) bool {
    _ = dt;
    _ = bytes;
    _ = stream_seeds;
    _ = coeffs;
    _ = scale;
    _ = n;
    return false;
}

pub fn esAnchor(dt: EsDType, bytes: []u8, anchor: []const u8, decay_step: f32, is_l1: bool, n: usize) bool {
    _ = dt;
    _ = bytes;
    _ = anchor;
    _ = decay_step;
    _ = is_l1;
    _ = n;
    return false;
}

pub fn allocResidentBytes(len: usize) ?[]u8 {
    if (len == 0 or len > std.math.maxInt(i64)) return null;
    const ctx = context() orelse return null;
    const p = fucina_metal_alloc_resident_bytes(ctx, @intCast(len)) orelse return null;
    registerResidentRange(@intFromPtr(p), len);
    if (trace_on) {
        tinc(&trace.dev_alloc_calls, 1);
        tinc(&trace.dev_alloc_bytes, len);
    }
    return p[0..len];
}

/// Zig-side ranges of the shim's resident allocations, so dispatch paths can
/// recognize a resident operand WITHOUT the caller flagging it (the generic
/// f16/f32 Tensor paths pass rhs_cacheable=false — they cannot prove
/// process-lifetime stability, but bytes inside a resident allocation carry
/// that proof themselves: the ADDRESS is stable for the process even when
/// the contents mutate, e.g. weights trained in place by fucina.es, and the
/// shim's cached wrap reads the live unified-memory pages).
const resident_range_slots = 512;
var resident_ranges_lock: thread.Mutex = .{};
var resident_ranges: [resident_range_slots]struct { base: usize, len: usize } = undefined;
var resident_range_count: usize = 0;

fn registerResidentRange(base: usize, len: usize) void {
    resident_ranges_lock.lock();
    defer resident_ranges_lock.unlock();
    if (resident_range_count >= resident_range_slots) return; // lookup misses stay correct
    resident_ranges[resident_range_count] = .{ .base = base, .len = len };
    resident_range_count += 1;
}

fn unregisterResidentRange(base: usize) void {
    resident_ranges_lock.lock();
    defer resident_ranges_lock.unlock();
    for (resident_ranges[0..resident_range_count], 0..) |range, i| {
        if (range.base == base) {
            resident_ranges[i] = resident_ranges[resident_range_count - 1];
            resident_range_count -= 1;
            return;
        }
    }
}

/// Whether `bytes` lies fully inside a live resident allocation.
fn isResidentRange(bytes: []const u8) bool {
    const base = @intFromPtr(bytes.ptr);
    resident_ranges_lock.lock();
    defer resident_ranges_lock.unlock();
    for (resident_ranges[0..resident_range_count]) |range| {
        if (base >= range.base and base + bytes.len <= range.base + range.len) return true;
    }
    return false;
}

/// Release bytes returned by `allocResidentBytes`. Safe no-op when the Metal
/// context is gone or the slice did not come from the resident allocator.
pub fn freeResidentBytes(bytes: []const u8) void {
    if (bytes.len == 0) return;
    unregisterResidentRange(@intFromPtr(bytes.ptr));
    const ctx = state.ctx orelse return;
    _ = fucina_metal_free_resident_bytes(ctx, bytes.ptr);
}

/// Process-global serialization for quantized-GEMM staging panels. Hold across
/// `qmoeStage` + CPU panel writes + `gemmQGroupedNt` dispatches + readback:
/// the shim owns one grow-only in/out panel pair and the command is blocking.
/// This deliberately matches Fucina's eager BLAS-like contract; any future
/// concurrent/async GPU runtime should replace this whole staging contract.
pub var qmoe_lock: thread.Mutex = .{};

pub const QMoeStage = struct {
    in: [*]f32,
    out: [*]f32,
};

/// Acquire the staging panels (grow-only shared MTLBuffers): `in` for the
/// activation rows the CPU gathers, `out` for the GEMM results. Pointers stay
/// valid until the next `qmoeStage` call — hold `qmoe_lock` across the whole
/// stage/dispatch/readback sequence. Null when the GPU is unavailable.
pub fn qmoeStage(in_bytes: usize, out_bytes: usize) ?QMoeStage {
    const ctx = context() orelse return null;
    var in_ptr: ?*anyopaque = null;
    var out_ptr: ?*anyopaque = null;
    const rc = fucina_metal_qmoe_stage(ctx, @intCast(in_bytes), @intCast(out_bytes), &in_ptr, &out_ptr);
    if (rc != 0) return null;
    return .{
        .in = @ptrCast(@alignCast(in_ptr.?)),
        .out = @ptrCast(@alignCast(out_ptr.?)),
    };
}

/// Grouped NT GEMM over the staged panels: for every tile, panel rows
/// `[base_row, base_row+m)` of expert `expert` produce
/// `out[row, 0..n_out) = in[row, 0..k) · dequant(W[expert])ᵀ`.
/// `rhs_bytes` = raw quantized blocks, row-major `[n_out, k]` per expert with
/// uniform byte strides `nb01` (row) / `nb02` (expert). `rhs_cacheable` must
/// only be true for stable storage: resident device-owned bytes
/// (`allocResidentBytes`) whose owner evicts via `freeResidentBytes` before
/// freeing. A cached wrap of freed-and-reused pages reads stale data.
/// Returns false when the GPU didn't run — caller falls back to CPU.
pub fn gemmQGroupedNt(
    format: QFormat,
    rhs_bytes: []const u8,
    rhs_cacheable: bool,
    nb01: usize,
    nb02: usize,
    n_out: usize,
    k: usize,
    tiles: []const QMMTile,
) bool {
    if (k == 0 or k % 32 != 0 or k % format.kMultiple() != 0) return false;
    if (n_out == 0 or n_out % 4 != 0) return false; // float4 row copies in the store
    if (tiles.len == 0) return false;
    const ctx = context() orelse return false;
    var timing: CommandTiming = .{ .gpu_ns = 0, .sched_ns = 0 };
    const timer = tstart();
    const rc = fucina_metal_gemm_q_grouped_nt(
        ctx,
        @intFromEnum(format),
        rhs_bytes.ptr,
        @intCast(rhs_bytes.len),
        @intFromBool(rhs_cacheable),
        @intCast(nb01),
        @intCast(nb02),
        @intCast(n_out),
        @intCast(k),
        tiles.ptr,
        @intCast(tiles.len),
        if (trace_on) &timing else null,
    );
    if (trace_on) {
        const wall_ns = tfinish(timer);
        tinc(&trace.quant_ns, wall_ns);
        tinc(&trace.quant_gpu_ns, timing.gpu_ns);
        tinc(&trace.quant_sched_ns, timing.sched_ns);
        tinc(&trace.quant_calls, 1);
        traceRhsCache(rhs_cacheable);
        traceRecordShape(.quant, rowsCoveredByTiles(tiles), n_out, k, 1, tiles.len, wall_ns, timing);
        if (rc != 0) tinc(&trace.shim_err, 1);
    }
    return rc == 0;
}

fn rowsCoveredByTiles(tiles: []const QMMTile) usize {
    var end: usize = 0;
    for (tiles) |tile| {
        if (tile.base_row < 0 or tile.m < 0) continue;
        const base: usize = @intCast(tile.base_row);
        const rows: usize = @intCast(tile.m);
        end = @max(end, base + rows);
    }
    return end;
}

/// Single quantized NT GEMM with host-memory operands, staged through the
/// shared panels: `c[m,n] = a[m,k] · dequant(W)ᵀ` (one "expert"). Convenience
/// for parity tests and dense offload; takes `qmoe_lock` itself.
/// `rhs_cacheable` routes the RHS wrap through the page cache — pass true
/// ONLY for stable storage (`internal.gpu.allocResidentBytes`), false for
/// transient buffers. A cached wrap of a freed-and-reused page reads stale
/// data.
pub fn gemmQuantNt(
    format: QFormat,
    rhs_bytes: []const u8,
    rhs_cacheable: bool,
    nb01: usize,
    a: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
) bool {
    if (m == 0 or m > std.math.maxInt(i32)) return false;
    const in_elems = std.math.mul(usize, m, k) catch return false;
    const out_elems = std.math.mul(usize, m, n) catch return false;
    if (a.len < in_elems or c.len < out_elems) return false;
    const in_bytes = std.math.mul(usize, in_elems, @sizeOf(f32)) catch return false;
    const out_bytes = std.math.mul(usize, out_elems, @sizeOf(f32)) catch return false;

    const lock_timer = tstart();
    qmoe_lock.lock();
    defer qmoe_lock.unlock();
    telapsed(&trace.quant_lock_ns, lock_timer);
    const stage = qmoeStage(in_bytes, out_bytes) orelse return false;
    const in_timer = tstart();
    @memcpy(stage.in[0..in_elems], a[0..in_elems]);
    telapsed(&trace.quant_stage_ns, in_timer);
    var tiles_buf: [64]QMMTile = undefined;
    const n_tiles = (m + 31) / 32;
    if (n_tiles > tiles_buf.len) return false;
    for (0..n_tiles) |t| {
        tiles_buf[t] = .{ .expert = 0, .base_row = 0, .m = @intCast(m), .tile_m = @intCast(t) };
    }
    if (!gemmQGroupedNt(format, rhs_bytes, rhs_cacheable, nb01, 0, n, k, tiles_buf[0..n_tiles])) return false;
    const out_timer = tstart();
    @memcpy(c[0..out_elems], stage.out[0..out_elems]);
    telapsed(&trace.quant_stage_ns, out_timer);
    return true;
}

/// Batched dense quantized NT GEMM over one shared activation matrix:
/// for each batch `b`, `c[b,m,n] = a[m,k] · dequant(W[b,n,k])^T`.
/// This is the narrow eager command-batching seam: it uses the existing grouped
/// quant kernel's expert dimension to collapse same-shape independent linears
/// into one Metal command, while the caller still owns an ordinary CPU-visible
/// result tensor. `nb02` is the byte stride between consecutive RHS operands.
pub fn gemmQuantNtSharedABatch(
    format: QFormat,
    rhs_bytes: []const u8,
    rhs_cacheable: bool,
    nb01: usize,
    nb02: usize,
    a: []const f32,
    c: []f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) bool {
    if (batch_count == 0 or m == 0 or m > std.math.maxInt(i32)) return false;
    const in_elems = std.math.mul(usize, m, k) catch return false;
    const rows_total = std.math.mul(usize, batch_count, m) catch return false;
    if (rows_total > std.math.maxInt(i32)) return false;
    const out_elems = std.math.mul(usize, rows_total, n) catch return false;
    if (a.len < in_elems or c.len < out_elems) return false;
    const n_tiles_per_batch = (m + 31) / 32;
    const n_tiles_total = std.math.mul(usize, n_tiles_per_batch, batch_count) catch return false;
    var tiles_buf: [2048]QMMTile = undefined;
    if (n_tiles_total > tiles_buf.len) return false;

    const lock_timer = tstart();
    qmoe_lock.lock();
    defer qmoe_lock.unlock();
    telapsed(&trace.quant_lock_ns, lock_timer);

    const row_bytes = std.math.mul(usize, k, @sizeOf(f32)) catch return false;
    const in_bytes = std.math.mul(usize, rows_total, row_bytes) catch return false;
    const out_bytes = std.math.mul(usize, out_elems, @sizeOf(f32)) catch return false;
    const stage = qmoeStage(in_bytes, out_bytes) orelse return false;
    const in_timer = tstart();
    for (0..batch_count) |bi| {
        @memcpy(stage.in[bi * in_elems ..][0..in_elems], a[0..in_elems]);
    }
    telapsed(&trace.quant_stage_ns, in_timer);

    var tile_i: usize = 0;
    for (0..batch_count) |bi| {
        const base_row = bi * m;
        for (0..n_tiles_per_batch) |t| {
            tiles_buf[tile_i] = .{
                .expert = @intCast(bi),
                .base_row = @intCast(base_row),
                .m = @intCast(m),
                .tile_m = @intCast(t),
            };
            tile_i += 1;
        }
    }
    if (!gemmQGroupedNt(format, rhs_bytes, rhs_cacheable, nb01, nb02, n, k, tiles_buf[0..n_tiles_total])) return false;
    const out_timer = tstart();
    @memcpy(c[0..out_elems], stage.out[0..out_elems]);
    telapsed(&trace.quant_stage_ns, out_timer);
    return true;
}

pub fn shouldUseGpuQMoe(total_work: u64) bool {
    ensureConfig();
    const pass = state.gpu_enabled and total_work >= state.min_work_qmoe;
    tgate(pass);
    return pass;
}

/// Occupancy arm of the grouped-MoE gate: `rows` real panel rows spread over
/// `n_tiles` 32-row token tiles must reach the configured minimum fill
/// percentage (see `default_qmoe_min_fill_pct`). Callers pass the exact tile
/// table they are about to dispatch.
pub fn qmoeFillAcceptable(rows: usize, n_tiles: usize) bool {
    ensureConfig();
    if (n_tiles == 0) return false;
    const filled = std.math.mul(u64, @as(u64, rows), 100) catch return true;
    return filled >= @as(u64, n_tiles) * 32 * state.qmoe_min_fill_pct;
}

pub fn shouldUseGpuDenseQuant(format: QFormat, total_work: u64) bool {
    ensureConfig();
    const min_work = switch (format) {
        .q6_k => state.min_work_dense_q6,
        .q4_k, .q8_0, .tq2_0 => state.min_work_qmoe,
    };
    const pass = state.gpu_enabled and total_work >= min_work;
    tgate(pass);
    return pass;
}

/// Dense model-weight gate against the load-time-packed CPU fallback.
pub fn shouldUseGpuDenseQuantPacked(format: QFormat, total_work: u64) bool {
    ensureConfig();
    const min_work = switch (format) {
        .q4_k => state.min_work_packed_q4,
        .q6_k => state.min_work_packed_q6,
        .q8_0 => state.min_work_packed_q8,
        .tq2_0 => state.min_work_packed_tq2,
    };
    const pass = state.gpu_enabled and total_work >= min_work;
    tgate(pass);
    return pass;
}

/// Test seam: unit-test shapes never reach the real threshold.
pub fn setMinWorkQMoeForTest(v: u64) void {
    ensureConfig();
    state.min_work_qmoe = v;
}

/// Decode-GEMV capability gate (exec's m <= 8 arm in denseQuantMatmulGpu).
/// The Metal provider keeps decode on CPU by design — always false here;
/// the CUDA provider opts in via FUCINA_GPU_DECODE=1.
pub fn decodeGemvEnabled() bool {
    return false;
}

pub fn shouldUseGpuQuantDecode(format: QFormat, m: usize, n: usize, k: usize) bool {
    _ = format;
    _ = m;
    _ = n;
    _ = k;
    return false;
}

/// Prefill-attention offload gate (exec's seam in
/// groupedCausalAttentionTiledRun) — not implemented on Metal; the CPU tiled
/// kernel runs. The CUDA provider implements the arm.
pub fn shouldUseGpuAttn(q_seq: usize, kv_seq: usize, heads: usize, d: usize) bool {
    _ = q_seq;
    _ = kv_seq;
    _ = heads;
    _ = d;
    return false;
}

/// Not implemented on Metal — always false (the gate above already refuses).
pub fn attnPrefillF16(
    q: []const f32,
    k: []const f16,
    v: []const f16,
    out: []f32,
    kv_head_for_head: []const i32,
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    source_offset: usize,
    scale: f32,
    window: usize,
    causal: bool,
) bool {
    _ = q;
    _ = k;
    _ = v;
    _ = out;
    _ = kv_head_for_head;
    _ = q_seq;
    _ = kv_seq;
    _ = heads;
    _ = kv_heads;
    _ = d;
    _ = source_offset;
    _ = scale;
    _ = window;
    _ = causal;
    return false;
}

// ---------------------------------------------------------------------------
// Tests (compiled and run only on -Dgpu=metal builds)
// ---------------------------------------------------------------------------

fn cpuReference(orient: Orient, a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f64 = 0;
            for (0..k) |p| {
                const av: f64 = switch (orient) {
                    .nn, .nt => a[i * k + p],
                    .tn => a[p * m + i],
                };
                const bv: f64 = switch (orient) {
                    .nn, .tn => b[p * n + j],
                    .nt => b[j * k + p],
                };
                acc += av * bv;
            }
            c[i * n + j] = @floatCast(acc);
        }
    }
}

test "metal gemm f32 parity vs reference (all orientations, edge tiles)" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(3);
    const random = prng.random();

    const Case = struct { m: usize, n: usize, k: usize };
    const cases = [_]Case{
        .{ .m = 64, .n = 64, .k = 64 }, // fully aligned
        .{ .m = 33, .n = 47, .k = 17 }, // every edge path
        .{ .m = 128, .n = 96, .k = 33 }, // K tail only
        .{ .m = 65, .n = 64, .k = 48 }, // M edge only
    };
    for (cases) |case| {
        const m = case.m;
        const n = case.n;
        const k = case.k;
        const a = try allocator.alloc(f32, m * k);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, k * n);
        defer allocator.free(b);
        const c = try allocator.alloc(f32, m * n);
        defer allocator.free(c);
        const expected = try allocator.alloc(f32, m * n);
        defer allocator.free(expected);

        for ([_]Orient{ .nn, .tn, .nt }) |orient| {
            for (a) |*x| x.* = random.floatNorm(f32);
            for (b) |*x| x.* = random.floatNorm(f32);
            @memset(c, std.math.nan(f32));
            try std.testing.expect(gemmF32(orient, a, b, c, m, n, k));
            cpuReference(orient, a, b, expected, m, n, k);
            for (c, expected) |got, want| {
                const tol = @max(2e-5 * @max(@abs(want), @abs(got)), 2e-5);
                try std.testing.expect(@abs(got - want) <= tol);
            }
        }
    }
}

test "metal gemm f16 NT parity vs f64 reference (f16-rounded output)" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(9);
    const random = prng.random();

    const Case = struct { m: usize, n: usize, k: usize };
    const cases = [_]Case{
        .{ .m = 64, .n = 64, .k = 64 },
        .{ .m = 33, .n = 47, .k = 17 },
        .{ .m = 65, .n = 96, .k = 33 },
    };
    for (cases) |case| {
        const m = case.m;
        const n = case.n;
        const k = case.k;
        const a = try allocator.alloc(f16, m * k);
        defer allocator.free(a);
        const b = try allocator.alloc(f16, n * k);
        defer allocator.free(b);
        for (a) |*x| x.* = @floatCast(random.floatNorm(f32));
        for (b) |*x| x.* = @floatCast(random.floatNorm(f32));

        f16_lock.lock();
        defer f16_lock.unlock();
        // transient test buffer: must not enter the wrap cache
        const staging = gemmF16Nt(a, b, m, n, k, false) orelse return error.TestUnexpectedResult;
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f64 = 0;
                for (0..k) |p| acc += @as(f64, a[i * k + p]) * @as(f64, b[j * k + p]);
                const got: f32 = staging[i * n + j];
                const want: f32 = @floatCast(acc);
                // f32 accumulate, f16-rounded store: tolerance = one f16 ulp
                // of the result magnitude plus accumulation slack.
                const tol = @max(2e-3 * @max(@abs(want), @abs(got)), 2e-3);
                try std.testing.expect(@abs(got - want) <= tol);
            }
        }
    }
}

/// Quantize random rows into `blocks` ([n rows] x [k/block] blocks) and return
/// the dequantized f32 matrix the GPU kernel effectively sees (the kernel's
/// extra f32->f16 rounding is covered by the test tolerances).
fn buildQuantWeights(
    comptime fmt: QFormat,
    allocator: std.mem.Allocator,
    random: std.Random,
    blocks: anytype,
    n: usize,
    k: usize,
) ![]f32 {
    const qm = @import("quant.zig");
    const dt = comptime switch (fmt) {
        .q8_0 => .q8_0,
        .q6_k => .q6_k,
        .q4_k => .q4_k,
        .tq2_0 => .tq2_0,
    };
    const bpr = blocks.len / n;
    const wref = try allocator.alloc(f32, n * k);
    errdefer allocator.free(wref);
    const row_src = try allocator.alloc(f32, k);
    defer allocator.free(row_src);
    for (0..n) |r| {
        for (row_src) |*x| x.* = random.floatNorm(f32);
        try qm.quantizeRowForDType(dt, blocks[r * bpr ..][0..bpr], row_src);
        try qm.dequantizeRowForDType(dt, wref[r * k ..][0..k], blocks[r * bpr ..][0..bpr]);
    }
    return wref;
}

fn expectQuantGemmRows(
    a: []const f32,
    wref: []const f32,
    c: []const f32,
    m: usize,
    n: usize,
    k: usize,
) !void {
    for (0..m) |i| {
        for (0..n) |j| {
            // reference over f16-rounded operands — the GPU stores both the
            // dequantized weights and the f32 activations as half in
            // threadgroup memory and accumulates in f32
            var acc: f64 = 0;
            for (0..k) |p| {
                const av: f16 = @floatCast(a[i * k + p]);
                const wv: f16 = @floatCast(wref[j * k + p]);
                acc += @as(f64, av) * @as(f64, wv);
            }
            const got: f32 = c[i * n + j];
            const want: f32 = @floatCast(acc);
            const tol = @max(5e-3 * @max(@abs(want), @abs(got)), 5e-3);
            if (@abs(got - want) > tol) {
                std.debug.print(
                    "quant gemm mismatch m={d} n={d} k={d} at ({d},{d}): got={e} want={e}\n",
                    .{ m, n, k, i, j, got, want },
                );
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "metal quant gemm q6_K/q4_K/q8_0 parity vs dequantized reference" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    const dtype_mod = @import("../dtype.zig");
    var prng = std.Random.DefaultPrng.init(17);
    const random = prng.random();

    const Case = struct { m: usize, n: usize, k: usize };
    inline for (.{ QFormat.q6_k, QFormat.q4_k, QFormat.q8_0, QFormat.tq2_0 }) |fmt| {
        const Block = switch (fmt) {
            .q6_k => dtype_mod.BlockQ6_K,
            .q4_k => dtype_mod.BlockQ4_K,
            .q8_0 => dtype_mod.BlockQ8_0,
            .tq2_0 => dtype_mod.BlockTQ2_0,
        };
        const k_mult = comptime fmt.kMultiple();
        const cases = [_]Case{
            .{ .m = 64, .n = 64, .k = 2 * k_mult }, // fully aligned
            .{ .m = 33, .n = 96, .k = k_mult }, // m edge, multiple n tiles
            .{ .m = 1, .n = 68, .k = 2 * k_mult }, // single row, n edge
        };
        for (cases) |case| {
            const m = case.m;
            const n = case.n;
            const k = case.k;
            const bpr = k / k_mult;
            const blocks = try allocator.alloc(Block, n * bpr);
            defer allocator.free(blocks);
            const wref = try buildQuantWeights(fmt, allocator, random, blocks, n, k);
            defer allocator.free(wref);

            const a = try allocator.alloc(f32, m * k);
            defer allocator.free(a);
            for (a) |*x| x.* = random.floatNorm(f32);
            const c = try allocator.alloc(f32, m * n);
            defer allocator.free(c);
            @memset(c, std.math.nan(f32));

            try std.testing.expect(gemmQuantNt(
                fmt,
                std.mem.sliceAsBytes(blocks),
                false, // transient test buffer: must not enter the wrap cache
                bpr * @sizeOf(Block),
                a,
                c,
                m,
                n,
                k,
            ));
            try expectQuantGemmRows(a, wref, c, m, n, k);
        }
    }
}

test "metal quant gemm grouped expert tiles parity" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    const dtype_mod = @import("../dtype.zig");
    var prng = std.Random.DefaultPrng.init(23);
    const random = prng.random();

    inline for (.{ QFormat.q6_k, QFormat.q4_k, QFormat.q8_0, QFormat.tq2_0 }) |fmt| {
        const Block = switch (fmt) {
            .q6_k => dtype_mod.BlockQ6_K,
            .q4_k => dtype_mod.BlockQ4_K,
            .q8_0 => dtype_mod.BlockQ8_0,
            .tq2_0 => dtype_mod.BlockTQ2_0,
        };
        const k = 2 * comptime fmt.kMultiple();
        const n = 64;
        const bpr = k / comptime fmt.kMultiple();
        const n_expert = 4;
        // tile edges: 1 row, sub-tile, exactly one tile + 1, multi-tile
        const ms = [n_expert]usize{ 1, 7, 33, 40 };

        const blocks = try allocator.alloc(Block, n_expert * n * bpr);
        defer allocator.free(blocks);
        const wref = try buildQuantWeights(fmt, allocator, random, blocks, n_expert * n, k);
        defer allocator.free(wref);

        var total_rows: usize = 0;
        var tiles_buf: [16]QMMTile = undefined;
        var n_tiles: usize = 0;
        var bases: [n_expert]usize = undefined;
        for (ms, 0..) |m_e, e| {
            bases[e] = total_rows;
            var t: usize = 0;
            while (t * 32 < m_e) : (t += 1) {
                tiles_buf[n_tiles] = .{
                    .expert = @intCast(e),
                    .base_row = @intCast(total_rows),
                    .m = @intCast(m_e),
                    .tile_m = @intCast(t),
                };
                n_tiles += 1;
            }
            total_rows += m_e;
        }

        const a = try allocator.alloc(f32, total_rows * k);
        defer allocator.free(a);
        for (a) |*x| x.* = random.floatNorm(f32);

        qmoe_lock.lock();
        defer qmoe_lock.unlock();
        const stage = qmoeStage(total_rows * k * @sizeOf(f32), total_rows * n * @sizeOf(f32)) orelse
            return error.TestUnexpectedResult;
        @memcpy(stage.in[0 .. total_rows * k], a);
        try std.testing.expect(gemmQGroupedNt(
            fmt,
            std.mem.sliceAsBytes(blocks),
            false, // transient test buffer: must not enter the wrap cache
            bpr * @sizeOf(Block),
            n * bpr * @sizeOf(Block),
            n,
            k,
            tiles_buf[0..n_tiles],
        ));
        for (ms, 0..) |m_e, e| {
            try expectQuantGemmRows(
                a[bases[e] * k ..][0 .. m_e * k],
                wref[e * n * k ..][0 .. n * k],
                stage.out[bases[e] * n ..][0 .. m_e * n],
                m_e,
                n,
                k,
            );
        }
    }
}

test "metal eager async dense quant Q4_K/Q6_K/Q8_0 uses direct tensor storage" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    const dtype_mod = @import("../dtype.zig");
    var prng = std.Random.DefaultPrng.init(31);
    const random = prng.random();
    inline for (.{ QFormat.q6_k, QFormat.q4_k, QFormat.q8_0, QFormat.tq2_0 }) |fmt| {
        const Block = switch (fmt) {
            .q6_k => dtype_mod.BlockQ6_K,
            .q4_k => dtype_mod.BlockQ4_K,
            .q8_0 => dtype_mod.BlockQ8_0,
            .tq2_0 => dtype_mod.BlockTQ2_0,
        };
        const m = 65;
        const n = 68;
        const k = 2 * comptime fmt.kMultiple();
        const batch_count = 2;
        const bpr = k / comptime fmt.kMultiple();
        const resident = allocResidentBytes(batch_count * n * bpr * @sizeOf(Block)) orelse return error.SkipZigTest;
        defer freeResidentBytes(resident);
        const blocks: []Block = @alignCast(std.mem.bytesAsSlice(Block, resident));
        const wref = try buildQuantWeights(fmt, allocator, random, blocks, batch_count * n, k);
        defer allocator.free(wref);

        const av = try allocator.alloc(f32, m * k);
        defer allocator.free(av);
        for (av) |*x| x.* = random.floatNorm(f32);
        var input = try Tensor.fromSlice(allocator, &.{ m, k }, av);
        defer input.deinit();
        var out = try Tensor.zeros(allocator, &.{ batch_count * m, n });
        defer out.deinit();

        try std.testing.expect(gemmQuantNtAsync(
            fmt,
            resident,
            true,
            bpr * @sizeOf(Block),
            n * bpr * @sizeOf(Block),
            &input,
            &out,
            batch_count,
            m,
            n,
            k,
        ));
        try std.testing.expect(out.buffer.pending() != null);
        input.data()[0] += 100;
        const got = out.dataConst();
        for (0..batch_count) |bi| {
            try expectQuantGemmRows(
                av,
                wref[bi * n * k ..][0 .. n * k],
                got[bi * m * n ..][0 .. m * n],
                m,
                n,
                k,
            );
        }
    }
}

test "qmoe fill gate arithmetic" {
    if (comptime !enabled) return error.SkipZigTest;
    ensureConfig();
    const saved = state.qmoe_min_fill_pct;
    defer state.qmoe_min_fill_pct = saved;

    state.qmoe_min_fill_pct = 50;
    // 2048 rows over 128 tiles = exactly 50% of the 32-row slots
    try std.testing.expect(qmoeFillAcceptable(2048, 128));
    try std.testing.expect(!qmoeFillAcceptable(2047, 128));
    // full tiles always pass
    try std.testing.expect(qmoeFillAcceptable(4096, 128));
    // empty tile table never dispatches
    try std.testing.expect(!qmoeFillAcceptable(0, 0));

    state.qmoe_min_fill_pct = 0; // occupancy-blind escape hatch
    try std.testing.expect(qmoeFillAcceptable(1, 128));

    state.qmoe_min_fill_pct = 101; // >100 = grouped GPU path never engages
    try std.testing.expect(!qmoeFillAcceptable(4096, 128));
}

test "metal gemm f32 batched matches per-matrix calls" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(5);
    const random = prng.random();
    const m = 48;
    const n = 40;
    const k = 32;
    const batch = 3;

    const a = try allocator.alloc(f32, batch * m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, batch * k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, batch * m * n);
    defer allocator.free(c);
    const expected = try allocator.alloc(f32, batch * m * n);
    defer allocator.free(expected);
    for (a) |*x| x.* = random.floatNorm(f32);
    for (b) |*x| x.* = random.floatNorm(f32);
    @memset(c, 0);

    try std.testing.expect(gemmBatchedF32(.nn, a, b, c, m, n, k, batch, m * k, k * n, m * n));
    for (0..batch) |bi| {
        cpuReference(.nn, a[bi * m * k ..][0 .. m * k], b[bi * k * n ..][0 .. k * n], expected[bi * m * n ..][0 .. m * n], m, n, k);
    }
    for (c, expected) |got, want| {
        const tol = @max(2e-5 * @max(@abs(want), @abs(got)), 2e-5);
        try std.testing.expect(@abs(got - want) <= tol);
    }
}

test "metal eager async gemm chains on the queue and synchronizes on host read" {
    if (!enabled) return error.SkipZigTest;
    if (context() == null) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const m = 65;
    const n = 67;
    const k = 33;

    const av = try allocator.alloc(f32, m * k);
    defer allocator.free(av);
    const bv = try allocator.alloc(f32, n * k);
    defer allocator.free(bv);
    for (av, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.03125 - 0.25;
    for (bv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.015625 - 0.125;

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, av);
    defer a.deinit();
    var b = try Tensor.fromSlice(allocator, &.{ n, k }, bv);
    defer b.deinit();
    var first = try Tensor.zeros(allocator, &.{ m, n });
    defer first.deinit();
    var second = try Tensor.zeros(allocator, &.{ m, k });
    defer second.deinit();

    // Both calls submit immediately.  `first` is consumed by the second
    // command directly from shared memory; no host wait occurs between them.
    try std.testing.expect(gemmF32Async(.nt, &a, &b, &first, m, n, k));
    try std.testing.expect(first.buffer.pending() != null);
    try std.testing.expect(gemmF32Async(.nn, &first, &b, &second, m, k, n));
    try std.testing.expect(second.buffer.pending() != null);

    const got = second.dataConst(); // the first unavoidable host boundary
    try std.testing.expect(second.buffer.pending() == null);

    const tmp = try allocator.alloc(f64, m * n);
    defer allocator.free(tmp);
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f64 = 0;
            for (0..k) |p| sum += @as(f64, av[row * k + p]) * @as(f64, bv[col * k + p]);
            tmp[row * n + col] = sum;
        }
    }
    for (0..m) |row| {
        for (0..k) |col| {
            var sum: f64 = 0;
            for (0..n) |p| sum += tmp[row * n + p] * @as(f64, bv[p * k + col]);
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(sum)), got[row * k + col], 3e-4);
        }
    }
}

test "metal eager async input mutation waits for the device reader" {
    if (!enabled) return error.SkipZigTest;
    if (context() == null) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const m = 33;
    const n = 35;
    const k = 31;
    const av = try allocator.alloc(f32, m * k);
    defer allocator.free(av);
    const bv = try allocator.alloc(f32, k * n);
    defer allocator.free(bv);
    const expected = try allocator.alloc(f32, m * n);
    defer allocator.free(expected);
    for (av, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.03125 - 0.125;
    for (bv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.015625 - 0.0625;
    cpuReference(.nn, av, bv, expected, m, n, k);

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, av);
    defer a.deinit();
    var b = try Tensor.fromSlice(allocator, &.{ k, n }, bv);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();
    try std.testing.expect(gemmF32Async(.nn, &a, &b, &out, m, n, k));
    try std.testing.expect(a.buffer.pending_use.load(.acquire) != null);
    a.data()[0] += 100; // mutable host boundary must wait for the old value's reader
    try std.testing.expect(a.buffer.pending_use.load(.acquire) == null);
    for (out.dataConst(), expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 2e-4);
}

test "metal eager async f16 NT writes f32 directly and fences input mutation" {
    if (!enabled) return error.SkipZigTest;
    if (context() == null) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const m = 33;
    const n = 47;
    const k = 17;
    const av = try allocator.alloc(f16, m * k);
    defer allocator.free(av);
    const bv = try allocator.alloc(f16, n * k);
    defer allocator.free(bv);
    const expected = try allocator.alloc(f32, m * n);
    defer allocator.free(expected);
    for (av, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i % 17)) * 0.03125 - 0.25);
    for (bv, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i % 13)) * 0.015625 - 0.125);
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k) |p| sum += @as(f32, av[row * k + p]) * @as(f32, bv[col * k + p]);
            expected[row * n + col] = sum;
        }
    }

    var a = try TensorF16.fromSlice(allocator, &.{ m, k }, av);
    defer a.deinit();
    var b = try TensorF16.fromSlice(allocator, &.{ n, k }, bv);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();
    try std.testing.expect(gemmF16NtAsync(&a, &b, &out, m, n, k));
    try std.testing.expect(out.buffer.pending() != null);
    try std.testing.expect(a.buffer.pending_use.load(.acquire) != null);
    a.data()[0] += 10;
    try std.testing.expect(a.buffer.pending_use.load(.acquire) == null);
    for (out.dataConst(), expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 4e-4);
    try std.testing.expect(out.buffer.pending() == null);
}

test "metal eager async bf16 NT matches the CPU bf16 reference" {
    if (!enabled) return error.SkipZigTest;
    if (context() == null) return error.SkipZigTest;
    const dtype_mod = @import("../dtype.zig");
    const allocator = std.testing.allocator;
    // Unaligned on every tile edge (bm 32 / bn 32 / bk 16) to run the
    // partial-tile arms too.
    const m = 33;
    const n = 47;
    const k = 17;
    const av = try allocator.alloc(f32, m * k);
    defer allocator.free(av);
    const bv = try allocator.alloc(u16, n * k);
    defer allocator.free(bv);
    const expected = try allocator.alloc(f32, m * n);
    defer allocator.free(expected);
    for (av, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.03125 - 0.25;
    for (bv, 0..) |*v, i| v.* = dtype_mod.f32ToBf16(@as(f32, @floatFromInt(i % 13)) * 0.015625 - 0.125);
    // Reference: the kernel's exact operand semantics — A rounded to bf16,
    // B read as bf16, f32 accumulation.
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k) |p| {
                const a_b = dtype_mod.bf16ToF32(dtype_mod.f32ToBf16(av[row * k + p]));
                sum += a_b * dtype_mod.bf16ToF32(bv[col * k + p]);
            }
            expected[row * n + col] = sum;
        }
    }

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, av);
    defer a.deinit();
    var b = try TensorBf16.fromSlice(allocator, &.{ n, k }, bv);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();
    try std.testing.expect(gemmBf16NtAsync(&a, &b, &out, m, n, k));
    try std.testing.expect(out.buffer.pending() != null);
    try std.testing.expect(b.buffer.pending_use.load(.acquire) != null);
    for (out.dataConst(), expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 4e-4);
    try std.testing.expect(out.buffer.pending() == null);
    try std.testing.expect(b.buffer.pending_use.load(.acquire) == null);
}
