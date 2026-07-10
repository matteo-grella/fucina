//! Metal GPU GEMM provider (`-Dgpu=metal`) — Zig host side.
//!
//! The objective is training-shaped f32 GEMM offload: run the heavy matrix
//! multiplications on the GPU, keep everything else on CPU. The kernel is the
//! vendored MLX "steel" GEMM (`metal/mlx_gemm.metal`, MIT, Apple copyright) —
//! simdgroup-matrix 8x8, 32x32x16 tiles, alignment-specialized edge handling —
//! compiled once at lazy init from embedded source by the ObjC shim
//! (`metal/shim.m`).
//!
//! Contract with the dispatch sites in native.zig (mirrors the BLAS arm):
//! `gemmF32`/`gemmBatchedF32` return `false` when the GPU did not run (init
//! failure, kill switch, shim error) and the caller falls through to
//! BLAS/vector — correctness never depends on the GPU. Calls are synchronous
//! (commit + waitUntilCompleted on the caller thread, like cblas_sgemm); a
//! batched call is ONE dispatch with grid depth = batch.
//!
//! Quantized dense/MoE offload uses one process-global Metal context, one
//! process-global staging panel pair, and a process-global lock. That is an
//! explicit eager-runtime choice: dispatches are blocking and serialized like a
//! BLAS call, while stable RHS operands may opt into resident storage owned by
//! the model/session.
//! The public Tensor API has no device/location state.
//!
//! Heuristics: `shouldUseGpu` gates on m*n*k work (default 2^30 ≈ 1024³, the
//! measured M1 Max crossover vs Accelerate/AMX — see `default_min_work`) with
//! `FUCINA_GPU_MIN_WORK` to experiment and `FUCINA_GPU=0` as a runtime kill
//! switch. Dense Q6_K dequant-in-kernel GEMM has a lower default gate, tuned on
//! Parakeet 110M, and can be overridden with `FUCINA_GPU_MIN_WORK_DENSE_Q6`.
//! Tune f32/f16 via `zig build bench-gemm -Dgpu=metal`.
const std = @import("std");
const build_options = @import("build_options");
const thread = @import("../thread.zig");

pub const enabled = build_options.gpu_kind == .metal;

/// Provider capability: dequant-in-kernel quantized GEMM (dense + grouped
/// MoE) is implemented. Loaders that reshape CPU-side representations for the
/// GPU quant path (gemma4's single-raw-expert choice, borrow arm) key on this
/// rather than `enabled` — providers whose quant arms are still stubs keep
/// the plain CPU story. Must be `enabled and ...`: a false `enabled` keeps
/// the whole module comptime-dead on other builds (extern symbols, libc).
pub const has_quant_gemm = enabled;

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

// One library: the MLX steel f32/f16 GEMM plus the vendored ggml quantized
// mul_mm (dequant-in-kernel). Both files are self-contained MSL; metal_stdlib
// include-guards make the concatenation safe.
const msl_source = @embedFile("metal/mlx_gemm.metal") ++ "\n" ++ @embedFile("metal/ggml_mul_mm.metal");

const State = struct {
    ctx: ?*anyopaque = null,
    gpu_enabled: bool = true,
    min_work: u64 = default_min_work,
    min_work_f16: u64 = default_min_work_f16,
    min_work_qmoe: u64 = default_min_work_qmoe,
    min_work_dense_q6: u64 = default_min_work_dense_q6,
    qmoe_min_fill_pct: u64 = default_qmoe_min_fill_pct,
};

/// Default offload threshold, set from `bench-gemm -Dgpu=metal` on M1 Max
/// (2026-06-12): the GPU crosses Accelerate/AMX at ~1024³ m*n*k work
/// (GPU 1361 vs 1302 GF/s there; 2048x1024x1024 "train" shape 2820 vs 1539,
/// +83%; 2048³ 3859 vs 3175). Below it AMX wins decisively (768³: 2522 vs
/// 1017) and the ~0.1-0.4 ms fixed round trip dominates small shapes.
const default_min_work: u64 = 1 << 30;
/// The f16-operands NT entry competes with the CPU f16 row kernels (no AMX
/// arm — Accelerate has no f16 GEMM here), which run an order of magnitude
/// below AMX f32, so the GPU pays off much earlier. 2^27 ≈ 134M m*n*k keeps
/// the ~0.3-0.5 ms round trip + output widen safely amortized.
const default_min_work_f16: u64 = 1 << 27;
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
    f16_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_gpu_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_sched_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_gpu_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_sched_ns: std.atomic.Value(u64) = .{ .raw = 0 },
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
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_QMOE")) |v_ptr| {
        const parsed = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_qmoe;
        state.min_work_qmoe = parsed;
        state.min_work_dense_q6 = parsed;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_DENSE_Q6")) |v_ptr| {
        state.min_work_dense_q6 = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_dense_q6;
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

    /// K (the reduced dim) must be a whole number of blocks.
    pub fn kMultiple(self: QFormat) usize {
        return switch (self) {
            .q8_0 => 32,
            .q6_k => 256,
            .q4_k => 256,
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
        .q4_k, .q8_0 => state.min_work_qmoe,
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
    inline for (.{ QFormat.q6_k, QFormat.q4_k, QFormat.q8_0 }) |fmt| {
        const Block = switch (fmt) {
            .q6_k => dtype_mod.BlockQ6_K,
            .q4_k => dtype_mod.BlockQ4_K,
            .q8_0 => dtype_mod.BlockQ8_0,
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

    inline for (.{ QFormat.q6_k, QFormat.q4_k, QFormat.q8_0 }) |fmt| {
        const Block = switch (fmt) {
            .q6_k => dtype_mod.BlockQ6_K,
            .q4_k => dtype_mod.BlockQ4_K,
            .q8_0 => dtype_mod.BlockQ8_0,
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
