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
//! M1 Max async crossover vs Accelerate/AMX — see `tuning.Table.Gpu`) with
//! `FUCINA_GPU_MIN_WORK` to experiment and `FUCINA_GPU=0` as a runtime kill
//! switch. Compact/raw Q6_K retains its Parakeet-tuned gate; dense model
//! weights use per-format gates measured against their faster packed CPU
//! fallback. Tune with `bench-gpu-dispatch` and `bench-gpu-formats`.
const std = @import("std");
const accelerator = @import("../accelerator.zig");
const build_options = @import("build_options");
const dtype_mod = @import("../dtype.zig");
const storage = @import("../storage.zig");
const gpu_policy = @import("gpu_policy.zig");
const gpu_provider = @import("gpu_provider.zig");
const Fmt = gpu_provider.QuantFormat;
const gpu_trace = @import("gpu_trace.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");
const tuning = @import("../tuning.zig");

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

pub const Orient = gpu_provider.Orient;

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
const msl_source = @embedFile("metal/mlx_gemm.metal") ++ "\n" ++ @embedFile("metal/ggml_mul_mm.metal") ++ "\n" ++ @embedFile("metal/attention.metal");

/// Offload policy is read live from the process tuning table
/// (`tuning.get().gpu` — loaded once from the environment and cached; the
/// `*ForTest` seams pin leaves in place). Only two things latch at the
/// one-time configuration read: the derived Q6 floors (their seeding rule
/// consults `tuning.wasSet`, so later test pins must not re-derive them)
/// and the trace flag. Crossover provenance lives on the table field docs
/// (`src/tuning.zig`).
var q6_floors: tuning.GpuQ6Floors = .{ .dense_q6 = 0, .packed_q6 = 0 };
var device_ctx: ?*anyopaque = null;
var config_done = std.atomic.Value(bool).init(false);
var init_done = std.atomic.Value(bool).init(false);
var init_mutex: thread.Mutex = .{};

/// The live tuning table's GPU group.
inline fn gpuT() *const tuning.Table.Gpu {
    return &tuning.get().gpu;
}

// --- Optional dispatch tracing (FUCINA_GPU_TRACE=1) -------------------------
// The counter table and timing helpers are the shared shell (`gpu_trace.zig`):
// zero overhead when off (`trace.on`, latched once at config), atomic
// counters (gates run on worker threads; dispatches are lock-serialized).
// The synchronous-dispatch wall-time per kind is the combined GPU-compute +
// waitUntilCompleted envelope (we commit+wait inline); qmoe lock-wait and
// stage-copy are split out. When tracing is on, the shim also returns
// command-buffer GPU/kernel timestamps so traceDump can show wall-vs-GPU
// overhead and the dominant shape buckets.
var trace: gpu_trace.Table = .{};

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

inline fn traceRhsCache(flag: bool) void {
    trace.inc(if (flag) .rhs_cacheable else .rhs_transient, 1);
}
inline fn overheadNs(wall_ns: u64, gpu_ns: u64) u64 {
    return if (wall_ns > gpu_ns) wall_ns - gpu_ns else 0;
}
fn traceRecordShape(kind: TraceKind, m: usize, n: usize, k: usize, batch: usize, tiles: usize, wall_ns: u64, timing: CommandTiming) void {
    if (!trace.on) return;
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
    trace.inc(.shape_overflow, 1);
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
    return trace.on;
}
/// Reset counters (call before a warm measurement window). Single-threaded.
pub fn traceReset() void {
    if (!trace.on) return;
    trace.reset();
    traceResetShapes();
}
/// Print the accumulated breakdown to stderr (no-op when tracing is off).
pub fn traceDump() void {
    if (!trace.on) return;
    const ms = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.f;
    const f32_wall = trace.get(.f32_ns);
    const f16_wall = trace.get(.f16_ns);
    const quant_wall = trace.get(.quant_ns);
    const attn_wall = trace.get(.attn_ns);
    const f32_gpu = trace.get(.f32_gpu_ns);
    const f16_gpu = trace.get(.f16_gpu_ns);
    const quant_gpu = trace.get(.quant_gpu_ns);
    const attn_gpu = trace.get(.attn_gpu_ns);
    std.debug.print(
        "[gpu-trace] async: f32 calls={d} submit={d:.1}ms host-wait={d:.1}ms | f16 calls={d} submit={d:.1}ms host-wait={d:.1}ms | bf16 calls={d} submit={d:.1}ms host-wait={d:.1}ms gpu={d:.1}ms | quant calls={d} submit={d:.1}ms host-wait={d:.1}ms\n",
        .{
            trace.get(.f32_async_calls),   ms(trace.get(.f32_submit_ns)),
            ms(trace.get(.f32_wait_ns)),   trace.get(.f16_async_calls),
            ms(trace.get(.f16_submit_ns)), ms(trace.get(.f16_wait_ns)),
            trace.get(.bf16_async_calls),  ms(trace.get(.bf16_submit_ns)),
            ms(trace.get(.bf16_wait_ns)),  ms(trace.get(.bf16_gpu_ns)),
            trace.get(.quant_async_calls), ms(trace.get(.quant_submit_ns)),
            ms(trace.get(.quant_wait_ns)),
        },
    );
    std.debug.print(
        \\[gpu-trace] dispatch: f32={d} ({d:.1}ms) f16={d} ({d:.1}ms) quant={d} ({d:.1}ms) attn={d} ({d:.1}ms)
        \\[gpu-trace] gpu-time: f32={d:.1}ms overhead={d:.1}ms | f16={d:.1}ms overhead={d:.1}ms | quant={d:.1}ms overhead={d:.1}ms | attn={d:.1}ms overhead={d:.1}ms
        \\[gpu-trace] kernel-sched: f32={d:.1}ms f16={d:.1}ms quant={d:.1}ms attn={d:.1}ms
        \\[gpu-trace] quant overhead: lock-wait={d:.1}ms stage-copy={d:.1}ms
        \\[gpu-trace] rhs-cache: stable={d} transient={d} | resident-bytes allocs={d} ({d:.1} MB) refused={d}
        \\[gpu-trace] gate decisions: pass={d} below-gate={d} shape-reject={d} shim-error={d} shape-overflow={d}
        \\
    , .{
        trace.get(.f32_calls),          ms(f32_wall),
        trace.get(.f16_calls),          ms(f16_wall),
        trace.get(.quant_calls),        ms(quant_wall),
        trace.get(.attn_calls),         ms(attn_wall),
        ms(f32_gpu),                    ms(overheadNs(f32_wall, f32_gpu)),
        ms(f16_gpu),                    ms(overheadNs(f16_wall, f16_gpu)),
        ms(quant_gpu),                  ms(overheadNs(quant_wall, quant_gpu)),
        ms(attn_gpu),                   ms(overheadNs(attn_wall, attn_gpu)),
        ms(trace.get(.f32_sched_ns)),   ms(trace.get(.f16_sched_ns)),
        ms(trace.get(.quant_sched_ns)), ms(trace.get(.attn_sched_ns)),
        ms(trace.get(.quant_lock_ns)),  ms(trace.get(.quant_stage_ns)),
        trace.get(.rhs_cacheable),      trace.get(.rhs_transient),
        trace.get(.dev_alloc_calls),    @as(f64, @floatFromInt(trace.get(.dev_alloc_bytes))) / 1e6,
        trace.get(.resident_refusals),  trace.get(.gate_pass),
        trace.get(.gate_below),         trace.get(.gate_shape),
        trace.get(.shim_err),           trace.get(.shape_overflow),
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
    // The dense-Q6/packed-Q6/qmoe seeding rule lives in the tuning table
    // (`tuning.gpuQ6Seeding`), stated once for both providers.
    q6_floors = tuning.gpuQ6Seeding();
    trace.on = gpuT().trace;
}

/// One-time lazy device init (std.once is gone in Zig 0.16): double-checked
/// under a mutex so concurrent ExecContexts share one device/library.
fn ensureInit() void {
    ensureConfig();
    if (!gpuT().enabled) return;
    if (init_done.load(.acquire)) return;
    init_mutex.lock();
    defer init_mutex.unlock();
    if (!init_done.load(.monotonic)) {
        initOnce();
        init_done.store(true, .release);
    }
}

fn initOnce() void {
    device_ctx = fucina_metal_init(msl_source);
    if (device_ctx == null) {
        std.log.warn("fucina-metal: init failed; GPU GEMM disabled for this process", .{});
    }
}

/// Lazy device init; null = GPU unavailable/disabled.
fn context() ?*anyopaque {
    ensureInit();
    return device_ctx;
}

pub fn deviceName() ?[]const u8 {
    const ctx = context() orelse return null;
    return std.mem.span(fucina_metal_device_name(ctx));
}

/// Maps a shared work-gate decision onto the trace counters.
fn gateDecision(d: gpu_policy.Decision) bool {
    if (d == .shape) {
        trace.inc(.gate_shape, 1);
        return false;
    }
    const pass = d == .pass;
    trace.gate(pass);
    return pass;
}

pub fn shouldUseGpu(m: usize, n: usize, k: usize) bool {
    ensureConfig();
    return gateDecision(gpu_policy.f32Gate(gpuT(), m, n, k, gpu_policy.work(m, n, k)));
}

pub fn shouldUseGpuBatched(m: usize, n: usize, k: usize, batch_count: usize) bool {
    ensureConfig();
    return gateDecision(gpu_policy.f32Gate(gpuT(), m, n, k, gpu_policy.workBatched(m, n, k, batch_count)));
}

pub fn shouldUseGpuF16(m: usize, n: usize, k: usize) bool {
    ensureConfig();
    return gateDecision(gpu_policy.f16Gate(gpuT(), m, n, k));
}

/// Resident small-m admission (the CUDA gate's idea, wrap-based): a decode
/// or batched-decode GEMM over weights that already carry a Metal page wrap
/// reads the whole RHS once, so bandwidth — not the m>=32 tile shape —
/// decides. Legal only with an existing wrap: first touch happens on a
/// large prefill, never on the latency-sensitive decode path.
fn residentSmallM16bit(buffer: anytype, m: usize, n: usize, k: usize) bool {
    const t = gpuT();
    if (m == 0 or m >= 32 or n < 256 or k < 256) return false;
    if (!t.enabled) return false;
    if (gpu_policy.work(m, n, k) < t.min_work.@"16bit_resident") return false;
    const resident = buffer.acceleratorResource(.metal) != null;
    trace.gate(resident);
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
    if (!gpu_policy.gemvEligible(gpuT(), m, n, k)) return false;
    const bytes = std.mem.sliceAsBytes(b.buffer.data);
    const resident = isResidentRange(bytes) or b.buffer.acceleratorResource(.metal) != null;
    trace.gate(resident);
    return resident;
}

/// Tensor-aware native-dispatch gate. Metal's ordinary large-op gate needs
/// no residency distinction (host and device share memory); the separate arm
/// only admits the small resident GEMV shape rejected by that gate.
pub fn shouldUseGpuForRhs(b: *const Tensor, m: usize, n: usize, k: usize) bool {
    return shouldUseGpuGemv(b, m, n, k) or shouldUseGpu(m, n, k);
}

/// Provider capability flag for the grouped-causal-attention forward arm
/// (exec/attention.zig's GPU tier). CUDA implements the same contract over
/// its streaming attention kernel.
pub const has_attention_fwd = true;

/// Attention-forward gate: prefill-length rows over the work floor
/// (`q_seq*kv_seq*heads*d >= min_work_attn`), plus the kernel's structural
/// limits — the score span must fit the 32 KB threadgroup budget and `d`
/// stays within the CPU tiled kernel's own ceiling.
pub fn shouldUseGpuAttentionFwd(q_seq: usize, kv_seq: usize, heads: usize, d: usize) bool {
    ensureConfig();
    if (!gpuT().enabled) return false;
    if (kv_seq > 7680 or d > 256 or d == 0) return false;
    const pass = gpu_policy.attnWork(gpuT(), q_seq, kv_seq, heads, d);
    trace.gate(pass);
    return pass;
}

/// Grouped causal attention forward on the GPU, f32, blocking — the exact
/// CPU-kernel contract: rows `[q_seq, heads, d]` against `[kv_seq,
/// kv_heads, d]` K/V with the uniform `heads_per_kv` GQA mapping, analytic
/// causal/window bounds, optional `{row max, sum exp}` stats capture for
/// the training backward. Values differ from the CPU kernels in summation
/// order only (the tier-shared ~1e-6 relative class). Returns false when
/// the GPU did not run (caller falls through to the CPU tiers).
pub fn attentionFwdF32(
    q_data: []const f32,
    k_data: []const f32,
    v_data: []const f32,
    out_data: []f32,
    stats: ?[]f32,
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    window: usize,
    causal: bool,
    heads_per_kv: usize,
    scale_value: f32,
) bool {
    return attentionFwdImpl(f32, q_data, k_data, v_data, out_data, stats, q_seq, kv_seq, heads, kv_heads, d, window, causal, heads_per_kv, scale_value);
}

/// The f16-KV-cache instantiation of the same kernel (inference prefill:
/// f32 queries against the half K/V cache, widened per load, f32
/// accumulation — the CPU f16-KV tier's contract).
pub fn attentionFwdF16Kv(
    q_data: []const f32,
    k_data: []const f16,
    v_data: []const f16,
    out_data: []f32,
    stats: ?[]f32,
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    window: usize,
    causal: bool,
    heads_per_kv: usize,
    scale_value: f32,
) bool {
    return attentionFwdImpl(f16, q_data, k_data, v_data, out_data, stats, q_seq, kv_seq, heads, kv_heads, d, window, causal, heads_per_kv, scale_value);
}

fn attentionFwdImpl(
    comptime KvElem: type,
    q_data: []const f32,
    k_data: []const KvElem,
    v_data: []const KvElem,
    out_data: []f32,
    stats: ?[]f32,
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    window: usize,
    causal: bool,
    heads_per_kv: usize,
    scale_value: f32,
) bool {
    const ctx = context() orelse return false;
    if (q_seq == 0 or kv_seq == 0 or heads == 0 or d == 0 or heads_per_kv == 0) return false;
    if (q_seq > std.math.maxInt(i32) or kv_seq > std.math.maxInt(i32) or d > std.math.maxInt(i32)) return false;
    const q_elems = q_seq * heads * d;
    const kv_elems = kv_seq * kv_heads * d;
    if (q_data.len < q_elems or k_data.len < kv_elems or v_data.len < kv_elems or out_data.len < q_elems) return false;
    if (stats) |values| {
        if (values.len < heads * q_seq * 2) return false;
    }
    var timing: CommandTiming = .{ .gpu_ns = 0, .sched_ns = 0 };
    const timer = trace.start();
    const rc = fucina_metal_attention_fwd_f32(
        ctx,
        q_data.ptr,
        @ptrCast(k_data.ptr),
        @ptrCast(v_data.ptr),
        out_data.ptr,
        if (stats) |values| values.ptr else out_data.ptr,
        @intCast(q_seq),
        @intCast(kv_seq),
        @intCast(heads),
        @intCast(kv_heads),
        @intCast(d),
        @intCast(kv_seq - q_seq),
        @intCast(window),
        @intFromBool(causal),
        @intCast(heads_per_kv),
        @intFromBool(stats != null),
        @intFromBool(KvElem == f16),
        scale_value,
        if (trace.on) &timing else null,
    );
    if (trace.on) {
        const wall_ns = trace.finish(timer);
        trace.inc(.attn_ns, wall_ns);
        trace.inc(.attn_gpu_ns, timing.gpu_ns);
        trace.inc(.attn_sched_ns, timing.sched_ns);
        trace.inc(.attn_calls, 1);
        if (rc != 0) trace.inc(.shim_err, 1);
    }
    return rc == 0;
}

extern fn fucina_metal_attention_fwd_f32(
    ctx: *anyopaque,
    q: [*]const f32,
    k: *const anyopaque,
    v: *const anyopaque,
    out: [*]f32,
    stats: [*]f32,
    q_seq: i64,
    kv_seq: i64,
    heads: i64,
    kv_heads: i64,
    d: i64,
    source_offset: i64,
    window: i64,
    causal: i32,
    heads_per_kv: i64,
    has_stats: i32,
    kv_half: i32,
    scale: f32,
    timing: ?*CommandTiming,
) c_int;

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
    const timer = trace.start();
    const rc = fucina_metal_gemm_f16_nt(ctx, a.ptr, b.ptr, @intCast(m), @intCast(n), @intCast(k), @intFromBool(cacheable), &staging, if (trace.on) &timing else null);
    if (trace.on) {
        const wall_ns = trace.finish(timer);
        trace.inc(.f16_ns, wall_ns);
        trace.inc(.f16_gpu_ns, timing.gpu_ns);
        trace.inc(.f16_sched_ns, timing.sched_ns);
        trace.inc(.f16_calls, 1);
        traceRhsCache(cacheable);
        traceRecordShape(.f16, m, n, k, 1, 0, wall_ns, timing);
        if (rc != 0) trace.inc(.shim_err, 1);
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
    const timer = trace.start();
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
        if (trace.on) &timing else null,
    );
    if (trace.on) {
        const wall_ns = trace.finish(timer);
        trace.inc(.f32_ns, wall_ns);
        trace.inc(.f32_gpu_ns, timing.gpu_ns);
        trace.inc(.f32_sched_ns, timing.sched_ns);
        trace.inc(.f32_calls, 1);
        traceRecordShape(.f32, m, n, k, batch_count, 0, wall_ns, timing);
        if (rc != 0) trace.inc(.shim_err, 1);
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
        const wait_started = trace.start();
        const rc = fucina_metal_ticket_wait(self.ticket, if (trace.on) &timing else null);
        if (trace.on) {
            trace.inc(.f32_wait_ns, trace.finish(wait_started));
            trace.inc(.f32_gpu_ns, timing.gpu_ns);
            trace.inc(.f32_sched_ns, timing.sched_ns);
            if (rc != 0) trace.inc(.shim_err, 1);
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
    const submit_started = trace.start();
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
    if (trace.on) {
        trace.inc(.f32_async_calls, 1);
        trace.inc(.f32_submit_ns, trace.finish(submit_started));
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
        const wait_started = trace.start();
        const rc = fucina_metal_ticket_wait(self.ticket, if (trace.on) &timing else null);
        if (trace.on) {
            trace.inc(.f16_wait_ns, trace.finish(wait_started));
            trace.inc(.f16_gpu_ns, timing.gpu_ns);
            trace.inc(.f16_sched_ns, timing.sched_ns);
            if (rc != 0) trace.inc(.shim_err, 1);
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
    const submit_started = trace.start();
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
    if (trace.on) {
        trace.inc(.f16_async_calls, 1);
        trace.inc(.f16_submit_ns, trace.finish(submit_started));
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
        const wait_started = trace.start();
        const rc = fucina_metal_ticket_wait(self.ticket, if (trace.on) &timing else null);
        if (trace.on) {
            trace.inc(.bf16_wait_ns, trace.finish(wait_started));
            trace.inc(.bf16_gpu_ns, timing.gpu_ns);
            trace.inc(.bf16_sched_ns, timing.sched_ns);
            if (rc != 0) trace.inc(.shim_err, 1);
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
    a.buffer.waitReady();
    const scratch = storage.BufferOf(.bf16).create(std.heap.c_allocator, a_elems) catch return false;
    var scratch_owned = true;
    defer if (scratch_owned) scratch.release();
    for (scratch.data, a.buffer.data[a.offset..][0..a_elems]) |*dst, src| {
        dst.* = dtype_mod.f32ToBf16(src);
    }

    const holder = std.heap.c_allocator.create(MetalBf16Work) catch return false;
    const submit_started = trace.start();
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
    if (trace.on) {
        trace.inc(.bf16_async_calls, 1);
        trace.inc(.bf16_submit_ns, trace.finish(submit_started));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Quantized (dequant-in-kernel) grouped GEMM — the MoE prefill path.
// Kernel: metal/ggml_mul_mm.metal (vendored llama.cpp legacy mul_mm).
// ---------------------------------------------------------------------------

/// Kernel-side ABI integers for the weight block formats the quantized
/// kernel reads directly. Must mirror the FUCINA_QFMT_* enum in shim.m.
/// Provider-private: the format identity is `gpu_provider.QuantFormat`
/// (tag names match, so `abiValue` maps by name; `tq2_0_folded` is the
/// fused PTQTP plane-pair layout with no `DType`).
const AbiTag = enum(c_int) {
    q8_0 = 0,
    q6_k = 1,
    q4_k = 2,
    tq2_0 = 3,
    tq2_0_folded = 4,
};

/// The kernel-side ABI integer for `fmt`, or null when this provider has
/// no kernel for it (null IS the capability answer: `offload.supportsQuant`
/// derives from it).
pub fn abiValue(comptime fmt: Fmt) ?c_int {
    if (!@hasField(AbiTag, @tagName(fmt))) return null;
    return @intFromEnum(@field(AbiTag, @tagName(fmt)));
}

/// Runtime form for the dispatch entries; unsupported formats are gated out
/// before dispatch, so reaching one here is a bug.
fn abi(fmt: Fmt) c_int {
    return switch (fmt) {
        inline else => |f| if (comptime abiValue(f)) |v| v else unreachable,
    };
}

/// One 32-row output tile of one expert group. Must mirror FucinaQMMTile in
/// shim.m / fucina_qmm_tile in the kernel.
pub const QMMTile = gpu_provider.QMMTile;

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
        const wait_started = trace.start();
        const rc = fucina_metal_ticket_wait(self.ticket, if (trace.on) &timing else null);
        if (trace.on) {
            trace.inc(.quant_wait_ns, trace.finish(wait_started));
            trace.inc(.quant_gpu_ns, timing.gpu_ns);
            trace.inc(.quant_sched_ns, timing.sched_ns);
            if (rc != 0) trace.inc(.shim_err, 1);
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
    format: Fmt,
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
    const submit_started = trace.start();
    const ticket = fucina_metal_gemm_q_dense_nt_async(
        ctx,
        abi(format),
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
    if (trace.on) {
        trace.inc(.quant_async_calls, 1);
        trace.inc(.quant_submit_ns, trace.finish(submit_started));
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
/// Flat-parameter device arm (see cuda.zig): not needed on unified memory —
/// the CPU kernels already mutate shared pages the GPU reads zero-copy — so
/// the stubs report "not handled" and callers keep their CPU path.
pub const FlatDType = gpu_provider.FlatDType;

pub fn flatPerturb(dt: FlatDType, bytes: []u8, stream_seed: u64, scaled: f32, n: usize) bool {
    _ = dt;
    _ = bytes;
    _ = stream_seed;
    _ = scaled;
    _ = n;
    return false;
}

pub fn flatWeightedUpdate(dt: FlatDType, bytes: []u8, stream_seeds: []const u64, coeffs: []const f32, scale: f32, n: usize) bool {
    _ = dt;
    _ = bytes;
    _ = stream_seeds;
    _ = coeffs;
    _ = scale;
    _ = n;
    return false;
}

pub fn flatAnchorDecay(dt: FlatDType, bytes: []u8, anchor: []const u8, decay_step: f32, is_l1: bool, n: usize) bool {
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
    if (!registerResidentRange(@intFromPtr(p), len)) {
        // Refuse (the CUDA behaviour) rather than hand out bytes the
        // dispatch paths would not recognize and would re-stage as
        // transient on every call: the caller keeps its host copy.
        _ = fucina_metal_free_resident_bytes(ctx, p);
        if (trace.on) trace.inc(.resident_refusals, 1);
        return null;
    }
    if (trace.on) {
        trace.inc(.dev_alloc_calls, 1);
        trace.inc(.dev_alloc_bytes, len);
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
///
/// The registry grows with the model (one entry per resident allocation,
/// under the lock): there is no fixed entry cap past which later weights
/// would silently stop being recognized. Only an allocator failure while
/// growing refuses the registration, and then `allocResidentBytes`
/// refuses the allocation itself (`Trace.resident_refusals`).
const ResidentRange = struct { base: usize, len: usize };
var resident_ranges_lock: thread.Mutex = .{};
var resident_ranges: std.ArrayList(ResidentRange) = .empty;

fn registerResidentRange(base: usize, len: usize) bool {
    resident_ranges_lock.lock();
    defer resident_ranges_lock.unlock();
    resident_ranges.append(std.heap.c_allocator, .{ .base = base, .len = len }) catch return false;
    return true;
}

fn unregisterResidentRange(base: usize) void {
    resident_ranges_lock.lock();
    defer resident_ranges_lock.unlock();
    for (resident_ranges.items, 0..) |range, i| {
        if (range.base == base) {
            _ = resident_ranges.swapRemove(i);
            return;
        }
    }
}

/// Whether `bytes` lies fully inside a live resident allocation.
fn isResidentRange(bytes: []const u8) bool {
    const base = @intFromPtr(bytes.ptr);
    resident_ranges_lock.lock();
    defer resident_ranges_lock.unlock();
    for (resident_ranges.items) |range| {
        if (base >= range.base and base + bytes.len <= range.base + range.len) return true;
    }
    return false;
}

fn residentRangeCount() usize {
    resident_ranges_lock.lock();
    defer resident_ranges_lock.unlock();
    return resident_ranges.items.len;
}

test "metal resident-range registry grows past the retired 512-entry cap" {
    if (comptime !enabled) return error.SkipZigTest;
    if (context() == null) return error.SkipZigTest;

    const count = 600;
    const before = residentRangeCount();
    var ranges: [count][]u8 = undefined;
    var allocated: usize = 0;
    defer for (ranges[0..allocated]) |bytes| freeResidentBytes(bytes);
    while (allocated < count) : (allocated += 1) {
        ranges[allocated] = allocResidentBytes(4096) orelse return error.TestUnexpectedResult;
    }
    try std.testing.expectEqual(before + count, residentRangeCount());
    // Every allocation, first and last alike, is recognized whole and by
    // an interior sub-range; the old table stopped recording at 512.
    for (ranges) |bytes| {
        try std.testing.expect(isResidentRange(bytes));
        try std.testing.expect(isResidentRange(bytes[1024..2048]));
    }
    for (ranges[0..allocated]) |bytes| freeResidentBytes(bytes);
    allocated = 0;
    try std.testing.expectEqual(before, residentRangeCount());
    try std.testing.expect(!isResidentRange(ranges[count - 1]));
}

/// Release bytes returned by `allocResidentBytes`. Safe no-op when the Metal
/// context is gone or the slice did not come from the resident allocator.
pub fn freeResidentBytes(bytes: []const u8) void {
    if (bytes.len == 0) return;
    unregisterResidentRange(@intFromPtr(bytes.ptr));
    const ctx = device_ctx orelse return;
    _ = fucina_metal_free_resident_bytes(ctx, bytes.ptr);
}

/// Process-global serialization for quantized-GEMM staging panels. Hold across
/// `qmoeStage` + CPU panel writes + `gemmQGroupedNt` dispatches + readback:
/// the shim owns one grow-only in/out panel pair and the command is blocking.
/// This deliberately matches Fucina's eager BLAS-like contract; any future
/// concurrent/async GPU runtime should replace this whole staging contract.
pub var qmoe_lock: thread.Mutex = .{};

pub const QMoeStage = gpu_provider.QMoeStage;

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
    format: Fmt,
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
    const timer = trace.start();
    const rc = fucina_metal_gemm_q_grouped_nt(
        ctx,
        abi(format),
        rhs_bytes.ptr,
        @intCast(rhs_bytes.len),
        @intFromBool(rhs_cacheable),
        @intCast(nb01),
        @intCast(nb02),
        @intCast(n_out),
        @intCast(k),
        tiles.ptr,
        @intCast(tiles.len),
        if (trace.on) &timing else null,
    );
    if (trace.on) {
        const wall_ns = trace.finish(timer);
        trace.inc(.quant_ns, wall_ns);
        trace.inc(.quant_gpu_ns, timing.gpu_ns);
        trace.inc(.quant_sched_ns, timing.sched_ns);
        trace.inc(.quant_calls, 1);
        traceRhsCache(rhs_cacheable);
        traceRecordShape(.quant, rowsCoveredByTiles(tiles), n_out, k, 1, tiles.len, wall_ns, timing);
        if (rc != 0) trace.inc(.shim_err, 1);
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
    format: Fmt,
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

    const lock_timer = trace.start();
    qmoe_lock.lock();
    defer qmoe_lock.unlock();
    trace.elapsed(.quant_lock_ns, lock_timer);
    const stage = qmoeStage(in_bytes, out_bytes) orelse return false;
    const in_timer = trace.start();
    @memcpy(stage.in[0..in_elems], a[0..in_elems]);
    trace.elapsed(.quant_stage_ns, in_timer);
    var tiles_buf: [64]QMMTile = undefined;
    const n_tiles = (m + 31) / 32;
    if (n_tiles > tiles_buf.len) return false;
    for (0..n_tiles) |t| {
        tiles_buf[t] = .{ .expert = 0, .base_row = 0, .m = @intCast(m), .tile_m = @intCast(t) };
    }
    if (!gemmQGroupedNt(format, rhs_bytes, rhs_cacheable, nb01, 0, n, k, tiles_buf[0..n_tiles])) return false;
    const out_timer = trace.start();
    @memcpy(c[0..out_elems], stage.out[0..out_elems]);
    trace.elapsed(.quant_stage_ns, out_timer);
    return true;
}

/// Batched dense quantized NT GEMM over one shared activation matrix:
/// for each batch `b`, `c[b,m,n] = a[m,k] · dequant(W[b,n,k])^T`.
/// This is the narrow eager command-batching seam: it uses the existing grouped
/// quant kernel's expert dimension to collapse same-shape independent linears
/// into one Metal command, while the caller still owns an ordinary CPU-visible
/// result tensor. `nb02` is the byte stride between consecutive RHS operands.
pub fn gemmQuantNtSharedABatch(
    format: Fmt,
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

    const lock_timer = trace.start();
    qmoe_lock.lock();
    defer qmoe_lock.unlock();
    trace.elapsed(.quant_lock_ns, lock_timer);

    const row_bytes = std.math.mul(usize, k, @sizeOf(f32)) catch return false;
    const in_bytes = std.math.mul(usize, rows_total, row_bytes) catch return false;
    const out_bytes = std.math.mul(usize, out_elems, @sizeOf(f32)) catch return false;
    const stage = qmoeStage(in_bytes, out_bytes) orelse return false;
    const in_timer = trace.start();
    for (0..batch_count) |bi| {
        @memcpy(stage.in[bi * in_elems ..][0..in_elems], a[0..in_elems]);
    }
    trace.elapsed(.quant_stage_ns, in_timer);

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
    const out_timer = trace.start();
    @memcpy(c[0..out_elems], stage.out[0..out_elems]);
    trace.elapsed(.quant_stage_ns, out_timer);
    return true;
}

pub fn shouldUseGpuQMoe(total_work: u64) bool {
    ensureConfig();
    const pass = gpu_policy.qmoeWork(gpuT(), total_work);
    trace.gate(pass);
    return pass;
}

/// Occupancy arm of the grouped-MoE gate: `rows` real panel rows spread over
/// `n_tiles` 32-row token tiles must reach the configured minimum fill
/// percentage (see `tuning.Table.Gpu.qmoe_min_fill`). Callers pass the exact tile
/// table they are about to dispatch.
pub fn qmoeFillAcceptable(rows: usize, n_tiles: usize) bool {
    ensureConfig();
    return gpu_policy.qmoeFillAcceptable(gpuT(), rows, n_tiles);
}

pub fn shouldUseGpuDenseQuant(format: Fmt, total_work: u64) bool {
    ensureConfig();
    std.debug.assert(format != .q5_k); // no Metal Q5_K kernel (supportsQuant gates first)
    const pass = gpu_policy.denseQuantWork(gpuT(), q6_floors, format, total_work);
    trace.gate(pass);
    return pass;
}

/// Dense model-weight gate against the load-time-packed CPU fallback.
pub fn shouldUseGpuDenseQuantPacked(format: Fmt, total_work: u64) bool {
    ensureConfig();
    std.debug.assert(format != .q5_k); // no Metal Q5_K kernel (supportsQuant gates first)
    const pass = gpu_policy.denseQuantPackedWork(gpuT(), q6_floors, format, total_work);
    trace.gate(pass);
    return pass;
}

/// Test seam: unit-test shapes never reach the real threshold.
pub fn setMinWorkQMoeForTest(v: u64) void {
    ensureConfig();
    tuning.setField("gpu.min_work.qmoe", v);
}

/// True when this provider is compiled in AND holds a live device context.
/// The GPU test suites gate on this rather than on `deviceName`, which can
/// also be null for a live context with no reportable name.
pub fn deviceAvailableForTest() bool {
    if (comptime !enabled) return false;
    return context() != null;
}

/// The grouped-MoE tile-occupancy gate, as a save/restore pair for tests.
pub fn qmoeMinFillForTest() u64 {
    ensureConfig();
    return gpuT().qmoe_min_fill;
}

pub fn setQmoeMinFillForTest(v: u64) void {
    ensureConfig();
    tuning.setField("gpu.qmoe_min_fill", v);
}

/// Quantized decode-GEMV gate — the decode arm of `offload.quantGemmAccepts`
/// (`src/exec/quant_matmul.zig`). The Metal provider keeps decode on CPU by
/// design — always false; the CUDA provider opts in via FUCINA_GPU_DECODE=1.
pub fn shouldUseGpuQuantDecode(format: Fmt, m: usize, n: usize, k: usize) bool {
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

const gpu_test_util = @import("gpu_test_util.zig");
const cpuReference = gpu_test_util.cpuReference;
const buildQuantWeights = gpu_test_util.buildQuantWeights;
const expectQuantGemmRows = gpu_test_util.expectQuantGemmRows;

test "metal attention forward parity vs scalar reference (causal, window, offset, stats)" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();

    const Case = struct { q_seq: usize, kv_seq: usize, heads: usize, kv_heads: usize, d: usize, window: usize, causal: bool };
    const cases = [_]Case{
        .{ .q_seq = 96, .kv_seq = 96, .heads = 4, .kv_heads = 2, .d = 64, .window = 0, .causal = true },
        .{ .q_seq = 64, .kv_seq = 96, .heads = 4, .kv_heads = 4, .d = 64, .window = 0, .causal = true }, // decode-suffix offset
        .{ .q_seq = 96, .kv_seq = 96, .heads = 2, .kv_heads = 1, .d = 80, .window = 40, .causal = true }, // window + d chunk tail
        .{ .q_seq = 48, .kv_seq = 48, .heads = 2, .kv_heads = 2, .d = 64, .window = 0, .causal = false }, // bidirectional
    };
    for (cases) |case| {
        const q = try allocator.alloc(f32, case.q_seq * case.heads * case.d);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, case.kv_seq * case.kv_heads * case.d);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, case.kv_seq * case.kv_heads * case.d);
        defer allocator.free(v);
        const out = try allocator.alloc(f32, q.len);
        defer allocator.free(out);
        const stats = try allocator.alloc(f32, case.heads * case.q_seq * 2);
        defer allocator.free(stats);
        for (q) |*x| x.* = random.floatNorm(f32);
        for (k) |*x| x.* = random.floatNorm(f32);
        for (v) |*x| x.* = random.floatNorm(f32);
        @memset(out, std.math.nan(f32));

        const scale: f32 = 0.125;
        const heads_per_kv = case.heads / case.kv_heads;
        try std.testing.expect(attentionFwdF32(q, k, v, out, stats, case.q_seq, case.kv_seq, case.heads, case.kv_heads, case.d, case.window, case.causal, heads_per_kv, scale));

        const source_offset = case.kv_seq - case.q_seq;
        const scores = try allocator.alloc(f64, case.kv_seq);
        defer allocator.free(scores);
        for (0..case.heads) |head_i| {
            const kv_base = (head_i / heads_per_kv) * case.d;
            for (0..case.q_seq) |q_i| {
                const p_abs = source_offset + q_i;
                const hi = if (case.causal) @min(p_abs + 1, case.kv_seq) else case.kv_seq;
                const lo = if (!case.causal or case.window == 0) 0 else (p_abs + 1) -| case.window;
                var m_row: f64 = -std.math.inf(f64);
                for (lo..hi) |j| {
                    var s: f64 = 0;
                    for (0..case.d) |c| {
                        s += @as(f64, q[(q_i * case.heads + head_i) * case.d + c]) * k[(j * case.kv_heads) * case.d + kv_base + c];
                    }
                    s *= scale;
                    scores[j] = s;
                    m_row = @max(m_row, s);
                }
                var sum: f64 = 0;
                for (lo..hi) |j| {
                    scores[j] = @exp(scores[j] - m_row);
                    sum += scores[j];
                }
                const sb = (head_i * case.q_seq + q_i) * 2;
                try std.testing.expectApproxEqRel(@as(f32, @floatCast(m_row)), stats[sb], 1e-4);
                try std.testing.expectApproxEqRel(@as(f32, @floatCast(sum)), stats[sb + 1], 1e-4);
                for (0..case.d) |c| {
                    var acc: f64 = 0;
                    for (lo..hi) |j| {
                        acc += scores[j] * v[(j * case.kv_heads) * case.d + kv_base + c];
                    }
                    const want: f32 = @floatCast(acc / sum);
                    const got = out[(q_i * case.heads + head_i) * case.d + c];
                    const tol = @max(2e-5 * @max(@abs(want), @abs(got)), 2e-5);
                    try std.testing.expect(@abs(got - want) <= tol);
                }
            }
        }
    }
}

test "metal attention forward f16-KV parity vs scalar reference" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();

    // Causal + GQA + decode-suffix offset, the inference prefill shape
    // class. The reference reads the same f16-rounded K/V the kernel sees.
    const q_seq = 64;
    const kv_seq = 96;
    const heads = 4;
    const kv_heads = 2;
    const d = 64;
    const heads_per_kv = heads / kv_heads;
    const scale: f32 = 0.125;

    const q = try allocator.alloc(f32, q_seq * heads * d);
    defer allocator.free(q);
    const k16 = try allocator.alloc(f16, kv_seq * kv_heads * d);
    defer allocator.free(k16);
    const v16 = try allocator.alloc(f16, kv_seq * kv_heads * d);
    defer allocator.free(v16);
    const out = try allocator.alloc(f32, q.len);
    defer allocator.free(out);
    const stats = try allocator.alloc(f32, heads * q_seq * 2);
    defer allocator.free(stats);
    for (q) |*x| x.* = random.floatNorm(f32);
    for (k16) |*x| x.* = @floatCast(random.floatNorm(f32));
    for (v16) |*x| x.* = @floatCast(random.floatNorm(f32));
    @memset(out, std.math.nan(f32));

    try std.testing.expect(attentionFwdF16Kv(q, k16, v16, out, stats, q_seq, kv_seq, heads, kv_heads, d, 0, true, heads_per_kv, scale));

    const source_offset = kv_seq - q_seq;
    const scores = try allocator.alloc(f64, kv_seq);
    defer allocator.free(scores);
    for (0..heads) |head_i| {
        const kv_base = (head_i / heads_per_kv) * d;
        for (0..q_seq) |q_i| {
            const hi = @min(source_offset + q_i + 1, kv_seq);
            var m_row: f64 = -std.math.inf(f64);
            for (0..hi) |j| {
                var sum_qk: f64 = 0;
                for (0..d) |c| {
                    sum_qk += @as(f64, q[(q_i * heads + head_i) * d + c]) * @as(f64, @floatCast(k16[(j * kv_heads) * d + kv_base + c]));
                }
                sum_qk *= scale;
                scores[j] = sum_qk;
                m_row = @max(m_row, sum_qk);
            }
            var sum: f64 = 0;
            for (0..hi) |j| {
                scores[j] = @exp(scores[j] - m_row);
                sum += scores[j];
            }
            const sb = (head_i * q_seq + q_i) * 2;
            try std.testing.expectApproxEqRel(@as(f32, @floatCast(m_row)), stats[sb], 1e-4);
            try std.testing.expectApproxEqRel(@as(f32, @floatCast(sum)), stats[sb + 1], 1e-4);
            for (0..d) |c| {
                var acc: f64 = 0;
                for (0..hi) |j| {
                    acc += scores[j] * @as(f64, @floatCast(v16[(j * kv_heads) * d + kv_base + c]));
                }
                const want: f32 = @floatCast(acc / sum);
                const got = out[(q_i * heads + head_i) * d + c];
                const tol = @max(2e-5 * @max(@abs(want), @abs(got)), 2e-5);
                try std.testing.expect(@abs(got - want) <= tol);
            }
        }
    }
}

test "metal quant gemm q6_K/q4_K/q8_0 parity vs dequantized reference" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(17);
    const random = prng.random();

    const Case = struct { m: usize, n: usize, k: usize };
    inline for (.{ Fmt.q6_k, Fmt.q4_k, Fmt.q8_0, Fmt.tq2_0 }) |fmt| {
        const Block = switch (fmt) {
            .q6_k => dtype_mod.BlockQ6_K,
            .q4_k => dtype_mod.BlockQ4_K,
            .q8_0 => dtype_mod.BlockQ8_0,
            .tq2_0 => dtype_mod.BlockTQ2_0,
            .tq2_0_folded => unreachable, // covered by its dedicated parity test
            .q5_k => unreachable, // no Metal Q5_K kernel
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

test "metal quant gemm tq2_0_folded parity vs dequantized reference" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xf01ded);
    const random = prng.random();

    const m = 33;
    const n = 8;
    const k = 512;
    const bpr = k / 256;
    const Block = @import("quant/types.zig").BlockTQ2_0Folded;

    // Random folded blocks: nibble codes in {0..8}, small positive f16 d.
    const blocks = try allocator.alloc(Block, n * bpr);
    defer allocator.free(blocks);
    const wref = try allocator.alloc(f32, n * k);
    defer allocator.free(wref);
    for (blocks, 0..) |*b, bi| {
        const d: f32 = 0.005 + random.float(f32) * 0.05;
        b.d = @bitCast(@as(f16, @floatCast(d)));
        const d_f32: f32 = @floatCast(@as(f16, @bitCast(b.d)));
        for (0..8) |sub| {
            for (0..16) |j| {
                const lo: u8 = random.intRangeAtMost(u8, 0, 8);
                const hi: u8 = random.intRangeAtMost(u8, 0, 8);
                b.qs[sub * 16 + j] = lo | (hi << 4);
                const row = bi / bpr;
                const base = (bi % bpr) * 256 + sub * 32;
                wref[row * k + base + j] = d_f32 * @as(f32, @floatFromInt(@as(i32, lo) - 4));
                wref[row * k + base + 16 + j] = d_f32 * @as(f32, @floatFromInt(@as(i32, hi) - 4));
            }
        }
    }

    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    for (a) |*v| v.* = (random.float(f32) * 2.0 - 1.0);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    try std.testing.expect(gemmQuantNt(
        .tq2_0_folded,
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

test "metal eager async dense quant Q4_K/Q6_K/Q8_0 uses direct tensor storage" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(31);
    const random = prng.random();
    inline for (.{ Fmt.q6_k, Fmt.q4_k, Fmt.q8_0, Fmt.tq2_0 }) |fmt| {
        const Block = switch (fmt) {
            .q6_k => dtype_mod.BlockQ6_K,
            .q4_k => dtype_mod.BlockQ4_K,
            .q8_0 => dtype_mod.BlockQ8_0,
            .tq2_0 => dtype_mod.BlockTQ2_0,
            .tq2_0_folded => unreachable, // covered by its dedicated parity test
            .q5_k => unreachable, // no Metal Q5_K kernel
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
    try std.testing.expect(gemmF32Async(.trans_b, &a, &b, &first, m, n, k));
    try std.testing.expect(first.buffer.pending() != null);
    try std.testing.expect(gemmF32Async(.plain, &first, &b, &second, m, k, n));
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

test "metal eager async bf16 NT matches the CPU bf16 reference" {
    if (!enabled) return error.SkipZigTest;
    if (context() == null) return error.SkipZigTest;
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
    try std.testing.expect(b.buffer.accel.pending_use.load(.acquire) != null);
    for (out.dataConst(), expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 4e-4);
    try std.testing.expect(out.buffer.pending() == null);
    try std.testing.expect(b.buffer.accel.pending_use.load(.acquire) == null);
}
