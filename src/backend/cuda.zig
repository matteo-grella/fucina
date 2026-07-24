//! CUDA GPU GEMM provider (`-Dgpu=cuda`) — Zig host side.
//!
//! Same eager provider contract as Metal: dense f32/f16 and stable-weight
//! Q4_K/Q5_K/Q6_K/Q8_0 commands submit to a
//! persistent stream and return with a completion token on output storage;
//! dependent GPU calls consume the producer device pointer and CPU access
//! performs the deferred visibility wait. Submission failure falls through to
//! BLAS/vector. Direct-slice, grouped-MoE phase boundaries, and attention
//! entries remain blocking. Submission is serialized briefly around provider
//! state and the shared cuBLAS handle.
//!
//! Host binding is dlopen (src/backend/cuda/api.zig): no CUDA SDK at build
//! time, so `-Dgpu=cuda -Dtarget=x86_64-linux-gnu` cross-compiles from any
//! dev machine. Missing libraries degrade per-capability (no libcublas ⇒ only
//! the f32/f16 GEMM arms are disabled).
//!
//! Offloaded ops:
//!   - f32 GEMM nn/tn/nt + strided-batched via cuBLAS, strict FP32 math
//!     (FUCINA_GPU_TF32=1 opts into TF32 tensor cores). Non-resident operands
//!     stream over PCIe, so a transient work floor (default 2^33 m·n·k and
//!     m ≥ 128, FUCINA_GPU_MIN_WORK_TRANSIENT) sits behind the ordinary gates.
//!   - f16 NT GEMM (cublasGemmEx, f32 accumulate + direct f32 async output).
//!   - Quantized dequant-in-kernel GEMM (Q4_K/Q5_K/Q6_K/Q8_0) via the vendored
//!     kernels (`cuda/kernels.cu` → committed `cuda/kernels.ptx`, driver JIT;
//!     NVRTC recompile fallback, FUCINA_GPU_KERNELS=src forces it). Adaptive
//!     N32/N64 WMMA tiles use tensor cores on capable devices; the original
//!     scalar-FFMA kernel remains the compatibility/diagnostic fallback. Both
//!     keep the Metal provider's tile-table and half-rounded operand contract;
//!     dense stable-weight calls use direct tensor storage and deferred D2H.
//!   - Grouped MoE expert FFN over pinned staging panels + device twins
//!     (`qmoeStage`/`gemmQGroupedNt`); the llm-tier phase chain (CPU
//!     gather/GeGLU/scatter between the two grouped dispatches) runs
//!     unchanged against the pinned panels, with transfers/kernel event-chained
//!     before each required CPU fence.
//!   - Decode GEMV (m <= 8, opt-in FUCINA_GPU_DECODE=1): warp-per-row
//!     dequant-dot against RESIDENT weights only; kept opt-in pending a
//!     parity-oracle pass on sampled-token streams.
//!   - Fused prefill attention over f16 KV (`attnPrefillF16`).
//!
//! Weight residency: `allocResidentBytes` = cuMemAllocManaged +
//! SET_READ_MOSTLY + prefetch-on-first-use — unified addressing means a
//! resident RHS dispatches with zero weight transfer while the CPU fallback
//! reads the same pointer at full host bandwidth. Stable (cacheable) RHS
//! bytes are additionally ADOPTED into the managed registry on first use —
//! the analog of the Metal shim's page wrap cache: mmap'd weights cross PCIe
//! once per process, not per dispatch. FUCINA_GPU_VRAM_BUDGET bounds both.
const std = @import("std");
const accelerator = @import("../accelerator.zig");
const build_options = @import("build_options");
const storage = @import("../storage.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");
const api = @import("cuda/api.zig");

const Tensor = tensor.Tensor;
const TensorF16 = tensor.TensorOf(.f16);

pub const enabled = build_options.gpu_kind == .cuda;

/// Provider capability: dequant-in-kernel quantized GEMM (dense + grouped
/// MoE) is implemented. Must stay `enabled and ...` so the module is
/// comptime-dead on non-cuda builds.
pub const has_quant_gemm = enabled;
/// CUDA's vendored dequant kernels cover Q5_K in addition to the common
/// Q4_K/Q6_K/Q8_0 provider surface. Metal deliberately leaves Q5_K on CPU.
pub const has_q5_k_quant = enabled;
/// No ternary CUDA kernel yet — the exec seam prunes the tq2_0 arm.
pub const has_tq2_0_quant = false;
pub const has_tq2_0_folded_quant = false;

pub const Orient = enum(c_int) { nn = 0, tn = 1, nt = 2 };

const State = struct {
    gpu_enabled: bool = true,
    tf32: bool = false,
    min_work: u64 = default_min_work,
    min_work_resident: u64 = default_min_work_resident,
    min_work_f16: u64 = default_min_work_f16,
    min_work_f16_resident: u64 = default_min_work_f16_resident,
    min_work_gemv: u64 = default_min_work_gemv,
    min_work_qmoe: u64 = default_min_work_qmoe,
    min_work_dense_q6: u64 = default_min_work_dense_q6,
    min_work_packed_q4: u64 = default_min_work_packed_q4,
    min_work_packed_q5: u64 = default_min_work_packed_q5,
    min_work_packed_q6: u64 = default_min_work_packed_q6,
    min_work_packed_q8: u64 = default_min_work_packed_q8,
    qmoe_min_fill_pct: u64 = default_qmoe_min_fill_pct,
    min_work_transient: u64 = default_min_work_transient,
    transient_min_m: usize = default_transient_min_m,
    /// FUCINA_GPU_VRAM_BUDGET override (bytes); null = ~80% of free VRAM at init.
    vram_budget_env: ?usize = null,
    /// FUCINA_GPU_KERNELS=src forces the NVRTC recompile path (dev loop).
    kernels_from_src: bool = false,
    /// Tensor-core quantized prefill kernel. FUCINA_GPU_QUANT_MMA=0 keeps the
    /// scalar-FFMA fallback for parity/performance diagnosis.
    quant_mma: bool = true,
    /// Split substantially underfilled quantized prefill along K, then reduce
    /// on-stream. The per-slot partial buffer is grow-only, so steady-state
    /// dispatch remains allocation-free. FUCINA_GPU_QUANT_SPLIT_K=0 keeps one
    /// block/output tile.
    quant_split_k: bool = true,
    /// FUCINA_GPU_DECODE=1 opts into experimental m<=8 quantized decode
    /// (GEMV normally; Q5_K switches to tiled MMA at m=4..8). Default off because
    /// CPU/GPU quant arithmetic is tolerance-equivalent, not bit-identical.
    decode_enabled: bool = false,
    /// Q5_K's compact CPU decode kernel is unusually strong; require enough
    /// work to cross its measured RTX/CPU boundary even when decode is on.
    min_work_decode_q5: u64 = default_min_work_decode_q5,
    /// Prefill-attention offload gate (q·kv·heads·d work).
    min_work_attn: u64 = default_min_work_attn,
};

/// Base f32 threshold. Ordinary host RHS operands must also pass the much
/// higher transient floor below; device-resident RHS uses its separately
/// measured lower threshold.
const default_min_work: u64 = 1 << 30;
/// Dense f32 GEMM with an already device-resident RHS. Against OpenBLAS-32 on
/// the reference RTX 5000 Ada host, 256^3 loses, 512^3 is a narrow GPU win,
/// and 640^3+ is decisive; 2^27 is therefore the first competitive tier.
const default_min_work_resident: u64 = 1 << 27;
/// f16 NT gate: the CPU f16 row kernels run far below the f32 blocked path
/// and f16 operands halve the PCIe bytes, so offload pays off early.
const default_min_work_f16: u64 = 1 << 27;
/// Resident f16 decode only crosses a tiny activation/result payload. On the
/// reference Ada GPU, 1x4096x1024 is 4.2x faster than the 32-thread-capable
/// CPU kernel (18.3 vs 77.4 us); 2^20 keeps smaller launch-bound dots on CPU.
const default_min_work_f16_resident: u64 = 1 << 20;
/// Resident f32 GEMV/GEMM: no RHS transfer, only a small activation/output
/// crossing. The reference RTX 5000 Ada beats OpenBLAS-32 by 12x at the
/// 4096-wide GEMV; 16 Mi work keeps launch/copy overhead amortized.
const default_min_work_gemv: u64 = 1 << 24;
/// Quantized grouped-MoE / dense-quant gates.
const default_min_work_qmoe: u64 = 1 << 30;
const default_min_work_dense_q6: u64 = 1 << 22;
/// Packed-CPU dense-linear crossovers on the reference Ada host. Q5_K wins at
/// the smallest admitted 32x1024x512 shape (47.7 vs 63.3 us), so 2^24 keeps
/// the measured boundary while rejecting lower-work calls.
const default_min_work_packed_q4: u64 = 1 << 27;
const default_min_work_packed_q5: u64 = 1 << 24;
const default_min_work_packed_q6: u64 = 1 << 24;
const default_min_work_packed_q8: u64 = 1 << 24;
/// Q5_K decode crossover against the compact (not x8-packed) CPU route:
/// 4096x4096 loses narrowly, while 6144x4096 and m>=2 at 4096 square win.
const default_min_work_decode_q5: u64 = 3 << 23;
const default_qmoe_min_fill_pct: u64 = 50;
/// Without residency every RHS streams over PCIe (measured 10.6 GB/s pageable
/// on the reference rig — ~9.5 ms per ffn-sized f32 matrix). The measured
/// break-even vs the CPU blocked kernel is m ≈ 35–40 at LLM widths; the
/// defaults keep a ~3× safety margin. `FUCINA_GPU_MIN_WORK_TRANSIENT`
/// overrides the work floor.
const default_min_work_transient: u64 = 1 << 33;
const default_transient_min_m: usize = 128;
/// Prefill attention: arithmetic intensity grows with q_seq (FLOPs O(q·kv·d),
/// bytes O(q·d)), so streamed offload pays off once the score work is a few
/// hundred MFLOP. 2^28 q·kv·heads·d keeps the ~0.5–1 ms round trip amortized;
/// decode (q_seq < the tiled floor) never reaches this seam.
const default_min_work_attn: u64 = 1 << 28;

var state: State = .{};
var config_done = std.atomic.Value(bool).init(false);
var init_done = std.atomic.Value(bool).init(false);
var init_mutex: thread.Mutex = .{};

const Ctx = struct {
    driver: api.Driver,
    device: api.CUdevice,
    context: api.CUcontext,
    stream: api.CUstream,
    upload_stream: api.CUstream,
    transfer_stream: api.CUstream,
    blas: ?api.Cublas,
    blas_handle: api.CublasHandle,
    /// CONCURRENT_MANAGED_ACCESS == 1: managed memory may be touched by the
    /// CPU while the GPU is active — the residency model's prerequisite.
    /// 0 on WSL2/some Jetson targets → residency disabled.
    managed_ok: bool,
    compute_major: c_int,
    sm_count: c_int,
    /// Tracked-allocation VRAM budget in bytes (env override or ~80% of the
    /// free VRAM sampled at init). Resident allocations beyond it return
    /// null → callers fall back to host bytes + transient, the Metal OOM path.
    vram_budget: usize,
    name_buf: [256]u8,
    name_len: usize,
};
var ctx_storage: Ctx = undefined;
var ctx_ptr: ?*Ctx = null;

// --- Optional dispatch tracing (FUCINA_GPU_TRACE=1) -------------------------
// Mirrors the Metal provider's tracing contract: zero overhead when off,
// `traceReset`/`traceDump` callable unconditionally.
var trace_on: bool = false;
const Trace = struct {
    f32_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    f32_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f32_async_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    f32_submit_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f32_wait_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_async_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_submit_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    f16_wait_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_async_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_submit_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    quant_wait_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    gemv_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    attn_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    attn_ns: std.atomic.Value(u64) = .{ .raw = 0 },
    h2d_bytes: std.atomic.Value(u64) = .{ .raw = 0 },
    d2h_bytes: std.atomic.Value(u64) = .{ .raw = 0 },
    rhs_resident: std.atomic.Value(u64) = .{ .raw = 0 },
    rhs_streamed: std.atomic.Value(u64) = .{ .raw = 0 },
    dev_alloc_calls: std.atomic.Value(u64) = .{ .raw = 0 },
    dev_alloc_bytes: std.atomic.Value(u64) = .{ .raw = 0 },
    gate_pass: std.atomic.Value(u64) = .{ .raw = 0 },
    gate_below: std.atomic.Value(u64) = .{ .raw = 0 },
    gate_shape: std.atomic.Value(u64) = .{ .raw = 0 },
    transient_below: std.atomic.Value(u64) = .{ .raw = 0 },
    cuda_err: std.atomic.Value(u64) = .{ .raw = 0 },
};
var trace: Trace = .{};

// Monotonic ns without an `std.Io` handle (backend dispatch sites have none;
// `std.time.Timer` is gone in 0.16). This file only builds into Linux
// `-Dgpu=cuda` binaries, which always link libc (build.zig configureGpu).
const CTimespec = extern struct { sec: c_long, nsec: c_long };
extern "c" fn clock_gettime(clk_id: c_int, tp: *CTimespec) c_int;
const clock_monotonic: c_int = 1; // Linux CLOCK_MONOTONIC

inline fn tinc(c: *std.atomic.Value(u64), v: u64) void {
    _ = c.fetchAdd(v, .monotonic);
}
inline fn tstart() u64 {
    if (!trace_on) return 0;
    var ts: CTimespec = undefined;
    if (clock_gettime(clock_monotonic, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}
inline fn telapsed(c: *std.atomic.Value(u64), start: u64) void {
    if (!trace_on) return;
    var ts: CTimespec = undefined;
    if (clock_gettime(clock_monotonic, &ts) != 0) return;
    const now = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    _ = c.fetchAdd(now -% start, .monotonic);
}
inline fn tfinish(start: u64) u64 {
    if (!trace_on) return 0;
    var ts: CTimespec = undefined;
    if (clock_gettime(clock_monotonic, &ts) != 0) return 0;
    const now = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    return now -% start;
}
inline fn tgate(pass: bool) void {
    if (!trace_on) return;
    tinc(if (pass) &trace.gate_pass else &trace.gate_below, 1);
}

pub fn traceEnabled() bool {
    ensureConfig();
    return trace_on;
}
/// Reset counters (call before a warm measurement window). Single-threaded.
pub fn traceReset() void {
    if (!trace_on) return;
    trace = .{};
}
/// Print the accumulated breakdown to stderr (no-op when tracing is off).
pub fn traceDump() void {
    if (!trace_on) return;
    const mb = struct {
        fn f(bytes: u64) f64 {
            return @as(f64, @floatFromInt(bytes)) / 1e6;
        }
    }.f;
    std.debug.print(
        \\[gpu-trace] cuda dispatch: f32={d} ({d:.1}ms) f16={d} ({d:.1}ms) quant={d} ({d:.1}ms) gemv={d} attn={d} ({d:.1}ms) | h2d={d:.1}MB d2h={d:.1}MB
        \\[gpu-trace] async: f32 calls={d} submit={d:.1}ms host-wait={d:.1}ms | f16 calls={d} submit={d:.1}ms host-wait={d:.1}ms | quant calls={d} submit={d:.1}ms host-wait={d:.1}ms
        \\[gpu-trace] rhs: resident={d} streamed={d} | resident allocs={d} ({d:.1}MB)
        \\[gpu-trace] gate decisions: pass={d} below-gate={d} shape-reject={d} transient-floor={d} cuda-error={d}
        \\
    , .{
        trace.f32_calls.load(.monotonic),
        @as(f64, @floatFromInt(trace.f32_ns.load(.monotonic))) / 1e6,
        trace.f16_calls.load(.monotonic),
        @as(f64, @floatFromInt(trace.f16_ns.load(.monotonic))) / 1e6,
        trace.quant_calls.load(.monotonic),
        @as(f64, @floatFromInt(trace.quant_ns.load(.monotonic))) / 1e6,
        trace.gemv_calls.load(.monotonic),
        trace.attn_calls.load(.monotonic),
        @as(f64, @floatFromInt(trace.attn_ns.load(.monotonic))) / 1e6,
        mb(trace.h2d_bytes.load(.monotonic)),
        mb(trace.d2h_bytes.load(.monotonic)),
        trace.f32_async_calls.load(.monotonic),
        @as(f64, @floatFromInt(trace.f32_submit_ns.load(.monotonic))) / 1e6,
        @as(f64, @floatFromInt(trace.f32_wait_ns.load(.monotonic))) / 1e6,
        trace.f16_async_calls.load(.monotonic),
        @as(f64, @floatFromInt(trace.f16_submit_ns.load(.monotonic))) / 1e6,
        @as(f64, @floatFromInt(trace.f16_wait_ns.load(.monotonic))) / 1e6,
        trace.quant_async_calls.load(.monotonic),
        @as(f64, @floatFromInt(trace.quant_submit_ns.load(.monotonic))) / 1e6,
        @as(f64, @floatFromInt(trace.quant_wait_ns.load(.monotonic))) / 1e6,
        trace.rhs_resident.load(.monotonic),
        trace.rhs_streamed.load(.monotonic),
        trace.dev_alloc_calls.load(.monotonic),
        mb(trace.dev_alloc_bytes.load(.monotonic)),
        trace.gate_pass.load(.monotonic),
        trace.gate_below.load(.monotonic),
        trace.gate_shape.load(.monotonic),
        trace.transient_below.load(.monotonic),
        trace.cuda_err.load(.monotonic),
    });
}

/// One-time runtime configuration read; deliberately does not touch the
/// driver so below-threshold probes stay cheap.
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
    if (std.c.getenv("FUCINA_GPU_TF32")) |v_ptr| {
        const v = std.mem.span(v_ptr);
        if (v.len > 0 and v[0] != '0') state.tf32 = true;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK")) |v_ptr| {
        state.min_work = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_RESIDENT")) |v_ptr| {
        state.min_work_resident = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_resident;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_F16")) |v_ptr| {
        state.min_work_f16 = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_f16;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_F16_RESIDENT")) |v_ptr| {
        state.min_work_f16_resident = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_f16_resident;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_GEMV")) |v_ptr| {
        state.min_work_gemv = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_gemv;
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
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_DENSE_Q5")) |v_ptr| {
        state.min_work_packed_q5 = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_packed_q5;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_DENSE_Q8")) |v_ptr| {
        state.min_work_packed_q8 = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_packed_q8;
    }
    if (std.c.getenv("FUCINA_GPU_QMOE_MIN_FILL")) |v_ptr| {
        state.qmoe_min_fill_pct = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_qmoe_min_fill_pct;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_TRANSIENT")) |v_ptr| {
        state.min_work_transient = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_transient;
    }
    if (std.c.getenv("FUCINA_GPU_VRAM_BUDGET")) |v_ptr| {
        state.vram_budget_env = std.fmt.parseInt(usize, std.mem.span(v_ptr), 10) catch null;
    }
    if (std.c.getenv("FUCINA_GPU_KERNELS")) |v_ptr| {
        state.kernels_from_src = std.mem.eql(u8, std.mem.span(v_ptr), "src");
    }
    if (std.c.getenv("FUCINA_GPU_QUANT_MMA")) |v_ptr| {
        const v = std.mem.span(v_ptr);
        if (v.len > 0 and v[0] == '0') state.quant_mma = false;
    }
    if (std.c.getenv("FUCINA_GPU_QUANT_SPLIT_K")) |v_ptr| {
        const v = std.mem.span(v_ptr);
        if (v.len > 0 and v[0] == '0') state.quant_split_k = false;
    }
    if (std.c.getenv("FUCINA_GPU_DECODE")) |v_ptr| {
        const v = std.mem.span(v_ptr);
        if (v.len > 0 and v[0] != '0') state.decode_enabled = true;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_DECODE_Q5")) |v_ptr| {
        state.min_work_decode_q5 = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_decode_q5;
    }
    if (std.c.getenv("FUCINA_GPU_MIN_WORK_ATTN")) |v_ptr| {
        state.min_work_attn = std.fmt.parseInt(u64, std.mem.span(v_ptr), 10) catch default_min_work_attn;
    }
    if (std.c.getenv("FUCINA_GPU_TRACE")) |v_ptr| {
        const v = std.mem.span(v_ptr);
        if (v.len > 0 and v[0] != '0') trace_on = true;
    }
}

/// One-time lazy device init: double-checked under a mutex so concurrent
/// ExecContexts share one context/stream/handle.
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
    var c: *Ctx = &ctx_storage;
    c.driver = api.Driver.load() catch |e| {
        std.log.warn("fucina-cuda: {t} loading libcuda.so.1; GPU offload disabled for this process", .{e});
        return;
    };
    if (c.driver.cuInit(0) != 0) {
        std.log.warn("fucina-cuda: cuInit failed; GPU offload disabled for this process", .{});
        return;
    }
    var count: c_int = 0;
    if (c.driver.cuDeviceGetCount(&count) != 0 or count < 1) {
        std.log.warn("fucina-cuda: no CUDA device; GPU offload disabled for this process", .{});
        return;
    }
    if (c.driver.cuDeviceGet(&c.device, 0) != 0) return;
    if (c.driver.cuDevicePrimaryCtxRetain(&c.context, c.device) != 0) {
        std.log.warn("fucina-cuda: cuDevicePrimaryCtxRetain failed; GPU offload disabled", .{});
        return;
    }
    if (c.driver.cuCtxSetCurrent(c.context) != 0) return;
    c.stream = null;
    if (c.driver.cuStreamCreate(&c.stream, 0) != 0) {
        std.log.warn("fucina-cuda: cuStreamCreate failed; GPU offload disabled", .{});
        return;
    }
    c.upload_stream = null;
    if (c.driver.cuStreamCreate(&c.upload_stream, 0) != 0) {
        _ = c.driver.cuStreamDestroy(c.stream);
        std.log.warn("fucina-cuda: upload stream creation failed; GPU offload disabled", .{});
        return;
    }
    c.transfer_stream = null;
    if (c.driver.cuStreamCreate(&c.transfer_stream, 0) != 0) {
        _ = c.driver.cuStreamDestroy(c.upload_stream);
        _ = c.driver.cuStreamDestroy(c.stream);
        std.log.warn("fucina-cuda: transfer stream creation failed; GPU offload disabled", .{});
        return;
    }
    c.name_len = 0;
    if (c.driver.cuDeviceGetName(&c.name_buf, c.name_buf.len, c.device) == 0) {
        c.name_len = std.mem.sliceTo(&c.name_buf, 0).len;
    }
    var cma: c_int = 0;
    _ = c.driver.cuDeviceGetAttribute(&cma, api.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, c.device);
    c.managed_ok = cma == 1;
    c.compute_major = 0;
    _ = c.driver.cuDeviceGetAttribute(&c.compute_major, api.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, c.device);
    c.sm_count = 0;
    _ = c.driver.cuDeviceGetAttribute(&c.sm_count, api.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, c.device);
    if (!c.managed_ok) {
        std.log.warn("fucina-cuda: no concurrent managed access on this platform; weight residency disabled (transient/CPU fallback)", .{});
    }
    c.vram_budget = state.vram_budget_env orelse blk: {
        var free_b: usize = 0;
        var total_b: usize = 0;
        if (c.driver.cuMemGetInfo(&free_b, &total_b) != 0) break :blk 0;
        break :blk free_b / 5 * 4;
    };
    c.blas = null;
    c.blas_handle = null;
    if (api.Cublas.load()) |loaded| {
        c.blas = loaded.api;
        var handle: api.CublasHandle = null;
        if (loaded.api.cublasCreate(&handle) == 0) {
            _ = loaded.api.cublasSetStream(handle, c.stream);
            _ = loaded.api.cublasSetMathMode(handle, if (state.tf32) api.CUBLAS_TF32_TENSOR_OP_MATH else api.CUBLAS_DEFAULT_MATH);
            c.blas_handle = handle;
            if (trace_on) std.debug.print("[gpu-trace] cuda: {s} via {s}{s}\n", .{
                c.name_buf[0..c.name_len], loaded.soname, if (state.tf32) " (TF32)" else "",
            });
        } else {
            std.log.warn("fucina-cuda: cublasCreate failed; f32 GEMM offload disabled", .{});
        }
    } else |e| {
        std.log.warn("fucina-cuda: {t} loading libcublas; f32 GEMM offload disabled", .{e});
    }
    ctx_ptr = c;
}

/// Lazy device init; null = GPU unavailable/disabled.
fn context() ?*Ctx {
    ensureInit();
    return ctx_ptr;
}

pub fn deviceName() ?[]const u8 {
    const c = context() orelse return null;
    if (c.name_len == 0) return null;
    return c.name_buf[0..c.name_len];
}

/// Gate contract (same as the Metal provider): gates decide, dispatchers run
/// unconditionally when called directly — bench-gemm bypasses the gates on
/// purpose to measure every shape, and that data is what these thresholds are
/// tuned from. The f32 tier has no RHS-lifetime information, so the transient
/// floor (`min_work_transient`, `transient_min_m`) lives HERE in the gates;
/// a residency-keyed dispatcher-side floor could replace it once the gates
/// learn RHS lifetimes, dropping these back to `min_work`.
fn gateWork(m: usize, n: usize, k: usize, work: u64) bool {
    if (m < 32 or n < 32 or k < 16) {
        if (trace_on) tinc(&trace.gate_shape, 1);
        return false;
    }
    if (!state.gpu_enabled or work < state.min_work) {
        tgate(false);
        return false;
    }
    // Shapes that pass the plain gate but fall to the transient floor are
    // counted separately: this split is the gate-retuning evidence.
    if (m < state.transient_min_m or work < state.min_work_transient) {
        if (trace_on) tinc(&trace.transient_below, 1);
        return false;
    }
    tgate(true);
    return true;
}

pub fn shouldUseGpu(m: usize, n: usize, k: usize) bool {
    ensureConfig();
    // Saturate on overflow: astronomically large work passes the work
    // thresholds but never bypasses the kill switch or shape floors.
    const work = std.math.mul(u64, std.math.mul(u64, m, n) catch std.math.maxInt(u64), k) catch std.math.maxInt(u64);
    return gateWork(m, n, k, work);
}

pub fn shouldUseGpuBatched(m: usize, n: usize, k: usize, batch_count: usize) bool {
    ensureConfig();
    const per = std.math.mul(u64, std.math.mul(u64, m, n) catch std.math.maxInt(u64), k) catch std.math.maxInt(u64);
    const work = std.math.mul(u64, per, batch_count) catch std.math.maxInt(u64);
    return gateWork(m, n, k, work);
}

/// f16 gate: no transient floor on top of `min_work_f16` — the CPU f16 row
/// kernels run an order of magnitude below the f32 blocked path (no AMX-class
/// arm, same as the Metal rationale) and f16 operands halve the PCIe bytes,
/// so streamed f16 offload pays off around the Metal-tuned 2^27 already.
pub fn shouldUseGpuF16(m: usize, n: usize, k: usize) bool {
    ensureConfig();
    if (m < 32 or n < 32 or k < 16) {
        if (trace_on) tinc(&trace.gate_shape, 1);
        return false;
    }
    const work = std.math.mul(u64, std.math.mul(u64, m, n) catch std.math.maxInt(u64), k) catch std.math.maxInt(u64);
    const pass = state.gpu_enabled and work >= state.min_work_f16;
    tgate(pass);
    return pass;
}

fn tensorHasDeviceStorage(b: anytype) bool {
    if (b.buffer.pending()) |pending| {
        if (pending.devicePtr(.cuda) != null) return true;
    }
    const ptr = @intFromPtr(b.buffer.data.ptr);
    const Elem = @TypeOf(b.buffer.data[0]);
    const len = b.buffer.data.len * @sizeOf(Elem);
    resident_lock.lock();
    const resident = residentLookupLocked(ptr, len) != null;
    resident_lock.unlock();
    return resident;
}

/// Tensor-aware f16 gate: resident model weights admit decode/small-batch
/// calls that would be disastrous with a streamed RHS. The ordinary gate
/// retains its m>=32 and high work floor for transient operands.
pub fn shouldUseGpuF16ForRhs(b: *const TensorF16, m: usize, n: usize, k: usize) bool {
    ensureConfig();
    if (m != 0 and n >= 32 and k >= 16 and state.gpu_enabled) {
        const work = std.math.mul(u64, std.math.mul(u64, m, n) catch std.math.maxInt(u64), k) catch std.math.maxInt(u64);
        if (work >= state.min_work_f16_resident) {
            const resident = tensorHasDeviceStorage(b);
            tgate(resident);
            if (resident) return true;
        }
    }
    return shouldUseGpuF16(m, n, k);
}

/// bf16 weight GEMMs do not offload on CUDA yet (Metal has the arm; the
/// cuBLAS CUDA_R_16BF route is the recorded follow-up) — the CPU bf16
/// streaming kernel serves them.
pub fn shouldUseGpuBf16ForRhs(_: *const tensor.TensorOf(.bf16), _: usize, _: usize, _: usize) bool {
    return false;
}

pub fn gemmBf16NtAsync(_: *const Tensor, _: *const tensor.TensorOf(.bf16), _: *Tensor, _: usize, _: usize, _: usize) bool {
    return false;
}

pub fn shouldUseGpuGemv(b: *const Tensor, m: usize, n: usize, k: usize) bool {
    ensureConfig();
    if (m == 0 or m > 8 or n < 256 or k < 256) return false;
    const work = std.math.mul(u64, std.math.mul(u64, m, n) catch std.math.maxInt(u64), k) catch std.math.maxInt(u64);
    if (work < state.min_work_gemv or !state.gpu_enabled) return false;
    const resident = tensorHasDeviceStorage(b);
    tgate(resident);
    return resident;
}

/// Tensor-aware native-dispatch gate. Resident weights skip the conservative
/// transient-RHS PCIe floor, while ordinary host weights retain it. This is
/// still a per-call decision: residency is storage metadata, not a graph.
pub fn shouldUseGpuForRhs(b: *const Tensor, m: usize, n: usize, k: usize) bool {
    if (shouldUseGpuGemv(b, m, n, k)) return true;
    ensureConfig();
    const work = std.math.mul(u64, std.math.mul(u64, m, n) catch std.math.maxInt(u64), k) catch std.math.maxInt(u64);
    if (state.gpu_enabled and m != 0 and n >= 32 and k >= 16 and work >= state.min_work_resident) {
        const resident = tensorHasDeviceStorage(b);
        tgate(resident);
        if (resident) return true;
    }
    return shouldUseGpu(m, n, k);
}

pub fn shouldUseGpuBatchedForRhs(b: *const Tensor, m: usize, n: usize, k: usize, batch_count: usize) bool {
    ensureConfig();
    const per = std.math.mul(u64, std.math.mul(u64, m, n) catch std.math.maxInt(u64), k) catch std.math.maxInt(u64);
    const work = std.math.mul(u64, per, batch_count) catch std.math.maxInt(u64);
    if (state.gpu_enabled and m >= 32 and n >= 32 and k >= 16 and work >= state.min_work_resident) {
        const resident = tensorHasDeviceStorage(b);
        tgate(resident);
        if (resident) return true;
    }
    return shouldUseGpuBatched(m, n, k, batch_count);
}

/// Serializes f16 GEMMs: the provider's f16 output staging (pinned host
/// memory) is reused across calls, so the caller must hold this across
/// `gemmF16Nt` + the widen of its result. Same contract as the Metal provider.
pub var f16_lock: thread.Mutex = .{};

/// Grow-only pinned host staging for the f16 GEMM result (valid until the
/// next f16 call; guarded by the caller-held `f16_lock`).
var f16_out_ptr: ?[*]f16 = null;
var f16_out_cap: usize = 0;

/// C16 = A16[m,k] · B16[n,k]ᵀ via cublasGemmEx (f16 operands, f32 accumulate
/// — the repo's dot dtype policy). Returns pinned staging valid until the
/// next f16 call, or null when the GPU didn't run. `rhs_cacheable` keeps the
/// Metal signature; residency is detected by registry lookup instead.
pub fn gemmF16Nt(a: []const f16, b: []const f16, m: usize, n: usize, k: usize, rhs_cacheable: bool) ?[]const f16 {
    _ = rhs_cacheable;
    ensureConfig();
    if (m == 0 or n == 0 or k == 0) return null;
    if (m > std.math.maxInt(i32) or n > std.math.maxInt(i32) or k > std.math.maxInt(i32)) return null;
    const a_elems = std.math.mul(usize, m, k) catch return null;
    const b_elems = std.math.mul(usize, n, k) catch return null;
    const c_elems = std.math.mul(usize, m, n) catch return null;
    if (a.len < a_elems or b.len < b_elems) return null;

    const ctx = context() orelse return null;
    if (ctx.blas_handle == null) return null;
    const blas = &(ctx.blas.?);

    const timer = tstart();
    dispatch_lock.lock();
    defer dispatch_lock.unlock();
    const d = &ctx.driver;
    if (d.cuCtxSetCurrent(ctx.context) != 0) return null;

    if (f16_out_cap < c_elems) {
        if (f16_out_ptr) |p| _ = d.cuMemFreeHost(@ptrCast(p));
        f16_out_ptr = null;
        f16_out_cap = 0;
        var raw: ?*anyopaque = null;
        if (d.cuMemHostAlloc(&raw, c_elems * 2, 0) != 0) return null;
        f16_out_ptr = @ptrCast(@alignCast(raw.?));
        f16_out_cap = c_elems;
    }
    if (!dev_a16.ensure(d, a_elems * 2) or !dev_c16.ensure(d, c_elems * 2)) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return null;
    }
    if (d.cuMemcpyHtoD(dev_a16.ptr, a.ptr, a_elems * 2) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return null;
    }
    const b_dev: api.CUdeviceptr = if (residentDevPtr(ctx, std.mem.sliceAsBytes(b[0..b_elems]), false)) |dev| blk: {
        if (trace_on) tinc(&trace.rhs_resident, 1);
        break :blk dev;
    } else blk: {
        if (trace_on) tinc(&trace.rhs_streamed, 1);
        if (!dev_b16.ensure(d, b_elems * 2)) {
            if (trace_on) tinc(&trace.cuda_err, 1);
            return null;
        }
        if (d.cuMemcpyHtoD(dev_b16.ptr, b.ptr, b_elems * 2) != 0) {
            if (trace_on) tinc(&trace.cuda_err, 1);
            return null;
        }
        if (trace_on) tinc(&trace.h2d_bytes, b_elems * 2);
        break :blk dev_b16.ptr;
    };

    // Same column-major mapping as gemmBatchedF32's .nt arm (Cᵀ = B·Aᵀ):
    // first operand = B stored [n,k] with OP_T, ld=k; second = A, OP_N, ld=k.
    const one: f32 = 1.0;
    const zero: f32 = 0.0;
    const rc = blas.cublasGemmEx(
        ctx.blas_handle,
        api.CUBLAS_OP_T,
        api.CUBLAS_OP_N,
        @intCast(n),
        @intCast(m),
        @intCast(k),
        @ptrCast(&one),
        b_dev,
        api.CUDA_R_16F,
        @intCast(k),
        dev_a16.ptr,
        api.CUDA_R_16F,
        @intCast(k),
        @ptrCast(&zero),
        dev_c16.ptr,
        api.CUDA_R_16F,
        @intCast(n),
        api.CUBLAS_COMPUTE_32F,
        api.CUBLAS_GEMM_DEFAULT,
    );
    if (rc != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return null;
    }
    if (d.cuStreamSynchronize(ctx.stream) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return null;
    }
    if (d.cuMemcpyDtoH(@ptrCast(f16_out_ptr.?), dev_c16.ptr, c_elems * 2) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return null;
    }
    if (trace_on) {
        tinc(&trace.f16_calls, 1);
        telapsed(&trace.f16_ns, timer);
        tinc(&trace.h2d_bytes, a_elems * 2);
        tinc(&trace.d2h_bytes, c_elems * 2);
    }
    return f16_out_ptr.?[0..c_elems];
}

/// C[m,n] = op(A)·op(B), f32 row-major, overwrite (beta = 0). Same operand
/// conventions as the BLAS arm and the Metal provider: nn A[m,k]/B[k,n];
/// tn A stored [k,m]; nt B stored [n,k]. Returns false when the GPU didn't run.
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

/// Grow-only device operand buffers, guarded by `dispatch_lock`. A and C
/// cross PCIe on every call, B only when non-resident — the transient floor
/// in the gates prices that in.
const DeviceBuf = struct {
    ptr: api.CUdeviceptr = 0,
    cap: usize = 0,

    fn ensure(self: *DeviceBuf, driver: *const api.Driver, bytes: usize) bool {
        if (self.cap >= bytes) return true;
        if (self.ptr != 0) {
            _ = driver.cuMemFree(self.ptr);
            self.ptr = 0;
            self.cap = 0;
        }
        var p: api.CUdeviceptr = 0;
        if (driver.cuMemAlloc(&p, bytes) != 0) return false;
        self.ptr = p;
        self.cap = bytes;
        return true;
    }
};

/// Serializes dispatches: one provider stream and one shared operand-buffer
/// set. This is a deliberate simplification the Metal f32 path does not
/// have (its command buffers run concurrently); per-thread streams are the
/// escape hatch if bench-backward-diamond shows serialization.
var dispatch_lock: thread.Mutex = .{};
var dev_a: DeviceBuf = .{};
var dev_b: DeviceBuf = .{};
var dev_c: DeviceBuf = .{};
var dev_a16: DeviceBuf = .{};
var dev_b16: DeviceBuf = .{};
var dev_c16: DeviceBuf = .{};

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
    ensureConfig();
    if (batch_count == 0 or m == 0 or n == 0 or k == 0) return false;
    if (m > std.math.maxInt(i32) or n > std.math.maxInt(i32) or k > std.math.maxInt(i32)) return false;
    if (batch_count > std.math.maxInt(i32)) return false;

    // No work gate here: gates decide, direct calls run (see shouldUseGpu —
    // bench-gemm measures every shape through this entry on purpose).
    const ctx = context() orelse return false;
    if (ctx.blas_handle == null) return false;
    const blas = &(ctx.blas.?);

    const block_a = std.math.mul(usize, m, k) catch return false; // == k*m for .tn
    const block_b = std.math.mul(usize, k, n) catch return false; // == n*k for .nt
    const block_c = std.math.mul(usize, m, n) catch return false;
    const total_a = std.math.add(usize, std.math.mul(usize, stride_a, batch_count - 1) catch return false, block_a) catch return false;
    const total_b = std.math.add(usize, std.math.mul(usize, stride_b, batch_count - 1) catch return false, block_b) catch return false;
    const total_c = std.math.add(usize, std.math.mul(usize, stride_c, batch_count - 1) catch return false, block_c) catch return false;
    if (a.len < total_a or b.len < total_b or c.len < total_c) return false;

    const timer = tstart();
    dispatch_lock.lock();
    defer dispatch_lock.unlock();
    const d = &ctx.driver;
    if (d.cuCtxSetCurrent(ctx.context) != 0) return false;

    if (!dev_a.ensure(d, total_a * 4) or !dev_c.ensure(d, total_c * 4)) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (d.cuMemcpyHtoD(dev_a.ptr, a.ptr, total_a * 4) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    // Resident RHS (managed memory, unified addressing): zero weight
    // transfer. Transient RHS streams (the backend f32 tier has no lifetime
    // info, so no adoption here).
    const b_dev: api.CUdeviceptr = if (residentDevPtr(ctx, std.mem.sliceAsBytes(b[0..total_b]), false)) |dev| blk: {
        if (trace_on) tinc(&trace.rhs_resident, 1);
        break :blk dev;
    } else blk: {
        if (trace_on) tinc(&trace.rhs_streamed, 1);
        if (!dev_b.ensure(d, total_b * 4)) {
            if (trace_on) tinc(&trace.cuda_err, 1);
            return false;
        }
        if (d.cuMemcpyHtoD(dev_b.ptr, b.ptr, total_b * 4) != 0) {
            if (trace_on) tinc(&trace.cuda_err, 1);
            return false;
        }
        if (trace_on) tinc(&trace.h2d_bytes, total_b * 4);
        break :blk dev_b.ptr;
    };

    // Row-major C = op(A)·op(B) maps to column-major as Cᵀ = op(B)ᵀ·op(A)ᵀ:
    // cuBLAS M = our n, N = our m, K = our k, first operand = B, second = A.
    // Operand ops/leading dims per orientation:
    //   nn: B op=N ld=n | A op=N ld=k
    //   nt: B op=T ld=k | A op=N ld=k   (B stored [n,k])
    //   tn: B op=N ld=n | A op=T ld=m   (A stored [k,m])
    const one: f32 = 1.0;
    const zero: f32 = 0.0;
    const cm: c_int = @intCast(n);
    const cn: c_int = @intCast(m);
    const ck: c_int = @intCast(k);
    const op_b: c_int = if (orient == .nt) api.CUBLAS_OP_T else api.CUBLAS_OP_N;
    const op_a: c_int = if (orient == .tn) api.CUBLAS_OP_T else api.CUBLAS_OP_N;
    const ld_b: c_int = if (orient == .nt) @intCast(k) else @intCast(n);
    const ld_a: c_int = if (orient == .tn) @intCast(m) else @intCast(k);
    const ld_c: c_int = @intCast(n);

    const rc = if (batch_count == 1)
        blas.cublasSgemm(ctx.blas_handle, op_b, op_a, cm, cn, ck, &one, b_dev, ld_b, dev_a.ptr, ld_a, &zero, dev_c.ptr, ld_c)
    else
        blas.cublasSgemmStridedBatched(ctx.blas_handle, op_b, op_a, cm, cn, ck, &one, b_dev, ld_b, @intCast(stride_b), dev_a.ptr, ld_a, @intCast(stride_a), &zero, dev_c.ptr, ld_c, @intCast(stride_c), @intCast(batch_count));
    if (rc != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (d.cuStreamSynchronize(ctx.stream) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (batch_count == 1 or stride_c == block_c) {
        if (d.cuMemcpyDtoH(c.ptr, dev_c.ptr, total_c * 4) != 0) {
            if (trace_on) tinc(&trace.cuda_err, 1);
            return false;
        }
    } else {
        // Strided output: copy per batch block. The gaps between blocks in
        // the device buffer are uninitialized (cuBLAS never writes them), and
        // the CPU/Metal arms leave the caller's gap bytes untouched — a bulk
        // copy would clobber live host data with garbage.
        for (0..batch_count) |bi| {
            const off = bi * stride_c;
            if (d.cuMemcpyDtoH(c[off .. off + block_c].ptr, dev_c.ptr + off * 4, block_c * 4) != 0) {
                if (trace_on) tinc(&trace.cuda_err, 1);
                return false;
            }
        }
    }
    if (trace_on) {
        tinc(&trace.f32_calls, 1);
        telapsed(&trace.f32_ns, timer);
        tinc(&trace.h2d_bytes, total_a * 4);
        tinc(&trace.d2h_bytes, total_c * 4);
    }
    return true;
}

const PinnedBuf = struct {
    ptr: ?[*]u8 = null,
    cap: usize = 0,

    fn ensure(self: *PinnedBuf, driver: *const api.Driver, bytes: usize) bool {
        if (self.cap >= bytes) return true;
        if (self.ptr) |p| _ = driver.cuMemFreeHost(@ptrCast(p));
        self.ptr = null;
        self.cap = 0;
        var raw: ?*anyopaque = null;
        if (driver.cuMemHostAlloc(&raw, bytes, 0) != 0) return false;
        self.ptr = @ptrCast(raw.?);
        self.cap = bytes;
        return true;
    }
};

/// Page-lock one ordinary Fucina allocation for its whole storage lifetime.
/// BufferPool reuse therefore amortizes registration just like its host
/// allocation, and CUDA DMA can target the tensor bytes directly.  This is a
/// resource cache only; it does not imply that the bytes are device-resident.
const CudaHostResource = struct {
    resource: accelerator.Resource,
    ctx: *Ctx,
    ptr: *anyopaque,

    const vtable: accelerator.ResourceVTable = .{ .destroy = destroy };

    fn destroy(ctx_opaque: *anyopaque) void {
        const self: *CudaHostResource = @ptrCast(@alignCast(ctx_opaque));
        _ = self.ctx.driver.cuCtxSetCurrent(self.ctx.context);
        _ = self.ctx.driver.cuMemHostUnregister(self.ptr);
        std.heap.c_allocator.destroy(self);
    }
};

fn ensureHostRegistered(ctx: *Ctx, buffer: anytype) bool {
    if (buffer.acceleratorResource(.cuda) != null) return true;
    const holder = std.heap.c_allocator.create(CudaHostResource) catch return false;
    const Elem = @TypeOf(buffer.data[0]);
    const bytes = std.math.mul(usize, buffer.data.len, @sizeOf(Elem)) catch {
        std.heap.c_allocator.destroy(holder);
        return false;
    };
    if (ctx.driver.cuMemHostRegister(buffer.data.ptr, bytes, 0) != 0) {
        std.heap.c_allocator.destroy(holder);
        return buffer.acceleratorResource(.cuda) != null;
    }
    holder.* = .{
        .resource = .{ .provider = .cuda, .ctx = holder, .vtable = &CudaHostResource.vtable },
        .ctx = ctx,
        .ptr = buffer.data.ptr,
    };
    if (buffer.setAcceleratorResource(&holder.resource)) return true;
    holder.resource.destroy();
    return buffer.acceleratorResource(.cuda) != null;
}

/// One reusable in-flight operand set.  Streams and cuBLAS stay process-open;
/// these slots remove cuMemAlloc/cuMemHostAlloc from steady-state calls while
/// allowing a short queue of independent/dependent eager ops.
const AsyncSlot = struct {
    busy: bool = false,
    a_dev: DeviceBuf = .{},
    b_dev: DeviceBuf = .{},
    c_dev: DeviceBuf = .{},
    partial_dev: DeviceBuf = .{},
    c_host: PinnedBuf = .{},
    aux_dev: DeviceBuf = .{},
    aux_host: PinnedBuf = .{},
    inputs_ready: api.CUevent = null,
    done: api.CUevent = null,
};

const async_slot_count = 8;
var async_slots: [async_slot_count]AsyncSlot = [_]AsyncSlot{.{}} ** async_slot_count;
var async_slots_lock: thread.Mutex = .{};

fn acquireAsyncSlot(ctx: *Ctx) ?*AsyncSlot {
    async_slots_lock.lock();
    defer async_slots_lock.unlock();
    for (&async_slots) |*slot| {
        if (slot.busy) continue;
        if (slot.done == null and ctx.driver.cuEventCreate(&slot.done, 2) != 0) return null; // CU_EVENT_DISABLE_TIMING
        if (slot.inputs_ready == null and ctx.driver.cuEventCreate(&slot.inputs_ready, 2) != 0) return null;
        slot.busy = true;
        return slot;
    }
    return null;
}

fn releaseAsyncSlot(slot: *AsyncSlot) void {
    async_slots_lock.lock();
    defer async_slots_lock.unlock();
    std.debug.assert(slot.busy);
    slot.busy = false;
}

const AsyncWorkKind = enum { f32, f16, quant };

fn CudaWorkFor(comptime input_dtype: storage.DType) type {
    return struct {
        work: accelerator.Work,
        ctx: *Ctx,
        slot: *AsyncSlot,
        output: [*]f32,
        total_c: usize,
        block_c: usize,
        stride_c: usize,
        batch_count: usize,
        device_base: usize,
        dep_a: ?*accelerator.Work,
        dep_b: ?*accelerator.Work,
        a_buffer: *storage.BufferOf(input_dtype),
        b_buffer: ?*storage.BufferOf(input_dtype),
        output_registered: bool,
        kind: AsyncWorkKind,

        const Self = @This();
        const vtable: accelerator.WorkVTable = .{
            .finish = finish,
            .device_ptr = devicePtr,
            .destroy = destroy,
        };

        fn finish(ctx_opaque: *anyopaque, copy_to_host: bool) bool {
            const self: *Self = @ptrCast(@alignCast(ctx_opaque));
            defer {
                self.a_buffer.clearPendingUse(&self.work);
                if (self.b_buffer) |buffer| buffer.clearPendingUse(&self.work);
            }
            const d = &self.ctx.driver;
            const started = tstart();
            if (d.cuCtxSetCurrent(self.ctx.context) != 0) return false;
            if (copy_to_host) {
                const bytes = self.total_c * @sizeOf(f32);
                // Put the dependency on the persistent download lane instead of
                // blocking the calling CPU until compute completes and only then
                // starting DMA.  The final stream wait is the sole host fence.
                if (d.cuStreamWaitEvent(self.ctx.transfer_stream, self.slot.done, 0) != 0) return false;
                if (self.output_registered) {
                    if (self.batch_count == 1 or self.stride_c == self.block_c) {
                        if (d.cuMemcpyDtoHAsync(self.output, self.slot.c_dev.ptr, bytes, self.ctx.transfer_stream) != 0) return false;
                    } else {
                        for (0..self.batch_count) |bi| {
                            const off = bi * self.stride_c;
                            if (d.cuMemcpyDtoHAsync(self.output + off, self.slot.c_dev.ptr + off * @sizeOf(f32), self.block_c * @sizeOf(f32), self.ctx.transfer_stream) != 0) return false;
                        }
                    }
                    if (d.cuStreamSynchronize(self.ctx.transfer_stream) != 0) return false;
                } else {
                    if (d.cuMemcpyDtoHAsync(self.slot.c_host.ptr.?, self.slot.c_dev.ptr, bytes, self.ctx.transfer_stream) != 0 or
                        d.cuStreamSynchronize(self.ctx.transfer_stream) != 0)
                        return false;
                    const staged: [*]const f32 = @ptrCast(@alignCast(self.slot.c_host.ptr.?));
                    if (self.batch_count == 1 or self.stride_c == self.block_c) {
                        @memcpy(self.output[0..self.total_c], staged[0..self.total_c]);
                    } else {
                        for (0..self.batch_count) |bi| {
                            const off = bi * self.stride_c;
                            @memcpy(self.output[off..][0..self.block_c], staged[off..][0..self.block_c]);
                        }
                    }
                }
                if (trace_on) tinc(&trace.d2h_bytes, bytes);
            } else if (d.cuEventSynchronize(self.slot.done) != 0) {
                // Discard has no host transfer to carry the dependency, but the
                // slot cannot be recycled while compute still touches it.
                return false;
            }
            if (trace_on) tinc(switch (self.kind) {
                .f32 => &trace.f32_wait_ns,
                .f16 => &trace.f16_wait_ns,
                .quant => &trace.quant_wait_ns,
            }, tfinish(started));
            return true;
        }

        fn devicePtr(ctx_opaque: *anyopaque) ?usize {
            const self: *Self = @ptrCast(@alignCast(ctx_opaque));
            return self.device_base;
        }

        fn destroy(ctx_opaque: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx_opaque));
            if (self.dep_a) |dep| dep.release();
            if (self.dep_b) |dep| dep.release();
            self.a_buffer.release();
            if (self.b_buffer) |buffer| buffer.release();
            releaseAsyncSlot(self.slot);
            std.heap.c_allocator.destroy(self);
        }
    };
}

const CudaWork = CudaWorkFor(.f32);
const CudaF16Work = CudaWorkFor(.f16);

fn pendingDeviceInput(x: *const Tensor, dep: *?*accelerator.Work) ?api.CUdeviceptr {
    const work = x.buffer.pending() orelse return null;
    const base = work.devicePtr(.cuda) orelse return null;
    work.retain();
    dep.* = work;
    return @intCast(base + x.offset * @sizeOf(f32));
}

fn stageInput(
    ctx: *Ctx,
    x: *const Tensor,
    elems: usize,
    dev: *DeviceBuf,
    copy_queued: *bool,
) ?api.CUdeviceptr {
    @constCast(x.buffer).waitReady();
    const bytes = std.math.mul(usize, elems, @sizeOf(f32)) catch return null;
    const values = x.buffer.data[x.offset..][0..elems];
    if (residentDevPtr(ctx, std.mem.sliceAsBytes(values), false)) |resident| return resident;
    if (!dev.ensure(&ctx.driver, bytes)) return null;
    // Registration is an amortized best effort: allocators unsupported by
    // the driver retain the correct pageable async fallback.
    _ = ensureHostRegistered(ctx, x.buffer);
    if (ctx.driver.cuMemcpyHtoDAsync(dev.ptr, values.ptr, bytes, ctx.upload_stream) != 0) return null;
    copy_queued.* = true;
    if (trace_on) tinc(&trace.h2d_bytes, bytes);
    return dev.ptr;
}

fn stageInputF16(
    ctx: *Ctx,
    x: *const TensorF16,
    elems: usize,
    dev: *DeviceBuf,
    copy_queued: *bool,
) ?api.CUdeviceptr {
    @constCast(x.buffer).waitReady();
    const bytes = std.math.mul(usize, elems, @sizeOf(f16)) catch return null;
    const values = x.buffer.data[x.offset..][0..elems];
    if (residentDevPtr(ctx, std.mem.sliceAsBytes(values), false)) |resident| return resident;
    if (!dev.ensure(&ctx.driver, bytes)) return null;
    _ = ensureHostRegistered(ctx, x.buffer);
    if (ctx.driver.cuMemcpyHtoDAsync(dev.ptr, values.ptr, bytes, ctx.upload_stream) != 0) return null;
    copy_queued.* = true;
    if (trace_on) tinc(&trace.h2d_bytes, bytes);
    return dev.ptr;
}

/// CUDA twin of the Metal eager-async seam.  H2D input copies run on a
/// persistent upload stream and feed the persistent compute stream by event.
/// A dependent op consumes the
/// producer slot directly; host materialization happens only at a CPU
/// boundary, through the persistent transfer stream.
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
    if (m > std.math.maxInt(i32) or n > std.math.maxInt(i32) or k > std.math.maxInt(i32) or batch_count > std.math.maxInt(i32)) return false;
    const block_a = std.math.mul(usize, m, k) catch return false;
    const block_b = std.math.mul(usize, k, n) catch return false;
    const block_c = std.math.mul(usize, m, n) catch return false;
    const total_a = std.math.add(usize, std.math.mul(usize, stride_a, batch_count - 1) catch return false, block_a) catch return false;
    const total_b = std.math.add(usize, std.math.mul(usize, stride_b, batch_count - 1) catch return false, block_b) catch return false;
    const total_c = std.math.add(usize, std.math.mul(usize, stride_c, batch_count - 1) catch return false, block_c) catch return false;
    if (a.offset + total_a > a.buffer.data.len or b.offset + total_b > b.buffer.data.len or out.offset + total_c > out.buffer.data.len) return false;
    if (out.buffer.pending() != null) return false;

    const ctx = context() orelse return false;
    if (ctx.blas_handle == null) return false;
    dispatch_lock.lock();
    defer dispatch_lock.unlock();
    if (ctx.driver.cuCtxSetCurrent(ctx.context) != 0) return false;
    const slot = acquireAsyncSlot(ctx) orelse return false;
    const holder = std.heap.c_allocator.create(CudaWork) catch {
        releaseAsyncSlot(slot);
        return false;
    };
    var dep_a: ?*accelerator.Work = null;
    var dep_b: ?*accelerator.Work = null;
    var copy_queued = false;
    var success = false;
    defer if (!success) {
        _ = ctx.driver.cuStreamSynchronize(ctx.upload_stream);
        _ = ctx.driver.cuStreamSynchronize(ctx.stream);
        if (dep_a) |dep| dep.release();
        if (dep_b) |dep| dep.release();
        releaseAsyncSlot(slot);
        std.heap.c_allocator.destroy(holder);
    };

    const submit_started = tstart();
    if (!slot.c_dev.ensure(&ctx.driver, total_c * @sizeOf(f32))) return false;
    const output_registered = ensureHostRegistered(ctx, out.buffer);
    if (!output_registered and !slot.c_host.ensure(&ctx.driver, total_c * @sizeOf(f32))) return false;
    const a_dev = pendingDeviceInput(a, &dep_a) orelse stageInput(ctx, a, total_a, &slot.a_dev, &copy_queued) orelse return false;
    const b_dev = pendingDeviceInput(b, &dep_b) orelse stageInput(ctx, b, total_b, &slot.b_dev, &copy_queued) orelse return false;
    if (copy_queued) {
        if (ctx.driver.cuEventRecord(slot.inputs_ready, ctx.upload_stream) != 0 or
            ctx.driver.cuStreamWaitEvent(ctx.stream, slot.inputs_ready, 0) != 0) return false;
    }

    const one: f32 = 1.0;
    const zero: f32 = 0.0;
    const op_b: c_int = if (orient == .nt) api.CUBLAS_OP_T else api.CUBLAS_OP_N;
    const op_a: c_int = if (orient == .tn) api.CUBLAS_OP_T else api.CUBLAS_OP_N;
    const ld_b: c_int = if (orient == .nt) @intCast(k) else @intCast(n);
    const ld_a: c_int = if (orient == .tn) @intCast(m) else @intCast(k);
    const blas = &(ctx.blas.?);
    const rc = if (batch_count == 1)
        blas.cublasSgemm(ctx.blas_handle, op_b, op_a, @intCast(n), @intCast(m), @intCast(k), &one, b_dev, ld_b, a_dev, ld_a, &zero, slot.c_dev.ptr, @intCast(n))
    else
        blas.cublasSgemmStridedBatched(ctx.blas_handle, op_b, op_a, @intCast(n), @intCast(m), @intCast(k), &one, b_dev, ld_b, @intCast(stride_b), a_dev, ld_a, @intCast(stride_a), &zero, slot.c_dev.ptr, @intCast(n), @intCast(stride_c), @intCast(batch_count));
    if (rc != 0 or ctx.driver.cuEventRecord(slot.done, ctx.stream) != 0) return false;

    a.buffer.retain();
    b.buffer.retain();
    holder.* = .{
        .work = accelerator.Work.init(.cuda, holder, &CudaWork.vtable),
        .ctx = ctx,
        .slot = slot,
        .output = out.buffer.data[out.offset..].ptr,
        .total_c = total_c,
        .block_c = block_c,
        .stride_c = stride_c,
        .batch_count = batch_count,
        .device_base = @as(usize, @intCast(slot.c_dev.ptr)) - out.offset * @sizeOf(f32),
        .dep_a = dep_a,
        .dep_b = dep_b,
        .a_buffer = a.buffer,
        .b_buffer = b.buffer,
        .output_registered = output_registered,
        .kind = .f32,
    };
    a.buffer.setPendingUse(&holder.work);
    b.buffer.setPendingUse(&holder.work);
    out.buffer.setPending(&holder.work);
    success = true;
    if (trace_on) {
        tinc(&trace.f32_async_calls, 1);
        tinc(&trace.f32_submit_ns, tfinish(submit_started));
    }
    return true;
}

pub fn gemmF32Async(orient: Orient, a: *const Tensor, b: *const Tensor, out: *Tensor, m: usize, n: usize, k: usize) bool {
    return gemmBatchedF32Async(orient, a, b, out, m, n, k, 1, 0, 0, 0);
}

/// Eager f16-operands NT GEMM. cuBLAS accumulates and writes f32 directly;
/// uploads, compute, and deferred download use the same persistent streams and
/// bounded slots as f32 GEMM. A resident GGUF RHS therefore crosses PCIe only
/// at model load/first residency and dependent f32 ops consume `c_dev` in place.
pub fn gemmF16NtAsync(a: *const TensorF16, b: *const TensorF16, out: *Tensor, m: usize, n: usize, k: usize) bool {
    if (m == 0 or n == 0 or k == 0) return false;
    if (m > std.math.maxInt(i32) or n > std.math.maxInt(i32) or k > std.math.maxInt(i32)) return false;
    const a_elems = std.math.mul(usize, m, k) catch return false;
    const b_elems = std.math.mul(usize, n, k) catch return false;
    const c_elems = std.math.mul(usize, m, n) catch return false;
    if (a.offset + a_elems > a.buffer.data.len or b.offset + b_elems > b.buffer.data.len or out.offset + c_elems > out.buffer.data.len) return false;
    if (out.buffer.pending() != null) return false;

    const ctx = context() orelse return false;
    if (ctx.blas_handle == null) return false;
    dispatch_lock.lock();
    defer dispatch_lock.unlock();
    if (ctx.driver.cuCtxSetCurrent(ctx.context) != 0) return false;
    const slot = acquireAsyncSlot(ctx) orelse return false;
    const holder = std.heap.c_allocator.create(CudaF16Work) catch {
        releaseAsyncSlot(slot);
        return false;
    };
    var copy_queued = false;
    var success = false;
    defer if (!success) {
        _ = ctx.driver.cuStreamSynchronize(ctx.upload_stream);
        _ = ctx.driver.cuStreamSynchronize(ctx.stream);
        releaseAsyncSlot(slot);
        std.heap.c_allocator.destroy(holder);
    };

    const submit_started = tstart();
    if (!slot.c_dev.ensure(&ctx.driver, c_elems * @sizeOf(f32))) return false;
    const output_registered = ensureHostRegistered(ctx, out.buffer);
    if (!output_registered and !slot.c_host.ensure(&ctx.driver, c_elems * @sizeOf(f32))) return false;
    const a_dev = stageInputF16(ctx, a, a_elems, &slot.a_dev, &copy_queued) orelse return false;
    const b_dev = stageInputF16(ctx, b, b_elems, &slot.b_dev, &copy_queued) orelse return false;
    if (copy_queued) {
        if (ctx.driver.cuEventRecord(slot.inputs_ready, ctx.upload_stream) != 0 or
            ctx.driver.cuStreamWaitEvent(ctx.stream, slot.inputs_ready, 0) != 0) return false;
    }

    const one: f32 = 1.0;
    const zero: f32 = 0.0;
    const rc = ctx.blas.?.cublasGemmEx(
        ctx.blas_handle,
        api.CUBLAS_OP_T,
        api.CUBLAS_OP_N,
        @intCast(n),
        @intCast(m),
        @intCast(k),
        @ptrCast(&one),
        b_dev,
        api.CUDA_R_16F,
        @intCast(k),
        a_dev,
        api.CUDA_R_16F,
        @intCast(k),
        @ptrCast(&zero),
        slot.c_dev.ptr,
        api.CUDA_R_32F,
        @intCast(n),
        api.CUBLAS_COMPUTE_32F,
        api.CUBLAS_GEMM_DEFAULT,
    );
    if (rc != 0 or ctx.driver.cuEventRecord(slot.done, ctx.stream) != 0) return false;

    a.buffer.retain();
    b.buffer.retain();
    holder.* = .{
        .work = accelerator.Work.init(.cuda, holder, &CudaF16Work.vtable),
        .ctx = ctx,
        .slot = slot,
        .output = out.buffer.data[out.offset..].ptr,
        .total_c = c_elems,
        .block_c = c_elems,
        .stride_c = c_elems,
        .batch_count = 1,
        .device_base = @as(usize, @intCast(slot.c_dev.ptr)) - out.offset * @sizeOf(f32),
        .dep_a = null,
        .dep_b = null,
        .a_buffer = a.buffer,
        .b_buffer = b.buffer,
        .output_registered = output_registered,
        .kind = .f16,
    };
    a.buffer.setPendingUse(&holder.work);
    b.buffer.setPendingUse(&holder.work);
    out.buffer.setPending(&holder.work);
    success = true;
    if (trace_on) {
        tinc(&trace.f16_async_calls, 1);
        tinc(&trace.f16_submit_ns, tfinish(submit_started));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Quantized (dequant-in-kernel) grouped GEMM. Types and wrappers mirror the
// Metal provider exactly so exec/llm consumers compile unchanged.
// ---------------------------------------------------------------------------

/// Weight block formats the quantized kernel reads directly.
pub const QFormat = enum(c_int) {
    q8_0 = 0,
    q6_k = 1,
    q4_k = 2,
    q5_k = 3,

    /// K (the reduced dim) must be a whole number of blocks.
    pub fn kMultiple(self: QFormat) usize {
        return switch (self) {
            .q8_0 => 32,
            .q4_k, .q5_k, .q6_k => 256,
        };
    }
};

/// One 32-row output tile of one expert group (the CPU-built tile table
/// protocol shared with the Metal provider).
pub const QMMTile = extern struct {
    expert: i32,
    base_row: i32,
    m: i32,
    tile_m: i32,
};

// --- Weight residency ---------------------------------------------------------
// `cuMemAllocManaged` + `CU_MEM_ADVISE_SET_READ_MOSTLY` preserves the Metal
// contract exactly: one pointer, CPU-readable bytes (the CPU fallback and
// decode paths read the SAME storage at full host bandwidth — READ_MOSTLY
// duplicates read-only pages instead of ping-ponging them), and unified addressing lets dispatchers use the pointer as a
// device address with zero weight transfer. Entries are prefetched to device
// on first GPU use.

const ResidentEntry = struct {
    /// Host-visible source range this entry answers for. For `owned` entries
    /// (allocResidentBytes) this IS the managed allocation; for `adopted`
    /// entries it is the caller's stable bytes (e.g. mmap'd GGUF blocks) and
    /// `dev_base` is the one-time managed copy.
    src_base: usize,
    len: usize,
    dev_base: usize,
    owned: bool,
    prefetched: bool,
};
const resident_slots = 512;
var resident_lock: thread.Mutex = .{};
var resident_entries: [resident_slots]ResidentEntry = undefined;
var resident_count: usize = 0;
var resident_outstanding: usize = 0;

fn residentLookupLocked(ptr: usize, len: usize) ?usize {
    for (resident_entries[0..resident_count], 0..) |e, i| {
        if (ptr >= e.src_base and ptr + len <= e.src_base + e.len) return i;
    }
    return null;
}

/// Resolve `bytes` to a zero-transfer device pointer:
/// - registry hit (owned managed bytes, or a previously adopted stable
///   range) → prefetch-once → device address (+ interior offset);
/// - miss with `adopt_if_stable` (RhsLifetime.stable_process, i.e. the
///   caller's `rhs_cacheable`) → one-time managed copy registered by SOURCE
///   address — the direct analog of the Metal shim's page wrap cache, with
///   the same stale-pages rule: only process-lifetime bytes may be adopted;
/// - otherwise null → the caller streams (or, for the decode arm, refuses).
fn residentDevPtr(ctx: *Ctx, bytes: []const u8, adopt_if_stable: bool) ?api.CUdeviceptr {
    const ptr = @intFromPtr(bytes.ptr);
    resident_lock.lock();
    if (residentLookupLocked(ptr, bytes.len)) |i| {
        const e = &resident_entries[i];
        const need = !e.prefetched;
        if (need) e.prefetched = true;
        const dev = e.dev_base + (ptr - e.src_base);
        const pf_base = e.dev_base;
        const pf_len = e.len;
        resident_lock.unlock();
        if (need) _ = ctx.driver.cuMemPrefetchAsync(@intCast(pf_base), pf_len, ctx.device, ctx.stream);
        return @intCast(dev);
    }
    const over_budget = resident_count >= resident_slots or
        (ctx.vram_budget > 0 and resident_outstanding + bytes.len > ctx.vram_budget);
    resident_lock.unlock();
    if (!adopt_if_stable or !ctx.managed_ok or over_budget) return null;

    // Adopt: managed copy of the stable source bytes, READ_MOSTLY-advised.
    if (ctx.driver.cuCtxSetCurrent(ctx.context) != 0) return null;
    var dptr: api.CUdeviceptr = 0;
    if (ctx.driver.cuMemAllocManaged(&dptr, bytes.len, api.CU_MEM_ATTACH_GLOBAL) != 0) return null;
    _ = ctx.driver.cuMemAdvise(dptr, bytes.len, api.CU_MEM_ADVISE_SET_READ_MOSTLY, ctx.device);
    const copy: [*]u8 = @ptrFromInt(@as(usize, @intCast(dptr)));
    @memcpy(copy[0..bytes.len], bytes);

    resident_lock.lock();
    if (residentLookupLocked(ptr, bytes.len)) |i| {
        // Raced with another adopter; keep theirs.
        const dev = resident_entries[i].dev_base + (ptr - resident_entries[i].src_base);
        resident_lock.unlock();
        _ = ctx.driver.cuMemFree(dptr);
        return @intCast(dev);
    }
    // Recheck BOTH bounds under the lock: the earlier probe was check-then-act
    // across a release, and concurrent adopters must not jointly exceed the
    // budget (oversubscribed managed memory silently page-thrashes).
    if (resident_count >= resident_slots or
        (ctx.vram_budget > 0 and resident_outstanding + bytes.len > ctx.vram_budget))
    {
        resident_lock.unlock();
        _ = ctx.driver.cuMemFree(dptr);
        return null;
    }
    resident_entries[resident_count] = .{
        .src_base = ptr,
        .len = bytes.len,
        .dev_base = @intCast(dptr),
        .owned = false,
        .prefetched = true,
    };
    resident_count += 1;
    resident_outstanding += bytes.len;
    resident_lock.unlock();
    _ = ctx.driver.cuMemPrefetchAsync(dptr, bytes.len, ctx.device, ctx.stream);
    if (trace_on) {
        tinc(&trace.dev_alloc_calls, 1);
        tinc(&trace.dev_alloc_bytes, bytes.len);
    }
    return dptr;
}

/// Managed, READ_MOSTLY-advised, budget-tracked resident bytes. Null when the
/// GPU/managed access is unavailable, the registry is full, or the allocation
/// would exceed FUCINA_GPU_VRAM_BUDGET — callers fall back to host bytes +
/// `.transient`, exactly the Metal OOM path.
// --- Evolution-strategies device arm (fucina.es) ----------------------------
//
// Runs perturb/update/anchored-decay directly on RESIDENT parameters so the
// population loop never migrates managed pages to the CPU. The kernels
// reproduce the CPU noise contract bitwise (rn-intrinsics, no FMA
// contraction — see kernels.cu); `false` means "device did not run it" and
// the caller keeps its CPU path (non-resident bytes, missing kernels, or a
// GPU-less process).

pub const EsDType = enum(usize) { f16 = 0, f32 = 1 };

fn esDevicePtr(ctx: *Ctx, bytes: []const u8) ?api.CUdeviceptr {
    // Never adopt: the ES arm must write the caller's live storage, not a
    // snapshot copy. Only owned managed allocations qualify.
    return residentDevPtr(ctx, bytes, false);
}

fn esLaunchGrid(work_items: u64) c_uint {
    const threads = 256;
    const blocks = (work_items + threads - 1) / threads;
    return @intCast(@min(blocks, 4096));
}

pub fn esPerturb(dt: EsDType, bytes: []u8, stream_seed: u64, scaled: f32, n: usize) bool {
    ensureConfig();
    const ctx = context() orelse return false;
    dispatch_lock.lock();
    defer dispatch_lock.unlock();
    const d = &ctx.driver;
    if (d.cuCtxSetCurrent(ctx.context) != 0) return false;
    const ks = ensureKernels(ctx) orelse return false;
    const f = ks.es_perturb[@intFromEnum(dt)] orelse return false;
    const dev = esDevicePtr(ctx, bytes) orelse return false;

    var p_data = dev;
    var p_seed: u64 = stream_seed;
    var p_scaled: f32 = scaled;
    var p_n: u64 = @intCast(n);
    var params = [_]?*anyopaque{ @ptrCast(&p_data), @ptrCast(&p_seed), @ptrCast(&p_scaled), @ptrCast(&p_n) };
    const pairs = (@as(u64, n) + 1) / 2;
    if (d.cuLaunchKernel(f, esLaunchGrid(pairs), 1, 1, 256, 1, 1, 0, ctx.stream, &params, null) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (d.cuStreamSynchronize(ctx.stream) != 0) return false;
    return true;
}

var dev_es_seeds: DeviceBuf = .{};
var dev_es_coeffs: DeviceBuf = .{};

pub fn esUpdate(dt: EsDType, bytes: []u8, stream_seeds: []const u64, coeffs: []const f32, scale: f32, n: usize) bool {
    ensureConfig();
    if (stream_seeds.len != coeffs.len or stream_seeds.len == 0) return false;
    if (stream_seeds.len > std.math.maxInt(u32)) return false;
    const ctx = context() orelse return false;
    dispatch_lock.lock();
    defer dispatch_lock.unlock();
    const d = &ctx.driver;
    if (d.cuCtxSetCurrent(ctx.context) != 0) return false;
    const ks = ensureKernels(ctx) orelse return false;
    const f = ks.es_update[@intFromEnum(dt)] orelse return false;
    const dev = esDevicePtr(ctx, bytes) orelse return false;

    if (!dev_es_seeds.ensure(d, stream_seeds.len * 8) or !dev_es_coeffs.ensure(d, coeffs.len * 4)) return false;
    if (d.cuMemcpyHtoD(dev_es_seeds.ptr, stream_seeds.ptr, stream_seeds.len * 8) != 0) return false;
    if (d.cuMemcpyHtoD(dev_es_coeffs.ptr, coeffs.ptr, coeffs.len * 4) != 0) return false;

    var p_data = dev;
    var p_seeds = dev_es_seeds.ptr;
    var p_coeffs = dev_es_coeffs.ptr;
    var p_streams: u32 = @intCast(stream_seeds.len);
    var p_scale: f32 = scale;
    var p_n: u64 = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(&p_data),    @ptrCast(&p_seeds), @ptrCast(&p_coeffs),
        @ptrCast(&p_streams), @ptrCast(&p_scale), @ptrCast(&p_n),
    };
    const pairs = (@as(u64, n) + 1) / 2;
    if (d.cuLaunchKernel(f, esLaunchGrid(pairs), 1, 1, 256, 1, 1, 0, ctx.stream, &params, null) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (d.cuStreamSynchronize(ctx.stream) != 0) return false;
    return true;
}

pub fn esAnchor(dt: EsDType, bytes: []u8, anchor: []const u8, decay_step: f32, is_l1: bool, n: usize) bool {
    ensureConfig();
    if (anchor.len != bytes.len) return false;
    const ctx = context() orelse return false;
    dispatch_lock.lock();
    defer dispatch_lock.unlock();
    const d = &ctx.driver;
    if (d.cuCtxSetCurrent(ctx.context) != 0) return false;
    const ks = ensureKernels(ctx) orelse return false;
    const f = ks.es_anchor[@intFromEnum(dt)] orelse return false;
    const dev = esDevicePtr(ctx, bytes) orelse return false;
    const anchor_dev = esDevicePtr(ctx, anchor) orelse return false;

    var p_data = dev;
    var p_anchor = anchor_dev;
    var p_step: f32 = decay_step;
    var p_l1: c_int = @intFromBool(is_l1);
    var p_n: u64 = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(&p_data), @ptrCast(&p_anchor), @ptrCast(&p_step), @ptrCast(&p_l1), @ptrCast(&p_n),
    };
    if (d.cuLaunchKernel(f, esLaunchGrid(@intCast(n)), 1, 1, 256, 1, 1, 0, ctx.stream, &params, null) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (d.cuStreamSynchronize(ctx.stream) != 0) return false;
    return true;
}

pub fn allocResidentBytes(len: usize) ?[]u8 {
    if (len == 0) return null;
    const ctx = context() orelse return null;
    if (!ctx.managed_ok) return null;
    resident_lock.lock();
    const over_budget = resident_count >= resident_slots or
        (ctx.vram_budget > 0 and resident_outstanding + len > ctx.vram_budget);
    resident_lock.unlock();
    if (over_budget) return null;

    if (ctx.driver.cuCtxSetCurrent(ctx.context) != 0) return null;
    var dptr: api.CUdeviceptr = 0;
    if (ctx.driver.cuMemAllocManaged(&dptr, len, api.CU_MEM_ATTACH_GLOBAL) != 0) return null;
    // Advisory only: failure just loses the page-duplication optimization.
    _ = ctx.driver.cuMemAdvise(dptr, len, api.CU_MEM_ADVISE_SET_READ_MOSTLY, ctx.device);

    resident_lock.lock();
    if (resident_count >= resident_slots or
        (ctx.vram_budget > 0 and resident_outstanding + len > ctx.vram_budget))
    {
        resident_lock.unlock();
        _ = ctx.driver.cuMemFree(dptr);
        return null;
    }
    resident_entries[resident_count] = .{
        .src_base = @intCast(dptr),
        .len = len,
        .dev_base = @intCast(dptr),
        .owned = true,
        .prefetched = false,
    };
    resident_count += 1;
    resident_outstanding += len;
    resident_lock.unlock();

    if (trace_on) {
        tinc(&trace.dev_alloc_calls, 1);
        tinc(&trace.dev_alloc_bytes, len);
    }
    const p: [*]u8 = @ptrFromInt(@as(usize, @intCast(dptr)));
    return p[0..len];
}

/// Release bytes returned by `allocResidentBytes`. Safe no-op for slices that
/// did not come from the resident allocator. Synchronizes the provider stream
/// before freeing (an in-flight prefetch on the range must not race the free)
/// and sets the context — release hooks run on arbitrary threads.
pub fn freeResidentBytes(bytes: []const u8) void {
    if (bytes.len == 0) return;
    const ctx = ctx_ptr orelse return;
    const base = @intFromPtr(bytes.ptr);

    resident_lock.lock();
    var dev_base: ?usize = null;
    for (resident_entries[0..resident_count], 0..) |e, i| {
        // Owned entries match their managed base; adopted entries match the
        // SOURCE base — an owner freeing adopted source bytes must evict the
        // stale mapping (and its managed copy) or a later allocation at the
        // same address would false-hit the registry.
        if (e.src_base == base) {
            dev_base = e.dev_base;
            resident_outstanding -= e.len;
            resident_entries[i] = resident_entries[resident_count - 1];
            resident_count -= 1;
            break;
        }
    }
    resident_lock.unlock();
    const dev = dev_base orelse return;

    if (ctx.driver.cuCtxSetCurrent(ctx.context) != 0) return;
    _ = ctx.driver.cuStreamSynchronize(ctx.stream);
    _ = ctx.driver.cuMemFree(@intCast(dev));
}

// --- Kernel module (vendored PTX -> driver JIT; NVRTC dev-loop fallback) ----

const kernels_ptx = @embedFile("cuda/kernels.ptx");
const kernels_src = @embedFile("cuda/kernels.cu");

const Kernels = struct {
    mul_mm: [4]api.CUfunction, // indexed by @intFromEnum(QFormat)
    mul_mm_mma: [4]?api.CUfunction,
    mul_mm_mma_n32: [4]?api.CUfunction,
    reduce_split_k: ?api.CUfunction,
    gemv: [4]api.CUfunction,
    attn_f16: api.CUfunction,
    // ES kernels ([0] = f16, [1] = f32). Optional: a stale vendored PTX
    // without them only disables the device ES arm (CPU fallback), never
    // the quant/attention arms.
    es_perturb: [2]?api.CUfunction,
    es_update: [2]?api.CUfunction,
    es_anchor: [2]?api.CUfunction,
};
var kernels_state: ?Kernels = null;
// Publication flag for the lock-free fast path (same acquire/release pattern
// as init_done): callers may reach ensureKernels holding different locks
// (qmoe_lock vs dispatch_lock), so the fast-path read must not race the
// under-mutex initialization of the Kernels struct.
var kernels_ready = std.atomic.Value(bool).init(false);
var kernels_tried: bool = false;

fn nvrtcCompilePtx() ?[:0]u8 {
    const nvrtc = api.Nvrtc.load() catch return null;
    var prog: api.NvrtcProgram = null;
    if (nvrtc.nvrtcCreateProgram(&prog, kernels_src, "fucina_kernels.cu", 0, null, null) != 0) return null;
    defer _ = nvrtc.nvrtcDestroyProgram(&prog);
    const opts = [_][*:0]const u8{
        "--gpu-architecture=compute_70",
        "--std=c++17",
        "-I/usr/include",
        "-I/usr/local/cuda/include",
    };
    if (nvrtc.nvrtcCompileProgram(prog, opts.len, &opts) != 0) {
        var log_size: usize = 0;
        _ = nvrtc.nvrtcGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            const log = std.heap.page_allocator.alloc(u8, log_size) catch return null;
            defer std.heap.page_allocator.free(log);
            _ = nvrtc.nvrtcGetProgramLog(prog, log.ptr);
            std.log.warn("fucina-cuda: NVRTC compile failed:\n{s}", .{log[0..log_size]});
        }
        return null;
    }
    var size: usize = 0;
    if (nvrtc.nvrtcGetPTXSize(prog, &size) != 0 or size == 0) return null;
    const buf = std.heap.page_allocator.allocSentinel(u8, size - 1, 0) catch return null;
    if (nvrtc.nvrtcGetPTX(prog, buf.ptr) != 0) {
        std.heap.page_allocator.free(buf);
        return null;
    }
    return buf;
}

/// Lazy kernel-module load: embedded PTX (driver JIT, disk-cached; measured
/// ~26 ms cold measured) -> NVRTC recompile of the embedded
/// source -> quant/gemv arms disabled (cuBLAS arms unaffected). One attempt
/// per process, under `init_mutex`.
fn ensureKernels(ctx: *Ctx) ?*Kernels {
    if (kernels_ready.load(.acquire)) return &kernels_state.?;
    init_mutex.lock();
    defer init_mutex.unlock();
    if (kernels_ready.load(.monotonic)) return &kernels_state.?;
    if (kernels_tried) return null;
    kernels_tried = true;

    const d = &ctx.driver;
    if (d.cuCtxSetCurrent(ctx.context) != 0) return null;
    var module: api.CUmodule = null;
    if (!state.kernels_from_src) {
        if (d.cuModuleLoadData(&module, kernels_ptx.ptr) != 0) module = null;
    }
    if (module == null) {
        if (nvrtcCompilePtx()) |ptx| {
            defer std.heap.page_allocator.free(ptx);
            if (d.cuModuleLoadData(&module, ptx.ptr) != 0) module = null;
        }
    }
    if (module == null) {
        std.log.warn("fucina-cuda: kernel module load failed; quantized GPU arms disabled for this process", .{});
        return null;
    }
    var ks: Kernels = undefined;
    const mul_names = [_][:0]const u8{ "fucina_mul_mm_q8_0", "fucina_mul_mm_q6_K", "fucina_mul_mm_q4_K", "fucina_mul_mm_q5_K" };
    const mul_mma_names = [_][:0]const u8{ "fucina_mul_mm_mma_q8_0", "fucina_mul_mm_mma_q6_K", "fucina_mul_mm_mma_q4_K", "fucina_mul_mm_mma_q5_K" };
    const mul_mma_n32_names = [_][:0]const u8{ "fucina_mul_mm_mma_n32_q8_0", "fucina_mul_mm_mma_n32_q6_K", "fucina_mul_mm_mma_n32_q4_K", "fucina_mul_mm_mma_n32_q5_K" };
    const gemv_names = [_][:0]const u8{ "fucina_gemv_q8_0", "fucina_gemv_q6_K", "fucina_gemv_q4_K", "fucina_gemv_q5_K" };
    for (mul_names, 0..) |name, i| {
        if (d.cuModuleGetFunction(&ks.mul_mm[i], module, name.ptr) != 0) return null;
    }
    for (mul_mma_names, 0..) |name, i| {
        var f: api.CUfunction = null;
        ks.mul_mm_mma[i] = if (d.cuModuleGetFunction(&f, module, name.ptr) == 0) f else null;
    }
    for (mul_mma_n32_names, 0..) |name, i| {
        var f: api.CUfunction = null;
        ks.mul_mm_mma_n32[i] = if (d.cuModuleGetFunction(&f, module, name.ptr) == 0) f else null;
    }
    var reduce_split_k: api.CUfunction = null;
    ks.reduce_split_k = if (d.cuModuleGetFunction(&reduce_split_k, module, "fucina_reduce_split_k") == 0) reduce_split_k else null;
    for (gemv_names, 0..) |name, i| {
        if (d.cuModuleGetFunction(&ks.gemv[i], module, name.ptr) != 0) return null;
    }
    if (d.cuModuleGetFunction(&ks.attn_f16, module, "fucina_attn_f16") != 0) return null;
    const es_perturb_names = [_][:0]const u8{ "fucina_es_perturb_f16", "fucina_es_perturb_f32" };
    const es_update_names = [_][:0]const u8{ "fucina_es_update_f16", "fucina_es_update_f32" };
    const es_anchor_names = [_][:0]const u8{ "fucina_es_anchor_f16", "fucina_es_anchor_f32" };
    inline for (0..2) |i| {
        var f: api.CUfunction = null;
        ks.es_perturb[i] = if (d.cuModuleGetFunction(&f, module, es_perturb_names[i].ptr) == 0) f else null;
        ks.es_update[i] = if (d.cuModuleGetFunction(&f, module, es_update_names[i].ptr) == 0) f else null;
        ks.es_anchor[i] = if (d.cuModuleGetFunction(&f, module, es_anchor_names[i].ptr) == 0) f else null;
    }
    kernels_state = ks;
    kernels_ready.store(true, .release);
    return &kernels_state.?;
}

const QuantMulMmLaunch = struct {
    kernel: api.CUfunction,
    n_tile: usize,
    block_y: c_uint,
    split_k: c_uint = 1,
};

fn quantMulMmLaunch(ctx: *const Ctx, kernels: *const Kernels, format: QFormat, grid_x: usize, n: usize, k: usize, allow_split_k: bool) QuantMulMmLaunch {
    const index: usize = @intCast(@intFromEnum(format));
    if (state.quant_mma and ctx.compute_major >= 7) {
        if (ctx.sm_count > 0) {
            const n64_tiles = n / 64 + @intFromBool(n % 64 != 0);
            const n64_blocks = std.math.mul(usize, grid_x, n64_tiles) catch std.math.maxInt(usize);
            const sm_count: usize = @intCast(ctx.sm_count);
            // A graphless adaptation of the Stream-K occupancy principle used
            // by ik_llama.cpp's CUDA MMQ (ggml-cuda/mmq.cuh, audited at
            // b90939934add9ba4fbb37e8c6470809a70b78f0a): when an eager N64
            // output grid cannot occupy every SM, partition K and queue a
            // deterministic reduction on the same persistent stream. Keeping
            // the split small beats a fixed nSM launch for Fucina's short,
            // already-tiled calls and bounds the reusable partial allocation.
            // Do not double the grid merely to fill the last few SMs: the
            // reduction then costs more than the small occupancy deficit. The
            // 7/8 cutoff still admits the measured 64/76 Qwen grid.
            const split_block_ceiling = sm_count - sm_count / 8;
            if (allow_split_k and state.quant_split_k and kernels.reduce_split_k != null and n64_blocks < split_block_ceiling) {
                const desired_split = sm_count / n64_blocks + @intFromBool(sm_count % n64_blocks != 0);
                // On the 76-SM Ada reference host, three Q6 partitions fill 72
                // slots for the 24-tile Parakeet shape; a fourth wave only adds
                // reduction cost. Q4/Q8 reach their optimum with two.
                const split_cap: usize = switch (format) {
                    .q6_k => 3,
                    .q4_k, .q5_k, .q8_0 => 2,
                };
                const max_split = @min(split_cap, @max(@as(usize, 1), k / 128));
                const split_k = @min(desired_split, max_split);
                if (split_k > 1) {
                    if (kernels.mul_mm_mma[index]) |kernel| return .{
                        .kernel = kernel,
                        .n_tile = 64,
                        .block_y = 16,
                        .split_k = @intCast(split_k),
                    };
                }
            }
            const weighted_blocks = std.math.mul(usize, n64_blocks, 3) catch std.math.maxInt(usize);
            const sm_target = std.math.mul(usize, sm_count, 2) catch std.math.maxInt(usize);
            if (weighted_blocks < sm_target) {
                if (kernels.mul_mm_mma_n32[index]) |kernel| return .{ .kernel = kernel, .n_tile = 32, .block_y = 8 };
            }
        }
        if (kernels.mul_mm_mma[index]) |kernel| return .{ .kernel = kernel, .n_tile = 64, .block_y = 16 };
    }
    return .{ .kernel = kernels.mul_mm[index], .n_tile = 64, .block_y = 16 };
}

fn quantDecodeUsesGemv(format: QFormat, m: usize) bool {
    // Q5_K switches to Fucina's lane-packed CPU kernel at m=4. Keep its
    // warp-per-row CUDA path on the compact-decode rows (m<4); the tiled MMA
    // path is the relevant GPU contender for batch rows 4..8.
    return m <= 8 and (format != .q5_k or m < 4);
}

/// Eager dense/shared-input quantized NT GEMM. Stable model weights resolve to
/// one managed resident allocation; input/tile uploads feed the persistent
/// compute stream and host output is deferred through the standard Work. For
/// batch_count > 1 the same input is consumed by one launch per weight matrix
/// without materializing repeated activation rows.
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
    if (!rhs_cacheable or rhs_bytes.len == 0 or batch_count == 0 or m == 0 or n == 0 or k == 0) return false;
    if (m > std.math.maxInt(i32) or n > std.math.maxInt(i32) or k > std.math.maxInt(i32) or batch_count > std.math.maxInt(i32)) return false;
    if (k % 32 != 0 or k % format.kMultiple() != 0 or n % 4 != 0) return false;
    if (m <= 8 and batch_count != 1) return false;
    const input_elems = std.math.mul(usize, m, k) catch return false;
    const output_rows = std.math.mul(usize, batch_count, m) catch return false;
    const output_elems = std.math.mul(usize, output_rows, n) catch return false;
    if (input.offset + input_elems > input.buffer.data.len or out.offset + output_elems > out.buffer.data.len) return false;
    if (out.buffer.pending() != null) return false;

    const ctx = context() orelse return false;
    dispatch_lock.lock();
    defer dispatch_lock.unlock();
    const d = &ctx.driver;
    if (d.cuCtxSetCurrent(ctx.context) != 0) return false;
    const ks = ensureKernels(ctx) orelse return false;
    const slot = acquireAsyncSlot(ctx) orelse return false;
    const holder = std.heap.c_allocator.create(CudaWork) catch {
        releaseAsyncSlot(slot);
        return false;
    };
    var dep_input: ?*accelerator.Work = null;
    var copy_queued = false;
    var success = false;
    defer if (!success) {
        _ = d.cuStreamSynchronize(ctx.upload_stream);
        _ = d.cuStreamSynchronize(ctx.stream);
        if (dep_input) |dep| dep.release();
        releaseAsyncSlot(slot);
        std.heap.c_allocator.destroy(holder);
    };

    const submit_started = tstart();
    if (!slot.c_dev.ensure(d, output_elems * @sizeOf(f32))) return false;
    const output_registered = ensureHostRegistered(ctx, out.buffer);
    if (!output_registered and !slot.c_host.ensure(d, output_elems * @sizeOf(f32))) return false;
    const input_dev = pendingDeviceInput(input, &dep_input) orelse stageInput(ctx, input, input_elems, &slot.a_dev, &copy_queued) orelse return false;
    const rhs_dev = residentDevPtr(ctx, rhs_bytes, true) orelse return false;
    if (trace_on) tinc(&trace.rhs_resident, 1);

    const use_gemv = quantDecodeUsesGemv(format, m);
    if (use_gemv) {
        if (copy_queued and (d.cuEventRecord(slot.inputs_ready, ctx.upload_stream) != 0 or
            d.cuStreamWaitEvent(ctx.stream, slot.inputs_ready, 0) != 0)) return false;
        var p_src0 = rhs_dev;
        var p_x = input_dev;
        var p_y = slot.c_dev.ptr;
        var p_ne00: c_int = @intCast(k);
        var p_ne01: c_int = @intCast(n);
        var p_m: c_int = @intCast(m);
        var p_nb01: u64 = @intCast(nb01);
        var params = [_]?*anyopaque{
            @ptrCast(&p_src0), @ptrCast(&p_x),    @ptrCast(&p_y),
            @ptrCast(&p_ne00), @ptrCast(&p_ne01), @ptrCast(&p_m),
            @ptrCast(&p_nb01),
        };
        const warps_per_block = 4;
        const grid: c_uint = @intCast((n + warps_per_block - 1) / warps_per_block);
        const f = ks.gemv[@intCast(@intFromEnum(format))];
        if (d.cuLaunchKernel(f, grid, 1, 1, 32 * warps_per_block, 1, 1, 0, ctx.stream, &params, null) != 0) return false;
    } else {
        const tiles_per_batch = (m + 31) / 32;
        const total_tiles = std.math.mul(usize, batch_count, tiles_per_batch) catch return false;
        const tiles_bytes = std.math.mul(usize, total_tiles, @sizeOf(QMMTile)) catch return false;
        if (!slot.aux_host.ensure(d, tiles_bytes) or !slot.aux_dev.ensure(d, tiles_bytes)) return false;
        const tiles: [*]QMMTile = @ptrCast(@alignCast(slot.aux_host.ptr.?));
        for (0..batch_count) |bi| {
            for (0..tiles_per_batch) |ti| {
                tiles[bi * tiles_per_batch + ti] = .{
                    .expert = @intCast(bi),
                    .base_row = 0,
                    .m = @intCast(m),
                    .tile_m = @intCast(ti),
                };
            }
        }
        if (d.cuMemcpyHtoDAsync(slot.aux_dev.ptr, tiles, tiles_bytes, ctx.upload_stream) != 0) return false;
        copy_queued = true;
        if (d.cuEventRecord(slot.inputs_ready, ctx.upload_stream) != 0 or
            d.cuStreamWaitEvent(ctx.stream, slot.inputs_ready, 0) != 0) return false;

        // Staged/resident allocations are normally 16-byte aligned. A pending
        // producer view may carry an arbitrary element offset; keep it on the
        // scalar CUDA kernel rather than adding an alignment branch to every
        // vector load in the hot WMMA K loop.
        const launch = if (input_dev & 15 == 0)
            quantMulMmLaunch(ctx, ks, format, tiles_per_batch, n, k, true)
        else
            QuantMulMmLaunch{
                .kernel = ks.mul_mm[@intCast(@intFromEnum(format))],
                .n_tile = 64,
                .block_y = 16,
            };
        const grid_y: c_uint = @intCast((n + launch.n_tile - 1) / launch.n_tile);
        const output_elems_per_batch = std.math.mul(usize, m, n) catch return false;
        const partial_elems_per_batch = std.math.mul(usize, output_elems_per_batch, @intCast(launch.split_k)) catch return false;
        if (launch.split_k > 1) {
            const partial_elems = std.math.mul(usize, batch_count, partial_elems_per_batch) catch return false;
            const partial_bytes = std.math.mul(usize, partial_elems, @sizeOf(f32)) catch return false;
            if (!slot.partial_dev.ensure(d, partial_bytes)) return false;
        }
        for (0..batch_count) |bi| {
            var p_src0 = rhs_dev;
            var p_src1 = input_dev;
            var p_tiles = slot.aux_dev.ptr + bi * tiles_per_batch * @sizeOf(QMMTile);
            var p_dst = slot.c_dev.ptr + bi * m * n * @sizeOf(f32);
            var p_partial: api.CUdeviceptr = if (launch.split_k > 1)
                slot.partial_dev.ptr + bi * partial_elems_per_batch * @sizeOf(f32)
            else
                0;
            var p_ne00: c_int = @intCast(k);
            var p_ne01: c_int = @intCast(n);
            var p_nb01: u64 = @intCast(nb01);
            var p_nb02: u64 = @intCast(nb02);
            var p_split_k: c_int = @intCast(launch.split_k);
            var p_partial_stride: u64 = @intCast(output_elems_per_batch);
            var params = [_]?*anyopaque{
                @ptrCast(&p_src0),    @ptrCast(&p_src1),    @ptrCast(&p_tiles),          @ptrCast(&p_dst),
                @ptrCast(&p_ne00),    @ptrCast(&p_ne01),    @ptrCast(&p_nb01),           @ptrCast(&p_nb02),
                @ptrCast(&p_partial), @ptrCast(&p_split_k), @ptrCast(&p_partial_stride),
            };
            if (d.cuLaunchKernel(launch.kernel, @intCast(tiles_per_batch), grid_y, launch.split_k, 16, launch.block_y, 1, 0, ctx.stream, &params, null) != 0) return false;
            if (launch.split_k > 1) {
                var p_elements: u64 = @intCast(output_elems_per_batch);
                var reduce_params = [_]?*anyopaque{
                    @ptrCast(&p_partial), @ptrCast(&p_dst), @ptrCast(&p_elements), @ptrCast(&p_split_k),
                };
                const reduce_grid: c_uint = @intCast((output_elems_per_batch + 1023) / 1024);
                if (d.cuLaunchKernel(ks.reduce_split_k.?, reduce_grid, 1, 1, 256, 1, 1, 0, ctx.stream, &reduce_params, null) != 0) return false;
            }
        }
        if (trace_on) tinc(&trace.h2d_bytes, tiles_bytes);
    }
    if (d.cuEventRecord(slot.done, ctx.stream) != 0) return false;

    input.buffer.retain();
    holder.* = .{
        .work = accelerator.Work.init(.cuda, holder, &CudaWork.vtable),
        .ctx = ctx,
        .slot = slot,
        .output = out.buffer.data[out.offset..].ptr,
        .total_c = output_elems,
        .block_c = output_elems,
        .stride_c = output_elems,
        .batch_count = 1,
        .device_base = @as(usize, @intCast(slot.c_dev.ptr)) - out.offset * @sizeOf(f32),
        .dep_a = dep_input,
        .dep_b = null,
        .a_buffer = input.buffer,
        .b_buffer = null,
        .output_registered = output_registered,
        .kind = .quant,
    };
    input.buffer.setPendingUse(&holder.work);
    out.buffer.setPending(&holder.work);
    success = true;
    if (trace_on) {
        tinc(&trace.quant_async_calls, 1);
        tinc(&trace.quant_submit_ns, tfinish(submit_started));
    }
    return true;
}

/// Process-global serialization for the quantized staging-panel protocol.
/// Hold across `qmoeStage` + CPU panel writes + `gemmQGroupedNt` dispatches +
/// readback — the provider owns one grow-only pinned panel pair and their
/// device twins. Same eager BLAS-like contract as the Metal provider.
pub var qmoe_lock: thread.Mutex = .{};

pub const QMoeStage = struct {
    in: [*]f32,
    out: [*]f32,
};

// Panels: pinned host (the CPU gather/GeGLU/scatter targets — llm-tier code
// is unchanged) + device twins; `gemmQGroupedNt` crosses PCIe once per
// direction per dispatch. Sizes recorded by the last `qmoeStage` call.
var qmoe_in_host: ?[*]f32 = null;
var qmoe_in_cap: usize = 0;
var qmoe_out_host: ?[*]f32 = null;
var qmoe_out_cap: usize = 0;
var qmoe_in_bytes: usize = 0;
var qmoe_out_bytes: usize = 0;
var qmoe_in_dev: DeviceBuf = .{};
var qmoe_out_dev: DeviceBuf = .{};
var qmoe_rhs_dev: DeviceBuf = .{};
var qmoe_tiles_dev: DeviceBuf = .{};
var qmoe_inputs_ready: api.CUevent = null;
var qmoe_done: api.CUevent = null;

fn ensurePinned(d: *const api.Driver, ptr: *?[*]f32, cap: *usize, bytes: usize) bool {
    if (cap.* >= bytes) return true;
    if (ptr.*) |p| _ = d.cuMemFreeHost(@ptrCast(p));
    ptr.* = null;
    cap.* = 0;
    var raw: ?*anyopaque = null;
    if (d.cuMemHostAlloc(&raw, bytes, 0) != 0) return false;
    ptr.* = @ptrCast(@alignCast(raw.?));
    cap.* = bytes;
    return true;
}

/// Acquire the staging panels (grow-only pinned host memory backed by device
/// twins): `in` for the activation rows the CPU gathers, `out` for the GEMM
/// results. Pointers stay valid until the next `qmoeStage` call — hold
/// `qmoe_lock` across the whole stage/dispatch/readback sequence. Null when
/// the GPU or the kernel module is unavailable.
pub fn qmoeStage(in_bytes: usize, out_bytes: usize) ?QMoeStage {
    const ctx = context() orelse return null;
    if (ensureKernels(ctx) == null) return null;
    const d = &ctx.driver;
    if (d.cuCtxSetCurrent(ctx.context) != 0) return null;
    if (!ensurePinned(d, &qmoe_in_host, &qmoe_in_cap, in_bytes)) return null;
    if (!ensurePinned(d, &qmoe_out_host, &qmoe_out_cap, out_bytes)) return null;
    if (!qmoe_in_dev.ensure(d, in_bytes) or !qmoe_out_dev.ensure(d, out_bytes)) return null;
    if (qmoe_inputs_ready == null and d.cuEventCreate(&qmoe_inputs_ready, 2) != 0) return null;
    if (qmoe_done == null and d.cuEventCreate(&qmoe_done, 2) != 0) return null;
    qmoe_in_bytes = in_bytes;
    qmoe_out_bytes = out_bytes;
    return .{ .in = qmoe_in_host.?, .out = qmoe_out_host.? };
}

/// Grouped NT GEMM over the staged panels — the same CPU-built tile-table
/// protocol as the Metal provider: for every tile, panel rows
/// `[base_row, base_row+m)` of expert `expert` produce
/// `out[row, 0..n_out) = in[row, 0..k) · dequant(W[expert])ᵀ`.
/// `rhs_bytes` = raw quantized blocks, row-major `[n_out, k]` per expert with
/// byte strides `nb01`/`nb02`. Caller holds `qmoe_lock`. A resident RHS
/// (managed registry hit) dispatches with zero weight transfer; a transient
/// RHS streams. Returns false when the GPU didn't run.
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
    if (n_out == 0 or n_out % 4 != 0) return false;
    if (tiles.len == 0 or tiles.len > 1 << 24) return false;
    if (n_out > std.math.maxInt(i32) or k > std.math.maxInt(i32)) return false;
    const ctx = context() orelse return false;
    const ks = ensureKernels(ctx) orelse return false;
    const d = &ctx.driver;
    if (d.cuCtxSetCurrent(ctx.context) != 0) return false;
    if (qmoe_inputs_ready == null or qmoe_done == null) return false;
    var queued = false;
    var success = false;
    defer if (!success and queued) {
        _ = d.cuStreamSynchronize(ctx.upload_stream);
        _ = d.cuStreamSynchronize(ctx.stream);
        _ = d.cuStreamSynchronize(ctx.transfer_stream);
    };

    const timer = tstart();
    // Stable (cacheable) RHS is adopted into the managed registry on first
    // use — the Metal wrap-cache analog — so mmap'd model weights cross PCIe
    // once per process, not once per dispatch. Transient RHS streams.
    const rhs_dev: api.CUdeviceptr = if (residentDevPtr(ctx, rhs_bytes, rhs_cacheable)) |dev| blk: {
        if (trace_on) tinc(&trace.rhs_resident, 1);
        break :blk dev;
    } else blk: {
        if (trace_on) tinc(&trace.rhs_streamed, 1);
        if (!qmoe_rhs_dev.ensure(d, rhs_bytes.len)) return false;
        if (d.cuMemcpyHtoDAsync(qmoe_rhs_dev.ptr, rhs_bytes.ptr, rhs_bytes.len, ctx.upload_stream) != 0) return false;
        queued = true;
        if (trace_on) tinc(&trace.h2d_bytes, rhs_bytes.len);
        break :blk qmoe_rhs_dev.ptr;
    };

    const tiles_bytes = tiles.len * @sizeOf(QMMTile);
    if (!qmoe_tiles_dev.ensure(d, tiles_bytes)) return false;
    if (d.cuMemcpyHtoDAsync(qmoe_tiles_dev.ptr, tiles.ptr, tiles_bytes, ctx.upload_stream) != 0 or
        d.cuMemcpyHtoDAsync(qmoe_in_dev.ptr, qmoe_in_host.?, qmoe_in_bytes, ctx.upload_stream) != 0) return false;
    queued = true;
    if (d.cuEventRecord(qmoe_inputs_ready, ctx.upload_stream) != 0 or
        d.cuStreamWaitEvent(ctx.stream, qmoe_inputs_ready, 0) != 0) return false;

    var p_src0 = rhs_dev;
    var p_src1 = qmoe_in_dev.ptr;
    var p_tiles = qmoe_tiles_dev.ptr;
    var p_dst = qmoe_out_dev.ptr;
    var p_ne00: c_int = @intCast(k);
    var p_ne01: c_int = @intCast(n_out);
    var p_nb01: u64 = @intCast(nb01);
    var p_nb02: u64 = @intCast(nb02);
    const launch = quantMulMmLaunch(ctx, ks, format, tiles.len, n_out, k, false);
    const grid_y: c_uint = @intCast((n_out + launch.n_tile - 1) / launch.n_tile);
    var p_partial: api.CUdeviceptr = 0;
    var p_split_k: c_int = 1;
    var p_partial_stride: u64 = 0;
    var launch_params = [_]?*anyopaque{
        @ptrCast(&p_src0),    @ptrCast(&p_src1),    @ptrCast(&p_tiles),          @ptrCast(&p_dst),
        @ptrCast(&p_ne00),    @ptrCast(&p_ne01),    @ptrCast(&p_nb01),           @ptrCast(&p_nb02),
        @ptrCast(&p_partial), @ptrCast(&p_split_k), @ptrCast(&p_partial_stride),
    };
    if (d.cuLaunchKernel(launch.kernel, @intCast(tiles.len), grid_y, 1, 16, launch.block_y, 1, 0, ctx.stream, &launch_params, null) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (d.cuEventRecord(qmoe_done, ctx.stream) != 0 or
        d.cuStreamWaitEvent(ctx.transfer_stream, qmoe_done, 0) != 0 or
        d.cuMemcpyDtoHAsync(qmoe_out_host.?, qmoe_out_dev.ptr, qmoe_out_bytes, ctx.transfer_stream) != 0 or
        d.cuStreamSynchronize(ctx.transfer_stream) != 0)
    {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (trace_on) {
        tinc(&trace.quant_calls, 1);
        telapsed(&trace.quant_ns, timer);
        tinc(&trace.h2d_bytes, qmoe_in_bytes + tiles_bytes);
        tinc(&trace.d2h_bytes, qmoe_out_bytes);
    }
    success = true;
    return true;
}

/// Decode arm (FUCINA_GPU_DECODE=1): warp-per-row dequant-dot GEMV for the
/// rows selected by `quantDecodeUsesGemv`. **Resident-or-adoptable weights
/// only**: at decode shapes
/// the op is bytes-bound, so streaming the weights over PCIe per token is a
/// strict loss vs the CPU int8 kernels — a registry miss on transient RHS
/// refuses and the caller stays on CPU. f32 dequant (no f16 rounding),
/// same 5e-3 quant tier. Takes `dispatch_lock` (shares the f32 staging bufs).
fn gemvQuant(
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
    const ctx = context() orelse return false;
    const ks = ensureKernels(ctx) orelse return false;
    const d = &ctx.driver;

    dispatch_lock.lock();
    defer dispatch_lock.unlock();
    if (d.cuCtxSetCurrent(ctx.context) != 0) return false;

    const rhs_dev = residentDevPtr(ctx, rhs_bytes, rhs_cacheable) orelse return false;
    if (trace_on) tinc(&trace.rhs_resident, 1);

    const in_bytes = m * k * 4;
    const out_bytes = m * n * 4;
    if (!dev_a.ensure(d, in_bytes) or !dev_c.ensure(d, out_bytes)) return false;
    if (d.cuMemcpyHtoD(dev_a.ptr, a.ptr, in_bytes) != 0) return false;

    var p_src0 = rhs_dev;
    var p_x = dev_a.ptr;
    var p_y = dev_c.ptr;
    var p_ne00: c_int = @intCast(k);
    var p_ne01: c_int = @intCast(n);
    var p_m: c_int = @intCast(m);
    var p_nb01: u64 = @intCast(nb01);
    var params = [_]?*anyopaque{
        @ptrCast(&p_src0), @ptrCast(&p_x),    @ptrCast(&p_y),
        @ptrCast(&p_ne00), @ptrCast(&p_ne01), @ptrCast(&p_m),
        @ptrCast(&p_nb01),
    };
    const warps_per_block = 4;
    const grid: c_uint = @intCast((n + warps_per_block - 1) / warps_per_block);
    const f = ks.gemv[@intCast(@intFromEnum(format))];
    if (d.cuLaunchKernel(f, grid, 1, 1, 32 * warps_per_block, 1, 1, 0, ctx.stream, &params, null) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (d.cuStreamSynchronize(ctx.stream) != 0) return false;
    if (d.cuMemcpyDtoH(c.ptr, dev_c.ptr, out_bytes) != 0) return false;
    if (trace_on) tinc(&trace.gemv_calls, 1);
    return true;
}

/// Single quantized NT GEMM with host-memory operands, staged through the
/// shared panels: `c[m,n] = a[m,k] · dequant(W)ᵀ` (one "expert"). Same
/// wrapper shape as the Metal provider; takes `qmoe_lock` itself. With
/// FUCINA_GPU_DECODE=1 it takes the selected decode route instead.
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
    if (k == 0 or k > std.math.maxInt(i32) or k % 32 != 0 or k % format.kMultiple() != 0) return false;
    const in_elems = std.math.mul(usize, m, k) catch return false;
    const out_elems = std.math.mul(usize, m, n) catch return false;
    if (a.len < in_elems or c.len < out_elems) return false;

    if (state.decode_enabled and quantDecodeUsesGemv(format, m) and n % 4 == 0 and n <= std.math.maxInt(i32)) {
        return gemvQuant(format, rhs_bytes, rhs_cacheable, nb01, a, c, m, n, k);
    }

    const in_bytes = std.math.mul(usize, in_elems, @sizeOf(f32)) catch return false;
    const out_bytes = std.math.mul(usize, out_elems, @sizeOf(f32)) catch return false;

    qmoe_lock.lock();
    defer qmoe_lock.unlock();
    const stage = qmoeStage(in_bytes, out_bytes) orelse return false;
    @memcpy(stage.in[0..in_elems], a[0..in_elems]);
    var tiles_buf: [64]QMMTile = undefined;
    const n_tiles = (m + 31) / 32;
    if (n_tiles > tiles_buf.len) return false;
    for (0..n_tiles) |t| {
        tiles_buf[t] = .{ .expert = 0, .base_row = 0, .m = @intCast(m), .tile_m = @intCast(t) };
    }
    if (!gemmQGroupedNt(format, rhs_bytes, rhs_cacheable, nb01, 0, n, k, tiles_buf[0..n_tiles])) return false;
    @memcpy(c[0..out_elems], stage.out[0..out_elems]);
    return true;
}

/// Batched dense quantized NT GEMM over one shared activation matrix:
/// for each batch `b`, `c[b,m,n] = a[m,k] · dequant(W[b,n,k])ᵀ` — the narrow
/// eager command-batching seam, one launch via the expert dimension. Same
/// wrapper shape as the Metal provider.
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

    qmoe_lock.lock();
    defer qmoe_lock.unlock();

    const row_bytes = std.math.mul(usize, k, @sizeOf(f32)) catch return false;
    const in_bytes = std.math.mul(usize, rows_total, row_bytes) catch return false;
    const out_bytes = std.math.mul(usize, out_elems, @sizeOf(f32)) catch return false;
    const stage = qmoeStage(in_bytes, out_bytes) orelse return false;
    for (0..batch_count) |bi| {
        @memcpy(stage.in[bi * in_elems ..][0..in_elems], a[0..in_elems]);
    }

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
    @memcpy(c[0..out_elems], stage.out[0..out_elems]);
    return true;
}

pub fn shouldUseGpuQMoe(total_work: u64) bool {
    ensureConfig();
    const pass = state.gpu_enabled and total_work >= state.min_work_qmoe;
    tgate(pass);
    return pass;
}

/// Occupancy arm of the grouped-MoE gate; arithmetic mirrors the Metal
/// provider so gate behavior is provider-independent once the arm lands.
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
        .q5_k => state.min_work_packed_q5,
        .q4_k, .q8_0 => state.min_work_qmoe,
    };
    const pass = state.gpu_enabled and total_work >= min_work;
    tgate(pass);
    return pass;
}

/// Dense model-weight gate against Fucina's load-time-packed CPU fallback.
pub fn shouldUseGpuDenseQuantPacked(format: QFormat, total_work: u64) bool {
    ensureConfig();
    const min_work = switch (format) {
        .q4_k => state.min_work_packed_q4,
        .q5_k => state.min_work_packed_q5,
        .q6_k => state.min_work_packed_q6,
        .q8_0 => state.min_work_packed_q8,
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

/// Decode-GEMV capability gate for exec's m <= 8 arm: FUCINA_GPU_DECODE=1
/// opts in, default off pending a sampled-token parity-oracle pass.
pub fn decodeGemvEnabled() bool {
    ensureConfig();
    return state.decode_enabled;
}

pub fn shouldUseGpuQuantDecode(format: QFormat, m: usize, n: usize, k: usize) bool {
    if (!decodeGemvEnabled()) return false;
    if (format != .q5_k) return true;
    const work = std.math.mul(u64, std.math.mul(u64, m, n) catch std.math.maxInt(u64), k) catch std.math.maxInt(u64);
    const pass = work >= state.min_work_decode_q5;
    tgate(pass);
    return pass;
}

/// Test seam: parity tests pin the decode arm off so their m <= 8 cases
/// exercise the staged panel path regardless of FUCINA_GPU_DECODE.
pub fn setDecodeForTest(v: bool) void {
    ensureConfig();
    state.decode_enabled = v;
}

fn setQuantMmaForTest(v: bool) void {
    ensureConfig();
    state.quant_mma = v;
}

fn setQuantSplitKForTest(v: bool) void {
    ensureConfig();
    state.quant_split_k = v;
}

// --- Prefill attention offload ------------------------------------------------

var attn_q_dev: DeviceBuf = .{};
var attn_k_dev: DeviceBuf = .{};
var attn_v_dev: DeviceBuf = .{};
var attn_o_dev: DeviceBuf = .{};
var attn_map_dev: DeviceBuf = .{};

/// Gate for the fused prefill-attention offload: score work q·kv·heads·d
/// against FUCINA_GPU_MIN_WORK_ATTN. Attention offload needs no residency —
/// arithmetic intensity grows with q_seq (FLOPs O(q·kv·d), bytes O(q·d)),
/// so streaming Q/K/V per
/// call already pays at prefill shapes; decode never reaches this seam.
pub fn shouldUseGpuAttn(q_seq: usize, kv_seq: usize, heads: usize, d: usize) bool {
    ensureConfig();
    const work = std.math.mul(u64, std.math.mul(u64, q_seq, kv_seq) catch std.math.maxInt(u64), std.math.mul(u64, heads, d) catch std.math.maxInt(u64)) catch std.math.maxInt(u64);
    const pass = state.gpu_enabled and work >= state.min_work_attn;
    tgate(pass);
    return pass;
}

/// Fused online-softmax grouped attention over f16 KV — the CPU tiled
/// kernel's exact semantics (query row i at absolute position
/// source_offset + i; sliding window pre-clamped by the caller; causal or
/// bidirectional; per-head kv mapping). Stateless and blocking: Q/K/V stream
/// in, the output streams back, `false` -> untouched CPU path. `d <= 256`,
/// `heads` bounded by the caller's map buffer.
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
    if (q_seq == 0 or kv_seq == 0 or heads == 0 or kv_heads == 0 or d == 0 or d > 256) return false;
    if (q_seq > std.math.maxInt(i32) or kv_seq > std.math.maxInt(i32) or heads > std.math.maxInt(i32) or
        kv_heads > std.math.maxInt(i32) or d > std.math.maxInt(i32) or source_offset > std.math.maxInt(i32) or
        window > std.math.maxInt(i32)) return false;
    if (kv_head_for_head.len < heads) return false;
    const q_elems = std.math.mul(usize, std.math.mul(usize, q_seq, heads) catch return false, d) catch return false;
    const kv_elems = std.math.mul(usize, std.math.mul(usize, kv_seq, kv_heads) catch return false, d) catch return false;
    if (q.len < q_elems or out.len < q_elems or k.len < kv_elems or v.len < kv_elems) return false;

    const ctx = context() orelse return false;
    const ks = ensureKernels(ctx) orelse return false;
    const d_ = &ctx.driver;

    const timer = tstart();
    dispatch_lock.lock();
    defer dispatch_lock.unlock();
    if (d_.cuCtxSetCurrent(ctx.context) != 0) return false;

    if (!attn_q_dev.ensure(d_, q_elems * 4) or !attn_o_dev.ensure(d_, q_elems * 4) or
        !attn_k_dev.ensure(d_, kv_elems * 2) or !attn_v_dev.ensure(d_, kv_elems * 2) or
        !attn_map_dev.ensure(d_, heads * 4))
    {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (d_.cuMemcpyHtoD(attn_q_dev.ptr, q.ptr, q_elems * 4) != 0 or
        d_.cuMemcpyHtoD(attn_k_dev.ptr, k.ptr, kv_elems * 2) != 0 or
        d_.cuMemcpyHtoD(attn_v_dev.ptr, v.ptr, kv_elems * 2) != 0 or
        d_.cuMemcpyHtoD(attn_map_dev.ptr, kv_head_for_head.ptr, heads * 4) != 0)
    {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }

    var p_q = attn_q_dev.ptr;
    var p_k = attn_k_dev.ptr;
    var p_v = attn_v_dev.ptr;
    var p_o = attn_o_dev.ptr;
    var p_map = attn_map_dev.ptr;
    var p_qs: c_int = @intCast(q_seq);
    var p_ks: c_int = @intCast(kv_seq);
    var p_h: c_int = @intCast(heads);
    var p_kh: c_int = @intCast(kv_heads);
    var p_d: c_int = @intCast(d);
    var p_off: c_int = @intCast(source_offset);
    var p_scale: f32 = scale;
    var p_win: c_int = @intCast(window);
    var p_causal: c_int = @intFromBool(causal);
    var params = [_]?*anyopaque{
        @ptrCast(&p_q),   @ptrCast(&p_k),     @ptrCast(&p_v),   @ptrCast(&p_o),      @ptrCast(&p_map),
        @ptrCast(&p_qs),  @ptrCast(&p_ks),    @ptrCast(&p_h),   @ptrCast(&p_kh),     @ptrCast(&p_d),
        @ptrCast(&p_off), @ptrCast(&p_scale), @ptrCast(&p_win), @ptrCast(&p_causal),
    };
    const warps_per_block = 4;
    const grid_x: c_uint = @intCast((q_seq + warps_per_block - 1) / warps_per_block);
    if (d_.cuLaunchKernel(ks.attn_f16, grid_x, @intCast(heads), 1, 32 * warps_per_block, 1, 1, 0, ctx.stream, &params, null) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (d_.cuStreamSynchronize(ctx.stream) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (d_.cuMemcpyDtoH(out.ptr, attn_o_dev.ptr, q_elems * 4) != 0) {
        if (trace_on) tinc(&trace.cuda_err, 1);
        return false;
    }
    if (trace_on) {
        tinc(&trace.attn_calls, 1);
        telapsed(&trace.attn_ns, timer);
        tinc(&trace.h2d_bytes, q_elems * 4 + kv_elems * 4 + heads * 4);
        tinc(&trace.d2h_bytes, q_elems * 4);
    }
    return true;
}

/// Test seam: parity tests use small shapes that the transient-RHS floor
/// would otherwise refuse.
pub fn setTransientFloorForTest(min_work: u64, min_m: usize) void {
    ensureConfig();
    state.min_work_transient = min_work;
    state.transient_min_m = min_m;
}

// ---------------------------------------------------------------------------
// Tests (compiled and run only on -Dgpu=cuda builds; skip without a device)
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

test "cuda gemm f32 parity vs reference (all orientations, edge tiles)" {
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
                // Strict FP32 cuBLAS (default math mode): same tolerance as
                // the Metal f32 parity test.
                const tol = @max(2e-5 * @max(@abs(want), @abs(got)), 2e-5);
                try std.testing.expect(@abs(got - want) <= tol);
            }
        }
    }
}

test "cuda gemm f32 batched matches per-matrix reference" {
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

test "cuda gate applies the transient floor above min_work" {
    if (comptime !enabled) return error.SkipZigTest;
    ensureConfig();
    const saved_work = state.min_work_transient;
    const saved_m = state.transient_min_m;
    defer setTransientFloorForTest(saved_work, saved_m);

    // 512·2048·2048 = 2^31: above min_work (2^30), below the default
    // transient floor (2^33) — refused until residency lands.
    setTransientFloorForTest(default_min_work_transient, default_transient_min_m);
    try std.testing.expect(!shouldUseGpu(512, 2048, 2048));
    // m below transient_min_m is refused even at huge work.
    try std.testing.expect(!shouldUseGpu(64, 1 << 14, 1 << 14));
    // With the floor lowered to the plain gate, the same shape passes
    // (when the kill switch hasn't disabled the GPU in this environment).
    setTransientFloorForTest(0, 0);
    try std.testing.expect(shouldUseGpu(512, 2048, 2048) == state.gpu_enabled);
}

test "cuda Q5_K decode gate preserves the compact CPU crossover" {
    if (comptime !enabled) return error.SkipZigTest;
    ensureConfig();
    const saved_decode = state.decode_enabled;
    const saved_q5 = state.min_work_decode_q5;
    defer {
        state.decode_enabled = saved_decode;
        state.min_work_decode_q5 = saved_q5;
    }

    state.decode_enabled = true;
    state.min_work_decode_q5 = default_min_work_decode_q5;
    try std.testing.expect(!shouldUseGpuQuantDecode(.q5_k, 1, 4096, 4096));
    try std.testing.expect(shouldUseGpuQuantDecode(.q5_k, 1, 6144, 4096));
    try std.testing.expect(shouldUseGpuQuantDecode(.q5_k, 2, 4096, 4096));
    // Existing formats retain the global opt-in behavior unchanged.
    try std.testing.expect(shouldUseGpuQuantDecode(.q4_k, 1, 256, 256));
    state.decode_enabled = false;
    try std.testing.expect(!shouldUseGpuQuantDecode(.q5_k, 8, 16384, 16384));
}

test "cuda Q5_K decode selects GEMV below row four and tiled GEMM above" {
    if (comptime !enabled) return error.SkipZigTest;
    try std.testing.expect(quantDecodeUsesGemv(.q5_k, 3));
    try std.testing.expect(!quantDecodeUsesGemv(.q5_k, 4));
    try std.testing.expect(!quantDecodeUsesGemv(.q5_k, 8));
    try std.testing.expect(quantDecodeUsesGemv(.q4_k, 8));
}

test "cuda gemm f16 NT parity vs f64 reference (f16-rounded output)" {
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
        const staging = gemmF16Nt(a, b, m, n, k, false) orelse return error.TestUnexpectedResult;
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f64 = 0;
                for (0..k) |p| acc += @as(f64, a[i * k + p]) * @as(f64, b[j * k + p]);
                const got: f32 = staging[i * n + j];
                const want: f32 = @floatCast(acc);
                // f32 accumulate, f16-rounded store — same tier as the Metal
                // f16 parity test.
                const tol = @max(2e-3 * @max(@abs(want), @abs(got)), 2e-3);
                try std.testing.expect(@abs(got - want) <= tol);
            }
        }
    }
}

test "cuda resident bytes: CPU-readable roundtrip + zero-copy RHS dispatch" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    const n: usize = 64;
    const k: usize = 48;
    const m: usize = 40;
    const bytes = allocResidentBytes(n * k * @sizeOf(f32)) orelse return error.SkipZigTest; // no managed access on this platform
    defer freeResidentBytes(bytes);

    // The Metal residency contract: loaders memcpy payloads in, CPU reads the
    // same bytes back.
    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();
    const w: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bytes));
    for (w) |*x| x.* = random.floatNorm(f32);
    resident_lock.lock();
    const registered = residentLookupLocked(@intFromPtr(bytes.ptr), bytes.len) != null;
    resident_lock.unlock();
    try std.testing.expect(registered);

    // Resident RHS dispatch (registry hit → host pointer used as device
    // pointer, no upload) must match the CPU reference like any other gemm.
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    for (a) |*x| x.* = random.floatNorm(f32);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);
    @memset(c, std.math.nan(f32));
    const expected = try allocator.alloc(f32, m * n);
    defer allocator.free(expected);

    try std.testing.expect(gemmF32(.nt, a, w, c, m, n, k));
    cpuReference(.nt, a, w, expected, m, n, k);
    for (c, expected) |got, want| {
        const tol = @max(2e-5 * @max(@abs(want), @abs(got)), 2e-5);
        try std.testing.expect(@abs(got - want) <= tol);
    }

    // CPU can still read the managed bytes after GPU use (READ_MOSTLY keeps
    // duplicated pages, so the read must not disturb later GPU access).
    var acc: f64 = 0;
    for (w) |x| acc += x;
    std.mem.doNotOptimizeAway(acc);
}

/// Quantize random rows into `blocks` and return the dequantized f32 matrix
/// the GPU kernel effectively sees (same helper as the Metal test suite).
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
        .q5_k => .q5_k,
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

/// Reference over f16-rounded operands — the GPU kernel stores dequantized
/// weights and f32 activations as half in shared memory and accumulates f32
/// (the Metal numerics contract; same 5e-3 tier).
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

fn expectQuantKernelAgreement(mma: []const f32, scalar: []const f32) !void {
    try std.testing.expectEqual(mma.len, scalar.len);
    for (mma, scalar, 0..) |a, b, i| {
        const tol = @max(1e-3 * @max(@abs(a), @abs(b)), 1e-3);
        if (@abs(a - b) > tol) {
            std.debug.print("quant MMA/scalar mismatch at {d}: mma={e} scalar={e}\n", .{ i, a, b });
            return error.TestUnexpectedResult;
        }
    }
}

test "cuda quant gemm q4_K/q5_K/q6_K/q8_0 parity vs dequantized reference" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    // Pin the decode arm off: the m=1 case must exercise the staged panel
    // path even when the suite runs with FUCINA_GPU_DECODE=1.
    ensureConfig();
    const saved_decode = state.decode_enabled;
    setDecodeForTest(false);
    defer setDecodeForTest(saved_decode);
    const saved_mma = state.quant_mma;
    defer setQuantMmaForTest(saved_mma);

    const dtype_mod = @import("../dtype.zig");
    var prng = std.Random.DefaultPrng.init(17);
    const random = prng.random();

    const Case = struct { m: usize, n: usize, k: usize };
    inline for (.{ QFormat.q6_k, QFormat.q4_k, QFormat.q5_k, QFormat.q8_0 }) |fmt| {
        const Block = switch (fmt) {
            .q6_k => dtype_mod.BlockQ6_K,
            .q4_k => dtype_mod.BlockQ4_K,
            .q5_k => dtype_mod.BlockQ5_K,
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
            const c_scalar = try allocator.alloc(f32, m * n);
            defer allocator.free(c_scalar);
            @memset(c_scalar, std.math.nan(f32));

            setQuantMmaForTest(true);
            try std.testing.expect(gemmQuantNt(
                fmt,
                std.mem.sliceAsBytes(blocks),
                false,
                bpr * @sizeOf(Block),
                a,
                c,
                m,
                n,
                k,
            ));
            try expectQuantGemmRows(a, wref, c, m, n, k);

            setQuantMmaForTest(false);
            try std.testing.expect(gemmQuantNt(
                fmt,
                std.mem.sliceAsBytes(blocks),
                false,
                bpr * @sizeOf(Block),
                a,
                c_scalar,
                m,
                n,
                k,
            ));
            try expectQuantGemmRows(a, wref, c_scalar, m, n, k);
            try expectQuantKernelAgreement(c, c_scalar);
        }
    }
}

test "cuda quant gemm grouped expert tiles parity" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    const dtype_mod = @import("../dtype.zig");
    var prng = std.Random.DefaultPrng.init(23);
    const random = prng.random();

    inline for (.{ QFormat.q6_k, QFormat.q4_k, QFormat.q5_k, QFormat.q8_0 }) |fmt| {
        const Block = switch (fmt) {
            .q6_k => dtype_mod.BlockQ6_K,
            .q4_k => dtype_mod.BlockQ4_K,
            .q5_k => dtype_mod.BlockQ5_K,
            .q8_0 => dtype_mod.BlockQ8_0,
        };
        const k = 2 * comptime fmt.kMultiple();
        const n = 64;
        const bpr = k / comptime fmt.kMultiple();
        const n_expert = 4;
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
            return error.SkipZigTest; // kernel module unavailable
        @memcpy(stage.in[0 .. total_rows * k], a);
        try std.testing.expect(gemmQGroupedNt(
            fmt,
            std.mem.sliceAsBytes(blocks),
            false,
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

test "cuda eager async dense quant Q4_K/Q5_K/Q6_K/Q8_0 uses direct tensor storage" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;
    ensureConfig();
    const saved_mma = state.quant_mma;
    const saved_split_k = state.quant_split_k;
    setQuantMmaForTest(true);
    setQuantSplitKForTest(true);
    defer {
        setQuantMmaForTest(saved_mma);
        setQuantSplitKForTest(saved_split_k);
    }

    const dtype_mod = @import("../dtype.zig");
    var prng = std.Random.DefaultPrng.init(31);
    const random = prng.random();
    inline for (.{ QFormat.q6_k, QFormat.q4_k, QFormat.q5_k, QFormat.q8_0 }) |fmt| {
        const Block = switch (fmt) {
            .q6_k => dtype_mod.BlockQ6_K,
            .q4_k => dtype_mod.BlockQ4_K,
            .q5_k => dtype_mod.BlockQ5_K,
            .q8_0 => dtype_mod.BlockQ8_0,
        };
        const m = 65;
        const n = 68;
        const k: usize = @max(512, 2 * comptime fmt.kMultiple());
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

test "cuda decode gemv q4_K/q5_K/q6_K/q8_0 parity vs dequantized reference" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    const dtype_mod = @import("../dtype.zig");
    var prng = std.Random.DefaultPrng.init(29);
    const random = prng.random();

    inline for (.{ QFormat.q6_k, QFormat.q4_k, QFormat.q5_k, QFormat.q8_0 }) |fmt| {
        const Block = switch (fmt) {
            .q6_k => dtype_mod.BlockQ6_K,
            .q4_k => dtype_mod.BlockQ4_K,
            .q5_k => dtype_mod.BlockQ5_K,
            .q8_0 => dtype_mod.BlockQ8_0,
        };
        const k = 2 * comptime fmt.kMultiple();
        const n = 68; // n % 4 == 0, not a multiple of the warp group
        // Q5_K uses GEMV for the compact-decode rows only; m>=4 deliberately
        // takes the tiled MMA route (covered by the async GEMM test above).
        const m = if (fmt == .q5_k) 3 else 4;
        const bpr = k / comptime fmt.kMultiple();
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

        // The heap blocks are stable for the test's duration; adoption is the
        // path under test (a plain transient RHS would be refused by design).
        // Evict before the allocator frees them — stale-wrap hygiene.
        const blocks_bytes = std.mem.sliceAsBytes(blocks);
        defer freeResidentBytes(blocks_bytes);
        try std.testing.expect(gemvQuant(fmt, blocks_bytes, true, bpr * @sizeOf(Block), a, c, m, n, k));

        // f32 dequant-dot (no f16 rounding) vs the dequantized reference.
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f64 = 0;
                for (0..k) |p| acc += @as(f64, a[i * k + p]) * @as(f64, wref[j * k + p]);
                const got: f32 = c[i * n + j];
                const want: f32 = @floatCast(acc);
                const tol = @max(5e-3 * @max(@abs(want), @abs(got)), 5e-3);
                try std.testing.expect(@abs(got - want) <= tol);
            }
        }
    }
}

test "cuda eager async gemm chains device results and synchronizes on host read" {
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

    try std.testing.expect(gemmF32Async(.nt, &a, &b, &first, m, n, k));
    const producer = first.buffer.pending() orelse return error.TestUnexpectedResult;
    try std.testing.expect(producer.devicePtr(.cuda) != null);
    try std.testing.expect(gemmF32Async(.nn, &first, &b, &second, m, k, n));
    try std.testing.expect(second.buffer.pending() != null);
    const got = second.dataConst();
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

test "cuda eager async input mutation waits for upload/device use" {
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
    a.data()[0] += 100;
    try std.testing.expect(a.buffer.pending_use.load(.acquire) == null);
    for (out.dataConst(), expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 2e-4);
}

test "cuda eager async f16 NT writes f32 directly and fences input mutation" {
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

test "cuda resident f16 gate admits decode without admitting streamed decode" {
    if (!enabled) return error.SkipZigTest;
    if (context() == null) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const n = 4096;
    const k = 1024;
    const resident = allocResidentBytes(n * k * @sizeOf(f16)) orelse return error.SkipZigTest;
    defer freeResidentBytes(resident);
    const values: []f16 = @alignCast(std.mem.bytesAsSlice(f16, resident));
    var rhs = try TensorF16.fromBorrowedSlice(allocator, &.{ n, k }, values);
    defer rhs.deinit();
    try std.testing.expect(shouldUseGpuF16ForRhs(&rhs, 1, n, k));

    const host = try allocator.alloc(f16, n * k);
    defer allocator.free(host);
    var transient = try TensorF16.fromBorrowedSlice(allocator, &.{ n, k }, host);
    defer transient.deinit();
    try std.testing.expect(!shouldUseGpuF16ForRhs(&transient, 1, n, k));
}

test "cuda prefill attention parity vs f64 reference (gqa, offset, window, bidi)" {
    if (comptime !enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    if (context() == null) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(41);
    const random = prng.random();

    const Case = struct { q_seq: usize, kv_seq: usize, heads: usize, kv_heads: usize, d: usize, window: usize, causal: bool };
    const cases = [_]Case{
        .{ .q_seq = 64, .kv_seq = 64, .heads = 8, .kv_heads = 4, .d = 64, .window = 0, .causal = true }, // plain gqa causal
        .{ .q_seq = 33, .kv_seq = 97, .heads = 4, .kv_heads = 4, .d = 48, .window = 0, .causal = true }, // chunked prefill (offset 64)
        .{ .q_seq = 40, .kv_seq = 40, .heads = 8, .kv_heads = 2, .d = 128, .window = 16, .causal = true }, // sliding window
        .{ .q_seq = 37, .kv_seq = 37, .heads = 6, .kv_heads = 3, .d = 80, .window = 0, .causal = false }, // bidirectional
    };
    for (cases) |case| {
        const qs = case.q_seq;
        const ks = case.kv_seq;
        const nh = case.heads;
        const nkh = case.kv_heads;
        const d = case.d;
        const offset = ks - qs;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

        const q = try allocator.alloc(f32, qs * nh * d);
        defer allocator.free(q);
        const k = try allocator.alloc(f16, ks * nkh * d);
        defer allocator.free(k);
        const v = try allocator.alloc(f16, ks * nkh * d);
        defer allocator.free(v);
        const out = try allocator.alloc(f32, qs * nh * d);
        defer allocator.free(out);
        for (q) |*x| x.* = random.floatNorm(f32);
        for (k) |*x| x.* = @floatCast(random.floatNorm(f32));
        for (v) |*x| x.* = @floatCast(random.floatNorm(f32));
        @memset(out, std.math.nan(f32));

        var map: [8]i32 = undefined;
        for (0..nh) |h| map[h] = @intCast(h * nkh / nh);

        try std.testing.expect(attnPrefillF16(q, k, v, out, map[0..nh], qs, ks, nh, nkh, d, offset, scale, case.window, case.causal));

        const scores = try allocator.alloc(f64, ks);
        defer allocator.free(scores);
        for (0..qs) |qi| {
            const p_abs = offset + qi;
            const end = if (case.causal) @min(ks, p_abs + 1) else ks;
            const start = if (case.causal and case.window > 0 and p_abs + 1 > case.window) p_abs + 1 - case.window else 0;
            for (0..nh) |h| {
                const kvh: usize = @intCast(map[h]);
                var m: f64 = -std.math.inf(f64);
                for (start..end) |j| {
                    var dot: f64 = 0;
                    for (0..d) |t| dot += @as(f64, q[(qi * nh + h) * d + t]) * @as(f64, k[(j * nkh + kvh) * d + t]);
                    scores[j] = dot * scale;
                    m = @max(m, scores[j]);
                }
                var l: f64 = 0;
                for (start..end) |j| {
                    scores[j] = @exp(scores[j] - m);
                    l += scores[j];
                }
                for (0..d) |t| {
                    var acc: f64 = 0;
                    for (start..end) |j| acc += scores[j] * @as(f64, v[(j * nkh + kvh) * d + t]);
                    const want: f32 = @floatCast(acc / l);
                    const got = out[(qi * nh + h) * d + t];
                    const tol = @max(2e-3 * @max(@abs(want), @abs(got)), 2e-3);
                    try std.testing.expect(@abs(got - want) <= tol);
                }
            }
        }
    }
}

test "cuda resident allocation refuses beyond the VRAM budget" {
    if (comptime !enabled) return error.SkipZigTest;
    if (context() == null) return error.SkipZigTest;
    // An allocation larger than any plausible budget must refuse (tracked
    // budget or slot bound), not crash — callers fall back to host+transient.
    try std.testing.expect(allocResidentBytes(std.math.maxInt(usize) / 4) == null);
}

test "qmoe fill gate arithmetic" {
    if (comptime !enabled) return error.SkipZigTest;
    ensureConfig();
    const saved = state.qmoe_min_fill_pct;
    defer state.qmoe_min_fill_pct = saved;

    state.qmoe_min_fill_pct = 50;
    try std.testing.expect(qmoeFillAcceptable(2048, 128));
    try std.testing.expect(!qmoeFillAcceptable(2047, 128));
    try std.testing.expect(qmoeFillAcceptable(4096, 128));
    try std.testing.expect(!qmoeFillAcceptable(0, 0));

    state.qmoe_min_fill_pct = 0;
    try std.testing.expect(qmoeFillAcceptable(1, 128));

    state.qmoe_min_fill_pct = 101;
    try std.testing.expect(!qmoeFillAcceptable(4096, 128));
}
