//! Shared dispatch-trace shell for the GPU providers (FUCINA_GPU_TRACE=1):
//! one atomic counter table keyed by `Counter`, plus the timing helpers.
//! Zero overhead when off — `Table.on` is a plain bool latched once at the
//! provider's configuration read, and every helper no-ops while it is
//! false. Counters are atomic because gates run on worker threads while
//! dispatches are lock-serialized.
//!
//! Provider-specific counters are enum members only that provider
//! increments; the dump format stays with the provider (the two report
//! different capabilities: Metal has command-buffer GPU/sched timestamps,
//! CUDA has PCIe byte counts).

const std = @import("std");
const builtin = @import("builtin");

/// Every counter either provider accumulates. Values are ns for `_ns`
/// members, bytes for `_bytes`, plain event counts otherwise.
pub const Counter = enum {
    // Shared: per-kind dispatch counts and blocking wall time.
    f32_calls,
    f32_ns,
    f16_calls,
    f16_ns,
    quant_calls,
    quant_ns,
    attn_calls,
    attn_ns,
    // Shared: eager-async submit/host-wait split.
    f32_async_calls,
    f32_submit_ns,
    f32_wait_ns,
    f16_async_calls,
    f16_submit_ns,
    f16_wait_ns,
    quant_async_calls,
    quant_submit_ns,
    quant_wait_ns,
    // Shared: device-owned resident bytes; refusals mean the caller kept
    // its host bytes (VRAM budget, or the range registry could not grow).
    dev_alloc_calls,
    dev_alloc_bytes,
    resident_refusals,
    // Shared: gate decisions.
    gate_pass,
    gate_below,
    gate_shape,
    // Metal only: command-buffer GPU/kernel-scheduling timestamps.
    f32_gpu_ns,
    f32_sched_ns,
    f16_gpu_ns,
    f16_sched_ns,
    attn_gpu_ns,
    attn_sched_ns,
    quant_gpu_ns,
    quant_sched_ns,
    // Metal only: the bf16 NT async arm (CUDA has no bf16 kernel).
    bf16_async_calls,
    bf16_submit_ns,
    bf16_wait_ns,
    bf16_gpu_ns,
    bf16_sched_ns,
    // Metal only: grouped-MoE lock-wait/stage-copy split, RHS wrap-cache
    // admission, shim errors, shape-bucket overflow.
    quant_lock_ns,
    quant_stage_ns,
    rhs_cacheable,
    rhs_transient,
    shim_err,
    shape_overflow,
    // CUDA only: GEMV dispatches, PCIe traffic, RHS residency split, the
    // transient-floor rejection tier, driver errors.
    gemv_calls,
    h2d_bytes,
    d2h_bytes,
    rhs_resident,
    rhs_streamed,
    transient_below,
    cuda_err,
};

const counter_count = @typeInfo(Counter).@"enum".fields.len;

/// The counter table. Providers hold one as a file-scope `var trace` and
/// latch `on` from `tuning.Table.Gpu.trace` at their one-time
/// configuration read.
pub const Table = struct {
    on: bool = false,
    counts: [counter_count]std.atomic.Value(u64) = @splat(.{ .raw = 0 }),

    pub inline fn inc(self: *Table, c: Counter, v: u64) void {
        if (!self.on) return;
        _ = self.counts[@intFromEnum(c)].fetchAdd(v, .monotonic);
    }

    pub fn get(self: *const Table, c: Counter) u64 {
        return self.counts[@intFromEnum(c)].load(.monotonic);
    }

    /// Zero every counter, keeping `on`. Single-threaded (call before a
    /// warm measurement window).
    pub fn reset(self: *Table) void {
        for (&self.counts) |*c| c.store(0, .monotonic);
    }

    pub inline fn gate(self: *Table, pass: bool) void {
        self.inc(if (pass) .gate_pass else .gate_below, 1);
    }

    /// A start timestamp (0 when tracing is off → `elapsed`/`finish` cost
    /// nothing on the same check).
    pub inline fn start(self: *const Table) u64 {
        return if (self.on) now() else 0;
    }

    /// Accumulates now-start into `c`.
    pub inline fn elapsed(self: *Table, c: Counter, started: u64) void {
        if (!self.on) return;
        _ = self.counts[@intFromEnum(c)].fetchAdd(now() -% started, .monotonic);
    }

    /// Returns now-start (0 when tracing is off).
    pub inline fn finish(self: *const Table, started: u64) u64 {
        return if (self.on) now() -% started else 0;
    }
};

pub fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

pub fn mb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1e6;
}

// Monotonic ns without an `std.Io` handle (the backend dispatch sites have
// none, and `std.time.Timer` was removed in 0.16's Io migration). Both
// providers' builds link libc (Metal is macOS-only; Linux `-Dgpu=cuda`
// binaries always link libc, build.zig configureGpu), so the OS clock is
// declared per target and the unused extern stays unreferenced.
extern fn clock_gettime_nsec_np(clock_id: c_int) u64;
const clock_uptime_raw: c_int = 8; // Darwin <sys/_clock_id.h> CLOCK_UPTIME_RAW

const CTimespec = extern struct { sec: c_long, nsec: c_long };
extern "c" fn clock_gettime(clk_id: c_int, tp: *CTimespec) c_int;
const clock_monotonic: c_int = 1; // Linux CLOCK_MONOTONIC

fn now() u64 {
    if (comptime builtin.os.tag.isDarwin()) {
        return clock_gettime_nsec_np(clock_uptime_raw);
    } else {
        var ts: CTimespec = undefined;
        if (clock_gettime(clock_monotonic, &ts) != 0) return 0;
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    }
}
