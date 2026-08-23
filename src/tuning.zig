//! Tuning policy: the one shape every FUCINA_* boolean route gate shares,
//! the numeric route defaults, and the per-context overrides an
//! `ExecContext` can carry.
//!
//! Route gates ("use the fused kernel / the compact decode route / the
//! Winograd tier?") follow one contract across the tree: a comptime-named
//! pair of env switches (`FUCINA_<X>` forces on, `FUCINA_NO_<X>` forces
//! off — the A/B and emergency-revert switches), a measured default, a
//! read-once process cache, and a programmatic `set` used by tests and CLI
//! flags. `Switch` is that contract as a type; `Threshold` is the same
//! read-once contract for a numeric crossover. Both read the environment
//! through `parallel.envFlag`/`envPositiveUsize` — the only sanctioned env
//! readers (libc-free Linux builds have no `std.c.getenv`).
//!
//! Process-wide gates answer "which kernel implements this op here" — a
//! per-machine fact. Policy that can differ per *workload* (two
//! `ExecContext`s in one process wanting different route choices) goes
//! through `Overrides`, carried by the context's `Runtime` and consulted
//! before the process gate.

const std = @import("std");
const parallel = @import("parallel.zig");

/// Numeric route defaults for the env-overridable crossovers (each backs
/// a `Threshold` at its consuming route). Fixed measured crossovers with
/// no env override live as named constants beside their consuming route,
/// with provenance in that route's comment block (machine, benchmark,
/// date).
pub const defaults = struct {
    /// CPU f32 weight-shadow crossover: rows `m >=` this take the BLAS
    /// route over the widened-once shadow (exec/matmul.zig block comment;
    /// measured bench-f16gemm, M1 Max + Accelerate).
    pub const cpu_f32_shadow_min_m: u64 = 32;
};

/// Per-context tuning overrides (`ExecContext.setTuning`); every field
/// `null` means "follow the process default" (env switch or measured
/// default). Fields are consulted by the routes that support per-context
/// policy; process-wide gates ignore them by design.
pub const Overrides = struct {
    /// CPU f32 weight-shadow route: force on/off for this context
    /// (process default: the `FUCINA_CPU_F32_SHADOW` gate, default off).
    cpu_f32_shadow: ?bool = null,
    /// Shadow crossover row count for this context (process default:
    /// `FUCINA_CPU_F32_SHADOW_MIN_M`, else `defaults.cpu_f32_shadow_min_m`).
    cpu_f32_shadow_min_m: ?u64 = null,
};

pub const SwitchConfig = struct {
    /// Env name that forces the route ON (getenv-family truthiness).
    on: ?[:0]const u8 = null,
    /// Env name that forces the route OFF; wins over `on`.
    off: ?[:0]const u8 = null,
    /// The measured default when neither env is set. May be
    /// comptime-computed (e.g. keyed on the BLAS provider).
    default: bool,
};

/// A read-once boolean route gate. Each instantiation owns one process-wide
/// cached state; `enabled()` reads the environment on first use, `set`
/// pre-seeds or re-arms the cache (the tests' A/B hook, also usable from a
/// CLI flag).
pub fn Switch(comptime cfg: SwitchConfig) type {
    return struct {
        var state = std.atomic.Value(u8).init(0); // 0 unread, 1 on, 2 off

        pub fn enabled() bool {
            const s = state.load(.acquire);
            if (s != 0) return s == 1;
            var on = cfg.default;
            if (cfg.on) |name| {
                if (parallel.envFlag(name)) on = true;
            }
            if (cfg.off) |name| {
                if (parallel.envFlag(name)) on = false;
            }
            state.store(if (on) 1 else 2, .release);
            return on;
        }

        /// `true`/`false` pin the route; `null` re-arms the env read.
        pub fn set(on: ?bool) void {
            state.store(if (on) |o| @as(u8, if (o) 1 else 2) else 0, .release);
        }
    };
}

/// A read-once numeric crossover with an env override. Same cache contract
/// as `Switch`; `set(null)` restores the default and re-arms the env read.
pub fn Threshold(comptime name: [:0]const u8, comptime default: u64) type {
    return struct {
        var read = std.atomic.Value(bool).init(false);
        var value = std.atomic.Value(u64).init(default);

        pub fn get() u64 {
            if (!read.load(.acquire)) {
                if (parallel.envPositiveUsize(name)) |v| value.store(v, .release);
                read.store(true, .release);
            }
            return value.load(.acquire);
        }

        pub fn set(v: ?u64) void {
            if (v) |x| {
                value.store(x, .release);
                read.store(true, .release);
            } else {
                value.store(default, .release);
                read.store(false, .release);
            }
        }
    };
}

test {
    _ = @import("tuning_tests.zig");
}
