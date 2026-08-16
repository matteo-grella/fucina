//! The IEEE floating-point environment: capability inquiry, scoped control,
//! and accrued exception flags.
//!
//! Fucina makes bitwise promises — the RNG's (seed → values) checkpoint
//! contract, scalar-vs-native backend parity, thread-count-invariant kernels —
//! and every one of them assumes the default IEEE environment: round to
//! nearest-even, gradual underflow, no trapping. Nothing in the language
//! guarantees that. The rounding mode and the flush-to-zero bit live in a
//! per-thread control register that any code sharing the thread can change,
//! and this process shares its threads with an external CBLAS and a GPU
//! driver. A run whose results silently disagree with the pinned oracles
//! because a vendor library left flush-to-zero on is otherwise indistinguishable
//! from a real numeric bug.
//!
//! The accrued exception flags answer the other half: which IEEE exceptions did
//! a computation actually raise? For a library whose quantization formats live
//! near the edges of the f32 range (block scales, ternary planes, f16 KV rows),
//! "this pass overflowed" is a fact worth reading rather than inferring from
//! output NaNs.
//!
//! **Inquiry first.** Every entry point reports whether the target supports it
//! rather than assuming: `supported()` is the gate, and the getters return
//! `null` where the registers are not reachable. Unsupported targets degrade to
//! no-ops, never to a wrong answer.
//!
//! **No hot-path cost.** Nothing here is called by a kernel. Reads and writes
//! touch one control register and appear only at context construction, in
//! explicitly scoped guards, and in tests.
//!
//! Implemented for aarch64 (`FPCR`/`FPSR`) and x86_64 (`MXCSR`), the two
//! architectures the kernels target. No libc dependency: the control registers
//! are read directly, so this works on libc-free static builds where `fenv.h`
//! is unavailable.

const std = @import("std");
const builtin = @import("builtin");

/// True when this target exposes the floating-point control/status registers.
/// Every getter below returns `null` and every setter is a no-op when false.
pub const supported = switch (builtin.cpu.arch) {
    .aarch64, .aarch64_be, .x86_64 => true,
    else => false,
};

/// IEEE rounding direction (Fortran's `ieee_round_type`).
pub const RoundingMode = enum {
    /// Round to nearest, ties to even. The IEEE default and the mode every
    /// pinned numeric contract in this repo assumes.
    nearest_even,
    /// Round toward zero (truncate).
    toward_zero,
    /// Round toward +infinity.
    upward,
    /// Round toward -infinity.
    downward,
};

/// How subnormal (denormal) values are handled.
///
/// `.gradual` is the IEEE behavior: subnormals are representable and arithmetic
/// on them is exact where the format allows. `.flush_to_zero` replaces them
/// with zero, which is faster on some hardware and is what several vendor math
/// libraries switch on. It changes results, so it is never the default here.
pub const UnderflowMode = enum { gradual, flush_to_zero };

/// The accrued IEEE exception flags (Fortran's `ieee_exceptions` set).
///
/// These are *sticky*: hardware sets a bit when the condition occurs and it
/// stays set until explicitly cleared. Reading them tells you what happened
/// since the last clear, not what is happening now.
pub const Exceptions = struct {
    /// An invalid operation: 0/0, inf-inf, sqrt of a negative, a comparison
    /// with a signaling NaN.
    invalid: bool = false,
    /// Division of a finite nonzero value by zero.
    division_by_zero: bool = false,
    /// The rounded result exceeded the format's largest finite magnitude.
    overflow: bool = false,
    /// The rounded result was subnormal and inexact.
    underflow: bool = false,
    /// The result had to be rounded. Raised by most ordinary arithmetic, so it
    /// is only interesting when you are auditing exactness.
    inexact: bool = false,
    /// A subnormal *operand* entered the computation. Not one of the five IEEE
    /// exceptions; both supported architectures expose it (x86 `DE`, aarch64
    /// `IDC`) and it is informative when auditing quantization block scales.
    denormal: bool = false,

    /// True when any flag is set.
    pub fn any(self: Exceptions) bool {
        return self.invalid or self.division_by_zero or self.overflow or
            self.underflow or self.inexact or self.denormal;
    }

    /// True when any flag other than `inexact` is set. Ordinary floating-point
    /// arithmetic raises `inexact` constantly; this is the "something worth
    /// looking at happened" predicate.
    pub fn anySignificant(self: Exceptions) bool {
        return self.invalid or self.division_by_zero or self.overflow or
            self.underflow or self.denormal;
    }

    /// Union of two flag sets.
    pub fn unionWith(self: Exceptions, other: Exceptions) Exceptions {
        return .{
            .invalid = self.invalid or other.invalid,
            .division_by_zero = self.division_by_zero or other.division_by_zero,
            .overflow = self.overflow or other.overflow,
            .underflow = self.underflow or other.underflow,
            .inexact = self.inexact or other.inexact,
            .denormal = self.denormal or other.denormal,
        };
    }

    pub fn format(self: Exceptions, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        if (!self.any()) return writer.writeAll("none");
        var first = true;
        inline for (@typeInfo(Exceptions).@"struct".fields) |field| {
            if (comptime field.type != bool) continue;
            if (@field(self, field.name)) {
                if (!first) try writer.writeAll("|");
                try writer.writeAll(field.name);
                first = false;
            }
        }
    }
};

/// The controllable part of the environment: what a `Guard` saves and restores.
pub const Environment = struct {
    rounding: RoundingMode,
    underflow: UnderflowMode,

    /// True when this is the IEEE default environment every pinned numeric
    /// contract in the repo assumes.
    pub fn isDefault(self: Environment) bool {
        return self.rounding == .nearest_even and self.underflow == .gradual;
    }
};

/// The IEEE default environment: round to nearest-even, gradual underflow.
pub const default_environment: Environment = .{ .rounding = .nearest_even, .underflow = .gradual };

pub const Error = error{
    /// The floating-point environment is not the IEEE default the repo's
    /// numeric contracts assume.
    NonDefaultFloatEnvironment,
};

// ---------------------------------------------------------------------------
// Inquiry and control
// ---------------------------------------------------------------------------

/// The current rounding mode, or null on an unsupported target.
pub fn roundingMode() ?RoundingMode {
    if (comptime !supported) return null;
    return impl.roundingMode();
}

/// The current subnormal handling, or null on an unsupported target.
pub fn underflowMode() ?UnderflowMode {
    if (comptime !supported) return null;
    return impl.underflowMode();
}

/// The full controllable environment, or null on an unsupported target.
pub fn get() ?Environment {
    if (comptime !supported) return null;
    return .{ .rounding = impl.roundingMode(), .underflow = impl.underflowMode() };
}

/// Install `env` on the calling thread. A no-op on an unsupported target.
///
/// This changes the numerics of every subsequent floating-point operation on
/// this thread, including inside kernels and vendor libraries. Prefer `Guard`,
/// which pairs the change with its restore.
pub fn set(env: Environment) void {
    if (comptime !supported) return;
    impl.set(env);
}

/// The accrued exception flags. All-false on an unsupported target.
pub fn testExceptions() Exceptions {
    if (comptime !supported) return .{};
    return impl.testExceptions();
}

/// Clear every accrued exception flag on the calling thread.
pub fn clearExceptions() void {
    if (comptime !supported) return;
    impl.clearExceptions();
}

/// Replace the accrued exception flags with `flags`.
pub fn raiseExceptions(flags: Exceptions) void {
    if (comptime !supported) return;
    impl.setExceptions(flags);
}

/// `error.NonDefaultFloatEnvironment` when the environment is not the IEEE
/// default. Succeeds on unsupported targets, where there is nothing to check.
///
/// Call this at a boundary you want to hold the default across: after loading a
/// plugin, after the first call into a vendor library, or at the top of a run
/// whose results are compared against pinned oracles.
pub fn assertDefault() Error!void {
    const env = get() orelse return;
    if (!env.isDefault()) return Error.NonDefaultFloatEnvironment;
}

// ---------------------------------------------------------------------------
// Scoped helpers
// ---------------------------------------------------------------------------

/// Saves the environment on construction and puts it back on `restore`.
///
/// This is Fortran's rule that a procedure which changes the environment must
/// restore it before returning, made explicit:
///
/// ```
/// var guard = fpenv.Guard.begin();
/// defer guard.restore();
/// fpenv.set(.{ .rounding = .toward_zero, .underflow = .gradual });
/// ```
///
/// `restore` is idempotent and safe on unsupported targets.
pub const Guard = struct {
    saved: ?Environment,

    pub fn begin() Guard {
        return .{ .saved = get() };
    }

    pub fn restore(self: *const Guard) void {
        if (self.saved) |env| set(env);
    }
};

/// Measures which IEEE exceptions a region of code raises.
///
/// `begin` takes over the accrued flags (saving whatever had accumulated and
/// clearing them), `sample` reads what has accrued since, and `end` reads the
/// final set and merges the saved outer flags back in, so probes nest without
/// a inner region hiding an outer one's history:
///
/// ```
/// var probe = fpenv.Probe.begin();
/// ... suspect computation ...
/// const raised = probe.end();
/// if (raised.overflow) ...
/// ```
///
/// The flags are per-thread. A probe around a parallel op observes only what the
/// calling thread raised, not what the worker team raised.
pub const Probe = struct {
    outer: Exceptions,

    pub fn begin() Probe {
        const outer = testExceptions();
        clearExceptions();
        return .{ .outer = outer };
    }

    /// What has accrued since `begin`, leaving the probe running.
    pub fn sample(_: *const Probe) Exceptions {
        return testExceptions();
    }

    /// What accrued since `begin`. Restores the pre-probe flags on top of the
    /// current ones so an enclosing probe keeps its history.
    pub fn end(self: *const Probe) Exceptions {
        const raised = testExceptions();
        raiseExceptions(raised.unionWith(self.outer));
        return raised;
    }
};

// ---------------------------------------------------------------------------
// Per-architecture register access
// ---------------------------------------------------------------------------

const impl = switch (builtin.cpu.arch) {
    .aarch64, .aarch64_be => aarch64_impl,
    .x86_64 => x86_64_impl,
    else => unsupported_impl,
};

const unsupported_impl = struct {
    fn roundingMode() RoundingMode {
        unreachable;
    }
    fn underflowMode() UnderflowMode {
        unreachable;
    }
    fn set(_: Environment) void {
        unreachable;
    }
    fn testExceptions() Exceptions {
        unreachable;
    }
    fn clearExceptions() void {
        unreachable;
    }
    fn setExceptions(_: Exceptions) void {
        unreachable;
    }
};

/// aarch64: `FPCR` holds the control bits, `FPSR` the accrued status bits.
/// Both are readable and writable at EL0.
const aarch64_impl = struct {
    // FPCR
    const rmode_shift: u6 = 22; // RMode, 2 bits
    const fz_bit: u64 = 1 << 24; // flush-to-zero
    const fz16_bit: u64 = 1 << 19; // flush-to-zero for f16
    // FPSR accrued exception bits.
    const ioc: u64 = 1 << 0; // invalid operation
    const dzc: u64 = 1 << 1; // divide by zero
    const ofc: u64 = 1 << 2; // overflow
    const ufc: u64 = 1 << 3; // underflow
    const ixc: u64 = 1 << 4; // inexact
    const idc: u64 = 1 << 7; // input denormal
    const flag_mask: u64 = ioc | dzc | ofc | ufc | ixc | idc;

    fn readFpcr() u64 {
        return asm volatile ("mrs %[out], fpcr"
            : [out] "=r" (-> u64),
        );
    }

    fn writeFpcr(value: u64) void {
        asm volatile ("msr fpcr, %[value]"
            :
            : [value] "r" (value),
        );
    }

    fn readFpsr() u64 {
        return asm volatile ("mrs %[out], fpsr"
            : [out] "=r" (-> u64),
        );
    }

    fn writeFpsr(value: u64) void {
        asm volatile ("msr fpsr, %[value]"
            :
            : [value] "r" (value),
        );
    }

    fn roundingMode() RoundingMode {
        // RMode: 00 = nearest-even, 01 = +inf, 10 = -inf, 11 = zero.
        return switch ((readFpcr() >> rmode_shift) & 0b11) {
            0b00 => .nearest_even,
            0b01 => .upward,
            0b10 => .downward,
            else => .toward_zero,
        };
    }

    fn underflowMode() UnderflowMode {
        return if (readFpcr() & fz_bit != 0) .flush_to_zero else .gradual;
    }

    fn set(env: Environment) void {
        var fpcr = readFpcr();
        fpcr &= ~(@as(u64, 0b11) << rmode_shift);
        fpcr |= @as(u64, switch (env.rounding) {
            .nearest_even => 0b00,
            .upward => 0b01,
            .downward => 0b10,
            .toward_zero => 0b11,
        }) << rmode_shift;
        switch (env.underflow) {
            .gradual => fpcr &= ~(fz_bit | fz16_bit),
            .flush_to_zero => fpcr |= fz_bit | fz16_bit,
        }
        writeFpcr(fpcr);
    }

    fn testExceptions() Exceptions {
        const fpsr = readFpsr();
        return .{
            .invalid = fpsr & ioc != 0,
            .division_by_zero = fpsr & dzc != 0,
            .overflow = fpsr & ofc != 0,
            .underflow = fpsr & ufc != 0,
            .inexact = fpsr & ixc != 0,
            .denormal = fpsr & idc != 0,
        };
    }

    fn clearExceptions() void {
        writeFpsr(readFpsr() & ~flag_mask);
    }

    fn setExceptions(flags: Exceptions) void {
        var fpsr = readFpsr() & ~flag_mask;
        if (flags.invalid) fpsr |= ioc;
        if (flags.division_by_zero) fpsr |= dzc;
        if (flags.overflow) fpsr |= ofc;
        if (flags.underflow) fpsr |= ufc;
        if (flags.inexact) fpsr |= ixc;
        if (flags.denormal) fpsr |= idc;
        writeFpsr(fpsr);
    }
};

/// x86_64: `MXCSR` holds both the control bits and the accrued status bits for
/// SSE/AVX arithmetic, which is what Zig's float operations compile to. The x87
/// control word governs only legacy x87 math and is deliberately not touched.
const x86_64_impl = struct {
    const ie: u32 = 1 << 0; // invalid operation
    const de: u32 = 1 << 1; // denormal operand
    const ze: u32 = 1 << 2; // divide by zero
    const oe: u32 = 1 << 3; // overflow
    const ue: u32 = 1 << 4; // underflow
    const pe: u32 = 1 << 5; // precision (inexact)
    const flag_mask: u32 = ie | de | ze | oe | ue | pe;
    const daz_bit: u32 = 1 << 6; // denormals-are-zero (inputs)
    const rc_shift: u5 = 13; // rounding control, 2 bits
    const ftz_bit: u32 = 1 << 15; // flush-to-zero (results)

    fn readMxcsr() u32 {
        var value: u32 = undefined;
        asm volatile ("stmxcsr %[out]"
            : [out] "=m" (value),
        );
        return value;
    }

    fn writeMxcsr(value: u32) void {
        asm volatile ("ldmxcsr %[value]"
            :
            : [value] "m" (value),
        );
    }

    fn roundingMode() RoundingMode {
        // RC: 00 = nearest-even, 01 = -inf, 10 = +inf, 11 = zero. Note the
        // middle two are the opposite order from aarch64's RMode.
        return switch ((readMxcsr() >> rc_shift) & 0b11) {
            0b00 => .nearest_even,
            0b01 => .downward,
            0b10 => .upward,
            else => .toward_zero,
        };
    }

    fn underflowMode() UnderflowMode {
        const csr = readMxcsr();
        return if (csr & (ftz_bit | daz_bit) != 0) .flush_to_zero else .gradual;
    }

    fn set(env: Environment) void {
        var csr = readMxcsr();
        csr &= ~(@as(u32, 0b11) << rc_shift);
        csr |= @as(u32, switch (env.rounding) {
            .nearest_even => 0b00,
            .downward => 0b01,
            .upward => 0b10,
            .toward_zero => 0b11,
        }) << rc_shift;
        switch (env.underflow) {
            .gradual => csr &= ~(ftz_bit | daz_bit),
            .flush_to_zero => csr |= ftz_bit | daz_bit,
        }
        writeMxcsr(csr);
    }

    fn testExceptions() Exceptions {
        const csr = readMxcsr();
        return .{
            .invalid = csr & ie != 0,
            .division_by_zero = csr & ze != 0,
            .overflow = csr & oe != 0,
            .underflow = csr & ue != 0,
            .inexact = csr & pe != 0,
            .denormal = csr & de != 0,
        };
    }

    fn clearExceptions() void {
        writeMxcsr(readMxcsr() & ~flag_mask);
    }

    fn setExceptions(flags: Exceptions) void {
        var csr = readMxcsr() & ~flag_mask;
        if (flags.invalid) csr |= ie;
        if (flags.division_by_zero) csr |= ze;
        if (flags.overflow) csr |= oe;
        if (flags.underflow) csr |= ue;
        if (flags.inexact) csr |= pe;
        if (flags.denormal) csr |= de;
        writeMxcsr(csr);
    }
};

test {
    _ = @import("fpenv_tests.zig");
}
