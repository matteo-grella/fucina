//! Behavior tests for the IEEE floating-point environment facility.
//!
//! Every test that changes the environment pairs the change with a `Guard`, so
//! a failing assertion cannot leave the mode installed for the tests that
//! follow. Tests that need the hardware skip cleanly on targets where the
//! control registers are unreachable.

const std = @import("std");
const fpenv = @import("fpenv.zig");

/// Opaque float source. The rounding-mode and flush-to-zero tests only mean
/// something if the arithmetic actually executes at run time; routing operands
/// through a `noinline` boundary keeps the optimizer from folding them.
noinline fn opaqueF32(value: f32) f32 {
    std.mem.doNotOptimizeAway(&value);
    return value;
}

noinline fn mulF32(a: f32, b: f32) f32 {
    const product = a * b;
    std.mem.doNotOptimizeAway(&product);
    return product;
}

noinline fn divF32(a: f32, b: f32) f32 {
    const quotient = a / b;
    std.mem.doNotOptimizeAway(&quotient);
    return quotient;
}

fn requireSupport() !void {
    if (!fpenv.supported) return error.SkipZigTest;
}

test "inquiry agrees with itself" {
    // The contract on an unsupported target is null, not a wrong answer.
    if (!fpenv.supported) {
        try std.testing.expect(fpenv.get() == null);
        try std.testing.expect(fpenv.roundingMode() == null);
        try std.testing.expect(fpenv.underflowMode() == null);
        return;
    }
    const env = fpenv.get().?;
    try std.testing.expectEqual(env.rounding, fpenv.roundingMode().?);
    try std.testing.expectEqual(env.underflow, fpenv.underflowMode().?);
}

test "the process starts in the IEEE default environment" {
    try requireSupport();
    // If this fails, something in the process (a vendor library, a driver, a
    // startup hook) changed the environment before the tests ran, and every
    // bitwise contract in the repo is being evaluated under different numerics.
    try fpenv.assertDefault();
    try std.testing.expect(fpenv.get().?.isDefault());
}

test "Guard restores the rounding mode" {
    try requireSupport();
    const before = fpenv.get().?;
    {
        var guard = fpenv.Guard.begin();
        defer guard.restore();
        fpenv.set(.{ .rounding = .toward_zero, .underflow = .gradual });
        try std.testing.expectEqual(fpenv.RoundingMode.toward_zero, fpenv.roundingMode().?);
    }
    try std.testing.expectEqual(before.rounding, fpenv.roundingMode().?);
    try std.testing.expectEqual(before.underflow, fpenv.underflowMode().?);
}

test "every rounding mode round-trips through set and get" {
    try requireSupport();
    var guard = fpenv.Guard.begin();
    defer guard.restore();
    for ([_]fpenv.RoundingMode{ .nearest_even, .toward_zero, .upward, .downward }) |mode| {
        fpenv.set(.{ .rounding = mode, .underflow = .gradual });
        try std.testing.expectEqual(mode, fpenv.roundingMode().?);
        // Setting the rounding mode must not disturb the underflow mode.
        try std.testing.expectEqual(fpenv.UnderflowMode.gradual, fpenv.underflowMode().?);
    }
}

test "the rounding mode changes arithmetic results" {
    try requireSupport();
    var guard = fpenv.Guard.begin();
    defer guard.restore();

    // 1/3 is not representable, so the two directed roundings straddle it.
    fpenv.set(.{ .rounding = .downward, .underflow = .gradual });
    const down = divF32(opaqueF32(1.0), opaqueF32(3.0));
    fpenv.set(.{ .rounding = .upward, .underflow = .gradual });
    const up = divF32(opaqueF32(1.0), opaqueF32(3.0));
    guard.restore();

    try std.testing.expect(down < up);
    const nearest = divF32(opaqueF32(1.0), opaqueF32(3.0));
    try std.testing.expect(nearest == down or nearest == up);
}

test "flush-to-zero is observable and restorable" {
    try requireSupport();
    const denormal: f32 = 1.0e-40; // subnormal in binary32
    var guard = fpenv.Guard.begin();
    defer guard.restore();

    fpenv.set(.{ .rounding = .nearest_even, .underflow = .flush_to_zero });
    try std.testing.expectEqual(fpenv.UnderflowMode.flush_to_zero, fpenv.underflowMode().?);
    const flushed = mulF32(opaqueF32(denormal), opaqueF32(1.0));

    fpenv.set(.{ .rounding = .nearest_even, .underflow = .gradual });
    const gradual = mulF32(opaqueF32(denormal), opaqueF32(1.0));

    try std.testing.expectEqual(@as(f32, 0.0), flushed);
    try std.testing.expect(gradual != 0.0);
}

test "assertDefault rejects a non-default environment" {
    try requireSupport();
    var guard = fpenv.Guard.begin();
    defer guard.restore();

    fpenv.set(.{ .rounding = .toward_zero, .underflow = .gradual });
    try std.testing.expectError(fpenv.Error.NonDefaultFloatEnvironment, fpenv.assertDefault());

    fpenv.set(.{ .rounding = .nearest_even, .underflow = .flush_to_zero });
    try std.testing.expectError(fpenv.Error.NonDefaultFloatEnvironment, fpenv.assertDefault());

    guard.restore();
    try fpenv.assertDefault();
}

test "Probe reports the exceptions a computation raises" {
    try requireSupport();

    var overflow_probe = fpenv.Probe.begin();
    const huge = mulF32(opaqueF32(3.0e38), opaqueF32(3.0e38));
    const overflowed = overflow_probe.end();
    try std.testing.expect(std.math.isInf(huge));
    try std.testing.expect(overflowed.overflow);
    try std.testing.expect(overflowed.anySignificant());

    var divzero_probe = fpenv.Probe.begin();
    const inf = divF32(opaqueF32(1.0), opaqueF32(0.0));
    const divided = divzero_probe.end();
    try std.testing.expect(std.math.isInf(inf));
    try std.testing.expect(divided.division_by_zero);

    var invalid_probe = fpenv.Probe.begin();
    const nan = divF32(opaqueF32(0.0), opaqueF32(0.0));
    const invalid = invalid_probe.end();
    try std.testing.expect(std.math.isNan(nan));
    try std.testing.expect(invalid.invalid);

    fpenv.clearExceptions();
}

test "an exact computation raises nothing significant" {
    try requireSupport();
    var probe = fpenv.Probe.begin();
    // Powers of two add exactly: no rounding, no exception at all.
    const sum = mulF32(opaqueF32(2.0), opaqueF32(4.0));
    const raised = probe.end();
    try std.testing.expectEqual(@as(f32, 8.0), sum);
    try std.testing.expect(!raised.anySignificant());
    fpenv.clearExceptions();
}

test "nested probes do not lose the outer history" {
    try requireSupport();
    fpenv.clearExceptions();

    var outer = fpenv.Probe.begin();
    _ = divF32(opaqueF32(1.0), opaqueF32(0.0)); // raises division_by_zero

    var inner = fpenv.Probe.begin();
    _ = mulF32(opaqueF32(3.0e38), opaqueF32(3.0e38)); // raises overflow
    const inner_raised = inner.end();
    try std.testing.expect(inner_raised.overflow);
    // The inner probe saw only its own region.
    try std.testing.expect(!inner_raised.division_by_zero);

    const outer_raised = outer.end();
    // The outer probe still sees what happened before the inner one started.
    try std.testing.expect(outer_raised.division_by_zero);
    try std.testing.expect(outer_raised.overflow);

    fpenv.clearExceptions();
}

test "clearExceptions and raiseExceptions round-trip" {
    try requireSupport();
    fpenv.clearExceptions();
    try std.testing.expect(!fpenv.testExceptions().any());

    fpenv.raiseExceptions(.{ .overflow = true, .inexact = true });
    const read_back = fpenv.testExceptions();
    try std.testing.expect(read_back.overflow);
    try std.testing.expect(read_back.inexact);
    try std.testing.expect(!read_back.invalid);
    try std.testing.expect(!read_back.underflow);

    fpenv.clearExceptions();
    try std.testing.expect(!fpenv.testExceptions().any());
}

test "Exceptions set algebra" {
    const none: fpenv.Exceptions = .{};
    try std.testing.expect(!none.any());
    try std.testing.expect(!none.anySignificant());

    const inexact_only: fpenv.Exceptions = .{ .inexact = true };
    try std.testing.expect(inexact_only.any());
    // inexact alone is ordinary arithmetic, not a finding.
    try std.testing.expect(!inexact_only.anySignificant());

    const merged = inexact_only.unionWith(.{ .overflow = true });
    try std.testing.expect(merged.inexact and merged.overflow);
    try std.testing.expect(merged.anySignificant());
}

test "Exceptions formats as a flag list" {
    var buf: [64]u8 = undefined;
    const none = try std.fmt.bufPrint(&buf, "{f}", .{fpenv.Exceptions{}});
    try std.testing.expectEqualStrings("none", none);

    const some = try std.fmt.bufPrint(&buf, "{f}", .{
        fpenv.Exceptions{ .overflow = true, .inexact = true },
    });
    try std.testing.expectEqualStrings("overflow|inexact", some);
}

test "Guard and Probe are safe on an unsupported target" {
    // The whole facility must be callable unconditionally; only the answers
    // change. This test runs everywhere, including where `supported` is false.
    var guard = fpenv.Guard.begin();
    defer guard.restore();
    var probe = fpenv.Probe.begin();
    _ = probe.sample();
    _ = probe.end();
    fpenv.clearExceptions();
    try fpenv.assertDefault();
}
