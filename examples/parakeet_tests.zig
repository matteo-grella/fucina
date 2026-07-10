//! Tests for examples/parakeet.zig. Hermetic (no fixtures): the `--compare`
//! parity gate (`compareGate`) must be mechanically enforcing — return `false`
//! when a comparison exceeds its tolerance / cosine bound and `true` otherwise,
//! so `main` can turn a gated `--compare` FAIL into a nonzero exit.
const std = @import("std");
const parakeet = @import("parakeet.zig");

test "compareGate enforces the max-abs tolerance" {
    // Identical -> within any tol.
    try std.testing.expect(parakeet.compareGate(&.{ 1, 2, 3 }, &.{ 1, 2, 3 }, 1e-6, null));
    // max_abs = 10 > tol -> FAIL.
    try std.testing.expect(!parakeet.compareGate(&.{ 0, 0, 0 }, &.{ 0, 10, 0 }, 1e-6, null));
    // max_abs = 0.5, just under a generous tol -> PASS.
    try std.testing.expect(parakeet.compareGate(&.{1.0}, &.{1.5}, 1.0, null));
}

test "compareGate enforces the cosine bound for intermediate stages" {
    // Same direction -> cosine 1.0 >= 0.9999 -> PASS (tol null = informational).
    try std.testing.expect(parakeet.compareGate(&.{ 1, 2, 3 }, &.{ 2, 4, 6 }, null, 0.9999));
    // Orthogonal -> cosine 0 < 0.9999 -> FAIL.
    try std.testing.expect(!parakeet.compareGate(&.{ 1, 0 }, &.{ 0, 1 }, null, 0.9999));
}

test "compareGate combines tol AND cosine gates" {
    // Passes tol but fails cosine -> overall FAIL.
    try std.testing.expect(!parakeet.compareGate(&.{ 1, 0 }, &.{ 1.0000001, 0.5 }, 1.0, 0.9999));
    // No gates -> vacuously true.
    try std.testing.expect(parakeet.compareGate(&.{ 1, 2 }, &.{ 9, 9 }, null, null));
}
