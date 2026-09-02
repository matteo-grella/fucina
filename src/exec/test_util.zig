//! Shared fixtures for the exec test files: f64 closeness checks, f16 bit
//! packing, and the deterministic Q5_K MoE RHS builder.

const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const backend_mod = @import("../backend.zig");
const exec = @import("../exec.zig");
const Allocator = std.mem.Allocator;
const ExecContext = exec.ExecContext;

pub fn expectCloseToF64(want: f64, got: f32, rtol: f64, atol: f64) !void {
    const diff = @abs(@as(f64, got) - want);
    if (diff <= atol or diff <= rtol * @abs(want)) return;
    std.debug.print("expected {d}, got {d} (diff {d})\n", .{ want, got, diff });
    return error.TestUnexpectedResult;
}

pub fn f16BitsFromF32(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

// Deterministic valid-domain Q5_K expert stack for the batched-MoE tests:
// `rows` stacked expert columns of `k_dim` features, same block pattern as
// the q5_k kernel-test fixtures, offset by `seed` so gate/up/down differ.
