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
pub fn buildTestMoeRhsQ5K(allocator: Allocator, rows: usize, k_dim: usize, seed: usize) !exec.ExecContext.MoeRhs {
    const qm = backend_mod.quantized_matmul;
    const bpc = k_dim / qm.types.qk_k_block_size;
    const blocks = try allocator.alloc(dtype_mod.BlockQ5_K, rows * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, block_i| {
        const bi = block_i + seed;
        b.dm[0] = f16BitsFromF32(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f16BitsFromF32(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    return .{ .q5_k = try qm.q8k.quantizedMatmulRhsQ5_KFromBlocks(allocator, k_dim, rows, blocks) };
}
