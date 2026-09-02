//! Test helpers of the MoE band.
const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const backend_mod = @import("../backend.zig");
const moe = @import("../moe.zig");
const exec_util = @import("../exec/test_util.zig");

const Allocator = std.mem.Allocator;
const f16BitsFromF32 = exec_util.f16BitsFromF32;

pub fn buildTestMoeRhsQ5K(allocator: Allocator, rows: usize, k_dim: usize, seed: usize) !moe.MoeRhs {
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
