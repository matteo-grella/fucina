//! Behavioral tests for the disk-streamed MoE expert tier
//! (`exec/expert_store.zig` + the `MoeRhs.streamed` arm): the streamed path
//! must be BIT-EXACT vs the resident path (same blocks, same kernels — any
//! difference is a resolve/geometry bug), across cold misses, warm hits, LRU
//! eviction, and batched prefill whose active set overflows the cache. Plus
//! store lifecycle/geometry validation.
const std = @import("std");
const backend_mod = @import("../backend.zig");
const exec = @import("../exec.zig");
const expert_store = @import("expert_store.zig");

const qm = backend_mod.quantized_matmul;
const ExecContext = exec.ExecContext;
const MoeRhs = ExecContext.MoeRhs;
const ExpertStore = expert_store.ExpertStore;

const hidden: usize = 256; // one Q8_K superblock per row
const out_pe: usize = 256;
const n_expert: usize = 8;

fn f16Bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

// Valid-domain deterministic block patterns, mirroring the batched-MoE
// fixtures in exec_tests.zig; `seed` differentiates gate/up/down.
fn fillQ5KBlocks(blocks: []qm.BlockQ5_K, seed: usize) void {
    for (blocks, 0..) |*b, block_i| {
        const bi = block_i + seed;
        b.dm[0] = f16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
}

fn fillQ6KBlocks(blocks: []qm.BlockQ6_K, seed: usize) void {
    for (blocks, 0..) |*b, block_i| {
        const bi = block_i + seed;
        b.d = f16Bits(0.04 + 0.001 * @as(f32, @floatFromInt(bi % 5)));
        for (&b.scales, 0..) |*s, i| s.* = @intCast(@as(i32, @intCast((i * 5 + bi) % 15)) - 7);
        for (&b.ql, 0..) |*q, i| q.* = @intCast((i * 19 + bi * 7) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 23 + bi * 3) % 256);
    }
}

/// Both views of the same expert stacks: resident MoeRhs arms built from the
/// block arrays, and an ExpertStore streaming the identical bytes from a
/// temp file (gate/up q5_k, down q6_k — mixed per-projection quants, like
/// real MoE GGUFs).
const Fixture = struct {
    allocator: std.mem.Allocator,
    path_buf: [128]u8 = undefined,
    path: []const u8 = &.{},
    gate_blocks: []qm.BlockQ5_K,
    up_blocks: []qm.BlockQ5_K,
    down_blocks: []qm.BlockQ6_K,
    resident_gate: MoeRhs,
    resident_up: MoeRhs,
    resident_down: MoeRhs,
    store: *ExpertStore,
    streamed_gate: MoeRhs,
    streamed_up: MoeRhs,
    streamed_down: MoeRhs,

    fn init(self: *Fixture, allocator: std.mem.Allocator, cache_slots: usize) !void {
        const gate_rows = n_expert * out_pe;
        const down_rows = n_expert * hidden;
        const bpc_in = hidden / qm.qk_k_block_size;
        const bpc_g = out_pe / qm.qk_k_block_size;

        self.allocator = allocator;
        self.gate_blocks = try allocator.alloc(qm.BlockQ5_K, gate_rows * bpc_in);
        self.up_blocks = try allocator.alloc(qm.BlockQ5_K, gate_rows * bpc_in);
        self.down_blocks = try allocator.alloc(qm.BlockQ6_K, down_rows * bpc_g);
        fillQ5KBlocks(self.gate_blocks, 0);
        fillQ5KBlocks(self.up_blocks, 1);
        fillQ6KBlocks(self.down_blocks, 2);

        self.path = try std.fmt.bufPrint(&self.path_buf, "expert_store_test_{d}.bin", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
        {
            var file = try std.Io.Dir.cwd().createFile(std.testing.io, self.path, .{});
            defer file.close(std.testing.io);
            var write_buffer: [4096]u8 = undefined;
            var writer = file.writer(std.testing.io, &write_buffer);
            try writer.interface.writeAll(std.mem.sliceAsBytes(self.gate_blocks));
            try writer.interface.writeAll(std.mem.sliceAsBytes(self.up_blocks));
            try writer.interface.writeAll(std.mem.sliceAsBytes(self.down_blocks));
            try writer.interface.flush();
        }

        self.resident_gate = .{ .q5_k = try qm.quantizedMatmulRhsQ5_KFromBlocks(allocator, hidden, gate_rows, self.gate_blocks) };
        self.resident_up = .{ .q5_k = try qm.quantizedMatmulRhsQ5_KFromBlocks(allocator, hidden, gate_rows, self.up_blocks) };
        self.resident_down = .{ .q6_k = try qm.quantizedMatmulRhsQ6_KFromBlocks(allocator, out_pe, down_rows, self.down_blocks) };

        const gate_bytes = self.gate_blocks.len * @sizeOf(qm.BlockQ5_K);
        const up_bytes = self.up_blocks.len * @sizeOf(qm.BlockQ5_K);
        const down_bytes = self.down_blocks.len * @sizeOf(qm.BlockQ6_K);
        self.store = try ExpertStore.create(allocator, self.path, 1, .{ .cache_slots_per_layer = cache_slots });
        try self.store.addLayer(0, .{
            .{ .quant = .q5_k, .file_offset = 0, .byte_len = gate_bytes, .in_dim = hidden, .out_dim = out_pe },
            .{ .quant = .q5_k, .file_offset = gate_bytes, .byte_len = up_bytes, .in_dim = hidden, .out_dim = out_pe },
            .{ .quant = .q6_k, .file_offset = gate_bytes + up_bytes, .byte_len = down_bytes, .in_dim = out_pe, .out_dim = hidden },
        }, n_expert);
        try self.store.finalize();
        self.streamed_gate = .{ .streamed = self.store.streamedRhs(0, .gate) };
        self.streamed_up = .{ .streamed = self.store.streamedRhs(0, .up) };
        self.streamed_down = .{ .streamed = self.store.streamedRhs(0, .down) };
    }

    fn deinit(self: *Fixture) void {
        self.store.destroy();
        self.resident_down.deinit();
        self.resident_up.deinit();
        self.resident_gate.deinit();
        self.allocator.free(self.down_blocks);
        self.allocator.free(self.up_blocks);
        self.allocator.free(self.gate_blocks);
        std.Io.Dir.cwd().deleteFile(std.testing.io, self.path) catch {};
    }

    /// One single-token decode on both paths; asserts bitwise-equal outputs.
    fn expectDecodeMatches(self: *Fixture, ctx: *ExecContext, selected: []const usize, weights: []const f32) !void {
        const x_vals = try self.allocator.alloc(f32, hidden);
        defer self.allocator.free(x_vals);
        for (x_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 13) % 199)) - 99);
        var x = try ctx.fromSliceRank(2, .{ 1, hidden }, x_vals);
        defer x.deinit();

        var want = try ctx.moeExpertFfn(&x, &self.resident_gate, &self.resident_up, &self.resident_down, selected, weights, out_pe, .swiglu, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &self.streamed_gate, &self.streamed_up, &self.streamed_down, selected, weights, out_pe, .swiglu, null, null);
        defer got.deinit();
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    }
};

test "streamed MoE decode is bit-exact vs resident across cold, warm, and evicting acquires" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 2); // cap 2: the third distinct pair evicts
    defer fx.deinit();

    // Cold: both experts read from disk.
    try fx.expectDecodeMatches(&ctx, &.{ 0, 3 }, &.{ 0.6, 0.4 });
    try std.testing.expectEqual(@as(u64, 0), fx.store.stats.hits);
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.misses);

    // Warm: same pair — pure cache hits, still exact.
    try fx.expectDecodeMatches(&ctx, &.{ 0, 3 }, &.{ 0.6, 0.4 });
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.hits);
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.misses);

    // Different pair: misses, and its promotion evicts {0, 3} (cap 2).
    try fx.expectDecodeMatches(&ctx, &.{ 1, 2 }, &.{ 0.5, 0.5 });
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.hits);
    try std.testing.expectEqual(@as(u64, 4), fx.store.stats.misses);

    // Evicted pair again: misses again, output still exact.
    try fx.expectDecodeMatches(&ctx, &.{ 0, 3 }, &.{ 0.6, 0.4 });
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.hits);
    try std.testing.expectEqual(@as(u64, 6), fx.store.stats.misses);
}

test "streamed MoE batched prefill is bit-exact when the active set overflows the cache" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 2); // all 8 experts active >> cap 2: working-set overflow
    defer fx.deinit();

    const seq: usize = 32;
    const top_k: usize = 2; // 64 pairs: the phased chain path engages
    const x_vals = try allocator.alloc(f32, seq * hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var x = try ctx.fromSliceRank(2, .{ seq, hidden }, x_vals);
    defer x.deinit();

    var selected: [seq * top_k]usize = undefined;
    var weights: [seq * top_k]f32 = undefined;
    for (&selected, &weights, 0..) |*s, *w, p| {
        s.* = (p * 5) % n_expert;
        w.* = 0.25 + 0.01 * @as(f32, @floatFromInt(p % 13));
    }

    var want = try ctx.moeExpertFfnBatch(&x, &fx.resident_gate, &fx.resident_up, &fx.resident_down, &selected, &weights, top_k, out_pe, .swiglu, null, null);
    defer want.deinit();
    var got = try ctx.moeExpertFfnBatch(&x, &fx.streamed_gate, &fx.streamed_up, &fx.streamed_down, &selected, &weights, top_k, out_pe, .swiglu, null, null);
    defer got.deinit();
    try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());

    // The batch touched all 8 experts once each (batch-union), promoted 2.
    try std.testing.expectEqual(@as(u64, 8), fx.store.stats.misses);

    // A follow-up decode routed to the promoted experts is warm and exact.
    const before_hits = fx.store.stats.hits;
    try fx.expectDecodeMatches(&ctx, &.{ 6, 7 }, &.{ 0.5, 0.5 });
    try std.testing.expect(fx.store.stats.hits > before_hits);
}

test "expert store validates geometry and lifecycle" {
    const allocator = std.testing.allocator;

    var fx: Fixture = undefined;
    try fx.init(allocator, 1);
    defer fx.deinit();

    // Double registration of the same layer.
    try std.testing.expectError(error.LayerAlreadyRegistered, fx.store.addLayer(0, .{
        .{ .quant = .q5_k, .file_offset = 0, .byte_len = 1, .in_dim = hidden, .out_dim = out_pe },
        .{ .quant = .q5_k, .file_offset = 0, .byte_len = 1, .in_dim = hidden, .out_dim = out_pe },
        .{ .quant = .q6_k, .file_offset = 0, .byte_len = 1, .in_dim = out_pe, .out_dim = hidden },
    }, n_expert));

    // Geometry that disagrees with the tensor's byte length, and an
    // unfinalized store refusing to acquire.
    var store2 = try ExpertStore.create(allocator, fx.path, 1, .{ .cache_slots_per_layer = 1 });
    defer store2.destroy();
    try std.testing.expectError(error.InvalidExpertGeometry, store2.addLayer(0, .{
        .{ .quant = .q5_k, .file_offset = 0, .byte_len = 12345, .in_dim = hidden, .out_dim = out_pe },
        .{ .quant = .q5_k, .file_offset = 0, .byte_len = 12345, .in_dim = hidden, .out_dim = out_pe },
        .{ .quant = .q6_k, .file_offset = 0, .byte_len = 12345, .in_dim = out_pe, .out_dim = hidden },
    }, n_expert));
    try std.testing.expectError(error.StoreNotFinalized, store2.acquire(0, &.{0}));
}
