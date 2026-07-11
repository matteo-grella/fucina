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

fn cleanupSidecar(path: []const u8) void {
    var buf: [160]u8 = undefined;
    const sidecar = std.fmt.bufPrint(&buf, "{s}.experts", .{path}) catch return;
    std.Io.Dir.cwd().deleteFile(std.testing.io, sidecar) catch {};
}

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

    /// Register this fixture's file layout (same geometry) on any store
    /// created over `self.path` — the reload/auto-pin tests build second
    /// stores against the same bytes.
    fn registerLayer(self: *const Fixture, store: *ExpertStore) !void {
        const gate_bytes = self.gate_blocks.len * @sizeOf(qm.BlockQ5_K);
        const up_bytes = self.up_blocks.len * @sizeOf(qm.BlockQ5_K);
        const down_bytes = self.down_blocks.len * @sizeOf(qm.BlockQ6_K);
        try store.addLayer(0, .{
            .{ .quant = .q5_k, .file_offset = 0, .byte_len = gate_bytes, .in_dim = hidden, .out_dim = out_pe },
            .{ .quant = .q5_k, .file_offset = gate_bytes, .byte_len = up_bytes, .in_dim = hidden, .out_dim = out_pe },
            .{ .quant = .q6_k, .file_offset = gate_bytes + up_bytes, .byte_len = down_bytes, .in_dim = out_pe, .out_dim = hidden },
        }, n_expert);
    }

    /// One single-token decode through `store`'s streamed arms; asserts the
    /// output is bitwise-equal to the resident path.
    fn expectDecodeWith(self: *Fixture, ctx: *ExecContext, store: *ExpertStore, selected: []const usize, weights: []const f32) !void {
        var gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
        var up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
        var down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };
        const x_vals = try self.allocator.alloc(f32, hidden);
        defer self.allocator.free(x_vals);
        for (x_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 13) % 199)) - 99);
        var x = try ctx.fromSliceRank(2, .{ 1, hidden }, x_vals);
        defer x.deinit();

        var want = try ctx.moeExpertFfn(&x, &self.resident_gate, &self.resident_up, &self.resident_down, selected, weights, out_pe, .swiglu, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &gate, &up, &down, selected, weights, out_pe, .swiglu, null, null);
        defer got.deinit();
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    }

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
        self.store = try ExpertStore.create(allocator, &.{self.path}, 1, .{ .cache_slots_per_layer = cache_slots });
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
        cleanupSidecar(self.path);
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
    var store2 = try ExpertStore.create(allocator, &.{fx.path}, 1, .{ .cache_slots_per_layer = 1 });
    defer store2.destroy();
    try std.testing.expectError(error.InvalidExpertGeometry, store2.addLayer(0, .{
        .{ .quant = .q5_k, .file_offset = 0, .byte_len = 12345, .in_dim = hidden, .out_dim = out_pe },
        .{ .quant = .q5_k, .file_offset = 0, .byte_len = 12345, .in_dim = hidden, .out_dim = out_pe },
        .{ .quant = .q6_k, .file_offset = 0, .byte_len = 12345, .in_dim = out_pe, .out_dim = hidden },
    }, n_expert));
    try std.testing.expectError(error.StoreNotFinalized, store2.acquire(0, &.{0}));
}

test "pilot: hints spin up the I/O thread and prediction recall is scored on acquire" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 2);
    defer fx.deinit();

    // Predict {5, 6, 3}, then route to {5, 6}: recall = 2/2 (both routed
    // experts were predicted; the over-prediction of 3 is bandwidth, not a
    // recall miss). All three were uncached -> 9 ranges enqueued (3 projs).
    fx.store.pilotHint(0, &.{ 5, 6, 3 });
    try std.testing.expectEqual(@as(u64, 9), fx.store.stats.pilot_ranges);
    try fx.expectDecodeMatches(&ctx, &.{ 5, 6 }, &.{ 0.7, 0.3 });
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.pilot_recall_hits);
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.pilot_recall_total);

    // One prediction scores exactly one acquire: a second decode on the
    // same layer does not double-count.
    try fx.expectDecodeMatches(&ctx, &.{ 5, 6 }, &.{ 0.7, 0.3 });
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.pilot_recall_total);

    // Cached/pinned experts are not re-hinted; a wrong prediction scores 0.
    fx.store.pilotHint(0, &.{ 5, 1 }); // 5 is now cached: only 1 enqueues
    try std.testing.expectEqual(@as(u64, 12), fx.store.stats.pilot_ranges);
    try fx.expectDecodeMatches(&ctx, &.{ 0, 2 }, &.{ 0.5, 0.5 });
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.pilot_recall_hits);
    try std.testing.expectEqual(@as(u64, 4), fx.store.stats.pilot_recall_total);
}

test "q8_0 experts with non-256-aligned dims: streamed decode is bit-exact vs resident" {
    // The deepseek2 shape: K-quant gate/up over a 256-aligned hidden, q8_0
    // down whose input width (the expert FFN dim) is only 32-aligned — the
    // fused decode op must produce Q8_K activations for the K-quant arms
    // and Q8_0 activations for the q8_0 arm in the same pass.
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const ds_hidden: usize = 256;
    const ds_ffn: usize = 96; // 3 q8_0 blocks; NOT a K-quant multiple
    const ds_experts: usize = 4;

    // gate/up: q5_k stacks [experts * ffn rows, hidden].
    const gu_rows = ds_experts * ds_ffn;
    const gate_blocks = try allocator.alloc(qm.BlockQ5_K, gu_rows * (ds_hidden / qm.qk_k_block_size));
    defer allocator.free(gate_blocks);
    const up_blocks = try allocator.alloc(qm.BlockQ5_K, gate_blocks.len);
    defer allocator.free(up_blocks);
    // Mild scales: the SwiGLU square of these activations must stay well
    // inside Q8_0's f16 scale range (see the NaN guard below).
    fillQ5KBlocks(gate_blocks, 3);
    fillQ5KBlocks(up_blocks, 4);
    for (gate_blocks) |*b| {
        b.dm[0] = f16Bits(0.0004);
        b.dm[1] = f16Bits(0.0002);
    }
    for (up_blocks) |*b| {
        b.dm[0] = f16Bits(0.0005);
        b.dm[1] = f16Bits(0.0002);
    }

    // down: q8_0 stack [experts * hidden rows, ffn] built by quantizing
    // deterministic f32 rows (valid blocks by construction).
    const down_rows = ds_experts * ds_hidden;
    const down_bpc = ds_ffn / 32;
    const down_blocks = try allocator.alloc(qm.BlockQ8_0, down_rows * down_bpc);
    defer allocator.free(down_blocks);
    {
        var row: [ds_ffn]f32 = undefined;
        for (0..down_rows) |r| {
            for (&row, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(r * 31 + i)) * 0.11) * 1.7;
            try qm.quantizeRowQ8_0Into(down_blocks[r * down_bpc ..][0..down_bpc], &row);
        }
    }

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "expert_store_q8_{d}.bin", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(std.mem.sliceAsBytes(gate_blocks));
        try writer.interface.writeAll(std.mem.sliceAsBytes(up_blocks));
        try writer.interface.writeAll(std.mem.sliceAsBytes(down_blocks));
        try writer.interface.flush();
    }
    defer {
        var buf: [160]u8 = undefined;
        const sidecar = std.fmt.bufPrint(&buf, "{s}.experts", .{path}) catch unreachable;
        std.Io.Dir.cwd().deleteFile(std.testing.io, sidecar) catch {};
    }

    var resident_gate: MoeRhs = .{ .q5_k = try qm.quantizedMatmulRhsQ5_KFromBlocks(allocator, ds_hidden, gu_rows, gate_blocks) };
    defer resident_gate.deinit();
    var resident_up: MoeRhs = .{ .q5_k = try qm.quantizedMatmulRhsQ5_KFromBlocks(allocator, ds_hidden, gu_rows, up_blocks) };
    defer resident_up.deinit();
    var resident_down: MoeRhs = .{ .q8_0 = .{
        .rows = .{ .allocator = null, .blocks = down_blocks, .rows = down_rows, .cols = ds_ffn, .blocks_per_row = down_bpc },
        .k = ds_ffn,
        .n = down_rows,
    } };
    defer resident_down.deinit();

    const gate_bytes = gate_blocks.len * @sizeOf(qm.BlockQ5_K);
    const up_bytes = up_blocks.len * @sizeOf(qm.BlockQ5_K);
    const down_bytes = down_blocks.len * @sizeOf(qm.BlockQ8_0);
    var store = try ExpertStore.create(allocator, &.{path}, 1, .{ .cache_slots_per_layer = 2 });
    defer store.destroy();
    try store.addLayer(0, .{
        .{ .quant = .q5_k, .file_offset = 0, .byte_len = gate_bytes, .in_dim = ds_hidden, .out_dim = ds_ffn },
        .{ .quant = .q5_k, .file_offset = gate_bytes, .byte_len = up_bytes, .in_dim = ds_hidden, .out_dim = ds_ffn },
        .{ .quant = .q8_0, .file_offset = gate_bytes + up_bytes, .byte_len = down_bytes, .in_dim = ds_ffn, .out_dim = ds_hidden },
    }, ds_experts);
    try store.finalize();
    var streamed_gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
    var streamed_up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
    var streamed_down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };

    const x_vals = try allocator.alloc(f32, ds_hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 151)) - 75)) / 75.0;
    var x = try ctx.fromSliceRank(2, .{ 1, ds_hidden }, x_vals);
    defer x.deinit();

    // Cold, warm, and evicting decodes must all match the resident path
    // bit-for-bit.
    for ([_][2]usize{ .{ 0, 3 }, .{ 0, 3 }, .{ 1, 2 }, .{ 0, 3 } }) |pair| {
        var want = try ctx.moeExpertFfn(&x, &resident_gate, &resident_up, &resident_down, &pair, &.{ 0.6, 0.4 }, ds_ffn, .swiglu, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &streamed_gate, &streamed_up, &streamed_down, &pair, &.{ 0.6, 0.4 }, ds_ffn, .swiglu, null, null);
        defer got.deinit();
        // A sanity guard on the fixture itself: Q8_0's f16 block scale
        // overflows past |activation| ~8.3e6, which would NaN both paths
        // and vacuously "match".
        for (want.dataConst()) |v| try std.testing.expect(!std.math.isNan(v));
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    }
}

test "tq2_0 ternary experts: streamed decode and batch are bit-exact vs resident" {
    // The PTQTP/ternary tier: all three projections TQ2_0 (256-elem blocks,
    // Q8_K activations like the K-quants). Streamed must match resident
    // bit-for-bit on both the decode GEMV and the batched-prefill path
    // (tq2_0 is not lane-packed, so batch flows through the row tile).
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const t_hidden: usize = 256;
    const t_ffn: usize = 512;
    const t_experts: usize = 4;

    const gu_rows = t_experts * t_ffn;
    const gu_bpc = t_hidden / qm.qk_k_block_size;
    const down_rows = t_experts * t_hidden;
    const down_bpc = t_ffn / qm.qk_k_block_size;
    const gate_blocks = try allocator.alloc(qm.BlockTQ2_0, gu_rows * gu_bpc);
    defer allocator.free(gate_blocks);
    const up_blocks = try allocator.alloc(qm.BlockTQ2_0, gate_blocks.len);
    defer allocator.free(up_blocks);
    const down_blocks = try allocator.alloc(qm.BlockTQ2_0, down_rows * down_bpc);
    defer allocator.free(down_blocks);
    {
        var row: [t_hidden]f32 = undefined;
        for (0..gu_rows) |r| {
            for (&row, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(r * 13 + i)) * 0.23) * 0.9;
            try qm.quantizeRowTQ2_0Into(gate_blocks[r * gu_bpc ..][0..gu_bpc], &row);
            for (&row, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(r * 7 + i)) * 0.31) * 1.1;
            try qm.quantizeRowTQ2_0Into(up_blocks[r * gu_bpc ..][0..gu_bpc], &row);
        }
        var drow: [t_ffn]f32 = undefined;
        for (0..down_rows) |r| {
            for (&drow, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(r * 31 + i)) * 0.11) * 0.8;
            try qm.quantizeRowTQ2_0Into(down_blocks[r * down_bpc ..][0..down_bpc], &drow);
        }
    }

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "expert_store_tq2_{d}.bin", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(std.mem.sliceAsBytes(gate_blocks));
        try writer.interface.writeAll(std.mem.sliceAsBytes(up_blocks));
        try writer.interface.writeAll(std.mem.sliceAsBytes(down_blocks));
        try writer.interface.flush();
    }
    defer {
        var buf: [160]u8 = undefined;
        const sidecar = std.fmt.bufPrint(&buf, "{s}.experts", .{path}) catch unreachable;
        std.Io.Dir.cwd().deleteFile(std.testing.io, sidecar) catch {};
    }

    const tq2View = struct {
        fn go(blocks: []qm.BlockTQ2_0, k: usize, rows: usize, bpc: usize) MoeRhs {
            return .{ .tq2_0 = .{
                .rows = .{ .allocator = null, .blocks = blocks, .rows = rows, .cols = k, .blocks_per_row = bpc },
                .k = k,
                .n = rows,
            } };
        }
    }.go;
    var resident_gate = tq2View(gate_blocks, t_hidden, gu_rows, gu_bpc);
    defer resident_gate.deinit();
    var resident_up = tq2View(up_blocks, t_hidden, gu_rows, gu_bpc);
    defer resident_up.deinit();
    var resident_down = tq2View(down_blocks, t_ffn, down_rows, down_bpc);
    defer resident_down.deinit();

    const gate_bytes = gate_blocks.len * @sizeOf(qm.BlockTQ2_0);
    const up_bytes = up_blocks.len * @sizeOf(qm.BlockTQ2_0);
    const down_bytes = down_blocks.len * @sizeOf(qm.BlockTQ2_0);
    var store = try ExpertStore.create(allocator, &.{path}, 1, .{ .cache_slots_per_layer = 2 });
    defer store.destroy();
    try store.addLayer(0, .{
        .{ .quant = .tq2_0, .file_offset = 0, .byte_len = gate_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .tq2_0, .file_offset = gate_bytes, .byte_len = up_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .tq2_0, .file_offset = gate_bytes + up_bytes, .byte_len = down_bytes, .in_dim = t_ffn, .out_dim = t_hidden },
    }, t_experts);
    try store.finalize();
    var streamed_gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
    var streamed_up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
    var streamed_down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };

    // Decode: cold, warm, and evicting acquires.
    const x_vals = try allocator.alloc(f32, t_hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11) % 173)) - 86)) / 86.0;
    var x = try ctx.fromSliceRank(2, .{ 1, t_hidden }, x_vals);
    defer x.deinit();
    for ([_][2]usize{ .{ 0, 3 }, .{ 0, 3 }, .{ 1, 2 }, .{ 0, 3 } }) |pair| {
        var want = try ctx.moeExpertFfn(&x, &resident_gate, &resident_up, &resident_down, &pair, &.{ 0.6, 0.4 }, t_ffn, .swiglu, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &streamed_gate, &streamed_up, &streamed_down, &pair, &.{ 0.6, 0.4 }, t_ffn, .swiglu, null, null);
        defer got.deinit();
        for (want.dataConst()) |v| try std.testing.expect(!std.math.isNan(v));
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    }

    // Batched prefill (m = 5 rows spanning all experts).
    const m: usize = 5;
    const xb_vals = try allocator.alloc(f32, m * t_hidden);
    defer allocator.free(xb_vals);
    for (xb_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 5) % 199)) - 99)) / 99.0;
    var xb = try ctx.fromSliceRank(2, .{ m, t_hidden }, xb_vals);
    defer xb.deinit();
    const selected = [_]usize{ 0, 3, 1, 2, 2, 0, 3, 1, 0, 1 };
    const routing = [_]f32{ 0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.2, 0.8, 0.9, 0.1 };
    var want_b = try ctx.moeExpertFfnBatch(&xb, &resident_gate, &resident_up, &resident_down, &selected, &routing, 2, t_ffn, .swiglu, null, null);
    defer want_b.deinit();
    var got_b = try ctx.moeExpertFfnBatch(&xb, &streamed_gate, &streamed_up, &streamed_down, &selected, &routing, 2, t_ffn, .swiglu, null, null);
    defer got_b.deinit();
    for (want_b.dataConst()) |v| try std.testing.expect(!std.math.isNan(v));
    try std.testing.expectEqualSlices(f32, want_b.dataConst(), got_b.dataConst());
}

test "learning cache: saved usage auto-pins the hot experts on reload, bit-exact and miss-free" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 2);
    defer fx.deinit();

    // Session 1: route consistently to {5, 6}, persist the histogram.
    for (0..3) |_| try fx.expectDecodeMatches(&ctx, &.{ 5, 6 }, &.{ 0.7, 0.3 });
    try fx.store.saveUsage();

    // Session 2 (fresh store, same file): history qualifies, budget fits
    // exactly two pinned experts -> 5 and 6 are read at finalize and every
    // decode routed to them is a pure pin hit.
    var store2 = try ExpertStore.create(allocator, &.{fx.path}, 1, .{
        .cache_slots_per_layer = 1,
        .auto_pin_min_history = 1,
    });
    defer store2.destroy();
    try fx.registerLayer(store2);
    store2.options.pin_bytes = 2 * store2.layers[0].slab_bytes;
    try store2.finalize();
    try std.testing.expectEqual(@as(usize, 2), store2.pinned_experts);

    try fx.expectDecodeWith(&ctx, store2, &.{ 5, 6 }, &.{ 0.7, 0.3 });
    try std.testing.expectEqual(@as(u64, 0), store2.stats.misses);
    try std.testing.expectEqual(@as(u64, 2), store2.stats.pin_hits);

    // A histogram from a different geometry is ignored wholesale: a store
    // pretending the model has more layers loads nothing and pins nothing.
    var store3 = try ExpertStore.create(allocator, &.{fx.path}, 2, .{
        .cache_slots_per_layer = 1,
        .auto_pin_min_history = 1,
    });
    defer store3.destroy();
    try fx.registerLayer(store3);
    try store3.finalize();
    try std.testing.expectEqual(@as(usize, 0), store3.pinned_experts);
}

test "learning cache: repin pass swaps cold pins for hot streamed experts with hysteresis" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 2);
    defer fx.deinit();

    // Pin {5, 6} via saved history (as above).
    for (0..3) |_| try fx.expectDecodeMatches(&ctx, &.{ 5, 6 }, &.{ 0.7, 0.3 });
    try fx.store.saveUsage();
    var store2 = try ExpertStore.create(allocator, &.{fx.path}, 1, .{
        .cache_slots_per_layer = 1,
        .auto_pin_min_history = 1,
    });
    defer store2.destroy();
    try fx.registerLayer(store2);
    store2.options.pin_bytes = 2 * store2.layers[0].slab_bytes;
    try store2.finalize();
    try std.testing.expectEqual(@as(usize, 2), store2.pinned_experts);

    // Below the hysteresis margin (fixed +4 with zero pinned heat) nothing
    // swaps: 4 routed pairs of heat are not enough evidence.
    for (0..4) |_| try fx.expectDecodeWith(&ctx, store2, &.{ 1, 2 }, &.{ 0.5, 0.5 });
    try std.testing.expectEqual(@as(usize, 0), store2.repinPass(4));

    // Past it (heat halved to 2 by the failed pass, then +5 = 7 > 4), both
    // cold pins swap to the new hot pair; decode then pin-hits and stays
    // bit-exact.
    for (0..5) |_| try fx.expectDecodeWith(&ctx, store2, &.{ 1, 2 }, &.{ 0.5, 0.5 });
    try std.testing.expectEqual(@as(usize, 2), store2.repinPass(4));
    const pin_hits_before = store2.stats.pin_hits;
    try fx.expectDecodeWith(&ctx, store2, &.{ 1, 2 }, &.{ 0.5, 0.5 });
    try std.testing.expectEqual(pin_hits_before + 2, store2.stats.pin_hits);
}
