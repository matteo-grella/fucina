//! Behavioral tests for the disk-streamed MoE expert tier
//! (`store/expert_store.zig` + the `MoeRhs.streamed` arm): the streamed path
//! must be BIT-EXACT vs the resident path (same blocks, same kernels — any
//! difference is a resolve/geometry bug), across cold misses, warm hits, LRU
//! eviction, and batched prefill whose active set overflows the cache. Plus
//! store lifecycle/geometry validation.
const std = @import("std");
const dtype_mod = @import("../dtype.zig");
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
fn fillQ5KBlocks(blocks: []dtype_mod.BlockQ5_K, seed: usize) void {
    for (blocks, 0..) |*b, block_i| {
        const bi = block_i + seed;
        b.dm[0] = f16Bits(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f16Bits(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
}

fn fillQ6KBlocks(blocks: []dtype_mod.BlockQ6_K, seed: usize) void {
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
    gate_blocks: []dtype_mod.BlockQ5_K,
    up_blocks: []dtype_mod.BlockQ5_K,
    down_blocks: []dtype_mod.BlockQ6_K,
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
        const gate_bytes = self.gate_blocks.len * @sizeOf(dtype_mod.BlockQ5_K);
        const up_bytes = self.up_blocks.len * @sizeOf(dtype_mod.BlockQ5_K);
        const down_bytes = self.down_blocks.len * @sizeOf(dtype_mod.BlockQ6_K);
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
        var x = try ctx.fromSlice(.f32, .{ 1, hidden }, x_vals);
        defer x.deinit();

        var want = try ctx.moeExpertFfn(&x, &self.resident_gate, &self.resident_up, &self.resident_down, selected, weights, out_pe, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &gate, &up, &down, selected, weights, out_pe, .{ .op = .swiglu }, null, null);
        defer got.deinit();
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    }

    fn init(self: *Fixture, allocator: std.mem.Allocator, cache_slots: usize) !void {
        const gate_rows = n_expert * out_pe;
        const down_rows = n_expert * hidden;
        const bpc_in = hidden / qm.types.qk_k_block_size;
        const bpc_g = out_pe / qm.types.qk_k_block_size;

        self.allocator = allocator;
        self.gate_blocks = try allocator.alloc(dtype_mod.BlockQ5_K, gate_rows * bpc_in);
        self.up_blocks = try allocator.alloc(dtype_mod.BlockQ5_K, gate_rows * bpc_in);
        self.down_blocks = try allocator.alloc(dtype_mod.BlockQ6_K, down_rows * bpc_g);
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

        self.resident_gate = .{ .q5_k = try qm.q8k.quantizedMatmulRhsQ5_KFromBlocks(allocator, hidden, gate_rows, self.gate_blocks) };
        self.resident_up = .{ .q5_k = try qm.q8k.quantizedMatmulRhsQ5_KFromBlocks(allocator, hidden, gate_rows, self.up_blocks) };
        self.resident_down = .{ .q6_k = try qm.q8k.quantizedMatmulRhsQ6_KFromBlocks(allocator, out_pe, down_rows, self.down_blocks) };

        const gate_bytes = self.gate_blocks.len * @sizeOf(dtype_mod.BlockQ5_K);
        const up_bytes = self.up_blocks.len * @sizeOf(dtype_mod.BlockQ5_K);
        const down_bytes = self.down_blocks.len * @sizeOf(dtype_mod.BlockQ6_K);
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
        var x = try ctx.fromSlice(.f32, .{ 1, hidden }, x_vals);
        defer x.deinit();

        var want = try ctx.moeExpertFfn(&x, &self.resident_gate, &self.resident_up, &self.resident_down, selected, weights, out_pe, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &self.streamed_gate, &self.streamed_up, &self.streamed_down, selected, weights, out_pe, .{ .op = .swiglu }, null, null);
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
    var x = try ctx.fromSlice(.f32, .{ seq, hidden }, x_vals);
    defer x.deinit();

    var selected: [seq * top_k]usize = undefined;
    var weights: [seq * top_k]f32 = undefined;
    for (&selected, &weights, 0..) |*s, *w, p| {
        s.* = (p * 5) % n_expert;
        w.* = 0.25 + 0.01 * @as(f32, @floatFromInt(p % 13));
    }

    var want = try ctx.moeExpertFfnBatch(&x, &fx.resident_gate, &fx.resident_up, &fx.resident_down, &selected, &weights, top_k, out_pe, .{ .op = .swiglu }, null, null);
    defer want.deinit();
    var got = try ctx.moeExpertFfnBatch(&x, &fx.streamed_gate, &fx.streamed_up, &fx.streamed_down, &selected, &weights, top_k, out_pe, .{ .op = .swiglu }, null, null);
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

    // The fold flag's geometry legs (ProjGeometry.init): fold with one
    // plane would make readExpertFolded write a 520-byte-per-group pack
    // into a 264-byte-per-group section — the guard is load-bearing for
    // memory safety, so both rejection legs are pinned. Plane count first:
    try std.testing.expectError(error.InvalidExpertGeometry, store2.addLayer(0, .{
        .{ .quant = .tq2_0, .file_offset = 0, .byte_len = 1, .in_dim = hidden, .out_dim = out_pe, .plane_count = 1, .fold = true },
        .{ .quant = .tq2_0, .file_offset = 0, .byte_len = 1, .in_dim = hidden, .out_dim = out_pe },
        .{ .quant = .tq2_0, .file_offset = 0, .byte_len = 1, .in_dim = out_pe, .out_dim = hidden },
    }, n_expert));
    // ... and the 4-column-group rule:
    try std.testing.expectError(error.InvalidExpertGeometry, store2.addLayer(0, .{
        .{ .quant = .tq2_0, .file_offset = 0, .byte_len = 1, .in_dim = hidden, .out_dim = out_pe + 2, .plane_count = 2, .fold = true },
        .{ .quant = .tq2_0, .file_offset = 0, .byte_len = 1, .in_dim = hidden, .out_dim = out_pe },
        .{ .quant = .tq2_0, .file_offset = 0, .byte_len = 1, .in_dim = out_pe, .out_dim = hidden },
    }, n_expert));
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
    // recall miss). All three were uncached -> 3 stage requests enqueued
    // (one per expert; the worker loads whole experts, not per-proj ranges).
    fx.store.pilotHint(0, &.{ 5, 6, 3 });
    try std.testing.expectEqual(@as(u64, 3), fx.store.stats.pilot_ranges);
    try fx.expectDecodeMatches(&ctx, &.{ 5, 6 }, &.{ 0.7, 0.3 });
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.pilot_recall_hits);
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.pilot_recall_total);

    // One prediction scores exactly one acquire: a second decode on the
    // same layer does not double-count.
    try fx.expectDecodeMatches(&ctx, &.{ 5, 6 }, &.{ 0.7, 0.3 });
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.pilot_recall_total);

    // Cached/pinned experts are not re-hinted; a wrong prediction scores 0.
    fx.store.pilotHint(0, &.{ 5, 1 }); // 5 is now cached: only 1 enqueues
    try std.testing.expectEqual(@as(u64, 4), fx.store.stats.pilot_ranges);
    try fx.expectDecodeMatches(&ctx, &.{ 0, 2 }, &.{ 0.5, 0.5 });
    try std.testing.expectEqual(@as(u64, 2), fx.store.stats.pilot_recall_hits);
    try std.testing.expectEqual(@as(u64, 4), fx.store.stats.pilot_recall_total);
}

test "staging tier: pilot hints racing acquires and slot reclaim stay bit-exact" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 2);
    defer fx.deinit();

    // A second store over the same bytes with a deliberately tiny staging
    // tier: 3 slots for 8 experts under a 2-slot LRU forces constant
    // ready-slot reclaim (staged_wasted), duplicate-hint dedup, and
    // consume-by-swap while the worker is mid-load on the other slots.
    var store = try ExpertStore.create(allocator, &.{fx.path}, 1, .{
        .cache_slots_per_layer = 2,
        .prefetch_stage_slots = 3,
    });
    defer store.destroy();
    try fx.registerLayer(store);
    try store.finalize();

    const Hammer = struct {
        fn run(s: *ExpertStore, salt: usize, stop: *std.atomic.Value(bool)) void {
            var i: usize = salt;
            while (!stop.load(.acquire)) : (i +%= 1) {
                s.pilotHint(0, &.{ i % n_expert, (i + 2) % n_expert, (i + 5) % n_expert });
            }
        }
    };
    var stop = std.atomic.Value(bool).init(false);
    const t1 = try std.Thread.spawn(.{}, Hammer.run, .{ store, 0, &stop });
    const t2 = try std.Thread.spawn(.{}, Hammer.run, .{ store, 3, &stop });

    // Every decode races the staging worker; each one must stay bitwise
    // equal to the resident path whether it hit a staged slab, a sync
    // read, or the LRU cache.
    var round: usize = 0;
    while (round < 120) : (round += 1) {
        const a = round % n_expert;
        const b = (round + 3) % n_expert;
        fx.expectDecodeWith(&ctx, store, &.{ a, b }, &.{ 0.6, 0.4 }) catch |err| {
            stop.store(true, .release);
            t1.join();
            t2.join();
            return err;
        };
    }
    stop.store(true, .release);
    t1.join();
    t2.join();

    // The tier was actually exercised, and its ledger stays consistent:
    // consumed swaps can never exceed published loads.
    try std.testing.expect(store.stats.staged_loads > 0);
    try std.testing.expect(store.stats.staged_consumed <= store.stats.staged_loads);
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
    const gate_blocks = try allocator.alloc(dtype_mod.BlockQ5_K, gu_rows * (ds_hidden / qm.types.qk_k_block_size));
    defer allocator.free(gate_blocks);
    const up_blocks = try allocator.alloc(dtype_mod.BlockQ5_K, gate_blocks.len);
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
    const down_blocks = try allocator.alloc(dtype_mod.BlockQ8_0, down_rows * down_bpc);
    defer allocator.free(down_blocks);
    {
        var row: [ds_ffn]f32 = undefined;
        for (0..down_rows) |r| {
            for (&row, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(r * 31 + i)) * 0.11) * 1.7;
            try qm.q8k.quantizeRowQ8_0Into(down_blocks[r * down_bpc ..][0..down_bpc], &row);
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

    var resident_gate: MoeRhs = .{ .q5_k = try qm.q8k.quantizedMatmulRhsQ5_KFromBlocks(allocator, ds_hidden, gu_rows, gate_blocks) };
    defer resident_gate.deinit();
    var resident_up: MoeRhs = .{ .q5_k = try qm.q8k.quantizedMatmulRhsQ5_KFromBlocks(allocator, ds_hidden, gu_rows, up_blocks) };
    defer resident_up.deinit();
    var resident_down: MoeRhs = .{ .q8_0 = .{
        .rows = .{ .allocator = null, .blocks = down_blocks, .rows = down_rows, .cols = ds_ffn, .blocks_per_row = down_bpc },
        .k = ds_ffn,
        .n = down_rows,
    } };
    defer resident_down.deinit();

    const gate_bytes = gate_blocks.len * @sizeOf(dtype_mod.BlockQ5_K);
    const up_bytes = up_blocks.len * @sizeOf(dtype_mod.BlockQ5_K);
    const down_bytes = down_blocks.len * @sizeOf(dtype_mod.BlockQ8_0);
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
    var x = try ctx.fromSlice(.f32, .{ 1, ds_hidden }, x_vals);
    defer x.deinit();

    // Cold, warm, and evicting decodes must all match the resident path
    // bit-for-bit.
    for ([_][2]usize{ .{ 0, 3 }, .{ 0, 3 }, .{ 1, 2 }, .{ 0, 3 } }) |pair| {
        var want = try ctx.moeExpertFfn(&x, &resident_gate, &resident_up, &resident_down, &pair, &.{ 0.6, 0.4 }, ds_ffn, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &streamed_gate, &streamed_up, &streamed_down, &pair, &.{ 0.6, 0.4 }, ds_ffn, .{ .op = .swiglu }, null, null);
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
    const gu_bpc = t_hidden / qm.types.qk_k_block_size;
    const down_rows = t_experts * t_hidden;
    const down_bpc = t_ffn / qm.types.qk_k_block_size;
    const gate_blocks = try allocator.alloc(dtype_mod.BlockTQ2_0, gu_rows * gu_bpc);
    defer allocator.free(gate_blocks);
    const up_blocks = try allocator.alloc(dtype_mod.BlockTQ2_0, gate_blocks.len);
    defer allocator.free(up_blocks);
    const down_blocks = try allocator.alloc(dtype_mod.BlockTQ2_0, down_rows * down_bpc);
    defer allocator.free(down_blocks);
    {
        var row: [t_hidden]f32 = undefined;
        for (0..gu_rows) |r| {
            for (&row, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(r * 13 + i)) * 0.23) * 0.9;
            try qm.ternary.quantizeRowTQ2_0Into(gate_blocks[r * gu_bpc ..][0..gu_bpc], &row);
            for (&row, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(r * 7 + i)) * 0.31) * 1.1;
            try qm.ternary.quantizeRowTQ2_0Into(up_blocks[r * gu_bpc ..][0..gu_bpc], &row);
        }
        var drow: [t_ffn]f32 = undefined;
        for (0..down_rows) |r| {
            for (&drow, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(r * 31 + i)) * 0.11) * 0.8;
            try qm.ternary.quantizeRowTQ2_0Into(down_blocks[r * down_bpc ..][0..down_bpc], &drow);
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
        fn go(blocks: []dtype_mod.BlockTQ2_0, k: usize, rows: usize, bpc: usize) MoeRhs {
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

    const gate_bytes = gate_blocks.len * @sizeOf(dtype_mod.BlockTQ2_0);
    const up_bytes = up_blocks.len * @sizeOf(dtype_mod.BlockTQ2_0);
    const down_bytes = down_blocks.len * @sizeOf(dtype_mod.BlockTQ2_0);
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
    var x = try ctx.fromSlice(.f32, .{ 1, t_hidden }, x_vals);
    defer x.deinit();
    for ([_][2]usize{ .{ 0, 3 }, .{ 0, 3 }, .{ 1, 2 }, .{ 0, 3 } }) |pair| {
        var want = try ctx.moeExpertFfn(&x, &resident_gate, &resident_up, &resident_down, &pair, &.{ 0.6, 0.4 }, t_ffn, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &streamed_gate, &streamed_up, &streamed_down, &pair, &.{ 0.6, 0.4 }, t_ffn, .{ .op = .swiglu }, null, null);
        defer got.deinit();
        for (want.dataConst()) |v| try std.testing.expect(!std.math.isNan(v));
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    }

    // Batched prefill (m = 5 rows spanning all experts).
    const m: usize = 5;
    const xb_vals = try allocator.alloc(f32, m * t_hidden);
    defer allocator.free(xb_vals);
    for (xb_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 5) % 199)) - 99)) / 99.0;
    var xb = try ctx.fromSlice(.f32, .{ m, t_hidden }, xb_vals);
    defer xb.deinit();
    const selected = [_]usize{ 0, 3, 1, 2, 2, 0, 3, 1, 0, 1 };
    const routing = [_]f32{ 0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.2, 0.8, 0.9, 0.1 };
    var want_b = try ctx.moeExpertFfnBatch(&xb, &resident_gate, &resident_up, &resident_down, &selected, &routing, 2, t_ffn, .{ .op = .swiglu }, null, null);
    defer want_b.deinit();
    var got_b = try ctx.moeExpertFfnBatch(&xb, &streamed_gate, &streamed_up, &streamed_down, &selected, &routing, 2, t_ffn, .{ .op = .swiglu }, null, null);
    defer got_b.deinit();
    for (want_b.dataConst()) |v| try std.testing.expect(!std.math.isNan(v));
    try std.testing.expectEqualSlices(f32, want_b.dataConst(), got_b.dataConst());
}

// ---- PTQTP multi-plane reference helpers ---------------------------------

/// One expert's multi-plane projection, reference style: the single-plane
/// (K=1 tq2_0) ternary tile once per plane over the FULL row, summed
/// host-side in fixed plane order (p0 + p1 (+ p2), left-associated). This
/// is the dense fused PTQTP linear's contract — verified by reading
/// src/weights.zig `linearSeqPtqtpFused`, which computes each plane's full
/// tile dot and adds the results elementwise in plane order (sum of dots,
/// NOT interleaved accumulation) — so the fused MoE `ptqtp` arm must
/// reproduce this bitwise.
fn ptqtpPlaneSumDot(
    planes: []const []const dtype_mod.BlockTQ2_0,
    qx: []const dtype_mod.BlockQ8_K,
    e: usize,
    k: usize,
    out_dim: usize,
    out: []f32,
    tmp: []f32,
) void {
    const bpc = k / qm.types.qk_k_block_size;
    for (planes, 0..) |plane, p| {
        const blocks = plane[e * out_dim * bpc ..][0 .. out_dim * bpc];
        const view = backend_mod.QuantizedMatmulRhsTQ2_0{
            .rows = .{ .allocator = null, .blocks = @constCast(blocks), .rows = out_dim, .cols = k, .blocks_per_row = bpc },
            .k = k,
            .n = out_dim,
        };
        const dst = if (p == 0) out else tmp[0..out_dim];
        qm.ternary.matmulTQ2_0RhsTile(dst, qx, &view, out_dim, 0, 1, 0, out_dim);
        if (p != 0) {
            for (out, tmp[0..out_dim]) |*o, s| o.* += s;
        }
    }
}

/// One (token, expert) pair's UNWEIGHTED down-projection row through the
/// per-plane-sum reference pipeline: gate/up plane sums, gated activation,
/// Q8_K requantize, down plane sum — exactly the fused op's per-expert
/// arithmetic with every multi-plane dot replaced by the host-side sum.
const PtqtpRefBufs = struct {
    qx: []dtype_mod.BlockQ8_K,
    qg: []dtype_mod.BlockQ8_K,
    gate_buf: []f32,
    up_buf: []f32,
    g_buf: []f32,
    tmp: []f32,
};

fn ptqtpExpertDownReference(
    bufs: *const PtqtpRefBufs,
    gate_planes: []const []const dtype_mod.BlockTQ2_0,
    up_planes: []const []const dtype_mod.BlockTQ2_0,
    down_planes: []const []const dtype_mod.BlockTQ2_0,
    e: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    down_out: []f32,
) !void {
    ptqtpPlaneSumDot(gate_planes, bufs.qx, e, hidden_dim, ffn_dim, bufs.gate_buf, bufs.tmp);
    ptqtpPlaneSumDot(up_planes, bufs.qx, e, hidden_dim, ffn_dim, bufs.up_buf, bufs.tmp);
    for (bufs.g_buf, bufs.gate_buf, bufs.up_buf) |*g, gate_v, up_v| {
        g.* = backend_mod.ops.gatedPairScalar(.swiglu, gate_v, up_v);
    }
    try qm.q8k.quantizeRowQ8_KInto(bufs.qg, bufs.g_buf);
    ptqtpPlaneSumDot(down_planes, bufs.qg, e, ffn_dim, hidden_dim, down_out, bufs.tmp);
}

test "ptqtp multi-plane experts: fused MoE sums planes like the dense path; streamed bit-exact vs resident" {
    // The PTQTP expert tier (K=2 gate/up, K=3 down): the fused op must sum
    // the per-plane dots per projection BEFORE the gated nonlinearity, in
    // the dense fused linear's order — asserted here as (a) resident ptqtp
    // arm == (b) host-side per-plane K=1 sums (the sum-of-dots contract,
    // see ptqtpPlaneSumDot) — and the streamed arm (c), whose ProjSpec
    // gathers the plane-major sibling tensors into expert-major slab
    // sections, must equal (a) bitwise across cold/warm/evicting decode
    // and the batched path.
    const allocator = std.testing.allocator;
    const ptqtp = @import("../ptqtp.zig");
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const t_hidden: usize = 256;
    const t_ffn: usize = 512;
    const t_experts: usize = 2;
    const gu_rows = t_experts * t_ffn;
    const gu_bpc = t_hidden / qm.types.qk_k_block_size;
    const down_rows = t_experts * t_hidden;
    const down_bpc = t_ffn / qm.types.qk_k_block_size;

    // Quantize synthetic expert stacks with the real PTQTP solver (rows are
    // independent groups, so one whole-stack solve equals per-expert
    // solves). Few iterations keep the fixture fast; determinism holds for
    // any iteration budget.
    const gate_w = try allocator.alloc(f32, gu_rows * t_hidden);
    defer allocator.free(gate_w);
    const up_w = try allocator.alloc(f32, gu_rows * t_hidden);
    defer allocator.free(up_w);
    const down_w = try allocator.alloc(f32, down_rows * t_ffn);
    defer allocator.free(down_w);
    for (gate_w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.017) * 0.8;
    for (up_w, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.023) * 1.1;
    for (down_w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.011 + 0.5) * 0.7;

    var gate_pair = try ptqtp.quantizeMatrix(&ctx, gate_w, gu_rows, t_hidden, .{ .planes = 2, .max_iterations = 8 });
    defer gate_pair.deinit(allocator);
    var up_pair = try ptqtp.quantizeMatrix(&ctx, up_w, gu_rows, t_hidden, .{ .planes = 2, .max_iterations = 8 });
    defer up_pair.deinit(allocator);
    var down_pair = try ptqtp.quantizeMatrix(&ctx, down_w, down_rows, t_ffn, .{ .planes = 3, .max_iterations = 8 });
    defer down_pair.deinit(allocator);

    const gate_planes = [_][]const dtype_mod.BlockTQ2_0{ gate_pair.plane1, gate_pair.plane2 };
    const up_planes = [_][]const dtype_mod.BlockTQ2_0{ up_pair.plane1, up_pair.plane2 };
    const down_planes = [_][]const dtype_mod.BlockTQ2_0{ down_pair.plane1, down_pair.plane2, down_pair.plane3 };

    // (a) resident multi-plane arms.
    var resident_gate: MoeRhs = .{ .ptqtp = .{
        .allocator = null,
        .planes = .{ gate_pair.plane1, gate_pair.plane2, &.{} },
        .plane_count = 2,
        .k = t_hidden,
        .n = gu_rows,
        .blocks_per_column = gu_bpc,
    } };
    var resident_up: MoeRhs = .{ .ptqtp = .{
        .allocator = null,
        .planes = .{ up_pair.plane1, up_pair.plane2, &.{} },
        .plane_count = 2,
        .k = t_hidden,
        .n = gu_rows,
        .blocks_per_column = gu_bpc,
    } };
    var resident_down: MoeRhs = .{ .ptqtp = .{
        .allocator = null,
        .planes = .{ down_pair.plane1, down_pair.plane2, down_pair.plane3 },
        .plane_count = 3,
        .k = t_ffn,
        .n = down_rows,
        .blocks_per_column = down_bpc,
    } };

    // (c) the same planes on disk, plane-major sibling layout (the GGUF
    // convention): every plane is one contiguous stack; the ProjSpec's
    // plane offsets point the store at them.
    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "expert_store_ptqtp_{d}.bin", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    defer cleanupSidecar(path);
    const gu_plane_bytes = gate_pair.plane1.len * @sizeOf(dtype_mod.BlockTQ2_0);
    const down_plane_bytes = down_pair.plane1.len * @sizeOf(dtype_mod.BlockTQ2_0);
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        for ([_][]const dtype_mod.BlockTQ2_0{
            gate_pair.plane1, gate_pair.plane2,
            up_pair.plane1,   up_pair.plane2,
            down_pair.plane1, down_pair.plane2,
            down_pair.plane3,
        }) |plane| try writer.interface.writeAll(std.mem.sliceAsBytes(plane));
        try writer.interface.flush();
    }

    var store = try ExpertStore.create(allocator, &.{path}, 1, .{ .cache_slots_per_layer = 1 });
    defer store.destroy();
    try store.addLayer(0, .{
        .{ .quant = .tq2_0, .file_offset = 0, .byte_len = gu_plane_bytes, .in_dim = t_hidden, .out_dim = t_ffn, .plane_count = 2, .plane_offsets = .{ gu_plane_bytes, 0 } },
        .{ .quant = .tq2_0, .file_offset = 2 * gu_plane_bytes, .byte_len = gu_plane_bytes, .in_dim = t_hidden, .out_dim = t_ffn, .plane_count = 2, .plane_offsets = .{ 3 * gu_plane_bytes, 0 } },
        .{ .quant = .tq2_0, .file_offset = 4 * gu_plane_bytes, .byte_len = down_plane_bytes, .in_dim = t_ffn, .out_dim = t_hidden, .plane_count = 3, .plane_offsets = .{ 4 * gu_plane_bytes + down_plane_bytes, 4 * gu_plane_bytes + 2 * down_plane_bytes } },
    }, t_experts);
    try store.finalize();
    var streamed_gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
    var streamed_up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
    var streamed_down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };

    // (b) scratch for the reference pipeline.
    const bufs = PtqtpRefBufs{
        .qx = try allocator.alloc(dtype_mod.BlockQ8_K, t_hidden / qm.types.qk_k_block_size),
        .qg = try allocator.alloc(dtype_mod.BlockQ8_K, t_ffn / qm.types.qk_k_block_size),
        .gate_buf = try allocator.alloc(f32, t_ffn),
        .up_buf = try allocator.alloc(f32, t_ffn),
        .g_buf = try allocator.alloc(f32, t_ffn),
        .tmp = try allocator.alloc(f32, t_ffn),
    };
    defer {
        allocator.free(bufs.tmp);
        allocator.free(bufs.g_buf);
        allocator.free(bufs.up_buf);
        allocator.free(bufs.gate_buf);
        allocator.free(bufs.qg);
        allocator.free(bufs.qx);
    }
    const dbuf = try allocator.alloc(f32, t_hidden);
    defer allocator.free(dbuf);
    const ref = try allocator.alloc(f32, t_hidden);
    defer allocator.free(ref);

    // Decode (m=1): cold, warm, and evicting acquires (cap 1 < 2 active).
    const x_vals = try allocator.alloc(f32, t_hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11) % 173)) - 86)) / 86.0;
    var x = try ctx.fromSlice(.f32, .{ 1, t_hidden }, x_vals);
    defer x.deinit();
    for ([_][2]usize{ .{ 0, 1 }, .{ 0, 1 }, .{ 1, 0 } }) |pair| {
        const routing = [_]f32{ 0.6, 0.4 };
        var want = try ctx.moeExpertFfn(&x, &resident_gate, &resident_up, &resident_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        for (want.dataConst()) |v| try std.testing.expect(!std.math.isNan(v));

        // (b): per-plane K=1 sums, then the decode op's exact assembly
        // (down row scaled by its routing weight, expert rows added in
        // routed order onto a zeroed accumulator).
        try qm.q8k.quantizeRowQ8_KInto(bufs.qx, x_vals);
        @memset(ref, 0);
        for (pair, routing) |e, w| {
            try ptqtpExpertDownReference(&bufs, &gate_planes, &up_planes, &down_planes, e, t_hidden, t_ffn, dbuf);
            for (dbuf) |*v| v.* *= w;
            for (ref, dbuf) |*o, s| o.* += s;
        }
        try std.testing.expectEqualSlices(f32, ref, want.dataConst());

        // (c): streamed == resident, always.
        var got = try ctx.moeExpertFfn(&x, &streamed_gate, &streamed_up, &streamed_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer got.deinit();
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    }

    // Batched prefill (m=5 rows spanning both experts).
    const m: usize = 5;
    const top_k: usize = 2;
    const xb_vals = try allocator.alloc(f32, m * t_hidden);
    defer allocator.free(xb_vals);
    for (xb_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 199)) - 99)) / 99.0;
    var xb = try ctx.fromSlice(.f32, .{ m, t_hidden }, xb_vals);
    defer xb.deinit();
    const selected = [_]usize{ 0, 1, 1, 0, 0, 1, 1, 0, 0, 0 };
    const routing = [_]f32{ 0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.2, 0.8, 0.9, 0.1 };
    var want_b = try ctx.moeExpertFfnBatch(&xb, &resident_gate, &resident_up, &resident_down, &selected, &routing, top_k, t_ffn, .{ .op = .swiglu }, null, null);
    defer want_b.deinit();
    for (want_b.dataConst()) |v| try std.testing.expect(!std.math.isNan(v));

    // (b) per token: identical per-pair arithmetic (each output element of
    // an m-row tile is an independent dot, so per-token m=1 references
    // match), assembled with the batch scatter's contract (first routed
    // pair assigns w*row, later pairs add w*row, in per-token k order).
    const want_b_data = want_b.dataConst();
    for (0..m) |t| {
        try qm.q8k.quantizeRowQ8_KInto(bufs.qx, xb_vals[t * t_hidden ..][0..t_hidden]);
        for (0..top_k) |j| {
            const e = selected[t * top_k + j];
            const w = routing[t * top_k + j];
            try ptqtpExpertDownReference(&bufs, &gate_planes, &up_planes, &down_planes, e, t_hidden, t_ffn, dbuf);
            if (j == 0) {
                for (ref, dbuf) |*o, s| o.* = w * s;
            } else {
                for (ref, dbuf) |*o, s| o.* += w * s;
            }
        }
        try std.testing.expectEqualSlices(f32, ref, want_b_data[t * t_hidden ..][0..t_hidden]);
    }

    // (c) streamed batch == resident batch.
    var got_b = try ctx.moeExpertFfnBatch(&xb, &streamed_gate, &streamed_up, &streamed_down, &selected, &routing, top_k, t_ffn, .{ .op = .swiglu }, null, null);
    defer got_b.deinit();
    try std.testing.expectEqualSlices(f32, want_b.dataConst(), got_b.dataConst());
}

fn foldedExpertDot(
    folded: []const qm.BlockTQ2_0Foldedx4,
    qx: []const dtype_mod.BlockQ8_K,
    e: usize,
    k: usize,
    out_dim: usize,
    out: []f32,
) void {
    const bpc = k / qm.types.qk_k_block_size;
    const fg = (out_dim / 4) * bpc;
    qm.ternary.matmulTQ2_0FoldedX4RhsRange(out, qx, folded[e * fg ..][0..fg], bpc, out_dim, 0, 1);
}

fn foldedExpertDownReference(
    bufs: *const PtqtpRefBufs,
    gate_folded: []const qm.BlockTQ2_0Foldedx4,
    up_folded: []const qm.BlockTQ2_0Foldedx4,
    down_folded: []const qm.BlockTQ2_0Foldedx4,
    e: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    down_out: []f32,
) !void {
    foldedExpertDot(gate_folded, bufs.qx, e, hidden_dim, ffn_dim, bufs.gate_buf);
    foldedExpertDot(up_folded, bufs.qx, e, hidden_dim, ffn_dim, bufs.up_buf);
    for (bufs.g_buf, bufs.gate_buf, bufs.up_buf) |*g, gate_v, up_v| {
        g.* = backend_mod.ops.gatedPairScalar(.swiglu, gate_v, up_v);
    }
    try qm.q8k.quantizeRowQ8_KInto(bufs.qg, bufs.g_buf);
    foldedExpertDot(down_folded, bufs.qg, e, ffn_dim, hidden_dim, down_out);
}

/// Expert-major folded stack from two plane stacks — the layout
/// `loadMoeRhsPtqtp` builds and the streamed fill reproduces per slab.
fn foldExpertStack(
    allocator: std.mem.Allocator,
    plane1: []const dtype_mod.BlockTQ2_0,
    plane2: []const dtype_mod.BlockTQ2_0,
    n_experts: usize,
    k: usize,
    out_dim: usize,
) ![]qm.BlockTQ2_0Foldedx4 {
    const bpc = k / qm.types.qk_k_block_size;
    const expert_blocks = out_dim * bpc;
    const fg = (out_dim / 4) * bpc;
    const folded = try allocator.alloc(qm.BlockTQ2_0Foldedx4, n_experts * fg);
    errdefer allocator.free(folded);
    for (0..n_experts) |e| {
        var views: [2]backend_mod.QuantizedMatmulRhsTQ2_0 = undefined;
        for ([2][]const dtype_mod.BlockTQ2_0{ plane1, plane2 }, 0..) |plane, p| {
            views[p] = .{
                .rows = .{
                    .allocator = null,
                    .blocks = @constCast(plane[e * expert_blocks ..][0..expert_blocks]),
                    .rows = out_dim,
                    .cols = k,
                    .blocks_per_row = bpc,
                },
                .k = k,
                .n = out_dim,
            };
        }
        try qm.ternary.packMatmulRhsTQ2_0Foldedx4Into(folded[e * fg ..][0..fg], &views[0], &views[1]);
    }
    return folded;
}

test "tie-folded ptqtp experts: resident fold serves the one-pass kernel; streamed fill folds bit-exact" {
    // Tie-fitted K=2 stacks (docs/PTQTP.md): the resident arm carries the
    // expert-major folded pack and the expert dot takes the one-pass
    // folded kernel; the streamed tier folds the two plane reads into the
    // slab at fill (`readExpert`). Pins: (a) the resident folded serve ==
    // a host reference assembled on the direct folded kernel — the
    // exact-ratio semantics, deliberately NOT the 2-pass plane sum, whose
    // independently rounded coarse f16 scale differs in final ulps; (b)
    // streamed folded == resident folded bitwise across cold, warm, and
    // evicting acquires, decode and batch.
    const allocator = std.testing.allocator;
    const ptqtp = @import("../ptqtp.zig");
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const t_hidden: usize = 256;
    const t_ffn: usize = 512;
    const t_experts: usize = 2;
    const gu_rows = t_experts * t_ffn;
    const gu_bpc = t_hidden / qm.types.qk_k_block_size;
    const down_rows = t_experts * t_hidden;
    const down_bpc = t_ffn / qm.types.qk_k_block_size;

    const gate_w = try allocator.alloc(f32, gu_rows * t_hidden);
    defer allocator.free(gate_w);
    const up_w = try allocator.alloc(f32, gu_rows * t_hidden);
    defer allocator.free(up_w);
    const down_w = try allocator.alloc(f32, down_rows * t_ffn);
    defer allocator.free(down_w);
    for (gate_w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.019) * 0.9;
    for (up_w, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.029) * 1.2;
    for (down_w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.013 + 0.3) * 0.6;

    const tied_opts = ptqtp.Options{ .planes = 2, .max_iterations = 8, .tie_scales = true };
    var gate_pair = try ptqtp.quantizeMatrix(&ctx, gate_w, gu_rows, t_hidden, tied_opts);
    defer gate_pair.deinit(allocator);
    var up_pair = try ptqtp.quantizeMatrix(&ctx, up_w, gu_rows, t_hidden, tied_opts);
    defer up_pair.deinit(allocator);
    var down_pair = try ptqtp.quantizeMatrix(&ctx, down_w, down_rows, t_ffn, tied_opts);
    defer down_pair.deinit(allocator);

    const gate_folded = try foldExpertStack(allocator, gate_pair.plane1, gate_pair.plane2, t_experts, t_hidden, t_ffn);
    defer allocator.free(gate_folded);
    const up_folded = try foldExpertStack(allocator, up_pair.plane1, up_pair.plane2, t_experts, t_hidden, t_ffn);
    defer allocator.free(up_folded);
    const down_folded = try foldExpertStack(allocator, down_pair.plane1, down_pair.plane2, t_experts, t_ffn, t_hidden);
    defer allocator.free(down_folded);

    var resident_gate: MoeRhs = .{ .ptqtp = .{
        .allocator = null,
        .planes = .{ gate_pair.plane1, gate_pair.plane2, &.{} },
        .plane_count = 2,
        .k = t_hidden,
        .n = gu_rows,
        .blocks_per_column = gu_bpc,
        .folded = gate_folded,
    } };
    var resident_up: MoeRhs = .{ .ptqtp = .{
        .allocator = null,
        .planes = .{ up_pair.plane1, up_pair.plane2, &.{} },
        .plane_count = 2,
        .k = t_hidden,
        .n = gu_rows,
        .blocks_per_column = gu_bpc,
        .folded = up_folded,
    } };
    var resident_down: MoeRhs = .{ .ptqtp = .{
        .allocator = null,
        .planes = .{ down_pair.plane1, down_pair.plane2, &.{} },
        .plane_count = 2,
        .k = t_ffn,
        .n = down_rows,
        .blocks_per_column = down_bpc,
        .folded = down_folded,
    } };

    // The same planes on disk, plane-major sibling layout; the fold flag
    // makes every fill bounce the two plane reads through the scratch and
    // land the 4-bit pack in the slab section.
    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "expert_store_ptqtp_fold_{d}.bin", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    defer cleanupSidecar(path);
    const gu_plane_bytes = gate_pair.plane1.len * @sizeOf(dtype_mod.BlockTQ2_0);
    const down_plane_bytes = down_pair.plane1.len * @sizeOf(dtype_mod.BlockTQ2_0);
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        for ([_][]const dtype_mod.BlockTQ2_0{
            gate_pair.plane1, gate_pair.plane2,
            up_pair.plane1,   up_pair.plane2,
            down_pair.plane1, down_pair.plane2,
        }) |plane| try writer.interface.writeAll(std.mem.sliceAsBytes(plane));
        try writer.interface.flush();
    }

    var store = try ExpertStore.create(allocator, &.{path}, 1, .{ .cache_slots_per_layer = 1 });
    defer store.destroy();
    try store.addLayer(0, .{
        .{ .quant = .tq2_0, .file_offset = 0, .byte_len = gu_plane_bytes, .in_dim = t_hidden, .out_dim = t_ffn, .plane_count = 2, .plane_offsets = .{ gu_plane_bytes, 0 }, .fold = true },
        .{ .quant = .tq2_0, .file_offset = 2 * gu_plane_bytes, .byte_len = gu_plane_bytes, .in_dim = t_hidden, .out_dim = t_ffn, .plane_count = 2, .plane_offsets = .{ 3 * gu_plane_bytes, 0 }, .fold = true },
        .{ .quant = .tq2_0, .file_offset = 4 * gu_plane_bytes, .byte_len = down_plane_bytes, .in_dim = t_ffn, .out_dim = t_hidden, .plane_count = 2, .plane_offsets = .{ 4 * gu_plane_bytes + down_plane_bytes, 0 }, .fold = true },
    }, t_experts);
    try store.finalize();
    var streamed_gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
    var streamed_up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
    var streamed_down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };

    const bufs = PtqtpRefBufs{
        .qx = try allocator.alloc(dtype_mod.BlockQ8_K, t_hidden / qm.types.qk_k_block_size),
        .qg = try allocator.alloc(dtype_mod.BlockQ8_K, t_ffn / qm.types.qk_k_block_size),
        .gate_buf = try allocator.alloc(f32, t_ffn),
        .up_buf = try allocator.alloc(f32, t_ffn),
        .g_buf = try allocator.alloc(f32, t_ffn),
        .tmp = try allocator.alloc(f32, t_ffn),
    };
    defer {
        allocator.free(bufs.tmp);
        allocator.free(bufs.g_buf);
        allocator.free(bufs.up_buf);
        allocator.free(bufs.gate_buf);
        allocator.free(bufs.qg);
        allocator.free(bufs.qx);
    }
    const dbuf = try allocator.alloc(f32, t_hidden);
    defer allocator.free(dbuf);
    const ref = try allocator.alloc(f32, t_hidden);
    defer allocator.free(ref);

    // Decode: cold, warm, and evicting acquires (cap 1 < 2 active).
    const x_vals = try allocator.alloc(f32, t_hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13) % 167)) - 83)) / 83.0;
    var x = try ctx.fromSlice(.f32, .{ 1, t_hidden }, x_vals);
    defer x.deinit();
    for ([_][2]usize{ .{ 0, 1 }, .{ 0, 1 }, .{ 1, 0 } }) |pair| {
        const routing = [_]f32{ 0.6, 0.4 };
        var want = try ctx.moeExpertFfn(&x, &resident_gate, &resident_up, &resident_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        for (want.dataConst()) |v| try std.testing.expect(!std.math.isNan(v));

        // (a) the folded-kernel reference, assembled exactly like the op.
        try qm.q8k.quantizeRowQ8_KInto(bufs.qx, x_vals);
        @memset(ref, 0);
        for (pair, routing) |e, w| {
            try foldedExpertDownReference(&bufs, gate_folded, up_folded, down_folded, e, t_hidden, t_ffn, dbuf);
            for (dbuf) |*v| v.* *= w;
            for (ref, dbuf) |*o, s| o.* += s;
        }
        try std.testing.expectEqualSlices(f32, ref, want.dataConst());

        // (b) streamed fill-fold == resident fold, always.
        var got = try ctx.moeExpertFfn(&x, &streamed_gate, &streamed_up, &streamed_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer got.deinit();
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    }

    // Batched prefill (m=5 rows spanning both experts).
    const m: usize = 5;
    const top_k: usize = 2;
    const xb_vals = try allocator.alloc(f32, m * t_hidden);
    defer allocator.free(xb_vals);
    for (xb_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17) % 191)) - 95)) / 95.0;
    var xb = try ctx.fromSlice(.f32, .{ m, t_hidden }, xb_vals);
    defer xb.deinit();
    const selected = [_]usize{ 0, 1, 1, 0, 0, 1, 1, 0, 0, 0 };
    const routing = [_]f32{ 0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.2, 0.8, 0.9, 0.1 };
    var want_b = try ctx.moeExpertFfnBatch(&xb, &resident_gate, &resident_up, &resident_down, &selected, &routing, top_k, t_ffn, .{ .op = .swiglu }, null, null);
    defer want_b.deinit();
    const want_b_data = want_b.dataConst();
    for (0..m) |t| {
        try qm.q8k.quantizeRowQ8_KInto(bufs.qx, xb_vals[t * t_hidden ..][0..t_hidden]);
        for (0..top_k) |j| {
            const e = selected[t * top_k + j];
            const w = routing[t * top_k + j];
            try foldedExpertDownReference(&bufs, gate_folded, up_folded, down_folded, e, t_hidden, t_ffn, dbuf);
            if (j == 0) {
                for (ref, dbuf) |*o, s| o.* = w * s;
            } else {
                for (ref, dbuf) |*o, s| o.* += w * s;
            }
        }
        try std.testing.expectEqualSlices(f32, ref, want_b_data[t * t_hidden ..][0..t_hidden]);
    }
    var got_b = try ctx.moeExpertFfnBatch(&xb, &streamed_gate, &streamed_up, &streamed_down, &selected, &routing, top_k, t_ffn, .{ .op = .swiglu }, null, null);
    defer got_b.deinit();
    try std.testing.expectEqualSlices(f32, want_b.dataConst(), got_b.dataConst());
}

test "q2_k, iq2_xxs, and iq3_xxs experts: streamed decode and batch are bit-exact vs resident" {
    // The community 2-bit tier (UD-IQ2_XXS files mix iq2_xxs, iq3_xxs, and
    // q2_k expert stacks): gate iq2_xxs, up iq3_xxs, down q2_k. Fixtures are
    // synthetic — every
    // byte pattern is a valid block for these formats (grid indices index
    // 256-entry tables, sign words index 128-entry tables), so patterned
    // bytes with mild f16 scales exercise the kernels deterministically.
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const t_hidden: usize = 256;
    const t_ffn: usize = 512;
    const t_experts: usize = 4;

    const gu_rows = t_experts * t_ffn;
    const gu_bpc = t_hidden / qm.types.qk_k_block_size;
    const down_rows = t_experts * t_hidden;
    const down_bpc = t_ffn / qm.types.qk_k_block_size;
    const gate_blocks = try allocator.alloc(dtype_mod.BlockIQ2_XXS, gu_rows * gu_bpc);
    defer allocator.free(gate_blocks);
    const up_blocks = try allocator.alloc(dtype_mod.BlockIQ3_XXS, gu_rows * gu_bpc);
    defer allocator.free(up_blocks);
    const down_blocks = try allocator.alloc(dtype_mod.BlockQ2_K, down_rows * down_bpc);
    defer allocator.free(down_blocks);
    for (gate_blocks, 0..) |*b, i| {
        b.d = f16Bits(0.02);
        for (&b.qs, 0..) |*q, j| q.* = @truncate((i * 7919 + j * 104729) % 65536);
    }
    for (up_blocks, 0..) |*b, i| {
        b.d = f16Bits(0.015);
        for (&b.qs, 0..) |*q, j| q.* = @truncate((i * 6007 + j * 92821 + 17) % 256);
    }
    for (down_blocks, 0..) |*b, i| {
        for (&b.scales, 0..) |*v, j| v.* = @truncate((i * 31 + j * 7) % 256);
        for (&b.qs, 0..) |*v, j| v.* = @truncate((i * 13 + j * 11) % 256);
        b.dm[0] = f16Bits(0.01);
        b.dm[1] = f16Bits(0.002);
    }

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "expert_store_iq2_{d}.bin", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
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

    var resident_gate: MoeRhs = .{ .iq2_xxs = .{
        .rows = .{ .allocator = null, .blocks = gate_blocks, .rows = gu_rows, .cols = t_hidden, .blocks_per_row = gu_bpc },
        .k = t_hidden,
        .n = gu_rows,
    } };
    defer resident_gate.deinit();
    var resident_up: MoeRhs = .{ .iq3_xxs = .{
        .rows = .{ .allocator = null, .blocks = up_blocks, .rows = gu_rows, .cols = t_hidden, .blocks_per_row = gu_bpc },
        .k = t_hidden,
        .n = gu_rows,
    } };
    defer resident_up.deinit();
    var resident_down: MoeRhs = .{ .q2_k = .{
        .allocator = null,
        .blocks = down_blocks,
        .k = t_ffn,
        .n = down_rows,
        .blocks_per_column = down_bpc,
    } };
    defer resident_down.deinit();

    const gate_bytes = gate_blocks.len * @sizeOf(dtype_mod.BlockIQ2_XXS);
    const up_bytes = up_blocks.len * @sizeOf(dtype_mod.BlockIQ3_XXS);
    const down_bytes = down_blocks.len * @sizeOf(dtype_mod.BlockQ2_K);
    var store = try ExpertStore.create(allocator, &.{path}, 1, .{ .cache_slots_per_layer = 2 });
    defer store.destroy();
    try store.addLayer(0, .{
        .{ .quant = .iq2_xxs, .file_offset = 0, .byte_len = gate_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .iq3_xxs, .file_offset = gate_bytes, .byte_len = up_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .q2_k, .file_offset = gate_bytes + up_bytes, .byte_len = down_bytes, .in_dim = t_ffn, .out_dim = t_hidden },
    }, t_experts);
    try store.finalize();
    var streamed_gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
    var streamed_up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
    var streamed_down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };

    const x_vals = try allocator.alloc(f32, t_hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17) % 191)) - 95)) / 95.0;
    var x = try ctx.fromSlice(.f32, .{ 1, t_hidden }, x_vals);
    defer x.deinit();
    for ([_][2]usize{ .{ 0, 3 }, .{ 0, 3 }, .{ 1, 2 }, .{ 0, 3 } }) |pair| {
        var want = try ctx.moeExpertFfn(&x, &resident_gate, &resident_up, &resident_down, &pair, &.{ 0.6, 0.4 }, t_ffn, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &streamed_gate, &streamed_up, &streamed_down, &pair, &.{ 0.6, 0.4 }, t_ffn, .{ .op = .swiglu }, null, null);
        defer got.deinit();
        for (want.dataConst()) |v| try std.testing.expect(!std.math.isNan(v));
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    }

    const m: usize = 5;
    const xb_vals = try allocator.alloc(f32, m * t_hidden);
    defer allocator.free(xb_vals);
    for (xb_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 3) % 157)) - 78)) / 78.0;
    var xb = try ctx.fromSlice(.f32, .{ m, t_hidden }, xb_vals);
    defer xb.deinit();
    const selected = [_]usize{ 0, 3, 1, 2, 2, 0, 3, 1, 0, 1 };
    const routing = [_]f32{ 0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.2, 0.8, 0.9, 0.1 };
    var want_b = try ctx.moeExpertFfnBatch(&xb, &resident_gate, &resident_up, &resident_down, &selected, &routing, 2, t_ffn, .{ .op = .swiglu }, null, null);
    defer want_b.deinit();
    var got_b = try ctx.moeExpertFfnBatch(&xb, &streamed_gate, &streamed_up, &streamed_down, &selected, &routing, 2, t_ffn, .{ .op = .swiglu }, null, null);
    defer got_b.deinit();
    for (want_b.dataConst()) |v| try std.testing.expect(!std.math.isNan(v));
    try std.testing.expectEqualSlices(f32, want_b.dataConst(), got_b.dataConst());
}

test "cache-aware routing: sacred ranks kept, fill prefers resident, swaps counted" {
    const allocator = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.init(allocator, 2);
    defer fx.deinit();

    var store = try ExpertStore.create(allocator, &.{fx.path}, 1, .{
        .cache_slots_per_layer = 2,
        .cache_route = .{ .sacred = 1, .window = 4 },
    });
    defer store.destroy();
    try fx.registerLayer(store);
    try store.finalize();

    // Descending scores: the true top-2 is {0, 1}.
    const choice = [_]f32{ 0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.05, 0.01 };
    var sel: [2]usize = undefined;

    // Nothing resident yet: cache-aware selection equals the true top-k.
    try std.testing.expect(store.cacheRouteTopK(0, &choice, &sel));
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, &sel);
    try std.testing.expectEqual(@as(u64, 0), store.stats.route_swaps);

    // Make expert 3 resident (miss promotes it into the LRU tier).
    try store.acquire(0, &.{3});
    store.release(0);

    // Rank 0 stays sacred; the fill slot prefers resident 3 (rank 3,
    // inside the window) over the true rank-1 expert.
    try std.testing.expect(store.cacheRouteTopK(0, &choice, &sel));
    try std.testing.expectEqualSlices(usize, &.{ 0, 3 }, &sel);
    try std.testing.expectEqual(@as(u64, 1), store.stats.route_swaps);
    try std.testing.expectEqual(@as(u64, 4), store.stats.route_slots);

    // Outside the window nothing changes: expert 5 resident but ranked 6th.
    try store.acquire(0, &.{5});
    store.release(0);
    var sel3: [2]usize = undefined;
    const steep = [_]f32{ 0.9, 0.8, 0.75, 0.7, 0.65, 0.1, 0.05, 0.01 };
    try std.testing.expect(store.cacheRouteTopK(0, &steep, &sel3));
    // Resident 3 (rank 3) still wins the fill slot; resident 5 (rank 5)
    // sits outside window=4 and is never considered.
    try std.testing.expectEqualSlices(usize, &.{ 0, 3 }, &sel3);

    // The store without the option keeps declining.
    var plain_sel: [2]usize = undefined;
    try std.testing.expect(!fx.store.cacheRouteTopK(0, &choice, &plain_sel));
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

test "mirror copies: reads split across drives, stay bit-exact, and a broken mirror falls back" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, n_expert); // every expert cached: each read once
    defer fx.deinit();

    // A full second copy of the model bytes — "the other drive".
    var mirror_buf: [160]u8 = undefined;
    const mirror_path = try std.fmt.bufPrint(&mirror_buf, "{s}.copy", .{fx.path});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, mirror_path) catch {};
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, mirror_path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(std.mem.sliceAsBytes(fx.gate_blocks));
        try writer.interface.writeAll(std.mem.sliceAsBytes(fx.up_blocks));
        try writer.interface.writeAll(std.mem.sliceAsBytes(fx.down_blocks));
        try writer.interface.flush();
    }

    // Attach validation: part-count mismatch and an unopenable path are
    // refused (a mirror is configuration — fail loud, unlike read errors).
    var store = try ExpertStore.create(allocator, &.{fx.path}, 1, .{ .cache_slots_per_layer = n_expert });
    defer store.destroy();
    try std.testing.expectError(error.InvalidExpertGeometry, store.addMirror(&.{ mirror_path, mirror_path }, 1.0));
    try std.testing.expectError(error.InvalidExpertGeometry, store.addMirror(&.{mirror_path}, 0));
    try std.testing.expectError(error.ExpertFileOpenFailed, store.addMirror(&.{"expert_store_no_such_copy.bin"}, 1.0));

    // Even split: every expert decodes bit-exactly while the reads land on
    // both copies (the (layer, expert) hash is deterministic, so this split
    // is stable), with no fallbacks.
    try store.addMirror(&.{mirror_path}, 1.0);
    try fx.registerLayer(store);
    try store.finalize();
    var e: usize = 0;
    while (e < n_expert) : (e += 2) {
        try fx.expectDecodeWith(&ctx, store, &.{ e, e + 1 }, &.{ 0.6, 0.4 });
    }
    try std.testing.expect(store.copy_bytes[0].load(.monotonic) > 0);
    try std.testing.expect(store.copy_bytes[1].load(.monotonic) > 0);
    try std.testing.expectEqual(@as(u64, 0), store.mirror_fallbacks.load(.monotonic));

    // A TRUNCATED mirror (gate section only) weighted to draw ~all reads:
    // up/down preads past its EOF fail and fall back to the primary —
    // output stays bit-exact, and the fallbacks are counted.
    var short_buf: [160]u8 = undefined;
    const short_path = try std.fmt.bufPrint(&short_buf, "{s}.short", .{fx.path});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, short_path) catch {};
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, short_path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(std.mem.sliceAsBytes(fx.gate_blocks));
        try writer.interface.flush();
    }
    var store2 = try ExpertStore.create(allocator, &.{fx.path}, 1, .{ .cache_slots_per_layer = n_expert });
    defer store2.destroy();
    try store2.addMirror(&.{short_path}, 1000.0);
    try fx.registerLayer(store2);
    try store2.finalize();
    e = 0;
    while (e < n_expert) : (e += 2) {
        try fx.expectDecodeWith(&ctx, store2, &.{ e, e + 1 }, &.{ 0.6, 0.4 });
    }
    try std.testing.expect(store2.mirror_fallbacks.load(.monotonic) > 0);
}

test "parallel demand reads: fan-out stays bit-exact, drives a mirror concurrently, and surfaces worker read errors" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 2); // cap 2: every acquire below misses
    defer fx.deinit();

    // A second full copy so the fan-out exercises BOTH drives inside one
    // acquire — the combination the mirror exists for.
    var mirror_buf: [160]u8 = undefined;
    const mirror_path = try std.fmt.bufPrint(&mirror_buf, "{s}.pcopy", .{fx.path});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, mirror_path) catch {};
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, mirror_path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(std.mem.sliceAsBytes(fx.gate_blocks));
        try writer.interface.writeAll(std.mem.sliceAsBytes(fx.up_blocks));
        try writer.interface.writeAll(std.mem.sliceAsBytes(fx.down_blocks));
        try writer.interface.flush();
    }
    var store = try ExpertStore.create(allocator, &.{fx.path}, 1, .{
        .cache_slots_per_layer = 2,
        .io_workers = 3,
    });
    defer store.destroy();
    try store.addMirror(&.{mirror_path}, 1.0);
    try fx.registerLayer(store);
    try store.finalize();

    // A batched prefill whose union routes ALL 8 experts is one acquire
    // with 8 misses — an 8-wide read batch over 4 threads and 2 copies —
    // and it must be bitwise equal to the resident path.
    const seq: usize = 16;
    const top_k: usize = 2;
    const x_vals = try allocator.alloc(f32, seq * hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var x = try ctx.fromSlice(.f32, .{ seq, hidden }, x_vals);
    defer x.deinit();
    var selected: [seq * top_k]usize = undefined;
    var routing: [seq * top_k]f32 = undefined;
    for (&selected, &routing, 0..) |*s, *w, p| {
        s.* = (p * 5) % n_expert;
        w.* = 0.25 + 0.01 * @as(f32, @floatFromInt(p % 13));
    }
    var gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
    var up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
    var down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };
    var want = try ctx.moeExpertFfnBatch(&x, &fx.resident_gate, &fx.resident_up, &fx.resident_down, &selected, &routing, top_k, out_pe, .{ .op = .swiglu }, null, null);
    defer want.deinit();
    var got = try ctx.moeExpertFfnBatch(&x, &gate, &up, &down, &selected, &routing, top_k, out_pe, .{ .op = .swiglu }, null, null);
    defer got.deinit();
    try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    try std.testing.expect(store.copy_bytes[0].load(.monotonic) > 0);
    try std.testing.expect(store.copy_bytes[1].load(.monotonic) > 0);

    // Evicting decode rounds keep hammering the parallel path bit-exactly.
    var round: usize = 0;
    while (round < 24) : (round += 1) {
        const a = round % n_expert;
        const b = (round + 3) % n_expert;
        if (a == b) continue;
        try fx.expectDecodeWith(&ctx, store, &.{ a, b }, &.{ 0.6, 0.4 });
    }

    // Worker read failures surface with their true error: a store whose
    // PRIMARY is a truncated copy (gate section only, no mirror) fails
    // the up/down preads inside the workers; the caller's synchronous
    // retry re-fails and the acquire reports it.
    var short_buf: [160]u8 = undefined;
    const short_path = try std.fmt.bufPrint(&short_buf, "{s}.pshort", .{fx.path});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, short_path) catch {};
    defer cleanupSidecar(short_path);
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, short_path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(std.mem.sliceAsBytes(fx.gate_blocks));
        try writer.interface.flush();
    }
    var store2 = try ExpertStore.create(allocator, &.{short_path}, 1, .{
        .cache_slots_per_layer = 2,
        .io_workers = 3,
    });
    defer store2.destroy();
    try fx.registerLayer(store2);
    try store2.finalize();
    try std.testing.expectError(error.UnexpectedEndOfFile, store2.acquire(0, &.{ 0, 3, 5 }));
}

test "wave-split acquire: start resolves hits and launches misses, finish lands them" {
    const allocator = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.init(allocator, 4);
    defer fx.deinit();

    // Warm 0 and 3 into the LRU.
    try fx.store.acquire(0, &.{ 0, 3 });
    fx.store.release(0);

    // Start over {hit, miss, hit, miss}: the hits resolve immediately (and
    // only they are computable), the misses are in flight on the I/O pool.
    const ls = &fx.store.layers[0];
    const mask = try fx.store.acquireStart(0, &.{ 0, 5, 3, 6 });
    try std.testing.expectEqual(@as(u64, 0b0101), mask);
    try std.testing.expect(ls.resolved[0][0] != null);
    try std.testing.expect(ls.resolved[3][0] != null);
    try std.testing.expect(ls.resolved[5][0] == null);
    try std.testing.expect(ls.resolved[6][0] == null);
    try fx.store.acquireFinish();
    try std.testing.expect(ls.resolved[5][0] != null);
    try std.testing.expect(ls.resolved[6][0] != null);
    fx.store.release(0);

    // The finished misses were promoted: re-acquiring them is hit-only —
    // full mask at start, and finish is a no-op.
    const misses_before = fx.store.stats.misses;
    const mask2 = try fx.store.acquireStart(0, &.{ 5, 6 });
    try std.testing.expectEqual(@as(u64, 0b11), mask2);
    try fx.store.acquireFinish();
    fx.store.release(0);
    try std.testing.expectEqual(misses_before, fx.store.stats.misses);
}

test "wave-split decode: partial-resident selections are bit-exact through both waves" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 4);
    defer fx.deinit();

    // Warm {1, 4}, then route {1, 2, 4, 7}: wave 1 computes slots 0 and 2
    // under the in-flight reads of 2 and 7, wave 2 computes slots 1 and 3.
    try fx.store.acquire(0, &.{ 1, 4 });
    fx.store.release(0);
    try fx.expectDecodeMatches(&ctx, &.{ 1, 2, 4, 7 }, &.{ 0.4, 0.3, 0.2, 0.1 });

    // All 8 with exactly the LRU's 4 resident: interleaved hit/miss slots.
    try fx.expectDecodeMatches(&ctx, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, &.{ 0.2, 0.05, 0.15, 0.1, 0.05, 0.2, 0.05, 0.2 });

    // Duplicate selection slots of a resident expert plus one miss: the
    // dup slots share a wave, each with its own j-based output strip.
    // (After the full-8 decode's promotions the LRU holds {0, 3, 5, 6}.)
    try fx.expectDecodeMatches(&ctx, &.{ 0, 1, 0 }, &.{ 0.5, 0.3, 0.2 });
}

test "wave-split acquire degrades to synchronous reads without an io pool" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 2);
    defer fx.deinit();

    var store2 = try ExpertStore.create(allocator, &.{fx.path}, 1, .{
        .cache_slots_per_layer = 2,
        .io_workers = 0,
    });
    defer store2.destroy();
    try fx.registerLayer(store2);
    try store2.finalize();

    // No pool: start reads synchronously and reports everything resident.
    const mask = try store2.acquireStart(0, &.{ 0, 5 });
    try std.testing.expectEqual(@as(u64, 0b11), mask);
    try store2.acquireFinish(); // no-op
    store2.release(0);

    try fx.expectDecodeWith(&ctx, store2, &.{ 0, 5 }, &.{ 0.5, 0.5 });
}

test "wave-split acquire: worker read failures surface at finish and release stays safe" {
    const allocator = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.init(allocator, 2);
    defer fx.deinit();

    // Primary truncated to the gate section only: the workers' up/down
    // preads fail, the finish retry re-fails with the true error, and the
    // release that follows (the op's guard unwinding) must not publish the
    // half-read slabs into the LRU.
    var short_buf: [160]u8 = undefined;
    const short_path = try std.fmt.bufPrint(&short_buf, "{s}.wshort", .{fx.path});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, short_path) catch {};
    defer cleanupSidecar(short_path);
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, short_path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(std.mem.sliceAsBytes(fx.gate_blocks));
        try writer.interface.flush();
    }
    var store2 = try ExpertStore.create(allocator, &.{short_path}, 1, .{
        .cache_slots_per_layer = 2,
        .io_workers = 3,
    });
    defer store2.destroy();
    try fx.registerLayer(store2);
    try store2.finalize();

    const mask = try store2.acquireStart(0, &.{ 0, 3, 5 });
    try std.testing.expectEqual(@as(u64, 0), mask);
    try std.testing.expectError(error.UnexpectedEndOfFile, store2.acquireFinish());
    store2.release(0);
    // Nothing findable was promoted: the next acquire still misses (and
    // still fails on the truncated file, via the blocking path).
    try std.testing.expectError(error.UnexpectedEndOfFile, store2.acquire(0, &.{0}));
}

test "routing trace: request order round-trips through the sidecar" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 2);
    defer fx.deinit();

    var trace_buf: [160]u8 = undefined;
    const stamp = std.Io.Clock.real.now(std.testing.io).nanoseconds;
    const trace_path = try std.fmt.bufPrint(&trace_buf, "expert_trace_{d}.bin", .{stamp});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, trace_path) catch {};

    // The trace must record what the model ASKED for, in order — including
    // requests that were cache hits.
    const turns = [_][2]usize{ .{ 5, 6 }, .{ 1, 5 }, .{ 6, 5 } };
    {
        var store = try ExpertStore.create(allocator, &.{fx.path}, 1, .{
            .cache_slots_per_layer = 2,
            .trace_path = trace_path,
        });
        defer store.destroy();
        try fx.registerLayer(store);
        try store.finalize();
        for (&turns) |sel| try fx.expectDecodeWith(&ctx, store, &.{ sel[0], sel[1] }, &.{ 0.7, 0.3 });
    } // destroy writes the trace

    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, trace_path, allocator, .limited(1 << 20));
    defer allocator.free(bytes);
    try std.testing.expect(std.mem.eql(u8, bytes[0..8], ExpertStore.trace_magic));
    try std.testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, bytes[8..12], .little));
    try std.testing.expectEqual(@as(u64, 6), std.mem.readInt(u64, bytes[12..20], .little));
    var at: usize = 20;
    for (&turns) |sel| {
        for (sel) |eid| {
            try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, bytes[at..][0..4], .little));
            try std.testing.expectEqual(@as(u32, @intCast(eid)), std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little));
            at += 8;
        }
    }
    try std.testing.expectEqual(bytes.len, at);
}

test "auto-pin flatness guard: flat usage declines pinning, skewed usage pins" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 2);
    defer fx.deinit();

    // FLAT: every expert routed exactly once (a quantile-balanced router's
    // signature). Budget holds 2 of 8 used experts; the "hottest" two cover
    // exactly 2/8 of the traffic = the flat baseline -> pinning declined,
    // the whole budget stays with the LRU.
    for (0..n_expert / 2) |i| {
        try fx.expectDecodeMatches(&ctx, &.{ 2 * i, 2 * i + 1 }, &.{ 0.5, 0.5 });
    }
    try fx.store.saveUsage();
    {
        var store2 = try ExpertStore.create(allocator, &.{fx.path}, 1, .{
            .cache_slots_per_layer = 1,
            .auto_pin_min_history = 1,
        });
        defer store2.destroy();
        try fx.registerLayer(store2);
        store2.options.pin_bytes = 2 * fxSlabBytes(store2);
        try store2.finalize();
        try std.testing.expectEqual(@as(usize, 0), store2.pinned_experts);
        try std.testing.expect(store2.pins_declined_flat);
    }

    // SKEWED on top of the flat base: {5, 6} now dominate. The same budget
    // retains far more than the flat baseline -> pinning proceeds.
    for (0..8) |_| try fx.expectDecodeMatches(&ctx, &.{ 5, 6 }, &.{ 0.7, 0.3 });
    try fx.store.saveUsage();
    {
        var store3 = try ExpertStore.create(allocator, &.{fx.path}, 1, .{
            .cache_slots_per_layer = 1,
            .auto_pin_min_history = 1,
        });
        defer store3.destroy();
        try fx.registerLayer(store3);
        store3.options.pin_bytes = 2 * fxSlabBytes(store3);
        try store3.finalize();
        try std.testing.expectEqual(@as(usize, 2), store3.pinned_experts);
        try std.testing.expect(!store3.pins_declined_flat);
        try fx.expectDecodeWith(&ctx, store3, &.{ 5, 6 }, &.{ 0.7, 0.3 });
        try std.testing.expectEqual(@as(u64, 2), store3.stats.pin_hits);
    }
}

fn fxSlabBytes(store: *ExpertStore) usize {
    return store.layers[0].slab_bytes;
}

test "heat eviction: the frequent expert survives where pure LRU would evict it" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 3); // cap 3
    defer fx.deinit();

    // Build heat 0:3, 1:2, 2:1 with all three resident, then touch {1, 2}
    // so 0 ends LEAST-recent while staying HOTTEST — the state where the
    // two policies diverge (heat 0:3+1=4? no: after {1,2}: 0:3, 1:3, 2:2).
    try fx.expectDecodeMatches(&ctx, &.{ 0, 1 }, &.{ 0.6, 0.4 });
    try fx.expectDecodeMatches(&ctx, &.{ 0, 2 }, &.{ 0.6, 0.4 });
    try fx.expectDecodeMatches(&ctx, &.{ 0, 1 }, &.{ 0.6, 0.4 });
    try fx.expectDecodeMatches(&ctx, &.{ 1, 2 }, &.{ 0.5, 0.5 });
    // Miss on 3 (0 is protected as this acquire's hit): pure LRU would
    // evict the least-recent unprotected resident by stamp order (1,
    // touched before 2 in the {1, 2} acquire); heat eviction evicts 2
    // (heat 2 vs 1's heat 3).
    try fx.expectDecodeMatches(&ctx, &.{ 3, 0 }, &.{ 0.5, 0.5 });
    const misses_before = fx.store.stats.misses;
    // 1 must still be resident — heat kept it; pure LRU would have
    // evicted it and missed here.
    try fx.expectDecodeMatches(&ctx, &.{ 0, 1 }, &.{ 0.6, 0.4 });
    try std.testing.expectEqual(misses_before, fx.store.stats.misses);
}

test "l2 tier: built sparse mirror serves every miss bit-exact and reopens across stores" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var fx: Fixture = undefined;
    try fx.init(allocator, 2);
    defer fx.deinit();

    var l2_buf: [160]u8 = undefined;
    const l2_path = try std.fmt.bufPrint(&l2_buf, "{s}.l2", .{fx.path});
    defer {
        var buf: [176]u8 = undefined;
        std.Io.Dir.cwd().deleteFile(std.testing.io, l2_path) catch {};
        const idx = std.fmt.bufPrint(&buf, "{s}.idx", .{l2_path}) catch &.{};
        if (idx.len > 0) std.Io.Dir.cwd().deleteFile(std.testing.io, idx) catch {};
    }

    // Build + serve: 1 LRU slot so every distinct pair misses; the budget
    // covers all 8 experts (no usage history -> natural order fill).
    {
        var store = try ExpertStore.create(allocator, &.{fx.path}, 1, .{
            .cache_slots_per_layer = 1,
            .l2_path = l2_path,
            .l2_build_bytes = 1 << 30,
        });
        defer store.destroy();
        try fx.registerLayer(store);
        try store.finalize();

        try fx.expectDecodeWith(&ctx, store, &.{ 0, 3 }, &.{ 0.6, 0.4 });
        try fx.expectDecodeWith(&ctx, store, &.{ 5, 7 }, &.{ 0.5, 0.5 });
        // Every miss so far was L2-present (the build covered the pool).
        const hits = store.l2_expert_hits.load(.monotonic);
        try std.testing.expectEqual(store.stats.misses, hits);
        try std.testing.expect(hits >= 4);
        try std.testing.expectEqual(@as(u64, 0), store.l2_fallbacks.load(.monotonic));
    }

    // Reopen WITHOUT rebuilding: the persisted index must serve as-is.
    {
        var store = try ExpertStore.create(allocator, &.{fx.path}, 1, .{
            .cache_slots_per_layer = 1,
            .l2_path = l2_path,
        });
        defer store.destroy();
        try fx.registerLayer(store);
        try store.finalize();
        try fx.expectDecodeWith(&ctx, store, &.{ 2, 6 }, &.{ 0.5, 0.5 });
        try std.testing.expect(store.l2_expert_hits.load(.monotonic) >= 2);
    }
}

test "l2 tier: fold-mode flip drops coverage instead of corrupting folded slabs; all-folded build refuses" {
    // A tier snapshot records slab-prefix offsets of PRIMARY file bytes,
    // which is only valid for unfolded serving (e.g. FUCINA_PTQTP_FOLD=0
    // over a tie-fitted file). Pins for the two mode-flip hazards:
    // (a) l2Build over a store whose every layer is fold-served refuses
    //     (L2NoStripeableLayers) BEFORE truncating an existing tier;
    // (b) a tier built from unfolded specs, reopened with fold-serving
    //     specs, drops that layer's coverage — `readExpertPrefix` would
    //     otherwise overwrite the folded pack's head with plane bytes —
    //     and decode stays bit-exact vs the resident folded arm.
    const allocator = std.testing.allocator;
    const ptqtp = @import("../ptqtp.zig");
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const t_hidden: usize = 256;
    const t_ffn: usize = 512;
    const t_experts: usize = 2;
    const gu_rows = t_experts * t_ffn;
    const gu_bpc = t_hidden / qm.types.qk_k_block_size;
    const down_rows = t_experts * t_hidden;
    const down_bpc = t_ffn / qm.types.qk_k_block_size;

    const gate_w = try allocator.alloc(f32, gu_rows * t_hidden);
    defer allocator.free(gate_w);
    const up_w = try allocator.alloc(f32, gu_rows * t_hidden);
    defer allocator.free(up_w);
    const down_w = try allocator.alloc(f32, down_rows * t_ffn);
    defer allocator.free(down_w);
    for (gate_w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.023) * 0.8;
    for (up_w, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.031) * 1.1;
    for (down_w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.017 + 0.5) * 0.7;

    const tied_opts = ptqtp.Options{ .planes = 2, .max_iterations = 8, .tie_scales = true };
    var gate_pair = try ptqtp.quantizeMatrix(&ctx, gate_w, gu_rows, t_hidden, tied_opts);
    defer gate_pair.deinit(allocator);
    var up_pair = try ptqtp.quantizeMatrix(&ctx, up_w, gu_rows, t_hidden, tied_opts);
    defer up_pair.deinit(allocator);
    var down_pair = try ptqtp.quantizeMatrix(&ctx, down_w, down_rows, t_ffn, tied_opts);
    defer down_pair.deinit(allocator);

    const gate_folded = try foldExpertStack(allocator, gate_pair.plane1, gate_pair.plane2, t_experts, t_hidden, t_ffn);
    defer allocator.free(gate_folded);
    const up_folded = try foldExpertStack(allocator, up_pair.plane1, up_pair.plane2, t_experts, t_hidden, t_ffn);
    defer allocator.free(up_folded);
    const down_folded = try foldExpertStack(allocator, down_pair.plane1, down_pair.plane2, t_experts, t_ffn, t_hidden);
    defer allocator.free(down_folded);

    var resident_gate_planes: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ gate_pair.plane1, gate_pair.plane2, &.{} }, .plane_count = 2, .k = t_hidden, .n = gu_rows, .blocks_per_column = gu_bpc } };
    var resident_up_planes: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ up_pair.plane1, up_pair.plane2, &.{} }, .plane_count = 2, .k = t_hidden, .n = gu_rows, .blocks_per_column = gu_bpc } };
    var resident_down_planes: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ down_pair.plane1, down_pair.plane2, &.{} }, .plane_count = 2, .k = t_ffn, .n = down_rows, .blocks_per_column = down_bpc } };
    var resident_gate_folded: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ gate_pair.plane1, gate_pair.plane2, &.{} }, .plane_count = 2, .k = t_hidden, .n = gu_rows, .blocks_per_column = gu_bpc, .folded = gate_folded } };
    var resident_up_folded: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ up_pair.plane1, up_pair.plane2, &.{} }, .plane_count = 2, .k = t_hidden, .n = gu_rows, .blocks_per_column = gu_bpc, .folded = up_folded } };
    var resident_down_folded: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ down_pair.plane1, down_pair.plane2, &.{} }, .plane_count = 2, .k = t_ffn, .n = down_rows, .blocks_per_column = down_bpc, .folded = down_folded } };

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "expert_store_l2_foldflip_{d}.bin", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    defer cleanupSidecar(path);
    const gu_plane_bytes = gate_pair.plane1.len * @sizeOf(dtype_mod.BlockTQ2_0);
    const down_plane_bytes = down_pair.plane1.len * @sizeOf(dtype_mod.BlockTQ2_0);
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        for ([_][]const dtype_mod.BlockTQ2_0{
            gate_pair.plane1, gate_pair.plane2,
            up_pair.plane1,   up_pair.plane2,
            down_pair.plane1, down_pair.plane2,
        }) |plane| try writer.interface.writeAll(std.mem.sliceAsBytes(plane));
        try writer.interface.flush();
    }
    const specs_unfolded = [3]expert_store.ProjSpec{
        .{ .quant = .tq2_0, .file_offset = 0, .byte_len = gu_plane_bytes, .in_dim = t_hidden, .out_dim = t_ffn, .plane_count = 2, .plane_offsets = .{ gu_plane_bytes, 0 } },
        .{ .quant = .tq2_0, .file_offset = 2 * gu_plane_bytes, .byte_len = gu_plane_bytes, .in_dim = t_hidden, .out_dim = t_ffn, .plane_count = 2, .plane_offsets = .{ 3 * gu_plane_bytes, 0 } },
        .{ .quant = .tq2_0, .file_offset = 4 * gu_plane_bytes, .byte_len = down_plane_bytes, .in_dim = t_ffn, .out_dim = t_hidden, .plane_count = 2, .plane_offsets = .{ 4 * gu_plane_bytes + down_plane_bytes, 0 } },
    };
    var specs_folded = specs_unfolded;
    for (&specs_folded) |*s| s.fold = true;

    var l2_buf: [160]u8 = undefined;
    const l2_path = try std.fmt.bufPrint(&l2_buf, "{s}.l2", .{path});
    defer {
        var buf: [176]u8 = undefined;
        std.Io.Dir.cwd().deleteFile(std.testing.io, l2_path) catch {};
        const idx = std.fmt.bufPrint(&buf, "{s}.idx", .{l2_path}) catch &.{};
        if (idx.len > 0) std.Io.Dir.cwd().deleteFile(std.testing.io, idx) catch {};
    }

    const x_vals = try allocator.alloc(f32, t_hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13) % 167)) - 83)) / 83.0;
    var x = try ctx.fromSlice(.f32, .{ 1, t_hidden }, x_vals);
    defer x.deinit();
    const pair = [2]usize{ 0, 1 };
    const routing = [_]f32{ 0.6, 0.4 };

    // (1) Unfolded store builds + serves the tier; decode bit-exact vs the
    // resident per-plane arm and every miss covered.
    {
        var store = try ExpertStore.create(allocator, &.{path}, 1, .{
            .cache_slots_per_layer = 1,
            .l2_path = l2_path,
            .l2_build_bytes = 1 << 30,
        });
        defer store.destroy();
        try store.addLayer(0, specs_unfolded, t_experts);
        try store.finalize();
        var s_gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
        var s_up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
        var s_down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };
        var want = try ctx.moeExpertFfn(&x, &resident_gate_planes, &resident_up_planes, &resident_down_planes, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &s_gate, &s_up, &s_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer got.deinit();
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
        try std.testing.expect(store.l2_expert_hits.load(.monotonic) >= 2);
        try std.testing.expectEqual(@as(u64, 0), store.l2_fallbacks.load(.monotonic));
    }

    // (2) All-folded build refuses before truncating the existing tier.
    {
        var store = try ExpertStore.create(allocator, &.{path}, 1, .{
            .cache_slots_per_layer = 1,
            .l2_path = l2_path,
            .l2_build_bytes = 1 << 30,
        });
        defer store.destroy();
        try store.addLayer(0, specs_folded, t_experts);
        try std.testing.expectError(error.L2NoStripeableLayers, store.finalize());
    }

    // (3) Folded store over the unfolded-built tier: coverage dropped
    // (zero l2 reads), decode bit-exact vs the resident folded arm.
    {
        var store = try ExpertStore.create(allocator, &.{path}, 1, .{
            .cache_slots_per_layer = 1,
            .l2_path = l2_path,
        });
        defer store.destroy();
        try store.addLayer(0, specs_folded, t_experts);
        try store.finalize();
        var s_gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
        var s_up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
        var s_down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };
        var want = try ctx.moeExpertFfn(&x, &resident_gate_folded, &resident_up_folded, &resident_down_folded, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &s_gate, &s_up, &s_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer got.deinit();
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
        try std.testing.expectEqual(@as(u64, 0), store.l2_expert_hits.load(.monotonic));
        try std.testing.expectEqual(@as(u64, 0), store.l2_fallbacks.load(.monotonic));
    }
}

test "native folded (tq2_0_fx4) experts: streamed pack serves the one-pass kernel bit-exact; L2 stripes it" {
    // The pre-folded on-disk format: the pack bytes fill-time folding
    // would produce, stored as ONE expert-major tensor region — a single
    // pread per projection per miss, and slab bytes == file bytes, so the
    // L2 tier stripes these layers like any plain quant (the whole point
    // of the format). Pins: (a) streamed fx4 == resident folded arm
    // bitwise across cold/warm/evicting decode and batch; (b) an L2 tier
    // built over fx4 layers serves misses bit-exact with hits recorded;
    // (c) geometry legs (plane_count/fold/out_dim%4 rejections).
    const allocator = std.testing.allocator;
    const ptqtp = @import("../ptqtp.zig");
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const t_hidden: usize = 256;
    const t_ffn: usize = 512;
    const t_experts: usize = 2;
    const gu_rows = t_experts * t_ffn;
    const gu_bpc = t_hidden / qm.types.qk_k_block_size;
    const down_rows = t_experts * t_hidden;
    const down_bpc = t_ffn / qm.types.qk_k_block_size;

    const gate_w = try allocator.alloc(f32, gu_rows * t_hidden);
    defer allocator.free(gate_w);
    const up_w = try allocator.alloc(f32, gu_rows * t_hidden);
    defer allocator.free(up_w);
    const down_w = try allocator.alloc(f32, down_rows * t_ffn);
    defer allocator.free(down_w);
    for (gate_w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.021) * 0.85;
    for (up_w, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.027) * 1.15;
    for (down_w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.015 + 0.4) * 0.65;

    const tied_opts = ptqtp.Options{ .planes = 2, .max_iterations = 8, .tie_scales = true };
    var gate_pair = try ptqtp.quantizeMatrix(&ctx, gate_w, gu_rows, t_hidden, tied_opts);
    defer gate_pair.deinit(allocator);
    var up_pair = try ptqtp.quantizeMatrix(&ctx, up_w, gu_rows, t_hidden, tied_opts);
    defer up_pair.deinit(allocator);
    var down_pair = try ptqtp.quantizeMatrix(&ctx, down_w, down_rows, t_ffn, tied_opts);
    defer down_pair.deinit(allocator);

    const gate_folded = try foldExpertStack(allocator, gate_pair.plane1, gate_pair.plane2, t_experts, t_hidden, t_ffn);
    defer allocator.free(gate_folded);
    const up_folded = try foldExpertStack(allocator, up_pair.plane1, up_pair.plane2, t_experts, t_hidden, t_ffn);
    defer allocator.free(up_folded);
    const down_folded = try foldExpertStack(allocator, down_pair.plane1, down_pair.plane2, t_experts, t_ffn, t_hidden);
    defer allocator.free(down_folded);

    var resident_gate: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ gate_pair.plane1, gate_pair.plane2, &.{} }, .plane_count = 2, .k = t_hidden, .n = gu_rows, .blocks_per_column = gu_bpc, .folded = gate_folded } };
    var resident_up: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ up_pair.plane1, up_pair.plane2, &.{} }, .plane_count = 2, .k = t_hidden, .n = gu_rows, .blocks_per_column = gu_bpc, .folded = up_folded } };
    var resident_down: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ down_pair.plane1, down_pair.plane2, &.{} }, .plane_count = 2, .k = t_ffn, .n = down_rows, .blocks_per_column = down_bpc, .folded = down_folded } };

    // On disk: the expert-major PACK per projection — what --repack-native
    // writes as the base-named tq2_0_fx4 tensor.
    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "expert_store_fx4_{d}.bin", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    defer cleanupSidecar(path);
    const gu_pack_bytes = gate_folded.len * @sizeOf(qm.BlockTQ2_0Foldedx4);
    const down_pack_bytes = down_folded.len * @sizeOf(qm.BlockTQ2_0Foldedx4);
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        for ([_][]const qm.BlockTQ2_0Foldedx4{ gate_folded, up_folded, down_folded }) |pack|
            try writer.interface.writeAll(std.mem.sliceAsBytes(pack));
        try writer.interface.flush();
    }
    const specs_fx4 = [3]expert_store.ProjSpec{
        .{ .quant = .tq2_0_fx4, .file_offset = 0, .byte_len = gu_pack_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .tq2_0_fx4, .file_offset = gu_pack_bytes, .byte_len = gu_pack_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .tq2_0_fx4, .file_offset = 2 * gu_pack_bytes, .byte_len = down_pack_bytes, .in_dim = t_ffn, .out_dim = t_hidden },
    };

    // Geometry legs: fx4 rejects multi-plane, fill-fold, and out_dim % 4.
    {
        var store = try ExpertStore.create(allocator, &.{path}, 1, .{ .cache_slots_per_layer = 1 });
        defer store.destroy();
        var bad = specs_fx4;
        bad[0].plane_count = 2;
        try std.testing.expectError(error.InvalidExpertGeometry, store.addLayer(0, bad, t_experts));
        bad = specs_fx4;
        bad[0].fold = true;
        try std.testing.expectError(error.InvalidExpertGeometry, store.addLayer(0, bad, t_experts));
        bad = specs_fx4;
        bad[1].out_dim = t_ffn + 2;
        try std.testing.expectError(error.InvalidExpertGeometry, store.addLayer(0, bad, t_experts));
    }

    var l2_buf: [160]u8 = undefined;
    const l2_path = try std.fmt.bufPrint(&l2_buf, "{s}.l2", .{path});
    defer {
        var buf: [176]u8 = undefined;
        std.Io.Dir.cwd().deleteFile(std.testing.io, l2_path) catch {};
        const idx = std.fmt.bufPrint(&buf, "{s}.idx", .{l2_path}) catch &.{};
        if (idx.len > 0) std.Io.Dir.cwd().deleteFile(std.testing.io, idx) catch {};
    }

    var store = try ExpertStore.create(allocator, &.{path}, 1, .{
        .cache_slots_per_layer = 1,
        .l2_path = l2_path,
        .l2_build_bytes = 1 << 30,
    });
    defer store.destroy();
    try store.addLayer(0, specs_fx4, t_experts);
    try store.finalize();
    var s_gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
    var s_up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
    var s_down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };

    const x_vals = try allocator.alloc(f32, t_hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11) % 173)) - 86)) / 86.0;
    var x = try ctx.fromSlice(.f32, .{ 1, t_hidden }, x_vals);
    defer x.deinit();
    for ([_][2]usize{ .{ 0, 1 }, .{ 0, 1 }, .{ 1, 0 } }) |pair| {
        const routing = [_]f32{ 0.55, 0.45 };
        var want = try ctx.moeExpertFfn(&x, &resident_gate, &resident_up, &resident_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &s_gate, &s_up, &s_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer got.deinit();
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    }
    // The point of the format: fx4 layers ARE striped (vs the fill-folded
    // sibling format, which gets zero coverage).
    try std.testing.expect(store.l2_expert_hits.load(.monotonic) >= 2);
    try std.testing.expectEqual(@as(u64, 0), store.l2_fallbacks.load(.monotonic));

    // Batched prefill across both experts.
    const m: usize = 5;
    const top_k: usize = 2;
    const xb_vals = try allocator.alloc(f32, m * t_hidden);
    defer allocator.free(xb_vals);
    for (xb_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 19) % 181)) - 90)) / 90.0;
    var xb = try ctx.fromSlice(.f32, .{ m, t_hidden }, xb_vals);
    defer xb.deinit();
    const selected = [_]usize{ 0, 1, 1, 0, 0, 1, 1, 0, 0, 0 };
    const routing_b = [_]f32{ 0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.2, 0.8, 0.9, 0.1 };
    var want_b = try ctx.moeExpertFfnBatch(&xb, &resident_gate, &resident_up, &resident_down, &selected, &routing_b, top_k, t_ffn, .{ .op = .swiglu }, null, null);
    defer want_b.deinit();
    var got_b = try ctx.moeExpertFfnBatch(&xb, &s_gate, &s_up, &s_down, &selected, &routing_b, top_k, t_ffn, .{ .op = .swiglu }, null, null);
    defer got_b.deinit();
    try std.testing.expectEqualSlices(f32, want_b.dataConst(), got_b.dataConst());
}

test "slab-native fx4 records: one-pread misses serve bit-exact; geometry mismatch rejected; L2 stripes" {
    // The slab-native container: the on-disk record IS the RAM slab
    // (sections at their 16 KiB-aligned proj_off positions, tail padded),
    // registered via addLayerSlab. Pins: (a) streamed decode/batch over
    // one-pread misses == the resident folded arm bitwise; (b) a record
    // size that disagrees with the geometry-derived slab is rejected;
    // (c) the L2 tier builds and serves over slab-native layers.
    const allocator = std.testing.allocator;
    const ptqtp = @import("../ptqtp.zig");
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const t_hidden: usize = 256;
    const t_ffn: usize = 512;
    const t_experts: usize = 2;
    const gu_rows = t_experts * t_ffn;
    const gu_bpc = t_hidden / qm.types.qk_k_block_size;
    const down_rows = t_experts * t_hidden;
    const down_bpc = t_ffn / qm.types.qk_k_block_size;

    const gate_w = try allocator.alloc(f32, gu_rows * t_hidden);
    defer allocator.free(gate_w);
    const up_w = try allocator.alloc(f32, gu_rows * t_hidden);
    defer allocator.free(up_w);
    const down_w = try allocator.alloc(f32, down_rows * t_ffn);
    defer allocator.free(down_w);
    for (gate_w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.017) * 0.8;
    for (up_w, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.033) * 1.05;
    for (down_w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.011 + 0.6) * 0.7;

    const tied_opts = ptqtp.Options{ .planes = 2, .max_iterations = 8, .tie_scales = true };
    var gate_pair = try ptqtp.quantizeMatrix(&ctx, gate_w, gu_rows, t_hidden, tied_opts);
    defer gate_pair.deinit(allocator);
    var up_pair = try ptqtp.quantizeMatrix(&ctx, up_w, gu_rows, t_hidden, tied_opts);
    defer up_pair.deinit(allocator);
    var down_pair = try ptqtp.quantizeMatrix(&ctx, down_w, down_rows, t_ffn, tied_opts);
    defer down_pair.deinit(allocator);

    const gate_folded = try foldExpertStack(allocator, gate_pair.plane1, gate_pair.plane2, t_experts, t_hidden, t_ffn);
    defer allocator.free(gate_folded);
    const up_folded = try foldExpertStack(allocator, up_pair.plane1, up_pair.plane2, t_experts, t_hidden, t_ffn);
    defer allocator.free(up_folded);
    const down_folded = try foldExpertStack(allocator, down_pair.plane1, down_pair.plane2, t_experts, t_ffn, t_hidden);
    defer allocator.free(down_folded);

    var resident_gate: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ gate_pair.plane1, gate_pair.plane2, &.{} }, .plane_count = 2, .k = t_hidden, .n = gu_rows, .blocks_per_column = gu_bpc, .folded = gate_folded } };
    var resident_up: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ up_pair.plane1, up_pair.plane2, &.{} }, .plane_count = 2, .k = t_hidden, .n = gu_rows, .blocks_per_column = gu_bpc, .folded = up_folded } };
    var resident_down: MoeRhs = .{ .ptqtp = .{ .allocator = null, .planes = .{ down_pair.plane1, down_pair.plane2, &.{} }, .plane_count = 2, .k = t_ffn, .n = down_rows, .blocks_per_column = down_bpc, .folded = down_folded } };

    // Record layout mirroring expertSlabOffsets: 16 KiB-aligned sections.
    const gu_expert_bytes = (t_ffn / 4) * gu_bpc * @sizeOf(qm.BlockTQ2_0Foldedx4);
    const down_expert_bytes = (t_hidden / 4) * down_bpc * @sizeOf(qm.BlockTQ2_0Foldedx4);
    const salign: usize = 16384;
    var off: [3]usize = undefined;
    var at: usize = 0;
    for ([_]usize{ gu_expert_bytes, gu_expert_bytes, down_expert_bytes }, 0..) |sz, p| {
        at = std.mem.alignForward(usize, at, salign);
        off[p] = at;
        at += sz;
    }
    const record = std.mem.alignForward(usize, at, salign);

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "expert_store_slab_{d}.bin", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    defer cleanupSidecar(path);
    {
        const stack = try allocator.alloc(u8, record * t_experts);
        defer allocator.free(stack);
        @memset(stack, 0);
        const gu_fg = (t_ffn / 4) * gu_bpc;
        const down_fg = (t_hidden / 4) * down_bpc;
        for (0..t_experts) |e| {
            const dst = stack[e * record ..][0..record];
            @memcpy(dst[off[0]..][0..gu_expert_bytes], std.mem.sliceAsBytes(gate_folded[e * gu_fg ..][0..gu_fg]));
            @memcpy(dst[off[1]..][0..gu_expert_bytes], std.mem.sliceAsBytes(up_folded[e * gu_fg ..][0..gu_fg]));
            @memcpy(dst[off[2]..][0..down_expert_bytes], std.mem.sliceAsBytes(down_folded[e * down_fg ..][0..down_fg]));
        }
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(stack);
        try writer.interface.flush();
    }

    const specs = [3]expert_store.ProjSpec{
        .{ .quant = .tq2_0_fx4, .file_offset = 0, .byte_len = t_experts * gu_expert_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .tq2_0_fx4, .file_offset = 0, .byte_len = t_experts * gu_expert_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .tq2_0_fx4, .file_offset = 0, .byte_len = t_experts * down_expert_bytes, .in_dim = t_ffn, .out_dim = t_hidden },
    };

    // (b) record-size mismatch is a loud failure.
    {
        var store = try ExpertStore.create(allocator, &.{path}, 1, .{ .cache_slots_per_layer = 1 });
        defer store.destroy();
        try std.testing.expectError(error.InvalidExpertGeometry, store.addLayerSlab(0, specs, 0, 0, record + salign, t_experts));
    }

    var l2_buf: [160]u8 = undefined;
    const l2_path = try std.fmt.bufPrint(&l2_buf, "{s}.l2", .{path});
    defer {
        var buf: [176]u8 = undefined;
        std.Io.Dir.cwd().deleteFile(std.testing.io, l2_path) catch {};
        const idx = std.fmt.bufPrint(&buf, "{s}.idx", .{l2_path}) catch &.{};
        if (idx.len > 0) std.Io.Dir.cwd().deleteFile(std.testing.io, idx) catch {};
    }

    var store = try ExpertStore.create(allocator, &.{path}, 1, .{
        .cache_slots_per_layer = 1,
        .l2_path = l2_path,
        .l2_build_bytes = 1 << 30,
    });
    defer store.destroy();
    try store.addLayerSlab(0, specs, 0, 0, record, t_experts);
    try store.finalize();
    var s_gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
    var s_up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
    var s_down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };

    const x_vals = try allocator.alloc(f32, t_hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 157)) - 78)) / 78.0;
    var x = try ctx.fromSlice(.f32, .{ 1, t_hidden }, x_vals);
    defer x.deinit();
    for ([_][2]usize{ .{ 0, 1 }, .{ 0, 1 }, .{ 1, 0 } }) |pair| {
        const routing = [_]f32{ 0.65, 0.35 };
        var want = try ctx.moeExpertFfn(&x, &resident_gate, &resident_up, &resident_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &s_gate, &s_up, &s_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer got.deinit();
        try std.testing.expectEqualSlices(f32, want.dataConst(), got.dataConst());
    }
    try std.testing.expect(store.l2_expert_hits.load(.monotonic) >= 2);
    try std.testing.expectEqual(@as(u64, 0), store.l2_fallbacks.load(.monotonic));

    const m: usize = 4;
    const top_k: usize = 2;
    const xb_vals = try allocator.alloc(f32, m * t_hidden);
    defer allocator.free(xb_vals);
    for (xb_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 23) % 199)) - 99)) / 99.0;
    var xb = try ctx.fromSlice(.f32, .{ m, t_hidden }, xb_vals);
    defer xb.deinit();
    const selected = [_]usize{ 0, 1, 1, 0, 0, 1, 1, 0 };
    const routing_b = [_]f32{ 0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.2, 0.8 };
    var want_b = try ctx.moeExpertFfnBatch(&xb, &resident_gate, &resident_up, &resident_down, &selected, &routing_b, top_k, t_ffn, .{ .op = .swiglu }, null, null);
    defer want_b.deinit();
    var got_b = try ctx.moeExpertFfnBatch(&xb, &s_gate, &s_up, &s_down, &selected, &routing_b, top_k, t_ffn, .{ .op = .swiglu }, null, null);
    defer got_b.deinit();
    try std.testing.expectEqualSlices(f32, want_b.dataConst(), got_b.dataConst());
}

test "mxfp4 experts: streamed serving matches an exact q8_0 mirror; miss==hit bitwise; L2 stripes" {
    // Native fp4 expert stacks served straight from disk. The reference arm
    // exploits exact representability: with power-of-two block scales in the
    // f16-normal range and doubled-e2m1 codes (|c| <= 12), the SAME real
    // weights load losslessly as resident q8_0 — so the proven q8_0 path
    // cross-checks the whole new dispatch (decode, lhs pairing, geometry)
    // within float-schedule tolerance. Store-consistency pins are bitwise:
    // a cold-miss serve must equal the cached repeat, and the L2 tier must
    // stripe these plain-quant layers with zero fallbacks.
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const kv = [16]i8{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
    const t_hidden: usize = 256;
    const t_ffn: usize = 512;
    const t_experts: usize = 2;
    const gu_rows = t_experts * t_ffn;
    const gu_bpc = t_hidden / 32;
    const down_rows = t_experts * t_hidden;
    const down_bpc = t_ffn / 32;

    const Mirror = struct {
        fn halfScale(e: u8) f32 {
            return @bitCast(@as(u32, e - 1) << 23);
        }
        fn fill(mx: []dtype_mod.BlockMXFP4, q8: []dtype_mod.BlockQ8_0, seed: usize) void {
            for (mx, q8, 0..) |*w, *r, i| {
                // Deterministic pattern; scales 2^-24..2^-6 stay exact f16
                // (incl. the subnormal floor) and keep the FFN intermediates
                // small enough that their own Q8_0 block scales fit f16.
                w.e = @intCast(104 + (seed + i * 7) % 19);
                for (&w.qs, 0..) |*q, j| q.* = @intCast((seed * 31 + i * 17 + j * 13) % 256);
                r.d = f16Bits(halfScale(w.e));
                for (0..16) |j| {
                    r.qs[j] = kv[w.qs[j] & 0x0f];
                    r.qs[j + 16] = kv[w.qs[j] >> 4];
                }
            }
        }
    };

    var stacks: [3][]dtype_mod.BlockMXFP4 = undefined;
    var mirrors: [3][]dtype_mod.BlockQ8_0 = undefined;
    const counts = [3]usize{ gu_rows * gu_bpc, gu_rows * gu_bpc, down_rows * down_bpc };
    for (0..3) |p| {
        stacks[p] = try allocator.alloc(dtype_mod.BlockMXFP4, counts[p]);
        mirrors[p] = try allocator.alloc(dtype_mod.BlockQ8_0, counts[p]);
        Mirror.fill(stacks[p], mirrors[p], 5 + p * 97);
    }
    defer for (0..3) |p| {
        allocator.free(stacks[p]);
        allocator.free(mirrors[p]);
    };

    var resident_gate: MoeRhs = .{ .q8_0 = .{ .rows = .{ .allocator = null, .blocks = mirrors[0], .rows = gu_rows, .cols = t_hidden, .blocks_per_row = gu_bpc }, .k = t_hidden, .n = gu_rows } };
    var resident_up: MoeRhs = .{ .q8_0 = .{ .rows = .{ .allocator = null, .blocks = mirrors[1], .rows = gu_rows, .cols = t_hidden, .blocks_per_row = gu_bpc }, .k = t_hidden, .n = gu_rows } };
    var resident_down: MoeRhs = .{ .q8_0 = .{ .rows = .{ .allocator = null, .blocks = mirrors[2], .rows = down_rows, .cols = t_ffn, .blocks_per_row = down_bpc }, .k = t_ffn, .n = down_rows } };

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "expert_store_mxfp4_{d}.bin", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    defer cleanupSidecar(path);
    const gu_bytes = counts[0] * @sizeOf(dtype_mod.BlockMXFP4);
    const down_bytes = counts[2] * @sizeOf(dtype_mod.BlockMXFP4);
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        for (stacks) |pack| try writer.interface.writeAll(std.mem.sliceAsBytes(pack));
        try writer.interface.flush();
    }
    const specs = [3]expert_store.ProjSpec{
        .{ .quant = .mxfp4, .file_offset = 0, .byte_len = gu_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .mxfp4, .file_offset = gu_bytes, .byte_len = gu_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .mxfp4, .file_offset = 2 * gu_bytes, .byte_len = down_bytes, .in_dim = t_ffn, .out_dim = t_hidden },
    };

    const Approx = struct {
        fn expectClose(want: []const f32, got: []const f32) !void {
            try std.testing.expectEqual(want.len, got.len);
            var max_abs: f32 = 0;
            var max_rel: f32 = 0;
            var max_mag: f32 = 0;
            var nans: usize = 0;
            for (want, got) |w, g| {
                if (std.math.isNan(g) or std.math.isNan(w)) {
                    nans += 1;
                    continue;
                }
                const d = @abs(w - g);
                max_abs = @max(max_abs, d);
                max_mag = @max(max_mag, @abs(w));
                if (@abs(w) > 1e-3) max_rel = @max(max_rel, d / @abs(w));
            }
            // rel 2e-3: the mxfp4 and q8_0 kernels have different (per-arch)
            // accumulation schedules; x86's diverges up to ~9e-4 on near-1e-3
            // magnitudes. Real decode/layout bugs show up as percent-level.
            if (nans != 0 or max_rel > 2e-3 or max_abs > 1e-2) {
                std.debug.print("mxfp4-vs-q8_0 mirror divergence: nans={d} max_abs={e} max_rel={e} max_mag={e} want[0..4]={e} {e} {e} {e} got[0..4]={e} {e} {e} {e}\n", .{ nans, max_abs, max_rel, max_mag, want[0], want[1], want[2], want[3], got[0], got[1], got[2], got[3] });
                return error.TestExpectedApproxEqRel;
            }
        }
    };

    // Geometry leg: a non-multiple-of-32 in_dim must be rejected.
    {
        var store = try ExpertStore.create(allocator, &.{path}, 1, .{ .cache_slots_per_layer = 1 });
        defer store.destroy();
        var bad = specs;
        bad[0].in_dim = 240;
        try std.testing.expectError(error.InvalidExpertGeometry, store.addLayer(0, bad, t_experts));
    }

    // No-L2 leg first: isolates the serving path from the stripe tier.
    {
        var plain = try ExpertStore.create(allocator, &.{path}, 1, .{ .cache_slots_per_layer = 1 });
        defer plain.destroy();
        try plain.addLayer(0, specs, t_experts);
        try plain.finalize();
        var p_gate: MoeRhs = .{ .streamed = plain.streamedRhs(0, .gate) };
        var p_up: MoeRhs = .{ .streamed = plain.streamedRhs(0, .up) };
        var p_down: MoeRhs = .{ .streamed = plain.streamedRhs(0, .down) };
        const x0_vals = try allocator.alloc(f32, t_hidden);
        defer allocator.free(x0_vals);
        for (x0_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11) % 173)) - 86)) / 86.0;
        var x0 = try ctx.fromSlice(.f32, .{ 1, t_hidden }, x0_vals);
        defer x0.deinit();
        const pair0 = [2]usize{ 0, 1 };
        const routing0 = [_]f32{ 0.55, 0.45 };
        var want0 = try ctx.moeExpertFfn(&x0, &resident_gate, &resident_up, &resident_down, &pair0, &routing0, t_ffn, .{ .op = .swiglu }, null, null);
        defer want0.deinit();
        var got0 = try ctx.moeExpertFfn(&x0, &p_gate, &p_up, &p_down, &pair0, &routing0, t_ffn, .{ .op = .swiglu }, null, null);
        defer got0.deinit();
        try Approx.expectClose(want0.dataConst(), got0.dataConst());
    }

    var l2_buf: [160]u8 = undefined;
    const l2_path = try std.fmt.bufPrint(&l2_buf, "{s}.l2", .{path});
    defer {
        var buf: [176]u8 = undefined;
        std.Io.Dir.cwd().deleteFile(std.testing.io, l2_path) catch {};
        const idx = std.fmt.bufPrint(&buf, "{s}.idx", .{l2_path}) catch &.{};
        if (idx.len > 0) std.Io.Dir.cwd().deleteFile(std.testing.io, idx) catch {};
    }

    var store = try ExpertStore.create(allocator, &.{path}, 1, .{
        .cache_slots_per_layer = 1,
        .l2_path = l2_path,
        .l2_build_bytes = 1 << 30,
    });
    defer store.destroy();
    try store.addLayer(0, specs, t_experts);
    try store.finalize();
    var s_gate: MoeRhs = .{ .streamed = store.streamedRhs(0, .gate) };
    var s_up: MoeRhs = .{ .streamed = store.streamedRhs(0, .up) };
    var s_down: MoeRhs = .{ .streamed = store.streamedRhs(0, .down) };

    const x_vals = try allocator.alloc(f32, t_hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11) % 173)) - 86)) / 86.0;
    var x = try ctx.fromSlice(.f32, .{ 1, t_hidden }, x_vals);
    defer x.deinit();

    var first_pass: ?[]f32 = null;
    defer if (first_pass) |p| allocator.free(p);
    for (0..2) |round| {
        const pair = [2]usize{ 0, 1 };
        const routing = [_]f32{ 0.55, 0.45 };
        var want = try ctx.moeExpertFfn(&x, &resident_gate, &resident_up, &resident_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer want.deinit();
        var got = try ctx.moeExpertFfn(&x, &s_gate, &s_up, &s_down, &pair, &routing, t_ffn, .{ .op = .swiglu }, null, null);
        defer got.deinit();
        try Approx.expectClose(want.dataConst(), got.dataConst());
        if (round == 0) {
            first_pass = try allocator.dupe(f32, got.dataConst());
        } else {
            // Cold-miss serve vs cached repeat: BITWISE.
            try std.testing.expectEqualSlices(f32, first_pass.?, got.dataConst());
        }
    }
    // Plain-quant layers must be striped with no fallback reads.
    try std.testing.expect(store.l2_expert_hits.load(.monotonic) >= 1);
    try std.testing.expectEqual(@as(u64, 0), store.l2_fallbacks.load(.monotonic));

    // Batched prefill across both experts.
    const m: usize = 5;
    const top_k: usize = 2;
    const xb_vals = try allocator.alloc(f32, m * t_hidden);
    defer allocator.free(xb_vals);
    for (xb_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 19) % 181)) - 90)) / 90.0;
    var xb = try ctx.fromSlice(.f32, .{ m, t_hidden }, xb_vals);
    defer xb.deinit();
    const selected = [_]usize{ 0, 1, 1, 0, 0, 1, 1, 0, 0, 0 };
    const routing_b = [_]f32{ 0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.2, 0.8, 0.9, 0.1 };
    var want_b = try ctx.moeExpertFfnBatch(&xb, &resident_gate, &resident_up, &resident_down, &selected, &routing_b, top_k, t_ffn, .{ .op = .swiglu }, null, null);
    defer want_b.deinit();
    var got_b = try ctx.moeExpertFfnBatch(&xb, &s_gate, &s_up, &s_down, &selected, &routing_b, top_k, t_ffn, .{ .op = .swiglu }, null, null);
    defer got_b.deinit();
    try Approx.expectClose(want_b.dataConst(), got_b.dataConst());
}

test "l2 tier refuses a file whose expert bytes differ (content fingerprint)" {
    // A v3 tier index carries no content fingerprint (structure only:
    // layer/expert counts, prefix bounds), so it cannot detect a
    // same-shaped file with different expert bytes. The v4 index records
    // a per-layer content fingerprint of the primary bytes; opening
    // against a file with different expert content must fail loudly.
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const t_hidden: usize = 256;
    const t_ffn: usize = 512;
    const t_experts: usize = 2;
    const gu_bpc = t_hidden / 32;
    const down_bpc = t_ffn / 32;
    const gu_count = t_experts * t_ffn * gu_bpc;
    const down_count = t_experts * t_hidden * down_bpc;

    const blocks = try allocator.alloc(dtype_mod.BlockQ8_0, 2 * gu_count + down_count);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, i| {
        b.d = f16Bits(0.01);
        for (&b.qs, 0..) |*q, j| q.* = @intCast(@as(i32, @intCast((i * 13 + j * 7) % 255)) - 127);
    }

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "expert_store_l2fp_{d}.bin", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    defer cleanupSidecar(path);
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(std.mem.sliceAsBytes(blocks));
        try writer.interface.flush();
    }
    const gu_bytes = gu_count * @sizeOf(dtype_mod.BlockQ8_0);
    const down_bytes = down_count * @sizeOf(dtype_mod.BlockQ8_0);
    const specs = [3]expert_store.ProjSpec{
        .{ .quant = .q8_0, .file_offset = 0, .byte_len = gu_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .q8_0, .file_offset = gu_bytes, .byte_len = gu_bytes, .in_dim = t_hidden, .out_dim = t_ffn },
        .{ .quant = .q8_0, .file_offset = 2 * gu_bytes, .byte_len = down_bytes, .in_dim = t_ffn, .out_dim = t_hidden },
    };

    var l2_buf: [160]u8 = undefined;
    const l2_path = try std.fmt.bufPrint(&l2_buf, "{s}.l2", .{path});
    defer {
        var buf: [176]u8 = undefined;
        std.Io.Dir.cwd().deleteFile(std.testing.io, l2_path) catch {};
        const idx = std.fmt.bufPrint(&buf, "{s}.idx", .{l2_path}) catch &.{};
        if (idx.len > 0) std.Io.Dir.cwd().deleteFile(std.testing.io, idx) catch {};
    }
    {
        var store = try ExpertStore.create(allocator, &.{path}, 1, .{
            .cache_slots_per_layer = 1,
            .l2_path = l2_path,
            .l2_build_bytes = 1 << 30,
        });
        defer store.destroy();
        try store.addLayer(0, specs, t_experts);
        try store.finalize();
    }

    // Same shape, different expert content: flip one code in the first
    // expert's gate bytes and rewrite the file.
    blocks[0].qs[2] ^= 0x55;
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [4096]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(std.mem.sliceAsBytes(blocks));
        try writer.interface.flush();
    }
    {
        var store = try ExpertStore.create(allocator, &.{path}, 1, .{
            .cache_slots_per_layer = 1,
            .l2_path = l2_path,
        });
        defer store.destroy();
        try store.addLayer(0, specs, t_experts);
        try std.testing.expectError(error.InvalidExpertGeometry, store.finalize());
    }
}
