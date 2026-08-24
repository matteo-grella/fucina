const std = @import("std");
const fucina = @import("fucina");
const engram = @import("engram.zig");

const ExecContext = fucina.ExecContext;

// A small everything-on config: two n-gram orders, two heads per order,
// two hyper-connection streams, dilation = max_ngram (the reference
// default).
fn testConfig() engram.Config {
    return .{
        .hidden_size = 12,
        .hc_mult = 2,
        .max_ngram_size = 3,
        .n_embed_per_ngram = 8,
        .n_head_per_ngram = 2,
        .engram_vocab_size = &.{ 23, 31 },
        .kernel_size = 3,
        .pad_id = 2,
    };
}

const test_layer_ids = [_]usize{ 0, 2 };
// Realistic magnitudes: half_bound-scale odd multipliers force i64 wrap.
const test_multipliers = [_]i64{
    6148914691236517205,  -7905747460161236407, 4611686018427387905,
    -1234567890123456789, 987654321987654321,   -43,
};

test "engram prime chain is global across layers and orders" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var plan = try engram.HashPlan.initWithMultipliers(allocator, testConfig(), &test_layer_ids, &test_multipliers, null);
    defer plan.deinit();

    // Layer 0: order-2 heads search up from 22 -> 23, 29; order-3 heads
    // from 30 -> 31, 37. Layer 2 continues the GLOBAL seen set: order-2
    // from 22 again but 23/29 are taken -> 37 is taken -> 41, 43; order-3
    // from 30: 31/37/41/43 taken -> 47, 53.
    try std.testing.expectEqualSlices(i64, &.{ 23, 29, 31, 37 }, plan.head_mods[0..4]);
    try std.testing.expectEqualSlices(i64, &.{ 41, 43, 47, 53 }, plan.head_mods[4..8]);
    try std.testing.expectEqualSlices(usize, &.{ 0, 23, 52, 83 }, plan.head_offsets[0..4]);
    try std.testing.expectEqual(@as(usize, 23 + 29 + 31 + 37), plan.table_rows[0]);
    try std.testing.expectEqual(@as(usize, 41 + 43 + 47 + 53), plan.table_rows[1]);
}

test "engram hashInto matches the tensor-op hash and stays in range" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var plan = try engram.HashPlan.initWithMultipliers(allocator, testConfig(), &test_layer_ids, &test_multipliers, null);
    defer plan.deinit();

    const ids = [_]i64{ 5, 1, 3, 3, 7, 0, 2, 11, 5 };
    const heads = testConfig().headsPerLayer();

    for (0..test_layer_ids.len) |slot| {
        const rows = try allocator.alloc(usize, ids.len * heads);
        defer allocator.free(rows);
        try plan.hashInto(slot, &ids, rows);

        for (rows) |row| try std.testing.expect(row < plan.table_rows[slot]);

        var rows_t = try plan.hashTensor(&ctx, slot, &ids);
        defer rows_t.deinit();
        const tensor_rows = try rows_t.dataConst();
        try std.testing.expectEqual(rows.len, tensor_rows.len);
        for (rows, tensor_rows) |host, tens| {
            try std.testing.expectEqual(@as(i64, @intCast(host)), tens);
        }
    }
}

test "engram compression lookup maps ids and the pad id" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    // Raw vocab of 6 collapsing to 3 compressed ids; pad_id 2 -> 1.
    const lookup = [_]i64{ 0, 1, 1, 2, 0, 2 };
    var plan = try engram.HashPlan.initWithMultipliers(allocator, testConfig(), &test_layer_ids, &test_multipliers, &lookup);
    defer plan.deinit();
    try std.testing.expectEqual(@as(i64, 1), plan.pad_compressed);

    const raw = [_]i64{ 0, 5, 4, 2 };
    var compressed: [4]i64 = undefined;
    try plan.compressInto(&raw, &compressed);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 0, 1 }, &compressed);
}

test "engram graft zero-init layer emits exactly zero and trains" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const cfg = testConfig();
    var plan = try engram.HashPlan.initWithMultipliers(allocator, cfg, &test_layer_ids, &test_multipliers, null);
    defer plan.deinit();

    var layer = try engram.Layer.initRandom(&ctx, allocator, cfg, plan.table_rows[0], 7, .{ .graft_zero_init = true });
    defer layer.deinit();

    const seq = 5;
    const ids = [_]i64{ 5, 1, 3, 3, 7 };
    const rows = try allocator.alloc(usize, seq * cfg.headsPerLayer());
    defer allocator.free(rows);
    try plan.hashInto(0, &ids, rows);

    const hidden_len = seq * cfg.hc_mult * cfg.hidden_size;
    const hidden_data = try allocator.alloc(f32, hidden_len);
    defer allocator.free(hidden_data);
    var prng = std.Random.DefaultPrng.init(3);
    for (hidden_data) |*v| v.* = prng.random().floatNorm(f32);

    var hidden = try engram.Hidden.variableFromSlice(&ctx, .{ seq, cfg.hc_mult, cfg.hidden_size }, hidden_data);
    defer hidden.deinit();

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);

    var out = try layer.forward(&ctx, &hidden, rows, null);
    defer out.deinit();

    // Zero value projection => the whole side branch is exactly zero:
    // grafting onto a frozen model is bitwise identity at step 0.
    for (out.asRawTensor().dataConst()) |v| try std.testing.expectEqual(@as(f32, 0), v);

    // ...but gradients still reach the value projection (and only flow
    // into the rest through it).
    var loss = try out.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gw = (try layer.value_w.grad(&ctx)).?;
    defer gw.deinit();
    var nonzero = false;
    for (gw.asRawTensor().dataConst()) |v| {
        if (v != 0) nonzero = true;
    }
    try std.testing.expect(nonzero);
}

test "engram forwardResidual squeezes the single-stream path" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var cfg = testConfig();
    cfg.hc_mult = 1;
    var plan = try engram.HashPlan.initWithMultipliers(allocator, cfg, &test_layer_ids, &test_multipliers, null);
    defer plan.deinit();
    var layer = try engram.Layer.initRandom(&ctx, allocator, cfg, plan.table_rows[0], 11, .{});
    defer layer.deinit();

    const seq = 4;
    const ids = [_]i64{ 9, 0, 4, 4 };
    const rows = try allocator.alloc(usize, seq * cfg.headsPerLayer());
    defer allocator.free(rows);
    try plan.hashInto(0, &ids, rows);

    const hidden_data = try allocator.alloc(f32, seq * cfg.hidden_size);
    defer allocator.free(hidden_data);
    for (hidden_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.25 - 0.5;
    var hidden = try fucina.Tensor(.{ .seq, .d }).fromSlice(&ctx, .{ seq, cfg.hidden_size }, hidden_data);
    defer hidden.deinit();

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);

    var out = try layer.forwardResidual(&ctx, &hidden, rows, null);
    defer out.deinit();
    try std.testing.expectEqual(seq, out.dim(.seq));
    try std.testing.expectEqual(cfg.hidden_size, out.dim(.d));
}

test "engram whole-model state dict roundtrips including multipliers" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const cfg = testConfig();
    var model = try engram.Engram.init(&ctx, allocator, cfg, &test_layer_ids, 17, null, .{});
    defer model.deinit();

    const buf = try allocator.alloc(u8, 1 << 20);
    defer allocator.free(buf);
    var writer = std.Io.Writer.fixed(buf);
    try model.saveStateDict(&writer);
    const written = writer.buffered();

    var other = try engram.Engram.init(&ctx, allocator, cfg, &test_layer_ids, 99, null, .{});
    defer other.deinit();
    // Different seed => different multipliers before the load.
    try std.testing.expect(!std.mem.eql(i64, model.plan.multipliers, other.plan.multipliers));

    var reader = std.Io.Reader.fixed(written);
    try other.loadStateDict(&reader, .{});
    try std.testing.expectEqualSlices(i64, model.plan.multipliers, other.plan.multipliers);

    const a = try model.layers[0].table.dataConst();
    const b = try other.layers[0].table.dataConst();
    try std.testing.expectEqualSlices(f32, a, b);
}
