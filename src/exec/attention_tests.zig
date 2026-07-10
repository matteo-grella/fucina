//! Group-B tiled-attention parity tests. These drive already-`pub`
//! `attention.zig` internals (the tiled kernel + its Task), so they live
//! beside the module under test rather than inline in `exec.zig` (whose inline
//! Group-A tests deliberately drive `exec.zig`-private methods). Force-imported
//! by `attention.zig`'s `test` block. Excluded from arch-check (a `_tests.zig`
//! file).

const std = @import("std");

const exec_mod = @import("../exec.zig");
const exec_attention = @import("attention.zig");

const ExecContext = exec_mod.ExecContext;

const attention_tiled_min_q_seq = exec_attention.attention_tiled_min_q_seq;
const attention_tile_rows = exec_attention.attention_tile_rows;
const GroupedCausalAttentionTiledTask = exec_attention.GroupedCausalAttentionTiledTask;
const runGroupedCausalAttentionTiledTask = exec_attention.runGroupedCausalAttentionTiledTask;
const groupedCausalAttentionHeadPairs = exec_attention.groupedCausalAttentionHeadPairs;
const hasAdjacentKvHeadPairs = exec_attention.hasAdjacentKvHeadPairs;

// Tiled-vs-per-query parity is relative (1e-5), not bitwise: the online
// softmax visits keys in the same order but groups the summation differently
// (running-max rescale, normalization after accumulation, fused
// multiply-adds), so the two kernels agree only to rounding.
fn expectTiledAttentionClose(expected: []const f32, got: []const f32) !void {
    try std.testing.expectEqual(expected.len, got.len);
    for (expected, got) |e, g| {
        const tol = @max(1e-5 * @max(@abs(e), @abs(g)), 1e-6);
        try std.testing.expect(@abs(e - g) <= tol);
    }
}

fn checkTiledAttentionParity(
    ctx: *ExecContext,
    comptime kv_f16: bool,
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    kv_heads: usize,
    kv_head_for_head: []const usize,
    window: usize,
    d: usize,
    seed: u64,
) !void {
    // The reference must route through the unchanged per-query kernels.
    std.debug.assert(q_seq < attention_tiled_min_q_seq);
    const KvElem = if (kv_f16) f16 else f32;
    const allocator = std.testing.allocator;
    const scale_value: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const q_vals = try allocator.alloc(f32, q_seq * heads * d);
    defer allocator.free(q_vals);
    const k_vals = try allocator.alloc(KvElem, kv_seq * kv_heads * d);
    defer allocator.free(k_vals);
    const v_vals = try allocator.alloc(KvElem, kv_seq * kv_heads * d);
    defer allocator.free(v_vals);
    for (q_vals) |*x| x.* = random.floatNorm(f32);
    for (k_vals) |*x| x.* = if (kv_f16) @floatCast(random.floatNorm(f32)) else random.floatNorm(f32);
    for (v_vals) |*x| x.* = if (kv_f16) @floatCast(random.floatNorm(f32)) else random.floatNorm(f32);

    var q = try ctx.fromSliceRank(3, .{ q_seq, heads, d }, q_vals);
    defer q.deinit();
    var k = if (kv_f16)
        try ctx.fromSliceRankTyped(.f16, 3, .{ kv_seq, kv_heads, d }, k_vals)
    else
        try ctx.fromSliceRank(3, .{ kv_seq, kv_heads, d }, k_vals);
    defer k.deinit();
    var v = if (kv_f16)
        try ctx.fromSliceRankTyped(.f16, 3, .{ kv_seq, kv_heads, d }, v_vals)
    else
        try ctx.fromSliceRank(3, .{ kv_seq, kv_heads, d }, v_vals);
    defer v.deinit();

    var ref = if (kv_f16)
        try ctx.groupedCausalAttentionF16KvWindowed(&q, &k, &v, kv_head_for_head, scale_value, window)
    else
        try ctx.groupedCausalAttentionWindowed(&q, &k, &v, kv_head_for_head, scale_value, window);
    defer ref.deinit();

    var got = try ctx.emptyRank(2, .{ q_seq, heads * d });
    defer got.deinit();
    const base = GroupedCausalAttentionTiledTask(KvElem){
        .q_data = q.dataConst(),
        .k_data = k.dataConst(),
        .v_data = v.dataConst(),
        .out_data = got.data(),
        .kv_head_for_head = kv_head_for_head,
        .q_seq = q_seq,
        .kv_seq = kv_seq,
        .source_offset = kv_seq - q_seq,
        .heads = heads,
        .d = d,
        .kv_heads = kv_heads,
        .scale_value = scale_value,
        .window = window,
        .n_tiles = 0, // set by groupedCausalAttentionTiledRun
        .work_start = 0,
        .work_end = 0,
    };
    if (hasAdjacentKvHeadPairs(kv_head_for_head, heads, kv_heads)) {
        exec_attention.groupedCausalAttentionTiledRun(&ctx.rt, KvElem, 2, base);
    } else {
        exec_attention.groupedCausalAttentionTiledRun(&ctx.rt, KvElem, 1, base);
    }

    try expectTiledAttentionClose(ref.dataConst(), got.dataConst());
}

test "grouped causal attention query-tiled kernel parity vs per-query kernels" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const map_pair = [_]usize{ 0, 0, 1, 1 }; // 2:1 adjacent GQA — pair path
    const map_one = [_]usize{ 0, 1 }; // 1:1 — general path
    const map_eight = [_]usize{0} ** 8; // 8:1 — general path

    // q_seq around the tile size (1 .. 2*Q_TILE+1): exact tiles, partial last
    // tiles, all-duplicate tiles; causal offset 0 and > 0.
    var s: usize = 1;
    while (s <= 2 * attention_tile_rows + 1) : (s += 1) {
        try checkTiledAttentionParity(&ctx, false, s, s, 4, 2, &map_pair, 0, 64, 1000 + s);
        try checkTiledAttentionParity(&ctx, false, s, s + 9, 4, 2, &map_pair, 0, 64, 2000 + s);
    }
    // GQA mappings 1:1 and 8:1 (general path), d = 128.
    try checkTiledAttentionParity(&ctx, false, 9, 9, 2, 2, &map_one, 0, 128, 31);
    try checkTiledAttentionParity(&ctx, false, 9, 21, 8, 1, &map_eight, 0, 128, 32);
    // d not a multiple of the SIMD width exercises the scalar tails.
    try checkTiledAttentionParity(&ctx, false, 7, 12, 4, 2, &map_pair, 0, 12, 33);
    // Windowed: window < kv_seq, window > kv_seq, and windows below the tile
    // span (the fully-masked-row edge where m stays -inf for early keys).
    try checkTiledAttentionParity(&ctx, false, 11, 19, 4, 2, &map_pair, 5, 64, 41);
    try checkTiledAttentionParity(&ctx, false, 11, 19, 4, 2, &map_pair, 64, 64, 42);
    try checkTiledAttentionParity(&ctx, false, 11, 19, 4, 2, &map_pair, 1, 64, 43);
    try checkTiledAttentionParity(&ctx, false, 11, 19, 2, 2, &map_one, 3, 64, 44);
    // kv_seq extremes: 1, and a long odd 4099 with a large causal offset.
    try checkTiledAttentionParity(&ctx, false, 1, 1, 4, 2, &map_pair, 0, 64, 51);
    try checkTiledAttentionParity(&ctx, false, 16, 4099, 4, 2, &map_pair, 0, 64, 52);
    try checkTiledAttentionParity(&ctx, false, 17, 4099, 2, 2, &map_one, 600, 64, 53);
    // f16 KV mirrors: pair + general mapping, offset, windowed, long odd kv.
    try checkTiledAttentionParity(&ctx, true, 9, 9, 4, 2, &map_pair, 0, 64, 61);
    try checkTiledAttentionParity(&ctx, true, 9, 17, 8, 1, &map_eight, 0, 128, 62);
    try checkTiledAttentionParity(&ctx, true, 11, 19, 4, 2, &map_pair, 5, 64, 63);
    try checkTiledAttentionParity(&ctx, true, 16, 4099, 4, 2, &map_pair, 0, 64, 64);
}

test "tiled attention NaN logit poisons the query row like the per-query kernels" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const S = 6;
    const H = 4;
    const KVH = 2;
    const D = 16;
    const kv_head_for_head = [_]usize{ 0, 0, 1, 1 };
    const nan_query = 3;
    const nan_head = 1;

    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();
    var q_vals: [S * H * D]f32 = undefined;
    var k_vals: [S * KVH * D]f32 = undefined;
    var v_vals: [S * KVH * D]f32 = undefined;
    for (&q_vals) |*x| x.* = random.floatNorm(f32);
    for (&k_vals) |*x| x.* = random.floatNorm(f32);
    for (&v_vals) |*x| x.* = random.floatNorm(f32);
    for (0..D) |f| q_vals[(nan_query * H + nan_head) * D + f] = std.math.nan(f32);

    var q = try ctx.fromSliceRank(3, .{ S, H, D }, &q_vals);
    defer q.deinit();
    var k = try ctx.fromSliceRank(3, .{ S, KVH, D }, &k_vals);
    defer k.deinit();
    var v = try ctx.fromSliceRank(3, .{ S, KVH, D }, &v_vals);
    defer v.deinit();

    var ref = try ctx.groupedCausalAttention(&q, &k, &v, &kv_head_for_head, 0.5);
    defer ref.deinit();

    var got = try ctx.emptyRank(2, .{ S, H * D });
    defer got.deinit();
    exec_attention.groupedCausalAttentionTiledRun(&ctx.rt, f32, 2, .{
        .q_data = q.dataConst(),
        .k_data = k.dataConst(),
        .v_data = v.dataConst(),
        .out_data = got.data(),
        .kv_head_for_head = &kv_head_for_head,
        .q_seq = S,
        .kv_seq = S,
        .source_offset = 0,
        .heads = H,
        .d = D,
        .kv_heads = KVH,
        .scale_value = 0.5,
        .window = 0,
        .n_tiles = 0, // set by groupedCausalAttentionTiledRun
        .work_start = 0,
        .work_end = 0,
    });

    for (0..S) |qi| {
        for (0..H) |h| {
            const base = (qi * H + h) * D;
            if (qi == nan_query and h == nan_head) {
                // The NaN row poisons fully on both paths (vexpf propagates NaN).
                for (0..D) |f| {
                    try std.testing.expect(std.math.isNan(ref.dataConst()[base + f]));
                    try std.testing.expect(std.math.isNan(got.dataConst()[base + f]));
                }
            } else {
                try expectTiledAttentionClose(ref.dataConst()[base..][0..D], got.dataConst()[base..][0..D]);
            }
        }
    }
}

test "tiled attention: huge usize SWA windows behave as full causal (dispatch clamp)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // q_seq >= attention_tiled_min_q_seq, so the public entry routes to the
    // tiled kernel — whose i32 SWA-bound math was UB for window >= 2^31
    // before the dispatch clamp. Any window >= kv_seq must reproduce the
    // unwindowed result; the reference is the unchanged per-query pair
    // kernel, called directly (the public entry would also route to tiled).
    const S = 64;
    const KV = 80;
    const H = 4;
    const KVH = 2;
    const D = 16;
    const kv_head_for_head = [_]usize{ 0, 0, 1, 1 };
    const scale_value: f32 = 0.25;

    var prng = std.Random.DefaultPrng.init(0x5117);
    const random = prng.random();
    const q_vals = try allocator.alloc(f32, S * H * D);
    defer allocator.free(q_vals);
    const k_vals = try allocator.alloc(f32, KV * KVH * D);
    defer allocator.free(k_vals);
    const v_vals = try allocator.alloc(f32, KV * KVH * D);
    defer allocator.free(v_vals);
    for (q_vals) |*x| x.* = random.floatNorm(f32);
    for (k_vals) |*x| x.* = random.floatNorm(f32);
    for (v_vals) |*x| x.* = random.floatNorm(f32);

    var q = try ctx.fromSliceRank(3, .{ S, H, D }, q_vals);
    defer q.deinit();
    var k = try ctx.fromSliceRank(3, .{ KV, KVH, D }, k_vals);
    defer k.deinit();
    var v = try ctx.fromSliceRank(3, .{ KV, KVH, D }, v_vals);
    defer v.deinit();

    var ref = try ctx.emptyRank(2, .{ S, H * D });
    defer ref.deinit();
    var scores: [KV * 2]f32 = undefined;
    groupedCausalAttentionHeadPairs(f32, .{
        .q_data = q.dataConst(),
        .k_data = k.dataConst(),
        .v_data = v.dataConst(),
        .out_data = ref.data(),
        .q_seq = S,
        .kv_seq = KV,
        .source_offset = KV - S,
        .heads = H,
        .d = D,
        .kv_heads = KVH,
        .scale_value = scale_value,
        .window = 0,
        .kv_head_start = 0,
        .kv_head_end = KVH,
        .scores = &scores,
    });

    var unwindowed = try ctx.groupedCausalAttention(&q, &k, &v, &kv_head_for_head, scale_value);
    defer unwindowed.deinit();

    for ([_]usize{ std.math.maxInt(usize), @as(usize, 1) << 40 }) |window| {
        var got = try ctx.groupedCausalAttentionWindowed(&q, &k, &v, &kv_head_for_head, scale_value, window);
        defer got.deinit();
        // Tiled vs per-query: same math, different summation grouping.
        try expectTiledAttentionClose(ref.dataConst(), got.dataConst());
        // Tiled vs tiled: the clamped window's mask is identical to the
        // unwindowed mask, so the result is bitwise equal.
        try std.testing.expectEqualSlices(f32, unwindowed.dataConst(), got.dataConst());
    }
}

test "tiled attention pool gate: small jobs stay serial and match the parallel split bitwise" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x6a7e);
    const random = prng.random();

    // Below the work gate (48*48*2*16 = 73728 < vector_matmul_work_threshold/2)
    // the tiled dispatch must not spin up the work pool: the whole job runs
    // as one task on the calling thread.
    {
        const S = 48;
        const H = 2;
        const KVH = 1;
        const D = 16;
        const kv_head_for_head = [_]usize{ 0, 0 };
        const q_vals = try allocator.alloc(f32, S * H * D);
        defer allocator.free(q_vals);
        const kv_vals = try allocator.alloc(f32, S * KVH * D);
        defer allocator.free(kv_vals);
        for (q_vals) |*x| x.* = random.floatNorm(f32);
        for (kv_vals) |*x| x.* = random.floatNorm(f32);
        var q = try ctx.fromSliceRank(3, .{ S, H, D }, q_vals);
        defer q.deinit();
        var k = try ctx.fromSliceRank(3, .{ S, KVH, D }, kv_vals);
        defer k.deinit();
        var v = try ctx.fromSliceRank(3, .{ S, KVH, D }, kv_vals);
        defer v.deinit();
        var out = try ctx.groupedCausalAttention(&q, &k, &v, &kv_head_for_head, 0.25);
        defer out.deinit();
        try std.testing.expect(!ctx.rt.work_pool_ready);
    }

    // Above the gate (48*48*8*64 = 1179648) the job splits across the pool.
    // Each (head, query) output row is written by exactly one task, so the
    // partitioned result must be BITWISE identical to the same job run as a
    // single task on the calling thread.
    {
        const S = 48;
        const KV = 48;
        const H = 8;
        const KVH = 4;
        const D = 64;
        const kv_head_for_head = [_]usize{ 0, 0, 1, 1, 2, 2, 3, 3 };
        const q_vals = try allocator.alloc(f32, S * H * D);
        defer allocator.free(q_vals);
        const k_vals = try allocator.alloc(f32, KV * KVH * D);
        defer allocator.free(k_vals);
        const v_vals = try allocator.alloc(f32, KV * KVH * D);
        defer allocator.free(v_vals);
        for (q_vals) |*x| x.* = random.floatNorm(f32);
        for (k_vals) |*x| x.* = random.floatNorm(f32);
        for (v_vals) |*x| x.* = random.floatNorm(f32);
        var q = try ctx.fromSliceRank(3, .{ S, H, D }, q_vals);
        defer q.deinit();
        var k = try ctx.fromSliceRank(3, .{ KV, KVH, D }, k_vals);
        defer k.deinit();
        var v = try ctx.fromSliceRank(3, .{ KV, KVH, D }, v_vals);
        defer v.deinit();

        var serial_out = try ctx.emptyRank(2, .{ S, H * D });
        defer serial_out.deinit();
        const q_tile = attention_tile_rows / 2;
        var task = GroupedCausalAttentionTiledTask(f32){
            .q_data = q.dataConst(),
            .k_data = k.dataConst(),
            .v_data = v.dataConst(),
            .out_data = serial_out.data(),
            .kv_head_for_head = &kv_head_for_head,
            .q_seq = S,
            .kv_seq = KV,
            .source_offset = 0,
            .heads = H,
            .d = D,
            .kv_heads = KVH,
            .scale_value = 0.25,
            .window = 0,
            .n_tiles = (S + q_tile - 1) / q_tile,
            .work_start = 0,
            .work_end = 0,
        };
        task.work_end = KVH * task.n_tiles;
        const run = runGroupedCausalAttentionTiledTask(f32, 2);
        run(&task);

        var pooled_out = try ctx.emptyRank(2, .{ S, H * D });
        defer pooled_out.deinit();
        var base = task;
        base.out_data = pooled_out.data();
        base.n_tiles = 0; // set by groupedCausalAttentionTiledRun
        base.work_end = 0;
        exec_attention.groupedCausalAttentionTiledRun(&ctx.rt, f32, 2, base);

        try std.testing.expectEqualSlices(f32, serial_out.dataConst(), pooled_out.dataConst());
    }
}

/// Multi-stream ragged decode attention vs per-stream single calls: the
/// multi entry runs the SAME per-query kernels per (stream, head unit), so
/// each stream's rows must be BITWISE identical to its own single-stream
/// `groupedCausalAttention{F16,Q8}Kv` call — regardless of batch
/// composition or the parallel/inline dispatch arm taken.
fn checkMultiKvAttentionParity(
    ctx: *ExecContext,
    comptime q8: bool,
    lens: []const usize,
    heads: usize,
    kv_heads: usize,
    kv_head_for_head: []const usize,
    d: usize,
    seed: u64,
) !void {
    const allocator = std.testing.allocator;
    const BlockQ8_0 = exec_attention.BlockQ8_0;
    const n = lens.len;
    const scale_value: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    const q_vals = try allocator.alloc(f32, n * heads * d);
    defer allocator.free(q_vals);
    for (q_vals) |*x| x.* = random.floatNorm(f32);
    var q = try ctx.fromSliceRank(3, .{ n, heads, d }, q_vals);
    defer q.deinit();

    const KvElem = if (q8) BlockQ8_0 else f16;
    const row_elems = if (q8) kv_heads * (d / exec_attention.q8_0_block_size) else kv_heads * d;

    const k_owned = try allocator.alloc([]KvElem, n);
    defer allocator.free(k_owned);
    const v_owned = try allocator.alloc([]KvElem, n);
    defer allocator.free(v_owned);
    var built: usize = 0;
    defer for (0..built) |s| {
        allocator.free(k_owned[s]);
        allocator.free(v_owned[s]);
    };
    for (lens, 0..) |len_s, s| {
        k_owned[s] = try allocator.alloc(KvElem, len_s * row_elems);
        errdefer allocator.free(k_owned[s]);
        v_owned[s] = try allocator.alloc(KvElem, len_s * row_elems);
        errdefer allocator.free(v_owned[s]);
        if (comptime q8) {
            const f32_vals = try allocator.alloc(f32, len_s * kv_heads * d);
            defer allocator.free(f32_vals);
            for (f32_vals) |*x| x.* = random.floatNorm(f32);
            var rows = try ctx.fromSliceRank(3, .{ len_s, kv_heads, d }, f32_vals);
            defer rows.deinit();
            try ctx.quantizeF32RowsToQ8_0Into(&rows, k_owned[s]);
            for (f32_vals) |*x| x.* = random.floatNorm(f32);
            var v_rows = try ctx.fromSliceRank(3, .{ len_s, kv_heads, d }, f32_vals);
            defer v_rows.deinit();
            try ctx.quantizeF32RowsToQ8_0Into(&v_rows, v_owned[s]);
        } else {
            for (k_owned[s]) |*x| x.* = @floatCast(random.floatNorm(f32));
            for (v_owned[s]) |*x| x.* = @floatCast(random.floatNorm(f32));
        }
        // Last: the iteration errdefers above cover a mid-iteration failure;
        // once counted, the function-level defer owns the pair.
        built += 1;
    }

    const ks = try allocator.alloc([]const KvElem, n);
    defer allocator.free(ks);
    const vs = try allocator.alloc([]const KvElem, n);
    defer allocator.free(vs);
    for (ks, vs, k_owned, v_owned) |*k_s, *v_s, k_o, v_o| {
        k_s.* = k_o;
        v_s.* = v_o;
    }

    var out = if (comptime q8)
        try ctx.groupedCausalAttentionMultiQ8Kv(&q, ks, vs, lens, kv_heads, kv_head_for_head, scale_value)
    else
        try ctx.groupedCausalAttentionMultiF16Kv(&q, ks, vs, lens, kv_heads, kv_head_for_head, scale_value);
    defer out.deinit();

    for (lens, 0..) |len_s, s| {
        var q_s = try ctx.fromSliceRank(3, .{ 1, heads, d }, q_vals[s * heads * d ..][0 .. heads * d]);
        defer q_s.deinit();
        var ref = if (comptime q8)
            try ctx.groupedCausalAttentionQ8Kv(&q_s, ks[s], vs[s], len_s, kv_heads, kv_head_for_head, scale_value)
        else blk: {
            var k_t = try ctx.fromSliceRankTyped(.f16, 3, .{ len_s, kv_heads, d }, k_owned[s]);
            defer k_t.deinit();
            var v_t = try ctx.fromSliceRankTyped(.f16, 3, .{ len_s, kv_heads, d }, v_owned[s]);
            defer v_t.deinit();
            break :blk try ctx.groupedCausalAttentionF16Kv(&q_s, &k_t, &v_t, kv_head_for_head, scale_value);
        };
        defer ref.deinit();
        try std.testing.expectEqualSlices(f32, ref.dataConst(), out.dataConst()[s * heads * d ..][0 .. heads * d]);
    }
}

test "multi-stream ragged decode attention == per-stream single calls (f16 + q8_0)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const map_pair = [_]usize{ 0, 0, 1, 1 }; // 2:1 adjacent GQA — pair path
    const map_one = [_]usize{ 0, 1 }; // 1:1 — general path
    const map_four = [_]usize{ 0, 0, 0, 0 }; // 4:1 — general path

    // Ragged lens, single stream, len-1 streams; d exercises SIMD tails.
    try checkMultiKvAttentionParity(&ctx, false, &.{ 1, 5, 3 }, 4, 2, &map_pair, 32, 101);
    try checkMultiKvAttentionParity(&ctx, false, &.{7}, 4, 2, &map_pair, 64, 102);
    try checkMultiKvAttentionParity(&ctx, false, &.{ 4, 4, 4, 4 }, 2, 2, &map_one, 48, 103);
    try checkMultiKvAttentionParity(&ctx, false, &.{ 2, 9 }, 4, 1, &map_four, 20, 104);
    // Crosses the parallel-dispatch gate (sum(lens) * heads * d >= 512 Ki):
    // exercises the length-weighted task partition + pooled scratch arm.
    try checkMultiKvAttentionParity(&ctx, false, &.{ 700, 300, 500, 100 }, 4, 2, &map_pair, 128, 105);
    try checkMultiKvAttentionParity(&ctx, false, &.{ 1700, 5 }, 2, 2, &map_one, 128, 106);
    // q8_0 mirrors (d % 32 == 0).
    try checkMultiKvAttentionParity(&ctx, true, &.{ 1, 5, 3 }, 4, 2, &map_pair, 32, 201);
    try checkMultiKvAttentionParity(&ctx, true, &.{ 4, 4 }, 2, 2, &map_one, 64, 202);
    try checkMultiKvAttentionParity(&ctx, true, &.{ 700, 300, 500, 100 }, 4, 2, &map_pair, 128, 203);
}

test "multi-stream attention rejects bad shapes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const map = [_]usize{ 0, 0 };
    const q_vals = [_]f32{0} ** (2 * 2 * 16);
    var q = try ctx.fromSliceRank(3, .{ 2, 2, 16 }, &q_vals);
    defer q.deinit();
    const k_vals = [_]f16{0} ** (3 * 16);
    const ks = [_][]const f16{ k_vals[0..], k_vals[0..] };

    // Length-0 stream.
    try std.testing.expectError(
        error.InvalidShape,
        ctx.groupedCausalAttentionMultiF16Kv(&q, &ks, &ks, &.{ 3, 0 }, 1, &map, 0.25),
    );
    // Span shorter than its declared len.
    try std.testing.expectError(
        error.InvalidShape,
        ctx.groupedCausalAttentionMultiF16Kv(&q, &ks, &ks, &.{ 3, 4 }, 1, &map, 0.25),
    );
    // lens count != stream count.
    try std.testing.expectError(
        error.InvalidShape,
        ctx.groupedCausalAttentionMultiF16Kv(&q, &ks, &ks, &.{3}, 1, &map, 0.25),
    );
}

/// How the bias tensor of a biased-bidirectional parity case is filled.
const BiasKind = enum {
    /// Uniform random values in [-2, 2] — the general mixed-row case.
    random,
    /// A single constant everywhere: softmax shift-invariance makes the
    /// result equal the UNbiased path (up to summation-order rounding).
    constant,
    /// OmniVoice's uncond CFG pattern (lm.zig buildUncondBias): rows
    /// sq < u_len get +1.0 on keys [0, u_len) and 0.0 on the tail; rows
    /// sq >= u_len get +1.0 only on their own diagonal.
    uncond,
};

/// Cross-implementation bound for the biased-bidirectional checks: the
/// reference below is computed in f64 (an effectively exact oracle), so the
/// full f32 rounding of the kernel under test — dot, vexpf-vs-libm exp, and
/// the (kv_seq)-term weighted V sum, whose near-uniform softmax rows shrink
/// the output to ~1/sqrt(kv_seq) while the absolute rounding stays put —
/// lands on one side of the comparison. Looser than the tiled-vs-per-query
/// bound (both sides f32, same exp) but still ~3 decimal orders below any
/// real bias-application bug.
fn expectBiasedAttentionClose(expected: []const f64, got: []const f32) !void {
    try std.testing.expectEqual(expected.len, got.len);
    for (expected, got) |e64, g| {
        const e: f32 = @floatCast(e64);
        const tol = @max(5e-5 * @max(@abs(e), @abs(g)), 5e-6);
        try std.testing.expect(@abs(e - g) <= tol);
    }
}

/// Biased bidirectional attention vs a naive per-head row-softmax reference
/// (the same composition lm.zig's per-head oracle uses, computed in f64):
/// probs = softmax(dot * scale + bias[q][kv]) over the FULL key range,
/// out = probs·V. Relative tolerance, not bitwise — the fused kernels
/// reorder the summation (online softmax on the tiled path, 3-pass on the
/// per-query paths) and run entirely in f32.
fn checkBiasedBidirectionalParity(
    ctx: *ExecContext,
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    kv_heads: usize,
    kv_head_for_head: []const usize,
    d: usize,
    seed: u64,
    bias_kind: BiasKind,
) !void {
    const allocator = std.testing.allocator;
    const scale_value: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const q_vals = try allocator.alloc(f32, q_seq * heads * d);
    defer allocator.free(q_vals);
    const k_vals = try allocator.alloc(f32, kv_seq * kv_heads * d);
    defer allocator.free(k_vals);
    const v_vals = try allocator.alloc(f32, kv_seq * kv_heads * d);
    defer allocator.free(v_vals);
    for (q_vals) |*x| x.* = random.floatNorm(f32);
    for (k_vals) |*x| x.* = random.floatNorm(f32);
    for (v_vals) |*x| x.* = random.floatNorm(f32);

    const bias_vals = try allocator.alloc(f32, q_seq * kv_seq);
    defer allocator.free(bias_vals);
    switch (bias_kind) {
        .random => for (bias_vals) |*x| {
            x.* = 4 * (random.float(f32) - 0.5);
        },
        .constant => @memset(bias_vals, 1.0),
        .uncond => {
            const u_len = @max(q_seq / 2, 1);
            @memset(bias_vals, 0.0);
            for (0..@min(u_len, q_seq)) |sq| @memset(bias_vals[sq * kv_seq ..][0..u_len], 1.0);
            for (u_len..q_seq) |sq| bias_vals[sq * kv_seq + @min(sq, kv_seq - 1)] = 1.0;
        },
    }

    var q = try ctx.fromSliceRank(3, .{ q_seq, heads, d }, q_vals);
    defer q.deinit();
    var k = try ctx.fromSliceRank(3, .{ kv_seq, kv_heads, d }, k_vals);
    defer k.deinit();
    var v = try ctx.fromSliceRank(3, .{ kv_seq, kv_heads, d }, v_vals);
    defer v.deinit();
    var bias = try ctx.fromSliceRank(2, .{ q_seq, kv_seq }, bias_vals);
    defer bias.deinit();

    var got = try ctx.groupedBidirectionalAttentionBiased(&q, &k, &v, kv_head_for_head, scale_value, &bias);
    defer got.deinit();

    // Naive f64 reference: full [kv_seq] score row per (head, query),
    // row-max shift, plain sequential sums.
    const ref = try allocator.alloc(f64, q_seq * heads * d);
    defer allocator.free(ref);
    const row = try allocator.alloc(f64, kv_seq);
    defer allocator.free(row);
    for (0..heads) |head_i| {
        const kv_head_i = kv_head_for_head[head_i];
        for (0..q_seq) |qi| {
            var max_score = -std.math.inf(f64);
            for (0..kv_seq) |si| {
                var dot: f64 = 0;
                for (0..d) |f| dot += @as(f64, q_vals[(qi * heads + head_i) * d + f]) * @as(f64, k_vals[(si * kv_heads + kv_head_i) * d + f]);
                const score = dot * scale_value + bias_vals[qi * kv_seq + si];
                row[si] = score;
                max_score = @max(max_score, score);
            }
            var sum_exp: f64 = 0;
            for (row) |*x| {
                x.* = @exp(x.* - max_score);
                sum_exp += x.*;
            }
            const out_base = (qi * heads + head_i) * d;
            @memset(ref[out_base..][0..d], 0);
            for (0..kv_seq) |si| {
                const weight = row[si] / sum_exp;
                for (0..d) |f| ref[out_base + f] += weight * @as(f64, v_vals[(si * kv_heads + kv_head_i) * d + f]);
            }
        }
    }
    try expectBiasedAttentionClose(ref, got.dataConst());

    // Constant bias: softmax shift-invariance — must also match the PLAIN
    // bidirectional path (same tolerance; the two may take different
    // summation orders through exp).
    if (bias_kind == .constant) {
        var plain = try ctx.groupedBidirectionalAttention(&q, &k, &v, kv_head_for_head, scale_value);
        defer plain.deinit();
        try expectTiledAttentionClose(plain.dataConst(), got.dataConst());
    }
}

test "grouped bidirectional biased attention matches the naive row-softmax composition" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const map_pair = [_]usize{ 0, 0, 1, 1 }; // 2:1 adjacent GQA — pair path
    const map_one = [_]usize{ 0, 1 }; // 1:1 — general path
    const map_eight = [_]usize{0} ** 8; // 8:1 — general path

    // Short prefill (q_seq < attention_tiled_min_q_seq): the per-query
    // 3-pass kernels, pair + general mappings, q_seq < kv_seq, odd-d tail.
    try checkBiasedBidirectionalParity(&ctx, 7, 7, 4, 2, &map_pair, 64, 301, .random);
    try checkBiasedBidirectionalParity(&ctx, 9, 21, 2, 2, &map_one, 128, 302, .random);
    try checkBiasedBidirectionalParity(&ctx, 5, 12, 8, 1, &map_eight, 64, 303, .random);
    try checkBiasedBidirectionalParity(&ctx, 7, 12, 4, 2, &map_pair, 12, 304, .random);
    // Long prefill (q_seq >= attention_tiled_min_q_seq = 48): the tiled
    // online-softmax kernel, incl. a partial last tile (q_seq = 65) and the
    // pooled dispatch (253^2-scale work like the OmniVoice design clip).
    std.debug.assert(64 >= attention_tiled_min_q_seq);
    try checkBiasedBidirectionalParity(&ctx, 64, 64, 4, 2, &map_pair, 64, 311, .random);
    try checkBiasedBidirectionalParity(&ctx, 65, 80, 2, 2, &map_one, 64, 312, .random);
    try checkBiasedBidirectionalParity(&ctx, 253, 253, 4, 2, &map_pair, 128, 313, .random);
    // The OmniVoice uncond +1/0 bias pattern (mixed rows: prompt-span rows +
    // diagonal-only padding rows), per-query and tiled.
    try checkBiasedBidirectionalParity(&ctx, 10, 10, 4, 2, &map_pair, 64, 321, .uncond);
    try checkBiasedBidirectionalParity(&ctx, 64, 64, 4, 2, &map_pair, 64, 322, .uncond);
    try checkBiasedBidirectionalParity(&ctx, 65, 65, 8, 1, &map_eight, 128, 323, .uncond);
}

test "grouped bidirectional biased attention: constant bias equals the plain bidirectional path" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const map_pair = [_]usize{ 0, 0, 1, 1 };
    const map_one = [_]usize{ 0, 1 };
    // Per-query kernels and the tiled kernel: softmax shift-invariance.
    try checkBiasedBidirectionalParity(&ctx, 9, 9, 4, 2, &map_pair, 64, 331, .constant);
    try checkBiasedBidirectionalParity(&ctx, 11, 19, 2, 2, &map_one, 128, 332, .constant);
    try checkBiasedBidirectionalParity(&ctx, 64, 72, 4, 2, &map_pair, 64, 333, .constant);
}

test "grouped bidirectional biased attention rejects a mis-shaped bias" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const map = [_]usize{ 0, 0 };
    const q_vals = [_]f32{0} ** (3 * 2 * 16);
    const kv_vals = [_]f32{0} ** (3 * 1 * 16);
    var q = try ctx.fromSliceRank(3, .{ 3, 2, 16 }, &q_vals);
    defer q.deinit();
    var k = try ctx.fromSliceRank(3, .{ 3, 1, 16 }, &kv_vals);
    defer k.deinit();
    var v = try ctx.fromSliceRank(3, .{ 3, 1, 16 }, &kv_vals);
    defer v.deinit();

    const bias_vals = [_]f32{0} ** (3 * 4);
    var bias = try ctx.fromSliceRank(2, .{ 3, 4 }, &bias_vals);
    defer bias.deinit();
    try std.testing.expectError(
        error.InvalidShape,
        ctx.groupedBidirectionalAttentionBiased(&q, &k, &v, &map, 0.25, &bias),
    );
}
