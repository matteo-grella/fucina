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
        exec_attention.groupedCausalAttentionTiledRun(ctx, KvElem, 2, base);
    } else {
        exec_attention.groupedCausalAttentionTiledRun(ctx, KvElem, 1, base);
    }

    try expectTiledAttentionClose(ref.dataConst(), got.dataConst());
}

test "BLAS-strip attention backward matches the register-tiled route" {
    // Drives both backward route functions directly on the same task (no env
    // flag), so the BLAS route is exercised in every CI run on BLAS builds.
    // Tolerance pin: the routes differ only in contraction association
    // (sgemm vs register tiles), so gradients agree to f32 contraction
    // tolerance across causal/window, GQA-partial vs direct, stats on/off.
    const backend_mod = @import("../backend.zig");
    if (comptime !(backend_mod.active_kind == .native and backend_mod.native_uses_blas)) return;

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const random = prng.random();

    const cases = [_]struct {
        q_seq: usize,
        kv_seq: usize,
        heads: usize,
        kv_heads: usize,
        d: usize,
        window: usize,
        use_stats: bool,
    }{
        .{ .q_seq = 96, .kv_seq = 96, .heads = 4, .kv_heads = 4, .d = 32, .window = 0, .use_stats = false },
        .{ .q_seq = 80, .kv_seq = 80, .heads = 4, .kv_heads = 2, .d = 16, .window = 0, .use_stats = false },
        .{ .q_seq = 70, .kv_seq = 70, .heads = 2, .kv_heads = 1, .d = 24, .window = 33, .use_stats = false },
        .{ .q_seq = 96, .kv_seq = 96, .heads = 2, .kv_heads = 2, .d = 32, .window = 0, .use_stats = true },
    };

    for (cases) |case| {
        const q_len = case.q_seq * case.heads * case.d;
        const kv_len = case.kv_seq * case.kv_heads * case.d;
        const q_data = try allocator.alloc(f32, q_len);
        defer allocator.free(q_data);
        const k_data = try allocator.alloc(f32, kv_len);
        defer allocator.free(k_data);
        const v_data = try allocator.alloc(f32, kv_len);
        defer allocator.free(v_data);
        const gy_data = try allocator.alloc(f32, q_len);
        defer allocator.free(gy_data);
        for (q_data) |*value| value.* = random.floatNorm(f32) * 0.5;
        for (k_data) |*value| value.* = random.floatNorm(f32) * 0.5;
        for (v_data) |*value| value.* = random.floatNorm(f32) * 0.5;
        for (gy_data) |*value| value.* = random.floatNorm(f32) * 0.5;

        var kv_map: [8]usize = undefined;
        const group = case.heads / case.kv_heads;
        for (0..case.heads) |head_i| kv_map[head_i] = head_i / group;
        const partial_mode = group > 1;

        const scale_value: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(case.d)));

        // Forward-saved {max, sum_exp} stats when requested: max over the
        // row's scaled scores, then the exp sum — matching the forward's
        // definition (scaled scores, row-local max).
        var stats: ?[]f32 = null;
        defer if (stats) |values| allocator.free(values);
        if (case.use_stats) {
            const values = try allocator.alloc(f32, case.heads * case.q_seq * 2);
            for (0..case.heads) |head_i| {
                const kv_head_base = kv_map[head_i] * case.d;
                for (0..case.q_seq) |query_i| {
                    var max_score: f32 = -std.math.inf(f32);
                    var scores_buf: [128]f32 = undefined;
                    const active = query_i + 1;
                    for (0..active) |key_i| {
                        var dot: f32 = 0;
                        for (0..case.d) |feature_i| {
                            dot += q_data[query_i * case.heads * case.d + head_i * case.d + feature_i] *
                                k_data[key_i * case.kv_heads * case.d + kv_head_base + feature_i];
                        }
                        scores_buf[key_i] = dot * scale_value;
                        max_score = @max(max_score, scores_buf[key_i]);
                    }
                    var sum_exp: f32 = 0;
                    for (0..active) |key_i| sum_exp += @exp(scores_buf[key_i] - max_score);
                    values[head_i * case.q_seq * 2 + query_i * 2] = max_score;
                    values[head_i * case.q_seq * 2 + query_i * 2 + 1] = sum_exp;
                }
            }
            stats = values;
        }

        const target_len = if (partial_mode) case.heads * case.kv_seq * case.d else kv_len;
        const RouteOut = struct { dq: []f32, dk: []f32, dv: []f32 };
        var outs: [2]RouteOut = undefined;
        for (0..2) |route_i| {
            const dq = try allocator.alloc(f32, q_len);
            const dk = try allocator.alloc(f32, target_len);
            const dv = try allocator.alloc(f32, target_len);
            @memset(dq, 0);
            @memset(dk, 0);
            @memset(dv, 0);
            const tile_rows = if (route_i == 0)
                exec_attention.attention_bwd_tile_rows
            else
                exec_attention.attention_bwd_blas_tile_rows;
            const scratch = try allocator.alloc(f32, 2 * tile_rows * case.kv_seq);
            defer allocator.free(scratch);

            const task = exec_attention.GroupedCausalAttentionBackwardTiledTask{
                .q_data = q_data,
                .k_data = k_data,
                .v_data = v_data,
                .gy_data = gy_data,
                .stats = stats,
                .q_grad = dq,
                .dk_target = dk,
                .dv_target = dv,
                .partial_mode = partial_mode,
                .kv_head_for_head = kv_map[0..case.heads],
                .q_seq = case.q_seq,
                .kv_seq = case.kv_seq,
                .source_offset = 0,
                .heads = case.heads,
                .d = case.d,
                .kv_heads = case.kv_heads,
                .scale_value = scale_value,
                .window = case.window,
                .causal = true,
                .scratch = scratch,
                .head_start = 0,
                .head_end = case.heads,
            };
            if (route_i == 0) {
                exec_attention.groupedCausalAttentionBackwardTiles(task);
            } else {
                exec_attention.groupedCausalAttentionBackwardBlasTiles(task);
            }
            outs[route_i] = .{ .dq = dq, .dk = dk, .dv = dv };
        }
        defer for (outs) |out| {
            allocator.free(out.dq);
            allocator.free(out.dk);
            allocator.free(out.dv);
        };

        for ([_][2][]const f32{
            .{ outs[0].dq, outs[1].dq },
            .{ outs[0].dk, outs[1].dk },
            .{ outs[0].dv, outs[1].dv },
        }) |pair| {
            for (pair[0], pair[1]) |reference, actual| {
                try std.testing.expectApproxEqAbs(reference, actual, 2e-4);
            }
        }
    }
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
    exec_attention.groupedCausalAttentionTiledRun(&ctx, f32, 2, .{
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
        try std.testing.expect(!ctx.work_pool_ready);
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
        exec_attention.groupedCausalAttentionTiledRun(&ctx, f32, 2, base);

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

const exec = @import("../exec.zig");
const parallel = @import("../parallel.zig");

fn checkWindowedAttention(
    ctx: *ExecContext,
    comptime S: usize,
    comptime H: usize,
    comptime KV: usize,
    kv_head_for_head: []const usize,
    window: usize,
    scale_value: f32,
) !void {
    const D = 4;
    var q_vals: [S * H * D]f32 = undefined;
    var k_vals: [S * KV * D]f32 = undefined;
    var v_vals: [S * KV * D]f32 = undefined;
    for (&q_vals, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.3) * 1.3;
    for (&k_vals, 0..) |*x, i| x.* = @cos(@as(f32, @floatFromInt(i)) * 0.21) - 0.2;
    for (&v_vals, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.17 + 0.4);

    var q = try ctx.fromSliceRank(3, .{ S, H, D }, &q_vals);
    defer q.deinit();
    var k = try ctx.fromSliceRank(3, .{ S, KV, D }, &k_vals);
    defer k.deinit();
    var v = try ctx.fromSliceRank(3, .{ S, KV, D }, &v_vals);
    defer v.deinit();

    var got = try ctx.groupedCausalAttentionWindowed(&q, &k, &v, kv_head_for_head, scale_value, window);
    defer got.deinit();

    // Scalar reference: query p attends keys [max(0, p-window+1), p] (window 0 = full).
    var expected: [S * H * D]f32 = undefined;
    for (0..H) |h| {
        const kvh = kv_head_for_head[h];
        for (0..S) |p| {
            const lo = if (window != 0 and p + 1 > window) p + 1 - window else 0;
            var scores: [S]f32 = undefined;
            var maxs: f32 = -std.math.inf(f32);
            for (lo..p + 1) |j| {
                var dot: f32 = 0;
                for (0..D) |d| dot += q_vals[(p * H + h) * D + d] * k_vals[(j * KV + kvh) * D + d];
                scores[j] = dot * scale_value;
                maxs = @max(maxs, scores[j]);
            }
            var sum: f32 = 0;
            for (lo..p + 1) |j| {
                scores[j] = @exp(scores[j] - maxs);
                sum += scores[j];
            }
            for (0..D) |d| {
                var acc: f32 = 0;
                for (lo..p + 1) |j| acc += (scores[j] / sum) * v_vals[(j * KV + kvh) * D + d];
                expected[(p * H + h) * D + d] = acc;
            }
        }
    }
    for (got.dataConst(), expected) |g, e| try std.testing.expectApproxEqAbs(e, g, 1e-5);
}

test "grouped causal attention sliding window (pair + heads kernels)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Head-pair kernel (heads == 2*kv_heads, adjacent mapping) — Qwen-style GQA.
    try checkWindowedAttention(&ctx, 7, 4, 2, &.{ 0, 0, 1, 1 }, 3, 0.5);
    // Single-head kernel (heads == kv_heads).
    try checkWindowedAttention(&ctx, 7, 2, 2, &.{ 0, 1 }, 3, 0.5);
    // window >= seq behaves as full causal (must match the no-window result).
    try checkWindowedAttention(&ctx, 5, 4, 2, &.{ 0, 0, 1, 1 }, 100, 0.25);
    try checkWindowedAttention(&ctx, 5, 4, 2, &.{ 0, 0, 1, 1 }, 0, 0.25);
    // window == 1: each query attends only its own position.
    try checkWindowedAttention(&ctx, 6, 2, 2, &.{ 0, 1 }, 1, 0.5);
}

test "grouped causal attention tiled dispatch matches a naive reference at long q_seq" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Above the dispatch threshold the public entry points route to the tiled
    // kernel (parallel split included); check against a naive f64 reference.
    const S = 67; // odd, > attention_tiled_min_q_seq
    const KV = 80;
    const H = 4;
    const KVH = 2;
    const D = 16;
    const kv_head_for_head = [_]usize{ 0, 0, 1, 1 };
    const scale_value: f32 = 0.25;

    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    var q_vals: [S * H * D]f32 = undefined;
    var k_vals: [KV * KVH * D]f32 = undefined;
    var v_vals: [KV * KVH * D]f32 = undefined;
    for (&q_vals) |*x| x.* = random.floatNorm(f32);
    for (&k_vals) |*x| x.* = random.floatNorm(f32);
    for (&v_vals) |*x| x.* = random.floatNorm(f32);

    var q = try ctx.fromSliceRank(3, .{ S, H, D }, &q_vals);
    defer q.deinit();
    var k = try ctx.fromSliceRank(3, .{ KV, KVH, D }, &k_vals);
    defer k.deinit();
    var v = try ctx.fromSliceRank(3, .{ KV, KVH, D }, &v_vals);
    defer v.deinit();

    for ([_]usize{ 0, 13 }) |window| {
        var got = try ctx.groupedCausalAttentionWindowed(&q, &k, &v, &kv_head_for_head, scale_value, window);
        defer got.deinit();

        const source_offset = KV - S;
        for (0..H) |h| {
            const kvh = kv_head_for_head[h];
            for (0..S) |qi| {
                const p = source_offset + qi;
                const lo = if (window == 0) 0 else (p + 1) -| window;
                var weights: [KV]f64 = undefined;
                var max_score: f64 = -std.math.inf(f64);
                for (lo..p + 1) |j| {
                    var dot: f64 = 0;
                    for (0..D) |f| dot += @as(f64, q_vals[(qi * H + h) * D + f]) * @as(f64, k_vals[(j * KVH + kvh) * D + f]);
                    weights[j] = dot * scale_value;
                    max_score = @max(max_score, weights[j]);
                }
                var sum: f64 = 0;
                for (lo..p + 1) |j| {
                    weights[j] = @exp(weights[j] - max_score);
                    sum += weights[j];
                }
                for (0..D) |f| {
                    var acc: f64 = 0;
                    for (lo..p + 1) |j| acc += (weights[j] / sum) * @as(f64, v_vals[(j * KVH + kvh) * D + f]);
                    const g = got.dataConst()[(qi * H + h) * D + f];
                    const e: f32 = @floatCast(acc);
                    const tol = @max(1e-5 * @max(@abs(e), @abs(g)), 2e-6);
                    try std.testing.expect(@abs(e - g) <= tol);
                }
            }
        }
    }
}

test "grouped bidirectional attention matches a naive full-range reference" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Shapes chosen to route through every forward kernel: long q_seq with
    // adjacent-pair GQA = tiled pair path; long q_seq with a non-adjacent map
    // = tiled general path; short q_seq = the per-query kernels; the f16 KV
    // entry covers the widening lanes. Every query must attend ALL keys —
    // including keys at positions later than its own (the causal kernels
    // would mask those), which is what the q[0]-sees-k[last] checks verify.
    const Case = struct { s: usize, kv: usize, h: usize, kvh: usize, d: usize, pair: bool };
    const cases = [_]Case{
        .{ .s = 67, .kv = 80, .h = 4, .kvh = 2, .d = 16, .pair = true },
        .{ .s = 67, .kv = 80, .h = 4, .kvh = 2, .d = 16, .pair = false },
        .{ .s = 5, .kv = 9, .h = 4, .kvh = 2, .d = 8, .pair = true },
        .{ .s = 5, .kv = 9, .h = 4, .kvh = 2, .d = 8, .pair = false },
        // d > attention_tile_max_d: stays on the per-query kernels even at
        // long q_seq (the gemma4 global-layer head width regime).
        .{ .s = 49, .kv = 60, .h = 2, .kvh = 1, .d = 288, .pair = false },
    };
    const scale_value: f32 = 0.25;

    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();

    for (cases) |case| {
        const q_len = case.s * case.h * case.d;
        const kv_len = case.kv * case.kvh * case.d;
        const q_vals = try allocator.alloc(f32, q_len);
        defer allocator.free(q_vals);
        const k_vals = try allocator.alloc(f32, kv_len);
        defer allocator.free(k_vals);
        const v_vals = try allocator.alloc(f32, kv_len);
        defer allocator.free(v_vals);
        for (q_vals) |*x| x.* = random.floatNorm(f32);
        for (k_vals) |*x| x.* = random.floatNorm(f32);
        for (v_vals) |*x| x.* = random.floatNorm(f32);

        var kv_head_for_head: [4]usize = undefined;
        for (0..case.h) |h| {
            kv_head_for_head[h] = if (case.pair) h / (case.h / case.kvh) else h % case.kvh;
        }
        const head_map = kv_head_for_head[0..case.h];

        var q = try ctx.fromSliceRank(3, .{ case.s, case.h, case.d }, q_vals);
        defer q.deinit();
        var k = try ctx.fromSliceRank(3, .{ case.kv, case.kvh, case.d }, k_vals);
        defer k.deinit();
        var v = try ctx.fromSliceRank(3, .{ case.kv, case.kvh, case.d }, v_vals);
        defer v.deinit();

        var got = try ctx.groupedBidirectionalAttention(&q, &k, &v, head_map, scale_value);
        defer got.deinit();

        var k16 = try ctx.castTyped(.f32, .f16, &k);
        defer k16.deinit();
        var v16 = try ctx.castTyped(.f32, .f16, &v);
        defer v16.deinit();
        var got16 = try ctx.groupedBidirectionalAttentionF16Kv(&q, &k16, &v16, head_map, scale_value);
        defer got16.deinit();

        for (0..case.h) |h| {
            const kvh = head_map[h];
            for (0..case.s) |qi| {
                var weights: [80]f64 = undefined;
                var max_score: f64 = -std.math.inf(f64);
                for (0..case.kv) |j| {
                    var dot: f64 = 0;
                    for (0..case.d) |f| dot += @as(f64, q_vals[(qi * case.h + h) * case.d + f]) * @as(f64, k_vals[(j * case.kvh + kvh) * case.d + f]);
                    weights[j] = dot * scale_value;
                    max_score = @max(max_score, weights[j]);
                }
                var sum: f64 = 0;
                for (0..case.kv) |j| {
                    weights[j] = @exp(weights[j] - max_score);
                    sum += weights[j];
                }
                for (0..case.d) |f| {
                    var acc: f64 = 0;
                    for (0..case.kv) |j| acc += (weights[j] / sum) * @as(f64, v_vals[(j * case.kvh + kvh) * case.d + f]);
                    // Tolerance covers f32-vs-f64 accumulation order at the
                    // widest case (d=288, 60 keys); a mask bug is O(0.1).
                    const e: f32 = @floatCast(acc);
                    const g = got.dataConst()[(qi * case.h + h) * case.d + f];
                    const tol = @max(5e-5 * @max(@abs(e), @abs(g)), 5e-6);
                    try std.testing.expect(@abs(e - g) <= tol);
                    // f16 K/V: logit noise (~1e-3) shifts softmax weights, so
                    // the f16 lane check is for mask correctness (a causal
                    // leak is an O(1) error), not numeric precision.
                    const g16 = got16.dataConst()[(qi * case.h + h) * case.d + f];
                    const tol16 = @max(2e-2 * @max(@abs(e), @abs(g16)), 1e-2);
                    try std.testing.expect(@abs(e - g16) <= tol16);
                }
            }
        }
    }
}

test "exec attention stats capture is output-neutral and feeds the backward stats route" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0xa77e);
    const random = prng.random();
    const scale: f32 = 0.125;

    // {q_seq, kv_seq, heads, kv_heads, d}: query-tiled (q >= 48), per-query
    // pair, and per-query general kernels.
    const shapes = [_][5]usize{ .{ 64, 64, 4, 2, 16 }, .{ 8, 8, 4, 2, 16 }, .{ 8, 8, 3, 3, 16 } };
    for (shapes) |shape| {
        const q_seq = shape[0];
        const kv_seq = shape[1];
        const heads = shape[2];
        const kv_heads = shape[3];
        const d = shape[4];
        var map_storage: [8]usize = undefined;
        const kv_head_for_head = map_storage[0..heads];
        for (kv_head_for_head, 0..) |*m, i| m.* = if (heads == 2 * kv_heads) i / 2 else i;

        const q_data = try allocator.alloc(f32, q_seq * heads * d);
        defer allocator.free(q_data);
        const k_data = try allocator.alloc(f32, kv_seq * kv_heads * d);
        defer allocator.free(k_data);
        const v_data = try allocator.alloc(f32, kv_seq * kv_heads * d);
        defer allocator.free(v_data);
        for (q_data) |*value| value.* = random.floatNorm(f32);
        for (k_data) |*value| value.* = random.floatNorm(f32);
        for (v_data) |*value| value.* = random.floatNorm(f32);

        var q = try ctx.fromSliceRank(3, .{ q_seq, heads, d }, q_data);
        defer q.deinit();
        var k = try ctx.fromSliceRank(3, .{ kv_seq, kv_heads, d }, k_data);
        defer k.deinit();
        var v = try ctx.fromSliceRank(3, .{ kv_seq, kv_heads, d }, v_data);
        defer v.deinit();

        var out_plain = try ctx.groupedCausalAttention(&q, &k, &v, kv_head_for_head, scale);
        defer out_plain.deinit();
        const stats = try allocator.alloc(f32, heads * q_seq * 2);
        defer allocator.free(stats);
        var out_stats = try ctx.groupedCausalAttentionStatsOut(&q, &k, &v, kv_head_for_head, scale, 0, true, stats);
        defer out_stats.deinit();
        // Capture is write-only: the forward output must be BITWISE identical.
        try std.testing.expectEqualSlices(f32, out_plain.dataConst(), out_stats.dataConst());

        // f64 reference sanity of the captured normalizers.
        for (0..heads) |head_i| {
            const kv_head_i = kv_head_for_head[head_i];
            for (0..q_seq) |query_i| {
                const active = kv_seq - q_seq + query_i + 1;
                var max64: f64 = -std.math.inf(f64);
                for (0..active) |source_i| {
                    var dot: f64 = 0;
                    for (0..d) |f| {
                        dot += @as(f64, q_data[query_i * heads * d + head_i * d + f]) *
                            @as(f64, k_data[source_i * kv_heads * d + kv_head_i * d + f]);
                    }
                    max64 = @max(max64, dot * scale);
                }
                const stat_max = stats[(head_i * q_seq + query_i) * 2];
                const stat_sum = stats[(head_i * q_seq + query_i) * 2 + 1];
                try std.testing.expect(@abs(@as(f64, stat_max) - max64) <= 1e-3 + 1e-3 * @abs(max64));
                var sum64: f64 = 0;
                for (0..active) |source_i| {
                    var dot: f64 = 0;
                    for (0..d) |f| {
                        dot += @as(f64, q_data[query_i * heads * d + head_i * d + f]) *
                            @as(f64, k_data[source_i * kv_heads * d + kv_head_i * d + f]);
                    }
                    sum64 += @exp(dot * scale - @as(f64, stat_max));
                }
                try std.testing.expect(@abs(@as(f64, stat_sum) - sum64) <= 2e-3 * sum64);
            }
        }
    }

    // Backward route parity on the GEMM route (work >= threshold): the
    // stats route rebuilds the FORWARD's probabilities where the recompute
    // route re-derives them from the GEMM scores; gradients agree to f32
    // roundoff, not bitwise.
    const q_seq = 64;
    const kv_seq = 64;
    const heads = 8;
    const kv_heads = 4;
    const d = 32;
    const kv_head_for_head = [_]usize{ 0, 0, 1, 1, 2, 2, 3, 3 };
    const q_data = try allocator.alloc(f32, q_seq * heads * d);
    defer allocator.free(q_data);
    const k_data = try allocator.alloc(f32, kv_seq * kv_heads * d);
    defer allocator.free(k_data);
    const v_data = try allocator.alloc(f32, kv_seq * kv_heads * d);
    defer allocator.free(v_data);
    const gy_data = try allocator.alloc(f32, q_seq * heads * d);
    defer allocator.free(gy_data);
    for (q_data) |*value| value.* = random.floatNorm(f32);
    for (k_data) |*value| value.* = random.floatNorm(f32);
    for (v_data) |*value| value.* = random.floatNorm(f32);
    for (gy_data) |*value| value.* = random.floatNorm(f32);
    var q = try ctx.fromSliceRank(3, .{ q_seq, heads, d }, q_data);
    defer q.deinit();
    var k = try ctx.fromSliceRank(3, .{ kv_seq, kv_heads, d }, k_data);
    defer k.deinit();
    var v = try ctx.fromSliceRank(3, .{ kv_seq, kv_heads, d }, v_data);
    defer v.deinit();
    var gy = try ctx.fromSliceRank(2, .{ q_seq, heads * d }, gy_data);
    defer gy.deinit();

    for ([_][2]usize{ .{ 0, 1 }, .{ 16, 1 }, .{ 0, 0 } }) |variant| {
        const window = variant[0];
        const causal = variant[1] == 1;
        const stats = try allocator.alloc(f32, heads * q_seq * 2);
        defer allocator.free(stats);
        var out = try ctx.groupedCausalAttentionStatsOut(&q, &k, &v, &kv_head_for_head, 0.125, window, causal, stats);
        defer out.deinit();

        var ref = try ctx.groupedCausalAttentionBackward(&q, &k, &v, &gy, &kv_head_for_head, 0.125, window, causal, null, null, true, true, true);
        defer ref.deinit();
        // Stats + output: the autograd-record route (one-pass softmax rebuild
        // AND the gy.O row dot).
        var got = try ctx.groupedCausalAttentionBackward(&q, &k, &v, &gy, &kv_head_for_head, 0.125, window, causal, stats, &out, true, true, true);
        defer got.deinit();
        // Stats without output: the in-panel row dot fallback.
        var got_no_out = try ctx.groupedCausalAttentionBackward(&q, &k, &v, &gy, &kv_head_for_head, 0.125, window, causal, stats, null, true, true, true);
        defer got_no_out.deinit();
        for ([_][2][]const f32{
            .{ ref.q.?.dataConst(), got.q.?.dataConst() },
            .{ ref.k.?.dataConst(), got.k.?.dataConst() },
            .{ ref.v.?.dataConst(), got.v.?.dataConst() },
            .{ ref.q.?.dataConst(), got_no_out.q.?.dataConst() },
            .{ ref.k.?.dataConst(), got_no_out.k.?.dataConst() },
            .{ ref.v.?.dataConst(), got_no_out.v.?.dataConst() },
        }) |pair| {
            for (pair[0], pair[1]) |want, gotv| {
                try std.testing.expect(@abs(gotv - want) <= 2e-5 + 2e-4 * @abs(want));
            }
        }

        // Ground truth: an f64 naive backward over the same masks — pins the
        // tiled kernel's values themselves, not just cross-arm agreement.
        const dq64 = try allocator.alloc(f64, q_seq * heads * d);
        defer allocator.free(dq64);
        const dk64 = try allocator.alloc(f64, kv_seq * kv_heads * d);
        defer allocator.free(dk64);
        const dv64 = try allocator.alloc(f64, kv_seq * kv_heads * d);
        defer allocator.free(dv64);
        @memset(dq64, 0);
        @memset(dk64, 0);
        @memset(dv64, 0);
        const scores64 = try allocator.alloc(f64, kv_seq);
        defer allocator.free(scores64);
        const dp64 = try allocator.alloc(f64, kv_seq);
        defer allocator.free(dp64);
        for (0..heads) |head_i| {
            const kv_head_i = kv_head_for_head[head_i];
            for (0..q_seq) |query_i| {
                const active = if (causal) kv_seq - q_seq + query_i + 1 else kv_seq;
                const lo = if (!causal or window == 0) 0 else active -| window;
                var max64: f64 = -std.math.inf(f64);
                for (lo..active) |source_i| {
                    var dot: f64 = 0;
                    for (0..d) |f| {
                        dot += @as(f64, q_data[query_i * heads * d + head_i * d + f]) *
                            @as(f64, k_data[source_i * kv_heads * d + kv_head_i * d + f]);
                    }
                    scores64[source_i] = dot * 0.125;
                    max64 = @max(max64, scores64[source_i]);
                }
                var sum64: f64 = 0;
                for (lo..active) |source_i| {
                    scores64[source_i] = @exp(scores64[source_i] - max64);
                    sum64 += scores64[source_i];
                }
                var row_dot64: f64 = 0;
                for (lo..active) |source_i| {
                    scores64[source_i] /= sum64;
                    var dot: f64 = 0;
                    for (0..d) |f| {
                        dot += @as(f64, gy_data[query_i * heads * d + head_i * d + f]) *
                            @as(f64, v_data[source_i * kv_heads * d + kv_head_i * d + f]);
                    }
                    dp64[source_i] = dot;
                    row_dot64 += scores64[source_i] * dot;
                }
                for (lo..active) |source_i| {
                    const ds = 0.125 * scores64[source_i] * (dp64[source_i] - row_dot64);
                    for (0..d) |f| {
                        dq64[query_i * heads * d + head_i * d + f] +=
                            ds * @as(f64, k_data[source_i * kv_heads * d + kv_head_i * d + f]);
                        dk64[source_i * kv_heads * d + kv_head_i * d + f] +=
                            ds * @as(f64, q_data[query_i * heads * d + head_i * d + f]);
                        dv64[source_i * kv_heads * d + kv_head_i * d + f] +=
                            @as(f64, scores64[source_i]) * @as(f64, gy_data[query_i * heads * d + head_i * d + f]);
                    }
                }
            }
        }
        for ([_]struct { want: []const f64, got: []const f32 }{
            .{ .want = dq64, .got = ref.q.?.dataConst() },
            .{ .want = dk64, .got = ref.k.?.dataConst() },
            .{ .want = dv64, .got = ref.v.?.dataConst() },
            .{ .want = dq64, .got = got.q.?.dataConst() },
            .{ .want = dk64, .got = got.k.?.dataConst() },
            .{ .want = dv64, .got = got.v.?.dataConst() },
        }) |pair| {
            for (pair.want, pair.got) |want, gotv| {
                try std.testing.expect(@abs(@as(f64, gotv) - want) <= 5e-5 + 5e-4 * @abs(want));
            }
        }
    }
}
