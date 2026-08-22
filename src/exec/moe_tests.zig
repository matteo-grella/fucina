//! MoE kernels: route plans, token-major scatter parity, col-width and
//! phase-chunk tiling, and phased-chain determinism. Force-imported by
//! `exec.zig`.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const exec = @import("../exec.zig");
const exec_elementwise = @import("elementwise.zig");
const exec_row_ops = @import("row_ops.zig");
const exec_moe_chain = @import("moe_chain.zig");
const dtype_mod = @import("../dtype.zig");
const fpenv = @import("../fpenv.zig");
const parallel = @import("../parallel.zig");
const rng = @import("../rng.zig");
const tensor = @import("../tensor.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;
const ExecContext = exec.ExecContext;
const LayoutClass = exec.LayoutClass;
const CrossEntropyOptions = exec.CrossEntropyOptions;
const Reduction = exec.Reduction;

const util = @import("test_util.zig");
const buildTestMoeRhsQ5K = util.buildTestMoeRhsQ5K;

test "moe route plan groups pairs by expert with a consistent inverse" {
    const allocator = std.testing.allocator;
    const n_expert: usize = 5;
    // 13 tokens x top_k 3, including an expert with zero routed pairs (4).
    const selected = [_]usize{
        0, 2, 1, 3, 3, 0, 1, 1, 2, 0, 0, 3, 2,
        1, 0, 2, 3, 1, 0, 2, 2, 1, 0, 3, 3, 1,
        0, 2, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0,
    };

    const result = try exec_moe_chain.buildMoeRoutePlan(allocator, &selected, n_expert, false, null);
    var route = result.plan;
    defer route.deinit();

    try std.testing.expectEqual(selected.len, route.pairCount());
    try std.testing.expectEqual(n_expert, route.expertCount());
    try std.testing.expectEqual(@as(usize, 4), route.active_experts);
    try std.testing.expectEqual(@as(usize, 0), route.count[4]);

    var total: usize = 0;
    var max_m: usize = 0;
    for (0..n_expert) |e| {
        try std.testing.expectEqual(route.offset[e], total);
        total += route.count[e];
        max_m = @max(max_m, route.count[e]);
    }
    try std.testing.expectEqual(selected.len, total);
    try std.testing.expectEqual(max_m, route.max_expert_m);

    // order[offset[e]..] lists exactly expert e's pairs in pair order, and
    // inv is its inverse permutation.
    for (0..n_expert) |e| {
        for (route.order[route.offset[e]..][0..route.count[e]]) |pair| {
            try std.testing.expectEqual(e, selected[pair]);
        }
    }
    for (0..selected.len) |p| try std.testing.expectEqual(p, route.order[route.inv[p]]);

    try std.testing.expectError(error.IndexOutOfBounds, exec_moe_chain.buildMoeRoutePlan(allocator, &selected, 3, false, null));
}

test "moe token-major scatter: range split is bit-identical to serial" {
    const allocator = std.testing.allocator;
    const seq: usize = 13;
    const top_k: usize = 3;
    const n_expert: usize = 5;
    const hidden: usize = 8;
    const n_pairs = seq * top_k;

    var prng = std.Random.DefaultPrng.init(97);
    const random = prng.random();
    const selected = try allocator.alloc(usize, n_pairs);
    defer allocator.free(selected);
    const weights = try allocator.alloc(f32, n_pairs);
    defer allocator.free(weights);
    for (selected, weights) |*s, *w| {
        s.* = random.uintLessThan(usize, n_expert);
        w.* = 0.1 + random.float(f32);
    }

    const result = try exec_moe_chain.buildMoeRoutePlan(allocator, selected, n_expert, false, null);
    var route = result.plan;
    defer route.deinit();

    // Pair-major rows, then placed at their grouped position inv[p] — the
    // layout the expert GEMMs leave behind.
    const pair_rows = try allocator.alloc(f32, n_pairs * hidden);
    defer allocator.free(pair_rows);
    for (pair_rows) |*v| v.* = random.floatNorm(f32);
    const down_rows = try allocator.alloc(f32, n_pairs * hidden);
    defer allocator.free(down_rows);
    for (0..n_pairs) |p| {
        @memcpy(down_rows[route.inv[p] * hidden ..][0..hidden], pair_rows[p * hidden ..][0..hidden]);
    }

    const out_full = try allocator.alloc(f32, seq * hidden);
    defer allocator.free(out_full);
    exec_moe_chain.scatterTokenMajor(out_full, down_rows, weights, route.inv, hidden, top_k, 0, seq);

    // The parallel path's decomposition: disjoint token ranges, any split.
    const out_split = try allocator.alloc(f32, seq * hidden);
    defer allocator.free(out_split);
    exec_moe_chain.scatterTokenMajor(out_split, down_rows, weights, route.inv, hidden, top_k, 0, 4);
    exec_moe_chain.scatterTokenMajor(out_split, down_rows, weights, route.inv, hidden, top_k, 4, 5);
    exec_moe_chain.scatterTokenMajor(out_split, down_rows, weights, route.inv, hidden, top_k, 5, seq);
    try std.testing.expectEqualSlices(f32, out_full, out_split);

    // Exact reference in the same per-row accumulation order, straight from
    // the pair-major rows: validates the inv[] gather indexing.
    const out_ref = try allocator.alloc(f32, seq * hidden);
    defer allocator.free(out_ref);
    for (0..seq) |t| {
        const dst = out_ref[t * hidden ..][0..hidden];
        for (0..top_k) |k| {
            const p = t * top_k + k;
            const w = weights[p];
            const src = pair_rows[p * hidden ..][0..hidden];
            if (k == 0) {
                for (dst, src) |*o, s| o.* = w * s;
            } else {
                for (dst, src) |*o, s| o.* += w * s;
            }
        }
    }
    try std.testing.expectEqualSlices(f32, out_ref, out_full);
}

test "moe small-m col width: 256 strictly below the worker task budget, 0 at and above" {
    const chunk = exec_moe_chain.moe_phase_small_m_col_chunk;
    const workers: usize = 8;
    // workers * moe_phase_small_m_task_budget_mul = 128 active experts.
    try std.testing.expectEqual(@as(usize, 128), workers * exec_moe_chain.moe_phase_small_m_task_budget_mul);
    try std.testing.expectEqual(@as(usize, 0), exec_moe_chain.moeSmallMColWidth(0, workers));
    try std.testing.expectEqual(@as(usize, chunk), exec_moe_chain.moeSmallMColWidth(1, workers));
    try std.testing.expectEqual(@as(usize, chunk), exec_moe_chain.moeSmallMColWidth(127, workers));
    try std.testing.expectEqual(@as(usize, 0), exec_moe_chain.moeSmallMColWidth(128, workers));
    try std.testing.expectEqual(@as(usize, 0), exec_moe_chain.moeSmallMColWidth(129, workers));
}

test "moe phase chunks tile [0, out_dim) contiguously with 256-aligned splits" {
    // Count and bounds both derive from moePhaseColWidth, so for every
    // (m, out_dim, small_m_width) combination the chunk sequence must be
    // non-empty, gapless, and — whenever a phase is actually split —
    // superblock-aligned (c0 % 256 == 0 keeps the Q4_K/Q5_K/Q6_K column
    // kernels on whole K-quant blocks).
    const out_dims = [_]usize{ 33, 255, 256, 768, 2048 };
    const ms = [_]usize{ 1, 15, 16, 17 };
    const small_m_widths = [_]usize{ 0, 256 };
    for (out_dims) |out_dim| {
        for (ms) |m| {
            for (small_m_widths) |small_m_width| {
                const width = exec_moe_chain.moePhaseColWidth(m, out_dim, small_m_width);
                const chunks = exec_moe_chain.moePhaseChunkCount(width, out_dim);
                try std.testing.expect(chunks >= 1);
                var expected_c0: usize = 0;
                for (0..chunks) |chunk| {
                    const b = exec_moe_chain.moePhaseChunkBounds(chunk, width, out_dim);
                    try std.testing.expect(b.c0 < b.c1);
                    try std.testing.expectEqual(expected_c0, b.c0);
                    if (chunks > 1) try std.testing.expectEqual(@as(usize, 0), b.c0 % 256);
                    expected_c0 = b.c1;
                }
                try std.testing.expectEqual(out_dim, expected_c0);
            }
        }
    }
}

test "moe batched ffn: phased chain output is deterministic across identical runs" {
    // seq * top_k = 64 pairs meets moe_batch_phase_min_pairs, so every run
    // drives the real gather -> gate/up -> act -> down phase chain (with
    // small-m column chunking: active experts << workers * 16) on the shared
    // work pool. An enqueue-contract violation or overlapping chunk write
    // shows up as a bitwise diff between runs; in Debug the chain's safety
    // panics fire as well.
    const allocator = std.testing.allocator;
    const seq: usize = 32;
    const top_k: usize = 2;
    const n_expert: usize = 8;
    const hidden: usize = 512;
    const out_pe: usize = 512;
    try std.testing.expect(seq * top_k >= exec_moe_chain.moe_batch_phase_min_pairs);

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var gate = try buildTestMoeRhsQ5K(allocator, n_expert * out_pe, hidden, 0);
    defer gate.deinit();
    var up = try buildTestMoeRhsQ5K(allocator, n_expert * out_pe, hidden, 1);
    defer up.deinit();
    var down = try buildTestMoeRhsQ5K(allocator, n_expert * hidden, out_pe, 2);
    defer down.deinit();

    const x_vals = try allocator.alloc(f32, seq * hidden);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
    var x = try ctx.fromSliceRank(2, .{ seq, hidden }, x_vals);
    defer x.deinit();

    // Every expert active with uneven per-expert m; per-pair routing weights.
    var selected: [seq * top_k]usize = undefined;
    var weights: [seq * top_k]f32 = undefined;
    for (&selected, &weights, 0..) |*s, *w, p| {
        s.* = (p * 5) % n_expert;
        w.* = 0.25 + 0.01 * @as(f32, @floatFromInt(p % 13));
    }

    const first = try allocator.alloc(f32, seq * hidden);
    defer allocator.free(first);
    for (0..8) |run| {
        var out = try ctx.moeExpertFfnBatch(&x, &gate, &up, &down, &selected, &weights, top_k, out_pe, .swiglu, null, null);
        defer out.deinit();
        if (run == 0) {
            @memcpy(first, out.dataConst());
        } else {
            try std.testing.expectEqualSlices(f32, first, out.dataConst());
        }
    }
}
