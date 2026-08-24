//! Function tests for the SubQ attention operator: the batched/gauged exact
//! path must reproduce a straightforward f64 dense-attention oracle when
//! every cluster opens (tau = 0 and the calibration path), and the pure-tail
//! extreme (tau = 1) must match an f64 reimplementation of the zeroth-order
//! moment accounting. Guards the online-softmax gauge, vexpf scoring, and
//! packed-read batching against numeric or bookkeeping regressions.

const std = @import("std");
const fucina = @import("fucina");
const subq = @import("subq.zig");

const ExecContext = fucina.ExecContext;

const num_layers = 1;
const q_heads = 2;
const kv_heads = 1;
const d = 32;
const cached_len = 128;

fn testConfig() subq.Config {
    return .{
        // The f16-oracle tests pin the f16 packed path; the q8_0/ptqtp_k2
        // tests override the format themselves.
        .packed_format = .f16,
        .cluster_size = 8,
        .rebuild_interval = 32,
        .sink = 2,
        .recent = 4,
        .rank = 2,
        .power_iterations = 4,
    };
}

const Fixture = struct {
    k: [cached_len * kv_heads * d]f16,
    v: [cached_len * kv_heads * d]f16,
    q: [q_heads * d]f32,

    fn init(query_scale: f32) Fixture {
        var self: Fixture = undefined;
        var s: u64 = 0x9e3779b97f4a7c15;
        for (&self.k) |*x| x.* = @floatCast(lcgUniform(&s));
        for (&self.v) |*x| x.* = @floatCast(lcgUniform(&s));
        for (&self.q) |*x| x.* = query_scale * lcgUniform(&s);
        return self;
    }
};

fn lcgUniform(s: *u64) f32 {
    s.* = s.* *% 6364136223846793005 +% 1442695040888963407;
    const bits: u32 = @truncate(s.* >> 33);
    return @as(f32, @floatFromInt(bits)) / @as(f32, @floatFromInt(std.math.maxInt(u32))) * 2.0 - 1.0;
}

/// Full dense attention in f64 over the same f16 values the kernel reads.
fn denseOracle(fx: *const Fixture, head_i: usize, out: []f64) void {
    const beta = 1.0 / @sqrt(@as(f64, d));
    const query = fx.q[head_i * d ..][0..d];
    var z: f64 = 0;
    @memset(out, 0);
    for (0..cached_len) |pos| {
        const base = pos * kv_heads * d;
        var dot: f64 = 0;
        for (0..d) |i| dot += @as(f64, query[i]) * @as(f64, @floatCast(fx.k[base + i]));
        const w = @exp(beta * dot);
        z += w;
        for (0..d) |i| out[i] += w * @as(f64, @floatCast(fx.v[base + i]));
    }
    for (out) |*o| o.* /= z;
}

fn relErr(got: []const f32, want: []const f64) f64 {
    var err2: f64 = 0;
    var norm2: f64 = 0;
    for (got, want) |g, w| {
        const delta = @as(f64, g) - w;
        err2 += delta * delta;
        norm2 += w * w;
    }
    return @sqrt(err2) / @max(@sqrt(norm2), 1e-12);
}

fn attendOnce(state: *subq.State, ctx: *ExecContext, fx: *const Fixture, out: []f32) !void {
    try state.attend(ctx, 0, &fx.q, &fx.k, &fx.v, cached_len, out);
}

test "subq attend with tau 0 matches the f64 dense oracle" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var state = try subq.State.init(allocator, num_layers, q_heads, kv_heads, d, testConfig());
    defer state.deinit();
    @memset(state.taus, 0);

    var fx = Fixture.init(1.0);
    var out: [q_heads * d]f32 = undefined;
    try attendOnce(&state, &ctx, &fx, &out);
    try std.testing.expect(state.plans[0].clusters > 0);

    var want: [d]f64 = undefined;
    for (0..q_heads) |head_i| {
        denseOracle(&fx, head_i, &want);
        try std.testing.expect(relErr(out[head_i * d ..][0..d], &want) < 1e-3);
    }
}

test "subq attend stays accurate and finite at large score magnitudes" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var state = try subq.State.init(allocator, num_layers, q_heads, kv_heads, d, testConfig());
    defer state.deinit();
    @memset(state.taus, 0);

    // Scores reach O(100): far past f32 exp range without the gauge.
    var fx = Fixture.init(25.0);
    var out: [q_heads * d]f32 = undefined;
    try attendOnce(&state, &ctx, &fx, &out);

    var want: [d]f64 = undefined;
    for (0..q_heads) |head_i| {
        denseOracle(&fx, head_i, &want);
        for (out[head_i * d ..][0..d]) |x| try std.testing.expect(std.math.isFinite(x));
        try std.testing.expect(relErr(out[head_i * d ..][0..d], &want) < 1e-3);
    }
}

test "subq attend with tau 1 matches an f64 moment-tail reference" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var state = try subq.State.init(allocator, num_layers, q_heads, kv_heads, d, testConfig());
    defer state.deinit();
    @memset(state.taus, 1.0);

    var fx = Fixture.init(1.0);
    var out: [q_heads * d]f32 = undefined;
    try attendOnce(&state, &ctx, &fx, &out);
    const plan = &state.plans[0];
    try std.testing.expect(plan.clusters > 0);

    const beta = 1.0 / @sqrt(@as(f64, d));
    const cfg = testConfig();
    for (0..q_heads) |head_i| {
        const query = fx.q[head_i * d ..][0..d];
        var z: f64 = 0;
        var want: [d]f64 = @splat(0);
        // Exact rows: sinks plus the unsealed suffix.
        for (0..cached_len) |pos| {
            if (pos >= cfg.sink and pos < plan.seal_end) continue;
            const base = pos * kv_heads * d;
            var dot: f64 = 0;
            for (0..d) |i| dot += @as(f64, query[i]) * @as(f64, @floatCast(fx.k[base + i]));
            const w = @exp(beta * dot);
            z += w;
            for (0..d) |i| want[i] += w * @as(f64, @floatCast(fx.v[base + i]));
        }
        // Zeroth-order tail: count * exp(centroid score) at the value mean.
        for (0..plan.clusters) |c| {
            var dot: f64 = 0;
            for (0..d) |i| dot += @as(f64, query[i]) * @as(f64, plan.centroid[c * d + i]);
            const w = @as(f64, plan.counts[c]) * @exp(beta * dot);
            z += w;
            for (0..d) |i| want[i] += w * @as(f64, plan.vmean[c * d + i]);
        }
        for (&want) |*o| o.* /= z;
        try std.testing.expect(relErr(out[head_i * d ..][0..d], &want) < 1e-3);
    }
}

test "subq calibration pass returns the exact output and freezes taus" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var state = try subq.State.init(allocator, num_layers, q_heads, kv_heads, d, testConfig());
    defer state.deinit();

    var fx = Fixture.init(1.0);
    var out: [q_heads * d]f32 = undefined;
    try state.startCalibration(0.025);
    try attendOnce(&state, &ctx, &fx, &out);
    var want: [d]f64 = undefined;
    for (0..q_heads) |head_i| {
        denseOracle(&fx, head_i, &want);
        try std.testing.expect(relErr(out[head_i * d ..][0..d], &want) < 1e-3);
    }
    state.finishCalibration();
    try std.testing.expect(!state.calibrating);
    for (state.taus) |tau| {
        try std.testing.expect(tau >= subq.calib_grid[subq.calib_grid.len - 1]);
        try std.testing.expect(tau <= subq.calib_grid[0]);
    }
}

test "subq q8_0 packed format matches a dequantized-oracle at tau 0" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var config = testConfig();
    config.packed_format = .q8_0;
    var state = try subq.State.init(allocator, num_layers, q_heads, kv_heads, d, config);
    defer state.deinit();
    @memset(state.taus, 0);

    var fx = Fixture.init(1.0);
    var out: [q_heads * d]f32 = undefined;
    try attendOnce(&state, &ctx, &fx, &out);
    const plan = &state.plans[0];
    try std.testing.expect(plan.clusters > 0);
    try std.testing.expect(plan.packed_k_q8.len > 0);

    // Oracle in f64 over the values the q8 kernel actually reads: exact f16
    // rows for sinks and the unsealed suffix, dequantized q8 rows (with the
    // query itself q8-quantized, matching the integer score path) for the
    // sealed clusters. Sealed rows live at packed position p for original
    // position sink + p (identity order fixture-independent check via the
    // builder's `order` is unavailable here, so compare against the kernel's
    // own packed data instead of the raw cache).
    const beta = 1.0 / @sqrt(@as(f64, d));
    const bpr = d / 32;
    for (0..q_heads) |head_i| {
        const query = fx.q[head_i * d ..][0..d];
        var q_q8: [16]fucina.quant.BlockQ8_0 = undefined;
        try @import("fucina").internal.backend_mod.quantized_matmul.q8k.quantizeRowQ8_0Into(q_q8[0..bpr], query);
        var qdeq: [d]f32 = undefined;
        try @import("fucina").internal.backend_mod.quantized_matmul.q8k.dequantizeRowQ8_0Into(&qdeq, q_q8[0..bpr]);
        var z: f64 = 0;
        var want: [d]f64 = @splat(0);
        // Exact rows: outside [sink, seal_end) read from the f16 cache with
        // the f32 query.
        const cfg = testConfig();
        for (0..cached_len) |pos| {
            if (pos >= cfg.sink and pos < plan.seal_end) continue;
            const base = pos * kv_heads * d;
            var dot: f64 = 0;
            for (0..d) |i| dot += @as(f64, query[i]) * @as(f64, @floatCast(fx.k[base + i]));
            const w = @exp(beta * dot);
            z += w;
            for (0..d) |i| want[i] += w * @as(f64, @floatCast(fx.v[base + i]));
        }
        // Sealed rows: from the kernel's own packed q8 data, scored with the
        // quantized query (the integer path's semantics).
        const n_packed = plan.offsets[plan.clusters];
        var krow: [d]f32 = undefined;
        var vrow: [d]f32 = undefined;
        for (0..n_packed) |row| {
            try @import("fucina").internal.backend_mod.quantized_matmul.q8k.dequantizeRowQ8_0Into(&krow, plan.packed_k_q8[row * bpr ..][0..bpr]);
            try @import("fucina").internal.backend_mod.quantized_matmul.q8k.dequantizeRowQ8_0Into(&vrow, plan.packed_v_q8[row * bpr ..][0..bpr]);
            var dot: f64 = 0;
            for (0..d) |i| dot += @as(f64, qdeq[i]) * @as(f64, krow[i]);
            const w = @exp(beta * dot);
            z += w;
            for (0..d) |i| want[i] += w * @as(f64, vrow[i]);
        }
        for (&want) |*o| o.* /= z;
        try std.testing.expect(relErr(out[head_i * d ..][0..d], &want) < 1e-3);
    }
}

test "subq incremental frozen-block maintenance matches the oracle across growth" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var config = testConfig();
    config.rebuild_interval = 16;
    config.block_size = 32;
    var state = try subq.State.init(allocator, num_layers, q_heads, kv_heads, d, config);
    defer state.deinit();
    @memset(state.taus, 0);

    var fx = Fixture.init(1.0);
    var out: [q_heads * d]f32 = undefined;
    // Simulated decode: repeated attends at growing lengths force several
    // region rebuilds and at least one block freeze.
    var len: usize = 24;
    while (len <= cached_len) : (len += 8) {
        try state.attend(&ctx, 0, &fx.q, &fx.k, &fx.v, len, &out);
    }
    const plan = &state.plans[0];
    try std.testing.expect(plan.frozen_rows > 0);
    try std.testing.expect(plan.clusters > plan.frozen_clusters or plan.seal_end == plan.frozen_end);

    // Final attend at full length: exact (tau 0) output must match the
    // dense f64 oracle despite the blockwise-built plan.
    try state.attend(&ctx, 0, &fx.q, &fx.k, &fx.v, cached_len, &out);
    var want: [d]f64 = undefined;
    for (0..q_heads) |head_i| {
        denseOracle(&fx, head_i, &want);
        try std.testing.expect(relErr(out[head_i * d ..][0..d], &want) < 1e-3);
    }
}

test "subq hierarchical frontier matches the dense oracle at tau 0" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var config = testConfig();
    config.hierarchical = true;
    var state = try subq.State.init(allocator, num_layers, q_heads, kv_heads, d, config);
    defer state.deinit();
    @memset(state.taus, 0);

    var fx = Fixture.init(1.0);
    var out: [q_heads * d]f32 = undefined;
    try attendOnce(&state, &ctx, &fx, &out);
    const plan = &state.plans[0];
    try std.testing.expect(plan.clusters > 1);
    try std.testing.expect(plan.node_count == plan.clusters - 1);

    var want: [d]f64 = undefined;
    for (0..q_heads) |head_i| {
        denseOracle(&fx, head_i, &want);
        try std.testing.expect(relErr(out[head_i * d ..][0..d], &want) < 1e-3);
    }
}

test "subq hierarchical frontier matches a root-tail reference at tau 1" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var config = testConfig();
    config.hierarchical = true;
    var state = try subq.State.init(allocator, num_layers, q_heads, kv_heads, d, config);
    defer state.deinit();
    @memset(state.taus, 1.0);

    var fx = Fixture.init(1.0);
    var out: [q_heads * d]f32 = undefined;
    try attendOnce(&state, &ctx, &fx, &out);
    const plan = &state.plans[0];
    try std.testing.expect(plan.node_count > 0);

    const beta = 1.0 / @sqrt(@as(f64, d));
    const cfg = testConfig();
    for (0..q_heads) |head_i| {
        const query = fx.q[head_i * d ..][0..d];
        var z: f64 = 0;
        var want: [d]f64 = @splat(0);
        for (0..cached_len) |pos| {
            if (pos >= cfg.sink and pos < plan.seal_end) continue;
            const base = pos * kv_heads * d;
            var dot: f64 = 0;
            for (0..d) |i| dot += @as(f64, query[i]) * @as(f64, @floatCast(fx.k[base + i]));
            const w = @exp(beta * dot);
            z += w;
            for (0..d) |i| want[i] += w * @as(f64, @floatCast(fx.v[base + i]));
        }
        // Frontier = root: one aggregated zeroth-order term.
        var dot: f64 = 0;
        for (0..d) |i| dot += @as(f64, query[i]) * @as(f64, plan.node_centroid[i]);
        const w = @as(f64, plan.node_counts[0]) * @exp(beta * dot);
        z += w;
        for (0..d) |i| want[i] += w * @as(f64, plan.node_vmean[i]);
        for (&want) |*o| o.* /= z;
        try std.testing.expect(relErr(out[head_i * d ..][0..d], &want) < 2e-3);
    }
}

test "subq hierarchical calibration returns exact output and freezes taus" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var config = testConfig();
    config.hierarchical = true;
    var state = try subq.State.init(allocator, num_layers, q_heads, kv_heads, d, config);
    defer state.deinit();

    var fx = Fixture.init(1.0);
    var out: [q_heads * d]f32 = undefined;
    try state.startCalibration(0.025);
    try attendOnce(&state, &ctx, &fx, &out);
    var want: [d]f64 = undefined;
    for (0..q_heads) |head_i| {
        denseOracle(&fx, head_i, &want);
        try std.testing.expect(relErr(out[head_i * d ..][0..d], &want) < 1e-3);
    }
    state.finishCalibration();
    for (state.taus) |tau| {
        try std.testing.expect(tau >= subq.calib_grid[subq.calib_grid.len - 1]);
        try std.testing.expect(tau <= subq.calib_grid[0]);
    }
}
