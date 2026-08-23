//! Microbenchmark for the f16 row-block attention primitives: isolates
//! scoreRows4F16 / weightedAccumRows4F16 throughput (GB/s) from the model
//! context, and A/Bs a comptime-d specialization.

const std = @import("std");
const fucina = @import("fucina");
const simd = fucina.internal.backend_mod.vector_impl;

const d = 128;
const rows_n = 64 * 1024; // 16 MB of f16 K rows: far beyond L2

fn scoreRows4F16Comptime(comptime dim: usize, scores: []f32, query: *const [dim]f32, rows: []const f16) void {
    const V = @Vector(8, f32);
    const H = @Vector(8, f16);
    var i: usize = 0;
    while (i + 4 <= scores.len) : (i += 4) {
        const r0 = rows[i * dim ..][0..dim];
        const r1 = rows[(i + 1) * dim ..][0..dim];
        const r2 = rows[(i + 2) * dim ..][0..dim];
        const r3 = rows[(i + 3) * dim ..][0..dim];
        var a0: V = @splat(0);
        var a1: V = @splat(0);
        var a2: V = @splat(0);
        var a3: V = @splat(0);
        var b0: V = @splat(0);
        var b1: V = @splat(0);
        var b2: V = @splat(0);
        var b3: V = @splat(0);
        comptime var j: usize = 0;
        inline while (j + 16 <= dim) : (j += 16) {
            const q0: V = query[j..][0..8].*;
            const q1: V = query[j + 8 ..][0..8].*;
            a0 = @mulAdd(V, q0, @floatCast(@as(H, r0[j..][0..8].*)), a0);
            a1 = @mulAdd(V, q0, @floatCast(@as(H, r1[j..][0..8].*)), a1);
            a2 = @mulAdd(V, q0, @floatCast(@as(H, r2[j..][0..8].*)), a2);
            a3 = @mulAdd(V, q0, @floatCast(@as(H, r3[j..][0..8].*)), a3);
            b0 = @mulAdd(V, q1, @floatCast(@as(H, r0[j + 8 ..][0..8].*)), b0);
            b1 = @mulAdd(V, q1, @floatCast(@as(H, r1[j + 8 ..][0..8].*)), b1);
            b2 = @mulAdd(V, q1, @floatCast(@as(H, r2[j + 8 ..][0..8].*)), b2);
            b3 = @mulAdd(V, q1, @floatCast(@as(H, r3[j + 8 ..][0..8].*)), b3);
        }
        scores[i] = @reduce(.Add, a0 + b0);
        scores[i + 1] = @reduce(.Add, a1 + b1);
        scores[i + 2] = @reduce(.Add, a2 + b2);
        scores[i + 3] = @reduce(.Add, a3 + b3);
    }
    while (i < scores.len) : (i += 1) scores[i] = simd.primitives.dotF32F16(query, rows[i * dim ..][0..dim]);
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const io = init.io;
    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    const stdout = &stdout_w.interface;

    const rows = try allocator.alloc(f16, rows_n * d);
    defer allocator.free(rows);
    var seed: u64 = 12345;
    for (rows) |*x| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        x.* = @floatCast(@as(f32, @floatFromInt(@as(u16, @truncate(seed >> 40)))) / 65536.0 - 0.5);
    }
    var query: [d]f32 = undefined;
    for (&query, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 17)) * 0.1 - 0.8;
    const scores = try allocator.alloc(f32, rows_n);
    defer allocator.free(scores);
    const numer = try allocator.alloc(f32, d);
    defer allocator.free(numer);
    @memset(numer, 0);

    const bytes_per_pass: f64 = @floatFromInt(rows_n * d * 2);
    const reps = 20;

    // Context-shaped variant: the decode loop's exactBatch sequence over
    // scattered 64-row cluster batches (score + max + exp + weighted accum,
    // K and V arrays both touched), batch order randomized.
    const vrows = try allocator.alloc(f16, rows_n * d);
    defer allocator.free(vrows);
    @memcpy(vrows, rows);
    const n_clusters = rows_n / 64;
    const order = try allocator.alloc(u32, n_clusters);
    defer allocator.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    var oseed: u64 = 777;
    var oi: usize = n_clusters - 1;
    while (oi > 0) : (oi -= 1) {
        oseed = oseed *% 6364136223846793005 +% 1442695040888963407;
        const pick: usize = @intCast(oseed % (oi + 1));
        const tmp = order[oi];
        order[oi] = order[pick];
        order[pick] = tmp;
    }
    {
        var checksum: f64 = 0;
        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        for (0..reps) |_| {
            var gauge: f32 = -std.math.inf(f32);
            var exact_w: f64 = 0;
            @memset(numer, 0);
            for (order, 0..) |c, ci| {
                const off = @as(usize, c) * 64;
                if (ci + 1 < order.len) {
                    // Prefetch the next scattered batch's K and V rows while
                    // this batch computes.
                    const noff = @as(usize, order[ci + 1]) * 64 * d;
                    var pl: usize = 0;
                    while (pl < 64 * d) : (pl += 64) {
                        @prefetch(&rows[noff + pl], .{ .rw = .read, .locality = 2 });
                        @prefetch(&vrows[noff + pl], .{ .rw = .read, .locality = 2 });
                    }
                }
                const sc = scores[0..64];
                simd.primitives.scoreRows4F16(sc, &query, rows[off * d ..], d);
                const m = 0.088 * simd.primitives.vecMaxReduce(sc);
                if (m > gauge) {
                    if (gauge != -std.math.inf(f32)) {
                        const rescale: f32 = @exp(gauge - m);
                        simd.primitives.vecScale(numer, numer, rescale);
                        exact_w *= rescale;
                    }
                    gauge = m;
                }
                exact_w += simd.primitives.vecExpAffineSumInPlace(sc, 0.088, -gauge);
                simd.primitives.weightedAccumRows4F16(numer, sc, vrows[off * d ..], d);
            }
            checksum += exact_w;
        }
        const t1 = std.Io.Clock.awake.now(io).nanoseconds;
        const secs = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
        try stdout.print("exactBatch context-shape (64-row scattered, K+V, prefetch-next): {d:.2} GB/s eff ({d:.2} ns/row) [chk {d:.3}]\n", .{
            2 * bytes_per_pass * reps / secs / 1e9,
            secs * 1e9 / @as(f64, reps * rows_n),
            checksum,
        });
        try stdout.flush();
    }

    inline for (.{ "scoreRows4F16 (runtime d)", "scoreRows4F16 (comptime d=128)", "weightedAccumRows4F16" }, 0..) |name, which| {
        var checksum: f64 = 0;
        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        for (0..reps) |_| {
            switch (which) {
                0 => simd.primitives.scoreRows4F16(scores, &query, rows, d),
                1 => scoreRows4F16Comptime(d, scores, &query, rows),
                2 => simd.primitives.weightedAccumRows4F16(numer, scores, rows, d),
                else => unreachable,
            }
            checksum += scores[rows_n - 1] + numer[0];
        }
        const t1 = std.Io.Clock.awake.now(io).nanoseconds;
        const secs = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
        try stdout.print("{s}: {d:.2} GB/s ({d:.2} ns/row) [chk {d:.3}]\n", .{
            name,
            bytes_per_pass * reps / secs / 1e9,
            secs * 1e9 / @as(f64, reps * rows_n),
            checksum,
        });
        try stdout.flush();
    }
}
