//! Behavioral tests for the eager runtime (`exec.zig`): buffer-pool reuse,
//! matmul/elementwise/broadcast dispatch, typed non-f32 kernels, grouped
//! attention forward parity, RoPE tables, cross-entropy/layer-norm/softmax/
//! reduction kernels vs naive f64 references, scatter-add, and dropout. Tests
//! that exercise the non-pub tiled-attention internals stay inline in exec.zig.
const std = @import("std");
const backend_mod = @import("backend.zig");
const exec = @import("exec.zig");
const exec_elementwise = @import("exec/elementwise.zig");
const exec_row_ops = @import("exec/row_ops.zig");
const exec_moe_chain = @import("exec/moe_chain.zig");
const dtype_mod = @import("dtype.zig");
const parallel = @import("parallel.zig");
const rng = @import("rng.zig");
const tensor = @import("tensor.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;
const ExecContext = exec.ExecContext;
const LayoutClass = exec.LayoutClass;
const CrossEntropyOptions = exec.CrossEntropyOptions;
const Reduction = exec.Reduction;

// Spot-check `got` (row-major m x n) against naive f64 dot products at a
// deterministic sample of positions; ground truth for the big matmuls below
// without an O(m*n*k) reference pass in Debug test runs.
fn expectSampledMatmulParity(
    got: []const f32,
    a_data: []const f32,
    b_data: []const f32,
    m: usize,
    n: usize,
    k: usize,
    tolerance: f32,
) !void {
    var s: usize = 0;
    while (s < 64) : (s += 1) {
        const i = (s * 769) % m;
        const j = (s * 521) % n;
        var acc: f64 = 0;
        for (0..k) |p| acc += @as(f64, a_data[i * k + p]) * @as(f64, b_data[p * n + j]);
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), got[i * n + j], tolerance);
    }
}

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

fn expectCloseToF64(want: f64, got: f32, rtol: f64, atol: f64) !void {
    const diff = @abs(@as(f64, got) - want);
    if (diff <= atol or diff <= rtol * @abs(want)) return;
    std.debug.print("expected {d}, got {d} (diff {d})\n", .{ want, got, diff });
    return error.TestUnexpectedResult;
}

const TestNaiveCrossEntropy = struct {
    loss: f64,
    row_losses: []f64,
    grads: []f64,

    fn deinit(self: *TestNaiveCrossEntropy, allocator: Allocator) void {
        allocator.free(self.row_losses);
        allocator.free(self.grads);
        self.* = undefined;
    }
};

/// Plain-loop f64 reference for the cross-entropy Ex kernels over a
/// contiguous (outer, class, inner) layout. `upstream_scale` is the scalar
/// upstream gradient; for `.none` it multiplies `per_row_upstream`.
fn testNaiveCrossEntropy(
    allocator: Allocator,
    input: []const f32,
    outer: usize,
    class_count: usize,
    inner: usize,
    labels: []const usize,
    options: CrossEntropyOptions,
    upstream_scale: f64,
    per_row_upstream: ?[]const f32,
) !TestNaiveCrossEntropy {
    const eps: f64 = options.label_smoothing;
    const k_f: f64 = @floatFromInt(class_count);
    const position_count = outer * inner;
    const row_losses = try allocator.alloc(f64, position_count);
    errdefer allocator.free(row_losses);
    const grads = try allocator.alloc(f64, outer * class_count * inner);
    errdefer allocator.free(grads);
    @memset(grads, 0);

    var valid_count: usize = 0;
    for (labels) |label| {
        if (options.ignore_index) |ignore_index| {
            if (label == ignore_index) continue;
        }
        valid_count += 1;
    }

    var loss_sum: f64 = 0;
    for (0..outer) |outer_i| {
        const base = outer_i * class_count * inner;
        for (0..inner) |inner_i| {
            const row = outer_i * inner + inner_i;
            const label = labels[row];
            const ignored = if (options.ignore_index) |ignore_index| label == ignore_index else false;
            if (ignored) {
                row_losses[row] = 0;
                continue;
            }

            var max_value = -std.math.inf(f64);
            for (0..class_count) |class_i| {
                max_value = @max(max_value, @as(f64, input[base + class_i * inner + inner_i]));
            }
            var sum_exp: f64 = 0;
            var logit_sum: f64 = 0;
            for (0..class_count) |class_i| {
                const value: f64 = input[base + class_i * inner + inner_i];
                sum_exp += @exp(value - max_value);
                logit_sum += value;
            }
            const lse = @log(sum_exp) + max_value;
            row_losses[row] = lse - (1 - eps) * @as(f64, input[base + label * inner + inner_i]) - (eps / k_f) * logit_sum;
            loss_sum += row_losses[row];

            const row_scale: f64 = switch (options.reduction) {
                .mean => if (valid_count == 0) 0 else upstream_scale / @as(f64, @floatFromInt(valid_count)),
                .sum => upstream_scale,
                .none => upstream_scale * @as(f64, per_row_upstream.?[row]),
            };
            for (0..class_count) |class_i| {
                const value: f64 = input[base + class_i * inner + inner_i];
                var grad = @exp(value - lse) - eps / k_f;
                if (class_i == label) grad -= 1 - eps;
                grads[base + class_i * inner + inner_i] = grad * row_scale;
            }
        }
    }

    const loss: f64 = switch (options.reduction) {
        .mean => if (valid_count == 0) 0 else loss_sum / @as(f64, @floatFromInt(valid_count)),
        .sum => loss_sum,
        .none => 0,
    };
    return .{ .loss = loss, .row_losses = row_losses, .grads = grads };
}

fn testNaiveLayerNorm(
    allocator: Allocator,
    data: []const f32,
    outer: usize,
    axis_dim: usize,
    inner: usize,
    weights: ?[]const f32,
    biases: ?[]const f32,
    eps: f64,
) ![]f64 {
    const out = try allocator.alloc(f64, data.len);
    errdefer allocator.free(out);
    const n = @as(f64, @floatFromInt(axis_dim));
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sum_acc: f64 = 0;
            for (0..axis_dim) |axis_i| sum_acc += data[base + axis_i * inner + inner_i];
            const mean_value = sum_acc / n;
            var sumsq: f64 = 0;
            for (0..axis_dim) |axis_i| {
                const centered = @as(f64, data[base + axis_i * inner + inner_i]) - mean_value;
                sumsq += centered * centered;
            }
            const inv_sigma = 1 / @sqrt(sumsq / n + eps);
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                var value = (@as(f64, data[offset]) - mean_value) * inv_sigma;
                if (weights) |w| value = value * w[axis_i] + biases.?[axis_i];
                out[offset] = value;
            }
        }
    }
    return out;
}

const TestNaiveLayerNormGrads = struct {
    dx: []f64,
    dweight: []f64,
    dbias: []f64,

    fn deinit(self: *TestNaiveLayerNormGrads, allocator: Allocator) void {
        allocator.free(self.dx);
        allocator.free(self.dweight);
        allocator.free(self.dbias);
        self.* = undefined;
    }
};

fn testNaiveLayerNormBackward(
    allocator: Allocator,
    data: []const f32,
    grad: []const f32,
    outer: usize,
    axis_dim: usize,
    inner: usize,
    weights: ?[]const f32,
    eps: f64,
) !TestNaiveLayerNormGrads {
    const dx = try allocator.alloc(f64, data.len);
    errdefer allocator.free(dx);
    const dweight = try allocator.alloc(f64, axis_dim);
    errdefer allocator.free(dweight);
    const dbias = try allocator.alloc(f64, axis_dim);
    errdefer allocator.free(dbias);
    @memset(dweight, 0);
    @memset(dbias, 0);

    const n = @as(f64, @floatFromInt(axis_dim));
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sum_acc: f64 = 0;
            for (0..axis_dim) |axis_i| sum_acc += data[base + axis_i * inner + inner_i];
            const mean_value = sum_acc / n;
            var sumsq: f64 = 0;
            for (0..axis_dim) |axis_i| {
                const centered = @as(f64, data[base + axis_i * inner + inner_i]) - mean_value;
                sumsq += centered * centered;
            }
            const inv_sigma = 1 / @sqrt(sumsq / n + eps);

            var gsum: f64 = 0;
            var gdot: f64 = 0;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const upstream = @as(f64, grad[offset]) * (if (weights) |w| @as(f64, w[axis_i]) else 1);
                const normalized = (@as(f64, data[offset]) - mean_value) * inv_sigma;
                gsum += upstream;
                gdot += upstream * normalized;
            }
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const upstream = @as(f64, grad[offset]) * (if (weights) |w| @as(f64, w[axis_i]) else 1);
                const normalized = (@as(f64, data[offset]) - mean_value) * inv_sigma;
                dx[offset] = inv_sigma * (upstream - gsum / n - normalized * gdot / n);
                dweight[axis_i] += @as(f64, grad[offset]) * normalized;
                dbias[axis_i] += grad[offset];
            }
        }
    }
    return .{ .dx = dx, .dweight = dweight, .dbias = dbias };
}

fn testNaiveSoftmaxRow(allocator: Allocator, row: []const f32) ![]f64 {
    const out = try allocator.alloc(f64, row.len);
    errdefer allocator.free(out);
    var max_value = -std.math.inf(f64);
    for (row) |value| max_value = @max(max_value, @as(f64, value));
    var sum_exp: f64 = 0;
    for (row, out) |value, *e| {
        e.* = @exp(@as(f64, value) - max_value);
        sum_exp += e.*;
    }
    for (out) |*value| value.* /= sum_exp;
    return out;
}

/// Serial reference for `scatterAddAxisRank` with axis == 0: dense zeros plus
/// row accumulation in index order — the exact algorithm of the serial path,
/// so parity assertions against it are BITWISE.
fn scatterAddAxis0Reference(expected: []f32, grad: []const f32, row_len: usize, indices: []const usize) void {
    @memset(expected, 0);
    for (indices, 0..) |index, row| {
        for (expected[index * row_len ..][0..row_len], grad[row * row_len ..][0..row_len]) |*d, v| {
            d.* += v;
        }
    }
}

/// Test-side dropout mask: keep element i iff the 53-bit uniform of
/// rng.at(seed, i) is < 1 - p — the exact predicate of dropoutRange.
fn dropoutKeeps(seed: u64, i: usize, p: f32) bool {
    const uniform = @as(f64, @floatFromInt(rng.at(seed, i) >> 11)) * 0x1.0p-53;
    return uniform < 1.0 - @as(f64, p);
}

test "exec context reuses released output buffers" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(&.{3}, &.{ 1, 2, 3 });
    defer a.deinit();
    var b = try ctx.fromSlice(&.{3}, &.{ 4, 5, 6 });
    defer b.deinit();

    var first = try ctx.add(&a, &b);
    const first_buffer = first.buffer;
    first.deinit();

    var second = try ctx.add(&a, &b);
    defer second.deinit();

    try std.testing.expect(second.buffer == first_buffer);
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, second.dataConst());
}

test "exec context wraps borrowed ranked slices without copying" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var values = [_]f32{ 1, 2, 3, 4 };
    var x = try ctx.fromBorrowedSliceRank(2, .{ 2, 2 }, values[0..]);
    defer x.deinit();

    values[3] = 40;
    try std.testing.expectEqual(@as(f32, 40), x.dataConst()[3]);

    var doubled = try ctx.add(&x, &x);
    defer doubled.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6, 80 }, doubled.dataConst());
}

test "exec context matmul uses backend into pooled output" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(&.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSlice(&.{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();

    var c = try ctx.matmul(&a, &b);
    defer c.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, c.dataConst());
}

test "exec context matmul transpose variants use backend outputs" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(&.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSlice(&.{ 2, 3 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();
    var c = try ctx.fromSlice(&.{ 2, 2 }, &.{ 7, 8, 9, 10 });
    defer c.deinit();

    var nt = try ctx.matmulTransB(&a, &b);
    defer nt.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 50, 68, 122, 167 }, nt.dataConst());

    var tn = try ctx.matmulTransA(&a, &c);
    defer tn.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 43, 48, 59, 66, 75, 84 }, tn.dataConst());

    var column = try ctx.fromSlice(&.{ 3, 1 }, &.{ 2, 3, 4 });
    defer column.deinit();
    var gemv = try ctx.matmul(&a, &column);
    defer gemv.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 20, 47 }, gemv.dataConst());
}

test "exec context matmul around the blocked-gemm work threshold stays consistent" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // 768 x 512 x 512 sits exactly at the blocked-GEMM work threshold and
    // routes to the cache-blocked kernel on the no-BLAS native path; m = 767
    // sits one below and stays on the register-tiled row kernels. Rows of C
    // depend only on the matching rows of A, so the 767-row result must agree
    // with the first 767 rows of the 768-row result across the dispatch
    // boundary (tolerance covers FMA-vs-mul/add rounding and the BLAS/scalar
    // dispatches in the other build configs).
    const n = 512;
    const k = 512;

    const a_data = try allocator.alloc(f32, 768 * k);
    defer allocator.free(a_data);
    const b_data = try allocator.alloc(f32, k * n);
    defer allocator.free(b_data);
    var prng = std.Random.DefaultPrng.init(0xb10c);
    const rand = prng.random();
    for (a_data) |*v| v.* = rand.float(f32) - 0.5;
    for (b_data) |*v| v.* = rand.float(f32) - 0.5;

    var b = try ctx.fromSlice(&.{ k, n }, b_data);
    defer b.deinit();

    var a_above = try ctx.fromSlice(&.{ 768, k }, a_data);
    defer a_above.deinit();
    var above = try ctx.matmul(&a_above, &b);
    defer above.deinit();
    try expectSampledMatmulParity(above.dataConst(), a_data, b_data, 768, n, k, 2e-3);

    var a_below = try ctx.fromSlice(&.{ 767, k }, a_data[0 .. 767 * k]);
    defer a_below.deinit();
    var below = try ctx.matmul(&a_below, &b);
    defer below.deinit();

    for (below.dataConst(), above.dataConst()[0 .. 767 * n]) |lo, hi| {
        try std.testing.expectApproxEqAbs(hi, lo, 2e-3);
    }
}

test "exec context matmul blocked path covers transposed and strided inputs" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Above the blocked work threshold with a k that is not a kc multiple.
    const m = 512;
    const n = 512;
    const k = 770;

    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);
    const at_data = try allocator.alloc(f32, k * m);
    defer allocator.free(at_data);
    const b_data = try allocator.alloc(f32, k * n);
    defer allocator.free(b_data);
    const bt_data = try allocator.alloc(f32, n * k);
    defer allocator.free(bt_data);
    var prng = std.Random.DefaultPrng.init(0xb10c2);
    const rand = prng.random();
    for (a_data) |*v| v.* = rand.float(f32) - 0.5;
    for (b_data) |*v| v.* = rand.float(f32) - 0.5;
    for (0..m) |i| {
        for (0..k) |p| at_data[p * m + i] = a_data[i * k + p];
    }
    for (0..k) |p| {
        for (0..n) |j| bt_data[j * k + p] = b_data[p * n + j];
    }

    var a = try ctx.fromSlice(&.{ m, k }, a_data);
    defer a.deinit();
    var a_t = try ctx.fromSlice(&.{ k, m }, at_data);
    defer a_t.deinit();
    var b = try ctx.fromSlice(&.{ k, n }, b_data);
    defer b.deinit();
    var b_t = try ctx.fromSlice(&.{ n, k }, bt_data);
    defer b_t.deinit();

    var want = try ctx.matmul(&a, &b);
    defer want.deinit();
    try expectSampledMatmulParity(want.dataConst(), a_data, b_data, m, n, k, 2e-3);

    var tn = try ctx.matmulTransA(&a_t, &b);
    defer tn.deinit();
    for (want.dataConst(), tn.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 2e-3);

    var nt = try ctx.matmulTransB(&a, &b_t);
    defer nt.deinit();
    for (want.dataConst(), nt.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 2e-3);

    // Strided non-contiguous input: a transposed view of A^T is A again;
    // prepareContiguous materializes it before the kernel dispatch.
    var a_view = try a_t.viewWithStrides(&.{ m, k }, &.{ 1, m });
    defer a_view.deinit();
    var nn = try ctx.matmul(&a_view, &b);
    defer nn.deinit();
    for (want.dataConst(), nn.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 2e-3);
}

test "exec context matmul transposed f16 RHS uses backend output" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(&.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSliceRankTyped(.f16, 2, .{ 2, 3 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();

    var got = try ctx.matmulTransB2DWithF16Rhs(&a, &b);
    defer got.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 50, 68, 122, 167 }, got.dataConst());
}

test "exec context matmul transposed bf16 RHS uses backend output" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(&.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSliceRankTyped(.bf16, 2, .{ 2, 3 }, &.{
        dtype_mod.f32ToBf16(7),
        dtype_mod.f32ToBf16(8),
        dtype_mod.f32ToBf16(9),
        dtype_mod.f32ToBf16(10),
        dtype_mod.f32ToBf16(11),
        dtype_mod.f32ToBf16(12),
    });
    defer b.deinit();

    var got = try ctx.matmulTransB2DWithBf16Rhs(&a, &b);
    defer got.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 50, 68, 122, 167 }, got.dataConst());
}

test "cpu f32 shadow route matches the streaming kernels and caches per buffer" {
    if (@import("build_options").use_gpu) return error.SkipZigTest;
    const exec_matmul = @import("exec/matmul.zig");
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Above the crossover so the shadow arm engages (min_m forced to 4).
    const m = 5;
    const n = 7;
    const k = 33;
    var prng = std.Random.DefaultPrng.init(99);
    const rand = prng.random();
    var a_data: [m * k]f32 = undefined;
    for (&a_data) |*v| v.* = rand.floatNorm(f32);
    var b16_data: [n * k]f16 = undefined;
    var bbf_data: [n * k]u16 = undefined;
    for (&b16_data, &bbf_data) |*h, *bb| {
        const value = rand.floatNorm(f32) * 0.1;
        h.* = @floatCast(value);
        bb.* = dtype_mod.f32ToBf16(value);
    }

    var a = try ctx.fromSlice(&.{ m, k }, &a_data);
    defer a.deinit();
    var b16 = try ctx.fromSliceRankTyped(.f16, 2, .{ n, k }, &b16_data);
    defer b16.deinit();
    var bbf = try ctx.fromSliceRankTyped(.bf16, 2, .{ n, k }, &bbf_data);
    defer bbf.deinit();

    exec_matmul.setCpuF32Shadow(false, null);
    var want16 = try ctx.matmulTransB2DWithF16Rhs(&a, &b16);
    defer want16.deinit();
    var wantbf = try ctx.matmulTransB2DWithBf16Rhs(&a, &bbf);
    defer wantbf.deinit();

    exec_matmul.setCpuF32Shadow(true, 4);
    defer exec_matmul.setCpuF32Shadow(null, 32);
    var got16 = try ctx.matmulTransB2DWithF16Rhs(&a, &b16);
    defer got16.deinit();
    var gotbf = try ctx.matmulTransB2DWithBf16Rhs(&a, &bbf);
    defer gotbf.deinit();

    // The shadow's widen is exact for both formats; results differ from the
    // streaming kernels only by accumulation order (and, for f16, by the
    // skipped A cast — the shadow route is the MORE precise one).
    for (want16.dataConst(), got16.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 2e-3);
    for (wantbf.dataConst(), gotbf.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 2e-3);

    // Second call reuses the cached shadow (the buffer's .cpu resource).
    try std.testing.expect(b16.buffer.acceleratorResource(.cpu) != null);
    const first = b16.buffer.acceleratorResource(.cpu).?;
    var again = try ctx.matmulTransB2DWithF16Rhs(&a, &b16);
    defer again.deinit();
    try std.testing.expect(b16.buffer.acceleratorResource(.cpu).? == first);
    try std.testing.expectEqualSlices(f32, got16.dataConst(), again.dataConst());
}

test "exec context applies unary ops through materialized inputs" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(&.{ 1, 3 }, &.{ -1, 2, -3 });
    defer x.deinit();
    var broadcast = try ctx.broadcastToRank(2, &x, .{ 2, 3 });
    defer broadcast.deinit();

    var y = try ctx.unary(.relu, &broadcast);
    defer y.deinit();

    try std.testing.expect(y.isContiguous());
    try std.testing.expectEqualSlices(f32, &.{ 0, 2, 0, 0, 2, 0 }, y.dataConst());

    var z = try ctx.relu(&broadcast);
    defer z.deinit();
    try std.testing.expectEqualSlices(f32, y.dataConst(), z.dataConst());
}

test "exec context reduces a compile-time axis" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(3, .{ 2, 2, 3 }, &.{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    });
    defer x.deinit();

    var rows = try ctx.sumAxisRank(3, &x, 1);
    defer rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, rows.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9, 17, 19, 21 }, rows.dataConst());

    var vector = try ctx.fromSliceRank(1, .{3}, &.{ 2, 4, 6 });
    defer vector.deinit();

    var all = try ctx.sumAxisRank(1, &vector, 0);
    defer all.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, all.shape.slice());
    try std.testing.expectEqual(@as(f32, 12), all.item());
}

test "exec context exposes fixed-rank construction and elementwise execution" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSliceRank(3, .{ 2, 1, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSliceRank(3, .{ 2, 1, 3 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer b.deinit();

    var c = try ctx.addRank(3, &a, &b);
    defer c.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 3 }, c.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44, 55, 66 }, c.dataConst());
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.addRank(2, &a, &b));
}

test "exec context applies explicit tail broadcast without materializing the view" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(&.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try ctx.fromSlice(&.{3}, &.{ 10, 20, 30 });
    defer bias.deinit();
    var broadcast = try ctx.broadcastTo(&bias, &.{ 2, 3 });
    defer broadcast.deinit();

    try std.testing.expect(broadcast.buffer == bias.buffer);
    try std.testing.expectEqual(LayoutClass.tail_broadcast, ctx.classify(&broadcast));

    var y = try ctx.add(&x, &broadcast);
    defer y.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, y.dataConst());

    var z = try ctx.sub(&x, &broadcast);
    defer z.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -9, -18, -27, -6, -15, -24 }, z.dataConst());

    var m = try ctx.mul(&x, &broadcast);
    defer m.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 40, 90, 40, 100, 180 }, m.dataConst());
}

test "exec context applies in-place elementwise ops with contiguous and broadcast operands" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try ctx.fromSliceRank(1, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();
    var gate = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer gate.deinit();
    var broadcast = try ctx.broadcastToRank(2, &bias, .{ 2, 3 });
    defer broadcast.deinit();

    try ctx.addInPlace(&x, &broadcast);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, x.dataConst());

    try ctx.mulInPlace(&x, &gate);
    try std.testing.expectEqualSlices(f32, &.{ 11, 44, 99, 56, 125, 216 }, x.dataConst());
}

test "exec context take elementwise reuses unique contiguous input" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    var bias = try ctx.fromSliceRank(1, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();
    var broadcast = try ctx.broadcastToRank(2, &bias, .{ 2, 3 });
    defer broadcast.deinit();

    const original_buffer = x.buffer;
    var y = try ctx.takeAdd(&x, &broadcast);
    defer y.deinit();

    try std.testing.expect(y.buffer == original_buffer);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, y.dataConst());
}

test "exec context take unary and scale reuse unique contiguous input" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(&.{4}, &.{ -1, 2, -3, 4 });
    const original_buffer = x.buffer;

    var y = try ctx.takeRelu(&x);
    try std.testing.expect(y.buffer == original_buffer);
    try std.testing.expectEqualSlices(f32, &.{ 0, 2, 0, 4 }, y.dataConst());

    y = try ctx.takeScale(&y, 0.5);
    defer y.deinit();
    try std.testing.expect(y.buffer == original_buffer);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 0, 2 }, y.dataConst());
}

test "exec context take elementwise falls back for shared buffers" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(&.{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var shared = try x.cloneView();
    var b = try ctx.fromSlice(&.{3}, &.{ 10, 20, 30 });
    defer b.deinit();

    var y = try ctx.takeMul(&shared, &b);
    defer y.deinit();

    try std.testing.expect(y.buffer != x.buffer);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, x.dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 10, 40, 90 }, y.dataConst());
}

test "exec context take elementwise falls back for views and preserves input on error" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var source = try ctx.fromSlice(&.{3}, &.{ 1, 2, 3 });
    defer source.deinit();
    var broadcast = try ctx.broadcastTo(&source, &.{ 2, 3 });
    var x = try ctx.fromSlice(&.{ 2, 3 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer x.deinit();

    var y = try ctx.takeSub(&broadcast, &x);
    defer y.deinit();
    try std.testing.expect(y.buffer != source.buffer);
    try std.testing.expectEqualSlices(f32, &.{ -9, -18, -27, -39, -48, -57 }, y.dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, source.dataConst());

    var a = try ctx.fromSlice(&.{2}, &.{ 1, 2 });
    defer a.deinit();
    var b = try ctx.fromSlice(&.{3}, &.{ 1, 2, 3 });
    defer b.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.takeAdd(&a, &b));
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, a.dataConst());
}

test "exec context combines fixed-rank ops with explicit broadcast views" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try ctx.fromSliceRank(1, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();
    var broadcast = try ctx.broadcastToRank(2, &bias, .{ 2, 3 });
    defer broadcast.deinit();

    var y = try ctx.addRank(2, &x, &broadcast);
    defer y.deinit();

    try std.testing.expect(broadcast.buffer == bias.buffer);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, y.dataConst());
}

test "exec context handles broadcast operands on both sides of elementwise ops" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try ctx.fromSliceRank(1, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();
    var bias_b = try ctx.broadcastToRank(2, &bias, .{ 2, 3 });
    defer bias_b.deinit();
    var scalar_value = try ctx.scalar(2);
    defer scalar_value.deinit();
    var scalar_b = try ctx.broadcastToRank(2, &scalar_value, .{ 2, 3 });
    defer scalar_b.deinit();

    var left_sub = try ctx.subRank(2, &bias_b, &x);
    defer left_sub.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 9, 18, 27, 6, 15, 24 }, left_sub.dataConst());

    var right_sub = try ctx.subRank(2, &x, &bias_b);
    defer right_sub.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -9, -18, -27, -6, -15, -24 }, right_sub.dataConst());

    var both_broadcast_sub = try ctx.subRank(2, &bias_b, &scalar_b);
    defer both_broadcast_sub.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 8, 18, 28, 8, 18, 28 }, both_broadcast_sub.dataConst());

    var both_broadcast_mul = try ctx.mulRank(2, &bias_b, &scalar_b);
    defer both_broadcast_mul.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 20, 40, 60, 20, 40, 60 }, both_broadcast_mul.dataConst());
}

test "exec context rank-specializes elementwise ops above rank four" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSliceRank(5, .{ 1, 1, 1, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSliceRank(5, .{ 1, 1, 1, 2, 3 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer b.deinit();

    var sum = try ctx.add(&a, &b);
    defer sum.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44, 55, 66 }, sum.dataConst());

    var diff = try ctx.subRank(5, &b, &a);
    defer diff.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 9, 18, 27, 36, 45, 54 }, diff.dataConst());

    var product = try ctx.mulRank(5, &a, &b);
    defer product.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 40, 90, 160, 250, 360 }, product.dataConst());
}

test "exec context reduces broadcast gradient to source shape" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var gy = try ctx.fromSlice(&.{ 2, 3 }, &.{ 1, 1, 1, 1, 1, 1 });
    defer gy.deinit();

    var reduced = try ctx.reduceBroadcast(&gy, &.{3});
    defer reduced.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2 }, reduced.dataConst());
}

test "exec context handles scalar broadcast and non-tail broadcast fallback" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(&.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var scalar_value = try ctx.scalar(10);
    defer scalar_value.deinit();
    var scalar_b = try ctx.broadcastTo(&scalar_value, &.{ 2, 3 });
    defer scalar_b.deinit();

    var y = try ctx.add(&x, &scalar_b);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 12, 13, 14, 15, 16 }, y.dataConst());

    var middle = try ctx.fromSlice(&.{ 2, 1, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer middle.deinit();
    var middle_b = try ctx.broadcastTo(&middle, &.{ 2, 4, 3 });
    defer middle_b.deinit();
    try std.testing.expectEqual(LayoutClass.arbitrary, ctx.classify(&middle_b));

    var zeros = try ctx.zeros(&.{ 2, 4, 3 });
    defer zeros.deinit();
    var copied = try ctx.add(&zeros, &middle_b);
    defer copied.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
        4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
    }, copied.dataConst());
}

test "exec context optimizes last-axis concat and updates" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer b.deinit();

    var inputs = [_]*const Tensor{ &a, &b };
    var joined = try ctx.concatAxisRank(2, &inputs, 1);
    defer joined.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, joined.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 10, 20, 30, 3, 4, 40, 50, 60 }, joined.dataConst());

    var update = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 7, 8, 9, 10 });
    defer update.deinit();
    var sliced = try ctx.setSliceAxisRank(2, &joined, &update, 1, 1);
    defer sliced.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 7, 8, 20, 30, 3, 9, 10, 50, 60 }, sliced.dataConst());

    var row_update = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 100, 200, 300, 400 });
    defer row_update.deinit();
    var rows = try ctx.setRowsAxisRank(2, &joined, &row_update, 1, &.{ 4, 0 });
    defer rows.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 200, 2, 10, 20, 100, 400, 4, 40, 50, 300 }, rows.dataConst());
}

test "exec context zeros indexed rows on non-leading axes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(3, .{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    var middle = try ctx.zeroRowsAxisRank(3, &x, 1, &.{ 2, 0 });
    defer middle.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 3, 4, 0, 0, 0, 0, 9, 10, 0, 0 }, middle.dataConst());

    var last = try ctx.zeroRowsAxisRank(3, &x, 2, &.{1});
    defer last.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0 }, last.dataConst());
}

test "exec context uses contiguous reduction paths for rank-one and last-axis sums" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var v = try ctx.fromSliceRank(1, .{4}, &.{ 1, 2, 3, 4 });
    defer v.deinit();
    var total = try ctx.sumAxisRank(1, &v, 0);
    defer total.deinit();
    try std.testing.expectEqualSlices(f32, &.{10}, total.dataConst());

    var x = try ctx.fromSliceRank(3, .{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();
    var rows = try ctx.sumAxisRank(3, &x, 2);
    defer rows.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, rows.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 6, 15, 24, 33 }, rows.dataConst());
}

test "exec context runs typed data movement and indexing kernels" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var ids = try ctx.fromSliceRankTyped(.u16, 2, .{ 3, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer ids.deinit();

    var rows = try ctx.gatherAxisRankTyped(.u16, 2, &ids, 0, &.{ 2, 0 });
    defer rows.deinit();
    try std.testing.expectEqualSlices(u16, &.{ 7, 8, 9, 1, 2, 3 }, rows.dataConst());

    var narrowed = try ctx.narrowAxisRankTyped(.u16, 2, &ids, 1, 1, 2);
    defer narrowed.deinit();
    var narrowed_data: [6]u16 = undefined;
    try narrowed.copyTo(&narrowed_data);
    try std.testing.expectEqualSlices(u16, &.{ 2, 3, 5, 6, 8, 9 }, &narrowed_data);

    var extra = try ctx.fromSliceRankTyped(.u16, 2, .{ 3, 1 }, &.{ 10, 11, 12 });
    defer extra.deinit();
    var concat_inputs = [_]*const tensor.TensorOf(.u16){ &ids, &extra };
    var joined = try ctx.concatAxisRankTyped(.u16, 2, &concat_inputs, 1);
    defer joined.deinit();
    try std.testing.expectEqualSlices(u16, &.{ 1, 2, 3, 10, 4, 5, 6, 11, 7, 8, 9, 12 }, joined.dataConst());

    var update = try ctx.fromSliceRankTyped(.u16, 2, .{ 3, 2 }, &.{ 20, 21, 22, 23, 24, 25 });
    defer update.deinit();
    var sliced = try ctx.setSliceAxisRankTyped(.u16, 2, &joined, &update, 1, 1);
    defer sliced.deinit();
    try std.testing.expectEqualSlices(u16, &.{ 1, 20, 21, 10, 4, 22, 23, 11, 7, 24, 25, 12 }, sliced.dataConst());

    var row_update = try ctx.fromSliceRankTyped(.u16, 2, .{ 2, 4 }, &.{ 30, 31, 32, 33, 40, 41, 42, 43 });
    defer row_update.deinit();
    var replaced = try ctx.setRowsAxisRankTyped(.u16, 2, &joined, &row_update, 0, &.{ 2, 0 });
    defer replaced.deinit();
    try std.testing.expectEqualSlices(u16, &.{ 40, 41, 42, 43, 4, 5, 6, 11, 30, 31, 32, 33 }, replaced.dataConst());
}

test "exec context runs typed float forward math kernels" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSliceRankTyped(.f64, 2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try ctx.fromSliceRankTyped(.f64, 2, .{ 2, 2 }, &.{ 10, 20, 30, 40 });
    defer b.deinit();

    var sum = try ctx.addRankTyped(.f64, 2, &a, &b);
    defer sum.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 11, 22, 33, 44 }, sum.dataConst());

    var reduced = try ctx.sumAxisRankTyped(.f64, 2, &sum, 1);
    defer reduced.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 33, 77 }, reduced.dataConst());

    var dot64 = try ctx.dotTyped(.f64, &a, &b);
    defer dot64.deinit();
    try std.testing.expectEqual(@as(f64, 300), dot64.dataConst()[0]);

    var matmul64 = try ctx.matmul2DTyped(.f64, &a, &b);
    defer matmul64.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 70, 100, 150, 220 }, matmul64.dataConst());

    var h1 = try ctx.fromSliceRankTyped(.f16, 2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer h1.deinit();
    var h2 = try ctx.fromSliceRankTyped(.f16, 2, .{ 2, 2 }, &.{ 2, 3, 4, 5 });
    defer h2.deinit();
    var hmul = try ctx.mulRankTyped(.f16, 2, &h1, &h2);
    defer hmul.deinit();
    try std.testing.expectEqualSlices(f16, &.{ 2, 6, 12, 20 }, hmul.dataConst());

    var hsum = try ctx.sumAxisRankTyped(.f16, 2, &hmul, 1);
    defer hsum.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 8, 32 }, hsum.dataConst());

    var hdot = try ctx.dotTyped(.f16, &h1, &h2);
    defer hdot.deinit();
    try std.testing.expectEqual(@as(f16, 40), hdot.dataConst()[0]);

    var hleft = try ctx.fromSliceRankTyped(.f16, 2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer hleft.deinit();
    var hright = try ctx.fromSliceRankTyped(.f16, 2, .{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer hright.deinit();
    var hproduct = try ctx.matmul2DTyped(.f16, &hleft, &hright);
    defer hproduct.deinit();
    try std.testing.expectEqualSlices(f16, &.{ 58, 64, 139, 154 }, hproduct.dataConst());

    var packed_rhs = try ctx.packMatmulRhsTyped(.f16, &hright);
    defer packed_rhs.deinit();
    var packed_product = try ctx.matmul2DWithPackedRhsTyped(.f16, &hleft, &packed_rhs);
    defer packed_product.deinit();
    try std.testing.expectEqualSlices(f16, hproduct.dataConst(), packed_product.dataConst());

    var left = try ctx.fromSliceRankTyped(.bf16, 2, .{ 2, 3 }, &.{
        dtype_mod.f32ToBf16(1),
        dtype_mod.f32ToBf16(2),
        dtype_mod.f32ToBf16(3),
        dtype_mod.f32ToBf16(4),
        dtype_mod.f32ToBf16(5),
        dtype_mod.f32ToBf16(6),
    });
    defer left.deinit();
    var bf16_sum = try ctx.sumTyped(.bf16, &left);
    defer bf16_sum.deinit();
    try std.testing.expectEqualSlices(f32, &.{21}, bf16_sum.dataConst());

    var right = try ctx.fromSliceRankTyped(.bf16, 2, .{ 3, 2 }, &.{
        dtype_mod.f32ToBf16(7),
        dtype_mod.f32ToBf16(8),
        dtype_mod.f32ToBf16(9),
        dtype_mod.f32ToBf16(10),
        dtype_mod.f32ToBf16(11),
        dtype_mod.f32ToBf16(12),
    });
    defer right.deinit();
    var product = try ctx.matmul2DTyped(.bf16, &left, &right);
    defer product.deinit();
    try std.testing.expectEqual(@as(f32, 58), dtype_mod.bf16ToF32(product.dataConst()[0]));
    try std.testing.expectEqual(@as(f32, 154), dtype_mod.bf16ToF32(product.dataConst()[3]));

    var bf16_packed_rhs = try ctx.packMatmulRhsTyped(.bf16, &right);
    defer bf16_packed_rhs.deinit();
    var bf16_packed_product = try ctx.matmul2DWithPackedRhsTyped(.bf16, &left, &bf16_packed_rhs);
    defer bf16_packed_product.deinit();
    try std.testing.expectEqualSlices(u16, product.dataConst(), bf16_packed_product.dataConst());

    var cast = try ctx.castTyped(.bf16, .f32, &product);
    defer cast.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, cast.dataConst());
}

test "buffer pool reuses bucket-rounded buffers across many small temporaries" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(&.{3}, &.{ 1, 2, 3 });
    defer a.deinit();
    var b = try ctx.fromSlice(&.{3}, &.{ 4, 5, 6 });
    defer b.deinit();

    for (0..100) |_| {
        var y = try ctx.add(&a, &b);
        try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, y.dataConst());
        y.deinit();
    }

    try std.testing.expect(ctx.rt.buffers.cachedBuffers() >= 1);
    try std.testing.expectEqual(@as(usize, 2), ctx.rt.buffers.outstandingBuffers());
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

test "prepareRopeTableFactors scales frequencies; null reproduces plain RoPE" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const positions = [_]i32{ 0, 1, 5 };
    const feature_dim: usize = 8;
    const theta_base: f32 = 10000;
    const ff = [_]f32{ 1.0, 2.0, 4.0, 8.0 };

    var plain = try ctx.prepareRopeTable(&positions, feature_dim, theta_base, false);
    defer plain.deinit();
    var plain_null = try ctx.prepareRopeTableFactors(&positions, feature_dim, theta_base, false, null);
    defer plain_null.deinit();
    try std.testing.expectEqualSlices(f32, plain.values, plain_null.values);

    var scaled = try ctx.prepareRopeTableFactors(&positions, feature_dim, theta_base, false, &ff);
    defer scaled.deinit();
    const pair = scaled.pair_count;
    const angle_count = positions.len * pair;
    for (positions, 0..) |posv, pi| {
        const pos = @as(f32, @floatFromInt(posv));
        for (0..pair) |j| {
            const exponent = @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(feature_dim));
            const theta = (pos / std.math.pow(f32, theta_base, exponent)) / ff[j];
            const idx = pi * pair + j;
            try std.testing.expectApproxEqAbs(@sin(theta), scaled.values[idx], 1e-6);
            try std.testing.expectApproxEqAbs(@cos(theta), scaled.values[angle_count + idx], 1e-6);
        }
    }
}

test "exec context cross entropy ex matches a naive reference across options" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x5eed);
    const random = prng.random();

    const class_counts = [_]usize{ 1, 2, 3, 8, 17, 1000, 4099 };
    const outers = [_]usize{ 1, 2, 5, 64 };
    const reductions = [_]Reduction{ .mean, .sum, .none };
    const upstream: f32 = 0.75;

    for (class_counts) |class_count| {
        for (outers) |outer| {
            inline for (.{ 1, 3 }) |inner| {
                const rank = if (inner == 1) 2 else 3;
                const position_count = outer * inner;
                const data = try allocator.alloc(f32, outer * class_count * inner);
                defer allocator.free(data);
                for (data) |*value| value.* = random.floatNorm(f32) * 3;

                var logits = if (inner == 1)
                    try ctx.fromSliceRank(2, .{ outer, class_count }, data)
                else
                    try ctx.fromSliceRank(3, .{ outer, class_count, inner }, data);
                defer logits.deinit();

                const labels = try allocator.alloc(usize, position_count);
                defer allocator.free(labels);
                const per_row = try allocator.alloc(f32, position_count);
                defer allocator.free(per_row);
                for (per_row) |*value| value.* = random.floatNorm(f32);

                for ([_]?usize{ null, class_count }) |ignore_index| {
                    for (labels, 0..) |*label, i| {
                        label.* = random.uintLessThan(usize, class_count);
                        if (ignore_index != null and i % 3 == 1) label.* = ignore_index.?;
                    }
                    for ([_]f32{ 0, 0.1 }) |label_smoothing| {
                        for (reductions) |reduction| {
                            const options = CrossEntropyOptions{
                                .ignore_index = ignore_index,
                                .reduction = reduction,
                                .label_smoothing = label_smoothing,
                            };
                            var ref = try testNaiveCrossEntropy(allocator, data, outer, class_count, inner, labels, options, upstream, per_row);
                            defer ref.deinit(allocator);

                            var loss = try ctx.crossEntropyLossExAxisRank(rank, &logits, 1, labels, options);
                            defer loss.deinit();
                            if (reduction == .none) {
                                const losses = loss.dataConst();
                                try std.testing.expectEqual(position_count, losses.len);
                                for (losses, ref.row_losses) |got, want| {
                                    try expectCloseToF64(want, got, 2e-4, 2e-5);
                                }
                            } else {
                                try expectCloseToF64(ref.loss, loss.item(), 2e-4, 2e-5);
                            }

                            var grad = try ctx.crossEntropyBackwardExAxisRank(
                                rank,
                                &logits,
                                1,
                                labels,
                                options,
                                upstream,
                                if (reduction == .none) per_row else null,
                            );
                            defer grad.deinit();
                            for (grad.dataConst(), ref.grads) |got, want| {
                                try expectCloseToF64(want, got, 2e-4, 2e-5);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "exec cross entropy backward with saved stats is bitwise identical to recompute" {
    // The autograd node saves the forward's per-row {max, sum_exp} and the
    // backward takes the one-pass route; the contract is BITWISE equality
    // with the recompute route across both kernel layouts (inner == 1
    // vectorized rows + inner > 1 scalar strides), the parallel/serial
    // dispatch threshold, reductions, ignore_index, and label smoothing.
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x57a75);
    const random = prng.random();

    const class_counts = [_]usize{ 1, 3, 17, 4099 };
    const outers = [_]usize{ 1, 5, 64 };
    const reductions = [_]Reduction{ .mean, .sum, .none };
    const upstream: f32 = 0.75;

    for (class_counts) |class_count| {
        for (outers) |outer| {
            inline for (.{ 1, 3 }) |inner| {
                const rank = if (inner == 1) 2 else 3;
                const position_count = outer * inner;
                const data = try allocator.alloc(f32, outer * class_count * inner);
                defer allocator.free(data);
                for (data) |*value| value.* = random.floatNorm(f32) * 3;

                var logits = if (inner == 1)
                    try ctx.fromSliceRank(2, .{ outer, class_count }, data)
                else
                    try ctx.fromSliceRank(3, .{ outer, class_count, inner }, data);
                defer logits.deinit();

                const labels = try allocator.alloc(usize, position_count);
                defer allocator.free(labels);
                const per_row = try allocator.alloc(f32, position_count);
                defer allocator.free(per_row);
                for (per_row) |*value| value.* = random.floatNorm(f32);
                const row_stats = try allocator.alloc(f32, 2 * position_count);
                defer allocator.free(row_stats);

                for ([_]?usize{ null, class_count }) |ignore_index| {
                    for (labels, 0..) |*label, i| {
                        label.* = random.uintLessThan(usize, class_count);
                        if (ignore_index != null and i % 3 == 1) label.* = ignore_index.?;
                    }
                    for ([_]f32{ 0, 0.1 }) |label_smoothing| {
                        for (reductions) |reduction| {
                            const options = CrossEntropyOptions{
                                .ignore_index = ignore_index,
                                .reduction = reduction,
                                .label_smoothing = label_smoothing,
                            };
                            const per_row_arg: ?[]const f32 = if (reduction == .none) per_row else null;

                            var plain_loss = try ctx.crossEntropyLossExAxisRank(rank, &logits, 1, labels, options);
                            defer plain_loss.deinit();
                            var stats_loss = try ctx.crossEntropyLossExStatsAxisRank(rank, &logits, 1, labels, options, row_stats);
                            defer stats_loss.deinit();
                            try std.testing.expectEqualSlices(f32, plain_loss.dataConst(), stats_loss.dataConst());

                            var grad_recompute = try ctx.crossEntropyBackwardExAxisRank(rank, &logits, 1, labels, options, upstream, per_row_arg);
                            defer grad_recompute.deinit();
                            var grad_stats = try ctx.crossEntropyBackwardExStatsAxisRank(rank, &logits, 1, labels, options, upstream, per_row_arg, row_stats);
                            defer grad_stats.deinit();
                            try std.testing.expectEqualSlices(f32, grad_recompute.dataConst(), grad_stats.dataConst());
                        }
                    }
                }
            }
        }
    }

    // Stats length is validated (must be 2 * position count).
    var logits = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer logits.deinit();
    var short_stats = [_]f32{ 0, 0 };
    try std.testing.expectError(tensor.TensorError.InvalidDataLength, ctx.crossEntropyLossExStatsAxisRank(2, &logits, 1, &.{ 0, 1 }, .{}, &short_stats));
    try std.testing.expectError(tensor.TensorError.InvalidDataLength, ctx.crossEntropyBackwardExStatsAxisRank(2, &logits, 1, &.{ 0, 1 }, .{}, 1, null, &.{ 0, 0, 0 }));
}

test "exec fused linear cross-entropy backward matches the composed two-GEMM path" {
    // linearCrossEntropyBackwardUpstream overwrites the logits with the
    // logit gradient in place (fresh-buffer fallback when shared) and runs
    // the SAME two monolithic GEMMs as the composed route, so dx/dweight are
    // BITWISE equal to the reference.
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x11cea);
    const random = prng.random();

    const shapes = [_][3]usize{ .{ 1, 8, 6 }, .{ 5, 17, 4099 }, .{ 9, 8, 200 } };
    const reductions = [_]Reduction{ .mean, .sum, .none };

    for (shapes) |shape| {
        const rows = shape[0];
        const in_dim = shape[1];
        const class_count = shape[2];

        const x_data = try allocator.alloc(f32, rows * in_dim);
        defer allocator.free(x_data);
        for (x_data) |*value| value.* = random.floatNorm(f32);
        const w_data = try allocator.alloc(f32, class_count * in_dim);
        defer allocator.free(w_data);
        for (w_data) |*value| value.* = random.floatNorm(f32) * 0.3;

        var x = try ctx.fromSliceRank(2, .{ rows, in_dim }, x_data);
        defer x.deinit();
        var w = try ctx.fromSliceRank(2, .{ class_count, in_dim }, w_data);
        defer w.deinit();

        const labels = try allocator.alloc(usize, rows);
        defer allocator.free(labels);
        const per_row = try allocator.alloc(f32, rows);
        defer allocator.free(per_row);
        for (per_row) |*value| value.* = random.floatNorm(f32);
        const row_stats = try allocator.alloc(f32, 2 * rows);
        defer allocator.free(row_stats);

        for ([_]?usize{ null, class_count }) |ignore_index| {
            for (labels, 0..) |*label, i| {
                label.* = random.uintLessThan(usize, class_count);
                if (ignore_index != null and i % 3 == 1) label.* = ignore_index.?;
            }
            for ([_]f32{ 0, 0.1 }) |label_smoothing| {
                for (reductions) |reduction| {
                    const options = CrossEntropyOptions{
                        .ignore_index = ignore_index,
                        .reduction = reduction,
                        .label_smoothing = label_smoothing,
                    };
                    // Fresh logits per case: the fused VJP consumes them.
                    var logits = try ctx.matmulTransB(&x, &w);
                    defer logits.deinit();
                    var loss = try ctx.crossEntropyLossExStatsAxisRank(2, &logits, 1, labels, options, row_stats);
                    loss.deinit();

                    var gy = if (reduction == .none)
                        try ctx.fromSliceRank(1, .{rows}, per_row)
                    else
                        try ctx.fromSliceRank(1, .{1}, &.{0.75});
                    defer gy.deinit();

                    // Composed reference (before the fused call eats logits).
                    var dlogits = try ctx.crossEntropyBackwardExStatsAxisRank(
                        2,
                        &logits,
                        1,
                        labels,
                        options,
                        if (reduction == .none) 1 else 0.75,
                        if (reduction == .none) per_row else null,
                        row_stats,
                    );
                    defer dlogits.deinit();
                    var dx_ref = try ctx.matmul2D(&dlogits, &w);
                    defer dx_ref.deinit();
                    var dw_ref = try ctx.matmulTransA(&dlogits, &x);
                    defer dw_ref.deinit();

                    var grads = try ctx.linearCrossEntropyBackwardUpstream(&x, &w, &logits, labels, options, &gy, row_stats, true, true);
                    defer grads.deinit();
                    try std.testing.expectEqualSlices(f32, dx_ref.dataConst(), grads.dx.?.dataConst());
                    try std.testing.expectEqualSlices(f32, dw_ref.dataConst(), grads.dweight.?.dataConst());
                    // In place: the logits now HOLD the logit gradient.
                    try std.testing.expectEqualSlices(f32, dlogits.dataConst(), logits.dataConst());
                }
            }
        }

        // Shared logits buffer: the VJP falls back to a fresh gradient
        // tensor with identical values and leaves the logits intact.
        for (labels) |*label| label.* = random.uintLessThan(usize, class_count);
        var logits = try ctx.matmulTransB(&x, &w);
        defer logits.deinit();
        var keeper = try logits.cloneView();
        defer keeper.deinit();
        const before = try allocator.dupe(f32, logits.dataConst());
        defer allocator.free(before);
        var loss = try ctx.crossEntropyLossExStatsAxisRank(2, &logits, 1, labels, .{}, row_stats);
        loss.deinit();
        var gy = try ctx.fromSliceRank(1, .{1}, &.{1});
        defer gy.deinit();
        var dlogits = try ctx.crossEntropyBackwardExStatsAxisRank(2, &logits, 1, labels, .{}, 1, null, row_stats);
        defer dlogits.deinit();
        var dx_ref = try ctx.matmul2D(&dlogits, &w);
        defer dx_ref.deinit();
        var grads = try ctx.linearCrossEntropyBackwardUpstream(&x, &w, &logits, labels, .{}, &gy, row_stats, true, true);
        defer grads.deinit();
        try std.testing.expectEqualSlices(f32, before, logits.dataConst());
        try std.testing.expectEqualSlices(f32, dx_ref.dataConst(), grads.dx.?.dataConst());

        // Partial needs: only the requested gradients are produced.
        var logits2 = try ctx.matmulTransB(&x, &w);
        defer logits2.deinit();
        var only_x = try ctx.linearCrossEntropyBackwardUpstream(&x, &w, &logits2, labels, .{}, &gy, row_stats, true, false);
        defer only_x.deinit();
        try std.testing.expect(only_x.dx != null and only_x.dweight == null);
        var logits3 = try ctx.matmulTransB(&x, &w);
        defer logits3.deinit();
        var only_w = try ctx.linearCrossEntropyBackwardUpstream(&x, &w, &logits3, labels, .{}, &gy, row_stats, false, true);
        defer only_w.deinit();
        try std.testing.expect(only_w.dx == null and only_w.dweight != null);
    }
}

test "exec context cross entropy ex handles ignored labels and validation" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var logits = try ctx.fromSliceRank(2, .{ 3, 5 }, &.{
        1,  2, 3,  4, 5,
        -1, 0, 1,  2, 3,
        2,  2, -2, 0, 1,
    });
    defer logits.deinit();

    // Every position ignored (in-range ignore_index): loss 0, grads exactly 0.
    // (Deliberate divergence from PyTorch's NaN.)
    const all_ignored = CrossEntropyOptions{ .ignore_index = 2, .reduction = .mean };
    var loss = try ctx.crossEntropyLossExAxisRank(2, &logits, 1, &.{ 2, 2, 2 }, all_ignored);
    defer loss.deinit();
    try std.testing.expectEqual(@as(f32, 0), loss.item());
    var grad = try ctx.crossEntropyBackwardExAxisRank(2, &logits, 1, &.{ 2, 2, 2 }, all_ignored, 1, null);
    defer grad.deinit();
    for (grad.dataConst()) |value| try std.testing.expectEqual(@as(f32, 0), value);

    // An in-range ignore_index drops exactly the matching positions, and the
    // mean denominator counts only the remaining ones.
    var partial = try ctx.crossEntropyLossExAxisRank(2, &logits, 1, &.{ 4, 2, 0 }, all_ignored);
    defer partial.deinit();
    var row0 = try ctx.fromSliceRank(2, .{ 1, 5 }, &.{ 1, 2, 3, 4, 5 });
    defer row0.deinit();
    var row2 = try ctx.fromSliceRank(2, .{ 1, 5 }, &.{ 2, 2, -2, 0, 1 });
    defer row2.deinit();
    var loss0 = try ctx.crossEntropyLossExAxisRank(2, &row0, 1, &.{4}, .{ .reduction = .sum });
    defer loss0.deinit();
    var loss2 = try ctx.crossEntropyLossExAxisRank(2, &row2, 1, &.{0}, .{ .reduction = .sum });
    defer loss2.deinit();
    try std.testing.expectApproxEqAbs((loss0.item() + loss2.item()) / 2, partial.item(), 1e-6);

    // Labels must be < class_count or == ignore_index.
    try std.testing.expectError(tensor.TensorError.IndexOutOfBounds, ctx.crossEntropyLossExAxisRank(2, &logits, 1, &.{ 0, 5, 1 }, .{}));
    try std.testing.expectError(tensor.TensorError.IndexOutOfBounds, ctx.crossEntropyBackwardExAxisRank(2, &logits, 1, &.{ 0, 9, 1 }, .{ .ignore_index = 7 }, 1, null));
    // label_smoothing must be in [0, 1).
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.crossEntropyLossExAxisRank(2, &logits, 1, &.{ 0, 1, 2 }, .{ .label_smoothing = 1 }));
    // .none requires a per-row upstream of matching length; mean/sum forbid it.
    try std.testing.expectError(tensor.TensorError.InvalidDataLength, ctx.crossEntropyBackwardExAxisRank(2, &logits, 1, &.{ 0, 1, 2 }, .{ .reduction = .none }, 1, null));
    try std.testing.expectError(tensor.TensorError.InvalidDataLength, ctx.crossEntropyBackwardExAxisRank(2, &logits, 1, &.{ 0, 1, 2 }, .{}, 1, &.{ 1, 1, 1 }));

    // Default options keep the legacy behavior: mean over all positions.
    var legacy = try ctx.crossEntropyLossAxisRank(2, &logits, 1, &.{ 4, 2, 0 });
    defer legacy.deinit();
    var ex_default = try ctx.crossEntropyLossExAxisRank(2, &logits, 1, &.{ 4, 2, 0 }, .{});
    defer ex_default.deinit();
    try std.testing.expectEqual(legacy.item(), ex_default.item());
}

test "exec context layer norm matches a naive f64 reference" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x1a7e);
    const random = prng.random();

    const axis_dims = [_]usize{ 1, 2, 3, 8, 17, 1000, 4099 };
    const outers = [_]usize{ 1, 2, 5, 64 };

    for (axis_dims) |axis_dim| {
        for (outers) |outer| {
            inline for (.{ 1, 3 }) |inner| {
                const rank = if (inner == 1) 2 else 3;
                const data = try allocator.alloc(f32, outer * axis_dim * inner);
                defer allocator.free(data);
                for (data) |*value| value.* = random.floatNorm(f32) * 3;
                const weights = try allocator.alloc(f32, axis_dim);
                defer allocator.free(weights);
                const biases = try allocator.alloc(f32, axis_dim);
                defer allocator.free(biases);
                for (weights) |*value| value.* = random.floatNorm(f32);
                for (biases) |*value| value.* = random.floatNorm(f32);

                var x = if (inner == 1)
                    try ctx.fromSliceRank(2, .{ outer, axis_dim }, data)
                else
                    try ctx.fromSliceRank(3, .{ outer, axis_dim, inner }, data);
                defer x.deinit();
                var w = try ctx.fromSliceRank(1, .{axis_dim}, weights);
                defer w.deinit();
                var b = try ctx.fromSliceRank(1, .{axis_dim}, biases);
                defer b.deinit();

                for ([_]f32{ 1e-5, 1e-6 }) |eps| {
                    const ref_plain = try testNaiveLayerNorm(allocator, data, outer, axis_dim, inner, null, null, eps);
                    defer allocator.free(ref_plain);
                    var y_plain = try ctx.layerNormAxisRank(rank, &x, 1, eps);
                    defer y_plain.deinit();
                    for (y_plain.dataConst(), ref_plain) |got, want| {
                        try expectCloseToF64(want, got, 5e-4, 5e-5);
                    }

                    const ref_affine = try testNaiveLayerNorm(allocator, data, outer, axis_dim, inner, weights, biases, eps);
                    defer allocator.free(ref_affine);
                    var y_affine = try ctx.layerNormAffineAxisRank(rank, &x, &w, &b, 1, eps);
                    defer y_affine.deinit();
                    for (y_affine.dataConst(), ref_affine) |got, want| {
                        try expectCloseToF64(want, got, 5e-4, 5e-5);
                    }
                }
            }
        }
    }
}

test "exec context layer norm backward matches a naive f64 reference and is bitwise deterministic" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x1a7f);
    const random = prng.random();

    // {outer, axis_dim, inner}: small SIMD rows, the inner>1 scalar fallback,
    // a degenerate single-element axis, and a shape big enough for the
    // parallel dx dispatch (70*2050 >= threshold/2).
    const cases = [_][3]usize{
        .{ 3, 5, 1 },
        .{ 2, 8, 3 },
        .{ 1, 1, 1 },
        .{ 5, 17, 1 },
        .{ 70, 2050, 1 },
    };
    const eps: f32 = 1e-5;

    inline for (.{ 2, 3 }) |rank| {
        for (cases) |case| {
            const outer = case[0];
            const axis_dim = case[1];
            const inner = case[2];
            if ((rank == 2) != (inner == 1)) continue;
            const data = try allocator.alloc(f32, outer * axis_dim * inner);
            defer allocator.free(data);
            const grad = try allocator.alloc(f32, outer * axis_dim * inner);
            defer allocator.free(grad);
            for (data) |*value| value.* = random.floatNorm(f32) * 2;
            for (grad) |*value| value.* = random.floatNorm(f32);
            const weights = try allocator.alloc(f32, axis_dim);
            defer allocator.free(weights);
            for (weights) |*value| value.* = random.floatNorm(f32);

            var shape: [rank]usize = undefined;
            shape[0] = outer;
            shape[1] = axis_dim;
            if (rank == 3) shape[2] = inner;
            var x = try ctx.fromSliceRank(rank, shape, data);
            defer x.deinit();
            var gy = try ctx.fromSliceRank(rank, shape, grad);
            defer gy.deinit();
            var w = try ctx.fromSliceRank(1, .{axis_dim}, weights);
            defer w.deinit();

            // Plain dx.
            var ref_plain = try testNaiveLayerNormBackward(allocator, data, grad, outer, axis_dim, inner, null, eps);
            defer ref_plain.deinit(allocator);
            var gx_plain = try ctx.layerNormBackwardAxisRank(rank, &x, &gy, 1, eps);
            defer gx_plain.deinit();
            for (gx_plain.dataConst(), ref_plain.dx) |got, want| {
                try expectCloseToF64(want, got, 5e-4, 5e-5);
            }

            // Affine dx + dweight + dbias.
            var ref_affine = try testNaiveLayerNormBackward(allocator, data, grad, outer, axis_dim, inner, weights, eps);
            defer ref_affine.deinit(allocator);
            var full = try ctx.layerNormAffineBackwardAxisRank(rank, &x, &w, &gy, 1, eps, true, true, true);
            defer full.deinit();
            for (full.input.?.dataConst(), ref_affine.dx) |got, want| {
                try expectCloseToF64(want, got, 5e-4, 5e-5);
            }
            for (full.weight.?.dataConst(), ref_affine.dweight) |got, want| {
                try expectCloseToF64(want, got, 5e-4, 5e-5);
            }
            for (full.bias.?.dataConst(), ref_affine.dbias) |got, want| {
                try expectCloseToF64(want, got, 5e-4, 5e-5);
            }

            // needs-grad pruning at the kernel level: partial runs return
            // only the requested gradients and match the full run bitwise
            // (the param pass is the same serial code either way).
            var weight_only = try ctx.layerNormAffineBackwardAxisRank(rank, &x, &w, &gy, 1, eps, false, true, false);
            defer weight_only.deinit();
            try std.testing.expect(weight_only.input == null);
            try std.testing.expect(weight_only.bias == null);
            try std.testing.expectEqualSlices(f32, full.weight.?.dataConst(), weight_only.weight.?.dataConst());

            var bias_only = try ctx.layerNormAffineBackwardAxisRank(rank, &x, &w, &gy, 1, eps, false, false, true);
            defer bias_only.deinit();
            try std.testing.expect(bias_only.input == null);
            try std.testing.expect(bias_only.weight == null);
            try std.testing.expectEqualSlices(f32, full.bias.?.dataConst(), bias_only.bias.?.dataConst());

            // Bitwise determinism across runs (the big case exercises the
            // parallel dx dispatch; dweight/dbias are one serial row pass,
            // bitwise identical for any thread count by construction).
            var again = try ctx.layerNormAffineBackwardAxisRank(rank, &x, &w, &gy, 1, eps, true, true, true);
            defer again.deinit();
            try std.testing.expectEqualSlices(f32, full.input.?.dataConst(), again.input.?.dataConst());
            try std.testing.expectEqualSlices(f32, full.weight.?.dataConst(), again.weight.?.dataConst());
            try std.testing.expectEqualSlices(f32, full.bias.?.dataConst(), again.bias.?.dataConst());

            // Forward determinism on the same shapes.
            var y_one = try ctx.layerNormAffineAxisRank(rank, &x, &w, &w, 1, eps);
            defer y_one.deinit();
            var y_two = try ctx.layerNormAffineAxisRank(rank, &x, &w, &w, 1, eps);
            defer y_two.deinit();
            try std.testing.expectEqualSlices(f32, y_one.dataConst(), y_two.dataConst());
        }
    }
}

test "exec context max min var match naive references" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x3a7c);
    const random = prng.random();

    // inner == 1 SIMD rows (axis_dim 11 exercises the scalar tail; 4099 the
    // vector body) and the inner>1 scalar layout.
    const cases = [_][3]usize{
        .{ 4, 11, 1 },
        .{ 2, 4099, 1 },
        .{ 3, 7, 2 },
        .{ 1, 1, 1 },
    };
    inline for (.{ 2, 3 }) |rank| {
        for (cases) |case| {
            const outer = case[0];
            const axis_dim = case[1];
            const inner = case[2];
            if ((rank == 2) != (inner == 1)) continue;
            const data = try allocator.alloc(f32, outer * axis_dim * inner);
            defer allocator.free(data);
            for (data) |*value| value.* = random.floatNorm(f32) * 3;

            var shape: [rank]usize = undefined;
            shape[0] = outer;
            shape[1] = axis_dim;
            if (rank == 3) shape[2] = inner;
            var x = try ctx.fromSliceRank(rank, shape, data);
            defer x.deinit();

            var max_result = try ctx.maxAxisRank(rank, &x, 1);
            defer max_result.deinit();
            var min_result = try ctx.minAxisRank(rank, &x, 1);
            defer min_result.deinit();

            for (0..outer) |outer_i| {
                const base = outer_i * axis_dim * inner;
                for (0..inner) |inner_i| {
                    var max_i: usize = 0;
                    var min_i: usize = 0;
                    var max_value = data[base + inner_i];
                    var min_value = data[base + inner_i];
                    for (1..axis_dim) |axis_i| {
                        const value = data[base + axis_i * inner + inner_i];
                        if (value > max_value) {
                            max_value = value;
                            max_i = axis_i;
                        }
                        if (value < min_value) {
                            min_value = value;
                            min_i = axis_i;
                        }
                    }
                    const flat = outer_i * inner + inner_i;
                    try std.testing.expectEqual(max_value, max_result.values.dataConst()[flat]);
                    try std.testing.expectEqual(@as(i64, @intCast(max_i)), max_result.indices.dataConst()[flat]);
                    try std.testing.expectEqual(min_value, min_result.values.dataConst()[flat]);
                    try std.testing.expectEqual(@as(i64, @intCast(min_i)), min_result.indices.dataConst()[flat]);
                }
            }

            for ([_]u1{ 0, 1 }) |ddof| {
                var v = try ctx.varAxisRank(rank, &x, 1, ddof);
                defer v.deinit();
                const n = @as(f64, @floatFromInt(axis_dim));
                for (0..outer) |outer_i| {
                    const base = outer_i * axis_dim * inner;
                    for (0..inner) |inner_i| {
                        var sum_acc: f64 = 0;
                        for (0..axis_dim) |axis_i| sum_acc += data[base + axis_i * inner + inner_i];
                        const mean_value = sum_acc / n;
                        var sumsq: f64 = 0;
                        for (0..axis_dim) |axis_i| {
                            const centered = @as(f64, data[base + axis_i * inner + inner_i]) - mean_value;
                            sumsq += centered * centered;
                        }
                        const got = v.dataConst()[outer_i * inner + inner_i];
                        if (axis_dim == 1 and ddof == 1) {
                            // torch.var on one element with Bessel: 0/0 = NaN.
                            try std.testing.expect(std.math.isNan(got));
                        } else {
                            try expectCloseToF64(sumsq / (n - @as(f64, @floatFromInt(ddof))), got, 5e-4, 5e-6);
                        }
                    }
                }
            }
        }
    }

    // Tie-break: duplicate extrema report the FIRST index on both layouts.
    var ties = try ctx.fromSliceRank(2, .{ 2, 4 }, &.{
        1, 3,  3, 2,
        5, -1, 5, 4,
    });
    defer ties.deinit();
    var tie_max = try ctx.maxAxisRank(2, &ties, 1);
    defer tie_max.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 5 }, tie_max.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, tie_max.indices.dataConst());
    var tie_min = try ctx.fromSliceRank(2, .{ 1, 5 }, &.{ 4, -2, 7, -2, 0 });
    defer tie_min.deinit();
    var tie_min_result = try ctx.minAxisRank(2, &tie_min, 1);
    defer tie_min_result.deinit();
    try std.testing.expectEqualSlices(f32, &.{-2}, tie_min_result.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{1}, tie_min_result.indices.dataConst());

    // Tie on axis 0 with inner > 1 (the scalar layout).
    var ties_inner = try ctx.fromSliceRank(2, .{ 3, 2 }, &.{
        2, 1,
        2, 5,
        0, 5,
    });
    defer ties_inner.deinit();
    var tie_axis0 = try ctx.maxAxisRank(2, &ties_inner, 0);
    defer tie_axis0.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 5 }, tie_axis0.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 0, 1 }, tie_axis0.indices.dataConst());
}

test "max/min over an axis: NaN drops and all-NaN rows degrade identically on both layouts" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // One NaN contract for both layouts (see maxAxisRank): NaN never wins —
    // regardless of position (a leading NaN used to poison the inner > 1
    // path, which seeded from input[first]); an all-NaN row degrades to
    // -inf/+inf with index 0.
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    const row_a = [_]f32{ nan, 2, -1, 7, 7, 0, 3, -5, 1, 2, 6 };
    const row_b = [_]f32{ 4, nan, nan, -2, 9, nan, 9, -8, nan, 5, -8 };
    const row_nan = [_]f32{nan} ** 11;

    // inner == 1 layout (SIMD body + scalar tail at axis_dim 11).
    var rows: [3 * 11]f32 = undefined;
    @memcpy(rows[0..11], &row_a);
    @memcpy(rows[11..22], &row_b);
    @memcpy(rows[22..33], &row_nan);
    var x = try ctx.fromSliceRank(2, .{ 3, 11 }, &rows);
    defer x.deinit();
    var mx = try ctx.maxAxisRank(2, &x, 1);
    defer mx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 7, 9, -inf }, mx.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 3, 4, 0 }, mx.indices.dataConst());
    var mn = try ctx.minAxisRank(2, &x, 1);
    defer mn.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -5, -8, inf }, mn.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 7, 7, 0 }, mn.indices.dataConst());

    // inner > 1 layout (generic strided path): the same rows as columns of a
    // {11, 3} tensor reduced over axis 0 — value and index semantics must be
    // identical to the inner == 1 results above.
    var cols: [11 * 3]f32 = undefined;
    for (0..11) |i| {
        cols[i * 3 + 0] = row_a[i];
        cols[i * 3 + 1] = row_b[i];
        cols[i * 3 + 2] = row_nan[i];
    }
    var xt = try ctx.fromSliceRank(2, .{ 11, 3 }, &cols);
    defer xt.deinit();
    var mxt = try ctx.maxAxisRank(2, &xt, 0);
    defer mxt.deinit();
    try std.testing.expectEqualSlices(f32, mx.values.dataConst(), mxt.values.dataConst());
    try std.testing.expectEqualSlices(i64, mx.indices.dataConst(), mxt.indices.dataConst());
    var mnt = try ctx.minAxisRank(2, &xt, 0);
    defer mnt.deinit();
    try std.testing.expectEqualSlices(f32, mn.values.dataConst(), mnt.values.dataConst());
    try std.testing.expectEqualSlices(i64, mn.indices.dataConst(), mnt.indices.dataConst());
}

test "argmax/topK over an axis: NaN never places, matching the max contract" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Same rows and contract as the max/min test above: a NaN never wins
    // (a leading NaN used to poison argmax, which seeded from input[first],
    // and used to land in topK slot 0, which admitted via `value <= min`);
    // an all-NaN row degrades to index 0 (argmax) / (-inf, 0) slots (topK).
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    const row_a = [_]f32{ nan, 2, -1, 7, 7, 0, 3, -5, 1, 2, 6 };
    const row_b = [_]f32{ 4, nan, nan, -2, 9, nan, 9, -8, nan, 5, -8 };
    const row_nan = [_]f32{nan} ** 11;

    var rows: [3 * 11]f32 = undefined;
    @memcpy(rows[0..11], &row_a);
    @memcpy(rows[11..22], &row_b);
    @memcpy(rows[22..33], &row_nan);
    var x = try ctx.fromSliceRank(2, .{ 3, 11 }, &rows);
    defer x.deinit();

    var arg = try ctx.argmaxAxisRank(2, &x, 1);
    defer arg.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 3, 4, 0 }, arg.dataConst());

    var top = try ctx.topKAxisRank(2, &x, 1, 3);
    defer top.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 7, 7, 6, 9, 9, 5, -inf, -inf, -inf }, top.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 3, 4, 10, 4, 6, 9, 0, 0, 0 }, top.indices.dataConst());

    // The same rows as columns of a {11, 3} tensor reduced over axis 0:
    // identical winners in the strided layout.
    var cols: [11 * 3]f32 = undefined;
    for (0..11) |i| {
        cols[i * 3 + 0] = row_a[i];
        cols[i * 3 + 1] = row_b[i];
        cols[i * 3 + 2] = row_nan[i];
    }
    var xt = try ctx.fromSliceRank(2, .{ 11, 3 }, &cols);
    defer xt.deinit();

    var argt = try ctx.argmaxAxisRank(2, &xt, 0);
    defer argt.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 3, 4, 0 }, argt.dataConst());

    var topt = try ctx.topKAxisRank(2, &xt, 0, 3);
    defer topt.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 7, 9, -inf, 7, 9, -inf, 6, 5, -inf }, topt.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 3, 4, 0, 4, 6, 0, 10, 9, 0 }, topt.indices.dataConst());
}

test "exec context softmax fast path matches generic layout and naive reference" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x50f7);
    const random = prng.random();

    // Big enough to take the parallel dispatch (rows*cols >= threshold/2),
    // and a small shape for the inline single-task body.
    const cases = [_][2]usize{ .{ 70, 2050 }, .{ 3, 5 } };
    for (cases) |case| {
        const rows = case[0];
        const cols = case[1];
        const data = try allocator.alloc(f32, rows * cols);
        defer allocator.free(data);
        for (data) |*value| value.* = random.floatNorm(f32) * 3;

        var x = try ctx.fromSliceRank(2, .{ rows, cols }, data);
        defer x.deinit();
        var y = try ctx.softmaxAxisRank(2, &x, 1);
        defer y.deinit();
        const yd = y.dataConst();

        for (0..rows) |row| {
            const expected = try testNaiveSoftmaxRow(allocator, data[row * cols ..][0..cols]);
            defer allocator.free(expected);
            for (expected, yd[row * cols ..][0..cols]) |want, got| {
                try expectCloseToF64(want, got, 1e-5, 1e-12);
            }
        }

        // Same data transposed with axis 0 forces inner > 1, i.e. the scalar
        // generic path; the SIMD fast path must agree with it.
        const transposed = try allocator.alloc(f32, rows * cols);
        defer allocator.free(transposed);
        for (0..rows) |row| {
            for (0..cols) |col| transposed[col * rows + row] = data[row * cols + col];
        }
        var xt = try ctx.fromSliceRank(2, .{ cols, rows }, transposed);
        defer xt.deinit();
        var yt = try ctx.softmaxAxisRank(2, &xt, 0);
        defer yt.deinit();
        const ytd = yt.dataConst();
        for (0..rows) |row| {
            for (0..cols) |col| {
                try expectCloseToF64(ytd[col * rows + row], yd[row * cols + col], 1e-5, 1e-12);
            }
        }
    }
}

test "softmax NaN logits poison the row on both SIMD and scalar paths" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const nan = std.math.nan(f32);
    const cols: usize = 11; // odd: exercises both vector and scalar tails
    var data: [2 * cols]f32 = undefined;
    for (&data, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 7)) * 0.5 - 1;
    data[3] = nan; // row 0 carries a NaN logit; row 1 stays clean

    // Last-axis softmax: contiguous rows, the SIMD vexpf path.
    var x = try ctx.fromSliceRank(2, .{ 2, cols }, &data);
    defer x.deinit();
    var y = try ctx.softmaxAxisRank(2, &x, 1);
    defer y.deinit();
    const yd = y.dataConst();

    // Same data transposed, softmax along axis 0: inner > 1, the strided
    // scalar path. Both paths must agree on NaN poisoning.
    var transposed: [2 * cols]f32 = undefined;
    for (0..2) |row| {
        for (0..cols) |col| transposed[col * 2 + row] = data[row * cols + col];
    }
    var xt = try ctx.fromSliceRank(2, .{ cols, 2 }, &transposed);
    defer xt.deinit();
    var yt = try ctx.softmaxAxisRank(2, &xt, 0);
    defer yt.deinit();
    const ytd = yt.dataConst();

    for (0..cols) |col| {
        // The NaN row poisons every output on both paths.
        try std.testing.expect(std.math.isNan(yd[col]));
        try std.testing.expect(std.math.isNan(ytd[col * 2]));
        // The clean row stays finite and matches across paths.
        try std.testing.expect(!std.math.isNan(yd[cols + col]));
        try std.testing.expectApproxEqAbs(yd[cols + col], ytd[col * 2 + 1], 1e-6);
    }
}

test "exec context softmax backward fast path matches generic layout" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x50f8);
    const random = prng.random();

    const rows: usize = 70;
    const cols: usize = 2050;
    const y_data = try allocator.alloc(f32, rows * cols);
    defer allocator.free(y_data);
    const gy_data = try allocator.alloc(f32, rows * cols);
    defer allocator.free(gy_data);
    for (y_data) |*value| value.* = random.float(f32);
    for (gy_data) |*value| value.* = random.floatNorm(f32);
    // Normalize the rows like real softmax outputs so the row dot is O(1).
    for (0..rows) |row| {
        var row_sum: f32 = 0;
        for (y_data[row * cols ..][0..cols]) |value| row_sum += value;
        for (y_data[row * cols ..][0..cols]) |*value| value.* /= row_sum;
    }

    var y = try ctx.fromSliceRank(2, .{ rows, cols }, y_data);
    defer y.deinit();
    var gy = try ctx.fromSliceRank(2, .{ rows, cols }, gy_data);
    defer gy.deinit();
    var gx = try ctx.softmaxExtBackwardAxisRank(2, &y, &gy, 1, 0.5);
    defer gx.deinit();
    const gxd = gx.dataConst();

    const yt_data = try allocator.alloc(f32, rows * cols);
    defer allocator.free(yt_data);
    const gyt_data = try allocator.alloc(f32, rows * cols);
    defer allocator.free(gyt_data);
    for (0..rows) |row| {
        for (0..cols) |col| {
            yt_data[col * rows + row] = y_data[row * cols + col];
            gyt_data[col * rows + row] = gy_data[row * cols + col];
        }
    }
    var yt = try ctx.fromSliceRank(2, .{ cols, rows }, yt_data);
    defer yt.deinit();
    var gyt = try ctx.fromSliceRank(2, .{ cols, rows }, gyt_data);
    defer gyt.deinit();
    var gxt = try ctx.softmaxExtBackwardAxisRank(2, &yt, &gyt, 0, 0.5);
    defer gxt.deinit();
    const gxtd = gxt.dataConst();

    for (0..rows) |row| {
        for (0..cols) |col| {
            try expectCloseToF64(gxtd[col * rows + row], gxd[row * cols + col], 1e-4, 1e-6);
        }
    }
}

test "exec context softmaxExt SIMD rows match strided scalar rows across option combos" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x50f9);
    const random = prng.random();

    // {heads, q, src} with the softmax on src: inner == 1 -> SIMD rows (and
    // big enough for the parallel dispatch). The permuted layout
    // {heads, src, q} with the softmax on axis 1 has inner == q -> the scalar
    // per-row body. Both must agree under mask + scale + causal offset +
    // ALiBi + sinks all at once.
    const heads: usize = 4;
    const q_dim: usize = 64;
    const src_dim: usize = 600;
    const source_offset = src_dim - q_dim;

    const data = try allocator.alloc(f32, heads * q_dim * src_dim);
    defer allocator.free(data);
    const mask_data = try allocator.alloc(f32, heads * q_dim * src_dim);
    defer allocator.free(mask_data);
    for (data) |*value| value.* = random.floatNorm(f32) * 2;
    for (mask_data) |*value| value.* = random.floatNorm(f32);
    const sinks = [_]f32{ 0.3, -0.2, 0.8, 0.1 };

    var x = try ctx.fromSliceRank(3, .{ heads, q_dim, src_dim }, data);
    defer x.deinit();
    var mask = try ctx.fromSliceRank(3, .{ heads, q_dim, src_dim }, mask_data);
    defer mask.deinit();
    var y = try ctx.softmaxExtAxisRank(3, &x, 2, .{
        .mask = &mask,
        .sinks = &sinks,
        .scale = 0.5,
        .max_bias = 8,
        .head_axis = 0,
        .causal_query_axis = 1,
        .causal_source_offset = source_offset,
    });
    defer y.deinit();
    const yd = y.dataConst();

    const data_t = try allocator.alloc(f32, heads * q_dim * src_dim);
    defer allocator.free(data_t);
    const mask_t = try allocator.alloc(f32, heads * q_dim * src_dim);
    defer allocator.free(mask_t);
    for (0..heads) |h| {
        for (0..q_dim) |qq| {
            for (0..src_dim) |s| {
                data_t[(h * src_dim + s) * q_dim + qq] = data[(h * q_dim + qq) * src_dim + s];
                mask_t[(h * src_dim + s) * q_dim + qq] = mask_data[(h * q_dim + qq) * src_dim + s];
            }
        }
    }
    var xt = try ctx.fromSliceRank(3, .{ heads, src_dim, q_dim }, data_t);
    defer xt.deinit();
    var maskt = try ctx.fromSliceRank(3, .{ heads, src_dim, q_dim }, mask_t);
    defer maskt.deinit();
    var yt = try ctx.softmaxExtAxisRank(3, &xt, 1, .{
        .mask = &maskt,
        .sinks = &sinks,
        .scale = 0.5,
        .max_bias = 8,
        .head_axis = 0,
        .causal_query_axis = 2,
        .causal_source_offset = source_offset,
    });
    defer yt.deinit();
    const ytd = yt.dataConst();

    for (0..heads) |h| {
        for (0..q_dim) |qq| {
            for (0..src_dim) |s| {
                const fast = yd[(h * q_dim + qq) * src_dim + s];
                const scalar_path = ytd[(h * src_dim + s) * q_dim + qq];
                try expectCloseToF64(scalar_path, fast, 1e-4, 1e-9);
            }
        }
    }
}

test "exec context softmaxExt mask broadcast along the softmax axis takes the scalar rows path" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(3, .{ 2, 4, 6 }, &.{
        0.1,  -0.4, 0.7,  1.2,  -0.9, 0.3,
        -1.1, 0.5,  0.2,  -0.6, 1.4,  0.8,
        0.9,  -0.2, -1.3, 0.4,  0.6,  -0.7,
        1.0,  0.0,  -0.5, 0.2,  -1.2, 0.5,
        -0.3, 0.8,  1.1,  -0.4, 0.2,  -0.8,
        0.6,  -1.0, 0.3,  0.9,  -0.1, 0.4,
        -0.7, 0.2,  0.5,  -1.4, 0.8,  0.1,
        0.3,  1.2,  -0.6, 0.7,  -0.2, -0.9,
    });
    defer x.deinit();

    // The mask has dim 1 on the softmax axis: broadcast gives it stride 0
    // there, so the rows are not SIMD-eligible and fall back to the scalar
    // body. The same mask materialized to full shape takes the SIMD body —
    // the two must agree.
    const mask_rows = [_]f32{ 0.5, -0.5, 0, -1, 1, 0.25, -0.25, 2 };
    var mask_thin = try ctx.fromSliceRank(3, .{ 2, 4, 1 }, &mask_rows);
    defer mask_thin.deinit();
    var mask_full_data: [2 * 4 * 6]f32 = undefined;
    for (0..8) |row| {
        for (0..6) |col| mask_full_data[row * 6 + col] = mask_rows[row];
    }
    var mask_full = try ctx.fromSliceRank(3, .{ 2, 4, 6 }, &mask_full_data);
    defer mask_full.deinit();

    var y_thin = try ctx.softmaxExtAxisRank(3, &x, 2, .{ .mask = &mask_thin, .scale = 0.7 });
    defer y_thin.deinit();
    var y_full = try ctx.softmaxExtAxisRank(3, &x, 2, .{ .mask = &mask_full, .scale = 0.7 });
    defer y_full.deinit();

    for (y_thin.dataConst(), y_full.dataConst()) |scalar_path, fast| {
        try expectCloseToF64(scalar_path, fast, 1e-5, 1e-9);
    }
}

test "scatter add axis0 parallel path matches serial reference bitwise" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // 4096x96 source + 1024-row grad = 491520 elements of work: above the
    // parallel threshold, so the pool path runs (duplicates included).
    const rows = 4096;
    const row_len = 96;
    const index_count = 1024;

    var prng = std.Random.DefaultPrng.init(0x5ca77e);
    const random = prng.random();
    const grad_data = try allocator.alloc(f32, index_count * row_len);
    defer allocator.free(grad_data);
    for (grad_data) |*value| value.* = random.floatNorm(f32);
    const indices = try allocator.alloc(usize, index_count);
    defer allocator.free(indices);
    // Heavy duplicates: half the indices land in a 13-row band.
    for (indices, 0..) |*index, i| {
        index.* = if (i % 2 == 0) random.uintLessThan(usize, rows) else 100 + random.uintLessThan(usize, 13);
    }

    var grad = try ctx.fromSliceRank(2, .{ index_count, row_len }, grad_data);
    defer grad.deinit();

    const expected = try allocator.alloc(f32, rows * row_len);
    defer allocator.free(expected);
    scatterAddAxis0Reference(expected, grad_data, row_len, indices);

    var out = try ctx.scatterAddAxisRank(2, &grad, .{ rows, row_len }, 0, indices);
    defer out.deinit();
    try std.testing.expectEqualSlices(f32, expected, out.dataConst());

    // Determinism: a second run is bitwise identical.
    var out2 = try ctx.scatterAddAxisRank(2, &grad, .{ rows, row_len }, 0, indices);
    defer out2.deinit();
    try std.testing.expectEqualSlices(f32, out.dataConst(), out2.dataConst());
}

test "scatter add axis0 single repeated index and more indices than rows" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0xd0011ca7e);
    const random = prng.random();

    // Every index hits the same destination row (worst-case duplicate skew on
    // the parallel path: one task accumulates everything, the rest only zero).
    {
        const rows = 2048;
        const row_len = 192;
        const index_count = 2048;
        const grad_data = try allocator.alloc(f32, index_count * row_len);
        defer allocator.free(grad_data);
        for (grad_data) |*value| value.* = random.floatNorm(f32);
        const indices = try allocator.alloc(usize, index_count);
        defer allocator.free(indices);
        @memset(indices, 7);

        var grad = try ctx.fromSliceRank(2, .{ index_count, row_len }, grad_data);
        defer grad.deinit();
        const expected = try allocator.alloc(f32, rows * row_len);
        defer allocator.free(expected);
        scatterAddAxis0Reference(expected, grad_data, row_len, indices);

        var out = try ctx.scatterAddAxisRank(2, &grad, .{ rows, row_len }, 0, indices);
        defer out.deinit();
        try std.testing.expectEqualSlices(f32, expected, out.dataConst());
    }

    // indices.len far above the source row count: every row is hit many times.
    {
        const rows = 8;
        const row_len = 64;
        const index_count = 4096;
        const grad_data = try allocator.alloc(f32, index_count * row_len);
        defer allocator.free(grad_data);
        for (grad_data) |*value| value.* = random.floatNorm(f32);
        const indices = try allocator.alloc(usize, index_count);
        defer allocator.free(indices);
        for (indices) |*index| index.* = random.uintLessThan(usize, rows);

        var grad = try ctx.fromSliceRank(2, .{ index_count, row_len }, grad_data);
        defer grad.deinit();
        const expected = try allocator.alloc(f32, rows * row_len);
        defer allocator.free(expected);
        scatterAddAxis0Reference(expected, grad_data, row_len, indices);

        var out = try ctx.scatterAddAxisRank(2, &grad, .{ rows, row_len }, 0, indices);
        defer out.deinit();
        try std.testing.expectEqualSlices(f32, expected, out.dataConst());
    }
}

test "scatter add axis0 parallel threshold boundary is bitwise seamless" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // row_len == 1 makes total work = rows + indices.len exactly, so the three
    // index counts straddle the threshold: -1 stays serial, 0/+1 go parallel.
    const rows = 131072;
    const base_index_count = parallel.vector_elementwise_len_threshold - rows;

    var prng = std.Random.DefaultPrng.init(0xb0a2d);
    const random = prng.random();

    for ([_]usize{ base_index_count - 1, base_index_count, base_index_count + 1 }) |index_count| {
        const grad_data = try allocator.alloc(f32, index_count);
        defer allocator.free(grad_data);
        for (grad_data) |*value| value.* = random.floatNorm(f32);
        const indices = try allocator.alloc(usize, index_count);
        defer allocator.free(indices);
        for (indices) |*index| index.* = random.uintLessThan(usize, rows);

        var grad = try ctx.fromSliceRank(2, .{ index_count, 1 }, grad_data);
        defer grad.deinit();
        const expected = try allocator.alloc(f32, rows);
        defer allocator.free(expected);
        scatterAddAxis0Reference(expected, grad_data, 1, indices);

        var out = try ctx.scatterAddAxisRank(2, &grad, .{ rows, 1 }, 0, indices);
        defer out.deinit();
        try std.testing.expectEqualSlices(f32, expected, out.dataConst());
    }
}

test "scatter add rank3 axis0 parallel and axis1 generic paths" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0xa715);
    const random = prng.random();

    // Rank 3, axis 0, above threshold: rows are (32*256)-element planes.
    {
        const rows = 64;
        const row_len = 32 * 256;
        const index_count = 96;
        const grad_data = try allocator.alloc(f32, index_count * row_len);
        defer allocator.free(grad_data);
        for (grad_data) |*value| value.* = random.floatNorm(f32);
        const indices = try allocator.alloc(usize, index_count);
        defer allocator.free(indices);
        for (indices) |*index| index.* = random.uintLessThan(usize, rows);

        var grad = try ctx.fromSliceRank(3, .{ index_count, 32, 256 }, grad_data);
        defer grad.deinit();
        const expected = try allocator.alloc(f32, rows * row_len);
        defer allocator.free(expected);
        scatterAddAxis0Reference(expected, grad_data, row_len, indices);

        var out = try ctx.scatterAddAxisRank(3, &grad, .{ rows, 32, 256 }, 0, indices);
        defer out.deinit();
        try std.testing.expectEqualSlices(f32, expected, out.dataConst());
    }

    // axis != 0 keeps the generic strided path (small reference check).
    {
        var grad = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 10, 20, 30 });
        defer grad.deinit();
        var out = try ctx.scatterAddAxisRank(2, &grad, .{ 2, 2 }, 1, &.{ 1, 0, 1 });
        defer out.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 2, 4, 20, 40 }, out.dataConst());
    }
}

test "exec dropout applies the counter-based mask with exact inverted scaling" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const len = 4096;
    const seed: u64 = 0xd20b0a7;
    var prng = std.Random.DefaultPrng.init(0xfeed);
    const random = prng.random();
    const data = try allocator.alloc(f32, len);
    defer allocator.free(data);
    for (data) |*value| value.* = random.floatNorm(f32) + 0.5;

    var x = try ctx.fromSliceRank(2, .{ 64, 64 }, data);
    defer x.deinit();

    for ([_]f32{ 0.25, 0.5 }) |p| {
        const scale = 1.0 / (1.0 - p);
        var y = try ctx.dropoutForward(&x, p, seed);
        defer y.deinit();
        var kept: usize = 0;
        for (y.dataConst(), data, 0..) |out_value, in_value, i| {
            if (dropoutKeeps(seed, i, p)) {
                try std.testing.expectEqual(in_value * scale, out_value);
                kept += 1;
            } else {
                try std.testing.expectEqual(@as(f32, 0), out_value);
            }
        }
        // Drop rate is ~p (loose tolerance at this size; the tight check runs
        // on the large parallel tensor below).
        const keep_rate = @as(f64, @floatFromInt(kept)) / len;
        try std.testing.expectApproxEqAbs(1.0 - @as(f64, p), keep_rate, 0.05);

        // Same seed -> bitwise identical; different seed -> different mask.
        var y2 = try ctx.dropoutForward(&x, p, seed);
        defer y2.deinit();
        try std.testing.expectEqualSlices(f32, y.dataConst(), y2.dataConst());
        var y3 = try ctx.dropoutForward(&x, p, seed + 1);
        defer y3.deinit();
        try std.testing.expect(!std.mem.eql(f32, y.dataConst(), y3.dataConst()));

        // The backward kernel applies the identical mask/scale to gy.
        var gy = try ctx.dropoutBackward(&x, p, seed);
        defer gy.deinit();
        try std.testing.expectEqualSlices(f32, y.dataConst(), gy.dataConst());
    }

    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.dropoutForward(&x, 1.0, seed));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.dropoutForward(&x, 1.5, seed));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.dropoutForward(&x, -0.1, seed));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.dropoutForward(&x, std.math.nan(f32), seed));
}

test "exec dropout parallel path is bitwise identical across the threshold" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const par_len = parallel.vector_elementwise_len_threshold; // pool path
    const ser_len = par_len - 1; // serial path
    const p = 0.5;
    const seed: u64 = 0x7ead5;

    var prng = std.Random.DefaultPrng.init(0xc0ffee);
    const random = prng.random();
    const data = try allocator.alloc(f32, par_len);
    defer allocator.free(data);
    for (data) |*value| value.* = random.floatNorm(f32);

    var x_par = try ctx.fromSliceRank(1, .{par_len}, data);
    defer x_par.deinit();
    var x_ser = try ctx.fromSliceRank(1, .{ser_len}, data[0..ser_len]);
    defer x_ser.deinit();

    var y_par = try ctx.dropoutForward(&x_par, p, seed);
    defer y_par.deinit();
    var y_ser = try ctx.dropoutForward(&x_ser, p, seed);
    defer y_ser.deinit();

    // The mask depends only on (seed, element index), so the serial result is
    // the exact prefix of the parallel one — bitwise.
    try std.testing.expectEqualSlices(f32, y_ser.dataConst(), y_par.dataConst()[0..ser_len]);

    // Determinism: a second parallel run is bitwise identical.
    var y_par2 = try ctx.dropoutForward(&x_par, p, seed);
    defer y_par2.deinit();
    try std.testing.expectEqualSlices(f32, y_par.dataConst(), y_par2.dataConst());

    // Tight drop-rate check at this size.
    var kept: usize = 0;
    for (y_par.dataConst()) |value| {
        if (value != 0) kept += 1;
    }
    const keep_rate = @as(f64, @floatFromInt(kept)) / par_len;
    try std.testing.expectApproxEqAbs(1.0 - @as(f64, p), keep_rate, 0.005);
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
    }
}

test "exec dropout integer cutoff equals the f64 keep predicate at boundaries" {
    // The kernel compares `rng.at >> 11` against dropoutKeepCutoff(p); the
    // contract is the historical f64 predicate (see dropoutKeeps). Check the
    // exact boundary integers for a battery of p values: exact-cutoff cases
    // (t·2^53 an integer), irrational-looking cases, and the extremes.
    const p_values = [_]f32{ 0, 0.5, 0.25, 0.1, 0.3, 1.0 / 3.0, 0x1p-24, 1.0 - 0x1p-24, 0.999, 1e-7, 0.9999999 };
    for (p_values) |p| {
        const cutoff = exec_row_ops.dropoutKeepCutoff(p);
        var boundary = [_]u64{ 0, 1, 0, 0, 0, (1 << 53) - 1 };
        boundary[2] = cutoff -| 1;
        boundary[3] = cutoff;
        boundary[4] = @min(cutoff + 1, (1 << 53) - 1);
        for (boundary) |k| {
            const uniform = @as(f64, @floatFromInt(k)) * 0x1.0p-53;
            const keeps_f64 = uniform < 1.0 - @as(f64, p);
            try std.testing.expectEqual(keeps_f64, k < cutoff);
        }
    }
}

// --- conv2d: hand-computed cases; run under native and -Dbackend=scalar ---

test "conv2d: pointwise 1x1 (groups=1) with bias" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // input [H=2, W=1, Cin=2]: h0=[1,2], h1=[3,4]
    var input = try ctx.fromSliceRank(3, .{ 2, 1, 2 }, &[_]f32{ 1, 2, 3, 4 });
    defer input.deinit();
    // weight [Cout=1, KH=1, KW=1, Cinpg=2] = [5,6]
    var weight = try ctx.fromSliceRank(4, .{ 1, 1, 1, 2 }, &[_]f32{ 5, 6 });
    defer weight.deinit();
    var bias = try ctx.fromSliceRank(1, .{1}, &[_]f32{100});
    defer bias.deinit();

    var out = try ctx.conv2d(&input, &weight, &bias, .{ 1, 1 }, .{ 0, 0 }, 1);
    defer out.deinit();
    // out[h] = in[h,0]*5 + in[h,1]*6 + 100
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1 * 5 + 2 * 6 + 100, 3 * 5 + 4 * 6 + 100 }, out.dataConst());
}

test "conv2d: 3x3 same-padding (stride 1, pad 1), ones kernel on ones image" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // input [3,3,1] all ones; weight [1,3,3,1] all ones; pad 1 -> output 3x3.
    var input = try ctx.fromSliceRank(3, .{ 3, 3, 1 }, &[_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1 });
    defer input.deinit();
    var weight = try ctx.fromSliceRank(4, .{ 1, 3, 3, 1 }, &[_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1 });
    defer weight.deinit();

    var out = try ctx.conv2d(&input, &weight, null, .{ 1, 1 }, .{ 1, 1 }, 1);
    defer out.deinit();
    // neighborhood sums with zero pad: corners 4, edges 6, center 9.
    try std.testing.expectEqualSlices(f32, &[_]f32{ 4, 6, 4, 6, 9, 6, 4, 6, 4 }, out.dataConst());
}

test "conv2d: depthwise (groups=Cin) with stride 2" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // input [H=4, W=1, Cin=2], interleaved h*2+c: ch0=[1,2,3,4], ch1=[10,20,30,40]
    var input = try ctx.fromSliceRank(3, .{ 4, 1, 2 }, &[_]f32{ 1, 10, 2, 20, 3, 30, 4, 40 });
    defer input.deinit();
    // weight [Cout=2, KH=2, KW=1, Cinpg=1] all ones; oc0->ch0, oc1->ch1 (depthwise).
    var weight = try ctx.fromSliceRank(4, .{ 2, 2, 1, 1 }, &[_]f32{ 1, 1, 1, 1 });
    defer weight.deinit();

    var out = try ctx.conv2d(&input, &weight, null, .{ 2, 1 }, .{ 0, 0 }, 2);
    defer out.deinit();
    // OH=(4-2)/2+1=2. oh0: ch0=1+2=3, ch1=10+20=30 ; oh1: ch0=3+4=7, ch1=30+40=70.
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3, 30, 7, 70 }, out.dataConst());
}

test "conv2d: rejects mismatched groups / shapes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var input = try ctx.fromSliceRank(3, .{ 2, 1, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();
    var weight = try ctx.fromSliceRank(4, .{ 1, 1, 1, 2 }, &[_]f32{ 1, 1 }); // Cinpg=2 != Cin/groups=3
    defer weight.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv2d(&input, &weight, null, .{ 1, 1 }, .{ 0, 0 }, 1));
}

test "conv2d prepared winograd weights: exec-level parity, .empty fallback, shape mismatch" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Pin the Winograd route ON (BLAS builds default it off).
    const exec_conv = @import("exec/conv.zig");
    exec_conv.setWinogradForTest(true);
    defer exec_conv.setWinogradForTest(null);

    // 8x8x8 -> F2-eligible 3x3 s1 p1 conv.
    const h = 8;
    const w = 8;
    const cin = 8;
    const cout = 4;
    const xd = try allocator.alloc(f32, h * w * cin);
    defer allocator.free(xd);
    const wd = try allocator.alloc(f32, cout * 9 * cin);
    defer allocator.free(wd);
    const bd = try allocator.alloc(f32, cout);
    defer allocator.free(bd);
    rng.gaussianFill(7, xd, 1.0);
    rng.gaussianFill(8, wd, 0.5);
    rng.gaussianFill(9, bd, 0.5);

    var input = try ctx.fromSliceRank(3, .{ h, w, cin }, xd);
    defer input.deinit();
    var weight = try ctx.fromSliceRank(4, .{ cout, 3, 3, cin }, wd);
    defer weight.deinit();
    var bias = try ctx.fromSliceRank(1, .{cout}, bd);
    defer bias.deinit();

    var ref = try ctx.conv2d(&input, &weight, &bias, .{ 1, 1 }, .{ 1, 1 }, 1);
    defer ref.deinit();

    var prep = try ctx.prepareConv2dWeights(&weight);
    defer prep.deinit();
    try std.testing.expect(prep.f2 != null);
    var got = try ctx.conv2dPrepared(&input, &weight, &prep, &bias, .{ 1, 1 }, .{ 1, 1 }, 1);
    defer got.deinit();
    try std.testing.expectEqualSlices(f32, ref.dataConst(), got.dataConst());

    // Fused-relu entry against the same planes.
    var ref_relu = try ctx.conv2dRelu(&input, &weight, &bias, .{ 1, 1 }, .{ 1, 1 }, 1);
    defer ref_relu.deinit();
    var got_relu = try ctx.conv2dPreparedRelu(&input, &weight, &prep, &bias, .{ 1, 1 }, .{ 1, 1 }, 1);
    defer got_relu.deinit();
    try std.testing.expectEqualSlices(f32, ref_relu.dataConst(), got_relu.dataConst());

    // `.empty` is inert: conv2dPreparedExt falls back to the per-call
    // transform and produces the same bytes.
    var empty = ExecContext.PreparedConvWeights.empty;
    defer empty.deinit();
    var got_empty = try exec_conv.conv2dPreparedExt(&ctx.rt, &input, &weight, &empty, &bias, .{ 1, 1 }, .{ 1, 1 }, 1, false);
    defer got_empty.deinit();
    try std.testing.expectEqualSlices(f32, ref.dataConst(), got_empty.dataConst());

    // Planes prepared for a DIFFERENT weight shape are rejected on the
    // Winograd route (caller wiring bug, not a fallback case).
    const wd2 = try allocator.alloc(f32, cout * 9 * cin * 2);
    defer allocator.free(wd2);
    rng.gaussianFill(10, wd2, 0.5);
    var weight2 = try ctx.fromSliceRank(4, .{ cout, 3, 3, cin * 2 }, wd2);
    defer weight2.deinit();
    var prep2 = try ctx.prepareConv2dWeights(&weight2);
    defer prep2.deinit();
    try std.testing.expect(prep2.f2 != null and prep2.cin == cin * 2);
    try std.testing.expectError(
        tensor.TensorError.ShapeMismatch,
        ctx.conv2dPrepared(&input, &weight, &prep2, &bias, .{ 1, 1 }, .{ 1, 1 }, 1),
    );
}

// Moved from exec.zig: these drive the Runtime alloc primitives (rt.zerosRank,
// rt.scalarTyped) and the elementwise reduce-broadcast VJP directly, so they
// belong beside the other exec_tests rather than inline in the exec.zig facade.
test "exec context reuses buffers for arbitrary broadcast materialization" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.rt.zerosRank(3, .{ 2, 4, 3 });
    defer x.deinit();
    var middle = try ctx.fromSliceRank(3, .{ 2, 1, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer middle.deinit();
    var middle_b = try ctx.broadcastToRank(3, &middle, .{ 2, 4, 3 });
    defer middle_b.deinit();

    try std.testing.expectEqual(LayoutClass.arbitrary, ctx.classify(&middle_b));
    try std.testing.expectEqual(@as(usize, 2), ctx.rt.buffers.outstandingBuffers());
    try std.testing.expectEqual(@as(usize, 0), ctx.rt.buffers.cachedBuffers());

    var first = try ctx.addRank(3, &x, &middle_b);
    try std.testing.expectEqualSlices(f32, &.{
        1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
        4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
    }, first.dataConst());
    try std.testing.expectEqual(@as(usize, 3), ctx.rt.buffers.outstandingBuffers());
    try std.testing.expect(ctx.rt.buffers.cachedBuffers() >= 1);
    first.deinit();

    try std.testing.expectEqual(@as(usize, 2), ctx.rt.buffers.outstandingBuffers());
    const cached_after_first = ctx.rt.buffers.cachedBuffers();
    try std.testing.expect(cached_after_first >= 2);

    var second = try ctx.addRank(3, &x, &middle_b);
    second.deinit();

    try std.testing.expectEqual(@as(usize, 2), ctx.rt.buffers.outstandingBuffers());
    try std.testing.expectEqual(cached_after_first, ctx.rt.buffers.cachedBuffers());
}

test "exec context reduces higher-rank and scalar broadcast gradients" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var gy = try ctx.fromSlice(&.{ 2, 2, 3 }, &.{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    });
    defer gy.deinit();

    var tail = try exec_elementwise.reduceBroadcastRank(&ctx.rt, 1, &gy, .{3});
    defer tail.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 22, 26, 30 }, tail.dataConst());

    var exact = try exec_elementwise.reduceBroadcastRank(&ctx.rt, 3, &gy, .{ 2, 2, 3 });
    defer exact.deinit();
    try std.testing.expectEqualSlices(f32, gy.dataConst(), exact.dataConst());
    try std.testing.expectEqual(@as(usize, 2), ctx.rt.buffers.outstandingBuffers());

    var scalar_reduced = try ctx.reduceBroadcast(&gy, &.{1});
    defer scalar_reduced.deinit();
    try std.testing.expectEqual(@as(f32, 78), scalar_reduced.item());

    var singleton_middle = try exec_elementwise.reduceBroadcastRank(&ctx.rt, 3, &gy, .{ 2, 1, 3 });
    defer singleton_middle.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9, 17, 19, 21 }, singleton_middle.dataConst());

    var singleton_prefix = try exec_elementwise.reduceBroadcastRank(&ctx.rt, 2, &gy, .{ 1, 3 });
    defer singleton_prefix.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 22, 26, 30 }, singleton_prefix.dataConst());

    try std.testing.expectError(tensor.TensorError.ShapeMismatch, exec_elementwise.reduceBroadcastRank(&ctx.rt, 2, &gy, .{ 2, 2 }));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.reduceBroadcast(&gy, &.{ 0, 3 }));
}

test "exec context allocates typed non-f32 tensors without using f32 kernels" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var ids = try ctx.fromSliceRankTyped(.u16, 2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ids.deinit();
    try std.testing.expect(@TypeOf(ids).dtype == .u16);
    try std.testing.expectEqualSlices(u16, &.{ 1, 2, 3, 4, 5, 6 }, ids.dataConst());

    var flags = try ctx.onesTyped(.bool, &.{3});
    defer flags.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, flags.dataConst());

    var scalar_id = try ctx.rt.scalarTyped(.i64, 42);
    defer scalar_id.deinit();
    try std.testing.expectEqual(@as(i64, 42), scalar_id.item());
}

// --- conv1d / col2im1d / convTranspose1d / snake / groupNorm / elu / gelu_erf
// (omnivoice op set): hand-computed cases + shape-error rejection; run under
// native and -Dbackend=scalar ---

test "conv1d: hand-computed same-pad, stride+dilation, and grouped cases" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // k=3, s=1, p=1, d=1: x=[1,2,3,4], w=[1,2,3].
    // PyTorch: F.conv1d(x, w, padding=1) = [8, 14, 20, 11].
    var x = try ctx.fromSliceRank(2, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w = try ctx.fromSliceRank(3, .{ 3, 1, 1 }, &.{ 1, 2, 3 });
    defer w.deinit();
    var y = try ctx.conv1dAxisRank(2, &x, &w, 0, 1, 1, 1, 1, 1);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 8, 14, 20, 11 }, y.dataConst());

    // k=2, s=2, p=0, d=2: T_out = (5 - 2 - 1)/2 + 1 = 2; y[t] = 10*x[2t] + x[2t+2].
    // PyTorch: F.conv1d([1..5], [10,1], stride=2, dilation=2) = [13, 35].
    var xs = try ctx.fromSliceRank(2, .{ 5, 1 }, &.{ 1, 2, 3, 4, 5 });
    defer xs.deinit();
    var ws = try ctx.fromSliceRank(3, .{ 2, 1, 1 }, &.{ 10, 1 });
    defer ws.deinit();
    var ys = try ctx.conv1dAxisRank(2, &xs, &ws, 0, 1, 2, 0, 2, 1);
    defer ys.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 13, 35 }, ys.dataConst());

    // groups=2, k=1: in=[a,b] rows, w [1, 1, 2] = [[10, 100]] so
    // y[t] = [10*a, 100*b] (each output channel sees only its group's input).
    var xg = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer xg.deinit();
    var wg = try ctx.fromSliceRank(3, .{ 1, 1, 2 }, &.{ 10, 100 });
    defer wg.deinit();
    var yg = try ctx.conv1dAxisRank(2, &xg, &wg, 0, 1, 1, 0, 1, 2);
    defer yg.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 200, 30, 400 }, yg.dataConst());

    // Channel mixing, no pad: in=2 channels, out=1, k=2.
    // y[t] = 1*x[t,0] + 2*x[t,1] + 3*x[t+1,0] + 4*x[t+1,1]
    // (tap k reads xpad[t*s + k*d], so k=0 is the OLDEST sample — non-causal
    // orientation, unlike causalConv1d where the last tap is newest).
    var xm = try ctx.fromSliceRank(2, .{ 3, 2 }, &.{ 1, 10, 2, 20, 3, 30 });
    defer xm.deinit();
    var wm = try ctx.fromSliceRank(3, .{ 2, 2, 1 }, &.{ 1, 2, 3, 4 });
    defer wm.deinit();
    var ym = try ctx.conv1dAxisRank(2, &xm, &wm, 0, 1, 1, 0, 1, 1);
    defer ym.deinit();
    // t=0: 1*1 + 2*10 + 3*2 + 4*20 = 107 ; t=1: 1*2 + 2*20 + 3*3 + 4*30 = 171.
    try std.testing.expectEqualSlices(f32, &.{ 107, 171 }, ym.dataConst());
}

test "conv1d: rejects invalid geometry and mismatched shapes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 4, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer x.deinit();
    var w = try ctx.fromSliceRank(3, .{ 2, 2, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer w.deinit();

    // stride/dilation must be positive.
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.conv1dAxisRank(2, &x, &w, 0, 1, 0, 0, 1, 1));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.conv1dAxisRank(2, &x, &w, 0, 1, 1, 0, 0, 1));
    // groups must divide in and out channels.
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv1dAxisRank(2, &x, &w, 0, 1, 1, 0, 1, 3));
    // weight in_per_group mismatch: groups=2 wants weight.shape[1] == 1.
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv1dAxisRank(2, &x, &w, 0, 1, 1, 0, 1, 2));
    // Receptive field longer than the padded input: T + 2p < d*(K-1)+1.
    var wide = try ctx.fromSliceRank(3, .{ 6, 2, 2 }, &([_]f32{0.5} ** 24));
    defer wide.deinit();
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.conv1dAxisRank(2, &x, &wide, 0, 1, 1, 0, 1, 1));
}

test "col2im1d: hand-computed gather with crop and output_pad" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // t_in=2, taps=2, oc=1, stride=2: col rows [10,20],[30,40] (column = k).
    var col = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 10, 20, 30, 40 });
    defer col.deinit();

    // pad=0: T_out = 4, disjoint scatter.
    var y = try ctx.col2im1dAxisRank(&col, 1, 2, 2, 0, 0);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30, 40 }, y.dataConst());

    // pad=1: crops one frame each side -> [20, 30]; output_pad=1 appends a zero row.
    var yc = try ctx.col2im1dAxisRank(&col, 1, 2, 2, 1, 1);
    defer yc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 20, 30, 0 }, yc.dataConst());

    // Rejects col width != taps*out_channels and degenerate T_out.
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.col2im1dAxisRank(&col, 2, 2, 2, 0, 0));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.col2im1dAxisRank(&col, 1, 2, 1, 2, 0));
}

test "convTranspose1d: matches a naive direct scatter reference (DAC combos)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // (stride, taps, pad, output_pad): two of the DAC decoder combos plus a
    // small overlapping case.
    const combos = [_][4]usize{
        .{ 2, 4, 1, 0 },
        .{ 3, 6, 2, 1 },
        .{ 1, 3, 1, 0 },
    };
    for (combos) |combo| {
        const stride = combo[0];
        const taps = combo[1];
        const pad = combo[2];
        const output_pad = combo[3];
        const t_in: usize = 5;
        const in_channels: usize = 3;
        const out_channels: usize = 2;
        const t_out = (t_in - 1) * stride + taps - 2 * pad;
        const out_len = t_out + output_pad;

        var x = try ctx.rt.zerosRank(2, .{ t_in, in_channels });
        defer x.deinit();
        for (x.data()) |*v| v.* = random.float(f32) * 2 - 1;
        // weight2[(oc*K + k)*IC + ic] — the reference's load-time repack.
        var w2 = try ctx.rt.zerosRank(2, .{ taps * out_channels, in_channels });
        defer w2.deinit();
        for (w2.data()) |*v| v.* = random.float(f32) * 2 - 1;
        var bias = try ctx.rt.zerosRank(1, .{out_channels});
        defer bias.deinit();
        for (bias.data()) |*v| v.* = random.float(f32);

        var got = try ctx.convTranspose1d(&x, &w2, &bias, out_channels, taps, stride, pad, output_pad);
        defer got.deinit();
        try std.testing.expectEqualSlices(usize, &.{ out_len, out_channels }, got.shape.slice());

        // Naive direct ConvTranspose1d: every input frame scatters its taps to
        // t_in*stride + k - pad, dropping positions outside [0, T_out); the
        // output_pad rows stay at bias.
        const want = try allocator.alloc(f32, out_len * out_channels);
        defer allocator.free(want);
        for (0..out_len) |t| {
            for (0..out_channels) |oc| want[t * out_channels + oc] = bias.dataConst()[oc];
        }
        const xd = x.dataConst();
        const wd = w2.dataConst();
        for (0..t_in) |ti| {
            for (0..taps) |k| {
                const pos = ti * stride + k;
                if (pos < pad) continue;
                const t = pos - pad;
                if (t >= t_out) continue;
                for (0..out_channels) |oc| {
                    var acc: f32 = 0;
                    for (0..in_channels) |ic| {
                        acc += xd[ti * in_channels + ic] * wd[(oc * taps + k) * in_channels + ic];
                    }
                    want[t * out_channels + oc] += acc;
                }
            }
        }
        for (want, got.dataConst()) |wv, gv| {
            try std.testing.expectApproxEqAbs(wv, gv, 1e-5);
        }
    }
}

test "convTranspose1d: rejects mismatched weight2 and bias shapes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w2 = try ctx.fromSliceRank(2, .{ 4, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer w2.deinit();

    // weight2 rows must be taps*out_channels (here 2*1=2 != 4 for taps=2, oc=1).
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.convTranspose1d(&x, &w2, null, 1, 2, 2, 0, 0));
    // bias length must equal out_channels.
    var bad_bias = try ctx.fromSliceRank(1, .{3}, &.{ 1, 2, 3 });
    defer bad_bias.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.convTranspose1d(&x, &w2, &bad_bias, 2, 2, 2, 0, 0));
    // stride 0 rejected.
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.convTranspose1d(&x, &w2, null, 2, 2, 0, 0, 0));
}

test "snake: hand-computed per-channel activation + shape rejection" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 0.5, -1.0, 2.0, 0.0 });
    defer x.deinit();
    var alpha = try ctx.fromSliceRank(1, .{2}, &.{ 1.0, 2.0 });
    defer alpha.deinit();
    var inv_b = try ctx.fromSliceRank(1, .{2}, &.{ 1.0, 0.5 });
    defer inv_b.deinit();

    var y = try ctx.snakeRows(&x, &alpha, &inv_b);
    defer y.deinit();

    // y[t,c] = x + inv_b[c]*sin(alpha[c]*x)^2:
    //   sin(0.5)^2 = 0.2298488, sin(-2)^2 = 0.8268218,
    //   sin(2)^2 = 0.8268218, sin(0) = 0.
    const expected = [_]f32{
        0.5 + 0.2298488,
        -1.0 + 0.5 * 0.8268218,
        2.0 + 0.8268218,
        0.0,
    };
    for (expected, y.dataConst()) |w, g| {
        try std.testing.expectApproxEqAbs(w, g, 1e-6);
    }

    var short_alpha = try ctx.fromSliceRank(1, .{1}, &.{1.0});
    defer short_alpha.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.snakeRows(&x, &short_alpha, &inv_b));
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.snakeRows(&x, &alpha, &short_alpha));
}

test "groupNorm: hand-computed G=1, G=C, and affine cases + rejection" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const eps: f32 = 1e-5;

    // G=1: one group over ALL T*C elements [1,2,3,4]: mean 2.5, biased var 1.25.
    var x = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var y1 = try ctx.groupNormAxisRank(&x, 1, eps, null, null);
    defer y1.deinit();
    const inv1 = 1.0 / @sqrt(@as(f32, 1.25) + eps);
    const want1 = [_]f32{ -1.5 * inv1, -0.5 * inv1, 0.5 * inv1, 1.5 * inv1 };
    for (want1, y1.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);

    // G=C: per-channel over time (InstanceNorm over T; the HuBERT layer-0
    // configuration). col0 = {1,3}: mean 2, var 1; col1 = {2,4}: mean 3, var 1.
    var y2 = try ctx.groupNormAxisRank(&x, 2, eps, null, null);
    defer y2.deinit();
    const inv2 = 1.0 / @sqrt(@as(f32, 1.0) + eps);
    const want2 = [_]f32{ -inv2, -inv2, inv2, inv2 };
    for (want2, y2.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);

    // Affine applied AFTER normalization: y*w + b.
    var wt = try ctx.fromSliceRank(1, .{2}, &.{ 2.0, 3.0 });
    defer wt.deinit();
    var bt = try ctx.fromSliceRank(1, .{2}, &.{ 10.0, 20.0 });
    defer bt.deinit();
    var y3 = try ctx.groupNormAxisRank(&x, 2, eps, &wt, &bt);
    defer y3.deinit();
    const want3 = [_]f32{ -inv2 * 2 + 10, -inv2 * 3 + 20, inv2 * 2 + 10, inv2 * 3 + 20 };
    for (want3, y3.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);

    // groups must divide C; affine vectors must be [C].
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.groupNormAxisRank(&x, 3, eps, null, null));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.groupNormAxisRank(&x, 0, eps, null, null));
    var bad_w = try ctx.fromSliceRank(1, .{3}, &.{ 1, 2, 3 });
    defer bad_w.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.groupNormAxisRank(&x, 2, eps, &bad_w, null));
}

// NOTE: this test stays EXACT (expectEqualSlices) across the direct-gather
// and GEMM+im2col/col2im backward routes: every operand is an integer-valued
// f32 and every reduction is a sum of integer products well below 2^24, so
// f32 accumulation is exact in ANY association order. Keep the values
// integer if you edit the cases — fractional values would make the
// route-dependent summation order observable.
test "conv2d backward: hand-computed input/weight gradients (channel-last)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Case A: 3x3x1 input, weight [cout1,kh2,kw2,cin1]=[[1,0],[0,2]], gy=ones 2x2, s1 p0 g1.
    var x = try ctx.fromSliceRank(3, .{ 3, 3, 1 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    defer x.deinit();
    var w = try ctx.fromSliceRank(4, .{ 1, 2, 2, 1 }, &.{ 1, 0, 0, 2 });
    defer w.deinit();
    var gy = try ctx.fromSliceRank(3, .{ 2, 2, 1 }, &.{ 1, 1, 1, 1 });
    defer gy.deinit();
    var gx = try ctx.conv2dBackwardInput(&gy, &w, 3, 3, .{ 1, 1 }, .{ 0, 0 }, 1);
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 0, 1, 3, 2, 0, 2, 2 }, gx.dataConst());
    var gw = try ctx.conv2dBackwardWeight(&x, &gy, 2, 2, .{ 1, 1 }, .{ 0, 0 }, 1);
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 12, 16, 24, 28 }, gw.dataConst());

    // Case B: stride 2. 4x4x1 input [1..16], weight ones 2x2, gy=ones 2x2, s2 p0 g1.
    // Every input pixel is read by exactly one output window -> gx all ones;
    // gw[kh,kw] = sum of in[2oh+kh, 2ow+kw].
    var x2 = try ctx.fromSliceRank(3, .{ 4, 4, 1 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    defer x2.deinit();
    var w2 = try ctx.fromSliceRank(4, .{ 1, 2, 2, 1 }, &.{ 1, 1, 1, 1 });
    defer w2.deinit();
    var gy2 = try ctx.fromSliceRank(3, .{ 2, 2, 1 }, &.{ 1, 1, 1, 1 });
    defer gy2.deinit();
    var gx2 = try ctx.conv2dBackwardInput(&gy2, &w2, 4, 4, .{ 2, 2 }, .{ 0, 0 }, 1);
    defer gx2.deinit();
    try std.testing.expectEqualSlices(f32, &([_]f32{1} ** 16), gx2.dataConst());
    var gw2 = try ctx.conv2dBackwardWeight(&x2, &gy2, 2, 2, .{ 2, 2 }, .{ 0, 0 }, 1);
    defer gw2.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 24, 28, 40, 44 }, gw2.dataConst());

    // Case C: depthwise 1x1 (groups=cin=cout=2). in [1,1,2]=[3,5], w[c]=[2,4], gy=[1,1].
    // gx[c] = gy[c]*w[c] = [2,4]; gw[c] = gy[c]*in[c] = [3,5].
    var x3 = try ctx.fromSliceRank(3, .{ 1, 1, 2 }, &.{ 3, 5 });
    defer x3.deinit();
    var w3 = try ctx.fromSliceRank(4, .{ 2, 1, 1, 1 }, &.{ 2, 4 });
    defer w3.deinit();
    var gy3 = try ctx.fromSliceRank(3, .{ 1, 1, 2 }, &.{ 1, 1 });
    defer gy3.deinit();
    var gx3 = try ctx.conv2dBackwardInput(&gy3, &w3, 1, 1, .{ 1, 1 }, .{ 0, 0 }, 2);
    defer gx3.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4 }, gx3.dataConst());
    var gw3 = try ctx.conv2dBackwardWeight(&x3, &gy3, 1, 1, .{ 1, 1 }, .{ 0, 0 }, 2);
    defer gw3.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 5 }, gw3.dataConst());
}

test "conv2d backward GEMM routes match the direct gather kernels" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Pin the GEMM route ON so every groups == 1 case below exercises it
    // (the depthwise case stays on the direct kernel either way).
    const exec_conv = @import("exec/conv.zig");
    exec_conv.setConvBwdGemmForTest(true);
    defer exec_conv.setConvBwdGemmForTest(null);

    const vector_conv = @import("backend/vector/conv.zig");

    const Case = struct {
        h: usize,
        w: usize,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
        groups: usize,
    };
    const cases = [_]Case{
        .{ .h = 6, .w = 5, .cin = 8, .cout = 4, .k = 1, .stride = 1, .pad = 0, .groups = 1 }, // 1x1 s1 p0 plain-GEMM route
        .{ .h = 7, .w = 5, .cin = 6, .cout = 3, .k = 1, .stride = 2, .pad = 0, .groups = 1 }, // 1x1 s2: general GEMM route
        .{ .h = 6, .w = 6, .cin = 5, .cout = 4, .k = 2, .stride = 1, .pad = 0, .groups = 1 },
        .{ .h = 8, .w = 7, .cin = 4, .cout = 6, .k = 3, .stride = 1, .pad = 1, .groups = 1 },
        .{ .h = 9, .w = 8, .cin = 4, .cout = 5, .k = 3, .stride = 2, .pad = 1, .groups = 1 },
        .{ .h = 9, .w = 9, .cin = 3, .cout = 4, .k = 5, .stride = 1, .pad = 2, .groups = 1 },
        .{ .h = 8, .w = 8, .cin = 6, .cout = 6, .k = 3, .stride = 1, .pad = 1, .groups = 6 }, // depthwise: direct route
    };

    for (cases, 0..) |case, case_i| {
        const cin_pg = case.cin / case.groups;
        const oh = (case.h + 2 * case.pad - case.k) / case.stride + 1;
        const ow = (case.w + 2 * case.pad - case.k) / case.stride + 1;

        var input = try ctx.emptyRank(3, .{ case.h, case.w, case.cin });
        defer input.deinit();
        rng.gaussianFill(100 + case_i, input.data(), 1.0);
        var weight = try ctx.emptyRank(4, .{ case.cout, case.k, case.k, cin_pg });
        defer weight.deinit();
        rng.gaussianFill(200 + case_i, weight.data(), 1.0);
        var gy = try ctx.emptyRank(3, .{ oh, ow, case.cout });
        defer gy.deinit();
        rng.gaussianFill(300 + case_i, gy.data(), 1.0);

        const d: backend_mod.Conv2dDims = .{
            .h = case.h,
            .w = case.w,
            .cin = case.cin,
            .oh = oh,
            .ow = ow,
            .cout = case.cout,
            .kh = case.k,
            .kw = case.k,
            .stride_h = case.stride,
            .stride_w = case.stride,
            .pad_h = case.pad,
            .pad_w = case.pad,
            .groups = case.groups,
        };

        // Reference: the config-less direct gather cores on the raw tensors.
        var gx_ref = try ctx.emptyRank(3, .{ case.h, case.w, case.cin });
        defer gx_ref.deinit();
        vector_conv.conv2dBackwardInputInto(&gx_ref, &gy, &weight, d);
        var gw_ref = try ctx.emptyRank(4, .{ case.cout, case.k, case.k, cin_pg });
        defer gw_ref.deinit();
        vector_conv.conv2dBackwardWeightInto(&gw_ref, &input, &gy, d);

        var gx = try ctx.conv2dBackwardInput(&gy, &weight, case.h, case.w, .{ case.stride, case.stride }, .{ case.pad, case.pad }, case.groups);
        defer gx.deinit();
        for (gx_ref.dataConst(), gx.dataConst()) |expected, got| {
            const tol = 1e-4 * @max(1.0, @abs(expected));
            try std.testing.expectApproxEqAbs(expected, got, tol);
        }

        var gw = try ctx.conv2dBackwardWeight(&input, &gy, case.k, case.k, .{ case.stride, case.stride }, .{ case.pad, case.pad }, case.groups);
        defer gw.deinit();
        for (gw_ref.dataConst(), gw.dataConst()) |expected, got| {
            const tol = 1e-4 * @max(1.0, @abs(expected));
            try std.testing.expectApproxEqAbs(expected, got, tol);
        }
    }
}

test "conv1d backward: hand-computed input/weight gradients + geometry rejection" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Adjoint of the k=3, s=1, p=1 forward hand case with gy = ones:
    // gx[ti] = sum of the taps that touch x[ti], gw[k] = sum of the x rows
    // tap k reads.
    var x = try ctx.fromSliceRank(2, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var w = try ctx.fromSliceRank(3, .{ 3, 1, 1 }, &.{ 1, 2, 3 });
    defer w.deinit();
    var gy = try ctx.fromSliceRank(2, .{ 4, 1 }, &.{ 1, 1, 1, 1 });
    defer gy.deinit();

    var gx = try ctx.conv1dBackwardInputAxisRank(2, &gy, &w, 0, 1, 4, 1, 1, 1, 1);
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 6, 6, 5 }, gx.dataConst());

    var gw = try ctx.conv1dBackwardWeightAxisRank(2, &x, &gy, 0, 1, 3, 1, 1, 1, 1);
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 10, 9 }, gw.dataConst());

    // Adjoint of the k=2, s=2, d=2 case (y[t] = 10*x[2t] + x[2t+2], out_len 2).
    var xs = try ctx.fromSliceRank(2, .{ 5, 1 }, &.{ 1, 2, 3, 4, 5 });
    defer xs.deinit();
    var ws = try ctx.fromSliceRank(3, .{ 2, 1, 1 }, &.{ 10, 1 });
    defer ws.deinit();
    var gys = try ctx.fromSliceRank(2, .{ 2, 1 }, &.{ 1, 1 });
    defer gys.deinit();
    var gxs = try ctx.conv1dBackwardInputAxisRank(2, &gys, &ws, 0, 1, 5, 2, 0, 2, 1);
    defer gxs.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 0, 11, 0, 1 }, gxs.dataConst());
    var gws = try ctx.conv1dBackwardWeightAxisRank(2, &xs, &gys, 0, 1, 2, 2, 0, 2, 1);
    defer gws.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 4, 8 }, gws.dataConst());

    // seq inconsistent with gy's out_len is rejected (both directions).
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv1dBackwardInputAxisRank(2, &gy, &w, 0, 1, 5, 1, 1, 1, 1));
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv1dBackwardWeightAxisRank(2, &xs, &gy, 0, 1, 3, 1, 1, 1, 1));
    // Degenerate geometry is rejected.
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.conv1dBackwardInputAxisRank(2, &gy, &w, 0, 1, 4, 0, 1, 1, 1));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.conv1dBackwardWeightAxisRank(2, &x, &gy, 0, 1, 3, 1, 1, 0, 1));
    // groups must divide the channel counts.
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.conv1dBackwardInputAxisRank(2, &gy, &w, 0, 1, 4, 1, 1, 1, 3));
}

test "col2im1d backward: hand-computed gather transpose + rejects short gy" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Adjoint of the s=2, k=2, pad=0 forward case: gcol[ti, k] = gy[2*ti + k].
    var gy = try ctx.fromSliceRank(2, .{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer gy.deinit();
    var gcol = try ctx.col2im1dBackwardAxisRank(&gy, 2, 1, 2, 2, 0);
    defer gcol.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, gcol.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, gcol.dataConst());

    // pad=1, output_pad=1 (t_conv = 2, gy has 3 rows): cropped taps and the
    // output_pad row read back as zero.
    var gy_pad = try ctx.fromSliceRank(2, .{ 3, 1 }, &.{ 5, 6, 7 });
    defer gy_pad.deinit();
    var gcol_pad = try ctx.col2im1dBackwardAxisRank(&gy_pad, 2, 1, 2, 2, 1);
    defer gcol_pad.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 5, 6, 0 }, gcol_pad.dataConst());

    // gy channel count must match; gy must cover t_conv rows; stride > 0.
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.col2im1dBackwardAxisRank(&gy, 2, 2, 2, 2, 0));
    var short_gy = try ctx.fromSliceRank(2, .{ 3, 1 }, &.{ 1, 2, 3 });
    defer short_gy.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.col2im1dBackwardAxisRank(&short_gy, 2, 1, 2, 2, 0));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.col2im1dBackwardAxisRank(&gy, 2, 1, 2, 0, 0));
}

test "snake backward: hand-computed gradients + shape rejection" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const x_vals = [_]f32{ 0.5, -1.0, 2.0, 0.25 };
    const gy_vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const a_vals = [_]f32{ 1.0, 2.0 };
    const ib_vals = [_]f32{ 1.0, 0.5 };

    var x = try ctx.fromSliceRank(2, .{ 2, 2 }, &x_vals);
    defer x.deinit();
    var gy = try ctx.fromSliceRank(2, .{ 2, 2 }, &gy_vals);
    defer gy.deinit();
    var alpha = try ctx.fromSliceRank(1, .{2}, &a_vals);
    defer alpha.deinit();
    var inv_b = try ctx.fromSliceRank(1, .{2}, &ib_vals);
    defer inv_b.deinit();

    var gx = try ctx.snakeRowsBackwardInput(&x, &gy, &alpha, &inv_b);
    defer gx.deinit();
    var params = try ctx.snakeRowsBackwardParams(&x, &gy, &alpha, &inv_b);
    defer params.deinit();

    // gx = gy*(1 + ib*a*sin(2ax)); ga = Σ gy*ib*x*sin(2ax); gib = Σ gy*sin²(ax).
    var want_gx: [4]f32 = undefined;
    var want_ga = [_]f32{ 0, 0 };
    var want_gib = [_]f32{ 0, 0 };
    for (0..2) |t| {
        for (0..2) |c| {
            const v = x_vals[t * 2 + c];
            const g = gy_vals[t * 2 + c];
            const s = @sin(a_vals[c] * v);
            const s2 = @sin(2 * a_vals[c] * v);
            want_gx[t * 2 + c] = g * (1 + ib_vals[c] * a_vals[c] * s2);
            want_ga[c] += g * ib_vals[c] * v * s2;
            want_gib[c] += g * s * s;
        }
    }
    for (want_gx, gx.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);
    for (want_ga, params.alpha.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);
    for (want_gib, params.inv_b.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);

    // gy must match x; the channel vectors must be [C].
    var bad_gy = try ctx.fromSliceRank(2, .{ 1, 2 }, &.{ 1, 2 });
    defer bad_gy.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.snakeRowsBackwardInput(&x, &bad_gy, &alpha, &inv_b));
    var short_alpha = try ctx.fromSliceRank(1, .{1}, &.{1.0});
    defer short_alpha.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.snakeRowsBackwardInput(&x, &gy, &short_alpha, &inv_b));
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.snakeRowsBackwardParams(&x, &gy, &alpha, &short_alpha));
}

test "groupNorm backward: hand-computed G=C case + rejection" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // G=C over time: col0 = {1,3} (mean 2), col1 = {2,4} (mean 3), biased
    // var 1 each; eps=0.25 keeps the dx terms far from cancellation.
    // s = 1/sqrt(1.25); x̂ = {−s, s} per column.
    // gy = {1,0 ; 0,2}: col0 ĝ={1,0} → mean_ĝ=0.5, mean(ĝx̂)=−s/2;
    // col1 ĝ={0,2} → mean_ĝ=1, mean(ĝx̂)=s.
    const eps: f32 = 0.25;
    const s: f32 = 1.0 / @sqrt(@as(f32, 1.0) + eps);
    var x = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var gy = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 1, 0, 0, 2 });
    defer gy.deinit();

    var result = try ctx.groupNormBackwardAxisRank(&x, &gy, 2, eps, null, true, true, true);
    defer result.deinit();

    const want_dx = [_]f32{
        s * (0.5 - s * s / 2.0),  s * (-1.0 + s * s),
        s * (-0.5 + s * s / 2.0), s * (1.0 - s * s),
    };
    for (want_dx, result.input.?.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);
    const want_dw = [_]f32{ -s, 2 * s };
    for (want_dw, result.weight.?.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), result.bias.?.dataConst()[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), result.bias.?.dataConst()[1], 1e-6);

    // Only the requested gradients are returned.
    var dx_only = try ctx.groupNormBackwardAxisRank(&x, &gy, 2, eps, null, true, false, false);
    defer dx_only.deinit();
    try std.testing.expect(dx_only.input != null);
    try std.testing.expect(dx_only.weight == null);
    try std.testing.expect(dx_only.bias == null);
    try std.testing.expectEqualSlices(f32, result.input.?.dataConst(), dx_only.input.?.dataConst());

    // gy must match x; groups must divide C; the affine weight must be [C].
    var bad_gy = try ctx.fromSliceRank(2, .{ 1, 2 }, &.{ 1, 2 });
    defer bad_gy.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.groupNormBackwardAxisRank(&x, &bad_gy, 2, eps, null, true, false, false));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.groupNormBackwardAxisRank(&x, &gy, 3, eps, null, true, false, false));
    var bad_w = try ctx.fromSliceRank(1, .{3}, &.{ 1, 2, 3 });
    defer bad_w.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.groupNormBackwardAxisRank(&x, &gy, 2, eps, &bad_w, true, false, false));
}

test "exec context applies elu and gelu_erf unary ops" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(1, .{4}, &.{ -1.0, 0.0, 1.0, 2.0 });
    defer x.deinit();

    var elu_y = try ctx.unary(.elu, &x);
    defer elu_y.deinit();
    const elu_want = [_]f32{ -0.6321206, 0.0, 1.0, 2.0 };
    for (elu_want, elu_y.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);

    var gelu_y = try ctx.unary(.gelu_erf, &x);
    defer gelu_y.deinit();
    const gelu_want = [_]f32{ -0.15865526, 0.0, 0.8413447, 1.9544997 };
    for (gelu_want, gelu_y.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);
}

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

fn f16BitsFromF32(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

// Deterministic valid-domain Q5_K expert stack for the batched-MoE tests:
// `rows` stacked expert columns of `k_dim` features, same block pattern as
// the q5_k kernel-test fixtures, offset by `seed` so gate/up/down differ.
fn buildTestMoeRhsQ5K(allocator: Allocator, rows: usize, k_dim: usize, seed: usize) !exec.ExecContext.MoeRhs {
    const qm = backend_mod.quantized_matmul;
    const bpc = k_dim / qm.qk_k_block_size;
    const blocks = try allocator.alloc(qm.BlockQ5_K, rows * bpc);
    defer allocator.free(blocks);
    for (blocks, 0..) |*b, block_i| {
        const bi = block_i + seed;
        b.dm[0] = f16BitsFromF32(0.05 + 0.001 * @as(f32, @floatFromInt(bi % 7)));
        b.dm[1] = f16BitsFromF32(0.02);
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }
    return .{ .q5_k = try qm.quantizedMatmulRhsQ5_KFromBlocks(allocator, k_dim, rows, blocks) };
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

// --- comparison/logical masks, cumsum, pad, sort ----------------------------

test "exec compare and compareScalar produce IEEE 0/1 masks (NaN false except ne)" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const nan = std.math.nan(f32);
    // torch: torch.eq/ne/lt/le/gt/ge(a, b).float() with a NaN lane — every
    // comparison involving NaN is false except ne, which is true.
    var a = try ctx.fromSliceRank(1, .{4}, &.{ 1, 2, nan, -3 });
    defer a.deinit();
    var b = try ctx.fromSliceRank(1, .{4}, &.{ 1, 5, nan, -4 });
    defer b.deinit();

    var eq = try ctx.compare(.eq, &a, &b);
    defer eq.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, eq.dataConst());
    var ne = try ctx.compare(.ne, &a, &b);
    defer ne.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, ne.dataConst());
    var lt = try ctx.compare(.lt, &a, &b);
    defer lt.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, lt.dataConst());
    var le = try ctx.compare(.le, &a, &b);
    defer le.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, le.dataConst());
    var gt = try ctx.compare(.gt, &a, &b);
    defer gt.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, gt.dataConst());
    var ge = try ctx.compare(.ge, &a, &b);
    defer ge.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, ge.dataConst());

    // Scalar RHS: x > 1.5 -> {0, 1, 0, 0}; x != 1.5 with the NaN lane true.
    var gt_s = try ctx.compareScalar(.gt, &a, 1.5);
    defer gt_s.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, gt_s.dataConst());
    var ne_s = try ctx.compareScalar(.ne, &a, 1.5);
    defer ne_s.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, ne_s.dataConst());
    var eq_s = try ctx.compareScalar(.eq, &a, nan);
    defer eq_s.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, eq_s.dataConst());

    // Same-shape contract, like where.
    var short = try ctx.fromSliceRank(1, .{2}, &.{ 1, 2 });
    defer short.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.compare(.eq, &a, &short));
}

test "exec logical ops use != 0 truthiness with 0/1 outputs" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // Nonzero (incl. negatives and NaN) is true; 0 is false.
    const nan = std.math.nan(f32);
    var a = try ctx.fromSliceRank(1, .{4}, &.{ 0, 2, 0, -3 });
    defer a.deinit();
    var b = try ctx.fromSliceRank(1, .{4}, &.{ 0, 0, nan, 5 });
    defer b.deinit();

    var land = try ctx.logicalTyped(.l_and, .f32, .f32, &a, &b);
    defer land.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, land.dataConst());
    var lor = try ctx.logicalTyped(.l_or, .f32, .f32, &a, &b);
    defer lor.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lor.dataConst());
    var lxor = try ctx.logicalTyped(.l_xor, .f32, .f32, &a, &b);
    defer lxor.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, lxor.dataConst());
    var lnot = try ctx.logicalNotTyped(.f32, &a);
    defer lnot.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, lnot.dataConst());

    // Same-shape contract, like where.
    var short = try ctx.fromSliceRank(1, .{2}, &.{ 1, 0 });
    defer short.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.logicalTyped(.l_and, .f32, .f32, &a, &short));
}

test "exec cumsum forward and reverse match torch.cumsum along both axes" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    // torch.cumsum(x, dim=1): rows {1,3,6} and {4,9,15}.
    var last = try ctx.cumsumAxisRank(2, &x, 1);
    defer last.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 6, 4, 9, 15 }, last.dataConst());

    // torch.cumsum(x, dim=0): {1,2,3} then {5,7,9} (inner > 1 layout).
    var first = try ctx.cumsumAxisRank(2, &x, 0);
    defer first.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 5, 7, 9 }, first.dataConst());

    // Reverse (suffix) sums — the cumsum VJP: torch.cumsum(x.flip(1), 1).flip(1).
    var rev = try ctx.cumsumReverseAxisRank(2, &x, 1);
    defer rev.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 5, 3, 15, 11, 6 }, rev.dataConst());
    var rev0 = try ctx.cumsumReverseAxisRank(2, &x, 0);
    defer rev0.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9, 4, 5, 6 }, rev0.dataConst());
}

test "exec pad places the body at offset before and fills the rest" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var x = try ctx.fromSliceRank(2, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    // torch F.pad(x, (1, 2), value=9): last axis grows 2 -> 5.
    var last = try ctx.padAxisRank(2, &x, 1, 1, 2, 9);
    defer last.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, last.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 9, 1, 2, 9, 9, 9, 3, 4, 9, 9 }, last.dataConst());

    // torch F.pad(x, (0, 0, 2, 1), value=0): first axis 2 -> 5.
    var first = try ctx.padAxisRank(2, &x, 0, 2, 1, 0);
    defer first.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 5, 2 }, first.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 1, 2, 3, 4, 0, 0 }, first.dataConst());

    // before == after == 0 is an identity copy.
    var same = try ctx.padAxisRank(2, &x, 1, 0, 0, 7);
    defer same.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, same.dataConst());
}

test "exec sort orders rows both directions with NaN last and exact indices" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // torch.sort(x, dim=1): values {{1,3,4,7},{-2,0,5,8}},
    // indices {{2,1,0,3},{1,3,2,0}} (all values distinct, no tie ambiguity).
    var x = try ctx.fromSliceRank(2, .{ 2, 4 }, &.{ 4, 3, 1, 7, 8, -2, 5, 0 });
    defer x.deinit();
    var asc = try ctx.sortAxisRank(2, &x, 1, false);
    defer asc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 4, 7, -2, 0, 5, 8 }, asc.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 0, 3, 1, 3, 2, 0 }, asc.indices.dataConst());

    // torch.sort(x, dim=1, descending=True).
    var desc = try ctx.sortAxisRank(2, &x, 1, true);
    defer desc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 7, 4, 3, 1, 8, 5, 0, -2 }, desc.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 3, 0, 1, 2, 0, 2, 3, 1 }, desc.indices.dataConst());

    // Axis-0 sort exercises the inner > 1 (strided) layout: columns sorted
    // independently — torch.sort(x, dim=0).
    var cols = try ctx.sortAxisRank(2, &x, 0, false);
    defer cols.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 4, -2, 1, 0, 8, 3, 5, 7 }, cols.values.dataConst());
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 1, 1, 0, 1, 0 }, cols.indices.dataConst());

    // NaN contract: NaN sorts LAST in BOTH directions (diverges from torch,
    // which puts NaN first when descending).
    const nan = std.math.nan(f32);
    var with_nan = try ctx.fromSliceRank(1, .{4}, &.{ 2, nan, 1, 3 });
    defer with_nan.deinit();
    var nan_asc = try ctx.sortAxisRank(1, &with_nan, 0, false);
    defer nan_asc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, nan_asc.values.dataConst()[0..3]);
    try std.testing.expect(std.math.isNan(nan_asc.values.dataConst()[3]));
    try std.testing.expectEqualSlices(i64, &.{ 2, 0, 3, 1 }, nan_asc.indices.dataConst());
    var nan_desc = try ctx.sortAxisRank(1, &with_nan, 0, true);
    defer nan_desc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 2, 1 }, nan_desc.values.dataConst()[0..3]);
    try std.testing.expect(std.math.isNan(nan_desc.values.dataConst()[3]));
    try std.testing.expectEqualSlices(i64, &.{ 3, 0, 2, 1 }, nan_desc.indices.dataConst());
}

test "materialize of a large permuted view goes through the chunked parallel copy" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // 768x512 transposed view: 393216 elements, above the parallel
    // materialize threshold, innermost axis strided.
    const rows: usize = 768;
    const cols: usize = 512;
    var x = try ctx.emptyRank(2, .{ rows, cols });
    defer x.deinit();
    for (x.data(), 0..) |*v, i| v.* = @floatFromInt(i % 1013);
    var t = try x.viewWithStrides(&.{ cols, rows }, &.{ 1, cols });
    defer t.deinit();

    var m = try ctx.materialize(&t);
    defer m.deinit();
    try std.testing.expect(m.isContiguous());
    try std.testing.expectEqualSlices(usize, &.{ cols, rows }, m.shape.slice());

    // Reference: the sequential range copy of the same view.
    const expected = try allocator.alloc(f32, rows * cols);
    defer allocator.free(expected);
    t.copyRangeTo(expected, 0, expected.len);
    try std.testing.expectEqualSlices(f32, expected, m.dataConst());

    // Spot-check against the analytic transpose.
    try std.testing.expectEqual(x.dataConst()[3 * cols + 7], m.dataConst()[7 * rows + 3]);
}

test "castTyped bf16 vector lanes match the scalar converters bit-for-bit" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Edge patterns: signed zeros, exact/tie/rounding mantissas, ±inf,
    // quiet + signaling NaN payloads, subnormals, max finite. 67 elements
    // force both the vector body and the scalar tail.
    const patterns = [_]u32{
        0x00000000, 0x80000000, 0x3f800000, 0x3f808000, 0x3f818000, 0x3f7fffff,
        0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001, 0xffc12345, 0x80000001,
        0x00000001, 0x807fffff, 0x40490fdb, 0x7f7fffff, 0xff7fffff, 0x33800000,
        0xc2f6e979, 0x3d800800,
    };
    var values: [67]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @bitCast(patterns[i % patterns.len]);

    var x = try ctx.fromSlice(&.{values.len}, &values);
    defer x.deinit();
    var narrowed = try ctx.castTyped(.f32, .bf16, &x);
    defer narrowed.deinit();
    for (narrowed.dataConst(), values) |got, value| {
        try std.testing.expectEqual(dtype_mod.f32ToBf16(value), got);
    }

    var widened = try ctx.castTyped(.bf16, .f32, &narrowed);
    defer widened.deinit();
    for (widened.dataConst(), narrowed.dataConst()) |got, bits| {
        // Bit compare: NaN payloads must survive, and nan != nan by value.
        try std.testing.expectEqual(@as(u32, @bitCast(dtype_mod.bf16ToF32(bits))), @as(u32, @bitCast(got)));
    }
}
