//! Matmul dispatch through ExecContext: backend outputs, transpose variants,
//! blocked-GEMM thresholds, f16/bf16 RHS routes, the cpu f32 shadow route,
//! and the IEEE float-environment gate. Force-imported by `exec.zig`.

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

test "exec context matmul uses backend into pooled output" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(.f32, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSlice(.f32, &.{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();

    var c = try ctx.matmul(.f32, &a, &b);
    defer c.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, c.dataConst());
}

test "the GEMM dispatch tier leaves the IEEE float environment untouched" {
    // The rounding mode and the flush-to-zero bit are per-thread state this
    // process shares with whatever CBLAS the build linked. A vendor kernel that
    // switches flush-to-zero on and forgets to restore it silently changes the
    // numerics of every op that follows, including the ones whose backend
    // parity tolerances and thread-count invariance are pinned by tests. This
    // is the gate for that: cross the dispatch tier at shapes that reach BLAS
    // (m, n, k >= 16 is `shouldUseBlas`) and confirm the environment survives.
    if (!fpenv.supported) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const before = fpenv.get().?;
    try std.testing.expect(before.isDefault());

    // Denormal and huge operands, so a kernel that flushes subnormals or traps
    // on overflow shows up here rather than in a distant parity failure.
    const n = 64;
    var lhs = try ctx.zeros(.f32, &.{ n, n });
    defer lhs.deinit();
    var rhs = try ctx.zeros(.f32, &.{ n, n });
    defer rhs.deinit();
    for (lhs.data(), 0..) |*v, i| v.* = if (i % 7 == 0) 1.0e-40 else @floatFromInt(i % 13);
    for (rhs.data(), 0..) |*v, i| v.* = if (i % 5 == 0) 1.0e-40 else @floatFromInt(i % 11);

    var nn = try ctx.matmul(.f32, &lhs, &rhs);
    defer nn.deinit();
    var nt = try ctx.matmulTransB(&lhs, &rhs);
    defer nt.deinit();
    var tn = try ctx.matmulTransA(&lhs, &rhs);
    defer tn.deinit();

    try std.testing.expectEqual(before, fpenv.get().?);
    try ctx.checkFloatEnvironment();
}

test "checkFloatEnvironment catches a changed environment" {
    if (!fpenv.supported) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    try ctx.checkFloatEnvironment();
    try std.testing.expect(ctx.floatEnvironmentAtInit().?.isDefault());

    var guard = fpenv.Guard.begin();
    defer guard.restore();
    fpenv.set(.{ .rounding = .nearest_even, .underflow = .flush_to_zero });
    try std.testing.expectError(error.FloatEnvironmentChanged, ctx.checkFloatEnvironment());

    guard.restore();
    try ctx.checkFloatEnvironment();
}

test "exec context matmul transpose variants use backend outputs" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(.f32, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSlice(.f32, &.{ 2, 3 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();
    var c = try ctx.fromSlice(.f32, &.{ 2, 2 }, &.{ 7, 8, 9, 10 });
    defer c.deinit();

    var nt = try ctx.matmulTransB(&a, &b);
    defer nt.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 50, 68, 122, 167 }, nt.dataConst());

    var tn = try ctx.matmulTransA(&a, &c);
    defer tn.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 43, 48, 59, 66, 75, 84 }, tn.dataConst());

    var column = try ctx.fromSlice(.f32, &.{ 3, 1 }, &.{ 2, 3, 4 });
    defer column.deinit();
    var gemv = try ctx.matmul(.f32, &a, &column);
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

    var b = try ctx.fromSlice(.f32, &.{ k, n }, b_data);
    defer b.deinit();

    var a_above = try ctx.fromSlice(.f32, &.{ 768, k }, a_data);
    defer a_above.deinit();
    var above = try ctx.matmul(.f32, &a_above, &b);
    defer above.deinit();
    try expectSampledMatmulParity(above.dataConst(), a_data, b_data, 768, n, k, 2e-3);

    var a_below = try ctx.fromSlice(.f32, &.{ 767, k }, a_data[0 .. 767 * k]);
    defer a_below.deinit();
    var below = try ctx.matmul(.f32, &a_below, &b);
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

    var a = try ctx.fromSlice(.f32, &.{ m, k }, a_data);
    defer a.deinit();
    var a_t = try ctx.fromSlice(.f32, &.{ k, m }, at_data);
    defer a_t.deinit();
    var b = try ctx.fromSlice(.f32, &.{ k, n }, b_data);
    defer b.deinit();
    var b_t = try ctx.fromSlice(.f32, &.{ n, k }, bt_data);
    defer b_t.deinit();

    var want = try ctx.matmul(.f32, &a, &b);
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
    var nn = try ctx.matmul(.f32, &a_view, &b);
    defer nn.deinit();
    for (want.dataConst(), nn.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 2e-3);
}

test "exec context matmul transposed f16 RHS uses backend output" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(.f32, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSlice(.f16, .{ 2, 3 }, &.{ 7, 8, 9, 10, 11, 12 });
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

    var a = try ctx.fromSlice(.f32, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSlice(.bf16, .{ 2, 3 }, &.{
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
    const exec_matmul = @import("matmul.zig");
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

    var a = try ctx.fromSlice(.f32, &.{ m, k }, &a_data);
    defer a.deinit();
    var b16 = try ctx.fromSlice(.f16, .{ n, k }, &b16_data);
    defer b16.deinit();
    var bbf = try ctx.fromSlice(.bf16, .{ n, k }, &bbf_data);
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

    // Per-context override beats the process gate: with the process gate
    // pinned OFF, a context whose Overrides force the route ON still takes
    // it (fresh weight tensor so the shadow resource is created here), and
    // a second untouched context stays on the streaming kernels — two
    // contexts in one process running different policy.
    exec_matmul.setCpuF32Shadow(false, null);
    var ctx_on: ExecContext = undefined;
    ctx_on.init(allocator);
    defer ctx_on.deinit();
    ctx_on.setTuning(.{ .cpu_f32_shadow = true, .cpu_f32_shadow_min_m = 4 });
    var b16_ovr = try ctx_on.fromSlice(.f16, .{ n, k }, &b16_data);
    defer b16_ovr.deinit();
    var a_ovr = try ctx_on.fromSlice(.f32, &.{ m, k }, &a_data);
    defer a_ovr.deinit();
    var got_ovr = try ctx_on.matmulTransB2DWithF16Rhs(&a_ovr, &b16_ovr);
    defer got_ovr.deinit();
    try std.testing.expect(b16_ovr.buffer.acceleratorResource(.cpu) != null); // shadow route ran
    for (want16.dataConst(), got_ovr.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 2e-3);

    var b16_plain = try ctx.fromSlice(.f16, .{ n, k }, &b16_data);
    defer b16_plain.deinit();
    var got_plain = try ctx.matmulTransB2DWithF16Rhs(&a, &b16_plain);
    defer got_plain.deinit();
    try std.testing.expect(b16_plain.buffer.acceleratorResource(.cpu) == null); // gate off, no shadow
}
