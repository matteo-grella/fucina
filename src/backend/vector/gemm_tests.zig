//! Behavioral tests for the dense f32/f16/bf16 GEMM vector kernels
//! (`gemm.zig`): the typed NN path over f16/bf16 row tiles + tails, and the
//! f32 x bf16-RHS / f16 x f16-RHS TransB kernels across row splits, column
//! tails, and odd k.
const std = @import("std");
const builtin = @import("builtin");
const gemm = @import("gemm.zig");
const dtype_mod = @import("../../dtype.zig");
const tensor = @import("../../tensor.zig");

const Tensor = tensor.Tensor;

const matmul2DIntoUncheckedTypedWithConfig = gemm.matmul2DIntoUncheckedTypedWithConfig;
const matmulTransB2DIntoUncheckedBf16RhsWithConfig = gemm.matmulTransB2DIntoUncheckedBf16RhsWithConfig;
const matmulTransB2DIntoUncheckedF16OperandsWithConfig = gemm.matmulTransB2DIntoUncheckedF16OperandsWithConfig;

test "f16 matmul vector kernel covers row tiles and tails" {
    const allocator = std.testing.allocator;
    const m = 21;
    const n = 17;
    const k = 13;

    var a_data: [m * k]f16 = undefined;
    var b_data: [k * n]f16 = undefined;
    for (&a_data, 0..) |*value, idx| {
        const centered: i32 = @intCast(idx % 11);
        value.* = @floatCast(@as(f32, @floatFromInt(centered - 5)) * 0.125);
    }
    for (&b_data, 0..) |*value, idx| {
        const centered: i32 = @intCast((idx * 3) % 13);
        value.* = @floatCast(@as(f32, @floatFromInt(centered - 6)) * 0.0625);
    }

    var a = try tensor.TensorOf(.f16).fromSlice(allocator, &.{ m, k }, &a_data);
    defer a.deinit();
    var b = try tensor.TensorOf(.f16).fromSlice(allocator, &.{ k, n }, &b_data);
    defer b.deinit();
    var out = try tensor.TensorOf(.f16).zeros(allocator, &.{ m, n });
    defer out.deinit();

    matmul2DIntoUncheckedTypedWithConfig(.f16, &out, &a, &b, m, n, k, .{});

    for (0..m) |i| {
        for (0..n) |j| {
            var expected: f32 = 0;
            for (0..k) |p| {
                expected += @as(f32, @floatCast(a_data[i * k + p])) * @as(f32, @floatCast(b_data[p * n + j]));
            }
            try std.testing.expectEqual(@as(f16, @floatCast(expected)), out.dataConst()[i * n + j]);
        }
    }
}

test "bf16 matmul vector kernel covers row tiles and tails" {
    const allocator = std.testing.allocator;
    const m = 21;
    const n = 17;
    const k = 13;

    var a_data: [m * k]u16 = undefined;
    var b_data: [k * n]u16 = undefined;
    for (&a_data, 0..) |*value, idx| {
        const centered: i32 = @intCast(idx % 11);
        value.* = dtype_mod.f32ToBf16(@as(f32, @floatFromInt(centered - 5)) * 0.125);
    }
    for (&b_data, 0..) |*value, idx| {
        const centered: i32 = @intCast((idx * 3) % 13);
        value.* = dtype_mod.f32ToBf16(@as(f32, @floatFromInt(centered - 6)) * 0.0625);
    }

    var a = try tensor.TensorOf(.bf16).fromSlice(allocator, &.{ m, k }, &a_data);
    defer a.deinit();
    var b = try tensor.TensorOf(.bf16).fromSlice(allocator, &.{ k, n }, &b_data);
    defer b.deinit();
    var out = try tensor.TensorOf(.bf16).zeros(allocator, &.{ m, n });
    defer out.deinit();

    matmul2DIntoUncheckedTypedWithConfig(.bf16, &out, &a, &b, m, n, k, .{});

    for (0..m) |i| {
        for (0..n) |j| {
            var expected: f32 = 0;
            for (0..k) |p| {
                expected += dtype_mod.bf16ToF32(a_data[i * k + p]) * dtype_mod.bf16ToF32(b_data[p * n + j]);
            }
            try std.testing.expectEqual(dtype_mod.f32ToBf16(expected), out.dataConst()[i * n + j]);
        }
    }
}

test "f16 x f16 RHS TransB kernel covers row tiles, column tails, and odd k" {
    const allocator = std.testing.allocator;
    // m exercises the 6/6/6/4/1 row split (incl. the dot4 + scalar single-row
    // leaves), n the dot4 column blocks + scalar column tail, k the vector
    // tail of the inner loop on both the f16-lane (aarch64) and f32-lane
    // (widened) arms.
    const m = 23;
    const n = 17;
    const k = 37;

    var a_data: [m * k]f16 = undefined;
    var b_data: [n * k]f16 = undefined;
    for (&a_data, 0..) |*value, idx| {
        const centered: i32 = @intCast(idx % 11);
        value.* = @floatCast(@as(f32, @floatFromInt(centered - 5)) * 0.125);
    }
    for (&b_data, 0..) |*value, idx| {
        const centered: i32 = @intCast((idx * 3) % 13);
        value.* = @floatCast(@as(f32, @floatFromInt(centered - 6)) * 0.0625);
    }

    var a = try tensor.TensorOf(.f16).fromSlice(allocator, &.{ m, k }, &a_data);
    defer a.deinit();
    var b = try tensor.TensorOf(.f16).fromSlice(allocator, &.{ n, k }, &b_data);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    matmulTransB2DIntoUncheckedF16OperandsWithConfig(&out, &a, &b, m, n, k, .{});

    // aarch64 accumulates in f16 (native fmla) — per-step rounding error up
    // to ~1e-1 abs at these magnitudes; the widened arms accumulate in f32
    // and track the reference to fp noise.
    const tol: f32 = if (builtin.cpu.arch.isAARCH64()) 0.25 else 1e-3;
    for (0..m) |i| {
        for (0..n) |j| {
            var expected: f32 = 0;
            for (0..k) |p| {
                expected += @as(f32, @floatCast(a_data[i * k + p])) * @as(f32, @floatCast(b_data[j * k + p]));
            }
            try std.testing.expectApproxEqAbs(expected, out.dataConst()[i * n + j], tol);
        }
    }
}

test "f32 x bf16 RHS TransB kernel covers row tiles, column tails, and odd k" {
    const allocator = std.testing.allocator;
    // m exercises the 6/4/3/2/1 row split, n the dot4 + scalar column tail,
    // k the vector tail of the widening inner loop.
    const m = 21;
    const n = 17;
    const k = 37;

    var a_data: [m * k]f32 = undefined;
    var b_data: [n * k]u16 = undefined;
    for (&a_data, 0..) |*value, idx| {
        const centered: i32 = @intCast(idx % 11);
        value.* = @as(f32, @floatFromInt(centered - 5)) * 0.125;
    }
    for (&b_data, 0..) |*value, idx| {
        const centered: i32 = @intCast((idx * 3) % 13);
        value.* = dtype_mod.f32ToBf16(@as(f32, @floatFromInt(centered - 6)) * 0.0625);
    }

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, &a_data);
    defer a.deinit();
    var b = try tensor.TensorOf(.bf16).fromSlice(allocator, &.{ n, k }, &b_data);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();

    matmulTransB2DIntoUncheckedBf16RhsWithConfig(&out, &a, &b, m, n, k, .{});

    for (0..m) |i| {
        for (0..n) |j| {
            var expected: f32 = 0;
            for (0..k) |p| {
                expected += a_data[i * k + p] * dtype_mod.bf16ToF32(b_data[j * k + p]);
            }
            // Exact small binary fractions: widening is exact and the f32
            // accumulation has no rounding for these values.
            try std.testing.expectApproxEqAbs(expected, out.dataConst()[i * n + j], 1e-5);
        }
    }
}
