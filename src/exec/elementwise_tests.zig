//! Elementwise kernels through ExecContext: dispatch, broadcast and in-place
//! paths, take-reuse, typed float math, dropout, snake, elu/gelu_erf, and
//! comparison/logical masks. Force-imported by `exec.zig`.

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
const CrossEntropyOptions = exec.CrossEntropyOptions;
const Reduction = exec.Reduction;

/// Test-side dropout mask: keep element i iff the 53-bit uniform of
/// rng.at(seed, i) is < 1 - p — the exact predicate of dropoutRange.
fn dropoutKeeps(seed: u64, i: usize, p: f32) bool {
    const uniform = @as(f64, @floatFromInt(rng.at(seed, i) >> 11)) * 0x1.0p-53;
    return uniform < 1.0 - @as(f64, p);
}

test "exec context applies unary ops through materialized inputs" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, &.{ 1, 3 }, &.{ -1, 2, -3 });
    defer x.deinit();
    var broadcast = try ctx.broadcastTo(&x, .{ 2, 3 });
    defer broadcast.deinit();

    var y = try ctx.unary(.f32, .relu, &broadcast);
    defer y.deinit();

    try std.testing.expect(y.isContiguous());
    try std.testing.expectEqualSlices(f32, &.{ 0, 2, 0, 0, 2, 0 }, y.dataConst());

    var z = try ctx.unary(.f32, .relu, &broadcast);
    defer z.deinit();
    try std.testing.expectEqualSlices(f32, y.dataConst(), z.dataConst());
}

test "exec context exposes fixed-rank construction and elementwise execution" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(.f32, .{ 2, 1, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSlice(.f32, .{ 2, 1, 3 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer b.deinit();

    var c = try ctx.add(.f32, 3, &a, &b);
    defer c.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 3 }, c.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44, 55, 66 }, c.dataConst());
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.add(.f32, 2, &a, &b));
}

test "exec context applies explicit tail broadcast without materializing the view" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try ctx.fromSlice(.f32, &.{3}, &.{ 10, 20, 30 });
    defer bias.deinit();
    var broadcast = try ctx.broadcastTo(&bias, &.{ 2, 3 });
    defer broadcast.deinit();

    try std.testing.expect(broadcast.buffer == bias.buffer);

    var y = try ctx.elementwise(.f32, .add, &x, &broadcast);
    defer y.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, y.dataConst());

    var z = try ctx.elementwise(.f32, .sub, &x, &broadcast);
    defer z.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -9, -18, -27, -6, -15, -24 }, z.dataConst());

    var m = try ctx.elementwise(.f32, .mul, &x, &broadcast);
    defer m.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 40, 90, 40, 100, 180 }, m.dataConst());
}

test "exec context applies in-place elementwise ops with contiguous and broadcast operands" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try ctx.fromSlice(.f32, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();
    var gate = try ctx.fromSlice(.f32, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer gate.deinit();
    var broadcast = try ctx.broadcastTo(&bias, .{ 2, 3 });
    defer broadcast.deinit();

    try ctx.elementwiseInPlace(.add, &x, &broadcast);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, x.dataConst());

    try ctx.elementwiseInPlace(.mul, &x, &gate);
    try std.testing.expectEqualSlices(f32, &.{ 11, 44, 99, 56, 125, 216 }, x.dataConst());
}

test "exec context take elementwise reuses unique contiguous input" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    var bias = try ctx.fromSlice(.f32, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();
    var broadcast = try ctx.broadcastTo(&bias, .{ 2, 3 });
    defer broadcast.deinit();

    const original_buffer = x.buffer;
    var y = try ctx.takeElementwise(.add, &x, &broadcast);
    defer y.deinit();

    try std.testing.expect(y.buffer == original_buffer);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, y.dataConst());
}

test "exec context take unary and scale reuse unique contiguous input" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, &.{4}, &.{ -1, 2, -3, 4 });
    const original_buffer = x.buffer;

    var y = try ctx.takeUnary(.relu, &x);
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

    var x = try ctx.fromSlice(.f32, &.{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var shared = try x.cloneView();
    var b = try ctx.fromSlice(.f32, &.{3}, &.{ 10, 20, 30 });
    defer b.deinit();

    var y = try ctx.takeElementwise(.mul, &shared, &b);
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

    var source = try ctx.fromSlice(.f32, &.{3}, &.{ 1, 2, 3 });
    defer source.deinit();
    var broadcast = try ctx.broadcastTo(&source, &.{ 2, 3 });
    var x = try ctx.fromSlice(.f32, &.{ 2, 3 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer x.deinit();

    var y = try ctx.takeElementwise(.sub, &broadcast, &x);
    defer y.deinit();
    try std.testing.expect(y.buffer != source.buffer);
    try std.testing.expectEqualSlices(f32, &.{ -9, -18, -27, -39, -48, -57 }, y.dataConst());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, source.dataConst());

    var a = try ctx.fromSlice(.f32, &.{2}, &.{ 1, 2 });
    defer a.deinit();
    var b = try ctx.fromSlice(.f32, &.{3}, &.{ 1, 2, 3 });
    defer b.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.takeElementwise(.add, &a, &b));
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, a.dataConst());
}

test "exec context combines fixed-rank ops with explicit broadcast views" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try ctx.fromSlice(.f32, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();
    var broadcast = try ctx.broadcastTo(&bias, .{ 2, 3 });
    defer broadcast.deinit();

    var y = try ctx.add(.f32, 2, &x, &broadcast);
    defer y.deinit();

    try std.testing.expect(broadcast.buffer == bias.buffer);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, y.dataConst());
}

test "exec context handles broadcast operands on both sides of elementwise ops" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try ctx.fromSlice(.f32, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();
    var bias_b = try ctx.broadcastTo(&bias, .{ 2, 3 });
    defer bias_b.deinit();
    var scalar_value = try ctx.scalar(.f32, 2);
    defer scalar_value.deinit();
    var scalar_b = try ctx.broadcastTo(&scalar_value, .{ 2, 3 });
    defer scalar_b.deinit();

    var left_sub = try ctx.sub(.f32, 2, &bias_b, &x);
    defer left_sub.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 9, 18, 27, 6, 15, 24 }, left_sub.dataConst());

    var right_sub = try ctx.sub(.f32, 2, &x, &bias_b);
    defer right_sub.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -9, -18, -27, -6, -15, -24 }, right_sub.dataConst());

    var both_broadcast_sub = try ctx.sub(.f32, 2, &bias_b, &scalar_b);
    defer both_broadcast_sub.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 8, 18, 28, 8, 18, 28 }, both_broadcast_sub.dataConst());

    var both_broadcast_mul = try ctx.mul(.f32, 2, &bias_b, &scalar_b);
    defer both_broadcast_mul.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 20, 40, 60, 20, 40, 60 }, both_broadcast_mul.dataConst());
}

test "exec context rank-specializes elementwise ops above rank four" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(.f32, .{ 1, 1, 1, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try ctx.fromSlice(.f32, .{ 1, 1, 1, 2, 3 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer b.deinit();

    var sum = try ctx.elementwise(.f32, .add, &a, &b);
    defer sum.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44, 55, 66 }, sum.dataConst());

    var diff = try ctx.sub(.f32, 5, &b, &a);
    defer diff.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 9, 18, 27, 36, 45, 54 }, diff.dataConst());

    var product = try ctx.mul(.f32, 5, &a, &b);
    defer product.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 40, 90, 160, 250, 360 }, product.dataConst());
}

test "exec context reduces broadcast gradient to source shape" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var gy = try ctx.fromSlice(.f32, &.{ 2, 3 }, &.{ 1, 1, 1, 1, 1, 1 });
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

    var x = try ctx.fromSlice(.f32, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var scalar_value = try ctx.scalar(.f32, 10);
    defer scalar_value.deinit();
    var scalar_b = try ctx.broadcastTo(&scalar_value, &.{ 2, 3 });
    defer scalar_b.deinit();

    var y = try ctx.elementwise(.f32, .add, &x, &scalar_b);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 12, 13, 14, 15, 16 }, y.dataConst());

    var middle = try ctx.fromSlice(.f32, &.{ 2, 1, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer middle.deinit();
    var middle_b = try ctx.broadcastTo(&middle, &.{ 2, 4, 3 });
    defer middle_b.deinit();

    var zeros = try ctx.zeros(.f32, &.{ 2, 4, 3 });
    defer zeros.deinit();
    var copied = try ctx.elementwise(.f32, .add, &zeros, &middle_b);
    defer copied.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
        4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
    }, copied.dataConst());
}

test "exec context runs typed float forward math kernels" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(.f64, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try ctx.fromSlice(.f64, .{ 2, 2 }, &.{ 10, 20, 30, 40 });
    defer b.deinit();

    var sum = try ctx.add(.f64, 2, &a, &b);
    defer sum.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 11, 22, 33, 44 }, sum.dataConst());

    var reduced = try ctx.sumAxis(.f64, 2, &sum, 1);
    defer reduced.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 33, 77 }, reduced.dataConst());

    var dot64 = try ctx.dot(.f64, &a, &b);
    defer dot64.deinit();
    try std.testing.expectEqual(@as(f64, 300), dot64.dataConst()[0]);

    var matmul64 = try ctx.matmul(.f64, .plain, &a, &b);
    defer matmul64.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 70, 100, 150, 220 }, matmul64.dataConst());

    var h1 = try ctx.fromSlice(.f16, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer h1.deinit();
    var h2 = try ctx.fromSlice(.f16, .{ 2, 2 }, &.{ 2, 3, 4, 5 });
    defer h2.deinit();
    var hmul = try ctx.mul(.f16, 2, &h1, &h2);
    defer hmul.deinit();
    try std.testing.expectEqualSlices(f16, &.{ 2, 6, 12, 20 }, hmul.dataConst());

    var hsum = try ctx.sumAxis(.f16, 2, &hmul, 1);
    defer hsum.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 8, 32 }, hsum.dataConst());

    var hdot = try ctx.dot(.f16, &h1, &h2);
    defer hdot.deinit();
    try std.testing.expectEqual(@as(f16, 40), hdot.dataConst()[0]);

    var hleft = try ctx.fromSlice(.f16, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer hleft.deinit();
    var hright = try ctx.fromSlice(.f16, .{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer hright.deinit();
    var hproduct = try ctx.matmul(.f16, .plain, &hleft, &hright);
    defer hproduct.deinit();
    try std.testing.expectEqualSlices(f16, &.{ 58, 64, 139, 154 }, hproduct.dataConst());

    var left = try ctx.fromSlice(.bf16, .{ 2, 3 }, &.{
        dtype_mod.f32ToBf16(1),
        dtype_mod.f32ToBf16(2),
        dtype_mod.f32ToBf16(3),
        dtype_mod.f32ToBf16(4),
        dtype_mod.f32ToBf16(5),
        dtype_mod.f32ToBf16(6),
    });
    defer left.deinit();
    var bf16_sum = try ctx.sum(.bf16, &left);
    defer bf16_sum.deinit();
    try std.testing.expectEqualSlices(f32, &.{21}, bf16_sum.dataConst());

    var right = try ctx.fromSlice(.bf16, .{ 3, 2 }, &.{
        dtype_mod.f32ToBf16(7),
        dtype_mod.f32ToBf16(8),
        dtype_mod.f32ToBf16(9),
        dtype_mod.f32ToBf16(10),
        dtype_mod.f32ToBf16(11),
        dtype_mod.f32ToBf16(12),
    });
    defer right.deinit();
    var product = try ctx.matmul(.bf16, .plain, &left, &right);
    defer product.deinit();
    try std.testing.expectEqual(@as(f32, 58), dtype_mod.bf16ToF32(product.dataConst()[0]));
    try std.testing.expectEqual(@as(f32, 154), dtype_mod.bf16ToF32(product.dataConst()[3]));

    var cast = try ctx.cast(.bf16, .f32, &product);
    defer cast.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, cast.dataConst());
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

    var x = try ctx.fromSlice(.f32, .{ 64, 64 }, data);
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

    var x_par = try ctx.fromSlice(.f32, .{par_len}, data);
    defer x_par.deinit();
    var x_ser = try ctx.fromSlice(.f32, .{ser_len}, data[0..ser_len]);
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

test "exec context reduces higher-rank and scalar broadcast gradients" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var gy = try ctx.fromSlice(.f32, &.{ 2, 2, 3 }, &.{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    });
    defer gy.deinit();

    var tail = try exec_elementwise.reduceBroadcast(&ctx, &gy, .{3});
    defer tail.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 22, 26, 30 }, tail.dataConst());

    var exact = try exec_elementwise.reduceBroadcast(&ctx, &gy, .{ 2, 2, 3 });
    defer exact.deinit();
    try std.testing.expectEqualSlices(f32, gy.dataConst(), exact.dataConst());
    try std.testing.expectEqual(@as(usize, 2), ctx.buffers.outstandingBuffers());

    var scalar_reduced = try ctx.reduceBroadcast(&gy, &.{1});
    defer scalar_reduced.deinit();
    try std.testing.expectEqual(@as(f32, 78), scalar_reduced.item());

    var singleton_middle = try exec_elementwise.reduceBroadcast(&ctx, &gy, .{ 2, 1, 3 });
    defer singleton_middle.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9, 17, 19, 21 }, singleton_middle.dataConst());

    var singleton_prefix = try exec_elementwise.reduceBroadcast(&ctx, &gy, .{ 1, 3 });
    defer singleton_prefix.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 22, 26, 30 }, singleton_prefix.dataConst());

    try std.testing.expectError(tensor.TensorError.ShapeMismatch, exec_elementwise.reduceBroadcast(&ctx, &gy, .{ 2, 2 }));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.reduceBroadcast(&gy, &.{ 0, 3 }));
}

test "snake: hand-computed per-channel activation + shape rejection" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, .{ 2, 2 }, &.{ 0.5, -1.0, 2.0, 0.0 });
    defer x.deinit();
    var alpha = try ctx.fromSlice(.f32, .{2}, &.{ 1.0, 2.0 });
    defer alpha.deinit();
    var inv_b = try ctx.fromSlice(.f32, .{2}, &.{ 1.0, 0.5 });
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

    var short_alpha = try ctx.fromSlice(.f32, .{1}, &.{1.0});
    defer short_alpha.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.snakeRows(&x, &short_alpha, &inv_b));
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.snakeRows(&x, &alpha, &short_alpha));
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

    var x = try ctx.fromSlice(.f32, .{ 2, 2 }, &x_vals);
    defer x.deinit();
    var gy = try ctx.fromSlice(.f32, .{ 2, 2 }, &gy_vals);
    defer gy.deinit();
    var alpha = try ctx.fromSlice(.f32, .{2}, &a_vals);
    defer alpha.deinit();
    var inv_b = try ctx.fromSlice(.f32, .{2}, &ib_vals);
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
    var bad_gy = try ctx.fromSlice(.f32, .{ 1, 2 }, &.{ 1, 2 });
    defer bad_gy.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.snakeRowsBackwardInput(&x, &bad_gy, &alpha, &inv_b));
    var short_alpha = try ctx.fromSlice(.f32, .{1}, &.{1.0});
    defer short_alpha.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.snakeRowsBackwardInput(&x, &gy, &short_alpha, &inv_b));
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.snakeRowsBackwardParams(&x, &gy, &alpha, &short_alpha));
}

test "exec context applies elu and gelu_erf unary ops" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.fromSlice(.f32, .{4}, &.{ -1.0, 0.0, 1.0, 2.0 });
    defer x.deinit();

    var elu_y = try ctx.unary(.f32, .elu, &x);
    defer elu_y.deinit();
    const elu_want = [_]f32{ -0.6321206, 0.0, 1.0, 2.0 };
    for (elu_want, elu_y.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);

    var gelu_y = try ctx.unary(.f32, .gelu_erf, &x);
    defer gelu_y.deinit();
    const gelu_want = [_]f32{ -0.15865526, 0.0, 0.8413447, 1.9544997 };
    for (gelu_want, gelu_y.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);

    var erf_y = try ctx.unary(.f32, .erf, &x);
    defer erf_y.deinit();
    const erf_want = [_]f32{ -0.8427008, 0.0, 0.8427008, 0.9953223 };
    for (erf_want, erf_y.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);
}

test "exec compare and compareScalar produce IEEE 0/1 masks (NaN false except ne)" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const nan = std.math.nan(f32);
    // torch: torch.eq/ne/lt/le/gt/ge(a, b).float() with a NaN lane — every
    // comparison involving NaN is false except ne, which is true.
    var a = try ctx.fromSlice(.f32, .{4}, &.{ 1, 2, nan, -3 });
    defer a.deinit();
    var b = try ctx.fromSlice(.f32, .{4}, &.{ 1, 5, nan, -4 });
    defer b.deinit();

    var eq = try ctx.compare(.f32, .eq, &a, &b);
    defer eq.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, eq.dataConst());
    var ne = try ctx.compare(.f32, .ne, &a, &b);
    defer ne.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, ne.dataConst());
    var lt = try ctx.compare(.f32, .lt, &a, &b);
    defer lt.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, lt.dataConst());
    var le = try ctx.compare(.f32, .le, &a, &b);
    defer le.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, le.dataConst());
    var gt = try ctx.compare(.f32, .gt, &a, &b);
    defer gt.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, gt.dataConst());
    var ge = try ctx.compare(.f32, .ge, &a, &b);
    defer ge.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, ge.dataConst());

    // Scalar RHS: x > 1.5 -> {0, 1, 0, 0}; x != 1.5 with the NaN lane true.
    var gt_s = try ctx.compareScalar(.f32, .gt, &a, 1.5);
    defer gt_s.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, gt_s.dataConst());
    var ne_s = try ctx.compareScalar(.f32, .ne, &a, 1.5);
    defer ne_s.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, ne_s.dataConst());
    var eq_s = try ctx.compareScalar(.f32, .eq, &a, nan);
    defer eq_s.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, eq_s.dataConst());

    // Same-shape contract, like where.
    var short = try ctx.fromSlice(.f32, .{2}, &.{ 1, 2 });
    defer short.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.compare(.f32, .eq, &a, &short));
}

test "exec logical ops use != 0 truthiness with 0/1 outputs" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // Nonzero (incl. negatives and NaN) is true; 0 is false.
    const nan = std.math.nan(f32);
    var a = try ctx.fromSlice(.f32, .{4}, &.{ 0, 2, 0, -3 });
    defer a.deinit();
    var b = try ctx.fromSlice(.f32, .{4}, &.{ 0, 0, nan, 5 });
    defer b.deinit();

    var land = try ctx.logical(.l_and, .f32, .f32, &a, &b);
    defer land.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, land.dataConst());
    var lor = try ctx.logical(.l_or, .f32, .f32, &a, &b);
    defer lor.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lor.dataConst());
    var lxor = try ctx.logical(.l_xor, .f32, .f32, &a, &b);
    defer lxor.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, lxor.dataConst());
    var lnot = try ctx.logicalNot(.f32, &a);
    defer lnot.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, lnot.dataConst());

    // Same-shape contract, like where.
    var short = try ctx.fromSlice(.f32, .{2}, &.{ 1, 0 });
    defer short.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.logical(.l_and, .f32, .f32, &a, &short));
}
