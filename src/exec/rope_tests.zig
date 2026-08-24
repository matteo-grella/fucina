//! RoPE table construction: the `.range` positions source against the
//! `.explicit` form, Fortran-style origins, and `.theta` frequency factors.
//! Force-imported by `exec.zig`.

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

test "a rope table built from an AxisRange is bitwise equal to the array form" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // The refactor this pins: ~17 call sites used to allocate an i32 array,
    // fill it with `origin + i`, build a table, and free it, purely to express
    // `origin`. Both forms feed one arithmetic body, so the tables must agree
    // BITWISE — a tolerance here would not be evidence, since every model
    // family's rotary parity now rides on this equality.
    const feature_dim = 64;
    const theta: f32 = 10000;
    const count = 37; // deliberately not a vector multiple
    for ([_]i64{ 0, 1, 7, 4096, 131072 }) |origin| {
        const positions = try allocator.alloc(i32, count);
        defer allocator.free(positions);
        for (positions, 0..) |*p, i| p.* = @intCast(origin + @as(i64, @intCast(i)));

        for ([_]bool{ false, true }) |inverse| {
            var from_array = try ctx.prepareRopeTable(.{ .positions = .{ .explicit = positions }, .feature_dim = feature_dim, .freqs = .{ .theta = .{ .base = theta } }, .inverse = inverse });
            defer from_array.deinit();
            var from_range = try ctx.prepareRopeTable(.{ .positions = .{ .range = .{ .origin = origin, .len = count } }, .feature_dim = feature_dim, .freqs = .{ .theta = .{ .base = theta } }, .inverse = inverse });
            defer from_range.deinit();

            try std.testing.expectEqualSlices(f32, from_array.sinValues(), from_range.sinValues());
            try std.testing.expectEqualSlices(f32, from_array.cosValues(), from_range.cosValues());
            try std.testing.expectEqual(from_array.feature_dim, from_range.feature_dim);
            try std.testing.expectEqual(from_array.pair_count, from_range.pair_count);
        }

        // Same equality with per-pair frequency factors (the Llama-3 /
        // Gemma-global arm).
        const factors = try allocator.alloc(f32, feature_dim / 2);
        defer allocator.free(factors);
        for (factors, 0..) |*f, i| f.* = 1.0 + @as(f32, @floatFromInt(i)) * 0.03;

        var array_factors = try ctx.prepareRopeTable(.{ .positions = .{ .explicit = positions }, .feature_dim = feature_dim, .freqs = .{ .theta = .{ .base = theta, .factors = factors } } });
        defer array_factors.deinit();
        var range_factors = try ctx.prepareRopeTable(.{ .positions = .{ .range = .{ .origin = origin, .len = count } }, .feature_dim = feature_dim, .freqs = .{ .theta = .{ .base = theta, .factors = factors } } });
        defer range_factors.deinit();
        try std.testing.expectEqualSlices(f32, array_factors.sinValues(), range_factors.sinValues());
        try std.testing.expectEqualSlices(f32, array_factors.cosValues(), range_factors.cosValues());
    }
}

test "AxisRange carries an origin the way a Fortran lower bound does" {
    const AxisRange = tensor.AxisRange;

    // The plain 0-origin axis is the default, so `.{ .len = n }` is the
    // ordinary case and costs nothing to write.
    const plain: AxisRange = .{ .len = 4 };
    try std.testing.expectEqual(@as(i64, 0), plain.origin);
    try std.testing.expectEqual(@as(i64, 4), plain.end());
    try std.testing.expectEqual(@as(i64, 2), plain.at(2));

    const decode: AxisRange = .{ .origin = 4096, .len = 3 };
    try std.testing.expectEqual(@as(i64, 4096), decode.at(0));
    try std.testing.expectEqual(@as(i64, 4099), decode.end());
    try std.testing.expect(decode.contains(4098));
    try std.testing.expect(!decode.contains(4099));
    try std.testing.expect(!decode.contains(4095));

    // narrowed is the operation a 0-origin axis cannot express: take a local
    // sub-run and keep knowing where it sits absolutely.
    const tail = decode.narrowed(1, 2);
    try std.testing.expectEqual(@as(i64, 4097), tail.origin);
    try std.testing.expectEqual(@as(usize, 2), tail.len);

    try std.testing.expectEqual(@as(i64, 4106), decode.shifted(10).origin);
    try std.testing.expectEqual(@as(i64, 0), decode.rebased(0).origin);
    try std.testing.expectEqual(@as(usize, 3), decode.rebased(0).len);

    // The materialization it exists to avoid, still available for ragged
    // interop boundaries.
    var out: [3]i32 = undefined;
    try decode.writeInto(&out);
    try std.testing.expectEqualSlices(i32, &.{ 4096, 4097, 4098 }, &out);
    var wrong: [2]i32 = undefined;
    try std.testing.expectError(error.InvalidDataLength, decode.writeInto(&wrong));
}

test "theta factors scale frequencies; null reproduces plain RoPE" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const positions = [_]i32{ 0, 1, 5 };
    const feature_dim: usize = 8;
    const theta_base: f32 = 10000;
    const ff = [_]f32{ 1.0, 2.0, 4.0, 8.0 };

    var plain = try ctx.prepareRopeTable(.{ .positions = .{ .explicit = &positions }, .feature_dim = feature_dim, .freqs = .{ .theta = .{ .base = theta_base } } });
    defer plain.deinit();
    var plain_null = try ctx.prepareRopeTable(.{ .positions = .{ .explicit = &positions }, .feature_dim = feature_dim, .freqs = .{ .theta = .{ .base = theta_base } } });
    defer plain_null.deinit();
    try std.testing.expectEqualSlices(f32, plain.values, plain_null.values);

    var scaled = try ctx.prepareRopeTable(.{ .positions = .{ .explicit = &positions }, .feature_dim = feature_dim, .freqs = .{ .theta = .{ .base = theta_base, .factors = &ff } } });
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
