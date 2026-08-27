//! RoPE table construction: the `.range` positions source against the
//! `.explicit` form, Fortran-style origins, and `.theta` frequency factors.
//! Force-imported by `exec.zig`.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const exec = @import("../exec.zig");
const exec_elementwise = @import("elementwise.zig");
const exec_row_ops = @import("../backend.zig").rows;
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
    const AxisRange = exec.AxisRange;

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

test "ropeWithTable splits vectors across the pool bitwise identically to the serial walk" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // [seq, heads, d] at a prefill size well above the pooling gate
    // (4 Mi elements vs the 32 Ki gate), so the pooled arm is the one under
    // test; the serial walk is the same body with the team forced to one.
    const seq = 2048;
    const heads = 16;
    const feature_dim = 128;
    const data = try allocator.alloc(f32, seq * heads * feature_dim);
    defer allocator.free(data);
    var prng = std.Random.DefaultPrng.init(0x50e);
    const random = prng.random();
    for (data) |*v| v.* = random.floatNorm(f32);
    var x = try ctx.fromSlice(.f32, .{ seq, heads, feature_dim }, data);
    defer x.deinit();

    var full = try ctx.prepareRopeTable(.{ .positions = .{ .range = .{ .origin = 3, .len = seq } }, .feature_dim = feature_dim, .freqs = .{ .theta = .{ .base = 10000 } } });
    defer full.deinit();
    // A partial span (tail-64) exercises the pass-through copy + offset arm.
    var partial = try ctx.prepareRopeTable(.{ .positions = .{ .range = .{ .origin = 3, .len = seq } }, .feature_dim = 64, .freqs = .{ .theta = .{ .base = 10000 } } });
    defer partial.deinit();

    // The team is built lazily and sized at creation, so it is created at
    // the full count BEFORE the override drops the count to one (a team
    // created under the override would have no workers and every later
    // dispatch would run on the caller, testing nothing).
    const saved_threads = parallel.cpuThreadCount(parallel.vector_max_threads);
    _ = try ctx.tryWorkPool();
    defer parallel.setMaxThreads(saved_threads);

    inline for ([_]exec.RopeMode{ .half, .interleaved, .interleaved_tail }) |mode| {
        inline for (.{ true, false }) |full_span| {
            const table = if (full_span) &full else &partial;
            const rotary_dim: usize = if (full_span) feature_dim else 64;
            const rotary_offset: usize = if (!full_span and mode == .interleaved_tail) feature_dim - rotary_dim else 0;
            const pair_count = rotary_dim / 2;

            parallel.setMaxThreads(saved_threads);
            var pooled = try ctx.ropeWithTable(3, &x, 0, 2, table, mode);
            defer pooled.deinit();
            parallel.setMaxThreads(1);
            var serial = try ctx.ropeWithTable(3, &x, 0, 2, table, mode);
            defer serial.deinit();
            try std.testing.expectEqualSlices(f32, serial.dataConst(), pooled.dataConst());

            // Both equal the scalar per-pair formula (the serial walk's
            // own tail loop), so the equality above is not two copies of
            // one mistake.
            const sin_values = table.sinValues();
            const cos_values = table.cosValues();
            const got = pooled.dataConst();
            for (0..seq * heads) |vector_i| {
                const position = vector_i / heads;
                const base = vector_i * feature_dim;
                for (0..feature_dim) |feature_i| {
                    if (feature_i < rotary_offset or feature_i >= rotary_offset + rotary_dim) {
                        try std.testing.expectEqual(data[base + feature_i], got[base + feature_i]);
                    }
                }
                for (0..pair_count) |pair_i| {
                    const sin_value = sin_values[position * pair_count + pair_i];
                    const cos_value = cos_values[position * pair_count + pair_i];
                    const first_i = base + rotary_offset + (if (mode == .half) pair_i else 2 * pair_i);
                    const second_i = base + rotary_offset + (if (mode == .half) pair_i + pair_count else 2 * pair_i + 1);
                    const first = data[first_i];
                    const second = data[second_i];
                    try std.testing.expectEqual(first * cos_value - second * sin_value, got[first_i]);
                    try std.testing.expectEqual(first * sin_value + second * cos_value, got[second_i]);
                }
            }
        }
    }
}
