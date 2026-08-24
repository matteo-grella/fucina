//! Softmax kernels through ExecContext: fast vs generic layouts, NaN rows,
//! backward paths, strided inner kernels, and softmaxExt option combos.
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

const util = @import("test_util.zig");
const expectCloseToF64 = util.expectCloseToF64;

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

        var x = try ctx.fromSlice(.f32, .{ rows, cols }, data);
        defer x.deinit();
        var y = try ctx.softmax(2, &x, 1);
        defer y.deinit();
        const yd = y.dataConst();

        for (0..rows) |row| {
            const expected = try testNaiveSoftmaxRow(allocator, data[row * cols ..][0..cols]);
            defer allocator.free(expected);
            for (expected, yd[row * cols ..][0..cols]) |want, got| {
                try expectCloseToF64(want, got, 1e-5, 1e-12);
            }
        }

        // Same data transposed with axis 0 forces inner > 1, i.e. the strided
        // inner-lane path; the row-kernel fast path must agree with it.
        const transposed = try allocator.alloc(f32, rows * cols);
        defer allocator.free(transposed);
        for (0..rows) |row| {
            for (0..cols) |col| transposed[col * rows + row] = data[row * cols + col];
        }
        var xt = try ctx.fromSlice(.f32, .{ cols, rows }, transposed);
        defer xt.deinit();
        var yt = try ctx.softmax(2, &xt, 0);
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
    var x = try ctx.fromSlice(.f32, .{ 2, cols }, &data);
    defer x.deinit();
    var y = try ctx.softmax(2, &x, 1);
    defer y.deinit();
    const yd = y.dataConst();

    // Same data transposed, softmax along axis 0: inner > 1, the strided
    // inner-lane path. Both paths must agree on NaN poisoning.
    var transposed: [2 * cols]f32 = undefined;
    for (0..2) |row| {
        for (0..cols) |col| transposed[col * 2 + row] = data[row * cols + col];
    }
    var xt = try ctx.fromSlice(.f32, .{ cols, 2 }, &transposed);
    defer xt.deinit();
    var yt = try ctx.softmax(2, &xt, 0);
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

    var y = try ctx.fromSlice(.f32, .{ rows, cols }, y_data);
    defer y.deinit();
    var gy = try ctx.fromSlice(.f32, .{ rows, cols }, gy_data);
    defer gy.deinit();
    var gx = try ctx.softmaxBackward(2, &y, &gy, 1, 0.5);
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
    var yt = try ctx.fromSlice(.f32, .{ cols, rows }, yt_data);
    defer yt.deinit();
    var gyt = try ctx.fromSlice(.f32, .{ cols, rows }, gyt_data);
    defer gyt.deinit();
    var gxt = try ctx.softmaxBackward(2, &yt, &gyt, 0, 0.5);
    defer gxt.deinit();
    const gxtd = gxt.dataConst();

    for (0..rows) |row| {
        for (0..cols) |col| {
            try expectCloseToF64(gxtd[col * rows + row], gxd[row * cols + col], 1e-4, 1e-6);
        }
    }
}

test "softmax family strided inner kernels match last-axis rows across the parallel lane split" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x50f9);
    const random = prng.random();

    // inner = 192 after the transpose: multiple lane-split tasks (192/64 = 3)
    // plus a vector tail, and rows*cols clears the parallel threshold — the
    // widths where a scratch-column or split-boundary bug would show.
    const rows: usize = 192;
    const cols: usize = 900;
    const data = try allocator.alloc(f32, rows * cols);
    defer allocator.free(data);
    const gy_data = try allocator.alloc(f32, rows * cols);
    defer allocator.free(gy_data);
    for (data) |*value| value.* = random.floatNorm(f32) * 3;
    for (gy_data) |*value| value.* = random.floatNorm(f32);

    const transposed = try allocator.alloc(f32, rows * cols);
    defer allocator.free(transposed);
    const gyt_data = try allocator.alloc(f32, rows * cols);
    defer allocator.free(gyt_data);
    for (0..rows) |row| {
        for (0..cols) |col| {
            transposed[col * rows + row] = data[row * cols + col];
            gyt_data[col * rows + row] = gy_data[row * cols + col];
        }
    }

    var x = try ctx.fromSlice(.f32, .{ rows, cols }, data);
    defer x.deinit();
    var xt = try ctx.fromSlice(.f32, .{ cols, rows }, transposed);
    defer xt.deinit();
    var gy = try ctx.fromSlice(.f32, .{ rows, cols }, gy_data);
    defer gy.deinit();
    var gyt = try ctx.fromSlice(.f32, .{ cols, rows }, gyt_data);
    defer gyt.deinit();

    var y = try ctx.softmax(2, &x, 1);
    defer y.deinit();
    var yt = try ctx.softmax(2, &xt, 0);
    defer yt.deinit();
    for (0..rows) |row| {
        for (0..cols) |col| {
            try expectCloseToF64(yt.dataConst()[col * rows + row], y.dataConst()[row * cols + col], 1e-5, 1e-12);
        }
    }

    // Log-space outputs compare across two different exp-sum accumulation
    // orders (vector-tree rows kernel vs sequential per-lane inner kernel),
    // so the band is wider than the probability-space checks above.
    var ls = try ctx.logSoftmax(2, &x, 1);
    defer ls.deinit();
    var lst = try ctx.logSoftmax(2, &xt, 0);
    defer lst.deinit();
    for (0..rows) |row| {
        for (0..cols) |col| {
            try expectCloseToF64(lst.dataConst()[col * rows + row], ls.dataConst()[row * cols + col], 5e-5, 1e-5);
        }
    }

    var lse = try ctx.logsumexp(2, &x, 1);
    defer lse.deinit();
    var lset = try ctx.logsumexp(2, &xt, 0);
    defer lset.deinit();
    for (0..rows) |row| {
        try expectCloseToF64(lset.dataConst()[row], lse.dataConst()[row], 5e-5, 1e-5);
    }

    var gx = try ctx.softmaxBackward(2, &y, &gy, 1, 1);
    defer gx.deinit();
    var gxt = try ctx.softmaxBackward(2, &yt, &gyt, 0, 1);
    defer gxt.deinit();
    for (0..rows) |row| {
        for (0..cols) |col| {
            try expectCloseToF64(gxt.dataConst()[col * rows + row], gx.dataConst()[row * cols + col], 1e-4, 1e-6);
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

    var x = try ctx.fromSlice(.f32, .{ heads, q_dim, src_dim }, data);
    defer x.deinit();
    var mask = try ctx.fromSlice(.f32, .{ heads, q_dim, src_dim }, mask_data);
    defer mask.deinit();
    var y = try ctx.softmaxExt(3, &x, 2, .{
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
    var xt = try ctx.fromSlice(.f32, .{ heads, src_dim, q_dim }, data_t);
    defer xt.deinit();
    var maskt = try ctx.fromSlice(.f32, .{ heads, src_dim, q_dim }, mask_t);
    defer maskt.deinit();
    var yt = try ctx.softmaxExt(3, &xt, 1, .{
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

    var x = try ctx.fromSlice(.f32, .{ 2, 4, 6 }, &.{
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
    var mask_thin = try ctx.fromSlice(.f32, .{ 2, 4, 1 }, &mask_rows);
    defer mask_thin.deinit();
    var mask_full_data: [2 * 4 * 6]f32 = undefined;
    for (0..8) |row| {
        for (0..6) |col| mask_full_data[row * 6 + col] = mask_rows[row];
    }
    var mask_full = try ctx.fromSlice(.f32, .{ 2, 4, 6 }, &mask_full_data);
    defer mask_full.deinit();

    var y_thin = try ctx.softmaxExt(3, &x, 2, .{ .mask = &mask_thin, .scale = 0.7 });
    defer y_thin.deinit();
    var y_full = try ctx.softmaxExt(3, &x, 2, .{ .mask = &mask_full, .scale = 0.7 });
    defer y_full.deinit();

    for (y_thin.dataConst(), y_full.dataConst()) |scalar_path, fast| {
        try expectCloseToF64(scalar_path, fast, 1e-5, 1e-9);
    }
}
