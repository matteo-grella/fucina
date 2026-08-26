//! Normalization kernels through ExecContext: layer norm forward and
//! backward against naive f64 references, and groupNorm hand-computed cases
//! with shape rejection. Force-imported by `exec.zig`.

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

const util = @import("test_util.zig");
const expectCloseToF64 = util.expectCloseToF64;

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
                    try ctx.fromSlice(.f32, .{ outer, axis_dim }, data)
                else
                    try ctx.fromSlice(.f32, .{ outer, axis_dim, inner }, data);
                defer x.deinit();
                var w = try ctx.fromSlice(.f32, .{axis_dim}, weights);
                defer w.deinit();
                var b = try ctx.fromSlice(.f32, .{axis_dim}, biases);
                defer b.deinit();

                for ([_]f32{ 1e-5, 1e-6 }) |eps| {
                    const ref_plain = try testNaiveLayerNorm(allocator, data, outer, axis_dim, inner, null, null, eps);
                    defer allocator.free(ref_plain);
                    var y_plain = try ctx.layerNorm(.f32, rank, &x, 1, eps, .{});
                    defer y_plain.deinit();
                    for (y_plain.dataConst(), ref_plain) |got, want| {
                        try expectCloseToF64(want, got, 5e-4, 5e-5);
                    }

                    const ref_affine = try testNaiveLayerNorm(allocator, data, outer, axis_dim, inner, weights, biases, eps);
                    defer allocator.free(ref_affine);
                    var y_affine = try ctx.layerNorm(.f32, rank, &x, 1, eps, .{ .weight = &w, .bias = &b });
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

    // {outer, axis_dim, inner}: small SIMD rows, the inner>1 streaming
    // kernel at scalar-tail (inner=3) and vector-lane (inner=70) widths, a
    // degenerate single-element axis, and a shape big enough for the
    // parallel dx dispatch (70*2050 >= threshold/2).
    const cases = [_][3]usize{
        .{ 3, 5, 1 },
        .{ 2, 8, 3 },
        .{ 2, 9, 70 },
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
            var x = try ctx.fromSlice(.f32, shape, data);
            defer x.deinit();
            var gy = try ctx.fromSlice(.f32, shape, grad);
            defer gy.deinit();
            var w = try ctx.fromSlice(.f32, .{axis_dim}, weights);
            defer w.deinit();

            // Plain dx.
            var ref_plain = try testNaiveLayerNormBackward(allocator, data, grad, outer, axis_dim, inner, null, eps);
            defer ref_plain.deinit(allocator);
            var gx_plain = (try ctx.layerNormBackward(rank, &x, &gy, 1, eps, .{})).input.?;
            defer gx_plain.deinit();
            for (gx_plain.dataConst(), ref_plain.dx) |got, want| {
                try expectCloseToF64(want, got, 5e-4, 5e-5);
            }

            // Affine dx + dweight + dbias.
            var ref_affine = try testNaiveLayerNormBackward(allocator, data, grad, outer, axis_dim, inner, weights, eps);
            defer ref_affine.deinit(allocator);
            var full = try ctx.layerNormBackward(rank, &x, &gy, 1, eps, .{ .weight = &w, .need_input = true, .need_weight = true, .need_bias = true });
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
            var weight_only = try ctx.layerNormBackward(rank, &x, &gy, 1, eps, .{ .weight = &w, .need_input = false, .need_weight = true, .need_bias = false });
            defer weight_only.deinit();
            try std.testing.expect(weight_only.input == null);
            try std.testing.expect(weight_only.bias == null);
            try std.testing.expectEqualSlices(f32, full.weight.?.dataConst(), weight_only.weight.?.dataConst());

            var bias_only = try ctx.layerNormBackward(rank, &x, &gy, 1, eps, .{ .weight = &w, .need_input = false, .need_weight = false, .need_bias = true });
            defer bias_only.deinit();
            try std.testing.expect(bias_only.input == null);
            try std.testing.expect(bias_only.weight == null);
            try std.testing.expectEqualSlices(f32, full.bias.?.dataConst(), bias_only.bias.?.dataConst());

            // Bitwise determinism across runs (the big case exercises the
            // parallel dx dispatch; dweight/dbias are column-partitioned
            // across the pool, every column accumulated in row order by
            // exactly one task, so they are bitwise identical for any
            // thread count).
            var again = try ctx.layerNormBackward(rank, &x, &gy, 1, eps, .{ .weight = &w, .need_input = true, .need_weight = true, .need_bias = true });
            defer again.deinit();
            try std.testing.expectEqualSlices(f32, full.input.?.dataConst(), again.input.?.dataConst());
            try std.testing.expectEqualSlices(f32, full.weight.?.dataConst(), again.weight.?.dataConst());
            try std.testing.expectEqualSlices(f32, full.bias.?.dataConst(), again.bias.?.dataConst());

            // Forward determinism on the same shapes.
            var y_one = try ctx.layerNorm(.f32, rank, &x, 1, eps, .{ .weight = &w, .bias = &w });
            defer y_one.deinit();
            var y_two = try ctx.layerNorm(.f32, rank, &x, 1, eps, .{ .weight = &w, .bias = &w });
            defer y_two.deinit();
            try std.testing.expectEqualSlices(f32, y_one.dataConst(), y_two.dataConst());
        }
    }
}

test "groupNorm: hand-computed G=1, G=C, and affine cases + rejection" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const eps: f32 = 1e-5;

    // G=1: one group over ALL T*C elements [1,2,3,4]: mean 2.5, biased var 1.25.
    var x = try ctx.fromSlice(.f32, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var y1 = try ctx.groupNorm(&x, 1, eps, .{});
    defer y1.deinit();
    const inv1 = 1.0 / @sqrt(@as(f32, 1.25) + eps);
    const want1 = [_]f32{ -1.5 * inv1, -0.5 * inv1, 0.5 * inv1, 1.5 * inv1 };
    for (want1, y1.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);

    // G=C: per-channel over time (InstanceNorm over T; the HuBERT layer-0
    // configuration). col0 = {1,3}: mean 2, var 1; col1 = {2,4}: mean 3, var 1.
    var y2 = try ctx.groupNorm(&x, 2, eps, .{});
    defer y2.deinit();
    const inv2 = 1.0 / @sqrt(@as(f32, 1.0) + eps);
    const want2 = [_]f32{ -inv2, -inv2, inv2, inv2 };
    for (want2, y2.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);

    // Affine applied AFTER normalization: y*w + b.
    var wt = try ctx.fromSlice(.f32, .{2}, &.{ 2.0, 3.0 });
    defer wt.deinit();
    var bt = try ctx.fromSlice(.f32, .{2}, &.{ 10.0, 20.0 });
    defer bt.deinit();
    var y3 = try ctx.groupNorm(&x, 2, eps, .{ .weight = &wt, .bias = &bt });
    defer y3.deinit();
    const want3 = [_]f32{ -inv2 * 2 + 10, -inv2 * 3 + 20, inv2 * 2 + 10, inv2 * 3 + 20 };
    for (want3, y3.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);

    // groups must divide C; affine vectors must be [C].
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.groupNorm(&x, 3, eps, .{}));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.groupNorm(&x, 0, eps, .{}));
    var bad_w = try ctx.fromSlice(.f32, .{3}, &.{ 1, 2, 3 });
    defer bad_w.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.groupNorm(&x, 2, eps, .{ .weight = &bad_w }));
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
    var x = try ctx.fromSlice(.f32, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var gy = try ctx.fromSlice(.f32, .{ 2, 2 }, &.{ 1, 0, 0, 2 });
    defer gy.deinit();

    var result = try ctx.groupNormBackward(&x, &gy, 2, eps, .{ .need_input = true, .need_weight = true, .need_bias = true });
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
    var dx_only = try ctx.groupNormBackward(&x, &gy, 2, eps, .{ .need_input = true, .need_weight = false, .need_bias = false });
    defer dx_only.deinit();
    try std.testing.expect(dx_only.input != null);
    try std.testing.expect(dx_only.weight == null);
    try std.testing.expect(dx_only.bias == null);
    try std.testing.expectEqualSlices(f32, result.input.?.dataConst(), dx_only.input.?.dataConst());

    // gy must match x; groups must divide C; the affine weight must be [C].
    var bad_gy = try ctx.fromSlice(.f32, .{ 1, 2 }, &.{ 1, 2 });
    defer bad_gy.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.groupNormBackward(&x, &bad_gy, 2, eps, .{ .need_input = true, .need_weight = false, .need_bias = false }));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.groupNormBackward(&x, &gy, 3, eps, .{ .need_input = true, .need_weight = false, .need_bias = false }));
    var bad_w = try ctx.fromSlice(.f32, .{3}, &.{ 1, 2, 3 });
    defer bad_w.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.groupNormBackward(&x, &gy, 2, eps, .{ .weight = &bad_w, .need_input = true, .need_weight = false, .need_bias = false }));
}

// ---- Non-last-axis lane kernels vs the retired scalar strided arms ----
//
// The oracles below are the strided scalar loops the inner-lane kernels
// replaced (one scalar accumulator per output lane, stride `inner`), kept
// verbatim per forward/backward arm: the kernels promise the same
// per-element accumulation order, so the comparison is bitwise.

fn oracleRmsNormPlainStrided(input: []const f32, residual_data: ?[]const f32, output: []f32, outer: usize, axis_dim: usize, inner: usize, eps: f32) void {
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const value = input[base + axis_i * inner + inner_i];
                sumsq += value * value;
            }
            const scale_value = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const value = input[offset] * scale_value;
                output[offset] = if (residual_data) |r| r[offset] + value else value;
            }
        }
    }
}

fn oracleRmsNormMulStrided(input: []const f32, weights: []const f32, residual_data: ?[]const f32, output: []f32, outer: usize, axis_dim: usize, inner: usize, eps: f32) void {
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const value = input[base + axis_i * inner + inner_i];
                sumsq += value * value;
            }
            const scale_value = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                output[offset] = if (residual_data) |r|
                    r[offset] + input[offset] * scale_value * weights[axis_i]
                else
                    input[offset] * scale_value * weights[axis_i];
            }
        }
    }
}

fn oracleRmsNormBackwardInputWeightedStrided(input: []const f32, weights: []const f32, grad: []const f32, output: []f32, outer: usize, axis_dim: usize, inner: usize, eps: f32) void {
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            var dot_acc: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const value = input[offset];
                sumsq += value * value;
                dot_acc += grad[offset] * weights[axis_i] * value;
            }
            const rms_scale = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            const correction_scale = rms_scale * rms_scale * rms_scale * inv_axis_dim * dot_acc;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                output[offset] = grad[offset] * weights[axis_i] * rms_scale - input[offset] * correction_scale;
            }
        }
    }
}

fn oracleRmsNormBackwardInputPlainStrided(input: []const f32, gyd: []const f32, output: []f32, outer: usize, axis_dim: usize, inner: usize, eps: f32) void {
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            var dot_acc: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                const value = input[offset];
                sumsq += value * value;
                dot_acc += gyd[offset] * value;
            }
            const inv_rms = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            const correction = dot_acc * inv_axis_dim * inv_rms * inv_rms * inv_rms;
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                output[offset] = gyd[offset] * inv_rms - input[offset] * correction;
            }
        }
    }
}

fn oracleRmsNormBackwardWeightStrided(input: []const f32, grad: []const f32, output: []f32, outer: usize, axis_dim: usize, inner: usize, eps: f32) void {
    @memset(output, 0);
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const value = input[base + axis_i * inner + inner_i];
                sumsq += value * value;
            }
            const rms_scale = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                output[axis_i] += grad[offset] * input[offset] * rms_scale;
            }
        }
    }
}

fn oracleLayerNormStrided(input: []const f32, weights: ?[]const f32, biases: ?[]const f32, output: []f32, outer: usize, axis_dim: usize, inner: usize, eps: f32) void {
    const inv_axis_dim = 1 / @as(f32, @floatFromInt(axis_dim));
    for (0..outer) |outer_i| {
        const base = outer_i * axis_dim * inner;
        for (0..inner) |inner_i| {
            var sum_acc: f32 = 0;
            for (0..axis_dim) |axis_i| {
                sum_acc += input[base + axis_i * inner + inner_i];
            }
            const mean_value = sum_acc * inv_axis_dim;
            var sumsq: f32 = 0;
            for (0..axis_dim) |axis_i| {
                const centered = input[base + axis_i * inner + inner_i] - mean_value;
                sumsq += centered * centered;
            }
            const inv_sigma = 1 / @sqrt(sumsq * inv_axis_dim + eps);
            for (0..axis_dim) |axis_i| {
                const offset = base + axis_i * inner + inner_i;
                var value = (input[offset] - mean_value) * inv_sigma;
                if (weights) |w| value = value * w[axis_i];
                if (biases) |b| value = value + b[axis_i];
                output[offset] = value;
            }
        }
    }
}

fn testFillRandom(data: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (data) |*value| value.* = random.floatNorm(f32) * 3;
}

fn expectBitwiseF32(expected: []const f32, actual: []const f32) !void {
    try std.testing.expectEqualSlices(u32, @ptrCast(expected), @ptrCast(actual));
}

/// Every non-last axis of one shape, through rmsNorm (plain / weighted /
/// weighted + residual), rmsNormBackward (dx plain and weighted, dweight)
/// and layerNorm (plain / affine), against the strided oracles.
fn expectNormLaneKernelsMatchOracle(ctx: *ExecContext, comptime rank: usize, shape: [rank]usize, seed: u64) !void {
    const allocator = std.testing.allocator;
    var len: usize = 1;
    for (shape) |dim| len *= dim;
    const data = try allocator.alloc(f32, len);
    defer allocator.free(data);
    testFillRandom(data, seed);
    const grad = try allocator.alloc(f32, len);
    defer allocator.free(grad);
    testFillRandom(grad, seed + 1);
    const expected = try allocator.alloc(f32, len);
    defer allocator.free(expected);
    const eps: f32 = 1e-5;

    var x = try ctx.fromSlice(.f32, shape, data);
    defer x.deinit();
    var gy = try ctx.fromSlice(.f32, shape, grad);
    defer gy.deinit();

    inline for (0..rank - 1) |axis| {
        const axis_dim = shape[axis];
        var outer: usize = 1;
        for (0..axis) |dim_i| outer *= shape[dim_i];
        const inner = len / (outer * axis_dim);

        const weight_data = try allocator.alloc(f32, axis_dim);
        defer allocator.free(weight_data);
        testFillRandom(weight_data, seed + 2 + axis);
        const bias_data = try allocator.alloc(f32, axis_dim);
        defer allocator.free(bias_data);
        testFillRandom(bias_data, seed + 3 + axis);
        var weight = try ctx.fromSlice(.f32, .{axis_dim}, weight_data);
        defer weight.deinit();
        var bias = try ctx.fromSlice(.f32, .{axis_dim}, bias_data);
        defer bias.deinit();

        {
            var got = try ctx.rmsNorm(.f32, rank, &x, axis, eps, .{});
            defer got.deinit();
            oracleRmsNormPlainStrided(data, null, expected, outer, axis_dim, inner, eps);
            try expectBitwiseF32(expected, got.dataConst());
        }
        {
            var got = try ctx.rmsNorm(.f32, rank, &x, axis, eps, .{ .residual = &gy });
            defer got.deinit();
            oracleRmsNormPlainStrided(data, grad, expected, outer, axis_dim, inner, eps);
            try expectBitwiseF32(expected, got.dataConst());
        }
        {
            var got = try ctx.rmsNorm(.f32, rank, &x, axis, eps, .{ .weight = &weight });
            defer got.deinit();
            oracleRmsNormMulStrided(data, weight_data, null, expected, outer, axis_dim, inner, eps);
            try expectBitwiseF32(expected, got.dataConst());
        }
        {
            var got = try ctx.rmsNorm(.f32, rank, &x, axis, eps, .{ .weight = &weight, .residual = &gy });
            defer got.deinit();
            oracleRmsNormMulStrided(data, weight_data, grad, expected, outer, axis_dim, inner, eps);
            try expectBitwiseF32(expected, got.dataConst());
        }
        {
            var got = try ctx.rmsNormBackward(rank, &x, &gy, axis, eps, .{ .need_input = true, .need_weight = true });
            defer got.deinit();
            oracleRmsNormBackwardInputPlainStrided(data, grad, expected, outer, axis_dim, inner, eps);
            try expectBitwiseF32(expected, got.input.?.dataConst());
            oracleRmsNormBackwardWeightStrided(data, grad, expected[0..axis_dim], outer, axis_dim, inner, eps);
            try expectBitwiseF32(expected[0..axis_dim], got.weight.?.dataConst());
        }
        {
            var got = try ctx.rmsNormBackward(rank, &x, &gy, axis, eps, .{ .need_input = true, .weight = &weight });
            defer got.deinit();
            oracleRmsNormBackwardInputWeightedStrided(data, weight_data, grad, expected, outer, axis_dim, inner, eps);
            try expectBitwiseF32(expected, got.input.?.dataConst());
        }
        {
            var got = try ctx.layerNorm(.f32, rank, &x, axis, eps, .{});
            defer got.deinit();
            oracleLayerNormStrided(data, null, null, expected, outer, axis_dim, inner, eps);
            try expectBitwiseF32(expected, got.dataConst());
        }
        {
            var got = try ctx.layerNorm(.f32, rank, &x, axis, eps, .{ .weight = &weight, .bias = &bias });
            defer got.deinit();
            oracleLayerNormStrided(data, weight_data, bias_data, expected, outer, axis_dim, inner, eps);
            try expectBitwiseF32(expected, got.dataConst());
        }
    }
}

test "norm non-last-axis lane kernels are bitwise the strided scalar arms" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // Below the row-kernel dispatch threshold (serial lane kernel, odd lane
    // counts for the vector tails) and above it (lane ranges split across
    // the pool, tails again).
    try expectNormLaneKernelsMatchOracle(&ctx, 3, .{ 3, 5, 7 }, 0x9e11);
    try expectNormLaneKernelsMatchOracle(&ctx, 4, .{ 2, 3, 4, 5 }, 0x9e12);
    try expectNormLaneKernelsMatchOracle(&ctx, 3, .{ 2, 64, 1027 }, 0x9e13);
    try expectNormLaneKernelsMatchOracle(&ctx, 4, .{ 2, 3, 32, 701 }, 0x9e14);
}

test "rmsNorm dweight is bitwise identical for any thread count" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // Above the row-kernel threshold (pooled column partition) with a
    // column count that leaves a vector tail in some task splits.
    const rows = 260;
    const cols = 1013;
    const data = try std.testing.allocator.alloc(f32, rows * cols);
    defer std.testing.allocator.free(data);
    testFillRandom(data, 0x4d5e);
    const grad = try std.testing.allocator.alloc(f32, rows * cols);
    defer std.testing.allocator.free(grad);
    testFillRandom(grad, 0x4d5f);
    var x = try ctx.fromSlice(.f32, .{ rows, cols }, data);
    defer x.deinit();
    var gy = try ctx.fromSlice(.f32, .{ rows, cols }, grad);
    defer gy.deinit();

    const saved_threads = parallel.cpuThreadCount(parallel.vector_max_threads);
    defer parallel.setMaxThreads(saved_threads);

    // One task: the pooled dispatch degenerates and the serial row kernel
    // runs; every larger team must reproduce its bytes.
    parallel.setMaxThreads(1);
    var serial = try ctx.rmsNormBackward(2, &x, &gy, 1, 1e-5, .{ .need_input = false, .need_weight = true });
    defer serial.deinit();
    for ([_]usize{ 2, 3, 5, saved_threads }) |threads| {
        parallel.setMaxThreads(threads);
        var pooled = try ctx.rmsNormBackward(2, &x, &gy, 1, 1e-5, .{ .need_input = false, .need_weight = true });
        defer pooled.deinit();
        try expectBitwiseF32(serial.weight.?.dataConst(), pooled.weight.?.dataConst());
    }

    // The serial form itself is the row-order column accumulation.
    const expected = try std.testing.allocator.alloc(f64, cols);
    defer std.testing.allocator.free(expected);
    @memset(expected, 0);
    for (0..rows) |row_i| {
        const row = data[row_i * cols ..][0..cols];
        const row_g = grad[row_i * cols ..][0..cols];
        var sumsq: f64 = 0;
        for (row) |value| sumsq += @as(f64, value) * value;
        const inv_rms = 1 / @sqrt(sumsq / @as(f64, @floatFromInt(cols)) + 1e-5);
        for (expected, row_g, row) |*acc, g, value| acc.* += @as(f64, g) * value * inv_rms;
    }
    for (expected, serial.weight.?.dataConst()) |want, got| {
        try expectCloseToF64(want, got, 1e-4, 1e-4);
    }
}
