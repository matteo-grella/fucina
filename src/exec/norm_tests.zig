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
const LayoutClass = exec.LayoutClass;
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
                    var y_plain = try ctx.layerNorm(rank, &x, 1, eps);
                    defer y_plain.deinit();
                    for (y_plain.dataConst(), ref_plain) |got, want| {
                        try expectCloseToF64(want, got, 5e-4, 5e-5);
                    }

                    const ref_affine = try testNaiveLayerNorm(allocator, data, outer, axis_dim, inner, weights, biases, eps);
                    defer allocator.free(ref_affine);
                    var y_affine = try ctx.layerNormAffine(rank, &x, &w, &b, 1, eps);
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
            var gx_plain = try ctx.layerNormBackward(rank, &x, &gy, 1, eps);
            defer gx_plain.deinit();
            for (gx_plain.dataConst(), ref_plain.dx) |got, want| {
                try expectCloseToF64(want, got, 5e-4, 5e-5);
            }

            // Affine dx + dweight + dbias.
            var ref_affine = try testNaiveLayerNormBackward(allocator, data, grad, outer, axis_dim, inner, weights, eps);
            defer ref_affine.deinit(allocator);
            var full = try ctx.layerNormAffineBackward(rank, &x, &w, &gy, 1, eps, true, true, true);
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
            var weight_only = try ctx.layerNormAffineBackward(rank, &x, &w, &gy, 1, eps, false, true, false);
            defer weight_only.deinit();
            try std.testing.expect(weight_only.input == null);
            try std.testing.expect(weight_only.bias == null);
            try std.testing.expectEqualSlices(f32, full.weight.?.dataConst(), weight_only.weight.?.dataConst());

            var bias_only = try ctx.layerNormAffineBackward(rank, &x, &w, &gy, 1, eps, false, false, true);
            defer bias_only.deinit();
            try std.testing.expect(bias_only.input == null);
            try std.testing.expect(bias_only.weight == null);
            try std.testing.expectEqualSlices(f32, full.bias.?.dataConst(), bias_only.bias.?.dataConst());

            // Bitwise determinism across runs (the big case exercises the
            // parallel dx dispatch; dweight/dbias are one serial row pass,
            // bitwise identical for any thread count by construction).
            var again = try ctx.layerNormAffineBackward(rank, &x, &w, &gy, 1, eps, true, true, true);
            defer again.deinit();
            try std.testing.expectEqualSlices(f32, full.input.?.dataConst(), again.input.?.dataConst());
            try std.testing.expectEqualSlices(f32, full.weight.?.dataConst(), again.weight.?.dataConst());
            try std.testing.expectEqualSlices(f32, full.bias.?.dataConst(), again.bias.?.dataConst());

            // Forward determinism on the same shapes.
            var y_one = try ctx.layerNormAffine(rank, &x, &w, &w, 1, eps);
            defer y_one.deinit();
            var y_two = try ctx.layerNormAffine(rank, &x, &w, &w, 1, eps);
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
    var y1 = try ctx.groupNorm(&x, 1, eps, null, null);
    defer y1.deinit();
    const inv1 = 1.0 / @sqrt(@as(f32, 1.25) + eps);
    const want1 = [_]f32{ -1.5 * inv1, -0.5 * inv1, 0.5 * inv1, 1.5 * inv1 };
    for (want1, y1.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);

    // G=C: per-channel over time (InstanceNorm over T; the HuBERT layer-0
    // configuration). col0 = {1,3}: mean 2, var 1; col1 = {2,4}: mean 3, var 1.
    var y2 = try ctx.groupNorm(&x, 2, eps, null, null);
    defer y2.deinit();
    const inv2 = 1.0 / @sqrt(@as(f32, 1.0) + eps);
    const want2 = [_]f32{ -inv2, -inv2, inv2, inv2 };
    for (want2, y2.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-6);

    // Affine applied AFTER normalization: y*w + b.
    var wt = try ctx.fromSlice(.f32, .{2}, &.{ 2.0, 3.0 });
    defer wt.deinit();
    var bt = try ctx.fromSlice(.f32, .{2}, &.{ 10.0, 20.0 });
    defer bt.deinit();
    var y3 = try ctx.groupNorm(&x, 2, eps, &wt, &bt);
    defer y3.deinit();
    const want3 = [_]f32{ -inv2 * 2 + 10, -inv2 * 3 + 20, inv2 * 2 + 10, inv2 * 3 + 20 };
    for (want3, y3.dataConst()) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);

    // groups must divide C; affine vectors must be [C].
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.groupNorm(&x, 3, eps, null, null));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.groupNorm(&x, 0, eps, null, null));
    var bad_w = try ctx.fromSlice(.f32, .{3}, &.{ 1, 2, 3 });
    defer bad_w.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.groupNorm(&x, 2, eps, &bad_w, null));
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

    var result = try ctx.groupNormBackward(&x, &gy, 2, eps, null, true, true, true);
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
    var dx_only = try ctx.groupNormBackward(&x, &gy, 2, eps, null, true, false, false);
    defer dx_only.deinit();
    try std.testing.expect(dx_only.input != null);
    try std.testing.expect(dx_only.weight == null);
    try std.testing.expect(dx_only.bias == null);
    try std.testing.expectEqualSlices(f32, result.input.?.dataConst(), dx_only.input.?.dataConst());

    // gy must match x; groups must divide C; the affine weight must be [C].
    var bad_gy = try ctx.fromSlice(.f32, .{ 1, 2 }, &.{ 1, 2 });
    defer bad_gy.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.groupNormBackward(&x, &bad_gy, 2, eps, null, true, false, false));
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.groupNormBackward(&x, &gy, 3, eps, null, true, false, false));
    var bad_w = try ctx.fromSlice(.f32, .{3}, &.{ 1, 2, 3 });
    defer bad_w.deinit();
    try std.testing.expectError(tensor.TensorError.ShapeMismatch, ctx.groupNormBackward(&x, &gy, 2, eps, &bad_w, true, false, false));
}
