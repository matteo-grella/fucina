//! Facade-level tests for the eager runtime (`exec.zig`): buffer reuse and
//! borrowed slices, cross-entropy against naive references, broadcast
//! materialization, typed allocation, and the chunked parallel materialize.
//! Per-module kernel tests live beside their modules in `src/exec/`.

const std = @import("std");
const exec = @import("exec.zig");
const parallel = @import("parallel.zig");
const tensor = @import("tensor.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;
const ExecContext = exec.ExecContext;
const LayoutClass = exec.LayoutClass;
const CrossEntropyOptions = exec.CrossEntropyOptions;
const Reduction = exec.Reduction;

const util = @import("exec/test_util.zig");
const expectCloseToF64 = util.expectCloseToF64;

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

test "exec context reuses released output buffers" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(.f32, &.{3}, &.{ 1, 2, 3 });
    defer a.deinit();
    var b = try ctx.fromSlice(.f32, &.{3}, &.{ 4, 5, 6 });
    defer b.deinit();

    var first = try ctx.elementwise(.f32, .add, &a, &b);
    const first_buffer = first.buffer;
    first.deinit();

    var second = try ctx.elementwise(.f32, .add, &a, &b);
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
    var x = try ctx.fromBorrowedSlice(.f32, .{ 2, 2 }, values[0..]);
    defer x.deinit();

    values[3] = 40;
    try std.testing.expectEqual(@as(f32, 40), x.dataConst()[3]);

    var doubled = try ctx.elementwise(.f32, .add, &x, &x);
    defer doubled.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6, 80 }, doubled.dataConst());
}

test "buffer pool reuses bucket-rounded buffers across many small temporaries" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var a = try ctx.fromSlice(.f32, &.{3}, &.{ 1, 2, 3 });
    defer a.deinit();
    var b = try ctx.fromSlice(.f32, &.{3}, &.{ 4, 5, 6 });
    defer b.deinit();

    for (0..100) |_| {
        var y = try ctx.elementwise(.f32, .add, &a, &b);
        try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, y.dataConst());
        y.deinit();
    }

    try std.testing.expect(ctx.buffers.cachedBuffers() >= 1);
    try std.testing.expectEqual(@as(usize, 2), ctx.buffers.outstandingBuffers());
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
                    try ctx.fromSlice(.f32, .{ outer, class_count }, data)
                else
                    try ctx.fromSlice(.f32, .{ outer, class_count, inner }, data);
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

                            var loss = try ctx.crossEntropyLoss(rank, &logits, 1, labels, options);
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

                            const up: exec.CrossEntropyUpstream = if (reduction == .none)
                                .{ .rows = .{ .per_row = per_row, .scale = upstream } }
                            else
                                .{ .scale = upstream };
                            var grad = try ctx.crossEntropyBackward(rank, &logits, 1, labels, options, up);
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
                    try ctx.fromSlice(.f32, .{ outer, class_count }, data)
                else
                    try ctx.fromSlice(.f32, .{ outer, class_count, inner }, data);
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
                            const up: exec.CrossEntropyUpstream = if (reduction == .none)
                                .{ .rows = .{ .per_row = per_row, .scale = upstream } }
                            else
                                .{ .scale = upstream };
                            var stats_options = options;
                            stats_options.row_stats = row_stats;

                            var plain_loss = try ctx.crossEntropyLoss(rank, &logits, 1, labels, options);
                            defer plain_loss.deinit();
                            var stats_loss = try ctx.crossEntropyLoss(rank, &logits, 1, labels, stats_options);
                            defer stats_loss.deinit();
                            try std.testing.expectEqualSlices(f32, plain_loss.dataConst(), stats_loss.dataConst());

                            var grad_recompute = try ctx.crossEntropyBackward(rank, &logits, 1, labels, options, up);
                            defer grad_recompute.deinit();
                            var grad_stats = try ctx.crossEntropyBackward(rank, &logits, 1, labels, stats_options, up);
                            defer grad_stats.deinit();
                            try std.testing.expectEqualSlices(f32, grad_recompute.dataConst(), grad_stats.dataConst());
                        }
                    }
                }
            }
        }
    }

    // Stats length is validated (must be 2 * position count).
    var logits = try ctx.fromSlice(.f32, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer logits.deinit();
    var short_stats = [_]f32{ 0, 0 };
    try std.testing.expectError(tensor.TensorError.InvalidDataLength, ctx.crossEntropyLoss(2, &logits, 1, &.{ 0, 1 }, .{ .row_stats = &short_stats }));
    var bad_stats = [_]f32{ 0, 0, 0 };
    try std.testing.expectError(tensor.TensorError.InvalidDataLength, ctx.crossEntropyBackward(2, &logits, 1, &.{ 0, 1 }, .{ .row_stats = &bad_stats }, .{ .scale = 1 }));
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

        var x = try ctx.fromSlice(.f32, .{ rows, in_dim }, x_data);
        defer x.deinit();
        var w = try ctx.fromSlice(.f32, .{ class_count, in_dim }, w_data);
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
                    var stats_options = options;
                    stats_options.row_stats = row_stats;
                    var loss = try ctx.crossEntropyLoss(2, &logits, 1, labels, stats_options);
                    loss.deinit();

                    var gy = if (reduction == .none)
                        try ctx.fromSlice(.f32, .{rows}, per_row)
                    else
                        try ctx.fromSlice(.f32, .{1}, &.{0.75});
                    defer gy.deinit();

                    // Composed reference (before the fused call eats logits).
                    const up: exec.CrossEntropyUpstream = if (reduction == .none)
                        .{ .rows = .{ .per_row = per_row } }
                    else
                        .{ .scale = 0.75 };
                    var dlogits = try ctx.crossEntropyBackward(2, &logits, 1, labels, stats_options, up);
                    defer dlogits.deinit();
                    var dx_ref = try ctx.matmul(.f32, &dlogits, &w);
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
        var loss = try ctx.crossEntropyLoss(2, &logits, 1, labels, .{ .row_stats = row_stats });
        loss.deinit();
        var gy = try ctx.fromSlice(.f32, .{1}, &.{1});
        defer gy.deinit();
        var dlogits = try ctx.crossEntropyBackward(2, &logits, 1, labels, .{ .row_stats = row_stats }, .{ .scale = 1 });
        defer dlogits.deinit();
        var dx_ref = try ctx.matmul(.f32, &dlogits, &w);
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

    var logits = try ctx.fromSlice(.f32, .{ 3, 5 }, &.{
        1,  2, 3,  4, 5,
        -1, 0, 1,  2, 3,
        2,  2, -2, 0, 1,
    });
    defer logits.deinit();

    // Every position ignored (in-range ignore_index): loss 0, grads exactly 0.
    // (Deliberate divergence from PyTorch's NaN.)
    const all_ignored = CrossEntropyOptions{ .ignore_index = 2, .reduction = .mean };
    var loss = try ctx.crossEntropyLoss(2, &logits, 1, &.{ 2, 2, 2 }, all_ignored);
    defer loss.deinit();
    try std.testing.expectEqual(@as(f32, 0), loss.item());
    var grad = try ctx.crossEntropyBackward(2, &logits, 1, &.{ 2, 2, 2 }, all_ignored, .{ .scale = 1 });
    defer grad.deinit();
    for (grad.dataConst()) |value| try std.testing.expectEqual(@as(f32, 0), value);

    // An in-range ignore_index drops exactly the matching positions, and the
    // mean denominator counts only the remaining ones.
    var partial = try ctx.crossEntropyLoss(2, &logits, 1, &.{ 4, 2, 0 }, all_ignored);
    defer partial.deinit();
    var row0 = try ctx.fromSlice(.f32, .{ 1, 5 }, &.{ 1, 2, 3, 4, 5 });
    defer row0.deinit();
    var row2 = try ctx.fromSlice(.f32, .{ 1, 5 }, &.{ 2, 2, -2, 0, 1 });
    defer row2.deinit();
    var loss0 = try ctx.crossEntropyLoss(2, &row0, 1, &.{4}, .{ .reduction = .sum });
    defer loss0.deinit();
    var loss2 = try ctx.crossEntropyLoss(2, &row2, 1, &.{0}, .{ .reduction = .sum });
    defer loss2.deinit();
    try std.testing.expectApproxEqAbs((loss0.item() + loss2.item()) / 2, partial.item(), 1e-6);

    // Labels must be < class_count or == ignore_index.
    try std.testing.expectError(tensor.TensorError.IndexOutOfBounds, ctx.crossEntropyLoss(2, &logits, 1, &.{ 0, 5, 1 }, .{}));
    try std.testing.expectError(tensor.TensorError.IndexOutOfBounds, ctx.crossEntropyBackward(2, &logits, 1, &.{ 0, 9, 1 }, .{ .ignore_index = 7 }, .{ .scale = 1 }));
    // label_smoothing must be in [0, 1).
    try std.testing.expectError(tensor.TensorError.InvalidShape, ctx.crossEntropyLoss(2, &logits, 1, &.{ 0, 1, 2 }, .{ .label_smoothing = 1 }));
    // .none requires a per-row upstream of matching length; mean/sum forbid it.
    try std.testing.expectError(tensor.TensorError.InvalidDataLength, ctx.crossEntropyBackward(2, &logits, 1, &.{ 0, 1, 2 }, .{ .reduction = .none }, .{ .scale = 1 }));
    try std.testing.expectError(tensor.TensorError.InvalidDataLength, ctx.crossEntropyBackward(2, &logits, 1, &.{ 0, 1, 2 }, .{}, .{ .rows = .{ .per_row = &.{ 1, 1, 1 } } }));

    // `.{}` is the mean over all positions.
    var mean_default = try ctx.crossEntropyLoss(2, &logits, 1, &.{ 4, 2, 0 }, .{});
    defer mean_default.deinit();
    var mean_explicit = try ctx.crossEntropyLoss(2, &logits, 1, &.{ 4, 2, 0 }, .{ .reduction = .mean });
    defer mean_explicit.deinit();
    try std.testing.expectEqual(mean_default.item(), mean_explicit.item());
}

// These drive the substrate alloc primitives (ctx.zeros,
// ctx.scalar) and the elementwise reduce-broadcast VJP directly, so they
// belong beside the other exec_tests rather than inline in exec.zig.
test "exec context reuses buffers for arbitrary broadcast materialization" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var x = try ctx.zeros(.f32, .{ 2, 4, 3 });
    defer x.deinit();
    var middle = try ctx.fromSlice(.f32, .{ 2, 1, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer middle.deinit();
    var middle_b = try ctx.broadcastTo(&middle, .{ 2, 4, 3 });
    defer middle_b.deinit();

    try std.testing.expectEqual(LayoutClass.arbitrary, ctx.classify(&middle_b));
    try std.testing.expectEqual(@as(usize, 2), ctx.buffers.outstandingBuffers());
    try std.testing.expectEqual(@as(usize, 0), ctx.buffers.cachedBuffers());

    var first = try ctx.add(.f32, 3, &x, &middle_b);
    try std.testing.expectEqualSlices(f32, &.{
        1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
        4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
    }, first.dataConst());
    try std.testing.expectEqual(@as(usize, 3), ctx.buffers.outstandingBuffers());
    try std.testing.expect(ctx.buffers.cachedBuffers() >= 1);
    first.deinit();

    try std.testing.expectEqual(@as(usize, 2), ctx.buffers.outstandingBuffers());
    const cached_after_first = ctx.buffers.cachedBuffers();
    try std.testing.expect(cached_after_first >= 2);

    var second = try ctx.add(.f32, 3, &x, &middle_b);
    second.deinit();

    try std.testing.expectEqual(@as(usize, 2), ctx.buffers.outstandingBuffers());
    try std.testing.expectEqual(cached_after_first, ctx.buffers.cachedBuffers());
}

test "exec context allocates typed non-f32 tensors without using f32 kernels" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var ids = try ctx.fromSlice(.u16, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ids.deinit();
    try std.testing.expect(@TypeOf(ids).dtype == .u16);
    try std.testing.expectEqualSlices(u16, &.{ 1, 2, 3, 4, 5, 6 }, ids.dataConst());

    var flags = try ctx.ones(.bool, &.{3});
    defer flags.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, flags.dataConst());

    var scalar_id = try ctx.scalar(.i64, 42);
    defer scalar_id.deinit();
    try std.testing.expectEqual(@as(i64, 42), scalar_id.item());
}
// native and -Dbackend=scalar ---

test "materialize of a large permuted view goes through the chunked parallel copy" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // 768x512 transposed view: 393216 elements, above the parallel
    // materialize threshold, innermost axis strided.
    const rows: usize = 768;
    const cols: usize = 512;
    var x = try ctx.empty(.f32, .{ rows, cols });
    defer x.deinit();
    for (x.data(), 0..) |*v, i| v.* = @floatFromInt(i % 1013);
    var t = try x.viewWithStrides(&.{ cols, rows }, &.{ 1, cols });
    defer t.deinit();

    var m = try ctx.materialize(.f32, &t);
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
