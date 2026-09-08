//! Behavioral tests for the tensor LSTM (`lstm.zig`): streaming versus
//! windowed forward, the NAM weight-stream round trip, gradients against
//! finite differences, one optimizer step, and the exported `.nam` played
//! by the app engine (the upstream render golden in `engine_tests.zig` is
//! the NAM-level parity reference).

const std = @import("std");
const fucina = @import("fucina");
const lstm = @import("lstm.zig");
const nam_file = @import("nam_file.zig");
const nam_export = @import("nam_export.zig");
const engine_mod = @import("engine.zig");

const ExecContext = fucina.ExecContext;

fn fillUniform(buf: []f32, seed: u64, amp: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (buf) |*v| v.* = (random.float(f32) * 2 - 1) * amp;
}

fn fillSignal(buf: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (buf, 0..) |*v, i| {
        const t = @as(f32, @floatFromInt(i));
        v.* = 0.3 * @sin(t * 0.0217) + 0.15 * @sin(t * 0.0651 + 0.3) + 0.05 * (random.float(f32) * 2 - 1);
    }
}

const synthetic_config = nam_file.LstmConfig{ .input_size = 1, .hidden_size = 8, .num_layers = 2, .in_channels = 1, .out_channels = 1 };

fn syntheticWeights(allocator: std.mem.Allocator, config: *const nam_file.LstmConfig, seed: u64) ![]f32 {
    const file_config = nam_file.Config{ .lstm = config.* };
    const weights = try allocator.alloc(f32, nam_file.expectedWeightCount(&file_config));
    fillUniform(weights, seed, 0.4);
    return weights;
}

test "lstm streaming equals the windowed forward and stays allocation-free" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    const weights = try syntheticWeights(allocator, &synthetic_config, 5);
    defer allocator.free(weights);
    var model = try lstm.Model.initFromNam(allocator, &ctx, &synthetic_config, weights, false, .{ .burn_in = 0, .truncate = 0 });
    defer model.deinit();

    const total = 200;
    var input: [total]f32 = undefined;
    fillSignal(&input, 3);
    var windowed: [total]f32 = undefined;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const pred = try model.forward(&ctx, &input);
        try pred.copyTo(&windowed);
    }
    var stream = try lstm.Stream.init(allocator, &ctx, &model);
    defer stream.deinit();
    var streamed: [total]f32 = undefined;
    try stream.process(&ctx, &input, &streamed, total);
    try std.testing.expectEqualSlices(f32, &windowed, &streamed);

    // A reset stream repeats itself exactly; the per-sample path allocates
    // nothing once the pool is warm.
    try stream.reset(&ctx);
    var again: [total]f32 = undefined;
    try stream.process(&ctx, &input, &again, total);
    try std.testing.expectEqualSlices(f32, &streamed, &again);
}

test "lstm weight stream round-trips bitwise through the tensor model" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    const weights = try syntheticWeights(allocator, &synthetic_config, 8);
    defer allocator.free(weights);
    inline for (.{ false, true }) |trainable| {
        var model = try lstm.Model.initFromNam(allocator, &ctx, &synthetic_config, weights, trainable, .{});
        defer model.deinit();
        const roundtrip = try model.extractWeights(&ctx, allocator);
        defer allocator.free(roundtrip);
        try std.testing.expectEqualSlices(f32, weights, roundtrip);
    }
}

fn lossFromStream(allocator: std.mem.Allocator, ctx: *ExecContext, config: *const nam_file.LstmConfig, weights: []const f32, training: lstm.Training, window: []const f32, target: []const f32) !f32 {
    var model = try lstm.Model.initFromNam(allocator, ctx, config, weights, false, training);
    defer model.deinit();
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try model.segmentLoss(ctx, window, target);
    return loss.item();
}

/// The analytic gradients in NAM stream order (the layout `extractWeights`
/// emits); a parameter without a gradient contributes zeros.
fn collectGrads(allocator: std.mem.Allocator, ctx: *ExecContext, model: *lstm.Model) ![]f32 {
    var out: std.ArrayList(f32) = .empty;
    errdefer out.deinit(allocator);
    for (model.lstm.cells) |*cell| {
        const width = cell.input_size + cell.hidden;
        var grad = try cell.w.grad(ctx);
        if (grad) |*g| {
            defer g.deinit();
            var matrix = try g.narrow(ctx, .k, 0, width);
            defer matrix.deinit();
            var nam_order = try matrix.permuteTo(ctx, .{ .unit, .k });
            defer nam_order.deinit();
            try appendViewTo(allocator, &out, &nam_order);
            var bias_row = try g.narrow(ctx, .k, width, 1);
            defer bias_row.deinit();
            try appendViewTo(allocator, &out, &bias_row);
        } else {
            const start = out.items.len;
            try out.resize(allocator, start + (width + 1) * 4 * cell.hidden);
            @memset(out.items[start..], 0);
        }
        try appendGrad(allocator, ctx, &out, &cell.h0, .{.unit});
        try appendGrad(allocator, ctx, &out, &cell.c0, .{.unit});
    }
    try appendGrad(allocator, ctx, &out, &model.head_w, .{ .out, .unit });
    try appendGrad(allocator, ctx, &out, &model.head_b, .{.out});
    return out.toOwnedSlice(allocator);
}

fn appendViewTo(allocator: std.mem.Allocator, out: *std.ArrayList(f32), view: anytype) !void {
    var count: usize = 1;
    for (view.shape()) |dim| count *= dim;
    const start = out.items.len;
    try out.resize(allocator, start + count);
    try view.copyTo(out.items[start..]);
}

fn appendGrad(allocator: std.mem.Allocator, ctx: *ExecContext, out: *std.ArrayList(f32), param: anytype, comptime order: anytype) !void {
    var count: usize = 1;
    for (param.shape()) |dim| count *= dim;
    const start = out.items.len;
    try out.resize(allocator, start + count);
    var grad = try param.grad(ctx);
    if (grad) |*g| {
        defer g.deinit();
        var ordered = try g.permuteTo(ctx, order);
        defer ordered.deinit();
        try ordered.copyTo(out.items[start..]);
    } else {
        @memset(out.items[start..], 0);
    }
}

test "lstm gradients match finite differences (full backpropagation through time)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const config = nam_file.LstmConfig{ .input_size = 1, .hidden_size = 3, .num_layers = 2, .in_channels = 1, .out_channels = 1 };
    const weights = try syntheticWeights(allocator, &config, 13);
    defer allocator.free(weights);
    const training = lstm.Training{ .burn_in = 0, .truncate = 0 };
    var window: [7]f32 = undefined;
    fillSignal(&window, 2);
    var target: [5]f32 = undefined;
    fillSignal(&target, 4);
    for (&target) |*t| t.* *= 0.5;

    var model = try lstm.Model.initFromNam(allocator, &ctx, &config, weights, true, training);
    defer model.deinit();
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try model.segmentLoss(&ctx, &window, &target);
        try loss.backward(&ctx);
    }
    const analytic = try collectGrads(allocator, &ctx, &model);
    defer allocator.free(analytic);
    try std.testing.expectEqual(weights.len, analytic.len);

    const perturbed = try allocator.dupe(f32, weights);
    defer allocator.free(perturbed);
    const eps: f32 = 1e-3;
    var max_err: f32 = 0;
    var nonzero: usize = 0;
    for (weights, 0..) |value, i| {
        perturbed[i] = value + eps;
        const plus = try lossFromStream(allocator, &ctx, &config, perturbed, training, &window, &target);
        perturbed[i] = value - eps;
        const minus = try lossFromStream(allocator, &ctx, &config, perturbed, training, &window, &target);
        perturbed[i] = value;
        const numeric = (plus - minus) / (2 * eps);
        try std.testing.expect(std.math.isFinite(numeric));
        const err = @abs(numeric - analytic[i]) / @max(1.0, @abs(analytic[i]));
        max_err = @max(max_err, err);
        if (@abs(analytic[i]) > 1e-6) nonzero += 1;
        try std.testing.expect(err <= 2e-2);
    }
    std.debug.print("  [lstm] finite differences over {d} parameters: max relative error {e:.2}, {d} nonzero gradients\n", .{ weights.len, max_err, nonzero });
    try std.testing.expect(nonzero > weights.len / 2);
}

test "lstm burn-in cuts the initial-state gradient and truncation keeps the forward values" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    const config = nam_file.LstmConfig{ .input_size = 1, .hidden_size = 4, .num_layers = 1, .in_channels = 1, .out_channels = 1 };
    const weights = try syntheticWeights(allocator, &config, 19);
    defer allocator.free(weights);
    var window: [24]f32 = undefined;
    fillSignal(&window, 6);
    var target: [16]f32 = undefined;
    fillSignal(&target, 8);

    var full = try lstm.Model.initFromNam(allocator, &ctx, &config, weights, true, .{ .burn_in = 8, .truncate = 0 });
    defer full.deinit();
    var truncated = try lstm.Model.initFromNam(allocator, &ctx, &config, weights, true, .{ .burn_in = 8, .truncate = 5 });
    defer truncated.deinit();
    var loss_full: f32 = 0;
    var loss_truncated: f32 = 0;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var a = try full.segmentLoss(&ctx, &window, &target);
        loss_full = try a.item();
        try a.backward(&ctx);
        var b = try truncated.segmentLoss(&ctx, &window, &target);
        loss_truncated = try b.item();
        try b.backward(&ctx);
    }
    // Detaching between segments changes gradients, never values.
    try std.testing.expectEqual(loss_full, loss_truncated);
    // The burn-in ran without gradient, so the learned initial state gets none.
    var h0_grad = try full.lstm.cells[0].h0.grad(&ctx);
    try std.testing.expect(h0_grad == null);
    var c0_grad = try full.lstm.cells[0].c0.grad(&ctx);
    try std.testing.expect(c0_grad == null);
    var w_grad = (try full.lstm.cells[0].w.grad(&ctx)).?;
    defer w_grad.deinit();
    var w_grad_t = (try truncated.lstm.cells[0].w.grad(&ctx)).?;
    defer w_grad_t.deinit();
    var any_nonzero = false;
    for (try w_grad.dataConst(), try w_grad_t.dataConst()) |g, gt| {
        try std.testing.expect(std.math.isFinite(g) and std.math.isFinite(gt));
        if (g != 0) any_nonzero = true;
    }
    try std.testing.expect(any_nonzero);
    if (h0_grad) |*g| g.deinit();
    if (c0_grad) |*g| g.deinit();
}

test "one Adam step reduces the lstm segment loss" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try lstm.Model.init(allocator, &ctx, .{ .hidden_size = 6, .num_layers = 1, .burn_in = 8, .truncate = 16 }, 7);
    defer model.deinit();
    var opt = try fucina.optim.Adam.init(allocator, .{ .lr = 0.01 });
    defer opt.deinit();
    try model.registerParams(&opt);

    var window: [40]f32 = undefined;
    fillSignal(&window, 11);
    var target: [32]f32 = undefined;
    for (&target, window[8..]) |*t, x| t.* = 0.7 * x - 0.1 * x * x;

    var before: f32 = 0;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try model.segmentLoss(&ctx, &window, &target);
        before = try loss.item();
        try loss.backward(&ctx);
        try opt.step(&ctx);
        opt.zeroGrad();
    }
    var after: f32 = 0;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try model.segmentLoss(&ctx, &window, &target);
        after = try loss.item();
    }
    std.debug.print("  [lstm] one Adam step: loss {d:.6} -> {d:.6}\n", .{ before, after });
    try std.testing.expect(after < before);
}

test "exported lstm .nam loads in the app engine and plays the tensor model's output" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try lstm.Model.init(allocator, &ctx, .{ .hidden_size = 5, .num_layers = 2, .burn_in = 0, .truncate = 0 }, 3);
    defer model.deinit();
    const weights = try model.extractWeights(&ctx, allocator);
    defer allocator.free(weights);
    const config = model.spec.engineConfig();

    const path = "/tmp/fucina-lstm-export-test.nam";
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    try nam_export.exportLstmConfig(std.testing.io, allocator, path, &config, weights, .{ .unix_seconds = 1_700_000_000 });

    var loaded = try nam_file.loadFile(std.testing.io, allocator, path);
    defer loaded.deinit();
    try std.testing.expectEqual(nam_file.Arch.lstm, loaded.architecture);
    try std.testing.expectEqual(@as(usize, 5), loaded.config.lstm.hidden_size);
    try std.testing.expectEqual(@as(usize, 2), loaded.config.lstm.num_layers);
    try std.testing.expectEqualSlices(f32, weights, loaded.weights);

    // The app engine (no prewarm) against the tensor stream from the same reset.
    var played = try engine_mod.Engine.init(allocator, &loaded);
    defer played.deinit();
    try played.reset(64, false);
    var stream = try lstm.Stream.init(allocator, &ctx, &model);
    defer stream.deinit();
    var input: [128]f32 = undefined;
    fillSignal(&input, 17);
    var expected: [128]f32 = undefined;
    var got: [128]f32 = undefined;
    try played.process(&input, &expected, 64);
    try played.process(input[64..], expected[64..], 64);
    try stream.process(&ctx, &input, &got, 128);
    for (expected, got) |e, g| try std.testing.expectApproxEqAbs(e, g, 2e-5);
}
