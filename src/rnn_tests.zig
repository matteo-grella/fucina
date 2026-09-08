//! Behavioral tests for `rnn.zig`: the LSTM step against a scalar reference
//! of PyTorch's equations, streaming versus the windowed forward, the
//! stacked-layout round trip, gradients against finite differences, and the
//! burn-in / truncation semantics.

const std = @import("std");
const rnn = @import("rnn.zig");
const ag = @import("ag.zig");
const exec_mod = @import("exec.zig");

const ExecContext = exec_mod.ExecContext;
const Tensor = ag.Tensor;

fn fillUniform(buf: []f32, seed: u64, amp: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (buf) |*v| v.* = (random.float(f32) * 2 - 1) * amp;
}

/// One layer's parameters in PyTorch's stacked layout, on the host.
const HostLayer = struct {
    stacked: []f32, // [4H, in + H]
    bias: []f32, // [4H]
    h0: []f32,
    c0: []f32,
    input_size: usize,
    hidden: usize,

    fn init(allocator: std.mem.Allocator, input_size: usize, hidden: usize, seed: u64) !HostLayer {
        const width = input_size + hidden;
        const self = HostLayer{
            .stacked = try allocator.alloc(f32, 4 * hidden * width),
            .bias = try allocator.alloc(f32, 4 * hidden),
            .h0 = try allocator.alloc(f32, hidden),
            .c0 = try allocator.alloc(f32, hidden),
            .input_size = input_size,
            .hidden = hidden,
        };
        fillUniform(self.stacked, seed, 0.5);
        fillUniform(self.bias, seed + 1, 0.3);
        fillUniform(self.h0, seed + 2, 0.4);
        fillUniform(self.c0, seed + 3, 0.4);
        return self;
    }

    fn deinit(self: *HostLayer, allocator: std.mem.Allocator) void {
        allocator.free(self.stacked);
        allocator.free(self.bias);
        allocator.free(self.h0);
        allocator.free(self.c0);
    }

    fn cell(self: *const HostLayer, ctx: *ExecContext, trainable: bool) !rnn.LstmCell {
        var stacked = try rnn.StackedWeight.fromBorrowedConstSlice(ctx, .{ 4 * self.hidden, self.input_size + self.hidden }, self.stacked);
        defer stacked.deinit();
        var bias = try rnn.Units.fromBorrowedConstSlice(ctx, .{4 * self.hidden}, self.bias);
        defer bias.deinit();
        var h0 = try rnn.Units.fromBorrowedConstSlice(ctx, .{self.hidden}, self.h0);
        defer h0.deinit();
        var c0 = try rnn.Units.fromBorrowedConstSlice(ctx, .{self.hidden}, self.c0);
        defer c0.deinit();
        return rnn.LstmCell.fromStacked(ctx, &stacked, &bias, &h0, &c0, trainable);
    }
};

fn buildLstm(allocator: std.mem.Allocator, ctx: *ExecContext, layers: []const HostLayer, trainable: bool) !rnn.Lstm {
    const cells = try allocator.alloc(rnn.LstmCell, layers.len);
    errdefer allocator.free(cells);
    var built: usize = 0;
    errdefer for (cells[0..built]) |*c| c.deinit();
    for (cells, layers) |*c, *layer| {
        c.* = try layer.cell(ctx, trainable);
        built += 1;
    }
    return rnn.Lstm.fromCells(allocator, ctx, cells);
}

fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

/// PyTorch's LSTM equations in f64 over the host layout, one step.
fn referenceStep(layer: *const HostLayer, x: []const f64, h: []f64, c: []f64, allocator: std.mem.Allocator) !void {
    const hidden = layer.hidden;
    const width = layer.input_size + hidden;
    const gates = try allocator.alloc(f64, 4 * hidden);
    defer allocator.free(gates);
    for (0..4 * hidden) |row| {
        var acc: f64 = layer.bias[row];
        for (0..layer.input_size) |i| acc += @as(f64, layer.stacked[row * width + i]) * x[i];
        for (0..hidden) |j| acc += @as(f64, layer.stacked[row * width + layer.input_size + j]) * h[j];
        gates[row] = acc;
    }
    for (0..hidden) |j| {
        const ig = sigmoid(gates[j]);
        const fg = sigmoid(gates[hidden + j]);
        const gg = std.math.tanh(gates[2 * hidden + j]);
        const og = sigmoid(gates[3 * hidden + j]);
        c[j] = fg * c[j] + ig * gg;
        h[j] = og * std.math.tanh(c[j]);
    }
}

test "lstm step matches PyTorch's equations in f64 over a two-layer stack" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var layers = [_]HostLayer{ try HostLayer.init(allocator, 2, 3, 10), try HostLayer.init(allocator, 3, 3, 20) };
    defer for (&layers) |*l| l.deinit(allocator);
    var lstm = try buildLstm(allocator, &ctx, &layers, false);
    defer lstm.deinit();
    var stream = try rnn.Lstm.Stream.init(allocator, &ctx, &lstm);
    defer stream.deinit();

    // Reference state, f64.
    var h = [_][3]f64{ undefined, undefined };
    var c = [_][3]f64{ undefined, undefined };
    for (0..2) |l| for (0..3) |j| {
        h[l][j] = layers[l].h0[j];
        c[l][j] = layers[l].c0[j];
    };
    var x_t = try rnn.Input.zeros(&ctx, .{2});
    defer x_t.deinit();
    var max_err: f64 = 0;
    for (0..12) |t| {
        const xf = [_]f32{ 0.3 * @sin(@as(f32, @floatFromInt(t))), 0.2 * @cos(@as(f32, @floatFromInt(t)) * 0.7) };
        try x_t.copyFrom(&xf);
        const got = try stream.step(&ctx, &x_t);
        const x64 = [_]f64{ xf[0], xf[1] };
        try referenceStep(&layers[0], &x64, &h[0], &c[0], allocator);
        try referenceStep(&layers[1], &h[0], &h[1], &c[1], allocator);
        for (try got.dataConst(), h[1]) |g, r| max_err = @max(max_err, @abs(@as(f64, g) - r));
    }
    try std.testing.expect(max_err <= 2e-6);
}

test "lstm streaming is bitwise the windowed forward, repeats after reset, and refuses a scope" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var layers = [_]HostLayer{try HostLayer.init(allocator, 1, 4, 30)};
    defer for (&layers) |*l| l.deinit(allocator);
    var lstm = try buildLstm(allocator, &ctx, &layers, false);
    defer lstm.deinit();

    const frames = 50;
    var signal: [frames]f32 = undefined;
    fillUniform(&signal, 31, 0.8);
    var windowed: [frames * 4]f32 = undefined;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var x = try rnn.Sequence.fromSlice(&ctx, .{ frames, 1 }, &signal);
        defer x.deinit();
        const hs = try lstm.forward(&ctx, &x, .{});
        try hs.copyTo(&windowed);
    }
    var stream = try rnn.Lstm.Stream.init(allocator, &ctx, &lstm);
    defer stream.deinit();
    var x_t = try rnn.Input.zeros(&ctx, .{1});
    defer x_t.deinit();
    var streamed: [frames * 4]f32 = undefined;
    for (signal, 0..) |v, t| {
        try x_t.copyFrom(&[_]f32{v});
        try (try stream.step(&ctx, &x_t)).copyTo(streamed[t * 4 ..][0..4]);
    }
    try std.testing.expectEqualSlices(f32, &windowed, &streamed);

    try stream.reset(&ctx);
    var again: [frames * 4]f32 = undefined;
    for (signal, 0..) |v, t| {
        try x_t.copyFrom(&[_]f32{v});
        try (try stream.step(&ctx, &x_t)).copyTo(again[t * 4 ..][0..4]);
    }
    try std.testing.expectEqualSlices(f32, &streamed, &again);

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    try std.testing.expectError(error.ActiveExecScopeUnsupported, stream.step(&ctx, &x_t));
}

test "lstm stacked layout round-trips through fromStacked, stackedWeight and bias" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var layer = try HostLayer.init(allocator, 2, 3, 40);
    defer layer.deinit(allocator);
    inline for (.{ false, true }) |trainable| {
        var cell = try layer.cell(&ctx, trainable);
        defer cell.deinit();
        try std.testing.expectEqual(@as(usize, 2), cell.input_size);
        try std.testing.expectEqual(@as(usize, 3), cell.hidden);
        try std.testing.expectEqual(trainable, cell.w.requiresGrad());
        var stacked = try cell.stackedWeight(&ctx);
        defer stacked.deinit();
        var stacked_out: [4 * 3 * 5]f32 = undefined;
        try stacked.copyTo(&stacked_out);
        try std.testing.expectEqualSlices(f32, layer.stacked, &stacked_out);
        var bias = try cell.bias(&ctx);
        defer bias.deinit();
        var bias_out: [12]f32 = undefined;
        try bias.copyTo(&bias_out);
        try std.testing.expectEqualSlices(f32, layer.bias, &bias_out);
        try std.testing.expectEqualSlices(f32, layer.h0, try cell.h0.dataConst());
        try std.testing.expectEqualSlices(f32, layer.c0, try cell.c0.dataConst());
    }
}

/// A fixed weighting of the hidden sequence: the scalar loss the gradient
/// tests use.
fn weightedLoss(ctx: *ExecContext, lstm: *const rnn.Lstm, signal: []const f32, options: rnn.Lstm.ForwardOptions, weights: []const f32) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    var x = try rnn.Sequence.fromSlice(ctx, .{ signal.len, 1 }, signal);
    defer x.deinit();
    const hs = try lstm.forward(ctx, &x, options);
    var w = try rnn.HiddenSequence.fromSlice(ctx, .{ signal.len, lstm.hiddenSize() }, weights);
    defer w.deinit();
    const weighted = try hs.mul(ctx, &w);
    var loss = try weighted.sumAll(ctx);
    try loss.backward(ctx);
    return loss.item();
}

fn weightedLossValue(allocator: std.mem.Allocator, ctx: *ExecContext, layers: []const HostLayer, signal: []const f32, options: rnn.Lstm.ForwardOptions, weights: []const f32) !f32 {
    var lstm = try buildLstm(allocator, ctx, layers, false);
    defer lstm.deinit();
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    var x = try rnn.Sequence.fromSlice(ctx, .{ signal.len, 1 }, signal);
    defer x.deinit();
    const hs = try lstm.forward(ctx, &x, options);
    var w = try rnn.HiddenSequence.fromSlice(ctx, .{ signal.len, lstm.hiddenSize() }, weights);
    defer w.deinit();
    const weighted = try hs.mul(ctx, &w);
    const loss = try weighted.sumAll(ctx);
    return loss.item();
}

fn expectFd(allocator: std.mem.Allocator, ctx: *ExecContext, layers: []HostLayer, values: []f32, analytic: []const f32, signal: []const f32, weights: []const f32) !void {
    const eps: f32 = 1e-3;
    for (values, analytic) |*value, a| {
        const original = value.*;
        value.* = original + eps;
        const plus = try weightedLossValue(allocator, ctx, layers, signal, .{}, weights);
        value.* = original - eps;
        const minus = try weightedLossValue(allocator, ctx, layers, signal, .{}, weights);
        value.* = original;
        const numeric = (plus - minus) / (2 * eps);
        try std.testing.expect(std.math.isFinite(numeric));
        try std.testing.expect(@abs(numeric - a) <= 2e-2 * @max(1.0, @abs(a)));
    }
}

test "lstm gradients match finite differences for the weight, bias and initial state" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var layers = [_]HostLayer{ try HostLayer.init(allocator, 1, 2, 50), try HostLayer.init(allocator, 2, 2, 60) };
    defer for (&layers) |*l| l.deinit(allocator);
    var signal: [6]f32 = undefined;
    fillUniform(&signal, 51, 0.9);
    var weights: [6 * 2]f32 = undefined;
    fillUniform(&weights, 52, 1.0);

    var lstm = try buildLstm(allocator, &ctx, &layers, true);
    defer lstm.deinit();
    _ = try weightedLoss(&ctx, &lstm, &signal, .{}, &weights);

    for (lstm.cells, &layers) |*cell, *layer| {
        var w_grad = (try cell.w.grad(&ctx)).?;
        defer w_grad.deinit();
        // The stored gradient is [in + H + 1, 4H]: its matrix rows in the
        // stacked order, its last row the bias gradient.
        var matrix = try w_grad.narrow(&ctx, .k, 0, layer.input_size + layer.hidden);
        defer matrix.deinit();
        var stacked_grad = try matrix.permuteTo(&ctx, .{ .unit, .k });
        defer stacked_grad.deinit();
        const analytic_stacked = try allocator.alloc(f32, layer.stacked.len);
        defer allocator.free(analytic_stacked);
        try stacked_grad.copyTo(analytic_stacked);
        try expectFd(allocator, &ctx, &layers, layer.stacked, analytic_stacked, &signal, &weights);
        var bias_row = try w_grad.narrow(&ctx, .k, layer.input_size + layer.hidden, 1);
        defer bias_row.deinit();
        const analytic_bias = try allocator.alloc(f32, layer.bias.len);
        defer allocator.free(analytic_bias);
        try bias_row.copyTo(analytic_bias);
        try expectFd(allocator, &ctx, &layers, layer.bias, analytic_bias, &signal, &weights);
        var h0_grad = (try cell.h0.grad(&ctx)).?;
        defer h0_grad.deinit();
        try expectFd(allocator, &ctx, &layers, layer.h0, try h0_grad.dataConst(), &signal, &weights);
        var c0_grad = (try cell.c0.grad(&ctx)).?;
        defer c0_grad.deinit();
        try expectFd(allocator, &ctx, &layers, layer.c0, try c0_grad.dataConst(), &signal, &weights);
    }
}

test "lstm burn-in cuts the initial-state gradient and truncation keeps the forward values" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var layers = [_]HostLayer{try HostLayer.init(allocator, 1, 3, 70)};
    defer for (&layers) |*l| l.deinit(allocator);
    var signal: [20]f32 = undefined;
    fillUniform(&signal, 71, 0.9);
    var weights: [20 * 3]f32 = undefined;
    fillUniform(&weights, 72, 1.0);

    var full = try buildLstm(allocator, &ctx, &layers, true);
    defer full.deinit();
    var truncated = try buildLstm(allocator, &ctx, &layers, true);
    defer truncated.deinit();
    const loss_full = try weightedLoss(&ctx, &full, &signal, .{ .burn_in = 6, .truncate = 0 }, &weights);
    const loss_truncated = try weightedLoss(&ctx, &truncated, &signal, .{ .burn_in = 6, .truncate = 4 }, &weights);
    try std.testing.expectEqual(loss_full, loss_truncated);
    try std.testing.expect((try full.cells[0].h0.grad(&ctx)) == null);
    try std.testing.expect((try full.cells[0].c0.grad(&ctx)) == null);
    var w_full = (try full.cells[0].w.grad(&ctx)).?;
    defer w_full.deinit();
    var w_truncated = (try truncated.cells[0].w.grad(&ctx)).?;
    defer w_truncated.deinit();
    var differs = false;
    for (try w_full.dataConst(), try w_truncated.dataConst()) |a, b| {
        try std.testing.expect(std.math.isFinite(a) and std.math.isFinite(b));
        if (a != b) differs = true;
    }
    // A shorter gradient horizon is a different gradient.
    try std.testing.expect(differs);
}
