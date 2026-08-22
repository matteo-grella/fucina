//! Grouped attention on the public Tensor: causal, GEMM-backward,
//! bidirectional, biased, windowed, and f16-KV variants against finite
//! differences.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const ag_tensor = @import("../tensor.zig");
const gradcheck_mod = @import("../gradcheck.zig");

const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tensor = ag_tensor.Tensor;
const RawTensor = @import("../../tensor.zig").Tensor;

const util = @import("util.zig");
const expectCloseSlices = util.expectCloseSlices;

test "tagged autograd grouped causal attention matches finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const q_values = [_]f32{ 0.2, -0.4, 0.5, 0.1 };
    const k_values = [_]f32{ 0.3, -0.2, -0.1, 0.4 };
    const v_values = [_]f32{ 0.7, -0.6, 0.2, 0.5 };
    const kv_head_for_head = [_]usize{0};
    const scale_value: f32 = 0.7;

    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &v_values);
    defer v.deinit();

    var y = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{});
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gq = (try q.grad(&ctx)).?;
    defer gq.deinit();
    var gk = (try k.grad(&ctx)).?;
    defer gk.deinit();
    var gv = (try v.grad(&ctx)).?;
    defer gv.deinit();

    const eps: f32 = 1e-3;
    var q_work = q_values;
    for (q_values, 0..) |_, i| {
        q_work = q_values;
        q_work[i] += eps;
        const plus = try groupedAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        q_work[i] -= 2 * eps;
        const minus = try groupedAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gq.asRawTensor().dataConst()[i], 2e-2);
    }

    var k_work = k_values;
    for (k_values, 0..) |_, i| {
        k_work = k_values;
        k_work[i] += eps;
        const plus = try groupedAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        k_work[i] -= 2 * eps;
        const minus = try groupedAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gk.asRawTensor().dataConst()[i], 2e-2);
    }

    var v_work = v_values;
    for (v_values, 0..) |_, i| {
        v_work = v_values;
        v_work[i] += eps;
        const plus = try groupedAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        v_work[i] -= 2 * eps;
        const minus = try groupedAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gv.asRawTensor().dataConst()[i], 2e-2);
    }
}

test "tagged autograd grouped causal attention GEMM backward matches finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const S = 16;
    const H = 4;
    const KV = 2;
    const D = 16;
    const kv_head_for_head = [_]usize{ 0, 0, 1, 1 };
    const scale_value: f32 = 0.35;

    var q_values: [S * H * D]f32 = undefined;
    var k_values: [S * KV * D]f32 = undefined;
    var v_values: [S * KV * D]f32 = undefined;
    for (&q_values, 0..) |*value, i| {
        const x = @as(f32, @floatFromInt(i));
        value.* = @sin(x * 0.071) * 0.4 + @cos(x * 0.037) * 0.11;
    }
    for (&k_values, 0..) |*value, i| {
        const x = @as(f32, @floatFromInt(i));
        value.* = @cos(x * 0.053) * 0.3 - @sin(x * 0.019) * 0.07;
    }
    for (&v_values, 0..) |*value, i| {
        const x = @as(f32, @floatFromInt(i));
        value.* = @sin(x * 0.041 + 0.2) * 0.5;
    }

    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ S, H, D }, &q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ S, KV, D }, &k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ S, KV, D }, &v_values);
    defer v.deinit();

    var y = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{});
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gq = (try q.grad(&ctx)).?;
    defer gq.deinit();
    var gk = (try k.grad(&ctx)).?;
    defer gk.deinit();
    var gv = (try v.grad(&ctx)).?;
    defer gv.deinit();

    const eps: f32 = 1e-2;
    const q_probe = [_]usize{ 0, 137, q_values.len - 5 };
    var q_work = q_values;
    for (q_probe) |i| {
        q_work = q_values;
        q_work[i] += eps;
        const plus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        q_work[i] -= 2 * eps;
        const minus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gq.asRawTensor().dataConst()[i], 2e-2);
    }

    const k_probe = [_]usize{ 3, 91, k_values.len - 7 };
    var k_work = k_values;
    for (k_probe) |i| {
        k_work = k_values;
        k_work[i] += eps;
        const plus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        k_work[i] -= 2 * eps;
        const minus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gk.asRawTensor().dataConst()[i], 2e-2);
    }

    const v_probe = [_]usize{ 11, 173, v_values.len - 1 };
    var v_work = v_values;
    for (v_probe) |i| {
        v_work = v_values;
        v_work[i] += eps;
        const plus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        v_work[i] -= 2 * eps;
        const minus = try groupedAttentionLoss(&ctx, S, H, KV, D, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gv.asRawTensor().dataConst()[i], 2e-2);
    }
}

fn bidirectionalAttentionTinyLoss(
    ctx: *ExecContext,
    q_values: []const f32,
    k_values: []const f32,
    v_values: []const f32,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !f32 {
    var q = try Tensor(.{ .seq, .head, .d }).fromSlice(ctx, .{ 2, 1, 2 }, q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 3, 1, 2 }, k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 3, 1, 2 }, v_values);
    defer v.deinit();
    var y = try q.groupedAttention(ctx, &k, &v, kv_head_for_head, .out, scale_value, .{ .mask = .bidirectional });
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();
    return loss.item();
}

test "tagged autograd grouped bidirectional attention matches finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // q_seq (2) < kv_seq (3): query 0 attends key 2 — a position the causal
    // kernels would mask — so the FD check fails if any causal bound leaks
    // into the bidirectional forward or backward.
    const q_values = [_]f32{ 0.2, -0.4, 0.5, 0.1 };
    const k_values = [_]f32{ 0.3, -0.2, -0.1, 0.4, 0.6, -0.5 };
    const v_values = [_]f32{ 0.7, -0.6, 0.2, 0.5, -0.3, 0.8 };
    const kv_head_for_head = [_]usize{0};
    const scale_value: f32 = 0.7;

    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 3, 1, 2 }, &k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 3, 1, 2 }, &v_values);
    defer v.deinit();

    var y = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .mask = .bidirectional });
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gq = (try q.grad(&ctx)).?;
    defer gq.deinit();
    var gk = (try k.grad(&ctx)).?;
    defer gk.deinit();
    var gv = (try v.grad(&ctx)).?;
    defer gv.deinit();

    const eps: f32 = 1e-3;
    var q_work = q_values;
    for (q_values, 0..) |_, i| {
        q_work = q_values;
        q_work[i] += eps;
        const plus = try bidirectionalAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        q_work[i] -= 2 * eps;
        const minus = try bidirectionalAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gq.asRawTensor().dataConst()[i], 2e-2);
    }

    var k_work = k_values;
    for (k_values, 0..) |_, i| {
        k_work = k_values;
        k_work[i] += eps;
        const plus = try bidirectionalAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        k_work[i] -= 2 * eps;
        const minus = try bidirectionalAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gk.asRawTensor().dataConst()[i], 2e-2);
    }

    var v_work = v_values;
    for (v_values, 0..) |_, i| {
        v_work = v_values;
        v_work[i] += eps;
        const plus = try bidirectionalAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        v_work[i] -= 2 * eps;
        const minus = try bidirectionalAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gv.asRawTensor().dataConst()[i], 2e-2);
    }
}

test "tagged grouped bidirectional biased attention: constant bias matches plain, grads rejected" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const q_values = [_]f32{ 0.2, -0.4, 0.5, 0.1 };
    const k_values = [_]f32{ 0.3, -0.2, -0.1, 0.4, 0.6, -0.5 };
    const v_values = [_]f32{ 0.7, -0.6, 0.2, 0.5, -0.3, 0.8 };
    const kv_head_for_head = [_]usize{0};
    const scale_value: f32 = 0.7;
    // Constant bias: softmax shift-invariance makes the biased result equal
    // the plain bidirectional path up to summation-order rounding.
    const bias_values = [_]f32{1.0} ** (2 * 3);

    var q = try Tensor(.{ .seq, .head, .d }).fromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 3, 1, 2 }, &k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 3, 1, 2 }, &v_values);
    defer v.deinit();
    var bias = try Tensor(.{ .sq, .skv }).fromSlice(&ctx, .{ 2, 3 }, &bias_values);
    defer bias.deinit();

    var got = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .mask = .bidirectional, .bias = &bias });
    defer got.deinit();
    var plain = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .mask = .bidirectional });
    defer plain.deinit();
    for (plain.asRawTensor().dataConst(), got.asRawTensor().dataConst()) |e, g| {
        try std.testing.expectApproxEqAbs(e, g, 1e-5);
    }

    // Inference-only: no VJP exists for the biased forward, so ANY
    // grad-requiring operand is rejected — q here, and the bias itself.
    var qg = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer qg.deinit();
    try std.testing.expectError(
        error.UnsupportedGradient,
        qg.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .mask = .bidirectional, .bias = &bias }),
    );
    var bias_grad = try Tensor(.{ .sq, .skv }).variableFromSlice(&ctx, .{ 2, 3 }, &bias_values);
    defer bias_grad.deinit();
    try std.testing.expectError(
        error.UnsupportedGradient,
        q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .mask = .bidirectional, .bias = &bias_grad }),
    );
}

fn groupedAttentionTinyLoss(
    ctx: *ExecContext,
    q_values: []const f32,
    k_values: []const f32,
    v_values: []const f32,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !f32 {
    var q = try Tensor(.{ .seq, .head, .d }).fromSlice(ctx, .{ 2, 1, 2 }, q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 2, 1, 2 }, k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 2, 1, 2 }, v_values);
    defer v.deinit();
    var y = try q.groupedAttention(ctx, &k, &v, kv_head_for_head, .out, scale_value, .{});
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();
    return loss.item();
}

fn groupedAttentionLoss(
    ctx: *ExecContext,
    comptime S: usize,
    comptime H: usize,
    comptime KV: usize,
    comptime D: usize,
    q_values: []const f32,
    k_values: []const f32,
    v_values: []const f32,
    kv_head_for_head: []const usize,
    scale_value: f32,
) !f32 {
    var q = try Tensor(.{ .seq, .head, .d }).fromSlice(ctx, .{ S, H, D }, q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ S, KV, D }, k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ S, KV, D }, v_values);
    defer v.deinit();
    var y = try q.groupedAttention(ctx, &k, &v, kv_head_for_head, .out, scale_value, .{});
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();
    return loss.item();
}

test "tagged autograd windowed grouped causal attention matches finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const q_values = [_]f32{ 0.2, -0.4, 0.5, 0.1, -0.3, 0.6, 0.05, -0.15 };
    const k_values = [_]f32{ 0.3, -0.2, -0.1, 0.4, 0.25, -0.35, 0.15, 0.45 };
    const v_values = [_]f32{ 0.7, -0.6, 0.2, 0.5, -0.4, 0.3, 0.1, -0.2 };
    const kv_head_for_head = [_]usize{0};
    const scale_value: f32 = 0.7;
    const window: usize = 2;

    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 4, 1, 2 }, &q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 4, 1, 2 }, &k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).variableFromSlice(&ctx, .{ 4, 1, 2 }, &v_values);
    defer v.deinit();

    var y = try q.groupedAttention(&ctx, &k, &v, kv_head_for_head[0..], .out, scale_value, .{ .window = window });
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gq = (try q.grad(&ctx)).?;
    defer gq.deinit();
    var gk = (try k.grad(&ctx)).?;
    defer gk.deinit();
    var gv = (try v.grad(&ctx)).?;
    defer gv.deinit();

    const eps: f32 = 1e-3;
    var q_work = q_values;
    for (q_values, 0..) |_, i| {
        q_work = q_values;
        q_work[i] += eps;
        const plus = try windowedAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value, window);
        q_work[i] -= 2 * eps;
        const minus = try windowedAttentionTinyLoss(&ctx, q_work[0..], k_values[0..], v_values[0..], kv_head_for_head[0..], scale_value, window);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gq.asRawTensor().dataConst()[i], 2e-2);
    }
    var k_work = k_values;
    for (k_values, 0..) |_, i| {
        k_work = k_values;
        k_work[i] += eps;
        const plus = try windowedAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value, window);
        k_work[i] -= 2 * eps;
        const minus = try windowedAttentionTinyLoss(&ctx, q_values[0..], k_work[0..], v_values[0..], kv_head_for_head[0..], scale_value, window);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gk.asRawTensor().dataConst()[i], 2e-2);
    }
    var v_work = v_values;
    for (v_values, 0..) |_, i| {
        v_work = v_values;
        v_work[i] += eps;
        const plus = try windowedAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value, window);
        v_work[i] -= 2 * eps;
        const minus = try windowedAttentionTinyLoss(&ctx, q_values[0..], k_values[0..], v_work[0..], kv_head_for_head[0..], scale_value, window);
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, gv.asRawTensor().dataConst()[i], 2e-2);
    }
}

fn windowedAttentionTinyLoss(
    ctx: *ExecContext,
    q_values: []const f32,
    k_values: []const f32,
    v_values: []const f32,
    kv_head_for_head: []const usize,
    scale_value: f32,
    window: usize,
) !f32 {
    var q = try Tensor(.{ .seq, .head, .d }).fromSlice(ctx, .{ 4, 1, 2 }, q_values);
    defer q.deinit();
    var k = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 4, 1, 2 }, k_values);
    defer k.deinit();
    var v = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ 4, 1, 2 }, v_values);
    defer v.deinit();
    var y = try q.groupedAttention(ctx, &k, &v, kv_head_for_head, .out, scale_value, .{ .window = window });
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();
    return loss.item();
}

test "tagged autograd f16 KV attention propagates q gradient" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Exactly representable in f16 so the f32 reference sees identical K/V.
    const q_values = [_]f32{ 0.25, -0.5, 0.125, 0.75 };
    const k_f16 = [_]f16{ 0.5, -0.25, 0.375, 0.625 };
    const v_f16 = [_]f16{ 0.75, -0.125, 0.25, 0.5 };
    const k_f32 = [_]f32{ 0.5, -0.25, 0.375, 0.625 };
    const v_f32 = [_]f32{ 0.75, -0.125, 0.25, 0.5 };
    const kv_head_for_head = [_]usize{0};
    const scale_value: f32 = 0.6;

    var q = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q.deinit();
    var k16 = try Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } }).fromSlice(&ctx, .{ 2, 1, 2 }, &k_f16);
    defer k16.deinit();
    var v16 = try Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } }).fromSlice(&ctx, .{ 2, 1, 2 }, &v_f16);
    defer v16.deinit();

    var y = try q.groupedAttention(&ctx, &k16, &v16, kv_head_for_head[0..], .out, scale_value, .{});
    defer y.deinit();
    try std.testing.expect(y.requiresGrad());
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gq = (try q.grad(&ctx)).?;
    defer gq.deinit();

    var q_ref = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q_ref.deinit();
    var k32 = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 2, 1, 2 }, &k_f32);
    defer k32.deinit();
    var v32 = try Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 2, 1, 2 }, &v_f32);
    defer v32.deinit();
    var y_ref = try q_ref.groupedAttention(&ctx, &k32, &v32, kv_head_for_head[0..], .out, scale_value, .{});
    defer y_ref.deinit();
    var loss_ref = try y_ref.sumAll(&ctx);
    defer loss_ref.deinit();
    try loss_ref.backward(&ctx);
    var gq_ref = (try q_ref.grad(&ctx)).?;
    defer gq_ref.deinit();

    try expectCloseSlices(y_ref.asRawTensor().dataConst(), y.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(gq_ref.asRawTensor().dataConst(), gq.asRawTensor().dataConst(), 1e-6);

    // Windowed f16-KV fallback against the windowed f32 reference.
    var q_w = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q_w.deinit();
    var y_w = try q_w.groupedAttention(&ctx, &k16, &v16, kv_head_for_head[0..], .out, scale_value, .{ .window = 1 });
    defer y_w.deinit();
    var loss_w = try y_w.sumAll(&ctx);
    defer loss_w.deinit();
    try loss_w.backward(&ctx);
    var gq_w = (try q_w.grad(&ctx)).?;
    defer gq_w.deinit();

    var q_wref = try Tensor(.{ .seq, .head, .d }).variableFromSlice(&ctx, .{ 2, 1, 2 }, &q_values);
    defer q_wref.deinit();
    var y_wref = try q_wref.groupedAttention(&ctx, &k32, &v32, kv_head_for_head[0..], .out, scale_value, .{ .window = 1 });
    defer y_wref.deinit();
    var loss_wref = try y_wref.sumAll(&ctx);
    defer loss_wref.deinit();
    try loss_wref.backward(&ctx);
    var gq_wref = (try q_wref.grad(&ctx)).?;
    defer gq_wref.deinit();

    try expectCloseSlices(y_wref.asRawTensor().dataConst(), y_w.asRawTensor().dataConst(), 1e-6);
    try expectCloseSlices(gq_wref.asRawTensor().dataConst(), gq_w.asRawTensor().dataConst(), 1e-6);
}
