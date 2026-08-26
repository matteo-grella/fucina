//! Shared helpers for the tensor_tests files: slice closeness checks,
//! softmaxExt reference math, and finite-difference gradcheck scaffolding.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const dtype_mod = @import("../../dtype.zig");
const exec_mod = @import("../../exec.zig");
const control = @import("../control.zig");
const core = @import("../core.zig");
const ag_tensor = @import("../tensor.zig");

const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const GradState = core.GradState;
const Tensor = ag_tensor.Tensor;
const RawTensor = @import("../../tensor.zig").Tensor;

pub fn expectCloseSlices(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try std.testing.expectApproxEqAbs(e, a, tolerance);
    }
}

pub fn expectedSoftmaxExtProbs(comptime logits: anytype, comptime mask: anytype, scale_value: f32, slope: f32, sink: ?f32, out: []f32) void {
    const len = logits.len;
    std.debug.assert(mask.len == len);
    std.debug.assert(out.len == len);

    var max_value = logits[0] * scale_value + mask[0] * slope;
    inline for (1..len) |i| {
        max_value = @max(max_value, logits[i] * scale_value + mask[i] * slope);
    }
    if (sink) |sink_value| max_value = @max(max_value, sink_value);

    var sum_exp: f32 = 0;
    inline for (0..len) |i| {
        out[i] = @exp(logits[i] * scale_value + mask[i] * slope - max_value);
        sum_exp += out[i];
    }
    if (sink) |sink_value| sum_exp += @exp(sink_value - max_value);

    inline for (0..len) |i| out[i] /= sum_exp;
}

pub fn expectedSoftmaxExtWeighted(
    comptime logits: anytype,
    comptime mask: anytype,
    comptime weights: anytype,
    scale_value: f32,
    slope: f32,
    sink: ?f32,
    probs_out: []f32,
    grad_out: []f32,
) void {
    const len = logits.len;
    std.debug.assert(weights.len == len);
    std.debug.assert(grad_out.len == len);
    expectedSoftmaxExtProbs(logits, mask, scale_value, slope, sink, probs_out);

    var dot: f32 = 0;
    inline for (0..len) |i| dot += probs_out[i] * weights[i];
    inline for (0..len) |i| grad_out[i] = scale_value * probs_out[i] * (weights[i] - dot);
}

/// Deterministic smooth fill for the FD gradcheck fixtures (values in
/// roughly [-0.75, 0.75], no two operands alike thanks to `phase`).
pub fn fdFillPattern(values: []f32, phase: f32) void {
    for (values, 0..) |*v, i| {
        const x = @as(f32, @floatFromInt(i)) + phase;
        v.* = @sin(x * 0.7) * 0.5 + @cos(x * 0.3) * 0.25;
    }
}

/// f64-accumulated weighted sum — the FD side of `loss = sum(y * coef)`.
pub fn fdWeightedSum(y: []const f32, coef: []const f32) f32 {
    var acc: f64 = 0;
    for (y, coef) |value, c| acc += @as(f64, value) * @as(f64, c);
    return @floatCast(acc);
}

/// Central-difference check of `grad` against `lossAt` over every element of
/// `values` (mutated in place and restored).
pub fn fdCheckGrad(
    values: []f32,
    grad: []const f32,
    eps: f32,
    tol: f32,
    context: anytype,
    comptime lossAt: fn (@TypeOf(context)) anyerror!f32,
) !void {
    try std.testing.expectEqual(values.len, grad.len);
    for (0..values.len) |i| {
        const original = values[i];
        values[i] = original + eps;
        const plus = try lossAt(context);
        values[i] = original - eps;
        const minus = try lossAt(context);
        values[i] = original;
        const expected = (plus - minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(expected, grad[i], tol);
    }
}
