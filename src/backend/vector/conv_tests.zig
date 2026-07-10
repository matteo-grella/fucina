//! Behavioral tests for the small-tap causal convolution kernels (`conv.zig`):
//! general channel-mixing forward/backward with dilation and state, grouped
//! variants, the 1x1 fast path, and depthwise channel-tail handling — checked
//! against hand-computed values and naive references at SIMD-body widths.
const std = @import("std");
const conv = @import("conv.zig");
const tensor = @import("../../tensor.zig");
const thread = @import("../../thread.zig");

const Tensor = tensor.Tensor;

const causalConv1dIntoWithConfig = conv.causalConv1dIntoWithConfig;
const causalConv1dBackwardInputIntoWithConfig = conv.causalConv1dBackwardInputIntoWithConfig;
const causalConv1dBackwardWeightIntoWithConfig = conv.causalConv1dBackwardWeightIntoWithConfig;
const groupedCausalConv1dIntoWithConfig = conv.groupedCausalConv1dIntoWithConfig;
const groupedCausalConv1dBackwardInputIntoWithConfig = conv.groupedCausalConv1dBackwardInputIntoWithConfig;
const groupedCausalConv1dBackwardWeightIntoWithConfig = conv.groupedCausalConv1dBackwardWeightIntoWithConfig;
const causalDepthwiseConv1dIntoWithConfig = conv.causalDepthwiseConv1dIntoWithConfig;

test "general causal conv vector kernel: channel mixing, dilation, state, vector tails" {
    const allocator = std.testing.allocator;

    // seq=3, in=2, out=2, taps=2, dilation=1; w[k][i][o] with k=1 the newest tap.
    var input = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &.{
        1, 10,
        2, 20,
        3, 30,
    });
    defer input.deinit();
    var weight = try Tensor.fromSlice(allocator, &.{ 2, 2, 2 }, &.{
        10, 20, // k=0 (oldest), i=0
        30, 40, // k=0, i=1
        1, 2, // k=1 (newest), i=0
        3, 4, // k=1, i=1
    });
    defer weight.deinit();
    var out = try Tensor.zeros(allocator, &.{ 3, 2 });
    defer out.deinit();

    causalConv1dIntoWithConfig(&out, &input, &weight, null, 3, 2, 2, 2, 1, .{});
    try std.testing.expectEqualSlices(f32, &.{
        31,  42,
        372, 504,
        713, 966,
    }, out.dataConst());

    // Same conv with one state row [5, 7] feeding the t=0 oldest tap.
    const state = [_]f32{ 5, 7 };
    causalConv1dIntoWithConfig(&out, &input, &weight, &state, 3, 2, 2, 2, 1, .{});
    try std.testing.expectEqualSlices(f32, &.{
        291, 422,
        372, 504,
        713, 966,
    }, out.dataConst());

    // Dilation 2, in=out=1: y[t] = 10*x[t-2] + x[t]; state rows fill t<2 history.
    var dilated_input = try Tensor.fromSlice(allocator, &.{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer dilated_input.deinit();
    var dilated_weight = try Tensor.fromSlice(allocator, &.{ 2, 1, 1 }, &.{ 10, 1 });
    defer dilated_weight.deinit();
    var dilated_out = try Tensor.zeros(allocator, &.{ 4, 1 });
    defer dilated_out.deinit();

    causalConv1dIntoWithConfig(&dilated_out, &dilated_input, &dilated_weight, null, 4, 1, 1, 2, 2, .{});
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 13, 24 }, dilated_out.dataConst());

    const dilated_state = [_]f32{ 100, 200 };
    causalConv1dIntoWithConfig(&dilated_out, &dilated_input, &dilated_weight, &dilated_state, 4, 1, 1, 2, 2, .{});
    try std.testing.expectEqualSlices(f32, &.{ 1001, 2002, 13, 24 }, dilated_out.dataConst());

    // out=5 exercises both the SIMD body and the scalar tail of axpyRow.
    var wide_weight = try Tensor.fromSlice(allocator, &.{ 1, 1, 5 }, &.{ 1, 2, 3, 4, 5 });
    defer wide_weight.deinit();
    var wide_out = try Tensor.zeros(allocator, &.{ 4, 5 });
    defer wide_out.deinit();
    causalConv1dIntoWithConfig(&wide_out, &dilated_input, &wide_weight, null, 4, 1, 5, 1, 1, .{});
    try std.testing.expectEqualSlices(f32, &.{
        1, 2, 3,  4,  5,
        2, 4, 6,  8,  10,
        3, 6, 9,  12, 15,
        4, 8, 12, 16, 20,
    }, wide_out.dataConst());
}

test "general causal conv vector backward kernels match hand-computed gradients" {
    const allocator = std.testing.allocator;

    // Mirrors the forward test: seq=3, in=2, out=2, taps=2, dilation=1,
    // state [5, 7], upstream gradient of ones (loss = sum of outputs).
    var input = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &.{
        1, 10,
        2, 20,
        3, 30,
    });
    defer input.deinit();
    var weight = try Tensor.fromSlice(allocator, &.{ 2, 2, 2 }, &.{
        10, 20,
        30, 40,
        1,  2,
        3,  4,
    });
    defer weight.deinit();
    var gy = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &.{
        1, 1,
        1, 1,
        1, 1,
    });
    defer gy.deinit();
    const state = [_]f32{ 5, 7 };

    var gx = try Tensor.zeros(allocator, &.{ 3, 2 });
    defer gx.deinit();
    causalConv1dBackwardInputIntoWithConfig(&gx, &gy, &weight, 3, 2, 2, 2, 1, .{});
    try std.testing.expectEqualSlices(f32, &.{
        33, 77,
        33, 77,
        3,  7,
    }, gx.dataConst());

    var gw = try Tensor.zeros(allocator, &.{ 2, 2, 2 });
    defer gw.deinit();
    causalConv1dBackwardWeightIntoWithConfig(&gw, &input, &gy, &state, 3, 2, 2, 2, 1, .{});
    try std.testing.expectEqualSlices(f32, &.{
        8,  8,
        37, 37,
        6,  6,
        60, 60,
    }, gw.dataConst());
}

test "general causal conv vector kernels match a naive reference at SIMD-body widths" {
    // out=19 covers the vector body AND tail of axpyRow/dotRow on every ISA
    // (4-lane NEON through 16-lane AVX-512); the small-channel tests above
    // only reach the scalar tails.
    const allocator = std.testing.allocator;
    const seq = 11;
    const in_ch = 5;
    const out_ch = 19;
    const taps = 3;
    const dilation = 2;
    const pad = dilation * (taps - 1);

    var input_data: [seq * in_ch]f32 = undefined;
    for (&input_data, 0..) |*v, idx| v.* = @sin(@as(f32, @floatFromInt(idx)) * 0.7) + 0.1;
    var weight_data: [taps * in_ch * out_ch]f32 = undefined;
    for (&weight_data, 0..) |*v, idx| v.* = @cos(@as(f32, @floatFromInt(idx)) * 0.3) - 0.05;
    var gy_data: [seq * out_ch]f32 = undefined;
    for (&gy_data, 0..) |*v, idx| v.* = @sin(@as(f32, @floatFromInt(idx)) * 0.11 + 1.0);
    var state_data: [pad * in_ch]f32 = undefined;
    for (&state_data, 0..) |*v, idx| v.* = @cos(@as(f32, @floatFromInt(idx)) * 0.9);

    var input = try Tensor.fromSlice(allocator, &.{ seq, in_ch }, &input_data);
    defer input.deinit();
    var weight = try Tensor.fromSlice(allocator, &.{ taps, in_ch, out_ch }, &weight_data);
    defer weight.deinit();
    var gy = try Tensor.fromSlice(allocator, &.{ seq, out_ch }, &gy_data);
    defer gy.deinit();

    const ref_x = struct {
        fn at(x: []const f32, s: []const f32, t: usize, k: usize, i: usize) f32 {
            const shifted = t + k * dilation;
            if (shifted >= pad) return x[(shifted - pad) * in_ch + i];
            return s[shifted * in_ch + i];
        }
    }.at;

    var out = try Tensor.zeros(allocator, &.{ seq, out_ch });
    defer out.deinit();
    causalConv1dIntoWithConfig(&out, &input, &weight, &state_data, seq, in_ch, out_ch, taps, dilation, .{});
    for (0..seq) |t| {
        for (0..out_ch) |o| {
            var acc: f64 = 0;
            for (0..taps) |k| for (0..in_ch) |i| {
                acc += @as(f64, ref_x(&input_data, &state_data, t, k, i)) * weight_data[(k * in_ch + i) * out_ch + o];
            };
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), out.dataConst()[t * out_ch + o], 1e-5);
        }
    }

    var gx = try Tensor.zeros(allocator, &.{ seq, in_ch });
    defer gx.deinit();
    causalConv1dBackwardInputIntoWithConfig(&gx, &gy, &weight, seq, in_ch, out_ch, taps, dilation, .{});
    for (0..seq) |p| {
        for (0..in_ch) |i| {
            var acc: f64 = 0;
            for (0..taps) |k| {
                const t = p + pad - k * dilation;
                if (t >= seq) continue;
                for (0..out_ch) |o| acc += @as(f64, gy_data[t * out_ch + o]) * weight_data[(k * in_ch + i) * out_ch + o];
            }
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), gx.dataConst()[p * in_ch + i], 1e-5);
        }
    }

    var gw = try Tensor.zeros(allocator, &.{ taps, in_ch, out_ch });
    defer gw.deinit();
    causalConv1dBackwardWeightIntoWithConfig(&gw, &input, &gy, &state_data, seq, in_ch, out_ch, taps, dilation, .{});
    for (0..taps) |k| {
        for (0..in_ch) |i| {
            for (0..out_ch) |o| {
                var acc: f64 = 0;
                for (0..seq) |t| acc += @as(f64, gy_data[t * out_ch + o]) * ref_x(&input_data, &state_data, t, k, i);
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), gw.dataConst()[(k * in_ch + i) * out_ch + o], 1e-5);
            }
        }
    }
}

test "grouped general causal conv vector kernels match a naive reference at SIMD-body widths" {
    const allocator = std.testing.allocator;
    const seq = 9;
    const groups = 3;
    const in_per_group = 2;
    const out_per_group = 5;
    const in_ch = groups * in_per_group;
    const out_ch = groups * out_per_group;
    const taps = 3;
    const dilation = 2;
    const pad = dilation * (taps - 1);

    var input_data: [seq * in_ch]f32 = undefined;
    for (&input_data, 0..) |*v, idx| v.* = @sin(@as(f32, @floatFromInt(idx)) * 0.17) + 0.03;
    var weight_data: [taps * in_per_group * out_ch]f32 = undefined;
    for (&weight_data, 0..) |*v, idx| v.* = 0.2 * @cos(@as(f32, @floatFromInt(idx)) * 0.23);
    var gy_data: [seq * out_ch]f32 = undefined;
    for (&gy_data, 0..) |*v, idx| v.* = @sin(@as(f32, @floatFromInt(idx)) * 0.07 + 0.4);
    var state_data: [pad * in_ch]f32 = undefined;
    for (&state_data, 0..) |*v, idx| v.* = @cos(@as(f32, @floatFromInt(idx)) * 0.31);

    var input = try Tensor.fromSlice(allocator, &.{ seq, in_ch }, &input_data);
    defer input.deinit();
    var weight = try Tensor.fromSlice(allocator, &.{ taps, in_per_group, out_ch }, &weight_data);
    defer weight.deinit();
    var gy = try Tensor.fromSlice(allocator, &.{ seq, out_ch }, &gy_data);
    defer gy.deinit();

    const ref_x = struct {
        fn at(x: []const f32, s: []const f32, t: usize, k: usize, i: usize) f32 {
            const shifted = t + k * dilation;
            if (shifted >= pad) return x[(shifted - pad) * in_ch + i];
            return s[shifted * in_ch + i];
        }
    }.at;

    var out = try Tensor.zeros(allocator, &.{ seq, out_ch });
    defer out.deinit();
    groupedCausalConv1dIntoWithConfig(&out, &input, &weight, &state_data, seq, in_ch, out_ch, taps, dilation, groups, .{});
    for (0..seq) |t| {
        for (0..out_ch) |o| {
            const group = o / out_per_group;
            var acc: f64 = 0;
            for (0..taps) |k| for (0..in_per_group) |local_i| {
                const i = group * in_per_group + local_i;
                acc += @as(f64, ref_x(&input_data, &state_data, t, k, i)) * weight_data[(k * in_per_group + local_i) * out_ch + o];
            };
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), out.dataConst()[t * out_ch + o], 1e-5);
        }
    }

    var gx = try Tensor.zeros(allocator, &.{ seq, in_ch });
    defer gx.deinit();
    groupedCausalConv1dBackwardInputIntoWithConfig(&gx, &gy, &weight, seq, in_ch, out_ch, taps, dilation, groups, .{});
    for (0..seq) |p| {
        for (0..in_ch) |i| {
            const group = i / in_per_group;
            const local_i = i - group * in_per_group;
            const out_start = group * out_per_group;
            var acc: f64 = 0;
            for (0..taps) |k| {
                const t = p + pad - k * dilation;
                if (t >= seq) continue;
                for (out_start..out_start + out_per_group) |o| {
                    acc += @as(f64, gy_data[t * out_ch + o]) * weight_data[(k * in_per_group + local_i) * out_ch + o];
                }
            }
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), gx.dataConst()[p * in_ch + i], 1e-5);
        }
    }

    var gw = try Tensor.zeros(allocator, &.{ taps, in_per_group, out_ch });
    defer gw.deinit();
    groupedCausalConv1dBackwardWeightIntoWithConfig(&gw, &input, &gy, &state_data, seq, in_ch, out_ch, taps, dilation, groups, .{});
    for (0..taps) |k| {
        for (0..in_per_group) |local_i| {
            for (0..out_ch) |o| {
                const group = o / out_per_group;
                const i = group * in_per_group + local_i;
                var acc: f64 = 0;
                for (0..seq) |t| acc += @as(f64, ref_x(&input_data, &state_data, t, k, i)) * gy_data[t * out_ch + o];
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), gw.dataConst()[(k * in_per_group + local_i) * out_ch + o], 1e-5);
            }
        }
    }
}

test "grouped 1x1 general causal conv fast path matches a naive reference" {
    const allocator = std.testing.allocator;
    const seq = 10;
    const groups = 2;
    const in_per_group = 3;
    const out_per_group = 7;
    const in_ch = groups * in_per_group;
    const out_ch = groups * out_per_group;
    const taps = 1;
    const dilation = 5;

    var input_data: [seq * in_ch]f32 = undefined;
    for (&input_data, 0..) |*v, idx| v.* = 0.4 * @sin(@as(f32, @floatFromInt(idx)) * 0.13) + 0.01;
    var weight_data: [taps * in_per_group * out_ch]f32 = undefined;
    for (&weight_data, 0..) |*v, idx| v.* = 0.3 * @cos(@as(f32, @floatFromInt(idx)) * 0.19) - 0.02;
    var gy_data: [seq * out_ch]f32 = undefined;
    for (&gy_data, 0..) |*v, idx| v.* = @sin(@as(f32, @floatFromInt(idx)) * 0.05 + 0.7);
    var unused_state: [in_ch]f32 = undefined;
    for (&unused_state) |*v| v.* = std.math.nan(f32);

    var input = try Tensor.fromSlice(allocator, &.{ seq, in_ch }, &input_data);
    defer input.deinit();
    var weight = try Tensor.fromSlice(allocator, &.{ taps, in_per_group, out_ch }, &weight_data);
    defer weight.deinit();
    var gy = try Tensor.fromSlice(allocator, &.{ seq, out_ch }, &gy_data);
    defer gy.deinit();

    var out = try Tensor.zeros(allocator, &.{ seq, out_ch });
    defer out.deinit();
    groupedCausalConv1dIntoWithConfig(&out, &input, &weight, &unused_state, seq, in_ch, out_ch, taps, dilation, groups, .{});
    for (0..seq) |t| {
        for (0..out_ch) |o| {
            const group = o / out_per_group;
            var acc: f64 = 0;
            for (0..in_per_group) |local_i| {
                const i = group * in_per_group + local_i;
                acc += @as(f64, input_data[t * in_ch + i]) * weight_data[local_i * out_ch + o];
            }
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), out.dataConst()[t * out_ch + o], 1e-5);
        }
    }

    var gx = try Tensor.zeros(allocator, &.{ seq, in_ch });
    defer gx.deinit();
    groupedCausalConv1dBackwardInputIntoWithConfig(&gx, &gy, &weight, seq, in_ch, out_ch, taps, dilation, groups, .{});
    for (0..seq) |p| {
        for (0..in_ch) |i| {
            const group = i / in_per_group;
            const local_i = i - group * in_per_group;
            const out_start = group * out_per_group;
            var acc: f64 = 0;
            for (out_start..out_start + out_per_group) |o| {
                acc += @as(f64, gy_data[p * out_ch + o]) * weight_data[local_i * out_ch + o];
            }
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), gx.dataConst()[p * in_ch + i], 1e-5);
        }
    }

    var gw = try Tensor.zeros(allocator, &.{ taps, in_per_group, out_ch });
    defer gw.deinit();
    groupedCausalConv1dBackwardWeightIntoWithConfig(&gw, &input, &gy, &unused_state, seq, in_ch, out_ch, taps, dilation, groups, .{});
    for (0..in_per_group) |local_i| {
        for (0..out_ch) |o| {
            const group = o / out_per_group;
            const i = group * in_per_group + local_i;
            var acc: f64 = 0;
            for (0..seq) |t| acc += @as(f64, input_data[t * in_ch + i]) * gy_data[t * out_ch + o];
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), gw.dataConst()[local_i * out_ch + o], 1e-5);
        }
    }
}

test "causal depthwise conv vector kernel handles channel tails" {
    const allocator = std.testing.allocator;
    var input = try Tensor.fromSlice(allocator, &.{ 2, 5 }, &.{
        1, 10, 100, 1000, 10000,
        2, 20, 200, 2000, 20000,
    });
    defer input.deinit();
    var kernel = try Tensor.fromSlice(allocator, &.{ 5, 2 }, &.{
        1, 2,
        1, 2,
        1, 2,
        1, 2,
        1, 2,
    });
    defer kernel.deinit();
    var out = try Tensor.zeros(allocator, &.{ 2, 5 });
    defer out.deinit();

    causalDepthwiseConv1dIntoWithConfig(&out, &input, &kernel, null, 2, 5, 2, .{});

    try std.testing.expectEqualSlices(f32, &.{
        2, 20, 200, 2000, 20000,
        5, 50, 500, 5000, 50000,
    }, out.dataConst());
}

// ===========================================================================
// conv1d (general non-causal) + col2im1d — vs hand-rolled naive references.
// ===========================================================================

const conv1dIntoWithConfig = conv.conv1dIntoWithConfig;
const conv1dBackwardInputIntoWithConfig = conv.conv1dBackwardInputIntoWithConfig;
const conv1dBackwardWeightIntoWithConfig = conv.conv1dBackwardWeightIntoWithConfig;
const col2im1dIntoWithConfig = conv.col2im1dIntoWithConfig;
const col2im1dBackwardIntoWithConfig = conv.col2im1dBackwardIntoWithConfig;
const Conv1dDims = conv.Conv1dDims;

/// Deterministic pseudo-random fill (splitmix-style) in [-2, 2).
fn fillPseudoRandom(values: []f32, seed: u64) void {
    var state = seed;
    for (values) |*v| {
        state +%= 0x9e3779b97f4a7c15;
        var z = state;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        z ^= z >> 31;
        const unit = @as(f32, @floatFromInt(z >> 40)) / @as(f32, @floatFromInt(@as(u64, 1) << 24));
        v.* = unit * 4.0 - 2.0;
    }
}

fn naiveConv1d(out: []f32, input: []const f32, weight: []const f32, d: Conv1dDims) void {
    const in_per_group = d.in_channels / d.groups;
    const out_per_group = d.out_channels / d.groups;
    for (0..d.out_len) |t| {
        for (0..d.out_channels) |o| {
            const g = o / out_per_group;
            var acc: f32 = 0;
            for (0..d.taps) |k| {
                const pos = t * d.stride + k * d.dilation;
                if (pos < d.pad) continue;
                const src = pos - d.pad;
                if (src >= d.seq) continue;
                for (0..in_per_group) |li| {
                    acc += input[src * d.in_channels + g * in_per_group + li] * weight[(k * in_per_group + li) * d.out_channels + o];
                }
            }
            out[t * d.out_channels + o] = acc;
        }
    }
}

fn conv1dOutLen(seq: usize, taps: usize, stride: usize, pad: usize, dilation: usize) usize {
    return (seq + 2 * pad - dilation * (taps - 1) - 1) / stride + 1;
}

test "conv1d vector kernel matches naive reference across stride/pad/dilation/groups" {
    const allocator = std.testing.allocator;

    const Case = struct {
        seq: usize,
        in: usize,
        out: usize,
        taps: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        groups: usize,
    };
    const cases = [_]Case{
        // SIMD-awkward channel counts, same-length pad.
        .{ .seq = 7, .in = 3, .out = 5, .taps = 3, .stride = 1, .pad = 1, .dilation = 1, .groups = 1 },
        // HuBERT feature layer 0-like: k=10, s=5, valid pad, mono in.
        .{ .seq = 33, .in = 1, .out = 8, .taps = 10, .stride = 5, .pad = 0, .dilation = 1, .groups = 1 },
        // Dilated DAC res-unit-like: k=7, d=3, p=9 (same-length).
        .{ .seq = 12, .in = 4, .out = 6, .taps = 7, .stride = 1, .pad = 9, .dilation = 3, .groups = 1 },
        // DAC encoder block 0-like: k=16, s=8, p=4.
        .{ .seq = 40, .in = 8, .out = 4, .taps = 16, .stride = 8, .pad = 4, .dilation = 1, .groups = 1 },
        // Grouped (HuBERT pos_conv-like shape family).
        .{ .seq = 9, .in = 6, .out = 6, .taps = 3, .stride = 1, .pad = 1, .dilation = 1, .groups = 3 },
        // Grouped with distinct in/out per group.
        .{ .seq = 11, .in = 4, .out = 8, .taps = 5, .stride = 2, .pad = 2, .dilation = 1, .groups = 2 },
        // Pointwise.
        .{ .seq = 6, .in = 5, .out = 7, .taps = 1, .stride = 1, .pad = 0, .dilation = 1, .groups = 1 },
        // Even kernel, stride == taps.
        .{ .seq = 10, .in = 2, .out = 3, .taps = 2, .stride = 2, .pad = 0, .dilation = 1, .groups = 1 },
        // Pad larger than the receptive field start (leading all-pad rows).
        .{ .seq = 4, .in = 2, .out = 2, .taps = 3, .stride = 1, .pad = 4, .dilation = 1, .groups = 1 },
    };

    for (cases, 0..) |case, case_i| {
        const d: Conv1dDims = .{
            .seq = case.seq,
            .out_len = conv1dOutLen(case.seq, case.taps, case.stride, case.pad, case.dilation),
            .in_channels = case.in,
            .out_channels = case.out,
            .taps = case.taps,
            .stride = case.stride,
            .pad = case.pad,
            .dilation = case.dilation,
            .groups = case.groups,
        };

        var input = try Tensor.zeros(allocator, &.{ d.seq, d.in_channels });
        defer input.deinit();
        fillPseudoRandom(input.data(), 1000 + case_i);
        const in_per_group = d.in_channels / d.groups;
        var weight = try Tensor.zeros(allocator, &.{ d.taps, in_per_group, d.out_channels });
        defer weight.deinit();
        fillPseudoRandom(weight.data(), 2000 + case_i);

        var out = try Tensor.zeros(allocator, &.{ d.out_len, d.out_channels });
        defer out.deinit();
        conv1dIntoWithConfig(&out, &input, &weight, d, .{});

        const want = try allocator.alloc(f32, d.out_len * d.out_channels);
        defer allocator.free(want);
        naiveConv1d(want, input.dataConst(), weight.dataConst(), d);

        for (want, out.dataConst()) |w, g| {
            try std.testing.expectApproxEqAbs(w, g, 1e-5);
        }
    }
}

test "conv1d vector kernel hand-computed case (k=3, s=1, p=1)" {
    const allocator = std.testing.allocator;

    // x = [1,2,3,4], w = [1,2,3] -> y[t] = sum_k w[k]*xpad[t+k], xpad = [0,x,0].
    // PyTorch: conv1d(x, w, padding=1) = [8, 14, 20, 11].
    var input = try Tensor.fromSlice(allocator, &.{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    var weight = try Tensor.fromSlice(allocator, &.{ 3, 1, 1 }, &.{ 1, 2, 3 });
    defer weight.deinit();
    var out = try Tensor.zeros(allocator, &.{ 4, 1 });
    defer out.deinit();

    conv1dIntoWithConfig(&out, &input, &weight, .{
        .seq = 4,
        .out_len = 4,
        .in_channels = 1,
        .out_channels = 1,
        .taps = 3,
        .stride = 1,
        .pad = 1,
        .dilation = 1,
        .groups = 1,
    }, .{});
    try std.testing.expectEqualSlices(f32, &.{ 8, 14, 20, 11 }, out.dataConst());
}

fn naiveCol2im1dScatter(
    out: []f32,
    col: []const f32,
    t_in: usize,
    out_len: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
) void {
    @memset(out, 0);
    const t_conv = (t_in - 1) * stride + taps - 2 * pad;
    for (0..t_in) |ti| {
        for (0..out_channels) |oc| {
            for (0..taps) |k| {
                const pos = ti * stride + k;
                if (pos < pad) continue;
                const t_out = pos - pad;
                if (t_out >= t_conv or t_out >= out_len) continue;
                out[t_out * out_channels + oc] += col[ti * (taps * out_channels) + oc * taps + k];
            }
        }
    }
}

test "col2im1d vector kernel matches naive scatter for the 5 DAC decoder combos" {
    const allocator = std.testing.allocator;

    // (stride, taps, pad, output_pad) — the 5 DAC decoder upsampling blocks.
    const combos = [_][4]usize{
        .{ 8, 16, 4, 0 },
        .{ 5, 10, 3, 1 },
        .{ 4, 8, 2, 0 },
        .{ 2, 4, 1, 0 },
        .{ 3, 6, 2, 1 },
    };
    for (combos, 0..) |combo, combo_i| {
        const stride = combo[0];
        const taps = combo[1];
        const pad = combo[2];
        const output_pad = combo[3];
        const t_in: usize = 4;
        const out_channels: usize = 3;
        const t_conv = (t_in - 1) * stride + taps - 2 * pad;
        const out_len = t_conv + output_pad;

        var col = try Tensor.zeros(allocator, &.{ t_in, taps * out_channels });
        defer col.deinit();
        fillPseudoRandom(col.data(), 3000 + combo_i);

        var out = try Tensor.zeros(allocator, &.{ out_len, out_channels });
        defer out.deinit();
        // Poison the output to prove every element (incl. the output_pad rows)
        // is written.
        @memset(out.data(), std.math.nan(f32));
        col2im1dIntoWithConfig(&out, &col, t_in, out_len, out_channels, taps, stride, pad, .{});

        const want = try allocator.alloc(f32, out_len * out_channels);
        defer allocator.free(want);
        naiveCol2im1dScatter(want, col.dataConst(), t_in, out_len, out_channels, taps, stride, pad);

        for (want, out.dataConst()) |w, g| {
            try std.testing.expectApproxEqAbs(w, g, 1e-6);
        }
        // output_pad rows are exactly zero.
        for (0..output_pad) |i| {
            const row = out.dataConst()[(t_conv + i) * out_channels ..][0..out_channels];
            for (row) |v| try std.testing.expectEqual(@as(f32, 0), v);
        }
    }
}

test "col2im1d vector kernel hand-computed gather (s=2, k=2)" {
    const allocator = std.testing.allocator;

    // col rows (t_in=2, oc=1): [c00, c01], [c10, c11] with column index k.
    var col = try Tensor.fromSlice(allocator, &.{ 2, 2 }, &.{ 10, 20, 30, 40 });
    defer col.deinit();

    // pad=0: T_out = (2-1)*2 + 2 = 4; out[t] = col[t/2, t%2] scattered disjointly.
    var out = try Tensor.zeros(allocator, &.{ 4, 1 });
    defer out.deinit();
    col2im1dIntoWithConfig(&out, &col, 2, 4, 1, 2, 2, 0, .{});
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30, 40 }, out.dataConst());

    // pad=1 crops one frame on each side: T_out = 2, out = [c01, c10].
    var cropped = try Tensor.zeros(allocator, &.{ 2, 1 });
    defer cropped.deinit();
    col2im1dIntoWithConfig(&cropped, &col, 2, 2, 1, 2, 2, 1, .{});
    try std.testing.expectEqualSlices(f32, &.{ 20, 30 }, cropped.dataConst());
}

/// Naive adjoint of naiveConv1d: scatter every forward (t, k, channel) pair
/// into gx and gw.
fn naiveConv1dBackward(
    gx: []f32,
    gw: []f32,
    input: []const f32,
    weight: []const f32,
    gy: []const f32,
    d: Conv1dDims,
) void {
    @memset(gx, 0);
    @memset(gw, 0);
    const in_per_group = d.in_channels / d.groups;
    const out_per_group = d.out_channels / d.groups;
    for (0..d.out_len) |t| {
        for (0..d.out_channels) |o| {
            const g = o / out_per_group;
            const gyv = gy[t * d.out_channels + o];
            for (0..d.taps) |k| {
                const pos = t * d.stride + k * d.dilation;
                if (pos < d.pad) continue;
                const src = pos - d.pad;
                if (src >= d.seq) continue;
                for (0..in_per_group) |li| {
                    const ic = g * in_per_group + li;
                    gx[src * d.in_channels + ic] += gyv * weight[(k * in_per_group + li) * d.out_channels + o];
                    gw[(k * in_per_group + li) * d.out_channels + o] += gyv * input[src * d.in_channels + ic];
                }
            }
        }
    }
}

test "conv1d backward vector kernels match the naive adjoint across stride/pad/dilation/groups" {
    const allocator = std.testing.allocator;

    const Case = struct {
        seq: usize,
        in: usize,
        out: usize,
        taps: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        groups: usize,
    };
    const cases = [_]Case{
        // SIMD-awkward channel counts (dotRow/axpyRow body + tail).
        .{ .seq = 11, .in = 5, .out = 19, .taps = 3, .stride = 1, .pad = 1, .dilation = 1, .groups = 1 },
        // Strided + padded (n % stride skipping in backward-input).
        .{ .seq = 12, .in = 3, .out = 5, .taps = 4, .stride = 2, .pad = 3, .dilation = 1, .groups = 1 },
        // Dilated with long pad.
        .{ .seq = 10, .in = 4, .out = 6, .taps = 5, .stride = 1, .pad = 6, .dilation = 3, .groups = 1 },
        // Grouped k=8 p=64 (the omnivoice grouped shape).
        .{ .seq = 9, .in = 8, .out = 8, .taps = 8, .stride = 1, .pad = 64, .dilation = 1, .groups = 4 },
        // Grouped with distinct in/out per group + stride.
        .{ .seq = 11, .in = 4, .out = 8, .taps = 5, .stride = 2, .pad = 2, .dilation = 1, .groups = 2 },
        // Pointwise.
        .{ .seq = 6, .in = 5, .out = 7, .taps = 1, .stride = 1, .pad = 0, .dilation = 1, .groups = 1 },
    };

    for (cases, 0..) |case, case_i| {
        const d: Conv1dDims = .{
            .seq = case.seq,
            .out_len = conv1dOutLen(case.seq, case.taps, case.stride, case.pad, case.dilation),
            .in_channels = case.in,
            .out_channels = case.out,
            .taps = case.taps,
            .stride = case.stride,
            .pad = case.pad,
            .dilation = case.dilation,
            .groups = case.groups,
        };
        const in_per_group = d.in_channels / d.groups;

        var input = try Tensor.zeros(allocator, &.{ d.seq, d.in_channels });
        defer input.deinit();
        fillPseudoRandom(input.data(), 4000 + case_i);
        var weight = try Tensor.zeros(allocator, &.{ d.taps, in_per_group, d.out_channels });
        defer weight.deinit();
        fillPseudoRandom(weight.data(), 5000 + case_i);
        var gy = try Tensor.zeros(allocator, &.{ d.out_len, d.out_channels });
        defer gy.deinit();
        fillPseudoRandom(gy.data(), 6000 + case_i);

        var gx = try Tensor.zeros(allocator, &.{ d.seq, d.in_channels });
        defer gx.deinit();
        @memset(gx.data(), std.math.nan(f32));
        conv1dBackwardInputIntoWithConfig(&gx, &gy, &weight, d, .{});

        var gw = try Tensor.zeros(allocator, &.{ d.taps, in_per_group, d.out_channels });
        defer gw.deinit();
        @memset(gw.data(), std.math.nan(f32));
        conv1dBackwardWeightIntoWithConfig(&gw, &input, &gy, d, .{});

        const want_gx = try allocator.alloc(f32, d.seq * d.in_channels);
        defer allocator.free(want_gx);
        const want_gw = try allocator.alloc(f32, d.taps * in_per_group * d.out_channels);
        defer allocator.free(want_gw);
        naiveConv1dBackward(want_gx, want_gw, input.dataConst(), weight.dataConst(), gy.dataConst(), d);

        for (want_gx, gx.dataConst()) |w, g| {
            try std.testing.expectApproxEqAbs(w, g, 1e-4);
        }
        for (want_gw, gw.dataConst()) |w, g| {
            try std.testing.expectApproxEqAbs(w, g, 1e-4);
        }
    }
}

test "conv1d backward vector kernels hand-computed case (k=3, s=1, p=1, gy=1)" {
    const allocator = std.testing.allocator;

    // Mirrors the forward hand case: x=[1,2,3,4], w=[1,2,3], p=1, gy = ones.
    // gx[ti] = sum of w taps that touch x[ti]; gw[k] = sum of x rows tap k reads.
    var input = try Tensor.fromSlice(allocator, &.{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    var weight = try Tensor.fromSlice(allocator, &.{ 3, 1, 1 }, &.{ 1, 2, 3 });
    defer weight.deinit();
    var gy = try Tensor.fromSlice(allocator, &.{ 4, 1 }, &.{ 1, 1, 1, 1 });
    defer gy.deinit();
    const d: Conv1dDims = .{
        .seq = 4,
        .out_len = 4,
        .in_channels = 1,
        .out_channels = 1,
        .taps = 3,
        .stride = 1,
        .pad = 1,
        .dilation = 1,
        .groups = 1,
    };

    var gx = try Tensor.zeros(allocator, &.{ 4, 1 });
    defer gx.deinit();
    conv1dBackwardInputIntoWithConfig(&gx, &gy, &weight, d, .{});
    try std.testing.expectEqualSlices(f32, &.{ 3, 6, 6, 5 }, gx.dataConst());

    var gw = try Tensor.zeros(allocator, &.{ 3, 1, 1 });
    defer gw.deinit();
    conv1dBackwardWeightIntoWithConfig(&gw, &input, &gy, d, .{});
    try std.testing.expectEqualSlices(f32, &.{ 6, 10, 9 }, gw.dataConst());
}

/// Naive adjoint of naiveCol2im1dScatter: every (ti, oc, k) cell reads back
/// the gy element its forward scatter target received (or 0 when cropped /
/// output_pad).
fn naiveCol2im1dBackward(
    gcol: []f32,
    gy: []const f32,
    t_in: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
) void {
    @memset(gcol, 0);
    const t_conv = (t_in - 1) * stride + taps - 2 * pad;
    for (0..t_in) |ti| {
        for (0..out_channels) |oc| {
            for (0..taps) |k| {
                const pos = ti * stride + k;
                if (pos < pad) continue;
                const t_out = pos - pad;
                if (t_out >= t_conv) continue;
                gcol[ti * (taps * out_channels) + oc * taps + k] = gy[t_out * out_channels + oc];
            }
        }
    }
}

test "col2im1d backward vector kernel matches the naive adjoint for the 5 DAC decoder combos" {
    const allocator = std.testing.allocator;

    const combos = [_][4]usize{
        .{ 8, 16, 4, 0 },
        .{ 5, 10, 3, 1 },
        .{ 4, 8, 2, 0 },
        .{ 2, 4, 1, 0 },
        .{ 3, 6, 2, 1 },
    };
    for (combos, 0..) |combo, combo_i| {
        const stride = combo[0];
        const taps = combo[1];
        const pad = combo[2];
        const output_pad = combo[3];
        const t_in: usize = 4;
        const out_channels: usize = 3;
        const t_conv = (t_in - 1) * stride + taps - 2 * pad;
        const gy_len = t_conv + output_pad;

        var gy = try Tensor.zeros(allocator, &.{ gy_len, out_channels });
        defer gy.deinit();
        fillPseudoRandom(gy.data(), 7000 + combo_i);

        var gcol = try Tensor.zeros(allocator, &.{ t_in, taps * out_channels });
        defer gcol.deinit();
        // Poison to prove every cell (incl. cropped/out-of-range taps) is written.
        @memset(gcol.data(), std.math.nan(f32));
        col2im1dBackwardIntoWithConfig(&gcol, &gy, t_in, gy_len, out_channels, taps, stride, pad, .{});

        const want = try allocator.alloc(f32, t_in * taps * out_channels);
        defer allocator.free(want);
        naiveCol2im1dBackward(want, gy.dataConst(), t_in, out_channels, taps, stride, pad);

        try std.testing.expectEqualSlices(f32, want, gcol.dataConst());
    }
}

test "col2im1d backward vector kernel hand-computed gather transpose (s=2, k=2)" {
    const allocator = std.testing.allocator;

    // Adjoint of the forward hand case: pad=0, t_conv=4 — gcol[ti,k] = gy[2ti+k].
    var gy = try Tensor.fromSlice(allocator, &.{ 4, 1 }, &.{ 1, 2, 3, 4 });
    defer gy.deinit();
    var gcol = try Tensor.zeros(allocator, &.{ 2, 2 });
    defer gcol.deinit();
    col2im1dBackwardIntoWithConfig(&gcol, &gy, 2, 4, 1, 2, 2, 0, .{});
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, gcol.dataConst());

    // pad=1, output_pad=1 (t_conv=2, gy has 3 rows): the cropped taps and the
    // output_pad row read back as 0 — gcol = [[0, gy0], [gy1, 0]].
    var gy_pad = try Tensor.fromSlice(allocator, &.{ 3, 1 }, &.{ 5, 6, 7 });
    defer gy_pad.deinit();
    var gcol_pad = try Tensor.zeros(allocator, &.{ 2, 2 });
    defer gcol_pad.deinit();
    col2im1dBackwardIntoWithConfig(&gcol_pad, &gy_pad, 2, 3, 1, 2, 2, 1, .{});
    try std.testing.expectEqualSlices(f32, &.{ 0, 5, 6, 0 }, gcol_pad.dataConst());
}

test "col2im1d vector kernel overlapping taps accumulate (s=1, k=3)" {
    const allocator = std.testing.allocator;

    // t_in=3, oc=1, stride=1, taps=3, pad=0: T_out = 2 + 3 = 5.
    // out[t] = sum over ti of col[ti, t - ti] where 0 <= t - ti < 3.
    var col = try Tensor.fromSlice(allocator, &.{ 3, 3 }, &.{
        1,  2,   4,
        8,  16,  32,
        64, 128, 256,
    });
    defer col.deinit();
    var out = try Tensor.zeros(allocator, &.{ 5, 1 });
    defer out.deinit();
    col2im1dIntoWithConfig(&out, &col, 3, 5, 1, 3, 1, 0, .{});
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 + 8, 4 + 16 + 64, 32 + 128, 256 }, out.dataConst());
}

// ---------------------------------------------------------------------------
// conv2d col2im + parallel backward splits
// ---------------------------------------------------------------------------

const Conv2dDims = conv.Conv2dDims;

/// Naive adjoint of the 2-D im2col gather: scatter every col entry back onto
/// the input position its forward tap read (out-of-range taps read padding
/// and never scatter). Output positions are visited DESCENDING: for a fixed
/// input element the valid (oy, ky) pairs satisfy oy*stride_h + ky = iy +
/// pad_h (one ky per oy), so descending oy/ox delivers each element's
/// contributions in ascending (ky, kx) order — exactly the gather kernel's
/// accumulation order, keeping the comparison free of reassociation noise.
fn naiveCol2imScatter(out: []f32, col: []const f32, d: Conv2dDims) void {
    @memset(out, 0);
    const ksz = d.kh * d.kw * d.cin;
    var oy = d.oh;
    while (oy > 0) {
        oy -= 1;
        var ox = d.ow;
        while (ox > 0) {
            ox -= 1;
            for (0..d.kh) |ky| {
                const iy_s = @as(isize, @intCast(oy * d.stride_h + ky)) - @as(isize, @intCast(d.pad_h));
                if (iy_s < 0 or iy_s >= @as(isize, @intCast(d.h))) continue;
                const iy: usize = @intCast(iy_s);
                for (0..d.kw) |kx| {
                    const ix_s = @as(isize, @intCast(ox * d.stride_w + kx)) - @as(isize, @intCast(d.pad_w));
                    if (ix_s < 0 or ix_s >= @as(isize, @intCast(d.w))) continue;
                    const ix: usize = @intCast(ix_s);
                    for (0..d.cin) |ic| {
                        out[(iy * d.w + ix) * d.cin + ic] += col[(oy * d.ow + ox) * ksz + (ky * d.kw + kx) * d.cin + ic];
                    }
                }
            }
        }
    }
}

test "conv2d col2im matches the naive im2col adjoint" {
    const allocator = std.testing.allocator;

    // (kh, kw, stride_h, stride_w, pad_h, pad_w) — s1/s2 and pad 0/1/2 mixes.
    const combos = [_][6]usize{
        .{ 3, 3, 1, 1, 1, 1 },
        .{ 3, 3, 2, 2, 1, 1 },
        .{ 2, 2, 2, 2, 0, 0 },
        .{ 5, 5, 1, 1, 2, 2 },
        .{ 3, 2, 2, 1, 2, 1 },
        .{ 1, 1, 2, 2, 0, 0 },
    };
    for (combos, 0..) |combo, combo_i| {
        const h: usize = 9;
        const w: usize = 8;
        const cin: usize = 3; // exercises the addRow scalar tail
        const d: Conv2dDims = .{
            .h = h,
            .w = w,
            .cin = cin,
            .oh = (h + 2 * combo[4] - combo[0]) / combo[2] + 1,
            .ow = (w + 2 * combo[5] - combo[1]) / combo[3] + 1,
            .cout = 1, // unused by col2im
            .kh = combo[0],
            .kw = combo[1],
            .stride_h = combo[2],
            .stride_w = combo[3],
            .pad_h = combo[4],
            .pad_w = combo[5],
            .groups = 1,
        };
        const ksz = d.kh * d.kw * d.cin;

        var col = try Tensor.zeros(allocator, &.{ d.oh * d.ow, ksz });
        defer col.deinit();
        fillPseudoRandom(col.data(), 9000 + combo_i);

        var out = try Tensor.zeros(allocator, &.{ h, w, cin });
        defer out.deinit();
        // Poison the output to prove every element is written.
        @memset(out.data(), std.math.nan(f32));
        conv.col2imIntoWithConfig(&out, &col, d, .{});

        const want = try allocator.alloc(f32, h * w * cin);
        defer allocator.free(want);
        naiveCol2imScatter(want, col.dataConst(), d);

        for (want, out.dataConst()) |expected, got| {
            try std.testing.expectApproxEqAbs(expected, got, 1e-6);
        }
    }
}

test "conv2d backward kernels + col2im: pooled split is bit-identical to serial" {
    const allocator = std.testing.allocator;

    var pool: thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .max_workers = 3 });
    defer pool.deinit();

    // 48x48x8 -> 16 with k3 s1 p1 clears the 256Ki work gate in every kernel:
    // backward work = 48*48*16*cin_pg*9 (2.65M dense / 1.33M grouped), col2im
    // work = 48*48*8*16 = 294912.
    const h: usize = 48;
    const w: usize = 48;
    const cin: usize = 8;
    const cout: usize = 16;

    var input = try Tensor.zeros(allocator, &.{ h, w, cin });
    defer input.deinit();
    fillPseudoRandom(input.data(), 41);
    var gy = try Tensor.zeros(allocator, &.{ h, w, cout });
    defer gy.deinit();
    fillPseudoRandom(gy.data(), 42);

    for ([_]usize{ 1, 2 }) |groups| {
        const cin_pg = cin / groups;
        const d: Conv2dDims = .{
            .h = h,
            .w = w,
            .cin = cin,
            .oh = h,
            .ow = w,
            .cout = cout,
            .kh = 3,
            .kw = 3,
            .stride_h = 1,
            .stride_w = 1,
            .pad_h = 1,
            .pad_w = 1,
            .groups = groups,
        };

        var weight = try Tensor.zeros(allocator, &.{ cout, 3, 3, cin_pg });
        defer weight.deinit();
        fillPseudoRandom(weight.data(), 43 + groups);

        var gx_serial = try Tensor.zeros(allocator, &.{ h, w, cin });
        defer gx_serial.deinit();
        var gx_pooled = try Tensor.zeros(allocator, &.{ h, w, cin });
        defer gx_pooled.deinit();
        conv.conv2dBackwardInputIntoWithConfig(&gx_serial, &gy, &weight, d, .{});
        conv.conv2dBackwardInputIntoWithConfig(&gx_pooled, &gy, &weight, d, .{ .pool = &pool });
        try std.testing.expectEqualSlices(f32, gx_serial.dataConst(), gx_pooled.dataConst());

        var gw_serial = try Tensor.zeros(allocator, &.{ cout, 3, 3, cin_pg });
        defer gw_serial.deinit();
        var gw_pooled = try Tensor.zeros(allocator, &.{ cout, 3, 3, cin_pg });
        defer gw_pooled.deinit();
        conv.conv2dBackwardWeightIntoWithConfig(&gw_serial, &input, &gy, d, .{});
        conv.conv2dBackwardWeightIntoWithConfig(&gw_pooled, &input, &gy, d, .{ .pool = &pool });
        try std.testing.expectEqualSlices(f32, gw_serial.dataConst(), gw_pooled.dataConst());

        if (groups == 1) {
            const ksz = d.kh * d.kw * d.cin;
            var col = try Tensor.zeros(allocator, &.{ d.oh * d.ow, ksz });
            defer col.deinit();
            fillPseudoRandom(col.data(), 45);
            var c2i_serial = try Tensor.zeros(allocator, &.{ h, w, cin });
            defer c2i_serial.deinit();
            var c2i_pooled = try Tensor.zeros(allocator, &.{ h, w, cin });
            defer c2i_pooled.deinit();
            conv.col2imIntoWithConfig(&c2i_serial, &col, d, .{});
            conv.col2imIntoWithConfig(&c2i_pooled, &col, d, .{ .pool = &pool });
            try std.testing.expectEqualSlices(f32, c2i_serial.dataConst(), c2i_pooled.dataConst());
        }
    }
}
