//! Small-tap causal convolution kernels.
//!
//! Depthwise family: covers the causal FIR used by DeltaNet-style blocks —
//! input/output are contiguous `[time, channel]`, kernel is `[channel, tap]`,
//! and optional state is the `tap - 1` historical rows preceding the input.
//!
//! General family (`causalConv1d*`): channel-mixing dilated causal conv —
//! input `[time, in]`, weight `[tap, in, out]` (tap `taps-1` is the newest
//! sample), output `[time, out]`, optional state is the `dilation*(taps-1)`
//! historical input rows preceding the chunk, oldest first. The `[tap, in,
//! out]` weight layout keeps every kernel on contiguous out-channel rows:
//! forward and backward-weight are axpy accumulations, backward-input is a
//! contiguous dot.

const parallel = @import("../../parallel.zig");
const std = @import("std");
const tensor = @import("../../tensor.zig");
const vm = @import("common.zig");

const Tensor = tensor.Tensor;
const ParallelConfig = vm.ParallelConfig;
const Vf32 = vm.Vf32;
const vector_len = vm.vector_len;

pub fn causalDepthwiseConv1dIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    kernel: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, seq * channels);
    const input_data = vm.contiguousDataConst(input, seq * channels);
    const kernel_data = vm.contiguousDataConst(kernel, channels * taps);
    if (maybeParallelConv(config, runForwardTask, output, input_data, kernel_data, null, state, seq, channels, taps)) return;
    forwardRange(output, input_data, kernel_data, state, seq, channels, taps, 0, channels);
}

pub fn causalDepthwiseConv1dBackwardInputIntoWithConfig(
    out: *Tensor,
    gy: *const Tensor,
    kernel: *const Tensor,
    seq: usize,
    channels: usize,
    taps: usize,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, seq * channels);
    const gy_data = vm.contiguousDataConst(gy, seq * channels);
    const kernel_data = vm.contiguousDataConst(kernel, channels * taps);
    if (maybeParallelConv(config, runBackwardInputTask, output, gy_data, kernel_data, null, null, seq, channels, taps)) return;
    backwardInputRange(output, gy_data, kernel_data, seq, channels, taps, 0, channels);
}

pub fn causalDepthwiseConv1dBackwardKernelIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, channels * taps);
    const input_data = vm.contiguousDataConst(input, seq * channels);
    const gy_data = vm.contiguousDataConst(gy, seq * channels);
    if (maybeParallelConv(config, runBackwardKernelTask, output, input_data, undefined, gy_data, state, seq, channels, taps)) return;
    backwardKernelRange(output, input_data, gy_data, state, seq, channels, taps, 0, channels);
}

const ConvTask = struct {
    out: []f32,
    input: []const f32,
    kernel: []const f32,
    gy: []const f32,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    channel_start: usize,
    channel_end: usize,
};

fn maybeParallelConv(
    config: ParallelConfig,
    comptime runTask: fn (*const ConvTask) void,
    out: []f32,
    input: []const f32,
    kernel: []const f32,
    gy: ?[]const f32,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
) bool {
    const pool = config.pool orelse return false;
    const thread_count = vm.depthwiseConvThreadCount(seq, channels, taps);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]ConvTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .out = out,
            .input = input,
            .kernel = kernel,
            .gy = gy orelse &.{},
            .state = state,
            .seq = seq,
            .channels = channels,
            .taps = taps,
            .channel_start = task_i * channels / thread_count,
            .channel_end = (task_i + 1) * channels / thread_count,
        };
    }
    pool.parallelChunks(ConvTask, tasks[0..thread_count], runTask);
    return true;
}

fn runForwardTask(task: *const ConvTask) void {
    forwardRange(task.out, task.input, task.kernel, task.state, task.seq, task.channels, task.taps, task.channel_start, task.channel_end);
}

fn runBackwardInputTask(task: *const ConvTask) void {
    backwardInputRange(task.out, task.input, task.kernel, task.seq, task.channels, task.taps, task.channel_start, task.channel_end);
}

fn runBackwardKernelTask(task: *const ConvTask) void {
    backwardKernelRange(task.out, task.input, task.gy, task.state, task.seq, task.channels, task.taps, task.channel_start, task.channel_end);
}

fn forwardRange(
    out: []f32,
    input: []const f32,
    kernel: []const f32,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = taps - 1;
    for (0..seq) |t| {
        var c = channel_start;
        while (c + vector_len <= channel_end) : (c += vector_len) {
            var acc: Vf32 = @splat(0);
            for (0..taps) |k| {
                const x: Vf32 = if (t + k >= pad)
                    input[(t + k - pad) * channels + c ..][0..vector_len].*
                else if (state) |s|
                    s[(t + k) * channels + c ..][0..vector_len].*
                else
                    @splat(0);
                acc += x * loadKernelVector(kernel, c, taps, k);
            }
            out[t * channels + c ..][0..vector_len].* = acc;
        }
        while (c < channel_end) : (c += 1) {
            var acc: f32 = 0;
            for (0..taps) |k| {
                acc += inputValue(input, state, channels, pad, t, c, k) * kernel[c * taps + k];
            }
            out[t * channels + c] = acc;
        }
    }
}

fn backwardInputRange(
    out: []f32,
    gy: []const f32,
    kernel: []const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = taps - 1;
    for (0..seq) |p| {
        var c = channel_start;
        while (c + vector_len <= channel_end) : (c += vector_len) {
            var acc: Vf32 = @splat(0);
            for (0..taps) |k| {
                const t_base = p + pad;
                if (k > t_base) continue;
                const t = t_base - k;
                if (t < seq) {
                    const g: Vf32 = gy[t * channels + c ..][0..vector_len].*;
                    acc += g * loadKernelVector(kernel, c, taps, k);
                }
            }
            out[p * channels + c ..][0..vector_len].* = acc;
        }
        while (c < channel_end) : (c += 1) {
            var acc: f32 = 0;
            for (0..taps) |k| {
                const t_base = p + pad;
                if (k > t_base) continue;
                const t = t_base - k;
                if (t < seq) acc += gy[t * channels + c] * kernel[c * taps + k];
            }
            out[p * channels + c] = acc;
        }
    }
}

fn backwardKernelRange(
    out: []f32,
    input: []const f32,
    gy: []const f32,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = taps - 1;
    for (channel_start..channel_end) |c| {
        for (0..taps) |k| {
            var acc: f32 = 0;
            for (0..seq) |t| {
                acc += gy[t * channels + c] * inputValue(input, state, channels, pad, t, c, k);
            }
            out[c * taps + k] = acc;
        }
    }
}

pub fn causalConv1dIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, seq * out_channels);
    const input_data = vm.contiguousDataConst(input, seq * in_channels);
    const weight_data = vm.contiguousDataConst(weight, taps * in_channels * out_channels);
    if (maybeParallelGeneralConv(config, runGeneralForwardTask, seq, output, input_data, weight_data, null, state, seq, in_channels, out_channels, taps, dilation, 1)) return;
    generalForwardRange(output, input_data, weight_data, state, in_channels, out_channels, taps, dilation, 1, 0, seq);
}

pub fn causalConv1dBackwardInputIntoWithConfig(
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, seq * in_channels);
    const gy_data = vm.contiguousDataConst(gy, seq * out_channels);
    const weight_data = vm.contiguousDataConst(weight, taps * in_channels * out_channels);
    if (maybeParallelGeneralConv(config, runGeneralBackwardInputTask, seq, output, &.{}, weight_data, gy_data, null, seq, in_channels, out_channels, taps, dilation, 1)) return;
    generalBackwardInputRange(output, gy_data, weight_data, seq, in_channels, out_channels, taps, dilation, 1, 0, seq);
}

pub fn causalConv1dBackwardWeightIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, taps * in_channels * out_channels);
    const input_data = vm.contiguousDataConst(input, seq * in_channels);
    const gy_data = vm.contiguousDataConst(gy, seq * out_channels);
    const rows = taps * in_channels;
    if (maybeParallelGeneralConv(config, runGeneralBackwardWeightTask, rows, output, input_data, &.{}, gy_data, state, seq, in_channels, out_channels, taps, dilation, 1)) return;
    generalBackwardWeightRange(output, input_data, gy_data, state, seq, in_channels, out_channels, taps, dilation, 1, 0, rows);
}

pub fn groupedCausalConv1dIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, seq * out_channels);
    const input_data = vm.contiguousDataConst(input, seq * in_channels);
    const in_per_group = in_channels / groups;
    const weight_data = vm.contiguousDataConst(weight, taps * in_per_group * out_channels);
    if (maybeParallelGeneralConv(config, runGeneralForwardTask, seq, output, input_data, weight_data, null, state, seq, in_channels, out_channels, taps, dilation, groups)) return;
    generalForwardRange(output, input_data, weight_data, state, in_channels, out_channels, taps, dilation, groups, 0, seq);
}

pub fn groupedCausalConv1dBackwardInputIntoWithConfig(
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, seq * in_channels);
    const gy_data = vm.contiguousDataConst(gy, seq * out_channels);
    const in_per_group = in_channels / groups;
    const weight_data = vm.contiguousDataConst(weight, taps * in_per_group * out_channels);
    if (maybeParallelGeneralConv(config, runGeneralBackwardInputTask, seq, output, &.{}, weight_data, gy_data, null, seq, in_channels, out_channels, taps, dilation, groups)) return;
    generalBackwardInputRange(output, gy_data, weight_data, seq, in_channels, out_channels, taps, dilation, groups, 0, seq);
}

pub fn groupedCausalConv1dBackwardWeightIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    config: ParallelConfig,
) void {
    const in_per_group = in_channels / groups;
    const output = vm.contiguousData(out, taps * in_per_group * out_channels);
    const input_data = vm.contiguousDataConst(input, seq * in_channels);
    const gy_data = vm.contiguousDataConst(gy, seq * out_channels);
    const rows = taps * in_per_group;
    if (maybeParallelGeneralConv(config, runGeneralBackwardWeightTask, rows, output, input_data, &.{}, gy_data, state, seq, in_channels, out_channels, taps, dilation, groups)) return;
    generalBackwardWeightRange(output, input_data, gy_data, state, seq, in_channels, out_channels, taps, dilation, groups, 0, rows);
}

const GeneralConvTask = struct {
    out: []f32,
    input: []const f32,
    weight: []const f32,
    gy: []const f32,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    start: usize,
    end: usize,
};

fn maybeParallelGeneralConv(
    config: ParallelConfig,
    comptime runTask: fn (*const GeneralConvTask) void,
    split: usize,
    out: []f32,
    input: []const f32,
    weight: []const f32,
    gy: ?[]const f32,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
) bool {
    const pool = config.pool orelse return false;
    const work = std.math.mul(usize, parallel.saturatedMul3(seq, in_channels, out_channels), taps) catch std.math.maxInt(usize);
    const thread_count = vm.generalConvThreadCount(split, work);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]GeneralConvTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .out = out,
            .input = input,
            .weight = weight,
            .gy = gy orelse &.{},
            .state = state,
            .seq = seq,
            .in_channels = in_channels,
            .out_channels = out_channels,
            .taps = taps,
            .dilation = dilation,
            .groups = groups,
            .start = task_i * split / thread_count,
            .end = (task_i + 1) * split / thread_count,
        };
    }
    pool.parallelChunks(GeneralConvTask, tasks[0..thread_count], runTask);
    return true;
}

fn runGeneralForwardTask(task: *const GeneralConvTask) void {
    generalForwardRange(task.out, task.input, task.weight, task.state, task.in_channels, task.out_channels, task.taps, task.dilation, task.groups, task.start, task.end);
}

fn runGeneralBackwardInputTask(task: *const GeneralConvTask) void {
    generalBackwardInputRange(task.out, task.gy, task.weight, task.seq, task.in_channels, task.out_channels, task.taps, task.dilation, task.groups, task.start, task.end);
}

fn runGeneralBackwardWeightTask(task: *const GeneralConvTask) void {
    generalBackwardWeightRange(task.out, task.input, task.gy, task.state, task.seq, task.in_channels, task.out_channels, task.taps, task.dilation, task.groups, task.start, task.end);
}

/// Resolves the input row feeding tap `k` at output time `t`: the chunk's own
/// rows once `t + k*dilation` clears the causal pad, the state rows before
/// that, zeros (null) when no state is given.
inline fn generalConvInputRow(
    input: []const f32,
    state: ?[]const f32,
    in_channels: usize,
    pad: usize,
    t: usize,
    k: usize,
    dilation: usize,
) ?[]const f32 {
    const shifted = t + k * dilation;
    if (shifted >= pad) return input[(shifted - pad) * in_channels ..][0..in_channels];
    const s = state orelse return null;
    return s[shifted * in_channels ..][0..in_channels];
}

fn generalForwardRange(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    state: ?[]const f32,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    t_start: usize,
    t_end: usize,
) void {
    if (taps == 1) {
        generalForward1x1Range(out, input, weight, in_channels, out_channels, groups, t_start, t_end);
        return;
    }
    if (groups == 1) {
        if (in_channels == 8 and out_channels == 16) {
            if (comptime 16 % vector_len == 0) {
                generalForwardFixedDenseRange(8, 16, out, input, weight, state, taps, dilation, t_start, t_end);
            } else {
                generalForwardFixedDenseScalarRange(8, 16, out, input, weight, state, taps, dilation, t_start, t_end);
            }
            return;
        }
        if (in_channels == 8 and out_channels == 8) {
            if (comptime 8 % vector_len == 0) {
                generalForwardFixedDenseRange(8, 8, out, input, weight, state, taps, dilation, t_start, t_end);
            } else {
                generalForwardFixedDenseScalarRange(8, 8, out, input, weight, state, taps, dilation, t_start, t_end);
            }
            return;
        }
        if (in_channels == 8 and out_channels == 1) {
            generalForwardFixedDenseScalarRange(8, 1, out, input, weight, state, taps, dilation, t_start, t_end);
            return;
        }
        if (in_channels == 4 and out_channels == 8) {
            if (comptime 8 % vector_len == 0) {
                generalForwardFixedDenseRange(4, 8, out, input, weight, state, taps, dilation, t_start, t_end);
            } else {
                generalForwardFixedDenseScalarRange(4, 8, out, input, weight, state, taps, dilation, t_start, t_end);
            }
            return;
        }
        if (in_channels == 4 and out_channels == 4) {
            if (comptime 4 % vector_len == 0) {
                generalForwardFixedDenseRange(4, 4, out, input, weight, state, taps, dilation, t_start, t_end);
            } else {
                generalForwardFixedDenseScalarRange(4, 4, out, input, weight, state, taps, dilation, t_start, t_end);
            }
            return;
        }
        if (in_channels == 3 and out_channels == 3) {
            generalForwardFixedDenseScalarRange(3, 3, out, input, weight, state, taps, dilation, t_start, t_end);
            return;
        }
        if (in_channels == 3 and out_channels == 1) {
            generalForwardFixedDenseScalarRange(3, 1, out, input, weight, state, taps, dilation, t_start, t_end);
            return;
        }
        if (in_channels == 2 and out_channels == 4) {
            if (comptime 4 % vector_len == 0) {
                generalForwardFixedDenseRange(2, 4, out, input, weight, state, taps, dilation, t_start, t_end);
            } else {
                generalForwardFixedDenseScalarRange(2, 4, out, input, weight, state, taps, dilation, t_start, t_end);
            }
            return;
        }
        if (in_channels == 2 and out_channels == 2) {
            generalForwardFixedDenseScalarRange(2, 2, out, input, weight, state, taps, dilation, t_start, t_end);
            return;
        }
        if (in_channels == 2 and out_channels == 1) {
            generalForwardFixedDenseScalarRange(2, 1, out, input, weight, state, taps, dilation, t_start, t_end);
            return;
        }
    }
    const pad = dilation * (taps - 1);
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    for (t_start..t_end) |t| {
        const out_row = out[t * out_channels ..][0..out_channels];
        @memset(out_row, 0);
        for (0..taps) |k| {
            const x_row = generalConvInputRow(input, state, in_channels, pad, t, k, dilation) orelse continue;
            if (groups == 1) {
                for (0..in_channels) |i| {
                    axpyRow(out_row, x_row[i], weight[(k * in_channels + i) * out_channels ..][0..out_channels]);
                }
                continue;
            }
            for (0..groups) |group| {
                const input_start = group * in_per_group;
                const out_start = group * out_per_group;
                const out_part = out_row[out_start..][0..out_per_group];
                for (0..in_per_group) |local_i| {
                    axpyRow(out_part, x_row[input_start + local_i], weight[(k * in_per_group + local_i) * out_channels + out_start ..][0..out_per_group]);
                }
            }
        }
    }
}

fn generalForwardFixedDenseRange(
    comptime in_channels: usize,
    comptime out_channels: usize,
    out: []f32,
    input: []const f32,
    weight: []const f32,
    state: ?[]const f32,
    taps: usize,
    dilation: usize,
    t_start: usize,
    t_end: usize,
) void {
    comptime std.debug.assert(out_channels % vector_len == 0);
    const vec_blocks = out_channels / vector_len;
    const pad = dilation * (taps - 1);
    for (t_start..t_end) |t| {
        var acc: [vec_blocks]Vf32 = undefined;
        inline for (0..vec_blocks) |b| acc[b] = @splat(0);
        for (0..taps) |k| {
            const x_row = generalConvInputRow(input, state, in_channels, pad, t, k, dilation) orelse continue;
            inline for (0..in_channels) |i| {
                const sv: Vf32 = @splat(x_row[i]);
                const base = (k * in_channels + i) * out_channels;
                inline for (0..vec_blocks) |b| {
                    const w: Vf32 = weight[base + b * vector_len ..][0..vector_len].*;
                    acc[b] = @mulAdd(Vf32, sv, w, acc[b]);
                }
            }
        }
        const out_row = out[t * out_channels ..][0..out_channels];
        inline for (0..vec_blocks) |b| {
            out_row[b * vector_len ..][0..vector_len].* = acc[b];
        }
    }
}

fn generalForwardFixedDenseScalarRange(
    comptime in_channels: usize,
    comptime out_channels: usize,
    out: []f32,
    input: []const f32,
    weight: []const f32,
    state: ?[]const f32,
    taps: usize,
    dilation: usize,
    t_start: usize,
    t_end: usize,
) void {
    const pad = dilation * (taps - 1);
    for (t_start..t_end) |t| {
        var acc: [out_channels]f32 = [_]f32{0} ** out_channels;
        for (0..taps) |k| {
            const x_row = generalConvInputRow(input, state, in_channels, pad, t, k, dilation) orelse continue;
            inline for (0..in_channels) |i| {
                const s = x_row[i];
                const base = (k * in_channels + i) * out_channels;
                inline for (0..out_channels) |o| {
                    acc[o] = @mulAdd(f32, s, weight[base + o], acc[o]);
                }
            }
        }
        const out_row = out[t * out_channels ..][0..out_channels];
        inline for (0..out_channels) |o| {
            out_row[o] = acc[o];
        }
    }
}

fn generalForward1x1Range(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    t_start: usize,
    t_end: usize,
) void {
    if (groups == 1) {
        if (in_channels == 8 and out_channels == 8) {
            if (comptime 8 % vector_len == 0) {
                generalForwardFixedDense1x1Range(8, 8, out, input, weight, t_start, t_end);
            } else {
                generalForwardFixedDense1x1ScalarRange(8, 8, out, input, weight, t_start, t_end);
            }
            return;
        }
        if (in_channels == 4 and out_channels == 4) {
            if (comptime 4 % vector_len == 0) {
                generalForwardFixedDense1x1Range(4, 4, out, input, weight, t_start, t_end);
            } else {
                generalForwardFixedDense1x1ScalarRange(4, 4, out, input, weight, t_start, t_end);
            }
            return;
        }
        if (in_channels == 3 and out_channels == 3) {
            generalForwardFixedDense1x1ScalarRange(3, 3, out, input, weight, t_start, t_end);
            return;
        }
        if (in_channels == 2 and out_channels == 2) {
            generalForwardFixedDense1x1ScalarRange(2, 2, out, input, weight, t_start, t_end);
            return;
        }
    }
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    for (t_start..t_end) |t| {
        const x_row = input[t * in_channels ..][0..in_channels];
        const out_row = out[t * out_channels ..][0..out_channels];
        if (groups == 1) {
            if (in_channels == 1) {
                scaleRow(out_row, x_row[0], weight[0..out_channels]);
                continue;
            }
            @memset(out_row, 0);
            for (0..in_channels) |i| {
                axpyRow(out_row, x_row[i], weight[i * out_channels ..][0..out_channels]);
            }
            continue;
        }
        if (in_per_group == 1) {
            for (0..groups) |group| {
                const out_start = group * out_per_group;
                const out_part = out_row[out_start..][0..out_per_group];
                scaleRow(out_part, x_row[group], weight[out_start..][0..out_per_group]);
            }
            continue;
        }
        @memset(out_row, 0);
        for (0..groups) |group| {
            const input_start = group * in_per_group;
            const out_start = group * out_per_group;
            const out_part = out_row[out_start..][0..out_per_group];
            for (0..in_per_group) |local_i| {
                axpyRow(out_part, x_row[input_start + local_i], weight[local_i * out_channels + out_start ..][0..out_per_group]);
            }
        }
    }
}

fn generalForwardFixedDense1x1Range(
    comptime in_channels: usize,
    comptime out_channels: usize,
    out: []f32,
    input: []const f32,
    weight: []const f32,
    t_start: usize,
    t_end: usize,
) void {
    comptime std.debug.assert(out_channels % vector_len == 0);
    const vec_blocks = out_channels / vector_len;
    for (t_start..t_end) |t| {
        const x_row = input[t * in_channels ..][0..in_channels];
        var acc: [vec_blocks]Vf32 = undefined;
        inline for (0..vec_blocks) |b| acc[b] = @splat(0);
        inline for (0..in_channels) |i| {
            const sv: Vf32 = @splat(x_row[i]);
            const base = i * out_channels;
            inline for (0..vec_blocks) |b| {
                const w: Vf32 = weight[base + b * vector_len ..][0..vector_len].*;
                acc[b] = @mulAdd(Vf32, sv, w, acc[b]);
            }
        }
        const out_row = out[t * out_channels ..][0..out_channels];
        inline for (0..vec_blocks) |b| {
            out_row[b * vector_len ..][0..vector_len].* = acc[b];
        }
    }
}

fn generalForwardFixedDense1x1ScalarRange(
    comptime in_channels: usize,
    comptime out_channels: usize,
    out: []f32,
    input: []const f32,
    weight: []const f32,
    t_start: usize,
    t_end: usize,
) void {
    for (t_start..t_end) |t| {
        const x_row = input[t * in_channels ..][0..in_channels];
        var acc: [out_channels]f32 = [_]f32{0} ** out_channels;
        inline for (0..in_channels) |i| {
            const s = x_row[i];
            const base = i * out_channels;
            inline for (0..out_channels) |o| {
                acc[o] = @mulAdd(f32, s, weight[base + o], acc[o]);
            }
        }
        const out_row = out[t * out_channels ..][0..out_channels];
        inline for (0..out_channels) |o| {
            out_row[o] = acc[o];
        }
    }
}

fn generalBackwardInputRange(
    out: []f32,
    gy: []const f32,
    weight: []const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    p_start: usize,
    p_end: usize,
) void {
    if (taps == 1) {
        generalBackwardInput1x1Range(out, gy, weight, in_channels, out_channels, groups, p_start, p_end);
        return;
    }
    const pad = dilation * (taps - 1);
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    for (p_start..p_end) |p| {
        const gx_row = out[p * in_channels ..][0..in_channels];
        @memset(gx_row, 0);
        for (0..taps) |k| {
            const t = p + pad - k * dilation;
            if (t >= seq) continue;
            const gy_row = gy[t * out_channels ..][0..out_channels];
            if (groups == 1) {
                for (0..in_channels) |i| {
                    gx_row[i] += dotRow(gy_row, weight[(k * in_channels + i) * out_channels ..][0..out_channels]);
                }
                continue;
            }
            for (0..groups) |group| {
                const out_start = group * out_per_group;
                const input_start = group * in_per_group;
                const gy_part = gy_row[out_start..][0..out_per_group];
                for (0..in_per_group) |local_i| {
                    gx_row[input_start + local_i] += dotRow(gy_part, weight[(k * in_per_group + local_i) * out_channels + out_start ..][0..out_per_group]);
                }
            }
        }
    }
}

fn generalBackwardInput1x1Range(
    out: []f32,
    gy: []const f32,
    weight: []const f32,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    p_start: usize,
    p_end: usize,
) void {
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    for (p_start..p_end) |p| {
        const gx_row = out[p * in_channels ..][0..in_channels];
        const gy_row = gy[p * out_channels ..][0..out_channels];
        if (groups == 1) {
            for (0..in_channels) |i| {
                gx_row[i] = dotRow(gy_row, weight[i * out_channels ..][0..out_channels]);
            }
            continue;
        }
        for (0..groups) |group| {
            const input_start = group * in_per_group;
            const out_start = group * out_per_group;
            const gy_part = gy_row[out_start..][0..out_per_group];
            for (0..in_per_group) |local_i| {
                gx_row[input_start + local_i] = dotRow(gy_part, weight[local_i * out_channels + out_start ..][0..out_per_group]);
            }
        }
    }
}

fn generalBackwardWeightRange(
    out: []f32,
    input: []const f32,
    gy: []const f32,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    row_start: usize,
    row_end: usize,
) void {
    if (taps == 1) {
        generalBackwardWeight1x1Range(out, input, gy, seq, in_channels, out_channels, groups, row_start, row_end);
        return;
    }
    if (state == null) {
        generalBackwardWeightNoStateRange(out, input, gy, seq, in_channels, out_channels, taps, dilation, groups, row_start, row_end);
        return;
    }
    const pad = dilation * (taps - 1);
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    for (row_start..row_end) |row| {
        const k = row / in_per_group;
        const local_i = row % in_per_group;
        const gw_row = out[row * out_channels ..][0..out_channels];
        @memset(gw_row, 0);
        for (0..seq) |t| {
            const x_row = generalConvInputRow(input, state, in_channels, pad, t, k, dilation) orelse continue;
            if (groups == 1) {
                axpyRow(gw_row, x_row[local_i], gy[t * out_channels ..][0..out_channels]);
                continue;
            }
            const gy_row = gy[t * out_channels ..][0..out_channels];
            for (0..groups) |group| {
                const out_start = group * out_per_group;
                const i = group * in_per_group + local_i;
                axpyRow(gw_row[out_start..][0..out_per_group], x_row[i], gy_row[out_start..][0..out_per_group]);
            }
        }
    }
}

fn generalBackwardWeightNoStateRange(
    out: []f32,
    input: []const f32,
    gy: []const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    row_start: usize,
    row_end: usize,
) void {
    const pad = dilation * (taps - 1);
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    for (row_start..row_end) |row| {
        const k = row / in_per_group;
        const local_i = row % in_per_group;
        const gw_row = out[row * out_channels ..][0..out_channels];
        @memset(gw_row, 0);
        const source_offset = k * dilation;
        const t_start = if (source_offset >= pad) 0 else pad - source_offset;
        if (t_start >= seq) continue;
        for (t_start..seq) |t| {
            const x_row = input[(t + source_offset - pad) * in_channels ..][0..in_channels];
            if (groups == 1) {
                axpyRow(gw_row, x_row[local_i], gy[t * out_channels ..][0..out_channels]);
                continue;
            }
            const gy_row = gy[t * out_channels ..][0..out_channels];
            for (0..groups) |group| {
                const out_start = group * out_per_group;
                const i = group * in_per_group + local_i;
                axpyRow(gw_row[out_start..][0..out_per_group], x_row[i], gy_row[out_start..][0..out_per_group]);
            }
        }
    }
}

fn generalBackwardWeight1x1Range(
    out: []f32,
    input: []const f32,
    gy: []const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    row_start: usize,
    row_end: usize,
) void {
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    for (row_start..row_end) |local_i| {
        const gw_row = out[local_i * out_channels ..][0..out_channels];
        @memset(gw_row, 0);
        for (0..seq) |t| {
            const x_row = input[t * in_channels ..][0..in_channels];
            if (groups == 1) {
                axpyRow(gw_row, x_row[local_i], gy[t * out_channels ..][0..out_channels]);
                continue;
            }
            const gy_row = gy[t * out_channels ..][0..out_channels];
            for (0..groups) |group| {
                const out_start = group * out_per_group;
                const i = group * in_per_group + local_i;
                axpyRow(gw_row[out_start..][0..out_per_group], x_row[i], gy_row[out_start..][0..out_per_group]);
            }
        }
    }
}

inline fn axpyRow(acc: []f32, scalar: f32, row: []const f32) void {
    const sv: Vf32 = @splat(scalar);
    var o: usize = 0;
    while (o + vector_len <= acc.len) : (o += vector_len) {
        const cur: Vf32 = acc[o..][0..vector_len].*;
        const rv: Vf32 = row[o..][0..vector_len].*;
        acc[o..][0..vector_len].* = cur + sv * rv;
    }
    while (o < acc.len) : (o += 1) acc[o] += scalar * row[o];
}

/// Vector-chunked `acc += row` (axpyRow without the multiply).
inline fn addRow(acc: []f32, row: []const f32) void {
    var o: usize = 0;
    while (o + vector_len <= acc.len) : (o += vector_len) {
        const cur: Vf32 = acc[o..][0..vector_len].*;
        const rv: Vf32 = row[o..][0..vector_len].*;
        acc[o..][0..vector_len].* = cur + rv;
    }
    while (o < acc.len) : (o += 1) acc[o] += row[o];
}

inline fn scaleRow(out: []f32, scalar: f32, row: []const f32) void {
    const sv: Vf32 = @splat(scalar);
    var o: usize = 0;
    while (o + vector_len <= out.len) : (o += vector_len) {
        const rv: Vf32 = row[o..][0..vector_len].*;
        out[o..][0..vector_len].* = sv * rv;
    }
    while (o < out.len) : (o += 1) out[o] = scalar * row[o];
}

inline fn dotRow(a: []const f32, b: []const f32) f32 {
    var accv: Vf32 = @splat(0);
    var o: usize = 0;
    while (o + vector_len <= a.len) : (o += vector_len) {
        const av: Vf32 = a[o..][0..vector_len].*;
        const bv: Vf32 = b[o..][0..vector_len].*;
        accv += av * bv;
    }
    var acc: f32 = @reduce(.Add, accv);
    while (o < a.len) : (o += 1) acc += a[o] * b[o];
    return acc;
}

inline fn loadKernelVector(kernel: []const f32, c: usize, taps: usize, k: usize) Vf32 {
    var values: Vf32 = undefined;
    inline for (0..vector_len) |lane| {
        values[lane] = kernel[(c + lane) * taps + k];
    }
    return values;
}

inline fn inputValue(
    input: []const f32,
    state: ?[]const f32,
    channels: usize,
    pad: usize,
    t: usize,
    c: usize,
    k: usize,
) f32 {
    if (t + k >= pad) return input[(t + k - pad) * channels + c];
    const s = state orelse return 0;
    return s[(t + k) * channels + c];
}

// ===========================================================================
// conv2d — rank-3 channel-last [H,W,Cin] -> [OH,OW,Cout], with stride,
// explicit zero padding, and grouped/depthwise support. Used by the Parakeet
// FastConformer subsampling stem. Allocation-free; caller validates shapes.
// ===========================================================================

/// Geometry for `conv2dIntoWithConfig`. Channel-last tensors:
///   input  `in[(h*W + w)*Cin + c]`   weight `w[((oc*KH+kh)*KW+kw)*Cin_pg + ic]`
///   output `out[(oh*OW + ow)*Cout + oc]`
/// Cin_pg = Cin/groups, Cout_pg = Cout/groups; output channel `oc` is in group
/// `oc / Cout_pg` and reads input channels `[g*Cin_pg, (g+1)*Cin_pg)`. Depthwise
/// is groups == Cin (Cin_pg = 1).
pub const Conv2dDims = struct {
    h: usize,
    w: usize,
    cin: usize,
    oh: usize,
    ow: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    groups: usize,
};

const Conv2dTask = struct {
    out: []f32,
    in: []const f32,
    w: []const f32,
    bias: ?[]const f32,
    d: Conv2dDims,
    oh_start: usize,
    oh_end: usize,
};

fn runConv2dTask(task: *const Conv2dTask) void {
    conv2dRangeRows(task.out, task.in, task.w, task.bias, task.d, task.oh_start, task.oh_end);
}

pub fn conv2dIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    bias: ?[]const f32,
    d: Conv2dDims,
    config: ParallelConfig,
) void {
    const o = out.data();
    const in = input.dataConst();
    const wt = weight.dataConst();
    // Parallelize over output rows (oh) — each oh writes a disjoint output
    // range and its accumulation is independent, so the result is bit-identical to
    // the serial path (pure parallelization, no numeric change). The general conv
    // stem (subsampling) was the #1 cost at ~48% — single-threaded before.
    if (config.pool) |pool| {
        const cin_pg = d.cin / d.groups;
        const work = d.oh * d.ow * d.cout * d.kh * d.kw * cin_pg;
        const tc = vm.generalConvThreadCount(d.oh, work);
        if (tc > 1) {
            var tasks: [parallel.vector_max_threads]Conv2dTask = undefined;
            for (0..tc) |ti| {
                tasks[ti] = .{
                    .out = o,
                    .in = in,
                    .w = wt,
                    .bias = bias,
                    .d = d,
                    .oh_start = ti * d.oh / tc,
                    .oh_end = (ti + 1) * d.oh / tc,
                };
            }
            pool.parallelChunks(Conv2dTask, tasks[0..tc], runConv2dTask);
            return;
        }
    }
    conv2dRangeRows(o, in, wt, bias, d, 0, d.oh);
}

/// Direct conv2d (correctness-first; f32 accumulation to match ggml's f32 conv
/// path). Full-output convenience wrapper over `conv2dRangeRows`.
pub fn conv2dRange(out: []f32, in: []const f32, w: []const f32, bias: ?[]const f32, d: Conv2dDims) void {
    conv2dRangeRows(out, in, w, bias, d, 0, d.oh);
}

const Im2colTask = struct {
    col: []f32,
    in: []const f32,
    d: Conv2dDims,
    oh_start: usize,
    oh_end: usize,
};

fn runIm2colTask(task: *const Im2colTask) void {
    im2colRangeRows(task.col, task.in, task.d, task.oh_start, task.oh_end);
}

/// im2col gather for the groups==1 conv2d GEMM route:
/// `col[(oy·OW+ox)·ksz + (ky·KW+kx)·Cin + ic]` = the padded input tap, with
/// `ksz = KH·KW·Cin` and out-of-range taps left zero. Pure data movement
/// (zero-fill + `Cin`-wide `@memcpy`), parallel over output rows — each `oy`
/// writes a disjoint `col` range, so the result is bit-identical to serial.
pub fn im2colIntoWithConfig(col: *Tensor, input: *const Tensor, d: Conv2dDims, config: ParallelConfig) void {
    const cd = col.data();
    const in = input.dataConst();
    if (config.pool) |pool| {
        const work = d.oh * d.ow * d.kh * d.kw * d.cin;
        const tc = vm.generalConvThreadCount(d.oh, work);
        if (tc > 1) {
            var tasks: [parallel.vector_max_threads]Im2colTask = undefined;
            for (0..tc) |ti| {
                tasks[ti] = .{
                    .col = cd,
                    .in = in,
                    .d = d,
                    .oh_start = ti * d.oh / tc,
                    .oh_end = (ti + 1) * d.oh / tc,
                };
            }
            pool.parallelChunks(Im2colTask, tasks[0..tc], runIm2colTask);
            return;
        }
    }
    im2colRangeRows(cd, in, d, 0, d.oh);
}

fn im2colRangeRows(col: []f32, in: []const f32, d: Conv2dDims, oh_start: usize, oh_end: usize) void {
    const ksz = d.kh * d.kw * d.cin;
    @memset(col[oh_start * d.ow * ksz .. oh_end * d.ow * ksz], 0);
    var oy: usize = oh_start;
    while (oy < oh_end) : (oy += 1) {
        var ox: usize = 0;
        while (ox < d.ow) : (ox += 1) {
            const dst_pos = (oy * d.ow + ox) * ksz;
            var ky: usize = 0;
            while (ky < d.kh) : (ky += 1) {
                const iy_s = @as(isize, @intCast(oy * d.stride_h + ky)) - @as(isize, @intCast(d.pad_h));
                if (iy_s < 0 or iy_s >= @as(isize, @intCast(d.h))) continue;
                const iy: usize = @intCast(iy_s);
                var kx: usize = 0;
                while (kx < d.kw) : (kx += 1) {
                    const ix_s = @as(isize, @intCast(ox * d.stride_w + kx)) - @as(isize, @intCast(d.pad_w));
                    if (ix_s < 0 or ix_s >= @as(isize, @intCast(d.w))) continue;
                    const ix: usize = @intCast(ix_s);
                    @memcpy(col[dst_pos + (ky * d.kw + kx) * d.cin ..][0..d.cin], in[(iy * d.w + ix) * d.cin ..][0..d.cin]);
                }
            }
        }
    }
}

/// Compute output rows `[oh_start, oh_end)` only (the per-worker range for the
/// threaded path). Each row is independent — no cross-row state.
fn conv2dRangeRows(out: []f32, in: []const f32, w: []const f32, bias: ?[]const f32, d: Conv2dDims, oh_start: usize, oh_end: usize) void {
    const cin_pg = d.cin / d.groups;
    const cout_pg = d.cout / d.groups;
    var oh: usize = oh_start;
    while (oh < oh_end) : (oh += 1) {
        var ow: usize = 0;
        while (ow < d.ow) : (ow += 1) {
            const out_base = (oh * d.ow + ow) * d.cout;
            var oc: usize = 0;
            while (oc < d.cout) : (oc += 1) {
                const g = oc / cout_pg;
                const ic0 = g * cin_pg;
                var acc: f32 = if (bias) |b| b[oc] else 0;
                var kh: usize = 0;
                while (kh < d.kh) : (kh += 1) {
                    const ih_s = @as(isize, @intCast(oh * d.stride_h + kh)) - @as(isize, @intCast(d.pad_h));
                    if (ih_s < 0 or ih_s >= @as(isize, @intCast(d.h))) continue;
                    const ih: usize = @intCast(ih_s);
                    var kw: usize = 0;
                    while (kw < d.kw) : (kw += 1) {
                        const iw_s = @as(isize, @intCast(ow * d.stride_w + kw)) - @as(isize, @intCast(d.pad_w));
                        if (iw_s < 0 or iw_s >= @as(isize, @intCast(d.w))) continue;
                        const iw: usize = @intCast(iw_s);
                        const in_base = (ih * d.w + iw) * d.cin + ic0;
                        const w_base = ((oc * d.kh + kh) * d.kw + kw) * cin_pg;
                        var ic: usize = 0;
                        while (ic < cin_pg) : (ic += 1) {
                            acc += in[in_base + ic] * w[w_base + ic];
                        }
                    }
                }
                out[out_base + oc] = acc;
            }
        }
    }
}

// ===========================================================================
// conv1d — general non-causal 1-D convolution (PyTorch Conv1d semantics:
// standard cross-correlation with symmetric zero padding, stride, dilation,
// and grouped channels). Used by the omnivoice codec port (HuBERT feature
// extractor, DAC encoder, SemanticEncoder). Allocation-free; the exec domain
// validates shapes and computes `out_len`.
// ===========================================================================

/// Geometry for `conv1dIntoWithConfig`. Row-major tensors:
///   input  `in[t*in_channels + i]`                    (t in [0, seq))
///   weight `w[(k*in_per_group + i)*out_channels + o]` (out-channel contiguous,
///                                                      same layout family as
///                                                      causalConv1d)
///   output `out[t*out_channels + o]`                  (t in [0, out_len))
/// with in_per_group = in_channels/groups; output channel `o` belongs to group
/// `g = o / (out_channels/groups)` and reads input channels
/// `[g*in_per_group, (g+1)*in_per_group)`. The input is virtually zero-padded
/// `pad` rows on BOTH sides (out-of-range rows are skipped, never
/// materialized); `out_len = (seq + 2*pad - dilation*(taps-1) - 1)/stride + 1`.
// ===========================================================================
// conv2d backward (VJPs of the channel-last direct conv2d). Gather kernels
// (no dilation in 2-D) with disjoint output writes: backward-input splits the
// input rows (d.h), backward-weight splits the output channels (d.cout), so
// the parallel splits are bit-identical to the serial paths. f32 accumulation
// to match the forward.
// ===========================================================================

const Conv2dGradTask = struct {
    out: []f32,
    a: []const f32, // gy (backward-input) / input (backward-weight)
    b: []const f32, // weight (backward-input) / gy (backward-weight)
    d: Conv2dDims,
    start: usize,
    end: usize,
};

fn runConv2dBackwardInputTask(task: *const Conv2dGradTask) void {
    conv2dBackwardInputRangeRows(task.out, task.a, task.b, task.d, task.start, task.end);
}

fn runConv2dBackwardWeightTask(task: *const Conv2dGradTask) void {
    conv2dBackwardWeightRangeCout(task.out, task.a, task.b, task.d, task.start, task.end);
}

fn maybeParallelConv2dGrad(
    config: ParallelConfig,
    comptime runTask: fn (*const Conv2dGradTask) void,
    split: usize,
    out: []f32,
    a: []const f32,
    b: []const f32,
    d: Conv2dDims,
) bool {
    const pool = config.pool orelse return false;
    const cin_pg = d.cin / d.groups;
    const work = std.math.mul(usize, parallel.saturatedMul3(d.oh * d.ow, d.cout, cin_pg), d.kh * d.kw) catch std.math.maxInt(usize);
    const thread_count = vm.generalConvThreadCount(split, work);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]Conv2dGradTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .out = out,
            .a = a,
            .b = b,
            .d = d,
            .start = task_i * split / thread_count,
            .end = (task_i + 1) * split / thread_count,
        };
    }
    pool.parallelChunks(Conv2dGradTask, tasks[0..thread_count], runTask);
    return true;
}

/// grad wrt input: gx[h,w,ci] = Σ over valid (oh,ow,kh,kw, co∈group(ci)) of
/// gy[oh,ow,co] · w[co,kh,kw, ci_local]. `out` is [H,W,Cin]. Parallel split
/// over input rows (d.h) — disjoint writes, bit-identical to serial.
pub fn conv2dBackwardInputIntoWithConfig(out: *Tensor, gy: *const Tensor, weight: *const Tensor, d: Conv2dDims, config: ParallelConfig) void {
    const gx = out.data();
    const gyd = gy.dataConst();
    const wt = weight.dataConst();
    if (maybeParallelConv2dGrad(config, runConv2dBackwardInputTask, d.h, gx, gyd, wt, d)) return;
    conv2dBackwardInputRangeRows(gx, gyd, wt, d, 0, d.h);
}

/// Config-less core, callable from both backends (their WithConfig wrappers
/// pass their own ParallelConfig type).
pub fn conv2dBackwardInputInto(out: *Tensor, gy: *const Tensor, weight: *const Tensor, d: Conv2dDims) void {
    conv2dBackwardInputRangeRows(out.data(), gy.dataConst(), weight.dataConst(), d, 0, d.h);
}

/// Compute input rows `[h_start, h_end)` only (the per-worker range for the
/// threaded path). Each row is independent — no cross-row state.
fn conv2dBackwardInputRangeRows(gx: []f32, gyd: []const f32, wt: []const f32, d: Conv2dDims, h_start: usize, h_end: usize) void {
    const cin_pg = d.cin / d.groups;
    const cout_pg = d.cout / d.groups;
    // Owned rows only; every element below is assigned anyway.
    @memset(gx[h_start * d.w * d.cin .. h_end * d.w * d.cin], 0);
    var h: usize = h_start;
    while (h < h_end) : (h += 1) {
        var w: usize = 0;
        while (w < d.w) : (w += 1) {
            var ci: usize = 0;
            while (ci < d.cin) : (ci += 1) {
                const group = ci / cin_pg;
                const ci_local = ci % cin_pg;
                var acc: f32 = 0;
                var kh: usize = 0;
                while (kh < d.kh) : (kh += 1) {
                    if (h + d.pad_h < kh) continue;
                    const nh = h + d.pad_h - kh;
                    if (nh % d.stride_h != 0) continue;
                    const oh = nh / d.stride_h;
                    if (oh >= d.oh) continue;
                    var kw: usize = 0;
                    while (kw < d.kw) : (kw += 1) {
                        if (w + d.pad_w < kw) continue;
                        const nw = w + d.pad_w - kw;
                        if (nw % d.stride_w != 0) continue;
                        const ow = nw / d.stride_w;
                        if (ow >= d.ow) continue;
                        const gy_base = (oh * d.ow + ow) * d.cout;
                        var co_local: usize = 0;
                        while (co_local < cout_pg) : (co_local += 1) {
                            const co = group * cout_pg + co_local;
                            acc += gyd[gy_base + co] * wt[((co * d.kh + kh) * d.kw + kw) * cin_pg + ci_local];
                        }
                    }
                }
                gx[(h * d.w + w) * d.cin + ci] = acc;
            }
        }
    }
}

/// grad wrt weight: gw[co,kh,kw,ci_local] = Σ over valid (oh,ow) of
/// gy[oh,ow,co] · in[oh·sh+kh−ph, ow·sw+kw−pw, group(co)·cin_pg+ci_local].
/// `out` is [Cout,KH,KW,Cin/groups]. Parallel split over output channels
/// (d.cout) — disjoint writes, bit-identical to serial.
pub fn conv2dBackwardWeightIntoWithConfig(out: *Tensor, input: *const Tensor, gy: *const Tensor, d: Conv2dDims, config: ParallelConfig) void {
    const gw = out.data();
    const ind = input.dataConst();
    const gyd = gy.dataConst();
    if (maybeParallelConv2dGrad(config, runConv2dBackwardWeightTask, d.cout, gw, ind, gyd, d)) return;
    conv2dBackwardWeightRangeCout(gw, ind, gyd, d, 0, d.cout);
}

/// Config-less core, callable from both backends (their WithConfig wrappers
/// pass their own ParallelConfig type).
pub fn conv2dBackwardWeightInto(out: *Tensor, input: *const Tensor, gy: *const Tensor, d: Conv2dDims) void {
    conv2dBackwardWeightRangeCout(out.data(), input.dataConst(), gy.dataConst(), d, 0, d.cout);
}

/// Compute output channels `[co_start, co_end)` only (the per-worker range for
/// the threaded path). Each channel's gw rows are independent.
fn conv2dBackwardWeightRangeCout(gw: []f32, ind: []const f32, gyd: []const f32, d: Conv2dDims, co_start: usize, co_end: usize) void {
    const cin_pg = d.cin / d.groups;
    const cout_pg = d.cout / d.groups;
    // Owned channels only; every element below is assigned anyway.
    @memset(gw[co_start * d.kh * d.kw * cin_pg .. co_end * d.kh * d.kw * cin_pg], 0);
    var co: usize = co_start;
    while (co < co_end) : (co += 1) {
        const group = co / cout_pg;
        var kh: usize = 0;
        while (kh < d.kh) : (kh += 1) {
            var kw: usize = 0;
            while (kw < d.kw) : (kw += 1) {
                var ci_local: usize = 0;
                while (ci_local < cin_pg) : (ci_local += 1) {
                    const ci = group * cin_pg + ci_local;
                    var acc: f32 = 0;
                    var oh: usize = 0;
                    while (oh < d.oh) : (oh += 1) {
                        const hh = oh * d.stride_h + kh;
                        if (hh < d.pad_h) continue;
                        const h = hh - d.pad_h;
                        if (h >= d.h) continue;
                        var ow: usize = 0;
                        while (ow < d.ow) : (ow += 1) {
                            const ww = ow * d.stride_w + kw;
                            if (ww < d.pad_w) continue;
                            const wpos = ww - d.pad_w;
                            if (wpos >= d.w) continue;
                            acc += gyd[(oh * d.ow + ow) * d.cout + co] * ind[(h * d.w + wpos) * d.cin + ci];
                        }
                    }
                    gw[((co * d.kh + kh) * d.kw + kw) * cin_pg + ci_local] = acc;
                }
            }
        }
    }
}

// ===========================================================================
// col2im — the adjoint of the 2-D im2col gather above (the second half of the
// backward-input GEMM decomposition: gcol = gy · w, gx = col2im(gcol)).
// Written as a GATHER over input rows (each input element sums the col
// entries whose forward tap read it; out-of-range taps never enter the
// enumeration, so the padding's gradient is dropped exactly as the adjoint
// requires), never a scatter — the parallel row split has disjoint writes and
// is bit-identical to serial.
// ===========================================================================

const Col2imTask = struct {
    out: []f32,
    col: []const f32,
    d: Conv2dDims,
    h_start: usize,
    h_end: usize,
};

fn runCol2imTask(task: *const Col2imTask) void {
    col2imRangeRows(task.out, task.col, task.d, task.h_start, task.h_end);
}

/// `out` is [H,W,Cin]; `col` is the im2col layout
/// `col[(oy·OW+ox)·ksz + (ky·KW+kx)·Cin + ic]` with `ksz = KH·KW·Cin`
/// (groups == 1, matching `im2colIntoWithConfig`). Parallel over input rows.
pub fn col2imIntoWithConfig(out: *Tensor, col: *const Tensor, d: Conv2dDims, config: ParallelConfig) void {
    const o = out.data();
    const cd = col.dataConst();
    if (config.pool) |pool| {
        // Each input element gathers at most (kh/stride_h + 1)*(kw/stride_w + 1)
        // col rows.
        const work = parallel.saturatedMul3(d.h * d.w, d.cin, (d.kh / d.stride_h + 1) * (d.kw / d.stride_w + 1));
        const tc = vm.generalConvThreadCount(d.h, work);
        if (tc > 1) {
            var tasks: [parallel.vector_max_threads]Col2imTask = undefined;
            for (0..tc) |ti| {
                tasks[ti] = .{
                    .out = o,
                    .col = cd,
                    .d = d,
                    .h_start = ti * d.h / tc,
                    .h_end = (ti + 1) * d.h / tc,
                };
            }
            pool.parallelChunks(Col2imTask, tasks[0..tc], runCol2imTask);
            return;
        }
    }
    col2imRangeRows(o, cd, d, 0, d.h);
}

fn col2imRangeRows(gx: []f32, col: []const f32, d: Conv2dDims, h_start: usize, h_end: usize) void {
    const ksz = d.kh * d.kw * d.cin;
    var h: usize = h_start;
    while (h < h_end) : (h += 1) {
        var w: usize = 0;
        while (w < d.w) : (w += 1) {
            const gx_row = gx[(h * d.w + w) * d.cin ..][0..d.cin];
            @memset(gx_row, 0);
            var kh: usize = 0;
            while (kh < d.kh) : (kh += 1) {
                if (h + d.pad_h < kh) continue;
                const nh = h + d.pad_h - kh;
                if (nh % d.stride_h != 0) continue;
                const oh = nh / d.stride_h;
                if (oh >= d.oh) continue;
                var kw: usize = 0;
                while (kw < d.kw) : (kw += 1) {
                    if (w + d.pad_w < kw) continue;
                    const nw = w + d.pad_w - kw;
                    if (nw % d.stride_w != 0) continue;
                    const ow = nw / d.stride_w;
                    if (ow >= d.ow) continue;
                    addRow(gx_row, col[(oh * d.ow + ow) * ksz + (kh * d.kw + kw) * d.cin ..][0..d.cin]);
                }
            }
        }
    }
}

pub const Conv1dDims = struct {
    seq: usize,
    out_len: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
};

pub fn conv1dIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    d: Conv1dDims,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, d.out_len * d.out_channels);
    const input_data = vm.contiguousDataConst(input, d.seq * d.in_channels);
    const in_per_group = d.in_channels / d.groups;
    const weight_data = vm.contiguousDataConst(weight, d.taps * in_per_group * d.out_channels);
    if (maybeParallelConv1d(config, output, input_data, weight_data, d)) return;
    conv1dForwardRange(output, input_data, weight_data, d, 0, d.out_len);
}

const Conv1dTask = struct {
    out: []f32,
    input: []const f32,
    weight: []const f32,
    d: Conv1dDims,
    start: usize,
    end: usize,
};

fn runConv1dTask(task: *const Conv1dTask) void {
    conv1dForwardRange(task.out, task.input, task.weight, task.d, task.start, task.end);
}

fn maybeParallelConv1d(
    config: ParallelConfig,
    out: []f32,
    input: []const f32,
    weight: []const f32,
    d: Conv1dDims,
) bool {
    const pool = config.pool orelse return false;
    const in_per_group = d.in_channels / d.groups;
    const work = std.math.mul(usize, parallel.saturatedMul3(d.out_len, in_per_group, d.out_channels), d.taps) catch std.math.maxInt(usize);
    const thread_count = vm.generalConvThreadCount(d.out_len, work);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]Conv1dTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .out = out,
            .input = input,
            .weight = weight,
            .d = d,
            .start = task_i * d.out_len / thread_count,
            .end = (task_i + 1) * d.out_len / thread_count,
        };
    }
    pool.parallelChunks(Conv1dTask, tasks[0..thread_count], runConv1dTask);
    return true;
}

/// Parallel split over OUTPUT rows `[t_start, t_end)` — disjoint writes, so
/// the threaded result is bit-identical to the serial path. Per output row:
/// zero the row (bias is not fused — it composes via addAxisVectorInPlace like
/// causalConv1d), then for each tap resolve the padded input row
/// `t*stride + k*dilation - pad` (skipped when outside `[0, seq)`) and axpy
/// its channels against the contiguous out-channel weight rows.
fn conv1dForwardRange(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    d: Conv1dDims,
    t_start: usize,
    t_end: usize,
) void {
    const in_per_group = d.in_channels / d.groups;
    const out_per_group = d.out_channels / d.groups;
    for (t_start..t_end) |t| {
        const out_row = out[t * d.out_channels ..][0..d.out_channels];
        @memset(out_row, 0);
        for (0..d.taps) |k| {
            const pos = t * d.stride + k * d.dilation; // position in the padded input
            if (pos < d.pad) continue;
            const src = pos - d.pad;
            if (src >= d.seq) continue;
            const x_row = input[src * d.in_channels ..][0..d.in_channels];
            if (d.groups == 1) {
                for (0..d.in_channels) |i| {
                    axpyRow(out_row, x_row[i], weight[(k * d.in_channels + i) * d.out_channels ..][0..d.out_channels]);
                }
                continue;
            }
            for (0..d.groups) |group| {
                const input_start = group * in_per_group;
                const out_start = group * out_per_group;
                const out_part = out_row[out_start..][0..out_per_group];
                for (0..in_per_group) |local_i| {
                    axpyRow(out_part, x_row[input_start + local_i], weight[(k * in_per_group + local_i) * d.out_channels + out_start ..][0..out_per_group]);
                }
            }
        }
    }
}

// ===========================================================================
// col2im1d — the ggml `col2im_1d` gather, the second half of the
// ConvTranspose1d = GEMM + col2im decomposition (omnivoice DAC decoder).
// Written as a GATHER over output rows (each output element sums its
// contributors; the crop by `pad` is folded in via `t_abs`), never a scatter,
// so the parallel row split has disjoint writes.
// ===========================================================================

/// `col` is `[t_in, taps*out_channels]` rows with column index `oc*taps + k`
/// (k varying fastest inside each oc block); `out` is `[out_len, out_channels]`
/// rows with the channel fast (the Fucina row convention — this differs from
/// ggml's channel-planar dst but the math per element is identical). Rows
/// `t_out >= t_conv`, where `t_conv = (t_in-1)*stride + taps - 2*pad`, are the
/// ConvTranspose `output_padding` and are zeroed.
pub fn col2im1dIntoWithConfig(
    out: *Tensor,
    col: *const Tensor,
    t_in: usize,
    out_len: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, out_len * out_channels);
    const col_data = vm.contiguousDataConst(col, t_in * taps * out_channels);
    if (maybeParallelCol2im1d(config, output, col_data, t_in, out_len, out_channels, taps, stride, pad)) return;
    col2im1dRange(output, col_data, t_in, out_channels, taps, stride, pad, 0, out_len);
}

const Col2im1dTask = struct {
    out: []f32,
    col: []const f32,
    t_in: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    start: usize,
    end: usize,
};

fn runCol2im1dTask(task: *const Col2im1dTask) void {
    col2im1dRange(task.out, task.col, task.t_in, task.out_channels, task.taps, task.stride, task.pad, task.start, task.end);
}

fn maybeParallelCol2im1d(
    config: ParallelConfig,
    out: []f32,
    col: []const f32,
    t_in: usize,
    out_len: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
) bool {
    const pool = config.pool orelse return false;
    // Each output element gathers at most taps/stride + 1 col entries.
    const work = parallel.saturatedMul3(out_len, out_channels, taps / stride + 1);
    const thread_count = vm.generalConvThreadCount(out_len, work);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]Col2im1dTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .out = out,
            .col = col,
            .t_in = t_in,
            .out_channels = out_channels,
            .taps = taps,
            .stride = stride,
            .pad = pad,
            .start = task_i * out_len / thread_count,
            .end = (task_i + 1) * out_len / thread_count,
        };
    }
    pool.parallelChunks(Col2im1dTask, tasks[0..thread_count], runCol2im1dTask);
    return true;
}

fn col2im1dRange(
    out: []f32,
    col: []const f32,
    t_in: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    t_start: usize,
    t_end: usize,
) void {
    const t_conv = (t_in - 1) * stride + taps - 2 * pad;
    const row_stride = taps * out_channels;
    for (t_start..t_end) |t_out| {
        const out_row = out[t_out * out_channels ..][0..out_channels];
        if (t_out >= t_conv) {
            @memset(out_row, 0); // ConvTranspose output_padding rows
            continue;
        }
        const t_abs = t_out + pad; // position in the uncropped signal
        // ceil((t_abs - taps + 1)/stride), clamped at 0 (the numerator can be
        // negative — branch before subtracting in usize).
        const t_in_min: usize = if (t_abs + 1 > taps) (t_abs + 1 - taps + stride - 1) / stride else 0;
        const t_in_max: usize = @min(t_in - 1, t_abs / stride);
        for (0..out_channels) |oc| {
            var acc: f32 = 0;
            var ti = t_in_min;
            while (ti <= t_in_max) : (ti += 1) {
                const k = t_abs - ti * stride;
                std.debug.assert(k < taps);
                acc += col[ti * row_stride + oc * taps + k];
            }
            out_row[oc] = acc;
        }
    }
}

// ===========================================================================
// conv1d backward — VJPs of the general non-causal conv1d above. Both are
// written with disjoint output writes so the parallel splits are bit-identical
// to the serial paths: backward-input splits INPUT time rows (each row gathers
// the gy positions that read it), backward-weight splits the taps*in_per_group
// weight rows.
// ===========================================================================

/// VJP of conv1dIntoWithConfig wrt the input. `out` is `[seq, in_channels]`,
/// `gy` is `[out_len, out_channels]`, `weight` the forward
/// `[tap, in_per_group, out_channels]`. Per input row `ti` and tap `k` the
/// contributing output row is `n/stride` with `n = ti + pad - k*dilation`,
/// valid when `n >= 0`, `n % stride == 0`, and `n/stride < out_len`; the
/// channel sum is a contiguous dot over the group's out-channel slice (same
/// memory trick as generalBackwardInputRange).
pub fn conv1dBackwardInputIntoWithConfig(
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    d: Conv1dDims,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, d.seq * d.in_channels);
    const gy_data = vm.contiguousDataConst(gy, d.out_len * d.out_channels);
    const in_per_group = d.in_channels / d.groups;
    const weight_data = vm.contiguousDataConst(weight, d.taps * in_per_group * d.out_channels);
    if (maybeParallelConv1dGrad(config, runConv1dBackwardInputTask, d.seq, output, gy_data, weight_data, d)) return;
    conv1dBackwardInputRange(output, gy_data, weight_data, d, 0, d.seq);
}

/// VJP of conv1dIntoWithConfig wrt the weight. `out` is
/// `[taps, in_per_group, out_channels]`, `input` and `gy` are the forward
/// operand and the upstream gradient. Splits the taps*in_per_group weight
/// rows; each row axpy-accumulates the valid `(t_out, src)` pairs of the
/// forward geometry (out-of-range padded input rows are skipped).
pub fn conv1dBackwardWeightIntoWithConfig(
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    d: Conv1dDims,
    config: ParallelConfig,
) void {
    const in_per_group = d.in_channels / d.groups;
    const output = vm.contiguousData(out, d.taps * in_per_group * d.out_channels);
    const input_data = vm.contiguousDataConst(input, d.seq * d.in_channels);
    const gy_data = vm.contiguousDataConst(gy, d.out_len * d.out_channels);
    const rows = d.taps * in_per_group;
    if (maybeParallelConv1dGrad(config, runConv1dBackwardWeightTask, rows, output, input_data, gy_data, d)) return;
    conv1dBackwardWeightRange(output, input_data, gy_data, d, 0, rows);
}

const Conv1dGradTask = struct {
    out: []f32,
    a: []const f32, // gy (backward-input) / input (backward-weight)
    b: []const f32, // weight (backward-input) / gy (backward-weight)
    d: Conv1dDims,
    start: usize,
    end: usize,
};

fn runConv1dBackwardInputTask(task: *const Conv1dGradTask) void {
    conv1dBackwardInputRange(task.out, task.a, task.b, task.d, task.start, task.end);
}

fn runConv1dBackwardWeightTask(task: *const Conv1dGradTask) void {
    conv1dBackwardWeightRange(task.out, task.a, task.b, task.d, task.start, task.end);
}

fn maybeParallelConv1dGrad(
    config: ParallelConfig,
    comptime runTask: fn (*const Conv1dGradTask) void,
    split: usize,
    out: []f32,
    a: []const f32,
    b: []const f32,
    d: Conv1dDims,
) bool {
    const pool = config.pool orelse return false;
    const in_per_group = d.in_channels / d.groups;
    const work = std.math.mul(usize, parallel.saturatedMul3(d.out_len, in_per_group, d.out_channels), d.taps) catch std.math.maxInt(usize);
    const thread_count = vm.generalConvThreadCount(split, work);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]Conv1dGradTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .out = out,
            .a = a,
            .b = b,
            .d = d,
            .start = task_i * split / thread_count,
            .end = (task_i + 1) * split / thread_count,
        };
    }
    pool.parallelChunks(Conv1dGradTask, tasks[0..thread_count], runTask);
    return true;
}

fn conv1dBackwardInputRange(
    out: []f32,
    gy: []const f32,
    weight: []const f32,
    d: Conv1dDims,
    ti_start: usize,
    ti_end: usize,
) void {
    const in_per_group = d.in_channels / d.groups;
    const out_per_group = d.out_channels / d.groups;
    for (ti_start..ti_end) |ti| {
        const gx_row = out[ti * d.in_channels ..][0..d.in_channels];
        @memset(gx_row, 0);
        for (0..d.taps) |k| {
            const shifted = k * d.dilation;
            if (shifted > ti + d.pad) continue; // n would be negative
            const n = ti + d.pad - shifted;
            if (n % d.stride != 0) continue;
            const t = n / d.stride;
            if (t >= d.out_len) continue;
            const gy_row = gy[t * d.out_channels ..][0..d.out_channels];
            if (d.groups == 1) {
                for (0..d.in_channels) |i| {
                    gx_row[i] += dotRow(gy_row, weight[(k * d.in_channels + i) * d.out_channels ..][0..d.out_channels]);
                }
                continue;
            }
            for (0..d.groups) |group| {
                const input_start = group * in_per_group;
                const out_start = group * out_per_group;
                const gy_part = gy_row[out_start..][0..out_per_group];
                for (0..in_per_group) |local_i| {
                    gx_row[input_start + local_i] += dotRow(gy_part, weight[(k * in_per_group + local_i) * d.out_channels + out_start ..][0..out_per_group]);
                }
            }
        }
    }
}

fn conv1dBackwardWeightRange(
    out: []f32,
    input: []const f32,
    gy: []const f32,
    d: Conv1dDims,
    row_start: usize,
    row_end: usize,
) void {
    const in_per_group = d.in_channels / d.groups;
    const out_per_group = d.out_channels / d.groups;
    for (row_start..row_end) |row| {
        const k = row / in_per_group;
        const local_i = row % in_per_group;
        const gw_row = out[row * d.out_channels ..][0..d.out_channels];
        @memset(gw_row, 0);
        for (0..d.out_len) |t| {
            const pos = t * d.stride + k * d.dilation; // position in the padded input
            if (pos < d.pad) continue;
            const src = pos - d.pad;
            if (src >= d.seq) continue;
            const x_row = input[src * d.in_channels ..][0..d.in_channels];
            if (d.groups == 1) {
                axpyRow(gw_row, x_row[local_i], gy[t * d.out_channels ..][0..d.out_channels]);
                continue;
            }
            const gy_row = gy[t * d.out_channels ..][0..d.out_channels];
            for (0..d.groups) |group| {
                const out_start = group * out_per_group;
                axpyRow(gw_row[out_start..][0..out_per_group], x_row[group * in_per_group + local_i], gy_row[out_start..][0..out_per_group]);
            }
        }
    }
}

// ===========================================================================
// col2im1d backward — the im2col-style GATHER that transposes the forward
// col2im gather. Each `(t_in, k)` cell reads exactly one gy row, so the
// parallel split over t_in rows has disjoint writes.
// ===========================================================================

/// VJP of col2im1dIntoWithConfig: `out` (gcol) is `[t_in, taps*out_channels]`
/// with column index `oc*taps + k` (the forward col layout);
/// `gcol[t_in, oc*taps + k] = gy[t_in*stride + k - pad, oc]` when that row
/// index lands in `[0, t_conv)` with `t_conv = (t_in-1)*stride + taps - 2*pad`,
/// else 0. `gy` has `gy_len >= t_conv` rows — the trailing `output_pad` rows
/// were forward-zeroed and never map back.
pub fn col2im1dBackwardIntoWithConfig(
    out: *Tensor,
    gy: *const Tensor,
    t_in: usize,
    gy_len: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    config: ParallelConfig,
) void {
    const output = vm.contiguousData(out, t_in * taps * out_channels);
    const gy_data = vm.contiguousDataConst(gy, gy_len * out_channels);
    const t_conv = (t_in - 1) * stride + taps - 2 * pad;
    std.debug.assert(gy_len >= t_conv);
    if (maybeParallelCol2im1dBackward(config, output, gy_data, t_in, t_conv, out_channels, taps, stride, pad)) return;
    col2im1dBackwardRange(output, gy_data, t_conv, out_channels, taps, stride, pad, 0, t_in);
}

const Col2im1dBackwardTask = struct {
    out: []f32,
    gy: []const f32,
    t_conv: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    start: usize,
    end: usize,
};

fn runCol2im1dBackwardTask(task: *const Col2im1dBackwardTask) void {
    col2im1dBackwardRange(task.out, task.gy, task.t_conv, task.out_channels, task.taps, task.stride, task.pad, task.start, task.end);
}

fn maybeParallelCol2im1dBackward(
    config: ParallelConfig,
    out: []f32,
    gy: []const f32,
    t_in: usize,
    t_conv: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
) bool {
    const pool = config.pool orelse return false;
    const work = parallel.saturatedMul3(t_in, out_channels, taps);
    const thread_count = vm.generalConvThreadCount(t_in, work);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]Col2im1dBackwardTask = undefined;
    for (0..thread_count) |task_i| {
        tasks[task_i] = .{
            .out = out,
            .gy = gy,
            .t_conv = t_conv,
            .out_channels = out_channels,
            .taps = taps,
            .stride = stride,
            .pad = pad,
            .start = task_i * t_in / thread_count,
            .end = (task_i + 1) * t_in / thread_count,
        };
    }
    pool.parallelChunks(Col2im1dBackwardTask, tasks[0..thread_count], runCol2im1dBackwardTask);
    return true;
}

fn col2im1dBackwardRange(
    out: []f32,
    gy: []const f32,
    t_conv: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
    ti_start: usize,
    ti_end: usize,
) void {
    const row_stride = taps * out_channels;
    for (ti_start..ti_end) |ti| {
        const col_row = out[ti * row_stride ..][0..row_stride];
        for (0..taps) |k| {
            const pos = ti * stride + k;
            if (pos < pad) {
                for (0..out_channels) |oc| col_row[oc * taps + k] = 0;
                continue;
            }
            const t_out = pos - pad;
            if (t_out >= t_conv) {
                for (0..out_channels) |oc| col_row[oc * taps + k] = 0;
                continue;
            }
            const gy_row = gy[t_out * out_channels ..][0..out_channels];
            for (0..out_channels) |oc| col_row[oc * taps + k] = gy_row[oc];
        }
    }
}

test {
    _ = @import("conv_tests.zig");
}
