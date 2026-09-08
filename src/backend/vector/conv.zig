//! Small-tap causal convolution kernels.
//!
//! Depthwise family: covers the causal FIR used by DeltaNet-style blocks
//! and Engram's ShortConv — input/output are contiguous `[time, channel]`,
//! kernel is `[channel, tap]` (tap `taps-1` = the newest sample), and
//! optional state is the `dilation * (taps - 1)` historical rows preceding
//! the input, oldest first.
//!
//! General family (`causalConv1d*`): channel-mixing dilated causal conv —
//! input `[time, in]`, weight `[tap, in, out]` (tap `taps-1` is the newest
//! sample), output `[time, out]`, optional state is the `dilation*(taps-1)`
//! historical input rows preceding the chunk, oldest first. The `[tap, in,
//! out]` weight layout keeps every kernel on contiguous out-channel rows:
//! forward and backward-weight are axpy accumulations, backward-input is a
//! contiguous dot.

const isa = @import("../isa.zig");
const parallel = @import("../../parallel.zig");
const std = @import("std");
const tensor = @import("../../tensor.zig");
const common = @import("common.zig");
const tile = @import("tile.zig");

const Tensor = tensor.Tensor;
const ParallelConfig = common.ParallelConfig;
const Vf32 = common.Vf32;
const vector_len = common.vector_len;

pub fn causalDepthwiseConv1dInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    kernel: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
) void {
    if (comptime isa.reference) return scalar.causalDepthwiseConv1dInto(out, input, kernel, state, seq, channels, taps, dilation);
    const output = common.contiguousData(out, seq * channels);
    const input_data = common.contiguousDataConst(input, seq * channels);
    const kernel_data = common.contiguousDataConst(kernel, channels * taps);
    if (maybeParallelConv(pc, runForwardChannels, output, input_data, kernel_data, null, state, seq, channels, taps, dilation)) return;
    forwardRange(output, input_data, kernel_data, state, seq, channels, taps, dilation, 0, channels);
}

pub fn causalDepthwiseConv1dBackwardInputInto(
    pc: ParallelConfig,
    out: *Tensor,
    gy: *const Tensor,
    kernel: *const Tensor,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
) void {
    if (comptime isa.reference) return scalar.causalDepthwiseConv1dBackwardInputInto(out, gy, kernel, seq, channels, taps, dilation);
    const output = common.contiguousData(out, seq * channels);
    const gy_data = common.contiguousDataConst(gy, seq * channels);
    const kernel_data = common.contiguousDataConst(kernel, channels * taps);
    if (maybeParallelConv(pc, runBackwardInputChannels, output, gy_data, kernel_data, null, null, seq, channels, taps, dilation)) return;
    backwardInputRange(output, gy_data, kernel_data, seq, channels, taps, dilation, 0, channels);
}

pub fn causalDepthwiseConv1dBackwardKernelInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
) void {
    if (comptime isa.reference) return scalar.causalDepthwiseConv1dBackwardKernelInto(out, input, gy, state, seq, channels, taps, dilation);
    const output = common.contiguousData(out, channels * taps);
    const input_data = common.contiguousDataConst(input, seq * channels);
    const gy_data = common.contiguousDataConst(gy, seq * channels);
    if (maybeParallelConv(pc, runBackwardKernelChannels, output, input_data, undefined, gy_data, state, seq, channels, taps, dilation)) return;
    backwardKernelRange(output, input_data, gy_data, state, seq, channels, taps, dilation, 0, channels);
}

const ConvCtx = struct {
    out: []f32,
    input: []const f32,
    kernel: []const f32,
    gy: []const f32,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
};

fn maybeParallelConv(
    pc: ParallelConfig,
    comptime runFn: fn (ConvCtx, usize, usize) void,
    out: []f32,
    input: []const f32,
    kernel: []const f32,
    gy: ?[]const f32,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
) bool {
    const pool = pc.pool orelse return false;
    const thread_count = common.depthwiseConvThreadCount(seq, channels, taps);
    if (thread_count == 1) return false;
    tile.forRange(pool, ConvCtx, .{
        .out = out,
        .input = input,
        .kernel = kernel,
        .gy = gy orelse &.{},
        .state = state,
        .seq = seq,
        .channels = channels,
        .taps = taps,
        .dilation = dilation,
    }, channels, thread_count, runFn);
    return true;
}

fn runForwardChannels(c: ConvCtx, channel_start: usize, channel_end: usize) void {
    forwardRange(c.out, c.input, c.kernel, c.state, c.seq, c.channels, c.taps, c.dilation, channel_start, channel_end);
}

fn runBackwardInputChannels(c: ConvCtx, channel_start: usize, channel_end: usize) void {
    backwardInputRange(c.out, c.input, c.kernel, c.seq, c.channels, c.taps, c.dilation, channel_start, channel_end);
}

fn runBackwardKernelChannels(c: ConvCtx, channel_start: usize, channel_end: usize) void {
    backwardKernelRange(c.out, c.input, c.gy, c.state, c.seq, c.channels, c.taps, c.dilation, channel_start, channel_end);
}

fn forwardRange(
    out: []f32,
    input: []const f32,
    kernel: []const f32,
    state: ?[]const f32,
    seq: usize,
    channels: usize,
    taps: usize,
    dilation: usize,
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = dilation * (taps - 1);
    for (0..seq) |t| {
        var c = channel_start;
        while (c + vector_len <= channel_end) : (c += vector_len) {
            var acc: Vf32 = @splat(0);
            for (0..taps) |k| {
                const u = t + k * dilation;
                const x: Vf32 = if (u >= pad)
                    input[(u - pad) * channels + c ..][0..vector_len].*
                else if (state) |s|
                    s[u * channels + c ..][0..vector_len].*
                else
                    @splat(0);
                acc += x * loadKernelVector(kernel, c, taps, k);
            }
            out[t * channels + c ..][0..vector_len].* = acc;
        }
        while (c < channel_end) : (c += 1) {
            var acc: f32 = 0;
            for (0..taps) |k| {
                acc += inputValue(input, state, channels, pad, dilation, t, c, k) * kernel[c * taps + k];
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
    dilation: usize,
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = dilation * (taps - 1);
    for (0..seq) |p| {
        var c = channel_start;
        while (c + vector_len <= channel_end) : (c += vector_len) {
            var acc: Vf32 = @splat(0);
            for (0..taps) |k| {
                const t_base = p + pad;
                if (k * dilation > t_base) continue;
                const t = t_base - k * dilation;
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
                if (k * dilation > t_base) continue;
                const t = t_base - k * dilation;
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
    dilation: usize,
    channel_start: usize,
    channel_end: usize,
) void {
    const pad = dilation * (taps - 1);
    for (channel_start..channel_end) |c| {
        for (0..taps) |k| {
            var acc: f32 = 0;
            for (0..seq) |t| {
                acc += gy[t * channels + c] * inputValue(input, state, channels, pad, dilation, t, c, k);
            }
            out[c * taps + k] = acc;
        }
    }
}

pub fn causalConv1dInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    state: ?[]const f32,
    bias: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
) void {
    if (comptime isa.reference) return scalar.causalConv1dInto(out, input, weight, state, bias, seq, in_channels, out_channels, taps, dilation);
    const output = common.contiguousData(out, seq * out_channels);
    const input_data = common.contiguousDataConst(input, seq * in_channels);
    const weight_data = common.contiguousDataConst(weight, taps * in_channels * out_channels);
    if (maybeParallelGeneralConv(pc, runGeneralForwardRows, seq, output, input_data, weight_data, null, state, bias, seq, in_channels, out_channels, taps, dilation, 1)) return;
    generalForwardRange(output, input_data, weight_data, state, bias, in_channels, out_channels, taps, dilation, 1, 0, seq);
}

pub fn causalConv1dBackwardInputInto(
    pc: ParallelConfig,
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
) void {
    if (comptime isa.reference) return scalar.causalConv1dBackwardInputInto(out, gy, weight, seq, in_channels, out_channels, taps, dilation);
    const output = common.contiguousData(out, seq * in_channels);
    const gy_data = common.contiguousDataConst(gy, seq * out_channels);
    const weight_data = common.contiguousDataConst(weight, taps * in_channels * out_channels);
    if (maybeParallelGeneralConv(pc, runGeneralBackwardInputRows, seq, output, &.{}, weight_data, gy_data, null, null, seq, in_channels, out_channels, taps, dilation, 1)) return;
    generalBackwardInputRange(output, gy_data, weight_data, seq, in_channels, out_channels, taps, dilation, 1, 0, seq);
}

pub fn causalConv1dBackwardWeightInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    state: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
) void {
    if (comptime isa.reference) return scalar.causalConv1dBackwardWeightInto(out, input, gy, state, seq, in_channels, out_channels, taps, dilation);
    const output = common.contiguousData(out, taps * in_channels * out_channels);
    const input_data = common.contiguousDataConst(input, seq * in_channels);
    const gy_data = common.contiguousDataConst(gy, seq * out_channels);
    const rows = taps * in_channels;
    if (maybeParallelGeneralConv(pc, runGeneralBackwardWeightRows, rows, output, input_data, &.{}, gy_data, state, null, seq, in_channels, out_channels, taps, dilation, 1)) return;
    generalBackwardWeightRange(output, input_data, gy_data, state, seq, in_channels, out_channels, taps, dilation, 1, 0, rows);
}

pub fn groupedCausalConv1dInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    state: ?[]const f32,
    bias: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
) void {
    if (comptime isa.reference) return scalar.groupedCausalConv1dInto(out, input, weight, state, bias, seq, in_channels, out_channels, taps, dilation, groups);
    const output = common.contiguousData(out, seq * out_channels);
    const input_data = common.contiguousDataConst(input, seq * in_channels);
    const in_per_group = in_channels / groups;
    const weight_data = common.contiguousDataConst(weight, taps * in_per_group * out_channels);
    if (maybeParallelGeneralConv(pc, runGeneralForwardRows, seq, output, input_data, weight_data, null, state, bias, seq, in_channels, out_channels, taps, dilation, groups)) return;
    generalForwardRange(output, input_data, weight_data, state, bias, in_channels, out_channels, taps, dilation, groups, 0, seq);
}

pub fn groupedCausalConv1dBackwardInputInto(
    pc: ParallelConfig,
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
) void {
    if (comptime isa.reference) return scalar.groupedCausalConv1dBackwardInputInto(out, gy, weight, seq, in_channels, out_channels, taps, dilation, groups);
    const output = common.contiguousData(out, seq * in_channels);
    const gy_data = common.contiguousDataConst(gy, seq * out_channels);
    const in_per_group = in_channels / groups;
    const weight_data = common.contiguousDataConst(weight, taps * in_per_group * out_channels);
    if (maybeParallelGeneralConv(pc, runGeneralBackwardInputRows, seq, output, &.{}, weight_data, gy_data, null, null, seq, in_channels, out_channels, taps, dilation, groups)) return;
    generalBackwardInputRange(output, gy_data, weight_data, seq, in_channels, out_channels, taps, dilation, groups, 0, seq);
}

pub fn groupedCausalConv1dBackwardWeightInto(
    pc: ParallelConfig,
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
) void {
    if (comptime isa.reference) return scalar.groupedCausalConv1dBackwardWeightInto(out, input, gy, state, seq, in_channels, out_channels, taps, dilation, groups);
    const in_per_group = in_channels / groups;
    const output = common.contiguousData(out, taps * in_per_group * out_channels);
    const input_data = common.contiguousDataConst(input, seq * in_channels);
    const gy_data = common.contiguousDataConst(gy, seq * out_channels);
    const rows = taps * in_per_group;
    if (maybeParallelGeneralConv(pc, runGeneralBackwardWeightRows, rows, output, input_data, &.{}, gy_data, state, null, seq, in_channels, out_channels, taps, dilation, groups)) return;
    generalBackwardWeightRange(output, input_data, gy_data, state, seq, in_channels, out_channels, taps, dilation, groups, 0, rows);
}

const GeneralConvCtx = struct {
    out: []f32,
    input: []const f32,
    weight: []const f32,
    gy: []const f32,
    state: ?[]const f32,
    bias: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
};

fn maybeParallelGeneralConv(
    pc: ParallelConfig,
    comptime runFn: fn (GeneralConvCtx, usize, usize) void,
    split: usize,
    out: []f32,
    input: []const f32,
    weight: []const f32,
    gy: ?[]const f32,
    state: ?[]const f32,
    bias: ?[]const f32,
    seq: usize,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
) bool {
    const pool = pc.pool orelse return false;
    const work = std.math.mul(usize, parallel.saturatedMul3(seq, in_channels, out_channels), taps) catch std.math.maxInt(usize);
    const thread_count = common.generalConvThreadCount(split, work);
    if (thread_count == 1) return false;
    tile.forRange(pool, GeneralConvCtx, .{
        .out = out,
        .input = input,
        .weight = weight,
        .gy = gy orelse &.{},
        .state = state,
        .bias = bias,
        .seq = seq,
        .in_channels = in_channels,
        .out_channels = out_channels,
        .taps = taps,
        .dilation = dilation,
        .groups = groups,
    }, split, thread_count, runFn);
    return true;
}

fn runGeneralForwardRows(c: GeneralConvCtx, start: usize, end: usize) void {
    generalForwardRange(c.out, c.input, c.weight, c.state, c.bias, c.in_channels, c.out_channels, c.taps, c.dilation, c.groups, start, end);
}

fn runGeneralBackwardInputRows(c: GeneralConvCtx, start: usize, end: usize) void {
    generalBackwardInputRange(c.out, c.gy, c.weight, c.seq, c.in_channels, c.out_channels, c.taps, c.dilation, c.groups, start, end);
}

fn runGeneralBackwardWeightRows(c: GeneralConvCtx, start: usize, end: usize) void {
    generalBackwardWeightRange(c.out, c.input, c.gy, c.state, c.seq, c.in_channels, c.out_channels, c.taps, c.dilation, c.groups, start, end);
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

/// Frames per register tile of the general causal conv forward: every
/// weight vector load feeds `time_tile` fused multiply-adds with the
/// accumulators in registers. Without the tile the walk is one FMA per
/// weight load with the accumulator round-tripping through memory, which
/// measures 10x slower at the 16-channel shapes of the NAM WaveNet.
const time_tile = 8;
/// Taps up to this many have their tile's input rows resolved once per tile
/// (8 KB of slices on the stack); longer kernels resolve per output block.
const max_cached_taps = 64;

/// The `time_tile` input rows feeding tap `k` of the tile at `t`, every one
/// of which resolves (the tile path's contract).
inline fn resolveTileRows(
    dst: *[time_tile][]const f32,
    input: []const f32,
    state: ?[]const f32,
    in_channels: usize,
    pad: usize,
    t: usize,
    k: usize,
    dilation: usize,
) *const [time_tile][]const f32 {
    inline for (0..time_tile) |tt| dst[tt] = generalConvInputRow(input, state, in_channels, pad, t + tt, k, dilation) orelse unreachable;
    return dst;
}
/// The (tap, in-channel) walk is blocked so the weight slab one output
/// vector visits per block stays in L1 across the output blocks and the
/// slab as a whole stays in L2 across the frame tiles: a block is at most
/// `max_block_pairs` (tap, in) pairs, one cache line of out channels each.
/// Partial sums round-trip through the output rows between blocks, which
/// is exact, so the per-element order stays tap-major, in-channel-inner
/// from zero, then bias, whatever the blocking. Without it the DAC entry
/// conv (1024 -> 1536, 7 taps, 44 MB of weights) re-streamed its weights
/// from DRAM once per output vector, 1.7x slower than a plain per-frame walk.
const max_block_pairs = 384;

/// General causal conv forward over output rows `[t_start, t_end)`.
/// Per output element the accumulation is `k`-major, in-channel-inner
/// fused multiply-add from zero, then the optional bias, the same in the
/// tile body, the frame tail and the channel tail, so the result is
/// bitwise independent of the tiling, the blocking and any row split. Rows
/// whose oldest tap precedes the chunk read zeros when no state is given;
/// those frames take the per-frame path, every later frame resolves all
/// its rows and takes the tile path.
fn generalForwardRange(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    state: ?[]const f32,
    bias: ?[]const f32,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    t_start: usize,
    t_end: usize,
) void {
    if (groups == 1 and in_channels == 1 and out_channels == 1 and dilation == 1 and taps > 1) {
        firForwardRange(out, input, weight, state, bias, taps, t_start, t_end);
        return;
    }
    const pad = dilation * (taps - 1);
    const tile_start = if (state == null) @min(@max(t_start, pad), t_end) else t_start;
    if (tile_start > t_start) forwardFramesRange(out, input, weight, state, bias, in_channels, out_channels, taps, dilation, groups, t_start, tile_start);
    forwardTilesRange(out, input, weight, state, bias, in_channels, out_channels, taps, dilation, groups, tile_start, t_end);
}

/// One (tap, in-channel) block of the walk: pairs `[p0, p1)` of the flat
/// tap-major sequence, as the tap range they span.
const PairBlock = struct {
    p0: usize,
    p1: usize,
    in_per_group: usize,

    fn first(self: PairBlock) bool {
        return self.p0 == 0;
    }

    fn k0(self: PairBlock) usize {
        return self.p0 / self.in_per_group;
    }

    fn k1(self: PairBlock) usize {
        return (self.p1 - 1) / self.in_per_group + 1;
    }

    /// The in-channel range of tap `k` inside the block.
    fn range(self: PairBlock, k: usize) [2]usize {
        const base = k * self.in_per_group;
        const lo = if (self.p0 > base) self.p0 - base else 0;
        const hi = @min(self.in_per_group, self.p1 - base);
        return .{ lo, hi };
    }
};

/// The tile body: `time_tile` frames per weight load, blocked over the
/// (tap, in-channel) walk. Every row resolves (the caller's contract); the
/// remainder frames take the per-frame path.
fn forwardTilesRange(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    state: ?[]const f32,
    bias: ?[]const f32,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    t_start: usize,
    t_end: usize,
) void {
    const in_per_group = in_channels / groups;
    const pairs = taps * in_per_group;
    if (pairs <= max_block_pairs) {
        // One block, the common case: no partial-sum round trip anywhere.
        const block = PairBlock{ .p0 = 0, .p1 = pairs, .in_per_group = in_per_group };
        forwardTilesBlock(true, true, out, input, weight, state, bias, in_channels, out_channels, taps, dilation, groups, block, t_start, t_end);
        return;
    }
    var p0: usize = 0;
    while (p0 < pairs) : (p0 += max_block_pairs) {
        const block = PairBlock{ .p0 = p0, .p1 = @min(pairs, p0 + max_block_pairs), .in_per_group = in_per_group };
        const last = block.p1 == pairs;
        if (block.first()) {
            forwardTilesBlock(true, false, out, input, weight, state, bias, in_channels, out_channels, taps, dilation, groups, block, t_start, t_end);
        } else if (last) {
            forwardTilesBlock(false, true, out, input, weight, state, bias, in_channels, out_channels, taps, dilation, groups, block, t_start, t_end);
        } else {
            forwardTilesBlock(false, false, out, input, weight, state, bias, in_channels, out_channels, taps, dilation, groups, block, t_start, t_end);
        }
    }
}

/// The tile body of one (tap, in-channel) block: `first` starts the
/// accumulators from zero (else from the output rows), `last` adds the
/// bias before the store.
fn forwardTilesBlock(
    comptime first: bool,
    comptime last: bool,
    out: []f32,
    input: []const f32,
    weight: []const f32,
    state: ?[]const f32,
    bias: ?[]const f32,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    block: PairBlock,
    t_start: usize,
    t_end: usize,
) void {
    const pad = dilation * (taps - 1);
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    const k0 = block.k0();
    const k1 = block.k1();
    var t = t_start;
    while (t + time_tile <= t_end) : (t += time_tile) {
        var cached: [max_cached_taps][time_tile][]const f32 = undefined;
        const cache_rows = taps <= max_cached_taps;
        if (cache_rows) {
            for (k0..k1) |k| _ = resolveTileRows(&cached[k], input, state, in_channels, pad, t, k, dilation);
        }
        for (0..groups) |g| {
            const in_start = g * in_per_group;
            const out_start = g * out_per_group;
            var o: usize = 0;
            while (o + vector_len <= out_per_group) : (o += vector_len) {
                var acc: [time_tile]Vf32 = undefined;
                inline for (0..time_tile) |tt| acc[tt] = if (first) @splat(0) else out[(t + tt) * out_channels + out_start + o ..][0..vector_len].*;
                for (k0..k1) |k| {
                    var local: [time_tile][]const f32 = undefined;
                    const rows = if (cache_rows) &cached[k] else resolveTileRows(&local, input, state, in_channels, pad, t, k, dilation);
                    const r = block.range(k);
                    for (r[0]..r[1]) |local_i| {
                        const wv: Vf32 = weight[(k * in_per_group + local_i) * out_channels + out_start + o ..][0..vector_len].*;
                        inline for (0..time_tile) |tt| acc[tt] = @mulAdd(Vf32, @splat(rows[tt][in_start + local_i]), wv, acc[tt]);
                    }
                }
                if (last) {
                    if (bias) |b| {
                        const bv: Vf32 = b[out_start + o ..][0..vector_len].*;
                        inline for (0..time_tile) |tt| acc[tt] += bv;
                    }
                }
                inline for (0..time_tile) |tt| out[(t + tt) * out_channels + out_start + o ..][0..vector_len].* = acc[tt];
            }
            while (o < out_per_group) : (o += 1) {
                var acc: [time_tile]f32 = undefined;
                inline for (0..time_tile) |tt| acc[tt] = if (first) 0 else out[(t + tt) * out_channels + out_start + o];
                for (k0..k1) |k| {
                    var local: [time_tile][]const f32 = undefined;
                    const rows = if (cache_rows) &cached[k] else resolveTileRows(&local, input, state, in_channels, pad, t, k, dilation);
                    const r = block.range(k);
                    for (r[0]..r[1]) |local_i| {
                        const w = weight[(k * in_per_group + local_i) * out_channels + out_start + o];
                        inline for (0..time_tile) |tt| acc[tt] = @mulAdd(f32, rows[tt][in_start + local_i], w, acc[tt]);
                    }
                }
                if (last) {
                    if (bias) |b| {
                        inline for (0..time_tile) |tt| acc[tt] += b[out_start + o];
                    }
                }
                inline for (0..time_tile) |tt| out[(t + tt) * out_channels + out_start + o] = acc[tt];
            }
        }
    }
    if (t < t_end) forwardFramesBlock(first, last, out, input, weight, state, bias, in_channels, out_channels, taps, dilation, groups, block, t, t_end);
}

/// One frame at a time with the accumulators in registers, over the whole
/// (tap, in-channel) walk in blocks.
fn forwardFramesRange(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    state: ?[]const f32,
    bias: ?[]const f32,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    t_start: usize,
    t_end: usize,
) void {
    const in_per_group = in_channels / groups;
    const pairs = taps * in_per_group;
    if (pairs <= max_block_pairs) {
        const block = PairBlock{ .p0 = 0, .p1 = pairs, .in_per_group = in_per_group };
        forwardFramesBlock(true, true, out, input, weight, state, bias, in_channels, out_channels, taps, dilation, groups, block, t_start, t_end);
        return;
    }
    var p0: usize = 0;
    while (p0 < pairs) : (p0 += max_block_pairs) {
        const block = PairBlock{ .p0 = p0, .p1 = @min(pairs, p0 + max_block_pairs), .in_per_group = in_per_group };
        const last = block.p1 == pairs;
        if (block.first()) {
            forwardFramesBlock(true, false, out, input, weight, state, bias, in_channels, out_channels, taps, dilation, groups, block, t_start, t_end);
        } else if (last) {
            forwardFramesBlock(false, true, out, input, weight, state, bias, in_channels, out_channels, taps, dilation, groups, block, t_start, t_end);
        } else {
            forwardFramesBlock(false, false, out, input, weight, state, bias, in_channels, out_channels, taps, dilation, groups, block, t_start, t_end);
        }
    }
}

/// The per-frame body of one block: a missing row (before the chunk, no
/// state) contributes nothing, which is what the zero row would.
fn forwardFramesBlock(
    comptime first: bool,
    comptime last: bool,
    out: []f32,
    input: []const f32,
    weight: []const f32,
    state: ?[]const f32,
    bias: ?[]const f32,
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    dilation: usize,
    groups: usize,
    block: PairBlock,
    t_start: usize,
    t_end: usize,
) void {
    const pad = dilation * (taps - 1);
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    const k0 = block.k0();
    const k1 = block.k1();
    for (t_start..t_end) |t| {
        for (0..groups) |g| {
            const in_start = g * in_per_group;
            const out_start = g * out_per_group;
            const out_row = out[t * out_channels + out_start ..][0..out_per_group];
            var o: usize = 0;
            while (o + vector_len <= out_per_group) : (o += vector_len) {
                var acc: Vf32 = if (first) @splat(0) else out_row[o..][0..vector_len].*;
                for (k0..k1) |k| {
                    const row = generalConvInputRow(input, state, in_channels, pad, t, k, dilation) orelse continue;
                    const r = block.range(k);
                    for (r[0]..r[1]) |local_i| {
                        const wv: Vf32 = weight[(k * in_per_group + local_i) * out_channels + out_start + o ..][0..vector_len].*;
                        acc = @mulAdd(Vf32, @splat(row[in_start + local_i]), wv, acc);
                    }
                }
                if (last) {
                    if (bias) |b| acc += @as(Vf32, b[out_start + o ..][0..vector_len].*);
                }
                out_row[o..][0..vector_len].* = acc;
            }
            while (o < out_per_group) : (o += 1) {
                var acc: f32 = if (first) 0 else out_row[o];
                for (k0..k1) |k| {
                    const row = generalConvInputRow(input, state, in_channels, pad, t, k, dilation) orelse continue;
                    const r = block.range(k);
                    for (r[0]..r[1]) |local_i| {
                        acc = @mulAdd(f32, row[in_start + local_i], weight[(k * in_per_group + local_i) * out_channels + out_start + o], acc);
                    }
                }
                if (last) {
                    if (bias) |b| acc += b[out_start + o];
                }
                out_row[o] = acc;
            }
        }
    }
}

/// Single-channel undilated FIR (`in = out = 1`, the cab-IR and linear
/// model shapes, thousands of taps): each output is one contiguous dot of
/// the taps against the signal window, vectorized along the taps. A window
/// that starts inside the state is two dots (state part, chunk part).
fn firForwardRange(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    state: ?[]const f32,
    bias: ?[]const f32,
    taps: usize,
    t_start: usize,
    t_end: usize,
) void {
    const pad = taps - 1;
    const b: f32 = if (bias) |values| values[0] else 0;
    for (t_start..t_end) |t| {
        var acc: f32 = undefined;
        if (t >= pad) {
            acc = firDot(weight, input[t - pad ..][0..taps]);
        } else {
            const from_state = pad - t;
            acc = if (state) |s| firDot(weight[0..from_state], s[t..][0..from_state]) else 0;
            acc += firDot(weight[from_state..], input[0 .. t + 1]);
        }
        out[t] = if (bias != null) acc + b else acc;
    }
}

/// Contiguous dot with fused multiply-adds along the vector body and a
/// scalar tail (the NAM cab's summation order).
inline fn firDot(w: []const f32, x: []const f32) f32 {
    var accv: Vf32 = @splat(0);
    var k: usize = 0;
    while (k + vector_len <= w.len) : (k += vector_len) {
        accv = @mulAdd(Vf32, w[k..][0..vector_len].*, x[k..][0..vector_len].*, accv);
    }
    var acc: f32 = @reduce(.Add, accv);
    while (k < w.len) : (k += 1) acc += w[k] * x[k];
    return acc;
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

inline fn axpyRow(acc: []f32, scalar_value: f32, row: []const f32) void {
    const sv: Vf32 = @splat(scalar_value);
    var o: usize = 0;
    while (o + vector_len <= acc.len) : (o += vector_len) {
        const cur: Vf32 = acc[o..][0..vector_len].*;
        const rv: Vf32 = row[o..][0..vector_len].*;
        acc[o..][0..vector_len].* = cur + sv * rv;
    }
    while (o < acc.len) : (o += 1) acc[o] += scalar_value * row[o];
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

inline fn scaleRow(out: []f32, scalar_value: f32, row: []const f32) void {
    const sv: Vf32 = @splat(scalar_value);
    var o: usize = 0;
    while (o + vector_len <= out.len) : (o += vector_len) {
        const rv: Vf32 = row[o..][0..vector_len].*;
        out[o..][0..vector_len].* = sv * rv;
    }
    while (o < out.len) : (o += 1) out[o] = scalar_value * row[o];
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
    dilation: usize,
    t: usize,
    c: usize,
    k: usize,
) f32 {
    const u = t + k * dilation;
    if (u >= pad) return input[(u - pad) * channels + c];
    const s = state orelse return 0;
    return s[u * channels + c];
}

// ===========================================================================
// conv2d — rank-3 channel-last [H,W,Cin] -> [OH,OW,Cout], with stride,
// explicit zero padding, and grouped/depthwise support. Used by the Parakeet
// FastConformer subsampling stem. Allocation-free; caller validates shapes.
// ===========================================================================

/// Geometry for `conv2dInto`. Channel-last tensors:
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

const Conv2dCtx = struct {
    out: []f32,
    in: []const f32,
    w: []const f32,
    bias: ?[]const f32,
    d: Conv2dDims,
};

fn runConv2dRows(c: Conv2dCtx, oh_start: usize, oh_end: usize) void {
    conv2dRangeRows(c.out, c.in, c.w, c.bias, c.d, oh_start, oh_end);
}

pub fn conv2dInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    bias: ?[]const f32,
    d: Conv2dDims,
) void {
    if (comptime isa.reference) return scalar.conv2dInto(out, input, weight, bias, d);
    const o = out.data();
    const in = input.dataConst();
    const wt = weight.dataConst();
    // Parallelize over output rows (oh) — each oh writes a disjoint output
    // range and its accumulation is independent, so the result is bit-identical to
    // the serial path (pure parallelization, no numeric change). The general conv
    // stem (subsampling) was the #1 cost at ~48% — single-threaded before.
    if (pc.pool) |pool| {
        const cin_pg = d.cin / d.groups;
        const work = d.oh * d.ow * d.cout * d.kh * d.kw * cin_pg;
        const tc = common.generalConvThreadCount(d.oh, work);
        if (tc > 1) {
            tile.forRange(pool, Conv2dCtx, .{ .out = o, .in = in, .w = wt, .bias = bias, .d = d }, d.oh, tc, runConv2dRows);
            return;
        }
    }
    conv2dRangeRows(o, in, wt, bias, d, 0, d.oh);
}

const DepthwiseCtx = struct {
    out: []f32,
    in: []const f32,
    taps: []const f32,
    bias: ?[]const f32,
    d: Conv2dDims,
};

fn runDepthwiseRows(c: DepthwiseCtx, oh_start: usize, oh_end: usize) void {
    conv2dDepthwiseRangeRows(c.out, c.in, c.taps, c.bias, c.d, oh_start, oh_end);
}

/// Depthwise conv2d (`groups == cin == cout`, one input channel per group)
/// over a TAP-MAJOR weight `taps[(ky·KW + kx)·C + c]` (the exec domain
/// repacks the `[C, KH, KW, 1]` weight once per call). Vectorized across
/// the contiguous channel axis: every output position starts from its bias
/// and accumulates its taps in the same `(ky, kx)` order as
/// `conv2dRangeRows`, per channel lane, multiply then add (no fused
/// multiply-add), so the result is bit-identical to the direct kernel.
/// The tap bounds are evaluated once per position instead of once per
/// output channel. Parallel over output rows like `conv2dInto`.
pub fn conv2dDepthwiseInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    taps: []const f32,
    bias: ?[]const f32,
    d: Conv2dDims,
) void {
    if (comptime isa.reference) return scalar.conv2dDepthwiseInto(out, input, taps, bias, d);
    const o = out.data();
    const in = input.dataConst();
    if (pc.pool) |pool| {
        const work = d.oh * d.ow * d.cout * d.kh * d.kw;
        const tc = common.generalConvThreadCount(d.oh, work);
        if (tc > 1) {
            tile.forRange(pool, DepthwiseCtx, .{ .out = o, .in = in, .taps = taps, .bias = bias, .d = d }, d.oh, tc, runDepthwiseRows);
            return;
        }
    }
    conv2dDepthwiseRangeRows(o, in, taps, bias, d, 0, d.oh);
}

fn conv2dDepthwiseRangeRows(out: []f32, in: []const f32, taps: []const f32, bias: ?[]const f32, d: Conv2dDims, oh_start: usize, oh_end: usize) void {
    const c = d.cin;
    var oh: usize = oh_start;
    while (oh < oh_end) : (oh += 1) {
        var ow: usize = 0;
        while (ow < d.ow) : (ow += 1) {
            const o = out[(oh * d.ow + ow) * c ..][0..c];
            if (bias) |b| @memcpy(o, b[0..c]) else @memset(o, 0);
            var ky: usize = 0;
            while (ky < d.kh) : (ky += 1) {
                const ih_s = @as(isize, @intCast(oh * d.stride_h + ky)) - @as(isize, @intCast(d.pad_h));
                if (ih_s < 0 or ih_s >= @as(isize, @intCast(d.h))) continue;
                const ih: usize = @intCast(ih_s);
                var kx: usize = 0;
                while (kx < d.kw) : (kx += 1) {
                    const iw_s = @as(isize, @intCast(ow * d.stride_w + kx)) - @as(isize, @intCast(d.pad_w));
                    if (iw_s < 0 or iw_s >= @as(isize, @intCast(d.w))) continue;
                    const iw: usize = @intCast(iw_s);
                    const in_row = in[(ih * d.w + iw) * c ..][0..c];
                    const w_tap = taps[(ky * d.kw + kx) * c ..][0..c];
                    var ci: usize = 0;
                    while (ci + vector_len <= c) : (ci += vector_len) {
                        const iv: Vf32 = in_row[ci..][0..vector_len].*;
                        const wv: Vf32 = w_tap[ci..][0..vector_len].*;
                        var acc: Vf32 = o[ci..][0..vector_len].*;
                        acc += iv * wv;
                        o[ci..][0..vector_len].* = acc;
                    }
                    while (ci < c) : (ci += 1) o[ci] += in_row[ci] * w_tap[ci];
                }
            }
        }
    }
}

const Im2colCtx = struct {
    col: []f32,
    in: []const f32,
    d: Conv2dDims,
};

fn runIm2colRows(c: Im2colCtx, oh_start: usize, oh_end: usize) void {
    im2colRangeRows(c.col, c.in, c.d, oh_start, oh_end);
}

/// im2col gather for the groups==1 conv2d GEMM route:
/// `col[(oy·OW+ox)·ksz + (ky·KW+kx)·Cin + ic]` = the padded input tap, with
/// `ksz = KH·KW·Cin` and out-of-range taps left zero. Pure data movement
/// (zero-fill + `Cin`-wide `@memcpy`), parallel over output rows — each `oy`
/// writes a disjoint `col` range, so the result is bit-identical to serial.
pub fn im2colInto(pc: ParallelConfig, col: *Tensor, input: *const Tensor, d: Conv2dDims) void {
    const cd = col.data();
    const in = input.dataConst();
    if (common.refSerial(pc).pool) |pool| {
        const work = d.oh * d.ow * d.kh * d.kw * d.cin;
        const tc = common.generalConvThreadCount(d.oh, work);
        if (tc > 1) {
            tile.forRange(pool, Im2colCtx, .{ .col = cd, .in = in, .d = d }, d.oh, tc, runIm2colRows);
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
/// Direct conv2d over an output-row range (correctness-first; f32 accumulation
/// to match ggml's f32 conv path).
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

/// Geometry for `conv1dInto`. Row-major tensors:
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

const Conv2dGradCtx = struct {
    out: []f32,
    a: []const f32, // gy (backward-input) / input (backward-weight)
    b: []const f32, // weight (backward-input) / gy (backward-weight)
    d: Conv2dDims,
};

fn runConv2dBackwardInputRows(c: Conv2dGradCtx, start: usize, end: usize) void {
    conv2dBackwardInputRangeRows(c.out, c.a, c.b, c.d, start, end);
}

fn runConv2dBackwardWeightCout(c: Conv2dGradCtx, start: usize, end: usize) void {
    conv2dBackwardWeightRangeCout(c.out, c.a, c.b, c.d, start, end);
}

fn maybeParallelConv2dGrad(
    pc: ParallelConfig,
    comptime runFn: fn (Conv2dGradCtx, usize, usize) void,
    split: usize,
    out: []f32,
    a: []const f32,
    b: []const f32,
    d: Conv2dDims,
) bool {
    const pool = pc.pool orelse return false;
    const cin_pg = d.cin / d.groups;
    const work = std.math.mul(usize, parallel.saturatedMul3(d.oh * d.ow, d.cout, cin_pg), d.kh * d.kw) catch std.math.maxInt(usize);
    const thread_count = common.generalConvThreadCount(split, work);
    if (thread_count == 1) return false;
    tile.forRange(pool, Conv2dGradCtx, .{ .out = out, .a = a, .b = b, .d = d }, split, thread_count, runFn);
    return true;
}

/// grad wrt input: gx[h,w,ci] = Σ over valid (oh,ow,kh,kw, co∈group(ci)) of
/// gy[oh,ow,co] · w[co,kh,kw, ci_local]. `out` is [H,W,Cin]. Parallel split
/// over input rows (d.h) — disjoint writes, bit-identical to serial.
pub fn conv2dBackwardInputInto(pc: ParallelConfig, out: *Tensor, gy: *const Tensor, weight: *const Tensor, d: Conv2dDims) void {
    const gx = out.data();
    const gyd = gy.dataConst();
    const wt = weight.dataConst();
    if (maybeParallelConv2dGrad(common.refSerial(pc), runConv2dBackwardInputRows, d.h, gx, gyd, wt, d)) return;
    conv2dBackwardInputRangeRows(gx, gyd, wt, d, 0, d.h);
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
pub fn conv2dBackwardWeightInto(pc: ParallelConfig, out: *Tensor, input: *const Tensor, gy: *const Tensor, d: Conv2dDims) void {
    const gw = out.data();
    const ind = input.dataConst();
    const gyd = gy.dataConst();
    if (maybeParallelConv2dGrad(common.refSerial(pc), runConv2dBackwardWeightCout, d.cout, gw, ind, gyd, d)) return;
    conv2dBackwardWeightRangeCout(gw, ind, gyd, d, 0, d.cout);
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

const Col2imCtx = struct {
    out: []f32,
    col: []const f32,
    d: Conv2dDims,
};

fn runCol2imRows(c: Col2imCtx, h_start: usize, h_end: usize) void {
    col2imRangeRows(c.out, c.col, c.d, h_start, h_end);
}

/// `out` is [H,W,Cin]; `col` is the im2col layout
/// `col[(oy·OW+ox)·ksz + (ky·KW+kx)·Cin + ic]` with `ksz = KH·KW·Cin`
/// (groups == 1, matching `im2colInto`). Parallel over input rows.
pub fn col2imInto(pc: ParallelConfig, out: *Tensor, col: *const Tensor, d: Conv2dDims) void {
    const o = out.data();
    const cd = col.dataConst();
    if (common.refSerial(pc).pool) |pool| {
        // Each input element gathers at most (kh/stride_h + 1)*(kw/stride_w + 1)
        // col rows.
        const work = parallel.saturatedMul3(d.h * d.w, d.cin, (d.kh / d.stride_h + 1) * (d.kw / d.stride_w + 1));
        const tc = common.generalConvThreadCount(d.h, work);
        if (tc > 1) {
            tile.forRange(pool, Col2imCtx, .{ .out = o, .col = cd, .d = d }, d.h, tc, runCol2imRows);
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

pub fn conv1dInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    weight: *const Tensor,
    d: Conv1dDims,
) void {
    if (comptime isa.reference) return scalar.conv1dInto(out, input, weight, d);
    const output = common.contiguousData(out, d.out_len * d.out_channels);
    const input_data = common.contiguousDataConst(input, d.seq * d.in_channels);
    const in_per_group = d.in_channels / d.groups;
    const weight_data = common.contiguousDataConst(weight, d.taps * in_per_group * d.out_channels);
    if (maybeParallelConv1d(pc, output, input_data, weight_data, d)) return;
    conv1dForwardRange(output, input_data, weight_data, d, 0, d.out_len);
}

const Conv1dCtx = struct {
    out: []f32,
    input: []const f32,
    weight: []const f32,
    d: Conv1dDims,
};

fn runConv1dRows(c: Conv1dCtx, start: usize, end: usize) void {
    conv1dForwardRange(c.out, c.input, c.weight, c.d, start, end);
}

fn maybeParallelConv1d(
    pc: ParallelConfig,
    out: []f32,
    input: []const f32,
    weight: []const f32,
    d: Conv1dDims,
) bool {
    const pool = pc.pool orelse return false;
    const in_per_group = d.in_channels / d.groups;
    const work = std.math.mul(usize, parallel.saturatedMul3(d.out_len, in_per_group, d.out_channels), d.taps) catch std.math.maxInt(usize);
    const thread_count = common.generalConvThreadCount(d.out_len, work);
    if (thread_count == 1) return false;
    tile.forRange(pool, Conv1dCtx, .{ .out = out, .input = input, .weight = weight, .d = d }, d.out_len, thread_count, runConv1dRows);
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
pub fn col2im1dInto(
    pc: ParallelConfig,
    out: *Tensor,
    col: *const Tensor,
    t_in: usize,
    out_len: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
) void {
    if (comptime isa.reference) return scalar.col2im1dInto(out, col, t_in, out_len, out_channels, taps, stride, pad);
    const output = common.contiguousData(out, out_len * out_channels);
    const col_data = common.contiguousDataConst(col, t_in * taps * out_channels);
    if (maybeParallelCol2im1d(pc, output, col_data, t_in, out_len, out_channels, taps, stride, pad)) return;
    col2im1dRange(output, col_data, t_in, out_channels, taps, stride, pad, 0, out_len);
}

const Col2im1dCtx = struct {
    out: []f32,
    col: []const f32,
    t_in: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
};

fn runCol2im1dRows(c: Col2im1dCtx, start: usize, end: usize) void {
    col2im1dRange(c.out, c.col, c.t_in, c.out_channels, c.taps, c.stride, c.pad, start, end);
}

fn maybeParallelCol2im1d(
    pc: ParallelConfig,
    out: []f32,
    col: []const f32,
    t_in: usize,
    out_len: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
) bool {
    const pool = pc.pool orelse return false;
    // Each output element gathers at most taps/stride + 1 col entries.
    const work = parallel.saturatedMul3(out_len, out_channels, taps / stride + 1);
    const thread_count = common.generalConvThreadCount(out_len, work);
    if (thread_count == 1) return false;
    tile.forRange(pool, Col2im1dCtx, .{ .out = out, .col = col, .t_in = t_in, .out_channels = out_channels, .taps = taps, .stride = stride, .pad = pad }, out_len, thread_count, runCol2im1dRows);
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

/// VJP of conv1dInto wrt the input. `out` is `[seq, in_channels]`,
/// `gy` is `[out_len, out_channels]`, `weight` the forward
/// `[tap, in_per_group, out_channels]`. Per input row `ti` and tap `k` the
/// contributing output row is `n/stride` with `n = ti + pad - k*dilation`,
/// valid when `n >= 0`, `n % stride == 0`, and `n/stride < out_len`; the
/// channel sum is a contiguous dot over the group's out-channel slice (same
/// memory trick as generalBackwardInputRange).
pub fn conv1dBackwardInputInto(
    pc: ParallelConfig,
    out: *Tensor,
    gy: *const Tensor,
    weight: *const Tensor,
    d: Conv1dDims,
) void {
    if (comptime isa.reference) return scalar.conv1dBackwardInputInto(out, gy, weight, d);
    const output = common.contiguousData(out, d.seq * d.in_channels);
    const gy_data = common.contiguousDataConst(gy, d.out_len * d.out_channels);
    const in_per_group = d.in_channels / d.groups;
    const weight_data = common.contiguousDataConst(weight, d.taps * in_per_group * d.out_channels);
    if (maybeParallelConv1dGrad(pc, runConv1dBackwardInputRows, d.seq, output, gy_data, weight_data, d)) return;
    conv1dBackwardInputRange(output, gy_data, weight_data, d, 0, d.seq);
}

/// VJP of conv1dInto wrt the weight. `out` is
/// `[taps, in_per_group, out_channels]`, `input` and `gy` are the forward
/// operand and the upstream gradient. Splits the taps*in_per_group weight
/// rows; each row axpy-accumulates the valid `(t_out, src)` pairs of the
/// forward geometry (out-of-range padded input rows are skipped).
pub fn conv1dBackwardWeightInto(
    pc: ParallelConfig,
    out: *Tensor,
    input: *const Tensor,
    gy: *const Tensor,
    d: Conv1dDims,
) void {
    if (comptime isa.reference) return scalar.conv1dBackwardWeightInto(out, input, gy, d);
    const in_per_group = d.in_channels / d.groups;
    const output = common.contiguousData(out, d.taps * in_per_group * d.out_channels);
    const input_data = common.contiguousDataConst(input, d.seq * d.in_channels);
    const gy_data = common.contiguousDataConst(gy, d.out_len * d.out_channels);
    const rows = d.taps * in_per_group;
    if (maybeParallelConv1dGrad(pc, runConv1dBackwardWeightRows, rows, output, input_data, gy_data, d)) return;
    conv1dBackwardWeightRange(output, input_data, gy_data, d, 0, rows);
}

const Conv1dGradCtx = struct {
    out: []f32,
    a: []const f32, // gy (backward-input) / input (backward-weight)
    b: []const f32, // weight (backward-input) / gy (backward-weight)
    d: Conv1dDims,
};

fn runConv1dBackwardInputRows(c: Conv1dGradCtx, start: usize, end: usize) void {
    conv1dBackwardInputRange(c.out, c.a, c.b, c.d, start, end);
}

fn runConv1dBackwardWeightRows(c: Conv1dGradCtx, start: usize, end: usize) void {
    conv1dBackwardWeightRange(c.out, c.a, c.b, c.d, start, end);
}

fn maybeParallelConv1dGrad(
    pc: ParallelConfig,
    comptime runFn: fn (Conv1dGradCtx, usize, usize) void,
    split: usize,
    out: []f32,
    a: []const f32,
    b: []const f32,
    d: Conv1dDims,
) bool {
    const pool = pc.pool orelse return false;
    const in_per_group = d.in_channels / d.groups;
    const work = std.math.mul(usize, parallel.saturatedMul3(d.out_len, in_per_group, d.out_channels), d.taps) catch std.math.maxInt(usize);
    const thread_count = common.generalConvThreadCount(split, work);
    if (thread_count == 1) return false;
    tile.forRange(pool, Conv1dGradCtx, .{ .out = out, .a = a, .b = b, .d = d }, split, thread_count, runFn);
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

/// VJP of col2im1dInto: `out` (gcol) is `[t_in, taps*out_channels]`
/// with column index `oc*taps + k` (the forward col layout);
/// `gcol[t_in, oc*taps + k] = gy[t_in*stride + k - pad, oc]` when that row
/// index lands in `[0, t_conv)` with `t_conv = (t_in-1)*stride + taps - 2*pad`,
/// else 0. `gy` has `gy_len >= t_conv` rows — the trailing `output_pad` rows
/// were forward-zeroed and never map back.
pub fn col2im1dBackwardInto(
    pc: ParallelConfig,
    out: *Tensor,
    gy: *const Tensor,
    t_in: usize,
    gy_len: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
) void {
    if (comptime isa.reference) return scalar.col2im1dBackwardInto(out, gy, t_in, gy_len, out_channels, taps, stride, pad);
    const output = common.contiguousData(out, t_in * taps * out_channels);
    const gy_data = common.contiguousDataConst(gy, gy_len * out_channels);
    const t_conv = (t_in - 1) * stride + taps - 2 * pad;
    std.debug.assert(gy_len >= t_conv);
    if (maybeParallelCol2im1dBackward(pc, output, gy_data, t_in, t_conv, out_channels, taps, stride, pad)) return;
    col2im1dBackwardRange(output, gy_data, t_conv, out_channels, taps, stride, pad, 0, t_in);
}

const Col2im1dBackwardCtx = struct {
    out: []f32,
    gy: []const f32,
    t_conv: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
};

fn runCol2im1dBackwardRows(c: Col2im1dBackwardCtx, start: usize, end: usize) void {
    col2im1dBackwardRange(c.out, c.gy, c.t_conv, c.out_channels, c.taps, c.stride, c.pad, start, end);
}

fn maybeParallelCol2im1dBackward(
    pc: ParallelConfig,
    out: []f32,
    gy: []const f32,
    t_in: usize,
    t_conv: usize,
    out_channels: usize,
    taps: usize,
    stride: usize,
    pad: usize,
) bool {
    const pool = pc.pool orelse return false;
    const work = parallel.saturatedMul3(t_in, out_channels, taps);
    const thread_count = common.generalConvThreadCount(t_in, work);
    if (thread_count == 1) return false;
    tile.forRange(pool, Col2im1dBackwardCtx, .{ .out = out, .gy = gy, .t_conv = t_conv, .out_channels = out_channels, .taps = taps, .stride = stride, .pad = pad }, t_in, thread_count, runCol2im1dBackwardRows);
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

// ---------------- The scalar reference arms ----------------

/// The scalar reference twins of this file's kernel entries: plain serial
/// loops, no SIMD, no pool. On `-Dbackend=scalar` builds (`isa.reference`)
/// the entries above dispatch here; on native builds the twins stay
/// reachable for `backend/parity_test.zig` and the ag conv tests. The
/// shared-core routes (im2col/col2im, the conv2d backward gathers, the
/// Winograd transforms) have no twin: their vector body IS the reference,
/// run serially on the reference build via `common.refSerial`.
pub const scalar = struct {
    pub fn causalDepthwiseConv1dInto(
        out: *Tensor,
        input: *const Tensor,
        kernel: *const Tensor,
        state: ?[]const f32,
        seq: usize,
        channels: usize,
        taps: usize,
        dilation: usize,
    ) void {
        causalDepthwiseConv1dRange(out.data(), input.dataConst(), kernel.dataConst(), state, seq, channels, taps, dilation, 0, channels);
    }

    pub fn causalDepthwiseConv1dBackwardInputInto(
        out: *Tensor,
        gy: *const Tensor,
        kernel: *const Tensor,
        seq: usize,
        channels: usize,
        taps: usize,
        dilation: usize,
    ) void {
        causalDepthwiseConv1dBackwardInputRange(out.data(), gy.dataConst(), kernel.dataConst(), seq, channels, taps, dilation, 0, channels);
    }

    pub fn causalDepthwiseConv1dBackwardKernelInto(
        out: *Tensor,
        input: *const Tensor,
        gy: *const Tensor,
        state: ?[]const f32,
        seq: usize,
        channels: usize,
        taps: usize,
        dilation: usize,
    ) void {
        causalDepthwiseConv1dBackwardKernelRange(out.data(), input.dataConst(), gy.dataConst(), state, seq, channels, taps, dilation, 0, channels);
    }

    pub fn causalConv1dInto(
        out: *Tensor,
        input: *const Tensor,
        weight: *const Tensor,
        state: ?[]const f32,
        bias: ?[]const f32,
        seq: usize,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
    ) void {
        groupedCausalConv1dRange(out.data(), input.dataConst(), weight.dataConst(), state, bias, in_channels, out_channels, taps, dilation, 1, 0, seq);
    }

    pub fn causalConv1dBackwardInputInto(
        out: *Tensor,
        gy: *const Tensor,
        weight: *const Tensor,
        seq: usize,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
    ) void {
        groupedCausalConv1dBackwardInputRange(out.data(), gy.dataConst(), weight.dataConst(), seq, in_channels, out_channels, taps, dilation, 1, 0, seq);
    }

    pub fn causalConv1dBackwardWeightInto(
        out: *Tensor,
        input: *const Tensor,
        gy: *const Tensor,
        state: ?[]const f32,
        seq: usize,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
    ) void {
        groupedCausalConv1dBackwardWeightRange(out.data(), input.dataConst(), gy.dataConst(), state, seq, in_channels, out_channels, taps, dilation, 1, 0, taps * in_channels);
    }

    pub fn groupedCausalConv1dInto(
        out: *Tensor,
        input: *const Tensor,
        weight: *const Tensor,
        state: ?[]const f32,
        bias: ?[]const f32,
        seq: usize,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
        groups: usize,
    ) void {
        groupedCausalConv1dRange(out.data(), input.dataConst(), weight.dataConst(), state, bias, in_channels, out_channels, taps, dilation, groups, 0, seq);
    }

    pub fn groupedCausalConv1dBackwardInputInto(
        out: *Tensor,
        gy: *const Tensor,
        weight: *const Tensor,
        seq: usize,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
        groups: usize,
    ) void {
        groupedCausalConv1dBackwardInputRange(out.data(), gy.dataConst(), weight.dataConst(), seq, in_channels, out_channels, taps, dilation, groups, 0, seq);
    }

    pub fn groupedCausalConv1dBackwardWeightInto(
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
    ) void {
        const in_per_group = in_channels / groups;
        groupedCausalConv1dBackwardWeightRange(out.data(), input.dataConst(), gy.dataConst(), state, seq, in_channels, out_channels, taps, dilation, groups, 0, taps * in_per_group);
    }

    /// Scalar reference of the depthwise entry: the same tap-major walk,
    /// one channel at a time.
    pub fn conv2dDepthwiseInto(
        out: *Tensor,
        input: *const Tensor,
        taps: []const f32,
        bias: ?[]const f32,
        d: Conv2dDims,
    ) void {
        const o = out.data();
        const in = input.dataConst();
        const c = d.cin;
        for (0..d.oh) |oh| {
            for (0..d.ow) |ow| {
                const dst = o[(oh * d.ow + ow) * c ..][0..c];
                for (0..c) |ci| dst[ci] = if (bias) |b| b[ci] else 0;
                for (0..d.kh) |ky| {
                    const ih_s = @as(isize, @intCast(oh * d.stride_h + ky)) - @as(isize, @intCast(d.pad_h));
                    if (ih_s < 0 or ih_s >= @as(isize, @intCast(d.h))) continue;
                    const ih: usize = @intCast(ih_s);
                    for (0..d.kw) |kx| {
                        const iw_s = @as(isize, @intCast(ow * d.stride_w + kx)) - @as(isize, @intCast(d.pad_w));
                        if (iw_s < 0 or iw_s >= @as(isize, @intCast(d.w))) continue;
                        const iw: usize = @intCast(iw_s);
                        const in_row = in[(ih * d.w + iw) * c ..][0..c];
                        const w_tap = taps[(ky * d.kw + kx) * c ..][0..c];
                        for (0..c) |ci| dst[ci] += in_row[ci] * w_tap[ci];
                    }
                }
            }
        }
    }

    /// Scalar reference conv2d (channel-last [H,W,Cin] -> [OH,OW,Cout] with
    /// stride, explicit zero pad, grouped/depthwise; see `Conv2dDims`).
    pub fn conv2dInto(
        out: *Tensor,
        input: *const Tensor,
        weight: *const Tensor,
        bias: ?[]const f32,
        d: Conv2dDims,
    ) void {
        const o = out.data();
        const in = input.dataConst();
        const w = weight.dataConst();
        const cin_pg = d.cin / d.groups;
        const cout_pg = d.cout / d.groups;
        for (0..d.oh) |oh| {
            for (0..d.ow) |ow| {
                for (0..d.cout) |oc| {
                    const g = oc / cout_pg;
                    var acc: f32 = if (bias) |b| b[oc] else 0;
                    for (0..d.kh) |kh| {
                        const ih_i = @as(isize, @intCast(oh * d.stride_h + kh)) - @as(isize, @intCast(d.pad_h));
                        if (ih_i < 0 or ih_i >= @as(isize, @intCast(d.h))) continue;
                        for (0..d.kw) |kw| {
                            const iw_i = @as(isize, @intCast(ow * d.stride_w + kw)) - @as(isize, @intCast(d.pad_w));
                            if (iw_i < 0 or iw_i >= @as(isize, @intCast(d.w))) continue;
                            const ih: usize = @intCast(ih_i);
                            const iw: usize = @intCast(iw_i);
                            for (0..cin_pg) |ic| {
                                const iv = in[(ih * d.w + iw) * d.cin + g * cin_pg + ic];
                                const wv = w[((oc * d.kh + kh) * d.kw + kw) * cin_pg + ic];
                                acc += iv * wv;
                            }
                        }
                    }
                    o[(oh * d.ow + ow) * d.cout + oc] = acc;
                }
            }
        }
    }

    /// Scalar reference conv1d (general non-causal, symmetric zero pad,
    /// stride, dilation, groups; see `Conv1dDims`).
    pub fn conv1dInto(
        out: *Tensor,
        input: *const Tensor,
        weight: *const Tensor,
        d: Conv1dDims,
    ) void {
        const o = out.data();
        const in = input.dataConst();
        const w = weight.dataConst();
        const in_per_group = d.in_channels / d.groups;
        const out_per_group = d.out_channels / d.groups;
        for (0..d.out_len) |t| {
            for (0..d.out_channels) |oc| {
                const g = oc / out_per_group;
                var acc: f32 = 0;
                for (0..d.taps) |k| {
                    const pos = t * d.stride + k * d.dilation;
                    if (pos < d.pad) continue;
                    const src = pos - d.pad;
                    if (src >= d.seq) continue;
                    for (0..in_per_group) |local_i| {
                        const iv = in[src * d.in_channels + g * in_per_group + local_i];
                        const wv = w[(k * in_per_group + local_i) * d.out_channels + oc];
                        acc += iv * wv;
                    }
                }
                o[t * d.out_channels + oc] = acc;
            }
        }
    }

    /// Scalar reference col2im1d gather (layout contract on the entry above).
    pub fn col2im1dInto(
        out: *Tensor,
        col: *const Tensor,
        t_in: usize,
        out_len: usize,
        out_channels: usize,
        taps: usize,
        stride: usize,
        pad: usize,
    ) void {
        const o = out.data();
        const c = col.dataConst();
        const t_conv = (t_in - 1) * stride + taps - 2 * pad;
        for (0..out_len) |t_out| {
            if (t_out >= t_conv) {
                for (0..out_channels) |oc| o[t_out * out_channels + oc] = 0;
                continue;
            }
            const t_abs = t_out + pad;
            const t_in_min: usize = if (t_abs + 1 > taps) (t_abs + 1 - taps + stride - 1) / stride else 0;
            const t_in_max: usize = @min(t_in - 1, t_abs / stride);
            for (0..out_channels) |oc| {
                var acc: f32 = 0;
                var ti = t_in_min;
                while (ti <= t_in_max) : (ti += 1) {
                    const k = t_abs - ti * stride;
                    std.debug.assert(k < taps);
                    acc += c[ti * (taps * out_channels) + oc * taps + k];
                }
                o[t_out * out_channels + oc] = acc;
            }
        }
    }

    /// Scalar reference conv1d backward-input (formula on the entry above).
    pub fn conv1dBackwardInputInto(
        out: *Tensor,
        gy: *const Tensor,
        weight: *const Tensor,
        d: Conv1dDims,
    ) void {
        const o = out.data();
        const g = gy.dataConst();
        const w = weight.dataConst();
        const in_per_group = d.in_channels / d.groups;
        const out_per_group = d.out_channels / d.groups;
        for (0..d.seq) |ti| {
            for (0..d.in_channels) |ic| {
                const group = ic / in_per_group;
                const local_i = ic % in_per_group;
                var acc: f32 = 0;
                for (0..d.taps) |k| {
                    const shifted = k * d.dilation;
                    if (shifted > ti + d.pad) continue;
                    const n = ti + d.pad - shifted;
                    if (n % d.stride != 0) continue;
                    const t = n / d.stride;
                    if (t >= d.out_len) continue;
                    for (0..out_per_group) |local_o| {
                        const oc = group * out_per_group + local_o;
                        acc += g[t * d.out_channels + oc] * w[(k * in_per_group + local_i) * d.out_channels + oc];
                    }
                }
                o[ti * d.in_channels + ic] = acc;
            }
        }
    }

    /// Scalar reference conv1d backward-weight (formula on the entry above).
    pub fn conv1dBackwardWeightInto(
        out: *Tensor,
        input: *const Tensor,
        gy: *const Tensor,
        d: Conv1dDims,
    ) void {
        const o = out.data();
        const in = input.dataConst();
        const g = gy.dataConst();
        const in_per_group = d.in_channels / d.groups;
        const out_per_group = d.out_channels / d.groups;
        for (0..d.taps) |k| {
            for (0..in_per_group) |local_i| {
                for (0..d.out_channels) |oc| {
                    const group = oc / out_per_group;
                    var acc: f32 = 0;
                    for (0..d.out_len) |t| {
                        const pos = t * d.stride + k * d.dilation;
                        if (pos < d.pad) continue;
                        const src = pos - d.pad;
                        if (src >= d.seq) continue;
                        acc += g[t * d.out_channels + oc] * in[src * d.in_channels + group * in_per_group + local_i];
                    }
                    o[(k * in_per_group + local_i) * d.out_channels + oc] = acc;
                }
            }
        }
    }

    /// Scalar reference col2im1d backward (formula on the entry above).
    pub fn col2im1dBackwardInto(
        out: *Tensor,
        gy: *const Tensor,
        t_in: usize,
        gy_len: usize,
        out_channels: usize,
        taps: usize,
        stride: usize,
        pad: usize,
    ) void {
        const o = out.data();
        const g = gy.dataConst();
        const t_conv = (t_in - 1) * stride + taps - 2 * pad;
        std.debug.assert(gy_len >= t_conv);
        const row_stride = taps * out_channels;
        for (0..t_in) |ti| {
            for (0..out_channels) |oc| {
                for (0..taps) |k| {
                    const pos = ti * stride + k;
                    var value: f32 = 0;
                    if (pos >= pad) {
                        const t_out = pos - pad;
                        if (t_out < t_conv) value = g[t_out * out_channels + oc];
                    }
                    o[ti * row_stride + oc * taps + k] = value;
                }
            }
        }
    }

    fn causalDepthwiseConv1dRange(
        out: []f32,
        input: []const f32,
        kernel: []const f32,
        state: ?[]const f32,
        seq: usize,
        channels: usize,
        taps: usize,
        dilation: usize,
        channel_start: usize,
        channel_end: usize,
    ) void {
        const pad = dilation * (taps - 1);
        for (0..seq) |t| {
            for (channel_start..channel_end) |c| {
                var acc: f32 = 0;
                for (0..taps) |k| {
                    acc += causalDepthwiseInputValue(input, state, seq, channels, pad, dilation, t, c, k) * kernel[c * taps + k];
                }
                out[t * channels + c] = acc;
            }
        }
    }

    fn causalDepthwiseConv1dBackwardInputRange(
        out: []f32,
        gy: []const f32,
        kernel: []const f32,
        seq: usize,
        channels: usize,
        taps: usize,
        dilation: usize,
        channel_start: usize,
        channel_end: usize,
    ) void {
        const pad = dilation * (taps - 1);
        for (0..seq) |p| {
            for (channel_start..channel_end) |c| {
                var acc: f32 = 0;
                for (0..taps) |k| {
                    const t_base = p + pad;
                    if (k * dilation > t_base) continue;
                    const t = t_base - k * dilation;
                    if (t < seq) acc += gy[t * channels + c] * kernel[c * taps + k];
                }
                out[p * channels + c] = acc;
            }
        }
    }

    fn causalDepthwiseConv1dBackwardKernelRange(
        out: []f32,
        input: []const f32,
        gy: []const f32,
        state: ?[]const f32,
        seq: usize,
        channels: usize,
        taps: usize,
        dilation: usize,
        channel_start: usize,
        channel_end: usize,
    ) void {
        const pad = dilation * (taps - 1);
        for (channel_start..channel_end) |c| {
            for (0..taps) |k| {
                var acc: f32 = 0;
                for (0..seq) |t| {
                    acc += gy[t * channels + c] * causalDepthwiseInputValue(input, state, seq, channels, pad, dilation, t, c, k);
                }
                out[c * taps + k] = acc;
            }
        }
    }

    fn groupedCausalConv1dRange(
        out: []f32,
        input: []const f32,
        weight: []const f32,
        state: ?[]const f32,
        bias: ?[]const f32,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
        groups: usize,
        t_start: usize,
        t_end: usize,
    ) void {
        const pad = dilation * (taps - 1);
        const in_per_group = in_channels / groups;
        const out_per_group = out_channels / groups;
        for (t_start..t_end) |t| {
            for (0..out_channels) |o| {
                const group = o / out_per_group;
                const input_start = group * in_per_group;
                var acc: f32 = 0;
                for (0..taps) |k| {
                    for (0..in_per_group) |local_i| {
                        const i = input_start + local_i;
                        acc += causalConvInputValue(input, state, in_channels, pad, t, i, k, dilation) * weight[(k * in_per_group + local_i) * out_channels + o];
                    }
                }
                if (bias) |b| acc += b[o];
                out[t * out_channels + o] = acc;
            }
        }
    }

    fn groupedCausalConv1dBackwardInputRange(
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
        const pad = dilation * (taps - 1);
        const in_per_group = in_channels / groups;
        const out_per_group = out_channels / groups;
        for (p_start..p_end) |p| {
            for (0..in_channels) |i| {
                const group = i / in_per_group;
                const local_i = i - group * in_per_group;
                const out_start = group * out_per_group;
                var acc: f32 = 0;
                for (0..taps) |k| {
                    const t = p + pad - k * dilation;
                    if (t >= seq) continue;
                    for (out_start..out_start + out_per_group) |o| {
                        acc += gy[t * out_channels + o] * weight[(k * in_per_group + local_i) * out_channels + o];
                    }
                }
                out[p * in_channels + i] = acc;
            }
        }
    }

    fn groupedCausalConv1dBackwardWeightRange(
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
        const pad = dilation * (taps - 1);
        const in_per_group = in_channels / groups;
        const out_per_group = out_channels / groups;
        for (row_start..row_end) |row| {
            const k = row / in_per_group;
            const local_i = row % in_per_group;
            for (0..out_channels) |o| {
                const group = o / out_per_group;
                const i = group * in_per_group + local_i;
                var acc: f32 = 0;
                for (0..seq) |t| {
                    acc += gy[t * out_channels + o] * causalConvInputValue(input, state, in_channels, pad, t, i, k, dilation);
                }
                out[row * out_channels + o] = acc;
            }
        }
    }

    fn causalConvInputValue(
        input: []const f32,
        state: ?[]const f32,
        in_channels: usize,
        pad: usize,
        t: usize,
        i: usize,
        k: usize,
        dilation: usize,
    ) f32 {
        const shifted = t + k * dilation;
        if (shifted >= pad) return input[(shifted - pad) * in_channels + i];
        const st = state orelse return 0;
        return st[shifted * in_channels + i];
    }

    fn causalDepthwiseInputValue(
        input: []const f32,
        state: ?[]const f32,
        seq: usize,
        channels: usize,
        pad: usize,
        dilation: usize,
        t: usize,
        c: usize,
        k: usize,
    ) f32 {
        _ = seq;
        const u = t + k * dilation;
        if (u >= pad) {
            return input[(u - pad) * channels + c];
        }
        const st = state orelse return 0;
        return st[u * channels + c];
    }
};
