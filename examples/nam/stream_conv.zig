//! Streaming causal 1-D convolution for the realtime engines: the example-
//! local counterpart of upstream's Conv1D + RingBuffer pair
//! (NeuralAmpModelerCore NAM/conv1d.cpp, NAM/ring_buffer.cpp). Weights are
//! stored [tap, in, out] (tap K-1 = newest sample — the same orientation as
//! fucina's core `causalConv1d` and PyTorch's causal-pad cross-correlation);
//! the per-conv history holds the last `dilation*(K-1)` input rows so output
//! is independent of how the stream is chunked.
//!
//! All buffers are allocated at init; `process` is allocation-free.

const std = @import("std");

const vector_len: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;
const Vf32 = @Vector(vector_len, f32);

pub const StreamConv = struct {
    /// [tap, in, out] row-major.
    weight: []f32,
    /// len out_channels, or empty for bias-less convs.
    bias: []f32,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    taps: usize,
    dilation: usize,
    /// [pad, in] rows preceding the next chunk (pad = dilation*(taps-1)),
    /// oldest first.
    history: []f32,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
        has_bias: bool,
    ) !StreamConv {
        return initGrouped(allocator, in_channels, out_channels, taps, dilation, has_bias, 1);
    }

    pub fn initGrouped(
        allocator: std.mem.Allocator,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
        has_bias: bool,
        groups: usize,
    ) !StreamConv {
        // Untrusted .nam shapes reach here: enforce the real bounds with a
        // returned error, not just a debug assert. taps==0 underflows `pad`
        // below; taps>max_taps would OOB the fixed [max_taps] tile arrays in
        // process(); dilation==0 is degenerate; a huge dilation blows up the
        // history allocation. (IR cabs use their own engine, not StreamConv.)
        if (taps < 1 or taps > max_taps or dilation < 1 or dilation > max_dilation or groups == 0) {
            return error.InvalidConvShape;
        }
        if (in_channels % groups != 0 or out_channels % groups != 0) {
            return error.InvalidConvShape;
        }
        const in_per_group = in_channels / groups;
        const weight = try allocator.alloc(f32, taps * in_per_group * out_channels);
        errdefer allocator.free(weight);
        const bias = try allocator.alloc(f32, if (has_bias) out_channels else 0);
        errdefer allocator.free(bias);
        const pad = dilation * (taps - 1);
        const history = try allocator.alloc(f32, pad * in_channels);
        @memset(history, 0);
        return .{
            .weight = weight,
            .bias = bias,
            .in_channels = in_channels,
            .out_channels = out_channels,
            .groups = groups,
            .taps = taps,
            .dilation = dilation,
            .history = history,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *StreamConv) void {
        self.allocator.free(self.weight);
        self.allocator.free(self.bias);
        self.allocator.free(self.history);
        self.* = undefined;
    }

    pub fn weightCount(self: *const StreamConv) usize {
        return self.weight.len + self.bias.len;
    }

    /// Loads from the NAM flat-weight stream (PyTorch (out, in, k) row-major
    /// order, spec §5.1/§5.2), permuting into our [tap, in, out] layout;
    /// bias follows when present. Returns the number of floats consumed.
    pub fn loadNamWeights(self: *StreamConv, stream: []const f32) usize {
        var idx: usize = 0;
        const in_per_group = self.in_channels / self.groups;
        const out_per_group = self.out_channels / self.groups;
        for (0..self.groups) |g| {
            for (0..out_per_group) |local_o| {
                const o = g * out_per_group + local_o;
                for (0..in_per_group) |local_i| {
                    for (0..self.taps) |k| {
                        self.weight[(k * in_per_group + local_i) * self.out_channels + o] = stream[idx];
                        idx += 1;
                    }
                }
            }
        }
        for (self.bias) |*b| {
            b.* = stream[idx];
            idx += 1;
        }
        return idx;
    }

    pub fn loadDenseWeights(self: *StreamConv, stream: []const f32) usize {
        std.debug.assert(self.groups == 1);
        var idx: usize = 0;
        for (0..self.out_channels) |o| {
            for (0..self.in_channels) |i| {
                for (0..self.taps) |k| {
                    self.weight[(k * self.in_channels + i) * self.out_channels + o] = stream[idx];
                    idx += 1;
                }
            }
        }
        for (self.bias) |*b| {
            b.* = stream[idx];
            idx += 1;
        }
        return idx;
    }

    pub fn reset(self: *StreamConv) void {
        @memset(self.history, 0);
    }

    pub const max_taps = 64;
    /// Generous dilation ceiling for untrusted .nam files (real configs use
    /// <= ~512): bounds the history allocation so a malformed huge dilation
    /// fails with InvalidConvShape instead of OOM/overflow.
    pub const max_dilation = 1 << 16;
    /// Frames processed per register tile: each weight vector load feeds
    /// `time_tile` FMAs (the Eigen-GEMM-style amortization upstream gets
    /// from processing whole blocks as matrix products).
    pub const time_tile = 8;

    /// y[t,o] = bias[o] + sum_{k,i} xhat[t + k*dilation - pad, i]*w[k,i,o]
    /// over `frames` rows; xhat reads `history` for pre-chunk rows. When
    /// `accumulate` is true the result is added into `out` instead of
    /// overwriting it. The main loop register-tiles over out-channel
    /// vectors AND `time_tile` frames; per output element the accumulation
    /// order (bias, then k-major i-inner FMA) is identical in every path,
    /// so results are bit-identical across tile boundaries and chunkings.
    /// Does NOT advance history — call `push` with the same input
    /// afterwards.
    pub fn process(self: *const StreamConv, input: []const f32, out: []f32, frames: usize, comptime accumulate: bool) void {
        if (self.groups != 1) return self.processGrouped(input, out, frames, accumulate);

        const in_ch = self.in_channels;
        const out_ch = self.out_channels;
        const pad = self.dilation * (self.taps - 1);
        std.debug.assert(self.taps <= max_taps);

        var t: usize = 0;
        while (t + time_tile <= frames) : (t += time_tile) {
            var tile_rows: [max_taps][time_tile][]const f32 = undefined;
            for (0..self.taps) |k| {
                inline for (0..time_tile) |tt| {
                    const shifted = t + tt + k * self.dilation;
                    tile_rows[k][tt] = if (shifted >= pad)
                        input[(shifted - pad) * in_ch ..][0..in_ch]
                    else
                        self.history[shifted * in_ch ..][0..in_ch];
                }
            }

            var o: usize = 0;
            while (o + vector_len <= out_ch) : (o += vector_len) {
                var acc: [time_tile]Vf32 = undefined;
                inline for (0..time_tile) |tt| {
                    acc[tt] = if (accumulate)
                        out[(t + tt) * out_ch + o ..][0..vector_len].*
                    else
                        @splat(0);
                }
                for (0..self.taps) |k| {
                    const rows = tile_rows[k];
                    for (0..in_ch) |i| {
                        const wv: Vf32 = self.weight[(k * in_ch + i) * out_ch + o ..][0..vector_len].*;
                        inline for (0..time_tile) |tt| {
                            acc[tt] = @mulAdd(Vf32, @splat(rows[tt][i]), wv, acc[tt]);
                        }
                    }
                }
                if (self.bias.len > 0) {
                    const bv: Vf32 = self.bias[o..][0..vector_len].*;
                    inline for (0..time_tile) |tt| acc[tt] += bv;
                }
                inline for (0..time_tile) |tt| {
                    out[(t + tt) * out_ch + o ..][0..vector_len].* = acc[tt];
                }
            }
            while (o < out_ch) : (o += 1) {
                inline for (0..time_tile) |tt| {
                    var acc: f32 = if (accumulate) out[(t + tt) * out_ch + o] else 0;
                    for (0..self.taps) |k| {
                        const x_row = tile_rows[k][tt];
                        for (0..in_ch) |i| {
                            acc = @mulAdd(f32, x_row[i], self.weight[(k * in_ch + i) * out_ch + o], acc);
                        }
                    }
                    if (self.bias.len > 0) acc += self.bias[o];
                    out[(t + tt) * out_ch + o] = acc;
                }
            }
        }

        // Remaining frames (< time_tile): the per-frame path.
        var x_rows: [max_taps][]const f32 = undefined;
        while (t < frames) : (t += 1) {
            for (0..self.taps) |k| {
                const shifted = t + k * self.dilation;
                x_rows[k] = if (shifted >= pad)
                    input[(shifted - pad) * in_ch ..][0..in_ch]
                else
                    self.history[shifted * in_ch ..][0..in_ch];
            }
            const out_row = out[t * out_ch ..][0..out_ch];

            var o: usize = 0;
            while (o + vector_len <= out_ch) : (o += vector_len) {
                var acc: Vf32 = if (accumulate)
                    out_row[o..][0..vector_len].*
                else
                    @splat(0);
                for (0..self.taps) |k| {
                    const x_row = x_rows[k];
                    for (0..in_ch) |i| {
                        const wv: Vf32 = self.weight[(k * in_ch + i) * out_ch + o ..][0..vector_len].*;
                        acc = @mulAdd(Vf32, @splat(x_row[i]), wv, acc);
                    }
                }
                if (self.bias.len > 0) {
                    const bv: Vf32 = self.bias[o..][0..vector_len].*;
                    acc += bv;
                }
                out_row[o..][0..vector_len].* = acc;
            }
            while (o < out_ch) : (o += 1) {
                var acc: f32 = if (accumulate) out_row[o] else 0;
                for (0..self.taps) |k| {
                    const x_row = x_rows[k];
                    for (0..in_ch) |i| {
                        acc = @mulAdd(f32, x_row[i], self.weight[(k * in_ch + i) * out_ch + o], acc);
                    }
                }
                if (self.bias.len > 0) acc += self.bias[o];
                out_row[o] = acc;
            }
        }
    }

    fn processGrouped(self: *const StreamConv, input: []const f32, out: []f32, frames: usize, comptime accumulate: bool) void {
        const in_ch = self.in_channels;
        const out_ch = self.out_channels;
        const in_per_group = in_ch / self.groups;
        const out_per_group = out_ch / self.groups;
        const pad = self.dilation * (self.taps - 1);

        var x_rows: [max_taps][]const f32 = undefined;
        for (0..frames) |t| {
            for (0..self.taps) |k| {
                const shifted = t + k * self.dilation;
                x_rows[k] = if (shifted >= pad)
                    input[(shifted - pad) * in_ch ..][0..in_ch]
                else
                    self.history[shifted * in_ch ..][0..in_ch];
            }
            const out_row = out[t * out_ch ..][0..out_ch];
            for (0..out_ch) |o| {
                const group = o / out_per_group;
                const input_start = group * in_per_group;
                var acc: f32 = if (accumulate) out_row[o] else 0;
                for (0..self.taps) |k| {
                    const x_row = x_rows[k];
                    for (0..in_per_group) |local_i| {
                        acc = @mulAdd(f32, x_row[input_start + local_i], self.weight[(k * in_per_group + local_i) * out_ch + o], acc);
                    }
                }
                if (self.bias.len > 0) acc += self.bias[o];
                out_row[o] = acc;
            }
        }
    }

    /// Advances the history by `frames` rows of `input` (the same buffer
    /// `process` consumed).
    pub fn push(self: *StreamConv, input: []const f32, frames: usize) void {
        const in_ch = self.in_channels;
        if (in_ch == 0 or self.history.len == 0) return;
        const pad = self.history.len / in_ch;
        if (frames >= pad) {
            @memcpy(self.history, input[(frames - pad) * in_ch ..][0 .. pad * in_ch]);
            return;
        }
        const keep = pad - frames;
        std.mem.copyForwards(f32, self.history[0 .. keep * in_ch], self.history[frames * in_ch ..][0 .. keep * in_ch]);
        @memcpy(self.history[keep * in_ch ..], input[0 .. frames * in_ch]);
    }
};

test {
    _ = @import("stream_conv_tests.zig");
}
