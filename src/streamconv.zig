//! Streaming causal 1-D convolutions — the state discipline every neural
//! audio codec decoder needs (SEANet / Mimi / GTCRN lineage), written once:
//!
//! - `StreamingConv1d`: keeps the last `kernel_eff − stride` input columns
//!   as the causal left context; feeding the stream chunk-by-chunk produces
//!   bit-identical output to convolving the whole signal.
//! - `StreamingConvTranspose1d`: keeps a `kernel − stride` overlap-add tail;
//!   the carried partial excludes the bias (added once on emission) — the
//!   correction the reference implementations all need and every port
//!   otherwise re-derives.
//!
//! Channel-major f32 layout (`[channels][time]`), matching the hand-rolled
//! port style these were promoted from (pocket-tts Mimi; the aec.zig GTCRN
//! rings are the same discipline). Weights: Conv1d `[out][in][k]`,
//! ConvTranspose1d `[in][out][k]` (torch layouts); `depthwise` serves the
//! groups==channels upsamplers. Unit tests pin chunked streaming against
//! whole-signal convolution.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Weights = struct {
    w: []const f32,
    b: ?[]const f32,
    in_ch: usize,
    out_ch: usize,
    k: usize,
};

fn dot(a: []const f32, b: []const f32) f32 {
    const V = @Vector(8, f32);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var i: usize = 0;
    while (i + 16 <= a.len) : (i += 16) {
        acc0 += @as(V, a[i..][0..8].*) * @as(V, b[i..][0..8].*);
        acc1 += @as(V, a[i + 8 ..][0..8].*) * @as(V, b[i + 8 ..][0..8].*);
    }
    var s: f32 = @reduce(.Add, acc0 + acc1);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

pub const StreamingConv1d = struct {
    w: *const Weights,
    prev: []f32, // [in_ch][k_eff - stride]
    stride: usize,

    pub fn init(allocator: Allocator, w: *const Weights, stride: usize) !StreamingConv1d {
        const tail = w.k - stride;
        const prev = try allocator.alloc(f32, w.in_ch * tail);
        @memset(prev, 0);
        return .{ .w = w, .prev = prev, .stride = stride };
    }

    /// x: [in_ch][T] channel-major; out [out_ch][T/stride].
    pub fn run(self: *StreamingConv1d, allocator: Allocator, x: []const f32, t_in: usize, out: []f32) !void {
        const w = self.w;
        const tail = w.k - self.stride;
        const total = tail + t_in;
        const xt = try allocator.alloc(f32, w.in_ch * total);
        defer allocator.free(xt);
        for (0..w.in_ch) |c| {
            @memcpy(xt[c * total ..][0..tail], self.prev[c * tail ..][0..tail]);
            @memcpy(xt[c * total ..][tail..][0..t_in], x[c * t_in ..][0..t_in]);
        }
        const t_out = (total - w.k) / self.stride + 1;
        if (self.stride == 1) {
            // vectorize over output positions: out[oc][:] += w[oc,ic,k]·x[ic][k:]
            for (0..w.out_ch) |oc| {
                const orow = out[oc * t_out ..][0..t_out];
                const bias: f32 = if (w.b) |b| b[oc] else 0;
                @memset(orow, bias);
                for (0..w.in_ch) |ic| {
                    const wrow = w.w[(oc * w.in_ch + ic) * w.k ..][0..w.k];
                    const xrow = xt[ic * total ..][0..total];
                    for (0..w.k) |kk| {
                        const wv = wrow[kk];
                        if (wv == 0) continue;
                        const V = @Vector(8, f32);
                        const wsplat: V = @splat(wv);
                        var t: usize = 0;
                        while (t + 8 <= t_out) : (t += 8) {
                            var ov: V = orow[t..][0..8].*;
                            ov += wsplat * @as(V, xrow[kk + t ..][0..8].*);
                            orow[t..][0..8].* = ov;
                        }
                        while (t < t_out) : (t += 1) orow[t] += wv * xrow[kk + t];
                    }
                }
            }
        } else {
            for (0..w.out_ch) |oc| {
                const orow = out[oc * t_out ..][0..t_out];
                for (0..t_out) |t| {
                    var acc: f32 = if (w.b) |b| b[oc] else 0;
                    for (0..w.in_ch) |ic| {
                        const wrow = w.w[(oc * w.in_ch + ic) * w.k ..][0..w.k];
                        const xrow = xt[ic * total + t * self.stride ..][0..w.k];
                        acc += dot(wrow, xrow);
                    }
                    orow[t] = acc;
                }
            }
        }
        for (0..w.in_ch) |c| {
            @memcpy(self.prev[c * tail ..][0..tail], xt[c * total ..][total - tail ..][0..tail]);
        }
    }
};

pub const StreamingConvTranspose1d = struct {
    w: *const Weights, // weight [in][out][k] (transposed-conv layout)
    partial: []f32, // [out_ch][k - stride]
    stride: usize,
    depthwise: bool,

    pub fn init(allocator: Allocator, w: *const Weights, stride: usize, depthwise: bool) !StreamingConvTranspose1d {
        const oc = if (depthwise) w.in_ch else w.out_ch;
        const partial = try allocator.alloc(f32, oc * (w.k - stride));
        @memset(partial, 0);
        return .{ .w = w, .partial = partial, .stride = stride, .depthwise = depthwise };
    }

    /// x: [in_ch][T]; out [out_ch][T*stride].
    pub fn run(self: *StreamingConvTranspose1d, allocator: Allocator, x: []const f32, t_in: usize, out: []f32) !void {
        const w = self.w;
        const oc = if (self.depthwise) w.in_ch else w.out_ch;
        const pt = w.k - self.stride;
        const full = (t_in - 1) * self.stride + w.k;
        const y = try allocator.alloc(f32, oc * full);
        defer allocator.free(y);
        @memset(y, 0);
        if (self.depthwise) {
            // weight ne [k, 1, in] → per-channel taps w[c*k..]
            for (0..oc) |c| {
                const taps = w.w[c * w.k ..][0..w.k];
                const yrow = y[c * full ..][0..full];
                const xrow = x[c * t_in ..][0..t_in];
                for (0..t_in) |t| {
                    const xv = xrow[t];
                    for (0..w.k) |kk| yrow[t * self.stride + kk] += xv * taps[kk];
                }
            }
        } else {
            // torch ConvTranspose1d weight [in, out, k]
            for (0..w.in_ch) |ic| {
                const xrow = x[ic * t_in ..][0..t_in];
                for (0..w.out_ch) |ocx| {
                    const taps = w.w[(ic * w.out_ch + ocx) * w.k ..][0..w.k];
                    const yrow = y[ocx * full ..][0..full];
                    for (0..t_in) |t| {
                        const xv = xrow[t];
                        if (xv == 0) continue;
                        const V = @Vector(8, f32);
                        const xsplat: V = @splat(xv);
                        const base = t * self.stride;
                        var kk: usize = 0;
                        while (kk + 8 <= w.k) : (kk += 8) {
                            var yv: V = yrow[base + kk ..][0..8].*;
                            yv += xsplat * @as(V, taps[kk..][0..8].*);
                            yrow[base + kk ..][0..8].* = yv;
                        }
                        while (kk < w.k) : (kk += 1) yrow[base + kk] += xv * taps[kk];
                    }
                }
            }
        }
        const emit = full - pt;
        for (0..oc) |c| {
            const yrow = y[c * full ..][0..full];
            const bias: f32 = if (w.b) |b| b[c] else 0;
            // add bias to every emitted sample + incoming partial
            for (0..pt) |i| yrow[i] += self.partial[c * pt + i];
            for (0..emit) |i| out[c * emit + i] = yrow[i] + bias;
            // new partial excludes bias (added on emission next time)
            for (0..pt) |i| self.partial[c * pt + i] = yrow[emit + i];
        }
    }
};

test "streaming conv1d chunked == whole-signal causal conv" {
    const allocator = std.testing.allocator;
    // 2 in, 3 out, k=5, stride 1; signal of 24 samples fed as 6+10+8.
    var w: [3 * 2 * 5]f32 = undefined;
    var b: [3]f32 = .{ 0.1, -0.2, 0.3 };
    var x: [2 * 24]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (&w) |*v| v.* = prng.random().floatNorm(f32);
    for (&x) |*v| v.* = prng.random().floatNorm(f32);
    const cw = Weights{ .w = &w, .b = &b, .in_ch = 2, .out_ch = 3, .k = 5 };

    // whole-signal reference with 4 left-pad zeros (causal)
    var ref: [3 * 24]f32 = undefined;
    for (0..3) |oc| {
        for (0..24) |t| {
            var acc: f32 = b[oc];
            for (0..2) |ic| {
                for (0..5) |kk| {
                    const ti = @as(i64, @intCast(t)) - 4 + @as(i64, @intCast(kk));
                    if (ti >= 0) acc += w[(oc * 2 + ic) * 5 + kk] * x[ic * 24 + @as(usize, @intCast(ti))];
                }
            }
            ref[oc * 24 + t] = acc;
        }
    }

    var sc = try StreamingConv1d.init(allocator, &cw, 1);
    defer allocator.free(sc.prev);
    var got: [3 * 24]f32 = undefined;
    var t0: usize = 0;
    for ([_]usize{ 6, 10, 8 }) |n| {
        var chunk: [2 * 10]f32 = undefined;
        for (0..2) |c| @memcpy(chunk[c * n ..][0..n], x[c * 24 + t0 ..][0..n]);
        var out_chunk: [3 * 10]f32 = undefined;
        try sc.run(allocator, chunk[0 .. 2 * n], n, out_chunk[0 .. 3 * n]);
        for (0..3) |c| @memcpy(got[c * 24 + t0 ..][0..n], out_chunk[c * n ..][0..n]);
        t0 += n;
    }
    for (ref, got) |r, g| try std.testing.expectApproxEqAbs(r, g, 1e-5);
}

test "streaming transposed conv chunked == whole-signal (bias-corrected OLA)" {
    const allocator = std.testing.allocator;
    // 2 in, 2 out, k=6, stride 3 (SEANet-style ratio); 12 input steps as 5+7.
    var w: [2 * 2 * 6]f32 = undefined;
    var b: [2]f32 = .{ 0.05, -0.07 };
    var x: [2 * 12]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(11);
    for (&w) |*v| v.* = prng.random().floatNorm(f32);
    for (&x) |*v| v.* = prng.random().floatNorm(f32);
    const cw = Weights{ .w = &w, .b = &b, .in_ch = 2, .out_ch = 2, .k = 6 };

    // whole-signal reference, emitting (T*stride) samples (tail cropped)
    const full = (12 - 1) * 3 + 6;
    var acc = [_]f32{0} ** (2 * full);
    for (0..2) |ic| {
        for (0..2) |oc| {
            for (0..12) |t| {
                for (0..6) |kk| acc[oc * full + t * 3 + kk] += x[ic * 12 + t] * w[(ic * 2 + oc) * 6 + kk];
            }
        }
    }
    var ref: [2 * 36]f32 = undefined;
    for (0..2) |oc| {
        for (0..36) |t| ref[oc * 36 + t] = acc[oc * full + t] + b[oc];
    }

    var sc = try StreamingConvTranspose1d.init(allocator, &cw, 3, false);
    defer allocator.free(sc.partial);
    var got: [2 * 36]f32 = undefined;
    var t0: usize = 0;
    for ([_]usize{ 5, 7 }) |n| {
        var chunk: [2 * 7]f32 = undefined;
        for (0..2) |c| @memcpy(chunk[c * n ..][0..n], x[c * 12 + t0 ..][0..n]);
        var out_chunk: [2 * 21]f32 = undefined;
        try sc.run(allocator, chunk[0 .. 2 * n], n, out_chunk[0 .. 2 * n * 3]);
        for (0..2) |c| @memcpy(got[c * 36 + t0 * 3 ..][0 .. n * 3], out_chunk[c * n * 3 ..][0 .. n * 3]);
        t0 += n;
    }
    for (ref, got) |r, g| try std.testing.expectApproxEqAbs(r, g, 1e-5);
}
