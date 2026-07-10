//! Channel-last 2-D pooling and nearest-neighbour upsampling kernels.
//!
//! Layout matches the channel-last conv2d family: input `[H, W, C]` contiguous,
//! output `[OH, OW, C]` with `OH = (H + 2*pad_h - KH)/stride_h + 1` (likewise
//! `OW`). The channel axis is innermost, so every window step is a contiguous
//! `C`-wide vector op — the layout the SCRFD/ArcFace ports run.
//!
//! Semantics:
//!   .max — out-of-range taps are skipped (identical to a −inf border, the
//!          ONNX MaxPool convention).
//!   .avg — mean over the VALID taps only (count excludes padding; the ONNX
//!          AveragePool `count_include_pad=0` default).
//!   .sum — sum over the valid taps (the upsample2x VJP; not exposed publicly).
//!
//! Forward parallelizes over output rows — each `oh` writes a disjoint output
//! range and reads immutably, so the threaded result is bit-identical to the
//! serial path. Backward kernels are correctness-first serial scatters
//! (gradient path only), matching the conv2d backward convention.

const std = @import("std");
const parallel = @import("../../parallel.zig");
const tensor = @import("../../tensor.zig");
const vm = @import("common.zig");

const Tensor = tensor.Tensor;
const ParallelConfig = vm.ParallelConfig;
const Vf32 = vm.Vf32;
const vector_len = vm.vector_len;

pub const PoolKind = enum { avg, max, sum };

/// pool2d geometry (channel-last `[H,W,C]` → `[OH,OW,C]`).
pub const Pool2dDims = struct {
    h: usize,
    w: usize,
    c: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
};

const Pool2dTask = struct {
    out: []f32,
    in: []const f32,
    kind: PoolKind,
    d: Pool2dDims,
    oh_start: usize,
    oh_end: usize,
};

fn runPool2dTask(task: *const Pool2dTask) void {
    switch (task.kind) {
        inline else => |k| pool2dRangeRows(k, task.out, task.in, task.d, task.oh_start, task.oh_end),
    }
}

pub fn pool2dIntoWithConfig(
    comptime kind: PoolKind,
    out: *Tensor,
    input: *const Tensor,
    d: Pool2dDims,
    config: ParallelConfig,
) void {
    const o = out.data();
    const in = input.dataConst();
    if (config.pool) |pool| {
        const work = d.oh * d.ow * d.c * d.kh * d.kw;
        const tc = vm.generalConvThreadCount(d.oh, work);
        if (tc > 1) {
            var tasks: [parallel.vector_max_threads]Pool2dTask = undefined;
            for (0..tc) |ti| {
                tasks[ti] = .{
                    .out = o,
                    .in = in,
                    .kind = kind,
                    .d = d,
                    .oh_start = ti * d.oh / tc,
                    .oh_end = (ti + 1) * d.oh / tc,
                };
            }
            pool.parallelChunks(Pool2dTask, tasks[0..tc], runPool2dTask);
            return;
        }
    }
    pool2dRangeRows(kind, o, in, d, 0, d.oh);
}

/// Compute output rows `[oh_start, oh_end)` — the per-worker range. Window
/// accumulation runs vectorized over the contiguous channel axis; `.avg`
/// divides by the valid-tap count, `.max` starts at −inf so a fully padded
/// window yields −inf (the ONNX border value).
fn pool2dRangeRows(comptime kind: PoolKind, out: []f32, in: []const f32, d: Pool2dDims, oh_start: usize, oh_end: usize) void {
    const c = d.c;
    var oh: usize = oh_start;
    while (oh < oh_end) : (oh += 1) {
        var ow: usize = 0;
        while (ow < d.ow) : (ow += 1) {
            const out_base = (oh * d.ow + ow) * c;
            const init_val: f32 = if (kind == .max) -std.math.inf(f32) else 0;
            @memset(out[out_base..][0..c], init_val);
            var count: usize = 0;
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
                    count += 1;
                    accumulateChannels(kind, out[out_base..][0..c], in[(ih * d.w + iw) * c ..][0..c]);
                }
            }
            if (kind == .avg and count > 0) {
                scaleChannels(out[out_base..][0..c], 1.0 / @as(f32, @floatFromInt(count)));
            }
        }
    }
}

inline fn accumulateChannels(comptime kind: PoolKind, acc: []f32, x: []const f32) void {
    const n = acc.len;
    var i: usize = 0;
    while (i + vector_len <= n) : (i += vector_len) {
        const va: Vf32 = acc[i..][0..vector_len].*;
        const vx: Vf32 = x[i..][0..vector_len].*;
        acc[i..][0..vector_len].* = switch (kind) {
            .max => @max(va, vx),
            .avg, .sum => va + vx,
        };
    }
    while (i < n) : (i += 1) {
        acc[i] = switch (kind) {
            .max => @max(acc[i], x[i]),
            .avg, .sum => acc[i] + x[i],
        };
    }
}

inline fn scaleChannels(acc: []f32, s: f32) void {
    const n = acc.len;
    const vs: Vf32 = @splat(s);
    var i: usize = 0;
    while (i + vector_len <= n) : (i += vector_len) {
        const va: Vf32 = acc[i..][0..vector_len].*;
        acc[i..][0..vector_len].* = va * vs;
    }
    while (i < n) : (i += 1) acc[i] *= s;
}

/// avg-pool VJP: scatter `gy[oh,ow,c] / valid_count(oh,ow)` back over the
/// window's valid taps. `out` is `[H,W,C]`, zeroed here. Serial
/// (correctness-first; each input cell may receive from overlapping windows).
pub fn avgPool2dBackwardIntoWithConfig(out: *Tensor, gy: *const Tensor, d: Pool2dDims, config: ParallelConfig) void {
    _ = config;
    const gx = out.data();
    const g = gy.dataConst();
    @memset(gx, 0);
    const c = d.c;
    var oh: usize = 0;
    while (oh < d.oh) : (oh += 1) {
        var ow: usize = 0;
        while (ow < d.ow) : (ow += 1) {
            const gy_base = (oh * d.ow + ow) * c;
            var count: usize = 0;
            var kh: usize = 0;
            while (kh < d.kh) : (kh += 1) {
                const ih_s = @as(isize, @intCast(oh * d.stride_h + kh)) - @as(isize, @intCast(d.pad_h));
                if (ih_s < 0 or ih_s >= @as(isize, @intCast(d.h))) continue;
                var kw: usize = 0;
                while (kw < d.kw) : (kw += 1) {
                    const iw_s = @as(isize, @intCast(ow * d.stride_w + kw)) - @as(isize, @intCast(d.pad_w));
                    if (iw_s < 0 or iw_s >= @as(isize, @intCast(d.w))) continue;
                    count += 1;
                }
            }
            if (count == 0) continue;
            const inv: f32 = 1.0 / @as(f32, @floatFromInt(count));
            kh = 0;
            while (kh < d.kh) : (kh += 1) {
                const ih_s = @as(isize, @intCast(oh * d.stride_h + kh)) - @as(isize, @intCast(d.pad_h));
                if (ih_s < 0 or ih_s >= @as(isize, @intCast(d.h))) continue;
                const ih: usize = @intCast(ih_s);
                var kw: usize = 0;
                while (kw < d.kw) : (kw += 1) {
                    const iw_s = @as(isize, @intCast(ow * d.stride_w + kw)) - @as(isize, @intCast(d.pad_w));
                    if (iw_s < 0 or iw_s >= @as(isize, @intCast(d.w))) continue;
                    const iw: usize = @intCast(iw_s);
                    const gx_row = gx[(ih * d.w + iw) * c ..][0..c];
                    const gy_row = g[gy_base..][0..c];
                    for (gx_row, gy_row) |*a, b| a.* += b * inv;
                }
            }
        }
    }
}

/// max-pool VJP: route `gy[oh,ow,c]` to the window's argmax tap, first
/// occurrence in `(kh,kw)` scan order winning ties (recomputed from the saved
/// forward input — no index tensor is stored). `out` is `[H,W,C]`, zeroed
/// here. Serial (correctness-first).
pub fn maxPool2dBackwardIntoWithConfig(out: *Tensor, input: *const Tensor, gy: *const Tensor, d: Pool2dDims, config: ParallelConfig) void {
    _ = config;
    const gx = out.data();
    const in = input.dataConst();
    const g = gy.dataConst();
    @memset(gx, 0);
    const c = d.c;
    var oh: usize = 0;
    while (oh < d.oh) : (oh += 1) {
        var ow: usize = 0;
        while (ow < d.ow) : (ow += 1) {
            const gy_base = (oh * d.ow + ow) * c;
            var ci: usize = 0;
            while (ci < c) : (ci += 1) {
                var best: f32 = -std.math.inf(f32);
                var best_idx: ?usize = null;
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
                        const v = in[(ih * d.w + iw) * c + ci];
                        if (v > best) {
                            best = v;
                            best_idx = (ih * d.w + iw) * c + ci;
                        }
                    }
                }
                if (best_idx) |bi| gx[bi] += g[gy_base + ci];
            }
        }
    }
}

const Upsample2xTask = struct {
    out: []f32,
    in: []const f32,
    h: usize,
    w: usize,
    c: usize,
    ih_start: usize,
    ih_end: usize,
};

fn runUpsample2xTask(task: *const Upsample2xTask) void {
    upsample2xRangeRows(task.out, task.in, task.h, task.w, task.c, task.ih_start, task.ih_end);
}

/// 2× nearest-neighbour upsample: `out[2h+i, 2w+j, :] = in[h, w, :]`
/// (`i,j ∈ {0,1}`). One duplicated output row is built by widening each
/// channel block, then the sibling row is a single row `@memcpy`. Parallel
/// over input rows (disjoint output ranges — bit-identical to serial).
pub fn upsample2xNearestIntoWithConfig(out: *Tensor, input: *const Tensor, h: usize, w: usize, c: usize, config: ParallelConfig) void {
    const o = out.data();
    const in = input.dataConst();
    if (config.pool) |pool| {
        const tc = vm.generalConvThreadCount(h, 4 * h * w * c);
        if (tc > 1) {
            var tasks: [parallel.vector_max_threads]Upsample2xTask = undefined;
            for (0..tc) |ti| {
                tasks[ti] = .{
                    .out = o,
                    .in = in,
                    .h = h,
                    .w = w,
                    .c = c,
                    .ih_start = ti * h / tc,
                    .ih_end = (ti + 1) * h / tc,
                };
            }
            pool.parallelChunks(Upsample2xTask, tasks[0..tc], runUpsample2xTask);
            return;
        }
    }
    upsample2xRangeRows(o, in, h, w, c, 0, h);
}

fn upsample2xRangeRows(out: []f32, in: []const f32, h: usize, w: usize, c: usize, ih_start: usize, ih_end: usize) void {
    _ = h;
    const orow_len = 2 * w * c;
    var ih: usize = ih_start;
    while (ih < ih_end) : (ih += 1) {
        const irow = in[ih * w * c ..][0 .. w * c];
        const orow0 = out[(2 * ih) * orow_len ..][0..orow_len];
        var iw: usize = 0;
        while (iw < w) : (iw += 1) {
            const src = irow[iw * c ..][0..c];
            @memcpy(orow0[(2 * iw) * c ..][0..c], src);
            @memcpy(orow0[(2 * iw + 1) * c ..][0..c], src);
        }
        @memcpy(out[(2 * ih + 1) * orow_len ..][0..orow_len], orow0);
    }
}
