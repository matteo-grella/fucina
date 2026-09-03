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
const isa = @import("../isa.zig");
const tensor = @import("../../tensor.zig");
const common = @import("common.zig");
const tile = @import("tile.zig");

const Tensor = tensor.Tensor;

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

pub fn pool2dInto(
    pc: common.ParallelConfig,
    comptime kind: PoolKind,
    out: *Tensor,
    input: *const Tensor,
    d: Pool2dDims,
) void {
    if (comptime isa.reference) return scalar.pool2dInto(kind, out, input, d);
    const Ctx = struct { out: []f32, in: []const f32, d: Pool2dDims };
    const rows = struct {
        fn go(c: Ctx, oh_start: usize, oh_end: usize) void {
            pool2dRangeRows(kind, c.out, c.in, c.d, oh_start, oh_end);
        }
    };
    const ctx: Ctx = .{ .out = out.data(), .in = input.dataConst(), .d = d };
    if (pc.pool) |pool| {
        const tc = common.generalConvThreadCount(d.oh, d.oh * d.ow * d.c * d.kh * d.kw);
        if (tc > 1) return tile.forRange(pool, Ctx, ctx, d.oh, tc, rows.go);
    }
    rows.go(ctx, 0, d.oh);
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
    while (i + common.vector_len <= n) : (i += common.vector_len) {
        const va: common.Vf32 = acc[i..][0..common.vector_len].*;
        const vx: common.Vf32 = x[i..][0..common.vector_len].*;
        acc[i..][0..common.vector_len].* = switch (kind) {
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
    const vs: common.Vf32 = @splat(s);
    var i: usize = 0;
    while (i + common.vector_len <= n) : (i += common.vector_len) {
        const va: common.Vf32 = acc[i..][0..common.vector_len].*;
        acc[i..][0..common.vector_len].* = va * vs;
    }
    while (i < n) : (i += 1) acc[i] *= s;
}

/// avg-pool VJP: scatter `gy[oh,ow,c] / valid_count(oh,ow)` back over the
/// window's valid taps. `out` is `[H,W,C]`, zeroed here. Serial
/// (correctness-first; each input cell may receive from overlapping windows).
pub fn avgPool2dBackwardInto(out: *Tensor, gy: *const Tensor, d: Pool2dDims) void {
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
pub fn maxPool2dBackwardInto(out: *Tensor, input: *const Tensor, gy: *const Tensor, d: Pool2dDims) void {
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

/// 2× nearest-neighbour upsample: `out[2h+i, 2w+j, :] = in[h, w, :]`
/// (`i,j ∈ {0,1}`). One duplicated output row is built by widening each
/// channel block, then the sibling row is a single row `@memcpy`. Parallel
/// over input rows (disjoint output ranges — bit-identical to serial).
pub fn upsample2xNearestInto(pc: common.ParallelConfig, out: *Tensor, input: *const Tensor, h: usize, w: usize, c: usize) void {
    if (comptime isa.reference) return scalar.upsample2xNearestInto(out, input, h, w, c);
    const Ctx = struct { out: []f32, in: []const f32, w: usize, c: usize };
    const rows = struct {
        fn go(ctx: Ctx, ih_start: usize, ih_end: usize) void {
            upsample2xRangeRows(ctx.out, ctx.in, ctx.w, ctx.c, ih_start, ih_end);
        }
    };
    const ctx: Ctx = .{ .out = out.data(), .in = input.dataConst(), .w = w, .c = c };
    if (pc.pool) |pool| {
        const tc = common.generalConvThreadCount(h, 4 * h * w * c);
        if (tc > 1) return tile.forRange(pool, Ctx, ctx, h, tc, rows.go);
    }
    rows.go(ctx, 0, h);
}

fn upsample2xRangeRows(out: []f32, in: []const f32, w: usize, c: usize, ih_start: usize, ih_end: usize) void {
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

// ---------------- The scalar reference arms ----------------

/// The scalar reference twins: plain serial loops, no SIMD, no pool. On
/// `-Dbackend=scalar` builds (`isa.reference`) the forward entries above
/// dispatch here; the backward scatters are serial scalar loops already and
/// need no twin.
pub const scalar = struct {
    /// Scalar reference pool2d (independent of the vector kernel; layout and
    /// border semantics stated on `Pool2dDims` above).
    pub fn pool2dInto(comptime kind: PoolKind, out: *Tensor, input: *const Tensor, d: Pool2dDims) void {
        const o = out.data();
        const in = input.dataConst();
        for (0..d.oh) |oh| {
            for (0..d.ow) |ow| {
                for (0..d.c) |c| {
                    var acc: f32 = if (kind == .max) -std.math.inf(f32) else 0;
                    var count: usize = 0;
                    for (0..d.kh) |kh| {
                        const ih_i = @as(isize, @intCast(oh * d.stride_h + kh)) - @as(isize, @intCast(d.pad_h));
                        if (ih_i < 0 or ih_i >= @as(isize, @intCast(d.h))) continue;
                        for (0..d.kw) |kw| {
                            const iw_i = @as(isize, @intCast(ow * d.stride_w + kw)) - @as(isize, @intCast(d.pad_w));
                            if (iw_i < 0 or iw_i >= @as(isize, @intCast(d.w))) continue;
                            const v = in[(@as(usize, @intCast(ih_i)) * d.w + @as(usize, @intCast(iw_i))) * d.c + c];
                            switch (kind) {
                                .max => acc = @max(acc, v),
                                .avg, .sum => acc += v,
                            }
                            count += 1;
                        }
                    }
                    if (kind == .avg and count > 0) acc /= @floatFromInt(count);
                    o[(oh * d.ow + ow) * d.c + c] = acc;
                }
            }
        }
    }

    /// Scalar reference 2x nearest-neighbour upsample.
    pub fn upsample2xNearestInto(out: *Tensor, input: *const Tensor, h: usize, w: usize, c: usize) void {
        const o = out.data();
        const in = input.dataConst();
        for (0..2 * h) |oy| {
            for (0..2 * w) |ox| {
                for (0..c) |ci| {
                    o[(oy * 2 * w + ox) * c + ci] = in[((oy / 2) * w + ox / 2) * c + ci];
                }
            }
        }
    }
};
