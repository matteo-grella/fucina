//! Winograd F(2×2, 3×3) transform kernels for the channel-last conv2d fast
//! path (Lavin & Gray, arXiv:1509.09308). The exec route (`exec/conv.zig`)
//! decomposes an eligible 3×3 stride-1 conv into:
//!
//!   U = G·g·Gᵀ   (weight transform, per (oc, ic) — 16 coefficient planes)
//!   V = Bᵀ·d·B   (input transform, per 4×4 input tile → 2×2 output tile)
//!   M_e = V_e · U_eᵀ   (16 independent [tiles×Cin]×[Cout×Cin]ᵀ GEMMs,
//!                       dispatched through the ordinary matmul path)
//!   y = Aᵀ·M·A + bias   (output transform; bias is ONE add per element,
//!                        value-identical to the post-GEMM bias pass)
//!
//! ~2.25× fewer MACs than direct/im2col 3×3 and no 9× col-matrix traffic.
//! Numerics: the transforms reassociate the 3×3 reduction (adds/subs plus
//! exact ·0.5 in G), so results differ from the direct kernel at the
//! ~1e-6-relative level. `FUCINA_NO_WINOGRAD=1` reverts to im2col at
//! runtime.
//!
//! Channel-last layouts: x `[H,W,Cin]`, w `[Cout,3,3,Cin]`, y `[OH,OW,Cout]`;
//! every transform is vectorized over the contiguous channel axis and
//! parallelized over disjoint row ranges (bit-identical to serial).

const std = @import("std");
const parallel = @import("../../parallel.zig");
const vm = @import("common.zig");

const ParallelConfig = vm.ParallelConfig;
const vector_len = vm.vector_len;

/// Geometry shared by the transforms. `pad_h/pad_w` ∈ {0, 1}; `tiles_y/x`
/// cover ceil(oh/2) × ceil(ow/2) output tiles.
pub const F2Dims = struct {
    h: usize,
    w: usize,
    cin: usize,
    oh: usize,
    ow: usize,
    cout: usize,
    pad_h: usize,
    pad_w: usize,
    tiles_y: usize,
    tiles_x: usize,
};

// --- weight transform -------------------------------------------------------

const WeightTask = struct {
    u: *const [16][]f32,
    w: []const f32,
    cout: usize,
    cin: usize,
    oc_start: usize,
    oc_end: usize,
};

fn runWeightTask(task: *const WeightTask) void {
    weightTransformRange(task.u, task.w, task.cout, task.cin, task.oc_start, task.oc_end);
}

/// U = G·g·Gᵀ per (oc, channel): `w[(oc·3+ky)·3+kx)·cin + ic]` →
/// `u[e][oc·cin + ic]`, e = 4·i + j. Parallel over output channels.
pub fn f2WeightTransformIntoWithConfig(u: *const [16][]f32, w: []const f32, cout: usize, cin: usize, config: ParallelConfig) void {
    if (config.pool) |pool| {
        const tc = vm.generalConvThreadCount(cout, 16 * cout * cin);
        if (tc > 1) {
            var tasks: [parallel.vector_max_threads]WeightTask = undefined;
            for (0..tc) |ti| {
                tasks[ti] = .{ .u = u, .w = w, .cout = cout, .cin = cin, .oc_start = ti * cout / tc, .oc_end = (ti + 1) * cout / tc };
            }
            pool.parallelChunks(WeightTask, tasks[0..tc], runWeightTask);
            return;
        }
    }
    weightTransformRange(u, w, cout, cin, 0, cout);
}

fn weightTransformRange(u: *const [16][]f32, w: []const f32, cout: usize, cin: usize, oc_start: usize, oc_end: usize) void {
    _ = cout;
    var oc = oc_start;
    while (oc < oc_end) : (oc += 1) {
        var ic: usize = 0;
        while (ic + vector_len <= cin) : (ic += vector_len) {
            weightTileChunk(vector_len, u, w, cin, oc, ic);
        }
        while (ic < cin) : (ic += 1) {
            weightTileChunk(1, u, w, cin, oc, ic);
        }
    }
}

inline fn weightTileChunk(comptime L: usize, u: *const [16][]f32, w: []const f32, cin: usize, oc: usize, ic: usize) void {
    const V = @Vector(L, f32);
    const half: V = @splat(0.5);
    var g: [3][3]V = undefined;
    inline for (0..3) |ky| {
        inline for (0..3) |kx| {
            g[ky][kx] = w[((oc * 3 + ky) * 3 + kx) * cin + ic ..][0..L].*;
        }
    }
    // rows: t = G·g (4×3)
    var t: [4][3]V = undefined;
    inline for (0..3) |k| {
        t[0][k] = g[0][k];
        t[1][k] = (g[0][k] + g[1][k] + g[2][k]) * half;
        t[2][k] = (g[0][k] - g[1][k] + g[2][k]) * half;
        t[3][k] = g[2][k];
    }
    // cols: U = t·Gᵀ (4×4)
    inline for (0..4) |i| {
        const c0 = t[i][0];
        const c1 = t[i][1];
        const c2 = t[i][2];
        const dst = oc * cin + ic;
        u[i * 4 + 0][dst..][0..L].* = c0;
        u[i * 4 + 1][dst..][0..L].* = (c0 + c1 + c2) * half;
        u[i * 4 + 2][dst..][0..L].* = (c0 - c1 + c2) * half;
        u[i * 4 + 3][dst..][0..L].* = c2;
    }
}

// --- input transform --------------------------------------------------------

const InputTask = struct {
    v: *const [16][]f32,
    x: []const f32,
    d: F2Dims,
    ty_start: usize,
    ty_end: usize,
};

fn runInputTask(task: *const InputTask) void {
    inputTransformRange(task.v, task.x, task.d, task.ty_start, task.ty_end);
}

/// V = Bᵀ·d·B per 4×4 input tile (top-left at `(2ty − pad, 2tx − pad)`,
/// out-of-range taps read as zero): `v[e][(ty·tiles_x+tx)·cin + ic]`.
/// Parallel over tile rows.
pub fn f2InputTransformIntoWithConfig(v: *const [16][]f32, x: []const f32, d: F2Dims, config: ParallelConfig) void {
    if (config.pool) |pool| {
        const work = 16 * d.tiles_y * d.tiles_x * d.cin;
        const tc = vm.generalConvThreadCount(d.tiles_y, work);
        if (tc > 1) {
            var tasks: [parallel.vector_max_threads]InputTask = undefined;
            for (0..tc) |ti| {
                tasks[ti] = .{ .v = v, .x = x, .d = d, .ty_start = ti * d.tiles_y / tc, .ty_end = (ti + 1) * d.tiles_y / tc };
            }
            pool.parallelChunks(InputTask, tasks[0..tc], runInputTask);
            return;
        }
    }
    inputTransformRange(v, x, d, 0, d.tiles_y);
}

fn inputTransformRange(v: *const [16][]f32, x: []const f32, d: F2Dims, ty_start: usize, ty_end: usize) void {
    var ty = ty_start;
    while (ty < ty_end) : (ty += 1) {
        const iy0 = @as(isize, @intCast(2 * ty)) - @as(isize, @intCast(d.pad_h));
        var tx: usize = 0;
        while (tx < d.tiles_x) : (tx += 1) {
            const ix0 = @as(isize, @intCast(2 * tx)) - @as(isize, @intCast(d.pad_w));
            const tile = ty * d.tiles_x + tx;
            const interior = iy0 >= 0 and ix0 >= 0 and iy0 + 4 <= @as(isize, @intCast(d.h)) and ix0 + 4 <= @as(isize, @intCast(d.w));
            var ic: usize = 0;
            while (ic + vector_len <= d.cin) : (ic += vector_len) {
                inputTileChunk(vector_len, v, x, d, tile, iy0, ix0, ic, interior);
            }
            while (ic < d.cin) : (ic += 1) {
                inputTileChunk(1, v, x, d, tile, iy0, ix0, ic, interior);
            }
        }
    }
}

inline fn inputTileChunk(comptime L: usize, v: *const [16][]f32, x: []const f32, d: F2Dims, tile: usize, iy0: isize, ix0: isize, ic: usize, interior: bool) void {
    const V = @Vector(L, f32);
    var dd: [4][4]V = undefined;
    if (interior) {
        const base: usize = @intCast((iy0 * @as(isize, @intCast(d.w)) + ix0));
        inline for (0..4) |i| {
            inline for (0..4) |j| {
                dd[i][j] = x[(base + i * d.w + j) * d.cin + ic ..][0..L].*;
            }
        }
    } else {
        inline for (0..4) |i| {
            const iy = iy0 + @as(isize, i);
            inline for (0..4) |j| {
                const ix = ix0 + @as(isize, j);
                dd[i][j] = if (iy >= 0 and ix >= 0 and iy < @as(isize, @intCast(d.h)) and ix < @as(isize, @intCast(d.w)))
                    x[(@as(usize, @intCast(iy)) * d.w + @as(usize, @intCast(ix))) * d.cin + ic ..][0..L].*
                else
                    @as(V, @splat(0));
            }
        }
    }
    // rows: t = Bᵀ·d
    var t: [4][4]V = undefined;
    inline for (0..4) |j| {
        t[0][j] = dd[0][j] - dd[2][j];
        t[1][j] = dd[1][j] + dd[2][j];
        t[2][j] = dd[2][j] - dd[1][j];
        t[3][j] = dd[1][j] - dd[3][j];
    }
    // cols: V = t·B
    const dst = tile * d.cin + ic;
    inline for (0..4) |i| {
        v[i * 4 + 0][dst..][0..L].* = t[i][0] - t[i][2];
        v[i * 4 + 1][dst..][0..L].* = t[i][1] + t[i][2];
        v[i * 4 + 2][dst..][0..L].* = t[i][2] - t[i][1];
        v[i * 4 + 3][dst..][0..L].* = t[i][1] - t[i][3];
    }
}

// --- output transform -------------------------------------------------------

const OutputTask = struct {
    y: []f32,
    m: *const [16][]const f32,
    bias: ?[]const f32,
    fuse_relu: bool,
    d: F2Dims,
    ty_start: usize,
    ty_end: usize,
};

fn runOutputTask(task: *const OutputTask) void {
    if (task.fuse_relu) {
        outputTransformRange(true, task.y, task.m, task.bias, task.d, task.ty_start, task.ty_end);
    } else {
        outputTransformRange(false, task.y, task.m, task.bias, task.d, task.ty_start, task.ty_end);
    }
}

/// y = Aᵀ·M·A (+ bias, then optional fused relu — the same single relu the
/// caller would apply, evaluated on identical values) per tile; writes only
/// the valid output positions (`2ty+r < oh`, `2tx+s < ow`). Parallel over
/// tile rows (disjoint `y` rows).
pub fn f2OutputTransformIntoWithConfig(y: []f32, m: *const [16][]const f32, bias: ?[]const f32, fuse_relu: bool, d: F2Dims, config: ParallelConfig) void {
    if (config.pool) |pool| {
        const work = 16 * d.tiles_y * d.tiles_x * d.cout;
        const tc = vm.generalConvThreadCount(d.tiles_y, work);
        if (tc > 1) {
            var tasks: [parallel.vector_max_threads]OutputTask = undefined;
            for (0..tc) |ti| {
                tasks[ti] = .{ .y = y, .m = m, .bias = bias, .fuse_relu = fuse_relu, .d = d, .ty_start = ti * d.tiles_y / tc, .ty_end = (ti + 1) * d.tiles_y / tc };
            }
            pool.parallelChunks(OutputTask, tasks[0..tc], runOutputTask);
            return;
        }
    }
    if (fuse_relu) {
        outputTransformRange(true, y, m, bias, d, 0, d.tiles_y);
    } else {
        outputTransformRange(false, y, m, bias, d, 0, d.tiles_y);
    }
}

fn outputTransformRange(comptime fuse_relu: bool, y: []f32, m: *const [16][]const f32, bias: ?[]const f32, d: F2Dims, ty_start: usize, ty_end: usize) void {
    var ty = ty_start;
    while (ty < ty_end) : (ty += 1) {
        var tx: usize = 0;
        while (tx < d.tiles_x) : (tx += 1) {
            const tile = ty * d.tiles_x + tx;
            var oc: usize = 0;
            while (oc + vector_len <= d.cout) : (oc += vector_len) {
                outputTileChunk(fuse_relu, vector_len, y, m, bias, d, ty, tx, tile, oc);
            }
            while (oc < d.cout) : (oc += 1) {
                outputTileChunk(fuse_relu, 1, y, m, bias, d, ty, tx, tile, oc);
            }
        }
    }
}

inline fn outputTileChunk(comptime fuse_relu: bool, comptime L: usize, y: []f32, m: *const [16][]const f32, bias: ?[]const f32, d: F2Dims, ty: usize, tx: usize, tile: usize, oc: usize) void {
    const V = @Vector(L, f32);
    const src = tile * d.cout + oc;
    var mm: [4][4]V = undefined;
    inline for (0..4) |i| {
        inline for (0..4) |j| {
            mm[i][j] = m[i * 4 + j][src..][0..L].*;
        }
    }
    // rows: r = Aᵀ·M (2×4)
    var r: [2][4]V = undefined;
    inline for (0..4) |j| {
        r[0][j] = mm[0][j] + mm[1][j] + mm[2][j];
        r[1][j] = mm[1][j] - mm[2][j] - mm[3][j];
    }
    // cols: y = r·A (2×2), plus bias.
    const bv: V = if (bias) |b| b[oc..][0..L].* else @splat(0);
    const vzero: V = @splat(0);
    var out: [2][2]V = undefined;
    inline for (0..2) |i| {
        out[i][0] = r[i][0] + r[i][1] + r[i][2] + bv;
        out[i][1] = r[i][1] - r[i][2] - r[i][3] + bv;
        if (fuse_relu) {
            out[i][0] = @max(out[i][0], vzero);
            out[i][1] = @max(out[i][1], vzero);
        }
    }
    inline for (0..2) |i| {
        const oy = 2 * ty + i;
        if (oy < d.oh) {
            inline for (0..2) |j| {
                const ox = 2 * tx + j;
                if (ox < d.ow) {
                    y[(oy * d.ow + ox) * d.cout + oc ..][0..L].* = out[i][j];
                }
            }
        }
    }
}

// ===========================================================================
// F(4×4, 3×3) — 6×6 input tiles → 4×4 output tiles, 36 coefficient planes.
// 4× fewer MACs than direct (vs F2's 2.25×); the G/Bᵀ/Aᵀ entries include
// non-dyadic fractions (1/6, 1/12, 1/24), so the drift envelope is wider
// than F2 (~1e-5-relative). Selected by the exec route for large spatial
// maps (min(oh,ow) ≥ 14, mirroring the reference's gate);
// FUCINA_NO_WINOGRAD_F4=1 pins those back to F2.
// Reuses `F2Dims` with tiles covering ceil(oh/4) × ceil(ow/4).
// ===========================================================================

const F4WeightTask = struct {
    u: *const [36][]f32,
    w: []const f32,
    cin: usize,
    oc_start: usize,
    oc_end: usize,
};

fn runF4WeightTask(task: *const F4WeightTask) void {
    f4WeightTransformRange(task.u, task.w, task.cin, task.oc_start, task.oc_end);
}

pub fn f4WeightTransformIntoWithConfig(u: *const [36][]f32, w: []const f32, cout: usize, cin: usize, config: ParallelConfig) void {
    if (config.pool) |pool| {
        const tc = vm.generalConvThreadCount(cout, 36 * cout * cin);
        if (tc > 1) {
            var tasks: [parallel.vector_max_threads]F4WeightTask = undefined;
            for (0..tc) |ti| {
                tasks[ti] = .{ .u = u, .w = w, .cin = cin, .oc_start = ti * cout / tc, .oc_end = (ti + 1) * cout / tc };
            }
            pool.parallelChunks(F4WeightTask, tasks[0..tc], runF4WeightTask);
            return;
        }
    }
    f4WeightTransformRange(u, w, cin, 0, cout);
}

fn f4WeightTransformRange(u: *const [36][]f32, w: []const f32, cin: usize, oc_start: usize, oc_end: usize) void {
    var oc = oc_start;
    while (oc < oc_end) : (oc += 1) {
        var ic: usize = 0;
        while (ic + vector_len <= cin) : (ic += vector_len) {
            f4WeightTileChunk(vector_len, u, w, cin, oc, ic);
        }
        while (ic < cin) : (ic += 1) {
            f4WeightTileChunk(1, u, w, cin, oc, ic);
        }
    }
}

/// One row (or column) of the F4 kernel transform: G·[a,b,c]ᵀ with
/// G = [¼,0,0; −⅙,−⅙,−⅙; −⅙,⅙,−⅙; 1/24,1/12,⅙; 1/24,−1/12,⅙; 0,0,1].
inline fn f4GRow(comptime L: usize, a: @Vector(L, f32), b: @Vector(L, f32), c: @Vector(L, f32)) [6]@Vector(L, f32) {
    const V = @Vector(L, f32);
    const q: V = @splat(0.25);
    const s6: V = @splat(-1.0 / 6.0);
    const s24: V = @splat(1.0 / 24.0);
    const s12: V = @splat(1.0 / 12.0);
    const s3: V = @splat(1.0 / 6.0);
    return .{
        a * q,
        (a + b + c) * s6,
        (a - b + c) * s6,
        a * s24 + b * s12 + c * s3,
        a * s24 - b * s12 + c * s3,
        c,
    };
}

inline fn f4WeightTileChunk(comptime L: usize, u: *const [36][]f32, w: []const f32, cin: usize, oc: usize, ic: usize) void {
    const V = @Vector(L, f32);
    var g: [3][3]V = undefined;
    inline for (0..3) |ky| {
        inline for (0..3) |kx| {
            g[ky][kx] = w[((oc * 3 + ky) * 3 + kx) * cin + ic ..][0..L].*;
        }
    }
    // rows: t = G·g (6×3)
    var t: [6][3]V = undefined;
    inline for (0..3) |k| {
        const col = f4GRow(L, g[0][k], g[1][k], g[2][k]);
        inline for (0..6) |i| t[i][k] = col[i];
    }
    // cols: U = t·Gᵀ (6×6)
    const dst = oc * cin + ic;
    inline for (0..6) |i| {
        const row = f4GRow(L, t[i][0], t[i][1], t[i][2]);
        inline for (0..6) |j| {
            u[i * 6 + j][dst..][0..L].* = row[j];
        }
    }
}

const F4InputTask = struct {
    v: *const [36][]f32,
    x: []const f32,
    d: F2Dims,
    ty_start: usize,
    ty_end: usize,
};

fn runF4InputTask(task: *const F4InputTask) void {
    f4InputTransformRange(task.v, task.x, task.d, task.ty_start, task.ty_end);
}

pub fn f4InputTransformIntoWithConfig(v: *const [36][]f32, x: []const f32, d: F2Dims, config: ParallelConfig) void {
    if (config.pool) |pool| {
        const work = 36 * d.tiles_y * d.tiles_x * d.cin;
        const tc = vm.generalConvThreadCount(d.tiles_y, work);
        if (tc > 1) {
            var tasks: [parallel.vector_max_threads]F4InputTask = undefined;
            for (0..tc) |ti| {
                tasks[ti] = .{ .v = v, .x = x, .d = d, .ty_start = ti * d.tiles_y / tc, .ty_end = (ti + 1) * d.tiles_y / tc };
            }
            pool.parallelChunks(F4InputTask, tasks[0..tc], runF4InputTask);
            return;
        }
    }
    f4InputTransformRange(v, x, d, 0, d.tiles_y);
}

fn f4InputTransformRange(v: *const [36][]f32, x: []const f32, d: F2Dims, ty_start: usize, ty_end: usize) void {
    var ty = ty_start;
    while (ty < ty_end) : (ty += 1) {
        const iy0 = @as(isize, @intCast(4 * ty)) - @as(isize, @intCast(d.pad_h));
        var tx: usize = 0;
        while (tx < d.tiles_x) : (tx += 1) {
            const ix0 = @as(isize, @intCast(4 * tx)) - @as(isize, @intCast(d.pad_w));
            const tile = ty * d.tiles_x + tx;
            const interior = iy0 >= 0 and ix0 >= 0 and iy0 + 6 <= @as(isize, @intCast(d.h)) and ix0 + 6 <= @as(isize, @intCast(d.w));
            var ic: usize = 0;
            while (ic + vector_len <= d.cin) : (ic += vector_len) {
                f4InputTileChunk(vector_len, v, x, d, tile, iy0, ix0, ic, interior);
            }
            while (ic < d.cin) : (ic += 1) {
                f4InputTileChunk(1, v, x, d, tile, iy0, ix0, ic, interior);
            }
        }
    }
}

/// One row (or column) of the F4 data transform: Bᵀ·[d0..d5]ᵀ with the
/// standard F(4×4,3×3) Bᵀ (rows 4,0,−5,0,1,0 / 0,∓4,−4,±1,1,0 /
/// 0,∓2,−1,±2,1,0 / 0,4,0,−5,0,1).
inline fn f4BRow(comptime L: usize, d0: @Vector(L, f32), d1: @Vector(L, f32), d2: @Vector(L, f32), d3: @Vector(L, f32), d4: @Vector(L, f32), d5: @Vector(L, f32)) [6]@Vector(L, f32) {
    const V = @Vector(L, f32);
    const four: V = @splat(4.0);
    const five: V = @splat(5.0);
    const two: V = @splat(2.0);
    return .{
        four * d0 - five * d2 + d4,
        -four * d1 - four * d2 + d3 + d4,
        four * d1 - four * d2 - d3 + d4,
        -two * d1 - d2 + two * d3 + d4,
        two * d1 - d2 - two * d3 + d4,
        four * d1 - five * d3 + d5,
    };
}

inline fn f4InputTileChunk(comptime L: usize, v: *const [36][]f32, x: []const f32, d: F2Dims, tile: usize, iy0: isize, ix0: isize, ic: usize, interior: bool) void {
    const V = @Vector(L, f32);
    var dd: [6][6]V = undefined;
    if (interior) {
        const base: usize = @intCast((iy0 * @as(isize, @intCast(d.w)) + ix0));
        inline for (0..6) |i| {
            inline for (0..6) |j| {
                dd[i][j] = x[(base + i * d.w + j) * d.cin + ic ..][0..L].*;
            }
        }
    } else {
        inline for (0..6) |i| {
            const iy = iy0 + @as(isize, i);
            inline for (0..6) |j| {
                const ix = ix0 + @as(isize, j);
                dd[i][j] = if (iy >= 0 and ix >= 0 and iy < @as(isize, @intCast(d.h)) and ix < @as(isize, @intCast(d.w)))
                    x[(@as(usize, @intCast(iy)) * d.w + @as(usize, @intCast(ix))) * d.cin + ic ..][0..L].*
                else
                    @as(V, @splat(0));
            }
        }
    }
    // rows: t = Bᵀ·d
    var t: [6][6]V = undefined;
    inline for (0..6) |j| {
        const col = f4BRow(L, dd[0][j], dd[1][j], dd[2][j], dd[3][j], dd[4][j], dd[5][j]);
        inline for (0..6) |i| t[i][j] = col[i];
    }
    // cols: V = t·B
    const dst = tile * d.cin + ic;
    inline for (0..6) |i| {
        const row = f4BRow(L, t[i][0], t[i][1], t[i][2], t[i][3], t[i][4], t[i][5]);
        inline for (0..6) |j| {
            v[i * 6 + j][dst..][0..L].* = row[j];
        }
    }
}

const F4OutputTask = struct {
    y: []f32,
    m: *const [36][]const f32,
    bias: ?[]const f32,
    fuse_relu: bool,
    d: F2Dims,
    ty_start: usize,
    ty_end: usize,
};

fn runF4OutputTask(task: *const F4OutputTask) void {
    if (task.fuse_relu) {
        f4OutputTransformRange(true, task.y, task.m, task.bias, task.d, task.ty_start, task.ty_end);
    } else {
        f4OutputTransformRange(false, task.y, task.m, task.bias, task.d, task.ty_start, task.ty_end);
    }
}

pub fn f4OutputTransformIntoWithConfig(y: []f32, m: *const [36][]const f32, bias: ?[]const f32, fuse_relu: bool, d: F2Dims, config: ParallelConfig) void {
    if (config.pool) |pool| {
        const work = 36 * d.tiles_y * d.tiles_x * d.cout;
        const tc = vm.generalConvThreadCount(d.tiles_y, work);
        if (tc > 1) {
            var tasks: [parallel.vector_max_threads]F4OutputTask = undefined;
            for (0..tc) |ti| {
                tasks[ti] = .{ .y = y, .m = m, .bias = bias, .fuse_relu = fuse_relu, .d = d, .ty_start = ti * d.tiles_y / tc, .ty_end = (ti + 1) * d.tiles_y / tc };
            }
            pool.parallelChunks(F4OutputTask, tasks[0..tc], runF4OutputTask);
            return;
        }
    }
    if (fuse_relu) {
        f4OutputTransformRange(true, y, m, bias, d, 0, d.tiles_y);
    } else {
        f4OutputTransformRange(false, y, m, bias, d, 0, d.tiles_y);
    }
}

fn f4OutputTransformRange(comptime fuse_relu: bool, y: []f32, m: *const [36][]const f32, bias: ?[]const f32, d: F2Dims, ty_start: usize, ty_end: usize) void {
    var ty = ty_start;
    while (ty < ty_end) : (ty += 1) {
        var tx: usize = 0;
        while (tx < d.tiles_x) : (tx += 1) {
            const tile = ty * d.tiles_x + tx;
            var oc: usize = 0;
            while (oc + vector_len <= d.cout) : (oc += vector_len) {
                f4OutputTileChunk(fuse_relu, vector_len, y, m, bias, d, ty, tx, tile, oc);
            }
            while (oc < d.cout) : (oc += 1) {
                f4OutputTileChunk(fuse_relu, 1, y, m, bias, d, ty, tx, tile, oc);
            }
        }
    }
}

/// One row (or column) of the F4 output transform: Aᵀ·[m0..m5]ᵀ with
/// Aᵀ = [1,1,1,1,1,0; 0,1,−1,2,−2,0; 0,1,1,4,4,0; 0,1,−1,8,−8,1].
inline fn f4ARow(comptime L: usize, m0: @Vector(L, f32), m1: @Vector(L, f32), m2: @Vector(L, f32), m3: @Vector(L, f32), m4: @Vector(L, f32), m5: @Vector(L, f32)) [4]@Vector(L, f32) {
    const V = @Vector(L, f32);
    const two: V = @splat(2.0);
    const four: V = @splat(4.0);
    const eight: V = @splat(8.0);
    return .{
        m0 + m1 + m2 + m3 + m4,
        m1 - m2 + two * m3 - two * m4,
        m1 + m2 + four * m3 + four * m4,
        m1 - m2 + eight * m3 - eight * m4 + m5,
    };
}

inline fn f4OutputTileChunk(comptime fuse_relu: bool, comptime L: usize, y: []f32, m: *const [36][]const f32, bias: ?[]const f32, d: F2Dims, ty: usize, tx: usize, tile: usize, oc: usize) void {
    const V = @Vector(L, f32);
    const src = tile * d.cout + oc;
    var mm: [6][6]V = undefined;
    inline for (0..6) |i| {
        inline for (0..6) |j| {
            mm[i][j] = m[i * 6 + j][src..][0..L].*;
        }
    }
    // rows: r = Aᵀ·M (4×6)
    var r: [4][6]V = undefined;
    inline for (0..6) |j| {
        const col = f4ARow(L, mm[0][j], mm[1][j], mm[2][j], mm[3][j], mm[4][j], mm[5][j]);
        inline for (0..4) |i| r[i][j] = col[i];
    }
    // cols: y = r·A (4×4), plus bias.
    const bv: V = if (bias) |b| b[oc..][0..L].* else @splat(0);
    const vzero: V = @splat(0);
    inline for (0..4) |i| {
        const row = f4ARow(L, r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5]);
        const oy = 4 * ty + i;
        if (oy < d.oh) {
            inline for (0..4) |j| {
                const ox = 4 * tx + j;
                if (ox < d.ow) {
                    const val = row[j] + bv;
                    y[(oy * d.ow + ox) * d.cout + oc ..][0..L].* = if (fuse_relu) @max(val, vzero) else val;
                }
            }
        }
    }
}
