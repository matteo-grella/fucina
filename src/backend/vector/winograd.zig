//! Winograd F(2×2, 3×3) and F(4×4, 3×3) transform kernels for the
//! channel-last conv2d fast path (Lavin & Gray, arXiv:1509.09308). The exec
//! route (`exec/conv.zig`) decomposes an eligible 3×3 stride-1 conv into:
//!
//!   U = G·g·Gᵀ   (weight transform, per (oc, ic) — one plane per tile cell)
//!   V = Bᵀ·d·B   (input transform, per input tile → output tile)
//!   M_e = V_e · U_eᵀ   (one [tiles×Cin]×[Cout×Cin]ᵀ GEMM per plane,
//!                       dispatched through the ordinary matmul path)
//!   y = Aᵀ·M·A + bias   (output transform; bias is ONE add per element,
//!                        value-identical to the post-GEMM bias pass)
//!
//! F2 maps 4×4 input tiles to 2×2 output tiles over 16 planes (~2.25× fewer
//! MACs than direct/im2col 3×3, no 9× col-matrix traffic; the transforms
//! reassociate the 3×3 reduction with adds/subs plus an exact ·0.5 in G, so
//! results differ from the direct kernel at the ~1e-6-relative level). F4
//! maps 6×6 tiles to 4×4 over 36 planes (4× fewer MACs; G/Bᵀ/Aᵀ carry
//! non-dyadic fractions, ~1e-5-relative drift), selected by the exec route
//! for large spatial maps (min(oh,ow) ≥ 14, mirroring the reference's
//! gate). `FUCINA_WINOGRAD=0` reverts to im2col at runtime,
//! `FUCINA_WINOGRAD_F4=0` pins F4 shapes back to F2.
//!
//! Both are the one `Winograd` module below over a tile size and its three
//! row transforms. Channel-last layouts: x `[H,W,Cin]`, w `[Cout,3,3,Cin]`,
//! y `[OH,OW,Cout]`; every transform is vectorized over the contiguous
//! channel axis and parallelized over disjoint row ranges (bit-identical
//! to serial).

const common = @import("common.zig");
const tile = @import("tile.zig");

const vector_len = common.vector_len;

/// Geometry shared by the transforms of both tile sizes. `pad_h/pad_w` ∈
/// {0, 1}; `tiles_y/x` cover ceil(oh/out_tile) × ceil(ow/out_tile) output
/// tiles.
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

/// The transform family over one tile size: `in_tile`² input taps map to
/// `out_tile`² outputs through the `Rows` row transforms (`gRow` = G·[3],
/// `bRow` = Bᵀ·[in_tile], `aRow` = Aᵀ·[in_tile]), each applied to the rows
/// and then the columns of a tile. Every entry splits its row range over
/// `tile.forRange` once `generalConvThreadCount` opens (`refSerial` keeps
/// the reference build serial).
fn Winograd(comptime in_tile: usize, comptime out_tile: usize, comptime Rows: type) type {
    return struct {
        const planes = in_tile * in_tile;
        const Planes = [planes][]f32;
        const PlanesConst = [planes][]const f32;
        const in_span: isize = @intCast(in_tile);

        fn dispatch(
            pc: common.ParallelConfig,
            comptime Ctx: type,
            ctx: Ctx,
            split: usize,
            work: usize,
            comptime rangeFn: fn (Ctx, usize, usize) void,
        ) void {
            if (common.refSerial(pc).pool) |pool| {
                const tc = common.generalConvThreadCount(split, work);
                if (tc > 1) return tile.forRange(pool, Ctx, ctx, split, tc, rangeFn);
            }
            rangeFn(ctx, 0, split);
        }

        // --- weight transform ---------------------------------------------

        const WeightCtx = struct { u: *const Planes, w: []const f32, cin: usize };

        /// U = G·g·Gᵀ per (oc, channel): `w[((oc·3+ky)·3+kx)·cin + ic]` →
        /// `u[e][oc·cin + ic]`, e = in_tile·i + j. Parallel over output
        /// channels.
        pub fn weightTransformInto(pc: common.ParallelConfig, u: *const Planes, w: []const f32, cout: usize, cin: usize) void {
            dispatch(pc, WeightCtx, .{ .u = u, .w = w, .cin = cin }, cout, planes * cout * cin, weightRange);
        }

        fn weightRange(c: WeightCtx, oc_start: usize, oc_end: usize) void {
            var oc = oc_start;
            while (oc < oc_end) : (oc += 1) {
                var ic: usize = 0;
                while (ic + vector_len <= c.cin) : (ic += vector_len) {
                    weightTile(vector_len, c.u, c.w, c.cin, oc, ic);
                }
                while (ic < c.cin) : (ic += 1) {
                    weightTile(1, c.u, c.w, c.cin, oc, ic);
                }
            }
        }

        inline fn weightTile(comptime L: usize, u: *const Planes, w: []const f32, cin: usize, oc: usize, ic: usize) void {
            const V = @Vector(L, f32);
            var g: [3][3]V = undefined;
            inline for (0..3) |ky| {
                inline for (0..3) |kx| {
                    g[ky][kx] = w[((oc * 3 + ky) * 3 + kx) * cin + ic ..][0..L].*;
                }
            }
            // rows: t = G·g (in_tile × 3)
            var t: [in_tile][3]V = undefined;
            inline for (0..3) |k| {
                const col = Rows.gRow(L, .{ g[0][k], g[1][k], g[2][k] });
                inline for (0..in_tile) |i| t[i][k] = col[i];
            }
            // cols: U = t·Gᵀ (in_tile × in_tile)
            const dst = oc * cin + ic;
            inline for (0..in_tile) |i| {
                const row = Rows.gRow(L, t[i]);
                inline for (0..in_tile) |j| {
                    u[i * in_tile + j][dst..][0..L].* = row[j];
                }
            }
        }

        // --- input transform ----------------------------------------------

        const InputCtx = struct { v: *const Planes, x: []const f32, d: F2Dims };

        /// V = Bᵀ·d·B per input tile (top-left at `(out_tile·ty − pad,
        /// out_tile·tx − pad)`, out-of-range taps read as zero):
        /// `v[e][(ty·tiles_x+tx)·cin + ic]`. Parallel over tile rows.
        pub fn inputTransformInto(pc: common.ParallelConfig, v: *const Planes, x: []const f32, d: F2Dims) void {
            dispatch(pc, InputCtx, .{ .v = v, .x = x, .d = d }, d.tiles_y, planes * d.tiles_y * d.tiles_x * d.cin, inputRange);
        }

        fn inputRange(c: InputCtx, ty_start: usize, ty_end: usize) void {
            const d = c.d;
            var ty = ty_start;
            while (ty < ty_end) : (ty += 1) {
                const iy0 = @as(isize, @intCast(out_tile * ty)) - @as(isize, @intCast(d.pad_h));
                var tx: usize = 0;
                while (tx < d.tiles_x) : (tx += 1) {
                    const ix0 = @as(isize, @intCast(out_tile * tx)) - @as(isize, @intCast(d.pad_w));
                    const tile_index = ty * d.tiles_x + tx;
                    const interior = iy0 >= 0 and ix0 >= 0 and iy0 + in_span <= @as(isize, @intCast(d.h)) and ix0 + in_span <= @as(isize, @intCast(d.w));
                    var ic: usize = 0;
                    while (ic + vector_len <= d.cin) : (ic += vector_len) {
                        inputTile(vector_len, c.v, c.x, d, tile_index, iy0, ix0, ic, interior);
                    }
                    while (ic < d.cin) : (ic += 1) {
                        inputTile(1, c.v, c.x, d, tile_index, iy0, ix0, ic, interior);
                    }
                }
            }
        }

        inline fn inputTile(comptime L: usize, v: *const Planes, x: []const f32, d: F2Dims, tile_index: usize, iy0: isize, ix0: isize, ic: usize, interior: bool) void {
            const V = @Vector(L, f32);
            var dd: [in_tile][in_tile]V = undefined;
            if (interior) {
                const base: usize = @intCast((iy0 * @as(isize, @intCast(d.w)) + ix0));
                inline for (0..in_tile) |i| {
                    inline for (0..in_tile) |j| {
                        dd[i][j] = x[(base + i * d.w + j) * d.cin + ic ..][0..L].*;
                    }
                }
            } else {
                inline for (0..in_tile) |i| {
                    const iy = iy0 + @as(isize, i);
                    inline for (0..in_tile) |j| {
                        const ix = ix0 + @as(isize, j);
                        dd[i][j] = if (iy >= 0 and ix >= 0 and iy < @as(isize, @intCast(d.h)) and ix < @as(isize, @intCast(d.w)))
                            x[(@as(usize, @intCast(iy)) * d.w + @as(usize, @intCast(ix))) * d.cin + ic ..][0..L].*
                        else
                            @as(V, @splat(0));
                    }
                }
            }
            // rows: t = Bᵀ·d
            var t: [in_tile][in_tile]V = undefined;
            inline for (0..in_tile) |j| {
                var column: [in_tile]V = undefined;
                inline for (0..in_tile) |i| column[i] = dd[i][j];
                const col = Rows.bRow(L, column);
                inline for (0..in_tile) |i| t[i][j] = col[i];
            }
            // cols: V = t·B
            const dst = tile_index * d.cin + ic;
            inline for (0..in_tile) |i| {
                const row = Rows.bRow(L, t[i]);
                inline for (0..in_tile) |j| {
                    v[i * in_tile + j][dst..][0..L].* = row[j];
                }
            }
        }

        // --- output transform ---------------------------------------------

        const OutputCtx = struct { y: []f32, m: *const PlanesConst, bias: ?[]const f32, d: F2Dims };

        /// y = Aᵀ·M·A (+ bias, then optional fused relu — the same single
        /// relu the caller would apply, evaluated on identical values) per
        /// tile; writes only the valid output positions (`out_tile·ty+r <
        /// oh`, `out_tile·tx+s < ow`). Parallel over tile rows (disjoint
        /// `y` rows).
        pub fn outputTransformInto(pc: common.ParallelConfig, y: []f32, m: *const PlanesConst, bias: ?[]const f32, fuse_relu: bool, d: F2Dims) void {
            const ctx: OutputCtx = .{ .y = y, .m = m, .bias = bias, .d = d };
            const work = planes * d.tiles_y * d.tiles_x * d.cout;
            if (fuse_relu) {
                dispatch(pc, OutputCtx, ctx, d.tiles_y, work, outputRange(true));
            } else {
                dispatch(pc, OutputCtx, ctx, d.tiles_y, work, outputRange(false));
            }
        }

        fn outputRange(comptime fuse_relu: bool) fn (OutputCtx, usize, usize) void {
            return struct {
                fn go(c: OutputCtx, ty_start: usize, ty_end: usize) void {
                    const d = c.d;
                    var ty = ty_start;
                    while (ty < ty_end) : (ty += 1) {
                        var tx: usize = 0;
                        while (tx < d.tiles_x) : (tx += 1) {
                            const tile_index = ty * d.tiles_x + tx;
                            var oc: usize = 0;
                            while (oc + vector_len <= d.cout) : (oc += vector_len) {
                                outputTile(fuse_relu, vector_len, c.y, c.m, c.bias, d, ty, tx, tile_index, oc);
                            }
                            while (oc < d.cout) : (oc += 1) {
                                outputTile(fuse_relu, 1, c.y, c.m, c.bias, d, ty, tx, tile_index, oc);
                            }
                        }
                    }
                }
            }.go;
        }

        inline fn outputTile(comptime fuse_relu: bool, comptime L: usize, y: []f32, m: *const PlanesConst, bias: ?[]const f32, d: F2Dims, ty: usize, tx: usize, tile_index: usize, oc: usize) void {
            const V = @Vector(L, f32);
            const src = tile_index * d.cout + oc;
            var mm: [in_tile][in_tile]V = undefined;
            inline for (0..in_tile) |i| {
                inline for (0..in_tile) |j| {
                    mm[i][j] = m[i * in_tile + j][src..][0..L].*;
                }
            }
            // rows: r = Aᵀ·M (out_tile × in_tile)
            var r: [out_tile][in_tile]V = undefined;
            inline for (0..in_tile) |j| {
                var column: [in_tile]V = undefined;
                inline for (0..in_tile) |i| column[i] = mm[i][j];
                const col = Rows.aRow(L, column);
                inline for (0..out_tile) |i| r[i][j] = col[i];
            }
            // cols: y = r·A (out_tile × out_tile), plus bias.
            const bv: V = if (bias) |b| b[oc..][0..L].* else @splat(0);
            const vzero: V = @splat(0);
            inline for (0..out_tile) |i| {
                const row = Rows.aRow(L, r[i]);
                const oy = out_tile * ty + i;
                if (oy < d.oh) {
                    inline for (0..out_tile) |j| {
                        const ox = out_tile * tx + j;
                        if (ox < d.ow) {
                            const val = row[j] + bv;
                            y[(oy * d.ow + ox) * d.cout + oc ..][0..L].* = if (fuse_relu) @max(val, vzero) else val;
                        }
                    }
                }
            }
        }
    };
}

/// F(2×2, 3×3) rows: G = [1,0,0; ½,½,½; ½,−½,½; 0,0,1], Bᵀ = [1,0,−1,0;
/// 0,1,1,0; 0,−1,1,0; 0,1,0,−1], Aᵀ = [1,1,1,0; 0,1,−1,−1].
const F2Rows = struct {
    inline fn gRow(comptime L: usize, g: [3]@Vector(L, f32)) [4]@Vector(L, f32) {
        const half: @Vector(L, f32) = @splat(0.5);
        return .{ g[0], (g[0] + g[1] + g[2]) * half, (g[0] - g[1] + g[2]) * half, g[2] };
    }

    inline fn bRow(comptime L: usize, d: [4]@Vector(L, f32)) [4]@Vector(L, f32) {
        return .{ d[0] - d[2], d[1] + d[2], d[2] - d[1], d[1] - d[3] };
    }

    inline fn aRow(comptime L: usize, m: [4]@Vector(L, f32)) [2]@Vector(L, f32) {
        return .{ m[0] + m[1] + m[2], m[1] - m[2] - m[3] };
    }
};

/// F(4×4, 3×3) rows: G = [¼,0,0; −⅙,−⅙,−⅙; −⅙,⅙,−⅙; 1/24,1/12,⅙;
/// 1/24,−1/12,⅙; 0,0,1], the standard Bᵀ (rows 4,0,−5,0,1,0 / 0,∓4,−4,±1,1,0
/// / 0,∓2,−1,±2,1,0 / 0,4,0,−5,0,1) and Aᵀ = [1,1,1,1,1,0; 0,1,−1,2,−2,0;
/// 0,1,1,4,4,0; 0,1,−1,8,−8,1].
const F4Rows = struct {
    inline fn gRow(comptime L: usize, g: [3]@Vector(L, f32)) [6]@Vector(L, f32) {
        const V = @Vector(L, f32);
        const q: V = @splat(0.25);
        const s6: V = @splat(-1.0 / 6.0);
        const s24: V = @splat(1.0 / 24.0);
        const s12: V = @splat(1.0 / 12.0);
        const s3: V = @splat(1.0 / 6.0);
        const a = g[0];
        const b = g[1];
        const c = g[2];
        return .{
            a * q,
            (a + b + c) * s6,
            (a - b + c) * s6,
            a * s24 + b * s12 + c * s3,
            a * s24 - b * s12 + c * s3,
            c,
        };
    }

    inline fn bRow(comptime L: usize, d: [6]@Vector(L, f32)) [6]@Vector(L, f32) {
        const V = @Vector(L, f32);
        const four: V = @splat(4.0);
        const five: V = @splat(5.0);
        const two: V = @splat(2.0);
        return .{
            four * d[0] - five * d[2] + d[4],
            -four * d[1] - four * d[2] + d[3] + d[4],
            four * d[1] - four * d[2] - d[3] + d[4],
            -two * d[1] - d[2] + two * d[3] + d[4],
            two * d[1] - d[2] - two * d[3] + d[4],
            four * d[1] - five * d[3] + d[5],
        };
    }

    inline fn aRow(comptime L: usize, m: [6]@Vector(L, f32)) [4]@Vector(L, f32) {
        const V = @Vector(L, f32);
        const two: V = @splat(2.0);
        const four: V = @splat(4.0);
        const eight: V = @splat(8.0);
        return .{
            m[0] + m[1] + m[2] + m[3] + m[4],
            m[1] - m[2] + two * m[3] - two * m[4],
            m[1] + m[2] + four * m[3] + four * m[4],
            m[1] - m[2] + eight * m[3] - eight * m[4] + m[5],
        };
    }
};

const F2 = Winograd(4, 2, F2Rows);
const F4 = Winograd(6, 4, F4Rows);

pub const f2WeightTransformInto = F2.weightTransformInto;
pub const f2InputTransformInto = F2.inputTransformInto;
pub const f2OutputTransformInto = F2.outputTransformInto;
pub const f4WeightTransformInto = F4.weightTransformInto;
pub const f4InputTransformInto = F4.inputTransformInto;
pub const f4OutputTransformInto = F4.outputTransformInto;
