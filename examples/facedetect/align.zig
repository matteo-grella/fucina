//! Face alignment: umeyama similarity fit of the 5 detected landmarks
//! to the insightface arcface_dst template + OpenCV-faithful warpAffine
//! (INTER_LINEAR, 5-bit sub-pixel / 10-bit fixed point) → the 112² aligned RGB
//! crop. Direct transcription of refs/face-detect.cpp `src/align.cpp` (f64 math).

const std = @import("std");
const image = @import("image.zig");

/// insightface arcface_dst — the 5-point template for a 112×112 aligned face.
const arcface_dst = [5][2]f64{
    .{ 38.2946, 51.6963 },
    .{ 73.5318, 51.5014 },
    .{ 56.0252, 71.7366 },
    .{ 41.5493, 92.3655 },
    .{ 70.7299, 92.2041 },
};

/// Round half to even (C `lrint` with the default FP rounding mode).
fn rintEven(v: f64) i64 {
    const f = std.math.floor(v);
    const diff = v - f;
    if (diff < 0.5) return @intFromFloat(f);
    if (diff > 0.5) return @intFromFloat(f + 1.0);
    const fi: i64 = @intFromFloat(f);
    return if (@rem(fi, 2) == 0) fi else fi + 1;
}

fn svd2x2(a: f64, b: f64, c: f64, d: f64, U: *[2][2]f64, S: *[2]f64, Vt: *[2][2]f64) void {
    const ata00 = a * a + c * c;
    const ata01 = a * b + c * d;
    const ata11 = b * b + d * d;
    const phi = 0.5 * std.math.atan2(2.0 * ata01, ata00 - ata11);
    const cphi = @cos(phi);
    const sphi = @sin(phi);
    const t1 = ata00 + ata11;
    const t2 = std.math.hypot(ata00 - ata11, 2.0 * ata01);
    const s0 = @sqrt(@max((t1 + t2) * 0.5, 0.0));
    const s1 = @sqrt(@max((t1 - t2) * 0.5, 0.0));
    const V = [2][2]f64{ .{ cphi, -sphi }, .{ sphi, cphi } };
    const av00 = a * V[0][0] + b * V[1][0];
    const av01 = a * V[0][1] + b * V[1][1];
    const av10 = c * V[0][0] + d * V[1][0];
    const av11 = c * V[0][1] + d * V[1][1];
    if (s0 > 1e-12) {
        U[0][0] = av00 / s0;
        U[1][0] = av10 / s0;
    } else {
        U[0][0] = av00;
        U[1][0] = av10;
    }
    if (s1 > 1e-12) {
        U[0][1] = av01 / s1;
        U[1][1] = av11 / s1;
    } else {
        U[0][1] = av01;
        U[1][1] = av11;
    }
    S[0] = s0;
    S[1] = s1;
    Vt[0][0] = V[0][0];
    Vt[0][1] = V[1][0];
    Vt[1][0] = V[0][1];
    Vt[1][1] = V[1][1];
}

/// Umeyama similarity (rotation + uniform scale + translation), src→dst → 2×3 M.
fn estimateNorm(lmk: [5][2]f64) [6]f64 {
    const N: f64 = 5;
    var sx: f64 = 0;
    var sy: f64 = 0;
    var dx: f64 = 0;
    var dy: f64 = 0;
    for (0..5) |i| {
        sx += lmk[i][0];
        sy += lmk[i][1];
        dx += arcface_dst[i][0];
        dy += arcface_dst[i][1];
    }
    sx /= N;
    sy /= N;
    dx /= N;
    dy /= N;
    var var_s: f64 = 0;
    var H = [2][2]f64{ .{ 0, 0 }, .{ 0, 0 } };
    for (0..5) |i| {
        const sxx = lmk[i][0] - sx;
        const syy = lmk[i][1] - sy;
        const dxx = arcface_dst[i][0] - dx;
        const dyy = arcface_dst[i][1] - dy;
        var_s += sxx * sxx + syy * syy;
        H[0][0] += dxx * sxx;
        H[0][1] += dxx * syy;
        H[1][0] += dyy * sxx;
        H[1][1] += dyy * syy;
    }
    var_s /= N;
    H[0][0] /= N;
    H[0][1] /= N;
    H[1][0] /= N;
    H[1][1] /= N;
    var U: [2][2]f64 = undefined;
    var S: [2]f64 = undefined;
    var Vt: [2][2]f64 = undefined;
    svd2x2(H[0][0], H[0][1], H[1][0], H[1][1], &U, &S, &Vt);
    const detH = H[0][0] * H[1][1] - H[0][1] * H[1][0];
    const D0: f64 = 1.0;
    const D1: f64 = if (detH < 0) -1.0 else 1.0;
    var Rm: [2][2]f64 = undefined;
    Rm[0][0] = U[0][0] * D0 * Vt[0][0] + U[0][1] * D1 * Vt[1][0];
    Rm[0][1] = U[0][0] * D0 * Vt[0][1] + U[0][1] * D1 * Vt[1][1];
    Rm[1][0] = U[1][0] * D0 * Vt[0][0] + U[1][1] * D1 * Vt[1][0];
    Rm[1][1] = U[1][0] * D0 * Vt[0][1] + U[1][1] * D1 * Vt[1][1];
    const scale: f64 = if (var_s > 0) (S[0] * D0 + S[1] * D1) / var_s else 1.0;
    var M: [6]f64 = undefined;
    M[0] = scale * Rm[0][0];
    M[1] = scale * Rm[0][1];
    M[3] = scale * Rm[1][0];
    M[4] = scale * Rm[1][1];
    M[2] = dx - (M[0] * sx + M[1] * sy);
    M[5] = dy - (M[3] * sx + M[4] * sy);
    return M;
}

/// OpenCV-faithful backward warpAffine INTER_LINEAR into a new `out_w×out_h` RGB.
fn warpAffine(allocator: std.mem.Allocator, src: *const image.Image, M: [6]f64, out_w: usize, out_h: usize) ![]u8 {
    const det = M[0] * M[4] - M[1] * M[3];
    const inv00 = M[4] / det;
    const inv01 = -M[1] / det;
    const inv10 = -M[3] / det;
    const inv11 = M[0] / det;
    const inv02 = -(inv00 * M[2] + inv01 * M[5]);
    const inv12 = -(inv10 * M[2] + inv11 * M[5]);

    const INTER_BITS: u6 = 5;
    const INTER_TAB: i64 = 1 << 5; // 32
    const AB_SCALE: f64 = 1024.0; // 1<<10
    const ROUND_DELTA: i64 = 16; // AB_SCALE/INTER_TAB/2
    const SHIFT: u6 = 10 - 5; // AB_BITS - INTER_BITS

    const sw: i64 = @intCast(src.width);
    const sh: i64 = @intCast(src.height);
    const S = struct {
        fn at(s: *const image.Image, w: i64, h: i64, xx: i64, yy: i64, ch: usize) f64 {
            if (xx < 0 or yy < 0 or xx >= w or yy >= h) return 0.0;
            return @floatFromInt(s.pixels[(@as(usize, @intCast(yy)) * s.width + @as(usize, @intCast(xx))) * 3 + ch]);
        }
    };

    const out = try allocator.alloc(u8, out_w * out_h * 3);
    const adelta = try allocator.alloc(i64, out_w);
    defer allocator.free(adelta);
    const bdelta = try allocator.alloc(i64, out_w);
    defer allocator.free(bdelta);
    for (0..out_w) |dxx| {
        const dxf: f64 = @floatFromInt(dxx);
        adelta[dxx] = rintEven(inv00 * dxf * AB_SCALE);
        bdelta[dxx] = rintEven(inv10 * dxf * AB_SCALE);
    }
    for (0..out_h) |dyy| {
        const dyf: f64 = @floatFromInt(dyy);
        const X0 = rintEven((inv01 * dyf + inv02) * AB_SCALE) + ROUND_DELTA;
        const Y0 = rintEven((inv11 * dyf + inv12) * AB_SCALE) + ROUND_DELTA;
        for (0..out_w) |dxx| {
            const X = (X0 + adelta[dxx]) >> SHIFT;
            const Y = (Y0 + bdelta[dxx]) >> SHIFT;
            const x0 = X >> INTER_BITS;
            const y0 = Y >> INTER_BITS;
            const ax: f64 = @as(f64, @floatFromInt(X & (INTER_TAB - 1))) / 32.0;
            const ay: f64 = @as(f64, @floatFromInt(Y & (INTER_TAB - 1))) / 32.0;
            for (0..3) |ch| {
                const v = S.at(src, sw, sh, x0, y0, ch) * (1 - ax) * (1 - ay) +
                    S.at(src, sw, sh, x0 + 1, y0, ch) * ax * (1 - ay) +
                    S.at(src, sw, sh, x0, y0 + 1, ch) * (1 - ax) * ay +
                    S.at(src, sw, sh, x0 + 1, y0 + 1, ch) * ax * ay;
                out[(dyy * out_w + dxx) * 3 + ch] = @intFromFloat(std.math.round(v));
            }
        }
    }
    return out;
}

/// Aligned 112² RGB crop from the source image + 5 landmarks (caller frees).
pub fn normCrop(allocator: std.mem.Allocator, src: *const image.Image, lmk: [5][2]f64, size: usize) ![]u8 {
    return warpAffine(allocator, src, estimateNorm(lmk), size, size);
}

/// Box-center scale-fit crop (the genderage / dense-landmark geometry): expand
/// max(w,h) by `expand`, scale-fit into `size²` centered. Mirrors the genderage
/// caller `s = size/(max(w,h)·expand); M = [s,0,-s·cx+size/2; 0,s,-s·cy+size/2]`.
pub fn centerScaleCrop(allocator: std.mem.Allocator, src: *const image.Image, box: [4]f32, size: usize, expand: f64) ![]u8 {
    const w = @as(f64, box[2]) - @as(f64, box[0]);
    const h = @as(f64, box[3]) - @as(f64, box[1]);
    const cx = (@as(f64, box[0]) + @as(f64, box[2])) * 0.5;
    const cy = (@as(f64, box[1]) + @as(f64, box[3])) * 0.5;
    const sz: f64 = @floatFromInt(size);
    const s = sz / (@max(w, h) * expand);
    const M = [6]f64{ s, 0, -s * cx + sz * 0.5, 0, s, -s * cy + sz * 0.5 };
    return warpAffine(allocator, src, M, size, size);
}
