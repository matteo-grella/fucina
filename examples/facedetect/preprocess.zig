//! Detector preprocess: SCRFD letterbox — aspect-preserving cv2
//! INTER_LINEAR resize (11-bit fixed point) into a top-left zero-padded `size²`
//! canvas, plus the geometry-only `det_scale`. Direct transcription of
//! refs/face-detect.cpp `scrfd_letterbox` + `cv_resize_linear_u8`. Pure integer
//! fixed point (no trig) → bit-exact to the reference.

const std = @import("std");
const image = @import("image.zig");

/// Round half to even (C `lrint`).
fn rintEven(v: f64) i64 {
    const f = std.math.floor(v);
    const diff = v - f;
    if (diff < 0.5) return @intFromFloat(f);
    if (diff > 0.5) return @intFromFloat(f + 1.0);
    const fi: i64 = @intFromFloat(f);
    return if (@rem(fi, 2) == 0) fi else fi + 1;
}

fn satShort(v: f32) i64 {
    const i = rintEven(@as(f64, v));
    return @max(@as(i64, -32768), @min(@as(i64, 32767), i));
}

/// cv2.resize INTER_LINEAR for uint8, 11-bit fixed point (OpenCV 8U path).
fn cvResize(allocator: std.mem.Allocator, src: []const u8, sw: usize, sh: usize, dst: []u8, dw: usize, dh: usize, cn: usize) !void {
    const SCALE: f32 = 2048.0;
    const xofs = try allocator.alloc(i64, dw);
    defer allocator.free(xofs);
    const ialpha = try allocator.alloc(i64, dw * 2);
    defer allocator.free(ialpha);
    const yofs = try allocator.alloc(i64, dh);
    defer allocator.free(yofs);
    const ibeta = try allocator.alloc(i64, dh * 2);
    defer allocator.free(ibeta);

    const sxs = @as(f64, @floatFromInt(sw)) / @as(f64, @floatFromInt(dw));
    const sys = @as(f64, @floatFromInt(sh)) / @as(f64, @floatFromInt(dh));
    for (0..dw) |dx| {
        var fx: f32 = @floatCast((@as(f64, @floatFromInt(dx)) + 0.5) * sxs - 0.5);
        var sx: i64 = @intFromFloat(@floor(fx));
        fx -= @floatFromInt(sx);
        if (sx < 0) {
            sx = 0;
            fx = 0;
        }
        if (sx >= @as(i64, @intCast(sw)) - 1) {
            sx = @as(i64, @intCast(sw)) - 1;
            fx = 0;
        }
        xofs[dx] = sx;
        ialpha[dx * 2] = satShort((1.0 - fx) * SCALE);
        ialpha[dx * 2 + 1] = satShort(fx * SCALE);
    }
    for (0..dh) |dy| {
        var fy: f32 = @floatCast((@as(f64, @floatFromInt(dy)) + 0.5) * sys - 0.5);
        var sy: i64 = @intFromFloat(@floor(fy));
        fy -= @floatFromInt(sy);
        if (sy < 0) {
            sy = 0;
            fy = 0;
        }
        if (sy >= @as(i64, @intCast(sh)) - 1) {
            sy = @as(i64, @intCast(sh)) - 1;
            fy = 0;
        }
        yofs[dy] = sy;
        ibeta[dy * 2] = satShort((1.0 - fy) * SCALE);
        ibeta[dy * 2 + 1] = satShort(fy * SCALE);
    }

    for (0..dh) |dy| {
        const sy0: usize = @intCast(yofs[dy]);
        const sy1: usize = @min(sy0 + 1, sh - 1);
        const b0 = ibeta[dy * 2];
        const b1 = ibeta[dy * 2 + 1];
        for (0..dw) |dx| {
            const sx: usize = @intCast(xofs[dx]);
            const sx1: usize = @min(sx + 1, sw - 1);
            const a0 = ialpha[dx * 2];
            const a1 = ialpha[dx * 2 + 1];
            for (0..cn) |c| {
                const r0a: i64 = src[(sy0 * sw + sx) * cn + c];
                const r0b: i64 = src[(sy0 * sw + sx1) * cn + c];
                const r1a: i64 = src[(sy1 * sw + sx) * cn + c];
                const r1b: i64 = src[(sy1 * sw + sx1) * cn + c];
                const p0 = r0a * a0 + r0b * a1;
                const p1 = r1a * a0 + r1b * a1;
                var v = (p0 * b0 + p1 * b1 + (1 << 21)) >> 22;
                v = @max(@as(i64, 0), @min(@as(i64, 255), v));
                dst[(dy * dw + dx) * cn + c] = @intCast(v);
            }
        }
    }
}

pub const Letterboxed = struct {
    img: image.Image,
    det_scale: f32,
};

/// Letterbox `src` into a `size×size` top-left zero-padded RGB canvas + det_scale.
pub fn letterbox(allocator: std.mem.Allocator, src: *const image.Image, size: usize) !Letterboxed {
    const ow = src.width;
    const oh = src.height;
    const im_ratio = @as(f32, @floatFromInt(oh)) / @as(f32, @floatFromInt(ow));
    var new_w: usize = undefined;
    var new_h: usize = undefined;
    if (im_ratio > 1.0) {
        new_h = size;
        new_w = @intFromFloat(@as(f32, @floatFromInt(size)) / im_ratio);
    } else {
        new_w = size;
        new_h = @intFromFloat(@as(f32, @floatFromInt(size)) * im_ratio);
    }
    const det_scale = @as(f32, @floatFromInt(new_h)) / @as(f32, @floatFromInt(oh));

    const resized = try allocator.alloc(u8, new_w * new_h * 3);
    defer allocator.free(resized);
    try cvResize(allocator, src.pixels, ow, oh, resized, new_w, new_h, 3);

    var out = try image.Image.initRgb(allocator, size, size);
    @memset(out.pixels, 0);
    for (0..new_h) |y| {
        @memcpy(out.pixels[(y * size) * 3 ..][0 .. new_w * 3], resized[(y * new_w) * 3 ..][0 .. new_w * 3]);
    }
    return .{ .img = out, .det_scale = det_scale };
}
