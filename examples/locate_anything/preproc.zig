//! Image preprocessing: PIL-exact bicubic resize -> normalize -> patchify.
//!
//! Faithful port of refs/locate-anything.cpp/src/pil_resize.cpp (itself a port
//! of Pillow's libImaging/Resample.c BICUBIC) and src/image_io.cpp
//! `preprocess`/`preproc_target`. Deliberately scalar host code: it runs once
//! per image, is never the bottleneck, and byte-exact uint8 resampling decides
//! the pixel values every later stage consumes.
//! Pillow specifics reproduced: a = -0.5 (torch uses -0.75),
//! antialias on downscale (support scaled by in/out), coefficients accumulated
//! in f64, and the u8 round-trip BETWEEN the two passes.

const std = @import("std");

const Allocator = std.mem.Allocator;

/// Pillow cubic_filter, a = -0.5.
fn cubicFilter(x_in: f64) f64 {
    const a = -0.5;
    var x = x_in;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

const bicubic_support = 2.0;

const Coeffs = struct {
    ksize: usize,
    /// Per output pixel: [xmin, xsize].
    bounds: []usize,
    /// Per output pixel: ksize weights.
    kk: []f64,

    fn deinit(self: *Coeffs, allocator: Allocator) void {
        allocator.free(self.bounds);
        allocator.free(self.kk);
        self.* = undefined;
    }
};

/// Pillow precompute_coeffs for a full-extent box.
fn precomputeCoeffs(allocator: Allocator, in_size: usize, out_size: usize) !Coeffs {
    const scale = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
    const filterscale = @max(scale, 1.0);
    const support = bicubic_support * filterscale;
    const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;

    var coeffs = Coeffs{
        .ksize = ksize,
        .bounds = try allocator.alloc(usize, out_size * 2),
        .kk = try allocator.alloc(f64, out_size * ksize),
    };
    errdefer coeffs.deinit(allocator);

    for (0..out_size) |xx| {
        const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
        const ss = 1.0 / filterscale;
        // int casts truncate toward zero, exactly like the C reference.
        var xmin_i: isize = @intFromFloat(center - support + 0.5);
        if (xmin_i < 0) xmin_i = 0;
        const xmin: usize = @intCast(xmin_i);
        var xmax_i: isize = @intFromFloat(center + support + 0.5);
        if (xmax_i > @as(isize, @intCast(in_size))) xmax_i = @intCast(in_size);
        const xmax: usize = @as(usize, @intCast(xmax_i)) - xmin;

        const k = coeffs.kk[xx * ksize ..][0..ksize];
        var ww: f64 = 0.0;
        for (0..xmax) |x| {
            const weight = cubicFilter((@as(f64, @floatFromInt(x + xmin)) - center + 0.5) * ss);
            k[x] = weight;
            ww += weight;
        }
        for (0..xmax) |x| {
            if (ww != 0.0) k[x] /= ww;
        }
        for (xmax..ksize) |x| k[x] = 0.0;
        coeffs.bounds[xx * 2 + 0] = xmin;
        coeffs.bounds[xx * 2 + 1] = xmax;
    }
    return coeffs;
}

/// Round-to-nearest with clamp to [0,255]; Pillow's fixed-point bias is
/// round-half-up, which C lround (round half away from zero — @round here)
/// matches for the non-negative magnitudes that occur after coefficient
/// normalization.
fn clip8(v: f64) u8 {
    const r: i64 = @intFromFloat(@round(v));
    if (r < 0) return 0;
    if (r > 255) return 255;
    return @intCast(r);
}

/// Two-pass (horizontal then vertical) Pillow BICUBIC resize of an RGB8 image.
/// The intermediate image is 8bpc: the u8 rounding between passes is part of
/// the reference behavior.
pub fn pilBicubicResize(
    allocator: Allocator,
    src: []const u8,
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) ![]u8 {
    std.debug.assert(src.len == sw * sh * 3);

    // ---- Horizontal pass: (sw, sh) -> (dw, sh) ----
    var hc = try precomputeCoeffs(allocator, sw, dw);
    defer hc.deinit(allocator);
    const tmp = try allocator.alloc(u8, dw * sh * 3);
    defer allocator.free(tmp);
    for (0..sh) |y| {
        const row = src[y * sw * 3 ..][0 .. sw * 3];
        for (0..dw) |xx| {
            const xmin = hc.bounds[xx * 2 + 0];
            const xsize = hc.bounds[xx * 2 + 1];
            const k = hc.kk[xx * hc.ksize ..];
            var s0: f64 = 0.0;
            var s1: f64 = 0.0;
            var s2: f64 = 0.0;
            for (0..xsize) |i| {
                const p = row[(xmin + i) * 3 ..][0..3];
                s0 += @as(f64, @floatFromInt(p[0])) * k[i];
                s1 += @as(f64, @floatFromInt(p[1])) * k[i];
                s2 += @as(f64, @floatFromInt(p[2])) * k[i];
            }
            const out = tmp[(y * dw + xx) * 3 ..][0..3];
            out[0] = clip8(s0);
            out[1] = clip8(s1);
            out[2] = clip8(s2);
        }
    }

    // ---- Vertical pass: (dw, sh) -> (dw, dh) ----
    var vc = try precomputeCoeffs(allocator, sh, dh);
    defer vc.deinit(allocator);
    const out = try allocator.alloc(u8, dw * dh * 3);
    errdefer allocator.free(out);
    for (0..dh) |yy| {
        const ymin = vc.bounds[yy * 2 + 0];
        const ysize = vc.bounds[yy * 2 + 1];
        const k = vc.kk[yy * vc.ksize ..];
        for (0..dw) |x| {
            var s0: f64 = 0.0;
            var s1: f64 = 0.0;
            var s2: f64 = 0.0;
            for (0..ysize) |i| {
                const p = tmp[((ymin + i) * dw + x) * 3 ..][0..3];
                s0 += @as(f64, @floatFromInt(p[0])) * k[i];
                s1 += @as(f64, @floatFromInt(p[1])) * k[i];
                s2 += @as(f64, @floatFromInt(p[2])) * k[i];
            }
            const o = out[(yy * dw + x) * 3 ..][0..3];
            o[0] = clip8(s0);
            o[1] = clip8(s1);
            o[2] = clip8(s2);
        }
    }
    return out;
}

pub const Preprocessed = struct {
    allocator: Allocator,
    /// Patch grid (rows, cols): target_h/patch, target_w/patch.
    gh: usize,
    gw: usize,
    target_w: usize,
    target_h: usize,
    /// Patchified normalized pixels: [gh*gw, patch_dim] row-major, patch
    /// token t = row*gw+col, within-patch (c, i, j) order, v = raw/127.5 - 1.
    pixel_values: []f32,

    pub fn deinit(self: *Preprocessed) void {
        self.allocator.free(self.pixel_values);
        self.* = undefined;
    }
};

pub const Limits = struct {
    patch: usize,
    merge_h: usize,
    merge_w: usize,
    in_token_limit: usize,
};

/// LocateAnythingImageProcessor.rescale target-size rule: (A) downscale if the
/// patch grid would exceed in_token_limit, then (B) round each dimension UP to
/// a multiple of merge_kernel * patch.
pub fn preprocTarget(w0: usize, h0: usize, limits: Limits, target_w: *usize, target_h: *usize) void {
    var w = w0;
    var h = h0;
    if ((w / limits.patch) * (h / limits.patch) > limits.in_token_limit) {
        const scale = @sqrt(@as(f64, @floatFromInt(limits.in_token_limit)) /
            (@as(f64, @floatFromInt(w / limits.patch)) * @as(f64, @floatFromInt(h / limits.patch))));
        w = @intFromFloat(@as(f64, @floatFromInt(w0)) * scale); // truncate, like int()
        h = @intFromFloat(@as(f64, @floatFromInt(h0)) * scale);
    }
    const pad_w = limits.merge_w * limits.patch;
    const pad_h = limits.merge_h * limits.patch;
    target_w.* = std.math.divCeil(usize, w, pad_w) catch unreachable;
    target_w.* *= pad_w;
    target_h.* = std.math.divCeil(usize, h, pad_h) catch unreachable;
    target_h.* *= pad_h;
}

/// Full preprocessing chain on an RGB8 image: (optional) token-limit
/// downscale, pad-resize to the merge-aligned target, normalize, patchify.
pub fn preprocess(
    allocator: Allocator,
    rgb: []const u8,
    w0: usize,
    h0: usize,
    limits: Limits,
) !Preprocessed {
    std.debug.assert(rgb.len == w0 * h0 * 3);
    const patch = limits.patch;

    var cur: []u8 = try allocator.dupe(u8, rgb);
    defer allocator.free(cur);
    var w = w0;
    var h = h0;

    if ((w / patch) * (h / patch) > limits.in_token_limit) {
        const scale = @sqrt(@as(f64, @floatFromInt(limits.in_token_limit)) /
            (@as(f64, @floatFromInt(w / patch)) * @as(f64, @floatFromInt(h / patch))));
        const w1: usize = @intFromFloat(@as(f64, @floatFromInt(w0)) * scale);
        const h1: usize = @intFromFloat(@as(f64, @floatFromInt(h0)) * scale);
        // Extreme aspect ratios can truncate a dimension to 0 (e.g. 14 x 6M):
        // resizing to a zero extent divides by zero in the coefficient
        // precompute. The reference has the same latent failure (its int()
        // truncation feeds Pillow's resample the same 0); erroring out is the
        // only well-defined behavior for such an image.
        if (w1 == 0 or h1 == 0) return error.ImageTooLarge;
        const resized = try pilBicubicResize(allocator, cur, w, h, w1, h1);
        allocator.free(cur);
        cur = resized;
        w = w1;
        h = h1;
    }

    var target_w: usize = 0;
    var target_h: usize = 0;
    preprocTarget(w0, h0, limits, &target_w, &target_h);
    if (target_w != w or target_h != h) {
        const resized = try pilBicubicResize(allocator, cur, w, h, target_w, target_h);
        allocator.free(cur);
        cur = resized;
    }

    // Upstream asserts the final grid stays under 512 patches per axis.
    if (target_w / patch >= 512 or target_h / patch >= 512) return error.ImageTooLarge;

    const gh = target_h / patch;
    const gw = target_w / patch;
    const patch_dim = patch * patch * 3;

    // Normalize + patchify in one pass:
    // pv[t*patch_dim + c*patch*patch + i*patch + j] = cur[(row*patch+i)*W + col*patch+j][c]/127.5 - 1.
    const pixel_values = try allocator.alloc(f32, gh * gw * patch_dim);
    errdefer allocator.free(pixel_values);
    for (0..gh) |row| {
        for (0..gw) |col| {
            const t = row * gw + col;
            const dst = pixel_values[t * patch_dim ..][0..patch_dim];
            for (0..patch) |i| {
                const y = row * patch + i;
                const src_row = cur[(y * target_w + col * patch) * 3 ..][0 .. patch * 3];
                for (0..patch) |j| {
                    inline for (0..3) |c| {
                        dst[c * patch * patch + i * patch + j] =
                            @as(f32, @floatFromInt(src_row[j * 3 + c])) / 127.5 - 1.0;
                    }
                }
            }
        }
    }

    return .{
        .allocator = allocator,
        .gh = gh,
        .gw = gw,
        .target_w = target_w,
        .target_h = target_h,
        .pixel_values = pixel_values,
    };
}

test {
    _ = @import("preproc_tests.zig");
}
