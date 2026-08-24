//! torchaudio.functional.resample-compatible Hann-windowed-sinc polyphase
//! resampler (lowpass_filter_width=6, rolloff=0.99) for the OmniVoice example.
//!
//! Port of refs/omnivoice.cpp/src/audio-resample.h. Kernel math is all f64
//! (stored f32); the apply loop accumulates sequentially in f32 — both widths
//! and the accumulation order match the reference exactly.

const std = @import("std");

pub const lowpass_filter_width = 6;
pub const rolloff = 0.99;

/// Euclidean gcd (audio_resample_gcd).
pub fn gcd(a0: u32, b0: u32) u32 {
    var a = a0;
    var b = b0;
    while (b != 0) {
        const t = b;
        b = a % b;
        a = t;
    }
    return a;
}

/// Hann-sinc polyphase kernel built by `buildKernel`. `weights` is
/// [newf][kernel_size] row major, owned by the caller's allocator.
pub const Kernel = struct {
    weights: []f32,
    width: usize,
    kernel_size: usize,

    pub fn deinit(self: *Kernel, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        self.* = undefined;
    }
};

/// Builds the polyphase kernel for reduced rates orig=sr_in/gcd, newf=sr_out/gcd
/// (audio_resample_build_kernel). All intermediate math in f64, stored f32.
pub fn buildKernel(allocator: std.mem.Allocator, orig: u32, newf: u32) !Kernel {
    const base_int: u32 = @min(orig, newf);
    const base: f64 = @as(f64, @floatFromInt(base_int)) * rolloff;
    const width: usize = @intFromFloat(@ceil(@as(f64, lowpass_filter_width) * @as(f64, @floatFromInt(orig)) / base));
    const k_size: usize = 2 * width + orig;

    const weights = try allocator.alloc(f32, @as(usize, newf) * k_size);
    errdefer allocator.free(weights);

    const scale = base / @as(f64, @floatFromInt(orig));
    const inv_o = 1.0 / @as(f64, @floatFromInt(orig));
    const inv_n = 1.0 / @as(f64, @floatFromInt(newf));
    const pi: f64 = std.math.pi;

    for (0..newf) |j| {
        const t_off = @as(f64, @floatFromInt(-@as(i64, @intCast(j)))) * inv_n;
        for (0..k_size) |k| {
            const idx_k = @as(f64, @floatFromInt(@as(i64, @intCast(k)) - @as(i64, @intCast(width)))) * inv_o;
            var t = (t_off + idx_k) * base;
            if (t < -@as(f64, lowpass_filter_width)) t = -lowpass_filter_width;
            if (t > lowpass_filter_width) t = lowpass_filter_width;

            var w = @cos(t * pi / @as(f64, lowpass_filter_width) / 2.0);
            w = w * w;

            const tp = t * pi;
            const sinc: f64 = if (tp == 0.0) 1.0 else @sin(tp) / tp;

            weights[j * k_size + k] = @floatCast(sinc * w * scale);
        }
    }

    return .{ .weights = weights, .width = width, .kernel_size = k_size };
}

/// Resamples one mono channel (audio_resample_apply_mono): zero pad width in
/// front and width+orig behind, then strided polyphase dot products. Writes
/// min(out.len, n_per_chan*newf) samples into `out` (for the reference's
/// target length that minimum is always out.len). f32 sequential accumulate.
pub fn applyMono(allocator: std.mem.Allocator, in: []const f32, orig: u32, newf: u32, kernel: *const Kernel, out: []f32) !void {
    const k_size = kernel.kernel_size;
    const np = in.len + 2 * kernel.width + orig;
    const padded = try allocator.alloc(f32, np);
    defer allocator.free(padded);
    @memset(padded, 0);
    @memcpy(padded[kernel.width..][0..in.len], in);

    const n_per_chan = (np - k_size) / orig + 1;
    const total = n_per_chan * newf;
    const out_len = @min(out.len, total);

    for (0..out_len) |t_out| {
        const chan = t_out % newf;
        const pos = t_out / newf;
        const w = kernel.weights[chan * k_size ..][0..k_size];
        const x = padded[pos * orig ..][0..k_size];
        var sum: f32 = 0.0;
        for (0..k_size) |k| {
            sum += x[k] * w[k];
        }
        out[t_out] = sum;
    }
}

/// Resamples a mono f32 buffer from sr_in to sr_out (audio_resample, nch=1).
/// Same-rate returns a copy. Output length = ceil(sr_out*n_in/sr_in) computed
/// in f64, exactly as the reference. Caller owns the returned slice.
pub fn resample(allocator: std.mem.Allocator, in: []const f32, sr_in: u32, sr_out: u32) ![]f32 {
    if (in.len == 0 or sr_in == 0 or sr_out == 0) return error.InvalidInput;

    if (sr_in == sr_out) return allocator.dupe(f32, in);

    const g = gcd(sr_in, sr_out);
    const orig = sr_in / g;
    const newf = sr_out / g;

    var kernel = try buildKernel(allocator, orig, newf);
    defer kernel.deinit(allocator);

    const target_f = @ceil(@as(f64, @floatFromInt(sr_out)) * @as(f64, @floatFromInt(in.len)) / @as(f64, @floatFromInt(sr_in)));
    const target: usize = @intFromFloat(target_f);
    if (target == 0) return error.InvalidInput;

    const out = try allocator.alloc(f32, target);
    errdefer allocator.free(out);
    try applyMono(allocator, in, orig, newf, &kernel, out);
    return out;
}

test {
    _ = @import("resample_tests.zig");
}
