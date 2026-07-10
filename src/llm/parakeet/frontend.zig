//! Parakeet mel front-end: WAV decode (→ mono 16 kHz f32 in [-1,1)) + NeMo
//! preemphasis, STFT power spectrum, mel filterbank + log + per-feature
//! normalization.
//!
//! Matches NeMo `AudioToMelSpectrogramPreprocessor` / parakeet.cpp
//! `src/audio_io.cpp` + `src/mel.cpp`. The WAV reader mirrors the normalization
//! of the NAM example's `wav.zig` (int / 2^(bits-1)); it is reimplemented here
//! (small, focused on 16 kHz mono) because that file is example-scoped and not
//! importable from `src/llm`.
const std = @import("std");
const fucina = @import("fucina");
const ExecContext = fucina.ExecContext;

pub const Error = error{
    NotWav,
    CorruptWav,
    UnsupportedWavFormat,
    InvalidMelParameters,
    UnsupportedSampleRate,
};

/// Decoded mono PCM as f32 in [-1, 1). Caller owns `samples`.
pub const Audio = struct {
    samples: []f32,
    sample_rate: u32,

    pub fn deinit(self: *Audio, allocator: std.mem.Allocator) void {
        allocator.free(self.samples);
        self.* = undefined;
    }
};

const FormatTag = enum { pcm, ieee_float };

/// Linear resample `in` from `in_sr` to `out_sr` (byte-for-byte port of
/// parakeet.cpp `audio_io.cpp::resample_linear`): `n_out = floor(in.len * out/in)`,
/// `out[i] = a + (b-a)*frac` with `src = i/ratio`, `i0 = floor(src)`. The `(b-a)`
/// is an f32 subtraction (then promoted to f64 for the multiply) to match the C++.
/// Returns an owned copy even when `in_sr == out_sr`.
pub fn resampleLinear(allocator: std.mem.Allocator, in: []const f32, in_sr: usize, out_sr: usize) ![]f32 {
    if (in_sr == out_sr or in.len == 0) return allocator.dupe(f32, in);
    const ratio = @as(f64, @floatFromInt(out_sr)) / @as(f64, @floatFromInt(in_sr));
    const n_out: usize = @intFromFloat(@floor(@as(f64, @floatFromInt(in.len)) * ratio));
    const out = try allocator.alloc(f32, n_out);
    errdefer allocator.free(out);
    var i: usize = 0;
    while (i < n_out) : (i += 1) {
        const src = @as(f64, @floatFromInt(i)) / ratio;
        const idx0: usize = @intFromFloat(src); // floor (src >= 0)
        const frac = src - @as(f64, @floatFromInt(idx0));
        const a = in[idx0];
        const b = if (idx0 + 1 < in.len) in[idx0 + 1] else a;
        const diff: f32 = b - a; // f32 subtraction (matches C++ `(b - a)`)
        out[i] = @floatCast(@as(f64, a) + @as(f64, diff) * frac);
    }
    return out;
}

/// Decode a WAV file image to mono f32 [-1,1) at 16 kHz. Multi-channel
/// audio is downmixed by averaging (`mono[i] = acc / channels`), then linearly
/// resampled to 16 kHz if needed (matching parakeet.cpp `load_audio_16k_mono`:
/// decode → mono → `resample_linear`). Supports PCM 16/24/32-bit int and 32-bit
/// IEEE float (incl. WAVE_FORMAT_EXTENSIBLE).
pub fn loadWav16kMono(allocator: std.mem.Allocator, bytes: []const u8) !Audio {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE"))
        return Error.NotWav;

    var format: ?FormatTag = null;
    var bits: u16 = 0;
    var channels: u16 = 0;
    var sample_rate: u32 = 0;
    var data: ?[]const u8 = null;

    var pos: usize = 12;
    while (pos + 8 <= bytes.len) {
        const chunk_id = bytes[pos .. pos + 4];
        const chunk_len: usize = std.mem.readInt(u32, bytes[pos + 4 ..][0..4], .little);
        pos += 8;
        if (chunk_len > bytes.len - pos) return Error.CorruptWav;
        const chunk = bytes[pos .. pos + chunk_len];

        if (std.mem.eql(u8, chunk_id, "fmt ")) {
            if (chunk.len < 16) return Error.CorruptWav;
            var tag = std.mem.readInt(u16, chunk[0..2], .little);
            channels = std.mem.readInt(u16, chunk[2..4], .little);
            sample_rate = std.mem.readInt(u32, chunk[4..8], .little);
            bits = std.mem.readInt(u16, chunk[14..16], .little);
            if (tag == 0xFFFE) { // WAVE_FORMAT_EXTENSIBLE: real tag in SubFormat GUID
                if (chunk.len < 26) return Error.CorruptWav;
                tag = std.mem.readInt(u16, chunk[24..26], .little);
            }
            format = switch (tag) {
                1 => .pcm,
                3 => .ieee_float,
                else => return Error.UnsupportedWavFormat,
            };
        } else if (std.mem.eql(u8, chunk_id, "data")) {
            data = chunk;
        }
        pos += chunk_len + (chunk_len & 1); // chunks are word-aligned
    }

    const fmt = format orelse return Error.CorruptWav;
    const payload = data orelse return Error.CorruptWav;
    if (channels == 0 or sample_rate == 0) return Error.CorruptWav;

    const bytes_per_sample: usize = switch (fmt) {
        .pcm => switch (bits) {
            16 => 2,
            24 => 3,
            32 => 4,
            else => return Error.UnsupportedWavFormat,
        },
        .ieee_float => switch (bits) {
            32 => 4,
            else => return Error.UnsupportedWavFormat,
        },
    };
    const ch: usize = channels;
    const total_samples = payload.len / bytes_per_sample;
    const frames = total_samples / ch;

    const mono = try allocator.alloc(f32, frames);
    errdefer allocator.free(mono);

    var f: usize = 0;
    while (f < frames) : (f += 1) {
        var acc: f64 = 0;
        var c: usize = 0;
        while (c < ch) : (c += 1) {
            const i = f * ch + c;
            acc += decodeSample(fmt, bits, payload, i);
        }
        mono[f] = @floatCast(acc / @as(f64, @floatFromInt(ch)));
    }

    if (sample_rate == 16000) return .{ .samples = mono, .sample_rate = 16000 };
    defer allocator.free(mono);
    const resampled = try resampleLinear(allocator, mono, sample_rate, 16000);
    return .{ .samples = resampled, .sample_rate = 16000 };
}

fn decodeSample(fmt: FormatTag, bits: u16, payload: []const u8, i: usize) f64 {
    return switch (fmt) {
        .pcm => switch (bits) {
            16 => @as(f64, @floatFromInt(std.mem.readInt(i16, payload[i * 2 ..][0..2], .little))) / 32768.0,
            24 => blk: {
                const lo: u32 = payload[i * 3];
                const mid: u32 = payload[i * 3 + 1];
                const hi: u32 = payload[i * 3 + 2];
                const raw: u32 = lo | (mid << 8) | (hi << 16);
                const v: i32 = if (raw & 0x800000 != 0) @bitCast(raw | 0xFF000000) else @bitCast(raw);
                break :blk @as(f64, @floatFromInt(v)) / 8388608.0;
            },
            32 => @as(f64, @floatFromInt(std.mem.readInt(i32, payload[i * 4 ..][0..4], .little))) / 2147483648.0,
            else => unreachable,
        },
        .ieee_float => blk: {
            const raw = std.mem.readInt(u32, payload[i * 4 ..][0..4], .little);
            const fv: f32 = @bitCast(raw);
            break :blk @as(f64, fv);
        },
    };
}

/// Read a WAV file from disk and decode it (mono 16 kHz f32).
pub fn loadWav16kMonoFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Audio {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return Error.NotWav;
    const len: usize = @intCast(stat.size);
    const bytes = try allocator.alloc(u8, len);
    defer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return loadWav16kMono(allocator, bytes);
}

/// NeMo preemphasis: the first sample is kept unchanged, then
/// `y[n] = x[n] - coeff * x[n-1]` (parakeet.cpp `src/mel.cpp` computes the
/// subtraction in double — followed here so the value feeds the mel stage
/// parity-faithfully). In-place safe (`out` may alias `in`); `out.len == in.len`.
pub fn preemphasis(in: []const f32, out: []f32, coeff: f32) !void {
    if (out.len != in.len) return Error.InvalidMelParameters;
    if (in.len == 0) return;
    out[0] = in[0];
    var prev: f64 = in[0];
    var n: usize = 1;
    while (n < in.len) : (n += 1) {
        const cur: f64 = in[n];
        out[n] = @floatCast(cur - @as(f64, coeff) * prev);
        prev = cur;
    }
}

// --- STFT power spectrum ---

pub const StftParams = struct {
    n_fft: usize,
    hop: usize,
    win_length: usize,
    mag_power: f32,
    preemph: f32,
};

/// Per-frame power spectrum, frame-major: `power[t * n_bins + b]`.
pub const Spectrogram = struct {
    power: []f32,
    n_frames: usize,
    n_bins: usize,

    pub fn deinit(self: *Spectrogram, allocator: std.mem.Allocator) void {
        allocator.free(self.power);
        self.* = undefined;
    }
};

pub const DftBasis = struct {
    allocator: std.mem.Allocator,
    n_fft: usize,
    n_bins: usize,
    cosT: []f32,
    nsinT: []f32,

    pub fn init(allocator: std.mem.Allocator, n_fft: usize) !DftBasis {
        if (n_fft == 0) return Error.InvalidMelParameters;
        const n_bins = n_fft / 2 + 1;
        const len = try std.math.mul(usize, n_fft, n_bins);
        const cosT = try allocator.alloc(f32, len);
        errdefer allocator.free(cosT);
        const nsinT = try allocator.alloc(f32, len);
        errdefer allocator.free(nsinT);
        fillDftBasis(n_fft, n_bins, cosT, nsinT);
        return .{ .allocator = allocator, .n_fft = n_fft, .n_bins = n_bins, .cosT = cosT, .nsinT = nsinT };
    }

    pub fn deinit(self: *DftBasis) void {
        self.allocator.free(self.cosT);
        self.allocator.free(self.nsinT);
        self.* = undefined;
    }
};

fn fillDftBasis(n_fft: usize, n_bins: usize, cosT: []f32, nsinT: []f32) void {
    for (0..n_fft) |i| {
        for (0..n_bins) |b| {
            const angle = 2.0 * std.math.pi * @as(f64, @floatFromInt(b * i)) / @as(f64, @floatFromInt(n_fft));
            cosT[i * n_bins + b] = @floatCast(std.math.cos(angle));
            nsinT[i * n_bins + b] = @floatCast(-std.math.sin(angle));
        }
    }
}

/// STFT power spectrum matching NeMo `AudioToMelSpectrogramPreprocessor` /
/// parakeet.cpp (`src/mel.cpp` + `src/fft.cpp`):
///   - preemphasis in double (first sample kept);
///   - center pad with n_fft/2 ZEROS each side (NeMo pad_mode="constant", NOT
///     reflect), frame count `1 + (S + 2*pad - n_fft)/hop`;
///   - windowed frame `f32(padded_f64 * window_f64)`, window = the win_length
///     Hann (from the GGUF) center-padded to n_fft (left = (n_fft-win_length)/2);
///   - DFT in double; re/im truncated to f32 (parakeet's rfft casts its output);
///   - `power[b] = mag^mag_power`, `mag = sqrt(re^2+im^2)` in double.
/// A direct (gather) DFT — correctness first; swap for an FFT if it's ever the
/// bottleneck. `window` is the raw win_length-length window.
pub fn stftPower(
    allocator: std.mem.Allocator,
    samples: []const f32,
    p: StftParams,
    window: []const f32,
) !Spectrogram {
    return stftPowerImpl(allocator, samples, p, window, null);
}

fn stftPowerImpl(
    allocator: std.mem.Allocator,
    samples: []const f32,
    p: StftParams,
    window: []const f32,
    basis: ?*const DftBasis,
) !Spectrogram {
    const n_fft = p.n_fft;
    const hop = p.hop;
    const n_bins = n_fft / 2 + 1;
    if (n_fft == 0 or hop == 0 or p.win_length == 0 or p.win_length > n_fft) return Error.InvalidMelParameters;
    if (window.len != p.win_length) return Error.InvalidMelParameters;

    const s = samples.len;
    if (s == 0) return .{ .power = try allocator.alloc(f32, 0), .n_frames = 0, .n_bins = n_bins };

    // Preemphasis in double (parakeet keeps x[] as double).
    const x = try allocator.alloc(f64, s);
    defer allocator.free(x);
    if (p.preemph > 0.0) {
        x[0] = samples[0];
        var t: usize = 1;
        while (t < s) : (t += 1) x[t] = @as(f64, samples[t]) - @as(f64, p.preemph) * @as(f64, samples[t - 1]);
    } else {
        for (samples, 0..) |v, i| x[i] = v;
    }

    // Center zero-pad.
    const pad = n_fft / 2;
    const padded_len = try std.math.add(usize, s, try std.math.mul(usize, 2, pad));
    const padded = try allocator.alloc(f64, padded_len);
    defer allocator.free(padded);
    @memset(padded, 0);
    for (0..s) |j| padded[pad + j] = x[j];

    const n_frames: usize = if (padded_len >= n_fft) 1 + (padded_len - n_fft) / hop else 0;

    // Center-pad the window to n_fft (f32, matching the float window_ in mel.cpp).
    const win = try allocator.alloc(f32, n_fft);
    defer allocator.free(win);
    @memset(win, 0);
    const left = (n_fft - p.win_length) / 2;
    for (0..p.win_length) |i| win[left + i] = window[i];

    // The direct DFT becomes a matmul. Build the windowed frames
    // `[T, n_fft]` and the DFT basis as `[n_fft, n_bins]` (cos and -sin, computed
    // in double, stored f32), then `re = windowed @ cosᵀ`, `im = windowed @ (-sin)ᵀ`
    // via the threaded `ctx.matmul2D` (m=T is large, so it parallelizes well). re/im
    // land in f32 — parakeet's rfft casts its output to f32 too — and the magnitude
    // is finished in double, matching the reference's structure.
    const windowed_len = try std.math.mul(usize, n_frames, n_fft);
    const windowed = try allocator.alloc(f32, windowed_len);
    defer allocator.free(windowed);
    for (0..n_frames) |t| {
        const start = t * hop;
        for (0..n_fft) |i| windowed[t * n_fft + i] = @floatCast(padded[start + i] * @as(f64, win[i]));
    }

    var owned_cosT: ?[]f32 = null;
    var owned_nsinT: ?[]f32 = null;
    const cosT, const nsinT = if (basis) |bp| blk: {
        if (bp.n_fft != n_fft or bp.n_bins != n_bins) return Error.InvalidMelParameters;
        break :blk .{ bp.cosT, bp.nsinT };
    } else blk: {
        const basis_len = try std.math.mul(usize, n_fft, n_bins);
        owned_cosT = try allocator.alloc(f32, basis_len);
        errdefer allocator.free(owned_cosT.?);
        owned_nsinT = try allocator.alloc(f32, basis_len);
        errdefer allocator.free(owned_nsinT.?);
        fillDftBasis(n_fft, n_bins, owned_cosT.?, owned_nsinT.?);
        break :blk .{ owned_cosT.?, owned_nsinT.? };
    };
    defer if (owned_cosT) |buf| allocator.free(buf);
    defer if (owned_nsinT) |buf| allocator.free(buf);

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var wt = try fucina.Tensor(2).fromBorrowedConstSlice(&ctx, .{ n_frames, n_fft }, windowed);
    defer wt.deinit();
    var ct = try fucina.Tensor(2).fromBorrowedConstSlice(&ctx, .{ n_fft, n_bins }, cosT);
    defer ct.deinit();
    var nst = try fucina.Tensor(2).fromBorrowedConstSlice(&ctx, .{ n_fft, n_bins }, nsinT);
    defer nst.deinit();
    // Public facade matmul (.plain NN) — same threaded f32 GEMM kernel.
    var re_t = try wt.matmul(&ctx, ct, .plain, 2); // [T, n_bins]
    defer re_t.deinit();
    var im_t = try wt.matmul(&ctx, nst, .plain, 2); // [T, n_bins]
    defer im_t.deinit();
    const re = try re_t.dataConst();
    const im = try im_t.dataConst();

    const power_len = try std.math.mul(usize, n_frames, n_bins);
    const power = try allocator.alloc(f32, power_len);
    errdefer allocator.free(power);
    const mag_power: f64 = p.mag_power;
    for (0..power_len) |idx| {
        const r: f64 = re[idx];
        const m: f64 = im[idx];
        const mag = @sqrt(r * r + m * m);
        power[idx] = @floatCast(if (p.mag_power == 1.0) mag else std.math.pow(f64, mag, mag_power));
    }

    return .{ .power = power, .n_frames = n_frames, .n_bins = n_bins };
}

// --- Mel filterbank + log + per-feature normalization ---

pub const MelParams = struct {
    stft: StftParams,
    n_mels: usize,
    log_guard: f32,
    normalize_per_feature: bool,
};

/// Log-mel features, feat-major: `feats[m * n_frames + t]` (mel m, frame t).
pub const MelSpectrogram = struct {
    feats: []f32,
    n_mels: usize,
    n_frames: usize,

    pub fn deinit(self: *MelSpectrogram, allocator: std.mem.Allocator) void {
        allocator.free(self.feats);
        self.* = undefined;
    }
};

/// CONSTANT added to std in NeMo normalize_batch (features.py: CONSTANT = 1e-5).
const norm_eps: f64 = 1e-5;

fn normalizePerFeature(ctx: *ExecContext, feats: []f32, n_mels: usize, n_frames: usize, valid: usize) !void {
    var features = try fucina.Tensor(2).fromBorrowedSlice(ctx, .{ n_mels, n_frames }, feats);
    defer features.deinit();
    var normalized = try features.standardizeAxis(ctx, ._1, .{
        .valid_len = valid,
        .ddof = 1,
        .eps = @as(f32, @floatCast(norm_eps)),
        .accumulation = .f64,
    });
    defer normalized.deinit();
    try normalized.copyTo(feats);
}

/// Full NeMo log-mel front end (parakeet.cpp `MelFrontend::compute`):
///   feats[m,t] = log( Σ_b fb[m,b]·power[b,t] + log_guard )   (acc in double)
/// then per-feature (per mel-channel) normalization over the first
/// `seq_len = floor(S/hop)` frames: subtract the row mean, divide by the unbiased
/// (ddof=1) std + 1e-5; frames ≥ seq_len are zeroed (valid_mask). `fb` is the GGUF
/// filterbank `fb[m*n_bins + b]`; `window` the raw win_length Hann window.
pub fn melSpectrogram(
    allocator: std.mem.Allocator,
    samples: []const f32,
    p: MelParams,
    fb: []const f32,
    window: []const f32,
) !MelSpectrogram {
    var spec = try stftPower(allocator, samples, p.stft, window);
    defer spec.deinit(allocator);
    const T = spec.n_frames;
    const n_bins = spec.n_bins;
    const n_mels = p.n_mels;
    if (n_mels == 0) return Error.InvalidMelParameters;
    if (fb.len != try std.math.mul(usize, n_mels, n_bins)) return Error.InvalidMelParameters;

    const feats = try allocator.alloc(f32, try std.math.mul(usize, n_mels, T));
    errdefer allocator.free(feats);

    // Mel projection + log (acc in double), feat-major out[m*T + t].
    const log_guard: f64 = p.log_guard;
    for (0..T) |t| {
        const pw = spec.power[t * n_bins ..][0..n_bins];
        for (0..n_mels) |m| {
            const fbm = fb[m * n_bins ..][0..n_bins];
            var acc: f64 = 0;
            for (0..n_bins) |b| acc += @as(f64, fbm[b]) * @as(f64, pw[b]);
            feats[m * T + t] = @floatCast(@log(acc + log_guard));
        }
    }

    // Per-feature normalization (normalize_batch, per_feature, B=1).
    if (p.normalize_per_feature and T > 0) {
        const seq_len = samples.len / p.stft.hop;
        const valid = @min(seq_len, T);
        var ctx: ExecContext = undefined;
        ctx.init(allocator);
        defer ctx.deinit();
        try normalizePerFeature(&ctx, feats, n_mels, T, valid);
    }

    return .{ .feats = feats, .n_mels = n_mels, .n_frames = T };
}

pub fn melSpectrogramFast(
    allocator: std.mem.Allocator,
    samples: []const f32,
    p: MelParams,
    fb: []const f32,
    window: []const f32,
) !MelSpectrogram {
    return melSpectrogramFastImpl(allocator, samples, p, fb, window, null);
}

pub fn melSpectrogramFastWithBasis(
    allocator: std.mem.Allocator,
    samples: []const f32,
    p: MelParams,
    fb: []const f32,
    window: []const f32,
    basis: *const DftBasis,
) !MelSpectrogram {
    return melSpectrogramFastImpl(allocator, samples, p, fb, window, basis);
}

fn melSpectrogramFastImpl(
    allocator: std.mem.Allocator,
    samples: []const f32,
    p: MelParams,
    fb: []const f32,
    window: []const f32,
    basis: ?*const DftBasis,
) !MelSpectrogram {
    var spec = try stftPowerImpl(allocator, samples, p.stft, window, basis);
    defer spec.deinit(allocator);
    const T = spec.n_frames;
    const n_bins = spec.n_bins;
    const n_mels = p.n_mels;
    if (n_mels == 0) return Error.InvalidMelParameters;
    if (fb.len != try std.math.mul(usize, n_mels, n_bins)) return Error.InvalidMelParameters;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var pt = try fucina.Tensor(2).fromBorrowedConstSlice(&ctx, .{ T, n_bins }, spec.power);
    defer pt.deinit();
    var fbt = try fucina.Tensor(2).fromBorrowedConstSlice(&ctx, .{ n_mels, n_bins }, fb);
    defer fbt.deinit();
    // Public facade matmul (.trans_b) — same f32 GEMM kernel.
    var mel_t = try pt.matmul(&ctx, fbt, .trans_b, 2); // [T, n_mels]
    defer mel_t.deinit();
    const mel = try mel_t.dataConst();

    const feats = try allocator.alloc(f32, try std.math.mul(usize, n_mels, T));
    errdefer allocator.free(feats);
    const log_guard: f64 = p.log_guard;
    for (0..T) |t| {
        for (0..n_mels) |m| {
            feats[m * T + t] = @floatCast(@log(@as(f64, mel[t * n_mels + m]) + log_guard));
        }
    }

    if (p.normalize_per_feature and T > 0) {
        const seq_len = samples.len / p.stft.hop;
        const valid = @min(seq_len, T);
        try normalizePerFeature(&ctx, feats, n_mels, T, valid);
    }

    return .{ .feats = feats, .n_mels = n_mels, .n_frames = T };
}

test {
    _ = @import("frontend_tests.zig");
}
