//! Hermetic tests for the parakeet mel front-end: in-memory WAV decode +
//! preemphasis. No dependency on the (gitignored) audio fixtures.
const std = @import("std");
const fe = @import("frontend.zig");

/// Hand-build a 16-bit PCM WAV image (interleaved i16) for testing.
fn buildWavPcm16(allocator: std.mem.Allocator, samples: []const i16, channels: u16, sr: u32) ![]u8 {
    const data_len = samples.len * 2;
    const out = try allocator.alloc(u8, 44 + data_len);
    @memcpy(out[0..4], "RIFF");
    std.mem.writeInt(u32, out[4..8], @intCast(out.len - 8), .little);
    @memcpy(out[8..12], "WAVE");
    @memcpy(out[12..16], "fmt ");
    std.mem.writeInt(u32, out[16..20], 16, .little);
    std.mem.writeInt(u16, out[20..22], 1, .little); // PCM
    std.mem.writeInt(u16, out[22..24], channels, .little);
    std.mem.writeInt(u32, out[24..28], sr, .little);
    const block_align: u16 = channels * 2;
    std.mem.writeInt(u32, out[28..32], sr * block_align, .little);
    std.mem.writeInt(u16, out[32..34], block_align, .little);
    std.mem.writeInt(u16, out[34..36], 16, .little);
    @memcpy(out[36..40], "data");
    std.mem.writeInt(u32, out[40..44], @intCast(data_len), .little);
    for (samples, 0..) |s, i| std.mem.writeInt(i16, out[44 + i * 2 ..][0..2], s, .little);
    return out;
}

test "front-end: decode 16k mono 16-bit PCM, normalized by 32768" {
    const allocator = std.testing.allocator;
    const pcm = [_]i16{ 0, 16384, -32768, 8192 };
    const wav = try buildWavPcm16(allocator, &pcm, 1, 16000);
    defer allocator.free(wav);

    var audio = try fe.loadWav16kMono(allocator, wav);
    defer audio.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 16000), audio.sample_rate);
    try std.testing.expectEqual(@as(usize, 4), audio.samples.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), audio.samples[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), audio.samples[1], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), audio.samples[2], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), audio.samples[3], 1e-9);
}

test "front-end: stereo downmix averages channels" {
    const allocator = std.testing.allocator;
    // frame0: L=16384(0.5), R=-16384(-0.5) -> 0 ; frame1: L=16384(0.5), R=0 -> 0.25
    const pcm = [_]i16{ 16384, -16384, 16384, 0 };
    const wav = try buildWavPcm16(allocator, &pcm, 2, 16000);
    defer allocator.free(wav);

    var audio = try fe.loadWav16kMono(allocator, wav);
    defer audio.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), audio.samples.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), audio.samples[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), audio.samples[1], 1e-9);
}

test "front-end: non-16k sample rate is resampled to 16k (P10.5)" {
    const allocator = std.testing.allocator;
    const pcm = [_]i16{ 1, 2, 3 };
    const wav = try buildWavPcm16(allocator, &pcm, 1, 44100);
    defer allocator.free(wav);
    const audio = try fe.loadWav16kMono(allocator, wav); // 44.1k -> 16k (no longer rejected)
    defer allocator.free(audio.samples);
    try std.testing.expectEqual(@as(u32, 16000), audio.sample_rate);
    // n_out = floor(3 * 16000/44100) = floor(1.088) = 1
    try std.testing.expectEqual(@as(usize, 1), audio.samples.len);
}

test "front-end: bad magic rejected" {
    const allocator = std.testing.allocator;
    var junk = [_]u8{0} ** 16;
    try std.testing.expectError(fe.Error.NotWav, fe.loadWav16kMono(allocator, &junk));
}

test "front-end: preemphasis keeps first sample, then y[n]=x[n]-c*x[n-1]" {
    const in = [_]f32{ 1.0, 2.0, 3.0, -1.0 };
    var out: [4]f32 = undefined;
    try fe.preemphasis(&in, &out, 0.97);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6); // unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 - 0.97 * 1.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0 - 0.97 * 2.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0 - 0.97 * 3.0), out[3], 1e-6);
}

test "front-end: preemphasis is in-place safe (out aliases in)" {
    const in = [_]f32{ 0.5, -0.25, 0.125, 1.0, -0.75 };
    var ref: [5]f32 = undefined;
    try fe.preemphasis(&in, &ref, 0.97);

    var buf: [5]f32 = in;
    try fe.preemphasis(&buf, &buf, 0.97); // alias
    try std.testing.expectEqualSlices(f32, &ref, &buf);
}

test "front-end: preemphasis with zero coeff is identity" {
    const in = [_]f32{ 0.1, 0.2, 0.3 };
    var out: [3]f32 = undefined;
    try fe.preemphasis(&in, &out, 0.0);
    try std.testing.expectEqualSlices(f32, &in, &out);
}

test "front-end: preemphasis rejects mismatched output length" {
    const in = [_]f32{ 0.1, 0.2, 0.3 };
    var out: [2]f32 = undefined;
    try std.testing.expectError(fe.Error.InvalidMelParameters, fe.preemphasis(&in, &out, 0.97));
}

test "stft: analytic impulse case (n_fft=4, hop=2, rect window)" {
    const allocator = std.testing.allocator;
    // samples=[1,0,0,0], pad=2 -> padded=[0,0,1,0,0,0,0,0], T=1+(8-4)/2=3.
    // frame0=[0,0,1,0] -> |X[b]|=1 all bins; frame1=[1,0,0,0] -> |X[b]|=1;
    // frame2=[0,0,0,0] -> 0. mag_power=2 -> power = mag^2.
    const samples = [_]f32{ 1, 0, 0, 0 };
    const window = [_]f32{ 1, 1, 1, 1 };
    var spec = try fe.stftPower(allocator, &samples, .{
        .n_fft = 4,
        .hop = 2,
        .win_length = 4,
        .mag_power = 2.0,
        .preemph = 0.0,
    }, &window);
    defer spec.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), spec.n_frames);
    try std.testing.expectEqual(@as(usize, 3), spec.n_bins);
    const expected = [_]f32{ 1, 1, 1, 1, 1, 1, 0, 0, 0 };
    for (expected, 0..) |e, i|
        try std.testing.expectApproxEqAbs(e, spec.power[i], 1e-4);
}

test "stft: malformed parameters return errors in release modes" {
    const allocator = std.testing.allocator;
    const samples = [_]f32{ 1, 0, 0, 0 };
    const window = [_]f32{ 1, 1, 1, 1 };

    try std.testing.expectError(fe.Error.InvalidMelParameters, fe.stftPower(allocator, &samples, .{
        .n_fft = 4,
        .hop = 0,
        .win_length = 4,
        .mag_power = 2.0,
        .preemph = 0.0,
    }, &window));
    try std.testing.expectError(fe.Error.InvalidMelParameters, fe.stftPower(allocator, &samples, .{
        .n_fft = 4,
        .hop = 2,
        .win_length = 5,
        .mag_power = 2.0,
        .preemph = 0.0,
    }, &window));
    try std.testing.expectError(fe.Error.InvalidMelParameters, fe.stftPower(allocator, &samples, .{
        .n_fft = 4,
        .hop = 2,
        .win_length = 3,
        .mag_power = 2.0,
        .preemph = 0.0,
    }, &window));
}

test "stft: frame count matches NeMo center=True formula" {
    const allocator = std.testing.allocator;
    const window = [_]f32{ 1, 1, 1, 1, 1, 1 }; // win_length=6
    const s: usize = 50;
    const samples = try allocator.alloc(f32, s);
    defer allocator.free(samples);
    for (samples, 0..) |*v, i| v.* = @floatCast(std.math.sin(0.3 * @as(f64, @floatFromInt(i))));
    var spec = try fe.stftPower(allocator, samples, .{
        .n_fft = 8,
        .hop = 4,
        .win_length = 6,
        .mag_power = 2.0,
        .preemph = 0.97,
    }, &window);
    defer spec.deinit(allocator);
    // pad=4, padded_len=58, T=1+(58-8)/4 = 1+12 = 13.
    try std.testing.expectEqual(@as(usize, 13), spec.n_frames);
    try std.testing.expectEqual(@as(usize, 5), spec.n_bins);
}

// Independent, table-free reference (recomputes cos/sin inline per term) — catches
// basis-table indexing / sign / center-pad bugs in the optimized stftPower.
fn naivePower(allocator: std.mem.Allocator, samples: []const f32, p: fe.StftParams, window: []const f32) ![]f32 {
    const n_fft = p.n_fft;
    const hop = p.hop;
    const nb = n_fft / 2 + 1;
    const s = samples.len;
    const x = try allocator.alloc(f64, s);
    defer allocator.free(x);
    if (p.preemph > 0) {
        x[0] = samples[0];
        for (1..s) |i| x[i] = @as(f64, samples[i]) - @as(f64, p.preemph) * @as(f64, samples[i - 1]);
    } else for (samples, 0..) |v, i| {
        x[i] = v;
    }
    const pad = n_fft / 2;
    const plen = s + 2 * pad;
    const padded = try allocator.alloc(f64, plen);
    defer allocator.free(padded);
    @memset(padded, 0);
    for (0..s) |j| padded[pad + j] = x[j];
    const T = if (plen >= n_fft) 1 + (plen - n_fft) / hop else 0;
    const win = try allocator.alloc(f32, n_fft);
    defer allocator.free(win);
    @memset(win, 0);
    const left = (n_fft - p.win_length) / 2;
    for (0..p.win_length) |i| win[left + i] = window[i];
    const power = try allocator.alloc(f32, T * nb);
    for (0..T) |t| {
        for (0..nb) |b| {
            var re: f64 = 0;
            var im: f64 = 0;
            for (0..n_fft) |i| {
                const fr: f32 = @floatCast(padded[t * hop + i] * @as(f64, win[i]));
                const ang = 2.0 * std.math.pi * @as(f64, @floatFromInt(b * i)) / @as(f64, @floatFromInt(n_fft));
                re += @as(f64, fr) * std.math.cos(ang);
                im -= @as(f64, fr) * std.math.sin(ang);
            }
            const re32: f32 = @floatCast(re);
            const im32: f32 = @floatCast(im);
            const mag = @sqrt(@as(f64, re32) * re32 + @as(f64, im32) * im32);
            power[t * nb + b] = @floatCast(if (p.mag_power == 1.0) mag else std.math.pow(f64, mag, @as(f64, p.mag_power)));
        }
    }
    return power;
}

test "mel: projection + log (no norm) on the analytic impulse spectrum" {
    const allocator = std.testing.allocator;
    // Same setup as the stft impulse test -> power = [[1,1,1],[1,1,1],[0,0,0]].
    const samples = [_]f32{ 1, 0, 0, 0 };
    const window = [_]f32{ 1, 1, 1, 1 };
    // fb [n_mels=2, n_bins=3]: m0=[1,0,0], m1=[0,1,1].
    const fb = [_]f32{ 1, 0, 0, 0, 1, 1 };
    const guard: f32 = 1e-5;
    var mel = try fe.melSpectrogram(allocator, &samples, .{
        .stft = .{ .n_fft = 4, .hop = 2, .win_length = 4, .mag_power = 2.0, .preemph = 0.0 },
        .n_mels = 2,
        .log_guard = guard,
        .normalize_per_feature = false,
    }, &fb, &window);
    defer mel.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), mel.n_mels);
    try std.testing.expectEqual(@as(usize, 3), mel.n_frames);
    const ln = std.math.log;
    // feat-major feats[m*T + t], T=3.
    const e = [_]f32{
        @floatCast(ln(f64, std.math.e, 1.0 + 1e-5)), @floatCast(ln(f64, std.math.e, 1.0 + 1e-5)), @floatCast(ln(f64, std.math.e, 1e-5)),
        @floatCast(ln(f64, std.math.e, 2.0 + 1e-5)), @floatCast(ln(f64, std.math.e, 2.0 + 1e-5)), @floatCast(ln(f64, std.math.e, 1e-5)),
    };
    for (e, 0..) |x, i| try std.testing.expectApproxEqAbs(x, mel.feats[i], 1e-4);
}

test "mel: malformed filterbank shape returns an error" {
    const allocator = std.testing.allocator;
    const samples = [_]f32{ 1, 0, 0, 0 };
    const window = [_]f32{ 1, 1, 1, 1 };
    const fb = [_]f32{ 1, 0, 0, 0, 1 }; // expected 2 * 3
    try std.testing.expectError(fe.Error.InvalidMelParameters, fe.melSpectrogram(allocator, &samples, .{
        .stft = .{ .n_fft = 4, .hop = 2, .win_length = 4, .mag_power = 2.0, .preemph = 0.0 },
        .n_mels = 2,
        .log_guard = 1e-5,
        .normalize_per_feature = false,
    }, &fb, &window));
}

test "mel: per-feature norm masks frames >= seq_len and zero-means the rest" {
    const allocator = std.testing.allocator;
    const samples = [_]f32{ 0.5, -0.3, 0.8, -0.1, 0.2, 0.6, -0.4, 0.9 }; // S=8
    const window = [_]f32{ 1, 1, 1, 1 };
    const fb = [_]f32{ 1, 1, 0, 0, 1, 1 }; // n_mels=2, n_bins=3
    var mel = try fe.melSpectrogram(allocator, &samples, .{
        .stft = .{ .n_fft = 4, .hop = 2, .win_length = 4, .mag_power = 2.0, .preemph = 0.97 },
        .n_mels = 2,
        .log_guard = 1e-5,
        .normalize_per_feature = true,
    }, &fb, &window);
    defer mel.deinit(allocator);

    // T = 1 + (8+4-4)/2 = 5 ; seq_len = 8/2 = 4 ; valid = 4 -> frame 4 masked.
    try std.testing.expectEqual(@as(usize, 5), mel.n_frames);
    const T = mel.n_frames;
    const valid = 4;
    for (0..mel.n_mels) |m| {
        const row = mel.feats[m * T ..][0..T];
        try std.testing.expectEqual(@as(f32, 0.0), row[T - 1]); // masked
        var mean: f64 = 0;
        for (0..valid) |t| mean += row[t];
        mean /= valid;
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), mean, 1e-5); // zero-mean
    }
}

test "stft: matches an independent inline-DFT reference (windowed, preemph)" {
    const allocator = std.testing.allocator;
    const window = [_]f32{ 0.1, 0.5, 0.9, 1.0, 0.9, 0.5 }; // win_length=6, asymmetric edges
    const s: usize = 40;
    const samples = try allocator.alloc(f32, s);
    defer allocator.free(samples);
    for (samples, 0..) |*v, i| v.* = @floatCast(0.3 * std.math.sin(0.21 * @as(f64, @floatFromInt(i))) + 0.1 * @as(f64, @floatFromInt(i % 7)));
    const p = fe.StftParams{ .n_fft = 16, .hop = 5, .win_length = 6, .mag_power = 2.0, .preemph = 0.97 };

    var spec = try fe.stftPower(allocator, samples, p, &window);
    defer spec.deinit(allocator);
    const ref = try naivePower(allocator, samples, p, &window);
    defer allocator.free(ref);

    try std.testing.expectEqual(ref.len, spec.power.len);
    for (ref, 0..) |r, i|
        try std.testing.expectApproxEqAbs(r, spec.power[i], 1e-5);
}

test "resampleLinear: upsample 2x, downsample 0.5x, identity (matches resample_linear)" {
    const a = std.testing.allocator;
    // upsample [0,1,2,3] 4->8: ratio 2, n_out=8, linear interp.
    const up = try fe.resampleLinear(a, &.{ 0, 1, 2, 3 }, 4, 8);
    defer a.free(up);
    try std.testing.expectEqual(@as(usize, 8), up.len);
    const exp_up = [_]f32{ 0, 0.5, 1, 1.5, 2, 2.5, 3, 3 }; // last clamps to in[3]
    for (exp_up, 0..) |e, i| try std.testing.expectApproxEqAbs(e, up[i], 1e-6);
    // downsample [0,2,4,6] 8->4: ratio 0.5, n_out=2 -> [0,4].
    const down = try fe.resampleLinear(a, &.{ 0, 2, 4, 6 }, 8, 4);
    defer a.free(down);
    try std.testing.expectEqual(@as(usize, 2), down.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0), down[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), down[1], 1e-6);
    // identity -> owned copy.
    const same = try fe.resampleLinear(a, &.{ 1, 2, 3 }, 16000, 16000);
    defer a.free(same);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, same);
}

test "loadWav16kMono: 8 kHz stereo PCM16 -> downmix + resample to 16 kHz" {
    const a = std.testing.allocator;
    // 4 stereo frames at 8 kHz; mono = channel mean, then 8k->16k doubles the count.
    const wav = try buildWavPcm16(a, &.{ 100, 300, 200, 400, 300, 500, 400, 600 }, 2, 8000);
    defer a.free(wav);
    const audio = try fe.loadWav16kMono(a, wav);
    defer a.free(audio.samples);
    try std.testing.expectEqual(@as(u32, 16000), audio.sample_rate);
    try std.testing.expectEqual(@as(usize, 8), audio.samples.len); // 4 frames * 2 (8k->16k)
    // first mono sample = mean(100,300)/32768
    try std.testing.expectApproxEqAbs(@as(f32, 200.0 / 32768.0), audio.samples[0], 1e-6);
}
