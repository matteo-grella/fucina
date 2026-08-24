//! Async guitar tuner for the live loop. The audio callback only memcpys the
//! raw (pre-trim) input into a wait-free SPSC ring (Tap.push: one copy + one
//! release store — no locks, no allocation); everything else runs on a
//! dedicated analysis thread, so the tuner can never interfere with the
//! realtime NAM inference.
//!
//! The DSP is deliberately plain scalar f64 rather than fucina Tensor ops:
//! the working sets are 63-tap FIR dots and 2-4k-sample correlations at a
//! ~15 Hz cadence — far below the shapes where the pool-parallel Tensor
//! pipeline amortizes its dispatch/allocation — and the sub-0.1-cent
//! accuracy target needs f64 accumulation, which the f32-centric Tensor
//! facade would forfeit.
//!
//! Detection pipeline (Analyzer, pure and single-threaded for testability):
//!   1. DC blocker + 63-tap windowed-sinc FIR decimator to ~12 kHz (integer
//!      factor; anti-aliased). All analysis math is f64.
//!   2. Coarse f0 via the McLeod Pitch Method: NSDF (normalized square
//!      difference, prefix-sum normalization), key-maxima picking between
//!      zero crossings with the 0.9·max threshold, parabolic interpolation.
//!      Time-domain, so a weak fundamental with strong harmonics still yields
//!      the true period (no octave-up errors).
//!   3. Precision: each partial k·f0 (k = 1..6, inside the anti-alias band)
//!      is refined by an iterated 3-point parabolic search on the Hann-
//!      windowed |DFT|² evaluated at exact frequencies (oscillator-recurrence
//!      correlation, not FFT bins) — final search granularity ≪ 0.05 cent.
//!      A weighted least-squares fit of f_k² = f0²(1 + B·k²) across the
//!      measured partials estimates the string's inharmonicity B and removes
//!      the sharp bias higher partials would otherwise inject; the reported
//!      frequency is the actual first-partial frequency f0·sqrt(1+B).
//!   4. Display stability: median of the recent per-frame estimates plus
//!      2-frame hysteresis on the note name.
//!
//! Polyphonic strum check (PolyTune-style, standard tuning E A D G B e):
//!   per-string harmonic-salience scan over ±120 cents around the expected
//!   open frequency, fundamental refined like the mono path, then a
//!   low-to-high masking pass so a lower string's exact-ratio partials
//!   (E2·3 ≈ B3, A2·3 = E4, E2·4 = E4) don't read as a sounding higher
//!   string. Per-string accuracy target ±2 cents; the mono path is the
//!   precision instrument.

const std = @import("std");

pub const a4_default: f64 = 440.0;
/// Detectable fundamental range (Hz): low B0 (5-string bass, dropped
/// tunings) up to the 24th fret of a guitar's high e.
pub const min_f0: f64 = 26.0;
pub const max_f0: f64 = 1350.0;

// ---------------------------------------------------------------------------
// Note naming (12-TET around a configurable A4)
// ---------------------------------------------------------------------------

pub const note_names = [_][]const u8{ "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };

pub const NoteInfo = struct { midi: i32, cents: f64 };

/// Nearest equal-tempered note to `f` and the signed cents offset from it.
pub fn nearestNote(f: f64, a4: f64) NoteInfo {
    const semis = 69.0 + 12.0 * std.math.log2(f / a4);
    const midi: i32 = @intFromFloat(@round(semis));
    return .{ .midi = midi, .cents = (semis - @round(semis)) * 100.0 };
}

pub fn midiFreq(midi: i32, a4: f64) f64 {
    const exp = @as(f64, @floatFromInt(midi - 69)) / 12.0;
    return a4 * std.math.pow(f64, 2.0, exp);
}

/// "E2", "F#3", ... (empty on out-of-range midi).
pub fn noteLabel(midi: i32, buf: []u8) []const u8 {
    if (midi < 0 or midi > 127) return "";
    const pc: usize = @intCast(@mod(midi, 12));
    const octave = @divFloor(midi, 12) - 1;
    return std.fmt.bufPrint(buf, "{s}{d}", .{ note_names[pc], octave }) catch "";
}

pub fn centsBetween(f: f64, ref: f64) f64 {
    return 1200.0 * std.math.log2(f / ref);
}

// ---------------------------------------------------------------------------
// Tap: SPSC sample ring. Writer = realtime audio callback (wait-free: one or
// two memcpys + a release store; disabled = a single monotonic load). Reader
// = the tuner thread, which drops backlog to capacity/2 when it falls behind;
// the writer never blocks or checks the reader, so an extreme lag could in
// principle overwrite samples mid-read — at 48 kHz the writer needs ~340 ms
// to cross the guard gap while a full reader drain takes well under 1 ms, and
// a torn window would only yield one low-clarity frame that the median
// filter absorbs.
// ---------------------------------------------------------------------------

pub const Tap = struct {
    pub const capacity = 1 << 15; // power of two; 0.68 s @ 48 kHz
    ring: [capacity]f32 = undefined,
    write_pos: std.atomic.Value(usize) = .init(0),
    enabled: std.atomic.Value(bool) = .init(false),

    /// Realtime-safe append; drops silently when the tuner is off.
    pub fn push(self: *Tap, samples: []const f32) void {
        if (!self.enabled.load(.monotonic)) return;
        var pos = self.write_pos.load(.monotonic); // single writer
        var src = samples;
        if (src.len > capacity) { // pathological block: keep the newest window
            pos += src.len - capacity;
            src = src[src.len - capacity ..];
        }
        const i = pos & (capacity - 1);
        const first = @min(capacity - i, src.len);
        @memcpy(self.ring[i..][0..first], src[0..first]);
        @memcpy(self.ring[0 .. src.len - first], src[first..]);
        self.write_pos.store(pos + src.len, .release);
    }
};

// ---------------------------------------------------------------------------
// Analyzer: the DSP. No threads, no atomics — feed() raw samples, then call
// analyzeMono()/analyzePoly(). Owned buffers only (no allocator), so the
// whole thing lives in one heap allocation inside Tuner.
// ---------------------------------------------------------------------------

pub const MonoResult = struct {
    valid: bool = false,
    /// Smoothed first-partial frequency (Hz).
    f1: f64 = 0,
    /// Hysteresis-locked nearest note and the smoothed offset from it.
    midi: i32 = 0,
    cents: f64 = 0,
    /// NSDF peak value (0..1): periodicity confidence of the current frame.
    clarity: f64 = 0,
    /// Window RMS in dBFS of the (band-limited) input.
    level_db: f64 = -140,
};

pub const StringResult = struct {
    active: bool = false,
    /// Offset from the string's equal-tempered target (cents).
    cents: f64 = 0,
};

pub const PolyResult = struct {
    strings: [6]StringResult = @splat(.{}),
    active_count: u32 = 0,
};

pub const Analyzer = struct {
    const fir_len = 63;
    pub const mono_window = 2048; // decimated samples for NSDF
    pub const poly_window = 4096; // decimated samples for spectral passes
    const hist_cap = 8192; // power of two
    const max_partials = 6;
    /// NSDF peak must reach this for a frame to count as pitched.
    const clarity_min = 0.80;
    /// McLeod key-maxima threshold relative to the highest peak.
    const mpm_k = 0.90;
    const level_gate_db = -70.0;
    /// Partials below max_partial_mag/this are too weak to trust.
    const partial_rel_gate = 32.0;
    const median_len = 5;

    // Configuration (set by setRate/init).
    fs: f64 = 48000,
    decim: u32 = 4,
    fd: f64 = 12000,
    a4: f64 = a4_default,
    fir: [fir_len]f32 = @splat(0),
    /// Upper edge of the trustworthy band after decimation.
    band_hi: f64 = 3600,

    // Streaming state.
    dc_x1: f64 = 0,
    dc_y1: f64 = 0,
    dc_r: f64 = 0.999,
    delay: [fir_len]f32 = @splat(0),
    delay_pos: usize = 0,
    phase: u32 = 0,
    hist: [hist_cap]f32 = undefined,
    hist_total: usize = 0,

    // Display-stability state.
    locked_midi: i32 = -1,
    cand_midi: i32 = -1,
    cand_count: u32 = 0,
    recent_f1: [median_len]f64 = @splat(0),
    recent_len: usize = 0,
    recent_pos: usize = 0,

    // Scratch (analysis-time only).
    abuf: [poly_window]f32 = undefined,
    wx: [poly_window]f64 = undefined,
    hann_m: [mono_window]f32 = undefined,
    hann_p: [poly_window]f32 = undefined,
    nsdf: [mono_window / 2 + 1]f32 = undefined,
    sq_prefix: [mono_window + 1]f64 = undefined,

    pub fn init(fs: u32, a4: f64) Analyzer {
        var self = Analyzer{ .a4 = a4 };
        fillHann(&self.hann_m);
        fillHann(&self.hann_p);
        self.setRate(fs);
        return self;
    }

    /// Reconfigure for a new device rate; clears all streaming state.
    pub fn setRate(self: *Analyzer, fs: u32) void {
        const fs_f: f64 = @floatFromInt(@max(fs, 8000));
        self.fs = fs_f;
        self.decim = @max(1, @as(u32, @intFromFloat(@round(fs_f / 12000.0))));
        self.fd = fs_f / @as(f64, @floatFromInt(self.decim));
        const fc = @min(0.30 * self.fd, 3600.0);
        self.band_hi = fc;
        designLowpass(&self.fir, fc / fs_f);
        self.dc_r = 1.0 - 30.0 / fs_f; // ~5 Hz corner
        self.reset();
    }

    /// Drop all signal history (enable toggles, device switches).
    pub fn reset(self: *Analyzer) void {
        self.dc_x1 = 0;
        self.dc_y1 = 0;
        self.delay = @splat(0);
        self.delay_pos = 0;
        self.phase = 0;
        self.hist_total = 0;
        self.locked_midi = -1;
        self.cand_midi = -1;
        self.cand_count = 0;
        self.recent_len = 0;
        self.recent_pos = 0;
    }

    /// Stream raw device-rate samples through DC block + decimator into the
    /// decimated history ring.
    pub fn feed(self: *Analyzer, samples: []const f32) void {
        for (samples) |s| {
            const x: f64 = s;
            const y = x - self.dc_x1 + self.dc_r * self.dc_y1;
            self.dc_x1 = x;
            self.dc_y1 = y;
            self.delay[self.delay_pos] = @floatCast(y);
            self.delay_pos = if (self.delay_pos + 1 == fir_len) 0 else self.delay_pos + 1;
            self.phase += 1;
            if (self.phase == self.decim) {
                self.phase = 0;
                var acc: f64 = 0;
                var j: usize = 0;
                var p = self.delay_pos; // oldest sample slot
                while (j < fir_len) : (j += 1) {
                    // fir is symmetric, so ordering doesn't matter.
                    acc += @as(f64, self.fir[j]) * self.delay[p];
                    p = if (p + 1 == fir_len) 0 else p + 1;
                }
                self.hist[self.hist_total & (hist_cap - 1)] = @floatCast(acc);
                self.hist_total += 1;
            }
        }
    }

    /// Decimated samples accumulated so far (monotone; for cadence decisions).
    pub fn decimatedTotal(self: *const Analyzer) usize {
        return self.hist_total;
    }

    fn copyLast(self: *const Analyzer, n: usize, dst: []f32) void {
        std.debug.assert(self.hist_total >= n and n <= hist_cap);
        var start = (self.hist_total - n) & (hist_cap - 1);
        for (dst[0..n]) |*d| {
            d.* = self.hist[start];
            start = (start + 1) & (hist_cap - 1);
        }
    }

    pub fn analyzeMono(self: *Analyzer) MonoResult {
        if (self.hist_total < mono_window) return .{};
        self.copyLast(mono_window, self.abuf[0..mono_window]);
        const win = self.abuf[0..mono_window];

        var sum_sq: f64 = 0;
        for (win) |v| sum_sq += @as(f64, v) * v;
        const rms = @sqrt(sum_sq / mono_window);
        const level_db = if (rms > 1e-9) 20.0 * std.math.log10(rms) else -140.0;
        if (level_db < level_gate_db) {
            self.noteUnlock();
            return .{ .level_db = level_db };
        }

        const tau_min: usize = @max(4, @as(usize, @intFromFloat(self.fd / max_f0)));
        const tau_max: usize = @min(mono_window / 2 - 1, @as(usize, @intFromFloat(self.fd / min_f0)) + 1);
        const peak = self.mpm(win, tau_min, tau_max) orelse {
            self.noteUnlock();
            return .{ .level_db = level_db };
        };
        if (peak.clarity < clarity_min) {
            self.noteUnlock();
            return .{ .clarity = peak.clarity, .level_db = level_db };
        }
        const f0c = self.fd / peak.tau;

        // Precision stage: longer window for low notes (partials 27-60 Hz
        // apart need the narrower lobe), otherwise the mono window.
        const n: usize = if (f0c < 60.0 and self.hist_total >= poly_window) poly_window else mono_window;
        self.copyLast(n, self.abuf[0..n]);
        const hann: []const f32 = if (n == poly_window) &self.hann_p else &self.hann_m;
        for (self.wx[0..n], self.abuf[0..n], hann[0..n]) |*d, x, w| d.* = @as(f64, x) * w;
        const f1_raw = self.refineF1(self.wx[0..n], f0c);

        const smoothed = self.noteTrack(f1_raw);
        return .{
            .valid = true,
            .f1 = smoothed.f1,
            .midi = smoothed.midi,
            .cents = smoothed.cents,
            .clarity = peak.clarity,
            .level_db = level_db,
        };
    }

    const MpmPeak = struct { tau: f64, clarity: f64 };

    /// NSDF + McLeod key-maxima picking. Returns the chosen peak's
    /// parabolically refined lag and its value (clarity).
    fn mpm(self: *Analyzer, x: []const f32, tau_min: usize, tau_max: usize) ?MpmPeak {
        const w = x.len;
        self.sq_prefix[0] = 0;
        for (x, 0..) |v, i| self.sq_prefix[i + 1] = self.sq_prefix[i] + @as(f64, v) * v;

        // NSDF from tau=1: the zero-crossing walk below must see the whole
        // tau→0 lobe even when tau_min lands past its first crossing (very
        // high notes), or the true peak's region would be skipped and the
        // 2·tau peak picked instead (an octave-down error).
        var tau: usize = 1;
        while (tau <= tau_max) : (tau += 1) {
            var acf: f64 = 0;
            for (0..w - tau) |i| acf += @as(f64, x[i]) * x[i + tau];
            const m = (self.sq_prefix[w - tau] - self.sq_prefix[0]) + (self.sq_prefix[w] - self.sq_prefix[tau]);
            self.nsdf[tau] = if (m > 1e-12) @floatCast(2.0 * acf / m) else 0;
        }

        // Key maxima: one candidate per positive NSDF region, starting only
        // after the curve first dips below zero (skips the tau→0 lobe);
        // region maxima count only inside [tau_min, tau_max].
        const Cand = struct { tau: usize, val: f32 };
        var cands: [64]Cand = undefined;
        var cand_n: usize = 0;
        var seen_negative = false;
        var in_region = false;
        var best_tau: usize = 0;
        var best_val: f32 = -1;
        tau = 1;
        while (tau <= tau_max) : (tau += 1) {
            const v = self.nsdf[tau];
            if (v <= 0) {
                seen_negative = true;
                if (in_region and best_tau > 0 and cand_n < cands.len) {
                    cands[cand_n] = .{ .tau = best_tau, .val = best_val };
                    cand_n += 1;
                }
                in_region = false;
                continue;
            }
            if (!seen_negative) continue;
            if (!in_region) {
                in_region = true;
                best_tau = 0;
                best_val = -1;
            }
            if (tau >= tau_min and v > best_val) {
                best_tau = tau;
                best_val = v;
            }
        }
        if (in_region and best_tau > 0 and cand_n < cands.len) {
            cands[cand_n] = .{ .tau = best_tau, .val = best_val };
            cand_n += 1;
        }
        if (cand_n == 0) return null;

        var nmax: f32 = 0;
        for (cands[0..cand_n]) |c| nmax = @max(nmax, c.val);
        const threshold = mpm_k * nmax;
        var chosen: ?Cand = null;
        for (cands[0..cand_n]) |c| {
            if (c.val >= threshold) {
                chosen = c;
                break;
            }
        }
        const c = chosen orelse return null;

        // Parabolic refinement of the lag on the NSDF samples.
        var t_ref: f64 = @floatFromInt(c.tau);
        var v_ref: f64 = c.val;
        if (c.tau > 1 and c.tau < tau_max) {
            const ym: f64 = self.nsdf[c.tau - 1];
            const y0: f64 = self.nsdf[c.tau];
            const yp: f64 = self.nsdf[c.tau + 1];
            const den = ym - 2.0 * y0 + yp;
            if (@abs(den) > 1e-12) {
                const d = 0.5 * (ym - yp) / den;
                if (@abs(d) < 1.0) {
                    t_ref += d;
                    v_ref = y0 - 0.25 * (ym - yp) * d;
                }
            }
        }
        return .{ .tau = t_ref, .clarity = @min(v_ref, 1.0) };
    }

    /// |DFT|² of the pre-windowed buffer at an exact frequency (Hz), via a
    /// complex oscillator recurrence (no FFT grid).
    fn spectralPower(self: *const Analyzer, wx: []const f64, f: f64) f64 {
        const th = -2.0 * std.math.pi * (f / self.fd);
        const cr = @cos(th);
        const ci = @sin(th);
        var osc_r: f64 = 1;
        var osc_i: f64 = 0;
        var ar: f64 = 0;
        var ai: f64 = 0;
        for (wx) |v| {
            ar += v * osc_r;
            ai += v * osc_i;
            const nr = osc_r * cr - osc_i * ci;
            osc_i = osc_r * ci + osc_i * cr;
            osc_r = nr;
        }
        return ar * ar + ai * ai;
    }

    /// Peak amplitude (linear, ~sinusoid peak) implied by a spectral power.
    fn powerToAmp(wx_len: usize, hann_sum: f64, p: f64) f64 {
        _ = wx_len;
        return 2.0 * @sqrt(p) / hann_sum;
    }

    const Refined = struct { f: f64, power: f64 };

    /// Iterated 3-point parabolic maximization of |X(f)|² over
    /// [f0·2^(-band/1200), f0·2^(band/1200)]: a coarse 9-point scan, then 4
    /// shrink-and-refit rounds. Final granularity ≈ band/2048 cents.
    fn refinePeak(self: *const Analyzer, wx: []const f64, center: f64, band_cents: f64) Refined {
        const lo = center * centsFactor(-band_cents);
        const hi = center * centsFactor(band_cents);
        var best_f = center;
        var best_p: f64 = -1;
        for (0..9) |i| {
            const f = lo + (hi - lo) * @as(f64, @floatFromInt(i)) / 8.0;
            const p = self.spectralPower(wx, f);
            if (p > best_p) {
                best_p = p;
                best_f = f;
            }
        }
        var step = (hi - lo) / 8.0;
        for (0..4) |_| {
            step *= 0.25;
            const pm = self.spectralPower(wx, best_f - step);
            const pp = self.spectralPower(wx, best_f + step);
            const den = pm - 2.0 * best_p + pp;
            if (pm > best_p or pp > best_p) {
                if (pp > pm) {
                    best_f += step;
                    best_p = pp;
                } else {
                    best_f -= step;
                    best_p = pm;
                }
            } else if (@abs(den) > 0) {
                const d = 0.5 * (pm - pp) / den * step;
                const f = best_f + std.math.clamp(d, -step, step);
                const p = self.spectralPower(wx, f);
                if (p >= best_p) {
                    best_f = f;
                    best_p = p;
                }
            }
        }
        return .{ .f = best_f, .power = best_p };
    }

    /// Measure partials k·f0c, fit f_k² = f0²(1+B·k²) weighted by partial
    /// amplitude, and return the first-partial frequency f0·sqrt(1+B).
    fn refineF1(self: *const Analyzer, wx: []const f64, f0c: f64) f64 {
        var hann_sum: f64 = 0;
        const hann: []const f32 = if (wx.len == poly_window) &self.hann_p else &self.hann_m;
        for (hann[0..wx.len]) |w| hann_sum += w;

        var ks: [max_partials]f64 = undefined;
        var fs_meas: [max_partials]f64 = undefined;
        var amps: [max_partials]f64 = undefined;
        var count: usize = 0;
        var amp_max: f64 = 0;
        var k: usize = 1;
        while (k <= max_partials) : (k += 1) {
            const target = f0c * @as(f64, @floatFromInt(k));
            if (target > self.band_hi or target > 0.45 * self.fd) break;
            const r = self.refinePeak(wx, target, 40.0);
            const amp = powerToAmp(wx.len, hann_sum, r.power);
            // Reject a refinement that wandered off (>35 cents means the
            // parabola latched onto a neighbor/noise, not this partial).
            if (@abs(centsBetween(r.f, target)) > 35.0) continue;
            ks[count] = @floatFromInt(k);
            fs_meas[count] = r.f;
            amps[count] = amp;
            count += 1;
            amp_max = @max(amp_max, amp);
        }
        if (count == 0) return f0c;

        // Drop partials that are too weak to carry information.
        var kept: usize = 0;
        for (0..count) |i| {
            if (amps[i] >= amp_max / partial_rel_gate) {
                ks[kept] = ks[i];
                fs_meas[kept] = fs_meas[i];
                amps[kept] = amps[i];
                kept += 1;
            }
        }
        if (kept == 1) return fs_meas[0] / ks[0];

        // Weighted LS of y = a + c·u with y = (f/k)², u = k².
        var sw: f64 = 0;
        var su: f64 = 0;
        var sy: f64 = 0;
        var suu: f64 = 0;
        var suy: f64 = 0;
        for (0..kept) |i| {
            const u = ks[i] * ks[i];
            const y = (fs_meas[i] / ks[i]) * (fs_meas[i] / ks[i]);
            const wgt = amps[i];
            sw += wgt;
            su += wgt * u;
            sy += wgt * y;
            suu += wgt * u * u;
            suy += wgt * u * y;
        }
        const den = sw * suu - su * su;
        var a = sy / sw;
        var b_inh: f64 = 0;
        if (@abs(den) > 1e-9) {
            const c = (sw * suy - su * sy) / den;
            const a_fit = (sy - c * su) / sw;
            if (a_fit > 0) {
                const b_fit = c / a_fit;
                if (b_fit > 0 and b_fit < 3e-3) {
                    a = a_fit;
                    b_inh = b_fit;
                }
            }
        }
        if (a <= 0) return fs_meas[0] / ks[0];
        return @sqrt(a * (1.0 + b_inh));
    }

    const Tracked = struct { f1: f64, midi: i32, cents: f64 };

    /// Note-name hysteresis (2 consecutive frames to switch) + median of the
    /// recent estimates for the displayed frequency.
    fn noteTrack(self: *Analyzer, f1: f64) Tracked {
        const nn = nearestNote(f1, self.a4);
        if (self.locked_midi < 0) {
            self.lockNote(nn.midi);
        } else if (nn.midi != self.locked_midi) {
            if (self.cand_midi == nn.midi) {
                self.cand_count += 1;
                if (self.cand_count >= 2) self.lockNote(nn.midi);
            } else {
                self.cand_midi = nn.midi;
                self.cand_count = 1;
            }
        } else {
            self.cand_midi = -1;
            self.cand_count = 0;
        }

        self.recent_f1[self.recent_pos] = f1;
        self.recent_pos = (self.recent_pos + 1) % median_len;
        if (self.recent_len < median_len) self.recent_len += 1;

        var tmp: [median_len]f64 = undefined;
        @memcpy(tmp[0..self.recent_len], self.recent_f1[0..self.recent_len]);
        std.mem.sort(f64, tmp[0..self.recent_len], {}, std.sort.asc(f64));
        const med = tmp[self.recent_len / 2];
        return .{
            .f1 = med,
            .midi = self.locked_midi,
            .cents = centsBetween(med, midiFreq(self.locked_midi, self.a4)),
        };
    }

    fn lockNote(self: *Analyzer, midi: i32) void {
        self.locked_midi = midi;
        self.cand_midi = -1;
        self.cand_count = 0;
        self.recent_len = 0;
        self.recent_pos = 0;
    }

    fn noteUnlock(self: *Analyzer) void {
        self.locked_midi = -1;
        self.cand_midi = -1;
        self.cand_count = 0;
        self.recent_len = 0;
        self.recent_pos = 0;
    }

    /// Standard-tuning open-string midi notes: E2 A2 D3 G3 B3 E4.
    pub const string_midis = [6]i32{ 40, 45, 50, 55, 59, 64 };

    pub fn analyzePoly(self: *Analyzer) PolyResult {
        if (self.hist_total < poly_window) return .{};
        self.copyLast(poly_window, self.abuf[0..poly_window]);
        var sum_sq: f64 = 0;
        for (self.abuf[0..poly_window]) |v| sum_sq += @as(f64, v) * v;
        const rms = @sqrt(sum_sq / poly_window);
        if (rms < 3.2e-4) return .{}; // ~ -70 dBFS

        for (self.wx[0..poly_window], self.abuf[0..poly_window], self.hann_p) |*d, x, w| d.* = @as(f64, x) * w;
        const wx = self.wx[0..poly_window];
        var hann_sum: f64 = 0;
        for (self.hann_p) |w| hann_sum += w;

        const Detected = struct { f: f64, amp: f64, cents: f64, edge: bool };
        var det: [6]Detected = undefined;
        for (0..6) |s| {
            const target = midiFreq(string_midis[s], self.a4);
            // Fundamental-only scan: ±120 cents in 6-cent steps. Harmonic
            // salience is deliberately NOT used to locate the peak — at the
            // low strings ±120 cents is only a few Hz wide, so a colliding
            // partial of another string (E2·3 ≈ B3) at 3f would drag the
            // scan off the (nearly flat) fundamental lobe. Ghost activity is
            // handled by the amplitude gates + masking pass below instead.
            var best_c: f64 = 0;
            var best_p: f64 = -1;
            var best_idx: usize = 0;
            for (0..41) |i| {
                const cents = -120.0 + 6.0 * @as(f64, @floatFromInt(i));
                const p = self.spectralPower(wx, target * centsFactor(cents));
                if (p > best_p) {
                    best_p = p;
                    best_c = cents;
                    best_idx = i;
                }
            }
            const r = self.refinePeak(wx, target * centsFactor(best_c), 8.0);
            const amp = powerToAmp(poly_window, hann_sum, r.power);
            det[s] = .{
                .f = r.f,
                .amp = amp,
                .cents = centsBetween(r.f, target),
                .edge = best_idx == 0 or best_idx == 40,
            };
        }

        var amp_all_max: f64 = 0;
        for (det) |d| amp_all_max = @max(amp_all_max, d.amp);

        var out = PolyResult{};
        for (0..6) |s| {
            const d = det[s];
            if (d.edge) continue;
            if (d.amp < 3.2e-4 or d.amp < amp_all_max / 12.0) continue;
            // Masking: a lower ACTIVE string whose integer partial lands on
            // this candidate (within 30 cents) explains it unless the
            // candidate is clearly louder than that partial would be
            // (expected rolloff ~ amp_lo/m).
            var masked = false;
            for (0..s) |lo| {
                if (!out.strings[lo].active) continue;
                const m = @round(d.f / det[lo].f);
                if (m < 2 or m > 8) continue;
                if (@abs(centsBetween(d.f, det[lo].f * m)) < 30.0 and d.amp < 0.6 * det[lo].amp / m) {
                    masked = true;
                    break;
                }
            }
            if (masked) continue;
            out.strings[s] = .{ .active = true, .cents = d.cents };
            out.active_count += 1;
        }
        return out;
    }
};

fn centsFactor(cents: f64) f64 {
    return std.math.pow(f64, 2.0, cents / 1200.0);
}

fn fillHann(buf: []f32) void {
    const n = buf.len;
    for (buf, 0..) |*w, i| {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n - 1));
        w.* = @floatCast(0.5 - 0.5 * @cos(2.0 * std.math.pi * t));
    }
}

/// Windowed-sinc (Blackman) lowpass, unity DC gain. `fc_norm` = fc/fs.
fn designLowpass(taps: []f32, fc_norm: f64) void {
    const n = taps.len;
    const mid = @as(f64, @floatFromInt(n - 1)) / 2.0;
    var sum: f64 = 0;
    for (taps, 0..) |*t, i| {
        const m = @as(f64, @floatFromInt(i)) - mid;
        const x = 2.0 * fc_norm * m;
        const sinc = if (@abs(x) < 1e-12) 1.0 else @sin(std.math.pi * x) / (std.math.pi * x);
        const u = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n - 1));
        const w = 0.42 - 0.5 * @cos(2.0 * std.math.pi * u) + 0.08 * @cos(4.0 * std.math.pi * u);
        const v = 2.0 * fc_norm * sinc * w;
        t.* = @floatCast(v);
        sum += v;
    }
    for (taps) |*t| t.* = @floatCast(@as(f64, t.*) / sum);
}

// ---------------------------------------------------------------------------
// Tuner: the analysis thread + lock-free published snapshot. All display
// fields are independent atomics (tearing across fields only ever mis-renders
// one 33 ms UI frame).
// ---------------------------------------------------------------------------

pub const Snapshot = struct {
    valid: bool = false,
    midi: i32 = 0,
    cents: f32 = 0,
    hz: f32 = 0,
    clarity: f32 = 0,
    poly_count: u32 = 0,
    strings: [6]StringResult = @splat(.{}),
};

pub const Tuner = struct {
    tap: Tap = .{},
    analyzer: Analyzer,
    io: std.Io,
    thread: std.Thread = undefined,
    running: std.atomic.Value(bool) = .init(true),
    wake_word: std.atomic.Value(u32) = .init(0),
    rate_req: std.atomic.Value(u32),
    /// Bumped on enable so the thread drops stale history before analyzing.
    epoch: std.atomic.Value(u32) = .init(0),

    // Published mono result.
    out_valid: std.atomic.Value(bool) = .init(false),
    out_midi: std.atomic.Value(i32) = .init(0),
    out_cents_x100: std.atomic.Value(i32) = .init(0),
    out_hz_bits: std.atomic.Value(u32) = .init(0),
    out_clarity_x1000: std.atomic.Value(u32) = .init(0),
    // Published poly result: bit16 = active, low 16 bits = cents*10 as i16.
    out_strings: [6]std.atomic.Value(u32) = @splat(.init(0)),
    out_poly_count: std.atomic.Value(u32) = .init(0),

    read_pos: usize = 0,
    last_mono_total: usize = 0,
    last_poly_total: usize = 0,
    seen_epoch: u32 = 0,

    pub fn create(allocator: std.mem.Allocator, io: std.Io, fs: u32, a4: f64, start_enabled: bool) !*Tuner {
        const self = try allocator.create(Tuner);
        errdefer allocator.destroy(self);
        self.* = .{
            .analyzer = Analyzer.init(fs, a4),
            .io = io,
            .rate_req = .init(fs),
        };
        if (start_enabled) self.setEnabled(true);
        self.thread = try std.Thread.spawn(.{}, threadMain, .{self});
        return self;
    }

    pub fn destroy(self: *Tuner, allocator: std.mem.Allocator) void {
        self.running.store(false, .release);
        self.wakeThread();
        self.thread.join();
        allocator.destroy(self);
    }

    pub fn enabled(self: *const Tuner) bool {
        return self.tap.enabled.load(.monotonic);
    }

    pub fn setEnabled(self: *Tuner, on: bool) void {
        if (on and !self.enabled()) {
            _ = self.epoch.fetchAdd(1, .release);
            self.out_valid.store(false, .monotonic);
            self.out_poly_count.store(0, .monotonic);
        }
        self.tap.enabled.store(on, .monotonic);
        self.wakeThread();
    }

    /// Follow the device's actual rate (cheap; call every UI tick).
    pub fn setRate(self: *Tuner, fs: u32) void {
        self.rate_req.store(fs, .monotonic);
    }

    fn wakeThread(self: *Tuner) void {
        _ = self.wake_word.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.wake_word.raw, 1);
    }

    pub fn snapshot(self: *const Tuner) Snapshot {
        var s = Snapshot{
            .valid = self.out_valid.load(.monotonic),
            .midi = self.out_midi.load(.monotonic),
            .cents = @as(f32, @floatFromInt(self.out_cents_x100.load(.monotonic))) / 100.0,
            .hz = @bitCast(self.out_hz_bits.load(.monotonic)),
            .clarity = @as(f32, @floatFromInt(self.out_clarity_x1000.load(.monotonic))) / 1000.0,
            .poly_count = self.out_poly_count.load(.monotonic),
        };
        for (&s.strings, &self.out_strings) |*dst, *src| {
            const raw = src.load(.monotonic);
            dst.* = .{
                .active = raw & (1 << 16) != 0,
                .cents = @as(f32, @floatFromInt(@as(i16, @bitCast(@as(u16, @truncate(raw)))))) / 10.0,
            };
        }
        return s;
    }

    fn publishMono(self: *Tuner, r: MonoResult) void {
        self.out_midi.store(r.midi, .monotonic);
        self.out_cents_x100.store(@intFromFloat(@round(std.math.clamp(r.cents, -99.0, 99.0) * 100.0)), .monotonic);
        self.out_hz_bits.store(@bitCast(@as(f32, @floatCast(r.f1))), .monotonic);
        self.out_clarity_x1000.store(@intFromFloat(@round(std.math.clamp(r.clarity, 0, 1) * 1000.0)), .monotonic);
        self.out_valid.store(r.valid, .monotonic);
    }

    fn publishPoly(self: *Tuner, r: PolyResult) void {
        for (&self.out_strings, r.strings) |*dst, s| {
            const cents_x10: i16 = @intFromFloat(@round(std.math.clamp(s.cents, -99.0, 99.0) * 10.0));
            const active_bit: u32 = if (s.active) 1 << 16 else 0;
            dst.store(active_bit | @as(u16, @bitCast(cents_x10)), .monotonic);
        }
        self.out_poly_count.store(r.active_count, .monotonic);
    }

    fn threadMain(self: *Tuner) void {
        while (self.running.load(.acquire)) {
            // Disabled => park with NO timeout: the tuner costs zero CPU
            // until setEnabled/destroy bumps wake_word and wakes the futex
            // (the snapshot-then-wait pattern makes the wake un-losable).
            const seen = self.wake_word.load(.monotonic);
            const timeout: std.Io.Timeout = if (self.tap.enabled.load(.monotonic))
                .{ .duration = .{ .raw = .{ .nanoseconds = 30 * std.time.ns_per_ms }, .clock = .awake } }
            else
                .none;
            self.io.futexWaitTimeout(u32, &self.wake_word.raw, seen, timeout) catch {};
            if (!self.running.load(.acquire)) break;

            const rate = self.rate_req.load(.monotonic);
            if (@as(f64, @floatFromInt(rate)) != self.analyzer.fs) {
                self.analyzer.setRate(rate);
                self.read_pos = self.tap.write_pos.load(.acquire);
                self.last_mono_total = 0;
                self.last_poly_total = 0;
            }

            if (!self.tap.enabled.load(.monotonic)) {
                // Drop backlog so re-enable starts from fresh audio.
                self.read_pos = self.tap.write_pos.load(.acquire);
                continue;
            }
            const e = self.epoch.load(.acquire);
            if (e != self.seen_epoch) {
                self.seen_epoch = e;
                self.analyzer.reset();
                self.read_pos = self.tap.write_pos.load(.acquire);
                self.last_mono_total = 0;
                self.last_poly_total = 0;
            }

            self.drainTap();

            const fd = self.analyzer.fd;
            const mono_hop: usize = @intFromFloat(fd / 15.0);
            const poly_hop: usize = @intFromFloat(fd / 4.0);
            const total = self.analyzer.decimatedTotal();
            if (total >= self.last_mono_total + mono_hop) {
                self.last_mono_total = total;
                self.publishMono(self.analyzer.analyzeMono());
            }
            if (total >= self.last_poly_total + poly_hop) {
                self.last_poly_total = total;
                self.publishPoly(self.analyzer.analyzePoly());
            }
        }
    }

    fn drainTap(self: *Tuner) void {
        const cap = Tap.capacity;
        var write = self.tap.write_pos.load(.acquire);
        if (write - self.read_pos > cap / 2) self.read_pos = write - cap / 2;
        while (self.read_pos < write) {
            const i = self.read_pos & (cap - 1);
            const n = @min(write - self.read_pos, cap - i);
            self.analyzer.feed(self.tap.ring[i..][0..n]);
            self.read_pos += n;
            write = self.tap.write_pos.load(.acquire);
            if (write - self.read_pos > cap / 2) self.read_pos = write - cap / 2;
        }
    }
};

test {
    _ = @import("tuner_tests.zig");
}
