//! Behavioral tests for the live tuner (tuner.zig): note naming, SPSC tap
//! semantics, mono accuracy on pure and inharmonic-plucked tones (including
//! octave robustness with a weak fundamental), silence/noise rejection, and
//! the polyphonic strum check with harmonic-ghost masking. All signals are
//! synthesized deterministically; accuracy assertions are in cents.

const std = @import("std");
const tuner = @import("tuner.zig");

const Analyzer = tuner.Analyzer;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const Partial = struct { k: f64, amp: f64 };

/// Additive stiff-string synthesis: partial k at f0·k·sqrt(1 + B·k²), with a
/// global exponential decay. Adds into `buf`.
fn synthInto(buf: []f32, fs: f64, f0: f64, partials: []const Partial, b_inh: f64, decay: f64, gain: f64) void {
    for (buf, 0..) |*s, i| {
        const t = @as(f64, @floatFromInt(i)) / fs;
        var acc: f64 = 0;
        for (partials) |p| {
            const fk = f0 * p.k * @sqrt(1.0 + b_inh * p.k * p.k);
            acc += p.amp * @sin(2.0 * std.math.pi * fk * t);
        }
        s.* += @floatCast(gain * acc * @exp(-decay * t));
    }
}

const pure = [_]Partial{.{ .k = 1, .amp = 1 }};

/// Feed `samples` in callback-sized chunks, running a mono analysis at the
/// live thread's cadence; returns the last result.
fn runMono(an: *Analyzer, samples: []const f32, fs: usize) tuner.MonoResult {
    var last = tuner.MonoResult{};
    const hop = fs / 15;
    var next = hop;
    var fed: usize = 0;
    while (fed < samples.len) {
        const n = @min(@as(usize, 480), samples.len - fed);
        an.feed(samples[fed..][0..n]);
        fed += n;
        if (fed >= next) {
            next += hop;
            last = an.analyzeMono();
        }
    }
    return last;
}

fn newAnalyzer(fs: u32, a4: f64) !*Analyzer {
    const an = try std.testing.allocator.create(Analyzer);
    an.* = Analyzer.init(fs, a4);
    return an;
}

fn expectCentsWithin(f_measured: f64, f_true: f64, tol: f64) !void {
    const err = tuner.centsBetween(f_measured, f_true);
    if (@abs(err) > tol) {
        std.debug.print("cents error {d:.3} exceeds {d:.3} (measured {d:.4} Hz vs {d:.4} Hz)\n", .{ err, tol, f_measured, f_true });
        return error.TestExpectedApproxEq;
    }
}

// ---------------------------------------------------------------------------
// Note naming
// ---------------------------------------------------------------------------

test "note naming: nearest note, labels, a4 override" {
    const a440 = tuner.nearestNote(440.0, 440.0);
    try std.testing.expectEqual(@as(i32, 69), a440.midi);
    try std.testing.expectApproxEqAbs(@as(f64, 0), a440.cents, 1e-9);

    const sharp = tuner.nearestNote(445.0, 440.0);
    try std.testing.expectEqual(@as(i32, 69), sharp.midi);
    try std.testing.expectApproxEqAbs(@as(f64, 19.56), sharp.cents, 0.01);

    // A4 reference shifts the grid: 440 Hz against a4=432 reads sharp.
    const vs432 = tuner.nearestNote(440.0, 432.0);
    try std.testing.expectEqual(@as(i32, 69), vs432.midi);
    try std.testing.expectApproxEqAbs(@as(f64, 31.77), vs432.cents, 0.01);

    var buf: [8]u8 = undefined;
    try std.testing.expectEqualStrings("E2", tuner.noteLabel(40, &buf));
    try std.testing.expectEqualStrings("F#4", tuner.noteLabel(66, &buf));
    try std.testing.expectEqualStrings("A4", tuner.noteLabel(69, &buf));
    try std.testing.expectApproxEqAbs(@as(f64, 82.4069), tuner.midiFreq(40, 440.0), 1e-3);
}

// ---------------------------------------------------------------------------
// Tap ring
// ---------------------------------------------------------------------------

test "tap: disabled drops, enabled preserves the newest samples across wraps" {
    const tap = try std.testing.allocator.create(tuner.Tap);
    defer std.testing.allocator.destroy(tap);
    tap.* = .{};

    var block: [193]f32 = undefined;
    tap.push(&block); // disabled: no write
    try std.testing.expectEqual(@as(usize, 0), tap.write_pos.load(.monotonic));

    tap.enabled.store(true, .monotonic);
    var value: f32 = 0;
    var total: usize = 0;
    while (total < tuner.Tap.capacity + 5000) {
        for (&block) |*s| {
            s.* = value;
            value += 1;
        }
        tap.push(&block);
        total += block.len;
    }
    try std.testing.expectEqual(total, tap.write_pos.load(.monotonic));
    // The last `capacity` samples must be intact: ring[(total-1-k) & mask]
    // holds the ramp value total-1-k.
    var k: usize = 0;
    while (k < tuner.Tap.capacity) : (k += 997) {
        const pos = total - 1 - k;
        const expect: f32 = @floatFromInt(pos);
        try std.testing.expectEqual(expect, tap.ring[pos & (tuner.Tap.capacity - 1)]);
    }
}

// ---------------------------------------------------------------------------
// Mono accuracy
// ---------------------------------------------------------------------------

test "mono: pure sines across the range detect within 0.1 cent (48 kHz)" {
    const an = try newAnalyzer(48000, 440.0);
    defer std.testing.allocator.destroy(an);
    const allocator = std.testing.allocator;
    const samples = try allocator.alloc(f32, 24000); // 0.5 s
    defer allocator.free(samples);

    const cases = [_]struct { f: f64, midi: i32 }{
        .{ .f = 82.4069, .midi = 40 }, // E2
        .{ .f = 110.0, .midi = 45 }, // A2
        .{ .f = 146.8324, .midi = 50 }, // D3
        .{ .f = 195.9977, .midi = 55 }, // G3
        .{ .f = 246.9417, .midi = 59 }, // B3
        .{ .f = 329.6276, .midi = 64 }, // E4
        .{ .f = 440.0, .midi = 69 }, // A4
        .{ .f = 659.2551, .midi = 76 }, // E5
        .{ .f = 987.7666, .midi = 83 }, // B5
        .{ .f = 1318.5102, .midi = 88 }, // E6 (24th fret high e; octave-pick guard)
        // Detuned strings (the actual tuning use case).
        .{ .f = 82.4069 * 0.98266, .midi = 40 }, // E2 - 30.3 cents
        .{ .f = 110.0 * 1.01162, .midi = 45 }, // A2 + 20 cents
    };
    for (cases) |case| {
        an.reset();
        @memset(samples, 0);
        synthInto(samples, 48000, case.f, &pure, 0, 0, 0.25);
        const r = runMono(an, samples, 48000);
        try std.testing.expect(r.valid);
        try std.testing.expectEqual(case.midi, r.midi);
        try expectCentsWithin(r.f1, case.f, 0.1);
    }
}

test "mono: pure sines at 44.1 kHz detect within 0.1 cent" {
    const an = try newAnalyzer(44100, 440.0);
    defer std.testing.allocator.destroy(an);
    const allocator = std.testing.allocator;
    const samples = try allocator.alloc(f32, 22050);
    defer allocator.free(samples);

    for ([_]f64{ 82.4069, 196.5, 440.0 }) |f| {
        an.reset();
        @memset(samples, 0);
        synthInto(samples, 44100, f, &pure, 0, 0, 0.25);
        const r = runMono(an, samples, 44100);
        try std.testing.expect(r.valid);
        try expectCentsWithin(r.f1, f, 0.1);
    }
}

test "mono: inharmonic pluck reports the true first partial within 0.5 cent" {
    const an = try newAnalyzer(48000, 440.0);
    defer std.testing.allocator.destroy(an);
    const allocator = std.testing.allocator;
    const samples = try allocator.alloc(f32, 36000); // 0.75 s
    defer allocator.free(samples);
    @memset(samples, 0);

    const f0 = 110.0;
    const b_inh = 3e-4;
    const partials = [_]Partial{
        .{ .k = 1, .amp = 1.0 },  .{ .k = 2, .amp = 0.55 },
        .{ .k = 3, .amp = 0.32 }, .{ .k = 4, .amp = 0.22 },
        .{ .k = 5, .amp = 0.14 }, .{ .k = 6, .amp = 0.09 },
        .{ .k = 7, .amp = 0.06 }, .{ .k = 8, .amp = 0.04 },
    };
    synthInto(samples, 48000, f0, &partials, b_inh, 1.5, 0.2);
    const f1_true = f0 * @sqrt(1.0 + b_inh);

    const r = runMono(an, samples, 48000);
    try std.testing.expect(r.valid);
    try std.testing.expectEqual(@as(i32, 45), r.midi);
    try expectCentsWithin(r.f1, f1_true, 0.5);
}

test "mono: weak fundamental with dominant harmonics keeps the right octave" {
    const an = try newAnalyzer(48000, 440.0);
    defer std.testing.allocator.destroy(an);
    const allocator = std.testing.allocator;
    const samples = try allocator.alloc(f32, 30000);
    defer allocator.free(samples);
    @memset(samples, 0);

    const f0 = 82.4069;
    const b_inh = 2e-4;
    const partials = [_]Partial{
        .{ .k = 1, .amp = 0.05 }, .{ .k = 2, .amp = 1.0 },
        .{ .k = 3, .amp = 0.45 }, .{ .k = 4, .amp = 0.30 },
        .{ .k = 5, .amp = 0.15 },
    };
    synthInto(samples, 48000, f0, &partials, b_inh, 1.0, 0.2);
    const f1_true = f0 * @sqrt(1.0 + b_inh);

    const r = runMono(an, samples, 48000);
    try std.testing.expect(r.valid);
    try std.testing.expectEqual(@as(i32, 40), r.midi); // E2, not E3
    try expectCentsWithin(r.f1, f1_true, 1.0);
}

test "mono: silence and noise are rejected" {
    const an = try newAnalyzer(48000, 440.0);
    defer std.testing.allocator.destroy(an);
    const allocator = std.testing.allocator;
    const samples = try allocator.alloc(f32, 16000);
    defer allocator.free(samples);

    @memset(samples, 0);
    var r = runMono(an, samples, 48000);
    try std.testing.expect(!r.valid);
    try std.testing.expect(r.level_db < -90);

    // Deterministic white-ish noise at a healthy level.
    an.reset();
    var state: u64 = 0x9E3779B97F4A7C15;
    for (samples) |*s| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const u: f64 = @floatFromInt(state >> 11);
        s.* = @floatCast((u / @as(f64, @floatFromInt(@as(u64, 1) << 53)) - 0.5) * 0.2);
    }
    r = runMono(an, samples, 48000);
    try std.testing.expect(!r.valid);
}

test "mono: sustained note change switches the lock" {
    const an = try newAnalyzer(48000, 440.0);
    defer std.testing.allocator.destroy(an);
    const allocator = std.testing.allocator;
    const samples = try allocator.alloc(f32, 24000);
    defer allocator.free(samples);

    @memset(samples, 0);
    synthInto(samples, 48000, 110.0, &pure, 0, 0, 0.25);
    var r = runMono(an, samples, 48000);
    try std.testing.expectEqual(@as(i32, 45), r.midi);

    @memset(samples, 0);
    synthInto(samples, 48000, 123.4708, &pure, 0, 0, 0.25); // B2
    r = runMono(an, samples, 48000);
    try std.testing.expect(r.valid);
    try std.testing.expectEqual(@as(i32, 47), r.midi);
}

// ---------------------------------------------------------------------------
// Poly strum check
// ---------------------------------------------------------------------------

/// Rolloff 1/k^1.7 keeps synthetic partial 3 below the masking gate.
fn pluckPartials(comptime n: usize) [n]Partial {
    var out: [n]Partial = undefined;
    for (&out, 1..) |*p, k| {
        const kf: f64 = @floatFromInt(k);
        p.* = .{ .k = kf, .amp = 1.0 / std.math.pow(f64, kf, 1.7) };
    }
    return out;
}

test "poly: full strum recovers every string's detune within 3.5 cents" {
    const an = try newAnalyzer(48000, 440.0);
    defer std.testing.allocator.destroy(an);
    const allocator = std.testing.allocator;
    const samples = try allocator.alloc(f32, 48000); // 1 s
    defer allocator.free(samples);
    @memset(samples, 0);

    const detunes = [6]f64{ -15, 8, 0, -4, 20, 3 };
    const gains = [6]f64{ 0.11, 0.105, 0.1, 0.095, 0.09, 0.085 };
    const partials = pluckPartials(5);
    for (0..6) |s| {
        const target = tuner.midiFreq(Analyzer.string_midis[s], 440.0);
        const f1 = target * std.math.pow(f64, 2.0, detunes[s] / 1200.0);
        // synthInto's first partial lands at f0·sqrt(1+B); place it on f1.
        const b_inh = 2e-4;
        synthInto(samples, 48000, f1 / @sqrt(1.0 + b_inh), &partials, b_inh, 1.0, gains[s]);
    }

    an.feed(samples);
    const r = an.analyzePoly();
    try std.testing.expectEqual(@as(u32, 6), r.active_count);
    for (0..6) |s| {
        try std.testing.expect(r.strings[s].active);
        try std.testing.expectApproxEqAbs(detunes[s], r.strings[s].cents, 3.5);
    }
}

test "poly: harmonic ghosts of sounding strings stay masked" {
    const an = try newAnalyzer(48000, 440.0);
    defer std.testing.allocator.destroy(an);
    const allocator = std.testing.allocator;
    const samples = try allocator.alloc(f32, 48000);
    defer allocator.free(samples);
    @memset(samples, 0);

    // Only A2 (+5 cents) and G3 (-7 cents) sound. A2's 3rd partial lands in
    // the high-e band (110·3 = 330 = E4) and must not read as a sixth string.
    const partials = pluckPartials(5);
    const b_inh = 2e-4;
    const a2 = tuner.midiFreq(45, 440.0) * std.math.pow(f64, 2.0, 5.0 / 1200.0);
    const g3 = tuner.midiFreq(55, 440.0) * std.math.pow(f64, 2.0, -7.0 / 1200.0);
    synthInto(samples, 48000, a2 / @sqrt(1.0 + b_inh), &partials, b_inh, 1.0, 0.15);
    synthInto(samples, 48000, g3 / @sqrt(1.0 + b_inh), &partials, b_inh, 1.0, 0.13);

    an.feed(samples);
    const r = an.analyzePoly();
    try std.testing.expect(r.strings[1].active); // A
    try std.testing.expect(r.strings[3].active); // G
    try std.testing.expect(!r.strings[0].active); // E2: nothing there
    try std.testing.expect(!r.strings[2].active); // D3: nothing there
    try std.testing.expect(!r.strings[5].active); // e: A2's 3rd partial, masked
    try std.testing.expectApproxEqAbs(@as(f64, 5), r.strings[1].cents, 3.0);
    try std.testing.expectApproxEqAbs(@as(f64, -7), r.strings[3].cents, 3.0);
}

test "poly: silence reports no strings" {
    const an = try newAnalyzer(48000, 440.0);
    defer std.testing.allocator.destroy(an);
    var zeros: [4096]f32 = @splat(0);
    for (0..8) |_| an.feed(&zeros);
    const r = an.analyzePoly();
    try std.testing.expectEqual(@as(u32, 0), r.active_count);
}

// ---------------------------------------------------------------------------
// Tuner thread integration
// ---------------------------------------------------------------------------

test "tuner thread: tap-fed sine is published, disable clears staleness" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const tun = try tuner.Tuner.create(allocator, io, 48000, 440.0, true);
    defer tun.destroy(allocator);

    var block: [480]f32 = undefined;
    var i: usize = 0;
    var found: ?tuner.Snapshot = null;
    // Feed ~real-time-ish and poll; bounded by sample count, not wall time.
    outer: while (i < 200) : (i += 1) {
        for (&block, 0..) |*s, j| {
            const t = @as(f64, @floatFromInt(i * block.len + j)) / 48000.0;
            s.* = @floatCast(0.25 * @sin(2.0 * std.math.pi * 196.5 * t));
        }
        tun.tap.push(&block);
        std.Io.sleep(io, .{ .nanoseconds = 5 * std.time.ns_per_ms }, .awake) catch {};
        const snap = tun.snapshot();
        if (snap.valid) {
            found = snap;
            if (i > 60) break :outer; // let smoothing settle a little
        }
    }
    const snap = found orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(i32, 55), snap.midi); // G3
    try std.testing.expectApproxEqAbs(@as(f32, 4.4), snap.cents, 2.0); // 196.5 Hz = G3 + 4.4 cents

    tun.setEnabled(false);
    try std.testing.expect(!tun.enabled());
}
