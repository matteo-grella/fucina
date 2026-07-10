//! Live processing: guitar in -> chain of stages (NAM models + cab IRs) -> out,
//! inline in the audio callback (safe: a standard WaveNet costs ~67 us per
//! 64-frame block vs the ~1333 us real-time budget at 48 kHz — measured
//! 2026-06-12, M1 Max, one core; serial stages add up). All chains are
//! preloaded and prewarmed up front, so switching is one atomic index store and
//! the callback never allocates, locks, or touches the Fucina thread pool.

const std = @import("std");
const nam_file = @import("nam_file.zig");
const engine_mod = @import("engine.zig");
const ir_cab = @import("ir_cab.zig");
const audio_mod = @import("audio.zig");
const midi_mod = @import("midi.zig");
const tuner_mod = @import("tuner.zig");
const ui = @import("ui.zig");

/// Sentinel for "the MIDI port needs (re)connecting": parks the stored
/// source signature on a value no real topology hash will match, so the
/// rescan retries even when the topology reverted after a failed start.
const midi_reconnect_pending: i64 = std.math.minInt(i64);

// Control ranges shared by the keyboard steps and the absolute MIDI CC
// sweeps (CC 0..127 maps linearly in dB across the same span).
const out_gain_min_db = -40.0;
const out_gain_max_db = 24.0;
const in_gain_min_db = -20.0;
const in_gain_max_db = 40.0;
const gate_min_db = -90.0;
const gate_max_db = -30.0;

/// Cab-advisory classification of a NAM capture, derived once at load from
/// metadata.gear_type (+ a Tone3000 full-rig hint). The realtime callback never
/// reads this — it exists only for the load-time advisory.
pub const GearClass = enum { has_cab, needs_cab, neutral };

/// One processing stage. Tagged union -> single switch, no vtable. Each stage
/// OWNS its instance (single-owner; a file used twice is duplicate-loaded so an
/// instance is advanced at most once per block).
pub const Stage = union(enum) {
    nam: NamStage,
    cab: *ir_cab.IrCab,

    pub const NamStage = struct {
        engine: *engine_mod.Engine,
        gear: GearClass = .neutral,
    };
};

/// A stage plus its manifest pre-trim (linear gain applied to the stage's INPUT
/// before it runs). 1.0 = no trim. Not live-adjustable in v1; stage 0's trim
/// composes with the live global input-gain already baked into scratch_in.
pub const ChainStage = struct {
    stage: Stage,
    in_trim: f32 = 1.0,
};

pub const Chain = struct {
    name: []const u8,
    stages: []ChainStage,
    /// Loudness comp of the LAST NAM stage (1.0 if cab-only); read once in the
    /// output tail, same role as the old per-profile norm_gain.
    norm_gain: f32 = 1.0,
};

/// Single owner of all chains + their stage instances; one deinit tears it all
/// down. `chains` is allocator-backed; names + the per-chain ChainStage arrays
/// live in `arena`; each stage instance is heap-allocated and freed here.
pub const ChainSet = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    chains: []Chain,

    pub fn deinit(self: *ChainSet) void {
        for (self.chains) |*c| for (c.stages) |*cs| destroyStage(self.allocator, cs);
        self.allocator.free(self.chains);
        self.arena.deinit();
    }
};

pub fn destroyStage(a: std.mem.Allocator, cs: *ChainStage) void {
    switch (cs.stage) {
        .nam => |ns| {
            ns.engine.deinit();
            a.destroy(ns.engine);
        },
        .cab => |cab| {
            cab.deinit();
            a.destroy(cab);
        },
    }
}

pub const Shared = struct {
    /// Index into profiles; the callback loads it acquire.
    current: std.atomic.Value(usize) = .init(0),
    bypass: std.atomic.Value(bool) = .init(false),
    /// Output mute (silent tuning, guitar swaps). The chain still runs so
    /// the stateful WaveNet streams stay warm — only the device buffer is
    /// zeroed, so unmute is click-free.
    mute: std.atomic.Value(bool) = .init(false),
    /// Output gain as f32 bits.
    gain_bits: std.atomic.Value(u32) = .init(@bitCast(@as(f32, 1.0))),
    /// Input trim (drives the model harder/softer) as f32 bits.
    input_gain_bits: std.atomic.Value(u32) = .init(@bitCast(@as(f32, 1.0))),
    /// Loudness normalization to -18 dBFS from metadata (player feature;
    /// the engine itself never applies loudness — same as upstream core).
    normalize: std.atomic.Value(bool) = .init(true),
    /// Period-sized input scratch, touched only by the audio thread.
    scratch_in: []f32 = &.{},
    /// Gate enable + threshold (power-dB of the smoothed level) as f32 bits.
    gate_on: std.atomic.Value(bool) = .init(false),
    gate_threshold_db_bits: std.atomic.Value(u32) = .init(@bitCast(@as(f32, -65.0))),
    /// Per-sample linear gains computed by the trigger from the DRY input
    /// and applied to the WET output — audio thread only.
    gate_gains: []f32 = &.{},
    gate: NoiseGate = .{},
    gate_was_on: bool = false,
    in_peak_bits: std.atomic.Value(u32) = .init(0),
    out_peak_bits: std.atomic.Value(u32) = .init(0),
    clipped: std.atomic.Value(bool) = .init(false),
    /// Set when a callback observed a frame count above the configured
    /// period (would overrun the engines' max buffer).
    oversize_blocks: std.atomic.Value(usize) = .init(0),
    /// Selectable chains (a bare profile is a 1-stage chain). `current` indexes
    /// this; immutable descriptors, so hot-swap is one atomic index store.
    chains: []Chain = &.{},
    /// Two frame_cap ping-pong scratch buffers for the inter-stage signal.
    /// Audio-thread-only; distinct from scratch_in and from the output buffer.
    ping: [2][]f32 = .{ &.{}, &.{} },
    period: usize = 0,
    /// Tuner input tap (raw, pre-trim). Set before audio.start and never
    /// changed while the stream runs; the tap itself is wait-free and a
    /// single load when the tuner is off.
    tap: ?*tuner_mod.Tap = null,

    pub fn gain(self: *const Shared) f32 {
        return @bitCast(self.gain_bits.load(.monotonic));
    }

    pub fn setGain(self: *Shared, value: f32) void {
        self.gain_bits.store(@bitCast(value), .monotonic);
    }

    pub fn inputGain(self: *const Shared) f32 {
        return @bitCast(self.input_gain_bits.load(.monotonic));
    }

    pub fn setInputGain(self: *Shared, value: f32) void {
        self.input_gain_bits.store(@bitCast(value), .monotonic);
    }

    pub fn gateThresholdDb(self: *const Shared) f32 {
        return @bitCast(self.gate_threshold_db_bits.load(.monotonic));
    }

    pub fn setGateThresholdDb(self: *Shared, value: f32) void {
        self.gate_threshold_db_bits.store(@bitCast(value), .monotonic);
    }
};

/// Faithful port of the official NAM plugin's noise gate
/// (AudioDSPTools dsp/NoiseGate.{h,cpp}, MIT): a Trigger analyzes the DRY
/// input — one-pole power envelope (50 ms half-life), quadratic soft-knee
/// gain reduction -ratio*(levelDB-threshold)^2 below threshold, a
/// MOVING/HOLDING state machine with 2 ms open, 50 ms hold, 50 ms close —
/// and the resulting per-sample gains are applied to the WET (post-model)
/// output, so the model's amplified hiss is what actually gets cut.
/// Constants and update rules match upstream exactly, including the
/// 10^(dB/10) gain application and the 0.5x step toward target clamped by
/// the open/close rates.
const NoiseGate = struct {
    // TriggerParams(0.05, threshold, 1.5, 0.002, 0.050, 0.050) — the
    // plugin's guitar-tuned defaults; threshold is the user knob.
    const time_s: f64 = 0.05;
    const ratio: f64 = 1.5;
    const open_s: f64 = 0.002;
    const hold_s: f64 = 0.050;
    const close_s: f64 = 0.050;
    const min_loudness_db: f64 = -120.0;
    const min_loudness_power: f64 = 1e-12; // 10^(min_db/10)

    level: f64 = min_loudness_power,
    last_gr_db: f64 = 0,
    holding: bool = false,
    time_held: f64 = 0,
    // Rate-derived (set by prepare()).
    alpha: f64 = 0,
    beta: f64 = 0,
    dt: f64 = 0,

    pub fn prepare(self: *NoiseGate, sample_rate: f64) void {
        self.alpha = std.math.pow(f64, 0.5, 1.0 / (time_s * sample_rate));
        self.beta = 1.0 - self.alpha;
        self.dt = 1.0 / sample_rate;
        self.reset(-65.0);
    }

    pub fn reset(self: *NoiseGate, threshold_db: f64) void {
        self.level = min_loudness_power;
        self.last_gr_db = maxGainReduction(threshold_db);
        self.holding = false;
        self.time_held = 0;
    }

    fn gainReduction(threshold_db: f64, level_db: f64) f64 {
        if (level_db >= threshold_db) return 0.0;
        const d = level_db - threshold_db;
        return -ratio * d * d; // quadratic soft knee (upstream's curve)
    }

    fn maxGainReduction(threshold_db: f64) f64 {
        return gainReduction(threshold_db, min_loudness_db);
    }

    /// One sample of the trigger; returns the LINEAR gain to apply to the
    /// wet output (upstream's 10^(dB/10) convention).
    pub fn step(self: *NoiseGate, threshold_db: f64, x: f32) f32 {
        const max_gr = maxGainReduction(threshold_db);
        const d_open = -max_gr / open_s * self.dt; // > 0
        const d_close = max_gr / close_s * self.dt; // < 0

        const xx = @as(f64, x) * @as(f64, x);
        self.level = std.math.clamp(self.alpha * self.level + self.beta * xx, min_loudness_power, 1000.0);
        const level_db = 10.0 * std.math.log10(self.level);

        var gr_db: f64 = undefined;
        if (self.holding) {
            gr_db = 0.0;
            self.last_gr_db = 0.0;
            if (level_db < threshold_db) {
                self.time_held += self.dt;
                if (self.time_held >= hold_s) self.holding = false;
            } else {
                self.time_held = 0.0;
            }
        } else {
            const target = gainReduction(threshold_db, level_db);
            if (target > self.last_gr_db) {
                self.last_gr_db += std.math.clamp(0.5 * (target - self.last_gr_db), 0.0, d_open);
                if (self.last_gr_db >= 0.0) {
                    self.last_gr_db = 0.0;
                    self.holding = true;
                    self.time_held = 0.0;
                }
            } else if (target < self.last_gr_db) {
                self.last_gr_db += std.math.clamp(0.5 * (target - self.last_gr_db), d_close, 0.0);
                if (self.last_gr_db < max_gr) self.last_gr_db = max_gr;
            }
            gr_db = self.last_gr_db;
        }
        if (gr_db == 0.0) return 1.0;
        return @floatCast(std.math.pow(f64, 10.0, gr_db / 10.0));
    }
};

/// Lock-free peak-hold publish: raise the cell to the largest f32 the callback
/// has seen since the render thread last reset it. The reader swap()s it to 0
/// each tick — so the meter reflects the loudest of the ~24 blocks per 33 ms
/// frame instead of just the last one (otherwise it undersamples fast transients).
/// Peaks are non-negative, so their IEEE-754 bit patterns order like the values:
/// one unsigned atomic max publishes max(cell, value) in a single RMW, with no
/// lost update against the reader's swap(0) (a pre-load + early-return CAS could
/// skip the store when a reset interleaves the load).
fn atomicMaxF32(cell: *std.atomic.Value(u32), value: f32) void {
    _ = cell.fetchMax(@bitCast(value), .monotonic);
}

/// The realtime data callback. No allocation, no locks, no syscalls.
pub fn audioCallback(user: ?*anyopaque, output: ?[*]f32, input: ?[*]const f32, frame_count: c_uint) callconv(.c) void {
    const shared: *Shared = @ptrCast(@alignCast(user.?));
    const frames: usize = frame_count;
    const out_ptr = output orelse return;
    const out = out_ptr[0..frames];
    const in_ptr = input orelse {
        @memset(out, 0);
        return;
    };
    const raw_in = in_ptr[0..frames];

    // Tuner tap first: raw (pre-trim) input, so the reading is independent
    // of the trim knob; fed even for oversize blocks.
    if (shared.tap) |tap| tap.push(raw_in);

    if (frames > shared.period) {
        // Should not happen with fixed-size callbacks; refuse to overrun
        // the engine buffers (deviation 3) and pass dry audio through.
        _ = shared.oversize_blocks.fetchAdd(1, .monotonic);
        @memcpy(out, raw_in);
        return;
    }

    // Input trim BEFORE the model: an amp model is nonlinear, so this is
    // the "how hard you hit the amp" knob, not just volume. The meter
    // shows the trimmed level (what the model actually sees).
    const in_gain = shared.inputGain();
    const in = shared.scratch_in[0..frames];
    var in_peak: f32 = 0;
    for (in, raw_in) |*dst, v| {
        dst.* = v * in_gain;
        in_peak = @max(in_peak, @abs(dst.*));
    }
    atomicMaxF32(&shared.in_peak_bits, in_peak);

    // Noise gate trigger on the DRY input (upstream NAM-plugin split):
    // the decision uses the clean signal's dynamics; the gains land on
    // the post-model output below, where the amplified hiss lives.
    const gate_on = shared.gate_on.load(.monotonic);
    if (gate_on and !shared.gate_was_on) {
        shared.gate.reset(shared.gateThresholdDb());
    }
    shared.gate_was_on = gate_on;
    if (gate_on) {
        const threshold_db: f64 = @floatCast(shared.gateThresholdDb());
        for (in, shared.gate_gains[0..frames]) |v, *g| {
            g.* = shared.gate.step(threshold_db, v);
        }
    }

    const index = shared.current.load(.acquire);
    // Snapshot bypass once so a mid-block toggle can't split this block across
    // states (model on one path, cab/gate/gain on the other).
    const bypass = shared.bypass.load(.monotonic);
    const chain = &shared.chains[index];
    if (bypass) {
        @memcpy(out, in); // dry monitor: `in` is the globally-trimmed signal
    } else {
        runChain(shared, chain, in, out, frames);
    }

    // Apply the trigger's gains to the wet signal (skip in bypass: dry
    // monitoring stays untouched, like the plugin's bypass).
    if (gate_on and !bypass) {
        for (out, shared.gate_gains[0..frames]) |*v, g| v.* *= g;
    }

    // Mute replaces the output tail entirely: processing above already ran
    // (streaming state stays warm), the out meter honestly reads silence,
    // and no CLIP can latch from a signal nobody hears.
    if (shared.mute.load(.monotonic)) {
        @memset(out, 0);
        return;
    }

    var gain_value = shared.gain();
    if (shared.normalize.load(.monotonic) and !bypass) {
        gain_value *= chain.norm_gain;
    }
    var out_peak: f32 = 0;
    for (out) |*v| {
        v.* *= gain_value;
        const mag = @abs(v.*);
        out_peak = @max(out_peak, mag);
        if (mag >= 1.0) shared.clipped.store(true, .monotonic);
    }
    atomicMaxF32(&shared.out_peak_bits, out_peak);
}

/// Runs `chain.stages` over `in` (already globally-trimmed, len `frames`),
/// landing the final result in `out`. Allocation/lock/syscall-free. `in` is
/// shared.scratch_in — read-only here (the gate's DRY source + the in meter).
/// ping[0]/ping[1] are the inter-stage scratch.
///
/// Per stage we pick `dst != src` (the LAST stage's dst is `out`). A per-stage
/// trim scales the stage input: an owned inter-stage ping buffer is scaled in
/// place (no spare needed); stage 0's source is read-only `in`, so its trimmed
/// copy goes into the free ping buffer. A nam stage never runs src==dst —
/// WaveNet is not in-place safe (reads `input` across the pass, pushes history
/// after writing `output`); cabs are in-place safe but still get distinct
/// buffers here.
fn runChain(shared: *Shared, chain: *const Chain, in: []const f32, out: []f32, frames: usize) void {
    const stages = chain.stages;
    if (stages.len == 0) {
        @memcpy(out, in); // unreachable: empty chains are rejected at load
        return;
    }
    const p0 = shared.ping[0][0..frames];
    const p1 = shared.ping[1][0..frames];
    // The routing assumes these are four distinct allocations (out = device
    // buffer, in = scratch_in, p0/p1 = our ping). Pin that precondition.
    std.debug.assert(out.ptr != in.ptr and out.ptr != p0.ptr and out.ptr != p1.ptr and
        in.ptr != p0.ptr and in.ptr != p1.ptr and p0.ptr != p1.ptr);

    var src: []const f32 = in;
    var src_owned = false; // true once src is a ping buffer we may mutate
    for (stages, 0..) |*slot, si| {
        const last = si + 1 == stages.len;
        const dst: []f32 = if (last) out else if (src.ptr == p0.ptr) p1 else p0;

        var stage_in: []const f32 = src;
        if (slot.in_trim != 1.0) {
            if (src_owned) {
                // Scale the owned inter-stage buffer in place.
                const m: []f32 = if (src.ptr == p0.ptr) p0 else p1;
                for (m) |*v| v.* *= slot.in_trim;
                stage_in = m;
            } else {
                // src is read-only `in`: trimmed copy into the free ping (!= dst).
                const spare: []f32 = if (dst.ptr == p0.ptr) p1 else p0;
                for (spare, src) |*d, v| d.* = v * slot.in_trim;
                stage_in = spare;
            }
        }

        switch (slot.stage) {
            .nam => |ns| {
                std.debug.assert(stage_in.ptr != dst.ptr); // never in-place for nam
                ns.engine.process(stage_in, dst, frames) catch {
                    if (stage_in.ptr != dst.ptr) @memcpy(dst, stage_in);
                };
            },
            .cab => |cab| cab.process(stage_in, dst, frames),
        }
        src = dst;
        src_owned = true; // dst is a ping buffer (intermediate) or out (last)
    }
}

/// Map a NAM capture's gear_type (+ a Tone3000 full-rig hint) to a cab-advisory
/// class. Unknown / absent => neutral (no false alarms).
pub fn classifyGear(gear_type: ?[]const u8, full_rig_hint: bool) GearClass {
    if (full_rig_hint) return .has_cab;
    const gt = gear_type orelse return .neutral;
    const eq = std.ascii.eqlIgnoreCase;
    if (eq(gt, "amp_cab") or eq(gt, "amp_pedal_cab") or eq(gt, "studio") or eq(gt, "full-rig")) return .has_cab;
    if (eq(gt, "amp") or eq(gt, "preamp") or eq(gt, "pedal_amp")) return .needs_cab;
    return .neutral; // "pedal" or unknown/absent
}

pub const CabAdvice = struct { redundant_cab: bool = false, missing_cab: bool = false };

/// Pure, bi-directional cab-advisory decision over a built chain's ordered
/// stages: `redundant_cab` when a cab IR follows a NAM stage that already bakes
/// in a cab; `missing_cab` when the chain's last NAM stage is an amp/preamp with
/// no cab after it.
pub fn cabAdvice(stages: []const ChainStage) CabAdvice {
    var upstream_has_cab = false; // does the nearest upstream NAM stage bake in a cab?
    var last_nam: ?GearClass = null;
    var cab_after_last_nam = false;
    var redundant = false;
    for (stages) |cs| switch (cs.stage) {
        .nam => |n| {
            upstream_has_cab = n.gear == .has_cab;
            last_nam = n.gear;
            cab_after_last_nam = false;
        },
        .cab => {
            // Redundant only when the cab immediately follows a cab-inclusive
            // capture (so [has_cab amp, needs_cab amp, cab] is NOT flagged — the
            // cab serves the second amp).
            if (upstream_has_cab) redundant = true;
            upstream_has_cab = false;
            cab_after_last_nam = true;
        },
    };
    return .{ .redundant_cab = redundant, .missing_cab = last_nam == .needs_cab and !cab_after_last_nam };
}

/// Emit the (non-fatal) load-time cab advisory notes for `stages` to `out`.
pub fn adviseChain(out: *std.Io.Writer, name: []const u8, stages: []const ChainStage) !void {
    const a = cabAdvice(stages);
    if (a.redundant_cab)
        try out.print("[{s}] note: cab IR is redundant — an earlier capture already includes a speaker cab/mic; the doubled cab will sound dull/boxy\n", .{name});
    if (a.missing_cab)
        try out.print("[{s}] note: this chain ends in an amp/preamp capture with no speaker cab — add a cab IR (--ir cab.wav, or a cab stage) for a full tone\n", .{name});
}

pub const Options = struct {
    capture: ?usize = null,
    playback: ?usize = null,
    sample_rate: u32 = 48000,
    /// 64 frames = 1.3 ms per period; the engine has ~20x headroom even at
    /// this size, and every period saved counts 4x in the duplex path.
    period: u32 = 64,
    gain_db: f32 = 0.0,
    input_gain_db: f32 = 0.0,
    normalize: bool = true,
    /// Noise-gate threshold in dBFS on the (trimmed) input; null = off.
    gate_db: ?f32 = null,
    /// Probe every capture device at startup and pick the cleanest signal.
    auto_input: bool = false,
    /// MIDI control of the live knobs (--no-midi turns it off).
    midi: bool = true,
    /// Connect only this MIDI source (`devices` lists them); null = all
    /// sources, with a periodic rescan so hot-plugged controllers join.
    midi_source: ?usize = null,
    /// React only to this MIDI channel (1-16); null = omni.
    midi_channel: ?u8 = null,
    /// CC-number assignments (--midi-map name=cc,...).
    cc_map: midi_mod.CcMap = .{},
    /// Start with the tuner display on ('t' toggles it live).
    tuner: bool = false,
    /// A4 reference for the tuner (Hz).
    a4: f64 = tuner_mod.a4_default,
};

// ---------------------------------------------------------------------------
// Input auto-detection: record ~1.5 s from every capture device while the
// player plays, estimate signal (90th-percentile window RMS) and noise
// floor (10th percentile), and pick the device with the best SNR. A direct
// interface/DI signal has a near-silent floor between notes, so it beats a
// microphone hearing the same guitar acoustically; silent devices (incl.
// loopbacks, which hear nothing while we play nothing) are excluded.
// ---------------------------------------------------------------------------

/// Known virtual/loopback device-name fragments (case-insensitive): they
/// route application audio, not guitars, so the probe never selects them
/// (a loopback can carry a strong, clean signal and still be wrong).
const virtual_name_fragments = [_][]const u8{
    "teams", "vb-cable", "vb cable", "blackhole", "soundflower",
    "loopback", "zoomaudio", "camo", "aggregate", "virtual",
};

fn isVirtualDevice(name: []const u8) bool {
    var lower_buf: [audio_mod.name_cap]u8 = undefined;
    if (name.len > lower_buf.len) return false;
    const lower = std.ascii.lowerString(&lower_buf, name);
    for (virtual_name_fragments) |fragment| {
        if (std.mem.indexOf(u8, lower, fragment) != null) return true;
    }
    return false;
}

/// A playback device with the same name as a capture device = the two
/// sides of one physical interface (one clock, no drift).
fn findPlaybackTwin(name: []const u8, playback_devices: []const audio_mod.DeviceInfo) ?usize {
    for (playback_devices, 0..) |*info, index| {
        if (std.mem.eql(u8, info.nameSlice(), name)) return index;
    }
    return null;
}

pub const AutoInputResult = struct {
    capture: usize,
    /// Playback side of the same physical device, when it exists.
    playback_twin: ?usize,
};

const ProbeState = struct {
    samples: []f32,
    cursor: std.atomic.Value(usize) = .init(0),
};

fn probeCallback(user: ?*anyopaque, output: ?[*]f32, input: ?[*]const f32, frame_count: c_uint) callconv(.c) void {
    _ = output;
    const state: *ProbeState = @ptrCast(@alignCast(user.?));
    const in_ptr = input orelse return;
    const cursor = state.cursor.load(.monotonic);
    if (cursor >= state.samples.len) return;
    const n = @min(@as(usize, frame_count), state.samples.len - cursor);
    @memcpy(state.samples[cursor..][0..n], in_ptr[0..n]);
    state.cursor.store(cursor + n, .release);
}

const min_signal_db: f64 = -50.0;
/// Weak-signal path: real dynamics (snr) above a quiet floor still count —
/// a direct input with the gain knob low reads ~-60 dB but shows clear
/// playing dynamics, while an idle input shows none.
const weak_signal_db: f64 = -65.0;
const weak_snr_db: f64 = 6.0;

/// Returns the chosen capture index (+ its playback twin when the device
/// is a full interface), or null when no device carries signal.
pub fn autoDetectInput(
    io: std.Io,
    allocator: std.mem.Allocator,
    audio: *audio_mod.Audio,
    devices: []const audio_mod.DeviceInfo,
    playback_devices: []const audio_mod.DeviceInfo,
    options: Options,
) !?AutoInputResult {
    const probe_samples = options.sample_rate * 3 / 2; // 1.5 s per device
    const samples = try allocator.alloc(f32, probe_samples);
    defer allocator.free(samples);

    ui.plainLine(io, "auto-detecting input over {d} devices (~{d} s) — KEEP PLAYING your guitar...", .{ devices.len, devices.len * 2 });

    var best: ?usize = null;
    var best_score = -std.math.inf(f64);
    var best_is_interface = false;
    var best_weak = false;
    var best_p90: f64 = 0;
    for (devices, 0..) |*info, index| {
        if (isVirtualDevice(info.nameSlice())) {
            ui.plainLine(io, "  [{d}] {s:<32} (virtual device, skipped)", .{ index, info.nameSlice() });
            continue;
        }
        var state = ProbeState{ .samples = samples };
        audio.startCapture(index, options.sample_rate, options.period, probeCallback, &state) catch {
            ui.plainLine(io, "  [{d}] {s:<32} (failed to open, skipped)", .{ index, info.nameSlice() });
            continue;
        };
        // Wait for the buffer (with a timeout for stalled devices).
        var waited_ms: usize = 0;
        while (state.cursor.load(.acquire) < samples.len and waited_ms < 2500) : (waited_ms += 50) {
            std.Io.sleep(io, .{ .nanoseconds = 50 * std.time.ns_per_ms }, .awake) catch {};
        }
        audio.stop();
        const got = @min(state.cursor.load(.acquire), samples.len);
        if (got < options.sample_rate / 2) {
            ui.plainLine(io, "  [{d}] {s:<32} (no data, skipped)", .{ index, info.nameSlice() });
            continue;
        }

        const stats = windowStats(samples[0..got], options.sample_rate);
        const snr = stats.p90_db - stats.p10_db;
        const strong = stats.p90_db > min_signal_db;
        const weak = !strong and snr >= weak_snr_db and stats.p90_db > weak_signal_db;
        const eligible = strong or weak;
        const is_interface = findPlaybackTwin(info.nameSlice(), playback_devices) != null;
        const score = snr + 0.1 * stats.p90_db;
        ui.plainLine(io, "  [{d}] {s:<32} signal {d:>6.1} dB  floor {d:>6.1} dB  snr {d:>5.1} dB{s}{s}", .{
            index,
            info.nameSlice(),
            stats.p90_db,
            stats.p10_db,
            snr,
            if (is_interface) "  [interface]" else "",
            if (strong) "" else if (weak) "  (weak signal)" else "  (no signal)",
        });
        if (!eligible) continue;
        // Tiered choice: a full interface carrying ANY meaningful signal
        // beats every one-way source (webcam/builtin mic hearing the same
        // guitar acoustically) — within a tier, the cleanest wins.
        const better = if (is_interface != best_is_interface)
            is_interface
        else
            score > best_score;
        if (best == null or better) {
            best = index;
            best_score = score;
            best_is_interface = is_interface;
            best_weak = weak;
            best_p90 = stats.p90_db;
        }
    }

    if (best) |index| {
        const name = devices[index].nameSlice();
        ui.plainLine(io, "selected [{d}] {s}{s}", .{
            index, name, if (best_is_interface) " (audio interface preferred over microphones)" else " (cleanest signal)",
        });
        if (best_weak) {
            ui.plainLine(io, "note: its signal is weak ({d:.1} dB) — raise the interface's input gain, use input 1, engage Inst/Hi-Z; meanwhile bump the trim with '.'", .{best_p90});
        }
        return .{ .capture = index, .playback_twin = findPlaybackTwin(name, playback_devices) };
    }
    ui.plainLine(io, "no device carried signal — were you playing? keeping the current input", .{});
    return null;
}

const WindowStats = struct { p90_db: f64, p10_db: f64 };

/// 50 ms window RMS percentiles in dBFS.
fn windowStats(samples: []const f32, sample_rate: u32) WindowStats {
    const window = @max(@as(usize, 1), sample_rate / 20);
    var rms_db: [64]f64 = undefined;
    var count: usize = 0;
    var offset: usize = 0;
    while (offset + window <= samples.len and count < rms_db.len) : (offset += window) {
        var sum_sq: f64 = 0;
        for (samples[offset..][0..window]) |v| sum_sq += @as(f64, v) * v;
        const rms = @sqrt(sum_sq / @as(f64, @floatFromInt(window)));
        rms_db[count] = 20.0 * std.math.log10(@max(rms, 1e-9));
        count += 1;
    }
    if (count == 0) return .{ .p90_db = -180.0, .p10_db = -180.0 };
    std.mem.sort(f64, rms_db[0..count], {}, std.sort.asc(f64));
    return .{
        .p90_db = rms_db[@min(count - 1, count * 9 / 10)],
        .p10_db = rms_db[count / 10],
    };
}

/// Runs the live loop until 'q'. `profiles` must be preloaded engines
/// already reset(period, prewarm=true).
pub fn run(io: std.Io, allocator: std.mem.Allocator, shared: *Shared, audio: *audio_mod.Audio, options: Options) !void {
    var terminal = try ui.RawTerminal.enable();
    defer terminal.restore();

    // Enumerated once so 'i'/'o' can cycle devices at runtime.
    var capture_storage: [audio_mod.max_devices]audio_mod.DeviceInfo = undefined;
    var playback_storage: [audio_mod.max_devices]audio_mod.DeviceInfo = undefined;
    const capture_devices = try audio.listDevices(.capture, &capture_storage);
    const playback_devices = try audio.listDevices(.playback, &playback_storage);

    var capture_index = options.capture;
    var playback_index = options.playback;
    // Explicit --playback (and later manual 'o' cycling) wins over the
    // same-device suggestion; defaults get upgraded automatically.
    var playback_user_set = options.playback != null;
    var suggested_playback: ?usize = null;
    if (options.auto_input) {
        if (try autoDetectInput(io, allocator, audio, capture_devices, playback_devices, options)) |found| {
            capture_index = found.capture;
            applyTwin(io, found.playback_twin, playback_user_set, playback_devices, &playback_index, &suggested_playback);
        }
    }
    // Tuner analysis thread + tap. Created (and shared.tap set) BEFORE the
    // stream starts so the callback never races a plain-field write, and the
    // deferred destroy is registered before defer audio.stop, so LIFO order
    // stops the callbacks first and only then joins/frees the tap. Failure
    // to spawn degrades to no-tuner, never a failed session.
    const tun: ?*tuner_mod.Tuner = tuner_mod.Tuner.create(allocator, io, options.sample_rate, options.a4, options.tuner) catch |err| blk: {
        ui.plainLine(io, "tuner: unavailable ({s})", .{@errorName(err)});
        break :blk null;
    };
    defer if (tun) |t| t.destroy(allocator);
    if (tun) |t| shared.tap = &t.tap;

    try audio.start(capture_index, playback_index, options.sample_rate, options.period, audioCallback, shared);
    defer audio.stop();

    // The pinned status block reserves the bottom rows now, so every log line
    // below (device announce, MIDI, hot-plug) scrolls above it. When stdout is
    // not a tty the block stays inactive and run() falls back to one rewritten
    // status line. deinit() (resets the scroll region, shows the cursor) runs
    // before audio.stop() and the terminal restore via defer LIFO.
    var dash = ui.Dashboard.init(io);
    defer dash.deinit();
    announceDevices(io, audio, options);
    if (!dash.active) printKeys(io); // active: keys live in the pinned block

    // MIDI control is best-effort: no backend / no server / bad source
    // index degrades to keyboard-only with a note, never a failed start.
    var midi_stream = midi_mod.Stream{};
    var midi: ?midi_mod.Midi = null;
    defer if (midi) |*m| m.deinit();
    var midi_signature: i64 = midi_reconnect_pending;
    if (options.midi) {
        if (midi_mod.Midi.init()) |m| {
            midi = m;
            // Signature BEFORE start: a source arriving inside the start
            // window then reads as a change on the first rescan tick (one
            // redundant reconnect) instead of being baked unconnected into
            // the baseline forever.
            const signature = midi.?.sourcesSignature();
            const connected = midi.?.start(options.midi_source, &midi_stream) catch |err| blk: {
                ui.plainLine(io, "midi: could not open input ({s}) — keyboard control only", .{@errorName(err)});
                midi.?.deinit();
                midi = null;
                break :blk 0;
            };
            if (midi != null) {
                midi_signature = signature;
                announceMidi(io, &midi.?, connected, options);
            }
        } else |_| {
            // Non-macOS build or unreachable MIDI server: silently keyboard-only.
        }
    }

    var running = true;
    var silent_loops: usize = 0;
    var midi_rescan_loops: usize = 0;
    var midi_label_buf: [24]u8 = undefined;
    var midi_label: []const u8 = "";
    var midi_label_age: usize = 0;
    var gate_db: f32 = options.gate_db orelse -65.0;
    shared.gate.prepare(@floatFromInt(options.sample_rate));
    shared.setGateThresholdDb(gate_db);
    if (options.gate_db != null) shared.gate_on.store(true, .monotonic);
    // Decaying peak-hold meter state (render thread only): rises instantly to
    // the window peak, falls ~0.8 dB/tick (~24 dB/s) so the bar reads like a
    // hardware meter instead of strobing.
    var in_disp_db: f32 = -140;
    var out_disp_db: f32 = -140;
    const meter_decay_db: f32 = 0.8;
    // Tuner display state (render thread only): the last valid reading is
    // held ~0.5 s across attack transients so the needle doesn't flicker.
    var tuner_note_buf: [8]u8 = undefined;
    var tuner_held = ui.TunerView{};
    var tuner_held_age: usize = 1000;
    while (running) {
        while (terminal.poll()) |key| {
            switch (key) {
                'q', 3, 4 => running = false, // q / ^C / ^D (ISIG is off)
                ' ' => _ = shared.bypass.store(!shared.bypass.load(.monotonic), .monotonic),
                '[' => switchProfile(shared, -1),
                ']' => switchProfile(shared, 1),
                '1'...'9' => {
                    const slot: usize = key - '1';
                    if (slot < shared.chains.len) shared.current.store(slot, .release);
                },
                '+', '=' => adjustGain(shared, 1.0),
                '-' => adjustGain(shared, -1.0),
                ',' => adjustInputGain(shared, -1.0),
                '.' => adjustInputGain(shared, 1.0),
                'n' => _ = shared.normalize.store(!shared.normalize.load(.monotonic), .monotonic),
                'g' => _ = shared.gate_on.store(!shared.gate_on.load(.monotonic), .monotonic),
                't' => if (tun) |t| t.setEnabled(!t.enabled()),
                'm' => _ = shared.mute.store(!shared.mute.load(.monotonic), .monotonic),
                '<' => {
                    gate_db = @max(gate_db - 5.0, gate_min_db);
                    shared.setGateThresholdDb(gate_db);
                },
                '>' => {
                    gate_db = @min(gate_db + 5.0, gate_max_db);
                    shared.setGateThresholdDb(gate_db);
                },
                'c' => {
                    // Clear both latched warnings (clip + oversize); neither has
                    // any other reset path.
                    shared.clipped.store(false, .monotonic);
                    shared.oversize_blocks.store(0, .monotonic);
                },
                '?', 'h' => {
                    printKeys(io);
                    if (midi) |*m| announceMidi(io, m, null, options);
                },
                'i' => {
                    if (capture_devices.len > 0) {
                        capture_index = if (capture_index) |current| (current + 1) % capture_devices.len else 0;
                        switchDevices(io, audio, shared, options, &capture_index, &playback_index);
                        silent_loops = 0;
                    }
                },
                'o' => {
                    if (playback_devices.len > 0) {
                        playback_index = if (playback_index) |current| (current + 1) % playback_devices.len else 0;
                        playback_user_set = true;
                        suggested_playback = null;
                        switchDevices(io, audio, shared, options, &capture_index, &playback_index);
                        silent_loops = 0;
                    }
                },
                'a' => {
                    audio.stop();
                    if (autoDetectInput(io, allocator, audio, capture_devices, playback_devices, options) catch null) |found| {
                        capture_index = found.capture;
                        applyTwin(io, found.playback_twin, playback_user_set, playback_devices, &playback_index, &suggested_playback);
                    }
                    switchDevices(io, audio, shared, options, &capture_index, &playback_index);
                    silent_loops = 0;
                },
                'y' => {
                    if (suggested_playback) |twin| {
                        playback_index = twin;
                        suggested_playback = null;
                        switchDevices(io, audio, shared, options, &capture_index, &playback_index);
                    }
                },
                else => {},
            }
        }

        // Non-tty stdin reached EOF (piped/redirected, no terminal): there is no
        // way to receive a quit key, so exit cleanly instead of spinning. (Ctrl-D
        // on a real tty arrives as byte 4 above, handled as quit — not via eof.)
        if (terminal.eof and running) {
            ui.plainLine(io, "stdin is not an interactive terminal — exiting (run `live` from a terminal for keyboard control).", .{});
            running = false;
        }

        while (midi_stream.queue.pop()) |msg| {
            if (applyMidi(shared, &gate_db, options, msg, &midi_label_buf)) |label| {
                midi_label = label;
                midi_label_age = 0;
            }
        }
        midi_label_age += 1;

        // Hot-plug: CoreMIDI notifications need a CFRunLoop we never spin,
        // so poll the source IDENTITY (count + unique IDs — a bare count
        // misses an unplug+replug landing inside one window) every ~2 s and
        // reconnect in omni mode when it changes (turning a pedalboard on
        // after launch). A failed start has already torn down the old port,
        // so it parks the stored signature on the sentinel: the next tick
        // retries even if the topology reverted meanwhile.
        midi_rescan_loops += 1;
        if (midi != null and options.midi_source == null and midi_rescan_loops >= 60) {
            midi_rescan_loops = 0;
            const signature = midi.?.sourcesSignature();
            if (signature != midi_signature) {
                if (midi.?.start(null, &midi_stream)) |connected| {
                    midi_signature = signature;
                    announceMidi(io, &midi.?, connected, options);
                } else |err| {
                    if (midi_signature != midi_reconnect_pending) {
                        ui.plainLine(io, "midi: reconnect failed ({s}) — retrying every 2 s", .{@errorName(err)});
                    }
                    midi_signature = midi_reconnect_pending;
                }
            }
        }

        const index = shared.current.load(.monotonic);
        // swap()-to-0 drains the peak-hold the callback accumulated this frame;
        // the display rises to it instantly and decays otherwise.
        const in_peak: f32 = @bitCast(shared.in_peak_bits.swap(0, .monotonic));
        const out_peak: f32 = @bitCast(shared.out_peak_bits.swap(0, .monotonic));
        in_disp_db = @max(ui.dbfs(in_peak), in_disp_db - meter_decay_db);
        out_disp_db = @max(ui.dbfs(out_peak), out_disp_db - meter_decay_db);
        const gain_db = 20.0 * std.math.log10(@max(shared.gain(), 1e-6));
        const in_gain_db = 20.0 * std.math.log10(@max(shared.inputGain(), 1e-6));
        // ~5 s of dead input = almost certainly the wrong capture device or
        // a denied mic permission; surface a hint instead of staying mute.
        silent_loops = if (in_peak < 1e-6) silent_loops + 1 else 0;

        const tuner_on = if (tun) |t| t.enabled() else false;
        var tuner_view = ui.TunerView{};
        if (tun) |t| {
            // Follow the device's real rate (changes on 'i'/'o'/'a' switches).
            const rate = audio.actualSampleRate();
            if (rate != 0) t.setRate(rate);
            if (tuner_on) {
                const snap = t.snapshot();
                if (snap.poly_count >= 3) {
                    tuner_view.mode = .poly;
                    for (&tuner_view.strings, snap.strings) |*dst, s| {
                        dst.* = .{ .active = s.active, .cents = @floatCast(s.cents) };
                    }
                    tuner_held = tuner_view;
                    tuner_held_age = 0;
                } else if (snap.valid) {
                    tuner_view = .{
                        .mode = .mono,
                        .note = tuner_mod.noteLabel(snap.midi, &tuner_note_buf),
                        .cents = snap.cents,
                        .hz = snap.hz,
                    };
                    tuner_held = tuner_view;
                    tuner_held_age = 0;
                } else if (tuner_held_age < 15) {
                    tuner_held_age += 1;
                    tuner_view = tuner_held;
                }
            }
        }

        const view = ui.View{
            .chain_index = index,
            .chain_count = shared.chains.len,
            .chain_name = shared.chains[index].name,
            .stage_count = shared.chains[index].stages.len,
            .in_db = in_disp_db,
            .out_db = out_disp_db,
            .in_trim_db = in_gain_db,
            .out_gain_db = gain_db,
            .bypass = shared.bypass.load(.monotonic),
            .muted = shared.mute.load(.monotonic),
            .normalize = shared.normalize.load(.monotonic),
            .gate_on = shared.gate_on.load(.monotonic),
            .gate_db = gate_db,
            .clipped = shared.clipped.load(.monotonic),
            .oversize = shared.oversize_blocks.load(.monotonic) > 0,
            .silent = silent_loops > 150,
            // Transient echo of the last applied MIDI event (~1.5 s).
            .midi_label = if (midi_label.len > 0 and midi_label_age < 45) midi_label else "",
            .tuner_on = tuner_on,
            .tuner = tuner_view,
        };
        if (dash.active) dash.render(view) else ui.statusLineView(io, view);

        std.Io.sleep(io, .{ .nanoseconds = 33 * std.time.ns_per_ms }, .awake) catch {};
    }
    ui.plainLine(io, "", .{});
}

/// Applies (or suggests) using the detected input's playback twin as the
/// output: auto-applied when the user never chose an output, suggested
/// (press 'y') when they did.
fn applyTwin(
    io: std.Io,
    twin: ?usize,
    playback_user_set: bool,
    playback_devices: []const audio_mod.DeviceInfo,
    playback_index: *?usize,
    suggested_playback: *?usize,
) void {
    const twin_index = twin orelse return;
    if (playback_index.*) |current| {
        if (current == twin_index) return; // already there
    }
    if (!playback_user_set) {
        playback_index.* = twin_index;
        ui.plainLine(io, "output -> [{d}] {s} (same device as the input: one clock, no drift)", .{ twin_index, playback_devices[twin_index].nameSlice() });
    } else {
        suggested_playback.* = twin_index;
        ui.plainLine(io, "suggestion: the detected input is a full interface — press y to also use [{d}] {s} as the output (one clock, no drift)", .{ twin_index, playback_devices[twin_index].nameSlice() });
    }
}

fn announceDevices(io: std.Io, audio: *audio_mod.Audio, options: Options) void {
    var capture_name: [audio_mod.name_cap]u8 = undefined;
    var playback_name: [audio_mod.name_cap]u8 = undefined;
    audio.runningNames(&capture_name, &playback_name);
    const rate: f64 = @floatFromInt(audio.actualSampleRate());

    // Honest end-to-end estimate: real CoreAudio device latencies (device
    // latency + safety offset + device buffer per side) + the period +
    // miniaudio's duplex ring pre-seek (2x the INTERNAL capture period —
    // which can exceed the requested one; report the real value).
    const capture_dev: f64 = @floatFromInt(audio.deviceLatencyFrames(.capture));
    const playback_dev: f64 = @floatFromInt(audio.deviceLatencyFrames(.playback));
    const internal_capture: f64 = @floatFromInt(@max(audio.internalPeriodFrames(.capture), options.period));
    const period: f64 = @floatFromInt(options.period);
    const duplex_slack = 2.0 * internal_capture;
    const total_ms = (capture_dev + playback_dev + period + duplex_slack) / rate * 1000.0;
    ui.plainLine(io, "live: {s} -> {s} @ {d:.0} Hz, period {d} frames", .{
        std.mem.sliceTo(&capture_name, 0), std.mem.sliceTo(&playback_name, 0), rate, options.period,
    });
    if (capture_dev > 0 or playback_dev > 0) {
        ui.plainLine(io, "latency ~{d:.1} ms total: input device {d:.1} ms + duplex+period {d:.1} ms + output device {d:.1} ms", .{
            total_ms, capture_dev / rate * 1000.0, (period + duplex_slack) / rate * 1000.0, playback_dev / rate * 1000.0,
        });
    }
    // A driver that refuses small buffers dominates the latency; make it
    // visible instead of silently inflating the estimate.
    const negotiated = audio.internalPeriodFrames(.capture);
    if (negotiated > options.period * 2) {
        ui.plainLine(io, "warning: the input device negotiated {d}-frame buffers (asked {d}) — its driver imposes ~{d:.1} ms of duplex slack; lower its buffer size in the vendor panel if possible", .{
            negotiated, options.period, 2.0 * @as(f64, @floatFromInt(negotiated)) / rate * 1000.0,
        });
    }

    const capture_native = audio.internalSampleRate(.capture);
    const playback_native = audio.internalSampleRate(.playback);
    if (capture_native != 0 and capture_native != audio.actualSampleRate()) {
        ui.plainLine(io, "warning: input device runs at {d} Hz natively — macOS/miniaudio is resampling (extra latency). Set it to {d} Hz in Audio MIDI Setup.", .{ capture_native, audio.actualSampleRate() });
    }
    if (playback_native != 0 and playback_native != audio.actualSampleRate()) {
        ui.plainLine(io, "warning: output device runs at {d} Hz natively — macOS/miniaudio is resampling (extra latency). Set it to {d} Hz in Audio MIDI Setup.", .{ playback_native, audio.actualSampleRate() });
    }
}

/// Restarts the duplex stream on new device indices; on failure walks back
/// to the system defaults so the session keeps running.
fn switchDevices(io: std.Io, audio: *audio_mod.Audio, shared: *Shared, options: Options, capture_index: *?usize, playback_index: *?usize) void {
    audio.stop();
    audio.start(capture_index.*, playback_index.*, options.sample_rate, options.period, audioCallback, shared) catch {
        ui.plainLine(io, "device open failed (capture {?d}, playback {?d}); falling back to system defaults", .{ capture_index.*, playback_index.* });
        capture_index.* = null;
        playback_index.* = null;
        audio.start(null, null, options.sample_rate, options.period, audioCallback, shared) catch {
            ui.plainLine(io, "could not reopen any device; quit and re-run with --capture/--playback from `devices`", .{});
            return;
        };
    };
    announceDevices(io, audio, options);
}

fn printKeys(io: std.Io) void {
    ui.plainLine(io, "keys: space=bypass  m=mute  [ ]=profile  1-9=slot  a=auto-input  i/o=cycle in/out device  ,/.=input gain  +/-=output gain  g=gate  </>=gate threshold  t=tuner  n=normalize  c=clear warnings  ?=help  q=quit", .{});
}

/// Prints which sources we listen to (when `connected` is known) and the
/// active CC map — the live-mode counterpart of printKeys.
fn announceMidi(io: std.Io, midi: *midi_mod.Midi, connected: ?usize, options: Options) void {
    if (connected) |count| {
        if (count == 0) {
            ui.plainLine(io, "midi: no sources found — plug in a controller (rescans every 2 s)", .{});
        } else {
            var storage: [midi_mod.max_sources]midi_mod.SourceInfo = undefined;
            const sources = midi.listSources(&storage);
            if (options.midi_source) |index| {
                const name = if (index < sources.len) sources[index].nameSlice() else "?";
                ui.plainLine(io, "midi: listening on [{d}] {s}", .{ index, name });
            } else {
                ui.plainLine(io, "midi: listening on {d} source(s):", .{count});
                for (sources, 0..) |*info, index| {
                    ui.plainLine(io, "  [{d}] {s}", .{ index, info.nameSlice() });
                }
            }
        }
    }
    const map = options.cc_map;
    var channel_buf: [16]u8 = undefined;
    const channel_label = if (options.midi_channel) |ch|
        std.fmt.bufPrint(&channel_buf, "  (ch {d})", .{ch}) catch ""
    else
        "";
    ui.plainLine(io, "midi map: CC{d}=output gain  CC{d}=input trim  CC{d}=gate threshold  CC{d}=bypass  CC{d}=gate  CC{d}=normalize  CC{d}=mute  PC=profile slot{s}", .{
        map.out_gain, map.in_gain, map.gate_threshold, map.bypass, map.gate, map.normalize, map.mute, channel_label,
    });
}

fn switchProfile(shared: *Shared, direction: i64) void {
    const count = shared.chains.len;
    if (count == 0) return;
    const current = shared.current.load(.monotonic);
    const next = if (direction > 0)
        (current + 1) % count
    else
        (current + count - 1) % count;
    shared.current.store(next, .release);
}

fn adjustGain(shared: *Shared, delta_db: f32) void {
    const current_db = 20.0 * std.math.log10(@max(shared.gain(), 1e-6));
    const next_db = std.math.clamp(current_db + delta_db, out_gain_min_db, out_gain_max_db);
    shared.setGain(std.math.pow(f32, 10.0, next_db / 20.0));
}

fn adjustInputGain(shared: *Shared, delta_db: f32) void {
    const current_db = 20.0 * std.math.log10(@max(shared.inputGain(), 1e-6));
    const next_db = std.math.clamp(current_db + delta_db, in_gain_min_db, in_gain_max_db);
    shared.setInputGain(std.math.pow(f32, 10.0, next_db / 20.0));
}

fn dbToLinear(db: f32) f32 {
    return std.math.pow(f32, 10.0, db / 20.0);
}

/// Applies one MIDI message to the shared controls; returns a short label
/// for the status line when the message changed something. Runs on the UI
/// thread (the queue decouples it from CoreMIDI), so it may touch the same
/// state the keyboard handlers do — including run()'s gate_db.
fn applyMidi(shared: *Shared, gate_db: *f32, options: Options, msg: midi_mod.Message, label_buf: []u8) ?[]const u8 {
    if (options.midi_channel) |ch| {
        if (msg.channel() != ch - 1) return null;
    }
    switch (msg.kind()) {
        midi_mod.program_change => {
            if (msg.data1 < shared.chains.len) {
                shared.current.store(msg.data1, .release);
                return std.fmt.bufPrint(label_buf, "PC{d}", .{msg.data1}) catch null;
            }
            return null;
        },
        midi_mod.control_change => {
            const map = options.cc_map;
            const cc = msg.data1;
            const value = msg.data2;
            if (cc == map.out_gain) {
                shared.setGain(dbToLinear(midi_mod.ccToDb(value, out_gain_min_db, out_gain_max_db)));
            } else if (cc == map.in_gain) {
                shared.setInputGain(dbToLinear(midi_mod.ccToDb(value, in_gain_min_db, in_gain_max_db)));
            } else if (cc == map.gate_threshold) {
                gate_db.* = midi_mod.ccToDb(value, gate_min_db, gate_max_db);
                shared.setGateThresholdDb(gate_db.*);
            } else if (cc == map.bypass) {
                shared.bypass.store(midi_mod.ccOn(value), .monotonic);
            } else if (cc == map.gate) {
                shared.gate_on.store(midi_mod.ccOn(value), .monotonic);
            } else if (cc == map.normalize) {
                shared.normalize.store(midi_mod.ccOn(value), .monotonic);
            } else if (cc == map.mute) {
                shared.mute.store(midi_mod.ccOn(value), .monotonic);
            } else {
                return null;
            }
            return std.fmt.bufPrint(label_buf, "CC{d}={d}", .{ cc, value }) catch null;
        },
        else => return null,
    }
}

test "midi mapping: CC sweeps, switches, channel filter, PC slot bounds" {
    var chains = [_]Chain{
        .{ .name = "a", .stages = &.{} },
        .{ .name = "b", .stages = &.{} },
    };
    var shared = Shared{ .chains = &chains };
    var gate_db: f32 = -65.0;
    var buf: [24]u8 = undefined;
    const options = Options{};

    // Continuous controls map CC 0..127 onto the keyboard's dB ranges.
    try std.testing.expect(applyMidi(&shared, &gate_db, options, .{ .status = 0xB0, .data1 = 7, .data2 = 127 }, &buf) != null);
    try std.testing.expectApproxEqAbs(dbToLinear(out_gain_max_db), shared.gain(), 1e-4);
    try std.testing.expect(applyMidi(&shared, &gate_db, options, .{ .status = 0xB0, .data1 = 11, .data2 = 0 }, &buf) != null);
    try std.testing.expectApproxEqAbs(dbToLinear(in_gain_min_db), shared.inputGain(), 1e-4);
    try std.testing.expect(applyMidi(&shared, &gate_db, options, .{ .status = 0xB0, .data1 = 1, .data2 = 127 }, &buf) != null);
    try std.testing.expectApproxEqAbs(@as(f32, gate_max_db), gate_db, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, gate_max_db), shared.gateThresholdDb(), 1e-4);

    // Switches follow the sustain-pedal convention (>= 64 on).
    _ = applyMidi(&shared, &gate_db, options, .{ .status = 0xB0, .data1 = 64, .data2 = 127 }, &buf);
    try std.testing.expect(shared.bypass.load(.monotonic));
    _ = applyMidi(&shared, &gate_db, options, .{ .status = 0xB0, .data1 = 64, .data2 = 0 }, &buf);
    try std.testing.expect(!shared.bypass.load(.monotonic));
    _ = applyMidi(&shared, &gate_db, options, .{ .status = 0xB0, .data1 = 80, .data2 = 127 }, &buf);
    try std.testing.expect(shared.gate_on.load(.monotonic));
    _ = applyMidi(&shared, &gate_db, options, .{ .status = 0xB0, .data1 = 85, .data2 = 127 }, &buf);
    try std.testing.expect(shared.mute.load(.monotonic));
    _ = applyMidi(&shared, &gate_db, options, .{ .status = 0xB0, .data1 = 85, .data2 = 0 }, &buf);
    try std.testing.expect(!shared.mute.load(.monotonic));

    // Program change selects the slot; out-of-range is ignored.
    try std.testing.expect(applyMidi(&shared, &gate_db, options, .{ .status = 0xC0, .data1 = 1, .data2 = 0 }, &buf) != null);
    try std.testing.expectEqual(@as(usize, 1), shared.current.load(.monotonic));
    try std.testing.expect(applyMidi(&shared, &gate_db, options, .{ .status = 0xC0, .data1 = 9, .data2 = 0 }, &buf) == null);
    try std.testing.expectEqual(@as(usize, 1), shared.current.load(.monotonic));

    // Unmapped CCs do nothing; a channel filter mutes other channels.
    try std.testing.expect(applyMidi(&shared, &gate_db, options, .{ .status = 0xB0, .data1 = 23, .data2 = 64 }, &buf) == null);
    const filtered = Options{ .midi_channel = 2 };
    try std.testing.expect(applyMidi(&shared, &gate_db, filtered, .{ .status = 0xB0, .data1 = 7, .data2 = 0 }, &buf) == null);
    try std.testing.expect(applyMidi(&shared, &gate_db, filtered, .{ .status = 0xB1, .data1 = 7, .data2 = 0 }, &buf) != null);
    try std.testing.expectApproxEqAbs(dbToLinear(out_gain_min_db), shared.gain(), 1e-4);
}

test "noise gate port: opens fast on signal, holds, then closes on silence" {
    var gate = NoiseGate{};
    gate.prepare(48000.0);
    const threshold_db: f64 = -65.0;

    // Silence: stays heavily attenuated.
    var gain: f32 = 1.0;
    for (0..4800) |_| gain = gate.step(threshold_db, 0.0);
    try std.testing.expect(gain < 1e-4);

    // Loud signal (~-12 dBFS sine): opens to unity within ~10 ms
    // (2 ms ramp + envelope rise) and holds there.
    var t: usize = 0;
    while (t < 480) : (t += 1) {
        const x = 0.25 * @sin(@as(f32, @floatFromInt(t)) * 0.13);
        gain = gate.step(threshold_db, x);
    }
    try std.testing.expectEqual(@as(f32, 1.0), gain);

    // Back to silence: stays open well past the hold — the 50 ms
    // power-envelope half-life decays the level reading only ~3 dB per
    // 50 ms, so dropping from -15 dB to the -65 dB threshold takes
    // ~830 ms. That slow, musical release is upstream's behavior.
    for (0..24000) |_| gain = gate.step(threshold_db, 0.0);
    try std.testing.expectEqual(@as(f32, 1.0), gain);
    // ~1.5 s of silence total: envelope under threshold, hold elapsed,
    // close ramp done.
    for (0..48000) |_| gain = gate.step(threshold_db, 0.0);
    try std.testing.expect(gain < 1e-3);
}

test "runChain: 3 cab stages + mid trim == manual sequential; dry input preserved" {
    const allocator = std.testing.allocator;
    const frames = 32;
    const cap = 64;
    const p0 = try allocator.alloc(f32, cap);
    defer allocator.free(p0);
    const p1 = try allocator.alloc(f32, cap);
    defer allocator.free(p1);
    var shared = Shared{ .ping = .{ p0, p1 }, .period = cap };

    const ir_a = [_]f32{ 1.0, 0.5, -0.25 };
    const ir_b = [_]f32{ 0.8, -0.3 };
    const ir_c = [_]f32{ 0.6, 0.2, 0.1, -0.05 };
    // Chain instances and an identical manual set (same rate => no resample).
    var ca = try ir_cab.IrCab.init(allocator, &ir_a, 48000, 48000, cap);
    defer ca.deinit();
    var cb = try ir_cab.IrCab.init(allocator, &ir_b, 48000, 48000, cap);
    defer cb.deinit();
    var cc = try ir_cab.IrCab.init(allocator, &ir_c, 48000, 48000, cap);
    defer cc.deinit();
    var ma = try ir_cab.IrCab.init(allocator, &ir_a, 48000, 48000, cap);
    defer ma.deinit();
    var mb = try ir_cab.IrCab.init(allocator, &ir_b, 48000, 48000, cap);
    defer mb.deinit();
    var mc = try ir_cab.IrCab.init(allocator, &ir_c, 48000, 48000, cap);
    defer mc.deinit();

    const mid_trim: f32 = 0.5; // trim into the middle stage
    var stages = [_]ChainStage{
        .{ .stage = .{ .cab = &ca } },
        .{ .stage = .{ .cab = &cb }, .in_trim = mid_trim },
        .{ .stage = .{ .cab = &cc } },
    };
    const chain = Chain{ .name = "t", .stages = &stages };

    var input: [frames]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.3);
    const dry = input;

    var out: [frames]f32 = undefined;
    runChain(&shared, &chain, &input, &out, frames);

    // Manual sequential with the same per-stage trim.
    var t0: [frames]f32 = undefined;
    var t1: [frames]f32 = undefined;
    var t1b: [frames]f32 = undefined;
    var ref: [frames]f32 = undefined;
    ma.process(&input, &t0, frames);
    for (&t1, t0) |*d, v| d.* = v * mid_trim;
    mb.process(&t1, &t1b, frames);
    mc.process(&t1b, &ref, frames);

    try std.testing.expectEqualSlices(f32, &ref, &out);
    // The chain must never write its input (the gate's DRY source + in meter).
    try std.testing.expectEqualSlices(f32, &dry, &input);
}

test "audioCallback: mute zeroes the output but keeps the chain streaming" {
    const allocator = std.testing.allocator;
    const frames = 32;
    const cap = 64;
    const scratch_in = try allocator.alloc(f32, cap);
    defer allocator.free(scratch_in);
    const gate_gains = try allocator.alloc(f32, cap);
    defer allocator.free(gate_gains);
    const p0 = try allocator.alloc(f32, cap);
    defer allocator.free(p0);
    const p1 = try allocator.alloc(f32, cap);
    defer allocator.free(p1);

    // A 2-tap IR so the convolution carries history across blocks: the
    // muted block must still advance it (click-free unmute contract).
    const ir = [_]f32{ 0.5, 0.5 };
    var c = try ir_cab.IrCab.init(allocator, &ir, 48000, 48000, cap);
    defer c.deinit();
    var m = try ir_cab.IrCab.init(allocator, &ir, 48000, 48000, cap);
    defer m.deinit();
    var stages = [_]ChainStage{.{ .stage = .{ .cab = &c } }};
    var chains = [_]Chain{.{ .name = "t", .stages = &stages }};
    var shared = Shared{
        .chains = &chains,
        .period = cap,
        .scratch_in = scratch_in,
        .gate_gains = gate_gains,
        .ping = .{ p0, p1 },
    };

    var input: [frames]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.4);
    var out: [frames]f32 = undefined;

    audioCallback(&shared, &out, &input, frames);
    var nonzero = false;
    for (out) |v| nonzero = nonzero or v != 0;
    try std.testing.expect(nonzero);

    shared.mute.store(true, .monotonic);
    audioCallback(&shared, &out, &input, frames);
    for (out) |v| try std.testing.expectEqual(@as(f32, 0), v);

    // Unmuted again: identical to a reference instance that processed all
    // three blocks unmuted — proof the muted block advanced the stream.
    shared.mute.store(false, .monotonic);
    audioCallback(&shared, &out, &input, frames);
    var ref: [frames]f32 = undefined;
    m.process(&input, &ref, frames);
    m.process(&input, &ref, frames);
    m.process(&input, &ref, frames);
    try std.testing.expectEqualSlices(f32, &ref, &out);
}

test "runChain: single stage lands in out" {
    const allocator = std.testing.allocator;
    const frames = 16;
    const cap = 32;
    const p0 = try allocator.alloc(f32, cap);
    defer allocator.free(p0);
    const p1 = try allocator.alloc(f32, cap);
    defer allocator.free(p1);
    var shared = Shared{ .ping = .{ p0, p1 }, .period = cap };

    const ir = [_]f32{ 0.7, -0.2, 0.1 };
    var c = try ir_cab.IrCab.init(allocator, &ir, 48000, 48000, cap);
    defer c.deinit();
    var m = try ir_cab.IrCab.init(allocator, &ir, 48000, 48000, cap);
    defer m.deinit();

    var stages = [_]ChainStage{.{ .stage = .{ .cab = &c } }};
    const chain = Chain{ .name = "t", .stages = &stages };

    var input: [frames]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.5);
    var out: [frames]f32 = undefined;
    runChain(&shared, &chain, &input, &out, frames);

    var ref: [frames]f32 = undefined;
    m.process(&input, &ref, frames);
    try std.testing.expectEqualSlices(f32, &ref, &out);
}

test {
    _ = @import("live_tests.zig");
}
