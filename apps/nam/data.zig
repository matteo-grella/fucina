//! Training-data pipeline: capture-file version detection, blip latency
//! calibration, data checks, and dataset windowing — faithful ports of
//! neural-amp-modeler@a11ed88 nam/train/core.py + nam/data.py (inline
//! file:line cites refer to those upstream sources).
//!
//! v1 supports the v3 standardized capture file (the current upstream
//! standard; v1/v2/v4 are deprecated upstream and refused) plus a generic
//! input/output pair mode with manual latency.

const std = @import("std");

pub const standard_sample_rate: f64 = 48000.0;

/// V3 capture-signal structure (_V3_DATA_INFO, core.py:301-320), all in
/// samples @48k. Total file length 9,120,000 (3:10).
pub const v3 = struct {
    pub const rate: u32 = 48000;
    pub const t_blips: usize = 96000;
    pub const first_blips_start: usize = 480000;
    pub const t_validate: usize = 432000;
    pub const train_start: usize = 480000;
    /// Negative offset from the end.
    pub const validation_start: usize = 432000;
    pub const noise_interval_start: usize = 492000;
    pub const noise_interval_stop: usize = 498000;
    pub const blip_locations = [_]usize{ 504000, 552000 };
};

/// Strong-match MD5s of the standardized capture files (core.py:90-96).
pub const InputVersion = enum { v1_0_0, v1_1_1, v2_0_0, v3_0_0, v4_0_0, unknown };

pub fn detectInputVersion(file_bytes: []const u8) InputVersion {
    var digest: [16]u8 = undefined;
    std.crypto.hash.Md5.hash(file_bytes, &digest, .{});
    var hex_buf: [32]u8 = undefined;
    const hex = std.fmt.bufPrint(&hex_buf, "{x}", .{digest}) catch unreachable;

    const table = .{
        .{ "4d54a958861bf720ec4637f43d44a7ef", InputVersion.v1_0_0 },
        .{ "7c3b6119c74465f79d96c761a0e27370", InputVersion.v1_1_1 },
        .{ "ede3b9d82135ce10c7ace3bb27469422", InputVersion.v2_0_0 },
        .{ "36cd1af62985c2fac3e654333e36431e", InputVersion.v3_0_0 },
        .{ "80e224bd5622fd6153ff1fd9f34cb3bd", InputVersion.v4_0_0 },
    };
    inline for (table) |entry| {
        if (std.mem.eql(u8, hex, entry[0])) return entry[1];
    }
    return .unknown;
}

// ---------------------------------------------------------------------------
// ESR (losses.py:30-51, eq. 10 of mdpi 2076-3417/10/3/766): for single
// sequences, mean((a-b)^2)/mean(b^2) with b the reference/target.
// ---------------------------------------------------------------------------

pub fn esr(pred: []const f32, target: []const f32) f64 {
    std.debug.assert(pred.len == target.len and pred.len > 0);
    var num: f64 = 0;
    var den: f64 = 0;
    for (pred, target) |p, t| {
        const d = @as(f64, p) - @as(f64, t);
        num += d * d;
        den += @as(f64, t) * @as(f64, t);
    }
    if (den == 0) return std.math.inf(f64);
    return num / den;
}

/// The console quality bands (core.py:988-997).
pub fn esrComment(value: f64) []const u8 {
    if (value < 0.01) return "Great!";
    if (value < 0.035) return "Not bad!";
    if (value < 0.1) return "...This *might* sound ok!";
    if (value < 0.3) return "...This probably won't sound great :(";
    return "...Something seems to have gone wrong.";
}

// ---------------------------------------------------------------------------
// Latency calibration (_calibrate_latency_v_all, core.py:359-499): exact
// port, operating only on the reamp output y (full file).
// ---------------------------------------------------------------------------

pub const LatencyCalibration = struct {
    /// Algorithm v1 averages the blip scans into a single delay
    /// (core.py:440-476); the metadata schema stores it as a 1-element list.
    delay: ?i64,
    recommended: ?i64,
    warn_not_detected: bool,
    warn_matches_lookahead: bool,
    warn_disagreement_too_high: bool,

    pub const safety_factor: i64 = 1;
    pub const abs_threshold: f32 = 0.0003;
    pub const rel_threshold: f32 = 1.001;
    pub const lookahead: usize = 1000;
    pub const lookback: usize = 10000;
};

/// Calibrates against the v3 structure. `y` is the full reamp recording.
pub fn calibrateLatencyV3(y: []const f32) LatencyCalibration {
    const la = LatencyCalibration.lookahead;
    const lb = LatencyCalibration.lookback;
    if (y.len < v3.first_blips_start + v3.t_blips) {
        return .{ .delay = null, .recommended = null, .warn_not_detected = true, .warn_matches_lookahead = false, .warn_disagreement_too_high = false };
    }
    const blips = y[v3.first_blips_start .. v3.first_blips_start + v3.t_blips];

    // Trigger level from the known-silent stretch before the first blip
    // (core.py:419-431).
    var background: f32 = 0;
    for (blips[v3.noise_interval_start - v3.first_blips_start .. v3.noise_interval_stop - v3.first_blips_start]) |v| {
        background = @max(background, @abs(v));
    }
    const trigger = @max(background + LatencyCalibration.abs_threshold, LatencyCalibration.rel_threshold * background);

    // Average the per-blip scan windows elementwise, then find the first
    // sample above the trigger (core.py:434-441).
    var scan_average: [la + lb]f32 = undefined;
    @memset(&scan_average, 0);
    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(v3.blip_locations.len));
    for (v3.blip_locations) |i_abs| {
        const i_rel = i_abs - v3.first_blips_start;
        const window = blips[i_rel - la .. i_rel + lb];
        for (&scan_average, window) |*dst, v| dst.* += v * inv_n;
    }
    var first_trigger: ?usize = null;
    for (scan_average, 0..) |v, idx| {
        if (@abs(v) > trigger) {
            first_trigger = idx;
            break;
        }
    }

    const triggered = first_trigger orelse {
        return .{ .delay = null, .recommended = null, .warn_not_detected = true, .warn_matches_lookahead = false, .warn_disagreement_too_high = false };
    };
    // delay = first trigger relative to the nominal blip center
    // (core.py:476); recommended = delay - safety factor (core.py:478).
    const delay = @as(i64, @intCast(triggered)) - @as(i64, @intCast(la));
    return .{
        .delay = delay,
        .recommended = delay - LatencyCalibration.safety_factor,
        .warn_not_detected = false,
        .warn_matches_lookahead = delay == -@as(i64, @intCast(la)),
        .warn_disagreement_too_high = false,
    };
}

// ---------------------------------------------------------------------------
// v3 data check (_check_v3, core.py:729-754): the two validation replicates
// must sound the same (self-ESR <= 0.01). The output is end-cropped to the
// input's length first; the delay is NOT applied (both replicates shift
// identically).
// ---------------------------------------------------------------------------

pub const v3_check_threshold: f64 = 0.01;

pub fn checkV3(input_len: usize, y: []const f32) struct { passed: bool, replicate_esr: f64 } {
    const n = @min(input_len, y.len);
    if (n < v3.t_validate * 2) return .{ .passed = false, .replicate_esr = std.math.inf(f64) };
    const val_1 = y[0..v3.t_validate];
    const val_2 = y[n - v3.t_validate .. n];
    const replicate = esr(val_1, val_2);
    return .{ .passed = replicate <= v3_check_threshold, .replicate_esr = replicate };
}

// ---------------------------------------------------------------------------
// Dataset: aligned x/y windows with upstream's nx/ny semantics
// (data.py:390-504). Construction order matches upstream: split trim, then
// delay, then validations.
// ---------------------------------------------------------------------------

pub const DataError = error{
    UnequalLengths,
    OutputClipped,
    InputPreSilenceMissing,
    TooShort,
    BadDelay,
};

/// 0.4 s (data.py:316).
pub const require_input_pre_silence_seconds: f64 = 0.4;

pub const Split = struct {
    /// Absolute sample bounds into the aligned pair (after delay).
    start: usize,
    stop: usize,
};

pub const Dataset = struct {
    /// Aligned views into caller-owned buffers.
    x: []const f32,
    y: []const f32,
    nx: usize,
    ny: usize,

    /// Number of non-overlapping examples (tail dropped) — data.py:374.
    pub fn len(self: *const Dataset) usize {
        if (self.x.len < self.nx) return 0;
        return (self.x.len - self.nx + 1) / self.ny;
    }

    /// data.py:488-504: input window (nx+ny-1), target window (ny) offset
    /// by nx-1.
    pub fn get(self: *const Dataset, idx: usize) struct { input: []const f32, target: []const f32 } {
        const i = idx * self.ny;
        return .{
            .input = self.x[i .. i + self.nx + self.ny - 1],
            .target = self.y[i + self.nx - 1 .. i + self.nx - 1 + self.ny],
        };
    }
};

/// Applies the calibrated/manual delay to align x and y
/// (Dataset._apply_delay_int, data.py:646-655).
pub fn applyDelay(x: []const f32, y: []const f32, delay: i64) DataError!struct { x: []const f32, y: []const f32 } {
    const n = @min(x.len, y.len);
    const xt = x[0..n];
    const yt = y[0..n];
    if (delay == 0) return .{ .x = xt, .y = yt };
    const magnitude: usize = @intCast(@abs(delay));
    if (magnitude >= n) return DataError.BadDelay;
    if (delay > 0) {
        return .{ .x = xt[0 .. n - magnitude], .y = yt[magnitude..] };
    }
    return .{ .x = xt[magnitude..], .y = yt[0 .. n - magnitude] };
}

/// Output clipping check (data.py:793-797) — runs on the raw target.
pub fn checkOutputNotClipped(y: []const f32) DataError!void {
    for (y) |v| {
        if (@abs(v) >= 1.0) return DataError.OutputClipped;
    }
}

/// Pre-silence requirement (data.py:799-836): the `seconds` of input before
/// `split_start` must be exactly zero.
pub fn checkInputPreSilence(x: []const f32, split_start: usize, sample_rate: f64) DataError!void {
    const need: usize = @intFromFloat(require_input_pre_silence_seconds * sample_rate);
    if (split_start < need) return DataError.InputPreSilenceMissing;
    for (x[split_start - need .. split_start]) |v| {
        if (v != 0.0) return DataError.InputPreSilenceMissing;
    }
}

test {
    _ = @import("data_tests.zig");
}
