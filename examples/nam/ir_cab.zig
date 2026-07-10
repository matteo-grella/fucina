//! Cabinet impulse-response (IR) processing for the NAM example.
//!
//! A faithful port of AudioDSPTools `dsp::ImpulseResponse` + `ResampleCubic`
//! (refs/NeuralAmpModelerCore/Dependencies/AudioDSPTools/dsp/ImpulseResponse.{h,cpp},
//! Resample.h) — the same library this example already ports `NoiseGate` from.
//! It is the post-model "cab" stage: `NAM engine -> cab IR -> gate/output`.
//!
//! An IR cab is a fixed, *linear* FIR, so this is inference-only: no autograd,
//! no Fucina-framework dependency, no allocation in the realtime path (it
//! mirrors the sibling streaming engines). Everything stays example-local.
//!
//! Reference semantics reproduced exactly:
//!   - Direct time-domain convolution `y[n] = sum_j w[j]*x[n-j]` via a dot of
//!     the (reversed) weights with a sliding history window (no FFT —
//!     ImpulseResponse.cpp:43-47). FFT/partitioned convolution is unnecessary
//!     at cab-IR lengths and is deliberately not implemented.
//!   - The IR is REVERSED so weight[L-1] multiplies the newest sample
//!     (true convolution, IR[0]*x[n]; ImpulseResponse.cpp:81-82) — the same
//!     orientation `LinearEngine` uses (models.zig:352).
//!   - A fixed, sample-rate-compensated -18 dB headroom gain baked into the
//!     weights (ImpulseResponse.cpp:80) — NOT per-IR loudness normalization.
//!   - Cubic resampling to the session rate at load time, with one zero sample
//!     padded at each end (ImpulseResponse.cpp:59-73, Resample.h). Resampling a
//!     *linear* filter's impulse response is well-defined; the nonlinear .nam
//!     model still hard-rejects rate mismatch (it cannot be resampled).
//!   - IR truncated to 8192 taps (ImpulseResponse.h:46).
//!   - Mono only (wav.h ERROR_NOT_MONO); the result is broadcast to all output
//!     channels by the caller's mono pipeline.

const std = @import("std");
const wav = @import("wav.zig");

/// Upstream cap (ImpulseResponse.h:46 `mMaxLength`). Longer IRs are truncated.
pub const max_length: usize = 8192;

const vector_len: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;
const Vf32 = @Vector(vector_len, f32);

pub const Error = error{EmptyIr};

/// Fixed -18 dB headroom, sample-rate compensated (ImpulseResponse.cpp:80:
/// `pow(10, -18*0.05) * 48000/sampleRate`). Baked into the weights at load.
fn gainForRate(session_rate: u32) f32 {
    const minus18 = std.math.pow(f32, 10.0, -18.0 * 0.05);
    return minus18 * 48000.0 / @as(f32, @floatFromInt(session_rate));
}

/// Contiguous f32 dot product (both slices same length). The summation is
/// SIMD-blocked rather than strictly sequential, so it matches the reference
/// (Eigen's vectorized `dot`) to within the ecosystem's parity tolerance, not
/// bit-for-bit.
fn dot(w: []const f32, x: []const f32) f32 {
    std.debug.assert(w.len == x.len);
    var acc: Vf32 = @splat(0);
    var k: usize = 0;
    while (k + vector_len <= w.len) : (k += vector_len) {
        const wv: Vf32 = w[k..][0..vector_len].*;
        const xv: Vf32 = x[k..][0..vector_len].*;
        acc = @mulAdd(Vf32, wv, xv, acc);
    }
    var sum: f32 = @reduce(.Add, acc);
    while (k < w.len) : (k += 1) sum += w[k] * x[k];
    return sum;
}

/// Streaming direct-FIR cab. All buffers are allocated at `init`; `process`
/// is allocation-free, lock-free, and audio-thread-safe (its state is touched
/// only by the audio thread).
pub const IrCab = struct {
    allocator: std.mem.Allocator,
    /// Reversed + gained IR taps, length `taps`. `weight[taps-1]` multiplies
    /// the newest input sample.
    weight: []f32,
    /// The previous `taps-1` input samples (oldest first); the convolution
    /// carry between blocks. Empty when `taps == 1`.
    history: []f32,
    /// Scratch holding `history ++ this block`, sized `(taps-1)+max_frames`,
    /// so every output dot reads a contiguous window. Audio-thread only.
    work: []f32,
    taps: usize,
    max_frames: usize,

    /// Build a cab from raw mono IR samples (resampled to `session_rate` if
    /// needed). `max_frames` is the largest block `process` will be given.
    pub fn init(
        allocator: std.mem.Allocator,
        ir_samples: []const f32,
        ir_rate: u32,
        session_rate: u32,
        max_frames: usize,
    ) !IrCab {
        std.debug.assert(max_frames > 0);

        // Resample to the session rate (cubic), matching
        // ImpulseResponse::_SetWeights. Same rate => verbatim copy.
        const resampled = if (ir_rate == session_rate)
            try allocator.dupe(f32, ir_samples)
        else r: {
            // One zero sample padded at each end before resampling
            // (ImpulseResponse.cpp:67-71).
            const padded = try allocator.alloc(f32, ir_samples.len + 2);
            defer allocator.free(padded);
            padded[0] = 0;
            padded[padded.len - 1] = 0;
            @memcpy(padded[1 .. padded.len - 1], ir_samples);
            break :r try resampleIrCubic(allocator, padded, @floatFromInt(ir_rate), @floatFromInt(session_rate));
        };
        defer allocator.free(resampled);

        const taps = @min(resampled.len, max_length);
        if (taps == 0) return Error.EmptyIr;

        const gain = gainForRate(session_rate);
        const weight = try allocator.alloc(f32, taps);
        errdefer allocator.free(weight);
        // Reverse + gain: weight[taps-1-i] = gain*IR[i] (ImpulseResponse.cpp:81-82).
        for (0..taps) |i| weight[taps - 1 - i] = gain * resampled[i];

        const history = try allocator.alloc(f32, taps - 1);
        errdefer allocator.free(history);
        @memset(history, 0);

        const work = try allocator.alloc(f32, (taps - 1) + max_frames);
        errdefer allocator.free(work);

        return .{
            .allocator = allocator,
            .weight = weight,
            .history = history,
            .work = work,
            .taps = taps,
            .max_frames = max_frames,
        };
    }

    /// Load a mono .wav IR from disk and build the cab at `session_rate`.
    /// Non-mono files are rejected (wav.Error.NotMono), matching upstream.
    pub fn loadFile(
        io: std.Io,
        allocator: std.mem.Allocator,
        path: []const u8,
        session_rate: u32,
        max_frames: usize,
    ) !IrCab {
        var w = try wav.readFile(io, allocator, path);
        defer w.deinit();
        const mono = try w.requireMono();
        return init(allocator, mono, w.sample_rate, session_rate, max_frames);
    }

    pub fn deinit(self: *IrCab) void {
        self.allocator.free(self.weight);
        self.allocator.free(self.history);
        self.allocator.free(self.work);
        self.* = undefined;
    }

    /// Zero the convolution history.
    pub fn reset(self: *IrCab) void {
        @memset(self.history, 0);
    }

    /// Mono in -> mono out, `frames <= max_frames`; allocation-free. Safe for
    /// `input == output` (the block is copied into `work` before any output
    /// is written).
    pub fn process(self: *IrCab, input: []const f32, output: []f32, frames: usize) void {
        std.debug.assert(frames <= self.max_frames);
        const l = self.taps;
        const carry = l - 1;

        // Contiguous window: previous tail, then this block.
        @memcpy(self.work[0..carry], self.history[0..carry]);
        @memcpy(self.work[carry .. carry + frames], input[0..frames]);

        // y[i] = dot(weight, work[i .. i+l]); weight[l-1] hits work[i+l-1]
        // (the i-th new sample).
        for (0..frames) |i| {
            output[i] = dot(self.weight, self.work[i .. i + l]);
        }

        // Carry the trailing `carry` samples for the next block.
        @memcpy(self.history[0..carry], self.work[frames .. frames + carry]);
    }
};

/// Port of AudioDSPTools `dsp::ResampleCubic<float>` (Resample.h): cubic
/// (Catmull-Rom-style) interpolation from `original_rate` to `desired_rate`.
/// Produces points from the second input sample to the second-to-last,
/// exclusive (the cubic needs a neighbour on each side). Caller owns the
/// returned slice. The polynomial is evaluated in f64 and stored as f32, which
/// is how the reference's `float` template instantiation effectively behaves
/// (its constants promote the expression to double before the narrowing store).
pub fn resampleIrCubic(
    allocator: std.mem.Allocator,
    inputs: []const f32,
    original_rate: f64,
    desired_rate: f64,
) ![]f32 {
    var out: std.ArrayList(f32) = .empty;
    errdefer out.deinit(allocator);
    if (inputs.len < 2) {
        try out.appendSlice(allocator, inputs);
        return out.toOwnedSlice(allocator);
    }

    const time_increment = 1.0 / original_rate;
    const resampled_increment = 1.0 / desired_rate;
    const end_time = @as(f64, @floatFromInt(inputs.len - 1)) * time_increment;

    var time = time_increment; // tOutputStart = 0.0
    while (time < end_time) : (time += resampled_increment) {
        const index: usize = @intFromFloat(@floor(time / time_increment));
        const time_diff = time - @as(f64, @floatFromInt(index)) * time_increment;

        // Boundary clamps use `>=`, not the reference's `==` (Resample.h:69-70):
        // float accumulation of `time` can leave it one ULP below `end_time`
        // while `floor(time/inc)` still rounds to `len-1`, so `index+2` (and
        // `index+1`) can reach `len`/`len+1`. Upstream's identical `==` logic is
        // silent OOB UB behind a signed `long` + `std::vector::operator[]`; the
        // `>=` form is a deliberate, safer divergence (clamps to the last
        // sample) and changes no in-bounds result.
        var p: [4]f64 = undefined;
        p[0] = inputs[if (index == 0) 0 else index - 1];
        p[1] = inputs[index];
        p[2] = inputs[if (index + 1 >= inputs.len) inputs.len - 1 else index + 1];
        p[3] = inputs[if (index + 2 >= inputs.len) inputs.len - 1 else index + 2];

        const x = time_diff / time_increment;
        try out.append(allocator, @floatCast(cubicInterp(p, x)));
    }
    return out.toOwnedSlice(allocator);
}

/// Interpolate 4 equispaced points to x in [0,1) (Resample.h:28-34).
fn cubicInterp(p: [4]f64, x: f64) f64 {
    return p[1] + 0.5 * x * (p[2] - p[0] +
        x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] +
        x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}

test "ir cab: same-rate FIR matches gain-scaled direct convolution" {
    const allocator = std.testing.allocator;
    // weights[0] multiplies the newest sample (true convolution).
    const ir = [_]f32{ 0.5, -0.25, 0.125 };
    var cab = try IrCab.init(allocator, &ir, 48000, 48000, 64);
    defer cab.deinit();

    const input = [_]f32{ 1, 2, 3, 4 };
    var out: [4]f32 = undefined;
    cab.process(&input, &out, 4);

    // Direct convolution (same values as the LinearEngine known-values test),
    // scaled by the baked -18 dB IR gain.
    const gain = gainForRate(48000);
    const conv = [_]f32{ 0.5, 0.75, 1.125, 1.5 };
    for (conv, out) |c, o| try std.testing.expectApproxEqAbs(gain * c, o, 1e-5);
}

test {
    _ = @import("ir_cab_tests.zig");
}
