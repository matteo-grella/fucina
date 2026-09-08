//! Cabinet impulse-response (IR) processing for the NAM example.
//!
//! A port of AudioDSPTools `dsp::ImpulseResponse` + `ResampleCubic`
//! (refs/NeuralAmpModelerCore/Dependencies/AudioDSPTools/dsp/ImpulseResponse.{h,cpp},
//! Resample.h) — the same library this example already ports `NoiseGate` from.
//! It is the post-model "cab" stage: `NAM engine -> cab IR -> gate/output`.
//!
//! An IR cab is a fixed, linear FIR: the reversed, gained IR as one
//! single-channel causal conv over the core's streaming op
//! (`causalConv1dStreaming` with its `CausalState` context ring, the FIR
//! route of the kernel), on an `ExecContext` the cab owns. No autograd; no
//! allocation in the realtime path once the pool is warm.
//!
//! Reference semantics reproduced exactly:
//!   - Direct time-domain convolution `y[n] = sum_j w[j]*x[n-j]`
//!     (ImpulseResponse.cpp:43-47). FFT/partitioned convolution is
//!     unnecessary at cab-IR lengths and is deliberately not implemented.
//!   - The IR is REVERSED so weight[L-1] multiplies the newest sample
//!     (true convolution, IR[0]*x[n]; ImpulseResponse.cpp:81-82) — the same
//!     orientation the Linear engine uses.
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
const fucina = @import("fucina");
const wav = @import("wav.zig");
const wavenet = @import("wavenet.zig");

const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;
const CausalState = fucina.streamconv.CausalState;

/// Upstream cap (ImpulseResponse.h:46 `mMaxLength`). Longer IRs are truncated.
pub const max_length: usize = 8192;

pub const Error = error{EmptyIr};

/// Fixed -18 dB headroom, sample-rate compensated (ImpulseResponse.cpp:80:
/// `pow(10, -18*0.05) * 48000/sampleRate`). Baked into the weights at load.
fn gainForRate(session_rate: u32) f32 {
    const minus18 = std.math.pow(f32, 10.0, -18.0 * 0.05);
    return minus18 * 48000.0 / @as(f32, @floatFromInt(session_rate));
}

const Weight = Tensor(.{ .tap, .in, .out });

/// Streaming FIR cab. Everything is allocated at `init`; `process` is
/// allocation-free once the context's pool is warm and audio-thread-safe
/// (its state is touched only by the audio thread).
pub const IrCab = struct {
    allocator: std.mem.Allocator,
    ctx: *ExecContext,
    /// `[taps, 1, 1]`: reversed + gained IR taps; tap `taps-1` multiplies
    /// the newest input sample.
    weight: Weight,
    state: CausalState,
    input_slot: wavenet.InputSlot = .{},
    taps: usize,
    max_frames: usize,

    /// Build a cab from raw mono IR samples (resampled to `session_rate` if
    /// needed). `max_frames` is the block length the context ring is sized
    /// for (any block length stays correct).
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
        const ctx = try allocator.create(ExecContext);
        errdefer allocator.destroy(ctx);
        ctx.init(allocator);
        errdefer ctx.deinit();
        // Reverse + gain: weight[taps-1-i] = gain*IR[i] (ImpulseResponse.cpp:81-82),
        // as a flip and a scale of the resampled taps, owned by the cab.
        var taps_view = try Weight.fromBorrowedConstSlice(ctx, .{ taps, 1, 1 }, resampled[0..taps]);
        defer taps_view.deinit();
        var flipped = try taps_view.flip(ctx, .tap);
        defer flipped.deinit();
        var gained = try flipped.scale(ctx, gain);
        defer gained.deinit();
        var weight = try wavenet.own(Weight, ctx, &gained, false);
        errdefer weight.deinit();
        const state = try CausalState.init(allocator, 1, taps, 1, max_frames);
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .weight = weight,
            .state = state,
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
        self.input_slot.deinit();
        self.weight.deinit();
        self.state.deinit();
        self.ctx.deinit();
        self.allocator.destroy(self.ctx);
        self.* = undefined;
    }

    /// Zero the convolution history.
    pub fn reset(self: *IrCab) void {
        self.state.reset();
    }

    /// Mono in -> mono out. Safe for `input == output` (the block is copied
    /// into the input slot before any output is written).
    pub fn process(self: *IrCab, input: []const f32, output: []f32, frames: usize) !void {
        const ctx = self.ctx;
        const mark = ctx.openExecScope();
        defer ctx.closeExecScope(mark);
        const x = try self.input_slot.view(ctx, input, frames);
        const y = try x.causalConv1dStreaming(ctx, .time, .in, .tap, .out, &self.weight, null, 1, &self.state);
        try y.copyTo(output[0..frames]);
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
    try cab.process(&input, &out, 4);

    // Direct convolution (same values as the Linear known-values test),
    // scaled by the baked -18 dB IR gain.
    const gain = gainForRate(48000);
    const conv = [_]f32{ 0.5, 0.75, 1.125, 1.5 };
    for (conv, out) |c, o| try std.testing.expectApproxEqAbs(gain * c, o, 1e-5);
}

test {
    _ = @import("ir_cab_tests.zig");
}
