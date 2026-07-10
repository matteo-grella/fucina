//! NAM activation functions — ports of NeuralAmpModelerCore's
//! NAM/activations.h math. Applied in place over row-major [frames, channels]
//! buffers; PReLU is the only per-channel one. Like upstream tools/render, the
//! engine uses exact tanh by default (`tanhF32`, a SIMD evaluation that tracks
//! the scalar libm result to ≲2 ulp — see the contract comment on `tanhLanes`);
//! the rational `fast_tanh` approximation runs only when a model names
//! "Fasttanh" explicitly.

const std = @import("std");
const nam_file = @import("nam_file.zig");

const Activation = nam_file.Activation;

const tanh_vlen = std.simd.suggestVectorLength(f32) orelse 4;

/// Elementwise exact-contract tanh over N lanes.
///
/// Contract (what the parity gates rely on):
/// - Value-only dependence: a lane's result depends only on its input value,
///   never on lane position or N — so `tanhF32` (N=1) is bit-identical to the
///   bulk SIMD path and engine output stays byte-identical across block sizes.
/// - Deterministic across machines: IEEE add/mul/div/round and correctly
///   rounded @mulAdd only — no rcpps-style estimates, whose results differ
///   between CPU vendors. (Targets without FMA hardware get the same bits via
///   softfloat, just slower.)
/// - Accuracy: ≤ 3e-7 absolute vs the correctly rounded (f64) tanh over all of
///   f32 (measured ≲2 ulp; the activations test sweep enforces the bound) —
///   the same class as libm tanhf, ~20x inside the 5e-6 golden render gates.
/// - Specials: ±0 → ±0, subnormals → x, ±inf → ±1, NaN → NaN.
fn tanhLanes(comptime N: usize, x: @Vector(N, f32)) @Vector(N, f32) {
    const V = @Vector(N, f32);
    const U = @Vector(N, u32);
    const I = @Vector(N, i32);

    const sign_mask: U = @splat(0x8000_0000);
    const xb: U = @bitCast(x);
    const sign_bits = xb & sign_mask;
    const ax: V = @bitCast(xb & ~sign_mask);

    // Small branch, |x| <= 0.35: odd Taylor series through x^11,
    // tanh(x) = x·(1 + x²·(-1/3 + x²·(2/15 + ...))). Truncation ≤ 5e-9 at the
    // boundary; the x·(1+…) form keeps tanh(±0) = ±0 exactly. @mulAdd is the
    // correctly rounded IEEE fused op — bit-deterministic on every target
    // (softfloat where FMA hardware is missing), halves the poly op count.
    const t2 = x * x;
    var p: V = @splat(-1382.0 / 155925.0);
    p = @mulAdd(V, p, t2, @splat(62.0 / 2835.0));
    p = @mulAdd(V, p, t2, @splat(-17.0 / 315.0));
    p = @mulAdd(V, p, t2, @splat(2.0 / 15.0));
    p = @mulAdd(V, p, t2, @splat(-1.0 / 3.0));
    const small = x * @mulAdd(V, t2, p, @splat(1.0));

    // Large branch, |x| > 0.35: tanh|x| = 1 - 2/(e^{2|x|} + 1). The argument is
    // clamped at 18.5 (already past the point where the result rounds to 1.0,
    // reached at 2|x| ≥ 18.02) so k stays in [1, 27] and the 2^k exponent
    // arithmetic below cannot overflow, even for |x| = inf.
    const targ = @min(ax + ax, @as(V, @splat(18.5)));
    // e^t = 2^k · e^f with k = round(t·log2e), f = t - k·ln2 via the fdlibm
    // hi/lo split: ln2_hi has 8 trailing zero mantissa bits, so kf·ln2_hi is
    // exact for k ≤ 2^8 and the first subtraction cancels exactly.
    const kf = @round(targ * @as(V, @splat(1.4426950408889634)));
    const k: I = @intFromFloat(kf);
    var f = @mulAdd(V, kf, @splat(-0x1.62E4p-1), targ); // ln2_hi
    f = @mulAdd(V, kf, @splat(-0x1.7F7D1Cp-20), f); // ln2_lo
    // e^f on |f| ≤ ln2/2, Taylor degree 7 (truncation ≤ 5e-9 relative).
    var q: V = @splat(1.0 / 5040.0);
    q = @mulAdd(V, q, f, @splat(1.0 / 720.0));
    q = @mulAdd(V, q, f, @splat(1.0 / 120.0));
    q = @mulAdd(V, q, f, @splat(1.0 / 24.0));
    q = @mulAdd(V, q, f, @splat(1.0 / 6.0));
    q = @mulAdd(V, q, f, @splat(0.5));
    q = @mulAdd(V, q, f, @splat(1.0));
    q = @mulAdd(V, q, f, @splat(1.0));
    const scale: V = @bitCast((k + @as(I, @splat(127))) << @as(@Vector(N, u5), @splat(23)));
    const z = q * scale;
    const large = @as(V, @splat(1.0)) - @as(V, @splat(2.0)) / (z + @as(V, @splat(1.0)));
    const large_signed: V = @bitCast(@as(U, @bitCast(large)) | sign_bits);

    var r = @select(f32, ax <= @as(V, @splat(0.35)), small, large_signed);
    r = @select(f32, x != x, x, r); // NaN propagates (the clamp above would eat it)
    return r;
}

/// Scalar exact-contract tanh — bit-identical to the bulk SIMD path.
pub fn tanhF32(x: f32) f32 {
    return tanhLanes(1, .{x})[0];
}

fn tanhRows(data: []f32) void {
    const N = tanh_vlen;
    var i: usize = 0;
    while (i + N <= data.len) : (i += N) {
        const v: @Vector(N, f32) = data[i..][0..N].*;
        data[i..][0..N].* = tanhLanes(N, v);
    }
    if (i < data.len) {
        // Same kernel on a zero-padded tail: results are value-only, so
        // padding lanes cannot affect the written elements.
        var tail: [N]f32 = @splat(0);
        const rem = data.len - i;
        @memcpy(tail[0..rem], data[i..]);
        const out: [N]f32 = tanhLanes(N, tail);
        @memcpy(data[i..], out[0..rem]);
    }
}

/// activations.h:91-98 — five-constant rational approximation.
pub fn fastTanh(x: f32) f32 {
    const ax = @abs(x);
    const x2 = x * x;
    return (x * (2.45550750702956 + 2.45550750702956 * ax +
        (0.893229853513558 + 0.821226666969744 * ax) * x2) /
        (2.44506634652299 + (2.44506634652299 + x2) *
            @abs(x + 0.814642734961073 * x * ax)));
}

/// activations.h:100-103.
pub fn fastSigmoid(x: f32) f32 {
    return 0.5 * (fastTanh(x * 0.5) + 1.0);
}

/// activations.h:64-67 — 1/(1+expf(-x)).
pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn scalarApply(act: *const Activation, x: f32, channel: usize) f32 {
    return switch (act.kind) {
        .tanh => tanhF32(x),
        .hardtanh => std.math.clamp(x, -1.0, 1.0),
        .fasttanh => fastTanh(x),
        .relu => @max(x, 0.0),
        .leaky_relu => if (x >= 0) x else act.negative_slope * x,
        .prelu => blk: {
            const slope = if (act.negative_slopes.len > 0)
                act.negative_slopes[channel % act.negative_slopes.len]
            else
                act.negative_slope;
            break :blk if (x >= 0) x else slope * x;
        },
        .sigmoid => sigmoid(x),
        .silu => x * sigmoid(x),
        // activations.h:120-128 — multiply by 1/6, not divide.
        .hardswish => x * std.math.clamp(x + 3.0, 0.0, 6.0) * (1.0 / 6.0),
        .leaky_hardtanh => blk: {
            if (x < act.min_val) break :blk act.min_val + act.min_slope * (x - act.min_val);
            if (x > act.max_val) break :blk act.max_val + act.max_slope * (x - act.max_val);
            break :blk x;
        },
        .softsign => x / (1.0 + @abs(x)),
    };
}

/// In-place over a row-major [frames, channels] buffer.
pub fn applyRows(act: *const Activation, data: []f32, channels: usize) void {
    switch (act.kind) {
        .tanh => tanhRows(data),
        .prelu => {
            var i: usize = 0;
            while (i < data.len) : (i += 1) {
                data[i] = scalarApply(act, data[i], i % channels);
            }
        },
        else => for (data) |*v| {
            v.* = scalarApply(act, v.*, 0);
        },
    }
}

/// Gated path (gating_activations.h:53-113): the first `width` of each
/// `2*width` input row is the primary, the second the gate;
/// out = primary_act(z_top) * secondary_act(z_bottom). `z` is row-major
/// [frames, 2*width], `out` row-major [frames, width]; in-place-safe when
/// `out` aliases the head of `z` row-block-wise is NOT assumed — pass
/// distinct buffers.
pub fn applyGated(
    primary: *const Activation,
    secondary: *const Activation,
    z: []const f32,
    out: []f32,
    frames: usize,
    width: usize,
) void {
    for (0..frames) |t| {
        const row = z[t * 2 * width ..][0 .. 2 * width];
        const dst = out[t * width ..][0..width];
        for (0..width) |c| {
            dst[c] = scalarApply(primary, row[c], c) * scalarApply(secondary, row[width + c], c);
        }
    }
}

/// Blended path (gating_activations.h:188-204): the first `width` values are
/// both the pre-activation and the primary input; the second half produces
/// alpha through the secondary activation. `out = alpha*act(x) + (1-alpha)*x`.
pub fn applyBlended(
    primary: *const Activation,
    secondary: *const Activation,
    z: []const f32,
    out: []f32,
    frames: usize,
    width: usize,
) void {
    for (0..frames) |t| {
        const row = z[t * 2 * width ..][0 .. 2 * width];
        const dst = out[t * width ..][0..width];
        for (0..width) |c| {
            const pre = row[c];
            const activated = scalarApply(primary, pre, c);
            const alpha = scalarApply(secondary, row[width + c], c);
            dst[c] = alpha * activated + (1.0 - alpha) * pre;
        }
    }
}

test "activation known values" {
    const tanh_act = Activation{ .kind = .tanh };
    const hard = Activation{ .kind = .hardtanh };
    const leaky = Activation{ .kind = .leaky_relu, .negative_slope = 0.1 };
    const soft = Activation{ .kind = .softsign };
    const hswish = Activation{ .kind = .hardswish };

    try std.testing.expectApproxEqAbs(@as(f32, 0.7615941559557649), scalarApply(&tanh_act, 1.0, 0), 1e-6);
    try std.testing.expectEqual(@as(f32, 1.0), scalarApply(&hard, 2.5, 0));
    try std.testing.expectEqual(@as(f32, -1.0), scalarApply(&hard, -2.5, 0));
    try std.testing.expectApproxEqAbs(@as(f32, -0.2), scalarApply(&leaky, -2.0, 0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), scalarApply(&soft, 1.0, 0), 1e-7);
    // hardswish(-1) = -1 * 2/6
    try std.testing.expectApproxEqAbs(@as(f32, -1.0 / 3.0), scalarApply(&hswish, -1.0, 0), 1e-7);
    // fast_tanh is a close approximation, saturates like tanh
    try std.testing.expect(@abs(fastTanh(1.0) - 0.7615941559557649) < 5e-3);
    try std.testing.expect(@abs(fastTanh(10.0) - 1.0) < 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), fastSigmoid(0.0), 1e-7);
}

test "tanh activation matches libm tanh" {
    var x: f32 = -10.0;
    while (x <= 10.0) : (x += 0.0137) {
        const expected = std.math.tanh(x);
        const got = scalarApply(&Activation{ .kind = .tanh }, x, 0);
        try std.testing.expect(@abs(expected - got) <= 3e-7);
    }

    // The bulk SIMD path (incl. the padded tail: 13 % vector width != 0) is
    // bit-identical to the scalar contract.
    var data: [13]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = -3.0 + 0.5 * @as(f32, @floatFromInt(i));
    var expected: [13]f32 = undefined;
    for (&expected, data) |*e, v| e.* = tanhF32(v);
    const act = Activation{ .kind = .tanh };
    applyRows(&act, &data, 13);
    try std.testing.expectEqualSlices(f32, &expected, &data);
}

test {
    _ = @import("activations_tests.zig");
}
