//! Repo-owned deterministic RNG for training (splitmix64-based).
//!
//! The (seed -> values) mapping of every function here is a CHECKPOINT
//! CONTRACT: APOLLO projections (optim.zig) are regenerated from their stored
//! seed instead of being serialized, and dropout masks (exec.zig) are
//! regenerated from their stored seed instead of being kept alive — so none of
//! this may depend on std.Random internals, which are free to change across
//! Zig releases. splitmix64 is additionally counter-based (`at`): the i-th
//! output of a stream is a pure function of (seed, i), which is what makes
//! dropout elements independently computable (parallel kernels, checkpoint
//! recompute) with bitwise-stable results.

const std = @import("std");

/// splitmix64 golden-gamma increment (2^64 / phi, Steele et al. 2014).
const gamma: u64 = 0x9E3779B97F4A7C15;

/// One splitmix64 step: advance `state` by the golden gamma, then mix.
/// Sequential form of `at` (the i-th output equals `at(initial_state, i)`).
pub fn splitmix64(state: *u64) u64 {
    state.* +%= gamma;
    var z = state.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// Counter-based access into the splitmix64 stream started at `seed`: the
/// i-th sequential output computed directly, `mix(seed +% (i+1) *% gamma)`.
/// O(1) per element and independent across `i` — any element of a stream can
/// be computed without the preceding ones.
pub fn at(seed: u64, i: u64) u64 {
    var z = seed +% (i +% 1) *% gamma;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// Repo-owned Gaussian fill (splitmix64 + Box-Muller). Consumes two stream
/// outputs per value pair: `u_a = ((a >> 11) + 1) * 2^-53` in (0, 1] and
/// `u_b = (b >> 11) * 2^-53` in [0, 1), then
/// `out[i] = sqrt(-2 ln u_a) * cos(2 pi u_b) * scale` and
/// `out[i+1] = sqrt(-2 ln u_a) * sin(2 pi u_b) * scale`.
/// The (seed -> values) mapping is part of the APOLLO checkpoint contract —
/// projections are regenerated, not stored.
pub fn gaussianFill(seed: u64, out: []f32, scale: f32) void {
    var state = seed;
    var i: usize = 0;
    while (i < out.len) : (i += 2) {
        const a = splitmix64(&state);
        const b = splitmix64(&state);
        const uniform_a = (@as(f64, @floatFromInt(a >> 11)) + 1) * 0x1.0p-53; // (0, 1]
        const uniform_b = @as(f64, @floatFromInt(b >> 11)) * 0x1.0p-53; // [0, 1)
        const radius = @sqrt(-2.0 * @log(uniform_a));
        const angle = 2.0 * std.math.pi * uniform_b;
        out[i] = @floatCast(radius * @cos(angle) * scale);
        if (i + 1 < out.len) out[i + 1] = @floatCast(radius * @sin(angle) * scale);
    }
}

/// Counter-based `gaussianFill`: fill `out` with elements `first ..
/// first + out.len` of the gaussian stream that `gaussianFill(seed, ...)`
/// produces, bitwise identically. Element j of that stream is a pure function
/// of the aligned Box-Muller pair (at(seed, 2*(j/2)), at(seed, 2*(j/2)+1)):
/// even j takes the cos half, odd j the sin half. Any range decomposition
/// therefore reproduces the sequential fill exactly — this is what makes ES
/// perturbations (es.zig) regenerable in parallel chunks with results
/// independent of the chunking. Same checkpoint contract as `gaussianFill`.
pub fn gaussianFillAt(seed: u64, first: u64, out: []f32, scale: f32) void {
    var j = first;
    var o: usize = 0;
    while (o < out.len) {
        const pair = j & ~@as(u64, 1);
        const a = at(seed, pair);
        const b = at(seed, pair + 1);
        const uniform_a = (@as(f64, @floatFromInt(a >> 11)) + 1) * 0x1.0p-53; // (0, 1]
        const uniform_b = @as(f64, @floatFromInt(b >> 11)) * 0x1.0p-53; // [0, 1)
        const radius = @sqrt(-2.0 * @log(uniform_a));
        const angle = 2.0 * std.math.pi * uniform_b;
        if (j % 2 == 0) {
            out[o] = @floatCast(radius * @cos(angle) * scale);
            o += 1;
            j += 1;
            if (o < out.len) {
                out[o] = @floatCast(radius * @sin(angle) * scale);
                o += 1;
                j += 1;
            }
        } else {
            out[o] = @floatCast(radius * @sin(angle) * scale);
            o += 1;
            j += 1;
        }
    }
}

/// FAST counter-based gaussian: the same splitmix64 pair stream and 53-bit
/// uniform construction as `gaussianFillAt`, with the f64 libm
/// transcendentals replaced by f32 polynomial ln/sin/cos (Cephes
/// coefficients, ~1-2 ulp) evaluated 8 pairs per step with `@Vector` lanes.
/// This is a DISTINCT (seed -> values) mapping from `gaussianFillAt`:
/// values agree to a few f32 ulps but are not bitwise equal, so the two
/// functions are separate checkpoint contracts — this one is the ES noise
/// contract (es.zig); APOLLO/dropout stay on the scalar mapping.
/// Chunking-invariant like `gaussianFillAt`: each pair is a pure function
/// of its counter index, and the vector body and scalar edges run the
/// identical elementwise ops (IEEE lane-exact), so any range decomposition
/// reproduces the same bits.
pub fn gaussianFillAtFast(seed: u64, first: u64, out: []f32, scale: f32) void {
    const w = fast_gaussian_width;
    var j = first;
    var o: usize = 0;
    // Head: an odd start consumes only the sin half of its pair.
    if (o < out.len and j % 2 == 1) {
        const pair = fastGaussianPairs(1, seed, .{j - 1});
        out[o] = pair.odd[0] * scale;
        o += 1;
        j += 1;
    }
    // Body: w pairs (2w values) at a time.
    while (o + 2 * w <= out.len) {
        var pair_index: @Vector(w, u64) = undefined;
        inline for (0..w) |lane| pair_index[lane] = j + 2 * lane;
        const pair = fastGaussianPairs(w, seed, pair_index);
        inline for (0..w) |lane| {
            out[o + 2 * lane] = pair.even[lane] * scale;
            out[o + 2 * lane + 1] = pair.odd[lane] * scale;
        }
        o += 2 * w;
        j += 2 * w;
    }
    // Tail: one pair at a time (the final pair may emit only its cos half).
    while (o < out.len) {
        const pair = fastGaussianPairs(1, seed, .{j});
        out[o] = pair.even[0] * scale;
        o += 1;
        j += 1;
        if (o < out.len) {
            out[o] = pair.odd[0] * scale;
            o += 1;
            j += 1;
        }
    }
}

/// f32 lanes of the fast gaussian body: 8 = two NEON registers / one AVX2
/// register (the repo's hand-vectorization width, see optim.zig).
const fast_gaussian_width = 8;

/// The Box-Muller pair (z_even, z_odd) for `count` pairs whose EVEN counter
/// indices are `pair_index` (each must be even). Pure per-lane function —
/// the chunking-invariance and lane-exactness contract of
/// `gaussianFillAtFast` rests on every lane running this exact op sequence.
fn fastGaussianPairs(comptime count: usize, seed: u64, pair_index: @Vector(count, u64)) struct {
    even: @Vector(count, f32),
    odd: @Vector(count, f32),
} {
    const Vu = @Vector(count, u64);
    const Vf = @Vector(count, f32);
    const Vd = @Vector(count, f64);

    const a = splitmixAtVec(count, seed, pair_index);
    const b = splitmixAtVec(count, seed, pair_index + @as(Vu, @splat(1)));
    // The scalar kernel's exact uniforms, then one f64 -> f32 rounding
    // (relative error <= 2^-24 for every magnitude, so the far tail keeps
    // its full range).
    const ua_wide = (@as(Vd, @floatFromInt(shrVec(count, a, 11))) + @as(Vd, @splat(1))) * @as(Vd, @splat(0x1.0p-53)); // (0, 1]
    const ub_wide = @as(Vd, @floatFromInt(shrVec(count, b, 11))) * @as(Vd, @splat(0x1.0p-53)); // [0, 1)
    const ua: Vf = @floatCast(ua_wide);
    const ub: Vf = @floatCast(ub_wide);

    const radius = @sqrt(@as(Vf, @splat(-2.0)) * lnPoly(count, ua));
    const turn = sinCosTurn(count, ub);
    return .{ .even = radius * turn.cos, .odd = radius * turn.sin };
}

/// `at(seed, i)` over vector lanes (same mixing constants as `at`).
fn splitmixAtVec(comptime count: usize, seed: u64, i: @Vector(count, u64)) @Vector(count, u64) {
    const Vu = @Vector(count, u64);
    var z = @as(Vu, @splat(seed)) +% (i +% @as(Vu, @splat(1))) *% @as(Vu, @splat(gamma));
    z = (z ^ shrVec(count, z, 30)) *% @as(Vu, @splat(0xBF58476D1CE4E5B9));
    z = (z ^ shrVec(count, z, 27)) *% @as(Vu, @splat(0x94D049BB133111EB));
    return z ^ shrVec(count, z, 31);
}

fn shrVec(comptime count: usize, v: @Vector(count, u64), comptime n: u6) @Vector(count, u64) {
    return v >> @as(@Vector(count, u6), @splat(n));
}

/// Natural log of a normal positive f32 (here always in (0, 1]) via the
/// Cephes logf polynomial (~1 ulp): frexp-style mantissa/exponent split,
/// degree-8 minimax on [sqrt(0.5)-1, sqrt(2)-1], split-ln2 exponent
/// recombination. Op order fixed — part of the fast-noise contract.
fn lnPoly(comptime count: usize, x: @Vector(count, f32)) @Vector(count, f32) {
    const Vf = @Vector(count, f32);
    const Vu32 = @Vector(count, u32);
    const bits: Vu32 = @bitCast(x);
    var e: Vf = @floatFromInt(@as(@Vector(count, i32), @intCast(bits >> @as(@Vector(count, u5), @splat(23)))) - @as(@Vector(count, i32), @splat(126)));
    var m: Vf = @bitCast((bits & @as(Vu32, @splat(0x007fffff))) | @as(Vu32, @splat(0x3f000000))); // [0.5, 1)
    const low = m < @as(Vf, @splat(0.7071067811865476));
    e = @select(f32, low, e - @as(Vf, @splat(1)), e);
    m = @select(f32, low, m + m, m);
    const f = m - @as(Vf, @splat(1));
    const z = f * f;
    var p: Vf = @splat(7.0376836292e-2);
    p = p * f + @as(Vf, @splat(-1.1514610310e-1));
    p = p * f + @as(Vf, @splat(1.1676998740e-1));
    p = p * f + @as(Vf, @splat(-1.2420140846e-1));
    p = p * f + @as(Vf, @splat(1.4249322787e-1));
    p = p * f + @as(Vf, @splat(-1.6668057665e-1));
    p = p * f + @as(Vf, @splat(2.0000714765e-1));
    p = p * f + @as(Vf, @splat(-2.4999993993e-1));
    p = p * f + @as(Vf, @splat(3.3333331174e-1));
    var y = f * z * p;
    y += e * @as(Vf, @splat(-2.12194440e-4)); // ln2 low
    y -= @as(Vf, @splat(0.5)) * z;
    y = f + y;
    return y + e * @as(Vf, @splat(0.693359375)); // ln2 high
}

/// sin(2*pi*u) and cos(2*pi*u) for u in [0, 1) via quadrant reduction on the
/// TURN fraction (no big-argument reduction needed) + the Cephes sinf/cosf
/// polynomials on [-pi/4, pi/4] (~1 ulp). Quadrant j = floor(4u + 0.5)
/// (round-half-up — part of the contract).
fn sinCosTurn(comptime count: usize, u: @Vector(count, f32)) struct {
    sin: @Vector(count, f32),
    cos: @Vector(count, f32),
} {
    const Vf = @Vector(count, f32);
    const Vu32 = @Vector(count, u32);
    const t4 = u * @as(Vf, @splat(4));
    const jf = @floor(t4 + @as(Vf, @splat(0.5)));
    const y = t4 - jf; // [-0.5, 0.5] quarter-turns
    const arg = y * @as(Vf, @splat(1.5707963267948966)); // [-pi/4, pi/4]
    const z = arg * arg;

    var sp: Vf = @splat(-1.9515295891e-4);
    sp = sp * z + @as(Vf, @splat(8.3321608736e-3));
    sp = sp * z + @as(Vf, @splat(-1.6666654611e-1));
    const sin_arg = arg + arg * z * sp;

    var cp: Vf = @splat(2.443315711809948e-5);
    cp = cp * z + @as(Vf, @splat(-1.388731625493765e-3));
    cp = cp * z + @as(Vf, @splat(4.166664568298827e-2));
    const cos_arg = @as(Vf, @splat(1)) - @as(Vf, @splat(0.5)) * z + z * z * cp;

    const q = @as(Vu32, @intFromFloat(jf)) & @as(Vu32, @splat(3));
    const swap = (q & @as(Vu32, @splat(1))) == @as(Vu32, @splat(1));
    const sin_base = @select(f32, swap, cos_arg, sin_arg);
    const cos_base = @select(f32, swap, sin_arg, cos_arg);
    const sin_neg = (q & @as(Vu32, @splat(2))) == @as(Vu32, @splat(2));
    const cos_neg = ((q +% @as(Vu32, @splat(1))) & @as(Vu32, @splat(2))) == @as(Vu32, @splat(2));
    return .{
        .sin = @select(f32, sin_neg, -sin_base, sin_base),
        .cos = @select(f32, cos_neg, -cos_base, cos_base),
    };
}

/// Uniform fill over [lo, hi): one stream output per value, mapped to [0, 1)
/// via `u = (x >> 11) * 2^-53` (the 53-bit mantissa mapping gaussianFill
/// uses), then `out[i] = lo + (hi - lo) * u`. The f64 -> f32 cast rounds to
/// nearest, so a `u` within ~2^-25 of 1 can round up to exactly `hi`; such
/// values are clamped to the largest f32 below `hi` to keep the documented
/// half-open bound exact.
pub fn uniformFill(seed: u64, out: []f32, lo: f32, hi: f32) void {
    var state = seed;
    const lo64: f64 = lo;
    const span: f64 = @as(f64, hi) - lo64;
    for (out) |*value| {
        const u = @as(f64, @floatFromInt(splitmix64(&state) >> 11)) * 0x1.0p-53; // [0, 1)
        var v: f32 = @floatCast(lo64 + span * u);
        if (v >= hi) v = std.math.nextAfter(f32, hi, lo);
        value.* = v;
    }
}

/// Standard Gumbel(0, 1) fill via inverse-CDF over the splitmix64 stream:
/// one stream output per value, mapped to the OPEN interval (0, 1) as
/// `u = ((x >> 11) + 0.5) * 2^-53` (min 2^-54, max 1 - 2^-54, so both log
/// arguments stay finite), then `out[i] = -ln(-ln(u))` computed in f64 and
/// rounded once to f32. Same checkpoint contract as `uniformFill`: the
/// (seed -> values) mapping is stable across releases.
pub fn gumbelFill(seed: u64, out: []f32) void {
    var state = seed;
    for (out) |*value| {
        const u = (@as(f64, @floatFromInt(splitmix64(&state) >> 11)) + 0.5) * 0x1.0p-53; // (0, 1)
        value.* = @floatCast(-@log(-@log(u)));
    }
}

/// Uniform i64 fill over [low, high): one stream output per value, mapped by
/// the widening multiply-shift `low + ((x · span) >> 64)` (Lemire's
/// multiply-shift; bias below span · 2^-64, branch-free). The span is
/// computed in two's-complement u64 arithmetic, so the full i64 range works.
/// Same checkpoint contract as `uniformFill`. Asserts `low < high`.
pub fn randintFill(seed: u64, out: []i64, low: i64, high: i64) void {
    std.debug.assert(low < high);
    const span: u64 = @as(u64, @bitCast(high)) -% @as(u64, @bitCast(low));
    var state = seed;
    for (out) |*value| {
        const offset: u64 = @truncate((@as(u128, splitmix64(&state)) * @as(u128, span)) >> 64);
        value.* = @bitCast(@as(u64, @bitCast(low)) +% offset);
    }
}

/// Fisher–Yates permutation of {0, …, out.len-1}: step k (k = n-1 … 1) swaps
/// position k with position `(at(seed, n-1-k) · (k+1)) >> 64` (the same
/// multiply-shift index map as `randintFill`). One counter-based stream
/// output per step; the (seed -> permutation) mapping shares the
/// `uniformFill` checkpoint contract.
pub fn randpermFill(seed: u64, out: []i64) void {
    for (out, 0..) |*value, i| value.* = @intCast(i);
    if (out.len < 2) return;
    var k: usize = out.len - 1;
    var counter: u64 = 0;
    while (k >= 1) : (k -= 1) {
        const j: usize = @intCast(@as(u64, @truncate((@as(u128, at(seed, counter)) * (@as(u128, k) + 1)) >> 64)));
        counter += 1;
        std.mem.swap(i64, &out[k], &out[j]);
    }
}

/// PyTorch `nn.init.kaiming_uniform_` with `a = sqrt(5)` — the default
/// `nn.Linear` / LoRA-A weight init: gain = sqrt(2 / (1 + a^2)) = sqrt(1/3),
/// bound = gain * sqrt(3 / fan_in) = sqrt(6 / ((1 + 5) * fan_in))
/// = sqrt(1 / fan_in), uniform over [-bound, bound).
pub fn kaimingUniformFill(seed: u64, out: []f32, fan_in: usize) void {
    const bound: f32 = @floatCast(@sqrt(1.0 / @as(f64, @floatFromInt(fan_in))));
    uniformFill(seed, out, -bound, bound);
}

/// Gaussian fill with explicit moments: `out[i] = mean + std_dev * z_i` where
/// `z_i` are the standard-normal draws of `gaussianFill` (splitmix64 +
/// Box-Muller, formula documented there).
pub fn normalFill(seed: u64, out: []f32, mean: f32, std_dev: f32) void {
    gaussianFill(seed, out, std_dev);
    if (mean != 0) {
        for (out) |*value| value.* += mean;
    }
}

test {
    _ = @import("rng_tests.zig");
}
