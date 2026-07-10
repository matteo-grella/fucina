//! Philox4x32-10 counter-based PRNG + Box-Muller normals for OmniVoice.
//!
//! Port of refs/omnivoice.cpp/src/philox.h. Matches PyTorch CUDA
//! torch.randn()/torch.rand() output (cuRAND Philox4_32_10): per-element
//! subsequences, counter layout [offset_lo, offset_hi, subseq_lo, subseq_hi],
//! cuRAND uniform conversion, and the torch.bfloat16 round-trip. All float
//! math is f32 with the reference's exact operation order.

const std = @import("std");

const philox_m0: u32 = 0xD2511F53;
const philox_m1: u32 = 0xCD9E8D57;
const philox_w0: u32 = 0x9E3779B9;
const philox_w1: u32 = 0xBB67AE85;

/// cuRAND uniform conversion constants: 1/2^32 and 2*pi/2^32.
pub const curand_2pow32_inv: f32 = 2.3283064365386963e-10;
pub const curand_2pow32_inv_2pi: f32 = 1.4629180792671596e-09;

fn philoxRound(ctr: [4]u32, k0: u32, k1: u32) [4]u32 {
    const prod0 = @as(u64, philox_m0) * @as(u64, ctr[0]);
    const prod1 = @as(u64, philox_m1) * @as(u64, ctr[2]);
    const hi0: u32 = @truncate(prod0 >> 32);
    const lo0: u32 = @truncate(prod0);
    const hi1: u32 = @truncate(prod1 >> 32);
    const lo1: u32 = @truncate(prod1);
    return .{ hi1 ^ ctr[1] ^ k0, lo1, hi0 ^ ctr[3] ^ k1, lo0 };
}

/// Philox4x32-10 block function: 10 rounds, key bumped by (W0, W1) after each
/// of rounds 1..9.
pub fn philox4x32_10(ctr_in: [4]u32, key0: u32, key1: u32) [4]u32 {
    var ctr = ctr_in;
    var k0 = key0;
    var k1 = key1;
    var round: usize = 0;
    while (round < 10) : (round += 1) {
        ctr = philoxRound(ctr, k0, k1);
        if (round < 9) {
            k0 +%= philox_w0;
            k1 +%= philox_w1;
        }
    }
    return ctr;
}

/// cuRAND Box-Muller: two u32 draws -> two N(0,1) f32 samples.
pub fn boxMuller(bits0: u32, bits1: u32) [2]f32 {
    const u = @as(f32, @floatFromInt(bits0)) * curand_2pow32_inv + curand_2pow32_inv * 0.5;
    const v = @as(f32, @floatFromInt(bits1)) * curand_2pow32_inv_2pi + curand_2pow32_inv_2pi * 0.5;
    const s = @sqrt(-2.0 * @log(u));
    return .{ s * @sin(v), s * @cos(v) };
}

/// Four N(0,1) samples for (seed, subsequence, offset); counter layout
/// [offset_lo, offset_hi, subseq_lo, subseq_hi].
pub fn normal4(seed: u64, subsequence: u64, offset: u64) [4]f32 {
    const ctr = [4]u32{
        @truncate(offset),
        @truncate(offset >> 32),
        @truncate(subsequence),
        @truncate(subsequence >> 32),
    };
    const r = philox4x32_10(ctr, @truncate(seed), @truncate(seed >> 32));
    const n01 = boxMuller(r[0], r[1]);
    const n23 = boxMuller(r[2], r[3]);
    return .{ n01[0], n01[1], n23[0], n23[1] };
}

/// f32 -> bf16 -> f32 round-trip (round-to-nearest-even), matching
/// torch.bfloat16 precision.
pub fn f32ToBf16ToF32(x: f32) f32 {
    var bits: u32 = @bitCast(x);
    bits +%= 0x7FFF + ((bits >> 16) & 1);
    bits &= 0xFFFF_0000;
    return @bitCast(bits);
}

/// Single N(0,1) f32 sample: element `index` of torch.randn on CUDA before
/// dtype rounding (subsequence = index, offset = 0, Box-Muller output 0;
/// outputs 1..3 discarded, one thread per element).
pub fn randn(seed: u64, index: u64) f32 {
    return normal4(seed, index, 0)[0];
}

/// `randn` with the torch.bfloat16 round-trip applied.
pub fn randnBf16(seed: u64, index: u64) f32 {
    return f32ToBf16ToF32(randn(seed, index));
}

/// Fills `out` with N(0,1) matching torch.randn(generator=manual_seed(seed),
/// device="cuda"); `bf16_round` reproduces dtype=torch.bfloat16.
pub fn randnFill(seed: u64, out: []f32, bf16_round: bool) void {
    for (out, 0..) |*dst, k| {
        const vals = normal4(seed, k, 0);
        dst.* = if (bf16_round) f32ToBf16ToF32(vals[0]) else vals[0];
    }
}

/// Fills `out` with uniform (0, 1) matching PyTorch CUDA torch.rand kernels:
/// per element k, ctr = {ctr_lo, 0, lo32(subseq_start+k), hi32(subseq_start+k)},
/// out[k] = (f32(r.x) + 0.5) * 2^-32. `ctr_lo` is the cumulative Philox block
/// counter the caller advances between successive kernels (0 on the first call
/// after manual_seed).
pub fn uniformFill(seed: u64, subseq_start: u64, ctr_lo: u32, out: []f32) void {
    const slo: u32 = @truncate(seed);
    const shi: u32 = @truncate(seed >> 32);
    for (out, 0..) |*dst, k| {
        const s = subseq_start +% k;
        const ctr = [4]u32{ ctr_lo, 0, @truncate(s), @truncate(s >> 32) };
        const r = philox4x32_10(ctr, slo, shi);
        dst.* = (@as(f32, @floatFromInt(r[0])) + 0.5) * curand_2pow32_inv;
    }
}

test {
    _ = @import("philox_tests.zig");
}
