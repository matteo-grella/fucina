//! Qwen3-TTS codec-token sampler (qwentts.cpp sampling.h + philox.h):
//! suppress → repetition penalty (HF rule, unique history) → temperature →
//! top-k (nth-element threshold, ties kept) → top-p (full-vocab softmax,
//! descending sort, keep-first-crossing) → unnormalized-exp multinomial with
//! a Philox4x32-10 draw (cuRAND/PyTorch-CUDA convention: key = seed,
//! ctr = (0, 0, subseq_lo, subseq_hi), u = (r.x + 0.5) · 2⁻³²).
//! Greedy: temperature ≤ 0 → argmax over the suppressed logits, no penalty,
//! no draw. One Philox subsequence per primitive sample; frame t consumes
//! subsequences [16t, 16t+15] (codebook-0 first, then the 15 predictor draws).

const std = @import("std");

const philox_m0: u32 = 0xD2511F53;
const philox_m1: u32 = 0xCD9E8D57;
const philox_w0: u32 = 0x9E3779B9;
const philox_w1: u32 = 0xBB67AE85;
const two_pow32_inv: f32 = 2.3283064365386963e-10; // 1 / 2^32

const Ctr = struct { x: u32, y: u32, z: u32, w: u32 };

fn round(ctr: Ctr, k0: u32, k1: u32) Ctr {
    const p0 = @as(u64, philox_m0) * ctr.x;
    const p1 = @as(u64, philox_m1) * ctr.z;
    return .{
        .x = @as(u32, @truncate(p1 >> 32)) ^ ctr.y ^ k0,
        .y = @truncate(p1),
        .z = @as(u32, @truncate(p0 >> 32)) ^ ctr.w ^ k1,
        .w = @truncate(p0),
    };
}

/// One uniform (0,1) draw for (seed, subsequence) — Philox4x32-10.
pub fn philoxUniform(seed: i64, subseq: u64) f32 {
    var k0: u32 = @truncate(@as(u64, @bitCast(seed)));
    var k1: u32 = @truncate(@as(u64, @bitCast(seed)) >> 32);
    var ctr = Ctr{ .x = 0, .y = 0, .z = @truncate(subseq), .w = @truncate(subseq >> 32) };
    inline for (0..10) |i| {
        ctr = round(ctr, k0, k1);
        if (i < 9) {
            k0 +%= philox_w0;
            k1 +%= philox_w1;
        }
    }
    return (@as(f32, @floatFromInt(ctr.x)) + 0.5) * two_pow32_inv;
}

/// Mask logits in `[lo, hi)` to −inf, sparing `keep`.
pub fn applySuppress(logits: []f32, lo: usize, hi: usize, keep: usize) void {
    for (@min(lo, logits.len)..@min(hi, logits.len)) |i| {
        if (i != keep) logits[i] = -std.math.inf(f32);
    }
}

pub const Params = struct {
    temperature: f32 = 0.9,
    top_k: usize = 50,
    top_p: f32 = 1.0,
    repetition_penalty: f32 = 1.05,
};

pub const Sampled = struct { id: usize, u: f32 };

/// HF-order stochastic sampler over `logits` (mutated in place). `history`
/// applies the repetition penalty over its unique token ids (pass empty for
/// the predictor stacks). `scratch` must hold ≥ 2·V f32 (thread-reuse
/// buffer, caller-owned). Greedy (`temperature <= 0`) needs no scratch.
pub fn sample(
    logits: []f32,
    params: Params,
    history: []const i32,
    seed: i64,
    subseq: u64,
    scratch: []f32,
) Sampled {
    const v = logits.len;
    if (params.temperature <= 0) {
        var best: usize = 0;
        for (logits, 0..) |l, i| {
            if (l > logits[best]) best = i;
        }
        return .{ .id = best, .u = -1.0 };
    }

    // Repetition penalty, each unique token touched once.
    if (params.repetition_penalty != 1.0 and history.len > 0) {
        const seen = scratch[0..v];
        @memset(seen, 0);
        for (history) |tok| {
            if (tok < 0 or tok >= v) continue;
            const ti: usize = @intCast(tok);
            if (seen[ti] != 0) continue;
            seen[ti] = 1;
            const s = logits[ti];
            logits[ti] = if (s < 0) s * params.repetition_penalty else s / params.repetition_penalty;
        }
    }

    const inv_temp = 1.0 / params.temperature;
    for (logits) |*l| l.* *= inv_temp;

    if (params.top_k > 0 and params.top_k < v) {
        const tmp = scratch[0..v];
        @memcpy(tmp, logits);
        std.mem.sort(f32, tmp, {}, std.sort.desc(f32));
        const threshold = tmp[params.top_k - 1];
        for (logits) |*l| {
            if (l.* < threshold) l.* = -std.math.inf(f32);
        }
    }

    if (params.top_p > 0.0 and params.top_p < 1.0) {
        var max_logit = -std.math.inf(f32);
        for (logits) |l| max_logit = @max(max_logit, l);
        var sum_exp: f32 = 0;
        for (logits) |l| sum_exp += @exp(l - max_logit);
        const inv_sum = 1.0 / sum_exp;

        const cutoff = max_logit - 16.0;
        const Entry = struct { id: u32, prob: f32 };
        var entries: [4096]Entry = undefined;
        var k: usize = 0;
        for (logits, 0..) |*l, i| {
            if (l.* >= cutoff) {
                if (k < entries.len) {
                    entries[k] = .{ .id = @intCast(i), .prob = @exp(l.* - max_logit) * inv_sum };
                    k += 1;
                }
            } else l.* = -std.math.inf(f32);
        }
        if (k > 0) {
            std.mem.sort(Entry, entries[0..k], {}, struct {
                fn lt(_: void, a: Entry, b: Entry) bool {
                    return a.prob > b.prob;
                }
            }.lt);
            var cum: f32 = 0;
            for (entries[0..k], 0..) |e, i| {
                if (i > 0 and cum >= params.top_p) logits[e.id] = -std.math.inf(f32);
                cum += e.prob;
            }
        }
    }

    // Unnormalized-exp multinomial: u·sum inverse-CDF.
    var max_val = -std.math.inf(f32);
    for (logits) |l| max_val = @max(max_val, l);
    var sum: f32 = 0;
    for (logits) |*l| {
        l.* = @exp(l.* - max_val);
        sum += l.*;
    }
    const u = philoxUniform(seed, subseq);
    const r = u * sum;
    var acc: f32 = 0;
    for (logits, 0..) |l, i| {
        acc += l;
        if (acc >= r) return .{ .id = i, .u = u };
    }
    return .{ .id = v - 1, .u = u };
}

test "philox matches the oracle's seed-42 sample traces" {
    // [Sample]/[Sample-CP] u values logged by qwen-tts --seed 42
    // (goldens-qwen3tts/seed42-aiden.log): u depends only on (seed, subseq).
    const expected = [_]f32{
        0.6129598618, 0.0100588445, 0.3984136879, 0.0403083973,
        0.1562667489, 0.4824730754, 0.7362473011, 0.4059819579,
    };
    for (expected, 0..) |e, subseq| {
        try std.testing.expectApproxEqAbs(e, philoxUniform(42, subseq), 5e-10);
    }
}
