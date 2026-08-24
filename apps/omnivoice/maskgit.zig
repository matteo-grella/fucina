//! MaskGIT per-step math for OmniVoice TTS.
//!
//! Port of the pure algorithmic core of refs/omnivoice.cpp/src/maskgit-tts.h:
//! the cosine timestep schedule, per-step demask counts, log-softmax, CFG
//! combine, top-k keep filter, Gumbel noise, argmax/confidence scans, and
//! top-k slot selection. The decode-loop orchestration (maskgit_generate)
//! lands with the pipeline stage.
//!
//! Numeric contract: every float op preserves the reference's width and
//! order — f32 scalar sequential loops (the reference documents that its f32
//! accumulation deliberately matches PyTorch CUDA log_softmax; f64 or SIMD
//! reordering flips near-tie argmaxes), f64 only where the C code casts to
//! double (schedule frac/ceil, top-k count), C-cast truncation for
//! float→int. Zig never contracts mul+add into fma, matching the IEEE
//! source semantics of the reference (and PyTorch's unfused kernels); a
//! clang -O2 default build of the reference fuses a few expressions and may
//! wiggle 1-2 ulp there (see maskgit_tests.zig).

const std = @import("std");
const philox = @import("philox.zig");

/// MaskgitConfig with the reference defaults (maskgit-tts.h:23-31).
pub const Config = struct {
    num_step: usize = 32,
    guidance_scale: f32 = 2.0,
    t_shift: f32 = 0.1,
    layer_penalty_factor: f32 = 5.0,
    position_temperature: f32 = 5.0,
    class_temperature: f32 = 0.0,
    /// Only consulted when temperatures > 0.
    seed: u64 = 42,
};

/// Cosine timesteps (maskgit_timesteps): ts[i] = t_shift*t / (1 + (t_shift-1)*t)
/// with t = i/num_step, all f32. `out.len` must be num_step + 1.
pub fn timesteps(t_shift: f32, num_step: usize, out: []f32) void {
    std.debug.assert(num_step >= 1 and out.len == num_step + 1);
    for (out, 0..) |*ts, i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(num_step));
        ts.* = t_shift * t / (1.0 + (t_shift - 1.0) * t);
    }
}

/// Per-step demask schedule (maskgit_schedule): out[step] counts how many
/// slots to fill; frac/ceil in f64 (the f32 timestep difference widened),
/// C-cast truncation to int, min(target, rem); the last step takes the
/// remainder. `ts.len` must be out.len + 1; sum(out) == total.
pub fn schedule(total: usize, ts: []const f32, out: []i32) void {
    std.debug.assert(out.len >= 1 and ts.len == out.len + 1);
    const num_step = out.len;
    var rem: i32 = @intCast(total);
    for (out, 0..) |*sched, step| {
        var num: i32 = undefined;
        if (step == num_step - 1) {
            num = rem;
        } else {
            const frac: f64 = ts[step + 1] - ts[step]; // f32 subtract, then widen
            const target: i32 = @intFromFloat(@ceil(@as(f64, @floatFromInt(total)) * frac));
            num = @min(target, rem);
        }
        sched.* = num;
        rem -= num;
    }
}

/// In-place log_softmax (maskgit_log_softmax_inplace): single f32 max scan
/// (strict >), sequential f32 exp-sum in ascending index order, lse = m +
/// log(sum), x[v] -= lse. Deliberately scalar sequential — the reference
/// documents this as a PyTorch-parity requirement (f64 accumulation flips
/// near-tie argmaxes). Do not SIMD/reorder.
pub fn logSoftmaxInplace(x: []f32) void {
    std.debug.assert(x.len >= 1);
    var m: f32 = x[0];
    for (x[1..]) |v| {
        if (v > m) m = v;
    }
    var sum: f32 = 0.0;
    for (x) |v| {
        sum += @exp(v - m);
    }
    const lse = m + @log(sum);
    for (x) |*v| {
        v.* -= lse;
    }
}

/// Top-k keep filter (maskgit_top_k_filter_inplace): kk = (int)ceil((f64)ratio
/// * n); threshold = the kk-th largest value (duplicates counted, as
/// nth_element on the sorted order); every x[v] < threshold becomes -inf, so
/// values tied with the threshold all survive. No-op when kk <= 0 or kk >= n.
pub fn topKFilterInplace(x: []f32, ratio: f32) void {
    const kk: i64 = @intFromFloat(@ceil(@as(f64, ratio) * @as(f64, @floatFromInt(x.len))));
    if (kk <= 0 or kk >= @as(i64, @intCast(x.len))) return;
    const threshold = kthLargest(x, @intCast(kk));
    const neg_inf = -std.math.inf(f32);
    for (x) |*v| {
        if (v.* < threshold) v.* = neg_inf;
    }
}

/// The k-th largest value of x, duplicates counted (== sorted-descending
/// x[k-1], the value the C++ reads via nth_element on an index array).
/// Allocation-free: repeatedly finds the next distinct maximum and its
/// multiplicity, descending, until k values are consumed. O(x.len * distinct
/// values above the answer) — bounded by O(x.len * k).
fn kthLargest(x: []const f32, k: usize) f32 {
    std.debug.assert(k >= 1 and k <= x.len);
    var remaining = k;
    var have_bound = false;
    var bound: f32 = undefined; // values >= bound were consumed in earlier rounds
    while (true) {
        var have_cur = false;
        var cur: f32 = undefined;
        var count: usize = 0;
        for (x) |v| {
            if (have_bound and !(v < bound)) continue;
            if (!have_cur or v > cur) {
                cur = v;
                have_cur = true;
                count = 1;
            } else if (v == cur) {
                count += 1;
            }
        }
        std.debug.assert(have_cur);
        if (count >= remaining) return cur;
        remaining -= count;
        bound = cur;
        have_bound = true;
    }
}

/// Gumbel augmented sampling (maskgit_gumbel_inplace): inv_t = 1/temperature
/// precomputed; n uniforms drawn via philox.uniformFill (subsequence = element
/// index, fixed ctr_lo); per element g = -log(-log(u + 1e-10) + 1e-10) in f32
/// and x[i] = x[i]*inv_t + g (unfused; -inf stays -inf through the formula).
/// Then ctr_lo += 1: PyTorch advances the Philox offset by ceil(numel/slab) =
/// 1 block per kernel at OmniVoice sizes (numel <= K*V ~ 8200 << ~1.15M slab).
/// The uniforms are drawn in fixed-size chunks (bit-identical to one flat
/// fill: each element only depends on its own subsequence index).
pub fn gumbelInplace(x: []f32, temperature: f32, seed: u64, ctr_lo: *u32) void {
    const inv_t: f32 = 1.0 / temperature;
    var u_buf: [256]f32 = undefined;
    var start: usize = 0;
    while (start < x.len) {
        const n = @min(u_buf.len, x.len - start);
        const u = u_buf[0..n];
        philox.uniformFill(seed, start, ctr_lo.*, u);
        for (u, x[start..][0..n]) |uv, *xv| {
            const g = -@log(-@log(uv + 1e-10) + 1e-10);
            xv.* = xv.* * inv_t + g;
        }
        start += n;
    }
    ctr_lo.* +%= 1;
}

/// CFG + log_softmax for one V-length row (maskgit-tts.h:242-254): if g != 0,
/// log-softmax `c` and `u` IN PLACE (mirroring the reference, which reuses the
/// softmaxed rows), lp[v] = c[v] + g*(c[v] - u[v]) (unfused), log-softmax lp;
/// else lp = log_softmax(copy of c), with `c`/`u` untouched. Setting
/// lp[mask_id] = -inf afterwards is the CALLER's job, mirroring the reference
/// loop structure (line 255).
pub fn cfgCombine(c: []f32, u: []f32, guidance_scale: f32, lp: []f32) void {
    std.debug.assert(c.len == lp.len and u.len == lp.len);
    if (guidance_scale != 0.0) {
        logSoftmaxInplace(c);
        logSoftmaxInplace(u);
        for (lp, c, u) |*out, cv, uv| {
            out.* = cv + guidance_scale * (cv - uv);
        }
        logSoftmaxInplace(lp);
    } else {
        @memcpy(lp, c);
        logSoftmaxInplace(lp);
    }
}

/// Argmax with the reference tie rule: strict > scan, first (lowest) index
/// wins ties (maskgit-tts.h:275-283).
pub fn argmaxStrict(x: []const f32) usize {
    std.debug.assert(x.len >= 1);
    var best = x[0];
    var best_i: usize = 0;
    for (x[1..], 1..) |v, i| {
        if (v > best) {
            best = v;
            best_i = i;
        }
    }
    return best_i;
}

/// Max value with the same strict-> first-wins scan (confidence, :285-291).
pub fn maxValue(x: []const f32) f32 {
    std.debug.assert(x.len >= 1);
    var m = x[0];
    for (x[1..]) |v| {
        if (v > m) m = v;
    }
    return m;
}

/// The k highest-confidence flat indices, written to out_indices[0..k] in
/// descending confidence order. The reference uses std::partial_sort with an
/// a>b comparator, whose tie order is unspecified; this port breaks ties by
/// ascending index (deterministic). Greedy parity never depends on tie order:
/// post-layer-penalty scores are distinct in practice, and with temperatures
/// > 0 Gumbel noise makes ties measure-zero.
pub fn topKSelect(confidence: []const f32, k: usize, out_indices: []usize) void {
    std.debug.assert(k <= out_indices.len and k <= confidence.len);
    if (k == 0) return;
    var count: usize = 0;
    for (confidence, 0..) |c, i| {
        if (count == k and !(c > confidence[out_indices[k - 1]])) continue;
        // Insertion position: after all kept entries >= c (stable for ties).
        var p = count;
        while (p > 0 and c > confidence[out_indices[p - 1]]) : (p -= 1) {}
        var q = if (count < k) count else k - 1;
        while (q > p) : (q -= 1) {
            out_indices[q] = out_indices[q - 1];
        }
        out_indices[p] = i;
        if (count < k) count += 1;
    }
}

test {
    _ = @import("maskgit_tests.zig");
}
