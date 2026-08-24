//! Element kernels of the ES module: chunk-parallel perturb/update/anchor
//! over one slot (GPU arms when resident), the serial replica writer, and
//! the f64 reward statistics. Every parallel loop is an element-independent
//! map — bitwise identical to the serial path for any thread count.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const exec_mod = @import("../exec.zig");
const parallel = @import("../parallel.zig");
const rng = @import("../rng.zig");
const common = @import("common.zig");
const slots_mod = @import("slots.zig");

const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;

const AnchorDecay = common.AnchorDecay;
const Stats = common.Stats;
const perturb_min_len = common.perturb_min_len;
const update_min_len = common.update_min_len;
const chunk_len = common.chunk_len;

const SlotCache = slots_mod.SlotCache;
const UpdateCacheView = slots_mod.UpdateCacheView;

/// Pre-normalization reward statistics: mean/std in f64 (ddof = 0), the
/// reference's float64 numpy stats — with plain SEQUENTIAL summation, which
/// may differ from numpy's pairwise summation in the last f64 ulp at larger
/// populations (invisible after the f32 coefficient cast; the golden
/// generator and parity checker compute their stats sequentially to match).
pub fn rewardStats(rewards: []const f32) Stats {
    var mean: f64 = 0;
    var min_reward: f32 = rewards[0];
    var max_reward: f32 = rewards[0];
    for (rewards) |r| {
        mean += @as(f64, r);
        min_reward = @min(min_reward, r);
        max_reward = @max(max_reward, r);
    }
    mean /= @floatFromInt(rewards.len);
    var variance: f64 = 0;
    for (rewards) |r| {
        const d = @as(f64, r) - mean;
        variance += d * d;
    }
    variance /= @floatFromInt(rewards.len);
    return .{
        .mean_reward = mean,
        .std_reward = @sqrt(variance),
        .min_reward = min_reward,
        .max_reward = max_reward,
    };
}

/// The GPU provider's ES dtype tag for a comptime DType (f16/f32 only —
/// bf16 has no device arm and stays on the CPU kernels).
fn esGpuDType(comptime dt: DType) ?backend_mod.gpu_impl.FlatDType {
    return switch (dt) {
        .f16 => .f16,
        .f32 => .f32,
        else => null,
    };
}

/// data[j] += scaled * eps[j] over [0, data.len), chunk-parallel. `scaled`
/// already carries sign * sigma; per element the delta `scaled * eps[j]`
/// rounds once in f32, narrows to the parameter dtype (the reference's
/// `p.data.add_(sign*scale*noise)` semantics — the scalar multiply rounds to
/// p.dtype before the in-place add), and applies widen-add-narrow. The
/// +sigma and -sigma passes use the bitwise-same delta, so they cancel to
/// the usual `(x + t) - t` drift.
pub fn perturbSlot(
    comptime dt: DType,
    ctx: *ExecContext,
    data: []dtype_mod.Scalar(dt),
    stream_seed: u64,
    scaled: f32,
    cache: ?SlotCache,
) void {
    // Device arm: resident parameters perturb ON the GPU (bitwise-identical
    // kernels, no managed-page migration back to the CPU). The stream cache
    // is a CPU-path feature — when it is active the CPU path runs so the
    // cache fills truthfully.
    if (comptime backend_mod.gpu_impl.enabled) {
        if (comptime esGpuDType(dt)) |gpu_dt| {
            if (cache == null and backend_mod.gpu_impl.flatPerturb(gpu_dt, std.mem.sliceAsBytes(data), stream_seed, scaled, data.len)) {
                return;
            }
        }
    }
    const Context = struct {
        data: []dtype_mod.Scalar(dt),
        stream_seed: u64,
        scaled: f32,
        cache: ?SlotCache,
    };
    ctx.parallelMap(data.len, perturb_min_len, Context{
        .data = data,
        .stream_seed = stream_seed,
        .scaled = scaled,
        .cache = cache,
    }, struct {
        fn runRange(c: Context, start: usize, end: usize) void {
            var scratch: [chunk_len]f32 = undefined;
            var j = start;
            while (j < end) {
                const len = @min(chunk_len, end - j);
                // The noise source: replay the cache, regenerate INTO the
                // cache (first use this iteration; chunk ranges are
                // disjoint), or regenerate into the stack scratch. Values
                // are identical on every path — caching is bitwise-neutral.
                const noise: []const f32 = if (c.cache) |slot_cache| blk: {
                    const region = slot_cache.data[j..][0..len];
                    if (!slot_cache.filled) rng.gaussianFillAtFast(c.stream_seed, j, region, 1.0);
                    break :blk region;
                } else blk: {
                    rng.gaussianFillAtFast(c.stream_seed, j, scratch[0..len], 1.0);
                    break :blk scratch[0..len];
                };
                for (c.data[j..][0..len], noise) |*p, eps| {
                    const t = dtype_mod.fromF32(dt, c.scaled * eps);
                    p.* = dtype_mod.fromF32(dt, dtype_mod.toF32(dt, p.*) + dtype_mod.toF32(dt, t));
                }
                j += len;
            }
        }
    }.runRange);
}

/// dst[j] = src[j] + sigma * eps[j] — the replica writer (serial; see
/// `materializeMember`). Same per-element math as `perturbSlot`.
pub fn materializeSlot(
    comptime dt: DType,
    src: []const dtype_mod.Scalar(dt),
    dst: []dtype_mod.Scalar(dt),
    stream_seed: u64,
    sigma: f32,
) void {
    var noise: [chunk_len]f32 = undefined;
    var j: usize = 0;
    while (j < src.len) {
        const len = @min(chunk_len, src.len - j);
        rng.gaussianFillAtFast(stream_seed, j, noise[0..len], 1.0);
        for (dst[j..][0..len], src[j..][0..len], noise[0..len]) |*d, s, eps| {
            const t = dtype_mod.fromF32(dt, sigma * eps);
            d.* = dtype_mod.fromF32(dt, dtype_mod.toF32(dt, s) + dtype_mod.toF32(dt, t));
        }
        j += len;
    }
}

/// data[j] += scale * sum_n coeffs[n] * eps_n[j], chunk-parallel with fp32
/// accumulation and the reference's rounding placement: `coeffs[n] * eps`
/// rounds, the accumulation adds round, the final `scale *` rounds once, the
/// delta narrows to the parameter dtype, and the add widens-adds-narrows.
/// Member order inside each element's accumulation is fixed (0..N), so the
/// result is bitwise independent of the chunking.
pub fn updateSlot(
    comptime dt: DType,
    ctx: *ExecContext,
    data: []dtype_mod.Scalar(dt),
    stream_seeds: []const u64,
    coeffs: []const f32,
    scale: f32,
    cache: ?UpdateCacheView,
) void {
    if (comptime backend_mod.gpu_impl.enabled) {
        if (comptime esGpuDType(dt)) |gpu_dt| {
            if (cache == null and backend_mod.gpu_impl.flatWeightedUpdate(gpu_dt, std.mem.sliceAsBytes(data), stream_seeds, coeffs, scale, data.len)) {
                return;
            }
        }
    }
    const Context = struct {
        data: []dtype_mod.Scalar(dt),
        stream_seeds: []const u64,
        coeffs: []const f32,
        scale: f32,
        cache: ?UpdateCacheView,
    };
    ctx.parallelMap(data.len, update_min_len, Context{
        .data = data,
        .stream_seeds = stream_seeds,
        .coeffs = coeffs,
        .scale = scale,
        .cache = cache,
    }, struct {
        fn runRange(c: Context, start: usize, end: usize) void {
            var scratch: [chunk_len]f32 = undefined;
            var acc: [chunk_len]f32 = undefined;
            var j = start;
            while (j < end) {
                const len = @min(chunk_len, end - j);
                @memset(acc[0..len], 0);
                for (c.stream_seeds, c.coeffs, 0..) |stream_seed, coeff, stream| {
                    const noise: []const f32 = if (c.cache) |cache_view| blk: {
                        const view = cache_view.regions[stream];
                        const region = view.slotRegion(cache_view.cache, cache_view.slot_index)[j..][0..len];
                        if (!view.filled) rng.gaussianFillAtFast(stream_seed, j, region, 1.0);
                        break :blk region;
                    } else blk: {
                        rng.gaussianFillAtFast(stream_seed, j, scratch[0..len], 1.0);
                        break :blk scratch[0..len];
                    };
                    for (acc[0..len], noise) |*a, eps| {
                        a.* += coeff * eps;
                    }
                }
                for (c.data[j..][0..len], acc[0..len]) |*p, a| {
                    const delta = dtype_mod.fromF32(dt, c.scale * a);
                    p.* = dtype_mod.fromF32(dt, dtype_mod.toF32(dt, p.*) + dtype_mod.toF32(dt, delta));
                }
                j += len;
            }
        }
    }.runRange);
}

/// The AWD proximal step on one slot, with the reference's exact in-place
/// op chain (each step computes in f32 and rounds to the parameter dtype,
/// mirroring torch's per-op `sub_`/`mul_`/`add_` semantics so parity is
/// bitwise for every dtype): d = round(theta - theta_0); l2 rounds
/// d * (1 - decay_step), l1 rounds |d| - decay_step, clamps at 0 and
/// restores the sign (torch sign(0) = 0 collapses to the same result);
/// finally theta = round(d' + theta_0). Chunk-parallel and
/// element-independent like the other kernels.
pub fn anchorSlot(
    comptime dt: DType,
    ctx: *ExecContext,
    data: []dtype_mod.Scalar(dt),
    anchor: []const dtype_mod.Scalar(dt),
    decay_step: f32,
    decay: AnchorDecay,
) void {
    if (comptime backend_mod.gpu_impl.enabled) {
        if (comptime esGpuDType(dt)) |gpu_dt| {
            if (backend_mod.gpu_impl.flatAnchorDecay(gpu_dt, std.mem.sliceAsBytes(data), std.mem.sliceAsBytes(anchor), decay_step, decay == .l1, data.len)) {
                return;
            }
        }
    }
    const Context = struct {
        data: []dtype_mod.Scalar(dt),
        anchor: []const dtype_mod.Scalar(dt),
        decay_step: f32,
        decay: AnchorDecay,
    };
    ctx.parallelMap(data.len, perturb_min_len, Context{
        .data = data,
        .anchor = anchor,
        .decay_step = decay_step,
        .decay = decay,
    }, struct {
        fn runRange(c: Context, start: usize, end: usize) void {
            switch (c.decay) {
                .l2 => {
                    const keep = 1 - c.decay_step;
                    for (c.data[start..end], c.anchor[start..end]) |*p, a| {
                        const anchor_wide = dtype_mod.toF32(dt, a);
                        const d = dtype_mod.fromF32(dt, dtype_mod.toF32(dt, p.*) - anchor_wide);
                        const kept = dtype_mod.fromF32(dt, dtype_mod.toF32(dt, d) * keep);
                        p.* = dtype_mod.fromF32(dt, dtype_mod.toF32(dt, kept) + anchor_wide);
                    }
                },
                .l1 => {
                    for (c.data[start..end], c.anchor[start..end]) |*p, a| {
                        const anchor_wide = dtype_mod.toF32(dt, a);
                        const d = dtype_mod.fromF32(dt, dtype_mod.toF32(dt, p.*) - anchor_wide);
                        const d_wide = dtype_mod.toF32(dt, d);
                        const thresholded = dtype_mod.fromF32(dt, @abs(d_wide) - c.decay_step);
                        const clamped = @max(dtype_mod.toF32(dt, thresholded), 0);
                        const shrunk: f32 = if (d_wide < 0) -clamped else clamped;
                        p.* = dtype_mod.fromF32(dt, shrunk + anchor_wide);
                    }
                },
                .none => unreachable,
            }
        }
    }.runRange);
}
