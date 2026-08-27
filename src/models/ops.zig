//! Model-band op library (`fucina_models`): kernels and helpers that exist
//! for exactly one model family group and therefore live above the core
//! runtime, reaching it through the public `ExecContext` surface. Family
//! subtrees keep their own heavy kernels (`gemma/moe_gu.zig`,
//! `research/kimi3/delta_attention.zig`); this file holds the small
//! cross-family leftovers.

const std = @import("std");
const fucina = @import("fucina");

const ExecContext = fucina.ExecContext;

/// DeepSeek-family YaRN inverse-frequency blend, in f64 (the HF
/// DeepseekV2Yarn reference): the plain pow schedule
/// `base^(-2i/dim)`, linearly blended toward `freq/factor` across the
/// correction ramp [beta_fast = 32, beta_slow = 1 rotations]. The cos/sin
/// magnitude correction (mscale) is the caller's business — the DeepSeek
/// ports fold it into the attention scale. `factor <= 1` or
/// `orig_ctx == 0` returns the unblended schedule, so one call covers both
/// rope families. Caller frees the returned slice.
pub fn yarnBlendInvFreqsF64(ctx: *ExecContext, dim: usize, base: f64, factor: f64, orig_ctx: usize) ![]f64 {
    const pairs = dim / 2;
    const inv_freq = try ctx.allocator().alloc(f64, pairs);
    errdefer ctx.allocator().free(inv_freq);
    for (inv_freq, 0..) |*f, i| {
        f.* = std.math.pow(f64, base, -(@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(dim))));
    }
    if (factor > 1.0 and orig_ctx > 0) {
        const orig: f64 = @floatFromInt(orig_ctx);
        const dimFor = struct {
            fn go(rotations: f64, d: f64, b: f64, o: f64) f64 {
                return d * @log(o / (rotations * 2.0 * std.math.pi)) / (2.0 * @log(b));
            }
        }.go;
        const d_f: f64 = @floatFromInt(dim);
        var low = @floor(dimFor(32.0, d_f, base, orig));
        var high = @ceil(dimFor(1.0, d_f, base, orig));
        low = @max(low, 0);
        high = @min(high, d_f - 1);
        for (inv_freq, 0..) |*f, i| {
            const extra = f.*;
            const inter = extra / factor;
            // ramp 0 -> 1 across [low, high]; mask = 1 - ramp keeps the
            // fast-rotating dims extrapolated (original freq).
            var ramp = (@as(f64, @floatFromInt(i)) - low) / @max(high - low, 0.001);
            ramp = @min(@max(ramp, 0.0), 1.0);
            const mask = 1.0 - ramp;
            f.* = inter * (1.0 - mask) + extra * mask;
        }
    }
    return inv_freq;
}
