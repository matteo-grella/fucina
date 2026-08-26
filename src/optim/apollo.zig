//! APOLLO: the official apollo_torch optimizer (arXiv 2412.05270) —
//! random low-rank gradient projection, AdamW moments in the compressed
//! space, channel/tensor scaling, the Fira norm-growth limiter, a scaled
//! SGD update, and the reference's legacy-HF AdamW fallback path (eps
//! outside the bias correction, decay after the step). Frame magics
//! FZP3/FZP5 (state is always f32, never dtype-tagged; the fallback's slot
//! list rides inline in the same frame); projections are regenerated from
//! (seed, step), never stored.

const std = @import("std");
const common = @import("common.zig");
const frame = @import("frame.zig");
const optimizer = @import("optimizer.zig");
// The (seed -> values) mapping of rng.gaussianFill is part of the APOLLO
// checkpoint contract (projections are regenerated from seed, not stored).
const rng = @import("../rng.zig");
const gaussianFill = rng.gaussianFill;

const Allocator = std.mem.Allocator;
const RawTensor = common.RawTensor;
const ExecContext = common.ExecContext;
const Param = common.Param;
const parallelMap = common.parallelMap;
const sumSquares = common.sumSquares;
const Optimizer = optimizer.Optimizer;
const FallbackFrame = optimizer.FallbackFrame;
const Magics = frame.Magics;

// ---------------------------------------------------------------------------
// APOLLO — official apollo_torch semantics (arXiv 2412.05270).
// ---------------------------------------------------------------------------

pub const ApolloScaleType = enum { channel, tensor };

pub const ApolloConfig = struct {
    lr: f32 = 0.01,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    /// Reference-class default (legacy HF AdamW lineage); HF Trainer overrides
    /// it to 1e-8.
    eps: f32 = 1e-6,
    weight_decay: f32 = 0,
    rank: usize = 128,
    update_proj_gap: u64 = 200,
    /// `apollo_scale`; the update multiplies sqrt(scale).
    scale: f32 = 1.0,
    scale_type: ApolloScaleType = .channel,
    correct_bias: bool = true,
    /// Apply sqrt(scale) BEFORE the norm-growth limiter (reference knob for
    /// short-warmup runs); default is after.
    scale_front: bool = false,
    disable_norm_growth_limiter: bool = false,
    seed: u64 = 0,

    /// APOLLO-Mini: rank-1 random projection with tensor-wise scaling and the
    /// sqrt(128) heuristic gradient scale.
    pub fn mini() ApolloConfig {
        return .{ .rank = 1, .scale_type = .tensor, .scale = 128 };
    }
};

const ApolloKernel = struct {
    pub const Config = ApolloConfig;
    pub const magics: Magics = .{ .v3 = "FZP3", .v5 = "FZP5" };
    pub const pinned_config_fields = [_][]const u8{ "rank", "update_proj_gap", "scale", "scale_type", "correct_bias", "scale_front", "disable_norm_growth_limiter" };
    pub const record = [_][]const u8{ "step", "seed", "prev_norm", "m", "v" };

    /// 2D params get the APOLLO low-rank path; everything else (biases,
    /// norms) and explicitly routed params get the reference's plain-AdamW
    /// path (legacy HF order, NOT `optim.AdamW`), whose slot list shares
    /// this frame.
    pub const Fallback = Optimizer(HfAdamwKernel);
    pub const fallback_frame: FallbackFrame = .inline_slots;

    pub fn fallbackConfig(config: ApolloConfig) ApolloConfig {
        return config;
    }

    pub fn routesToFallback(param: *const Param) bool {
        return param.raw_rank != 2;
    }

    pub const State = struct {
        m: []f32,
        v: []f32,
        /// Per-channel (or single-element, for tensor scaling) factor scratch.
        scaling: []f32,
        /// f64 accumulators for the per-channel norm sums (2 per channel),
        /// preallocated so step() stays allocation-free.
        norms: []f64,
        proj: ?RawTensor = null,
        proj_chunk: u64 = std.math.maxInt(u64),
        seed: u64,
        step: u64 = 0,
        /// Norm-growth-limiter memory; negative means "not recorded yet".
        prev_norm: f32 = -1,

        pub fn deinit(self: *State, allocator: Allocator) void {
            allocator.free(self.m);
            allocator.free(self.v);
            allocator.free(self.scaling);
            allocator.free(self.norms);
            if (self.proj) |*proj| proj.deinit();
            self.* = undefined;
        }
    };

    pub fn initState(allocator: Allocator, config: ApolloConfig, param: *const Param, index: usize) !State {
        const compressed = compressedLen(param, config.rank);
        const m = try allocator.alloc(f32, compressed);
        errdefer allocator.free(m);
        const v = try allocator.alloc(f32, compressed);
        errdefer allocator.free(v);
        const channels: usize = switch (config.scale_type) {
            .channel => if (param.rows >= param.cols) param.rows else param.cols,
            .tensor => 1,
        };
        const scaling = try allocator.alloc(f32, channels);
        errdefer allocator.free(scaling);
        const norms = try allocator.alloc(f64, 2 * channels);
        errdefer allocator.free(norms);
        @memset(m, 0);
        @memset(v, 0);
        // Distinct per-param seed (base + 1-based rank-slot index). The
        // reference enumerates every param for its torch RNG; only "distinct
        // seed, i.i.d. N(0, 1/rank) entries" is semantically required, and the
        // torch RNG stream is not reproducible here anyway.
        const seed = config.seed +% (index + 1);
        return .{ .m = m, .v = v, .scaling = scaling, .norms = norms, .seed = seed };
    }

    fn compressedLen(param: *const Param, rank: usize) usize {
        // Tall/square (rows >= cols): R = G @ P^T is (rows, rank).
        // Wide (rows < cols): R = P^T @ G is (rank, cols).
        return if (param.rows >= param.cols) param.rows * rank else rank * param.cols;
    }

    /// The projection is a pure function of (seed, step/gap); force
    /// regeneration on the next step after a load.
    pub fn afterLoad(state: *State) void {
        state.proj_chunk = std.math.maxInt(u64);
    }

    pub fn update(ctx: *ExecContext, config: ApolloConfig, param: *Param, state: *State, grad: *const RawTensor) !void {
        const rows = param.rows;
        const cols = param.cols;
        const tall = rows >= cols;
        const rank = config.rank;

        // Projection regeneration uses the PRE-increment step counter:
        // chunks are [0,T), [T,2T), ... States are NOT reset on regeneration.
        const chunk = state.step / config.update_proj_gap;
        if (state.proj == null or state.proj_chunk != chunk) {
            try regenerateProjection(ctx, config, param, state, chunk, tall);
        }

        var r_t = if (tall)
            try ctx.matmul(.f32, .trans_b, grad, &state.proj.?) // (rows, rank)
        else
            try ctx.matmul(.f32, .trans_a, &state.proj.?, grad); // (rank, cols)
        defer r_t.deinit();
        const r_data = r_t.dataConst();

        state.step += 1;
        const t: f64 = @floatFromInt(state.step);

        parallelMap(ctx, r_data.len, MomentMapContext{
            .m = state.m,
            .v = state.v,
            .r = r_data,
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .one_minus_b1 = @floatCast(1.0 - @as(f64, config.beta1)),
            .one_minus_b2 = @floatCast(1.0 - @as(f64, config.beta2)),
        }, momentMapRange);

        var step_size: f32 = config.lr;
        if (config.correct_bias) {
            const bc1 = 1 - std.math.pow(f64, config.beta1, t);
            const bc2 = 1 - std.math.pow(f64, config.beta2, t);
            step_size = @floatCast(@as(f64, config.lr) * @sqrt(bc2) / bc1);
        }

        // Scaling factors from the UN-bias-corrected R~ = m/(sqrt(v)+eps),
        // norms taken along the rank axis; +1e-8 guards the division.
        const scaling = state.scaling;
        switch (config.scale_type) {
            .channel => {
                const channels = scaling.len;
                const sum_opt = state.norms[0..channels];
                const sum_raw = state.norms[channels..];
                @memset(sum_opt, 0);
                @memset(sum_raw, 0);
                if (tall) {
                    // R is (rows=channels, rank): channel = row index.
                    for (state.m, state.v, r_data, 0..) |mi, vi, ri, i| {
                        const channel = i / rank;
                        const opt = mi / (@sqrt(vi) + config.eps);
                        sum_opt[channel] += @as(f64, opt) * opt;
                        sum_raw[channel] += @as(f64, ri) * ri;
                    }
                } else {
                    // R is (rank, cols=channels): channel = column index.
                    for (state.m, state.v, r_data, 0..) |mi, vi, ri, i| {
                        const channel = i % cols;
                        const opt = mi / (@sqrt(vi) + config.eps);
                        sum_opt[channel] += @as(f64, opt) * opt;
                        sum_raw[channel] += @as(f64, ri) * ri;
                    }
                }
                for (scaling, sum_opt, sum_raw) |*s, so, sr| {
                    s.* = @floatCast(@sqrt(so) / (@sqrt(sr) + 1e-8));
                }
            },
            .tensor => {
                var sum_opt: f64 = 0;
                var sum_raw: f64 = 0;
                for (state.m, state.v, r_data) |mi, vi, ri| {
                    const opt = mi / (@sqrt(vi) + config.eps);
                    sum_opt += @as(f64, opt) * opt;
                    sum_raw += @as(f64, ri) * ri;
                }
                scaling[0] = @floatCast(@sqrt(sum_opt) / (@sqrt(sum_raw) + 1e-8));
            },
        }

        // U = G (elementwise) scaled per channel (rows for tall, cols for
        // wide); the optional front sqrt(scale) is fused (same per-element op
        // order as the reference's separate pass).
        var update_buf = try ctx.empty(.f32, .{ rows, cols });
        defer update_buf.deinit();
        const ud = update_buf.data();
        const sqrt_scale = @sqrt(config.scale);
        parallelMap(ctx, ud.len, BuildMapContext{
            .u = ud,
            .g = grad.dataConst(),
            .scaling = scaling,
            .cols = cols,
            .kind = switch (config.scale_type) {
                .channel => if (tall) BuildKind.channel_tall else BuildKind.channel_wide,
                .tensor => BuildKind.tensor,
            },
            .front_scale = if (config.scale_front and config.scale != 1) sqrt_scale else 1,
        }, buildMapRange);

        // Fira norm-growth limiter (gamma = 1.01): clip the per-step growth of
        // ||U||_F; the recorded norm is the POST-limit one. The norm reduction
        // is the deterministic chunked `sumSquares`; the division and the
        // trailing sqrt(scale) fuse into the apply pass below with the same
        // per-element op order as the reference's separate passes.
        var limiter: f32 = 1;
        if (!config.disable_norm_growth_limiter) {
            const cur: f32 = @floatCast(@sqrt(try sumSquares(ctx, ud)));
            if (state.prev_norm >= 0) {
                limiter = @max(cur / (state.prev_norm + 1e-8), 1.01) / 1.01;
                state.prev_norm = cur / limiter;
            } else {
                state.prev_norm = cur;
            }
        }

        // Scaled-SGD step, then decoupled decay AFTER the step with the raw lr
        // (reference order; differs from AdamW/Muon).
        parallelMap(ctx, param.len(), ApplyMapContext{
            .p = param.data(),
            .u = ud,
            .limiter = limiter,
            .back_scale = if (!config.scale_front and config.scale != 1) sqrt_scale else 1,
            .step_size = step_size,
            .decay_alpha = if (config.weight_decay > 0) @floatCast(-(@as(f64, config.lr) * @as(f64, config.weight_decay))) else 0,
        }, applyMapRange);
    }

    /// P entries are i.i.d. N(0, 1/rank): standard normal draws divided by
    /// sqrt(rank). Deterministic in (seed, chunk), so checkpoints don't need to
    /// store P. Tall params project the column space: P is (rank, cols); wide
    /// params project the row space: P is (rows, rank).
    fn regenerateProjection(ctx: *ExecContext, config: ApolloConfig, param: *const Param, state: *State, chunk: u64, tall: bool) !void {
        const shape: [2]usize = if (tall)
            .{ config.rank, param.cols }
        else
            .{ param.rows, config.rank };
        if (state.proj == null) {
            state.proj = try ctx.empty(.f32, shape);
        }
        const inv_sqrt_rank = 1.0 / @sqrt(@as(f32, @floatFromInt(config.rank)));
        gaussianFill(state.seed +% chunk *% 0x9E3779B97F4A7C15, state.proj.?.data(), inv_sqrt_rank);
        state.proj_chunk = chunk;
    }
};

pub const Apollo = Optimizer(ApolloKernel);

const MomentMapContext = struct {
    m: []f32,
    v: []f32,
    r: []const f32,
    beta1: f32,
    beta2: f32,
    one_minus_b1: f32,
    one_minus_b2: f32,
};

fn momentMapRange(c: MomentMapContext, start: usize, end: usize) void {
    for (c.m[start..end], c.v[start..end], c.r[start..end]) |*mi, *vi, ri| {
        mi.* = c.beta1 * mi.* + c.one_minus_b1 * ri;
        vi.* = c.beta2 * vi.* + c.one_minus_b2 * ri * ri;
    }
}

const BuildKind = enum { channel_tall, channel_wide, tensor };

const BuildMapContext = struct {
    u: []f32,
    g: []const f32,
    scaling: []const f32,
    cols: usize,
    kind: BuildKind,
    front_scale: f32,
};

fn buildMapRange(c: BuildMapContext, start: usize, end: usize) void {
    switch (c.kind) {
        .channel_tall => for (c.u[start..end], c.g[start..end], start..) |*ui, gi, i| {
            ui.* = gi * c.scaling[i / c.cols] * c.front_scale;
        },
        .channel_wide => for (c.u[start..end], c.g[start..end], start..) |*ui, gi, i| {
            ui.* = gi * c.scaling[i % c.cols] * c.front_scale;
        },
        .tensor => for (c.u[start..end], c.g[start..end]) |*ui, gi| {
            ui.* = gi * c.scaling[0] * c.front_scale;
        },
    }
}

const ApplyMapContext = struct {
    p: []f32,
    u: []const f32,
    limiter: f32,
    back_scale: f32,
    step_size: f32,
    decay_alpha: f32,
};

fn applyMapRange(c: ApplyMapContext, start: usize, end: usize) void {
    for (c.p[start..end], c.u[start..end]) |*pi, ui| {
        const adjusted = ui / c.limiter * c.back_scale;
        // Decay AFTER the step, in the reference's additive form
        // `p.add_(p, alpha=-lr*wd)`.
        const stepped = pi.* - c.step_size * adjusted;
        pi.* = stepped + c.decay_alpha * stepped;
    }
}

// ---------------------------------------------------------------------------
// The fallback path: the reference's legacy-HF AdamW.
// ---------------------------------------------------------------------------

/// The APOLLO reference's fallback AdamW is the legacy HF formulation, NOT
/// PyTorch AdamW: `denom = sqrt(v) + eps` (eps outside the bias correction),
/// the bias correction folded into a scalar `lr*sqrt(bc2)/bc1`, and decoupled
/// decay applied AFTER the step to the already-updated parameter. Its slots
/// only ever frame inline within Apollo's, so the kernel carries no magics.
const HfAdamwKernel = struct {
    pub const Config = ApolloConfig;
    pub const record = [_][]const u8{ "step", "m", "v" };

    pub const State = struct {
        m: []f32,
        v: []f32,
        step: u64 = 0,

        pub fn deinit(self: *State, allocator: Allocator) void {
            allocator.free(self.m);
            allocator.free(self.v);
            self.* = undefined;
        }
    };

    pub fn initState(allocator: Allocator, _: ApolloConfig, param: *const Param, _: usize) !State {
        const n = param.len();
        const m = try allocator.alloc(f32, n);
        errdefer allocator.free(m);
        const v = try allocator.alloc(f32, n);
        @memset(m, 0);
        @memset(v, 0);
        return .{ .m = m, .v = v };
    }

    pub fn update(ctx: *ExecContext, config: ApolloConfig, param: *Param, state: *State, grad: *const RawTensor) !void {
        state.step += 1;
        hfAdamwUpdate(ctx, config, param.data(), grad.dataConst(), state.m, state.v, state.step);
    }
};

const HfAdamwMapContext = struct {
    p: []f32,
    g: []const f32,
    m: []f32,
    v: []f32,
    beta1: f32,
    beta2: f32,
    one_minus_b1: f32,
    one_minus_b2: f32,
    step_size: f32,
    eps: f32,
    decay_alpha: f32,
};

fn hfAdamwMapRange(c: HfAdamwMapContext, start: usize, end: usize) void {
    for (c.p[start..end], c.g[start..end], c.m[start..end], c.v[start..end]) |*pi, gi, *mi, *vi| {
        mi.* = c.beta1 * mi.* + c.one_minus_b1 * gi;
        vi.* = c.beta2 * vi.* + c.one_minus_b2 * gi * gi;
        // Decay AFTER the step (legacy-HF order), fused per element in the
        // reference's additive form `p.add_(p, alpha=-lr*wd)`.
        const stepped = pi.* - c.step_size * (mi.* / (@sqrt(vi.*) + c.eps));
        pi.* = stepped + c.decay_alpha * stepped;
    }
}

fn hfAdamwUpdate(ctx: *ExecContext, config: ApolloConfig, p: []f32, g: []const f32, m: []f32, v: []f32, step_count: u64) void {
    const t: f64 = @floatFromInt(step_count);
    var step_size: f32 = config.lr;
    if (config.correct_bias) {
        const bc1 = 1 - std.math.pow(f64, config.beta1, t);
        const bc2 = 1 - std.math.pow(f64, config.beta2, t);
        step_size = @floatCast(@as(f64, config.lr) * @sqrt(bc2) / bc1);
    }
    parallelMap(ctx, p.len, HfAdamwMapContext{
        .p = p,
        .g = g,
        .m = m,
        .v = v,
        .beta1 = config.beta1,
        .beta2 = config.beta2,
        .one_minus_b1 = @floatCast(1.0 - @as(f64, config.beta1)),
        .one_minus_b2 = @floatCast(1.0 - @as(f64, config.beta2)),
        .step_size = step_size,
        .eps = config.eps,
        .decay_alpha = if (config.weight_decay > 0) @floatCast(-(@as(f64, config.lr) * @as(f64, config.weight_decay))) else 0,
    }, hfAdamwMapRange);
}
