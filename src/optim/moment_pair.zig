//! Adam and AdamW: PyTorch single-tensor semantics, one comptime kernel
//! (`MomentPairKernel`) serving both with separate reference-faithful
//! update kernels. Frame magics FZAD/FZD4/FZD5 (Adam) and FZA3/FZA4/FZA5
//! (AdamW).

const std = @import("std");
const common = @import("common.zig");
const frame = @import("frame.zig");
const optimizer = @import("optimizer.zig");

const Allocator = std.mem.Allocator;
const RawTensor = common.RawTensor;
const ExecContext = common.ExecContext;
const StateDType = common.StateDType;
const StateSlice = common.StateSlice;
const StateBuf = common.StateBuf;
const StateVec = common.StateVec;
const state_vec_len = common.state_vec_len;
const stateLoad = common.stateLoad;
const stateStore = common.stateStore;
const stateVecLoad = common.stateVecLoad;
const stateVecStore = common.stateVecStore;
const Param = common.Param;
const parallelMap = common.parallelMap;
const Optimizer = optimizer.Optimizer;
const Magics = frame.Magics;

// ---------------------------------------------------------------------------
// AdamW — PyTorch torch.optim.AdamW single-tensor semantics.
// ---------------------------------------------------------------------------

pub const AdamWConfig = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,
    /// Storage dtype of the FIRST moment (m); step math stays f32 either
    /// way. bf16 is safe for m: with beta1 = 0.9 the per-step relative
    /// change (~10%) is far above bf16's ~0.39% resolution.
    state_dtype: StateDType = .f32,
    /// Storage dtype of the SECOND moment (v) — a separate opt-in because v
    /// is precision-sensitive: with beta2 = 0.999 the per-step relative
    /// change (~0.1%) is BELOW bf16's ~2^-8 ≈ 0.39% resolution, so the EMA
    /// update can round to a no-op and stall (stale denominator →
    /// effective-LR drift). Keep .f32 unless memory forces the trade.
    second_moment_dtype: StateDType = .f32,
};

/// AdamW and Adam differ only in the per-element update kernel, the config
/// defaults, and the frame magic trio; one kernel serves both. The update
/// kernels stay separate reference-faithful ports, and the frame bytes are
/// pinned by the checkpoint tests.
fn MomentPairKernel(comptime ConfigT: type, comptime frame_magics: Magics, comptime updateFn: anytype) type {
    return struct {
        pub const Config = ConfigT;
        pub const magics = frame_magics;
        pub const pinned_config_fields = [_][]const u8{};
        pub const record = [_][]const u8{ "step", "m", "v" };

        pub const State = struct {
            m: StateBuf,
            v: StateBuf,
            step: u64 = 0,

            pub fn deinit(self: *State, allocator: Allocator) void {
                self.m.deinit(allocator);
                self.v.deinit(allocator);
                self.* = undefined;
            }
        };

        pub fn initState(allocator: Allocator, config: ConfigT, param: *const Param, _: usize) !State {
            const n = param.len();
            const m = try StateBuf.alloc(allocator, config.state_dtype, n);
            errdefer m.deinit(allocator);
            const v = try StateBuf.alloc(allocator, config.second_moment_dtype, n);
            return .{ .m = m, .v = v };
        }

        pub fn update(ctx: *ExecContext, config: ConfigT, param: *Param, state: *State, grad: *const RawTensor) !void {
            state.step += 1;
            updateFn(ctx, config, param.data(), grad.dataConst(), state.m, state.v, state.step);
        }
    };
}

pub const AdamW = Optimizer(MomentPairKernel(AdamWConfig, .{ .v3 = "FZA3", .v4 = "FZA4", .v5 = "FZA5" }, adamwUpdate));

/// The exact PyTorch `_single_tensor_adam(decoupled_weight_decay=True)` update.
/// Order matters: decay multiplies the parameter BEFORE the moment update and
/// Adam step; `eps` is added AFTER dividing sqrt(v) by sqrt(bias_correction2);
/// the 1/bias_correction1 lives in the scalar step size. Scalar prep runs in
/// f64 and is rounded once to f32 — matching torch's Python-float scalars to
/// within a few f32 ulps (bit parity is impossible with f32 config fields).
/// The three reference loops are fused into one element-independent pass
/// (same per-element op order, less memory traffic) and chunked across the
/// worker pool. All arithmetic is f32 for every state dtype: bf16 moments are
/// widened on read and narrowed on write, and the parameter update uses the
/// just-computed (pre-narrow) f32 moments; the f32/f32 instantiation is
/// op-for-op the pre-`StateDType` kernel.
const AdamWScalars = struct {
    keep: f32,
    beta2: f32,
    one_minus_b1: f32,
    one_minus_b2: f32,
    step_size: f32,
    bc2s: f32,
    eps: f32,
};

fn AdamWMap(comptime md: StateDType, comptime vd: StateDType) type {
    return struct {
        p: []f32,
        g: []const f32,
        m: StateSlice(md),
        v: StateSlice(vd),
        s: AdamWScalars,

        fn run(c: @This(), start: usize, end: usize) void {
            // Hand-vectorized update (see `state_vec_len`); bit-identical to
            // runScalar per element on every state dtype — the lanes are
            // independent and vector sqrt/divide are per-lane IEEE ops.
            const keep: StateVec = @splat(c.s.keep);
            const beta2: StateVec = @splat(c.s.beta2);
            const one_minus_b1: StateVec = @splat(c.s.one_minus_b1);
            const one_minus_b2: StateVec = @splat(c.s.one_minus_b2);
            const step_size: StateVec = @splat(c.s.step_size);
            const bc2s: StateVec = @splat(c.s.bc2s);
            const eps: StateVec = @splat(c.s.eps);
            var i = start;
            while (i + state_vec_len <= end) : (i += state_vec_len) {
                const pv: StateVec = c.p[i..][0..state_vec_len].*;
                const gv: StateVec = c.g[i..][0..state_vec_len].*;
                const decayed = pv * keep;
                const m0 = stateVecLoad(md, c.m[i..][0..state_vec_len]);
                const v0 = stateVecLoad(vd, c.v[i..][0..state_vec_len]);
                const m1 = m0 + one_minus_b1 * (gv - m0);
                const v1 = beta2 * v0 + one_minus_b2 * gv * gv;
                stateVecStore(md, c.m[i..][0..state_vec_len], m1);
                stateVecStore(vd, c.v[i..][0..state_vec_len], v1);
                c.p[i..][0..state_vec_len].* = decayed - step_size * (m1 / (@sqrt(v1) / bc2s + eps));
            }
            runScalar(c, i, end);
        }

        fn runScalar(c: @This(), start: usize, end: usize) void {
            for (c.p[start..end], c.g[start..end], c.m[start..end], c.v[start..end]) |*pi, gi, *mi, *vi| {
                const decayed = pi.* * c.s.keep;
                const m0 = stateLoad(md, mi);
                const v0 = stateLoad(vd, vi);
                const m1 = m0 + c.s.one_minus_b1 * (gi - m0);
                const v1 = c.s.beta2 * v0 + c.s.one_minus_b2 * gi * gi;
                stateStore(md, mi, m1);
                stateStore(vd, vi, v1);
                pi.* = decayed - c.s.step_size * (m1 / (@sqrt(v1) / c.s.bc2s + c.s.eps));
            }
        }
    };
}

fn adamwRun(comptime md: StateDType, comptime vd: StateDType, ctx: *ExecContext, s: AdamWScalars, p: []f32, g: []const f32, m: StateSlice(md), v: StateSlice(vd)) void {
    const Map = AdamWMap(md, vd);
    parallelMap(ctx, p.len, Map{ .p = p, .g = g, .m = m, .v = v, .s = s }, Map.run);
}

fn adamwUpdate(ctx: *ExecContext, config: AdamWConfig, p: []f32, g: []const f32, m: StateBuf, v: StateBuf, step_count: u64) void {
    const t: f64 = @floatFromInt(step_count);
    const bc1 = 1 - std.math.pow(f64, config.beta1, t);
    const bc2_sqrt = @sqrt(1 - std.math.pow(f64, config.beta2, t));
    const s = AdamWScalars{
        .keep = if (config.weight_decay != 0) @floatCast(1.0 - @as(f64, config.lr) * @as(f64, config.weight_decay)) else 1,
        .beta2 = config.beta2,
        .one_minus_b1 = @floatCast(1.0 - @as(f64, config.beta1)),
        .one_minus_b2 = @floatCast(1.0 - @as(f64, config.beta2)),
        .step_size = @floatCast(@as(f64, config.lr) / bc1),
        .bc2s = @floatCast(bc2_sqrt),
        .eps = config.eps,
    };
    switch (m) {
        .f32 => |ms| switch (v) {
            .f32 => |vs| adamwRun(.f32, .f32, ctx, s, p, g, ms, vs),
            .bf16 => |vs| adamwRun(.f32, .bf16, ctx, s, p, g, ms, vs),
        },
        .bf16 => |ms| switch (v) {
            .f32 => |vs| adamwRun(.bf16, .f32, ctx, s, p, g, ms, vs),
            .bf16 => |vs| adamwRun(.bf16, .bf16, ctx, s, p, g, ms, vs),
        },
    }
}

// ---------------------------------------------------------------------------
// Adam — PyTorch torch.optim.Adam single-tensor semantics.
// ---------------------------------------------------------------------------

pub const AdamConfig = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0,
    /// Storage dtype of the FIRST moment (m); see `AdamWConfig.state_dtype`.
    state_dtype: StateDType = .f32,
    /// Storage dtype of the SECOND moment (v); see
    /// `AdamWConfig.second_moment_dtype` for the bf16 v-stall math.
    second_moment_dtype: StateDType = .f32,
};

pub const Adam = Optimizer(MomentPairKernel(AdamConfig, .{ .v3 = "FZAD", .v4 = "FZD4", .v5 = "FZD5" }, adamUpdate));

/// PyTorch Adam keeps weight decay coupled to the gradient. This differs from
/// AdamW only when `weight_decay != 0`; NAM packed training uses that path.
/// State dtype handling mirrors `AdamWMap` (widen-on-read / narrow-on-write,
/// f32 math, pre-narrow values used within the element).
const AdamScalars = struct {
    weight_decay: f32,
    beta2: f32,
    one_minus_b1: f32,
    one_minus_b2: f32,
    step_size: f32,
    bc2s: f32,
    eps: f32,
};

fn AdamMap(comptime md: StateDType, comptime vd: StateDType) type {
    return struct {
        p: []f32,
        g: []const f32,
        m: StateSlice(md),
        v: StateSlice(vd),
        s: AdamScalars,

        fn run(c: @This(), start: usize, end: usize) void {
            // Hand-vectorized update (see `state_vec_len`); bit-identical to
            // runScalar per element on every state dtype — the lanes are
            // independent and vector sqrt/divide are per-lane IEEE ops. The decay branch is loop-invariant.
            const wd: StateVec = @splat(c.s.weight_decay);
            const beta2: StateVec = @splat(c.s.beta2);
            const one_minus_b1: StateVec = @splat(c.s.one_minus_b1);
            const one_minus_b2: StateVec = @splat(c.s.one_minus_b2);
            const step_size: StateVec = @splat(c.s.step_size);
            const bc2s: StateVec = @splat(c.s.bc2s);
            const eps: StateVec = @splat(c.s.eps);
            var i = start;
            while (i + state_vec_len <= end) : (i += state_vec_len) {
                const pv: StateVec = c.p[i..][0..state_vec_len].*;
                const raw_gv: StateVec = c.g[i..][0..state_vec_len].*;
                const gv = if (c.s.weight_decay != 0) raw_gv + wd * pv else raw_gv;
                const m0 = stateVecLoad(md, c.m[i..][0..state_vec_len]);
                const v0 = stateVecLoad(vd, c.v[i..][0..state_vec_len]);
                const m1 = m0 + one_minus_b1 * (gv - m0);
                const v1 = beta2 * v0 + one_minus_b2 * gv * gv;
                stateVecStore(md, c.m[i..][0..state_vec_len], m1);
                stateVecStore(vd, c.v[i..][0..state_vec_len], v1);
                c.p[i..][0..state_vec_len].* = pv - step_size * (m1 / (@sqrt(v1) / bc2s + eps));
            }
            runScalar(c, i, end);
        }

        fn runScalar(c: @This(), start: usize, end: usize) void {
            for (c.p[start..end], c.g[start..end], c.m[start..end], c.v[start..end]) |*pi, raw_gi, *mi, *vi| {
                const gi = if (c.s.weight_decay != 0) raw_gi + c.s.weight_decay * pi.* else raw_gi;
                const m0 = stateLoad(md, mi);
                const v0 = stateLoad(vd, vi);
                const m1 = m0 + c.s.one_minus_b1 * (gi - m0);
                const v1 = c.s.beta2 * v0 + c.s.one_minus_b2 * gi * gi;
                stateStore(md, mi, m1);
                stateStore(vd, vi, v1);
                pi.* -= c.s.step_size * (m1 / (@sqrt(v1) / c.s.bc2s + c.s.eps));
            }
        }
    };
}

fn adamRun(comptime md: StateDType, comptime vd: StateDType, ctx: *ExecContext, s: AdamScalars, p: []f32, g: []const f32, m: StateSlice(md), v: StateSlice(vd)) void {
    const Map = AdamMap(md, vd);
    parallelMap(ctx, p.len, Map{ .p = p, .g = g, .m = m, .v = v, .s = s }, Map.run);
}

fn adamUpdate(ctx: *ExecContext, config: AdamConfig, p: []f32, g: []const f32, m: StateBuf, v: StateBuf, step_count: u64) void {
    const t: f64 = @floatFromInt(step_count);
    const bc1 = 1 - std.math.pow(f64, config.beta1, t);
    const bc2_sqrt = @sqrt(1 - std.math.pow(f64, config.beta2, t));
    const s = AdamScalars{
        .weight_decay = config.weight_decay,
        .beta2 = config.beta2,
        .one_minus_b1 = @floatCast(1.0 - @as(f64, config.beta1)),
        .one_minus_b2 = @floatCast(1.0 - @as(f64, config.beta2)),
        .step_size = @floatCast(@as(f64, config.lr) / bc1),
        .bc2s = @floatCast(bc2_sqrt),
        .eps = config.eps,
    };
    switch (m) {
        .f32 => |ms| switch (v) {
            .f32 => |vs| adamRun(.f32, .f32, ctx, s, p, g, ms, vs),
            .bf16 => |vs| adamRun(.f32, .bf16, ctx, s, p, g, ms, vs),
        },
        .bf16 => |ms| switch (v) {
            .f32 => |vs| adamRun(.bf16, .f32, ctx, s, p, g, ms, vs),
            .bf16 => |vs| adamRun(.bf16, .bf16, ctx, s, p, g, ms, vs),
        },
    }
}
