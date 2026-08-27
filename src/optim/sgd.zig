//! SGD: PyTorch torch.optim.SGD single-tensor semantics (coupled L2,
//! first-step buffer = decayed gradient, optional nesterov). Frame magics
//! FZS3/FZS4/FZS5.

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
// SGD — PyTorch torch.optim.SGD single-tensor semantics.
// ---------------------------------------------------------------------------

pub const SgdConfig = struct {
    lr: f32 = 1e-3,
    /// 0 disables the momentum buffer entirely (no state RAM).
    momentum: f32 = 0,
    dampening: f32 = 0,
    /// COUPLED L2 (g += wd*p), like PyTorch SGD — not AdamW-style decoupled.
    weight_decay: f32 = 0,
    /// Requires momentum > 0 and dampening == 0 (PyTorch constructor rule).
    nesterov: bool = false,
    /// Storage dtype of the momentum buffer; step math stays f32. bf16 is
    /// safe here (per-step relative change ~10% at momentum 0.9). Ignored
    /// when `momentum == 0` (no buffer exists). Note the first-step
    /// `buf = d_p.clone()` semantics store the NARROWED decayed gradient.
    state_dtype: StateDType = .f32,
};

const SgdKernel = struct {
    pub const Config = SgdConfig;
    pub const magics: Magics = .{ .v3 = "FZS3", .v4 = "FZS4", .v5 = "FZS5" };
    pub const pinned_config_fields = [_][]const u8{ "momentum", "dampening", "nesterov" };
    pub const record = [_][]const u8{ "step", "buf" };

    pub const State = struct {
        /// Momentum buffer; empty (f32-tagged, so the frame stays v3) when
        /// momentum == 0. PyTorch initializes it to a CLONE OF THE FIRST
        /// (decayed) GRADIENT, not zeros — `step` tracks whether that has
        /// happened.
        buf: StateBuf,
        step: u64 = 0,

        pub fn deinit(self: *State, allocator: Allocator) void {
            self.buf.deinit(allocator);
            self.* = undefined;
        }
    };

    /// PyTorch constructor rule, enforced in every build mode (a debug
    /// assert would vanish exactly where training runs: ReleaseFast):
    /// nesterov requires momentum > 0 and dampening == 0.
    pub fn checkConfig(config: SgdConfig) !void {
        if (config.nesterov and (config.momentum == 0 or config.dampening != 0)) {
            return error.InvalidOptimizerConfig;
        }
    }

    pub fn initState(allocator: Allocator, config: SgdConfig, param: *const Param, _: usize) !State {
        const buf: StateBuf = if (config.momentum != 0)
            try StateBuf.alloc(allocator, config.state_dtype, param.len())
        else
            .{ .f32 = &.{} };
        return .{ .buf = buf };
    }

    pub fn update(ctx: *ExecContext, config: SgdConfig, param: *Param, state: *State, grad: *const RawTensor) !void {
        state.step += 1;
        sgdUpdate(ctx, config, param.data(), grad.dataConst(), state.buf, state.step == 1);
    }
};

pub const SGD = Optimizer(SgdKernel);

/// The exact PyTorch SGD step. With weight decay the L2 term joins the
/// gradient BEFORE the momentum buffer sees it; on the very first step the
/// buffer is initialized to that (decayed) gradient itself — not zeros —
/// matching `buf = d_p.clone()`. One fused element-independent pass, chunked
/// across the worker pool. bf16 momentum widens on read and narrows on write;
/// the nesterov blend and the parameter update use the just-computed
/// (pre-narrow) f32 buffer value.
fn SgdMap(comptime sd: StateDType) type {
    return struct {
        p: []f32,
        g: []const f32,
        buf: StateSlice(sd),
        config: SgdConfig,
        one_minus_damp: f32,
        first_step: bool,

        fn run(c: @This(), start: usize, end: usize) void {
            if (comptime sd == .f32) {
                // The golden-pinned baseline stays on the original scalar loop.
                runScalar(c, start, end);
                return;
            }
            // Hand-vectorized bf16 arm (see `state_vec_len`); bit-identical
            // to runScalar per element. A bf16 buffer only exists with
            // momentum != 0; the decay/first-step/nesterov branches are
            // loop-invariant.
            const config = c.config;
            const wdv: StateVec = @splat(config.weight_decay);
            const momentumv: StateVec = @splat(config.momentum);
            const one_minus_dampv: StateVec = @splat(c.one_minus_damp);
            const lrv: StateVec = @splat(config.lr);
            var i = start;
            while (i + state_vec_len <= end) : (i += state_vec_len) {
                const pv: StateVec = c.p[i..][0..state_vec_len].*;
                var gv: StateVec = c.g[i..][0..state_vec_len].*;
                if (config.weight_decay != 0) gv += wdv * pv;
                if (config.momentum != 0) {
                    const b1 = if (c.first_step)
                        gv
                    else
                        momentumv * stateVecLoad(sd, c.buf[i..][0..state_vec_len]) + one_minus_dampv * gv;
                    stateVecStore(sd, c.buf[i..][0..state_vec_len], b1);
                    gv = if (config.nesterov) gv + momentumv * b1 else b1;
                }
                c.p[i..][0..state_vec_len].* = pv - lrv * gv;
            }
            runScalar(c, i, end);
        }

        fn runScalar(c: @This(), start: usize, end: usize) void {
            const config = c.config;
            for (c.p[start..end], c.g[start..end], start..) |*pi, gi_raw, i| {
                var gi = gi_raw;
                if (config.weight_decay != 0) gi += config.weight_decay * pi.*;
                if (config.momentum != 0) {
                    const b1 = if (c.first_step)
                        gi
                    else
                        config.momentum * stateLoad(sd, &c.buf[i]) + c.one_minus_damp * gi;
                    stateStore(sd, &c.buf[i], b1);
                    gi = if (config.nesterov) gi + config.momentum * b1 else b1;
                }
                pi.* -= config.lr * gi;
            }
        }
    };
}

fn sgdRun(comptime sd: StateDType, ctx: *ExecContext, config: SgdConfig, p: []f32, g: []const f32, buf: StateSlice(sd), first_step: bool) void {
    const Map = SgdMap(sd);
    parallelMap(ctx, p.len, Map{
        .p = p,
        .g = g,
        .buf = buf,
        .config = config,
        .one_minus_damp = @floatCast(1.0 - @as(f64, config.dampening)),
        .first_step = first_step,
    }, Map.run);
}

fn sgdUpdate(ctx: *ExecContext, config: SgdConfig, p: []f32, g: []const f32, buf: StateBuf, first_step: bool) void {
    switch (buf) {
        .f32 => |bs| sgdRun(.f32, ctx, config, p, g, bs, first_step),
        .bf16 => |bs| sgdRun(.bf16, ctx, config, p, g, bs, first_step),
    }
}
