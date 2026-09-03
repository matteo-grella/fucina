//! Muon: Keller Jordan's reference (lerp-form momentum, Newton-Schulz-5
//! orthogonalization with the transpose trick) with the Moonlight
//! RMS-matching scale variant, over an embedded AdamW fallback for
//! non-matrix params. Frame magics FZM3/FZM4/FZM5; the fallback's own
//! frame follows Muon's.

const std = @import("std");
const common = @import("common.zig");
const frame = @import("frame.zig");
const optimizer = @import("optimizer.zig");
const moment_pair = @import("moment_pair.zig");

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
const sumSquares = common.sumSquares;
const Optimizer = optimizer.Optimizer;
const FallbackFrame = optimizer.FallbackFrame;
const Magics = frame.Magics;
const AdamWConfig = moment_pair.AdamWConfig;
const AdamW = moment_pair.AdamW;

// ---------------------------------------------------------------------------
// Muon — Keller Jordan's reference with the Moonlight scale/decay variant.
// ---------------------------------------------------------------------------

pub const MuonScale = enum {
    /// Keller's original: update *= sqrt(max(1, rows/cols)); lr stays in
    /// spectral-norm units.
    spectral,
    /// Moonlight's `0.2*sqrt(max(rows, cols))`: makes Muon reuse lr/wd tuned
    /// for AdamW (update RMS ~= 0.2).
    match_rms_adamw,
};

pub const MuonConfig = struct {
    lr: f32 = 0.02,
    momentum: f32 = 0.95,
    nesterov: bool = true,
    ns_steps: u32 = 5,
    weight_decay: f32 = 0,
    scale: MuonScale = .spectral,
    /// Storage dtype of the momentum buffer; step math stays f32. bf16 is
    /// safe here (per-step relative change ~5% at momentum 0.95; the Keller
    /// reference even runs the whole pipeline in bf16 on GPU). The embedded
    /// AdamW fallback has its own `state_dtype`/`second_moment_dtype` below.
    state_dtype: StateDType = .f32,
    /// AdamW for everything Muon must not touch: 0D/1D params route here
    /// automatically; route embeddings and output heads here explicitly via
    /// `addFallbackParam` (they are 2D but are not hidden-space maps).
    fallback: AdamWConfig = .{ .lr = 3e-4, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-10, .weight_decay = 0 },
};

const MuonKernel = struct {
    pub const Config = MuonConfig;
    pub const magics: Magics = .{ .v3 = "FZM3", .v4 = "FZM4", .v5 = "FZM5" };
    pub const pinned_config_fields = [_][]const u8{ "scale", "nesterov", "ns_steps" };
    pub const record = [_][]const u8{"momentum"};

    /// Matrix params (raw rank >= 2; conv filters are flattened `[d0, rest]`)
    /// get Muon; lower-rank params route to the AdamW fallback, which
    /// frames its m/v independently after Muon's own slots.
    pub const Fallback = AdamW;
    pub const fallback_frame: FallbackFrame = .nested;

    pub fn fallbackConfig(config: MuonConfig) AdamWConfig {
        return config.fallback;
    }

    pub fn routesToFallback(param: *const Param) bool {
        return param.raw_rank < 2;
    }

    pub const State = struct {
        momentum: StateBuf,

        pub fn deinit(self: *State, allocator: Allocator) void {
            self.momentum.deinit(allocator);
            self.* = undefined;
        }
    };

    pub fn initState(allocator: Allocator, config: MuonConfig, param: *const Param, _: usize) !State {
        return .{ .momentum = try StateBuf.alloc(allocator, config.state_dtype, param.len()) };
    }

    pub fn update(ctx: *ExecContext, config: MuonConfig, param: *Param, state: *State, grad: *const RawTensor) !void {
        const rows = param.rows;
        const cols = param.cols;
        const g = grad.dataConst();
        var u = try ctx.empty(.f32, .{ rows, cols });
        defer u.deinit();
        switch (state.momentum) {
            .f32 => |ms| momentumRun(.f32, ctx, ms, g, u.data(), config.momentum, config.nesterov),
            .bf16 => |ms| momentumRun(.bf16, ctx, ms, g, u.data(), config.momentum, config.nesterov),
        }

        var ortho = try newtonSchulz5(ctx, &u, config.ns_steps);
        defer ortho.deinit();

        const rows_f: f32 = @floatFromInt(rows);
        const cols_f: f32 = @floatFromInt(cols);
        const lr_eff = switch (config.scale) {
            .spectral => config.lr * @sqrt(@max(1, rows_f / cols_f)),
            .match_rms_adamw => config.lr * 0.2 * @sqrt(@max(rows_f, cols_f)),
        };
        parallelMap(ctx, param.len(), ApplyMapContext{
            .p = param.data(),
            .o = ortho.dataConst(),
            .keep = if (config.weight_decay != 0) @floatCast(1.0 - @as(f64, config.lr) * @as(f64, config.weight_decay)) else 1,
            .lr_eff = lr_eff,
        }, applyMapRange);
    }
};

pub const Muon = Optimizer(MuonKernel);

fn MomentumMap(comptime sd: StateDType) type {
    return struct {
        m: StateSlice(sd),
        g: []const f32,
        u: []f32,
        beta: f32,
        nesterov: bool,

        fn run(c: @This(), start: usize, end: usize) void {
            if (comptime sd == .f32) {
                // The golden-pinned baseline stays on the original scalar loop.
                runScalar(c, start, end);
                return;
            }
            // Hand-vectorized bf16 arm (see `state_vec_len`); bit-identical
            // to runScalar per element. The lerp-form and nesterov branches
            // are loop-invariant.
            const w = 1 - c.beta;
            const wv: StateVec = @splat(w);
            const betav: StateVec = @splat(c.beta);
            var i = start;
            while (i + state_vec_len <= end) : (i += state_vec_len) {
                const gv: StateVec = c.g[i..][0..state_vec_len].*;
                const m0 = stateVecLoad(sd, c.m[i..][0..state_vec_len]);
                const m1 = if (w < 0.5) m0 + wv * (gv - m0) else gv - (gv - m0) * betav;
                stateVecStore(sd, c.m[i..][0..state_vec_len], m1);
                c.u[i..][0..state_vec_len].* = if (c.nesterov)
                    (if (c.beta < 0.5) gv + betav * (m1 - gv) else m1 - (m1 - gv) * wv)
                else
                    m1;
            }
            runScalar(c, i, end);
        }

        fn runScalar(c: @This(), start: usize, end: usize) void {
            // Keller's lerp form: M <- M + (1-beta)*(g - M). The nesterov update
            // blends the raw grad with the NEW buffer: U = lerp(g, M, beta).
            // ATen's lerp kernel switches forms at |weight| = 0.5; mirror both
            // branches so the FP op sequence matches the torch-run reference
            // (beta defaults to 0.95, taking the end-anchored form). bf16
            // momentum widens on read, narrows on write; U uses the
            // just-computed (pre-narrow) f32 buffer value.
            for (c.m[start..end], c.g[start..end], c.u[start..end]) |*mi, gi, *ui| {
                const w = 1 - c.beta;
                const m0 = stateLoad(sd, mi);
                const m1 = if (w < 0.5) m0 + w * (gi - m0) else gi - (gi - m0) * c.beta;
                stateStore(sd, mi, m1);
                ui.* = if (c.nesterov)
                    (if (c.beta < 0.5) gi + c.beta * (m1 - gi) else m1 - (m1 - gi) * w)
                else
                    m1;
            }
        }
    };
}

fn momentumRun(comptime sd: StateDType, ctx: *ExecContext, m: StateSlice(sd), g: []const f32, u: []f32, beta: f32, nesterov: bool) void {
    const Map = MomentumMap(sd);
    parallelMap(ctx, g.len, Map{ .m = m, .g = g, .u = u, .beta = beta, .nesterov = nesterov }, Map.run);
}

const ApplyMapContext = struct {
    p: []f32,
    o: []const f32,
    keep: f32,
    lr_eff: f32,
};

fn applyMapRange(c: ApplyMapContext, start: usize, end: usize) void {
    // Decoupled decay with the BASE lr, then the shape-scaled update.
    for (c.p[start..end], c.o[start..end]) |*pi, oi| {
        pi.* = pi.* * c.keep - c.lr_eff * oi;
    }
}

const ns_coeff_a: f32 = 3.4445;
const ns_coeff_b: f32 = -4.7750;
const ns_coeff_c: f32 = 2.0315;

/// Keller's `zeropower_via_newtonschulz5`: Frobenius-normalize so the spectral
/// norm is <= 1, then iterate the tuned quintic X <- a*X + (b*A + c*A*A)*X with
/// A = X*X^T. When rows > cols the iteration runs on X^T so the Gram matrix has
/// the small dimension. The reference runs in bf16 on GPU; f32 here is strictly
/// more accurate. The result approximates U*V^T with singular values in roughly
/// (0.5, 1.5) — by design, not a bug. `u` must be rank-2 and is never mutated.
/// The Newton-Schulz elementwise fixups as pool maps (`parallelMap`:
/// element-independent, so bitwise the serial loop for any part count).
const Scale = struct {
    z: []f32,
    s: f32,
    fn run(c: @This(), start: usize, end: usize) void {
        for (c.z[start..end]) |*value| value.* *= c.s;
    }
};

/// z = a·x + b·z
const Axpby = struct {
    z: []f32,
    x: []const f32,
    a: f32,
    b: f32,
    fn run(c: @This(), start: usize, end: usize) void {
        for (c.z[start..end], c.x[start..end]) |*zi, xi| zi.* = c.a * xi + c.b * zi.*;
    }
};

/// z += a·x
const Axpy = struct {
    z: []f32,
    x: []const f32,
    a: f32,
    fn run(c: @This(), start: usize, end: usize) void {
        for (c.z[start..end], c.x[start..end]) |*zi, xi| zi.* = c.a * xi + zi.*;
    }
};

pub fn newtonSchulz5(ctx: *ExecContext, u: *const RawTensor, steps: u32) !RawTensor {
    const rows = u.shape.at(0);
    const cols = u.shape.at(1);
    const transposed = rows > cols;
    var x = if (transposed) try transpose2D(ctx, u) else try ctx.materialize(.f32, u);
    errdefer x.deinit();

    const sumsq = try sumSquares(ctx, x.dataConst());
    const inv_norm: f32 = @floatCast(1.0 / (@sqrt(sumsq) + 1e-7));
    parallelMap(ctx, x.len(), Scale{ .z = x.data(), .s = inv_norm }, Scale.run);

    for (0..steps) |_| {
        var gram = try ctx.matmul(.f32, .trans_b, &x, &x);
        defer gram.deinit();
        var quad = try ctx.matmul(.f32, .plain, &gram, &gram);
        defer quad.deinit();
        parallelMap(ctx, quad.len(), Axpby{ .z = quad.data(), .x = gram.dataConst(), .a = ns_coeff_b, .b = ns_coeff_c }, Axpby.run);
        var bx = try ctx.matmul(.f32, .plain, &quad, &x);
        errdefer bx.deinit();
        parallelMap(ctx, bx.len(), Axpy{ .z = bx.data(), .x = x.dataConst(), .a = ns_coeff_a }, Axpy.run);
        x.deinit();
        x = bx;
    }

    if (transposed) {
        const out = try transpose2D(ctx, &x);
        x.deinit();
        return out;
    }
    return x;
}

fn transpose2D(ctx: *ExecContext, t: *const RawTensor) !RawTensor {
    const rows = t.shape.at(0);
    const cols = t.shape.at(1);
    var view = try t.viewWithStrides(&.{ cols, rows }, &.{ 1, cols });
    defer view.deinit();
    return try ctx.materialize(.f32, &view);
}
