//! Neural-net building blocks for the face-detect port, thin wrappers over
//! the public tagged `Tensor` facade. The pool / upsample / PReLU /
//! channel-affine primitives are core ops (`src/exec/pool.zig`,
//! `src/exec/elementwise.zig`): one fused, vectorized, row-parallel pass each,
//! with VJPs — value-identical to the equivalent multi-op compositions
//! (select == relu + α·(x−relu); mul-then-add is never fma-contracted).
//! Feature maps are channel-last `Tensor(.{ .h, .w, .c })`.

const fucina = @import("fucina");
const ExecContext = fucina.ExecContext;

pub const Map = fucina.Tensor(.{ .h, .w, .c });
pub const Channels = fucina.Tensor(.{.c});

/// 2×2 stride-2 average pool (SCRFD downsample-shortcut AvgPool).
pub fn avgPool2x2(ctx: *ExecContext, x: *const Map) !Map {
    return x.avgPool2d(ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 });
}

/// 2×2 stride-2 max pool (SCRFD stem MaxPool).
pub fn maxPool2x2(ctx: *ExecContext, x: *const Map) !Map {
    return x.maxPool2d(ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 });
}

/// PReLU with a per-channel LEARNABLE slope (insightface ArcFace/MiniFASNet):
/// prelu(x) = max(x,0) + α·min(x,0) = x > 0 ? x : α[c]·x — differentiable in
/// BOTH x and α (the learnable generalization of `leakyRelu`).
pub fn prelu(ctx: *ExecContext, x: *const Map, alpha: *const Channels) !Map {
    return x.prelu(ctx, alpha);
}

/// Frozen-stats BatchNorm folded to a per-channel (scale, shift) pair:
/// scale = γ/√(var+ε), shift = β − μ·scale (the reference's `bn_fold`).
/// Computed once at model-load time so every forward is a single
/// `channelAffine` pass.
pub const BnScaleShift = struct {
    scale: Channels,
    shift: Channels,

    pub fn deinit(self: *BnScaleShift) void {
        self.scale.deinit();
        self.shift.deinit();
        self.* = undefined;
    }
};

pub fn bnFold(
    ctx: *ExecContext,
    gamma: *const Channels,
    beta: *const Channels,
    mean: *const Channels,
    variance: *const Channels,
    eps: f32,
) !BnScaleShift {
    var ve = try variance.addScalar(ctx, eps); // var + ε
    defer ve.deinit();
    var std_ = try ve.sqrt(ctx); // √(var+ε)
    defer std_.deinit();
    var scale = try gamma.div(ctx, &std_); // γ/√(var+ε)   [.c]
    errdefer scale.deinit();
    var ms = try mean.mul(ctx, &scale); // μ·scale        [.c]
    defer ms.deinit();
    const shift = try beta.sub(ctx, &ms); // β − μ·scale   [.c]
    return .{ .scale = scale, .shift = shift };
}

/// Standalone inference BatchNorm (the explicit BN nodes in genderage /
/// MiniFASNet / ArcFace pre-activation): fold the frozen stats, then one
/// per-channel affine pass; `eps` is the per-node epsilon.
pub fn batchNormInfer(
    ctx: *ExecContext,
    x: *const Map,
    gamma: *const Channels,
    beta: *const Channels,
    mean: *const Channels,
    variance: *const Channels,
    eps: f32,
) !Map {
    var ss = try bnFold(ctx, gamma, beta, mean, variance, eps);
    defer ss.deinit();
    return x.channelAffine(ctx, &ss.scale, &ss.shift);
}

/// Train-mode BatchNorm: normalize each channel by its BATCH statistics (mean/
/// var over all spatial positions — here the single map's H·W), then affine.
/// Composed from merge → mean/variance(ddof=0) → (x−μ)/√(var+ε) → γ·+β → split,
/// all differentiable, so the (x, γ, β) VJP is inherited from the composition. The
/// eval-mode counterpart is `batchNormInfer` (frozen stats) — together they are
/// the train/eval switch. Running-stat EMA update is a non-gradient side effect
/// the caller owns; not needed for the VJP.
pub fn batchNormTrain(ctx: *ExecContext, x: *const Map, gamma: *const Channels, beta: *const Channels, eps: f32) !Map {
    const nh = x.dim(.h);
    const nw = x.dim(.w);
    var m = try x.merge(ctx, .n, .{ .h, .w }); // [n, c]
    defer m.deinit();
    var mu = try m.mean(ctx, .n); // [c]
    defer mu.deinit();
    var vr = try m.variance(ctx, .n, 0); // [c] population variance
    defer vr.deinit();
    var ve = try vr.addScalar(ctx, eps);
    defer ve.deinit();
    var sd = try ve.sqrt(ctx);
    defer sd.deinit();
    var centered = try m.sub(ctx, &mu); // [n,c] − μ broadcast
    defer centered.deinit();
    var norm = try centered.div(ctx, &sd); // / √(var+ε) broadcast
    defer norm.deinit();
    var scaled = try norm.mul(ctx, gamma);
    defer scaled.deinit();
    var affine = try scaled.add(ctx, beta);
    defer affine.deinit();
    return affine.split(ctx, .n, .{ .h, .w }, .{ nh, nw });
}

/// 2× nearest-neighbour upsample (SCRFD PAFPN top-down neck):
/// out[2h+i, 2w+j] = in[h,w] — one row-duplicating pass (VJP = 2×2 sum-pool).
pub fn upsample2xNearest(ctx: *ExecContext, x: *const Map) !Map {
    return x.upsample2xNearest(ctx);
}
