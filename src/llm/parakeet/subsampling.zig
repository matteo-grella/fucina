//! Parakeet FastConformer subsampling stem: mel `[n_mels, T]` (feat-major)
//! → `[T', d_model]` (row-major `out[t*d_model + c]`). The NeMo `dw_striding`
//! 8× subsampling: 3 stride-2 conv stages (conv.0 full, then depthwise+pointwise
//! ×2) with ReLU, channel-major flatten, then a linear projection to d_model.
//!
//! Matches the non-causal offline path in parakeet.cpp
//! `src/subsampling.cpp::build_graph` (the 110m is `causal_downsampling=false`,
//! so: symmetric pad=1 each stage, no input/output masking since valid_out==T').
//! All in f32 (ggml's conv path is f32). Conv weights are f32 in every quant
//! format; `out.weight` is f16/k-quant → dequantized to f32.
const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;
const loader = @import("loader.zig");
const pweights = @import("weights.zig");
const ParakeetWeights = pweights.ParakeetWeights;

const ExecContext = fucina.ExecContext;
const Allocator = std.mem.Allocator;

/// Borrow a contiguous f32 GGUF tensor's bytes as `[]const f32` (from the
/// mapping), guarded against misaligned/odd-length untrusted bytes. Conv
/// kernels/biases are always f32.
fn f32Data(info: *const gguf.TensorInfo) ![]const f32 {
    return pweights.borrowF32(info.data);
}

/// Build a conv weight tensor with the conv2d layout `[Cout, KH, KW, Cinpg]`.
/// The GGUF ggml memory order matches this layout for every subsampling conv
/// (each has IC=1 or KH=KW=1), so no repacking is needed.
fn convWeight(ctx: *ExecContext, file: *const gguf.File, name: []const u8, cout: usize, kh: usize, kw: usize, cinpg: usize) !fucina.Tensor(4) {
    const info = try file.get(name);
    return fucina.Tensor(4).fromSlice(ctx, .{ cout, kh, kw, cinpg }, try f32Data(info));
}

fn vecTensor(ctx: *ExecContext, file: *const gguf.File, name: []const u8, n: usize) !fucina.Tensor(1) {
    const info = try file.get(name);
    return fucina.Tensor(1).fromSlice(ctx, .{n}, try f32Data(info));
}

/// Route a raw conv2d through the PUBLIC facade conv2d: bridge
/// the borrowed operands to facade constants (cloneView retains them; the
/// wrappers are local and freed here) and unwrap the owned raw result. Same
/// kernel — zero numeric change. Conv input `[H,W,Cin]` (rank 3), weight
/// `[Cout,KH,KW,Cinpg]` (rank 4), bias `[Cout]` (rank 1). Shared with the encoder
/// conv-module depthwise.
pub fn conv2dPublic(ctx: *ExecContext, input: *const fucina.Tensor(3), weight: anytype, bias: anytype, stride: [2]usize, pad: [2]usize, groups: usize) !fucina.Tensor(3) {
    return input.conv2d(ctx, weight, bias, stride, pad, groups, 3);
}

/// ReLU of a rank-3 conv output `[H,W,C]` via the public facade `relu`.
fn reluRank3Public(ctx: *ExecContext, x: *const fucina.Tensor(3)) !fucina.Tensor(3) {
    return x.relu(ctx);
}

/// One depthwise(3×3,s2,p1)+bias → pointwise(1×1)+bias → ReLU stage. Consumes
/// `x` (deinits it) and returns the new tensor.
fn dwSepStage(
    ctx: *ExecContext,
    file: *const gguf.File,
    x: *const fucina.Tensor(3),
    c: usize,
    dw_w: []const u8,
    dw_b: []const u8,
    pw_w: []const u8,
    pw_b: []const u8,
) !fucina.Tensor(3) {
    var dww = try convWeight(ctx, file, dw_w, c, 3, 3, 1);
    defer dww.deinit();
    var dwb = try vecTensor(ctx, file, dw_b, c);
    defer dwb.deinit();
    var dw = try conv2dPublic(ctx, x, &dww, &dwb, .{ 2, 2 }, .{ 1, 1 }, c); // depthwise
    defer dw.deinit();

    var pww = try convWeight(ctx, file, pw_w, c, 1, 1, c);
    defer pww.deinit();
    var pwb = try vecTensor(ctx, file, pw_b, c);
    defer pwb.deinit();
    var pw = try conv2dPublic(ctx, &dw, &pww, &pwb, .{ 1, 1 }, .{ 0, 0 }, 1);
    defer pw.deinit();
    return reluRank3Public(ctx, &pw);
}

/// Run the subsampling stem. Returns a public `Tensor` `[T', d_model]` (row-major,
/// the subsample→encode module boundary); caller owns it (`.deinit()`).
pub fn subsample(
    ctx: *ExecContext,
    file: *const gguf.File,
    cfg: loader.Config,
    mel: []const f32, // [n_mels, T] feat-major: mel[f*T + t]
    n_mels: usize,
    t_in: usize,
) !fucina.Tensor(2) {
    return subsampleWithWeights(ctx, file, cfg, mel, n_mels, t_in, null);
}

pub fn subsampleWithWeights(
    ctx: *ExecContext,
    file: *const gguf.File,
    cfg: loader.Config,
    mel: []const f32, // [n_mels, T] feat-major: mel[f*T + t]
    n_mels: usize,
    t_in: usize,
    shared_w: ?*ParakeetWeights,
) !fucina.Tensor(2) {
    const c = cfg.subsampling_conv_channels;

    // --- mel -> conv input x[t*F + f] (conv2d layout [H=T, W=F, Cin=1]). Written
    // straight into the pooled tensor. ---
    var x_in = try fucina.Tensor(3).empty(ctx, .{ t_in, n_mels, 1 });
    defer x_in.deinit();
    const x_host = try x_in.data();
    for (0..t_in) |t| {
        for (0..n_mels) |f| x_host[t * n_mels + f] = mel[f * t_in + t];
    }

    // --- Stage 0: full conv2d(1->C, k3, s2, p1) + bias + ReLU. ---
    var w0 = try convWeight(ctx, file, "encoder.pre_encode.conv.0.weight", c, 3, 3, 1);
    defer w0.deinit();
    var b0 = try vecTensor(ctx, file, "encoder.pre_encode.conv.0.bias", c);
    defer b0.deinit();
    var c0 = try conv2dPublic(ctx, &x_in, &w0, &b0, .{ 2, 2 }, .{ 1, 1 }, 1);
    defer c0.deinit();
    var s0 = try reluRank3Public(ctx, &c0);
    // Stage 1 + 2: depthwise-separable.
    var s1 = try dwSepStage(ctx, file, &s0, c, "encoder.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.bias", "encoder.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.bias");
    s0.deinit();
    var s2 = try dwSepStage(ctx, file, &s1, c, "encoder.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.bias", "encoder.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.bias");
    s1.deinit();
    defer s2.deinit();

    // s2: conv2d output [T', F', C] channel-last (o[(t*F'+f)*C + c]). NeMo flatten:
    // per time, vector is channel-major idx = c*F' + f. Reorder into
    // flat[t*(C*F') + c*F' + f].
    const sv = s2.shape();
    const tp = sv[0];
    const fp = sv[1];
    if (sv[2] != c) return error.ShapeMismatch;
    const k = c * fp; // C*F'
    const conv_out = try s2.dataConst();
    var flat = try fucina.Tensor(2).empty(ctx, .{ tp, k });
    defer flat.deinit();
    const flat_host = try flat.data();
    for (0..tp) |t| {
        for (0..c) |cc| {
            for (0..fp) |f| {
                flat_host[t * k + cc * fp + f] = conv_out[(t * fp + f) * c + cc];
            }
        }
    }

    // --- Linear out via the Tensor-valued `linearD`: the packed/quantized linear
    // (out.weight ggml [k, d_model], loaded+packed once) + row bias.
    // y[t,d] = Σ_k flat[t,k]·W[d,k] + bias[d]. ---
    var local_w: ParakeetWeights = undefined;
    const w = if (shared_w) |pw| pw else blk: {
        local_w = ParakeetWeights.init(ctx, file);
        break :blk &local_w;
    };
    defer if (shared_w == null) local_w.deinit();
    const ob = try f32Data(try file.get("encoder.pre_encode.out.bias"));
    return w.linearD("encoder.pre_encode.out.weight", ob, &flat); // [T', d_model] (public Tensor boundary)
}

// ========================= Streaming (causal) subsampling =========================
//
// The realtime_eou model is `causal_downsampling=true`. Each k=3,s=2 stage pads
// BOTH spatial axes (time H and feature W) with left=k-1=2, right=stride-1=1
// (NeMo CausalConv2D), then convs with p=0 — so out = floor(in/2)+1 (all_paddings
// =3). Per-stage trailing-time masking (zero frames ≥ valid_t before each stage)
// reproduces NeMo MaskedConvSequential; for a fully-real chunk window it is a
// no-op. Matches `subsampling.cpp::build_graph(causal)`.

/// Asymmetric causal pad of a `[H, W, C]` conv input: left=2/right=1 on both the
/// time (H, axis 0) and feature (W, axis 1) axes → `[H+3, W+3, C]` (input copied
/// to `[2..2+H, 2..2+W]`, the rest zero). Caller owns the result.
fn causalPad(ctx: *ExecContext, x: *const fucina.Tensor(3)) !fucina.Tensor(3) {
    const v = x.shape();
    const h = v[0];
    const wid = v[1];
    const cc = v[2];
    var out = try fucina.Tensor(3).empty(ctx, .{ h + 3, wid + 3, cc });
    errdefer out.deinit();
    const od = try out.data();
    @memset(od, 0);
    const xd = try x.dataConst();
    const ow = wid + 3;
    for (0..h) |hh| {
        for (0..wid) |ww| {
            @memcpy(od[((hh + 2) * ow + (ww + 2)) * cc ..][0..cc], xd[(hh * wid + ww) * cc ..][0..cc]);
        }
    }
    return out;
}

/// Zero the trailing time rows `h >= valid_t` of a `[H, W, C]` tensor in place
/// (NeMo trailing-pad masking). No-op when `valid_t >= H`.
fn maskTimeInPlace(x: *fucina.Tensor(3), valid_t: usize) !void {
    const v = x.shape();
    const h = v[0];
    if (valid_t >= h) return;
    const wc = v[1] * v[2];
    @memset((try x.data())[valid_t * wc ..], 0);
}

fn dwSepStageCausal(
    ctx: *ExecContext,
    file: *const gguf.File,
    x: *fucina.Tensor(3),
    c: usize,
    valid_t: usize,
    dw_w: []const u8,
    dw_b: []const u8,
    pw_w: []const u8,
    pw_b: []const u8,
) !fucina.Tensor(3) {
    try maskTimeInPlace(x, valid_t);
    var xp = try causalPad(ctx, x);
    defer xp.deinit();
    var dww = try convWeight(ctx, file, dw_w, c, 3, 3, 1);
    defer dww.deinit();
    var dwb = try vecTensor(ctx, file, dw_b, c);
    defer dwb.deinit();
    var dw = try conv2dPublic(ctx, &xp, &dww, &dwb, .{ 2, 2 }, .{ 0, 0 }, c); // causal: p=0
    defer dw.deinit();

    var pww = try convWeight(ctx, file, pw_w, c, 1, 1, c);
    defer pww.deinit();
    var pwb = try vecTensor(ctx, file, pw_b, c);
    defer pwb.deinit();
    var pw = try conv2dPublic(ctx, &dw, &pww, &pwb, .{ 1, 1 }, .{ 0, 0 }, 1);
    defer pw.deinit();
    return reluRank3Public(ctx, &pw);
}

/// Streaming causal subsampling on one mel chunk `[n_mels, t_in]` (feat-major;
/// already includes the pre-encode overlap). `in_valid_frames` = the real frame
/// count of the window (= `t_in` for a full chunk). Returns the FULL public Tensor
/// `[T', d_model]` (row-major) — the caller drops `drop_extra_pre_encoded` leading frames.
pub fn streamingSubsample(
    ctx: *ExecContext,
    file: *const gguf.File,
    cfg: loader.Config,
    w: *ParakeetWeights,
    mel: []const f32,
    n_mels: usize,
    t_in: usize,
    in_valid_frames: usize,
) !fucina.Tensor(2) {
    const c = cfg.subsampling_conv_channels;

    var x_in = try fucina.Tensor(3).empty(ctx, .{ t_in, n_mels, 1 });
    defer x_in.deinit();
    const x_host = try x_in.data();
    for (0..t_in) |t| {
        for (0..n_mels) |f| x_host[t * n_mels + f] = mel[f * t_in + t];
    }

    // Per-stage valid TIME lengths (all_paddings=3 recurrence): v→v/2+1.
    const vt0 = in_valid_frames;
    const vt1 = vt0 / 2 + 1;
    const vt2 = vt1 / 2 + 1;

    // Stage 0: full conv2d(1→C, k3, s2) causal-padded + bias + ReLU.
    try maskTimeInPlace(&x_in, vt0);
    var xp = try causalPad(ctx, &x_in);
    defer xp.deinit();
    var w0 = try convWeight(ctx, file, "encoder.pre_encode.conv.0.weight", c, 3, 3, 1);
    defer w0.deinit();
    var b0 = try vecTensor(ctx, file, "encoder.pre_encode.conv.0.bias", c);
    defer b0.deinit();
    var c0 = try conv2dPublic(ctx, &xp, &w0, &b0, .{ 2, 2 }, .{ 0, 0 }, 1);
    defer c0.deinit();
    var s0 = try reluRank3Public(ctx, &c0);
    var s1 = try dwSepStageCausal(ctx, file, &s0, c, vt1, "encoder.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.bias", "encoder.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.bias");
    s0.deinit();
    var s2 = try dwSepStageCausal(ctx, file, &s1, c, vt2, "encoder.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.bias", "encoder.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.bias");
    s1.deinit();
    defer s2.deinit();

    // Flatten channel-major (idx = c*F' + f), out linear + bias. (Output-length
    // masking is a no-op when valid_out == T'; the caller's drop+slice handles
    // partial windows.)
    const sv = s2.shape();
    const tp = sv[0];
    const fp = sv[1];
    if (sv[2] != c) return error.ShapeMismatch;
    const k = c * fp;
    const conv_out = try s2.dataConst();
    var flat = try fucina.Tensor(2).empty(ctx, .{ tp, k });
    defer flat.deinit();
    const flat_host = try flat.data();
    for (0..tp) |t| {
        for (0..c) |cc| {
            for (0..fp) |f| flat_host[t * k + cc * fp + f] = conv_out[(t * fp + f) * c + cc];
        }
    }

    const ob = try f32Data(try file.get("encoder.pre_encode.out.bias"));
    return w.linearD("encoder.pre_encode.out.weight", ob, &flat); // [T', d_model] public Tensor boundary
}

test {
    _ = @import("subsampling_tests.zig");
}
