//! Parakeet FastConformer encoder: Transformer-XL relative-position multi-head
//! self-attention (NeMo `RelPositionMultiHeadAttention`) + the sinusoidal
//! relative-position table, plus the conv module and the macaron block + layer
//! stack built on top of it.
//!
//! Matches the non-causal offline path in parakeet.cpp
//! `src/relpos_attention.cpp::build_graph` + `src/pos_enc.cpp`. Projections use
//! the packed/quantized `linearSeq` path (see `linearWT`); the attention core
//! (the two bias terms + Transformer-XL rel-shift + softmax + context) runs as
//! batched matmul + a vectorized softmax (see `relposAttention`).
const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;
const loader = @import("loader.zig");

const subsampling = @import("subsampling.zig");
const pweights = @import("weights.zig");
const ParakeetWeights = pweights.ParakeetWeights;

const ExecContext = fucina.ExecContext;
const Tensor2 = fucina.Tensor(2);
const Allocator = std.mem.Allocator;

pub fn f32Data(info: *const gguf.TensorInfo) ![]const f32 {
    return pweights.borrowF32(info.data);
}

// --- Public-facade bridges for the rank-3 relpos attention. Each
// cloneView-retains the borrowed raw operands into facade constants, calls the
// PUBLIC facade op, and unwraps the owned raw result. Same kernels → zero
// numeric change. Shared with the streaming attention. ---

/// NeMo `RelPositionalEncoding` table: `[P=2T-1, d_model]`, caller-owned.
/// `pos = (T-1)-p` (row 0 = +(T-1) … last = -(T-1)); even dims sin, odd cos,
/// `div_term[i] = exp(2i · -(ln 10000 / d_model))`. Computed in double.
pub fn relPosEncoding(allocator: Allocator, t: usize, d_model: usize) ![]f32 {
    if (d_model == 0 or d_model % 2 != 0 or t == 0) return error.InvalidShape;
    const p_count = try std.math.sub(usize, try std.math.mul(usize, 2, t), 1);
    const half = d_model / 2;
    const out = try allocator.alloc(f32, try std.math.mul(usize, p_count, d_model));
    errdefer allocator.free(out);
    @memset(out, 0);

    const div = try allocator.alloc(f64, half);
    defer allocator.free(div);
    const factor = -(@log(@as(f64, 10000.0)) / @as(f64, @floatFromInt(d_model)));
    for (0..half) |i| div[i] = @exp(@as(f64, @floatFromInt(2 * i)) * factor);

    for (0..p_count) |p| {
        const pos: f64 = @floatFromInt(@as(i64, @intCast(t - 1)) - @as(i64, @intCast(p)));
        const row = out[p * d_model ..][0..d_model];
        for (0..half) |i| {
            const arg = pos * div[i];
            row[2 * i] = @floatCast(@sin(arg));
            row[2 * i + 1] = @floatCast(@cos(arg));
        }
    }
    return out;
}

pub fn attnName(buf: []u8, il: usize, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "encoder.layers.{d}.self_attn.{s}", .{ il, suffix }) catch unreachable;
}

/// Linear `x[m,in] @ W^T (+ bias)` using the packed/quantized `linearSeq` path:
/// the weight is loaded + packed ONCE by `ParakeetWeights` (cached by name) and
/// reused, instead of a per-forward `decodeF32` + transpose + f32 matmul. GGUF
/// `W` is ggml `[in,out]` (or `[1,in,out]` for the conv pointwise); `getLinear`
/// handles both. Returns `[m,out]` (caller owns). `w_name`/`b_name` must be
/// valid only for the duration of this call.
/// Tensor-valued linear + optional row bias: `[T,in] → [T,out]` over the
/// encoder's `.{._0,._1}` activations. Resolves the bias-by-name and forwards to
/// the shared `ParakeetWeights.linearD` (also used by subsampling).
pub fn linearWT(w: *ParakeetWeights, w_name: []const u8, b_name: ?[]const u8, x: *const fucina.Tensor(2)) !fucina.Tensor(2) {
    const bias: ?[]const f32 = if (b_name) |bn| (if (w.file.maybeGet(bn)) |bi| try f32Data(bi) else null) else null;
    return w.linearD(w_name, bias, x);
}

/// Transformer-XL relative-position MHSA for one encoder layer (non-causal, full
/// attention — the offline path; no pad/window mask when valid_len==T). `x` is
/// `[T, d_model]`, `pos_emb` is `[2T-1, d_model]` (from `relPosEncoding`).
/// Returns `[T, d_model]` (caller owns).
///
/// scores[qi,kj] = scale · ( (q[qi]+u)·k[kj] + (q[qi]+v)·p[kj-qi+T-1] )
/// The rel-shift `p[kj-qi+T-1]` is the closed form of parakeet.cpp's
/// pad/reshape/view skew (derived + unit-tested in the sibling tests).
pub fn relposAttention(
    ctx: *ExecContext,
    w: *ParakeetWeights,
    cfg: loader.Config,
    layer_idx: usize,
    x: *const fucina.Tensor(2),
    pos_emb: *const fucina.Tensor(2),
    pos_proj_all: ?*const fucina.Tensor(2),
    mask: ?[]const f32, // optional pre-softmax additive mask [t_len*t_len] (streaming); null = full attention
) !fucina.Tensor(2) {
    const file = w.file;
    const d_model = cfg.d_model;
    const h_count = cfg.n_heads;
    const dk = cfg.head_dim;

    const t_len = x.shape()[0];
    const p_count = try std.math.sub(usize, try std.math.mul(usize, 2, t_len), 1);
    if (x.shape()[1] != d_model) return error.ShapeMismatch;
    if (pos_emb.shape()[0] != p_count or pos_emb.shape()[1] != d_model) return error.ShapeMismatch;

    var q_wbuf: [128]u8 = undefined;
    var k_wbuf: [128]u8 = undefined;
    var v_wbuf: [128]u8 = undefined;
    var p_wbuf: [128]u8 = undefined;
    var q_bbuf: [128]u8 = undefined;
    var k_bbuf: [128]u8 = undefined;
    var v_bbuf: [128]u8 = undefined;
    const q_w_name = attnName(&q_wbuf, layer_idx, "linear_q.weight");
    const k_w_name = attnName(&k_wbuf, layer_idx, "linear_k.weight");
    const v_w_name = attnName(&v_wbuf, layer_idx, "linear_v.weight");
    const q_b_name = attnName(&q_bbuf, layer_idx, "linear_q.bias");
    const k_b_name = attnName(&k_bbuf, layer_idx, "linear_k.bias");
    const v_b_name = attnName(&v_bbuf, layer_idx, "linear_v.bias");
    const q_bias: ?[]const f32 = if (file.maybeGet(q_b_name)) |bi| try f32Data(bi) else null;
    const k_bias: ?[]const f32 = if (file.maybeGet(k_b_name)) |bi| try f32Data(bi) else null;
    const v_bias: ?[]const f32 = if (file.maybeGet(v_b_name)) |bi| try f32Data(bi) else null;

    var qkv = try w.linearQkvD(layer_idx, q_w_name, k_w_name, v_w_name, q_bias, k_bias, v_bias, x);
    defer if (qkv) |*t| t.deinit();
    var q: ?fucina.Tensor(2) = null;
    defer if (q) |*t| t.deinit();
    var k: ?fucina.Tensor(2) = null;
    defer if (k) |*t| t.deinit();
    var vproj: ?fucina.Tensor(2) = null;
    defer if (vproj) |*t| t.deinit();
    if (qkv == null) {
        q = try linearWT(w, q_w_name, q_b_name, x);
        k = try linearWT(w, k_w_name, k_b_name, x);
        vproj = try linearWT(w, v_w_name, v_b_name, x);
    }
    var p_owned: ?fucina.Tensor(2) = null;
    defer if (p_owned) |*t| t.deinit();
    if (pos_proj_all == null) {
        p_owned = try linearWT(w, attnName(&p_wbuf, layer_idx, "linear_pos.weight"), null, pos_emb);
    }

    const qkv_d: ?[]const f32 = if (qkv) |*t| try t.dataConst() else null;
    const qd: ?[]const f32 = if (q) |*t| try t.dataConst() else null;
    const kd: ?[]const f32 = if (k) |*t| try t.dataConst() else null;
    const vd: ?[]const f32 = if (vproj) |*t| try t.dataConst() else null;
    const qkv_width = try std.math.mul(usize, 3, d_model);
    if (qkv) |*t| {
        if (t.shape()[0] != t_len or t.shape()[1] != qkv_width) return error.InvalidWeightShape;
    } else {
        if (q.?.shape()[0] != t_len or q.?.shape()[1] != d_model) return error.InvalidWeightShape;
        if (k.?.shape()[0] != t_len or k.?.shape()[1] != d_model) return error.InvalidWeightShape;
        if (vproj.?.shape()[0] != t_len or vproj.?.shape()[1] != d_model) return error.InvalidWeightShape;
    }
    const pd_all: ?[]const f32 = if (pos_proj_all) |pa| blk: {
        if (pa.shape()[0] != cfg.n_layers * p_count or pa.shape()[1] != d_model) {
            return error.InvalidWeightShape;
        }
        break :blk try pa.dataConst();
    } else null;
    const pd_single: ?[]const f32 = if (p_owned) |*p| try p.dataConst() else null;
    if (p_owned) |*p| {
        if (p.shape()[0] != p_count or p.shape()[1] != d_model) return error.InvalidWeightShape;
    }

    var wbuf: [128]u8 = undefined;
    var bbuf: [128]u8 = undefined;
    const bu = try f32Data(try file.get(attnName(&wbuf, layer_idx, "pos_bias_u"))); // u[h*dk + d]
    const bv = try f32Data(try file.get(attnName(&bbuf, layer_idx, "pos_bias_v")));

    // The attention core is batched matmul + a vectorized softmax. Build
    // per-head [H,T,dk] tensors (fold u/v into q), then: AC = (q+u)·kᵀ via
    // matmul .trans_b; BD_raw = (q+v)·pᵀ likewise; the Transformer-XL rel-shift
    // `p[kj-qi+T-1]` becomes a light O(H·T²) skew of BD_raw into scores;
    // softmaxExtAxisRank (scale = 1/√dk, axis = kj) → attn; context = attn·V
    // via matmul .plain; merge heads → [T,d_model]. The heavy O(H·T²·dk) work runs in the
    // threaded SIMD GEMM; only the index remaps stay scalar.
    // Per-head [H,T,dk] scratch as public facade Tensors straight from the
    // BufferPool (write into the pooled tensor's data — no raw alloc +
    // fromSliceRank copy): `try .data()` gives the mutable buffer for the local
    // Q/K/V/pos packing fills below (allowed glue), and the GEMMs call facade
    // `.matmul` (.plain/.trans_b) directly.
    var qut = try fucina.Tensor(3).empty(ctx, .{ h_count, t_len, dk });
    defer qut.deinit();
    var qvt = try fucina.Tensor(3).empty(ctx, .{ h_count, t_len, dk });
    defer qvt.deinit();
    var kht = try fucina.Tensor(3).empty(ctx, .{ h_count, t_len, dk });
    defer kht.deinit();
    var vht = try fucina.Tensor(3).empty(ctx, .{ h_count, t_len, dk });
    defer vht.deinit();
    var pht = try fucina.Tensor(3).empty(ctx, .{ h_count, p_count, dk });
    defer pht.deinit();
    const qu = try qut.data();
    const qv = try qvt.data();
    const kh = try kht.data();
    const vh = try vht.data();
    const ph = try pht.data();
    for (0..h_count) |h| {
        const hoff = h * dk;
        for (0..t_len) |i| {
            for (0..dk) |d| {
                const base = (h * t_len + i) * dk + d;
                const qval = if (qkv_d) |buf| buf[i * 3 * d_model + hoff + d] else qd.?[i * d_model + hoff + d];
                qu[base] = qval + bu[hoff + d];
                qv[base] = qval + bv[hoff + d];
                kh[base] = if (qkv_d) |buf| buf[i * 3 * d_model + d_model + hoff + d] else kd.?[i * d_model + hoff + d];
                vh[base] = if (qkv_d) |buf| buf[i * 3 * d_model + 2 * d_model + hoff + d] else vd.?[i * d_model + hoff + d];
            }
        }
        for (0..p_count) |m| {
            for (0..dk) |d| {
                ph[(h * p_count + m) * dk + d] = if (pd_all) |buf|
                    buf[(layer_idx * p_count + m) * d_model + hoff + d]
                else
                    pd_single.?[m * d_model + hoff + d];
            }
        }
    }

    var ac = try qut.matmul(ctx, &kht, .trans_b, 3); // [H,T,T]: (q+u)·kᵀ
    defer ac.deinit();
    var bdraw = try qvt.matmul(ctx, &pht, .trans_b, 3); // [H,T,2T-1]: (q+v)·pᵀ
    defer bdraw.deinit();

    const acd = try ac.dataConst();
    // Transformer-XL skew via the public relposShift op: bd[H,T,2T-1] ->
    // bd_shifted[H,T,T] with bd_shifted[h,qi,kj] = bdraw[h,qi, kj+(T-1)-qi] — the
    // closed-form remap as a single op.
    var bd_shifted = try bdraw.relposShift(ctx, t_len, 3);
    defer bd_shifted.deinit();
    const bsd = try bd_shifted.dataConst();
    var scorest = try fucina.Tensor(3).empty(ctx, .{ h_count, t_len, t_len });
    defer scorest.deinit();
    const scores = try scorest.data();
    for (0..h_count) |h| {
        for (0..t_len) |qi| {
            const ac_row = (h * t_len + qi) * t_len;
            for (0..t_len) |kj| {
                var s = acd[ac_row + kj] + bsd[ac_row + kj];
                if (mask) |m| s += m[qi * t_len + kj]; // additive mask (-inf masks; broadcast over heads)
                scores[ac_row + kj] = s;
            }
        }
    }

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dk)));
    var attn = try scorest.softmax(ctx, ._2, .{ .scale = scale }); // softmax over kj
    defer attn.deinit();
    var ctxt = try attn.matmul(ctx, &vht, .plain, 3); // [H,T,dk]: attn·V
    defer ctxt.deinit();

    const ctxd = try ctxt.dataConst();
    var ctx_t = try fucina.Tensor(2).empty(ctx, .{ t_len, d_model });
    defer ctx_t.deinit();
    const out_ctx = try ctx_t.data();
    for (0..h_count) |h| {
        const hoff = h * dk;
        for (0..t_len) |qi| {
            for (0..dk) |d| out_ctx[qi * d_model + hoff + d] = ctxd[(h * t_len + qi) * dk + d];
        }
    }

    return linearWT(w, attnName(&wbuf, layer_idx, "linear_out.weight"), attnName(&bbuf, layer_idx, "linear_out.bias"), &ctx_t);
}

pub fn convName(buf: []u8, il: usize, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "encoder.layers.{d}.conv.{s}", .{ il, suffix }) catch unreachable;
}

/// NeMo `ConformerConvolution` (the part AFTER norm_conv), non-causal/offline:
///   c[T,D] → pointwise1(d→2d) → GLU → depthwise(k,groups=d,sym pad) + bias →
///   BatchNorm1d(inference, folded) → SiLU → pointwise2(d→d) → [T,D].
/// Matches parakeet.cpp `conformer.cpp::build_conv_module` (bn_eps=1e-5).
/// No pad mask (offline full sequence; valid_len==T). Caller owns the result.
pub fn convModule(
    ctx: *ExecContext,
    w: *ParakeetWeights,
    cfg: loader.Config,
    layer_idx: usize,
    c: *const fucina.Tensor(2),
) !fucina.Tensor(2) {
    const file = w.file;
    const d = cfg.d_model;
    const kk = cfg.conv_kernel;
    if (kk == 0) return error.InvalidShape;
    const pad = (kk - 1) / 2; // symmetric (offline)

    const t_len = c.shape()[0];
    if (c.shape()[1] != d) return error.ShapeMismatch;

    var wbuf: [128]u8 = undefined;
    var bbuf: [128]u8 = undefined;

    // pointwise_conv1: d -> 2d (ggml weight [1, d, 2d]).
    var pw1 = try linearWT(w, convName(&wbuf, layer_idx, "pointwise_conv1.weight"), convName(&bbuf, layer_idx, "pointwise_conv1.bias"), c);
    defer pw1.deinit();

    // GLU over channels (facade splitGated .glu): a=first d, b=next d → a·sigmoid(b).
    var glu2 = try pw1.splitGated(ctx, .glu, ._1, ._1); // [T, d]
    defer glu2.deinit();
    // Structural view [T,d] -> [T,1,d] for the depthwise conv input.
    var glu_t = try glu2.viewWithStrides(ctx, 3, .{ t_len, 1, d }, .{ d, d, 1 });
    defer glu_t.deinit();

    // Depthwise conv (k, groups=d, sym pad) with BatchNorm1d folded into the conv
    // weight/bias — bn = conv_out·scale + shift = Σ glu·(w·scale) + (dwb·scale+shift)
    // — run via the threaded public conv2d, then SiLU.
    const dww = try f32Data(try file.get(convName(&wbuf, layer_idx, "depthwise_conv.weight"))); // w[c*K + k]
    // Depthwise bias is optional (e.g. parakeet-tdt-0.6b-v3 has none — clone_weight_opt);
    // absent → folded as 0 (only the BN shift remains).
    const dwb: ?[]const f32 = if (file.maybeGet(convName(&bbuf, layer_idx, "depthwise_conv.bias"))) |bi| try f32Data(bi) else null;
    const bn_g = try f32Data(try file.get(convName(&wbuf, layer_idx, "batch_norm.weight")));
    const bn_b = try f32Data(try file.get(convName(&bbuf, layer_idx, "batch_norm.bias")));
    var mbuf: [128]u8 = undefined;
    var vbuf: [128]u8 = undefined;
    const bn_m = try f32Data(try file.get(convName(&mbuf, layer_idx, "batch_norm.running_mean")));
    const bn_v = try f32Data(try file.get(convName(&vbuf, layer_idx, "batch_norm.running_var")));

    var dw_wt = try fucina.Tensor(4).empty(ctx, .{ d, kk, 1, 1 }); // conv2d weight [Cout=d, KH=kk, KW=1, 1]
    defer dw_wt.deinit();
    var dw_bt = try fucina.Tensor(1).empty(ctx, .{d}); // folded bias [Cout=d]
    defer dw_bt.deinit();
    const dw_w = try dw_wt.data();
    const dw_b = try dw_bt.data();
    const bn_eps: f64 = 1e-5;
    for (0..d) |cc| {
        const scale: f64 = @as(f64, bn_g[cc]) / @sqrt(@as(f64, bn_v[cc]) + bn_eps);
        for (0..kk) |k| dw_w[cc * kk + k] = @floatCast(@as(f64, dww[cc * kk + k]) * scale);
        const dwb_c: f64 = if (dwb) |b| @as(f64, b[cc]) else 0;
        dw_b[cc] = @floatCast(dwb_c * scale + (@as(f64, bn_b[cc]) - @as(f64, bn_m[cc]) * scale));
    }
    var bn_t = try glu_t.conv2d(ctx, dw_wt, dw_bt, .{ 1, 1 }, .{ pad, 0 }, d, 3); // [T,1,d] facade conv2d
    defer bn_t.deinit();
    var s_t = try bn_t.unary(ctx, .silu); // SiLU (facade unary)
    defer s_t.deinit();

    // pointwise_conv2: d -> d. conv2d out [T,1,d] is viewed as [T,d].
    var s2 = try s_t.viewWithStrides(ctx, 2, .{ t_len, d }, .{ d, 1 });
    defer s2.deinit();
    return linearWT(w, convName(&wbuf, layer_idx, "pointwise_conv2.weight"), convName(&bbuf, layer_idx, "pointwise_conv2.bias"), &s2);
}

// === Macaron block + layer stack ===

fn fullName(buf: []u8, il: usize, mid: []const u8, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "encoder.layers.{d}.{s}.{s}", .{ il, mid, suffix }) catch unreachable;
}

/// LayerNorm over the channel dim (d) per frame: biased variance, eps 1e-5,
/// affine (g,b). Input/output `[T, d]` (row-major).
fn layerNormRaw(ctx: *ExecContext, in: []const f32, t_len: usize, d: usize, g: []const f32, b: []const f32) !fucina.internal.RawTensor {
    // KEPT RAW (numerics exception): the slice-based row kernel
    // `layerNormAffineRows` and the public facade `layerNorm` affine arm
    // (the `layerNormAffineAxisRank` kernel) are DIFFERENT kernels — routing
    // through the facade shifted the `--compare encoder` cosine (0.99999766 →
    // 0.99999782), so per the no-parity-drift rule this stays on the row kernel.
    return ctx.layerNormAffineRows(in, t_len, d, g, b, 1e-5);
}

pub fn layerNorm(ctx: *ExecContext, in: []const f32, t_len: usize, d: usize, g: []const f32, b: []const f32) !Tensor2 {
    var raw = try layerNormRaw(ctx, in, t_len, d, g, b);
    errdefer raw.deinit();
    return Tensor2.constant(ctx, raw);
}

pub fn layerNormByName(ctx: *ExecContext, w: *ParakeetWeights, il: usize, norm: []const u8, in: []const f32, t_len: usize, d: usize) !Tensor2 {
    var wb: [160]u8 = undefined;
    var bb: [160]u8 = undefined;
    const g = try f32Data(try w.file.get(fullName(&wb, il, norm, "weight")));
    const b = try f32Data(try w.file.get(fullName(&bb, il, norm, "bias")));
    return layerNorm(ctx, in, t_len, d, g, b);
}

/// Compatibility spelling used by the Tensor-valued layer code.
pub fn layerNormByNameT(ctx: *ExecContext, w: *ParakeetWeights, il: usize, norm: []const u8, in: []const f32, t_len: usize, d: usize) !Tensor2 {
    return layerNormByName(ctx, w, il, norm, in, t_len, d);
}

/// ConformerFeedForward (Tensor-valued): linear1(d→ff) → SiLU → linear2(ff→d).
/// `x` is `[T,d]`. The fused bias+SiLU keeps the single-pass perf.
pub fn feedForwardT(w: *ParakeetWeights, il: usize, ff: []const u8, x: *const fucina.Tensor(2)) !fucina.Tensor(2) {
    var wb: [160]u8 = undefined;
    var bb: [160]u8 = undefined;
    var h = try linearWT(w, fullName(&wb, il, ff, "linear1.weight"), null, x); // [T, ff]
    defer h.deinit();
    // linear1 bias is optional (the streaming realtime_eou model has no FFN biases):
    // present -> fused bias+SiLU; absent -> plain SiLU.
    if (w.file.maybeGet(fullName(&bb, il, ff, "linear1.bias"))) |bi| {
        try h.addAxisVectorUnaryInPlace(w.ctx, .silu, try f32Data(bi), ._1);
        return linearWT(w, fullName(&wb, il, ff, "linear2.weight"), fullName(&bb, il, ff, "linear2.bias"), &h);
    }
    var act = try h.unary(w.ctx, .silu);
    defer act.deinit();
    return linearWT(w, fullName(&wb, il, ff, "linear2.weight"), fullName(&bb, il, ff, "linear2.bias"), &act);
}

/// One Conformer block (macaron): x + 0.5·FFN1(LN) ; +attn(LN) ; +conv(LN) ;
/// +0.5·FFN2(LN) ; out=LN. `x`/`pos_emb` are `[T,d]`/`[2T-1,d]`. Returns `[T,d]`.
pub fn conformerLayer(ctx: *ExecContext, w: *ParakeetWeights, cfg: loader.Config, il: usize, x: *const fucina.Tensor(2), pos_emb: *const fucina.Tensor(2), pos_proj_all: ?*const fucina.Tensor(2)) !fucina.Tensor(2) {
    const d = cfg.d_model;
    const t_len = x.shape()[0];

    // Pooled in-place residual accumulator. The macaron residual is updated
    // in place across the 4 stages via the facade `addScaledInPlace`. `r` aliases
    // r_t's buffer (local glue) so the parity-wall row LayerNorm reads the residual.
    var r_t = try fucina.Tensor(2).empty(ctx, .{ t_len, d });
    defer r_t.deinit();
    const r = try r_t.data();
    @memcpy(r, try x.dataConst());

    // Stage A: r += 0.5 * FFN1(norm_feed_forward1(r))
    {
        var n = try layerNormByNameT(ctx, w, il, "norm_feed_forward1", r, t_len, d);
        defer n.deinit();
        var f = try feedForwardT(w, il, "feed_forward1", &n);
        defer f.deinit();
        try r_t.addScaledInPlace(ctx, f, 0.5);
    }
    // Stage B: r += attn(norm_self_att(r))
    {
        var n = try layerNormByNameT(ctx, w, il, "norm_self_att", r, t_len, d);
        defer n.deinit();
        var a = try relposAttention(ctx, w, cfg, il, &n, pos_emb, pos_proj_all, null);
        defer a.deinit();
        try r_t.addScaledInPlace(ctx, a, 1);
    }
    // Stage C: r += conv(norm_conv(r))
    {
        var n = try layerNormByNameT(ctx, w, il, "norm_conv", r, t_len, d);
        defer n.deinit();
        var c = try convModule(ctx, w, cfg, il, &n);
        defer c.deinit();
        try r_t.addScaledInPlace(ctx, c, 1);
    }
    // Stage D: r += 0.5 * FFN2(norm_feed_forward2(r))
    {
        var n = try layerNormByNameT(ctx, w, il, "norm_feed_forward2", r, t_len, d);
        defer n.deinit();
        var f = try feedForwardT(w, il, "feed_forward2", &n);
        defer f.deinit();
        try r_t.addScaledInPlace(ctx, f, 0.5);
    }
    // out = norm_out(r)
    return layerNormByNameT(ctx, w, il, "norm_out", r, t_len, d);
}

/// Full FastConformer encoder (offline, non-causal full attention): mel
/// `[n_mels, T]` (feat-major) → subsampling → N Conformer blocks → `[T', d_model]`
/// (row-major). xscaling is off for the 110m. Caller owns the result.
pub fn encode(ctx: *ExecContext, file: *const gguf.File, cfg: loader.Config, mel: []const f32, n_mels: usize, t_in: usize) !fucina.Tensor(2) {
    var w = ParakeetWeights.init(ctx, file);
    defer w.deinit();
    return encodeWithWeights(ctx, file, cfg, mel, n_mels, t_in, &w);
}

/// Full FastConformer encode → `[T', d_model]` as a public `Tensor` (the
/// encoder→decoder module boundary). Caller owns it.
pub fn encodeWithWeights(ctx: *ExecContext, file: *const gguf.File, cfg: loader.Config, mel: []const f32, n_mels: usize, t_in: usize, w: *ParakeetWeights) !fucina.Tensor(2) {
    const d = cfg.d_model;
    const alloc = ctx.allocator;

    var cur = try subsampling.subsampleWithWeights(ctx, file, cfg, mel, n_mels, t_in, w); // [T', d_model] facade
    errdefer cur.deinit();
    const tp = cur.shape()[0];

    const pe = try relPosEncoding(alloc, tp, d);
    var pet = try fucina.Tensor(2).fromSlice(ctx, .{ 2 * tp - 1, d }, pe);
    alloc.free(pe); // fromSlice copied it
    defer pet.deinit();

    var pos_proj_all = try w.linearPosAllD(cfg.n_layers, &pet);
    defer if (pos_proj_all) |*t| t.deinit();

    for (0..cfg.n_layers) |il| {
        const pos_proj_ptr: ?*const fucina.Tensor(2) = if (pos_proj_all) |*p| p else null;
        const next = try conformerLayer(ctx, w, cfg, il, &cur, &pet, pos_proj_ptr);
        cur.deinit();
        cur = next;
    }
    return cur;
}

test {
    _ = @import("encoder_tests.zig");
}
