//! Parakeet streaming inference: cache-aware building blocks for the
//! realtime_eou streaming model (arch=rnnt, chunked_limited attention, causal
//! depthwise conv). Matches the non-NeMo C++ reference `refs/parakeet.cpp`
//! `src/streaming_encoder.{hpp,cpp}` chunk-by-chunk.
//!
//! ConvCache — the conformer conv module's causal depthwise conv left-context cache
//! (NeMo `cache_last_time`). The causal depthwise conv1d itself is the SHARED op
//! `ExecContext.causalDepthwiseConv1dAxisRank` (its `state` is the `taps-1`
//! historical rows preceding the chunk). This module owns the per-layer cache
//! and its chunk-to-chunk advance, so a chunked stream is byte-equivalent to the
//! full-sequence causal conv (the cache-equivalence property, pinned by the
//! chunked-vs-full test in `streaming_tests.zig`).
const std = @import("std");
const fucina = @import("fucina");

const ExecContext = fucina.ExecContext;
const Tensor2 = fucina.Tensor(2);
const Allocator = std.mem.Allocator;

const encoder = @import("encoder.zig");
const subsampling = @import("subsampling.zig");
const decoder = @import("decoder.zig");
const loader = @import("loader.zig");
const ParakeetWeights = @import("weights.zig").ParakeetWeights;

/// Advance a `[rows, channels]` ring of historical frames by NeMo `update_cache`:
/// drop the oldest `n_new` rows and append `new` (`[n_new, channels]`), keeping
/// `rows` rows total. In place — the conceptual buffer is `[old(rows) ; new(n_new)]`
/// and the result is its last `rows` rows (handles `n_new` ≥ or < `rows`). The
/// shift reads index `n_new+j > j`, so increasing `j` never reads overwritten data.
fn advanceFrames(data: []f32, rows: usize, ch: usize, new: []const f32, n_new: usize) void {
    for (0..rows) |j| {
        const ci = n_new + j;
        if (ci < rows) {
            std.mem.copyForwards(f32, data[j * ch ..][0..ch], data[ci * ch ..][0..ch]);
        } else {
            @memcpy(data[j * ch ..][0..ch], new[(ci - rows) * ch ..][0..ch]);
        }
    }
}

/// Per-layer causal-depthwise-conv left-context cache (NeMo `cache_last_time`):
/// the `taps-1` frames preceding the current chunk, `[taps-1, channels]`
/// row-major. Zero-filled at stream start (and on a stream reset). Owned by the
/// caller (one per conformer layer).
pub const ConvCache = struct {
    data: []f32, // [(taps-1) * channels], row-major [row, channel]
    rows: usize, // taps - 1 (the causal left_pad)
    channels: usize,

    pub fn init(allocator: Allocator, channels: usize, taps: usize) !ConvCache {
        if (taps == 0) return error.InvalidShape;
        const rows = taps - 1;
        const data = try allocator.alloc(f32, try std.math.mul(usize, rows, channels));
        @memset(data, 0);
        return .{ .data = data, .rows = rows, .channels = channels };
    }

    pub fn deinit(self: *ConvCache, allocator: Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }

    /// Reset to zero (NeMo stream reset / new utterance).
    pub fn reset(self: *ConvCache) void {
        @memset(self.data, 0);
    }
};

/// One chunk of the streaming causal depthwise conv. Computes
/// `y[T,C] = causal_depthwise([cache ; x], kernel)` over the time axis (the conv
/// op reads `cache` as the `taps-1` historical rows), then advances `cache` to
/// the last `taps-1` rows of `[cache ; x]`. `x` is `[T, C]` (time-major,
/// contiguous), `kernel` is `[C, taps]` (the GGUF depthwise weight `w[c*taps+k]`).
/// Returns `[T, C]` (caller owns); mutates `cache`. The depthwise bias / norm /
/// activation are added by the conv-module assembly (`streamingConvModule`), not here.
pub fn streamingDepthwiseConv(
    ctx: *ExecContext,
    x: *const Tensor2,
    kernel: *const Tensor2,
    cache: *ConvCache,
) !Tensor2 {
    const xv = x.shape();
    const seq = xv[0];
    const ch = xv[1];
    if (ch != cache.channels) return error.ShapeMismatch;

    // Retag the raw [C, taps] kernel for the typed facade signature
    // (`causalDepthwiseConv1d` wants kernel tags .{ channel_tag, tap_tag }).
    var kernel_t = try kernel.withTags(ctx, .{ ._1, .tap });
    defer kernel_t.deinit();
    var out = try x.causalDepthwiseConv1d(ctx, ._0, ._1, .tap, &kernel_t, 1, cache.data);
    errdefer out.deinit();

    // Advance the cache to the last `taps-1` frames of [cache ; x].
    advanceFrames(cache.data, cache.rows, ch, try x.dataConst(), seq);
    return out;
}

/// Per-layer attention K/V left-context cache (NeMo `cache_last_channel`): the
/// last `cache_len` post-`norm_self_att` frames preceding the chunk, `[cache_len,
/// channels]` row-major (oldest first). `valid` is the number of filled frames
/// (NeMo `cache_last_channel_len`), growing `0, tc, 2tc, … , cache_len` — the
/// unfilled leading rows are masked out. Owned by the caller (one per layer).
pub const ChannelCache = struct {
    data: []f32, // [cache_len * channels]
    cache_len: usize,
    channels: usize,
    valid: usize, // filled frames (cache_last_channel_len), <= cache_len

    pub fn init(allocator: Allocator, cache_len: usize, channels: usize) !ChannelCache {
        const data = try allocator.alloc(f32, try std.math.mul(usize, cache_len, channels));
        @memset(data, 0);
        return .{ .data = data, .cache_len = cache_len, .channels = channels, .valid = 0 };
    }

    pub fn deinit(self: *ChannelCache, allocator: Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }

    pub fn reset(self: *ChannelCache) void {
        @memset(self.data, 0);
        self.valid = 0;
    }

    /// Append `tc` new frames (`attn_in [tc, channels]`), dropping the oldest `tc`;
    /// grow `valid` toward `cache_len`.
    pub fn advance(self: *ChannelCache, attn_in: []const f32, tc: usize) void {
        advanceFrames(self.data, self.cache_len, self.channels, attn_in, tc);
        self.valid = @min(self.valid + tc, self.cache_len);
    }
};

/// Build the streaming attention additive mask `[Tc*Tk]` (`mask[qi*Tk + kj]`, 0
/// visible / -inf masked) for the `Tc` CHUNK query rows only (query `qi` is at the
/// absolute position `cache_len+qi`; keys `kj` span `[cache(cache_len) ;
/// chunk(Tc)]`, Tk = cache_len + Tc). Mirrors `refs/parakeet.cpp`
/// `streaming_encoder.cpp:195-222` exactly:
///   - cache cols `[0, cache_len - valid)` are unfilled → masked,
///   - chunked_limited: `chunk = att_right+1`, `left_chunks = att_left/chunk`,
///     visible iff `0 <= gq/chunk - kj/chunk <= left_chunks` (`gq = cache_len+qi`).
/// Caller frees.
pub fn streamingAttnMask(
    allocator: Allocator,
    tc: usize,
    tk: usize,
    cache_len: usize,
    valid: usize,
    att_left: i32,
    att_right: i32,
) ![]f32 {
    const mask = try allocator.alloc(f32, tc * tk);
    errdefer allocator.free(mask);
    const chunk: i64 = @as(i64, att_right) + 1;
    const left_chunks: i64 = if (chunk > 0) @divFloor(@as(i64, att_left), chunk) else 0;
    const empty_cache: i64 = @as(i64, @intCast(cache_len)) - @as(i64, @intCast(valid));
    const ninf = -std.math.inf(f32);
    for (0..tc) |qi| {
        const gq: i64 = @as(i64, @intCast(cache_len)) + @as(i64, @intCast(qi)); // absolute query pos
        for (0..tk) |kj| {
            var vis = @as(i64, @intCast(kj)) >= empty_cache;
            if (vis and chunk > 0) {
                const diff = @divFloor(gq, chunk) - @divFloor(@as(i64, @intCast(kj)), chunk);
                vis = diff >= 0 and diff <= left_chunks;
            }
            mask[qi * tk + kj] = if (vis) 0 else ninf;
        }
    }
    return mask;
}

/// One chunk of the streaming relpos MHSA (NeMo `RelPositionMultiHeadAttention`,
/// streaming). Computes Q over the CHUNK only (`Tc` rows) while K/V span `[cache ;
/// chunk]` (`Tk` rows) — avoiding the ~`Tk/Tc`× redundant query work of running the
/// full self-attention over the concatenation and slicing. `attn_in` is `[Tc,
/// d_model]` (post `norm_self_att`). Returns `[Tc, d_model]` (caller owns); mutates
/// `cache`. Matches `refs/parakeet.cpp/src/streaming_encoder.cpp::build_stream_layer`
/// Stage B; the Transformer-XL rel-shift index for chunk query `qi` (absolute pos
/// `cache_len+qi`) vs key `kj` is `kj - qi + (Tc-1)` (the offline `kj-qi+(T-1)`
/// specialized to the chunk-at-offset: `kj-(cache_len+qi)+(Tk-1) = kj-qi+(Tc-1)`).
pub fn streamingRelposAttention(
    ctx: *ExecContext,
    w: *ParakeetWeights,
    cfg: loader.Config,
    layer_idx: usize,
    attn_in: *const fucina.Tensor(2),
    pet: *const fucina.Tensor(2), // relPosEncoding(Tk) [2Tk-1, d], computed ONCE per chunk by the caller
    cache: *ChannelCache,
    att_left: i32,
    att_right: i32,
) !fucina.Tensor(2) {
    const file = w.file;
    const d_model = cfg.d_model;
    const h_count = cfg.n_heads;
    const dk = cfg.head_dim;
    const tc = attn_in.shape()[0];
    if (attn_in.shape()[1] != d_model or cache.channels != d_model) return error.ShapeMismatch;
    const cache_len = cache.cache_len;
    const tk = try std.math.add(usize, cache_len, tc);
    const p_count = try std.math.sub(usize, try std.math.mul(usize, 2, tk), 1);
    if (pet.shape()[0] != p_count or pet.shape()[1] != d_model) return error.ShapeMismatch;

    // K/V context = [cache ; attn_in]  [Tk, d_model]; Q = attn_in (chunk only).
    var kvt = try fucina.Tensor(2).empty(ctx, .{ tk, d_model });
    defer kvt.deinit();
    const kvt_host = try kvt.data();
    const cache_values = try std.math.mul(usize, cache_len, d_model);
    const chunk_values = try std.math.mul(usize, tc, d_model);
    @memcpy(kvt_host[0..cache_values], cache.data);
    @memcpy(kvt_host[cache_values..][0..chunk_values], try attn_in.dataConst());

    var wbuf: [128]u8 = undefined;
    var bbuf: [128]u8 = undefined;
    var q = try encoder.linearWT(w, encoder.attnName(&wbuf, layer_idx, "linear_q.weight"), encoder.attnName(&bbuf, layer_idx, "linear_q.bias"), attn_in); // [Tc, d]
    defer q.deinit();
    var k = try encoder.linearWT(w, encoder.attnName(&wbuf, layer_idx, "linear_k.weight"), encoder.attnName(&bbuf, layer_idx, "linear_k.bias"), &kvt); // [Tk, d]
    defer k.deinit();
    var vproj = try encoder.linearWT(w, encoder.attnName(&wbuf, layer_idx, "linear_v.weight"), encoder.attnName(&bbuf, layer_idx, "linear_v.bias"), &kvt); // [Tk, d]
    defer vproj.deinit();
    var p = try encoder.linearWT(w, encoder.attnName(&wbuf, layer_idx, "linear_pos.weight"), null, pet); // [P, d]
    defer p.deinit();

    const qd = try q.dataConst();
    const kd = try k.dataConst();
    const vd = try vproj.dataConst();
    const pd = try p.dataConst();
    const bu = try encoder.f32Data(try file.get(encoder.attnName(&wbuf, layer_idx, "pos_bias_u")));
    const bv = try encoder.f32Data(try file.get(encoder.attnName(&bbuf, layer_idx, "pos_bias_v")));

    // Per-head tensors: qu/qv [H,Tc,dk] (fold u/v); kh/vh [H,Tk,dk]; ph [H,P,dk].
    // Per-head scratch as facade Tensors: `try .data()` for the local Q/K/V/
    // pos packing fills; the GEMMs call `.matmul` (.trans_b/.plain) directly.
    var qut = try fucina.Tensor(3).empty(ctx, .{ h_count, tc, dk });
    defer qut.deinit();
    var qvt = try fucina.Tensor(3).empty(ctx, .{ h_count, tc, dk });
    defer qvt.deinit();
    var kht = try fucina.Tensor(3).empty(ctx, .{ h_count, tk, dk });
    defer kht.deinit();
    var vht = try fucina.Tensor(3).empty(ctx, .{ h_count, tk, dk });
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
        for (0..tc) |i| {
            for (0..dk) |dd| {
                const base = (h * tc + i) * dk + dd;
                const qval = qd[i * d_model + hoff + dd];
                qu[base] = qval + bu[hoff + dd];
                qv[base] = qval + bv[hoff + dd];
            }
        }
        for (0..tk) |i| {
            for (0..dk) |dd| {
                const base = (h * tk + i) * dk + dd;
                kh[base] = kd[i * d_model + hoff + dd];
                vh[base] = vd[i * d_model + hoff + dd];
            }
        }
        for (0..p_count) |m| {
            for (0..dk) |dd| ph[(h * p_count + m) * dk + dd] = pd[m * d_model + hoff + dd];
        }
    }

    var ac = try qut.matmul(ctx, &kht, .trans_b, 3); // [H,Tc,Tk]: (q+u)·kᵀ
    defer ac.deinit();
    var bdraw = try qvt.matmul(ctx, &pht, .trans_b, 3); // [H,Tc,P]: (q+v)·pᵀ
    defer bdraw.deinit();

    const mask = try streamingAttnMask(ctx.allocator, tc, tk, cache_len, cache.valid, att_left, att_right);
    defer ctx.allocator.free(mask);

    const acd = try ac.dataConst();
    // Transformer-XL skew via the public relposShift op: bd[H,Tc,P] ->
    // [H,Tc,Tk] with bd_shifted[h,qi,kj] = bdraw[h,qi, kj+(Tc-1)-qi].
    var bd_shifted = try bdraw.relposShift(ctx, tk, 3);
    defer bd_shifted.deinit();
    const bsd = try bd_shifted.dataConst();
    var scorest = try fucina.Tensor(3).empty(ctx, .{ h_count, tc, tk });
    defer scorest.deinit();
    const scores = try scorest.data();
    for (0..h_count) |h| {
        for (0..tc) |qi| {
            const ac_row = (h * tc + qi) * tk;
            for (0..tk) |kj| {
                scores[ac_row + kj] = acd[ac_row + kj] + bsd[ac_row + kj] + mask[qi * tk + kj];
            }
        }
    }

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dk)));
    var attn = try scorest.softmax(ctx, ._2, .{ .scale = scale }); // over kj
    defer attn.deinit();
    var ctxt = try attn.matmul(ctx, &vht, .plain, 3); // [H,Tc,dk]: attn·V
    defer ctxt.deinit();

    const ctxd = try ctxt.dataConst();
    var ctx_t = try fucina.Tensor(2).empty(ctx, .{ tc, d_model });
    defer ctx_t.deinit();
    const out_ctx = try ctx_t.data();
    for (0..h_count) |h| {
        const hoff = h * dk;
        for (0..tc) |qi| {
            for (0..dk) |dd| out_ctx[qi * d_model + hoff + dd] = ctxd[(h * tc + qi) * dk + dd];
        }
    }

    var out = try encoder.linearWT(w, encoder.attnName(&wbuf, layer_idx, "linear_out.weight"), encoder.attnName(&bbuf, layer_idx, "linear_out.bias"), &ctx_t);
    errdefer out.deinit();
    cache.advance(try attn_in.dataConst(), tc);
    return out;
}

/// Streaming conformer conv module (NeMo `ConformerConvolution`, streaming): `c`
/// `[Tc, d]` → pointwise1(d→2d) → GLU → streaming causal depthwise (`ConvCache`) +
/// bias → **layer_norm** (per-frame over channels; g/b = `conv.batch_norm.{weight,
/// bias}`) → SiLU → pointwise2(d→d). The realtime model uses layer_norm, not the
/// offline folded batch_norm. Matches `streaming_encoder.cpp:258-329`.
fn streamingConvModule(
    ctx: *ExecContext,
    w: *ParakeetWeights,
    cfg: loader.Config,
    il: usize,
    c: *const fucina.Tensor(2),
    conv_cache: *ConvCache,
) !fucina.Tensor(2) {
    const d = cfg.d_model;
    const kk = cfg.conv_kernel;
    var wb: [128]u8 = undefined;
    var bb: [128]u8 = undefined;

    var pw1 = try encoder.linearWT(w, encoder.convName(&wb, il, "pointwise_conv1.weight"), encoder.convName(&bb, il, "pointwise_conv1.bias"), c); // [tc, 2d]
    defer pw1.deinit();

    // GLU a·sigmoid(b) over the 2d→d split via the facade `splitGated`.
    var glu_t = try pw1.splitGated(ctx, .glu, ._1, ._1); // [tc, d]
    defer glu_t.deinit();

    const dww = try encoder.f32Data(try w.file.get(encoder.convName(&wb, il, "depthwise_conv.weight"))); // [d*K], w[c*K+k]
    var dwk = try fucina.Tensor(2).fromSlice(ctx, .{ d, kk }, dww);
    defer dwk.deinit();
    var dw = try streamingDepthwiseConv(ctx, &glu_t, &dwk, conv_cache); // [tc, d]
    defer dw.deinit();
    // Depthwise bias is optional (the realtime_eou model has none — clone_weight_opt).
    if (w.file.maybeGet(encoder.convName(&bb, il, "depthwise_conv.bias"))) |bi|
        try dw.addAxisVectorInPlace(ctx, try encoder.f32Data(bi), ._1);

    // layer_norm over channels per frame (g/b stored under the batch_norm names) via
    // the facade `layerNorm` affine arm (the layerNormAffineAxisRank kernel) — NOT the
    // offline encoder's layerNormAffineRows row kernel: the two differ numerically
    // (see `encoder.zig` layerNormRaw); streaming parity was validated on this one.
    var gt = try fucina.Tensor(1).fromSlice(ctx, .{d}, try encoder.f32Data(try w.file.get(encoder.convName(&wb, il, "batch_norm.weight"))));
    defer gt.deinit();
    var bt = try fucina.Tensor(1).fromSlice(ctx, .{d}, try encoder.f32Data(try w.file.get(encoder.convName(&bb, il, "batch_norm.bias"))));
    defer bt.deinit();
    var normed = try dw.layerNorm(ctx, ._1, 1e-5, .{ .weight = gt, .bias = bt });
    defer normed.deinit();

    var s = try normed.unary(ctx, .silu);
    defer s.deinit();
    return encoder.linearWT(w, encoder.convName(&wb, il, "pointwise_conv2.weight"), encoder.convName(&bb, il, "pointwise_conv2.bias"), &s);
}

/// One streaming conformer block (macaron): `x + 0.5·FFN1(LN)` → `+streamingAttn(LN)`
/// → `+streamingConv(LN)` → `+0.5·FFN2(LN)` → `LN_out`. Carries both per-layer
/// caches. `x` is `[Tc, d]` (a post-subsampling chunk). Returns `[Tc, d]`.
pub fn streamingConformerLayer(
    ctx: *ExecContext,
    w: *ParakeetWeights,
    cfg: loader.Config,
    il: usize,
    x: *const Tensor2,
    pet: *const Tensor2, // shared relPosEncoding(Tk) for this chunk
    ch_cache: *ChannelCache,
    conv_cache: *ConvCache,
    att_left: i32,
    att_right: i32,
) !Tensor2 {
    const d = cfg.d_model;
    const tc = x.shape()[0];

    var r_t = try fucina.Tensor(2).empty(ctx, .{ tc, d });
    defer r_t.deinit();
    const r = try r_t.data();
    @memcpy(r, try x.dataConst());

    {
        var n = try encoder.layerNormByNameT(ctx, w, il, "norm_feed_forward1", r, tc, d);
        defer n.deinit();
        var f = try encoder.feedForwardT(w, il, "feed_forward1", &n);
        defer f.deinit();
        const fd = try f.dataConst();
        for (0..tc * d) |i| r[i] += 0.5 * fd[i];
    }
    {
        var n = try encoder.layerNormByNameT(ctx, w, il, "norm_self_att", r, tc, d);
        defer n.deinit();
        var a = try streamingRelposAttention(ctx, w, cfg, il, &n, pet, ch_cache, att_left, att_right);
        defer a.deinit();
        const ad = try a.dataConst();
        for (0..tc * d) |i| r[i] += ad[i];
    }
    {
        var n = try encoder.layerNormByNameT(ctx, w, il, "norm_conv", r, tc, d);
        defer n.deinit();
        var cc = try streamingConvModule(ctx, w, cfg, il, &n, conv_cache);
        defer cc.deinit();
        const cd = try cc.dataConst();
        for (0..tc * d) |i| r[i] += cd[i];
    }
    {
        var n = try encoder.layerNormByNameT(ctx, w, il, "norm_feed_forward2", r, tc, d);
        defer n.deinit();
        var f = try encoder.feedForwardT(w, il, "feed_forward2", &n);
        defer f.deinit();
        const fd = try f.dataConst();
        for (0..tc * d) |i| r[i] += 0.5 * fd[i];
    }
    return encoder.layerNormByName(ctx, w, il, "norm_out", r, tc, d);
}

/// Cache-aware streaming conformer encoder layer stack: owns the per-layer
/// conv (`cache_last_time`) + attention (`cache_last_channel`) caches and runs the
/// full cache-aware encoder on each mel chunk: causal subsampling → drop
/// `drop_extra_pre_encoded` leading frames (steps ≥ 1) → 17-layer stack → slice to
/// `valid_out_len`. State (both caches + the step counter) carries across chunks.
pub const StreamingEncoder = struct {
    conv_caches: []ConvCache,
    chan_caches: []ChannelCache,
    n_layers: usize,
    att_left: i32,
    att_right: i32,
    drop_extra: usize,
    valid_out_len: usize,
    step_count: usize = 0,
    allocator: Allocator,

    pub fn init(allocator: Allocator, cfg: loader.Config, sc: loader.StreamingConfig) !StreamingEncoder {
        const n = cfg.n_layers;
        const d = cfg.d_model;
        const conv = try allocator.alloc(ConvCache, n);
        errdefer allocator.free(conv);
        const chan = try allocator.alloc(ChannelCache, n);
        errdefer allocator.free(chan);
        const ccl: usize = @intCast(sc.last_channel_cache_size);
        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |j| {
                conv[j].deinit(allocator);
                chan[j].deinit(allocator);
            }
        }
        for (0..n) |i| {
            conv[i] = try ConvCache.init(allocator, d, cfg.conv_kernel);
            errdefer conv[i].deinit(allocator);
            chan[i] = try ChannelCache.init(allocator, ccl, d);
            initialized = i + 1;
        }
        return .{
            .conv_caches = conv,
            .chan_caches = chan,
            .n_layers = n,
            .att_left = sc.att_context_left,
            .att_right = sc.att_context_right,
            .drop_extra = @intCast(@max(0, sc.drop_extra_pre_encoded)),
            .valid_out_len = @intCast(@max(0, sc.valid_out_len)),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *StreamingEncoder) void {
        for (0..self.n_layers) |i| {
            self.conv_caches[i].deinit(self.allocator);
            self.chan_caches[i].deinit(self.allocator);
        }
        self.allocator.free(self.conv_caches);
        self.allocator.free(self.chan_caches);
        self.* = undefined;
    }

    pub fn reset(self: *StreamingEncoder) void {
        for (0..self.n_layers) |i| {
            self.conv_caches[i].reset();
            self.chan_caches[i].reset();
        }
        self.step_count = 0;
    }

    /// Full cache-aware encoder step on one mel chunk `[n_mels, n_mel_frames]`
    /// (feat-major, including the pre-encode overlap). Returns the valid encoder
    /// frames `[valid, d_model]` (caller owns); advances both caches + the step
    /// counter. Matches `streaming_encoder.cpp::step`. (`valid = valid_out_len`
    /// mid-stream, all `Tc` frames on the final chunk.)
    pub fn step(
        self: *StreamingEncoder,
        ctx: *ExecContext,
        file: *const @import("fucina").gguf.File,
        cfg: loader.Config,
        w: *ParakeetWeights,
        mel_chunk: []const f32,
        n_mels: usize,
        n_mel_frames: usize,
        is_last: bool,
    ) !Tensor2 {
        const d = cfg.d_model;
        var sub = try subsampling.streamingSubsample(ctx, file, cfg, w, mel_chunk, n_mels, n_mel_frames, n_mel_frames);
        defer sub.deinit();
        const tsub = sub.shape()[0];
        const drop = if (self.step_count != 0) self.drop_extra else 0;
        self.step_count += 1;
        const tc = if (tsub > drop) tsub - drop else 0;
        if (tc == 0) return fucina.Tensor(2).empty(ctx, .{ 0, d });

        const sub_data = try sub.dataConst();
        var x = try fucina.Tensor(2).fromSlice(ctx, .{ tc, d }, sub_data[drop * d ..][0 .. tc * d]);
        defer x.deinit();
        var y_full = try self.layerStack(ctx, w, cfg, &x); // [tc, d]

        const valid = if (is_last) tc else @min(self.valid_out_len, tc);
        if (valid == tc) return y_full;
        defer y_full.deinit();
        var out = try fucina.Tensor(2).empty(ctx, .{ valid, d });
        errdefer out.deinit();
        @memcpy(try out.data(), (try y_full.dataConst())[0 .. valid * d]);
        return out;
    }

    /// Run the layer stack on one post-subsampling chunk `[Tc, d_model]`, carrying
    /// both caches. Returns `[Tc, d_model]` (caller owns). The relpos sinusoidal
    /// table (`relPosEncoding(Tk)`, Tk = cache_len + Tc) is identical across all
    /// layers of a chunk, so it is computed ONCE here and shared (each layer still
    /// applies its own `linear_pos`) — avoids 16 redundant f64 sin/cos tables/chunk.
    pub fn layerStack(self: *StreamingEncoder, ctx: *ExecContext, w: *ParakeetWeights, cfg: loader.Config, x: *const Tensor2) !Tensor2 {
        const d = cfg.d_model;
        const tc = x.shape()[0];
        const tk = self.chan_caches[0].cache_len + tc;
        const pe = try encoder.relPosEncoding(ctx.allocator, tk, d); // [2Tk-1, d]
        defer ctx.allocator.free(pe);
        var pet = try fucina.Tensor(2).fromSlice(ctx, .{ 2 * tk - 1, d }, pe);
        defer pet.deinit();

        var cur = try x.detach(ctx);
        errdefer cur.deinit();
        for (0..self.n_layers) |il| {
            const next = try streamingConformerLayer(ctx, w, cfg, il, &cur, &pet, &self.chan_caches[il], &self.conv_caches[il], self.att_left, self.att_right);
            cur.deinit();
            cur = next;
        }
        return cur;
    }
};

/// NeMo prompt-conditioning (multilingual `nemotron-3.5-asr-streaming`): project
/// the encoder output through `prompt_kernel` with a per-utterance language
/// one-hot. `enc` `[Tc, d_model]` → `[enc | onehot(prompt_index)]` `[Tc,
/// d_model+num_prompts]` → `W2·ReLU(W0·x + b0) + b2` `[Tc, d_model]`. Matches
/// `refs/parakeet.cpp/src/prompt_kernel.cpp`. Caller owns the result.
pub fn applyPromptKernel(ctx: *ExecContext, w: *ParakeetWeights, cfg: loader.Config, enc: *const Tensor2, num_prompts: usize, prompt_index: i32) !Tensor2 {
    const d = cfg.d_model;
    const tc = enc.shape()[0];
    const in_dim = d + num_prompts;
    var x = try fucina.Tensor(2).empty(ctx, .{ tc, in_dim }); // [enc | onehot]
    defer x.deinit();
    const xd = try x.data();
    @memset(xd, 0);
    const ed = try enc.dataConst();
    const pi: usize = @intCast(prompt_index);
    for (0..tc) |t| {
        @memcpy(xd[t * in_dim ..][0..d], ed[t * d ..][0..d]);
        xd[t * in_dim + d + pi] = 1.0;
    }
    var h = try encoder.linearWT(w, "prompt_kernel.0.weight", "prompt_kernel.0.bias", &x); // [Tc, 2D]
    defer h.deinit();
    var hr = try h.relu(ctx);
    defer hr.deinit();
    return encoder.linearWT(w, "prompt_kernel.2.weight", "prompt_kernel.2.bias", &hr); // [Tc, D]
}

/// Cache-aware streaming RNN-T session: drives the `StreamingEncoder` +
/// the carried-state RNN-T greedy decoder chunk by chunk. Owns the encoder caches,
/// the predictor LSTM + joint, the carried `RnntDecodeState`, and the accumulated
/// non-special token output. `<EOU>`/`<EOB>` tokens are recorded as events
/// (excluded from the output) and reset the DECODER state for the next utterance
/// (decoder-only, matching `streaming.cpp`). Borrows `w` (must outlive the session).
pub const StreamingSession = struct {
    enc: StreamingEncoder,
    pred: decoder.Predictor,
    joint: decoder.Joint,
    state: decoder.RnntDecodeState,
    eou_id: i32, // -1 if the model has no <EOU>
    eob_id: i32,
    eou_events: usize = 0, // count of EOU/EOB events seen
    tokens: std.ArrayList(i32), // accumulated non-special token ids
    collect_meta: bool = false, // gather per-token TokenInfo (timestamps)
    token_meta: std.ArrayList(decoder.TokenInfo) = .empty, // aligned with `tokens` when collect_meta
    frames_consumed: usize = 0, // GLOBAL encoder frames decoded so far (timestamp base)
    chunk0: usize, // first chunk size (mel frames)
    chunk_main: usize, // subsequent chunk size
    pre_cache: usize, // pre-encode overlap prepended to chunks ≥ 1
    prompt_present: bool, // multilingual prompt conditioning (nemotron)
    prompt_index: i32, // resolved language one-hot index (-1 = no prompt)
    prompt_num: usize, // num_prompts (one-hot width)
    cfg: loader.Config,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        file: *const @import("fucina").gguf.File,
        cfg: loader.Config,
        sc: loader.StreamingConfig,
        w: *ParakeetWeights,
        lang: []const u8, // target locale ("auto"/"en"/…); ignored by non-prompt models
    ) !StreamingSession {
        var enc = try StreamingEncoder.init(allocator, cfg, sc);
        errdefer enc.deinit();
        var pred = try decoder.Predictor.init(allocator, w, cfg);
        errdefer pred.deinit();
        var joint = try decoder.Joint.init(allocator, w, cfg);
        errdefer joint.deinit();
        var dstate = try decoder.RnntDecodeState.init(allocator, cfg.pred_hidden, cfg.pred_rnn_layers);
        errdefer dstate.deinit(allocator);

        // Resolve the <EOU>/<EOB> special-token ids from the tokenizer pieces.
        var eou: i32 = -1;
        var eob: i32 = -1;
        const pieces = try loader.loadPieces(file, allocator);
        defer allocator.free(pieces);
        for (pieces, 0..) |p, i| {
            if (std.mem.eql(u8, p, "<EOU>")) eou = @intCast(i) else if (std.mem.eql(u8, p, "<EOB>")) eob = @intCast(i);
        }

        // Resolve the multilingual prompt conditioning (no-op for non-prompt models).
        var prompt_present = false;
        var prompt_index: i32 = -1;
        var prompt_num: usize = 0;
        if (loader.PromptConfig.fromGguf(file)) |pc| {
            prompt_present = true;
            prompt_num = @intCast(@max(0, pc.num_prompts));
            prompt_index = (try pc.resolveLang(file, allocator, lang)) orelse return error.UnknownLang;
        }

        return .{
            .enc = enc,
            .pred = pred,
            .joint = joint,
            .state = dstate,
            .eou_id = eou,
            .eob_id = eob,
            .tokens = .empty,
            .chunk0 = @intCast(@max(1, sc.chunk_size[0])),
            .chunk_main = @intCast(@max(1, sc.chunk_size[1])),
            .pre_cache = @intCast(@max(0, sc.pre_encode_cache_size[1])),
            .prompt_present = prompt_present,
            .prompt_index = prompt_index,
            .prompt_num = prompt_num,
            .cfg = cfg,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *StreamingSession) void {
        self.enc.deinit();
        self.pred.deinit();
        self.joint.deinit();
        self.state.deinit(self.allocator);
        self.tokens.deinit(self.allocator);
        self.token_meta.deinit(self.allocator);
        self.* = undefined;
    }

    fn isSpecial(self: *const StreamingSession, tok: i32) bool {
        return (self.eou_id >= 0 and tok == self.eou_id) or (self.eob_id >= 0 and tok == self.eob_id);
    }

    /// Decode one chunk's encoder frames `[n, d_model]` (RNN-T greedy, carrying
    /// state). Appends non-special tokens to `self.tokens`; on an `<EOU>`/`<EOB>`
    /// event resets the decoder state (SOS) for the next utterance.
    pub fn feedEncoderFrames(self: *StreamingSession, ctx: *ExecContext, enc_frames: *const Tensor2) !void {
        const n = enc_frames.shape()[0]; // this chunk's encoder frames
        var emitted: std.ArrayList(i32) = .empty;
        defer emitted.deinit(self.allocator);
        var meta: std.ArrayList(decoder.TokenInfo) = .empty;
        defer meta.deinit(self.allocator);
        // frame_base = GLOBAL frames decoded by prior chunks → token frames are global, not per-chunk t.
        try decoder.rnntDecodeFrames(ctx, self.cfg, &self.pred, &self.joint, enc_frames, &self.state, &emitted, self.allocator, if (self.collect_meta) &meta else null, self.frames_consumed);
        self.frames_consumed += n;
        var had_eou = false;
        for (emitted.items, 0..) |tok, i| {
            if (self.isSpecial(tok)) {
                had_eou = true;
                self.eou_events += 1;
            } else {
                try self.tokens.append(self.allocator, tok);
                if (self.collect_meta) try self.token_meta.append(self.allocator, meta.items[i]);
            }
        }
        if (had_eou) self.state.reset(); // decoder-only EOU reset (streaming.cpp)
    }

    /// Encoder.step + (if prompt-conditioned) the prompt_kernel projection — the
    /// per-chunk encoder output the decoder consumes, matching `streaming.cpp`
    /// (prompt applied between enc.step and the decode). Returns `[valid, d_model]`.
    pub fn encodeChunkPrompted(
        self: *StreamingSession,
        ctx: *ExecContext,
        file: *const @import("fucina").gguf.File,
        w: *ParakeetWeights,
        mel_chunk: []const f32,
        n_mels: usize,
        n_mel_frames: usize,
        is_last: bool,
    ) !Tensor2 {
        var enc = try self.enc.step(ctx, file, self.cfg, w, mel_chunk, n_mels, n_mel_frames, is_last);
        const tc = enc.shape()[0];
        if (!self.prompt_present or tc == 0) return enc; // ownership to caller
        defer enc.deinit();
        return applyPromptKernel(ctx, w, self.cfg, &enc, self.prompt_num, self.prompt_index);
    }

    /// Full per-chunk step: mel chunk `[n_mels, n_mel_frames]` → encoder.step (+
    /// prompt) → RNN-T decode. (The chunk-schedule mel windowing front-end is `feedMel`.)
    pub fn feedMelChunk(
        self: *StreamingSession,
        ctx: *ExecContext,
        file: *const @import("fucina").gguf.File,
        w: *ParakeetWeights,
        mel_chunk: []const f32,
        n_mels: usize,
        n_mel_frames: usize,
        is_last: bool,
    ) !void {
        var enc_frames = try self.encodeChunkPrompted(ctx, file, w, mel_chunk, n_mels, n_mel_frames, is_last);
        defer enc_frames.deinit();
        if (enc_frames.shape()[0] == 0) return;
        try self.feedEncoderFrames(ctx, &enc_frames);
    }

    /// Chunk-schedule driver: window the full clip mel `[n_mels, T]`
    /// (feat-major `mel[m*T + t]`) and feed each chunk through `feedMelChunk`,
    /// carrying all caches + decoder state. Chunk 0 = `mel[:, 0:chunk0]`; chunk i =
    /// `mel[:, buffer_idx-pre_cache : buffer_idx+chunk_main]` (the pre-encode
    /// overlap), advancing `buffer_idx` by the chunk size. Matches
    /// `streaming.cpp::run_stream_over_pcm`. Accumulates tokens in `self.tokens`.
    pub fn feedMel(
        self: *StreamingSession,
        ctx: *ExecContext,
        file: *const @import("fucina").gguf.File,
        w: *ParakeetWeights,
        mel: []const f32,
        n_mels: usize,
        t: usize,
    ) !void {
        var buffer_idx: usize = 0;
        var first = true;
        while (buffer_idx < t) {
            const chunk_size = if (first) self.chunk0 else self.chunk_main;
            const chunk_hi = @min(buffer_idx + chunk_size, t);
            if (chunk_hi <= buffer_idx) break;
            const lo = if (first) buffer_idx else (if (buffer_idx > self.pre_cache) buffer_idx - self.pre_cache else 0);
            const win_frames = chunk_hi - lo;
            const is_last = chunk_hi >= t;

            const win = try self.allocator.alloc(f32, n_mels * win_frames);
            defer self.allocator.free(win);
            for (0..n_mels) |m| {
                for (0..win_frames) |tt| win[m * win_frames + tt] = mel[m * t + (lo + tt)];
            }
            try self.feedMelChunk(ctx, file, w, win, n_mels, win_frames, is_last);

            buffer_idx += chunk_size; // shift_size == chunk_size
            first = false;
        }
    }
};

test {
    _ = @import("streaming_tests.zig");
}
