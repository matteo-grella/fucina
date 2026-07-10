//! Parakeet decoders: CTC head + greedy collapse; RNNT/TDT transducer decode.
//!
//! Matches parakeet.cpp `src/ctc_decoder.cpp` (the head) + `src/search.cpp`
//! (`ctc_greedy`): logits = enc·Wᵀ + bias (the `ctc_decoder.decoder_layers.0`
//! linear, always f32), then per-frame argmax (strict `>`, lowest-index ties)
//! and NeMo's fold_consecutive collapse. argmax is invariant to the log_softmax,
//! so the decode works on raw logits.
const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;
const loader = @import("loader.zig");
const pweights = @import("weights.zig");
const ParakeetWeights = pweights.ParakeetWeights;

const ExecContext = fucina.ExecContext;
const Tensor2 = fucina.Tensor(2);
const Allocator = std.mem.Allocator;

/// Per-emitted-token decode metadata, mirroring parakeet.cpp
/// `decode_types.hpp::TokenInfo` / NeMo timestamps=True + confidence='max_prob':
///   id    = emitted token id (always < blank; same as the id-only path).
///   frame = encoder time frame attributed to the token. TDT/RNNT = the frame `t`
///           at emission (streaming: the GLOBAL frame, offset by prior chunks);
///           CTC = the collapsed token's NeMo `start_offset` (the run-start frame).
///           Time(s) = frame * frame_sec (hop * subsampling_factor / sample_rate).
///   conf  = NeMo rescaled max_prob in (0,1] (0 when no per-frame confs given).
///   span  = encoder frames for word-end timing: TDT = durations[d_k]; RNNT = 1;
///           CTC = 1 here (the timestamp path rewrites it to next_frame-frame,
///           matching the C++ transcribe_with_timestamps).
pub const TokenInfo = struct {
    id: i32,
    frame: i32,
    conf: f32 = 0,
    span: i32 = 1,
};

/// Optional per-token metadata sink threaded through the greedy decoders. When
/// `null`, the decoders take the exact id-only path (ids are unaffected).
pub const TokenMeta = ?*std.ArrayList(TokenInfo);

fn f32Data(info: *const gguf.TensorInfo) ![]const f32 {
    return pweights.borrowF32(info.data);
}

/// NeMo CTC fold_consecutive collapse: `previous=blank`; emit `p` iff
/// `(p != previous or previous == blank) and p != blank`; `previous=p` every
/// frame (incl. blanks). Caller owns the result.
pub fn ctcCollapse(allocator: Allocator, argmax_ids: []const i32, blank: i32) ![]i32 {
    return ctcCollapseWithMeta(allocator, argmax_ids, blank, null, null);
}

/// `ctcCollapse` + optional per-token metadata. `frame` = NeMo `start_offset`
/// (search.cpp:70-72): token 0 → max(0, peak0-1), token i → the PREVIOUS token's
/// emit frame (peak); `span = 1` (the timestamp path rewrites it to the run
/// length); `conf` = **run-min** of the per-frame `confs` over the token's
/// consecutive argmax run (search.cpp:67-88), when `confs` is supplied.
/// The id output is identical to the bare collapse.
pub fn ctcCollapseWithMeta(allocator: Allocator, argmax_ids: []const i32, blank: i32, meta: TokenMeta, confs: ?[]const f32) ![]i32 {
    var out: std.ArrayList(i32) = .empty;
    errdefer out.deinit(allocator);
    var previous: i32 = blank;
    var prev_peak: usize = 0; // emit frame of the previously emitted token
    var have_prev = false;
    var cur_run_min: f32 = 1.0; // running min per-frame conf over the current run
    for (argmax_ids, 0..) |p, t| {
        if ((p != previous or previous == blank) and p != blank) {
            try out.append(allocator, p);
            if (meta) |m| {
                if (m.items.len > 0) m.items[m.items.len - 1].conf = cur_run_min; // close prev token's run
                const frame: i32 = if (have_prev) @intCast(prev_peak) else @intCast(if (t > 0) t - 1 else 0);
                try m.append(allocator, .{ .id = p, .frame = frame, .conf = 0, .span = 1 });
                if (confs) |cf| cur_run_min = cf[t]; // first frame of the new run
                prev_peak = t;
                have_prev = true;
            }
        } else if (p == previous and p != blank) { // continuation of the current run
            if (confs) |cf| {
                if (cf[t] < cur_run_min) cur_run_min = cf[t];
            }
        }
        previous = p;
    }
    if (meta) |m| {
        if (m.items.len > 0) m.items[m.items.len - 1].conf = cur_run_min; // finalize last token
    }
    return out.toOwnedSlice(allocator);
}

/// CTC greedy decode: encoder output `[T, d_model]` (row-major) → token ids.
/// Computes the CTC head logits `[T, vocab+1]`, per-frame argmax, then collapse.
/// Caller owns the returned slice.
pub fn ctcDecode(ctx: *ExecContext, file: *const gguf.File, cfg: loader.Config, enc: *const Tensor2, allocator: Allocator, meta: TokenMeta) ![]i32 {
    const d = cfg.d_model;
    const v = std.math.add(usize, cfg.vocab_size, 1) catch return error.InvalidShape; // incl. blank
    const t_len = enc.shape()[0];
    if (enc.shape()[1] != d) return error.ShapeMismatch;

    // CTC head linear via the packed/quantized linearSeq path: weight
    // (ggml [1, d, V], f32) loaded + packed once instead of decodeF32+transpose.
    var w = ParakeetWeights.init(ctx, file);
    defer w.deinit();
    const bias = try f32Data(try file.get("ctc_decoder.decoder_layers.0.bias"));
    var logits = try w.linearD("ctc_decoder.decoder_layers.0.weight", bias, enc); // [T, V]
    defer logits.deinit();
    const ld = try logits.dataConst();

    // per-frame argmax (strict >, lowest-index ties).
    const am = try allocator.alloc(i32, t_len);
    defer allocator.free(am);
    for (0..t_len) |t| {
        const row = ld[t * v ..][0..v];
        var best: usize = 0;
        var best_val: f32 = row[0];
        for (1..v) |vv| {
            if (row[vv] > best_val) {
                best_val = row[vv];
                best = vv;
            }
        }
        am[t] = @intCast(best);
    }

    // Per-frame max_prob conf for the run-min aggregation (only when metadata is
    // requested). CTC head logits are raw here; maxProbConf softmaxes them → p_max
    // (== exp(log_softmax) the C++ uses). N = v = vocab+1.
    var confs: ?[]f32 = null;
    if (meta != null) {
        const cf = try allocator.alloc(f32, t_len);
        for (0..t_len) |t| cf[t] = maxProbConf(ld[t * v ..][0..v], @intCast(am[t]));
        confs = cf;
    }
    defer if (confs) |cf| allocator.free(cf);
    const ids = try ctcCollapseWithMeta(allocator, am, @intCast(cfg.blank_id), meta, confs);
    // NeMo CTC word end_offset = the NEXT collapsed token's start frame (model.cpp:352):
    // rewrite span = next_frame - frame so group_words' (frame+span) matches; the final
    // token keeps span == 1. Only when metadata is requested.
    if (meta) |m| {
        var i: usize = 0;
        while (i + 1 < m.items.len) : (i += 1) m.items[i].span = m.items[i + 1].frame - m.items[i].frame;
    }
    return ids;
}

// === RNNT/TDT predictor (LSTM) + joint network ===

fn dequant(allocator: Allocator, file: *const gguf.File, name: []const u8) ![]f32 {
    const wi = try file.get(name);
    var n: usize = 1;
    for (0..wi.n_dims) |i| n = try std.math.mul(usize, n, wi.dims[i]);
    const out = try allocator.alloc(f32, n);
    errdefer allocator.free(out);
    try gguf.decodeF32(wi.ggml_type, wi.data, out);
    return out;
}

/// 1-layer LSTM prediction network (`decoder.prediction.*`). Gate order i,f,g,o
/// (PyTorch nn.LSTM); `z=Wih·x+bih+Whh·h+bhh`; `c'=f·c+i·g`, `h'=o·tanh(c')`;
/// SOS → x=0, else x=embed[token]. The gate matmuls run through the public
/// `linearSeq` path (loaded + packed once via `ParakeetWeights`); only the
/// element-wise cell update is local glue. Matches parakeet.cpp `prediction.cpp`.
/// LSTM weight/bias tensor name for predictor layer `l`, e.g.
/// `decoder.prediction.dec_rnn.lstm.weight_hh_l1`.
fn lstmName(buf: []u8, field: []const u8, l: usize) []const u8 {
    return std.fmt.bufPrint(buf, "decoder.prediction.dec_rnn.lstm.{s}{d}", .{ field, l }) catch unreachable;
}

pub const Predictor = struct {
    pw: *ParakeetWeights,
    h: usize, // pred_hidden
    vp1: usize, // vocab + 1 (embedding rows)
    n_layers: usize, // pred_rnn_layers (1 for the English models, 2 for nemotron)
    bih: []f32, // [n_layers * 4H] (per-layer concatenated)
    bhh: []f32,
    embed: []f32, // ggml [H, vp1] → embed[tok*H + d]
    allocator: Allocator,

    pub fn init(allocator: Allocator, weights: *ParakeetWeights, cfg: loader.Config) !Predictor {
        const nl = cfg.pred_rnn_layers;
        const four_h = try std.math.mul(usize, 4, cfg.pred_hidden);
        const bias_len = try std.math.mul(usize, nl, four_h);
        const vp1 = std.math.add(usize, cfg.vocab_size, 1) catch return error.InvalidShape;
        const bih = try allocator.alloc(f32, bias_len);
        errdefer allocator.free(bih);
        const bhh = try allocator.alloc(f32, bias_len);
        errdefer allocator.free(bhh);
        var nb: [96]u8 = undefined;
        for (0..nl) |l| {
            const bi = try dequant(allocator, weights.file, lstmName(&nb, "bias_ih_l", l));
            defer allocator.free(bi);
            @memcpy(bih[l * four_h ..][0..four_h], bi);
        }
        for (0..nl) |l| {
            const bh = try dequant(allocator, weights.file, lstmName(&nb, "bias_hh_l", l));
            defer allocator.free(bh);
            @memcpy(bhh[l * four_h ..][0..four_h], bh);
        }
        const embed = try dequant(allocator, weights.file, "decoder.prediction.embed.weight");
        errdefer allocator.free(embed);
        return .{
            .pw = weights,
            .h = cfg.pred_hidden,
            .vp1 = vp1,
            .n_layers = nl,
            .bih = bih,
            .bhh = bhh,
            .embed = embed,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Predictor) void {
        self.allocator.free(self.bih);
        self.allocator.free(self.bhh);
        self.allocator.free(self.embed);
        self.* = undefined;
    }

    /// One step of the (stacked) LSTM predictor. `h_in`/`c_in`/`h_out`/`c_out` are
    /// `[n_layers * h]` (per-layer state); `g_out` is `[h]` (the last layer's
    /// hidden). Layer 0's input is `embed[token]` (or 0 at SOS); layer L>0's input
    /// is layer L-1's new hidden. The gate matmuls `W_hh·h_in` / `W_ih·x` go through
    /// `linearSeq`; the i,f,g,o activations + cell update are the element-wise LSTM
    /// recurrence. For `n_layers==1` this is byte-identical to the prior 1-layer step.
    pub fn step(self: *const Predictor, ctx: *ExecContext, token: i32, is_sos: bool, h_in: []const f32, c_in: []const f32, g_out: []f32, h_out: []f32, c_out: []f32) !void {
        const h = self.h;
        const four_h = try std.math.mul(usize, 4, h);
        const z = try self.allocator.alloc(f64, four_h); // gate pre-acts z[gate*h + o]
        defer self.allocator.free(z);
        var nb: [96]u8 = undefined;

        for (0..self.n_layers) |l| {
            const bih = self.bih[l * four_h ..][0..four_h];
            const bhh = self.bhh[l * four_h ..][0..four_h];
            for (0..four_h) |k| z[k] = @as(f64, bih[k]) + @as(f64, bhh[k]);

            var h_t = try fucina.Tensor(.{ .seq, .in }).fromSlice(ctx, .{ 1, h }, h_in[l * h ..][0..h]);
            defer h_t.deinit();
            var zhh = try self.pw.linear(lstmName(&nb, "weight_hh_l", l), &h_t); // [1, 4h]
            defer zhh.deinit();
            const zhd = try zhh.dataConst();
            for (0..four_h) |k| z[k] += zhd[k];

            // Input: layer 0 = embed[token] (skipped at SOS); layer L>0 = layer L-1's
            // new hidden (always present).
            const xrow: ?[]const f32 = if (l == 0)
                (if (is_sos) null else self.embed[@as(usize, @intCast(token)) * h ..][0..h])
            else
                h_out[(l - 1) * h ..][0..h];
            if (xrow) |xr| {
                var x_t = try fucina.Tensor(.{ .seq, .in }).fromSlice(ctx, .{ 1, h }, xr);
                defer x_t.deinit();
                var zih = try self.pw.linear(lstmName(&nb, "weight_ih_l", l), &x_t); // [1, 4h]
                defer zih.deinit();
                const zid = try zih.dataConst();
                for (0..four_h) |k| z[k] += zid[k];
            }

            for (0..h) |o| {
                const ig = sigmoid(z[0 * h + o]);
                const fg = sigmoid(z[1 * h + o]);
                const gg = std.math.tanh(z[2 * h + o]);
                const og = sigmoid(z[3 * h + o]);
                const cv = fg * @as(f64, c_in[l * h + o]) + ig * gg;
                const hv = og * std.math.tanh(cv);
                c_out[l * h + o] = @floatCast(cv);
                h_out[l * h + o] = @floatCast(hv);
            }
        }
        @memcpy(g_out, h_out[(self.n_layers - 1) * h ..][0..h]);
    }
};

inline fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Joint network (`joint.*`): enc_proj = enc.weight·enc + enc.bias [jh];
/// pred_proj = pred.weight·g + pred.bias [jh]; f = ReLU(enc_proj+pred_proj);
/// logits = joint_net.2.weight·f + joint_net.2.bias [V_plus]. enc/pred weights
/// may be f16 (dequant→f32); joint_net.2 is f32.
pub const Joint = struct {
    pw: *ParakeetWeights,
    e: usize, // enc_hidden (d_model)
    jh: usize, // joint_hidden
    p: usize, // pred_hidden
    vp: usize, // V_plus = vocab+1+num_durations
    enc_b: []f32, // [JH]
    pred_b: []f32,
    out_b: []f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, weights: *ParakeetWeights, cfg: loader.Config) !Joint {
        const vp = try cfg.checkedVPlus();
        const enc_b = try dequant(allocator, weights.file, "joint.enc.bias");
        errdefer allocator.free(enc_b);
        const pred_b = try dequant(allocator, weights.file, "joint.pred.bias");
        errdefer allocator.free(pred_b);
        const out_b = try dequant(allocator, weights.file, "joint.joint_net.2.bias");
        errdefer allocator.free(out_b);
        return .{
            .pw = weights,
            .e = cfg.d_model,
            .jh = cfg.joint_hidden,
            .p = cfg.pred_hidden,
            .vp = vp,
            .enc_b = enc_b,
            .pred_b = pred_b,
            .out_b = out_b,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Joint) void {
        self.allocator.free(self.enc_b);
        self.allocator.free(self.pred_b);
        self.allocator.free(self.out_b);
        self.* = undefined;
    }

    /// Batched enc projection over ALL frames: `enc` `[T,E]` → `[T,jh]` via one
    /// `linearSeq` GEMM (the optimized backend), then + bias. Caller owns it.
    pub fn encProjAll(self: *const Joint, enc: *const Tensor2) !Tensor2 {
        return self.pw.linearD("joint.enc.weight", self.enc_b, enc); // [T, jh]
    }

    /// Joint step: `enc_proj_t` `[jh]`, `g` `[p]` → `logits` `[vp]`. pred_proj and
    /// the output projection go through `linearSeq`; only the ReLU sum is local.
    pub fn step(self: *const Joint, ctx: *ExecContext, enc_proj_t: []const f32, g: []const f32, logits: []f32) !void {
        var g_t = try fucina.Tensor(.{ .seq, .in }).fromSlice(ctx, .{ 1, self.p }, g);
        defer g_t.deinit();
        var pp = try self.pw.linear("joint.pred.weight", &g_t); // [1, jh]
        defer pp.deinit();
        const ppd = try pp.dataConst();

        const f = try self.allocator.alloc(f32, self.jh);
        defer self.allocator.free(f);
        for (0..self.jh) |hh| {
            const sum = @as(f64, enc_proj_t[hh]) + @as(f64, ppd[hh]) + @as(f64, self.pred_b[hh]);
            f[hh] = if (sum > 0) @floatCast(sum) else 0; // ReLU
        }

        var f_t = try fucina.Tensor(.{ .seq, .in }).fromSlice(ctx, .{ 1, self.jh }, f);
        defer f_t.deinit();
        var lg = try self.pw.linear("joint.joint_net.2.weight", &f_t); // [1, vp]
        defer lg.deinit();
        const lgd = try lg.dataConst();
        for (0..self.vp) |v| logits[v] = @floatCast(@as(f64, lgd[v]) + @as(f64, self.out_b[v]));
    }
};

fn argmaxSlice(a: []const f32) usize {
    var best: usize = 0;
    var bv: f32 = a[0];
    for (1..a.len) |i| {
        if (a[i] > bv) { // strict >, lowest-index ties (matches decode_argmax)
            bv = a[i];
            best = i;
        }
    }
    return best;
}

/// NeMo rescaled `max_prob` confidence over `a[0..n)` at index `k`:
/// `conf = (N*p_max - 1)/(N - 1)`, `p_max = softmax(a)[k]`, `N = a.len`. Stable
/// softmax accumulated in f64 (matches parakeet.cpp `decode_common.hpp::
/// decode_max_prob_conf` to ≤1e-3). Model-glue (the NeMo formula), not a general
/// kernel — the inner softmax-of-one-index is a few lines, so it stays local.
fn maxProbConf(a: []const f32, k: usize) f32 {
    var mx: f32 = a[0];
    for (a[1..]) |x| {
        if (x > mx) mx = x;
    }
    var denom: f64 = 0;
    for (a) |x| denom += @exp(@as(f64, x) - @as(f64, mx));
    const p_max = @exp(@as(f64, a[k]) - @as(f64, mx)) / denom;
    const n: f64 = @floatFromInt(a.len);
    return @floatCast((n * p_max - 1.0) / (n - 1.0));
}

/// TDT/RNNT greedy transducer decode. Matches parakeet.cpp `tdt.cpp`:
/// per encoder frame, run the joint on the committed predictor state; argmax the
/// token slice `[0,vocab+1)` and the duration slice `[vocab+1,V_plus)`; emit the
/// token if non-blank (advance the predictor) and skip `durations[d_k]` frames;
/// `max_symbols` cap with a +1 frame guard. RNNT is the same with a single
/// duration of 1. Caller owns the returned ids.
pub fn tdtDecode(ctx: *ExecContext, file: *const gguf.File, cfg: loader.Config, enc: *const Tensor2, allocator: Allocator, meta: TokenMeta) ![]i32 {
    var weights = ParakeetWeights.init(ctx, file); // packed predictor/joint weights, loaded once
    defer weights.deinit();
    return tdtDecodeWithWeights(ctx, cfg, enc, allocator, &weights, meta);
}

pub fn tdtDecodeWithWeights(ctx: *ExecContext, cfg: loader.Config, enc: *const Tensor2, allocator: Allocator, weights: *ParakeetWeights, meta: TokenMeta) ![]i32 {
    const d = cfg.d_model;
    const t_total = enc.shape()[0];
    if (enc.shape()[1] != d) return error.ShapeMismatch;

    var pred = try Predictor.init(allocator, weights, cfg);
    defer pred.deinit();
    var joint = try Joint.init(allocator, weights, cfg);
    defer joint.deinit();

    const jh = joint.jh;
    const v_plus = try cfg.checkedVPlus();
    const num_dur = cfg.num_durations;
    const token_count = v_plus - num_dur; // vocab + 1 (incl. blank)
    const blank: i32 = @intCast(cfg.blank_id);
    const max_symbols = cfg.max_symbols;
    const durations = cfg.durationsSlice();
    if (num_dur == 0) return error.InvalidShape;

    // enc_proj[t] = joint.enc·enc[t] + bias, batched over all frames (one GEMM).
    var enc_proj_t = try joint.encProjAll(enc);
    defer enc_proj_t.deinit();
    const enc_proj = try enc_proj_t.dataConst();

    const h = cfg.pred_hidden;
    const hs = try std.math.mul(usize, cfg.pred_rnn_layers, h); // per-layer LSTM state
    const c_h = try allocator.alloc(f32, hs); // committed h (all layers)
    defer allocator.free(c_h);
    const c_c = try allocator.alloc(f32, hs); // committed c
    defer allocator.free(c_c);
    const o_h = try allocator.alloc(f32, hs);
    defer allocator.free(o_h);
    const o_c = try allocator.alloc(f32, hs);
    defer allocator.free(o_c);
    const g = try allocator.alloc(f32, h); // last-layer hidden (predictor output)
    defer allocator.free(g);
    @memset(c_h, 0);
    @memset(c_c, 0);
    const logits = try allocator.alloc(f32, v_plus);
    defer allocator.free(logits);

    var hyp: std.ArrayList(i32) = .empty;
    errdefer hyp.deinit(allocator);

    var last_token: i32 = -1;
    var emitted_any = false;
    var g_valid = false;

    var t: usize = 0;
    while (t < t_total) {
        var symbols_added: usize = 0;
        var need_loop = true;
        var skip: usize = 0;
        while (need_loop and symbols_added < max_symbols) {
            if (!g_valid) {
                const is_sos = !emitted_any;
                const last_label = if (emitted_any) last_token else blank;
                try pred.step(ctx, last_label, is_sos, c_h, c_c, g, o_h, o_c);
                g_valid = true;
            }
            try joint.step(ctx, enc_proj[t * jh ..][0..jh], g, logits);
            const k: i32 = @intCast(argmaxSlice(logits[0..token_count]));
            const d_k = argmaxSlice(logits[token_count..v_plus]);
            skip = @intCast(durations[d_k]); // durations >= 0

            if (k != blank) {
                try hyp.append(allocator, k);
                if (meta) |m| {
                    // conf over the TOKEN slice only (exclude duration logits), N=token_count.
                    const conf = maxProbConf(logits[0..token_count], @intCast(k));
                    try m.append(allocator, .{ .id = k, .frame = @intCast(t), .conf = conf, .span = @intCast(durations[d_k]) });
                }
                last_token = k;
                @memcpy(c_h, o_h);
                @memcpy(c_c, o_c);
                emitted_any = true;
                g_valid = false;
            }
            symbols_added += 1;
            t += skip;
            need_loop = (skip == 0);
        }
        if (symbols_added == max_symbols) t += 1; // progress guard (ref's skip=1 is vestigial)
    }

    return hyp.toOwnedSlice(allocator);
}

/// Carried RNN-T greedy decode state (NeMo `RnntDecodeState`): the predictor LSTM
/// hidden/cell + the last emitted token. Persists across streaming chunks; reset
/// to SOS on an end-of-utterance event. Caller owns.
pub const RnntDecodeState = struct {
    h: []f32, // committed LSTM hidden [n_layers * pred_hidden]
    c: []f32, // committed LSTM cell
    last_token: i32 = -1,
    have_token: bool = false,

    pub fn init(allocator: Allocator, pred_hidden: usize, n_layers: usize) !RnntDecodeState {
        const sz = try std.math.mul(usize, n_layers, pred_hidden);
        const h = try allocator.alloc(f32, sz);
        errdefer allocator.free(h);
        const c = try allocator.alloc(f32, sz);
        @memset(h, 0);
        @memset(c, 0);
        return .{ .h = h, .c = c };
    }
    pub fn deinit(self: *RnntDecodeState, allocator: Allocator) void {
        allocator.free(self.h);
        allocator.free(self.c);
        self.* = undefined;
    }
    pub fn reset(self: *RnntDecodeState) void {
        @memset(self.h, 0);
        @memset(self.c, 0);
        self.last_token = -1;
        self.have_token = false;
    }
};

/// Pure RNN-T greedy over `enc` `[T, d_model]` (no duration head — `num_durations
/// == 0`), carrying `st` across calls so a chunked stream is byte-identical to a
/// whole-utterance decode. Appends EVERY emitted token id (incl. <EOU>/<EOB>) to
/// `out`; the caller filters specials + drives the EOU reset. Reuses `Predictor`
/// + `Joint`. Mirrors `refs/parakeet.cpp/src/rnnt.cpp::rnnt_decode_frames`.
pub fn rnntDecodeFrames(
    ctx: *ExecContext,
    cfg: loader.Config,
    pred: *const Predictor,
    joint: *const Joint,
    enc: *const Tensor2,
    st: *RnntDecodeState,
    out: *std.ArrayList(i32),
    allocator: Allocator,
    meta: TokenMeta,
    frame_base: usize, // GLOBAL encoder-frame offset for this chunk (streaming); 0 offline
) !void {
    if (cfg.num_durations != 0) return error.InvalidShape; // pure RNN-T
    const d = cfg.d_model;
    const ev = enc.shape();
    const t_total = ev[0];
    if (ev[1] != d) return error.ShapeMismatch;

    const jh = joint.jh;
    const v_plus = try cfg.checkedVPlus(); // vocab + 1 (incl. blank), no durations
    const blank: i32 = @intCast(cfg.blank_id);
    const max_symbols = cfg.max_symbols;

    var enc_proj_t = try joint.encProjAll(enc); // [T, joint_hidden]
    defer enc_proj_t.deinit();
    const enc_proj = try enc_proj_t.dataConst();

    const h = cfg.pred_hidden;
    const hs = try std.math.mul(usize, cfg.pred_rnn_layers, h); // per-layer LSTM state
    const g = try allocator.alloc(f32, h);
    defer allocator.free(g);
    const o_h = try allocator.alloc(f32, hs);
    defer allocator.free(o_h);
    const o_c = try allocator.alloc(f32, hs);
    defer allocator.free(o_c);
    const logits = try allocator.alloc(f32, v_plus);
    defer allocator.free(logits);

    var g_valid = false;
    var t: usize = 0;
    while (t < t_total) : (t += 1) {
        var emitted: usize = 0;
        while (emitted < max_symbols) {
            if (!g_valid) {
                const is_sos = !st.have_token;
                const last_label = if (st.have_token) st.last_token else blank;
                try pred.step(ctx, last_label, is_sos, st.h, st.c, g, o_h, o_c);
                g_valid = true;
            }
            try joint.step(ctx, enc_proj[t * jh ..][0..jh], g, logits);
            const k: i32 = @intCast(argmaxSlice(logits[0..v_plus]));
            if (k == blank) break; // blank -> advance time
            try out.append(allocator, k);
            if (meta) |m| {
                const conf = maxProbConf(logits[0..v_plus], @intCast(k)); // N = V_plus = vocab+1
                try m.append(allocator, .{ .id = k, .frame = @intCast(frame_base + t), .conf = conf, .span = 1 });
            }
            st.last_token = k;
            @memcpy(st.h, o_h);
            @memcpy(st.c, o_c);
            st.have_token = true;
            g_valid = false; // committed state advanced
            emitted += 1;
        }
    }
}

test {
    _ = @import("decoder_tests.zig");
}
