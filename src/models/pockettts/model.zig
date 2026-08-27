//! Pocket TTS v2 (kyutai, CC-BY-4.0) — pure-Zig port of the streaming
//! text-to-speech model: a prefix-LM transformer that autoregressively emits
//! CONTINUOUS 32-dim latents (no codebooks), each sampled by a 1-step
//! flow-matching MLP head (MAR/LSD lineage), decoded by a VAE-style Mimi
//! (depthwise transposed upsample 12.5→200 Hz → 2-layer sliding-window
//! transformer → SEANet causal decoder) at 24 kHz, 1920 samples per frame.
//!
//! Reference: refs/pocket-tts (kyutai-labs/pocket-tts @ d108410d); parity
//! pinned per stage against refs/pocket-tts-dumps (see pocket_tests.zig).
//! Weights: one GGUF from refs/pocket_to_gguf.py, all f32, checkpoint tensor
//! names preserved, voices as `voice.<name>.cache.<layer>` KV prefixes
//! (roped K — loads directly into the cache).
//!
//! Hand-rolled scalar/SIMD in the aec.zig style: explicit streaming state,
//! no facade, f32 throughout (matching the reference dtype).

const std = @import("std");
const fucina = @import("fucina");
const weights = @import("fucina").weights;
const ptqtp_gguf = @import("fucina").ptqtp_gguf;

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;
const LinearWeight = weights.LinearWeight;
const Rows = fucina.Tensor(.{ .seq, .embed });
const VecEmbed = fucina.Tensor(.{.embed});
const Allocator = std.mem.Allocator;

pub const Error = error{
    MissingTensor,
    MissingMetadata,
    BadShape,
    UnknownVoice,
    KvOverflow,
};

pub const latent_dim = 32;
pub const frame_samples = 1920;
pub const sample_rate = 24000;
pub const default_eos_threshold: f32 = -4.0;

// --- small math ------------------------------------------------------------

fn dot(a: []const f32, b: []const f32) f32 {
    const V = @Vector(8, f32);
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    var i: usize = 0;
    while (i + 32 <= a.len) : (i += 32) {
        acc0 += @as(V, a[i..][0..8].*) * @as(V, b[i..][0..8].*);
        acc1 += @as(V, a[i + 8 ..][0..8].*) * @as(V, b[i + 8 ..][0..8].*);
        acc2 += @as(V, a[i + 16 ..][0..8].*) * @as(V, b[i + 16 ..][0..8].*);
        acc3 += @as(V, a[i + 24 ..][0..8].*) * @as(V, b[i + 24 ..][0..8].*);
    }
    while (i + 8 <= a.len) : (i += 8) {
        acc0 += @as(V, a[i..][0..8].*) * @as(V, b[i..][0..8].*);
    }
    var s: f32 = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

/// out[o] = bias[o] + w[o,:]·x — w row-major [out][in].
fn matvec(out: []f32, w: []const f32, x: []const f32, bias: ?[]const f32) void {
    const in = x.len;
    for (out, 0..) |*o, i| {
        const b: f32 = if (bias) |bb| bb[i] else 0;
        o.* = b + dot(w[i * in ..][0..in], x);
    }
}

/// Exact GELU: 0.5x(1+erf(x/√2)). erf via Abramowitz–Stegun 7.1.26 in f64
/// (|err| ≤ 1.5e-7 — inside the parity gates).
fn erf(x: f64) f64 {
    const sign: f64 = if (x < 0) -1 else 1;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * @exp(-ax * ax);
    return sign * y;
}

fn gelu(x: f32) f32 {
    return @floatCast(0.5 * @as(f64, x) * (1.0 + erf(@as(f64, x) / std.math.sqrt2)));
}

const silu = @import("../host_ops.zig").silu;

fn elu(x: f32) f32 {
    return if (x > 0) x else @exp(x) - 1.0;
}

/// torch nn.LayerNorm: biased variance, optional affine.
fn layerNorm(x: []f32, w: ?[]const f32, b: ?[]const f32, eps: f32) void {
    var mean: f64 = 0;
    for (x) |v| mean += v;
    mean /= @floatFromInt(x.len);
    var vs: f64 = 0;
    for (x) |v| vs += (v - mean) * (v - mean);
    vs /= @floatFromInt(x.len);
    const inv = 1.0 / @sqrt(vs + eps);
    for (x, 0..) |*v, i| {
        var y: f64 = (@as(f64, v.*) - mean) * inv;
        if (w) |ww| y = y * ww[i];
        if (b) |bb| y += bb[i];
        v.* = @floatCast(y);
    }
}

/// TimestepEmbedder RMSNorm: UNBIASED variance (mean-subtracted), x not
/// centered in the output — y = x·alpha·rsqrt(eps + var). eps 1e-5.
fn timestepRmsNorm(x: []f32, alpha: []const f32) void {
    var mean: f64 = 0;
    for (x) |v| mean += v;
    mean /= @floatFromInt(x.len);
    var vs: f64 = 0;
    for (x) |v| vs += (v - mean) * (v - mean);
    vs /= @floatFromInt(x.len - 1);
    const inv = 1.0 / @sqrt(1e-5 + vs);
    for (x, 0..) |*v, i| v.* = @floatCast(@as(f64, v.*) * alpha[i] * inv);
}

// --- weights ---------------------------------------------------------------

fn f32Data(file: *const gguf.File, name: []const u8) ![]const f32 {
    const t = file.get(name) catch return Error.MissingTensor;
    if (t.ggml_type != .f32) return Error.BadShape;
    if (!std.mem.isAligned(@intFromPtr(t.data.ptr), @alignOf(f32))) return Error.BadShape;
    return @as([*]const f32, @ptrCast(@alignCast(t.data.ptr)))[0 .. t.data.len / 4];
}

fn fmtName(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

/// A transformer linear served through the CORE tensor library
/// (`weights.LinearWeight`): any GGUF dtype, PTQTP plane pair-detection,
/// BLAS/quant kernel dispatch — the same serving path every facade-based
/// port uses. The rest of this module stays hand-rolled (streaming state),
/// but the matmuls — ~90% of the compute — go through the house path.
const Linear = struct {
    w: LinearWeight,
    out_dim: usize,
    in_dim: usize,
};

fn loadLinear(ctx: *ExecContext, file: *const gguf.File, name: []const u8, out_dim: usize, in_dim: usize) !Linear {
    if (try ptqtp_gguf.maybeLoadPlanes(ctx, file, name, out_dim, in_dim)) |w| {
        return .{ .w = w, .out_dim = out_dim, .in_dim = in_dim };
    }
    return .{ .w = try LinearWeight.load(ctx, try file.get(name), out_dim, in_dim), .out_dim = out_dim, .in_dim = in_dim };
}

const AttnW = struct {
    in_proj: Linear,
    out_proj: Linear,
};

const LayerW = struct {
    norm1_w: VecEmbed,
    norm1_b: VecEmbed,
    norm2_w: VecEmbed,
    norm2_b: VecEmbed,
    attn: AttnW,
    lin1: Linear, // [ffn, d]
    lin2: Linear, // [d, ffn]
};

fn loadVec(ctx: *ExecContext, file: *const gguf.File, name: []const u8, len: usize) !VecEmbed {
    const data = try f32Data(file, name);
    if (data.len != len) return Error.BadShape;
    return VecEmbed.fromSlice(ctx, .{len}, data);
}

/// LayerScale (mimi) is folded into out_proj/linear2 rows by the converter.
fn loadLayer(ctx: *ExecContext, file: *const gguf.File, comptime prefix: []const u8, i: usize, d: usize, ffn: usize) !LayerW {
    var buf: [128]u8 = undefined;
    return .{
        .norm1_w = try loadVec(ctx, file, fmtName(&buf, prefix ++ ".layers.{d}.norm1.weight", .{i}), d),
        .norm1_b = try loadVec(ctx, file, fmtName(&buf, prefix ++ ".layers.{d}.norm1.bias", .{i}), d),
        .norm2_w = try loadVec(ctx, file, fmtName(&buf, prefix ++ ".layers.{d}.norm2.weight", .{i}), d),
        .norm2_b = try loadVec(ctx, file, fmtName(&buf, prefix ++ ".layers.{d}.norm2.bias", .{i}), d),
        .attn = .{
            .in_proj = try loadLinear(ctx, file, fmtName(&buf, prefix ++ ".layers.{d}.self_attn.in_proj.weight", .{i}), 3 * d, d),
            .out_proj = try loadLinear(ctx, file, fmtName(&buf, prefix ++ ".layers.{d}.self_attn.out_proj.weight", .{i}), d, d),
        },
        .lin1 = try loadLinear(ctx, file, fmtName(&buf, prefix ++ ".layers.{d}.linear1.weight", .{i}), ffn, d),
        .lin2 = try loadLinear(ctx, file, fmtName(&buf, prefix ++ ".layers.{d}.linear2.weight", .{i}), d, ffn),
    };
}

fn deinitLayer(l: *LayerW) void {
    l.norm1_w.deinit();
    l.norm1_b.deinit();
    l.norm2_w.deinit();
    l.norm2_b.deinit();
    l.attn.in_proj.w.deinit();
    l.attn.out_proj.w.deinit();
    l.lin1.w.deinit();
    l.lin2.w.deinit();
}

const ResBlockW = struct {
    in_ln_w: []const f32,
    in_ln_b: []const f32,
    mlp0_w: []const f32,
    mlp0_b: []const f32,
    mlp2_w: []const f32,
    mlp2_b: []const f32,
    ada_w: []const f32, // [3h, h]
    ada_b: []const f32,
};

pub const Model = struct {
    allocator: Allocator,
    ctx: *ExecContext,
    // dims
    d: usize, // 1024
    layers: usize, // 6
    heads: usize, // 16
    hd: usize, // 64
    ffn: usize, // 4096
    fh: usize, // flow hidden 512
    fblocks: usize, // 6
    // tokenizer
    pieces: [][]const u8,
    scores: []f32,
    // flow-lm
    embed: []const f32, // [4001, d]
    input_linear: []const f32, // [d, 32]
    bos_emb: []const f32, // [32]
    emb_mean: []const f32,
    emb_std: []const f32,
    out_norm_w: []const f32,
    out_norm_b: []const f32,
    out_eos_w: []const f32,
    out_eos_b: []const f32,
    lm_layers: []LayerW,
    // flow net
    input_proj_w: []const f32,
    input_proj_b: []const f32,
    cond_embed_w: []const f32,
    cond_embed_b: []const f32,
    time_freqs: [2][]const f32, // [128] each
    time_mlp0_w: [2][]const f32,
    time_mlp0_b: [2][]const f32,
    time_mlp2_w: [2][]const f32,
    time_mlp2_b: [2][]const f32,
    time_alpha: [2][]const f32,
    res_blocks: []ResBlockW,
    final_ada_w: []const f32,
    final_ada_b: []const f32,
    final_lin_w: []const f32,
    final_lin_b: []const f32,
    // mimi
    mimi_d: usize, // 512
    mimi_layers_n: usize, // 2
    mimi_heads: usize, // 8
    mimi_ctx: usize, // 250
    output_proj: []const f32, // [512, 32] (1x1 conv squeezed)
    upsample_w: []const f32, // [512][32] depthwise k32
    mimi_layers: []LayerW,
    dec: SeanetW,

    pub fn load(ctx: *ExecContext, file: *const gguf.File) !Model {
        const allocator = ctx.allocator();
        const d: usize = @intCast(file.getInt("pocket.d_model") orelse return Error.MissingMetadata);
        const layers: usize = @intCast(file.getInt("pocket.layers") orelse return Error.MissingMetadata);
        const heads: usize = @intCast(file.getInt("pocket.heads") orelse return Error.MissingMetadata);
        const ffn: usize = @intCast(file.getInt("pocket.ffn") orelse return Error.MissingMetadata);
        const fh: usize = @intCast(file.getInt("pocket.flow.hidden") orelse return Error.MissingMetadata);
        const fblocks: usize = @intCast(file.getInt("pocket.flow.blocks") orelse return Error.MissingMetadata);

        const tokens_arr = file.getArray("tokenizer.pocket.tokens") orelse return Error.MissingMetadata;
        const scores_arr = file.getArray("tokenizer.pocket.scores") orelse return Error.MissingMetadata;
        const pieces = try tokens_arr.stringSlices(allocator);
        errdefer allocator.free(pieces);
        if (scores_arr.item_type != 6 or scores_arr.data.len != 4 * scores_arr.len) return Error.BadShape;
        const scores = try allocator.alloc(f32, scores_arr.len);
        errdefer allocator.free(scores);
        for (scores, 0..) |*sc, i| sc.* = @bitCast(std.mem.readInt(u32, scores_arr.data[i * 4 ..][0..4], .little));

        const lm_layers = try allocator.alloc(LayerW, layers);
        errdefer allocator.free(lm_layers);
        for (lm_layers, 0..) |*l, i| l.* = try loadLayer(ctx, file, "flow_lm.transformer", i, d, ffn);

        const res_blocks = try allocator.alloc(ResBlockW, fblocks);
        errdefer allocator.free(res_blocks);
        var buf: [128]u8 = undefined;
        for (res_blocks, 0..) |*r, i| {
            r.* = .{
                .in_ln_w = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.res_blocks.{d}.in_ln.weight", .{i})),
                .in_ln_b = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.res_blocks.{d}.in_ln.bias", .{i})),
                .mlp0_w = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.res_blocks.{d}.mlp.0.weight", .{i})),
                .mlp0_b = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.res_blocks.{d}.mlp.0.bias", .{i})),
                .mlp2_w = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.res_blocks.{d}.mlp.2.weight", .{i})),
                .mlp2_b = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.res_blocks.{d}.mlp.2.bias", .{i})),
                .ada_w = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.res_blocks.{d}.adaLN_modulation.1.weight", .{i})),
                .ada_b = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.res_blocks.{d}.adaLN_modulation.1.bias", .{i})),
            };
        }

        const mimi_d: usize = @intCast(file.getInt("pocket.mimi.d_model") orelse return Error.MissingMetadata);
        const mimi_layers_n: usize = @intCast(file.getInt("pocket.mimi.layers") orelse return Error.MissingMetadata);
        const mimi_layers = try allocator.alloc(LayerW, mimi_layers_n);
        errdefer allocator.free(mimi_layers);
        for (mimi_layers, 0..) |*l, i| l.* = try loadLayer(ctx, file, "mimi.decoder_transformer.transformer", i, mimi_d, 2048);

        var time_freqs: [2][]const f32 = undefined;
        var time_mlp0_w: [2][]const f32 = undefined;
        var time_mlp0_b: [2][]const f32 = undefined;
        var time_mlp2_w: [2][]const f32 = undefined;
        var time_mlp2_b: [2][]const f32 = undefined;
        var time_alpha: [2][]const f32 = undefined;
        for (0..2) |i| {
            time_freqs[i] = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.time_embed.{d}.freqs", .{i}));
            time_mlp0_w[i] = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.time_embed.{d}.mlp.0.weight", .{i}));
            time_mlp0_b[i] = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.time_embed.{d}.mlp.0.bias", .{i}));
            time_mlp2_w[i] = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.time_embed.{d}.mlp.2.weight", .{i}));
            time_mlp2_b[i] = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.time_embed.{d}.mlp.2.bias", .{i}));
            time_alpha[i] = try f32Data(file, fmtName(&buf, "flow_lm.flow_net.time_embed.{d}.mlp.3.alpha", .{i}));
        }

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .d = d,
            .layers = layers,
            .heads = heads,
            .hd = d / heads,
            .ffn = ffn,
            .fh = fh,
            .fblocks = fblocks,
            .pieces = pieces,
            .scores = scores,
            .embed = try f32Data(file, "flow_lm.conditioner.embed.weight"),
            .input_linear = try f32Data(file, "flow_lm.input_linear.weight"),
            .bos_emb = try f32Data(file, "flow_lm.bos_emb"),
            .emb_mean = try f32Data(file, "flow_lm.emb_mean"),
            .emb_std = try f32Data(file, "flow_lm.emb_std"),
            .out_norm_w = try f32Data(file, "flow_lm.out_norm.weight"),
            .out_norm_b = try f32Data(file, "flow_lm.out_norm.bias"),
            .out_eos_w = try f32Data(file, "flow_lm.out_eos.weight"),
            .out_eos_b = try f32Data(file, "flow_lm.out_eos.bias"),
            .lm_layers = lm_layers,
            .input_proj_w = try f32Data(file, "flow_lm.flow_net.input_proj.weight"),
            .input_proj_b = try f32Data(file, "flow_lm.flow_net.input_proj.bias"),
            .cond_embed_w = try f32Data(file, "flow_lm.flow_net.cond_embed.weight"),
            .cond_embed_b = try f32Data(file, "flow_lm.flow_net.cond_embed.bias"),
            .time_freqs = time_freqs,
            .time_mlp0_w = time_mlp0_w,
            .time_mlp0_b = time_mlp0_b,
            .time_mlp2_w = time_mlp2_w,
            .time_mlp2_b = time_mlp2_b,
            .time_alpha = time_alpha,
            .res_blocks = res_blocks,
            .final_ada_w = try f32Data(file, "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight"),
            .final_ada_b = try f32Data(file, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias"),
            .final_lin_w = try f32Data(file, "flow_lm.flow_net.final_layer.linear.weight"),
            .final_lin_b = try f32Data(file, "flow_lm.flow_net.final_layer.linear.bias"),
            .mimi_d = mimi_d,
            .mimi_layers_n = mimi_layers_n,
            .mimi_heads = @intCast(file.getInt("pocket.mimi.heads") orelse return Error.MissingMetadata),
            .mimi_ctx = @intCast(file.getInt("pocket.mimi.context") orelse return Error.MissingMetadata),
            .output_proj = try f32Data(file, "mimi.quantizer.output_proj.weight"),
            .upsample_w = try f32Data(file, "mimi.upsample.convtr.convtr.weight"),
            .mimi_layers = mimi_layers,
            .dec = try SeanetW.load(file),
        };
    }

    pub fn deinit(self: *Model) void {
        for (self.lm_layers) |*l| deinitLayer(l);
        for (self.mimi_layers) |*l| deinitLayer(l);
        self.allocator.free(self.pieces);
        self.allocator.free(self.scores);
        self.allocator.free(self.lm_layers);
        self.allocator.free(self.res_blocks);
        self.allocator.free(self.mimi_layers);
        self.* = undefined;
    }

    // --- tokenizer (SPM unigram Viterbi) ------------------------------------

    /// prepare_text_prompt + SentencePiece: normalize whitespace, uppercase
    /// the first char, ensure trailing '.', spaces → ▁ with a dummy prefix,
    /// then best-path segmentation by piece scores.
    pub fn tokenize(self: *const Model, allocator: Allocator, text: []const u8) ![]u32 {
        var norm: std.ArrayList(u8) = .empty;
        defer norm.deinit(allocator);
        const marker = "\xe2\x96\x81"; // ▁
        try norm.appendSlice(allocator, marker);
        var prev_space = true;
        const trimmed = std.mem.trim(u8, text, " \t\n\r");
        for (trimmed, 0..) |ch, i| {
            var c = ch;
            if (c == '\n' or c == '\r' or c == '\t') c = ' ';
            if (c == ';') c = ',';
            if (c == ' ') {
                if (prev_space) continue;
                prev_space = true;
                try norm.appendSlice(allocator, marker);
                continue;
            }
            prev_space = false;
            if (i == 0) c = std.ascii.toUpper(c);
            try norm.append(allocator, c);
        }
        if (trimmed.len > 0 and std.ascii.isAlphanumeric(trimmed[trimmed.len - 1])) {
            try norm.append(allocator, '.');
        }

        const s = norm.items;
        const n = s.len;
        const neg_inf = -std.math.inf(f32);
        const best = try allocator.alloc(f32, n + 1);
        defer allocator.free(best);
        const back = try allocator.alloc(usize, n + 1);
        defer allocator.free(back);
        const back_id = try allocator.alloc(u32, n + 1);
        defer allocator.free(back_id);
        @memset(best, neg_inf);
        best[0] = 0;
        for (0..n) |i| {
            if (best[i] == neg_inf) continue;
            for (self.pieces, 0..) |piece, id| {
                if (piece.len == 0 or i + piece.len > n) continue;
                if (!std.mem.eql(u8, s[i..][0..piece.len], piece)) continue;
                const cand = best[i] + self.scores[id];
                if (cand > best[i + piece.len]) {
                    best[i + piece.len] = cand;
                    back[i + piece.len] = i;
                    back_id[i + piece.len] = @intCast(id);
                }
            }
            // unknown byte fallback: skip one byte with a large penalty
            if (best[i] - 100.0 > best[i + 1]) {
                best[i + 1] = best[i] - 100.0;
                back[i + 1] = i;
                back_id[i + 1] = 0;
            }
        }
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(allocator);
        var pos = n;
        while (pos > 0) {
            try ids.append(allocator, back_id[pos]);
            pos = back[pos];
        }
        const out = try allocator.alloc(u32, ids.items.len);
        for (out, 0..) |*o, i| o.* = ids.items[ids.items.len - 1 - i];
        return out;
    }
};

// --- SEANet decoder weights ------------------------------------------------

const ConvW = fucina.streamconv.Weights;

fn loadConv(file: *const gguf.File, comptime fmt: []const u8, args: anytype, with_bias: bool, transposed: bool) !ConvW {
    var buf: [128]u8 = undefined;
    const wt = file.get(fmtName(&buf, fmt ++ ".weight", args)) catch return Error.MissingTensor;
    if (wt.ggml_type != .f32 or wt.n_dims != 3) return Error.BadShape;
    const w = @as([*]const f32, @ptrCast(@alignCast(wt.data.ptr)))[0 .. wt.data.len / 4];
    // Conv1d is [out, in, k] → ne [k, in, out]; ConvTranspose1d is
    // [in, out, k] → ne [k, out, in].
    const k = wt.dims[0];
    const in_ch = if (transposed) wt.dims[2] else wt.dims[1];
    const out_ch = if (transposed) wt.dims[1] else wt.dims[2];
    const b = if (with_bias) try f32Data(file, fmtName(&buf, fmt ++ ".bias", args)) else null;
    return .{ .w = w, .b = b, .in_ch = in_ch, .out_ch = out_ch, .k = k };
}

/// Decoder Sequential indices (n_residual_layers=1, ratios [6,5,4]):
/// 0 conv(512,512,k7); 1 ELU; 2 convtr(512,256,k12,s6); 3 res(256);
/// 4 ELU; 5 convtr(256,128,k10,s5); 6 res(128);
/// 7 ELU; 8 convtr(128,64,k8,s4); 9 res(64); 10 ELU; 11 conv(64,1,k3).
const SeanetW = struct {
    conv0: ConvW,
    up: [3]ConvW, // convtr per ratio (weights [in][out][k] for transposed!)
    res_c1: [3]ConvW, // k3
    res_c2: [3]ConvW, // k1
    last: ConvW,

    fn load(file: *const gguf.File) !SeanetW {
        var self: SeanetW = undefined;
        self.conv0 = try loadConv(file, "mimi.decoder.model.{d}.conv", .{@as(usize, 0)}, true, false);
        const up_idx = [3]usize{ 2, 5, 8 };
        const res_idx = [3]usize{ 3, 6, 9 };
        for (0..3) |i| {
            self.up[i] = try loadConv(file, "mimi.decoder.model.{d}.convtr", .{up_idx[i]}, true, true);
            self.res_c1[i] = try loadConv(file, "mimi.decoder.model.{d}.block.1.conv", .{res_idx[i]}, true, false);
            self.res_c2[i] = try loadConv(file, "mimi.decoder.model.{d}.block.3.conv", .{res_idx[i]}, true, false);
        }
        self.last = try loadConv(file, "mimi.decoder.model.{d}.conv", .{@as(usize, 11)}, true, false);
        return self;
    }
};

// --- KV cache --------------------------------------------------------------

pub const Kv = struct {
    allocator: Allocator,
    /// per layer: k and v as [cap][heads][hd]
    k: [][]f32,
    v: [][]f32,
    offset: usize = 0,
    cap: usize,
    heads: usize,
    hd: usize,

    pub fn init(allocator: Allocator, layers: usize, cap: usize, heads: usize, hd: usize) !Kv {
        const k = try allocator.alloc([]f32, layers);
        const v = try allocator.alloc([]f32, layers);
        for (k) |*b| b.* = try allocator.alloc(f32, cap * heads * hd);
        for (v) |*b| b.* = try allocator.alloc(f32, cap * heads * hd);
        return .{ .allocator = allocator, .k = k, .v = v, .cap = cap, .heads = heads, .hd = hd };
    }

    pub fn deinit(self: *Kv) void {
        for (self.k) |b| self.allocator.free(b);
        for (self.v) |b| self.allocator.free(b);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.* = undefined;
    }

    pub fn reset(self: *Kv) void {
        self.offset = 0;
    }
};

/// Load a voice prefix (`voice.<name>.cache.<layer>` = [2, T, heads, hd],
/// roped K then V) into the cache and set the offset.
pub fn loadVoice(model: *const Model, file: *const gguf.File, kv: *Kv, name: []const u8) !void {
    var buf: [128]u8 = undefined;
    var vlen: usize = 0;
    for (0..model.layers) |l| {
        const t = file.get(fmtName(&buf, "voice.{s}.cache.{d}", .{ name, l })) catch return Error.UnknownVoice;
        if (t.ggml_type != .f32 or t.n_dims != 4) return Error.BadShape;
        // ne order: [hd, heads, T, 2]
        const hd = t.dims[0];
        const heads = t.dims[1];
        const tlen = t.dims[2];
        if (hd != model.hd or heads != model.heads) return Error.BadShape;
        if (tlen > kv.cap) return Error.KvOverflow;
        const data = @as([*]const f32, @ptrCast(@alignCast(t.data.ptr)))[0 .. t.data.len / 4];
        const half = tlen * heads * hd;
        @memcpy(kv.k[l][0..half], data[0..half]);
        @memcpy(kv.v[l][0..half], data[half .. 2 * half]);
        vlen = tlen;
    }
    kv.offset = vlen;
}

// --- FlowLM ----------------------------------------------------------------

pub const FlowLm = struct {
    model: *const Model,
    allocator: Allocator,
    x: []f32, // last forward's pre-out_norm hidden rows (parity taps)
    head_map: []usize,
    s_max: usize,

    pub fn init(allocator: Allocator, model: *const Model, s_max: usize) !FlowLm {
        const head_map = try allocator.alloc(usize, model.heads);
        for (head_map, 0..) |*h, i| h.* = i;
        return .{
            .model = model,
            .allocator = allocator,
            .x = try allocator.alloc(f32, s_max * model.d),
            .head_map = head_map,
            .s_max = s_max,
        };
    }

    pub fn deinit(self: *FlowLm) void {
        self.allocator.free(self.x);
        self.allocator.free(self.head_map);
        self.* = undefined;
    }

    /// Forward `rows` ([S, d]) at positions kv.offset..; advances the
    /// offset. `last_out` receives out_norm(last hidden).
    pub fn forward(self: *FlowLm, kv: *Kv, rows: []const f32, s: usize, last_out: []f32) !void {
        const m = self.model;
        try transformerForward(m.ctx, m.lm_layers, kv, self.head_map, rows, s, m.d, m.heads, m.hd, 0, 1e-5, self.x);
        @memcpy(last_out, self.x[(s - 1) * m.d ..][0..m.d]);
        layerNorm(last_out, m.out_norm_w, m.out_norm_b, 1e-5);
    }
};

/// Shared pre-LN transformer over a host-side roped-K/V cache, run
/// ENTIRELY on fucina tensor ops (the qwen3tts Stack.forward pattern):
/// tagged layerNorm(+bias) → fused-qkv LinearWeight → interleaved RoPE via
/// the core table → groupedAttention (optional sliding window) → exact-erf
/// GELU FFN. Appends this step's K/V to `kv` and advances its offset.
/// Writes the PRE-out_norm hidden rows into `out_rows` ([s*d]).
fn transformerForward(
    ctx: *ExecContext,
    layers: []const LayerW,
    kv: *Kv,
    head_map: []const usize,
    rows: []const f32,
    s: usize,
    d: usize,
    heads: usize,
    hd: usize,
    window: usize,
    ln_eps: f32,
    out_rows: []f32,
) !void {
    if (kv.offset + s > kv.cap) return Error.KvOverflow;
    const allocator = ctx.allocator();
    const positions = try allocator.alloc(i32, s);
    defer allocator.free(positions);
    for (positions, 0..) |*pp, i| pp.* = @intCast(kv.offset + i);
    var rope_table = try ctx.prepareRopeTable(.{ .positions = .{ .explicit = positions }, .feature_dim = hd, .freqs = .{ .theta = .{ .base = 10000.0 } } });
    defer rope_table.deinit();

    var x = try Rows.fromBorrowedConstSlice(ctx, .{ s, d }, rows);
    errdefer x.deinit();
    const cached_len = kv.offset + s;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    for (layers, 0..) |*layer, li| {
        var n1 = try x.layerNorm(ctx, .embed, ln_eps, .{ .weight = &layer.norm1_w, .bias = &layer.norm1_b });
        defer n1.deinit();
        var qkv = try layer.attn.in_proj.w.linearSeq(ctx, &n1, .embed, .q);
        defer qkv.deinit();
        // fused [s, 3d] → three contiguous [s, heads, hd] (host gather; the
        // strided narrow is not a valid split view)
        const qkvd = try qkv.dataConst();
        var q3 = try fucina.Tensor(.{ .seq, .head, .d }).empty(ctx, .{ s, heads, hd });
        defer q3.deinit();
        var k3 = try fucina.Tensor(.{ .seq, .kv_head, .d }).empty(ctx, .{ s, heads, hd });
        defer k3.deinit();
        var v3 = try fucina.Tensor(.{ .seq, .kv_head, .d }).empty(ctx, .{ s, heads, hd });
        defer v3.deinit();
        const q_dst = try q3.data();
        const k_dst = try k3.data();
        const v_dst = try v3.data();
        for (0..s) |t| {
            @memcpy(q_dst[t * d ..][0..d], qkvd[t * 3 * d ..][0..d]);
            @memcpy(k_dst[t * d ..][0..d], qkvd[t * 3 * d + d ..][0..d]);
            @memcpy(v_dst[t * d ..][0..d], qkvd[t * 3 * d + 2 * d ..][0..d]);
        }

        const q_raw = try ctx.ropeWithTable(3, q3.asRawTensor(), 0, 2, &rope_table, .interleaved);
        var q_rope = try fucina.Tensor(.{ .seq, .head, .d }).fromTensor(ctx, q_raw);
        defer q_rope.deinit();
        const k_raw = try ctx.ropeWithTable(3, k3.asRawTensor(), 0, 2, &rope_table, .interleaved);
        var k_rope = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromTensor(ctx, k_raw);
        defer k_rope.deinit();

        {
            const kd = try k_rope.dataConst();
            const vd = try v3.dataConst();
            @memcpy(kv.k[li][kv.offset * heads * hd ..][0 .. s * heads * hd], kd);
            @memcpy(kv.v[li][kv.offset * heads * hd ..][0 .. s * heads * hd], vd);
        }
        // Borrowed ring views (rows appended above; attention reads only).
        var k_all = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromBorrowedConstSlice(ctx, .{ cached_len, heads, hd }, kv.k[li][0 .. cached_len * heads * hd]);
        defer k_all.deinit();
        var v_all = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromBorrowedConstSlice(ctx, .{ cached_len, heads, hd }, kv.v[li][0 .. cached_len * heads * hd]);
        defer v_all.deinit();

        var attn = try q_rope.groupedAttention(ctx, &k_all, &v_all, head_map, .attn, scale, .{ .window = window });
        defer attn.deinit();
        var attn_out = try layer.attn.out_proj.w.linearSeq(ctx, &attn, .attn, .embed);
        defer attn_out.deinit();
        const res1 = try x.add(ctx, &attn_out);
        x.deinit();
        x = res1;

        var n2 = try x.layerNorm(ctx, .embed, ln_eps, .{ .weight = &layer.norm2_w, .bias = &layer.norm2_b });
        defer n2.deinit();
        var h1 = try layer.lin1.w.linearSeq(ctx, &n2, .embed, .ffn);
        defer h1.deinit();
        var act = try h1.geluErf(ctx);
        defer act.deinit();
        var down = try layer.lin2.w.linearSeq(ctx, &act, .ffn, .embed);
        defer down.deinit();
        const res2 = try x.add(ctx, &down);
        x.deinit();
        x = res2;
    }
    kv.offset += s;
    @memcpy(out_rows[0 .. s * d], try x.dataConst());
    x.deinit();
}

// --- flow head -------------------------------------------------------------

/// v = flow_net(cond, s, t, x): SimpleMLPAdaLN with two timestep embedders.
pub fn flowNet(model: *const Model, allocator: Allocator, cond: []const f32, s_time: f32, t_time: f32, x_in: []const f32, out: []f32) !void {
    const fh = model.fh;
    const y = try allocator.alloc(f32, fh);
    defer allocator.free(y);
    const tmp = try allocator.alloc(f32, fh);
    defer allocator.free(tmp);
    const tmp2 = try allocator.alloc(f32, fh);
    defer allocator.free(tmp2);
    const freq_emb = try allocator.alloc(f32, 2 * model.time_freqs[0].len);
    defer allocator.free(freq_emb);

    // y = mean of two timestep embeddings + cond_embed(cond)
    @memset(y, 0);
    const times = [2]f32{ s_time, t_time };
    for (0..2) |ti| {
        const freqs = model.time_freqs[ti];
        for (freqs, 0..) |f, i| {
            freq_emb[i] = @cos(f * times[ti]);
            freq_emb[freqs.len + i] = @sin(f * times[ti]);
        }
        matvec(tmp, model.time_mlp0_w[ti], freq_emb, model.time_mlp0_b[ti]);
        for (tmp) |*v| v.* = silu(v.*);
        matvec(tmp2, model.time_mlp2_w[ti], tmp, model.time_mlp2_b[ti]);
        timestepRmsNorm(tmp2, model.time_alpha[ti]);
        for (y, tmp2) |*a, b| a.* += b * 0.5;
    }
    matvec(tmp, model.cond_embed_w, cond, model.cond_embed_b);
    for (y, tmp) |*a, b| a.* += b;

    // x path
    const h = try allocator.alloc(f32, fh);
    defer allocator.free(h);
    const hb = try allocator.alloc(f32, fh);
    defer allocator.free(hb);
    const ada = try allocator.alloc(f32, 3 * fh);
    defer allocator.free(ada);
    const ysilu = try allocator.alloc(f32, fh);
    defer allocator.free(ysilu);
    matvec(h, model.input_proj_w, x_in, model.input_proj_b);
    for (ysilu, y) |*a, b| a.* = silu(b);

    for (model.res_blocks) |*rb| {
        matvec(ada, rb.ada_w, ysilu, rb.ada_b);
        const shift = ada[0..fh];
        const scale = ada[fh .. 2 * fh];
        const gate = ada[2 * fh .. 3 * fh];
        @memcpy(hb, h);
        layerNorm(hb, rb.in_ln_w, rb.in_ln_b, 1e-6);
        for (hb, 0..) |*v, i| v.* = v.* * (1 + scale[i]) + shift[i];
        matvec(tmp, rb.mlp0_w, hb, rb.mlp0_b);
        for (tmp) |*v| v.* = silu(v.*);
        matvec(hb, rb.mlp2_w, tmp, rb.mlp2_b);
        for (h, hb, 0..) |*v, u, i| v.* += gate[i] * u;
    }
    // final layer: non-affine LN, 2-way adaLN, linear to 32
    const ada2 = try allocator.alloc(f32, 2 * fh);
    defer allocator.free(ada2);
    matvec(ada2, model.final_ada_w, ysilu, model.final_ada_b);
    @memcpy(hb, h);
    layerNorm(hb, null, null, 1e-6);
    for (hb, 0..) |*v, i| v.* = v.* * (1 + ada2[fh + i]) + ada2[i];
    matvec(out, model.final_lin_w, hb, model.final_lin_b);
}

/// One LSD step: x1 = x0 + flow_net(cond, 0, 1, x0).
pub fn lsdStep(model: *const Model, allocator: Allocator, cond: []const f32, x0: []const f32, x1: []f32) !void {
    try flowNet(model, allocator, cond, 0.0, 1.0, x0, x1);
    for (x1, x0) |*v, a| v.* += a;
}

// --- Mimi decoder session --------------------------------------------------

pub const Mimi = struct {
    allocator: Allocator,
    model: *const Model,
    kv: Kv,
    up_partial: []f32, // depthwise upsample partial [512][16]
    conv0: fucina.streamconv.StreamingConv1d,
    ups: [3]fucina.streamconv.StreamingConvTranspose1d,
    res1: [3]fucina.streamconv.StreamingConv1d,
    res2: [3]fucina.streamconv.StreamingConv1d,
    last: fucina.streamconv.StreamingConv1d,
    // transformer scratch
    x: []f32,
    head_map: []usize,
    // parity captures (tests set these to [512] / [512*16] buffers)
    cap_q: ?[]f32 = null,
    cap_up: ?[]f32 = null,
    cap_tf: ?[]f32 = null,

    pub fn init(allocator: Allocator, model: *const Model, max_frames: usize) !Mimi {
        const d = model.mimi_d;
        return .{
            .allocator = allocator,
            .model = model,
            .kv = try Kv.init(allocator, model.mimi_layers_n, max_frames * 16, model.mimi_heads, d / model.mimi_heads),
            .up_partial = blk: {
                const p = try allocator.alloc(f32, d * 16);
                @memset(p, 0);
                break :blk p;
            },
            .conv0 = try fucina.streamconv.StreamingConv1d.init(allocator, &model.dec.conv0, 1),
            .ups = .{
                try fucina.streamconv.StreamingConvTranspose1d.init(allocator, &model.dec.up[0], 6, false),
                try fucina.streamconv.StreamingConvTranspose1d.init(allocator, &model.dec.up[1], 5, false),
                try fucina.streamconv.StreamingConvTranspose1d.init(allocator, &model.dec.up[2], 4, false),
            },
            .res1 = .{
                try fucina.streamconv.StreamingConv1d.init(allocator, &model.dec.res_c1[0], 1),
                try fucina.streamconv.StreamingConv1d.init(allocator, &model.dec.res_c1[1], 1),
                try fucina.streamconv.StreamingConv1d.init(allocator, &model.dec.res_c1[2], 1),
            },
            .res2 = .{
                try fucina.streamconv.StreamingConv1d.init(allocator, &model.dec.res_c2[0], 1),
                try fucina.streamconv.StreamingConv1d.init(allocator, &model.dec.res_c2[1], 1),
                try fucina.streamconv.StreamingConv1d.init(allocator, &model.dec.res_c2[2], 1),
            },
            .last = try fucina.streamconv.StreamingConv1d.init(allocator, &model.dec.last, 1),
            .x = try allocator.alloc(f32, 16 * d),
            .head_map = blk: {
                const hm = try allocator.alloc(usize, model.mimi_heads);
                for (hm, 0..) |*h, i| h.* = i;
                break :blk hm;
            },
        };
    }

    pub fn deinit(self: *Mimi) void {
        self.kv.deinit();
        self.allocator.free(self.up_partial);
        self.allocator.free(self.conv0.prev);
        for (&self.ups) |*u| self.allocator.free(u.partial);
        for (&self.res1) |*r| self.allocator.free(r.prev);
        for (&self.res2) |*r| self.allocator.free(r.prev);
        self.allocator.free(self.last.prev);
        self.allocator.free(self.x);
        self.allocator.free(self.head_map);
        self.* = undefined;
    }

    /// Fresh stream: zero all conv tails/partials and the KV offset.
    pub fn reset(self: *Mimi) void {
        self.kv.reset();
        @memset(self.up_partial, 0);
        @memset(self.conv0.prev, 0);
        for (&self.ups) |*u| @memset(u.partial, 0);
        for (&self.res1) |*r| @memset(r.prev, 0);
        for (&self.res2) |*r| @memset(r.prev, 0);
        @memset(self.last.prev, 0);
    }

    /// One RAW flow latent [32] → 1920 samples at 24 kHz.
    pub fn decodeFrame(self: *Mimi, latent: []const f32, pcm: []f32) !void {
        const m = self.model;
        const d = m.mimi_d;
        const allocator = self.allocator;
        // de-normalize + output_proj (1x1)
        var z: [latent_dim]f32 = undefined;
        for (0..latent_dim) |i| z[i] = latent[i] * m.emb_std[i] + m.emb_mean[i];
        const q = try allocator.alloc(f32, d);
        defer allocator.free(q);
        matvec(q, m.output_proj, &z, null);
        if (self.cap_q) |cq| @memcpy(cq[0..d], q);

        // depthwise convtr upsample k32 s16, T=1: emit 16, partial 16
        const up = try allocator.alloc(f32, d * 16);
        defer allocator.free(up);
        for (0..d) |c| {
            const taps = m.upsample_w[c * 32 ..][0..32];
            const xv = q[c];
            for (0..16) |i| up[c * 16 + i] = self.up_partial[c * 16 + i] + xv * taps[i];
            for (0..16) |i| self.up_partial[c * 16 + i] = xv * taps[16 + i];
        }

        if (self.cap_up) |cu| @memcpy(cu[0 .. d * 16], up);
        // decoder transformer: 16 positions, sliding context, channel-major in
        // [d][16] → time-major rows
        var rows_in = try allocator.alloc(f32, 16 * d);
        defer allocator.free(rows_in);
        for (0..16) |t| {
            for (0..d) |c| rows_in[t * d + c] = up[c * 16 + t];
        }
        try transformerForward(m.ctx, m.mimi_layers, &self.kv, self.head_map, rows_in, 16, d, m.mimi_heads, d / m.mimi_heads, m.mimi_ctx, 1e-5, self.x);
        // back to channel-major
        const emb = try allocator.alloc(f32, d * 16);
        defer allocator.free(emb);
        for (0..16) |t| {
            for (0..d) |c| emb[c * 16 + t] = self.x[t * d + c];
        }
        if (self.cap_tf) |ct| @memcpy(ct[0 .. d * 16], emb);

        // SEANet: conv0 (T16) → [ELU, up, res]×3 → ELU, last conv
        // ping-pong: conv0 out (512×16) and ratio-2 out (64×1920) share `a`
        // (the conv0 content is consumed by ratio 0 before `a` is rewritten);
        // ratio 0 → b (256×96), ratio 1 → cbuf (128×480).
        const a = try allocator.alloc(f32, 64 * 1920);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, 256 * 96);
        defer allocator.free(b);
        const cbuf = try allocator.alloc(f32, 128 * 480);
        defer allocator.free(cbuf);

        try self.conv0.run(allocator, emb, 16, a[0 .. 512 * 16]);
        var cur: []f32 = a[0 .. 512 * 16];
        var t_cur: usize = 16;
        var ch: usize = 512;
        const bufs = [3][]f32{ b, cbuf, a };
        for (0..3) |ri| {
            for (cur) |*v| v.* = elu(v.*);
            const oc = ch / 2;
            const t_out = t_cur * self.ups[ri].stride;
            const dst = bufs[ri][0 .. oc * t_out];
            try self.ups[ri].run(allocator, cur, t_cur, dst);
            // residual block: ELU→conv k3→ELU→conv k1, + shortcut
            const h1 = try allocator.alloc(f32, (oc / 2) * t_out);
            defer allocator.free(h1);
            const h2 = try allocator.alloc(f32, oc * t_out);
            defer allocator.free(h2);
            const act = try allocator.alloc(f32, oc * t_out);
            defer allocator.free(act);
            @memcpy(act, dst);
            for (act) |*v| v.* = elu(v.*);
            try self.res1[ri].run(allocator, act, t_out, h1);
            for (h1) |*v| v.* = elu(v.*);
            try self.res2[ri].run(allocator, h1, t_out, h2);
            for (dst, h2) |*v, u| v.* += u;
            cur = dst;
            t_cur = t_out;
            ch = oc;
        }
        for (cur) |*v| v.* = elu(v.*);
        try self.last.run(allocator, cur, t_cur, pcm[0..t_cur]);
    }
};

// --- high-level engine ------------------------------------------------------

pub const SpeakOpts = struct {
    temp: f32,
    seed: u64 = 42,
    eos_threshold: f32 = default_eos_threshold,
    max_frames: usize = 500,
    frames_after_eos: usize = 2,
};

/// One loaded model + voice, reusable across replies. `speak` streams 80 ms
/// PCM frames (1920 samples, 24 kHz) to the callback; a `false` return
/// aborts generation. Long texts chunk at sentence boundaries (≤50 tokens)
/// against the same voice prefix.
pub const Engine = struct {
    allocator: Allocator,
    model: Model,
    kv: Kv,
    voice_len: usize,
    temp_default: f32,

    pub fn init(ctx: *ExecContext, file: *const gguf.File, voice: []const u8) !Engine {
        const allocator = ctx.allocator();
        var model = try Model.load(ctx, file);
        errdefer model.deinit();
        var kv = try Kv.init(allocator, model.layers, 4096, model.heads, model.hd);
        errdefer kv.deinit();
        try loadVoice(&model, file, &kv, voice);
        // NOTE: Engine is returned by value — nothing here may hold a
        // pointer into `model`; FlowLm/Mimi are created per speak() where
        // self.model's address is stable.
        return .{
            .allocator = allocator,
            .model = model,
            .kv = kv,
            .voice_len = kv.offset,
            .temp_default = @floatCast(file.getFloat("pocket.default_temperature") orelse 0.3),
        };
    }

    pub fn deinit(self: *Engine) void {
        self.kv.deinit();
        self.model.deinit();
        self.* = undefined;
    }

    /// Returns frames generated; stops early when the callback returns false.
    pub fn speak(self: *Engine, text_in: []const u8, opts: SpeakOpts, ctx: anytype, comptime cb: fn (@TypeOf(ctx), []const f32) bool) !usize {
        const allocator = self.allocator;
        const model = &self.model;
        var fl = try FlowLm.init(allocator, model, 128);
        defer fl.deinit();
        var mimi = try Mimi.init(allocator, model, opts.max_frames + 8);
        defer mimi.deinit();
        var prng = std.Random.DefaultPrng.init(opts.seed);
        const rng = prng.random();

        const cond = try allocator.alloc(f32, model.d);
        defer allocator.free(cond);
        const row = try allocator.alloc(f32, model.d);
        defer allocator.free(row);
        var pcm: [frame_samples]f32 = undefined;
        var x0: [latent_dim]f32 = undefined;
        var x1: [latent_dim]f32 = undefined;
        var frames_total: usize = 0;

        const trimmed = std.mem.trim(u8, text_in, " \t\r\n");
        var chunk_start: usize = 0;
        outer: while (chunk_start < trimmed.len) {
            var chunk_end = trimmed.len;
            {
                var last_ok: usize = chunk_start;
                var probe = chunk_start;
                while (probe < trimmed.len) {
                    probe += 1;
                    if (probe == trimmed.len or trimmed[probe - 1] == '.' or trimmed[probe - 1] == '!' or trimmed[probe - 1] == '?') {
                        const ids_probe = try model.tokenize(allocator, trimmed[chunk_start..probe]);
                        const fits = ids_probe.len <= 50;
                        allocator.free(ids_probe);
                        if (fits) {
                            last_ok = probe;
                            if (probe == trimmed.len) break;
                        } else break;
                    }
                }
                chunk_end = if (last_ok > chunk_start) last_ok else @min(trimmed.len, chunk_start + 160);
            }
            const chunk_text = std.mem.trim(u8, trimmed[chunk_start..chunk_end], " ");
            chunk_start = chunk_end;
            if (chunk_text.len == 0) continue;

            self.kv.offset = self.voice_len; // rewind to the voice prefix
            const ids = try model.tokenize(allocator, chunk_text);
            defer allocator.free(ids);
            if (ids.len == 0) continue;
            const rows = try allocator.alloc(f32, ids.len * model.d);
            defer allocator.free(rows);
            for (ids, 0..) |id, i| @memcpy(rows[i * model.d ..][0..model.d], model.embed[id * model.d ..][0..model.d]);
            try fl.forward(&self.kv, rows, ids.len, cond);

            for (row, 0..) |*v, i| {
                var acc: f32 = 0;
                for (0..latent_dim) |j| acc += model.input_linear[i * latent_dim + j] * model.bos_emb[j];
                v.* = acc;
            }
            var eos_at: ?usize = null;
            var frame: usize = 0;
            mimi.reset();
            while (frame < opts.max_frames) : (frame += 1) {
                try fl.forward(&self.kv, row, 1, cond);
                var eos_logit: f32 = model.out_eos_b[0];
                for (cond, model.out_eos_w) |c, w| eos_logit += c * w;
                if (eos_logit > opts.eos_threshold and eos_at == null) eos_at = frame;
                if (eos_at) |e| if (frame >= e + opts.frames_after_eos) break;
                const std_dev = @sqrt(opts.temp);
                for (&x0) |*v| v.* = rng.floatNorm(f32) * std_dev;
                try lsdStep(model, allocator, cond, &x0, &x1);
                try mimi.decodeFrame(&x1, &pcm);
                frames_total += 1;
                if (!cb(ctx, &pcm)) break :outer;
                for (row, 0..) |*v, i| {
                    var acc: f32 = 0;
                    for (0..latent_dim) |j| acc += model.input_linear[i * latent_dim + j] * x1[j];
                    v.* = acc;
                }
            }
        }
        return frames_total;
    }
};

test {
    _ = @import("model_tests.zig");
}
