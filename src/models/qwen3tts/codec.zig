//! Qwen3-TTS 12.5 Hz codec decoder (qwentts.cpp pipeline-codec decode side):
//! RVQ codes `[K=16, T]` → 24 kHz mono waveform `[T * 1920]`.
//!
//! Graph: RVQ dequant (16 fused codebooks `[2048, 256]`, 1 semantic + 15
//! acoustic; per-group gather+sum then a bias-free 256→512 out_proj, groups
//! ADDED) → causal Conv1d k=3 512→1024 → 8-layer pre-transformer (hidden 512,
//! 16 heads, head_dim 64, NEOX RoPE θ=10000, sliding-window-72 causal mask,
//! LayerScale, SwiGLU, input/output projections WITH bias) → 2× upsample
//! (CausalTransConv1d k=2 s=2 + ConvNeXt block) → DAC v2 (conv_pre k=7
//! 1024→1536; 4 blocks stride {8,5,4,3} of snake → CausalTransConv1d k=2s →
//! 3 dilated res units; snake_post; conv_post k=7 96→1) → clamp ±1.
//! Total upsample 4 · 480 = 1920 samples per frame.
//!
//! Decode is STATELESS whole-sequence; `decodeChunked` mirrors the
//! reference's chunked_decode (re-decode with left-context frames, drop the
//! context samples — bit-equal to one-shot only within a single chunk).
//!
//! Internal layout is Fucina's `[T, C]` rows (channel fast) everywhere.
//! SnakeBeta is folded at load: `a = exp(alpha)`, `inv_b = 1/(exp(beta)+1e-9)`
//! (the facade `snake` op computes `x + inv_b·sin²(a·x)`).

const std = @import("std");
const fucina = @import("fucina");
const weights = @import("fucina").weights;

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;
const Allocator = std.mem.Allocator;

pub const hop_length = 1920;
pub const sample_rate = 24000;
pub const num_codebooks = 16;
pub const code_bits = 11;

pub const Error = error{
    MissingMetadata,
    BadShape,
    InvalidCodes,
    CodeOutOfRange,
    NotF32,
};

/// Activation rows `[T, C]` (running channel axis is `.in`).
const Act = fucina.Tensor(.{ .seq, .in });
const Codebook = fucina.Tensor(.{ .code, .cdim });
const ConvW = fucina.Tensor(.{ .tap, .in, .out });
const DwW = fucina.Tensor(.{ .in, .tap });
const TConvW = fucina.Tensor(.{ .kout, .in });
const VecIn = fucina.Tensor(.{.in});
const VecD = fucina.Tensor(.{.d});
const LinearWeight = weights.LinearWeight;

pub const Config = struct {
    hidden: usize, // pre-transformer width (512)
    latent: usize, // conv trunk width (1024)
    decoder_dim: usize, // DAC entry width (1536)
    cdim: usize, // codebook entry dim (256)
    codebook_size: usize, // 2048
    vq_hidden: usize, // RVQ projection target (512)
    n_layers: usize, // 8
    n_heads: usize, // 16
    n_kv_heads: usize, // 16
    head_dim: usize, // 64
    ffn: usize, // 1024
    sliding_window: usize, // 72
    rope_theta: f32, // 10000
    rms_eps: f32, // 1e-5
    n_quantizers: usize, // 16
    n_semantic: usize, // 1

    fn metaU(file: *const gguf.File, comptime key: []const u8) !usize {
        const v = file.getInt("qwen3-tts-tokenizer.decoder." ++ key) orelse return Error.MissingMetadata;
        return @intCast(v);
    }
    fn metaF(file: *const gguf.File, comptime key: []const u8, default: f32) f32 {
        return if (file.getFloat("qwen3-tts-tokenizer.decoder." ++ key)) |v| @floatCast(v) else default;
    }

    pub fn fromGguf(file: *const gguf.File) !Config {
        return .{
            .hidden = try metaU(file, "hidden_size"),
            .latent = try metaU(file, "latent_dim"),
            .decoder_dim = try metaU(file, "decoder_dim"),
            .cdim = try metaU(file, "codebook_dim_internal"),
            .codebook_size = try metaU(file, "codebook_size"),
            .vq_hidden = try metaU(file, "vector_quantization_hidden_dim"),
            .n_layers = try metaU(file, "num_hidden_layers"),
            .n_heads = try metaU(file, "num_attention_heads"),
            .n_kv_heads = try metaU(file, "num_key_value_heads"),
            .head_dim = try metaU(file, "head_dim"),
            .ffn = try metaU(file, "intermediate_size"),
            .sliding_window = try metaU(file, "sliding_window"),
            .rope_theta = metaF(file, "rope_theta", 10000.0),
            .rms_eps = metaF(file, "rms_norm_eps", 1e-5),
            .n_quantizers = try metaU(file, "num_quantizers"),
            .n_semantic = try metaU(file, "num_semantic_quantizers"),
        };
    }
};

// --- raw f32 loading helpers ------------------------------------------------

/// Borrow an F32 GGUF tensor's payload as `[]const f32` (the codec GGUF is
/// all-F32 by construction — the converter writes every tensor as f32).
fn f32Data(t: *const gguf.TensorInfo) ![]const f32 {
    if (t.ggml_type != .f32) return Error.NotF32;
    if (!std.mem.isAligned(@intFromPtr(t.data.ptr), @alignOf(f32))) return Error.BadShape;
    return @as([*]const f32, @ptrCast(@alignCast(t.data.ptr)))[0 .. t.data.len / 4];
}

fn elemCount(t: *const gguf.TensorInfo) usize {
    var n: usize = 1;
    for (0..t.n_dims) |i| n *= t.dims[i];
    return n;
}

fn fmtName(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

fn loadNamed(comptime T: type, ctx: *ExecContext, file: *const gguf.File, name: []const u8, shape: anytype) !T {
    const t = try file.get(name);
    const data = try f32Data(t);
    return T.fromSlice(ctx, shape, data);
}

fn loadLin(ctx: *ExecContext, file: *const gguf.File, name: []const u8, out_n: usize, in_n: usize) !LinearWeight {
    return LinearWeight.load(ctx, try file.get(name), out_n, in_n);
}

/// A 1x1-Conv1d projection stored `[1, IC, OC]`: squeeze the tap axis and
/// bind it as a plain `(OC, IC)` matmul weight.
fn loadProj1x1(ctx: *ExecContext, file: *const gguf.File, name: []const u8, out_n: usize, in_n: usize) !LinearWeight {
    const t = try file.get(name);
    if (t.n_dims != 3 or t.dims[0] != 1 or t.dims[1] != in_n or t.dims[2] != out_n) return Error.BadShape;
    var flat = t.*;
    flat.n_dims = 2;
    flat.dims = .{ in_n, out_n, 1, 1 };
    return LinearWeight.load(ctx, &flat, out_n, in_n);
}

/// Borrow a bias/param vector straight from the F32 mmap.
fn loadSlice(file: *const gguf.File, name: []const u8, n: usize) ![]const f32 {
    const t = try file.get(name);
    if (elemCount(t) != n) return Error.BadShape;
    return f32Data(t);
}

fn loadVec(comptime T: type, ctx: *ExecContext, file: *const gguf.File, name: []const u8, n: usize) !T {
    const t = try file.get(name);
    if (elemCount(t) != n) return Error.BadShape;
    return loadNamed(T, ctx, file, name, .{n});
}

/// Conv1d weight: GGUF ne `[K, IC, OC]` (memory `w[oc][ic][k]`) → facade
/// `[tap, in, out]` (`w2[k][ic][oc]`, out fastest), tap taps−1 = newest.
fn loadConvW(ctx: *ExecContext, file: *const gguf.File, name: []const u8, k: usize, ic: usize, oc: usize) !ConvW {
    const t = try file.get(name);
    if (t.dims[0] != k or t.dims[1] != ic or t.dims[2] != oc) return Error.BadShape;
    const src = try f32Data(t);
    var w = try ConvW.empty(ctx, .{ k, ic, oc });
    errdefer w.deinit();
    const dst = try w.data();
    for (0..oc) |o| {
        for (0..ic) |i| {
            for (0..k) |kk| {
                dst[(kk * ic + i) * oc + o] = src[(o * ic + i) * k + kk];
            }
        }
    }
    return w;
}

/// Depthwise kernel: GGUF ne `[K, 1, C]` (memory `w[c][k]`) → facade
/// `[channel, tap]` — the source layout is already channel-major.
fn loadDwW(ctx: *ExecContext, file: *const gguf.File, name: []const u8, k: usize, c: usize) !DwW {
    const t = try file.get(name);
    if (t.dims[0] != k or elemCount(t) != k * c) return Error.BadShape;
    const src = try f32Data(t);
    return DwW.fromSlice(ctx, .{ c, k }, src);
}

/// ConvTranspose1d weight: GGUF ne `[K, OC, IC]` (PyTorch `(IC, OC, K)`,
/// memory `w[ic][oc][k]`) → facade packed `[K*OC, IC]` with
/// `w2[(oc*K + k)*IC + ic]`.
fn loadTConvW(ctx: *ExecContext, file: *const gguf.File, name: []const u8, k: usize, ic: usize, oc: usize) !TConvW {
    const t = try file.get(name);
    if (t.dims[0] != k or t.dims[1] != oc or t.dims[2] != ic) return Error.BadShape;
    const src = try f32Data(t);
    var w = try TConvW.empty(ctx, .{ k * oc, ic });
    errdefer w.deinit();
    const dst = try w.data();
    for (0..ic) |i| {
        for (0..oc) |o| {
            for (0..k) |kk| {
                dst[(o * k + kk) * ic + i] = src[(i * oc + o) * k + kk];
            }
        }
    }
    return w;
}

/// Snake params folded at load: `a = exp(alpha)`, `inv_b = 1/(exp(beta)+1e-9)`.
fn loadSnakeFolded(ctx: *ExecContext, at: *const gguf.TensorInfo, bt: *const gguf.TensorInfo, c: usize) !struct { a: VecIn, ib: VecIn } {
    if (elemCount(at) != c or elemCount(bt) != c) return Error.BadShape;
    const alpha = try f32Data(at);
    const beta = try f32Data(bt);
    var av = try VecIn.empty(ctx, .{c});
    errdefer av.deinit();
    var ibv = try VecIn.empty(ctx, .{c});
    errdefer ibv.deinit();
    for (try av.data(), try ibv.data(), alpha, beta) |*a, *ib, al, be| {
        a.* = @exp(al);
        ib.* = 1.0 / (@exp(be) + 1e-9);
    }
    return .{ .a = av, .ib = ibv };
}

// --- weights ----------------------------------------------------------------

const TransformerLayer = struct {
    attn_norm: VecD,
    q: LinearWeight,
    k: LinearWeight,
    v: LinearWeight,
    o: LinearWeight,
    attn_scale: VecD,
    ffn_norm: VecD,
    gate: LinearWeight,
    up: LinearWeight,
    down: LinearWeight,
    mlp_scale: VecD,

    fn deinit(self: *TransformerLayer) void {
        inline for (@typeInfo(TransformerLayer).@"struct".fields) |f| @field(self, f.name).deinit();
        self.* = undefined;
    }
};

const ResUnit = struct {
    act1_a: VecIn,
    act1_ib: VecIn,
    conv1_w: ConvW, // k=7 dilated
    conv1_b: []const f32,
    act2_a: VecIn,
    act2_ib: VecIn,
    conv2_w: ConvW, // k=1
    conv2_b: []const f32,
    dilation: usize,

    fn deinit(self: *ResUnit) void {
        self.act1_a.deinit();
        self.act1_ib.deinit();
        self.conv1_w.deinit();
        self.act2_a.deinit();
        self.act2_ib.deinit();
        self.conv2_w.deinit();
        self.* = undefined;
    }
};

const DacBlock = struct {
    snake_a: VecIn,
    snake_ib: VecIn,
    tconv_w: TConvW,
    tconv_b: []const f32,
    taps: usize,
    stride: usize,
    out_ch: usize,
    res: [3]ResUnit,

    fn deinit(self: *DacBlock) void {
        self.snake_a.deinit();
        self.snake_ib.deinit();
        self.tconv_w.deinit();
        for (&self.res) |*r| r.deinit();
        self.* = undefined;
    }
};

const UpsampleBlock = struct {
    tconv_w: TConvW, // k=2 s=2, latent→latent
    tconv_b: []const f32,
    dw_w: DwW, // depthwise k=7 causal
    dw_b: []const f32,
    norm_w: VecD,
    norm_b: VecD,
    pw1: LinearWeight, // 1024→4096
    pw1_b: []const f32,
    pw2: LinearWeight, // 4096→1024
    pw2_b: []const f32,
    gamma: VecD,

    fn deinit(self: *UpsampleBlock) void {
        self.tconv_w.deinit();
        self.dw_w.deinit();
        self.norm_w.deinit();
        self.norm_b.deinit();
        self.pw1.deinit();
        self.pw2.deinit();
        self.gamma.deinit();
        self.* = undefined;
    }
};

pub const Decoder = struct {
    allocator: Allocator,
    cfg: Config,
    kv_head_for_head: []usize,

    vq_first_books: []Codebook,
    vq_first_proj: LinearWeight, // 256→512, no bias
    vq_rest_books: []Codebook,
    vq_rest_proj: LinearWeight,

    pre_conv_w: ConvW, // k=3, 512→1024
    pre_conv_b: []const f32,
    in_proj: LinearWeight, // 1024→512, with bias
    in_proj_b: []const f32,
    layers: []TransformerLayer,
    final_norm: VecD,
    out_proj: LinearWeight, // 512→1024, with bias
    out_proj_b: []const f32,

    upsample: [2]UpsampleBlock,

    dac_conv_pre_w: ConvW, // k=7, 1024→1536
    dac_conv_pre_b: []const f32,
    dac_blocks: [4]DacBlock,
    dac_snake_post_a: VecIn,
    dac_snake_post_ib: VecIn,
    dac_conv_post_w: ConvW, // k=7, 96→1
    dac_conv_post_b: []const f32,

    pub fn deinit(self: *Decoder) void {
        self.allocator.free(self.kv_head_for_head);
        for (self.vq_first_books) |*b| b.deinit();
        self.allocator.free(self.vq_first_books);
        for (self.vq_rest_books) |*b| b.deinit();
        self.allocator.free(self.vq_rest_books);
        self.vq_first_proj.deinit();
        self.vq_rest_proj.deinit();
        self.pre_conv_w.deinit();
        self.in_proj.deinit();
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        self.final_norm.deinit();
        self.out_proj.deinit();
        for (&self.upsample) |*u| u.deinit();
        self.dac_conv_pre_w.deinit();
        for (&self.dac_blocks) |*b| b.deinit();
        self.dac_snake_post_a.deinit();
        self.dac_snake_post_ib.deinit();
        self.dac_conv_post_w.deinit();
        self.* = undefined;
    }
};

const dac_strides = [4]usize{ 8, 5, 4, 3 };
const res_dilations = [3]usize{ 1, 3, 9 };

pub fn load(ctx: *ExecContext, file: *const gguf.File) !Decoder {
    const allocator = ctx.allocator;
    const cfg = try Config.fromGguf(file);
    var buf: [128]u8 = undefined;

    var dec: Decoder = undefined;
    dec.allocator = allocator;
    dec.cfg = cfg;

    dec.kv_head_for_head = try allocator.alloc(usize, cfg.n_heads);
    errdefer allocator.free(dec.kv_head_for_head);
    const group = cfg.n_heads / cfg.n_kv_heads;
    for (dec.kv_head_for_head, 0..) |*m, h| m.* = h / group;

    const n_acoustic = cfg.n_quantizers - cfg.n_semantic;
    dec.vq_first_books = try allocator.alloc(Codebook, cfg.n_semantic);
    var first_built: usize = 0;
    errdefer {
        for (dec.vq_first_books[0..first_built]) |*b| b.deinit();
        allocator.free(dec.vq_first_books);
    }
    for (dec.vq_first_books, 0..) |*b, i| {
        b.* = try loadNamed(Codebook, ctx, file, fmtName(&buf, "tok_dec.vq_first.{d}.codebook", .{i}), .{ cfg.codebook_size, cfg.cdim });
        first_built += 1;
    }
    dec.vq_rest_books = try allocator.alloc(Codebook, n_acoustic);
    var rest_built: usize = 0;
    errdefer {
        for (dec.vq_rest_books[0..rest_built]) |*b| b.deinit();
        allocator.free(dec.vq_rest_books);
    }
    for (dec.vq_rest_books, 0..) |*b, i| {
        b.* = try loadNamed(Codebook, ctx, file, fmtName(&buf, "tok_dec.vq_rest.{d}.codebook", .{i}), .{ cfg.codebook_size, cfg.cdim });
        rest_built += 1;
    }
    dec.vq_first_proj = try loadProj1x1(ctx, file, "tok_dec.vq_first.output_proj.weight", cfg.vq_hidden, cfg.cdim);
    errdefer dec.vq_first_proj.deinit();
    dec.vq_rest_proj = try loadProj1x1(ctx, file, "tok_dec.vq_rest.output_proj.weight", cfg.vq_hidden, cfg.cdim);
    errdefer dec.vq_rest_proj.deinit();

    dec.pre_conv_w = try loadConvW(ctx, file, "tok_dec.pre_conv.weight", 3, cfg.vq_hidden, cfg.latent);
    errdefer dec.pre_conv_w.deinit();
    dec.pre_conv_b = try loadSlice(file, "tok_dec.pre_conv.bias", cfg.latent);

    dec.in_proj = try loadLin(ctx, file, "tok_dec.pre_tfm.input_proj.weight", cfg.hidden, cfg.latent);
    errdefer dec.in_proj.deinit();
    dec.in_proj_b = try loadSlice(file, "tok_dec.pre_tfm.input_proj.bias", cfg.hidden);

    dec.layers = try allocator.alloc(TransformerLayer, cfg.n_layers);
    var layers_built: usize = 0;
    errdefer {
        for (dec.layers[0..layers_built]) |*l| l.deinit();
        allocator.free(dec.layers);
    }
    const qkv = cfg.n_heads * cfg.head_dim;
    for (dec.layers, 0..) |*l, i| {
        l.attn_norm = try loadVec(VecD, ctx, file, fmtName(&buf, "tok_dec.pre_tfm.blk.{d}.attn_norm.weight", .{i}), cfg.hidden);
        l.q = try loadLin(ctx, file, fmtName(&buf, "tok_dec.pre_tfm.blk.{d}.attn_q.weight", .{i}), qkv, cfg.hidden);
        l.k = try loadLin(ctx, file, fmtName(&buf, "tok_dec.pre_tfm.blk.{d}.attn_k.weight", .{i}), qkv, cfg.hidden);
        l.v = try loadLin(ctx, file, fmtName(&buf, "tok_dec.pre_tfm.blk.{d}.attn_v.weight", .{i}), qkv, cfg.hidden);
        l.o = try loadLin(ctx, file, fmtName(&buf, "tok_dec.pre_tfm.blk.{d}.attn_output.weight", .{i}), cfg.hidden, qkv);
        l.attn_scale = try loadVec(VecD, ctx, file, fmtName(&buf, "tok_dec.pre_tfm.blk.{d}.attn_scale", .{i}), cfg.hidden);
        l.ffn_norm = try loadVec(VecD, ctx, file, fmtName(&buf, "tok_dec.pre_tfm.blk.{d}.ffn_norm.weight", .{i}), cfg.hidden);
        l.gate = try loadLin(ctx, file, fmtName(&buf, "tok_dec.pre_tfm.blk.{d}.ffn_gate.weight", .{i}), cfg.ffn, cfg.hidden);
        l.up = try loadLin(ctx, file, fmtName(&buf, "tok_dec.pre_tfm.blk.{d}.ffn_up.weight", .{i}), cfg.ffn, cfg.hidden);
        l.down = try loadLin(ctx, file, fmtName(&buf, "tok_dec.pre_tfm.blk.{d}.ffn_down.weight", .{i}), cfg.hidden, cfg.ffn);
        l.mlp_scale = try loadVec(VecD, ctx, file, fmtName(&buf, "tok_dec.pre_tfm.blk.{d}.ffn_scale", .{i}), cfg.hidden);
        layers_built += 1;
    }
    dec.final_norm = try loadVec(VecD, ctx, file, "tok_dec.pre_tfm.norm.weight", cfg.hidden);
    errdefer dec.final_norm.deinit();
    dec.out_proj = try loadLin(ctx, file, "tok_dec.pre_tfm.output_proj.weight", cfg.latent, cfg.hidden);
    errdefer dec.out_proj.deinit();
    dec.out_proj_b = try loadSlice(file, "tok_dec.pre_tfm.output_proj.bias", cfg.latent);

    var upsample_built: usize = 0;
    errdefer for (dec.upsample[0..upsample_built]) |*u| u.deinit();
    for (&dec.upsample, 0..) |*u, i| {
        u.tconv_w = try loadTConvW(ctx, file, fmtName(&buf, "tok_dec.upsample.{d}.conv.weight", .{i}), 2, cfg.latent, cfg.latent);
        u.tconv_b = try loadSlice(file, fmtName(&buf, "tok_dec.upsample.{d}.conv.bias", .{i}), cfg.latent);
        u.dw_w = try loadDwW(ctx, file, fmtName(&buf, "tok_dec.upsample.{d}.dwconv.weight", .{i}), 7, cfg.latent);
        u.dw_b = try loadSlice(file, fmtName(&buf, "tok_dec.upsample.{d}.dwconv.bias", .{i}), cfg.latent);
        u.norm_w = try loadVec(VecD, ctx, file, fmtName(&buf, "tok_dec.upsample.{d}.norm.weight", .{i}), cfg.latent);
        u.norm_b = try loadVec(VecD, ctx, file, fmtName(&buf, "tok_dec.upsample.{d}.norm.bias", .{i}), cfg.latent);
        u.pw1 = try loadLin(ctx, file, fmtName(&buf, "tok_dec.upsample.{d}.pwconv1.weight", .{i}), cfg.latent * 4, cfg.latent);
        u.pw1_b = try loadSlice(file, fmtName(&buf, "tok_dec.upsample.{d}.pwconv1.bias", .{i}), cfg.latent * 4);
        u.pw2 = try loadLin(ctx, file, fmtName(&buf, "tok_dec.upsample.{d}.pwconv2.weight", .{i}), cfg.latent, cfg.latent * 4);
        u.pw2_b = try loadSlice(file, fmtName(&buf, "tok_dec.upsample.{d}.pwconv2.bias", .{i}), cfg.latent);
        u.gamma = try loadVec(VecD, ctx, file, fmtName(&buf, "tok_dec.upsample.{d}.gamma", .{i}), cfg.latent);
        upsample_built += 1;
    }

    dec.dac_conv_pre_w = try loadConvW(ctx, file, "tok_dec.dec.0.conv.weight", 7, cfg.latent, cfg.decoder_dim);
    errdefer dec.dac_conv_pre_w.deinit();
    dec.dac_conv_pre_b = try loadSlice(file, "tok_dec.dec.0.conv.bias", cfg.decoder_dim);
    var ch_in = cfg.decoder_dim;
    var dac_built: usize = 0;
    errdefer for (dec.dac_blocks[0..dac_built]) |*blk| blk.deinit();
    for (&dec.dac_blocks, 0..) |*blk, bi| {
        const stride = dac_strides[bi];
        const ch_out = ch_in / 2;
        const base = bi + 1;
        const sn_at = try file.get(fmtName(&buf, "tok_dec.dec.{d}.snake.alpha", .{base}));
        const sn_bt = try file.get(fmtName(&buf, "tok_dec.dec.{d}.snake.beta", .{base}));
        const sn = try loadSnakeFolded(ctx, sn_at, sn_bt, ch_in);
        blk.snake_a = sn.a;
        blk.snake_ib = sn.ib;
        blk.tconv_w = try loadTConvW(ctx, file, fmtName(&buf, "tok_dec.dec.{d}.conv_t.weight", .{base}), 2 * stride, ch_in, ch_out);
        blk.tconv_b = try loadSlice(file, fmtName(&buf, "tok_dec.dec.{d}.conv_t.bias", .{base}), ch_out);
        blk.taps = 2 * stride;
        blk.stride = stride;
        blk.out_ch = ch_out;
        for (&blk.res, 0..) |*r, ri| {
            const a1_at = try file.get(fmtName(&buf, "tok_dec.dec.{d}.res.{d}.act1.alpha", .{ base, ri }));
            const a1_bt = try file.get(fmtName(&buf, "tok_dec.dec.{d}.res.{d}.act1.beta", .{ base, ri }));
            const a1 = try loadSnakeFolded(ctx, a1_at, a1_bt, ch_out);
            r.act1_a = a1.a;
            r.act1_ib = a1.ib;
            r.conv1_w = try loadConvW(ctx, file, fmtName(&buf, "tok_dec.dec.{d}.res.{d}.conv1.weight", .{ base, ri }), 7, ch_out, ch_out);
            r.conv1_b = try loadSlice(file, fmtName(&buf, "tok_dec.dec.{d}.res.{d}.conv1.bias", .{ base, ri }), ch_out);
            const a2_at = try file.get(fmtName(&buf, "tok_dec.dec.{d}.res.{d}.act2.alpha", .{ base, ri }));
            const a2_bt = try file.get(fmtName(&buf, "tok_dec.dec.{d}.res.{d}.act2.beta", .{ base, ri }));
            const a2 = try loadSnakeFolded(ctx, a2_at, a2_bt, ch_out);
            r.act2_a = a2.a;
            r.act2_ib = a2.ib;
            r.conv2_w = try loadConvW(ctx, file, fmtName(&buf, "tok_dec.dec.{d}.res.{d}.conv2.weight", .{ base, ri }), 1, ch_out, ch_out);
            r.conv2_b = try loadSlice(file, fmtName(&buf, "tok_dec.dec.{d}.res.{d}.conv2.bias", .{ base, ri }), ch_out);
            r.dilation = res_dilations[ri];
        }
        ch_in = ch_out;
        dac_built += 1;
    }
    const sp = try loadSnakeFolded(ctx, try file.get("tok_dec.dec.5.snake.alpha"), try file.get("tok_dec.dec.5.snake.beta"), ch_in);
    dec.dac_snake_post_a = sp.a;
    dec.dac_snake_post_ib = sp.ib;
    errdefer {
        dec.dac_snake_post_a.deinit();
        dec.dac_snake_post_ib.deinit();
    }
    dec.dac_conv_post_w = try loadConvW(ctx, file, "tok_dec.dec.6.conv.weight", 7, ch_in, 1);
    dec.dac_conv_post_b = try loadSlice(file, "tok_dec.dec.6.conv.bias", 1);

    return dec;
}

// --- decode -----------------------------------------------------------------

fn convBias(ctx: *ExecContext, x: *const Act, w: *const ConvW, bias: []const f32, dilation: usize) !Act {
    var y = try x.causalConv1d(ctx, .seq, .in, .tap, .out, w, dilation, null);
    defer y.deinit();
    var yt = try y.withTags(ctx, .{ .seq, .in });
    errdefer yt.deinit();
    try yt.addAxisVectorInPlace(ctx, bias, .in);
    return yt;
}

/// CausalTransConv1d: full transposed conv then right-trim (taps − stride) rows.
fn transConvBias(ctx: *ExecContext, x: *const Act, w: *const TConvW, bias: []const f32, oc: usize, taps: usize, stride: usize) !Act {
    var full = try x.convTranspose1d(ctx, .seq, .in, .kout, .out, w, null, oc, taps, stride, 0, 0);
    defer full.deinit();
    const keep = full.dim(.seq) - (taps - stride);
    var trimmed = try full.narrow(ctx, .seq, 0, keep);
    defer trimmed.deinit();
    var tt = try trimmed.withTags(ctx, .{ .seq, .in });
    errdefer tt.deinit();
    try tt.addAxisVectorInPlace(ctx, bias, .in);
    return tt;
}

/// RVQ dequant: codes `[K, T]` row-major (T fastest) → `[T, 512]`.
fn rvqDecode(ctx: *ExecContext, dec: *const Decoder, codes: []const i32, t: usize) !Act {
    const cfg = &dec.cfg;
    const indices = try ctx.allocator.alloc(usize, t);
    defer ctx.allocator.free(indices);

    var group_out: ?Act = null;
    errdefer if (group_out) |*g| g.deinit();

    for (0..2) |side| {
        const books = if (side == 0) dec.vq_first_books else dec.vq_rest_books;
        const proj = if (side == 0) &dec.vq_first_proj else &dec.vq_rest_proj;
        const k0: usize = if (side == 0) 0 else cfg.n_semantic;

        var sum: ?fucina.Tensor(.{ .seq, .cdim }) = null;
        defer if (sum) |*s| s.deinit();
        for (books, 0..) |*book, ki| {
            for (indices, codes[(k0 + ki) * t ..][0..t]) |*dst, code| {
                if (code < 0 or @as(usize, @intCast(code)) >= cfg.codebook_size) return Error.CodeOutOfRange;
                dst.* = @intCast(code);
            }
            var rows = try book.gather(ctx, .code, indices, .seq); // [T, 256]
            if (sum) |*s| {
                defer rows.deinit();
                try s.addScaledInPlace(ctx, &rows, 1.0);
            } else sum = rows;
        }
        var projected = try proj.linearSeq(ctx, &sum.?, .cdim, .d); // [T, 512]
        defer projected.deinit();
        var pr = try projected.withTags(ctx, .{ .seq, .in });
        if (group_out) |*g| {
            defer pr.deinit();
            try g.addScaledInPlace(ctx, &pr, 1.0);
        } else group_out = pr;
    }
    return group_out.?;
}

fn transformerForward(ctx: *ExecContext, dec: *const Decoder, x: *const Act) !Act {
    const cfg = &dec.cfg;
    var cur = try dec.in_proj.linearSeq(ctx, x, .in, .d);
    errdefer cur.deinit();
    try cur.addAxisVectorInPlace(ctx, dec.in_proj_b, .d);

    const t = cur.dim(.seq);
    var rope_table = try ctx.prepareRopeTableRange(.{ .len = t }, cfg.head_dim, cfg.rope_theta, false);
    defer rope_table.deinit();

    for (dec.layers) |*l| {
        var attn_in = try cur.rmsNormMul(ctx, .d, &l.attn_norm, cfg.rms_eps);
        defer attn_in.deinit();

        var q = try l.q.linearSeq(ctx, &attn_in, .d, .qkv);
        defer q.deinit();
        var k = try l.k.linearSeq(ctx, &attn_in, .d, .qkv);
        defer k.deinit();
        var v = try l.v.linearSeq(ctx, &attn_in, .d, .qkv);
        defer v.deinit();

        var q3 = try q.split(ctx, .qkv, .{ .head, .d }, .{ cfg.n_heads, cfg.head_dim });
        defer q3.deinit();
        var k3 = try k.split(ctx, .qkv, .{ .kv_head, .d }, .{ cfg.n_kv_heads, cfg.head_dim });
        defer k3.deinit();
        var v3 = try v.split(ctx, .qkv, .{ .kv_head, .d }, .{ cfg.n_kv_heads, cfg.head_dim });
        defer v3.deinit();

        var q_rope = try q3.rope(ctx, .seq, .d, &rope_table, .half);
        defer q_rope.deinit();
        var k_rope = try k3.rope(ctx, .seq, .d, &rope_table, .half);
        defer k_rope.deinit();

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        var attn = try q_rope.groupedAttention(ctx, &k_rope, &v3, dec.kv_head_for_head, .attn, scale, .{ .window = cfg.sliding_window });
        defer attn.deinit();

        var o = try l.o.linearSeq(ctx, &attn, .attn, .d);
        defer o.deinit();
        var scaled = try o.mul(ctx, &l.attn_scale);
        defer scaled.deinit();
        const res1 = try cur.add(ctx, &scaled);
        cur.deinit();
        cur = res1;

        var ffn_in = try cur.rmsNormMul(ctx, .d, &l.ffn_norm, cfg.rms_eps);
        defer ffn_in.deinit();
        var gate = try l.gate.linearSeq(ctx, &ffn_in, .d, .ff);
        defer gate.deinit();
        var up = try l.up.linearSeq(ctx, &ffn_in, .d, .ff);
        defer up.deinit();
        var gated = try up.swiglu(ctx, &gate);
        defer gated.deinit();
        var down = try l.down.linearSeq(ctx, &gated, .ff, .d);
        defer down.deinit();
        var mscaled = try down.mul(ctx, &l.mlp_scale);
        defer mscaled.deinit();
        const res2 = try cur.add(ctx, &mscaled);
        cur.deinit();
        cur = res2;
    }

    const normed = try cur.rmsNormMul(ctx, .d, &dec.final_norm, cfg.rms_eps);
    cur.deinit();
    cur = normed;
    defer cur.deinit();
    var out = try dec.out_proj.linearSeq(ctx, &cur, .d, .in);
    errdefer out.deinit();
    try out.addAxisVectorInPlace(ctx, dec.out_proj_b, .in);
    return out;
}

fn convNext(ctx: *ExecContext, u: *const UpsampleBlock, x: *const Act) !Act {
    var dw = try x.causalDepthwiseConv1d(ctx, .seq, .in, .tap, &u.dw_w, 1, null);
    defer dw.deinit();
    try dw.addAxisVectorInPlace(ctx, u.dw_b, .in);
    var dwd = try dw.withTags(ctx, .{ .seq, .d });
    defer dwd.deinit();
    var normed = try dwd.layerNorm(ctx, .d, 1e-6, .{ .weight = &u.norm_w, .bias = &u.norm_b });
    defer normed.deinit();
    var p1 = try u.pw1.linearSeq(ctx, &normed, .d, .ff);
    defer p1.deinit();
    try p1.addAxisVectorInPlace(ctx, u.pw1_b, .ff);
    var act = try p1.gelu(ctx);
    defer act.deinit();
    var p2 = try u.pw2.linearSeq(ctx, &act, .ff, .d);
    defer p2.deinit();
    try p2.addAxisVectorInPlace(ctx, u.pw2_b, .d);
    var scaled = try p2.mul(ctx, &u.gamma);
    defer scaled.deinit();
    var xin = try x.withTags(ctx, .{ .seq, .d });
    defer xin.deinit();
    var summed = try xin.add(ctx, &scaled);
    defer summed.deinit();
    return summed.withTags(ctx, .{ .seq, .in });
}

fn snakeAct(ctx: *ExecContext, x: *const Act, a: *const VecIn, ib: *const VecIn) !Act {
    return x.snake(ctx, .in, a, ib);
}

/// Per-stage taps for parity debugging (all `[T, C]` row-major copies).
pub const Taps = struct {
    allocator: Allocator,
    rvq: ?[]f32 = null,
    preconv: ?[]f32 = null,
    tfm: ?[]f32 = null,
    up: ?[]f32 = null,

    pub fn deinit(self: *Taps) void {
        if (self.rvq) |b| self.allocator.free(b);
        if (self.preconv) |b| self.allocator.free(b);
        if (self.tfm) |b| self.allocator.free(b);
        if (self.up) |b| self.allocator.free(b);
        self.* = undefined;
    }
};

fn capture(ctx: *ExecContext, slot: *?[]f32, x: *const Act) !void {
    slot.* = try ctx.allocator.dupe(f32, try x.dataConst());
}

/// Decode `codes` (`[K=16, T]` row-major, T fastest) → `[T*1920]` samples,
/// clamped to ±1. Caller frees.
pub fn decode(ctx: *ExecContext, dec: *const Decoder, codes: []const i32, t: usize) ![]f32 {
    return decodeWithTaps(ctx, dec, codes, t, null);
}

pub fn decodeWithTaps(ctx: *ExecContext, dec: *const Decoder, codes: []const i32, t: usize, taps: ?*Taps) ![]f32 {
    if (t == 0 or codes.len != dec.cfg.n_quantizers * t) return Error.InvalidCodes;

    var h512 = try rvqDecode(ctx, dec, codes, t);
    defer h512.deinit();
    if (taps) |tp| try capture(ctx, &tp.rvq, &h512);

    var trunk = try convBias(ctx, &h512, &dec.pre_conv_w, dec.pre_conv_b, 1);
    defer trunk.deinit();
    if (taps) |tp| try capture(ctx, &tp.preconv, &trunk);

    var up = try transformerForward(ctx, dec, &trunk);
    defer up.deinit();
    if (taps) |tp| try capture(ctx, &tp.tfm, &up);
    for (&dec.upsample) |*u| {
        var tc = try transConvBias(ctx, &up, &u.tconv_w, u.tconv_b, dec.cfg.latent, 2, 2);
        defer tc.deinit();
        const nx = try convNext(ctx, u, &tc);
        up.deinit();
        up = nx;
    }
    if (taps) |tp| try capture(ctx, &tp.up, &up);

    var d = try convBias(ctx, &up, &dec.dac_conv_pre_w, dec.dac_conv_pre_b, 1);
    defer d.deinit();
    for (&dec.dac_blocks) |*blk| {
        var sn = try snakeAct(ctx, &d, &blk.snake_a, &blk.snake_ib);
        defer sn.deinit();
        const tc = try transConvBias(ctx, &sn, &blk.tconv_w, blk.tconv_b, blk.out_ch, blk.taps, blk.stride);
        d.deinit();
        d = tc;
        for (&blk.res) |*r| {
            var a1 = try snakeAct(ctx, &d, &r.act1_a, &r.act1_ib);
            defer a1.deinit();
            var c1 = try convBias(ctx, &a1, &r.conv1_w, r.conv1_b, r.dilation);
            defer c1.deinit();
            var a2 = try snakeAct(ctx, &c1, &r.act2_a, &r.act2_ib);
            defer a2.deinit();
            var c2 = try convBias(ctx, &a2, &r.conv2_w, r.conv2_b, 1);
            defer c2.deinit();
            const summed = try d.add(ctx, &c2);
            d.deinit();
            d = summed;
        }
    }
    var spost = try snakeAct(ctx, &d, &dec.dac_snake_post_a, &dec.dac_snake_post_ib);
    defer spost.deinit();
    var wave = try convBias(ctx, &spost, &dec.dac_conv_post_w, dec.dac_conv_post_b, 1);
    defer wave.deinit();

    const rows = try wave.dataConst();
    const out = try ctx.allocator.alloc(f32, rows.len);
    for (out, rows) |*o, s| o.* = std.math.clamp(s, -1.0, 1.0);
    return out;
}

/// Chunked decode (the reference `chunked_decode`): re-decode each chunk with
/// up to `left_ctx` context frames prepended, drop `ctx_frames * 1920` head
/// samples. Reference defaults: 300-frame chunks, 25-frame context.
pub fn decodeChunked(ctx: *ExecContext, dec: *const Decoder, codes: []const i32, t: usize, chunk_frames_in: usize, left_ctx: usize, out: *std.ArrayList(f32)) !void {
    const chunk_frames = @max(1, chunk_frames_in);
    const kq = dec.cfg.n_quantizers;
    if (t == 0 or codes.len != kq * t) return Error.InvalidCodes;
    const scratch = try ctx.allocator.alloc(i32, kq * @min(t, chunk_frames + left_ctx));
    defer ctx.allocator.free(scratch);
    var start: usize = 0;
    while (start < t) {
        const end = @min(start + chunk_frames, t);
        const ctx_frames = @min(left_ctx, start);
        const s0 = start - ctx_frames;
        const span = end - s0;
        for (0..kq) |k| {
            @memcpy(scratch[k * span ..][0..span], codes[k * t + s0 ..][0..span]);
        }
        const audio = try decode(ctx, dec, scratch[0 .. kq * span], span);
        defer ctx.allocator.free(audio);
        try out.appendSlice(ctx.allocator, audio[ctx_frames * hop_length ..]);
        start = end;
    }
}


// --- exact streaming decode --------------------------------------------------
//
// Every decoder op is causal and time-invariant, so streaming needs no new
// kernels: each conv-like stage keeps a host tail of its own INPUT columns,
// runs the SAME facade op on [tail ++ new], and drops the tail-region
// outputs; the transformer keeps a windowed KV with ABSOLUTE positions.
// Result: bit-path-identical to whole-clip `decode` (pinned by test), with
// no left-context recompute — the chunked decoder re-decodes 25 frames per
// 12 emitted (~3x); this decodes each frame once.

/// Host tail of a stage's input rows ([cols, C] row-major).
const TailBuf = struct {
    buf: []f32,
    cap_cols: usize,
    have: usize = 0,
    c: usize,

    fn init(allocator: Allocator, cap_cols: usize, c: usize) !TailBuf {
        return .{ .buf = try allocator.alloc(f32, cap_cols * c), .cap_cols = cap_cols, .c = c };
    }

    fn deinit(self: *TailBuf, allocator: Allocator) void {
        allocator.free(self.buf);
    }

    /// [tail ++ x] as a fresh tensor; returns it plus the drop count.
    fn concat(self: *TailBuf, ctx: *ExecContext, x: *const Act) !struct { t: Act, drop: usize } {
        const xd = try x.dataConst();
        const t_new = x.dim(.seq);
        const total = self.have + t_new;
        var t = try Act.empty(ctx, .{ total, self.c });
        errdefer t.deinit();
        const td = try t.data();
        @memcpy(td[0 .. self.have * self.c], self.buf[0 .. self.have * self.c]);
        @memcpy(td[self.have * self.c ..][0 .. t_new * self.c], xd);
        // save the new tail from the composite input
        const keep = @min(self.cap_cols, total);
        @memcpy(self.buf[0 .. keep * self.c], td[(total - keep) * self.c ..][0 .. keep * self.c]);
        const drop = self.have;
        self.have = keep;
        return .{ .t = t, .drop = drop };
    }

    fn reset(self: *TailBuf) void {
        self.have = 0;
    }
};

fn dropRows(ctx: *ExecContext, y: *Act, rows: usize) !Act {
    var kept = try y.narrow(ctx, .seq, rows, y.dim(.seq) - rows);
    defer kept.deinit();
    return kept.withTags(ctx, .{ .seq, .in });
}

/// Streaming decoder session: feed codec frames incrementally, get
/// `frames * 1920` samples per step, exactly equal to whole-clip decode.
pub const Streaming = struct {
    allocator: Allocator,
    dec: *const Decoder,
    // transformer KV: roped-K/V host rows per layer, windowed (72) with
    // compaction; absolute positions survive compaction.
    kv_k: [][]f32,
    kv_v: [][]f32,
    kv_len: usize = 0,
    pos_base: usize = 0, // absolute position of kv row 0
    // per-stage input tails, in decodeWithTaps order
    t_pre: TailBuf,
    t_next: [2]TailBuf, // convnext depthwise input (whole block input)
    t_dacpre: TailBuf,
    t_tconv: [4]TailBuf,
    t_res: [4][3]TailBuf,
    t_post: TailBuf,

    const kv_cap = 256; // >= window(72) + max step; compacts to the window

    pub fn init(allocator: Allocator, dec: *const Decoder) !Streaming {
        const cfg = &dec.cfg;
        const kv_dim = cfg.n_kv_heads * cfg.head_dim;
        const kv_k = try allocator.alloc([]f32, cfg.n_layers);
        const kv_v = try allocator.alloc([]f32, cfg.n_layers);
        for (kv_k) |*b| b.* = try allocator.alloc(f32, kv_cap * kv_dim);
        for (kv_v) |*b| b.* = try allocator.alloc(f32, kv_cap * kv_dim);

        var self = Streaming{
            .allocator = allocator,
            .dec = dec,
            .kv_k = kv_k,
            .kv_v = kv_v,
            .t_pre = try TailBuf.init(allocator, dec.pre_conv_w.dim(.tap) - 1, cfg.hidden),
            .t_next = undefined,
            .t_dacpre = try TailBuf.init(allocator, dec.dac_conv_pre_w.dim(.tap) - 1, cfg.latent),
            .t_tconv = undefined,
            .t_res = undefined,
            .t_post = try TailBuf.init(allocator, dec.dac_conv_post_w.dim(.tap) - 1, dec.dac_blocks[3].out_ch),
        };
        for (&self.t_next, 0..) |*tb, i| {
            tb.* = try TailBuf.init(allocator, dec.upsample[i].dw_w.dim(.tap) - 1, cfg.latent);
        }
        var in_ch = cfg.decoder_dim;
        for (&self.t_tconv, 0..) |*tb, bi| {
            const blk = &dec.dac_blocks[bi];
            const tail_in = if (blk.taps > blk.stride) (blk.taps - blk.stride + blk.stride - 1) / blk.stride else 0;
            tb.* = try TailBuf.init(allocator, tail_in, in_ch);
            for (&self.t_res[bi], 0..) |*rb, ri| {
                const r = &blk.res[ri];
                rb.* = try TailBuf.init(allocator, (r.conv1_w.dim(.tap) - 1) * r.dilation, blk.out_ch);
            }
            in_ch = blk.out_ch;
        }
        return self;
    }

    pub fn deinit(self: *Streaming) void {
        for (self.kv_k) |b| self.allocator.free(b);
        for (self.kv_v) |b| self.allocator.free(b);
        self.allocator.free(self.kv_k);
        self.allocator.free(self.kv_v);
        self.t_pre.deinit(self.allocator);
        for (&self.t_next) |*tb| tb.deinit(self.allocator);
        self.t_dacpre.deinit(self.allocator);
        for (&self.t_tconv) |*tb| tb.deinit(self.allocator);
        for (&self.t_res) |*row| for (row) |*rb| rb.deinit(self.allocator);
        self.t_post.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn reset(self: *Streaming) void {
        self.kv_len = 0;
        self.pos_base = 0;
        self.t_pre.reset();
        for (&self.t_next) |*tb| tb.reset();
        self.t_dacpre.reset();
        for (&self.t_tconv) |*tb| tb.reset();
        for (&self.t_res) |*row| for (row) |*rb| rb.reset();
        self.t_post.reset();
    }

    fn tailedConv(self: *Streaming, ctx: *ExecContext, tb: *TailBuf, x: *const Act, w: *const ConvW, bias: []const f32, dilation: usize) !Act {
        _ = self;
        var cc = try tb.concat(ctx, x);
        defer cc.t.deinit();
        var y = try convBias(ctx, &cc.t, w, bias, dilation);
        defer y.deinit();
        return dropRows(ctx, &y, cc.drop);
    }

    fn tailedTConv(self: *Streaming, ctx: *ExecContext, tb: *TailBuf, x: *const Act, w: *const TConvW, bias: []const f32, oc: usize, taps: usize, stride: usize) !Act {
        _ = self;
        var cc = try tb.concat(ctx, x);
        defer cc.t.deinit();
        var y = try transConvBias(ctx, &cc.t, w, bias, oc, taps, stride);
        defer y.deinit();
        return dropRows(ctx, &y, cc.drop * stride);
    }

    /// Transformer step with absolute positions + windowed host KV.
    fn transformerStep(self: *Streaming, ctx: *ExecContext, x: *const Act) !Act {
        const dec = self.dec;
        const cfg = &dec.cfg;
        const kv_dim = cfg.n_kv_heads * cfg.head_dim;

        var cur = try dec.in_proj.linearSeq(ctx, x, .in, .d);
        errdefer cur.deinit();
        try cur.addAxisVectorInPlace(ctx, dec.in_proj_b, .d);
        const t = cur.dim(.seq);

        // compact the window before appending (keep >= window rows)
        if (self.kv_len + t > kv_cap) {
            const keep = cfg.sliding_window;
            const drop = self.kv_len - keep;
            for (0..cfg.n_layers) |li| {
                std.mem.copyForwards(f32, self.kv_k[li][0 .. keep * kv_dim], self.kv_k[li][drop * kv_dim ..][0 .. keep * kv_dim]);
                std.mem.copyForwards(f32, self.kv_v[li][0 .. keep * kv_dim], self.kv_v[li][drop * kv_dim ..][0 .. keep * kv_dim]);
            }
            self.pos_base += drop;
            self.kv_len = keep;
        }

        const abs0 = self.pos_base + self.kv_len;
        var rope_table = try ctx.prepareRopeTableRange(.{ .origin = @intCast(abs0), .len = t }, cfg.head_dim, cfg.rope_theta, false);
        defer rope_table.deinit();

        for (dec.layers, 0..) |*l, li| {
            var attn_in = try cur.rmsNormMul(ctx, .d, &l.attn_norm, cfg.rms_eps);
            defer attn_in.deinit();
            var q = try l.q.linearSeq(ctx, &attn_in, .d, .qkv);
            defer q.deinit();
            var k = try l.k.linearSeq(ctx, &attn_in, .d, .qkv);
            defer k.deinit();
            var v = try l.v.linearSeq(ctx, &attn_in, .d, .qkv);
            defer v.deinit();
            var q3 = try q.split(ctx, .qkv, .{ .head, .d }, .{ cfg.n_heads, cfg.head_dim });
            defer q3.deinit();
            var k3 = try k.split(ctx, .qkv, .{ .kv_head, .d }, .{ cfg.n_kv_heads, cfg.head_dim });
            defer k3.deinit();
            var v3 = try v.split(ctx, .qkv, .{ .kv_head, .d }, .{ cfg.n_kv_heads, cfg.head_dim });
            defer v3.deinit();
            var q_rope = try q3.rope(ctx, .seq, .d, &rope_table, .half);
            defer q_rope.deinit();
            var k_rope = try k3.rope(ctx, .seq, .d, &rope_table, .half);
            defer k_rope.deinit();
            {
                const kd = try k_rope.dataConst();
                const vd = try v3.dataConst();
                @memcpy(self.kv_k[li][self.kv_len * kv_dim ..][0 .. t * kv_dim], kd);
                @memcpy(self.kv_v[li][self.kv_len * kv_dim ..][0 .. t * kv_dim], vd);
            }
            const cached = self.kv_len + t;
            // Borrowed ring views (rows appended above; attention reads only).
            var k_all = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromBorrowedConstSlice(ctx, .{ cached, cfg.n_kv_heads, cfg.head_dim }, self.kv_k[li][0 .. cached * kv_dim]);
            defer k_all.deinit();
            var v_all = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromBorrowedConstSlice(ctx, .{ cached, cfg.n_kv_heads, cfg.head_dim }, self.kv_v[li][0 .. cached * kv_dim]);
            defer v_all.deinit();
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
            var attn = try q_rope.groupedAttention(ctx, &k_all, &v_all, dec.kv_head_for_head, .attn, scale, .{ .window = cfg.sliding_window });
            defer attn.deinit();
            var o = try l.o.linearSeq(ctx, &attn, .attn, .d);
            defer o.deinit();
            var scaled = try o.mul(ctx, &l.attn_scale);
            defer scaled.deinit();
            const res1 = try cur.add(ctx, &scaled);
            cur.deinit();
            cur = res1;

            var ffn_in = try cur.rmsNormMul(ctx, .d, &l.ffn_norm, cfg.rms_eps);
            defer ffn_in.deinit();
            var gate = try l.gate.linearSeq(ctx, &ffn_in, .d, .ff);
            defer gate.deinit();
            var up = try l.up.linearSeq(ctx, &ffn_in, .d, .ff);
            defer up.deinit();
            var gated = try up.swiglu(ctx, &gate);
            defer gated.deinit();
            var down = try l.down.linearSeq(ctx, &gated, .ff, .d);
            defer down.deinit();
            var dscaled = try down.mul(ctx, &l.mlp_scale);
            defer dscaled.deinit();
            const res2 = try cur.add(ctx, &dscaled);
            cur.deinit();
            cur = res2;
        }
        self.kv_len += t;

        var normed = try cur.rmsNormMul(ctx, .d, &dec.final_norm, cfg.rms_eps);
        cur.deinit();
        defer normed.deinit();
        var out = try dec.out_proj.linearSeq(ctx, &normed, .d, .in);
        errdefer out.deinit();
        try out.addAxisVectorInPlace(ctx, dec.out_proj_b, .in);
        return out;
    }

    /// Decode `t` NEW frames; returns `t * 1920` samples (caller frees).
    pub fn step(self: *Streaming, ctx: *ExecContext, codes: []const i32, t: usize) ![]f32 {
        const dec = self.dec;
        var h512 = try rvqDecode(ctx, dec, codes, t);
        defer h512.deinit();
        var trunk = try self.tailedConv(ctx, &self.t_pre, &h512, &dec.pre_conv_w, dec.pre_conv_b, 1);
        defer trunk.deinit();

        var up = try self.transformerStep(ctx, &trunk);
        errdefer up.deinit();
        for (&dec.upsample, 0..) |*u, ui| {
            // k2 s2 tconv has no cross-input overlap: stateless
            var tc = try transConvBias(ctx, &up, &u.tconv_w, u.tconv_b, dec.cfg.latent, 2, 2);
            defer tc.deinit();
            var cc = try self.t_next[ui].concat(ctx, &tc);
            defer cc.t.deinit();
            var nx = try convNext(ctx, u, &cc.t);
            defer nx.deinit();
            const kept = try dropRows(ctx, &nx, cc.drop);
            up.deinit();
            up = kept;
        }

        var d = try self.tailedConv(ctx, &self.t_dacpre, &up, &dec.dac_conv_pre_w, dec.dac_conv_pre_b, 1);
        up.deinit();
        errdefer d.deinit();
        for (&dec.dac_blocks, 0..) |*blk, bi| {
            var sn = try snakeAct(ctx, &d, &blk.snake_a, &blk.snake_ib);
            defer sn.deinit();
            const tc = try self.tailedTConv(ctx, &self.t_tconv[bi], &sn, &blk.tconv_w, blk.tconv_b, blk.out_ch, blk.taps, blk.stride);
            d.deinit();
            d = tc;
            for (&blk.res, 0..) |*r, ri| {
                var a1 = try snakeAct(ctx, &d, &r.act1_a, &r.act1_ib);
                defer a1.deinit();
                var c1 = try self.tailedConv(ctx, &self.t_res[bi][ri], &a1, &r.conv1_w, r.conv1_b, r.dilation);
                defer c1.deinit();
                var a2 = try snakeAct(ctx, &c1, &r.act2_a, &r.act2_ib);
                defer a2.deinit();
                var c2 = try convBias(ctx, &a2, &r.conv2_w, r.conv2_b, 1);
                defer c2.deinit();
                // the tailed conv already emits only the NEW region: residual
                // adds against the equally-new `d` rows
                var d_new = try d.narrow(ctx, .seq, d.dim(.seq) - c2.dim(.seq), c2.dim(.seq));
                defer d_new.deinit();
                const summed = try d_new.add(ctx, &c2);
                d.deinit();
                d = summed;
            }
        }
        var spost = try snakeAct(ctx, &d, &dec.dac_snake_post_a, &dec.dac_snake_post_ib);
        d.deinit();
        defer spost.deinit();
        var wave = try self.tailedConv(ctx, &self.t_post, &spost, &dec.dac_conv_post_w, dec.dac_conv_post_b, 1);
        defer wave.deinit();

        const rows = try wave.dataConst();
        const out = try ctx.allocator.alloc(f32, rows.len);
        for (out, rows) |*o, sv| o.* = std.math.clamp(sv, -1.0, 1.0);
        return out;
    }
};

/// Unpack an `.rvq` byte stream (11-bit codes, LSB-first, `[K=16, T]`
/// row-major, no header) into i32 codes. Caller frees.
pub fn unpackRvq(allocator: Allocator, bytes: []const u8) !struct { codes: []i32, t: usize } {
    const total_codes = (bytes.len * 8) / code_bits;
    const t = total_codes / num_codebooks;
    if (t == 0) return Error.InvalidCodes;
    const codes = try allocator.alloc(i32, num_codebooks * t);
    errdefer allocator.free(codes);
    var acc: u32 = 0;
    var bits: u5 = 0;
    var idx: usize = 0;
    for (bytes) |b| {
        acc |= @as(u32, b) << bits;
        bits += 8;
        while (bits >= code_bits and idx < codes.len) {
            codes[idx] = @intCast(acc & 0x7FF);
            acc >>= code_bits;
            bits -= code_bits;
            idx += 1;
        }
        if (idx == codes.len) break;
    }
    if (idx != codes.len) return Error.InvalidCodes;
    return .{ .codes = codes, .t = t };
}

test {
    _ = @import("codec_tests.zig");
}
