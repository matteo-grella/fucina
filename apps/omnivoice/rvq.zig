//! RVQ decode (rvq-codec.h rvq_decode_graph) + the fc2 projection
//! (pipeline-codec.cpp:224-255): codes `[8, T]` (k-slow) → per codebook k,
//! gather codebook rows and apply `project_out` (bias added EVERY k — the 8
//! biases accumulate into the sum), then `fc2` maps the `[T, 1024]` latent to
//! the `[T, 256]` DAC decoder input.
//!
//! ENCODE side: `encodeCodes` is the 8-codebook residual quantization chain
//! (rvq_encode_graph), and `encode` is the full audio→codes pipeline
//! (pipeline_codec_encode) — it lives here, at the sink of the module DAG
//! (codec ← {hubert, semantic, dac} ← rvq), keeping imports cycle-free.
//!
//! Internal layout is Fucina's `[T, C]` rows (channel fast) throughout.

const std = @import("std");
const fucina = @import("fucina");

const codec = @import("codec.zig");
const dac = @import("dac.zig");
const hubert = @import("hubert.zig");
const semantic = @import("semantic.zig");
const wav = @import("wav.zig");

const ExecContext = fucina.ExecContext;
const Allocator = std.mem.Allocator;

pub const Error = error{
    InvalidCodes,
    CodeOutOfRange,
    FrameCountMismatch,
    InputTooShort,
};

/// RVQ latent `[T, 1024]` (pre-fc2).
pub const Latent = fucina.Tensor(.{ .seq, .d });
/// fc2 output `[T, 256]` (the DAC decoder input).
pub const Fc2Out = fucina.Tensor(.{ .seq, .fc });

pub const DecodeOut = struct {
    latent: Latent,
    fc2_out: Fc2Out,

    pub fn deinit(self: *DecodeOut) void {
        self.latent.deinit();
        self.fc2_out.deinit();
        self.* = undefined;
    }
};

/// Decodes `codes` (`[8, T]` row-major, k slow, values in
/// `[0, codebook_size)`) into the RVQ latent and its fc2 projection.
pub fn decode(ctx: *ExecContext, dec: *const codec.RvqDecoder, codes: []const i32, t: usize) !DecodeOut {
    if (t == 0 or codes.len != codec.n_codebooks * t) return Error.InvalidCodes;

    const indices = try ctx.allocator.alloc(usize, t);
    defer ctx.allocator.free(indices);

    var acc: ?Latent = null;
    errdefer if (acc) |*a| a.deinit();
    for (0..codec.n_codebooks) |k| {
        const q = &dec.quantizers[k];
        const v = q.embed.dim(.code);
        for (indices, codes[k * t ..][0..t]) |*dst, code| {
            if (code < 0) return Error.CodeOutOfRange;
            const idx: usize = @intCast(code);
            if (idx >= v) return Error.CodeOutOfRange;
            dst.* = idx;
        }

        var rows = try q.embed.gather(ctx, .code, indices, .seq); // [T, 64]
        defer rows.deinit();
        var proj = try q.project_out.linearSeq(ctx, &rows, .cdim, .d); // [T, 1024]
        try proj.addAxisVectorInPlace(ctx, q.project_out_bias, .d);
        if (acc) |*a| {
            defer proj.deinit();
            try a.addScaledInPlace(ctx, proj, 1.0);
        } else {
            acc = proj;
        }
    }

    var latent = acc.?;
    acc = null;
    errdefer latent.deinit();

    var fc2_out = try dec.fc2.linearSeq(ctx, &latent, .d, .fc); // [T, 256]
    errdefer fc2_out.deinit();
    try fc2_out.addAxisVectorInPlace(ctx, dec.fc2_bias, .fc);

    return .{ .latent = latent, .fc2_out = fc2_out };
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// One captured encode stage: `data` is `[t, c]` row-major.
pub const EncodeTap = struct {
    t: usize,
    c: usize,
    data: []f32,
};

/// Per-stage taps for encode parity dumps (all optional; filled by `encode`
/// when passed in). Layouts match the reference dumps: HuBERT taps and
/// pre_fc/embed are (T, C) row-major flat buffers.
pub const EncodeTaps = struct {
    allocator: Allocator,
    /// Post-resample, PRE-pad 16 kHz buffer (the reference's ref-audio-16k).
    audio_16k: ?[]f32 = null,
    hubert: hubert.Taps,
    /// Post mean+decimate HuBERT features (ref-hubert-features).
    features: ?EncodeTap = null,
    e_semantic: ?EncodeTap = null,
    e_acoustic: ?EncodeTap = null,
    pre_fc: ?EncodeTap = null,
    embed: ?EncodeTap = null,

    pub fn init(allocator: Allocator) EncodeTaps {
        return .{ .allocator = allocator, .hubert = hubert.Taps.init(allocator) };
    }

    pub fn deinit(self: *EncodeTaps) void {
        if (self.audio_16k) |buf| self.allocator.free(buf);
        self.hubert.deinit();
        inline for (.{ &self.features, &self.e_semantic, &self.e_acoustic, &self.pre_fc, &self.embed }) |maybe_tap| {
            if (maybe_tap.*) |tap| self.allocator.free(tap.data);
        }
        self.* = undefined;
    }
};

/// Argmax over one score row with ggml's tie-break (`ggml_vec_argmax`):
/// scan keeping the running max; ANY element equal to the running max
/// overwrites the index — among equal maxima the LARGEST index wins. The
/// score is the materialized f32 `2*dot − embed_sq` (one f32 mul + one f32
/// sub per element, exactly the reference's scale+sub node pair; the `‖h‖²`
/// term is dropped as frame-constant).
pub fn scoreArgmaxRow(dot_row: []const f32, embed_sq: []const f32) usize {
    var best = -std.math.inf(f32);
    var idx: usize = 0;
    for (dot_row, embed_sq, 0..) |d, sq, j| {
        const v = 2.0 * d - sq;
        if (v > best) best = v;
        if (best == v) idx = j;
    }
    return idx;
}

/// RVQ encode chain (rvq_encode_graph): 8-codebook residual quantization of
/// the fc output. `embed` (`[T, 1024]`) is consumed as the running residual
/// and MUTATED in place. Returns codes `[8, T]` row-major (k slow), owned by
/// `allocator`. The decode-side `quantizers` supply the codebooks +
/// `embed_sq` + `project_out`; `enc` supplies `project_in`.
pub fn encodeCodes(
    ctx: *ExecContext,
    allocator: Allocator,
    dec: *const codec.RvqDecoder,
    enc: *const codec.Encoder,
    embed: *Latent,
) ![]i32 {
    const t = embed.dim(.seq);
    if (t == 0) return Error.InvalidCodes;

    const codes = try allocator.alloc(i32, codec.n_codebooks * t);
    errdefer allocator.free(codes);
    const indices = try ctx.allocator.alloc(usize, t);
    defer ctx.allocator.free(indices);

    for (0..codec.n_codebooks) |k| {
        const q = &dec.quantizers[k];
        const pin = &enc.project_in[k];
        const v = q.embed.dim(.code);

        // h64 = project_in(residual) + bias: [T, 64].
        var h = try pin.weight.linearSeq(ctx, embed, .d, .cdim);
        defer h.deinit();
        try h.addAxisVectorInPlace(ctx, pin.bias, .cdim);

        // dot = h @ embedᵀ: [T, V]; score = 2*dot − embed_sq (host-fused).
        var dot = try h.matmul(ctx, &q.embed, .trans_b, .{ .seq, .code });
        defer dot.deinit();
        const dot_data = try dot.dataConst();
        for (0..t) |ti| {
            const idx = scoreArgmaxRow(dot_data[ti * v ..][0..v], q.embed_sq);
            indices[ti] = idx;
            codes[k * t + ti] = @intCast(idx);
        }

        // quant = project_out(embed[idx]) + bias; residual −= quant. The
        // last subtraction (k = 7) mirrors the reference's wasted node.
        var rows = try q.embed.gather(ctx, .code, indices, .seq); // [T, 64]
        defer rows.deinit();
        var quant = try q.project_out.linearSeq(ctx, &rows, .cdim, .d); // [T, 1024]
        defer quant.deinit();
        try quant.addAxisVectorInPlace(ctx, q.project_out_bias, .d);
        try embed.addScaledInPlace(ctx, quant, -1.0);
    }
    return codes;
}

/// Full encode pipeline (pipeline_codec_encode): hop-aligned 24 kHz mono →
/// codes `[8, T]` (k slow), owned by `allocator`.
///
/// 1. resample 24k→16k → HuBERT (±160 pad inside) → features `[T_s, 768]`.
/// 2. SemanticEncoder → e_semantic `[T_s, 768]`.
/// 3. DAC encoder on the ORIGINAL 24 kHz buffer; if the analytic frame count
///    differs from T_s, zero-pad the audio hop/2 = 480 each side first, then
///    REQUIRE T_a == T_s.
/// 4. pre_fc = concat channels [acoustic 0..255; semantic 256..1023]; fc →
///    embed `[T, 1024]`.
/// 5. RVQ residual chain → codes.
pub fn encode(
    ctx: *ExecContext,
    allocator: Allocator,
    config: codec.Config,
    rvq_dec: *const codec.RvqDecoder,
    enc: *const codec.Encoder,
    audio_24k: []const f32,
    taps: ?*EncodeTaps,
) ![]i32 {
    if (audio_24k.len == 0) return Error.InputTooShort;

    // Step 1: 16 kHz semantic input → HuBERT features [T_s, 768]. The
    // FMA-contracted resample matches the shipped reference binary.
    const audio_16k = try wav.resampleFma(ctx.allocator, audio_24k, 24000, 16000);
    defer ctx.allocator.free(audio_16k);
    if (taps) |tp| tp.audio_16k = try tp.allocator.dupe(f32, audio_16k);

    const hubert_taps: ?*hubert.Taps = if (taps) |tp| &tp.hubert else null;
    var features = try hubert.forward(ctx, &enc.hubert, audio_16k, hubert_taps);
    defer features.deinit();
    const t_s = features.dim(.seq);
    if (taps) |tp| tp.features = try captureTap(tp.allocator, &features);

    // Step 2: SemanticEncoder → e_semantic [T_s, 768].
    var e_semantic = try semantic.forward(ctx, &enc.semantic, &features);
    defer e_semantic.deinit();
    if (taps) |tp| tp.e_semantic = try captureTap(tp.allocator, &e_semantic);

    // Step 3: DAC encoder on the original 24 kHz audio (± conditional pad).
    const t_a_no_pad = dac.encodeOutputLength(audio_24k.len);
    var e_acoustic = blk: {
        if (t_a_no_pad != @as(isize, @intCast(t_s))) {
            const p = config.hop_length / 2; // 480
            const padded = try ctx.allocator.alloc(f32, audio_24k.len + 2 * p);
            defer ctx.allocator.free(padded);
            @memset(padded, 0);
            @memcpy(padded[p..][0..audio_24k.len], audio_24k);
            break :blk try dac.encodeForward(ctx, &enc.dac, padded);
        }
        break :blk try dac.encodeForward(ctx, &enc.dac, audio_24k);
    };
    defer e_acoustic.deinit();
    if (e_acoustic.dim(.seq) != t_s) return Error.FrameCountMismatch;
    if (taps) |tp| tp.e_acoustic = try captureTap(tp.allocator, &e_acoustic);

    // Step 4: concat channels [acoustic; semantic] → fc → embed [T, 1024].
    var pre_fc = try e_acoustic.concat(ctx, .in, &.{&e_semantic});
    defer pre_fc.deinit();
    if (taps) |tp| tp.pre_fc = try captureTap(tp.allocator, &pre_fc);

    var embed = try enc.fc.linearSeq(ctx, &pre_fc, .in, .d);
    defer embed.deinit();
    try embed.addAxisVectorInPlace(ctx, enc.fc_bias, .d);
    if (taps) |tp| tp.embed = try captureTap(tp.allocator, &embed);

    // Step 5: RVQ residual chain (mutates embed as the residual).
    return encodeCodes(ctx, allocator, rvq_dec, enc, &embed);
}

fn captureTap(allocator: Allocator, x: anytype) !EncodeTap {
    const data = try allocator.dupe(f32, try x.dataConst());
    const raw = x.asRawTensor();
    return .{ .t = raw.shape.at(0), .c = raw.shape.at(1), .data = data };
}

test {
    _ = @import("rvq_tests.zig");
}
