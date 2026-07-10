//! OmniVoice TTS LM — the Qwen3-0.6B-variant backbone: bidirectional
//! attention, hybrid text+audio input embedding over 8 codebooks, and the
//! `audio_heads` [K*V, H] output projection. Ports refs/omnivoice.cpp
//! `omnivoice-llm.h` + `qwen3-enc.h`.
//!
//! Single-sequence forward only (no batch axis): the CFG cond/uncond pair of
//! the MaskGIT loop runs as two independent forward calls at the pipeline
//! level — numerically equivalent to the reference's own per-row debug loop
//! (pipeline-tts.cpp:394-425).

const std = @import("std");
const builtin = @import("builtin");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;
const weights = llm.weights;
const LinearWeight = weights.LinearWeight;

/// Whether to output-stack q/k/v (and gate/up) into fused GEMM weights.
/// Measured (i9-13950HX, design clip, 8 P-cores, 2026-07-04):
/// quantized bases want fusion (Q4_K_M 44.5s fused vs 46.9s separate — the
/// fused splitSwiGlu→down path only exists on the fused arm; Q8_0 tied), but
/// on the no-BLAS dense-float row/blocked-kernel builds fusion REGRESSES
/// hard (F32 64.2s fused vs 51.2s separate; BF16 76.3s vs 67.0s — the wider
/// fused outputs pay strided split-view materializations with no dispatch
/// economy to amortize them). Vendor-BLAS builds (M1/Accelerate) keep fusion
/// for all dtypes: measured faster there (design Q8_0 ~24s fused vs ~28s).
fn shouldFuse(sample: *const LinearWeight) bool {
    return switch (sample.*) {
        .f32, .f16, .bf16 => fucina.native_uses_blas,
        else => true,
    };
}

pub const Error = weights.Error || error{
    InvalidConfig,
    InvalidArchitecture,
    InvalidSequenceLength,
    InvalidLogitsRange,
    InvalidTokenId,
};

pub const arch_name = "omnivoice-lm";

/// The seven prompt special-token ids from the `omnivoice.special.*` GGUF
/// metadata (docs/ARCHITECTURE.md:102-108). Consumed by the prompt builder (next
/// stage), which splices them programmatically around BPE-encoded spans.
pub const SpecialTokens = struct {
    denoise: u32,
    lang_start: u32,
    lang_end: u32,
    instruct_start: u32,
    instruct_end: u32,
    text_start: u32,
    text_end: u32,
};

pub const Config = struct {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    num_audio_codebook: usize,
    audio_vocab_size: usize,
    audio_mask_id: usize,
    specials: SpecialTokens,

    /// Rows of `audio_embeddings.weight` / `audio_heads.weight`: the K
    /// codebooks flattened codebook-major (row index = k*V + v).
    pub fn audioTableRows(self: Config) usize {
        return self.num_audio_codebook * self.audio_vocab_size;
    }

    pub fn fromGguf(file: *const gguf.File) !Config {
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        if (!std.mem.eql(u8, arch, arch_name)) return Error.InvalidArchitecture;

        // vocab_size from the embedding table's logical {vocab, hidden} shape
        // (the omnivoice-lm.vocab_size KV exists but the tensor is the truth).
        const embd = try file.get("llm.embed_tokens.weight");
        const embd_shape = try embd.logicalMatrixShape();

        const config = Config{
            .vocab_size = embd_shape[0],
            .hidden_size = try metaUsize(file, "omnivoice-lm.embedding_length"),
            .intermediate_size = try metaUsize(file, "omnivoice-lm.feed_forward_length"),
            .num_layers = try metaUsize(file, "omnivoice-lm.block_count"),
            .num_attention_heads = try metaUsize(file, "omnivoice-lm.attention.head_count"),
            .num_key_value_heads = try metaUsize(file, "omnivoice-lm.attention.head_count_kv"),
            .head_dim = try metaUsize(file, "omnivoice-lm.attention.key_length"),
            .rms_norm_eps = try metaF32(file, "omnivoice-lm.attention.layer_norm_rms_epsilon"),
            .rope_theta = try metaF32(file, "omnivoice-lm.rope.freq_base"),
            .num_audio_codebook = try metaUsize(file, "omnivoice.num_audio_codebook"),
            .audio_vocab_size = try metaUsize(file, "omnivoice.audio_vocab_size"),
            .audio_mask_id = try metaUsize(file, "omnivoice.audio_mask_id"),
            .specials = .{
                .denoise = try metaU32(file, "omnivoice.special.denoise"),
                .lang_start = try metaU32(file, "omnivoice.special.lang_start"),
                .lang_end = try metaU32(file, "omnivoice.special.lang_end"),
                .instruct_start = try metaU32(file, "omnivoice.special.instruct_start"),
                .instruct_end = try metaU32(file, "omnivoice.special.instruct_end"),
                .text_start = try metaU32(file, "omnivoice.special.text_start"),
                .text_end = try metaU32(file, "omnivoice.special.text_end"),
            },
        };
        try config.validate();
        if (embd_shape[1] != config.hidden_size) return Error.InvalidConfig;
        return config;
    }

    fn validate(self: Config) !void {
        if (self.num_attention_heads == 0 or self.num_key_value_heads == 0) return Error.InvalidConfig;
        if (self.num_attention_heads % self.num_key_value_heads != 0) return Error.InvalidConfig;
        if (self.head_dim % 2 != 0) return Error.InvalidConfig;
        if (self.num_audio_codebook == 0 or self.audio_vocab_size == 0) return Error.InvalidConfig;
        if (self.audio_mask_id >= self.audio_vocab_size) return Error.InvalidConfig;
    }
};

fn metaUsize(file: *const gguf.File, key: []const u8) !usize {
    const v = file.getInt(key) orelse return Error.InvalidConfig;
    if (v < 0) return Error.InvalidConfig;
    return @intCast(v);
}

fn metaF32(file: *const gguf.File, key: []const u8) !f32 {
    const v = file.getFloat(key) orelse return Error.InvalidConfig;
    return @floatCast(v);
}

fn metaU32(file: *const gguf.File, key: []const u8) !u32 {
    const v = file.getInt(key) orelse return Error.InvalidConfig;
    if (v < 0 or v > std.math.maxInt(u32)) return Error.InvalidConfig;
    return @intCast(v);
}

/// The layer indices whose post-layer hidden state the reference dumps
/// (`lm-hidden-step0-*-l{i}`), mirrored by `TapSink` captures.
pub const tap_layer_indices = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20 };
/// The layer whose four pre-residual sub-module outputs are tapped
/// (`l1-{norm1,attn,norm2,mlp}` — reference dump_sub_layer = 1).
pub const sub_tap_layer: usize = 1;

/// Optional forward-pass tap sink: named owned [rows, hidden] f32 copies of
/// intermediate hidden states (embed, l0..l6, l13..l20, l1-norm1/l1-attn/
/// l1-norm2/l1-mlp, final), matching the reference `lm-hidden-step0-*` dump
/// taps. Used by later stages for `--dump`.
pub const TapSink = struct {
    allocator: Allocator,
    entries: std.ArrayList(Entry) = .empty,

    pub const Entry = struct {
        name: []const u8,
        rows: usize,
        cols: usize,
        data: []const f32,
    };

    pub fn init(allocator: Allocator) TapSink {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *TapSink) void {
        for (self.entries.items) |entry| {
            self.allocator.free(entry.name);
            self.allocator.free(entry.data);
        }
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn get(self: *const TapSink, name: []const u8) ?*const Entry {
        for (self.entries.items) |*entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry;
        }
        return null;
    }

    /// Copies the tensor's contiguous data (dataConst + dupe) under an owned
    /// copy of `name`.
    pub fn capture(self: *TapSink, name: []const u8, t: *const fucina.Tensor(.{ .seq, .embed })) !void {
        const src = try t.dataConst();
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);
        const data_copy = try self.allocator.dupe(f32, src);
        errdefer self.allocator.free(data_copy);
        try self.entries.append(self.allocator, .{
            .name = name_copy,
            .rows = t.dim(.seq),
            .cols = t.dim(.embed),
            .data = data_copy,
        });
    }
};

/// The uncond CFG row's additive attention bias: [S, S] f32, row sq
/// contiguous over skv, values exactly 1.0/0.0. Query rows sq < u_len get
/// +1.0 on keys [0, u_len) and +0.0 on the padding tail; padding rows
/// sq >= u_len get +1.0 ONLY on their own diagonal. The bias is added AFTER
/// the 1/sqrt(D) score scaling and is NOT a hard mask: blocked keys still
/// receive softmax mass (down-weighted by e^-1 relative) — the model was
/// TRAINED with this soft bias (the Python reference adds a boolean mask
/// promoted to 0/1 floats), so -inf masking would break parity.
pub const UncondBias = fucina.Tensor(.{ .sq, .skv });

/// Builds the uncond bias once per decode call (it is constant across the
/// MaskGIT steps of a chunk, mirroring the reference MaskgitBatchedCtx).
pub fn buildUncondBias(ctx: *ExecContext, seq_len: usize, u_len: usize) !UncondBias {
    if (u_len > seq_len or seq_len == 0) return Error.InvalidSequenceLength;
    const data = try ctx.allocator.alloc(f32, seq_len * seq_len);
    defer ctx.allocator.free(data);
    @memset(data, 0.0);
    for (0..u_len) |sq| {
        @memset(data[sq * seq_len ..][0..u_len], 1.0);
    }
    for (u_len..seq_len) |sq| {
        data[sq * seq_len + sq] = 1.0;
    }
    return UncondBias.fromSlice(ctx, .{ seq_len, seq_len }, data);
}

/// shifted[s] = ids[k*S + s] * m[s] + k*V with m = audio_mask (0/1) — the
/// reference's exact shifted-index formula (pipeline-tts.cpp:435-446). On
/// text positions (m = 0) the id is gated to 0, so the index stays valid
/// (k*V) and the gathered row is later discarded by the hybrid-embed stitch.
pub fn fillShiftedIds(
    ids: []const i32,
    audio_mask: []const i32,
    k: usize,
    audio_vocab_size: usize,
    out: []usize,
) !void {
    const seq_len = audio_mask.len;
    for (out, 0..) |*dst, s| {
        const raw = ids[k * seq_len + s];
        if (raw < 0) return Error.InvalidTokenId;
        const gated: usize = if (audio_mask[s] != 0) @intCast(raw) else 0;
        if (gated >= audio_vocab_size) return Error.InvalidTokenId;
        dst.* = gated + k * audio_vocab_size;
    }
}

pub fn loadModel(ctx: *ExecContext, file: *const gguf.File) !Model {
    const config = try Config.fromGguf(file);
    return Model.load(ctx, file, config);
}

pub const Model = struct {
    allocator: Allocator,
    config: Config,
    embed_tokens: LinearWeight, // [vocab, hidden]
    audio_embeddings: LinearWeight, // [K*V, hidden], row = k*V + v
    audio_heads: LinearWeight, // [K*V, hidden], NOT tied to audio_embeddings
    final_norm: fucina.Tensor(.{.embed}), // llm.norm.weight [hidden]
    layers: []Layer,
    kv_head_for_head: []usize,

    pub fn load(ctx: *ExecContext, file: *const gguf.File, config: Config) !Model {
        const allocator = ctx.allocator;

        var embed_tokens = try LinearWeight.load(ctx, try file.get("llm.embed_tokens.weight"), config.vocab_size, config.hidden_size);
        errdefer embed_tokens.deinit();

        var audio_embeddings = try LinearWeight.load(ctx, try file.get("audio_embeddings.weight"), config.audioTableRows(), config.hidden_size);
        errdefer audio_embeddings.deinit();

        var audio_heads = try LinearWeight.load(ctx, try file.get("audio_heads.weight"), config.audioTableRows(), config.hidden_size);
        errdefer audio_heads.deinit();

        var final_norm = try weights.loadVector(ctx, try file.get("llm.norm.weight"), config.hidden_size, .embed);
        errdefer final_norm.deinit();

        const kv_head_for_head = try allocator.alloc(usize, config.num_attention_heads);
        errdefer allocator.free(kv_head_for_head);
        const heads_per_kv = config.num_attention_heads / config.num_key_value_heads;
        for (kv_head_for_head, 0..) |*kv_head, head_i| kv_head.* = head_i / heads_per_kv;

        const layers = try allocator.alloc(Layer, config.num_layers);
        errdefer allocator.free(layers);
        var loaded: usize = 0;
        errdefer for (layers[0..loaded]) |*layer| layer.deinit();
        for (layers, 0..) |*layer, layer_i| {
            layer.* = try Layer.load(ctx, file, config, layer_i);
            loaded += 1;
        }

        return .{
            .allocator = allocator,
            .config = config,
            .embed_tokens = embed_tokens,
            .audio_embeddings = audio_embeddings,
            .audio_heads = audio_heads,
            .final_norm = final_norm,
            .layers = layers,
            .kv_head_for_head = kv_head_for_head,
        };
    }

    pub fn deinit(self: *Model) void {
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.allocator.free(self.kv_head_for_head);
        self.final_norm.deinit();
        self.audio_heads.deinit();
        self.audio_embeddings.deinit();
        self.embed_tokens.deinit();
        self.* = undefined;
    }

    /// Full bidirectional forward of ONE sequence.
    ///
    /// - `ids`: [K, S] i32, k slow / s fast (row 0 doubles as text ids on
    ///   non-audio positions).
    /// - `audio_mask`: [S] 0/1 i32 (1 = audio position).
    /// - Returns audio logits `[logits_len, K*V]` for positions
    ///   `[logits_start, logits_start + logits_len)`: flat index
    ///   `s_rel*(K*V) + k*V + v` — identical to the reference dump layout
    ///   ([V fast, K mid, S slow]). The hidden states are narrowed BEFORE the
    ///   head GEMM (T_audio narrowing): the GEMM is row-independent, so this
    ///   is math-identical to the reference's full-GEMM-then-view while
    ///   skipping the head-GEMM rows whose logits are never read.
    pub fn forward(
        self: *const Model,
        ctx: *ExecContext,
        ids: []const i32,
        audio_mask: []const i32,
        logits_start: usize,
        logits_len: usize,
        taps: ?*TapSink,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        const seq_len = audio_mask.len;
        if (logits_len == 0 or logits_start + logits_len > seq_len) return Error.InvalidLogitsRange;

        var final = try self.hiddenForward(ctx, ids, audio_mask, taps);
        defer final.deinit();

        var head_in = try final.narrow(ctx, .seq, logits_start, logits_len);
        defer head_in.deinit();
        return self.audio_heads.linearSeq(ctx, &head_in, .embed, .vocab);
    }

    /// The uncond CFG row forward, PADDED to S_max with the reference's
    /// additive +1.0/0.0 attention bias: the same stack as `forward`, but
    /// every layer's attention runs the hand-composed biased path
    /// (softmax(scale*q·kT + bias) per head) instead of the plain
    /// bidirectional kernel. Returns audio logits for positions [0, logits_len)
    /// (the uncond target window sits at the row head), layout identical to
    /// `forward`. `bias` comes from `buildUncondBias(ctx, S, u_len)` and is
    /// built once per decode call.
    pub fn forwardUncondPadded(
        self: *const Model,
        ctx: *ExecContext,
        ids: []const i32,
        audio_mask: []const i32,
        bias: *const UncondBias,
        logits_len: usize,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        const seq_len = audio_mask.len;
        if (logits_len == 0 or logits_len > seq_len) return Error.InvalidLogitsRange;
        if (bias.dim(.sq) != seq_len or bias.dim(.skv) != seq_len) return Error.InvalidSequenceLength;

        var final = try self.stackForward(ctx, ids, audio_mask, null, bias);
        defer final.deinit();

        var head_in = try final.narrow(ctx, .seq, 0, logits_len);
        defer head_in.deinit();
        return self.audio_heads.linearSeq(ctx, &head_in, .embed, .vocab);
    }

    /// The backbone up to and including the final RMSNorm: returns the
    /// post-final-norm hidden states `[S, hidden]` (the reference's
    /// `lm-hidden-step0-*` "final" tensor, pre audio_heads).
    pub fn hiddenForward(
        self: *const Model,
        ctx: *ExecContext,
        ids: []const i32,
        audio_mask: []const i32,
        taps: ?*TapSink,
    ) !fucina.Tensor(.{ .seq, .embed }) {
        return self.stackForward(ctx, ids, audio_mask, taps, null);
    }

    /// Shared 28L stack: hybrid embed, attention (plain bidirectional when
    /// `bias` is null, additive-bias composition otherwise), SwiGLU FFN,
    /// final RMSNorm.
    fn stackForward(
        self: *const Model,
        ctx: *ExecContext,
        ids: []const i32,
        audio_mask: []const i32,
        taps: ?*TapSink,
        bias: ?*const UncondBias,
    ) !fucina.Tensor(.{ .seq, .embed }) {
        const cfg = &self.config;
        const seq_len = audio_mask.len;
        if (seq_len == 0) return Error.InvalidSequenceLength;
        if (ids.len != cfg.num_audio_codebook * seq_len) return Error.InvalidSequenceLength;

        // Positions 0..S-1, identical for cond and uncond rows (reference
        // pipeline-tts.cpp:477-480); NEOX half-rotation RoPE over all
        // head_dim dims, theta 1e6, no freq factors.
        const positions = try ctx.allocator.alloc(i32, seq_len);
        defer ctx.allocator.free(positions);
        for (positions, 0..) |*position, i| position.* = @intCast(i);
        var rope_table = try ctx.prepareRopeTable(positions, cfg.head_dim, cfg.rope_theta, false);
        defer rope_table.deinit();

        var final = blk: {
            var x = try self.hybridEmbed(ctx, ids, audio_mask);
            errdefer x.deinit();
            if (taps) |sink| try sink.capture("embed", &x);

            for (self.layers, 0..) |*layer, layer_i| {
                x = try ctx.replace(x, self.attnBlock(ctx, layer, &x, &rope_table, bias, taps, layer_i));
                x = try ctx.replace(x, self.ffnBlock(ctx, layer, &x, taps, layer_i));
                if (taps) |sink| {
                    if (std.mem.indexOfScalar(usize, &tap_layer_indices, layer_i) != null) {
                        var name_buf: [16]u8 = undefined;
                        const name = std.fmt.bufPrint(&name_buf, "l{d}", .{layer_i}) catch unreachable;
                        try sink.capture(name, &x);
                    }
                }
            }

            const normed = try x.rmsNormMul(ctx, .embed, &self.final_norm, cfg.rms_norm_eps);
            x.deinit();
            break :blk normed;
        };
        errdefer final.deinit();
        if (taps) |sink| try sink.capture("final", &final);
        return final;
    }

    /// Hybrid text+audio input embedding (reference pipeline-tts.cpp:431-507,
    /// exact math): text_emb = get_rows(embed_tokens, ids[0]); audio_emb =
    /// left-fold sum over k=0..K-1 of get_rows(audio_embeddings, shifted_k)
    /// (sequential adds, ascending k — the reference's f32 accumulation
    /// order); inputs = audio row where m=1, text row where m=0. The
    /// reference gates by exact 0.0/1.0 f32 multiplies + add, so a host-side
    /// per-position row select is bit-identical (one addend is exactly
    /// zeroed).
    fn hybridEmbed(
        self: *const Model,
        ctx: *ExecContext,
        ids: []const i32,
        audio_mask: []const i32,
    ) !fucina.Tensor(.{ .seq, .embed }) {
        const cfg = &self.config;
        const seq_len = audio_mask.len;
        const hidden = cfg.hidden_size;
        const allocator = ctx.allocator;

        // Text rows: row k=0 of ids, gathered UNMASKED (audio positions look
        // up their in-range audio id in the text table; the row is discarded
        // by the stitch — keep index validity like the reference).
        const text_ids = try allocator.alloc(usize, seq_len);
        defer allocator.free(text_ids);
        for (text_ids, 0..) |*dst, s| {
            const raw = ids[s];
            if (raw < 0) return Error.InvalidTokenId;
            const idx: usize = @intCast(raw);
            if (idx >= cfg.vocab_size) return Error.InvalidTokenId;
            dst.* = idx;
        }
        var text_emb = try self.embed_tokens.getRowsAs(ctx, text_ids, .embed);
        defer text_emb.deinit();

        const shifted = try allocator.alloc(usize, seq_len);
        defer allocator.free(shifted);

        var audio_emb = blk: {
            try fillShiftedIds(ids, audio_mask, 0, cfg.audio_vocab_size, shifted);
            var acc = try self.audio_embeddings.getRowsAs(ctx, shifted, .embed);
            errdefer acc.deinit();
            for (1..cfg.num_audio_codebook) |k| {
                try fillShiftedIds(ids, audio_mask, k, cfg.audio_vocab_size, shifted);
                var rows = try self.audio_embeddings.getRowsAs(ctx, shifted, .embed);
                defer rows.deinit();
                acc = try ctx.replace(acc, acc.add(ctx, &rows));
            }
            break :blk acc;
        };
        defer audio_emb.deinit();

        const text_data = try text_emb.dataConst();
        const audio_data = try audio_emb.dataConst();
        const stitched = try allocator.alloc(f32, seq_len * hidden);
        defer allocator.free(stitched);
        for (0..seq_len) |s| {
            const src = if (audio_mask[s] != 0) audio_data else text_data;
            @memcpy(stitched[s * hidden ..][0..hidden], src[s * hidden ..][0..hidden]);
        }
        return fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ seq_len, hidden }, stitched);
    }

    /// Qwen3-semantics self-attention block, bidirectional (is_causal=false).
    ///
    /// Attention-bias note: the reference adds an ADDITIVE +1.0/0.0 f32 bias
    /// to the scaled scores pre-softmax (NOT a -inf mask). When the bias is
    /// CONSTANT over keys (the llm-test harness passes no mask, i.e.
    /// all-ones; the cond prompt row is all-ones too), softmax
    /// shift-invariance makes plain bidirectional groupedAttention with no
    /// bias mathematically exact — that is the `bias == null` path. The
    /// mixed-bias uncond row (ones on [0,u_len) plus diagonal-only tail)
    /// passes its bias tensor and runs the composed per-head path.
    fn attnBlock(
        self: *const Model,
        ctx: *ExecContext,
        layer: *const Layer,
        input: *const fucina.Tensor(.{ .seq, .embed }),
        rope_table: *const fucina.RopeTable,
        bias: ?*const UncondBias,
        taps: ?*TapSink,
        layer_i: usize,
    ) !fucina.Tensor(.{ .seq, .embed }) {
        const cfg = &self.config;

        var attn_in = try input.rmsNormMul(ctx, .embed, &layer.attn_norm, cfg.rms_norm_eps);
        defer attn_in.deinit();
        if (taps) |sink| if (layer_i == sub_tap_layer) try sink.capture("l1-norm1", &attn_in);

        var qkv_linear = try layer.attn_proj.project(ctx, &attn_in, cfg);
        defer qkv_linear.deinit();

        var q3 = try qkv_linear.q.split(ctx, .q, .{ .head, .d }, .{ cfg.num_attention_heads, cfg.head_dim });
        defer q3.deinit();
        var k3 = try qkv_linear.k.split(ctx, .k, .{ .kv_head, .d }, .{ cfg.num_key_value_heads, cfg.head_dim });
        defer k3.deinit();
        var v3 = try qkv_linear.v.split(ctx, .v, .{ .kv_head, .d }, .{ cfg.num_key_value_heads, cfg.head_dim });
        defer v3.deinit();

        // Per-head QK-RMSNorm (same [head_dim] weight for all heads) BEFORE
        // RoPE, fused with the half (NEOX) rotation; v is neither normed nor
        // roped — qwen3-enc.h:177-186 order.
        var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, cfg.rms_norm_eps, rope_table);
        defer q_rope.deinit();
        var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm, cfg.rms_norm_eps, rope_table);
        defer k_rope.deinit();

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        var attn = if (bias) |bias_tensor|
            try self.biasedAttention(ctx, &q_rope, &k_rope, &v3, bias_tensor, scale)
        else
            try q_rope.groupedAttention(ctx, &k_rope, &v3, self.kv_head_for_head, .attn, scale, .{ .mask = .bidirectional });
        defer attn.deinit();

        var attn_out = try layer.o_proj.linearSeq(ctx, &attn, .attn, .embed);
        defer attn_out.deinit();
        if (taps) |sink| if (layer_i == sub_tap_layer) try sink.capture("l1-attn", &attn_out);

        return input.add(ctx, &attn_out);
    }

    /// Additive-bias attention (the reference qwen3_attn_f32,
    /// qwen3-enc.h:82-93): per q head (GQA kv_head = head / heads_per_kv),
    /// scores[sq][skv] = q_h[sq]·k_h[skv], probs = softmax(scale*scores +
    /// bias) — matching ggml_soft_max_ext — then out_h = probs·v_h; heads
    /// concatenated on the feature axis. f32 throughout. Arch-gated
    /// realization: aarch64 keeps the hand-composed per-head path (narrow +
    /// GEMM + softmaxExt row kernel + GEMM — bit-identical to the reference
    /// composition, the golden-parity leg); other arches run the fused
    /// additive-bias online-softmax kernel
    /// (bidirectional `groupedAttention` with `.bias`), which agrees to
    /// ~1e-6 relative but not bitwise (online vs row softmax).
    fn biasedAttention(
        self: *const Model,
        ctx: *ExecContext,
        q: *const fucina.Tensor(.{ .seq, .head, .d }),
        k: *const fucina.Tensor(.{ .seq, .kv_head, .d }),
        v: *const fucina.Tensor(.{ .seq, .kv_head, .d }),
        bias: *const UncondBias,
        scale: f32,
    ) !fucina.Tensor(.{ .seq, .attn }) {
        if (comptime builtin.cpu.arch == .aarch64) {
            // aarch64 keeps the composed per-head path EXACTLY: the golden
            // token-exactness contract lives here, and BOTH macOS parity legs
            // (Accelerate and -Dblas=none) must stay bit-identical.
            // Per-head is also the measured Accelerate winner (design Q8_0
            // clip: 12.65s vs batched 13.3s — per-batch sgemm calls contend
            // on the AMX). biasedAttentionBatched stays as the bit-identical,
            // A/B-tested alternate (see the inline test), like the scalar
            // reference kernels.
            return biasedAttentionPerHead(ctx, q, k, v, self.kv_head_for_head, bias, scale);
        } else {
            // Elsewhere (the x86 target): the fused additive-bias
            // online-softmax attention kernel — one dispatch replaces the
            // per-head narrow/GEMM/softmaxExt/GEMM/concat composition (~8.6%
            // of design-run f32 GEMM cycles, growing quadratically with the
            // clone seq). Low-bit differences vs the per-head row softmax
            // are expected and acceptable here: same-build determinism
            // holds, and cross-implementation F32 token equality is
            // corpus-dependent on x86 anyway.
            return q.groupedAttention(ctx, k, v, self.kv_head_for_head, .attn, scale, .{ .mask = .bidirectional, .bias = bias });
        }
    }

    /// SwiGLU FFN: down(silu(gate(x)) * up(x)) — qwen3-enc.h:219-234.
    fn ffnBlock(
        self: *const Model,
        ctx: *ExecContext,
        layer: *const Layer,
        input: *const fucina.Tensor(.{ .seq, .embed }),
        taps: ?*TapSink,
        layer_i: usize,
    ) !fucina.Tensor(.{ .seq, .embed }) {
        const cfg = &self.config;

        var ffn_in = try input.rmsNormMul(ctx, .embed, &layer.ffn_norm, cfg.rms_norm_eps);
        defer ffn_in.deinit();
        if (taps) |sink| if (layer_i == sub_tap_layer) try sink.capture("l1-norm2", &ffn_in);

        var down = try self.ffnDown(ctx, layer, &ffn_in);
        defer down.deinit();
        if (taps) |sink| if (layer_i == sub_tap_layer) try sink.capture("l1-mlp", &down);

        return input.add(ctx, &down);
    }

    /// down(silu(gate(x)) * up(x)) with the qwen3 fused-GEMM structure: when
    /// gate/up were fused at load time, one [2*ffn, hidden] GEMM replaces two;
    /// quantized down weights additionally take the fused
    /// split-SwiGLU+LHS-quantization+down-GEMM kernel (the gated m*ffn tensor
    /// is never materialized).
    fn ffnDown(
        self: *const Model,
        ctx: *ExecContext,
        layer: *const Layer,
        ffn_in: *const fucina.Tensor(.{ .seq, .embed }),
    ) !fucina.Tensor(.{ .seq, .embed }) {
        const cfg = &self.config;

        // Multi-token fused fast path (qwen3 denseFfnFusedDown). The fused
        // kernel's split-SwiGLU form ((up*gate)*sigmoid vs the gated kernel's
        // up*(gate*sigmoid)) is NOT bit-identical, so it is confined to the
        // quantized-down arms whose parity gates are tolerance-based; the
        // float arms below keep the token-exact F32 golden contract.
        if (ffn_in.dim(.seq) >= 12 and layer.ffn_input == .fused) {
            const gate_up_weight = &layer.ffn_input.fused;
            switch (layer.down_proj) {
                .q4_k => |*down| if (comptime !fucina.supports_q4_k_mmla) {
                    return ffnFusedDown(ctx, gate_up_weight, &down.packed_rhs, ffn_in);
                },
                .q5_k => |*down| return ffnFusedDown(ctx, gate_up_weight, &down.packed_rhs, ffn_in),
                .q6_k => |*down| return ffnFusedDown(ctx, gate_up_weight, &down.packed_rhs, ffn_in),
                .q8_0 => |*down| return ffnFusedDown(ctx, gate_up_weight, &down.packed_rhs, ffn_in),
                else => {},
            }
        }

        var gated = switch (layer.ffn_input) {
            .separate => |*separate| blk: {
                var gate = try separate.gate_proj.linearSeq(ctx, ffn_in, .embed, .ffn);
                defer gate.deinit();
                var up = try separate.up_proj.linearSeq(ctx, ffn_in, .embed, .ffn);
                defer up.deinit();

                // swiglu convention: self * silu(other) — so up.swiglu(gate)
                // is the reference's silu(gate) * up.
                break :blk try up.swiglu(ctx, &gate);
            },
            .fused => |*weight| blk: {
                var gate_up = try weight.linearSeq(ctx, ffn_in, .embed, .gate_up);
                defer gate_up.deinit();

                // BIT-EXACTNESS: split into gate/up VIEWS and run the same
                // gated swiglu kernel as the separate path — NOT
                // gate_up.splitGated(.swiglu). splitSwiGluRows computes
                // (up*gate)*sigmoid while the gated kernel computes
                // up*(gate*sigmoid); f32 multiply is non-associative, so
                // ~35% of elements differ in the last ulp (measured), which
                // would break the token-exact F32 parity gate.
                var gate_view = try gate_up.narrow(ctx, .gate_up, 0, cfg.intermediate_size);
                defer gate_view.deinit();
                var gate = try gate_view.withTags(ctx, .{ .seq, .ffn });
                defer gate.deinit();
                var up_view = try gate_up.narrow(ctx, .gate_up, cfg.intermediate_size, cfg.intermediate_size);
                defer up_view.deinit();
                var up = try up_view.withTags(ctx, .{ .seq, .ffn });
                defer up.deinit();
                break :blk try up.swiglu(ctx, &gate);
            },
        };
        defer gated.deinit();

        return layer.down_proj.linearSeq(ctx, &gated, .ffn, .embed);
    }
};

/// One gate_up GEMM, then the fused split-SwiGLU + LHS quantization + packed
/// down GEMM (qwen3's denseFfnFusedDown, without the profile plumbing).
fn ffnFusedDown(
    ctx: *ExecContext,
    gate_up_weight: *const LinearWeight,
    rhs: anytype,
    ffn_in: *const fucina.Tensor(.{ .seq, .embed }),
) !fucina.Tensor(.{ .seq, .embed }) {
    var gate_up = try gate_up_weight.linearSeq(ctx, ffn_in, .embed, .gate_up);
    defer gate_up.deinit();
    return gate_up.splitSwiGluDotPacked(ctx, rhs, .gate_up, .embed);
}

/// Head-batched broadcast-bmm form of the additive-bias attention: one
/// bmmTransB + one rank-4 softmaxExt + one bmm replace the per-head loop
/// (biasedAttentionPerHead below). q is viewed [kv_head, g, seq, d] with
/// head = kv_head*heads_per_kv + g; k/v are viewed [kv_head, 1, seq, d], so
/// bmm's BROADCAST batch mode dispatches the very same backend 2-D entries
/// (matmulTransB2DIntoUnchecked / matmul2DIntoUnchecked) per head at the same
/// (m, n, k). Kernel choice depends only on shape + comptime flags and the
/// row kernels never split the k reduction, so every head's output is
/// BIT-IDENTICAL to the per-head loop; the rank-4 softmaxExt broadcasts the
/// [sq, skv] bias stride-0 over the two head axes and runs the identical
/// per-row SIMD softmax(x*scale + mask) body. The win is dispatch structure,
/// not math: per layer, ~48 sequential pool fork/joins and 2 prepare copies
/// per kv head collapse into 3 batched ops with one copy per operand, and the
/// final [seq, kv_head, g, d] materialize reproduces the head-concat layout
/// in one pass. Used on no-BLAS builds only (see biasedAttention): with a
/// vendor BLAS the concurrent per-batch sgemm calls contend on the AMX and
/// lose to the sequential per-head loop (measured, byte-identical outputs).
fn biasedAttentionBatched(
    ctx: *ExecContext,
    q: *const fucina.Tensor(.{ .seq, .head, .d }),
    k: *const fucina.Tensor(.{ .seq, .kv_head, .d }),
    v: *const fucina.Tensor(.{ .seq, .kv_head, .d }),
    kv_heads: usize,
    heads_per_kv: usize,
    bias: *const UncondBias,
    scale: f32,
) !fucina.Tensor(.{ .seq, .attn }) {
    // Views only — the bmm dispatch materializes each operand once (each kv
    // head copied once, vs twice in the per-head loop).
    var q_split = try q.split(ctx, .head, .{ .kv_head, .g }, .{ kv_heads, heads_per_kv });
    defer q_split.deinit();
    var q4 = try q_split.permuteTo(ctx, .{ .kv_head, .g, .seq, .d });
    defer q4.deinit();
    var k_grouped = try k.permuteTo(ctx, .{ .kv_head, .seq, .d });
    defer k_grouped.deinit();
    var k4 = try k_grouped.insertAxis(ctx, .g, 1);
    defer k4.deinit();
    var v_grouped = try v.permuteTo(ctx, .{ .kv_head, .seq, .d });
    defer v_grouped.deinit();
    var v4 = try v_grouped.insertAxis(ctx, .g, 1);
    defer v4.deinit();

    var scores = try q4.matmul(ctx, &k4, .trans_b, .{ .kv_head, .g, .sq, .skv });
    defer scores.deinit();
    var probs = try scores.softmax(ctx, .skv, .{ .mask = bias, .scale = scale });
    defer probs.deinit();
    var out = try probs.matmul(ctx, &v4, .plain, .{ .kv_head, .g, .seq, .d });
    defer out.deinit();

    // [kv_head, g, seq, d] -> [seq, kv_head, g, d]: one materialize copy
    // reproduces the head-major concat order (head = kv_head*heads_per_kv+g).
    var out_perm = try out.permuteTo(ctx, .{ .seq, .kv_head, .g, .d });
    defer out_perm.deinit();
    var out_contig = try out_perm.materialize(ctx);
    defer out_contig.deinit();
    return out_contig.merge(ctx, .attn, .{ .kv_head, .g, .d });
}

/// The original per-head 2-D composition: per q head, scores = q_h·k_hᵀ,
/// probs = softmax(scale*scores + bias), out_h = probs·v_h, heads
/// concatenated on the feature axis. The production path on BLAS builds
/// (sequential sgemm calls keep the AMX uncontended) and the bit-exactness
/// oracle for biasedAttentionBatched (see the inline A/B test at the bottom
/// of this file).
fn biasedAttentionPerHead(
    ctx: *ExecContext,
    q: *const fucina.Tensor(.{ .seq, .head, .d }),
    k: *const fucina.Tensor(.{ .seq, .kv_head, .d }),
    v: *const fucina.Tensor(.{ .seq, .kv_head, .d }),
    kv_head_for_head: []const usize,
    bias: *const UncondBias,
    scale: f32,
) !fucina.Tensor(.{ .seq, .attn }) {
    const seq_len = q.dim(.seq);
    const num_heads = q.dim(.head);
    const head_dim = q.dim(.d);
    const attn_dim = num_heads * head_dim;

    const out_buf = try ctx.allocator.alloc(f32, seq_len * attn_dim);
    defer ctx.allocator.free(out_buf);

    for (0..num_heads) |head_i| {
        const kv_i = kv_head_for_head[head_i];

        var q_view = try q.narrow(ctx, .head, head_i, 1);
        defer q_view.deinit();
        var q_h = try q_view.squeeze(ctx, .head);
        defer q_h.deinit();
        var k_view = try k.narrow(ctx, .kv_head, kv_i, 1);
        defer k_view.deinit();
        var k_h = try k_view.squeeze(ctx, .kv_head);
        defer k_h.deinit();
        var v_view = try v.narrow(ctx, .kv_head, kv_i, 1);
        defer v_view.deinit();
        var v_h = try v_view.squeeze(ctx, .kv_head);
        defer v_h.deinit();

        var scores = try q_h.matmul(ctx, &k_h, .trans_b, .{ .sq, .skv });
        defer scores.deinit();
        var probs = try scores.softmax(ctx, .skv, .{ .mask = bias, .scale = scale });
        defer probs.deinit();
        var out_h = try probs.matmul(ctx, &v_h, .plain, .{ .seq, .d });
        defer out_h.deinit();

        const src = try out_h.dataConst();
        for (0..seq_len) |s| {
            @memcpy(out_buf[s * attn_dim + head_i * head_dim ..][0..head_dim], src[s * head_dim ..][0..head_dim]);
        }
    }

    return fucina.Tensor(.{ .seq, .attn }).fromSlice(ctx, .{ seq_len, attn_dim }, out_buf);
}

const QkvProjection = struct {
    q: fucina.Tensor(.{ .seq, .q }),
    k: fucina.Tensor(.{ .seq, .k }),
    v: fucina.Tensor(.{ .seq, .v }),

    fn deinit(self: *QkvProjection) void {
        self.v.deinit();
        self.k.deinit();
        self.q.deinit();
        self.* = undefined;
    }
};

const SeparateAttentionProjection = struct {
    q_proj: LinearWeight,
    k_proj: LinearWeight,
    v_proj: LinearWeight,

    fn deinit(self: *SeparateAttentionProjection) void {
        self.v_proj.deinit();
        self.k_proj.deinit();
        self.q_proj.deinit();
        self.* = undefined;
    }
};

/// q/k/v projections, output-stacked into ONE [q_dim+2*kv_dim, hidden] GEMM
/// weight when all three share a fusable dtype (qwen3's AttentionProjection
/// pattern). Per-output-element dot products are unchanged by row stacking, so
/// the fused path is bit-identical to the separate fallback.
const AttentionProjection = union(enum) {
    separate: SeparateAttentionProjection,
    fused: LinearWeight,

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize) !AttentionProjection {
        var name_buf: [96]u8 = undefined;
        const q_dim = config.num_attention_heads * config.head_dim;
        const kv_dim = config.num_key_value_heads * config.head_dim;

        var q_proj = try LinearWeight.loadForFusion(ctx, try file.get(try tensorName(&name_buf, layer_i, "self_attn.q_proj.weight")), q_dim, config.hidden_size);
        errdefer q_proj.deinit();

        var k_proj = try LinearWeight.loadForFusion(ctx, try file.get(try tensorName(&name_buf, layer_i, "self_attn.k_proj.weight")), kv_dim, config.hidden_size);
        errdefer k_proj.deinit();

        var v_proj = try LinearWeight.loadForFusion(ctx, try file.get(try tensorName(&name_buf, layer_i, "self_attn.v_proj.weight")), kv_dim, config.hidden_size);
        errdefer v_proj.deinit();

        var fuse_parts = [_]*LinearWeight{ &q_proj, &k_proj, &v_proj };
        if (shouldFuse(fuse_parts[0])) {
            if (try weights.fuseLinear(ctx, &fuse_parts)) |fused| {
                return .{ .fused = fused };
            }
        }
        return .{ .separate = .{
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
        } };
    }

    fn deinit(self: *AttentionProjection) void {
        switch (self.*) {
            .separate => |*separate| separate.deinit(),
            .fused => |*weight| weight.deinit(),
        }
        self.* = undefined;
    }

    fn project(self: *const AttentionProjection, ctx: *ExecContext, input: *const fucina.Tensor(.{ .seq, .embed }), config: *const Config) !QkvProjection {
        return switch (self.*) {
            .separate => |*separate| blk: {
                var q = try separate.q_proj.linearSeq(ctx, input, .embed, .q);
                errdefer q.deinit();
                var k = try separate.k_proj.linearSeq(ctx, input, .embed, .k);
                errdefer k.deinit();
                var v = try separate.v_proj.linearSeq(ctx, input, .embed, .v);
                errdefer v.deinit();
                break :blk .{ .q = q, .k = k, .v = v };
            },
            .fused => |*weight| blk: {
                var qkv = try weight.linearSeq(ctx, input, .embed, .qkv);
                defer qkv.deinit();
                break :blk try splitQkv(ctx, &qkv, config);
            },
        };
    }
};

fn splitQkv(ctx: *ExecContext, qkv: *const fucina.Tensor(.{ .seq, .qkv }), config: *const Config) !QkvProjection {
    const q_dim = config.num_attention_heads * config.head_dim;
    const kv_dim = config.num_key_value_heads * config.head_dim;

    var q_view = try qkv.narrow(ctx, .qkv, 0, q_dim);
    defer q_view.deinit();
    var q = try q_view.withTags(ctx, .{ .seq, .q });
    errdefer q.deinit();

    var k_view = try qkv.narrow(ctx, .qkv, q_dim, kv_dim);
    defer k_view.deinit();
    var k = try k_view.withTags(ctx, .{ .seq, .k });
    errdefer k.deinit();

    var v_view = try qkv.narrow(ctx, .qkv, q_dim + kv_dim, kv_dim);
    defer v_view.deinit();
    var v = try v_view.withTags(ctx, .{ .seq, .v });
    errdefer v.deinit();

    return .{ .q = q, .k = k, .v = v };
}

const SeparateFfnInputProjection = struct {
    gate_proj: LinearWeight,
    up_proj: LinearWeight,

    fn deinit(self: *SeparateFfnInputProjection) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.* = undefined;
    }
};

/// gate/up projections, output-stacked into ONE [2*ffn, hidden] GEMM weight
/// (fused layout [gate; up] — gate FIRST, qwen3's FfnInputProjection pattern)
/// when both share a fusable dtype.
const FfnInputProjection = union(enum) {
    separate: SeparateFfnInputProjection,
    fused: LinearWeight,

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize) !FfnInputProjection {
        var name_buf: [96]u8 = undefined;

        var gate_proj = try LinearWeight.loadForFusion(ctx, try file.get(try tensorName(&name_buf, layer_i, "mlp.gate_proj.weight")), config.intermediate_size, config.hidden_size);
        errdefer gate_proj.deinit();

        var up_proj = try LinearWeight.loadForFusion(ctx, try file.get(try tensorName(&name_buf, layer_i, "mlp.up_proj.weight")), config.intermediate_size, config.hidden_size);
        errdefer up_proj.deinit();

        var fuse_parts = [_]*LinearWeight{ &gate_proj, &up_proj };
        if (shouldFuse(fuse_parts[0])) {
            if (try weights.fuseLinear(ctx, &fuse_parts)) |fused| {
                return .{ .fused = fused };
            }
        }
        return .{ .separate = .{
            .gate_proj = gate_proj,
            .up_proj = up_proj,
        } };
    }

    fn deinit(self: *FfnInputProjection) void {
        switch (self.*) {
            .separate => |*separate| separate.deinit(),
            .fused => |*weight| weight.deinit(),
        }
        self.* = undefined;
    }
};

pub const Layer = struct {
    attn_norm: fucina.Tensor(.{.embed}), // input_layernorm.weight
    q_norm: fucina.Tensor(.{.d}),
    k_norm: fucina.Tensor(.{.d}),
    ffn_norm: fucina.Tensor(.{.embed}), // post_attention_layernorm.weight
    attn_proj: AttentionProjection, // q [2048,1024] + k/v [1024,1024], fused when same-dtype
    o_proj: LinearWeight, // [hidden, heads*d] = [1024, 2048]
    ffn_input: FfnInputProjection, // gate + up [ffn, hidden], fused when same-dtype
    down_proj: LinearWeight, // [hidden, ffn]

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize) !Layer {
        var name_buf: [96]u8 = undefined;
        const q_dim = config.num_attention_heads * config.head_dim;

        var attn_norm = try weights.loadVector(ctx, try file.get(try tensorName(&name_buf, layer_i, "input_layernorm.weight")), config.hidden_size, .embed);
        errdefer attn_norm.deinit();
        var q_norm = try weights.loadVector(ctx, try file.get(try tensorName(&name_buf, layer_i, "self_attn.q_norm.weight")), config.head_dim, .d);
        errdefer q_norm.deinit();
        var k_norm = try weights.loadVector(ctx, try file.get(try tensorName(&name_buf, layer_i, "self_attn.k_norm.weight")), config.head_dim, .d);
        errdefer k_norm.deinit();
        var ffn_norm = try weights.loadVector(ctx, try file.get(try tensorName(&name_buf, layer_i, "post_attention_layernorm.weight")), config.hidden_size, .embed);
        errdefer ffn_norm.deinit();

        var attn_proj = try AttentionProjection.load(ctx, file, config, layer_i);
        errdefer attn_proj.deinit();
        var o_proj = try LinearWeight.load(ctx, try file.get(try tensorName(&name_buf, layer_i, "self_attn.o_proj.weight")), config.hidden_size, q_dim);
        errdefer o_proj.deinit();

        var ffn_input = try FfnInputProjection.load(ctx, file, config, layer_i);
        errdefer ffn_input.deinit();
        var down_proj = try LinearWeight.load(ctx, try file.get(try tensorName(&name_buf, layer_i, "mlp.down_proj.weight")), config.hidden_size, config.intermediate_size);
        errdefer down_proj.deinit();

        return .{
            .attn_norm = attn_norm,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .ffn_norm = ffn_norm,
            .attn_proj = attn_proj,
            .o_proj = o_proj,
            .ffn_input = ffn_input,
            .down_proj = down_proj,
        };
    }

    fn deinit(self: *Layer) void {
        self.down_proj.deinit();
        self.ffn_input.deinit();
        self.o_proj.deinit();
        self.attn_proj.deinit();
        self.ffn_norm.deinit();
        self.k_norm.deinit();
        self.q_norm.deinit();
        self.attn_norm.deinit();
        self.* = undefined;
    }
};

/// HF-style GGUF tensor names: `llm.layers.{i}.{suffix}` (NOT `blk.*`).
fn tensorName(buf: []u8, layer_i: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "llm.layers.{d}.{s}", .{ layer_i, suffix });
}

test {
    _ = @import("lm_tests.zig");
}

// Inline (needs the file-private attention functions — policy: never add
// `pub` just to move a test): the batched broadcast-bmm attention must be
// BYTEWISE identical to the per-head reference composition on every backend.
test "biasedAttentionBatched is bytewise identical to the per-head reference" {
    var ctx: fucina.ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // Case 1: odd sizes (d not a multiple of the SIMD width), total bmm work
    // below bmm_loop_work_threshold -> serial broadcast loop. Case 2: work
    // above the threshold -> parallel batch loop; wider g grouping.
    const cases = [_]struct { heads: usize, kv_heads: usize, d: usize, seq: usize, u_len: usize }{
        .{ .heads = 6, .kv_heads = 3, .d = 24, .seq = 19, .u_len = 7 },
        .{ .heads = 8, .kv_heads = 2, .d = 32, .seq = 64, .u_len = 20 },
    };
    var prng = std.Random.DefaultPrng.init(0x0b1a5ed);
    const random = prng.random();
    const allocator = std.testing.allocator;

    for (cases) |case| {
        const heads_per_kv = case.heads / case.kv_heads;

        const q_data = try allocator.alloc(f32, case.seq * case.heads * case.d);
        defer allocator.free(q_data);
        const k_data = try allocator.alloc(f32, case.seq * case.kv_heads * case.d);
        defer allocator.free(k_data);
        const v_data = try allocator.alloc(f32, case.seq * case.kv_heads * case.d);
        defer allocator.free(v_data);
        for (q_data) |*x| x.* = random.floatNorm(f32);
        for (k_data) |*x| x.* = random.floatNorm(f32);
        for (v_data) |*x| x.* = random.floatNorm(f32);

        var q = try fucina.Tensor(.{ .seq, .head, .d }).fromSlice(&ctx, .{ case.seq, case.heads, case.d }, q_data);
        defer q.deinit();
        var k = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ case.seq, case.kv_heads, case.d }, k_data);
        defer k.deinit();
        var v = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ case.seq, case.kv_heads, case.d }, v_data);
        defer v.deinit();
        var bias = try buildUncondBias(&ctx, case.seq, case.u_len);
        defer bias.deinit();

        const kv_head_for_head = try allocator.alloc(usize, case.heads);
        defer allocator.free(kv_head_for_head);
        for (kv_head_for_head, 0..) |*kv_i, head_i| kv_i.* = head_i / heads_per_kv;

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(case.d)));
        var want = try biasedAttentionPerHead(&ctx, &q, &k, &v, kv_head_for_head, &bias, scale);
        defer want.deinit();
        var got = try biasedAttentionBatched(&ctx, &q, &k, &v, case.kv_heads, heads_per_kv, &bias, scale);
        defer got.deinit();

        try std.testing.expectEqualSlices(
            u8,
            std.mem.sliceAsBytes(try want.dataConst()),
            std.mem.sliceAsBytes(try got.dataConst()),
        );
    }
}
