//! DiffusionGemma (`diffusion-gemma` GGUF arch) — block text-diffusion on the
//! Gemma 4 26B-A4B backbone, text-only CPU inference.
//!
//! Sources of truth: llama.cpp draft PR #24423 (`src/models/diffusion-gemma.cpp`
//! + `examples/diffusion/diffusion.cpp`, author-verified logit parity vs
//! transformers) and the HF reference (`modeling/generation_diffusion_gemma.py`).
//!
//! The transformer is EXACTLY gemma4 (encoder and decoder share all weights);
//! this file reuses gemma4.zig's layer loader and attn/ffn blocks. The deltas:
//!   - +34 tensors: per-layer `enc_layer_output_scale` (the encoder-pass
//!     variant of `layer_output_scale`) and the `self_cond_*` GeGLU MLP.
//!   - Two forward modes over one weight set:
//!       encodeStep   — causal prefix pass (prompt, then each finalized
//!                      canvas); appends K/V and advances the cache; per-layer
//!                      scale = enc_layer_output_scale; NO logits (the lm head
//!                      is skipped — only the KV matters).
//!       canvasForward — bidirectional denoiser pass over the canvas_length
//!                      canvas at absolute positions [P, P+C): canvas K/V are
//!                      written into the cache's scratch region [P, P+C)
//!                      WITHOUT advancing kv.len (appendLayer writes at
//!                      kv.len; the next step overwrites), every canvas query
//!                      attends keys [lo, P+C) with lo = (P+1)-|window on SWA
//!                      layers (llama.cpp: "last (n_swa-1) prompt positions +
//!                      all canvas") else 0 — one contiguous range, so the
//!                      bidirectional attention op needs no mask; per-layer
//!                      scale = layer_output_scale; logits for EVERY row.
//!   - Canvas embedding: rms_norm_noscale(embed*sqrt(E) + SC); SC = the
//!     self-conditioning signal from the previous step's logits (zero on the
//!     first step — the rms_norm still applies).
//!   - The entropy-bound (EB) sampler: uniform-random canvas init, per-step
//!     temperature schedule t_max→t_min, per-position entropy + multinomial
//!     draw, lowest-entropy acceptance under the mutual-information bound,
//!     full renoise of the rest, stable+confident adaptive stop.
//!
//! Self-conditioning on CPU: the reference computes softmax(prev_logits/t) @
//! token_embd densely (a 256x262144x2816 GEMM vs a dequantized-transposed
//! embedding). Here the sampler pass — which already computes exactly that
//! softmax row-by-row — collects the ids with p >= sc_p_min (the
//! distributions are near-one-hot at t <= 0.8), renormalizes over the kept
//! mass, and the canvas pass gathers + weight-sums just those embedding rows.
//! The downstream `sc_pre_norm` is an rms norm, so the kept-mass
//! renormalization error largely cancels; validated against the llama.cpp
//! dense oracle (the dense-equivalent path stays available for parity
//! re-runs via the eval harness's `--sc-logits` flag).
const std = @import("std");
const fucina = @import("fucina");
const gemma4 = @import("../gemma/gemma4.zig");
const weights = @import("../weights.zig");
const kv_cache = @import("../kv_cache.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;
const gguf = fucina.gguf;
const rng = fucina.rng;
const LinearWeight = weights.LinearWeight;

pub const Error = gemma4.Error || error{
    MissingCanvasLength,
    MissingLayerScale,
    CanvasLengthMismatch,
    KvCapacityTooSmall,
    SelfConditioningUnavailable,
};

const Vec = fucina.Tensor(.{.embed});

/// Entropy-bound sampler parameters. Defaults are the reference values from
/// the model's generation_config.json (also llama.cpp PR #24423's defaults);
/// GGUF `diffusion.eb_*` keys override when present.
pub const EbParams = struct {
    max_steps: usize = 48,
    t_min: f32 = 0.4, // temperature at the LAST denoising step
    t_max: f32 = 0.8, // temperature at the FIRST denoising step
    entropy_bound: f32 = 0.1,
    stability_threshold: usize = 1,
    confidence_threshold: f32 = 0.005,
};

pub const Config = struct {
    base: gemma4.Config,
    canvas_length: usize,
    eb: EbParams,

    pub fn fromGguf(file: *const gguf.File) !Config {
        const base = try gemma4.Config.fromGgufArch(file, "diffusion-gemma");
        const canvas_raw = file.getInt("diffusion.canvas_length") orelse return Error.MissingCanvasLength;
        if (canvas_raw <= 0) return Error.MissingCanvasLength;

        var eb = EbParams{};
        if (file.getInt("diffusion.eb_max_steps")) |v| {
            if (v > 0) eb.max_steps = @intCast(v);
        }
        if (file.getFloat("diffusion.eb_t_min")) |v| eb.t_min = @floatCast(v);
        if (file.getFloat("diffusion.eb_t_max")) |v| eb.t_max = @floatCast(v);
        if (file.getFloat("diffusion.eb_entropy_bound")) |v| eb.entropy_bound = @floatCast(v);
        if (file.getInt("diffusion.eb_stability_threshold")) |v| {
            if (v >= 0) eb.stability_threshold = @intCast(v);
        }
        if (file.getFloat("diffusion.eb_confidence_threshold")) |v| eb.confidence_threshold = @floatCast(v);

        return .{
            .base = base,
            .canvas_length = @intCast(canvas_raw),
            .eb = eb,
        };
    }
};

/// Self-conditioning MLP weights (decoder-only; optional in the GGUF — the
/// zero-SC forward is exact without them).
const SelfCond = struct {
    pre_norm: Vec,
    gate: LinearWeight,
    up: LinearWeight,
    down: LinearWeight,

    fn deinit(self: *SelfCond) void {
        self.down.deinit();
        self.up.deinit();
        self.gate.deinit();
        self.pre_norm.deinit();
        self.* = undefined;
    }
};

pub const Model = struct {
    allocator: Allocator,
    config: Config,
    geom: gemma4.LayerGeometry,
    token_embedding: LinearWeight,
    output_norm: Vec,
    output: LinearWeight,
    rope_freqs: ?fucina.Tensor(.{.rope}),
    layers: []gemma4.Layer,
    /// Per-layer encoder-pass output scale (`blk.N.enc_layer_output_scale`).
    /// The canvas-pass scale lives in `layers[il].out_scale` (required here,
    /// unlike plain gemma4 where it is optional).
    enc_scale: []f32,
    sc: ?SelfCond,
    /// GGUF mapping owned by the model when MoE experts borrow from it
    /// (`--experts=borrow`); unmapped last in deinit. null on the packed path.
    weight_mapping: ?gguf.File.MappedRegion = null,

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, config: Config) !Model {
        var file = try gguf.File.loadMmap(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFile(ctx, &file, config);
    }

    pub fn loadGgufFromFile(ctx: *ExecContext, file: *gguf.File, config: Config) !Model {
        try config.base.validate();
        // The DiffusionGemma family has no per-layer-embeddings arm; the
        // forward modes here do not implement PLE injection.
        if (config.base.per_layer_input_size != 0) return Error.InvalidConfig;
        const allocator = ctx.allocator;

        const swa_pattern = try gemma4.readU32OrBoolArray(allocator, file, "diffusion-gemma.attention.sliding_window_pattern", config.base.num_layers, bool);
        defer allocator.free(swa_pattern);
        const kv_heads = try gemma4.readU32OrBoolArray(allocator, file, "diffusion-gemma.attention.head_count_kv", config.base.num_layers, usize);
        defer allocator.free(kv_heads);
        for (kv_heads) |kvh| {
            if (kvh == 0 or config.base.num_attention_heads % kvh != 0) return Error.InvalidConfig;
        }

        var geom = try gemma4.deriveGeometry(allocator, config.base.num_layers, swa_pattern, kv_heads, config.base.shared_kv_layers, config.base.head_dim_global, config.base.head_dim_swa);
        errdefer geom.deinit(allocator);

        var token_embedding = try LinearWeight.load(ctx, try file.get("token_embd.weight"), config.base.vocab_size, config.base.hidden_size);
        errdefer token_embedding.deinit();

        var output_norm = try weights.loadVector(ctx, try file.get("output_norm.weight"), config.base.hidden_size, .embed);
        errdefer output_norm.deinit();

        var output = if (file.maybeGet("output.weight")) |info|
            try LinearWeight.load(ctx, info, config.base.vocab_size, config.base.hidden_size)
        else
            try token_embedding.cloneView(ctx);
        errdefer output.deinit();

        var rope_freqs: ?fucina.Tensor(.{.rope}) = null;
        if (file.maybeGet("rope_freqs.weight")) |info|
            rope_freqs = try weights.loadVector(ctx, info, config.base.head_dim_global / 2, .rope);
        errdefer if (rope_freqs) |*t| t.deinit();

        var sc: ?SelfCond = null;
        errdefer if (sc) |*s| s.deinit();
        if (file.maybeGet("self_cond_pre_norm.weight")) |pre_info| {
            var pre_norm = try weights.loadVector(ctx, pre_info, config.base.hidden_size, .embed);
            errdefer pre_norm.deinit();
            var gate = try LinearWeight.load(ctx, try file.get("self_cond_gate.weight"), config.base.intermediate_size, config.base.hidden_size);
            errdefer gate.deinit();
            var up = try LinearWeight.load(ctx, try file.get("self_cond_up.weight"), config.base.intermediate_size, config.base.hidden_size);
            errdefer up.deinit();
            var down = try LinearWeight.load(ctx, try file.get("self_cond_down.weight"), config.base.hidden_size, config.base.intermediate_size);
            errdefer down.deinit();
            sc = .{ .pre_norm = pre_norm, .gate = gate, .up = up, .down = down };
        }

        const layers = try allocator.alloc(gemma4.Layer, config.base.num_layers);
        errdefer allocator.free(layers);
        try gemma4.loadLayers(ctx, file, config.base, geom, layers);
        errdefer for (layers) |*layer| layer.deinit(allocator);

        // Both per-layer scales are REQUIRED for this arch (llama.cpp marks
        // them non-optional): the decoder scale was loaded by loadLayers into
        // out_scale; the encoder scale is the diffusion-only extra tensor.
        const enc_scale = try allocator.alloc(f32, config.base.num_layers);
        errdefer allocator.free(enc_scale);
        var nb: [96]u8 = undefined;
        for (0..config.base.num_layers) |il| {
            if (layers[il].out_scale == null) return Error.MissingLayerScale;
            const info = file.maybeGet(try weights.layerName(&nb, il, "enc_layer_output_scale.weight")) orelse return Error.MissingLayerScale;
            var t = try weights.loadVector(ctx, info, 1, .scalar);
            defer t.deinit();
            enc_scale[il] = try t.item();
        }

        // Experts borrow from the mapping (gemma4.loadMoe) when requested; the
        // model then owns it. The default packed path leaves nothing mapped.
        const weight_mapping = if (config.base.num_experts > 0 and config.base.borrow_experts) file.takeMapping() else null;

        return .{
            .allocator = allocator,
            .config = config,
            .geom = geom,
            .token_embedding = token_embedding,
            .output_norm = output_norm,
            .output = output,
            .rope_freqs = rope_freqs,
            .layers = layers,
            .enc_scale = enc_scale,
            .sc = sc,
            .weight_mapping = weight_mapping,
        };
    }

    pub fn deinit(self: *Model) void {
        for (self.layers) |*layer| layer.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.allocator.free(self.enc_scale);
        if (self.sc) |*s| s.deinit();
        if (self.rope_freqs) |*t| t.deinit();
        self.output.deinit();
        self.output_norm.deinit();
        self.token_embedding.deinit();
        self.geom.deinit(self.allocator);
        // Unmap LAST: borrowed expert blocks point into this region.
        if (self.weight_mapping) |*m| m.deinit();
        self.* = undefined;
    }

    /// KV capacity must cover prefix + one canvas: the canvas pass writes its
    /// K/V into [kv.len, kv.len + canvas_length) without advancing.
    pub fn initKvCache(self: *const Model, ctx: *ExecContext, capacity: usize) !KvCache {
        return KvCache.initPerLayer(ctx, self.geom.kv_heads, self.geom.head_dim, capacity);
    }

    /// Dequantize the dense-GEMM-heavy weights to RESIDENT f16 — attention
    /// q/k/v/o, the shared dense FFN, the self-conditioning MLP, and the lm
    /// head — so the canvas forward's big matmuls take the f16-operands NT
    /// path (the `-Dgpu=metal` f16 GEMM offload; without a GPU build this
    /// path is the slower CPU f16 kernel — don't enable it then). Every
    /// canvas-step matmul on these weights is prefill-shaped (m = 256), so
    /// they all clear the GPU work gate. The MoE experts (the bulk of the
    /// parameters but only ~2% of the canvas-step FLOPs — 8-of-128 sparse)
    /// stay on the packed quant kernels. Adds ~4.6 GB resident on 26B-A4B
    /// (the embedding gather stays quantized: `output` is converted via its
    /// own arm, the tied `token_embedding` is untouched).
    pub fn convertDenseWeightsToF16(self: *Model, ctx: *ExecContext) !void {
        try self.output.toResidentF16(ctx);
        if (self.sc) |*sc| {
            try sc.gate.toResidentF16(ctx);
            try sc.up.toResidentF16(ctx);
            try sc.down.toResidentF16(ctx);
        }
        for (self.layers) |*layer| {
            try layer.attn_proj.toResidentF16(ctx);
            try layer.o_proj.toResidentF16(ctx);
            try layer.ffn_gate.toResidentF16(ctx);
            try layer.ffn_up.toResidentF16(ctx);
            try layer.ffn_down.toResidentF16(ctx);
        }
    }

    /// Causal encoder pass: append `token_ids` (the prompt, or a finalized
    /// canvas) to the KV cache and advance it. The lm head is skipped — the
    /// pass exists for its K/V side effect (llama.cpp PREFILL phase).
    pub fn encodeStep(
        self: *const Model,
        ctx: *ExecContext,
        kv: *KvCache,
        token_ids: []const usize,
        pos0: usize,
    ) !void {
        if (token_ids.len == 0) return Error.InvalidSequenceLength;
        try gemma4.requireF16KvCache(kv);
        if (kv.len != pos0) return Error.InvalidSequenceLength;
        if (kv.len + token_ids.len > kv.capacity) return kv_cache.Error.KvCacheOverflow;

        const cfg = self.config.base;
        const allocator = ctx.allocator;

        const positions = try allocator.alloc(i32, token_ids.len);
        defer allocator.free(positions);
        for (positions, 0..) |*p, i| p.* = @intCast(pos0 + i);

        const factors: ?[]const f32 = if (self.rope_freqs) |*t| try t.dataConst() else null;
        var swa_table = try ctx.prepareRopeTable(positions, cfg.head_dim_swa, cfg.rope_theta_swa, false);
        defer swa_table.deinit();
        var global_table = try ctx.prepareRopeTableFactors(positions, cfg.head_dim_global, cfg.rope_theta, false, factors);
        defer global_table.deinit();

        var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
        defer x.deinit();
        x = try ctx.replace(x, x.scale(ctx, @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)))));

        for (self.layers, 0..) |*layer, il| {
            x = try ctx.replace(x, gemma4.attnBlock(ctx, cfg, self.geom, layer, il, &x, &swa_table, &global_table, false, kv));
            x = try ctx.replace(x, gemma4.ffnBlock(ctx, null, cfg, layer, &x, null));
            x = try ctx.replace(x, x.scale(ctx, self.enc_scale[il]));
        }
        kv.advance(token_ids.len);
    }

    /// One bidirectional denoiser pass over the canvas at absolute positions
    /// [kv.len, kv.len + C). Returns `[C, vocab]` logits (softcapped). The KV
    /// cache is read-only from the caller's perspective: canvas K/V occupy
    /// the scratch region past kv.len and kv.len is unchanged on return.
    /// `sc` carries the previous step's self-conditioning signal (null on the
    /// first step — the zero-SC embedding path still rms-normalizes).
    pub fn canvasForward(
        self: *const Model,
        ctx: *ExecContext,
        kv: *KvCache,
        canvas_ids: []const usize,
        sc: ?*const ScSignal,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        const cfg = self.config.base;
        const c_len = canvas_ids.len;
        if (c_len == 0) return Error.InvalidSequenceLength;
        try gemma4.requireF16KvCache(kv);
        if (kv.len + c_len > kv.capacity) return kv_cache.Error.KvCacheOverflow;
        if (sc != null and self.sc == null) return Error.SelfConditioningUnavailable;

        const allocator = ctx.allocator;
        const prefix_len = kv.len;

        const positions = try allocator.alloc(i32, c_len);
        defer allocator.free(positions);
        for (positions, 0..) |*p, i| p.* = @intCast(prefix_len + i);

        const factors: ?[]const f32 = if (self.rope_freqs) |*t| try t.dataConst() else null;
        var swa_table = try ctx.prepareRopeTable(positions, cfg.head_dim_swa, cfg.rope_theta_swa, false);
        defer swa_table.deinit();
        var global_table = try ctx.prepareRopeTableFactors(positions, cfg.head_dim_global, cfg.rope_theta, false, factors);
        defer global_table.deinit();

        var x = try self.canvasEmbed(ctx, canvas_ids, sc);
        defer x.deinit();

        for (self.layers, 0..) |*layer, il| {
            x = try ctx.replace(x, self.canvasAttnBlock(ctx, layer, il, &x, &swa_table, &global_table, kv));
            x = try ctx.replace(x, gemma4.ffnBlock(ctx, null, cfg, layer, &x, null));
            x = try ctx.replace(x, x.scale(ctx, layer.out_scale.?));
        }

        var final_norm = try x.rmsNormMul(ctx, .embed, &self.output_norm, cfg.rms_norm_eps);
        defer final_norm.deinit();

        var logits = try self.output.linearSeq(ctx, &final_norm, .embed, .vocab);
        if (cfg.final_logit_softcapping != 0) {
            const cap = cfg.final_logit_softcapping;
            var down = try logits.scale(ctx, 1.0 / cap);
            logits.deinit();
            defer down.deinit();
            var t = try down.tanh(ctx);
            defer t.deinit();
            return t.scale(ctx, cap);
        }
        return logits;
    }

    /// Canvas embedding: scaled token rows, plus the self-conditioning signal
    /// when present, then the no-scale rms norm (applied in BOTH cases — the
    /// zero-SC path is `rms_norm(embed * sqrt(E))`).
    fn canvasEmbed(
        self: *const Model,
        ctx: *ExecContext,
        canvas_ids: []const usize,
        sc: ?*const ScSignal,
    ) !fucina.Tensor(.{ .seq, .embed }) {
        const cfg = self.config.base;
        const embed_scale = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));

        var x = try self.token_embedding.getRowsAs(ctx, canvas_ids, .embed);
        errdefer x.deinit();
        x = try ctx.replace(x, x.scale(ctx, embed_scale));

        if (sc) |signal| {
            const sc_w = &self.sc.?;
            var soft = try self.softEmbedding(ctx, canvas_ids.len, signal);
            defer soft.deinit();
            soft = try ctx.replace(soft, soft.scale(ctx, embed_scale));

            var normed = try soft.rmsNormMul(ctx, .embed, &sc_w.pre_norm, cfg.rms_norm_eps);
            defer normed.deinit();
            var gate = try sc_w.gate.linearSeq(ctx, &normed, .embed, .ffn);
            defer gate.deinit();
            var up = try sc_w.up.linearSeq(ctx, &normed, .embed, .ffn);
            defer up.deinit();
            // Same ggml-matching f16-LUT tanh-gelu the backbone's GeGLU uses
            // (llama.cpp builds this block with ggml_gelu).
            var gate_act = try gate.unary(ctx, .gelu_quant);
            defer gate_act.deinit();
            var gated = try up.mul(ctx, &gate_act);
            defer gated.deinit();
            var sig = try sc_w.down.linearSeq(ctx, &gated, .ffn, .embed);
            defer sig.deinit();
            x = try ctx.replace(x, x.add(ctx, &sig));
        }
        return ctx.replace(x, x.rmsNorm(ctx, .embed, cfg.rms_norm_eps));
    }

    /// soft_emb[c,:] = sum_i p_i * token_embd[id_i,:] over the sparse per-row
    /// candidate lists (one batched row gather; probs pre-renormalized by the
    /// sampler over the kept mass).
    fn softEmbedding(
        self: *const Model,
        ctx: *ExecContext,
        c_len: usize,
        signal: *const ScSignal,
    ) !fucina.Tensor(.{ .seq, .embed }) {
        if (signal.row_offsets.len != c_len + 1) return Error.CanvasLengthMismatch;
        const hidden = self.config.base.hidden_size;
        const allocator = ctx.allocator;

        var rows = try self.token_embedding.getRowsAs(ctx, signal.ids, .embed);
        defer rows.deinit();
        const rows_data = try rows.dataConst();

        const acc = try allocator.alloc(f32, c_len * hidden);
        defer allocator.free(acc);
        @memset(acc, 0);
        for (0..c_len) |c| {
            const out_row = acc[c * hidden ..][0..hidden];
            for (signal.row_offsets[c]..signal.row_offsets[c + 1]) |j| {
                const p = signal.probs[j];
                const src = rows_data[j * hidden ..][0..hidden];
                for (out_row, src) |*a, v| a.* += p * v;
            }
        }
        return fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ c_len, hidden }, acc);
    }

    /// gemma4.attnBlock with the canvas-pass deltas: K/V appended into the
    /// scratch region (kv.len unchanged), keys narrowed to [lo, P+C) with the
    /// SWA prompt-reach lower bound, bidirectional attention, no
    /// last-query-only fast path (every row's logits are needed).
    fn canvasAttnBlock(
        self: *const Model,
        ctx: *ExecContext,
        layer: *const gemma4.Layer,
        il: usize,
        input: *const fucina.Tensor(.{ .seq, .embed }),
        swa_table: *const fucina.RopeTable,
        global_table: *const fucina.RopeTable,
        kv: *KvCache,
    ) !fucina.Tensor(.{ .seq, .embed }) {
        const cfg = self.config.base;
        const geom = self.geom;
        const head_dim = geom.head_dim[il];
        const n_head = cfg.num_attention_heads;
        const n_kv = geom.kv_heads[il];
        const q_dim = n_head * head_dim;
        const kv_dim = n_kv * head_dim;
        const table = if (geom.is_swa[il]) swa_table else global_table;
        const m = input.dim(.seq);

        var kvhh: [gemma4.max_heads]usize = undefined;
        const heads_per_kv = n_head / n_kv;
        for (0..n_head) |h| kvhh[h] = h / heads_per_kv;
        const kv_head_for_head = kvhh[0..n_head];

        var attn_in = try input.rmsNormMul(ctx, .embed, &layer.attn_norm, cfg.rms_norm_eps);
        defer attn_in.deinit();

        var proj = try layer.attn_proj.project(ctx, &attn_in, q_dim, kv_dim);
        defer proj.deinit();
        var q3 = try proj.q.split(ctx, .q, .{ .head, .d }, .{ n_head, head_dim });
        defer q3.deinit();
        var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, cfg.rms_norm_eps, table);
        defer q_rope.deinit();

        if (geom.has_kv[il]) {
            var k3 = try proj.k.?.split(ctx, .k, .{ .kv_head, .d }, .{ n_kv, head_dim });
            defer k3.deinit();
            var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm.?, cfg.rms_norm_eps, table);
            defer k_rope.deinit();

            var v3 = blk: {
                if (proj.v) |*v| {
                    break :blk try v.split(ctx, .v, .{ .kv_head, .d }, .{ n_kv, head_dim });
                } else {
                    break :blk try k3.withTags(ctx, .{ .seq, .kv_head, .d });
                }
            };
            defer v3.deinit();
            var v_norm = try v3.rmsNorm(ctx, .d, cfg.rms_norm_eps);
            defer v_norm.deinit();

            // Writes rows [kv.len, kv.len + m); kv.len is NOT advanced — the
            // canvas scratch region is overwritten by the next denoise step.
            try kv.appendLayer(ctx, il, &k_rope, &v_norm);
        }

        const ref = geom.kv_ref[il];
        const prefix_len = kv.len;
        const cached_len = prefix_len + m;
        // SWA canvas reach: the last (window-1) prompt positions + the whole
        // canvas (llama.cpp: allow = k_is_canvas or k >= P - n_swa + 1).
        const lo = if (geom.is_swa[il]) (prefix_len + 1) -| cfg.sliding_window else 0;
        var k_view = try kv.k[ref].narrow(ctx, .seq, lo, cached_len - lo);
        defer k_view.deinit();
        var v_view = try kv.v[ref].narrow(ctx, .seq, lo, cached_len - lo);
        defer v_view.deinit();

        var attn = try q_rope.groupedAttention(
            ctx,
            &k_view,
            &v_view,
            kv_head_for_head,
            .attn,
            1.0, // Gemma 4 family: softmax scale = 1.0 (f_attention_scale)
            .{ .mask = .bidirectional },
        );
        defer attn.deinit();

        var attn_out = try layer.o_proj.linearSeq(ctx, &attn, .attn, .embed);
        defer attn_out.deinit();
        var post = try attn_out.rmsNormMul(ctx, .embed, &layer.attn_post_norm, cfg.rms_norm_eps);
        defer post.deinit();
        return input.add(ctx, &post);
    }
};

// ---------------------------------------------------------------------------
// Self-conditioning signal (sparse softmax(prev_logits / t_prev))
// ---------------------------------------------------------------------------

/// Sparse per-row probability lists feeding the next step's soft embedding.
/// Row c's candidates are `ids/probs[row_offsets[c]..row_offsets[c+1]]`;
/// probs are renormalized over the kept mass (the dropped tail is below
/// `sc_p_min` per token; `sc_pre_norm`'s rms normalization makes the result
/// insensitive to the missing mass).
pub const ScSignal = struct {
    ids: []usize,
    probs: []f32,
    row_offsets: []usize,
    allocator: Allocator,

    pub fn deinit(self: *ScSignal) void {
        self.allocator.free(self.ids);
        self.allocator.free(self.probs);
        self.allocator.free(self.row_offsets);
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// Entropy-bound sampler
// ---------------------------------------------------------------------------

/// Per-position results of one sampler pass over the `[C, vocab]` logits.
const RowResult = struct {
    argmax: usize,
    sampled: usize,
    entropy: f32, // nats, of softmax(z / t)
};

pub const SamplerOptions = struct {
    /// Keep softmax(z/t) candidates with p >= sc_p_min for self-conditioning
    /// (cap sc_max_per_row ids/row, highest-p kept via the threshold pass).
    sc_p_min: f32 = 1e-6,
    sc_max_per_row: usize = 512,
    collect_sc: bool = true,
};

/// One sampler pass: per canvas position over the vocab row, with `z` the raw
/// logit and `t` this step's temperature — argmax, entropy of softmax(z/t),
/// one multinomial draw (uniform `u` pre-drawn by the caller), and the sparse
/// SC candidate list for the NEXT step (which conditions on softmax(z/t) with
/// exactly this step's t — llama.cpp passes prev_temp_inv = 1/t).
pub const SamplerPass = struct {
    results: []RowResult,
    sc: ?ScSignal,

    pub fn deinit(self: *SamplerPass, allocator: Allocator) void {
        allocator.free(self.results);
        if (self.sc) |*s| s.deinit();
        self.* = undefined;
    }
};

const RowTask = struct {
    logits: []const f32,
    vocab: usize,
    temp_inv: f32,
    u: []const f32,
    results: []RowResult,
    sc_p_min: f32,
    sc_cap: usize,
    sc_ids: []usize, // [c_len * sc_cap], 0-length when SC is off
    sc_probs: []f32,
    sc_counts: []usize,
    row_start: usize,
    row_end: usize,
};

fn runRowTask(task: *const RowTask) void {
    const v = task.vocab;
    for (task.row_start..task.row_end) |c| {
        const row = task.logits[c * v ..][0..v];
        // Pass 1: max + argmax of z' = z * temp_inv.
        var m: f32 = -std.math.inf(f32);
        var argmax: usize = 0;
        for (row, 0..) |z, i| {
            const zt = z * task.temp_inv;
            if (zt > m) {
                m = zt;
                argmax = i;
            }
        }
        // Pass 2: Z and the entropy accumulator (f64 — 262k-term sums).
        var z_sum: f64 = 0;
        var s1: f64 = 0; // sum e^{z'-m} (z'-m)
        for (row) |z| {
            const d = z * task.temp_inv - m;
            const e = @exp(@as(f64, d));
            z_sum += e;
            s1 += e * d;
        }
        const entropy: f32 = @floatCast(@log(z_sum) - s1 / z_sum);
        // Pass 3: the multinomial draw (first index whose cumulative mass
        // reaches u*Z, vocab order — llama.cpp's exact scheme) + the SC
        // candidate collection (p >= p_min).
        const target = @as(f64, task.u[c]) * z_sum;
        var cum: f64 = 0;
        var sampled: usize = v - 1;
        var found = false;
        const collect = task.sc_cap > 0;
        const sc_base = c * task.sc_cap;
        var sc_n: usize = 0;
        var sc_kept: f64 = 0;
        const p_min_e = @as(f64, task.sc_p_min) * z_sum; // p >= p_min ⇔ e^{z'-m} >= p_min*Z
        for (row, 0..) |z, i| {
            const e = @exp(@as(f64, z * task.temp_inv - m));
            cum += e;
            if (!found and cum >= target) {
                sampled = i;
                found = true;
            }
            if (collect and e >= p_min_e and sc_n < task.sc_cap) {
                task.sc_ids[sc_base + sc_n] = i;
                task.sc_probs[sc_base + sc_n] = @floatCast(e);
                sc_kept += e;
                sc_n += 1;
            }
        }
        if (collect) {
            // Renormalize the kept mass (see ScSignal docs). A row can never
            // be empty: the argmax token has p >= 1/vocab >= p_min in any
            // realistic configuration, but guard anyway.
            if (sc_n == 0) {
                task.sc_ids[sc_base] = argmax;
                task.sc_probs[sc_base] = 1.0;
                sc_n = 1;
            } else {
                const inv: f32 = @floatCast(1.0 / sc_kept);
                for (task.sc_probs[sc_base..][0..sc_n]) |*p| p.* *= inv;
            }
            task.sc_counts[c] = sc_n;
        }
        task.results[c] = .{ .argmax = argmax, .sampled = sampled, .entropy = entropy };
    }
}

/// Run the sampler pass over `[c_len, vocab]` logits, parallelized over
/// positions. `u` = pre-drawn uniforms (one per position, drawn
/// single-threaded by the caller so results are thread-count independent).
pub fn samplerPass(
    ctx: *ExecContext,
    logits: *const fucina.Tensor(.{ .seq, .vocab }),
    temp: f32,
    u: []const f32,
    options: SamplerOptions,
) !SamplerPass {
    const allocator = ctx.allocator;
    const c_len = logits.dim(.seq);
    const vocab = logits.dim(.vocab);
    if (u.len != c_len) return Error.CanvasLengthMismatch;
    const data = try logits.dataConst();

    const results = try allocator.alloc(RowResult, c_len);
    errdefer allocator.free(results);

    const sc_cap = if (options.collect_sc) options.sc_max_per_row else 0;
    const sc_ids = try allocator.alloc(usize, c_len * sc_cap);
    errdefer allocator.free(sc_ids);
    const sc_probs = try allocator.alloc(f32, c_len * sc_cap);
    errdefer allocator.free(sc_probs);
    const sc_counts = try allocator.alloc(usize, c_len);
    defer allocator.free(sc_counts);
    @memset(sc_counts, 0);

    const base = RowTask{
        .logits = data,
        .vocab = vocab,
        .temp_inv = 1.0 / temp,
        .u = u,
        .results = results,
        .sc_p_min = options.sc_p_min,
        .sc_cap = sc_cap,
        .sc_ids = sc_ids,
        .sc_probs = sc_probs,
        .sc_counts = sc_counts,
        .row_start = 0,
        .row_end = c_len,
    };

    if (ctx.workPool()) |pool| {
        var tasks: [64]RowTask = undefined;
        const task_count = @min(@min(tasks.len, fucina.parallel.cpuThreadCount(fucina.parallel.vector_max_threads)), c_len);
        for (0..task_count) |t| {
            tasks[t] = base;
            tasks[t].row_start = t * c_len / task_count;
            tasks[t].row_end = (t + 1) * c_len / task_count;
        }
        pool.parallelChunks(RowTask, tasks[0..task_count], runRowTask);
    } else {
        runRowTask(&base);
    }

    var sc: ?ScSignal = null;
    if (options.collect_sc) {
        // Compact the fixed-cap per-row segments into the flat ScSignal.
        var total: usize = 0;
        for (sc_counts) |n| total += n;
        const ids = try allocator.alloc(usize, total);
        errdefer allocator.free(ids);
        const probs = try allocator.alloc(f32, total);
        errdefer allocator.free(probs);
        const offsets = try allocator.alloc(usize, c_len + 1);
        errdefer allocator.free(offsets);
        var w: usize = 0;
        for (0..c_len) |c| {
            offsets[c] = w;
            const n = sc_counts[c];
            @memcpy(ids[w..][0..n], sc_ids[c * sc_cap ..][0..n]);
            @memcpy(probs[w..][0..n], sc_probs[c * sc_cap ..][0..n]);
            w += n;
        }
        offsets[c_len] = w;
        sc = .{ .ids = ids, .probs = probs, .row_offsets = offsets, .allocator = allocator };
    }
    allocator.free(sc_ids);
    allocator.free(sc_probs);

    return .{ .results = results, .sc = sc };
}

/// Entropy-bound acceptance: sort positions by entropy ascending; accept a
/// position while the cumulative entropy of the strictly-lower-entropy
/// accepted set stays within the bound (the joint-MI upper bound of
/// arXiv:2505.24857; llama.cpp: `cumE - H[pos] <= entropy_bound`).
/// `accepted` is a caller-owned [c_len] bool buffer; `order` a [c_len]
/// scratch. Returns the number of accepted positions.
pub fn entropyBoundAccept(
    results: []const RowResult,
    entropy_bound: f32,
    order: []usize,
    accepted: []bool,
) usize {
    const c_len = results.len;
    std.debug.assert(order.len == c_len and accepted.len == c_len);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, results, struct {
        fn lessThan(ctx_results: []const RowResult, a: usize, b: usize) bool {
            return ctx_results[a].entropy < ctx_results[b].entropy;
        }
    }.lessThan);
    @memset(accepted, false);
    var cum: f64 = 0;
    var n_accepted: usize = 0;
    for (order) |pos| {
        const h = results[pos].entropy;
        cum += h;
        if (cum - h <= entropy_bound) {
            accepted[pos] = true;
            n_accepted += 1;
        }
    }
    return n_accepted;
}

/// Counter-based uniform f32 in [0, 1) from the repo RNG (rng.zig contract).
fn uniformAt(seed: u64, counter: *u64) f32 {
    const bits = rng.at(seed, counter.*);
    counter.* += 1;
    // Top 24 bits -> [0, 1) with full f32 mantissa coverage.
    return @as(f32, @floatFromInt(bits >> 40)) * (1.0 / @as(f32, 1 << 24));
}

fn randomTokenAt(seed: u64, counter: *u64, vocab: usize) usize {
    const bits = rng.at(seed, counter.*);
    counter.* += 1;
    // Gemma's vocab (262144) is a power of two, so the modulo is exact; for
    // other sizes the bias at 2^64 scale is negligible for renoising.
    return @intCast(bits % @as(u64, vocab));
}

// ---------------------------------------------------------------------------
// Denoising loop (one canvas) + block-autoregressive generation
// ---------------------------------------------------------------------------

/// Per-step snapshot handed to the streaming callback (the terminal
/// visualization renders these — see examples/diffusion_gemma/main.zig --visual).
pub const StepInfo = struct {
    /// 0-based step within this canvas; restarts at 0 on every block.
    step_index: usize,
    total_steps: usize,
    /// The per-position argmax (what becomes the output canvas when the
    /// denoising stops on this step).
    argmax_canvas: []const usize,
    /// This step's entropy-bound acceptance per position (accepted = the
    /// working canvas keeps the sampled token; rejected = renoised).
    accepted: []const bool,
    n_accepted: usize,
    mean_entropy: f32,
};

pub const DenoiseOptions = struct {
    eb: EbParams,
    seed: u64 = 0,
    self_conditioning: bool = true,
    sampler: SamplerOptions = .{},
    /// Streaming callback: called after every denoising step with the current
    /// argmax canvas + acceptance snapshot.
    on_step: ?*const fn (user: ?*anyopaque, info: *const StepInfo) void = null,
    on_step_user: ?*anyopaque = null,
};

pub const DenoiseResult = struct {
    /// Denoising steps actually run (adaptive stop counts).
    steps: usize,
};

/// Denoise one canvas in place: `canvas` (length C) holds the working token
/// ids and finishes as the OUTPUT canvas = the last step's per-position
/// argmax (the reference's output rule). The KV cache must hold exactly the
/// encoded prefix; it is left unchanged (kv.len identical on return).
pub fn denoiseCanvas(
    model: *const Model,
    ctx: *ExecContext,
    kv: *KvCache,
    canvas: []usize,
    options: DenoiseOptions,
) !DenoiseResult {
    const allocator = ctx.allocator;
    const c_len = canvas.len;
    const vocab = model.config.base.vocab_size;
    const eb = options.eb;
    if (c_len == 0) return Error.InvalidSequenceLength;
    if (kv.len + c_len > kv.capacity) return Error.KvCapacityTooSmall;

    var counter: u64 = 0;
    const seed = options.seed;

    // Canvas init: uniform-random token ids (NOT mask tokens — the EB
    // reference scheme; llama.cpp diffusion_generate_entropy_bound).
    for (canvas) |*t| t.* = randomTokenAt(seed, &counter, vocab);

    const u = try allocator.alloc(f32, c_len);
    defer allocator.free(u);
    const renoise = try allocator.alloc(usize, c_len);
    defer allocator.free(renoise);
    const order = try allocator.alloc(usize, c_len);
    defer allocator.free(order);
    const accepted = try allocator.alloc(bool, c_len);
    defer allocator.free(accepted);
    const argmax_canvas = try allocator.alloc(usize, c_len);
    defer allocator.free(argmax_canvas);
    const prev_argmax = try allocator.alloc(usize, c_len);
    defer allocator.free(prev_argmax);
    @memset(prev_argmax, std.math.maxInt(usize));

    var sc_signal: ?ScSignal = null;
    defer if (sc_signal) |*s| s.deinit();

    const use_sc = options.self_conditioning and model.sc != null;

    var held: usize = 0;
    var steps_run: usize = 0;
    var cur_step: usize = eb.max_steps;
    while (cur_step >= 1) : (cur_step -= 1) {
        const step_index = eb.max_steps - cur_step;
        const temp = eb.t_min + (eb.t_max - eb.t_min) * (@as(f32, @floatFromInt(cur_step)) / @as(f32, @floatFromInt(eb.max_steps)));

        // Pre-draw this step's uniforms and renoise tokens single-threaded
        // (deterministic at any thread count — the repo RNG contract).
        for (u) |*x| x.* = uniformAt(seed, &counter);
        for (renoise) |*t| t.* = randomTokenAt(seed, &counter, vocab);

        var logits = try model.canvasForward(ctx, kv, canvas, if (sc_signal) |*s| s else null);
        defer logits.deinit();

        var pass = try samplerPass(ctx, &logits, temp, u, .{
            .sc_p_min = options.sampler.sc_p_min,
            .sc_max_per_row = options.sampler.sc_max_per_row,
            .collect_sc = use_sc,
        });
        defer pass.deinit(allocator);
        steps_run += 1;

        const n_accepted = entropyBoundAccept(pass.results, eb.entropy_bound, order, accepted);

        var entropy_sum: f64 = 0;
        for (0..c_len) |c| {
            argmax_canvas[c] = pass.results[c].argmax;
            canvas[c] = if (accepted[c]) pass.results[c].sampled else renoise[c];
            entropy_sum += pass.results[c].entropy;
        }
        const mean_entropy: f32 = @floatCast(entropy_sum / @as(f64, @floatFromInt(c_len)));

        if (options.on_step) |cb| {
            const info = StepInfo{
                .step_index = step_index,
                .total_steps = eb.max_steps,
                .argmax_canvas = argmax_canvas,
                .accepted = accepted,
                .n_accepted = n_accepted,
                .mean_entropy = mean_entropy,
            };
            cb(options.on_step_user, &info);
        }

        // Hand this step's SC signal to the next step.
        if (use_sc) {
            if (sc_signal) |*s| s.deinit();
            sc_signal = pass.sc;
            pass.sc = null;
        }

        // Stable & confident adaptive stop.
        const stable = std.mem.eql(usize, prev_argmax, argmax_canvas);
        held = if (stable) held + 1 else 0;
        @memcpy(prev_argmax, argmax_canvas);
        if (held >= eb.stability_threshold and mean_entropy < eb.confidence_threshold) break;
    }

    @memcpy(canvas, argmax_canvas);
    return .{ .steps = steps_run };
}

pub const GenerateOptions = struct {
    denoise: DenoiseOptions,
    max_new_tokens: usize,
    /// Default end-of-generation ids from the model's generation_config:
    /// <eos>, <turn|>, <|tool_response>.
    eog_token_ids: []const usize = &.{ 1, 106, 50 },
    /// Called after each block is finalized (denoised + EOG/repetition
    /// trimmed) with the tokens kept from it. `finished` marks the last
    /// block of the generation. Drives the inline chat visualization.
    on_block: ?*const fn (user: ?*anyopaque, block_index: usize, kept_tokens: []const usize, finished: bool) void = null,
    on_block_user: ?*anyopaque = null,
};

pub const GenerateResult = struct {
    /// Tokens produced (written to out_tokens[0..produced]).
    produced: usize,
    /// Total denoising steps across all canvases.
    steps: usize,
    /// Canvases (blocks) denoised.
    blocks: usize,
};

/// Block-autoregressive generation: encode the prompt once, then per block
/// denoise a canvas, trim it at the first EOG token (or at a degenerate
/// repetition-loop onset — llama.cpp's heuristic), append the kept tokens to
/// the output, encoder-pass them into the KV cache, and continue until
/// `max_new_tokens` or an EOG-terminated block.
pub fn generate(
    model: *const Model,
    ctx: *ExecContext,
    kv: *KvCache,
    prompt_tokens: []const usize,
    out_tokens: []usize,
    options: GenerateOptions,
) !GenerateResult {
    if (prompt_tokens.len == 0) return Error.InvalidSequenceLength;
    const allocator = ctx.allocator;
    const c_len = model.config.canvas_length;
    const limit = @min(options.max_new_tokens, out_tokens.len);

    kv.reset();
    try model.encodeStep(ctx, kv, prompt_tokens, 0);

    const canvas = try allocator.alloc(usize, c_len);
    defer allocator.free(canvas);

    var produced: usize = 0;
    var steps_total: usize = 0;
    var blocks: usize = 0;
    var block_seed = options.denoise.seed;
    while (produced < limit) {
        if (kv.len + c_len > kv.capacity) return Error.KvCapacityTooSmall;
        var denoise_options = options.denoise;
        denoise_options.seed = block_seed;
        block_seed +%= 1; // llama.cpp visual server: per-block seed = seed + block index
        const result = try denoiseCanvas(model, ctx, kv, canvas, denoise_options);
        steps_total += result.steps;
        blocks += 1;

        // Trim: cut at the first EOG, else at a repetition-loop onset.
        var cut: usize = c_len;
        var finished = false;
        outer: for (canvas, 0..) |t, i| {
            for (options.eog_token_ids) |eog| {
                if (t == eog) {
                    cut = i;
                    finished = true;
                    break :outer;
                }
            }
        }
        if (!finished) {
            if (repetitionLoopOnset(canvas)) |onset| {
                cut = onset;
                finished = true;
            }
        }

        const keep = @min(cut, limit - produced);
        @memcpy(out_tokens[produced..][0..keep], canvas[0..keep]);
        produced += keep;

        const done = finished or produced >= limit;
        if (options.on_block) |cb| cb(options.on_block_user, blocks - 1, out_tokens[produced - keep .. produced], done);
        if (done) break;

        // Commit the full canvas to the prefix (the reference appends the
        // whole denoised block when continuing).
        try model.encodeStep(ctx, kv, canvas, kv.len);
    }
    return .{ .produced = produced, .steps = steps_total, .blocks = blocks };
}

/// llama.cpp's degenerate-loop trim: a token recurring at stride 1-2 for >= 6
/// consecutive steps marks a runaway repetition; returns the onset index.
fn repetitionLoopOnset(canvas: []const usize) ?usize {
    for ([_]usize{ 1, 2 }) |stride| {
        if (canvas.len < stride + 6) continue;
        var reps: usize = 0;
        for (0..canvas.len - stride) |j| {
            if (canvas[j] == canvas[j + stride]) {
                reps += 1;
                if (reps >= 6) return j + stride - reps;
            } else {
                reps = 0;
            }
        }
    }
    return null;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test {
    _ = @import("model_tests.zig");
}

test "entropy-bound acceptance matches the reference rule" {
    // Entropies 0.02, 0.05, 0.6, 0.01 with bound 0.1: ascending order is
    // [3 (0.01), 0 (0.02), 1 (0.05), 2 (0.6)]; cumE-h = 0, 0.01, 0.03, 0.08
    // -> all four satisfy <= 0.1 EXCEPT the last? cum before 2 is 0.08 <= 0.1
    // so even the high-entropy one is accepted once the others are tiny.
    var results = [_]RowResult{
        .{ .argmax = 0, .sampled = 0, .entropy = 0.02 },
        .{ .argmax = 0, .sampled = 0, .entropy = 0.05 },
        .{ .argmax = 0, .sampled = 0, .entropy = 0.6 },
        .{ .argmax = 0, .sampled = 0, .entropy = 0.01 },
    };
    var order: [4]usize = undefined;
    var accepted: [4]bool = undefined;
    const n = entropyBoundAccept(&results, 0.1, &order, &accepted);
    try std.testing.expectEqual(@as(usize, 4), n);
    try std.testing.expect(accepted[0] and accepted[1] and accepted[2] and accepted[3]);

    // Tighter bound 0.04: order [3,0,1,2]; cumE-h: 3->0 ok, 0->0.01 ok,
    // 1->0.03 ok, 2->0.08 reject.
    const n2 = entropyBoundAccept(&results, 0.04, &order, &accepted);
    try std.testing.expectEqual(@as(usize, 3), n2);
    try std.testing.expect(accepted[0] and accepted[1] and !accepted[2] and accepted[3]);
}

test "repetition-loop onset heuristic" {
    // No loop.
    try std.testing.expectEqual(@as(?usize, null), repetitionLoopOnset(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }));
    // Stride-1 run of 7 identical tokens starting at index 3.
    const loop1 = [_]usize{ 1, 2, 3, 9, 9, 9, 9, 9, 9, 9 };
    try std.testing.expect(repetitionLoopOnset(&loop1) != null);
    // Stride-2 alternation.
    const loop2 = [_]usize{ 5, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1 };
    try std.testing.expect(repetitionLoopOnset(&loop2) != null);
}

test "uniform draws are deterministic and in range" {
    var counter: u64 = 0;
    var counter2: u64 = 0;
    for (0..1000) |_| {
        const a = uniformAt(42, &counter);
        const b = uniformAt(42, &counter2);
        try std.testing.expectEqual(a, b);
        try std.testing.expect(a >= 0 and a < 1);
    }
    var c3: u64 = 0;
    for (0..1000) |_| {
        const t = randomTokenAt(7, &c3, 262144);
        try std.testing.expect(t < 262144);
    }
}
