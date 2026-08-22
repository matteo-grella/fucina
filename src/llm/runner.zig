//! The descriptor runner (Level 0 of the universal checkpoint runner,
//! docs/RUNNER.md): one family-independent decoder implementation driven by
//! a runtime `Descriptor` — dims, attention/MLP/norm variants, and the GGUF
//! metadata prefix — instead of per-family forward code. `fromGguf` is the
//! seed metadata compiler: it recognizes any architecture following the
//! qwen3-shaped GGUF key layout (dense or MoE) and emits the descriptor.
//!
//! The op vocabulary is the same facade the hand ports use (fused
//! norm+projection, half-rope QK-norm, grouped causal attention over the
//! shared KvCache, dense/MoE FFN with the packed fast paths), so the
//! runner inherits every kernel and the buffer-pool memory discipline.
//! The qwen3 family runs directly on this module (`qwen3/model.zig` is an
//! alias surface over it), which is why the `.fused` style also carries
//! the batched decode entries (`forwardStepBatch`, `forwardStepBatchSpans`)
//! and the nullable SubQ research seam threaded through `attentionBlock`.
//! Correctness is pinned by `runner_tests.zig`'s recorded-logits gates
//! (real Qwen3-0.6B GGUFs + synthetic dense/MoE/glm fixtures) and, for
//! `.host_reference`, bitwise parity against the hand glm4moe port.
//!
//! Stability: experimental (CHANGELOG.md tiers).

const std = @import("std");
const fucina = @import("fucina");
const weights = @import("fucina").weights;
const kv_cache = @import("kv_cache.zig");
const subq_mod = @import("subq.zig");
const gguf_meta = @import("fucina").gguf_meta;
const ptqtp_gguf = @import("fucina").ptqtp_gguf;

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;
const gguf = fucina.gguf;
const LinearWeight = weights.LinearWeight;
const WeightF32 = weights.WeightF32;
const WeightF16 = weights.WeightF16;
const WeightQ4_K = weights.WeightQ4_K;
const WeightQ5_K = weights.WeightQ5_K;
const WeightQ6_K = weights.WeightQ6_K;
const WeightQ8_0 = weights.WeightQ8_0;

pub const Error = weights.Error || error{
    InvalidConfig,
    InvalidSequenceLength,
    /// Batched entries require distinct sibling caches: one per stream,
    /// all the same dtype (all from this model's `initKvCache`).
    MismatchedKvCaches,
    KvCacheOverflow,
};

pub const ForwardProfile = struct {
    attn_prep_ns: i128 = 0,
    qkv_ns: i128 = 0,
    qk_norm_rope_ns: i128 = 0,
    attention_ns: i128 = 0,
    attn_out_ns: i128 = 0,
    attn_residual_ns: i128 = 0,
    ffn_prep_ns: i128 = 0,
    router_ns: i128 = 0,
    gate_up_ns: i128 = 0,
    swiglu_ns: i128 = 0,
    down_ns: i128 = 0,
    moe_batch: fucina.MoeBatchProfile = .{},
    ffn_residual_ns: i128 = 0,
    final_ns: i128 = 0,
    layers: usize = 0,
};

/// Which block implementation executes the descriptor. Structural, not a
/// family name: `.fused` is the fused-kernel decoder shape (QK-norm +
/// half-rope over the full head dim, unbiased QKV, softmax top-k MoE — the
/// qwen3/qwen3moe vocabulary); `.host_reference` is the auditable host-side
/// f32 shape (biased QKV, PARTIAL rope with selectable pairing, sigmoid
/// noaux MoE with router bias + shared experts + leading dense layers —
/// the GLM-4.5 / DeepSeek-MoE vocabulary, ported verbatim from the hand
/// glm4moe block so parity holds bitwise). Heavy linears and the fused MoE
/// mixture run on fucina kernels in both styles.
pub const BlockStyle = enum { fused, host_reference };

pub const RopePairing = enum { half, interleaved };

pub const Descriptor = struct {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    // Mixture-of-Experts (qwen3moe). `num_experts == 0` means a dense model and
    // the FFN follows the standard gate/up/down path.
    num_experts: usize = 0,
    num_experts_used: usize = 0,
    moe_intermediate_size: usize = 0,
    norm_topk_prob: bool = true,
    /// Adaptive expert top-p (`applyExpertTopP`): keep experts per token up
    /// to this cumulative routing weight. 1.0 (default) = full top-k,
    /// bit-identical baseline. A runtime knob, not GGUF metadata.
    moe_expert_top_p: f32 = 1.0,

    // --- block-style variants (see `BlockStyle`); the defaults describe
    // the fused qwen3 shape, `fromGguf` fills the host_reference shape.
    block_style: BlockStyle = .fused,
    /// Rotated dims per head (0 = the full head_dim; host_reference only).
    rope_dims: usize = 0,
    rope_pairing: RopePairing = .half,
    /// QKV projections carry additive biases (host_reference only).
    qkv_bias: bool = false,
    /// First N layers use the dense FFN even when `num_experts > 0`.
    leading_dense_layers: usize = 0,
    /// Always-on shared experts beside the routed mixture (their fused FFN
    /// width is `num_shared_experts * moe_intermediate_size`).
    num_shared_experts: usize = 0,
    /// Routed-mixture weight scale applied after (optional) renormalization.
    expert_weights_scale: f32 = 1.0,
    /// Router activation: 1 = softmax, 2 = sigmoid (the noaux shape).
    expert_gating_func: usize = 1,
    /// Renormalize the selected experts' routing weights to sum to 1.
    expert_weights_norm: bool = false,

    pub fn isMoe(self: Descriptor) bool {
        return self.num_experts > 0;
    }

    pub fn qwen3_0_6b() Descriptor {
        return .{
            .vocab_size = 151_936,
            .hidden_size = 1024,
            .intermediate_size = 3072,
            .num_layers = 28,
            .num_attention_heads = 16,
            .num_key_value_heads = 8,
            .head_dim = 128,
            .rms_norm_eps = 1e-6,
            .rope_theta = 1_000_000,
        };
    }

    /// Derive the config from GGUF metadata so any Qwen3-family size
    /// (0.6B/1.7B/4B/8B) loads without hardcoding. Keys are read under the
    /// `general.architecture` prefix (e.g. `qwen3.block_count`) — the standard
    /// GGUF naming convention — so this stays model-size and arch agnostic.
    pub fn fromGguf(file: *const gguf.File) !Descriptor {
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        const embd = try file.get("token_embd.weight");
        const shape = try embd.logicalMatrixShape(); // {vocab, hidden}

        // The GLM/DeepSeek-MoE metadata shape maps onto the host_reference
        // block style (partial interleaved rope, QKV biases, noaux MoE).
        if (std.mem.eql(u8, arch, "glm4moe")) {
            const block_count = try metaInt(file, arch, "block_count");
            const nextn = metaIntOpt(file, arch, "nextn_predict_layers") orelse 0;
            return .{
                .block_style = .host_reference,
                .vocab_size = shape[0],
                .hidden_size = try metaInt(file, arch, "embedding_length"),
                .intermediate_size = try metaInt(file, arch, "feed_forward_length"),
                .num_layers = block_count - nextn, // the runner runs the trunk; nextn/MTP stays with the hand port
                .num_attention_heads = try metaInt(file, arch, "attention.head_count"),
                .num_key_value_heads = try metaInt(file, arch, "attention.head_count_kv"),
                .head_dim = try metaInt(file, arch, "attention.key_length"),
                .rms_norm_eps = try metaFloat(file, arch, "attention.layer_norm_rms_epsilon"),
                .rope_theta = gguf_meta.metaFloatOpt(file, arch, "rope.freq_base") orelse 10_000.0,
                .rope_dims = try metaInt(file, arch, "rope.dimension_count"),
                .rope_pairing = .interleaved,
                .qkv_bias = true,
                .num_experts = metaIntOpt(file, arch, "expert_count") orelse 0,
                .num_experts_used = metaIntOpt(file, arch, "expert_used_count") orelse 0,
                .moe_intermediate_size = metaIntOpt(file, arch, "expert_feed_forward_length") orelse 0,
                .leading_dense_layers = metaIntOpt(file, arch, "leading_dense_block_count") orelse 0,
                .num_shared_experts = metaIntOpt(file, arch, "expert_shared_count") orelse 0,
                .expert_weights_scale = gguf_meta.metaFloatOpt(file, arch, "expert_weights_scale") orelse 1.0,
                .expert_gating_func = metaIntOpt(file, arch, "expert_gating_func") orelse 1,
                .expert_weights_norm = file.getBool("glm4moe.expert_weights_norm") orelse false,
            };
        }

        // MoE models (qwen3moe) declare experts and replace the dense FFN; a
        // dense model has no expert_count key (num_experts stays 0).
        const num_experts = metaIntOpt(file, arch, "expert_count") orelse 0;
        const is_moe = num_experts > 0;

        return .{
            .vocab_size = shape[0],
            .hidden_size = try metaInt(file, arch, "embedding_length"),
            // MoE GGUFs may omit/zero feed_forward_length; experts size the FFN.
            .intermediate_size = if (is_moe)
                (metaIntOpt(file, arch, "feed_forward_length") orelse 0)
            else
                try metaInt(file, arch, "feed_forward_length"),
            .num_layers = try metaInt(file, arch, "block_count"),
            .num_attention_heads = try metaInt(file, arch, "attention.head_count"),
            .num_key_value_heads = try metaInt(file, arch, "attention.head_count_kv"),
            .head_dim = try metaInt(file, arch, "attention.key_length"),
            .rms_norm_eps = try metaFloat(file, arch, "attention.layer_norm_rms_epsilon"),
            .rope_theta = try metaFloat(file, arch, "rope.freq_base"),
            .num_experts = num_experts,
            .num_experts_used = if (is_moe) try metaInt(file, arch, "expert_used_count") else 0,
            .moe_intermediate_size = if (is_moe) try metaInt(file, arch, "expert_feed_forward_length") else 0,
            .norm_topk_prob = true,
        };
    }

    fn validate(self: Descriptor) !void {
        if (self.num_attention_heads == 0 or self.num_key_value_heads == 0) return Error.InvalidConfig;
        if (self.num_attention_heads % self.num_key_value_heads != 0) return Error.InvalidConfig;
        if (self.head_dim % 2 != 0) return Error.InvalidConfig;
        if (self.isMoe()) {
            if (self.num_experts_used == 0 or self.num_experts_used > self.num_experts) return Error.InvalidConfig;
            if (self.moe_intermediate_size == 0) return Error.InvalidConfig;
        }
    }

    pub fn qProjectionDim(self: Descriptor) usize {
        return self.num_attention_heads * self.head_dim;
    }

    pub fn kvProjectionDim(self: Descriptor) usize {
        return self.num_key_value_heads * self.head_dim;
    }
};

// Every qwen3 config int is structurally positive, so a present-but-zero key
// is rejected like a missing one (`.reject_zero`).
fn metaInt(file: *const gguf.File, arch: []const u8, suffix: []const u8) !usize {
    return gguf_meta.metaInt(file, arch, suffix, .reject_zero);
}

const metaFloat = gguf_meta.metaFloat;

fn metaIntOpt(file: *const gguf.File, arch: []const u8, suffix: []const u8) ?usize {
    return gguf_meta.metaIntOpt(file, arch, suffix, .reject_zero);
}

/// Opt-in disk streaming for the MoE expert stacks (shared across the MoE
/// loaders — see `weights.MoeStreamOptions`).
pub const MoeStreamOptions = weights.MoeStreamOptions;

pub const LoadOptions = struct {
    moe_stream: ?MoeStreamOptions = null,
    /// host_reference only: the host rope table's position capacity
    /// (`hostStep` rejects positions beyond it).
    max_positions: usize = 4096,
};

pub const Model = struct {
    allocator: Allocator,
    config: Descriptor,
    token_embedding: LinearWeight,
    output_norm: fucina.Tensor(.{.embed}),
    output: LinearWeight,
    layers: []Layer,
    kv_head_for_head: []usize,
    /// The GGUF mmap, owned by the model when MoE expert blocks borrow from it
    /// (see loadMoeFfn); unmapped last in deinit.
    weight_mapping: ?gguf.File.MappedRegion = null,
    /// Disk-streaming tier for MoE experts (`MoeStreamOptions`); destroyed
    /// after the layers whose streamed arms point into it.
    expert_store: ?*fucina.ExpertStore = null,
    /// Router-lookahead prefetch (`MoeStreamOptions.pilot`).
    pilot_enabled: bool = false,
    /// The host_reference band (`BlockStyle.host_reference`): host-side
    /// layer state + rope table; null for `.fused` models.
    host: ?HostBand = null,

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, config: Descriptor) !Model {
        return loadGgufOptions(ctx, io, path, config, .{});
    }

    pub fn loadGgufOptions(ctx: *ExecContext, io: std.Io, path: []const u8, config: Descriptor, options: LoadOptions) !Model {
        // mmap, matching the CLI (examples/qwen3/main.zig) and the other loaders:
        // avoids an eager multi-GB heap read that coexists with the
        // materialized weights, and lets MoE experts borrow straight from the
        // mapping (loadGgufFromFile takes ownership of it via takeMapping).
        var file = try gguf.File.loadMmap(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFileOptions(ctx, &file, config, options);
    }

    /// Load weights from an already-parsed GGUF file. Lets the caller build a
    /// tokenizer from the same `file` (which carries the tokenizer metadata)
    /// without reading the model file twice.
    pub fn loadGgufFromFile(ctx: *ExecContext, file: *gguf.File, config: Descriptor) !Model {
        return loadGgufFromFileOptions(ctx, file, config, .{});
    }

    pub fn loadGgufFromFileOptions(ctx: *ExecContext, file: *gguf.File, config: Descriptor, options: LoadOptions) !Model {
        try config.validate();
        if (config.block_style == .host_reference) return loadHostReference(ctx, file, config, options);

        const allocator = ctx.allocator;

        var expert_store: ?*fucina.ExpertStore = null;
        if (options.moe_stream) |stream_options| {
            if (config.isMoe()) expert_store = try weights.createExpertStore(allocator, stream_options, config.num_layers);
        }
        errdefer if (expert_store) |store| store.destroy();

        var token_embedding = try LinearWeight.load(ctx, try file.get("token_embd.weight"), config.vocab_size, config.hidden_size);
        errdefer token_embedding.deinit();

        var output_norm = try weights.loadVector(ctx, try file.get("output_norm.weight"), config.hidden_size, .embed);
        errdefer output_norm.deinit();

        var output = blk: {
            // Persisted PTQTP head planes win over the base tensor; a
            // decorated head on a tied-embedding model has planes and no
            // base, while the embedding keeps its source precision.
            if (try ptqtp_gguf.maybeLoadPlanes(ctx, file, "output.weight", config.vocab_size, config.hidden_size)) |planes| break :blk planes;
            if (file.maybeGet("output.weight")) |info| break :blk try LinearWeight.load(ctx, info, config.vocab_size, config.hidden_size);
            break :blk try token_embedding.cloneView(ctx);
        };
        errdefer output.deinit();

        const kv_head_for_head = try allocator.alloc(usize, config.num_attention_heads);
        errdefer allocator.free(kv_head_for_head);
        const heads_per_kv = config.num_attention_heads / config.num_key_value_heads;
        for (kv_head_for_head, 0..) |*kv_head, head_i| kv_head.* = head_i / heads_per_kv;

        const layers = try allocator.alloc(Layer, config.num_layers);
        errdefer allocator.free(layers);
        try loadLayers(ctx, file, config, layers, expert_store);
        errdefer for (layers) |*layer| layer.deinit();

        if (expert_store) |store| try store.finalize();

        // MoE expert blocks borrow from the mmap (loadMoeFfn), so the model
        // takes ownership of the mapping; dense models keep nothing mapped.
        // Streamed MoE never touches the expert pages through the mapping, so
        // it keeps nothing mapped either: the caller's `file.deinit` munmaps
        // and resident memory stays dense weights + the expert cache.
        const weight_mapping = if (config.num_experts > 0 and expert_store == null) file.takeMapping() else null;

        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = token_embedding,
            .output_norm = output_norm,
            .output = output,
            .layers = layers,
            .kv_head_for_head = kv_head_for_head,
            .weight_mapping = weight_mapping,
            .expert_store = expert_store,
            .pilot_enabled = expert_store != null and options.moe_stream.?.pilot,
        };
    }

    pub fn deinit(self: *Model) void {
        if (self.host) |*band| band.deinit(self.allocator);
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.allocator.free(self.kv_head_for_head);
        self.output.deinit();
        self.output_norm.deinit();
        self.token_embedding.deinit();
        // Last: expert blocks borrowed from this mapping / streamed arms
        // pointing into the store.
        if (self.expert_store) |store| store.destroy();
        if (self.weight_mapping) |*mapping| mapping.deinit();
        self.* = undefined;
    }

    pub fn forwardLastLogits(self: *const Model, ctx: *ExecContext, token_ids: []const usize) !fucina.Tensor(.{ .seq, .vocab }) {
        return self.forwardLastLogitsImpl(ctx, null, token_ids, null);
    }

    pub fn forwardLastLogitsProfiled(self: *const Model, ctx: *ExecContext, io: std.Io, token_ids: []const usize, profile: *ForwardProfile) !fucina.Tensor(.{ .seq, .vocab }) {
        return self.forwardLastLogitsImpl(ctx, io, token_ids, profile);
    }

    fn forwardLastLogitsImpl(self: *const Model, ctx: *ExecContext, io: ?std.Io, token_ids: []const usize, profile: ?*ForwardProfile) !fucina.Tensor(.{ .seq, .vocab }) {
        if (token_ids.len == 0) return Error.InvalidSequenceLength;

        var rope_table = try ctx.prepareRopeTableRange(.{ .len = token_ids.len }, self.config.head_dim, self.config.rope_theta, false);
        defer rope_table.deinit();

        var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
        errdefer x.deinit();

        const cfg = self.config;
        for (self.layers, 0..) |*layer, layer_i| {
            const last_query_only = layer_i + 1 == cfg.num_layers and token_ids.len > 1;
            x = try ctx.replace(x, attentionBlock(ctx, io, cfg, layer, &x, &rope_table, self.kv_head_for_head, last_query_only, profile, null, layer_i, null));

            x = try ctx.replace(x, ffnBlock(ctx, io, cfg, layer, &x, profile));
            if (profile) |p| p.layers += 1;
        }

        const final_start = profileStart(profile, io);
        var final_norm = try x.rmsNormMul(ctx, .embed, &self.output_norm, self.config.rms_norm_eps);
        defer final_norm.deinit();
        x.deinit();

        var last = try final_norm.narrow(ctx, .seq, final_norm.dim(.seq) - 1, 1);
        defer last.deinit();

        const logits = try self.output.linearSeq(ctx, &last, .embed, .vocab);
        if (profile) |p| p.final_ns += profileElapsed(final_start, io);
        return logits;
    }

    /// A KV cache with this model's (uniform) attention geometry — the
    /// duck-typed construction seam generic embedders (chat.Conversation)
    /// use; gemma4's per-layer-geometry counterpart is initPerLayer-backed.
    pub fn initKvCache(self: *const Model, ctx: *ExecContext, capacity: usize) !KvCache {
        return KvCache.init(ctx, self.config.num_layers, self.config.num_key_value_heads, self.config.head_dim, capacity);
    }

    /// Process `token_ids` at absolute positions `pos0 .. pos0 + len`, appending
    /// their post-RoPE K/V into `kv`, and return the last token's logits.
    /// Attention runs the new queries against the whole cache (`kv.len + len`
    /// positions). With a fresh cache and `pos0 == 0` this is prefill and yields
    /// the same last-token logits as `forwardLastLogits`; with one token it is a
    /// single decode step.
    pub fn forwardStep(
        self: *const Model,
        ctx: *ExecContext,
        kv: *KvCache,
        token_ids: []const usize,
        pos0: usize,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        return self.forwardStepImpl(ctx, null, kv, token_ids, pos0, null, true, null);
    }

    pub fn forwardStepProfiled(
        self: *const Model,
        ctx: *ExecContext,
        io: std.Io,
        kv: *KvCache,
        token_ids: []const usize,
        pos0: usize,
        profile: *ForwardProfile,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        return self.forwardStepImpl(ctx, io, kv, token_ids, pos0, profile, true, null);
    }

    /// As `forwardStep`, but returns logits for EVERY appended position —
    /// `[token_ids.len, vocab]`, row `i` = the next-token distribution after
    /// `token_ids[0..i+1]` (given the cached prefix). KV semantics are
    /// identical to `forwardStep`: all rows are appended and `kv` advances by
    /// `token_ids.len`. This is the speculative-decoding verify entry: one
    /// batched pass scores all draft positions, so the caller pays ~one step's
    /// weight traffic instead of `token_ids.len` sequential steps. The per-row
    /// numerics match per-token `forwardStep` calls (same kernels, row-wise
    /// independent) as long as the batch stays below the m-dependent kernel
    /// thresholds (quantized-weight x4-packed kernels at seq >= 4, fused FFN
    /// at seq >= 12, tiled attention at seq >= 48); beyond them rows can
    /// differ by reassociation drift (~1e-6 rel).
    pub fn forwardStepAllLogits(
        self: *const Model,
        ctx: *ExecContext,
        kv: *KvCache,
        token_ids: []const usize,
        pos0: usize,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        return self.forwardStepImpl(ctx, null, kv, token_ids, pos0, null, false, null);
    }

    /// `forwardStep` with the SubQ research attention evaluator active on
    /// every layer's decode attention (f16 KV caches only; the evaluator
    /// falls back to dense until its warmup floor is reached).
    pub fn forwardStepSubq(
        self: *const Model,
        ctx: *ExecContext,
        kv: *KvCache,
        token_ids: []const usize,
        pos0: usize,
        sq: *subq_mod.State,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        return self.forwardStepImpl(ctx, null, kv, token_ids, pos0, null, true, sq);
    }

    fn forwardStepImpl(
        self: *const Model,
        ctx: *ExecContext,
        io: ?std.Io,
        kv: *KvCache,
        token_ids: []const usize,
        pos0: usize,
        profile: ?*ForwardProfile,
        last_only: bool,
        subq_state: ?*subq_mod.State,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        if (token_ids.len == 0) return Error.InvalidSequenceLength;
        if (kv.len != pos0) return Error.InvalidSequenceLength;
        if (kv.len + token_ids.len > kv.capacity) return kv_cache.Error.KvCacheOverflow;

        var rope_table = try ctx.prepareRopeTableRange(.{ .origin = @intCast(pos0), .len = token_ids.len }, self.config.head_dim, self.config.rope_theta, false);
        defer rope_table.deinit();

        var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
        errdefer x.deinit();

        const cfg = self.config;
        for (self.layers, 0..) |*layer, layer_i| {
            const last_query_only = last_only and layer_i + 1 == cfg.num_layers and token_ids.len > 1;
            x = try ctx.replace(x, attentionBlock(ctx, io, cfg, layer, &x, &rope_table, self.kv_head_for_head, last_query_only, profile, kv, layer_i, subq_state));
            // Router lookahead (pilot): predict the NEXT layer's experts from
            // this layer's post-attention state and start their disk
            // readahead in the background while this layer's FFN computes.
            // Decode-sized batches only — prefill's batch-union reads each
            // routed expert once regardless.
            if (self.pilot_enabled and token_ids.len <= 4 and layer_i + 1 < self.layers.len) {
                pilotPrefetchNext(ctx, cfg, &self.layers[layer_i + 1], layer_i + 1, &x) catch {};
            }
            x = try ctx.replace(x, ffnBlock(ctx, io, cfg, layer, &x, profile));
            if (profile) |p| p.layers += 1;
        }
        kv.advance(token_ids.len);

        const final_start = profileStart(profile, io);
        var final_norm = try x.rmsNormMul(ctx, .embed, &self.output_norm, self.config.rms_norm_eps);
        defer final_norm.deinit();
        x.deinit();

        // last_only keeps just the final row for the vocab projection; the
        // all-logits entry projects every position.
        const keep_from = if (last_only) final_norm.dim(.seq) - 1 else 0;
        var head_in = try final_norm.narrow(ctx, .seq, keep_from, final_norm.dim(.seq) - keep_from);
        defer head_in.deinit();

        const logits = try self.output.linearSeq(ctx, &head_in, .embed, .vocab);
        if (profile) |p| p.final_ns += profileElapsed(final_start, io);
        return logits;
    }

    /// Batched multi-sequence decode: one NEW token per stream, each stream
    /// backed by its own `KvCache` (distinct sibling caches from this
    /// model's `initKvCache`, all the same dtype). Row `s` of the returned
    /// `[n_streams, vocab]` logits is stream `s`'s next-token distribution,
    /// and each cache advances by one. The dense trunk (QKV/O-proj, FFN or
    /// MoE mixture, lm_head) runs as ONE m=n pass — weights are read once
    /// for all streams, the batch-decode bandwidth win — while RoPE
    /// positions, KV appends, and attention are per-stream (ragged, each
    /// row against its own cache at its own position). Per-row numerics
    /// match per-stream `forwardStep` under the same conditions as
    /// `forwardStepAllLogits`: bit-identical below the m-dependent kernel
    /// thresholds — for QUANTIZED weights the x4-packed kernels engage at
    /// n >= 4 (measured: 0.6B Q4_K/Q8_0 batch == sequential token-for-token
    /// at n <= 3, ~1e-6 reassociation drift at n >= 4); f32/f16 weights
    /// stay bitwise until the fused-FFN threshold at n >= 12.
    pub fn forwardStepBatch(
        self: *const Model,
        ctx: *ExecContext,
        caches: []const *KvCache,
        token_ids: []const usize,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        const n = token_ids.len;
        if (n == 0 or caches.len != n) return Error.InvalidSequenceLength;
        const dtype = caches[0].dtype;
        for (caches, 0..) |kv, i| {
            if (kv.dtype != dtype) return Error.MismatchedKvCaches;
            // A cache built for another model's layer stack would index its
            // per-layer slices out of bounds inside the layer loop.
            if (kv.head_dim.len != self.layers.len) return Error.MismatchedKvCaches;
            if (kv.len + 1 > kv.capacity) return kv_cache.Error.KvCacheOverflow;
            for (caches[0..i]) |prev| if (prev == kv) return Error.MismatchedKvCaches;
        }

        const a = ctx.allocator;
        const positions = try a.alloc(i32, n);
        defer a.free(positions);
        for (positions, caches) |*position, kv| position.* = @intCast(kv.len);

        var rope_table = try ctx.prepareRopeTable(positions, self.config.head_dim, self.config.rope_theta, false);
        defer rope_table.deinit();

        // Per-stream attention spans, refilled per layer; the lens are
        // constant across layers (appendLayer never advances the caches —
        // they advance once, below, after the layer loop).
        var spans = try BatchKvSpans.init(a, dtype, n);
        defer spans.deinit(a);
        for (spans.lens, caches) |*len, kv| len.* = kv.len + 1;

        var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
        // Released manually once final_norm is built; the flag keeps the
        // lm_head projection's error path from re-releasing it.
        var x_released = false;
        errdefer if (!x_released) x.deinit();

        const cfg = self.config;
        for (self.layers, 0..) |*layer, layer_i| {
            x = try ctx.replace(x, attentionBlockBatch(ctx, cfg, layer, &x, &rope_table, self.kv_head_for_head, caches, layer_i, &spans));
            x = try ctx.replace(x, ffnBlock(ctx, null, cfg, layer, &x, null));
        }
        for (caches) |kv| kv.advance(1);

        var final_norm = try x.rmsNormMul(ctx, .embed, &self.output_norm, self.config.rms_norm_eps);
        defer final_norm.deinit();
        x.deinit();
        x_released = true;

        return self.output.linearSeq(ctx, &final_norm, .embed, .vocab);
    }

    /// `forwardStepBatch` generalized to a SPAN of tokens per stream — the
    /// ragged batch a multi-stream speculative verify needs: stream `i`
    /// contributes `span_lens[i]` consecutive tokens of `token_ids` (its
    /// carried token plus drafted continuations). Every seq-parallel op runs
    /// packed over the concatenated rows (the batching win), and attention
    /// runs per stream against its own cache with the standard kernels
    /// (q_seq = span, end-aligned causal — per-stream attention IS the
    /// ragged-batch mask, the same decomposition as the trainer's packed
    /// segments). Appends each stream's rows to its cache and advances it
    /// by its span — callers rewind rejected drafts with `truncate`.
    /// Returns logits for EVERY row, in input order. With all spans == 1
    /// this computes exactly `forwardStepBatch`.
    pub fn forwardStepBatchSpans(
        self: *const Model,
        ctx: *ExecContext,
        caches: []const *KvCache,
        token_ids: []const usize,
        span_lens: []const usize,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        const n = caches.len;
        if (n == 0 or span_lens.len != n) return Error.InvalidSequenceLength;
        var total: usize = 0;
        for (span_lens) |span| {
            if (span == 0) return Error.InvalidSequenceLength;
            total += span;
        }
        if (total != token_ids.len) return Error.InvalidSequenceLength;
        const dtype = caches[0].dtype;
        for (caches, span_lens, 0..) |kv, span, i| {
            if (kv.dtype != dtype) return Error.MismatchedKvCaches;
            if (kv.head_dim.len != self.layers.len) return Error.MismatchedKvCaches;
            if (kv.len + span > kv.capacity) return kv_cache.Error.KvCacheOverflow;
            for (caches[0..i]) |prev| if (prev == kv) return Error.MismatchedKvCaches;
        }

        const a = ctx.allocator;
        const positions = try a.alloc(i32, total);
        defer a.free(positions);
        {
            var at: usize = 0;
            for (caches, span_lens) |kv, span| {
                for (0..span) |j| {
                    positions[at] = @intCast(kv.len + j);
                    at += 1;
                }
            }
        }
        var rope_table = try ctx.prepareRopeTable(positions, self.config.head_dim, self.config.rope_theta, false);
        defer rope_table.deinit();

        var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
        var x_released = false;
        errdefer if (!x_released) x.deinit();

        const cfg = self.config;
        for (self.layers, 0..) |*layer, layer_i| {
            x = try ctx.replace(x, attentionBlockBatchSpans(ctx, cfg, layer, &x, &rope_table, self.kv_head_for_head, caches, layer_i, span_lens));
            x = try ctx.replace(x, ffnBlock(ctx, null, cfg, layer, &x, null));
        }
        for (caches, span_lens) |kv, span| kv.advance(span);

        var final_norm = try x.rmsNormMul(ctx, .embed, &self.output_norm, self.config.rms_norm_eps);
        defer final_norm.deinit();
        x.deinit();
        x_released = true;

        return self.output.linearSeq(ctx, &final_norm, .embed, .vocab);
    }

    // ---- host_reference band (see `BlockStyle`) --------------------------

    fn loadHostReference(ctx: *ExecContext, file: *gguf.File, config: Descriptor, options: LoadOptions) !Model {
        const allocator = ctx.allocator;

        var expert_store: ?*fucina.ExpertStore = null;
        if (options.moe_stream) |stream_options| {
            if (config.isMoe()) expert_store = try weights.createExpertStore(allocator, stream_options, config.num_layers);
        }
        errdefer if (expert_store) |store| store.destroy();

        var token_embedding = try LinearWeight.load(ctx, try file.get("token_embd.weight"), config.vocab_size, config.hidden_size);
        errdefer token_embedding.deinit();
        var output_norm = try weights.loadVector(ctx, try file.get("output_norm.weight"), config.hidden_size, .embed);
        errdefer output_norm.deinit();
        var output = try LinearWeight.load(ctx, try file.get("output.weight"), config.vocab_size, config.hidden_size);
        errdefer output.deinit();

        const kv_head_for_head = try allocator.alloc(usize, config.num_attention_heads);
        errdefer allocator.free(kv_head_for_head);
        const heads_per_kv = config.num_attention_heads / config.num_key_value_heads;
        for (kv_head_for_head, 0..) |*kv_head, head_i| kv_head.* = head_i / heads_per_kv;

        const host_layers = try allocator.alloc(HostLayer, config.num_layers);
        errdefer allocator.free(host_layers);
        var built: usize = 0;
        errdefer for (host_layers[0..built]) |*l| l.deinit(allocator);
        for (host_layers, 0..) |*layer, i| {
            layer.* = try loadHostLayer(ctx, file, config, i, expert_store);
            built += 1;
        }

        if (expert_store) |store| try store.finalize();
        const weight_mapping = if (config.isMoe() and expert_store == null) file.takeMapping() else null;
        if (config.isMoe() and expert_store == null and weight_mapping == null) return Error.InvalidWeightShape;

        const host_output_norm = try weights.hostVector(allocator, file, "output_norm.weight", config.hidden_size);
        errdefer allocator.free(host_output_norm);
        var rope = try HostRope.init(allocator, config, options.max_positions);
        errdefer rope.deinit(allocator);

        const layers = try allocator.alloc(Layer, 0);
        errdefer allocator.free(layers);

        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = token_embedding,
            .output_norm = output_norm,
            .output = output,
            .layers = layers,
            .kv_head_for_head = kv_head_for_head,
            .weight_mapping = weight_mapping,
            .expert_store = expert_store,
            .host = .{
                .layers = host_layers,
                .rope = rope,
                .output_norm = host_output_norm,
                .attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.head_dim))),
            },
        };
    }

    pub fn initHostCache(self: *const Model, capacity: usize) !HostCache {
        if (self.host == null) return Error.InvalidConfig;
        return HostCache.init(self.allocator, self.config.num_layers, self.config.num_key_value_heads, self.config.head_dim, capacity);
    }

    /// host_reference forward: process `tokens` at positions
    /// [cache.len, cache.len + S) and return per-position next-token logits
    /// `[S, vocab]`. Positions are computed causally in sequence, so
    /// per-row numerics match S = 1 steps exactly (the hand glm4moe port's
    /// `step` contract, whose block this band mirrors verbatim).
    pub fn hostStep(self: *const Model, ctx: *ExecContext, cache: *HostCache, tokens: []const usize) !fucina.Tensor(.{ .seq, .vocab }) {
        const band = if (self.host) |*b| b else return Error.InvalidConfig;
        const cfg = self.config;
        const allocator = ctx.allocator;
        if (tokens.len == 0) return Error.InvalidSequenceLength;
        if (cache.len + tokens.len > cache.capacity or cache.len + tokens.len > band.rope.capacity) return Error.KvCacheOverflow;

        const S = tokens.len;
        const x = try allocator.alloc(f32, S * cfg.hidden_size);
        defer allocator.free(x);
        {
            var emb = try self.token_embedding.getRowsAs(ctx, tokens, .embed);
            defer emb.deinit();
            @memcpy(x, try emb.dataConst());
        }

        for (band.layers, 0..) |*layer, layer_i| {
            try hostLayerForward(ctx, cfg, band, cache, layer, layer_i, x, S, cache.len);
        }
        cache.len += S;

        var normed_t = try fucina.Tensor(.{ .seq, .embed }).empty(ctx, .{ S, cfg.hidden_size });
        defer normed_t.deinit();
        const normed = try normed_t.data();
        for (0..S) |r| {
            hostRmsNormInto(normed[r * cfg.hidden_size ..][0..cfg.hidden_size], x[r * cfg.hidden_size ..][0..cfg.hidden_size], band.output_norm, cfg.rms_norm_eps);
        }
        return self.output.linearSeq(ctx, &normed_t, .embed, .vocab);
    }

};

/// Dense-FFN weights (exported for train.zig's differentiable forward).
pub const DenseFfn = struct {
    input_proj: FfnInputProjection,
    down_proj: LinearWeight,

    fn deinit(self: *DenseFfn) void {
        self.down_proj.deinit();
        self.input_proj.deinit();
        self.* = undefined;
    }
};

const MoeFfn = struct {
    router: LinearWeight,
    gate: fucina.MoeRhs,
    up: fucina.MoeRhs,
    down: fucina.MoeRhs,

    fn deinit(self: *MoeFfn) void {
        self.down.deinit();
        self.up.deinit();
        self.gate.deinit();
        self.router.deinit();
        self.* = undefined;
    }
};

const Ffn = union(enum) {
    dense: DenseFfn,
    moe: MoeFfn,

    fn deinit(self: *Ffn) void {
        switch (self.*) {
            .dense => |*dense| dense.deinit(),
            .moe => |*moe| moe.deinit(),
        }
        self.* = undefined;
    }
};

/// Load one projection linear by GGUF name: persisted PTQTP planes when the
/// file carries them (ptqtp_gguf pair-detection; a no-op metadata lookup on
/// undecorated files), else the base tensor. Skip-layer tensors inside a
/// decorated file take the base branch — plane presence is per tensor.
fn loadProjection(ctx: *ExecContext, file: *const gguf.File, name: []const u8, rows: usize, cols: usize, for_fusion: bool) !LinearWeight {
    if (try ptqtp_gguf.maybeLoadPlanes(ctx, file, name, rows, cols)) |planes| return planes;
    const info = try file.get(name);
    return if (for_fusion)
        LinearWeight.loadForFusion(ctx, info, rows, cols)
    else
        LinearWeight.load(ctx, info, rows, cols);
}

/// MoE expert-stack counterpart of `loadProjection`: persisted PTQTP plane
/// stacks (`<name>.ptqtpK` siblings) when the file carries them, else the
/// base stacked tensor.
fn loadMoeProjection(ctx: *ExecContext, file: *const gguf.File, name: []const u8, in_dim: usize, out_dim: usize, n_expert: usize, borrow: bool) !fucina.MoeRhs {
    if (try ptqtp_gguf.maybeLoadMoeRhs(ctx, file, name, in_dim, out_dim, n_expert, borrow)) |rhs| return rhs;
    return weights.loadMoeRhs(ctx, try file.get(name), in_dim, out_dim, n_expert, borrow);
}

/// Streamed counterpart: an ExpertStore ProjSpec, pair-detecting PTQTP
/// plane sets the same way.
fn moeProjSpec(file: *const gguf.File, name: []const u8, in_dim: usize, out_dim: usize, n_expert: usize) !fucina.expert_store.ProjSpec {
    if (try ptqtp_gguf.maybeStreamedMoeProjSpec(file, name, in_dim, out_dim, n_expert)) |spec| return spec;
    return weights.streamedProjSpec(file, try file.get(name), in_dim, out_dim, n_expert);
}

fn loadDenseFfn(ctx: *ExecContext, file: *const gguf.File, config: Descriptor, layer_i: usize) !DenseFfn {
    var name_buf: [96]u8 = undefined;

    var input_proj = try FfnInputProjection.load(ctx, file, config, layer_i);
    errdefer input_proj.deinit();

    var down_proj = try loadProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_down.weight"), config.hidden_size, config.intermediate_size, false);
    errdefer down_proj.deinit();

    return .{ .input_proj = input_proj, .down_proj = down_proj };
}

fn loadMoeFfn(ctx: *ExecContext, file: *const gguf.File, config: Descriptor, layer_i: usize, store: ?*fucina.ExpertStore) !MoeFfn {
    var name_buf: [96]u8 = undefined;

    var router = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_gate_inp.weight")), config.num_experts, config.hidden_size);
    errdefer router.deinit();

    // Streamed: register geometry only — the expert stacks stay on disk and
    // never become resident. addLayer touches only this layer's state, so
    // the parallel layer loader may call it concurrently for distinct layers.
    if (store) |s| {
        const trio = try weights.registerStreamedMoeLayer(s, layer_i, .{
            try moeProjSpec(file, try weights.layerName(&name_buf, layer_i, "ffn_gate_exps.weight"), config.hidden_size, config.moe_intermediate_size, config.num_experts),
            try moeProjSpec(file, try weights.layerName(&name_buf, layer_i, "ffn_up_exps.weight"), config.hidden_size, config.moe_intermediate_size, config.num_experts),
            // down transposes the FFN: (out_pe -> hidden).
            try moeProjSpec(file, try weights.layerName(&name_buf, layer_i, "ffn_down_exps.weight"), config.moe_intermediate_size, config.hidden_size, config.num_experts),
        }, config.num_experts);
        return .{ .router = router, .gate = trio.gate, .up = trio.up, .down = trio.down };
    }

    // Expert blocks need no repack, so when the GGUF is mmap'd they are
    // borrowed straight from the mapping (the Model takes ownership of it in
    // loadGgufFromFile) instead of copying the multi-GB stacks. Split GGUFs
    // cannot hand over their multiple mappings (takeMapping declines), so
    // their experts are copied — stream them instead for the big models.
    const borrow = file.is_mmap and !file.isSplit();

    var gate = try loadMoeProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_gate_exps.weight"), config.hidden_size, config.moe_intermediate_size, config.num_experts, borrow);
    errdefer gate.deinit();
    var up = try loadMoeProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_up_exps.weight"), config.hidden_size, config.moe_intermediate_size, config.num_experts, borrow);
    errdefer up.deinit();
    var down = try loadMoeProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_down_exps.weight"), config.moe_intermediate_size, config.hidden_size, config.num_experts, borrow);
    errdefer down.deinit();

    return .{ .router = router, .gate = gate, .up = up, .down = down };
}

/// One transformer layer's weights. Exported: train.zig runs its
/// differentiable forward over the same layer structs (`layers` field), and
/// tests build synthetic layers directly.
pub const Layer = struct {
    attn_norm: fucina.Tensor(.{.embed}),
    q_norm: fucina.Tensor(.{.d}),
    k_norm: fucina.Tensor(.{.d}),
    ffn_norm: fucina.Tensor(.{.embed}),
    attn_proj: AttentionProjection,
    o_proj: LinearWeight,
    ffn: Ffn,

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Descriptor, layer_i: usize, store: ?*fucina.ExpertStore) !Layer {
        var name_buf: [96]u8 = undefined;

        var attn_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "attn_norm.weight")), config.hidden_size, .embed);
        errdefer attn_norm.deinit();

        var q_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "attn_q_norm.weight")), config.head_dim, .d);
        errdefer q_norm.deinit();

        var k_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "attn_k_norm.weight")), config.head_dim, .d);
        errdefer k_norm.deinit();

        var ffn_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_norm.weight")), config.hidden_size, .embed);
        errdefer ffn_norm.deinit();

        var attn_proj = try AttentionProjection.load(ctx, file, config, layer_i);
        errdefer attn_proj.deinit();

        var o_proj = try loadProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "attn_output.weight"), config.hidden_size, config.qProjectionDim(), false);
        errdefer o_proj.deinit();

        var ffn: Ffn = if (config.isMoe())
            .{ .moe = try loadMoeFfn(ctx, file, config, layer_i, store) }
        else
            .{ .dense = try loadDenseFfn(ctx, file, config, layer_i) };
        errdefer ffn.deinit();

        return .{
            .attn_norm = attn_norm,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .ffn_norm = ffn_norm,
            .attn_proj = attn_proj,
            .o_proj = o_proj,
            .ffn = ffn,
        };
    }

    pub fn deinit(self: *Layer) void {
        self.ffn.deinit();
        self.o_proj.deinit();
        self.attn_proj.deinit();
        self.ffn_norm.deinit();
        self.k_norm.deinit();
        self.q_norm.deinit();
        self.attn_norm.deinit();
        self.* = undefined;
    }
};

/// Per-family adapter for `gguf_meta.parallelLoadLayers`.
const LayerLoader = struct {
    ctx: *ExecContext,
    file: *const gguf.File,
    config: Descriptor,
    store: ?*fucina.ExpertStore,

    pub fn load(self: LayerLoader, layer_i: usize) !Layer {
        return Layer.load(self.ctx, self.file, self.config, layer_i, self.store);
    }

    pub fn deinitLayer(_: LayerLoader, layer: *Layer) void {
        layer.deinit();
    }
};

/// Load all transformer layers, in parallel across the work pool when
/// available (see `gguf_meta.parallelLoadLayers` for the failure semantics).
fn loadLayers(ctx: *ExecContext, file: *const gguf.File, config: Descriptor, layers: []Layer, store: ?*fucina.ExpertStore) !void {
    return gguf_meta.parallelLoadLayers(Layer, LayerLoader, ctx, .{ .ctx = ctx, .file = file, .config = config, .store = store }, layers);
}

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

pub const QkvProjection = struct {
    q: fucina.Tensor(.{ .seq, .q }),
    k: fucina.Tensor(.{ .seq, .k }),
    v: fucina.Tensor(.{ .seq, .v }),

    pub fn deinit(self: *QkvProjection) void {
        self.v.deinit();
        self.k.deinit();
        self.q.deinit();
        self.* = undefined;
    }
};

const AttentionProjection = union(enum) {
    separate: SeparateAttentionProjection,
    fused: LinearWeight,

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Descriptor, layer_i: usize) !AttentionProjection {
        var name_buf: [96]u8 = undefined;

        var q_proj = try loadProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "attn_q.weight"), config.qProjectionDim(), config.hidden_size, true);
        errdefer q_proj.deinit();

        var k_proj = try loadProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "attn_k.weight"), config.kvProjectionDim(), config.hidden_size, true);
        errdefer k_proj.deinit();

        var v_proj = try loadProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "attn_v.weight"), config.kvProjectionDim(), config.hidden_size, true);
        errdefer v_proj.deinit();

        var fuse_parts = [_]*LinearWeight{ &q_proj, &k_proj, &v_proj };
        if (try weights.fuseLinear(ctx, &fuse_parts)) |fused| {
            return .{ .fused = fused };
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

    /// `project` over `rmsNormMul(input, norm_weight, eps)`: when EVERY
    /// projection weight routes through the fused normalize+quantize+packed
    /// GEMM (see LinearWeight.supportsNormedFusion), the normalized [m, k]
    /// tensor is never materialized — f32-roundoff-identical to the unfused pair.
    /// Otherwise one rmsNormMul + the plain `project` run (never a per-
    /// projection re-normalize).
    fn projectNormed(
        self: *const AttentionProjection,
        ctx: *ExecContext,
        input: *const fucina.Tensor(.{ .seq, .embed }),
        norm_weight: *const fucina.Tensor(.{.embed}),
        eps: f32,
        config: Descriptor,
    ) !QkvProjection {
        const m = input.dim(.seq);
        const all_fused = switch (self.*) {
            .separate => |*separate| separate.q_proj.supportsNormedFusion(m) and
                separate.k_proj.supportsNormedFusion(m) and
                separate.v_proj.supportsNormedFusion(m),
            .fused => |*weight| weight.supportsNormedFusion(m),
        };
        if (!all_fused) {
            var normed = try input.rmsNormMul(ctx, .embed, norm_weight, eps);
            defer normed.deinit();
            return self.project(ctx, &normed, config);
        }
        return switch (self.*) {
            // The separate arm re-derives the row norms inside each fused
            // kernel (three cheap rms reductions) instead of materializing
            // and re-reading the normalized tensor three times.
            .separate => |*separate| blk: {
                var q = try separate.q_proj.linearSeqNormed(ctx, input, norm_weight, eps, .embed, .q);
                errdefer q.deinit();
                var k = try separate.k_proj.linearSeqNormed(ctx, input, norm_weight, eps, .embed, .k);
                errdefer k.deinit();
                var v = try separate.v_proj.linearSeqNormed(ctx, input, norm_weight, eps, .embed, .v);
                errdefer v.deinit();
                break :blk .{ .q = q, .k = k, .v = v };
            },
            .fused => |*weight| blk: {
                var qkv = try weight.linearSeqNormed(ctx, input, norm_weight, eps, .embed, .qkv);
                defer qkv.deinit();
                break :blk try splitQkv(ctx, &qkv, config);
            },
        };
    }

    fn project(self: *const AttentionProjection, ctx: *ExecContext, input: *const fucina.Tensor(.{ .seq, .embed }), config: Descriptor) !QkvProjection {
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

/// Zero-copy narrows of a fused-QKV projection result into per-projection
/// slices. Exported: train.zig's differentiable forward splits its fused-QKV
/// dot through this exact function, so the split stays op-for-op one place.
pub fn splitQkv(ctx: *ExecContext, qkv: *const fucina.Tensor(.{ .seq, .qkv }), config: Descriptor) !QkvProjection {
    var q_view = try qkv.narrow(ctx, .qkv, 0, config.qProjectionDim());
    defer q_view.deinit();
    var q = try q_view.withTags(ctx, .{ .seq, .q });
    errdefer q.deinit();

    var k_view = try qkv.narrow(ctx, .qkv, config.qProjectionDim(), config.kvProjectionDim());
    defer k_view.deinit();
    var k = try k_view.withTags(ctx, .{ .seq, .k });
    errdefer k.deinit();

    var v_view = try qkv.narrow(ctx, .qkv, config.qProjectionDim() + config.kvProjectionDim(), config.kvProjectionDim());
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

pub const GateUpProjection = struct {
    gate: fucina.Tensor(.{ .seq, .ffn }),
    up: fucina.Tensor(.{ .seq, .ffn }),

    pub fn deinit(self: *GateUpProjection) void {
        self.up.deinit();
        self.gate.deinit();
        self.* = undefined;
    }
};

const FfnInputProjection = union(enum) {
    separate: SeparateFfnInputProjection,
    fused: LinearWeight,

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Descriptor, layer_i: usize) !FfnInputProjection {
        var name_buf: [96]u8 = undefined;

        var gate_proj = try loadProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_gate.weight"), config.intermediate_size, config.hidden_size, true);
        errdefer gate_proj.deinit();

        var up_proj = try loadProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_up.weight"), config.intermediate_size, config.hidden_size, true);
        errdefer up_proj.deinit();

        var fuse_parts = [_]*LinearWeight{ &gate_proj, &up_proj };
        if (try weights.fuseLinear(ctx, &fuse_parts)) |fused| {
            return .{ .fused = fused };
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

    fn project(self: *const FfnInputProjection, ctx: *ExecContext, input: *const fucina.Tensor(.{ .seq, .embed }), config: Descriptor) !GateUpProjection {
        return switch (self.*) {
            .separate => |*separate| blk: {
                var gate = try separate.gate_proj.linearSeq(ctx, input, .embed, .ffn);
                errdefer gate.deinit();
                var up = try separate.up_proj.linearSeq(ctx, input, .embed, .ffn);
                errdefer up.deinit();
                break :blk .{ .gate = gate, .up = up };
            },
            .fused => |*weight| blk: {
                var gate_up = try weight.linearSeq(ctx, input, .embed, .gate_up);
                defer gate_up.deinit();
                break :blk try splitGateUp(ctx, &gate_up, config);
            },
        };
    }
};

/// Fused gate/up counterpart of `splitQkv` (also shared with train.zig).
pub fn splitGateUp(ctx: *ExecContext, gate_up: *const fucina.Tensor(.{ .seq, .gate_up }), config: Descriptor) !GateUpProjection {
    var gate_view = try gate_up.narrow(ctx, .gate_up, 0, config.intermediate_size);
    defer gate_view.deinit();
    var gate = try gate_view.withTags(ctx, .{ .seq, .ffn });
    errdefer gate.deinit();

    var up_view = try gate_up.narrow(ctx, .gate_up, config.intermediate_size, config.intermediate_size);
    defer up_view.deinit();
    var up = try up_view.withTags(ctx, .{ .seq, .ffn });
    errdefer up.deinit();

    return .{ .gate = gate, .up = up };
}

fn attentionBlock(
    ctx: *ExecContext,
    io: ?std.Io,
    config: Descriptor,
    layer: *const Layer,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    rope_table: *const fucina.RopeTable,
    kv_head_for_head: []const usize,
    last_query_only: bool,
    profile: ?*ForwardProfile,
    cache: ?*KvCache,
    layer_i: usize,
    subq_state: ?*subq_mod.State,
) !fucina.Tensor(.{ .seq, .embed }) {
    const prep_start = profileStart(profile, io);
    var qkv_linear = try layer.attn_proj.projectNormed(ctx, input, &layer.attn_norm, config.rms_norm_eps, config);
    defer qkv_linear.deinit();
    if (profile) |p| p.attn_prep_ns += profileElapsed(prep_start, io);

    const qkv_start = profileStart(profile, io);

    var q3 = try qkv_linear.q.split(ctx, .q, .{ .head, .d }, .{ config.num_attention_heads, config.head_dim });
    defer q3.deinit();
    var k3 = try qkv_linear.k.split(ctx, .k, .{ .kv_head, .d }, .{ config.num_key_value_heads, config.head_dim });
    defer k3.deinit();
    var v3 = try qkv_linear.v.split(ctx, .v, .{ .kv_head, .d }, .{ config.num_key_value_heads, config.head_dim });
    defer v3.deinit();
    if (profile) |p| p.qkv_ns += profileElapsed(qkv_start, io);

    const qk_norm_rope_start = profileStart(profile, io);
    var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, config.rms_norm_eps, rope_table);
    defer q_rope.deinit();
    var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm, config.rms_norm_eps, rope_table);
    defer k_rope.deinit();
    if (profile) |p| p.qk_norm_rope_ns += profileElapsed(qk_norm_rope_start, io);

    const attention_start = profileStart(profile, io);
    var q_last: ?fucina.Tensor(.{ .seq, .head, .d }) = null;
    defer if (q_last) |*value| value.deinit();
    if (last_query_only) {
        q_last = try q_rope.narrow(ctx, .seq, q_rope.dim(.seq) - 1, 1);
    }
    const q_attention = if (q_last) |*value| value else &q_rope;
    var attn = if (cache) |kv| blk: {
        try kv.appendLayer(ctx, layer_i, &k_rope, &v3);
        const cached_len = kv.len + k_rope.dim(.seq);
        switch (kv.dtype) {
            .f16 => {
                if (subq_state) |sq| {
                    if (q_attention.dim(.seq) == 1) {
                        break :blk try subqAttention(ctx, config, sq, layer_i, q_attention, kv, cached_len);
                    }
                }
                var k_view = try kv.k[layer_i].narrow(ctx, .seq, 0, cached_len);
                defer k_view.deinit();
                var v_view = try kv.v[layer_i].narrow(ctx, .seq, 0, cached_len);
                defer v_view.deinit();
                break :blk try causalAttention(ctx, config, q_attention, &k_view, &v_view, kv_head_for_head, .{});
            },
            .q8_0 => break :blk try causalAttention(
                ctx,
                config,
                q_attention,
                kv.kBlocks(layer_i, cached_len),
                kv.vBlocks(layer_i, cached_len),
                kv_head_for_head,
                .{ .kv_seq = cached_len, .kv_heads = config.num_key_value_heads },
            ),
        }
    } else try causalAttention(ctx, config, q_attention, &k_rope, &v3, kv_head_for_head, .{});
    defer attn.deinit();
    if (profile) |p| p.attention_ns += profileElapsed(attention_start, io);

    const out_start = profileStart(profile, io);
    var attn_out = try layer.o_proj.linearSeq(ctx, &attn, .attn, .embed);
    defer attn_out.deinit();
    if (profile) |p| p.attn_out_ns += profileElapsed(out_start, io);

    const residual_start = profileStart(profile, io);
    var input_last: ?fucina.Tensor(.{ .seq, .embed }) = null;
    defer if (input_last) |*value| value.deinit();
    if (last_query_only) {
        input_last = try input.narrow(ctx, .seq, input.dim(.seq) - 1, 1);
    }
    const residual_input = if (input_last) |*value| value else input;
    const out = try residual_input.add(ctx, &attn_out);
    if (profile) |p| p.attn_residual_ns += profileElapsed(residual_start, io);
    return out;
}

fn subqAttention(
    ctx: *ExecContext,
    config: Descriptor,
    sq: *subq_mod.State,
    layer_i: usize,
    q: *const fucina.Tensor(.{ .seq, .head, .d }),
    kv: *KvCache,
    cached_len: usize,
) !fucina.Tensor(.{ .seq, .attn }) {
    const heads = config.num_attention_heads;
    const d = config.head_dim;
    // Borrow the contiguous query row when possible; the state's persistent
    // bridge buffers cover the fallback and the output (no per-token
    // allocations in this glue).
    const q_flat: []const f32 = blk: {
        if (q.dataConst()) |qd| {
            if (qd.len == heads * d) break :blk qd;
        } else |_| {}
        try q.copyTo(sq.bridge_q);
        break :blk sq.bridge_q;
    };
    const out = sq.bridge_out;
    const row_len = cached_len * config.num_key_value_heads * d;
    try sq.attend(
        ctx,
        layer_i,
        q_flat,
        (try kv.k[layer_i].dataConst())[0..row_len],
        (try kv.v[layer_i].dataConst())[0..row_len],
        cached_len,
        out,
    );
    return fucina.Tensor(.{ .seq, .attn }).fromSlice(ctx, .{ 1, heads * d }, out);
}

/// Per-stream KV attention spans for `forwardStepBatch`: allocated once per
/// step, the span arm matching the caches' dtype refilled per layer.
const BatchKvSpans = struct {
    lens: []usize,
    ks_f16: [][]const f16 = &.{},
    vs_f16: [][]const f16 = &.{},
    ks_q8: [][]const fucina.BlockQ8_0 = &.{},
    vs_q8: [][]const fucina.BlockQ8_0 = &.{},

    fn init(allocator: Allocator, dtype: kv_cache.KvDtype, n: usize) !BatchKvSpans {
        var spans = BatchKvSpans{ .lens = try allocator.alloc(usize, n) };
        errdefer allocator.free(spans.lens);
        switch (dtype) {
            .f16 => {
                spans.ks_f16 = try allocator.alloc([]const f16, n);
                errdefer allocator.free(spans.ks_f16);
                spans.vs_f16 = try allocator.alloc([]const f16, n);
            },
            .q8_0 => {
                spans.ks_q8 = try allocator.alloc([]const fucina.BlockQ8_0, n);
                errdefer allocator.free(spans.ks_q8);
                spans.vs_q8 = try allocator.alloc([]const fucina.BlockQ8_0, n);
            },
        }
        return spans;
    }

    fn deinit(self: *BatchKvSpans, allocator: Allocator) void {
        if (self.vs_q8.len > 0) allocator.free(self.vs_q8);
        if (self.ks_q8.len > 0) allocator.free(self.ks_q8);
        if (self.vs_f16.len > 0) allocator.free(self.vs_f16);
        if (self.ks_f16.len > 0) allocator.free(self.ks_f16);
        allocator.free(self.lens);
        self.* = undefined;
    }
};

/// The batch-decode sibling of `attentionBlock`: the norm/QKV/QK-norm/RoPE
/// trunk runs on all n stream rows at once (the rope table carries each
/// stream's own position), then KV append and attention go per-stream —
/// row `s` appends to and attends `caches[s]` only, via the ragged
/// multi-stream attention entry. Every row is its stream's last (and only)
/// query, so no `last_query_only` arm exists; no profile plumbing either.
fn attentionBlockBatch(
    ctx: *ExecContext,
    config: Descriptor,
    layer: *const Layer,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    rope_table: *const fucina.RopeTable,
    kv_head_for_head: []const usize,
    caches: []const *KvCache,
    layer_i: usize,
    spans: *BatchKvSpans,
) !fucina.Tensor(.{ .seq, .embed }) {
    var qkv_linear = try layer.attn_proj.projectNormed(ctx, input, &layer.attn_norm, config.rms_norm_eps, config);
    defer qkv_linear.deinit();

    var q3 = try qkv_linear.q.split(ctx, .q, .{ .head, .d }, .{ config.num_attention_heads, config.head_dim });
    defer q3.deinit();
    var k3 = try qkv_linear.k.split(ctx, .k, .{ .kv_head, .d }, .{ config.num_key_value_heads, config.head_dim });
    defer k3.deinit();
    var v3 = try qkv_linear.v.split(ctx, .v, .{ .kv_head, .d }, .{ config.num_key_value_heads, config.head_dim });
    defer v3.deinit();

    var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, config.rms_norm_eps, rope_table);
    defer q_rope.deinit();
    var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm, config.rms_norm_eps, rope_table);
    defer k_rope.deinit();

    for (caches, 0..) |kv, s| {
        var k_row = try k_rope.narrow(ctx, .seq, s, 1);
        defer k_row.deinit();
        var v_row = try v3.narrow(ctx, .seq, s, 1);
        defer v_row.deinit();
        try kv.appendLayer(ctx, layer_i, &k_row, &v_row);
    }

    const scale = 1 / @sqrt(@as(f32, @floatFromInt(config.head_dim)));
    var attn = switch (caches[0].dtype) {
        .f16 => blk: {
            for (caches, spans.ks_f16, spans.vs_f16, spans.lens) |kv, *k_span, *v_span, len| {
                k_span.* = try kv.kSlice(layer_i, len);
                v_span.* = try kv.vSlice(layer_i, len);
            }
            break :blk try q_rope.groupedAttention(ctx, spans.ks_f16, spans.vs_f16, kv_head_for_head, .attn, scale, .{ .lens = spans.lens, .kv_heads = config.num_key_value_heads });
        },
        .q8_0 => blk: {
            for (caches, spans.ks_q8, spans.vs_q8, spans.lens) |kv, *k_span, *v_span, len| {
                k_span.* = kv.kBlocks(layer_i, len);
                v_span.* = kv.vBlocks(layer_i, len);
            }
            break :blk try q_rope.groupedAttention(ctx, spans.ks_q8, spans.vs_q8, kv_head_for_head, .attn, scale, .{ .lens = spans.lens, .kv_heads = config.num_key_value_heads });
        },
    };
    defer attn.deinit();

    var attn_out = try layer.o_proj.linearSeq(ctx, &attn, .attn, .embed);
    defer attn_out.deinit();

    return input.add(ctx, &attn_out);
}

/// `attentionBlockBatch` for a SPAN of rows per stream (see
/// `forwardStepBatchSpans`): projections, norms, and RoPE run packed over
/// the concatenated spans; each stream's rows append to its cache and
/// attend against it with the standard single-stream kernels (q_seq =
/// span, kv_seq = cache prefix + span, end-aligned causal).
fn attentionBlockBatchSpans(
    ctx: *ExecContext,
    config: Descriptor,
    layer: *const Layer,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    rope_table: *const fucina.RopeTable,
    kv_head_for_head: []const usize,
    caches: []const *KvCache,
    layer_i: usize,
    span_lens: []const usize,
) !fucina.Tensor(.{ .seq, .embed }) {
    var qkv_linear = try layer.attn_proj.projectNormed(ctx, input, &layer.attn_norm, config.rms_norm_eps, config);
    defer qkv_linear.deinit();

    var q3 = try qkv_linear.q.split(ctx, .q, .{ .head, .d }, .{ config.num_attention_heads, config.head_dim });
    defer q3.deinit();
    var k3 = try qkv_linear.k.split(ctx, .k, .{ .kv_head, .d }, .{ config.num_key_value_heads, config.head_dim });
    defer k3.deinit();
    var v3 = try qkv_linear.v.split(ctx, .v, .{ .kv_head, .d }, .{ config.num_key_value_heads, config.head_dim });
    defer v3.deinit();

    var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, config.rms_norm_eps, rope_table);
    defer q_rope.deinit();
    var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm, config.rms_norm_eps, rope_table);
    defer k_rope.deinit();

    const Out = fucina.Tensor(.{ .seq, .attn });
    const outs = try ctx.allocator.alloc(Out, caches.len);
    defer ctx.allocator.free(outs);
    var built: usize = 0;
    errdefer for (outs[0..built]) |*out| out.deinit();

    var start: usize = 0;
    for (caches, span_lens, 0..) |kv, span, s| {
        var k_rows = try k_rope.narrow(ctx, .seq, start, span);
        defer k_rows.deinit();
        var v_rows = try v3.narrow(ctx, .seq, start, span);
        defer v_rows.deinit();
        try kv.appendLayer(ctx, layer_i, &k_rows, &v_rows);
        const cached_len = kv.len + span;

        var q_seg = try q_rope.narrow(ctx, .seq, start, span);
        defer q_seg.deinit();
        outs[s] = switch (kv.dtype) {
            .f16 => blk: {
                var k_view = try kv.k[layer_i].narrow(ctx, .seq, 0, cached_len);
                defer k_view.deinit();
                var v_view = try kv.v[layer_i].narrow(ctx, .seq, 0, cached_len);
                defer v_view.deinit();
                break :blk try causalAttention(ctx, config, &q_seg, &k_view, &v_view, kv_head_for_head, .{});
            },
            .q8_0 => try causalAttention(
                ctx,
                config,
                &q_seg,
                kv.kBlocks(layer_i, cached_len),
                kv.vBlocks(layer_i, cached_len),
                kv_head_for_head,
                .{ .kv_seq = cached_len, .kv_heads = config.num_key_value_heads },
            ),
        };
        built += 1;
        start += span;
    }

    var attn: Out = undefined;
    if (outs.len == 1) {
        attn = outs[0];
        built = 0; // ownership moved
    } else {
        const rest = try ctx.allocator.alloc(*const Out, outs.len - 1);
        defer ctx.allocator.free(rest);
        for (rest, outs[1..]) |*ptr, *out| ptr.* = out;
        attn = try outs[0].concat(ctx, .seq, rest);
        for (outs[0..built]) |*out| out.deinit();
        built = 0;
    }
    defer attn.deinit();

    var attn_out = try layer.o_proj.linearSeq(ctx, &attn, .attn, .embed);
    defer attn_out.deinit();

    return input.add(ctx, &attn_out);
}

fn ffnBlock(
    ctx: *ExecContext,
    io: ?std.Io,
    config: Descriptor,
    layer: *const Layer,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    profile: ?*ForwardProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    // Dense fused-input models whose gate_up projection routes packed: the
    // FFN norm fuses into the projection's LHS quantization (f32-roundoff
    // identical — see LinearWeight.linearSeqNormed) and the normalized
    // tensor is never materialized.
    switch (layer.ffn) {
        .dense => |*dense| if (dense.input_proj == .fused and dense.input_proj.fused.supportsNormedFusion(input.dim(.seq))) {
            var contribution = try denseFfnNormed(ctx, io, dense, input, &layer.ffn_norm, config.rms_norm_eps, profile);
            defer contribution.deinit();
            const residual_start = profileStart(profile, io);
            const out = try input.add(ctx, &contribution);
            if (profile) |p| p.ffn_residual_ns += profileElapsed(residual_start, io);
            return out;
        },
        .moe => {},
    }

    const prep_start = profileStart(profile, io);
    var ffn_in = try input.rmsNormMul(ctx, .embed, &layer.ffn_norm, config.rms_norm_eps);
    defer ffn_in.deinit();
    if (profile) |p| p.ffn_prep_ns += profileElapsed(prep_start, io);

    // The FFN contribution (pre-residual): dense gate/up/SwiGLU/down, or the
    // MoE mixture of top-k experts.
    var contribution = switch (layer.ffn) {
        .dense => |*dense| try denseFfn(ctx, io, config, dense, &ffn_in, profile),
        .moe => |*moe| try moeFfn(ctx, io, config, moe, &ffn_in, profile),
    };
    defer contribution.deinit();

    const residual_start = profileStart(profile, io);
    const out = try input.add(ctx, &contribution);
    if (profile) |p| p.ffn_residual_ns += profileElapsed(residual_start, io);
    return out;
}

fn denseFfn(
    ctx: *ExecContext,
    io: ?std.Io,
    config: Descriptor,
    dense: *const DenseFfn,
    ffn_in: *const fucina.Tensor(.{ .seq, .embed }),
    profile: ?*ForwardProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    // Multi-token fused fast path (qwen35's Q8_0 pattern, extended to the
    // K-quants): one gate_up GEMM, then split-SwiGLU + LHS quantization + the
    // packed down GEMM in a single pass — the gated m*ffn tensor is never
    // materialized. Q4_K is x8-packed only; MMLA targets fall through.
    if (ffn_in.dim(.seq) >= 12 and dense.input_proj == .fused) {
        const gate_up_weight = &dense.input_proj.fused;
        switch (dense.down_proj) {
            .q4_k => |*down| if (comptime !fucina.supports_q4_k_mmla) {
                return denseFfnFusedDown(ctx, io, gate_up_weight, &down.packed_rhs, ffn_in, profile);
            },
            .q5_k => |*down| return denseFfnFusedDown(ctx, io, gate_up_weight, &down.packed_rhs, ffn_in, profile),
            .q6_k => |*down| return denseFfnFusedDown(ctx, io, gate_up_weight, &down.packed_rhs, ffn_in, profile),
            .q8_0 => |*down| return denseFfnFusedDown(ctx, io, gate_up_weight, &down.packed_rhs, ffn_in, profile),
            else => {},
        }
    }

    const gate_up_start = profileStart(profile, io);
    var gated = switch (dense.input_proj) {
        .separate => blk: {
            var gate_up = try dense.input_proj.project(ctx, ffn_in, config);
            defer gate_up.deinit();
            if (profile) |p| p.gate_up_ns += profileElapsed(gate_up_start, io);

            const swiglu_start = profileStart(profile, io);
            const out = try gate_up.up.swiglu(ctx, &gate_up.gate);
            if (profile) |p| p.swiglu_ns += profileElapsed(swiglu_start, io);
            break :blk out;
        },
        .fused => |*weight| blk: {
            var gate_up = try weight.linearSeq(ctx, ffn_in, .embed, .gate_up);
            defer gate_up.deinit();
            if (profile) |p| p.gate_up_ns += profileElapsed(gate_up_start, io);

            const swiglu_start = profileStart(profile, io);
            const out = try gate_up.splitGated(ctx, .swiglu, .gate_up, .ffn);
            if (profile) |p| p.swiglu_ns += profileElapsed(swiglu_start, io);
            break :blk out;
        },
    };
    defer gated.deinit();

    const down_start = profileStart(profile, io);
    const down = try dense.down_proj.linearSeq(ctx, &gated, .ffn, .embed);
    if (profile) |p| p.down_ns += profileElapsed(down_start, io);
    return down;
}

fn denseFfnFusedDown(
    ctx: *ExecContext,
    io: ?std.Io,
    gate_up_weight: *const LinearWeight,
    rhs: anytype,
    ffn_in: *const fucina.Tensor(.{ .seq, .embed }),
    profile: ?*ForwardProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    const gate_up_start = profileStart(profile, io);
    var gate_up = try gate_up_weight.linearSeq(ctx, ffn_in, .embed, .gate_up);
    defer gate_up.deinit();
    if (profile) |p| p.gate_up_ns += profileElapsed(gate_up_start, io);

    const down_start = profileStart(profile, io);
    const out = try gate_up.splitSwiGluDotPacked(ctx, rhs, .gate_up, .embed);
    if (profile) |p| p.down_ns += profileElapsed(down_start, io);
    return out;
}

/// `denseFfn` for fused-input models whose gate_up projection routes
/// through the fused normalize+quantize+packed GEMM: the FFN-norm output is
/// never materialized (f32-roundoff identical to rmsNormMul + denseFfn — see
/// LinearWeight.linearSeqNormed). Callers guarantee `dense.input_proj` is
/// `.fused` and supportsNormedFusion held.
fn denseFfnNormed(
    ctx: *ExecContext,
    io: ?std.Io,
    dense: *const DenseFfn,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    norm_weight: *const fucina.Tensor(.{.embed}),
    eps: f32,
    profile: ?*ForwardProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    const gate_up_weight = &dense.input_proj.fused;
    if (input.dim(.seq) >= 12) {
        switch (dense.down_proj) {
            .q4_k => |*down| if (comptime !fucina.supports_q4_k_mmla) {
                return denseFfnFusedDownNormed(ctx, io, gate_up_weight, &down.packed_rhs, input, norm_weight, eps, profile);
            },
            .q5_k => |*down| return denseFfnFusedDownNormed(ctx, io, gate_up_weight, &down.packed_rhs, input, norm_weight, eps, profile),
            .q6_k => |*down| return denseFfnFusedDownNormed(ctx, io, gate_up_weight, &down.packed_rhs, input, norm_weight, eps, profile),
            .q8_0 => |*down| return denseFfnFusedDownNormed(ctx, io, gate_up_weight, &down.packed_rhs, input, norm_weight, eps, profile),
            else => {},
        }
    }

    const gate_up_start = profileStart(profile, io);
    var gate_up = try gate_up_weight.linearSeqNormed(ctx, input, norm_weight, eps, .embed, .gate_up);
    defer gate_up.deinit();
    if (profile) |p| p.gate_up_ns += profileElapsed(gate_up_start, io);

    const swiglu_start = profileStart(profile, io);
    var gated = try gate_up.splitGated(ctx, .swiglu, .gate_up, .ffn);
    defer gated.deinit();
    if (profile) |p| p.swiglu_ns += profileElapsed(swiglu_start, io);

    const down_start = profileStart(profile, io);
    const down = try dense.down_proj.linearSeq(ctx, &gated, .ffn, .embed);
    if (profile) |p| p.down_ns += profileElapsed(down_start, io);
    return down;
}

/// `denseFfnFusedDown` with the FFN norm fused into the gate_up LHS
/// quantization (see denseFfnNormed).
fn denseFfnFusedDownNormed(
    ctx: *ExecContext,
    io: ?std.Io,
    gate_up_weight: *const LinearWeight,
    rhs: anytype,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    norm_weight: *const fucina.Tensor(.{.embed}),
    eps: f32,
    profile: ?*ForwardProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    const gate_up_start = profileStart(profile, io);
    var gate_up = try gate_up_weight.linearSeqNormed(ctx, input, norm_weight, eps, .embed, .gate_up);
    defer gate_up.deinit();
    if (profile) |p| p.gate_up_ns += profileElapsed(gate_up_start, io);

    const down_start = profileStart(profile, io);
    const out = try gate_up.splitSwiGluDotPacked(ctx, rhs, .gate_up, .embed);
    if (profile) |p| p.down_ns += profileElapsed(down_start, io);
    return out;
}

/// MoE FFN: route each token to its top-k experts (softmax over the tiny router
/// logits, on the host), then run the router-weighted SwiGLU mixture. Decode
/// (seq == 1) uses the fused expert-parallel GEMV; prefill (seq > 1) groups
/// tokens by expert and runs one m>1 GEMM per expert (weights read once, reused
/// across the batch) — far less weight traffic than per-token.
fn moeFfn(
    ctx: *ExecContext,
    io: ?std.Io,
    config: Descriptor,
    moe: *const MoeFfn,
    ffn_in: *const fucina.Tensor(.{ .seq, .embed }),
    profile: ?*ForwardProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    const router_start = profileStart(profile, io);
    const allocator = ctx.allocator;
    const seq = ffn_in.dim(.seq);
    const top_k = config.num_experts_used;

    var logits = try moe.router.linearSeq(ctx, ffn_in, .embed, .expert);
    defer logits.deinit();

    const sel = try allocator.alloc(usize, seq * top_k);
    defer allocator.free(sel);
    const wgt = try allocator.alloc(f32, seq * top_k);
    defer allocator.free(wgt);
    try logits.routerTopK(ctx, .expert, top_k, .{ .normalize_selected = config.norm_topk_prob }, sel, wgt);
    applyExpertTopP(sel, wgt, top_k, config.moe_expert_top_p);
    if (profile) |p| p.router_ns += profileElapsed(router_start, io);

    const moe_profile: ?*fucina.MoeBatchProfile = if (profile) |p| &p.moe_batch else null;
    return weights.moeSwiGluFfnSeq(
        ctx,
        ffn_in,
        &moe.gate,
        &moe.up,
        &moe.down,
        sel,
        wgt,
        top_k,
        config.moe_intermediate_size,
        io,
        moe_profile,
    );
}

/// Router lookahead (pilot): apply the NEXT layer's ffn_norm + router to the
/// current layer's post-attention state and hand the predicted top-k experts
/// to the expert store's background readahead thread. Pure prediction — no
/// routing state changes, and a failure costs only the overlap.
fn pilotPrefetchNext(
    ctx: *ExecContext,
    config: Descriptor,
    next: *Layer,
    next_layer_i: usize,
    x: *const fucina.Tensor(.{ .seq, .embed }),
) !void {
    const moe = switch (next.ffn) {
        .moe => |*m| m,
        else => return,
    };
    const store = switch (moe.gate) {
        .streamed => |*s| s.store,
        else => return,
    };
    const top_k = config.num_experts_used;
    const seq = x.dim(.seq);

    var nrm = try x.rmsNormMul(ctx, .embed, &next.ffn_norm, config.rms_norm_eps);
    defer nrm.deinit();
    var logits = try moe.router.linearSeq(ctx, &nrm, .embed, .expert);
    defer logits.deinit();

    const allocator = ctx.allocator;
    const sel = try allocator.alloc(usize, seq * top_k);
    defer allocator.free(sel);
    const wgt = try allocator.alloc(f32, seq * top_k);
    defer allocator.free(wgt);
    try logits.routerTopK(ctx, .expert, top_k, .{ .normalize_selected = false }, sel, wgt);
    store.pilotHint(next_layer_i, sel);
}

/// Adaptive expert top-p (routing sparsification, off at p >= 1): per token,
/// keep the smallest weight-descending prefix of the selected experts whose
/// cumulative routing weight reaches `p` of the selected total, rescale the
/// kept weights back to that total, and re-point every dropped pair at the
/// token's top expert with weight zero. The pair layout stays (seq, top_k),
/// so the fused MoE ops run unchanged — but dropped experts are neither read
/// from disk nor cached, which is the lever when experts stream from disk
/// (measured on the 30B MoE: p = 0.7 cut streamed disk traffic 55% for
/// modest quality cost). Quality-traded: outputs differ from full top-k.
pub fn applyExpertTopP(selected: []usize, routing_weights: []f32, top_k: usize, p: f32) void {
    if (p >= 1 or top_k <= 1) return;
    std.debug.assert(selected.len == routing_weights.len and selected.len % top_k == 0);
    const n_tokens = selected.len / top_k;
    for (0..n_tokens) |t| {
        const sel = selected[t * top_k ..][0..top_k];
        const wgt = routing_weights[t * top_k ..][0..top_k];
        // Insertion sort, weight-descending (top_k is single digits).
        for (1..top_k) |i| {
            const wi = wgt[i];
            const si = sel[i];
            var j = i;
            while (j > 0 and wgt[j - 1] < wi) : (j -= 1) {
                wgt[j] = wgt[j - 1];
                sel[j] = sel[j - 1];
            }
            wgt[j] = wi;
            sel[j] = si;
        }
        var total: f32 = 0;
        for (wgt) |w| total += w;
        if (!(total > 0)) continue;
        var cum: f32 = 0;
        var keep: usize = top_k;
        for (wgt, 0..) |w, i| {
            cum += w;
            if (cum >= p * total) {
                keep = i + 1;
                break;
            }
        }
        if (keep == top_k) continue;
        // Rescale the kept prefix back to the selected total, preserving the
        // router's normalization choice (norm_topk or raw softmax mass).
        const scale = total / cum;
        for (wgt[0..keep]) |*w| w.* *= scale;
        for (keep..top_k) |i| {
            sel[i] = sel[0];
            wgt[i] = 0;
        }
    }
}

test "applyExpertTopP keeps the cumulative-weight prefix and re-points dropped pairs" {
    // Token routing: weights 0.4, 0.3, 0.2, 0.1 over experts 7, 2, 5, 1.
    var sel = [_]usize{ 2, 7, 1, 5 };
    var wgt = [_]f32{ 0.3, 0.4, 0.1, 0.2 };

    // p = 1: untouched (bit-identical baseline).
    applyExpertTopP(&sel, &wgt, 4, 1.0);
    try std.testing.expectEqualSlices(usize, &.{ 2, 7, 1, 5 }, &sel);
    try std.testing.expectEqualSlices(f32, &.{ 0.3, 0.4, 0.1, 0.2 }, &wgt);

    // p = 0.65: sorted prefix 0.4 + 0.3 = 0.7 >= 0.65 -> keep two, rescale
    // them to the original total (1.0), drop the rest onto the top expert.
    applyExpertTopP(&sel, &wgt, 4, 0.65);
    try std.testing.expectEqualSlices(usize, &.{ 7, 2, 7, 7 }, &sel);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4 / 0.7), wgt[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3 / 0.7), wgt[1], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), wgt[2]);
    try std.testing.expectEqual(@as(f32, 0), wgt[3]);

    // A dominant top-1 collapses routing to one expert.
    var sel1 = [_]usize{ 3, 0 };
    var wgt1 = [_]f32{ 0.9, 0.1 };
    applyExpertTopP(&sel1, &wgt1, 2, 0.8);
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, &sel1);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 0.9) * 0.9, wgt1[0], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), wgt1[1]);
}

fn profileStart(profile: ?*ForwardProfile, io: ?std.Io) i128 {
    return if (profile != null) std.Io.Clock.awake.now(io.?).nanoseconds else 0;
}

fn profileElapsed(start: i128, io: ?std.Io) i128 {
    return std.Io.Clock.awake.now(io.?).nanoseconds - start;
}

/// Causal grouped attention at the model's scale; `k`/`v` may be any KV
/// representation `groupedAttention` accepts (f32 tensors, f16 cache views,
/// or q8_0 block slices with `.kv_seq`/`.kv_heads` in `opts`).
fn causalAttention(
    ctx: *ExecContext,
    config: Descriptor,
    q: *const fucina.Tensor(.{ .seq, .head, .d }),
    k: anytype,
    v: anytype,
    kv_head_for_head: []const usize,
    opts: anytype,
) !fucina.Tensor(.{ .seq, .attn }) {
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(config.head_dim)));
    return q.groupedAttention(ctx, k, v, kv_head_for_head, .attn, scale, opts);
}
test "Qwen3 0.6B config matches expected projection dimensions" {
    const config = Descriptor.qwen3_0_6b();
    try config.validate();
    try std.testing.expectEqual(@as(usize, 2048), config.qProjectionDim());
    try std.testing.expectEqual(@as(usize, 1024), config.kvProjectionDim());
}


// ---- host_reference band: types and block code ---------------------------
// Ported verbatim from the hand glm4moe port's block (src/llm/glm4moe/
// model.zig) and parameterized by the Descriptor, so parity with that
// oracle holds bitwise: same host-side f32 rope/attention/routing order,
// same fucina kernels for the heavy linears and the fused MoE mixture.

pub const HostBand = struct {
    layers: []HostLayer,
    rope: HostRope,
    output_norm: []f32,
    attn_scale: f32,

    fn deinit(self: *HostBand, allocator: Allocator) void {
        self.rope.deinit(allocator);
        allocator.free(self.output_norm);
        for (self.layers) |*l| l.deinit(allocator);
        allocator.free(self.layers);
        self.* = undefined;
    }
};

/// Plain-rope cos/sin table over the rotated dims (partial rope: the first
/// `rope_dims` of each head), pairing selected by the descriptor.
const HostRope = struct {
    cos: []f32,
    sin: []f32,
    pairs: usize,
    pairing: RopePairing,
    capacity: usize,

    fn init(allocator: Allocator, config: Descriptor, capacity: usize) !HostRope {
        const rope_dims = if (config.rope_dims == 0) config.head_dim else config.rope_dims;
        const pairs = rope_dims / 2;
        const cos = try allocator.alloc(f32, capacity * pairs);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, capacity * pairs);
        for (0..capacity) |pos| {
            for (0..pairs) |i| {
                const freq = std.math.pow(f64, config.rope_theta, -(@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rope_dims))));
                const angle = @as(f64, @floatFromInt(pos)) * freq;
                cos[pos * pairs + i] = @floatCast(@cos(angle));
                sin[pos * pairs + i] = @floatCast(@sin(angle));
            }
        }
        return .{ .cos = cos, .sin = sin, .pairs = pairs, .pairing = config.rope_pairing, .capacity = capacity };
    }

    fn deinit(self: *HostRope, allocator: Allocator) void {
        allocator.free(self.cos);
        allocator.free(self.sin);
        self.* = undefined;
    }

    /// Rotate the first `2*pairs` dims of one head slice, in place.
    fn apply(self: *const HostRope, head: []f32, pos: usize) void {
        const c = self.cos[pos * self.pairs ..][0..self.pairs];
        const s = self.sin[pos * self.pairs ..][0..self.pairs];
        switch (self.pairing) {
            .interleaved => for (0..self.pairs) |i| {
                const a = head[2 * i];
                const b = head[2 * i + 1];
                head[2 * i] = a * c[i] - b * s[i];
                head[2 * i + 1] = a * s[i] + b * c[i];
            },
            .half => for (0..self.pairs) |i| {
                const a = head[i];
                const b = head[i + self.pairs];
                head[i] = a * c[i] - b * s[i];
                head[i + self.pairs] = a * s[i] + b * c[i];
            },
        }
    }
};

/// Host K/V cache: per layer `[capacity, kv_heads, head_dim]` for K
/// (post-rope) and V. `truncate` is the speculative rewind.
pub const HostCache = struct {
    allocator: Allocator,
    k: [][]f32,
    v: [][]f32,
    len: usize = 0,
    capacity: usize,

    pub fn init(allocator: Allocator, n_layers: usize, kv_heads: usize, head_dim: usize, capacity: usize) !HostCache {
        const k = try allocator.alloc([]f32, n_layers);
        var built: usize = 0;
        errdefer {
            for (k[0..built]) |l| allocator.free(l);
            allocator.free(k);
        }
        for (0..n_layers) |i| {
            k[i] = try allocator.alloc(f32, capacity * kv_heads * head_dim);
            built += 1;
        }
        const v = try allocator.alloc([]f32, n_layers);
        errdefer allocator.free(v);
        var v_built: usize = 0;
        errdefer for (v[0..v_built]) |l| allocator.free(l);
        for (0..n_layers) |i| {
            v[i] = try allocator.alloc(f32, capacity * kv_heads * head_dim);
            v_built += 1;
        }
        return .{ .allocator = allocator, .k = k, .v = v, .capacity = capacity };
    }

    pub fn deinit(self: *HostCache) void {
        for (self.k) |l| self.allocator.free(l);
        for (self.v) |l| self.allocator.free(l);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.* = undefined;
    }

    pub fn truncate(self: *HostCache, keep: usize) void {
        if (keep < self.len) self.len = keep;
    }
};

const HostMoeFfn = struct {
    router: LinearWeight,
    router_bias: ?[]f32,
    gate: fucina.MoeRhs,
    up: fucina.MoeRhs,
    down: fucina.MoeRhs,
    shared_gate: LinearWeight,
    shared_up: LinearWeight,
    shared_down: LinearWeight,

    fn deinit(self: *HostMoeFfn, allocator: Allocator) void {
        self.shared_down.deinit();
        self.shared_up.deinit();
        self.shared_gate.deinit();
        self.down.deinit();
        self.up.deinit();
        self.gate.deinit();
        if (self.router_bias) |b| allocator.free(b);
        self.router.deinit();
        self.* = undefined;
    }
};

const HostDenseFfn = struct {
    gate: LinearWeight,
    up: LinearWeight,
    down: LinearWeight,

    fn deinit(self: *HostDenseFfn) void {
        self.down.deinit();
        self.up.deinit();
        self.gate.deinit();
        self.* = undefined;
    }
};

const HostFfn = union(enum) {
    dense: HostDenseFfn,
    moe: HostMoeFfn,

    fn deinit(self: *HostFfn, allocator: Allocator) void {
        switch (self.*) {
            .dense => |*d| d.deinit(),
            .moe => |*m| m.deinit(allocator),
        }
        self.* = undefined;
    }
};

pub const HostLayer = struct {
    attn_norm: []f32,
    post_attention_norm: []f32,
    q_proj: LinearWeight,
    q_bias: []f32,
    k_proj: LinearWeight,
    k_bias: []f32,
    v_proj: LinearWeight,
    v_bias: []f32,
    o_proj: LinearWeight,
    ffn: HostFfn,

    fn deinit(self: *HostLayer, allocator: Allocator) void {
        self.ffn.deinit(allocator);
        self.o_proj.deinit();
        allocator.free(self.v_bias);
        self.v_proj.deinit();
        allocator.free(self.k_bias);
        self.k_proj.deinit();
        allocator.free(self.q_bias);
        self.q_proj.deinit();
        allocator.free(self.post_attention_norm);
        allocator.free(self.attn_norm);
        self.* = undefined;
    }
};

fn loadHostLayer(ctx: *ExecContext, file: *const gguf.File, config: Descriptor, layer_i: usize, store: ?*fucina.ExpertStore) !HostLayer {
    const allocator = ctx.allocator;
    var name_buf: [96]u8 = undefined;

    const attn_norm = try weights.hostVector(allocator, file, try weights.layerName(&name_buf, layer_i, "attn_norm.weight"), config.hidden_size);
    errdefer allocator.free(attn_norm);
    const post_attention_norm = try weights.hostVector(allocator, file, try weights.layerName(&name_buf, layer_i, "post_attention_norm.weight"), config.hidden_size);
    errdefer allocator.free(post_attention_norm);

    const q_dim = config.qProjectionDim();
    const kv_dim = config.kvProjectionDim();
    var q_proj = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "attn_q.weight")), q_dim, config.hidden_size);
    errdefer q_proj.deinit();
    const q_bias = try weights.hostVector(allocator, file, try weights.layerName(&name_buf, layer_i, "attn_q.bias"), q_dim);
    errdefer allocator.free(q_bias);
    var k_proj = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "attn_k.weight")), kv_dim, config.hidden_size);
    errdefer k_proj.deinit();
    const k_bias = try weights.hostVector(allocator, file, try weights.layerName(&name_buf, layer_i, "attn_k.bias"), kv_dim);
    errdefer allocator.free(k_bias);
    var v_proj = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "attn_v.weight")), kv_dim, config.hidden_size);
    errdefer v_proj.deinit();
    const v_bias = try weights.hostVector(allocator, file, try weights.layerName(&name_buf, layer_i, "attn_v.bias"), kv_dim);
    errdefer allocator.free(v_bias);
    var o_proj = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "attn_output.weight")), config.hidden_size, q_dim);
    errdefer o_proj.deinit();

    var ffn: HostFfn = undefined;
    if (layer_i < config.leading_dense_layers or config.num_experts == 0) {
        var gate = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_gate.weight")), config.intermediate_size, config.hidden_size);
        errdefer gate.deinit();
        var up = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_up.weight")), config.intermediate_size, config.hidden_size);
        errdefer up.deinit();
        var down = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_down.weight")), config.hidden_size, config.intermediate_size);
        errdefer down.deinit();
        ffn = .{ .dense = .{ .gate = gate, .up = up, .down = down } };
    } else {
        var router = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_gate_inp.weight")), config.num_experts, config.hidden_size);
        errdefer router.deinit();
        var router_bias: ?[]f32 = null;
        errdefer if (router_bias) |b| allocator.free(b);
        var bias_buf: [96]u8 = undefined;
        if (file.maybeGet(try weights.layerName(&bias_buf, layer_i, "exp_probs_b.bias"))) |bias_info| {
            router_bias = try weights.hostVectorInfo(allocator, bias_info, config.num_experts);
        }
        var gate: fucina.MoeRhs = undefined;
        var up: fucina.MoeRhs = undefined;
        var down: fucina.MoeRhs = undefined;
        if (store) |st| {
            const trio = try weights.loadMoeRhsStreamed(st, file, layer_i, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_gate_exps.weight")), try file.get(try weights.layerName(&name_buf, layer_i, "ffn_up_exps.weight")), try file.get(try weights.layerName(&name_buf, layer_i, "ffn_down_exps.weight")), config.hidden_size, config.moe_intermediate_size, config.num_experts);
            gate = trio.gate;
            up = trio.up;
            down = trio.down;
        } else {
            const borrow = file.is_mmap and !file.isSplit();
            gate = try weights.loadMoeRhs(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_gate_exps.weight")), config.hidden_size, config.moe_intermediate_size, config.num_experts, borrow);
            up = try weights.loadMoeRhs(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_up_exps.weight")), config.hidden_size, config.moe_intermediate_size, config.num_experts, borrow);
            down = try weights.loadMoeRhs(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_down_exps.weight")), config.moe_intermediate_size, config.hidden_size, config.num_experts, borrow);
        }
        const shared_ffn = config.moe_intermediate_size * config.num_shared_experts;
        var shared_gate = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_gate_shexp.weight")), shared_ffn, config.hidden_size);
        errdefer shared_gate.deinit();
        var shared_up = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_up_shexp.weight")), shared_ffn, config.hidden_size);
        errdefer shared_up.deinit();
        var shared_down = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_down_shexp.weight")), config.hidden_size, shared_ffn);
        errdefer shared_down.deinit();
        ffn = .{ .moe = .{ .router = router, .router_bias = router_bias, .gate = gate, .up = up, .down = down, .shared_gate = shared_gate, .shared_up = shared_up, .shared_down = shared_down } };
    }

    return .{
        .attn_norm = attn_norm,
        .post_attention_norm = post_attention_norm,
        .q_proj = q_proj,
        .q_bias = q_bias,
        .k_proj = k_proj,
        .k_bias = k_bias,
        .v_proj = v_proj,
        .v_bias = v_bias,
        .o_proj = o_proj,
        .ffn = ffn,
    };
}

/// One transformer layer over `x` rows in place (the hand glm4moe
/// `layerForward` minus its MTP threading).
fn hostLayerForward(ctx: *ExecContext, cfg: Descriptor, band: *const HostBand, cache: *HostCache, layer: *const HostLayer, layer_i: usize, x: []f32, S: usize, pos0: usize) !void {
    const allocator = ctx.allocator;
    const heads_per_kv = cfg.num_attention_heads / cfg.num_key_value_heads;

    const h_norm = try allocator.alloc(f32, S * cfg.hidden_size);
    defer allocator.free(h_norm);
    for (0..S) |r| hostRmsNormInto(h_norm[r * cfg.hidden_size ..][0..cfg.hidden_size], x[r * cfg.hidden_size ..][0..cfg.hidden_size], layer.attn_norm, cfg.rms_norm_eps);
    var h_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ S, cfg.hidden_size }, h_norm);
    defer h_t.deinit();

    var q_t = try layer.q_proj.linearSeq(ctx, &h_t, .embed, .q);
    defer q_t.deinit();
    var k_t = try layer.k_proj.linearSeq(ctx, &h_t, .embed, .k);
    defer k_t.deinit();
    var v_t = try layer.v_proj.linearSeq(ctx, &h_t, .embed, .v);
    defer v_t.deinit();
    const q = try allocator.dupe(f32, try q_t.dataConst());
    defer allocator.free(q);
    const k = try allocator.dupe(f32, try k_t.dataConst());
    defer allocator.free(k);
    const v = try allocator.dupe(f32, try v_t.dataConst());
    defer allocator.free(v);
    const q_width = cfg.qProjectionDim();
    const kv_width = cfg.kvProjectionDim();
    if (cfg.qkv_bias) {
        for (0..S) |r| {
            const q_row = q[r * q_width ..][0..q_width];
            for (q_row, layer.q_bias) |*qv, b| qv.* += b;
            const k_row = k[r * kv_width ..][0..kv_width];
            for (k_row, layer.k_bias) |*kv, b| kv.* += b;
            const v_row = v[r * kv_width ..][0..kv_width];
            for (v_row, layer.v_bias) |*vv, b| vv.* += b;
        }
    }

    var attn_t = try fucina.Tensor(.{ .seq, .embed }).empty(ctx, .{ S, q_width });
    defer attn_t.deinit();
    const attn_out = try attn_t.data();
    const scores = try allocator.alloc(f32, pos0 + S);
    defer allocator.free(scores);

    for (0..S) |r| {
        const pos = pos0 + r;
        // Partial rope + append this position's K/V.
        const k_row = k[r * kv_width ..][0..kv_width];
        const v_row = v[r * kv_width ..][0..kv_width];
        for (0..cfg.num_key_value_heads) |h| {
            band.rope.apply(k_row[h * cfg.head_dim ..][0..cfg.head_dim], pos);
        }
        const k_dst = cache.k[layer_i][pos * kv_width ..][0..kv_width];
        const v_dst = cache.v[layer_i][pos * kv_width ..][0..kv_width];
        @memcpy(k_dst, k_row);
        @memcpy(v_dst, v_row);

        const t_len = pos + 1;
        const q_row = q[r * q_width ..][0..q_width];
        for (0..cfg.num_attention_heads) |h| {
            const q_head = q_row[h * cfg.head_dim ..][0..cfg.head_dim];
            band.rope.apply(q_head, pos);
            const kv_h = h / heads_per_kv;
            for (0..t_len) |t| {
                const kt = cache.k[layer_i][(t * cfg.num_key_value_heads + kv_h) * cfg.head_dim ..][0..cfg.head_dim];
                var dot: f32 = 0;
                for (q_head, kt) |a, b| dot += a * b;
                scores[t] = dot * band.attn_scale;
            }
            hostSoftmaxInPlace(scores[0..t_len]);
            const out_head = attn_out[r * q_width + h * cfg.head_dim ..][0..cfg.head_dim];
            @memset(out_head, 0);
            for (0..t_len) |t| {
                const w = scores[t];
                const vt = cache.v[layer_i][(t * cfg.num_key_value_heads + kv_h) * cfg.head_dim ..][0..cfg.head_dim];
                for (out_head, vt) |*o, val| o.* += w * val;
            }
        }
    }

    var o_t = try layer.o_proj.linearSeq(ctx, &attn_t, .embed, .attn);
    defer o_t.deinit();
    for (x, try o_t.dataConst()) |*xi, oi| xi.* += oi;

    // FFN with the sandwich naming (post_attention_norm = pre-FFN).
    for (0..S) |r| hostRmsNormInto(h_norm[r * cfg.hidden_size ..][0..cfg.hidden_size], x[r * cfg.hidden_size ..][0..cfg.hidden_size], layer.post_attention_norm, cfg.rms_norm_eps);
    switch (layer.ffn) {
        .dense => |*dense| {
            var f_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ S, cfg.hidden_size }, h_norm);
            defer f_t.deinit();
            const y = try hostSwigluLinear(ctx, allocator, &f_t, &dense.gate, &dense.up, &dense.down);
            defer allocator.free(y);
            for (x, y) |*xi, yi| xi.* += yi;
        },
        .moe => |*moe| {
            // Row-wise through the fused decode op (consecutive rows share
            // expert reads via the store's LRU when streaming).
            for (0..S) |r| {
                const row = h_norm[r * cfg.hidden_size ..][0..cfg.hidden_size];
                var f_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ 1, cfg.hidden_size }, row);
                defer f_t.deinit();
                const y = try hostMoeForward(ctx, cfg, allocator, moe, &f_t);
                defer allocator.free(y);
                for (x[r * cfg.hidden_size ..][0..cfg.hidden_size], y) |*xi, yi| xi.* += yi;
            }
        },
    }
}

/// DeepSeek-V3-style routing: sigmoid (or softmax) scores, selection bias
/// for the top-k choice only, renormalized weights, expert_weights_scale,
/// plus the always-on shared experts.
fn hostMoeForward(ctx: *ExecContext, cfg: Descriptor, allocator: Allocator, moe: *const HostMoeFfn, f_t: *const fucina.Tensor(.{ .seq, .embed })) ![]f32 {
    var logits_t = try moe.router.linearSeq(ctx, f_t, .embed, .expert);
    defer logits_t.deinit();
    const probs = try allocator.dupe(f32, try logits_t.dataConst());
    defer allocator.free(probs);
    switch (cfg.expert_gating_func) {
        2 => for (probs) |*p| {
            p.* = 1.0 / (1.0 + @exp(-p.*));
        },
        else => hostSoftmaxInPlace(probs),
    }
    const choice = try allocator.dupe(f32, probs);
    defer allocator.free(choice);
    if (moe.router_bias) |bias| {
        for (choice, bias) |*c, b| c.* += b;
    }

    var selected: [64]usize = undefined;
    var routing: [64]f32 = undefined;
    std.debug.assert(cfg.num_experts_used <= selected.len);
    if (weights.cacheRouteSel(&moe.gate, choice, selected[0..cfg.num_experts_used])) {
        for (selected[0..cfg.num_experts_used], routing[0..cfg.num_experts_used]) |e, *w| w.* = probs[e];
    } else for (0..cfg.num_experts_used) |slot| {
        var best: usize = 0;
        var best_c: f32 = -std.math.inf(f32);
        for (choice, 0..) |c, e| {
            if (c > best_c) {
                best_c = c;
                best = e;
            }
        }
        choice[best] = -std.math.inf(f32);
        selected[slot] = best;
        routing[slot] = probs[best];
    }
    if (cfg.expert_weights_norm) {
        var total: f32 = 1e-20;
        for (routing[0..cfg.num_experts_used]) |w| total += w;
        for (routing[0..cfg.num_experts_used]) |*w| w.* /= total;
    }
    for (routing[0..cfg.num_experts_used]) |*w| w.* *= cfg.expert_weights_scale;

    var mix = try weights.moeSwiGluFfnSeq(ctx, f_t, &moe.gate, &moe.up, &moe.down, selected[0..cfg.num_experts_used], routing[0..cfg.num_experts_used], cfg.num_experts_used, cfg.moe_intermediate_size, null, null);
    defer mix.deinit();

    const y = try allocator.dupe(f32, try mix.dataConst());
    errdefer allocator.free(y);
    const shared = try hostSwigluLinear(ctx, allocator, f_t, &moe.shared_gate, &moe.shared_up, &moe.shared_down);
    defer allocator.free(shared);
    for (y, shared) |*yi, si| yi.* += si;
    return y;
}

fn hostSwigluLinear(ctx: *ExecContext, allocator: Allocator, x: *const fucina.Tensor(.{ .seq, .embed }), gate: *const LinearWeight, up: *const LinearWeight, down: *const LinearWeight) ![]f32 {
    var gate_t = try gate.linearSeq(ctx, x, .embed, .gate_up);
    defer gate_t.deinit();
    var up_t = try up.linearSeq(ctx, x, .embed, .gate_up);
    defer up_t.deinit();
    const width = gate_t.dim(.gate_up);
    const rows = gate_t.dim(.seq);
    var g_t = try fucina.Tensor(.{ .seq, .embed }).empty(ctx, .{ rows, width });
    defer g_t.deinit();
    for (try g_t.data(), try gate_t.dataConst(), try up_t.dataConst()) |*gi, gv, uv| gi.* = hostSilu(gv) * uv;
    var down_t = try down.linearSeq(ctx, &g_t, .embed, .attn);
    defer down_t.deinit();
    return allocator.dupe(f32, try down_t.dataConst());
}

fn hostRmsNormInto(out: []f32, x: []const f32, weight: []const f32, eps: f32) void {
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, v) * v;
    const inv = 1.0 / @sqrt(sum / @as(f64, @floatFromInt(x.len)) + eps);
    for (out, x, weight) |*o, v, w| o.* = @floatCast(@as(f64, v) * inv * w);
}

fn hostSoftmaxInPlace(v: []f32) void {
    var max: f32 = -std.math.inf(f32);
    for (v) |x| max = @max(max, x);
    var sum: f32 = 0;
    for (v) |*x| {
        x.* = @exp(x.* - max);
        sum += x.*;
    }
    for (v) |*x| x.* /= sum;
}

fn hostSilu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}
