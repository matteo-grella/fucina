const std = @import("std");
const fucina = @import("fucina");
const weights = @import("../weights.zig");
const kv_cache = @import("../kv_cache.zig");
const gguf_meta = @import("../gguf_meta.zig");
const ptqtp_gguf = @import("../ptqtp_gguf.zig");

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
    /// `forwardStepBatch` requires distinct sibling caches: one per stream,
    /// all the same dtype (all from this model's `initKvCache`).
    MismatchedKvCaches,
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

    pub fn isMoe(self: Config) bool {
        return self.num_experts > 0;
    }

    pub fn qwen3_0_6b() Config {
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
    pub fn fromGguf(file: *const gguf.File) !Config {
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        const embd = try file.get("token_embd.weight");
        const shape = try embd.logicalMatrixShape(); // {vocab, hidden}

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

    fn validate(self: Config) !void {
        if (self.num_attention_heads == 0 or self.num_key_value_heads == 0) return Error.InvalidConfig;
        if (self.num_attention_heads % self.num_key_value_heads != 0) return Error.InvalidConfig;
        if (self.head_dim % 2 != 0) return Error.InvalidConfig;
        if (self.isMoe()) {
            if (self.num_experts_used == 0 or self.num_experts_used > self.num_experts) return Error.InvalidConfig;
            if (self.moe_intermediate_size == 0) return Error.InvalidConfig;
        }
    }

    fn qProjectionDim(self: Config) usize {
        return self.num_attention_heads * self.head_dim;
    }

    fn kvProjectionDim(self: Config) usize {
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

/// Opt-in disk streaming for the MoE expert stacks: experts stay on disk and
/// are `pread` on demand through a tiered store (LRU + working set), so a
/// mixture model loads with only its dense weights resident. Decode then
/// pays disk reads for expert-cache misses — the explicit trade that lets a
/// bigger-than-RAM model run at all (docs: out-of-core MoE).
pub const MoeStreamOptions = struct {
    /// Path of the same GGUF being loaded; the store opens its own read fd
    /// (the load-time mmap is released after load — resident memory stays
    /// dense weights + expert cache).
    gguf_path: []const u8,
    /// Total RAM budget for the expert LRU tier (all layers). Default: half
    /// of available memory at load time.
    cache_bytes: ?usize = null,
    /// Fixed LRU slots per layer; wins over `cache_bytes` when set.
    cache_slots_per_layer: ?usize = null,
    /// OS readahead hints for miss batches.
    readahead: bool = true,
    /// The learning cache: pin the hottest experts from the persisted usage
    /// sidecar (`<gguf>.experts`) at load; save updated counts with
    /// `ExpertStore.saveUsage` at generation/turn boundaries.
    auto_pin: bool = true,
    /// RAM for the pinned tier (default: half the budget when history
    /// qualifies).
    pin_bytes: ?usize = null,
    /// Router-lookahead prefetch: predict each next layer's experts from the
    /// current post-attention state and readahead them from a background
    /// I/O thread while the current layer computes. Prediction recall is
    /// measured in `ExpertStore.Stats`.
    pilot: bool = false,
};

pub const LoadOptions = struct {
    moe_stream: ?MoeStreamOptions = null,
};

pub const Model = struct {
    allocator: Allocator,
    config: Config,
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

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, config: Config) !Model {
        return loadGgufOptions(ctx, io, path, config, .{});
    }

    pub fn loadGgufOptions(ctx: *ExecContext, io: std.Io, path: []const u8, config: Config, options: LoadOptions) !Model {
        // mmap, matching the CLI (examples/qwen3.zig) and the other loaders:
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
    pub fn loadGgufFromFile(ctx: *ExecContext, file: *gguf.File, config: Config) !Model {
        return loadGgufFromFileOptions(ctx, file, config, .{});
    }

    pub fn loadGgufFromFileOptions(ctx: *ExecContext, file: *gguf.File, config: Config, options: LoadOptions) !Model {
        try config.validate();

        const allocator = ctx.allocator;

        var expert_store: ?*fucina.ExpertStore = null;
        if (options.moe_stream) |stream_options| {
            if (config.isMoe()) {
                // Split GGUFs: the store opens every part (TensorInfo.part
                // indexes them); single files pass through as one entry.
                const split_paths = try gguf.File.splitPartPaths(allocator, stream_options.gguf_path);
                defer if (split_paths) |paths| {
                    for (paths) |p| allocator.free(p);
                    allocator.free(paths);
                };
                var one_path = [_][]const u8{stream_options.gguf_path};
                const store_paths: []const []const u8 = if (split_paths) |paths| blk: {
                    const view = try allocator.alloc([]const u8, paths.len);
                    for (view, paths) |*d, src| d.* = src;
                    break :blk view;
                } else &one_path;
                defer if (split_paths != null) allocator.free(store_paths);
                expert_store = try fucina.ExpertStore.create(allocator, store_paths, config.num_layers, .{
                    .cache_bytes = stream_options.cache_bytes,
                    .cache_slots_per_layer = stream_options.cache_slots_per_layer,
                    .readahead = stream_options.readahead,
                    .auto_pin = stream_options.auto_pin,
                    .pin_bytes = stream_options.pin_bytes,
                });
            }
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

        const positions = try ctx.allocator.alloc(i32, token_ids.len);
        defer ctx.allocator.free(positions);
        for (positions, 0..) |*position, i| position.* = @intCast(i);

        var rope_table = try ctx.prepareRopeTable(positions, self.config.head_dim, self.config.rope_theta, false);
        defer rope_table.deinit();

        var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
        errdefer x.deinit();

        const cfg = self.config;
        for (self.layers, 0..) |*layer, layer_i| {
            const last_query_only = layer_i + 1 == cfg.num_layers and token_ids.len > 1;
            x = try ctx.replace(x, attentionBlock(ctx, io, cfg, layer, &x, &rope_table, self.kv_head_for_head, last_query_only, profile, null, layer_i));

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

    /// PTQTP-decorate every eligible layer linear in place (attention
    /// q/k/v — split or fused — o_proj, dense FFN gate/up/down): each
    /// becomes two packed TQ2_0 trit-planes and its original storage is
    /// dropped (docs/PTQTP.md). Embeddings, the lm_head, norms, and MoE
    /// expert stacks are left untouched (MoE FFNs count as skipped).
    pub const DecoratePtqtpOptions = struct {
        solver: fucina.ptqtp.Options = .{},
        /// Leave the first/last N layers in their source precision — the
        /// edge layers are the most quantization-sensitive in extreme
        /// low-bit practice (data-free: pure configuration, no inputs).
        skip_first_layers: usize = 0,
        skip_last_layers: usize = 0,
        /// Per-projection plane-count overrides (null = solver.planes):
        /// selective capacity for the sensitive projections — an extra
        /// trit-plane costs +2.06 bpw only where applied and keeps every
        /// op ternary, unlike layer skipping which retains source-dtype
        /// matmuls. down_proj (and o_proj) are the classic hot spots.
        down_planes: ?u8 = null,
        o_planes: ?u8 = null,
    };

    pub fn decoratePtqtp(self: *Model, ctx: *ExecContext, decorate_options: DecoratePtqtpOptions) !weights.PtqtpReport {
        const options = decorate_options.solver;
        var report = weights.PtqtpReport{};
        const n_layers = self.layers.len;
        for (self.layers, 0..) |*layer, layer_i| {
            if (layer_i < decorate_options.skip_first_layers or
                layer_i + decorate_options.skip_last_layers >= n_layers)
            {
                report.skipped_layers += 1;
                continue;
            }
            switch (layer.attn_proj) {
                .separate => |*separate| {
                    try weights.decoratePtqtpInto(&separate.q_proj, ctx, options, &report);
                    try weights.decoratePtqtpInto(&separate.k_proj, ctx, options, &report);
                    try weights.decoratePtqtpInto(&separate.v_proj, ctx, options, &report);
                },
                .fused => |*fused| try weights.decoratePtqtpInto(fused, ctx, options, &report),
            }
            var o_options = options;
            if (decorate_options.o_planes) |p| o_options.planes = p;
            try weights.decoratePtqtpInto(&layer.o_proj, ctx, o_options, &report);
            switch (layer.ffn) {
                .dense => |*dense| {
                    switch (dense.input_proj) {
                        .separate => |*separate| {
                            try weights.decoratePtqtpInto(&separate.gate_proj, ctx, options, &report);
                            try weights.decoratePtqtpInto(&separate.up_proj, ctx, options, &report);
                        },
                        .fused => |*fused| try weights.decoratePtqtpInto(fused, ctx, options, &report),
                    }
                    var down_options = options;
                    if (decorate_options.down_planes) |p| down_options.planes = p;
                    try weights.decoratePtqtpInto(&dense.down_proj, ctx, down_options, &report);
                },
                .moe => report.skipped += 1,
            }
        }
        return report;
    }

    /// Persist this (decorated) model as a GGUF beside its source file:
    /// every `.ptqtp` weight becomes per-source-name plane tensors, all
    /// other tensors and metadata pass through byte-verbatim
    /// (ptqtp_gguf.zig; docs/PTQTP.md). Fused in-memory weights are
    /// row-sliced back to their source tensor names, so the saved file is
    /// independent of this load's fusion decisions. `src` must be the
    /// still-open GGUF this model was loaded from.
    pub fn savePtqtpGguf(self: *const Model, ctx: *ExecContext, io: std.Io, src: *const gguf.File, out_path: []const u8) !ptqtp_gguf.SaveReport {
        var arena_state = std.heap.ArenaAllocator.init(ctx.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var entries: std.ArrayList(ptqtp_gguf.SaveEntry) = .empty;
        try entries.append(arena, .{ .name = "output.weight", .weight = &self.output });

        const q_dim = self.config.qProjectionDim();
        const kv_dim = self.config.kvProjectionDim();
        const ffn_dim = self.config.intermediate_size;
        for (self.layers, 0..) |*layer, layer_i| {
            switch (layer.attn_proj) {
                .separate => |*separate| {
                    try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_q.weight"), .weight = &separate.q_proj });
                    try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_k.weight"), .weight = &separate.k_proj });
                    try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_v.weight"), .weight = &separate.v_proj });
                },
                .fused => |*fused| {
                    try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_q.weight"), .weight = fused, .row0 = 0, .rows = q_dim });
                    try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_k.weight"), .weight = fused, .row0 = q_dim, .rows = kv_dim });
                    try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_v.weight"), .weight = fused, .row0 = q_dim + kv_dim, .rows = kv_dim });
                },
            }
            try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_output.weight"), .weight = &layer.o_proj });
            switch (layer.ffn) {
                .dense => |*dense| {
                    switch (dense.input_proj) {
                        .separate => |*separate| {
                            try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "ffn_gate.weight"), .weight = &separate.gate_proj });
                            try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "ffn_up.weight"), .weight = &separate.up_proj });
                        },
                        .fused => |*fused| {
                            try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "ffn_gate.weight"), .weight = fused, .row0 = 0, .rows = ffn_dim });
                            try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "ffn_up.weight"), .weight = fused, .row0 = ffn_dim, .rows = ffn_dim });
                        },
                    }
                    try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "ffn_down.weight"), .weight = &dense.down_proj });
                },
                .moe => {},
            }
        }

        // MoE loads moved the mmap into the model (loadMoeFfn borrows expert
        // blocks), so the metadata copy reads the region through us.
        const options = ptqtp_gguf.SaveOptions{
            .header_bytes = if (self.weight_mapping) |mapping| mapping.bytes else null,
        };
        return ptqtp_gguf.saveFile(ctx.allocator, io, src, entries.items, options, out_path);
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
        return self.forwardStepImpl(ctx, null, kv, token_ids, pos0, null, true);
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
        return self.forwardStepImpl(ctx, io, kv, token_ids, pos0, profile, true);
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
        return self.forwardStepImpl(ctx, null, kv, token_ids, pos0, null, false);
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
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        if (token_ids.len == 0) return Error.InvalidSequenceLength;
        if (kv.len != pos0) return Error.InvalidSequenceLength;
        if (kv.len + token_ids.len > kv.capacity) return kv_cache.Error.KvCacheOverflow;

        const positions = try ctx.allocator.alloc(i32, token_ids.len);
        defer ctx.allocator.free(positions);
        for (positions, 0..) |*position, i| position.* = @intCast(pos0 + i);

        var rope_table = try ctx.prepareRopeTable(positions, self.config.head_dim, self.config.rope_theta, false);
        defer rope_table.deinit();

        var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
        errdefer x.deinit();

        const cfg = self.config;
        for (self.layers, 0..) |*layer, layer_i| {
            const last_query_only = last_only and layer_i + 1 == cfg.num_layers and token_ids.len > 1;
            x = try ctx.replace(x, attentionBlock(ctx, io, cfg, layer, &x, &rope_table, self.kv_head_for_head, last_query_only, profile, kv, layer_i));
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

    /// Greedy autoregressive generation: prefill `prompt_tokens`, then sample
    /// the argmax token each step into `out_tokens` until `max_new_tokens`,
    /// `out_tokens.len`, or the optional `stop_token` is reached. Resets `kv`.
    /// Returns the number of tokens written.
    pub fn generate(
        self: *const Model,
        ctx: *ExecContext,
        kv: *KvCache,
        prompt_tokens: []const usize,
        out_tokens: []usize,
        options: GenerateOptions,
    ) !usize {
        if (prompt_tokens.len == 0) return Error.InvalidSequenceLength;
        kv.reset();

        var logits = try self.forwardStep(ctx, kv, prompt_tokens, 0);
        defer logits.deinit();

        const limit = @min(options.max_new_tokens, out_tokens.len);
        var produced: usize = 0;
        while (produced < limit) {
            const next = try argmaxLast(ctx, &logits);
            out_tokens[produced] = next;
            produced += 1;
            if (options.stop_token) |stop| if (next == stop) break;
            if (produced == limit) break;
            // Allocate the next step before freeing the current logits, so an
            // error here leaves `logits` valid for the function-scope defer
            // (deinit-then-reassign would leave it dangling on the error path).
            const fresh = try self.forwardStep(ctx, kv, &.{next}, kv.len);
            logits.deinit();
            logits = fresh;
        }
        return produced;
    }
};

pub const GenerateOptions = struct {
    max_new_tokens: usize,
    stop_token: ?usize = null,
};

fn argmaxLast(ctx: *ExecContext, logits: *const fucina.Tensor(.{ .seq, .vocab })) !usize {
    var last = try logits.narrow(ctx, .seq, logits.dim(.seq) - 1, 1);
    defer last.deinit();
    var index = try last.argmax(ctx, .vocab);
    defer index.deinit();
    return @intCast(try index.item());
}

const DenseFfn = struct {
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

/// `weights.layerName` with owned storage — `savePtqtpGguf` builds its
/// whole entry list before any name is consumed.
fn layerNameOwned(allocator: Allocator, layer_i: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.allocPrint(allocator, "blk.{d}.{s}", .{ layer_i, suffix });
}

fn loadDenseFfn(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize) !DenseFfn {
    var name_buf: [96]u8 = undefined;

    var input_proj = try FfnInputProjection.load(ctx, file, config, layer_i);
    errdefer input_proj.deinit();

    var down_proj = try loadProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_down.weight"), config.hidden_size, config.intermediate_size, false);
    errdefer down_proj.deinit();

    return .{ .input_proj = input_proj, .down_proj = down_proj };
}

fn loadMoeFfn(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize, store: ?*fucina.ExpertStore) !MoeFfn {
    var name_buf: [96]u8 = undefined;

    var router = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_gate_inp.weight")), config.num_experts, config.hidden_size);
    errdefer router.deinit();

    // Streamed: register geometry only — the expert stacks stay on disk and
    // never become resident. addLayer touches only this layer's state, so
    // the parallel layer loader may call it concurrently for distinct layers.
    if (store) |s| {
        const trio = try weights.loadMoeRhsStreamed(
            s,
            file,
            layer_i,
            try file.get(try weights.layerName(&name_buf, layer_i, "ffn_gate_exps.weight")),
            try file.get(try weights.layerName(&name_buf, layer_i, "ffn_up_exps.weight")),
            try file.get(try weights.layerName(&name_buf, layer_i, "ffn_down_exps.weight")),
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
        );
        return .{ .router = router, .gate = trio.gate, .up = trio.up, .down = trio.down };
    }

    // Expert blocks need no repack, so when the GGUF is mmap'd they are
    // borrowed straight from the mapping (the Model takes ownership of it in
    // loadGgufFromFile) instead of copying the multi-GB stacks. Split GGUFs
    // cannot hand over their multiple mappings (takeMapping declines), so
    // their experts are copied — stream them instead for the big models.
    const borrow = file.is_mmap and !file.isSplit();

    var gate = try weights.loadMoeRhs(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_gate_exps.weight")), config.hidden_size, config.moe_intermediate_size, config.num_experts, borrow);
    errdefer gate.deinit();
    var up = try weights.loadMoeRhs(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_up_exps.weight")), config.hidden_size, config.moe_intermediate_size, config.num_experts, borrow);
    errdefer up.deinit();
    var down = try weights.loadMoeRhs(ctx, try file.get(try weights.layerName(&name_buf, layer_i, "ffn_down_exps.weight")), config.moe_intermediate_size, config.hidden_size, config.num_experts, borrow);
    errdefer down.deinit();

    return .{ .router = router, .gate = gate, .up = up, .down = down };
}

const Layer = struct {
    attn_norm: fucina.Tensor(.{.embed}),
    q_norm: fucina.Tensor(.{.d}),
    k_norm: fucina.Tensor(.{.d}),
    ffn_norm: fucina.Tensor(.{.embed}),
    attn_proj: AttentionProjection,
    o_proj: LinearWeight,
    ffn: Ffn,

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize, store: ?*fucina.ExpertStore) !Layer {
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

    fn deinit(self: *Layer) void {
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
    config: Config,
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
fn loadLayers(ctx: *ExecContext, file: *const gguf.File, config: Config, layers: []Layer, store: ?*fucina.ExpertStore) !void {
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

const AttentionProjection = union(enum) {
    separate: SeparateAttentionProjection,
    fused: LinearWeight,

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize) !AttentionProjection {
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
        config: Config,
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

    fn project(self: *const AttentionProjection, ctx: *ExecContext, input: *const fucina.Tensor(.{ .seq, .embed }), config: Config) !QkvProjection {
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

fn splitQkv(ctx: *ExecContext, qkv: *const fucina.Tensor(.{ .seq, .qkv }), config: Config) !QkvProjection {
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

const GateUpProjection = struct {
    gate: fucina.Tensor(.{ .seq, .ffn }),
    up: fucina.Tensor(.{ .seq, .ffn }),

    fn deinit(self: *GateUpProjection) void {
        self.up.deinit();
        self.gate.deinit();
        self.* = undefined;
    }
};

const FfnInputProjection = union(enum) {
    separate: SeparateFfnInputProjection,
    fused: LinearWeight,

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize) !FfnInputProjection {
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

    fn project(self: *const FfnInputProjection, ctx: *ExecContext, input: *const fucina.Tensor(.{ .seq, .embed }), config: Config) !GateUpProjection {
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

fn splitGateUp(ctx: *ExecContext, gate_up: *const fucina.Tensor(.{ .seq, .gate_up }), config: Config) !GateUpProjection {
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
    config: Config,
    layer: *const Layer,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    rope_table: *const fucina.RopeTable,
    kv_head_for_head: []const usize,
    last_query_only: bool,
    profile: ?*ForwardProfile,
    cache: ?*KvCache,
    layer_i: usize,
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
    config: Config,
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

fn ffnBlock(
    ctx: *ExecContext,
    io: ?std.Io,
    config: Config,
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
    config: Config,
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
    config: Config,
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
    config: Config,
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
/// (colibri measured 30-40% fewer expert loads at p ~= 0.7 for modest
/// quality cost). Quality-traded: outputs differ from full top-k.
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
    config: Config,
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
    const config = Config.qwen3_0_6b();
    try config.validate();
    try std.testing.expectEqual(@as(usize, 2048), config.qProjectionDim());
    try std.testing.expectEqual(@as(usize, 1024), config.kvProjectionDim());
}

test {
    _ = @import("model_tests.zig");
}
