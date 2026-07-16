//! Inkling (`inkling` GGUF arch, thinkingmachines/Inkling): hybrid local/
//! global attention with a banded content-dependent relative-position bias
//! instead of RoPE, per-layer short causal convolutions on four sites
//! (k-proj, v-proj, attention output, FFN output), a fine-grained sigmoid-
//! routed MoE whose shared experts participate in the routing softmax as
//! sinks, log-N attention scaling on global layers, muP logit scaling, and
//! a padded vocabulary. Reference: llama.cpp PR #25731 @ 1cb0374
//! (`refs/llama.cpp-inkling`, see docs/PORTING.md for the method).
//!
//! Same correctness-first shape as the glm4moe/deepseek2 ports: heavy
//! linears (projections, experts, unembed) run on fucina kernels; the
//! parity-critical control paths — short-conv taps, per-head attention with
//! the rel-bias band, router selection and weighting — run host-side in
//! auditable f32 (docs/PORTING.md §4: verbatim scalar host code). `step`
//! handles S >= 1 causally and returns per-position logits.
//!
//! Deviations register (vs the llama.cpp reference):
//!   D1. Shared-expert gammas scale the swiglu activation BEFORE the down
//!       projection, exactly like the reference graph (no deviation — noted
//!       because the routed experts apply weights AFTER down, also like the
//!       reference; the asymmetry is intentional upstream).
//!   D2. The reference gathers rel-bias values through a flat index tensor
//!       with a zero pad column because its KV cells are not position-
//!       ordered; this cache is position-contiguous, so the bias is added
//!       directly from delta = pos_q - pos_k. Equivalent by construction.
//!   D3. Speculative cache truncate is not supported: the rolling conv
//!       states cannot rewind (would need state snapshots).
const std = @import("std");
const fucina = @import("fucina");
const weights = @import("../weights.zig");
const gguf_meta = @import("../gguf_meta.zig");
const gemma4 = @import("../gemma/gemma4.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;
const LinearWeight = weights.LinearWeight;

pub const Error = weights.Error || error{
    InvalidConfig,
    InvalidSequenceLength,
    KvCacheOverflow,
    MissingMetadata,
};

pub const Config = struct {
    vocab_size: usize,
    unpadded_vocab_size: usize, // ids >= this get -inf logits (0 = disabled)
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    /// Per-layer KV head count (local/SWA layers differ from global ones).
    kv_heads: []usize,
    /// Per-layer window flag (true = local/SWA layer).
    is_swa: []bool,
    sliding_window: usize, // visibility: pos_q - pos_k < sliding_window
    d_rel: usize,
    rel_extent: usize, // rel-bias band width on global layers
    rel_extent_swa: usize, // band width on local layers (= sliding_window)
    shortconv_kernel: usize, // K taps; rolling state is K-1 rows
    dense_layers: usize, // leading dense-FFN layers, rest are MoE
    dense_ffn_size: usize,
    num_experts: usize,
    num_experts_used: usize,
    expert_ffn_size: usize,
    num_shared_experts: usize,
    expert_weights_scale: f32, // route_scale
    logit_scale: f32, // muP: 1 / logit_scale_denom
    log_n_floor: usize, // 0 = log-N attention scaling disabled
    log_alpha: f32,
    rms_norm_eps: f32,

    pub fn fromGguf(allocator: Allocator, file: *const gguf.File) !Config {
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        if (!std.mem.eql(u8, arch, "inkling")) return Error.InvalidConfig;

        const num_layers = try metaInt(file, "inkling.block_count");
        const embd = try file.get("token_embd.weight");
        const embd_shape = try embd.logicalMatrixShape();

        const kv_heads = try gemma4.readU32OrBoolArray(allocator, file, "inkling.attention.head_count_kv", num_layers, usize);
        errdefer allocator.free(kv_heads);
        const is_swa = try gemma4.readU32OrBoolArray(allocator, file, "inkling.attention.sliding_window_pattern", num_layers, bool);
        errdefer allocator.free(is_swa);

        const logit_scale_denom = try metaFloat(file, "inkling.logit_scale_denom");
        if (logit_scale_denom == 0) return Error.InvalidConfig;

        const gating = gguf_meta.metaIntOpt(file, "inkling", "expert_gating_func", .accept_zero) orelse 2;
        if (gating != 2) return Error.InvalidConfig; // sigmoid is the only shipped variant

        return .{
            .vocab_size = embd_shape[0],
            .unpadded_vocab_size = gguf_meta.metaIntOpt(file, "inkling", "unpadded_vocab_size", .accept_zero) orelse 0,
            .hidden_size = try metaInt(file, "inkling.embedding_length"),
            .num_layers = num_layers,
            .num_heads = try metaInt(file, "inkling.attention.head_count"),
            .head_dim = try metaInt(file, "inkling.attention.key_length"),
            .kv_heads = kv_heads,
            .is_swa = is_swa,
            .sliding_window = try metaInt(file, "inkling.attention.sliding_window"),
            .d_rel = try metaInt(file, "inkling.d_rel"),
            .rel_extent = try metaInt(file, "inkling.rel_extent"),
            .rel_extent_swa = try metaInt(file, "inkling.rel_extent_swa"),
            .shortconv_kernel = try metaInt(file, "inkling.shortconv_kernel"),
            .dense_layers = gguf_meta.metaIntOpt(file, "inkling", "dense_block_count", .accept_zero) orelse 0,
            .dense_ffn_size = try metaInt(file, "inkling.feed_forward_length"),
            .num_experts = try metaInt(file, "inkling.expert_count"),
            .num_experts_used = try metaInt(file, "inkling.expert_used_count"),
            .expert_ffn_size = try metaInt(file, "inkling.expert_feed_forward_length"),
            .num_shared_experts = try metaInt(file, "inkling.expert_shared_count"),
            .expert_weights_scale = metaFloatOpt(file, "inkling.expert_weights_scale") orelse 1.0,
            .logit_scale = 1.0 / logit_scale_denom,
            .log_n_floor = gguf_meta.metaIntOpt(file, "inkling", "log_scaling_n_floor", .accept_zero) orelse 0,
            .log_alpha = metaFloatOpt(file, "inkling.log_scaling_alpha") orelse 0.0,
            .rms_norm_eps = try metaFloat(file, "inkling.attention.layer_norm_rms_epsilon"),
        };
    }

    pub fn deinit(self: *Config, allocator: Allocator) void {
        allocator.free(self.kv_heads);
        allocator.free(self.is_swa);
        self.* = undefined;
    }

    pub fn relExtent(self: *const Config, layer_i: usize) usize {
        return if (self.is_swa[layer_i]) self.rel_extent_swa else self.rel_extent;
    }

    fn metaInt(file: *const gguf.File, key: []const u8) !usize {
        const v = file.getInt(key) orelse return Error.InvalidConfig;
        if (v <= 0) return Error.InvalidConfig;
        return @intCast(v);
    }

    fn metaFloat(file: *const gguf.File, key: []const u8) !f32 {
        const v = file.getFloat(key) orelse return Error.InvalidConfig;
        return @floatCast(v);
    }

    fn metaFloatOpt(file: *const gguf.File, key: []const u8) ?f32 {
        const v = file.getFloat(key) orelse return null;
        return @floatCast(v);
    }
};

/// One expert's projections, loaded from per-expert sub-views of the stacked
/// 3-D GGUF banks so every weight dtype rides the existing LinearWeight arms.
const Expert = struct {
    gate_up: GateUp,
    down: LinearWeight,

    fn deinit(self: *Expert) void {
        self.down.deinit();
        self.gate_up.deinit();
        self.* = undefined;
    }
};

const MoeFfn = struct {
    router: LinearWeight, // [n_expert + n_shexp, hidden]; tail rows = shared sinks
    probs_bias: []f32, // selection-only bias over the n_expert routed logits
    experts: []Expert,
    shared: []Expert,

    fn deinit(self: *MoeFfn, allocator: Allocator) void {
        for (self.shared) |*e| e.deinit();
        allocator.free(self.shared);
        for (self.experts) |*e| e.deinit();
        allocator.free(self.experts);
        allocator.free(self.probs_bias);
        self.router.deinit();
        self.* = undefined;
    }
};

const DenseFfn = struct {
    gate_up: GateUp,
    down: LinearWeight,

    fn deinit(self: *DenseFfn) void {
        self.down.deinit();
        self.gate_up.deinit();
        self.* = undefined;
    }
};

const Ffn = union(enum) {
    dense: DenseFfn,
    moe: MoeFfn,

    fn deinit(self: *Ffn, allocator: Allocator) void {
        switch (self.*) {
            .dense => |*d| d.deinit(),
            .moe => |*m| m.deinit(allocator),
        }
        self.* = undefined;
    }
};

/// Q/K/V/R projections: fused into one GEMM when the four tensors share a
/// weight format, separate otherwise (mixed dynamic quants).
const QkvrProj = union(enum) {
    fused: LinearWeight,
    separate: struct {
        q: LinearWeight,
        k: LinearWeight,
        v: LinearWeight,
        r: LinearWeight,
    },

    fn deinit(self: *QkvrProj) void {
        switch (self.*) {
            .fused => |*w| w.deinit(),
            .separate => |*sep| {
                sep.r.deinit();
                sep.v.deinit();
                sep.k.deinit();
                sep.q.deinit();
            },
        }
        self.* = undefined;
    }
};

/// Gate+up pair: fused when formats match (one GEMM, output row = [gate | up]).
const GateUp = union(enum) {
    fused: LinearWeight,
    separate: struct {
        gate: LinearWeight,
        up: LinearWeight,
    },

    fn deinit(self: *GateUp) void {
        switch (self.*) {
            .fused => |*w| w.deinit(),
            .separate => |*sep| {
                sep.up.deinit();
                sep.gate.deinit();
            },
        }
        self.* = undefined;
    }
};

const Layer = struct {
    attn_norm: []f32,
    qkvr_proj: QkvrProj,
    o_proj: LinearWeight,
    q_norm: []f32, // [head_dim]
    k_norm: []f32, // [head_dim]
    /// Rel-bias projection table, checkpoint orientation [d_rel][extent]
    /// (GGUF ne {extent, d_rel}), F32 by converter contract.
    rel_proj: []f32,
    /// Short-conv kernels, [channels][K] (GGUF ne {K, channels}), F32.
    sconv_k: []f32,
    sconv_v: []f32,
    sconv_attn: []f32,
    sconv_mlp: []f32,
    ffn_norm: []f32,
    gscale: f32, // per-layer FFN global scale (F32 [1] tensor)
    ffn: Ffn,

    fn deinit(self: *Layer, allocator: Allocator) void {
        self.ffn.deinit(allocator);
        allocator.free(self.ffn_norm);
        allocator.free(self.sconv_mlp);
        allocator.free(self.sconv_attn);
        allocator.free(self.sconv_v);
        allocator.free(self.sconv_k);
        allocator.free(self.rel_proj);
        allocator.free(self.k_norm);
        allocator.free(self.q_norm);
        self.o_proj.deinit();
        self.qkvr_proj.deinit();
        allocator.free(self.attn_norm);
        self.* = undefined;
    }
};

/// Host cache: per layer post-norm K and V `[capacity, kv_heads_i, head_dim]`
/// plus the four rolling short-conv input states `[K-1, width]` (sites:
/// k-proj, v-proj, attention output, FFN output). No truncate: the conv
/// states only roll forward (deviation D3).
pub const Cache = struct {
    allocator: Allocator,
    k: [][]f32,
    v: [][]f32,
    /// conv_state[layer * 4 + site], time-major rows of the last K-1 inputs.
    conv_state: [][]f32,
    len: usize = 0,
    capacity: usize,

    pub const Site = enum(usize) { k = 0, v = 1, attn = 2, mlp = 3 };

    pub fn init(allocator: Allocator, config: *const Config, capacity: usize) !Cache {
        const n = config.num_layers;
        const taps = config.shortconv_kernel;

        var k = try allocator.alloc([]f32, n);
        var k_built: usize = 0;
        errdefer {
            for (k[0..k_built]) |l| allocator.free(l);
            allocator.free(k);
        }
        for (0..n) |i| {
            k[i] = try allocator.alloc(f32, capacity * config.kv_heads[i] * config.head_dim);
            k_built += 1;
        }

        var v = try allocator.alloc([]f32, n);
        var v_built: usize = 0;
        errdefer {
            for (v[0..v_built]) |l| allocator.free(l);
            allocator.free(v);
        }
        for (0..n) |i| {
            v[i] = try allocator.alloc(f32, capacity * config.kv_heads[i] * config.head_dim);
            v_built += 1;
        }

        var conv = try allocator.alloc([]f32, n * 4);
        var c_built: usize = 0;
        errdefer {
            for (conv[0..c_built]) |s| allocator.free(s);
            allocator.free(conv);
        }
        for (0..n) |i| {
            const kvw = config.kv_heads[i] * config.head_dim;
            const widths = [4]usize{ kvw, kvw, config.hidden_size, config.hidden_size };
            for (widths) |w| {
                conv[c_built] = try allocator.alloc(f32, (taps - 1) * w);
                @memset(conv[c_built], 0);
                c_built += 1;
            }
        }

        return .{ .allocator = allocator, .k = k, .v = v, .conv_state = conv, .capacity = capacity };
    }

    pub fn deinit(self: *Cache) void {
        for (self.conv_state) |s| self.allocator.free(s);
        self.allocator.free(self.conv_state);
        for (self.v) |l| self.allocator.free(l);
        self.allocator.free(self.v);
        for (self.k) |l| self.allocator.free(l);
        self.allocator.free(self.k);
        self.* = undefined;
    }

    fn state(self: *Cache, layer_i: usize, site: Site) []f32 {
        return self.conv_state[layer_i * 4 + @intFromEnum(site)];
    }
};

pub const Model = struct {
    allocator: Allocator,
    config: Config,
    token_embedding: LinearWeight,
    embed_norm: []f32, // applied to token lookups (rms)
    output_norm: []f32,
    output: LinearWeight,
    layers: []Layer,

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8) !Model {
        var file = try gguf.File.loadMmapAuto(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFile(ctx, &file);
    }

    pub fn loadGgufFromFile(ctx: *ExecContext, file: *gguf.File) !Model {
        const allocator = ctx.allocator;
        var config = try Config.fromGguf(allocator, file);
        errdefer config.deinit(allocator);

        var token_embedding = try LinearWeight.load(ctx, try file.get("token_embd.weight"), config.vocab_size, config.hidden_size);
        errdefer token_embedding.deinit();
        const embed_norm = try hostVector(allocator, file, "token_embd_norm.weight", config.hidden_size);
        errdefer allocator.free(embed_norm);
        const output_norm = try hostVector(allocator, file, "output_norm.weight", config.hidden_size);
        errdefer allocator.free(output_norm);
        var output = try LinearWeight.load(ctx, try file.get("output.weight"), config.vocab_size, config.hidden_size);
        errdefer output.deinit();

        const layers = try allocator.alloc(Layer, config.num_layers);
        errdefer allocator.free(layers);
        var built: usize = 0;
        errdefer for (layers[0..built]) |*l| l.deinit(allocator);
        for (layers, 0..) |*layer, i| {
            layer.* = try loadLayer(ctx, file, &config, i);
            built += 1;
        }

        // Total coverage: every tensor in the file must have been resolved
        // by name above (docs/PORTING.md §5 loader discipline).
        const expected = expectedTensorCount(&config);
        if (file.tensors.len != expected) {
            std.log.err("inkling: file has {d} tensors, loader resolved {d}", .{ file.tensors.len, expected });
            return Error.InvalidWeightShape;
        }

        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = token_embedding,
            .embed_norm = embed_norm,
            .output_norm = output_norm,
            .output = output,
            .layers = layers,
        };
    }

    fn expectedTensorCount(config: *const Config) usize {
        var count: usize = 4; // token_embd, token_embd_norm, output_norm, output
        for (0..config.num_layers) |i| {
            count += 14; // attn_norm q k v r o q_norm k_norm rel_proj 4*sconv ffn_norm
            count += 1; // ffn_gscale
            // dense: gate/up/down; moe: gate_inp, probs bias, 3 expert banks,
            // 3 shared banks
            count += if (i < config.dense_layers) 3 else 8;
        }
        return count;
    }

    pub fn deinit(self: *Model) void {
        for (self.layers) |*l| l.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.output.deinit();
        self.allocator.free(self.output_norm);
        self.allocator.free(self.embed_norm);
        self.token_embedding.deinit();
        self.config.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn initCache(self: *const Model, capacity: usize) !Cache {
        return Cache.init(self.allocator, &self.config, capacity);
    }

    /// One input row: a text token id, or a pre-computed embedding row
    /// (multimodal towers emit final-normed rows the decoder consumes
    /// as-is — the token embedding norm must NOT reapply).
    pub const Row = union(enum) {
        token: usize,
        embd: []const f32,
    };

    /// Process `tokens` at positions [cache.len, cache.len + S) and return
    /// the LAST position's next-token logits `[vocab]` (caller frees). Only
    /// the final row runs the unembed. Padded vocab ids carry -inf. Rows are
    /// computed jointly but causally: row r attends to cache positions
    /// <= cache.len + r only, so batch prefill matches S=1 stepping.
    pub fn step(self: *const Model, ctx: *ExecContext, cache: *Cache, tokens: []const usize) ![]f32 {
        const allocator = ctx.allocator;
        const rows = try allocator.alloc(Row, tokens.len);
        defer allocator.free(rows);
        for (rows, tokens) |*r, t| r.* = .{ .token = t };
        return self.stepMixed(ctx, cache, rows);
    }

    /// `step` over mixed token/embedding rows (multimodal prompts).
    pub fn stepMixed(self: *const Model, ctx: *ExecContext, cache: *Cache, items: []const Row) ![]f32 {
        const cfg = &self.config;
        const allocator = ctx.allocator;
        if (items.len == 0) return Error.InvalidSequenceLength;
        if (cache.len + items.len > cache.capacity) return Error.KvCacheOverflow;

        const S = items.len;
        const H = cfg.hidden_size;
        const x = try allocator.alloc(f32, S * H);
        defer allocator.free(x);

        // Batch consecutive token rows through one embedding lookup.
        var ids_buf = try allocator.alloc(usize, S);
        defer allocator.free(ids_buf);
        var run_start: usize = 0;
        var r: usize = 0;
        while (r <= S) : (r += 1) {
            const is_token = r < S and items[r] == .token;
            if (is_token) {
                ids_buf[r - run_start] = items[r].token;
                continue;
            }
            if (r > run_start) {
                var emb = try self.token_embedding.getRowsAs(ctx, ids_buf[0 .. r - run_start], .embed);
                defer emb.deinit();
                const rows = try emb.dataConst();
                for (run_start..r) |ri| {
                    // Embedding norm applies to token lookups only.
                    rmsNormInto(x[ri * H ..][0..H], rows[(ri - run_start) * H ..][0..H], self.embed_norm, cfg.rms_norm_eps);
                }
            }
            if (r < S) {
                const row = items[r].embd;
                if (row.len != H) return Error.InvalidWeightShape;
                @memcpy(x[r * H ..][0..H], row);
            }
            run_start = r + 1;
        }

        for (self.layers, 0..) |*layer, layer_i| {
            try self.layerForward(ctx, cache, layer, layer_i, x, S, cache.len);
        }
        cache.len += S;

        // Final norm, muP logit scale, unembed — last row only.
        const normed = try allocator.alloc(f32, H);
        defer allocator.free(normed);
        rmsNormInto(normed, x[(S - 1) * H ..][0..H], self.output_norm, cfg.rms_norm_eps);
        for (normed) |*nv| nv.* *= cfg.logit_scale;
        var normed_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, H }, normed);
        defer normed_t.deinit();
        var logits_t = try self.output.linearSeq(ctx, &normed_t, .embed, .vocab);
        defer logits_t.deinit();

        const row = try allocator.dupe(f32, try logits_t.dataConst());
        if (cfg.unpadded_vocab_size > 0 and cfg.unpadded_vocab_size < cfg.vocab_size) {
            for (row[cfg.unpadded_vocab_size..]) |*l| l.* = -std.math.inf(f32);
        }
        return row;
    }

    /// One layer over `x` rows in place.
    fn layerForward(self: *const Model, ctx: *ExecContext, cache: *Cache, layer: *const Layer, layer_i: usize, x: []f32, S: usize, pos0: usize) !void {
        const cfg = &self.config;
        const allocator = ctx.allocator;
        const H = cfg.hidden_size;
        const hd = cfg.head_dim;
        const n_head = cfg.num_heads;
        const kv_heads = cfg.kv_heads[layer_i];
        const heads_per_kv = n_head / kv_heads;
        const q_width = n_head * hd;
        const kv_width = kv_heads * hd;
        const is_swa = cfg.is_swa[layer_i];
        const extent = cfg.relExtent(layer_i);
        const taps = cfg.shortconv_kernel;

        // ---- attention block: h += sconv_attn(attn(norm(h))) ----
        const h_norm = try allocator.alloc(f32, S * H);
        defer allocator.free(h_norm);
        for (0..S) |r| rmsNormInto(h_norm[r * H ..][0..H], x[r * H ..][0..H], layer.attn_norm, cfg.rms_norm_eps);
        var h_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ S, H }, h_norm);
        defer h_t.deinit();

        const r_width = n_head * cfg.d_rel;
        const q = try allocator.alloc(f32, S * q_width);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, S * kv_width);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, S * kv_width);
        defer allocator.free(v);
        const r_flat = try allocator.alloc(f32, S * r_width);
        defer allocator.free(r_flat);
        switch (layer.qkvr_proj) {
            .fused => |*w| {
                // One GEMM; each output row is [q | k | v | r].
                var f_t = try w.linearSeq(ctx, &h_t, .embed, .q);
                defer f_t.deinit();
                const rows = try f_t.dataConst();
                const total = q_width + 2 * kv_width + r_width;
                for (0..S) |ri| {
                    const row = rows[ri * total ..][0..total];
                    @memcpy(q[ri * q_width ..][0..q_width], row[0..q_width]);
                    @memcpy(k[ri * kv_width ..][0..kv_width], row[q_width..][0..kv_width]);
                    @memcpy(v[ri * kv_width ..][0..kv_width], row[q_width + kv_width ..][0..kv_width]);
                    @memcpy(r_flat[ri * r_width ..][0..r_width], row[q_width + 2 * kv_width ..][0..r_width]);
                }
            },
            .separate => |*sep| {
                var q_t = try sep.q.linearSeq(ctx, &h_t, .embed, .q);
                defer q_t.deinit();
                var k_t = try sep.k.linearSeq(ctx, &h_t, .embed, .k);
                defer k_t.deinit();
                var v_t = try sep.v.linearSeq(ctx, &h_t, .embed, .v);
                defer v_t.deinit();
                var r_t = try sep.r.linearSeq(ctx, &h_t, .embed, .q);
                defer r_t.deinit();
                @memcpy(q, try q_t.dataConst());
                @memcpy(k, try k_t.dataConst());
                @memcpy(v, try v_t.dataConst());
                @memcpy(r_flat, try r_t.dataConst());
            },
        }

        // Short convs on the flat K/V projections, before head split/norm.
        try sconvInPlace(allocator, k, S, kv_width, layer.sconv_k, taps, cache.state(layer_i, .k));
        try sconvInPlace(allocator, v, S, kv_width, layer.sconv_v, taps, cache.state(layer_i, .v));

        // Rel-bias logits: rel[s, h, e] = sum_d r[s, h, d] * rel_proj[d][e],
        // one [S*n_head, d_rel] x [d_rel, extent] GEMM over the fused r rows.
        var r2 = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ S * n_head, cfg.d_rel }, r_flat);
        defer r2.deinit();
        var proj_t = try fucina.Tensor(.{ .embed, .attn }).fromSlice(ctx, .{ cfg.d_rel, extent }, layer.rel_proj);
        defer proj_t.deinit();
        var rel_t = try r2.dot(ctx, &proj_t, .embed);
        defer rel_t.deinit();
        const rel = try allocator.dupe(f32, try rel_t.dataConst());
        defer allocator.free(rel);

        // Per-head q/k rms norm; log-N tau on global layers (q and rel).
        for (0..S) |r| {
            const pos = pos0 + r;
            const q_row = q[r * q_width ..][0..q_width];
            for (0..n_head) |h| {
                const head = q_row[h * hd ..][0..hd];
                rmsNormInto(head, head, layer.q_norm, cfg.rms_norm_eps);
            }
            const k_row = k[r * kv_width ..][0..kv_width];
            for (0..kv_heads) |h| {
                const head = k_row[h * hd ..][0..hd];
                rmsNormInto(head, head, layer.k_norm, cfg.rms_norm_eps);
            }
            if (!is_swa and cfg.log_n_floor > 0) {
                const eff = @as(f32, @floatFromInt(pos + 1)) / @as(f32, @floatFromInt(cfg.log_n_floor));
                const tau = 1.0 + cfg.log_alpha * @log(@max(eff, 1.0));
                for (q_row) |*qv| qv.* *= tau;
                for (rel[r * n_head * extent ..][0 .. n_head * extent]) |*rv| rv.* *= tau;
            }

            // Append post-norm K and V for this position.
            @memcpy(cache.k[layer_i][pos * kv_width ..][0..kv_width], k_row);
            @memcpy(cache.v[layer_i][pos * kv_width ..][0..kv_width], v[r * kv_width ..][0..kv_width]);
        }

        // Scores + softmax + weighted V, per row and head. Visibility:
        // causal, and for SWA layers pos_q - pos_k < sliding_window; the
        // rel bias covers deltas in [0, extent). All rows' K/V are already
        // appended, so rows are independent — fan out over the hot team
        // (per-row arithmetic identical to the serial loop: bitwise-safe
        // scheduling-only parallelism).
        const attn_out = try allocator.alloc(f32, S * q_width);
        defer allocator.free(attn_out);
        const inv_d: f32 = 1.0 / @as(f32, @floatFromInt(hd));

        const max_t_len = pos0 + S;
        const scores_all = try allocator.alloc(f32, S * max_t_len);
        defer allocator.free(scores_all);

        const tasks = try allocator.alloc(AttnRowTask, S);
        defer allocator.free(tasks);
        for (tasks, 0..) |*task, r| {
            task.* = .{
                .k_cache = cache.k[layer_i],
                .v_cache = cache.v[layer_i],
                .q = q,
                .rel = rel,
                .attn_out = attn_out,
                .scores = scores_all[r * max_t_len ..][0..max_t_len],
                .r = r,
                .pos0 = pos0,
                .n_head = n_head,
                .hd = hd,
                .kv_heads = kv_heads,
                .heads_per_kv = heads_per_kv,
                .extent = extent,
                .window = if (is_swa) cfg.sliding_window else 0,
                .inv_d = inv_d,
            };
        }
        if (ctx.workPool()) |pool| {
            pool.parallelChunks(AttnRowTask, tasks, AttnRowTask.run);
        } else {
            for (tasks) |*t| AttnRowTask.run(t);
        }

        var attn_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ S, q_width }, attn_out);
        defer attn_t.deinit();
        var o_t = try layer.o_proj.linearSeq(ctx, &attn_t, .embed, .attn);
        defer o_t.deinit();
        const o = try allocator.dupe(f32, try o_t.dataConst());
        defer allocator.free(o);
        try sconvInPlace(allocator, o, S, H, layer.sconv_attn, taps, cache.state(layer_i, .attn));
        for (x, o) |*xi, oi| xi.* += oi;

        // ---- FFN block: h += sconv_mlp(ffn(norm(h))) ----
        for (0..S) |r| rmsNormInto(h_norm[r * H ..][0..H], x[r * H ..][0..H], layer.ffn_norm, cfg.rms_norm_eps);
        const ffn_out = try allocator.alloc(f32, S * H);
        defer allocator.free(ffn_out);
        switch (layer.ffn) {
            .dense => |*dense| {
                var f_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ S, H }, h_norm);
                defer f_t.deinit();
                const y = try swigluLinear(ctx, allocator, &f_t, &dense.gate_up, &dense.down, cfg.dense_ffn_size);
                defer allocator.free(y);
                for (ffn_out, y) |*fo, yi| fo.* = yi * layer.gscale;
            },
            .moe => |*moe| {
                try self.moeForwardBatch(ctx, allocator, moe, layer.gscale, h_norm, ffn_out, S);
            },
        }
        try sconvInPlace(allocator, ffn_out, S, H, layer.sconv_mlp, taps, cache.state(layer_i, .mlp));
        for (x, ffn_out) |*xi, fi| xi.* += fi;
    }

    /// Batched MoE over all S rows: one router GEMM, rows grouped per
    /// routed expert into one GEMM each, shared experts run across every
    /// row. Selection/weight semantics run per row on the host.
    fn moeForwardBatch(self: *const Model, ctx: *ExecContext, allocator: Allocator, moe: *const MoeFfn, gscale: f32, h_norm: []const f32, ffn_out: []f32, S: usize) !void {
        const cfg = &self.config;
        const H = cfg.hidden_size;
        const n_exp = cfg.num_experts;
        const n_used = cfg.num_experts_used;
        const n_shared = cfg.num_shared_experts;
        const fe = cfg.expert_ffn_size;

        var in_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ S, H }, h_norm);
        defer in_t.deinit();
        var logits_t = try moe.router.linearSeq(ctx, &in_t, .embed, .expert);
        defer logits_t.deinit();
        const logits = try logits_t.dataConst(); // [S, n_exp + n_shared]
        const n_logit = n_exp + n_shared;

        // Per-row selection + mixture weights (verbatim host control path).
        const selected = try allocator.alloc(usize, S * n_used);
        defer allocator.free(selected);
        const mix_w = try allocator.alloc(f32, S * (n_used + n_shared));
        defer allocator.free(mix_w);
        var scores_buf: [512]f32 = undefined;
        std.debug.assert(n_exp <= scores_buf.len);
        for (0..S) |r| {
            const row_logits = logits[r * n_logit ..][0..n_logit];
            const sel_scores = scores_buf[0..n_exp];
            for (sel_scores, row_logits[0..n_exp], moe.probs_bias) |*s, l, b| {
                s.* = 1.0 / (1.0 + @exp(-l)) + b;
            }
            const sel = selected[r * n_used ..][0..n_used];
            for (0..n_used) |slot| {
                var best: usize = 0;
                var best_s: f32 = -std.math.inf(f32);
                for (sel_scores, 0..) |s, e| {
                    if (s > best_s) { // strict: equal scores keep the lower id
                        best_s = s;
                        best = e;
                    }
                }
                sel_scores[best] = -std.math.inf(f32);
                sel[slot] = best;
            }
            const w = mix_w[r * (n_used + n_shared) ..][0 .. n_used + n_shared];
            for (0..n_used) |i| w[i] = logsigmoid(row_logits[sel[i]]);
            for (0..n_shared) |s| w[n_used + s] = logsigmoid(row_logits[n_exp + s]);
            softmaxInPlace(w);
            for (w) |*wv| wv.* *= cfg.expert_weights_scale * gscale;
        }

        @memset(ffn_out[0 .. S * H], 0);

        // Routed experts: gather assigned rows per expert, run one GEMM
        // chain, scatter the weighted outputs back (weights apply after
        // the down projection, like the reference).
        const count = try allocator.alloc(usize, n_exp);
        defer allocator.free(count);
        @memset(count, 0);
        for (0..S) |r| {
            for (selected[r * n_used ..][0..n_used]) |e| count[e] += 1;
        }
        const gather = try allocator.alloc(f32, S * n_used * H);
        defer allocator.free(gather);
        const gather_rows = try allocator.alloc(usize, S * n_used);
        defer allocator.free(gather_rows);
        const gather_w = try allocator.alloc(f32, S * n_used);
        defer allocator.free(gather_w);

        for (0..n_exp) |e| {
            if (count[e] == 0) continue;
            var n_rows: usize = 0;
            for (0..S) |r| {
                const sel = selected[r * n_used ..][0..n_used];
                for (sel, 0..) |se, slot| {
                    if (se != e) continue;
                    @memcpy(gather[n_rows * H ..][0..H], h_norm[r * H ..][0..H]);
                    gather_rows[n_rows] = r;
                    gather_w[n_rows] = mix_w[r * (n_used + n_shared) + slot];
                    n_rows += 1;
                }
            }
            var g_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ n_rows, H }, gather[0 .. n_rows * H]);
            defer g_t.deinit();
            const ex = &moe.experts[e];
            const y = try swigluLinear(ctx, allocator, &g_t, &ex.gate_up, &ex.down, fe);
            defer allocator.free(y);
            for (0..n_rows) |i| {
                vecAxpy(ffn_out[gather_rows[i] * H ..][0..H], gather_w[i], y[i * H ..][0..H]);
            }
        }

        // Shared experts: every row, batched; gamma scales the swiglu
        // activation BEFORE the down projection (D1).
        for (0..n_shared) |s| {
            const ex = &moe.shared[s];
            const hbuf = try allocator.alloc(f32, S * fe);
            defer allocator.free(hbuf);
            switch (ex.gate_up) {
                .fused => |*w| {
                    var gu_t = try w.linearSeq(ctx, &in_t, .embed, .gate_up);
                    defer gu_t.deinit();
                    const rows = try gu_t.dataConst();
                    for (0..S) |r| {
                        const gamma = mix_w[r * (n_used + n_shared) + n_used + s];
                        const dst = hbuf[r * fe ..][0..fe];
                        const row = rows[r * 2 * fe ..][0 .. 2 * fe];
                        for (dst, row[0..fe], row[fe..]) |*hv, gvv, uvv| hv.* = silu(gvv) * uvv * gamma;
                    }
                },
                .separate => |*sep| {
                    var gate_t = try sep.gate.linearSeq(ctx, &in_t, .embed, .gate_up);
                    defer gate_t.deinit();
                    var up_t = try sep.up.linearSeq(ctx, &in_t, .embed, .gate_up);
                    defer up_t.deinit();
                    const gv = try gate_t.dataConst();
                    const uv = try up_t.dataConst();
                    for (0..S) |r| {
                        const gamma = mix_w[r * (n_used + n_shared) + n_used + s];
                        const dst = hbuf[r * fe ..][0..fe];
                        for (dst, gv[r * fe ..][0..fe], uv[r * fe ..][0..fe]) |*hv, gvv, uvv| hv.* = silu(gvv) * uvv * gamma;
                    }
                },
            }
            var h_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ S, fe }, hbuf);
            defer h_t.deinit();
            var down_t = try ex.down.linearSeq(ctx, &h_t, .embed, .attn);
            defer down_t.deinit();
            const dv = try down_t.dataConst();
            for (0..S * H) |i| ffn_out[i] += dv[i];
        }
    }

};

/// Depthwise causal short conv with built-in residual, in place over
/// time-major rows `[S, width]`: y[t,c] = x[t,c] + sum_j w[j][c] *
/// xin[t-(K-1)+j, c], where negative times read the rolling state (last
/// K-1 input rows of the previous step; zeros on a fresh sequence).
/// `kernel` is TAP-MAJOR [taps][width] (transposed at load) so the channel
/// loop runs 8-lane SIMD. Updates `state` to the last K-1 input rows.
/// Mirrors ggml_ssm_conv plus the reference graph's `x + conv(x)` residual.
fn sconvInPlace(allocator: Allocator, x: []f32, S: usize, width: usize, kernel: []const f32, taps: usize, state: []f32) !void {
    const d_conv = taps - 1;
    std.debug.assert(kernel.len == width * taps);
    std.debug.assert(state.len == d_conv * width);

    // Save the input rows the NEXT step will need before overwriting x.
    const next_state = try allocator.alloc(f32, d_conv * width);
    defer allocator.free(next_state);
    for (0..d_conv) |r| {
        const dst = next_state[r * width ..][0..width];
        // Row index in the virtual [state | x] stream, from the tail.
        const virt = S + r; // rows S-d_conv+r of x when S >= d_conv
        if (virt >= d_conv) {
            const xi = virt - d_conv;
            @memcpy(dst, x[xi * width ..][0..width]);
        } else {
            @memcpy(dst, state[virt * width ..][0..width]);
        }
    }

    var t: usize = S;
    while (t > 0) {
        t -= 1;
        const y_row = x[t * width ..][0..width];
        var c: usize = 0;
        while (c + 8 <= width) : (c += 8) {
            var acc: Vf = @as(Vf, y_row[c..][0..8].*);
            for (0..taps) |j| {
                // time = t - (K-1) + j; negative reads the rolling state.
                const virt = t + j; // vs offset d_conv in the [state | x] stream
                const src = if (virt >= d_conv)
                    x[(virt - d_conv) * width + c ..][0..8]
                else
                    state[virt * width + c ..][0..8];
                const wv: Vf = kernel[j * width + c ..][0..8].*;
                acc += wv * @as(Vf, src.*);
            }
            y_row[c..][0..8].* = acc;
        }
        while (c < width) : (c += 1) {
            var acc: f32 = 0;
            for (0..taps) |j| {
                const virt = t + j;
                const xv = if (virt >= d_conv)
                    x[(virt - d_conv) * width + c]
                else
                    state[virt * width + c];
                acc += kernel[j * width + c] * xv;
            }
            y_row[c] += acc;
        }
    }
    @memcpy(state, next_state);
}

fn layerName(buf: []u8, layer_i: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer_i, suffix });
}

/// Per-expert 2-D sub-view of a stacked 3-D expert bank (ne {in, out, E}):
/// expert e's block is contiguous, so a shifted TensorInfo suffices.
fn expertInfo(bank: *const gguf.TensorInfo, e: usize) !gguf.TensorInfo {
    if (bank.n_dims != 3) return Error.InvalidWeightShape;
    const n_expert = bank.dims[2];
    if (e >= n_expert or bank.data.len % n_expert != 0) return Error.InvalidWeightShape;
    const bytes_per = bank.data.len / n_expert;
    var info = bank.*;
    info.n_dims = 2;
    info.dims[2] = 1;
    info.data = bank.data[e * bytes_per ..][0..bytes_per];
    return info;
}

/// Fused-or-separate loader for a gate/up projection pair (fuseLinear
/// deinits its parts on success; on decline the parts remain usable and
/// become the separate arm).
fn loadGateUp(ctx: *ExecContext, gate_info: *const gguf.TensorInfo, up_info: *const gguf.TensorInfo, out_dim: usize, in_dim: usize) !GateUp {
    var gate = try LinearWeight.loadForFusion(ctx, gate_info, out_dim, in_dim);
    errdefer gate.deinit();
    var up = try LinearWeight.loadForFusion(ctx, up_info, out_dim, in_dim);
    errdefer up.deinit();
    var parts = [_]*LinearWeight{ &gate, &up };
    if (try weights.fuseLinear(ctx, &parts)) |fused| {
        return .{ .fused = fused };
    }
    return .{ .separate = .{ .gate = gate, .up = up } };
}

fn loadQkvr(ctx: *ExecContext, file: *const gguf.File, layer_i: usize, q_dim: usize, kv_dim: usize, r_dim: usize, hidden: usize) !QkvrProj {
    var name_buf: [96]u8 = undefined;
    var q = try LinearWeight.loadForFusion(ctx, try file.get(try layerName(&name_buf, layer_i, "attn_q.weight")), q_dim, hidden);
    errdefer q.deinit();
    var k = try LinearWeight.loadForFusion(ctx, try file.get(try layerName(&name_buf, layer_i, "attn_k.weight")), kv_dim, hidden);
    errdefer k.deinit();
    var v = try LinearWeight.loadForFusion(ctx, try file.get(try layerName(&name_buf, layer_i, "attn_v.weight")), kv_dim, hidden);
    errdefer v.deinit();
    var r = try LinearWeight.loadForFusion(ctx, try file.get(try layerName(&name_buf, layer_i, "attn_r.weight")), r_dim, hidden);
    errdefer r.deinit();
    var parts = [_]*LinearWeight{ &q, &k, &v, &r };
    if (try weights.fuseLinear(ctx, &parts)) |fused| {
        return .{ .fused = fused };
    }
    return .{ .separate = .{ .q = q, .k = k, .v = v, .r = r } };
}

/// Load a gate+up bank pair per expert, fusing each expert's two
/// projections into one GEMM when formats match.
fn loadExpertGateUp(ctx: *ExecContext, gate_bank: *const gguf.TensorInfo, up_bank: *const gguf.TensorInfo, n_expert: usize, out_dim: usize, in_dim: usize, experts: []Expert) !void {
    for (0..n_expert) |e| {
        const g_info = try expertInfo(gate_bank, e);
        const u_info = try expertInfo(up_bank, e);
        experts[e].gate_up = try loadGateUp(ctx, &g_info, &u_info, out_dim, in_dim);
    }
}

fn loadExpertDown(ctx: *ExecContext, bank: *const gguf.TensorInfo, n_expert: usize, out_dim: usize, in_dim: usize, experts: []Expert) !void {
    for (0..n_expert) |e| {
        const info = try expertInfo(bank, e);
        experts[e].down = try LinearWeight.load(ctx, &info, out_dim, in_dim);
    }
}

fn loadLayer(ctx: *ExecContext, file: *const gguf.File, config: *const Config, layer_i: usize) !Layer {
    const allocator = ctx.allocator;
    var name_buf: [96]u8 = undefined;

    const H = config.hidden_size;
    const q_dim = config.num_heads * config.head_dim;
    const kv_dim = config.kv_heads[layer_i] * config.head_dim;
    const r_dim = config.num_heads * config.d_rel;
    const extent = config.relExtent(layer_i);
    const taps = config.shortconv_kernel;

    const attn_norm = try hostVector(allocator, file, try layerName(&name_buf, layer_i, "attn_norm.weight"), H);
    errdefer allocator.free(attn_norm);

    var qkvr_proj = try loadQkvr(ctx, file, layer_i, q_dim, kv_dim, r_dim, H);
    errdefer qkvr_proj.deinit();
    var o_proj = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "attn_output.weight")), H, q_dim);
    errdefer o_proj.deinit();

    const q_norm = try hostVector(allocator, file, try layerName(&name_buf, layer_i, "attn_q_norm.weight"), config.head_dim);
    errdefer allocator.free(q_norm);
    const k_norm = try hostVector(allocator, file, try layerName(&name_buf, layer_i, "attn_k_norm.weight"), config.head_dim);
    errdefer allocator.free(k_norm);

    const rel_proj = try hostMatrixF32(allocator, file, try layerName(&name_buf, layer_i, "attn_rel_proj.weight"), extent, config.d_rel);
    errdefer allocator.free(rel_proj);

    const sconv_k = try hostConvKernel(allocator, file, try layerName(&name_buf, layer_i, "shortconv_k.weight"), taps, kv_dim);
    errdefer allocator.free(sconv_k);
    const sconv_v = try hostConvKernel(allocator, file, try layerName(&name_buf, layer_i, "shortconv_v.weight"), taps, kv_dim);
    errdefer allocator.free(sconv_v);
    const sconv_attn = try hostConvKernel(allocator, file, try layerName(&name_buf, layer_i, "shortconv_attn.weight"), taps, H);
    errdefer allocator.free(sconv_attn);
    const sconv_mlp = try hostConvKernel(allocator, file, try layerName(&name_buf, layer_i, "shortconv_mlp.weight"), taps, H);
    errdefer allocator.free(sconv_mlp);

    const ffn_norm = try hostVector(allocator, file, try layerName(&name_buf, layer_i, "ffn_norm.weight"), H);
    errdefer allocator.free(ffn_norm);
    const gscale_v = try hostVector(allocator, file, try layerName(&name_buf, layer_i, "ffn_gscale.weight"), 1);
    defer allocator.free(gscale_v);
    const gscale = gscale_v[0];

    var ffn: Ffn = undefined;
    if (layer_i < config.dense_layers) {
        var gate_up = try loadGateUp(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_gate.weight")), try file.get(try layerName(&name_buf, layer_i, "ffn_up.weight")), config.dense_ffn_size, H);
        errdefer gate_up.deinit();
        var down = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_down.weight")), H, config.dense_ffn_size);
        errdefer down.deinit();
        ffn = .{ .dense = .{ .gate_up = gate_up, .down = down } };
    } else {
        const n_exp = config.num_experts;
        const n_shared = config.num_shared_experts;
        const fe = config.expert_ffn_size;

        // Router holds n_expert + n_shared rows; the tail rows produce the
        // shared-expert sink logits.
        var router = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_gate_inp.weight")), n_exp + n_shared, H);
        errdefer router.deinit();
        const probs_bias = try hostVector(allocator, file, try layerName(&name_buf, layer_i, "exp_probs_b.bias"), n_exp);
        errdefer allocator.free(probs_bias);

        const experts = try allocator.alloc(Expert, n_exp);
        errdefer allocator.free(experts);
        try loadExpertGateUp(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_gate_exps.weight")), try file.get(try layerName(&name_buf, layer_i, "ffn_up_exps.weight")), n_exp, fe, H, experts);
        try loadExpertDown(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_down_exps.weight")), n_exp, H, fe, experts);

        const shared = try allocator.alloc(Expert, n_shared);
        errdefer allocator.free(shared);
        try loadExpertGateUp(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_gate_shexp.weight")), try file.get(try layerName(&name_buf, layer_i, "ffn_up_shexp.weight")), n_shared, fe, H, shared);
        try loadExpertDown(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_down_shexp.weight")), n_shared, H, fe, shared);

        ffn = .{ .moe = .{ .router = router, .probs_bias = probs_bias, .experts = experts, .shared = shared } };
    }

    return .{
        .attn_norm = attn_norm,
        .qkvr_proj = qkvr_proj,
        .o_proj = o_proj,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .rel_proj = rel_proj,
        .sconv_k = sconv_k,
        .sconv_v = sconv_v,
        .sconv_attn = sconv_attn,
        .sconv_mlp = sconv_mlp,
        .ffn_norm = ffn_norm,
        .gscale = gscale,
        .ffn = ffn,
    };
}

fn swigluLinear(ctx: *ExecContext, allocator: Allocator, x: *const fucina.Tensor(.{ .seq, .embed }), gate_up: *const GateUp, down: *const LinearWeight, width: usize) ![]f32 {
    const rows = x.dim(.seq);
    const g = try allocator.alloc(f32, rows * width);
    defer allocator.free(g);
    switch (gate_up.*) {
        .fused => |*w| {
            var gu_t = try w.linearSeq(ctx, x, .embed, .gate_up);
            defer gu_t.deinit();
            const flat = try gu_t.dataConst();
            for (0..rows) |r| {
                const row = flat[r * 2 * width ..][0 .. 2 * width];
                const dst = g[r * width ..][0..width];
                for (dst, row[0..width], row[width..]) |*gi, gv, uv| gi.* = silu(gv) * uv;
            }
        },
        .separate => |*sep| {
            var gate_t = try sep.gate.linearSeq(ctx, x, .embed, .gate_up);
            defer gate_t.deinit();
            var up_t = try sep.up.linearSeq(ctx, x, .embed, .gate_up);
            defer up_t.deinit();
            for (g, try gate_t.dataConst(), try up_t.dataConst()) |*gi, gv, uv| gi.* = silu(gv) * uv;
        },
    }
    var g_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ rows, width }, g);
    defer g_t.deinit();
    var down_t = try down.linearSeq(ctx, &g_t, .embed, .attn);
    defer down_t.deinit();
    return allocator.dupe(f32, try down_t.dataConst());
}

fn hostVector(allocator: Allocator, file: *const gguf.File, tensor_name: []const u8, expected: usize) ![]f32 {
    const info = try file.get(tensor_name);
    if (info.n_dims != 1 or info.dims[0] != expected) return Error.InvalidWeightShape;
    const out = try allocator.alloc(f32, expected);
    errdefer allocator.free(out);
    try weights.fillF32(out, info);
    return out;
}

/// Load a 2-D F32 tensor (ne {d0, d1}) as a host row-major [d1][d0] slice.
fn hostMatrixF32(allocator: Allocator, file: *const gguf.File, tensor_name: []const u8, d0: usize, d1: usize) ![]f32 {
    const info = try file.get(tensor_name);
    if (info.n_dims != 2 or info.dims[0] != d0 or info.dims[1] != d1) return Error.InvalidWeightShape;
    const out = try allocator.alloc(f32, d0 * d1);
    errdefer allocator.free(out);
    try weights.fillF32(out, info);
    return out;
}

/// Load a shortconv kernel (GGUF ne {taps, width} = channel-major [C][K])
/// transposed to tap-major [K][C] for the SIMD conv loop.
fn hostConvKernel(allocator: Allocator, file: *const gguf.File, tensor_name: []const u8, taps: usize, width: usize) ![]f32 {
    const cw = try hostMatrixF32(allocator, file, tensor_name, taps, width);
    defer allocator.free(cw);
    const out = try allocator.alloc(f32, taps * width);
    errdefer allocator.free(out);
    for (0..width) |c| {
        for (0..taps) |j| out[j * width + c] = cw[c * taps + j];
    }
    return out;
}

/// One query row's attention (all heads): scores + softmax + weighted V.
/// Runs on the hot team; every field is read-only shared state except the
/// row's own attn_out and scores slices.
const AttnRowTask = struct {
    k_cache: []const f32,
    v_cache: []const f32,
    q: []const f32,
    rel: []const f32,
    attn_out: []f32,
    scores: []f32,
    r: usize,
    pos0: usize,
    n_head: usize,
    hd: usize,
    kv_heads: usize,
    heads_per_kv: usize,
    extent: usize,
    window: usize, // 0 = full causal
    inv_d: f32,

    fn run(t: *const AttnRowTask) void {
        const pos = t.pos0 + t.r;
        const t_first = if (t.window > 0 and pos + 1 > t.window) pos + 1 - t.window else 0;
        const q_width = t.n_head * t.hd;
        const q_row = t.q[t.r * q_width ..][0..q_width];
        for (0..t.n_head) |h| {
            const q_head = q_row[h * t.hd ..][0..t.hd];
            const kv_h = h / t.heads_per_kv;
            const rel_row = t.rel[(t.r * t.n_head + h) * t.extent ..][0..t.extent];
            for (t_first..pos + 1) |ti| {
                const kt = t.k_cache[(ti * t.kv_heads + kv_h) * t.hd ..][0..t.hd];
                var s = vecDot(q_head, kt) * t.inv_d;
                const delta = pos - ti;
                if (delta < t.extent) s += rel_row[delta];
                t.scores[ti - t_first] = s;
            }
            const t_len = pos + 1 - t_first;
            softmaxInPlace(t.scores[0..t_len]);
            const out_head = t.attn_out[t.r * q_width + h * t.hd ..][0..t.hd];
            @memset(out_head, 0);
            for (0..t_len) |i| {
                const vt = t.v_cache[((t_first + i) * t.kv_heads + kv_h) * t.hd ..][0..t.hd];
                vecAxpy(out_head, t.scores[i], vt);
            }
        }
    }
};

const Vf = @Vector(8, f32);

/// 8-lane SIMD dot with scalar tail (portable @Vector; reassociates the
/// f32 sum — covered by the tolerance-tier gates, generation stays exact).
fn vecDot(a: []const f32, b: []const f32) f32 {
    var acc: Vf = @splat(0);
    var i: usize = 0;
    while (i + 8 <= a.len) : (i += 8) {
        acc += @as(Vf, a[i..][0..8].*) * @as(Vf, b[i..][0..8].*);
    }
    var s = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

/// acc += w * v, 8-lane SIMD with scalar tail.
fn vecAxpy(acc: []f32, w: f32, v: []const f32) void {
    const wv: Vf = @splat(w);
    var i: usize = 0;
    while (i + 8 <= acc.len) : (i += 8) {
        const r: Vf = @as(Vf, acc[i..][0..8].*) + wv * @as(Vf, v[i..][0..8].*);
        acc[i..][0..8].* = r;
    }
    while (i < acc.len) : (i += 1) acc[i] += w * v[i];
}

fn rmsNormInto(out: []f32, x: []const f32, weight: []const f32, eps: f32) void {
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, v) * v;
    const inv = 1.0 / @sqrt(sum / @as(f64, @floatFromInt(x.len)) + eps);
    for (out, x, weight) |*o, v, w| o.* = @floatCast(@as(f64, v) * inv * w);
}

fn softmaxInPlace(v: []f32) void {
    var max: f32 = -std.math.inf(f32);
    for (v) |x| max = @max(max, x);
    var sum: f32 = 0;
    for (v) |*x| {
        x.* = @exp(x.* - max);
        sum += x.*;
    }
    for (v) |*x| x.* /= sum;
}

/// logsigmoid(x) = -softplus(-x) = -log(1 + exp(-x)).
fn logsigmoid(x: f32) f32 {
    return -std.math.log1p(@exp(-x));
}

fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

const Self = @This();

/// Internal hooks for model_tests.zig only.
pub const testing = struct {
    pub const sconvInPlace = Self.sconvInPlace;
    pub const logsigmoid = Self.logsigmoid;
};

test {
    _ = @import("model_tests.zig");
}
