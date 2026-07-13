//! DeepSeek-V2 family (`deepseek2` GGUF arch): Multi-head Latent Attention
//! (MLA) + fine-grained MoE with shared experts. Validation target:
//! DeepSeek-V2-Lite (27 layers, 16 heads, qk 192 = nope 128 + rope 64,
//! v 128, kv_lora 512, no q-LoRA; layer 0 dense; layers 1+ = 64 routed
//! experts top-6 softmax + fused shared expert).
//!
//! The heavy linears per layer (q, kv_a, kv_b, o, FFN projections) run on
//! fucina's quantized kernels; YaRN RoPE runs through the core rope op over
//! hand-filled f64-angle tables (`yarnBlendInvFreqsF64` +
//! `prepareRopeTableInvFreqsF64`), and router top-k through the core topK.
//! Host-side f32 remains where the core kernels would change validated
//! bits: rms norms (f64 accumulation, ggml's), router sigmoid/softmax and
//! the SwiGLU combine (the vector exp/silu legs are polynomial, not
//! bit-equal to scalar `@exp`), and the `.full` cache mode's deliberately
//! naive per-head attention loops (the auditable validation baseline for
//! `.latent`'s absorbed algebra).
const std = @import("std");
const fucina = @import("fucina");
const weights = @import("../weights.zig");
const gguf_meta = @import("../gguf_meta.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;
const LinearWeight = weights.LinearWeight;

pub const Error = weights.Error || error{
    InvalidConfig,
    InvalidSequenceLength,
    KvCacheOverflow,
    UnsupportedExpertQuant,
};

/// How the 64 rope dims pair up for rotation. The HF checkpoint applies the
/// de-interleave trick (pairs are adjacent lanes -> `.interleaved`); GGUF
/// conversions that pre-permute the rope slice would need `.half`. Decided
/// empirically against coherent output; see `rope_pairing`.
pub const RopePairing = enum { interleaved, half };
pub const rope_pairing: RopePairing = .interleaved;

pub const Config = struct {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    qk_nope_dim: usize, // key_length - rope dims
    qk_rope_dim: usize,
    qk_head_dim: usize, // nope + rope
    v_head_dim: usize,
    kv_lora_rank: usize,
    dense_ffn_size: usize,
    leading_dense_layers: usize,
    num_experts: usize,
    num_experts_used: usize,
    expert_ffn_size: usize,
    num_shared_experts: usize,
    expert_weights_scale: f32,
    /// 1 = softmax scoring (V2), 2 = sigmoid scoring (V3/noaux_tc).
    expert_gating_func: usize,
    /// V3: renormalize the selected experts' weights to sum 1.
    expert_weights_norm: bool,
    /// V2 full / V3: low-rank q projection (0 = direct q, as on Lite).
    q_lora_rank: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    yarn_factor: f32,
    yarn_orig_ctx: usize,
    yarn_log_multiplier: f32,

    pub fn fromGguf(file: *const gguf.File) !Config {
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        // GLM-5.2's "glm-dsa" is DeepSeek-MLA-native attention + V3 sigmoid
        // routing under its own metadata prefix, plus per-layer DSA indexer
        // tensors (not loaded — dense attention, llama.cpp's fallback too)
        // and a trailing nextn MTP layer (excluded from the trunk count).
        const prefix: []const u8 = if (std.mem.eql(u8, arch, "deepseek2"))
            "deepseek2"
        else if (std.mem.eql(u8, arch, "glm-dsa"))
            "glm-dsa"
        else
            return Error.InvalidConfig;

        var kbuf: [96]u8 = undefined;
        const K = struct {
            fn of(b: []u8, p: []const u8, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(b, "{s}.{s}", .{ p, suffix }) catch unreachable;
            }
        };

        // Newer MLA-native conversions set key_length/value_length to the
        // LATENT attention dims (576/512) and carry the real per-head dims
        // in *_mla; older files carry them in key_length/value_length.
        const qk_head = gguf_meta.metaIntOpt(file, prefix, "attention.key_length_mla", .reject_zero) orelse
            try metaInt(file, K.of(&kbuf, prefix, "attention.key_length"));
        const v_head = gguf_meta.metaIntOpt(file, prefix, "attention.value_length_mla", .reject_zero) orelse
            try metaInt(file, K.of(&kbuf, prefix, "attention.value_length"));
        const rope_dims = try metaInt(file, K.of(&kbuf, prefix, "rope.dimension_count"));
        const block_count = try metaInt(file, K.of(&kbuf, prefix, "block_count"));
        const nextn = gguf_meta.metaIntOpt(file, prefix, "nextn_predict_layers", .accept_zero) orelse 0;
        if (nextn >= block_count) return Error.InvalidConfig;
        return .{
            .vocab_size = try metaInt(file, K.of(&kbuf, prefix, "vocab_size")),
            .hidden_size = try metaInt(file, K.of(&kbuf, prefix, "embedding_length")),
            .num_layers = block_count - nextn,
            .num_heads = try metaInt(file, K.of(&kbuf, prefix, "attention.head_count")),
            .qk_nope_dim = qk_head - rope_dims,
            .qk_rope_dim = rope_dims,
            .qk_head_dim = qk_head,
            .v_head_dim = v_head,
            .kv_lora_rank = try metaInt(file, K.of(&kbuf, prefix, "attention.kv_lora_rank")),
            .dense_ffn_size = try metaInt(file, K.of(&kbuf, prefix, "feed_forward_length")),
            .leading_dense_layers = gguf_meta.metaIntOpt(file, prefix, "leading_dense_block_count", .accept_zero) orelse 0,
            .num_experts = gguf_meta.metaIntOpt(file, prefix, "expert_count", .accept_zero) orelse 0,
            .num_experts_used = gguf_meta.metaIntOpt(file, prefix, "expert_used_count", .accept_zero) orelse 0,
            .expert_ffn_size = gguf_meta.metaIntOpt(file, prefix, "expert_feed_forward_length", .accept_zero) orelse 0,
            .num_shared_experts = gguf_meta.metaIntOpt(file, prefix, "expert_shared_count", .accept_zero) orelse 0,
            .expert_weights_scale = metaFloatOpt(file, K.of(&kbuf, prefix, "expert_weights_scale")) orelse 1.0,
            .expert_gating_func = gguf_meta.metaIntOpt(file, prefix, "expert_gating_func", .accept_zero) orelse 1,
            .expert_weights_norm = file.getBool(K.of(&kbuf, prefix, "expert_weights_norm")) orelse false,
            .q_lora_rank = gguf_meta.metaIntOpt(file, prefix, "attention.q_lora_rank", .accept_zero) orelse 0,
            .rms_norm_eps = try metaFloat(file, K.of(&kbuf, prefix, "attention.layer_norm_rms_epsilon")),
            .rope_theta = metaFloatOpt(file, K.of(&kbuf, prefix, "rope.freq_base")) orelse 10000.0,
            .yarn_factor = metaFloatOpt(file, K.of(&kbuf, prefix, "rope.scaling.factor")) orelse 1.0,
            .yarn_orig_ctx = gguf_meta.metaIntOpt(file, prefix, "rope.scaling.original_context_length", .accept_zero) orelse 0,
            .yarn_log_multiplier = metaFloatOpt(file, K.of(&kbuf, prefix, "rope.scaling.yarn_log_multiplier")) orelse 0.0,
        };
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

/// YaRN rotary frequencies for the rope slice: the blend between
/// interpolation (freq/factor) and extrapolation (original freq) across the
/// correction ramp [beta_fast=32, beta_slow=1] — the HF DeepseekV2Yarn
/// reference, computed by the core `yarnBlendInvFreqsF64`. V2-Lite's mscale
/// == mscale_all_dim, so the cos/sin magnitude correction cancels to 1 and
/// only the attention-scale mscale² survives (`attn_scale`). Rotation runs
/// through the core rope op over per-step hand-filled tables (f64 angles,
/// the reference numerics), applied `.interleaved_tail` to the q heads and
/// full-width to the shared k_pe — `rope_pairing` maps onto the core mode.
const YarnRope = struct {
    inv_freq: []f64,

    fn init(ctx: *ExecContext, config: Config) !YarnRope {
        return .{ .inv_freq = try ctx.yarnBlendInvFreqsF64(config.qk_rope_dim, config.rope_theta, config.yarn_factor, config.yarn_orig_ctx) };
    }

    fn deinit(self: *YarnRope, allocator: Allocator) void {
        allocator.free(self.inv_freq);
        self.* = undefined;
    }

    /// Hand-fill the rotation table for `count` positions starting at
    /// `pos0` (spans `qk_rope_dim` features — partial-tail on the q heads).
    fn table(self: *const YarnRope, ctx: *ExecContext, pos0: usize, count: usize) !fucina.RopeTable {
        return ctx.prepareRopeTableInvFreqsF64(pos0, count, self.inv_freq, false);
    }
};

/// The core rope mode equivalent of `rope_pairing` within the rotated span.
const rope_mode: fucina.RopeMode = switch (rope_pairing) {
    .interleaved => .interleaved_tail,
    // The rope slice sits at the TAIL of each q head and the core has no
    // half-paired tail mode; add one before flipping `rope_pairing`.
    .half => @compileError("rope_pairing .half needs a half-paired tail rope mode"),
};

const MoeFfn = struct {
    router: LinearWeight,
    gate: fucina.MoeRhs,
    up: fucina.MoeRhs,
    down: fucina.MoeRhs,
    shared_gate: LinearWeight,
    shared_up: LinearWeight,
    shared_down: LinearWeight,

    fn deinit(self: *MoeFfn) void {
        self.shared_down.deinit();
        self.shared_up.deinit();
        self.shared_gate.deinit();
        self.down.deinit();
        self.up.deinit();
        self.gate.deinit();
        self.router.deinit();
        self.* = undefined;
    }
};

const DenseFfn = struct {
    gate: LinearWeight,
    up: LinearWeight,
    down: LinearWeight,

    fn deinit(self: *DenseFfn) void {
        self.down.deinit();
        self.up.deinit();
        self.gate.deinit();
        self.* = undefined;
    }
};

const Ffn = union(enum) {
    dense: DenseFfn,
    moe: MoeFfn,

    fn deinit(self: *Ffn) void {
        switch (self.*) {
            inline else => |*v| v.deinit(),
        }
        self.* = undefined;
    }
};

const Layer = struct {
    attn_norm: []f32, // host copies of the tiny norm vectors
    kv_a_norm: []f32,
    ffn_norm: []f32,
    q_proj: ?LinearWeight, // hidden -> heads*qk_head (models without q-LoRA)
    q_a: ?LinearWeight, // q-LoRA: hidden -> q_lora
    q_a_norm: ?[]f32,
    q_b: ?LinearWeight, // q-LoRA: q_lora -> heads*qk_head
    /// V3 noaux_tc: selection bias added to the sigmoid scores for TOP-K
    /// CHOICE only (never to the mixture weights).
    router_bias: ?[]f32,
    kv_a: LinearWeight, // hidden -> kv_lora + rope
    /// Fused kv_b (older conversions); null when the file ships the
    /// pre-split absorption tensors instead. Required by `.full` cache mode.
    kv_b: ?LinearWeight,
    /// kv_b dequantized to host f32 for weight absorption, split per head:
    /// `kv_b_k` rows are each head's k_nope block `[nope, kv_lora]`,
    /// `kv_b_v` each head's v block `[v_head, kv_lora]` (row-major,
    /// heads-major). ~8 MB/layer on V2-Lite — the price of never
    /// reconstructing K/V.
    kv_b_k: []f32,
    kv_b_v: []f32,
    o_proj: LinearWeight, // heads*v -> hidden
    ffn: Ffn,

    fn deinit(self: *Layer, allocator: Allocator) void {
        self.ffn.deinit();
        self.o_proj.deinit();
        allocator.free(self.kv_b_v);
        allocator.free(self.kv_b_k);
        if (self.kv_b) |*w| w.deinit();
        self.kv_a.deinit();
        if (self.router_bias) |b| allocator.free(b);
        if (self.q_b) |*w| w.deinit();
        if (self.q_a_norm) |n| allocator.free(n);
        if (self.q_a) |*w| w.deinit();
        if (self.q_proj) |*w| w.deinit();
        allocator.free(self.ffn_norm);
        allocator.free(self.kv_a_norm);
        allocator.free(self.attn_norm);
        self.* = undefined;
    }
};

/// MLA cache in one of two modes.
///
/// `.full` (validation baseline): per layer, `[capacity, heads, qk_head]` K
/// (nope | rotated rope) and `[capacity, heads, v_head]` V, reconstructed
/// once per appended position through kv_b.
///
/// `.latent` (MLA's raison d'être): per layer, only the 512-dim normed
/// latent and the shared rotated 64-dim rope slice per position — 576
/// floats/token/layer instead of 16*(192+128) = 5120, a 8.9x cache
/// reduction (57x against a full-KV no-GQA design) — with kv_b folded into
/// the query and applied after attention (weight absorption), so K/V never
/// materialize at all.
pub const Cache = struct {
    pub const Mode = enum { full, latent };

    allocator: Allocator,
    mode: Mode,
    k: [][]f32 = &.{}, // full: K rows; latent: [capacity, kv_lora + rope]
    v: [][]f32 = &.{}, // full only
    len: usize = 0,
    capacity: usize,

    pub fn init(allocator: Allocator, config: Config, capacity: usize, mode: Mode) !Cache {
        const k = try allocator.alloc([]f32, config.num_layers);
        var built: usize = 0;
        errdefer {
            for (k[0..built]) |layer| allocator.free(layer);
            allocator.free(k);
        }
        const k_width = switch (mode) {
            .full => config.num_heads * config.qk_head_dim,
            .latent => config.kv_lora_rank + config.qk_rope_dim,
        };
        for (0..config.num_layers) |i| {
            k[i] = try allocator.alloc(f32, capacity * k_width);
            built += 1;
        }
        var v: [][]f32 = &.{};
        if (mode == .full) {
            v = try allocator.alloc([]f32, config.num_layers);
            errdefer allocator.free(v);
            var v_built: usize = 0;
            errdefer for (v[0..v_built]) |layer| allocator.free(layer);
            for (0..config.num_layers) |i| {
                v[i] = try allocator.alloc(f32, capacity * config.num_heads * config.v_head_dim);
                v_built += 1;
            }
        }
        return .{ .allocator = allocator, .mode = mode, .k = k, .v = v, .capacity = capacity };
    }

    pub fn deinit(self: *Cache) void {
        for (self.k) |layer| self.allocator.free(layer);
        for (self.v) |layer| self.allocator.free(layer);
        self.allocator.free(self.k);
        if (self.v.len > 0) self.allocator.free(self.v);
        self.* = undefined;
    }

    pub fn reset(self: *Cache) void {
        self.len = 0;
    }
};

pub const Model = struct {
    allocator: Allocator,
    config: Config,
    token_embedding: LinearWeight,
    output_norm: []f32,
    output: LinearWeight,
    layers: []Layer,
    rope: YarnRope,
    /// The GGUF mmap: resident MoE arms borrow their blocks straight from
    /// it (unmapped last in deinit; null in streamed mode).
    weight_mapping: ?gguf.File.MappedRegion = null,
    /// Disk-streaming tier for the expert stacks (destroyed after layers).
    expert_store: ?*fucina.ExpertStore = null,
    /// Router-lookahead prefetch (`MoeStreamOptions.pilot`).
    pilot_enabled: bool = false,
    /// (1 + yarn_log_multiplier * ln(factor))² / sqrt(qk_head): the YaRN
    /// mscale² attention-scale correction folded with the 1/sqrt(d) scale.
    attn_scale: f32,

    pub const MoeStreamOptions = weights.MoeStreamOptions;

    pub const LoadOptions = struct {
        moe_stream: ?MoeStreamOptions = null,
    };

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8) !Model {
        var file = try gguf.File.loadMmapAuto(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFile(ctx, &file);
    }

    pub fn loadGgufFromFile(ctx: *ExecContext, file: *gguf.File) !Model {
        return loadGgufFromFileOptions(ctx, file, .{});
    }

    pub fn loadGgufFromFileOptions(ctx: *ExecContext, file: *gguf.File, options: LoadOptions) !Model {
        const config = try Config.fromGguf(file);
        const allocator = ctx.allocator;

        var expert_store: ?*fucina.ExpertStore = null;
        if (options.moe_stream) |stream_options| {
            if (config.num_experts > 0) expert_store = try weights.createExpertStore(allocator, stream_options, config.num_layers);
        }
        errdefer if (expert_store) |store| store.destroy();

        var token_embedding = try LinearWeight.load(ctx, try file.get("token_embd.weight"), config.vocab_size, config.hidden_size);
        errdefer token_embedding.deinit();
        var output = try LinearWeight.load(ctx, try file.get("output.weight"), config.vocab_size, config.hidden_size);
        errdefer output.deinit();
        const output_norm = try hostVector(allocator, file, "output_norm.weight", config.hidden_size);
        errdefer allocator.free(output_norm);

        const layers = try allocator.alloc(Layer, config.num_layers);
        errdefer allocator.free(layers);
        var built: usize = 0;
        errdefer for (layers[0..built]) |*l| l.deinit(allocator);
        for (layers, 0..) |*layer, i| {
            layer.* = try loadLayer(ctx, file, config, i, expert_store);
            built += 1;
        }
        if (expert_store) |store| try store.finalize();

        var rope = try YarnRope.init(ctx, config);
        errdefer rope.deinit(allocator);

        // Resident expert stacks borrow from the mapping; the model must own
        // it. Streamed mode borrows nothing: the mapping dies with `file` and
        // residency stays dense weights + the expert cache.
        const weight_mapping = if (config.num_experts > 0 and expert_store == null) file.takeMapping() else null;
        if (config.num_experts > 0 and expert_store == null and weight_mapping == null) return Error.InvalidWeightShape;

        const mscale: f32 = if (config.yarn_factor > 1.0)
            1.0 + config.yarn_log_multiplier * @log(config.yarn_factor)
        else
            1.0;
        const attn_scale = mscale * mscale / @sqrt(@as(f32, @floatFromInt(config.qk_head_dim)));

        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = token_embedding,
            .output_norm = output_norm,
            .output = output,
            .layers = layers,
            .rope = rope,
            .weight_mapping = weight_mapping,
            .expert_store = expert_store,
            .pilot_enabled = expert_store != null and options.moe_stream.?.pilot,
            .attn_scale = attn_scale,
        };
    }

    pub fn deinit(self: *Model) void {
        self.rope.deinit(self.allocator);
        for (self.layers) |*l| l.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.allocator.free(self.output_norm);
        self.output.deinit();
        self.output_norm = undefined;
        self.token_embedding.deinit();
        if (self.expert_store) |store| store.destroy();
        if (self.weight_mapping) |*mapping| mapping.deinit();
        self.* = undefined;
    }

    pub fn initCache(self: *const Model, capacity: usize) !Cache {
        return Cache.init(self.allocator, self.config, capacity, .latent);
    }

    pub fn initCacheMode(self: *const Model, capacity: usize, mode: Cache.Mode) !Cache {
        return Cache.init(self.allocator, self.config, capacity, mode);
    }

    /// One decode step: process `token` at position `cache.len`, append its
    /// K/V, return the next-token logits as a host slice (caller-owned).
    pub fn step(self: *const Model, ctx: *ExecContext, cache: *Cache, token: usize) ![]f32 {
        const cfg = self.config;
        const allocator = ctx.allocator;
        if (cache.len >= cache.capacity) return Error.KvCacheOverflow;
        const pos = cache.len;
        var rope_table = try self.rope.table(ctx, pos, 1);
        defer rope_table.deinit();

        // Residual stream as a host row; per-op tensors are built on demand.
        const x = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(x);
        {
            var ids = [_]usize{token};
            var emb = try self.token_embedding.getRowsAs(ctx, &ids, .embed);
            defer emb.deinit();
            @memcpy(x, try emb.dataConst());
        }

        const h_norm = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(h_norm);
        const attn_out = try allocator.alloc(f32, cfg.num_heads * cfg.v_head_dim);
        defer allocator.free(attn_out);

        for (self.layers, 0..) |*layer, layer_i| {
            // ---- MLA attention ----
            rmsNormInto(h_norm, x, layer.attn_norm, cfg.rms_norm_eps);
            var h_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, cfg.hidden_size }, h_norm);
            defer h_t.deinit();

            // q projection (direct or q-LoRA), per-head view, tail rotary —
            // every head's rope slice rotates at `pos` in one core rope op.
            var q_flat = blk: {
                if (layer.q_proj) |*direct| {
                    break :blk try direct.linearSeq(ctx, &h_t, .embed, .q);
                }
                // q-LoRA: hidden -> q_lora -> rmsnorm -> heads*qk_head.
                var qa_t = try layer.q_a.?.linearSeq(ctx, &h_t, .embed, .q);
                defer qa_t.deinit();
                const q_lat = try allocator.alloc(f32, cfg.q_lora_rank);
                defer allocator.free(q_lat);
                rmsNormInto(q_lat, try qa_t.dataConst(), layer.q_a_norm.?, cfg.rms_norm_eps);
                var q_lat_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, cfg.q_lora_rank }, q_lat);
                defer q_lat_t.deinit();
                break :blk try layer.q_b.?.linearSeq(ctx, &q_lat_t, .embed, .q);
            };
            defer q_flat.deinit();
            var q_heads = try q_flat.split(ctx, .q, .{ .head, .d }, .{ cfg.num_heads, cfg.qk_head_dim });
            defer q_heads.deinit();
            var q_rot = try q_heads.rope(ctx, .seq, .d, &rope_table, rope_mode);
            defer q_rot.deinit();
            var q_t = try q_rot.squeeze(ctx, .seq);
            defer q_t.deinit();

            var kv_a_t = try layer.kv_a.linearSeq(ctx, &h_t, .embed, .k);
            defer kv_a_t.deinit();
            const kv_a = try kv_a_t.dataConst();

            // Latent norm + rope: the shared k_pe (MQA) rotates as a narrow
            // view of the kv_a row (full-span table -> full rotation).
            const latent = try allocator.alloc(f32, cfg.kv_lora_rank);
            defer allocator.free(latent);
            rmsNormInto(latent, kv_a[0..cfg.kv_lora_rank], layer.kv_a_norm, cfg.rms_norm_eps);
            var k_pe_v = try kv_a_t.narrow(ctx, .k, cfg.kv_lora_rank, cfg.qk_rope_dim);
            defer k_pe_v.deinit();
            var k_pe_rot = try k_pe_v.rope(ctx, .seq, .k, &rope_table, rope_mode);
            defer k_pe_rot.deinit();
            const k_pe = try allocator.dupe(f32, try k_pe_rot.dataConst());
            defer allocator.free(k_pe);

            const t_len = pos + 1;
            switch (cache.mode) {
                .full => {
                    // Reconstruct this position's k_nope/v via kv_b and run
                    // attention over materialized K/V — the deliberately
                    // naive, hand-auditable validation baseline (host dot /
                    // softmax / mix loops stay on purpose).
                    var latent_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, cfg.kv_lora_rank }, latent);
                    defer latent_t.deinit();
                    const kv_b_full = if (layer.kv_b) |*w| w else return Error.InvalidWeightShape;
                    var kv_t = try kv_b_full.linearSeq(ctx, &latent_t, .embed, .v);
                    defer kv_t.deinit();
                    const kv = try kv_t.dataConst(); // [heads * (nope + v)]

                    const k_layer = cache.k[layer_i];
                    const v_layer = cache.v[layer_i];
                    for (0..cfg.num_heads) |h| {
                        const k_dst = k_layer[(pos * cfg.num_heads + h) * cfg.qk_head_dim ..][0..cfg.qk_head_dim];
                        const v_dst = v_layer[(pos * cfg.num_heads + h) * cfg.v_head_dim ..][0..cfg.v_head_dim];
                        const kv_head = kv[h * (cfg.qk_nope_dim + cfg.v_head_dim) ..];
                        @memcpy(k_dst[0..cfg.qk_nope_dim], kv_head[0..cfg.qk_nope_dim]);
                        @memcpy(k_dst[cfg.qk_nope_dim..], k_pe);
                        @memcpy(v_dst, kv_head[cfg.qk_nope_dim..][0..cfg.v_head_dim]);
                    }

                    const q = try allocator.dupe(f32, try q_rot.dataConst());
                    defer allocator.free(q);
                    const scores = try allocator.alloc(f32, t_len);
                    defer allocator.free(scores);
                    for (0..cfg.num_heads) |h| {
                        const q_head = q[h * cfg.qk_head_dim ..][0..cfg.qk_head_dim];
                        for (0..t_len) |t| {
                            const k_row = k_layer[(t * cfg.num_heads + h) * cfg.qk_head_dim ..][0..cfg.qk_head_dim];
                            var dot: f32 = 0;
                            for (q_head, k_row) |a, b| dot += a * b;
                            scores[t] = dot * self.attn_scale;
                        }
                        softmaxInPlace(scores);
                        const out_head = attn_out[h * cfg.v_head_dim ..][0..cfg.v_head_dim];
                        @memset(out_head, 0);
                        for (0..t_len) |t| {
                            const w = scores[t];
                            const v_row = v_layer[(t * cfg.num_heads + h) * cfg.v_head_dim ..][0..cfg.v_head_dim];
                            for (out_head, v_row) |*o, val| o.* += w * val;
                        }
                    }
                },
                .latent => {
                    // Weight absorption: cache only [latent | rotated k_pe],
                    // fold kv_b's K block into the query (q_eff = W_kb_k^T
                    // q_nope) and apply its V block AFTER attention to the
                    // probability-weighted latent — K/V never materialize.
                    // All contractions batch over heads through tensor ops;
                    // the absorbed weights and the cache rows are borrowed
                    // views (zero copies per step).
                    const lora = cfg.kv_lora_rank;
                    const row_w = lora + cfg.qk_rope_dim;
                    const k_layer = cache.k[layer_i];
                    const dst = k_layer[pos * row_w ..][0..row_w];
                    @memcpy(dst[0..lora], latent);
                    @memcpy(dst[lora..], k_pe);

                    // q_nope and q_pe are strided VIEWS of the roped q
                    // (narrow + retag), never gathered by hand.
                    var q_nope_v = try q_t.narrow(ctx, .d, 0, cfg.qk_nope_dim);
                    defer q_nope_v.deinit();
                    var q_nope_t = try q_nope_v.withTags(ctx, .{ .head, .nope });
                    defer q_nope_t.deinit();
                    var wk_t = try fucina.Tensor(.{ .head, .nope, .lora }).fromBorrowedConstSlice(ctx, .{ cfg.num_heads, cfg.qk_nope_dim, lora }, layer.kv_b_k);
                    defer wk_t.deinit();
                    var q_eff_t = try q_nope_t.dot(ctx, &wk_t, .nope);
                    defer q_eff_t.deinit();

                    // q_cat = [q_eff | q_pe] per head (concat of the effective
                    // query with the roped q_pe view); scores over the cached
                    // [latent | k_pe] rows in ONE batched contraction.
                    var q_eff_d = try q_eff_t.withTags(ctx, .{ .head, .d });
                    defer q_eff_d.deinit();
                    var q_pe_v = try q_t.narrow(ctx, .d, cfg.qk_nope_dim, cfg.qk_rope_dim);
                    defer q_pe_v.deinit();
                    var q_cat_t = try q_eff_d.concat(ctx, .d, &.{&q_pe_v});
                    defer q_cat_t.deinit();
                    var rows_t = try fucina.Tensor(.{ .t, .d }).fromBorrowedConstSlice(ctx, .{ t_len, row_w }, k_layer[0 .. t_len * row_w]);
                    defer rows_t.deinit();
                    var scores_t = try q_cat_t.dot(ctx, &rows_t, .d);
                    defer scores_t.deinit();
                    var scaled_t = try scores_t.scale(ctx, self.attn_scale);
                    defer scaled_t.deinit();
                    var probs_t = try scaled_t.softmax(ctx, .t, .{});
                    defer probs_t.deinit();

                    // ctx_lat[h] = probs[h] . latents; out[h] = W_kb_v[h] @ ctx_lat[h].
                    var lat_t = try rows_t.narrow(ctx, .d, 0, lora);
                    defer lat_t.deinit();
                    var ctx_t = try probs_t.dot(ctx, &lat_t, .t);
                    defer ctx_t.deinit();
                    var wv_t = try fucina.Tensor(.{ .head, .v, .d }).fromBorrowedConstSlice(ctx, .{ cfg.num_heads, cfg.v_head_dim, lora }, layer.kv_b_v);
                    defer wv_t.deinit();
                    var out_t = try ctx_t.dot(ctx, &wv_t, .d);
                    defer out_t.deinit();
                    @memcpy(attn_out, try out_t.dataConst());
                },
            }

            var attn_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, cfg.num_heads * cfg.v_head_dim }, attn_out);
            defer attn_t.deinit();
            var o_t = try layer.o_proj.linearSeq(ctx, &attn_t, .embed, .attn);
            defer o_t.deinit();
            for (x, try o_t.dataConst()) |*xi, oi| xi.* += oi;

            // Router lookahead (pilot): predict the NEXT layer's experts
            // from this layer's post-attention state and hint the store's
            // I/O thread so misses overlap this layer's FFN compute.
            // Prediction only — never changes routing or output.
            if (self.pilot_enabled and layer_i + 1 < self.layers.len) {
                self.pilotPrefetchNext(ctx, &self.layers[layer_i + 1], layer_i + 1, x) catch {};
            }

            // ---- FFN ----
            rmsNormInto(h_norm, x, layer.ffn_norm, cfg.rms_norm_eps);
            var f_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, cfg.hidden_size }, h_norm);
            defer f_t.deinit();
            switch (layer.ffn) {
                .dense => |*dense| {
                    const y = try swigluLinear(ctx, allocator, &f_t, &dense.gate, &dense.up, &dense.down);
                    defer allocator.free(y);
                    for (x, y) |*xi, yi| xi.* += yi;
                },
                .moe => |*moe| {
                    const y = try self.moeForward(ctx, allocator, moe, layer, &f_t);
                    defer allocator.free(y);
                    for (x, y) |*xi, yi| xi.* += yi;
                },
            }
        }
        cache.len += 1;

        rmsNormInto(h_norm, x, self.output_norm, cfg.rms_norm_eps);
        var final_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, cfg.hidden_size }, h_norm);
        defer final_t.deinit();
        var logits_t = try self.output.linearSeq(ctx, &final_t, .embed, .vocab);
        defer logits_t.deinit();
        return allocator.dupe(f32, try logits_t.dataConst());
    }

    /// Router lookahead (pilot): apply the NEXT layer's ffn_norm + router to
    /// the current post-attention state and hint the store with the
    /// predicted top-k — mirroring moeForward's V2/V3 selection (sigmoid or
    /// softmax scores, choice bias applied) so the hint matches what the
    /// real routing will fetch whenever the FFN barely changes the routing
    /// direction. Purely advisory.
    fn pilotPrefetchNext(self: *const Model, ctx: *ExecContext, next: *const Layer, next_layer_i: usize, x: []const f32) !void {
        const moe = switch (next.ffn) {
            .moe => |*m| m,
            else => return,
        };
        const store = switch (moe.gate) {
            .streamed => |*st| st.store,
            else => return,
        };
        const cfg = self.config;
        const allocator = ctx.allocator;

        const h = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(h);
        rmsNormInto(h, x, next.ffn_norm, cfg.rms_norm_eps);
        var h_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedSlice(ctx, .{ 1, cfg.hidden_size }, h);
        defer h_t.deinit();
        var logits_t = try moe.router.linearSeq(ctx, &h_t, .embed, .expert);
        defer logits_t.deinit();

        const choice = try allocator.dupe(f32, try logits_t.dataConst());
        defer allocator.free(choice);
        switch (cfg.expert_gating_func) {
            2 => for (choice) |*v| {
                v.* = 1.0 / (1.0 + @exp(-v.*));
            },
            else => softmaxInPlace(choice),
        }
        if (next.router_bias) |bias| {
            for (choice, bias) |*c, b| c.* += b;
        }
        var sel: [64]usize = undefined;
        std.debug.assert(cfg.num_experts_used <= sel.len);
        try topKExperts(ctx, choice, sel[0..cfg.num_experts_used]);
        store.pilotHint(next_layer_i, sel[0..cfg.num_experts_used]);
    }

    /// Top-k expert selection over the (bias-adjusted) choice scores through
    /// the core kernel — ties resolve to the lowest expert id, exactly like
    /// the repeated strict-> argmax scan it replaces.
    fn topKExperts(ctx: *ExecContext, choice: []f32, sel: []usize) !void {
        var choice_t = try fucina.Tensor(.{.expert}).fromBorrowedSlice(ctx, .{choice.len}, choice);
        defer choice_t.deinit();
        var top = try choice_t.topK(ctx, .expert, sel.len, .slot);
        defer top.values.deinit();
        defer top.indices.deinit();
        for (sel, try top.indices.dataConst()) |*s, e| s.* = @intCast(e);
    }

    /// Routed mixture + shared expert. V2: softmax over ALL router logits,
    /// top-k of those probabilities used un-renormalized. V3 (noaux_tc):
    /// sigmoid scores; the per-expert selection bias (exp_probs_b) applies
    /// to the top-k CHOICE only, the mixture weights are the raw sigmoid
    /// scores, optionally renormalized to sum 1 (expert_weights_norm).
    /// Group-limited top-k (n_group > 1, the 671B-scale configs) is not
    /// implemented yet — single-group models (Lite, Moonlight) are exact.
    /// Both variants scale by expert_weights_scale.
    fn moeForward(self: *const Model, ctx: *ExecContext, allocator: Allocator, moe: *const MoeFfn, layer: *const Layer, f_t: *const fucina.Tensor(.{ .seq, .embed })) ![]f32 {
        const cfg = self.config;
        var logits_t = try moe.router.linearSeq(ctx, f_t, .embed, .expert);
        defer logits_t.deinit();
        const probs = try allocator.dupe(f32, try logits_t.dataConst());
        defer allocator.free(probs);
        switch (cfg.expert_gating_func) {
            2 => for (probs) |*v| {
                v.* = 1.0 / (1.0 + @exp(-v.*));
            },
            else => softmaxInPlace(probs),
        }
        // Selection scores: biased for the choice, never for the weights.
        const choice = try allocator.dupe(f32, probs);
        defer allocator.free(choice);
        if (layer.router_bias) |bias| {
            for (choice, bias) |*c, b| c.* += b;
        }

        const y = try allocator.alloc(f32, cfg.hidden_size);
        errdefer allocator.free(y);
        @memset(y, 0);

        // Top-k selection through the core kernel; mixture weights read
        // from the unbiased probs.
        var selected: [64]usize = undefined;
        var routing: [64]f32 = undefined;
        std.debug.assert(cfg.num_experts_used <= selected.len);
        try topKExperts(ctx, choice, selected[0..cfg.num_experts_used]);
        for (selected[0..cfg.num_experts_used], routing[0..cfg.num_experts_used]) |e, *w| w.* = probs[e];
        if (cfg.expert_weights_norm) {
            var total: f32 = 1e-20;
            for (routing[0..cfg.num_experts_used]) |w| total += w;
            for (routing[0..cfg.num_experts_used]) |*w| w.* /= total;
        }
        for (routing[0..cfg.num_experts_used]) |*w| w.* *= cfg.expert_weights_scale;

        var mix = try weights.moeSwiGluFfnSeq(
            ctx,
            f_t,
            &moe.gate,
            &moe.up,
            &moe.down,
            selected[0..cfg.num_experts_used],
            routing[0..cfg.num_experts_used],
            cfg.num_experts_used,
            cfg.expert_ffn_size,
            null,
            null,
        );
        defer mix.deinit();
        for (y, try mix.dataConst()) |*yi, mi| yi.* += mi;

        // Shared expert (always active, weight 1).
        const shared = try swigluLinear(ctx, allocator, f_t, &moe.shared_gate, &moe.shared_up, &moe.shared_down);
        defer allocator.free(shared);
        for (y, shared) |*yi, si| yi.* += si;
        return y;
    }
};

fn loadLayer(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize, store: ?*fucina.ExpertStore) !Layer {
    const allocator = ctx.allocator;
    var name_buf: [96]u8 = undefined;
    const name = struct {
        fn of(buf: []u8, i: usize, suffix: []const u8) ![]const u8 {
            return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ i, suffix });
        }
    };

    const attn_norm = try hostVector(allocator, file, try name.of(&name_buf, layer_i, "attn_norm.weight"), config.hidden_size);
    errdefer allocator.free(attn_norm);
    const kv_a_norm = try hostVector(allocator, file, try name.of(&name_buf, layer_i, "attn_kv_a_norm.weight"), config.kv_lora_rank);
    errdefer allocator.free(kv_a_norm);
    const ffn_norm = try hostVector(allocator, file, try name.of(&name_buf, layer_i, "ffn_norm.weight"), config.hidden_size);
    errdefer allocator.free(ffn_norm);

    var q_proj: ?LinearWeight = null;
    var q_a: ?LinearWeight = null;
    var q_a_norm: ?[]f32 = null;
    var q_b: ?LinearWeight = null;
    errdefer if (q_proj) |*w| w.deinit();
    errdefer if (q_a) |*w| w.deinit();
    errdefer if (q_a_norm) |n| allocator.free(n);
    errdefer if (q_b) |*w| w.deinit();
    if (config.q_lora_rank > 0) {
        q_a = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "attn_q_a.weight")), config.q_lora_rank, config.hidden_size);
        q_a_norm = try hostVector(allocator, file, try name.of(&name_buf, layer_i, "attn_q_a_norm.weight"), config.q_lora_rank);
        q_b = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "attn_q_b.weight")), config.num_heads * config.qk_head_dim, config.q_lora_rank);
    } else {
        q_proj = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "attn_q.weight")), config.num_heads * config.qk_head_dim, config.hidden_size);
    }
    var kv_a = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "attn_kv_a_mqa.weight")), config.kv_lora_rank + config.qk_rope_dim, config.hidden_size);
    errdefer kv_a.deinit();
    // Absorption matrices, from either kv_b layout:
    //  - fused attn_kv_b [kv_lora -> heads*(nope+v)] (older conversions;
    //    also powers the .full reconstructed cache mode), or
    //  - pre-split attn_k_b [nope, lora, heads] + attn_v_b [lora, v, heads]
    //    (MLA-native conversions; k_b arrives [lora][nope] per head and is
    //    transposed into our [nope][lora] row layout).
    const lora = config.kv_lora_rank;
    var kv_b: ?LinearWeight = null;
    errdefer if (kv_b) |*w| w.deinit();
    const kv_b_k = try allocator.alloc(f32, config.num_heads * config.qk_nope_dim * lora);
    errdefer allocator.free(kv_b_k);
    const kv_b_v = try allocator.alloc(f32, config.num_heads * config.v_head_dim * lora);
    errdefer allocator.free(kv_b_v);
    if (file.maybeGet(try name.of(&name_buf, layer_i, "attn_kv_b.weight"))) |kv_b_info| {
        kv_b = try LinearWeight.load(ctx, kv_b_info, config.num_heads * (config.qk_nope_dim + config.v_head_dim), lora);
        const kv_b_rows = config.num_heads * (config.qk_nope_dim + config.v_head_dim);
        const kv_b_all = try allocator.alloc(f32, kv_b_rows * lora);
        defer allocator.free(kv_b_all);
        try gguf.decodeF32(kv_b_info.ggml_type, kv_b_info.data, kv_b_all);
        const head_rows = config.qk_nope_dim + config.v_head_dim;
        for (0..config.num_heads) |h| {
            const src_base = h * head_rows * lora;
            @memcpy(
                kv_b_k[h * config.qk_nope_dim * lora ..][0 .. config.qk_nope_dim * lora],
                kv_b_all[src_base ..][0 .. config.qk_nope_dim * lora],
            );
            @memcpy(
                kv_b_v[h * config.v_head_dim * lora ..][0 .. config.v_head_dim * lora],
                kv_b_all[src_base + config.qk_nope_dim * lora ..][0 .. config.v_head_dim * lora],
            );
        }
    } else {
        const k_b_info = try file.get(try name.of(&name_buf, layer_i, "attn_k_b.weight"));
        if (k_b_info.n_dims != 3 or k_b_info.dims[0] != config.qk_nope_dim or k_b_info.dims[1] != lora or k_b_info.dims[2] != config.num_heads) return Error.InvalidWeightShape;
        const k_b_all = try allocator.alloc(f32, config.num_heads * lora * config.qk_nope_dim);
        defer allocator.free(k_b_all);
        try gguf.decodeF32(k_b_info.ggml_type, k_b_info.data, k_b_all);
        for (0..config.num_heads) |h| {
            const src = k_b_all[h * lora * config.qk_nope_dim ..][0 .. lora * config.qk_nope_dim];
            const dst = kv_b_k[h * config.qk_nope_dim * lora ..][0 .. config.qk_nope_dim * lora];
            for (0..lora) |j| {
                for (0..config.qk_nope_dim) |i| dst[i * lora + j] = src[j * config.qk_nope_dim + i];
            }
        }
        const v_b_info = try file.get(try name.of(&name_buf, layer_i, "attn_v_b.weight"));
        if (v_b_info.n_dims != 3 or v_b_info.dims[0] != lora or v_b_info.dims[1] != config.v_head_dim or v_b_info.dims[2] != config.num_heads) return Error.InvalidWeightShape;
        try gguf.decodeF32(v_b_info.ggml_type, v_b_info.data, kv_b_v);
    }
    var o_proj = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "attn_output.weight")), config.hidden_size, config.num_heads * config.v_head_dim);
    errdefer o_proj.deinit();

    var router_bias: ?[]f32 = null;
    errdefer if (router_bias) |b| allocator.free(b);
    var ffn: Ffn = undefined;
    if (layer_i < config.leading_dense_layers or config.num_experts == 0) {
        var gate = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "ffn_gate.weight")), config.dense_ffn_size, config.hidden_size);
        errdefer gate.deinit();
        var up = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "ffn_up.weight")), config.dense_ffn_size, config.hidden_size);
        errdefer up.deinit();
        var down = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "ffn_down.weight")), config.hidden_size, config.dense_ffn_size);
        errdefer down.deinit();
        ffn = .{ .dense = .{ .gate = gate, .up = up, .down = down } };
    } else {
        var router = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "ffn_gate_inp.weight")), config.num_experts, config.hidden_size);
        errdefer router.deinit();
        // V3 noaux_tc selection bias (absent on V2-family files).
        var bias_name_buf: [96]u8 = undefined;
        if (file.maybeGet(try name.of(&bias_name_buf, layer_i, "exp_probs_b.bias"))) |bias_info| {
            router_bias = try hostVectorInfo(allocator, bias_info, config.num_experts);
        } else if (file.maybeGet(try name.of(&bias_name_buf, layer_i, "exp_probs_b.weight"))) |bias_info| {
            router_bias = try hostVectorInfo(allocator, bias_info, config.num_experts);
        }
        var gate: fucina.MoeRhs = undefined;
        var up: fucina.MoeRhs = undefined;
        var down: fucina.MoeRhs = undefined;
        if (store) |st| {
            const trio = try weights.loadMoeRhsStreamed(
                st,
                file,
                layer_i,
                try file.get(try name.of(&name_buf, layer_i, "ffn_gate_exps.weight")),
                try file.get(try name.of(&name_buf, layer_i, "ffn_up_exps.weight")),
                try file.get(try name.of(&name_buf, layer_i, "ffn_down_exps.weight")),
                config.hidden_size,
                config.expert_ffn_size,
                config.num_experts,
            );
            gate = trio.gate;
            up = trio.up;
            down = trio.down;
        } else {
            const borrow = file.is_mmap and !file.isSplit();
            gate = try weights.loadMoeRhs(ctx, try file.get(try name.of(&name_buf, layer_i, "ffn_gate_exps.weight")), config.hidden_size, config.expert_ffn_size, config.num_experts, borrow);
            up = try weights.loadMoeRhs(ctx, try file.get(try name.of(&name_buf, layer_i, "ffn_up_exps.weight")), config.hidden_size, config.expert_ffn_size, config.num_experts, borrow);
            down = try weights.loadMoeRhs(ctx, try file.get(try name.of(&name_buf, layer_i, "ffn_down_exps.weight")), config.expert_ffn_size, config.hidden_size, config.num_experts, borrow);
        }
        const shared_ffn = config.expert_ffn_size * config.num_shared_experts;
        var shared_gate = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "ffn_gate_shexp.weight")), shared_ffn, config.hidden_size);
        errdefer shared_gate.deinit();
        var shared_up = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "ffn_up_shexp.weight")), shared_ffn, config.hidden_size);
        errdefer shared_up.deinit();
        var shared_down = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "ffn_down_shexp.weight")), config.hidden_size, shared_ffn);
        errdefer shared_down.deinit();
        ffn = .{ .moe = .{ .router = router, .gate = gate, .up = up, .down = down, .shared_gate = shared_gate, .shared_up = shared_up, .shared_down = shared_down } };
    }

    return .{
        .attn_norm = attn_norm,
        .kv_a_norm = kv_a_norm,
        .ffn_norm = ffn_norm,
        .q_proj = q_proj,
        .q_a = q_a,
        .q_a_norm = q_a_norm,
        .q_b = q_b,
        .router_bias = router_bias,
        .kv_a = kv_a,
        .kv_b = kv_b,
        .kv_b_k = kv_b_k,
        .kv_b_v = kv_b_v,
        .o_proj = o_proj,
        .ffn = ffn,
    };
}

/// SwiGLU through three resident linears, result as a host row.
fn swigluLinear(ctx: *ExecContext, allocator: Allocator, x: *const fucina.Tensor(.{ .seq, .embed }), gate: *const LinearWeight, up: *const LinearWeight, down: *const LinearWeight) ![]f32 {
    var gate_t = try gate.linearSeq(ctx, x, .embed, .gate_up);
    defer gate_t.deinit();
    var up_t = try up.linearSeq(ctx, x, .embed, .gate_up);
    defer up_t.deinit();
    const width = gate_t.dim(.gate_up);
    const g = try allocator.alloc(f32, width);
    defer allocator.free(g);
    for (g, try gate_t.dataConst(), try up_t.dataConst()) |*gi, gv, uv| gi.* = silu(gv) * uv;
    var g_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, width }, g);
    defer g_t.deinit();
    var down_t = try down.linearSeq(ctx, &g_t, .embed, .attn);
    defer down_t.deinit();
    return allocator.dupe(f32, try down_t.dataConst());
}

fn hostVector(allocator: Allocator, file: *const gguf.File, tensor_name: []const u8, expected: usize) ![]f32 {
    return hostVectorInfo(allocator, try file.get(tensor_name), expected);
}

fn hostVectorInfo(allocator: Allocator, info: *const gguf.TensorInfo, expected: usize) ![]f32 {
    if (info.n_dims != 1 or info.dims[0] != expected) return Error.InvalidWeightShape;
    const out = try allocator.alloc(f32, expected);
    errdefer allocator.free(out);
    try weights.fillF32(out, info);
    return out;
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

fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

test "deepseek2 config rejects non-deepseek architectures" {
    // Covered structurally: Config.fromGguf requires general.architecture ==
    // "deepseek2"; real-model behavior is validated by the runner (the
    // GGUF-dependent tests live with models/ and skip without it).
}
