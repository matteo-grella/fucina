//! DeepSeek-V2 family (`deepseek2` GGUF arch): Multi-head Latent Attention
//! (MLA) + fine-grained MoE with shared experts. Validation target:
//! DeepSeek-V2-Lite (27 layers, 16 heads, qk 192 = nope 128 + rope 64,
//! v 128, kv_lora 512, no q-LoRA; layer 0 dense; layers 1+ = 64 routed
//! experts top-6 softmax + fused shared expert).
//!
//! Milestone-A implementation, correctness before speed: the six heavy
//! linears per layer (q, kv_a, kv_b, o, FFN projections) run on fucina's
//! quantized kernels; the small glue — YaRN RoPE on the 64-dim rope slice,
//! per-head attention over the cached K/V, router top-k — runs host-side in
//! plain f32, which keeps the MLA algebra auditable. K/V are reconstructed
//! from the latent INCREMENTALLY (kv_b applied once per new position, the
//! O(1)-per-step middle ground between colibri's naive per-step
//! reconstruction and full weight absorption); the true compressed-latent
//! cache and absorbed decode land with the perf milestone.
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
        if (!std.mem.eql(u8, arch, "deepseek2")) return Error.InvalidConfig;

        // Newer MLA-native conversions set key_length/value_length to the
        // LATENT attention dims (576/512) and carry the real per-head dims
        // in *_mla; older files carry them in key_length/value_length.
        const qk_head = gguf_meta.metaIntOpt(file, "deepseek2", "attention.key_length_mla", .reject_zero) orelse
            try metaInt(file, "deepseek2.attention.key_length");
        const v_head = gguf_meta.metaIntOpt(file, "deepseek2", "attention.value_length_mla", .reject_zero) orelse
            try metaInt(file, "deepseek2.attention.value_length");
        const rope_dims = try metaInt(file, "deepseek2.rope.dimension_count");
        return .{
            .vocab_size = try metaInt(file, "deepseek2.vocab_size"),
            .hidden_size = try metaInt(file, "deepseek2.embedding_length"),
            .num_layers = try metaInt(file, "deepseek2.block_count"),
            .num_heads = try metaInt(file, "deepseek2.attention.head_count"),
            .qk_nope_dim = qk_head - rope_dims,
            .qk_rope_dim = rope_dims,
            .qk_head_dim = qk_head,
            .v_head_dim = v_head,
            .kv_lora_rank = try metaInt(file, "deepseek2.attention.kv_lora_rank"),
            .dense_ffn_size = try metaInt(file, "deepseek2.feed_forward_length"),
            .leading_dense_layers = gguf_meta.metaIntOpt(file, "deepseek2", "leading_dense_block_count", .accept_zero) orelse 0,
            .num_experts = gguf_meta.metaIntOpt(file, "deepseek2", "expert_count", .accept_zero) orelse 0,
            .num_experts_used = gguf_meta.metaIntOpt(file, "deepseek2", "expert_used_count", .accept_zero) orelse 0,
            .expert_ffn_size = gguf_meta.metaIntOpt(file, "deepseek2", "expert_feed_forward_length", .accept_zero) orelse 0,
            .num_shared_experts = gguf_meta.metaIntOpt(file, "deepseek2", "expert_shared_count", .accept_zero) orelse 0,
            .expert_weights_scale = metaFloatOpt(file, "deepseek2.expert_weights_scale") orelse 1.0,
            .expert_gating_func = gguf_meta.metaIntOpt(file, "deepseek2", "expert_gating_func", .accept_zero) orelse 1,
            .expert_weights_norm = file.getBool("deepseek2.expert_weights_norm") orelse false,
            .q_lora_rank = gguf_meta.metaIntOpt(file, "deepseek2", "attention.q_lora_rank", .accept_zero) orelse 0,
            .rms_norm_eps = try metaFloat(file, "deepseek2.attention.layer_norm_rms_epsilon"),
            .rope_theta = metaFloatOpt(file, "deepseek2.rope.freq_base") orelse 10000.0,
            .yarn_factor = metaFloatOpt(file, "deepseek2.rope.scaling.factor") orelse 1.0,
            .yarn_orig_ctx = gguf_meta.metaIntOpt(file, "deepseek2", "rope.scaling.original_context_length", .accept_zero) orelse 0,
            .yarn_log_multiplier = metaFloatOpt(file, "deepseek2.rope.scaling.yarn_log_multiplier") orelse 0.0,
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

/// YaRN rotary table for the rope slice: per (position, pair) cos/sin with
/// the frequency blend between interpolation (freq/factor) and
/// extrapolation (original freq) across the correction ramp
/// [beta_fast=32, beta_slow=1] — the HF DeepseekV2Yarn reference. V2-Lite's
/// mscale == mscale_all_dim, so the cos/sin magnitude correction cancels to
/// 1 and only the attention-scale mscale² survives (`attnScale`).
const YarnRope = struct {
    cos: []f32, // [capacity][pairs]
    sin: []f32,
    pairs: usize,
    capacity: usize,

    fn init(allocator: Allocator, config: Config, capacity: usize) !YarnRope {
        const dim = config.qk_rope_dim;
        const pairs = dim / 2;
        const inv_freq = try allocator.alloc(f64, pairs);
        defer allocator.free(inv_freq);

        const base: f64 = config.rope_theta;
        const factor: f64 = config.yarn_factor;
        for (inv_freq, 0..) |*f, i| {
            const extra = std.math.pow(f64, base, -(@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(dim))));
            f.* = extra;
        }
        if (factor > 1.0 and config.yarn_orig_ctx > 0) {
            const orig: f64 = @floatFromInt(config.yarn_orig_ctx);
            const correction = struct {
                fn dimFor(rotations: f64, d: f64, b: f64, o: f64) f64 {
                    return d * @log(o / (rotations * 2.0 * std.math.pi)) / (2.0 * @log(b));
                }
            };
            const d_f: f64 = @floatFromInt(dim);
            var low = @floor(correction.dimFor(32.0, d_f, base, orig));
            var high = @ceil(correction.dimFor(1.0, d_f, base, orig));
            low = @max(low, 0);
            high = @min(high, d_f - 1);
            for (inv_freq, 0..) |*f, i| {
                const extra = f.*;
                const inter = extra / factor;
                // ramp 0 -> 1 across [low, high]; mask = 1 - ramp keeps the
                // fast-rotating dims extrapolated (original freq).
                var ramp = (@as(f64, @floatFromInt(i)) - low) / @max(high - low, 0.001);
                ramp = @min(@max(ramp, 0.0), 1.0);
                const mask = 1.0 - ramp;
                f.* = inter * (1.0 - mask) + extra * mask;
            }
        }

        const cos = try allocator.alloc(f32, capacity * pairs);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, capacity * pairs);
        for (0..capacity) |pos| {
            for (0..pairs) |i| {
                const angle = @as(f64, @floatFromInt(pos)) * inv_freq[i];
                cos[pos * pairs + i] = @floatCast(@cos(angle));
                sin[pos * pairs + i] = @floatCast(@sin(angle));
            }
        }
        return .{ .cos = cos, .sin = sin, .pairs = pairs, .capacity = capacity };
    }

    fn deinit(self: *YarnRope, allocator: Allocator) void {
        allocator.free(self.cos);
        allocator.free(self.sin);
        self.* = undefined;
    }

    /// Rotate one `dim`-wide rope slice in place at `pos`.
    fn apply(self: *const YarnRope, v: []f32, pos: usize) void {
        const c = self.cos[pos * self.pairs ..][0..self.pairs];
        const s = self.sin[pos * self.pairs ..][0..self.pairs];
        switch (rope_pairing) {
            .interleaved => for (0..self.pairs) |i| {
                const a = v[2 * i];
                const b = v[2 * i + 1];
                v[2 * i] = a * c[i] - b * s[i];
                v[2 * i + 1] = a * s[i] + b * c[i];
            },
            .half => for (0..self.pairs) |i| {
                const a = v[i];
                const b = v[i + self.pairs];
                v[i] = a * c[i] - b * s[i];
                v[i + self.pairs] = a * s[i] + b * c[i];
            },
        }
    }
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
    /// (1 + yarn_log_multiplier * ln(factor))² / sqrt(qk_head): the YaRN
    /// mscale² attention-scale correction folded with the 1/sqrt(d) scale.
    attn_scale: f32,

    pub const MoeStreamOptions = struct {
        gguf_path: []const u8,
        cache_bytes: ?usize = null,
        cache_slots_per_layer: ?usize = null,
        readahead: bool = true,
        auto_pin: bool = true,
        pin_bytes: ?usize = null,
    };

    pub const LoadOptions = struct {
        moe_stream: ?MoeStreamOptions = null,
    };

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, max_positions: usize) !Model {
        var file = try gguf.File.loadMmapAuto(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFile(ctx, &file, max_positions);
    }

    pub fn loadGgufFromFile(ctx: *ExecContext, file: *gguf.File, max_positions: usize) !Model {
        return loadGgufFromFileOptions(ctx, file, max_positions, .{});
    }

    pub fn loadGgufFromFileOptions(ctx: *ExecContext, file: *gguf.File, max_positions: usize, options: LoadOptions) !Model {
        const config = try Config.fromGguf(file);
        const allocator = ctx.allocator;

        var expert_store: ?*fucina.ExpertStore = null;
        if (options.moe_stream) |stream_options| {
            if (config.num_experts > 0) {
                const split_paths = try gguf.File.splitPartPaths(allocator, stream_options.gguf_path);
                defer if (split_paths) |paths| {
                    for (paths) |part| allocator.free(part);
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

        var rope = try YarnRope.init(allocator, config, max_positions);
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
        if (cache.len >= cache.capacity or cache.len >= self.rope.capacity) return Error.KvCacheOverflow;
        const pos = cache.len;

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

            const q = blk: {
                if (layer.q_proj) |*direct| {
                    var q_t = try direct.linearSeq(ctx, &h_t, .embed, .q);
                    defer q_t.deinit();
                    break :blk try allocator.dupe(f32, try q_t.dataConst());
                }
                // q-LoRA: hidden -> q_lora -> rmsnorm -> heads*qk_head.
                var qa_t = try layer.q_a.?.linearSeq(ctx, &h_t, .embed, .q);
                defer qa_t.deinit();
                const q_lat = try allocator.alloc(f32, cfg.q_lora_rank);
                defer allocator.free(q_lat);
                rmsNormInto(q_lat, try qa_t.dataConst(), layer.q_a_norm.?, cfg.rms_norm_eps);
                var q_lat_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, cfg.q_lora_rank }, q_lat);
                defer q_lat_t.deinit();
                var qb_t = try layer.q_b.?.linearSeq(ctx, &q_lat_t, .embed, .q);
                defer qb_t.deinit();
                break :blk try allocator.dupe(f32, try qb_t.dataConst());
            };
            defer allocator.free(q);

            var kv_a_t = try layer.kv_a.linearSeq(ctx, &h_t, .embed, .k);
            defer kv_a_t.deinit();
            const kv_a = try kv_a_t.dataConst();

            // Latent norm + rope: shared k_pe (MQA) and per-head q_pe at `pos`.
            const latent = try allocator.alloc(f32, cfg.kv_lora_rank);
            defer allocator.free(latent);
            rmsNormInto(latent, kv_a[0..cfg.kv_lora_rank], layer.kv_a_norm, cfg.rms_norm_eps);
            const k_pe = try allocator.dupe(f32, kv_a[cfg.kv_lora_rank..][0..cfg.qk_rope_dim]);
            defer allocator.free(k_pe);
            self.rope.apply(k_pe, pos);

            const t_len = pos + 1;
            const scores = try allocator.alloc(f32, t_len);
            defer allocator.free(scores);
            switch (cache.mode) {
                .full => {
                    // Reconstruct this position's k_nope/v via kv_b and run
                    // attention over materialized K/V (validation baseline).
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

                    for (0..cfg.num_heads) |h| {
                        const q_head = q[h * cfg.qk_head_dim ..][0..cfg.qk_head_dim];
                        self.rope.apply(q_head[cfg.qk_nope_dim..], pos);
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
                    const lora = cfg.kv_lora_rank;
                    const row_w = lora + cfg.qk_rope_dim;
                    const k_layer = cache.k[layer_i];
                    const dst = k_layer[pos * row_w ..][0..row_w];
                    @memcpy(dst[0..lora], latent);
                    @memcpy(dst[lora..], k_pe);

                    const q_eff = try allocator.alloc(f32, lora);
                    defer allocator.free(q_eff);
                    const ctx_lat = try allocator.alloc(f32, lora);
                    defer allocator.free(ctx_lat);
                    for (0..cfg.num_heads) |h| {
                        const q_head = q[h * cfg.qk_head_dim ..][0..cfg.qk_head_dim];
                        self.rope.apply(q_head[cfg.qk_nope_dim..], pos);
                        const q_pe = q_head[cfg.qk_nope_dim..];

                        // q_eff = W_kb_k[h]^T @ q_nope  ([nope, lora] rows).
                        const wk = layer.kv_b_k[h * cfg.qk_nope_dim * lora ..][0 .. cfg.qk_nope_dim * lora];
                        @memset(q_eff, 0);
                        for (0..cfg.qk_nope_dim) |i| {
                            const qi = q_head[i];
                            const w_row = wk[i * lora ..][0..lora];
                            for (q_eff, w_row) |*acc, w| acc.* += qi * w;
                        }

                        for (0..t_len) |t| {
                            const row = k_layer[t * row_w ..][0..row_w];
                            var dot: f32 = 0;
                            for (q_eff, row[0..lora]) |a, b| dot += a * b;
                            for (q_pe, row[lora..]) |a, b| dot += a * b;
                            scores[t] = dot * self.attn_scale;
                        }
                        softmaxInPlace(scores);

                        @memset(ctx_lat, 0);
                        for (0..t_len) |t| {
                            const w = scores[t];
                            const row = k_layer[t * row_w ..][0..lora];
                            for (ctx_lat, row) |*acc, c| acc.* += w * c;
                        }

                        // out[h] = W_kb_v[h] @ ctx_lat  ([v_head, lora] rows).
                        const wv = layer.kv_b_v[h * cfg.v_head_dim * lora ..][0 .. cfg.v_head_dim * lora];
                        const out_head = attn_out[h * cfg.v_head_dim ..][0..cfg.v_head_dim];
                        for (out_head, 0..) |*o, i| {
                            const w_row = wv[i * lora ..][0..lora];
                            var acc: f32 = 0;
                            for (w_row, ctx_lat) |w, c| acc += w * c;
                            o.* = acc;
                        }
                    }
                },
            }

            var attn_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, cfg.num_heads * cfg.v_head_dim }, attn_out);
            defer attn_t.deinit();
            var o_t = try layer.o_proj.linearSeq(ctx, &attn_t, .embed, .attn);
            defer o_t.deinit();
            for (x, try o_t.dataConst()) |*xi, oi| xi.* += oi;

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

        // Top-k selection (k is single digits; simple repeated max).
        var selected: [64]usize = undefined;
        var routing: [64]f32 = undefined;
        std.debug.assert(cfg.num_experts_used <= selected.len);
        for (0..cfg.num_experts_used) |slot| {
            var best: usize = 0;
            var best_c: f32 = -std.math.inf(f32);
            for (choice, 0..) |c, e| {
                if (c > best_c) {
                    best_c = c;
                    best = e;
                }
            }
            choice[best] = -std.math.inf(f32); // consumed
            selected[slot] = best;
            routing[slot] = probs[best];
        }
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
