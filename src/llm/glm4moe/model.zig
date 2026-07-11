//! GLM-4.5 family (`glm4moe` GGUF arch): GQA attention with QKV biases and
//! partial rotary (64 of 128 dims), DeepSeek-V3-style MoE (sigmoid noaux
//! routing with a selection bias, renormalized weights, one shared expert,
//! one leading dense layer), and — the reason this family is here — a
//! native MTP (multi-token-prediction) `nextn` layer for lossless
//! self-speculative decoding.
//!
//! Same correctness-first shape as the deepseek2 port: the heavy linears
//! and the fused/streamed MoE run on fucina kernels; attention, rope, and
//! routing glue run host-side in auditable f32. `step` handles S >= 1
//! (positions processed causally in sequence within one call) and returns
//! per-position logits, which is what the MTP verify needs.
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
};

/// Rope pairing for the 64 rotated dims (empirical, like deepseek2's).
pub const RopePairing = enum { interleaved, half };
pub const rope_pairing: RopePairing = .interleaved;
/// The MTP stream's rope positions: entry i of the stream embeds token
/// i+1, so its sequence position is arguably i+1 (shift 1) rather than i.
/// Empirical, like the pairing.
pub const mtp_pos_shift: usize = 0;
pub const mtp_debug: bool = false;

fn maxAbs(v: []const f32) f32 {
    var m: f32 = 0;
    for (v) |x| m = @max(m, @abs(x));
    return m;
}

pub const Config = struct {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize, // trunk layers (block_count - nextn layers)
    num_nextn_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dims: usize,
    dense_ffn_size: usize,
    leading_dense_layers: usize,
    num_experts: usize,
    num_experts_used: usize,
    expert_ffn_size: usize,
    num_shared_experts: usize,
    expert_weights_scale: f32,
    expert_gating_func: usize,
    expert_weights_norm: bool,
    rms_norm_eps: f32,
    rope_theta: f32,

    pub fn fromGguf(file: *const gguf.File) !Config {
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        if (!std.mem.eql(u8, arch, "glm4moe")) return Error.InvalidConfig;
        const block_count = try metaInt(file, "glm4moe.block_count");
        const nextn = gguf_meta.metaIntOpt(file, "glm4moe", "nextn_predict_layers", .accept_zero) orelse 0;
        const embd = try file.get("token_embd.weight");
        const shape = try embd.logicalMatrixShape();
        return .{
            .vocab_size = shape[0],
            .hidden_size = try metaInt(file, "glm4moe.embedding_length"),
            .num_layers = block_count - nextn,
            .num_nextn_layers = nextn,
            .num_heads = try metaInt(file, "glm4moe.attention.head_count"),
            .num_kv_heads = try metaInt(file, "glm4moe.attention.head_count_kv"),
            .head_dim = try metaInt(file, "glm4moe.attention.key_length"),
            .rope_dims = try metaInt(file, "glm4moe.rope.dimension_count"),
            .dense_ffn_size = try metaInt(file, "glm4moe.feed_forward_length"),
            .leading_dense_layers = gguf_meta.metaIntOpt(file, "glm4moe", "leading_dense_block_count", .accept_zero) orelse 0,
            .num_experts = gguf_meta.metaIntOpt(file, "glm4moe", "expert_count", .accept_zero) orelse 0,
            .num_experts_used = gguf_meta.metaIntOpt(file, "glm4moe", "expert_used_count", .accept_zero) orelse 0,
            .expert_ffn_size = gguf_meta.metaIntOpt(file, "glm4moe", "expert_feed_forward_length", .accept_zero) orelse 0,
            .num_shared_experts = gguf_meta.metaIntOpt(file, "glm4moe", "expert_shared_count", .accept_zero) orelse 0,
            .expert_weights_scale = metaFloatOpt(file, "glm4moe.expert_weights_scale") orelse 1.0,
            .expert_gating_func = gguf_meta.metaIntOpt(file, "glm4moe", "expert_gating_func", .accept_zero) orelse 1,
            .expert_weights_norm = file.getBool("glm4moe.expert_weights_norm") orelse false,
            .rms_norm_eps = try metaFloat(file, "glm4moe.attention.layer_norm_rms_epsilon"),
            .rope_theta = metaFloatOpt(file, "glm4moe.rope.freq_base") orelse 10000.0,
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

/// Plain-rope cos/sin table over the rotated dims.
const Rope = struct {
    cos: []f32,
    sin: []f32,
    pairs: usize,
    capacity: usize,

    fn init(allocator: Allocator, config: Config, capacity: usize) !Rope {
        const pairs = config.rope_dims / 2;
        const cos = try allocator.alloc(f32, capacity * pairs);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, capacity * pairs);
        for (0..capacity) |pos| {
            for (0..pairs) |i| {
                const freq = std.math.pow(f64, config.rope_theta, -(@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(config.rope_dims))));
                const angle = @as(f64, @floatFromInt(pos)) * freq;
                cos[pos * pairs + i] = @floatCast(@cos(angle));
                sin[pos * pairs + i] = @floatCast(@sin(angle));
            }
        }
        return .{ .cos = cos, .sin = sin, .pairs = pairs, .capacity = capacity };
    }

    fn deinit(self: *Rope, allocator: Allocator) void {
        allocator.free(self.cos);
        allocator.free(self.sin);
        self.* = undefined;
    }

    /// Rotate the first `2*pairs` dims of one head slice, in place.
    fn apply(self: *const Rope, head: []f32, pos: usize) void {
        const c = self.cos[pos * self.pairs ..][0..self.pairs];
        const s = self.sin[pos * self.pairs ..][0..self.pairs];
        switch (rope_pairing) {
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

const MoeFfn = struct {
    router: LinearWeight,
    router_bias: ?[]f32,
    gate: fucina.MoeRhs,
    up: fucina.MoeRhs,
    down: fucina.MoeRhs,
    shared_gate: LinearWeight,
    shared_up: LinearWeight,
    shared_down: LinearWeight,

    fn deinit(self: *MoeFfn, allocator: Allocator) void {
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

    fn deinit(self: *Ffn, allocator: Allocator) void {
        switch (self.*) {
            .dense => |*d| d.deinit(),
            .moe => |*m| m.deinit(allocator),
        }
        self.* = undefined;
    }
};

const Layer = struct {
    attn_norm: []f32,
    post_attention_norm: []f32,
    q_proj: LinearWeight,
    q_bias: []f32,
    k_proj: LinearWeight,
    k_bias: []f32,
    v_proj: LinearWeight,
    v_bias: []f32,
    o_proj: LinearWeight,
    ffn: Ffn,

    fn deinit(self: *Layer, allocator: Allocator) void {
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

/// The MTP (`nextn`) head: its own token embedding and output head plus one
/// full trunk-shaped transformer layer. Draft recurrence:
///   h0 = eh_proj([enorm(embed(token)) | hnorm(h_prev)])
///   h1 = layer(h0)  (attention over the MTP stream's own cache)
///   logits = shared_head(shared_head_norm(h1)),  next h_prev = h1
pub const MtpHead = struct {
    embed: LinearWeight,
    eh_proj: LinearWeight, // 2*hidden -> hidden
    enorm: []f32,
    hnorm: []f32,
    shared_head_norm: []f32,
    shared_head: LinearWeight,
    layer: Layer,

    fn deinit(self: *MtpHead, allocator: Allocator) void {
        self.layer.deinit(allocator);
        self.shared_head.deinit();
        allocator.free(self.shared_head_norm);
        allocator.free(self.hnorm);
        allocator.free(self.enorm);
        self.eh_proj.deinit();
        self.embed.deinit();
        self.* = undefined;
    }
};

/// Host K/V cache: per layer `[capacity, kv_heads, head_dim]` for K
/// (post-rope) and V. `truncate` is the speculative rewind.
pub const Cache = struct {
    allocator: Allocator,
    k: [][]f32,
    v: [][]f32,
    len: usize = 0,
    capacity: usize,

    pub fn init(allocator: Allocator, n_layers: usize, kv_heads: usize, head_dim: usize, capacity: usize) !Cache {
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

    pub fn deinit(self: *Cache) void {
        for (self.k) |l| self.allocator.free(l);
        for (self.v) |l| self.allocator.free(l);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.* = undefined;
    }

    pub fn truncate(self: *Cache, keep: usize) void {
        if (keep < self.len) self.len = keep;
    }
};

pub const Model = struct {
    allocator: Allocator,
    config: Config,
    token_embedding: LinearWeight,
    output_norm: []f32,
    output: LinearWeight,
    layers: []Layer,
    mtp: ?MtpHead,
    rope: Rope,
    attn_scale: f32,
    weight_mapping: ?gguf.File.MappedRegion = null,
    expert_store: ?*fucina.ExpertStore = null,
    /// Trunk hidden state (pre-output-norm) of the LAST position of the
    /// most recent `step` — the MTP draft's h_prev seed.
    last_hidden: []f32,
    /// All pre-norm hiddens of the most recent `step` ([S, hidden], model-
    /// owned, valid until the next step) — the MTP stream consumes one per
    /// committed position.
    step_hiddens: []f32 = &.{},

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

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, max_positions: usize, options: LoadOptions) !Model {
        var file = try gguf.File.loadMmapAuto(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFileOptions(ctx, &file, max_positions, options);
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
                // block indices include the nextn layer (blk.46).
                expert_store = try fucina.ExpertStore.create(allocator, store_paths, config.num_layers + config.num_nextn_layers, .{
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

        var mtp: ?MtpHead = null;
        errdefer if (mtp) |*m| m.deinit(allocator);
        if (config.num_nextn_layers > 0) {
            const mtp_i = config.num_layers; // blk.46 on Air
            var name_buf: [96]u8 = undefined;
            var embed = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, mtp_i, "nextn.embed_tokens.weight")), config.vocab_size, config.hidden_size);
            errdefer embed.deinit();
            var eh_proj = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, mtp_i, "nextn.eh_proj.weight")), config.hidden_size, 2 * config.hidden_size);
            errdefer eh_proj.deinit();
            const enorm = try hostVector(allocator, file, try layerName(&name_buf, mtp_i, "nextn.enorm.weight"), config.hidden_size);
            errdefer allocator.free(enorm);
            const hnorm = try hostVector(allocator, file, try layerName(&name_buf, mtp_i, "nextn.hnorm.weight"), config.hidden_size);
            errdefer allocator.free(hnorm);
            const sh_norm = try hostVector(allocator, file, try layerName(&name_buf, mtp_i, "nextn.shared_head_norm.weight"), config.hidden_size);
            errdefer allocator.free(sh_norm);
            var sh_head = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, mtp_i, "nextn.shared_head_head.weight")), config.vocab_size, config.hidden_size);
            errdefer sh_head.deinit();
            var mtp_layer = try loadLayer(ctx, file, config, mtp_i, expert_store);
            errdefer mtp_layer.deinit(allocator);
            mtp = .{
                .embed = embed,
                .eh_proj = eh_proj,
                .enorm = enorm,
                .hnorm = hnorm,
                .shared_head_norm = sh_norm,
                .shared_head = sh_head,
                .layer = mtp_layer,
            };
        }

        if (expert_store) |store| try store.finalize();
        const weight_mapping = if (config.num_experts > 0 and expert_store == null) file.takeMapping() else null;
        if (config.num_experts > 0 and expert_store == null and weight_mapping == null) return Error.InvalidWeightShape;

        var rope = try Rope.init(allocator, config, max_positions);
        errdefer rope.deinit(allocator);
        const last_hidden = try allocator.alloc(f32, config.hidden_size);

        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = token_embedding,
            .output_norm = output_norm,
            .output = output,
            .layers = layers,
            .mtp = mtp,
            .rope = rope,
            .attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.head_dim))),
            .weight_mapping = weight_mapping,
            .expert_store = expert_store,
            .last_hidden = last_hidden,
        };
    }

    pub fn deinit(self: *Model) void {
        if (self.step_hiddens.len > 0) self.allocator.free(self.step_hiddens);
        self.allocator.free(self.last_hidden);
        self.rope.deinit(self.allocator);
        if (self.mtp) |*m| m.deinit(self.allocator);
        for (self.layers) |*l| l.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.allocator.free(self.output_norm);
        self.output.deinit();
        self.token_embedding.deinit();
        if (self.expert_store) |store| store.destroy();
        if (self.weight_mapping) |*mapping| mapping.deinit();
        self.* = undefined;
    }

    pub fn initCache(self: *const Model, capacity: usize) !Cache {
        return Cache.init(self.allocator, self.config.num_layers, self.config.num_kv_heads, self.config.head_dim, capacity);
    }

    pub fn initMtpCache(self: *const Model, capacity: usize) !Cache {
        return Cache.init(self.allocator, 1, self.config.num_kv_heads, self.config.head_dim, capacity);
    }

    /// Process `tokens` at positions [cache.len, cache.len + S) and return
    /// per-position next-token logits `[S][vocab]` (caller frees the outer
    /// and inner slices). Positions are computed causally in sequence, so
    /// per-row numerics match S=1 steps exactly — the MTP verify contract.
    /// Also refreshes `last_hidden` (pre-norm trunk state of the last row).
    pub fn step(self: *Model, ctx: *ExecContext, cache: *Cache, tokens: []const usize) ![][]f32 {
        const cfg = self.config;
        const allocator = ctx.allocator;
        if (tokens.len == 0) return Error.InvalidSequenceLength;
        if (cache.len + tokens.len > cache.capacity or cache.len + tokens.len > self.rope.capacity) return Error.KvCacheOverflow;

        const S = tokens.len;
        // Residual stream rows [S, hidden].
        const x = try allocator.alloc(f32, S * cfg.hidden_size);
        defer allocator.free(x);
        {
            var emb = try self.token_embedding.getRowsAs(ctx, tokens, .embed);
            defer emb.deinit();
            @memcpy(x, try emb.dataConst());
        }

        for (self.layers, 0..) |*layer, layer_i| {
            try self.layerForward(ctx, cache, layer, layer_i, x, S, cache.len, 0, null);
        }
        cache.len += S;

        // Per-position logits through the shared head. The hiddens handed
        // to the MTP stream are POST-output-norm (matching the reference
        // serving stacks, where the target model's forward output is the
        // final-normed hidden).
        @memcpy(self.last_hidden, x[(S - 1) * cfg.hidden_size ..][0..cfg.hidden_size]);
        if (self.step_hiddens.len != x.len) {
            if (self.step_hiddens.len > 0) self.allocator.free(self.step_hiddens);
            self.step_hiddens = try self.allocator.alloc(f32, x.len);
        }
        for (0..S) |r| {
            rmsNormInto(self.step_hiddens[r * cfg.hidden_size ..][0..cfg.hidden_size], x[r * cfg.hidden_size ..][0..cfg.hidden_size], self.output_norm, cfg.rms_norm_eps);
        }
        return self.headLogits(ctx, x, S, self.output_norm, &self.output);
    }

    fn headLogits(self: *Model, ctx: *ExecContext, x: []const f32, S: usize, norm: []const f32, head: *const LinearWeight) ![][]f32 {
        const cfg = self.config;
        const allocator = ctx.allocator;
        const normed = try allocator.alloc(f32, S * cfg.hidden_size);
        defer allocator.free(normed);
        for (0..S) |r| {
            rmsNormInto(normed[r * cfg.hidden_size ..][0..cfg.hidden_size], x[r * cfg.hidden_size ..][0..cfg.hidden_size], norm, cfg.rms_norm_eps);
        }
        var normed_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ S, cfg.hidden_size }, normed);
        defer normed_t.deinit();
        var logits_t = try head.linearSeq(ctx, &normed_t, .embed, .vocab);
        defer logits_t.deinit();
        const flat = try logits_t.dataConst();
        const out = try allocator.alloc([]f32, S);
        errdefer allocator.free(out);
        var built: usize = 0;
        errdefer for (out[0..built]) |row| allocator.free(row);
        for (0..S) |r| {
            out[r] = try allocator.dupe(f32, flat[r * cfg.vocab_size ..][0..cfg.vocab_size]);
            built += 1;
        }
        return out;
    }

    /// One transformer layer over `x` rows in place. `mtp_cache` non-null
    /// routes the K/V through the MTP stream's own single-layer cache.
    fn layerForward(self: *Model, ctx: *ExecContext, cache: ?*Cache, layer: *const Layer, layer_i: usize, x: []f32, S: usize, pos0: usize, rope_shift: usize, mtp_cache: ?*Cache) !void {
        const cfg = self.config;
        const allocator = ctx.allocator;
        const heads_per_kv = cfg.num_heads / cfg.num_kv_heads;

        const h_norm = try allocator.alloc(f32, S * cfg.hidden_size);
        defer allocator.free(h_norm);
        for (0..S) |r| rmsNormInto(h_norm[r * cfg.hidden_size ..][0..cfg.hidden_size], x[r * cfg.hidden_size ..][0..cfg.hidden_size], layer.attn_norm, cfg.rms_norm_eps);
        var h_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ S, cfg.hidden_size }, h_norm);
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
        const q_width = cfg.num_heads * cfg.head_dim;
        const kv_width = cfg.num_kv_heads * cfg.head_dim;
        for (0..S) |r| {
            const q_row = q[r * q_width ..][0..q_width];
            for (q_row, layer.q_bias) |*qv, b| qv.* += b;
            const k_row = k[r * kv_width ..][0..kv_width];
            for (k_row, layer.k_bias) |*kv, b| kv.* += b;
            const v_row = v[r * kv_width ..][0..kv_width];
            for (v_row, layer.v_bias) |*vv, b| vv.* += b;
        }

        const kv_cache = mtp_cache orelse cache.?;
        const cache_layer = if (mtp_cache != null) 0 else layer_i;
        const attn_out = try allocator.alloc(f32, S * q_width);
        defer allocator.free(attn_out);
        const scores = try allocator.alloc(f32, pos0 + S);
        defer allocator.free(scores);

        for (0..S) |r| {
            const pos = pos0 + r;
            const rope_pos = pos + rope_shift;
            // Partial rope + append this position's K/V.
            const k_row = k[r * kv_width ..][0..kv_width];
            const v_row = v[r * kv_width ..][0..kv_width];
            for (0..cfg.num_kv_heads) |h| {
                self.rope.apply(k_row[h * cfg.head_dim ..][0..cfg.head_dim], rope_pos);
            }
            const k_dst = kv_cache.k[cache_layer][pos * kv_width ..][0..kv_width];
            const v_dst = kv_cache.v[cache_layer][pos * kv_width ..][0..kv_width];
            @memcpy(k_dst, k_row);
            @memcpy(v_dst, v_row);

            const t_len = pos + 1;
            const q_row = q[r * q_width ..][0..q_width];
            for (0..cfg.num_heads) |h| {
                const q_head = q_row[h * cfg.head_dim ..][0..cfg.head_dim];
                self.rope.apply(q_head, rope_pos);
                const kv_h = h / heads_per_kv;
                for (0..t_len) |t| {
                    const kt = kv_cache.k[cache_layer][(t * cfg.num_kv_heads + kv_h) * cfg.head_dim ..][0..cfg.head_dim];
                    var dot: f32 = 0;
                    for (q_head, kt) |a, b| dot += a * b;
                    scores[t] = dot * self.attn_scale;
                }
                softmaxInPlace(scores[0..t_len]);
                const out_head = attn_out[r * q_width + h * cfg.head_dim ..][0..cfg.head_dim];
                @memset(out_head, 0);
                for (0..t_len) |t| {
                    const w = scores[t];
                    const vt = kv_cache.v[cache_layer][(t * cfg.num_kv_heads + kv_h) * cfg.head_dim ..][0..cfg.head_dim];
                    for (out_head, vt) |*o, val| o.* += w * val;
                }
            }
        }
        if (mtp_cache) |mc| mc.len = pos0 + S;

        var attn_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ S, q_width }, attn_out);
        defer attn_t.deinit();
        var o_t = try layer.o_proj.linearSeq(ctx, &attn_t, .embed, .attn);
        defer o_t.deinit();
        for (x, try o_t.dataConst()) |*xi, oi| xi.* += oi;

        // FFN with the GLM sandwich name (post_attention_norm = pre-FFN).
        for (0..S) |r| rmsNormInto(h_norm[r * cfg.hidden_size ..][0..cfg.hidden_size], x[r * cfg.hidden_size ..][0..cfg.hidden_size], layer.post_attention_norm, cfg.rms_norm_eps);
        switch (layer.ffn) {
            .dense => |*dense| {
                var f_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ S, cfg.hidden_size }, h_norm);
                defer f_t.deinit();
                const y = try swigluLinear(ctx, allocator, &f_t, &dense.gate, &dense.up, &dense.down);
                defer allocator.free(y);
                for (x, y) |*xi, yi| xi.* += yi;
            },
            .moe => |*moe| {
                // Row-wise through the fused decode op: q8_0 experts route
                // through the dual-format decode path, and consecutive rows
                // share expert reads via the store's LRU when streaming.
                for (0..S) |r| {
                    const row = h_norm[r * cfg.hidden_size ..][0..cfg.hidden_size];
                    var f_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, cfg.hidden_size }, row);
                    defer f_t.deinit();
                    const y = try self.moeForward(ctx, allocator, moe, &f_t);
                    defer allocator.free(y);
                    for (x[r * cfg.hidden_size ..][0..cfg.hidden_size], y) |*xi, yi| xi.* += yi;
                }
            },
        }
    }

    /// DeepSeek-V3-style routing (same semantics the deepseek2 port
    /// validated on Moonlight): sigmoid scores, selection bias for the
    /// top-k choice only, renormalized weights, expert_weights_scale.
    fn moeForward(self: *Model, ctx: *ExecContext, allocator: Allocator, moe: *const MoeFfn, f_t: *const fucina.Tensor(.{ .seq, .embed })) ![]f32 {
        const cfg = self.config;
        var logits_t = try moe.router.linearSeq(ctx, f_t, .embed, .expert);
        defer logits_t.deinit();
        const probs = try allocator.dupe(f32, try logits_t.dataConst());
        defer allocator.free(probs);
        switch (cfg.expert_gating_func) {
            2 => for (probs) |*p| {
                p.* = 1.0 / (1.0 + @exp(-p.*));
            },
            else => softmaxInPlace(probs),
        }
        const choice = try allocator.dupe(f32, probs);
        defer allocator.free(choice);
        if (moe.router_bias) |bias| {
            for (choice, bias) |*c, b| c.* += b;
        }

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

        var mix = try weights.moeSwiGluFfnSeq(ctx, f_t, &moe.gate, &moe.up, &moe.down, selected[0..cfg.num_experts_used], routing[0..cfg.num_experts_used], cfg.num_experts_used, cfg.expert_ffn_size, null, null);
        defer mix.deinit();

        const y = try allocator.dupe(f32, try mix.dataConst());
        errdefer allocator.free(y);
        const shared = try swigluLinear(ctx, allocator, f_t, &moe.shared_gate, &moe.shared_up, &moe.shared_down);
        defer allocator.free(shared);
        for (y, shared) |*yi, si| yi.* += si;
        return y;
    }

    /// One MTP draft step: combine the token embedding with the previous
    /// hidden, run the nextn layer over the MTP stream's cache at position
    /// `mtp_cache.len`, and return (greedy token, logits row, new hidden).
    /// `h_prev` is `last_hidden` for the first draft and the returned
    /// hidden for the chained drafts.
    pub fn mtpDraftStep(self: *Model, ctx: *ExecContext, mtp_cache: *Cache, token: usize, h_prev: []const f32, h_out: []f32) ![]f32 {
        const cfg = self.config;
        const allocator = ctx.allocator;
        const mtp = if (self.mtp) |*m| m else return Error.InvalidConfig;
        if (mtp_cache.len >= mtp_cache.capacity or mtp_cache.len >= self.rope.capacity) return Error.KvCacheOverflow;

        const x = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(x);
        {
            var ids = [_]usize{token};
            var emb = try mtp.embed.getRowsAs(ctx, &ids, .embed);
            defer emb.deinit();
            const cat = try allocator.alloc(f32, 2 * cfg.hidden_size);
            defer allocator.free(cat);
            // Concat order per the GLM/DeepSeek MTP reference: normed token
            // embedding first, normed trunk hidden second.
            rmsNormInto(cat[0..cfg.hidden_size], try emb.dataConst(), mtp.enorm, cfg.rms_norm_eps);
            rmsNormInto(cat[cfg.hidden_size..], h_prev, mtp.hnorm, cfg.rms_norm_eps);
            var cat_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, 2 * cfg.hidden_size }, cat);
            defer cat_t.deinit();
            var proj_t = try mtp.eh_proj.linearSeq(ctx, &cat_t, .embed, .attn);
            defer proj_t.deinit();
            @memcpy(x, try proj_t.dataConst());
        }

        if (mtp_debug) {
            std.debug.print("mtp dbg: |h_prev|max {d:.3} |x=eh_proj|max {d:.3}", .{ maxAbs(h_prev), maxAbs(x) });
        }
        try self.layerForward(ctx, null, &mtp.layer, 0, x, 1, mtp_cache.len, mtp_pos_shift, mtp_cache);
        @memcpy(h_out, x);
        if (mtp_debug) std.debug.print(" |h1|max {d:.3}\n", .{maxAbs(x)});

        const rows = try self.headLogits(ctx, x, 1, mtp.shared_head_norm, &mtp.shared_head);
        defer allocator.free(rows);
        return rows[0];
    }
};

fn layerName(buf: []u8, layer_i: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer_i, suffix });
}

fn loadLayer(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize, store: ?*fucina.ExpertStore) !Layer {
    const allocator = ctx.allocator;
    var name_buf: [96]u8 = undefined;

    const attn_norm = try hostVector(allocator, file, try layerName(&name_buf, layer_i, "attn_norm.weight"), config.hidden_size);
    errdefer allocator.free(attn_norm);
    const post_attention_norm = try hostVector(allocator, file, try layerName(&name_buf, layer_i, "post_attention_norm.weight"), config.hidden_size);
    errdefer allocator.free(post_attention_norm);

    const q_dim = config.num_heads * config.head_dim;
    const kv_dim = config.num_kv_heads * config.head_dim;
    var q_proj = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "attn_q.weight")), q_dim, config.hidden_size);
    errdefer q_proj.deinit();
    const q_bias = try hostVector(allocator, file, try layerName(&name_buf, layer_i, "attn_q.bias"), q_dim);
    errdefer allocator.free(q_bias);
    var k_proj = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "attn_k.weight")), kv_dim, config.hidden_size);
    errdefer k_proj.deinit();
    const k_bias = try hostVector(allocator, file, try layerName(&name_buf, layer_i, "attn_k.bias"), kv_dim);
    errdefer allocator.free(k_bias);
    var v_proj = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "attn_v.weight")), kv_dim, config.hidden_size);
    errdefer v_proj.deinit();
    const v_bias = try hostVector(allocator, file, try layerName(&name_buf, layer_i, "attn_v.bias"), kv_dim);
    errdefer allocator.free(v_bias);
    var o_proj = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "attn_output.weight")), config.hidden_size, q_dim);
    errdefer o_proj.deinit();

    var ffn: Ffn = undefined;
    if (layer_i < config.leading_dense_layers or config.num_experts == 0) {
        var gate = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_gate.weight")), config.dense_ffn_size, config.hidden_size);
        errdefer gate.deinit();
        var up = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_up.weight")), config.dense_ffn_size, config.hidden_size);
        errdefer up.deinit();
        var down = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_down.weight")), config.hidden_size, config.dense_ffn_size);
        errdefer down.deinit();
        ffn = .{ .dense = .{ .gate = gate, .up = up, .down = down } };
    } else {
        var router = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_gate_inp.weight")), config.num_experts, config.hidden_size);
        errdefer router.deinit();
        var router_bias: ?[]f32 = null;
        errdefer if (router_bias) |b| allocator.free(b);
        var bias_buf: [96]u8 = undefined;
        if (file.maybeGet(try layerName(&bias_buf, layer_i, "exp_probs_b.bias"))) |bias_info| {
            router_bias = try hostVectorInfo(allocator, bias_info, config.num_experts);
        }
        var gate: fucina.MoeRhs = undefined;
        var up: fucina.MoeRhs = undefined;
        var down: fucina.MoeRhs = undefined;
        if (store) |st| {
            const trio = try weights.loadMoeRhsStreamed(st, file, layer_i, try file.get(try layerName(&name_buf, layer_i, "ffn_gate_exps.weight")), try file.get(try layerName(&name_buf, layer_i, "ffn_up_exps.weight")), try file.get(try layerName(&name_buf, layer_i, "ffn_down_exps.weight")), config.hidden_size, config.expert_ffn_size, config.num_experts);
            gate = trio.gate;
            up = trio.up;
            down = trio.down;
        } else {
            const borrow = file.is_mmap and !file.isSplit();
            gate = try weights.loadMoeRhs(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_gate_exps.weight")), config.hidden_size, config.expert_ffn_size, config.num_experts, borrow);
            up = try weights.loadMoeRhs(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_up_exps.weight")), config.hidden_size, config.expert_ffn_size, config.num_experts, borrow);
            down = try weights.loadMoeRhs(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_down_exps.weight")), config.expert_ffn_size, config.hidden_size, config.num_experts, borrow);
        }
        const shared_ffn = config.expert_ffn_size * config.num_shared_experts;
        var shared_gate = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_gate_shexp.weight")), shared_ffn, config.hidden_size);
        errdefer shared_gate.deinit();
        var shared_up = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_up_shexp.weight")), shared_ffn, config.hidden_size);
        errdefer shared_up.deinit();
        var shared_down = try LinearWeight.load(ctx, try file.get(try layerName(&name_buf, layer_i, "ffn_down_shexp.weight")), config.hidden_size, shared_ffn);
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

fn swigluLinear(ctx: *ExecContext, allocator: Allocator, x: *const fucina.Tensor(.{ .seq, .embed }), gate: *const LinearWeight, up: *const LinearWeight, down: *const LinearWeight) ![]f32 {
    var gate_t = try gate.linearSeq(ctx, x, .embed, .gate_up);
    defer gate_t.deinit();
    var up_t = try up.linearSeq(ctx, x, .embed, .gate_up);
    defer up_t.deinit();
    const width = gate_t.dim(.gate_up);
    const rows = gate_t.dim(.seq);
    const g = try allocator.alloc(f32, rows * width);
    defer allocator.free(g);
    for (g, try gate_t.dataConst(), try up_t.dataConst()) |*gi, gv, uv| gi.* = silu(gv) * uv;
    var g_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ rows, width }, g);
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
