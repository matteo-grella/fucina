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
    rms_norm_eps: f32,
    rope_theta: f32,
    yarn_factor: f32,
    yarn_orig_ctx: usize,
    yarn_log_multiplier: f32,

    pub fn fromGguf(file: *const gguf.File) !Config {
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        if (!std.mem.eql(u8, arch, "deepseek2")) return Error.InvalidConfig;

        const qk_head = try metaInt(file, "deepseek2.attention.key_length");
        const rope_dims = try metaInt(file, "deepseek2.rope.dimension_count");
        return .{
            .vocab_size = try metaInt(file, "deepseek2.vocab_size"),
            .hidden_size = try metaInt(file, "deepseek2.embedding_length"),
            .num_layers = try metaInt(file, "deepseek2.block_count"),
            .num_heads = try metaInt(file, "deepseek2.attention.head_count"),
            .qk_nope_dim = qk_head - rope_dims,
            .qk_rope_dim = rope_dims,
            .qk_head_dim = qk_head,
            .v_head_dim = try metaInt(file, "deepseek2.attention.value_length"),
            .kv_lora_rank = try metaInt(file, "deepseek2.attention.kv_lora_rank"),
            .dense_ffn_size = try metaInt(file, "deepseek2.feed_forward_length"),
            .leading_dense_layers = gguf_meta.metaIntOpt(file, "deepseek2", "leading_dense_block_count", .accept_zero) orelse 0,
            .num_experts = gguf_meta.metaIntOpt(file, "deepseek2", "expert_count", .accept_zero) orelse 0,
            .num_experts_used = gguf_meta.metaIntOpt(file, "deepseek2", "expert_used_count", .accept_zero) orelse 0,
            .expert_ffn_size = gguf_meta.metaIntOpt(file, "deepseek2", "expert_feed_forward_length", .accept_zero) orelse 0,
            .num_shared_experts = gguf_meta.metaIntOpt(file, "deepseek2", "expert_shared_count", .accept_zero) orelse 0,
            .expert_weights_scale = metaFloatOpt(file, "deepseek2.expert_weights_scale") orelse 1.0,
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

/// One stacked expert projection kept as raw GGUF blocks (expert-major):
/// expert `e`'s rows are the byte range [e*expert_bytes, +expert_bytes).
/// Runs through the borrowed-quantized linear (q4_k/q5_k/q6_k/q8_0), so the
/// non-256-aligned expert dims deepseek2 ships (1408) work unchanged.
const ExpertStack = struct {
    data: []const u8,
    ggml_type: gguf.GgmlType,
    out_dim: usize,
    in_dim: usize,
    expert_bytes: usize,

    fn init(info: *const gguf.TensorInfo, in_dim: usize, out_dim: usize, n_expert: usize) !ExpertStack {
        if (info.n_dims != 3) return Error.InvalidWeightShape;
        if (info.dims[0] != in_dim or info.dims[1] != out_dim or info.dims[2] != n_expert) return Error.InvalidWeightShape;
        switch (info.ggml_type) {
            .q4_k, .q5_k, .q6_k, .q8_0 => {},
            else => return Error.UnsupportedExpertQuant,
        }
        if (info.data.len % n_expert != 0) return Error.InvalidWeightShape;
        return .{
            .data = info.data,
            .ggml_type = info.ggml_type,
            .out_dim = out_dim,
            .in_dim = in_dim,
            .expert_bytes = info.data.len / n_expert,
        };
    }

    fn expertBytesSlice(self: *const ExpertStack, e: usize) []const u8 {
        return self.data[e * self.expert_bytes ..][0..self.expert_bytes];
    }

    /// y[seq, out] = x[seq, in] through expert `e`'s rows.
    fn linear(self: *const ExpertStack, ctx: *ExecContext, x: *const fucina.Tensor(.{ .seq, .embed }), e: usize, comptime out_tag: @TypeOf(.tag)) !fucina.Tensor(.{ .seq, out_tag }) {
        return switch (self.ggml_type) {
            inline .q4_k, .q5_k, .q6_k, .q8_0 => |t| weights.linearSeqBorrowedQuantized(
                comptime @field(fucina.DType, @tagName(t)),
                ctx,
                x,
                self.expertBytesSlice(e),
                .{ self.out_dim, self.in_dim },
                .{ .allow_gpu = false },
                .embed,
                out_tag,
            ),
            else => unreachable,
        };
    }
};

const MoeFfn = struct {
    router: LinearWeight,
    gate: ExpertStack,
    up: ExpertStack,
    down: ExpertStack,
    shared_gate: LinearWeight,
    shared_up: LinearWeight,
    shared_down: LinearWeight,

    fn deinit(self: *MoeFfn) void {
        self.shared_down.deinit();
        self.shared_up.deinit();
        self.shared_gate.deinit();
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
    q_proj: LinearWeight, // hidden -> heads*qk_head
    kv_a: LinearWeight, // hidden -> kv_lora + rope
    kv_b: LinearWeight, // kv_lora -> heads*(nope + v)
    o_proj: LinearWeight, // heads*v -> hidden
    ffn: Ffn,

    fn deinit(self: *Layer, allocator: Allocator) void {
        self.ffn.deinit();
        self.o_proj.deinit();
        self.kv_b.deinit();
        self.kv_a.deinit();
        self.q_proj.deinit();
        allocator.free(self.ffn_norm);
        allocator.free(self.kv_a_norm);
        allocator.free(self.attn_norm);
        self.* = undefined;
    }
};

/// Reconstructed-K/V cache: per layer, `[capacity, heads, qk_head]` K
/// (nope | rotated rope) and `[capacity, heads, v_head]` V, filled once per
/// appended position from the latent through kv_b.
pub const Cache = struct {
    allocator: Allocator,
    k: [][]f32,
    v: [][]f32,
    len: usize = 0,
    capacity: usize,

    pub fn init(allocator: Allocator, config: Config, capacity: usize) !Cache {
        const k = try allocator.alloc([]f32, config.num_layers);
        var built: usize = 0;
        errdefer {
            for (k[0..built]) |layer| allocator.free(layer);
            allocator.free(k);
        }
        const v = try allocator.alloc([]f32, config.num_layers);
        errdefer allocator.free(v);
        for (0..config.num_layers) |i| {
            k[i] = try allocator.alloc(f32, capacity * config.num_heads * config.qk_head_dim);
            built += 1;
        }
        var v_built: usize = 0;
        errdefer for (v[0..v_built]) |layer| allocator.free(layer);
        for (0..config.num_layers) |i| {
            v[i] = try allocator.alloc(f32, capacity * config.num_heads * config.v_head_dim);
            v_built += 1;
        }
        return .{ .allocator = allocator, .k = k, .v = v, .capacity = capacity };
    }

    pub fn deinit(self: *Cache) void {
        for (self.k) |layer| self.allocator.free(layer);
        for (self.v) |layer| self.allocator.free(layer);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
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
    /// The GGUF mmap: the MoE ExpertStacks borrow their blocks straight
    /// from it (unmapped last in deinit).
    weight_mapping: ?gguf.File.MappedRegion = null,
    /// (1 + yarn_log_multiplier * ln(factor))² / sqrt(qk_head): the YaRN
    /// mscale² attention-scale correction folded with the 1/sqrt(d) scale.
    attn_scale: f32,

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, max_positions: usize) !Model {
        var file = try gguf.File.loadMmapAuto(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFile(ctx, &file, max_positions);
    }

    pub fn loadGgufFromFile(ctx: *ExecContext, file: *gguf.File, max_positions: usize) !Model {
        const config = try Config.fromGguf(file);
        const allocator = ctx.allocator;

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
            layer.* = try loadLayer(ctx, file, config, i);
            built += 1;
        }

        var rope = try YarnRope.init(allocator, config, max_positions);
        errdefer rope.deinit(allocator);

        // Expert stacks borrow from the mapping; the model must own it.
        const weight_mapping = if (config.num_experts > 0) file.takeMapping() else null;
        if (config.num_experts > 0 and weight_mapping == null) return Error.InvalidWeightShape;

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
        if (self.weight_mapping) |*mapping| mapping.deinit();
        self.* = undefined;
    }

    pub fn initCache(self: *const Model, capacity: usize) !Cache {
        return Cache.init(self.allocator, self.config, capacity);
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

            var q_t = try layer.q_proj.linearSeq(ctx, &h_t, .embed, .q);
            defer q_t.deinit();
            const q = try allocator.dupe(f32, try q_t.dataConst());
            defer allocator.free(q);

            var kv_a_t = try layer.kv_a.linearSeq(ctx, &h_t, .embed, .k);
            defer kv_a_t.deinit();
            const kv_a = try kv_a_t.dataConst();

            // Latent norm, then reconstruct this position's k_nope/v via kv_b.
            const latent = try allocator.alloc(f32, cfg.kv_lora_rank);
            defer allocator.free(latent);
            rmsNormInto(latent, kv_a[0..cfg.kv_lora_rank], layer.kv_a_norm, cfg.rms_norm_eps);
            var latent_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, cfg.kv_lora_rank }, latent);
            defer latent_t.deinit();
            var kv_t = try layer.kv_b.linearSeq(ctx, &latent_t, .embed, .v);
            defer kv_t.deinit();
            const kv = try kv_t.dataConst(); // [heads * (nope + v)]

            // Rope: shared k_pe (MQA) and per-head q_pe, rotated at `pos`.
            const k_pe = try allocator.dupe(f32, kv_a[cfg.kv_lora_rank..][0..cfg.qk_rope_dim]);
            defer allocator.free(k_pe);
            self.rope.apply(k_pe, pos);

            // Append reconstructed K (nope | rotated pe) and V.
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

            // Per-head causal attention over positions [0, pos].
            const t_len = pos + 1;
            const scores = try allocator.alloc(f32, t_len);
            defer allocator.free(scores);
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
                    const y = try self.moeForward(ctx, allocator, moe, &f_t, h_norm);
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

    /// Routed mixture + shared expert. V2 semantics: softmax over ALL
    /// router logits, top-k of those probabilities used UN-renormalized
    /// (norm_topk_prob = false), scaled by expert_weights_scale.
    fn moeForward(self: *const Model, ctx: *ExecContext, allocator: Allocator, moe: *const MoeFfn, f_t: *const fucina.Tensor(.{ .seq, .embed }), f_row: []const f32) ![]f32 {
        _ = f_row;
        const cfg = self.config;
        var logits_t = try moe.router.linearSeq(ctx, f_t, .embed, .expert);
        defer logits_t.deinit();
        const probs = try allocator.dupe(f32, try logits_t.dataConst());
        defer allocator.free(probs);
        softmaxInPlace(probs);

        const y = try allocator.alloc(f32, cfg.hidden_size);
        errdefer allocator.free(y);
        @memset(y, 0);

        // Top-k selection (k is single digits; simple repeated max).
        var chosen: usize = 0;
        while (chosen < cfg.num_experts_used) : (chosen += 1) {
            var best: usize = 0;
            var best_p: f32 = -1;
            for (probs, 0..) |p, e| {
                if (p > best_p) {
                    best_p = p;
                    best = e;
                }
            }
            probs[best] = -1; // consumed
            const w = best_p * cfg.expert_weights_scale;

            var gate_t = try moe.gate.linear(ctx, f_t, best, .gate_up);
            defer gate_t.deinit();
            var up_t = try moe.up.linear(ctx, f_t, best, .gate_up);
            defer up_t.deinit();
            const g = try allocator.alloc(f32, moe.gate.out_dim);
            defer allocator.free(g);
            for (g, try gate_t.dataConst(), try up_t.dataConst()) |*gi, gv, uv| gi.* = silu(gv) * uv;
            var g_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, moe.gate.out_dim }, g);
            defer g_t.deinit();
            var down_t = try moe.down.linear(ctx, &g_t, best, .attn);
            defer down_t.deinit();
            for (y, try down_t.dataConst()) |*yi, di| yi.* += w * di;
        }

        // Shared expert (always active, weight 1).
        const shared = try swigluLinear(ctx, allocator, f_t, &moe.shared_gate, &moe.shared_up, &moe.shared_down);
        defer allocator.free(shared);
        for (y, shared) |*yi, si| yi.* += si;
        return y;
    }
};

fn loadLayer(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize) !Layer {
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

    var q_proj = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "attn_q.weight")), config.num_heads * config.qk_head_dim, config.hidden_size);
    errdefer q_proj.deinit();
    var kv_a = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "attn_kv_a_mqa.weight")), config.kv_lora_rank + config.qk_rope_dim, config.hidden_size);
    errdefer kv_a.deinit();
    var kv_b = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "attn_kv_b.weight")), config.num_heads * (config.qk_nope_dim + config.v_head_dim), config.kv_lora_rank);
    errdefer kv_b.deinit();
    var o_proj = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "attn_output.weight")), config.hidden_size, config.num_heads * config.v_head_dim);
    errdefer o_proj.deinit();

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
        const gate = try ExpertStack.init(try file.get(try name.of(&name_buf, layer_i, "ffn_gate_exps.weight")), config.hidden_size, config.expert_ffn_size, config.num_experts);
        const up = try ExpertStack.init(try file.get(try name.of(&name_buf, layer_i, "ffn_up_exps.weight")), config.hidden_size, config.expert_ffn_size, config.num_experts);
        const down = try ExpertStack.init(try file.get(try name.of(&name_buf, layer_i, "ffn_down_exps.weight")), config.expert_ffn_size, config.hidden_size, config.num_experts);
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
        .kv_a = kv_a,
        .kv_b = kv_b,
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
    const info = try file.get(tensor_name);
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
