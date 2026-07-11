//! DeepSeek V4 Flash (`deepseek4` GGUF arch): 43 layers, 284B-A13B, 1M
//! context. The trunk composes, per layer: 4-stream hyper-connections with
//! Sinkhorn-normalized combine matrices; MQA attention where ONE 512-dim
//! row per position serves as both key and value (with per-head sink
//! logits, tail-64 rotary, and FP8-simulated cache rows); a raw 128-token
//! sliding window plus time-compressed KV (score-gated softmax pooling at
//! ratio 4 or 128, with an FP4/Hadamard-quantized indexer selecting the
//! top-512 compressed rows on ratio-4 layers); grouped low-rank attention
//! output; and a 256-expert MoE routed by sqrt-softplus scores — hash
//! routing from a token-id table on the first three layers — through the
//! clamped SwiGLU.
//!
//! Same correctness-first shape as the deepseek2/glm4moe ports: heavy
//! linears and the fused/streamed experts run on fucina kernels; the
//! novel glue runs host-side in auditable f32, mirroring the reference
//! implementation exactly (including its quantization round-trips, which
//! are part of the model's numerics, not an optimization).
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

pub const Config = struct {
    vocab_size: usize,
    hidden_size: usize, // 4096
    num_layers: usize, // trunk layers (block_count - nextn)
    num_nextn_layers: usize,
    num_heads: usize, // 64
    head_dim: usize, // 512 (the shared K==V row width)
    rope_dims: usize, // 64 (tail)
    n_swa: usize, // 128 raw sliding window
    indexer_heads: usize, // 64
    indexer_head_dim: usize, // 128
    indexer_top_k: usize, // 512
    q_lora_rank: usize, // 1024
    output_lora_rank: usize, // 1024
    output_groups: usize, // 8
    num_experts: usize, // 256
    num_experts_used: usize, // 6
    expert_ffn_size: usize, // 2048
    num_shared_experts: usize, // 1
    expert_weights_scale: f32, // 1.5
    hash_layers: usize, // 3
    n_hc: usize, // 4
    hc_sinkhorn_iters: usize, // 20
    hc_eps: f32,
    rms_norm_eps: f32,
    rope_theta: f32, // 10000
    compress_rope_theta: f32, // 160000
    yarn_factor: f32, // 16
    yarn_orig_ctx: usize, // 65536
    /// Per-layer compression ratio (0 = raw window only, else 4 or 128).
    compress_ratio: []u32,

    pub fn fromGguf(allocator: Allocator, file: *const gguf.File) !Config {
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        if (!std.mem.eql(u8, arch, "deepseek4")) return Error.InvalidConfig;
        const block_count = try metaInt(file, "deepseek4.block_count");
        const nextn = gguf_meta.metaIntOpt(file, "deepseek4", "nextn_predict_layers", .accept_zero) orelse 0;
        const n_layers = block_count - nextn;

        const ratios_arr = file.getArray("deepseek4.attention.compress_ratios") orelse return Error.InvalidConfig;
        if (ratios_arr.len < n_layers) return Error.InvalidConfig;
        const compress_ratio = try allocator.alloc(u32, n_layers);
        errdefer allocator.free(compress_ratio);
        for (compress_ratio, 0..) |*r, i| {
            r.* = @intCast(std.mem.readInt(i32, ratios_arr.data[i * 4 ..][0..4], .little));
            switch (r.*) {
                0, 4, 128 => {},
                else => return Error.InvalidConfig,
            }
        }

        return .{
            .vocab_size = try metaInt(file, "deepseek4.vocab_size"),
            .hidden_size = try metaInt(file, "deepseek4.embedding_length"),
            .num_layers = n_layers,
            .num_nextn_layers = nextn,
            .num_heads = try metaInt(file, "deepseek4.attention.head_count"),
            .head_dim = try metaInt(file, "deepseek4.attention.key_length"),
            .rope_dims = try metaInt(file, "deepseek4.rope.dimension_count"),
            .n_swa = try metaInt(file, "deepseek4.attention.sliding_window"),
            .indexer_heads = try metaInt(file, "deepseek4.attention.indexer.head_count"),
            .indexer_head_dim = try metaInt(file, "deepseek4.attention.indexer.key_length"),
            .indexer_top_k = try metaInt(file, "deepseek4.attention.indexer.top_k"),
            .q_lora_rank = try metaInt(file, "deepseek4.attention.q_lora_rank"),
            .output_lora_rank = try metaInt(file, "deepseek4.attention.output_lora_rank"),
            .output_groups = try metaInt(file, "deepseek4.attention.output_group_count"),
            .num_experts = try metaInt(file, "deepseek4.expert_count"),
            .num_experts_used = try metaInt(file, "deepseek4.expert_used_count"),
            .expert_ffn_size = try metaInt(file, "deepseek4.expert_feed_forward_length"),
            .num_shared_experts = try metaInt(file, "deepseek4.expert_shared_count"),
            .expert_weights_scale = metaFloat(file, "deepseek4.expert_weights_scale") orelse 1.0,
            .hash_layers = gguf_meta.metaIntOpt(file, "deepseek4", "hash_layer_count", .accept_zero) orelse 0,
            .n_hc = try metaInt(file, "deepseek4.hyper_connection.count"),
            .hc_sinkhorn_iters = try metaInt(file, "deepseek4.hyper_connection.sinkhorn_iterations"),
            .hc_eps = metaFloat(file, "deepseek4.hyper_connection.epsilon") orelse 1.0e-7,
            .rms_norm_eps = metaFloat(file, "deepseek4.attention.layer_norm_rms_epsilon") orelse 1.0e-6,
            .rope_theta = metaFloat(file, "deepseek4.rope.freq_base") orelse 10000.0,
            .compress_rope_theta = metaFloat(file, "deepseek4.attention.compress_rope_freq_base") orelse 10000.0,
            .yarn_factor = metaFloat(file, "deepseek4.rope.scaling.factor") orelse 1.0,
            .yarn_orig_ctx = gguf_meta.metaIntOpt(file, "deepseek4", "rope.scaling.original_context_length", .accept_zero) orelse 0,
            .compress_ratio = compress_ratio,
        };
    }

    fn metaInt(file: *const gguf.File, key: []const u8) !usize {
        const v = file.getInt(key) orelse return Error.InvalidConfig;
        if (v <= 0) return Error.InvalidConfig;
        return @intCast(v);
    }

    fn metaFloat(file: *const gguf.File, key: []const u8) ?f32 {
        const v = file.getFloat(key) orelse return null;
        return @floatCast(v);
    }
};

// =========================================================================
// Reference numerics: the model's own quantization round-trips. These are
// part of the graph (cache rows and indexer activations are stored through
// them), so parity requires bit-faithful ports.
// =========================================================================

fn e4m3Value(i: i32) f32 {
    const exp_scale = [16]f32{ 0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0 };
    const exp: usize = @intCast((i >> 3) & 0x0f);
    const mant: f32 = @floatFromInt(i & 0x07);
    return if (exp == 0) mant * 0.001953125 else (1.0 + mant * 0.125) * exp_scale[exp];
}

fn e4m3Round(x: f32) f32 {
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const ax = @min(@abs(x), 448.0);
    var lo: i32 = 0;
    var hi: i32 = 126;
    while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (e4m3Value(mid) <= ax) lo = mid else hi = mid - 1;
    }
    var best = lo;
    if (best < 126) {
        const best_diff = @abs(ax - e4m3Value(best));
        const next_diff = @abs(ax - e4m3Value(best + 1));
        if (next_diff < best_diff or (next_diff == best_diff and ((best + 1) & 1) == 0 and (best & 1) != 0)) {
            best += 1;
        }
    }
    return sign * e4m3Value(best);
}

/// FP8-simulate the non-rotary part of a KV row in place: per 64-dim group,
/// power-of-two scale from amax/448, clamp, e4m3 round trip.
fn fp8KvQuantRow(x: []f32, n_rot: usize) void {
    const n_nope = x.len - n_rot;
    var off: usize = 0;
    while (off < n_nope) : (off += 64) {
        var amax: f32 = 0;
        for (x[off..][0..64]) |v| amax = @max(amax, @abs(v));
        if (amax < 1.0e-4) amax = 1.0e-4;
        const scale = std.math.ldexp(@as(f32, 1.0), @intFromFloat(@ceil(@log2(amax / 448.0))));
        for (x[off..][0..64]) |*v| {
            const clamped = @min(@max(v.* / scale, -448.0), 448.0);
            v.* = e4m3Round(clamped) * scale;
        }
    }
}

fn e2m1Value(i: usize) f32 {
    const values = [8]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };
    return values[i & 7];
}

fn e2m1Round(x: f32) f32 {
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const ax = @min(@abs(x), 6.0);
    var best: usize = 0;
    var best_diff = @abs(ax - e2m1Value(0));
    for (1..8) |i| {
        const diff = @abs(ax - e2m1Value(i));
        if (diff < best_diff or (diff == best_diff and (i & 1) == 0 and (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * e2m1Value(best);
}

/// In-place 128-wide fast Walsh-Hadamard transform scaled by 1/sqrt(128).
fn hadamard128(x: *[128]f32) void {
    var stride: usize = 1;
    while (stride < 128) : (stride <<= 1) {
        var base: usize = 0;
        while (base < 128) : (base += 2 * stride) {
            for (0..stride) |i| {
                const a = x[base + i];
                const b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
    }
    const scale = 0.08838834764831845;
    for (x) |*v| v.* *= scale;
}

/// FP4-simulate an activation row: per 32-dim group, power-of-two scale from
/// amax/6, clamp, e2m1 round trip.
fn fp4ActQuantRow(x: []f32) void {
    std.debug.assert(x.len % 32 == 0);
    var off: usize = 0;
    while (off < x.len) : (off += 32) {
        var amax: f32 = 0;
        for (x[off..][0..32]) |v| amax = @max(amax, @abs(v));
        if (amax < 7.052966104933725e-38) amax = 7.052966104933725e-38;
        const scale = std.math.ldexp(@as(f32, 1.0), @intFromFloat(@ceil(@log2(amax / 6.0))));
        for (x[off..][0..32]) |*v| {
            const clamped = @min(@max(v.* / scale, -6.0), 6.0);
            v.* = e2m1Round(clamped) * scale;
        }
    }
}

/// The indexer QAT: 128-wide Hadamard rotation followed by the FP4
/// activation round trip (applies to indexer Q rows and indexer compressed
/// KV rows; without it the top-k selection is not the model's graph).
fn indexerQatRow(x: []f32) void {
    std.debug.assert(x.len == 128);
    hadamard128(x[0..128]);
    fp4ActQuantRow(x);
}

// =========================================================================
// Rotary: tail-64 rotation. Raw-window layers (ratio 0) use the plain base;
// compressed layers use the compress base with YaRN interpolation whose
// magnitude correction is cancelled (pure frequency blend).
// =========================================================================

const Rope = struct {
    /// cos/sin per (position, pair) for both layer families.
    raw_cos: []f32,
    raw_sin: []f32,
    comp_cos: []f32,
    comp_sin: []f32,
    pairs: usize,
    capacity: usize,

    fn buildFreqs(allocator: Allocator, config: Config, base: f64, yarn: bool) ![]f64 {
        const dim = config.rope_dims;
        const pairs = dim / 2;
        const inv_freq = try allocator.alloc(f64, pairs);
        for (inv_freq, 0..) |*f, i| {
            f.* = std.math.pow(f64, base, -(@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(dim))));
        }
        if (yarn and config.yarn_factor > 1.0 and config.yarn_orig_ctx > 0) {
            const orig: f64 = @floatFromInt(config.yarn_orig_ctx);
            const d_f: f64 = @floatFromInt(dim);
            const dimFor = struct {
                fn go(rot: f64, d: f64, b: f64, o: f64) f64 {
                    return d * @log(o / (rot * 2.0 * std.math.pi)) / (2.0 * @log(b));
                }
            }.go;
            var low = @floor(dimFor(32.0, d_f, base, orig));
            var high = @ceil(dimFor(1.0, d_f, base, orig));
            low = @max(low, 0);
            high = @min(high, d_f - 1);
            const factor: f64 = config.yarn_factor;
            for (inv_freq, 0..) |*f, i| {
                const extra = f.*;
                const inter = extra / factor;
                var ramp = (@as(f64, @floatFromInt(i)) - low) / @max(high - low, 0.001);
                ramp = @min(@max(ramp, 0.0), 1.0);
                const mask = 1.0 - ramp;
                f.* = inter * (1.0 - mask) + extra * mask;
            }
        }
        return inv_freq;
    }

    fn init(allocator: Allocator, config: Config, capacity: usize) !Rope {
        const pairs = config.rope_dims / 2;
        const raw_freq = try buildFreqs(allocator, config, config.rope_theta, false);
        defer allocator.free(raw_freq);
        const comp_freq = try buildFreqs(allocator, config, config.compress_rope_theta, true);
        defer allocator.free(comp_freq);

        const raw_cos = try allocator.alloc(f32, capacity * pairs);
        errdefer allocator.free(raw_cos);
        const raw_sin = try allocator.alloc(f32, capacity * pairs);
        errdefer allocator.free(raw_sin);
        const comp_cos = try allocator.alloc(f32, capacity * pairs);
        errdefer allocator.free(comp_cos);
        const comp_sin = try allocator.alloc(f32, capacity * pairs);
        for (0..capacity) |pos| {
            for (0..pairs) |i| {
                const raw_angle = @as(f64, @floatFromInt(pos)) * raw_freq[i];
                raw_cos[pos * pairs + i] = @floatCast(@cos(raw_angle));
                raw_sin[pos * pairs + i] = @floatCast(@sin(raw_angle));
                const comp_angle = @as(f64, @floatFromInt(pos)) * comp_freq[i];
                comp_cos[pos * pairs + i] = @floatCast(@cos(comp_angle));
                comp_sin[pos * pairs + i] = @floatCast(@sin(comp_angle));
            }
        }
        return .{ .raw_cos = raw_cos, .raw_sin = raw_sin, .comp_cos = comp_cos, .comp_sin = comp_sin, .pairs = pairs, .capacity = capacity };
    }

    fn deinit(self: *Rope, allocator: Allocator) void {
        allocator.free(self.raw_cos);
        allocator.free(self.raw_sin);
        allocator.free(self.comp_cos);
        allocator.free(self.comp_sin);
        self.* = undefined;
    }

    /// Rotate the TAIL `2*pairs` dims of one `head` slice at `pos`.
    /// Compressed-family layers use the blended frequencies; `inverse`
    /// un-rotates (the post-attention head correction). Pairing is
    /// half-split within the tail (adapted-ggml convention).
    fn applyTail(self: *const Rope, head: []f32, pos: usize, compressed: bool, inverse: bool) void {
        const pairs = self.pairs;
        const tail = head[head.len - 2 * pairs ..];
        const c = (if (compressed) self.comp_cos else self.raw_cos)[pos * pairs ..][0..pairs];
        const s = (if (compressed) self.comp_sin else self.raw_sin)[pos * pairs ..][0..pairs];
        for (0..pairs) |i| {
            const a = tail[i];
            const b = tail[i + pairs];
            const si = if (inverse) -s[i] else s[i];
            tail[i] = a * c[i] - b * si;
            tail[i + pairs] = a * si + b * c[i];
        }
    }
};


// =========================================================================
// Weights.
// =========================================================================

const MoeFfn = struct {
    router: LinearWeight, // f16, sqrt-softplus scores
    router_bias: ?[]f32, // exp_probs_b (top-k layers)
    tid2eid: ?[]const i32, // hash layers: [vocab][6] expert ids (borrowed)
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

const HcModule = struct {
    fn_proj: LinearWeight, // [n_hc*hidden -> 24] f16
    scale: []f32, // [3]
    base: []f32, // [24]

    fn deinit(self: *HcModule, allocator: Allocator) void {
        allocator.free(self.base);
        allocator.free(self.scale);
        self.fn_proj.deinit();
        self.* = undefined;
    }
};

const Compressor = struct {
    kv: LinearWeight, // hidden -> width
    gate: LinearWeight, // hidden -> width
    ape: []f32, // [ratio][width] additive positional embedding
    norm: []f32, // [head_dim]
    width: usize, // 2*head_dim for ratio 4, head_dim for ratio 128
    ratio: usize,

    fn deinit(self: *Compressor, allocator: Allocator) void {
        allocator.free(self.norm);
        allocator.free(self.ape);
        self.gate.deinit();
        self.kv.deinit();
        self.* = undefined;
    }
};

const Layer = struct {
    hc_attn: HcModule,
    hc_ffn: HcModule,
    attn_norm: []f32,
    ffn_norm: []f32,
    q_a: LinearWeight,
    q_a_norm: []f32,
    q_b: LinearWeight,
    kv: LinearWeight,
    kv_a_norm: []f32,
    sinks: []f32, // [heads]
    output_a: LinearWeight, // grouped low-rank (rows = groups*rank)
    output_b: LinearWeight,
    attn_compressor: ?Compressor,
    index_compressor: ?Compressor,
    indexer_q_b: ?LinearWeight, // q_lora -> idx_heads*idx_dim
    indexer_proj: ?LinearWeight, // hidden -> idx_heads
    moe: MoeFfn,

    fn deinit(self: *Layer, allocator: Allocator) void {
        self.moe.deinit(allocator);
        if (self.indexer_proj) |*w| w.deinit();
        if (self.indexer_q_b) |*w| w.deinit();
        if (self.index_compressor) |*c| c.deinit(allocator);
        if (self.attn_compressor) |*c| c.deinit(allocator);
        self.output_b.deinit();
        self.output_a.deinit();
        allocator.free(self.sinks);
        allocator.free(self.kv_a_norm);
        self.kv.deinit();
        allocator.free(self.q_a_norm);
        self.q_a.deinit();
        allocator.free(self.ffn_norm);
        allocator.free(self.attn_norm);
        self.hc_ffn.deinit(allocator);
        self.hc_attn.deinit(allocator);
        self.* = undefined;
    }
};

// =========================================================================
// Session state: raw sliding-window ring + compressed streams + compressor
// frontiers, per layer. All rows live FP8/FP4-simulated, exactly like the
// reference cache.
// =========================================================================

const LayerCache = struct {
    /// Raw ring, chronological order maintained by shifting (window is only
    /// 128 rows; a memmove per token is cheaper than ring index juggling).
    raw: []f32, // [n_swa][head_dim]
    n_raw: usize = 0,
    comp: std.ArrayList(f32) = .empty, // [n][head_dim]
    index_comp: std.ArrayList(f32) = .empty, // [n][idx_dim] (ratio-4)
    attn_state_kv: []f32 = &.{},
    attn_state_score: []f32 = &.{},
    index_state_kv: []f32 = &.{},
    index_state_score: []f32 = &.{},
};

pub const Cache = struct {
    allocator: Allocator,
    layers: []LayerCache,
    len: usize = 0,
    capacity: usize,

    pub fn deinit(self: *Cache) void {
        for (self.layers) |*lc| {
            self.allocator.free(lc.raw);
            lc.comp.deinit(self.allocator);
            lc.index_comp.deinit(self.allocator);
            if (lc.attn_state_kv.len > 0) self.allocator.free(lc.attn_state_kv);
            if (lc.attn_state_score.len > 0) self.allocator.free(lc.attn_state_score);
            if (lc.index_state_kv.len > 0) self.allocator.free(lc.index_state_kv);
            if (lc.index_state_score.len > 0) self.allocator.free(lc.index_state_score);
        }
        self.allocator.free(self.layers);
        self.* = undefined;
    }
};

pub const Model = struct {
    allocator: Allocator,
    config: Config,
    token_embedding: LinearWeight,
    output_hc: HcModule, // fn [16384->4], scale [1], base [4]
    output_norm: []f32,
    output: LinearWeight,
    layers: []Layer,
    rope: Rope,
    attn_scale: f32,
    weight_mapping: ?gguf.File.MappedRegion = null,
    expert_store: ?*fucina.ExpertStore = null,

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

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, options: LoadOptions) !Model {
        var file = try gguf.File.loadMmapAuto(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFileOptions(ctx, &file, options);
    }

    pub fn loadGgufFromFileOptions(ctx: *ExecContext, file: *gguf.File, options: LoadOptions) !Model {
        const allocator = ctx.allocator;
        const config = try Config.fromGguf(allocator, file);
        errdefer allocator.free(config.compress_ratio);

        var expert_store: ?*fucina.ExpertStore = null;
        if (options.moe_stream) |so| {
            if (config.num_experts > 0) {
                var one_path = [_][]const u8{so.gguf_path};
                expert_store = try fucina.ExpertStore.create(allocator, &one_path, config.num_layers, .{
                    .cache_bytes = so.cache_bytes,
                    .cache_slots_per_layer = so.cache_slots_per_layer,
                    .readahead = so.readahead,
                    .auto_pin = so.auto_pin,
                    .pin_bytes = so.pin_bytes,
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
        var output_hc = HcModule{
            .fn_proj = try LinearWeight.load(ctx, try file.get("output_hc_fn.weight"), config.n_hc, config.n_hc * config.hidden_size),
            .scale = try hostVector(allocator, file, "output_hc_scale.weight", 1),
            .base = try hostVector(allocator, file, "output_hc_base.weight", config.n_hc),
        };
        errdefer output_hc.deinit(allocator);

        const layers = try allocator.alloc(Layer, config.num_layers);
        errdefer allocator.free(layers);
        var built: usize = 0;
        errdefer for (layers[0..built]) |*l| l.deinit(allocator);
        for (layers, 0..) |*layer, i| {
            layer.* = try loadLayer(ctx, file, config, i, expert_store);
            built += 1;
        }
        if (expert_store) |store| try store.finalize();

        const weight_mapping = file.takeMapping();
        if (weight_mapping == null) return Error.InvalidWeightShape;

        var rope = try Rope.init(allocator, config, max_positions_default);
        errdefer rope.deinit(allocator);

        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = token_embedding,
            .output_hc = output_hc,
            .output_norm = output_norm,
            .output = output,
            .layers = layers,
            .rope = rope,
            .attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.head_dim))),
            .weight_mapping = weight_mapping,
            .expert_store = expert_store,
        };
    }

    pub fn deinit(self: *Model) void {
        self.rope.deinit(self.allocator);
        for (self.layers) |*l| l.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.output_hc.deinit(self.allocator);
        self.allocator.free(self.output_norm);
        self.output.deinit();
        self.token_embedding.deinit();
        self.allocator.free(self.config.compress_ratio);
        if (self.expert_store) |store| store.destroy();
        if (self.weight_mapping) |*mapping| mapping.deinit();
        self.* = undefined;
    }

    pub fn initCache(self: *const Model, capacity: usize) !Cache {
        const cfg = self.config;
        const allocator = self.allocator;
        const layers = try allocator.alloc(LayerCache, cfg.num_layers);
        var built: usize = 0;
        errdefer {
            for (layers[0..built]) |*lc| allocator.free(lc.raw);
            allocator.free(layers);
        }
        for (layers, 0..) |*lc, i| {
            lc.* = .{ .raw = try allocator.alloc(f32, cfg.n_swa * cfg.head_dim) };
            const ratio = cfg.compress_ratio[i];
            if (ratio != 0) {
                // Ratio-4 keeps an overlapped double window (2x rows of 2x
                // width); ratio-128 a plain window.
                const rows: usize = if (ratio == 4) 2 * ratio else ratio;
                const width: usize = if (ratio == 4) 2 * cfg.head_dim else cfg.head_dim;
                lc.attn_state_kv = try allocator.alloc(f32, rows * width);
                lc.attn_state_score = try allocator.alloc(f32, rows * width);
            }
            if (ratio == 4) {
                const iw = 2 * cfg.indexer_head_dim;
                lc.index_state_kv = try allocator.alloc(f32, 2 * ratio * iw);
                lc.index_state_score = try allocator.alloc(f32, 2 * ratio * iw);
            }
            built += 1;
        }
        return .{ .allocator = allocator, .layers = layers, .capacity = capacity };
    }
};

const max_positions_default: usize = 65536;

fn loadLayer(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize, store: ?*fucina.ExpertStore) !Layer {
    const allocator = ctx.allocator;
    var buf: [96]u8 = undefined;
    const name = struct {
        fn of(b: []u8, i: usize, suffix: []const u8) ![]const u8 {
            return std.fmt.bufPrint(b, "blk.{d}.{s}", .{ i, suffix });
        }
    };
    const hidden = config.hidden_size;
    const ratio = config.compress_ratio[layer_i];

    var hc_attn = HcModule{
        .fn_proj = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "hc_attn_fn.weight")), 6 * config.n_hc, config.n_hc * hidden),
        .scale = try hostVector(allocator, file, try name.of(&buf, layer_i, "hc_attn_scale.weight"), 3),
        .base = try hostVector(allocator, file, try name.of(&buf, layer_i, "hc_attn_base.weight"), 6 * config.n_hc),
    };
    errdefer hc_attn.deinit(allocator);
    var hc_ffn = HcModule{
        .fn_proj = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "hc_ffn_fn.weight")), 6 * config.n_hc, config.n_hc * hidden),
        .scale = try hostVector(allocator, file, try name.of(&buf, layer_i, "hc_ffn_scale.weight"), 3),
        .base = try hostVector(allocator, file, try name.of(&buf, layer_i, "hc_ffn_base.weight"), 6 * config.n_hc),
    };
    errdefer hc_ffn.deinit(allocator);

    const attn_norm = try hostVector(allocator, file, try name.of(&buf, layer_i, "attn_norm.weight"), hidden);
    errdefer allocator.free(attn_norm);
    const ffn_norm = try hostVector(allocator, file, try name.of(&buf, layer_i, "ffn_norm.weight"), hidden);
    errdefer allocator.free(ffn_norm);

    var q_a = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "attn_q_a.weight")), config.q_lora_rank, hidden);
    errdefer q_a.deinit();
    const q_a_norm = try hostVector(allocator, file, try name.of(&buf, layer_i, "attn_q_a_norm.weight"), config.q_lora_rank);
    errdefer allocator.free(q_a_norm);
    var q_b = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "attn_q_b.weight")), config.num_heads * config.head_dim, config.q_lora_rank);
    errdefer q_b.deinit();
    var kv = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "attn_kv.weight")), config.head_dim, hidden);
    errdefer kv.deinit();
    const kv_a_norm = try hostVector(allocator, file, try name.of(&buf, layer_i, "attn_kv_a_norm.weight"), config.head_dim);
    errdefer allocator.free(kv_a_norm);
    const sinks = try hostVector(allocator, file, try name.of(&buf, layer_i, "attn_sinks.weight"), config.num_heads);
    errdefer allocator.free(sinks);
    var output_a = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "attn_output_a.weight")), config.output_groups * config.output_lora_rank, config.num_heads * config.head_dim);
    errdefer output_a.deinit();
    var output_b = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "attn_output_b.weight")), hidden, config.output_groups * config.output_lora_rank);
    errdefer output_b.deinit();

    var attn_compressor: ?Compressor = null;
    errdefer if (attn_compressor) |*c| c.deinit(allocator);
    var index_compressor: ?Compressor = null;
    errdefer if (index_compressor) |*c| c.deinit(allocator);
    var indexer_q_b: ?LinearWeight = null;
    errdefer if (indexer_q_b) |*w| w.deinit();
    var indexer_proj: ?LinearWeight = null;
    errdefer if (indexer_proj) |*w| w.deinit();
    if (ratio != 0) {
        const width: usize = if (ratio == 4) 2 * config.head_dim else config.head_dim;
        attn_compressor = .{
            .kv = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "attn_compressor_kv.weight")), width, hidden),
            .gate = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "attn_compressor_gate.weight")), width, hidden),
            .ape = try hostMatrix(allocator, file, try name.of(&buf, layer_i, "attn_compressor_ape.weight"), width, ratio),
            .norm = try hostVector(allocator, file, try name.of(&buf, layer_i, "attn_compressor_norm.weight"), config.head_dim),
            .width = width,
            .ratio = ratio,
        };
    }
    if (ratio == 4) {
        const iw = 2 * config.indexer_head_dim;
        index_compressor = .{
            .kv = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "indexer_compressor_kv.weight")), iw, hidden),
            .gate = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "indexer_compressor_gate.weight")), iw, hidden),
            .ape = try hostMatrix(allocator, file, try name.of(&buf, layer_i, "indexer_compressor_ape.weight"), iw, ratio),
            .norm = try hostVector(allocator, file, try name.of(&buf, layer_i, "indexer_compressor_norm.weight"), config.indexer_head_dim),
            .width = iw,
            .ratio = ratio,
        };
        indexer_q_b = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "indexer.attn_q_b.weight")), config.indexer_heads * config.indexer_head_dim, config.q_lora_rank);
        indexer_proj = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "indexer.proj.weight")), config.indexer_heads, hidden);
    }

    // MoE: router f16 + hash table on the leading layers + bias later on.
    var router = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "ffn_gate_inp.weight")), config.num_experts, hidden);
    errdefer router.deinit();
    var router_bias: ?[]f32 = null;
    errdefer if (router_bias) |b| allocator.free(b);
    var bias_buf: [96]u8 = undefined;
    if (file.maybeGet(try name.of(&bias_buf, layer_i, "exp_probs_b.bias"))) |bias_info| {
        router_bias = try hostVectorInfo(allocator, bias_info, config.num_experts);
    }
    var tid2eid: ?[]const i32 = null;
    if (file.maybeGet(try name.of(&bias_buf, layer_i, "ffn_gate_tid2eid.weight"))) |tid_info| {
        if (tid_info.n_dims != 2 or tid_info.dims[0] != config.num_experts_used or tid_info.dims[1] != config.vocab_size) return Error.InvalidWeightShape;
        const raw = std.mem.bytesAsSlice(i32, @as([]align(4) const u8, @alignCast(tid_info.data)));
        tid2eid = raw;
    }
    if ((layer_i < config.hash_layers) != (tid2eid != null)) return Error.InvalidWeightShape;

    var gate: fucina.MoeRhs = undefined;
    var up: fucina.MoeRhs = undefined;
    var down: fucina.MoeRhs = undefined;
    if (store) |st| {
        const trio = try weights.loadMoeRhsStreamed(st, file, layer_i, try file.get(try name.of(&buf, layer_i, "ffn_gate_exps.weight")), try file.get(try name.of(&buf, layer_i, "ffn_up_exps.weight")), try file.get(try name.of(&buf, layer_i, "ffn_down_exps.weight")), hidden, config.expert_ffn_size, config.num_experts);
        gate = trio.gate;
        up = trio.up;
        down = trio.down;
    } else {
        const borrow = file.is_mmap and !file.isSplit();
        gate = try weights.loadMoeRhs(ctx, try file.get(try name.of(&buf, layer_i, "ffn_gate_exps.weight")), hidden, config.expert_ffn_size, config.num_experts, borrow);
        up = try weights.loadMoeRhs(ctx, try file.get(try name.of(&buf, layer_i, "ffn_up_exps.weight")), hidden, config.expert_ffn_size, config.num_experts, borrow);
        down = try weights.loadMoeRhs(ctx, try file.get(try name.of(&buf, layer_i, "ffn_down_exps.weight")), config.expert_ffn_size, hidden, config.num_experts, borrow);
    }
    const shared_ffn = config.expert_ffn_size * config.num_shared_experts;
    var shared_gate = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "ffn_gate_shexp.weight")), shared_ffn, hidden);
    errdefer shared_gate.deinit();
    var shared_up = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "ffn_up_shexp.weight")), shared_ffn, hidden);
    errdefer shared_up.deinit();
    var shared_down = try LinearWeight.load(ctx, try file.get(try name.of(&buf, layer_i, "ffn_down_shexp.weight")), hidden, shared_ffn);
    errdefer shared_down.deinit();

    return .{
        .hc_attn = hc_attn,
        .hc_ffn = hc_ffn,
        .attn_norm = attn_norm,
        .ffn_norm = ffn_norm,
        .q_a = q_a,
        .q_a_norm = q_a_norm,
        .q_b = q_b,
        .kv = kv,
        .kv_a_norm = kv_a_norm,
        .sinks = sinks,
        .output_a = output_a,
        .output_b = output_b,
        .attn_compressor = attn_compressor,
        .index_compressor = index_compressor,
        .indexer_q_b = indexer_q_b,
        .indexer_proj = indexer_proj,
        .moe = .{
            .router = router,
            .router_bias = router_bias,
            .tid2eid = tid2eid,
            .gate = gate,
            .up = up,
            .down = down,
            .shared_gate = shared_gate,
            .shared_up = shared_up,
            .shared_down = shared_down,
        },
    };
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

/// 2D host matrix as [rows][cols] f32 (GGUF dims [cols, rows]).
fn hostMatrix(allocator: Allocator, file: *const gguf.File, tensor_name: []const u8, cols: usize, rows: usize) ![]f32 {
    const info = try file.get(tensor_name);
    if (info.n_dims != 2 or info.dims[0] != cols or info.dims[1] != rows) return Error.InvalidWeightShape;
    const out = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(out);
    try weights.fillF32(out, info);
    return out;
}

fn rmsNormInto(out: []f32, x: []const f32, weight: ?[]const f32, eps: f32) void {
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, v) * v;
    const inv = 1.0 / @sqrt(sum / @as(f64, @floatFromInt(x.len)) + eps);
    if (weight) |w| {
        for (out, x, w) |*o, v, wv| o.* = @floatCast(@as(f64, v) * inv * wv);
    } else {
        for (out, x) |*o, v| o.* = @floatCast(@as(f64, v) * inv);
    }
}

fn sigmoidStable(x: f32) f32 {
    if (x >= 0) {
        const e = @exp(-x);
        return 1.0 / (1.0 + e);
    }
    const e = @exp(x);
    return e / (1.0 + e);
}

fn softplusStable(x: f32) f32 {
    // log(1 + e^x), sign-stable.
    return if (x > 0) x + std.math.log1p(@exp(-x)) else std.math.log1p(@exp(x));
}

test {
    // Numerics sanity: Hadamard is an involution up to scale; e4m3 grid is
    // monotone; fp8/fp4 round trips are idempotent.
    var v: [128]f32 = undefined;
    for (&v, 0..) |*x, i| x.* = @floatFromInt(@as(i32, @intCast(i % 17)) - 8);
    var w = v;
    hadamard128(&w);
    hadamard128(&w);
    for (v, w) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-4);

    var prev: f32 = -1;
    for (0..127) |i| {
        const val = e4m3Value(@intCast(i));
        try std.testing.expect(val > prev);
        prev = val;
    }

    var row: [64]f32 = undefined;
    for (&row, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i))) * 100.0;
    fp8KvQuantRow(&row, 0);
    var again = row;
    fp8KvQuantRow(&again, 0);
    for (row, again) |a, b| try std.testing.expectEqual(a, b);
}
