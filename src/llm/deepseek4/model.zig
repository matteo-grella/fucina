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
fn f16Round(v: f32) f32 {
    return @floatCast(@as(f16, @floatCast(v)));
}

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
    /// ADJACENT within the tail — (tail[2i], tail[2i+1]) shares frequency
    /// i — matching the reference loop, not the half-split convention the
    /// other DeepSeek-family ports use.
    fn applyTail(self: *const Rope, head: []f32, pos: usize, compressed: bool, inverse: bool) void {
        const pairs = self.pairs;
        const tail = head[head.len - 2 * pairs ..];
        const c = (if (compressed) self.comp_cos else self.raw_cos)[pos * pairs ..][0..pairs];
        const s = (if (compressed) self.comp_sin else self.raw_sin)[pos * pairs ..][0..pairs];
        for (0..pairs) |i| {
            const a = tail[2 * i];
            const b = tail[2 * i + 1];
            const si = if (inverse) -s[i] else s[i];
            tail[2 * i] = a * c[i] - b * si;
            tail[2 * i + 1] = a * si + b * c[i];
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
    /// Grouped low-rank output stage A: raw q8_0 rows, `groups*rank` rows of
    /// `group_dim` (borrowed from the mapping; per-group row-block slices).
    output_a: []const u8,
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
        allocator.free(self.sinks);
        allocator.free(self.kv_a_norm);
        self.kv.deinit();
        self.q_b.deinit();
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
            @memset(lc.raw, 0);
            const ratio = cfg.compress_ratio[i];
            if (ratio != 0) {
                // Ratio-4 keeps an overlapped double window (2x rows of 2x
                // width); ratio-128 a plain window. Unwritten window slots
                // must pool as empty: kv 0, score -inf (the pooling max
                // guard skips them) — exactly the reference cache init.
                const rows: usize = if (ratio == 4) 2 * ratio else ratio;
                const width: usize = if (ratio == 4) 2 * cfg.head_dim else cfg.head_dim;
                lc.attn_state_kv = try allocator.alloc(f32, rows * width);
                @memset(lc.attn_state_kv, 0);
                lc.attn_state_score = try allocator.alloc(f32, rows * width);
                @memset(lc.attn_state_score, -std.math.inf(f32));
            }
            if (ratio == 4) {
                const iw = 2 * cfg.indexer_head_dim;
                lc.index_state_kv = try allocator.alloc(f32, 2 * ratio * iw);
                @memset(lc.index_state_kv, 0);
                lc.index_state_score = try allocator.alloc(f32, 2 * ratio * iw);
                @memset(lc.index_state_score, -std.math.inf(f32));
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
    // Grouped low-rank output stage A: 8 stacked [rank x group_dim] blocks
    // ([8192 x 4096] on Flash) applied per head-group; kept as the raw q8_0
    // byte slice so each group runs through the borrowed-quantized linear.
    const output_a_info = try file.get(try name.of(&buf, layer_i, "attn_output_a.weight"));
    const group_dim = (config.num_heads / config.output_groups) * config.head_dim;
    if (output_a_info.n_dims != 2 or output_a_info.dims[0] != group_dim or output_a_info.dims[1] != config.output_groups * config.output_lora_rank) return Error.InvalidWeightShape;
    if (output_a_info.ggml_type != .q8_0) return Error.UnsupportedWeightType;
    const output_a = output_a_info.data;
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


// =========================================================================
// Forward pass. Tensor ops carry everything expressible in the public API;
// host f32 remains only for the reference's bit-exact quantization grids,
// the 4x4 Sinkhorn iteration, the tail rotary, and the sliding-ring
// bookkeeping (whose rows live pre-quantized between steps by design).
// =========================================================================

const HcSplit = struct {
    pre: [4]f32,
    post: [4]f32,
    comb: [16]f32, // [dst + src*4]
};

fn hcSplitSinkhorn(mix: []const f32, scale: []const f32, base: []const f32, n_hc: usize, iters: usize) HcSplit {
    std.debug.assert(n_hc == 4 and mix.len == 24);
    var out: HcSplit = undefined;
    const eps: f32 = 1.0e-6;
    for (0..n_hc) |i| out.pre[i] = sigmoidStable(mix[i] * scale[0] + base[i]) + eps;
    for (0..n_hc) |i| out.post[i] = 2.0 * sigmoidStable(mix[n_hc + i] * scale[1] + base[n_hc + i]);

    var c: [16]f32 = undefined;
    for (0..n_hc) |dst| {
        var row_max = -std.math.inf(f32);
        for (0..n_hc) |src| {
            const v = mix[2 * n_hc + src + dst * n_hc] * scale[2] + base[2 * n_hc + src + dst * n_hc];
            c[src + dst * n_hc] = v;
            row_max = @max(row_max, v);
        }
        var row_sum: f32 = 0;
        for (0..n_hc) |src| {
            const v = @exp(c[src + dst * n_hc] - row_max);
            c[src + dst * n_hc] = v;
            row_sum += v;
        }
        for (0..n_hc) |src| c[src + dst * n_hc] = c[src + dst * n_hc] / row_sum + eps;
    }
    // Sinkhorn: first a column pass, then alternating row/column passes.
    for (0..n_hc) |src| {
        var sum: f32 = 0;
        for (0..n_hc) |dst| sum += c[src + dst * n_hc];
        const inv = 1.0 / (sum + eps);
        for (0..n_hc) |dst| c[src + dst * n_hc] *= inv;
    }
    for (1..iters) |_| {
        for (0..n_hc) |dst| {
            var sum: f32 = 0;
            for (0..n_hc) |src| sum += c[src + dst * n_hc];
            const inv = 1.0 / (sum + eps);
            for (0..n_hc) |src| c[src + dst * n_hc] *= inv;
        }
        for (0..n_hc) |src| {
            var sum: f32 = 0;
            for (0..n_hc) |dst| sum += c[src + dst * n_hc];
            const inv = 1.0 / (sum + eps);
            for (0..n_hc) |dst| c[src + dst * n_hc] *= inv;
        }
    }
    out.comb = c;
    return out;
}

pub const StepScratch = struct {
    /// HC stream state [n_hc * hidden], persistent across layers of one step.
    streams: []f32,

    pub fn init(allocator: Allocator, config: Config) !StepScratch {
        return .{ .streams = try allocator.alloc(f32, config.n_hc * config.hidden_size) };
    }

    pub fn deinit(self: *StepScratch, allocator: Allocator) void {
        allocator.free(self.streams);
        self.* = undefined;
    }
};



// (continued in Model.step below)

fn hcPre(
    ctx: *ExecContext,
    config: Config,
    module: *const HcModule,
    streams: []const f32,
) !struct { sub_in: []f32, split: HcSplit } {
    const allocator = ctx.allocator;
    const hc_dim = config.n_hc * config.hidden_size;

    // flat = rms_norm_no_weight(streams) over the full 4*hidden vector.
    var flat_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, hc_dim }, streams);
    defer flat_t.deinit();
    var normed_t = try flat_t.rmsNorm(ctx, .embed, config.rms_norm_eps);
    defer normed_t.deinit();
    var mix_t = try module.fn_proj.linearSeq(ctx, &normed_t, .embed, .attn);
    defer mix_t.deinit();
    const split = hcSplitSinkhorn(try mix_t.dataConst(), module.scale, module.base, config.n_hc, config.hc_sinkhorn_iters);

    // sublayer input = sum_h pre[h] * stream_h : [stream, embed] x [stream].
    var streams_t = try fucina.Tensor(.{ .stream, .embed }).fromSlice(ctx, .{ config.n_hc, config.hidden_size }, streams);
    defer streams_t.deinit();
    var pre_t = try fucina.Tensor(.{.stream}).fromSlice(ctx, .{config.n_hc}, &split.pre);
    defer pre_t.deinit();
    var weighted = try streams_t.mul(ctx, &pre_t);
    defer weighted.deinit();
    var summed = try weighted.sum(ctx, .stream);
    defer summed.deinit();
    const sub_in = try allocator.dupe(f32, try summed.dataConst());
    return .{ .sub_in = sub_in, .split = split };
}

fn hcPost(
    ctx: *ExecContext,
    config: Config,
    split: *const HcSplit,
    block_out: []const f32,
    streams: []f32, // residual in, next state out
) !void {
    // streams'[dst] = post[dst]*block_out + sum_src comb[dst + src*4]*streams[src]
    var streams_t = try fucina.Tensor(.{ .stream, .embed }).fromSlice(ctx, .{ config.n_hc, config.hidden_size }, streams);
    defer streams_t.deinit();
    var comb_t = try fucina.Tensor(.{ .stream_dst, .stream }).fromSlice(ctx, .{ config.n_hc, config.n_hc }, blk: {
        // comb is addressed [dst + src*n_hc]; lay it out row-major [dst][src].
        var host: [16]f32 = undefined;
        for (0..config.n_hc) |dst| {
            for (0..config.n_hc) |src| host[dst * config.n_hc + src] = split.comb[dst + src * config.n_hc];
        }
        break :blk &host;
    });
    defer comb_t.deinit();
    var mixed = try comb_t.dot(ctx, &streams_t, .stream);
    defer mixed.deinit();
    var out_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, config.hidden_size }, block_out);
    defer out_t.deinit();
    var post_t = try fucina.Tensor(.{.stream_dst}).fromSlice(ctx, .{config.n_hc}, &split.post);
    defer post_t.deinit();
    var injected = try out_t.mul(ctx, &post_t);
    defer injected.deinit();
    var next = try mixed.add(ctx, &injected);
    defer next.deinit();
    @memcpy(streams, try next.dataConst());
}

// =========================================================================
// Sublayer blocks + the step orchestration.
// =========================================================================

fn embedTag(ctx: *ExecContext, host: []const f32) !fucina.Tensor(.{.embed}) {
    return fucina.Tensor(.{.embed}).fromSlice(ctx, .{host.len}, host);
}

fn rowTensor(ctx: *ExecContext, host: []const f32, width: usize) !fucina.Tensor(.{ .seq, .embed }) {
    return fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ host.len / width, width }, host);
}

/// The reference's streaming compressor for one token: project, add the
/// window-slot APE, update the rolling frontier, and emit a pooled row on
/// ratio boundaries (rope'd at the compressed position and quantized
/// through the family grid). Returns true when a row was appended.
fn compressorStep(
    self: *const Model,
    ctx: *ExecContext,
    comp: *const Compressor,
    attn_norm_t: *const fucina.Tensor(.{ .seq, .embed }),
    state_kv: []f32,
    state_score: []f32,
    out_rows: *std.ArrayList(f32),
    head_dim: usize,
    layer_compressed: bool,
    pos: usize,
) !bool {
    var kv_t = try comp.kv.linearSeq(ctx, attn_norm_t, .embed, .attn);
    defer kv_t.deinit();
    var sc_t = try comp.gate.linearSeq(ctx, attn_norm_t, .embed, .attn);
    defer sc_t.deinit();
    return compressorAdvance(self, ctx, comp, try kv_t.dataConst(), try sc_t.dataConst(), state_kv, state_score, out_rows, head_dim, layer_compressed, pos);
}

/// State-advance half of the streaming compressor: consume this token's
/// already-projected kv/score rows (so batched prefill can compute all
/// projections in one GEMM), update the window, emit on the boundary.
fn compressorAdvance(
    self: *const Model,
    ctx: *ExecContext,
    comp: *const Compressor,
    kv_cur: []const f32,
    sc_cur: []const f32,
    state_kv: []f32,
    state_score: []f32,
    out_rows: *std.ArrayList(f32),
    head_dim: usize,
    layer_compressed: bool,
    pos: usize,
) !bool {
    const allocator = ctx.allocator;
    const ratio = comp.ratio;
    const width = comp.width;
    const pos_mod = pos % ratio;
    const row: usize = if (ratio == 4) ratio + pos_mod else pos_mod;
    const should_compress = ((pos + 1) % ratio) == 0;

    const kv_dst = state_kv[row * width ..][0..width];
    const sc_dst = state_score[row * width ..][0..width];
    @memcpy(kv_dst, kv_cur);
    const ape_row = comp.ape[pos_mod * width ..][0..width];
    for (sc_dst, sc_cur, ape_row) |*d, v, a| d.* = v + a;

    if (!should_compress) return false;

    // Score-gated per-dim softmax pooling over the window (ratio-4 pools the
    // overlapped prev/cur halves; see the port notes for the addressing).
    const pooled = try allocator.alloc(f32, head_dim);
    defer allocator.free(pooled);
    for (0..head_dim) |j| {
        var max_score = -std.math.inf(f32);
        if (ratio == 4) {
            for (0..ratio) |r| {
                max_score = @max(max_score, state_score[r * width + j]);
                max_score = @max(max_score, state_score[(ratio + r) * width + head_dim + j]);
            }
        } else {
            for (0..ratio) |r| max_score = @max(max_score, state_score[r * width + j]);
        }
        if (max_score <= -std.math.inf(f32) * 0.5) {
            pooled[j] = 0;
            continue;
        }
        var denom: f32 = 0;
        var sum: f32 = 0;
        if (ratio == 4) {
            for (0..ratio) |r| {
                const wp = @exp(state_score[r * width + j] - max_score);
                const wc = @exp(state_score[(ratio + r) * width + head_dim + j] - max_score);
                denom += wp + wc;
                sum += wp * state_kv[r * width + j];
                sum += wc * state_kv[(ratio + r) * width + head_dim + j];
            }
        } else {
            for (0..ratio) |r| {
                const w = @exp(state_score[r * width + j] - max_score);
                denom += w;
                sum += w * state_kv[r * width + j];
            }
        }
        pooled[j] = if (denom > 0) sum / denom else 0;
    }

    // rms-norm with the compressor weight, rope at the compressed position,
    // then the family quantization round-trip.
    const out_row = try out_rows.addManyAsSlice(ctx.allocator, head_dim);
    rmsNormInto(out_row, pooled, comp.norm, self.config.rms_norm_eps);
    const comp_pos = pos + 1 - ratio;
    self.rope.applyTail(out_row, comp_pos, layer_compressed, false);
    if (head_dim == self.config.head_dim) {
        fp8KvQuantRow(out_row, self.config.rope_dims);
    } else {
        indexerQatRow(out_row);
    }
    // Cached rows live f16-rounded (the reference cache stores f16).
    for (out_row) |*v| v.* = f16Round(v.*);

    if (ratio == 4) {
        // Shift: cur half becomes prev, then mirrored back (reference-exact).
        for (0..ratio) |r| {
            @memcpy(state_kv[r * width ..][0..width], state_kv[(ratio + r) * width ..][0..width]);
            @memcpy(state_score[r * width ..][0..width], state_score[(ratio + r) * width ..][0..width]);
        }
        for (0..ratio) |r| {
            @memcpy(state_kv[(ratio + r) * width ..][0..width], state_kv[r * width ..][0..width]);
            @memcpy(state_score[(ratio + r) * width ..][0..width], state_score[r * width ..][0..width]);
        }
    }
    return true;
}

pub const Session = struct {
    cache: Cache,
    scratch: StepScratch,

    pub fn init(model: *const Model, capacity: usize) !Session {
        var cache = try model.initCache(capacity);
        errdefer cache.deinit();
        const scratch = try StepScratch.init(model.allocator, model.config);
        return .{ .cache = cache, .scratch = scratch };
    }

    pub fn deinit(self: *Session, model: *const Model) void {
        self.scratch.deinit(model.allocator);
        self.cache.deinit();
        self.* = undefined;
    }
};

pub fn step(self: *Model, ctx: *ExecContext, session: *Session, token: usize) ![]f32 {
    const cfg = self.config;
    const allocator = ctx.allocator;
    const cache = &session.cache;
    if (cache.len >= cache.capacity or cache.len >= self.rope.capacity) return Error.KvCacheOverflow;
    const pos = cache.len;
    const streams = session.scratch.streams;

    // All HC streams start as the token embedding.
    {
        var ids = [_]usize{token};
        var emb = try self.token_embedding.getRowsAs(ctx, &ids, .embed);
        defer emb.deinit();
        const row = try emb.dataConst();
        for (0..cfg.n_hc) |h| @memcpy(streams[h * cfg.hidden_size ..][0..cfg.hidden_size], row);
    }

    for (self.layers, 0..) |*layer, layer_i| {
        // ---- attention sublayer ----
        {
            const pre = try hcPre(ctx, cfg, &layer.hc_attn, streams);
            defer allocator.free(pre.sub_in);
            const block_out = try attnBlock(self, ctx, cache, layer, layer_i, pre.sub_in, pos, token);
            defer allocator.free(block_out);
            try hcPost(ctx, cfg, &pre.split, block_out, streams);
        }
        // ---- FFN sublayer ----
        {
            const pre = try hcPre(ctx, cfg, &layer.hc_ffn, streams);
            defer allocator.free(pre.sub_in);
            const block_out = try moeBlock(self, ctx, layer, pre.sub_in, token);
            defer allocator.free(block_out);
            try hcPost(ctx, cfg, &pre.split, block_out, streams);
        }
    }
    cache.len += 1;
    return outputLogits(self, ctx, streams);
}

/// Output: sigmoid-gated HC merge, output norm, vocab head — for the stream
/// state of one position.
fn outputLogits(self: *Model, ctx: *ExecContext, streams: []const f32) ![]f32 {
    const cfg = self.config;
    const allocator = ctx.allocator;
    const hc_dim = cfg.n_hc * cfg.hidden_size;
    var flat_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ 1, hc_dim }, streams);
    defer flat_t.deinit();
    var flat_norm = try flat_t.rmsNorm(ctx, .embed, cfg.rms_norm_eps);
    defer flat_norm.deinit();
    var pre_t = try self.output_hc.fn_proj.linearSeq(ctx, &flat_norm, .embed, .attn);
    defer pre_t.deinit();
    const pre_vals = try pre_t.dataConst();
    var merge_w: [4]f32 = undefined;
    for (0..cfg.n_hc) |i| merge_w[i] = sigmoidStable(pre_vals[i] * self.output_hc.scale[0] + self.output_hc.base[i]) + cfg.hc_eps;

    var streams_t = try fucina.Tensor(.{ .stream, .embed }).fromBorrowedConstSlice(ctx, .{ cfg.n_hc, cfg.hidden_size }, streams);
    defer streams_t.deinit();
    var w_t = try fucina.Tensor(.{.stream}).fromSlice(ctx, .{cfg.n_hc}, merge_w[0..cfg.n_hc]);
    defer w_t.deinit();
    var weighted = try streams_t.mul(ctx, &w_t);
    defer weighted.deinit();
    var merged = try weighted.sum(ctx, .stream);
    defer merged.deinit();

    var norm_w = try embedTag(ctx, self.output_norm);
    defer norm_w.deinit();
    var merged_row = try rowTensor(ctx, try merged.dataConst(), cfg.hidden_size);
    defer merged_row.deinit();
    var final_norm = try merged_row.rmsNormMul(ctx, .embed, &norm_w, cfg.rms_norm_eps);
    defer final_norm.deinit();
    var logits_t = try self.output.linearSeq(ctx, &final_norm, .embed, .vocab);
    defer logits_t.deinit();
    return allocator.dupe(f32, try logits_t.dataConst());
}

/// Batched prefill: one chunk of positions through all layers with S-row
/// GEMMs and union-routed expert fetches (the fused batch MoE path fetches
/// each expert once per layer per chunk — the whole point for streamed
/// weights). Per-token state (compressor windows, raw ring, indexer
/// visibility) advances sequentially inside each layer exactly as decode
/// would, so a batched prefill leaves the caches in the same state as the
/// equivalent sequence of single steps. Returns the LAST position's logits.
pub fn stepBatch(self: *Model, ctx: *ExecContext, session: *Session, tokens: []const usize) ![]f32 {
    const cfg = self.config;
    const allocator = ctx.allocator;
    const cache = &session.cache;
    const S = tokens.len;
    if (S == 0) return Error.KvCacheOverflow;
    if (S == 1) return step(self, ctx, session, tokens[0]);
    if (cache.len + S > cache.capacity or cache.len + S > self.rope.capacity) return Error.KvCacheOverflow;
    const pos0 = cache.len;
    const hc_dim = cfg.n_hc * cfg.hidden_size;

    // Every batch row carries its own HC stream state.
    const streams_all = try allocator.alloc(f32, S * hc_dim);
    defer allocator.free(streams_all);
    {
        var emb = try self.token_embedding.getRowsAs(ctx, tokens, .embed);
        defer emb.deinit();
        const rows = try emb.dataConst();
        for (0..S) |s| {
            const row = rows[s * cfg.hidden_size ..][0..cfg.hidden_size];
            for (0..cfg.n_hc) |h| @memcpy(streams_all[s * hc_dim + h * cfg.hidden_size ..][0..cfg.hidden_size], row);
        }
    }

    const splits = try allocator.alloc(HcSplit, S);
    defer allocator.free(splits);

    for (self.layers, 0..) |*layer, layer_i| {
        // ---- attention sublayer ----
        {
            const sub_in = try hcPreBatch(ctx, cfg, &layer.hc_attn, streams_all, S, splits);
            defer allocator.free(sub_in);
            const block_out = try attnBlockBatch(self, ctx, cache, layer, layer_i, sub_in, pos0, S);
            defer allocator.free(block_out);
            try hcPostBatch(ctx, cfg, splits, block_out, streams_all, S);
        }
        // ---- FFN sublayer ----
        {
            const sub_in = try hcPreBatch(ctx, cfg, &layer.hc_ffn, streams_all, S, splits);
            defer allocator.free(sub_in);
            const block_out = try moeBlockBatch(self, ctx, layer, sub_in, tokens);
            defer allocator.free(block_out);
            try hcPostBatch(ctx, cfg, splits, block_out, streams_all, S);
        }
    }
    cache.len += S;

    return outputLogits(self, ctx, streams_all[(S - 1) * hc_dim ..][0..hc_dim]);
}

/// Batched hcPre: one flat rms-norm + fn projection over all rows, Sinkhorn
/// per row (4x4 host math), and the pre-weighted stream sum as one batched
/// contraction. Fills `splits` for the matching hcPostBatch.
fn hcPreBatch(ctx: *ExecContext, config: Config, module: *const HcModule, streams_all: []const f32, S: usize, splits: []HcSplit) ![]f32 {
    const allocator = ctx.allocator;
    const hc_dim = config.n_hc * config.hidden_size;

    var flat_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ S, hc_dim }, streams_all[0 .. S * hc_dim]);
    defer flat_t.deinit();
    var normed_t = try flat_t.rmsNorm(ctx, .embed, config.rms_norm_eps);
    defer normed_t.deinit();
    var mix_t = try module.fn_proj.linearSeq(ctx, &normed_t, .embed, .attn);
    defer mix_t.deinit();
    const mix = try mix_t.dataConst();

    const n_mix = 6 * config.n_hc;
    const pre_all = try allocator.alloc(f32, S * config.n_hc);
    defer allocator.free(pre_all);
    for (0..S) |s| {
        splits[s] = hcSplitSinkhorn(mix[s * n_mix ..][0..n_mix], module.scale, module.base, config.n_hc, config.hc_sinkhorn_iters);
        @memcpy(pre_all[s * config.n_hc ..][0..config.n_hc], &splits[s].pre);
    }

    var streams3 = try fucina.Tensor(.{ .seq, .stream, .embed }).fromBorrowedConstSlice(ctx, .{ S, config.n_hc, config.hidden_size }, streams_all[0 .. S * hc_dim]);
    defer streams3.deinit();
    var pre_t = try fucina.Tensor(.{ .seq, .stream }).fromBorrowedSlice(ctx, .{ S, config.n_hc }, pre_all);
    defer pre_t.deinit();
    var weighted = try streams3.mul(ctx, &pre_t);
    defer weighted.deinit();
    var summed = try weighted.sum(ctx, .stream);
    defer summed.deinit();
    return allocator.dupe(f32, try summed.dataConst());
}

/// Batched hcPost: streams'[s,dst] = post[s,dst]*block_out[s] +
/// sum_src comb[s,dst,src]*streams[s,src], all rows at once.
fn hcPostBatch(ctx: *ExecContext, config: Config, splits: []const HcSplit, block_out: []const f32, streams_all: []f32, S: usize) !void {
    const allocator = ctx.allocator;
    const n = config.n_hc;
    const hc_dim = n * config.hidden_size;

    const comb_all = try allocator.alloc(f32, S * n * n);
    defer allocator.free(comb_all);
    const post_all = try allocator.alloc(f32, S * n);
    defer allocator.free(post_all);
    for (0..S) |s| {
        // comb is addressed [dst + src*n]; lay each row out as [dst][src].
        for (0..n) |dst| {
            for (0..n) |src| comb_all[s * n * n + dst * n + src] = splits[s].comb[dst + src * n];
        }
        @memcpy(post_all[s * n ..][0..n], &splits[s].post);
    }

    var streams3 = try fucina.Tensor(.{ .seq, .stream, .embed }).fromBorrowedConstSlice(ctx, .{ S, n, config.hidden_size }, streams_all[0 .. S * hc_dim]);
    defer streams3.deinit();
    var comb_t = try fucina.Tensor(.{ .seq, .stream_dst, .stream }).fromBorrowedSlice(ctx, .{ S, n, n }, comb_all);
    defer comb_t.deinit();
    var mixed = try comb_t.dot(ctx, &streams3, .stream);
    defer mixed.deinit();
    var out_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ S, config.hidden_size }, block_out);
    defer out_t.deinit();
    var post_t = try fucina.Tensor(.{ .seq, .stream_dst }).fromBorrowedSlice(ctx, .{ S, n }, post_all);
    defer post_t.deinit();
    var injected = try out_t.einsum(ctx, &post_t, .{ .seq, .stream_dst, .embed });
    defer injected.deinit();
    var next = try mixed.add(ctx, &injected);
    defer next.deinit();
    @memcpy(streams_all[0 .. S * hc_dim], try next.dataConst());
}

/// Batched attention sublayer: all projections and norms run as S-row GEMMs;
/// rope/fp8/window state and the row-visibility bookkeeping advance per
/// token. Raw rows for the whole chunk live in a temporary [carry+S] buffer
/// (each token sees the trailing <= n_swa rows at its own position); the
/// persistent ring receives the final window afterwards.
fn attnBlockBatch(self: *Model, ctx: *ExecContext, cache: *Cache, layer: *const Layer, layer_i: usize, sub_in: []const f32, pos0: usize, S: usize) ![]f32 {
    const cfg = self.config;
    const allocator = ctx.allocator;
    const lc = &cache.layers[layer_i];
    const ratio = cfg.compress_ratio[layer_i];
    const compressed_family = ratio != 0;
    const hd = cfg.head_dim;
    const q_dim = cfg.num_heads * hd;

    var in_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ S, cfg.hidden_size }, sub_in);
    defer in_t.deinit();
    var attn_norm_w = try embedTag(ctx, layer.attn_norm);
    defer attn_norm_w.deinit();
    var x_norm = try in_t.rmsNormMul(ctx, .embed, &attn_norm_w, cfg.rms_norm_eps);
    defer x_norm.deinit();

    // q-LoRA + per-head rms norm, all rows at once.
    var q_lat = try layer.q_a.linearSeq(ctx, &x_norm, .embed, .q);
    defer q_lat.deinit();
    var q_a_norm_w = try fucina.Tensor(.{.q}).fromSlice(ctx, .{cfg.q_lora_rank}, layer.q_a_norm);
    defer q_a_norm_w.deinit();
    var qr_norm_t = try q_lat.rmsNormMul(ctx, .q, &q_a_norm_w, cfg.rms_norm_eps);
    defer qr_norm_t.deinit();
    var q_full = try layer.q_b.linearSeq(ctx, &qr_norm_t, .q, .attn);
    defer q_full.deinit();
    const q_all = try allocator.alloc(f32, S * q_dim);
    defer allocator.free(q_all);
    {
        var heads_t = try fucina.Tensor(.{ .r, .d }).fromBorrowedConstSlice(ctx, .{ S * cfg.num_heads, hd }, try q_full.dataConst());
        defer heads_t.deinit();
        var normed = try heads_t.rmsNorm(ctx, .d, cfg.rms_norm_eps);
        defer normed.deinit();
        @memcpy(q_all, try normed.dataConst());
    }

    var kv_lin = try layer.kv.linearSeq(ctx, &x_norm, .embed, .k);
    defer kv_lin.deinit();
    var kv_norm_w = try fucina.Tensor(.{.k}).fromSlice(ctx, .{hd}, layer.kv_a_norm);
    defer kv_norm_w.deinit();
    var kv_normed = try kv_lin.rmsNormMul(ctx, .k, &kv_norm_w, cfg.rms_norm_eps);
    defer kv_normed.deinit();
    const kv_all = try allocator.dupe(f32, try kv_normed.dataConst());
    defer allocator.free(kv_all);

    // Rope + fp8 per token; chunk-local raw buffer = ring carry + S new rows.
    const carry = lc.n_raw;
    const raw_buf = try allocator.alloc(f32, (carry + S) * hd);
    defer allocator.free(raw_buf);
    @memcpy(raw_buf[0 .. carry * hd], lc.raw[0 .. carry * hd]);
    for (0..S) |s| {
        const pos = pos0 + s;
        for (0..cfg.num_heads) |h| self.rope.applyTail(q_all[s * q_dim + h * hd ..][0..hd], pos, compressed_family, false);
        const kv_row = kv_all[s * hd ..][0..hd];
        self.rope.applyTail(kv_row, pos, compressed_family, false);
        fp8KvQuantRow(kv_row, cfg.rope_dims);
        for (raw_buf[(carry + s) * hd ..][0..hd], kv_row) |*d, v| d.* = f16Round(v);
    }

    // Compressors: projections batch, the window state advances per token;
    // record each token's visible compressed-row count.
    const count_attn = try allocator.alloc(usize, S);
    defer allocator.free(count_attn);
    const count_idx = try allocator.alloc(usize, S);
    defer allocator.free(count_idx);
    if (layer.attn_compressor) |*comp| {
        var kv_t = try comp.kv.linearSeq(ctx, &x_norm, .embed, .attn);
        defer kv_t.deinit();
        var sc_t = try comp.gate.linearSeq(ctx, &x_norm, .embed, .attn);
        defer sc_t.deinit();
        const kvd = try kv_t.dataConst();
        const scd = try sc_t.dataConst();
        for (0..S) |s| {
            _ = try compressorAdvance(self, ctx, comp, kvd[s * comp.width ..][0..comp.width], scd[s * comp.width ..][0..comp.width], lc.attn_state_kv, lc.attn_state_score, &lc.comp, hd, compressed_family, pos0 + s);
            count_attn[s] = lc.comp.items.len / hd;
        }
    } else @memset(count_attn, 0);
    if (layer.index_compressor) |*comp| {
        var kv_t = try comp.kv.linearSeq(ctx, &x_norm, .embed, .attn);
        defer kv_t.deinit();
        var sc_t = try comp.gate.linearSeq(ctx, &x_norm, .embed, .attn);
        defer sc_t.deinit();
        const kvd = try kv_t.dataConst();
        const scd = try sc_t.dataConst();
        for (0..S) |s| {
            _ = try compressorAdvance(self, ctx, comp, kvd[s * comp.width ..][0..comp.width], scd[s * comp.width ..][0..comp.width], lc.index_state_kv, lc.index_state_score, &lc.index_comp, cfg.indexer_head_dim, compressed_family, pos0 + s);
            count_idx[s] = lc.index_comp.items.len / cfg.indexer_head_dim;
        }
    } else @memset(count_idx, 0);

    // Indexer projections batch; selection runs per token below.
    const idx_q_dim = cfg.indexer_heads * cfg.indexer_head_dim;
    var qi_all: []f32 = &.{};
    defer if (qi_all.len > 0) allocator.free(qi_all);
    var wp_all: []f32 = &.{};
    defer if (wp_all.len > 0) allocator.free(wp_all);
    if (ratio == 4) {
        var qi_t = try layer.indexer_q_b.?.linearSeq(ctx, &qr_norm_t, .q, .attn);
        defer qi_t.deinit();
        qi_all = try allocator.dupe(f32, try qi_t.dataConst());
        var wp_t = try layer.indexer_proj.?.linearSeq(ctx, &x_norm, .embed, .attn);
        defer wp_t.deinit();
        wp_all = try allocator.dupe(f32, try wp_t.dataConst());
    }

    // Per-token attention with per-position visibility.
    const out_heads_all = try allocator.alloc(f32, S * q_dim);
    defer allocator.free(out_heads_all);
    var rows: std.ArrayList(f32) = .empty;
    defer rows.deinit(allocator);
    for (0..S) |s| {
        const pos = pos0 + s;
        const n_vis = count_attn[s];
        var allowed: ?[]bool = null;
        defer if (allowed) |a| allocator.free(a);
        if (ratio == 4 and n_vis > cfg.indexer_top_k) {
            allowed = try indexerSelectFrom(self, ctx, qi_all[s * idx_q_dim ..][0..idx_q_dim], wp_all[s * cfg.indexer_heads ..][0..cfg.indexer_heads], lc.index_comp.items[0 .. count_idx[s] * cfg.indexer_head_dim], n_vis, pos);
        }
        rows.clearRetainingCapacity();
        const t_abs = carry + s;
        const lo = if (t_abs + 1 > cfg.n_swa) t_abs + 1 - cfg.n_swa else 0;
        try rows.appendSlice(allocator, raw_buf[lo * hd .. (t_abs + 1) * hd]);
        for (0..n_vis) |c| {
            if (allowed) |a| {
                if (!a[c]) continue;
            }
            try rows.appendSlice(allocator, lc.comp.items[c * hd ..][0..hd]);
        }
        const out_row = out_heads_all[s * q_dim ..][0..q_dim];
        try attendRowsSink(self, ctx, layer, q_all[s * q_dim ..][0..q_dim], rows.items, out_row);
        for (0..cfg.num_heads) |h| self.rope.applyTail(out_row[h * hd ..][0..hd], pos, compressed_family, true);
    }

    // The persistent ring keeps the trailing window.
    const total = carry + S;
    const keep = @min(total, cfg.n_swa);
    std.mem.copyForwards(f32, lc.raw[0 .. keep * hd], raw_buf[(total - keep) * hd .. total * hd]);
    lc.n_raw = keep;

    // Grouped low-rank output, batched per group: each group's head block is
    // a narrow VIEW of the head buffer (materialized once for the quantized
    // GEMM) and the per-group ranks concat into stage B's input — no hand
    // gather/scatter.
    const group_heads = cfg.num_heads / cfg.output_groups;
    const group_dim = group_heads * hd;
    const rank = cfg.output_lora_rank;
    const group_row_bytes = (group_dim / 32) * @sizeOf(fucina.BlockQ8_0);
    var heads_full = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ S, q_dim }, out_heads_all);
    defer heads_full.deinit();
    var lows: [8]fucina.Tensor(.{ .seq, .attn }) = undefined;
    var n_lows: usize = 0;
    defer for (lows[0..n_lows]) |*t| t.deinit();
    std.debug.assert(cfg.output_groups <= lows.len);
    for (0..cfg.output_groups) |g| {
        // The quantized GEMM entry materializes non-contiguous LHS views
        // itself (pooled, once) — feed the narrow view directly.
        var gs_v = try heads_full.narrow(ctx, .embed, g * group_dim, group_dim);
        defer gs_v.deinit();
        lows[n_lows] = try weights.linearSeqBorrowedQuantized(
            .q8_0,
            ctx,
            &gs_v,
            layer.output_a[g * rank * group_row_bytes ..][0 .. rank * group_row_bytes],
            .{ rank, group_dim },
            .{ .allow_gpu = false },
            .embed,
            .attn,
        );
        n_lows += 1;
    }
    var low_refs: [7]*const fucina.Tensor(.{ .seq, .attn }) = undefined;
    for (lows[1..n_lows], 0..) |*t, i| low_refs[i] = t;
    var low_cat = try lows[0].concat(ctx, .attn, low_refs[0 .. n_lows - 1]);
    defer low_cat.deinit();
    var low_in = try low_cat.withTags(ctx, .{ .seq, .embed });
    defer low_in.deinit();
    var out_t = try layer.output_b.linearSeq(ctx, &low_in, .embed, .attn);
    defer out_t.deinit();
    return allocator.dupe(f32, try out_t.dataConst());
}

fn attnBlock(self: *Model, ctx: *ExecContext, cache: *Cache, layer: *const Layer, layer_i: usize, sub_in: []const f32, pos: usize, token: usize) ![]f32 {
    _ = token;
    const cfg = self.config;
    const allocator = ctx.allocator;
    const lc = &cache.layers[layer_i];
    const ratio = cfg.compress_ratio[layer_i];
    const compressed_family = ratio != 0;

    var in_t = try rowTensor(ctx, sub_in, cfg.hidden_size);
    defer in_t.deinit();
    var attn_norm_w = try embedTag(ctx, layer.attn_norm);
    defer attn_norm_w.deinit();
    var x_norm = try in_t.rmsNormMul(ctx, .embed, &attn_norm_w, cfg.rms_norm_eps);
    defer x_norm.deinit();

    // q-LoRA; the normed latent (qr_norm) also feeds the indexer.
    var q_lat = try layer.q_a.linearSeq(ctx, &x_norm, .embed, .q);
    defer q_lat.deinit();
    var q_a_norm_w = try fucina.Tensor(.{.q}).fromSlice(ctx, .{cfg.q_lora_rank}, layer.q_a_norm);
    defer q_a_norm_w.deinit();
    var qr_norm_t = try q_lat.rmsNormMul(ctx, .q, &q_a_norm_w, cfg.rms_norm_eps);
    defer qr_norm_t.deinit();
    var q_full = try layer.q_b.linearSeq(ctx, &qr_norm_t, .q, .attn);
    defer q_full.deinit();
    const q = try allocator.dupe(f32, try q_full.dataConst());
    defer allocator.free(q);

    // Each q head is RMS-normalized (no weight) after the LoRA projection.
    {
        var heads_t = try fucina.Tensor(.{ .head, .d }).fromSlice(ctx, .{ cfg.num_heads, cfg.head_dim }, q);
        defer heads_t.deinit();
        var normed = try heads_t.rmsNorm(ctx, .d, cfg.rms_norm_eps);
        defer normed.deinit();
        @memcpy(q, try normed.dataConst());
    }

    var kv_lin = try layer.kv.linearSeq(ctx, &x_norm, .embed, .k);
    defer kv_lin.deinit();
    var kv_norm_w = try fucina.Tensor(.{.k}).fromSlice(ctx, .{cfg.head_dim}, layer.kv_a_norm);
    defer kv_norm_w.deinit();
    var kv_normed = try kv_lin.rmsNormMul(ctx, .k, &kv_norm_w, cfg.rms_norm_eps);
    defer kv_normed.deinit();
    const kv_row = try allocator.dupe(f32, try kv_normed.dataConst());
    defer allocator.free(kv_row);

    // Tail rotary on q heads and the shared kv row, FP8-simulate the row,
    // push into the raw sliding window (shift when full).
    for (0..cfg.num_heads) |h| self.rope.applyTail(q[h * cfg.head_dim ..][0..cfg.head_dim], pos, compressed_family, false);
    self.rope.applyTail(kv_row, pos, compressed_family, false);
    fp8KvQuantRow(kv_row, cfg.rope_dims);
    if (lc.n_raw == cfg.n_swa) {
        std.mem.copyForwards(f32, lc.raw[0 .. (cfg.n_swa - 1) * cfg.head_dim], lc.raw[cfg.head_dim..]);
        lc.n_raw -= 1;
    }
    // Cached rows live f16-rounded (the reference cache stores f16).
    for (lc.raw[lc.n_raw * cfg.head_dim ..][0..cfg.head_dim], kv_row) |*d, v| d.* = f16Round(v);
    lc.n_raw += 1;

    // Compressed streams + indexer selection.
    var allowed: ?[]bool = null;
    defer if (allowed) |a| allocator.free(a);
    if (layer.attn_compressor) |*comp| {
        _ = try compressorStep(self, ctx, comp, &x_norm, lc.attn_state_kv, lc.attn_state_score, &lc.comp, cfg.head_dim, compressed_family, pos);
    }
    if (layer.index_compressor) |*comp| {
        _ = try compressorStep(self, ctx, comp, &x_norm, lc.index_state_kv, lc.index_state_score, &lc.index_comp, cfg.indexer_head_dim, compressed_family, pos);
    }
    const n_comp = lc.comp.items.len / cfg.head_dim;
    if (ratio == 4 and n_comp > cfg.indexer_top_k) {
        allowed = try indexerSelect(self, ctx, layer, &x_norm, &qr_norm_t, lc, pos);
    }

    // Assemble the visible rows (exclusion == -inf under the sink softmax),
    // then attention through tensor ops: scores = q . rows / sqrt(d), the
    // sink as an extra softmax column, weights . rows as the output.
    var rows: std.ArrayList(f32) = .empty;
    defer rows.deinit(allocator);
    try rows.appendSlice(allocator, lc.raw[0 .. lc.n_raw * cfg.head_dim]);
    for (0..n_comp) |c| {
        if (allowed) |a| {
            if (!a[c]) continue;
        }
        try rows.appendSlice(allocator, lc.comp.items[c * cfg.head_dim ..][0..cfg.head_dim]);
    }
    const out_heads = try allocator.alloc(f32, cfg.num_heads * cfg.head_dim);
    defer allocator.free(out_heads);
    try attendRowsSink(self, ctx, layer, q, rows.items, out_heads);

    // Undo the value-side tail rotation carried by the K==V rows.
    for (0..cfg.num_heads) |h| self.rope.applyTail(out_heads[h * cfg.head_dim ..][0..cfg.head_dim], pos, compressed_family, true);

    // Grouped low-rank output: per group, [group_heads*d] -> rank (each
    // group's block is a borrowed view — one token's group slice is
    // contiguous), then the concatenated ranks through stage B.
    const group_heads = cfg.num_heads / cfg.output_groups;
    const group_dim = group_heads * cfg.head_dim;
    const rank = cfg.output_lora_rank;
    const group_row_bytes = blk: {
        // q8_0 rows of `group_dim` values: group g's rows start at
        // g*rank rows into the stacked stage-A weight.
        const bpr = group_dim / 32;
        break :blk bpr * @sizeOf(fucina.BlockQ8_0);
    };
    var lows: [8]fucina.Tensor(.{ .seq, .attn }) = undefined;
    var n_lows: usize = 0;
    defer for (lows[0..n_lows]) |*t| t.deinit();
    std.debug.assert(cfg.output_groups <= lows.len);
    for (0..cfg.output_groups) |g| {
        var head_slice = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ 1, group_dim }, out_heads[g * group_dim ..][0..group_dim]);
        defer head_slice.deinit();
        lows[n_lows] = try weights.linearSeqBorrowedQuantized(
            .q8_0,
            ctx,
            &head_slice,
            layer.output_a[g * rank * group_row_bytes ..][0 .. rank * group_row_bytes],
            .{ rank, group_dim },
            .{ .allow_gpu = false },
            .embed,
            .attn,
        );
        n_lows += 1;
    }
    var low_refs: [7]*const fucina.Tensor(.{ .seq, .attn }) = undefined;
    for (lows[1..n_lows], 0..) |*t, i| low_refs[i] = t;
    var low_cat = try lows[0].concat(ctx, .attn, low_refs[0 .. n_lows - 1]);
    defer low_cat.deinit();
    var low_in = try low_cat.withTags(ctx, .{ .seq, .embed });
    defer low_in.deinit();
    var out_t = try layer.output_b.linearSeq(ctx, &low_in, .embed, .attn);
    defer out_t.deinit();
    return allocator.dupe(f32, try out_t.dataConst());
}

/// Attention over an assembled row set with the per-head sink logit as an
/// extra softmax column: out_heads = softmax([q·rows·scale | sink]) · rows.
/// `q_row` is all heads of one token; both inputs enter as borrowed views.
fn attendRowsSink(self: *const Model, ctx: *ExecContext, layer: *const Layer, q_row: []const f32, rows: []const f32, out_heads: []f32) !void {
    const cfg = self.config;
    const n_rows = rows.len / cfg.head_dim;

    var q_t = try fucina.Tensor(.{ .head, .d }).fromBorrowedConstSlice(ctx, .{ cfg.num_heads, cfg.head_dim }, q_row);
    defer q_t.deinit();
    var rows_t = try fucina.Tensor(.{ .row, .d }).fromBorrowedConstSlice(ctx, .{ n_rows, cfg.head_dim }, rows);
    defer rows_t.deinit();
    var scores = try q_t.dot(ctx, &rows_t, .d);
    defer scores.deinit();
    var scaled = try scores.scale(ctx, self.attn_scale);
    defer scaled.deinit();

    // Sink column: concat the per-head sink logit as one extra row-axis
    // column (a [head,1] view of the flat sinks vector — zero copy).
    var sink_col = try fucina.Tensor(.{ .head, .row }).fromBorrowedConstSlice(ctx, .{ cfg.num_heads, 1 }, layer.sinks);
    defer sink_col.deinit();
    var ext_t = try scaled.concat(ctx, .row, &.{&sink_col});
    defer ext_t.deinit();
    var probs = try ext_t.softmax(ctx, .row, .{});
    defer probs.deinit();
    var probs_rows = try probs.narrow(ctx, .row, 0, n_rows);
    defer probs_rows.deinit();
    var out_heads_t = try probs_rows.dot(ctx, &rows_t, .row);
    defer out_heads_t.deinit();
    @memcpy(out_heads, try out_heads_t.dataConst());
}

fn indexerSelect(self: *Model, ctx: *ExecContext, layer: *const Layer, x_norm: anytype, qr_norm_t: anytype, lc: *LayerCache, pos: usize) ![]bool {
    const cfg = self.config;
    const allocator = ctx.allocator;

    var q_t = try layer.indexer_q_b.?.linearSeq(ctx, qr_norm_t, .q, .attn);
    defer q_t.deinit();
    const q = try allocator.dupe(f32, try q_t.dataConst());
    defer allocator.free(q);
    var w_lin = try layer.indexer_proj.?.linearSeq(ctx, x_norm, .embed, .attn);
    defer w_lin.deinit();
    const head_w = try allocator.dupe(f32, try w_lin.dataConst());
    defer allocator.free(head_w);

    return indexerSelectFrom(self, ctx, q, head_w, lc.index_comp.items, lc.comp.items.len / cfg.head_dim, pos);
}

/// Scoring/selection core shared by decode and batched prefill: takes this
/// token's already-projected indexer q heads (mutated in place: rope + QAT)
/// and unscaled head weights, scores the visible index-compressed rows, and
/// returns the allowed mask over the attention-compressed rows (1:1 counts).
fn indexerSelectFrom(self: *const Model, ctx: *ExecContext, q: []f32, head_w: []f32, index_comp: []const f32, n_allowed_rows: usize, pos: usize) ![]bool {
    const cfg = self.config;
    const allocator = ctx.allocator;
    const n_comp = index_comp.len / cfg.indexer_head_dim;
    const allowed = try allocator.alloc(bool, n_allowed_rows);
    @memset(allowed, false);
    const top_k = @min(cfg.indexer_top_k, n_comp);

    for (0..cfg.indexer_heads) |h| {
        const head = q[h * cfg.indexer_head_dim ..][0..cfg.indexer_head_dim];
        self.rope.applyTail(head, pos, true, false);
        indexerQatRow(head);
    }
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.indexer_heads * cfg.indexer_head_dim)));
    for (head_w) |*w| w.* *= scale;

    // score[c] = sum_h relu(comp_c . q_h) * w[h] over the QAT'd rows: one
    // batched contraction + relu + weighted head reduction (n_comp grows
    // with context, so this is the indexer's hot loop).
    var comp_t = try fucina.Tensor(.{ .row, .d }).fromBorrowedConstSlice(ctx, .{ n_comp, cfg.indexer_head_dim }, index_comp[0 .. n_comp * cfg.indexer_head_dim]);
    defer comp_t.deinit();
    var q_t2 = try fucina.Tensor(.{ .head, .d }).fromBorrowedSlice(ctx, .{ cfg.indexer_heads, cfg.indexer_head_dim }, q);
    defer q_t2.deinit();
    var dots = try comp_t.dot(ctx, &q_t2, .d);
    defer dots.deinit();
    var rectified = try dots.unary(ctx, .relu);
    defer rectified.deinit();
    var w_t = try fucina.Tensor(.{.head}).fromBorrowedSlice(ctx, .{cfg.indexer_heads}, head_w);
    defer w_t.deinit();
    var weighted = try rectified.mul(ctx, &w_t);
    defer weighted.deinit();
    var scores_t = try weighted.sum(ctx, .head);
    defer scores_t.deinit();
    const scores = try scores_t.dataConst();
    for (0..top_k) |_| {
        var best: usize = 0;
        var best_score = -std.math.inf(f32);
        for (scores, 0..) |sc, c| {
            if (!allowed[c] and sc > best_score) {
                best = c;
                best_score = sc;
            }
        }
        allowed[best] = true;
    }
    return allowed;
}

fn moeBlock(self: *Model, ctx: *ExecContext, layer: *const Layer, sub_in: []const f32, token: usize) ![]f32 {
    return moeBlockBatch(self, ctx, layer, sub_in, &.{token});
}

fn moeBlockBatch(self: *Model, ctx: *ExecContext, layer: *const Layer, sub_in: []const f32, tokens: []const usize) ![]f32 {
    const cfg = self.config;
    const allocator = ctx.allocator;
    const S = tokens.len;
    const used = cfg.num_experts_used;

    var in_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ S, cfg.hidden_size }, sub_in);
    defer in_t.deinit();
    var ffn_norm_w = try embedTag(ctx, layer.ffn_norm);
    defer ffn_norm_w.deinit();
    var x_norm = try in_t.rmsNormMul(ctx, .embed, &ffn_norm_w, cfg.rms_norm_eps);
    defer x_norm.deinit();

    // Router: probs = sqrt(softplus(logits)) — softplus is a core unary now.
    var logits_t = try layer.moe.router.linearSeq(ctx, &x_norm, .embed, .expert);
    defer logits_t.deinit();
    var sp = try logits_t.unary(ctx, .softplus);
    defer sp.deinit();
    var probs_t = try sp.unary(ctx, .sqrt);
    defer probs_t.deinit();
    const probs_all = try probs_t.dataConst();

    const selected = try allocator.alloc(usize, S * used);
    defer allocator.free(selected);
    const routing = try allocator.alloc(f32, S * used);
    defer allocator.free(routing);
    const choice = try allocator.alloc(f32, cfg.num_experts);
    defer allocator.free(choice);
    for (0..S) |s| {
        const probs = probs_all[s * cfg.num_experts ..][0..cfg.num_experts];
        const sel = selected[s * used ..][0..used];
        const wts = routing[s * used ..][0..used];
        if (layer.moe.tid2eid) |table| {
            const row = table[tokens[s] * used ..][0..used];
            for (0..used) |i| {
                const e: usize = @intCast(row[i]);
                if (e >= cfg.num_experts) return Error.InvalidWeightShape;
                sel[i] = e;
                wts[i] = probs[e];
            }
        } else {
            @memcpy(choice, probs);
            if (layer.moe.router_bias) |bias| {
                for (choice, bias) |*c, b| c.* += b;
            }
            for (0..used) |slot| {
                var best: usize = 0;
                var best_c = -std.math.inf(f32);
                for (choice, 0..) |c, e| {
                    if (c > best_c) {
                        best_c = c;
                        best = e;
                    }
                }
                choice[best] = -std.math.inf(f32);
                sel[slot] = best;
                wts[slot] = probs[best];
            }
        }
        var sum: f32 = 0;
        for (wts) |w| sum += w;
        if (sum < 6.103515625e-5) sum = 6.103515625e-5;
        for (wts) |*w| w.* = w.* / sum * cfg.expert_weights_scale;
    }

    var routed = try weights.moeGatedFfnSeq(ctx, &x_norm, &layer.moe.gate, &layer.moe.up, &layer.moe.down, selected, routing, used, cfg.expert_ffn_size, .swiglu_clamp10, null, null);
    defer routed.deinit();

    // Shared expert with the clamped SwiGLU, entirely through tensor ops.
    var gate_t = try layer.moe.shared_gate.linearSeq(ctx, &x_norm, .embed, .gate_up);
    defer gate_t.deinit();
    var up_t = try layer.moe.shared_up.linearSeq(ctx, &x_norm, .embed, .gate_up);
    defer up_t.deinit();
    var gate_c = try gate_t.clamp(ctx, -std.math.floatMax(f32), 10.0);
    defer gate_c.deinit();
    var gate_act = try gate_c.unary(ctx, .silu);
    defer gate_act.deinit();
    var up_c = try up_t.clamp(ctx, -10.0, 10.0);
    defer up_c.deinit();
    var mid = try gate_act.mul(ctx, &up_c);
    defer mid.deinit();
    var shared = try layer.moe.shared_down.linearSeq(ctx, &mid, .gate_up, .embed);
    defer shared.deinit();

    var total = try routed.add(ctx, &shared);
    defer total.deinit();
    return allocator.dupe(f32, try total.dataConst());
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
