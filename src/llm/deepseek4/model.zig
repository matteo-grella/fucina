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
