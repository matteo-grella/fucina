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
    /// Cross-layer index sharing hit a Shared layer with no Full-layer
    /// stash this step — a broken invariant (layer 0 is always Full).
    IndexShareMismatch,
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
    /// DSA lightning indexer (V3.2 / glm-dsa files; 0 on V2/V3 files).
    /// Presence in the file does not enable it — `LoadOptions.dsa` does;
    /// the default stays dense attention, llama.cpp's fallback too.
    indexer_heads: usize,
    indexer_key_dim: usize,
    indexer_top_k: usize,
    /// Indexer rope pairing: half-split (non-interleaved) on deepseek32,
    /// interleaved on glm-dsa — each differs from its own main attention in
    /// exactly this one axis (reference `Indexer` classes; the rotated
    /// slice is the LEADING qk_rope_dim dims either way).
    indexer_rope_half: bool,

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
        else if (std.mem.eql(u8, arch, "deepseek32"))
            "deepseek32"
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
            .indexer_heads = gguf_meta.metaIntOpt(file, prefix, "attention.indexer.head_count", .accept_zero) orelse 0,
            .indexer_key_dim = gguf_meta.metaIntOpt(file, prefix, "attention.indexer.key_length", .accept_zero) orelse 0,
            .indexer_top_k = gguf_meta.metaIntOpt(file, prefix, "attention.indexer.top_k", .accept_zero) orelse 0,
            .indexer_rope_half = std.mem.eql(u8, arch, "deepseek32"),
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
    /// DSA lightning indexer (LoadOptions.dsa on a V3.2/glm-dsa file):
    /// per-position key projection + LayerNorm(weight, bias), per-head
    /// queries from the normed q-LoRA latent, per-head score weights.
    /// null = dense attention.
    idx_k: ?LinearWeight, // hidden -> idx_dim
    idx_k_norm_w: ?[]f32, // [idx_dim]
    idx_k_norm_b: ?[]f32, // [idx_dim]
    idx_q_b: ?LinearWeight, // q_lora -> idx_heads*idx_dim
    idx_proj: ?LinearWeight, // hidden -> idx_heads
    ffn: Ffn,

    fn deinit(self: *Layer, allocator: Allocator) void {
        self.ffn.deinit();
        if (self.idx_proj) |*w| w.deinit();
        if (self.idx_q_b) |*w| w.deinit();
        if (self.idx_k_norm_b) |n| allocator.free(n);
        if (self.idx_k_norm_w) |n| allocator.free(n);
        if (self.idx_k) |*w| w.deinit();
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
    /// DSA indexer keys, [capacity, indexer_key_dim] per layer — allocated
    /// only when the model loaded the indexer (LoadOptions.dsa).
    idx: [][]f32 = &.{},
    idx_width: usize = 0,
    len: usize = 0,
    capacity: usize,
    /// Cross-layer index-share state for the CURRENT step (the ds4
    /// IndexCache pattern, position-space): the nearest Full layer's
    /// selected positions. Reset at every step entry. Decode holds one
    /// selection (`share_lens` empty); `stepBatch` flattens S per-token
    /// selections with `share_lens[s]` entries each (0 = no selection).
    share_sel: std.ArrayList(usize) = .empty,
    share_lens: std.ArrayList(usize) = .empty,
    share_valid: bool = false,
    share_computed: u64 = 0,
    share_reused: u64 = 0,
    /// Selection-overlap probe (null = off; enable via enableDsaProbe with
    /// sharing OFF — it measures the exact path).
    probe: ?DsaProbe = null,

    pub fn init(allocator: Allocator, config: Config, capacity: usize, mode: Mode, idx_width: usize) !Cache {
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
        var idx: [][]f32 = &.{};
        if (idx_width > 0) {
            idx = try allocator.alloc([]f32, config.num_layers);
            errdefer allocator.free(idx);
            var idx_built: usize = 0;
            errdefer for (idx[0..idx_built]) |layer| allocator.free(layer);
            for (0..config.num_layers) |i| {
                idx[i] = try allocator.alloc(f32, capacity * idx_width);
                idx_built += 1;
            }
        }
        return .{ .allocator = allocator, .mode = mode, .k = k, .v = v, .idx = idx, .idx_width = idx_width, .capacity = capacity };
    }

    pub fn enableDsaProbe(self: *Cache, n_layers: usize) !void {
        self.probe = try DsaProbe.init(self.allocator, n_layers);
    }

    pub fn deinit(self: *Cache) void {
        if (self.probe) |*p| p.deinit();
        self.share_lens.deinit(self.allocator);
        self.share_sel.deinit(self.allocator);
        for (self.idx) |layer| self.allocator.free(layer);
        if (self.idx.len > 0) self.allocator.free(self.idx);
        for (self.k) |layer| self.allocator.free(layer);
        for (self.v) |layer| self.allocator.free(layer);
        self.allocator.free(self.k);
        if (self.v.len > 0) self.allocator.free(self.v);
        self.* = undefined;
    }

    pub fn reset(self: *Cache) void {
        self.len = 0;
    }

    fn shareReset(self: *Cache) void {
        self.share_sel.clearRetainingCapacity();
        self.share_lens.clearRetainingCapacity();
        self.share_valid = false;
    }
};

/// Decode-time selection-overlap probe (the IndexCache calibration
/// instrument, ds4's design over raw positions): records every DSA layer's
/// selected position set each step and accumulates pairwise overlap
/// `|Si ∩ Sj| / |Si|`. Report via `report`.
pub const DsaProbe = struct {
    allocator: Allocator,
    n_layers: usize,
    masks: [][]bool,
    has: []bool,
    overlap_sum: []f64,
    overlap_cnt: []u64,
    steps: u64 = 0,

    fn init(allocator: Allocator, n_layers: usize) !DsaProbe {
        const masks = try allocator.alloc([]bool, n_layers);
        errdefer allocator.free(masks);
        for (masks) |*m| m.* = &.{};
        const has = try allocator.alloc(bool, n_layers);
        errdefer allocator.free(has);
        @memset(has, false);
        const sums = try allocator.alloc(f64, n_layers * n_layers);
        errdefer allocator.free(sums);
        @memset(sums, 0);
        const cnts = try allocator.alloc(u64, n_layers * n_layers);
        errdefer allocator.free(cnts);
        @memset(cnts, 0);
        return .{ .allocator = allocator, .n_layers = n_layers, .masks = masks, .has = has, .overlap_sum = sums, .overlap_cnt = cnts };
    }

    fn deinit(self: *DsaProbe) void {
        for (self.masks) |m| self.allocator.free(m);
        self.allocator.free(self.masks);
        self.allocator.free(self.has);
        self.allocator.free(self.overlap_sum);
        self.allocator.free(self.overlap_cnt);
        self.* = undefined;
    }

    fn record(self: *DsaProbe, layer_i: usize, selected: []const usize, t_len: usize) !void {
        if (self.masks[layer_i].len != t_len) {
            self.allocator.free(self.masks[layer_i]);
            self.masks[layer_i] = try self.allocator.alloc(bool, t_len);
        }
        @memset(self.masks[layer_i], false);
        for (selected) |p| self.masks[layer_i][p] = true;
        self.has[layer_i] = true;
    }

    fn stepDone(self: *DsaProbe) void {
        var any = false;
        for (0..self.n_layers) |i| {
            if (!self.has[i]) continue;
            any = true;
            var selected_i: usize = 0;
            for (self.masks[i]) |b| selected_i += @intFromBool(b);
            if (selected_i == 0) continue;
            for (i + 1..self.n_layers) |j| {
                if (!self.has[j] or self.masks[j].len != self.masks[i].len) continue;
                var inter: usize = 0;
                for (self.masks[i], self.masks[j]) |a, b| inter += @intFromBool(a and b);
                self.overlap_sum[i * self.n_layers + j] += @as(f64, @floatFromInt(inter)) / @as(f64, @floatFromInt(selected_i));
                self.overlap_cnt[i * self.n_layers + j] += 1;
            }
        }
        if (any) self.steps += 1;
        @memset(self.has, false);
    }

    /// Mean overlap by layer distance, then the adjacent-pair detail row.
    pub fn report(self: *const DsaProbe, writer: anytype) !void {
        try writer.print("dsa probe: {d} selecting steps, {d} layers\n", .{ self.steps, self.n_layers });
        var d: usize = 1;
        while (d < @min(self.n_layers, 9)) : (d += 1) {
            var sum: f64 = 0;
            var cnt: u64 = 0;
            var min_pair: f64 = 1.0;
            var min_i: usize = 0;
            for (0..self.n_layers - d) |i| {
                const c = self.overlap_cnt[i * self.n_layers + (i + d)];
                if (c == 0) continue;
                const mean = self.overlap_sum[i * self.n_layers + (i + d)] / @as(f64, @floatFromInt(c));
                sum += mean;
                cnt += 1;
                if (mean < min_pair) {
                    min_pair = mean;
                    min_i = i;
                }
            }
            if (cnt == 0) continue;
            try writer.print("  distance {d}: mean {d:.1}% (weakest {d:.1}% at layer {d}->{d})\n", .{ d, 100.0 * sum / @as(f64, @floatFromInt(cnt)), 100.0 * min_pair, min_i, min_i + d });
        }
        try writer.print("  adjacent pairs:", .{});
        for (0..self.n_layers -| 1) |i| {
            const c = self.overlap_cnt[i * self.n_layers + i + 1];
            const mean = if (c == 0) 0.0 else self.overlap_sum[i * self.n_layers + i + 1] / @as(f64, @floatFromInt(c));
            try writer.print(" {d:.0}", .{100.0 * mean});
        }
        try writer.print("\n", .{});
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
    /// DSA indexer tensors loaded (LoadOptions.dsa): `.latent` decode
    /// attends sparsely past `Config.indexer_top_k` positions.
    dsa_enabled: bool = false,
    /// Cross-layer indexer reuse over the DSA layers (IndexCache,
    /// arXiv:2603.12201; same contract as deepseek4's field): 0/1 = off,
    /// N >= 2 = every Nth layer computes its selection, the layers between
    /// reuse it. Approximate BY DESIGN; calibrate with the probe first.
    index_share_every: usize = 0,
    /// Dynamic expert dropping, dial 1 (opt-in; 1.0 = off): keep routed
    /// experts in weight order until they cover this fraction of the
    /// selected gate mass — confident tokens drop the tail, uncertain
    /// tokens keep all k. Deterministic. Applies to the decode/NLL path.
    moe_top_p: f32 = 1.0,
    /// Dynamic expert dropping, dial 2 (opt-in; 0 = off): drop a routed
    /// expert whose share of the selected gate mass is below this AND
    /// which is neither pinned nor cached — the disk read is skipped,
    /// resident experts are always kept (free). Output depends on cache
    /// state: serve-time tradeoff mode, NOT bit-reproducible across runs.
    moe_skip_miss_below: f32 = 0,
    /// (1 + yarn_log_multiplier * ln(factor))² / sqrt(qk_head): the YaRN
    /// mscale² attention-scale correction folded with the 1/sqrt(d) scale.
    attn_scale: f32,

    pub const MoeStreamOptions = weights.MoeStreamOptions;

    pub const LoadOptions = struct {
        moe_stream: ?MoeStreamOptions = null,
        /// Load the DSA lightning-indexer tensors and attend sparsely
        /// (top-`Config.indexer_top_k` positions) once the context exceeds
        /// top_k — the TRAINED behavior of V3.2/glm-dsa checkpoints; the
        /// default false keeps today's dense fallback bit-identically.
        dsa: bool = false,
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
            layer.* = try loadLayer(ctx, file, config, i, expert_store, options.dsa);
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
            .dsa_enabled = options.dsa,
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
        return self.initCacheMode(capacity, .latent);
    }

    pub fn initCacheMode(self: *const Model, capacity: usize, mode: Cache.Mode) !Cache {
        const idx_width = if (self.dsa_enabled and mode == .latent) self.config.indexer_key_dim else 0;
        return Cache.init(self.allocator, self.config, capacity, mode, idx_width);
    }

    /// True when the DSA indexer tensors were loaded (LoadOptions.dsa).
    pub fn dsaEnabled(self: *const Model) bool {
        return self.dsa_enabled;
    }

    /// The indexer's rope pairing is an arch property: DeepSeek V3.2 pairs
    /// half-split (neox), GLM pairs interleaved — each differs from its own
    /// MAIN attention in exactly that axis, so the mode comes from config,
    /// never from the main-attention constant.
    fn dsaRopeHalf(self: *const Model) bool {
        return self.config.indexer_rope_half;
    }

    /// Rope the LEADING qk_rope_dim dims of `n_rows` CONTIGUOUS key rows of
    /// `indexer_key_dim` floats (already normed) in place; `rope_table`
    /// spans those rows' positions. The trailing dims pass through unroped
    /// (the reference indexer's [pe | nope] row layout).
    fn dsaRopeKeyRows(self: *const Model, ctx: *ExecContext, rows: []f32, n_rows: usize, rope_table: anytype) !void {
        const cfg = self.config;
        const allocator = ctx.allocator;
        const dim = cfg.indexer_key_dim;
        const rd = cfg.qk_rope_dim;
        const tmp = try allocator.alloc(f32, n_rows * rd);
        defer allocator.free(tmp);
        for (0..n_rows) |r| @memcpy(tmp[r * rd ..][0..rd], rows[r * dim ..][0..rd]);
        var t = try fucina.Tensor(.{ .seq, .k }).fromSlice(ctx, .{ n_rows, rd }, tmp);
        defer t.deinit();
        var rot = if (self.dsaRopeHalf())
            try t.rope(ctx, .seq, .k, rope_table, .half)
        else
            try t.rope(ctx, .seq, .k, rope_table, rope_mode);
        defer rot.deinit();
        const out = try rot.dataConst();
        for (0..n_rows) |r| @memcpy(rows[r * dim ..][0..rd], out[r * rd ..][0..rd]);
    }

    /// Fill `buf` ([n_rows*heads*dim]) with the indexer q heads from the
    /// projected `[seq, heads*dim]` tensor (head-major), the LEADING
    /// qk_rope_dim dims of every head roped like the keys.
    fn dsaRopeQBuf(self: *const Model, ctx: *ExecContext, idx_q_t: anytype, buf: []f32, n_rows: usize, rope_table: anytype) !void {
        const cfg = self.config;
        const heads = cfg.indexer_heads;
        const dim = cfg.indexer_key_dim;
        const rd = cfg.qk_rope_dim;
        var q_heads = try idx_q_t.split(ctx, .attn, .{ .head, .d }, .{ heads, dim });
        defer q_heads.deinit();
        @memcpy(buf, try q_heads.dataConst());
        var pe_v = try q_heads.narrow(ctx, .d, 0, rd);
        defer pe_v.deinit();
        var rot = if (self.dsaRopeHalf())
            try pe_v.rope(ctx, .seq, .d, rope_table, .half)
        else
            try pe_v.rope(ctx, .seq, .d, rope_table, rope_mode);
        defer rot.deinit();
        const pe = try rot.dataConst();
        for (0..n_rows * heads) |h| @memcpy(buf[h * dim ..][0..rd], pe[h * rd ..][0..rd]);
    }

    /// One decode step: process `token` at position `cache.len`, append its
    /// K/V, return the next-token logits as a host slice (caller-owned).
    pub fn step(self: *const Model, ctx: *ExecContext, cache: *Cache, token: usize) ![]f32 {
        const cfg = self.config;
        const allocator = ctx.allocator;
        if (cache.len >= cache.capacity) return Error.KvCacheOverflow;
        const pos = cache.len;
        cache.shareReset();
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

            // Cross-layer index sharing, two sources composed: GLM-5.2's
            // NATIVE IndexShare (layers without indexer weights MUST reuse
            // the previous Full layer's selection — the trained behavior)
            // and the experimental `Model.index_share_every` knob, which
            // additionally turns every non-Nth weighted layer into a Shared
            // one (IndexCache, arXiv:2603.12201 — calibrate with the probe).
            const dsa_shared_layer = layer.idx_k == null or
                (self.index_share_every >= 2 and (layer_i % self.index_share_every) != 0);
            const dsa_here = cache.idx_width > 0 and layer.idx_k != null and cache.mode == .latent and !dsa_shared_layer;

            // q projection (direct or q-LoRA), per-head view, tail rotary —
            // every head's rope slice rotates at `pos` in one core rope op.
            // The normed q-LoRA latent also feeds the DSA indexer queries.
            var idx_q_t: ?fucina.Tensor(.{ .seq, .attn }) = null;
            defer if (idx_q_t) |*t| t.deinit();
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
                if (dsa_here and pos + 1 > cfg.indexer_top_k) {
                    idx_q_t = try layer.idx_q_b.?.linearSeq(ctx, &q_lat_t, .embed, .attn);
                }
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

                    // DSA: append this position's indexer key; past top_k
                    // positions, select and attend only the chosen rows
                    // (below/at top_k every row is visible — the dense path
                    // runs untouched, bit-identically). Shared layers reuse
                    // the step's stashed selection instead.
                    var sel_buf: ?[]usize = null;
                    defer if (sel_buf) |s| allocator.free(s);
                    var gathered: ?[]f32 = null;
                    defer if (gathered) |g| allocator.free(g);
                    var att_rows: []const f32 = k_layer[0 .. t_len * row_w];
                    var att_n: usize = t_len;
                    if (dsa_here) {
                        try dsaAppendKey(self, ctx, cache, layer, layer_i, &h_t, pos, &rope_table);
                    }
                    if (cache.idx_width > 0 and t_len > cfg.indexer_top_k) {
                        var sel: []usize = undefined;
                        if (dsa_shared_layer) {
                            if (!cache.share_valid) return Error.IndexShareMismatch;
                            sel = try allocator.dupe(usize, cache.share_sel.items);
                            cache.share_reused += 1;
                        } else {
                            sel = try dsaSelect(self, ctx, cache, layer, layer_i, &idx_q_t.?, &h_t, t_len, &rope_table);
                            cache.share_computed += 1;
                            cache.share_sel.clearRetainingCapacity();
                            try cache.share_sel.appendSlice(cache.allocator, sel);
                            cache.share_valid = true;
                            if (cache.probe) |*p| try p.record(layer_i, sel, t_len);
                        }
                        sel_buf = sel;
                        const g = try allocator.alloc(f32, sel.len * row_w);
                        for (sel, 0..) |p, i| @memcpy(g[i * row_w ..][0..row_w], k_layer[p * row_w ..][0..row_w]);
                        gathered = g;
                        att_rows = g;
                        att_n = sel.len;
                    }

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
                    var rows_t = try fucina.Tensor(.{ .t, .d }).fromBorrowedConstSlice(ctx, .{ att_n, row_w }, att_rows);
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
        if (cache.probe) |*p| p.stepDone();
        cache.len += 1;

        rmsNormInto(h_norm, x, self.output_norm, cfg.rms_norm_eps);
        var final_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, cfg.hidden_size }, h_norm);
        defer final_t.deinit();
        var logits_t = try self.output.linearSeq(ctx, &final_t, .embed, .vocab);
        defer logits_t.deinit();
        return allocator.dupe(f32, try logits_t.dataConst());
    }

    /// Batched prefill: one chunk of positions through all layers with S-row
    /// projections and union-routed expert fetches (each routed expert reads
    /// once per layer per chunk — the point for streamed weights). Per-token
    /// state (cache rows, DSA keys and selection) advances sequentially
    /// inside each layer exactly as decode would, so a batched prefill
    /// leaves the caches in the same state as the equivalent sequence of
    /// single steps. Returns the LAST position's logits. Batched GEMMs may
    /// reassociate vs S=1 (~1e-6 rel) past the m-dependent kernel
    /// thresholds — the qwen3 forwardStepAllLogits convention. `.latent`
    /// cache mode only (the production path).
    pub fn stepBatch(self: *const Model, ctx: *ExecContext, cache: *Cache, tokens: []const usize) ![]f32 {
        const cfg = self.config;
        const allocator = ctx.allocator;
        const S = tokens.len;
        if (S == 0) return Error.InvalidSequenceLength;
        if (S == 1) return self.step(ctx, cache, tokens[0]);
        if (cache.mode != .latent) return Error.InvalidConfig;
        if (cache.len + S > cache.capacity) return Error.KvCacheOverflow;
        const pos0 = cache.len;
        cache.shareReset();
        var rope_table = try self.rope.table(ctx, pos0, S);
        defer rope_table.deinit();

        const hidden = cfg.hidden_size;
        const xs = try allocator.alloc(f32, S * hidden);
        defer allocator.free(xs);
        {
            var emb = try self.token_embedding.getRowsAs(ctx, tokens, .embed);
            defer emb.deinit();
            @memcpy(xs, try emb.dataConst());
        }
        const hn = try allocator.alloc(f32, S * hidden);
        defer allocator.free(hn);
        const attn_out = try allocator.alloc(f32, S * cfg.num_heads * cfg.v_head_dim);
        defer allocator.free(attn_out);
        const latent_tmp = try allocator.alloc(f32, cfg.kv_lora_rank);
        defer allocator.free(latent_tmp);

        for (self.layers, 0..) |*layer, layer_i| {
            // ---- MLA attention: S-row projections, per-token core ----
            for (0..S) |s| rmsNormInto(hn[s * hidden ..][0..hidden], xs[s * hidden ..][0..hidden], layer.attn_norm, cfg.rms_norm_eps);
            var h_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ S, hidden }, hn);
            defer h_t.deinit();

            const dsa_shared_layer = layer.idx_k == null or
                (self.index_share_every >= 2 and (layer_i % self.index_share_every) != 0);
            const dsa_on = cache.idx_width > 0;
            const dsa_here = dsa_on and layer.idx_k != null and !dsa_shared_layer;
            const selecting = dsa_on and pos0 + S > cfg.indexer_top_k;

            var idx_q_t: ?fucina.Tensor(.{ .seq, .attn }) = null;
            defer if (idx_q_t) |*t| t.deinit();
            var q_flat = blk: {
                if (layer.q_proj) |*direct| {
                    break :blk try direct.linearSeq(ctx, &h_t, .embed, .q);
                }
                var qa_t = try layer.q_a.?.linearSeq(ctx, &h_t, .embed, .q);
                defer qa_t.deinit();
                const q_lat = try allocator.alloc(f32, S * cfg.q_lora_rank);
                defer allocator.free(q_lat);
                const qa = try qa_t.dataConst();
                for (0..S) |s| rmsNormInto(q_lat[s * cfg.q_lora_rank ..][0..cfg.q_lora_rank], qa[s * cfg.q_lora_rank ..][0..cfg.q_lora_rank], layer.q_a_norm.?, cfg.rms_norm_eps);
                var q_lat_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ S, cfg.q_lora_rank }, q_lat);
                defer q_lat_t.deinit();
                if (dsa_here and selecting) {
                    idx_q_t = try layer.idx_q_b.?.linearSeq(ctx, &q_lat_t, .embed, .attn);
                }
                break :blk try layer.q_b.?.linearSeq(ctx, &q_lat_t, .embed, .q);
            };
            defer q_flat.deinit();
            var q_heads = try q_flat.split(ctx, .q, .{ .head, .d }, .{ cfg.num_heads, cfg.qk_head_dim });
            defer q_heads.deinit();
            var q_rot = try q_heads.rope(ctx, .seq, .d, &rope_table, rope_mode);
            defer q_rot.deinit();

            var kv_a_t = try layer.kv_a.linearSeq(ctx, &h_t, .embed, .k);
            defer kv_a_t.deinit();
            const kv_a = try kv_a_t.dataConst();
            var k_pe_v = try kv_a_t.narrow(ctx, .k, cfg.kv_lora_rank, cfg.qk_rope_dim);
            defer k_pe_v.deinit();
            var k_pe_rot = try k_pe_v.rope(ctx, .seq, .k, &rope_table, rope_mode);
            defer k_pe_rot.deinit();
            const k_pe_all = try k_pe_rot.dataConst();

            // Absorption weights, built once per layer; the per-token loop
            // runs the decode-shaped contractions on its own q row.
            const lora = cfg.kv_lora_rank;
            const row_w = lora + cfg.qk_rope_dim;
            var wk_t = try fucina.Tensor(.{ .head, .nope, .lora }).fromBorrowedConstSlice(ctx, .{ cfg.num_heads, cfg.qk_nope_dim, lora }, layer.kv_b_k);
            defer wk_t.deinit();
            var wv_t = try fucina.Tensor(.{ .head, .v, .d }).fromBorrowedConstSlice(ctx, .{ cfg.num_heads, cfg.v_head_dim, lora }, layer.kv_b_v);
            defer wv_t.deinit();

            // DSA batched pieces: keys for every position (Full layers), and
            // pre-roped indexer queries + head weights when selecting.
            var idx_q_buf: []f32 = &.{};
            defer if (idx_q_buf.len > 0) allocator.free(idx_q_buf);
            var idx_w_all: []const f32 = &.{};
            var idx_w_t: ?fucina.Tensor(.{ .seq, .attn }) = null;
            defer if (idx_w_t) |*t| t.deinit();
            if (dsa_here) {
                const dim = cfg.indexer_key_dim;
                var kt = try layer.idx_k.?.linearSeq(ctx, &h_t, .embed, .k);
                defer kt.deinit();
                const kd = try kt.dataConst();
                for (0..S) |s| {
                    const dst = cache.idx[layer_i][(pos0 + s) * dim ..][0..dim];
                    layerNormInto(dst, kd[s * dim ..][0..dim], layer.idx_k_norm_w.?, layer.idx_k_norm_b.?, dsa_ln_eps);
                }
                try self.dsaRopeKeyRows(ctx, cache.idx[layer_i][pos0 * dim .. (pos0 + S) * dim], S, &rope_table);

                if (selecting) {
                    idx_q_buf = try allocator.alloc(f32, S * cfg.indexer_heads * dim);
                    try self.dsaRopeQBuf(ctx, &idx_q_t.?, idx_q_buf, S, &rope_table);
                    idx_w_t = try layer.idx_proj.?.linearSeq(ctx, &h_t, .embed, .attn);
                    idx_w_all = try idx_w_t.?.dataConst();
                }
            }
            const stashing = selecting and dsa_here;
            if (stashing) cache.shareReset();
            if (selecting and dsa_shared_layer and (!cache.share_valid or cache.share_lens.items.len != S)) return Error.IndexShareMismatch;

            // Per-token attention with per-position visibility.
            const k_layer = cache.k[layer_i];
            var share_off: usize = 0;
            var rows_buf: std.ArrayList(f32) = .empty;
            defer rows_buf.deinit(allocator);
            for (0..S) |s| {
                const pos = pos0 + s;
                const t_len = pos + 1;
                rmsNormInto(latent_tmp, kv_a[s * (lora + cfg.qk_rope_dim) ..][0..lora], layer.kv_a_norm, cfg.rms_norm_eps);
                const dst = k_layer[pos * row_w ..][0..row_w];
                @memcpy(dst[0..lora], latent_tmp);
                @memcpy(dst[lora..], k_pe_all[s * cfg.qk_rope_dim ..][0..cfg.qk_rope_dim]);

                var sel_buf: ?[]usize = null;
                defer if (sel_buf) |sl| allocator.free(sl);
                var att_rows: []const f32 = k_layer[0 .. t_len * row_w];
                var att_n: usize = t_len;
                if (dsa_on and t_len > cfg.indexer_top_k) {
                    var sel: []usize = undefined;
                    if (dsa_shared_layer) {
                        const len = cache.share_lens.items[s];
                        if (len != cfg.indexer_top_k) return Error.IndexShareMismatch;
                        sel = try allocator.dupe(usize, cache.share_sel.items[share_off..][0..len]);
                        cache.share_reused += 1;
                    } else {
                        sel = try dsaScoreSelect(self, ctx, cache, layer_i, idx_q_buf[s * cfg.indexer_heads * cfg.indexer_key_dim ..][0 .. cfg.indexer_heads * cfg.indexer_key_dim], idx_w_all[s * cfg.indexer_heads ..][0..cfg.indexer_heads], t_len);
                        cache.share_computed += 1;
                        if (cache.probe) |*p| try p.record(layer_i, sel, t_len);
                    }
                    sel_buf = sel;
                    rows_buf.clearRetainingCapacity();
                    try rows_buf.ensureTotalCapacity(allocator, sel.len * row_w);
                    for (sel) |p| rows_buf.appendSliceAssumeCapacity(k_layer[p * row_w ..][0..row_w]);
                    att_rows = rows_buf.items;
                    att_n = sel.len;
                }
                if (dsa_shared_layer and dsa_on and t_len > cfg.indexer_top_k) share_off += cache.share_lens.items[s];
                if (stashing) {
                    if (sel_buf) |sl| {
                        try cache.share_sel.appendSlice(cache.allocator, sl);
                        try cache.share_lens.append(cache.allocator, sl.len);
                    } else {
                        try cache.share_lens.append(cache.allocator, 0);
                    }
                }

                // This token's q row through the decode-shaped contractions.
                var q_view = try q_rot.select(ctx, .seq, @intCast(s));
                defer q_view.deinit();
                var q_nope_v = try q_view.narrow(ctx, .d, 0, cfg.qk_nope_dim);
                defer q_nope_v.deinit();
                var q_nope_t = try q_nope_v.withTags(ctx, .{ .head, .nope });
                defer q_nope_t.deinit();
                var q_eff_t = try q_nope_t.dot(ctx, &wk_t, .nope);
                defer q_eff_t.deinit();
                var q_eff_d = try q_eff_t.withTags(ctx, .{ .head, .d });
                defer q_eff_d.deinit();
                var q_pe_v2 = try q_view.narrow(ctx, .d, cfg.qk_nope_dim, cfg.qk_rope_dim);
                defer q_pe_v2.deinit();
                var q_cat_t = try q_eff_d.concat(ctx, .d, &.{&q_pe_v2});
                defer q_cat_t.deinit();
                var rows_t = try fucina.Tensor(.{ .t, .d }).fromBorrowedConstSlice(ctx, .{ att_n, row_w }, att_rows);
                defer rows_t.deinit();
                var scores_t = try q_cat_t.dot(ctx, &rows_t, .d);
                defer scores_t.deinit();
                var scaled_t = try scores_t.scale(ctx, self.attn_scale);
                defer scaled_t.deinit();
                var probs_t = try scaled_t.softmax(ctx, .t, .{});
                defer probs_t.deinit();
                var lat_t = try rows_t.narrow(ctx, .d, 0, lora);
                defer lat_t.deinit();
                var ctx_t = try probs_t.dot(ctx, &lat_t, .t);
                defer ctx_t.deinit();
                var out_t = try ctx_t.dot(ctx, &wv_t, .d);
                defer out_t.deinit();
                try out_t.copyTo(attn_out[s * cfg.num_heads * cfg.v_head_dim ..][0 .. cfg.num_heads * cfg.v_head_dim]);
            }
            if (stashing) cache.share_valid = true;

            var attn_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ S, cfg.num_heads * cfg.v_head_dim }, attn_out);
            defer attn_t.deinit();
            var o_t = try layer.o_proj.linearSeq(ctx, &attn_t, .embed, .attn);
            defer o_t.deinit();
            for (xs, try o_t.dataConst()) |*xi, oi| xi.* += oi;

            // Router lookahead across the chunk: predict the NEXT layer's
            // experts from every row's post-attention state and stage the
            // UNION's loads while THIS layer's FFN computes. One hint per
            // layer per chunk — per-row hints would bump the store's
            // prediction epoch S times and flood the ring with repeats.
            if (self.pilot_enabled and layer_i + 1 < self.layers.len) {
                self.pilotPrefetchNextBatch(ctx, &self.layers[layer_i + 1], layer_i + 1, xs, S) catch {};
            }

            // ---- FFN (batched; MoE = union-routed fused batch) ----
            // Final layer: only row S-1 reaches the logits, and this layer's
            // KV was appended by the attention above — the FFN output of
            // rows 0..S-2 feeds nothing, so the last layer's FFN runs on the
            // final row alone (the qwen3/gemma4 truncation, MoE-batch form).
            const ffn_rows = if (layer_i + 1 == self.layers.len) 1 else S;
            const row0 = S - ffn_rows;
            const xs_ffn = xs[row0 * hidden ..][0 .. ffn_rows * hidden];
            for (0..ffn_rows) |s| rmsNormInto(hn[s * hidden ..][0..hidden], xs_ffn[s * hidden ..][0..hidden], layer.ffn_norm, cfg.rms_norm_eps);
            var f_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ ffn_rows, hidden }, hn[0 .. ffn_rows * hidden]);
            defer f_t.deinit();
            switch (layer.ffn) {
                .dense => |*dense| {
                    const y = try swigluLinear(ctx, allocator, &f_t, &dense.gate, &dense.up, &dense.down);
                    defer allocator.free(y);
                    for (xs_ffn, y) |*xi, yi| xi.* += yi;
                },
                .moe => |*moe| {
                    const y = try self.moeForwardBatch(ctx, allocator, moe, layer, &f_t, ffn_rows);
                    defer allocator.free(y);
                    for (xs_ffn, y) |*xi, yi| xi.* += yi;
                },
            }
        }
        if (cache.probe) |*p| p.stepDone();
        cache.len += S;

        rmsNormInto(hn[0..hidden], xs[(S - 1) * hidden ..][0..hidden], self.output_norm, cfg.rms_norm_eps);
        var final_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, hidden }, hn[0..hidden]);
        defer final_t.deinit();
        var logits_t = try self.output.linearSeq(ctx, &final_t, .embed, .vocab);
        defer logits_t.deinit();
        return allocator.dupe(f32, try logits_t.dataConst());
    }

    /// Batched routed mixture: per-row V2/V3 routing (same math as
    /// `moeForward`), one union-routed fused batch call, batched shared
    /// expert.
    fn moeForwardBatch(self: *const Model, ctx: *ExecContext, allocator: Allocator, moe: *const MoeFfn, layer: *const Layer, f_t: *const fucina.Tensor(.{ .seq, .embed }), S: usize) ![]f32 {
        const cfg = self.config;
        const used = cfg.num_experts_used;

        // The fused batch entry serves Q8_K-LHS arms only; q8_0-LHS expert
        // stacks (Q8_0 files) fall back to per-row decode-path calls —
        // correct everywhere, union-fetched where supported.
        if (moe.gate.wantsQ8_0Lhs() or moe.up.wantsQ8_0Lhs() or moe.down.wantsQ8_0Lhs()) {
            const y = try allocator.alloc(f32, S * cfg.hidden_size);
            errdefer allocator.free(y);
            for (0..S) |s| {
                var row_t = try f_t.narrow(ctx, .seq, s, 1);
                defer row_t.deinit();
                const row_y = try self.moeForward(ctx, allocator, moe, layer, &row_t);
                defer allocator.free(row_y);
                @memcpy(y[s * cfg.hidden_size ..][0..cfg.hidden_size], row_y);
            }
            return y;
        }
        var logits_t = try moe.router.linearSeq(ctx, f_t, .embed, .expert);
        defer logits_t.deinit();
        const probs = try allocator.dupe(f32, try logits_t.dataConst());
        defer allocator.free(probs);

        const selected = try allocator.alloc(usize, S * used);
        defer allocator.free(selected);
        const routing = try allocator.alloc(f32, S * used);
        defer allocator.free(routing);
        const choice = try allocator.alloc(f32, cfg.num_experts);
        defer allocator.free(choice);
        for (0..S) |s| {
            const row = probs[s * cfg.num_experts ..][0..cfg.num_experts];
            switch (cfg.expert_gating_func) {
                2 => for (row) |*v| {
                    v.* = 1.0 / (1.0 + @exp(-v.*));
                },
                else => softmaxInPlace(row),
            }
            @memcpy(choice, row);
            if (layer.router_bias) |bias| {
                for (choice, bias) |*c, b| c.* += b;
            }
            const sel = selected[s * used ..][0..used];
            const wts = routing[s * used ..][0..used];
            try topKExperts(ctx, choice, sel);
            for (sel, wts) |e, *w| w.* = row[e];
            if (cfg.expert_weights_norm) {
                var total: f32 = 1e-20;
                for (wts) |w| total += w;
                for (wts) |*w| w.* /= total;
            }
            for (wts) |*w| w.* *= cfg.expert_weights_scale;
        }

        var mix = try weights.moeSwiGluFfnSeq(ctx, f_t, &moe.gate, &moe.up, &moe.down, selected, routing, used, cfg.expert_ffn_size, null, null);
        defer mix.deinit();
        const y = try allocator.dupe(f32, try mix.dataConst());
        errdefer allocator.free(y);

        const shared = try swigluLinear(ctx, allocator, f_t, &moe.shared_gate, &moe.shared_up, &moe.shared_down);
        defer allocator.free(shared);
        for (y, shared) |*yi, si| yi.* += si;
        return y;
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

    /// Chunk-wide router lookahead: one batched router pass over every
    /// row's post-attention state, one `pilotHint` with the DEDUPED union
    /// of predicted experts — the per-row selection math is identical to
    /// `pilotPrefetchNext` (the decode pilot keeps that entry).
    fn pilotPrefetchNextBatch(self: *const Model, ctx: *ExecContext, next: *const Layer, next_layer_i: usize, xs: []const f32, S: usize) !void {
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
        const hidden = cfg.hidden_size;

        const h = try allocator.alloc(f32, S * hidden);
        defer allocator.free(h);
        for (0..S) |s| rmsNormInto(h[s * hidden ..][0..hidden], xs[s * hidden ..][0..hidden], next.ffn_norm, cfg.rms_norm_eps);
        var h_t = try fucina.Tensor(.{ .seq, .embed }).fromBorrowedConstSlice(ctx, .{ S, hidden }, h);
        defer h_t.deinit();
        var logits_t = try moe.router.linearSeq(ctx, &h_t, .embed, .expert);
        defer logits_t.deinit();
        const logits = try logits_t.dataConst();

        const in_union = try allocator.alloc(bool, cfg.num_experts);
        defer allocator.free(in_union);
        @memset(in_union, false);
        const choice = try allocator.alloc(f32, cfg.num_experts);
        defer allocator.free(choice);
        var uni = try std.ArrayList(usize).initCapacity(allocator, S * cfg.num_experts_used);
        defer uni.deinit(allocator);
        var sel: [64]usize = undefined;
        std.debug.assert(cfg.num_experts_used <= sel.len);
        for (0..S) |s| {
            @memcpy(choice, logits[s * cfg.num_experts ..][0..cfg.num_experts]);
            switch (cfg.expert_gating_func) {
                2 => for (choice) |*v| {
                    v.* = 1.0 / (1.0 + @exp(-v.*));
                },
                else => softmaxInPlace(choice),
            }
            if (next.router_bias) |bias| {
                for (choice, bias) |*c, b| c.* += b;
            }
            try topKExperts(ctx, choice, sel[0..cfg.num_experts_used]);
            for (sel[0..cfg.num_experts_used]) |e| {
                if (!in_union[e]) {
                    in_union[e] = true;
                    uni.appendAssumeCapacity(e);
                }
            }
        }
        store.pilotHint(next_layer_i, uni.items);
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
    /// Dynamic expert-drop filter (both dials off = never called): sorts
    /// the selection weight-descending, then keeps experts until
    /// `moe_top_p` of the selected gate mass is covered; independently, a
    /// below-`moe_skip_miss_below` expert that is neither pinned nor
    /// cached is dropped instead of read from disk (resident experts are
    /// always kept — they cost nothing). The top-weight expert always
    /// survives. Compacts `sel`/`wts` in place and returns the kept count;
    /// weights stay raw (the caller renormalizes over what remains).
    fn dynamicExpertFilter(self: *const Model, moe: *const MoeFfn, sel: []usize, wts: []f32) usize {
        const k = sel.len;
        var i: usize = 1;
        while (i < k) : (i += 1) {
            const e = sel[i];
            const w = wts[i];
            var j = i;
            while (j > 0 and wts[j - 1] < w) : (j -= 1) {
                sel[j] = sel[j - 1];
                wts[j] = wts[j - 1];
            }
            sel[j] = e;
            wts[j] = w;
        }
        var total: f32 = 1e-20;
        for (wts) |w| total += w;
        const streamed = switch (moe.gate) {
            .streamed => |*st| st,
            else => null,
        };
        var kept: usize = 1;
        var cum: f32 = wts[0];
        for (1..k) |idx| {
            if (self.moe_top_p < 1.0 and cum >= self.moe_top_p * total) break;
            const w = wts[idx];
            if (self.moe_skip_miss_below > 0 and w < self.moe_skip_miss_below * total) {
                if (streamed) |st| {
                    if (!st.store.isResident(st.layer, sel[idx])) continue;
                }
            }
            sel[kept] = sel[idx];
            wts[kept] = w;
            kept += 1;
            cum += w;
        }
        return kept;
    }

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
        var used = cfg.num_experts_used;
        try topKExperts(ctx, choice, selected[0..used]);
        for (selected[0..used], routing[0..used]) |e, *w| w.* = probs[e];
        if (self.moe_top_p < 1.0 or self.moe_skip_miss_below > 0) {
            used = self.dynamicExpertFilter(moe, selected[0..used], routing[0..used]);
        }
        if (cfg.expert_weights_norm) {
            var total: f32 = 1e-20;
            for (routing[0..used]) |w| total += w;
            for (routing[0..used]) |*w| w.* /= total;
        }
        for (routing[0..used]) |*w| w.* *= cfg.expert_weights_scale;

        var mix = try weights.moeSwiGluFfnSeq(
            ctx,
            f_t,
            &moe.gate,
            &moe.up,
            &moe.down,
            selected[0..used],
            routing[0..used],
            used,
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

/// LayerNorm (weight + bias) for the DSA indexer key — the family's one
/// non-RMS norm. eps per the V3.2 reference indexer.
const dsa_ln_eps: f32 = 1e-6;

fn layerNormInto(dst: []f32, src: []const f32, w: []const f32, b: []const f32, eps: f32) void {
    var mean: f64 = 0;
    for (src) |v| mean += v;
    mean /= @as(f64, @floatFromInt(src.len));
    var variance: f64 = 0;
    for (src) |v| {
        const d = @as(f64, v) - mean;
        variance += d * d;
    }
    variance /= @as(f64, @floatFromInt(src.len));
    const inv: f32 = @floatCast(1.0 / @sqrt(variance + eps));
    for (dst, src, w, b) |*o, v, wi, bi| o.* = @as(f32, @floatCast(@as(f64, v) - mean)) * inv * wi + bi;
}

/// Compute and cache this position's DSA indexer key:
/// `rope(LayerNorm(idx_k(h)))` with the rotation on the LEADING
/// qk_rope_dim dims (the reference indexer's [pe | nope] head layout),
/// through the same yarn table and pairing as the main attention.
fn dsaAppendKey(self: *const Model, ctx: *ExecContext, cache: *Cache, layer: *const Layer, layer_i: usize, h_t: anytype, pos: usize, rope_table: anytype) !void {
    const cfg = self.config;
    var kt = try layer.idx_k.?.linearSeq(ctx, h_t, .embed, .k);
    defer kt.deinit();
    const w = cache.idx_width;
    const dst = cache.idx[layer_i][pos * w ..][0..w];
    layerNormInto(dst, try kt.dataConst(), layer.idx_k_norm_w.?, layer.idx_k_norm_b.?, dsa_ln_eps);
    _ = cfg;
    try self.dsaRopeKeyRows(ctx, dst, 1, rope_table);
}

/// DSA lightning-indexer selection over the cached keys:
/// `score_s = Σ_h w_h · ReLU(q_h · k_s)`, top-`indexer_top_k` positions
/// ascending. The reference's positive global scales (heads^-0.5, the
/// softmax scale) cannot change the ranking and are omitted; `w` keeps its
/// sign. Returned slice is caller-owned.
fn dsaSelect(self: *const Model, ctx: *ExecContext, cache: *Cache, layer: *const Layer, layer_i: usize, idx_q_t: anytype, h_t: anytype, t_len: usize, rope_table: anytype) ![]usize {
    const cfg = self.config;
    const allocator = ctx.allocator;
    const heads = cfg.indexer_heads;
    const dim = cfg.indexer_key_dim;

    // Queries: [heads, dim] with the rope treatment applied.
    const q_buf = try allocator.alloc(f32, heads * dim);
    defer allocator.free(q_buf);
    try self.dsaRopeQBuf(ctx, idx_q_t, q_buf, 1, rope_table);

    // Per-head score weights from the pre-attention hidden state.
    var w_t = try layer.idx_proj.?.linearSeq(ctx, h_t, .embed, .attn);
    defer w_t.deinit();

    return dsaScoreSelect(self, ctx, cache, layer_i, q_buf, try w_t.dataConst(), t_len);
}

/// Scoring/selection core shared by decode and batched prefill: takes ONE
/// token's already-roped indexer q heads and its raw head weights, scores
/// the cached keys, returns the top-k positions ascending (caller-owned).
fn dsaScoreSelect(self: *const Model, ctx: *ExecContext, cache: *Cache, layer_i: usize, q_buf: []const f32, head_w: []const f32, t_len: usize) ![]usize {
    const cfg = self.config;
    const allocator = ctx.allocator;
    const heads = cfg.indexer_heads;
    const dim = cfg.indexer_key_dim;

    // score = sum_h relu(k . q_h) * w_h in one batched contraction chain.
    var keys_t = try fucina.Tensor(.{ .t, .d }).fromBorrowedConstSlice(ctx, .{ t_len, dim }, cache.idx[layer_i][0 .. t_len * dim]);
    defer keys_t.deinit();
    var q_t2 = try fucina.Tensor(.{ .head, .d }).fromBorrowedConstSlice(ctx, .{ heads, dim }, q_buf);
    defer q_t2.deinit();
    var dots = try keys_t.dot(ctx, &q_t2, .d);
    defer dots.deinit();
    var rectified = try dots.unary(ctx, .relu);
    defer rectified.deinit();
    var w_row = try fucina.Tensor(.{.head}).fromBorrowedConstSlice(ctx, .{heads}, head_w);
    defer w_row.deinit();
    var weighted = try rectified.mul(ctx, &w_row);
    defer weighted.deinit();
    var scores_t = try weighted.sum(ctx, .head);
    defer scores_t.deinit();

    var top = try scores_t.topK(ctx, .t, cfg.indexer_top_k, .k);
    defer top.values.deinit();
    defer top.indices.deinit();
    const sel = try allocator.alloc(usize, cfg.indexer_top_k);
    errdefer allocator.free(sel);
    for (sel, try top.indices.dataConst()) |*s, v| s.* = @intCast(v);
    std.mem.sort(usize, sel, {}, std.sort.asc(usize));
    return sel;
}

fn loadLayer(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize, store: ?*fucina.ExpertStore, dsa: bool) !Layer {
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

    var idx_k: ?LinearWeight = null;
    errdefer if (idx_k) |*w| w.deinit();
    var idx_k_norm_w: ?[]f32 = null;
    errdefer if (idx_k_norm_w) |n| allocator.free(n);
    var idx_k_norm_b: ?[]f32 = null;
    errdefer if (idx_k_norm_b) |n| allocator.free(n);
    var idx_q_b: ?LinearWeight = null;
    errdefer if (idx_q_b) |*w| w.deinit();
    var idx_proj: ?LinearWeight = null;
    errdefer if (idx_proj) |*w| w.deinit();
    if (dsa) {
        if (config.indexer_top_k == 0 or config.q_lora_rank == 0) return Error.InvalidConfig;
        // Per-layer presence is the Full/Shared discriminator: GLM-5.2's
        // native IndexShare ships indexer weights only on its Full layers
        // (~1 in 4); weightless layers reuse the previous Full layer's
        // selection at run time (the trained behavior — transformers raises
        // when the stash is missing, and so do we).
        // Some GGUF conversions pad the Shared layers with byte-identical
        // COPIES of the previous Full layer's indexer instead of omitting
        // the tensors. The trained behavior on those layers is to REUSE
        // the previous Full layer's selection — running the copied indexer
        // against this layer's activations produces garbage selections —
        // so a byte-equal indexer demotes the layer to Shared.
        const copied_from_prev = layer_i > 0 and blk: {
            var prev_buf: [96]u8 = undefined;
            const cur_k = file.maybeGet(try name.of(&name_buf, layer_i, "indexer.attn_k.weight")) orelse break :blk false;
            const prev_k = file.maybeGet(try name.of(&prev_buf, layer_i - 1, "indexer.attn_k.weight")) orelse break :blk false;
            if (cur_k.ggml_type != prev_k.ggml_type or !std.mem.eql(u8, prev_k.data, cur_k.data)) break :blk false;
            const cur_q = file.maybeGet(try name.of(&name_buf, layer_i, "indexer.attn_q_b.weight")) orelse break :blk false;
            const prev_q = file.maybeGet(try name.of(&prev_buf, layer_i - 1, "indexer.attn_q_b.weight")) orelse break :blk false;
            break :blk cur_q.ggml_type == prev_q.ggml_type and std.mem.eql(u8, prev_q.data, cur_q.data);
        };
        if (!copied_from_prev) if (file.maybeGet(try name.of(&name_buf, layer_i, "indexer.attn_k.weight"))) |ik_info| {
            idx_k = try LinearWeight.load(ctx, ik_info, config.indexer_key_dim, config.hidden_size);
            idx_k_norm_w = try hostVector(allocator, file, try name.of(&name_buf, layer_i, "indexer.k_norm.weight"), config.indexer_key_dim);
            idx_k_norm_b = try hostVector(allocator, file, try name.of(&name_buf, layer_i, "indexer.k_norm.bias"), config.indexer_key_dim);
            idx_q_b = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "indexer.attn_q_b.weight")), config.indexer_heads * config.indexer_key_dim, config.q_lora_rank);
            idx_proj = try LinearWeight.load(ctx, try file.get(try name.of(&name_buf, layer_i, "indexer.proj.weight")), config.indexer_heads, config.hidden_size);
        };
    }

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
        .idx_k = idx_k,
        .idx_k_norm_w = idx_k_norm_w,
        .idx_k_norm_b = idx_k_norm_b,
        .idx_q_b = idx_q_b,
        .idx_proj = idx_proj,
        .ffn = ffn,
    };
}

/// SwiGLU through three resident linears, result as a host row.
fn swigluLinear(ctx: *ExecContext, allocator: Allocator, x: *const fucina.Tensor(.{ .seq, .embed }), gate: *const LinearWeight, up: *const LinearWeight, down: *const LinearWeight) ![]f32 {
    const rows = x.dim(.seq);
    var gate_t = try gate.linearSeq(ctx, x, .embed, .gate_up);
    defer gate_t.deinit();
    var up_t = try up.linearSeq(ctx, x, .embed, .gate_up);
    defer up_t.deinit();
    const width = gate_t.dim(.gate_up);
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

test "deepseek2 config rejects non-deepseek architectures" {
    // Covered structurally: Config.fromGguf requires general.architecture ==
    // "deepseek2"; real-model behavior is validated by the runner (the
    // GGUF-dependent tests live with models/ and skip without it).
}
