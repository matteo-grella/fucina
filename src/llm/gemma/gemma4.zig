//! Gemma 4 (`gemma4` GGUF arch) text-only CPU inference.
//!
//! Source of truth: `refs/llama.cpp/src/models/gemma4.cpp` (graph) +
//! `load_arch_hparams`/`load_arch_tensors` in that file, validated against the
//! actual `unsloth/gemma-4-26B-A4B-it` GGUF. North star is logit parity with
//! llama.cpp on the same GGUF. The forward is built from the general
//! `ExecContext`/facade ops; Gemma-specific composition lives here.
//!
//! Confirmed shape (26B-A4B, from the GGUF metadata):
//!   - 30 layers, 16 query heads; KV heads PER LAYER: 8 on local-SWA layers,
//!     2 on the 5 global layers (il where sliding_window_pattern[il]==false:
//!     5,11,17,23,29). head_dim 256 (SWA) / 512 (global). softmax scale 1.0.
//!   - sandwich RMS norms (pre+post on attn and FFN), QK-norm, no-weight V-norm.
//!   - FFN = always-on shared dense GeGLU MLP (Q8_0) + 128-expert top-8 MoE with
//!     FUSED `ffn_gate_up_exps` (Q6_K, out = 2*n_ff_exp) and a `ffn_down_exps`
//!     (Q8_0, contraction 704 — not 256-aligned, hence Q8_0 not K-quant) plus a
//!     per-expert `ffn_down_exps.scale`. Router runs on rms_norm(attn_out).
//!   - per-layer freq_base (global layers add proportional RoPE via rope_freqs).
//!   - final logit softcap. shared_kv_layers=0 and per-layer-embeddings=0 in this
//!     GGUF (both paths are kept but inactive).
const std = @import("std");
const fucina = @import("fucina");
const weights = @import("../weights.zig");
const kv_cache = @import("../kv_cache.zig");
const gguf_meta = @import("../gguf_meta.zig");
const gemma_moe = @import("moe.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;
const gguf = fucina.gguf;
const LinearWeight = weights.LinearWeight;
const RhsQ6_K = fucina.QuantizedMatmulRhsQ6_Kx4;
const RhsQ8_0 = fucina.QuantizedMatmulRhsQ8_0x4;

pub const Error = weights.Error || error{
    InvalidConfig,
    InvalidSequenceLength,
    MismatchedKvCaches,
    MissingMetadata,
    PleUnsupported,
    UnsupportedExpertType,
    UnsupportedKvCacheDtype,
};

pub const ForwardProfile = struct {
    embed_ns: i128 = 0,
    attn_ns: i128 = 0,
    ffn_ns: i128 = 0,
    dense_ns: i128 = 0,
    moe_router_ns: i128 = 0,
    moe_count_sort_ns: i128 = 0,
    moe_gather_ns: i128 = 0,
    moe_expert_wall_ns: i128 = 0,
    moe_gate_up_ns: i128 = 0,
    moe_act_ns: i128 = 0,
    moe_down_ns: i128 = 0,
    moe_task_gate_up_ns: i128 = 0,
    moe_task_act_ns: i128 = 0,
    moe_task_down_ns: i128 = 0,
    moe_scatter_ns: i128 = 0,
    final_ns: i128 = 0,
    layers: usize = 0,
};

pub const Config = struct {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_attention_heads: usize, // query heads (16); KV heads vary per layer (geometry)
    head_dim_global: usize, // GGUF attention.key_length
    head_dim_swa: usize, // GGUF attention.key_length_swa
    sliding_window: usize,
    shared_kv_layers: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_theta_swa: f32,
    num_experts: usize,
    num_experts_used: usize,
    moe_intermediate_size: usize, // per-expert FFN width (expert_feed_forward_length)
    intermediate_size: usize, // shared dense MLP width (feed_forward_length)
    per_layer_input_size: usize, // PLE width (0 = disabled)
    final_logit_softcapping: f32, // 0 = disabled
    /// Load-time policy (not a GGUF hparam): when true AND the file is mmap'd
    /// AND this is a CPU build, MoE experts are borrowed zero-copy from the
    /// mapping instead of x4-packed/copied — near-instant load, ~half memory.
    /// Set by `--experts=borrow`; default false keeps the packed fast-inference
    /// path. See `loadMoe`.
    borrow_experts: bool = false,

    pub fn fromGguf(file: *const gguf.File) !Config {
        return fromGgufArch(file, "gemma4");
    }

    /// As `fromGguf` for an architecture that shares the gemma4 hparam key
    /// set under its own prefix (diffusion-gemma's `diffusion-gemma.*` keys
    /// are 1:1 with `gemma4.*`).
    pub fn fromGgufArch(file: *const gguf.File, expected_arch: []const u8) !Config {
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        if (!std.mem.eql(u8, arch, expected_arch)) return Error.InvalidConfig;
        const embd = try file.get("token_embd.weight");
        const shape = try embd.logicalMatrixShape(); // {vocab, hidden}
        const head_dim_global = try metaInt(file, arch, "attention.key_length");
        return .{
            .vocab_size = shape[0],
            .hidden_size = try metaInt(file, arch, "embedding_length"),
            .num_layers = try metaInt(file, arch, "block_count"),
            .num_attention_heads = try metaInt(file, arch, "attention.head_count"),
            .head_dim_global = head_dim_global,
            .head_dim_swa = metaIntOpt(file, arch, "attention.key_length_swa") orelse head_dim_global,
            .sliding_window = metaIntOpt(file, arch, "attention.sliding_window") orelse 0,
            .shared_kv_layers = metaIntOpt(file, arch, "attention.shared_kv_layers") orelse 0,
            .rms_norm_eps = try metaFloat(file, arch, "attention.layer_norm_rms_epsilon"),
            .rope_theta = metaFloatOpt(file, arch, "rope.freq_base") orelse 10_000,
            .rope_theta_swa = metaFloatOpt(file, arch, "rope.freq_base_swa") orelse 10_000,
            .num_experts = metaIntOpt(file, arch, "expert_count") orelse 0,
            .num_experts_used = metaIntOpt(file, arch, "expert_used_count") orelse 0,
            .moe_intermediate_size = metaIntOpt(file, arch, "expert_feed_forward_length") orelse 0,
            .intermediate_size = try metaInt(file, arch, "feed_forward_length"),
            .per_layer_input_size = metaIntOpt(file, arch, "embedding_length_per_layer_input") orelse 0,
            .final_logit_softcapping = metaFloatOpt(file, arch, "final_logit_softcapping") orelse 0,
        };
    }

    pub fn validate(self: Config) !void {
        if (self.num_attention_heads == 0) return Error.InvalidConfig;
        if (self.head_dim_global % 2 != 0 or self.head_dim_swa % 2 != 0) return Error.InvalidConfig;
        if (self.num_experts > 0 and (self.num_experts_used == 0 or self.num_experts_used > self.num_experts)) return Error.InvalidConfig;
        if (self.shared_kv_layers >= self.num_layers) return Error.InvalidConfig;
        if (self.num_attention_heads > max_heads) return Error.InvalidConfig;
    }
};

pub const max_heads = 64;

// Gemma reads legitimately-zero keys (e.g. `attention.shared_kv_layers`), so
// zero-valued ints are accepted (`.accept_zero`).
pub fn metaInt(file: *const gguf.File, arch: []const u8, suffix: []const u8) !usize {
    return gguf_meta.metaInt(file, arch, suffix, .accept_zero);
}

pub fn metaIntOpt(file: *const gguf.File, arch: []const u8, suffix: []const u8) ?usize {
    return gguf_meta.metaIntOpt(file, arch, suffix, .accept_zero);
}

pub const metaFloat = gguf_meta.metaFloat;
pub const metaFloatOpt = gguf_meta.metaFloatOpt;

/// Per-layer geometry derived from the SWA pattern, the per-layer KV-head array,
/// and the shared-KV count. Mirrors llama.cpp's `is_swa`/`n_embd_head_k`/
/// `n_head_kv`/`has_kv` accessors.
pub const LayerGeometry = struct {
    is_swa: []bool,
    head_dim: []usize,
    kv_heads: []usize,
    has_kv: []bool,
    kv_ref: []usize,

    pub fn deinit(self: *LayerGeometry, allocator: Allocator) void {
        allocator.free(self.is_swa);
        allocator.free(self.head_dim);
        allocator.free(self.kv_heads);
        allocator.free(self.has_kv);
        allocator.free(self.kv_ref);
        self.* = undefined;
    }
};

/// Derive per-layer geometry. `swa_pattern[il]==true` marks a local SWA layer
/// (false = global full-attention); `kv_heads_in[il]` is that layer's KV-head
/// count. The trailing `shared_kv_layers` layers reuse an earlier same-type
/// layer's K/V (offset 2 for SWA, 1 for global).
pub fn deriveGeometry(
    allocator: Allocator,
    n_layer: usize,
    swa_pattern: []const bool,
    kv_heads_in: []const usize,
    shared_kv_layers: usize,
    head_dim_global: usize,
    head_dim_swa: usize,
) !LayerGeometry {
    std.debug.assert(swa_pattern.len == n_layer and kv_heads_in.len == n_layer);
    const is_swa = try allocator.alloc(bool, n_layer);
    errdefer allocator.free(is_swa);
    const head_dim = try allocator.alloc(usize, n_layer);
    errdefer allocator.free(head_dim);
    const kv_heads = try allocator.alloc(usize, n_layer);
    errdefer allocator.free(kv_heads);
    const has_kv = try allocator.alloc(bool, n_layer);
    errdefer allocator.free(has_kv);
    const kv_ref = try allocator.alloc(usize, n_layer);
    errdefer allocator.free(kv_ref);

    const kv_from_start = n_layer - shared_kv_layers;
    for (0..n_layer) |il| {
        is_swa[il] = swa_pattern[il];
        head_dim[il] = if (swa_pattern[il]) head_dim_swa else head_dim_global;
        kv_heads[il] = kv_heads_in[il];
        has_kv[il] = il < kv_from_start;
        if (has_kv[il]) {
            kv_ref[il] = il;
        } else {
            const offset: usize = if (swa_pattern[il]) 2 else 1;
            kv_ref[il] = if (kv_from_start >= offset) kv_from_start - offset else 0;
        }
    }
    return .{ .is_swa = is_swa, .head_dim = head_dim, .kv_heads = kv_heads, .has_kv = has_kv, .kv_ref = kv_ref };
}

/// Read a per-layer metadata array (bool/int), broadcasting a scalar value
/// across all layers (mirrors llama.cpp `get_key_or_arr`).
pub fn readU32OrBoolArray(allocator: Allocator, file: *const gguf.File, key: []const u8, n_layer: usize, comptime T: type) ![]T {
    const out = try allocator.alloc(T, n_layer);
    errdefer allocator.free(out);
    if (file.getArray(key)) |arr| {
        if (arr.len != n_layer) return Error.InvalidConfig;
        switch (arr.item_type) {
            7, 0, 1 => for (out, 0..) |*s, i| {
                s.* = if (T == bool) (arr.data[i] != 0) else @as(T, arr.data[i]);
            },
            4, 5 => for (out, 0..) |*s, i| {
                const v = std.mem.readInt(u32, arr.data[i * 4 ..][0..4], .little);
                s.* = if (T == bool) (v != 0) else @intCast(v);
            },
            10, 11 => for (out, 0..) |*s, i| {
                const v = std.mem.readInt(u64, arr.data[i * 8 ..][0..8], .little);
                s.* = if (T == bool) (v != 0) else @intCast(v);
            },
            else => return Error.InvalidConfig,
        }
        return out;
    }
    // scalar broadcast
    const scalar = file.getInt(key) orelse return Error.MissingMetadata;
    for (out) |*s| s.* = if (T == bool) (scalar != 0) else @intCast(scalar);
    return out;
}

const Vec = fucina.Tensor(.{.embed});

const PerLayerEmbeddings = struct {
    tok_embd: LinearWeight,
    model_proj: LinearWeight,
    proj_norm: fucina.Tensor(.{.ple}),

    fn deinit(self: *PerLayerEmbeddings) void {
        self.proj_norm.deinit();
        self.model_proj.deinit();
        self.tok_embd.deinit();
        self.* = undefined;
    }
};

/// MoE weights. Experts are stored as per-expert packed matmul RHS so the FFN
/// composes from the tested optimized Q6_K / Q8_0 packed kernels: the fused
/// `ffn_gate_up_exps` (Q6_K) is split per expert into `gate`/`up` (first/second
/// half of the 2*n_ff output), and `down` is the Q8_0 down projection.
/// Q4_K-transcoded gate_up experts (an experts-only requantize via
/// `export-gguf --experts-dtype q4_k`, cutting decode weight bandwidth) skip
/// the x4 packing on every build and run the raw-block paths over
/// `gpu_weights`.
pub const MoeFfn = struct {
    router: LinearWeight, // ffn_gate_inp.weight, f32 [n_expert, hidden]
    router_weight: Vec, // ffn_gate_inp.scale folded with 1/sqrt(hidden)
    pre_norm_2: Vec,
    post_norm_1: Vec,
    post_norm_2: Vec,
    gate: []RhsQ6_K,
    up: []RhsQ6_K,
    down: []RhsQ8_0,
    down_scale: []f32, // ffn_down_exps.scale, per-expert output scale
    // Resident copies of the raw GGUF expert blocks, populated on
    // -Dgpu=metal builds (the single expert representation there: the grouped
    // Metal GEMM and the raw CPU paths both read them) and on CPU builds
    // whose gate_up experts are Q4_K (no x4 packing exists for that arm — the
    // raw CPU paths are the consumers). Preferably DEVICE-OWNED memory
    // (`fucina.internal.gpu.allocResidentBytes`): pageable client wraps are re-wired into the
    // GPU address space on every dispatch (~45 µs/MB, tens of ms per layer);
    // device-owned buffers stay GPU-resident, and the CPU reads the same
    // bytes (unified memory). The `device_owned = false` plain-allocator
    // fallback (GPU alloc failed, or any CPU build) is freed by deinit and
    // is dispatched with uncached wraps (see gemma_moe.RawExpertWeights).
    gpu_weights: ?gemma_moe.RawExpertWeights,

    fn deinit(self: *MoeFfn, allocator: Allocator) void {
        if (self.gpu_weights) |gw| {
            // device-owned storage belongs to the shim (process lifetime —
            // the wrap cache may still hold its GPU mapping); borrowed blocks
            // live in the GGUF mapping the Model owns. Only heap copies free.
            if (!gw.device_owned and !gw.borrowed) {
                switch (gw.gu) {
                    inline else => |gu_blocks| allocator.free(gu_blocks),
                }
                allocator.free(gw.dn_blocks);
            }
        }
        for (self.gate) |*r| r.deinit();
        for (self.up) |*r| r.deinit();
        for (self.down) |*r| r.deinit();
        allocator.free(self.gate);
        allocator.free(self.up);
        allocator.free(self.down);
        allocator.free(self.down_scale);
        self.post_norm_2.deinit();
        self.post_norm_1.deinit();
        self.pre_norm_2.deinit();
        self.router_weight.deinit();
        self.router.deinit();
        self.* = undefined;
    }
};

pub const PerLayerInject = struct {
    inp_gate: LinearWeight,
    proj: LinearWeight,
    post_norm: Vec,

    fn deinit(self: *PerLayerInject) void {
        self.post_norm.deinit();
        self.proj.deinit();
        self.inp_gate.deinit();
        self.* = undefined;
    }
};

pub const SeparateAttentionProjection = struct {
    q_proj: LinearWeight,
    k_proj: ?LinearWeight,
    v_proj: ?LinearWeight,

    fn toResidentF16(self: *SeparateAttentionProjection, ctx: *ExecContext) !void {
        try self.q_proj.toResidentF16(ctx);
        if (self.k_proj) |*w| try w.toResidentF16(ctx);
        if (self.v_proj) |*w| try w.toResidentF16(ctx);
    }

    fn deinit(self: *SeparateAttentionProjection) void {
        if (self.v_proj) |*w| w.deinit();
        if (self.k_proj) |*w| w.deinit();
        self.q_proj.deinit();
        self.* = undefined;
    }
};

pub const FusedAttentionProjectionKind = enum { qk, qkv };

pub const FusedAttentionProjection = struct {
    weight: LinearWeight,
    kind: FusedAttentionProjectionKind,

    fn toResidentF16(self: *FusedAttentionProjection, ctx: *ExecContext) !void {
        try self.weight.toResidentF16(ctx);
    }

    fn deinit(self: *FusedAttentionProjection) void {
        self.weight.deinit();
        self.* = undefined;
    }
};

pub const AttentionProjectionResult = struct {
    q: fucina.Tensor(.{ .seq, .q }),
    k: ?fucina.Tensor(.{ .seq, .k }) = null,
    v: ?fucina.Tensor(.{ .seq, .v }) = null,

    pub fn deinit(self: *AttentionProjectionResult) void {
        if (self.v) |*value| value.deinit();
        if (self.k) |*value| value.deinit();
        self.q.deinit();
        self.* = undefined;
    }
};

pub const AttentionProjection = union(enum) {
    separate: SeparateAttentionProjection,
    fused: FusedAttentionProjection,

    pub fn toResidentF16(self: *AttentionProjection, ctx: *ExecContext) !void {
        switch (self.*) {
            .separate => |*separate| try separate.toResidentF16(ctx),
            .fused => |*fused| try fused.toResidentF16(ctx),
        }
    }

    fn deinit(self: *AttentionProjection) void {
        switch (self.*) {
            .separate => |*separate| separate.deinit(),
            .fused => |*fused| fused.deinit(),
        }
        self.* = undefined;
    }

    pub fn project(
        self: *const AttentionProjection,
        ctx: *ExecContext,
        input: *const fucina.Tensor(.{ .seq, .embed }),
        q_dim: usize,
        kv_dim: usize,
    ) !AttentionProjectionResult {
        return switch (self.*) {
            .separate => |*separate| blk: {
                var q = try separate.q_proj.linearSeq(ctx, input, .embed, .q);
                errdefer q.deinit();

                var k: ?fucina.Tensor(.{ .seq, .k }) = null;
                errdefer if (k) |*value| value.deinit();
                if (separate.k_proj) |*k_proj| {
                    k = try k_proj.linearSeq(ctx, input, .embed, .k);
                }

                var v: ?fucina.Tensor(.{ .seq, .v }) = null;
                errdefer if (v) |*value| value.deinit();
                if (separate.v_proj) |*v_proj| {
                    v = try v_proj.linearSeq(ctx, input, .embed, .v);
                }

                break :blk .{ .q = q, .k = k, .v = v };
            },
            .fused => |*fused| switch (fused.kind) {
                .qk => blk: {
                    var qk = try fused.weight.linearSeq(ctx, input, .embed, .qk);
                    defer qk.deinit();
                    break :blk try splitFusedQk(ctx, &qk, q_dim, kv_dim);
                },
                .qkv => blk: {
                    var qkv = try fused.weight.linearSeq(ctx, input, .embed, .qkv);
                    defer qkv.deinit();
                    break :blk try splitFusedQkv(ctx, &qkv, q_dim, kv_dim);
                },
            },
        };
    }
};

fn splitFusedQk(
    ctx: *ExecContext,
    qk: *const fucina.Tensor(.{ .seq, .qk }),
    q_dim: usize,
    kv_dim: usize,
) !AttentionProjectionResult {
    var q_view = try qk.narrow(ctx, .qk, 0, q_dim);
    defer q_view.deinit();
    var q = try q_view.withTags(ctx, .{ .seq, .q });
    errdefer q.deinit();

    var k_view = try qk.narrow(ctx, .qk, q_dim, kv_dim);
    defer k_view.deinit();
    var k = try k_view.withTags(ctx, .{ .seq, .k });
    errdefer k.deinit();

    return .{ .q = q, .k = k };
}

fn splitFusedQkv(
    ctx: *ExecContext,
    qkv: *const fucina.Tensor(.{ .seq, .qkv }),
    q_dim: usize,
    kv_dim: usize,
) !AttentionProjectionResult {
    var q_view = try qkv.narrow(ctx, .qkv, 0, q_dim);
    defer q_view.deinit();
    var q = try q_view.withTags(ctx, .{ .seq, .q });
    errdefer q.deinit();

    var k_view = try qkv.narrow(ctx, .qkv, q_dim, kv_dim);
    defer k_view.deinit();
    var k = try k_view.withTags(ctx, .{ .seq, .k });
    errdefer k.deinit();

    var v_view = try qkv.narrow(ctx, .qkv, q_dim + kv_dim, kv_dim);
    defer v_view.deinit();
    var v = try v_view.withTags(ctx, .{ .seq, .v });
    errdefer v.deinit();

    return .{ .q = q, .k = k, .v = v };
}

pub const Layer = struct {
    attn_norm: Vec,
    attn_post_norm: Vec,
    attn_proj: AttentionProjection,
    q_norm: fucina.Tensor(.{.d}),
    k_norm: ?fucina.Tensor(.{.d}),
    o_proj: LinearWeight,
    ffn_norm: Vec,
    ffn_gate: LinearWeight,
    ffn_up: LinearWeight,
    ffn_down: LinearWeight,
    ffn_post_norm: Vec,
    moe: ?MoeFfn,
    ple: ?PerLayerInject,
    out_scale: ?f32,

    pub fn deinit(self: *Layer, allocator: Allocator) void {
        if (self.ple) |*p| p.deinit();
        if (self.moe) |*m| m.deinit(allocator);
        self.ffn_post_norm.deinit();
        self.ffn_down.deinit();
        self.ffn_up.deinit();
        self.ffn_gate.deinit();
        self.ffn_norm.deinit();
        self.o_proj.deinit();
        if (self.k_norm) |*t| t.deinit();
        self.q_norm.deinit();
        self.attn_proj.deinit();
        self.attn_post_norm.deinit();
        self.attn_norm.deinit();
        self.* = undefined;
    }
};

pub const Model = struct {
    allocator: Allocator,
    config: Config,
    geom: LayerGeometry,
    token_embedding: LinearWeight,
    output_norm: Vec,
    output: LinearWeight,
    rope_freqs: ?fucina.Tensor(.{.rope}),
    layers: []Layer,
    ple: ?PerLayerEmbeddings,
    /// The GGUF mapping, owned by the model when MoE experts borrow from it
    /// (`--experts=borrow`, see loadMoe); unmapped last in deinit. null when
    /// nothing borrows (the default packed path copies everything out).
    weight_mapping: ?gguf.File.MappedRegion = null,

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, config: Config) !Model {
        var file = try gguf.File.loadMmap(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFile(ctx, &file, config);
    }

    pub fn loadGgufFromFile(ctx: *ExecContext, file: *gguf.File, config: Config) !Model {
        try config.validate();
        const allocator = ctx.allocator;

        const swa_pattern = try readU32OrBoolArray(allocator, file, "gemma4.attention.sliding_window_pattern", config.num_layers, bool);
        defer allocator.free(swa_pattern);
        const kv_heads = try readU32OrBoolArray(allocator, file, "gemma4.attention.head_count_kv", config.num_layers, usize);
        defer allocator.free(kv_heads);
        for (kv_heads) |kvh| {
            if (kvh == 0 or config.num_attention_heads % kvh != 0) return Error.InvalidConfig;
        }

        var geom = try deriveGeometry(allocator, config.num_layers, swa_pattern, kv_heads, config.shared_kv_layers, config.head_dim_global, config.head_dim_swa);
        errdefer geom.deinit(allocator);

        var token_embedding = try LinearWeight.load(ctx, try file.get("token_embd.weight"), config.vocab_size, config.hidden_size);
        errdefer token_embedding.deinit();

        var output_norm = try weights.loadVector(ctx, try file.get("output_norm.weight"), config.hidden_size, .embed);
        errdefer output_norm.deinit();

        var output = if (file.maybeGet("output.weight")) |info|
            try LinearWeight.load(ctx, info, config.vocab_size, config.hidden_size)
        else
            try token_embedding.cloneView(ctx);
        errdefer output.deinit();

        var rope_freqs: ?fucina.Tensor(.{.rope}) = null;
        if (file.maybeGet("rope_freqs.weight")) |info|
            rope_freqs = try weights.loadVector(ctx, info, config.head_dim_global / 2, .rope);
        errdefer if (rope_freqs) |*t| t.deinit();

        var ple: ?PerLayerEmbeddings = null;
        if (config.per_layer_input_size > 0) ple = try loadPerLayerEmbeddings(ctx, file, config);
        errdefer if (ple) |*p| p.deinit();

        const layers = try allocator.alloc(Layer, config.num_layers);
        errdefer allocator.free(layers);
        try loadLayers(ctx, file, config, geom, layers);
        errdefer for (layers) |*layer| layer.deinit(allocator);

        // When experts borrow from the mapping (loadMoe), the model takes
        // ownership of it; the packed/copied default leaves nothing mapped.
        const weight_mapping = if (config.num_experts > 0 and config.borrow_experts) file.takeMapping() else null;

        return .{
            .allocator = allocator,
            .config = config,
            .geom = geom,
            .token_embedding = token_embedding,
            .output_norm = output_norm,
            .output = output,
            .rope_freqs = rope_freqs,
            .layers = layers,
            .ple = ple,
            .weight_mapping = weight_mapping,
        };
    }

    pub fn deinit(self: *Model) void {
        for (self.layers) |*layer| layer.deinit(self.allocator);
        self.allocator.free(self.layers);
        if (self.ple) |*p| p.deinit();
        if (self.rope_freqs) |*t| t.deinit();
        self.output.deinit();
        self.output_norm.deinit();
        self.token_embedding.deinit();
        self.geom.deinit(self.allocator);
        // Unmap LAST: borrowed expert blocks (freed with the layers above)
        // point into this region.
        if (self.weight_mapping) |*m| m.deinit();
        self.* = undefined;
    }

    pub fn initKvCache(self: *const Model, ctx: *ExecContext, capacity: usize) !KvCache {
        return KvCache.initPerLayer(ctx, self.geom.kv_heads, self.geom.head_dim, capacity);
    }

    pub fn forwardLastLogits(self: *const Model, ctx: *ExecContext, token_ids: []const usize) !fucina.Tensor(.{ .seq, .vocab }) {
        if (token_ids.len == 0) return Error.InvalidSequenceLength;
        var kv = try self.initKvCache(ctx, token_ids.len);
        defer kv.deinit();
        return self.forwardStep(ctx, &kv, token_ids, 0);
    }

    pub fn forwardLastLogitsProfiled(self: *const Model, ctx: *ExecContext, io: std.Io, token_ids: []const usize, profile: *ForwardProfile) !fucina.Tensor(.{ .seq, .vocab }) {
        if (token_ids.len == 0) return Error.InvalidSequenceLength;
        var kv = try self.initKvCache(ctx, token_ids.len);
        defer kv.deinit();
        return self.forwardStepProfiled(ctx, io, &kv, token_ids, 0, profile);
    }

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
    /// identical to `forwardStep` (all rows appended, `kv` advances by
    /// `token_ids.len`); final-logit softcapping applies to every row. The
    /// speculative-decoding verify entry (see qwen3.forwardStepAllLogits for
    /// the batching/numerics notes).
    pub fn forwardStepAllLogits(
        self: *const Model,
        ctx: *ExecContext,
        kv: *KvCache,
        token_ids: []const usize,
        pos0: usize,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        return self.forwardStepImpl(ctx, null, kv, token_ids, pos0, null, false);
    }

    /// Ragged multi-stream forward: one span of new tokens per stream, all
    /// spans packed through the weights (embedding, projections, norms,
    /// FFN/MoE, lm head) as ONE [total, ...] pass; attention runs per
    /// stream against its own cache (per-layer sliding windows and
    /// shared-KV refs included). All-spans-1 is the lockstep batched
    /// decode (`forwardStepBatch`). Returns [total, vocab] logits — every
    /// appended position, soft-capped — in stream order; each cache
    /// advances by its span. f16 caches only; PLE models are not routed.
    pub fn forwardStepBatchSpans(
        self: *const Model,
        ctx: *ExecContext,
        caches: []const *KvCache,
        token_ids: []const usize,
        span_lens: []const usize,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        const n = caches.len;
        if (n == 0 or span_lens.len != n) return Error.InvalidSequenceLength;
        if (self.ple != null) return Error.PleUnsupported;
        var total: usize = 0;
        for (span_lens) |span| {
            if (span == 0) return Error.InvalidSequenceLength;
            total += span;
        }
        if (total != token_ids.len) return Error.InvalidSequenceLength;
        for (caches, span_lens, 0..) |kv, span, i| {
            try requireF16KvCache(kv);
            if (kv.head_dim.len != self.layers.len) return Error.MismatchedKvCaches;
            if (kv.len + span > kv.capacity) return kv_cache.Error.KvCacheOverflow;
            for (caches[0..i]) |prev| if (prev == kv) return Error.MismatchedKvCaches;
        }

        const cfg = self.config;
        const allocator = ctx.allocator;
        const positions = try allocator.alloc(i32, total);
        defer allocator.free(positions);
        {
            var at: usize = 0;
            for (caches, span_lens) |kv, span| {
                for (0..span) |j| {
                    positions[at] = @intCast(kv.len + j);
                    at += 1;
                }
            }
        }
        const factors: ?[]const f32 = if (self.rope_freqs) |*t| try t.dataConst() else null;
        var swa_table = try ctx.prepareRopeTable(positions, cfg.head_dim_swa, cfg.rope_theta_swa, false);
        defer swa_table.deinit();
        var global_table = try ctx.prepareRopeTableFactors(positions, cfg.head_dim_global, cfg.rope_theta, false, factors);
        defer global_table.deinit();

        var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
        errdefer x.deinit();
        x = try ctx.replace(x, x.scale(ctx, @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)))));

        for (self.layers, 0..) |*layer, il| {
            x = try ctx.replace(x, attnBlockBatchSpans(ctx, cfg, self.geom, layer, il, &x, &swa_table, &global_table, caches, span_lens));
            x = try ctx.replace(x, ffnBlock(ctx, null, cfg, layer, &x, null));
            if (layer.out_scale) |sc| x = try ctx.replace(x, x.scale(ctx, sc));
        }
        for (caches, span_lens) |kv, span| kv.advance(span);

        var final_norm = try x.rmsNormMul(ctx, .embed, &self.output_norm, cfg.rms_norm_eps);
        defer final_norm.deinit();
        x.deinit();

        var logits = try self.output.linearSeq(ctx, &final_norm, .embed, .vocab);
        if (cfg.final_logit_softcapping != 0) {
            const sc = cfg.final_logit_softcapping;
            if (sc == 30.0) {
                const out = try logits.softcap30(ctx);
                logits.deinit();
                return out;
            }
            var down = try logits.scale(ctx, 1.0 / sc);
            logits.deinit();
            defer down.deinit();
            var t = try down.tanh(ctx);
            defer t.deinit();
            return t.scale(ctx, sc);
        }
        return logits;
    }

    /// Lockstep batched decode: one new token per stream — `forwardStepBatchSpans`
    /// with every span 1. Returns [streams, vocab] logits in stream order.
    pub fn forwardStepBatch(
        self: *const Model,
        ctx: *ExecContext,
        caches: []const *KvCache,
        token_ids: []const usize,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        const spans = try ctx.allocator.alloc(usize, caches.len);
        defer ctx.allocator.free(spans);
        @memset(spans, 1);
        return self.forwardStepBatchSpans(ctx, caches, token_ids, spans);
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
        try requireF16KvCache(kv);
        if (kv.len != pos0) return Error.InvalidSequenceLength;
        if (kv.len + token_ids.len > kv.capacity) return kv_cache.Error.KvCacheOverflow;

        const cfg = self.config;
        const allocator = ctx.allocator;

        const positions = try allocator.alloc(i32, token_ids.len);
        defer allocator.free(positions);
        for (positions, 0..) |*p, i| p.* = @intCast(pos0 + i);

        const factors: ?[]const f32 = if (self.rope_freqs) |*t| try t.dataConst() else null;
        var swa_table = try ctx.prepareRopeTable(positions, cfg.head_dim_swa, cfg.rope_theta_swa, false);
        defer swa_table.deinit();
        var global_table = try ctx.prepareRopeTableFactors(positions, cfg.head_dim_global, cfg.rope_theta, false, factors);
        defer global_table.deinit();

        const embed_start = profileStart(profile, io);
        var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
        errdefer x.deinit();
        x = try ctx.replace(x, x.scale(ctx, @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)))));
        if (profile) |p| p.embed_ns += profileElapsed(embed_start, io);

        var ple_inputs: ?[]fucina.Tensor(.{ .seq, .ple }) = null;
        defer if (ple_inputs) |inputs| {
            for (inputs) |*t| t.deinit();
            allocator.free(inputs);
        };
        if (self.ple) |*ple_w| ple_inputs = try buildPerLayerInputs(ctx, cfg, ple_w, &x, token_ids);

        for (self.layers, 0..) |*layer, il| {
            const last_query_only = last_only and il + 1 == self.layers.len and token_ids.len > 1;
            const attn_start = profileStart(profile, io);
            x = try ctx.replace(x, attnBlock(ctx, cfg, self.geom, layer, il, &x, &swa_table, &global_table, last_query_only, kv));
            if (profile) |p| p.attn_ns += profileElapsed(attn_start, io);

            const ffn_start = profileStart(profile, io);
            x = try ctx.replace(x, ffnBlock(ctx, io, cfg, layer, &x, profile));
            if (profile) |p| {
                p.ffn_ns += profileElapsed(ffn_start, io);
                p.layers += 1;
            }
            if (ple_inputs) |inputs| {
                if (last_query_only) {
                    var ple_last = try inputs[il].narrow(ctx, .seq, inputs[il].dim(.seq) - 1, 1);
                    defer ple_last.deinit();
                    x = try ctx.replace(x, pleInject(ctx, layer, &x, &ple_last, cfg.rms_norm_eps));
                } else {
                    x = try ctx.replace(x, pleInject(ctx, layer, &x, &inputs[il], cfg.rms_norm_eps));
                }
            }
            if (layer.out_scale) |s| x = try ctx.replace(x, x.scale(ctx, s));
        }
        kv.advance(token_ids.len);

        const final_start = profileStart(profile, io);
        var final_norm = try x.rmsNormMul(ctx, .embed, &self.output_norm, cfg.rms_norm_eps);
        defer final_norm.deinit();
        x.deinit();

        // last_only keeps just the final row for the vocab projection; the
        // all-logits entry projects every position.
        const keep_from = if (last_only) final_norm.dim(.seq) - 1 else 0;
        var head_in = try final_norm.narrow(ctx, .seq, keep_from, final_norm.dim(.seq) - keep_from);
        defer head_in.deinit();

        var logits = try self.output.linearSeq(ctx, &head_in, .embed, .vocab);
        if (cfg.final_logit_softcapping != 0) {
            const sc = cfg.final_logit_softcapping;
            if (sc == 30.0) {
                const out = try logits.softcap30(ctx);
                logits.deinit();
                if (profile) |p| p.final_ns += profileElapsed(final_start, io);
                return out;
            }
            var down = try logits.scale(ctx, 1.0 / sc);
            logits.deinit();
            defer down.deinit();
            var t = try down.tanh(ctx);
            defer t.deinit();
            const out = try t.scale(ctx, sc);
            if (profile) |p| p.final_ns += profileElapsed(final_start, io);
            return out;
        }
        if (profile) |p| p.final_ns += profileElapsed(final_start, io);
        return logits;
    }

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

// ---------------------------------------------------------------------------
// Forward blocks
// ---------------------------------------------------------------------------

/// gemma4's attention reads the `kv.k`/`kv.v` f16 tensor views directly; a
/// q8_0 cache stores blocks in `k_q8`/`v_q8` and leaves those views EMPTY, so
/// indexing them would be out of bounds. Reject non-f16 caches at the forward
/// seam (qwen3 is the only model with a q8_0 attention path).
pub fn requireF16KvCache(kv: *const KvCache) Error!void {
    switch (kv.dtype) {
        .f16 => {},
        .q8_0 => return Error.UnsupportedKvCacheDtype,
    }
}

/// The ragged-batch twin of `attnBlock` (see `forwardStepBatchSpans`):
/// norms/projections/rope run over the packed rows; K/V append and
/// attention run per stream against that stream's cache (per-layer window,
/// shared-KV ref, per-layer GQA map), and the per-stream outputs concat
/// back into the packed row order.
fn attnBlockBatchSpans(
    ctx: *ExecContext,
    config: Config,
    geom: LayerGeometry,
    layer: *const Layer,
    il: usize,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    swa_table: *const fucina.RopeTable,
    global_table: *const fucina.RopeTable,
    caches: []const *KvCache,
    span_lens: []const usize,
) !fucina.Tensor(.{ .seq, .embed }) {
    const head_dim = geom.head_dim[il];
    const n_head = config.num_attention_heads;
    const n_kv = geom.kv_heads[il];
    const q_dim = n_head * head_dim;
    const kv_dim = n_kv * head_dim;
    const window: usize = if (geom.is_swa[il]) config.sliding_window else 0;
    const table = if (geom.is_swa[il]) swa_table else global_table;

    var kvhh: [max_heads]usize = undefined;
    const heads_per_kv = n_head / n_kv;
    for (0..n_head) |h| kvhh[h] = h / heads_per_kv;
    const kv_head_for_head = kvhh[0..n_head];

    var attn_in = try input.rmsNormMul(ctx, .embed, &layer.attn_norm, config.rms_norm_eps);
    defer attn_in.deinit();
    var proj = try layer.attn_proj.project(ctx, &attn_in, q_dim, kv_dim);
    defer proj.deinit();
    var q3 = try proj.q.split(ctx, .q, .{ .head, .d }, .{ n_head, head_dim });
    defer q3.deinit();
    var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, config.rms_norm_eps, table);
    defer q_rope.deinit();

    var k_rope: ?fucina.Tensor(.{ .seq, .kv_head, .d }) = null;
    defer if (k_rope) |*t| t.deinit();
    var v_norm: ?fucina.Tensor(.{ .seq, .kv_head, .d }) = null;
    defer if (v_norm) |*t| t.deinit();
    if (geom.has_kv[il]) {
        var k3 = try proj.k.?.split(ctx, .k, .{ .kv_head, .d }, .{ n_kv, head_dim });
        defer k3.deinit();
        k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm.?, config.rms_norm_eps, table);
        var v3 = blk: {
            if (proj.v) |*v| {
                break :blk try v.split(ctx, .v, .{ .kv_head, .d }, .{ n_kv, head_dim });
            } else {
                break :blk try k3.withTags(ctx, .{ .seq, .kv_head, .d });
            }
        };
        defer v3.deinit();
        v_norm = try v3.rmsNorm(ctx, .d, config.rms_norm_eps);
    }

    const Out = fucina.Tensor(.{ .seq, .attn });
    const outs = try ctx.allocator.alloc(Out, caches.len);
    defer ctx.allocator.free(outs);
    var built: usize = 0;
    errdefer for (outs[0..built]) |*out| out.deinit();

    var start: usize = 0;
    for (caches, span_lens, 0..) |kv, span, si| {
        if (geom.has_kv[il]) {
            var k_rows = try k_rope.?.narrow(ctx, .seq, start, span);
            defer k_rows.deinit();
            var v_rows = try v_norm.?.narrow(ctx, .seq, start, span);
            defer v_rows.deinit();
            try kv.appendLayer(ctx, il, &k_rows, &v_rows);
        }
        const ref = geom.kv_ref[il];
        const cached_len = kv.len + span;
        var k_view = try kv.k[ref].narrow(ctx, .seq, 0, cached_len);
        defer k_view.deinit();
        var v_view = try kv.v[ref].narrow(ctx, .seq, 0, cached_len);
        defer v_view.deinit();
        var q_seg = try q_rope.narrow(ctx, .seq, start, span);
        defer q_seg.deinit();
        outs[si] = try q_seg.groupedAttention(
            ctx,
            &k_view,
            &v_view,
            kv_head_for_head,
            .attn,
            1.0, // Gemma 4: softmax scale = 1.0 (f_attention_scale)
            .{ .window = window },
        );
        built += 1;
        start += span;
    }

    var attn: Out = undefined;
    if (outs.len == 1) {
        attn = outs[0];
        built = 0; // ownership moved
    } else {
        const rest = try ctx.allocator.alloc(*const Out, outs.len - 1);
        defer ctx.allocator.free(rest);
        for (rest, outs[1..]) |*ptr, *out| ptr.* = out;
        attn = try outs[0].concat(ctx, .seq, rest);
        for (outs[0..built]) |*out| out.deinit();
        built = 0;
    }
    defer attn.deinit();

    var attn_out = try layer.o_proj.linearSeq(ctx, &attn, .attn, .embed);
    defer attn_out.deinit();
    return attn_out.rmsNormMulAdd(ctx, .embed, &layer.attn_post_norm, input, config.rms_norm_eps);
}

pub fn attnBlock(
    ctx: *ExecContext,
    config: Config,
    geom: LayerGeometry,
    layer: *const Layer,
    il: usize,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    swa_table: *const fucina.RopeTable,
    global_table: *const fucina.RopeTable,
    last_query_only: bool,
    kv: *KvCache,
) !fucina.Tensor(.{ .seq, .embed }) {
    const head_dim = geom.head_dim[il];
    const n_head = config.num_attention_heads;
    const n_kv = geom.kv_heads[il];
    const q_dim = n_head * head_dim;
    const kv_dim = n_kv * head_dim;
    const window: usize = if (geom.is_swa[il]) config.sliding_window else 0;
    const table = if (geom.is_swa[il]) swa_table else global_table;
    const m = input.dim(.seq);

    // Per-layer GQA head→kv-head map (n_kv varies by layer).
    var kvhh: [max_heads]usize = undefined;
    const heads_per_kv = n_head / n_kv;
    for (0..n_head) |h| kvhh[h] = h / heads_per_kv;
    const kv_head_for_head = kvhh[0..n_head];

    var attn_in = try input.rmsNormMul(ctx, .embed, &layer.attn_norm, config.rms_norm_eps);
    defer attn_in.deinit();

    var proj = try layer.attn_proj.project(ctx, &attn_in, q_dim, kv_dim);
    defer proj.deinit();
    var q3 = try proj.q.split(ctx, .q, .{ .head, .d }, .{ n_head, head_dim });
    defer q3.deinit();
    var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, config.rms_norm_eps, table);
    defer q_rope.deinit();

    if (geom.has_kv[il]) {
        var k3 = try proj.k.?.split(ctx, .k, .{ .kv_head, .d }, .{ n_kv, head_dim });
        defer k3.deinit();
        var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm.?, config.rms_norm_eps, table);
        defer k_rope.deinit();

        var v3 = blk: {
            if (proj.v) |*v| {
                break :blk try v.split(ctx, .v, .{ .kv_head, .d }, .{ n_kv, head_dim });
            } else {
                break :blk try k3.withTags(ctx, .{ .seq, .kv_head, .d });
            }
        };
        defer v3.deinit();
        var v_norm = try v3.rmsNorm(ctx, .d, config.rms_norm_eps);
        defer v_norm.deinit();

        try kv.appendLayer(ctx, il, &k_rope, &v_norm);
    }

    const ref = geom.kv_ref[il];
    const cached_len = kv.len + m;
    var k_view = try kv.k[ref].narrow(ctx, .seq, 0, cached_len);
    defer k_view.deinit();
    var v_view = try kv.v[ref].narrow(ctx, .seq, 0, cached_len);
    defer v_view.deinit();

    // Multi-token prefill on the last layer: only the final position feeds the
    // logits, so attend (and project/residual below) for that row alone. K/V
    // were already appended full-width above.
    var q_last: ?fucina.Tensor(.{ .seq, .head, .d }) = null;
    defer if (q_last) |*value| value.deinit();
    if (last_query_only) {
        q_last = try q_rope.narrow(ctx, .seq, q_rope.dim(.seq) - 1, 1);
    }
    const q_attention = if (q_last) |*value| value else &q_rope;
    var attn = try q_attention.groupedAttention(
        ctx,
        &k_view,
        &v_view,
        kv_head_for_head,
        .attn,
        1.0, // Gemma 4: softmax scale = 1.0 (f_attention_scale)
        .{ .window = window },
    );
    defer attn.deinit();

    var attn_out = try layer.o_proj.linearSeq(ctx, &attn, .attn, .embed);
    defer attn_out.deinit();
    var input_last: ?fucina.Tensor(.{ .seq, .embed }) = null;
    defer if (input_last) |*value| value.deinit();
    if (last_query_only) {
        input_last = try input.narrow(ctx, .seq, input.dim(.seq) - 1, 1);
    }
    const residual_input = if (input_last) |*value| value else input;
    return attn_out.rmsNormMulAdd(ctx, .embed, &layer.attn_post_norm, residual_input, config.rms_norm_eps);
}

pub fn ffnBlock(
    ctx: *ExecContext,
    io: ?std.Io,
    config: Config,
    layer: *const Layer,
    attn_out: *const fucina.Tensor(.{ .seq, .embed }),
    profile: ?*ForwardProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    var mlp_in = try attn_out.rmsNormMul(ctx, .embed, &layer.ffn_norm, config.rms_norm_eps);
    defer mlp_in.deinit();
    const dense_start = profileStart(profile, io);
    var mlp = try denseGeglu(ctx, &layer.ffn_gate, &layer.ffn_up, &layer.ffn_down, &mlp_in);
    defer mlp.deinit();
    if (profile) |p| p.dense_ns += profileElapsed(dense_start, io);

    var combined: fucina.Tensor(.{ .seq, .embed }) = blk: {
        if (layer.moe) |*moe| {
            var mlp_post = try mlp.rmsNormMul(ctx, .embed, &moe.post_norm_1, config.rms_norm_eps);
            defer mlp_post.deinit();

            var moe_in = try attn_out.rmsNormMul(ctx, .embed, &moe.pre_norm_2, config.rms_norm_eps);
            defer moe_in.deinit();
            var moe_out = try moeFfn(ctx, io, config, moe, attn_out, &moe_in, profile);
            defer moe_out.deinit();
            var moe_post = try moe_out.rmsNormMul(ctx, .embed, &moe.post_norm_2, config.rms_norm_eps);
            defer moe_post.deinit();

            break :blk try mlp_post.add(ctx, &moe_post);
        } else {
            break :blk try mlp.withTags(ctx, .{ .seq, .embed });
        }
    };
    defer combined.deinit();

    return combined.rmsNormMulAdd(ctx, .embed, &layer.ffn_post_norm, attn_out, config.rms_norm_eps);
}

fn denseGeglu(
    ctx: *ExecContext,
    gate_w: *const LinearWeight,
    up_w: *const LinearWeight,
    down_w: *const LinearWeight,
    input: *const fucina.Tensor(.{ .seq, .embed }),
) !fucina.Tensor(.{ .seq, .embed }) {
    var gate = try gate_w.linearSeq(ctx, input, .embed, .ffn);
    defer gate.deinit();
    var up = try up_w.linearSeq(ctx, input, .embed, .ffn);
    defer up.deinit();
    // Multi-token fused fast path: GeGLU (same f16-LUT gelu_quant semantics)
    // + LHS quantization + the packed down GEMM in one pass, skipping the two
    // m*ffn intermediates the unfused path materializes below.
    if (input.dim(.seq) >= 2) {
        switch (down_w.*) {
            .q8_0 => |*down| return gate.gegluQuantDotPacked(ctx, &up, &down.packed_rhs, .ffn, .embed),
            else => {},
        }
    }
    // GeGLU with ggml-matching f16-LUT gelu on the gate branch (× the f32 up).
    var gate_act = try gate.unary(ctx, .gelu_quant);
    defer gate_act.deinit();
    var gated = try up.mul(ctx, &gate_act);
    defer gated.deinit();
    return down_w.linearSeq(ctx, &gated, .ffn, .embed);
}

/// MoE FFN. Router input = rms_norm(attn_out) · router_weight (= ffn_gate_inp.scale
/// / sqrt(hidden)); routing = softmax over all experts → top-k → renormalize. Each
/// selected expert: GeGLU(gate(x), up(x)) → down → ×(routing_weight × down_scale).
/// `moe_in` is pre_ffw_norm_2(attn_out) (the expert input).
fn moeFfn(
    ctx: *ExecContext,
    io: ?std.Io,
    config: Config,
    moe: *const MoeFfn,
    attn_out: *const fucina.Tensor(.{ .seq, .embed }),
    moe_in: *const fucina.Tensor(.{ .seq, .embed }),
    profile: ?*ForwardProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    const allocator = ctx.allocator;
    const seq = moe_in.dim(.seq);
    const hidden = config.hidden_size;
    const n_expert = config.num_experts;
    const top_k = config.num_experts_used;

    const router_start = profileStart(profile, io);
    var router_in = try attn_out.rmsNormMul(ctx, .embed, &moe.router_weight, config.rms_norm_eps);
    defer router_in.deinit();
    var logits = try moe.router.linearSeq(ctx, &router_in, .embed, .expert);
    defer logits.deinit();

    const n_pairs = seq * top_k;
    var sel_stack: [512]usize = undefined;
    var wgt_stack: [512]f32 = undefined;
    var sel_heap: ?[]usize = null;
    defer if (sel_heap) |buf| allocator.free(buf);
    var wgt_heap: ?[]f32 = null;
    defer if (wgt_heap) |buf| allocator.free(buf);
    const sel = if (n_pairs <= sel_stack.len) sel_stack[0..n_pairs] else blk: {
        sel_heap = try allocator.alloc(usize, n_pairs);
        break :blk sel_heap.?;
    };
    const wgt = if (n_pairs <= wgt_stack.len) wgt_stack[0..n_pairs] else blk: {
        wgt_heap = try allocator.alloc(f32, n_pairs);
        break :blk wgt_heap.?;
    };
    try logits.routerTopK(ctx, .expert, top_k, .{}, sel, wgt);
    // Fold the per-expert down output scale into the routing weight (llama.cpp
    // applies down_exps_s after the down projection, before the routing weight).
    for (sel, wgt) |e, *w| w.* *= moe.down_scale[e];
    if (profile) |p| p.moe_router_ns += profileElapsed(router_start, io);

    // Raw-block expert representation: always set on -Dgpu=metal builds
    // (batch tries the grouped Metal GEMM first and falls back to the raw CPU
    // path; decode stays on the CPU, memory-bound, over the same blocks) and
    // on CPU builds with Q4_K gate_up experts (raw CPU paths only — no x4
    // packing exists for that arm).
    if (moe.gpu_weights) |gw| {
        var batch_profile: fucina.MoeBatchProfile = .{};
        const batch_profile_ptr: ?*fucina.MoeBatchProfile = if (profile != null) &batch_profile else null;
        if (seq == 1) {
            const out = try gemma_moe.decodeRawTensor(
                ctx,
                moe_in,
                gw,
                n_expert,
                sel[0..top_k],
                wgt[0..top_k],
                config.moe_intermediate_size,
                io,
                batch_profile_ptr,
            );
            if (profile) |p| {
                p.moe_count_sort_ns += batch_profile.count_sort_ns;
                p.moe_gather_ns += batch_profile.gather_quant_ns;
                p.moe_expert_wall_ns += batch_profile.expert_wall_ns;
                p.moe_task_gate_up_ns += batch_profile.gate_up_ns;
                p.moe_task_act_ns += batch_profile.swiglu_requant_ns;
                p.moe_task_down_ns += batch_profile.down_ns;
                p.moe_scatter_ns += batch_profile.scatter_ns;
            }
            return out;
        }
        const out = try gemma_moe.batchRawTensor(
            ctx,
            moe_in,
            gw,
            n_expert,
            sel,
            wgt,
            top_k,
            config.moe_intermediate_size,
            io,
            batch_profile_ptr,
        );
        if (profile) |p| {
            p.moe_count_sort_ns += batch_profile.count_sort_ns;
            p.moe_gather_ns += batch_profile.gather_quant_ns;
            p.moe_gate_up_ns += batch_profile.gate_up_ns;
            p.moe_act_ns += batch_profile.swiglu_requant_ns;
            p.moe_down_ns += batch_profile.down_ns;
            p.moe_scatter_ns += batch_profile.scatter_ns;
        }
        return out;
    }

    // Decode (seq==1): one GEMV per selected expert straight off the input row —
    // no grouping/gather overhead (which would only cost on a single token).
    if (seq == 1) {
        var batch_profile: fucina.MoeBatchProfile = .{};
        const batch_profile_ptr: ?*fucina.MoeBatchProfile = if (profile != null) &batch_profile else null;
        const out = try gemma_moe.decodePackedTensor(
            ctx,
            moe_in,
            moe.gate,
            moe.up,
            moe.down,
            sel[0..top_k],
            wgt[0..top_k],
            config.moe_intermediate_size,
            io,
            batch_profile_ptr,
        );
        if (profile) |p| {
            p.moe_count_sort_ns += batch_profile.count_sort_ns;
            p.moe_gather_ns += batch_profile.gather_quant_ns;
            p.moe_expert_wall_ns += batch_profile.expert_wall_ns;
            p.moe_task_gate_up_ns += batch_profile.gate_up_ns;
            p.moe_task_act_ns += batch_profile.swiglu_requant_ns;
            p.moe_task_down_ns += batch_profile.down_ns;
            p.moe_scatter_ns += batch_profile.scatter_ns;
        }
        return out;
    }

    if (seq * top_k >= 1) {
        var batch_profile: fucina.MoeBatchProfile = .{};
        const batch_profile_ptr: ?*fucina.MoeBatchProfile = if (profile != null) &batch_profile else null;
        const out = try gemma_moe.batchPackedTensor(
            ctx,
            moe_in,
            moe.gate,
            moe.up,
            moe.down,
            sel,
            wgt,
            top_k,
            config.moe_intermediate_size,
            io,
            batch_profile_ptr,
        );
        if (profile) |p| {
            p.moe_count_sort_ns += batch_profile.count_sort_ns;
            p.moe_gather_ns += batch_profile.gather_quant_ns;
            p.moe_gate_up_ns += batch_profile.gate_up_ns;
            p.moe_act_ns += batch_profile.swiglu_requant_ns;
            p.moe_down_ns += batch_profile.down_ns;
            p.moe_scatter_ns += batch_profile.scatter_ns;
        }
        return out;
    }

    // Prefill (seq>1): group the (token, expert) pairs by expert so each
    // expert's gate/up/down weights are read ONCE and reused across all m_e
    // tokens routed to it (one m>1 batched GEMM per expert via the packed
    // Q6_K/Q8_0 kernels) — far less weight traffic than a GEMV per token.
    const acc = try allocator.alloc(f32, seq * hidden);
    defer allocator.free(acc);
    @memset(acc, 0);

    const count = try allocator.alloc(usize, n_expert);
    defer allocator.free(count);
    @memset(count, 0);
    for (sel) |e| count[e] += 1;
    const offset = try allocator.alloc(usize, n_expert);
    defer allocator.free(offset);
    const cursor = try allocator.alloc(usize, n_expert);
    defer allocator.free(cursor);
    {
        var running: usize = 0;
        for (0..n_expert) |e| {
            offset[e] = running;
            cursor[e] = running;
            running += count[e];
        }
    }
    const order = try allocator.alloc(usize, n_pairs); // pair indices grouped by expert
    defer allocator.free(order);
    for (sel, 0..) |e, p| {
        order[cursor[e]] = p;
        cursor[e] += 1;
    }

    const moe_in_data = try moe_in.dataConst();
    const gathered = try allocator.alloc(f32, seq * hidden); // m_e <= seq (distinct experts per token)
    defer allocator.free(gathered);

    for (0..n_expert) |e| {
        const m = count[e];
        if (m == 0) continue;
        const base = offset[e];
        for (0..m) |i| {
            const token = order[base + i] / top_k;
            @memcpy(gathered[i * hidden ..][0..hidden], moe_in_data[token * hidden ..][0..hidden]);
        }
        var gx = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ m, hidden }, gathered[0 .. m * hidden]);
        defer gx.deinit();
        const gate_up_start = profileStart(profile, io);
        var gate = try gx.dotPacked(ctx, &moe.gate[e], .embed, .ffn);
        defer gate.deinit();
        var up = try gx.dotPacked(ctx, &moe.up[e], .embed, .ffn);
        defer up.deinit();
        if (profile) |p| p.moe_gate_up_ns += profileElapsed(gate_up_start, io);

        const act_start = profileStart(profile, io);
        var gate_act = try gate.unary(ctx, .gelu_quant);
        defer gate_act.deinit();
        var g = try up.mul(ctx, &gate_act);
        defer g.deinit();
        if (profile) |p| p.moe_act_ns += profileElapsed(act_start, io);

        const down_start = profileStart(profile, io);
        var d = try g.dotPacked(ctx, &moe.down[e], .ffn, .embed);
        defer d.deinit();
        if (profile) |p| p.moe_down_ns += profileElapsed(down_start, io);

        const scatter_start = profileStart(profile, io);
        const dd = try d.dataConst();
        for (0..m) |i| {
            const p = order[base + i];
            const token = p / top_k;
            const w = wgt[p];
            const src = dd[i * hidden ..][0..hidden];
            for (acc[token * hidden ..][0..hidden], src) |*a, v| a.* += w * v;
        }
        if (profile) |p| p.moe_scatter_ns += profileElapsed(scatter_start, io);
    }
    return fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ seq, hidden }, acc);
}

fn profileStart(profile: ?*ForwardProfile, io: ?std.Io) i128 {
    return if (profile != null) std.Io.Clock.awake.now(io.?).nanoseconds else 0;
}

fn profileElapsed(start: i128, io: ?std.Io) i128 {
    return std.Io.Clock.awake.now(io.?).nanoseconds - start;
}

// ---------------------------------------------------------------------------
// Per-Layer Embeddings (inactive when per_layer_input_size == 0)
// ---------------------------------------------------------------------------

fn buildPerLayerInputs(
    ctx: *ExecContext,
    config: Config,
    ple: *const PerLayerEmbeddings,
    x_scaled: *const fucina.Tensor(.{ .seq, .embed }),
    token_ids: []const usize,
) ![]fucina.Tensor(.{ .seq, .ple }) {
    const allocator = ctx.allocator;
    const n_layer = config.num_layers;
    const ple_w = config.per_layer_input_size;

    var tok_all = try ple.tok_embd.getRowsAs(ctx, token_ids, .ple_all);
    defer tok_all.deinit();
    tok_all = try ctx.replace(tok_all, tok_all.scale(ctx, @sqrt(@as(f32, @floatFromInt(ple_w)))));

    var proj_all = try ple.model_proj.linearSeq(ctx, x_scaled, .embed, .ple_all);
    defer proj_all.deinit();
    proj_all = try ctx.replace(proj_all, proj_all.scale(ctx, 1.0 / @sqrt(@as(f32, @floatFromInt(config.hidden_size)))));

    const inputs = try allocator.alloc(fucina.Tensor(.{ .seq, .ple }), n_layer);
    var built: usize = 0;
    errdefer {
        for (inputs[0..built]) |*t| t.deinit();
        allocator.free(inputs);
    }
    const inv_sqrt2: f32 = 1.0 / @sqrt(2.0);
    for (0..n_layer) |il| {
        var proj_block = try proj_all.narrow(ctx, .ple_all, il * ple_w, ple_w);
        defer proj_block.deinit();
        var proj_ple = try proj_block.withTags(ctx, .{ .seq, .ple });
        defer proj_ple.deinit();
        var proj_norm = try proj_ple.rmsNormMul(ctx, .ple, &ple.proj_norm, config.rms_norm_eps);
        defer proj_norm.deinit();

        var tok_block = try tok_all.narrow(ctx, .ple_all, il * ple_w, ple_w);
        defer tok_block.deinit();
        var tok_ple = try tok_block.withTags(ctx, .{ .seq, .ple });
        defer tok_ple.deinit();

        var summed = try proj_norm.add(ctx, &tok_ple);
        defer summed.deinit();
        inputs[il] = try summed.scale(ctx, inv_sqrt2);
        built += 1;
    }
    return inputs;
}

fn pleInject(
    ctx: *ExecContext,
    layer: *const Layer,
    pe_in: *const fucina.Tensor(.{ .seq, .embed }),
    inp: *const fucina.Tensor(.{ .seq, .ple }),
    eps: f32,
) !fucina.Tensor(.{ .seq, .embed }) {
    const ple = &layer.ple.?;
    var gate = try ple.inp_gate.linearSeq(ctx, pe_in, .embed, .ple);
    defer gate.deinit();
    var gate_act = try gate.gelu(ctx);
    defer gate_act.deinit();
    var mixed = try gate_act.mul(ctx, inp);
    defer mixed.deinit();
    var proj = try ple.proj.linearSeq(ctx, &mixed, .ple, .embed);
    defer proj.deinit();
    var proj_norm = try proj.rmsNormMul(ctx, .embed, &ple.post_norm, eps);
    defer proj_norm.deinit();
    return pe_in.add(ctx, &proj_norm);
}

// ---------------------------------------------------------------------------
// Loaders
// ---------------------------------------------------------------------------

fn loadPerLayerEmbeddings(ctx: *ExecContext, file: *const gguf.File, config: Config) !PerLayerEmbeddings {
    const ple_all = config.per_layer_input_size * config.num_layers;
    var tok_embd = try LinearWeight.load(ctx, try file.get("per_layer_token_embd.weight"), config.vocab_size, ple_all);
    errdefer tok_embd.deinit();
    var model_proj = try LinearWeight.load(ctx, try file.get("per_layer_model_proj.weight"), ple_all, config.hidden_size);
    errdefer model_proj.deinit();
    var proj_norm = try weights.loadVector(ctx, try file.get("per_layer_proj_norm.weight"), config.per_layer_input_size, .ple);
    errdefer proj_norm.deinit();
    return .{ .tok_embd = tok_embd, .model_proj = model_proj, .proj_norm = proj_norm };
}

fn blocksOf(comptime Block: type, data: []const u8) ![]const Block {
    if (data.len % @sizeOf(Block) != 0) return Error.InvalidWeightShape;
    if (@intFromPtr(data.ptr) % @alignOf(Block) != 0) return Error.InvalidWeightShape;
    const aligned: []align(@alignOf(Block)) const u8 = @alignCast(data);
    return std.mem.bytesAsSlice(Block, aligned);
}

fn packExpertQ6_K(ctx: *ExecContext, blocks: []const fucina.BlockQ6_K, out_rows: usize, in_cols: usize) !RhsQ6_K {
    var raw = try fucina.Tensor(.{ .dtype = .q6_k, .tags = .{ .out, .in } }).fromBlocks(ctx, .{ out_rows, in_cols }, blocks);
    defer raw.deinit();
    return raw.packRhs(ctx);
}

fn packExpertQ8_0(ctx: *ExecContext, blocks: []const fucina.BlockQ8_0, out_rows: usize, in_cols: usize) !RhsQ8_0 {
    var raw = try fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } }).fromBlocks(ctx, .{ out_rows, in_cols }, blocks);
    defer raw.deinit();
    return raw.packRhs(ctx);
}

fn loadMoe(ctx: *ExecContext, file: *const gguf.File, config: Config, il: usize, router_info: *const gguf.TensorInfo) !MoeFfn {
    var nb: [96]u8 = undefined;
    const allocator = ctx.allocator;
    const hidden = config.hidden_size;
    const n_expert = config.num_experts;
    const n_ff = config.moe_intermediate_size;

    var router = try LinearWeight.load(ctx, router_info, n_expert, hidden);
    errdefer router.deinit();

    var router_weight = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "ffn_gate_inp.scale")), hidden, .embed);
    errdefer router_weight.deinit();
    {
        const data = try router_weight.data();
        const s = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden)));
        for (data) |*w| w.* *= s;
    }

    var pre_norm_2 = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "pre_ffw_norm_2.weight")), hidden, .embed);
    errdefer pre_norm_2.deinit();
    var post_norm_1 = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "post_ffw_norm_1.weight")), hidden, .embed);
    errdefer post_norm_1.deinit();
    var post_norm_2 = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "post_ffw_norm_2.weight")), hidden, .embed);
    errdefer post_norm_2.deinit();

    // Fused gate_up experts (Q6_K as shipped, or Q4_K from an experts-only
    // transcode — `export-gguf --experts-dtype q4_k`): GGUF dims
    // [in=hidden, out=2*n_ff, n_expert].
    const gu_info = try file.get(try weights.layerName(&nb, il, "ffn_gate_up_exps.weight"));
    if (gu_info.n_dims != 3) return Error.UnsupportedExpertType;
    if (gu_info.dims[0] != hidden or gu_info.dims[1] != 2 * n_ff or gu_info.dims[2] != n_expert) return Error.InvalidWeightShape;
    const gu: gemma_moe.RawExpertWeights.GuBlocks = switch (gu_info.ggml_type) {
        .q6_k => .{ .q6_k = try blocksOf(fucina.BlockQ6_K, gu_info.data) },
        .q4_k => .{ .q4_k = try blocksOf(fucina.BlockQ4_K, gu_info.data) },
        else => return Error.UnsupportedExpertType,
    };
    const bpr_gu = hidden / 256; // 256-elem super-blocks per output row (Q6_K and Q4_K)

    // Down experts (Q8_0): GGUF dims [in=n_ff, out=hidden, n_expert].
    const dn_info = try file.get(try weights.layerName(&nb, il, "ffn_down_exps.weight"));
    if (dn_info.ggml_type != .q8_0 or dn_info.n_dims != 3) return Error.UnsupportedExpertType;
    if (dn_info.dims[0] != n_ff or dn_info.dims[1] != hidden or dn_info.dims[2] != n_expert) return Error.InvalidWeightShape;
    const dn_blocks = try blocksOf(fucina.BlockQ8_0, dn_info.data);
    const bpr_dn = n_ff / 32; // Q8_0 blocks per output row

    var down_scale_t = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "ffn_down_exps.scale")), n_expert, .expert);
    defer down_scale_t.deinit();
    const down_scale = try allocator.dupe(f32, try down_scale_t.dataConst());
    errdefer allocator.free(down_scale);

    // -Dgpu=metal builds keep a SINGLE expert representation: resident copies
    // of the raw blocks, read by both the grouped Metal GEMM and the raw CPU
    // paths. Widening them into x4 packs as well would double ~20 GB of
    // expert weights and thrash memory — measured as a page-fault storm that
    // made both the GPU dispatches and CPU decode several times slower.
    // Prefer device-owned storage when the GPU is up (pageable client memory
    // gets re-wired into the GPU address space on every dispatch, ~45 µs/MB).
    // If resident allocation fails, plain copies still feed the raw CPU path
    // and may still be tried by Metal with uncached wraps; correctness falls
    // back to CPU either way.
    // CPU builds keep the same raw representation when gate_up is Q4_K —
    // there is no x4 packing for that arm, the raw paths are the consumers.
    // CPU build + `--experts=borrow` + an mmap'd GGUF: point the raw paths
    // straight at the mapping (the Model keeps it alive via gguf.File.takeMapping
    // in loadGgufFromFile). No copy and no x4 widening — near-instant load and
    // ~half the resident memory, trading a little Q6_K CPU throughput (the raw
    // kernels are numerically identical to the x4 packs — see gemma_moe.zig
    // raw-vs-x4 parity tests). Default stays pack (no inference regression).
    // Keyed on the quant-GEMM capability, not `enabled`: a GPU build whose
    // quantized arms are still stubs (cuda M1) keeps the -Dgpu=none CPU
    // story — borrow arm and x4 packs included.
    const borrow = (comptime !fucina.internal.gpu.has_quant_gemm) and config.borrow_experts and file.is_mmap;
    if (!borrow) {
        // Every other path reads the full expert stacks (device/heap copy or
        // x4 pack), so kick off readahead for the cold-mapped bytes first.
        gguf.prefetch(gu_info.data);
        gguf.prefetch(dn_info.data);
    }

    var gpu_weights: ?gemma_moe.RawExpertWeights = null;
    if (comptime fucina.internal.gpu.has_quant_gemm) {
        const gu_bytes: []const u8 = switch (gu) {
            inline else => |gu_blocks| std.mem.sliceAsBytes(gu_blocks),
        };
        const dn_bytes = std.mem.sliceAsBytes(dn_blocks);
        if (fucina.internal.gpu.allocResidentBytes(gu_bytes.len)) |gu_dev| {
            if (fucina.internal.gpu.allocResidentBytes(dn_bytes.len)) |dn_dev| {
                @memcpy(gu_dev, gu_bytes);
                @memcpy(dn_dev, dn_bytes);
                gpu_weights = .{
                    .gu = switch (gu) {
                        .q6_k => .{ .q6_k = @alignCast(std.mem.bytesAsSlice(fucina.BlockQ6_K, gu_dev)) },
                        .q4_k => .{ .q4_k = @alignCast(std.mem.bytesAsSlice(fucina.BlockQ4_K, gu_dev)) },
                    },
                    .dn_blocks = @alignCast(std.mem.bytesAsSlice(fucina.BlockQ8_0, dn_dev)),
                    .device_owned = true,
                };
            }
            // a half-allocated pair stays with the shim until process exit —
            // only reachable when the second allocation fails mid-load
        }
        if (gpu_weights == null) {
            const gu_owned = switch (gu) {
                .q6_k => |gu_blocks| gemma_moe.RawExpertWeights.GuBlocks{ .q6_k = try allocator.dupe(fucina.BlockQ6_K, gu_blocks) },
                .q4_k => |gu_blocks| gemma_moe.RawExpertWeights.GuBlocks{ .q4_k = try allocator.dupe(fucina.BlockQ4_K, gu_blocks) },
            };
            errdefer switch (gu_owned) {
                inline else => |gu_blocks| allocator.free(gu_blocks),
            };
            const dn_owned = try allocator.dupe(fucina.BlockQ8_0, dn_blocks);
            gpu_weights = .{ .gu = gu_owned, .dn_blocks = dn_owned, .device_owned = false };
        }
    } else if (borrow) {
        // Zero-copy: the blocks live in the GGUF mapping the Model now owns.
        gpu_weights = .{ .gu = gu, .dn_blocks = dn_blocks, .device_owned = false, .borrowed = true };
    } else if (gu == .q4_k) {
        // No borrow: the GGUF mapping is released after load, so the raw
        // representation is an allocator-owned copy. (No x4 packing exists for
        // the Q4_K gate_up arm — the raw CPU paths are its only consumers.)
        const gu_owned = try allocator.dupe(fucina.BlockQ4_K, gu.q4_k);
        errdefer allocator.free(gu_owned);
        const dn_owned = try allocator.dupe(fucina.BlockQ8_0, dn_blocks);
        gpu_weights = .{ .gu = .{ .q4_k = gu_owned }, .dn_blocks = dn_owned, .device_owned = false };
    }
    errdefer if (gpu_weights) |gw| {
        if (!gw.device_owned and !gw.borrowed) {
            switch (gw.gu) {
                inline else => |gu_blocks| allocator.free(gu_blocks),
            }
            allocator.free(gw.dn_blocks);
        }
    };
    // x4 packs exist only for the Q6_K-experts CPU PACK path; gpu builds, Q4_K
    // experts, and the borrow path all run the raw kernels over `gpu_weights`.
    const n_pack = if (gpu_weights != null) 0 else switch (gu) {
        .q6_k => n_expert,
        .q4_k => 0,
    };
    const gate = try allocator.alloc(RhsQ6_K, n_pack);
    errdefer allocator.free(gate);
    const up = try allocator.alloc(RhsQ6_K, n_pack);
    errdefer allocator.free(up);
    const down = try allocator.alloc(RhsQ8_0, n_pack);
    errdefer allocator.free(down);

    var built: usize = 0;
    errdefer for (0..built) |e| {
        gate[e].deinit();
        up[e].deinit();
        down[e].deinit();
    };
    var current_gate: ?RhsQ6_K = null;
    var current_up: ?RhsQ6_K = null;
    var current_down: ?RhsQ8_0 = null;
    errdefer {
        if (current_gate) |*r| r.deinit();
        if (current_up) |*r| r.deinit();
        if (current_down) |*r| r.deinit();
    }

    const gu_per_expert = 2 * n_ff * bpr_gu;
    const gate_blocks_per_expert = n_ff * bpr_gu;
    const dn_per_expert = hidden * bpr_dn;
    // The per-expert pack loop stays SERIAL on purpose: loadLayers already runs
    // one task per transformer layer across the work pool (every gemma/diffusion
    // layer is MoE, so the packing is balanced ≈ n_layers/n_cores per thread),
    // and the pool runs nested parallelChunks serially anyway — an inner fan-out
    // would not add parallelism. The remaining cost is ~20 GB of copy+widen,
    // which is memory-bandwidth-bound (it saturates a few cores), so the real
    // levers are avoiding the copy entirely (`--experts=borrow`) and prefetching
    // the cold pages (gguf.prefetch above), not more threads.
    if (n_pack != 0) {
        const gu_q6 = gu.q6_k; // n_pack > 0 only for the Q6_K arm
        for (0..n_pack) |e| {
            const eg = gu_q6[e * gu_per_expert ..][0..gu_per_expert];
            const gate_blocks = eg[0..gate_blocks_per_expert];
            const up_blocks = eg[gate_blocks_per_expert..gu_per_expert];
            current_gate = try packExpertQ6_K(ctx, gate_blocks, n_ff, hidden);
            current_up = try packExpertQ6_K(ctx, up_blocks, n_ff, hidden);
            const ed = dn_blocks[e * dn_per_expert ..][0..dn_per_expert];
            current_down = try packExpertQ8_0(ctx, ed, hidden, n_ff);

            gate[e] = current_gate.?;
            up[e] = current_up.?;
            down[e] = current_down.?;
            current_gate = null;
            current_up = null;
            current_down = null;
            built += 1;
        }
    }

    return .{
        .router = router,
        .router_weight = router_weight,
        .pre_norm_2 = pre_norm_2,
        .post_norm_1 = post_norm_1,
        .post_norm_2 = post_norm_2,
        .gate = gate,
        .up = up,
        .down = down,
        .down_scale = down_scale,
        .gpu_weights = gpu_weights,
    };
}

fn loadLayer(ctx: *ExecContext, file: *const gguf.File, config: Config, geom: LayerGeometry, il: usize) !Layer {
    var nb: [96]u8 = undefined;
    const hidden = config.hidden_size;
    const head_dim = geom.head_dim[il];
    const q_dim = config.num_attention_heads * head_dim;
    const kv_dim = geom.kv_heads[il] * head_dim;

    var attn_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "attn_norm.weight")), hidden, .embed);
    errdefer attn_norm.deinit();
    var attn_post_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "post_attention_norm.weight")), hidden, .embed);
    errdefer attn_post_norm.deinit();

    // On layers with KV, q/k/v are fuse parts (consumed by fuseLinear below);
    // shared-KV layers keep q_proj as the long-lived weight, so it loads with
    // the default (device-resident on gpu builds) path.
    const q_info = try file.get(try weights.layerName(&nb, il, "attn_q.weight"));
    var q_proj = if (geom.has_kv[il])
        try LinearWeight.loadForFusion(ctx, q_info, q_dim, hidden)
    else
        try LinearWeight.load(ctx, q_info, q_dim, hidden);
    var q_proj_owned = true;
    errdefer if (q_proj_owned) q_proj.deinit();
    var q_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "attn_q_norm.weight")), head_dim, .d);
    errdefer q_norm.deinit();

    var k_proj: ?LinearWeight = null;
    var v_proj: ?LinearWeight = null;
    var k_proj_owned = false;
    var v_proj_owned = false;
    var k_norm: ?fucina.Tensor(.{.d}) = null;
    errdefer {
        if (k_norm) |*t| t.deinit();
        if (v_proj_owned) v_proj.?.deinit();
        if (k_proj_owned) k_proj.?.deinit();
    }
    if (geom.has_kv[il]) {
        k_proj = try LinearWeight.loadForFusion(ctx, try file.get(try weights.layerName(&nb, il, "attn_k.weight")), kv_dim, hidden);
        k_proj_owned = true;
        k_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "attn_k_norm.weight")), head_dim, .d);
        if (file.maybeGet(try weights.layerName(&nb, il, "attn_v.weight"))) |info| {
            v_proj = try LinearWeight.loadForFusion(ctx, info, kv_dim, hidden);
            v_proj_owned = true;
        }
    }

    var attn_proj: AttentionProjection = undefined;
    var attn_proj_loaded = false;
    errdefer if (attn_proj_loaded) attn_proj.deinit();
    if (geom.has_kv[il]) {
        if (v_proj) |*v_w| {
            var fuse_parts = [_]*LinearWeight{ &q_proj, &k_proj.?, v_w };
            if (try weights.fuseLinear(ctx, &fuse_parts)) |fused| {
                q_proj_owned = false;
                k_proj_owned = false;
                v_proj_owned = false;
                attn_proj = .{ .fused = .{ .weight = fused, .kind = .qkv } };
            } else {
                q_proj_owned = false;
                k_proj_owned = false;
                v_proj_owned = false;
                attn_proj = .{ .separate = .{
                    .q_proj = q_proj,
                    .k_proj = k_proj,
                    .v_proj = v_proj,
                } };
            }
        } else {
            var fuse_parts = [_]*LinearWeight{ &q_proj, &k_proj.? };
            if (try weights.fuseLinear(ctx, &fuse_parts)) |fused| {
                q_proj_owned = false;
                k_proj_owned = false;
                attn_proj = .{ .fused = .{ .weight = fused, .kind = .qk } };
            } else {
                q_proj_owned = false;
                k_proj_owned = false;
                attn_proj = .{ .separate = .{
                    .q_proj = q_proj,
                    .k_proj = k_proj,
                    .v_proj = null,
                } };
            }
        }
    } else {
        q_proj_owned = false;
        attn_proj = .{ .separate = .{
            .q_proj = q_proj,
            .k_proj = null,
            .v_proj = null,
        } };
    }
    attn_proj_loaded = true;

    var o_proj = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "attn_output.weight")), hidden, q_dim);
    errdefer o_proj.deinit();

    var ffn_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "ffn_norm.weight")), hidden, .embed);
    errdefer ffn_norm.deinit();
    var ffn_gate = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "ffn_gate.weight")), config.intermediate_size, hidden);
    errdefer ffn_gate.deinit();
    var ffn_up = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "ffn_up.weight")), config.intermediate_size, hidden);
    errdefer ffn_up.deinit();
    var ffn_down = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "ffn_down.weight")), hidden, config.intermediate_size);
    errdefer ffn_down.deinit();
    var ffn_post_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "post_ffw_norm.weight")), hidden, .embed);
    errdefer ffn_post_norm.deinit();

    var moe: ?MoeFfn = null;
    errdefer if (moe) |*m| m.deinit(ctx.allocator);
    if (file.maybeGet(try weights.layerName(&nb, il, "ffn_gate_inp.weight"))) |router_info| {
        moe = try loadMoe(ctx, file, config, il, router_info);
    }

    var ple: ?PerLayerInject = null;
    errdefer if (ple) |*p| p.deinit();
    if (config.per_layer_input_size > 0) {
        var inp_gate = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "inp_gate.weight")), config.per_layer_input_size, hidden);
        errdefer inp_gate.deinit();
        var proj = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "proj.weight")), hidden, config.per_layer_input_size);
        errdefer proj.deinit();
        var post_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "post_norm.weight")), hidden, .embed);
        errdefer post_norm.deinit();
        ple = .{ .inp_gate = inp_gate, .proj = proj, .post_norm = post_norm };
    }

    var out_scale: ?f32 = null;
    if (file.maybeGet(try weights.layerName(&nb, il, "layer_output_scale.weight"))) |info|
        out_scale = try readScalar(ctx, info);

    return .{
        .attn_norm = attn_norm,
        .attn_post_norm = attn_post_norm,
        .attn_proj = attn_proj,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .o_proj = o_proj,
        .ffn_norm = ffn_norm,
        .ffn_gate = ffn_gate,
        .ffn_up = ffn_up,
        .ffn_down = ffn_down,
        .ffn_post_norm = ffn_post_norm,
        .moe = moe,
        .ple = ple,
        .out_scale = out_scale,
    };
}

fn readScalar(ctx: *ExecContext, info: *const gguf.TensorInfo) !f32 {
    var t = try weights.loadVector(ctx, info, 1, .scalar);
    defer t.deinit();
    return t.item();
}

/// Per-family adapter for `gguf_meta.parallelLoadLayers` (gemma adds the
/// derived per-layer geometry and an allocator-taking Layer.deinit).
const LayerLoader = struct {
    ctx: *ExecContext,
    file: *const gguf.File,
    config: Config,
    geom: LayerGeometry,

    pub fn load(self: LayerLoader, il: usize) !Layer {
        return loadLayer(self.ctx, self.file, self.config, self.geom, il);
    }

    pub fn deinitLayer(self: LayerLoader, layer: *Layer) void {
        layer.deinit(self.ctx.allocator);
    }
};

pub fn loadLayers(ctx: *ExecContext, file: *const gguf.File, config: Config, geom: LayerGeometry, layers: []Layer) !void {
    return gguf_meta.parallelLoadLayers(Layer, LayerLoader, ctx, .{ .ctx = ctx, .file = file, .config = config, .geom = geom }, layers);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test {
    _ = @import("gemma4_tests.zig");
}
