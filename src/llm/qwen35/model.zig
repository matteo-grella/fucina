//! Qwen3.5 (`qwen35` GGUF arch) — a **hybrid Gated-DeltaNet (linear attention) +
//! transformer**, ported from llama.cpp (`refs/llama.cpp/src/models/qwen35.cpp`,
//! `delta-net-base.cpp`). Sibling of `qwen3next`; NOT a Qwen3 variant.
//!
//! Each decoder block is one of two kinds, selected per index by
//! `Config.isRecurrent(il)` (every `full_attention_interval`-th block is full
//! attention; the rest are linear DeltaNet):
//!   - **full attention**: GQA with a fused Q+gate projection, per-head q/k
//!     RMSNorm, multi-section (partial) RoPE, sigmoid output gate.
//!   - **linear (DeltaNet)**: a causal depthwise conv1d feeding a gated
//!     delta-rule recurrent scan over per-v-head state matrices, with a gated
//!     RMSNorm and L2-normed q/k.
//! Both feed a SiLU dense FFN. The dense Qwen3.5 text path supports GGUF
//! loading, prefill, decode, KV/recurrent cache, and an opt-in forward profiler;
//! `qwen35moe` and MTP/NextN variants are still rejected at load time.

const std = @import("std");
const fucina = @import("fucina");
const weights = @import("../weights.zig");
const kv_cache = @import("../kv_cache.zig");
const gguf_meta = @import("../gguf_meta.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;
const LinearWeight = weights.LinearWeight;
const WeightF32 = weights.WeightF32;
const WeightF16 = weights.WeightF16;
const WeightQ4_K = weights.WeightQ4_K;
const WeightQ5_K = weights.WeightQ5_K;
const WeightQ6_K = weights.WeightQ6_K;
const WeightQ8_0 = weights.WeightQ8_0;
const KvCache = kv_cache.KvCache;

pub const Error = weights.Error || error{
    InvalidConfig,
    InvalidSequenceLength,
    UnsupportedVariant,
    UnsupportedKvCacheDtype,
};

pub const ForwardProfile = struct {
    tokens: usize = 0,
    total_ns: i128 = 0,
    prep_ns: i128 = 0,
    embed_ns: i128 = 0,
    attn_layers: usize = 0,
    attn_total_ns: i128 = 0,
    attn_proj_ns: i128 = 0,
    attn_rope_ns: i128 = 0,
    attn_sdpa_ns: i128 = 0,
    attn_out_ns: i128 = 0,
    linear_layers: usize = 0,
    linear_total_ns: i128 = 0,
    linear_norm_ns: i128 = 0,
    linear_qkv_ns: i128 = 0,
    linear_conv_ns: i128 = 0,
    linear_gate_ns: i128 = 0,
    linear_z_ns: i128 = 0,
    linear_alpha_ns: i128 = 0,
    linear_beta_ns: i128 = 0,
    linear_scan_ns: i128 = 0,
    linear_out_ns: i128 = 0,
    ffn_layers: usize = 0,
    ffn_ns: i128 = 0,
    ffn_norm_ns: i128 = 0,
    ffn_gate_up_ns: i128 = 0,
    ffn_gate_ns: i128 = 0,
    ffn_up_ns: i128 = 0,
    ffn_act_ns: i128 = 0,
    ffn_down_ns: i128 = 0,
    ffn_residual_ns: i128 = 0,
    final_ns: i128 = 0,
};

pub const Config = struct {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    // --- full-attention blocks ---
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    /// Number of head dims that get rotated (`rope.dimension_count`); partial
    /// RoPE when `< head_dim` (Qwen3.5-0.8B: 64 of 256).
    rope_n_rot: usize,
    /// Multi-section (IMROPE) split, `rope.dimension_sections` (e.g. 11/11/10/0).
    rope_sections: [4]i32,
    /// A block `il` is full-attention iff `(il + 1) % full_attention_interval == 0`.
    full_attention_interval: usize,
    // --- linear DeltaNet blocks (SSM hyperparameters) ---
    ssm_d_conv: usize,
    ssm_d_inner: usize,
    ssm_d_state: usize,
    ssm_dt_rank: usize,
    ssm_n_group: usize,
    // --- optional MTP/NextN trailer (0 = none) ---
    nextn_predict_layers: usize = 0,
    // --- MoE (qwen35moe; 0 = dense) ---
    num_experts: usize = 0,

    /// llama.cpp `hparams.is_recurrent(il)`: linear (DeltaNet) layers are all
    /// trunk blocks except every `full_attention_interval`-th; MTP blocks (the
    /// trailing `nextn_predict_layers`) are dense-attention, never recurrent.
    pub fn isRecurrent(self: Config, il: usize) bool {
        const n_main = self.num_layers - self.nextn_predict_layers;
        return il < n_main and ((il + 1) % self.full_attention_interval != 0);
    }

    pub fn isMoe(self: Config) bool {
        return self.num_experts > 0;
    }

    // --- DeltaNet derived dims (see qwen35.cpp load_block_trunk) ---
    fn headKDim(self: Config) usize {
        return self.ssm_d_state;
    }
    fn numKHeads(self: Config) usize {
        return self.ssm_n_group;
    }
    fn numVHeads(self: Config) usize {
        return self.ssm_dt_rank;
    }
    fn headVDim(self: Config) usize {
        return self.ssm_d_inner / self.ssm_dt_rank;
    }
    fn keyDim(self: Config) usize {
        return self.headKDim() * self.numKHeads();
    }
    fn valueDim(self: Config) usize {
        return self.headVDim() * self.numVHeads();
    }
    /// Channels of the fused causal conv1d (`key_dim*2 + value_dim`), == the
    /// `attn_qkv` output width.
    fn convDim(self: Config) usize {
        return self.keyDim() * 2 + self.valueDim();
    }
    /// Full-attention projection widths.
    fn qGateDim(self: Config) usize {
        // wq emits query + gate interleaved per head → head_dim * n_head * 2.
        return self.head_dim * self.num_attention_heads * 2;
    }
    fn kvDim(self: Config) usize {
        return self.head_dim * self.num_key_value_heads;
    }
    fn attnOutInDim(self: Config) usize {
        return self.head_dim * self.num_attention_heads;
    }

    pub fn fromGguf(file: *const gguf.File) !Config {
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        const embd = try file.get("token_embd.weight");
        const shape = try embd.logicalMatrixShape(); // {vocab, hidden}

        return .{
            .vocab_size = shape[0],
            .hidden_size = try metaInt(file, arch, "embedding_length"),
            .intermediate_size = try metaInt(file, arch, "feed_forward_length"),
            .num_layers = try metaInt(file, arch, "block_count"),
            .num_attention_heads = try metaInt(file, arch, "attention.head_count"),
            .num_key_value_heads = try metaInt(file, arch, "attention.head_count_kv"),
            .head_dim = try metaInt(file, arch, "attention.key_length"),
            .rms_norm_eps = try metaFloat(file, arch, "attention.layer_norm_rms_epsilon"),
            .rope_theta = try metaFloat(file, arch, "rope.freq_base"),
            .rope_n_rot = metaIntOpt(file, arch, "rope.dimension_count") orelse try metaInt(file, arch, "attention.key_length"),
            .rope_sections = readSections(file, arch),
            .full_attention_interval = metaIntOpt(file, arch, "full_attention_interval") orelse 4,
            .ssm_d_conv = try metaInt(file, arch, "ssm.conv_kernel"),
            .ssm_d_inner = try metaInt(file, arch, "ssm.inner_size"),
            .ssm_d_state = try metaInt(file, arch, "ssm.state_size"),
            .ssm_dt_rank = try metaInt(file, arch, "ssm.time_step_rank"),
            .ssm_n_group = try metaInt(file, arch, "ssm.group_count"),
            .nextn_predict_layers = metaIntOpt(file, arch, "nextn_predict_layers") orelse 0,
            .num_experts = metaIntOpt(file, arch, "expert_count") orelse 0,
        };
    }

    fn validate(self: Config) !void {
        if (self.isMoe()) return Error.UnsupportedVariant; // qwen35moe: later phase
        if (self.nextn_predict_layers != 0) return Error.UnsupportedVariant; // MTP: later phase
        if (self.num_attention_heads == 0 or self.num_key_value_heads == 0) return Error.InvalidConfig;
        if (self.num_attention_heads % self.num_key_value_heads != 0) return Error.InvalidConfig;
        if (self.full_attention_interval == 0) return Error.InvalidConfig;
        if (self.ssm_dt_rank == 0 or self.ssm_d_inner % self.ssm_dt_rank != 0) return Error.InvalidConfig;
        // This loader assumes uniform DeltaNet heads (no GQA-repeat in the scan).
        if (self.numKHeads() != self.numVHeads()) return Error.UnsupportedVariant;
    }
};

pub const LinearScanMode = enum {
    /// Exact batched chunked-GEMM DeltaNet prefill path.
    chunked,
    /// Exact token-by-token recurrent DeltaNet scan, forced even for prefill.
    recurrent,
};

// Required config ints are structurally positive (`.reject_zero`), but
// optional keys legitimately carry 0 (e.g. `nextn_predict_layers`).
fn metaInt(file: *const gguf.File, arch: []const u8, suffix: []const u8) !usize {
    return gguf_meta.metaInt(file, arch, suffix, .reject_zero);
}

const metaFloat = gguf_meta.metaFloat;

fn metaIntOpt(file: *const gguf.File, arch: []const u8, suffix: []const u8) ?usize {
    return gguf_meta.metaIntOpt(file, arch, suffix, .accept_zero);
}

/// Read the (up to) 4-entry `rope.dimension_sections` int array; missing → zeros.
fn readSections(file: *const gguf.File, arch: []const u8) [4]i32 {
    var out = [4]i32{ 0, 0, 0, 0 };
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.rope.dimension_sections", .{arch}) catch return out;
    const arr = file.getArray(key) orelse return out;
    // int32 (5) or uint32 (4); 4 bytes per element, little-endian.
    if (arr.item_type != 5 and arr.item_type != 4) return out;
    const n = @min(arr.len, 4);
    var i: usize = 0;
    while (i < n) : (i += 1) out[i] = std.mem.readInt(i32, arr.data[i * 4 ..][0..4], .little);
    return out;
}

const SeparateFfnInputProjection = struct {
    gate: LinearWeight,
    up: LinearWeight,

    fn deinit(self: *SeparateFfnInputProjection) void {
        self.up.deinit();
        self.gate.deinit();
        self.* = undefined;
    }
};

const FfnInputProjection = union(enum) {
    separate: SeparateFfnInputProjection,
    fused: LinearWeight,

    fn load(ctx: *ExecContext, file: *const gguf.File, cfg: Config, il: usize) !FfnInputProjection {
        var nb: [96]u8 = undefined;
        var gate = try LinearWeight.loadForFusion(ctx, try file.get(try weights.layerName(&nb, il, "ffn_gate.weight")), cfg.intermediate_size, cfg.hidden_size);
        errdefer gate.deinit();
        var up = try LinearWeight.loadForFusion(ctx, try file.get(try weights.layerName(&nb, il, "ffn_up.weight")), cfg.intermediate_size, cfg.hidden_size);
        errdefer up.deinit();

        var fuse_parts = [_]*LinearWeight{ &gate, &up };
        if (try weights.fuseLinear(ctx, &fuse_parts)) |fused| return .{ .fused = fused };

        return .{ .separate = .{ .gate = gate, .up = up } };
    }

    fn deinit(self: *FfnInputProjection) void {
        switch (self.*) {
            .separate => |*separate| separate.deinit(),
            .fused => |*weight| weight.deinit(),
        }
        self.* = undefined;
    }
};

/// SiLU gate/up/down dense FFN. Gate+up weights are fused when they share a
/// dtype/layout, so prefill can issue one wider GEMM and run split-SwiGLU.
const DenseFfn = struct {
    input_proj: FfnInputProjection,
    down: LinearWeight,

    fn load(ctx: *ExecContext, file: *const gguf.File, cfg: Config, il: usize) !DenseFfn {
        var nb: [96]u8 = undefined;
        var input_proj = try FfnInputProjection.load(ctx, file, cfg, il);
        errdefer input_proj.deinit();
        var down = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "ffn_down.weight")), cfg.hidden_size, cfg.intermediate_size);
        errdefer down.deinit();
        return .{ .input_proj = input_proj, .down = down };
    }

    fn deinit(self: *DenseFfn) void {
        self.down.deinit();
        self.input_proj.deinit();
        self.* = undefined;
    }
};

const SeparateLinearGateProjection = struct {
    z: LinearWeight,
    alpha: LinearWeight,
    beta: LinearWeight,

    fn deinit(self: *SeparateLinearGateProjection) void {
        self.beta.deinit();
        self.alpha.deinit();
        self.z.deinit();
        self.* = undefined;
    }
};

const SeparateLinearGateData = struct {
    z: fucina.Tensor(.{ .seq, .vd }),
    alpha: fucina.Tensor(.{ .seq, .vhead }),
    beta: fucina.Tensor(.{ .seq, .vhead }),

    fn deinit(self: *SeparateLinearGateData) void {
        self.beta.deinit();
        self.alpha.deinit();
        self.z.deinit();
        self.* = undefined;
    }
};

const LinearGateDataStorage = union(enum) {
    separate: SeparateLinearGateData,
    fused: fucina.Tensor(.{ .seq, .linear_gate }),

    fn deinit(self: *LinearGateDataStorage) void {
        switch (self.*) {
            .separate => |*separate| separate.deinit(),
            .fused => |*fused| fused.deinit(),
        }
        self.* = undefined;
    }
};

const LinearGateData = struct {
    z: []const f32,
    z_stride: usize,
    alpha: []const f32,
    alpha_stride: usize,
    beta: []const f32,
    beta_stride: usize,
    storage: LinearGateDataStorage,

    fn deinit(self: *LinearGateData) void {
        self.storage.deinit();
        self.* = undefined;
    }
};

const LinearGateProjection = union(enum) {
    separate: SeparateLinearGateProjection,
    fused: LinearWeight,

    fn load(ctx: *ExecContext, file: *const gguf.File, cfg: Config, il: usize) !LinearGateProjection {
        var nb: [96]u8 = undefined;
        var z = try LinearWeight.loadForFusion(ctx, try file.get(try weights.layerName(&nb, il, "attn_gate.weight")), cfg.valueDim(), cfg.hidden_size);
        errdefer z.deinit();
        var alpha = try LinearWeight.loadForFusion(ctx, try file.get(try weights.layerName(&nb, il, "ssm_alpha.weight")), cfg.numVHeads(), cfg.hidden_size);
        errdefer alpha.deinit();
        var beta = try LinearWeight.loadForFusion(ctx, try file.get(try weights.layerName(&nb, il, "ssm_beta.weight")), cfg.numVHeads(), cfg.hidden_size);
        errdefer beta.deinit();

        var fuse_parts = [_]*LinearWeight{ &z, &alpha, &beta };
        if (try weights.fuseLinear(ctx, &fuse_parts)) |fused| return .{ .fused = fused };

        return .{ .separate = .{ .z = z, .alpha = alpha, .beta = beta } };
    }

    fn deinit(self: *LinearGateProjection) void {
        switch (self.*) {
            .separate => |*separate| separate.deinit(),
            .fused => |*weight| weight.deinit(),
        }
        self.* = undefined;
    }

    fn project(
        self: *const LinearGateProjection,
        ctx: *ExecContext,
        cfg: Config,
        attn_in: *const fucina.Tensor(.{ .seq, .embed }),
        io: ?std.Io,
        profile: ?*ForwardProfile,
    ) !LinearGateData {
        return switch (self.*) {
            .separate => |*separate| blk: {
                const z_start = profileStart(profile, io);
                var z = try separate.z.linearSeq(ctx, attn_in, .embed, .vd);
                errdefer z.deinit();
                const zd = try z.dataConst();
                profileAdd(profile, io, z_start, .linear_z);

                const alpha_start = profileStart(profile, io);
                var alpha = try separate.alpha.linearSeq(ctx, attn_in, .embed, .vhead);
                errdefer alpha.deinit();
                const ad = try alpha.dataConst();
                profileAdd(profile, io, alpha_start, .linear_alpha);

                const beta_start = profileStart(profile, io);
                var beta = try separate.beta.linearSeq(ctx, attn_in, .embed, .vhead);
                errdefer beta.deinit();
                const bd = try beta.dataConst();
                profileAdd(profile, io, beta_start, .linear_beta);

                break :blk .{
                    .z = zd,
                    .z_stride = cfg.valueDim(),
                    .alpha = ad,
                    .alpha_stride = cfg.numVHeads(),
                    .beta = bd,
                    .beta_stride = cfg.numVHeads(),
                    .storage = .{ .separate = .{ .z = z, .alpha = alpha, .beta = beta } },
                };
            },
            .fused => |*weight| try projectFusedLinearGate(weight, ctx, cfg, attn_in),
        };
    }
};

fn projectFusedLinearGate(
    weight: *const LinearWeight,
    ctx: *ExecContext,
    cfg: Config,
    attn_in: *const fucina.Tensor(.{ .seq, .embed }),
) !LinearGateData {
    var gate = try weight.linearSeq(ctx, attn_in, .embed, .linear_gate);
    errdefer gate.deinit();

    const gd = try gate.dataConst();
    const vd = cfg.valueDim();
    const H = cfg.numVHeads();
    const stride = vd + 2 * H;

    return .{
        .z = gd,
        .z_stride = stride,
        .alpha = gd[vd..],
        .alpha_stride = stride,
        .beta = gd[vd + H ..],
        .beta_stride = stride,
        .storage = .{ .fused = gate },
    };
}

/// A full-attention block: fused Q+gate, separate K/V, per-head q/k RMSNorm.
const AttnLayer = struct {
    attn_norm: fucina.Tensor(.{.embed}),
    post_norm: fucina.Tensor(.{.embed}),
    q_gate_proj: LinearWeight, // wq: hidden -> head_dim*n_head*2 (query interleaved with gate)
    k_proj: LinearWeight,
    v_proj: LinearWeight,
    q_norm: fucina.Tensor(.{.d}),
    k_norm: fucina.Tensor(.{.d}),
    o_proj: LinearWeight,
    ffn: DenseFfn,

    fn load(ctx: *ExecContext, file: *const gguf.File, cfg: Config, il: usize) !AttnLayer {
        var nb: [96]u8 = undefined;
        var attn_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "attn_norm.weight")), cfg.hidden_size, .embed);
        errdefer attn_norm.deinit();
        var post_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "post_attention_norm.weight")), cfg.hidden_size, .embed);
        errdefer post_norm.deinit();
        var q_gate = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "attn_q.weight")), cfg.qGateDim(), cfg.hidden_size);
        errdefer q_gate.deinit();
        var k_proj = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "attn_k.weight")), cfg.kvDim(), cfg.hidden_size);
        errdefer k_proj.deinit();
        var v_proj = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "attn_v.weight")), cfg.kvDim(), cfg.hidden_size);
        errdefer v_proj.deinit();
        var q_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "attn_q_norm.weight")), cfg.head_dim, .d);
        errdefer q_norm.deinit();
        var k_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "attn_k_norm.weight")), cfg.head_dim, .d);
        errdefer k_norm.deinit();
        var o_proj = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "attn_output.weight")), cfg.hidden_size, cfg.attnOutInDim());
        errdefer o_proj.deinit();
        var ffn = try DenseFfn.load(ctx, file, cfg, il);
        errdefer ffn.deinit();
        return .{
            .attn_norm = attn_norm,
            .post_norm = post_norm,
            .q_gate_proj = q_gate,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .o_proj = o_proj,
            .ffn = ffn,
        };
    }

    fn deinit(self: *AttnLayer) void {
        self.ffn.deinit();
        self.o_proj.deinit();
        self.k_norm.deinit();
        self.q_norm.deinit();
        self.v_proj.deinit();
        self.k_proj.deinit();
        self.q_gate_proj.deinit();
        self.post_norm.deinit();
        self.attn_norm.deinit();
        self.* = undefined;
    }
};

/// Per-family adapter for `gguf_meta.parallelLoadLayers`.
const LayerLoader = struct {
    ctx: *ExecContext,
    file: *const gguf.File,
    config: Config,

    pub fn load(self: LayerLoader, layer_i: usize) !Layer {
        return Layer.load(self.ctx, self.file, self.config, layer_i);
    }

    pub fn deinitLayer(_: LayerLoader, layer: *Layer) void {
        layer.deinit();
    }
};

/// Load all transformer layers, in parallel across the work pool when
/// available (see `gguf_meta.parallelLoadLayers` for the failure semantics).
fn loadLayers(ctx: *ExecContext, file: *const gguf.File, config: Config, layers: []Layer) !void {
    return gguf_meta.parallelLoadLayers(Layer, LayerLoader, ctx, .{ .ctx = ctx, .file = file, .config = config }, layers);
}

/// A linear (Gated DeltaNet) block.
const LinearLayer = struct {
    attn_norm: fucina.Tensor(.{.embed}),
    post_norm: fucina.Tensor(.{.embed}),
    qkv_proj: LinearWeight, // hidden -> key_dim*2 + value_dim (fused q,k,v)
    gate_proj: LinearGateProjection, // hidden -> value_dim + 2*num_v_heads (z, alpha, beta)
    conv1d: LinearWeight, // [conv_dim, d_conv] f32 causal depthwise kernel
    ssm_a: fucina.Tensor(.{.vhead}), // [num_v_heads] (-A_log)
    ssm_dt: fucina.Tensor(.{.vhead}), // [num_v_heads] dt bias
    ssm_norm: fucina.Tensor(.{.d}), // [head_v_dim] gated RMSNorm weight
    out_proj: LinearWeight, // value_dim -> hidden
    ffn: DenseFfn,

    fn load(ctx: *ExecContext, file: *const gguf.File, cfg: Config, il: usize) !LinearLayer {
        var nb: [96]u8 = undefined;
        var attn_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "attn_norm.weight")), cfg.hidden_size, .embed);
        errdefer attn_norm.deinit();
        var post_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "post_attention_norm.weight")), cfg.hidden_size, .embed);
        errdefer post_norm.deinit();
        var qkv = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "attn_qkv.weight")), cfg.convDim(), cfg.hidden_size);
        errdefer qkv.deinit();
        var gate = try LinearGateProjection.load(ctx, file, cfg, il);
        errdefer gate.deinit();
        // conv1d: GGUF shape ne=[d_conv, conv_dim] → load as [rows=conv_dim, cols=d_conv].
        var conv1d = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "ssm_conv1d.weight")), cfg.convDim(), cfg.ssm_d_conv);
        errdefer conv1d.deinit();
        var ssm_a = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "ssm_a")), cfg.numVHeads(), .vhead);
        errdefer ssm_a.deinit();
        var ssm_dt = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "ssm_dt.bias")), cfg.numVHeads(), .vhead);
        errdefer ssm_dt.deinit();
        var ssm_norm = try weights.loadVector(ctx, try file.get(try weights.layerName(&nb, il, "ssm_norm.weight")), cfg.headVDim(), .d);
        errdefer ssm_norm.deinit();
        var out = try LinearWeight.load(ctx, try file.get(try weights.layerName(&nb, il, "ssm_out.weight")), cfg.hidden_size, cfg.valueDim());
        errdefer out.deinit();
        var ffn = try DenseFfn.load(ctx, file, cfg, il);
        errdefer ffn.deinit();
        return .{
            .attn_norm = attn_norm,
            .post_norm = post_norm,
            .qkv_proj = qkv,
            .gate_proj = gate,
            .conv1d = conv1d,
            .ssm_a = ssm_a,
            .ssm_dt = ssm_dt,
            .ssm_norm = ssm_norm,
            .out_proj = out,
            .ffn = ffn,
        };
    }

    fn deinit(self: *LinearLayer) void {
        self.ffn.deinit();
        self.out_proj.deinit();
        self.ssm_norm.deinit();
        self.ssm_dt.deinit();
        self.ssm_a.deinit();
        self.conv1d.deinit();
        self.gate_proj.deinit();
        self.qkv_proj.deinit();
        self.post_norm.deinit();
        self.attn_norm.deinit();
        self.* = undefined;
    }
};

const Layer = union(enum) {
    attn: AttnLayer,
    linear: LinearLayer,

    fn load(ctx: *ExecContext, file: *const gguf.File, cfg: Config, il: usize) !Layer {
        if (cfg.isRecurrent(il)) return .{ .linear = try LinearLayer.load(ctx, file, cfg, il) };
        return .{ .attn = try AttnLayer.load(ctx, file, cfg, il) };
    }

    fn deinit(self: *Layer) void {
        switch (self.*) {
            .attn => |*a| a.deinit(),
            .linear => |*l| l.deinit(),
        }
        self.* = undefined;
    }
};

pub const Model = struct {
    allocator: Allocator,
    config: Config,
    token_embedding: LinearWeight,
    output_norm: fucina.Tensor(.{.embed}),
    output: LinearWeight,
    layers: []Layer,
    /// head index → its KV-head index (GQA grouping) for the attention layers.
    kv_head_for_head: []usize,

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, config: Config) !Model {
        var file = try gguf.File.load(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFile(ctx, &file, config);
    }

    pub fn loadGgufFromFile(ctx: *ExecContext, file: *const gguf.File, config: Config) !Model {
        try config.validate();
        const allocator = ctx.allocator;

        var token_embedding = try LinearWeight.load(ctx, try file.get("token_embd.weight"), config.vocab_size, config.hidden_size);
        errdefer token_embedding.deinit();

        var output_norm = try weights.loadVector(ctx, try file.get("output_norm.weight"), config.hidden_size, .embed);
        errdefer output_norm.deinit();

        var output = if (file.maybeGet("output.weight")) |info|
            try LinearWeight.load(ctx, info, config.vocab_size, config.hidden_size)
        else
            try token_embedding.cloneView(ctx); // tied embeddings
        errdefer output.deinit();

        const kv_head_for_head = try allocator.alloc(usize, config.num_attention_heads);
        errdefer allocator.free(kv_head_for_head);
        const heads_per_kv = config.num_attention_heads / config.num_key_value_heads;
        for (kv_head_for_head, 0..) |*kv_head, head_i| kv_head.* = head_i / heads_per_kv;

        const layers = try allocator.alloc(Layer, config.num_layers);
        errdefer allocator.free(layers);
        try loadLayers(ctx, file, config, layers);

        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = token_embedding,
            .output_norm = output_norm,
            .output = output,
            .layers = layers,
            .kv_head_for_head = kv_head_for_head,
        };
    }

    pub fn deinit(self: *Model) void {
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.allocator.free(self.kv_head_for_head);
        self.output.deinit();
        self.output_norm.deinit();
        self.token_embedding.deinit();
        self.* = undefined;
    }

    /// Count blocks by kind (for the loader smoke check / `--info`).
    pub fn blockCounts(self: *const Model) struct { attn: usize, linear: usize } {
        var a: usize = 0;
        var l: usize = 0;
        for (self.layers) |*layer| switch (layer.*) {
            .attn => a += 1,
            .linear => l += 1,
        };
        return .{ .attn = a, .linear = l };
    }

    /// Whole-sequence forward (no KV cache), returning the **last** token's
    /// logits. Runs the full hybrid stack — full-attention blocks and DeltaNet
    /// linear blocks (per-token recurrent scan) — and is logit-parity-validated
    /// (argmax-aligned, mean|Δ|≈0.04 vs Q8_0 rounding) against llama.cpp on
    /// Qwen3.5-0.8B. KV-cache decode + the chunked-prefill perf path come later.
    pub fn forwardLastLogits(self: *const Model, ctx: *ExecContext, token_ids: []const usize) !fucina.Tensor(.{ .seq, .vocab }) {
        if (token_ids.len == 0) return Error.InvalidSequenceLength;
        const cfg = self.config;

        const positions = try ctx.allocator.alloc(i32, token_ids.len);
        defer ctx.allocator.free(positions);
        for (positions, 0..) |*p, i| p.* = @intCast(i);

        // For text input, multi-section / IMROPE reduces to standard NEOX RoPE
        // (all position components equal), so a plain table over the rotary dims
        // (`rope_n_rot`) suffices.
        var rope_table = try ctx.prepareRopeTable(positions, cfg.rope_n_rot, cfg.rope_theta, false);
        defer rope_table.deinit();

        var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
        errdefer x.deinit();

        for (self.layers, 0..) |*layer, il| {
            // Mixing sublayer: full attention or DeltaNet linear attention.
            switch (layer.*) {
                .attn => |*a| x = try ctx.replace(x, attnForward(ctx, cfg, a, &x, &rope_table, self.kv_head_for_head, null, 0, null, null)),
                .linear => |*l| x = try ctx.replace(x, linearForward(ctx, cfg, l, &x, null, null, .chunked, null, null)),
            }
            // Multi-token input on the last layer: only the final position
            // feeds the logits, so its FFN runs on one row.
            if (il + 1 == self.layers.len and token_ids.len > 1) {
                x = try ctx.replace(x, x.narrow(ctx, .seq, x.dim(.seq) - 1, 1));
            }
            // FFN sublayer (post-attention-norm → dense SiLU → residual).
            const post_norm, const ffn = switch (layer.*) {
                .attn => |*a| .{ &a.post_norm, &a.ffn },
                .linear => |*l| .{ &l.post_norm, &l.ffn },
            };
            x = try ctx.replace(x, ffnForward(ctx, cfg, post_norm, ffn, &x, null, null));
        }

        var final_norm = try x.rmsNormMul(ctx, .embed, &self.output_norm, cfg.rms_norm_eps);
        defer final_norm.deinit();
        x.deinit();

        var last = try final_norm.narrow(ctx, .seq, final_norm.dim(.seq) - 1, 1);
        defer last.deinit();
        return self.output.linearSeq(ctx, &last, .embed, .vocab);
    }

    /// Initialize the streaming cache for this model (attention KV cache +
    /// per-linear-layer recurrent state). Capacity is the max sequence length.
    pub fn initCache(self: *const Model, ctx: *ExecContext, capacity: usize) !Cache {
        return Cache.init(self, ctx, capacity);
    }

    /// Process `token_ids` at absolute positions `pos0 .. pos0+len`, updating the
    /// caches (attention K/V + DeltaNet conv/SSM state), and return the last
    /// token's logits. Prefill = one call with `pos0 == 0`; each subsequent
    /// single-token call is a decode step. Equivalent to `forwardLastLogits` for
    /// the same prefix, but O(1)-state per linear layer instead of recomputing.
    pub fn forwardStep(self: *const Model, ctx: *ExecContext, cache: *Cache, token_ids: []const usize, pos0: usize) !fucina.Tensor(.{ .seq, .vocab }) {
        return self.forwardStepImpl(ctx, cache, token_ids, pos0, .chunked, null, null);
    }

    pub fn forwardStepWithScanMode(
        self: *const Model,
        ctx: *ExecContext,
        cache: *Cache,
        token_ids: []const usize,
        pos0: usize,
        scan_mode: LinearScanMode,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        return self.forwardStepImpl(ctx, cache, token_ids, pos0, scan_mode, null, null);
    }

    pub fn forwardStepProfiled(
        self: *const Model,
        ctx: *ExecContext,
        cache: *Cache,
        token_ids: []const usize,
        pos0: usize,
        io: std.Io,
        profile: *ForwardProfile,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        return self.forwardStepImpl(ctx, cache, token_ids, pos0, .chunked, io, profile);
    }

    pub fn forwardStepProfiledWithScanMode(
        self: *const Model,
        ctx: *ExecContext,
        cache: *Cache,
        token_ids: []const usize,
        pos0: usize,
        scan_mode: LinearScanMode,
        io: std.Io,
        profile: *ForwardProfile,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        return self.forwardStepImpl(ctx, cache, token_ids, pos0, scan_mode, io, profile);
    }

    fn forwardStepImpl(
        self: *const Model,
        ctx: *ExecContext,
        cache: *Cache,
        token_ids: []const usize,
        pos0: usize,
        scan_mode: LinearScanMode,
        io: ?std.Io,
        profile: ?*ForwardProfile,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        const total_start = profileStart(profile, io);
        if (profile) |p| p.tokens += token_ids.len;
        if (token_ids.len == 0) return Error.InvalidSequenceLength;
        try requireF16KvCache(&cache.kv);
        if (cache.kv.len != pos0) return Error.InvalidSequenceLength;
        if (cache.kv.len + token_ids.len > cache.kv.capacity) return kv_cache.Error.KvCacheOverflow;
        const cfg = self.config;

        const prep_start = profileStart(profile, io);
        const positions = try ctx.allocator.alloc(i32, token_ids.len);
        defer ctx.allocator.free(positions);
        for (positions, 0..) |*p, i| p.* = @intCast(pos0 + i);

        var rope_table = try ctx.prepareRopeTable(positions, cfg.rope_n_rot, cfg.rope_theta, false);
        defer rope_table.deinit();
        profileAdd(profile, io, prep_start, .prep);

        const embed_start = profileStart(profile, io);
        var x = try self.token_embedding.getRowsAs(ctx, token_ids, .embed);
        errdefer x.deinit();
        profileAdd(profile, io, embed_start, .embed);

        for (self.layers, 0..) |*layer, il| {
            switch (layer.*) {
                .attn => |*a| x = try ctx.replace(x, attnForward(ctx, cfg, a, &x, &rope_table, self.kv_head_for_head, &cache.kv, il, io, profile)),
                .linear => |*l| x = try ctx.replace(x, linearForward(ctx, cfg, l, &x, cache.convSlice(il), cache.ssmSlice(il), scan_mode, io, profile)),
            }
            // Multi-token prefill on the last layer: only the final position
            // feeds the logits, so its FFN (and the final norm below) run on
            // one row. Mixing already updated the caches full-width above.
            if (il + 1 == self.layers.len and token_ids.len > 1) {
                x = try ctx.replace(x, x.narrow(ctx, .seq, x.dim(.seq) - 1, 1));
            }
            const post_norm, const ffn = switch (layer.*) {
                .attn => |*a| .{ &a.post_norm, &a.ffn },
                .linear => |*l| .{ &l.post_norm, &l.ffn },
            };
            const ffn_start = profileStart(profile, io);
            x = try ctx.replace(x, ffnForward(ctx, cfg, post_norm, ffn, &x, io, profile));
            profileAdd(profile, io, ffn_start, .ffn);
        }
        cache.kv.advance(token_ids.len);

        const final_start = profileStart(profile, io);
        var final_norm = try x.rmsNormMul(ctx, .embed, &self.output_norm, cfg.rms_norm_eps);
        defer final_norm.deinit();
        x.deinit();

        var last = try final_norm.narrow(ctx, .seq, final_norm.dim(.seq) - 1, 1);
        defer last.deinit();
        var logits = try self.output.linearSeq(ctx, &last, .embed, .vocab);
        errdefer logits.deinit();
        profileAdd(profile, io, final_start, .final);
        profileAdd(profile, io, total_start, .total);
        return logits;
    }
};

/// Streaming-decode cache: the attention KV cache plus, per DeltaNet-linear
/// layer, the conv window (`[(d_conv-1)·conv_dim]`) and the recurrent state
/// (`[H·Sd·Sd]`). State for the 6 attention layers in `conv`/`ssm` is unused.
pub const Cache = struct {
    allocator: Allocator,
    kv: KvCache,
    conv: []f32,
    ssm: []f32,
    conv_stride: usize,
    ssm_stride: usize,

    fn init(model: *const Model, ctx: *ExecContext, capacity: usize) !Cache {
        const cfg = model.config;
        var kv = try KvCache.init(ctx, cfg.num_layers, cfg.num_key_value_heads, cfg.head_dim, capacity);
        errdefer kv.deinit();
        const conv_stride = (cfg.ssm_d_conv - 1) * cfg.convDim();
        const ssm_stride = cfg.numVHeads() * cfg.headVDim() * cfg.headVDim();
        const conv = try ctx.allocator.alloc(f32, cfg.num_layers * conv_stride);
        errdefer ctx.allocator.free(conv);
        @memset(conv, 0);
        const ssm = try ctx.allocator.alloc(f32, cfg.num_layers * ssm_stride);
        @memset(ssm, 0);
        return .{ .allocator = ctx.allocator, .kv = kv, .conv = conv, .ssm = ssm, .conv_stride = conv_stride, .ssm_stride = ssm_stride };
    }

    pub fn deinit(self: *Cache) void {
        self.allocator.free(self.ssm);
        self.allocator.free(self.conv);
        self.kv.deinit();
        self.* = undefined;
    }

    /// Reset to an empty sequence (zero all carried state).
    pub fn reset(self: *Cache) void {
        self.kv.reset();
        @memset(self.conv, 0);
        @memset(self.ssm, 0);
    }

    pub fn len(self: *const Cache) usize {
        return self.kv.len;
    }

    fn convSlice(self: *Cache, il: usize) []f32 {
        return self.conv[il * self.conv_stride ..][0..self.conv_stride];
    }
    fn ssmSlice(self: *Cache, il: usize) []f32 {
        return self.ssm[il * self.ssm_stride ..][0..self.ssm_stride];
    }
};

const ProfileBucket = enum {
    total,
    prep,
    embed,
    attn_total,
    attn_proj,
    attn_rope,
    attn_sdpa,
    attn_out,
    linear_total,
    linear_norm,
    linear_qkv,
    linear_conv,
    linear_gate,
    linear_z,
    linear_alpha,
    linear_beta,
    linear_scan,
    linear_out,
    ffn,
    ffn_norm,
    ffn_gate_up,
    ffn_gate,
    ffn_up,
    ffn_act,
    ffn_down,
    ffn_residual,
    final,
};

fn profileStart(profile: ?*ForwardProfile, io: ?std.Io) i128 {
    if (profile == null) return 0;
    return std.Io.Clock.awake.now(io.?).nanoseconds;
}

fn profileAdd(profile: ?*ForwardProfile, io: ?std.Io, start: i128, bucket: ProfileBucket) void {
    const p = profile orelse return;
    const elapsed = std.Io.Clock.awake.now(io.?).nanoseconds - start;
    switch (bucket) {
        .total => p.total_ns += elapsed,
        .prep => p.prep_ns += elapsed,
        .embed => p.embed_ns += elapsed,
        .attn_total => p.attn_total_ns += elapsed,
        .attn_proj => p.attn_proj_ns += elapsed,
        .attn_rope => p.attn_rope_ns += elapsed,
        .attn_sdpa => p.attn_sdpa_ns += elapsed,
        .attn_out => p.attn_out_ns += elapsed,
        .linear_total => p.linear_total_ns += elapsed,
        .linear_norm => p.linear_norm_ns += elapsed,
        .linear_qkv => p.linear_qkv_ns += elapsed,
        .linear_conv => p.linear_conv_ns += elapsed,
        .linear_gate => p.linear_gate_ns += elapsed,
        .linear_z => p.linear_z_ns += elapsed,
        .linear_alpha => p.linear_alpha_ns += elapsed,
        .linear_beta => p.linear_beta_ns += elapsed,
        .linear_scan => p.linear_scan_ns += elapsed,
        .linear_out => p.linear_out_ns += elapsed,
        .ffn => p.ffn_ns += elapsed,
        .ffn_norm => p.ffn_norm_ns += elapsed,
        .ffn_gate_up => p.ffn_gate_up_ns += elapsed,
        .ffn_gate => p.ffn_gate_ns += elapsed,
        .ffn_up => p.ffn_up_ns += elapsed,
        .ffn_act => p.ffn_act_ns += elapsed,
        .ffn_down => p.ffn_down_ns += elapsed,
        .ffn_residual => p.ffn_residual_ns += elapsed,
        .final => p.final_ns += elapsed,
    }
}

/// Partial NEOX RoPE: rotate the first `n_rot` feature dims (using `table`,
/// whose `feature_dim` must equal `n_rot`), pass the rest through unchanged.
/// Works for any `[.seq, .*, .d]` tensor (query or key heads).
fn partialRope(ctx: *ExecContext, x: anytype, n_rot: usize, table: *const fucina.RopeTable) !@TypeOf(x.*) {
    // `table.feature_dim` is `rope`'s authoritative rotary span (full when it
    // equals d, prefix rotation when smaller): reject a mismatched table
    // before it silently rotates the wrong prefix.
    if (n_rot != table.feature_dim) return Error.InvalidConfig;
    return x.rope(ctx, .seq, .d, table, .half);
}

/// qwen35's attention reads the `kv.k`/`kv.v` f16 tensor views directly; a
/// q8_0 cache stores blocks in `k_q8`/`v_q8` and leaves those views EMPTY, so
/// indexing them would be out of bounds. Reject non-f16 caches at the forward
/// seam (qwen3 is the only model with a q8_0 attention path).
fn requireF16KvCache(kv: *const KvCache) Error!void {
    switch (kv.dtype) {
        .f16 => {},
        .q8_0 => return Error.UnsupportedKvCacheDtype,
    }
}

/// One full-attention block: returns `input + attn_out`. Fused Q+gate
/// projection (per head: [query | gate]), per-head q/k RMSNorm, partial RoPE,
/// GQA SDPA, sigmoid output gate, output projection.
fn attnForward(
    ctx: *ExecContext,
    cfg: Config,
    layer: *const AttnLayer,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    rope_table: *const fucina.RopeTable,
    kv_head_for_head: []const usize,
    kv: ?*KvCache, // when non-null: append this layer's K/V and attend over the cache
    layer_i: usize,
    io: ?std.Io,
    profile: ?*ForwardProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    const total_start = profileStart(profile, io);
    if (profile) |p| p.attn_layers += 1;

    const proj_start = profileStart(profile, io);
    var attn_in = try input.rmsNormMul(ctx, .embed, &layer.attn_norm, cfg.rms_norm_eps);
    defer attn_in.deinit();

    // Fused Q+gate: [seq, n_head * (2*head_dim)] → split heads, then [query|gate].
    var qg = try layer.q_gate_proj.linearSeq(ctx, &attn_in, .embed, .qg);
    defer qg.deinit();
    var qg3 = try qg.split(ctx, .qg, .{ .head, .dd }, .{ cfg.num_attention_heads, 2 * cfg.head_dim });
    defer qg3.deinit();
    var q_view = try qg3.narrow(ctx, .dd, 0, cfg.head_dim);
    defer q_view.deinit();
    var q3 = try q_view.withTags(ctx, .{ .seq, .head, .d });
    defer q3.deinit();
    var g_view = try qg3.narrow(ctx, .dd, cfg.head_dim, cfg.head_dim);
    defer g_view.deinit();
    var gate3 = try g_view.withTags(ctx, .{ .seq, .head, .d });
    defer gate3.deinit();

    var k = try layer.k_proj.linearSeq(ctx, &attn_in, .embed, .k);
    defer k.deinit();
    var k3 = try k.split(ctx, .k, .{ .kv_head, .d }, .{ cfg.num_key_value_heads, cfg.head_dim });
    defer k3.deinit();
    var v = try layer.v_proj.linearSeq(ctx, &attn_in, .embed, .v);
    defer v.deinit();
    var v3 = try v.split(ctx, .v, .{ .kv_head, .d }, .{ cfg.num_key_value_heads, cfg.head_dim });
    defer v3.deinit();

    // Per-head RMSNorm (over head_dim) then partial RoPE.
    var q_norm = try q3.rmsNormMul(ctx, .d, &layer.q_norm, cfg.rms_norm_eps);
    defer q_norm.deinit();
    var k_norm = try k3.rmsNormMul(ctx, .d, &layer.k_norm, cfg.rms_norm_eps);
    defer k_norm.deinit();
    profileAdd(profile, io, proj_start, .attn_proj);

    const rope_start = profileStart(profile, io);
    var q_rope = try partialRope(ctx, &q_norm, cfg.rope_n_rot, rope_table);
    defer q_rope.deinit();
    var k_rope = try partialRope(ctx, &k_norm, cfg.rope_n_rot, rope_table);
    defer k_rope.deinit();
    profileAdd(profile, io, rope_start, .attn_rope);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const sdpa_start = profileStart(profile, io);
    var attn = if (kv) |cache| blk: {
        try cache.appendLayer(ctx, layer_i, &k_rope, &v3);
        const cached_len = cache.len + k_rope.dim(.seq);
        var k_view = try cache.k[layer_i].narrow(ctx, .seq, 0, cached_len);
        defer k_view.deinit();
        var v_view = try cache.v[layer_i].narrow(ctx, .seq, 0, cached_len);
        defer v_view.deinit();
        break :blk try q_rope.groupedAttention(ctx, &k_view, &v_view, kv_head_for_head, .attn, scale, .{});
    } else try q_rope.groupedAttention(ctx, &k_rope, &v3, kv_head_for_head, .attn, scale, .{});
    defer attn.deinit();
    profileAdd(profile, io, sdpa_start, .attn_sdpa);

    // Output gate: attn * sigmoid(gate), then output projection. sigmoid first
    // (its output is contiguous) so the per-head [query|gate] strided `gate3`
    // can be merged head-major to match the attention output's [seq, attn].
    const out_start = profileStart(profile, io);
    var gate_sig = try gate3.sigmoid(ctx);
    defer gate_sig.deinit();
    var gate_flat = try gate_sig.merge(ctx, .attn, .{ .head, .d });
    defer gate_flat.deinit();
    var gated = try attn.mul(ctx, &gate_flat);
    defer gated.deinit();
    var attn_out = try layer.o_proj.linearSeq(ctx, &gated, .attn, .embed);
    defer attn_out.deinit();

    var result = try input.add(ctx, &attn_out);
    errdefer result.deinit();
    profileAdd(profile, io, out_start, .attn_out);
    profileAdd(profile, io, total_start, .attn_total);
    return result;
}

fn softplus(x: f32) f32 {
    return if (x > 20.0) x else @log(1.0 + @exp(x));
}
fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}
fn siluScalar(x: f32) f32 {
    return x * sigmoid(x);
}

/// One v-head's recurrent DeltaNet scan over the whole sequence. Heads are
/// independent, so these run in parallel across the work pool; each task owns
/// its `state`/`uo` scratch and writes only its head's slice of `out`.
const HeadScanTask = struct {
    h: usize,
    seq: usize,
    Sd: usize,
    H: usize,
    conv_dim: usize,
    vd: usize,
    k_off: usize,
    v_off: usize,
    scale: f32,
    eps: f32,
    cd: []const f32, // conv(silu) output [seq][conv_dim], q|k|v head-major
    zd: []const f32, // z projection, row stride `z_stride`
    z_stride: usize,
    ad: []const f32, // alpha, row stride `alpha_stride`
    alpha_stride: usize,
    bd: []const f32, // beta, row stride `beta_stride`
    beta_stride: usize,
    ssm_a: []const f32,
    ssm_dt: []const f32,
    ssm_norm: []const f32,
    out: []f32, // [seq][vd]
    state: []f32, // [Sd*Sd] — scratch (whole-seq) or the cache's per-head S (decode)
    uo: []f32, // [2*Sd] : u (then d) | o
    reset_state: bool, // true = start from zero (prefill); false = carry cached S (decode)
};

fn runHeadScan(task: *const HeadScanTask) void {
    const h = task.h;
    const Sd = task.Sd;
    const eps = task.eps;
    const cd = task.cd;
    const state = task.state;
    const u = task.uo[0..Sd];
    const o = task.uo[Sd..][0..Sd];

    if (task.reset_state) @memset(state, 0);
    for (0..task.seq) |t| {
        const qb = t * task.conv_dim + h * Sd;
        const kb = t * task.conv_dim + task.k_off + h * Sd;
        const vb = t * task.conv_dim + task.v_off + h * Sd;

        // L2-norm scales for q and k (1 / max(sqrt(Σx²), eps)).
        var qss: f32 = 0;
        var kss: f32 = 0;
        for (0..Sd) |i| {
            qss += cd[qb + i] * cd[qb + i];
            kss += cd[kb + i] * cd[kb + i];
        }
        const qn = 1.0 / @max(@sqrt(qss), eps);
        const kn = 1.0 / @max(@sqrt(kss), eps);

        // Gates.
        const g = @exp(softplus(task.ad[t * task.alpha_stride + h] + task.ssm_dt[h]) * task.ssm_a[h]);
        const beta = sigmoid(task.bd[t * task.beta_stride + h]);

        // S *= g
        for (state) |*s| s.* *= g;
        // u[j] = Σ_i S[i,j]·k̂[i]
        @memset(u, 0);
        for (0..Sd) |i| {
            const ki = cd[kb + i] * kn;
            const row = i * Sd;
            for (0..Sd) |j| u[j] += state[row + j] * ki;
        }
        // d[j] = (v[j] − u[j])·beta  (store back into u)
        for (0..Sd) |j| u[j] = (cd[vb + j] - u[j]) * beta;
        // S[i,j] += k̂[i]·d[j]
        for (0..Sd) |i| {
            const ki = cd[kb + i] * kn;
            const row = i * Sd;
            for (0..Sd) |j| state[row + j] += ki * u[j];
        }
        // o[j] = Σ_i S[i,j]·(q̂[i]·scale)
        @memset(o, 0);
        for (0..Sd) |i| {
            const qi = cd[qb + i] * qn * task.scale;
            const row = i * Sd;
            for (0..Sd) |j| o[j] += state[row + j] * qi;
        }
        // Gated RMSNorm: rmsnorm(o)·ssm_norm·silu(z)
        var oss: f32 = 0;
        for (0..Sd) |j| oss += o[j] * o[j];
        const rms = 1.0 / @sqrt(oss / @as(f32, @floatFromInt(Sd)) + eps);
        const ob = t * task.vd + h * Sd;
        const zb = t * task.z_stride + h * Sd;
        for (0..Sd) |j| task.out[ob + j] = o[j] * rms * task.ssm_norm[j] * siluScalar(task.zd[zb + j]);
    }
}

/// Chunked-parallel form of the per-head DeltaNet scan (the `runHeadScan`
/// recurrence), derived from that recurrence so it is mathematically identical.
///
/// Process the sequence in chunks of `CS`, carrying the `[Sd×Sd]` state `S`
/// across chunks. Within a chunk (incoming state S₀, cumulative log-decay
/// `G_t = Σ_{τ≤t} lg_τ`, `P_t = exp(G_t)`):
///   RHS_t = β_t·(v_t − P_t·Sᵀ₀k_t)
///   δ solves the unit-lower-triangular system  δ_t = RHS_t − Σ_{s<t} β_t·
///        exp(G_t−G_s)·(k_s·k_t)·δ_s          (forward substitution)
///   o_t = P_t·Sᵀ₀q_t + Σ_{s≤t} exp(G_t−G_s)·(k_s·q_t)·δ_s     (q pre-scaled)
///   S ← P_{C−1}·S₀ + Σ_s exp(G_{C−1}−G_s)·k_s⊗δ_s
/// `q` is the L2-normed query already multiplied by `1/√Sd`. Ratios use
/// `exp(G_t−G_s)` (bounded) rather than `P_t/P_s` to avoid underflow.
///
/// Scratch: `delta` `[CS·Sd]`, `G` `[CS]`, `tmp` `[Sd]`.
fn deltaNetChunked(
    o: []f32,
    state: []f32,
    k: []const f32,
    v: []const f32,
    q: []const f32,
    lg: []const f32,
    beta: []const f32,
    seq: usize,
    Sd: usize,
    CS: usize,
    delta: []f32,
    gcs: []f32,
    tmp: []f32,
) void {
    var c0: usize = 0;
    while (c0 < seq) : (c0 += CS) {
        const C = @min(CS, seq - c0);

        // Cumulative log-decay over the chunk.
        var acc: f32 = 0;
        for (0..C) |t| {
            acc += lg[c0 + t];
            gcs[t] = acc;
        }

        // RHS_t = β_t·(v_t − P_t·Sᵀ₀k_t), stored into delta.
        for (0..C) |t| {
            @memset(tmp[0..Sd], 0);
            for (0..Sd) |i| {
                const ki = k[(c0 + t) * Sd + i];
                const row = i * Sd;
                for (0..Sd) |j| tmp[j] += state[row + j] * ki;
            }
            const pt = @exp(gcs[t]);
            const bt = beta[c0 + t];
            const vb = (c0 + t) * Sd;
            for (0..Sd) |j| delta[t * Sd + j] = bt * (v[vb + j] - pt * tmp[j]);
        }

        // Forward substitution: δ_t −= β_t·exp(G_t−G_s)·(k_s·k_t)·δ_s for s<t.
        for (0..C) |t| {
            const bt = beta[c0 + t];
            const kt = (c0 + t) * Sd;
            for (0..t) |s| {
                var kk: f32 = 0;
                const ks = (c0 + s) * Sd;
                for (0..Sd) |i| kk += k[ks + i] * k[kt + i];
                const coef = bt * @exp(gcs[t] - gcs[s]) * kk;
                const dt = t * Sd;
                const ds = s * Sd;
                for (0..Sd) |j| delta[dt + j] -= coef * delta[ds + j];
            }
        }

        // Outputs: o_t = P_t·Sᵀ₀q_t + Σ_{s≤t} exp(G_t−G_s)·(k_s·q_t)·δ_s.
        for (0..C) |t| {
            @memset(tmp[0..Sd], 0);
            const qt = (c0 + t) * Sd;
            for (0..Sd) |i| {
                const qi = q[qt + i];
                const row = i * Sd;
                for (0..Sd) |j| tmp[j] += state[row + j] * qi;
            }
            const pt = @exp(gcs[t]);
            const ob = (c0 + t) * Sd;
            for (0..Sd) |j| o[ob + j] = pt * tmp[j];
            for (0..t + 1) |s| {
                var kq: f32 = 0;
                const ks = (c0 + s) * Sd;
                for (0..Sd) |i| kq += k[ks + i] * q[qt + i];
                const coef = @exp(gcs[t] - gcs[s]) * kq;
                const ds = s * Sd;
                for (0..Sd) |j| o[ob + j] += coef * delta[ds + j];
            }
        }

        // Carry state: S ← P_{C−1}·S₀ + Σ_s exp(G_{C−1}−G_s)·k_s⊗δ_s.
        const plast = @exp(gcs[C - 1]);
        for (state) |*s| s.* *= plast;
        for (0..C) |s| {
            const coef = @exp(gcs[C - 1] - gcs[s]);
            const ks = (c0 + s) * Sd;
            const ds = s * Sd;
            for (0..Sd) |i| {
                const ki = k[ks + i] * coef;
                const row = i * Sd;
                for (0..Sd) |j| state[row + j] += ki * delta[ds + j];
            }
        }
    }
}

/// Inputs to the batched chunked-GEMM prefill scan (`deltaNetScanBatched`).
const BatchedScan = struct {
    H: usize, // num v-heads (== num k-heads)
    Sd: usize, // head dim (== state dim)
    seq: usize,
    conv_dim: usize,
    vd: usize, // value_dim = H*Sd
    k_off: usize, // q|k|v offsets within a conv-silu row
    v_off: usize,
    scale: f32, // 1/sqrt(Sd)
    eps: f32,
    cd: []const f32, // conv(silu) output [seq][conv_dim], q|k|v head-major
    zd: []const f32, // z projection, row stride `z_stride`
    z_stride: usize,
    ad: []const f32, // alpha, row stride `alpha_stride`
    alpha_stride: usize,
    bd: []const f32, // beta, row stride `beta_stride`
    beta_stride: usize,
    ssm_a: []const f32, // [H] (-A_log)
    ssm_dt: []const f32, // [H] dt bias
    ssm_norm: []const f32, // [Sd] gated RMSNorm weight
    out: []f32, // [seq][vd] gated output
    state: []f32, // [H*Sd*Sd] per-head state S (carried in/out)
    reset_state: bool, // true = zero S first (fresh prefill)
};

/// Mutable per-chunk context shared by the per-head scalar phases of the batched
/// scan. The `*_out` fields point at the (transient) batched-GEMM result buffers
/// and are refreshed by the main thread between phase barriers; everything else
/// is fixed for the call. Phases only ever touch their own head's slices, so the
/// 16 heads run concurrently with no synchronization.
const ScanCtx = struct {
    H: usize,
    Sd: usize,
    C: usize, // current chunk length
    c0: usize, // current chunk start
    conv_dim: usize,
    vd: usize,
    k_off: usize,
    v_off: usize,
    scale: f32,
    eps: f32,
    cd: []const f32,
    zd: []const f32,
    z_stride: usize,
    ad: []const f32,
    alpha_stride: usize,
    bd: []const f32,
    beta_stride: usize,
    ssm_a: []const f32,
    ssm_dt: []const f32,
    ssm_norm: []const f32,
    out: []f32,
    state: []f32,
    kf: []f32,
    qf: []f32,
    vf: []f32,
    kw: []f32,
    delta: []f32,
    am: []f32,
    gcs: []f32,
    betac: []f32,
    // Batched-GEMM outputs for the current chunk (set just before the phase that reads them).
    vp_out: []const f32 = &.{},
    oq_out: []const f32 = &.{},
    kk_out: []const f32 = &.{},
    qk_out: []const f32 = &.{},
    va_out: []const f32 = &.{},
    kgv_out: []const f32 = &.{},
};

const HeadTask = struct { cx: *const ScanCtx, h: usize };

/// Run a per-head phase across the work pool (one task per head). Safe to call
/// from `deltaNetScanBatched`'s main thread: the phase bodies contain no `dot`/
/// pooled GEMM, so there is no nesting with the batched GEMMs run between phases.
fn dispatchHeads(ctx: *ExecContext, tasks: []const HeadTask, comptime run: fn (*const HeadTask) void) void {
    if (ctx.workPool()) |pool| {
        pool.parallelChunks(HeadTask, tasks, run);
    } else {
        for (tasks) |*t| run(t);
    }
}

fn borrowedTaggedTensor(
    comptime tags_spec: anytype,
    ctx: *ExecContext,
    raw_shape: [fucina.Tensor(tags_spec).tensor_rank]usize,
    values: []f32,
) !fucina.Tensor(tags_spec) {
    // The public facade `fromBorrowedSlice` IS this borrow+wrap
    // (ctx.fromBorrowedSliceRank → constant), no local raw bridging needed.
    return fucina.Tensor(tags_spec).fromBorrowedSlice(ctx, raw_shape, values);
}

// Phase 0: repack q/k/v into contiguous [H,C,Sd] (L2-norm k,q; scale q), gates,
// cumulative log-decay G, and the decay-weighted key Kw (state-update operand).
fn scanRepack(t: *const HeadTask) void {
    const cx = t.cx;
    const h = t.h;
    const Sd = cx.Sd;
    const C = cx.C;
    var acc: f32 = 0;
    for (0..C) |ti| {
        const tt = cx.c0 + ti;
        const bq = tt * cx.conv_dim + h * Sd;
        const bk = tt * cx.conv_dim + cx.k_off + h * Sd;
        const bv = tt * cx.conv_dim + cx.v_off + h * Sd;
        var qss: f32 = 0;
        var kss: f32 = 0;
        for (0..Sd) |i| {
            qss += cx.cd[bq + i] * cx.cd[bq + i];
            kss += cx.cd[bk + i] * cx.cd[bk + i];
        }
        const qn = 1.0 / @max(@sqrt(qss), cx.eps);
        const kn = 1.0 / @max(@sqrt(kss), cx.eps);
        const dst = (h * C + ti) * Sd;
        for (0..Sd) |i| {
            cx.kf[dst + i] = cx.cd[bk + i] * kn;
            cx.qf[dst + i] = cx.cd[bq + i] * qn * cx.scale;
            cx.vf[dst + i] = cx.cd[bv + i];
        }
        acc += softplus(cx.ad[tt * cx.alpha_stride + h] + cx.ssm_dt[h]) * cx.ssm_a[h];
        cx.gcs[h * C + ti] = acc;
        cx.betac[h * C + ti] = sigmoid(cx.bd[tt * cx.beta_stride + h]);
    }
    // Kw_s = exp(G_{C−1}−G_s)·k_s (needs this head's full cumsum).
    const glast = cx.gcs[h * C + C - 1];
    for (0..C) |s| {
        const w = @exp(glast - cx.gcs[h * C + s]);
        const off = (h * C + s) * Sd;
        for (0..Sd) |i| cx.kw[off + i] = cx.kf[off + i] * w;
    }
}

// Phase 1: RHS δ_t = β_t·(v_t − P_t·Vp_t).
fn scanRHS(t: *const HeadTask) void {
    const cx = t.cx;
    const h = t.h;
    const Sd = cx.Sd;
    const C = cx.C;
    for (0..C) |ti| {
        const pt = @exp(cx.gcs[h * C + ti]);
        const bt = cx.betac[h * C + ti];
        const off = (h * C + ti) * Sd;
        for (0..Sd) |j| cx.delta[off + j] = bt * (cx.vf[off + j] - pt * cx.vp_out[off + j]);
    }
}

// Phase 2: forward substitution δ_t −= β_t·exp(G_t−G_s)·(k_s·k_t)·δ_s, s<t.
fn scanFwdSubst(t: *const HeadTask) void {
    const cx = t.cx;
    const h = t.h;
    const Sd = cx.Sd;
    const C = cx.C;
    for (0..C) |ti| {
        const bt = cx.betac[h * C + ti];
        const dt_off = (h * C + ti) * Sd;
        for (0..ti) |s| {
            const coef = bt * @exp(cx.gcs[h * C + ti] - cx.gcs[h * C + s]) * cx.kk_out[(h * C + s) * C + ti];
            const ds_off = (h * C + s) * Sd;
            for (0..Sd) |j| cx.delta[dt_off + j] -= coef * cx.delta[ds_off + j];
        }
    }
}

// Phase 3: build A[t,s] = (s≤t ? exp(G_t−G_s)·QK[s,t] : 0).
fn scanABuild(t: *const HeadTask) void {
    const cx = t.cx;
    const h = t.h;
    const C = cx.C;
    for (0..C) |ti| for (0..C) |s| {
        const amoff = (h * C + ti) * C + s;
        cx.am[amoff] = if (s <= ti) @exp(cx.gcs[h * C + ti] - cx.gcs[h * C + s]) * cx.qk_out[(h * C + s) * C + ti] else 0;
    };
}

// Phase 4: o_t = P_t·Oq_t + Vattn_t; gated RMSNorm·ssm_norm·silu(z) → out.
fn scanOutput(t: *const HeadTask) void {
    const cx = t.cx;
    const h = t.h;
    const Sd = cx.Sd;
    const C = cx.C;
    for (0..C) |ti| {
        const pt = @exp(cx.gcs[h * C + ti]);
        const off = (h * C + ti) * Sd;
        var oss: f32 = 0;
        for (0..Sd) |j| {
            const oj = pt * cx.oq_out[off + j] + cx.va_out[off + j];
            oss += oj * oj;
        }
        const rms = 1.0 / @sqrt(oss / @as(f32, @floatFromInt(Sd)) + cx.eps);
        const ob = (cx.c0 + ti) * cx.vd + h * Sd;
        const zb = (cx.c0 + ti) * cx.z_stride + h * Sd;
        for (0..Sd) |j| {
            const oj = pt * cx.oq_out[off + j] + cx.va_out[off + j];
            cx.out[ob + j] = oj * rms * cx.ssm_norm[j] * siluScalar(cx.zd[zb + j]);
        }
    }
}

// Phase 5: state carry S ← P_{C−1}·S₀ + KGV.
fn scanCarry(t: *const HeadTask) void {
    const cx = t.cx;
    const h = t.h;
    const Sd = cx.Sd;
    const plast = @exp(cx.gcs[h * cx.C + cx.C - 1]);
    const so = h * Sd * Sd;
    for (0..Sd * Sd) |ij| cx.state[so + ij] = plast * cx.state[so + ij] + cx.kgv_out[so + ij];
}

/// Batched chunked-GEMM DeltaNet prefill scan — the perf path that replaces the
/// per-token recurrent `runHeadScan` for sequences longer than one token. It is
/// the GEMM-ized form of `deltaNetChunked` (its scalar oracle): the sequence is
/// processed in chunks of `CS`, carrying the `[Sd×Sd]` per-head state, and every
/// per-head matmul is issued as ONE batched `dot` over the `.head` axis (16 heads
/// → a single `cblas_sgemm_batch` at `CS=64`). The element-wise / forward-subst
/// work is split across the pool one-task-per-head (`scanRepack` … `scanCarry`),
/// interleaved with the main-thread batched GEMMs. The two parallelisms never
/// nest: the GEMMs run on the main thread (between phase barriers), so the BLAS
/// pool and the per-head phase pool are never active at the same time.
///
/// Per chunk (incoming state S₀, cumulative log-decay `G_t`, `P_t=exp(G_t)`):
///   Vp = K·S₀, Oq = Q·S₀                     (batched, S₀-dependent)
///   δ ← β·(v − P·Vp)                          (RHS)
///   KK = K·Kᵀ  → forward-subst δ              (transB dot + scalar solve)
///   QK = K·Qᵀ  → A[t,s]=exp(G_t−G_s)·QK[s,t]  (transB dot + element-wise mask)
///   Vattn = A·δ                               (batched dot)
///   o = P·Oq + Vattn  → gated RMSNorm·silu(z) (→ `out`)
///   KGV = Kwᵀ·δ,  Kw_s = exp(G_{C−1}−G_s)·k_s (transA dot)
///   S ← P_{C−1}·S₀ + KGV                       (carry)
fn deltaNetScanBatched(ctx: *ExecContext, p: BatchedScan) !void {
    const a = ctx.allocator;
    const H = p.H;
    const Sd = p.Sd;
    const seq = p.seq;
    const eps = p.eps;
    // GDA chunk size (llama.cpp uses 64 for scalar-per-head gates); capped at seq
    // so short prompts run in a single chunk.
    const CS = @min(@as(usize, 64), seq);

    if (p.reset_state) @memset(p.state, 0);

    // Chunk-local scratch (sized for the max chunk CS, reused across chunks).
    const kf = try a.alloc(f32, H * CS * Sd);
    defer a.free(kf);
    const qf = try a.alloc(f32, H * CS * Sd);
    defer a.free(qf);
    const vf = try a.alloc(f32, H * CS * Sd);
    defer a.free(vf);
    const kw = try a.alloc(f32, H * CS * Sd);
    defer a.free(kw);
    const delta = try a.alloc(f32, H * CS * Sd);
    defer a.free(delta);
    const am = try a.alloc(f32, H * CS * CS);
    defer a.free(am);
    const gcs = try a.alloc(f32, H * CS); // cumulative log-decay G_{h,t}
    defer a.free(gcs);
    const betac = try a.alloc(f32, H * CS);
    defer a.free(betac);

    var cx = ScanCtx{
        .H = H,
        .Sd = Sd,
        .C = 0,
        .c0 = 0,
        .conv_dim = p.conv_dim,
        .vd = p.vd,
        .k_off = p.k_off,
        .v_off = p.v_off,
        .scale = p.scale,
        .eps = eps,
        .cd = p.cd,
        .zd = p.zd,
        .ad = p.ad,
        .bd = p.bd,
        .ssm_a = p.ssm_a,
        .ssm_dt = p.ssm_dt,
        .ssm_norm = p.ssm_norm,
        .out = p.out,
        .state = p.state,
        .kf = kf,
        .qf = qf,
        .vf = vf,
        .kw = kw,
        .delta = delta,
        .am = am,
        .gcs = gcs,
        .betac = betac,
        .z_stride = p.z_stride,
        .alpha_stride = p.alpha_stride,
        .beta_stride = p.beta_stride,
    };
    const tasks = try a.alloc(HeadTask, H);
    defer a.free(tasks);
    for (tasks, 0..) |*t, h| t.* = .{ .cx = &cx, .h = h };

    var c0: usize = 0;
    while (c0 < seq) : (c0 += CS) {
        const C = @min(CS, seq - c0);
        cx.C = C;
        cx.c0 = c0;

        dispatchHeads(ctx, tasks, scanRepack);

        var k_chunk = try borrowedTaggedTensor(.{ .head, .c, .i }, ctx, .{ H, C, Sd }, kf[0 .. H * C * Sd]);
        defer k_chunk.deinit();
        var q_chunk = try borrowedTaggedTensor(.{ .head, .c, .i }, ctx, .{ H, C, Sd }, qf[0 .. H * C * Sd]);
        defer q_chunk.deinit();
        var s_mat = try borrowedTaggedTensor(.{ .head, .i, .j }, ctx, .{ H, Sd, Sd }, p.state);
        defer s_mat.deinit();

        // Vp = K·S₀, Oq = Q·S₀ (batched plain dots; S₀-dependent) → RHS.
        var vp_t = try k_chunk.dot(ctx, &s_mat, .i); // {.head,.c,.j}
        defer vp_t.deinit();
        var oq_t = try q_chunk.dot(ctx, &s_mat, .i);
        defer oq_t.deinit();
        cx.vp_out = try vp_t.dataConst();
        cx.oq_out = try oq_t.dataConst();
        dispatchHeads(ctx, tasks, scanRHS);

        // KK = K·Kᵀ (transB) → forward substitution.
        var k_s = try k_chunk.withTags(ctx, .{ .head, .s, .i });
        defer k_s.deinit();
        var k_t = try k_chunk.withTags(ctx, .{ .head, .t, .i });
        defer k_t.deinit();
        var kk_t = try k_s.dot(ctx, &k_t, .i); // {.head,.s,.t}
        defer kk_t.deinit();
        cx.kk_out = try kk_t.dataConst();
        dispatchHeads(ctx, tasks, scanFwdSubst);

        // QK = K·Qᵀ (transB) → build A.
        var q_t = try q_chunk.withTags(ctx, .{ .head, .t, .i });
        defer q_t.deinit();
        var qk_t = try k_s.dot(ctx, &q_t, .i); // {.head,.s,.t}
        defer qk_t.deinit();
        cx.qk_out = try qk_t.dataConst();
        dispatchHeads(ctx, tasks, scanABuild);

        // Vattn = A·δ (batched plain) → gated output.
        var delta_t = try borrowedTaggedTensor(.{ .head, .s, .j }, ctx, .{ H, C, Sd }, delta[0 .. H * C * Sd]);
        defer delta_t.deinit();
        var a_mat = try borrowedTaggedTensor(.{ .head, .t, .s }, ctx, .{ H, C, C }, am[0 .. H * C * C]);
        defer a_mat.deinit();
        var vattn_t = try a_mat.dot(ctx, &delta_t, .s); // {.head,.t,.j}
        defer vattn_t.deinit();
        cx.va_out = try vattn_t.dataConst();
        dispatchHeads(ctx, tasks, scanOutput);

        // KGV = Kwᵀ·δ (transA) → state carry.
        var kw_t = try borrowedTaggedTensor(.{ .head, .s, .i }, ctx, .{ H, C, Sd }, kw[0 .. H * C * Sd]);
        defer kw_t.deinit();
        var kgv_t = try kw_t.dot(ctx, &delta_t, .s); // {.head,.i,.j}
        defer kgv_t.deinit();
        cx.kgv_out = try kgv_t.dataConst();
        dispatchHeads(ctx, tasks, scanCarry);
    }
}

/// One DeltaNet (linear-attention) block: returns `input + ssm_out`.
///
/// Port of qwen35.cpp `build_layer_attn_linear` + delta-net-base.cpp
/// `build_delta_net_autoregressive`. Whole-sequence, stateless (fresh prefill):
/// the recurrence is run token-by-token carrying a per-v-head `[S×S]` state — the
/// simple O(seq) form that is mathematically equivalent to llama.cpp's chunked
/// prefill (the chunked kernel is a Phase-5 perf optimization). The recurrent
/// state cache for streaming decode also comes later.
///
/// Per head `h`, per token `t` (S = head dim, state `S[i,j]`, i=key j=value):
///   S *= exp(softplus(αₜ+dt_bias)·ssm_a)            (gated decay)
///   u[j] = Σᵢ S[i,j]·k̂[i]                           (k̂ = L2-normed key)
///   d[j] = (v[j] − u[j])·sigmoid(βₜ)
///   S[i,j] += k̂[i]·d[j]                             (rank-1 update)
///   o[j] = Σᵢ S[i,j]·(q̂[i]/√S)                       (q̂ = L2-normed query)
///   out[t,h,j] = rmsnorm(o)·ssm_norm[j]·silu(z[t,h,j])
fn linearForward(
    ctx: *ExecContext,
    cfg: Config,
    layer: *const LinearLayer,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    conv_state: ?[]f32, // [(d_conv-1)·conv_dim] carried conv window; null = fresh prefill
    ssm_state: ?[]f32, // [H·Sd·Sd] carried per-head state S; null = fresh (zero) state
    scan_mode: LinearScanMode,
    io: ?std.Io,
    profile: ?*ForwardProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    const a = ctx.allocator;
    const total_start = profileStart(profile, io);
    if (profile) |p| p.linear_layers += 1;

    const seq = input.dim(.seq);
    const H = cfg.numVHeads(); // 16 (== numKHeads)
    const Sd = cfg.headVDim(); // 128 (== headKDim)
    const eps = cfg.rms_norm_eps;
    const conv_dim = cfg.convDim(); // 6144
    const vd = H * Sd; // 2048 (value_dim)
    const k_off = vd; // q [0,vd) | k [vd,2vd) | v [2vd,3vd)
    const v_off = 2 * vd;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(Sd)));

    const norm_start = profileStart(profile, io);
    var attn_in = try input.rmsNormMul(ctx, .embed, &layer.attn_norm, eps);
    defer attn_in.deinit();
    profileAdd(profile, io, norm_start, .linear_norm);

    // Projections + causal conv + SiLU; then read everything as f32 for the scan.
    const qkv_start = profileStart(profile, io);
    var qkv = try layer.qkv_proj.linearSeq(ctx, &attn_in, .embed, .conv); // [seq, conv_dim]
    defer qkv.deinit();
    profileAdd(profile, io, qkv_start, .linear_qkv);

    const conv_start = profileStart(profile, io);
    var conv_kernel = switch (layer.conv1d) {
        .f32 => |*t| try t.withTags(ctx, .{ .conv, .tap }),
        else => return Error.UnsupportedVariant, // qwen35 GGUFs store ssm_conv1d as f32
    };
    defer conv_kernel.deinit();
    var conv = try qkv.causalDepthwiseConv1d(ctx, .seq, .conv, .tap, &conv_kernel, 1, conv_state);
    defer conv.deinit();
    // Refresh the conv window for the next step (last d_conv-1 tokens of qkv).
    if (conv_state) |cs| updateConvState(cs, try qkv.dataConst(), seq, conv_dim, cfg.ssm_d_conv);
    var conv_silu = try conv.silu(ctx);
    defer conv_silu.deinit();
    const cd = try conv_silu.dataConst(); // [seq][conv_dim], q|k|v head-major
    profileAdd(profile, io, conv_start, .linear_conv);

    const gate_start = profileStart(profile, io);
    var gates = try layer.gate_proj.project(ctx, cfg, &attn_in, io, profile);
    defer gates.deinit();
    profileAdd(profile, io, gate_start, .linear_gate);

    const scan_start = profileStart(profile, io);
    var final_t = try fucina.Tensor(.{ .seq, .vd }).empty(ctx, .{ seq, vd });
    defer final_t.deinit();
    const out_final = try final_t.data(); // [seq][h*Sd + j]

    // State scratch: when decoding, the per-head state lives in `ssm_state`
    // (carried across steps); for whole-sequence prefill it's a zeroed scratch.
    const hsz = Sd * Sd;
    const state: []f32 = if (ssm_state) |ss| ss else try a.alloc(f32, H * hsz);
    defer if (ssm_state == null) a.free(state);

    const ssm_a = try layer.ssm_a.dataConst();
    const ssm_dt = try layer.ssm_dt.dataConst();
    const ssm_norm = try layer.ssm_norm.dataConst();

    if (seq > 1 and scan_mode == .chunked) {
        // Prefill: batched chunked-GEMM scan (heads batched across the matmuls).
        try deltaNetScanBatched(ctx, .{
            .H = H,
            .Sd = Sd,
            .seq = seq,
            .conv_dim = conv_dim,
            .vd = vd,
            .k_off = k_off,
            .v_off = v_off,
            .scale = scale,
            .eps = eps,
            .cd = cd,
            .zd = gates.z,
            .z_stride = gates.z_stride,
            .ad = gates.alpha,
            .alpha_stride = gates.alpha_stride,
            .bd = gates.beta,
            .beta_stride = gates.beta_stride,
            .ssm_a = ssm_a,
            .ssm_dt = ssm_dt,
            .ssm_norm = ssm_norm,
            .out = out_final,
            .state = state,
            .reset_state = ssm_state == null,
        });
    } else {
        // Decode, or forced recurrent prefill: per-token scan parallel across heads.
        const uo = try a.alloc(f32, H * 2 * Sd); // u/o scratch is always transient
        defer a.free(uo);
        const tasks = try a.alloc(HeadScanTask, H);
        defer a.free(tasks);
        for (tasks, 0..) |*task, h| task.* = .{
            .h = h,
            .seq = seq,
            .Sd = Sd,
            .H = H,
            .conv_dim = conv_dim,
            .vd = vd,
            .k_off = k_off,
            .v_off = v_off,
            .scale = scale,
            .eps = eps,
            .cd = cd,
            .zd = gates.z,
            .z_stride = gates.z_stride,
            .ad = gates.alpha,
            .alpha_stride = gates.alpha_stride,
            .bd = gates.beta,
            .beta_stride = gates.beta_stride,
            .ssm_a = ssm_a,
            .ssm_dt = ssm_dt,
            .ssm_norm = ssm_norm,
            .out = out_final,
            .state = state[h * hsz ..][0..hsz],
            .uo = uo[h * 2 * Sd ..][0 .. 2 * Sd],
            .reset_state = ssm_state == null,
        };
        // The 16 v-heads are fully independent — run the recurrent scan in parallel.
        if (ctx.workPool()) |pool| {
            pool.parallelChunks(HeadScanTask, tasks, runHeadScan);
        } else {
            for (tasks) |*task| runHeadScan(task);
        }
    }
    profileAdd(profile, io, scan_start, .linear_scan);

    const out_start = profileStart(profile, io);
    var out = try layer.out_proj.linearSeq(ctx, &final_t, .vd, .embed);
    defer out.deinit();
    var result = try input.add(ctx, &out);
    errdefer result.deinit();
    profileAdd(profile, io, out_start, .linear_out);
    profileAdd(profile, io, total_start, .linear_total);
    return result;
}

/// Refresh a conv window in place to the most recent `d_conv-1` token vectors of
/// `[old_state ++ qkv]`, so the next step's conv sees the right causal context.
/// `qkv` is `[seq][conv_dim]` row-major (the pre-conv projection); `state` is
/// `[(d_conv-1)·conv_dim]` row-major `[t][c]` (t=0 oldest).
fn updateConvState(state: []f32, qkv: []const f32, seq: usize, conv_dim: usize, d_conv: usize) void {
    const keep = d_conv - 1;
    if (seq >= keep) {
        @memcpy(state, qkv[(seq - keep) * conv_dim ..][0 .. keep * conv_dim]);
    } else {
        const shift = keep - seq; // slide the retained tail forward, then append qkv
        std.mem.copyForwards(f32, state[0 .. shift * conv_dim], state[seq * conv_dim ..][0 .. shift * conv_dim]);
        @memcpy(state[shift * conv_dim ..][0 .. seq * conv_dim], qkv[0 .. seq * conv_dim]);
    }
}

/// FFN sublayer: `input + DenseSiLU(post_attention_norm(input))`.
fn ffnForward(
    ctx: *ExecContext,
    cfg: Config,
    post_norm: *const fucina.Tensor(.{.embed}),
    ffn: *const DenseFfn,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    io: ?std.Io,
    profile: ?*ForwardProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    if (profile) |p| p.ffn_layers += 1;

    const norm_start = profileStart(profile, io);
    var ffn_in = try input.rmsNormMul(ctx, .embed, post_norm, cfg.rms_norm_eps);
    defer ffn_in.deinit();
    profileAdd(profile, io, norm_start, .ffn_norm);

    const ffn_rows = ffn_in.dim(.seq);
    if (ffn_rows >= 12) {
        switch (ffn.input_proj) {
            .fused => |*fused_weight| switch (fused_weight.*) {
                .q8_0 => |*gate_up_weight| switch (ffn.down) {
                    .q8_0 => |*down_weight| {
                        const gate_up_start = profileStart(profile, io);
                        var gate_up = try weights.linearSeqQ8_0(gate_up_weight, ctx, &ffn_in, .embed, .gate_up);
                        defer gate_up.deinit();
                        profileAdd(profile, io, gate_up_start, .ffn_gate_up);

                        const down_start = profileStart(profile, io);
                        var contribution = try gate_up.splitSwiGluDotPacked(ctx, &down_weight.packed_rhs, .gate_up, .embed);
                        defer contribution.deinit();
                        profileAdd(profile, io, down_start, .ffn_down);

                        const residual_start = profileStart(profile, io);
                        var result = try input.add(ctx, &contribution);
                        errdefer result.deinit();
                        profileAdd(profile, io, residual_start, .ffn_residual);
                        return result;
                    },
                    else => {},
                },
                else => {},
            },
            else => {},
        }
    }

    var gated = switch (ffn.input_proj) {
        .separate => |*separate| blk: {
            const gate_start = profileStart(profile, io);
            var gate = try separate.gate.linearSeq(ctx, &ffn_in, .embed, .ffn);
            defer gate.deinit();
            profileAdd(profile, io, gate_start, .ffn_gate);

            const up_start = profileStart(profile, io);
            var up = try separate.up.linearSeq(ctx, &ffn_in, .embed, .ffn);
            defer up.deinit();
            profileAdd(profile, io, up_start, .ffn_up);

            const act_start = profileStart(profile, io);
            const out = try up.swiglu(ctx, &gate);
            profileAdd(profile, io, act_start, .ffn_act);
            break :blk out;
        },
        .fused => |*weight| blk: {
            const gate_up_start = profileStart(profile, io);
            var gate_up = try weight.linearSeq(ctx, &ffn_in, .embed, .gate_up);
            defer gate_up.deinit();
            profileAdd(profile, io, gate_up_start, .ffn_gate_up);

            const act_start = profileStart(profile, io);
            const out = try gate_up.splitGated(ctx, .swiglu, .gate_up, .ffn);
            profileAdd(profile, io, act_start, .ffn_act);
            break :blk out;
        },
    };
    defer gated.deinit();

    const down_start = profileStart(profile, io);
    var contribution = try ffn.down.linearSeq(ctx, &gated, .ffn, .embed);
    defer contribution.deinit();
    profileAdd(profile, io, down_start, .ffn_down);

    const residual_start = profileStart(profile, io);
    var result = try input.add(ctx, &contribution);
    errdefer result.deinit();
    profileAdd(profile, io, residual_start, .ffn_residual);
    return result;
}

test "qwen35 Config.isRecurrent + derived dims (0.8B shape)" {
    const cfg = Config{
        .vocab_size = 248320,
        .hidden_size = 1024,
        .intermediate_size = 3584,
        .num_layers = 24,
        .num_attention_heads = 8,
        .num_key_value_heads = 2,
        .head_dim = 256,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1e7,
        .rope_n_rot = 64,
        .rope_sections = .{ 11, 11, 10, 0 },
        .full_attention_interval = 4,
        .ssm_d_conv = 4,
        .ssm_d_inner = 2048,
        .ssm_d_state = 128,
        .ssm_dt_rank = 16,
        .ssm_n_group = 16,
    };
    // Every 4th block (1-indexed) is full attention: il 3,7,11,15,19,23.
    const expect_attn = [_]usize{ 3, 7, 11, 15, 19, 23 };
    var n_attn: usize = 0;
    for (0..cfg.num_layers) |il| {
        const rec = cfg.isRecurrent(il);
        const is_attn = std.mem.indexOfScalar(usize, &expect_attn, il) != null;
        try std.testing.expectEqual(is_attn, !rec);
        if (!rec) n_attn += 1;
    }
    try std.testing.expectEqual(@as(usize, 6), n_attn);
    // DeltaNet derived dims.
    try std.testing.expectEqual(@as(usize, 128), cfg.headKDim());
    try std.testing.expectEqual(@as(usize, 128), cfg.headVDim());
    try std.testing.expectEqual(@as(usize, 16), cfg.numKHeads());
    try std.testing.expectEqual(@as(usize, 16), cfg.numVHeads());
    try std.testing.expectEqual(@as(usize, 6144), cfg.convDim());
    try std.testing.expectEqual(@as(usize, 2048), cfg.valueDim());
    // Full-attn projection widths.
    try std.testing.expectEqual(@as(usize, 4096), cfg.qGateDim());
    try std.testing.expectEqual(@as(usize, 512), cfg.kvDim());
    try std.testing.expectEqual(@as(usize, 2048), cfg.attnOutInDim());
    try cfg.validate();
}

test "partialRope rotates first n_rot dims and passes the rest through" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();

    const seq = 2;
    const head = 1;
    const d = 6;
    const n_rot = 4;
    var data: [seq * head * d]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast(i)) + 1);
    var x = try fucina.Tensor(.{ .seq, .head, .d }).fromSlice(&ctx, .{ seq, head, d }, &data);
    defer x.deinit();

    var positions = [_]i32{ 0, 1 };
    var table = try ctx.prepareRopeTable(&positions, n_rot, 10000.0, false);
    defer table.deinit();

    var out = try partialRope(&ctx, &x, n_rot, &table);
    defer out.deinit();
    const od = try out.dataConst();

    // The trailing [n_rot, d) dims must be untouched.
    for (0..seq) |s| {
        for (n_rot..d) |j| try std.testing.expectApproxEqAbs(data[s * d + j], od[s * d + j], 1e-6);
    }

    // The leading [0, n_rot) dims must match a full half-RoPE over just those dims.
    var sub: [seq * head * n_rot]f32 = undefined;
    for (0..seq) |s| {
        for (0..n_rot) |j| sub[s * n_rot + j] = data[s * d + j];
    }
    var xr = try fucina.Tensor(.{ .seq, .head, .d }).fromSlice(&ctx, .{ seq, head, n_rot }, &sub);
    defer xr.deinit();
    var ref = try xr.rope(&ctx, .seq, .d, &table, .half);
    defer ref.deinit();
    const rd = try ref.dataConst();
    for (0..seq) |s| {
        for (0..n_rot) |j| try std.testing.expectApproxEqAbs(rd[s * n_rot + j], od[s * d + j], 1e-5);
    }
}

test {
    _ = @import("model_tests.zig");
}

test "deltaNetChunked matches the per-token recurrence" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();
    const Sd = 8;
    const seq = 37; // intentionally not a multiple of any chunk size

    const k = try a.alloc(f32, seq * Sd);
    defer a.free(k);
    const v = try a.alloc(f32, seq * Sd);
    defer a.free(v);
    const q = try a.alloc(f32, seq * Sd);
    defer a.free(q);
    const lg = try a.alloc(f32, seq);
    defer a.free(lg);
    const beta = try a.alloc(f32, seq);
    defer a.free(beta);
    for (0..seq) |t| {
        var kn: f32 = 0;
        for (0..Sd) |i| {
            const x = rnd.float(f32) * 2 - 1;
            k[t * Sd + i] = x;
            kn += x * x;
        }
        kn = 1.0 / @sqrt(kn); // L2-normalize keys (as the layer does)
        for (0..Sd) |i| k[t * Sd + i] *= kn;
        for (0..Sd) |i| {
            v[t * Sd + i] = rnd.float(f32) * 2 - 1;
            q[t * Sd + i] = (rnd.float(f32) * 2 - 1) * 0.35; // pre-scaled query
        }
        lg[t] = -(rnd.float(f32) * 1.5 + 0.05); // log-decay (negative)
        beta[t] = rnd.float(f32);
    }

    // Reference: the exact per-token recurrence (same as runHeadScan core).
    const o_ref = try a.alloc(f32, seq * Sd);
    defer a.free(o_ref);
    const s_ref = try a.alloc(f32, Sd * Sd);
    defer a.free(s_ref);
    @memset(s_ref, 0);
    const u = try a.alloc(f32, Sd);
    defer a.free(u);
    for (0..seq) |t| {
        for (s_ref) |*s| s.* *= @exp(lg[t]);
        @memset(u, 0);
        for (0..Sd) |i| {
            const ki = k[t * Sd + i];
            for (0..Sd) |j| u[j] += s_ref[i * Sd + j] * ki;
        }
        for (0..Sd) |j| u[j] = (v[t * Sd + j] - u[j]) * beta[t];
        for (0..Sd) |i| {
            const ki = k[t * Sd + i];
            for (0..Sd) |j| s_ref[i * Sd + j] += ki * u[j];
        }
        @memset(o_ref[t * Sd ..][0..Sd], 0);
        for (0..Sd) |i| {
            const qi = q[t * Sd + i];
            for (0..Sd) |j| o_ref[t * Sd + j] += s_ref[i * Sd + j] * qi;
        }
    }

    for ([_]usize{ 4, 16, 64 }) |CS| {
        const o_c = try a.alloc(f32, seq * Sd);
        defer a.free(o_c);
        const s_c = try a.alloc(f32, Sd * Sd);
        defer a.free(s_c);
        @memset(s_c, 0);
        const delta = try a.alloc(f32, CS * Sd);
        defer a.free(delta);
        const gcs = try a.alloc(f32, CS);
        defer a.free(gcs);
        const tmp = try a.alloc(f32, Sd);
        defer a.free(tmp);
        deltaNetChunked(o_c, s_c, k, v, q, lg, beta, seq, Sd, CS, delta, gcs, tmp);
        for (0..seq * Sd) |i| try std.testing.expectApproxEqAbs(o_ref[i], o_c[i], 1e-3);
        for (0..Sd * Sd) |i| try std.testing.expectApproxEqAbs(s_ref[i], s_c[i], 1e-3);
    }
}

test "deltaNetScanBatched matches the per-token runHeadScan (gated output + carried state)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    var ctx: ExecContext = undefined;
    ctx.init(gpa.allocator());
    defer ctx.deinit();
    const a = ctx.allocator;

    var prng = std.Random.DefaultPrng.init(0xDE17A);
    const rnd = prng.random();

    // Dims chosen so the heavy chunk crosses the BLAS threshold (Sd, C ≥ 16) and
    // seq forces both a full chunk (C=64) and a short tail chunk (C=6).
    const H = 2;
    const Sd = 16;
    const seq = 70;
    const vd = H * Sd;
    const conv_dim = 3 * vd; // q | k | v, head-major
    const k_off = vd;
    const v_off = 2 * vd;
    const gate_stride = vd + 2 * H + 3; // z | alpha | beta | padding
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(Sd)));
    const eps: f32 = 1e-6;

    const cd = try a.alloc(f32, seq * conv_dim);
    defer a.free(cd);
    const gate = try a.alloc(f32, seq * gate_stride);
    defer a.free(gate);
    const zd = gate[0..];
    const ad = gate[vd..];
    const bd = gate[vd + H ..];
    for (cd) |*x| x.* = rnd.float(f32) * 2 - 1;
    @memset(gate, 1234.0);
    for (0..seq) |t| {
        for (0..vd) |i| gate[t * gate_stride + i] = rnd.float(f32) * 2 - 1;
        for (0..H) |i| gate[t * gate_stride + vd + i] = rnd.float(f32) * 2 - 1;
        for (0..H) |i| gate[t * gate_stride + vd + H + i] = rnd.float(f32) * 2 - 1;
    }

    const ssm_a = try a.alloc(f32, H);
    defer a.free(ssm_a);
    const ssm_dt = try a.alloc(f32, H);
    defer a.free(ssm_dt);
    const ssm_norm = try a.alloc(f32, Sd);
    defer a.free(ssm_norm);
    for (ssm_a) |*x| x.* = -(rnd.float(f32) + 0.25); // -A_log (negative)
    for (ssm_dt) |*x| x.* = rnd.float(f32) * 0.5;
    for (ssm_norm) |*x| x.* = rnd.float(f32) + 0.5;

    const hsz = Sd * Sd;

    // Reference: per-token recurrent scan (the shipping, llama-validated path).
    const out_ref = try a.alloc(f32, seq * vd);
    defer a.free(out_ref);
    const state_ref = try a.alloc(f32, H * hsz);
    defer a.free(state_ref);
    const uo = try a.alloc(f32, H * 2 * Sd);
    defer a.free(uo);
    for (0..H) |h| {
        var task = HeadScanTask{
            .h = h,
            .seq = seq,
            .Sd = Sd,
            .H = H,
            .conv_dim = conv_dim,
            .vd = vd,
            .k_off = k_off,
            .v_off = v_off,
            .scale = scale,
            .eps = eps,
            .cd = cd,
            .zd = zd,
            .z_stride = gate_stride,
            .ad = ad,
            .alpha_stride = gate_stride,
            .bd = bd,
            .beta_stride = gate_stride,
            .ssm_a = ssm_a,
            .ssm_dt = ssm_dt,
            .ssm_norm = ssm_norm,
            .out = out_ref,
            .state = state_ref[h * hsz ..][0..hsz],
            .uo = uo[h * 2 * Sd ..][0 .. 2 * Sd],
            .reset_state = true,
        };
        runHeadScan(&task);
    }

    // Batched chunked-GEMM scan over the same inputs.
    const out_b = try a.alloc(f32, seq * vd);
    defer a.free(out_b);
    const state_b = try a.alloc(f32, H * hsz);
    defer a.free(state_b);
    try deltaNetScanBatched(&ctx, .{
        .H = H,
        .Sd = Sd,
        .seq = seq,
        .conv_dim = conv_dim,
        .vd = vd,
        .k_off = k_off,
        .v_off = v_off,
        .scale = scale,
        .eps = eps,
        .cd = cd,
        .zd = zd,
        .z_stride = gate_stride,
        .ad = ad,
        .alpha_stride = gate_stride,
        .bd = bd,
        .beta_stride = gate_stride,
        .ssm_a = ssm_a,
        .ssm_dt = ssm_dt,
        .ssm_norm = ssm_norm,
        .out = out_b,
        .state = state_b,
        .reset_state = true,
    });

    for (0..seq * vd) |i| try std.testing.expectApproxEqAbs(out_ref[i], out_b[i], 2e-3);
    for (0..H * hsz) |i| try std.testing.expectApproxEqAbs(state_ref[i], state_b[i], 2e-3);
}

test "qwen35 rejects a q8_0 KV cache at the forward seam" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var q8 = try KvCache.initWithDtype(&ctx, 1, 2, 64, 4, .q8_0);
    defer q8.deinit();
    try std.testing.expectError(Error.UnsupportedKvCacheDtype, requireF16KvCache(&q8));

    var f16_cache = try KvCache.initWithDtype(&ctx, 1, 2, 64, 4, .f16);
    defer f16_cache.deinit();
    try requireF16KvCache(&f16_cache);
}
