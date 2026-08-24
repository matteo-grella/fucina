//! Kimi-K3 architecture (the Kimi-Linear lineage): a KDA linear-attention /
//! Gated-MLA-NoPE hybrid with a latent sigmoid-routed MoE, cross-layer
//! attention residuals, and the SiTU activation.
//!
//! This is the architecture-parity model over the tiny f32 reference
//! checkpoint (safetensors + config.json): every component reproduces the
//! reference equations and is pinned against reference activations by the
//! golden tests. Heavy lifting goes through the exec ops (`matmulTransB`,
//! `causalDepthwiseConv1d`, `kdaRecurrent`, `gated(.situ)`,
//! `rmsNormMul`); the depth-mixture and the tiny MLA core are
//! model-local routines. A serving-scale variant (GGUF weights, packed/
//! quant routes, fused attention, KV/state caches) builds on the same
//! layout when real K3-family checkpoints become a target.
//!
//! Reference structure per layer (attention residuals always on in K3):
//!   entry x → [depth-mix vs the block-residual bank] → rmsnorm → KDA|MLA →
//!   prefix accumulate → [depth-mix] → rmsnorm → dense-MLP|latent-MoE →
//!   prefix accumulate; layers at index % block_size == 0 snapshot their
//!   entry into the bank; the final hidden depth-mixes once more before the
//!   output norm and the untied lm head.

const std = @import("std");
const fucina = @import("fucina");
const safetensors = fucina.safetensors;

const Allocator = std.mem.Allocator;
pub const ExecContext = fucina.ExecContext;
const Tensor = fucina.internal.RawTensor;

pub const Config = struct {
    hidden_size: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    // KDA
    kda_head_dim: usize,
    kda_num_heads: usize,
    short_conv_kernel: usize,
    // MLA
    num_attention_heads: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    // MoE
    num_experts: usize,
    num_experts_per_token: usize,
    moe_intermediate_size: usize,
    routed_expert_hidden_size: usize,
    num_shared_experts: usize,
    routed_scaling_factor: f32,
    first_k_dense_replace: usize,
    intermediate_size: usize,
    attn_res_block_size: usize,
    /// 1-based KDA layer indices from linear_attn_config (the remainder are
    /// full-attention MLA layers).
    kda_layers: []usize,

    pub fn deinit(self: *Config, allocator: Allocator) void {
        allocator.free(self.kda_layers);
        self.* = undefined;
    }

    pub fn isKdaLayer(self: *const Config, layer_idx: usize) bool {
        for (self.kda_layers) |one_based| {
            if (one_based == layer_idx + 1) return true;
        }
        return false;
    }

    pub fn fromJsonFile(allocator: Allocator, io: std.Io, path: []const u8) !Config {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 20));
        defer allocator.free(bytes);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        const text = root.get("text_config").?.object;
        const linear = text.get("linear_attn_config").?.object;

        // The situ betas are checkpoint metadata; the GatedOp carries K3's
        // 4/25 form, so anything else must fail at load, not drift silently.
        if (asF32(text.get("activation_situ_beta").?) != 4.0 or
            asF32(text.get("activation_situ_linear_beta").?) != 25.0)
            return error.UnsupportedSituBetas;
        if (!std.mem.eql(u8, text.get("hidden_act").?.string, "situ"))
            return error.UnsupportedActivation;

        const kda_array = linear.get("kda_layers").?.array;
        const kda_layers = try allocator.alloc(usize, kda_array.items.len);
        errdefer allocator.free(kda_layers);
        for (kda_layers, kda_array.items) |*dst, item| dst.* = @intCast(item.integer);

        return .{
            .hidden_size = asUsize(text.get("hidden_size").?),
            .num_hidden_layers = asUsize(text.get("num_hidden_layers").?),
            .vocab_size = asUsize(text.get("vocab_size").?),
            .rms_norm_eps = asF32(text.get("rms_norm_eps").?),
            .kda_head_dim = asUsize(linear.get("head_dim").?),
            .kda_num_heads = asUsize(linear.get("num_heads").?),
            .short_conv_kernel = asUsize(linear.get("short_conv_kernel_size").?),
            .num_attention_heads = asUsize(text.get("num_attention_heads").?),
            .q_lora_rank = asUsize(text.get("q_lora_rank").?),
            .kv_lora_rank = asUsize(text.get("kv_lora_rank").?),
            .qk_nope_head_dim = asUsize(text.get("qk_nope_head_dim").?),
            .qk_rope_head_dim = asUsize(text.get("qk_rope_head_dim").?),
            .v_head_dim = asUsize(text.get("v_head_dim").?),
            .num_experts = asUsize(text.get("num_experts").?),
            .num_experts_per_token = asUsize(text.get("num_experts_per_token").?),
            .moe_intermediate_size = asUsize(text.get("moe_intermediate_size").?),
            .routed_expert_hidden_size = asUsize(text.get("routed_expert_hidden_size").?),
            .num_shared_experts = asUsize(text.get("num_shared_experts").?),
            .routed_scaling_factor = asF32(text.get("routed_scaling_factor").?),
            .first_k_dense_replace = asUsize(text.get("first_k_dense_replace").?),
            .intermediate_size = asUsize(text.get("intermediate_size").?),
            .attn_res_block_size = asUsize(text.get("attn_res_block_size").?),
            .kda_layers = kda_layers,
        };
    }

    fn asUsize(value: std.json.Value) usize {
        return @intCast(value.integer);
    }

    fn asF32(value: std.json.Value) f32 {
        return switch (value) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => unreachable,
        };
    }
};

const KdaWeights = struct {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    q_conv: Tensor,
    k_conv: Tensor,
    v_conv: Tensor,
    f_a: Tensor,
    f_b: Tensor,
    b_proj: Tensor,
    g_proj: Tensor,
    o_norm: Tensor,
    o_proj: Tensor,
    a_log: []f32,
    dt_bias: []f32,
};

const MlaWeights = struct {
    q_a: Tensor,
    q_a_norm: Tensor,
    q_b: Tensor,
    kv_a: Tensor,
    kv_a_norm: Tensor,
    kv_b: Tensor,
    g_proj: Tensor,
    o_proj: Tensor,
};

const DenseMlp = struct {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
};

const Expert = struct {
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
};

const MoeWeights = struct {
    gate_weight: Tensor,
    e_score_bias: []f32,
    experts: []Expert,
    down_proj: Tensor,
    up_proj: Tensor,
    latent_norm: Tensor,
    shared: DenseMlp,
};

const Attn = union(enum) {
    kda: KdaWeights,
    mla: MlaWeights,
};

const Ffn = union(enum) {
    dense: DenseMlp,
    moe: MoeWeights,
};

const Layer = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    attn: Attn,
    ffn: Ffn,
    sa_res_norm: []f32,
    sa_res_proj: []f32,
    mlp_res_norm: []f32,
    mlp_res_proj: []f32,
};

pub const Model = struct {
    allocator: Allocator,
    config: Config,
    embed: []f32,
    lm_head: Tensor,
    final_norm: Tensor,
    out_res_norm: []f32,
    out_res_proj: []f32,
    layers: []Layer,

    pub fn deinit(self: *Model) void {
        const a = self.allocator;
        for (self.layers) |*layer| {
            layer.input_norm.deinit();
            layer.post_norm.deinit();
            switch (layer.attn) {
                .kda => |*w| {
                    inline for (.{ &w.q_proj, &w.k_proj, &w.v_proj, &w.q_conv, &w.k_conv, &w.v_conv, &w.f_a, &w.f_b, &w.b_proj, &w.g_proj, &w.o_norm, &w.o_proj }) |t| t.deinit();
                    a.free(w.a_log);
                    a.free(w.dt_bias);
                },
                .mla => |*w| {
                    inline for (.{ &w.q_a, &w.q_a_norm, &w.q_b, &w.kv_a, &w.kv_a_norm, &w.kv_b, &w.g_proj, &w.o_proj }) |t| t.deinit();
                },
            }
            switch (layer.ffn) {
                .dense => |*w| {
                    inline for (.{ &w.gate, &w.up, &w.down }) |t| t.deinit();
                },
                .moe => |*w| {
                    w.gate_weight.deinit();
                    a.free(w.e_score_bias);
                    for (w.experts) |*e| {
                        inline for (.{ &e.w1, &e.w2, &e.w3 }) |t| t.deinit();
                    }
                    a.free(w.experts);
                    w.down_proj.deinit();
                    w.up_proj.deinit();
                    w.latent_norm.deinit();
                    inline for (.{ &w.shared.gate, &w.shared.up, &w.shared.down }) |t| t.deinit();
                },
            }
            a.free(layer.sa_res_norm);
            a.free(layer.sa_res_proj);
            a.free(layer.mlp_res_norm);
            a.free(layer.mlp_res_proj);
        }
        a.free(self.layers);
        a.free(self.embed);
        self.lm_head.deinit();
        self.final_norm.deinit();
        a.free(self.out_res_norm);
        a.free(self.out_res_proj);
        self.config.deinit(a);
        self.* = undefined;
    }

    pub fn load(allocator: Allocator, io: std.Io, ctx: *ExecContext, dir: []const u8) !Model {
        var path_buf: [512]u8 = undefined;
        var config = try Config.fromJsonFile(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/config.json", .{dir}));
        errdefer config.deinit(allocator);

        var file = try safetensors.File.loadMmap(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/model.safetensors", .{dir}));
        defer file.deinit();

        var loader = Loader{ .allocator = allocator, .ctx = ctx, .file = &file };

        const layers = try allocator.alloc(Layer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        var name_buf: [256]u8 = undefined;
        for (layers, 0..) |*layer, i| {
            const prefix = "language_model.model.layers.";
            layer.input_norm = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.input_layernorm.weight", .{ prefix, i }));
            layer.post_norm = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.post_attention_layernorm.weight", .{ prefix, i }));
            layer.sa_res_norm = try loader.slice(try std.fmt.bufPrint(&name_buf, "{s}{d}.self_attention_res_norm.weight", .{ prefix, i }));
            layer.sa_res_proj = try loader.slice(try std.fmt.bufPrint(&name_buf, "{s}{d}.self_attention_res_proj.weight", .{ prefix, i }));
            layer.mlp_res_norm = try loader.slice(try std.fmt.bufPrint(&name_buf, "{s}{d}.mlp_res_norm.weight", .{ prefix, i }));
            layer.mlp_res_proj = try loader.slice(try std.fmt.bufPrint(&name_buf, "{s}{d}.mlp_res_proj.weight", .{ prefix, i }));

            if (config.isKdaLayer(i)) {
                layer.attn = .{ .kda = .{
                    .q_proj = try loader.attn(&name_buf, i, "q_proj.weight"),
                    .k_proj = try loader.attn(&name_buf, i, "k_proj.weight"),
                    .v_proj = try loader.attn(&name_buf, i, "v_proj.weight"),
                    .q_conv = try loader.conv(&name_buf, i, "q_conv1d.weight"),
                    .k_conv = try loader.conv(&name_buf, i, "k_conv1d.weight"),
                    .v_conv = try loader.conv(&name_buf, i, "v_conv1d.weight"),
                    .f_a = try loader.attn(&name_buf, i, "f_a_proj.weight"),
                    .f_b = try loader.attn(&name_buf, i, "f_b_proj.weight"),
                    .b_proj = try loader.attn(&name_buf, i, "b_proj.weight"),
                    .g_proj = try loader.attn(&name_buf, i, "g_proj.weight"),
                    .o_norm = try loader.attn(&name_buf, i, "o_norm.weight"),
                    .o_proj = try loader.attn(&name_buf, i, "o_proj.weight"),
                    .a_log = try loader.slice(try std.fmt.bufPrint(&name_buf, "{s}{d}.self_attn.A_log", .{ prefix, i })),
                    .dt_bias = try loader.slice(try std.fmt.bufPrint(&name_buf, "{s}{d}.self_attn.dt_bias", .{ prefix, i })),
                } };
            } else {
                layer.attn = .{ .mla = .{
                    .q_a = try loader.attn(&name_buf, i, "q_a_proj.weight"),
                    .q_a_norm = try loader.attn(&name_buf, i, "q_a_layernorm.weight"),
                    .q_b = try loader.attn(&name_buf, i, "q_b_proj.weight"),
                    .kv_a = try loader.attn(&name_buf, i, "kv_a_proj_with_mqa.weight"),
                    .kv_a_norm = try loader.attn(&name_buf, i, "kv_a_layernorm.weight"),
                    .kv_b = try loader.attn(&name_buf, i, "kv_b_proj.weight"),
                    .g_proj = try loader.attn(&name_buf, i, "g_proj.weight"),
                    .o_proj = try loader.attn(&name_buf, i, "o_proj.weight"),
                } };
            }

            if (i < config.first_k_dense_replace) {
                layer.ffn = .{ .dense = .{
                    .gate = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.mlp.gate_proj.weight", .{ prefix, i })),
                    .up = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.mlp.up_proj.weight", .{ prefix, i })),
                    .down = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.mlp.down_proj.weight", .{ prefix, i })),
                } };
            } else {
                const experts = try allocator.alloc(Expert, config.num_experts);
                for (experts, 0..) |*expert, e| {
                    expert.w1 = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.block_sparse_moe.experts.{d}.w1.weight", .{ prefix, i, e }));
                    expert.w2 = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.block_sparse_moe.experts.{d}.w2.weight", .{ prefix, i, e }));
                    expert.w3 = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.block_sparse_moe.experts.{d}.w3.weight", .{ prefix, i, e }));
                }
                layer.ffn = .{ .moe = .{
                    .gate_weight = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.block_sparse_moe.gate.weight", .{ prefix, i })),
                    .e_score_bias = try loader.slice(try std.fmt.bufPrint(&name_buf, "{s}{d}.block_sparse_moe.gate.e_score_correction_bias", .{ prefix, i })),
                    .experts = experts,
                    .down_proj = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.block_sparse_moe.routed_expert_down_proj.weight", .{ prefix, i })),
                    .up_proj = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.block_sparse_moe.routed_expert_up_proj.weight", .{ prefix, i })),
                    .latent_norm = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.block_sparse_moe.routed_expert_norm.weight", .{ prefix, i })),
                    .shared = .{
                        .gate = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.block_sparse_moe.shared_experts.gate_proj.weight", .{ prefix, i })),
                        .up = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.block_sparse_moe.shared_experts.up_proj.weight", .{ prefix, i })),
                        .down = try loader.tensor(try std.fmt.bufPrint(&name_buf, "{s}{d}.block_sparse_moe.shared_experts.down_proj.weight", .{ prefix, i })),
                    },
                } };
            }
        }

        return .{
            .allocator = allocator,
            .config = config,
            .embed = try loader.slice("language_model.model.embed_tokens.weight"),
            .lm_head = try loader.tensor("language_model.lm_head.weight"),
            .final_norm = try loader.tensor("language_model.model.norm.weight"),
            .out_res_norm = try loader.slice("language_model.model.output_attn_res_norm.weight"),
            .out_res_proj = try loader.slice("language_model.model.output_attn_res_proj.weight"),
            .layers = layers,
        };
    }

    const Loader = struct {
        allocator: Allocator,
        ctx: *ExecContext,
        file: *const safetensors.File,

        fn slice(self: *Loader, name: []const u8) ![]f32 {
            const info = try self.file.tensor(name);
            if (info.dtype != .F32) return error.UnsupportedDtype;
            const out = try self.allocator.alloc(f32, info.data.len / 4);
            @memcpy(std.mem.sliceAsBytes(out), info.data);
            return out;
        }

        fn tensor(self: *Loader, name: []const u8) !Tensor {
            const values = try self.slice(name);
            defer self.allocator.free(values);
            const info = try self.file.tensor(name);
            return self.ctx.fromSlice(.f32, info.shape, values);
        }

        fn attn(self: *Loader, buf: []u8, layer_idx: usize, comptime leaf: []const u8) !Tensor {
            return self.tensor(try std.fmt.bufPrint(buf, "language_model.model.layers.{d}.self_attn." ++ leaf, .{layer_idx}));
        }

        fn conv(self: *Loader, buf: []u8, layer_idx: usize, comptime leaf: []const u8) !Tensor {
            // ShortConvolution stores depthwise [D, 1, W]; the causal conv op
            // takes [channel, tap] with tap W-1 the newest sample — the same
            // orientation, so squeezing the middle axis is the whole map.
            const name = try std.fmt.bufPrint(buf, "language_model.model.layers.{d}.self_attn." ++ leaf, .{layer_idx});
            const values = try self.slice(name);
            defer self.allocator.free(values);
            const info = try self.file.tensor(name);
            return self.ctx.fromSlice(.f32, &.{ info.shape[0], info.shape[2] }, values);
        }
    };

    /// Stage observer for parity work: receives the same intermediates the
    /// reference dumps (per-layer normed input and attention output).
    pub const Probe = struct {
        context: *anyopaque,
        callback: *const fn (context: *anyopaque, name: []const u8, layer_idx: usize, values: []const f32) void,

        fn emit(self: *const Probe, name: []const u8, layer_idx: usize, values: []const f32) void {
            self.callback(self.context, name, layer_idx, values);
        }
    };

    /// Full-sequence forward: token ids -> [seq, vocab] logits.
    pub fn forward(self: *const Model, ctx: *ExecContext, tokens: []const u32) !Tensor {
        return self.forwardProbed(ctx, tokens, null);
    }

    pub fn forwardProbed(self: *const Model, ctx: *ExecContext, tokens: []const u32, probe: ?*const Probe) !Tensor {
        const cfg = &self.config;
        const d = cfg.hidden_size;
        const seq = tokens.len;
        const allocator = self.allocator;

        // Embedding rows, gathered straight into the residual tensor.
        var x = try ctx.empty(.f32, &.{ seq, d });
        {
            const x_values = x.data();
            for (tokens, 0..) |token, t| {
                @memcpy(x_values[t * d ..][0..d], self.embed[token * d ..][0..d]);
            }
        }

        // The block-residual bank: entry snapshots of block-boundary layers.
        var bank: std.ArrayList([]f32) = .empty;
        defer {
            for (bank.items) |snapshot| allocator.free(snapshot);
            bank.deinit(allocator);
        }

        for (self.layers, 0..) |*layer, layer_idx| {
            // Entry snapshot (prefix_sum in the reference).
            const entry = try allocator.dupe(f32, x.dataConst());
            var entry_owned = true;
            defer if (entry_owned) allocator.free(entry);

            // `x` stays the pre-mix entry (the reference's prefix_sum);
            // `hidden` is the depth-mixed view feeding the attention.
            var hidden = if (bank.items.len > 0)
                try self.applyAttnRes(ctx, &x, bank.items, layer.sa_res_norm, layer.sa_res_proj)
            else
                try x.cloneView();
            defer hidden.deinit();

            var boundary = false;
            if (layer_idx % cfg.attn_res_block_size == 0) {
                try bank.append(allocator, entry);
                entry_owned = false;
                boundary = true;
            }

            var normed = try ctx.rmsNorm(.f32, 2, &hidden, 1, cfg.rms_norm_eps, .{ .weight = &layer.input_norm });
            defer normed.deinit();
            if (probe) |pr| pr.emit("input_layernorm", layer_idx, normed.dataConst());
            var attn_out = switch (layer.attn) {
                .kda => |*w| try self.kdaAttention(ctx, &normed, w),
                .mla => |*w| try self.mlaAttention(ctx, &normed, w),
            };
            defer attn_out.deinit();
            if (probe) |pr| pr.emit("self_attn", layer_idx, attn_out.dataConst());

            // prefix = pre-mix entry + attn (attn alone on a boundary layer,
            // whose entry moved into the bank).
            var prefix = if (boundary)
                try attn_out.cloneView()
            else
                try ctx.elementwise(.f32, .add, &x, &attn_out);
            errdefer prefix.deinit();
            x.deinit();

            var mixed2 = try self.applyAttnRes(ctx, &prefix, bank.items, layer.mlp_res_norm, layer.mlp_res_proj);
            defer mixed2.deinit();
            var normed2 = try ctx.rmsNorm(.f32, 2, &mixed2, 1, cfg.rms_norm_eps, .{ .weight = &layer.post_norm });
            defer normed2.deinit();
            var ffn_out = switch (layer.ffn) {
                .dense => |*w| try self.denseMlp(ctx, &normed2, w),
                .moe => |*w| try self.latentMoe(ctx, &normed2, w),
            };
            defer ffn_out.deinit();

            x = try ctx.elementwise(.f32, .add, &prefix, &ffn_out);
            prefix.deinit();
        }

        var final_mix = try self.applyAttnRes(ctx, &x, bank.items, self.out_res_norm, self.out_res_proj);
        x.deinit();
        defer final_mix.deinit();
        var final_normed = try ctx.rmsNorm(.f32, 2, &final_mix, 1, cfg.rms_norm_eps, .{ .weight = &self.final_norm });
        defer final_normed.deinit();
        return ctx.matmul(.f32, .trans_b, &final_normed, &self.lm_head);
    }

    /// The cross-layer attention-residual mixture: per token, softmax over
    /// {bank snapshots..., current} scored by rmsnorm(v)·(norm_w ⊙ proj_w),
    /// mixing the RAW candidates.
    fn applyAttnRes(self: *const Model, ctx: *ExecContext, current: *const Tensor, bank: []const []f32, norm_w: []const f32, proj_w: []const f32) !Tensor {
        const d = self.config.hidden_size;
        const eps = self.config.rms_norm_eps;
        const cur = current.dataConst();
        const seq = cur.len / d;
        const candidates = bank.len + 1;

        var out = try ctx.empty(.f32, &.{ seq, d });
        errdefer out.deinit();
        const out_values = out.data();

        var scores_buf: [64]f32 = undefined;
        const scores = scores_buf[0..candidates];
        for (0..seq) |t| {
            for (0..candidates) |c| {
                const row = if (c < bank.len) bank[c][t * d ..][0..d] else cur[t * d ..][0..d];
                var sumsq: f64 = 0;
                for (row) |value| sumsq += @as(f64, value) * value;
                const inv = 1.0 / @sqrt(@as(f32, @floatCast(sumsq / @as(f64, @floatFromInt(d)))) + eps);
                var score: f32 = 0;
                for (row, norm_w, proj_w) |value, nw, pw| score += value * inv * nw * pw;
                scores[c] = score;
            }
            var max_score: f32 = -std.math.inf(f32);
            for (scores) |s| max_score = @max(max_score, s);
            var denom: f32 = 0;
            for (scores) |*s| {
                s.* = @exp(s.* - max_score);
                denom += s.*;
            }
            const out_row = out_values[t * d ..][0..d];
            @memset(out_row, 0);
            for (0..candidates) |c| {
                const row = if (c < bank.len) bank[c][t * d ..][0..d] else cur[t * d ..][0..d];
                const p = scores[c] / denom;
                for (out_row, row) |*dst, value| dst.* += p * value;
            }
        }
        return out;
    }

    fn kdaAttention(self: *const Model, ctx: *ExecContext, h: *const Tensor, w: *const KdaWeights) !Tensor {
        const cfg = &self.config;
        const heads = cfg.kda_num_heads;
        const head_dim = cfg.kda_head_dim;
        const seq = h.dataConst().len / cfg.hidden_size;

        var q = try self.convProj(ctx, h, &w.q_proj, &w.q_conv);
        defer q.deinit();
        var k = try self.convProj(ctx, h, &w.k_proj, &w.k_conv);
        defer k.deinit();
        var v = try self.convProj(ctx, h, &w.v_proj, &w.v_conv);
        defer v.deinit();

        var f_low = try ctx.matmul(.f32, .trans_b, h, &w.f_a);
        defer f_low.deinit();
        var g_raw = try ctx.matmul(.f32, .trans_b, &f_low, &w.f_b);
        defer g_raw.deinit();
        var beta_raw = try ctx.matmul(.f32, .trans_b, h, &w.b_proj);
        defer beta_raw.deinit();

        var q3 = try q.reshape(&.{ seq, heads, head_dim });
        defer q3.deinit();
        var k3 = try k.reshape(&.{ seq, heads, head_dim });
        defer k3.deinit();
        var v3 = try v.reshape(&.{ seq, heads, head_dim });
        defer v3.deinit();
        var g3 = try g_raw.reshape(&.{ seq, heads, head_dim });
        defer g3.deinit();

        var result = try ctx.kdaRecurrent(&q3, &k3, &v3, &g3, &beta_raw, w.a_log, w.dt_bias, null, 0);
        defer result.deinit();

        // o_norm: per-head RMSNorm(o)·weight × sigmoid(full-rank gate).
        var o_normed = try ctx.rmsNorm(.f32, 3, &result.o, 2, cfg.rms_norm_eps, .{ .weight = &w.o_norm });
        defer o_normed.deinit();
        var gate = try ctx.matmul(.f32, .trans_b, h, &w.g_proj);
        defer gate.deinit();
        var gate3 = try gate.reshape(&.{ seq, heads, head_dim });
        defer gate3.deinit();
        var gate_sig = try ctx.unary(.f32, .sigmoid, &gate3);
        defer gate_sig.deinit();
        var gated = try ctx.elementwise(.f32, .mul, &o_normed, &gate_sig);
        defer gated.deinit();
        var flat = try gated.reshape(&.{ seq, heads * head_dim });
        defer flat.deinit();
        return ctx.matmul(.f32, .trans_b, &flat, &w.o_proj);
    }

    fn convProj(self: *const Model, ctx: *ExecContext, h: *const Tensor, proj: *const Tensor, conv: *const Tensor) !Tensor {
        _ = self;
        var projected = try ctx.matmul(.f32, .trans_b, h, proj);
        defer projected.deinit();
        var convolved = try ctx.causalDepthwiseConv1d(2, &projected, conv, 0, 1, 1, null);
        defer convolved.deinit();
        return ctx.unary(.f32, .silu, &convolved);
    }

    /// Gated MLA without positional encoding: plain causal softmax attention
    /// over [nope|rope-slot] q/k (the rope slot never rotates — NoPE), with
    /// the shared-across-heads k_rot from the compressed kv projection, a
    /// sigmoid output gate, and the o projection. Model-local core: the
    /// parity checkpoint is tiny; serving scale adopts the fused attention.
    fn mlaAttention(self: *const Model, ctx: *ExecContext, h: *const Tensor, w: *const MlaWeights) !Tensor {
        const cfg = &self.config;
        const heads = cfg.num_attention_heads;
        const nope = cfg.qk_nope_head_dim;
        const rope = cfg.qk_rope_head_dim;
        const v_dim = cfg.v_head_dim;
        const q_dim = nope + rope;
        const kv_lora = cfg.kv_lora_rank;
        const seq = h.dataConst().len / cfg.hidden_size;
        const allocator = self.allocator;

        var q_low = try ctx.matmul(.f32, .trans_b, h, &w.q_a);
        defer q_low.deinit();
        var q_low_n = try ctx.rmsNorm(.f32, 2, &q_low, 1, cfg.rms_norm_eps, .{ .weight = &w.q_a_norm });
        defer q_low_n.deinit();
        var q_full = try ctx.matmul(.f32, .trans_b, &q_low_n, &w.q_b);
        defer q_full.deinit();

        var ckv = try ctx.matmul(.f32, .trans_b, h, &w.kv_a);
        defer ckv.deinit();
        const ckv_data = ckv.dataConst();

        // Split compressed kv into the lora half (normalized, expanded by
        // kv_b) and the shared rope-slot half (used raw — NoPE).
        var kv_low = try ctx.empty(.f32, &.{ seq, kv_lora });
        defer kv_low.deinit();
        {
            const kv_low_values = kv_low.data();
            for (0..seq) |t| {
                @memcpy(kv_low_values[t * kv_lora ..][0..kv_lora], ckv_data[t * (kv_lora + rope) ..][0..kv_lora]);
            }
        }
        var kv_low_n = try ctx.rmsNorm(.f32, 2, &kv_low, 1, cfg.rms_norm_eps, .{ .weight = &w.kv_a_norm });
        defer kv_low_n.deinit();
        var kv_full = try ctx.matmul(.f32, .trans_b, &kv_low_n, &w.kv_b);
        defer kv_full.deinit();

        // Attention per head, eager causal softmax, scale q_dim^-1/2.
        const q_data = q_full.dataConst(); // [seq, heads*(nope+rope)]
        const kv_data = kv_full.dataConst(); // [seq, heads*(nope+v)]
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(q_dim)));

        var attn = try ctx.empty(.f32, &.{ seq, heads * v_dim });
        defer attn.deinit();
        const attn_values = attn.data();
        const row_scores = try allocator.alloc(f32, seq);
        defer allocator.free(row_scores);

        for (0..heads) |head| {
            for (0..seq) |t| {
                const q_row = q_data[t * heads * q_dim + head * q_dim ..][0..q_dim];
                const q_rot = q_row[nope..];
                for (0..t + 1) |s| {
                    const k_nope = kv_data[s * heads * (nope + v_dim) + head * (nope + v_dim) ..][0..nope];
                    const k_rot = ckv_data[s * (kv_lora + rope) + kv_lora ..][0..rope];
                    var dot: f32 = 0;
                    for (q_row[0..nope], k_nope) |qv, kv| dot += qv * kv;
                    for (q_rot, k_rot) |qv, kv| dot += qv * kv;
                    row_scores[s] = dot * scale;
                }
                var max_score: f32 = -std.math.inf(f32);
                for (row_scores[0 .. t + 1]) |s| max_score = @max(max_score, s);
                var denom: f32 = 0;
                for (row_scores[0 .. t + 1]) |*s| {
                    s.* = @exp(s.* - max_score);
                    denom += s.*;
                }
                const out_row = attn_values[t * heads * v_dim + head * v_dim ..][0..v_dim];
                @memset(out_row, 0);
                for (0..t + 1) |s| {
                    const value_row = kv_data[s * heads * (nope + v_dim) + head * (nope + v_dim) + nope ..][0..v_dim];
                    const p = row_scores[s] / denom;
                    for (out_row, value_row) |*dst, vv| dst.* += p * vv;
                }
            }
        }

        var gate = try ctx.matmul(.f32, .trans_b, h, &w.g_proj);
        defer gate.deinit();
        var gate_sig = try ctx.unary(.f32, .sigmoid, &gate);
        defer gate_sig.deinit();
        var gated = try ctx.elementwise(.f32, .mul, &attn, &gate_sig);
        defer gated.deinit();
        return ctx.matmul(.f32, .trans_b, &gated, &w.o_proj);
    }

    fn denseMlp(self: *const Model, ctx: *ExecContext, h: *const Tensor, w: *const DenseMlp) !Tensor {
        _ = self;
        var gate = try ctx.matmul(.f32, .trans_b, h, &w.gate);
        defer gate.deinit();
        var up = try ctx.matmul(.f32, .trans_b, h, &w.up);
        defer up.deinit();
        var act = try ctx.gated(.f32, 2, .situ, &up, &gate);
        defer act.deinit();
        return ctx.matmul(.f32, .trans_b, &act, &w.down);
    }

    /// DeepSeek-V3-style sigmoid router (bias-corrected choice, uncorrected
    /// weights, renormalized) over experts running at the latent width.
    fn latentMoe(self: *const Model, ctx: *ExecContext, h: *const Tensor, w: *const MoeWeights) !Tensor {
        const cfg = &self.config;
        const top_k = cfg.num_experts_per_token;
        const experts = cfg.num_experts;
        const latent = cfg.routed_expert_hidden_size;
        const seq = h.dataConst().len / cfg.hidden_size;
        const allocator = self.allocator;

        var logits = try ctx.matmul(.f32, .trans_b, h, &w.gate_weight);
        defer logits.deinit();
        const logits_data = logits.dataConst();

        var lat = try ctx.matmul(.f32, .trans_b, h, &w.down_proj);
        defer lat.deinit();

        // Every expert over the full latent sequence (8 tiny matmuls beat
        // per-token gathers at parity scale), then the routed weighted sum.
        const expert_outs = try allocator.alloc(Tensor, experts);
        // Free only the experts the loop below actually built: a mid-loop
        // failure leaves the tail undefined.
        var built: usize = 0;
        defer {
            for (expert_outs[0..built]) |*t| t.deinit();
            allocator.free(expert_outs);
        }
        for (expert_outs, w.experts) |*out, *expert| {
            var gate = try ctx.matmul(.f32, .trans_b, &lat, &expert.w1);
            defer gate.deinit();
            var up = try ctx.matmul(.f32, .trans_b, &lat, &expert.w3);
            defer up.deinit();
            var act = try ctx.gated(.f32, 2, .situ, &up, &gate);
            defer act.deinit();
            out.* = try ctx.matmul(.f32, .trans_b, &act, &expert.w2);
            built += 1;
        }

        var routed = try ctx.empty(.f32, &.{ seq, latent });
        defer routed.deinit();
        const routed_values = routed.data();
        @memset(routed_values, 0);
        var top_idx_buf: [64]usize = undefined;
        const top_idx = top_idx_buf[0..top_k];
        for (0..seq) |t| {
            const row = logits_data[t * experts ..][0..experts];
            var chosen: usize = 0;
            while (chosen < top_k) : (chosen += 1) {
                var best: usize = std.math.maxInt(usize);
                var best_score: f32 = -std.math.inf(f32);
                expert_loop: for (0..experts) |e| {
                    for (top_idx[0..chosen]) |used| {
                        if (used == e) continue :expert_loop;
                    }
                    const score = 1.0 / (1.0 + @exp(-row[e])) + w.e_score_bias[e];
                    if (score > best_score) {
                        best_score = score;
                        best = e;
                    }
                }
                top_idx[chosen] = best;
            }
            var denom: f32 = 1e-20;
            for (top_idx) |e| denom += 1.0 / (1.0 + @exp(-row[e]));
            for (top_idx) |e| {
                const weight = (1.0 / (1.0 + @exp(-row[e]))) / denom * cfg.routed_scaling_factor;
                const expert_row = expert_outs[e].dataConst()[t * latent ..][0..latent];
                const dst = routed_values[t * latent ..][0..latent];
                for (dst, expert_row) |*acc, value| acc.* += weight * value;
            }
        }

        var routed_n = try ctx.rmsNorm(.f32, 2, &routed, 1, cfg.rms_norm_eps, .{ .weight = &w.latent_norm });
        defer routed_n.deinit();
        var up_out = try ctx.matmul(.f32, .trans_b, &routed_n, &w.up_proj);
        defer up_out.deinit();

        var shared_out = try self.denseMlp(ctx, h, &w.shared);
        defer shared_out.deinit();
        return ctx.elementwise(.f32, .add, &up_out, &shared_out);
    }
};

test {
    _ = @import("model_tests.zig");
}
