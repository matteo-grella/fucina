//! GLM-4.5 family (`glm4moe` GGUF arch): GQA attention with QKV biases and
//! partial interleaved rotary, DeepSeek-V3-style MoE (sigmoid noaux routing
//! with a selection bias, renormalized weights, one shared expert, one
//! leading dense layer), and — the reason this family is here — a native
//! MTP (multi-token-prediction) `nextn` layer for lossless self-speculative
//! decoding.
//!
//! The trunk runs on the descriptor runner's host_reference band
//! (`llm/runner.zig`: `HostLayer` blocks, `HostRope`, `HostCache`, host
//! attention/routing in auditable f32, fucina kernels for the heavy linears
//! and the fused/streamed MoE). This module keeps what is family-specific:
//! the MTP head and draft step, and the `step` verify API — S >= 1
//! positions processed causally in one call, per-position logits returned,
//! which is what the MTP verify needs.
const std = @import("std");
const fucina = @import("fucina");
const weights = @import("fucina").weights;
const gguf_meta = @import("fucina").gguf_meta;
const runner = @import("../runner.zig");
const host_ops = @import("../host_ops.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;
const LinearWeight = weights.LinearWeight;

pub const Error = runner.Error;

/// The trunk configuration IS the runner descriptor (`fromGguf` fills the
/// host_reference shape from glm4moe metadata; `num_layers` counts the
/// trunk only, the `nextn` MTP layer is loaded separately by this module).
pub const Config = runner.Descriptor;

/// Host K/V cache (`truncate` is the speculative rewind).
pub const Cache = runner.HostCache;

pub const mtp_debug: bool = false;

fn maxAbs(v: []const f32) f32 {
    var m: f32 = 0;
    for (v) |x| m = @max(m, @abs(x));
    return m;
}

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
    layer: runner.HostLayer,

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

pub const Model = struct {
    allocator: Allocator,
    config: Config,
    /// MTP (`nextn`) layer count from the GGUF metadata; `config.num_layers`
    /// is the trunk only.
    nextn: usize,
    token_embedding: LinearWeight,
    output: LinearWeight,
    /// Trunk layers, rope table, output norm, attention scale — the
    /// runner's host_reference state, driven through its block code.
    band: runner.HostBand,
    mtp: ?MtpHead,
    weight_mapping: ?gguf.File.MappedRegion = null,
    expert_store: ?*fucina.ExpertStore = null,
    /// Trunk hidden state (pre-output-norm) of the LAST position of the
    /// most recent `step` — the MTP draft's h_prev seed.
    last_hidden: []f32,
    /// All pre-norm hiddens of the most recent `step` ([S, hidden], model-
    /// owned, valid until the next step) — the MTP stream consumes one per
    /// committed position.
    step_hiddens: []f32 = &.{},

    pub const MoeStreamOptions = weights.MoeStreamOptions;

    pub const LoadOptions = struct {
        moe_stream: ?MoeStreamOptions = null,
    };

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, max_positions: usize, options: LoadOptions) !Model {
        var file = try gguf.File.loadMmapAuto(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFileOptions(ctx, &file, max_positions, options);
    }

    pub fn loadGgufFromFileOptions(ctx: *ExecContext, file: *gguf.File, max_positions: usize, options: LoadOptions) !Model {
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        if (!std.mem.eql(u8, arch, "glm4moe")) return Error.InvalidConfig;
        const config = try Config.fromGguf(file);
        const nextn = gguf_meta.metaIntOpt(file, "glm4moe", "nextn_predict_layers", .accept_zero) orelse 0;
        const allocator = ctx.allocator;

        var expert_store: ?*fucina.ExpertStore = null;
        if (options.moe_stream) |stream_options| {
            // block indices include the nextn layer (blk.46).
            if (config.num_experts > 0) expert_store = try weights.createExpertStore(allocator, stream_options, config.num_layers + nextn);
        }
        errdefer if (expert_store) |store| store.destroy();

        var token_embedding = try LinearWeight.load(ctx, try file.get("token_embd.weight"), config.vocab_size, config.hidden_size);
        errdefer token_embedding.deinit();
        var output = try LinearWeight.load(ctx, try file.get("output.weight"), config.vocab_size, config.hidden_size);
        errdefer output.deinit();
        const output_norm = try weights.hostVector(allocator, file, "output_norm.weight", config.hidden_size);
        errdefer allocator.free(output_norm);

        const layers = try allocator.alloc(runner.HostLayer, config.num_layers);
        errdefer allocator.free(layers);
        var built: usize = 0;
        errdefer for (layers[0..built]) |*l| l.deinit(allocator);
        for (layers, 0..) |*layer, i| {
            layer.* = try runner.loadHostLayer(ctx, file, config, i, expert_store);
            built += 1;
        }

        var mtp: ?MtpHead = null;
        errdefer if (mtp) |*m| m.deinit(allocator);
        if (nextn > 0) {
            const mtp_i = config.num_layers; // blk.46 on Air
            var name_buf: [96]u8 = undefined;
            var embed = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, mtp_i, "nextn.embed_tokens.weight")), config.vocab_size, config.hidden_size);
            errdefer embed.deinit();
            var eh_proj = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, mtp_i, "nextn.eh_proj.weight")), config.hidden_size, 2 * config.hidden_size);
            errdefer eh_proj.deinit();
            const enorm = try weights.hostVector(allocator, file, try weights.layerName(&name_buf, mtp_i, "nextn.enorm.weight"), config.hidden_size);
            errdefer allocator.free(enorm);
            const hnorm = try weights.hostVector(allocator, file, try weights.layerName(&name_buf, mtp_i, "nextn.hnorm.weight"), config.hidden_size);
            errdefer allocator.free(hnorm);
            const sh_norm = try weights.hostVector(allocator, file, try weights.layerName(&name_buf, mtp_i, "nextn.shared_head_norm.weight"), config.hidden_size);
            errdefer allocator.free(sh_norm);
            var sh_head = try LinearWeight.load(ctx, try file.get(try weights.layerName(&name_buf, mtp_i, "nextn.shared_head_head.weight")), config.vocab_size, config.hidden_size);
            errdefer sh_head.deinit();
            var mtp_layer = try runner.loadHostLayer(ctx, file, config, mtp_i, expert_store);
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

        var rope = try runner.HostRope.init(allocator, config, max_positions);
        errdefer rope.deinit(allocator);
        const last_hidden = try allocator.alloc(f32, config.hidden_size);

        return .{
            .allocator = allocator,
            .config = config,
            .nextn = nextn,
            .token_embedding = token_embedding,
            .output = output,
            .band = .{
                .layers = layers,
                .rope = rope,
                .output_norm = output_norm,
                .attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.head_dim))),
            },
            .mtp = mtp,
            .weight_mapping = weight_mapping,
            .expert_store = expert_store,
            .last_hidden = last_hidden,
        };
    }

    pub fn deinit(self: *Model) void {
        if (self.step_hiddens.len > 0) self.allocator.free(self.step_hiddens);
        self.allocator.free(self.last_hidden);
        if (self.mtp) |*m| m.deinit(self.allocator);
        self.band.deinit(self.allocator);
        self.output.deinit();
        self.token_embedding.deinit();
        if (self.expert_store) |store| store.destroy();
        if (self.weight_mapping) |*mapping| mapping.deinit();
        self.* = undefined;
    }

    pub fn initCache(self: *const Model, capacity: usize) !Cache {
        return Cache.init(self.allocator, self.config.num_layers, self.config.num_key_value_heads, self.config.head_dim, capacity);
    }

    pub fn initMtpCache(self: *const Model, capacity: usize) !Cache {
        return Cache.init(self.allocator, 1, self.config.num_key_value_heads, self.config.head_dim, capacity);
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
        if (cache.len + tokens.len > cache.capacity or cache.len + tokens.len > self.band.rope.capacity) return Error.KvCacheOverflow;

        const S = tokens.len;
        // Residual stream rows [S, hidden].
        const x = try allocator.alloc(f32, S * cfg.hidden_size);
        defer allocator.free(x);
        {
            var emb = try self.token_embedding.getRowsAs(ctx, tokens, .embed);
            defer emb.deinit();
            @memcpy(x, try emb.dataConst());
        }

        for (self.band.layers, 0..) |*layer, layer_i| {
            try runner.hostLayerForward(ctx, cfg, &self.band, cache, layer, layer_i, x, S, cache.len, null);
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
            host_ops.rmsNormInto(self.step_hiddens[r * cfg.hidden_size ..][0..cfg.hidden_size], x[r * cfg.hidden_size ..][0..cfg.hidden_size], self.band.output_norm, cfg.rms_norm_eps);
        }
        return self.headLogits(ctx, x, S, self.band.output_norm, &self.output);
    }

    fn headLogits(self: *Model, ctx: *ExecContext, x: []const f32, S: usize, norm: []const f32, head: *const LinearWeight) ![][]f32 {
        const cfg = self.config;
        const allocator = ctx.allocator;
        var normed_t = try fucina.Tensor(.{ .seq, .embed }).empty(ctx, .{ S, cfg.hidden_size });
        defer normed_t.deinit();
        const normed = try normed_t.data();
        for (0..S) |r| {
            host_ops.rmsNormInto(normed[r * cfg.hidden_size ..][0..cfg.hidden_size], x[r * cfg.hidden_size ..][0..cfg.hidden_size], norm, cfg.rms_norm_eps);
        }
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

    /// One MTP draft step: combine the token embedding with the previous
    /// hidden, run the nextn layer over the MTP stream's cache at position
    /// `mtp_cache.len`, and return (greedy token, logits row, new hidden).
    /// `h_prev` is `last_hidden` for the first draft and the returned
    /// hidden for the chained drafts. The MTP stream's rope position for
    /// entry i is i (validated empirically against the reference stacks).
    pub fn mtpDraftStep(self: *Model, ctx: *ExecContext, mtp_cache: *Cache, token: usize, h_prev: []const f32, h_out: []f32) ![]f32 {
        const cfg = self.config;
        const allocator = ctx.allocator;
        const mtp = if (self.mtp) |*m| m else return Error.InvalidConfig;
        if (mtp_cache.len >= mtp_cache.capacity or mtp_cache.len >= self.band.rope.capacity) return Error.KvCacheOverflow;

        const x = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(x);
        {
            var ids = [_]usize{token};
            var emb = try mtp.embed.getRowsAs(ctx, &ids, .embed);
            defer emb.deinit();
            var cat_t = try fucina.Tensor(.{ .seq, .embed }).empty(ctx, .{ 1, 2 * cfg.hidden_size });
            defer cat_t.deinit();
            const cat = try cat_t.data();
            // Concat order per the GLM/DeepSeek MTP reference: normed token
            // embedding first, normed trunk hidden second.
            host_ops.rmsNormInto(cat[0..cfg.hidden_size], try emb.dataConst(), mtp.enorm, cfg.rms_norm_eps);
            host_ops.rmsNormInto(cat[cfg.hidden_size..], h_prev, mtp.hnorm, cfg.rms_norm_eps);
            var proj_t = try mtp.eh_proj.linearSeq(ctx, &cat_t, .embed, .attn);
            defer proj_t.deinit();
            @memcpy(x, try proj_t.dataConst());
        }

        if (mtp_debug) {
            std.debug.print("mtp dbg: |h_prev|max {d:.3} |x=eh_proj|max {d:.3}", .{ maxAbs(h_prev), maxAbs(x) });
        }
        try runner.hostLayerForward(ctx, cfg, &self.band, null, &mtp.layer, 0, x, 1, mtp_cache.len, mtp_cache);
        @memcpy(h_out, x);
        if (mtp_debug) std.debug.print(" |h1|max {d:.3}\n", .{maxAbs(x)});

        const rows = try self.headLogits(ctx, x, 1, mtp.shared_head_norm, &mtp.shared_head);
        defer allocator.free(rows);
        return rows[0];
    }
};
