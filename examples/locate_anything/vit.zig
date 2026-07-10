//! MoonViT vision tower + 2x2 patch merger + MLP projector.
//!
//! Mirrors refs/locate-anything.cpp/src/{vit_encoder,vit_rope,vit_posemb,
//! projector}.cpp on Fucina tensor ops end to end: patch embedding is a
//! [hidden, patch_dim] linear over pre-patchified pixels, the interpolated
//! position embedding is torch-bicubic (a=-0.75) host code added as a tensor,
//! each encoder block is LayerNorm -> fused-QKV linear -> interleaved 2D RoPE
//! (hand-filled RopeTable over the shared fused kernel) -> bidirectional
//! grouped attention -> LayerNorm -> fc0+gelu(tanh) -> fc1, and the merger is
//! gather + axis split/merge views. All matmuls run through
//! `LinearWeight.linearSeq`, so BLAS/Metal/CUDA dispatch applies unchanged.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const config_mod = @import("config.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;
const weights = llm.weights;
const LinearWeight = weights.LinearWeight;
const Config = config_mod.Config;

const HiddenVec = fucina.Tensor(.{.hidden});
const ChannelVec = fucina.Tensor(.{.channel});
const EmbedVec = fucina.Tensor(.{.embed});
const Tokens = fucina.Tensor(.{ .seq, .hidden });
pub const Projected = fucina.Tensor(.{ .seq, .embed });

/// Load a 1-d f32 tensor as an owned raw slice (linear biases and the raw
/// pos-emb table are consumed host-side / via the in-place bias kernels).
fn loadRawVector(allocator: Allocator, info: *const gguf.TensorInfo, expected_len: usize) ![]f32 {
    if (info.ggml_type != .f32) return weights.Error.UnsupportedWeightType;
    const count = blk: {
        var n: usize = 1;
        for (info.dims[0..info.n_dims]) |d| n *= d;
        break :blk n;
    };
    if (count != expected_len) return weights.Error.InvalidWeightShape;
    const values = try allocator.alloc(f32, count);
    errdefer allocator.free(values);
    const bytes = info.data;
    if (bytes.len != count * 4) return weights.Error.InvalidWeightShape;
    for (values, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return values;
}

const Layer = struct {
    norm0_w: HiddenVec,
    norm0_b: HiddenVec,
    wqkv: LinearWeight,
    wqkv_b: []f32,
    wo: LinearWeight,
    wo_b: []f32,
    norm1_w: HiddenVec,
    norm1_b: HiddenVec,
    fc0: LinearWeight,
    fc0_b: []f32,
    fc1: LinearWeight,
    fc1_b: []f32,

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize) !Layer {
        var name_buf: [64]u8 = undefined;
        const t = struct {
            fn name(buf: []u8, i: usize, suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "vit.blk.{d}.{s}", .{ i, suffix });
            }
        }.name;
        const hid = config.vit_hidden;
        const qkv_dim = 3 * config.vit_n_heads * config.vit_head_dim;

        var norm0_w = try weights.loadVector(ctx, try file.get(try t(&name_buf, layer_i, "norm0.weight")), hid, .hidden);
        errdefer norm0_w.deinit();
        var norm0_b = try weights.loadVector(ctx, try file.get(try t(&name_buf, layer_i, "norm0.bias")), hid, .hidden);
        errdefer norm0_b.deinit();
        var wqkv = try LinearWeight.load(ctx, try file.get(try t(&name_buf, layer_i, "wqkv.weight")), qkv_dim, hid);
        errdefer wqkv.deinit();
        const wqkv_b = try loadRawVector(ctx.allocator, try file.get(try t(&name_buf, layer_i, "wqkv.bias")), qkv_dim);
        errdefer ctx.allocator.free(wqkv_b);
        var wo = try LinearWeight.load(ctx, try file.get(try t(&name_buf, layer_i, "wo.weight")), hid, hid);
        errdefer wo.deinit();
        const wo_b = try loadRawVector(ctx.allocator, try file.get(try t(&name_buf, layer_i, "wo.bias")), hid);
        errdefer ctx.allocator.free(wo_b);
        var norm1_w = try weights.loadVector(ctx, try file.get(try t(&name_buf, layer_i, "norm1.weight")), hid, .hidden);
        errdefer norm1_w.deinit();
        var norm1_b = try weights.loadVector(ctx, try file.get(try t(&name_buf, layer_i, "norm1.bias")), hid, .hidden);
        errdefer norm1_b.deinit();
        var fc0 = try LinearWeight.load(ctx, try file.get(try t(&name_buf, layer_i, "fc0.weight")), config.vit_intermediate, hid);
        errdefer fc0.deinit();
        const fc0_b = try loadRawVector(ctx.allocator, try file.get(try t(&name_buf, layer_i, "fc0.bias")), config.vit_intermediate);
        errdefer ctx.allocator.free(fc0_b);
        var fc1 = try LinearWeight.load(ctx, try file.get(try t(&name_buf, layer_i, "fc1.weight")), hid, config.vit_intermediate);
        errdefer fc1.deinit();
        const fc1_b = try loadRawVector(ctx.allocator, try file.get(try t(&name_buf, layer_i, "fc1.bias")), hid);
        errdefer ctx.allocator.free(fc1_b);

        return .{
            .norm0_w = norm0_w,
            .norm0_b = norm0_b,
            .wqkv = wqkv,
            .wqkv_b = wqkv_b,
            .wo = wo,
            .wo_b = wo_b,
            .norm1_w = norm1_w,
            .norm1_b = norm1_b,
            .fc0 = fc0,
            .fc0_b = fc0_b,
            .fc1 = fc1,
            .fc1_b = fc1_b,
        };
    }

    fn deinit(self: *Layer, allocator: Allocator) void {
        allocator.free(self.fc1_b);
        self.fc1.deinit();
        allocator.free(self.fc0_b);
        self.fc0.deinit();
        self.norm1_b.deinit();
        self.norm1_w.deinit();
        allocator.free(self.wo_b);
        self.wo.deinit();
        allocator.free(self.wqkv_b);
        self.wqkv.deinit();
        self.norm0_b.deinit();
        self.norm0_w.deinit();
        self.* = undefined;
    }
};

pub const Vit = struct {
    allocator: Allocator,
    config: Config,
    patch_embed: LinearWeight,
    patch_embed_b: []f32,
    /// Raw 64x64x[hidden] position-embedding table, host-interpolated per grid.
    pos_emb: []f32,
    layers: []Layer,
    final_norm_w: HiddenVec,
    final_norm_b: HiddenVec,
    // Projector (mlp1): LayerNorm(4*hidden) -> Linear -> gelu_erf -> Linear.
    proj_norm_w: ChannelVec,
    proj_norm_b: ChannelVec,
    proj_fc0: LinearWeight,
    proj_fc0_b: []f32,
    proj_fc1: LinearWeight,
    proj_fc1_b: []f32,
    kv_head_for_head: []usize,

    pub fn load(ctx: *ExecContext, file: *const gguf.File, config: Config) !Vit {
        const allocator = ctx.allocator;
        const hid = config.vit_hidden;
        const patch_dim = config.vit_patch * config.vit_patch * 3;
        const merged_dim = config.vit_merge_h * config.vit_merge_w * hid;
        const lm_hidden = config.lm_hidden;

        // patch_embed ships as the conv kernel [14,14,3,1152] (ggml ne, inner
        // first); its flat layout IS the [hid, patch_dim] linear over the
        // per-patch (c,i,j) pixel vector, so reshape the TensorInfo in place.
        var patch_embed_info = (try file.get("vit.patch_embed.weight")).*;
        if (patch_embed_info.n_dims == 4) {
            patch_embed_info.dims = .{ patch_embed_info.dims[0] * patch_embed_info.dims[1] * patch_embed_info.dims[2], patch_embed_info.dims[3], 1, 1 };
            patch_embed_info.n_dims = 2;
        }
        var patch_embed = try LinearWeight.load(ctx, &patch_embed_info, hid, patch_dim);
        errdefer patch_embed.deinit();
        const patch_embed_b = try loadRawVector(allocator, try file.get("vit.patch_embed.bias"), hid);
        errdefer allocator.free(patch_embed_b);
        const pos_emb = try loadRawVector(allocator, try file.get("vit.pos_emb.weight"), config.vit_pos_emb_hw * config.vit_pos_emb_hw * hid);
        errdefer allocator.free(pos_emb);

        var final_norm_w = try weights.loadVector(ctx, try file.get("vit.final_norm.weight"), hid, .hidden);
        errdefer final_norm_w.deinit();
        var final_norm_b = try weights.loadVector(ctx, try file.get("vit.final_norm.bias"), hid, .hidden);
        errdefer final_norm_b.deinit();

        var proj_norm_w = try weights.loadVector(ctx, try file.get("proj.0.weight"), merged_dim, .channel);
        errdefer proj_norm_w.deinit();
        var proj_norm_b = try weights.loadVector(ctx, try file.get("proj.0.bias"), merged_dim, .channel);
        errdefer proj_norm_b.deinit();
        var proj_fc0 = try LinearWeight.load(ctx, try file.get("proj.1.weight"), lm_hidden, merged_dim);
        errdefer proj_fc0.deinit();
        const proj_fc0_b = try loadRawVector(allocator, try file.get("proj.1.bias"), lm_hidden);
        errdefer allocator.free(proj_fc0_b);
        var proj_fc1 = try LinearWeight.load(ctx, try file.get("proj.3.weight"), lm_hidden, lm_hidden);
        errdefer proj_fc1.deinit();
        const proj_fc1_b = try loadRawVector(allocator, try file.get("proj.3.bias"), lm_hidden);
        errdefer allocator.free(proj_fc1_b);

        const kv_head_for_head = try allocator.alloc(usize, config.vit_n_heads);
        errdefer allocator.free(kv_head_for_head);
        for (kv_head_for_head, 0..) |*slot, i| slot.* = i; // MHA: identity

        const layers = try allocator.alloc(Layer, config.vit_n_layers);
        errdefer allocator.free(layers);
        var loaded: usize = 0;
        errdefer for (layers[0..loaded]) |*layer| layer.deinit(allocator);
        for (layers, 0..) |*layer, i| {
            layer.* = try Layer.load(ctx, file, config, i);
            loaded += 1;
        }

        return .{
            .allocator = allocator,
            .config = config,
            .patch_embed = patch_embed,
            .patch_embed_b = patch_embed_b,
            .pos_emb = pos_emb,
            .layers = layers,
            .final_norm_w = final_norm_w,
            .final_norm_b = final_norm_b,
            .proj_norm_w = proj_norm_w,
            .proj_norm_b = proj_norm_b,
            .proj_fc0 = proj_fc0,
            .proj_fc0_b = proj_fc0_b,
            .proj_fc1 = proj_fc1,
            .proj_fc1_b = proj_fc1_b,
            .kv_head_for_head = kv_head_for_head,
        };
    }

    pub fn deinit(self: *Vit) void {
        const allocator = self.allocator;
        for (self.layers) |*layer| layer.deinit(allocator);
        allocator.free(self.layers);
        allocator.free(self.kv_head_for_head);
        allocator.free(self.proj_fc1_b);
        self.proj_fc1.deinit();
        allocator.free(self.proj_fc0_b);
        self.proj_fc0.deinit();
        self.proj_norm_b.deinit();
        self.proj_norm_w.deinit();
        self.final_norm_b.deinit();
        self.final_norm_w.deinit();
        allocator.free(self.pos_emb);
        allocator.free(self.patch_embed_b);
        self.patch_embed.deinit();
        self.* = undefined;
    }

    /// patch_embed linear + bias + interpolated pos-emb: [n_tok, hidden].
    pub fn patchAndPos(self: *const Vit, ctx: *ExecContext, pixel_values: []const f32, gh: usize, gw: usize) !Tokens {
        const hid = self.config.vit_hidden;
        const patch_dim = self.config.vit_patch * self.config.vit_patch * 3;
        const n_tok = gh * gw;
        std.debug.assert(pixel_values.len == n_tok * patch_dim);

        var px = try fucina.Tensor(.{ .seq, .pixel }).fromSlice(ctx, .{ n_tok, patch_dim }, pixel_values);
        defer px.deinit();
        var x = try self.patch_embed.linearSeq(ctx, &px, .pixel, .hidden);
        errdefer x.deinit();
        try x.addAxisVectorInPlace(ctx, self.patch_embed_b, .hidden);

        const pos = try bicubicPosEmb(ctx.allocator, self.pos_emb, self.config.vit_pos_emb_hw, self.config.vit_pos_emb_hw, hid, gh, gw);
        defer ctx.allocator.free(pos);
        var pos_t = try Tokens.fromSlice(ctx, .{ n_tok, hid }, pos);
        defer pos_t.deinit();
        try x.addScaledInPlace(ctx, &pos_t, 1.0);
        return x;
    }

    /// One encoder block over [seq, hidden].
    fn block(self: *const Vit, ctx: *ExecContext, x: *Tokens, layer: *const Layer, rope_table: *const fucina.RopeTable) !void {
        const heads = self.config.vit_n_heads;
        const head_dim = self.config.vit_head_dim;
        const hid = self.config.vit_hidden;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // --- attention ---
        var xn = try x.layerNorm(ctx, .hidden, 1e-5, .{ .weight = &layer.norm0_w, .bias = &layer.norm0_b });
        defer xn.deinit();
        var qkv = try layer.wqkv.linearSeq(ctx, &xn, .hidden, .qkv);
        defer qkv.deinit();
        try qkv.addAxisVectorInPlace(ctx, layer.wqkv_b, .qkv);

        var q_flat = try qkv.narrow(ctx, .qkv, 0, hid);
        defer q_flat.deinit();
        var k_flat = try qkv.narrow(ctx, .qkv, hid, hid);
        defer k_flat.deinit();
        var v_flat = try qkv.narrow(ctx, .qkv, 2 * hid, hid);
        defer v_flat.deinit();

        var q = try q_flat.split(ctx, .qkv, .{ .head, .d }, .{ heads, head_dim });
        defer q.deinit();
        var k_pre = try k_flat.split(ctx, .qkv, .{ .kv_head, .d }, .{ heads, head_dim });
        defer k_pre.deinit();
        var v = try v_flat.split(ctx, .qkv, .{ .kv_head, .d }, .{ heads, head_dim });
        defer v.deinit();

        var q_rope = try q.rope(ctx, .seq, .d, rope_table, .interleaved);
        defer q_rope.deinit();
        var k_rope = try k_pre.rope(ctx, .seq, .d, rope_table, .interleaved);
        defer k_rope.deinit();

        var attn = try q_rope.groupedAttention(ctx, &k_rope, &v, self.kv_head_for_head, .attn, scale, .{ .mask = .bidirectional });
        defer attn.deinit();

        var o = try layer.wo.linearSeq(ctx, &attn, .attn, .hidden);
        defer o.deinit();
        try o.addAxisVectorInPlace(ctx, layer.wo_b, .hidden);
        try x.addScaledInPlace(ctx, &o, 1.0);

        // --- mlp ---
        var mn = try x.layerNorm(ctx, .hidden, 1e-5, .{ .weight = &layer.norm1_w, .bias = &layer.norm1_b });
        defer mn.deinit();
        var m = try layer.fc0.linearSeq(ctx, &mn, .hidden, .ffn);
        defer m.deinit();
        try m.addAxisVectorUnaryInPlace(ctx, .gelu, layer.fc0_b, .ffn);
        var m2 = try layer.fc1.linearSeq(ctx, &m, .ffn, .hidden);
        defer m2.deinit();
        try m2.addAxisVectorInPlace(ctx, layer.fc1_b, .hidden);
        try x.addScaledInPlace(ctx, &m2, 1.0);
    }

    /// Full tower: patch+pos -> blocks -> final LayerNorm. `stop_after_block`
    /// limits the block count (parity gates for block 0 / layer 26); null runs
    /// all layers plus the final norm.
    pub fn forward(self: *const Vit, ctx: *ExecContext, pixel_values: []const f32, gh: usize, gw: usize, stop_after_block: ?usize) !Tokens {
        var x = try self.patchAndPos(ctx, pixel_values, gh, gw);
        errdefer x.deinit();

        var rope_table = try buildRope2dTable(ctx.allocator, gh, gw, self.config.vit_head_dim, self.config.vit_rope_theta);
        defer rope_table.deinit();

        const n_blocks = if (stop_after_block) |last| last + 1 else self.layers.len;
        for (self.layers[0..n_blocks]) |*layer| {
            try self.block(ctx, &x, layer, &rope_table);
        }
        if (stop_after_block != null) return x;

        const final = try x.layerNorm(ctx, .hidden, 1e-5, .{ .weight = &self.final_norm_w, .bias = &self.final_norm_b });
        x.deinit();
        return final;
    }

    /// 2x2 patch merge: [gh*gw, hidden] -> [(gh/2)*(gw/2), 4*hidden], channel
    /// concat order TL,TR,BL,BR — a row gather + two axis views, no host math.
    pub fn mergePatches(self: *const Vit, ctx: *ExecContext, vit_final: *const Tokens, gh: usize, gw: usize) !fucina.Tensor(.{ .seq, .channel }) {
        const mh = gh / self.config.vit_merge_h;
        const mw = gw / self.config.vit_merge_w;
        const indices = try ctx.allocator.alloc(usize, mh * mw * 4);
        defer ctx.allocator.free(indices);
        var slot: usize = 0;
        for (0..mh) |a| {
            for (0..mw) |c| {
                inline for (0..2) |b| {
                    inline for (0..2) |e| {
                        indices[slot] = (2 * a + b) * gw + (2 * c + e);
                        slot += 1;
                    }
                }
            }
        }
        var gathered = try vit_final.gather(ctx, .seq, indices, .seq);
        defer gathered.deinit();
        var grouped = try gathered.split(ctx, .seq, .{ .seq, .four }, .{ mh * mw, 4 });
        defer grouped.deinit();
        return grouped.merge(ctx, .channel, .{ .four, .hidden });
    }

    /// Projector mlp1: LayerNorm -> Linear -> gelu_erf -> Linear: [seq, embed].
    pub fn project(self: *const Vit, ctx: *ExecContext, merged: *const fucina.Tensor(.{ .seq, .channel })) !Projected {
        var xn = try merged.layerNorm(ctx, .channel, 1e-5, .{ .weight = &self.proj_norm_w, .bias = &self.proj_norm_b });
        defer xn.deinit();
        var h = try self.proj_fc0.linearSeq(ctx, &xn, .channel, .proj);
        defer h.deinit();
        try h.addAxisVectorUnaryInPlace(ctx, .gelu_erf, self.proj_fc0_b, .proj);
        var out = try self.proj_fc1.linearSeq(ctx, &h, .proj, .embed);
        errdefer out.deinit();
        try out.addAxisVectorInPlace(ctx, self.proj_fc1_b, .embed);
        return out;
    }
};

/// The reference Rope2DPosEmb table (vit_rope.cpp build_rope_tables): pair k
/// rotates by angle coord * theta^(-4*(k/2)/head_dim), where even pairs take
/// the column and odd pairs the row of the token's grid position. Packed into
/// the shared `RopeTable` (values = sin then cos, positions.len * pair_count
/// each) and applied by the stock `.interleaved` rope kernel.
pub fn buildRope2dTable(allocator: Allocator, gh: usize, gw: usize, head_dim: usize, theta: f32) !fucina.RopeTable {
    const n_tok = gh * gw;
    const pair_count = head_dim / 2;
    const angle_count = n_tok * pair_count;

    const positions = try allocator.alloc(i32, n_tok);
    errdefer allocator.free(positions);
    for (positions, 0..) |*p, i| p.* = @intCast(i);

    const values = try allocator.alloc(f32, angle_count * 2);
    errdefer allocator.free(values);
    const sin_values = values[0..angle_count];
    const cos_values = values[angle_count..];

    for (0..gh) |h| {
        for (0..gw) |w| {
            const tok = h * gw + w;
            for (0..pair_count) |k| {
                const i = k / 2;
                const invf = std.math.pow(f32, theta, -4.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
                const coord: f32 = if (k % 2 == 0) @floatFromInt(w) else @floatFromInt(h);
                const ang = coord * invf;
                sin_values[tok * pair_count + k] = @sin(ang);
                cos_values[tok * pair_count + k] = @cos(ang);
            }
        }
    }

    return .{
        .allocator = allocator,
        .positions = positions,
        .theta_base = theta,
        .feature_dim = head_dim,
        .pair_count = pair_count,
        .values = values,
    };
}

/// Torch-bicubic (a=-0.75, align_corners=False, no antialias) interpolation of
/// the [base_h, base_w, c] pos-emb table to [gh, gw, c] (vit_posemb.cpp).
/// Parity-critical scalar host code: runs once per image, f32 accumulation
/// exactly like the reference.
pub fn bicubicPosEmb(allocator: Allocator, src: []const f32, base_h: usize, base_w: usize, c: usize, gh: usize, gw: usize) ![]f32 {
    const a: f32 = -0.75;
    const sh = @as(f32, @floatFromInt(base_h)) / @as(f32, @floatFromInt(gh));
    const sw = @as(f32, @floatFromInt(base_w)) / @as(f32, @floatFromInt(gw));

    const out = try allocator.alloc(f32, gh * gw * c);
    errdefer allocator.free(out);

    const base_h_i: isize = @intCast(base_h);
    const base_w_i: isize = @intCast(base_w);

    for (0..gh) |oy| {
        const fy = (@as(f32, @floatFromInt(oy)) + 0.5) * sh - 0.5;
        const iy: isize = @intFromFloat(@floor(fy));
        const ty = fy - @as(f32, @floatFromInt(iy));
        const wy = [4]f32{ cubicW(1.0 + ty, a), cubicW(ty, a), cubicW(1.0 - ty, a), cubicW(2.0 - ty, a) };
        for (0..gw) |ox| {
            const fx = (@as(f32, @floatFromInt(ox)) + 0.5) * sw - 0.5;
            const ix: isize = @intFromFloat(@floor(fx));
            const tx = fx - @as(f32, @floatFromInt(ix));
            const wx = [4]f32{ cubicW(1.0 + tx, a), cubicW(tx, a), cubicW(1.0 - tx, a), cubicW(2.0 - tx, a) };
            const dst = out[(oy * gw + ox) * c ..][0..c];
            for (0..c) |ch| {
                var acc: f32 = 0.0;
                inline for (0..4) |m| {
                    var row: f32 = 0.0;
                    inline for (0..4) |n| {
                        const y: usize = @intCast(std.math.clamp(iy - 1 + @as(isize, m), 0, base_h_i - 1));
                        const x: usize = @intCast(std.math.clamp(ix - 1 + @as(isize, n), 0, base_w_i - 1));
                        row += wx[n] * src[(y * base_w + x) * c + ch];
                    }
                    acc += wy[m] * row;
                }
                dst[ch] = acc;
            }
        }
    }
    return out;
}

fn cubicW(t_in: f32, a: f32) f32 {
    const t = @abs(t_in);
    if (t <= 1.0) return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    if (t < 2.0) return (((t - 5.0) * t + 8.0) * t - 4.0) * a;
    return 0.0;
}
