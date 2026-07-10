//! Qwen2.5-3B language model: resident-KV causal decode + the MTP
//! (parallel-box-decoding) block forward.
//!
//! Mirrors refs/locate-anything.cpp/src/{qwen2,lm}.cpp on Fucina ops. The KV
//! cache is f32 like the reference's ResidentKV (token-stream parity leaves no
//! room for an f16 round-trip here); the MTP round's block-diffusion mask is
//! the reference's build_mtp_mask with -1e9 standing in for -inf — the biased
//! attention kernel takes finite additive biases, and exp(-1e9 - m)
//! underflows to exactly 0.0f in f32, so masked keys get probability 0
//! bit-for-bit like ggml's -inf soft_max_ext path.
//!
//! Every linear goes through `LinearWeight.linearSeq` (f32/f16/q8_0/K-quant
//! arms + the BLAS/Metal/CUDA dispatch), attention through the shared
//! grouped-attention kernels, RoPE through the stock half (NeoX) kernel.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const config_mod = @import("config.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;
const weights = llm.weights;
const gguf_meta = llm.gguf_meta;
const LinearWeight = weights.LinearWeight;
const Config = config_mod.Config;

const EmbedVec = fucina.Tensor(.{.embed});
const Seq = fucina.Tensor(.{ .seq, .embed });
const KvTensor = fucina.Tensor(.{ .seq, .kv_head, .d });

/// Mask stand-in for -inf (see module doc).
pub const mask_neg = -1.0e9;

fn loadRawVector(allocator: Allocator, info: *const gguf.TensorInfo, expected_len: usize) ![]f32 {
    if (info.ggml_type != .f32) return weights.Error.UnsupportedWeightType;
    var count: usize = 1;
    for (info.dims[0..info.n_dims]) |d| count *= d;
    if (count != expected_len or info.data.len != count * 4) return weights.Error.InvalidWeightShape;
    const values = try allocator.alloc(f32, count);
    errdefer allocator.free(values);
    for (values, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, info.data[i * 4 ..][0..4], .little));
    return values;
}

const Layer = struct {
    attn_norm: EmbedVec,
    q_proj: LinearWeight,
    q_b: []f32,
    k_proj: LinearWeight,
    k_b: []f32,
    v_proj: LinearWeight,
    v_b: []f32,
    o_proj: LinearWeight,
    ffn_norm: EmbedVec,
    gate_proj: LinearWeight,
    up_proj: LinearWeight,
    down_proj: LinearWeight,

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Config, layer_i: usize) !Layer {
        var name_buf: [64]u8 = undefined;
        const t = struct {
            fn name(buf: []u8, i: usize, suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "lm.blk.{d}.{s}", .{ i, suffix });
            }
        }.name;
        const hid = config.lm_hidden;
        const q_dim = config.lm_n_heads * config.lm_head_dim;
        const kv_dim = config.lm_n_kv_heads * config.lm_head_dim;

        var attn_norm = try weights.loadVector(ctx, try file.get(try t(&name_buf, layer_i, "attn_norm.weight")), hid, .embed);
        errdefer attn_norm.deinit();
        var q_proj = try LinearWeight.load(ctx, try file.get(try t(&name_buf, layer_i, "attn_q.weight")), q_dim, hid);
        errdefer q_proj.deinit();
        const q_b = try loadRawVector(ctx.allocator, try file.get(try t(&name_buf, layer_i, "attn_q.bias")), q_dim);
        errdefer ctx.allocator.free(q_b);
        var k_proj = try LinearWeight.load(ctx, try file.get(try t(&name_buf, layer_i, "attn_k.weight")), kv_dim, hid);
        errdefer k_proj.deinit();
        const k_b = try loadRawVector(ctx.allocator, try file.get(try t(&name_buf, layer_i, "attn_k.bias")), kv_dim);
        errdefer ctx.allocator.free(k_b);
        var v_proj = try LinearWeight.load(ctx, try file.get(try t(&name_buf, layer_i, "attn_v.weight")), kv_dim, hid);
        errdefer v_proj.deinit();
        const v_b = try loadRawVector(ctx.allocator, try file.get(try t(&name_buf, layer_i, "attn_v.bias")), kv_dim);
        errdefer ctx.allocator.free(v_b);
        var o_proj = try LinearWeight.load(ctx, try file.get(try t(&name_buf, layer_i, "attn_o.weight")), hid, q_dim);
        errdefer o_proj.deinit();
        var ffn_norm = try weights.loadVector(ctx, try file.get(try t(&name_buf, layer_i, "ffn_norm.weight")), hid, .embed);
        errdefer ffn_norm.deinit();
        var gate_proj = try LinearWeight.load(ctx, try file.get(try t(&name_buf, layer_i, "ffn_gate.weight")), config.lm_intermediate, hid);
        errdefer gate_proj.deinit();
        var up_proj = try LinearWeight.load(ctx, try file.get(try t(&name_buf, layer_i, "ffn_up.weight")), config.lm_intermediate, hid);
        errdefer up_proj.deinit();
        var down_proj = try LinearWeight.load(ctx, try file.get(try t(&name_buf, layer_i, "ffn_down.weight")), hid, config.lm_intermediate);
        errdefer down_proj.deinit();

        return .{
            .attn_norm = attn_norm,
            .q_proj = q_proj,
            .q_b = q_b,
            .k_proj = k_proj,
            .k_b = k_b,
            .v_proj = v_proj,
            .v_b = v_b,
            .o_proj = o_proj,
            .ffn_norm = ffn_norm,
            .gate_proj = gate_proj,
            .up_proj = up_proj,
            .down_proj = down_proj,
        };
    }

    fn deinit(self: *Layer, allocator: Allocator) void {
        self.down_proj.deinit();
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.ffn_norm.deinit();
        self.o_proj.deinit();
        allocator.free(self.v_b);
        self.v_proj.deinit();
        allocator.free(self.k_b);
        self.k_proj.deinit();
        allocator.free(self.q_b);
        self.q_proj.deinit();
        self.attn_norm.deinit();
        self.* = undefined;
    }
};

const LayerLoader = struct {
    ctx: *ExecContext,
    file: *const gguf.File,
    config: Config,

    pub fn load(self: LayerLoader, layer_i: usize) !Layer {
        return Layer.load(self.ctx, self.file, self.config, layer_i);
    }

    pub fn deinitLayer(self: LayerLoader, layer: *Layer) void {
        layer.deinit(self.ctx.allocator);
    }
};

/// Per-layer resident f32 K/V, written in place at an explicit offset —
/// the reference's ResidentKV. `len` is the committed prefix; MTP rounds
/// write scratch rows past `len` without advancing it.
pub const Cache = struct {
    allocator: Allocator,
    k: []KvTensor,
    v: []KvTensor,
    capacity: usize,
    len: usize,

    pub fn deinit(self: *Cache) void {
        for (self.k) |*t| t.deinit();
        for (self.v) |*t| t.deinit();
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.* = undefined;
    }
};

pub const Lm = struct {
    allocator: Allocator,
    config: Config,
    /// f32 [vocab, hidden] token-embedding bytes (mmap view; host row gather).
    tok_embd_bytes: []const u8,
    /// lm.output.weight (or the tied tok_embd fallback), first `output_rows`
    /// rows. Quantized heads whose row count is not a multiple of 8 (this
    /// model's vocab is 152681) are split: the lane-packed RHS layouts group
    /// 4 or 8 rows, so the largest x8-aligned prefix loads packed and the
    /// 1..7 remainder rows dequantize to `output_tail` (f32), with the two
    /// logit strips concatenated in projectLogits.
    output: LinearWeight,
    output_rows: usize,
    output_tail: ?fucina.Tensor(.{ .vocab, .embed }),
    output_norm: EmbedVec,
    layers: []Layer,
    kv_head_for_head: []usize,

    pub fn load(ctx: *ExecContext, file: *const gguf.File, config: Config) !Lm {
        const allocator = ctx.allocator;

        const tok_embd_info = try file.get("lm.tok_embd.weight");
        if (tok_embd_info.ggml_type != .f32) return weights.Error.UnsupportedWeightType;
        if (tok_embd_info.n_dims != 2 or tok_embd_info.dims[0] != config.lm_hidden or tok_embd_info.dims[1] != config.lm_vocab)
            return weights.Error.InvalidWeightShape;

        const output_info = file.maybeGet("lm.output.weight") orelse tok_embd_info;
        const head_quantized = output_info.ggml_type != .f32 and output_info.ggml_type != .f16 and output_info.ggml_type != .bf16;
        const output_rows = if (head_quantized) config.lm_vocab & ~@as(usize, 7) else config.lm_vocab;
        const tail_rows = config.lm_vocab - output_rows;

        var main_info = output_info.*;
        if (tail_rows > 0) {
            if (output_info.n_dims != 2 or output_info.dims[1] != config.lm_vocab) return weights.Error.InvalidWeightShape;
            if (output_info.data.len % config.lm_vocab != 0) return weights.Error.InvalidWeightShape;
            const row_bytes = output_info.data.len / config.lm_vocab;
            main_info.dims[1] = output_rows;
            main_info.data = output_info.data[0 .. output_rows * row_bytes];
        }
        var output = try LinearWeight.load(ctx, &main_info, output_rows, config.lm_hidden);
        errdefer output.deinit();

        var output_tail: ?fucina.Tensor(.{ .vocab, .embed }) = null;
        errdefer if (output_tail) |*t| t.deinit();
        if (tail_rows > 0) {
            const row_bytes = output_info.data.len / config.lm_vocab;
            const tail_values = try allocator.alloc(f32, tail_rows * config.lm_hidden);
            defer allocator.free(tail_values);
            try gguf.decodeF32(output_info.ggml_type, output_info.data[output_rows * row_bytes ..], tail_values);
            output_tail = try fucina.Tensor(.{ .vocab, .embed }).fromSlice(ctx, .{ tail_rows, config.lm_hidden }, tail_values);
        }

        var output_norm = try weights.loadVector(ctx, try file.get("lm.output_norm.weight"), config.lm_hidden, .embed);
        errdefer output_norm.deinit();

        const kv_head_for_head = try allocator.alloc(usize, config.lm_n_heads);
        errdefer allocator.free(kv_head_for_head);
        const group = config.lm_n_heads / config.lm_n_kv_heads;
        for (kv_head_for_head, 0..) |*slot, i| slot.* = i / group;

        const layers = try allocator.alloc(Layer, config.lm_n_layers);
        errdefer allocator.free(layers);
        try gguf_meta.parallelLoadLayers(Layer, LayerLoader, ctx, .{ .ctx = ctx, .file = file, .config = config }, layers);

        return .{
            .allocator = allocator,
            .config = config,
            .tok_embd_bytes = tok_embd_info.data,
            .output = output,
            .output_rows = output_rows,
            .output_tail = output_tail,
            .output_norm = output_norm,
            .layers = layers,
            .kv_head_for_head = kv_head_for_head,
        };
    }

    pub fn deinit(self: *Lm) void {
        for (self.layers) |*layer| layer.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.allocator.free(self.kv_head_for_head);
        self.output_norm.deinit();
        if (self.output_tail) |*t| t.deinit();
        self.output.deinit();
        self.* = undefined;
    }

    pub fn initCache(self: *const Lm, ctx: *ExecContext, capacity: usize) !Cache {
        const allocator = ctx.allocator;
        const n_layers = self.config.lm_n_layers;
        const k = try allocator.alloc(KvTensor, n_layers);
        errdefer allocator.free(k);
        const v = try allocator.alloc(KvTensor, n_layers);
        errdefer allocator.free(v);
        var made: usize = 0;
        errdefer for (0..made) |i| {
            k[i].deinit();
            v[i].deinit();
        };
        for (0..n_layers) |i| {
            k[i] = try KvTensor.empty(ctx, .{ capacity, self.config.lm_n_kv_heads, self.config.lm_head_dim });
            errdefer k[i].deinit();
            v[i] = try KvTensor.empty(ctx, .{ capacity, self.config.lm_n_kv_heads, self.config.lm_head_dim });
            made += 1;
        }
        return .{ .allocator = allocator, .k = k, .v = v, .capacity = capacity, .len = 0 };
    }

    /// Host row-gather from the f32 token-embedding table:
    /// out[i] = tok_embd[ids[i]], flat [ids.len, hidden]. Caller frees.
    pub fn embedTokens(self: *const Lm, allocator: Allocator, ids: []const u32) ![]f32 {
        const hid = self.config.lm_hidden;
        const out = try allocator.alloc(f32, ids.len * hid);
        errdefer allocator.free(out);
        const row_bytes = hid * 4;
        for (ids, 0..) |id, i| {
            if (id >= self.config.lm_vocab) return error.TokenIdOutOfRange;
            const src = self.tok_embd_bytes[@as(usize, id) * row_bytes ..][0..row_bytes];
            @memcpy(std.mem.sliceAsBytes(out[i * hid ..][0..hid]), src);
        }
        return out;
    }

    const AttentionMode = union(enum) {
        /// Plain causal chunk: queries at absolute positions
        /// [kv_len - n_new, kv_len) over the cache prefix [0, kv_len).
        causal,
        /// Additive-bias attention over the cache prefix [0, kv_len)
        /// (the MTP block-diffusion mask; bias is [n_new, kv_len], row-major).
        biased: []const f32,
    };

    /// Run all layers over `x_host` ([n_new, hidden] flat), writing this
    /// chunk's K/V into the cache at row `write_offset` and attending over the
    /// prefix [0, kv_len). `positions` are the absolute RoPE positions of the
    /// n_new rows. Does NOT advance cache.len. Returns the hidden states
    /// [n_new, hidden] (pre final-norm).
    fn forwardLayers(
        self: *const Lm,
        ctx: *ExecContext,
        cache: *Cache,
        x_host: []const f32,
        n_new: usize,
        positions: []const i32,
        write_offset: usize,
        kv_len: usize,
        mode: AttentionMode,
    ) !Seq {
        const config = self.config;
        const hid = config.lm_hidden;
        std.debug.assert(x_host.len == n_new * hid);
        std.debug.assert(positions.len == n_new);
        std.debug.assert(write_offset + n_new <= cache.capacity);
        std.debug.assert(kv_len <= write_offset + n_new);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.lm_head_dim)));

        var rope_table = try ctx.prepareRopeTable(positions, config.lm_head_dim, config.lm_rope_theta, false);
        defer rope_table.deinit();

        var bias_tensor: ?fucina.Tensor(.{ .q_seq, .kv_seq }) = null;
        defer if (bias_tensor) |*t| t.deinit();
        if (mode == .biased) {
            std.debug.assert(mode.biased.len == n_new * kv_len);
            bias_tensor = try fucina.Tensor(.{ .q_seq, .kv_seq }).fromSlice(ctx, .{ n_new, kv_len }, mode.biased);
        }

        var x = try Seq.fromSlice(ctx, .{ n_new, hid }, x_host);
        errdefer x.deinit();

        for (self.layers, 0..) |*layer, layer_i| {
            var xn = try x.rmsNormMul(ctx, .embed, &layer.attn_norm, config.lm_rms_eps);
            defer xn.deinit();

            var q_flat = try layer.q_proj.linearSeq(ctx, &xn, .embed, .q);
            defer q_flat.deinit();
            try q_flat.addAxisVectorInPlace(ctx, layer.q_b, .q);
            var k_flat = try layer.k_proj.linearSeq(ctx, &xn, .embed, .k);
            defer k_flat.deinit();
            try k_flat.addAxisVectorInPlace(ctx, layer.k_b, .k);
            var v_flat = try layer.v_proj.linearSeq(ctx, &xn, .embed, .v);
            defer v_flat.deinit();
            try v_flat.addAxisVectorInPlace(ctx, layer.v_b, .v);

            var q3 = try q_flat.split(ctx, .q, .{ .head, .d }, .{ config.lm_n_heads, config.lm_head_dim });
            defer q3.deinit();
            var k3 = try k_flat.split(ctx, .k, .{ .kv_head, .d }, .{ config.lm_n_kv_heads, config.lm_head_dim });
            defer k3.deinit();
            var v3 = try v_flat.split(ctx, .v, .{ .kv_head, .d }, .{ config.lm_n_kv_heads, config.lm_head_dim });
            defer v3.deinit();

            var q_rope = try q3.rope(ctx, .seq, .d, &rope_table, .half);
            defer q_rope.deinit();
            var k_rope = try k3.rope(ctx, .seq, .d, &rope_table, .half);
            defer k_rope.deinit();

            // Write the chunk's K/V into the resident cache at write_offset.
            {
                const row = config.lm_n_kv_heads * config.lm_head_dim;
                const k_dst = (try cache.k[layer_i].data())[write_offset * row ..][0 .. n_new * row];
                const v_dst = (try cache.v[layer_i].data())[write_offset * row ..][0 .. n_new * row];
                @memcpy(k_dst, try k_rope.dataConst());
                @memcpy(v_dst, try v3.dataConst());
            }

            var k_view = try cache.k[layer_i].narrow(ctx, .seq, 0, kv_len);
            defer k_view.deinit();
            var v_view = try cache.v[layer_i].narrow(ctx, .seq, 0, kv_len);
            defer v_view.deinit();

            var attn = switch (mode) {
                .causal => try q_rope.groupedAttention(ctx, &k_view, &v_view, self.kv_head_for_head, .attn, scale, .{}),
                .biased => try q_rope.groupedAttention(ctx, &k_view, &v_view, self.kv_head_for_head, .attn, scale, .{ .mask = .bidirectional, .bias = &bias_tensor.? }),
            };
            defer attn.deinit();

            var attn_out = try layer.o_proj.linearSeq(ctx, &attn, .attn, .embed);
            defer attn_out.deinit();
            try x.addScaledInPlace(ctx, &attn_out, 1.0);

            var hn = try x.rmsNormMul(ctx, .embed, &layer.ffn_norm, config.lm_rms_eps);
            defer hn.deinit();
            var gate = try layer.gate_proj.linearSeq(ctx, &hn, .embed, .ffn);
            defer gate.deinit();
            var up = try layer.up_proj.linearSeq(ctx, &hn, .embed, .ffn);
            defer up.deinit();
            var gated = try up.swiglu(ctx, &gate);
            defer gated.deinit();
            var ffn_out = try layer.down_proj.linearSeq(ctx, &gated, .ffn, .embed);
            defer ffn_out.deinit();
            try x.addScaledInPlace(ctx, &ffn_out, 1.0);
        }
        return x;
    }

    /// Final RMSNorm + lm_head over the LAST `n_logits` rows of `hidden`;
    /// writes [n_logits, vocab] row-major into `logits_out`.
    fn projectLogits(self: *const Lm, ctx: *ExecContext, hidden: *const Seq, n_logits: usize, logits_out: []f32) !void {
        const n = hidden.dim(.seq);
        std.debug.assert(logits_out.len == n_logits * self.config.lm_vocab);
        var last = try hidden.narrow(ctx, .seq, n - n_logits, n_logits);
        defer last.deinit();
        var normed = try last.rmsNormMul(ctx, .embed, &self.output_norm, self.config.lm_rms_eps);
        defer normed.deinit();
        var logits = try self.output.linearSeq(ctx, &normed, .embed, .vocab);
        defer logits.deinit();
        if (self.output_tail) |*tail| {
            var tail_logits = try normed.dot(ctx, tail, .embed);
            defer tail_logits.deinit();
            var full = try logits.concat(ctx, .vocab, &.{&tail_logits});
            defer full.deinit();
            try full.copyTo(logits_out);
            return;
        }
        try logits.copyTo(logits_out);
    }

    /// Reference `run_resident_causal`: forward [n_new, hidden] at absolute
    /// position pos0 (plain causal), commit the K/V, and return the
    /// last-position logits in `logits_out` ([vocab]). Advances cache.len.
    pub fn forwardCausal(
        self: *const Lm,
        ctx: *ExecContext,
        cache: *Cache,
        x_host: []const f32,
        n_new: usize,
        pos0: usize,
        logits_out: []f32,
    ) !void {
        const allocator = ctx.allocator;
        const positions = try allocator.alloc(i32, n_new);
        defer allocator.free(positions);
        for (positions, 0..) |*p, i| p.* = @intCast(pos0 + i);

        var hidden = try self.forwardLayers(ctx, cache, x_host, n_new, positions, pos0, pos0 + n_new, .causal);
        defer hidden.deinit();
        try self.projectLogits(ctx, &hidden, 1, logits_out);
        cache.len = pos0 + n_new;
    }

    /// Reference `build_mtp_positions` (lm.cpp): recompute rows take
    /// consecutive absolute positions; the `block` slots take
    /// [base-1, base, .., base+block-2] (slot 0 duplicates the last committed
    /// token's own position).
    pub fn buildMtpPositions(allocator: Allocator, cached_len: usize, n_recompute: usize, block: usize) ![]i32 {
        const n_new = n_recompute + block;
        const base = cached_len + n_recompute;
        const positions = try allocator.alloc(i32, n_new);
        for (0..n_recompute) |i| positions[i] = @intCast(cached_len + i);
        for (0..block) |b| positions[n_recompute + b] = @intCast(base + b - 1);
        return positions;
    }

    /// Reference `build_mtp_mask` (lm.cpp; ported from
    /// mask_sdpa_utils.update_causal_mask_for_one_gen_window_2d): causal by
    /// absolute position, then (2) the block window fully bidirectional among
    /// its own keys and (3) the last committed token's key masked out for all
    /// block rows. 0 = visible, `mask_neg` = masked. [n_new, full] row-major.
    pub fn buildMtpMask(allocator: Allocator, cached_len: usize, n_recompute: usize, block: usize) ![]f32 {
        const n_new = n_recompute + block;
        const full = cached_len + n_new;
        const base = cached_len + n_recompute;
        const mask = try allocator.alloc(f32, n_new * full);
        for (0..n_new) |q| {
            const apos = if (q < n_recompute) cached_len + q else base + (q - n_recompute) - 1;
            for (0..full) |key| {
                mask[q * full + key] = if (key > apos) mask_neg else 0.0;
            }
        }
        for (n_new - block..n_new) |q| {
            for (full - block..full) |key| mask[q * full + key] = 0.0;
            mask[q * full + (full - block - 1)] = mask_neg;
        }
        return mask;
    }

    /// Reference `mtp_block_forward`: one MTP round over the resident cache.
    /// `x_host` is [n_recompute + block, hidden]; K/V scratch rows land at
    /// cache.len (NOT advanced). Writes the `block` block-position logits
    /// (position-major, [block, vocab]) into `logits_block_out`.
    pub fn forwardMtpBlock(
        self: *const Lm,
        ctx: *ExecContext,
        cache: *Cache,
        x_host: []const f32,
        n_recompute: usize,
        logits_block_out: []f32,
    ) !void {
        const allocator = ctx.allocator;
        const block = self.config.lm_block_size;
        const n_new = n_recompute + block;
        const cached_len = cache.len;
        const full = cached_len + n_new;

        const positions = try buildMtpPositions(allocator, cached_len, n_recompute, block);
        defer allocator.free(positions);
        const mask = try buildMtpMask(allocator, cached_len, n_recompute, block);
        defer allocator.free(mask);

        var hidden = try self.forwardLayers(ctx, cache, x_host, n_new, positions, cached_len, full, .{ .biased = mask });
        defer hidden.deinit();
        try self.projectLogits(ctx, &hidden, block, logits_block_out);
    }
};

test {
    _ = @import("lm_tests.zig");
}
