//! SHINE (arXiv 2602.06358): an in-context hypernetwork that maps a context
//! passage to a rank-8 LoRA over every linear of a frozen Qwen3 in ONE
//! forward pass — the amortized cousin of the cartridge mechanism: the
//! artifact is adapter weights instead of a trained KV prefix, and producing
//! it costs one prefill instead of a training run. Serving with the adapter
//! carries zero context tokens and zero extra KV.
//!
//! Pipeline (reference: refs/SHINE, checkpoint HF Yewei-Liu/SHINE-ift_mqa_1qa,
//! MIT; converted by tools/convert_shine.py, goldens by tools/gen_shine_goldens.py):
//!
//!   1. `encodeMemoryStates`: embed the context ids, append the learned
//!      memory embeddings, run the base model once with the rank-128 Meta
//!      LoRA active on all 7 linears of every layer (side path
//!      `base + (x @ A) @ B`, never merged), and capture the residual stream
//!      at the memory rows after EVERY layer -> (layer, mem, embed).
//!   2. `m2pForward`: post-LN transformer encoder layers (gelu-erf, biased,
//!      LayerNorm) alternating LAYER-mixing (attention along the layer axis,
//!      one sequence per memory token) and TOKEN-mixing (attention along the
//!      token axis, one sequence per layer), with learned layer/token
//!      positional embeddings added once at the input.
//!   3. `sliceLora` ("rl" method): scale the M2P output by sqrt(scale) and
//!      cut each layer's flattened (mem * embed) row into per-module
//!      A [in, r] / B [r, out] pairs in order q,k,v,o,gate,up,down.
//!   4. `forwardStep`/`greedy`: the standard decode loop with the GENERATED
//!      LoRA side paths on every linear (f16 KV cache).
//!
//! Everything here composes public facade ops and the exported qwen3 Layer
//! weights; model.zig is untouched.
const std = @import("std");
const fucina = @import("fucina");
const weights = @import("fucina").weights;
const decoder = @import("../decoder.zig");
const kv_cache = @import("../kv_cache.zig");
const qwen3 = @import("model.zig");
const cartridge_mod = @import("../cartridge.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;
const gguf = fucina.gguf;
const Tag = @TypeOf(.tag);

pub const Error = weights.Error || qwen3.Error || kv_cache.Error || error{
    InvalidShineConfig,
    /// SHINE decodes against the f16 KV representation only.
    UnsupportedKvDtype,
    /// The base model must be a dense Qwen3 (the reference adapts Qwen3-8B).
    MoeNotSupported,
};

pub const Config = struct {
    hidden_size: usize,
    num_layers: usize,
    num_mem_token: usize,
    metalora_r: usize,
    lora_r: usize,
    scale: f32,
    m2p_layers: usize,
    m2p_heads: usize,
    m2p_ffn: usize,
    m2p_eps: f32,
    layer_transformer_first: bool,
    /// Cartridge readout: > 0 selects the KV-prefix interpretation of the
    /// generated per-layer block — `cartridge_rows` prefix positions, each
    /// a K row and a V row (`M * hidden == 2 * rows * kv_dim`). 0 keeps
    /// the LoRA "rl" slicing.
    cartridge_rows: usize = 0,

    pub fn fromGguf(file: *const gguf.File) !Config {
        const arch = file.getString("general.architecture") orelse return Error.InvalidShineConfig;
        if (!std.mem.eql(u8, arch, "shine")) return Error.InvalidShineConfig;
        return .{
            .hidden_size = try metaInt(file, "shine.hidden_size"),
            .num_layers = try metaInt(file, "shine.num_layers"),
            .num_mem_token = try metaInt(file, "shine.num_mem_token"),
            .metalora_r = try metaInt(file, "shine.metalora_r"),
            .lora_r = try metaInt(file, "shine.lora_r"),
            .scale = @floatCast(file.getFloat("shine.scale") orelse return Error.InvalidShineConfig),
            .m2p_layers = try metaInt(file, "shine.m2p.num_layers"),
            .m2p_heads = try metaInt(file, "shine.m2p.head_count"),
            .m2p_ffn = try metaInt(file, "shine.m2p.feed_forward_length"),
            .m2p_eps = @floatCast(file.getFloat("shine.m2p.layer_norm_eps") orelse return Error.InvalidShineConfig),
            .cartridge_rows = if (file.getInt("shine.cartridge_rows")) |rows| @intCast(@max(rows, 0)) else 0,
            .layer_transformer_first = file.getBool("shine.m2p.layer_transformer_first") orelse true,
        };
    }

    pub fn validate(self: Config, base: qwen3.Config) !void {
        if (self.hidden_size != base.hidden_size) return Error.InvalidShineConfig;
        if (self.num_layers != base.num_layers) return Error.InvalidShineConfig;
        if (self.m2p_layers % 2 != 0) return Error.InvalidShineConfig;
        if (self.hidden_size % self.m2p_heads != 0) return Error.InvalidShineConfig;
        if (base.isMoe()) return Error.MoeNotSupported;
        if (self.cartridge_rows > 0) {
            // Cartridge readout: the budget is one K and one V row per
            // prefix position, per layer.
            const kv_dim = base.num_key_value_heads * base.head_dim;
            if (self.num_mem_token * self.hidden_size != 2 * self.cartridge_rows * kv_dim)
                return Error.InvalidShineConfig;
        } else {
            // The memory-token count IS the LoRA budget: M * hidden must
            // equal the per-layer generated-parameter count
            // (SHINE's M = ceil(rD/H)).
            if (self.num_mem_token * self.hidden_size != loraParamsPerLayer(base, self.lora_r))
                return Error.InvalidShineConfig;
        }
    }
};

fn metaInt(file: *const gguf.File, key: []const u8) !usize {
    const value = file.getInt(key) orelse return Error.InvalidShineConfig;
    if (value <= 0) return Error.InvalidShineConfig;
    return @intCast(value);
}

/// The 7 adapted linears of one dense qwen3 layer, in the reference's
/// generation/slicing order.
pub const Module = enum { q, k, v, o, gate, up, down };
pub const modules = [_]Module{ .q, .k, .v, .o, .gate, .up, .down };
/// GGUF-style tensor-name stems, index-aligned with `modules` (shared by
/// the Meta LoRA tensors and the standalone adapter artifact).
const module_gguf_names = [_][]const u8{ "attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate", "ffn_up", "ffn_down" };

/// {in, out} features of one adapted linear.
pub fn moduleDims(base: qwen3.Config, module: Module) [2]usize {
    return switch (module) {
        .q => .{ base.hidden_size, base.qProjectionDim() },
        .k => .{ base.hidden_size, base.kvProjectionDim() },
        .v => .{ base.hidden_size, base.kvProjectionDim() },
        .o => .{ base.qProjectionDim(), base.hidden_size },
        .gate => .{ base.hidden_size, base.intermediate_size },
        .up => .{ base.hidden_size, base.intermediate_size },
        .down => .{ base.intermediate_size, base.hidden_size },
    };
}

pub fn loraParamsPerLayer(base: qwen3.Config, r: usize) usize {
    var total: usize = 0;
    for (modules) |module| {
        const dims = moduleDims(base, module);
        total += r * (dims[0] + dims[1]);
    }
    return total;
}

/// One low-rank adapter: `delta(x) = (x @ a) @ b`, PyTorch layouts
/// (a: [in, r], b: [r, out]); any sqrt(scale) factor is baked into the
/// stored values, exactly like the reference.
pub const LoraPair = struct {
    a: fucina.Tensor(.{ .lin, .lora_r }),
    b: fucina.Tensor(.{ .lora_r, .lout }),

    pub fn deinit(self: *LoraPair) void {
        self.b.deinit();
        self.a.deinit();
        self.* = undefined;
    }
};

pub const LayerLora = struct {
    q: LoraPair,
    k: LoraPair,
    v: LoraPair,
    o: LoraPair,
    gate: LoraPair,
    up: LoraPair,
    down: LoraPair,

    pub fn pair(self: *const LayerLora, comptime module: Module) *const LoraPair {
        return &@field(self, @tagName(module));
    }

    pub fn deinit(self: *LayerLora) void {
        inline for (comptime std.meta.fieldNames(LayerLora)) |name| @field(self, name).deinit();
        self.* = undefined;
    }
};

/// A full per-layer adapter set — the Meta LoRA (loaded from the SHINE GGUF)
/// and generated adapters (produced by `sliceLora`) share this shape.
pub const LoraSet = struct {
    allocator: Allocator,
    layers: []LayerLora,

    pub fn deinit(self: *LoraSet) void {
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.* = undefined;
    }
};

/// One M2P encoder layer's weights (torch nn.TransformerEncoderLayer,
/// post-LN, biased throughout). Field tensors may be constants (inference)
/// or gradient-carrying variables (the trainer) — same type, same code.
pub const M2pBlock = struct {
    in_w: fucina.Tensor(.{ .proj, .embed }), // [3H, H] packed q;k;v
    in_b: fucina.Tensor(.{.proj}),
    out_w: fucina.Tensor(.{ .eout, .hmerge }), // [H, H]
    out_b: fucina.Tensor(.{.eout}),
    ff1_w: fucina.Tensor(.{ .m2pffn, .embed }), // [FF, H]
    ff1_b: fucina.Tensor(.{.m2pffn}),
    ff2_w: fucina.Tensor(.{ .eout, .m2pffn }), // [H, FF]
    ff2_b: fucina.Tensor(.{.eout}),
    norm1_w: fucina.Tensor(.{.embed}),
    norm1_b: fucina.Tensor(.{.embed}),
    norm2_w: fucina.Tensor(.{.embed}),
    norm2_b: fucina.Tensor(.{.embed}),

    fn load(ctx: *ExecContext, file: *const gguf.File, config: Config, i: usize) !M2pBlock {
        const hidden = config.hidden_size;
        var name_buf: [64]u8 = undefined;

        var in_w = try loadMatrix(ctx, file, try m2pName(&name_buf, i, "attn_in.weight"), 3 * hidden, hidden, .{ .proj, .embed });
        errdefer in_w.deinit();
        var in_b = try weights.loadVector(ctx, try file.get(try m2pName(&name_buf, i, "attn_in.bias")), 3 * hidden, .proj);
        errdefer in_b.deinit();
        var out_w = try loadMatrix(ctx, file, try m2pName(&name_buf, i, "attn_out.weight"), hidden, hidden, .{ .eout, .hmerge });
        errdefer out_w.deinit();
        var out_b = try weights.loadVector(ctx, try file.get(try m2pName(&name_buf, i, "attn_out.bias")), hidden, .eout);
        errdefer out_b.deinit();
        var ff1_w = try loadMatrix(ctx, file, try m2pName(&name_buf, i, "ffn_up.weight"), config.m2p_ffn, hidden, .{ .m2pffn, .embed });
        errdefer ff1_w.deinit();
        var ff1_b = try weights.loadVector(ctx, try file.get(try m2pName(&name_buf, i, "ffn_up.bias")), config.m2p_ffn, .m2pffn);
        errdefer ff1_b.deinit();
        var ff2_w = try loadMatrix(ctx, file, try m2pName(&name_buf, i, "ffn_down.weight"), hidden, config.m2p_ffn, .{ .eout, .m2pffn });
        errdefer ff2_w.deinit();
        var ff2_b = try weights.loadVector(ctx, try file.get(try m2pName(&name_buf, i, "ffn_down.bias")), hidden, .eout);
        errdefer ff2_b.deinit();
        var norm1_w = try weights.loadVector(ctx, try file.get(try m2pName(&name_buf, i, "norm1.weight")), hidden, .embed);
        errdefer norm1_w.deinit();
        var norm1_b = try weights.loadVector(ctx, try file.get(try m2pName(&name_buf, i, "norm1.bias")), hidden, .embed);
        errdefer norm1_b.deinit();
        var norm2_w = try weights.loadVector(ctx, try file.get(try m2pName(&name_buf, i, "norm2.weight")), hidden, .embed);
        errdefer norm2_w.deinit();
        var norm2_b = try weights.loadVector(ctx, try file.get(try m2pName(&name_buf, i, "norm2.bias")), hidden, .embed);
        errdefer norm2_b.deinit();

        return .{
            .in_w = in_w,
            .in_b = in_b,
            .out_w = out_w,
            .out_b = out_b,
            .ff1_w = ff1_w,
            .ff1_b = ff1_b,
            .ff2_w = ff2_w,
            .ff2_b = ff2_b,
            .norm1_w = norm1_w,
            .norm1_b = norm1_b,
            .norm2_w = norm2_w,
            .norm2_b = norm2_b,
        };
    }

    pub fn deinit(self: *M2pBlock, allocator: Allocator) void {
        _ = allocator;
        self.norm2_b.deinit();
        self.norm2_w.deinit();
        self.norm1_b.deinit();
        self.norm1_w.deinit();
        self.ff2_b.deinit();
        self.ff2_w.deinit();
        self.ff1_b.deinit();
        self.ff1_w.deinit();
        self.out_b.deinit();
        self.out_w.deinit();
        self.in_b.deinit();
        self.in_w.deinit();
        self.* = undefined;
    }
};

pub const Shine = struct {
    allocator: Allocator,
    config: Config,
    mem_tokens: fucina.Tensor(.{ .mem, .embed }),
    layer_pe: fucina.Tensor(.{ .layer, .embed }),
    token_pe: fucina.Tensor(.{ .mem, .embed }),
    m2p: []M2pBlock,
    metalora: LoraSet,

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, base: qwen3.Config) !Shine {
        var file = try gguf.File.loadMmap(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFile(ctx, &file, base);
    }

    pub fn loadGgufFromFile(ctx: *ExecContext, file: *const gguf.File, base: qwen3.Config) !Shine {
        const config = try Config.fromGguf(file);
        try config.validate(base);
        const allocator = ctx.allocator;

        var mem_tokens = try loadMatrix(ctx, file, "mem_tokens", config.num_mem_token, config.hidden_size, .{ .mem, .embed });
        errdefer mem_tokens.deinit();
        var layer_pe = try loadMatrix(ctx, file, "m2p.layer_pe", config.num_layers, config.hidden_size, .{ .layer, .embed });
        errdefer layer_pe.deinit();
        var token_pe = try loadMatrix(ctx, file, "m2p.token_pe", config.num_mem_token, config.hidden_size, .{ .mem, .embed });
        errdefer token_pe.deinit();

        const m2p = try allocator.alloc(M2pBlock, config.m2p_layers);
        var m2p_built: usize = 0;
        errdefer {
            for (m2p[0..m2p_built]) |*block| block.deinit(allocator);
            allocator.free(m2p);
        }
        for (m2p, 0..) |*block, i| {
            block.* = try M2pBlock.load(ctx, file, config, i);
            m2p_built += 1;
        }

        var metalora = try loadMetalora(ctx, file, base, config);
        errdefer metalora.deinit();

        return .{
            .allocator = allocator,
            .config = config,
            .mem_tokens = mem_tokens,
            .layer_pe = layer_pe,
            .token_pe = token_pe,
            .m2p = m2p,
            .metalora = metalora,
        };
    }

    pub fn deinit(self: *Shine) void {
        self.metalora.deinit();
        for (self.m2p) |*block| block.deinit(self.allocator);
        self.allocator.free(self.m2p);
        self.token_pe.deinit();
        self.layer_pe.deinit();
        self.mem_tokens.deinit();
        self.* = undefined;
    }
};

fn m2pName(buf: []u8, layer_i: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "m2p.{d}.{s}", .{ layer_i, suffix });
}

fn loadMatrix(ctx: *ExecContext, file: *const gguf.File, name: []const u8, rows: usize, cols: usize, comptime tags: anytype) !fucina.Tensor(tags) {
    const info = try file.get(name);
    const shape = try info.logicalMatrixShape();
    if (shape[0] != rows or shape[1] != cols) return Error.InvalidShineConfig;
    var out = try fucina.Tensor(tags).empty(ctx, .{ rows, cols });
    errdefer out.deinit();
    try weights.fillF32(try out.data(), info);
    return out;
}

/// `prefix` is a comptime format stem ("metalora" / "blk"); tensor names
/// are `<prefix>.<layer>.<module>.{a,b}`.
fn loadPair(ctx: *ExecContext, file: *const gguf.File, base: qwen3.Config, r: usize, layer_i: usize, comptime prefix: []const u8, comptime module: Module, gguf_name: []const u8) !LoraPair {
    const dims = moduleDims(base, module);
    var name_buf: [64]u8 = undefined;
    var a = try loadMatrix(ctx, file, try std.fmt.bufPrint(&name_buf, prefix ++ ".{d}.{s}.a", .{ layer_i, gguf_name }), dims[0], r, .{ .lin, .lora_r });
    errdefer a.deinit();
    var b = try loadMatrix(ctx, file, try std.fmt.bufPrint(&name_buf, prefix ++ ".{d}.{s}.b", .{ layer_i, gguf_name }), r, dims[1], .{ .lora_r, .lout });
    errdefer b.deinit();
    return .{ .a = a, .b = b };
}

fn loadLoraSet(ctx: *ExecContext, file: *const gguf.File, base: qwen3.Config, r: usize, num_layers: usize, comptime prefix: []const u8) !LoraSet {
    const allocator = ctx.allocator;
    const layers = try allocator.alloc(LayerLora, num_layers);
    var loaded: usize = 0;
    errdefer {
        for (layers[0..loaded]) |*layer| layer.deinit();
        allocator.free(layers);
    }
    for (layers, 0..) |*layer, layer_i| {
        var q = try loadPair(ctx, file, base, r, layer_i, prefix, .q, "attn_q");
        errdefer q.deinit();
        var k = try loadPair(ctx, file, base, r, layer_i, prefix, .k, "attn_k");
        errdefer k.deinit();
        var v = try loadPair(ctx, file, base, r, layer_i, prefix, .v, "attn_v");
        errdefer v.deinit();
        var o = try loadPair(ctx, file, base, r, layer_i, prefix, .o, "attn_o");
        errdefer o.deinit();
        var gate = try loadPair(ctx, file, base, r, layer_i, prefix, .gate, "ffn_gate");
        errdefer gate.deinit();
        var up = try loadPair(ctx, file, base, r, layer_i, prefix, .up, "ffn_up");
        errdefer up.deinit();
        const down = try loadPair(ctx, file, base, r, layer_i, prefix, .down, "ffn_down");
        layer.* = .{ .q = q, .k = k, .v = v, .o = o, .gate = gate, .up = up, .down = down };
        loaded += 1;
    }
    return .{ .allocator = allocator, .layers = layers };
}

fn loadMetalora(ctx: *ExecContext, file: *const gguf.File, base: qwen3.Config, config: Config) !LoraSet {
    return loadLoraSet(ctx, file, base, config.metalora_r, config.num_layers, "metalora");
}

/// Serialize an adapter as a standalone `shine-adapter` GGUF: generate once,
/// serve forever — the reload path needs neither the SHINE weights nor the
/// hypernetwork pass. Tensors are `blk.<layer>.<module>.{a,b}` f32 in the
/// stored (PyTorch) layouts; the sqrt(scale) factor is already baked in.
pub fn saveLora(set: *const LoraSet, base: qwen3.Config, allocator: Allocator, out: *std.Io.Writer) !void {
    if (set.layers.len == 0) return Error.InvalidShineConfig;
    const r = set.layers[0].q.a.dim(.lora_r);
    var writer = gguf.Writer.init(allocator);
    defer writer.deinit();
    try writer.addMetaString("general.architecture", "shine-adapter");
    try writer.addMetaInt("shine.hidden_size", u32, @intCast(base.hidden_size));
    try writer.addMetaInt("shine.num_layers", u32, @intCast(set.layers.len));
    try writer.addMetaInt("shine.lora_r", u32, @intCast(r));
    for (set.layers, 0..) |*layer, layer_i| {
        inline for (modules, module_gguf_names) |module, gguf_name| {
            const pair = layer.pair(module);
            const dims = moduleDims(base, module);
            var name_buf: [64]u8 = undefined;
            // GGUF dims are innermost-first: a is [in, r] row-major, b [r, out].
            try writer.addTensor(
                try std.fmt.bufPrint(&name_buf, "blk.{d}.{s}.a", .{ layer_i, gguf_name }),
                .f32,
                &.{ r, dims[0] },
                std.mem.sliceAsBytes(try pair.a.dataConst()),
            );
            try writer.addTensor(
                try std.fmt.bufPrint(&name_buf, "blk.{d}.{s}.b", .{ layer_i, gguf_name }),
                .f32,
                &.{ dims[1], r },
                std.mem.sliceAsBytes(try pair.b.dataConst()),
            );
        }
    }
    try writer.finish(out);
}

/// Reload a saved adapter, cross-checked against the base model's geometry.
pub fn loadLoraFromFile(ctx: *ExecContext, file: *const gguf.File, base: qwen3.Config) !LoraSet {
    const arch = file.getString("general.architecture") orelse return Error.InvalidShineConfig;
    if (!std.mem.eql(u8, arch, "shine-adapter")) return Error.InvalidShineConfig;
    if (try metaInt(file, "shine.hidden_size") != base.hidden_size) return Error.InvalidShineConfig;
    const num_layers = try metaInt(file, "shine.num_layers");
    if (num_layers != base.num_layers) return Error.InvalidShineConfig;
    const r = try metaInt(file, "shine.lora_r");
    return loadLoraSet(ctx, file, base, r, num_layers, "blk");
}

pub fn loadLoraGguf(ctx: *ExecContext, io: std.Io, path: []const u8, base: qwen3.Config) !LoraSet {
    var file = try gguf.File.loadMmap(ctx.allocator, io, path);
    defer file.deinit();
    return loadLoraFromFile(ctx, &file, base);
}

/// `(x @ a) @ b` re-tagged to `{ .seq, out_tag }` — the LoRA side path added
/// to a base projection's output. `x` is any `{ .seq, * }` activation.
/// Differentiable end to end when the operands carry grad state (the
/// trainer's route); shared with the no-grad inference path.
pub fn loraDelta(ctx: *ExecContext, x: anytype, pair: *const LoraPair, comptime out_tag: Tag) !fucina.Tensor(.{ .seq, out_tag }) {
    var xr = try x.withTags(ctx, .{ .seq, .lin });
    defer xr.deinit();
    var mid = try xr.dot(ctx, &pair.a, .lin);
    defer mid.deinit();
    var out = try mid.dot(ctx, &pair.b, .lora_r);
    defer out.deinit();
    return out.withTags(ctx, .{ .seq, out_tag });
}

fn addLora(ctx: *ExecContext, base: anytype, x: anytype, pair: *const LoraPair, comptime out_tag: Tag) !@TypeOf(base.*) {
    var delta = try loraDelta(ctx, x, pair, out_tag);
    defer delta.deinit();
    return base.add(ctx, &delta);
}

/// One dense qwen3 layer with LoRA side paths on all 7 linears — the
/// reference LoraQwen3DecoderLayer, over the exported Layer weights. With
/// `kv` null this is the full-sequence (encoder) form; with a cache it is
/// the decode form (f16 KV only).
fn loraLayerForward(
    ctx: *ExecContext,
    base: qwen3.Config,
    layer: *const qwen3.Layer,
    lora: *const LayerLora,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    rope_table: *const fucina.RopeTable,
    kv_head_for_head: []const usize,
    kv: ?*KvCache,
    layer_i: usize,
) !fucina.Tensor(.{ .seq, .embed }) {
    const attn_scale = 1 / @sqrt(@as(f32, @floatFromInt(base.head_dim)));

    var normed = try input.rmsNormMul(ctx, .embed, &layer.attn_norm, base.rms_norm_eps);
    defer normed.deinit();

    var qkv = switch (layer.attn_proj) {
        .separate => |*separate| blk: {
            var q = try separate.q_proj.linearSeq(ctx, &normed, .embed, .q);
            errdefer q.deinit();
            var k = try separate.k_proj.linearSeq(ctx, &normed, .embed, .k);
            errdefer k.deinit();
            const v = try separate.v_proj.linearSeq(ctx, &normed, .embed, .v);
            break :blk qwen3.QkvProjection{ .q = q, .k = k, .v = v };
        },
        .fused => |*weight| blk: {
            var fused = try weight.linearSeq(ctx, &normed, .embed, .qkv);
            defer fused.deinit();
            break :blk try qwen3.splitQkv(ctx, &fused, base);
        },
    };
    defer qkv.deinit();
    qkv.q = try ctx.replace(qkv.q, addLora(ctx, &qkv.q, &normed, lora.pair(.q), .q));
    qkv.k = try ctx.replace(qkv.k, addLora(ctx, &qkv.k, &normed, lora.pair(.k), .k));
    qkv.v = try ctx.replace(qkv.v, addLora(ctx, &qkv.v, &normed, lora.pair(.v), .v));

    var q3 = try qkv.q.split(ctx, .q, .{ .head, .d }, .{ base.num_attention_heads, base.head_dim });
    defer q3.deinit();
    var k3 = try qkv.k.split(ctx, .k, .{ .kv_head, .d }, .{ base.num_key_value_heads, base.head_dim });
    defer k3.deinit();
    var v3 = try qkv.v.split(ctx, .v, .{ .kv_head, .d }, .{ base.num_key_value_heads, base.head_dim });
    defer v3.deinit();

    var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, base.rms_norm_eps, rope_table);
    defer q_rope.deinit();
    var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm, base.rms_norm_eps, rope_table);
    defer k_rope.deinit();

    var attn = if (kv) |cache| blk: {
        if (cache.dtype != .f16) return Error.UnsupportedKvDtype;
        try cache.appendLayer(ctx, layer_i, &k_rope, &v3);
        const cached_len = cache.len() + k_rope.dim(.seq);
        var k_view = try cache.k[layer_i].narrow(ctx, .seq, 0, cached_len);
        defer k_view.deinit();
        var v_view = try cache.v[layer_i].narrow(ctx, .seq, 0, cached_len);
        defer v_view.deinit();
        break :blk try q_rope.groupedAttention(ctx, &k_view, &v_view, kv_head_for_head, .attn, attn_scale, .{});
    } else try q_rope.groupedAttention(ctx, &k_rope, &v3, kv_head_for_head, .attn, attn_scale, .{});
    defer attn.deinit();

    var attn_out = try layer.o_proj.linearSeq(ctx, &attn, .attn, .embed);
    defer attn_out.deinit();
    attn_out = try ctx.replace(attn_out, addLora(ctx, &attn_out, &attn, lora.pair(.o), .embed));

    var with_attn = try input.add(ctx, &attn_out);
    defer with_attn.deinit();

    var ffn_in = try with_attn.rmsNormMul(ctx, .embed, &layer.ffn_norm, base.rms_norm_eps);
    defer ffn_in.deinit();

    const dense = switch (layer.ffn) {
        .dense => |*d| d,
        .moe => return Error.MoeNotSupported,
    };
    var gate_up = switch (dense.input_proj) {
        .separate => |*separate| blk: {
            var gate = try separate.gate_proj.linearSeq(ctx, &ffn_in, .embed, .ffn);
            errdefer gate.deinit();
            const up = try separate.up_proj.linearSeq(ctx, &ffn_in, .embed, .ffn);
            break :blk qwen3.GateUpProjection{ .gate = gate, .up = up };
        },
        .fused => |*weight| blk: {
            var fused = try weight.linearSeq(ctx, &ffn_in, .embed, .gate_up);
            defer fused.deinit();
            break :blk try qwen3.splitGateUp(ctx, &fused, base);
        },
    };
    defer gate_up.deinit();
    gate_up.gate = try ctx.replace(gate_up.gate, addLora(ctx, &gate_up.gate, &ffn_in, lora.pair(.gate), .ffn));
    gate_up.up = try ctx.replace(gate_up.up, addLora(ctx, &gate_up.up, &ffn_in, lora.pair(.up), .ffn));

    var gated = try gate_up.up.swiglu(ctx, &gate_up.gate);
    defer gated.deinit();

    var down = try dense.down_proj.linearSeq(ctx, &gated, .ffn, .embed);
    defer down.deinit();
    down = try ctx.replace(down, addLora(ctx, &down, &gated, lora.pair(.down), .embed));

    return with_attn.add(ctx, &down);
}

/// Stage 1: context ids -> per-layer memory states `(layer, mem, embed)`.
/// One causal full-sequence pass with the Meta LoRA; no KV cache.
pub fn encodeMemoryStates(
    model: *const qwen3.Model,
    sh: *const Shine,
    ctx: *ExecContext,
    token_ids: []const usize,
) !fucina.Tensor(.{ .layer, .mem, .embed }) {
    if (token_ids.len == 0) return qwen3.Error.InvalidSequenceLength;
    const base = model.config;
    const mem = sh.config.num_mem_token;
    const hidden = base.hidden_size;
    const total_len = token_ids.len + mem;

    var rope_table = try ctx.prepareRopeTableRange(.{ .len = total_len }, base.head_dim, base.rope_theta, false);
    defer rope_table.deinit();

    var embedded = try model.token_embedding.getRowsAs(ctx, token_ids, .embed);
    var x = blk: {
        defer embedded.deinit();
        var mem_rows = try sh.mem_tokens.withTags(ctx, .{ .seq, .embed });
        defer mem_rows.deinit();
        break :blk try embedded.concat(ctx, .seq, &.{&mem_rows});
    };
    var x_released = false;
    errdefer if (!x_released) x.deinit();

    const capture = try ctx.allocator.alloc(f32, base.num_layers * mem * hidden);
    defer ctx.allocator.free(capture);

    for (model.layers, 0..) |*layer, layer_i| {
        x = try ctx.replace(x, loraLayerForward(ctx, base, layer, &sh.metalora.layers[layer_i], &x, &rope_table, model.kv_head_for_head, null, layer_i));
        var mem_view = try x.narrow(ctx, .seq, token_ids.len, mem);
        defer mem_view.deinit();
        try mem_view.copyTo(capture[layer_i * mem * hidden ..][0 .. mem * hidden]);
    }
    x.deinit();
    x_released = true;

    return fucina.Tensor(.{ .layer, .mem, .embed }).fromSlice(ctx, .{ base.num_layers, mem, hidden }, capture);
}

/// One M2P encoder layer (torch TransformerEncoderLayer, post-LN, gelu-erf),
/// attending along `.layer` (layer-mixing) or `.mem` (token-mixing) with the
/// other axis as batch — both passes fully batched through tag-aligned dots.
fn m2pEncoderLayer(
    config: Config,
    ctx: *ExecContext,
    block: *const M2pBlock,
    x: *const fucina.Tensor(.{ .layer, .mem, .embed }),
    comptime attend_layer: bool,
) !fucina.Tensor(.{ .layer, .mem, .embed }) {
    const hidden = config.hidden_size;
    const heads = config.m2p_heads;
    const head_dim = hidden / heads;
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    var qkv = try x.dot(ctx, &block.in_w, .embed);
    defer qkv.deinit();
    qkv = try ctx.replace(qkv, qkv.add(ctx, &block.in_b));

    var q_slice = try qkv.narrow(ctx, .proj, 0, hidden);
    defer q_slice.deinit();
    var k_slice = try qkv.narrow(ctx, .proj, hidden, hidden);
    defer k_slice.deinit();
    var v_slice = try qkv.narrow(ctx, .proj, 2 * hidden, hidden);
    defer v_slice.deinit();

    var q = try q_slice.split(ctx, .proj, .{ .head, .d }, .{ heads, head_dim });
    defer q.deinit();
    var k = try k_slice.split(ctx, .proj, .{ .head, .d }, .{ heads, head_dim });
    defer k.deinit();
    var v = try v_slice.split(ctx, .proj, .{ .head, .d }, .{ heads, head_dim });
    defer v.deinit();

    const kv_tags = comptime if (attend_layer) .{ .layer2, .mem, .head, .d } else .{ .layer, .mem2, .head, .d };
    const attend_tag: Tag = comptime if (attend_layer) .layer2 else .mem2;
    var k2 = try k.withTags(ctx, kv_tags);
    defer k2.deinit();
    var v2 = try v.withTags(ctx, kv_tags);
    defer v2.deinit();

    var scores = try q.dot(ctx, &k2, .d);
    defer scores.deinit();
    var probs = try scores.softmax(ctx, attend_tag, .{ .scale = scale });
    defer probs.deinit();
    var attn = try probs.dot(ctx, &v2, attend_tag);
    defer attn.deinit();
    var attn_p = try attn.permuteTo(ctx, .{ .layer, .mem, .head, .d });
    defer attn_p.deinit();
    // The permuted head/d axes are not memory-adjacent, so the head merge
    // is not view-legal — materialize first.
    var attn_c = try attn_p.contiguous(ctx);
    defer attn_c.deinit();
    var merged = try attn_c.merge(ctx, .hmerge, .{ .head, .d });
    defer merged.deinit();

    var sa = try merged.dot(ctx, &block.out_w, .hmerge);
    defer sa.deinit();
    sa = try ctx.replace(sa, sa.add(ctx, &block.out_b));
    var sa_e = try sa.withTags(ctx, .{ .layer, .mem, .embed });
    defer sa_e.deinit();

    var post_attn = try x.add(ctx, &sa_e);
    defer post_attn.deinit();
    var normed1 = try post_attn.layerNorm(ctx, .embed, config.m2p_eps, .{ .weight = &block.norm1_w, .bias = &block.norm1_b });
    defer normed1.deinit();

    var ff = try normed1.dot(ctx, &block.ff1_w, .embed);
    defer ff.deinit();
    ff = try ctx.replace(ff, ff.add(ctx, &block.ff1_b));
    ff = try ctx.replace(ff, ff.unary(ctx, .gelu_erf));
    var ff_out = try ff.dot(ctx, &block.ff2_w, .m2pffn);
    defer ff_out.deinit();
    ff_out = try ctx.replace(ff_out, ff_out.add(ctx, &block.ff2_b));
    var ff_e = try ff_out.withTags(ctx, .{ .layer, .mem, .embed });
    defer ff_e.deinit();

    var post_ff = try normed1.add(ctx, &ff_e);
    defer post_ff.deinit();
    return post_ff.layerNorm(ctx, .embed, config.m2p_eps, .{ .weight = &block.norm2_w, .bias = &block.norm2_b });
}

/// The M2P input: memory states plus the learned layer/token positional
/// embeddings (`ms + layer_pe[:, None, :] + token_pe`). The two PEs combine
/// into one (layer, mem, embed) addend up front — a plain same-shape add,
/// no cross-axis broadcast. Split out (with `m2pStage`) so the golden test
/// can walk the reference's per-stage probes.
pub fn m2pInput(
    sh: *const Shine,
    ctx: *ExecContext,
    memory_states: *const fucina.Tensor(.{ .layer, .mem, .embed }),
) !fucina.Tensor(.{ .layer, .mem, .embed }) {
    const layers = sh.config.num_layers;
    const mem = sh.config.num_mem_token;
    const hidden = sh.config.hidden_size;
    const pe = try ctx.allocator.alloc(f32, layers * mem * hidden);
    defer ctx.allocator.free(pe);
    const layer_pe = try sh.layer_pe.dataConst();
    const token_pe = try sh.token_pe.dataConst();
    for (0..layers) |l| {
        for (0..mem) |m| {
            const row = pe[(l * mem + m) * hidden ..][0..hidden];
            for (row, layer_pe[l * hidden ..][0..hidden], token_pe[m * hidden ..][0..hidden]) |*dst, lp, tp| dst.* = lp + tp;
        }
    }
    var pe_tensor = try fucina.Tensor(.{ .layer, .mem, .embed }).fromSlice(ctx, .{ layers, mem, hidden }, pe);
    defer pe_tensor.deinit();
    return memory_states.add(ctx, &pe_tensor);
}

/// One M2P stage: encoder layer `i` with its alternation choice
/// (layer_transformer_first: even passes mix along the layer axis).
pub fn m2pStage(
    config: Config,
    m2p: []const M2pBlock,
    ctx: *ExecContext,
    i: usize,
    x: *const fucina.Tensor(.{ .layer, .mem, .embed }),
) !fucina.Tensor(.{ .layer, .mem, .embed }) {
    const layer_pass = (i % 2 == 0) == config.layer_transformer_first;
    return if (layer_pass)
        m2pEncoderLayer(config, ctx, &m2p[i], x, true)
    else
        m2pEncoderLayer(config, ctx, &m2p[i], x, false);
}

/// Stage 2: memory states -> the flat generated-parameter tensor (still
/// `(layer, mem, embed)`; `sliceLora` cuts it into adapters).
pub fn m2pForward(
    sh: *const Shine,
    ctx: *ExecContext,
    memory_states: *const fucina.Tensor(.{ .layer, .mem, .embed }),
) !fucina.Tensor(.{ .layer, .mem, .embed }) {
    var x = try m2pInput(sh, ctx, memory_states);
    errdefer x.deinit();
    for (0..sh.m2p.len) |i| {
        x = try ctx.replace(x, m2pStage(sh.config, sh.m2p, ctx, i, &x));
    }
    return x;
}

/// Stage 3 ("rl"): scale by sqrt(scale) once (A and B both carry sqrt(s) in
/// the reference), then per layer cut the flattened (mem * embed) row into
/// A [in, r] / B [r, out] pairs in module order q,k,v,o,gate,up,down.
pub fn sliceLora(
    ctx: *ExecContext,
    base: qwen3.Config,
    r: usize,
    scale: f32,
    plain: *const fucina.Tensor(.{ .layer, .mem, .embed }),
) !LoraSet {
    const allocator = ctx.allocator;

    var scaled = try plain.scale(ctx, @sqrt(scale));
    defer scaled.deinit();

    const layers = try allocator.alloc(LayerLora, base.num_layers);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*layer| layer.deinit();
        allocator.free(layers);
    }
    for (layers, 0..) |*layer, layer_i| {
        var row = try scaled.narrow(ctx, .layer, layer_i, 1);
        defer row.deinit();
        const values = try row.dataConst();
        var at: usize = 0;
        inline for (modules) |module| {
            const dims = moduleDims(base, module);
            var a = try fucina.Tensor(.{ .lin, .lora_r }).fromSlice(ctx, .{ dims[0], r }, values[at .. at + dims[0] * r]);
            errdefer a.deinit();
            at += dims[0] * r;
            var b = try fucina.Tensor(.{ .lora_r, .lout }).fromSlice(ctx, .{ r, dims[1] }, values[at .. at + r * dims[1]]);
            errdefer b.deinit();
            at += r * dims[1];
            @field(layer, @tagName(module)) = .{ .a = a, .b = b };
        }
        std.debug.assert(at == values.len);
        built += 1;
    }
    return .{ .allocator = allocator, .layers = layers };
}

/// The full hypernetwork: context ids -> generated adapter, one pass.
pub fn generateAdapter(
    model: *const qwen3.Model,
    sh: *const Shine,
    ctx: *ExecContext,
    token_ids: []const usize,
) !LoraSet {
    var memory_states = try encodeMemoryStates(model, sh, ctx, token_ids);
    defer memory_states.deinit();
    var plain = try m2pForward(sh, ctx, &memory_states);
    defer plain.deinit();
    return sliceLora(ctx, model.config, sh.config.lora_r, sh.config.scale, &plain);
}

/// Cartridge ("KV-prefix") readout of the generated block: each layer's
/// flat `M * hidden` row is `rows` K rows then `rows` V rows (kv-cache row
/// order `[rows, kv_heads * head_dim]`), both scaled by `sqrt(scale)` (the
/// "rl" discipline: generation starts near zero, so the prefix begins with
/// near-uniform logits and near-zero value content). The result is a
/// STANDARD `llm.cartridge.Cartridge` (leaf rows, registry, no sink):
/// save it with `saveState`, serve it with `--cartridge`/fleets, evaluate
/// it exactly like a distilled one. Prefix K rows are read as post-
/// norm/rope kv-cache rows at positions `0..rows-1` — the cartridge
/// serving contract (tokens shift to offset `rows`).
pub fn sliceCartridge(
    ctx: *ExecContext,
    allocator: Allocator,
    base: qwen3.Config,
    config: Config,
    plain: *const fucina.Tensor(.{ .layer, .mem, .embed }),
) !cartridge_mod.Cartridge {
    if (config.cartridge_rows == 0) return Error.InvalidShineConfig;
    const rows = config.cartridge_rows;
    const kv_dim = base.num_key_value_heads * base.head_dim;

    var scaled = try plain.scale(ctx, @sqrt(config.scale));
    defer scaled.deinit();
    const values = try scaled.dataConst();
    const per_layer = 2 * rows * kv_dim;

    const k_rows = try allocator.alloc([]const f32, base.num_layers);
    defer allocator.free(k_rows);
    const v_rows = try allocator.alloc([]const f32, base.num_layers);
    defer allocator.free(v_rows);
    for (0..base.num_layers) |layer_i| {
        const layer_base = layer_i * per_layer;
        k_rows[layer_i] = values[layer_base..][0 .. rows * kv_dim];
        v_rows[layer_i] = values[layer_base + rows * kv_dim ..][0 .. rows * kv_dim];
    }
    return cartridge_mod.Cartridge.initFromRows(ctx, allocator, 0, rows, base.num_key_value_heads, base.head_dim, k_rows, v_rows);
}

/// The full hypernetwork with the cartridge readout: context ids -> a
/// standard KV-prefix cartridge, one pass (the amortized counterpart of
/// the distillation run that normally produces one).
pub fn generateCartridge(
    model: *const qwen3.Model,
    sh: *const Shine,
    ctx: *ExecContext,
    allocator: Allocator,
    token_ids: []const usize,
) !cartridge_mod.Cartridge {
    var memory_states = try encodeMemoryStates(model, sh, ctx, token_ids);
    defer memory_states.deinit();
    var plain = try m2pForward(sh, ctx, &memory_states);
    defer plain.deinit();
    return sliceCartridge(ctx, allocator, model.config, sh.config, &plain);
}

/// `qwen3.Model.forwardStep` with a LoRA set active on every linear: appends
/// the tokens' K/V to `kv` and returns the LAST position's logits.
pub fn forwardStep(
    model: *const qwen3.Model,
    lora: *const LoraSet,
    ctx: *ExecContext,
    kv: *KvCache,
    token_ids: []const usize,
    pos0: usize,
) !fucina.Tensor(.{ .seq, .vocab }) {
    return forwardStepImpl(model, lora, ctx, kv, token_ids, pos0, true);
}

/// `qwen3.Model.forwardStepAllLogits` with a LoRA set active — logits for
/// EVERY appended position (the speculative-verify entry).
pub fn forwardStepAllLogits(
    model: *const qwen3.Model,
    lora: *const LoraSet,
    ctx: *ExecContext,
    kv: *KvCache,
    token_ids: []const usize,
    pos0: usize,
) !fucina.Tensor(.{ .seq, .vocab }) {
    return forwardStepImpl(model, lora, ctx, kv, token_ids, pos0, false);
}

fn forwardStepImpl(
    model: *const qwen3.Model,
    lora: *const LoraSet,
    ctx: *ExecContext,
    kv: *KvCache,
    token_ids: []const usize,
    pos0: usize,
    last_only: bool,
) !fucina.Tensor(.{ .seq, .vocab }) {
    if (token_ids.len == 0) return qwen3.Error.InvalidSequenceLength;
    if (kv.len() != pos0) return qwen3.Error.InvalidSequenceLength;
    if (kv.len() + token_ids.len > kv.capacity) return kv_cache.Error.KvCacheOverflow;
    const base = model.config;

    var rope_table = try ctx.prepareRopeTableRange(.{ .origin = @intCast(pos0), .len = token_ids.len }, base.head_dim, base.rope_theta, false);
    defer rope_table.deinit();

    var x = try model.token_embedding.getRowsAs(ctx, token_ids, .embed);
    var x_released = false;
    errdefer if (!x_released) x.deinit();
    for (model.layers, 0..) |*layer, layer_i| {
        x = try ctx.replace(x, loraLayerForward(ctx, base, layer, &lora.layers[layer_i], &x, &rope_table, model.kv_head_for_head, kv, layer_i));
    }
    kv.advance(token_ids.len);

    var final_norm = try x.rmsNormMul(ctx, .embed, &model.output_norm, base.rms_norm_eps);
    defer final_norm.deinit();
    x.deinit();
    x_released = true;

    const keep_from = if (last_only) final_norm.dim(.seq) - 1 else 0;
    var head_in = try final_norm.narrow(ctx, .seq, keep_from, final_norm.dim(.seq) - keep_from);
    defer head_in.deinit();
    return model.output.linearSeq(ctx, &head_in, .embed, .vocab);
}

/// File-scope aliases so AdaptedModel's methods can reach the free
/// functions they shadow by name.
const loraForwardStep = forwardStep;
const loraForwardStepAllLogits = forwardStepAllLogits;

/// The frozen base plus a SWAPPABLE adapter behind the duck-typed model
/// surface `chat.Conversation` consumes (`config.vocab_size`,
/// `initCache`, `forwardStep`) — the serving seam for adapter fleets:
/// one box serves every request, and the single inference worker points
/// `adapter` at the request's selection before decoding (lmserve's
/// scheduler owns the backend from exactly one thread, so a plain field
/// swap is safe). `null` decodes the plain base. Deliberately NO
/// `forwardStepBatch`: batch mode is per-slot-heterogeneous and a single
/// box holds one adapter at a time.
pub const AdaptedModel = struct {
    /// Decoder-contract decode state (`llm.decoder`): the shared KV cache.
    pub const Cache = KvCache;
    /// Decoder-contract capabilities: rewind rides the shared cache;
    /// deliberately NO batch entry (see above).
    pub const caps: decoder.Caps = .{ .rewind = true, .batch = false };

    base: *const qwen3.Model,
    adapter: ?*const LoraSet = null,
    config: qwen3.Config,

    pub fn init(base: *const qwen3.Model) AdaptedModel {
        return .{ .base = base, .config = base.config };
    }

    pub fn initCache(self: *const AdaptedModel, ctx: *ExecContext, capacity: usize) !KvCache {
        return self.base.initCache(ctx, capacity);
    }

    pub fn forwardStep(
        self: *const AdaptedModel,
        ctx: *ExecContext,
        kv: *KvCache,
        token_ids: []const usize,
        pos0: usize,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        if (self.adapter) |set| return loraForwardStep(self.base, set, ctx, kv, token_ids, pos0);
        return self.base.forwardStep(ctx, kv, token_ids, pos0);
    }

    pub fn forwardStepAllLogits(
        self: *const AdaptedModel,
        ctx: *ExecContext,
        kv: *KvCache,
        token_ids: []const usize,
        pos0: usize,
    ) !fucina.Tensor(.{ .seq, .vocab }) {
        if (self.adapter) |set| return loraForwardStepAllLogits(self.base, set, ctx, kv, token_ids, pos0);
        return self.base.forwardStepAllLogits(ctx, kv, token_ids, pos0);
    }
};

pub const GreedyOptions = struct {
    max_new_tokens: usize,
    stop_token: ?usize = null,
    /// Restrict argmax to the first `vocab_limit` logits. The reference
    /// resizes the head to len(tokenizer) (151,672 for the released run,
    /// below the GGUF's padded 151,936 rows), so parity needs the same cut.
    vocab_limit: ?usize = null,
};

/// Greedy decode with a LoRA set active (generate.zig's loop shape).
/// Resets `kv`. Returns the number of tokens written.
pub fn greedy(
    model: *const qwen3.Model,
    lora: *const LoraSet,
    ctx: *ExecContext,
    kv: *KvCache,
    prompt_tokens: []const usize,
    out_tokens: []usize,
    options: GreedyOptions,
) !usize {
    if (prompt_tokens.len == 0) return qwen3.Error.InvalidSequenceLength;
    kv.reset();

    var logits = try forwardStep(model, lora, ctx, kv, prompt_tokens, 0);
    defer logits.deinit();

    const limit = @min(options.max_new_tokens, out_tokens.len);
    var produced: usize = 0;
    while (produced < limit) {
        const next = try argmaxLast(ctx, &logits, options.vocab_limit);
        out_tokens[produced] = next;
        produced += 1;
        if (options.stop_token) |stop| if (next == stop) break;
        if (produced == limit) break;
        // Allocate the next step before freeing the current logits, so an
        // error here leaves `logits` valid for the function-scope defer.
        const fresh = try forwardStep(model, lora, ctx, kv, &.{next}, kv.len());
        logits.deinit();
        logits = fresh;
    }
    return produced;
}

fn argmaxLast(ctx: *ExecContext, logits: *const fucina.Tensor(.{ .seq, .vocab }), vocab_limit: ?usize) !usize {
    var last = try logits.narrow(ctx, .seq, logits.dim(.seq) - 1, 1);
    defer last.deinit();
    var limited: ?fucina.Tensor(.{ .seq, .vocab }) = null;
    defer if (limited) |*value| value.deinit();
    if (vocab_limit) |cut| {
        if (cut < last.dim(.vocab)) limited = try last.narrow(ctx, .vocab, 0, cut);
    }
    const scan: *const fucina.Tensor(.{ .seq, .vocab }) = if (limited) |*value| value else &last;
    var index = try scan.argmax(ctx, .vocab);
    defer index.deinit();
    return @intCast(try index.item());
}

test {
    _ = @import("shine_tests.zig");
    _ = @import("shine_golden_tests.zig");
}
