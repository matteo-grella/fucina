//! Shared GGUF-family load helpers: PTQTP-aware projection loading, the
//! dense-FFN weight containers (gate/up, fused when layouts allow, plus
//! down), the MoE expert-trio loader (streamed or resident), the
//! embed/output-norm/lm-head trio, and the GQA head map. Plain structs
//! and functions — every family keeps its own explicit forward loop;
//! only load-band semantics that are identical across families live
//! here. Consumers: runner.zig (the qwen3 substrate) and qwen35.

const std = @import("std");
const fucina = @import("fucina");
const weights = @import("fucina").weights;
const gguf_mod = @import("fucina").gguf;
const ptqtp_gguf = @import("fucina").ptqtp_gguf;

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const LinearWeight = weights.LinearWeight;

/// Load one projection linear by GGUF name: persisted PTQTP planes when the
/// file carries them (ptqtp_gguf pair-detection; a no-op metadata lookup on
/// undecorated files), else the base tensor. Skip-layer tensors inside a
/// decorated file take the base branch — plane presence is per tensor.
pub fn loadProjection(ctx: *ExecContext, file: *const gguf_mod.File, name: []const u8, rows: usize, cols: usize, for_fusion: bool) !LinearWeight {
    if (try ptqtp_gguf.maybeLoadPlanes(ctx, file, name, rows, cols)) |planes| return planes;
    const info = try file.get(name);
    return if (for_fusion)
        LinearWeight.loadForFusion(ctx, info, rows, cols)
    else
        LinearWeight.load(ctx, info, rows, cols);
}

pub const SeparateFfnInputProjection = struct {
    gate_proj: LinearWeight,
    up_proj: LinearWeight,

    pub fn deinit(self: *SeparateFfnInputProjection) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.* = undefined;
    }
};

pub const GateUpProjection = struct {
    gate: fucina.Tensor(.{ .seq, .ffn }),
    up: fucina.Tensor(.{ .seq, .ffn }),

    pub fn deinit(self: *GateUpProjection) void {
        self.up.deinit();
        self.gate.deinit();
        self.* = undefined;
    }
};

/// Gate/up input projection of a dense FFN (or a MoE shared expert via
/// `suffix`): two `width x hidden` linears under
/// `ffn_gate<suffix>`/`ffn_up<suffix>`, fused into one wider weight when
/// their dtype/layout allows so prefill issues a single GEMM.
pub const FfnInputProjection = union(enum) {
    separate: SeparateFfnInputProjection,
    fused: LinearWeight,

    pub fn load(ctx: *ExecContext, file: *const gguf_mod.File, hidden: usize, width: usize, layer_i: usize, comptime suffix: []const u8) !FfnInputProjection {
        var name_buf: [96]u8 = undefined;

        var gate = try loadProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_gate" ++ suffix ++ ".weight"), width, hidden, true);
        errdefer gate.deinit();
        var up = try loadProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_up" ++ suffix ++ ".weight"), width, hidden, true);
        errdefer up.deinit();

        var fuse_parts = [_]*LinearWeight{ &gate, &up };
        if (try weights.fuseLinear(ctx, &fuse_parts)) |fused| return .{ .fused = fused };

        return .{ .separate = .{ .gate_proj = gate, .up_proj = up } };
    }

    pub fn deinit(self: *FfnInputProjection) void {
        switch (self.*) {
            .separate => |*separate| separate.deinit(),
            .fused => |*weight| weight.deinit(),
        }
        self.* = undefined;
    }

    pub fn project(self: *const FfnInputProjection, ctx: *ExecContext, input: *const fucina.Tensor(.{ .seq, .embed }), width: usize) !GateUpProjection {
        return switch (self.*) {
            .separate => |*separate| blk: {
                var gate = try separate.gate_proj.linearSeq(ctx, input, .embed, .ffn);
                errdefer gate.deinit();
                var up = try separate.up_proj.linearSeq(ctx, input, .embed, .ffn);
                errdefer up.deinit();
                break :blk .{ .gate = gate, .up = up };
            },
            .fused => |*weight| blk: {
                var gate_up = try weight.linearSeq(ctx, input, .embed, .gate_up);
                defer gate_up.deinit();
                break :blk try splitGateUp(ctx, &gate_up, width);
            },
        };
    }
};

/// Zero-copy narrows of a fused gate/up projection result into the two
/// per-projection views (`width` columns each).
pub fn splitGateUp(ctx: *ExecContext, gate_up: *const fucina.Tensor(.{ .seq, .gate_up }), width: usize) !GateUpProjection {
    var gate_view = try gate_up.narrow(ctx, .gate_up, 0, width);
    defer gate_view.deinit();
    var gate = try gate_view.withTags(ctx, .{ .seq, .ffn });
    errdefer gate.deinit();

    var up_view = try gate_up.narrow(ctx, .gate_up, width, width);
    defer up_view.deinit();
    var up = try up_view.withTags(ctx, .{ .seq, .ffn });
    errdefer up.deinit();

    return .{ .gate = gate, .up = up };
}

/// Dense-FFN weights: gate/up input projection + down, all PTQTP-aware.
/// `suffix` selects the plain FFN (`""`) or a MoE shared expert
/// (`"_shexp"`, its own width).
pub const DenseFfn = struct {
    input_proj: FfnInputProjection,
    down_proj: LinearWeight,

    pub fn load(ctx: *ExecContext, file: *const gguf_mod.File, hidden: usize, width: usize, layer_i: usize, comptime suffix: []const u8) !DenseFfn {
        var name_buf: [96]u8 = undefined;

        var input_proj = try FfnInputProjection.load(ctx, file, hidden, width, layer_i, suffix);
        errdefer input_proj.deinit();
        var down = try loadProjection(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_down" ++ suffix ++ ".weight"), hidden, width, false);
        errdefer down.deinit();

        return .{ .input_proj = input_proj, .down_proj = down };
    }

    pub fn deinit(self: *DenseFfn) void {
        self.down_proj.deinit();
        self.input_proj.deinit();
        self.* = undefined;
    }
};

pub const MoeTrio = struct {
    gate: fucina.MoeRhs,
    up: fucina.MoeRhs,
    down: fucina.MoeRhs,
};

/// Load one layer's MoE expert trio (the `ffn_{gate,up,down}_exps.weight`
/// stacks), PTQTP plane siblings detected on every projection.
///
/// Streamed (`store` non-null): register geometry only — the expert stacks
/// stay on disk and are fetched on demand through the store's cache tiers.
/// `registerStreamedMoeLayer` touches only this layer's state, so the
/// parallel layer loader may call this concurrently for distinct layers.
///
/// Resident: expert blocks need no repack, so when the GGUF is mmap'd they
/// are borrowed straight from the mapping (the caller's Model takes
/// ownership of it) instead of copying the multi-GB stacks. Split GGUFs
/// cannot hand over their multiple mappings, so their experts are copied —
/// stream those instead for the big models.
pub fn loadMoeTrio(ctx: *ExecContext, file: *const gguf_mod.File, store: ?*fucina.ExpertStore, layer_i: usize, hidden: usize, moe_ffn: usize, n_experts: usize) !MoeTrio {
    var name_buf: [96]u8 = undefined;

    if (store) |s| {
        const trio = try weights.registerStreamedMoeLayer(s, layer_i, .{
            try ptqtp_gguf.streamedProjSpecAuto(file, try weights.layerName(&name_buf, layer_i, "ffn_gate_exps.weight"), hidden, moe_ffn, n_experts),
            try ptqtp_gguf.streamedProjSpecAuto(file, try weights.layerName(&name_buf, layer_i, "ffn_up_exps.weight"), hidden, moe_ffn, n_experts),
            // down transposes the FFN: (out_pe -> hidden).
            try ptqtp_gguf.streamedProjSpecAuto(file, try weights.layerName(&name_buf, layer_i, "ffn_down_exps.weight"), moe_ffn, hidden, n_experts),
        }, n_experts);
        return .{ .gate = trio.gate, .up = trio.up, .down = trio.down };
    }

    const borrow = file.is_mmap and !file.isSplit();
    var gate = try ptqtp_gguf.loadMoeRhsAuto(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_gate_exps.weight"), hidden, moe_ffn, n_experts, borrow);
    errdefer gate.deinit();
    var up = try ptqtp_gguf.loadMoeRhsAuto(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_up_exps.weight"), hidden, moe_ffn, n_experts, borrow);
    errdefer up.deinit();
    var down = try ptqtp_gguf.loadMoeRhsAuto(ctx, file, try weights.layerName(&name_buf, layer_i, "ffn_down_exps.weight"), moe_ffn, hidden, n_experts, borrow);
    errdefer down.deinit();

    return .{ .gate = gate, .up = up, .down = down };
}

/// The embed/output-norm/lm-head trio every flat-GGUF family loads first.
pub const EmbedHead = struct {
    token_embedding: LinearWeight,
    output_norm: fucina.Tensor(.{.embed}),
    output: LinearWeight,
};

/// Load `token_embd.weight`, `output_norm.weight`, and the lm head:
/// persisted PTQTP head planes win over the base tensor (a decorated head
/// on a tied-embedding model has planes and no base, while the embedding
/// keeps its source precision); a missing `output.weight` ties the head to
/// the embedding.
pub fn loadEmbedHead(ctx: *ExecContext, file: *const gguf_mod.File, vocab: usize, hidden: usize) !EmbedHead {
    var token_embedding = try LinearWeight.load(ctx, try file.get("token_embd.weight"), vocab, hidden);
    errdefer token_embedding.deinit();

    var output_norm = try weights.loadVector(ctx, try file.get("output_norm.weight"), hidden, .embed);
    errdefer output_norm.deinit();

    const output = blk: {
        if (try ptqtp_gguf.maybeLoadPlanes(ctx, file, "output.weight", vocab, hidden)) |planes| break :blk planes;
        if (file.maybeGet("output.weight")) |info| break :blk try LinearWeight.load(ctx, info, vocab, hidden);
        break :blk try token_embedding.cloneView(ctx); // tied embeddings
    };

    return .{ .token_embedding = token_embedding, .output_norm = output_norm, .output = output };
}

/// The GQA q-head -> kv-head map (`head_i / heads_per_kv`), allocated for
/// the model's lifetime.
pub fn kvHeadMap(allocator: Allocator, n_heads: usize, n_kv_heads: usize) ![]usize {
    const map = try allocator.alloc(usize, n_heads);
    const heads_per_kv = n_heads / n_kv_heads;
    for (map, 0..) |*kv_head, head_i| kv_head.* = head_i / heads_per_kv;
    return map;
}
