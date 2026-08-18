//! PTQTP decoration + GGUF persistence for a loaded Qwen3 `Model`: the
//! quantization capabilities AROUND the model, split out of model.zig
//! (which keeps the Qwen3 forward itself). See docs/PTQTP.md.
const std = @import("std");
const fucina = @import("fucina");
const qwen3 = @import("model.zig");
const weights = @import("../weights.zig");
const ptqtp_gguf = @import("../ptqtp_gguf.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;

/// Options for `decorate`: PTQTP-decorate every eligible layer linear in
/// place (attention q/k/v — split or fused — o_proj, dense FFN
/// gate/up/down): each becomes two packed TQ2_0 trit-planes and its
/// original storage is dropped (docs/PTQTP.md). Embeddings, the lm_head,
/// norms, and MoE expert stacks are left untouched (MoE FFNs count as
/// skipped).
pub const DecorateOptions = struct {
    solver: fucina.ptqtp.Options = .{},
    /// Leave the first/last N layers in their source precision — the
    /// edge layers are the most quantization-sensitive in extreme
    /// low-bit practice (data-free: pure configuration, no inputs).
    skip_first_layers: usize = 0,
    skip_last_layers: usize = 0,
    /// Per-projection plane-count overrides (null = solver.planes):
    /// selective capacity for the sensitive projections — an extra
    /// trit-plane costs +2.06 bpw only where applied and keeps every
    /// op ternary, unlike layer skipping which retains source-dtype
    /// matmuls. down_proj (and o_proj) are the classic hot spots.
    down_planes: ?u8 = null,
    o_planes: ?u8 = null,
};

pub fn decorate(model: *qwen3.Model, ctx: *ExecContext, decorate_options: DecorateOptions) !weights.PtqtpReport {
    const options = decorate_options.solver;
    var report = weights.PtqtpReport{};
    const n_layers = model.layers.len;
    for (model.layers, 0..) |*layer, layer_i| {
        if (layer_i < decorate_options.skip_first_layers or
            layer_i + decorate_options.skip_last_layers >= n_layers)
        {
            report.skipped_layers += 1;
            continue;
        }
        switch (layer.attn_proj) {
            .separate => |*separate| {
                try weights.decoratePtqtpInto(&separate.q_proj, ctx, options, &report);
                try weights.decoratePtqtpInto(&separate.k_proj, ctx, options, &report);
                try weights.decoratePtqtpInto(&separate.v_proj, ctx, options, &report);
            },
            .fused => |*fused| try weights.decoratePtqtpInto(fused, ctx, options, &report),
        }
        var o_options = options;
        if (decorate_options.o_planes) |p| o_options.planes = p;
        try weights.decoratePtqtpInto(&layer.o_proj, ctx, o_options, &report);
        switch (layer.ffn) {
            .dense => |*dense| {
                switch (dense.input_proj) {
                    .separate => |*separate| {
                        try weights.decoratePtqtpInto(&separate.gate_proj, ctx, options, &report);
                        try weights.decoratePtqtpInto(&separate.up_proj, ctx, options, &report);
                    },
                    .fused => |*fused| try weights.decoratePtqtpInto(fused, ctx, options, &report),
                }
                var down_options = options;
                if (decorate_options.down_planes) |p| down_options.planes = p;
                try weights.decoratePtqtpInto(&dense.down_proj, ctx, down_options, &report);
            },
            .moe => report.skipped += 1,
        }
    }
    return report;
}

/// Persist a (decorated) model as a GGUF beside its source file: every
/// `.ptqtp` weight becomes per-source-name plane tensors, all other tensors
/// and metadata pass through byte-verbatim (ptqtp_gguf.zig; docs/PTQTP.md).
/// Fused in-memory weights are row-sliced back to their source tensor
/// names, so the saved file is independent of this load's fusion decisions.
/// `src` must be the still-open GGUF this model was loaded from.
pub fn save(model: *const qwen3.Model, ctx: *ExecContext, io: std.Io, src: *const gguf.File, out_path: []const u8) !ptqtp_gguf.SaveReport {
    var arena_state = std.heap.ArenaAllocator.init(ctx.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var entries: std.ArrayList(ptqtp_gguf.SaveEntry) = .empty;
    try entries.append(arena, .{ .name = "output.weight", .weight = &model.output });

    const q_dim = model.config.qProjectionDim();
    const kv_dim = model.config.kvProjectionDim();
    const ffn_dim = model.config.intermediate_size;
    for (model.layers, 0..) |*layer, layer_i| {
        switch (layer.attn_proj) {
            .separate => |*separate| {
                try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_q.weight"), .weight = &separate.q_proj });
                try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_k.weight"), .weight = &separate.k_proj });
                try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_v.weight"), .weight = &separate.v_proj });
            },
            .fused => |*fused| {
                try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_q.weight"), .weight = fused, .row0 = 0, .rows = q_dim });
                try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_k.weight"), .weight = fused, .row0 = q_dim, .rows = kv_dim });
                try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_v.weight"), .weight = fused, .row0 = q_dim + kv_dim, .rows = kv_dim });
            },
        }
        try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "attn_output.weight"), .weight = &layer.o_proj });
        switch (layer.ffn) {
            .dense => |*dense| {
                switch (dense.input_proj) {
                    .separate => |*separate| {
                        try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "ffn_gate.weight"), .weight = &separate.gate_proj });
                        try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "ffn_up.weight"), .weight = &separate.up_proj });
                    },
                    .fused => |*fused| {
                        try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "ffn_gate.weight"), .weight = fused, .row0 = 0, .rows = ffn_dim });
                        try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "ffn_up.weight"), .weight = fused, .row0 = ffn_dim, .rows = ffn_dim });
                    },
                }
                try entries.append(arena, .{ .name = try layerNameOwned(arena, layer_i, "ffn_down.weight"), .weight = &dense.down_proj });
            },
            .moe => {},
        }
    }

    // MoE loads moved the mmap into the model (loadMoeFfn borrows expert
    // blocks), so the metadata copy reads the region through the model.
    const options = ptqtp_gguf.SaveOptions{
        .header_bytes = if (model.weight_mapping) |mapping| mapping.bytes else null,
    };
    return ptqtp_gguf.saveFile(ctx.allocator, io, src, entries.items, options, out_path);
}

/// `weights.layerName` with owned storage — `save` builds its entry list in
/// an arena, so the names must outlive the stack name buffer.
fn layerNameOwned(allocator: Allocator, layer_i: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.allocPrint(allocator, "blk.{d}.{s}", .{ layer_i, suffix });
}
