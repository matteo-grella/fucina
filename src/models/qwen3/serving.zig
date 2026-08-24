//! qwen3/qwen3moe serving wiring (`registry.Entry.Serving`): the
//! `Conversation`-hosted generic engine box (`adapter_common.openChat`)
//! with this family's serving traits. Lives beside the model so the traits
//! can name the family trainer (`train.zig`, the `--fleet` query
//! embedder). Public as `models.qwen3.serving`.

const std = @import("std");
const fucina = @import("fucina");
const adapter_common = @import("../text/serving/adapter_common.zig");
const contract = @import("../text/serving/contract.zig");
const qwen3_model = @import("model.zig");
const qwen3_train = @import("train.zig");

/// Served through the generic `Conversation` engine box: the full option
/// surface (cartridges, fleets, KV reuse tiers, speculative decode)
/// applies.
pub const conversation_hosted = true;

const traits = adapter_common.ChatTraits{
    .think_markers = .{ .open = "<think>", .close = "</think>" },
    .supports_think = true,
    .tool_style = .hermes,
    // Qwen3's recommended no-think chat settings (the server default;
    // per-request reasoning switches nothing here — clients override).
    .sampling = .{ .fixed = .{ .temperature = 0.7, .top_k = 20, .top_p = 0.8 } },
    .allow_spec = true,
    .Trainer = qwen3_train.Trainer(.{ .q = false, .v = false }),
};

/// `serving.openFromFile` dispatches here for the qwen3/qwen3moe archs.
/// Takes ownership of `file` on every path.
pub fn openFromFile(
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: std.mem.Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    options: contract.OpenOptions,
    stderr: *std.Io.Writer,
) !contract.Opened {
    return adapter_common.openChat(qwen3_model.Family, traits, ctx, io, allocator, file, model_id, options, stderr);
}
