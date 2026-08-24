//! gemma4 serving wiring (`registry.Entry.Serving`): the
//! `Conversation`-hosted generic engine box (`adapter_common.openChat`)
//! with this family's serving traits. Lives beside the model so the traits
//! can name the family trainer (`train.zig`, the `--fleet` query
//! embedder). Public as `models.gemma.serving`.

const std = @import("std");
const fucina = @import("fucina");
const adapter_common = @import("../text/serving/adapter_common.zig");
const contract = @import("../text/serving/contract.zig");
const gemma_model = @import("model.zig");
const gemma_train = @import("train.zig");

/// Served through the generic `Conversation` engine box: the full option
/// surface (cartridges, fleets, KV reuse tiers) applies.
pub const conversation_hosted = true;

const traits = adapter_common.ChatTraits{
    .sampling = .from_gguf,
    .gemma_extra_stops = true,
    .Trainer = gemma_train.Trainer(.{ .q = false, .v = false }),
    .fleet_needs_borrow = true,
};

/// `serving.openFromFile` dispatches here for the gemma4 arch. Takes
/// ownership of `file` on every path.
pub fn openFromFile(
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: std.mem.Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    options: contract.OpenOptions,
    stderr: *std.Io.Writer,
) !contract.Opened {
    return adapter_common.openChat(gemma_model.Family, traits, ctx, io, allocator, file, model_id, options, stderr);
}
