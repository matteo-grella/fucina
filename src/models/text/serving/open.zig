//! Load-and-serve entry for the serving band: `open` (and `openFromFile`)
//! sniff a GGUF's `general.architecture`, resolve the family through the
//! architecture registry (`models.registry`), and return a ready `Backend`.
//! The `Conversation`-hosted families (qwen3, qwen3moe, gemma4) share one
//! generic engine box with the full option surface (KV slot pool + disk
//! tier, speculative decode, cartridges, cartridge fleets); the
//! engine-hosted families dispatch to their family serving adapters
//! (`qwen35.serving`, `inkling.serving`, `deepseek4.serving`). Registered
//! families without a serving adapter (deepseek2, glm4moe) and unknown
//! architectures return `error.UnsupportedArchitecture`; nanochat
//! checkpoints and diffusion-gemma stay with the CLI front end
//! (`apps/lmserve`).

const std = @import("std");
const fucina = @import("fucina");
const contract = @import("contract.zig");
const adapter_common = @import("adapter_common.zig");
const registry = @import("../../registry.zig");
const qwen3_model = @import("../../qwen3/model.zig");
const qwen3_train = @import("../../qwen3/train.zig");
const gemma4_mod = @import("../../gemma/model.zig");
const gemma_train = @import("../../gemma/train.zig");
const qwen35_serving = @import("../../qwen35/serving.zig");
const inkling_serving = @import("../../inkling/serving.zig");
const deepseek4_serving = @import("../../deepseek4/serving.zig");
const qwen35_model = @import("../../qwen35/model.zig");
const inkling_model = @import("../../inkling/model.zig");
const deepseek4_model = @import("../../deepseek4/model.zig");

const Allocator = std.mem.Allocator;

pub const OpenOptions = contract.OpenOptions;
pub const Opened = contract.Opened;
pub const samplingFromGguf = contract.samplingFromGguf;
/// A lockstep batch needs one resident KV slot per stream (shared with
/// `models.qwen3.shine_serving`).
pub const slotsForBatch = adapter_common.slotsForBatch;

/// Open `gguf_path` and return a ready `Backend` for its
/// `general.architecture`. Families served: qwen3, qwen3moe, gemma4 (the
/// `Conversation`-hosted set) and qwen35, qwen35moe, inkling, deepseek4
/// (engine-hosted). `stderr` is the diagnostic sink (load-time guard
/// arithmetic and error detail); a host may pass a discarding writer.
pub fn open(
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    gguf_path: []const u8,
    options: OpenOptions,
    stderr: *std.Io.Writer,
) !Opened {
    // loadMmapAuto follows llama.cpp split GGUFs; single-file paths take
    // the plain loadMmap route.
    var file = try fucina.gguf.File.loadMmapAuto(allocator, io, gguf_path);
    const model_id = std.fs.path.stem(std.fs.path.basename(gguf_path));
    return openFromFile(ctx, io, allocator, &file, model_id, options, stderr);
}

/// `open` over an already-loaded GGUF (a host that sniffed the architecture
/// itself). Takes ownership of `file` on every path: the weights are loaded
/// (or the error is returned) and the file is deinitialized before this
/// returns. `model_id` is copied.
pub fn openFromFile(
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    options: OpenOptions,
    stderr: *std.Io.Writer,
) !Opened {
    if (options.fleet_dir != null and
        (options.cartridge_path != null or options.kv_cache_dir != null or options.batch > 1))
    {
        file.deinit();
        return error.InvalidOptions;
    }
    const arch = file.getString("general.architecture") orelse {
        file.deinit();
        return error.UnknownArchitecture;
    };
    inline for (registry.families) |entry| {
        if (std.mem.eql(u8, arch, entry.arch)) {
            return openFamily(entry.Family, ctx, io, allocator, file, model_id, options, stderr);
        }
    }
    file.deinit();
    return error.UnsupportedArchitecture;
}

/// Comptime dispatch on the resolved family: the `Conversation`-hosted set
/// takes the generic chat box with its serving traits; the engine-hosted
/// set forwards to the family serving adapters; registered families
/// without an adapter reject.
fn openFamily(
    comptime Family: type,
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    options: OpenOptions,
    stderr: *std.Io.Writer,
) !Opened {
    if (comptime Family == qwen3_model.Family) {
        return adapter_common.openChat(Family, qwen3_traits, ctx, io, allocator, file, model_id, options, stderr);
    }
    if (comptime Family == gemma4_mod.Family) {
        return adapter_common.openChat(Family, gemma4_traits, ctx, io, allocator, file, model_id, options, stderr);
    }
    // Engine-hosted families: no Conversation, so no cartridges, fleets or
    // KV reuse tiers; reject those options loudly (the caps philosophy).
    if (comptime Family == qwen35_model.Family or Family == inkling_model.Family or Family == deepseek4_model.Family) {
        if (options.cartridge_path != null or options.fleet_dir != null or options.kv_cache_dir != null) {
            try stderr.writeAll("--cartridge/--fleet/--kv-cache-dir need the Conversation-hosted families (qwen3/qwen3moe/gemma4)\n");
            file.deinit();
            return error.InvalidOptions;
        }
        if (comptime Family == qwen35_model.Family)
            return qwen35_serving.openFromFile(ctx, io, allocator, file, model_id, options, stderr);
        if (comptime Family == inkling_model.Family)
            return inkling_serving.openFromFile(ctx, io, allocator, file, model_id, options, stderr);
        return deepseek4_serving.openFromFile(ctx, io, allocator, file, model_id, options, stderr);
    }
    // Registered for the comptime lookup, no serving adapter (deepseek2,
    // glm4moe).
    file.deinit();
    return error.UnsupportedArchitecture;
}

const qwen3_traits = adapter_common.ChatTraits{
    .think_markers = .{ .open = "<think>", .close = "</think>" },
    .supports_think = true,
    .tool_style = .hermes,
    // Qwen3's recommended no-think chat settings (the server default;
    // per-request reasoning switches nothing here — clients override).
    .sampling = .{ .fixed = .{ .temperature = 0.7, .top_k = 20, .top_p = 0.8 } },
    .allow_spec = true,
    .Trainer = qwen3_train.Trainer(.{ .q = false, .v = false }),
};

const gemma4_traits = adapter_common.ChatTraits{
    .sampling = .from_gguf,
    .gemma_extra_stops = true,
    .Trainer = gemma_train.Trainer(.{ .q = false, .v = false }),
    .fleet_needs_borrow = true,
};
