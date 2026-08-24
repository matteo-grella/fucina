//! Load-and-serve entry for the serving band: `open` (and `openFromFile`)
//! sniff a GGUF's `general.architecture`, resolve the family through the
//! architecture registry (`models.registry`), and dispatch to the row's
//! serving wiring (`registry.Entry.Serving`, the family's `serving.zig`).
//! The `Conversation`-hosted families (qwen3, qwen3moe, gemma4) share one
//! generic engine box with the full option surface (KV slot pool + disk
//! tier, speculative decode, cartridges, cartridge fleets); the
//! engine-hosted families (qwen35, qwen35moe, inkling, deepseek4) drive
//! their own engines through the shared adapter skeleton
//! (`adapter_common.zig`). Registered families without a serving wiring
//! (deepseek2, glm4moe — the registry rows state the blockers) and
//! unknown architectures return `error.UnsupportedArchitecture`; nanochat
//! checkpoints and diffusion-gemma stay with the CLI front end
//! (`apps/lmserve`).

const std = @import("std");
const fucina = @import("fucina");
const contract = @import("contract.zig");
const adapter_common = @import("adapter_common.zig");
const registry = @import("../../registry.zig");

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
            // Registered for the comptime lookup only (deepseek2, glm4moe).
            const Serving = entry.Serving orelse {
                file.deinit();
                return error.UnsupportedArchitecture;
            };
            if (comptime !Serving.conversation_hosted) {
                // Engine-hosted families: no Conversation, so no cartridges,
                // fleets or KV reuse tiers; reject those options loudly (the
                // caps philosophy).
                if (options.cartridge_path != null or options.fleet_dir != null or options.kv_cache_dir != null) {
                    try stderr.writeAll("--cartridge/--fleet/--kv-cache-dir need the Conversation-hosted families (qwen3/qwen3moe/gemma4)\n");
                    file.deinit();
                    return error.InvalidOptions;
                }
            }
            return Serving.openFromFile(ctx, io, allocator, file, model_id, options, stderr);
        }
    }
    file.deinit();
    return error.UnsupportedArchitecture;
}
