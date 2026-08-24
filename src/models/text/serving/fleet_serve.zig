//! `--fleet` serving state for the `Conversation`-hosted families:
//! `FleetServeFor` (the fleet manifest + cosine retrieval index + query
//! embedder behind `gguf_chat.FleetOptions`) and `loadCartridge` (the
//! trained-cartridge loader with its KV-geometry probe). The adapter
//! skeleton (`adapter_common.zig`) wires both into the generic engine box.

const std = @import("std");
const fucina = @import("fucina");
const cartridge_mod = @import("../cartridge.zig");
const cartridge_fleet = @import("../cartridge_fleet.zig");

const Allocator = std.mem.Allocator;

/// --fleet serving state: the fleet's manifest + cosine index plus a
/// no-adapter trainer whose `embedLastHidden` implements the fleet's
/// retrieval-embedding contract (`cartridge_fleet.embed_suffix`) for
/// incoming queries. One comptime instantiation per (model, trainer,
/// tokenizer) family; everything is borrowed by the backend's
/// `FleetOptions`, so this must outlive the served backend.
pub fn FleetServeFor(comptime ModelT: type, comptime TrainerT: type, comptime TokT: type) type {
    return struct {
        const FleetServe = @This();
        allocator: Allocator,
        ctx: *fucina.ExecContext,
        tokenizer: *const TokT,
        fleet: cartridge_fleet.Fleet,
        index: cartridge_fleet.EmbedIndex,
        trainer: TrainerT,

        pub fn init(
            io: std.Io,
            allocator: Allocator,
            stderr: *std.Io.Writer,
            ctx: *fucina.ExecContext,
            model: *const ModelT,
            tokenizer: *const TokT,
            dir: []const u8,
        ) !FleetServe {
            var fleet = cartridge_fleet.Fleet.open(allocator, io, dir, 0, .{ .budget = 1 }) catch |err| {
                try stderr.print("--fleet {s}: cannot open the fleet manifest ({t})\n", .{ dir, err });
                return err;
            };
            errdefer fleet.deinit();
            if (fleet.manifest.docs.items.len == 0) {
                try stderr.print("--fleet {s}: the manifest lists no documents\n", .{dir});
                return error.EmptyFleet;
            }
            if (fleet.manifest.embed_dim != model.config.hidden_size) {
                try stderr.print(
                    "--fleet {s}: no retrieval index for this model (index dim {d}, model hidden {d}) — rebuild with `zig build cartridge-fleet -- --resume --rounds 0 --docs ...`\n",
                    .{ dir, fleet.manifest.embed_dim, model.config.hidden_size },
                );
                return error.MissingIndex;
            }
            const index_path = try fleet.indexPath();
            defer allocator.free(index_path);
            var mapped = try cartridge_fleet.mmapFile(io, index_path);
            defer mapped.deinit();
            var index = try cartridge_fleet.EmbedIndex.initFromBytes(allocator, mapped.bytes);
            errdefer index.deinit();

            // The query embedder runs through the family's no-adapter trainer.
            var trainer = TrainerT.init(ctx, model, .{ .rank = 1, .alpha = 1 }, 0) catch |err| {
                try stderr.writeAll("--fleet: this GGUF cannot host the query-embedding trainer (dense qwen3, or gemma4 with --experts=borrow)\n");
                return err;
            };
            errdefer trainer.deinit();

            // Probe doc 0's cartridge against the model's KV geometry so a
            // foreign fleet fails at startup, not mid-request.
            {
                const cart_path = try std.fs.path.join(allocator, &.{ dir, fleet.manifest.docs.items[0].cart_file });
                defer allocator.free(cart_path);
                var cart_mapped = try cartridge_fleet.mmapFile(io, cart_path);
                defer cart_mapped.deinit();
                var cart = try cartridge_mod.Cartridge.initFromStateDict(ctx, allocator, cart_mapped.bytes);
                defer cart.deinit();
                var probe = try model.initCache(ctx, cart.p + 1);
                defer probe.deinit();
                cart.writeToCache(ctx, &probe) catch |err| {
                    try stderr.print("--fleet {s}: its cartridges do not fit this model's KV geometry\n", .{dir});
                    return err;
                };
            }

            return .{
                .allocator = allocator,
                .ctx = ctx,
                .tokenizer = tokenizer,
                .fleet = fleet,
                .index = index,
                .trainer = trainer,
            };
        }

        pub fn deinit(self: *FleetServe) void {
            self.trainer.deinit();
            self.index.deinit();
            self.fleet.deinit();
        }

        /// `gguf_chat.FleetOptions.embedFn`: the exact recipe the index was
        /// built with — text ids ++ separately tokenized `embed_suffix` ids,
        /// then the final-norm last hidden state (worker thread only, like
        /// generation).
        pub fn embed(ptr: *anyopaque, text: []const u8, out: []f32) anyerror!void {
            const self: *FleetServe = @ptrCast(@alignCast(ptr));
            const a = self.allocator;
            const text_ids = try self.tokenizer.encode(a, text);
            defer a.free(text_ids);
            const suffix_ids = try self.tokenizer.encode(a, cartridge_fleet.embed_suffix);
            defer a.free(suffix_ids);
            const full = try a.alloc(usize, text_ids.len + suffix_ids.len);
            defer a.free(full);
            for (full[0..text_ids.len], text_ids) |*dst, id| dst.* = id;
            for (full[text_ids.len..], suffix_ids) |*dst, id| dst.* = id;
            try self.trainer.embedLastHidden(self.ctx, full, out);
        }
    };
}

/// Load a trained cartridge (docs/CARTRIDGES.md) and probe it against the
/// model's KV geometry, so a mismatched file fails at load instead of
/// mid-request.
pub fn loadCartridge(
    io: std.Io,
    allocator: Allocator,
    stderr: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: anytype,
    path: []const u8,
) !cartridge_mod.Cartridge {
    const bytes = blk: {
        var dir = std.Io.Dir.cwd();
        break :blk try dir.readFileAlloc(io, path, allocator, .limited(1024 * 1024 * 1024));
    };
    defer allocator.free(bytes);
    var cart = try cartridge_mod.Cartridge.initFromStateDict(ctx, allocator, bytes);
    errdefer cart.deinit();
    var probe = try model.initCache(ctx, cart.p + 1);
    defer probe.deinit();
    cart.writeToCache(ctx, &probe) catch |err| {
        try stderr.print("cartridge {s} does not fit this model's KV geometry\n", .{path});
        return err;
    };
    return cart;
}
