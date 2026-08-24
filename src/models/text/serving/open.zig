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
//! (`examples/lmserve`).

const std = @import("std");
const fucina = @import("fucina");
const contract = @import("contract.zig");
const gguf_chat = @import("gguf_chat.zig");
const chat = @import("../chat.zig");
const registry = @import("../../registry.zig");
const sampler = @import("../sampler.zig");
const cartridge_mod = @import("../cartridge.zig");
const cartridge_fleet = @import("../cartridge_fleet.zig");
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
        return openChat(Family, qwen3_traits, ctx, io, allocator, file, model_id, options, stderr);
    }
    if (comptime Family == gemma4_mod.Family) {
        return openChat(Family, gemma4_traits, ctx, io, allocator, file, model_id, options, stderr);
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

/// The per-family serving policy of the `Conversation`-hosted set: what
/// `GgufChatOptions` and the box's construction vary on.
const ChatTraits = struct {
    /// The reply's reasoning-block delimiters, when the family has one the
    /// server can toggle.
    think_markers: ?contract.ThinkMarkers = null,
    supports_think: bool = false,
    tool_style: contract.ToolStyle = .none,
    /// Default sampling: a fixed config, or the GGUF's own
    /// `general.sampling.*` metadata.
    sampling: union(enum) { fixed: sampler.Config, from_gguf },
    /// Wire `OpenOptions.spec` through (the qwen3 self-draft cascade).
    allow_spec: bool = false,
    /// Register the gemma4 turn-end extras (GGUF EOS + stray SPM `<eos>`).
    gemma_extra_stops: bool = false,
    /// The `--fleet` query-embedding trainer (null = no fleet serving).
    Trainer: ?type = null,
    /// Fleet precondition: MoE GGUFs need `--experts=borrow` (the query
    /// embedder forwards through raw expert blocks).
    fleet_needs_borrow: bool = false,
};

const qwen3_traits = ChatTraits{
    .think_markers = .{ .open = "<think>", .close = "</think>" },
    .supports_think = true,
    .tool_style = .hermes,
    // Qwen3's recommended no-think chat settings (the server default;
    // per-request reasoning switches nothing here — clients override).
    .sampling = .{ .fixed = .{ .temperature = 0.7, .top_k = 20, .top_p = 0.8 } },
    .allow_spec = true,
    .Trainer = qwen3_train.Trainer(.{ .q = false, .v = false }),
};

const gemma4_traits = ChatTraits{
    .sampling = .from_gguf,
    .gemma_extra_stops = true,
    .Trainer = gemma_train.Trainer(.{ .q = false, .v = false }),
    .fleet_needs_borrow = true,
};

/// The generic engine box of the `Conversation`-hosted families
/// (heap-pinned: the adapter and the fleet options hold pointers into it;
/// `extra_stops_buf` backs the adapter's borrowed `extra_stop_ids`).
fn ChatBox(comptime Family: type, comptime traits: ChatTraits) type {
    return struct {
        const Self = @This();
        const Adapter = gguf_chat.GgufChatBackend(Family.Model, Family.Tok);
        const FleetServe = if (traits.Trainer) |T| FleetServeFor(Family.Model, T, Family.Tokenizer) else void;

        allocator: Allocator,
        model_id: []u8,
        model: Family.Model,
        tokenizer: Family.Tokenizer,
        cart: ?cartridge_mod.Cartridge,
        fleet_serve: ?FleetServe,
        extra_stops_buf: [2]u32,
        extra_n: usize,
        adapter: Adapter,

        fn destroy(ptr: *anyopaque) void {
            const box: *Self = @ptrCast(@alignCast(ptr));
            const a = box.allocator;
            box.adapter.deinit();
            if (box.fleet_serve) |*fs| fs.deinit();
            if (box.cart) |*c| c.deinit();
            box.tokenizer.deinit();
            box.model.deinit();
            a.free(box.model_id);
            a.destroy(box);
        }
    };
}

fn openChat(
    comptime Family: type,
    comptime traits: ChatTraits,
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    options: OpenOptions,
    stderr: *std.Io.Writer,
) !Opened {
    const Box = ChatBox(Family, traits);
    var file_alive = true;
    errdefer if (file_alive) file.deinit();

    const box = try allocator.create(Box);
    errdefer allocator.destroy(box);
    box.allocator = allocator;
    box.cart = null;
    box.fleet_serve = null;
    box.model_id = try allocator.dupe(u8, model_id);
    errdefer allocator.free(box.model_id);

    box.tokenizer = Family.tokenizer(allocator, file) catch {
        try stderr.writeAll("this GGUF has no usable tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    errdefer box.tokenizer.deinit();
    const template = chat.Template.detect(file.getString("tokenizer.chat_template")) orelse blk: {
        if (Family.template_fallback) |format| break :blk chat.Template{ .format = format };
        try stderr.writeAll("this GGUF has no recognizable chat template\n");
        return error.NoChatTemplate;
    };
    const default_sampling = switch (traits.sampling) {
        .fixed => |config| config,
        .from_gguf => samplingFromGguf(file),
    };

    box.model = try Family.load(ctx, file, .{ .experts_borrow = options.experts_borrow });
    errdefer box.model.deinit();
    file.deinit();
    file_alive = false;

    if (options.cartridge_path) |path| box.cart = try loadCartridge(io, allocator, stderr, ctx, &box.model, path);
    errdefer if (box.cart) |*c| c.deinit();

    if (options.fleet_dir) |dir| {
        if (comptime traits.Trainer == null) return error.FleetUnsupported;
        if (comptime traits.fleet_needs_borrow) {
            if (box.model.config.num_experts > 0 and !box.model.config.borrow_experts) {
                // The query embedder forwards through the trainer, whose
                // MoE arm consumes raw expert blocks.
                try stderr.writeAll("--fleet on a gemma4 MoE GGUF needs --experts=borrow (the query embedder forwards through raw expert blocks)\n");
                return error.FleetUnsupported;
            }
        }
        box.fleet_serve = try Box.FleetServe.init(io, allocator, stderr, ctx, &box.model, &box.tokenizer, dir);
    }
    errdefer if (box.fleet_serve) |*fs| fs.deinit();
    if (box.fleet_serve) |*fs| {
        try stderr.print("fleet: {d} documents, {d} retrieval chunks, {d} docs composed per request\n", .{
            fs.fleet.manifest.docs.items.len,
            fs.index.len(),
            options.rag_docs,
        });
        try stderr.flush();
    }

    // Turn-end ids beyond the template stop marker (gemma4: the GGUF's own
    // EOS and a stray SPM <eos>, id 1 — the gemma4 chat harness registers
    // the same pair).
    box.extra_n = 0;
    if (traits.gemma_extra_stops) {
        if (box.tokenizer.eosId()) |e| {
            box.extra_stops_buf[box.extra_n] = e;
            box.extra_n += 1;
        }
        box.extra_stops_buf[box.extra_n] = 1;
        box.extra_n += 1;
    }

    const kv_slots = try gguf_chat.kvRamGuardSlots(Family.Model, ctx, &box.model, options.context_len, try slotsForBatch(stderr, options), options.kv_slots_force, stderr);

    box.adapter = Box.Adapter.init(
        allocator,
        ctx,
        &box.model,
        &box.tokenizer,
        template,
        .{
            .model_id = box.model_id,
            .context_len = options.context_len,
            .extra_stop_ids = box.extra_stops_buf[0..box.extra_n],
            .think_markers = traits.think_markers,
            .supports_think = traits.supports_think,
            .tool_style = traits.tool_style,
            .default_sampling = default_sampling,
            .speculation = traits.allow_spec and options.spec,
            .constraint_cache_len = @max(8, options.batch),
            .kv_slots = kv_slots,
            .kv_disk = kvDiskOptions(io, options),
            .cartridge = if (box.cart) |*c| c else null,
            .fleet = if (box.fleet_serve) |*fs| .{
                .io = io,
                .dir = options.fleet_dir.?,
                .manifest = &fs.fleet.manifest,
                .index = &fs.index,
                .embed_ctx = fs,
                .embedFn = Box.FleetServe.embed,
                .rag_docs = options.rag_docs,
                .rag_chunks = options.rag_chunks,
                .adaptive = options.rag_adaptive,
                .switch_margin = options.rag_margin,
            } else null,
        },
    );
    return .{ .ptr = box, .destroyFn = Box.destroy, .backend = box.adapter.backend() };
}

/// --fleet serving state: the fleet's manifest + cosine index plus a
/// no-adapter trainer whose `embedLastHidden` implements the fleet's
/// retrieval-embedding contract (`cartridge_fleet.embed_suffix`) for
/// incoming queries. One comptime instantiation per (model, trainer,
/// tokenizer) family; everything is borrowed by the backend's
/// `FleetOptions`, so this must outlive the served backend.
fn FleetServeFor(comptime ModelT: type, comptime TrainerT: type, comptime TokT: type) type {
    return struct {
        const FleetServe = @This();
        allocator: Allocator,
        ctx: *fucina.ExecContext,
        tokenizer: *const TokT,
        fleet: cartridge_fleet.Fleet,
        index: cartridge_fleet.EmbedIndex,
        trainer: TrainerT,

        fn init(
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

        fn deinit(self: *FleetServe) void {
            self.trainer.deinit();
            self.index.deinit();
            self.fleet.deinit();
        }

        /// `gguf_chat.FleetOptions.embedFn`: the exact recipe the index was
        /// built with — text ids ++ separately tokenized `embed_suffix` ids,
        /// then the final-norm last hidden state (worker thread only, like
        /// generation).
        fn embed(ptr: *anyopaque, text: []const u8, out: []f32) anyerror!void {
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
fn loadCartridge(
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

/// The gguf-chat backends' evict-to-disk tier config (`kv_cache_dir` arms
/// it; the directory must exist).
fn kvDiskOptions(io: std.Io, options: OpenOptions) ?gguf_chat.KvDiskOptions {
    const dir = options.kv_cache_dir orelse return null;
    return .{ .io = io, .dir = dir, .max_files = @max(options.kv_disk_slots, 1) };
}

/// A lockstep batch needs one resident KV slot per stream; raise the
/// requested pool to the batch width (the RAM guard then prices the total).
/// Shared with `models.qwen3.shine_serving`.
pub fn slotsForBatch(stderr: *std.Io.Writer, options: OpenOptions) !usize {
    if (options.batch > options.kv_slots) {
        try stderr.print("--batch {d}: raising --kv-slots {d} -> {d} (one resident slot per lockstep stream)\n", .{ options.batch, options.kv_slots, options.batch });
        try stderr.flush();
        return options.batch;
    }
    return options.kv_slots;
}
