//! Load-and-serve entry for the serving band: `open` (and `openFromFile`)
//! sniff a GGUF's `general.architecture` and return a ready `Backend` for
//! the families the shared `llm.chat.Conversation` hosts (qwen3, qwen3moe,
//! gemma4), with the full engine option surface (KV slot pool + disk tier,
//! speculative decode, cartridges, cartridge fleets, SHINE adapter fleets).
//! Architectures whose adapters cannot ride `Conversation` (nanochat,
//! diffusion-gemma, inkling, qwen35/qwen35moe, deepseek4) return
//! `error.UnsupportedArchitecture`: those adapters live with the CLI front
//! end (`examples/lmserve/backend_*.zig`), which dispatches to them itself
//! and falls back to `open` for the rest.

const std = @import("std");
const fucina = @import("fucina");
const contract = @import("contract.zig");
const gguf_chat = @import("gguf_chat.zig");
const chat = @import("../chat.zig");
const sampler = @import("../sampler.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const spm_tokenizer_mod = @import("../spm_tokenizer.zig");
const cartridge_mod = @import("../cartridge.zig");
const cartridge_fleet = @import("../cartridge_fleet.zig");
const qwen3_model = @import("../qwen3/model.zig");
const qwen3_train = @import("../qwen3/train.zig");
const qwen3_shine = @import("../qwen3/shine.zig");
const gemma4_mod = @import("../gemma/model.zig");
const gemma_train = @import("../gemma/train.zig");

const Allocator = std.mem.Allocator;

/// Engine options for `open`/`openFromFile`: the CLI-independent form of
/// lmserve's flags. Exclusions the engine cannot host are rejected with
/// `error.InvalidOptions`: `fleet_dir` excludes `cartridge_path`,
/// `kv_cache_dir` and `batch > 1`; `shine_fleet_dir` additionally excludes
/// `fleet_dir`.
pub const OpenOptions = struct {
    /// Per-request context budget in tokens (prompt + reply).
    context_len: usize = 4096,
    /// Speculative decoding for solo generations (qwen3/qwen3moe).
    spec: bool = false,
    /// Lockstep batch width the host intends to drive (sizes the constraint
    /// cache and raises the KV slot pool to one slot per stream).
    batch: usize = 1,
    /// Zero-copy MoE expert load (gemma4).
    experts_borrow: bool = false,
    /// Resident cross-request KV reuse slots (each a full `context_len`
    /// cache; the RAM guard prices the total at load).
    kv_slots: usize = 1,
    /// Keep the requested `kv_slots` even when the RAM guard would clamp.
    kv_slots_force: bool = false,
    /// Evict-to-disk KV tier directory (must exist; null = off).
    kv_cache_dir: ?[]const u8 = null,
    /// Max sidecar files under `kv_cache_dir`.
    kv_disk_slots: usize = 8,
    /// Trained KV-prefix cartridge (safetensors path; docs/CARTRIDGES.md).
    cartridge_path: ?[]const u8 = null,
    /// Per-document cartridge fleet directory (qwen3 dense, gemma4).
    fleet_dir: ?[]const u8 = null,
    /// Per-document SHINE adapter fleet directory (qwen3 dense).
    shine_fleet_dir: ?[]const u8 = null,
    /// Fleet: documents composed per request.
    rag_docs: usize = 2,
    /// Fleet: cosine top-N chunks scanned per selection.
    rag_chunks: usize = 8,
    /// Fleet: decisive-margin knowledge-base switching for continuing
    /// conversations (default fully sticky).
    rag_adaptive: bool = false,
    /// Fleet: the adaptive switch margin (cosine units).
    rag_margin: f32 = 0.05,
};

/// A loaded serving engine: the model, tokenizer, optional cartridge/fleet
/// state, and the adapter behind `backend`, heap-owned behind one handle.
/// `backend` stays valid until `deinit`.
pub const Opened = struct {
    ptr: *anyopaque,
    destroyFn: *const fn (ptr: *anyopaque) void,
    backend: contract.Backend,

    pub fn deinit(self: *Opened) void {
        self.destroyFn(self.ptr);
        self.* = undefined;
    }
};

/// Open `gguf_path` and return a ready `Backend` for its
/// `general.architecture`. Families served: qwen3, qwen3moe, gemma4 (the
/// `Conversation`-hosted set). nanochat, diffusion-gemma, inkling,
/// qwen35/qwen35moe and deepseek4 return `error.UnsupportedArchitecture`:
/// their adapters live with `examples/lmserve`. `stderr` is the diagnostic
/// sink (load-time guard arithmetic and error detail); a host may pass a
/// discarding writer.
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
    if (options.shine_fleet_dir != null and
        (options.fleet_dir != null or options.cartridge_path != null or
            options.kv_cache_dir != null or options.batch > 1))
    {
        file.deinit();
        return error.InvalidOptions;
    }
    const arch = file.getString("general.architecture") orelse {
        file.deinit();
        return error.UnknownArchitecture;
    };
    if (std.mem.eql(u8, arch, "qwen3") or std.mem.eql(u8, arch, "qwen3moe")) {
        return openQwen3(ctx, io, allocator, file, model_id, options, stderr);
    }
    if (std.mem.eql(u8, arch, "gemma4")) {
        return openGemma4(ctx, io, allocator, file, model_id, options, stderr);
    }
    file.deinit();
    return error.UnsupportedArchitecture;
}

const Qwen3Adapter = gguf_chat.GgufChatBackend(qwen3_model.Model, tokenizer_mod);
const Gemma4Adapter = gguf_chat.GgufChatBackend(gemma4_mod.Model, spm_tokenizer_mod);
const ShineAdapter = gguf_chat.GgufChatBackend(qwen3_shine.AdaptedModel, tokenizer_mod);

/// The qwen3/qwen3moe engine box (heap-pinned: the adapter and the fleet
/// options hold pointers into it).
const Qwen3Box = struct {
    allocator: Allocator,
    model_id: []u8,
    model: qwen3_model.Model,
    tokenizer: tokenizer_mod.Tokenizer,
    cart: ?cartridge_mod.Cartridge,
    fleet_serve: ?FleetServeQwen3,
    adapter: Qwen3Adapter,

    fn destroy(ptr: *anyopaque) void {
        const box: *Qwen3Box = @ptrCast(@alignCast(ptr));
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

fn openQwen3(
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    options: OpenOptions,
    stderr: *std.Io.Writer,
) !Opened {
    if (options.shine_fleet_dir != null) return openQwen3ShineFleet(ctx, io, allocator, file, model_id, options, stderr);
    var file_alive = true;
    errdefer if (file_alive) file.deinit();

    const box = try allocator.create(Qwen3Box);
    errdefer allocator.destroy(box);
    box.allocator = allocator;
    box.cart = null;
    box.fleet_serve = null;
    box.model_id = try allocator.dupe(u8, model_id);
    errdefer allocator.free(box.model_id);

    box.model = try qwen3_model.Model.loadGgufFromFile(ctx, file, try qwen3_model.Config.fromGguf(file));
    errdefer box.model.deinit();
    box.tokenizer = tokenizer_mod.Tokenizer.initFromGguf(allocator, file, .{}) catch {
        try stderr.writeAll("this GGUF has no usable tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    errdefer box.tokenizer.deinit();
    const template = chat.Template.detect(file.getString("tokenizer.chat_template")) orelse {
        try stderr.writeAll("this GGUF has no recognizable chat template\n");
        return error.NoChatTemplate;
    };
    file.deinit();
    file_alive = false;

    if (options.cartridge_path) |path| box.cart = try loadCartridge(io, allocator, stderr, ctx, &box.model, path);
    errdefer if (box.cart) |*c| c.deinit();

    if (options.fleet_dir) |dir| {
        box.fleet_serve = try FleetServeQwen3.init(io, allocator, stderr, ctx, &box.model, &box.tokenizer, dir);
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

    const kv_slots = try gguf_chat.kvRamGuardSlots(qwen3_model.Model, ctx, &box.model, options.context_len, try slotsForBatch(stderr, options), options.kv_slots_force, stderr);

    box.adapter = Qwen3Adapter.init(
        allocator,
        ctx,
        &box.model,
        &box.tokenizer,
        template,
        .{
            .model_id = box.model_id,
            .context_len = options.context_len,
            .think_markers = .{ .open = "<think>", .close = "</think>" },
            .supports_think = true,
            .tool_style = .hermes,
            // Qwen3's recommended no-think chat settings (the server default;
            // per-request reasoning switches nothing here — clients override).
            .default_sampling = .{ .temperature = 0.7, .top_k = 20, .top_p = 0.8 },
            .speculation = options.spec,
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
                .embedFn = FleetServeQwen3.embed,
                .rag_docs = options.rag_docs,
                .rag_chunks = options.rag_chunks,
                .adaptive = options.rag_adaptive,
                .switch_margin = options.rag_margin,
            } else null,
        },
    );
    return .{ .ptr = box, .destroyFn = Qwen3Box.destroy, .backend = box.adapter.backend() };
}

/// The qwen3 dense engine box behind `shine_fleet_dir`: the same chat
/// backend, decoding through the AdaptedModel box with per-request adapter
/// selection.
const ShineBox = struct {
    allocator: Allocator,
    model_id: []u8,
    model: qwen3_model.Model,
    tokenizer: tokenizer_mod.Tokenizer,
    fs: ShineFleetServe,
    adapter: ShineAdapter,

    fn destroy(ptr: *anyopaque) void {
        const box: *ShineBox = @ptrCast(@alignCast(ptr));
        const a = box.allocator;
        box.adapter.deinit();
        box.fs.deinit();
        box.tokenizer.deinit();
        box.model.deinit();
        a.free(box.model_id);
        a.destroy(box);
    }
};

fn openQwen3ShineFleet(
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    options: OpenOptions,
    stderr: *std.Io.Writer,
) !Opened {
    var file_alive = true;
    errdefer if (file_alive) file.deinit();

    const config = try qwen3_model.Config.fromGguf(file);
    if (config.isMoe()) {
        try stderr.writeAll("--shine-fleet needs a dense qwen3 base (SHINE adapts the dense linears)\n");
        return error.ShineFleetUnsupported;
    }

    const box = try allocator.create(ShineBox);
    errdefer allocator.destroy(box);
    box.allocator = allocator;
    box.model_id = try allocator.dupe(u8, model_id);
    errdefer allocator.free(box.model_id);

    box.model = try qwen3_model.Model.loadGgufFromFile(ctx, file, config);
    errdefer box.model.deinit();
    box.tokenizer = tokenizer_mod.Tokenizer.initFromGguf(allocator, file, .{}) catch {
        try stderr.writeAll("this GGUF has no usable tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    errdefer box.tokenizer.deinit();
    const template = chat.Template.detect(file.getString("tokenizer.chat_template")) orelse {
        try stderr.writeAll("this GGUF has no recognizable chat template\n");
        return error.NoChatTemplate;
    };
    file.deinit();
    file_alive = false;

    box.fs = try ShineFleetServe.init(io, allocator, stderr, ctx, &box.model, &box.tokenizer, options.shine_fleet_dir.?);
    errdefer box.fs.deinit();
    try stderr.print("shine fleet: {d} documents, {d} retrieval chunks, top-1 adapter per request\n", .{
        box.fs.manifest.value.docs.len,
        box.fs.index.len(),
    });
    try stderr.flush();

    const kv_slots = try gguf_chat.kvRamGuardSlots(qwen3_shine.AdaptedModel, ctx, &box.fs.box, options.context_len, try slotsForBatch(stderr, options), options.kv_slots_force, stderr);

    box.adapter = ShineAdapter.init(
        allocator,
        ctx,
        &box.fs.box,
        &box.tokenizer,
        template,
        .{
            .model_id = box.model_id,
            .context_len = options.context_len,
            .think_markers = .{ .open = "<think>", .close = "</think>" },
            .supports_think = true,
            .tool_style = .hermes,
            .default_sampling = .{ .temperature = 0.7, .top_k = 20, .top_p = 0.8 },
            .speculation = options.spec,
            .constraint_cache_len = @max(8, options.batch),
            .kv_slots = kv_slots,
            .shine_fleet = .{
                .index = &box.fs.index,
                .embed_ctx = &box.fs,
                .embedFn = ShineFleetServe.embed,
                .apply_ctx = &box.fs,
                .applyFn = ShineFleetServe.apply,
                .rag_chunks = options.rag_chunks,
            },
        },
    );
    return .{ .ptr = box, .destroyFn = ShineBox.destroy, .backend = box.adapter.backend() };
}

/// The gemma4 engine box (heap-pinned like `Qwen3Box`; `extra_stops_buf`
/// backs the adapter's borrowed `extra_stop_ids`).
const Gemma4Box = struct {
    allocator: Allocator,
    model_id: []u8,
    model: gemma4_mod.Model,
    tokenizer: spm_tokenizer_mod.Tokenizer,
    cart: ?cartridge_mod.Cartridge,
    fleet_serve: ?FleetServeGemma4,
    extra_stops_buf: [2]u32,
    extra_n: usize,
    adapter: Gemma4Adapter,

    fn destroy(ptr: *anyopaque) void {
        const box: *Gemma4Box = @ptrCast(@alignCast(ptr));
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

fn openGemma4(
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    options: OpenOptions,
    stderr: *std.Io.Writer,
) !Opened {
    var file_alive = true;
    errdefer if (file_alive) file.deinit();

    if (options.shine_fleet_dir != null) {
        try stderr.writeAll("--shine-fleet is a dense-qwen3 backend feature\n");
        return error.ShineFleetUnsupported;
    }
    var config = try gemma4_mod.Config.fromGguf(file);
    config.borrow_experts = options.experts_borrow;

    const box = try allocator.create(Gemma4Box);
    errdefer allocator.destroy(box);
    box.allocator = allocator;
    box.cart = null;
    box.fleet_serve = null;
    box.model_id = try allocator.dupe(u8, model_id);
    errdefer allocator.free(box.model_id);

    box.tokenizer = spm_tokenizer_mod.Tokenizer.initFromGguf(allocator, file, .{}) catch {
        try stderr.writeAll("this GGUF has no usable SPM tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    errdefer box.tokenizer.deinit();
    const template = chat.Template.detect(file.getString("tokenizer.chat_template")) orelse
        chat.Template{ .format = .gemma4 };
    const default_sampling = samplingFromGguf(file);

    box.model = try gemma4_mod.Model.loadGgufFromFile(ctx, file, config);
    errdefer box.model.deinit();
    file.deinit();
    file_alive = false;

    if (options.cartridge_path) |path| box.cart = try loadCartridge(io, allocator, stderr, ctx, &box.model, path);
    errdefer if (box.cart) |*c| c.deinit();

    if (options.fleet_dir) |dir| {
        if (config.num_experts > 0 and !config.borrow_experts) {
            // The query embedder forwards through the trainer, whose MoE
            // arm consumes raw expert blocks.
            try stderr.writeAll("--fleet on a gemma4 MoE GGUF needs --experts=borrow (the query embedder forwards through raw expert blocks)\n");
            return error.FleetUnsupported;
        }
        box.fleet_serve = try FleetServeGemma4.init(io, allocator, stderr, ctx, &box.model, &box.tokenizer, dir);
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

    // Turn-end ids beyond <turn|>: the GGUF's own EOS and a stray SPM <eos>
    // (id 1) — the gemma4 chat harness registers the same pair.
    box.extra_n = 0;
    if (box.tokenizer.eosId()) |e| {
        box.extra_stops_buf[box.extra_n] = e;
        box.extra_n += 1;
    }
    box.extra_stops_buf[box.extra_n] = 1;
    box.extra_n += 1;

    const kv_slots = try gguf_chat.kvRamGuardSlots(gemma4_mod.Model, ctx, &box.model, options.context_len, try slotsForBatch(stderr, options), options.kv_slots_force, stderr);

    box.adapter = Gemma4Adapter.init(
        allocator,
        ctx,
        &box.model,
        &box.tokenizer,
        template,
        .{
            .model_id = box.model_id,
            .context_len = options.context_len,
            .extra_stop_ids = box.extra_stops_buf[0..box.extra_n],
            .default_sampling = default_sampling,
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
                .embedFn = FleetServeGemma4.embed,
                .rag_docs = options.rag_docs,
                .rag_chunks = options.rag_chunks,
                .adaptive = options.rag_adaptive,
                .switch_margin = options.rag_margin,
            } else null,
        },
    );
    return .{ .ptr = box, .destroyFn = Gemma4Box.destroy, .backend = box.adapter.backend() };
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
                var probe = try model.initKvCache(ctx, cart.p + 1);
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

const FleetServeQwen3 = FleetServeFor(
    qwen3_model.Model,
    qwen3_train.Trainer(.{ .q = false, .v = false }),
    tokenizer_mod.Tokenizer,
);
const FleetServeGemma4 = FleetServeFor(
    gemma4_mod.Model,
    gemma_train.Trainer(.{ .q = false, .v = false }),
    spm_tokenizer_mod.Tokenizer,
);

/// SHINE adapter fleet serving state (REFERENCE §13.12): the manifest's doc
/// names and adapter files, the retrieval index (the cartridge fleets'
/// EmbedIndex, same `embed_suffix` contract), the query-embedding trainer,
/// a small adapter LRU, and the AdaptedModel box the backend decodes
/// through. `apply` runs on the single inference worker only.
const ShineFleetServe = struct {
    const Trainer = qwen3_train.Trainer(.{ .q = false, .v = false });
    const max_resident = 4;
    const Entry = struct { doc: usize, set: *qwen3_shine.LoraSet, last: u64 };
    const ManifestDoc = struct { name: []const u8, adapter_file: []const u8 };
    const ManifestJson = struct { embed_chunk: usize = 256, docs: []ManifestDoc };

    allocator: Allocator,
    io: std.Io,
    ctx: *fucina.ExecContext,
    tokenizer: *const tokenizer_mod.Tokenizer,
    dir: []const u8,
    manifest: std.json.Parsed(ManifestJson),
    index: cartridge_fleet.EmbedIndex,
    trainer: Trainer,
    box: qwen3_shine.AdaptedModel,
    resident: std.ArrayList(Entry) = .empty,
    clock: u64 = 0,

    fn init(
        io: std.Io,
        allocator: Allocator,
        stderr: *std.Io.Writer,
        ctx: *fucina.ExecContext,
        model: *const qwen3_model.Model,
        tokenizer: *const tokenizer_mod.Tokenizer,
        dir: []const u8,
    ) !ShineFleetServe {
        const manifest_path = try std.fs.path.join(allocator, &.{ dir, "shine-fleet.json" });
        defer allocator.free(manifest_path);
        const manifest_bytes = std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, .limited(1 << 20)) catch |err| {
            try stderr.print("--shine-fleet {s}: cannot read shine-fleet.json ({t})\n", .{ dir, err });
            return err;
        };
        defer allocator.free(manifest_bytes);
        // alloc_always: the parsed strings must not borrow manifest_bytes,
        // which is freed when init returns.
        var manifest = std.json.parseFromSlice(ManifestJson, allocator, manifest_bytes, .{ .allocate = .alloc_always }) catch |err| {
            try stderr.print("--shine-fleet {s}: malformed shine-fleet.json ({t})\n", .{ dir, err });
            return err;
        };
        errdefer manifest.deinit();
        if (manifest.value.docs.len == 0) {
            try stderr.print("--shine-fleet {s}: the manifest lists no documents\n", .{dir});
            return error.EmptyFleet;
        }

        const index_path = try std.fs.path.join(allocator, &.{ dir, "index.safetensors" });
        defer allocator.free(index_path);
        var mapped = cartridge_fleet.mmapFile(io, index_path) catch |err| {
            try stderr.print("--shine-fleet {s}: cannot open index.safetensors ({t})\n", .{ dir, err });
            return err;
        };
        defer mapped.deinit();
        var index = try cartridge_fleet.EmbedIndex.initFromBytes(allocator, mapped.bytes);
        errdefer index.deinit();
        if (index.dim != model.config.hidden_size) {
            try stderr.print("--shine-fleet {s}: index dim {d} does not match model hidden {d} — rebuild the fleet against this base\n", .{ dir, index.dim, model.config.hidden_size });
            return error.MissingIndex;
        }

        var trainer = Trainer.init(ctx, model, .{ .rank = 1, .alpha = 1 }, 0) catch |err| {
            try stderr.writeAll("--shine-fleet: this GGUF cannot host the query-embedding trainer (dense qwen3 required)\n");
            return err;
        };
        errdefer trainer.deinit();

        return .{
            .allocator = allocator,
            .io = io,
            .ctx = ctx,
            .tokenizer = tokenizer,
            .dir = dir,
            .manifest = manifest,
            .index = index,
            .trainer = trainer,
            .box = qwen3_shine.AdaptedModel.init(model),
        };
    }

    fn deinit(self: *ShineFleetServe) void {
        self.box.adapter = null;
        for (self.resident.items) |entry| {
            entry.set.deinit();
            self.allocator.destroy(entry.set);
        }
        self.resident.deinit(self.allocator);
        self.trainer.deinit();
        self.index.deinit();
        self.manifest.deinit();
    }

    /// `gguf_chat.ShineFleetOptions.embedFn`: the exact recipe the index was
    /// built with — text ids ++ `embed_suffix` ids, final-norm last hidden
    /// state through the PLAIN base (the trainer holds the base model, not
    /// the box, so retrieval is adapter-independent).
    fn embed(ptr: *anyopaque, text: []const u8, out: []f32) anyerror!void {
        const self: *ShineFleetServe = @ptrCast(@alignCast(ptr));
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

    /// `gguf_chat.ShineFleetOptions.applyFn`: point the box at `doc`'s
    /// adapter, loading it through the LRU on a miss.
    fn apply(ptr: *anyopaque, doc: ?usize) anyerror!void {
        const self: *ShineFleetServe = @ptrCast(@alignCast(ptr));
        const target = doc orelse {
            self.box.adapter = null;
            return;
        };
        if (target >= self.manifest.value.docs.len) return error.InvalidSelection;
        self.clock += 1;
        for (self.resident.items) |*entry| {
            if (entry.doc == target) {
                entry.last = self.clock;
                self.box.adapter = entry.set;
                return;
            }
        }
        // Miss: evict the LRU entry first (the box is re-pointed on every
        // request, so no live pointer survives an eviction).
        self.box.adapter = null;
        if (self.resident.items.len >= max_resident) {
            var lru_i: usize = 0;
            for (self.resident.items, 0..) |entry, i| {
                if (entry.last < self.resident.items[lru_i].last) lru_i = i;
            }
            const victim = self.resident.swapRemove(lru_i);
            victim.set.deinit();
            self.allocator.destroy(victim.set);
        }
        const path = try std.fs.path.join(self.allocator, &.{ self.dir, self.manifest.value.docs[target].adapter_file });
        defer self.allocator.free(path);
        const set = try self.allocator.create(qwen3_shine.LoraSet);
        errdefer self.allocator.destroy(set);
        set.* = try qwen3_shine.loadLoraGguf(self.ctx, self.io, path, self.box.config);
        errdefer set.deinit();
        try self.resident.append(self.allocator, .{ .doc = target, .set = set, .last = self.clock });
        self.box.adapter = set;
    }
};

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
    var probe = try model.initKvCache(ctx, cart.p + 1);
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

/// GGUF-recommended sampling (`general.sampling.*`), as the gemma4 chat
/// harness reads it.
pub fn samplingFromGguf(file: *const fucina.gguf.File) sampler.Config {
    return .{
        .temperature = if (file.getFloat("general.sampling.temp")) |v| @floatCast(v) else 1.0,
        .top_k = if (file.getInt("general.sampling.top_k")) |v| @intCast(@max(@as(i64, 0), v)) else 64,
        .top_p = if (file.getFloat("general.sampling.top_p")) |v| @floatCast(v) else 0.95,
        .min_p = if (file.getFloat("general.sampling.min_p")) |v| @floatCast(v) else 0.0,
        .repeat_penalty = if (file.getFloat("general.sampling.penalty_repeat")) |v| @floatCast(v) else 1.0,
        .freq_penalty = if (file.getFloat("general.sampling.penalty_freq")) |v| @floatCast(v) else 0.0,
        .presence_penalty = if (file.getFloat("general.sampling.penalty_present")) |v| @floatCast(v) else 0.0,
        .repeat_last_n = if (file.getInt("general.sampling.penalty_last_n")) |v| @intCast(@max(@as(i64, 0), v)) else 64,
    };
}

/// A lockstep batch needs one resident KV slot per stream; raise the
/// requested pool to the batch width (the RAM guard then prices the total).
fn slotsForBatch(stderr: *std.Io.Writer, options: OpenOptions) !usize {
    if (options.batch > options.kv_slots) {
        try stderr.print("--batch {d}: raising --kv-slots {d} -> {d} (one resident slot per lockstep stream)\n", .{ options.batch, options.kv_slots, options.batch });
        try stderr.flush();
        return options.batch;
    }
    return options.kv_slots;
}
