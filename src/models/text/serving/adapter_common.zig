//! The family serving adapters' shared skeleton. Every served family builds
//! the same heap-pinned box: model_id dupe, tokenizer from the GGUF,
//! template + default sampling per policy, `Family.load`, file handoff,
//! then the family's backend adapter over pointers into the box. The
//! engine-hosted families (qwen35, inkling, deepseek4, SHINE fleets)
//! instantiate `openFromFile` with a small comptime `Wiring`; the
//! `Conversation`-hosted families (qwen3, qwen3moe, gemma4) take the
//! generic `GgufChatBackend` box through `openChat` with `ChatTraits`.
//!
//! A Wiring provides:
//!   `Adapter` (the backend type stored in the box; `deinit` + `backend()`),
//!   `Extra` (box state between model and adapter; `void` = none, else it
//!   has `deinit` and `initExtra(built)` constructs it),
//!   `template: TemplatePolicy`, `sampling: SamplingPolicy`,
//!   `reports_expert_store: bool`, and `initAdapter(built)`.

const std = @import("std");
const fucina = @import("fucina");
const contract = @import("contract.zig");
const gguf_chat = @import("gguf_chat.zig");
const fleet_serve = @import("fleet_serve.zig");
const chat = @import("../chat.zig");
const sampler = @import("../sampler.zig");
const cartridge_mod = @import("../cartridge.zig");
const model_common = @import("../../model_common.zig");

const Allocator = std.mem.Allocator;

/// How the box resolves its chat template: `none` (the family renders chat
/// itself, at the token level), `detect_or_fallback` (the GGUF's template,
/// else `Family.template_fallback`, else `error.NoChatTemplate`), or
/// `detect_required` (the GGUF must carry a recognizable template).
pub const TemplatePolicy = enum { none, detect_or_fallback, detect_required };

/// Where the box's default sampling comes from: a fixed config, the GGUF's
/// own `general.sampling.*` metadata, or the GGUF metadata with a
/// family-specific fixup applied after the read.
pub const SamplingPolicy = union(enum) {
    fixed: sampler.Config,
    from_gguf,
    from_gguf_with: *const fn (file: *const fucina.gguf.File, cfg: *sampler.Config) void,
};

/// The family-independent form of `contract.OpenOptions` that `Family.load`
/// accepts; families ignore the levers they do not read.
pub fn loadOptionsFrom(options: contract.OpenOptions) model_common.FamilyLoadOptions {
    return .{ .experts_borrow = options.experts_borrow, .moe_stream = options.moe_stream };
}

/// The built box, as the Wiring hooks see it: pointers into the heap-pinned
/// box (stable until `Opened.deinit`) plus the open call's context.
pub fn Built(comptime Family: type, comptime W: type) type {
    return struct {
        allocator: Allocator,
        io: std.Io,
        ctx: *fucina.ExecContext,
        stderr: *std.Io.Writer,
        model_id: []u8,
        model: *Family.Model,
        tokenizer: *Family.Tokenizer,
        /// Null exactly when `W.template == .none`.
        template: ?chat.Template,
        default_sampling: sampler.Config,
        options: contract.OpenOptions,
        extra: *W.Extra,
    };
}

/// The served engine box (heap-pinned; the adapter holds pointers into it).
/// Takes ownership of `file` on every path.
fn EngineBox(comptime Family: type, comptime W: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        model_id: []u8,
        model: Family.Model,
        tokenizer: Family.Tokenizer,
        extra: W.Extra,
        adapter: W.Adapter,

        fn destroy(ptr: *anyopaque) void {
            const box: *Self = @ptrCast(@alignCast(ptr));
            const a = box.allocator;
            box.adapter.deinit();
            if (comptime W.Extra != void) box.extra.deinit();
            box.tokenizer.deinit();
            box.model.deinit();
            a.free(box.model_id);
            a.destroy(box);
        }
    };
}

/// The shared open path of the family serving adapters: build the engine
/// box for `Family` per `W`'s policies and return the ready `Opened`.
/// Takes ownership of `file` on every path: the weights are loaded (or the
/// error is returned) and the file is deinitialized before this returns.
/// `model_id` is copied.
pub fn openFromFile(
    comptime Family: type,
    comptime W: type,
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    options: contract.OpenOptions,
    stderr: *std.Io.Writer,
) !contract.Opened {
    const Box = EngineBox(Family, W);
    var file_alive = true;
    errdefer if (file_alive) file.deinit();

    const box = try allocator.create(Box);
    errdefer allocator.destroy(box);
    box.allocator = allocator;
    box.model_id = try allocator.dupe(u8, model_id);
    errdefer allocator.free(box.model_id);

    box.tokenizer = Family.tokenizer(allocator, file) catch {
        try stderr.writeAll("this GGUF has no usable tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    errdefer box.tokenizer.deinit();

    const template: ?chat.Template = switch (W.template) {
        .none => null,
        .detect_or_fallback => chat.Template.detect(file.getString("tokenizer.chat_template")) orelse blk: {
            if (Family.template_fallback) |format| break :blk chat.Template{ .format = format };
            try stderr.writeAll("this GGUF has no recognizable chat template\n");
            return error.NoChatTemplate;
        },
        .detect_required => chat.Template.detect(file.getString("tokenizer.chat_template")) orelse {
            try stderr.writeAll("this GGUF has no recognizable chat template\n");
            return error.NoChatTemplate;
        },
    };
    const default_sampling: sampler.Config = switch (W.sampling) {
        .fixed => |config| config,
        .from_gguf => contract.samplingFromGguf(file),
        .from_gguf_with => |fixup| blk: {
            var config = contract.samplingFromGguf(file);
            fixup(file, &config);
            break :blk config;
        },
    };

    box.model = try Family.load(ctx, file, loadOptionsFrom(options));
    errdefer box.model.deinit();
    file.deinit();
    file_alive = false;

    const built: Built(Family, W) = .{
        .allocator = allocator,
        .io = io,
        .ctx = ctx,
        .stderr = stderr,
        .model_id = box.model_id,
        .model = &box.model,
        .tokenizer = &box.tokenizer,
        .template = template,
        .default_sampling = default_sampling,
        .options = options,
        .extra = &box.extra,
    };
    if (comptime W.Extra != void) box.extra = try W.initExtra(built);
    errdefer if (comptime W.Extra != void) box.extra.deinit();

    box.adapter = try W.initAdapter(built);
    return .{
        .ptr = box,
        .destroyFn = Box.destroy,
        .backend = box.adapter.backend(),
        .expert_store = if (comptime W.reports_expert_store) box.model.expert_store else null,
    };
}

/// The per-family serving policy of the `Conversation`-hosted set: what
/// `GgufChatOptions` and the box's construction vary on.
pub const ChatTraits = struct {
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

/// The generic engine box of the `Conversation`-hosted families
/// (heap-pinned: the adapter and the fleet options hold pointers into it;
/// `extra_stops_buf` backs the adapter's borrowed `extra_stop_ids`).
fn ChatBox(comptime Family: type, comptime traits: ChatTraits) type {
    return struct {
        const Self = @This();
        const Adapter = gguf_chat.GgufChatBackend(Family.Model, Family.Tok);
        const FleetServe = if (traits.Trainer) |T| fleet_serve.FleetServeFor(Family.Model, T, Family.Tokenizer) else void;

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

/// The `Conversation`-hosted open path: the generic `GgufChatBackend` box
/// with the family's `ChatTraits` (KV slot pool + disk tier, speculative
/// decode, cartridges, cartridge fleets). Takes ownership of `file` on
/// every path, like `openFromFile`.
pub fn openChat(
    comptime Family: type,
    comptime traits: ChatTraits,
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    options: contract.OpenOptions,
    stderr: *std.Io.Writer,
) !contract.Opened {
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
        .from_gguf => contract.samplingFromGguf(file),
    };

    box.model = try Family.load(ctx, file, loadOptionsFrom(options));
    errdefer box.model.deinit();
    file.deinit();
    file_alive = false;

    if (options.cartridge_path) |path| box.cart = try fleet_serve.loadCartridge(io, allocator, stderr, ctx, &box.model, path);
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

/// The gguf-chat backends' evict-to-disk tier config (`kv_cache_dir` arms
/// it; the directory must exist).
fn kvDiskOptions(io: std.Io, options: contract.OpenOptions) ?gguf_chat.KvDiskOptions {
    const dir = options.kv_cache_dir orelse return null;
    return .{ .io = io, .dir = dir, .max_files = @max(options.kv_disk_slots, 1) };
}

/// A lockstep batch needs one resident KV slot per stream; raise the
/// requested pool to the batch width (the RAM guard then prices the total).
/// Shared with `models.qwen3.shine_serving`.
pub fn slotsForBatch(stderr: *std.Io.Writer, options: contract.OpenOptions) !usize {
    if (options.batch > options.kv_slots) {
        try stderr.print("--batch {d}: raising --kv-slots {d} -> {d} (one resident slot per lockstep stream)\n", .{ options.batch, options.kv_slots, options.batch });
        try stderr.flush();
        return options.batch;
    }
    return options.kv_slots;
}
