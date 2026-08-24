//! SHINE adapter-fleet serving for the dense qwen3 family: `open` /
//! `openFromFile` mirror `serving.open` / `serving.openFromFile` with the
//! fleet directory as an explicit argument and return the same `Opened`
//! handle, decoding through the `qwen3/shine.zig` AdaptedModel box with
//! per-request adapter selection (see docs/reference/13-the-model-stack-fucina_models.md). The generic engine
//! (`serving/open.zig`) carries no SHINE branch; `examples/lmserve` routes
//! its `--shine-fleet` flag here. Public as `models.qwen3.shine_serving`.

const std = @import("std");
const fucina = @import("fucina");
const serving_open = @import("../text/serving/open.zig");
const gguf_chat = @import("../text/serving/gguf_chat.zig");
const chat = @import("../text/chat.zig");
const tokenizer_mod = @import("../text/tokenizer.zig");
const cartridge_fleet = @import("../text/cartridge_fleet.zig");
const qwen3_model = @import("model.zig");
const qwen3_train = @import("train.zig");
const qwen3_shine = @import("shine.zig");

const Allocator = std.mem.Allocator;
const OpenOptions = serving_open.OpenOptions;
const Opened = serving_open.Opened;

/// `serving.open` counterpart for a SHINE adapter fleet: open `gguf_path`
/// (dense qwen3 only) and serve it with `fleet_dir`'s per-document
/// adapters. `options` is the shared engine surface; combinations the
/// adapter box cannot host (`fleet_dir`, `cartridge_path`, `kv_cache_dir`,
/// `batch > 1`) are rejected with `error.InvalidOptions`.
pub fn open(
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    gguf_path: []const u8,
    fleet_dir: []const u8,
    options: OpenOptions,
    stderr: *std.Io.Writer,
) !Opened {
    var file = try fucina.gguf.File.loadMmapAuto(allocator, io, gguf_path);
    const model_id = std.fs.path.stem(std.fs.path.basename(gguf_path));
    return openFromFile(ctx, io, allocator, &file, model_id, fleet_dir, options, stderr);
}

/// `open` over an already-loaded GGUF. Takes ownership of `file` on every
/// path, like `serving.openFromFile`.
pub fn openFromFile(
    ctx: *fucina.ExecContext,
    io: std.Io,
    allocator: Allocator,
    file: *fucina.gguf.File,
    model_id: []const u8,
    fleet_dir: []const u8,
    options: OpenOptions,
    stderr: *std.Io.Writer,
) !Opened {
    if (options.fleet_dir != null or options.cartridge_path != null or
        options.kv_cache_dir != null or options.batch > 1)
    {
        file.deinit();
        return error.InvalidOptions;
    }
    const arch = file.getString("general.architecture") orelse {
        file.deinit();
        return error.UnknownArchitecture;
    };
    if (std.mem.eql(u8, arch, "qwen3") or std.mem.eql(u8, arch, "qwen3moe")) {
        return openQwen3ShineFleet(ctx, io, allocator, file, model_id, fleet_dir, options, stderr);
    }
    if (std.mem.eql(u8, arch, "gemma4")) {
        file.deinit();
        try stderr.writeAll("--shine-fleet is a dense-qwen3 backend feature\n");
        return error.ShineFleetUnsupported;
    }
    file.deinit();
    return error.UnsupportedArchitecture;
}

const ShineAdapter = gguf_chat.GgufChatBackend(qwen3_shine.AdaptedModel, tokenizer_mod);

/// The qwen3 dense engine box behind the SHINE fleet: the same chat
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
    fleet_dir: []const u8,
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

    box.fs = try ShineFleetServe.init(io, allocator, stderr, ctx, &box.model, &box.tokenizer, fleet_dir);
    errdefer box.fs.deinit();
    try stderr.print("shine fleet: {d} documents, {d} retrieval chunks, top-1 adapter per request\n", .{
        box.fs.manifest.value.docs.len,
        box.fs.index.len(),
    });
    try stderr.flush();

    const kv_slots = try gguf_chat.kvRamGuardSlots(qwen3_shine.AdaptedModel, ctx, &box.fs.box, options.context_len, try serving_open.slotsForBatch(stderr, options), options.kv_slots_force, stderr);

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

/// SHINE adapter fleet serving state (see docs/reference/13-the-model-stack-fucina_models.md): the manifest's doc
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
