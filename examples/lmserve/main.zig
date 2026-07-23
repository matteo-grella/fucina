//! OpenAI-compatible HTTP server over the in-tree language models: Chat
//! Completions (`POST /v1/chat/completions`) plus the stateless Responses
//! API (`POST /v1/responses`), with SSE streaming, JSON-schema/regex/Lark
//! constrained output (`-Dllguidance=true` builds), and a bounded request
//! queue in front of one sequential inference worker.
//!
//! The model family is dispatched from the GGUF's `general.architecture`
//! (qwen3 / qwen3moe / gemma4 / diffusion-gemma); nanochat checkpoints load
//! via `--nanochat <dir>`. Run with `zig build lmserve -- <model.gguf> [flags]`.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const types = @import("types.zig");
const backend_mod = @import("backend.zig");
const backend_nanochat = @import("backend_nanochat.zig");
const backend_diffusion = @import("backend_diffusion.zig");
const backend_inkling = @import("backend_inkling.zig");
const backend_qwen35 = @import("backend_qwen35.zig");
const scheduler_mod = @import("scheduler.zig");
const http_mod = @import("http.zig");

const usage_text =
    \\fucina lmserve — OpenAI-compatible LM server (chat completions + responses)
    \\
    \\usage: zig build lmserve -Doptimize=ReleaseFast [-Dllguidance=true] -- <model.gguf> [flags]
    \\       zig build lmserve -Doptimize=ReleaseFast -- --nanochat <checkpoint dir> [flags]
    \\
    \\The GGUF's general.architecture picks the backend: qwen3, qwen3moe,
    \\qwen35 (Qwen3.5 / Ternary-Bonsai), gemma4, diffusion-gemma, inkling.
    \\--nanochat serves a nanochat checkpoint dir (model.safetensors +
    \\tokenizer.bin).
    \\
    \\  --host H            bind address (default 127.0.0.1)
    \\  --port N            port (default 8080)
    \\  --ctx N             per-request context budget in tokens (default 4096)
    \\  --api-key K         require Authorization: Bearer K
    \\  --queue N           max queued requests before 429 (default 16)
    \\  --conns N           max concurrent connections (default 32)
    \\  --experts=borrow    zero-copy MoE expert load (gemma4/diffusion-gemma)
    \\  --kv-slots N        resident KV-reuse slots (default 1); each holds a
    \\                      full --ctx cache, so N-1 extra slots cost real
    \\                      memory but keep interleaved conversations warm.
    \\                      A startup guard checks N x per-slot bytes against
    \\                      available RAM: Linux clamps on overcommit, macOS
    \\                      warns (its probe is conservative)
    \\  --kv-slots-force    keep the requested --kv-slots even when the guard
    \\                      would clamp (Linux; the warning still prints)
    \\  --kv-cache-dir D    spill evicted slots to sidecar files under D and
    \\                      restore them on prefix match (gguf chat backends)
    \\  --kv-disk-slots M   max sidecar files under --kv-cache-dir (default 8)
    \\  --cartridge F       preload a trained KV-prefix cartridge (safetensors from
    \\                      `zig build cartridge`; docs/CARTRIDGES.md) into every
    \\                      conversation — served "prior knowledge" without prompt
    \\                      tokens (qwen3/gemma4 backends; composes with the slot
    \\                      pool and the --kv-cache-dir disk tier)
    \\  --fleet DIR         serve a per-document cartridge fleet (from `zig build
    \\                      cartridge-fleet`; Cartridges at Scale): each request's
    \\                      last user message picks cartridges via the fleet's
    \\                      cosine index and they compose as the conversation's
    \\                      prefix (qwen3/gemma4 backends — gemma4 needs
    \\                      --experts=borrow on MoE GGUFs; excludes --cartridge
    \\                      and --kv-cache-dir; slot reuse is keyed by selection)
    \\  --rag-docs K        fleet: documents composed per request (default 2)
    \\  --rag-chunks N      fleet: cosine top-N chunks scanned (default 8)
    \\  --rag-adaptive      fleet: follow-up turns may SWITCH knowledge base when
    \\                      a document outside the conversation's selection
    \\                      decisively out-scores it (margin --rag-margin,
    \\                      default 0.05) under the contextual query; default is
    \\                      fully sticky (selection pinned at conversation start)
    \\  --rag-margin F      fleet: the adaptive switch margin (cosine units)
    \\
    \\Reasoning is off by default; clients enable it per request via
    \\reasoning_effort (chat) or reasoning.effort (responses).
    \\
    \\endpoints: POST /v1/chat/completions   POST /v1/responses
    \\           GET  /v1/models             GET  /health
    \\
;

const Args = struct {
    model_path: ?[]const u8 = null,
    nanochat_dir: ?[]const u8 = null,
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    ctx_len: usize = 4096,
    api_key: ?[]const u8 = null,
    queue: usize = 16,
    conns: usize = 32,
    experts_borrow: bool = false,
    kv_slots: usize = 1,
    kv_slots_force: bool = false,
    kv_cache_dir: ?[]const u8 = null,
    kv_disk_slots: usize = 8,
    cartridge_path: ?[]const u8 = null,
    fleet_dir: ?[]const u8 = null,
    rag_docs: usize = 2,
    rag_chunks: usize = 8,
    rag_adaptive: bool = false,
    rag_margin: f32 = 0.05,
};

var g_shutdown = std.atomic.Value(bool).init(false);
var g_listener_fd = std.atomic.Value(i64).init(-1);

fn onSignal(_: std.posix.SIG) callconv(.c) void {
    g_shutdown.store(true, .release);
    const fd = g_listener_fd.load(.acquire);
    // shutdown(2) is async-signal-safe; it unblocks the accept loop.
    if (fd >= 0) _ = std.c.shutdown(@intCast(fd), std.c.SHUT.RDWR);
}

fn installSignalHandlers() void {
    var action: std.posix.Sigaction = .{
        .handler = .{ .handler = onSignal },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(.INT, &action, null);
    std.posix.sigaction(.TERM, &action, null);
}

pub fn main(init: std.process.Init) !void {
    const args_slice = try init.minimal.args.toSlice(init.arena.allocator());
    const allocator = std.heap.smp_allocator;

    var stderr_buffer: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(init.io, &stderr_buffer);
    const stderr = &stderr_writer.interface;
    defer stderr.flush() catch {};

    var args = Args{};
    var i: usize = 1;
    while (i < args_slice.len) : (i += 1) {
        const arg = args_slice[i];
        if (std.mem.eql(u8, arg, "--host") and i + 1 < args_slice.len) {
            i += 1;
            args.host = args_slice[i];
        } else if (std.mem.eql(u8, arg, "--port") and i + 1 < args_slice.len) {
            i += 1;
            args.port = try std.fmt.parseInt(u16, args_slice[i], 10);
        } else if (std.mem.eql(u8, arg, "--ctx") and i + 1 < args_slice.len) {
            i += 1;
            args.ctx_len = try std.fmt.parseInt(usize, args_slice[i], 10);
        } else if (std.mem.eql(u8, arg, "--api-key") and i + 1 < args_slice.len) {
            i += 1;
            args.api_key = args_slice[i];
        } else if (std.mem.eql(u8, arg, "--queue") and i + 1 < args_slice.len) {
            i += 1;
            args.queue = try std.fmt.parseInt(usize, args_slice[i], 10);
        } else if (std.mem.eql(u8, arg, "--conns") and i + 1 < args_slice.len) {
            i += 1;
            args.conns = try std.fmt.parseInt(usize, args_slice[i], 10);
        } else if (std.mem.eql(u8, arg, "--nanochat") and i + 1 < args_slice.len) {
            i += 1;
            args.nanochat_dir = args_slice[i];
        } else if (std.mem.eql(u8, arg, "--kv-slots") and i + 1 < args_slice.len) {
            i += 1;
            args.kv_slots = try std.fmt.parseInt(usize, args_slice[i], 10);
        } else if (std.mem.eql(u8, arg, "--kv-slots-force")) {
            args.kv_slots_force = true;
        } else if (std.mem.eql(u8, arg, "--kv-cache-dir") and i + 1 < args_slice.len) {
            i += 1;
            args.kv_cache_dir = args_slice[i];
        } else if (std.mem.eql(u8, arg, "--kv-disk-slots") and i + 1 < args_slice.len) {
            i += 1;
            args.kv_disk_slots = try std.fmt.parseInt(usize, args_slice[i], 10);
        } else if (std.mem.eql(u8, arg, "--cartridge") and i + 1 < args_slice.len) {
            i += 1;
            args.cartridge_path = args_slice[i];
        } else if (std.mem.eql(u8, arg, "--fleet") and i + 1 < args_slice.len) {
            i += 1;
            args.fleet_dir = args_slice[i];
        } else if (std.mem.eql(u8, arg, "--rag-docs") and i + 1 < args_slice.len) {
            i += 1;
            args.rag_docs = try std.fmt.parseInt(usize, args_slice[i], 10);
        } else if (std.mem.eql(u8, arg, "--rag-chunks") and i + 1 < args_slice.len) {
            i += 1;
            args.rag_chunks = try std.fmt.parseInt(usize, args_slice[i], 10);
        } else if (std.mem.eql(u8, arg, "--rag-adaptive")) {
            args.rag_adaptive = true;
        } else if (std.mem.eql(u8, arg, "--rag-margin") and i + 1 < args_slice.len) {
            i += 1;
            args.rag_margin = try std.fmt.parseFloat(f32, args_slice[i]);
        } else if (std.mem.eql(u8, arg, "--experts=borrow")) {
            args.experts_borrow = true;
        } else if (std.mem.eql(u8, arg, "--experts=pack")) {
            args.experts_borrow = false;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try stderr.writeAll(usage_text);
            return;
        } else if (std.mem.startsWith(u8, arg, "--")) {
            try stderr.print("unknown flag: {s}\n\n{s}", .{ arg, usage_text });
            return error.UnknownArgument;
        } else {
            args.model_path = arg;
        }
    }
    if (args.fleet_dir != null) {
        if (args.cartridge_path != null) {
            try stderr.writeAll("--fleet and --cartridge are mutually exclusive (a fleet selects its own cartridges)\n");
            return error.InvalidArguments;
        }
        if (args.kv_cache_dir != null) {
            try stderr.writeAll("--fleet excludes --kv-cache-dir: KV sidecars do not record cartridge selections, so a restore could resurrect rows behind the wrong prefix\n");
            return error.InvalidArguments;
        }
    }
    if (args.kv_cache_dir) |dir| try std.Io.Dir.cwd().createDirPath(init.io, dir);

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    if (args.nanochat_dir) |dir| {
        if (args.cartridge_path != null or args.fleet_dir != null) {
            try stderr.writeAll("--cartridge/--fleet are supported by the GGUF chat backends only (qwen3/gemma4; --fleet is qwen3)\n");
            return error.CartridgeUnsupported;
        }
        const model_id = try allocator.dupe(u8, std.fs.path.basename(dir));
        defer allocator.free(model_id);
        var adapter = try backend_nanochat.NanochatBackend.load(allocator, &ctx, init.io, dir, model_id, args.ctx_len);
        defer adapter.deinit();
        return serveWith(init.io, allocator, adapter.backend(), args);
    }

    const model_path = args.model_path orelse {
        try stderr.writeAll(usage_text);
        return error.MissingModelPath;
    };

    var file = try fucina.gguf.File.loadMmap(allocator, init.io, model_path);
    const arch = file.getString("general.architecture") orelse {
        try stderr.writeAll("GGUF is missing general.architecture\n");
        return error.UnknownArchitecture;
    };

    const model_id = try allocator.dupe(u8, std.fs.path.stem(std.fs.path.basename(model_path)));
    defer allocator.free(model_id);

    if (std.mem.eql(u8, arch, "qwen3") or std.mem.eql(u8, arch, "qwen3moe")) {
        try serveQwen3(init.io, allocator, stderr, &ctx, &file, model_id, args);
    } else if (std.mem.eql(u8, arch, "gemma4")) {
        try serveGemma4(init.io, allocator, stderr, &ctx, &file, model_id, args);
    } else if (std.mem.eql(u8, arch, "diffusion-gemma")) {
        try serveDiffusion(init.io, allocator, stderr, &ctx, &file, model_id, args);
    } else if (std.mem.eql(u8, arch, "inkling")) {
        try serveInkling(init.io, allocator, stderr, &ctx, &file, model_id, args);
    } else if (std.mem.eql(u8, arch, "qwen35")) {
        try serveQwen35(init.io, allocator, stderr, &ctx, &file, model_id, args);
    } else {
        try stderr.print("unsupported architecture for serving: {s} (supported: qwen3, qwen3moe, qwen35, gemma4, diffusion-gemma, inkling)\n", .{arch});
        file.deinit();
        return error.UnsupportedArchitecture;
    }
}

fn serveDiffusion(
    io: std.Io,
    allocator: std.mem.Allocator,
    stderr: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    file: *fucina.gguf.File,
    model_id: []const u8,
    args: Args,
) !void {
    if (args.cartridge_path != null or args.fleet_dir != null) {
        try stderr.writeAll("--cartridge/--fleet are supported by the GGUF chat backends only (qwen3/gemma4; --fleet is qwen3)\n");
        return error.CartridgeUnsupported;
    }
    var config = try llm.diffusion_gemma.model.Config.fromGguf(file);
    config.base.borrow_experts = args.experts_borrow;
    var tokenizer = llm.spm_tokenizer.Tokenizer.initFromGguf(allocator, file, .{}) catch {
        try stderr.writeAll("this GGUF has no usable SPM tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    defer tokenizer.deinit();

    var model = try llm.diffusion_gemma.model.Model.loadGgufFromFile(ctx, file, config);
    defer model.deinit();
    file.deinit();

    var adapter = backend_diffusion.DiffusionBackend{
        .allocator = allocator,
        .ctx = ctx,
        .model = &model,
        .tokenizer = &tokenizer,
        .template = .{ .format = .gemma4 },
        .model_id = model_id,
        .context_len = args.ctx_len,
    };
    try serveWith(io, allocator, adapter.backend(), args);
}

fn serveInkling(
    io: std.Io,
    allocator: std.mem.Allocator,
    stderr: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    file: *fucina.gguf.File,
    model_id: []const u8,
    args: Args,
) !void {
    if (args.cartridge_path != null or args.fleet_dir != null) {
        try stderr.writeAll("--cartridge/--fleet are supported by the GGUF chat backends only (qwen3/gemma4; --fleet is qwen3)\n");
        return error.CartridgeUnsupported;
    }
    var tokenizer = llm.tokenizer.Tokenizer.initFromGguf(allocator, file, .{}) catch {
        try stderr.writeAll("this GGUF has no usable tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    defer tokenizer.deinit();

    var model = try llm.inkling.model.Model.loadGgufFromFile(ctx, file);
    defer model.deinit();
    file.deinit();

    var adapter = try backend_inkling.InklingBackend.init(allocator, ctx, &model, &tokenizer, model_id, args.ctx_len);
    defer adapter.deinit();
    try serveWith(io, allocator, adapter.backend(), args);
}

fn serveQwen35(
    io: std.Io,
    allocator: std.mem.Allocator,
    stderr: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    file: *fucina.gguf.File,
    model_id: []const u8,
    args: Args,
) !void {
    if (args.cartridge_path != null or args.fleet_dir != null) {
        try stderr.writeAll("--cartridge/--fleet are supported by the GGUF chat backends only (qwen3/gemma4; --fleet is qwen3)\n");
        return error.CartridgeUnsupported;
    }
    var tokenizer = llm.tokenizer.Tokenizer.initFromGguf(allocator, file, .{}) catch {
        try stderr.writeAll("this GGUF has no usable tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    defer tokenizer.deinit();
    const template = llm.chat.Template.detect(file.getString("tokenizer.chat_template")) orelse
        llm.chat.Template{ .format = .chatml };
    // Bonsai GGUFs carry their recommended sampling (`general.sampling.*`).
    const default_sampling = samplingFromGguf(file);

    const config = try llm.qwen35.model.Config.fromGguf(file);
    var model = try llm.qwen35.model.Model.loadGgufFromFile(ctx, file, config);
    defer model.deinit();
    file.deinit();

    var adapter = try backend_qwen35.Qwen35Backend.init(
        allocator,
        ctx,
        &model,
        &tokenizer,
        template,
        model_id,
        args.ctx_len,
        default_sampling,
    );
    defer adapter.deinit();
    try serveWith(io, allocator, adapter.backend(), args);
}

/// --fleet serving state: the fleet's manifest + cosine index plus a
/// no-adapter trainer whose `embedLastHidden` implements the fleet's
/// retrieval-embedding contract (`cartridge_fleet.embed_suffix`) for
/// incoming queries. One comptime instantiation per (model, trainer,
/// tokenizer) family; everything is borrowed by the backend's
/// `FleetOptions`, so this must outlive the server loop.
fn FleetServeFor(comptime ModelT: type, comptime TrainerT: type, comptime TokT: type) type {
    return struct {
    const FleetServe = @This();
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    tokenizer: *const TokT,
    fleet: llm.cartridge_fleet.Fleet,
    index: llm.cartridge_fleet.EmbedIndex,
    trainer: TrainerT,

    fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        stderr: *std.Io.Writer,
        ctx: *fucina.ExecContext,
        model: *const ModelT,
        tokenizer: *const TokT,
        dir: []const u8,
    ) !FleetServe {
        var fleet = llm.cartridge_fleet.Fleet.open(allocator, io, dir, 0, .{ .budget = 1 }) catch |err| {
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
        var mapped = try llm.cartridge_fleet.mmapFile(io, index_path);
        defer mapped.deinit();
        var index = try llm.cartridge_fleet.EmbedIndex.initFromBytes(allocator, mapped.bytes);
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
            var cart_mapped = try llm.cartridge_fleet.mmapFile(io, cart_path);
            defer cart_mapped.deinit();
            var cart = try llm.cartridge.Cartridge.initFromStateDict(ctx, allocator, cart_mapped.bytes);
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

    /// `backend.FleetOptions.embedFn`: the exact recipe the index was built
    /// with — text ids ++ separately tokenized `embed_suffix` ids, then the
    /// final-norm last hidden state (worker thread only, like generation).
    fn embed(ptr: *anyopaque, text: []const u8, out: []f32) anyerror!void {
        const self: *FleetServe = @ptrCast(@alignCast(ptr));
        const a = self.allocator;
        const text_ids = try self.tokenizer.encode(a, text);
        defer a.free(text_ids);
        const suffix_ids = try self.tokenizer.encode(a, llm.cartridge_fleet.embed_suffix);
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
    llm.qwen3.model.Model,
    llm.qwen3.train.Trainer(.{ .q = false, .v = false }),
    llm.tokenizer.Tokenizer,
);
const FleetServeGemma4 = FleetServeFor(
    llm.gemma.gemma4.Model,
    llm.gemma.gemma4_train.Trainer(.{ .q = false, .v = false }),
    llm.spm_tokenizer.Tokenizer,
);

fn serveQwen3(
    io: std.Io,
    allocator: std.mem.Allocator,
    stderr: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    file: *fucina.gguf.File,
    model_id: []const u8,
    args: Args,
) !void {
    var model = try llm.qwen3.model.Model.loadGgufFromFile(ctx, file, try llm.qwen3.model.Config.fromGguf(file));
    defer model.deinit();
    var tokenizer = llm.tokenizer.Tokenizer.initFromGguf(allocator, file, .{}) catch {
        try stderr.writeAll("this GGUF has no usable tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    defer tokenizer.deinit();
    const template = llm.chat.Template.detect(file.getString("tokenizer.chat_template")) orelse {
        try stderr.writeAll("this GGUF has no recognizable chat template\n");
        return error.NoChatTemplate;
    };
    file.deinit();

    var cart: ?llm.cartridge.Cartridge = null;
    defer if (cart) |*c| c.deinit();
    if (args.cartridge_path) |path| cart = try loadCartridge(io, allocator, stderr, ctx, &model, path);

    var fleet_serve: ?FleetServeQwen3 = null;
    defer if (fleet_serve) |*fs| fs.deinit();
    if (args.fleet_dir) |dir| {
        fleet_serve = try FleetServeQwen3.init(io, allocator, stderr, ctx, &model, &tokenizer, dir);
        try stderr.print("fleet: {d} documents, {d} retrieval chunks, {d} docs composed per request\n", .{
            fleet_serve.?.fleet.manifest.docs.items.len,
            fleet_serve.?.index.len(),
            args.rag_docs,
        });
        try stderr.flush();
    }

    const kv_slots = try backend_mod.kvRamGuardSlots(llm.qwen3.model.Model, ctx, &model, args.ctx_len, args.kv_slots, args.kv_slots_force, stderr);

    var adapter = backend_mod.GgufChatBackend(llm.qwen3.model.Model, llm.tokenizer).init(
        allocator,
        ctx,
        &model,
        &tokenizer,
        template,
        .{
            .model_id = model_id,
            .context_len = args.ctx_len,
            .think_markers = .{ .open = "<think>", .close = "</think>" },
            .supports_think = true,
            // Qwen3's recommended no-think chat settings (the server default;
            // per-request reasoning switches nothing here — clients override).
            .default_sampling = .{ .temperature = 0.7, .top_k = 20, .top_p = 0.8 },
            .kv_slots = kv_slots,
            .kv_disk = kvDiskOptions(io, args),
            .cartridge = if (cart) |*c| c else null,
            .fleet = if (fleet_serve) |*fs| .{
                .io = io,
                .dir = args.fleet_dir.?,
                .manifest = &fs.fleet.manifest,
                .index = &fs.index,
                .embed_ctx = fs,
                .embedFn = FleetServeQwen3.embed,
                .rag_docs = args.rag_docs,
                .rag_chunks = args.rag_chunks,
                .adaptive = args.rag_adaptive,
                .switch_margin = args.rag_margin,
            } else null,
        },
    );
    defer adapter.deinit();
    try serveWith(io, allocator, adapter.backend(), args);
}

fn serveGemma4(
    io: std.Io,
    allocator: std.mem.Allocator,
    stderr: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    file: *fucina.gguf.File,
    model_id: []const u8,
    args: Args,
) !void {
    var config = try llm.gemma.gemma4.Config.fromGguf(file);
    config.borrow_experts = args.experts_borrow;
    var tokenizer = llm.spm_tokenizer.Tokenizer.initFromGguf(allocator, file, .{}) catch {
        try stderr.writeAll("this GGUF has no usable SPM tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    defer tokenizer.deinit();
    const template = llm.chat.Template.detect(file.getString("tokenizer.chat_template")) orelse
        llm.chat.Template{ .format = .gemma4 };
    const default_sampling = samplingFromGguf(file);

    var model = try llm.gemma.gemma4.Model.loadGgufFromFile(ctx, file, config);
    defer model.deinit();
    file.deinit();

    var cart: ?llm.cartridge.Cartridge = null;
    defer if (cart) |*c| c.deinit();
    if (args.cartridge_path) |path| cart = try loadCartridge(io, allocator, stderr, ctx, &model, path);

    var fleet_serve: ?FleetServeGemma4 = null;
    defer if (fleet_serve) |*fs| fs.deinit();
    if (args.fleet_dir) |dir| {
        if (config.num_experts > 0 and !config.borrow_experts) {
            // The query embedder forwards through the trainer, whose MoE
            // arm consumes raw expert blocks.
            try stderr.writeAll("--fleet on a gemma4 MoE GGUF needs --experts=borrow (the query embedder forwards through raw expert blocks)\n");
            return error.FleetUnsupported;
        }
        fleet_serve = try FleetServeGemma4.init(io, allocator, stderr, ctx, &model, &tokenizer, dir);
        try stderr.print("fleet: {d} documents, {d} retrieval chunks, {d} docs composed per request\n", .{
            fleet_serve.?.fleet.manifest.docs.items.len,
            fleet_serve.?.index.len(),
            args.rag_docs,
        });
        try stderr.flush();
    }

    // Turn-end ids beyond <turn|>: the GGUF's own EOS and a stray SPM <eos>
    // (id 1) — the gemma4 chat harness registers the same pair.
    var extra_stops_buf: [2]u32 = undefined;
    var extra_n: usize = 0;
    if (tokenizer.eosId()) |e| {
        extra_stops_buf[extra_n] = e;
        extra_n += 1;
    }
    extra_stops_buf[extra_n] = 1;
    extra_n += 1;

    const kv_slots = try backend_mod.kvRamGuardSlots(llm.gemma.gemma4.Model, ctx, &model, args.ctx_len, args.kv_slots, args.kv_slots_force, stderr);

    var adapter = backend_mod.GgufChatBackend(llm.gemma.gemma4.Model, llm.spm_tokenizer).init(
        allocator,
        ctx,
        &model,
        &tokenizer,
        template,
        .{
            .model_id = model_id,
            .context_len = args.ctx_len,
            .extra_stop_ids = extra_stops_buf[0..extra_n],
            .default_sampling = default_sampling,
            .kv_slots = kv_slots,
            .kv_disk = kvDiskOptions(io, args),
            .cartridge = if (cart) |*c| c else null,
            .fleet = if (fleet_serve) |*fs| .{
                .io = io,
                .dir = args.fleet_dir.?,
                .manifest = &fs.fleet.manifest,
                .index = &fs.index,
                .embed_ctx = fs,
                .embedFn = FleetServeGemma4.embed,
                .rag_docs = args.rag_docs,
                .rag_chunks = args.rag_chunks,
                .adaptive = args.rag_adaptive,
                .switch_margin = args.rag_margin,
            } else null,
        },
    );
    defer adapter.deinit();
    try serveWith(io, allocator, adapter.backend(), args);
}

/// Load a trained cartridge (docs/CARTRIDGES.md) and probe it against the
/// model's KV geometry, so a mismatched file fails at startup instead of
/// mid-request.
fn loadCartridge(
    io: std.Io,
    allocator: std.mem.Allocator,
    stderr: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: anytype,
    path: []const u8,
) !llm.cartridge.Cartridge {
    const bytes = blk: {
        var dir = std.Io.Dir.cwd();
        break :blk try dir.readFileAlloc(io, path, allocator, .limited(1024 * 1024 * 1024));
    };
    defer allocator.free(bytes);
    var cart = try llm.cartridge.Cartridge.initFromStateDict(ctx, allocator, bytes);
    errdefer cart.deinit();
    var probe = try model.initKvCache(ctx, cart.p + 1);
    defer probe.deinit();
    cart.writeToCache(ctx, &probe) catch |err| {
        try stderr.print("cartridge {s} does not fit this model's KV geometry\n", .{path});
        return err;
    };
    return cart;
}

/// The gguf-chat backends' evict-to-disk tier config from the CLI flags
/// (`--kv-cache-dir` armed it; the directory exists — main created it).
fn kvDiskOptions(io: std.Io, args: Args) ?backend_mod.KvDiskOptions {
    const dir = args.kv_cache_dir orelse return null;
    return .{ .io = io, .dir = dir, .max_files = @max(args.kv_disk_slots, 1) };
}

/// Gemma's GGUF-recommended sampling (`general.sampling.*`), as the gemma4
/// chat harness reads it.
fn samplingFromGguf(file: *const fucina.gguf.File) llm.sampler.Config {
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

fn serveWith(io: std.Io, allocator: std.mem.Allocator, backend: types.Backend, args: Args) !void {
    var sched = scheduler_mod.Scheduler.init(allocator, io, backend, args.queue);
    try sched.start();
    defer sched.stop();

    var server = http_mod.Server{
        .allocator = allocator,
        .io = io,
        .opts = .{
            .host = args.host,
            .port = args.port,
            .api_key = args.api_key,
            .max_connections = args.conns,
        },
        .backend = backend,
        .sched = &sched,
        .shutdown = &g_shutdown,
    };

    try server.bind();
    g_listener_fd.store(@intCast(server.listenerHandle()), .release);
    installSignalHandlers();

    // A signal handler alone cannot end the accept loop: the Io layer
    // retries accept() on EINTR, and macOS does not wake a pending accept
    // when the handler shuts the listening socket down. The kicker thread
    // turns the flag flip into a real connection, which accept returns and
    // the loop then observes the flag on.
    const kicker = try std.Thread.spawn(.{}, shutdownKicker, .{ io, args.port });
    defer {
        g_shutdown.store(true, .release);
        kicker.join();
    }

    try server.run();
    std.log.info("shut down cleanly", .{});
}

fn shutdownKicker(io: std.Io, port: u16) void {
    while (!g_shutdown.load(.acquire)) {
        std.Io.sleep(io, .{ .nanoseconds = 100 * std.time.ns_per_ms }, .awake) catch {};
    }
    const addr = std.Io.net.IpAddress.parse("127.0.0.1", port) catch return;
    const stream = addr.connect(io, .{ .mode = .stream }) catch return;
    stream.close(io);
}

test {
    _ = @import("types.zig");
    _ = @import("backend.zig");
    _ = @import("backend_nanochat.zig");
    _ = @import("backend_diffusion.zig");
    _ = @import("scheduler.zig");
    _ = @import("openai.zig");
    _ = @import("emitter.zig");
    _ = @import("http.zig");
}
