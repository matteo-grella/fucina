//! OpenAI- and Anthropic-compatible HTTP server over the in-tree language
//! models: Chat Completions (`POST /v1/chat/completions`), the stateless
//! Responses API (`POST /v1/responses`), and the Anthropic Messages API
//! (`POST /v1/messages` — Claude Code works against it), with SSE
//! streaming, JSON-schema/regex/Lark constrained output
//! (`-Dllguidance=true` builds), and a bounded request queue in front of
//! one sequential inference worker.
//!
//! The model family is dispatched from the GGUF's `general.architecture`
//! (qwen3 / qwen3moe / qwen35 / qwen35moe / gemma4 / diffusion-gemma /
//! inkling / deepseek4); nanochat checkpoints load via `--nanochat <dir>`.
//! Run with `zig build lmserve -- <model.gguf> [flags]`.
//!
//! Thin front end: the transport (HTTP server, scheduler, wire dialects),
//! the generic engine (`GgufChatBackend`), and every GGUF family adapter
//! live in `models.text.serving` behind `serving.openFromFile` (dispatched through
//! the architecture registry); this main parses flags and keeps only the
//! two non-registry backends — diffusion-gemma (not an autoregressive
//! decoder) and nanochat (a checkpoint format, not GGUF).

const std = @import("std");
const fucina = @import("fucina");
const models = @import("fucina_models");

const types = @import("fucina_models").text.serving;
const backend_nanochat = @import("backend_nanochat.zig");
const backend_diffusion = @import("backend_diffusion.zig");
const scheduler_mod = types.scheduler;
const http_mod = types.http;

const usage_text =
    \\fucina lmserve — OpenAI- and Anthropic-compatible LM server
    \\
    \\usage: zig build lmserve -Doptimize=ReleaseFast [-Dllguidance=true] -- <model.gguf> [flags]
    \\       zig build lmserve -Doptimize=ReleaseFast -- --nanochat <checkpoint dir> [flags]
    \\
    \\The GGUF's general.architecture picks the backend: qwen3, qwen3moe,
    \\qwen35/qwen35moe (Qwen3.5 / Qwen3.6 / Ternary-Bonsai), gemma4,
    \\diffusion-gemma, inkling, deepseek4 (DeepSeek V4 Flash). --nanochat
    \\serves a nanochat checkpoint dir (model.safetensors + tokenizer.bin).
    \\
    \\  --host H            bind address (default 127.0.0.1)
    \\  --port N            port (default 8080)
    \\  --ctx N             per-request context budget in tokens (default 4096)
    \\  --api-key K         require Authorization: Bearer K
    \\  --queue N           max queued requests before 429 (default 16)
    \\  --conns N           max concurrent connections (default 32)
    \\  --allow-host H      accept Host header H (repeatable, max 8). The
    \\                      DNS-rebinding guard always accepts loopback names
    \\                      and the bind host; on non-loopback binds the
    \\                      check only arms when --allow-host is given
    \\  --cors-origin O     let browser pages from origin O ("*" for any)
    \\                      call the server. Default: no CORS headers, so
    \\                      cross-origin pages cannot read responses;
    \\                      non-browser clients are unaffected
    \\  --spec              speculative decoding for solo generations
    \\                      (qwen3/qwen3moe; self-draft cascade, lossless,
    \\                      stop sequences included). --batch groups of 2+
    \\                      decode plain automatically
    \\  --batch N           lockstep-decode up to N queued requests together
    \\                      (default 1 = strictly sequential; qwen3/qwen3moe/
    \\                      gemma4). Batching takes only what is already
    \\                      queued — an idle server keeps single-request
    \\                      latency — and raises --kv-slots to at least N.
    \\                      Streams in a batch of 4+ can differ from a solo
    \\                      run by float-reassociation drift (~1e-6), the
    \\                      speculative-verify caveat; excludes --fleet
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
    \\  --shine-fleet DIR   serve a per-document SHINE adapter fleet (from
    \\                      `zig build qwen3 -- ... --shine-fleet-build DIR`):
    \\                      each request's user messages pick ONE document by
    \\                      cosine retrieval and its saved adapter decodes the
    \\                      reply — zero context tokens, zero prefix rows
    \\                      (qwen3 dense; excludes --fleet/--cartridge/--batch)
    \\  --moe-stream        deepseek4: stream MoE expert weights from disk on
    \\                      demand (dense trunk stays resident); companions:
    \\                      --moe-cache-mb=N (streamed-tier RAM budget),
    \\                      --moe-pin-mb=N (pinned hot-expert tier budget),
    \\                      --moe-cache-slots=N (fixed LRU slots per layer),
    \\                      --moe-pilot (router-lookahead prefetch),
    \\                      --moe-no-learn (don't persist expert-usage counts),
    \\                      --moe-uncached, --moe-io-threads=N, --moe-mirror=P,
    \\                      --moe-mirror-weights=..., --moe-trace=P,
    \\                      --moe-l2=DIR, --moe-l2-build-gb=N
    \\                      (docs/RUNNING-MODELS.md documents the family)
    \\
    \\Reasoning is off by default; clients enable it per request via
    \\reasoning_effort (chat), reasoning.effort (responses), or thinking
    \\(anthropic messages). Anthropic clients (Claude Code, the SDKs) point
    \\ANTHROPIC_BASE_URL at this server; x-api-key carries --api-key.
    \\Function calling works on qwen3-family models in all three dialects
    \\(tool declarations render into the prompt; the client executes).
    \\
    \\endpoints: POST /v1/chat/completions   POST /v1/responses
    \\           POST /v1/messages           GET  /v1/models
    \\           GET  /health
    \\
;

pub const Args = struct {
    model_path: ?[]const u8 = null,
    nanochat_dir: ?[]const u8 = null,
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    ctx_len: usize = 4096,
    api_key: ?[]const u8 = null,
    cors_origin: ?[]const u8 = null,
    queue: usize = 16,
    conns: usize = 32,
    batch: usize = 1,
    spec: bool = false,
    experts_borrow: bool = false,
    kv_slots: usize = 1,
    kv_slots_force: bool = false,
    kv_cache_dir: ?[]const u8 = null,
    kv_disk_slots: usize = 8,
    cartridge_path: ?[]const u8 = null,
    fleet_dir: ?[]const u8 = null,
    shine_fleet_dir: ?[]const u8 = null,
    rag_docs: usize = 2,
    rag_chunks: usize = 8,
    rag_adaptive: bool = false,
    rag_margin: f32 = 0.05,
    allow_hosts: [8][]const u8 = undefined,
    allow_hosts_n: usize = 0,
    /// Streamed-expert flags (deepseek4 backend): the shared `--moe-*` set
    /// plus the family-specific levers the deepseek4 runner speaks. Slices
    /// parsed into `moe_cli` borrow argv.
    moe_cli: models.moe_stream_cli.MoeStreamCli = .{},
    moe_pilot: bool = false,
    moe_pin_mb: ?usize = null,
    moe_no_learn: bool = false,
    moe_cache_slots: ?usize = null,
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
        } else if (std.mem.eql(u8, arg, "--spec")) {
            args.spec = true;
        } else if (std.mem.eql(u8, arg, "--batch") and i + 1 < args_slice.len) {
            i += 1;
            args.batch = @max(try std.fmt.parseInt(usize, args_slice[i], 10), 1);
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
        } else if (std.mem.eql(u8, arg, "--shine-fleet") and i + 1 < args_slice.len) {
            i += 1;
            args.shine_fleet_dir = args_slice[i];
        } else if (std.mem.eql(u8, arg, "--cors-origin") and i + 1 < args_slice.len) {
            i += 1;
            args.cors_origin = args_slice[i];
        } else if (std.mem.eql(u8, arg, "--allow-host") and i + 1 < args_slice.len) {
            i += 1;
            if (args.allow_hosts_n >= args.allow_hosts.len) {
                try stderr.writeAll("too many --allow-host entries (max 8)\n");
                return error.InvalidArguments;
            }
            args.allow_hosts[args.allow_hosts_n] = args_slice[i];
            args.allow_hosts_n += 1;
        } else if (std.mem.eql(u8, arg, "--experts=borrow")) {
            args.experts_borrow = true;
        } else if (std.mem.eql(u8, arg, "--experts=pack")) {
            args.experts_borrow = false;
        } else if (try args.moe_cli.tryParse(arg)) {
            // Shared streamed-experts flags (models.moe_stream_cli.MoeStreamCli).
        } else if (std.mem.eql(u8, arg, "--moe-pilot")) {
            args.moe_cli.armed = true;
            args.moe_pilot = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-pin-mb=")) {
            args.moe_cli.armed = true;
            args.moe_pin_mb = try std.fmt.parseInt(usize, arg["--moe-pin-mb=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--moe-no-learn")) {
            args.moe_cli.armed = true;
            args.moe_no_learn = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-cache-slots=")) {
            args.moe_cli.armed = true;
            args.moe_cache_slots = try std.fmt.parseInt(usize, arg["--moe-cache-slots=".len..], 10);
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
        if (args.batch > 1) {
            try stderr.writeAll("--fleet excludes --batch: per-request retrieval and sticky slot adoption are single-stream logic\n");
            return error.InvalidArguments;
        }
    }
    if (args.shine_fleet_dir != null) {
        if (args.fleet_dir != null or args.cartridge_path != null) {
            try stderr.writeAll("--shine-fleet excludes --fleet/--cartridge: one knowledge mechanism per server\n");
            return error.InvalidArguments;
        }
        if (args.kv_cache_dir != null) {
            try stderr.writeAll("--shine-fleet excludes --kv-cache-dir: KV sidecars do not record adapter selections, so a restore could resurrect rows produced under another adapter\n");
            return error.InvalidArguments;
        }
        if (args.batch > 1) {
            try stderr.writeAll("--shine-fleet excludes --batch: the served adapter box holds one adapter at a time\n");
            return error.InvalidArguments;
        }
    }
    if (args.kv_cache_dir) |dir| try std.Io.Dir.cwd().createDirPath(init.io, dir);

    return serveBlocking(init.io, allocator, stderr, args);
}

/// Everything `main` does once the flags are parsed: open the GGUF, pick
/// the backend from `general.architecture` (the example-local adapters
/// here; `serving.openFromFile` for the `Conversation`-hosted families),
/// and run the server until it stops.
fn serveBlocking(
    io: std.Io,
    allocator: std.mem.Allocator,
    stderr: *std.Io.Writer,
    args: Args,
) !void {
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    if (args.nanochat_dir) |dir| {
        if (args.moe_cli.armed) {
            try stderr.writeAll("--moe-* streamed-expert flags are supported by the deepseek4 backend only\n");
            return error.MoeStreamUnsupported;
        }
        if (args.cartridge_path != null or args.fleet_dir != null or args.shine_fleet_dir != null) {
            try stderr.writeAll("--cartridge/--fleet are supported by the GGUF chat backends only (qwen3/gemma4; --fleet is qwen3)\n");
            return error.CartridgeUnsupported;
        }
        const model_id = try allocator.dupe(u8, std.fs.path.basename(dir));
        defer allocator.free(model_id);
        var adapter = try backend_nanochat.NanochatBackend.load(allocator, &ctx, io, dir, model_id, args.ctx_len);
        defer adapter.deinit();
        return serveWith(io, allocator, adapter.backend(), args);
    }

    const model_path = args.model_path orelse {
        try stderr.writeAll(usage_text);
        return error.MissingModelPath;
    };

    // loadMmapAuto follows llama.cpp split GGUFs (deepseek4-scale exports
    // ship in parts); single-file paths take the plain loadMmap route.
    var file = try fucina.gguf.File.loadMmapAuto(allocator, io, model_path);
    const arch = file.getString("general.architecture") orelse {
        try stderr.writeAll("GGUF is missing general.architecture\n");
        return error.UnknownArchitecture;
    };
    if (args.moe_cli.armed and !std.mem.eql(u8, arch, "deepseek4")) {
        // Reject instead of silently dropping (the caps philosophy).
        try stderr.print("--moe-* streamed-expert flags are supported by the deepseek4 backend only (this GGUF is {s})\n", .{arch});
        file.deinit();
        return error.MoeStreamUnsupported;
    }

    const model_id = try allocator.dupe(u8, std.fs.path.stem(std.fs.path.basename(model_path)));
    defer allocator.free(model_id);

    if (std.mem.eql(u8, arch, "diffusion-gemma")) {
        try serveDiffusion(io, allocator, stderr, &ctx, &file, model_id, args);
        return;
    }

    // Every GGUF family adapter lives in the library: the registry-driven
    // `serving.openFromFile` dispatches on the architecture; a SHINE
    // adapter fleet routes to its qwen3-family entry.
    // Streamed experts (`--moe-stream` + companions, deepseek4): the local
    // copy must outlive the open call — MoeStreamOptions borrows its
    // buffers.
    var moe_cli = args.moe_cli;
    var moe_stream = try moe_cli.options(model_path);
    if (moe_stream) |*m| {
        m.pilot = args.moe_pilot;
        if (args.moe_pin_mb) |mb| m.pin_bytes = mb << 20;
        if (args.moe_no_learn) m.auto_pin = false;
        if (args.moe_cache_slots) |n| m.cache_slots_per_layer = n;
    }
    const open_options = types.OpenOptions{
        .context_len = args.ctx_len,
        .spec = args.spec,
        .batch = args.batch,
        .experts_borrow = args.experts_borrow,
        .moe_stream = moe_stream,
        .kv_slots = args.kv_slots,
        .kv_slots_force = args.kv_slots_force,
        .kv_cache_dir = args.kv_cache_dir,
        .kv_disk_slots = args.kv_disk_slots,
        .cartridge_path = args.cartridge_path,
        .fleet_dir = args.fleet_dir,
        .rag_docs = args.rag_docs,
        .rag_chunks = args.rag_chunks,
        .rag_adaptive = args.rag_adaptive,
        .rag_margin = args.rag_margin,
    };
    var opened = if (args.shine_fleet_dir) |dir|
        try models.qwen3.shine_serving.openFromFile(&ctx, io, allocator, &file, model_id, dir, open_options, stderr)
    else
        types.openFromFile(&ctx, io, allocator, &file, model_id, open_options, stderr) catch |err| {
            if (err == error.UnsupportedArchitecture) {
                try stderr.print("unsupported architecture for serving: {s} (supported: qwen3, qwen3moe, qwen35, qwen35moe, gemma4, diffusion-gemma, inkling, deepseek4)\n", .{arch});
            }
            return err;
        };
    defer opened.deinit();
    // LIFO: the streamed-tier report + usage save runs BEFORE deinit
    // destroys the store.
    defer if (opened.expert_store) |store| {
        models.moe_stream_cli.reportAndSaveMoeStream(store, !args.moe_no_learn, stderr);
        stderr.flush() catch {};
    };
    try serveWith(io, allocator, opened.backend, args);
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
    if (args.cartridge_path != null or args.fleet_dir != null or args.shine_fleet_dir != null) {
        try stderr.writeAll("--cartridge/--fleet are supported by the GGUF chat backends only (qwen3/gemma4; --fleet is qwen3)\n");
        return error.CartridgeUnsupported;
    }
    var config = try models.diffusion_gemma.model.Config.fromGguf(file);
    config.base.borrow_experts = args.experts_borrow;
    var tokenizer = models.text.spm_tokenizer.Tokenizer.initFromGguf(allocator, file, .{}) catch {
        try stderr.writeAll("this GGUF has no usable SPM tokenizer metadata\n");
        return error.TokenizerUnavailable;
    };
    defer tokenizer.deinit();

    var model = try models.diffusion_gemma.model.Model.loadGgufFromFile(ctx, file, config);
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

fn serveWith(io: std.Io, allocator: std.mem.Allocator, backend: types.Backend, args: Args) !void {
    if (args.batch > 1 and !backend.supportsBatch()) {
        std.log.warn("this backend has no batched decode; serving --batch 1 (batching needs qwen3/qwen3moe/gemma4)", .{});
    }
    var sched = scheduler_mod.Scheduler.init(allocator, io, backend, args.queue, args.batch);
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
            .extra_hosts = args.allow_hosts[0..args.allow_hosts_n],
            .cors_origin = args.cors_origin,
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

// Every sibling .zig file in this directory is listed: a file referenced
// only from main()'s serve paths contributes ZERO tests to the test binary
// (Zig's lazy analysis — silently green), so presence in the directory must
// imply presence here. The models.text.serving band's tests live in the models root
// (`zig build test-models`): Zig collects tests from the root MODULE only, so
// a reference from this root cannot run them.
test {
    _ = @import("backend_nanochat.zig");
    _ = @import("backend_diffusion.zig");
}
