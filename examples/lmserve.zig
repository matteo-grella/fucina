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

const types = @import("lmserve/types.zig");
const backend_mod = @import("lmserve/backend.zig");
const backend_nanochat = @import("lmserve/backend_nanochat.zig");
const backend_diffusion = @import("lmserve/backend_diffusion.zig");
const scheduler_mod = @import("lmserve/scheduler.zig");
const http_mod = @import("lmserve/http.zig");

const usage_text =
    \\fucina lmserve — OpenAI-compatible LM server (chat completions + responses)
    \\
    \\usage: zig build lmserve -Doptimize=ReleaseFast [-Dllguidance=true] -- <model.gguf> [flags]
    \\       zig build lmserve -Doptimize=ReleaseFast -- --nanochat <checkpoint dir> [flags]
    \\
    \\The GGUF's general.architecture picks the backend: qwen3, qwen3moe,
    \\gemma4, diffusion-gemma. --nanochat serves a nanochat checkpoint dir
    \\(model.safetensors + tokenizer.bin).
    \\
    \\  --host H            bind address (default 127.0.0.1)
    \\  --port N            port (default 8080)
    \\  --ctx N             per-request context budget in tokens (default 4096)
    \\  --api-key K         require Authorization: Bearer K
    \\  --queue N           max queued requests before 429 (default 16)
    \\  --conns N           max concurrent connections (default 32)
    \\  --experts=borrow    zero-copy MoE expert load (gemma4/diffusion-gemma)
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
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    if (args.nanochat_dir) |dir| {
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
    } else {
        try stderr.print("unsupported architecture for serving: {s} (supported: qwen3, qwen3moe, gemma4, diffusion-gemma)\n", .{arch});
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
        },
    );
    defer adapter.deinit();
    try serveWith(io, allocator, adapter.backend(), args);
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
    _ = @import("lmserve/types.zig");
    _ = @import("lmserve/backend.zig");
    _ = @import("lmserve/backend_nanochat.zig");
    _ = @import("lmserve/backend_diffusion.zig");
    _ = @import("lmserve/scheduler.zig");
    _ = @import("lmserve/openai.zig");
    _ = @import("lmserve/emitter.zig");
    _ = @import("lmserve/http.zig");
}
