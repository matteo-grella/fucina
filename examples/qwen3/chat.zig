//! Chat mode of the Qwen3 runner: streamed single-turn (`--chat`) or
//! interactive multi-turn REPL (`--repl`), with optional KV persistence and
//! speculative decoding.
const std = @import("std");
const fucina = @import("fucina");
const models = @import("fucina_models");
const util = @import("util.zig");

/// Streamed single-turn or interactive multi-turn chat. The reply streams to
/// `stdout` token-by-token; a `Conversation` keeps the KV cache across turns.
pub fn runChat(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const models.qwen3.model.Model,
    tok: *const models.text.tokenizer.Tokenizer,
    template: models.text.chat.Template,
    system: ?[]const u8,
    no_think: bool,
    sampler_cfg: models.text.sampler.Config,
    chat_text: ?[]const u8,
    spec: bool,
    spec_refs: []const []const u8,
    processor: ?models.text.sampler.LogitProcessor,
    kv_save_path: ?[]const u8,
) !void {
    var convo = try models.text.chat.Conversation(models.qwen3.model.Model, models.text.tokenizer).init(ctx, model, tok, template, .{
        .system = system,
        .think_off = no_think,
        .sampler = sampler_cfg,
        .capacity = 4096,
        .max_response_tokens = 1024,
        .logit_processor = processor,
        .speculation = spec,
        .io = io,
    });
    defer convo.deinit();
    if (kv_save_path) |path| {
        const resumed = try convo.enablePersistence(io, path);
        if (resumed > 0) try stdout.print("[kv: conversation resumed from {s} — {d} tokens, no re-prefill]\n", .{ path, resumed });
    }
    for (spec_refs) |path| {
        const ids = try util.tokenizeFile(io, allocator, tok, path);
        defer allocator.free(ids);
        try convo.addSpecReference(ids);
    }

    if (chat_text) |msg| {
        _ = try convo.send(msg, stdout);
        try stdout.writeAll("\n");
        if (convo.specStats()) |stats| {
            try stats.writeSummary(stdout);
            try stdout.writeAll("\n");
        }
        try stdout.flush();
        return;
    }

    // REPL: read a line per turn, stream the reply, keep the cache.
    var stdin_buf: [8192]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);
    const in = &stdin_reader.interface;
    try stdout.print("chat ({s}) — type a message, empty line or Ctrl-D to quit:\n", .{@tagName(template.format)});
    while (true) {
        try stdout.writeAll("\n> ");
        try stdout.flush();
        const line = (try in.takeDelimiter('\n')) orelse break;
        const msg = std.mem.trim(u8, line, " \t\r");
        if (msg.len == 0) break;
        try stdout.writeAll("\n");
        _ = convo.send(msg, stdout) catch |e| switch (e) {
            error.ContextFull => {
                try stdout.print("[context full — restart to continue]\n", .{});
                break;
            },
            else => return e,
        };
        try stdout.writeAll("\n");
        try stdout.flush();
    }
}
