//! DeepSeek V4 Flash runner: greedy completion over the CSA/HCA trunk with
//! streamed experts. Validation against the official API vectors comes via
//! --vectors (chat-rendered prompts + greedy token comparison).
//!   zig build deepseek4 -- gguf/model.gguf --prompt "..." --gen 32 \
//!     --moe-stream --moe-cache-mb=20480
//!   zig build deepseek4 -- gguf/model.gguf --moe-stream \
//!     --vectors=path/to/ds4/tests/test-vectors/official
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const Model = llm.deepseek4.model.Model;
const Session = llm.deepseek4.model.Session;
const Tokenizer = llm.tokenizer.Tokenizer;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print("usage: zig build deepseek4 -- <model.gguf> --prompt \"...\" [--gen N] [--chat] [--moe-stream --moe-cache-mb=N] [--vectors=DIR [--vectors-max-prompt=N]]\n", .{});
        return;
    }

    var prompt_text: []const u8 = "The capital of France is";
    var gen_count: usize = 16;
    var moe_stream_flag = false;
    var moe_cache_mb: ?usize = null;
    var chat = false;
    var vectors_dir: ?[]const u8 = null;
    var vectors_max_prompt: usize = 256;
    var arg_i: usize = 2;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--prompt")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingPrompt;
            prompt_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--prompt=")) {
            prompt_text = arg["--prompt=".len..];
        } else if (std.mem.eql(u8, arg, "--gen")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingGenCount;
            gen_count = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--gen=")) {
            gen_count = try std.fmt.parseInt(usize, arg["--gen=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--chat")) {
            chat = true;
        } else if (std.mem.startsWith(u8, arg, "--vectors=")) {
            vectors_dir = arg["--vectors=".len..];
        } else if (std.mem.startsWith(u8, arg, "--vectors-max-prompt=")) {
            vectors_max_prompt = try std.fmt.parseInt(usize, arg["--vectors-max-prompt=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--moe-stream")) {
            moe_stream_flag = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-cache-mb=")) {
            moe_stream_flag = true;
            moe_cache_mb = try std.fmt.parseInt(usize, arg["--moe-cache-mb=".len..], 10);
        } else {
            try stdout.print("unknown flag: {s}\n", .{arg});
            return error.UnknownArgument;
        }
    }

    const allocator = std.heap.smp_allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const load_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    var file = try fucina.gguf.File.loadMmapAuto(allocator, init.io, args[1]);
    var tokenizer = try Tokenizer.initFromGguf(allocator, &file, .{});
    defer tokenizer.deinit();

    const load_options: Model.LoadOptions = if (moe_stream_flag) .{
        .moe_stream = .{
            .gguf_path = args[1],
            .cache_bytes = if (moe_cache_mb) |mb| mb << 20 else null,
        },
    } else .{};
    var model = try Model.loadGgufFromFileOptions(&ctx, &file, load_options);
    defer model.deinit();
    defer if (model.expert_store) |store| {
        const st = store.stats;
        std.debug.print("moe stream: hits {d} / misses {d} ({d:.1}% hit), {d:.2} GB read, cap {d} slots/layer, pinned {d}\n", .{ st.hits, st.misses, st.hitRate() * 100, @as(f64, @floatFromInt(st.bytes_read)) / 1e9, store.cap, store.pinned_experts });
        store.saveUsage() catch {};
    };
    file.deinit();
    try stdout.print("load: {d:.3} s\n", .{@as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - load_start)) / 1e9});

    if (vectors_dir) |dir_path| {
        return runVectors(init.io, allocator, &ctx, &model, &tokenizer, stdout, dir_path, vectors_max_prompt);
    }

    const eos = tokenizer.eosId();
    var tokens: std.ArrayList(usize) = .empty;
    defer tokens.deinit(allocator);
    if (chat) {
        try renderChat(allocator, &tokenizer, prompt_text, &tokens);
    } else {
        if (tokenizer.bosId()) |b| try tokens.append(allocator, b);
        const ids32 = try tokenizer.encode(allocator, prompt_text);
        defer allocator.free(ids32);
        for (ids32) |id| try tokens.append(allocator, id);
    }
    try stdout.print("prompt tokens: {d}\n", .{tokens.items.len});

    var session = try Session.init(&model, 8192);
    defer session.deinit(&model);

    const prefill_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    var logits: []f32 = &.{};
    defer if (logits.len > 0) allocator.free(logits);
    for (tokens.items) |token| {
        if (logits.len > 0) allocator.free(logits);
        logits = try llm.deepseek4.model.step(&model, &ctx, &session, token);
    }
    try stdout.print("prefill: {d:.1} ms ({d} sequential steps)\n", .{ @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - prefill_start)) / 1e6, tokens.items.len });

    var reply: std.ArrayList(u8) = .empty;
    defer reply.deinit(allocator);
    const decode_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    var produced: usize = 0;
    while (produced < gen_count) : (produced += 1) {
        const best = argmax(logits);
        if (eos != null and best == eos.?) break;
        try tokenizer.decodeAppend(allocator, @intCast(best), &reply);
        allocator.free(logits);
        logits = try llm.deepseek4.model.step(&model, &ctx, &session, best);
    }
    const decode_ns = std.Io.Clock.awake.now(init.io).nanoseconds - decode_start;
    try stdout.print("decode: {d} steps, {d:.1} ms, {d:.2} tok/s\n", .{ produced, @as(f64, @floatFromInt(decode_ns)) / 1e6, @as(f64, @floatFromInt(produced)) * 1e9 / @as(f64, @floatFromInt(decode_ns)) });
    try stdout.print("prompt: {s}\ntext:  {s}\n", .{ prompt_text, reply.items });
}

/// The reference chat rendering (thinking disabled): BOS, user marker,
/// prompt, assistant marker, closed think block.
fn renderChat(allocator: std.mem.Allocator, tokenizer: *const Tokenizer, prompt: []const u8, out: *std.ArrayList(usize)) !void {
    const bos = tokenizer.bosId() orelse return error.MissingChatTokens;
    const user_id = tokenizer.tokenId("<｜User｜>") orelse return error.MissingChatTokens;
    const assistant_id = tokenizer.tokenId("<｜Assistant｜>") orelse return error.MissingChatTokens;
    const think_end_id = tokenizer.tokenId("</think>") orelse return error.MissingChatTokens;
    try out.append(allocator, bos);
    try out.append(allocator, user_id);
    const ids32 = try tokenizer.encode(allocator, prompt);
    defer allocator.free(ids32);
    for (ids32) |id| try out.append(allocator, id);
    try out.append(allocator, assistant_id);
    try out.append(allocator, think_end_id);
}

fn argmax(logits: []const f32) usize {
    var best: usize = 0;
    var best_v: f32 = -std.math.inf(f32);
    for (logits, 0..) |v, i| {
        if (v > best_v) {
            best_v = v;
            best = i;
        }
    }
    return best;
}

/// One parsed `*.official.json` fixture: the exact user prompt, the API's
/// prompt token count, and the greedy continuation (one text per step).
const Vector = struct {
    id: []const u8,
    prompt: []const u8,
    prompt_tokens: usize,
    steps: [][]const u8,
};

fn parseVector(arena: std.mem.Allocator, bytes: []const u8) !Vector {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{});
    const root = parsed.object;
    const usage = root.get("usage") orelse return error.BadVector;
    const steps_v = root.get("steps") orelse return error.BadVector;
    const steps = try arena.alloc([]const u8, steps_v.array.items.len);
    for (steps_v.array.items, steps) |item, *out| {
        out.* = item.object.get("token").?.object.get("text").?.string;
    }
    return .{
        .id = (root.get("id") orelse return error.BadVector).string,
        .prompt = (root.get("prompt") orelse return error.BadVector).string,
        .prompt_tokens = @intCast(usage.object.get("prompt_tokens").?.integer),
        .steps = steps,
    };
}

/// Run every official fixture in `dir_path` with the reference chat rendering
/// and greedy decoding, and compare the continuation step-by-step against the
/// API's. Text is compared on concatenated bytes, so a different token
/// boundary with identical text still counts as a match. Fails when any run
/// vector diverges on the very first step (quantized weights legitimately
/// drift a few steps in; step 0 disagreeing means the forward is wrong).
fn runVectors(
    io: std.Io,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    model: *Model,
    tokenizer: *const Tokenizer,
    stdout: *std.Io.Writer,
    dir_path: []const u8,
    max_prompt: usize,
) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);

    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".official.json")) continue;
        try names.append(allocator, try allocator.dupe(u8, entry.name));
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    if (names.items.len == 0) return error.NoVectors;

    var failures: usize = 0;
    var ran: usize = 0;
    for (names.items) |name| {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var file = try dir.openFile(io, name, .{});
        defer file.close(io);
        const stat = try file.stat(io);
        const bytes = try arena.alloc(u8, @intCast(stat.size));
        var read_len: usize = 0;
        while (read_len < bytes.len) {
            const n = try file.readStreaming(io, &.{bytes[read_len..]});
            if (n == 0) return error.EndOfStream;
            read_len += n;
        }
        const vec = try parseVector(arena, bytes);

        if (vec.prompt_tokens > max_prompt) {
            try stdout.print("{s}: SKIP ({d} prompt tokens > --vectors-max-prompt={d})\n", .{ vec.id, vec.prompt_tokens, max_prompt });
            try stdout.flush();
            continue;
        }
        ran += 1;

        var tokens: std.ArrayList(usize) = .empty;
        defer tokens.deinit(allocator);
        try renderChat(allocator, tokenizer, vec.prompt, &tokens);

        var session = try Session.init(model, tokens.items.len + vec.steps.len + 8);
        defer session.deinit(model);

        var logits: []f32 = &.{};
        defer if (logits.len > 0) allocator.free(logits);
        for (tokens.items) |token| {
            if (logits.len > 0) allocator.free(logits);
            logits = try llm.deepseek4.model.step(model, ctx, &session, token);
        }

        var ours: std.ArrayList(u8) = .empty;
        defer ours.deinit(allocator);
        const eos = tokenizer.eosId();
        for (vec.steps) |_| {
            const best = argmax(logits);
            if (eos != null and best == eos.?) break;
            try tokenizer.decodeAppend(allocator, @intCast(best), &ours);
            allocator.free(logits);
            logits = try llm.deepseek4.model.step(model, ctx, &session, best);
        }

        // Longest official step-prefix our continuation reproduces.
        var official: std.ArrayList(u8) = .empty;
        defer official.deinit(allocator);
        var matched: usize = 0;
        for (vec.steps) |step_text| {
            try official.appendSlice(allocator, step_text);
            if (!std.mem.startsWith(u8, ours.items, official.items)) break;
            matched += 1;
        }

        const token_note = if (tokens.items.len == vec.prompt_tokens) "=" else "!";
        if (matched == 0) failures += 1;
        try stdout.print("{s}: {s} steps {d}/{d}, prompt tokens {d}{s}{d}\n  ours: {s}\n  api:  {s}\n", .{
            vec.id,
            if (matched > 0) "PASS" else "FAIL",
            matched,
            vec.steps.len,
            tokens.items.len,
            token_note,
            vec.prompt_tokens,
            ours.items,
            official.items,
        });
        try stdout.flush();
    }
    try stdout.print("vectors: {d} run, {d} failed\n", .{ ran, failures });
    if (failures > 0) return error.VectorMismatch;
}
