//! DeepSeek V4 Flash runner: greedy completion over the CSA/HCA trunk with
//! streamed experts. Validation against the official API vectors comes via
//! --vectors (chat-rendered prompts + greedy token comparison).
//!   zig build deepseek4 -- gguf/model.gguf --prompt "..." --gen 32 \
//!     --moe-stream --moe-cache-mb=20480
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print("usage: zig build deepseek4 -- <model.gguf> --prompt \"...\" [--gen N] [--moe-stream --moe-cache-mb=N]\n", .{});
        return;
    }

    var prompt_text: []const u8 = "The capital of France is";
    var gen_count: usize = 16;
    var moe_stream_flag = false;
    var moe_cache_mb: ?usize = null;
    var chat = false;
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
    var tokenizer = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tokenizer.deinit();

    const load_options: llm.deepseek4.model.Model.LoadOptions = if (moe_stream_flag) .{
        .moe_stream = .{
            .gguf_path = args[1],
            .cache_bytes = if (moe_cache_mb) |mb| mb << 20 else null,
        },
    } else .{};
    var model = try llm.deepseek4.model.Model.loadGgufFromFileOptions(&ctx, &file, load_options);
    defer model.deinit();
    defer if (model.expert_store) |store| {
        const st = store.stats;
        std.debug.print("moe stream: hits {d} / misses {d} ({d:.1}% hit), {d:.2} GB read, cap {d} slots/layer, pinned {d}\n", .{ st.hits, st.misses, st.hitRate() * 100, @as(f64, @floatFromInt(st.bytes_read)) / 1e9, store.cap, store.pinned_experts });
        store.saveUsage() catch {};
    };
    const bos: ?u32 = tokenizer.bosId();
    const eos = tokenizer.eosId();
    file.deinit();
    try stdout.print("load: {d:.3} s\n", .{@as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - load_start)) / 1e9});

    const ids32 = try tokenizer.encode(allocator, prompt_text);
    defer allocator.free(ids32);
    var tokens: std.ArrayList(usize) = .empty;
    defer tokens.deinit(allocator);
    if (bos) |b| try tokens.append(allocator, b);
    if (chat) {
        // The reference chat rendering (thinking disabled): BOS, user marker,
        // prompt, assistant marker, closed think block.
        const user_id = tokenizer.tokenId("<｜User｜>") orelse return error.MissingChatTokens;
        const assistant_id = tokenizer.tokenId("<｜Assistant｜>") orelse return error.MissingChatTokens;
        const think_end_id = tokenizer.tokenId("</think>") orelse return error.MissingChatTokens;
        try tokens.append(allocator, user_id);
        for (ids32) |id| try tokens.append(allocator, id);
        try tokens.append(allocator, assistant_id);
        try tokens.append(allocator, think_end_id);
    } else {
        for (ids32) |id| try tokens.append(allocator, id);
    }
    try stdout.print("prompt tokens: {d}\n", .{tokens.items.len});

    var session = try llm.deepseek4.model.Session.init(&model, 8192);
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
        var best: usize = 0;
        var best_v: f32 = -std.math.inf(f32);
        for (logits, 0..) |v, i| {
            if (v > best_v) {
                best_v = v;
                best = i;
            }
        }
        if (eos != null and best == eos.?) break;
        try tokenizer.decodeAppend(allocator, @intCast(best), &reply);
        allocator.free(logits);
        logits = try llm.deepseek4.model.step(&model, &ctx, &session, best);
    }
    const decode_ns = std.Io.Clock.awake.now(init.io).nanoseconds - decode_start;
    try stdout.print("decode: {d} steps, {d:.1} ms, {d:.2} tok/s\n", .{ produced, @as(f64, @floatFromInt(decode_ns)) / 1e6, @as(f64, @floatFromInt(produced)) * 1e9 / @as(f64, @floatFromInt(decode_ns)) });
    try stdout.print("prompt: {s}\ntext:  {s}\n", .{ prompt_text, reply.items });
}
