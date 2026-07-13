//! DeepSeek-V2 family runner (milestone A): raw greedy completion over the
//! MLA + MoE forward, sequential single-token steps. Usage:
//!   zig build deepseek2 -- models/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf \
//!     --prompt "..." --gen 64
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
        try stdout.print("usage: zig build deepseek2 -- <model.gguf> --prompt \"...\" [--gen N]\n", .{});
        return;
    }

    var prompt_text: []const u8 = "The capital of France is";
    var gen_count: usize = 32;
    var moe_stream_flag = false;
    var moe_cache_mb: ?usize = null;
    var moe_pilot = false;
    var mla_mode: llm.deepseek2.model.Cache.Mode = .latent;
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
        } else if (std.mem.eql(u8, arg, "--moe-stream")) {
            moe_stream_flag = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-cache-mb=")) {
            moe_stream_flag = true;
            moe_cache_mb = try std.fmt.parseInt(usize, arg["--moe-cache-mb=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--moe-pilot")) {
            moe_stream_flag = true;
            moe_pilot = true;
        } else if (std.mem.eql(u8, arg, "--mla=full")) {
            mla_mode = .full;
        } else if (std.mem.eql(u8, arg, "--mla=latent")) {
            mla_mode = .latent;
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

    const capacity: usize = 2048;
    const load_options: llm.deepseek2.model.Model.LoadOptions = if (moe_stream_flag) .{
        .moe_stream = .{
            .gguf_path = args[1],
            .cache_bytes = if (moe_cache_mb) |mb| mb << 20 else null,
            .pilot = moe_pilot,
        },
    } else .{};
    var model = try llm.deepseek2.model.Model.loadGgufFromFileOptions(&ctx, &file, load_options);
    defer model.deinit();
    // The stats go through the SAME buffered stdout writer as everything
    // else: stdout's positional writes and stderr's offset-advancing writes
    // cannot safely share one redirected file (`cmd > f 2>&1` interleaves
    // destructively), so a std.debug stats line would get overwritten.
    defer if (model.expert_store) |store| {
        const st = store.stats;
        stdout.print("moe stream: hits {d} / misses {d} ({d:.1}% hit), {d:.2} GB read, cap {d} slots/layer, pinned {d}\n", .{ st.hits, st.misses, st.hitRate() * 100, @as(f64, @floatFromInt(st.bytes_read)) / 1e9, store.cap, store.pinned_experts }) catch {};
        if (st.pilot_recall_total > 0) stdout.print("moe pilot: recall {d:.1}% ({d}/{d} routed experts predicted), {d} ranges hinted\n", .{ st.pilotRecall() * 100, st.pilot_recall_hits, st.pilot_recall_total, st.pilot_ranges }) catch {};
        // Persist the routing histogram so the next startup auto-pins the
        // hot experts (the learning cache; every other runner does this).
        store.saveUsage() catch {};
    };
    const bos: ?u32 = tokenizer.bosId();
    file.deinit();
    try stdout.print("load: {d:.3} s\n", .{@as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - load_start)) / 1e9});

    // Encode: deepseek2 adds BOS.
    const ids32 = try tokenizer.encode(allocator, prompt_text);
    defer allocator.free(ids32);
    var tokens: std.ArrayList(usize) = .empty;
    defer tokens.deinit(allocator);
    // GLM checkpoints (glm-dsa) open with [gMASK]<sop> instead of BOS —
    // resolved by string so vocab ids stay model-defined; without them GLM
    // trunks degenerate (the glm4moe lesson).
    if (tokenizer.tokenId("[gMASK]")) |gmask| {
        try tokens.append(allocator, gmask);
        if (tokenizer.tokenId("<sop>")) |sop| try tokens.append(allocator, sop);
    } else if (bos) |b| try tokens.append(allocator, b);
    for (ids32) |id| try tokens.append(allocator, id);
    try stdout.print("prompt tokens: {d}\n", .{tokens.items.len});

    var cache = try model.initCacheMode(capacity, mla_mode);
    defer cache.deinit();

    // Prefill: sequential steps (milestone A keeps one uniform S=1 path).
    const prefill_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    var logits: []f32 = &.{};
    defer if (logits.len > 0) allocator.free(logits);
    for (tokens.items) |token| {
        if (logits.len > 0) allocator.free(logits);
        logits = try model.step(&ctx, &cache, token);
    }
    try stdout.print("prefill: {d:.1} ms ({d} sequential steps)\n", .{ @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - prefill_start)) / 1e6, tokens.items.len });

    // Greedy decode.
    const eos = tokenizer.eosId();
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
        try tokens.append(allocator, best);
        allocator.free(logits);
        logits = try model.step(&ctx, &cache, best);
    }
    const decode_ns = std.Io.Clock.awake.now(init.io).nanoseconds - decode_start;
    try stdout.print("decode: {d} steps, {d:.1} ms, {d:.2} tok/s\n", .{ produced, @as(f64, @floatFromInt(decode_ns)) / 1e6, @as(f64, @floatFromInt(produced)) * 1e9 / @as(f64, @floatFromInt(decode_ns)) });
    try stdout.print("prompt: {s}\ntext:  {s}\n", .{ prompt_text, reply.items });
}
