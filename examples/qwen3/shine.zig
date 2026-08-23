//! SHINE mode of the qwen3 runner (`--shine <shine.gguf> --shine-context
//! TEXT|@FILE`): compile the context into a rank-8 LoRA adapter with one
//! hypernetwork pass (arXiv 2602.06358), then answer questions with the
//! adapted model — zero context tokens, zero context KV at question time.
//! One-shot with `--chat "question"`, interactive with `--repl`.
//!
//! Prompting matches the reference inference setup: no system turn, and the
//! assistant turn opens with an empty <think> block (the SHINE checkpoint
//! was tuned no-think). Decoding is greedy like the reference harness.
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const util = @import("util.zig");

const shine = llm.research.shine;
const qwen3 = llm.qwen3.model;

pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const qwen3.Model,
    tokenizer: *const llm.tokenizer.Tokenizer,
    shine_path: ?[]const u8,
    context_arg: ?[]const u8,
    adapter_path: ?[]const u8,
    save_path: ?[]const u8,
    save_cartridge_path: ?[]const u8,
    chat_text: ?[]const u8,
    repl: bool,
    max_new_tokens: usize,
) !void {
    var adapter: shine.LoraSet = if (adapter_path) |path| blk: {
        // Reload a saved artifact: no SHINE weights, no hypernetwork pass.
        const load_start = util.nowNs(io);
        const set = try shine.loadLoraGguf(ctx, io, path, model.config);
        try stdout.print("shine: loaded adapter {s} ({d} layers, r={d}) in {d:.2}s\n", .{
            path,
            set.layers.len,
            set.layers[0].q.a.dim(.lora_r),
            @as(f64, @floatFromInt(util.nowNs(io) - load_start)) / 1e9,
        });
        break :blk set;
    } else blk: {
        const sh_path = shine_path orelse {
            try stdout.print("shine mode needs --shine <shine.gguf> (or --shine-adapter <adapter.gguf>)\n", .{});
            return error.MissingShinePath;
        };
        const ctx_arg = context_arg orelse {
            try stdout.print("--shine requires --shine-context TEXT|@FILE\n", .{});
            return error.MissingShineContext;
        };
        const load_start = util.nowNs(io);
        var sh = try shine.Shine.loadGguf(ctx, io, sh_path, model.config);
        defer sh.deinit();
        try stdout.print("shine: loaded {s} (M={d}, lora r={d}, metalora r={d}) in {d:.2}s\n", .{
            sh_path,
            sh.config.num_mem_token,
            sh.config.lora_r,
            sh.config.metalora_r,
            @as(f64, @floatFromInt(util.nowNs(io) - load_start)) / 1e9,
        });

        // Context text: literal, or @FILE.
        var context_owned: ?[]u8 = null;
        defer if (context_owned) |c| allocator.free(c);
        const context_text = try util.grammarValue(io, allocator, ctx_arg, &context_owned);

        // The reference tokenizes the raw context with no template and no
        // special-marker parsing (HumanCollator).
        const context_ids32 = try tokenizer.encodeRaw(allocator, context_text);
        defer allocator.free(context_ids32);
        const context_ids = try allocator.alloc(usize, context_ids32.len);
        defer allocator.free(context_ids);
        for (context_ids, context_ids32) |*d, s| d.* = s;

        // Cartridge readout: a cartridge-mode checkpoint compiles the
        // context into a STANDARD KV-prefix cartridge (safetensors state
        // dict) instead of a LoRA adapter — train, evaluate, and serve it
        // exactly like `zig build cartridge` output (--cartridge, fleets).
        if (sh.config.cartridge_rows != 0 or save_cartridge_path != null) {
            if (sh.config.cartridge_rows == 0) {
                try stdout.print("--shine-save-cartridge needs a cartridge-mode SHINE checkpoint (shine.cartridge_rows > 0); this one uses the LoRA readout\n", .{});
                return error.InvalidShineConfig;
            }
            const cart_path = save_cartridge_path orelse {
                try stdout.print("cartridge-mode SHINE checkpoint: pass --shine-save-cartridge PATH, then serve with --cartridge PATH\n", .{});
                return error.MissingArgument;
            };
            const cart_start = util.nowNs(io);
            var cart = try shine.generateCartridge(model, &sh, ctx, allocator, context_ids);
            defer cart.deinit();
            var out_file = try std.Io.Dir.cwd().createFile(io, cart_path, .{});
            defer out_file.close(io);
            var write_buffer: [1 << 20]u8 = undefined;
            var out_writer = out_file.writer(io, &write_buffer);
            try cart.saveState(&out_writer.interface);
            try out_writer.interface.flush();
            try stdout.print("shine: context ({d} tokens) -> cartridge in {d:.2}s ({d} rows/layer) saved to {s}\n", .{
                context_ids.len,
                @as(f64, @floatFromInt(util.nowNs(io) - cart_start)) / 1e9,
                cart.p,
                cart_path,
            });
            try stdout.print("serve it like any cartridge: zig build qwen3 -- <base.gguf> --cartridge {s} --chat \"...\"\n", .{cart_path});
            return;
        }

        const gen_start = util.nowNs(io);
        const set = try shine.generateAdapter(model, &sh, ctx, context_ids);
        try stdout.print("shine: context ({d} tokens) -> adapter in {d:.2}s\n", .{
            context_ids.len,
            @as(f64, @floatFromInt(util.nowNs(io) - gen_start)) / 1e9,
        });
        break :blk set;
    };
    defer adapter.deinit();
    try stdout.flush();

    if (save_path) |path| {
        var out_file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer out_file.close(io);
        var write_buffer: [1 << 20]u8 = undefined;
        var out_writer = out_file.writer(io, &write_buffer);
        try shine.saveLora(&adapter, model.config, allocator, &out_writer.interface);
        try out_writer.interface.flush();
        try stdout.print("shine: adapter saved to {s}\n", .{path});
        try stdout.flush();
    }

    const stop = tokenizer.tokenId("<|im_end|>") orelse return error.TokenizerUnavailable;
    // The reference head is resized to len(tokenizer); everything past the
    // tokenizer's vocabulary is a padding row — never sample it.
    const vocab_limit = tokenizer.vocabSize();

    var history: std.ArrayList(u8) = .empty;
    defer history.deinit(allocator);

    if (chat_text) |question| {
        _ = try answer(io, allocator, stdout, ctx, model, &adapter, tokenizer, &history, question, stop, vocab_limit, max_new_tokens);
        try stdout.writeAll("\n");
        return;
    }
    if (!repl) return;

    var stdin_buf: [8192]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);
    const in = &stdin_reader.interface;
    try stdout.writeAll("shine chat — the context lives in the adapter; type a question, empty line or Ctrl-D to quit:\n");
    while (true) {
        try stdout.writeAll("\n> ");
        try stdout.flush();
        const line = (try in.takeDelimiter('\n')) orelse break;
        const msg = std.mem.trim(u8, line, " \t\r");
        if (msg.len == 0) break;
        try stdout.writeAll("\n");
        _ = try answer(io, allocator, stdout, ctx, model, &adapter, tokenizer, &history, msg, stop, vocab_limit, max_new_tokens);
        try stdout.writeAll("\n");
    }
}

/// `--shine-fleet-build OUT --shine-docs DIR`: compile every .txt/.md file
/// under DIR into a saved adapter plus retrieval embeddings, and write the
/// fleet lmserve's `--shine-fleet` serves: `doc-XXX.adapter.gguf` per
/// document, `index.safetensors` (the cartridge fleets' EmbedIndex, same
/// `embed_suffix` recipe, chunked at 256 tokens), and `shine-fleet.json`.
pub fn buildFleet(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const qwen3.Model,
    tokenizer: *const llm.tokenizer.Tokenizer,
    shine_path: []const u8,
    docs_dir: []const u8,
    out_dir: []const u8,
) !void {
    var sh = try shine.Shine.loadGguf(ctx, io, shine_path, model.config);
    defer sh.deinit();
    var trainer = try llm.qwen3.train.Trainer(.{ .q = false, .v = false }).init(ctx, model, .{ .rank = 1, .alpha = 1 }, 0);
    defer trainer.deinit();
    try std.Io.Dir.cwd().createDirPath(io, out_dir);

    // Stable doc ids: directory entries sorted by name.
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |name| allocator.free(name);
        names.deinit(allocator);
    }
    {
        var dir = try std.Io.Dir.cwd().openDir(io, docs_dir, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.ascii.endsWithIgnoreCase(entry.name, ".txt") and !std.ascii.endsWithIgnoreCase(entry.name, ".md")) continue;
            try names.append(allocator, try allocator.dupe(u8, entry.name));
        }
    }
    if (names.items.len == 0) return error.EmptyCorpus;
    std.mem.sort([]u8, names.items, {}, struct {
        fn lessThan(_: void, x: []u8, y: []u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lessThan);

    const embed_chunk: usize = 256;
    var index = llm.cartridge_fleet.EmbedIndex.init(allocator, model.config.hidden_size);
    defer index.deinit();
    const suffix_ids32 = try tokenizer.encode(allocator, llm.cartridge_fleet.embed_suffix);
    defer allocator.free(suffix_ids32);
    const vec = try allocator.alloc(f32, model.config.hidden_size);
    defer allocator.free(vec);

    const ManifestDoc = struct { name: []const u8, adapter_file: []const u8 };
    var manifest_docs: std.ArrayList(ManifestDoc) = .empty;
    defer {
        for (manifest_docs.items) |doc| allocator.free(doc.adapter_file);
        manifest_docs.deinit(allocator);
    }

    for (names.items, 0..) |name, doc_i| {
        const doc_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ docs_dir, name });
        defer allocator.free(doc_path);
        const text = try util.readTextFile(io, allocator, doc_path);
        defer allocator.free(text);
        const ids32 = try tokenizer.encodeRaw(allocator, text);
        defer allocator.free(ids32);
        const ids = try allocator.alloc(usize, ids32.len);
        defer allocator.free(ids);
        for (ids, ids32) |*dst, id| dst.* = id;

        const gen_start = util.nowNs(io);
        var adapter = try shine.generateAdapter(model, &sh, ctx, ids);
        defer adapter.deinit();

        const adapter_file = try std.fmt.allocPrint(allocator, "doc-{d:0>3}.adapter.gguf", .{doc_i});
        errdefer allocator.free(adapter_file);
        {
            const out_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ out_dir, adapter_file });
            defer allocator.free(out_path);
            var out_file = try std.Io.Dir.cwd().createFile(io, out_path, .{});
            defer out_file.close(io);
            var write_buffer: [1 << 20]u8 = undefined;
            var out_writer = out_file.writer(io, &write_buffer);
            try shine.saveLora(&adapter, model.config, allocator, &out_writer.interface);
            try out_writer.interface.flush();
        }

        var chunks: usize = 0;
        var at: usize = 0;
        while (at < ids.len) : (at += embed_chunk) {
            const end = @min(at + embed_chunk, ids.len);
            const full = try allocator.alloc(usize, (end - at) + suffix_ids32.len);
            defer allocator.free(full);
            for (full[0 .. end - at], ids[at..end]) |*dst, id| dst.* = id;
            for (full[end - at ..], suffix_ids32) |*dst, id| dst.* = id;
            try trainer.embedLastHidden(ctx, full, vec);
            try index.append(@intCast(doc_i), vec);
            chunks += 1;
        }
        try manifest_docs.append(allocator, .{ .name = name, .adapter_file = adapter_file });
        try stdout.print("shine fleet: [{d}/{d}] {s}: {d} tokens -> {s} ({d} chunks) in {d:.2}s\n", .{
            doc_i + 1,
            names.items.len,
            name,
            ids.len,
            adapter_file,
            chunks,
            @as(f64, @floatFromInt(util.nowNs(io) - gen_start)) / 1e9,
        });
        try stdout.flush();
    }

    try index.finalize();
    {
        const index_path = try std.fmt.allocPrint(allocator, "{s}/index.safetensors", .{out_dir});
        defer allocator.free(index_path);
        var out_file = try std.Io.Dir.cwd().createFile(io, index_path, .{});
        defer out_file.close(io);
        var write_buffer: [1 << 20]u8 = undefined;
        var out_writer = out_file.writer(io, &write_buffer);
        try index.serialize(allocator, &out_writer.interface);
        try out_writer.interface.flush();
    }
    {
        const manifest = .{ .embed_chunk = embed_chunk, .docs = manifest_docs.items };
        const json = try std.json.Stringify.valueAlloc(allocator, manifest, .{ .whitespace = .indent_2 });
        defer allocator.free(json);
        const manifest_path = try std.fmt.allocPrint(allocator, "{s}/shine-fleet.json", .{out_dir});
        defer allocator.free(manifest_path);
        var out_file = try std.Io.Dir.cwd().createFile(io, manifest_path, .{});
        defer out_file.close(io);
        var out_writer = out_file.writer(io, &.{});
        try out_writer.interface.writeAll(json);
        try out_writer.interface.flush();
    }
    try stdout.print("shine fleet: wrote {d} adapters + index ({d} chunks) + shine-fleet.json to {s}\n", .{ names.items.len, index.len(), out_dir });
    try stdout.flush();
}

/// One greedy turn: extend the running transcript with the question and the
/// no-think assistant opener, prefill + decode with the adapter, stream the
/// reply, and append it to the transcript for the next turn.
fn answer(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const qwen3.Model,
    adapter: *const shine.LoraSet,
    tokenizer: *const llm.tokenizer.Tokenizer,
    history: *std.ArrayList(u8),
    question: []const u8,
    stop: u32,
    vocab_limit: usize,
    max_new_tokens: usize,
) !usize {
    {
        const user_turn = try std.fmt.allocPrint(allocator, "<|im_start|>user\n{s}<|im_end|>\n", .{question});
        defer allocator.free(user_turn);
        try history.appendSlice(allocator, user_turn);
    }

    var prompt_text: std.ArrayList(u8) = .empty;
    defer prompt_text.deinit(allocator);
    try prompt_text.appendSlice(allocator, history.items);
    try prompt_text.appendSlice(allocator, "<|im_start|>assistant\n<think>\n\n</think>\n\n");

    const prompt_ids32 = try tokenizer.encode(allocator, prompt_text.items);
    defer allocator.free(prompt_ids32);
    const prompt_ids = try allocator.alloc(usize, prompt_ids32.len);
    defer allocator.free(prompt_ids);
    for (prompt_ids, prompt_ids32) |*d, s| d.* = s;

    var kv = try model.initKvCache(ctx, prompt_ids.len + max_new_tokens + 1);
    defer kv.deinit();

    const decode_start = util.nowNs(io);
    const out_tokens = try allocator.alloc(usize, max_new_tokens);
    defer allocator.free(out_tokens);
    const produced = try shine.greedy(model, adapter, ctx, &kv, prompt_ids, out_tokens, .{
        .max_new_tokens = max_new_tokens,
        .stop_token = stop,
        .vocab_limit = vocab_limit,
    });

    var reply32: std.ArrayList(u32) = .empty;
    defer reply32.deinit(allocator);
    for (out_tokens[0..produced]) |id| {
        if (id == stop) break;
        try reply32.append(allocator, @intCast(id));
    }
    const text = try tokenizer.decode(allocator, reply32.items);
    defer allocator.free(text);
    try stdout.writeAll(text);
    const elapsed_s = @as(f64, @floatFromInt(util.nowNs(io) - decode_start)) / 1e9;
    try stdout.print("\n[{d} tokens in {d:.2}s, {d:.2} tok/s]\n", .{ produced, elapsed_s, @as(f64, @floatFromInt(produced)) / elapsed_s });
    try stdout.flush();

    {
        const assistant_turn = try std.fmt.allocPrint(allocator, "<|im_start|>assistant\n{s}<|im_end|>\n", .{text});
        defer allocator.free(assistant_turn);
        try history.appendSlice(allocator, assistant_turn);
    }
    return produced;
}
