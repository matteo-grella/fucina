//! Talker-stack parity vs the qwentts.cpp oracle's greedy dump
//! (src/llm/qwen3tts/goldens/dump-greedy: CustomVoice Aiden, "The quick brown fox
//! jumps over the lazy dog.", --greedy --no-fa, natural EOS at 37 frames).
//! Gates: prompt ids exact, prompt embeds + prefill logits cosine, generated
//! codes token-for-token. Skips without the talker GGUF + dumps.
//!
//! Native backend only: these are real-model golden forwards — they pin
//! MODEL WIRING, not kernel math, so the scalar reference leg skips them
//! (its kernel coverage lives in the exec/backend suites).

const std = @import("std");
const fucina = @import("fucina");
const model_mod = @import("model.zig");
const prompt_mod = @import("prompt.zig");
const pipeline = @import("pipeline.zig");
const tokenizer_mod = @import("../tokenizer.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

const talker_path = "models/qwen3-tts/qwen-talker-0.6b-customvoice-F32.gguf";
const dump_dir = "src/llm/qwen3tts/goldens/dump-greedy";
const test_text = "The quick brown fox jumps over the lazy dog.";

fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(1 << 30));
}

const Dump = struct {
    dims: [4]usize,
    ndims: usize,
    data: []f32,

    fn deinit(self: *Dump, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};

fn loadDump(allocator: std.mem.Allocator, comptime name: []const u8) !Dump {
    const bytes = try readFile(allocator, dump_dir ++ "/" ++ name ++ ".bin");
    defer allocator.free(bytes);
    const ndims: usize = @intCast(std.mem.readInt(i32, bytes[0..4], .little));
    var dims = [4]usize{ 1, 1, 1, 1 };
    var n: usize = 1;
    for (0..ndims) |i| {
        dims[i] = @intCast(std.mem.readInt(i32, bytes[4 + i * 4 ..][0..4], .little));
        n *= dims[i];
    }
    const off = 4 + ndims * 4;
    const data = try allocator.alloc(f32, n);
    for (data, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[off + i * 4 ..][0..4], .little));
    return .{ .dims = dims, .ndims = ndims, .data = data };
}

fn cosine(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * @as(f64, y);
        na += @as(f64, x) * @as(f64, x);
        nb += @as(f64, y) * @as(f64, y);
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}

test "qwen3tts talker: greedy generation matches the oracle token-for-token" {
    if (comptime @import("fucina").internal.backend_mod.active_kind != .native) return error.SkipZigTest; // real-model goldens: native only
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = gguf.File.loadMmap(allocator, std.testing.io, talker_path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();
    var ids_dump = loadDump(allocator, "prompt-ids") catch return error.SkipZigTest;
    defer ids_dump.deinit(allocator);
    var embed_dump = try loadDump(allocator, "talker-input-embed");
    defer embed_dump.deinit(allocator);
    var logits_dump = try loadDump(allocator, "talker-logits-prefill");
    defer logits_dump.deinit(allocator);
    var codes_dump = try loadDump(allocator, "codes-full");
    defer codes_dump.deinit(allocator);
    var next_emb_dump = try loadDump(allocator, "next-emb-step0");
    defer next_emb_dump.deinit(allocator);
    var pad_dump = try loadDump(allocator, "tts-pad-embed");
    defer pad_dump.deinit(allocator);

    var model = try model_mod.Model.load(&ctx, &file);
    defer model.deinit();
    var tok = try tokenizer_mod.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tok.deinit();

    var prompt = try prompt_mod.build(allocator, &model, &tok, .{
        .text = test_text,
        .language = "english",
        .speaker = "aiden",
    });
    defer prompt.deinit();

    // Prompt ids exact (oracle stores them as f32).
    try std.testing.expectEqual(ids_dump.data.len, prompt.prompt_ids.len);
    for (ids_dump.data, prompt.prompt_ids) |want, got| {
        try std.testing.expectEqual(@as(u32, @intFromFloat(want)), got);
    }

    // Prompt embedding rows.
    try std.testing.expectEqual(embed_dump.data.len, prompt.input_embed.len);
    const embed_cos = cosine(embed_dump.data, prompt.input_embed);
    if (embed_cos < 0.999999) {
        std.debug.print("[talker] input-embed cosine {d:.8}\n", .{embed_cos});
        return error.TestUnexpectedResult;
    }
    const pad_cos = cosine(pad_dump.data, prompt.tts_pad_embed);
    try std.testing.expect(pad_cos > 0.999999);

    // Greedy generation with taps.
    var taps = pipeline.Taps{ .allocator = allocator };
    defer taps.deinit();
    var kvs = try pipeline.Kvs.init(allocator, &model);
    defer kvs.deinit();
    var result = try pipeline.generate(&ctx, &model, &prompt, .{ .greedy = true, .max_new_tokens = 256 }, &kvs, null, null, &taps);
    defer result.deinit();

    const logits_cos = cosine(logits_dump.data, taps.logits_prefill.?);
    if (logits_cos < 0.99999) {
        std.debug.print("[talker] prefill-logits cosine {d:.8}\n", .{logits_cos});
        return error.TestUnexpectedResult;
    }

    // Token-for-token equality with the oracle's [37, 16] frame matrix.
    const want_frames = codes_dump.dims[0];
    const want_groups = codes_dump.dims[1];
    try std.testing.expectEqual(@as(usize, 16), want_groups);
    if (result.frames != want_frames) {
        std.debug.print("[talker] frames {d} != oracle {d}\n", .{ result.frames, want_frames });
        return error.TestUnexpectedResult;
    }
    var mismatches: usize = 0;
    for (codes_dump.data, result.codes) |want, got| {
        if (@as(i32, @intFromFloat(want)) != got) mismatches += 1;
    }
    if (mismatches != 0) {
        std.debug.print("[talker] {d}/{d} code mismatches; next-emb-step0 cosine {d:.8}\n", .{
            mismatches, result.codes.len, cosine(next_emb_dump.data, taps.next_emb_step0.?),
        });
        return error.TestUnexpectedResult;
    }
}

test "qwen3tts talker: seeded sampling replays the oracle draw-for-draw" {
    if (comptime @import("fucina").internal.backend_mod.active_kind != .native) return error.SkipZigTest; // real-model goldens: native only
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = gguf.File.loadMmap(allocator, std.testing.io, talker_path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();
    const bytes = readFile(allocator, "src/llm/qwen3tts/goldens/dump-seed42/codes-full.bin") catch return error.SkipZigTest;
    allocator.free(bytes);
    var codes_dump = blk: {
        const b = try readFile(allocator, "src/llm/qwen3tts/goldens/dump-seed42/codes-full.bin");
        defer allocator.free(b);
        const ndims: usize = @intCast(std.mem.readInt(i32, b[0..4], .little));
        var n: usize = 1;
        var dims = [4]usize{ 1, 1, 1, 1 };
        for (0..ndims) |i| {
            dims[i] = @intCast(std.mem.readInt(i32, b[4 + i * 4 ..][0..4], .little));
            n *= dims[i];
        }
        const data = try allocator.alloc(f32, n);
        for (data, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, b[4 + ndims * 4 + i * 4 ..][0..4], .little));
        break :blk Dump{ .dims = dims, .ndims = ndims, .data = data };
    };
    defer codes_dump.deinit(allocator);

    var model = try model_mod.Model.load(&ctx, &file);
    defer model.deinit();
    var tok = try tokenizer_mod.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tok.deinit();
    var prompt = try prompt_mod.build(allocator, &model, &tok, .{
        .text = test_text,
        .language = "english",
        .speaker = "aiden",
    });
    defer prompt.deinit();

    var kvs = try pipeline.Kvs.init(allocator, &model);
    defer kvs.deinit();
    var result = try pipeline.generate(&ctx, &model, &prompt, .{ .seed = 42, .max_new_tokens = 256 }, &kvs, null, null, null);
    defer result.deinit();

    try std.testing.expectEqual(codes_dump.dims[0] * codes_dump.dims[1], result.codes.len);
    var mismatches: usize = 0;
    for (codes_dump.data, result.codes) |want, got| {
        if (@as(i32, @intFromFloat(want)) != got) mismatches += 1;
    }
    if (mismatches != 0) {
        std.debug.print("[talker seeded] {d}/{d} mismatches\n", .{ mismatches, result.codes.len });
        return error.TestUnexpectedResult;
    }
}
