//! Pocket TTS v2 from GGUF: text (stdin) → 24 kHz mono WAV.
//!
//!   zig build pockettts -- --model models/pocket-tts/pocket-tts-english-v2.gguf \
//!       [--voice alba] [--temp 0.3] [--seed N] [--eos-threshold -4]
//!       [--max-frames N] -o out.wav < text.txt
//!
//! Continuous-latent streaming: each AR step yields one 80 ms frame (1920
//! samples) — first audio after a single step. Long texts split at sentence
//! boundaries into ≤50-token chunks, each generated against the same voice
//! prefix (the reference's chunking discipline).

const std = @import("std");
const builtin = @import("builtin");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const pocket = llm.pockettts.pocket;

const usage =
    \\usage: zig build pockettts -- --model <pocket.gguf> [--voice NAME]
    \\         [--temp F] [--seed N] [--eos-threshold F] [--max-frames N] -o out.wav
    \\
;

fn flagVal(args: []const [:0]const u8, name: []const u8) ?[]const u8 {
    for (args, 0..) |a, i| if (std.mem.eql(u8, a, name) and i + 1 < args.len) return args[i + 1];
    return null;
}

fn writeWav16(io: std.Io, path: []const u8, samples: []const f32) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var w = file.writer(io, &buf);
    const data_len: u32 = @intCast(samples.len * 2);
    try w.interface.writeAll("RIFF");
    try w.interface.writeInt(u32, 36 + data_len, .little);
    try w.interface.writeAll("WAVEfmt ");
    try w.interface.writeInt(u32, 16, .little);
    try w.interface.writeInt(u16, 1, .little);
    try w.interface.writeInt(u16, 1, .little);
    try w.interface.writeInt(u32, pocket.sample_rate, .little);
    try w.interface.writeInt(u32, pocket.sample_rate * 2, .little);
    try w.interface.writeInt(u16, 2, .little);
    try w.interface.writeInt(u16, 16, .little);
    try w.interface.writeAll("data");
    try w.interface.writeInt(u32, data_len, .little);
    for (samples) |s| {
        const v = std.math.clamp(s, -1.0, 1.0);
        try w.interface.writeInt(i16, @intFromFloat(v * 32767.0), .little);
    }
    try w.interface.flush();
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(allocator);

    var err_buf: [4096]u8 = undefined;
    var err_writer = std.Io.File.stderr().writer(io, &err_buf);
    const stderr = &err_writer.interface;
    defer stderr.flush() catch {};

    const model_path = flagVal(args, "--model") orelse return stderr.print("{s}", .{usage});
    const voice = flagVal(args, "--voice") orelse "alba";
    const out_path = flagVal(args, "-o") orelse "pocket-out.wav";
    const seed: u64 = if (flagVal(args, "--seed")) |s| try std.fmt.parseInt(u64, s, 10) else 42;
    const max_frames: usize = if (flagVal(args, "--max-frames")) |s| try std.fmt.parseInt(usize, s, 10) else 1000;
    const eos_threshold: f32 = if (flagVal(args, "--eos-threshold")) |s| try std.fmt.parseFloat(f32, s) else pocket.default_eos_threshold;

    var file = try fucina.gguf.File.loadMmap(allocator, io, model_path);
    defer file.deinit();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try pocket.Model.load(&ctx, &file);
    defer model.deinit();
    const temp: f32 = if (flagVal(args, "--temp")) |s|
        try std.fmt.parseFloat(f32, s)
    else
        @floatCast(file.getFloat("pocket.default_temperature") orelse 0.3);

    // stdin text
    var in_buf: [1 << 16]u8 = undefined;
    var reader = std.Io.File.stdin().reader(io, &in_buf);
    var text: std.ArrayList(u8) = .empty;
    while (true) {
        const chunk = reader.interface.peekGreedy(1) catch break;
        if (chunk.len == 0) break;
        try text.appendSlice(allocator, chunk);
        reader.interface.toss(chunk.len);
    }
    const trimmed = std.mem.trim(u8, text.items, " \t\r\n");
    if (trimmed.len == 0) return stderr.print("no text on stdin\n{s}", .{usage});

    var kv = try pocket.Kv.init(allocator, model.layers, 4096, model.heads, model.hd);
    defer kv.deinit();
    try pocket.loadVoice(&model, &file, &kv, voice);
    const voice_len = kv.offset;

    var fl = try pocket.FlowLm.init(allocator, &model, 128);
    defer fl.deinit();
    var mimi = try pocket.Mimi.init(allocator, &model, max_frames + 8);
    defer mimi.deinit();

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    var audio: std.ArrayList(f32) = .empty;
    var pcm: [pocket.frame_samples]f32 = undefined;
    const cond = try allocator.alloc(f32, model.d);
    const row = try allocator.alloc(f32, model.d);
    var x0: [pocket.latent_dim]f32 = undefined;
    var x1: [pocket.latent_dim]f32 = undefined;

    const t0 = std.Io.Clock.awake.now(io).nanoseconds;
    var frames_total: usize = 0;

    // sentence chunks ≤50 tokens against the same voice prefix
    var chunk_start: usize = 0;
    while (chunk_start < trimmed.len) {
        var chunk_end = trimmed.len;
        {
            // grow chunk sentence-by-sentence while ≤50 tokens
            var last_ok: usize = chunk_start;
            var probe = chunk_start;
            while (probe < trimmed.len) {
                probe += 1;
                if (probe == trimmed.len or trimmed[probe - 1] == '.' or trimmed[probe - 1] == '!' or trimmed[probe - 1] == '?') {
                    const ids_probe = try model.tokenize(allocator, trimmed[chunk_start..probe]);
                    const fits = ids_probe.len <= 50;
                    allocator.free(ids_probe);
                    if (fits) {
                        last_ok = probe;
                        if (probe == trimmed.len) break;
                    } else break;
                }
            }
            chunk_end = if (last_ok > chunk_start) last_ok else @min(trimmed.len, chunk_start + 160);
        }
        const chunk_text = std.mem.trim(u8, trimmed[chunk_start..chunk_end], " ");
        chunk_start = chunk_end;
        if (chunk_text.len == 0) continue;

        kv.offset = voice_len; // rewind to the voice prefix per chunk
        const ids = try model.tokenize(allocator, chunk_text);
        defer allocator.free(ids);
        const rows = try allocator.alloc(f32, ids.len * model.d);
        defer allocator.free(rows);
        for (ids, 0..) |id, i| @memcpy(rows[i * model.d ..][0..model.d], model.embed[id * model.d ..][0..model.d]);
        try fl.forward(&kv, rows, ids.len, cond);

        // bos row
        for (row, 0..) |*v, i| {
            var acc: f32 = 0;
            for (0..pocket.latent_dim) |j| acc += model.input_linear[i * pocket.latent_dim + j] * model.bos_emb[j];
            v.* = acc;
        }
        var eos_at: ?usize = null;
        var frame: usize = 0;
        while (frame < max_frames) : (frame += 1) {
            try fl.forward(&kv, row, 1, cond);
            var eos_logit: f32 = model.out_eos_b[0];
            eos_logit += blk: {
                var acc: f32 = 0;
                for (cond, model.out_eos_w) |c, w| acc += c * w;
                break :blk acc;
            };
            if (eos_logit > eos_threshold and eos_at == null) eos_at = frame;
            if (eos_at) |e| if (frame >= e + 2) break;
            const std_dev = @sqrt(temp);
            for (&x0) |*v| v.* = rng.floatNorm(f32) * std_dev;
            try pocket.lsdStep(&model, allocator, cond, &x0, &x1);
            try mimi.decodeFrame(&x1, &pcm);
            try audio.appendSlice(allocator, &pcm);
            frames_total += 1;
            for (row, 0..) |*v, i| {
                var acc: f32 = 0;
                for (0..pocket.latent_dim) |j| acc += model.input_linear[i * pocket.latent_dim + j] * x1[j];
                v.* = acc;
            }
        }
    }
    const t1 = std.Io.Clock.awake.now(io).nanoseconds;

    const gen_s = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const audio_s = @as(f64, @floatFromInt(audio.items.len)) / @as(f64, pocket.sample_rate);
    try stderr.print("[timing:{s}] {d} frames = {d:.2} s audio in {d:.2} s | RTF {d:.3} | {d:.1} ms/frame\n", .{
        @tagName(builtin.mode),
        frames_total,
        audio_s,
        gen_s,
        gen_s / audio_s,
        gen_s * 1000.0 / @as(f64, @floatFromInt(@max(1, frames_total))),
    });
    try writeWav16(io, out_path, audio.items);
    try stderr.print("[out] {s}: {d} samples @ {d} Hz\n", .{ out_path, audio.items.len, pocket.sample_rate });
}
