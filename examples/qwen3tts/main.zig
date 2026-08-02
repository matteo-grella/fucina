//! Qwen3-TTS from GGUF: text (stdin) → 24 kHz mono WAV, CustomVoice speakers.
//! The Zig port of qwentts.cpp's `qwen-tts` tool over `fucina.llm.qwen3tts`
//! (talker + code predictor at oracle token parity, codec at stage parity).
//!
//!   zig build qwen3tts -- --model models/qwen3-tts/qwen-talker-0.6b-customvoice-F32.gguf \
//!       --codec models/qwen3-tts/qwen-tokenizer-12hz-F32.gguf \
//!       --speaker Aiden -o out.wav < text.txt
//!
//! Frames stream through the chunked codec decode (12-frame chunks, 25-frame
//! left context by default) as generation proceeds; `[timing]` reports
//! time-to-first-audio and the frame rate vs the 12.5 fps real-time budget.

const std = @import("std");
const fucina = @import("fucina");

const llm = @import("fucina_llm");
const qtts = llm.qwen3tts;
const ExecContext = fucina.ExecContext;

const usage =
    \\usage: zig build qwen3tts -- --model <talker.gguf> --codec <codec.gguf>
    \\         [--speaker NAME] [--lang NAME] [--seed N | --greedy]
    \\         [--max-new N] [--chunk-frames N] [--left-ctx N]
    \\         [--threads N] -o <out.wav>   (text on stdin)
    \\
;

fn flagVal(args: []const [:0]const u8, name: []const u8) ?[]const u8 {
    for (args, 0..) |a, i| if (std.mem.eql(u8, a, name) and i + 1 < args.len) return args[i + 1];
    return null;
}
fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}
fn hasFlag(args: []const [:0]const u8, name: []const u8) bool {
    for (args) |a| if (std.mem.eql(u8, a, name)) return true;
    return false;
}

fn writeWav16(io: std.Io, path: []const u8, samples: []const f32) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var writer = file.writer(io, &buf);
    const w = &writer.interface;
    const data_len: u32 = @intCast(samples.len * 2);
    try w.writeAll("RIFF");
    try w.writeInt(u32, 36 + data_len, .little);
    try w.writeAll("WAVEfmt ");
    try w.writeInt(u32, 16, .little);
    try w.writeInt(u16, 1, .little); // PCM
    try w.writeInt(u16, 1, .little); // mono
    try w.writeInt(u32, qtts.codec.sample_rate, .little);
    try w.writeInt(u32, qtts.codec.sample_rate * 2, .little);
    try w.writeInt(u16, 2, .little);
    try w.writeInt(u16, 16, .little);
    try w.writeAll("data");
    try w.writeInt(u32, data_len, .little);
    for (samples) |s| {
        const clamped = std.math.clamp(s, -1.0, 1.0);
        try w.writeInt(i16, @intFromFloat(@round(clamped * 32767.0)), .little);
    }
    try w.flush();
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(allocator);

    var stderr_buf: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buf);
    const stderr = &stderr_writer.interface;
    defer stderr.flush() catch {};

    const talker_path = flagVal(args, "--model") orelse {
        try stderr.print("{s}", .{usage});
        return;
    };
    const codec_path = flagVal(args, "--codec") orelse {
        try stderr.print("{s}", .{usage});
        return;
    };
    const out_path = flagVal(args, "-o") orelse "out.wav";
    const speaker = flagVal(args, "--speaker");
    const lang = flagVal(args, "--lang") orelse "english";
    const greedy = hasFlag(args, "--greedy");
    const seed: i64 = if (flagVal(args, "--seed")) |s| try std.fmt.parseInt(i64, s, 10) else -1;
    const max_new: usize = if (flagVal(args, "--max-new")) |s| try std.fmt.parseInt(usize, s, 10) else 2048;
    const chunk_frames: usize = if (flagVal(args, "--chunk-frames")) |s| try std.fmt.parseInt(usize, s, 10) else 12;
    const left_ctx: usize = if (flagVal(args, "--left-ctx")) |s| try std.fmt.parseInt(usize, s, 10) else 25;
    if (flagVal(args, "--threads")) |t| fucina.parallel.setMaxThreads(try std.fmt.parseInt(usize, t, 10));

    // Read the utterance text from stdin.
    var stdin_reader = std.Io.File.stdin().reader(io, &.{});
    const text_raw = try stdin_reader.interface.allocRemaining(allocator, .limited(1 << 20));
    const text = std.mem.trim(u8, text_raw, " \t\r\n");
    if (text.len == 0) {
        try stderr.print("error: empty stdin text\n{s}", .{usage});
        return;
    }

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const t_load0 = nowNs(io);
    var talker_file = try fucina.gguf.File.loadMmap(allocator, io, talker_path);
    defer talker_file.deinit();
    var codec_file = try fucina.gguf.File.loadMmap(allocator, io, codec_path);
    defer codec_file.deinit();

    var model = try qtts.model.Model.load(&ctx, &talker_file);
    defer model.deinit();
    var dec = try qtts.codec.load(&ctx, &codec_file);
    defer dec.deinit();
    var tok = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &talker_file, .{});
    defer tok.deinit();
    try stderr.print("[load] {d:.2} s\n", .{@as(f64, @floatFromInt(nowNs(io) - t_load0)) / 1e9});

    var prompt = try qtts.prompt.build(allocator, &model, &tok, .{
        .text = text,
        .language = lang,
        .speaker = speaker,
    });
    defer prompt.deinit();

    // Resolve the seed like the oracle: -1 → random, logged for replay.
    const resolved_seed: i64 = if (greedy) 0 else if (seed >= 0) seed else blk: {
        // -1 → derive a printable random seed from the wall clock.
        break :blk @intCast(@as(u64, @intCast(std.Io.Clock.real.now(io).nanoseconds)) & 0x7FFF_FFFF_FFFF_FFFF);
    };
    if (!greedy) try stderr.print("[seed] {d}\n", .{resolved_seed});

    var kvs = try qtts.pipeline.Kvs.init(allocator, &model);
    defer kvs.deinit();
    const t_gen0 = nowNs(io);
    var result = try qtts.pipeline.generate(&ctx, &model, &prompt, .{
        .greedy = greedy,
        .seed = resolved_seed,
        .max_new_tokens = max_new,
    }, &kvs, null, null, null);
    defer result.deinit();
    const gen_s = @as(f64, @floatFromInt(nowNs(io) - t_gen0)) / 1e9;

    if (result.frames == 0) {
        try stderr.print("[gen] immediate EOS — no audio\n", .{});
        return;
    }

    // Transpose frame-major codes to the codec's [K, T] layout and decode
    // chunked (streaming-shaped even for the buffered file output).
    const kq = model.specials.num_code_groups;
    const kt = try allocator.alloc(i32, result.codes.len);
    defer allocator.free(kt);
    for (0..result.frames) |t| {
        for (0..kq) |k| kt[k * result.frames + t] = result.codes[t * kq + k];
    }
    const t_dec0 = nowNs(io);
    var audio: std.ArrayList(f32) = .empty;
    defer audio.deinit(allocator);
    // exact streaming decode: each frame decoded once (no left-context
    // re-decode); chunk size only sets the feed cadence
    _ = left_ctx;
    var sess = try qtts.codec.Streaming.init(allocator, &dec);
    defer sess.deinit();
    {
        const kq2 = dec.cfg.n_quantizers;
        const scratch = try allocator.alloc(i32, kq2 * chunk_frames);
        defer allocator.free(scratch);
        var start: usize = 0;
        while (start < result.frames) {
            const n = @min(chunk_frames, result.frames - start);
            for (0..kq2) |k| @memcpy(scratch[k * n ..][0..n], kt[k * result.frames + start ..][0..n]);
            const chunk = try sess.step(&ctx, scratch[0 .. kq2 * n], n);
            defer allocator.free(chunk);
            try audio.appendSlice(allocator, chunk);
            start += n;
        }
    }
    const dec_s = @as(f64, @floatFromInt(nowNs(io) - t_dec0)) / 1e9;

    try writeWav16(io, out_path, audio.items);
    const audio_s = @as(f64, @floatFromInt(audio.items.len)) / @as(f64, @floatFromInt(qtts.codec.sample_rate));
    // Optimize mode in the line: a Debug binary silently clobbering
    // zig-out once produced a 20x-wrong benchmark verdict.
    try stderr.print("[timing:{s}] {d} frames = {d:.2} s audio | talker {d:.2} s ({d:.1} fps vs 12.5 target) | codec {d:.2} s | total RTF {d:.2}\n", .{
        @tagName(@import("builtin").mode),
        result.frames, audio_s, gen_s, @as(f64, @floatFromInt(result.frames)) / gen_s, dec_s, (gen_s + dec_s) / audio_s,
    });
    try stderr.print("[out] {s}: {d} samples @ {d} Hz\n", .{ out_path, audio.items.len, qtts.codec.sample_rate });
}
