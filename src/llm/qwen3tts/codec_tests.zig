//! Codec decode parity vs the qwentts.cpp oracle: unpack the golden `.rvq`
//! (137 frames of ref_audio_2), decode, and compare against the oracle's F32
//! WAV decode sample-by-sample. Skips without the codec GGUF + goldens.

const std = @import("std");
const fucina = @import("fucina");
const codec = @import("codec.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

const model_path = "models/qwen3-tts/qwen-tokenizer-12hz-F32.gguf";
const rvq_path = "goldens-qwen3tts/codes-137.rvq";
const wav_path = "goldens-qwen3tts/decoded-137-f32.wav";

fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(1 << 30));
}

/// Minimal RIFF parse for the oracle's F32 mono output: find the `data`
/// chunk and reinterpret as f32 little-endian.
fn wavF32Samples(allocator: std.mem.Allocator, bytes: []const u8) ![]f32 {
    if (bytes.len < 44 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE"))
        return error.BadWav;
    var off: usize = 12;
    while (off + 8 <= bytes.len) {
        const id = bytes[off..][0..4];
        const size = std.mem.readInt(u32, bytes[off + 4 ..][0..4], .little);
        if (std.mem.eql(u8, id, "data")) {
            const payload = bytes[off + 8 ..][0..@min(size, bytes.len - off - 8)];
            const n = payload.len / 4;
            const out = try allocator.alloc(f32, n);
            for (out, 0..) |*s, i| {
                s.* = @bitCast(std.mem.readInt(u32, payload[i * 4 ..][0..4], .little));
            }
            return out;
        }
        off += 8 + size + (size & 1);
    }
    return error.BadWav;
}

test "qwen3tts codec: golden rvq decodes to the oracle waveform" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = gguf.File.loadMmap(allocator, std.testing.io, model_path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();
    const rvq_bytes = readFile(allocator, rvq_path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(rvq_bytes);
    const golden_bytes = try readFile(allocator, wav_path);
    defer allocator.free(golden_bytes);
    const golden = try wavF32Samples(allocator, golden_bytes);
    defer allocator.free(golden);

    var dec = try codec.load(&ctx, &file);
    defer dec.deinit();

    const unpacked = try codec.unpackRvq(allocator, rvq_bytes);
    defer allocator.free(unpacked.codes);
    try std.testing.expectEqual(@as(usize, 137), unpacked.t);

    const audio = try codec.decode(&ctx, &dec, unpacked.codes, unpacked.t);
    defer allocator.free(audio);
    try std.testing.expectEqual(golden.len, audio.len);

    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var max_diff: f64 = 0;
    for (audio, golden) |a, b| {
        dot += @as(f64, a) * @as(f64, b);
        na += @as(f64, a) * @as(f64, a);
        nb += @as(f64, b) * @as(f64, b);
        max_diff = @max(max_diff, @abs(@as(f64, a) - @as(f64, b)));
    }
    const cosine = dot / (@sqrt(na) * @sqrt(nb));
    if (cosine < 0.9999) {
        std.debug.print("[qwen3tts codec] cosine {d:.7} max|diff| {d:.6}\n", .{ cosine, max_diff });
        return error.TestUnexpectedResult;
    }
}

fn loadDump(allocator: std.mem.Allocator, path: []const u8) !struct { rows: usize, cols: usize, data: []f32 } {
    const bytes = try readFile(allocator, path);
    defer allocator.free(bytes);
    const ndims = std.mem.readInt(i32, bytes[0..4], .little);
    if (ndims != 2) return error.BadDump;
    const d0: usize = @intCast(std.mem.readInt(i32, bytes[4..8], .little));
    const d1: usize = @intCast(std.mem.readInt(i32, bytes[8..12], .little));
    const data = try allocator.alloc(f32, d0 * d1);
    for (data, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[12 + i * 4 ..][0..4], .little));
    return .{ .rows = d0, .cols = d1, .data = data };
}

fn stageCosine(name: []const u8, got_tc: []const f32, dump: anytype, dump_is_ct: bool, t: usize, c: usize) void {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..t) |ti| {
        for (0..c) |ci| {
            const a = got_tc[ti * c + ci];
            const b = if (dump_is_ct) dump.data[ci * t + ti] else dump.data[ti * c + ci];
            dot += @as(f64, a) * @as(f64, b);
            na += @as(f64, a) * @as(f64, a);
            nb += @as(f64, b) * @as(f64, b);
        }
    }
    std.debug.print("[stage {s}] cosine {d:.7}\n", .{ name, dot / (@sqrt(na) * @sqrt(nb)) });
}

test "qwen3tts codec: stage bisect vs oracle dumps (debug, needs dump-codec/)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = gguf.File.loadMmap(allocator, std.testing.io, model_path) catch return error.SkipZigTest;
    defer file.deinit();
    const rvq_bytes = readFile(allocator, rvq_path) catch return error.SkipZigTest;
    defer allocator.free(rvq_bytes);
    const rvq_dump = loadDump(allocator, "goldens-qwen3tts/dump-codec/codec-rvq.bin") catch return error.SkipZigTest;
    defer allocator.free(rvq_dump.data);
    const preconv_dump = try loadDump(allocator, "goldens-qwen3tts/dump-codec/codec-preconv.bin");
    defer allocator.free(preconv_dump.data);
    const tfm_dump = try loadDump(allocator, "goldens-qwen3tts/dump-codec/codec-tfm.bin");
    defer allocator.free(tfm_dump.data);
    const up_dump = try loadDump(allocator, "goldens-qwen3tts/dump-codec/codec-up.bin");
    defer allocator.free(up_dump.data);

    var dec = try codec.load(&ctx, &file);
    defer dec.deinit();
    const unpacked = try codec.unpackRvq(allocator, rvq_bytes);
    defer allocator.free(unpacked.codes);
    const t = unpacked.t;

    var taps = codec.Taps{ .allocator = allocator };
    defer taps.deinit();
    const audio = try codec.decodeWithTaps(&ctx, &dec, unpacked.codes, t, &taps);
    defer allocator.free(audio);

    // Layouts: rvq dump [T,512] row-major; preconv [1024,T] (C-major);
    // tfm [T,1024]; up [1024, T*4] (C-major).
    stageCosine("rvq", taps.rvq.?, rvq_dump, false, t, 512);
    stageCosine("preconv", taps.preconv.?, preconv_dump, true, t, 1024);
    stageCosine("tfm", taps.tfm.?, tfm_dump, false, t, 1024);
    stageCosine("up", taps.up.?, up_dump, true, t * 4, 1024);
}

test "codec: streaming session equals whole-clip decode" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    var file = gguf.File.loadMmap(allocator, std.testing.io, model_path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var dec = try codec.load(&ctx, &file);
    defer dec.deinit();

    // random-but-fixed codes, 30 frames; feed as 7 + 12 + 11
    const t: usize = 30;
    const kq = dec.cfg.n_quantizers;
    const codes = try allocator.alloc(i32, kq * t);
    defer allocator.free(codes);
    var prng = std.Random.DefaultPrng.init(3);
    for (codes) |*c| c.* = @intCast(prng.random().uintLessThan(usize, dec.cfg.codebook_size));

    const whole = try codec.decode(&ctx, &dec, codes, t);
    defer allocator.free(whole);

    var sess = try codec.Streaming.init(allocator, &dec);
    defer sess.deinit();
    var got: std.ArrayList(f32) = .empty;
    defer got.deinit(allocator);
    const scratch = try allocator.alloc(i32, kq * 12);
    defer allocator.free(scratch);
    var start: usize = 0;
    for ([_]usize{ 7, 12, 11 }) |n| {
        for (0..kq) |k| @memcpy(scratch[k * n ..][0..n], codes[k * t + start ..][0..n]);
        const chunk = try sess.step(&ctx, scratch[0 .. kq * n], n);
        defer allocator.free(chunk);
        try got.appendSlice(allocator, chunk);
        start += n;
    }
    try std.testing.expectEqual(whole.len, got.items.len);
    var md: f64 = 0;
    for (whole, got.items) |a, b| md = @max(md, @abs(@as(f64, a) - @as(f64, b)));
    if (md > 1e-4) {
        std.debug.print("[codec-stream] max|diff| {d:.7}\n", .{md});
        return error.TestUnexpectedResult;
    }
    std.debug.print("[codec-stream] streaming == whole-clip (max|diff| {e:.2})\n", .{md});
}
