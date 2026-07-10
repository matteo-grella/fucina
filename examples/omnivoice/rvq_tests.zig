//! Tests for rvq.zig: synthetic tiny-dim decoder vs a naive host reference
//! (gather → project_out+bias per codebook, 8 bias adds accumulate, then
//! fc2+bias), input validation, the ggml argmax tie-break, and the
//! env-gated ENCODE parity gates vs the captured reference goldens:
//!
//!   OMNIVOICE_PARITY=1 zig build test [-Doptimize=ReleaseFast]
//!
//! (needs models/omnivoice/omnivoice-tokenizer-*.gguf + models/en_4.wav +
//! refs/omnivoice-research/goldens/; skipped cleanly otherwise).

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const codec = @import("codec.zig");
const dump = @import("dump.zig");
const postproc = @import("postproc.zig");
const rvq = @import("rvq.zig");
const rvq_file = @import("rvq_file.zig");
const wav = @import("wav.zig");

const v_size = 4; // codebook entries
const d_dim = 2; // codebook dim
const h_dim = 3; // latent (project_out rows)
const fc_dim = 2; // fc2 rows

/// Deterministic pseudo-random weight fill.
fn fill(values: []f32, seed: u64) void {
    var state = seed *% 0x9E3779B97F4A7C15 +% 1;
    for (values) |*value| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const bits: u32 = @truncate(state >> 33);
        value.* = (@as(f32, @floatFromInt(bits % 2000)) - 1000.0) / 500.0;
    }
}

const Synthetic = struct {
    dec: codec.RvqDecoder,
    embed: [codec.n_codebooks][v_size * d_dim]f32,
    proj_w: [codec.n_codebooks][h_dim * d_dim]f32,
    proj_b: [codec.n_codebooks][h_dim]f32,
    fc2_w: [fc_dim * h_dim]f32,
    fc2_b: [fc_dim]f32,
};

fn buildSynthetic(ctx: *fucina.ExecContext, allocator: std.mem.Allocator) !Synthetic {
    var out: Synthetic = undefined;
    for (0..codec.n_codebooks) |k| {
        fill(&out.embed[k], 11 + k);
        fill(&out.proj_w[k], 101 + k);
        fill(&out.proj_b[k], 201 + k);
    }
    fill(&out.fc2_w, 301);
    fill(&out.fc2_b, 302);

    for (0..codec.n_codebooks) |k| {
        var embed = try codec.Codebook.fromSlice(ctx, .{ v_size, d_dim }, &out.embed[k]);
        errdefer embed.deinit();
        const embed_sq = try allocator.alloc(f32, v_size);
        errdefer allocator.free(embed_sq);
        for (embed_sq, 0..) |*sq, j| {
            var sum: f32 = 0.0;
            for (out.embed[k][j * d_dim ..][0..d_dim]) |x| sum += x * x;
            sq.* = sum;
        }
        var weight = try llm.weights.WeightF32.fromSlice(ctx, .{ h_dim, d_dim }, &out.proj_w[k]);
        errdefer weight.deinit();
        const bias = try allocator.dupe(f32, &out.proj_b[k]);
        errdefer allocator.free(bias);
        out.dec.quantizers[k] = .{
            .embed = embed,
            .embed_sq = embed_sq,
            .project_out = .{ .f32 = weight },
            .project_out_bias = bias,
        };
    }
    var fc2_w = try llm.weights.WeightF32.fromSlice(ctx, .{ fc_dim, h_dim }, &out.fc2_w);
    errdefer fc2_w.deinit();
    out.dec.fc2 = .{ .f32 = fc2_w };
    out.dec.fc2_bias = try allocator.dupe(f32, &out.fc2_b);
    return out;
}

test "rvq decode matches the naive host reference (8 bias adds accumulate)" {
    const allocator = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var syn = try buildSynthetic(&ctx, allocator);
    defer syn.dec.deinit(allocator);

    const t = 3;
    // [8, T] k-slow codes.
    const codes = [codec.n_codebooks * t]i32{
        0, 3, 1,
        2, 2, 0,
        1, 0, 3,
        3, 1, 2,
        0, 0, 0,
        3, 3, 3,
        1, 2, 1,
        2, 1, 0,
    };

    var out = try rvq.decode(&ctx, &syn.dec, &codes, t);
    defer out.deinit();

    // Naive reference: latent[t][j] = Σ_k (Σ_d W_k[j][d]·embed_k[code][d] + b_k[j]).
    var expected_latent: [t * h_dim]f32 = @splat(0.0);
    for (0..codec.n_codebooks) |k| {
        for (0..t) |ti| {
            const code: usize = @intCast(codes[k * t + ti]);
            for (0..h_dim) |j| {
                var sum: f32 = syn.proj_b[k][j];
                for (0..d_dim) |dd| {
                    sum += syn.proj_w[k][j * d_dim + dd] * syn.embed[k][code * d_dim + dd];
                }
                expected_latent[ti * h_dim + j] += sum;
            }
        }
    }
    var expected_fc2: [t * fc_dim]f32 = undefined;
    for (0..t) |ti| {
        for (0..fc_dim) |o| {
            var sum: f32 = syn.fc2_b[o];
            for (0..h_dim) |j| {
                sum += syn.fc2_w[o * h_dim + j] * expected_latent[ti * h_dim + j];
            }
            expected_fc2[ti * fc_dim + o] = sum;
        }
    }

    try std.testing.expectEqual(@as(usize, t), out.latent.dim(.seq));
    try std.testing.expectEqual(@as(usize, h_dim), out.latent.dim(.d));
    try std.testing.expectEqual(@as(usize, t), out.fc2_out.dim(.seq));
    try std.testing.expectEqual(@as(usize, fc_dim), out.fc2_out.dim(.fc));

    const latent = try out.latent.dataConst();
    for (latent, expected_latent) |got, want| {
        try std.testing.expectApproxEqAbs(want, got, 1e-4);
    }
    const fc2_out = try out.fc2_out.dataConst();
    for (fc2_out, expected_fc2) |got, want| {
        try std.testing.expectApproxEqAbs(want, got, 1e-4);
    }
}

test "rvq decode rejects bad inputs" {
    const allocator = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var syn = try buildSynthetic(&ctx, allocator);
    defer syn.dec.deinit(allocator);

    const short = [_]i32{ 0, 1, 2 };
    try std.testing.expectError(rvq.Error.InvalidCodes, rvq.decode(&ctx, &syn.dec, &short, 1));
    try std.testing.expectError(rvq.Error.InvalidCodes, rvq.decode(&ctx, &syn.dec, &short, 0));

    var out_of_range: [codec.n_codebooks]i32 = @splat(0);
    out_of_range[3] = v_size; // == codebook size → out of range
    try std.testing.expectError(rvq.Error.CodeOutOfRange, rvq.decode(&ctx, &syn.dec, &out_of_range, 1));
    out_of_range[3] = -1;
    try std.testing.expectError(rvq.Error.CodeOutOfRange, rvq.decode(&ctx, &syn.dec, &out_of_range, 1));
}

test "scoreArgmaxRow: ggml last-index-wins tie-break on the fused score" {
    // score = 2*dot − sq; craft ties and strict maxima.
    const zero_sq = [_]f32{ 0, 0, 0, 0 };
    // Scores: [2, 6, 6, 4] — ties at indices 1 and 2 → LAST equal wins.
    try std.testing.expectEqual(@as(usize, 2), rvq.scoreArgmaxRow(&.{ 1, 3, 3, 2 }, &zero_sq));
    // Strictly increasing then decreasing: unique max wins.
    try std.testing.expectEqual(@as(usize, 1), rvq.scoreArgmaxRow(&.{ 1, 5, 3, 2 }, &zero_sq));
    // All-equal row: the LAST index wins.
    try std.testing.expectEqual(@as(usize, 3), rvq.scoreArgmaxRow(&.{ 7, 7, 7, 7 }, &zero_sq));
    // embed_sq shifts the score: 2*3−0 = 6 beats 2*4−3 = 5.
    try std.testing.expectEqual(@as(usize, 0), rvq.scoreArgmaxRow(&.{ 3, 4 }, &.{ 0, 3 }));
    // -inf scores at the front never overwrite once beaten.
    try std.testing.expectEqual(@as(usize, 1), rvq.scoreArgmaxRow(&.{ -std.math.inf(f32), 0, -1 }, &.{ 0, 0, 0 }));
}

// ---------------------------------------------------------------------------
// Encode parity gates (env-gated): full pipeline vs reference goldens
// ---------------------------------------------------------------------------

const goldens_dir = "refs/omnivoice-research/goldens";
const tokenizer_f32 = "models/omnivoice/omnivoice-tokenizer-F32.gguf";
const en4_wav = "models/en_4.wav";

const CosStats = struct {
    cosine: f64,
    max_abs: f64,
    exact: bool,
};

fn cosStats(a: []const f32, b: []const f32) CosStats {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var max_abs: f64 = 0;
    for (a, b) |av, bv| {
        const x: f64 = av;
        const y: f64 = bv;
        dot += x * y;
        na += x * x;
        nb += y * y;
        max_abs = @max(max_abs, @abs(x - y));
    }
    const cosine: f64 = if (na < 1e-30 or nb < 1e-30) 0.0 else dot / (@sqrt(na) * @sqrt(nb));
    return .{ .cosine = cosine, .max_abs = max_abs, .exact = max_abs == 0.0 };
}

fn checkTap(name: []const u8, golden: []const f32, ours: []const f32) !void {
    try std.testing.expectEqual(golden.len, ours.len);
    const stats = cosStats(golden, ours);
    std.debug.print(
        "omnivoice-codec parity {s}: cos={d:.9} max_abs={e:.3}{s}\n",
        .{ name, stats.cosine, stats.max_abs, if (stats.exact) " (bit-exact)" else "" },
    );
    try std.testing.expect(stats.cosine >= 0.9999);
}

fn readRawF32(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]f32 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const byte_len: usize = @intCast(stat.size);
    if (byte_len % 4 != 0) return error.CorruptRawFile;
    const bytes = try allocator.alloc(u8, byte_len);
    defer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    const values = try allocator.alloc(f32, byte_len / 4);
    for (values, 0..) |*dst, i| dst.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return values;
}

/// Reads en_4.wav → reference preprocessing → hop alignment (the codec CLI
/// encode front-end). Caller frees.
fn loadPreprocessedEn4(allocator: std.mem.Allocator, io: std.Io) ![]f32 {
    var samples = try wav.readMono(io, allocator, en4_wav, 24000);
    errdefer allocator.free(samples);
    _ = try postproc.refPreprocessAudio(allocator, &samples, 24000, true);
    const n_aligned = (samples.len / 960) * 960;
    if (n_aligned == 0) return error.InputTooShort;
    if (n_aligned != samples.len) {
        samples = try allocator.realloc(samples, n_aligned);
    }
    return samples;
}

fn perCodebookMatch(golden: []const i32, ours: []const i32, out: *[codec.n_codebooks]usize) usize {
    const t = golden.len / codec.n_codebooks;
    var total: usize = 0;
    for (0..codec.n_codebooks) |k| {
        var m: usize = 0;
        for (0..t) |ti| {
            if (golden[k * t + ti] == ours[k * t + ti]) m += 1;
        }
        out[k] = m;
        total += m;
    }
    return total;
}

test "OMNIVOICE_PARITY: encode taps + codes vs reference goldens (F32 tokenizer)" {
    if (std.c.getenv("OMNIVOICE_PARITY") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var file = fucina.gguf.File.loadMmap(allocator, io, tokenizer_f32) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var cdc = try codec.Codec.load(&ctx, &file);
    defer cdc.deinit();
    var enc = try codec.Encoder.load(&ctx, &file);
    defer enc.deinit();

    const audio = try loadPreprocessedEn4(allocator, io);
    defer allocator.free(audio);

    var taps = rvq.EncodeTaps.init(allocator);
    defer taps.deinit();

    const t0 = std.Io.Clock.awake.now(io).nanoseconds;
    const codes = try rvq.encode(&ctx, allocator, cdc.config, &cdc.rvq, &enc, audio, &taps);
    defer allocator.free(codes);
    const encode_ns = std.Io.Clock.awake.now(io).nanoseconds - t0;
    std.debug.print("omnivoice-codec parity: encode {d} samples in {d:.1} ms\n", .{ audio.len, @as(f64, @floatFromInt(encode_ns)) / 1e6 });

    // ref-audio-16k (post-resample, pre-pad).
    {
        const golden = try dump.readFile(allocator, io, goldens_dir ++ "/tts-clone/ref-audio-16k.bin");
        defer allocator.free(golden.shape);
        defer allocator.free(golden.data);
        try checkTap("ref-audio-16k", golden.data, taps.audio_16k.?);
    }

    // HuBERT taps. Our taps are [T, C] row-major; the golden feat-extract
    // payload is CHANNEL-PLANAR (the reference tensor is T-fast there, and
    // its dumper copies the flat ggml buffer), while all C-first transformer
    // taps land as (T, C) row-major directly.
    {
        const golden = try dump.readFile(allocator, io, goldens_dir ++ "/tts-clone/hubert-feat-extract.bin");
        defer allocator.free(golden.shape);
        defer allocator.free(golden.data);
        const tap = taps.hubert.feat_extract.?;
        try std.testing.expectEqual(golden.data.len, tap.data.len);
        const planar = try allocator.alloc(f32, tap.data.len);
        defer allocator.free(planar);
        for (0..tap.t) |ti| {
            for (0..tap.c) |ci| planar[ci * tap.t + ti] = tap.data[ti * tap.c + ci];
        }
        try checkTap("hubert-feat-extract", golden.data, planar);
    }
    const row_major_taps = [_]struct { name: []const u8, ours: []const f32 }{
        .{ .name = "hubert-feat-proj-ln", .ours = taps.hubert.proj_ln.?.data },
        .{ .name = "hubert-feat-proj", .ours = taps.hubert.proj.?.data },
        .{ .name = "hubert-enc-init", .ours = taps.hubert.enc_init.?.data },
        .{ .name = "hubert-l0", .ours = taps.hubert.layer[0].?.data },
        .{ .name = "hubert-l5", .ours = taps.hubert.layer[5].?.data },
        .{ .name = "hubert-l7", .ours = taps.hubert.layer[7].?.data },
        .{ .name = "hubert-l9", .ours = taps.hubert.layer[9].?.data },
        .{ .name = "hubert-l11", .ours = taps.hubert.layer[11].?.data },
        .{ .name = "ref-hubert-features", .ours = taps.features.?.data },
    };
    for (row_major_taps) |entry| {
        var path_buf: [128]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, goldens_dir ++ "/tts-clone/{s}.bin", .{entry.name});
        const golden = try dump.readFile(allocator, io, path);
        defer allocator.free(golden.shape);
        defer allocator.free(golden.data);
        try checkTap(entry.name, golden.data, entry.ours);
    }

    // pre_fc / embed (headerless cpp raw dumps, [T, 1024] row-major flat)
    // + the e_acoustic / e_semantic column slices of pre_fc.
    const golden_pre_fc = try readRawF32(allocator, io, goldens_dir ++ "/tts-clone/cpp_encode_pre_fc.raw");
    defer allocator.free(golden_pre_fc);
    try checkTap("encode_pre_fc", golden_pre_fc, taps.pre_fc.?.data);
    {
        const t = golden_pre_fc.len / 1024;
        const ac = try allocator.alloc(f32, t * 256);
        defer allocator.free(ac);
        const sem = try allocator.alloc(f32, t * 768);
        defer allocator.free(sem);
        for (0..t) |ti| {
            @memcpy(ac[ti * 256 ..][0..256], golden_pre_fc[ti * 1024 ..][0..256]);
            @memcpy(sem[ti * 768 ..][0..768], golden_pre_fc[ti * 1024 + 256 ..][0..768]);
        }
        try checkTap("e-acoustic", ac, taps.e_acoustic.?.data);
        try checkTap("e-semantic", sem, taps.e_semantic.?.data);
    }
    {
        const golden_embed = try readRawF32(allocator, io, goldens_dir ++ "/tts-clone/cpp_encode_embed.raw");
        defer allocator.free(golden_embed);
        try checkTap("encode_embed", golden_embed, taps.embed.?.data);
    }

    // Codes: the byte-exact oracle (en_4.rvq) + the TTS-run capture.
    const golden_codes = try rvq_file.readFile(allocator, io, goldens_dir ++ "/en_4.rvq", codec.n_codebooks);
    defer allocator.free(golden_codes);
    try std.testing.expectEqual(golden_codes.len, codes.len);
    const t_codes = codes.len / codec.n_codebooks;

    var per: [codec.n_codebooks]usize = undefined;
    const total = perCodebookMatch(golden_codes, codes, &per);
    std.debug.print("omnivoice-codec parity codes vs en_4.rvq: {d}/{d}", .{ total, codes.len });
    for (per) |m| std.debug.print(" {d}/{d}", .{ m, t_codes });
    std.debug.print("\n", .{});
    for (per) |m| {
        // Gate: ≥ 99.5% exact per codebook (expected: 100%).
        try std.testing.expect(@as(f64, @floatFromInt(m)) / @as(f64, @floatFromInt(t_codes)) >= 0.995);
    }

    {
        const rc = try dump.readFile(allocator, io, goldens_dir ++ "/tts-clone/ref-audio-codes.bin");
        defer allocator.free(rc.shape);
        defer allocator.free(rc.data);
        try std.testing.expectEqual(codes.len, rc.data.len);
        var mismatches: usize = 0;
        for (rc.data, codes) |golden_f, got| {
            const golden_code: i32 = @intFromFloat(@round(golden_f));
            if (golden_code != got) mismatches += 1;
        }
        std.debug.print("omnivoice-codec parity codes vs ref-audio-codes: {d}/{d}\n", .{ codes.len - mismatches, codes.len });
        try std.testing.expect(@as(f64, @floatFromInt(codes.len - mismatches)) / @as(f64, @floatFromInt(codes.len)) >= 0.995);
    }
}

test "OMNIVOICE_PARITY: all tokenizer dtypes encode (code match vs F32, informational)" {
    if (std.c.getenv("OMNIVOICE_PARITY") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const audio = loadPreprocessedEn4(allocator, io) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(audio);

    var f32_codes: ?[]i32 = null;
    defer if (f32_codes) |cs| allocator.free(cs);

    const dtypes = [_][]const u8{ "F32", "BF16", "Q8_0", "Q4_K_M" };
    var ran: usize = 0;
    for (dtypes) |dtype_name| {
        var model_path_buf: [128]u8 = undefined;
        const model_path = try std.fmt.bufPrint(&model_path_buf, "models/omnivoice/omnivoice-tokenizer-{s}.gguf", .{dtype_name});
        var file = fucina.gguf.File.loadMmap(allocator, io, model_path) catch |err| switch (err) {
            error.FileNotFound => {
                std.debug.print("omnivoice-codec encode {s}: model file missing, skipped\n", .{dtype_name});
                continue;
            },
            else => return err,
        };
        defer file.deinit();

        var ctx: fucina.ExecContext = undefined;
        ctx.init(allocator);
        defer ctx.deinit();

        var cdc = try codec.Codec.load(&ctx, &file);
        defer cdc.deinit();
        var enc = try codec.Encoder.load(&ctx, &file);
        defer enc.deinit();

        const codes = try rvq.encode(&ctx, allocator, cdc.config, &cdc.rvq, &enc, audio, null);
        errdefer allocator.free(codes);
        try std.testing.expectEqual(@as(usize, codec.n_codebooks * (audio.len / 960)), codes.len);

        if (f32_codes) |base| {
            defer allocator.free(codes);
            var per: [codec.n_codebooks]usize = undefined;
            const total = perCodebookMatch(base, codes, &per);
            std.debug.print(
                "omnivoice-codec encode {s}: code match vs F32 = {d}/{d} ({d:.2}%)\n",
                .{ dtype_name, total, codes.len, 100.0 * @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(codes.len)) },
            );
        } else {
            f32_codes = codes;
            std.debug.print("omnivoice-codec encode {s}: baseline codes captured\n", .{dtype_name});
        }
        ran += 1;
    }
    if (ran == 0) return error.SkipZigTest;
}
