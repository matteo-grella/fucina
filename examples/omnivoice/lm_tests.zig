//! Tests for lm.zig — OmniVoice TTS LM forward.
//!
//! The logits-parity tests against the captured reference goldens
//! (refs/omnivoice-research/goldens/llm/) need the multi-GB model files under
//! models/omnivoice/ and are gated behind the OMNIVOICE_PARITY env var so
//! plain `zig build test` runs stay fast:
//!
//!   OMNIVOICE_PARITY=1 zig build test [-Doptimize=ReleaseSafe]

const std = @import("std");
const fucina = @import("fucina");
const lm = @import("lm.zig");

// ---------------------------------------------------------------------------
// Fast unit tests (no model files)
// ---------------------------------------------------------------------------

test "fillShiftedIds gates text positions to k*V and offsets audio ids" {
    // K=2, S=3: row 0 = text/audio mix, row 1 = audio codebook 1.
    const ids = [_]i32{ 9707, 42, 7, 100, 1024, 3 };
    const mask = [_]i32{ 0, 1, 1 };
    var out: [3]usize = undefined;

    // k=0: text position gated to 0+0*V; audio positions keep id + 0*V.
    try lm.fillShiftedIds(&ids, &mask, 0, 1025, &out);
    try std.testing.expectEqualSlices(usize, &.{ 0, 42, 7 }, &out);

    // k=1: gated text index k*V stays valid; audio ids offset by k*V.
    try lm.fillShiftedIds(&ids, &mask, 1, 1025, &out);
    try std.testing.expectEqualSlices(usize, &.{ 1025, 1024 + 1025, 3 + 1025 }, &out);
}

test "fillShiftedIds rejects out-of-range and negative audio ids" {
    var out: [2]usize = undefined;
    // Audio-masked id == V is out of range (valid ids are 0..V-1, incl. mask 1024).
    const too_big = [_]i32{ 1025, 0 };
    const mask_on = [_]i32{ 1, 1 };
    try std.testing.expectError(lm.Error.InvalidTokenId, lm.fillShiftedIds(&too_big, &mask_on, 0, 1025, &out));
    const negative = [_]i32{ -1, 0 };
    try std.testing.expectError(lm.Error.InvalidTokenId, lm.fillShiftedIds(&negative, &mask_on, 0, 1025, &out));
    // The same big id on a TEXT position is gated to 0 and passes.
    const mask_off = [_]i32{ 0, 0 };
    try lm.fillShiftedIds(&too_big, &mask_off, 0, 1025, &out);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0 }, &out);
}

test "buildUncondBias: window rows 1.0 on [0,u_len), diagonal-only tail" {
    var ctx: fucina.ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var bias = try lm.buildUncondBias(&ctx, 5, 3);
    defer bias.deinit();
    try std.testing.expectEqual(@as(usize, 5), bias.dim(.sq));
    try std.testing.expectEqual(@as(usize, 5), bias.dim(.skv));

    const data = try bias.dataConst();
    const expected = [_]f32{
        1, 1, 1, 0, 0, // sq=0: keys [0,3) get +1.0, padding tail +0.0
        1, 1, 1, 0, 0,
        1, 1, 1, 0, 0,
        0, 0, 0, 1, 0, // sq=3: diagonal only
        0, 0, 0, 0, 1, // sq=4: diagonal only
    };
    try std.testing.expectEqualSlices(f32, &expected, data);
}

test "TapSink captures owned named copies" {
    var ctx: fucina.ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var sink = lm.TapSink.init(std.testing.allocator);
    defer sink.deinit();

    const values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ 2, 3 }, &values);
    defer t.deinit();

    try sink.capture("embed", &t);
    const entry = sink.get("embed") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 2), entry.rows);
    try std.testing.expectEqual(@as(usize, 3), entry.cols);
    try std.testing.expectEqualSlices(f32, &values, entry.data);
    try std.testing.expect(sink.get("missing") == null);
}

// ---------------------------------------------------------------------------
// Parity gate (env-gated): forward vs captured reference logits
// ---------------------------------------------------------------------------

const goldens_dir = "refs/omnivoice-research/goldens/llm";

const GoldenInput = struct {
    num_codebooks: usize,
    seq_len: usize,
    ids: []i32,
    audio_mask: []i32,

    fn deinit(self: *GoldenInput, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.audio_mask);
        self.* = undefined;
    }
};

/// input.bin: [i32 K][i32 S][K*S i32 ids k-slow][S i32 audio_mask].
fn readGoldenInput(allocator: std.mem.Allocator, io: std.Io) !GoldenInput {
    const bytes = try readFileBytes(allocator, io, goldens_dir ++ "/input.bin");
    defer allocator.free(bytes);
    if (bytes.len < 8) return error.CorruptGolden;

    const k_raw = std.mem.readInt(i32, bytes[0..4], .little);
    const s_raw = std.mem.readInt(i32, bytes[4..8], .little);
    if (k_raw <= 0 or s_raw <= 0) return error.CorruptGolden;
    const num_codebooks: usize = @intCast(k_raw);
    const seq_len: usize = @intCast(s_raw);
    if (bytes.len != 8 + (num_codebooks * seq_len + seq_len) * 4) return error.CorruptGolden;

    const ids = try allocator.alloc(i32, num_codebooks * seq_len);
    errdefer allocator.free(ids);
    for (ids, 0..) |*dst, i| {
        dst.* = std.mem.readInt(i32, bytes[8 + i * 4 ..][0..4], .little);
    }
    const audio_mask = try allocator.alloc(i32, seq_len);
    errdefer allocator.free(audio_mask);
    const mask_base = 8 + ids.len * 4;
    for (audio_mask, 0..) |*dst, i| {
        dst.* = std.mem.readInt(i32, bytes[mask_base + i * 4 ..][0..4], .little);
    }
    return .{ .num_codebooks = num_codebooks, .seq_len = seq_len, .ids = ids, .audio_mask = audio_mask };
}

/// logits-<dtype>.bin: [i32 V][i32 K][i32 S][V*K*S f32], flat [S,K,V] (v
/// fastest) — identical to our forward's [logits_len, K*V] row layout.
fn readGoldenLogits(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]f32 {
    const bytes = try readFileBytes(allocator, io, path);
    defer allocator.free(bytes);
    if (bytes.len < 12) return error.CorruptGolden;

    const v_raw = std.mem.readInt(i32, bytes[0..4], .little);
    const k_raw = std.mem.readInt(i32, bytes[4..8], .little);
    const s_raw = std.mem.readInt(i32, bytes[8..12], .little);
    if (v_raw <= 0 or k_raw <= 0 or s_raw <= 0) return error.CorruptGolden;
    const numel = @as(usize, @intCast(v_raw)) * @as(usize, @intCast(k_raw)) * @as(usize, @intCast(s_raw));
    if (bytes.len != 12 + numel * 4) return error.CorruptGolden;

    const data = try allocator.alloc(f32, numel);
    errdefer allocator.free(data);
    for (data, 0..) |*dst, i| {
        dst.* = @bitCast(std.mem.readInt(u32, bytes[12 + i * 4 ..][0..4], .little));
    }
    return data;
}

fn readFileBytes(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotAFile;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}

const CompareStats = struct {
    cosine: f64,
    min_row_cosine: f64,
    max_abs_diff: f64,
};

fn cosineOf(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |av, bv| {
        const x: f64 = av;
        const y: f64 = bv;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if (na < 1e-30 or nb < 1e-30) return 0.0;
    return dot / (@sqrt(na) * @sqrt(nb));
}

fn compareLogits(got: []const f32, want: []const f32, row_len: usize) CompareStats {
    var max_abs: f64 = 0;
    for (got, want) |gv, wv| {
        const diff = @abs(@as(f64, gv) - @as(f64, wv));
        if (diff > max_abs) max_abs = diff;
    }
    var min_row_cos: f64 = 1.0;
    var row: usize = 0;
    while (row * row_len < got.len) : (row += 1) {
        const cos = cosineOf(got[row * row_len ..][0..row_len], want[row * row_len ..][0..row_len]);
        if (cos < min_row_cos) min_row_cos = cos;
    }
    return .{ .cosine = cosineOf(got, want), .min_row_cosine = min_row_cos, .max_abs_diff = max_abs };
}

test "OMNIVOICE_PARITY: forward logits parity vs reference goldens (all base dtypes)" {
    if (std.c.getenv("OMNIVOICE_PARITY") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var input = try readGoldenInput(allocator, io);
    defer input.deinit(allocator);

    const dtypes = [_][]const u8{ "F32", "BF16", "Q8_0", "Q4_K_M" };
    var ran: usize = 0;
    for (dtypes) |dtype_name| {
        var model_path_buf: [128]u8 = undefined;
        const model_path = try std.fmt.bufPrint(&model_path_buf, "models/omnivoice/omnivoice-base-{s}.gguf", .{dtype_name});

        var file = fucina.gguf.File.loadMmap(allocator, io, model_path) catch |err| switch (err) {
            error.FileNotFound => {
                std.debug.print("omnivoice-lm parity {s}: model file missing, skipped\n", .{dtype_name});
                continue;
            },
            else => return err,
        };
        defer file.deinit();

        var golden_path_buf: [128]u8 = undefined;
        const golden_path = try std.fmt.bufPrint(&golden_path_buf, goldens_dir ++ "/logits-{s}.bin", .{dtype_name});
        const want = try readGoldenLogits(allocator, io, golden_path);
        defer allocator.free(want);

        var ctx: fucina.ExecContext = undefined;
        ctx.init(allocator);
        defer ctx.deinit();

        var model = try lm.loadModel(&ctx, &file);
        defer model.deinit();
        try std.testing.expectEqual(input.num_codebooks, model.config.num_audio_codebook);

        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        var logits = try model.forward(&ctx, input.ids, input.audio_mask, 0, input.seq_len, null);
        defer logits.deinit();
        const forward_ns = std.Io.Clock.awake.now(io).nanoseconds - t0;

        const got = try logits.dataConst();
        try std.testing.expectEqual(want.len, got.len);

        const row_len = model.config.audioTableRows(); // K*V = 8200
        const stats = compareLogits(got, want, row_len);
        std.debug.print(
            "omnivoice-lm parity {s}: cosine={d:.9} min_row_cosine={d:.9} max_abs={d:.6} forward={d:.1}ms\n",
            .{ dtype_name, stats.cosine, stats.min_row_cosine, stats.max_abs_diff, @as(f64, @floatFromInt(forward_ns)) / 1e6 },
        );

        try std.testing.expect(stats.cosine >= 0.9999);
        try std.testing.expect(stats.min_row_cosine >= 0.999);
        if (std.mem.eql(u8, dtype_name, "F32")) {
            try std.testing.expect(stats.max_abs_diff < 0.05);
        }
        ran += 1;
    }
    if (ran == 0) return error.SkipZigTest;
}

test "OMNIVOICE_PARITY: taps capture the reference dump set" {
    if (std.c.getenv("OMNIVOICE_PARITY") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var file = fucina.gguf.File.loadMmap(allocator, io, "models/omnivoice/omnivoice-base-Q4_K_M.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    var input = try readGoldenInput(allocator, io);
    defer input.deinit(allocator);

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try lm.loadModel(&ctx, &file);
    defer model.deinit();

    var sink = lm.TapSink.init(allocator);
    defer sink.deinit();

    var logits = try model.forward(&ctx, input.ids, input.audio_mask, 0, input.seq_len, &sink);
    defer logits.deinit();

    // embed + 15 layer taps + 4 layer-1 sub-taps + final = 21 entries, each [S, H].
    try std.testing.expectEqual(@as(usize, 21), sink.entries.items.len);
    const expected_names = [_][]const u8{ "embed", "l0", "l1-norm1", "l1-attn", "l1-norm2", "l1-mlp", "l1", "l6", "l13", "l20", "final" };
    for (expected_names) |name| {
        const entry = sink.get(name) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(input.seq_len, entry.rows);
        try std.testing.expectEqual(model.config.hidden_size, entry.cols);
    }
}
