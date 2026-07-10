//! Tests for mg_decode.zig — the MaskGIT decode loop.
//!
//! The token-parity gates against the captured reference goldens
//! (refs/omnivoice-research/goldens/{maskgit,tts-design}/) run 32 steps x 2
//! full LM forwards each, need the multi-GB model files under
//! models/omnivoice/, and are gated behind OMNIVOICE_PARITY (run them in an
//! optimized build — Debug takes far too long):
//!
//!   OMNIVOICE_PARITY=1 zig build test -Doptimize=ReleaseSafe

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const dump = @import("dump.zig");
const lm = @import("lm.zig");
const maskgit = @import("maskgit.zig");
const mg_decode = @import("mg_decode.zig");
const prompt = @import("prompt.zig");
const voicedesign = @import("voicedesign.zig");

// ---------------------------------------------------------------------------
// Fast unit tests (no model files)
// ---------------------------------------------------------------------------

test "extractToKTV repacks [T, K*V] rows into [K, T, V]" {
    // K=2, T=3, V=2; src[(t*K + k)*V + v] = 100*k + 10*t + v.
    var src: [12]f32 = undefined;
    for (0..3) |t| {
        for (0..2) |k| {
            for (0..2) |v| {
                src[(t * 2 + k) * 2 + v] = @floatFromInt(100 * k + 10 * t + v);
            }
        }
    }
    var dst: [12]f32 = undefined;
    mg_decode.extractToKTV(&src, 2, 3, 2, &dst);
    for (0..2) |k| {
        for (0..3) |t| {
            for (0..2) |v| {
                try std.testing.expectEqual(
                    @as(f32, @floatFromInt(100 * k + 10 * t + v)),
                    dst[(k * 3 + t) * 2 + v],
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers for the env-gated parity gates
// ---------------------------------------------------------------------------

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

/// Raw [K, T] i32 row-major, NO header (the reference --maskgit-test output).
fn readRawTokens(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expect_len: usize) ![]i32 {
    const bytes = try readFileBytes(allocator, io, path);
    defer allocator.free(bytes);
    if (bytes.len != expect_len * 4) return error.CorruptGolden;
    const values = try allocator.alloc(i32, expect_len);
    errdefer allocator.free(values);
    for (values, 0..) |*dst, i| {
        dst.* = std.mem.readInt(i32, bytes[i * 4 ..][0..4], .little);
    }
    return values;
}

fn countExact(got: []const i32, want: []const i32) usize {
    var exact: usize = 0;
    for (got, want) |g, w| {
        if (g == w) exact += 1;
    }
    return exact;
}

fn executedSteps(cfg: maskgit.Config, total: usize) !usize {
    const allocator = std.testing.allocator;
    const ts = try allocator.alloc(f32, cfg.num_step + 1);
    defer allocator.free(ts);
    maskgit.timesteps(cfg.t_shift, cfg.num_step, ts);
    const sched = try allocator.alloc(i32, cfg.num_step);
    defer allocator.free(sched);
    maskgit.schedule(total, ts, sched);
    var executed: usize = 0;
    for (sched) |n| {
        if (n > 0) executed += 1;
    }
    return executed;
}

const RunResult = struct {
    tokens: []i32,
    ctr_lo: u32,
    wall_ns: i96,
};

/// Load the base GGUF at `model_path`, build the prompt, run generate.
fn runGenerate(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
    instruct: []const u8,
    target_len: usize,
    cfg: maskgit.Config,
    dumps: ?mg_decode.DumpSink,
) !?RunResult {
    var file = fucina.gguf.File.loadMmap(allocator, io, model_path) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer file.deinit();

    const config = try lm.Config.fromGguf(&file);
    var tok = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tok.deinit();

    var built = try prompt.build(allocator, &tok, &config, .{
        .text = "The quick brown fox jumps over the lazy dog.",
        .lang = "English",
        .instruct = instruct,
        .num_target_tokens = target_len,
        .denoise = true, // no ref => no <|denoise|> token (reference default)
    });
    defer built.deinit();

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try lm.loadModel(&ctx, &file);
    defer model.deinit();

    var ctr_lo: u32 = 0;
    const t0 = std.Io.Clock.awake.now(io).nanoseconds;
    const tokens = try mg_decode.generate(allocator, &ctx, &model, &built, cfg, target_len, &ctr_lo, dumps, null);
    const wall_ns = std.Io.Clock.awake.now(io).nanoseconds - t0;
    return .{ .tokens = tokens, .ctr_lo = ctr_lo, .wall_ns = wall_ns };
}

// ---------------------------------------------------------------------------
// Gate 1: greedy decode, token-exact vs goldens/maskgit (F32 + Q8_0)
// ---------------------------------------------------------------------------

test "OMNIVOICE_PARITY: greedy MaskGIT tokens exact vs reference (F32, Q8_0)" {
    if (std.c.getenv("OMNIVOICE_PARITY") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const target_len = 75; // --duration 3 at the hardcoded 25 fps
    const num_k = 8;

    // The reference --maskgit-test forces both temperatures to zero; every
    // other knob keeps its default. Greedy performs no Philox calls.
    const cfg = maskgit.Config{ .class_temperature = 0.0, .position_temperature = 0.0, .seed = 42 };

    const dtypes = [_][]const u8{ "F32", "Q8_0" };
    var ran: usize = 0;
    for (dtypes) |dtype_name| {
        var model_path_buf: [128]u8 = undefined;
        const model_path = try std.fmt.bufPrint(&model_path_buf, "models/omnivoice/omnivoice-base-{s}.gguf", .{dtype_name});
        var golden_path_buf: [128]u8 = undefined;
        const golden_path = try std.fmt.bufPrint(&golden_path_buf, "refs/omnivoice-research/goldens/maskgit/tokens-{s}.bin", .{dtype_name});

        const result = (try runGenerate(allocator, io, model_path, "", target_len, cfg, null)) orelse {
            std.debug.print("omnivoice mg greedy {s}: model file missing, skipped\n", .{dtype_name});
            continue;
        };
        defer allocator.free(result.tokens);

        const want = try readRawTokens(allocator, io, golden_path, num_k * target_len);
        defer allocator.free(want);

        const exact = countExact(result.tokens, want);
        std.debug.print(
            "omnivoice mg greedy {s}: {d}/{d} tokens exact, ctr_lo={d}, wall={d:.1}s\n",
            .{ dtype_name, exact, want.len, result.ctr_lo, @as(f64, @floatFromInt(result.wall_ns)) / 1e9 },
        );
        try std.testing.expectEqual(@as(u32, 0), result.ctr_lo); // greedy: zero Philox calls

        // Every slot must be demasked and in-vocab regardless of dtype.
        for (result.tokens) |token| {
            try std.testing.expect(token >= 0 and token < 1024);
        }

        if (std.mem.eql(u8, dtype_name, "F32")) {
            // F32 is the token-exact oracle: the whole prompt+LM+CFG+loop
            // chain must reproduce the reference bit-for-bit at the decision
            // level (measured: 600/600).
            try std.testing.expectEqual(want.len, exact);
        } else {
            // Q8_0 CANNOT be token-exact across implementations: ggml
            // quantizes ACTIVATIONS to q8_0 per GEMM, so any sub-ulp
            // difference in upstream f32 ops (GEMM reduction order, exp
            // implementations) can flip an int8 rounding boundary, injecting
            // ~d-sized jumps that compound over 28 layers and 32 feedback
            // steps. Measured on the stage-D golden input: our Q8_0 logits
            // sit at the SAME distance from the F32 truth as the reference's
            // own Q8_0 (mean_abs 0.141 vs 0.148, argmax-vs-F32 agreement
            // 157/192 for both) — two equally valid q8_0 pipelines — while
            // per-(s,k) argmax agreement between them is 183/192 at step 0,
            // which the greedy feedback loop then amplifies chaotically.
            // Token exactness would require bit-identical kernels; the
            // stage-D Q8_0 gate is the row-cosine logits parity in lm_tests.
            std.debug.print(
                "omnivoice mg greedy Q8_0: cross-implementation agreement is informational (quantization chaos), not asserted\n",
                .{},
            );
        }
        ran += 1;
    }
    if (ran == 0) return error.SkipZigTest;
}

// ---------------------------------------------------------------------------
// Gate 2: seeded chain (defaults, seed 42) vs goldens/tts-design
// ---------------------------------------------------------------------------

test "OMNIVOICE_PARITY: seeded MaskGIT chain token-exact vs tts-design goldens" {
    if (std.c.getenv("OMNIVOICE_PARITY") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const goldens_dir = "refs/omnivoice-research/goldens/tts-design";
    const dump_dir = "/tmp/fucina-omnivoice-mg-decode-test";
    const target_len = 65;
    const num_k = 8;

    // Reference defaults: 32 steps, guidance 2.0, t_shift 0.1, layer penalty
    // 5.0, position temperature 5.0, class temperature 0, seed 42; ctr_lo 0.
    const cfg = maskgit.Config{};

    std.Io.Dir.cwd().createDirPath(io, dump_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    const normalized = try voicedesign.normalize(allocator, "male, young adult, moderate pitch", false);
    defer normalized.deinit(allocator);
    const instruct = switch (normalized) {
        .ok => |s| s,
        .invalid => return error.TestUnexpectedResult,
    };

    const dumps = mg_decode.DumpSink{ .io = io, .dir = dump_dir };
    const result = (try runGenerate(
        allocator,
        io,
        "models/omnivoice/omnivoice-base-F32.gguf",
        instruct,
        target_len,
        cfg,
        dumps,
    )) orelse return error.SkipZigTest;
    defer allocator.free(result.tokens);

    // Philox accounting: one uniform kernel per EXECUTED step (pos_temp > 0,
    // class_temp == 0).
    const expected_ctr: u32 = @intCast(try executedSteps(cfg, num_k * target_len));
    try std.testing.expectEqual(expected_ctr, result.ctr_lo);

    // Final tokens: 100% exact (the reference is seed-deterministic and
    // step-0 FP flips resorb over the 32 steps).
    const tokens_golden = try dump.readFile(allocator, io, goldens_dir ++ "/mg-tokens.bin");
    defer {
        allocator.free(tokens_golden.shape);
        allocator.free(tokens_golden.data);
    }
    try std.testing.expectEqualSlices(i32, &.{ num_k, target_len }, tokens_golden.shape);
    var tokens_exact: usize = 0;
    for (tokens_golden.data, result.tokens) |want, got| {
        if (want == @as(f32, @floatFromInt(got))) tokens_exact += 1;
    }

    // Step-0 dumps: pred tokens >= 99% exact (FP-epsilon argmax ties allowed),
    // confidence scores cosine >= 0.9999.
    const pred_ours = try dump.readFile(allocator, io, dump_dir ++ "/mg-pred-tokens-step0.bin");
    defer {
        allocator.free(pred_ours.shape);
        allocator.free(pred_ours.data);
    }
    const pred_golden = try dump.readFile(allocator, io, goldens_dir ++ "/mg-pred-tokens-step0.bin");
    defer {
        allocator.free(pred_golden.shape);
        allocator.free(pred_golden.data);
    }
    try std.testing.expectEqualSlices(i32, pred_golden.shape, pred_ours.shape);
    var pred_exact: usize = 0;
    for (pred_golden.data, pred_ours.data) |want, got| {
        if (want == got) pred_exact += 1;
    }

    const scores_ours = try dump.readFile(allocator, io, dump_dir ++ "/mg-scores-step0.bin");
    defer {
        allocator.free(scores_ours.shape);
        allocator.free(scores_ours.data);
    }
    const scores_golden = try dump.readFile(allocator, io, goldens_dir ++ "/mg-scores-step0.bin");
    defer {
        allocator.free(scores_golden.shape);
        allocator.free(scores_golden.data);
    }
    const scores_stats = dump.compare(scores_ours.data, scores_golden.data);

    std.debug.print(
        "omnivoice mg seeded F32: tokens {d}/{d} exact, step0 pred {d}/{d} exact, step0 scores cos={d:.7} max_abs={d:.6}, ctr_lo={d}, wall={d:.1}s\n",
        .{
            tokens_exact,        tokens_golden.data.len,
            pred_exact,          pred_golden.data.len,
            scores_stats.cosine, scores_stats.max_abs_diff,
            result.ctr_lo,       @as(f64, @floatFromInt(result.wall_ns)) / 1e9,
        },
    );

    try std.testing.expectEqual(tokens_golden.data.len, tokens_exact);
    try std.testing.expect(@as(f64, @floatFromInt(pred_exact)) >= 0.99 * @as(f64, @floatFromInt(pred_golden.data.len)));
    try std.testing.expect(scores_stats.cosine >= 0.9999);
}
