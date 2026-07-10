//! Tests for prompt.zig — combineText normalisation, buffer-fill offsets,
//! and (env-gated) token parity of the built prompt against the reference
//! goldens (refs/omnivoice-research/goldens/tts-design/prompt-*-ids.bin).
//!
//! The parity test needs models/omnivoice/omnivoice-base-*.gguf (tokenizer
//! metadata is identical across the four base dtypes) and is gated behind
//! OMNIVOICE_PARITY like lm_tests.zig:
//!
//!   OMNIVOICE_PARITY=1 zig build test [-Doptimize=ReleaseSafe]

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const dump = @import("dump.zig");
const lm = @import("lm.zig");
const prompt = @import("prompt.zig");
const voicedesign = @import("voicedesign.zig");

// ---------------------------------------------------------------------------
// combineText (fast, no tokenizer)
// ---------------------------------------------------------------------------

fn expectCombined(expected: []const u8, text: []const u8, ref_text: []const u8) !void {
    const got = try prompt.combineText(std.testing.allocator, text, ref_text);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(expected, got);
}

test "combineText strips space/tab and joins ref + text" {
    try expectCombined("hello world", "  hello world \t", "");
    try expectCombined("ref text hello", " hello", " ref text\t");
}

test "combineText drops CR/LF and maps full-width parens" {
    try expectCombined("ab", "a\r\nb", "");
    try expectCombined("(x)", "\xEF\xBC\x88x\xEF\xBC\x89", ""); // U+FF08 x U+FF09
    // The ref+text join space is inserted BEFORE the newline drop (strip only
    // removes spaces/tabs), so the trailing "\n" vanishes but the joining
    // space survives.
    try expectCombined("ref ab", "ab", "ref\n");
}

test "combineText collapses space/tab runs into one space" {
    try expectCombined("a b c", "a  \t b \t\tc", "");
}

test "combineText drops spaces adjacent to CJK ideographs" {
    try expectCombined("\xE4\xBD\xA0\xE5\xA5\xBD", "\xE4\xBD\xA0 \xE5\xA5\xBD", ""); // 你 好 -> 你好
    try expectCombined("hi\xE4\xBD\xA0", "hi \xE4\xBD\xA0", ""); // hi 你 -> hi你
    try expectCombined("\xE4\xBD\xA0hi", "\xE4\xBD\xA0 hi", ""); // 你 hi -> 你hi
    try expectCombined("a b", "a b", ""); // non-CJK spaces survive
}

test "combineText skips malformed UTF-8 lead bytes one byte at a time" {
    try expectCombined("ab", "a\xFFb", "");
}

// ---------------------------------------------------------------------------
// fillFromIds buffer offsets (fast, no tokenizer)
// ---------------------------------------------------------------------------

test "fillFromIds lays out cond/uncond rows and audio masks (no ref)" {
    const style = [_]i32{ 1, 2 };
    const text = [_]i32{ 3, 4, 5 };
    var built = try prompt.fillFromIds(std.testing.allocator, 2, 9, &style, &text, null, 0, 4);
    defer built.deinit();

    try std.testing.expectEqual(@as(usize, 9), built.c_len);
    try std.testing.expectEqual(@as(usize, 4), built.u_len);
    try std.testing.expectEqual(@as(usize, 9), built.s_max);

    // Cond rows: style + text duplicated across k, target window = mask.
    const expected_cond = [_]i32{ 1, 2, 3, 4, 5, 9, 9, 9, 9 };
    try std.testing.expectEqualSlices(i32, &expected_cond, built.condIds()[0..9]);
    try std.testing.expectEqualSlices(i32, &expected_cond, built.condIds()[9..18]);

    // Uncond rows: cond tail copy (all mask at build) + mask padding.
    const expected_uncond = [_]i32{ 9, 9, 9, 9, 9, 9, 9, 9, 9 };
    try std.testing.expectEqualSlices(i32, &expected_uncond, built.uncondIds()[0..9]);

    // audio_mask: cond 1 on [N1+N2, c_len), uncond 1 on [0, u_len).
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0, 0, 1, 1, 1, 1 }, built.condAudioMask());
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 1, 1, 0, 0, 0, 0, 0 }, built.uncondAudioMask());
}

test "fillFromIds places per-codebook ref tokens on the cond rows" {
    const style = [_]i32{7};
    const text = [_]i32{8};
    const ref = [_]i32{ 10, 11, 20, 21 }; // [K=2, ref_len=2], k slow
    var built = try prompt.fillFromIds(std.testing.allocator, 2, 9, &style, &text, &ref, 2, 3);
    defer built.deinit();

    try std.testing.expectEqual(@as(usize, 7), built.c_len);
    // k=0: style text ref0 ref1 mask mask mask
    try std.testing.expectEqualSlices(i32, &.{ 7, 8, 10, 11, 9, 9, 9 }, built.condIds()[0..7]);
    // k=1 carries the k=1 ref codes.
    try std.testing.expectEqualSlices(i32, &.{ 7, 8, 20, 21, 9, 9, 9 }, built.condIds()[7..14]);
    // audio_mask covers ref + target on the cond row.
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 1, 1, 1, 1, 1 }, built.condAudioMask());
}

// ---------------------------------------------------------------------------
// Parity gate (env-gated): built prompt ids vs reference goldens
// ---------------------------------------------------------------------------

const goldens_dir = "refs/omnivoice-research/goldens/tts-design";
const model_candidates = [_][]const u8{
    "models/omnivoice/omnivoice-base-F32.gguf",
    "models/omnivoice/omnivoice-base-Q8_0.gguf",
    "models/omnivoice/omnivoice-base-BF16.gguf",
    "models/omnivoice/omnivoice-base-Q4_K_M.gguf",
};

fn openAnyBaseGguf(allocator: std.mem.Allocator, io: std.Io) !?fucina.gguf.File {
    for (model_candidates) |path| {
        const file = fucina.gguf.File.loadMmap(allocator, io, path) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
        return file;
    }
    return null;
}

test "OMNIVOICE_PARITY: prompt ids match the tts-design goldens exactly" {
    if (std.c.getenv("OMNIVOICE_PARITY") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var file = (try openAnyBaseGguf(allocator, io)) orelse return error.SkipZigTest;
    defer file.deinit();

    const config = try lm.Config.fromGguf(&file);
    var tok = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tok.deinit();

    // The voice-design reference run: instruct resolved through
    // voice_design_normalize (use_zh = false, no CJK in the text).
    const normalized = try voicedesign.normalize(allocator, "male, young adult, moderate pitch", false);
    defer normalized.deinit(allocator);
    const instruct = switch (normalized) {
        .ok => |s| s,
        .invalid => return error.TestUnexpectedResult,
    };

    var built = try prompt.build(allocator, &tok, &config, .{
        .text = "The quick brown fox jumps over the lazy dog.",
        .lang = "English",
        .instruct = instruct,
        .num_target_tokens = 65,
        .denoise = true, // no ref => no <|denoise|> token
    });
    defer built.deinit();

    // Reference log: N1=12 N2=12 Sref=0 Stgt=65 c_len=89 u_len=65.
    try std.testing.expectEqual(@as(usize, 89), built.c_len);
    try std.testing.expectEqual(@as(usize, 65), built.u_len);

    // Golden rows are k=0 of each batch row, ids cast to f32.
    const cond_golden = try dump.readFile(allocator, io, goldens_dir ++ "/prompt-cond-ids.bin");
    defer {
        allocator.free(cond_golden.shape);
        allocator.free(cond_golden.data);
    }
    const uncond_golden = try dump.readFile(allocator, io, goldens_dir ++ "/prompt-uncond-ids.bin");
    defer {
        allocator.free(uncond_golden.shape);
        allocator.free(uncond_golden.data);
    }
    try std.testing.expectEqualSlices(i32, &.{89}, cond_golden.shape);
    try std.testing.expectEqualSlices(i32, &.{89}, uncond_golden.shape);

    const cond_row = built.condIds()[0..built.s_max];
    const uncond_row = built.uncondIds()[0..built.s_max];
    for (cond_golden.data, cond_row) |want, got| {
        try std.testing.expectEqual(want, @as(f32, @floatFromInt(got)));
    }
    for (uncond_golden.data, uncond_row) |want, got| {
        try std.testing.expectEqual(want, @as(f32, @floatFromInt(got)));
    }

    // Style/text ids are duplicated across all K codebooks.
    for (1..built.num_codebooks) |k| {
        try std.testing.expectEqualSlices(i32, cond_row, built.condIds()[k * built.s_max ..][0..built.s_max]);
    }
    std.debug.print("omnivoice prompt parity: cond+uncond ids exact ({d} positions)\n", .{2 * built.s_max});
}
