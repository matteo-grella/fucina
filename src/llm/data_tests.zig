//! Behavioral tests for the SFT dataset/dataloader helpers (`data.zig`):
//! prompt-mask/next-token-shift correctness on a tiny byte-BPE fixture,
//! truncation edges (seq_max hit, SampleTooLong), JSONL loading (happy path,
//! custom keys, escaped strings, malformed lines), and the `Loader` order
//! contract — sequential == round-robin, shuffled = valid permutation,
//! GOLDEN-PINNED (seed, epoch) → permutation (the checkpoint contract), and
//! mid-epoch `restore(state())` replay.

const std = @import("std");
const data_mod = @import("data.zig");
const chat_mod = @import("chat.zig");
const tok_mod = @import("tokenizer.zig");

const Tokenizer = tok_mod.Tokenizer;
const Pair = data_mod.Pair;
const SftText = data_mod.SftText;
const EncodeOptions = data_mod.EncodeOptions;
const Loader = data_mod.Loader;

// ---------------------------------------------------------------------------
// encodePair / encodePrompt
// ---------------------------------------------------------------------------

/// Merge-free byte-BPE fixture: the ChatML markers resolve as special tokens
/// and every other character of the rendered scaffolding ("user",
/// "assistant", the think block, newlines as the byte-level "Ċ") is a
/// single-char vocab entry, so chunks encode 1 token per character.
fn makeTokenizer(allocator: std.mem.Allocator) !Tokenizer {
    const vocab = [_][]const u8{
        "<|im_start|>", "<|im_end|>",
        "u",            "s",
        "e",            "r",
        "a",            "n",
        "t",            "i",
        "h",            "k",
        "y",            "o",
        "m",            "<",
        ">",            "/",
        "\xC4\x8A", // 'Ċ' = byte-level '\n'
    };
    const merges = [_][]const u8{};
    return Tokenizer.initFromParts(allocator, &vocab, &merges, .{});
}

const chatml = chat_mod.Template{ .format = .chatml };
const fixture_pair = Pair{ .instruction = "hi", .response = "yo" };
/// A sentinel distinct from the default proves the option is plumbed through.
const test_ignore: usize = 9999;

test "encodePair masks the prompt and shifts labels by one" {
    const allocator = std.testing.allocator;
    var tok = try makeTokenizer(allocator);
    defer tok.deinit();

    const opts = EncodeOptions{ .seq_max = 64, .ignore_index = test_ignore };
    var sample = try data_mod.encodePair(allocator, &tok, chatml, fixture_pair, opts);
    defer sample.deinit(allocator);

    // Reference pieces, encoded exactly as the contract states: the prompt is
    // the rendered turn; the response (+ stop marker) is encoded SEPARATELY.
    const prompt = try data_mod.encodePrompt(allocator, &tok, chatml, fixture_pair.instruction, opts);
    defer allocator.free(prompt);
    const response = try tok.encodeRaw(allocator, "yo<|im_end|>");
    defer allocator.free(response);
    try std.testing.expectEqualSlices(u32, &.{ 12, 13, 1 }, response); // y, o, <|im_end|>

    const total = prompt.len + response.len;
    try std.testing.expectEqual(total - 1, sample.inputs.len);
    try std.testing.expectEqual(total - 1, sample.labels.len);

    // inputs = prompt ++ response minus the final token.
    try std.testing.expectEqualSlices(usize, prompt, sample.inputs[0..prompt.len]);
    for (sample.inputs[prompt.len..], response[0 .. response.len - 1]) |got, want| {
        try std.testing.expectEqual(@as(usize, want), got);
    }

    // labels: position i-1 predicts token i — every prompt position masked,
    // every response token (incl. the stop marker) supervised.
    for (sample.labels[0 .. prompt.len - 1]) |label| {
        try std.testing.expectEqual(test_ignore, label);
    }
    for (sample.labels[prompt.len - 1 ..], response) |got, want| {
        try std.testing.expectEqual(@as(usize, want), got);
    }
    try std.testing.expectEqual(@as(usize, 1), sample.labels[sample.labels.len - 1]); // <|im_end|>
}

test "encodePair without prompt masking supervises the full shift" {
    const allocator = std.testing.allocator;
    var tok = try makeTokenizer(allocator);
    defer tok.deinit();

    const opts = EncodeOptions{ .seq_max = 64, .ignore_index = test_ignore, .mask_prompt = false };
    var sample = try data_mod.encodePair(allocator, &tok, chatml, fixture_pair, opts);
    defer sample.deinit(allocator);

    // labels[j] == sequence[j + 1] everywhere: labels[j] == inputs[j + 1] for
    // all but the last position, and no position carries the sentinel.
    for (sample.labels[0 .. sample.labels.len - 1], sample.inputs[1..]) |label, input| {
        try std.testing.expectEqual(input, label);
    }
    for (sample.labels) |label| try std.testing.expect(label != test_ignore);
}

test "encodePrompt honors the system option" {
    const allocator = std.testing.allocator;
    var tok = try makeTokenizer(allocator);
    defer tok.deinit();

    const bare = try data_mod.encodePrompt(allocator, &tok, chatml, "hi", .{});
    defer allocator.free(bare);
    const with_system = try data_mod.encodePrompt(allocator, &tok, chatml, "hi", .{ .system = "ok" });
    defer allocator.free(with_system);
    try std.testing.expect(with_system.len > bare.len);
}

test "encodePair truncates to seq_max inputs" {
    const allocator = std.testing.allocator;
    var tok = try makeTokenizer(allocator);
    defer tok.deinit();

    const prompt = try data_mod.encodePrompt(allocator, &tok, chatml, fixture_pair.instruction, .{});
    defer allocator.free(prompt);

    var full = try data_mod.encodePair(allocator, &tok, chatml, fixture_pair, .{
        .seq_max = 64,
        .ignore_index = test_ignore,
    });
    defer full.deinit(allocator);

    // seq_max = prompt.len + 1 keeps two of the three response tokens:
    // total = seq_max + 1 sequence tokens = seq_max inputs/labels.
    const seq_max = prompt.len + 1;
    var truncated = try data_mod.encodePair(allocator, &tok, chatml, fixture_pair, .{
        .seq_max = seq_max,
        .ignore_index = test_ignore,
    });
    defer truncated.deinit(allocator);

    try std.testing.expectEqual(seq_max, truncated.inputs.len);
    try std.testing.expectEqual(seq_max, truncated.labels.len);
    // The truncated sample is a prefix of the untruncated one.
    try std.testing.expectEqualSlices(usize, full.inputs[0..seq_max], truncated.inputs);
    try std.testing.expectEqualSlices(usize, full.labels[0..seq_max], truncated.labels);

    // seq_max == prompt.len is the tightest legal window: exactly one
    // supervised token (the first response token).
    var tight = try data_mod.encodePair(allocator, &tok, chatml, fixture_pair, .{
        .seq_max = prompt.len,
        .ignore_index = test_ignore,
    });
    defer tight.deinit(allocator);
    try std.testing.expectEqual(prompt.len, tight.labels.len);
    for (tight.labels[0 .. tight.labels.len - 1]) |label| {
        try std.testing.expectEqual(test_ignore, label);
    }
    try std.testing.expectEqual(@as(usize, 12), tight.labels[tight.labels.len - 1]); // 'y'
}

test "encodePair rejects windows with no supervised token" {
    const allocator = std.testing.allocator;
    var tok = try makeTokenizer(allocator);
    defer tok.deinit();

    const prompt = try data_mod.encodePrompt(allocator, &tok, chatml, fixture_pair.instruction, .{});
    defer allocator.free(prompt);

    // The prompt alone fills the window: no room for a supervised token.
    try std.testing.expectError(error.SampleTooLong, data_mod.encodePair(
        allocator,
        &tok,
        chatml,
        fixture_pair,
        .{ .seq_max = prompt.len - 1 },
    ));
    // Degenerate window (total < 2).
    try std.testing.expectError(error.SampleTooLong, data_mod.encodePair(
        allocator,
        &tok,
        chatml,
        fixture_pair,
        .{ .seq_max = 0 },
    ));
}

// ---------------------------------------------------------------------------
// SftText
// ---------------------------------------------------------------------------

fn writeTempFile(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
}

test "fromPairs borrows and deinit frees nothing" {
    const pairs = [_]Pair{
        .{ .instruction = "a", .response = "b" },
        .{ .instruction = "c", .response = "d" },
    };
    var text = SftText.fromPairs(&pairs);
    try std.testing.expectEqual(@as(usize, 2), text.pairs.len);
    try std.testing.expectEqual(&pairs[0], &text.pairs[0]); // zero-copy borrow
    try std.testing.expectEqual(@as(?[]u8, null), text.blob);
    text.deinit(std.testing.allocator); // no-op arm; leak checker proves it
}

test "fromJsonl loads pairs, skips blank lines, ignores extra keys" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "data_jsonl_test_{d}.jsonl", .{std.Io.Clock.real.now(io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    try writeTempFile(io, path,
        \\{"instruction": "What is two plus two?", "response": "Ahoy! Four, matey."}
        \\
        \\{"instruction": "Name a color.", "response": "Red.", "extra": [1, 2]}
        \\
    );
    var text = try SftText.fromJsonl(allocator, io, path, .{});
    defer text.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), text.pairs.len);
    try std.testing.expectEqualStrings("What is two plus two?", text.pairs[0].instruction);
    try std.testing.expectEqualStrings("Ahoy! Four, matey.", text.pairs[0].response);
    try std.testing.expectEqualStrings("Name a color.", text.pairs[1].instruction);
    try std.testing.expectEqualStrings("Red.", text.pairs[1].response);
    try std.testing.expect(text.blob != null);
}

test "fromJsonl honors custom keys and JSON escapes" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "data_jsonl_keys_test_{d}.jsonl", .{std.Io.Clock.real.now(io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    // Escaped strings are materialized in the std.json parse arena — the
    // pairs must survive it (duped into the owned blob).
    try writeTempFile(io, path,
        \\{"prompt": "line\nbreak \"quoted\"", "completion": "uniAcode"}
    );
    var text = try SftText.fromJsonl(allocator, io, path, .{
        .instruction_key = "prompt",
        .response_key = "completion",
    });
    defer text.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), text.pairs.len);
    try std.testing.expectEqualStrings("line\nbreak \"quoted\"", text.pairs[0].instruction);
    try std.testing.expectEqualStrings("uniAcode", text.pairs[0].response);
}

test "fromJsonl rejects malformed lines" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "data_jsonl_bad_test_{d}.jsonl", .{std.Io.Clock.real.now(io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    // Invalid JSON on line 2.
    try writeTempFile(io, path,
        \\{"instruction": "ok", "response": "ok"}
        \\not json at all
    );
    try std.testing.expectError(error.MalformedJsonl, SftText.fromJsonl(allocator, io, path, .{}));

    // Missing response key.
    try writeTempFile(io, path,
        \\{"instruction": "only"}
    );
    try std.testing.expectError(error.MalformedJsonl, SftText.fromJsonl(allocator, io, path, .{}));

    // Non-string value under a configured key.
    try writeTempFile(io, path,
        \\{"instruction": 5, "response": "y"}
    );
    try std.testing.expectError(error.MalformedJsonl, SftText.fromJsonl(allocator, io, path, .{}));

    // Non-object line.
    try writeTempFile(io, path,
        \\[1, 2, 3]
    );
    try std.testing.expectError(error.MalformedJsonl, SftText.fromJsonl(allocator, io, path, .{}));
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

test "sequential loader reproduces the round-robin order" {
    const allocator = std.testing.allocator;
    var loader = try Loader.init(allocator, 5, .sequential, 42);
    defer loader.deinit(allocator);
    for (0..12) |i| {
        try std.testing.expectEqual(i % 5, loader.next());
    }
    // Mid-epoch position after 12 draws over 5 samples.
    const s = loader.state();
    try std.testing.expectEqual(@as(u64, 2), s.epoch);
    try std.testing.expectEqual(@as(u64, 2), s.index);
}

test "shuffled loader emits a valid permutation per epoch" {
    const allocator = std.testing.allocator;
    var loader = try Loader.init(allocator, 7, .shuffled, 123);
    defer loader.deinit(allocator);
    var seen = [_]bool{false} ** 7;
    for (0..7) |_| {
        const idx = loader.next();
        try std.testing.expect(idx < 7);
        try std.testing.expect(!seen[idx]);
        seen[idx] = true;
    }
    try std.testing.expectEqual(@as(u64, 1), loader.state().epoch);
}

test "shuffled permutation is golden-pinned (checkpoint contract)" {
    // These values pin the documented formula — identity perm, Fisher–Yates
    // with j = splitmix64 % (i + 1) seeded by rng.at(seed, epoch). They may
    // NEVER change: checkpoints replay sample order from (seed, epoch).
    const allocator = std.testing.allocator;
    var loader = try Loader.init(allocator, 8, .shuffled, 42);
    defer loader.deinit(allocator);
    try std.testing.expectEqualSlices(usize, &.{ 3, 6, 0, 7, 1, 2, 5, 4 }, loader.perm);

    try loader.restore(.{ .seed = 42, .epoch = 1, .index = 0 });
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 4, 5, 7, 2, 1, 6 }, loader.perm);

    var other = try Loader.init(allocator, 5, .shuffled, 7);
    defer other.deinit(allocator);
    try other.restore(.{ .seed = 7, .epoch = 3, .index = 0 });
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 4, 0, 2 }, other.perm);
}

test "different epochs draw different permutations" {
    const allocator = std.testing.allocator;
    var loader = try Loader.init(allocator, 16, .shuffled, 99);
    defer loader.deinit(allocator);
    var epoch0: [16]usize = undefined;
    @memcpy(&epoch0, loader.perm);
    for (0..16) |_| _ = loader.next(); // wrap into epoch 1
    try std.testing.expectEqual(@as(u64, 1), loader.state().epoch);
    try std.testing.expect(!std.mem.eql(usize, &epoch0, loader.perm));

    // Same (seed, epoch) regenerates the identical permutation.
    var replay = try Loader.init(allocator, 16, .shuffled, 99);
    defer replay.deinit(allocator);
    try std.testing.expectEqualSlices(usize, &epoch0, replay.perm);
}

test "restore(state()) replays the identical remaining stream mid-epoch" {
    const allocator = std.testing.allocator;
    var a = try Loader.init(allocator, 7, .shuffled, 2026);
    defer a.deinit(allocator);
    for (0..3) |_| _ = a.next();
    const s = a.state();

    // Continue A across the epoch boundary (11 draws crosses into epoch 1).
    var expected: [11]usize = undefined;
    for (&expected) |*e| e.* = a.next();

    var b = try Loader.init(allocator, 7, .shuffled, 0); // seed overwritten by restore
    defer b.deinit(allocator);
    try b.restore(s);
    for (expected) |want| {
        try std.testing.expectEqual(want, b.next());
    }
}

test "loader rejects empty datasets and out-of-range restore" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.EmptyDataset, Loader.init(allocator, 0, .sequential, 1));

    var loader = try Loader.init(allocator, 7, .shuffled, 1);
    defer loader.deinit(allocator);
    try std.testing.expectError(error.InvalidLoaderState, loader.restore(.{ .seed = 1, .epoch = 0, .index = 7 }));
}
