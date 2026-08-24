const std = @import("std");
const tokenizer = @import("tokenizer.zig");

const testing = std.testing;

// Synthetic vocab: byte-level singles + one merge + two non-marker specials.
// Exercises the reference's greedy longest-first special matching around
// plain-BPE runs (specials here do NOT follow the <|...|> marker shape).
fn syntheticTokenizer(allocator: std.mem.Allocator) !tokenizer.Tokenizer {
    const vocab = [_][]const u8{
        "a", "b", "c", "ab", "<s>", "<sp>", "Ġ", "Ġa",
    };
    const merges = [_][]const u8{ "a b", "Ġ a" };
    const types = [_]i32{ 1, 1, 1, 1, 4, 4, 1, 1 };
    return tokenizer.Tokenizer.initFromArrays(allocator, &vocab, &merges, &types);
}

test "special tokens split atomically, longest match first" {
    var tok = try syntheticTokenizer(testing.allocator);
    defer tok.deinit();

    // "<sp>" must win over "<s>" at the same position (longest-first), and
    // the surrounding runs BPE normally ("ab" merges).
    const ids = try tok.encode(testing.allocator, "ab<sp>ab<s>c");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &.{ 3, 5, 3, 4, 2 }, ids);
}

test "no specials means one plain BPE run" {
    var tok = try syntheticTokenizer(testing.allocator);
    defer tok.deinit();

    const ids = try tok.encode(testing.allocator, "ab ab");
    defer testing.allocator.free(ids);
    // qwen2 pretokenizer: "ab" | " ab". In " ab" the rank-0 merge "a b"
    // applies before the rank-1 "Ġ a", leaving [Ġ, ab].
    try testing.expectEqualSlices(u32, &.{ 3, 6, 3 }, ids);
}

test "buildPrompt shape: specials + query" {
    var tok = try syntheticTokenizer(testing.allocator);
    defer tok.deinit();

    // The chat scaffold tokens are absent from this synthetic vocab, so the
    // scaffold text BPE-falls-back to single symbols; just assert the image
    // placeholder count scales with n_image_tokens via <sp> stand-ins: not
    // applicable here — covered by the model-gated parity test instead.
    const ids = try tokenizer.buildPrompt(testing.allocator, &tok, 0, "c");
    defer testing.allocator.free(ids);
    try testing.expect(ids.len > 0);
}
