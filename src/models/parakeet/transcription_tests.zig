const std = @import("std");
const tr = @import("transcription.zig");
const TokenInfo = @import("decoder.zig").TokenInfo;

const W = "\xe2\x96\x81"; // ▁ (SentencePiece meta-space)

test "groupWords: word + punctuation-close + min-conf + tail" {
    const allocator = std.testing.allocator;
    // ids: 0=<blk> 1=▁Well 2=, 3=▁I
    const pieces = [_][]const u8{ "<blk>", W ++ "Well", ",", W ++ "I" };
    const toks = [_]TokenInfo{
        .{ .id = 1, .frame = 6, .conf = 0.79, .span = 2 }, // ▁Well, frame 6, end 8
        .{ .id = 2, .frame = 8, .conf = 0.90, .span = 1 }, // "," (punct, refined to prev end)
        .{ .id = 3, .frame = 10, .conf = 1.00, .span = 1 }, // ▁I
    };
    const words = try tr.groupWords(allocator, &toks, &pieces, 0.08);
    defer tr.freeWords(allocator, words);

    try std.testing.expectEqual(@as(usize, 2), words.len);
    try std.testing.expectEqualStrings("Well,", words[0].text);
    try std.testing.expectApproxEqAbs(@as(f32, 0.48), words[0].start, 1e-5); // frame 6
    try std.testing.expectApproxEqAbs(@as(f32, 0.64), words[0].end, 1e-5); // refined to frame 8
    try std.testing.expectApproxEqAbs(@as(f32, 0.79), words[0].conf, 1e-5); // min(0.79,0.90)
    try std.testing.expectEqualStrings("I", words[1].text);
    try std.testing.expectApproxEqAbs(@as(f32, 0.80), words[1].start, 1e-5); // frame 10
    try std.testing.expectApproxEqAbs(@as(f32, 0.88), words[1].end, 1e-5); // frame 11
    try std.testing.expectApproxEqAbs(@as(f32, 1.00), words[1].conf, 1e-5);
}

test "toJson: schema + float precision + escaping" {
    const allocator = std.testing.allocator;
    const toks = [_]TokenInfo{.{ .id = 5, .frame = 10, .conf = 0.5, .span = 1 }};
    const words = [_]tr.Word{.{ .text = @constCast("Hi"), .start = 0.8, .end = 0.88, .conf = 0.5 }};
    const j = try tr.toJson(allocator, "Hi", 0.08, &toks, &words);
    defer allocator.free(j);
    try std.testing.expectEqualStrings(
        "{\"text\":\"Hi\",\"frame_sec\":0.080000,\"words\":[{\"w\":\"Hi\",\"start\":0.800,\"end\":0.880,\"conf\":0.5000}],\"tokens\":[{\"id\":5,\"t\":0.800,\"conf\":0.5000}]}",
        j,
    );
    // escaping: quote in the text
    const j2 = try tr.toJson(allocator, "a\"b", 0.08, &.{}, &.{});
    defer allocator.free(j2);
    try std.testing.expectEqualStrings("{\"text\":\"a\\\"b\",\"frame_sec\":0.080000,\"words\":[],\"tokens\":[]}", j2);
}

test "groupWords: empty input" {
    const allocator = std.testing.allocator;
    const pieces = [_][]const u8{ "<blk>", W ++ "x" };
    const words = try tr.groupWords(allocator, &.{}, &pieces, 0.08);
    defer tr.freeWords(allocator, words);
    try std.testing.expectEqual(@as(usize, 0), words.len);
}

test "groupWords: multi-subword single word (pure continuation)" {
    const allocator = std.testing.allocator;
    // "turning" = ▁turn + ing  (continuation sub-word, no new word, no punct)
    const pieces = [_][]const u8{ "<blk>", W ++ "turn", "ing" };
    const toks = [_]TokenInfo{
        .{ .id = 1, .frame = 43, .conf = 0.99, .span = 1 },
        .{ .id = 2, .frame = 44, .conf = 0.95, .span = 3 }, // end 44+3=47
    };
    const words = try tr.groupWords(allocator, &toks, &pieces, 0.08);
    defer tr.freeWords(allocator, words);
    try std.testing.expectEqual(@as(usize, 1), words.len);
    try std.testing.expectEqualStrings("turning", words[0].text);
    try std.testing.expectApproxEqAbs(@as(f32, 3.44), words[0].start, 1e-5); // frame 43
    try std.testing.expectApproxEqAbs(@as(f32, 3.76), words[0].end, 1e-5); // frame 47
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), words[0].conf, 1e-5); // min
}
