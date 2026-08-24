const std = @import("std");
const tok = @import("tokenizer.zig");

test "detokenize: ▁ -> space, concat, strip leading space" {
    const allocator = std.testing.allocator;
    // ▁ = "\xe2\x96\x81". pieces: 0=<unk>, 1=▁Hello, 2=▁world, 3=s.
    const pieces = [_][]const u8{ "<unk>", "\xe2\x96\x81Hello", "\xe2\x96\x81world", "s" };
    const out = try tok.detokenize(allocator, &pieces, &[_]i32{ 1, 2, 3 });
    defer allocator.free(out);
    try std.testing.expectEqualStrings("Hello worlds", out);
}

test "detokenize: subword pieces glue without ▁; out-of-range ids skipped" {
    const allocator = std.testing.allocator;
    const pieces = [_][]const u8{ "\xe2\x96\x81the", "re", "\xe2\x96\x81old" };
    // 0=▁the, 1=re, 2=▁old ; ids include an out-of-range id (99) that is skipped.
    const out = try tok.detokenize(allocator, &pieces, &[_]i32{ 0, 1, 99, 2 });
    defer allocator.free(out);
    try std.testing.expectEqualStrings("there old", out);
}

test "detokenize: empty ids -> empty string" {
    const allocator = std.testing.allocator;
    const pieces = [_][]const u8{"\xe2\x96\x81x"};
    const out = try tok.detokenize(allocator, &pieces, &[_]i32{});
    defer allocator.free(out);
    try std.testing.expectEqual(@as(usize, 0), out.len);
}
