//! Behavioral tests for the generated Unicode category tables
//! (`unicode_categories.zig`): spot checks of the public \p{L} / \p{N} / \s
//! classification predicates against known code points.
const std = @import("std");
const unicode_categories = @import("unicode_categories.zig");

const isLetter = unicode_categories.isLetter;
const isNumber = unicode_categories.isNumber;
const isWhitespace = unicode_categories.isWhitespace;

test "spot checks against known categories" {
    try std.testing.expect(isLetter('a') and isLetter('Z'));
    try std.testing.expect(isLetter(0x00E9)); // é
    try std.testing.expect(isLetter(0x4E2D)); // 中
    try std.testing.expect(!isLetter('1') and !isLetter(' ') and !isLetter(0x1F600));
    try std.testing.expect(isNumber('7'));
    try std.testing.expect(isNumber(0x0661)); // ١ arabic-indic one
    try std.testing.expect(isNumber(0x00B2)); // ² superscript two (\p{No})
    try std.testing.expect(!isNumber('x'));
    try std.testing.expect(isWhitespace(' ') and isWhitespace('\t') and isWhitespace('\n') and isWhitespace('\r'));
    try std.testing.expect(isWhitespace(0x00A0) and isWhitespace(0x3000));
    try std.testing.expect(!isWhitespace('a') and !isWhitespace(0x200B)); // ZWSP is \p{Cf}, not \s
}
