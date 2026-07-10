#!/usr/bin/env python3
"""Generate src/llm/unicode_categories.zig from llama.cpp's unicode tables.

Extracts the \\p{L} (LETTER) and \\p{N} (NUMBER) codepoint ranges and the \\s
whitespace set from refs/llama.cpp/src/unicode-data.cpp (itself generated from
the Unicode database by llama.cpp's scripts/gen-unicode-data.py), so Fucina's
pretokenizer classifies codepoints EXACTLY like llama.cpp's tokenizer — the
parity oracle for token-ID-exact encoding.

Usage (from the repo root):
    python3 tools/gen_unicode_categories.py > src/llm/unicode_categories.zig
"""

import re
import sys

SRC = "refs/llama.cpp/src/unicode-data.cpp"

LETTER = 0x0004  # unicode_cpt_flags::LETTER
NUMBER = 0x0002  # unicode_cpt_flags::NUMBER


def main() -> None:
    src = open(SRC).read()

    m = re.search(r"unicode_ranges_flags = \{(.*?)\};", src, re.S)
    rng = [
        (int(a, 16), int(b, 16))
        for a, b in re.findall(r"\{0x([0-9A-Fa-f]+), 0x([0-9A-Fa-f]+)\}", m.group(1))
    ]
    assert rng[0][0] == 0 and rng[-1][0] == 0x110000, "unexpected table sentinel"

    def merged(bit):
        out = []
        for i in range(len(rng) - 1):
            start, flags = rng[i]
            end = rng[i + 1][0]  # exclusive
            if flags & bit:
                if out and out[-1][1] == start:
                    out[-1] = (out[-1][0], end)
                else:
                    out.append((start, end))
        return out

    letters = merged(LETTER)
    numbers = merged(NUMBER)

    m = re.search(r"unicode_set_whitespace = \{(.*?)\};", src, re.S)
    whitespace = sorted(int(c, 16) for c in re.findall(r"0x([0-9A-Fa-f]+)", m.group(1)))

    w = sys.stdout.write
    w("//! GENERATED FILE — do not edit by hand.\n")
    w("//!\n")
    w("//! Unicode \\p{L} / \\p{N} / \\s classification tables matching llama.cpp's\n")
    w("//! tokenizer (refs/llama.cpp/src/unicode-data.cpp), for token-ID-exact\n")
    w("//! pretokenizer parity. Regenerate with:\n")
    w("//!\n")
    w("//!     python3 tools/gen_unicode_categories.py > src/llm/unicode_categories.zig\n")
    w("\n")
    w("const std = @import(\"std\");\n")
    w("\n")
    w("const Range = struct { lo: u32, hi: u32 }; // [lo, hi) — hi exclusive\n")
    w("\n")

    def emit_table(name, ranges):
        w(f"const {name} = [_]Range{{\n")
        for lo, hi in ranges:
            w(f"    .{{ .lo = 0x{lo:06X}, .hi = 0x{hi:06X} }},\n")
        w("};\n\n")

    emit_table("letter_ranges", letters)
    emit_table("number_ranges", numbers)

    w("fn inRanges(ranges: []const Range, cp: u32) bool {\n")
    w("    // Binary search: largest range with lo <= cp, then bounds check.\n")
    w("    var lo: usize = 0;\n")
    w("    var hi: usize = ranges.len;\n")
    w("    while (lo < hi) {\n")
    w("        const mid = lo + (hi - lo) / 2;\n")
    w("        if (ranges[mid].lo <= cp) lo = mid + 1 else hi = mid;\n")
    w("    }\n")
    w("    if (lo == 0) return false;\n")
    w("    return cp < ranges[lo - 1].hi;\n")
    w("}\n")
    w("\n")
    w("/// Unicode \\p{L} (any letter category).\n")
    w("pub fn isLetter(cp: u32) bool {\n")
    w("    if (cp < 0x80) return (cp >= 'a' and cp <= 'z') or (cp >= 'A' and cp <= 'Z');\n")
    w("    return inRanges(&letter_ranges, cp);\n")
    w("}\n")
    w("\n")
    w("/// Unicode \\p{N} (any number category).\n")
    w("pub fn isNumber(cp: u32) bool {\n")
    w("    if (cp < 0x80) return cp >= '0' and cp <= '9';\n")
    w("    return inRanges(&number_ranges, cp);\n")
    w("}\n")
    w("\n")
    w("/// llama.cpp's \\s whitespace set (unicode_set_whitespace).\n")
    w("pub fn isWhitespace(cp: u32) bool {\n")
    w("    return switch (cp) {\n")
    ws_hex = ", ".join(f"0x{c:04X}" for c in whitespace)
    w(f"        {ws_hex} => true,\n")
    w("        else => false,\n")
    w("    };\n")
    w("}\n")
    w("\n")
    w("test \"ascii fast paths agree with the range tables\" {\n")
    w("    var cp: u32 = 0;\n")
    w("    while (cp < 0x80) : (cp += 1) {\n")
    w("        try std.testing.expectEqual(inRanges(&letter_ranges, cp), isLetter(cp));\n")
    w("        try std.testing.expectEqual(inRanges(&number_ranges, cp), isNumber(cp));\n")
    w("    }\n")
    w("}\n")
    w("\n")
    w("test \"spot checks against known categories\" {\n")
    w("    try std.testing.expect(isLetter('a') and isLetter('Z'));\n")
    w("    try std.testing.expect(isLetter(0x00E9)); // é\n")
    w("    try std.testing.expect(isLetter(0x4E2D)); // 中\n")
    w("    try std.testing.expect(!isLetter('1') and !isLetter(' ') and !isLetter(0x1F600));\n")
    w("    try std.testing.expect(isNumber('7'));\n")
    w("    try std.testing.expect(isNumber(0x0661)); // ١ arabic-indic one\n")
    w("    try std.testing.expect(isNumber(0x00B2)); // ² superscript two (\\p{No})\n")
    w("    try std.testing.expect(!isNumber('x'));\n")
    w("    try std.testing.expect(isWhitespace(' ') and isWhitespace('\\t') and isWhitespace('\\n') and isWhitespace('\\r'));\n")
    w("    try std.testing.expect(isWhitespace(0x00A0) and isWhitespace(0x3000));\n")
    w("    try std.testing.expect(!isWhitespace('a') and !isWhitespace(0x200B)); // ZWSP is \\p{Cf}, not \\s\n")
    w("}\n")


if __name__ == "__main__":
    main()
