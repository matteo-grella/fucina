//! Long-form text splitter for the OmniVoice example. Port of
//! `refs/omnivoice.cpp/src/text-chunker.h` (itself a 1:1 port of
//! omnivoice/utils/text.py `chunk_text_punctuation` / `add_punctuation`).
//!
//! Splits text on sentence-ending punctuation (skipping abbreviation
//! periods), then merges sentences into chunks of at most `chunk_len` UTF-8
//! codepoints; optional `min_chunk_len` merges undersized chunks into a
//! neighbour. Strings are UTF-8 in, UTF-8 out; comparison and length are
//! codepoint-based, matching Python str semantics. The reference builds
//! vectors of per-codepoint strings; because every codepoint lands, in text
//! order, either in the current sentence or glued onto the previous one,
//! sentences/chunks are always contiguous byte ranges of the input — this
//! port tracks (byte range, codepoint count) instead, with identical output.

const std = @import("std");

/// Byte length of the UTF-8 codepoint starting at `b` (1, 2, 3 or 4).
/// Falls back to 1 on invalid first bytes so iteration always advances.
/// Mirrors `chunker_utf8_len`.
pub fn utf8Len(b: u8) usize {
    if (b & 0x80 == 0x00) return 1;
    if (b & 0xE0 == 0xC0) return 2;
    if (b & 0xF0 == 0xE0) return 3;
    if (b & 0xF8 == 0xF0) return 4;
    return 1;
}

/// Sentence-ending punctuation. Mirrors SPLIT_PUNCTUATION in text.py.
pub const split_punctuation = [_][]const u8{
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "\xe3\x80\x82", // U+3002 ideographic full stop
    "\xef\xbc\x8c", // U+FF0C fullwidth comma
    "\xef\xbc\x9b", // U+FF1B fullwidth semicolon
    "\xef\xbc\x9a", // U+FF1A fullwidth colon
    "\xef\xbc\x81", // U+FF01 fullwidth exclamation mark
    "\xef\xbc\x9f", // U+FF1F fullwidth question mark
};

/// Closing marks attach to the preceding sentence. Mirrors CLOSING_MARKS.
pub const closing_marks = [_][]const u8{
    "\"", "'", "]", ">",
    "\xe2\x80\x9c", // U+201C left double quotation mark
    "\xe2\x80\x9d", // U+201D right double quotation mark
    "\xe2\x80\x98", // U+2018 left single quotation mark
    "\xe2\x80\x99", // U+2019 right single quotation mark
    "\xef\xbc\x89", // U+FF09 fullwidth right parenthesis
    "\xe3\x80\x8b", // U+300B right double angle bracket
    "\xe3\x80\x8d", // U+300D right corner bracket
    "\xe3\x80\x91", // U+3011 right black lenticular bracket
};

/// Abbreviations that suppress the period as a sentence break. ASCII only,
/// matched on the last whitespace-delimited word ending with the period.
/// Mirrors ABBREVIATIONS in text.py.
pub const abbreviations = [_][]const u8{
    "Mr.",  "Mrs.",  "Ms.",  "Dr.",  "Prof.", "Sr.",   "Jr.",     "Rev.", "Fr.",   "Hon.", "Pres.",
    "Gov.", "Capt.", "Gen.", "Sen.", "Rep.",  "Col.",  "Maj.",    "Lt.",  "Cmdr.", "Sgt.", "Cpl.",
    "Co.",  "Corp.", "Inc.", "Ltd.", "Est.",  "Dept.", "St.",     "Ave.", "Blvd.", "Rd.",  "Mt.",
    "Ft.",  "No.",   "Jan.", "Feb.", "Mar.",  "Apr.",  "Aug.",    "Sep.", "Sept.", "Oct.", "Nov.",
    "Dec.", "i.e.",  "e.g.", "vs.",  "Vs.",   "Etc.",  "approx.", "fig.", "def.",
};

/// Punctuation considered "terminal" by `addPunctuation`. Mirrors
/// END_PUNCTUATION in text.py.
pub const end_punctuation = [_][]const u8{
    ";",
    ":",
    ",",
    ".",
    "!",
    "?",
    ")",
    "]",
    "}",
    "\"",
    "'",
    "\xe2\x80\xa6", // U+2026 horizontal ellipsis
    "\xe2\x80\x9c", // U+201C left double quotation mark
    "\xe2\x80\x9d", // U+201D right double quotation mark
    "\xe2\x80\x98", // U+2018 left single quotation mark
    "\xe2\x80\x99", // U+2019 right single quotation mark
    "\xef\xbc\x9b", // U+FF1B fullwidth semicolon
    "\xef\xbc\x9a", // U+FF1A fullwidth colon
    "\xef\xbc\x8c", // U+FF0C fullwidth comma
    "\xe3\x80\x82", // U+3002 ideographic full stop
    "\xef\xbc\x81", // U+FF01 fullwidth exclamation mark
    "\xef\xbc\x9f", // U+FF1F fullwidth question mark
    "\xe3\x80\x81", // U+3001 ideographic comma
    "\xef\xbc\x89", // U+FF09 fullwidth right parenthesis
    "\xe3\x80\x91", // U+3011 right black lenticular bracket
};

/// std::set<std::string>::count equivalent: byte-exact membership of a
/// codepoint string in one of the small sets above.
fn inSet(set: []const []const u8, cp: []const u8) bool {
    for (set) |entry| {
        if (std.mem.eql(u8, entry, cp)) return true;
    }
    return false;
}

/// True if `cp` (a UTF-8 codepoint string) is whitespace per Python's
/// str.split() / str.strip() definition: ASCII (\t \n \v \f \r space) plus
/// the Unicode whitespace block. Mirrors `chunker_is_unicode_whitespace`.
pub fn isUnicodeWhitespace(cp: []const u8) bool {
    if (cp.len == 1) {
        const b = cp[0];
        return b == ' ' or b == '\t' or b == '\n' or b == '\r' or b == 0x0B or b == 0x0C;
    }

    if (cp.len == 2) {
        // U+0085 NEL (C2 85), U+00A0 NBSP (C2 A0).
        if (cp[0] == 0xC2) {
            return cp[1] == 0x85 or cp[1] == 0xA0;
        }
        return false;
    }

    if (cp.len == 3) {
        const u: u32 = (@as(u32, cp[0] & 0x0F) << 12) | (@as(u32, cp[1] & 0x3F) << 6) | @as(u32, cp[2] & 0x3F);

        // U+1680 OGHAM SPACE MARK
        if (u == 0x1680) return true;
        // U+2000..U+200A en quad/em/thin/hair spaces
        if (u >= 0x2000 and u <= 0x200A) return true;
        // U+2028 LINE SEP, U+2029 PARAGRAPH SEP, U+202F NARROW NBSP
        if (u == 0x2028 or u == 0x2029 or u == 0x202F) return true;
        // U+205F MEDIUM MATHEMATICAL SPACE
        if (u == 0x205F) return true;
        // U+3000 IDEOGRAPHIC SPACE
        if (u == 0x3000) return true;
        // U+FEFF ZERO WIDTH NO-BREAK SPACE (BOM)
        if (u == 0xFEFF) return true;
        return false;
    }

    return false;
}

/// Last whitespace-delimited word of `s` (borrowed slice), or `s` itself if
/// no whitespace; empty if all-whitespace. Whitespace is the Unicode block,
/// matching Python str.split()[-1]. Mirrors `chunker_last_word`.
pub fn lastWord(s: []const u8) []const u8 {
    var last_word_byte_start: usize = 0;
    var prev_was_ws = true;
    var any_non_ws = false;
    var trailing_ws_start_at: usize = s.len;

    var i: usize = 0;
    while (i < s.len) {
        var n = utf8Len(s[i]);
        if (i + n > s.len) n = s.len - i;
        const cp = s[i .. i + n];

        const is_ws = isUnicodeWhitespace(cp);
        if (is_ws) {
            if (!prev_was_ws) trailing_ws_start_at = i;
        } else {
            if (prev_was_ws) last_word_byte_start = i;
            any_non_ws = true;
            trailing_ws_start_at = i + n;
        }
        prev_was_ws = is_ws;
        i += n;
    }

    if (!any_non_ws) return s[0..0];
    return s[last_word_byte_start..trailing_ws_start_at];
}

/// Strips leading and trailing Unicode whitespace (borrowed slice). Matches
/// Python str.strip(). Mirrors `chunker_strip`.
pub fn strip(s: []const u8) []const u8 {
    var start_byte: usize = 0;
    while (start_byte < s.len) {
        var n = utf8Len(s[start_byte]);
        if (start_byte + n > s.len) n = s.len - start_byte;
        if (!isUnicodeWhitespace(s[start_byte .. start_byte + n])) break;
        start_byte += n;
    }

    if (start_byte >= s.len) return s[0..0];

    var after_last_non_ws = start_byte;
    var i = start_byte;
    while (i < s.len) {
        var n = utf8Len(s[i]);
        if (i + n > s.len) n = s.len - i;
        if (!isUnicodeWhitespace(s[i .. i + n])) after_last_non_ws = i + n;
        i += n;
    }

    return s[start_byte..after_last_non_ws];
}

/// Mirrors min_chunk_len=3 hardcoded in omnivoice/models/omnivoice.py:815
/// (reference OMNIVOICE_MIN_CHUNK_LEN).
pub const min_chunk_len_default: i32 = 3;

/// Contiguous run of the input text: byte range + codepoint count.
const Run = struct {
    start: usize,
    end: usize,
    cps: usize,

    fn extend(self: *Run, other: Run) void {
        self.end = other.end;
        self.cps += other.cps;
    }
};

/// Splits text on sentence-ending punctuation (skipping abbreviations) and
/// merges sentences into chunks of at most `chunk_len` codepoints. If
/// `min_chunk_len > 0`, undersized chunks are merged with a neighbour.
/// Returns stripped chunk strings (UTF-8); empty chunks are dropped.
///
/// Ownership: the caller owns the returned slice AND every chunk string in
/// it — free with `freeChunks` (or free each element, then the slice).
///
/// Strict semantic port of `chunk_text_punctuation`.
pub fn chunkTextPunctuation(allocator: std.mem.Allocator, text: []const u8, chunk_len: i32, min_chunk_len: i32) ![][]u8 {
    // Step 1: walk UTF-8 codepoints, split on punctuation. Leading
    // punctuation glues onto the previous sentence; a period whose last
    // whitespace-delimited word is an abbreviation does not break.
    var sentences: std.ArrayList(Run) = .empty;
    defer sentences.deinit(allocator);
    var current: ?Run = null;

    var i: usize = 0;
    while (i < text.len) {
        var n = utf8Len(text[i]);
        if (i + n > text.len) n = text.len - i;
        const cp = text[i .. i + n];
        i += n;

        const is_split = inSet(&split_punctuation, cp);
        const is_closing = inSet(&closing_marks, cp);

        // Leading punctuation glues onto the previous sentence.
        if (current == null and sentences.items.len > 0 and (is_split or is_closing)) {
            // The previous sentence ends exactly where this codepoint starts.
            sentences.items[sentences.items.len - 1].extend(.{ .start = i - n, .end = i, .cps = 1 });
            continue;
        }

        if (current) |*cur| {
            cur.extend(.{ .start = i - n, .end = i, .cps = 1 });
        } else {
            current = .{ .start = i - n, .end = i, .cps = 1 };
        }

        if (!is_split) continue;

        // Period after an abbreviation does not break the sentence.
        var is_abbreviation = false;
        if (std.mem.eql(u8, cp, ".")) {
            const joined = text[current.?.start..current.?.end];
            const last = lastWord(joined);
            if (last.len != 0 and inSet(&abbreviations, last)) {
                is_abbreviation = true;
            }
        }

        if (!is_abbreviation) {
            try sentences.append(allocator, current.?);
            current = null;
        }
    }

    if (current) |cur| {
        try sentences.append(allocator, cur);
    }

    // Step 2: greedy merge of sentences into chunks of at most chunk_len
    // codepoints. A sentence that does not fit starts a new chunk by itself,
    // even if it is longer than chunk_len.
    var merged: std.ArrayList(Run) = .empty;
    defer merged.deinit(allocator);
    var cur_chunk: ?Run = null;

    for (sentences.items) |sent| {
        const cur_cps: usize = if (cur_chunk) |c| c.cps else 0;
        if (@as(i64, @intCast(cur_cps + sent.cps)) <= chunk_len) {
            if (cur_chunk) |*c| c.extend(sent) else cur_chunk = sent;
        } else {
            if (cur_chunk) |c| try merged.append(allocator, c);
            cur_chunk = sent;
        }
    }

    if (cur_chunk) |c| {
        try merged.append(allocator, c);
    }

    // Step 3: merge undersized chunks. The first chunk, if short, is folded
    // into the second. Subsequent short chunks fold into the previous one.
    var finals: std.ArrayList(Run) = .empty;
    defer finals.deinit(allocator);

    if (min_chunk_len > 0) {
        const first_short = merged.items.len > 0 and @as(i64, @intCast(merged.items[0].cps)) < min_chunk_len;

        for (merged.items, 0..) |chunk, idx| {
            if (idx == 1 and first_short) {
                finals.items[finals.items.len - 1].extend(chunk);
                continue;
            }

            if (@as(i64, @intCast(chunk.cps)) >= min_chunk_len) {
                try finals.append(allocator, chunk);
                continue;
            }

            if (finals.items.len == 0) {
                try finals.append(allocator, chunk);
            } else {
                finals.items[finals.items.len - 1].extend(chunk);
            }
        }
    } else {
        try finals.appendSlice(allocator, merged.items);
    }

    // Step 4: materialize, strip whitespace, drop empty.
    var result: std.ArrayList([]u8) = .empty;
    defer result.deinit(allocator);
    errdefer for (result.items) |chunk| allocator.free(chunk);

    for (finals.items) |chunk| {
        const stripped = strip(text[chunk.start..chunk.end]);
        if (stripped.len != 0) {
            const owned = try allocator.dupe(u8, stripped);
            errdefer allocator.free(owned);
            try result.append(allocator, owned);
        }
    }

    return result.toOwnedSlice(allocator);
}

/// Frees a chunk list returned by `chunkTextPunctuation`.
pub fn freeChunks(allocator: std.mem.Allocator, chunks: [][]u8) void {
    for (chunks) |chunk| allocator.free(chunk);
    allocator.free(chunks);
}

/// Counts UTF-8 codepoints in `text`, matching Python's len(text).
/// Mirrors `chunker_utf8_count`.
pub fn utf8Count(text: []const u8) usize {
    var count: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        var n = utf8Len(text[i]);
        if (i + n > text.len) n = text.len - i;
        i += n;
        count += 1;
    }
    return count;
}

/// Last UTF-8 codepoint of `s` (borrowed slice), or empty if `s` is empty.
/// A string of only continuation bytes yields `s` itself, exactly like the
/// reference. Mirrors `chunker_last_codepoint`.
pub fn lastCodepoint(s: []const u8) []const u8 {
    if (s.len == 0) return s;

    var i = s.len;
    while (i > 0) : (i -= 1) {
        if (s[i - 1] & 0xC0 != 0x80) return s[i - 1 ..];
    }
    return s;
}

/// True if any codepoint of `s` falls inside the CJK Unified Ideographs
/// block (U+4E00..U+9FFF). Note the reference bails out (false) on a
/// truncated trailing sequence instead of clipping it. Mirrors
/// `chunker_contains_chinese`.
pub fn containsChinese(s: []const u8) bool {
    var i: usize = 0;
    while (i < s.len) {
        const n = utf8Len(s[i]);
        if (i + n > s.len) return false;

        if (n == 3) {
            const cp: u32 = (@as(u32, s[i] & 0x0F) << 12) | (@as(u32, s[i + 1] & 0x3F) << 6) | @as(u32, s[i + 2] & 0x3F);
            if (cp >= 0x4E00 and cp <= 0x9FFF) return true;
        }

        i += n;
    }
    return false;
}

/// Strips text and appends a terminal punctuation if missing: "." for
/// non-Chinese text, the ideographic full stop U+3002 for text containing
/// CJK. Returns an allocator-owned string (possibly empty); caller frees.
/// Mirrors `add_punctuation`.
pub fn addPunctuation(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const s = strip(text);
    if (s.len == 0) return allocator.dupe(u8, s);

    const last = lastCodepoint(s);
    if (inSet(&end_punctuation, last)) return allocator.dupe(u8, s);

    const suffix: []const u8 = if (containsChinese(s)) "\xe3\x80\x82" else ".";
    const out = try allocator.alloc(u8, s.len + suffix.len);
    @memcpy(out[0..s.len], s);
    @memcpy(out[s.len..], suffix);
    return out;
}

test {
    _ = @import("chunker_tests.zig");
}
