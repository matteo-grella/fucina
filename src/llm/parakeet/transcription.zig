//! Parakeet word-level grouping (P10.3): faithful port of parakeet.cpp
//! `src/transcription.cpp::group_words`. Turns the per-token `TokenInfo` stream
//! into `Word{ text, start, end, conf }` matching NeMo's get_words_offsets
//! (tokenizer_type='bpe', word_delimiter=' ') + `_refine_timestamps` (punctuation
//! pinned to the previous token's end) + 'min' confidence aggregation.
//!
//!   word.start = first_token.frame              * frame_sec
//!   word.end   = (last_token.frame + last.span) * frame_sec
//!   word.conf  = min over the word's per-token confidences
//!   frame_sec  = hop_length * subsampling_factor / sample_rate
const std = @import("std");
const decoder = @import("decoder.zig");
const tokenizer = @import("tokenizer.zig");

const Allocator = std.mem.Allocator;
const TokenInfo = decoder.TokenInfo;

/// One decoded word with its time span (seconds) + aggregate confidence. `text` is
/// owned by the caller (free it; the `Word` slice from `groupWords` too).
pub const Word = struct {
    text: []u8,
    start: f32 = 0,
    end: f32 = 0,
    conf: f32 = 0,
};

// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece meta-space marker (3 bytes).
fn startsWithMeta(p: []const u8) bool {
    return p.len >= 3 and p[0] == 0xE2 and p[1] == 0x96 and p[2] == 0x81;
}

/// `▁`→space, strip one leading space (detokenize on a 1-element id list). Owned.
fn pieceToText(allocator: Allocator, p: []const u8) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var i: usize = 0;
    while (i < p.len) {
        if (i + 3 <= p.len and p[i] == 0xE2 and p[i + 1] == 0x96 and p[i + 2] == 0x81) {
            try buf.append(allocator, ' ');
            i += 3;
        } else {
            try buf.append(allocator, p[i]);
            i += 1;
        }
    }
    if (buf.items.len > 0 and buf.items[0] == ' ') _ = buf.orderedRemove(0);
    return buf.toOwnedSlice(allocator);
}

// Unicode category 'P*' ASCII code points (NeMo extract_punctuation_from_vocab).
fn isAsciiPunct(c: u8) bool {
    return switch (c) {
        '!', '"', '#', '%', '&', '\'', '(', ')', '*', ',', '-', '.', '/', ':', ';', '?', '@', '[', '\\', ']', '_', '{', '}' => true,
        else => false,
    };
}

fn isSpecialToken(tok: []const u8) bool {
    if (tok.len == 0) return true;
    if (tok[0] == '[' and tok[tok.len - 1] == ']') return true;
    if (tok[0] == '<' and tok[tok.len - 1] == '>') return true;
    if (tok.len >= 2 and tok[0] == '#' and tok[1] == '#') return true;
    if (startsWithMeta(tok)) return true;
    for (tok) |ch| {
        if (!std.ascii.isWhitespace(ch)) return false;
    }
    return true; // all-whitespace
}

const DELIM = " ";

fn isPunct(set: *const [256]bool, s: []const u8) bool {
    return s.len == 1 and s[0] != ' ' and set[s[0]];
}

/// Detokenize the `built` tokens' ids → owned word text.
fn detokBuilt(allocator: Allocator, tokens: []const TokenInfo, pieces: []const []const u8, built: []const usize) ![]u8 {
    const ids = try allocator.alloc(i32, built.len);
    defer allocator.free(ids);
    for (built, 0..) |k, j| ids[j] = tokens[k].id;
    return tokenizer.detokenize(allocator, pieces, ids);
}

fn minConf(conf: []const f32, built: []const usize) f32 {
    var m: f32 = 1.0;
    for (built) |k| m = @min(m, conf[k]);
    return m;
}

/// Group a per-token decode into words. Caller owns the result (free each
/// `Word.text` then the slice). `pieces` indexes by `TokenInfo.id`.
pub fn groupWords(allocator: Allocator, tokens: []const TokenInfo, pieces: []const []const u8, frame_sec: f32) ![]Word {
    var words: std.ArrayList(Word) = .empty;
    errdefer {
        for (words.items) |w| allocator.free(w.text);
        words.deinit(allocator);
    }
    const n = tokens.len;
    if (n == 0) return words.toOwnedSlice(allocator);

    // Supported-punctuation set (single-char strings present in non-special pieces).
    var punct: [256]bool = .{false} ** 256;
    for (pieces) |tok| {
        if (isSpecialToken(tok)) continue;
        for (tok) |c| {
            if (isAsciiPunct(c)) punct[c] = true;
        }
    }

    // Per-token char offsets: text = decode_ids_to_str([id]), tok = raw piece,
    // start/end in encoder frames (end = frame + span; CTC run-length span set by
    // the caller). conf = per-token confidence.
    const text = try allocator.alloc([]u8, n);
    defer allocator.free(text);
    const tokp = try allocator.alloc([]const u8, n);
    defer allocator.free(tokp);
    const start = try allocator.alloc(i32, n);
    defer allocator.free(start);
    const end = try allocator.alloc(i32, n);
    defer allocator.free(end);
    const conf = try allocator.alloc(f32, n);
    defer allocator.free(conf);
    var filled: usize = 0;
    // Registered after the array defers, so LIFO order frees the per-token
    // strings before their backing array.
    defer for (0..filled) |i| allocator.free(text[i]);
    for (0..n) |i| {
        const id = tokens[i].id;
        tokp[i] = if (id >= 0 and @as(usize, @intCast(id)) < pieces.len) pieces[@intCast(id)] else "";
        text[i] = try pieceToText(allocator, tokp[i]);
        filled = i + 1;
        start[i] = tokens[i].frame;
        end[i] = tokens[i].frame + tokens[i].span;
        conf[i] = tokens[i].conf;
    }

    // _refine_timestamps: a punctuation token (i>0) is pinned to the previous
    // token's end (start = prev end; end = start).
    for (0..n) |i| {
        if (text[i].len > 0 and i > 0 and isPunct(&punct, text[i][0..1])) {
            start[i] = end[i - 1];
            end[i] = start[i];
        }
    }

    var built: std.ArrayList(usize) = .empty;
    defer built.deinit(allocator);
    var prev: usize = 0;

    for (0..n) |i| {
        const ct = text[i];
        const tk = tokp[i];
        const curr_punct = isPunct(&punct, ct);

        // next non-delimiter token text (NeMo lookahead).
        var next_non_delim: []const u8 = "";
        var j = i;
        while (next_non_delim.len == 0 and j + 1 < n) {
            j += 1;
            if (!std.mem.eql(u8, text[j], DELIM)) next_non_delim = text[j];
        }
        const next_is_punct = next_non_delim.len > 0 and isPunct(&punct, next_non_delim);
        const word_start_cond = !std.mem.eql(u8, tk, ct) or (std.mem.eql(u8, ct, DELIM) and !next_is_punct);

        if (word_start_cond and !curr_punct) {
            if (built.items.len > 0) {
                try words.append(allocator, .{
                    .text = try detokBuilt(allocator, tokens, pieces, built.items),
                    .start = @as(f32, @floatFromInt(start[prev])) * frame_sec,
                    .end = @as(f32, @floatFromInt(end[built.items[built.items.len - 1]])) * frame_sec,
                    .conf = minConf(conf, built.items),
                });
            }
            built.clearRetainingCapacity();
            if (!std.mem.eql(u8, ct, DELIM)) {
                try built.append(allocator, i);
                prev = i;
            }
        } else if (curr_punct and built.items.len == 0 and words.items.len > 0) {
            // Punctuation with no open word: attach to the previous word (extend its
            // end, drop a trailing space, append the punctuation char).
            var lw = &words.items[words.items.len - 1];
            lw.end = @as(f32, @floatFromInt(end[i])) * frame_sec;
            var t = lw.text;
            if (t.len > 0 and t[t.len - 1] == ' ') t = t[0 .. t.len - 1];
            const nt = try std.mem.concat(allocator, u8, &.{ t, ct });
            allocator.free(lw.text);
            lw.text = nt;
            lw.conf = @min(lw.conf, conf[i]);
        } else if (curr_punct and built.items.len > 0) {
            // Punctuation closing an open word: drop a trailing delimiter token.
            const last = tokp[built.items[built.items.len - 1]];
            if (std.mem.eql(u8, last, " ") or std.mem.eql(u8, last, "_") or (last.len == 3 and startsWithMeta(last))) {
                _ = built.pop();
            }
            try built.append(allocator, i);
        } else {
            // Continuation sub-word: extend the current word.
            if (built.items.len == 0) prev = i;
            try built.append(allocator, i);
        }
    }

    // Tail: force the first word's start to the first token's start, flush remainder.
    if (words.items.len > 0) {
        words.items[0].start = @as(f32, @floatFromInt(start[0])) * frame_sec;
        if (built.items.len > 0) {
            try words.append(allocator, .{
                .text = try detokBuilt(allocator, tokens, pieces, built.items),
                .start = @as(f32, @floatFromInt(start[prev])) * frame_sec,
                .end = @as(f32, @floatFromInt(end[built.items[built.items.len - 1]])) * frame_sec,
                .conf = minConf(conf, built.items),
            });
        }
    } else if (built.items.len > 0) {
        try words.append(allocator, .{
            .text = try detokBuilt(allocator, tokens, pieces, built.items),
            .start = @as(f32, @floatFromInt(start[0])) * frame_sec,
            .end = @as(f32, @floatFromInt(end[built.items[built.items.len - 1]])) * frame_sec,
            .conf = minConf(conf, built.items),
        });
    }

    return words.toOwnedSlice(allocator);
}

/// Free a `Word` slice from `groupWords` (each `text` + the slice itself).
pub fn freeWords(allocator: Allocator, words: []Word) void {
    for (words) |w| allocator.free(w.text);
    allocator.free(words);
}

// === JSON output (P10.4) — byte-for-byte port of parakeet_capi.cpp ===

/// Append `s` as a JSON string (quoted + escaped), matching `append_json_string`.
fn appendJsonString(allocator: Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.append(allocator, '"');
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0C => try out.appendSlice(allocator, "\\f"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => if (c < 0x20) {
                var b: [6]u8 = undefined;
                try out.appendSlice(allocator, try std.fmt.bufPrint(&b, "\\u{x:0>4}", .{c}));
            } else try out.append(allocator, c),
        }
    }
    try out.append(allocator, '"');
}

/// Append `v` with `fmt` ("{d:.6}" etc.); NaN/huge → "0" (matches append_json_float).
fn appendJsonFloat(allocator: Allocator, out: *std.ArrayList(u8), comptime fmt: []const u8, v: f32) !void {
    if (v != v or v > 1e30 or v < -1e30) {
        try out.append(allocator, '0');
        return;
    }
    var b: [32]u8 = undefined;
    try out.appendSlice(allocator, try std.fmt.bufPrint(&b, fmt, .{v}));
}

/// Serialize to the C-API JSON document (`transcription_to_json`): owned string.
/// `{"text":..,"frame_sec":%.6f,"words":[{"w","start"%.3f,"end"%.3f,"conf"%.4f}],
///   "tokens":[{"id","t"%.3f,"conf"%.4f}]}`. Token `t` = frame * frame_sec.
pub fn toJson(allocator: Allocator, text: []const u8, frame_sec: f32, tokens: []const TokenInfo, words: []const Word) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"text\":");
    try appendJsonString(allocator, &out, text);
    try out.appendSlice(allocator, ",\"frame_sec\":");
    try appendJsonFloat(allocator, &out, "{d:.6}", frame_sec);
    try out.appendSlice(allocator, ",\"words\":[");
    for (words, 0..) |wd, i| {
        if (i > 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, "{\"w\":");
        try appendJsonString(allocator, &out, wd.text);
        try out.appendSlice(allocator, ",\"start\":");
        try appendJsonFloat(allocator, &out, "{d:.3}", wd.start);
        try out.appendSlice(allocator, ",\"end\":");
        try appendJsonFloat(allocator, &out, "{d:.3}", wd.end);
        try out.appendSlice(allocator, ",\"conf\":");
        try appendJsonFloat(allocator, &out, "{d:.4}", wd.conf);
        try out.append(allocator, '}');
    }
    try out.appendSlice(allocator, "],\"tokens\":[");
    for (tokens, 0..) |ti, i| {
        if (i > 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, "{\"id\":");
        var b: [16]u8 = undefined;
        try out.appendSlice(allocator, try std.fmt.bufPrint(&b, "{d}", .{ti.id}));
        try out.appendSlice(allocator, ",\"t\":");
        try appendJsonFloat(allocator, &out, "{d:.3}", @as(f32, @floatFromInt(ti.frame)) * frame_sec);
        try out.appendSlice(allocator, ",\"conf\":");
        try appendJsonFloat(allocator, &out, "{d:.4}", ti.conf);
        try out.append(allocator, '}');
    }
    try out.appendSlice(allocator, "]}");
    return out.toOwnedSlice(allocator);
}

test {
    _ = @import("transcription_tests.zig");
}
