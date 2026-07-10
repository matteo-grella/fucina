//! Speaker-attribute validation and normalisation for the OmniVoice instruct
//! string. Port of `refs/omnivoice.cpp/src/voice-design.h` (voice_design_init,
//! voice_design_normalize and their helpers, incl. the Ratcliff-Obershelp
//! SequenceMatcher.ratio port used for did-you-mean suggestions).
//!
//! Six mutually exclusive categories: gender / age / pitch / style (each with
//! EN+ZH forms), accent (English-only, 10) and dialect (Chinese-only, 12).
//! The reference keeps its vocabularies in `std::set<std::string>` and
//! iterates them for the did-you-mean scan and the error-message item lists,
//! so the sorted orders below (byte-wise, matching std::string comparison)
//! are load-bearing for byte-exact error messages and tie-breaking.

const std = @import("std");
const Allocator = std.mem.Allocator;

const Pair = struct { en: []const u8, zh: []const u8 };

const gender_pairs = [_]Pair{
    .{ .en = "male", .zh = "男" },
    .{ .en = "female", .zh = "女" },
};
const age_pairs = [_]Pair{
    .{ .en = "child", .zh = "儿童" },
    .{ .en = "teenager", .zh = "少年" },
    .{ .en = "young adult", .zh = "青年" },
    .{ .en = "middle-aged", .zh = "中年" },
    .{ .en = "elderly", .zh = "老年" },
};
const pitch_pairs = [_]Pair{
    .{ .en = "very low pitch", .zh = "极低音调" },
    .{ .en = "low pitch", .zh = "低音调" },
    .{ .en = "moderate pitch", .zh = "中音调" },
    .{ .en = "high pitch", .zh = "高音调" },
    .{ .en = "very high pitch", .zh = "极高音调" },
};
const style_pairs = [_]Pair{
    .{ .en = "whisper", .zh = "耳语" },
};

/// English-only accent labels (passed through untranslated).
pub const accents = [_][]const u8{
    "american accent", "british accent",  "australian accent", "chinese accent",
    "canadian accent", "indian accent",   "korean accent",     "portuguese accent",
    "russian accent",  "japanese accent",
};

/// Chinese-only dialect labels (passed through untranslated).
pub const dialects = [_][]const u8{
    "河南话",
    "陕西话",
    "四川话",
    "贵州话",
    "云南话",
    "桂林话",
    "济南话",
    "石家庄话",
    "甘肃话",
    "宁夏话",
    "青岛话",
    "东北话",
};

/// The en<->zh translation table (every category item existing in both
/// languages: gender + age + pitch + style).
const translate_pairs = gender_pairs ++ age_pairs ++ pitch_pairs ++ style_pairs;

fn pairMembers(comptime pairs: []const Pair) [pairs.len * 2][]const u8 {
    var out: [pairs.len * 2][]const u8 = undefined;
    for (pairs, 0..) |p, i| {
        out[2 * i] = p.en;
        out[2 * i + 1] = p.zh;
    }
    return out;
}

const category_gender = pairMembers(&gender_pairs);
const category_age = pairMembers(&age_pairs);
const category_pitch = pairMembers(&pitch_pairs);
const category_style = pairMembers(&style_pairs);

/// Reference `mutually_exclusive` category order: gender, age, pitch, style,
/// accent, dialect (order of the conflict groups in the error message).
const mutually_exclusive = [_][]const []const u8{
    &category_gender, &category_age, &category_pitch,
    &category_style,  &accents,      &dialects,
};

/// Byte-wise insertion sort, mirroring std::set's std::string ordering.
fn sorted(comptime n: usize, items: [n][]const u8) [n][]const u8 {
    var out = items;
    var i: usize = 1;
    while (i < n) : (i += 1) {
        const key = out[i];
        var j = i;
        while (j > 0 and std.mem.lessThan(u8, key, out[j - 1])) : (j -= 1) {
            out[j] = out[j - 1];
        }
        out[j] = key;
    }
    return out;
}

const en_items = blk: {
    var items: [translate_pairs.len + accents.len][]const u8 = undefined;
    for (translate_pairs, 0..) |p, i| items[i] = p.en;
    for (accents, 0..) |a, i| items[translate_pairs.len + i] = a;
    break :blk items;
};
const zh_items = blk: {
    var items: [translate_pairs.len + dialects.len][]const u8 = undefined;
    for (translate_pairs, 0..) |p, i| items[i] = p.zh;
    for (dialects, 0..) |d, i| items[translate_pairs.len + i] = d;
    break :blk items;
};

/// Reference `valid_en` in std::set iteration order (23 items).
pub const valid_en = blk: {
    @setEvalBranchQuota(100_000);
    break :blk sorted(en_items.len, en_items);
};
/// Reference `valid_zh` in std::set iteration order (25 items).
pub const valid_zh = blk: {
    @setEvalBranchQuota(100_000);
    break :blk sorted(zh_items.len, zh_items);
};
/// Reference `all_valid` in std::set iteration order (48 items — all distinct,
/// so no dedup is needed to match the set).
pub const all_valid = blk: {
    @setEvalBranchQuota(100_000);
    break :blk sorted(en_items.len + zh_items.len, en_items ++ zh_items);
};

/// Decodes one UTF-8 sequence at the start of `p`, writes the codepoint to
/// `cp` and returns the byte count (0 on malformed). Exact port of
/// `voice_design_utf8_decode` — deliberately permissive (no continuation-byte
/// or overlong checks), so do not swap in std.unicode.
fn utf8Decode(p: []const u8, cp: *u32) usize {
    if (p.len == 0) return 0;
    const c = p[0];
    if (c < 0x80) {
        cp.* = c;
        return 1;
    }
    if ((c & 0xE0) == 0xC0 and p.len >= 2) {
        cp.* = (@as(u32, c & 0x1F) << 6) | (@as(u32, p[1]) & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 and p.len >= 3) {
        cp.* = (@as(u32, c & 0x0F) << 12) | ((@as(u32, p[1]) & 0x3F) << 6) |
            (@as(u32, p[2]) & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 and p.len >= 4) {
        cp.* = (@as(u32, c & 0x07) << 18) | ((@as(u32, p[1]) & 0x3F) << 12) |
            ((@as(u32, p[2]) & 0x3F) << 6) | (@as(u32, p[3]) & 0x3F);
        return 4;
    }
    return 0;
}

/// True if the codepoint sits in the CJK Unified Ideographs main range
/// (U+4E00..U+9FFF), the same range the reference `_ZH_RE` uses.
fn isCjk(cp: u32) bool {
    return cp >= 0x4E00 and cp <= 0x9FFF;
}

/// True if the UTF-8 string contains at least one CJK ideograph
/// (reference `voice_design_has_cjk`; malformed bytes are skipped one at a
/// time, exactly like the reference resync).
pub fn hasCjk(text: []const u8) bool {
    var i: usize = 0;
    while (i < text.len) {
        var cp: u32 = 0;
        const n = utf8Decode(text[i..], &cp);
        if (n == 0) {
            i += 1;
            continue;
        }
        if (isCjk(cp)) return true;
        i += n;
    }
    return false;
}

/// Total matched character count of the recursive Ratcliff-Obershelp matching
/// within a[ai..ai+am) and b[bi..bi+bm). Exact port of `voice_design_match`
/// (same algorithm as Python difflib SequenceMatcher.get_matching_blocks()).
fn matchLen(
    allocator: Allocator,
    a: []const u8,
    ai: usize,
    am: usize,
    b: []const u8,
    bi: usize,
    bm: usize,
) Allocator.Error!usize {
    if (am == 0 or bm == 0) return 0;
    var best_size: usize = 0;
    var best_a: usize = ai;
    var best_b: usize = bi;
    var prev = try allocator.alloc(usize, bm + 1);
    defer allocator.free(prev);
    var curr = try allocator.alloc(usize, bm + 1);
    defer allocator.free(curr);
    @memset(prev, 0);
    @memset(curr, 0);
    for (0..am) |i| {
        for (0..bm) |j| {
            if (a[ai + i] == b[bi + j]) {
                curr[j + 1] = prev[j] + 1;
                if (curr[j + 1] > best_size) {
                    best_size = curr[j + 1];
                    best_a = ai + i + 1 - best_size;
                    best_b = bi + j + 1 - best_size;
                }
            } else {
                curr[j + 1] = 0;
            }
        }
        const tmp = prev;
        prev = curr;
        curr = tmp;
        @memset(curr, 0);
    }
    if (best_size == 0) return 0;
    const left = try matchLen(allocator, a, ai, best_a - ai, b, bi, best_b - bi);
    const right = try matchLen(allocator, a, best_a + best_size, ai + am - (best_a + best_size), b, best_b + best_size, bi + bm - (best_b + best_size));
    return best_size + left + right;
}

/// SequenceMatcher.ratio(): 2 * matched / (len(a) + len(b)), f32 arithmetic
/// exactly as the reference `voice_design_ratio`.
pub fn ratio(allocator: Allocator, a: []const u8, b: []const u8) Allocator.Error!f32 {
    if (a.len == 0 and b.len == 0) return 1.0;
    if (a.len == 0 or b.len == 0) return 0.0;
    const m = try matchLen(allocator, a, 0, a.len, b, 0, b.len);
    return 2.0 * @as(f32, @floatFromInt(m)) / @as(f32, @floatFromInt(a.len + b.len));
}

/// Whitespace strip (ASCII space and tab only), reference `voice_design_strip`.
/// Returns a subslice of `s` (borrow, no allocation).
fn strip(s: []const u8) []const u8 {
    var a: usize = 0;
    var b: usize = s.len;
    while (a < b and (s[a] == ' ' or s[a] == '\t')) a += 1;
    while (b > a and (s[b - 1] == ' ' or s[b - 1] == '\t')) b -= 1;
    return s[a..b];
}

/// ASCII tolower into a fresh owned copy; bytes outside 'A'..'Z' (incl. all
/// UTF-8 continuation/lead bytes) pass through, matching `voice_design_lower`.
fn lowerDupe(allocator: Allocator, s: []const u8) Allocator.Error![]u8 {
    const out = try allocator.dupe(u8, s);
    for (out) |*c| {
        if (c.* >= 'A' and c.* <= 'Z') c.* += 32;
    }
    return out;
}

fn listContains(list: []const []const u8, item: []const u8) bool {
    for (list) |v| {
        if (std.mem.eql(u8, v, item)) return true;
    }
    return false;
}

const mixed_error = "Cannot mix Chinese dialect and English accent in a single instruct. " ++
    "Dialects are for Chinese speech, accents for English speech.";

/// Result of `normalize`: either the normalised instruct string or the
/// reference's multi-line error text. Both variants are owned by the caller.
pub const Normalized = union(enum) {
    ok: []u8,
    invalid: []u8,

    pub fn deinit(self: Normalized, allocator: Allocator) void {
        switch (self) {
            inline else => |s| allocator.free(s),
        }
    }
};

/// Validates and normalises an instruct string; port of
/// `voice_design_normalize`. `.ok` holds the items joined with ", " (English)
/// or "，" (any CJK in the result); empty input yields an empty `.ok`.
/// `.invalid` carries the reference's error message byte-for-byte.
///
/// `use_zh` selects the target language when neither a dialect (forces
/// Chinese) nor an accent (forces English) decides it; the reference caller
/// passes true when the synthesis text contains CJK.
pub fn normalize(allocator: Allocator, instruct: []const u8, use_zh_hint: bool) Allocator.Error!Normalized {
    var use_zh = use_zh_hint;

    const instruct_str = strip(instruct);
    if (instruct_str.len == 0) {
        return .{ .ok = try allocator.dupe(u8, "") };
    }

    // Split on half-width ',' or full-width '，' (UTF-8 EF BC 8C), stripping
    // each item and dropping empties, matching the reference split loop
    // (Python regex r"\s*[,，]\s*").
    var raw_items: std.ArrayList([]const u8) = .empty;
    defer raw_items.deinit(allocator);
    {
        var start: usize = 0;
        var i: usize = 0;
        while (i < instruct_str.len) {
            var is_sep = false;
            var skip: usize = 0;
            if (instruct_str[i] == ',') {
                is_sep = true;
                skip = 1;
            } else if (i + 2 < instruct_str.len and instruct_str[i] == 0xEF and
                instruct_str[i + 1] == 0xBC and instruct_str[i + 2] == 0x8C)
            {
                is_sep = true;
                skip = 3;
            }
            if (is_sep) {
                const item = strip(instruct_str[start..i]);
                if (item.len != 0) try raw_items.append(allocator, item);
                start = i + skip;
                i = start;
            } else {
                i += 1;
            }
        }
        const item = strip(instruct_str[start..]);
        if (item.len != 0) try raw_items.append(allocator, item);
    }

    // Lowercased copies owned here; `normalised` / `unknown` borrow them (or
    // static vocabulary strings after translation).
    var owned: std.ArrayList([]u8) = .empty;
    defer {
        for (owned.items) |s| allocator.free(s);
        owned.deinit(allocator);
    }

    var normalised: std.ArrayList([]const u8) = .empty;
    defer normalised.deinit(allocator);

    const Unknown = struct { raw: []const u8, n: []const u8, sug: []const u8 };
    var unknown: std.ArrayList(Unknown) = .empty;
    defer unknown.deinit(allocator);

    for (raw_items.items) |raw| {
        const lowered = try lowerDupe(allocator, strip(raw));
        {
            errdefer allocator.free(lowered);
            try owned.append(allocator, lowered);
        }
        const n: []const u8 = lowered;
        if (listContains(&all_valid, n)) {
            try normalised.append(allocator, n);
        } else {
            // Did-you-mean scan in std::set (sorted) order; `>=` means later
            // items win ratio ties, exactly like the reference loop.
            var best: []const u8 = "";
            var best_ratio: f32 = 0.6;
            for (all_valid) |v| {
                const r = try ratio(allocator, n, v);
                if (r >= best_ratio) {
                    best = v;
                    best_ratio = r;
                }
            }
            try unknown.append(allocator, .{ .raw = raw, .n = n, .sug = best });
        }
    }

    if (unknown.items.len != 0) {
        var msg: std.ArrayList(u8) = .empty;
        defer msg.deinit(allocator);
        try msg.appendSlice(allocator, "Unsupported instruct items found in ");
        try msg.appendSlice(allocator, instruct_str);
        try msg.appendSlice(allocator, ":\n");
        for (unknown.items) |u| {
            try msg.appendSlice(allocator, "  '");
            try msg.appendSlice(allocator, u.raw);
            try msg.appendSlice(allocator, "' -> '");
            try msg.appendSlice(allocator, u.n);
            try msg.appendSlice(allocator, "' (unsupported");
            if (u.sug.len != 0) {
                try msg.appendSlice(allocator, "; did you mean '");
                try msg.appendSlice(allocator, u.sug);
                try msg.appendSlice(allocator, "'?");
            }
            try msg.appendSlice(allocator, ")\n");
        }
        try msg.appendSlice(allocator, "\nValid English items: ");
        for (valid_en, 0..) |v, i| {
            if (i != 0) try msg.appendSlice(allocator, ", ");
            try msg.appendSlice(allocator, v);
        }
        try msg.appendSlice(allocator, "\nValid Chinese items: ");
        for (valid_zh, 0..) |v, i| {
            if (i != 0) try msg.appendSlice(allocator, "，");
            try msg.appendSlice(allocator, v);
        }
        try msg.appendSlice(allocator, "\n\nTip: Use only English or only Chinese instructs. " ++
            "English instructs should use comma + space (e.g. " ++
            "'male, indian accent'),\nChinese instructs should use full-width " ++
            "comma (e.g. '男，河南话').");
        return .{ .invalid = try msg.toOwnedSlice(allocator) };
    }

    // Language consistency: dialect forces Chinese, accent forces English.
    var has_dialect = false;
    var has_accent = false;
    for (normalised.items) |n| {
        if (std.mem.endsWith(u8, n, "话")) has_dialect = true;
        if (std.mem.indexOf(u8, n, " accent") != null) has_accent = true;
    }
    if (has_dialect and has_accent) {
        return .{ .invalid = try allocator.dupe(u8, mixed_error) };
    }
    if (has_dialect) {
        use_zh = true;
    } else if (has_accent) {
        use_zh = false;
    }

    // Translate to the unified language.
    if (use_zh) {
        for (normalised.items) |*n| {
            for (translate_pairs) |p| {
                if (std.mem.eql(u8, n.*, p.en)) {
                    n.* = p.zh;
                    break;
                }
            }
        }
    } else {
        for (normalised.items) |*n| {
            for (translate_pairs) |p| {
                if (std.mem.eql(u8, n.*, p.zh)) {
                    n.* = p.en;
                    break;
                }
            }
        }
    }

    // Category conflict check: at most one item per mutually-exclusive set
    // (duplicates count — 'male, male' conflicts, like the reference).
    var conflict_groups: std.ArrayList(u8) = .empty;
    defer conflict_groups.deinit(allocator);
    var any_conflict = false;
    for (mutually_exclusive) |cat| {
        var hit_count: usize = 0;
        for (normalised.items) |n| {
            if (listContains(cat, n)) hit_count += 1;
        }
        if (hit_count <= 1) continue;
        if (any_conflict) try conflict_groups.appendSlice(allocator, "; ");
        var first = true;
        for (normalised.items) |n| {
            if (!listContains(cat, n)) continue;
            if (!first) try conflict_groups.appendSlice(allocator, " vs ");
            try conflict_groups.appendSlice(allocator, "'");
            try conflict_groups.appendSlice(allocator, n);
            try conflict_groups.appendSlice(allocator, "'");
            first = false;
        }
        any_conflict = true;
    }
    if (any_conflict) {
        var msg: std.ArrayList(u8) = .empty;
        defer msg.deinit(allocator);
        try msg.appendSlice(allocator, "Conflicting instruct items within the same category: ");
        try msg.appendSlice(allocator, conflict_groups.items);
        try msg.appendSlice(allocator, ". Each category (gender, age, pitch, style, accent, dialect) " ++
            "allows at most one item.");
        return .{ .invalid = try msg.toOwnedSlice(allocator) };
    }

    // Pick the separator from the language of the result.
    var any_zh = false;
    for (normalised.items) |n| {
        if (hasCjk(n)) {
            any_zh = true;
            break;
        }
    }
    const sep: []const u8 = if (any_zh) "，" else ", ";

    var result: std.ArrayList(u8) = .empty;
    defer result.deinit(allocator);
    for (normalised.items, 0..) |n, k| {
        if (k > 0) try result.appendSlice(allocator, sep);
        try result.appendSlice(allocator, n);
    }
    return .{ .ok = try result.toOwnedSlice(allocator) };
}

test {
    _ = @import("voicedesign_tests.zig");
}
