//! Golden-parity tests for the OmniVoice voice-design module
//! (`voicedesign.zig`) vs the C++ reference
//! (refs/omnivoice.cpp/src/voice-design.h). Golden strings and f32 bit
//! patterns were produced by a harness compiled against the reference header
//! (scratchpad vd_lang_golden.cpp: voice_design_normalize outputs/error
//! messages printed as raw bytes + hex, voice_design_ratio as u32 bit
//! patterns). All comparisons are byte/bit-exact.

const std = @import("std");
const voicedesign = @import("voicedesign.zig");

fn expectOk(instruct: []const u8, use_zh: bool, expected: []const u8) !void {
    const allocator = std.testing.allocator;
    const res = try voicedesign.normalize(allocator, instruct, use_zh);
    defer res.deinit(allocator);
    switch (res) {
        .ok => |s| try std.testing.expectEqualStrings(expected, s),
        .invalid => |e| {
            std.debug.print("unexpected invalid: {s}\n", .{e});
            return error.TestUnexpectedResult;
        },
    }
}

fn expectInvalid(instruct: []const u8, use_zh: bool, expected_err: []const u8) !void {
    const allocator = std.testing.allocator;
    const res = try voicedesign.normalize(allocator, instruct, use_zh);
    defer res.deinit(allocator);
    switch (res) {
        .ok => |s| {
            std.debug.print("unexpected ok: {s}\n", .{s});
            return error.TestUnexpectedResult;
        },
        .invalid => |e| try std.testing.expectEqualStrings(expected_err, e),
    }
}

test "normalize: valid combinations (golden)" {
    // en_valid
    try expectOk("male, young adult, moderate pitch", false, "male, young adult, moderate pitch");
    // en_valid_zh_target (en→zh translation)
    try expectOk("male, young adult, moderate pitch", true, "男，青年，中音调");
    // zh_valid
    try expectOk("女，老年，极高音调", true, "女，老年，极高音调");
    // zh_valid_en_target (zh→en translation)
    try expectOk("女，老年，极高音调", false, "female, elderly, very high pitch");
}

test "normalize: dialect forces zh, accent forces en (golden)" {
    // dialect_forces_zh
    try expectOk("female, 四川话", false, "女，四川话");
    // accent_forces_en
    try expectOk("男，british accent", true, "male, british accent");
    // single_dialect
    try expectOk("东北话", false, "东北话");
}

test "normalize: strip / case / separators (golden)" {
    // messy_case
    try expectOk(" Male ,\tYOUNG adult , moderate PITCH ", false, "male, young adult, moderate pitch");
    // fullwidth_en
    try expectOk("male，whisper", false, "male, whisper");
    // empty / spaces_only / commas_only
    try expectOk("", false, "");
    try expectOk("   ", false, "");
    try expectOk(",,,", false, "");
}

test "normalize: unknown items with did-you-mean (golden error strings)" {
    // typo_didyoumean
    try expectInvalid("male, moderate pich", false, "Unsupported instruct items found in male, moderate pich:\n  'moderate pich' -> 'moderate pich' (unsupported; did you mean 'moderate pitch'?)\n\nValid English items: american accent, australian accent, british accent, canadian accent, child, chinese accent, elderly, female, high pitch, indian accent, japanese accent, korean accent, low pitch, male, middle-aged, moderate pitch, portuguese accent, russian accent, teenager, very high pitch, very low pitch, whisper, young adult\nValid Chinese items: 东北话，中年，中音调，云南话，低音调，儿童，四川话，女，宁夏话，少年，极低音调，极高音调，桂林话，河南话，济南话，甘肃话，男，石家庄话，老年，耳语，贵州话，陕西话，青岛话，青年，高音调\n\nTip: Use only English or only Chinese instructs. English instructs should use comma + space (e.g. 'male, indian accent'),\nChinese instructs should use full-width comma (e.g. '男，河南话').");
    // typo_no_sugg (no candidate reaches ratio 0.6)
    try expectInvalid("xqzk", false, "Unsupported instruct items found in xqzk:\n  'xqzk' -> 'xqzk' (unsupported)\n\nValid English items: american accent, australian accent, british accent, canadian accent, child, chinese accent, elderly, female, high pitch, indian accent, japanese accent, korean accent, low pitch, male, middle-aged, moderate pitch, portuguese accent, russian accent, teenager, very high pitch, very low pitch, whisper, young adult\nValid Chinese items: 东北话，中年，中音调，云南话，低音调，儿童，四川话，女，宁夏话，少年，极低音调，极高音调，桂林话，河南话，济南话，甘肃话，男，石家庄话，老年，耳语，贵州话，陕西话，青岛话，青年，高音调\n\nTip: Use only English or only Chinese instructs. English instructs should use comma + space (e.g. 'male, indian accent'),\nChinese instructs should use full-width comma (e.g. '男，河南话').");
    // zh_typo (byte-based matcher suggests the Chinese item)
    try expectInvalid("极高音调调", true, "Unsupported instruct items found in 极高音调调:\n  '极高音调调' -> '极高音调调' (unsupported; did you mean '极高音调'?)\n\nValid English items: american accent, australian accent, british accent, canadian accent, child, chinese accent, elderly, female, high pitch, indian accent, japanese accent, korean accent, low pitch, male, middle-aged, moderate pitch, portuguese accent, russian accent, teenager, very high pitch, very low pitch, whisper, young adult\nValid Chinese items: 东北话，中年，中音调，云南话，低音调，儿童，四川话，女，宁夏话，少年，极低音调，极高音调，桂林话，河南话，济南话，甘肃话，男，石家庄话，老年，耳语，贵州话，陕西话，青岛话，青年，高音调\n\nTip: Use only English or only Chinese instructs. English instructs should use comma + space (e.g. 'male, indian accent'),\nChinese instructs should use full-width comma (e.g. '男，河南话').");
    // uppercase unknown: the message shows the raw item, then the lowered one
    try expectInvalid("Mole", false, "Unsupported instruct items found in Mole:\n  'Mole' -> 'mole' (unsupported; did you mean 'male'?)\n\nValid English items: american accent, australian accent, british accent, canadian accent, child, chinese accent, elderly, female, high pitch, indian accent, japanese accent, korean accent, low pitch, male, middle-aged, moderate pitch, portuguese accent, russian accent, teenager, very high pitch, very low pitch, whisper, young adult\nValid Chinese items: 东北话，中年，中音调，云南话，低音调，儿童，四川话，女，宁夏话，少年，极低音调，极高音调，桂林话，河南话，济南话，甘肃话，男，石家庄话，老年，耳语，贵州话，陕西话，青岛话，青年，高音调\n\nTip: Use only English or only Chinese instructs. English instructs should use comma + space (e.g. 'male, indian accent'),\nChinese instructs should use full-width comma (e.g. '男，河南话').");
    // multi_unknown (two items, both with suggestions, in input order)
    try expectInvalid("mole, whisperr", false, "Unsupported instruct items found in mole, whisperr:\n  'mole' -> 'mole' (unsupported; did you mean 'male'?)\n  'whisperr' -> 'whisperr' (unsupported; did you mean 'whisper'?)\n\nValid English items: american accent, australian accent, british accent, canadian accent, child, chinese accent, elderly, female, high pitch, indian accent, japanese accent, korean accent, low pitch, male, middle-aged, moderate pitch, portuguese accent, russian accent, teenager, very high pitch, very low pitch, whisper, young adult\nValid Chinese items: 东北话，中年，中音调，云南话，低音调，儿童，四川话，女，宁夏话，少年，极低音调，极高音调，桂林话，河南话，济南话，甘肃话，男，石家庄话，老年，耳语，贵州话，陕西话，青岛话，青年，高音调\n\nTip: Use only English or only Chinese instructs. English instructs should use comma + space (e.g. 'male, indian accent'),\nChinese instructs should use full-width comma (e.g. '男，河南话').");
}

test "normalize: category conflicts (golden error strings)" {
    // conflict_gender
    try expectInvalid("male, female", false, "Conflicting instruct items within the same category: 'male' vs 'female'. Each category (gender, age, pitch, style, accent, dialect) allows at most one item.");
    // conflict_cross_lang (conflict detected after translation)
    try expectInvalid("low pitch, 高音调", false, "Conflicting instruct items within the same category: 'low pitch' vs 'high pitch'. Each category (gender, age, pitch, style, accent, dialect) allows at most one item.");
    // conflict_dup (duplicates count as a conflict)
    try expectInvalid("male, male", false, "Conflicting instruct items within the same category: 'male' vs 'male'. Each category (gender, age, pitch, style, accent, dialect) allows at most one item.");
}

test "normalize: mixed dialect + accent (golden error string)" {
    try expectInvalid("四川话，british accent", false, "Cannot mix Chinese dialect and English accent in a single instruct. Dialects are for Chinese speech, accents for English speech.");
}

test "ratio: bit-exact vs voice_design_ratio" {
    const allocator = std.testing.allocator;
    const Case = struct { a: []const u8, b: []const u8, bits: u32 };
    const cases = [_]Case{
        .{ .a = "moderate pich", .b = "moderate pitch", .bits = 0x3F7684BE },
        .{ .a = "abcabba", .b = "abbabc", .bits = 0x3F1D89D9 },
        .{ .a = "male", .b = "female", .bits = 0x3F4CCCCD },
        .{ .a = "极高音调调", .b = "极高音调", .bits = 0x3F638E39 },
        .{ .a = "whisper", .b = "whisper", .bits = 0x3F800000 },
        .{ .a = "", .b = "", .bits = 0x3F800000 },
        .{ .a = "a", .b = "", .bits = 0x00000000 },
        .{ .a = "qrs", .b = "xyz", .bits = 0x00000000 },
        .{ .a = "whisperr", .b = "whisper", .bits = 0x3F6EEEEF },
        .{ .a = "mole", .b = "male", .bits = 0x3F400000 },
    };
    for (cases) |c| {
        const r = try voicedesign.ratio(allocator, c.a, c.b);
        try std.testing.expectEqual(c.bits, @as(u32, @bitCast(r)));
    }
}

test "hasCjk: CJK Unified Ideographs main range only, permissive decode" {
    try std.testing.expect(voicedesign.hasCjk("男"));
    try std.testing.expect(voicedesign.hasCjk("male 四川话"));
    try std.testing.expect(!voicedesign.hasCjk("male, british accent"));
    try std.testing.expect(!voicedesign.hasCjk(""));
    // Range boundaries: U+4E00 and U+9FFF are in, U+4DFF and U+A000 are out.
    try std.testing.expect(voicedesign.hasCjk("\xE4\xB8\x80")); // U+4E00
    try std.testing.expect(voicedesign.hasCjk("\xE9\xBF\xBF")); // U+9FFF
    try std.testing.expect(!voicedesign.hasCjk("\xE4\xB7\xBF")); // U+4DFF
    try std.testing.expect(!voicedesign.hasCjk("\xEA\x80\x80")); // U+A000
    // Malformed lead byte resyncs one byte at a time, like the reference.
    try std.testing.expect(voicedesign.hasCjk("\xFF男"));
    try std.testing.expect(!voicedesign.hasCjk("\xFF\xFE"));
    // Full-width comma U+FF0C is not CJK.
    try std.testing.expect(!voicedesign.hasCjk("，"));
}

test "vocabularies: sizes, sortedness and distinctness match the std::sets" {
    try std.testing.expectEqual(@as(usize, 23), voicedesign.valid_en.len);
    try std.testing.expectEqual(@as(usize, 25), voicedesign.valid_zh.len);
    try std.testing.expectEqual(@as(usize, 48), voicedesign.all_valid.len);
    for (voicedesign.all_valid[0 .. voicedesign.all_valid.len - 1], voicedesign.all_valid[1..]) |a, b| {
        try std.testing.expect(std.mem.lessThan(u8, a, b)); // strictly sorted = distinct
    }
    // std::set iteration order, spot-checked against the golden error message.
    try std.testing.expectEqualStrings("american accent", voicedesign.valid_en[0]);
    try std.testing.expectEqualStrings("young adult", voicedesign.valid_en[22]);
    try std.testing.expectEqualStrings("东北话", voicedesign.valid_zh[0]);
    try std.testing.expectEqualStrings("高音调", voicedesign.valid_zh[24]);
}
