//! Golden-parity tests for the OmniVoice language map (`langmap.zig`) vs the
//! C++ reference (refs/omnivoice.cpp/src/lang-map.h). resolve_language
//! goldens were produced by a harness compiled against the reference header
//! (scratchpad vd_lang_golden.cpp); the table itself is pinned verbatim by a
//! SHA-256 over "name\x00id\n" per entry, computed from the header by the
//! extraction script. All comparisons are byte-exact.

const std = @import("std");
const langmap = @import("langmap.zig");

test "table: verbatim vs the reference header (count + hash + endpoints)" {
    try std.testing.expectEqual(@as(usize, 646), langmap.lang_name_to_id_table.len);
    try std.testing.expectEqualStrings("abadi", langmap.lang_name_to_id_table[0][0]);
    try std.testing.expectEqualStrings("kbt", langmap.lang_name_to_id_table[0][1]);
    try std.testing.expectEqualStrings("ömie", langmap.lang_name_to_id_table[645][0]);
    try std.testing.expectEqualStrings("aom", langmap.lang_name_to_id_table[645][1]);

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    for (langmap.lang_name_to_id_table) |entry| {
        hasher.update(entry[0]);
        hasher.update("\x00");
        hasher.update(entry[1]);
        hasher.update("\n");
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    var hex: [64]u8 = undefined;
    for (digest, 0..) |b, i| {
        _ = std.fmt.bufPrint(hex[i * 2 ..][0..2], "{x:0>2}", .{b}) catch unreachable;
    }
    try std.testing.expectEqualStrings(
        "429e49f0a5bebc6afa049848cb421251245b116cf70bc6677dfe4b78423d5092",
        &hex,
    );
}

test "resolveLanguage matches the reference goldens" {
    const cases = [_][2][]const u8{
        .{ "English", "en" },
        .{ "english", "en" },
        .{ "en", "en" }, // ISO-id passthrough
        .{ "EN", "" }, // ids are case-sensitive, "en" is not a name
        .{ "None", "" },
        .{ "nOnE", "" },
        .{ "none", "" },
        .{ "cantonese", "yue" },
        .{ "Cantonese", "yue" },
        .{ "yue", "yue" },
        .{ "Klingon", "" },
        .{ "", "" },
        .{ "cen", "cen" }, // both a name and an id; passthrough wins
        .{ "Võro", "vro" }, // ASCII-only lowering leaves the õ byte intact
        .{ "VÕRO", "" }, // Õ is not ASCII-lowered, so the name misses
        .{ "min nan chinese", "nan" },
        .{ "nan", "nan" },
        .{ "NAN", "" },
    };
    for (cases) |c| {
        try std.testing.expectEqualStrings(c[1], langmap.resolveLanguage(c[0]));
    }
}

test "resolveLanguage: ISO-id passthrough returns the original slice" {
    const input = "yue";
    const out = langmap.resolveLanguage(input);
    try std.testing.expectEqual(@intFromPtr(input.ptr), @intFromPtr(out.ptr));
    try std.testing.expectEqual(input.len, out.len);
}
