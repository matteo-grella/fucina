//! Tests for duration.zig against golden outputs of the C++ reference
//! (refs/omnivoice.cpp/src/duration-estimator.h). Goldens were produced by a
//! standalone harness (scratchpad golden_text.cpp, clang++ -std=c++17 -O2)
//! that prints f32 results as u32 bit patterns. Weight sums and estimates
//! are asserted bit-exact; the powf boost path is asserted within 1 ulp
//! because std.math.pow(f32) and Apple libm powf may legitimately differ in
//! the last bit.

const std = @import("std");
const duration = @import("duration.zig");

fn bits(x: f32) u32 {
    return @bitCast(x);
}

/// 1-ulp tolerance for libm-backed results (powf). Everything else is
/// bit-exact.
fn expectWithin1Ulp(expected_bits: u32, got: f32) !void {
    const got_bits: u32 = @bitCast(got);
    if (expected_bits == got_bits) return;
    // Same sign and adjacent representable values.
    const diff = if (expected_bits > got_bits) expected_bits - got_bits else got_bits - expected_bits;
    if (diff > 1) {
        std.debug.print("expected 0x{X:0>8} got 0x{X:0>8}\n", .{ expected_bits, got_bits });
        return error.TestExpectedApproxEq;
    }
}

const WeightCase = struct {
    text: []const u8,
    weight_bits: u32,
};

const weight_cases = [_]WeightCase{
    .{ .text = "Nice to meet you.", .weight_bits = 0x41619999 }, // 14.0999994
    .{ .text = "The quick brown fox jumps over the lazy dog, 42 times!", .weight_bits = 0x42480001 }, // 50.0000038
    .{ .text = "\xE4\xBB\x8A\xE5\xA4\xA9\xE5\xA4\xA9\xE6\xB0\x94\xE5\xBE\x88\xE5\xA5\xBD\xE3\x80\x82\xE6\x88\x91\xE4\xBB\xAC\xE5\x8E\xBB\xE5\x85\xAC\xE5\x9B\xAD\xE6\x95\xA3\xE6\xAD\xA5\xE5\x90\xA7\xEF\xBC\x81", .weight_bits = 0x422C0000 }, // 43
    .{ .text = "\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF\xE3\x80\x81\xE4\xB8\x96\xE7\x95\x8C\xEF\xBC\x81\xE3\x82\xAB\xE3\x82\xBF\xE3\x82\xAB\xE3\x83\x8A\xE3\x82\x82", .weight_bits = 0x41E80002 }, // 29.0000038
    .{ .text = "\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD9\x8B\xD8\xA7 \xD8\xA8\xD8\xA7\xD9\x84\xD9\x80\xD9\x80\xD8\xB9\xD8\xA7\xD9\x84\xD9\x85", .weight_bits = 0x4191999A }, // 18.2000008 (incl. Tatweel = mark)
    .{ .text = "1234567890 12.5% \xD9\xA1\xD9\xA2\xD9\xA3", .weight_bits = 0x4265999A }, // 57.4000015
    .{ .text = "emoji \xF0\x9F\x98\x80\xF0\x9F\x8E\x89 and \xF0\xA0\x80\x80 done", .weight_bits = 0x416CCCCC }, // 14.7999992 (symbols + CJK Ext B)
    .{ .text = "\xEC\x95\x88\xEB\x85\x95\xED\x95\x98\xEC\x84\xB8\xEC\x9A\x94 \xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1\xE0\xB8\xAA\xE0\xB8\x94\xE0\xB8\xB5 \xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D", .weight_bits = 0x41C73334 }, // 24.9000015 (hangul + thai + hebrew)
    .{ .text = "bad \xFF\x80 bytes", .weight_bits = 0x41066666 }, // 8.39999962 (malformed bytes skip 1)
    .{ .text = "", .weight_bits = 0x00000000 }, // 0
};

test "totalWeight matches the C++ reference bit for bit" {
    for (weight_cases) |case| {
        try std.testing.expectEqual(case.weight_bits, bits(duration.totalWeight(case.text)));
    }
}

// f32 sums of getCharWeight over each 0x10000-codepoint block of
// [0, 0x110000), accumulated in codepoint order. Bit-equality with the C++
// harness proves both tables (879 category ranges + 88 script ranges), the
// binary searches and the check order are transcribed exactly.
const block_sums = [17]u32{
    0x481DF064, // block  0 sum 161729.562
    0x477F3480, // block  1 sum 65332.5
    0x483FFF80, // block  2 sum 196606
    0x48400000, // block  3 sum 196608
    0x48400000, // block  4 sum 196608
    0x48400000, // block  5 sum 196608
    0x48400000, // block  6 sum 196608
    0x48400000, // block  7 sum 196608
    0x48400000, // block  8 sum 196608
    0x48400000, // block  9 sum 196608
    0x48400000, // block 10 sum 196608
    0x48400000, // block 11 sum 196608
    0x48400000, // block 12 sum 196608
    0x48400000, // block 13 sum 196608
    0x483F4C00, // block 14 sum 195888
    0x48400000, // block 15 sum 196608
    0x48400000, // block 16 sum 196608
};

test "getCharWeight sweep over all of [0, 0x110000) matches C++ block sums bit for bit" {
    for (block_sums, 0..) |expected, block| {
        var sum: f32 = 0.0;
        var cp: u32 = @intCast(block * 0x10000);
        const end_cp: u32 = cp + 0x10000;
        while (cp < end_cp) : (cp += 1) {
            sum += duration.getCharWeight(cp);
        }
        try std.testing.expectEqual(expected, bits(sum));
    }
}

const EstimateCase = struct {
    target: []const u8,
    ref: []const u8,
    ref_dur_bits: u32,
    est_bits: u32,
    boost: bool, // linear estimate below the 50.0 threshold => powf path
};

const estimate_cases = [_]EstimateCase{
    .{ .target = "The quick brown fox jumps over the lazy dog, 42 times!", .ref = "Nice to meet you.", .ref_dur_bits = 0x41C80000, .est_bits = 0x42B14E14, .boost = false }, // 88.6524963
    .{ .target = "Hi.", .ref = "Nice to meet you.", .ref_dur_bits = 0x41C80000, .est_bits = 0x41B25B68, .boost = true }, // 22.294632
    .{ .target = "Ok", .ref = "Nice to meet you.", .ref_dur_bits = 0x41C80000, .est_bits = 0x41A59270, .boost = true }, // 20.6965027
    .{ .target = "\xE4\xBD\xA0\xE5\xA5\xBD", .ref = "Nice to meet you.", .ref_dur_bits = 0x41C80000, .est_bits = 0x41EECBD3, .boost = true }, // 29.8495235
    .{ .target = "A much longer target text that should comfortably exceed the low threshold of fifty units in the estimate.", .ref = "Nice to meet you.", .ref_dur_bits = 0x41C80000, .est_bits = 0x4322F179, .boost = false }, // 162.943253
    .{ .target = "same text", .ref = "same text", .ref_dur_bits = 0x42A00000, .est_bits = 0x42A00000, .boost = false }, // 80
    .{ .target = "anything", .ref = "", .ref_dur_bits = 0x41C80000, .est_bits = 0x00000000, .boost = false }, // 0: empty ref
    .{ .target = "anything", .ref = "ref text", .ref_dur_bits = 0x00000000, .est_bits = 0x00000000, .boost = false }, // 0: ref_duration <= 0
    .{ .target = "target", .ref = "\xCC\x80\xCC\x81", .ref_dur_bits = 0x41200000, .est_bits = 0x00000000, .boost = false }, // 0: marks-only ref weight 0
};

test "estimate matches the C++ reference (bit-exact; boost path within 1 ulp)" {
    for (estimate_cases) |case| {
        const got = duration.estimate(
            case.target,
            case.ref,
            @bitCast(case.ref_dur_bits),
            duration.default_low_threshold,
            duration.default_boost_strength,
        );
        if (case.boost) {
            try expectWithin1Ulp(case.est_bits, got);
        } else {
            try std.testing.expectEqual(case.est_bits, bits(got));
        }
    }
}

const TokensCase = struct {
    target: []const u8,
    ref: []const u8,
    ref_tokens: i32,
    expected: i32,
};

const tokens_cases = [_]TokensCase{
    .{ .target = "The quick brown fox jumps over the lazy dog, 42 times!", .ref = "", .ref_tokens = 0, .expected = 88 },
    .{ .target = "Hi.", .ref = "", .ref_tokens = 0, .expected = 22 },
    .{ .target = "", .ref = "", .ref_tokens = 0, .expected = 1 },
    .{ .target = "\xE4\xBB\x8A\xE5\xA4\xA9\xE5\xA4\xA9\xE6\xB0\x94\xE5\xBE\x88\xE5\xA5\xBD\xE3\x80\x82", .ref = "", .ref_tokens = 0, .expected = 43 },
    .{ .target = "A longer English sentence used as the synthesis target for token estimation purposes.", .ref = "Nice to meet you.", .ref_tokens = 25, .expected = 132 },
    .{ .target = "Short", .ref = "A reference transcript that is quite a bit longer than the target.", .ref_tokens = 512, .expected = 48 },
    // Non-positive ref_tokens falls back to the anchor even with ref text.
    .{ .target = "anything", .ref = "ref", .ref_tokens = -3, .expected = 32 },
    .{ .target = "\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD9\x8B\xD8\xA7 \xD8\xA8\xD8\xA7\xD9\x84\xD8\xB9\xD8\xA7\xD9\x84\xD9\x85", .ref = "", .ref_tokens = 0, .expected = 43 },
};

test "estimateTokens matches the C++ reference exactly" {
    // The truncation of the token count absorbs a potential final-bit powf
    // difference on the boost path for every golden here (none sits on an
    // integer boundary), so the integer results are asserted exactly.
    for (tokens_cases) |case| {
        try std.testing.expectEqual(case.expected, duration.estimateTokens(case.target, case.ref, case.ref_tokens));
    }
}

test "utf8Decode mirrors the reference (malformed => 0, truncated => 0)" {
    var cp: u32 = undefined;

    try std.testing.expectEqual(@as(usize, 1), duration.utf8Decode("a", &cp));
    try std.testing.expectEqual(@as(u32, 'a'), cp);

    try std.testing.expectEqual(@as(usize, 2), duration.utf8Decode("\xC2\xA0", &cp));
    try std.testing.expectEqual(@as(u32, 0xA0), cp);

    try std.testing.expectEqual(@as(usize, 3), duration.utf8Decode("\xE4\xBD\xA0", &cp));
    try std.testing.expectEqual(@as(u32, 0x4F60), cp);

    try std.testing.expectEqual(@as(usize, 4), duration.utf8Decode("\xF0\x9F\x98\x80", &cp));
    try std.testing.expectEqual(@as(u32, 0x1F600), cp);

    try std.testing.expectEqual(@as(usize, 0), duration.utf8Decode("", &cp));
    try std.testing.expectEqual(@as(usize, 0), duration.utf8Decode("\xFF", &cp));
    try std.testing.expectEqual(@as(usize, 0), duration.utf8Decode("\x80", &cp));
    // Truncated multi-byte sequences are malformed (no clipping).
    try std.testing.expectEqual(@as(usize, 0), duration.utf8Decode("\xE4\xBD", &cp));
    try std.testing.expectEqual(@as(usize, 0), duration.utf8Decode("\xF0\x9F\x98", &cp));
}
