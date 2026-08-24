//! Behavioral tests for the `.chain` manifest parser (`chain.zig`): name/spaces/
//! comments/trim/kind parsing, name-default behavior, and the parse error cases.

const std = @import("std");
const chain = @import("chain.zig");

const StageKind = chain.StageKind;
const ParseError = chain.ParseError;
const parse = chain.parse;

test "chain manifest: name, spaces, comments, trim, kinds" {
    const allocator = std.testing.allocator;
    const text =
        \\# my rig
        \\name:  5150 + Mesa OS
        \\
        \\Full Rig Peavey 5150 MXR Mesa OS SM57 - jp_is_out_of_tune.nam :: trim=+2
        \\Tone King Imperial Rhythm Drive, Volume 8, Treble 5,  Bass 4, 1x12 Fender.nam
        \\models/cabs/Mesa OS 4x12 SM57.wav :: trim=-3.5
    ;
    var spec = try parse(allocator, text);
    defer spec.deinit();

    try std.testing.expect(spec.name != null);
    try std.testing.expectEqualStrings("5150 + Mesa OS", spec.name.?);
    try std.testing.expectEqual(@as(usize, 3), spec.stages.len);

    try std.testing.expectEqualStrings("Full Rig Peavey 5150 MXR Mesa OS SM57 - jp_is_out_of_tune.nam", spec.stages[0].path);
    try std.testing.expectEqual(StageKind.nam, spec.stages[0].kind);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), spec.stages[0].trim_db, 1e-6);

    // Spaces, commas, and double-spaces in the path are preserved verbatim.
    try std.testing.expectEqualStrings("Tone King Imperial Rhythm Drive, Volume 8, Treble 5,  Bass 4, 1x12 Fender.nam", spec.stages[1].path);
    try std.testing.expectEqual(StageKind.nam, spec.stages[1].kind);
    try std.testing.expectEqual(@as(f32, 0.0), spec.stages[1].trim_db);

    try std.testing.expectEqualStrings("models/cabs/Mesa OS 4x12 SM57.wav", spec.stages[2].path);
    try std.testing.expectEqual(StageKind.cab, spec.stages[2].kind);
    try std.testing.expectApproxEqAbs(@as(f32, -3.5), spec.stages[2].trim_db, 1e-6);
}

test "chain manifest: name defaults to null, basename a non-name line" {
    const allocator = std.testing.allocator;
    // A '#' mid-filename is part of the path (only leading '#' is a comment).
    var spec = try parse(allocator, "Marshall #2.nam\n");
    defer spec.deinit();
    try std.testing.expect(spec.name == null);
    try std.testing.expectEqual(@as(usize, 1), spec.stages.len);
    try std.testing.expectEqualStrings("Marshall #2.nam", spec.stages[0].path);
}

test "chain manifest: error cases" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(ParseError.EmptyChain, parse(allocator, "# only comments\n\n"));
    try std.testing.expectError(ParseError.UnknownStageType, parse(allocator, "notes.txt\n"));
    try std.testing.expectError(ParseError.UnknownOption, parse(allocator, "amp.nam :: gain=3\n"));
    try std.testing.expectError(ParseError.BadTrim, parse(allocator, "amp.nam :: trim=abc\n"));
}
