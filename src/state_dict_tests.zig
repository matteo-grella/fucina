//! Tests for state_dict.zig. Focus: `loadStateDict` is TRANSACTIONAL — if any
//! stream entry fails validation, NO destination is mutated. safetensors
//! sorts entries by name, so "a" is validated/committed before "b" in the old
//! one-pass code; the two-pass load must leave "a" byte-unchanged on "b"'s error.
const std = @import("std");
const sd = @import("state_dict.zig");
const Shape = @import("tensor.zig").Shape;

test "loadStateDict is transactional: a mid-stream mismatch leaves prior dests unmutated" {
    const allocator = std.testing.allocator;

    // A 2-entry safetensors stream: "a" {2} and "b" {2}, both valid f32.
    const a_src = [_]f32{ 1.0, 2.0 };
    const b_src = [_]f32{ 3.0, 4.0 };
    const save_entries = [_]sd.NamedTensor{
        .{ .name = "a", .dtype = .f32, .shape = try Shape.init(&[_]usize{2}), .bytes = std.mem.sliceAsBytes(a_src[0..]) },
        .{ .name = "b", .dtype = .f32, .shape = try Shape.init(&[_]usize{2}), .bytes = std.mem.sliceAsBytes(b_src[0..]) },
    };
    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try sd.saveStateDict(allocator, &writer, &save_entries);
    const written = writer.buffered();

    // Destinations pre-seeded with a non-zero sentinel: "a" {2} matches the stream,
    // "b" {3} is a deliberate shape mismatch that fails validation.
    var dst_a = [_]f32{ 9.0, 9.0 };
    var dst_b = [_]f32{ 9.0, 9.0, 9.0 };
    const load_entries = [_]sd.NamedTensorMut{
        .{ .name = "a", .dtype = .f32, .shape = try Shape.init(&[_]usize{2}), .bytes = std.mem.sliceAsBytes(dst_a[0..]) },
        .{ .name = "b", .dtype = .f32, .shape = try Shape.init(&[_]usize{3}), .bytes = std.mem.sliceAsBytes(dst_b[0..]) },
    };
    var reader = std.Io.Reader.fixed(written);
    try std.testing.expectError(sd.Error.CheckpointShapeMismatch, sd.loadStateDict(allocator, &reader, &load_entries, .{}));

    // The transactional guarantee: "a" validated fine and (in the old one-pass code)
    // would already be committed — but the load failed on "b", so dst_a must still
    // hold the sentinel, byte-identical to pre-load.
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9.0, 9.0 }, &dst_a);
}

test "loadStateDict applies an alias map for a renamed field path" {
    const allocator = std.testing.allocator;

    // Save under the OLD field path "enc.w".
    const w_src = [_]f32{ 1, 2, 3, 4 };
    const save_entries = [_]sd.NamedTensor{
        .{ .name = "enc.w", .dtype = .f32, .shape = try Shape.init(&[_]usize{4}), .bytes = std.mem.sliceAsBytes(w_src[0..]) },
    };
    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try sd.saveStateDict(allocator, &writer, &save_entries);
    const written = writer.buffered();

    // Destination registered under the NEW path "encoder.w".
    var dst = [_]f32{ 0, 0, 0, 0 };
    const load_entries = [_]sd.NamedTensorMut{
        .{ .name = "encoder.w", .dtype = .f32, .shape = try Shape.init(&[_]usize{4}), .bytes = std.mem.sliceAsBytes(dst[0..]) },
    };

    // WITH the alias map: enc.w -> encoder.w, strict load succeeds + bytes round-trip.
    {
        var reader = std.Io.Reader.fixed(written);
        try sd.loadStateDict(allocator, &reader, &load_entries, .{
            .aliases = &.{.{ .old = "enc.w", .new = "encoder.w" }},
        });
        try std.testing.expectEqualSlices(f32, &w_src, &dst);
    }

    // WITHOUT the map: the stream name "enc.w" matches no destination -> strict error.
    {
        @memset(&dst, 0);
        var reader = std.Io.Reader.fixed(written);
        try std.testing.expectError(sd.Error.CheckpointUnknownName, sd.loadStateDict(allocator, &reader, &load_entries, .{}));
        try std.testing.expectEqualSlices(f32, &[_]f32{ 0, 0, 0, 0 }, &dst);
    }
}
