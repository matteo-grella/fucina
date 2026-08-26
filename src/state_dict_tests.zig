//! Tests for state_dict.zig. Focus: `loadStateDict` is TRANSACTIONAL — if any
//! stream entry fails validation, NO destination is mutated. safetensors
//! sorts entries by name, so "a" is validated/committed before "b" in the old
//! one-pass code; the two-pass load must leave "a" byte-unchanged on "b"'s error.
const std = @import("std");
const sd = @import("state_dict.zig");
const safetensors = @import("safetensors.zig");
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

/// Write one safetensors frame to a fresh file under cwd and return its path
/// (caller deletes). The file form of the load is exercised through
/// `File.loadMmap`, the way a full-model resume opens its payload.
fn writeFrameFile(buf: []u8, bytes: []const u8) ![]const u8 {
    const io = std.testing.io;
    const path = try std.fmt.bufPrint(buf, "state_dict_test_{d}.safetensors", .{std.Io.Clock.real.now(io).nanoseconds});
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var write_buf: [4096]u8 = undefined;
    var writer = file.writer(io, &write_buf);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
    return path;
}

test "loadStateDictFromFile round-trips through an mmap'd file byte-exactly" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const a_src = [_]f32{ 1.5, -2.25, 3.0 };
    const b_src = [_]f32{ 4.0, 5.0 };
    const save_entries = [_]sd.NamedTensor{
        .{ .name = "enc.a", .dtype = .f32, .shape = try Shape.init(&[_]usize{3}), .bytes = std.mem.sliceAsBytes(a_src[0..]) },
        .{ .name = "enc.b", .dtype = .f32, .shape = try Shape.init(&[_]usize{ 1, 2 }), .bytes = std.mem.sliceAsBytes(b_src[0..]) },
    };
    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try sd.saveStateDict(allocator, &writer, &save_entries);

    var path_buf: [128]u8 = undefined;
    const path = try writeFrameFile(&path_buf, writer.buffered());
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var file = try safetensors.File.loadMmap(allocator, io, path);
    defer file.deinit();

    // Destinations registered in the OPPOSITE order of the stream, under an
    // alias for one of them: name matching and the remap hold on this path.
    var dst_b = [_]f32{ 0, 0 };
    var dst_a = [_]f32{ 0, 0, 0 };
    const load_entries = [_]sd.NamedTensorMut{
        .{ .name = "encoder.b", .dtype = .f32, .shape = try Shape.init(&[_]usize{ 1, 2 }), .bytes = std.mem.sliceAsBytes(dst_b[0..]) },
        .{ .name = "enc.a", .dtype = .f32, .shape = try Shape.init(&[_]usize{3}), .bytes = std.mem.sliceAsBytes(dst_a[0..]) },
    };
    try sd.loadStateDictFromFile(allocator, &file, &load_entries, .{
        .aliases = &.{.{ .old = "enc.b", .new = "encoder.b" }},
    });
    try std.testing.expectEqualSlices(f32, &a_src, &dst_a);
    try std.testing.expectEqualSlices(f32, &b_src, &dst_b);
}

test "loadStateDictFromFile is transactional: a mismatch leaves every destination untouched" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const a_src = [_]f32{ 1.0, 2.0 };
    const b_src = [_]f32{ 3.0, 4.0 };
    const save_entries = [_]sd.NamedTensor{
        .{ .name = "a", .dtype = .f32, .shape = try Shape.init(&[_]usize{2}), .bytes = std.mem.sliceAsBytes(a_src[0..]) },
        .{ .name = "b", .dtype = .f32, .shape = try Shape.init(&[_]usize{2}), .bytes = std.mem.sliceAsBytes(b_src[0..]) },
    };
    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try sd.saveStateDict(allocator, &writer, &save_entries);

    var path_buf: [128]u8 = undefined;
    const path = try writeFrameFile(&path_buf, writer.buffered());
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var file = try safetensors.File.loadMmap(allocator, io, path);
    defer file.deinit();

    // "a" matches; "b" is registered with the wrong shape, so validation
    // fails after "a" has already been checked. The sentinel in "a" must
    // survive: no destination byte moves before every entry validates.
    var dst_a = [_]f32{ 9.0, 9.0 };
    var dst_b = [_]f32{ 9.0, 9.0, 9.0 };
    const shape_mismatch = [_]sd.NamedTensorMut{
        .{ .name = "a", .dtype = .f32, .shape = try Shape.init(&[_]usize{2}), .bytes = std.mem.sliceAsBytes(dst_a[0..]) },
        .{ .name = "b", .dtype = .f32, .shape = try Shape.init(&[_]usize{3}), .bytes = std.mem.sliceAsBytes(dst_b[0..]) },
    };
    try std.testing.expectError(sd.Error.CheckpointShapeMismatch, sd.loadStateDictFromFile(allocator, &file, &shape_mismatch, .{}));
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9.0, 9.0 }, &dst_a);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9.0, 9.0, 9.0 }, &dst_b);

    // Strict missing entry: the file lacks "c", so nothing is written even
    // though "a" and "b" both validate.
    var dst_b2 = [_]f32{ 9.0, 9.0 };
    var dst_c = [_]f32{9.0};
    const missing = [_]sd.NamedTensorMut{
        .{ .name = "a", .dtype = .f32, .shape = try Shape.init(&[_]usize{2}), .bytes = std.mem.sliceAsBytes(dst_a[0..]) },
        .{ .name = "b", .dtype = .f32, .shape = try Shape.init(&[_]usize{2}), .bytes = std.mem.sliceAsBytes(dst_b2[0..]) },
        .{ .name = "c", .dtype = .f32, .shape = try Shape.init(&[_]usize{1}), .bytes = std.mem.sliceAsBytes(dst_c[0..]) },
    };
    try std.testing.expectError(sd.Error.CheckpointMissingEntry, sd.loadStateDictFromFile(allocator, &file, &missing, .{}));
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9.0, 9.0 }, &dst_a);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9.0, 9.0 }, &dst_b2);
}
