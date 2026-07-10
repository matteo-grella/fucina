//! Behavioral tests for the refcounted owned storage (`storage.zig`):
//! refcount release-once semantics, typed-buffer element dtype,
//! borrowed-slice lifetime (caller keeps ownership after release), and the
//! borrowed-with-release hook (fires exactly once at refs==0).
const std = @import("std");
const dtype_mod = @import("dtype.zig");
const storage = @import("storage.zig");

const Buffer = storage.Buffer;
const BufferOf = storage.BufferOf;

test "buffer refcount releases once" {
    const allocator = std.testing.allocator;
    const buf = try Buffer.fromSlice(allocator, &.{ 1, 2, 3 });
    buf.retain();
    buf.release();
    buf.release();
}

test "typed buffers retain element dtype" {
    const allocator = std.testing.allocator;
    const tokens = try BufferOf(.u16).fromSlice(allocator, &.{ 1, 2, 3 });
    defer tokens.release();

    try std.testing.expect(tokens.data.len == 3);
    try std.testing.expect(@TypeOf(tokens.data[0]) == u16);
}

test "borrowed buffer release leaves caller-owned slice alive" {
    const allocator = std.testing.allocator;
    var values = [_]f32{ 1, 2, 3 };

    const buf = try Buffer.fromBorrowedSlice(allocator, values[0..]);
    try std.testing.expectEqual(@as(usize, 3), buf.data.len);
    buf.data[1] = 20;
    try std.testing.expectEqual(@as(f32, 20), values[1]);

    buf.release();
    try std.testing.expectEqual(@as(f32, 20), values[1]);
}

test "borrowed buffer with release hook fires once at refs==0" {
    const allocator = std.testing.allocator;
    var values = [_]f32{ 1, 2, 3 };

    const Hook = struct {
        var calls: usize = 0;
        fn release(_: *anyopaque, buf: *Buffer) void {
            calls += 1;
            buf.allocator.destroy(buf);
        }
    };
    Hook.calls = 0;

    const buf = try Buffer.fromBorrowedSliceWithRelease(allocator, values[0..], Hook.release);
    buf.retain();
    buf.release();
    try std.testing.expectEqual(@as(usize, 0), Hook.calls);
    buf.release();
    try std.testing.expectEqual(@as(usize, 1), Hook.calls);
    // The hook owns cleanup of the external data; the borrowed slice stays
    // caller-owned and intact here.
    try std.testing.expectEqual(@as(f32, 2), values[1]);
}
