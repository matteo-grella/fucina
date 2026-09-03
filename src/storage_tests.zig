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
            buf.destroyHeader();
        }
    };
    Hook.calls = 0;

    const buf = try Buffer.fromBorrowedSliceWithRelease(allocator, values[0..], .{ .run = Hook.release });
    buf.retain();
    buf.release();
    try std.testing.expectEqual(@as(usize, 0), Hook.calls);
    buf.release();
    try std.testing.expectEqual(@as(usize, 1), Hook.calls);
    // The hook owns cleanup of the external data; the borrowed slice stays
    // caller-owned and intact here.
    try std.testing.expectEqual(@as(f32, 2), values[1]);
}

test "buffer header carries the accelerator slots only with a provider" {
    // Without a GPU provider nothing can be pending on a buffer, so the four
    // accelerator slots cost no bytes: the header is the allocator, the data
    // slice, the refcount, the release hook, and the host-shadow slot.
    const base = 2 * @sizeOf(usize) + @sizeOf([]f32) + @sizeOf(u32) + @sizeOf(Buffer.Release) + @sizeOf(?*storage.HostShadow);
    if (comptime storage.has_accelerator) {
        try std.testing.expect(@sizeOf(storage.AcceleratorSlots) > 0);
        try std.testing.expect(@sizeOf(Buffer) > std.mem.alignForward(usize, base, @alignOf(Buffer)));
    } else {
        try std.testing.expectEqual(@as(usize, 0), @sizeOf(storage.AcceleratorSlots));
        try std.testing.expectEqual(std.mem.alignForward(usize, base, @alignOf(Buffer)), @sizeOf(Buffer));
        const buf = try Buffer.fromSlice(std.testing.allocator, &.{ 1, 2 });
        defer buf.release();
        // The accessors are inert: nothing is ever pending, no resource exists.
        buf.waitReady();
        buf.waitMutable();
        try std.testing.expect(buf.pending() == null);
        try std.testing.expect(buf.acceleratorResource(.metal) == null);
    }
}

test "host shadow slot is independent of the accelerator slots and dies with the header" {
    const Probe = struct {
        var destroys: usize = 0;
        fn destroy(_: *anyopaque) void {
            destroys += 1;
        }
    };
    Probe.destroys = 0;
    var first: storage.HostShadow = .{ .ctx = @ptrCast(&Probe.destroys), .destroy_fn = Probe.destroy };
    var second: storage.HostShadow = .{ .ctx = @ptrCast(&Probe.destroys), .destroy_fn = Probe.destroy };

    const buf = try Buffer.fromSlice(std.testing.allocator, &.{ 1, 2, 3 });
    try std.testing.expect(buf.hostShadow() == null);
    try std.testing.expect(buf.setHostShadow(&first));
    // A race loser keeps ownership of its own copy.
    try std.testing.expect(!buf.setHostShadow(&second));
    try std.testing.expect(buf.hostShadow() == &first);
    try std.testing.expect(buf.acceleratorResource(.metal) == null);
    buf.release();
    try std.testing.expectEqual(@as(usize, 1), Probe.destroys);
}

test "waitReady completes a pending work exactly once under concurrent readers" {
    // Only a compiled-in provider can attach work; the probe below exercises
    // the claim protocol through the real slots.
    if (comptime !storage.has_accelerator) return error.SkipZigTest;
    // Regression for the parallel-materialize crash: N chunk workers call
    // waitReady on the SAME buffer (copyRangeTo's disjoint-range contract).
    // Only the claimant may dereference — and thereby free — the Work; the
    // pre-claim form (load, ensureHost, clear, release) let a loser touch a
    // Work the winner had already destroyed. The probe Work is heap-owned
    // and freed by its destroy hook, so a stale dereference lands on freed
    // memory and trips the safety checks under the old code.
    const accelerator = @import("accelerator.zig");

    const counters = struct {
        var finishes = std.atomic.Value(u32).init(0);
        var destroys = std.atomic.Value(u32).init(0);
    };
    counters.finishes.store(0, .release);
    counters.destroys.store(0, .release);

    const HeapProbe = struct {
        work: accelerator.Work,
        allocator: std.mem.Allocator,

        const vtable: accelerator.WorkVTable = .{ .finish = finish, .destroy = destroy };

        fn finish(ctx: *anyopaque, copy_to_host: bool) bool {
            _ = ctx;
            _ = copy_to_host;
            // Dwell so every reader is inside waitReady while the claimant
            // completes: the widest possible race window.
            sleepMicros(2_000);
            _ = counters.finishes.fetchAdd(1, .monotonic);
            return true;
        }
        fn destroy(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            _ = counters.destroys.fetchAdd(1, .monotonic);
            self.allocator.destroy(self);
        }
    };

    const allocator = std.testing.allocator;
    const buf = try BufferOf(.f32).fromSlice(allocator, &.{ 1, 2, 3, 4 });
    defer buf.release();

    const probe = try allocator.create(HeapProbe);
    probe.* = .{
        .work = accelerator.Work.init(.metal, probe, &HeapProbe.vtable),
        .allocator = allocator,
    };
    buf.setPending(&probe.work);

    const Reader = struct {
        fn run(target: *BufferOf(.f32)) void {
            target.waitReady();
        }
    };
    var threads: [8]std.Thread = undefined;
    for (&threads) |*t| t.* = try std.Thread.spawn(.{}, Reader.run, .{buf});
    for (threads) |t| t.join();

    try std.testing.expectEqual(@as(u32, 1), counters.finishes.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), counters.destroys.load(.acquire));
    try std.testing.expect(buf.pending() == null);
}

/// Short cross-platform sleep for the dwell above (no std.Thread.sleep in
/// this std; the syscall detour mirrors store/expert_store.zig).
fn sleepMicros(us: u64) void {
    switch (@import("builtin").os.tag) {
        .linux => {
            var req = std.os.linux.timespec{ .sec = @intCast(us / 1_000_000), .nsec = @intCast((us % 1_000_000) * 1000) };
            _ = std.os.linux.nanosleep(&req, null);
        },
        else => {
            const c = struct {
                extern "c" fn nanosleep(req: *const std.c.timespec, rem: ?*std.c.timespec) c_int;
            };
            var req = std.c.timespec{ .sec = @intCast(us / 1_000_000), .nsec = @intCast((us % 1_000_000) * 1000) };
            _ = c.nanosleep(&req, null);
        },
    }
}
