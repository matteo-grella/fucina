const std = @import("std");
const dtype_mod = @import("dtype.zig");

const Allocator = std.mem.Allocator;
pub const DType = dtype_mod.DType;

pub fn BufferOf(comptime buffer_dtype: DType) type {
    const Elem = dtype_mod.Storage(buffer_dtype);

    return struct {
        allocator: Allocator,
        data: []Elem,
        refs: std.atomic.Value(u32),
        release_ctx: ?*anyopaque = null,
        release_fn: ?*const fn (*anyopaque, *Self) void = null,

        const Self = @This();
        pub const dtype = buffer_dtype;
        pub const Element = Elem;

        pub fn create(allocator: Allocator, len: usize) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.* = .{
                .allocator = allocator,
                .data = try allocator.alloc(Elem, len),
                .refs = std.atomic.Value(u32).init(1),
            };
            return self;
        }

        pub fn createWithRelease(
            allocator: Allocator,
            len: usize,
            release_ctx: *anyopaque,
            release_fn: *const fn (*anyopaque, *Self) void,
        ) !*Self {
            const self = try create(allocator, len);
            self.release_ctx = release_ctx;
            self.release_fn = release_fn;
            return self;
        }

        pub fn fromSlice(allocator: Allocator, values: []const Elem) !*Self {
            const self = try create(allocator, values.len);
            @memcpy(self.data, values);
            return self;
        }

        pub fn fromBorrowedSlice(allocator: Allocator, values: []Elem) !*Self {
            const self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .data = values,
                .refs = std.atomic.Value(u32).init(1),
            };
            self.release_ctx = @ptrCast(self);
            self.release_fn = releaseBorrowed;
            return self;
        }

        // Borrowed-data variant of `createWithRelease`: `values` stays external
        // and `release_fn` runs once at refs==0 with full cleanup responsibility
        // for both the external data and this header (`allocator.destroy(self)`).
        pub fn fromBorrowedSliceWithRelease(
            allocator: Allocator,
            values: []Elem,
            release_fn: *const fn (*anyopaque, *Self) void,
        ) !*Self {
            const self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .data = values,
                .refs = std.atomic.Value(u32).init(1),
            };
            self.release_ctx = @ptrCast(self);
            self.release_fn = release_fn;
            return self;
        }

        // Explicit-ctx mirror of `fromBorrowedSliceWithRelease` (the borrowed
        // analog of `createWithRelease`): `release_fn` receives `release_ctx`
        // instead of the header itself and keeps full cleanup responsibility
        // for both the external data and this header (`allocator.destroy(self)`).
        pub fn fromBorrowedSliceWithReleaseCtx(
            allocator: Allocator,
            values: []Elem,
            release_ctx: *anyopaque,
            release_fn: *const fn (*anyopaque, *Self) void,
        ) !*Self {
            const self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .data = values,
                .refs = std.atomic.Value(u32).init(1),
            };
            self.release_ctx = release_ctx;
            self.release_fn = release_fn;
            return self;
        }

        pub fn retain(self: *Self) void {
            _ = self.refs.fetchAdd(1, .monotonic);
        }

        // Snapshot only. Use for ownership-transfer APIs when the caller already
        // has exclusive access to the Tensor handle that points at this buffer.
        pub fn isUnique(self: *const Self) bool {
            return self.refs.load(.acquire) == 1;
        }

        pub fn release(self: *Self) void {
            const old = self.refs.fetchSub(1, .acq_rel);
            std.debug.assert(old > 0);
            if (old == 1) {
                if (self.release_fn) |release_fn| {
                    release_fn(self.release_ctx.?, self);
                } else {
                    self.destroy();
                }
            }
        }

        pub fn resetRefs(self: *Self) void {
            self.refs.store(1, .release);
        }

        pub fn destroy(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.destroy(self);
        }

        fn releaseBorrowed(_: *anyopaque, self: *Self) void {
            self.allocator.destroy(self);
        }
    };
}

pub const Buffer = BufferOf(.f32);

test {
    _ = @import("storage_tests.zig");
}
