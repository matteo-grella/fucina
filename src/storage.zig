const std = @import("std");
const accelerator = @import("accelerator.zig");
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
        pending_work: std.atomic.Value(?*accelerator.Work) = .init(null),
        pending_use: std.atomic.Value(?*accelerator.Work) = .init(null),
        accelerator_resource: std.atomic.Value(?*accelerator.Resource) = .init(null),

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
                self.discardPending();
                self.waitUnused();
                if (self.release_fn) |release_fn| {
                    release_fn(self.release_ctx.?, self);
                } else {
                    self.destroy();
                }
            }
        }

        pub fn resetRefs(self: *Self) void {
            std.debug.assert(self.pending_work.load(.acquire) == null);
            std.debug.assert(self.pending_use.load(.acquire) == null);
            self.refs.store(1, .release);
        }

        /// Attach one already-submitted accelerator operation to this output.
        /// The buffer owns the Work's initial reference until host access or
        /// final release consumes it.
        pub fn setPending(self: *Self, work: *accelerator.Work) void {
            const old = self.pending_work.cmpxchgStrong(null, work, .release, .acquire);
            std.debug.assert(old == null);
        }

        pub fn pending(self: *const Self) ?*accelerator.Work {
            return self.pending_work.load(.acquire);
        }

        pub fn waitReady(self: *Self) void {
            const work = self.pending_work.load(.acquire) orelse return;
            work.ensureHost();
            if (self.pending_work.cmpxchgStrong(work, null, .acq_rel, .acquire) == null) work.release();
        }

        pub fn discardPending(self: *Self) void {
            const work = self.pending_work.swap(null, .acq_rel) orelse return;
            work.discard();
            work.release();
        }

        /// Record that one already-submitted accelerator command reads this
        /// storage. Provider queue order lets a newer use subsume an older
        /// one; mutable CPU access waits for the latest command. The Work
        /// clears itself on completion, so this reference does not pin an
        /// in-flight slot after a normal host fence.
        pub fn setPendingUse(self: *Self, work: *accelerator.Work) void {
            work.retain();
            if (self.pending_use.swap(work, .acq_rel)) |old| old.release();
        }

        pub fn clearPendingUse(self: *Self, work: *accelerator.Work) void {
            if (self.pending_use.cmpxchgStrong(work, null, .acq_rel, .acquire) == null) work.release();
        }

        /// A mutable host accessor is an eager ordering boundary: all device
        /// readers of the old value must be finished before the caller may
        /// overwrite it. `ensureHost` may also materialize that command's
        /// output on discrete GPUs; mutation is rare enough that correctness
        /// is preferable to a second provider-specific fence protocol.
        pub fn waitUnused(self: *Self) void {
            while (self.pending_use.load(.acquire)) |work| {
                work.ensureHost();
                // Provider finish normally cleared it. Keep this fallback so
                // a future Work implementation cannot leave a stale token.
                self.clearPendingUse(work);
            }
        }

        pub fn waitMutable(self: *Self) void {
            self.waitReady();
            self.waitUnused();
        }

        /// Install a provider cache entry for this backing allocation.  On a
        /// race, the caller keeps ownership of `resource` and must destroy it.
        pub fn setAcceleratorResource(self: *Self, resource: *accelerator.Resource) bool {
            return self.accelerator_resource.cmpxchgStrong(null, resource, .release, .acquire) == null;
        }

        pub fn acceleratorResource(self: *const Self, provider: accelerator.Provider) ?*accelerator.Resource {
            const resource = self.accelerator_resource.load(.acquire) orelse return null;
            return if (resource.provider == provider) resource else null;
        }

        pub fn destroy(self: *Self) void {
            self.discardPending();
            self.waitUnused();
            self.destroyAcceleratorResource();
            self.allocator.free(self.data);
            self.allocator.destroy(self);
        }

        /// Destroy only the storage header. Release hooks that own borrowed
        /// data call this before freeing that data, so accelerator cache
        /// resources are torn down while the wrapped allocation is live.
        pub fn destroyHeader(self: *Self) void {
            self.discardPending();
            self.waitUnused();
            self.destroyAcceleratorResource();
            self.allocator.destroy(self);
        }

        fn destroyAcceleratorResource(self: *Self) void {
            const resource = self.accelerator_resource.swap(null, .acq_rel) orelse return;
            resource.destroy();
        }

        fn releaseBorrowed(_: *anyopaque, self: *Self) void {
            self.destroyHeader();
        }
    };
}

pub const Buffer = BufferOf(.f32);

test {
    _ = @import("storage_tests.zig");
}
