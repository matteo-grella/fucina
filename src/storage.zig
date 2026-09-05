//! Reference-counted byte storage backing raw tensors: allocation,
//! retain/release, and the borrowed/owned distinction views rely on.
//! Layer stack: docs/ARCHITECTURE.md.
const std = @import("std");
const build_options = @import("build_options");
const accelerator = @import("accelerator.zig");
const dtype_mod = @import("dtype.zig");

const Allocator = std.mem.Allocator;
pub const DType = dtype_mod.DType;

/// True when a GPU provider is compiled in: only then can a buffer carry
/// submitted accelerator work or a provider cache entry.
pub const has_accelerator = build_options.use_gpu;

/// One accelerator Work reference behind a claim. The slot owns exactly
/// one reference to the Work it holds, and every read that will
/// dereference the pointer goes through `acquire`, which hands back a
/// RETAINED reference under the claim; only the boolean observer
/// `hasPending` reads the pointer bare. The mutators (`install`, `take`,
/// `clear`) take the same claim and release a displaced or cleared Work
/// only after dropping it, so no reader can hold a Work whose last
/// reference another thread is releasing. The pre-claim form (load the
/// pointer, complete it, clear, release) let a reader on one buffer
/// dereference a Work that the reader of ANOTHER buffer — the output side
/// of the same command — had already completed and freed.
pub const WorkSlot = struct {
    ptr: std.atomic.Value(?*accelerator.Work) = .init(null),
    claim: std.atomic.Value(bool) = .init(false),

    /// A Work is held (no dereference; the answer may be stale by the
    /// time the caller acts on it).
    pub fn hasPending(self: *const WorkSlot) bool {
        return self.ptr.load(.acquire) != null;
    }

    /// The held Work with one reference retained for the caller, who
    /// releases it; null when the slot is empty (checked bare first, so
    /// the common empty case takes no claim).
    pub fn acquire(self: *WorkSlot) ?*accelerator.Work {
        if (self.ptr.load(.acquire) == null) return null;
        self.lock();
        defer self.unlock();
        const work = self.ptr.load(.monotonic) orelse return null;
        work.retain();
        return work;
    }

    /// Put `work` in the slot, moving one reference from the caller to
    /// the slot; the displaced Work, if any, comes back with the slot's
    /// former reference for the caller to release.
    pub fn install(self: *WorkSlot, work: *accelerator.Work) ?*accelerator.Work {
        self.lock();
        defer self.unlock();
        return self.ptr.swap(work, .acq_rel);
    }

    /// Empty the slot; the caller receives the slot's reference.
    pub fn take(self: *WorkSlot) ?*accelerator.Work {
        if (self.ptr.load(.acquire) == null) return null;
        self.lock();
        defer self.unlock();
        return self.ptr.swap(null, .acq_rel);
    }

    /// Empty the slot iff it still holds `expected`, releasing the slot's
    /// reference (after the claim is dropped: a release may complete the
    /// Work and clear other slots); true when it did.
    pub fn clear(self: *WorkSlot, expected: *accelerator.Work) bool {
        {
            self.lock();
            defer self.unlock();
            if (self.ptr.cmpxchgStrong(expected, null, .acq_rel, .acquire) != null) return false;
        }
        expected.release();
        return true;
    }

    fn lock(self: *WorkSlot) void {
        while (self.claim.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn unlock(self: *WorkSlot) void {
        self.claim.store(false, .release);
    }
};

/// The accelerator lifetime slots of a buffer header: the pending output
/// Work, the latest device reader, and the provider cache Resource. Only a
/// compiled-in provider can fill them, so without one
/// (`has_accelerator == false`) the type is empty: zero bytes on every
/// buffer, and the accessors below are inert.
pub const AcceleratorSlots = if (has_accelerator) struct {
    pending_work: WorkSlot = .{},
    pending_use: WorkSlot = .{},
    resource: std.atomic.Value(?*accelerator.Resource) = .init(null),
} else struct {};

/// A host-side derived copy tied to one allocation's lifetime (the widen-once
/// f32 weight shadow of `exec/matmul.zig`). Its own slot on every buffer,
/// independent of the accelerator slots: whoever installs it owns the
/// destroy hook, which the buffer runs when the header dies.
pub const HostShadow = struct {
    ctx: *anyopaque,
    destroy_fn: *const fn (ctx: *anyopaque) void,

    pub fn destroy(self: *HostShadow) void {
        self.destroy_fn(self.ctx);
    }
};

pub fn BufferOf(comptime buffer_dtype: DType) type {
    const Elem = dtype_mod.Storage(buffer_dtype);

    return struct {
        allocator: Allocator,
        data: []Elem,
        refs: std.atomic.Value(u32),
        /// Runs once at refs == 0 in place of `destroy`, with full cleanup
        /// responsibility for the data and this header; `run == null` means
        /// the plain `destroy`.
        release_hook: Release = .{},
        accel: AcceleratorSlots = .{},
        host_shadow: std.atomic.Value(?*HostShadow) = .init(null),

        const Self = @This();
        pub const dtype = buffer_dtype;
        pub const Element = Elem;

        /// A release hook: `run(ctx, buffer)` replaces `destroy` when the last
        /// reference drops and must free the data and the header
        /// (`destroyHeader`). A null `ctx` means the buffer itself.
        pub const Release = struct {
            ctx: ?*anyopaque = null,
            run: ?*const fn (*anyopaque, *Self) void = null,
        };

        fn bind(self: *Self, hook: Release) void {
            std.debug.assert(hook.run != null);
            self.release_hook = .{ .ctx = hook.ctx orelse @ptrCast(self), .run = hook.run };
        }

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

        /// Owned data with a release hook (the buffer pool's recycling).
        pub fn createWithRelease(allocator: Allocator, len: usize, hook: Release) !*Self {
            const self = try create(allocator, len);
            self.bind(hook);
            return self;
        }

        pub fn fromSlice(allocator: Allocator, values: []const Elem) !*Self {
            const self = try create(allocator, values.len);
            @memcpy(self.data, values);
            return self;
        }

        /// Borrowed data: `values` stays external and is never freed here; the
        /// header is destroyed when the last reference drops.
        pub fn fromBorrowedSlice(allocator: Allocator, values: []Elem) !*Self {
            return fromBorrowedSliceWithRelease(allocator, values, .{ .run = releaseBorrowed });
        }

        /// Borrowed data with a release hook that owns the cleanup of both the
        /// external data and this header (device-resident weight bytes, the
        /// buffer pool's slabs).
        pub fn fromBorrowedSliceWithRelease(allocator: Allocator, values: []Elem, hook: Release) !*Self {
            const self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .data = values,
                .refs = std.atomic.Value(u32).init(1),
            };
            self.bind(hook);
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
                if (self.release_hook.run) |run| {
                    run(self.release_hook.ctx.?, self);
                } else {
                    self.destroy();
                }
            }
        }

        pub fn resetRefs(self: *Self) void {
            if (comptime has_accelerator) {
                std.debug.assert(!self.accel.pending_work.hasPending());
                std.debug.assert(!self.accel.pending_use.hasPending());
            }
            self.refs.store(1, .release);
        }

        // The accelerator accessors keep one signature in every build. With
        // no provider compiled in (`has_accelerator == false`) the waits and
        // queries are inert (nothing can be pending) and the setters are
        // unreachable (nothing can produce a Work or a Resource).

        /// Attach one already-submitted accelerator operation to this output.
        /// The buffer owns the Work's initial reference until host access or
        /// final release consumes it.
        pub fn setPending(self: *Self, work: *accelerator.Work) void {
            if (comptime !has_accelerator) unreachable;
            const displaced = self.accel.pending_work.install(work);
            std.debug.assert(displaced == null);
        }

        /// A submitted accelerator output is still attached (a bare
        /// observation: nothing is dereferenced).
        pub fn hasPending(self: *const Self) bool {
            return if (comptime has_accelerator) self.accel.pending_work.hasPending() else false;
        }

        /// The pending output Work with one reference retained for the
        /// caller (who releases it), or null. The only way to reach the
        /// Work behind a buffer: a provider that wants to consume the
        /// device result of an earlier command holds this reference as its
        /// dependency.
        pub fn acquirePending(self: *const Self) ?*accelerator.Work {
            if (comptime !has_accelerator) return null;
            const atomics: *Self = @constCast(self);
            return atomics.accel.pending_work.acquire();
        }

        /// Block until any pending accelerator output is host-visible.
        ///
        /// Safe under CONCURRENT callers (`copyRangeTo`'s disjoint-range
        /// contract puts parallel chunk workers here on the same buffer):
        /// every caller acquires its own retained reference, completes the
        /// Work (the state machine runs the provider's finish once; the
        /// others wait on it and return only after the host copy is
        /// visible), clears the slot if it still holds that Work, and
        /// drops its reference. The Work outlives every caller that
        /// acquired it, whichever buffer's side frees it.
        ///
        /// Takes `*const`: the wait entries move only the slots' atomics,
        /// so a read-only accessor fences without a cast; the one cast
        /// lives here.
        pub fn waitReady(self: *const Self) void {
            if (comptime !has_accelerator) return;
            const atomics: *Self = @constCast(self);
            while (atomics.accel.pending_work.acquire()) |work| {
                work.ensureHost();
                _ = atomics.accel.pending_work.clear(work);
                work.release();
            }
        }

        pub fn discardPending(self: *Self) void {
            if (comptime !has_accelerator) return;
            const work = self.accel.pending_work.take() orelse return;
            work.discard();
            work.release();
        }

        /// Record that one already-submitted accelerator command reads this
        /// storage. Provider queue order lets a newer use subsume an older
        /// one; mutable CPU access waits for the latest command. The Work
        /// clears itself on completion, so this reference does not pin an
        /// in-flight slot after a normal host fence.
        pub fn setPendingUse(self: *Self, work: *accelerator.Work) void {
            if (comptime !has_accelerator) unreachable;
            work.retain();
            if (self.accel.pending_use.install(work)) |old| old.release();
        }

        pub fn clearPendingUse(self: *Self, work: *accelerator.Work) void {
            if (comptime !has_accelerator) unreachable;
            _ = self.accel.pending_use.clear(work);
        }

        /// A mutable host accessor is an eager ordering boundary: all device
        /// readers of the old value must be finished before the caller may
        /// overwrite it. Same acquire/complete/clear/release shape as
        /// `waitReady`, on the reader slot: the command is finished
        /// host-visibly if still pending (its output may be read later),
        /// and a Work whose output was already discarded counts as
        /// finished. Mutation is rare enough that this is preferable to a
        /// second provider-specific fence protocol.
        pub fn waitUnused(self: *const Self) void {
            if (comptime !has_accelerator) return;
            const atomics: *Self = @constCast(self);
            while (atomics.accel.pending_use.acquire()) |work| {
                work.ensureFinished();
                // Provider finish normally cleared it. Keep this fallback so
                // a Work implementation cannot leave a stale token.
                _ = atomics.accel.pending_use.clear(work);
                work.release();
            }
        }

        pub fn waitMutable(self: *const Self) void {
            self.waitReady();
            self.waitUnused();
        }

        /// Install a provider cache entry for this backing allocation.  On a
        /// race, the caller keeps ownership of `resource` and must destroy it.
        pub fn setAcceleratorResource(self: *Self, resource: *accelerator.Resource) bool {
            if (comptime !has_accelerator) unreachable;
            return self.accel.resource.cmpxchgStrong(null, resource, .release, .acquire) == null;
        }

        pub fn acceleratorResource(self: *const Self, provider: accelerator.Provider) ?*accelerator.Resource {
            if (comptime !has_accelerator) return null;
            const resource = self.accel.resource.load(.acquire) orelse return null;
            return if (resource.provider == provider) resource else null;
        }

        /// Install the host-side shadow of this allocation.  On a race, the
        /// caller keeps ownership of `shadow` and must destroy it.
        pub fn setHostShadow(self: *Self, shadow: *HostShadow) bool {
            return self.host_shadow.cmpxchgStrong(null, shadow, .release, .acquire) == null;
        }

        pub fn hostShadow(self: *const Self) ?*HostShadow {
            return self.host_shadow.load(.acquire);
        }

        pub fn destroy(self: *Self) void {
            self.discardPending();
            self.waitUnused();
            self.destroyAttachments();
            self.allocator.free(self.data);
            self.allocator.destroy(self);
        }

        /// Destroy only the storage header. Release hooks that own borrowed
        /// data call this before freeing that data, so accelerator cache
        /// resources are torn down while the wrapped allocation is live.
        pub fn destroyHeader(self: *Self) void {
            self.discardPending();
            self.waitUnused();
            self.destroyAttachments();
            self.allocator.destroy(self);
        }

        /// The provider cache entry and the host shadow die with the header.
        fn destroyAttachments(self: *Self) void {
            if (comptime has_accelerator) {
                if (self.accel.resource.swap(null, .acq_rel)) |resource| resource.destroy();
            }
            if (self.host_shadow.swap(null, .acq_rel)) |shadow| shadow.destroy();
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
