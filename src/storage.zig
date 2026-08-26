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

/// The accelerator lifetime slots of a buffer header: the pending output
/// Work, the latest device reader, the provider cache Resource, and the
/// completion claim of `waitReady`. Only a compiled-in provider can fill
/// them, so without one (`has_accelerator == false`) the type is empty:
/// zero bytes on every buffer, and the accessors below are inert.
pub const AcceleratorSlots = if (has_accelerator) struct {
    pending_work: std.atomic.Value(?*accelerator.Work) = .init(null),
    pending_use: std.atomic.Value(?*accelerator.Work) = .init(null),
    resource: std.atomic.Value(?*accelerator.Resource) = .init(null),
    /// Exclusive completion claim for `waitReady`: only the claim holder
    /// may dereference (and release) `pending_work` — see `waitReady`.
    pending_claim: std.atomic.Value(bool) = .init(false),
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
        release_ctx: ?*anyopaque = null,
        release_fn: ?*const fn (*anyopaque, *Self) void = null,
        accel: AcceleratorSlots = .{},
        host_shadow: std.atomic.Value(?*HostShadow) = .init(null),

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
            if (comptime has_accelerator) {
                std.debug.assert(self.accel.pending_work.load(.acquire) == null);
                std.debug.assert(self.accel.pending_use.load(.acquire) == null);
                std.debug.assert(!self.accel.pending_claim.load(.acquire));
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
            const old = self.accel.pending_work.cmpxchgStrong(null, work, .release, .acquire);
            std.debug.assert(old == null);
        }

        pub fn pending(self: *const Self) ?*accelerator.Work {
            return if (comptime has_accelerator) self.accel.pending_work.load(.acquire) else null;
        }

        /// Block until any pending accelerator output is host-visible.
        ///
        /// Safe under CONCURRENT callers (`copyRangeTo`'s disjoint-range
        /// contract puts parallel chunk workers here on the same buffer): a
        /// single claimant dereferences the Work, completes it, clears the
        /// slot, and drops the buffer's reference; everyone else spins until
        /// the slot clears — which the claimant does only AFTER the host
        /// copy is visible. The pre-claim naive form (load → ensureHost →
        /// clear → release) let a loser dereference a Work the winner had
        /// already freed.
        ///
        /// Takes `*const`: the wait entries move only the atomic fields
        /// (`pending_work`, `pending_claim`, `pending_use`), so a read-only
        /// accessor fences without a cast; the one cast lives here.
        pub fn waitReady(self: *const Self) void {
            if (comptime !has_accelerator) return;
            const atomics: *Self = @constCast(self);
            while (true) {
                if (self.accel.pending_work.load(.acquire) == null) return;
                if (atomics.accel.pending_claim.cmpxchgWeak(false, true, .acq_rel, .acquire) != null) {
                    std.atomic.spinLoopHint();
                    continue;
                }
                // Re-read under the claim: a previous claimant may have
                // completed and freed the work after our gate load.
                const work = self.accel.pending_work.load(.acquire) orelse {
                    atomics.accel.pending_claim.store(false, .release);
                    return;
                };
                work.ensureHost();
                const displaced = atomics.accel.pending_work.cmpxchgStrong(work, null, .acq_rel, .acquire);
                std.debug.assert(displaced == null); // sole clearer while claimed
                atomics.accel.pending_claim.store(false, .release);
                work.release();
                return;
            }
        }

        pub fn discardPending(self: *Self) void {
            if (comptime !has_accelerator) return;
            const work = self.accel.pending_work.swap(null, .acq_rel) orelse return;
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
            if (self.accel.pending_use.swap(work, .acq_rel)) |old| old.release();
        }

        pub fn clearPendingUse(self: *Self, work: *accelerator.Work) void {
            if (comptime !has_accelerator) unreachable;
            if (self.accel.pending_use.cmpxchgStrong(work, null, .acq_rel, .acquire) == null) work.release();
        }

        /// A mutable host accessor is an eager ordering boundary: all device
        /// readers of the old value must be finished before the caller may
        /// overwrite it. `ensureHost` may also materialize that command's
        /// output on discrete GPUs; mutation is rare enough that correctness
        /// is preferable to a second provider-specific fence protocol.
        pub fn waitUnused(self: *const Self) void {
            if (comptime !has_accelerator) return;
            const atomics: *Self = @constCast(self);
            while (self.accel.pending_use.load(.acquire)) |work| {
                work.ensureHost();
                // Provider finish normally cleared it. Keep this fallback so
                // a future Work implementation cannot leave a stale token.
                atomics.clearPendingUse(work);
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
