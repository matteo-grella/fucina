//! Reusable transient-buffer pool for the eager runtime.
//!
//! Owned, refcounted storage buffers are recycled across ops to avoid per-op
//! allocation churn (see docs/MEMORY-MODEL.md). Two arms share one byte budget:
//! the f32 arm (a free list of `*storage.Buffer`, the dominant transient
//! path) and a byte-granular slab arm serving every other storage dtype
//! (`acquireTyped`) plus non-DType packed block scratch (`acquireScratch`).
//! This is a LEAF module: it never references the ExecContext/Runtime, so
//! anything may depend on it without forming an import cycle (arch-check
//! 0-SCC invariant).

const std = @import("std");
const storage = @import("../storage.zig");
const thread = @import("../thread.zig");

const Allocator = std.mem.Allocator;

/// Alignment of every slab in the byte arm. Covers the max `@alignOf` of all
/// Storage/packed scratch element types (8) with cache-line headroom (the MoE
/// i128-align precedent argues against cutting this fine).
pub const slab_align = 64;

/// Byte-size rounding quantum for slabs. Must stay greater than the largest
/// `Storage(dtype)` element size (currently 292: BlockQ8_K) — the capacity
/// recovery in `reclaimTypedFor` rounds a typed view's byte length back up to
/// the slab's true capacity and is exact only under that bound.
pub const slab_size_quantum = 4096;

pub const Slab = []align(slab_align) u8;

/// Borrowed pooled scratch for non-tensor block types (e.g. the packed
/// quantized-LHS layouts, which are not DType members). `items` is valid until
/// `release`, which returns the slab to the pool on whatever thread calls it.
pub fn ScratchLease(comptime T: type) type {
    return struct {
        pool: *BufferPool,
        slab: Slab,
        items: []T,

        pub fn release(self: *@This()) void {
            self.pool.releaseSlab(self.slab);
            self.* = undefined;
        }
    };
}

pub const BufferPool = struct {
    allocator: Allocator,
    free_list: std.ArrayList(*storage.Buffer),
    slab_free_list: std.ArrayList(Slab),
    mutex: thread.Mutex = .{},
    cached_bytes: usize = 0,
    outstanding: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    // Large enough to keep big prefill transients cached: at pp4096 a single
    // fused gate_up activation on a 0.6B model is ~100MB, and mid-size dense
    // models carry several 30-50MB transients per layer. Retention is bounded
    // by the actual peak transient set, not this cap. The budget is SHARED by
    // the f32 arm and the byte-slab arm.
    max_cached_bytes: usize = 1024 * 1024 * 1024,

    pub fn init(allocator: Allocator) BufferPool {
        return .{
            .allocator = allocator,
            .free_list = .empty,
            .slab_free_list = .empty,
        };
    }

    pub fn deinit(self: *BufferPool) void {
        std.debug.assert(self.outstanding.load(.acquire) == 0);
        for (self.free_list.items) |buffer| {
            buffer.destroy();
        }
        self.free_list.deinit(self.allocator);
        for (self.slab_free_list.items) |slab| {
            self.allocator.free(slab);
        }
        self.slab_free_list.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn acquire(self: *BufferPool, len: usize) !*storage.Buffer {
        const requested_len = allocationLen(len);
        self.mutex.lock();
        var i: usize = 0;
        while (i < self.free_list.items.len) : (i += 1) {
            const buffer = self.free_list.items[i];
            if (buffer.data.len >= requested_len) {
                _ = self.free_list.orderedRemove(i);
                self.cached_bytes -= bytesOf(buffer);
                self.mutex.unlock();

                buffer.resetRefs();
                _ = self.outstanding.fetchAdd(1, .monotonic);
                return buffer;
            }
        }
        self.mutex.unlock();

        const buffer = try storage.Buffer.createWithRelease(
            self.allocator,
            requested_len,
            self,
            reclaim,
        );
        _ = self.outstanding.fetchAdd(1, .monotonic);
        return buffer;
    }

    /// Typed (non-f32) pooled buffer over a byte slab. The returned buffer
    /// wraps a slab view of `floor(slab_capacity / @sizeOf(Elem))` elements
    /// (>= storage_len; tensors use the shape-covered prefix, mirroring the
    /// oversized f32 arm) and its release hook returns the slab to the byte
    /// free list and destroys the header. One header `allocator.create` per
    /// acquire (header recycling is a possible follow-up if it ever shows).
    pub fn acquireTyped(self: *BufferPool, comptime dtype: storage.DType, storage_len: usize) !*storage.BufferOf(dtype) {
        if (comptime dtype == .f32) return self.acquire(storage_len);

        const Elem = storage.BufferOf(dtype).Element;
        comptime std.debug.assert(@alignOf(Elem) <= slab_align);
        // Capacity recovery in reclaimTypedFor requires elem size < quantum.
        comptime std.debug.assert(@sizeOf(Elem) > 0 and @sizeOf(Elem) < slab_size_quantum);

        const byte_len = std.math.mul(usize, storage_len, @sizeOf(Elem)) catch return error.OutOfMemory;
        const slab = try self.acquireSlab(byte_len);
        errdefer self.releaseSlab(slab);

        const view_len = slab.len / @sizeOf(Elem);
        const elems: [*]Elem = @ptrCast(@alignCast(slab.ptr));
        return storage.BufferOf(dtype).fromBorrowedSliceWithReleaseCtx(
            self.allocator,
            elems[0..view_len],
            self,
            reclaimTypedFor(dtype),
        );
    }

    /// Pooled scratch slab viewed as `[]T` for non-DType block types (packed
    /// quantized-LHS scratch). Caller must `lease.release()` exactly once.
    pub fn acquireScratch(self: *BufferPool, comptime T: type, len: usize) !ScratchLease(T) {
        comptime std.debug.assert(@alignOf(T) <= slab_align);

        const byte_len = std.math.mul(usize, len, @sizeOf(T)) catch return error.OutOfMemory;
        const slab = try self.acquireSlab(byte_len);
        const items: [*]T = @ptrCast(@alignCast(slab.ptr));
        return .{ .pool = self, .slab = slab, .items = items[0..len] };
    }

    pub fn cachedBuffers(self: *BufferPool) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.free_list.items.len;
    }

    pub fn cachedSlabs(self: *BufferPool) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.slab_free_list.items.len;
    }

    pub fn cachedBytes(self: *BufferPool) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.cached_bytes;
    }

    pub fn outstandingBuffers(self: *const BufferPool) usize {
        return self.outstanding.load(.acquire);
    }

    fn reclaim(ctx: *anyopaque, buffer: *storage.Buffer) void {
        const self: *BufferPool = @ptrCast(@alignCast(ctx));
        const old = self.outstanding.fetchSub(1, .acq_rel);
        std.debug.assert(old > 0);

        self.mutex.lock();
        defer self.mutex.unlock();

        const buffer_bytes = bytesOf(buffer);
        if (buffer_bytes > self.max_cached_bytes or self.cached_bytes + buffer_bytes > self.max_cached_bytes) {
            buffer.destroy();
            return;
        }

        // Insert BEFORE equal-size entries: within a size class the pool
        // hands back the most-recently-released buffer first (LIFO) — its
        // lines are the likeliest to still be cache-resident. Ordering is
        // address-only; numerics are unaffected.
        var index: usize = 0;
        while (index < self.free_list.items.len and self.free_list.items[index].data.len < buffer.data.len) : (index += 1) {}

        self.free_list.insert(self.allocator, index, buffer) catch {
            buffer.destroy();
            return;
        };
        self.cached_bytes += buffer_bytes;
    }

    // ------------------------------------------------------------------
    // Byte-slab arm internals. Same lock/counter discipline as the f32 arm:
    // acquire pops under the mutex (releasing it before a fresh alloc),
    // release runs on whatever thread drops the last ref.
    // ------------------------------------------------------------------

    fn acquireSlab(self: *BufferPool, min_bytes: usize) !Slab {
        const requested_bytes = slabAllocationBytes(min_bytes);
        self.mutex.lock();
        var i: usize = 0;
        while (i < self.slab_free_list.items.len) : (i += 1) {
            const slab = self.slab_free_list.items[i];
            if (slab.len >= requested_bytes) {
                _ = self.slab_free_list.orderedRemove(i);
                self.cached_bytes -= slab.len;
                self.mutex.unlock();

                _ = self.outstanding.fetchAdd(1, .monotonic);
                return slab;
            }
        }
        self.mutex.unlock();

        const slab = try self.allocator.alignedAlloc(u8, comptime std.mem.Alignment.fromByteUnits(slab_align), requested_bytes);
        _ = self.outstanding.fetchAdd(1, .monotonic);
        return slab;
    }

    fn releaseSlab(self: *BufferPool, slab: Slab) void {
        const old = self.outstanding.fetchSub(1, .acq_rel);
        std.debug.assert(old > 0);

        self.mutex.lock();
        defer self.mutex.unlock();

        if (slab.len > self.max_cached_bytes or self.cached_bytes + slab.len > self.max_cached_bytes) {
            self.allocator.free(slab);
            return;
        }

        // LIFO within a size class (see the f32 arm above).
        var index: usize = 0;
        while (index < self.slab_free_list.items.len and self.slab_free_list.items[index].len < slab.len) : (index += 1) {}

        self.slab_free_list.insert(self.allocator, index, slab) catch {
            self.allocator.free(slab);
            return;
        };
        self.cached_bytes += slab.len;
    }

    fn reclaimTypedFor(comptime dtype: storage.DType) *const fn (*anyopaque, *storage.BufferOf(dtype)) void {
        const Elem = storage.BufferOf(dtype).Element;
        return struct {
            fn reclaimTyped(ctx: *anyopaque, buffer: *storage.BufferOf(dtype)) void {
                const self: *BufferPool = @ptrCast(@alignCast(ctx));
                // Recover the slab's true rounded capacity: data.len is
                // floor(capacity / @sizeOf(Elem)) with capacity a multiple of
                // slab_size_quantum and @sizeOf(Elem) < slab_size_quantum, so
                // rounding the view's byte length back up is exact.
                const byte_len = std.mem.alignForward(usize, buffer.data.len * @sizeOf(Elem), slab_size_quantum);
                const base: [*]align(slab_align) u8 = @ptrCast(@alignCast(buffer.data.ptr));
                const slab: Slab = base[0..byte_len];
                buffer.destroyHeader();
                self.releaseSlab(slab);
            }
        }.reclaimTyped;
    }
};

fn bytesOf(buffer: *const storage.Buffer) usize {
    return buffer.data.len * @sizeOf(f32);
}

fn allocationLen(len: usize) usize {
    if (len <= 1024) {
        return std.math.ceilPowerOfTwo(usize, len) catch len;
    }
    return std.mem.alignForward(usize, len, 1024);
}

fn slabAllocationBytes(min_bytes: usize) usize {
    return std.mem.alignForward(usize, @max(min_bytes, 1), slab_size_quantum);
}

test {
    _ = @import("buffer_pool_tests.zig");
}
