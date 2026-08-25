//! Steady-state caching allocator for training loops: blocks of at least
//! 64 KB round up to the next power of two and recycle through per-class
//! freelists; nothing returns to the backing allocator until `deinit`.
//! Repeated steps then reuse warm pages instead of paying mmap/madvise
//! and first-touch page faults on every tensor allocation — the property
//! PyTorch's caching allocator provides for CUDA training. Small blocks
//! and over-page alignments pass through to the backing allocator.
//!
//! Trade: cached memory is retained at the high-water mark of live large
//! blocks (rounded up per class), and recycled blocks come back DIRTY —
//! callers own their initialization, as with any Zig allocator. Measured
//! on the 0.6B q8 LoRA step (1259 tokens): 6.1 s -> 4.9 s per step.
const std = @import("std");

pub const CachingAllocator = struct {
    backing: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    lists: [class_count]std.ArrayListUnmanaged([*]u8) = @splat(.empty),

    const min_shift = 16; // 64 KB: below this, churn is cheap and variety is high
    const max_shift = 34; // 16 GB ceiling per block
    /// Classes per octave: geometric steps of 2^(1/4). Power-of-two classes
    /// rounded a training step's 5.2 MB hidden-sized tensors up to 8 MiB;
    /// at 1280 tokens on the 0.6B LoRA step that was 7.7 GB of the 28 GB
    /// peak. Four steps per octave cap the worst-case rounding at 19% and
    /// measured 2.1 GB at the same peak, 10% less resident memory. The
    /// trade-off is more free lists: a block serves only its own class, so
    /// a workload with widely varying shapes hoards more than it would with
    /// coarser classes (the lists are never released before `deinit`).
    const steps_per_octave = 4;
    const class_count = (max_shift - min_shift) * steps_per_octave + 1;

    pub fn init(backing: std.mem.Allocator) CachingAllocator {
        return .{ .backing = backing };
    }

    /// Returns every cached block to the backing allocator.
    pub fn deinit(self: *CachingAllocator) void {
        for (&self.lists, 0..) |*list, class| {
            const class_len = classLen(class);
            for (list.items) |ptr| {
                self.backing.vtable.free(self.backing.ptr, ptr[0..class_len], page_align, @returnAddress());
            }
            list.deinit(self.backing);
        }
        self.* = undefined;
    }

    pub fn allocator(self: *CachingAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const page_align = std.mem.Alignment.fromByteUnits(std.heap.page_size_min);

    /// The class sizes: `2^(min_shift + c / steps) * 2^((c % steps) / steps)`,
    /// rounded up to whole pages.
    fn classLen(class: usize) usize {
        const octave: u6 = @intCast(min_shift + class / steps_per_octave);
        const sub = class % steps_per_octave;
        const base = @as(usize, 1) << octave;
        // 2^(sub/4) as a fixed-point multiplier over 2^16: 1, 1.1892, 1.4142, 1.6818.
        const mult = [_]usize{ 65536, 77936, 92682, 110218 };
        const raw = (base >> 16) * mult[sub];
        const page = std.heap.page_size_min;
        return (raw + page - 1) / page * page;
    }

    /// The smallest class whose size holds `len`; null below the caching
    /// floor and above the ceiling.
    fn classOf(len: usize) ?usize {
        if (len < (@as(usize, 1) << min_shift)) return null;
        if (std.math.log2_int_ceil(usize, len) > max_shift) return null;
        const floor_shift = std.math.log2_int(usize, len);
        var class = (@as(usize, @max(floor_shift, min_shift)) - min_shift) * steps_per_octave;
        while (classLen(class) < len) : (class += 1) {}
        if (class >= class_count) return null;
        return class;
    }

    const vtable = std.mem.Allocator.VTable{
        .alloc = allocImpl,
        .resize = resizeImpl,
        .remap = remapImpl,
        .free = freeImpl,
    };

    fn allocImpl(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CachingAllocator = @ptrCast(@alignCast(ctx));
        const class = classOf(len) orelse return self.backing.vtable.alloc(self.backing.ptr, len, alignment, ret_addr);
        if (alignment.toByteUnits() > std.heap.page_size_min) return self.backing.vtable.alloc(self.backing.ptr, len, alignment, ret_addr);
        std.Io.Threaded.mutexLock(&self.mutex);
        const cached = self.lists[class].pop();
        std.Io.Threaded.mutexUnlock(&self.mutex);
        if (cached) |ptr| return ptr;
        const class_len = classLen(class);
        return self.backing.vtable.alloc(self.backing.ptr, class_len, page_align, ret_addr);
    }

    fn resizeImpl(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *CachingAllocator = @ptrCast(@alignCast(ctx));
        const class = classOf(memory.len) orelse {
            // A passthrough block must never grow into the cached range:
            // its free() would then route to a freelist it never came from.
            if (classOf(new_len) != null) return false;
            return self.backing.vtable.resize(self.backing.ptr, memory, alignment, new_len, ret_addr);
        };
        return classOf(new_len) == class; // the block's capacity is its class size
    }

    fn remapImpl(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *CachingAllocator = @ptrCast(@alignCast(ctx));
        if (classOf(memory.len)) |class| {
            return if (classOf(new_len) == class) memory.ptr else null;
        }
        if (classOf(new_len) != null) return null;
        return self.backing.vtable.remap(self.backing.ptr, memory, alignment, new_len, ret_addr);
    }

    fn freeImpl(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *CachingAllocator = @ptrCast(@alignCast(ctx));
        const class = classOf(memory.len) orelse return self.backing.vtable.free(self.backing.ptr, memory, alignment, ret_addr);
        if (alignment.toByteUnits() > std.heap.page_size_min) return self.backing.vtable.free(self.backing.ptr, memory, alignment, ret_addr);
        std.Io.Threaded.mutexLock(&self.mutex);
        defer std.Io.Threaded.mutexUnlock(&self.mutex);
        self.lists[class].append(self.backing, memory.ptr) catch {
            const class_len = classLen(class);
            self.backing.vtable.free(self.backing.ptr, memory.ptr[0..class_len], page_align, ret_addr);
        };
    }
};

test "caching allocator recycles large blocks and passes small ones through" {
    var cache = CachingAllocator.init(std.testing.allocator);
    defer cache.deinit();
    const a = cache.allocator();

    // Small blocks: plain passthrough round-trip.
    const small = try a.alloc(u8, 1024);
    a.free(small);

    // Large blocks: the second alloc of the same class reuses the first
    // block's storage.
    const first = try a.alloc(f32, 100_000); // 400 KB: the 432 KiB class
    const first_ptr = first.ptr;
    a.free(first);
    const second = try a.alloc(f32, 105_000); // 420 KB: same class
    try std.testing.expectEqual(@intFromPtr(first_ptr), @intFromPtr(second.ptr));
    a.free(second);

    // Different class: distinct storage, both cached for deinit to release.
    const third = try a.alloc(f32, 400_000); // 1.6 MB: the 1728 KiB class
    try std.testing.expect(@intFromPtr(third.ptr) != @intFromPtr(first_ptr));
    a.free(third);
}

test "caching allocator resize stays inside the class" {
    var cache = CachingAllocator.init(std.testing.allocator);
    defer cache.deinit();
    const a = cache.allocator();

    var buf = try a.alloc(u8, 100_000); // the 112 KiB class
    try std.testing.expect(a.resize(buf, 110_000)); // same class: in place
    buf = buf.ptr[0..110_000];
    try std.testing.expect(!a.resize(buf, 200_000)); // crosses class: refused
    a.free(buf);
}
