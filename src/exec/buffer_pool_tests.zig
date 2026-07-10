//! Behavioral tests for the byte-slab (typed/non-f32) arm of the
//! `BufferPool`: typed address reuse, cross-dtype slab sharing, the shared
//! byte-cap eviction, accounting recovery across acquire/release cycles,
//! scratch leases, and slab alignment. The f32 arm's behavior is pinned by
//! `src/exec_tests.zig` (address reuse + outstanding/cached counts) and must
//! stay byte-identical. Force-imported by `buffer_pool.zig`'s `test` block.
//! Excluded from arch-check (a `_tests.zig` file).

const std = @import("std");

const buffer_pool = @import("buffer_pool.zig");
const runtime_mod = @import("runtime.zig");

const BufferPool = buffer_pool.BufferPool;
const Runtime = runtime_mod.Runtime;
const slab_align = buffer_pool.slab_align;
const slab_size_quantum = buffer_pool.slab_size_quantum;

// Local stand-in for the packed quantized-LHS scratch layouts (mirrors
// BlockQ8_Kx4: 1168 bytes, align 4) so the leaf test file does not have to
// import the backend.
const FakePackedBlock = extern struct {
    d: [4]f32,
    qs: [1024]i8,
    bsums: [64]i16,
};

test "typed emptyTyped reuses released slabs (f16 address reuse)" {
    var rt: Runtime = undefined;
    rt.init(std.testing.allocator);
    defer rt.deinit();

    var first = try rt.emptyTyped(.f16, &.{ 32, 64 });
    @memset(first.data(), 0);
    const first_ptr = first.dataConst().ptr;
    first.deinit();

    var second = try rt.emptyTyped(.f16, &.{ 32, 64 });
    defer second.deinit();
    try std.testing.expectEqual(first_ptr, second.dataConst().ptr);
}

test "cross-dtype slab reuse: f16 then q8_k share one slab" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    // 1024 f16 elements = 2048 bytes -> one quantum-rounded slab.
    const f16_buf = try pool.acquireTyped(.f16, 1024);
    const base = @intFromPtr(f16_buf.data.ptr);
    f16_buf.release();
    try std.testing.expectEqual(@as(usize, 1), pool.cachedSlabs());

    // 10 BlockQ8_K = 2920 bytes fits the same slab.
    const q8k_buf = try pool.acquireTyped(.q8_k, 10);
    defer q8k_buf.release();
    try std.testing.expect(q8k_buf.data.len >= 10);
    try std.testing.expectEqual(base, @intFromPtr(q8k_buf.data.ptr));
    try std.testing.expectEqual(@as(usize, 0), pool.cachedSlabs());
}

test "slab byte-cap eviction shares the f32 budget" {
    var pool = BufferPool.init(std.testing.allocator);
    pool.max_cached_bytes = slab_size_quantum;
    defer pool.deinit();

    // One cap-sized slab is cached.
    const a = try pool.acquireTyped(.f16, 16);
    a.release();
    try std.testing.expectEqual(@as(usize, 1), pool.cachedSlabs());
    try std.testing.expectEqual(@as(usize, slab_size_quantum), pool.cachedBytes());

    // An oversized slab is destroyed instead of cached.
    const b = try pool.acquireTyped(.f16, slab_size_quantum);
    b.release();
    try std.testing.expectEqual(@as(usize, 1), pool.cachedSlabs());
    try std.testing.expectEqual(@as(usize, slab_size_quantum), pool.cachedBytes());

    // A second cap-sized slab would exceed the budget: destroyed.
    const c = try pool.acquireTyped(.f16, 16); // pops the cached slab
    const d = try pool.acquireTyped(.f16, 16); // fresh slab
    c.release();
    d.release();
    try std.testing.expectEqual(@as(usize, 1), pool.cachedSlabs());
    try std.testing.expectEqual(@as(usize, slab_size_quantum), pool.cachedBytes());
}

test "slab accounting recovers exactly across cross-dtype cycles" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    const a = try pool.acquireTyped(.f16, 1024); // 2048 B -> 4096 B slab
    const b = try pool.acquireTyped(.q6_k, 32); // 6720 B -> 8192 B slab
    try std.testing.expectEqual(@as(usize, 0), pool.cachedBytes());
    try std.testing.expectEqual(@as(usize, 2), pool.outstandingBuffers());
    a.release();
    b.release();
    try std.testing.expectEqual(@as(usize, 3 * slab_size_quantum), pool.cachedBytes());
    try std.testing.expectEqual(@as(usize, 0), pool.outstandingBuffers());

    // Reinterpreting the same slabs under other dtypes must hand back the
    // identical rounded capacities (the capacity-recovery invariant).
    for (0..3) |_| {
        const x = try pool.acquireTyped(.bf16, 2048); // 4096 B -> the 4096 B slab
        const y = try pool.acquireTyped(.q8_k, 20); // 5840 B -> the 8192 B slab
        try std.testing.expectEqual(@as(usize, 0), pool.cachedBytes());
        try std.testing.expectEqual(@as(usize, 0), pool.cachedSlabs());
        x.release();
        y.release();
        try std.testing.expectEqual(@as(usize, 3 * slab_size_quantum), pool.cachedBytes());
        try std.testing.expectEqual(@as(usize, 2), pool.cachedSlabs());
    }
    try std.testing.expectEqual(@as(usize, 0), pool.outstandingBuffers());
}

test "scratch lease recycles slabs and returns aligned items" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    var lease = try pool.acquireScratch(FakePackedBlock, 3); // 3504 B -> 4096 B slab
    try std.testing.expectEqual(@as(usize, 3), lease.items.len);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(lease.items.ptr) % slab_align);
    lease.items[2].d[0] = 42.0;
    const base = @intFromPtr(lease.items.ptr);
    try std.testing.expectEqual(@as(usize, 1), pool.outstandingBuffers());
    lease.release();
    try std.testing.expectEqual(@as(usize, 0), pool.outstandingBuffers());
    try std.testing.expectEqual(@as(usize, 1), pool.cachedSlabs());
    try std.testing.expectEqual(@as(usize, slab_size_quantum), pool.cachedBytes());

    var again = try pool.acquireScratch(FakePackedBlock, 3);
    defer again.release();
    try std.testing.expectEqual(base, @intFromPtr(again.items.ptr));
}

test "typed acquires return slab-aligned storage" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    const a = try pool.acquireTyped(.f16, 33);
    defer a.release();
    const b = try pool.acquireTyped(.bf16, 7);
    defer b.release();
    const c = try pool.acquireTyped(.q6_k, 3);
    defer c.release();
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(a.data.ptr) % slab_align);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(b.data.ptr) % slab_align);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(c.data.ptr) % slab_align);
}

test "acquireTyped(.f32) routes to the f32 arm" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    const buf = try pool.acquireTyped(.f32, 100);
    buf.release();
    try std.testing.expectEqual(@as(usize, 1), pool.cachedBuffers());
    try std.testing.expectEqual(@as(usize, 0), pool.cachedSlabs());
}
