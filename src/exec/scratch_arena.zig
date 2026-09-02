//! A grow-only, mutex-guarded scratch arena for the per-token decode paths
//! that carve typed slices instead of allocating: the per-token region
//! sizes are model constants, so after the first token the hot path
//! performs no allocations and takes one uncontended lock instead of
//! several allocator round-trips per layer. Callers hold the lock for the
//! whole op because their tasks write into the carved slices. Region types
//! must align to <= 8 (the u64 backing store's natural alignment).
const std = @import("std");
const thread = @import("../thread.zig");
const tensor = @import("../tensor.zig");

const Allocator = std.mem.Allocator;

pub const ScratchArena = struct {
    mutex: thread.Mutex = .{},
    words: []u64 = &.{},

    pub fn lock(self: *ScratchArena) void {
        self.mutex.lock();
    }

    pub fn unlock(self: *ScratchArena) void {
        self.mutex.unlock();
    }

    /// The arena's base, grown once to hold `bytes`; valid until the next
    /// `reserve` that grows it. The caller holds the lock.
    pub fn reserve(self: *ScratchArena, allocator: Allocator, bytes: usize) ![*]u8 {
        const rounded = std.math.add(usize, bytes, @sizeOf(u64) - 1) catch return tensor.TensorError.InvalidDataLength;
        const words_needed = rounded / @sizeOf(u64);
        if (self.words.len < words_needed) {
            const grown = try allocator.alloc(u64, words_needed);
            if (self.words.len > 0) allocator.free(self.words);
            self.words = grown;
        }
        return @ptrCast(self.words.ptr);
    }

    pub fn deinit(self: *ScratchArena, allocator: Allocator) void {
        if (self.words.len > 0) allocator.free(self.words);
        self.* = undefined;
    }

    /// Carves typed regions out of a reserved base in order; `need` sizes
    /// the same sequence ahead of `reserve`.
    pub const Carver = struct {
        base: [*]u8,
        offset: usize = 0,

        pub fn carve(self: *Carver, comptime T: type, n: usize) ![]T {
            const start = std.mem.alignForward(usize, self.offset, @alignOf(T));
            const byte_len = std.math.mul(usize, n, @sizeOf(T)) catch return tensor.TensorError.InvalidDataLength;
            self.offset = std.math.add(usize, start, byte_len) catch return tensor.TensorError.InvalidDataLength;
            const ptr: [*]T = @ptrCast(@alignCast(self.base + start));
            return ptr[0..n];
        }

        pub fn need(comptime T: type, offset: usize, n: usize) !usize {
            const start = std.mem.alignForward(usize, offset, @alignOf(T));
            const byte_len = std.math.mul(usize, n, @sizeOf(T)) catch return tensor.TensorError.InvalidDataLength;
            return std.math.add(usize, start, byte_len) catch return tensor.TensorError.InvalidDataLength;
        }
    };
};
