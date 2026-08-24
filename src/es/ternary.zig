//! Ternary genome machinery of the ES module: tq2_0 crumb addressing
//! (crumbGet/crumbSet), the counter-based flip stream (ternaryFlipAt), the
//! clamped one-bin move (moveCode), the pinned vote order (voteBefore), and
//! the TernarySlot/UndoEntry/VoteEntry storage types the trainer drives.

const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const rng = @import("../rng.zig");
const common = @import("common.zig");

const Allocator = std.mem.Allocator;
const BlockTQ2_0 = common.BlockTQ2_0;

/// Logical elements per BlockTQ2_0 (the ggml QK_K super-block).
pub const ternary_block_len: usize = dtype_mod.qk_k_block_size;

/// One registered ternary genome. Blocks are BORROWED mutable storage; the
/// undo log (sized once at the per-member flip count) holds the active
/// member's (index, old code) pairs for reverse replay in `restore`.
pub const TernarySlot = struct {
    name: ?[]const u8, // borrowed
    blocks: []BlockTQ2_0, // mutable view aliasing the genome storage
    /// Logical element count (== blocks.len * 256).
    len: usize,
    undo: []UndoEntry,
    undo_len: usize = 0,
    /// Staged top-K entry count in the trainer's `ternary_entries` scratch
    /// (set by collectTernaryVotes, consumed by applyTernaryVotes within a
    /// single update call).
    pending: usize = 0,

    pub fn deinit(self: *TernarySlot, allocator: Allocator) void {
        allocator.free(self.undo);
        self.* = undefined;
    }
};

pub const UndoEntry = struct { index: usize, code: u8 };

/// A nonzero vote of the ternary update (see `Trainer.collectTernaryVotes`).
pub const VoteEntry = struct { index: usize, vote: f32 };

/// Ternary update application order: |vote| descending, index ascending —
/// a total order, so the applied top-K set is deterministic.
pub fn voteBefore(_: void, a: VoteEntry, b: VoteEntry) bool {
    if (@abs(a.vote) != @abs(b.vote)) return @abs(a.vote) > @abs(b.vote);
    return a.index < b.index;
}

/// Flip i of a ternary flip stream: index = at(seed, 2i) % len, delta bit =
/// at(seed, 2i+1) & 1 (1 = +1, 0 = -1), negated for the antithetic odd
/// member (`mirror`). Counter-based like the gaussian streams — any flip is
/// O(1)-addressable — and a checkpoint contract (see the module doc).
pub fn ternaryFlipAt(stream_seed: u64, i: u64, len: usize, mirror: bool) struct { index: usize, delta: i8 } {
    const index: usize = @intCast(rng.at(stream_seed, 2 * i) % @as(u64, len));
    const bit = rng.at(stream_seed, 2 * i + 1) & 1;
    const base: i8 = if (bit == 1) 1 else -1;
    return .{ .index = index, .delta = if (mirror) -base else base };
}

/// One clamped single-bin move on a stored code (w+1):
/// code' = clamp((code - 1) + delta, -1, +1) + 1.
pub fn moveCode(code: u8, delta: i8) u8 {
    const w = @as(i8, @intCast(code)) - 1;
    return @intCast(std.math.clamp(w + delta, -1, 1) + 1);
}

/// Crumb address of logical element e in tq2_0 layout: block e/256; within
/// the block, r = e % 256 splits into group g = r/128 (32-byte qs half),
/// lane l = (r%128)/32 (bit pair 2l+1:2l), byte (r%128)%32 — crumb lane l
/// of a group covers elements [g*128 + l*32, g*128 + (l+1)*32).
fn crumbAddr(e: usize) struct { block: usize, byte: usize, shift: u3 } {
    const r = e % ternary_block_len;
    const group = r / 128;
    const lane = (r % 128) / 32;
    const byte = (r % 128) % 32;
    return .{ .block = e / ternary_block_len, .byte = group * 32 + byte, .shift = @intCast(2 * lane) };
}

pub fn crumbGet(blocks: []const BlockTQ2_0, e: usize) u8 {
    const addr = crumbAddr(e);
    return (blocks[addr.block].qs[addr.byte] >> addr.shift) & 3;
}

pub fn crumbSet(blocks: []BlockTQ2_0, e: usize, code: u8) void {
    std.debug.assert(code <= 2);
    const addr = crumbAddr(e);
    const byte = &blocks[addr.block].qs[addr.byte];
    byte.* = (byte.* & ~(@as(u8, 3) << addr.shift)) | (code << addr.shift);
}
