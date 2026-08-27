//! Float-slot storage of the ES module: the registered `Slot` view, the
//! per-iteration `StreamCache` with its `CacheRegion`/`SlotCache`/
//! `UpdateCacheView` access types, and the facade-tensor registration
//! helpers (`makeRelease`, `validateFacadePtr`).

const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const tensor_mod = @import("../tensor.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;

pub const Slot = struct {
    name: ?[]const u8, // borrowed
    dtype: DType,
    bytes: []u8, // mutable view aliasing the parameter storage
    n: usize,
    retained: ?*anyopaque, // *TensorOf(dtype) for facade-registered slots
    release: ?*const fn (*anyopaque, Allocator) void,

    pub fn elems(self: *const Slot, comptime dt: DType) []dtype_mod.Scalar(dt) {
        // Slot bytes alias real Scalar(dt) storage, so the alignment holds.
        return @alignCast(std.mem.bytesAsSlice(dtype_mod.Scalar(dt), self.bytes));
    }

    pub fn elemsConst(self: *const Slot, comptime dt: DType) []const dtype_mod.Scalar(dt) {
        return self.elems(dt);
    }

    pub fn deinit(self: *Slot, allocator: Allocator) void {
        if (self.retained) |retained| self.release.?(retained, allocator);
        self.* = undefined;
    }
};

/// Per-iteration noise-stream cache (`Config.cache_streams`): stream-major
/// f32 storage — stream s's slot k lives at
/// data[s * total_elems + slot_offsets[k] ..][0 .. slot k len] — plus a
/// per-stream filled flag for the current iteration. Values are exactly the
/// regenerated ones (the kernels regenerate INTO the cache on first use),
/// so cached and uncached runs are bitwise identical.
pub const StreamCache = struct {
    data: []f32,
    filled: []bool,
    /// Prefix sums of slot element counts (len = n_slots + 1).
    slot_offsets: []usize,
    total_elems: usize,

    pub fn init(allocator: Allocator, slots: []const Slot, n_streams: usize) !StreamCache {
        const slot_offsets = try allocator.alloc(usize, slots.len + 1);
        errdefer allocator.free(slot_offsets);
        var total: usize = 0;
        for (slots, 0..) |*slot, k| {
            slot_offsets[k] = total;
            total += slot.n;
        }
        slot_offsets[slots.len] = total;
        const data = try allocator.alloc(f32, n_streams * total);
        errdefer allocator.free(data);
        const filled = try allocator.alloc(bool, n_streams);
        @memset(filled, false);
        return .{ .data = data, .filled = filled, .slot_offsets = slot_offsets, .total_elems = total };
    }

    pub fn deinit(self: *StreamCache, allocator: Allocator) void {
        allocator.free(self.filled);
        allocator.free(self.data);
        allocator.free(self.slot_offsets);
        self.* = undefined;
    }
};

/// One stream's region of a `StreamCache` for the current iteration.
pub const CacheRegion = struct {
    stream: usize,
    base: usize,
    filled: bool,

    pub fn slotRegion(self: CacheRegion, cache: *const StreamCache, slot_index: usize) []f32 {
        const offset = cache.slot_offsets[slot_index];
        const len = cache.slot_offsets[slot_index + 1] - offset;
        return cache.data[self.base + offset ..][0..len];
    }
};

/// One slot's view of a stream-cache region: `data` aliases the region
/// (global element index j maps to data[j]); `filled` says whether the
/// stream already holds this iteration's regenerated values.
pub const SlotCache = struct { data: []f32, filled: bool };

/// updateSlot's view of the cache: per-stream regions resolved to the
/// current slot inside the kernel (read-only cache access).
pub const UpdateCacheView = struct {
    cache: *const StreamCache,
    slot_index: usize,
    regions: []const CacheRegion,
};

/// The shared registry releaser (`tensor.makeRetainedRelease`), exported
/// here because this module owns the ES registration surface.
pub const makeRelease = tensor_mod.makeRetainedRelease;

pub fn validateFacadePtr(comptime P: type) type {
    const info = @typeInfo(P);
    if (info != .pointer or info.pointer.size != .one) {
        @compileError("es.Trainer.addParam expects a single-item pointer to a facade tensor");
    }
    const T = info.pointer.child;
    if (@typeInfo(T) != .@"struct" or !@hasDecl(T, "dtype") or !@hasField(T, "value")) {
        @compileError("es.Trainer.addParam expects a facade tensor, got " ++ @typeName(P));
    }
    if (T.dtype != .f32 and T.dtype != .f16 and T.dtype != .bf16) {
        @compileError("es.Trainer.addParam supports f32/f16/bf16 tensors only");
    }
    return T;
}
