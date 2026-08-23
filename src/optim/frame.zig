//! Optimizer checkpoint frames: the transactional staged-load helpers,
//! the v3/v4/v5 frame-version rules, per-slot name/dims/master/state
//! records, the name-matching `SlotMatcher`, and the positional f32
//! `saveTensors`/`loadTensors` (FZT1) parameter format. Wire bytes are
//! pinned by the checkpoint goldens in optim_tests.zig.

const std = @import("std");
const common = @import("common.zig");

const Allocator = std.mem.Allocator;
const OptimError = common.OptimError;
const StateDType = common.StateDType;
const StateBuf = common.StateBuf;
const Param = common.Param;

// ---------------------------------------------------------------------------
// Checkpoint helpers.
// ---------------------------------------------------------------------------
//
// Transactional load contract: every `loadState` / `loadTensors` here is
// all-or-nothing. Each record is decoded into freshly-allocated scratch and the
// WHOLE stream is validated (magic, config, names, dims, lengths, slot match)
// BEFORE any live parameter/optimizer buffer is written. A truncated, short, or
// otherwise-invalid stream therefore leaves every destination byte-unchanged —
// a half-applied checkpoint can silently corrupt training, so we never produce
// one. (`OptimizerSet.loadState` is transactional per member optimizer.)

/// One decoded-but-not-yet-committed slot record for a transactional `loadState`:
/// the destination slot index, the scalar fields, and a freshly-allocated RAW-BYTE
/// scratch holding the slot's contiguous state buffers (e.g. m then v, or a lone
/// momentum/buf) in their storage dtype — dtype validation already happened at
/// read time, so the commit is a plain byte copy for every `StateDType`.
/// `seed`/`prev_norm` are only meaningful for APOLLO main slots.
pub const StagedSlot = struct {
    idx: usize,
    data: []u8,
    step: u64 = 0,
    seed: u64 = 0,
    prev_norm: f32 = 0,
    /// Staged v5 f32 master weights (empty when the frame carried none).
    master: []f32 = &.{},
};

pub fn freeStaged(allocator: Allocator, staged: *std.ArrayList(StagedSlot)) void {
    for (staged.items) |s| {
        allocator.free(s.data);
        if (s.master.len != 0) allocator.free(s.master);
    }
    staged.deinit(allocator);
}

/// Optimizer state frame revision. v3 is the pre-`StateDType` format: state
/// buffers are raw f32 bytes with no tag. v4 prefixes every state buffer with
/// one u8 `StateDType` tag. Writers emit v3 whenever every buffer is f32 (so
/// the bytes stay identical to pre-bf16 builds) and v4 otherwise; readers
/// accept both and require the stored dtype to match the live buffer's.
pub const FrameVersion = enum { v3, v4, v5 };

/// The frame version for the common `{ m, v }` moment-pair slot layout
/// (Adam/AdamW): v3 iff every buffer of every slot is f32.
pub fn momentSlotsFrameVersion(slots: anytype) FrameVersion {
    for (slots) |*slot| {
        if (slot.m != .f32 or slot.v != .f32) return .v4;
    }
    return .v3;
}

/// Read a 4-byte magic and map it to the frame version it names.
pub fn expectMagicVersion(reader: *std.Io.Reader, comptime v3_magic: *const [4]u8, comptime v4_magic: *const [4]u8, comptime v5_magic: *const [4]u8) !FrameVersion {
    var buf: [4]u8 = undefined;
    try reader.readSliceAll(&buf);
    if (std.mem.eql(u8, &buf, v3_magic)) return .v3;
    if (std.mem.eql(u8, &buf, v4_magic)) return .v4;
    if (std.mem.eql(u8, &buf, v5_magic)) return .v5;
    return OptimError.CheckpointMagicMismatch;
}

/// v5 frames exist to persist f32 MASTER weights for 16-bit params: resuming
/// from the narrowed values instead would re-round the master and lose the
/// sub-bf16 update accumulation. Frames without 16-bit slots keep emitting
/// v3/v4, byte-identical to before.
pub fn slotsCarryMasters(slots: anytype) bool {
    for (slots) |*slot| {
        if (slot.param.master.len != 0) return true;
    }
    return false;
}

/// Per-slot master record (v5 frames only): u8 presence flag + raw f32 bytes.
pub fn writeSlotMaster(writer: *std.Io.Writer, version: FrameVersion, param: *const Param) !void {
    if (version != .v5) return;
    const has: u8 = @intFromBool(param.master.len != 0);
    try writer.writeInt(u8, has, .little);
    if (has == 1) try writeF32Slice(writer, param.master);
}

pub fn readSlotMaster(allocator: Allocator, reader: *std.Io.Reader, version: FrameVersion, param: *const Param) ![]f32 {
    if (version != .v5) return &.{};
    if (try reader.takeInt(u8, .little) == 0) return &.{};
    if (param.master.len == 0) return OptimError.CheckpointDtypeMismatch;
    const buf = try allocator.alloc(f32, param.master.len);
    errdefer allocator.free(buf);
    try readF32Slice(reader, buf);
    return buf;
}

/// Commit arm for a 16-bit slot's master: install the checkpoint master and
/// narrow it into the param storage, or — when the frame carried none — re-
/// widen from the (possibly just-loaded) param values so the master never
/// goes stale. No-op for f32 params.
pub fn commitSlotMaster(param: *Param, staged_master: []const f32) void {
    if (param.master.len == 0) return;
    if (staged_master.len != 0) {
        @memcpy(param.master, staged_master);
        param.publish();
    } else {
        param.refreshMasterFromValue();
    }
}

/// Write one state-buffer record: v3 = raw f32 bytes (the caller guarantees
/// every buffer is f32 before choosing v3), v4 = one u8 dtype tag + the raw
/// storage bytes. The f32 v3 arm is byte-identical to the old `writeF32Slice`.
pub fn writeStateSlice(writer: *std.Io.Writer, version: FrameVersion, buf: StateBuf) !void {
    switch (version) {
        .v3 => std.debug.assert(buf == .f32),
        .v4, .v5 => try writer.writeInt(u8, @intFromEnum(@as(StateDType, buf)), .little),
    }
    try writer.writeAll(buf.bytesConst());
}

/// Read one state-buffer record into `dest` (raw bytes, `buf.byteLen()` long).
/// The stored dtype — implicitly f32 for v3, the u8 tag for v4 — must equal
/// the live buffer's exactly: NO implicit conversion, or a resumed run would
/// silently break the bit-exact-resume contract.
pub fn readStateSlice(reader: *std.Io.Reader, version: FrameVersion, buf: StateBuf, dest: []u8) !void {
    const stored: StateDType = switch (version) {
        .v3 => .f32,
        .v4, .v5 => std.enums.fromInt(StateDType, try reader.takeInt(u8, .little)) orelse
            return OptimError.CheckpointUnsupportedDtype,
    };
    if (stored != @as(StateDType, buf)) return OptimError.CheckpointDtypeMismatch;
    try reader.readSliceAll(dest);
}

/// Serialize parameter values (shapes + f32 data, little-endian). `tensors` is
/// a tuple of pointers to contiguous f32 facade tensors (variables or
/// constants). This is all an inference-time consumer needs.
pub fn saveTensors(writer: *std.Io.Writer, tensors: anytype) !void {
    try writer.writeAll("FZT1");
    try writer.writeInt(u32, @intCast(tensors.len), .little);
    inline for (tensors) |t| {
        if (!t.value.isContiguous()) return OptimError.NonContiguousParam;
        const shape = t.value.shape.slice();
        try writer.writeInt(u32, @intCast(shape.len), .little);
        for (shape) |dim| try writer.writeInt(u64, dim, .little);
        try writeF32Slice(writer, t.value.dataConst());
    }
}

/// Load parameter values saved by `saveTensors` into existing tensors of the
/// same shapes (order-based, shape-validated). Transactional (see the contract
/// above): the whole stream is staged + validated before any tensor is written,
/// so a truncated/mismatched stream leaves every tensor byte-unchanged. Needs an
/// allocator for the per-tensor scratch.
pub fn loadTensors(allocator: Allocator, reader: *std.Io.Reader, tensors: anytype) !void {
    try expectMagic(reader, "FZT1");
    const count = try reader.takeInt(u32, .little);
    if (count != tensors.len) return OptimError.CheckpointShapeMismatch;
    var staged: [tensors.len][]f32 = undefined;
    var staged_n: usize = 0;
    defer for (staged[0..staged_n]) |buf| allocator.free(buf);
    // Pass 1 — validate shapes + read every tensor into scratch.
    inline for (tensors, 0..) |t, ti| {
        if (!t.value.isContiguous()) return OptimError.NonContiguousParam;
        const shape = t.value.shape.slice();
        const rank = try reader.takeInt(u32, .little);
        if (rank != shape.len) return OptimError.CheckpointShapeMismatch;
        for (shape) |dim| {
            const stored = try reader.takeInt(u64, .little);
            if (stored != dim) return OptimError.CheckpointShapeMismatch;
        }
        const buf = try allocator.alloc(f32, t.value.data().len);
        staged[ti] = buf;
        staged_n = ti + 1;
        try readF32Slice(reader, buf);
    }
    // Pass 2 — commit (all reads succeeded).
    inline for (tensors, 0..) |t, ti| {
        @memcpy(t.value.data(), staged[ti]);
    }
}

pub fn writeF32Slice(writer: *std.Io.Writer, values: []const f32) !void {
    try writer.writeAll(std.mem.sliceAsBytes(values));
}

pub fn readF32Slice(reader: *std.Io.Reader, values: []f32) !void {
    try reader.readSliceAll(std.mem.sliceAsBytes(values));
}

/// Longest serializable name; the wire length prefix is u16.
const max_name_len: usize = std.math.maxInt(u16);
/// "param" plus at most 20 digits of a usize slot index.
const auto_name_buf_len = "param".len + 20;

/// A slot's checkpoint identity: the explicit registration name, or the
/// auto-name "param<i>" from its index within its slot list.
fn slotName(param: *const Param, index: usize, buf: *[auto_name_buf_len]u8) []const u8 {
    return param.name orelse (std.fmt.bufPrint(buf, "param{d}", .{index}) catch unreachable);
}

fn validateName(name: []const u8) !void {
    if (name.len == 0 or name.len > max_name_len) return OptimError.CheckpointInvalidName;
    if (std.mem.indexOfScalar(u8, name, 0) != null) return OptimError.CheckpointInvalidName;
    if (!std.unicode.utf8ValidateSlice(name)) return OptimError.CheckpointInvalidName;
}

/// Validate the effective names of one name-matched slot list before saving:
/// well-formed and collision-free (an explicit name can also collide with an
/// auto-name). O(n^2) compares — checkpoint-time only, allocation-free.
pub fn validateSlotNames(slots: anytype) !void {
    for (slots, 0..) |*a, i| {
        var buf_a: [auto_name_buf_len]u8 = undefined;
        const name_a = slotName(&a.param, i, &buf_a);
        try validateName(name_a);
        for (slots[i + 1 ..], i + 1..) |*b, j| {
            var buf_b: [auto_name_buf_len]u8 = undefined;
            if (std.mem.eql(u8, name_a, slotName(&b.param, j, &buf_b))) {
                return OptimError.CheckpointDuplicateName;
            }
        }
    }
}

pub fn writeSlotName(writer: *std.Io.Writer, param: *const Param, index: usize) !void {
    var buf: [auto_name_buf_len]u8 = undefined;
    const name = slotName(param, index, &buf);
    try writer.writeInt(u16, @intCast(name.len), .little);
    try writer.writeAll(name);
}

/// Name-matches v3/v4 optimizer slot records to registered slots, enforcing the
/// fill-exactly-once contract: an unknown record name, a record matching an
/// already-filled slot, and a slot left unfilled at the end all error.
pub const SlotMatcher = struct {
    filled: []bool,
    name_buf: []u8,

    pub fn init(allocator: Allocator, slot_count: usize) !SlotMatcher {
        const filled = try allocator.alloc(bool, slot_count);
        errdefer allocator.free(filled);
        @memset(filled, false);
        const name_buf = try allocator.alloc(u8, max_name_len);
        return .{ .filled = filled, .name_buf = name_buf };
    }

    pub fn deinit(self: *SlotMatcher, allocator: Allocator) void {
        allocator.free(self.filled);
        allocator.free(self.name_buf);
        self.* = undefined;
    }

    /// Read one record's name and resolve it to a not-yet-filled slot index.
    pub fn match(self: *SlotMatcher, reader: *std.Io.Reader, slots: anytype) !usize {
        const name_len = try reader.takeInt(u16, .little);
        if (name_len == 0) return OptimError.CheckpointInvalidName;
        const name = self.name_buf[0..name_len];
        try reader.readSliceAll(name);
        for (slots, 0..) |*slot, i| {
            var buf: [auto_name_buf_len]u8 = undefined;
            if (!std.mem.eql(u8, name, slotName(&slot.param, i, &buf))) continue;
            if (self.filled[i]) return OptimError.CheckpointDuplicateName;
            self.filled[i] = true;
            return i;
        }
        return OptimError.CheckpointUnknownName;
    }

    pub fn requireAllFilled(self: *const SlotMatcher) !void {
        for (self.filled) |was_filled| if (!was_filled) return OptimError.CheckpointMissingEntry;
    }
};

pub fn writeSlotDims(writer: *std.Io.Writer, param: *const Param) !void {
    try writer.writeInt(u64, param.rows, .little);
    try writer.writeInt(u64, param.cols, .little);
}

pub fn expectSlotDims(reader: *std.Io.Reader, param: *const Param) !void {
    if (try reader.takeInt(u64, .little) != param.rows) return OptimError.CheckpointShapeMismatch;
    if (try reader.takeInt(u64, .little) != param.cols) return OptimError.CheckpointShapeMismatch;
}

pub fn expectMagic(reader: *std.Io.Reader, comptime magic: *const [4]u8) !void {
    var buf: [4]u8 = undefined;
    try reader.readSliceAll(&buf);
    if (!std.mem.eql(u8, &buf, magic)) return OptimError.CheckpointMagicMismatch;
}
