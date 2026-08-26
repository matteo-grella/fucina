//! Optimizer checkpoint frames: the frame magics and their version map,
//! the type-directed wire form of header/record scalars and state
//! buffers, per-slot name/dims/master records, the name-indexed
//! `SlotIndex`/`SlotMatcher`, and the positional f32 `saveTensors`/
//! `loadTensors` (FZT1) parameter format. The frame walk itself (version
//! rule, header, staged transactional load) is `common.Optimizer`. Wire
//! bytes are pinned by the checkpoint goldens in optim_tests.zig.

const std = @import("std");
const common = @import("common.zig");

const Allocator = std.mem.Allocator;
const OptimError = common.OptimError;
const StateDType = common.StateDType;
const StateBuf = common.StateBuf;
const Param = common.Param;

/// Optimizer state frame revision. v3 is the pre-`StateDType` format: state
/// buffers are raw f32 bytes with no tag. v4 prefixes every state buffer with
/// one u8 `StateDType` tag. v5 adds the per-slot f32 master record. Writers
/// emit the lowest version the slots need (`common.Optimizer.frameVersion`),
/// so all-f32 frames stay byte-identical to pre-bf16 builds; readers accept
/// every version and require the stored dtype to match the live buffer's.
pub const FrameVersion = enum(u8) { v3, v4, v5 };

/// The 4-byte magics of one optimizer's frame, one per version. A kernel
/// whose state is never dtype-tagged (Apollo: raw f32 moments) has no v4
/// magic; its frames go straight from v3 to v5.
pub const Magics = struct {
    v3: *const [4]u8,
    v4: ?*const [4]u8 = null,
    v5: *const [4]u8,

    pub fn of(self: Magics, version: FrameVersion) *const [4]u8 {
        return switch (version) {
            .v3 => self.v3,
            .v4 => self.v4.?,
            .v5 => self.v5,
        };
    }

    /// Read a 4-byte magic and map it to the frame version it names.
    pub fn expect(self: Magics, reader: *std.Io.Reader) !FrameVersion {
        var buf: [4]u8 = undefined;
        try reader.readSliceAll(&buf);
        if (std.mem.eql(u8, &buf, self.v3)) return .v3;
        if (self.v4) |v4| if (std.mem.eql(u8, &buf, v4)) return .v4;
        if (std.mem.eql(u8, &buf, self.v5)) return .v5;
        return OptimError.CheckpointMagicMismatch;
    }
};

/// Wire form of one header or record scalar, chosen by its type: f32 as
/// its IEEE bits (u32), bool and enum as one u8, u32 as u32, u64 and usize
/// as u64; all little-endian. The pinned config fields and the per-slot
/// counters go through here, so a kernel names a field and the encoding
/// follows from its type.
pub fn writeScalar(writer: *std.Io.Writer, value: anytype) !void {
    const T = @TypeOf(value);
    switch (T) {
        f32 => try writer.writeInt(u32, @bitCast(value), .little),
        bool => try writer.writeInt(u8, @intFromBool(value), .little),
        u32 => try writer.writeInt(u32, value, .little),
        u64 => try writer.writeInt(u64, value, .little),
        usize => try writer.writeInt(u64, @intCast(value), .little),
        else => switch (@typeInfo(T)) {
            .@"enum" => try writer.writeInt(u8, @intFromEnum(value), .little),
            else => @compileError("optim frame: no wire form for " ++ @typeName(T)),
        },
    }
}

/// Decode one record scalar written by `writeScalar` (the per-slot
/// counters: u64 and f32 only).
pub fn readScalar(reader: *std.Io.Reader, comptime T: type) !T {
    return switch (T) {
        f32 => @bitCast(try reader.takeInt(u32, .little)),
        u64 => try reader.takeInt(u64, .little),
        else => @compileError("optim frame: no record scalar of type " ++ @typeName(T)),
    };
}

/// Read one header scalar and require it to equal the live `value` (a
/// pinned structural config field), compared in wire form: f32 as bits,
/// enum as its tag, so a tag the enum does not name is a plain mismatch.
pub fn expectScalar(reader: *std.Io.Reader, value: anytype) !void {
    const T = @TypeOf(value);
    const same = switch (T) {
        f32 => try reader.takeInt(u32, .little) == @as(u32, @bitCast(value)),
        bool => try reader.takeInt(u8, .little) == @intFromBool(value),
        u32 => try reader.takeInt(u32, .little) == value,
        u64 => try reader.takeInt(u64, .little) == value,
        usize => try reader.takeInt(u64, .little) == value,
        else => switch (@typeInfo(T)) {
            .@"enum" => try reader.takeInt(u8, .little) == @intFromEnum(value),
            else => @compileError("optim frame: no wire form for " ++ @typeName(T)),
        },
    };
    if (!same) return OptimError.CheckpointConfigMismatch;
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

/// Storage bytes of one record field: `StateBuf`s and raw `[]f32`s carry
/// their buffer bytes (the v4/v5 dtype tag is not counted); scalars are
/// decoded into the staged State copy instead and contribute nothing.
pub fn recordFieldBytes(value: anytype) usize {
    return switch (@TypeOf(value)) {
        StateBuf => value.byteLen(),
        []f32 => 4 * value.len,
        else => 0,
    };
}

/// Write one per-slot record field by its type: a `StateBuf` as a
/// version-tagged state record, a raw `[]f32` as untagged f32 bytes
/// (Apollo's always-f32 moments, in every version), anything else through
/// `writeScalar`.
pub fn writeRecordField(writer: *std.Io.Writer, version: FrameVersion, value: anytype) !void {
    switch (@TypeOf(value)) {
        StateBuf => try writeStateSlice(writer, version, value),
        []f32 => try writeF32Slice(writer, value),
        else => try writeScalar(writer, value),
    }
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

/// The name -> slot-index map of one slot list, built once per save or
/// load. Building it validates every effective name: well-formed
/// (`validateName`) and collision-free, an explicit name colliding with an
/// unnamed slot's auto-name included. Auto-names are formatted into the
/// arena; explicit names are borrowed (they outlive the optimizer).
pub const SlotIndex = struct {
    arena: std.heap.ArenaAllocator,
    map: std.StringHashMapUnmanaged(usize) = .empty,

    pub fn build(allocator: Allocator, slots: anytype) !SlotIndex {
        var self: SlotIndex = .{ .arena = std.heap.ArenaAllocator.init(allocator) };
        errdefer self.deinit();
        const arena = self.arena.allocator();
        for (slots, 0..) |*slot, i| {
            const name = slot.param.name orelse try std.fmt.allocPrint(arena, "param{d}", .{i});
            try validateName(name);
            const entry = try self.map.getOrPut(arena, name);
            if (entry.found_existing) return OptimError.CheckpointDuplicateName;
            entry.value_ptr.* = i;
        }
        return self;
    }

    pub fn deinit(self: *SlotIndex) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

/// Validate one slot list's names before a save (see `SlotIndex.build`).
pub fn validateSlotNames(allocator: Allocator, slots: anytype) !void {
    var index = try SlotIndex.build(allocator, slots);
    index.deinit();
}

pub fn writeSlotName(writer: *std.Io.Writer, param: *const Param, index: usize) !void {
    var buf: [auto_name_buf_len]u8 = undefined;
    const name = slotName(param, index, &buf);
    try writer.writeInt(u16, @intCast(name.len), .little);
    try writer.writeAll(name);
}

/// Name-matches optimizer slot records to registered slots through a
/// `SlotIndex`, enforcing the fill-exactly-once contract: an unknown record
/// name, a record matching an already-filled slot, and a slot left unfilled
/// at the end all error. A slot list whose own names collide is rejected
/// at `init`, before any record is read.
pub const SlotMatcher = struct {
    index: SlotIndex,
    filled: []bool,
    name_buf: []u8,

    pub fn init(allocator: Allocator, slots: anytype) !SlotMatcher {
        var index = try SlotIndex.build(allocator, slots);
        errdefer index.deinit();
        const arena = index.arena.allocator();
        const filled = try arena.alloc(bool, slots.len);
        @memset(filled, false);
        const name_buf = try arena.alloc(u8, max_name_len);
        return .{ .index = index, .filled = filled, .name_buf = name_buf };
    }

    pub fn deinit(self: *SlotMatcher) void {
        self.index.deinit();
        self.* = undefined;
    }

    /// Read one record's name and resolve it to a not-yet-filled slot index.
    pub fn match(self: *SlotMatcher, reader: *std.Io.Reader) !usize {
        const name_len = try reader.takeInt(u16, .little);
        if (name_len == 0) return OptimError.CheckpointInvalidName;
        const name = self.name_buf[0..name_len];
        try reader.readSliceAll(name);
        const i = self.index.map.get(name) orelse return OptimError.CheckpointUnknownName;
        if (self.filled[i]) return OptimError.CheckpointDuplicateName;
        self.filled[i] = true;
        return i;
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
