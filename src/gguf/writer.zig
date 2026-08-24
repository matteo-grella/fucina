//! GGUF writing: the v3 `Writer` (metadata, tensor declarations, one-pass
//! `finish` or the `beginStream`/`DataStreamer` streaming path). Metadata
//! passthrough re-reads exact wire types from a parsed `File` through
//! reader.zig's `RawKvIter`.

const std = @import("std");

const wire = @import("wire.zig");
const reader = @import("reader.zig");

const Allocator = std.mem.Allocator;
const Cursor = wire.Cursor;
const Error = wire.Error;
const GgmlType = wire.GgmlType;
const MetaType = wire.MetaType;
const metaTypeForScalar = wire.metaTypeForScalar;
const writeScalarLittle = wire.writeScalarLittle;
const tensorByteLen = wire.tensorByteLen;
const File = reader.File;
const RawKvIter = reader.RawKvIter;

/// GGUF v3 writer: buffer metadata and tensor declarations, then `finish`
/// serializes header + KV section + tensor infos + alignment padding + tensor
/// data in one pass (tensor data offsets are precomputed, so nothing is
/// written twice).
///
/// Layout contract (mirrors ggml's gguf writer, `refs/llama.cpp/ggml/src/
/// gguf.cpp`): tensor offsets are relative to the data-section start and each
/// tensor's data is padded to `alignment` (default 32, tracked from any
/// `general.alignment` key added or copied) — llama.cpp's reader rejects
/// files whose offsets are not exactly the running padded total.
///
/// Ownership: metadata keys/payloads and tensor names are duplicated into the
/// writer (deinit frees them); tensor DATA is borrowed and must stay alive
/// until `finish` returns.
pub const Writer = struct {
    allocator: Allocator,
    /// Data-section alignment; updated when a `general.alignment` metadata
    /// key is added or copied, mirroring how the parser honors that key.
    alignment: usize,
    kvs: std.ArrayList(Kv),
    kv_index: std.StringHashMap(usize),
    tensors: std.ArrayList(PendingTensor),
    tensor_index: std.StringHashMap(usize),

    const Kv = struct {
        key: []u8,
        value_type: u32,
        /// Wire-encoded value bytes (strings carry their u64 length prefix,
        /// arrays their u32 item type + u64 length).
        payload: []u8,
    };

    const PendingTensor = struct {
        name: []u8,
        ggml_type: GgmlType,
        n_dims: usize,
        dims: [4]usize,
        /// Wire byte length (`tensorByteLen`), known at declaration time —
        /// offsets are computed from this, so headers never need the bytes.
        byte_len: usize,
        /// Borrowed until `finish` returns; null for tensors declared via
        /// `declareTensor`, whose bytes arrive through `beginStream`.
        data: ?[]const u8,
    };

    pub fn init(allocator: Allocator) Writer {
        return .{
            .allocator = allocator,
            .alignment = 32,
            .kvs = .empty,
            .kv_index = std.StringHashMap(usize).init(allocator),
            .tensors = .empty,
            .tensor_index = std.StringHashMap(usize).init(allocator),
        };
    }

    pub fn deinit(self: *Writer) void {
        for (self.kvs.items) |kv| {
            self.allocator.free(kv.key);
            self.allocator.free(kv.payload);
        }
        self.kvs.deinit(self.allocator);
        self.kv_index.deinit();
        for (self.tensors.items) |t| self.allocator.free(t.name);
        self.tensors.deinit(self.allocator);
        self.tensor_index.deinit();
        self.* = undefined;
    }

    /// Insert a KV record, taking ownership of `payload` (freed on error).
    /// Re-adding an existing key replaces its value in place (file order is
    /// kept) — GGUF keys must be unique.
    fn putKv(self: *Writer, key: []const u8, value_type: u32, payload: []u8) !void {
        errdefer self.allocator.free(payload);
        try self.noteSpecialKv(key, value_type, payload);
        if (self.kv_index.get(key)) |kv_i| {
            const kv = &self.kvs.items[kv_i];
            self.allocator.free(kv.payload);
            kv.value_type = value_type;
            kv.payload = payload;
            return;
        }
        const owned_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned_key);
        try self.kvs.append(self.allocator, .{ .key = owned_key, .value_type = value_type, .payload = payload });
        errdefer _ = self.kvs.pop();
        try self.kv_index.put(owned_key, self.kvs.items.len - 1);
    }

    /// `general.alignment` changes the data-section padding rule, so track it
    /// no matter how it was added. llama.cpp requires this key's wire type to
    /// be uint32 and the value to be a power of two.
    fn noteSpecialKv(self: *Writer, key: []const u8, value_type: u32, payload: []const u8) !void {
        if (!std.mem.eql(u8, key, "general.alignment")) return;
        if (value_type != @intFromEnum(MetaType.uint32)) return Error.InvalidAlignment;
        var cursor = Cursor{ .bytes = payload };
        const value = try cursor.readInt(u32);
        if (value == 0 or (value & (value - 1)) != 0) return Error.InvalidAlignment;
        // Upper bound: a runaway alignment would make `finish` pad every
        // tensor to gigabytes. 1 MiB is far beyond any real GGUF (default 32).
        if (value > (1 << 20)) return Error.InvalidAlignment;
        self.alignment = @intCast(value);
    }

    pub fn addMetaString(self: *Writer, key: []const u8, value: []const u8) !void {
        const payload = try self.allocator.alloc(u8, 8 + value.len);
        std.mem.writeInt(u64, payload[0..8], value.len, .little);
        @memcpy(payload[8..], value);
        try self.putKv(key, @intFromEnum(MetaType.string), payload);
    }

    /// `Int` selects the exact wire type (u8/i8/u16/i16/u32/i32/u64/i64) —
    /// llama.cpp type-checks many keys, so passthrough-adjacent metadata must
    /// keep its original width.
    pub fn addMetaInt(self: *Writer, key: []const u8, comptime Int: type, value: Int) !void {
        const payload = try self.allocator.alloc(u8, @sizeOf(Int));
        writeScalarLittle(Int, payload, value);
        try self.putKv(key, @intFromEnum(metaTypeForScalar(Int)), payload);
    }

    /// `Float` selects the wire type (f32/f64).
    pub fn addMetaFloat(self: *Writer, key: []const u8, comptime Float: type, value: Float) !void {
        const payload = try self.allocator.alloc(u8, @sizeOf(Float));
        writeScalarLittle(Float, payload, value);
        try self.putKv(key, @intFromEnum(metaTypeForScalar(Float)), payload);
    }

    pub fn addMetaBool(self: *Writer, key: []const u8, value: bool) !void {
        const payload = try self.allocator.alloc(u8, 1);
        payload[0] = @intFromBool(value);
        try self.putKv(key, @intFromEnum(MetaType.boolean), payload);
    }

    /// Array of scalars; `Elem` selects the wire item type.
    pub fn addMetaArray(self: *Writer, key: []const u8, comptime Elem: type, values: []const Elem) !void {
        const payload = try self.allocator.alloc(u8, 4 + 8 + values.len * @sizeOf(Elem));
        std.mem.writeInt(u32, payload[0..4], @intFromEnum(metaTypeForScalar(Elem)), .little);
        std.mem.writeInt(u64, payload[4..12], values.len, .little);
        for (values, 0..) |value, value_i| {
            writeScalarLittle(Elem, payload[12 + value_i * @sizeOf(Elem) ..], value);
        }
        try self.putKv(key, @intFromEnum(MetaType.array), payload);
    }

    pub fn addMetaStringArray(self: *Writer, key: []const u8, values: []const []const u8) !void {
        var payload_len: usize = 4 + 8;
        for (values) |value| payload_len += 8 + value.len;
        const payload = try self.allocator.alloc(u8, payload_len);
        std.mem.writeInt(u32, payload[0..4], @intFromEnum(MetaType.string), .little);
        std.mem.writeInt(u64, payload[4..12], values.len, .little);
        var offset: usize = 12;
        for (values) |value| {
            std.mem.writeInt(u64, payload[offset..][0..8], value.len, .little);
            @memcpy(payload[offset + 8 ..][0..value.len], value);
            offset += 8 + value.len;
        }
        try self.putKv(key, @intFromEnum(MetaType.array), payload);
    }

    /// Copy one metadata entry from a parsed file byte-verbatim, preserving
    /// the exact wire value type that the parser's widened `Value` map drops.
    /// `from` must still own its file bytes (before `deinit`/`takeMapping`).
    pub fn addMetaCopy(self: *Writer, from: *const File, key: []const u8) !void {
        var it = try RawKvIter.init(from.bytes);
        while (try it.next()) |kv| {
            if (!std.mem.eql(u8, kv.key, key)) continue;
            const payload = try self.allocator.dupe(u8, kv.payload);
            return self.putKv(kv.key, kv.value_type, payload);
        }
        return Error.KeyNotFound;
    }

    /// Byte-verbatim passthrough of every metadata entry except `skip_keys`,
    /// in the source file's order. Same lifetime requirement as `addMetaCopy`.
    pub fn copyAllMetadata(self: *Writer, from: *const File, skip_keys: []const []const u8) !void {
        return self.copyAllMetadataRaw(from.bytes, skip_keys);
    }

    /// `copyAllMetadata` over a raw GGUF byte region — for callers whose
    /// `File` transferred its mapping away (`takeMapping`) while the region
    /// itself is still alive.
    pub fn copyAllMetadataRaw(self: *Writer, file_bytes: []const u8, skip_keys: []const []const u8) !void {
        var it = try RawKvIter.init(file_bytes);
        outer: while (try it.next()) |kv| {
            for (skip_keys) |skip| {
                if (std.mem.eql(u8, kv.key, skip)) continue :outer;
            }
            const payload = try self.allocator.dupe(u8, kv.payload);
            try self.putKv(kv.key, kv.value_type, payload);
        }
    }

    /// Declare a tensor. `dims` are GGUF `ne[]` order — innermost/
    /// fastest-varying axis FIRST, exactly as the parser surfaces
    /// `TensorInfo.dims` (so re-emitting is `addTensor(info.name,
    /// info.ggml_type, info.dims[0..info.n_dims], info.data)`). A Fucina
    /// row-major logical [out, in] matrix is therefore `&.{ in, out }`.
    /// `data` are the wire bytes for `ggml_type` (length must equal
    /// `tensorByteLen`) and are BORROWED until `finish` returns.
    pub fn addTensor(self: *Writer, name: []const u8, ggml_type: GgmlType, dims: []const usize, data: []const u8) !void {
        try self.appendTensor(name, ggml_type, dims, data);
    }

    /// `addTensor` without the bytes: declares name/type/dims (same
    /// validation, offsets come from `tensorByteLen`) and defers the payload
    /// to the `beginStream` data phase. For outputs too large to hold every
    /// tensor buffer at once: declare the full tensor set, then stream each
    /// tensor's bytes in declaration order, releasing each buffer before
    /// producing the next.
    pub fn declareTensor(self: *Writer, name: []const u8, ggml_type: GgmlType, dims: []const usize) !void {
        try self.appendTensor(name, ggml_type, dims, null);
    }

    fn appendTensor(self: *Writer, name: []const u8, ggml_type: GgmlType, dims: []const usize, data: ?[]const u8) !void {
        if (name.len == 0) return Error.InvalidTensorInfo;
        if (dims.len == 0 or dims.len > 4) return Error.InvalidTensorInfo;
        const byte_len = try tensorByteLen(ggml_type, dims);
        if (data) |bytes| {
            if (bytes.len != byte_len) return Error.InvalidTensorInfo;
        }
        if (self.tensor_index.contains(name)) return Error.DuplicateTensorName;

        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        var entry = PendingTensor{
            .name = owned_name,
            .ggml_type = ggml_type,
            .n_dims = dims.len,
            .dims = .{ 0, 0, 0, 0 },
            .byte_len = byte_len,
            .data = data,
        };
        for (dims, 0..) |dim, dim_i| entry.dims[dim_i] = dim;
        try self.tensors.append(self.allocator, entry);
        errdefer _ = self.tensors.pop();
        try self.tensor_index.put(owned_name, self.tensors.items.len - 1);
    }

    /// Header, KV section, tensor infos (offsets are the llama.cpp running
    /// padded total, relative to the data-section start), padding to
    /// `alignment` — everything except tensor data.
    fn writeHeader(self: *const Writer, out: *std.Io.Writer) !void {
        try out.writeAll("GGUF");
        try out.writeInt(u32, 3, .little);
        try out.writeInt(u64, @intCast(self.tensors.items.len), .little);
        try out.writeInt(u64, @intCast(self.kvs.items.len), .little);
        var header_len: usize = 4 + 4 + 8 + 8;

        for (self.kvs.items) |kv| {
            try out.writeInt(u64, @intCast(kv.key.len), .little);
            try out.writeAll(kv.key);
            try out.writeInt(u32, kv.value_type, .little);
            try out.writeAll(kv.payload);
            header_len += 8 + kv.key.len + 4 + kv.payload.len;
        }

        var data_offset: usize = 0;
        for (self.tensors.items) |t| {
            try out.writeInt(u64, @intCast(t.name.len), .little);
            try out.writeAll(t.name);
            try out.writeInt(u32, @intCast(t.n_dims), .little);
            for (t.dims[0..t.n_dims]) |dim| try out.writeInt(u64, @intCast(dim), .little);
            try out.writeInt(u32, @intFromEnum(t.ggml_type), .little);
            try out.writeInt(u64, @intCast(data_offset), .little);
            header_len += 8 + t.name.len + 4 + t.n_dims * 8 + 4 + 8;
            data_offset = try std.math.add(usize, data_offset, std.mem.alignForward(usize, t.byte_len, self.alignment));
        }

        try out.splatByteAll(0, std.mem.alignForward(usize, header_len, self.alignment) - header_len);
    }

    /// Serialize everything: `writeHeader`, then each tensor's borrowed data
    /// padded to `alignment` (including the last, matching ggml's writer).
    /// Tensors declared without data (`declareTensor`) belong to the
    /// streaming path and make this fail with `Error.TensorDataMissing`.
    pub fn finish(self: *const Writer, out: *std.Io.Writer) !void {
        try self.writeHeader(out);
        for (self.tensors.items) |t| {
            const data = t.data orelse return Error.TensorDataMissing;
            try out.writeAll(data);
            try out.splatByteAll(0, std.mem.alignForward(usize, t.byte_len, self.alignment) - t.byte_len);
        }
    }

    /// The streaming counterpart of `finish`: write the complete header now
    /// (declare/add every tensor BEFORE calling — the header pins names,
    /// dims, and offsets) and return a `DataStreamer` that feeds each
    /// tensor's bytes in declaration order. Tensors added WITH data still
    /// stream — pass their bytes (or any equal-length buffer) at their turn.
    pub fn beginStream(self: *const Writer, out: *std.Io.Writer) !DataStreamer {
        try self.writeHeader(out);
        return .{ .writer = self, .out = out };
    }

    /// Data-phase companion of `beginStream`. Each `writeTensorData` call
    /// writes the next tensor's bytes plus alignment padding straight
    /// through, so the caller can free/release every buffer before producing
    /// the next — the writer never borrows tensor data in this mode.
    pub const DataStreamer = struct {
        writer: *const Writer,
        out: *std.Io.Writer,
        next_index: usize = 0,

        /// Name of the tensor the next `writeTensorData` call must supply;
        /// null once every declared tensor has been written.
        pub fn nextTensorName(self: *const DataStreamer) ?[]const u8 {
            if (self.next_index >= self.writer.tensors.items.len) return null;
            return self.writer.tensors.items[self.next_index].name;
        }

        /// Write the next tensor's wire bytes. Length must match the
        /// declaration (`Error.InvalidTensorInfo`); writing past the
        /// declared set is `Error.TensorDataMissing`. Written in <= 1 GiB
        /// slices: a single `write(2)` of 2 GiB+ fails with EINVAL on
        /// macOS, and slab-native expert records make multi-GiB tensors
        /// routine.
        pub fn writeTensorData(self: *DataStreamer, data: []const u8) !void {
            const tensors = self.writer.tensors.items;
            if (self.next_index >= tensors.len) return Error.TensorDataMissing;
            const t = &tensors[self.next_index];
            if (data.len != t.byte_len) return Error.InvalidTensorInfo;
            var index: usize = 0;
            while (index < data.len) {
                const n = @min(data.len - index, @as(usize, 1) << 30);
                try self.out.writeAll(data[index..][0..n]);
                index += n;
            }
            try self.out.splatByteAll(0, std.mem.alignForward(usize, t.byte_len, self.writer.alignment) - t.byte_len);
            self.next_index += 1;
        }

        /// The file is complete only when every declared tensor was
        /// streamed; anything less is `Error.TensorDataMissing`. The caller
        /// still owns the output flush.
        pub fn finish(self: *const DataStreamer) !void {
            if (self.next_index != self.writer.tensors.items.len) return Error.TensorDataMissing;
        }
    };
};
