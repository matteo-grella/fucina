//! Hugging Face safetensors reader/writer.
//!
//! Format summary (current upstream contract): u64 little-endian JSON header
//! length, UTF-8 JSON header, then one contiguous tensor data buffer. Tensor
//! offsets are relative to the start of that data buffer and must cover it
//! exactly with no holes. Header JSON is padded to 8 bytes with spaces when
//! writing so the first data byte is naturally aligned for scalar dtypes.
const std = @import("std");
const builtin = @import("builtin");
const dtype_mod = @import("dtype.zig");

const Allocator = std.mem.Allocator;
const FucinaDType = dtype_mod.DType;

const n_len: usize = 8;
pub const max_header_size: usize = 100_000_000;

pub const Error = error{
    HeaderTooLarge,
    HeaderTooSmall,
    InvalidHeaderLength,
    InvalidHeader,
    InvalidHeaderDeserialization,
    TensorNotFound,
    TensorInvalidInfo,
    InvalidOffset,
    MetadataIncompleteBuffer,
    ValidationOverflow,
    MisalignedSlice,
    DuplicateTensorName,
    InvalidTensorName,
    InvalidMetadata,
    UnsupportedDtype,
    InvalidSlice,
};

pub const DType = enum {
    BOOL,
    F4,
    F6_E2M3,
    F6_E3M2,
    U8,
    I8,
    F8_E5M2,
    F8_E4M3,
    F8_E8M0,
    F8_E4M3FNUZ,
    F8_E5M2FNUZ,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    C64,
    F64,
    I64,
    U64,

    pub fn bitsize(self: DType) usize {
        return switch (self) {
            .F4 => 4,
            .F6_E2M3, .F6_E3M2 => 6,
            .BOOL, .U8, .I8, .F8_E5M2, .F8_E4M3, .F8_E8M0, .F8_E4M3FNUZ, .F8_E5M2FNUZ => 8,
            .I16, .U16, .F16, .BF16 => 16,
            .I32, .U32, .F32 => 32,
            .C64, .F64, .I64, .U64 => 64,
        };
    }

    pub fn string(self: DType) []const u8 {
        return switch (self) {
            .BOOL => "BOOL",
            .F4 => "F4",
            .F6_E2M3 => "F6_E2M3",
            .F6_E3M2 => "F6_E3M2",
            .U8 => "U8",
            .I8 => "I8",
            .F8_E5M2 => "F8_E5M2",
            .F8_E4M3 => "F8_E4M3",
            .F8_E8M0 => "F8_E8M0",
            .F8_E4M3FNUZ => "F8_E4M3FNUZ",
            .F8_E5M2FNUZ => "F8_E5M2FNUZ",
            .I16 => "I16",
            .U16 => "U16",
            .F16 => "F16",
            .BF16 => "BF16",
            .I32 => "I32",
            .U32 => "U32",
            .F32 => "F32",
            .C64 => "C64",
            .F64 => "F64",
            .I64 => "I64",
            .U64 => "U64",
        };
    }

    fn fromString(value: []const u8) !DType {
        inline for (@typeInfo(DType).@"enum".fields) |field| {
            const dt: DType = @enumFromInt(field.value);
            if (std.mem.eql(u8, value, dt.string())) return dt;
        }
        return Error.UnsupportedDtype;
    }
};

pub fn dtypeFromFucina(dtype: FucinaDType) !DType {
    return switch (dtype) {
        .f32 => .F32,
        .f16 => .F16,
        .bf16 => .BF16,
        else => Error.UnsupportedDtype,
    };
}

pub fn dtypeToFucina(dtype: DType) !FucinaDType {
    return switch (dtype) {
        .F32 => .f32,
        .F16 => .f16,
        .BF16 => .bf16,
        else => Error.UnsupportedDtype,
    };
}

pub const MetadataEntry = struct {
    key: []const u8,
    value: []const u8,
};

pub const Tensor = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    data: []const u8,
};

pub const TensorInfo = struct {
    name: []const u8,
    dtype: DType,
    shape: []usize,
    data_offsets: [2]usize,
    data: []const u8,

    pub const Slice = struct {
        start: usize = 0,
        end: ?usize = null,
    };

    pub fn sliceBytesAlloc(self: *const TensorInfo, allocator: Allocator, ranges: []const Slice) ![]u8 {
        if (ranges.len > self.shape.len) return Error.InvalidSlice;
        const bits = self.dtype.bitsize();
        if (bits % 8 != 0) return Error.MisalignedSlice;
        const elem_size = bits / 8;

        const rank = self.shape.len;
        if (rank == 0) return allocator.dupe(u8, self.data);

        const starts = try allocator.alloc(usize, rank);
        defer allocator.free(starts);
        const extents = try allocator.alloc(usize, rank);
        defer allocator.free(extents);
        const source_strides = try allocator.alloc(usize, rank);
        defer allocator.free(source_strides);
        const output_strides = try allocator.alloc(usize, rank);
        defer allocator.free(output_strides);

        var source_stride: usize = 1;
        var axis = rank;
        while (axis > 0) {
            axis -= 1;
            source_strides[axis] = source_stride;
            source_stride = std.math.mul(usize, source_stride, self.shape[axis]) catch return Error.ValidationOverflow;
        }

        var total_elems: usize = 1;
        for (0..rank) |i| {
            const range = if (i < ranges.len) ranges[i] else Slice{};
            const dim = self.shape[i];
            const end = range.end orelse dim;
            if (range.start > end or end > dim) return Error.InvalidSlice;
            starts[i] = range.start;
            extents[i] = end - range.start;
            total_elems = std.math.mul(usize, total_elems, extents[i]) catch return Error.ValidationOverflow;
        }

        var output_stride: usize = 1;
        axis = rank;
        while (axis > 0) {
            axis -= 1;
            output_strides[axis] = output_stride;
            output_stride = std.math.mul(usize, output_stride, extents[axis]) catch return Error.ValidationOverflow;
        }

        const byte_len = std.math.mul(usize, total_elems, elem_size) catch return Error.ValidationOverflow;
        const out = try allocator.alloc(u8, byte_len);
        errdefer allocator.free(out);
        for (0..total_elems) |out_elem| {
            var src_elem: usize = 0;
            for (0..rank) |i| {
                const coord = if (extents[i] == 0) 0 else (out_elem / output_strides[i]) % extents[i];
                src_elem += (starts[i] + coord) * source_strides[i];
            }
            const dst_byte = out_elem * elem_size;
            const src_byte = src_elem * elem_size;
            @memcpy(out[dst_byte..][0..elem_size], self.data[src_byte..][0..elem_size]);
        }
        return out;
    }
};

pub const File = struct {
    allocator: Allocator,
    bytes: []const u8,
    tensors: []TensorInfo,
    metadata: std.StringHashMap([]const u8),
    index: std.StringHashMap(usize),
    ownership: Ownership = .borrowed,

    const Ownership = enum { borrowed, owned, mmap };

    pub fn parse(allocator: Allocator, bytes: []const u8) !File {
        return parseWithOwnership(allocator, bytes, .borrowed);
    }

    pub fn parseOwned(allocator: Allocator, bytes: []u8) !File {
        errdefer allocator.free(bytes);
        return parseWithOwnership(allocator, bytes, .owned);
    }

    pub fn load(allocator: Allocator, io: std.Io, path: []const u8) !File {
        var handle = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer handle.close(io);
        const stat = try handle.stat(io);
        if (stat.kind != .file) return error.IsDir;
        const file_len: usize = @intCast(stat.size);
        const bytes = try allocator.alloc(u8, file_len);
        {
            errdefer allocator.free(bytes);
            var read_len: usize = 0;
            while (read_len < bytes.len) {
                const n = try handle.readStreaming(io, &.{bytes[read_len..]});
                if (n == 0) return error.EndOfStream;
                read_len += n;
            }
        }
        return parseOwned(allocator, bytes);
    }

    pub fn loadMmap(allocator: Allocator, io: std.Io, path: []const u8) !File {
        var handle = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer handle.close(io);
        const stat = try handle.stat(io);
        if (stat.kind != .file) return error.IsDir;
        const file_len: usize = @intCast(stat.size);
        if (file_len == 0) return Error.HeaderTooSmall;
        const mapped = try std.posix.mmap(null, file_len, .{ .READ = true }, .{ .TYPE = .PRIVATE }, handle.handle, 0);
        errdefer std.posix.munmap(mapped);
        return parseWithOwnership(allocator, mapped, .mmap);
    }

    pub fn deinit(self: *File) void {
        for (self.tensors) |*info| {
            self.allocator.free(info.name);
            self.allocator.free(info.shape);
        }
        self.allocator.free(self.tensors);
        var meta_it = self.metadata.iterator();
        while (meta_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
        self.index.deinit();
        switch (self.ownership) {
            .borrowed => {},
            .owned => self.allocator.free(@constCast(self.bytes)),
            .mmap => std.posix.munmap(@alignCast(self.bytes)),
        }
        self.* = undefined;
    }

    pub fn tensor(self: *const File, name: []const u8) !*const TensorInfo {
        const i = self.index.get(name) orelse return Error.TensorNotFound;
        return &self.tensors[i];
    }

    pub fn maybeTensor(self: *const File, name: []const u8) ?*const TensorInfo {
        const i = self.index.get(name) orelse return null;
        return &self.tensors[i];
    }

    pub fn names(self: *const File) []const TensorInfo {
        return self.tensors;
    }

    pub fn tensorNames(self: *const File, allocator: Allocator) ![][]const u8 {
        const out = try allocator.alloc([]const u8, self.tensors.len);
        for (self.tensors, out) |*info, *name| name.* = info.name;
        return out;
    }

    pub fn len(self: *const File) usize {
        return self.tensors.len;
    }

    pub fn isEmpty(self: *const File) bool {
        return self.tensors.len == 0;
    }
};

pub fn readPrefix(allocator: Allocator, reader: *std.Io.Reader) !File {
    const n = try reader.takeInt(u64, .little);
    if (n > max_header_size) return Error.HeaderTooLarge;
    const header_len: usize = @intCast(n);
    const header = try allocator.alloc(u8, header_len);
    defer allocator.free(header);
    try reader.readSliceAll(header);

    var header_file = try parseHeader(allocator, header, &.{});
    defer deinitHeaderOnly(&header_file);
    const data_len = dataLen(&header_file);

    const header_total = std.math.add(usize, n_len, header_len) catch return Error.ValidationOverflow;
    const total_len = std.math.add(usize, header_total, data_len) catch return Error.ValidationOverflow;
    const bytes = try allocator.alloc(u8, total_len);
    {
        errdefer allocator.free(bytes);
        std.mem.writeInt(u64, bytes[0..8], n, .little);
        @memcpy(bytes[n_len..][0..header_len], header);
        try reader.readSliceAll(bytes[header_total..]);
    }

    return File.parseOwned(allocator, bytes);
}

pub fn serialize(allocator: Allocator, writer: *std.Io.Writer, tensors: []const Tensor, metadata: ?[]const MetadataEntry) !void {
    var prepared = try prepare(allocator, tensors, metadata);
    defer prepared.deinit(allocator);
    if (prepared.header.len > max_header_size) return Error.HeaderTooLarge;

    try writer.writeInt(u64, prepared.header.len, .little);
    try writer.writeAll(prepared.header);
    for (prepared.tensors) |*tensor| try writer.writeAll(tensor.data);
}

pub fn serializeAlloc(allocator: Allocator, tensors: []const Tensor, metadata: ?[]const MetadataEntry) ![]u8 {
    var prepared = try prepare(allocator, tensors, metadata);
    defer prepared.deinit(allocator);
    if (prepared.header.len > max_header_size) return Error.HeaderTooLarge;

    const header_total = std.math.add(usize, n_len, prepared.header.len) catch return Error.ValidationOverflow;
    const total_len = std.math.add(usize, header_total, prepared.data_len) catch return Error.ValidationOverflow;
    var out = try allocator.alloc(u8, total_len);
    errdefer allocator.free(out);
    std.mem.writeInt(u64, out[0..8], prepared.header.len, .little);
    @memcpy(out[n_len..][0..prepared.header.len], prepared.header);
    var offset = header_total;
    for (prepared.tensors) |*tensor| {
        @memcpy(out[offset..][0..tensor.data.len], tensor.data);
        offset += tensor.data.len;
    }
    return out;
}

pub fn saveFileAtomic(allocator: Allocator, io: std.Io, path: []const u8, tensors: []const Tensor, metadata: ?[]const MetadataEntry) !void {
    var prepared = try prepare(allocator, tensors, metadata);
    defer prepared.deinit(allocator);
    if (prepared.header.len > max_header_size) return Error.HeaderTooLarge;
    const header_total = try std.math.add(usize, n_len, prepared.header.len);
    const total_len = try std.math.add(usize, header_total, prepared.data_len);

    var tmp_buf: [512]u8 = undefined;
    const tmp_path = try std.fmt.bufPrint(&tmp_buf, "{s}.tmp.{d}", .{ path, std.Io.Clock.real.now(io).nanoseconds });
    {
        var file = try std.Io.Dir.cwd().createFile(io, tmp_path, .{});
        errdefer std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};
        defer file.close(io);
        try file.setLength(io, @intCast(total_len));
        optimizeSequentialWrite(file);
        var buffer: [1024 * 1024]u8 = undefined;
        var writer = file.writer(io, &buffer);
        try writer.interface.writeInt(u64, prepared.header.len, .little);
        try writer.interface.writeAll(prepared.header);
        for (prepared.tensors) |*tensor| try writer.interface.writeAll(tensor.data);
        try writer.interface.flush();
    }
    std.Io.Dir.cwd().rename(tmp_path, std.Io.Dir.cwd(), path, io) catch |err| {
        std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};
        return err;
    };
}

fn optimizeSequentialWrite(file: std.Io.File) void {
    if (builtin.os.tag == .macos) {
        _ = std.c.fcntl(file.handle, std.c.F.NOCACHE, @as(c_int, 1));
    }
}

const Prepared = struct {
    header: []u8,
    tensors: []Tensor,
    data_len: usize,

    fn deinit(self: *Prepared, allocator: Allocator) void {
        allocator.free(self.header);
        allocator.free(self.tensors);
        self.* = undefined;
    }
};

fn prepare(allocator: Allocator, tensors: []const Tensor, metadata: ?[]const MetadataEntry) !Prepared {
    const sorted = try allocator.dupe(Tensor, tensors);
    errdefer allocator.free(sorted);
    std.mem.sort(Tensor, sorted, {}, tensorLessThan);

    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();
    try seen.ensureTotalCapacity(@intCast(sorted.len));
    var offset: usize = 0;
    for (sorted) |*tensor| {
        try validateTensorName(tensor.name);
        if (tensor.data.len != try byteLen(tensor.dtype, tensor.shape)) return Error.TensorInvalidInfo;
        const gop = seen.getOrPutAssumeCapacity(tensor.name);
        if (gop.found_existing) return Error.DuplicateTensorName;
        offset = std.math.add(usize, offset, tensor.data.len) catch return Error.ValidationOverflow;
    }

    if (metadata) |entries| try validateMetadata(allocator, entries);

    var header: std.ArrayList(u8) = .empty;
    errdefer header.deinit(allocator);
    try header.append(allocator, '{');
    var needs_comma = false;
    if (metadata) |entries| {
        if (entries.len != 0) {
            try writeJsonString(&header, allocator, "__metadata__");
            try header.append(allocator, ':');
            try header.append(allocator, '{');
            for (entries, 0..) |entry, i| {
                if (i != 0) try header.append(allocator, ',');
                try writeJsonString(&header, allocator, entry.key);
                try header.append(allocator, ':');
                try writeJsonString(&header, allocator, entry.value);
            }
            try header.append(allocator, '}');
            needs_comma = true;
        }
    }

    offset = 0;
    for (sorted) |*tensor| {
        if (needs_comma) try header.append(allocator, ',');
        needs_comma = true;
        try writeJsonString(&header, allocator, tensor.name);
        try header.appendSlice(allocator, ":{\"dtype\":\"");
        try header.appendSlice(allocator, tensor.dtype.string());
        try header.appendSlice(allocator, "\",\"shape\":[");
        for (tensor.shape, 0..) |dim, i| {
            if (i != 0) try header.append(allocator, ',');
            try appendInt(&header, allocator, dim);
        }
        const end = std.math.add(usize, offset, tensor.data.len) catch return Error.ValidationOverflow;
        try header.appendSlice(allocator, "],\"data_offsets\":[");
        try appendInt(&header, allocator, offset);
        try header.append(allocator, ',');
        try appendInt(&header, allocator, end);
        try header.appendSlice(allocator, "]}");
        offset = end;
    }
    try header.append(allocator, '}');
    const aligned_len = std.mem.alignForward(usize, header.items.len, n_len);
    try header.appendNTimes(allocator, ' ', aligned_len - header.items.len);

    return .{ .header = try header.toOwnedSlice(allocator), .tensors = sorted, .data_len = offset };
}

fn tensorLessThan(_: void, lhs: Tensor, rhs: Tensor) bool {
    const l_ord = @intFromEnum(lhs.dtype);
    const r_ord = @intFromEnum(rhs.dtype);
    if (l_ord != r_ord) return l_ord > r_ord;
    return std.mem.order(u8, lhs.name, rhs.name) == .lt;
}

fn parseWithOwnership(allocator: Allocator, bytes: []const u8, ownership: File.Ownership) !File {
    if (bytes.len < n_len) return Error.HeaderTooSmall;
    const n_u64 = std.mem.readInt(u64, bytes[0..8], .little);
    if (n_u64 > max_header_size) return Error.HeaderTooLarge;
    const n: usize = @intCast(n_u64);
    const stop = std.math.add(usize, n_len, n) catch return Error.InvalidHeaderLength;
    if (stop > bytes.len) return Error.InvalidHeaderLength;

    var file = try parseHeader(allocator, bytes[n_len..stop], bytes[stop..]);
    errdefer file.deinit();
    const len = dataLen(&file);
    const expected_len = std.math.add(usize, stop, len) catch return Error.ValidationOverflow;
    if (expected_len != bytes.len) return Error.MetadataIncompleteBuffer;
    for (file.tensors) |*tensor| {
        tensor.data = bytes[stop + tensor.data_offsets[0] .. stop + tensor.data_offsets[1]];
    }
    file.bytes = bytes;
    file.ownership = ownership;
    return file;
}

fn parseHeader(allocator: Allocator, header: []const u8, data: []const u8) !File {
    if (!std.unicode.utf8ValidateSlice(header)) return Error.InvalidHeader;
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, header, .{}) catch return Error.InvalidHeaderDeserialization;
    defer parsed.deinit();
    if (parsed.value != .object) return Error.InvalidHeaderDeserialization;
    const object = parsed.value.object;

    var tensor_list: std.ArrayList(TensorInfo) = .empty;
    errdefer {
        for (tensor_list.items) |*tensor| {
            allocator.free(tensor.name);
            allocator.free(tensor.shape);
        }
        tensor_list.deinit(allocator);
    }
    var metadata = std.StringHashMap([]const u8).init(allocator);
    errdefer {
        var it = metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        metadata.deinit();
    }

    var it = object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.eql(u8, key, "__metadata__")) {
            try parseMetadataObject(allocator, &metadata, entry.value_ptr.*);
            continue;
        }
        try validateName(key);
        const info = try parseTensorInfo(allocator, key, entry.value_ptr.*);
        errdefer {
            allocator.free(info.name);
            allocator.free(info.shape);
        }
        try tensor_list.append(allocator, info);
    }
    std.mem.sort(TensorInfo, tensor_list.items, {}, tensorInfoLessThan);

    var index = std.StringHashMap(usize).init(allocator);
    errdefer index.deinit();
    try index.ensureTotalCapacity(@intCast(tensor_list.items.len));
    var start: usize = 0;
    for (tensor_list.items, 0..) |*tensor, i| {
        if (tensor.data_offsets[0] != start or tensor.data_offsets[1] < tensor.data_offsets[0]) return Error.InvalidOffset;
        const expected = try byteLen(tensor.dtype, tensor.shape);
        if (tensor.data_offsets[1] - tensor.data_offsets[0] != expected) return Error.TensorInvalidInfo;
        if (tensor.data_offsets[1] > data.len and data.len != 0) return Error.MetadataIncompleteBuffer;
        const gop = try index.getOrPut(tensor.name);
        if (gop.found_existing) return Error.DuplicateTensorName;
        gop.value_ptr.* = i;
        start = tensor.data_offsets[1];
    }

    return .{
        .allocator = allocator,
        .bytes = &.{},
        .tensors = try tensor_list.toOwnedSlice(allocator),
        .metadata = metadata,
        .index = index,
    };
}

fn parseMetadataObject(allocator: Allocator, metadata: *std.StringHashMap([]const u8), value: std.json.Value) !void {
    if (value != .object) return Error.InvalidMetadata;
    var it = value.object.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.* != .string) return Error.InvalidMetadata;
        const key = try allocator.dupe(u8, entry.key_ptr.*);
        errdefer allocator.free(key);
        const val = try allocator.dupe(u8, entry.value_ptr.*.string);
        errdefer allocator.free(val);
        const gop = try metadata.getOrPut(key);
        if (gop.found_existing) return Error.InvalidMetadata;
        gop.value_ptr.* = val;
    }
}

fn parseTensorInfo(allocator: Allocator, name: []const u8, value: std.json.Value) !TensorInfo {
    if (value != .object) return Error.InvalidHeaderDeserialization;
    const object = value.object;
    const dtype_value = object.get("dtype") orelse return Error.InvalidHeaderDeserialization;
    if (dtype_value != .string) return Error.InvalidHeaderDeserialization;
    const dtype = try DType.fromString(dtype_value.string);

    const shape_value = object.get("shape") orelse return Error.InvalidHeaderDeserialization;
    if (shape_value != .array) return Error.InvalidHeaderDeserialization;
    const shape = try allocator.alloc(usize, shape_value.array.items.len);
    errdefer allocator.free(shape);
    for (shape_value.array.items, shape) |item, *dim| dim.* = try jsonUsize(item);

    const offsets_value = object.get("data_offsets") orelse return Error.InvalidHeaderDeserialization;
    if (offsets_value != .array or offsets_value.array.items.len != 2) return Error.InvalidHeaderDeserialization;
    const begin = try jsonUsize(offsets_value.array.items[0]);
    const end = try jsonUsize(offsets_value.array.items[1]);

    const owned_name = try allocator.dupe(u8, name);
    errdefer allocator.free(owned_name);
    return .{
        .name = owned_name,
        .dtype = dtype,
        .shape = shape,
        .data_offsets = .{ begin, end },
        .data = &.{},
    };
}

fn tensorInfoLessThan(_: void, lhs: TensorInfo, rhs: TensorInfo) bool {
    if (lhs.data_offsets[0] != rhs.data_offsets[0]) return lhs.data_offsets[0] < rhs.data_offsets[0];
    return std.mem.order(u8, lhs.name, rhs.name) == .lt;
}

fn byteLen(dtype: DType, shape: []const usize) !usize {
    var elems: usize = 1;
    for (shape) |dim| elems = std.math.mul(usize, elems, dim) catch return Error.ValidationOverflow;
    const nbits = std.math.mul(usize, elems, dtype.bitsize()) catch return Error.ValidationOverflow;
    if (nbits % 8 != 0) return Error.MisalignedSlice;
    return nbits / 8;
}

fn jsonUsize(value: std.json.Value) !usize {
    return switch (value) {
        .integer => |v| if (v < 0) Error.InvalidHeaderDeserialization else std.math.cast(usize, v) orelse Error.ValidationOverflow,
        .number_string => |v| std.fmt.parseInt(usize, v, 10) catch |err| switch (err) {
            error.Overflow => Error.ValidationOverflow,
            error.InvalidCharacter => Error.InvalidHeaderDeserialization,
        },
        else => Error.InvalidHeaderDeserialization,
    };
}

fn validateMetadata(allocator: Allocator, entries: []const MetadataEntry) !void {
    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();
    try seen.ensureTotalCapacity(@intCast(entries.len));
    for (entries) |entry| {
        try validateName(entry.key);
        if (!std.unicode.utf8ValidateSlice(entry.value)) return Error.InvalidMetadata;
        const gop = seen.getOrPutAssumeCapacity(entry.key);
        if (gop.found_existing) return Error.InvalidMetadata;
    }
}

fn validateName(name: []const u8) !void {
    if (name.len == 0) return Error.InvalidTensorName;
    if (!std.unicode.utf8ValidateSlice(name)) return Error.InvalidTensorName;
}

fn validateTensorName(name: []const u8) !void {
    try validateName(name);
    if (std.mem.eql(u8, name, "__metadata__")) return Error.InvalidTensorName;
}

fn writeJsonString(out: *std.ArrayList(u8), allocator: Allocator, value: []const u8) !void {
    if (!std.unicode.utf8ValidateSlice(value)) return Error.InvalidHeader;
    try out.append(allocator, '"');
    for (value) |c| {
        switch (c) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0c => try out.appendSlice(allocator, "\\f"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0...0x07, 0x0b, 0x0e...0x1f => {
                try out.appendSlice(allocator, "\\u00");
                try out.append(allocator, hexDigit(c >> 4));
                try out.append(allocator, hexDigit(c & 0xf));
            },
            else => try out.append(allocator, c),
        }
    }
    try out.append(allocator, '"');
}

fn hexDigit(value: u8) u8 {
    return if (value < 10) '0' + value else 'a' + (value - 10);
}

fn appendInt(out: *std.ArrayList(u8), allocator: Allocator, value: usize) !void {
    var buf: [20]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{d}", .{value});
    try out.appendSlice(allocator, text);
}

fn deinitHeaderOnly(file: *File) void {
    for (file.tensors) |*tensor| {
        file.allocator.free(tensor.name);
        file.allocator.free(tensor.shape);
    }
    file.allocator.free(file.tensors);
    var meta_it = file.metadata.iterator();
    while (meta_it.next()) |entry| {
        file.allocator.free(entry.key_ptr.*);
        file.allocator.free(entry.value_ptr.*);
    }
    file.metadata.deinit();
    file.index.deinit();
    file.* = undefined;
}

fn dataLen(file: *const File) usize {
    return if (file.tensors.len == 0) 0 else file.tensors[file.tensors.len - 1].data_offsets[1];
}

test {
    _ = @import("safetensors_tests.zig");
}
