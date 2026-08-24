//! GGUF wire vocabulary: the error set, `GgmlType` and its `DType` mapping,
//! the parsed `Value`/`Array`/`TensorInfo` types, byte-length rules
//! (`tensorByteLen`), the metadata value-type codes (`MetaType`) with their
//! scalar encoders, and the little-endian `Cursor`. Leaf: reader, writer,
//! and codec all import this.

const std = @import("std");

const dtype_mod = @import("../dtype.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;

pub const Error = error{
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedValueType,
    UnsupportedGgmlType,
    InvalidTensorInfo,
    TensorNotFound,
    /// Two tensors (writer or parsed container) or two metadata entries
    /// (parsed container) share one name: refused, never last-wins — a
    /// silently-resolved duplicate lets an auditing tool and this reader
    /// see different payloads under the same name.
    DuplicateTensorName,
    DuplicateMetadataKey,
    // Writer-side errors.
    KeyNotFound,
    InvalidAlignment,
    TensorDataMissing,
    MetadataValueOutOfRange,
    EncoderUnavailable,
    DecoderUnavailable,
    NonFiniteValue,
};

pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    nvfp4 = 40,
    q1_0 = 41,
    q2_0 = 42,
    /// Fucina-custom: tie-fitted K=2 PTQTP planes pre-folded into the
    /// one-pass 4-bit pack (`BlockTQ2_0Foldedx4`, backend/quant/types.zig)
    /// — 520 bytes per 4 columns x 256 elements (4.0625 bpw). The expert
    /// streaming format: one contiguous read per expert projection, and
    /// slab bytes == file bytes, so the L2 tier stripes it like any plain
    /// quant. Requires dims[0] % 256 == 0 and dims[1] % 4 == 0.
    tq2_0_fx4 = 43,
};

pub fn dtypeForGgmlType(value: GgmlType) ?DType {
    switch (value) {
        .f32 => return .f32,
        .f16 => return .f16,
        .bf16 => return .bf16,
        else => {},
    }
    // Quant formats: GgmlType and DType share tag names, one registry
    // row per format (dtype.block_formats is the single source). A
    // registry format with no GgmlType tag is a compile error here —
    // every block format carries its wire id.
    @setEvalBranchQuota(10_000);
    inline for (dtype_mod.block_formats) |row| {
        if (value == @field(GgmlType, @tagName(row.dtype))) return row.dtype;
    }
    return null;
}

/// A parsed GGUF metadata value. Scalars are widened (all integer types to
/// i64, all floats to f64); strings and arrays are zero-copy slices into the
/// loaded file bytes, so they stay valid only while the owning `File` lives.
pub const Value = union(enum) {
    int: i64,
    float: f64,
    boolean: bool,
    string: []const u8,
    array: Array,
};

/// A GGUF metadata array: its element type and the raw bytes spanning all
/// elements (a slice into the file). Use `stringSlices` for string arrays.
pub const Array = struct {
    item_type: u32,
    len: usize,
    data: []const u8,

    /// Decode a string array into owned slices (each pointing into the file
    /// bytes; only the outer slice array is allocated). Caller frees the result.
    pub fn stringSlices(self: Array, allocator: Allocator) ![][]const u8 {
        if (self.item_type != 8) return Error.UnsupportedValueType;
        const out = try allocator.alloc([]const u8, self.len);
        errdefer allocator.free(out);
        var cursor = Cursor{ .bytes = self.data };
        for (out) |*slot| slot.* = try cursor.readString();
        return out;
    }
};

pub const TensorInfo = struct {
    name: []const u8,
    dims: [4]usize,
    n_dims: usize,
    ggml_type: GgmlType,
    offset: usize,
    data: []const u8,
    /// Which split part holds this tensor (0 for single-file GGUFs). The
    /// absolute on-disk position is `File.partDataOffset(part) + offset`.
    part: u16 = 0,

    pub fn dim(self: TensorInfo, index: usize) !usize {
        if (index >= self.n_dims) return Error.InvalidTensorInfo;
        return self.dims[index];
    }

    pub fn logicalMatrixShape(self: TensorInfo) ![2]usize {
        if (self.n_dims != 2) return Error.InvalidTensorInfo;
        return .{ self.dims[1], self.dims[0] };
    }
};

pub fn tensorByteLen(ggml_type: GgmlType, dims: []const usize) !usize {
    if (dims.len == 0) return Error.InvalidTensorInfo;

    // A 0-length dimension is a legitimate empty ggml tensor (0 elements => 0
    // bytes); ggml exports them (e.g. face-detect.cpp's SCRFD `det.392`, a rank-1
    // [0] initializer). Fold it through instead of rejecting: logical_count
    // becomes 0 and the byte length below is 0. The EOF check in `parseCore`
    // still catches genuinely truncated headers.
    var logical_count: usize = 1;
    for (dims) |dim| {
        logical_count = try std.math.mul(usize, logical_count, dim);
    }

    @setEvalBranchQuota(10_000);
    inline for (dtype_mod.block_formats) |row| {
        if (ggml_type == @field(GgmlType, @tagName(row.dtype)))
            return quantizedByteLen(row.dtype, dims, logical_count);
    }
    return switch (ggml_type) {
        .f32, .i32 => try std.math.mul(usize, logical_count, 4),
        .f16, .bf16, .i16 => try std.math.mul(usize, logical_count, 2),
        .f64, .i64 => try std.math.mul(usize, logical_count, 8),
        .i8 => logical_count,
        // 2-D block: 4 columns (dims[1]) x 256 elements (dims[0]) per
        // 520-byte pack — the only fucina type whose block spans dims[1].
        .tq2_0_fx4 => blk: {
            if (dims.len < 2 or dims[0] % 256 != 0 or dims[1] % 4 != 0) return Error.InvalidTensorInfo;
            break :blk try std.math.mul(usize, logical_count / 1024, 520);
        },
        // Every remaining tag is a registry block format, returned by the
        // `block_formats` loop above.
        else => unreachable,
    };
}

fn quantizedByteLen(comptime dtype: DType, dims: []const usize, logical_count: usize) !usize {
    // Blocks must not straddle the innermost dim: like ggml_row_size, require
    // ne[0] % block_size == 0, not just the total element count (ggml
    // guarantees ne[0] % block == 0 for valid files, so this only rejects
    // malformed shapes such as q8_0 [16, 4]).
    if (dims[0] % dtype_mod.blockSize(dtype) != 0) return Error.InvalidTensorInfo;
    return try std.math.mul(usize, logical_count / dtype_mod.blockSize(dtype), dtype_mod.blockByteSize(dtype));
}

pub fn ggmlTypeFromInt(value: u32) ?GgmlType {
    return switch (value) {
        0 => .f32,
        1 => .f16,
        2 => .q4_0,
        3 => .q4_1,
        6 => .q5_0,
        7 => .q5_1,
        8 => .q8_0,
        9 => .q8_1,
        10 => .q2_k,
        11 => .q3_k,
        12 => .q4_k,
        13 => .q5_k,
        14 => .q6_k,
        15 => .q8_k,
        16 => .iq2_xxs,
        17 => .iq2_xs,
        18 => .iq3_xxs,
        19 => .iq1_s,
        20 => .iq4_nl,
        21 => .iq3_s,
        22 => .iq2_s,
        23 => .iq4_xs,
        24 => .i8,
        25 => .i16,
        26 => .i32,
        27 => .i64,
        28 => .f64,
        29 => .iq1_m,
        30 => .bf16,
        34 => .tq1_0,
        35 => .tq2_0,
        39 => .mxfp4,
        40 => .nvfp4,
        41 => .q1_0,
        42 => .q2_0,
        43 => .tq2_0_fx4,
        else => null,
    };
}

pub const Cursor = struct {
    bytes: []const u8,
    offset: usize = 0,

    pub fn readBytes(self: *Cursor, len: usize) ![]const u8 {
        const end = try std.math.add(usize, self.offset, len);
        if (end > self.bytes.len) return error.EndOfStream;
        const out = self.bytes[self.offset..end];
        self.offset = end;
        return out;
    }

    pub fn readInt(self: *Cursor, comptime Int: type) !Int {
        return std.mem.readInt(Int, (try self.readBytes(@sizeOf(Int)))[0..@sizeOf(Int)], .little);
    }

    pub fn readString(self: *Cursor) ![]const u8 {
        const len: usize = @intCast(try self.readInt(u64));
        return self.readBytes(len);
    }

    fn readFloat(self: *Cursor, comptime Float: type) !Float {
        const Int = std.meta.Int(.unsigned, @bitSizeOf(Float));
        return @bitCast(try self.readInt(Int));
    }

    pub fn readValue(self: *Cursor, value_type: u32) !Value {
        return switch (value_type) {
            0 => .{ .int = @intCast(try self.readInt(u8)) },
            1 => .{ .int = @intCast(try self.readInt(i8)) },
            2 => .{ .int = @intCast(try self.readInt(u16)) },
            3 => .{ .int = @intCast(try self.readInt(i16)) },
            4 => .{ .int = @intCast(try self.readInt(u32)) },
            5 => .{ .int = @intCast(try self.readInt(i32)) },
            6 => .{ .float = try self.readFloat(f32) },
            7 => .{ .boolean = (try self.readInt(u8)) != 0 },
            8 => .{ .string = try self.readString() },
            9 => .{ .array = try self.readArray() },
            // A wire uint64 >= 2^63 does not fit the i64 `Value.int`; a checked
            // cast returns an error instead of illegal behaviour under ReleaseFast.
            10 => .{ .int = std.math.cast(i64, try self.readInt(u64)) orelse return Error.MetadataValueOutOfRange },
            11 => .{ .int = try self.readInt(i64) },
            12 => .{ .float = try self.readFloat(f64) },
            else => Error.UnsupportedValueType,
        };
    }

    /// Read + validate `general.alignment` directly from its wire value, before
    /// the lossy uint64->i64 narrowing in `readValue`. A non-int, negative, zero,
    /// non-power-of-two, or out-of-range (`>= 2^63` or `> 2^20`) alignment returns
    /// `Error.InvalidAlignment` rather than illegal behaviour at a cast or at
    /// `std.mem.alignForward`. Returns a validated power-of-two `<= 2^20`.
    pub fn readAlignment(self: *Cursor, value_type: u32) !usize {
        const raw: u64 = switch (value_type) {
            0 => try self.readInt(u8),
            2 => try self.readInt(u16),
            4 => try self.readInt(u32),
            10 => try self.readInt(u64),
            1, 3, 5, 11 => signed: {
                const sv: i64 = switch (value_type) {
                    1 => try self.readInt(i8),
                    3 => try self.readInt(i16),
                    5 => try self.readInt(i32),
                    else => try self.readInt(i64),
                };
                if (sv <= 0) return Error.InvalidAlignment;
                break :signed @intCast(sv);
            },
            else => return Error.InvalidAlignment, // non-int type (string/array/float/bool)
        };
        if (raw == 0 or (raw & (raw - 1)) != 0 or raw > (1 << 20)) return Error.InvalidAlignment;
        return @intCast(raw);
    }

    fn readArray(self: *Cursor) !Array {
        const item_type = try self.readInt(u32);
        if (item_type == 9) return Error.UnsupportedValueType; // nested arrays unsupported
        const len: usize = @intCast(try self.readInt(u64));
        const start = self.offset;
        for (0..len) |_| try self.skipValue(item_type);
        return .{ .item_type = item_type, .len = len, .data = self.bytes[start..self.offset] };
    }

    pub fn skipValue(self: *Cursor, value_type: u32) !void {
        switch (value_type) {
            0, 1, 7 => _ = try self.readBytes(1),
            2, 3 => _ = try self.readBytes(2),
            4, 5, 6 => _ = try self.readBytes(4),
            8 => _ = try self.readString(),
            9 => {
                const item_type = try self.readInt(u32);
                if (item_type == 9) return Error.UnsupportedValueType;
                const len: usize = @intCast(try self.readInt(u64));
                for (0..len) |_| try self.skipValue(item_type);
            },
            10, 11, 12 => _ = try self.readBytes(8),
            else => return Error.UnsupportedValueType,
        }
    }
};

/// GGUF metadata value-type wire codes (ggml's gguf.h `GGUF_TYPE_*`); the
/// same codes the parser's `Cursor.readValue` switches on.
pub const MetaType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

pub fn metaTypeForScalar(comptime T: type) MetaType {
    return switch (T) {
        u8 => .uint8,
        i8 => .int8,
        u16 => .uint16,
        i16 => .int16,
        u32 => .uint32,
        i32 => .int32,
        u64 => .uint64,
        i64 => .int64,
        f32 => .float32,
        f64 => .float64,
        else => @compileError("unsupported GGUF metadata scalar type " ++ @typeName(T)),
    };
}

pub fn writeScalarLittle(comptime T: type, buf: []u8, value: T) void {
    switch (@typeInfo(T)) {
        .int => std.mem.writeInt(T, buf[0..@sizeOf(T)], value, .little),
        .float => {
            const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
            std.mem.writeInt(Bits, buf[0..@sizeOf(T)], @bitCast(value), .little);
        },
        else => @compileError("unsupported GGUF metadata scalar type " ++ @typeName(T)),
    }
}
