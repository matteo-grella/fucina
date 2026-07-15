const std = @import("std");

const dtype_mod = @import("dtype.zig");
const quant = @import("backend/quant.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;

/// Hint the OS to start paging in a mapped weight region we are about to read
/// in full (the copy/pack load paths). Lets readahead run ahead of the
/// sequential pack loop, so a cold load doesn't stall one page fault at a time
/// — the dominant cost when a model isn't already in the page cache. A no-op on
/// heap-read files (already resident) and whenever the advice call is
/// unsupported or fails. Borrowed (zero-copy) expert blocks deliberately do NOT
/// call this, so they stay lazily paged. `madvise` requires a page-aligned
/// range, so the start is rounded down and the length extended to cover it.
pub fn prefetch(data: []const u8) void {
    if (data.len == 0) return;
    const page = std.heap.pageSize();
    const start = @intFromPtr(data.ptr);
    const aligned = std.mem.alignBackward(usize, start, page);
    const len = (start - aligned) + data.len;
    const ptr: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(aligned);
    std.posix.madvise(ptr, len, std.posix.MADV.WILLNEED) catch {};
}

/// The counterpart hint for a mapped region we are DONE reading (the
/// tensor-at-a-time streaming paths): drop its pages from residency now
/// instead of waiting for memory pressure. Only valid for read-only
/// file-backed mappings (`File.loadMmap*`) — the pages are clean, so
/// `MADV.DONTNEED` merely releases them and a later touch refaults from the
/// file. Best-effort: a no-op whenever the advice call is unsupported or
/// fails. Page-aligned like `prefetch` (rounding may cover neighbouring
/// header bytes on the shared first/last page; they refault harmlessly).
pub fn release(data: []const u8) void {
    if (data.len == 0) return;
    const page = std.heap.pageSize();
    const start = @intFromPtr(data.ptr);
    const aligned = std.mem.alignBackward(usize, start, page);
    const len = (start - aligned) + data.len;
    const ptr: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(aligned);
    std.posix.madvise(ptr, len, std.posix.MADV.DONTNEED) catch {};
}

pub const Error = error{
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedValueType,
    UnsupportedGgmlType,
    InvalidTensorInfo,
    TensorNotFound,
    // Writer-side errors.
    KeyNotFound,
    DuplicateTensorName,
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
};

pub fn dtypeForGgmlType(value: GgmlType) ?DType {
    return switch (value) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .q1_0 => .q1_0,
        .q2_0 => .q2_0,
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q5_0 => .q5_0,
        .q5_1 => .q5_1,
        .q8_0 => .q8_0,
        .q8_1 => .q8_1,
        .q2_k => .q2_k,
        .q3_k => .q3_k,
        .q4_k => .q4_k,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
        .q8_k => .q8_k,
        .iq1_s => .iq1_s,
        .iq1_m => .iq1_m,
        .iq2_xxs => .iq2_xxs,
        .iq2_xs => .iq2_xs,
        .iq2_s => .iq2_s,
        .iq3_xxs => .iq3_xxs,
        .iq3_s => .iq3_s,
        .iq4_nl => .iq4_nl,
        .iq4_xs => .iq4_xs,
        .tq1_0 => .tq1_0,
        .tq2_0 => .tq2_0,
        .mxfp4 => .mxfp4,
        .nvfp4 => .nvfp4,
        else => null,
    };
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

pub const File = struct {
    allocator: Allocator,
    bytes: []u8,
    tensors: []TensorInfo,
    index: std.StringHashMap(usize),
    metadata: std.StringHashMap(Value),
    alignment: usize,
    data_offset: usize,
    /// When true, `bytes` is a read-only mmap of the file (freed via munmap)
    /// rather than a heap allocation. Lets large models load without a
    /// multi-GB heap copy — pages are file-backed and evictable under pressure.
    is_mmap: bool = false,
    /// llama.cpp split GGUFs (`-00001-of-0000N`): mappings of parts 2..N
    /// (part 1 is `bytes`) and each part's data-section offset, indexed by
    /// `TensorInfo.part`. Empty for single-file GGUFs.
    extra_bytes: [][]u8 = &.{},
    part_data_offsets: []u64 = &.{},

    pub fn isSplit(self: *const File) bool {
        return self.extra_bytes.len > 0;
    }

    /// The data-section offset of `part` within its own file on disk.
    pub fn partDataOffset(self: *const File, part: u16) u64 {
        if (self.part_data_offsets.len == 0) {
            std.debug.assert(part == 0);
            return self.data_offset;
        }
        return self.part_data_offsets[part];
    }

    pub fn load(allocator: Allocator, io: std.Io, path: []const u8) !File {
        var handle = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer handle.close(io);

        const stat = try handle.stat(io);
        if (stat.kind != .file) return error.IsDir;

        const len: usize = @intCast(stat.size);
        const bytes = try allocator.alloc(u8, len);
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

    /// Memory-map the file read-only and parse it in place. Weight loaders copy
    /// the blocks they need, so the mapping can be released by `deinit` after
    /// loading. Preferred for large (multi-GB) models — avoids a giant heap copy
    /// that would otherwise coexist with the materialized weights and OOM.
    pub fn loadMmap(allocator: Allocator, io: std.Io, path: []const u8) !File {
        var handle = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer handle.close(io); // POSIX keeps the mapping valid after the fd closes.

        const stat = try handle.stat(io);
        if (stat.kind != .file) return error.IsDir;

        const len: usize = @intCast(stat.size);
        if (len == 0) return Error.InvalidMagic;

        const mapped = try std.posix.mmap(null, len, .{ .READ = true }, .{ .TYPE = .PRIVATE }, handle.handle, 0);
        errdefer std.posix.munmap(mapped);

        var file = try parseCore(allocator, mapped);
        file.is_mmap = true;
        return file;
    }

    /// `loadMmap`, transparently following llama.cpp split GGUFs: when
    /// `path` is a `-00001-of-0000N` part, every part is mapped and parsed,
    /// and the result is one merged File — part 1's metadata (splits carry
    /// the full metadata there), the union of all parts' tensors (each
    /// tagged with its `part`), one index over all of them.
    pub fn loadMmapAuto(allocator: Allocator, io: std.Io, path: []const u8) !File {
        const parts = try splitPartPaths(allocator, path) orelse return loadMmap(allocator, io, path);
        defer {
            for (parts) |p| allocator.free(p);
            allocator.free(parts);
        }

        var files = try allocator.alloc(File, parts.len);
        defer allocator.free(files);
        var n_loaded: usize = 0;
        errdefer for (files[0..n_loaded]) |*f| f.deinit();
        for (parts) |part_path| {
            files[n_loaded] = try loadMmap(allocator, io, part_path);
            n_loaded += 1;
        }

        var total_tensors: usize = 0;
        for (files) |*f| total_tensors += f.tensors.len;
        const tensors = try allocator.alloc(TensorInfo, total_tensors);
        errdefer allocator.free(tensors);
        const part_data_offsets = try allocator.alloc(u64, files.len);
        errdefer allocator.free(part_data_offsets);
        const extra_bytes = try allocator.alloc([]u8, files.len - 1);
        errdefer allocator.free(extra_bytes);

        var index = std.StringHashMap(usize).init(allocator);
        errdefer index.deinit();
        var at: usize = 0;
        for (files, 0..) |*f, part_i| {
            part_data_offsets[part_i] = f.data_offset;
            for (f.tensors) |info| {
                tensors[at] = info;
                tensors[at].part = @intCast(part_i);
                try index.put(tensors[at].name, at);
                at += 1;
            }
        }

        // The merged File adopts part 1's mapping/metadata and the other
        // parts' mappings; the sub-Files' own bookkeeping is released.
        var merged = files[0];
        allocator.free(merged.tensors);
        merged.index.deinit();
        for (files[1..], 0..) |*f, i| {
            extra_bytes[i] = f.bytes;
            allocator.free(f.tensors);
            f.index.deinit();
            f.metadata.deinit();
        }
        merged.tensors = tensors;
        merged.index = index;
        merged.extra_bytes = extra_bytes;
        merged.part_data_offsets = part_data_offsets;
        return merged;
    }

    /// When `path` names the FIRST part of a llama.cpp split GGUF
    /// (`...-00001-of-0000N.gguf`), the caller-owned list of all part
    /// paths; null otherwise.
    pub fn splitPartPaths(allocator: Allocator, path: []const u8) !?[][]u8 {
        // "<base>-00001-of-0000N.gguf": 16-char split suffix + extension.
        const ext = ".gguf";
        if (!std.mem.endsWith(u8, path, ext)) return null;
        const stem = path[0 .. path.len - ext.len];
        const suffix_len = "-00001-of-00001".len;
        if (stem.len < suffix_len) return null;
        const suffix = stem[stem.len - suffix_len ..];
        if (!std.mem.startsWith(u8, suffix, "-") or !std.mem.eql(u8, suffix[6..10], "-of-")) return null;
        const part_no = std.fmt.parseInt(usize, suffix[1..6], 10) catch return null;
        const n_parts = std.fmt.parseInt(usize, suffix[10..], 10) catch return null;
        if (part_no != 1 or n_parts < 2) return null;

        const base = stem[0 .. stem.len - suffix_len];
        var parts = try allocator.alloc([]u8, n_parts);
        var built: usize = 0;
        errdefer {
            for (parts[0..built]) |p| allocator.free(p);
            allocator.free(parts);
        }
        for (0..n_parts) |i| {
            parts[i] = try std.fmt.allocPrint(allocator, "{s}-{d:0>5}-of-{d:0>5}{s}", .{ base, i + 1, n_parts, ext });
            built += 1;
        }
        return parts;
    }

    pub fn parseOwned(allocator: Allocator, bytes: []u8) !File {
        errdefer allocator.free(bytes);
        return parseCore(allocator, bytes);
    }

    fn parseCore(allocator: Allocator, bytes: []u8) !File {
        var cursor = Cursor{ .bytes = bytes };
        if (!std.mem.eql(u8, try cursor.readBytes(4), "GGUF")) return Error.InvalidMagic;

        const version = try cursor.readInt(u32);
        if (version != 2 and version != 3) return Error.UnsupportedVersion;

        const tensor_count_raw = try cursor.readInt(u64);
        const metadata_count_raw = try cursor.readInt(u64);
        if (tensor_count_raw > bytes.len or metadata_count_raw > bytes.len) return Error.InvalidTensorInfo;
        const tensor_count: usize = @intCast(tensor_count_raw);
        const metadata_count: usize = @intCast(metadata_count_raw);
        const metadata_capacity = std.math.cast(u32, metadata_count) orelse return Error.InvalidTensorInfo;

        var metadata = std.StringHashMap(Value).init(allocator);
        errdefer metadata.deinit();
        try metadata.ensureTotalCapacity(metadata_capacity);

        var alignment: usize = 32;
        for (0..metadata_count) |_| {
            const key = try cursor.readString();
            const value_type = try cursor.readInt(u32);
            if (std.mem.eql(u8, key, "general.alignment")) {
                // Validate directly from the wire value (before readValue's lossy
                // uint64->i64 narrowing and before the unchecked i64->usize cast),
                // so a hostile alignment can't reach UB at the cast or alignForward.
                alignment = try cursor.readAlignment(value_type);
                metadata.putAssumeCapacity(key, .{ .int = @intCast(alignment) });
                continue;
            }
            const value = try cursor.readValue(value_type);
            metadata.putAssumeCapacity(key, value);
        }

        const tensors = try allocator.alloc(TensorInfo, tensor_count);
        errdefer allocator.free(tensors);

        for (tensors) |*info| {
            info.name = try cursor.readString();
            info.n_dims = @intCast(try cursor.readInt(u32));
            if (info.n_dims == 0 or info.n_dims > info.dims.len) return Error.InvalidTensorInfo;
            info.dims = .{ 0, 0, 0, 0 };
            for (0..info.n_dims) |dim_i| {
                info.dims[dim_i] = @intCast(try cursor.readInt(u64));
            }
            info.ggml_type = ggmlTypeFromInt(try cursor.readInt(u32)) orelse return Error.UnsupportedGgmlType;
            info.offset = @intCast(try cursor.readInt(u64));
            info.data = &.{};
            // The infos live in alloc'd (undefined) memory filled field-by-field,
            // so the struct-literal default for `part` never applies here.
            info.part = 0;
        }

        const data_offset = std.mem.alignForward(usize, cursor.offset, alignment);
        var index = std.StringHashMap(usize).init(allocator);
        errdefer index.deinit();

        for (tensors, 0..) |*info, tensor_i| {
            const byte_len = try tensorByteLen(info.ggml_type, info.dims[0..info.n_dims]);
            const start = try std.math.add(usize, data_offset, info.offset);
            const end = try std.math.add(usize, start, byte_len);
            if (end > bytes.len) {
                // The header describes a tensor that runs past EOF — almost
                // always a truncated/incomplete download or a botched export,
                // not a malformed header. Name the first offender and the
                // shortfall so it self-diagnoses (otherwise this surfaces as a
                // bare InvalidTensorInfo with no hint).
                std.log.err("gguf: '{s}' ends at {d} but file is only {d} bytes — short by {d} ({d:.2} GB); the GGUF is truncated/incomplete (re-download or re-export)", .{
                    info.name, end, bytes.len, end - bytes.len, @as(f64, @floatFromInt(end - bytes.len)) / 1e9,
                });
                return Error.InvalidTensorInfo;
            }
            info.data = bytes[start..end];
            try index.put(info.name, tensor_i);
        }

        return .{
            .allocator = allocator,
            .bytes = bytes,
            .tensors = tensors,
            .index = index,
            .metadata = metadata,
            .alignment = alignment,
            .data_offset = data_offset,
        };
    }

    pub fn deinit(self: *File) void {
        self.metadata.deinit();
        self.index.deinit();
        self.allocator.free(self.tensors);
        for (self.extra_bytes) |part_bytes| std.posix.munmap(@alignCast(part_bytes));
        if (self.extra_bytes.len > 0) self.allocator.free(self.extra_bytes);
        if (self.part_data_offsets.len > 0) self.allocator.free(self.part_data_offsets);
        if (self.is_mmap) {
            std.posix.munmap(@alignCast(self.bytes));
        } else if (self.bytes.len > 0) {
            self.allocator.free(self.bytes);
        }
        self.* = undefined;
    }

    /// A file mapping whose ownership was transferred out of the `File` (see
    /// `takeMapping`). Holder must keep it alive as long as anything borrows
    /// tensor data from it, then `deinit` to munmap.
    pub const MappedRegion = struct {
        bytes: []const u8,

        pub fn deinit(self: *MappedRegion) void {
            std.posix.munmap(@alignCast(self.bytes));
            self.* = undefined;
        }
    };

    /// Transfer ownership of the underlying mmap to the caller (e.g. a model
    /// that borrows quantized weight blocks straight from the mapping instead
    /// of copying them). Returns null when the file was heap-read. Afterwards
    /// `File.deinit` no longer unmaps; metadata and `TensorInfo.data` slices
    /// stay valid for as long as the returned region lives.
    pub fn takeMapping(self: *File) ?MappedRegion {
        if (!self.is_mmap) return null;
        // Split files: tensors point into ALL part mappings, but a
        // MappedRegion can carry only one — borrowing across a split load
        // is not supported (stream the experts instead).
        if (self.isSplit()) return null;
        self.is_mmap = false;
        const bytes = self.bytes;
        // Leave an empty slice so deinit's heap branch is a no-op; previously
        // parsed metadata/TensorInfo slices keep pointing into the (still
        // mapped) region now owned by the caller.
        self.bytes = &.{};
        return .{ .bytes = bytes };
    }

    /// Look up a raw metadata value by key (e.g. "tokenizer.ggml.tokens").
    pub fn meta(self: *const File, key: []const u8) ?Value {
        return self.metadata.get(key);
    }

    pub fn getString(self: *const File, key: []const u8) ?[]const u8 {
        return switch (self.metadata.get(key) orelse return null) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn getInt(self: *const File, key: []const u8) ?i64 {
        return switch (self.metadata.get(key) orelse return null) {
            .int => |v| v,
            else => null,
        };
    }

    pub fn getFloat(self: *const File, key: []const u8) ?f64 {
        return switch (self.metadata.get(key) orelse return null) {
            .float => |v| v,
            .int => |v| @floatFromInt(v),
            else => null,
        };
    }

    pub fn getBool(self: *const File, key: []const u8) ?bool {
        return switch (self.metadata.get(key) orelse return null) {
            .boolean => |b| b,
            .int => |v| v != 0,
            else => null,
        };
    }

    pub fn getArray(self: *const File, key: []const u8) ?Array {
        return switch (self.metadata.get(key) orelse return null) {
            .array => |a| a,
            else => null,
        };
    }

    pub fn get(self: *const File, name: []const u8) !*const TensorInfo {
        const tensor_i = self.index.get(name) orelse return Error.TensorNotFound;
        return &self.tensors[tensor_i];
    }

    pub fn maybeGet(self: *const File, name: []const u8) ?*const TensorInfo {
        const tensor_i = self.index.get(name) orelse return null;
        return &self.tensors[tensor_i];
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

    return switch (ggml_type) {
        .f32, .i32 => try std.math.mul(usize, logical_count, 4),
        .f16, .bf16, .i16 => try std.math.mul(usize, logical_count, 2),
        .f64, .i64 => try std.math.mul(usize, logical_count, 8),
        .i8 => logical_count,
        .q1_0 => quantizedByteLen(.q1_0, dims, logical_count),
        .q2_0 => quantizedByteLen(.q2_0, dims, logical_count),
        .q4_0 => quantizedByteLen(.q4_0, dims, logical_count),
        .q4_1 => quantizedByteLen(.q4_1, dims, logical_count),
        .q5_0 => quantizedByteLen(.q5_0, dims, logical_count),
        .q5_1 => quantizedByteLen(.q5_1, dims, logical_count),
        .q8_0 => quantizedByteLen(.q8_0, dims, logical_count),
        .q8_1 => quantizedByteLen(.q8_1, dims, logical_count),
        .q2_k => quantizedByteLen(.q2_k, dims, logical_count),
        .q3_k => quantizedByteLen(.q3_k, dims, logical_count),
        .q4_k => quantizedByteLen(.q4_k, dims, logical_count),
        .q5_k => quantizedByteLen(.q5_k, dims, logical_count),
        .q6_k => quantizedByteLen(.q6_k, dims, logical_count),
        .q8_k => quantizedByteLen(.q8_k, dims, logical_count),
        .iq1_s => quantizedByteLen(.iq1_s, dims, logical_count),
        .iq1_m => quantizedByteLen(.iq1_m, dims, logical_count),
        .iq2_xxs => quantizedByteLen(.iq2_xxs, dims, logical_count),
        .iq2_xs => quantizedByteLen(.iq2_xs, dims, logical_count),
        .iq2_s => quantizedByteLen(.iq2_s, dims, logical_count),
        .iq3_xxs => quantizedByteLen(.iq3_xxs, dims, logical_count),
        .iq3_s => quantizedByteLen(.iq3_s, dims, logical_count),
        .iq4_nl => quantizedByteLen(.iq4_nl, dims, logical_count),
        .iq4_xs => quantizedByteLen(.iq4_xs, dims, logical_count),
        .tq1_0 => quantizedByteLen(.tq1_0, dims, logical_count),
        .tq2_0 => quantizedByteLen(.tq2_0, dims, logical_count),
        .mxfp4 => quantizedByteLen(.mxfp4, dims, logical_count),
        .nvfp4 => quantizedByteLen(.nvfp4, dims, logical_count),
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

fn ggmlTypeFromInt(value: u32) ?GgmlType {
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
        else => null,
    };
}

const Cursor = struct {
    bytes: []const u8,
    offset: usize = 0,

    fn readBytes(self: *Cursor, len: usize) ![]const u8 {
        const end = try std.math.add(usize, self.offset, len);
        if (end > self.bytes.len) return error.EndOfStream;
        const out = self.bytes[self.offset..end];
        self.offset = end;
        return out;
    }

    fn readInt(self: *Cursor, comptime Int: type) !Int {
        return std.mem.readInt(Int, (try self.readBytes(@sizeOf(Int)))[0..@sizeOf(Int)], .little);
    }

    fn readString(self: *Cursor) ![]const u8 {
        const len: usize = @intCast(try self.readInt(u64));
        return self.readBytes(len);
    }

    fn readFloat(self: *Cursor, comptime Float: type) !Float {
        const Int = std.meta.Int(.unsigned, @bitSizeOf(Float));
        return @bitCast(try self.readInt(Int));
    }

    fn readValue(self: *Cursor, value_type: u32) !Value {
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
    fn readAlignment(self: *Cursor, value_type: u32) !usize {
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

    fn skipValue(self: *Cursor, value_type: u32) !void {
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

// ---------------------------------------------------------------------------
// GGUF writer.
// ---------------------------------------------------------------------------

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

fn metaTypeForScalar(comptime T: type) MetaType {
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

fn writeScalarLittle(comptime T: type, buf: []u8, value: T) void {
    switch (@typeInfo(T)) {
        .int => std.mem.writeInt(T, buf[0..@sizeOf(T)], value, .little),
        .float => {
            const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
            std.mem.writeInt(Bits, buf[0..@sizeOf(T)], @bitCast(value), .little);
        },
        else => @compileError("unsupported GGUF metadata scalar type " ++ @typeName(T)),
    }
}

/// Walks the raw key/value records of a serialized GGUF metadata section,
/// surfacing each value's exact wire type and encoded bytes. The parser's
/// widened `Value` map drops scalar wire widths (every int becomes i64), so
/// the writer's lossless metadata passthrough re-reads them from the file
/// bytes instead.
const RawKvIter = struct {
    cursor: Cursor,
    remaining: usize,

    const RawKv = struct {
        key: []const u8,
        value_type: u32,
        payload: []const u8,
    };

    fn init(bytes: []const u8) !RawKvIter {
        var cursor = Cursor{ .bytes = bytes };
        if (!std.mem.eql(u8, try cursor.readBytes(4), "GGUF")) return Error.InvalidMagic;
        const version = try cursor.readInt(u32);
        if (version != 2 and version != 3) return Error.UnsupportedVersion;
        _ = try cursor.readInt(u64); // tensor_count
        const metadata_count: usize = @intCast(try cursor.readInt(u64));
        return .{ .cursor = cursor, .remaining = metadata_count };
    }

    fn next(self: *RawKvIter) !?RawKv {
        if (self.remaining == 0) return null;
        self.remaining -= 1;
        const key = try self.cursor.readString();
        const value_type = try self.cursor.readInt(u32);
        const start = self.cursor.offset;
        try self.cursor.skipValue(value_type);
        return .{ .key = key, .value_type = value_type, .payload = self.cursor.bytes[start..self.cursor.offset] };
    }
};

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
        /// declared set is `Error.TensorDataMissing`.
        pub fn writeTensorData(self: *DataStreamer, data: []const u8) !void {
            const tensors = self.writer.tensors.items;
            if (self.next_index >= tensors.len) return Error.TensorDataMissing;
            const t = &tensors[self.next_index];
            if (data.len != t.byte_len) return Error.InvalidTensorInfo;
            try self.out.writeAll(data);
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

/// Encode `src` f32 values as `ggml_type` wire bytes into `dst`, whose length
/// must equal `tensorByteLen(ggml_type, &.{src.len})`. Little-endian targets
/// only for block formats (same assumption as the parser's zero-copy blocks).
///
/// This is the writer-side quantize seam: scalar formats cast element-wise,
/// block formats dispatch to the byte-exact ggml-parity encoders in
/// backend/quant.zig (quantizeRowForDType) after rejecting non-finite input
/// with `error.NonFiniteValue`. Formats without a from-float encoder return
/// `error.EncoderUnavailable`.
pub fn encodeF32(ggml_type: GgmlType, src: []const f32, dst: []u8) !void {
    if (dst.len != try tensorByteLen(ggml_type, &.{src.len})) return Error.InvalidTensorInfo;
    switch (ggml_type) {
        // The block encoders assume finite input (their finite-input contract
        // is otherwise enforced only by Debug asserts); src may come straight
        // from untrusted file bytes, so validate here in release builds too —
        // the same seam llama.cpp guards with ggml_validate_row_data
        // (refs/llama.cpp/src/llama-quant.cpp).
        .q2_0, .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q4_k, .q5_k, .q6_k, .tq2_0 => {
            if (!allFinite(src)) return Error.NonFiniteValue;
        },
        // Scalar casts stay unguarded: f16 inf-overflow on out-of-range
        // values matches ggml's scalar conversion behavior.
        else => {},
    }
    switch (ggml_type) {
        .f32 => for (src, 0..) |value, i| {
            std.mem.writeInt(u32, dst[i * 4 ..][0..4], @bitCast(value), .little);
        },
        .f16 => for (src, 0..) |value, i| {
            std.mem.writeInt(u16, dst[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(value))), .little);
        },
        .bf16 => for (src, 0..) |value, i| {
            std.mem.writeInt(u16, dst[i * 2 ..][0..2], dtype_mod.f32ToBf16(value), .little);
        },
        .q2_0 => try encodeBlocks(.q2_0, src, dst),
        .q4_0 => try encodeBlocks(.q4_0, src, dst),
        .q4_1 => try encodeBlocks(.q4_1, src, dst),
        .q5_0 => try encodeBlocks(.q5_0, src, dst),
        .q5_1 => try encodeBlocks(.q5_1, src, dst),
        .q8_0 => try encodeBlocks(.q8_0, src, dst),
        .q4_k => try encodeBlocks(.q4_k, src, dst),
        .q5_k => try encodeBlocks(.q5_k, src, dst),
        .q6_k => try encodeBlocks(.q6_k, src, dst),
        .tq2_0 => try encodeBlocks(.tq2_0, src, dst),
        .q2_k, .q3_k => return Error.EncoderUnavailable,
        else => return Error.EncoderUnavailable,
    }
}

/// One vectorizable pass: finite iff |x| < inf (NaN compares false).
fn allFinite(values: []const f32) bool {
    const lanes = 8;
    const V = @Vector(lanes, f32);
    const inf: V = @splat(std.math.inf(f32));
    var i: usize = 0;
    while (i + lanes <= values.len) : (i += lanes) {
        const v: V = values[i..][0..lanes].*;
        if (!@reduce(.And, @abs(v) < inf)) return false;
    }
    while (i < values.len) : (i += 1) {
        if (!std.math.isFinite(values[i])) return false;
    }
    return true;
}

fn encodeBlocks(comptime dt: dtype_mod.DType, src: []const f32, dst: []u8) !void {
    const Block = dtype_mod.Storage(dt);
    if (@intFromPtr(dst.ptr) % @alignOf(Block) != 0) return Error.InvalidTensorInfo;
    const blocks: []Block = @alignCast(std.mem.bytesAsSlice(Block, dst));
    quant.quantizeRowForDType(dt, blocks, src) catch return Error.InvalidTensorInfo;
}

/// Decode `ggml_type` wire bytes into f32 values — the reader-side mirror of
/// `encodeF32` (same supported formats, same length contract: `src.len` must
/// equal `tensorByteLen(ggml_type, &.{dst.len})`). Scalar formats widen
/// element-wise, block formats dispatch to the ggml-parity decoders in
/// backend/quant.zig (dequantizeRowForDType). Formats without a to-float
/// decoder return `error.DecoderUnavailable`.
pub fn decodeF32(ggml_type: GgmlType, src: []const u8, dst: []f32) !void {
    if (src.len != try tensorByteLen(ggml_type, &.{dst.len})) return Error.InvalidTensorInfo;
    switch (ggml_type) {
        .f32 => for (dst, 0..) |*value, i| {
            value.* = @bitCast(std.mem.readInt(u32, src[i * 4 ..][0..4], .little));
        },
        .f16 => for (dst, 0..) |*value, i| {
            value.* = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, src[i * 2 ..][0..2], .little))));
        },
        .bf16 => for (dst, 0..) |*value, i| {
            value.* = @bitCast(@as(u32, std.mem.readInt(u16, src[i * 2 ..][0..2], .little)) << 16);
        },
        .q2_0 => try decodeBlocks(.q2_0, src, dst),
        .q4_0 => try decodeBlocks(.q4_0, src, dst),
        .q4_1 => try decodeBlocks(.q4_1, src, dst),
        .q5_0 => try decodeBlocks(.q5_0, src, dst),
        .q5_1 => try decodeBlocks(.q5_1, src, dst),
        .q8_0 => try decodeBlocks(.q8_0, src, dst),
        .q4_k => try decodeBlocks(.q4_k, src, dst),
        .q5_k => try decodeBlocks(.q5_k, src, dst),
        .q6_k => try decodeBlocks(.q6_k, src, dst),
        .tq2_0 => try decodeBlocks(.tq2_0, src, dst),
        else => return Error.DecoderUnavailable,
    }
}

fn decodeBlocks(comptime dt: dtype_mod.DType, src: []const u8, dst: []f32) !void {
    const Block = dtype_mod.Storage(dt);
    if (@intFromPtr(src.ptr) % @alignOf(Block) != 0) return Error.InvalidTensorInfo;
    const blocks: []const Block = @alignCast(std.mem.bytesAsSlice(Block, src));
    quant.dequantizeRowForDType(dt, dst, blocks) catch return Error.InvalidTensorInfo;
}

test {
    _ = @import("gguf_tests.zig");
}
