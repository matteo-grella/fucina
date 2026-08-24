//! GGUF reading: the madvise prefetch/release hints, `File` (heap-read,
//! mmap, and llama.cpp split-part loading; metadata + tensor directory),
//! and the raw metadata walker (`RawKvIter`) the writer's byte-verbatim
//! passthrough re-reads wire types from.

const std = @import("std");

const wire = @import("wire.zig");

const Allocator = std.mem.Allocator;
const Cursor = wire.Cursor;
const Error = wire.Error;
const TensorInfo = wire.TensorInfo;
const Value = wire.Value;
const Array = wire.Array;
const ggmlTypeFromInt = wire.ggmlTypeFromInt;
const tensorByteLen = wire.tensorByteLen;

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
                const gop = try index.getOrPut(tensors[at].name);
                if (gop.found_existing) {
                    std.log.warn("gguf: duplicate tensor name '{s}' across split parts — refusing the file", .{tensors[at].name});
                    return Error.DuplicateTensorName;
                }
                gop.value_ptr.* = at;
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
            if (metadata.contains(key)) return Error.DuplicateMetadataKey;
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
            const gop = try index.getOrPut(info.name);
            if (gop.found_existing) {
                std.log.warn("gguf: duplicate tensor name '{s}' — refusing the file", .{info.name});
                return Error.DuplicateTensorName;
            }
            gop.value_ptr.* = tensor_i;
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

/// Walks the raw key/value records of a serialized GGUF metadata section,
/// surfacing each value's exact wire type and encoded bytes. The parser's
/// widened `Value` map drops scalar wire widths (every int becomes i64), so
/// the writer's lossless metadata passthrough re-reads them from the file
/// bytes instead.
pub const RawKvIter = struct {
    cursor: Cursor,
    remaining: usize,

    const RawKv = struct {
        key: []const u8,
        value_type: u32,
        payload: []const u8,
    };

    pub fn init(bytes: []const u8) !RawKvIter {
        var cursor = Cursor{ .bytes = bytes };
        if (!std.mem.eql(u8, try cursor.readBytes(4), "GGUF")) return Error.InvalidMagic;
        const version = try cursor.readInt(u32);
        if (version != 2 and version != 3) return Error.UnsupportedVersion;
        _ = try cursor.readInt(u64); // tensor_count
        const metadata_count: usize = @intCast(try cursor.readInt(u64));
        return .{ .cursor = cursor, .remaining = metadata_count };
    }

    pub fn next(self: *RawKvIter) !?RawKv {
        if (self.remaining == 0) return null;
        self.remaining -= 1;
        const key = try self.cursor.readString();
        const value_type = try self.cursor.readInt(u32);
        const start = self.cursor.offset;
        try self.cursor.skipValue(value_type);
        return .{ .key = key, .value_type = value_type, .payload = self.cursor.bytes[start..self.cursor.offset] };
    }
};
