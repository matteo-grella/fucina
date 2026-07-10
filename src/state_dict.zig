//! Neutral named tensor state-dict serialization.
//!
//! This is the framework-level tensor checkpoint layer: modules, LoRA
//! adapters, and optimizers can all speak in named tensor entries without
//! depending on each other. The wire format is Hugging Face safetensors:
//! a neutral tensor-only format with mmap-friendly contiguous data. GGUF
//! remains a separate LLM interop/export codec.
const std = @import("std");
const tensor_mod = @import("tensor.zig");
const dtype_mod = @import("dtype.zig");
const safetensors = @import("safetensors.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;
const Shape = tensor_mod.Shape;

pub const Error = error{
    NonContiguousParam,
    CheckpointMagicMismatch,
    CheckpointShapeMismatch,
    CheckpointInvalidName,
    CheckpointDuplicateName,
    CheckpointUnknownName,
    CheckpointMissingEntry,
    CheckpointDtypeMismatch,
    CheckpointUnsupportedDtype,
    CheckpointTooManyEntries,
};

/// One named state-dict entry: a borrowed name plus a type-erased, borrowed
/// view of one contiguous f32/f16/bf16 facade tensor (variable or constant).
/// Build with `of`; the tensor's storage and the name are BORROWED and must
/// outlive the entry.
pub const NamedTensor = struct {
    name: []const u8,
    dtype: DType,
    shape: Shape,
    bytes: []const u8,

    /// `t` is a pointer to a facade tensor. f32/f16/bf16 are supported;
    /// other dtypes are rejected at compile time. Non-contiguous tensors are
    /// rejected.
    pub fn of(name: []const u8, t: anytype) !NamedTensor {
        const dt = comptime stateDictDtype(@TypeOf(t));
        if (!t.value.isContiguous()) return Error.NonContiguousParam;
        return .{ .name = name, .dtype = dt, .shape = t.value.shape, .bytes = std.mem.sliceAsBytes(t.value.dataConst()) };
    }
};

/// Mutable counterpart of `NamedTensor` for `loadStateDict` destinations.
pub const NamedTensorMut = struct {
    name: []const u8,
    dtype: DType,
    shape: Shape,
    bytes: []u8,

    pub fn of(name: []const u8, t: anytype) !NamedTensorMut {
        const dt = comptime stateDictDtype(@TypeOf(t));
        comptime if (@typeInfo(@TypeOf(t)).pointer.is_const) {
            @compileError("NamedTensorMut.of requires a mutable facade tensor pointer");
        };
        if (!t.value.isContiguous()) return Error.NonContiguousParam;
        return .{ .name = name, .dtype = dt, .shape = t.value.shape, .bytes = std.mem.sliceAsBytes(t.value.data()) };
    }
};

/// A checkpoint field-rename rule: a stream entry named `old` is matched to the
/// destination registered as `new`. Lets a renamed parameter path load an older
/// checkpoint without re-saving — see the schema-stability contract in
/// `param_registry.zig`'s header.
pub const Alias = struct {
    old: []const u8,
    new: []const u8,
};

pub const LoadOptions = struct {
    strict: bool = true,
    /// Optional field-rename map applied to each STREAM name before matching it
    /// to a destination. First match wins; an empty map (the default) is a no-op.
    aliases: []const Alias = &.{},
};

/// Remap a stream name through the alias map (first `old` match → its `new`),
/// or return it unchanged when no rule applies.
fn remapName(name: []const u8, aliases: []const Alias) []const u8 {
    for (aliases) |alias| {
        if (std.mem.eql(u8, name, alias.old)) return alias.new;
    }
    return name;
}

/// Serialize named tensors as safetensors. Names must be non-empty, NUL-free
/// UTF-8 and unique; f32/f16/bf16 contiguous tensors are written as raw bytes.
/// Everything is validated before the first byte is written.
pub fn saveStateDict(allocator: Allocator, writer: *std.Io.Writer, entries: []const NamedTensor) !void {
    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();
    try seen.ensureTotalCapacity(@intCast(entries.len));
    const tensors = try allocator.alloc(safetensors.Tensor, entries.len);
    defer allocator.free(tensors);
    for (entries, 0..) |*entry, i| {
        try validateName(entry.name);
        const dims = entry.shape.slice();
        if (entry.bytes.len != try stateDictByteLen(entry.dtype, dims)) {
            return Error.CheckpointShapeMismatch;
        }
        const gop = seen.getOrPutAssumeCapacity(entry.name);
        if (gop.found_existing) return Error.CheckpointDuplicateName;
        tensors[i] = .{
            .name = entry.name,
            .dtype = safetensors.dtypeFromFucina(entry.dtype) catch return Error.CheckpointUnsupportedDtype,
            .shape = dims,
            .data = entry.bytes,
        };
    }
    try safetensors.serialize(allocator, writer, tensors, null);
}

/// Load a state dict saved by `saveStateDict` into `entries`, matching stream
/// entries by NAME (any order). Shape and dtype must match the destination
/// exactly. Strict (the default) demands a one-to-one match.
pub fn loadStateDict(allocator: Allocator, reader: *std.Io.Reader, entries: []const NamedTensorMut, options: LoadOptions) !void {
    var index = std.StringHashMap(usize).init(allocator);
    defer index.deinit();
    try index.ensureTotalCapacity(@intCast(entries.len));
    for (entries, 0..) |*entry, i| {
        const gop = index.getOrPutAssumeCapacity(entry.name);
        if (gop.found_existing) return Error.CheckpointDuplicateName;
        gop.value_ptr.* = i;
    }
    // Stage the load: validate EVERY stream entry against its destination before
    // mutating any destination, so a mismatch leaves all destinations
    // byte-unchanged (transactional). `srcs[i]` is the validated source byte slice
    // for entries[i] (null = absent from the stream); it doubles as the duplicate
    // tracker and the strict-missing check.
    const srcs = try allocator.alloc(?[]const u8, entries.len);
    defer allocator.free(srcs);
    @memset(srcs, null);

    var file = try safetensors.readPrefix(allocator, reader);
    defer file.deinit();

    // Pass 1 — validate only; no live buffer is written.
    for (file.tensors) |*tensor| {
        const lookup_name = remapName(tensor.name, options.aliases);
        const entry_index = index.get(lookup_name) orelse {
            if (options.strict) return Error.CheckpointUnknownName;
            continue;
        };
        if (srcs[entry_index] != null) return Error.CheckpointDuplicateName;
        const entry = &entries[entry_index];
        const stored_dtype = safetensors.dtypeToFucina(tensor.dtype) catch return Error.CheckpointDtypeMismatch;
        if (stored_dtype != entry.dtype) return Error.CheckpointDtypeMismatch;
        const shape = entry.shape.slice();
        if (tensor.shape.len != shape.len) return Error.CheckpointShapeMismatch;
        for (tensor.shape, shape) |stored, dim| {
            if (stored != dim) return Error.CheckpointShapeMismatch;
        }
        if (entry.bytes.len != tensor.data.len) return Error.CheckpointShapeMismatch;
        srcs[entry_index] = tensor.data;
    }
    if (options.strict) {
        for (srcs) |s| if (s == null) return Error.CheckpointMissingEntry;
    }

    // Pass 2 — commit. Every validated entry is copied; no failure points remain,
    // so destinations are mutated all-or-nothing.
    for (entries, srcs) |*entry, maybe_src| {
        if (maybe_src) |src| @memcpy(entry.bytes, src);
    }
}

fn stateDictDtype(comptime P: type) DType {
    const info = @typeInfo(P);
    if (info != .pointer) @compileError("NamedTensor.of expects a pointer to a facade tensor");
    const dt = info.pointer.child.dtype;
    return switch (dt) {
        .f32, .f16, .bf16 => dt,
        else => @compileError("state dicts support f32, f16, and bf16 tensors only; got " ++ @tagName(dt)),
    };
}

fn stateDictByteLen(dtype: DType, dims: []const usize) !usize {
    const elems = try tensor_mod.elementCount(dims);
    const elem_size: usize = switch (dtype) {
        .f32 => 4,
        .f16, .bf16 => 2,
        else => return Error.CheckpointUnsupportedDtype,
    };
    return std.math.mul(usize, elems, elem_size) catch return Error.CheckpointShapeMismatch;
}

fn validateName(name: []const u8) !void {
    if (name.len == 0) return Error.CheckpointInvalidName;
    if (std.mem.eql(u8, name, "__metadata__")) return Error.CheckpointInvalidName;
    if (std.mem.indexOfScalar(u8, name, 0) != null) return Error.CheckpointInvalidName;
    if (!std.unicode.utf8ValidateSlice(name)) return Error.CheckpointInvalidName;
}

test {
    _ = @import("state_dict_tests.zig");
}
