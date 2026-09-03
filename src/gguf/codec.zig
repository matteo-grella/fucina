//! GGUF tensor-payload codec: `encodeF32`/`decodeF32` between f32 and wire
//! bytes (scalar casts plus the byte-exact ggml-parity block encoders/
//! decoders in backend/quant.zig), the whole-tensor `decodeAllocF32`, and
//! the mmap-backed `RowTable` row-lookup view.

const std = @import("std");

const dtype_mod = @import("../dtype.zig");
const quant = @import("../backend.zig").quant;
const wire = @import("wire.zig");

const Allocator = std.mem.Allocator;
const Error = wire.Error;
const GgmlType = wire.GgmlType;
const TensorInfo = wire.TensorInfo;
const tensorByteLen = wire.tensorByteLen;

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

/// Decode a whole (possibly quantized) tensor into an owned f32 buffer.
pub fn decodeAllocF32(allocator: Allocator, t: *const TensorInfo) ![]f32 {
    var n: usize = 1;
    for (t.dims[0..t.n_dims]) |d| n *= d;
    const buf = try allocator.alloc(f32, n);
    errdefer allocator.free(buf);
    try decodeF32(t.ggml_type, t.data, buf);
    return buf;
}

/// Embedding-style table with on-demand row reads: zero-copy for f32
/// tables, per-row dequantization (Q8_0/K-quants/…) into a caller scratch
/// otherwise — huge tables (vocab-sized embeddings) stay in their quantized
/// mmap instead of ballooning RSS. Validated once at init so `row` cannot
/// fail.
pub const RowTable = struct {
    bytes: []const u8,
    typ: GgmlType,
    width: usize,
    row_bytes: usize,

    pub fn init(allocator: Allocator, t: *const TensorInfo) !RowTable {
        const shape = try t.logicalMatrixShape();
        const self = RowTable{
            .bytes = t.data,
            .typ = t.ggml_type,
            .width = shape[1],
            .row_bytes = t.data.len / shape[0],
        };
        // Validate dtype + alignment once (worst case: the last row).
        const probe = try allocator.alloc(f32, self.width);
        defer allocator.free(probe);
        try decodeF32(self.typ, self.bytes[(shape[0] - 1) * self.row_bytes ..][0..self.row_bytes], probe);
        return self;
    }

    /// Row `i` as f32; when dequantized, the result borrows `scratch` until
    /// the next call with the same scratch.
    pub fn row(self: *const RowTable, i: usize, scratch: []f32) []const f32 {
        const src = self.bytes[i * self.row_bytes ..][0..self.row_bytes];
        if (self.typ == .f32 and std.mem.isAligned(@intFromPtr(src.ptr), @alignOf(f32))) {
            return @as([*]const f32, @ptrCast(@alignCast(src.ptr)))[0..self.width];
        }
        decodeF32(self.typ, src, scratch[0..self.width]) catch unreachable; // validated in init
        return scratch[0..self.width];
    }
};

fn decodeBlocks(comptime dt: dtype_mod.DType, src: []const u8, dst: []f32) !void {
    const Block = dtype_mod.Storage(dt);
    if (@intFromPtr(src.ptr) % @alignOf(Block) != 0) return Error.InvalidTensorInfo;
    const blocks: []const Block = @alignCast(std.mem.bytesAsSlice(Block, src));
    quant.dequantizeRowForDType(dt, dst, blocks) catch return Error.InvalidTensorInfo;
}
