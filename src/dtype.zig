//! Element dtypes and the quantized block-format vocabulary: the `DType`
//! enum, scalar converters (bf16/f8), every GGML block struct, and the
//! comptime `block_formats` registry (one row per quantized dtype, claimed
//! exactly once — completeness is a compile error, and GGUF derives its
//! type mapping from it). Layer stack: docs/ARCHITECTURE.md.
const std = @import("std");

pub const DType = enum {
    bool,
    u8,
    u16,
    i8,
    i16,
    i32,
    i64,
    f16,
    bf16,
    f32,
    f64,
    f8_e4m3,
    f8_e5m2,
    q1_0,
    q2_0,
    q4_0,
    q4_1,
    q5_0,
    q5_1,
    q8_0,
    q8_1,
    q2_k,
    q3_k,
    q4_k,
    q5_k,
    q6_k,
    q8_k,
    iq1_s,
    iq1_m,
    iq2_xxs,
    iq2_xs,
    iq2_s,
    iq3_xxs,
    iq3_s,
    iq4_nl,
    iq4_xs,
    tq1_0,
    tq2_0,
    mxfp4,
    nvfp4,
};

pub const FloatOp = enum {
    pointwise,
    reduction,
    matmul,
};

pub const DTypeKind = enum {
    scalar,
    block_quantized,
};

pub const q1_0_block_size: usize = 128;
pub const q2_0_block_size: usize = 128;
pub const q4_0_block_size: usize = 32;
pub const q4_1_block_size: usize = 32;
pub const q5_0_block_size: usize = 32;
pub const q5_1_block_size: usize = 32;
pub const q8_0_block_size: usize = 32;
pub const q8_1_block_size: usize = 32;
pub const qk_k_block_size: usize = 256;
pub const k_scale_size: usize = 12;
pub const iq4_nl_block_size: usize = 32;
pub const mxfp4_block_size: usize = 32;
pub const nvfp4_block_size: usize = 64;
pub const nvfp4_subblock_size: usize = 16;
pub const iq3s_n_scale: usize = qk_k_block_size / 64;

pub const BlockQ1_0 = extern struct {
    d: u16,
    qs: [q1_0_block_size / 8]u8,
};

pub const BlockQ2_0 = extern struct {
    d: u16,
    qs: [q2_0_block_size / 4]u8,
};

pub const BlockQ8_0 = extern struct {
    d: u16,
    qs: [q8_0_block_size]i8,
};

pub const BlockQ8_1 = extern struct {
    ds: [2]u16,
    qs: [q8_1_block_size]i8,
};

pub const BlockQ4_0 = extern struct {
    d: u16,
    qs: [q4_0_block_size / 2]u8,
};

pub const BlockQ4_1 = extern struct {
    dm: [2]u16,
    qs: [q4_1_block_size / 2]u8,
};

pub const BlockQ5_0 = extern struct {
    d: u16,
    qh: [4]u8,
    qs: [q5_0_block_size / 2]u8,
};

pub const BlockQ5_1 = extern struct {
    dm: [2]u16,
    qh: [4]u8,
    qs: [q5_1_block_size / 2]u8,
};

pub const BlockQ2_K = extern struct {
    scales: [qk_k_block_size / 16]u8,
    qs: [qk_k_block_size / 4]u8,
    dm: [2]u16,
};

pub const BlockQ3_K = extern struct {
    hmask: [qk_k_block_size / 8]u8,
    qs: [qk_k_block_size / 4]u8,
    scales: [12]u8,
    d: u16,
};

pub const BlockQ4_K = extern struct {
    dm: [2]u16,
    scales: [k_scale_size]u8,
    qs: [qk_k_block_size / 2]u8,
};

pub const BlockQ5_K = extern struct {
    dm: [2]u16,
    scales: [k_scale_size]u8,
    qh: [qk_k_block_size / 8]u8,
    qs: [qk_k_block_size / 2]u8,
};

pub const BlockQ6_K = extern struct {
    ql: [qk_k_block_size / 2]u8,
    qh: [qk_k_block_size / 4]u8,
    scales: [qk_k_block_size / 16]i8,
    d: u16,
};

pub const BlockQ8_K = extern struct {
    d: f32,
    qs: [qk_k_block_size]i8,
    bsums: [qk_k_block_size / 16]i16,
};

pub const BlockIQ2_XXS = extern struct {
    d: u16,
    qs: [qk_k_block_size / 8]u16,
};

pub const BlockIQ2_XS = extern struct {
    d: u16,
    qs: [qk_k_block_size / 8]u16,
    scales: [qk_k_block_size / 32]u8,
};

pub const BlockIQ2_S = extern struct {
    d: u16,
    qs: [qk_k_block_size / 4]u8,
    qh: [qk_k_block_size / 32]u8,
    scales: [qk_k_block_size / 32]u8,
};

pub const BlockIQ3_XXS = extern struct {
    d: u16,
    qs: [3 * qk_k_block_size / 8]u8,
};

pub const BlockIQ3_S = extern struct {
    d: u16,
    qs: [qk_k_block_size / 4]u8,
    qh: [qk_k_block_size / 32]u8,
    signs: [qk_k_block_size / 8]u8,
    scales: [iq3s_n_scale]u8,
};

pub const BlockIQ1_S = extern struct {
    d: u16,
    qs: [qk_k_block_size / 8]u8,
    qh: [qk_k_block_size / 32]u16,
};

pub const BlockIQ1_M = extern struct {
    qs: [qk_k_block_size / 8]u8,
    qh: [qk_k_block_size / 16]u8,
    scales: [qk_k_block_size / 32]u8,
};

pub const BlockIQ4_NL = extern struct {
    d: u16,
    qs: [iq4_nl_block_size / 2]u8,
};

pub const BlockIQ4_XS = extern struct {
    d: u16,
    scales_h: u16,
    scales_l: [qk_k_block_size / 64]u8,
    qs: [qk_k_block_size / 2]u8,
};

pub const BlockTQ1_0 = extern struct {
    qs: [(qk_k_block_size - 4 * qk_k_block_size / 64) / 5]u8,
    qh: [qk_k_block_size / 64]u8,
    d: u16,
};

pub const BlockTQ2_0 = extern struct {
    qs: [qk_k_block_size / 4]u8,
    d: u16,
};

pub const BlockMXFP4 = extern struct {
    e: u8,
    qs: [mxfp4_block_size / 2]u8,
};

pub const BlockNVFP4 = extern struct {
    d: [nvfp4_block_size / nvfp4_subblock_size]u8,
    qs: [nvfp4_block_size / 2]u8,
};

comptime {
    std.debug.assert(@sizeOf(BlockQ1_0) == 18);
    std.debug.assert(@sizeOf(BlockQ2_0) == 34);
    std.debug.assert(@sizeOf(BlockQ4_0) == 18);
    std.debug.assert(@sizeOf(BlockQ4_1) == 20);
    std.debug.assert(@sizeOf(BlockQ5_0) == 22);
    std.debug.assert(@sizeOf(BlockQ5_1) == 24);
    std.debug.assert(@sizeOf(BlockQ8_0) == 34);
    std.debug.assert(@sizeOf(BlockQ8_1) == 36);
    std.debug.assert(@sizeOf(BlockQ2_K) == 84);
    std.debug.assert(@sizeOf(BlockQ3_K) == 110);
    std.debug.assert(@sizeOf(BlockQ4_K) == 144);
    std.debug.assert(@sizeOf(BlockQ5_K) == 176);
    std.debug.assert(@sizeOf(BlockQ6_K) == 210);
    std.debug.assert(@sizeOf(BlockQ8_K) == 292);
    std.debug.assert(@sizeOf(BlockIQ2_XXS) == 66);
    std.debug.assert(@sizeOf(BlockIQ2_XS) == 74);
    std.debug.assert(@sizeOf(BlockIQ2_S) == 82);
    std.debug.assert(@sizeOf(BlockIQ3_XXS) == 98);
    std.debug.assert(@sizeOf(BlockIQ3_S) == 110);
    std.debug.assert(@sizeOf(BlockIQ1_S) == 50);
    std.debug.assert(@sizeOf(BlockIQ1_M) == 56);
    std.debug.assert(@sizeOf(BlockIQ4_NL) == 18);
    std.debug.assert(@sizeOf(BlockIQ4_XS) == 136);
    std.debug.assert(@sizeOf(BlockTQ1_0) == 54);
    std.debug.assert(@sizeOf(BlockTQ2_0) == 66);
    std.debug.assert(@sizeOf(BlockMXFP4) == 17);
    std.debug.assert(@sizeOf(BlockNVFP4) == 36);
}

/// One registry row per block-quantized storage format.
pub const BlockFormat = struct {
    dtype: DType,
    Block: type,
    /// Logical elements per block (`blockSize`).
    elems: usize,
    /// Whether the format is a valid stored-RHS side of the quantized
    /// matmul boundary (`supportsQuantizedMatmulRhs` derives from this).
    /// False for the accumulator/LHS-side formats (q8_1, q8_k). No default
    /// on purpose: a new row must claim its capability explicitly.
    matmul_rhs: bool,
};

/// The block-format registry: the single source every per-format
/// enumeration derives from (`Storage`, `kind`, the scalar/accumulator
/// exclusions, GGUF's `dtypeForGgmlType`). Adding a format is: define its
/// packed Block struct (with the size assert above), add its `DType` tag,
/// add one row here — the completeness check below makes a tag claimed by
/// neither this table nor `scalar_dtypes` (or by both) a compile error, so
/// a forgotten row can never fall through a switch as a scalar.
pub const block_formats = [_]BlockFormat{
    .{ .dtype = .q1_0, .Block = BlockQ1_0, .elems = q1_0_block_size, .matmul_rhs = true },
    .{ .dtype = .q2_0, .Block = BlockQ2_0, .elems = q2_0_block_size, .matmul_rhs = true },
    .{ .dtype = .q4_0, .Block = BlockQ4_0, .elems = q4_0_block_size, .matmul_rhs = true },
    .{ .dtype = .q4_1, .Block = BlockQ4_1, .elems = q4_1_block_size, .matmul_rhs = true },
    .{ .dtype = .q5_0, .Block = BlockQ5_0, .elems = q5_0_block_size, .matmul_rhs = true },
    .{ .dtype = .q5_1, .Block = BlockQ5_1, .elems = q5_1_block_size, .matmul_rhs = true },
    .{ .dtype = .q8_0, .Block = BlockQ8_0, .elems = q8_0_block_size, .matmul_rhs = true },
    .{ .dtype = .q8_1, .Block = BlockQ8_1, .elems = q8_1_block_size, .matmul_rhs = false },
    .{ .dtype = .q2_k, .Block = BlockQ2_K, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .q3_k, .Block = BlockQ3_K, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .q4_k, .Block = BlockQ4_K, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .q5_k, .Block = BlockQ5_K, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .q6_k, .Block = BlockQ6_K, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .q8_k, .Block = BlockQ8_K, .elems = qk_k_block_size, .matmul_rhs = false },
    .{ .dtype = .iq1_s, .Block = BlockIQ1_S, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .iq1_m, .Block = BlockIQ1_M, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .iq2_xxs, .Block = BlockIQ2_XXS, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .iq2_xs, .Block = BlockIQ2_XS, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .iq2_s, .Block = BlockIQ2_S, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .iq3_xxs, .Block = BlockIQ3_XXS, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .iq3_s, .Block = BlockIQ3_S, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .iq4_nl, .Block = BlockIQ4_NL, .elems = iq4_nl_block_size, .matmul_rhs = true },
    .{ .dtype = .iq4_xs, .Block = BlockIQ4_XS, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .tq1_0, .Block = BlockTQ1_0, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .tq2_0, .Block = BlockTQ2_0, .elems = qk_k_block_size, .matmul_rhs = true },
    .{ .dtype = .mxfp4, .Block = BlockMXFP4, .elems = mxfp4_block_size, .matmul_rhs = true },
    .{ .dtype = .nvfp4, .Block = BlockNVFP4, .elems = nvfp4_block_size, .matmul_rhs = true },
};

/// The registry's complement: every non-block dtype, listed once.
const scalar_dtypes = [_]DType{
    .bool, .u8,   .u16, .i8,  .i16,     .i32,     .i64,
    .f16,  .bf16, .f32, .f64, .f8_e4m3, .f8_e5m2,
};

comptime {
    @setEvalBranchQuota(10_000);
    // Completeness: every DType tag appears exactly once across
    // `scalar_dtypes` and `block_formats`. Fires on any build that analyzes
    // this module, not just on the first mishandled use.
    for (@typeInfo(DType).@"enum".fields) |field| {
        const dt: DType = @enumFromInt(field.value);
        var claims: usize = 0;
        for (scalar_dtypes) |s| {
            if (s == dt) claims += 1;
        }
        for (block_formats) |row| {
            if (row.dtype == dt) claims += 1;
        }
        if (claims != 1) @compileError("DType ." ++ field.name ++
            " must appear exactly once across dtype.scalar_dtypes and dtype.block_formats");
    }
}

fn blockFormatIndex(comptime dtype: DType) ?usize {
    @setEvalBranchQuota(10_000);
    inline for (block_formats, 0..) |row, i| {
        if (row.dtype == dtype) return i;
    }
    return null;
}

pub fn Scalar(comptime dtype: DType) type {
    if (comptime blockFormatIndex(dtype) != null)
        @compileError("block-quantized dtypes do not have one scalar storage element per logical tensor element");
    return switch (dtype) {
        .bool => bool,
        .u8 => u8,
        .u16 => u16,
        .i8 => i8,
        .i16 => i16,
        .i32 => i32,
        .i64 => i64,
        .f16 => f16,
        .bf16 => u16,
        .f32 => f32,
        .f64 => f64,
        .f8_e4m3, .f8_e5m2 => u8,
        else => unreachable, // claimed by block_formats (completeness check above)
    };
}

pub fn Storage(comptime dtype: DType) type {
    if (comptime blockFormatIndex(dtype)) |i| return block_formats[i].Block;
    return Scalar(dtype);
}

pub fn Accumulator(comptime dtype: DType) type {
    if (comptime blockFormatIndex(dtype) != null)
        @compileError("block-quantized dtypes do not have scalar accumulators");
    return switch (dtype) {
        .f64 => f64,
        .f16, .bf16, .f32 => f32,
        .f8_e4m3, .f8_e5m2 => f32,
        .bool, .u8, .u16 => u64,
        .i8, .i16, .i32, .i64 => i64,
        else => unreachable, // claimed by block_formats (completeness check above)
    };
}

pub fn kind(comptime dtype: DType) DTypeKind {
    return if (comptime blockFormatIndex(dtype) != null) .block_quantized else .scalar;
}

pub fn isScalar(comptime dtype: DType) bool {
    return kind(dtype) == .scalar;
}

pub fn isBlockQuantized(comptime dtype: DType) bool {
    return kind(dtype) == .block_quantized;
}

pub fn isFloat(comptime dtype: DType) bool {
    return switch (dtype) {
        .f16, .bf16, .f32, .f64 => true,
        else => false,
    };
}

/// 8-bit storage floats (OCP FP8: f8_e4m3 is the E4M3FN variant with NaN
/// but no infinities; f8_e5m2 is IEEE-like with infinities). Storage-only:
/// they convert to/from f32 but are excluded from forward math and grads.
pub fn isF8(comptime dtype: DType) bool {
    return switch (dtype) {
        .f8_e4m3, .f8_e5m2 => true,
        else => false,
    };
}

pub fn isInteger(comptime dtype: DType) bool {
    return switch (dtype) {
        .u8, .u16, .i8, .i16, .i32, .i64 => true,
        else => false,
    };
}

pub fn isSignedInteger(comptime dtype: DType) bool {
    return switch (dtype) {
        .i8, .i16, .i32, .i64 => true,
        else => false,
    };
}

pub fn isUnsignedInteger(comptime dtype: DType) bool {
    return switch (dtype) {
        .u8, .u16 => true,
        else => false,
    };
}

pub fn supportsGrad(comptime dtype: DType) bool {
    return isFloat(dtype);
}

/// Ordinary integer pointwise math (wrapping add/sub/mul, max/min,
/// explicit divTrunc/divFloor) and i64-accumulated reductions. bool is
/// excluded from pointwise math but its reductions (sum = count) apply.
pub fn supportsIntMath(comptime dtype: DType) bool {
    return switch (dtype) {
        .u8, .u16, .i8, .i16, .i32, .i64 => true,
        else => false,
    };
}

fn isScalarIntegerOrBool(comptime dtype: DType) bool {
    return dtype == .bool or supportsIntMath(dtype);
}

pub fn supportsForwardFloatMath(comptime dtype: DType) bool {
    return switch (dtype) {
        .f16, .bf16, .f32, .f64 => true,
        else => false,
    };
}

pub fn supportsToFloat(comptime dtype: DType) bool {
    return supportsForwardFloatMath(dtype) or isF8(dtype) or isBlockQuantized(dtype);
}

pub fn supportsQuantizedMatmulRhs(comptime dtype: DType) bool {
    // Derived from the registry so the capability can never silently lag a
    // new format: every `block_formats` row claims `matmul_rhs` explicitly.
    return comptime blk: {
        for (block_formats) |row| {
            if (row.dtype == dtype) break :blk row.matmul_rhs;
        }
        break :blk false;
    };
}

pub fn supportsQuantizedGetRows(comptime dtype: DType) bool {
    return isBlockQuantized(dtype);
}

pub fn logicalDType(comptime dtype: DType) DType {
    return switch (dtype) {
        .q1_0,
        .q2_0,
        .q4_0,
        .q4_1,
        .q5_0,
        .q5_1,
        .q8_0,
        .q8_1,
        .q2_k,
        .q3_k,
        .q4_k,
        .q5_k,
        .q6_k,
        .q8_k,
        .iq1_s,
        .iq1_m,
        .iq2_xxs,
        .iq2_xs,
        .iq2_s,
        .iq3_xxs,
        .iq3_s,
        .iq4_nl,
        .iq4_xs,
        .tq1_0,
        .tq2_0,
        .mxfp4,
        .nvfp4,
        .f8_e4m3,
        .f8_e5m2,
        => .f32,
        else => dtype,
    };
}

pub fn blockSize(comptime dtype: DType) usize {
    if (comptime blockFormatIndex(dtype)) |i| return block_formats[i].elems;
    @compileError("scalar dtypes do not have quantized blocks");
}

pub fn blockByteSize(comptime dtype: DType) usize {
    return @sizeOf(Storage(dtype));
}

pub fn computeDType(comptime op: FloatOp, comptime input_dtype: DType) DType {
    // Integer/bool policy: pointwise math runs in the input dtype
    // (wrapping two's complement); reductions accumulate in i64.
    if (isScalarIntegerOrBool(input_dtype)) return switch (op) {
        .pointwise, .matmul => input_dtype,
        .reduction => .i64,
    };
    if (!supportsForwardFloatMath(input_dtype)) return input_dtype;
    return switch (op) {
        .pointwise => switch (input_dtype) {
            .bf16 => .f32,
            else => input_dtype,
        },
        .reduction, .matmul => switch (input_dtype) {
            .f16, .bf16, .f32 => .f32,
            .f64 => .f64,
            else => unreachable,
        },
    };
}

pub fn outputDType(comptime op: FloatOp, comptime input_dtype: DType) DType {
    // Integer/bool reductions RETURN i64 (torch's integer-sum dtype);
    // pointwise keeps the input dtype.
    if (isScalarIntegerOrBool(input_dtype)) return switch (op) {
        .pointwise, .matmul => input_dtype,
        .reduction => .i64,
    };
    if (!supportsForwardFloatMath(input_dtype)) return input_dtype;
    return switch (op) {
        .pointwise, .matmul => input_dtype,
        .reduction => switch (input_dtype) {
            .f16, .bf16 => .f32,
            .f32 => .f32,
            .f64 => .f64,
            else => unreachable,
        },
    };
}

pub fn zero(comptime dtype: DType) Scalar(dtype) {
    return switch (dtype) {
        .bool => false,
        else => @as(Scalar(dtype), 0),
    };
}

pub fn one(comptime dtype: DType) Scalar(dtype) {
    return switch (dtype) {
        .bool => true,
        .bf16 => 0x3f80,
        .f8_e4m3 => 0x38,
        .f8_e5m2 => 0x3c,
        else => @as(Scalar(dtype), 1),
    };
}

pub fn name(comptime dtype: DType) []const u8 {
    return @tagName(dtype);
}

pub fn toF32(comptime dtype: DType, value: Scalar(dtype)) f32 {
    return switch (dtype) {
        .f16 => @floatCast(value),
        .bf16 => bf16ToF32(value),
        .f32 => value,
        .f64 => @floatCast(value),
        .f8_e4m3 => f8e4m3ToF32(value),
        .f8_e5m2 => f8e5m2ToF32(value),
        else => @compileError("dtype cannot be converted to f32"),
    };
}

pub fn toF64(comptime dtype: DType, value: Scalar(dtype)) f64 {
    return switch (dtype) {
        .f16 => @floatCast(value),
        .bf16 => @floatCast(bf16ToF32(value)),
        .f32 => @floatCast(value),
        .f64 => value,
        .f8_e4m3 => @floatCast(f8e4m3ToF32(value)),
        .f8_e5m2 => @floatCast(f8e5m2ToF32(value)),
        else => @compileError("dtype cannot be converted to f64"),
    };
}

pub fn fromF32(comptime dtype: DType, value: f32) Scalar(dtype) {
    return switch (dtype) {
        .f16 => @floatCast(value),
        .bf16 => f32ToBf16(value),
        .f32 => value,
        .f64 => @floatCast(value),
        .f8_e4m3 => f32ToF8e4m3(value),
        .f8_e5m2 => f32ToF8e5m2(value),
        else => @compileError("dtype cannot be converted from f32"),
    };
}

pub fn fromF64(comptime dtype: DType, value: f64) Scalar(dtype) {
    return switch (dtype) {
        .f16 => @floatCast(value),
        .bf16 => f32ToBf16(@floatCast(value)),
        .f32 => @floatCast(value),
        .f64 => value,
        .f8_e4m3 => f32ToF8e4m3(@floatCast(value)),
        .f8_e5m2 => f32ToF8e5m2(@floatCast(value)),
        else => @compileError("dtype cannot be converted from f64"),
    };
}

pub fn castFloat(comptime source_dtype: DType, comptime target_dtype: DType, value: Scalar(source_dtype)) Scalar(target_dtype) {
    if (comptime source_dtype == target_dtype) return value;
    if (comptime target_dtype == .f64) return fromF64(target_dtype, toF64(source_dtype, value));
    return fromF32(target_dtype, @floatCast(toF64(source_dtype, value)));
}

pub fn toAccumulator(comptime dtype: DType, value: Scalar(dtype)) Accumulator(dtype) {
    return switch (dtype) {
        .f16 => @as(f32, @floatCast(value)),
        .bf16 => bf16ToF32(value),
        .f32 => value,
        .f64 => value,
        .f8_e4m3 => f8e4m3ToF32(value),
        .f8_e5m2 => f8e5m2ToF32(value),
        .bool => if (value) 1 else 0,
        .u8, .u16, .i8, .i16, .i32, .i64 => @intCast(value),
        else => @compileError("block-quantized dtypes have no scalar accumulator"),
    };
}

pub fn fromAccumulator(comptime dtype: DType, value: Accumulator(dtype)) Scalar(dtype) {
    return switch (dtype) {
        .f16 => @floatCast(value),
        .bf16 => f32ToBf16(value),
        .f32 => value,
        .f64 => value,
        .f8_e4m3 => f32ToF8e4m3(value),
        .f8_e5m2 => f32ToF8e5m2(value),
        .bool => value != 0,
        .u8, .u16, .i8, .i16, .i32, .i64 => @intCast(value),
        else => @compileError("block-quantized dtypes have no scalar accumulator"),
    };
}

fn intToI64(comptime from: DType, v: Scalar(from)) i64 {
    return switch (from) {
        .bool => @intFromBool(v),
        else => @as(i64, v),
    };
}

fn wrapFromI64(comptime to: DType, wide: i64) Scalar(to) {
    const bits: u64 = @bitCast(wide);
    return switch (to) {
        .bool => wide != 0,
        .u8 => @truncate(bits),
        .u16 => @truncate(bits),
        .i8 => @bitCast(@as(u8, @truncate(bits))),
        .i16 => @bitCast(@as(u16, @truncate(bits))),
        .i32 => @bitCast(@as(u32, @truncate(bits))),
        .i64 => wide,
        else => @compileError("wrapFromI64 targets integer/bool dtypes"),
    };
}

/// General scalar cast between the non-block-quantized dtypes:
/// float ↔ float = `castFloat`; integer ↔ integer wraps (two's
/// complement, the torch narrowing behavior); float → integer truncates
/// toward zero and SATURATES at the target bounds with NaN → 0 (defined
/// everywhere — torch's CPU float-to-int overflow is unspecified);
/// anything → bool is `!= 0` (NaN → true); bool → number is 0/1.
pub fn castScalar(comptime from: DType, comptime to: DType, v: Scalar(from)) Scalar(to) {
    if (comptime from == to) return v;
    // f8 bridges through f32 (its u8 storage must not take the integer
    // paths below); float-to-f8 overflow follows the format: e4m3 has no
    // infinities so it saturates to NaN, e5m2 overflows to ±inf.
    if (comptime isF8(from)) return castScalar(.f32, to, toF32(from, v));
    if (comptime isF8(to)) return fromF32(to, castScalar(from, .f32, v));
    const from_float = comptime supportsForwardFloatMath(from);
    const to_float = comptime supportsForwardFloatMath(to);
    if (comptime (from_float and to_float)) return castFloat(from, to, v);
    if (comptime (!from_float and !to_float)) return wrapFromI64(to, intToI64(from, v));
    if (comptime from_float) {
        const d = toF64(from, v);
        if (comptime to == .bool) return d != 0;
        if (std.math.isNan(d)) return 0;
        const t = @trunc(d);
        const lo: f64 = @floatFromInt(std.math.minInt(Scalar(to)));
        const hi: f64 = @floatFromInt(std.math.maxInt(Scalar(to)));
        if (t <= lo) return std.math.minInt(Scalar(to));
        if (t >= hi) return std.math.maxInt(Scalar(to));
        return @intFromFloat(t);
    }
    return fromF64(to, @floatFromInt(intToI64(from, v)));
}

/// Mask truthiness across the scalar dtypes: `!= 0` (bf16 goes through
/// the value bridge so -0.0 stays falsy; NaN is truthy).
pub fn isTruthy(comptime dtype: DType, value: Scalar(dtype)) bool {
    return switch (dtype) {
        .bool => value,
        .bf16 => bf16ToF32(value) != 0,
        .f8_e4m3 => f8e4m3ToF32(value) != 0,
        .f8_e5m2 => f8e5m2ToF32(value) != 0,
        else => value != 0,
    };
}

/// Decode tables for the OCP FP8 formats. Every representable value is
/// exact in f32; NaN codes decode to a NaN carrying the code's sign.
const f8_e4m3_decode = f8DecodeTable(4, 3, false);
const f8_e5m2_decode = f8DecodeTable(5, 2, true);

fn f8DecodeTable(comptime exp_bits: u4, comptime mant_bits: u4, comptime has_inf: bool) [256]f32 {
    @setEvalBranchQuota(10_000);
    const bias: i32 = (1 << (exp_bits - 1)) - 1;
    const max_exp_field: i32 = (1 << exp_bits) - 1;
    var table: [256]f32 = undefined;
    for (0..256) |code| {
        const negative = code >> 7 == 1;
        const e: i32 = @intCast((code >> mant_bits) & ((1 << exp_bits) - 1));
        const m: i32 = @intCast(code & ((1 << mant_bits) - 1));
        const v: f32 = if (has_inf and e == max_exp_field)
            (if (m == 0) std.math.inf(f32) else std.math.nan(f32))
        else if (!has_inf and e == max_exp_field and m == (1 << mant_bits) - 1)
            std.math.nan(f32)
        else if (e == 0)
            std.math.ldexp(@as(f32, @floatFromInt(m)), 1 - bias - mant_bits)
        else
            std.math.ldexp(@as(f32, @floatFromInt((1 << mant_bits) + m)), e - bias - mant_bits);
        table[code] = if (negative) -v else v;
    }
    return table;
}

pub fn f8e4m3ToF32(bits: u8) f32 {
    return f8_e4m3_decode[bits];
}

pub fn f8e5m2ToF32(bits: u8) f32 {
    return f8_e5m2_decode[bits];
}

pub fn f32ToF8e4m3(value: f32) u8 {
    return f32ToF8(4, 3, false, value);
}

pub fn f32ToF8e5m2(value: f32) u8 {
    return f32ToF8(5, 2, true, value);
}

/// Round-to-nearest-even encode. NaN keeps its sign with the format's
/// canonical NaN code. Overflow follows the format: e4m3 (no infinities)
/// goes to NaN like torch's float8_e4m3fn cast, e5m2 goes to ±inf.
fn f32ToF8(comptime exp_bits: u4, comptime mant_bits: u4, comptime has_inf: bool, value: f32) u8 {
    const bias: i32 = (1 << (exp_bits - 1)) - 1;
    const max_exp_field: i32 = (1 << exp_bits) - 1;
    const nan_code: u8 = if (has_inf)
        @intCast((max_exp_field << mant_bits) | (1 << (mant_bits - 1)))
    else
        @intCast((max_exp_field << mant_bits) | ((1 << mant_bits) - 1));
    const mshift: u5 = 23 - @as(u5, mant_bits);

    const bits: u32 = @bitCast(value);
    const sign: u8 = @intCast((bits >> 31) << 7);
    const abs: u32 = bits & 0x7fff_ffff;
    if (abs > 0x7f80_0000) return sign | nan_code;

    const e_f32: i32 = @as(i32, @intCast(abs >> 23)) - 127;
    const m23: u32 = abs & 0x7f_ffff;
    const min_normal_e: i32 = 1 - bias;
    var e_field: i32 = undefined;
    var m_field: u32 = undefined;
    if (e_f32 >= min_normal_e) {
        m_field = rneShift(m23, mshift);
        e_field = e_f32 + bias;
        if (m_field == (1 << mant_bits)) {
            m_field = 0;
            e_field += 1;
        }
        const overflow = if (has_inf)
            e_field >= max_exp_field
        else
            e_field > max_exp_field or (e_field == max_exp_field and m_field == (1 << mant_bits) - 1);
        if (overflow) {
            const inf_code: u8 = @intCast(max_exp_field << mant_bits);
            return sign | (if (has_inf) inf_code else nan_code);
        }
    } else {
        // Subnormal target: rescale so one target quantum is one ulp.
        // f32 subnormal inputs land in the rounds-to-zero branch for both
        // formats, so treating their exponent field as biased-0 is moot.
        const total: i32 = @as(i32, mshift) + (min_normal_e - e_f32);
        if (total >= 25) return sign;
        m_field = rneShift(m23 | 0x0080_0000, @intCast(total));
        e_field = 0;
        if (m_field == (1 << mant_bits)) {
            m_field = 0;
            e_field = 1;
        }
    }
    return sign | @as(u8, @intCast(e_field << mant_bits)) | @as(u8, @intCast(m_field));
}

fn rneShift(v: u32, shift: u5) u32 {
    const keep = v >> shift;
    const round = (v >> (shift - 1)) & 1;
    const sticky = (v & ((@as(u32, 1) << (shift - 1)) - 1)) != 0;
    return keep + (round & @intFromBool(sticky or (keep & 1) == 1));
}

pub fn bf16ToF32(bits: u16) f32 {
    const widened: u32 = @as(u32, bits) << 16;
    return @bitCast(widened);
}

pub fn f32ToBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    if ((bits & 0x7fff_ffff) > 0x7f80_0000) {
        return @truncate((bits >> 16) | 64);
    }
    const lsb = (bits >> 16) & 1;
    const rounded = bits + 0x7fff + lsb;
    return @truncate(rounded >> 16);
}

test {
    _ = @import("dtype_tests.zig");
}
