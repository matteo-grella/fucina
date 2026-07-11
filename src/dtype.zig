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
    q1_0,
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

pub fn Scalar(comptime dtype: DType) type {
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
        .q1_0,
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
        => @compileError("block-quantized dtypes do not have one scalar storage element per logical tensor element"),
    };
}

pub fn Storage(comptime dtype: DType) type {
    return switch (dtype) {
        .q1_0 => BlockQ1_0,
        .q4_0 => BlockQ4_0,
        .q4_1 => BlockQ4_1,
        .q5_0 => BlockQ5_0,
        .q5_1 => BlockQ5_1,
        .q8_0 => BlockQ8_0,
        .q8_1 => BlockQ8_1,
        .q2_k => BlockQ2_K,
        .q3_k => BlockQ3_K,
        .q4_k => BlockQ4_K,
        .q5_k => BlockQ5_K,
        .q6_k => BlockQ6_K,
        .q8_k => BlockQ8_K,
        .iq1_s => BlockIQ1_S,
        .iq1_m => BlockIQ1_M,
        .iq2_xxs => BlockIQ2_XXS,
        .iq2_xs => BlockIQ2_XS,
        .iq2_s => BlockIQ2_S,
        .iq3_xxs => BlockIQ3_XXS,
        .iq3_s => BlockIQ3_S,
        .iq4_nl => BlockIQ4_NL,
        .iq4_xs => BlockIQ4_XS,
        .tq1_0 => BlockTQ1_0,
        .tq2_0 => BlockTQ2_0,
        .mxfp4 => BlockMXFP4,
        .nvfp4 => BlockNVFP4,
        else => Scalar(dtype),
    };
}

pub fn Accumulator(comptime dtype: DType) type {
    return switch (dtype) {
        .f64 => f64,
        .f16, .bf16, .f32 => f32,
        .bool, .u8, .u16 => u64,
        .i8, .i16, .i32, .i64 => i64,
        .q1_0,
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
        => @compileError("block-quantized dtypes do not have scalar accumulators"),
    };
}

pub fn kind(comptime dtype: DType) DTypeKind {
    return switch (dtype) {
        .q1_0,
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
        => .block_quantized,
        else => .scalar,
    };
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

pub fn supportsForwardFloatMath(comptime dtype: DType) bool {
    return switch (dtype) {
        .f16, .bf16, .f32, .f64 => true,
        else => false,
    };
}

pub fn supportsToFloat(comptime dtype: DType) bool {
    return supportsForwardFloatMath(dtype) or isBlockQuantized(dtype);
}

pub fn supportsQuantizedMatmulRhs(comptime dtype: DType) bool {
    return switch (dtype) {
        .q1_0,
        .q4_0,
        .q4_1,
        .q5_0,
        .q5_1,
        .q8_0,
        .q2_k,
        .q3_k,
        .q4_k,
        .q5_k,
        .q6_k,
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
        => true,
        else => false,
    };
}

pub fn supportsQuantizedGetRows(comptime dtype: DType) bool {
    return isBlockQuantized(dtype);
}

pub fn logicalDType(comptime dtype: DType) DType {
    return switch (dtype) {
        .q1_0,
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
        => .f32,
        else => dtype,
    };
}

pub fn blockSize(comptime dtype: DType) usize {
    return switch (dtype) {
        .q1_0 => q1_0_block_size,
        .q4_0 => q4_0_block_size,
        .q4_1 => q4_1_block_size,
        .q5_0 => q5_0_block_size,
        .q5_1 => q5_1_block_size,
        .q8_0 => q8_0_block_size,
        .q8_1 => q8_1_block_size,
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
        .iq4_xs,
        .tq1_0,
        .tq2_0,
        => qk_k_block_size,
        .iq4_nl => iq4_nl_block_size,
        .mxfp4 => mxfp4_block_size,
        .nvfp4 => nvfp4_block_size,
        else => @compileError("scalar dtypes do not have quantized blocks"),
    };
}

pub fn blockByteSize(comptime dtype: DType) usize {
    return @sizeOf(Storage(dtype));
}

pub fn computeDType(comptime op: FloatOp, comptime input_dtype: DType) DType {
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
        else => @compileError("dtype cannot be converted to f32"),
    };
}

pub fn toF64(comptime dtype: DType, value: Scalar(dtype)) f64 {
    return switch (dtype) {
        .f16 => @floatCast(value),
        .bf16 => @floatCast(bf16ToF32(value)),
        .f32 => @floatCast(value),
        .f64 => value,
        else => @compileError("dtype cannot be converted to f64"),
    };
}

pub fn fromF32(comptime dtype: DType, value: f32) Scalar(dtype) {
    return switch (dtype) {
        .f16 => @floatCast(value),
        .bf16 => f32ToBf16(value),
        .f32 => value,
        .f64 => @floatCast(value),
        else => @compileError("dtype cannot be converted from f32"),
    };
}

pub fn fromF64(comptime dtype: DType, value: f64) Scalar(dtype) {
    return switch (dtype) {
        .f16 => @floatCast(value),
        .bf16 => f32ToBf16(@floatCast(value)),
        .f32 => @floatCast(value),
        .f64 => value,
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
        .bool => value != 0,
        .u8, .u16, .i8, .i16, .i32, .i64 => @intCast(value),
        else => @compileError("block-quantized dtypes have no scalar accumulator"),
    };
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
