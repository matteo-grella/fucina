//! Shared substrate for the `weights/` subtree: the weight error set, the
//! `QuantWeight` tensor shape, the GGUF block-payload reinterpreters
//! (`blockSlice`, `BlockStorage`), and the loader `LoadOptions`. A leaf:
//! every sibling imports it, it imports none of them.

const std = @import("std");

const dtype_mod = @import("../dtype.zig");
const backend_mod = @import("../backend.zig");
const ag_mod = @import("../ag.zig");

const Tensor = ag_mod.Tensor;
const DType = dtype_mod.DType;

pub const Error = error{
    InvalidWeightShape,
    UnsupportedWeightType,
    GradUnsupported,
};

pub fn QuantWeight(comptime dtype: DType) type {
    return Tensor(.{ .dtype = dtype, .tags = .{ .out, .in } });
}

pub const backend_quant = backend_mod.quantized_matmul;

/// Loader options for `LinearWeight.loadWithOptions` (re-exported there as
/// `LinearWeight.LoadOptions`).
pub const LoadOptions = struct {
    /// Copy provider-supported quant payloads into device-owned storage
    /// (q4_k/q6_k/q8_0 on Metal; those plus q5_k on CUDA) for stable RHS
    /// dequant-in-kernel GEMM. `loadForFusion` turns this off: parts are consumed
    /// by `fuseLinear`, whose concat re-acquires residency for the fused
    /// result, so per-part device copies would be alloc+memcpy+free waste.
    gpu_resident: bool = true,
};

pub fn blockSlice(comptime Elem: type, bytes: []const u8) ![]const Elem {
    if (bytes.len % @sizeOf(Elem) != 0) return Error.InvalidWeightShape;
    if (@intFromPtr(bytes.ptr) % @alignOf(Elem) != 0) return Error.InvalidWeightShape;
    const aligned: []align(@alignOf(Elem)) const u8 = @alignCast(bytes);
    return std.mem.bytesAsSlice(Elem, aligned);
}

pub fn BlockStorage(comptime dtype: DType) type {
    return switch (dtype) {
        .q1_0 => dtype_mod.BlockQ1_0,
        .q2_0 => dtype_mod.BlockQ2_0,
        .q4_0 => dtype_mod.BlockQ4_0,
        .q4_1 => dtype_mod.BlockQ4_1,
        .q5_0 => dtype_mod.BlockQ5_0,
        .q5_1 => dtype_mod.BlockQ5_1,
        .q8_0 => dtype_mod.BlockQ8_0,
        .q2_k => dtype_mod.BlockQ2_K,
        .q3_k => dtype_mod.BlockQ3_K,
        .q4_k => dtype_mod.BlockQ4_K,
        .q5_k => dtype_mod.BlockQ5_K,
        .q6_k => dtype_mod.BlockQ6_K,
        .iq1_s => dtype_mod.BlockIQ1_S,
        .iq1_m => dtype_mod.BlockIQ1_M,
        .iq2_xxs => dtype_mod.BlockIQ2_XXS,
        .iq2_xs => dtype_mod.BlockIQ2_XS,
        .iq2_s => dtype_mod.BlockIQ2_S,
        .iq3_xxs => dtype_mod.BlockIQ3_XXS,
        .iq3_s => dtype_mod.BlockIQ3_S,
        .iq4_nl => dtype_mod.BlockIQ4_NL,
        .iq4_xs => dtype_mod.BlockIQ4_XS,
        .tq1_0 => dtype_mod.BlockTQ1_0,
        .tq2_0 => dtype_mod.BlockTQ2_0,
        .mxfp4 => dtype_mod.BlockMXFP4,
        .nvfp4 => dtype_mod.BlockNVFP4,
        else => @compileError("unsupported quantized weight dtype"),
    };
}
