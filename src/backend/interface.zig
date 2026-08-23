//! The backend kernel interface: every kernel a backend provides, by name.
//! `cpu.zig` and `native.zig` each export `pub const kernels = struct { ... }`
//! with exactly this set, and `conform` is the single compile-time check that
//! holds them together. The interface is a comptime-checked namespace, not a
//! struct of function pointers: many kernels are generic over a `comptime`
//! dtype or op, and those stay generic across the two implementations.
//!
//! Signature rule: a kernel that needs the worker pool takes
//! `pc: ParallelConfig` as its FIRST parameter; a kernel that does not use
//! the pool does not take it. Both implementations follow the same rule for
//! every name, so an exec caller is backend-agnostic. The scalar reference
//! ignores `pc` where the native kernel threads on it.
const std = @import("std");
const ParallelConfig = @import("vector/common.zig").ParallelConfig;

/// The native kernel set is the reference signature of every entry.
const Reference = @import("native.zig").kernels;

/// Every kernel, by name. The order follows the kernel families: elementwise,
/// row/slice helpers, 1-D and 2-D convolution, Winograd, pooling, norm and
/// activation, reductions, dense GEMM, packed and quantized GEMM, batched GEMM.
pub const names = [_][]const u8{
    "addInto",
    "addContiguousIntoUnchecked",
    "divContiguousIntoUnchecked",
    "maximumContiguousIntoUnchecked",
    "minimumContiguousIntoUnchecked",
    "subInto",
    "subContiguousIntoUnchecked",
    "mulInto",
    "mulContiguousIntoUnchecked",
    "elementwiseContiguousIntoTyped",
    "scaleInto",
    "addScaledSlice",
    "addRowVectorSlice",
    "addRowVectorUnarySlice",
    "causalDepthwiseConv1dInto",
    "causalDepthwiseConv1dBackwardInputInto",
    "causalDepthwiseConv1dBackwardKernelInto",
    "causalConv1dInto",
    "conv2dInto",
    "conv2dBackwardInputInto",
    "conv2dBackwardWeightInto",
    "im2colInto",
    "col2imInto",
    "winogradF2WeightTransformInto",
    "winogradF2InputTransformInto",
    "winogradF2OutputTransformInto",
    "winogradF4WeightTransformInto",
    "winogradF4InputTransformInto",
    "winogradF4OutputTransformInto",
    "pool2dInto",
    "avgPool2dBackwardInto",
    "maxPool2dBackwardInto",
    "upsample2xNearestInto",
    "preluChannelsInto",
    "preluChannelsBackwardInputInto",
    "preluChannelsBackwardAlphaInto",
    "channelAffineInto",
    "conv1dInto",
    "conv1dBackwardInputInto",
    "conv1dBackwardWeightInto",
    "col2im1dInto",
    "col2im1dBackwardInto",
    "snakeInto",
    "snakeBackwardInputInto",
    "snakeBackwardParamsInto",
    "groupNormInto",
    "groupNormBackwardInto",
    "causalConv1dBackwardInputInto",
    "causalConv1dBackwardWeightInto",
    "groupedCausalConv1dInto",
    "groupedCausalConv1dBackwardInputInto",
    "groupedCausalConv1dBackwardWeightInto",
    "unaryContiguousIntoUnchecked",
    "leakyReluContiguousIntoUnchecked",
    "clampContiguousIntoUnchecked",
    "gatedContiguousIntoUnchecked",
    "sumInto",
    "sumSlice",
    "prodInto",
    "prodSlice",
    "sumSliceTyped",
    "dotInto",
    "dotIntoTyped",
    "matmulInto",
    "matmul2DIntoUnchecked",
    "matmul2DAccIntoUnchecked",
    "matmul2DIntoUncheckedTyped",
    "packMatmulRhsTyped",
    "packDenseMatmulRhsTyped",
    "matmul2DIntoUncheckedPackedDenseRhs",
    "matmul2DIntoUncheckedPackedRhsTyped",
    "quantizeMatmulRhsBlockwiseI8",
    "quantizeMatmulRhsQ4_0",
    "quantizeMatmulRhsQ8_0",
    "matmul2DQuantizedRhs",
    "matmul2DQuantizedRhsQ8_0x4",
    "matmul2DPackedQ8_0x4LhsRhs",
    "matmulPackedQ4_Kx8Q8_Kx4Slice",
    "matmulPackedQ4_Kx8RowsSlice",
    "matmulPackedQ5_Kx8Q8_Kx4Slice",
    "matmulPackedQ5_Kx8RowsSlice",
    "matmulPackedQ6_Kx4RowsSlice",
    "unaryRowSlice",
    "mulRowSlice",
    "matmul2DPackedPaddedQ8_0x4LhsRhs",
    "matmul2DQuantizedRhsQ6_Kx4",
    "matmul2DQuantizedRhsQ4_Kx4",
    "matmul2DQuantizedRhsQ4_Kx8",
    "matmul2DQuantizedRhsQ4_Kx2Mmla",
    "matmul2DQuantizedRhsQ5_Kx8",
    "matmulTransAInto",
    "matmulTransA2DIntoUnchecked",
    "matmulTransBInto",
    "matmulTransB2DIntoUnchecked",
    "matmulTransB2DIntoUncheckedF16Operands",
    "matmulTransB2DIntoUncheckedBf16Rhs",
    "matmulBatched2DIntoUnchecked",
    "matmulBatchedTransA2DIntoUnchecked",
    "matmulBatchedTransB2DIntoUnchecked",
};

/// Kernels with `comptime` parameters (dtype or op). Zig cannot compare two
/// generic function types, so these are checked by name, parameter count and
/// the `pc` rule only.
pub const generic_names = [_][]const u8{
    "elementwiseContiguousIntoTyped",
    "addRowVectorUnarySlice",
    "pool2dInto",
    "unaryContiguousIntoUnchecked",
    "gatedContiguousIntoUnchecked",
    "sumSliceTyped",
    "dotIntoTyped",
    "matmul2DIntoUncheckedTyped",
    "packMatmulRhsTyped",
    "packDenseMatmulRhsTyped",
    "matmul2DIntoUncheckedPackedRhsTyped",
    "unaryRowSlice",
};

/// Kernels that take no `pc`: they run on the calling thread in both
/// implementations.
pub const pool_free_names = [_][]const u8{
    "addInto",
    "subInto",
    "mulInto",
    "addScaledSlice",
    "addRowVectorSlice",
    "addRowVectorUnarySlice",
    "sumSlice",
    "prodSlice",
    "matmulInto",
    "packMatmulRhsTyped",
    "packDenseMatmulRhsTyped",
    "quantizeMatmulRhsBlockwiseI8",
    "quantizeMatmulRhsQ4_0",
    "quantizeMatmulRhsQ8_0",
    "unaryRowSlice",
    "mulRowSlice",
    "matmulTransAInto",
    "matmulTransBInto",
};

fn contains(comptime list: []const []const u8, comptime name: []const u8) bool {
    for (list) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

/// Two non-generic function types agree when every parameter type is the
/// same and the return types are the same, or are error unions over the same
/// payload. Fallible kernels return `!T` with an inferred error set, and two
/// inferred sets are distinct types by construction, so `@TypeOf` equality
/// alone would reject every fallible twin.
fn sameSignature(comptime Want: type, comptime Got: type) bool {
    const want = @typeInfo(Want).@"fn";
    const got = @typeInfo(Got).@"fn";
    if (want.params.len != got.params.len) return false;
    for (want.params, got.params) |wp, gp| if (wp.type != gp.type) return false;
    const wr = want.return_type.?;
    const gr = got.return_type.?;
    if (wr == gr) return true;
    const wi = @typeInfo(wr);
    const gi = @typeInfo(gr);
    return wi == .error_union and gi == .error_union and wi.error_union.payload == gi.error_union.payload;
}

/// Compile-time conformance: `Impl` declares every name; a non-generic entry
/// has the reference signature (`sameSignature`); a generic entry has the
/// same parameter count; the `pc` rule holds for every entry; and `Impl`
/// declares nothing beyond the set.
pub fn conform(comptime Impl: type) void {
    comptime {
        @setEvalBranchQuota(20_000);
        const impl_name = @typeName(Impl);
        for (names) |name| {
            if (!@hasDecl(Impl, name)) @compileError(impl_name ++ " lacks kernel `" ++ name ++ "`");
            const Want = @TypeOf(@field(Reference, name));
            const Got = @TypeOf(@field(Impl, name));
            const want = @typeInfo(Want).@"fn";
            const got = @typeInfo(Got).@"fn";
            const generic = contains(&generic_names, name);
            if (want.is_generic != generic or got.is_generic != generic) @compileError(
                "kernel `" ++ name ++ "` disagrees with `generic_names` on having comptime parameters",
            );
            if (generic) {
                if (want.params.len != got.params.len) @compileError(std.fmt.comptimePrint(
                    "{s}.{s} takes {d} parameters, the reference takes {d}",
                    .{ impl_name, name, got.params.len, want.params.len },
                ));
            } else if (!sameSignature(Want, Got)) @compileError(
                impl_name ++ "." ++ name ++ " is " ++ @typeName(Got) ++ ", the reference is " ++ @typeName(Want),
            );
            const takes_pc = got.params.len > 0 and got.params[0].type == ParallelConfig;
            if (takes_pc == contains(&pool_free_names, name)) @compileError(
                impl_name ++ "." ++ name ++ " disagrees with `pool_free_names` on taking `pc: ParallelConfig` first",
            );
        }
        const decl_count = @typeInfo(Impl).@"struct".decls.len;
        if (decl_count != names.len) @compileError(std.fmt.comptimePrint(
            "{s} declares {d} kernels, the interface names {d}",
            .{ impl_name, decl_count, names.len },
        ));
    }
}
