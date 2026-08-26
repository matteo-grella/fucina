//! Load-time packed dense matmul RHS: the f32 output-row panel every dense
//! weight dtype shares (`PackedDenseRhs`, `dtype == .f32`; f16/bf16 sources
//! widen exactly once at pack time). The container names the dtype it
//! serves with `pub const dtype`; the container type is the layout.
const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const tensor = @import("../tensor.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

/// Load-time f32 RHS panels for `A[m,k] * W[n,k]^T`. Rows are padded to the
/// four-output microkernel tile, while every logical row remains contiguous so
/// GPU and BLAS NT dispatch can consume the same stable storage. The source is
/// copied: the pack is an immutable snapshot and outlives the source tensor.
pub const PackedDenseRhs = struct {
    rhs: Tensor,
    k: usize,
    n: usize,
    padded_n: usize,

    const Self = @This();
    /// The panel's storage dtype; f16 and bf16 sources share it.
    pub const dtype: DType = .f32;

    pub fn deinit(self: *Self) void {
        self.rhs.deinit();
        self.* = undefined;
    }
};

/// Build dense f32 output-row panels from an f32, f16, or bf16 `[n,k]`
/// weight. The 16-bit conversions are exact widenings; arithmetic happens
/// only when the packed panel is consumed.
pub fn packDenseRhs(
    allocator: Allocator,
    comptime dtype: DType,
    rhs: *const tensor.TensorOf(dtype),
) !PackedDenseRhs {
    comptime if (dtype != .f32 and dtype != .f16 and dtype != .bf16)
        @compileError("dense packed matmul RHS supports f32, f16, and bf16 weights");

    const view = try rhs.rankView(2);
    if (!rhs.isContiguous()) return tensor.TensorError.UnsupportedView;
    const n = view.dim(0);
    const k = view.dim(1);
    const padded_n = std.mem.alignForward(usize, n, 4);
    var packed_rhs = try Tensor.zeros(allocator, &.{ padded_n, k });
    errdefer packed_rhs.deinit();
    const dst = packed_rhs.data()[0 .. n * k];
    switch (comptime dtype) {
        .f32 => @memcpy(dst, try rhs.dataConstChecked()),
        .f16 => widenF16ToF32(dst, try rhs.dataConstChecked()),
        .bf16 => widenBf16ToF32(dst, try rhs.dataConstChecked()),
        else => comptime unreachable,
    }
    return .{ .rhs = packed_rhs, .k = k, .n = n, .padded_n = padded_n };
}

/// Scalar-backend specification for the dense packed operation.
pub fn matmulDenseScalar(
    out: []f32,
    lhs: []const f32,
    rhs: *const PackedDenseRhs,
    m: usize,
) void {
    for (0..m) |i| {
        for (0..rhs.n) |j| {
            var sum: f32 = 0;
            for (0..rhs.k) |p| sum += lhs[i * rhs.k + p] * rhs.rhs.dataConst()[j * rhs.k + p];
            out[i * rhs.n + j] = sum;
        }
    }
}

const f16_bridge_vector_len: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;
const Vf16Bridge = @Vector(f16_bridge_vector_len, f16);
const Vf32Bridge = @Vector(f16_bridge_vector_len, f32);

pub fn widenF16ToF32(dst: []f32, src: []const f16) void {
    var i: usize = 0;
    while (i + 4 * f16_bridge_vector_len <= src.len) : (i += 4 * f16_bridge_vector_len) {
        dst[i..][0..f16_bridge_vector_len].* = @as(Vf32Bridge, @floatCast(@as(Vf16Bridge, src[i..][0..f16_bridge_vector_len].*)));
        dst[i + f16_bridge_vector_len ..][0..f16_bridge_vector_len].* = @as(Vf32Bridge, @floatCast(@as(Vf16Bridge, src[i + f16_bridge_vector_len ..][0..f16_bridge_vector_len].*)));
        dst[i + 2 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].* = @as(Vf32Bridge, @floatCast(@as(Vf16Bridge, src[i + 2 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].*)));
        dst[i + 3 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].* = @as(Vf32Bridge, @floatCast(@as(Vf16Bridge, src[i + 3 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].*)));
    }
    while (i + f16_bridge_vector_len <= src.len) : (i += f16_bridge_vector_len) {
        dst[i..][0..f16_bridge_vector_len].* = @as(Vf32Bridge, @floatCast(@as(Vf16Bridge, src[i..][0..f16_bridge_vector_len].*)));
    }
    while (i < src.len) : (i += 1) {
        dst[i] = @floatCast(src[i]);
    }
}

// bf16 is stored as raw u16 bits, so widening is bit manipulation rather
// than a float cast. Reuse the canonical scalar converter so rounding matches
// the rest of the runtime; the simple loop auto-vectorizes in release.
pub fn widenBf16ToF32(dst: []f32, src: []const u16) void {
    for (dst, src) |*d, s| d.* = dtype_mod.bf16ToF32(s);
}

test {
    _ = @import("packed_tests.zig");
}
