//! Shared core for the vector kernel family: the symbols every
//! `vector/<concern>.zig` section depends on — `ParallelConfig`, the `V*`
//! vector-width aliases, the thread-count gates, and the contiguous-data
//! accessors. A true leaf (imports only the runtime primitives, never a sibling
//! kernel or the `vector.zig` barrel), so the children depend on this directly
//! instead of importing the parent barrel — breaking the parent<->child cycle.
//! `vector.zig` re-exports these so `vector.<sym>` is unchanged.

const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const parallel = @import("../../parallel.zig");
const tensor = @import("../../tensor.zig");
const thread = @import("../../thread.zig");

const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

pub const ParallelConfig = struct {
    pool: ?*thread.Pool = null,
};

// Architecture-appropriate vector width. Falls back to 4 (a safe minimum on
// any SIMD-capable target) if the compiler can't infer a better one.
pub const vector_len: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;
pub const Vf32 = @Vector(vector_len, f32);
pub const vector_len_f64: comptime_int = std.simd.suggestVectorLength(f64) orelse 2;
pub const Vf64 = @Vector(vector_len_f64, f64);
pub const vector_len_f16: comptime_int = std.simd.suggestVectorLength(f16) orelse 8;
pub const Vf16 = @Vector(vector_len_f16, f16);
pub const Vf32ForF16 = @Vector(vector_len_f16, f32);
pub const Vf16ForF32 = @Vector(vector_len, f16);
pub const Vu16ForF16 = @Vector(vector_len_f16, u16);
pub const Vu32ForF16 = @Vector(vector_len_f16, u32);
pub const Vu16ForF32 = @Vector(vector_len, u16);
pub const Vu32ForF32 = @Vector(vector_len, u32);

// ---------------- Shared thread-count gates ----------------

// Like columnThreadCount but with the base matmul work gate (not the 8x column
// multiplier): decode-sized int8 GEMVs are worth splitting across cores.
pub fn i8ColumnThreadCount(m: usize, n: usize, k: usize) usize {
    if (m == 0 or n < parallel.vector_column_min_n or k == 0) return 1;
    const work = parallel.saturatedMul3(m, n, k);
    if (work < parallel.vector_matmul_work_threshold) return 1;
    const cpu_count = parallel.cpuThreadCount(parallel.vector_max_threads);
    return @max(@as(usize, 1), @min(@min(cpu_count, n / parallel.vector_column_chunk), n));
}

pub fn elementwiseThreadCount(len: usize) usize {
    if (len < parallel.vector_elementwise_len_threshold) return 1;
    const cpu_count = parallel.cpuThreadCount(parallel.vector_max_threads);
    return @max(@as(usize, 1), @min(cpu_count, len / parallel.vector_elementwise_len_threshold + 1));
}

pub fn matmulThreadCount(m: usize, n: usize, k: usize, threshold: usize) usize {
    if (m < parallel.vector_column_min_m or n == 0 or k == 0) return 1;
    const work = parallel.saturatedMul3(m, n, k);
    if (work < threshold) return 1;
    return @max(@as(usize, 1), @min(parallel.cpuThreadCount(parallel.vector_max_threads), m));
}

pub fn batchedThreadCount(batch_count: usize, m: usize, n: usize, k: usize) usize {
    if (batch_count < 2 or m == 0 or n == 0 or k == 0) return 1;
    const work = parallel.saturatedMul3(batch_count, parallel.saturatedMul3(m, n, 1), k);
    if (work < parallel.vector_batched_work_threshold) return 1;
    return @max(@as(usize, 1), @min(parallel.cpuThreadCount(parallel.vector_max_threads), batch_count));
}

pub fn depthwiseConvThreadCount(seq: usize, channels: usize, taps: usize) usize {
    if (seq == 0 or channels == 0 or taps == 0) return 1;
    const work = parallel.saturatedMul3(seq, channels, taps);
    if (work < parallel.vector_elementwise_len_threshold) return 1;
    const cpu_count = parallel.cpuThreadCount(parallel.vector_max_threads);
    return @max(@as(usize, 1), @min(@min(cpu_count, channels), channels / @min(channels, vector_len) + 1));
}

/// Thread count for the general causal conv kernels: `split` is the
/// parallel-split extent (time rows for forward/backward-input, `taps*in`
/// weight rows for backward-weight), `work` the total multiply count.
pub fn generalConvThreadCount(split: usize, work: usize) usize {
    if (split == 0) return 1;
    if (work < parallel.vector_elementwise_len_threshold) return 1;
    return @max(@as(usize, 1), @min(parallel.cpuThreadCount(parallel.vector_max_threads), split));
}

pub fn columnThreadCount(m: usize, n: usize, k: usize) usize {
    if (m == 0 or n < parallel.vector_column_min_n or k == 0) return 1;
    const work = parallel.saturatedMul3(m, n, k);
    if (work < parallel.vector_column_work_multiplier * parallel.vector_matmul_work_threshold) return 1;
    const cpu_count = parallel.cpuThreadCount(parallel.vector_max_threads);
    return @max(@as(usize, 1), @min(@min(cpu_count, n / parallel.vector_column_chunk), n));
}

// ---------------- Shared contiguous-data accessors ----------------

pub fn contiguousDataConstOf(comptime dtype: DType, x: *const tensor.TensorOf(dtype), len: usize) []const dtype_mod.Scalar(dtype) {
    return x.buffer.data[x.offset .. x.offset + len];
}

pub fn contiguousDataOf(comptime dtype: DType, x: *tensor.TensorOf(dtype), len: usize) []dtype_mod.Scalar(dtype) {
    return x.buffer.data[x.offset .. x.offset + len];
}

pub fn contiguousDataConst(x: *const Tensor, len: usize) []const f32 {
    return x.buffer.data[x.offset .. x.offset + len];
}

pub fn contiguousData(x: *Tensor, len: usize) []f32 {
    return x.buffer.data[x.offset .. x.offset + len];
}
