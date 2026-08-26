//! Null GPU provider (`-Dgpu=none`, the default): the full provider
//! interface with `enabled = false` and every capability flag false, so
//! every call site comptime-elides past it and the build analyzes no GPU
//! shim, no kernel source, and no target-specific library. Functions that
//! remain reachable behind runtime conditions return the "not available"
//! value their callers already handle (false, null, no-op); the format
//! tag set is empty in effect, so format-keyed dispatch never survives
//! comptime resolution.
//!
//! This file is also the interface's reference statement: `assertConforms`
//! (`gpu_provider.zig`) checks a real provider against the public
//! declarations HERE, so adding an entry means adding its stub, and a
//! stray public declaration would become a requirement on every provider.
const gpu_provider = @import("gpu_provider.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");

const Tensor = tensor.Tensor;
const TensorF16 = tensor.TensorOf(.f16);
const TensorBf16 = tensor.TensorOf(.bf16);

pub const enabled = false;
pub const has_quant_gemm = false;
pub const has_attention_fwd = false;

// Shared wire and request types (identical type identity per the provider
// interface).
pub const Orient = gpu_provider.Orient;
pub const QMMTile = gpu_provider.QMMTile;
pub const QMoeStage = gpu_provider.QMoeStage;
pub const FlatDType = gpu_provider.FlatDType;
pub const GemmRequest = gpu_provider.GemmRequest;
pub const QuantGemmRequest = gpu_provider.QuantGemmRequest;
pub const AttentionRequest = gpu_provider.AttentionRequest;

// The staging-panel locks exist for interface conformance; nothing stages.
pub var f16_lock: thread.Mutex = .{};
pub var qmoe_lock: thread.Mutex = .{};

/// No kernels exist: every format answers null, so format-keyed call
/// sites comptime-elide (`offload.supportsQuant` is false for all).
pub fn abiValue(comptime fmt: gpu_provider.QuantFormat) ?c_int {
    _ = fmt;
    return null;
}

// Dispatch tracing: off.
pub fn traceEnabled() bool {
    return false;
}
pub fn traceReset() void {}
pub fn traceDump() void {}
pub fn deviceName() ?[]const u8 {
    return null;
}

// Work gates: nothing goes to the GPU.
pub fn shouldUseGpu(_: usize, _: usize, _: usize) bool {
    return false;
}
pub fn shouldUseGpuBatched(_: usize, _: usize, _: usize, _: usize) bool {
    return false;
}
pub fn shouldUseGpuF16(_: usize, _: usize, _: usize) bool {
    return false;
}
pub fn shouldUseGpuF16ForRhs(_: *const TensorF16, _: usize, _: usize, _: usize) bool {
    return false;
}
pub fn shouldUseGpuBf16ForRhs(_: *const TensorBf16, _: usize, _: usize, _: usize) bool {
    return false;
}
pub fn shouldUseGpuGemv(_: *const Tensor, _: usize, _: usize, _: usize) bool {
    return false;
}
pub fn shouldUseGpuForRhs(_: *const Tensor, _: usize, _: usize, _: usize) bool {
    return false;
}
pub fn shouldUseGpuBatchedForRhs(_: *const Tensor, _: usize, _: usize, _: usize, _: usize) bool {
    return false;
}
pub fn shouldUseGpuAttentionFwd(_: usize, _: usize, _: usize, _: usize) bool {
    return false;
}
pub fn shouldUseGpuQMoe(_: u64) bool {
    return false;
}
pub fn qmoeFillAcceptable(_: usize, _: usize) bool {
    return false;
}
pub fn shouldUseGpuDenseQuant(_: gpu_provider.QuantFormat, _: u64) bool {
    return false;
}
pub fn shouldUseGpuDenseQuantPacked(_: gpu_provider.QuantFormat, _: u64) bool {
    return false;
}
pub fn shouldUseGpuQuantDecode(_: gpu_provider.QuantFormat, _: usize, _: usize, _: usize) bool {
    return false;
}
pub fn shouldUseGpuAttn(_: usize, _: usize, _: usize, _: usize) bool {
    return false;
}

// Conformance-suite seams: no device, nothing to save or restore.
pub fn deviceAvailableForTest() bool {
    return false;
}
pub fn setMinWorkQMoeForTest(_: u64) void {}
pub fn qmoeMinFillForTest() u64 {
    return 0;
}
pub fn setQmoeMinFillForTest(_: u64) void {}

// Attention forward: never handled; callers keep their CPU path.
pub fn attentionFwdF32(_: []const f32, _: []const f32, _: []const f32, _: []f32, _: ?[]f32, _: gpu_provider.AttentionRequest) bool {
    return false;
}
pub fn attentionFwdF16Kv(_: []const f32, _: []const f16, _: []const f16, _: []f32, _: ?[]f32, _: gpu_provider.AttentionRequest) bool {
    return false;
}
pub fn attnPrefillF16(_: []const f32, _: []const f16, _: []const f16, _: []f32, _: []const i32, _: gpu_provider.AttentionRequest) bool {
    return false;
}

// Dense GEMM: never handled. The blocking f32 entry serves plain and
// strided-batched calls through one request; the async twins keep their
// tensor-operand contract (they attach completion tokens to storage).
pub fn gemmF16Nt(_: []const f16, _: []const f16, _: usize, _: usize, _: usize, _: bool) ?[]const f16 {
    return null;
}
pub fn gemmF32(_: []const f32, _: []const f32, _: []f32, _: gpu_provider.GemmRequest) bool {
    return false;
}
pub fn gemmBatchedF32Async(_: Orient, _: *const Tensor, _: *const Tensor, _: *Tensor, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize) bool {
    return false;
}
pub fn gemmF32Async(_: Orient, _: *const Tensor, _: *const Tensor, _: *Tensor, _: usize, _: usize, _: usize) bool {
    return false;
}
pub fn gemmF16NtAsync(_: *const TensorF16, _: *const TensorF16, _: *Tensor, _: usize, _: usize, _: usize) bool {
    return false;
}
pub fn gemmBf16NtAsync(_: *const Tensor, _: *const TensorBf16, _: *Tensor, _: usize, _: usize, _: usize) bool {
    return false;
}

// Quantized GEMM: never handled.
pub fn gemmQuantNtAsync(_: gpu_provider.QuantGemmRequest, _: *const Tensor, _: *Tensor) bool {
    return false;
}
pub fn gemmQuantNt(_: gpu_provider.QuantGemmRequest, _: []const f32, _: []f32) bool {
    return false;
}
pub fn gemmQuantNtSharedABatch(_: gpu_provider.QuantGemmRequest, _: []const f32, _: []f32) bool {
    return false;
}
pub fn gemmQGroupedNt(_: gpu_provider.QuantGemmRequest, _: []const QMMTile) bool {
    return false;
}
pub fn qmoeStage(_: usize, _: usize) ?QMoeStage {
    return null;
}

// Device-owned bytes: no device.
pub fn allocResidentBytes(_: usize) ?[]u8 {
    return null;
}
pub fn freeResidentBytes(_: []const u8) void {}

// Evolution-strategies device kernels: never handled.
pub fn flatPerturb(_: FlatDType, _: []u8, _: u64, _: f32, _: usize) bool {
    return false;
}
pub fn flatWeightedUpdate(_: FlatDType, _: []u8, _: []const u64, _: []const f32, _: f32, _: usize) bool {
    return false;
}
pub fn flatAnchorDecay(_: FlatDType, _: []u8, _: []const u8, _: f32, _: bool, _: usize) bool {
    return false;
}
