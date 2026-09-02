//! Pure-Zig vector kernels using @Vector intrinsics. Portable across NEON
//! (Apple Silicon, ARM64), AVX2 / AVX-512 (x86_64), WASM SIMD, and any target
//! with vector support. Vector width is chosen by std.simd.suggestVectorLength
//! so the same source compiles to 4-wide on NEON, 8-wide on AVX2, 16-wide on
//! AVX-512.
//!
//! These are implementation details of the native backend: non-GEMM ops always
//! use them, and GEMM falls back to them when no platform BLAS is selected.
//! Every pool-taking kernel takes `pc: ParallelConfig` first.
//!
//! Child modules, each addressed as `vector.<child>.<fn>`:
//!   common       - ParallelConfig, the V* vector-width aliases, the
//!                  thread-count gates and the contiguous-data accessors.
//!   tile         - the payload-generic range splitter (forRange /
//!                  reduceRange) behind the kernels' parallel dispatch.
//!   primitives   - the @Vector cores: dot/add/mul, transcendental bodies,
//!                  the typed f64/f16/bf16 tails, the f16 row-block attention
//!                  inner loops.
//!   elementwise  - elementwise, reduction, activation and norm kernels with
//!                  their parallel dispatch.
//!   gemm         - dense f32/f16/f64/bf16 GEMM (NN/TN/NT) and the row paths.
//!   gemm_blocked - the cache-blocked packed f32 GEMM.
//!   gemm_packed  - the register-tiled GEMM over load-time f32 RHS panels.
//!   batched      - batched dense GEMM.
//!   matmul_quant - quantized matmul dispatch over the `quant/` kernels.
//!   conv         - causal, depthwise, grouped and channel-last 2-D convolution.
//!   pool         - channel-last 2-D pooling and nearest upsampling.
//!   winograd     - the Winograd F(2x2,3x3) and F(4x4,3x3) transforms.
//!   rows         - the fused row kernels (softmax/logsumexp, layer/RMS
//!                  norm, cross-entropy, dropout, scatter-add, gated
//!                  activations, the inner-lane strided-axis family) with
//!                  their Task payloads and run adapters.
//!   attention    - the grouped-causal attention kernels (per-query units,
//!                  query-tiled online-softmax prefill, tiled backward +
//!                  BLAS strips, multi-stream decode) with their Task
//!                  payloads and run adapters.
pub const common = @import("vector/common.zig");
pub const tile = @import("vector/tile.zig");

pub const primitives = @import("vector/primitives.zig");
pub const elementwise = @import("vector/elementwise.zig");
pub const gemm = @import("vector/gemm.zig");
pub const gemm_blocked = @import("vector/gemm_blocked.zig");
pub const gemm_packed = @import("vector/gemm_packed.zig");
pub const batched = @import("vector/batched.zig");
pub const matmul_quant = @import("vector/matmul_quant.zig");
pub const conv = @import("vector/conv.zig");
pub const pool = @import("vector/pool.zig");
pub const winograd = @import("vector/winograd.zig");
pub const rows = @import("vector/rows.zig");
pub const attention = @import("vector/attention.zig");

pub const ParallelConfig = common.ParallelConfig;
pub const Vf32 = common.Vf32;
pub const vector_len = common.vector_len;

test {
    _ = @import("vector/elementwise.zig");
    _ = @import("vector/gemm.zig");
    _ = @import("vector/gemm_blocked.zig");
    _ = @import("vector/gemm_packed.zig");
    _ = @import("vector/matmul_quant.zig");
    _ = @import("vector/batched.zig");
    _ = @import("vector/conv.zig");
    _ = @import("vector/pool.zig");
    _ = @import("vector/winograd.zig");
    _ = @import("vector/primitives.zig");
}
