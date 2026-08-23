//! The public surface of fucina. Root membership follows a closed rule —
//! a declaration lives here only if it is one of:
//!
//!   1. a pillar type: `Tensor` (the tagged autograd facade), `ExecContext`
//!      (the eager runtime), `DType`, and their option/result types;
//!   2. a graph-control free function with no natural receiver (`noGrad`,
//!      `checkpoint`, `customVjp`, `gradcheck`) or the N-ary fold of a
//!      method (`einsumMany`);
//!   3. a subsystem namespace, one name per band (`weights`, `gguf`,
//!      `optim`, `quant`, `tuning`, `simd`, ...);
//!   4. a backend identity/capability constant (`active_backend_kind`,
//!      `native_*`);
//!   5. a scalar converter bridging storage formats (`bf16ToF32`).
//!
//! Everything else earns a namespace, not a root name. `internal` is the
//! sibling-module seam (exact core type identity for `fucina_llm`), not
//! public API; the raw tensor type is deliberately unexported (see the
//! comptime guard below). Layer stack and band map: docs/ARCHITECTURE.md.

const std = @import("std");

const dtype = @import("dtype.zig");
const storage = @import("storage.zig");
const tensor = @import("tensor.zig");
const backend = @import("backend.zig");
const exec = @import("exec.zig");
const tag_ops = @import("tag_ops.zig");
const ag = @import("ag.zig");
const param_registry_mod = @import("param_registry.zig");
const state_dict_mod = @import("state_dict.zig");
const safetensors_mod = @import("safetensors.zig");
const training_checkpoint_mod = @import("training_checkpoint.zig");
const thread = @import("thread.zig");
/// Evolution strategies (experimental tier): NES/OpenAI-ES optimizers
/// over flat parameter slices.
pub const es = @import("es.zig");
/// GGUF container: mmap/streamed parsing, metadata, tensor infos, the
/// writer, and split-file handling.
pub const gguf = @import("gguf.zig");
/// Model I/O above the container parsers: GGUF tensors → executable
/// quantized weight containers (`LinearWeight`, `MoeRhs`, fusion, mmap
/// borrowing, ExpertStore streaming glue). Model-agnostic — LLM families,
/// vision encoders, and audio models load through the same band.
pub const weights = @import("weights.zig");
/// PTQTP sidecar planes as weight decorations (docs/PTQTP.md).
pub const ptqtp_gguf = @import("ptqtp_gguf.zig");
/// GGUF metadata readers + parallel layer loading shared by model loaders.
pub const gguf_meta = @import("gguf_meta.zig");
/// Streaming causal 1-D convolutions (codec-decoder state discipline).
pub const streamconv = @import("streamconv.zig");
/// LoRA adapters: low-rank deltas over `weights.LinearWeight` plus their
/// train-time plumbing.
pub const lora = @import("lora.zig");
/// Optimizers (SGD/AdamW/Muon/APOLLO...) over parameter registries, with
/// recorded-golden parity tests.
pub const optim = @import("optim.zig");
/// PTQTP ternary-plane quantization (experimental tier): K trit-plane
/// decomposition and its quality stats (docs/PTQTP.md).
pub const ptqtp = @import("ptqtp.zig");
/// Deterministic RNG streams (philox-style splittable seeds) shared by
/// init, dropout, and samplers.
pub const rng = @import("rng.zig");
/// Parallel execution policies over the worker pool: chunked loops,
/// reduction scaffolding, thread-count gates.
pub const parallel = @import("parallel.zig");
/// Tuning policy: the shared shape of every FUCINA_* route gate (read-once
/// env switch + measured default + programmatic `set`), the numeric route
/// defaults, and the per-context `Overrides` (`ExecContext.setTuning`).
pub const tuning = @import("tuning.zig");
/// IEEE floating-point environment: rounding/underflow inquiry and scoped
/// control, plus the accrued exception flags. The bitwise contracts elsewhere
/// in this library assume the default environment; this is how you check it.
pub const fpenv = @import("fpenv.zig");
/// Steady-state caching allocator for training loops (see the module doc:
/// large blocks recycle through power-of-two freelists, never returning to
/// the OS mid-run).
pub const CachingAllocator = @import("caching_allocator.zig").CachingAllocator;
/// Named parameter registry: the bridge between model structs and the
/// optimizers/serializers (registration order is the flat layout).
pub const ParamRegistry = param_registry_mod.ParamRegistry;
/// One registered parameter as `ParamRegistry.view` returns it: name,
/// dtype, shape, mutable byte view, and trainability.
pub const ParamView = param_registry_mod.ParamView;
/// Torch-style state-dict reading (name -> tensor) over safetensors.
pub const state_dict = state_dict_mod;
/// safetensors container: single-file and sharded (`index.json`) reading.
pub const safetensors = safetensors_mod;
/// Resumable training checkpoints (params + optimizer state + RNG).
pub const training_checkpoint = training_checkpoint_mod;

/// THE public tensor: comptime-tagged axes, eager forward, tape-recorded
/// autograd. `Tensor(.{ .seq, .embed })` names a distinct type; methods
/// live on it, documented in docs/REFERENCE.md.
pub const Tensor = ag.Tensor;
// Deliberately NO public `RawTensor` root export. Raw f32 tensors are an INTERNAL
// runtime/backend detail, not a stable public API — the no-grad `Tensor` facade
// has negligible forward overhead, so model/example code carries
// `fucina.Tensor(spec)` end-to-end. In-tree raw naming uses `fucina.internal.RawTensor`;
// microbenchmarks use `bench_raw.RawTensor`.
comptime {
    // Anti-regression guard: re-exporting the raw tensor type at the
    // PUBLIC ROOT is a COMPILE ERROR. This fires on any build that analyzes the
    // module root (every test/example/tool), not just `zig build test`. `internal`
    // and `bench_raw` are unaffected (this only inspects the root's own decls).
    if (@hasDecl(@This(), "RawTensor")) @compileError(
        "fucina.RawTensor must not be exported at the public root; raw tensors are internal. " ++
            "Use fucina.internal.RawTensor (in-tree raw naming) or bench_raw.RawTensor (microbench).",
    );
}
// Graph-control free functions (species 2 of the membership rule).
/// N-ary einsum over a comptime spec string (the fold of the binary `einsum` method).
pub const einsumMany = ag.einsumMany;
/// Gradient checkpointing: recompute this scope on backward instead of storing activations.
pub const checkpoint = ag.checkpoint;
/// `checkpoint` with a caller-supplied context pointer threaded to the recompute closure.
pub const checkpointWithContext = ag.checkpointWithContext;
/// Run a scope with gradient recording off (inference inside a training step).
pub const noGrad = ag.noGrad;
/// Whether gradient recording is currently on for this thread.
pub const isGradEnabled = ag.isGradEnabled;
/// RAII form of `noGrad`: recording off until `deinit`.
pub const NoGradScope = ag.NoGradScope;
/// Register a custom backward (VJP) for a user op.
pub const customVjp = ag.customVjp;
/// Finite-difference gradient checker for custom ops and model code.
pub const gradcheck = ag.gradcheck;
/// Tolerances and probe configuration for `gradcheck`.
pub const GradcheckOptions = ag.GradcheckOptions;
/// Per-parameter outcome of a `gradcheck` run.
pub const GradcheckResult = ag.GradcheckResult;
/// Element dtype enum (f32/f16/bf16/int/bool + every quantized block
/// format); the block-format registry lives in `quant`/`dtype`.
pub const DType = dtype.DType;
/// bf16 <-> f32 scalar converters (bf16 tensors store raw u16 bits): the
/// bridge for consumers of bf16 state dicts and 16-bit params.
pub const bf16ToF32 = dtype.bf16ToF32;
/// f32 -> bf16 scalar converter (round-to-nearest-even; bf16 tensors store raw u16 bits).
pub const f32ToBf16 = dtype.f32ToBf16;
/// Quantized-format vocabulary: the GGML block structs, the packed
/// quantized-matmul RHS container types, block sizes, and the quant
/// capability flags. The comptime completeness registry behind the block
/// set is `dtype.block_formats` (every quantized dtype claims exactly one
/// row or compilation fails).
pub const quant = struct {
    pub const supports_q4_k_mmla = backend.supports_q4_k_mmla;
    pub const BlockQ1_0 = dtype.BlockQ1_0;
    pub const BlockQ2_0 = dtype.BlockQ2_0;
    pub const BlockQ4_0 = dtype.BlockQ4_0;
    pub const BlockQ4_1 = dtype.BlockQ4_1;
    pub const BlockQ5_0 = dtype.BlockQ5_0;
    pub const BlockQ5_1 = dtype.BlockQ5_1;
    pub const BlockQ8_0 = dtype.BlockQ8_0;
    pub const q8_0_block_size = dtype.q8_0_block_size;
    pub const QuantizedMatmulRhsQ8_0x4 = backend.QuantizedMatmulRhsQ8_0x4;
    pub const QuantizedMatmulRhsQ4_Kx4 = backend.QuantizedMatmulRhsQ4_Kx4;
    pub const QuantizedMatmulRhsQ4_Kx8 = backend.QuantizedMatmulRhsQ4_Kx8;
    pub const QuantizedMatmulRhsQ4_Kx2Mmla = backend.QuantizedMatmulRhsQ4_Kx2Mmla;
    pub const QuantizedMatmulRhsQ5_Kx8 = backend.QuantizedMatmulRhsQ5_Kx8;
    pub const QuantizedMatmulRhsQ6_Kx4 = backend.QuantizedMatmulRhsQ6_Kx4;
    pub const QuantizedMatmulRhsQ2_K = backend.QuantizedMatmulRhsQ2_K;
    pub const QuantizedMatmulRhsQ3_K = backend.QuantizedMatmulRhsQ3_K;
    pub const QuantizedMatmulRhsQ4_K = backend.QuantizedMatmulRhsQ4_K;
    pub const QuantizedMatmulRhsQ5_K = backend.QuantizedMatmulRhsQ5_K;
    pub const QuantizedMatmulRhsQ6_K = backend.QuantizedMatmulRhsQ6_K;
    pub const BlockQ8_1 = dtype.BlockQ8_1;
    pub const BlockQ2_K = dtype.BlockQ2_K;
    pub const BlockQ3_K = dtype.BlockQ3_K;
    pub const BlockQ4_K = dtype.BlockQ4_K;
    pub const BlockQ5_K = dtype.BlockQ5_K;
    pub const BlockQ6_K = dtype.BlockQ6_K;
    pub const BlockQ8_K = dtype.BlockQ8_K;
    pub const BlockIQ1_S = dtype.BlockIQ1_S;
    pub const BlockIQ1_M = dtype.BlockIQ1_M;
    pub const BlockIQ2_XXS = dtype.BlockIQ2_XXS;
    pub const BlockIQ2_XS = dtype.BlockIQ2_XS;
    pub const BlockIQ2_S = dtype.BlockIQ2_S;
    pub const BlockIQ3_XXS = dtype.BlockIQ3_XXS;
    pub const BlockIQ3_S = dtype.BlockIQ3_S;
    pub const BlockIQ4_NL = dtype.BlockIQ4_NL;
    pub const BlockIQ4_XS = dtype.BlockIQ4_XS;
    pub const BlockTQ1_0 = dtype.BlockTQ1_0;
    pub const BlockTQ2_0 = dtype.BlockTQ2_0;
    pub const BlockMXFP4 = dtype.BlockMXFP4;
    pub const BlockNVFP4 = dtype.BlockNVFP4;
};

/// Pre-packed dense matmul RHS (weights repacked once at load for the packed GEMM arms).
pub const PackedRhs = ag.PackedRhs;
/// Layout tag carried by `PackedRhs` (which packed GEMM arm the bytes are shaped for).
pub const PackedRhsLayout = backend.PackedRhsLayout;
/// Half-open index range for tensor `slice` calls.
pub const SliceRange = ag.SliceRange;
/// A contiguous run of ABSOLUTE indices along one axis, `[origin, origin+len)`
/// — Fortran's array lower bound as a value. Tensor axes here are 0-origin, so
/// this is how a positional axis's origin travels alongside its length instead
/// of being materialized into an arithmetic array (see `prepareRopeTableRange`).
pub const AxisRange = tensor.AxisRange;
/// Conv weights pre-transformed at load for the streaming conv kernels.
pub const PreparedConvWeights = exec.ExecContext.PreparedConvWeights;
// Backend identity + capability constants (species 4 of the membership
// rule); the selected provider is fixed at build time.
/// The selected kernel provider (comptime; see `active_backend_kind`).
pub const Backend = backend.Backend;
/// Provider identity: `.native` (SIMD/BLAS/GPU seams) or `.scalar` (reference).
pub const BackendKind = backend.Kind;
/// Which provider this build compiled in (`-Dbackend`).
pub const active_backend_kind = backend.active_kind;
/// The native build's BLAS provider (`-Dblas`), `.none` when pure Zig kernels.
pub const native_blas_kind = backend.native_blas_kind;
/// True when a CBLAS provider is linked into the native backend.
pub const native_uses_blas = backend.native_uses_blas;
/// True when the linked BLAS is Apple Accelerate.
pub const native_uses_accelerate = backend.native_uses_accelerate;
/// Thread count requested from an explicit BLAS provider (0 = provider default).
pub const native_blas_threads = backend.native_blas_threads;
/// The eager execution runtime: owns allocation (buffer pool), validation,
/// and kernel dispatch. One per thread of model execution; every Tensor op
/// takes it explicitly.
pub const ExecContext = exec.ExecContext;
/// Caller promise about an RHS pointer's lifetime (gates the stable-RHS caches).
pub const RhsLifetime = exec.RhsLifetime;
/// A MoE expert stack's RHS: resident (borrowed or owned) or streamed through an `ExpertStore`.
pub const MoeRhs = exec.ExecContext.MoeRhs;
/// Per-phase timing counters for the batched MoE paths.
pub const MoeBatchProfile = exec.MoeBatchProfile;
/// Gated-activation selector for the fused gate/up FFN kernels (silu/gelu variants).
pub const GatedOp = exec.GatedOp;
/// Disk-backed MoE expert streaming: the store, its cache tiers, and the ProjSpec plumbing.
pub const expert_store = exec.expert_store;
/// The streaming MoE expert store (LRU RAM cache over disk-resident expert blocks).
pub const ExpertStore = exec.expert_store.ExpertStore;
/// Options for the router top-k kernel (normalization of the selected weights).
pub const RouterTopKOptions = exec.RouterTopKOptions;
/// Options for `standardize` (eps mode, accumulation dtype).
pub const StandardizeOptions = exec.StandardizeOptions;
/// Accumulation precision selector for `standardize`.
pub const StandardizeAccumulation = exec.StandardizeAccumulation;
/// Where eps enters the standardize denominator (inside or outside the sqrt).
pub const StandardizeEpsMode = exec.StandardizeEpsMode;
/// Loss reduction selector (none/mean/sum).
pub const Reduction = exec.Reduction;
/// Options for the cross-entropy ops (ignore index, reduction, label smoothing).
pub const CrossEntropyOptions = exec.CrossEntropyOptions;
/// Elementwise unary op selector shared by the `unary` kernels.
pub const UnaryOp = exec.UnaryOp;
/// RoPE pairing/application mode selector.
pub const RopeMode = exec.RopeMode;
/// Precomputed cos/sin rope table (`prepareRopeTable*`), consumed by the fused rope kernels.
pub const RopeTable = exec.RopeTable;
/// Rope base frequency parameter type for the table builders.
pub const RopeTheta = exec.RopeTheta;

/// SIMD vocabulary for user-defined elemental ops (`elementalUnary` /
/// `elementalBinary` vector bodies): the machine vector type, its width,
/// and the transcendental helpers the built-in kernels use.
pub const simd = struct {
    const vector_common = @import("backend/vector/common.zig");
    const vector_primitives = @import("backend/vector/primitives.zig");
    pub const Vf32 = vector_common.Vf32;
    pub const vector_len = vector_common.vector_len;
    pub const vexpf = vector_primitives.vexpf;
    pub const sigmoidVec = vector_primitives.sigmoidVec;
    pub const tanhVec = vector_primitives.tanhVec;
};
/// Fake-quantization round trips (FP8-E4M3 / FP4-E2M1 microscaling groups,
/// Hadamard rotation, f16 round trip) over host slices (§10.10).
pub const fakequant = exec.fakequant;

/// Internal surface for sibling modules such as `fucina_llm` that need exact
/// core type identity without importing a second copy of backend/exec files.
pub const internal = struct {
    pub const backend_mod = backend;
    pub const tensor_mod = tensor;
    pub const thread_mod = thread;
    /// Internal GPU hooks for model loaders and benchmark instrumentation.
    /// These are deliberately kept out of the public root: users keep ordinary
    /// eager `Tensor` values; residency and tracing are backend-owned details.
    pub const gpu = struct {
        /// True on GPU builds (`-Dgpu=metal` or `-Dgpu=cuda`): GPU GEMM
        /// offload is compiled in. Per-arm capability varies by provider —
        /// see `has_quant_gemm`.
        pub const enabled = backend.gpu_impl.enabled;
        /// True when the provider implements dequant-in-kernel quantized GEMM
        /// (dense + grouped MoE). Loaders that reshape CPU representations
        /// for the GPU quant path key on this, not on `enabled` — a provider
        /// can be enabled while its quantized arms are still CPU-only.
        pub const has_quant_gemm = backend.gpu_impl.has_quant_gemm;
        /// True when this provider additionally implements Q5_K dense/grouped
        /// quantized kernels (CUDA only at present).
        pub const has_q5_k_quant = backend.gpu_impl.has_q5_k_quant;
        pub const has_tq2_0_quant = backend.gpu_impl.has_tq2_0_quant;
        /// Device-owned bytes for GPU-build loaders; see the provider's
        /// `allocResidentBytes` (backend/metal.zig, backend/cuda.zig). Null
        /// when unavailable.
        pub const allocResidentBytes = backend.gpu_impl.allocResidentBytes;
        /// Release bytes returned by `allocResidentBytes`.
        pub const freeResidentBytes = backend.gpu_impl.freeResidentBytes;
        /// Opt-in GPU dispatch tracing (`FUCINA_GPU_TRACE=1`). `traceReset` and
        /// `traceDump` are no-ops when tracing is off, so callers can invoke them
        /// unconditionally; `traceEnabled` is the query.
        pub const traceEnabled = backend.gpu_impl.traceEnabled;
        pub const traceReset = backend.gpu_impl.traceReset;
        pub const traceDump = backend.gpu_impl.traceDump;
    };
    /// Canonical INTERNAL name for the raw, no-grad f32 tensor type. In-tree
    /// code that genuinely needs the raw type — runtime/backend internals,
    /// raw-kernel benchmarks, serialization/format byte work, tests that target
    /// raw runtime behavior — names it here instead of a top-level
    /// `fucina.RawTensor` (which the comptime guard above forbids).
    pub const RawTensor = tensor.Tensor;
};

test {
    _ = streamconv;
    _ = tuning;
    _ = dtype;
    _ = storage;
    _ = tensor;
    _ = backend;
    _ = exec;
    _ = tag_ops;
    _ = ag;
    _ = param_registry_mod;
    _ = state_dict_mod;
    _ = safetensors_mod;
    _ = training_checkpoint_mod;
    _ = thread;
    _ = es;
    _ = gguf;
    _ = weights;
    _ = ptqtp_gguf;
    _ = gguf_meta;
    _ = lora;
    _ = optim;
    _ = ptqtp;
    _ = rng;
    _ = fpenv;
    _ = @import("caching_allocator.zig");
    _ = @import("fucina_tests.zig");
}
