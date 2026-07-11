const std = @import("std");

const dtype = @import("dtype.zig");
const storage = @import("storage.zig");
const tensor = @import("tensor.zig");
const backend = @import("backend.zig");
const exec = @import("exec.zig");
const tagged = @import("tagged.zig");
const ag = @import("ag.zig");
const param_registry_mod = @import("param_registry.zig");
const state_dict_mod = @import("state_dict.zig");
const safetensors_mod = @import("safetensors.zig");
const training_checkpoint_mod = @import("training_checkpoint.zig");
const thread = @import("thread.zig");
pub const es = @import("es.zig");
pub const gguf = @import("gguf.zig");
pub const lora = @import("lora.zig");
pub const optim = @import("optim.zig");
pub const ptqtp = @import("ptqtp.zig");
pub const rng = @import("rng.zig");
pub const parallel = @import("parallel.zig");
pub const ParamRegistry = param_registry_mod.ParamRegistry;
pub const state_dict = state_dict_mod;
pub const safetensors = safetensors_mod;
pub const training_checkpoint = training_checkpoint_mod;

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
pub const einsumMany = ag.einsumMany;
pub const checkpoint = ag.checkpoint;
pub const checkpointWithContext = ag.checkpointWithContext;
pub const noGrad = ag.noGrad;
pub const isGradEnabled = ag.isGradEnabled;
pub const NoGradScope = ag.NoGradScope;
pub const customVjp = ag.customVjp;
pub const gradcheck = ag.gradcheck;
pub const GradcheckOptions = ag.GradcheckOptions;
pub const GradcheckResult = ag.GradcheckResult;
pub const DType = dtype.DType;
/// bf16 <-> f32 scalar converters (bf16 tensors store raw u16 bits): the
/// bridge for consumers of bf16 state dicts and 16-bit params.
pub const bf16ToF32 = dtype.bf16ToF32;
pub const f32ToBf16 = dtype.f32ToBf16;
pub const supports_q4_k_mmla = backend.supports_q4_k_mmla;
pub const PackedRhs = ag.PackedRhs;
pub const PackedRhsLayout = backend.PackedRhsLayout;
pub const SliceRange = ag.SliceRange;
pub const PreparedConvWeights = exec.ExecContext.PreparedConvWeights;
pub const BlockQ1_0 = dtype.BlockQ1_0;
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
pub const Backend = backend.Backend;
pub const BackendKind = backend.Kind;
pub const active_backend_kind = backend.active_kind;
pub const native_blas_kind = backend.native_blas_kind;
pub const native_uses_blas = backend.native_uses_blas;
pub const native_uses_accelerate = backend.native_uses_accelerate;
pub const native_blas_threads = backend.native_blas_threads;
pub const ExecContext = exec.ExecContext;
pub const RhsLifetime = exec.RhsLifetime;
pub const MoeRhs = exec.ExecContext.MoeRhs;
pub const MoeBatchProfile = exec.MoeBatchProfile;
pub const GatedOp = exec.GatedOp;
pub const expert_store = exec.expert_store;
pub const ExpertStore = exec.expert_store.ExpertStore;
pub const RouterTopKOptions = exec.RouterTopKOptions;
pub const StandardizeOptions = exec.StandardizeOptions;
pub const StandardizeAccumulation = exec.StandardizeAccumulation;
pub const StandardizeEpsMode = exec.StandardizeEpsMode;
pub const Reduction = exec.Reduction;
pub const CrossEntropyOptions = exec.CrossEntropyOptions;
pub const UnaryOp = exec.UnaryOp;
pub const RopeMode = exec.RopeMode;
pub const RopeTable = exec.RopeTable;
pub const RopeTheta = exec.RopeTheta;

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
    _ = dtype;
    _ = storage;
    _ = tensor;
    _ = backend;
    _ = exec;
    _ = tagged;
    _ = ag;
    _ = param_registry_mod;
    _ = state_dict_mod;
    _ = safetensors_mod;
    _ = training_checkpoint_mod;
    _ = thread;
    _ = es;
    _ = gguf;
    _ = lora;
    _ = optim;
    _ = ptqtp;
    _ = rng;
    _ = @import("fucina_tests.zig");
}
