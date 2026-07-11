# Fucina Reference

The detailed reference for the Fucina library: the public API surface, its
exact semantics, and the internal layers needed to extend it. Structure and
rationale live in [ARCHITECTURE.md](ARCHITECTURE.md); this document is the
API-level companion. Every Zig snippet is machine-verified against the tree
(see §1).

## Contents

- [1. Introduction and mental model](#1-introduction-and-mental-model)
  - [1.1 The two modules](#11-the-two-modules)
  - [1.2 Mental model](#12-mental-model)
  - [1.3 Layer stack](#13-layer-stack)
  - [1.4 A first program](#14-a-first-program)
  - [1.5 Stability](#15-stability)
- [2. Toolchain, build, and project wiring](#2-toolchain-build-and-project-wiring)
  - [2.1 Toolchain (`AGENTS.md`, `README.md`)](#21-toolchain-agentsmd-readmemd)
  - [2.2 Build options (`build.zig`)](#22-build-options-buildzig)
  - [2.3 Build steps (`build.zig`)](#23-build-steps-buildzig)
  - [2.4 Module graph and options wiring (`build.zig`)](#24-module-graph-and-options-wiring-buildzig)
  - [2.5 Consuming Fucina from another project](#25-consuming-fucina-from-another-project)
  - [2.6 Runtime environment variables](#26-runtime-environment-variables)
  - [2.7 Test organization (`src/`, `examples/`)](#27-test-organization-src-examples)
  - [2.8 Continuous integration (`.github/workflows/ci.yml`)](#28-continuous-integration-githubworkflowsciyml)
- [3. Tensors: types, construction, and data access](#3-tensors-types-construction-and-data-access)
  - [3.1 The `Tensor(spec)` type constructor (`src/ag/tensor.zig`, `src/tags.zig`)](#31-the-tensorspec-type-constructor-srcagtensorzig-srctagszig)
  - [3.2 The four facade branches (`src/ag/tensor.zig`)](#32-the-four-facade-branches-srcagtensorzig)
  - [3.3 Construction and ownership (`src/ag/tensor.zig`, `src/exec.zig`)](#33-construction-and-ownership-srcagtensorzig-srcexeczig)
  - [3.4 `deinit`, lifetime, and exec scopes (`src/ag/tensor.zig`, `src/tensor.zig`)](#34-deinit-lifetime-and-exec-scopes-srcagtensorzig-srctensorzig)
  - [3.5 Data access (`src/ag/tensor.zig`, `src/tensor.zig`)](#35-data-access-srcagtensorzig-srctensorzig)
  - [3.6 Shape and tag introspection (`src/ag/tensor.zig`, `src/tags.zig`)](#36-shape-and-tag-introspection-srcagtensorzig-srctagszig)
  - [3.7 Views and structural ops (`src/ag/tensor.zig`, `src/tagged.zig`, `src/exec/gather_scatter.zig`)](#37-views-and-structural-ops-srcagtensorzig-srctaggedzig-srcexecgather_scatterzig)
  - [3.8 Casting: `to(dtype)` (`src/ag/tensor.zig`, `src/exec/convert.zig`)](#38-casting-todtype-srcagtensorzig-srcexecconvertzig)
  - [3.9 Gradient accessors (`src/ag/tensor.zig`, `src/ag/core.zig`; mechanics in §5)](#39-gradient-accessors-srcagtensorzig-srcagcorezig-mechanics-in-5)
  - [3.10 Facade surface index](#310-facade-surface-index)
- [4. Tensor operations](#4-tensor-operations)
  - [4.1 The common operation contract (`src/ag/tensor.zig`)](#41-the-common-operation-contract-srcagtensorzig)
  - [4.2 Pointwise binary ops and tag-driven broadcasting (`src/ag/tensor.zig`, `src/tagged.zig`)](#42-pointwise-binary-ops-and-tag-driven-broadcasting-srcagtensorzig-srctaggedzig)
  - [4.3 Scalar variants and in-place/no-grad helpers (`src/ag/tensor.zig`)](#43-scalar-variants-and-in-placeno-grad-helpers-srcagtensorzig)
  - [4.4 Unary ops (`src/ag/tensor.zig`, `src/backend/ops.zig`)](#44-unary-ops-srcagtensorzig-srcbackendopszig)
  - [4.5 Gated activations (`src/ag/tensor.zig`)](#45-gated-activations-srcagtensorzig)
  - [4.6 Masks, comparisons, and conditionals (`src/ag/tensor.zig`)](#46-masks-comparisons-and-conditionals-srcagtensorzig)
  - [4.7 Reductions and scans (`src/ag/tensor.zig`)](#47-reductions-and-scans-srcagtensorzig)
  - [4.8 `dot`: tag-directed contraction (`src/ag/tensor.zig`, `src/tagged.zig`)](#48-dot-tag-directed-contraction-srcagtensorzig-srctaggedzig)
  - [4.9 Explicit matmul, ternary STE, and packed-RHS GEMMs (`src/ag/tensor.zig`)](#49-explicit-matmul-ternary-ste-and-packed-rhs-gemms-srcagtensorzig)
  - [4.10 Softmax family (`src/ag/tensor.zig`, `src/exec/softmax.zig`)](#410-softmax-family-srcagtensorzig-srcexecsoftmaxzig)
  - [4.11 Normalization family (`src/ag/tensor.zig`)](#411-normalization-family-srcagtensorzig)
  - [4.12 Rotary position embedding (`src/ag/tensor.zig`, `src/exec/rope.zig`)](#412-rotary-position-embedding-srcagtensorzig-srcexecropezig)
  - [4.13 Attention (`src/ag/tensor.zig`)](#413-attention-srcagtensorzig)
  - [4.14 Convolution and channel-last vision ops (`src/ag/tensor.zig`)](#414-convolution-and-channel-last-vision-ops-srcagtensorzig)
  - [4.15 Losses and similarity (`src/ag/tensor.zig`, `src/exec/loss.zig`)](#415-losses-and-similarity-srcagtensorzig-srcexeclosszig)
  - [4.16 Selection: argmax, topK, sort, routerTopK (`src/ag/tensor.zig`, `src/exec/topk.zig`)](#416-selection-argmax-topk-sort-routertopk-srcagtensorzig-srcexectopkzig)
  - [4.17 Indexing, assembly, and functional updates (`src/ag/tensor.zig`)](#417-indexing-assembly-and-functional-updates-srcagtensorzig)
  - [4.18 MoE facade entries (`src/exec/moe.zig`, `src/exec.zig`)](#418-moe-facade-entries-srcexecmoezig-srcexeczig)
  - [4.19 Math on non-f32 tensors (`src/ag/tensor.zig`)](#419-math-on-non-f32-tensors-srcagtensorzig)
- [5. Automatic differentiation](#5-automatic-differentiation)
  - [5.1 The gradient model (`src/ag/tensor.zig`, `src/ag/core.zig`)](#51-the-gradient-model-srcagtensorzig-srcagcorezig)
  - [5.2 Running backward (`src/ag/core.zig`)](#52-running-backward-srcagcorezig)
  - [5.3 Reading, seeding, and resetting gradients (`src/ag/tensor.zig`, `src/ag/core.zig`)](#53-reading-seeding-and-resetting-gradients-srcagtensorzig-srcagcorezig)
  - [5.4 noGrad scopes (`src/ag/control.zig`)](#54-nograd-scopes-srcagcontrolzig)
  - [5.5 Activation checkpointing (`src/ag/checkpoint.zig`)](#55-activation-checkpointing-srcagcheckpointzig)
  - [5.6 Custom VJPs (`src/ag/custom.zig`)](#56-custom-vjps-srcagcustomzig)
  - [5.7 Gradient checking (`src/ag/gradcheck.zig`)](#57-gradient-checking-srcaggradcheckzig)
  - [5.8 VJP coverage inventory (`src/ag/backward.zig`)](#58-vjp-coverage-inventory-srcagbackwardzig)
- [6. The execution runtime: ExecContext and the memory model](#6-the-execution-runtime-execcontext-and-the-memory-model)
  - [6.1 ExecContext: role and lifecycle (`src/exec.zig`, `src/exec/runtime.zig`)](#61-execcontext-role-and-lifecycle-srcexeczig-srcexecruntimezig)
  - [6.2 The memory model: who owns an op result (`docs/MEMORY-MODEL.md`)](#62-the-memory-model-who-owns-an-op-result-docsmemory-modelmd)
  - [6.3 Exec scopes: implicit ownership for training (`src/exec.zig`, `src/exec/runtime.zig`)](#63-exec-scopes-implicit-ownership-for-training-srcexeczig-srcexecruntimezig)
  - [6.4 Raw construction and copy helpers on ctx (`src/exec.zig`, `src/exec/runtime.zig`)](#64-raw-construction-and-copy-helpers-on-ctx-srcexeczig-srcexecruntimezig)
  - [6.5 BufferPool: transient reuse and scratch leases (`src/exec/buffer_pool.zig`)](#65-bufferpool-transient-reuse-and-scratch-leases-srcexecbuffer_poolzig)
  - [6.6 The worker team (`src/thread.zig`, `src/parallel.zig`)](#66-the-worker-team-srcthreadzig-srcparallelzig)
  - [6.7 RhsLifetime: address-keyed caching of RHS operands (`src/exec/quant_matmul.zig`)](#67-rhslifetime-address-keyed-caching-of-rhs-operands-srcexecquant_matmulzig)
  - [6.8 Determinism and the RNG contract (`src/rng.zig`)](#68-determinism-and-the-rng-contract-srcrngzig)
  - [6.9 The thread-safety contract](#69-the-thread-safety-contract)
- [7. Named axes: the tag algebra](#7-named-axes-the-tag-algebra)
  - [7.1 Tags, tag specs, and normalization (`src/tags.zig`)](#71-tags-tag-specs-and-normalization-srctagszig)
  - [7.2 Lookup, equality, and constraint helpers (`src/tags.zig`)](#72-lookup-equality-and-constraint-helpers-srctagszig)
  - [7.3 Tuple rewrites and axis maps (`src/tags.zig`)](#73-tuple-rewrites-and-axis-maps-srctagszig)
  - [7.4 Result-tag computation: pointwise and dot (`src/tags.zig`)](#74-result-tag-computation-pointwise-and-dot-srctagszig)
  - [7.5 The op library contract (`src/tagged.zig`)](#75-the-op-library-contract-srctaggedzig)
  - [7.6 Alignment, permutation, and broadcast views (`src/tagged.zig`)](#76-alignment-permutation-and-broadcast-views-srctaggedzig)
  - [7.7 Pointwise and gated broadcasting (`src/tagged.zig`)](#77-pointwise-and-gated-broadcasting-srctaggedzig)
  - [7.8 Split, merge, flatten, and multi-axis reduction (`src/tagged.zig`)](#78-split-merge-flatten-and-multi-axis-reduction-srctaggedzig)
  - [7.9 `taggedDot`: tag-directed contraction and its lowering (`src/tagged.zig`)](#79-taggeddot-tag-directed-contraction-and-its-lowering-srctaggedzig)
  - [7.10 Shared dtype-generic helpers (`src/tagged.zig`)](#710-shared-dtype-generic-helpers-srctaggedzig)
- [8. Data types, storage, and the raw tensor layer (internal)](#8-data-types-storage-and-the-raw-tensor-layer-internal)
  - [8.1 The `DType` enum (`src/dtype.zig`)](#81-the-dtype-enum-srcdtypezig)
  - [8.2 Storage mapping and dtype predicates (`src/dtype.zig`)](#82-storage-mapping-and-dtype-predicates-srcdtypezig)
  - [8.3 Float compute/output dtype policy (`src/dtype.zig`)](#83-float-computeoutput-dtype-policy-srcdtypezig)
  - [8.4 Refcounted storage: `BufferOf(dtype)` (`src/storage.zig`)](#84-refcounted-storage-bufferofdtype-srcstoragezig)
  - [8.5 The raw tensor: `TensorOf(dtype)` (`src/tensor.zig`)](#85-the-raw-tensor-tensorofdtype-srctensorzig)
  - [8.6 The `fucina.internal` escape hatch (`src/fucina.zig`)](#86-the-fucinainternal-escape-hatch-srcfucinazig)
- [9. Backends: CPU SIMD, BLAS, threading, and GPU offload](#9-backends-cpu-simd-blas-threading-and-gpu-offload)
  - [9.1 Build-time selection and the facade constants (`src/backend.zig`, `build.zig`)](#91-build-time-selection-and-the-facade-constants-srcbackendzig-buildzig)
  - [9.2 The `Backend` struct and the kernel contract (`src/backend.zig`)](#92-the-backend-struct-and-the-kernel-contract-srcbackendzig)
  - [9.3 The scalar backend and the parity contract (`src/backend/cpu.zig`, `src/backend/parity_test.zig`)](#93-the-scalar-backend-and-the-parity-contract-srcbackendcpuzig-srcbackendparity_testzig)
  - [9.4 Native backend: portable `@Vector` kernels (`src/backend/vector/`)](#94-native-backend-portable-vector-kernels-srcbackendvector)
  - [9.5 GEMM: dispatch precedence, BLAS, and the blocked packed kernel (`src/backend/native.zig`, `vector/gemm.zig`, `vector/gemm_blocked.zig`)](#95-gemm-dispatch-precedence-blas-and-the-blocked-packed-kernel-srcbackendnativezig-vectorgemmzig-vectorgemm_blockedzig)
  - [9.6 Convolution, pooling, and image kernels (`vector/conv.zig`, `vector/pool.zig`, `vector/winograd.zig`)](#96-convolution-pooling-and-image-kernels-vectorconvzig-vectorpoolzig-vectorwinogradzig)
  - [9.7 Quantized matmul dispatch, packed RHS, and the int8 dot arms](#97-quantized-matmul-dispatch-packed-rhs-and-the-int8-dot-arms)
  - [9.8 Threading: the worker team (`src/thread.zig`, `src/parallel.zig`)](#98-threading-the-worker-team-srcthreadzig-srcparallelzig)
  - [9.9 GPU offload (`src/backend/gpu.zig`, `metal.zig`, `cuda.zig`)](#99-gpu-offload-srcbackendgpuzig-metalzig-cudazig)
- [10. Quantization](#10-quantization)
  - [10.1 Format inventory (`src/backend/quant/types.zig`, `src/dtype.zig`)](#101-format-inventory-srcbackendquanttypeszig-srcdtypezig)
  - [10.2 The block-quantized public tensor (`src/ag/tensor.zig`)](#102-the-block-quantized-public-tensor-srcagtensorzig)
  - [10.3 RHS containers and packed layouts (`src/backend/quant/types.zig`, `src/exec/quant_matmul.zig`)](#103-rhs-containers-and-packed-layouts-srcbackendquanttypeszig-srcexecquant_matmulzig)
  - [10.4 `RhsLifetime` and the address-keyed caching rule (`src/exec/quant_matmul.zig`)](#104-rhslifetime-and-the-address-keyed-caching-rule-srcexecquant_matmulzig)
  - [10.5 LHS activation quantization (`src/backend/quant/q8k.zig`, `src/backend/cpu.zig`, `src/backend/native.zig`)](#105-lhs-activation-quantization-srcbackendquantq8kzig-srcbackendcpuzig-srcbackendnativezig)
  - [10.6 Encoders, `gguf.encodeF32`, and ggml parity (`src/backend/quant.zig`, `src/gguf.zig`)](#106-encoders-ggufencodef32-and-ggml-parity-srcbackendquantzig-srcggufzig)
  - [10.7 Ternary: TQ2_0 first-class, TQ1_0 decode-only (`src/backend/quant/ternary.zig`, TERNARY.md)](#107-ternary-tq2_0-first-class-tq1_0-decode-only-srcbackendquantternaryzig-ternarymd)
  - [10.8 Cold decode rules: IQ*, FP4, and friends (`src/backend/quant/cold.zig`, `src/backend/quant_tables.zig`)](#108-cold-decode-rules-iq-fp4-and-friends-srcbackendquantcoldzig-srcbackendquant_tableszig)
  - [10.9 PTQTP: multi-plane ternary decomposition (`src/ptqtp.zig`, PTQTP.md)](#109-ptqtp-multi-plane-ternary-decomposition-srcptqtpzig-ptqtpmd)
- [11. Training: optimizers, evolution strategies, LoRA, and checkpoints](#11-training-optimizers-evolution-strategies-lora-and-checkpoints)
  - [11.1 The shape of a training step](#111-the-shape-of-a-training-step)
  - [11.2 Optimizers (`src/optim.zig`)](#112-optimizers-srcoptimzig)
  - [11.3 Param groups: `OptimizerSet` (`src/optim.zig`)](#113-param-groups-optimizerset-srcoptimzig)
  - [11.4 Gradient clipping and LR schedules (`src/optim.zig`)](#114-gradient-clipping-and-lr-schedules-srcoptimzig)
  - [11.5 Optimizer-state persistence: FZT1 snapshots vs named state dicts (`src/optim.zig`)](#115-optimizer-state-persistence-fzt1-snapshots-vs-named-state-dicts-srcoptimzig)
  - [11.6 `ParamRegistry` (`src/param_registry.zig`)](#116-paramregistry-srcparam_registryzig)
  - [11.7 State dicts (`src/state_dict.zig`)](#117-state-dicts-srcstate_dictzig)
  - [11.8 safetensors read/write surface (`src/safetensors.zig`)](#118-safetensors-readwrite-surface-srcsafetensorszig)
  - [11.9 Checkpoint directories (`src/training_checkpoint.zig`)](#119-checkpoint-directories-srctraining_checkpointzig)
  - [11.10 LoRA adapters (`src/lora.zig`)](#1110-lora-adapters-srclorazig)
  - [11.11 Evolution strategies (`src/es.zig`)](#1111-evolution-strategies-srceszig)
- [12. Model I/O: GGUF and safetensors](#12-model-io-gguf-and-safetensors)
  - [12.1 GGUF reader (`src/gguf.zig`)](#121-gguf-reader-srcggufzig)
  - [12.2 GGUF writer (`src/gguf.zig`)](#122-gguf-writer-srcggufzig)
  - [12.3 The f32 transcode seam: `encodeF32` / `decodeF32` (`src/gguf.zig`)](#123-the-f32-transcode-seam-encodef32--decodef32-srcggufzig)
  - [12.4 The export-gguf tool (`tools/export_gguf.zig`)](#124-the-export-gguf-tool-toolsexport_ggufzig)
  - [12.5 safetensors (`src/safetensors.zig`)](#125-safetensors-srcsafetensorszig)
  - [12.6 Named state dicts (`src/state_dict.zig`)](#126-named-state-dicts-srcstate_dictzig)
  - [12.7 Training-checkpoint directory and native optimizer frames (`src/training_checkpoint.zig`, `src/optim.zig`)](#127-training-checkpoint-directory-and-native-optimizer-frames-srctraining_checkpointzig-srcoptimzig)
- [13. The LLM stack (fucina_llm)](#13-the-llm-stack-fucina_llm)
  - [13.1 Module layout (`src/llm.zig`)](#131-module-layout-srcllmzig)
  - [13.2 Weight loading (`src/llm/weights.zig`)](#132-weight-loading-srcllmweightszig)
  - [13.3 GGUF metadata glue (`src/llm/gguf_meta.zig`)](#133-gguf-metadata-glue-srcllmgguf_metazig)
  - [13.4 KV cache (`src/llm/kv_cache.zig`)](#134-kv-cache-srcllmkv_cachezig)
  - [13.5 Tokenizers](#135-tokenizers)
  - [13.6 Sampling (`src/llm/sampler.zig`)](#136-sampling-srcllmsamplerzig)
  - [13.7 SFT data (`src/llm/data.zig`)](#137-sft-data-srcllmdatazig)
  - [13.8 Chat (`src/llm/chat.zig`)](#138-chat-srcllmchatzig)
  - [13.9 Speculative decoding (`src/llm/speculative/`)](#139-speculative-decoding-srcllmspeculative)
- [14. Model families and example applications](#14-model-families-and-example-applications)
  - [14.1 Conventions shared by every family](#141-conventions-shared-by-every-family)
  - [14.2 Qwen3 — dense and MoE (`src/llm/qwen3/model.zig`)](#142-qwen3--dense-and-moe-srcllmqwen3modelzig)
  - [14.3 Qwen3.5 — Gated-DeltaNet hybrid (`src/llm/qwen35/model.zig`)](#143-qwen35--gated-deltanet-hybrid-srcllmqwen35modelzig)
  - [14.4 Gemma 4 — text + MoE (`src/llm/gemma/`)](#144-gemma-4--text--moe-srcllmgemma)
  - [14.5 DiffusionGemma — block text-diffusion (`src/llm/diffusion_gemma/model.zig`)](#145-diffusiongemma--block-text-diffusion-srcllmdiffusion_gemmamodelzig)
  - [14.6 Parakeet ASR (`src/llm/parakeet/`)](#146-parakeet-asr-srcllmparakeet)
  - [14.7 Example applications](#147-example-applications)
  - [14.8 Example → features → run command](#148-example--features--run-command)

## 1. Introduction and mental model

Fucina is an eager, close-to-metal CPU tensor/autograd runtime plus an
LLM/ASR inference stack, written in Zig 0.16. This document is the detailed
reference for the whole library: the public API surface, its exact semantics
(ownership, errors, defaults, thread-safety), and the internal layers you
need to understand to extend it. The structural overview lives in
[ARCHITECTURE.md](ARCHITECTURE.md); command cheat sheets and per-model
recipes live in `AGENTS.md` and [RUNNING-MODELS.md](RUNNING-MODELS.md).

Every runnable Zig snippet in this document is machine-verified against the tree:
`zig build snippet-check` (§2.7, a CI step) extracts every runnable
snippet written as a named `test` block and runs it against the real
modules; snippets that need model assets are non-test fragments the
harness ignores and are marked `// requires model assets to run`.

### 1.1 The two modules

The build exposes two library modules (§2):

- **`fucina`** (`src/fucina.zig`) — the tensor library: the public `Tensor`
  facade, the `ExecContext` runtime, autograd, quantized formats, training
  (optimizers, ES, LoRA), and persistence (GGUF, safetensors, checkpoints).
- **`fucina_llm`** (`src/llm.zig`) — the model stack built on top: GGUF
  weight binding, KV caches, tokenizers, samplers, chat sessions,
  speculative decoding, and the model families (Qwen3, Qwen3.5, Gemma 4,
  DiffusionGemma, DeepSeek V2/V3, GLM-4.5, DeepSeek V4 Flash,
  Parakeet ASR).

Applications (`examples/`, `tools/`, `bench/`) sit above both.

### 1.2 Mental model

Five ideas carry the whole library:

**Eager, explicit execution.** There is no graph compiler, no lazy
evaluation, no fusion pass. Every tensor operation validates its inputs,
allocates its output through the `ExecContext`, and calls a backend kernel
immediately. What you write is what runs, in the order you wrote it.

**One public tensor, typed at comptime.** `fucina.Tensor(spec)` is the only
user-facing tensor type. Its axis *tags* (names) and rank are part of the
Zig type — checked at compile time — while dimension *sizes* stay runtime
values. The same facade carries no-grad inference and gradient-tracked
training: a tensor with gradient state records backward information, a
constant does not, and the operation call sites are identical. The raw,
untagged tensor underneath is deliberately not exported (§8).

**Tags instead of axis numbers.** Operations name the axes they act on
(`x.dot(&ctx, &w, .in)` contracts the `.in` axis) and broadcasting is
tag-driven: axes align by name, not by position. The tag algebra is
comptime-only data — it compiles down to stride manipulation on the raw
tensor with zero runtime tagging cost (§7).

**Explicit ownership, deterministic cleanup.** Tensors are value handles
over reference-counted buffers. Operations return owned results;
`defer x.deinit()` is the norm. Training loops use exec scopes to own the
flood of intermediates implicitly (§6). Loaded model weights borrow mmap'd
bytes (holder-managed lifetime) or device-resident bytes (freed through
storage release hooks) instead of copying (§8, §12).

**Multi-dtype with a sealed policy.** Tensors span bool/integer dtypes,
`f16`/`bf16`/`f32`/`f64`, and the GGML block-quantized formats. What each
dtype branch can do is enforced by the type system, not runtime checks:
`.f32` is the differentiable branch; other scalar dtypes are constant typed
tensors (floats additionally get forward-only math); block-quantized tensors
are constant inference tensors that dequantize, gather rows, and serve as
matmul right-hand sides (§3, §10). Float compute/output dtypes follow a
fixed per-op-family policy (§8.3).

### 1.3 Layer stack

Top-down; a band depends only on bands at or below it. Acyclicity of the
production import graph is machine-enforced by `zig build arch-check` (§2);
the band *direction* is checked by a development-side dependency-structure
lint whose configuration is not part of this tree (see
[ARCHITECTURE.md](ARCHITECTURE.md)):

| Band | Contents | Reference |
| --- | --- | --- |
| apps | `examples/`, `tools/`, `bench/` | §14 |
| llm | `fucina_llm` module | §13, §14 |
| facade | `src/fucina.zig` public root | §1–§5 |
| autograd + training | `src/ag/`, optim/es/lora/persistence | §5, §11, §12 |
| tagged ops | `src/tagged.zig` | §7 |
| exec runtime | `ExecContext`, `src/exec/` | §6 |
| backends | CPU SIMD, BLAS, Metal/CUDA | §9, §10 |
| tensor/storage/dtype | raw value types | §8 |

### 1.4 A first program

The canonical smoke test — build two variables, contract them, reduce, and
differentiate:

```zig
const std = @import("std");
const fucina = @import("fucina");

test "first program" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // x: [batch=1, in=2], w: [in=2, out=1]
    var x = try fucina.Tensor(.{ .batch, .in }).variable(&ctx, try ctx.fromSlice(&.{ 1, 2 }, &.{ 2, 3 }));
    defer x.deinit();
    var w = try fucina.Tensor(.{ .in, .out }).variable(&ctx, try ctx.fromSlice(&.{ 2, 1 }, &.{ 4, 5 }));
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in); // contract .in => [batch, out]
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?; // dloss/dx = w^T = [4, 5]
    defer gx.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 23.0), try loss.item(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), (try gx.dataConst())[0], 1e-6);
}
```

Everything in this snippet — the context lifecycle, tensor specs,
construction, ownership, tagged contraction, backward — is unpacked in
§3–§6.

### 1.5 Stability

Fucina is a production-oriented core, not a versioned product: there is no
package manifest, no semver contract, and the public API may change (see
*Current Production Gaps* in [ARCHITECTURE.md](ARCHITECTURE.md)). This
reference describes the tree it ships with; sections marked *internal*
(§7 library level, §8, backend internals in §9) document machinery that is
explicitly not a stable API.

## 2. Toolchain, build, and project wiring

### 2.1 Toolchain (`AGENTS.md`, `README.md`)

Fucina is pinned to **Zig 0.16.0** — `zig version` must print `0.16.0`; other
versions do not build. There is no `build.zig.zon` and no package manifest:
every module, executable, and option is wired directly in `build.zig`. There
is also no C/C++ build system — the only non-Zig translation units are a few
vendored shims (`src/backend/metal/shim.m`, the miniaudio/MIDI shims under
`examples/`) compiled by `build.zig` itself when the relevant option or
example requires them. System dependencies appear only when options select
them: a CBLAS provider for `-Dblas=...`, Apple frameworks for
`-Dgpu=metal`/`-Dblas=accelerate` and the audio examples, libc for
`-Dgpu=cuda` (the CUDA driver and cuBLAS are `dlopen`ed at runtime — no CUDA
SDK at build time).

```sh
zig version        # 0.16.0
zig build test     # all test roots; no model assets needed
zig build --help   # lists every step and project option below
```

### 2.2 Build options (`build.zig`)

All project options are consumed at **comptime** through the generated
`build_options` module (§2.4) — backend dispatch is compiled away, and unused
kernel arms are not in the binary.

| Option | Values | Default | Effect | Constraints |
| --- | --- | --- | --- | --- |
| `-Dbackend` | `native` \| `scalar` \| `cpu` | `native` | Kernel implementation set. `native` = Zig SIMD vector kernels + optional BLAS; `scalar` = the reference backend (correctness oracle — native and scalar must agree). | `cpu` is a deprecated alias for `scalar`. |
| `-Dblas` | `none` \| `accelerate` \| `openblas` \| `mkl` \| `blis` \| `nvpl` \| `blas` | `accelerate` on macOS *targets*, `none` elsewhere | CBLAS provider backing the native backend's large-GEMM arms; `none` keeps the pure Zig vector kernels (including the blocked packed f32 GEMM). | `accelerate` on a non-macOS target **panics the build**. |
| `-Daccelerate` | `bool` | unset | Compatibility alias, consulted only when `-Dblas` is absent: `true` → `-Dblas=accelerate`, `false` → `-Dblas=none`. | An explicit `-Dblas` always wins. |
| `-Dblas-threads` | `u32` | `0` | Pins the vendor BLAS thread count for explicit providers (OpenBLAS/MKL/BLIS/NVPL); `0` keeps the provider default. | No effect with `-Dblas=none`. |
| `-Dmax-threads` | `usize` | `8` | Comptime worker-team ceiling **and** runtime default thread count (`src/parallel.zig`). Sized for M1 Max P-cores; many-core servers must raise it at build time (`FUCINA_MAX_THREADS` only lowers it at runtime). | Outside 1–64 **panics the build**. |
| `-Dgpu` | `none` \| `metal` \| `cuda` | `none` | GPU GEMM offload provider (§9). `metal`: big f32/f16 GEMMs, dense quantized prefill linears, and the MoE expert FFN on macOS. `cuda`: the same surface plus fused prefill attention and opt-in decode GEMV on Linux/NVIDIA, no SDK at build time. Decode below the work gates and training stay on CPU. | `metal` on a non-macOS target **panics**; `cuda` on a non-Linux target **panics** (cross-compiling from macOS with `-Dtarget=x86_64-linux-gnu` is the supported path). |
| `-Dparakeet-mic` | `bool` | `false` | Links the vendored miniaudio capture stack into the `parakeet` example so `--mic` (live microphone) works; default off keeps the parakeet build fast. | Only affects the parakeet executable/tests. |
| `-Dllguidance` | `bool` | `false` | Builds the vendored [llguidance](../vendor/llguidance/README.md) constrained-decoding engine (`cargo build` in `vendor/llguidance`) and links its staticlib into the qwen3/gemma4/lmserve examples and the llm, lmserve, and snippet-check test roots, enabling `llm.llguidance` grammar/JSON-schema token masking (§13.6). Off (the default) the build stays pure Zig and `llm.llguidance.Constraint.init` returns `error.LlguidanceNotEnabled`; the `LogitProcessor` seam itself is always available. | Requires a Rust toolchain >= 1.87 on PATH when enabled. |
| `-Dvector-scan` | `bool` | `false` | Vectorizes the scan kernels (`cumsum`/`cumprod` and cumsum's reverse VJP pass). Off = the documented serial-per-row scans. On: non-last-axis scans vectorize across independent columns (bitwise identical to serial); last-axis scans use an in-register prefix scan — still bitwise deterministic for any thread count, but the accumulation order differs from the serial default (the sum-SIMD-lanes rounding class; exact for integer-valued data). Measured M1 ReleaseFast 256×8192: cumsum 3.3×, cumprod 5.2× (last axis), 4.3× (non-last, bit-identical). |
| `-Doptimize` | `Debug` \| `ReleaseSafe` \| `ReleaseFast` \| `ReleaseSmall` | `Debug` | Standard Zig optimize mode. Build with `ReleaseFast` whenever speed matters (Debug is 10–50× slower); validate in Debug/ReleaseSafe, bench in ReleaseFast. | `x86dot-check` is always built ReleaseSafe regardless. |
| `-Dtarget`, `-Dcpu` | standard queries | host, native CPU | Cross-compilation target and CPU model. | See below — a bare `-Dtarget` silently loses the fast kernels. |

Constraint violations are **build-time panics** (`@panic`/`std.debug.panic`
inside `build()`), not recoverable configuration errors: the panic checks run
against the *target* OS (`target.result.os.tag`), not the host, so
`-Dgpu=cuda -Dtarget=x86_64-linux-gnu` from macOS builds fine while
`-Dgpu=cuda` alone on macOS panics.

**CPU targeting is native by default.** With no `-Dtarget`, Zig targets the
compiling machine's exact CPU (full detected feature set, like
`-march=native`), and the kernels' comptime feature gates
(`src/backend/quant/common.zig`) compile in the matching arms — NEON/sdot on
Apple Silicon, AVX2/AVX-VNNI on modern x86, smmla on I8MM-class ARM servers,
portable vectors elsewhere. Unused arms are compiled out entirely; there is
no runtime dispatch. Cross-compiling with `-Dtarget=...` drops to that
architecture's *baseline* unless `-Dcpu=...` names a model (`x86_64_v3`,
`alderlake`, `znver4`, `neoverse_v1`, …). Two rules follow: build on the
machine that will run the binary, or pin `-Dcpu` to match it.

The resolved configuration is visible on the `fucina` module root as
comptime constants (`active_backend_kind`, `native_blas_kind`,
`native_uses_blas`, `native_uses_accelerate`, `native_blas_threads`,
`parallel.vector_max_threads`):

```zig
const std = @import("std");
const fucina = @import("fucina");

test "build options are comptime facts on the module root" {
    // Baked in by build.zig's `build_options`; all comptime-known.
    const kind: fucina.BackendKind = fucina.active_backend_kind; // -Dbackend
    try std.testing.expect(kind == .native or kind == .scalar);
    if (fucina.native_uses_blas) // -Dblas != none
        try std.testing.expect(fucina.native_blas_kind != .none);
    try std.testing.expect(fucina.parallel.vector_max_threads >= 1); // -Dmax-threads
}
```

Note `fucina.BackendKind` has two members (`scalar`, `native`): build.zig
bakes the raw three-member `-Dbackend` value (including the deprecated
`cpu`) into `build_options.backend_kind`, and the `cpu → .scalar` mapping
happens at file scope of `src/backend.zig`. At runtime the effective worker
count never exceeds the comptime ceiling — `fucina.parallel.setMaxThreads(n)`
is the programmatic counterpart of `FUCINA_MAX_THREADS` (mirrors llama.cpp's
`-t`; call once at startup, before the first parallel op — the first
`cpuThreadCount` call latches the value). The two are not identical: the env
var only *lowers* the detected CPU count, while `setMaxThreads` *replaces*
it and can raise the team size above the detected count, up to the ceiling
(§6.6):

```zig
test "runtime worker count never exceeds the comptime ceiling" {
    fucina.parallel.setMaxThreads(4); // programmatic twin of FUCINA_MAX_THREADS
    const n = fucina.parallel.cpuThreadCount(fucina.parallel.vector_max_threads);
    try std.testing.expect(n >= 1 and n <= 4);
    try std.testing.expect(n <= fucina.parallel.vector_max_threads);
}
```

### 2.3 Build steps (`build.zig`)

`zig build` with no step runs the default **install** step: it compiles all
23 installed executables into `zig-out/bin/` (named `fucina-zig-<name>`).
Bench and check executables are *not* installed; they build on demand when
their step runs. Every example-runner step depends only on its own
executable's install-artifact step, so `zig build qwen3` builds just that
executable; among the `bench*` steps only `bench-gate` depends on the full
install step. Arguments after `--` are forwarded to
the launched program.

**Tests and gates:**

| Step | What it does |
| --- | --- |
| `test` | Runs the unit tests of all nine test roots (§2.7). No model assets needed. |
| `test-fucina` | Runs the `fucina`-root unit tests only (the routine `-Dbackend=scalar` leg); the full `test` matrix stays the pre-merge gate. |
| `bench-check` | Compiles every bench executable without running it — the cheap gate that keeps the bench suite building (bench mains are otherwise reachable only through their run steps). |
| `arch-check` | Builds and runs `tools/check_import_graph.zig`: the production (non-test) `src/**/*.zig` import graph must have zero strongly-connected components. AST-based and test-aware — imports reachable only from `test` decls or test-only private helpers are not counted. |
| `doc-check` | Builds and runs `tools/check_doc_links.zig`: every backtick-quoted `*.md` in `AGENTS.md`'s "## Doc index" section (root docs and `docs/<name>.md`) must exist on disk. |
| `snippet-check` | Builds and runs `tools/gen_snippet_tests.zig`: every runnable ```zig snippet in this document (a fenced block with a column-0 named `test "..."`) is extracted into a generated test root and run against the real `fucina`/`fucina_llm` modules with the build's option set — a snippet that stops compiling or asserting fails the gate (conventions in §2.7). |
| `x86dot-check` | Runs the cross-ISA int8/Q4_K/Q8_0/TQ2_0 dot-kernel parity checker (`src/x86dot_check.zig`, always ReleaseSafe, deterministic output diffable across environments). The run leg follows `-Dtarget` (so `-Dtarget=x86_64-macos -Dcpu=baseline` under Rosetta drives the emulated x86 legs); four additional compile-only legs (x86_64_v3, alderlake, znver4, neoverse_v1) catch bit-rot of the AVX2/AVX-VNNI/AVX512-VNNI/smmla inline-asm arms that the local machine cannot execute. |
| `cuda-check` | Compile-only `-Dgpu=cuda` legs: semantically analyzes the `fucina` and `fucina_llm` test roots for `x86_64-linux-gnu` with `gpu_kind=.cuda` (never run), so the CUDA provider cannot bit-rot on GPU-less/macOS machines. |
| `bench-gate` | Runs `python3 tools/bench_gate.py` (a system command, not a Zig artifact): the paired Fucina-vs-llama.cpp benchmark gate; protocol in [`BENCHMARK.md`](BENCHMARK.md). Requires `tools/fetch_refs.sh --build` first. |

**Example and tool runners** (each `zig build <step> -- <args>`; CLI details
in [`RUNNING-MODELS.md`](RUNNING-MODELS.md) and §14):

| Step | Program |
| --- | --- |
| `run` | `examples/smoke.zig` — the smoke example. |
| `qwen3` | Qwen3 dense/MoE GGUF inference: chat/REPL, `--spec`/`--spec-ref` lossless speculative decode, `--tokenize` tokenizer-parity oracle. |
| `gemma4` | Gemma 4 GGUF inference / logit-parity harness; chat/REPL/`--spec`. |
| `qwen35` | Qwen3.5 (hybrid Gated-DeltaNet) GGUF loader/parity harness. |
| `diffusion-gemma` | DiffusionGemma block text-diffusion (parity harness + EB chat). Links libc. |
| `deepseek2` | DeepSeek-V2 family (MLA + MoE) GGUF inference. |
| `glm4moe` | GLM-4.5 family GGUF inference; `--mtp` native multi-token-prediction speculative decode. |
| `deepseek4` | DeepSeek V4 Flash GGUF inference (CSA/HCA + streamed experts). |
| `nanochat` | nanochat port (karpathy/nanochat): tok-train / base-train / sft / eval-bpb / chat. |
| `lmserve` | OpenAI-compatible HTTP server (chat completions + responses; SSE streaming; JSON-schema constrained output with `-Dllguidance=true`) over qwen3/gemma4/diffusion-gemma GGUFs + nanochat checkpoints. Links libc. |
| `parakeet` | Parakeet ASR: WAV → text, `--stream`/`--manifest`/`--mic` (needs `-Dparakeet-mic`), `--compare` parity harness. |
| `omnivoice` | OmniVoice MaskGIT TTS: voice cloning/design, codec encode/decode. |
| `locate-anything` | LocateAnything-3B open-vocabulary detection: detect/info, exit-code parity gates, bench. |
| `facedetect` | buffalo_l face pipeline (SCRFD/ArcFace/genderage/anti-spoof/landmarks): detect/embed/verify/analyze. |
| `spirals` | Two-spirals training demo: SGD/AdamW/Muon/APOLLO, checkpoint, resume, infer. |
| `nam` | Neural Amp Modeler: `.nam` profiles, training, live amp sim (vendored miniaudio + CoreMIDI shims always linked). |
| `finetune` | LoRA fine-tune of a Qwen3 GGUF on a built-in SFT dataset. |
| `es-finetune` | Evolution-strategies fine-tune of a Qwen3 GGUF (`--mode lora\|full`, `--reward rule\|nll\|acc`). |
| `es-spirals` | Two-spirals MLP trained from scratch by ES (self-verifying). |
| `es-ternary-spirals` | Ternary-native ES on packed TQ2_0 layers (training state = the int8 inference model; see [`TERNARY.md`](TERNARY.md)). |
| `ptqtp-spirals` | Self-verifying PTQTP acceptance demo: float-trains an MLP, decorates it post-training with dual trit-planes, asserts accuracy holds on the deployed int8 path (§10.9, [`PTQTP.md`](PTQTP.md)). |
| `ptqtp-qwen3` | Decorate a Qwen3 GGUF's linears in place (any source dtype; `--planes 1\|2\|3`, `--down-planes/--o-planes`, `--skip-first/--skip-last`, `--head-planes`) with teacher-forced NLL before/after and greedy completion + decode timing; `--save FILE` persists the decorated model as a GGUF that reloads bitwise through the ordinary loaders (§13.2.1, [`PTQTP.md`](PTQTP.md)). |
| `export-gguf` | `tools/export_gguf.zig`: GGUF re-emit/transcode (`--dtype f16/bf16/f32/q8_0/q4_k/q5_k/q6_k/tq2_0/verbatim`, `--experts-dtype` override) or merge of Fucina LoRA adapters into dense weights (`--adapters`); see §12. |

**Microbenchmarks** (all in `bench/`; run under `-Doptimize=ReleaseFast`;
protocol and thermal discipline in [`BENCHMARK.md`](BENCHMARK.md)):

| Step | Measures |
| --- | --- |
| `bench` | MLP-shaped inference and backward (`bench/mlp.zig`). |
| `bench-optim` | Optimizer step kernels (SGD/AdamW/Muon/APOLLO) at LLM shapes. |
| `bench-ce` | Softmax / cross-entropy row kernels at LLM shapes. |
| `bench-conv` | conv2d forward/backward-input/backward-weight at CNN shapes. |
| `bench-scatter` | Scatter-add (embedding-gradient) kernel at vocab × dim shapes. |
| `bench-backward-diamond` | Serial vs manual-parallel independent GEMM VJPs. |
| `bench-attention-backward` | Grouped causal attention backward. |
| `bench-backend` | Scalar vs native backends on representative ops. |
| `bench-f16gemm` | f16 TransB GEMM parallel efficiency (Qwen3 shapes). |
| `bench-gemm` | Large-shape f32 GEMM: row kernels vs blocked packed kernel vs BLAS. |
| `bench-gpu-dispatch` | CPU CBLAS vs blocking/async eager GPU GEMM/GEMV: host-visible latency, submit latency, queued throughput, and parity. |
| `bench-gpu-formats` | Fucina f16/load-time-packed quant CPU kernels vs eager GPU f16/Q4_K/Q6_K/Q8_0 LLM linears: host-visible latency, submit latency, queued throughput, and parity. |
| `bench-q5kmoe` | Q5_K MoE-expert matmul variants. |
| `bench-ternary` | TQ2_0 ternary matmul: hot sdot/vpdpbusd tiles vs cold table path, f32 path, Q4_K, dense f32. |
| `bench-facade` | Raw tensor ops vs the public no-grad `Tensor` facade. |
| `bench-einsum` | `einsum` vs hand-written dot/permute contraction pipelines (parity + advantage cases). |

Common invocations:

```sh
zig build test                        # correctness, native backend
zig build test -Dbackend=scalar       # reference backend must agree
zig build test -Dblas=none            # native backend on pure Zig kernels
zig build arch-check doc-check snippet-check  # structure + doc gates
zig build -Doptimize=ReleaseFast      # install everything into zig-out/bin

zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "What is the capital of France?" --no-think

zig build -Dmax-threads=32 -Doptimize=ReleaseFast          # many-core server
zig build qwen3 -Dgpu=metal -Doptimize=ReleaseFast -- ...  # Metal offload (macOS)
zig build -Dgpu=cuda -Dtarget=x86_64-linux-gnu -Dcpu=znver4 \
  -Doptimize=ReleaseFast                                   # CUDA cross-build from macOS

FUCINA_MAX_THREADS=6 zig-out/bin/fucina-zig-qwen3 models/... --chat "..."
```

### 2.4 Module graph and options wiring (`build.zig`)

`build.zig` registers two library modules and two internal microbench roots
with `b.addModule`; executables get private root modules via
`b.createModule` and pull the libraries in with `addImport`.

- **`fucina`** — root `src/fucina.zig`. The public facade: tensors, autograd,
  `ExecContext`, optimizers, ES, LoRA, GGUF/safetensors I/O (§3–§12). It is
  the only one of the two *library* modules that receives the option set:
  `module.addOptions("build_options", options)` (the microbench roots below
  and the test-root module instances receive the same `options` object).
- **`fucina_llm`** — root `src/llm.zig`. The LLM/ASR stack (§13). It does
  *not* get `build_options`; every module built from `src/llm.zig` instead
  receives a single-key `llm_build_options` module (`llguidance: bool`, read
  by `src/llm/llguidance.zig`). It reaches the configured core exclusively
  through `llm_module.addImport("fucina", module)` and the `fucina.internal`
  seam, so there is exactly one copy of the backend/exec types.
- **`bench_raw`** — root `src/bench_raw.zig`, same options. Internal raw
  tensor surface (`RawTensor`, `ExecContext`, `optim`) for
  `bench/{mlp,optim,ce,conv,scatter,backward_diamond,attention_backward,facade,einsum}.zig`.
  Not part of the public facade — the root export guard in `src/fucina.zig`
  makes `fucina.RawTensor` a compile error.
- **`raw_backend`** — root `src/backend.zig`, same options. Direct kernel
  access for `bench/{backend,f16gemm,gemm,q5kmoe,ternary}.zig`. The
  `bench-backend` executable additionally receives a second options module
  named `bench_options` (`native_blas_kind: BlasKind`,
  `native_uses_blas: bool`, `native_blas_threads: u32`) so it can label its
  output with the native backend's BLAS configuration.

The `build_options` module is built with `b.addOptions()` and exactly these
keys (`options.addOption(T, name, value)`):

| Key | Type | Value |
| --- | --- | --- |
| `backend_kind` | `enum { scalar, native, cpu }` | `-Dbackend` |
| `blas_kind` | `enum { none, accelerate, openblas, mkl, blis, nvpl, blas }` | resolved `-Dblas` |
| `use_blas` | `bool` | `blas_kind != .none` |
| `blas_threads` | `u32` | `-Dblas-threads` |
| `max_threads` | `usize` | `-Dmax-threads` |
| `use_gpu` | `bool` | `gpu_kind != .none` |
| `gpu_kind` | `enum { none, metal, cuda }` | `-Dgpu` |
| `vector_scan` | `bool` | `-Dvector-scan` |

Only seven files outside tests import it, all inside the `fucina` module:
`src/parallel.zig`, `src/backend.zig`, `src/backend/native.zig`,
`src/backend/gpu.zig`, `src/backend/metal.zig`, `src/backend/cuda.zig`,
`src/exec/reduce.zig` (a `src/ag/tensor_tests.zig` test also branches on
`vector_scan`). The parakeet executable and
its test root get their *own* single-key `build_options`
(`parakeet_mic: bool`) — the name collides deliberately; the example reads
its key, the library module keeps its full set.

Linking is centralized in four helpers applied per executable:

- `configureBlas(step, blas_kind)` — per provider: link libc plus
  `Accelerate` (framework), `openblas`, `mkl_rt`, `blis`, `nvpl_blas`, or
  generic `blas`, with Homebrew/oneAPI/HPC-SDK library search paths *and*
  rpaths added (`/opt/homebrew/opt/{openblas,blis}`,
  `/usr/local/opt/{openblas,blis}`, `/opt/intel/oneapi/mkl/latest`,
  `/opt/nvidia/hpc_sdk`).
- `configureGpu(b, step, gpu_kind)` — `metal`: link libc + `Metal` +
  `Foundation` and compile `src/backend/metal/shim.m` (`-fobjc-arc`);
  `cuda`: link libc only (the provider `dlopen`s `libcuda.so.1`/cuBLAS via
  `std.DynLib` at runtime).
- `configureLlguidance(step, dep)` — no-op unless `-Dllguidance`; then links
  the cargo-built staticlib plus libc, and on non-macOS targets Zig's
  bundled LLVM libunwind via `link_libcpp` (the Rust FFI converts panics to
  error strings with `catch_unwind`, and glibc does not export
  `_Unwind_*`; macOS's libSystem ships an unwinder).
- `configureNamAudio` / `configureOmnivoiceAudio` / `configureParakeetAudio`
  — the vendored miniaudio C shims (`examples/nam/audio_shim.c`, plus
  `midi_shim.c` for NAM and `examples/omnivoice/play_shim.c` for playback),
  with CoreAudio/CoreMIDI frameworks on macOS; elsewhere miniaudio `dlopen`s
  its backend through libc.

### 2.5 Consuming Fucina from another project

There is no package manifest, so today a consumer vendors the repository
(git submodule, subtree, or plain copy) and wires the modules in its own
`build.zig` with the same `std.Build` calls the in-tree build uses. The
option enums must be re-declared, but only the *field names* matter — the
fucina sources switch on them by name — and all eight keys are required
(compilation of `src/parallel.zig`/`src/backend.zig`/`src/exec/reduce.zig`
fails on a missing key). Keep the two derived booleans consistent with
their enums.

```sh
git submodule add https://github.com/matteo-grella/fucina vendor/fucina
```

```zig
// build.zig (consumer) — verified against Zig 0.16.0
const std = @import("std");

const BackendKind = enum { scalar, native, cpu };
const BlasKind = enum { none, accelerate, openblas, mkl, blis, nvpl, blas };
const GpuKind = enum { none, metal, cuda };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The comptime configuration the fucina sources read as
    // `@import("build_options")`. All eight keys are required.
    const options = b.addOptions();
    options.addOption(BackendKind, "backend_kind", .native);
    options.addOption(BlasKind, "blas_kind", .none);
    options.addOption(bool, "use_blas", false); // keep == (blas_kind != .none)
    options.addOption(u32, "blas_threads", 0);
    options.addOption(usize, "max_threads", 8);
    options.addOption(bool, "use_gpu", false); // keep == (gpu_kind != .none)
    options.addOption(GpuKind, "gpu_kind", .none);
    options.addOption(bool, "vector_scan", false);

    const fucina = b.addModule("fucina", .{
        .root_source_file = b.path("vendor/fucina/src/fucina.zig"),
        .target = target,
        .optimize = optimize,
    });
    fucina.addOptions("build_options", options);

    const fucina_llm = b.addModule("fucina_llm", .{
        .root_source_file = b.path("vendor/fucina/src/llm.zig"),
        .target = target,
        .optimize = optimize,
    });
    fucina_llm.addImport("fucina", fucina);
    // fucina_llm's own comptime configuration, read as
    // `@import("llm_build_options")` — required by every module built from
    // `src/llm.zig` (src/llm/llguidance.zig reads the boolean `llguidance`
    // key; false keeps the engine stubbed). `true` additionally needs the
    // cargo staticlib build + link from fucina's build.zig (§2.2
    // `-Dllguidance`).
    const llm_options = b.addOptions();
    llm_options.addOption(bool, "llguidance", false);
    fucina_llm.addOptions("llm_build_options", llm_options);

    const exe = b.addExecutable(.{
        .name = "myapp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("fucina", fucina);
    exe.root_module.addImport("fucina_llm", fucina_llm);
    // Non-default -Dblas / -Dgpu configurations also need the link steps
    // from fucina's build.zig (configureBlas / configureGpu): frameworks,
    // system libraries, and the Metal shim C source.
    b.installArtifact(exe);
}
```

The application code then imports the modules by the names given to
`addImport`: `const fucina = @import("fucina");` and
`const llm = @import("fucina_llm");`. `fucina_llm` is optional — omit it
(and its `addImport`) for tensor/training-only consumers. For a BLAS or GPU
configuration, replicate the corresponding `configureBlas`/`configureGpu`
body from the in-tree `build.zig` on the consumer executable (the Metal shim
path becomes `vendor/fucina/src/backend/metal/shim.m`). The public API is
not yet stable (`README.md` says so explicitly); pin the vendored commit.

### 2.6 Runtime environment variables

Every knob in the core-runtime and GPU tables is read **once** and cached
(atomically or under a mutex) at first use; changing the process environment
afterwards has no effect. The example/test gates in the last table are plain
`getenv` calls re-read at every use.
Numeric knobs that fail to parse fall back to their defaults;
`FUCINA_MAX_THREADS`-style positive-integer knobs ignore unset/invalid/zero
values. On Linux without libc the lookup scans `/proc/self/environ`
(`src/parallel.zig`), so `FUCINA_MAX_THREADS` also works in static builds.

**Core runtime** (`src/parallel.zig`, `src/exec/conv.zig`,
`src/exec/attention.zig`):

| Variable | Effect | Default |
| --- | --- | --- |
| `FUCINA_MAX_THREADS` | Lowers the worker count below the `-Dmax-threads` ceiling (mirrors llama.cpp `-t`). Never raises it. Consulted on the first `cpuThreadCount` call; a prior `setMaxThreads` wins. | unset (detected CPU count, capped by the ceiling) |
| `FUCINA_SPIN_BUDGET` | Overrides the worker-team spin-then-park window (`src/thread.zig` BarrierPool). Read once per pool init. Workload-coupled; the default is deliberate. | unset (built-in budget) |
| `FUCINA_WINOGRAD=1` / `FUCINA_NO_WINOGRAD=1` | Force the Winograd conv2d route on/off (A/B + emergency revert switches). | on for no-BLAS builds, off when a platform BLAS backs the matmul |
| `FUCINA_NO_WINOGRAD_F4=1` | Pins Winograd-routed large maps to the F(2×2,3×3) tier. | F4 tier enabled |
| `FUCINA_WINOGRAD_F4_MIN` | Minimum output spatial size for the F4 tier. | `14` |
| `FUCINA_WINOGRAD_F4_MAXCIN` | Maximum input channels for the F4 tier (deep-channel maps run faster on F2). | `56` |
| `FUCINA_NO_CONV_BWD_GEMM=1` | Pins the `groups == 1` conv2d backward entries to the direct gather kernels instead of the GEMM (matmul + im2col/col2im) decomposition (A/B + emergency revert switch). | GEMM route on |
| `FUCINA_ATTN_BWD_STATS=1` / `FUCINA_NO_ATTN_BWD_STATS=1` | Force the forward-saved-stats route of the attention-backward softmax reconstruction (`src/exec/attention.zig`) on/off (A/B + emergency revert switches) — the two routes agree to f32 roundoff, not bitwise; only consulted when the autograd record saved forward stats (the stats-less exec path always recomputes). | on |

**GPU offload** (read by both providers unless noted;
`src/backend/metal.zig`, `src/backend/cuda.zig`; see §9):

| Variable | Effect | Default |
| --- | --- | --- |
| `FUCINA_GPU` | Kill switch: a value starting with `0` disables the GPU provider entirely. | enabled on `-Dgpu` builds |
| `FUCINA_GPU_MIN_WORK` | Base f32 GEMM offload gate, in m·n·k work units. | Metal `2^32` (cold single-op crossover); CUDA `2^30` (the transient floor below still dominates ordinary host RHS) |
| `FUCINA_GPU_MIN_WORK_F16` | f16 GEMM gate. | `2^27` (lower — the CPU f16 competitor has no AMX-class arm) |
| `FUCINA_GPU_MIN_WORK_F16_RESIDENT` (cuda) | f16 GEMM/GEMV gate when the RHS already has a device address; permits small-m decode without admitting a streamed weight. | `2^20` |
| `FUCINA_GPU_MIN_WORK_GEMV` | Resident dense-f32 GEMV/small-m GEMM gate (`m <= 8`; nonresident CUDA RHS is refused). | `2^24` |
| `FUCINA_GPU_MIN_WORK_RESIDENT` (cuda) | Dense-f32 GEMM/batched-GEMM gate when the RHS already has a device address. | `2^27` (512³; 256³ loses to OpenBLAS-32 on the reference host) |
| `FUCINA_GPU_MIN_WORK_QMOE` | Grouped quantized MoE GEMM gate; setting it also re-seeds the dense-Q6 gate. | `2^30` |
| `FUCINA_GPU_MIN_WORK_DENSE_Q4` | Dense Q4_K model-weight gate against the load-time-packed CPU fallback. | Metal `2^30`; CUDA `2^27` |
| `FUCINA_GPU_MIN_WORK_DENSE_Q6` | Dense Q6_K gate; overrides both the compact/raw and packed-CPU tiers. | compact/raw `2^22`; packed Metal `2^31`, CUDA `2^24` |
| `FUCINA_GPU_MIN_WORK_DENSE_Q8` | Dense Q8_0 model-weight gate against the load-time-packed CPU fallback. | Metal `2^29`; CUDA `2^24` |
| `FUCINA_GPU_QMOE_MIN_FILL` | Tile-occupancy gate (percent) for grouped MoE: small expert batches whose 32-row tiles would run mostly empty stay on CPU; `0` disables the gate, `>100` never passes it. | `50` |
| `FUCINA_GPU_TRACE` | Non-`0` first character enables dispatch tracing; dump via `fucina.internal.gpu.traceDump()` (no-op when off). | off |
| `FUCINA_GPU_TF32` (cuda) | Non-`0` opts f32 GEMMs into TF32 tensor cores (default is strict FP32). | off |
| `FUCINA_GPU_MIN_WORK_TRANSIENT` (cuda) | Work floor for *non-resident* operands (each crossing PCIe per call); an `m ≥ 128` row floor applies alongside it. | `2^33` |
| `FUCINA_GPU_MIN_WORK_ATTN` (cuda) | Fused prefill-attention gate, in q·kv·heads·d work units. | `2^28` |
| `FUCINA_GPU_DECODE` (cuda) | Non-`0` enables the opt-in dequant-dot decode GEMV (m ≤ 8, resident weights only). | off |
| `FUCINA_GPU_QUANT_MMA` (cuda) | A value starting with `0` disables the tensor-core Q4_K/Q6_K/Q8_0 prefill kernels and selects the scalar-FFMA fallback (diagnostic A/B switch). | enabled on compute capability ≥ 7 |
| `FUCINA_GPU_VRAM_BUDGET` (cuda) | Weight-residency budget in bytes; `0` disables the bound. | 80% of free VRAM at init |
| `FUCINA_GPU_KERNELS=src` (cuda) | NVRTC-recompiles the vendored kernels from `kernels.cu` instead of loading the committed PTX (dev loop; `tools/gen_cuda_ptx.sh` regenerates the PTX). | committed PTX |

**LLM weight loading** (`src/llm/weights.zig`, §13.2; read once and
cached like the tables above):

| Variable | Effect | Default |
| --- | --- | --- |
| `FUCINA_NORM_QUANT_FUSED=1` / `FUCINA_NO_NORM_QUANT_FUSED=1` | Force the fused normalize+quantize+packed-GEMM route of `linearSeqNormed` on/off (prefill shapes on the packed CPU arms only; the fused route matches the unfused `rmsNormMul` + linear pair to f32 roundoff, not bitwise). | on |
| `FUCINA_Q5K_DECODE_COMPACT=1` / `FUCINA_NO_Q5K_DECODE_COMPACT=1` | Route decode-shape (m < 4) no-grad Q5_K matmuls through the GGUF-native compact blocks instead of the byte-expanded packed layout — bitwise-equal, ~1.57× fewer weight bytes streamed. | on on x86_64, off elsewhere |
| `FUCINA_Q6K_DECODE_COMPACT=1` / `FUCINA_NO_Q6K_DECODE_COMPACT=1` | The same switch for Q6_K (1.30× byte ratio). | on on x86_64, off elsewhere |

**Examples and test gates** (`examples/`):

| Variable | Effect | Default |
| --- | --- | --- |
| `FUCINA_NAM_PROFILES` | Profile directory for the `nam` CLI (`--profiles-dir` overrides it). | `nam-profiles` |
| `OMNIVOICE_PARITY=1` | Enables the OmniVoice parity suites under `zig build test` (need model files under `models/omnivoice/` and locally captured reference goldens); unset, they `error.SkipZigTest`. | skipped |
| `OMNIVOICE_AUDIO_DEVICE_TESTS=1` | Enables the speaker-playback device tests. | skipped |
| `OMNIVOICE_TOKENIZER_GGUF=<path>` | Points the real-codec-GGUF load test at a tokenizer GGUF. | skipped |
| `NANOCHAT_PARITY=1` | Enables the nanochat parity suites under `zig build test` (need locally captured reference goldens); unset, they `error.SkipZigTest`. | skipped |
| `FUCINA_TEST_VERBOSE` | Any value re-enables the facedetect/nanochat per-case test-progress prints on stderr (`examples/{facedetect,nanochat}/testlog.zig`); failure-path prints stay on regardless. | silent |

### 2.7 Test organization (`src/`, `examples/`)

Tests live in **sibling `*_tests.zig` files** next to the production file
they cover (143 of them across `src/` and `examples/`): `exec.zig` ↔
`exec_tests.zig`, `src/llm/tokenizer.zig` ↔ `src/llm/tokenizer_tests.zig`,
and so on. The production file pulls its sibling in with a forwarding
stanza, so analyzing the production file analyzes its tests:

```zig
test {
    _ = @import("exec_tests.zig");
}
```

Module roots forward everything: `src/fucina.zig` ends in a `test` block
referencing every submodule (`_ = dtype; _ = exec; …`), and `src/llm.zig`
does the same for every family and helper, so one `addTest` per root
reaches the whole tree.

`zig build test` runs **nine test roots**, each compiled as its own test
binary with the same option set as the corresponding executable:

1. `src/fucina.zig` — the core (with `build_options`);
2. `src/llm.zig` — the LLM/ASR stack (imports `fucina`);
3. `examples/lmserve.zig` (imports `fucina` and `fucina_llm`; links libc);
4. `examples/nam.zig` (with the audio/MIDI shims linked);
5. `examples/parakeet.zig` (with its `parakeet_mic` options);
6. `examples/omnivoice.zig` (with the playback shim);
7. `examples/locate_anything.zig`;
8. `examples/facedetect.zig`;
9. `examples/nanochat.zig` (imports `fucina` and `fucina_llm` — the
   raw-byte BPE pretokenizer reuses the generated Unicode tables via
   `llm.unicode_categories`).

All nine pass with no model assets present. Suites that need external
material skip themselves cleanly rather than fail: the OmniVoice parity
suites gate on `OMNIVOICE_PARITY` (§2.6); asset-dependent tests (facedetect
goldens, the GGUF re-emit byte-identity test, tokenizer-parity fixtures,
NAM training goldens) translate `error.FileNotFound` into
`error.SkipZigTest`; GPU-dependent tests (`src/llm/gemma/moe_tests.zig`)
skip unless the build has a GPU provider *and* a device is actually present.
Tests for **opt-in build features** follow the same discipline through the
feature's comptime flag: every `src/llm/llguidance_tests.zig` case is
guarded on the flag — the enabled-path cases open with
`if (!llm.llguidance.enabled) return error.SkipZigTest;`, and one
disabled-build case inverts the guard to assert `error.LlguidanceNotEnabled`
— so the same test root compiles and passes under any flag combination and
gains coverage — never failures — when the flag is on.
Per `CONTRIBUTING.md`, numeric changes must additionally be green under
`-Dbackend=scalar` and `-Dblas=none` — the scalar backend is the reference,
and native must agree with it.

**Doc snippets are tests too.** `zig build snippet-check` extracts every
runnable ```zig block from this document — any fenced block containing a
column-0 named `test "..."` declaration — into a generated test root and
runs it against the real `fucina`/`fucina_llm` modules with the build's
option set (`tools/gen_snippet_tests.zig`). Authoring contract: snippets
assume an implicit prelude (`std`, `fucina`, `llm = @import("fucina_llm")`,
`optim = fucina.optim`; entries a snippet declares itself are not
re-emitted); a `<!-- snippet: helper -->` comment on the line before a
non-test fence marks a definition block (an Op/Spec/fn the prose
introduces) prepended to every later snippet in the same `## ` chapter; a
`<!-- snippet: skip -->` comment excludes a test-shaped block that cannot
run hermetically. Illustrative fragments (signature blocks, bare `test {`
stanzas, asset-dependent `fn` examples) are ignored automatically. A
snippet for an opt-in build feature stays RUNNABLE, not skip-marked: it
opens with the feature's comptime-flag guard (e.g.
`if (!llm.llguidance.enabled) return error.SkipZigTest;`), so
`snippet-check` compiles it under every flag combination and executes it
exactly when the enabling `-D` flag is passed.

### 2.8 Continuous integration (`.github/workflows/ci.yml`)

CI runs on pushes to `main` and on every pull request, on a two-OS matrix
(`fail-fast: false`): `ubuntu-latest` (x86-64) and `macos-15` (arm64 —
pinned rather than `-latest`, bumped deliberately). Zig 0.16.0 is installed
via `mlugg/setup-zig@v2`. Steps, in order:

1. `zig build test` — native backend (Accelerate on macOS, no BLAS on Linux,
   per the `-Dblas` default);
2. `zig build` — all executables compile;
3. `zig build bench-check` — every bench executable compiles (bench mains
   are reachable only through their run steps, so nothing else in the
   build graph exercises them);
4. `zig build arch-check` — import-graph gate;
5. `zig build doc-check` — doc-index link gate;
6. `zig build snippet-check` — REFERENCE.md runnable-snippet gate (§2.7);
7. `zig build x86dot-check` — dot-kernel parity on the host ISA (x86 on
   ubuntu, NEON/sdot on macOS) plus the compile-only bit-rot legs;
8. `zig build test -Dbackend=scalar` — ubuntu only (the reference backend);
9. `zig build test -Dblas=none` — macOS only (pure-Zig native kernels,
   complementing the Accelerate run in step 1);
10. `zig build test -Dllguidance=true` + `snippet-check -Dllguidance=true` —
    ubuntu only (the runner image ships cargo): un-skips the flag-gated
    llguidance tests and snippets (§2.7), keeping the extern ABI, the cargo
    build, and the Rust-staticlib link from bit-rotting behind a green
    default build — and continuously proving the Linux link of that
    staticlib.

Between the matrix and the conditional legs, every backend combination that
can run on stock CI hardware is covered: native+BLAS, native without BLAS,
and scalar, on both ISAs' unit-test surface, plus the opt-in llguidance
feature on Linux. The CUDA GPU provider is
covered by the compile-only `cuda-check` leg locally (not in CI); CPU dot
ISA arms that CI cannot execute (AVX-VNNI, AVX512-VNNI, smmla) are covered
by the compile-only legs and attestation records in `src/x86dot_check.zig`.

## 3. Tensors: types, construction, and data access

`fucina.Tensor` is the single public tensor type. It is a comptime type
constructor: `Tensor(spec)` returns a struct type whose axis names (tags),
rank, and dtype are fixed at compile time, while axis *sizes* are runtime
values. Every tensor operation goes through an explicit `*ExecContext`
(see §6); there is no global state. The raw, tag-less tensor underneath the
facade is deliberately **not** exported at the public root — a comptime guard
in `src/fucina.zig` makes `fucina.RawTensor` a compile error; in-tree code
that genuinely needs it names `fucina.internal.RawTensor` (§8). The no-grad
facade has negligible forward overhead, so model and example code carries
`fucina.Tensor(spec)` end-to-end.

Sources for this section: `src/ag/tensor.zig` (the facade), `src/tags.zig`
(spec normalization), `src/tensor.zig` (raw storage semantics visible through
the facade), `src/exec.zig` (raw-tensor producers). The ownership discipline
is documented in depth in [MEMORY-MODEL.md](MEMORY-MODEL.md).

Snippets in this section are runnable test blocks and assume:

```zig
const std = @import("std");
const fucina = @import("fucina");
```

### 3.1 The `Tensor(spec)` type constructor (`src/ag/tensor.zig`, `src/tags.zig`)

```zig
pub fn Tensor(comptime tags_spec: anytype) type
```

`spec` takes one of five forms, all normalized by `src/tags.zig`:

| Spec form | Example | Meaning |
|---|---|---|
| Named tag tuple | `Tensor(.{ .batch, .in })` | rank = tuple length, dtype `.f32` |
| Numeric rank | `Tensor(2)` | rank-2 f32 with auto tags `._0, ._1` |
| dtype + tags | `Tensor(.{ .dtype = .f16, .tags = .{ .seq, .d } })` | typed, named axes |
| dtype + rank | `Tensor(.{ .dtype = .i64, .rank = 2 })` | typed, auto tags `._0, ._1` |
| Scalar | `Tensor(.{})` | zero tags; stored as raw rank-1 shape `{1}` |

Tags are anonymous enum literals (`Tag = @TypeOf(.tag)`); any identifier
works (`.batch`, `.qkv`, `._0`). Auto tags for numeric-rank specs are
`._0 … ._7`. Rules enforced at compile time:

- **Unique tags.** Duplicate tags in one spec are a compile error
  (`validateUniqueTags`).
- **Max rank 8** (`tensor.max_rank`); more tags are a compile error.
- **dtype defaults to `.f32`** when the spec carries no `.dtype` field.
- A dtype-struct spec must include `.tags` or `.rank`.

**Type identity.** Specs that normalize to the same (dtype, tag list)
produce the *same* type: `Tensor(2) == Tensor(.{ ._0, ._1 })` and
`Tensor(.{ .batch, .in }) == Tensor(.{ .dtype = .f32, .tags = .{ .batch, .in } })`.
Tensors are therefore freely interchangeable across function boundaries
regardless of which spelling declared them.

**Comptime vs runtime.** Rank, tags, and dtype are comptime; sizes are
runtime. Every branch exposes the comptime constants

```zig
pub const axis_tags: [tag_count]Tag; // normalized tag list
pub const tag_count: usize;          // logical rank (0 for scalars)
pub const tensor_rank: usize;        // raw storage rank; rawRank(0) == 1
pub const dtype: DType;
```

**Scalars.** `Tensor(.{})` has `tag_count == 0` but `tensor_rank == 1`:
scalars are stored as a rank-1, single-element tensor of shape `{1}`
(zero-size and zero-rank raw tensors are not representable — `Shape.init`
rejects any zero dimension with `error.InvalidShape`; a deliberate design
stance, rationale in ARCHITECTURE.md "Tensor And Storage Model"). Full
reductions such as `sumAll` return `Tensor(.{})`.

```zig
test "Tensor spec forms and comptime introspection" {
    const A = fucina.Tensor(.{ .batch, .in }); // named tags, dtype defaults to f32
    const B = fucina.Tensor(2); // numeric rank: axes tagged ._0, ._1
    const C = fucina.Tensor(.{ .dtype = .i64, .rank = 2 }); // typed, auto tags
    const D = fucina.Tensor(.{ .dtype = .f16, .tags = .{ .seq, .d } }); // typed, named tags
    const S = fucina.Tensor(.{}); // scalar
    comptime {
        std.debug.assert(A.dtype == .f32 and A.tag_count == 2 and A.tensor_rank == 2);
        std.debug.assert(B.axis_tags[0] == ._0 and B.axis_tags[1] == ._1);
        std.debug.assert(C.dtype == .i64 and D.dtype == .f16);
        std.debug.assert(S.tag_count == 0 and S.tensor_rank == 1); // scalars store rank-1 [1]
        // Specs that normalize to the same (dtype, tags) are the SAME type:
        std.debug.assert(B == fucina.Tensor(.{ ._0, ._1 }));
        std.debug.assert(A == fucina.Tensor(.{ .dtype = .f32, .tags = .{ .batch, .in } }));
    }
}
```

### 3.2 The four facade branches (`src/ag/tensor.zig`)

`Tensor(spec)` comptime-dispatches on the dtype into four struct families.
The method set of each branch is decided at compile time, so calling an
unsupported operation is a compile error, never a runtime failure
(pinned by the `@hasDecl` guard tests in `src/ag/tensor_tests.zig`):

| Branch | dtypes | Capabilities |
|---|---|---|
| Float (differentiable) | `.f32` | Full surface: autograd (`variable`, `backward`, `grad`), all math/NN ops (§4), all views and structural ops |
| Typed float | `.f16`, `.bf16`, `.f64` (`supportsForwardFloatMath`) | Forward math: the native typed set (`add/sub/mul/div/sum/mean/sumAll/dot/scale/divScalar`, `to`), the full structural set (`split`/`merge`/`flatten`/`reshape`/`sliceStep`/`flip`/`roll`/`stack`/`repeatAxis` + the §3.10 base set), and — **f16/bf16 only** — the widened forward set (unary family, gated, softmax/norm family, remaining reductions, masks, `pad`, `einsum`; §4.19). **f16/bf16 only, autograd LEAVES**: `variable`/`variableFromSlice` with f32 gradients; differentiable `to` casts and mixed-RHS `dot`/`einsum` are the graph entries (§5.1) |
| Typed scalar constant | `.bool`, `.u8`, `.u16`, `.i8`, `.i16`, `.i32`, `.i64` | Constants only; construction, data access, structural ops, `to` (scalar casts, §3.8), and integer forward math (§4.19): wrapping `add`/`sub`/`mul`, `maximum`/`minimum`, explicit `divTrunc`/`divFloor`, i64-returning `sum`/`sumAll`, plus exact integer `compare` (§4.6). `.bool` keeps only `to`, the counting `sum`/`sumAll`, and the mask combinators `logicalAnd`/`logicalOr`/`logicalXor`/`logicalNot` |
| Block-quantized constant | `q1_0`, `q4_0 … q8_k`, `iq*`, `tq1_0/tq2_0`, `mxfp4`, `nvfp4` (`isBlockQuantized`) | Constants only; block construction, `to(.f32)` dequantize, `getRows`, row `concat`, `packRhs`/`packRhsLayout` (§10) |

Notes that follow from the dtype layer (`src/dtype.zig`, detailed in §8):

- `Scalar(.bf16)` is `u16` — bf16 tensors store and expose **raw bits**, not
  a native float type; `f16` uses Zig's `f16`.
- Block-quantized tensors have no per-element scalar; their element type is
  the block struct (`Storage(dtype)`, e.g. `fucina.BlockQ8_0`), and shapes
  count *logical* elements while storage counts blocks.
- Output dtypes of typed-float math follow `dtype_mod.outputDType`:
  pointwise and `dot` keep the input dtype; reductions (`sum`, `mean`,
  `sumAll`) widen `f16`/`bf16` to `f32` (`f64` stays `f64`).

Only the f32 and typed-float branches have gradient machinery: they carry
`grad_state: ?*GradState` and a `scope_owned: bool` flag (see §3.4; on the
typed-float branch only f16/bf16 tensors — autograd leaves and
differentiable cast results, §5.1 — ever populate it); the typed scalar and
block-quantized branches hold just the raw value and hard-code
`requiresGrad() == false`.

### 3.3 Construction and ownership (`src/ag/tensor.zig`, `src/exec.zig`)

All constructors are associated functions of the concrete tensor type and
take `ctx: *ExecContext` first (uniform signature even where the context is
unused, as in `constant`). Shape parameters are fixed-size arrays
`[tensor_rank]usize`, so passing the wrong-rank literal is a compile error.

**The core ownership contract.** `variable` and `constant` *consume* a raw
tensor produced by the context (`ctx.fromSlice(...)` and friends, §6) **on
success**; on error, ownership stays with the caller. Every returned facade
tensor is owned by the caller and must be released with `deinit()` (idiom:
`var x = try ...; defer x.deinit();`). Constructor failures never leak: the
facade holds `errdefer value.deinit()` internally around validation.

f32 branch:

```zig
pub fn variable(ctx: *ExecContext, value: RawTensor) !Self            // trainable leaf
pub fn variableFromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self
pub fn constant(ctx: *ExecContext, value: RawTensor) !Self            // no-grad wrap
pub fn fromTensor(ctx: *ExecContext, value: RawTensor) !Self          // alias of constant
pub fn fromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self
pub fn fromBorrowedSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []f32) !Self
pub fn fromBorrowedConstSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self
pub fn empty(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self  // uninitialized
pub fn zeros(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self
pub fn ones(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self
pub fn full(ctx: *ExecContext, raw_shape: [tensor_rank]usize, fill_value: f32) !Self
pub fn scalar(ctx: *ExecContext, scalar_value: f32) !Self             // single-element

pub fn emptyLike(self: *const Self, ctx: *ExecContext) !Self          // torch *_like
pub fn zerosLike(self: *const Self, ctx: *ExecContext) !Self
pub fn onesLike(self: *const Self, ctx: *ExecContext) !Self
pub fn fullLike(self: *const Self, ctx: *ExecContext, fill_value: f32) !Self

pub fn arange(ctx: *ExecContext, start: f32, end: f32, step: f32) !Self       // single-tag types only
pub fn linspace(ctx: *ExecContext, start: f32, end: f32, steps: usize) !Self  // single-tag types only
pub fn oneHot(ctx: *ExecContext, indices: []const usize, depth: usize) !Self  // two-tag types only

pub fn rand(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64) !Self       // uniform [0, 1)
pub fn uniform(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, lo: f32, hi: f32) !Self
pub fn randn(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64) !Self      // standard normal
pub fn normal(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, mean_value: f32, std_dev: f32) !Self
pub fn bernoulli(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, p: f32) !Self
```

Semantics:

- `variable` allocates a leaf `GradState`; the tensor participates in
  autograd (§5). `variableFromSlice` is `ctx.fromSliceRank` + `variable`.
- `fromSlice` **copies** `values` into context-owned storage.
- `fromBorrowedSlice` **borrows** caller-owned mutable storage zero-copy:
  the slice must stay alive and unmoved until the tensor's `deinit`;
  mutations of the backing slice are visible through the tensor. On a GPU
  build, mutate through the tensor's `data()` boundary (or synchronize
  externally) after submitting an op: direct writes through the external
  slice cannot be observed by the storage reader fence.
- `fromBorrowedConstSlice` borrows **read-only** storage (e.g. mmap'd GGUF
  weights) without a caller-side `@constCast`. The single internal
  `@constCast` is sound only under the contract that the data is never
  mutated through `.data()`; use `fromSlice` if a writable buffer is needed.
- `empty` returns uninitialized, buffer-pool-backed storage (§6); `zeros`,
  `ones`, `full`, `scalar` initialize it.
- The `*Like` forms are instance sugar over the same constructors
  (torch `zeros_like` and friends): same tags and dtype (both are part of
  the tensor type), shape taken from the receiver's logical shape (strided
  views included). Like every constructor the result is a fresh no-grad
  constant — the receiver's grad state does not carry over — and is never
  scope-owned. The typed scalar/float branches get `emptyLike`/`zerosLike`/
  `onesLike` (matching their static set, which has no `full`).
- `arange` is torch.arange with float semantics: element i is
  `start + i·step` (not accumulated), the end exclusive; `step == 0` or an
  empty range errors `InvalidShape` (zero-size tensors are not
  representable). `linspace` is torch.linspace: `steps` evenly spaced
  values, end INCLUSIVE and pinned exactly (`steps == 1` yields
  `{start}`; `steps == 0` is `InvalidShape`). `oneHot` builds the f32
  `[indices.len, depth]` one-hot matrix (torch F.one_hot with an explicit
  class count) from host-side indices — first tag rows, second tag
  classes; `indices[i] >= depth` is `IndexOutOfBounds`.
- The random constructors draw from the deterministic counter-based
  stream at `seed` (§6.8, `fucina.rng`): element i is a pure function of
  `(seed, i)`, so a stored seed regenerates the exact tensor — the stream
  IS the generator abstraction; pass a fresh seed per draw (reusing one
  reuses the values). `rand`/`uniform` map one stream output per element
  onto `[lo, hi)` (`uniformFill`); `randn`/`normal` are Box-Muller
  (`gaussianFill`/`normalFill`); `bernoulli` is 1.0 iff the `[0, 1)` draw
  at `(seed, i)` falls below `p` (`p` outside `[0, 1]` is
  `InvalidShape`). All are no-grad constants, like every constructor.
- Constructors are the f32 branch's; the *initialization* entry points on
  `ExecContext` they delegate to (`fromSlice`, `fromSliceRank`,
  `fromBorrowedSliceRank`, `fromSliceTyped`, `fromSliceRankTyped`,
  `fromBorrowedSliceRankTyped`, `fromStorageSliceRankTyped`,
  `fromBorrowedStorageSliceRankTyped`, `empty`, `emptyRank`,
  `emptyRankTyped`, `zeros`, `zerosTyped`, `ones`, `onesRank`, `onesTyped`,
  `onesRankTyped`, `full`, `fullTyped`, `scalar`) are catalogued in §6.

```zig
test "arange linspace and seed-deterministic random constructors" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var r = try fucina.Tensor(.{.d}).arange(&ctx, 0, 2, 0.5); // end exclusive
    defer r.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.5, 1, 1.5 }, try r.dataConst());
    var l = try fucina.Tensor(.{.d}).linspace(&ctx, 0, 1, 3); // end INCLUSIVE
    defer l.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.5, 1 }, try l.dataConst());

    // Same seed → the same tensor (the §6.8 counter-stream contract).
    const M = fucina.Tensor(.{ .row, .col });
    var a = try M.randn(&ctx, .{ 2, 4 }, 42);
    defer a.deinit();
    var b = try M.randn(&ctx, .{ 2, 4 }, 42);
    defer b.deinit();
    try std.testing.expectEqualSlices(f32, try a.dataConst(), try b.dataConst());
}
```

Error conditions (all recoverable errors, no panics):

| Error | Condition |
|---|---|
| `TensorError.InvalidShape` | raw rank ≠ `tensor_rank` in `variable`/`constant` (for `Tensor(.{})`: value not single-element); any zero dimension; rank 0 or > 8 |
| `TensorError.InvalidDataLength` | `values.len` ≠ product of `raw_shape` |
| `error.OutOfMemory` | allocation failure |

```zig
test "variable wraps a ctx-produced raw tensor" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .in })
        .variable(&ctx, try ctx.fromSlice(&.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 }));
    defer x.deinit();
    try std.testing.expect(x.requiresGrad());
    try std.testing.expectEqual(@as(usize, 3), x.dim(.in));
    try std.testing.expectEqual([2]usize{ 2, 3 }, x.shape());
    try std.testing.expectError(error.MutableDataRequiresNoGrad, x.data());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, try x.dataConst());
}
```

```zig
test "constant constructors" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var z = try fucina.Tensor(.{ .row, .col }).zeros(&ctx, .{ 2, 2 });
    defer z.deinit();
    var o = try fucina.Tensor(.{ .row, .col }).ones(&ctx, .{ 2, 2 });
    defer o.deinit();
    var f = try fucina.Tensor(.{ .row, .col }).full(&ctx, .{ 2, 2 }, 0.5);
    defer f.deinit();
    var u = try fucina.Tensor(.{ .row, .col }).empty(&ctx, .{ 2, 2 }); // uninitialized
    defer u.deinit();
    var s = try fucina.Tensor(.{}).scalar(&ctx, 3.5);
    defer s.deinit();
    try std.testing.expect(!a.requiresGrad());
    try std.testing.expectEqual(@as(f32, 3.5), try s.item());
    try std.testing.expectEqual([1]usize{1}, s.shape()); // scalar = raw rank-1 [1]
    try std.testing.expectError(error.InvalidDataLength, fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2 }));

    // *Like: same type (tags + dtype), shape from the receiver.
    var zl = try a.zerosLike(&ctx);
    defer zl.deinit();
    var mask = try a.fullLike(&ctx, -std.math.inf(f32));
    defer mask.deinit();
    try std.testing.expectEqual(a.shape(), zl.shape());
    try std.testing.expect(std.math.isNegativeInf((try mask.dataConst())[0]));
}
```

```zig
test "borrowed-storage constructors" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var backing = [_]f32{ 1, 2, 3, 4 };
    var t = try fucina.Tensor(.{ .row, .col }).fromBorrowedSlice(&ctx, .{ 2, 2 }, &backing);
    defer t.deinit();
    backing[0] = 42; // zero-copy: mutation is visible through the tensor
    try std.testing.expectEqual(@as(f32, 42), (try t.dataConst())[0]);

    const frozen = [_]f32{ 5, 6 }; // read-only source, no @constCast at the call site
    var c = try fucina.Tensor(.{.d}).fromBorrowedConstSlice(&ctx, .{2}, &frozen);
    defer c.deinit();
    try std.testing.expectEqual(@as(f32, 6), (try c.dataConst())[1]);
}
```

**Typed-constant branches** (int/bool and non-f32 float) share one
constructor set: `constant`, `fromTensor`, `fromSlice`,
`fromBorrowedConstSlice`, `empty`, `zeros`, `ones` — same semantics as the
f32 forms, elements typed `Scalar(dtype)`. There is no `full`, `scalar`, or
mutable `fromBorrowedSlice` on these branches. No `variable` except the
f16/bf16 leaf constructors `variable`/`variableFromSlice` (§3.2): gradients
are always f32.

**Block-quantized branch** constructors take *block* slices
(`Storage(dtype)`, e.g. `[]const fucina.BlockQ8_0`):

```zig
pub fn constant(ctx: *ExecContext, value: RawTypedTensor) !Self
pub fn fromTensor(ctx: *ExecContext, value: RawTypedTensor) !Self
pub fn fromBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self // copies
pub fn fromStorageSlice(...) !Self       // alias of fromBlocks
pub fn fromBorrowedBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []Elem) !Self // borrows
```

`raw_shape` counts logical elements (so the innermost dimension must be a
multiple of the block size); `values` counts blocks. Quantized *formats* and
the packed-RHS machinery are §10.

```zig
test "q8_0 constants: fromBlocks, dequantize, getRows" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const Q = fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });
    const bits = struct {
        fn of(x: f32) u16 {
            return @bitCast(@as(f16, @floatCast(x)));
        }
    }.of;
    var blocks = [_]fucina.BlockQ8_0{
        .{ .d = bits(1), .qs = [_]i8{1} ** fucina.q8_0_block_size },
        .{ .d = bits(2), .qs = [_]i8{3} ** fucina.q8_0_block_size },
    };
    var q = try Q.fromBlocks(&ctx, .{ 2, fucina.q8_0_block_size }, &blocks);
    defer q.deinit();

    var dense = try q.to(&ctx, .f32); // dequantize; .f32 is the only target
    defer dense.deinit();
    try std.testing.expectEqual(@as(f32, 6), (try dense.dataConst())[fucina.q8_0_block_size]);

    var row = try q.getRows(&ctx, .out, &.{1}, .batch); // dequantizing row gather
    defer row.deinit();
    comptime std.debug.assert(@TypeOf(row).dtype == .f32);
    try std.testing.expectEqual(@as(usize, 1), row.dim(.batch));
}
```

### 3.4 `deinit`, lifetime, and exec scopes (`src/ag/tensor.zig`, `src/tensor.zig`)

```zig
pub fn deinit(self: *Self) void
```

`deinit` releases one reference on the underlying refcounted buffer and
(f32 and typed-float branches) destroys the tensor's `GradState`; then sets
`self.* = undefined`,
so a second `deinit` on the same value is illegal (checked-UB in safe
builds, not a recoverable error). Buffer release is the driver of buffer-pool
recycling — the `defer x.deinit()` idiom returns transient storage to the
pool mid-forward (see [MEMORY-MODEL.md](MEMORY-MODEL.md) and §6).

- **Views retain.** Every view op (§3.7) bumps the source buffer's refcount;
  the storage is freed only when the last owner — parent or view — deinits.
  View lifetimes are independent of their parents'.
- **Exec scopes.** While `ctx.openExecScope()` is active (the training
  pattern, §5/§6), op *results* are adopted by the scope: the returned
  struct is a borrow with `scope_owned = true` and its `deinit` is a safe
  no-op — the scope releases value and graph node at `closeExecScope`. This
  lets the same defer-deinit forward code run scoped (training) and unscoped
  (inference). Tensors built by the *constructors* above are never
  scope-owned; only op results are.
- **No public clone.** `detach` and `materialize` are the copy/alias
  entry points (below); `grad()` returns an owned clone of the gradient.
- **Thread safety.** An `ExecContext` and the tensors flowing through it are
  single-threaded state: run ops on one context from one thread (parallelism
  happens *inside* ops via the context's worker pool, §9). Gradient
  *accumulation* is internally mutex-guarded (`GradState.grad_mutex`), but
  facade tensor values carry no cross-thread handle synchronization. A
  backend-internal accelerator completion on storage is a host-visibility
  fence, not permission to share a handle across threads (§9.9).

```zig
pub fn detach(self: *const Self, ctx: *ExecContext) !Self       // f32 + typed-float branches
pub fn materialize(self: *const Self, ctx: *ExecContext) !Self  // all branches
pub fn contiguous(self: *const Self, ctx: *ExecContext) !Self   // f32 branch only
```

`detach` returns a no-grad tensor **sharing storage** with `self` (a
refcounted view) — the values are live, the graph link is dropped.
`materialize` returns a **contiguous copy** in the tensor's logical order;
on the f32 branch it is differentiable (identity VJP through the strided
view). Use it to make a permuted/broadcast view exportable via `dataConst`.
`contiguous` is the borrow-if-contiguous variant (torch.contiguous): an
already-contiguous tensor returns a **zero-copy alias** of the same storage
(graph-linked through an identity VJP; in-place mutation of either handle is
visible through both), a strided view returns `materialize(ctx)` — an
independent snapshot. Either way the result is caller-owned (`deinit` it),
contiguous, and `dataConst`-safe; use `materialize` when a guaranteed copy
is wanted.

### 3.5 Data access (`src/ag/tensor.zig`, `src/tensor.zig`)

```zig
pub fn item(self: *const Self) !f32                 // f32; typed: !Scalar(dtype); absent on quantized
pub fn data(self: *Self) ![]f32                     // mutable element view
pub fn dataConst(self: *const Self) ![]const f32    // read-only element view
pub fn copyTo(self: *const Self, dst: []f32) !void  // stride-aware copy out
pub fn asRawTensor(self: *const Self) *const RawTensor
```

(Element types are `Scalar(dtype)` on typed branches and the block struct
`Storage(dtype)` on the quantized branch.)

- `item` requires a single-element tensor (`len() == 1`, any shape of
  all-ones); otherwise `error.InvalidShape`. It is how scalar losses are
  read out (`try loss.item()`).
- `data`/`dataConst` return a slice over the tensor's storage. Both require
  a **contiguous** layout and fail with `error.UnsupportedView` on strided
  views (permutes, broadcasts, inner narrows) — `materialize` first, or use
  `copyTo`. On the f32 branch, `data` additionally fails with
  `error.MutableDataRequiresNoGrad` on a `requiresGrad()` tensor: graph
  values must not be mutated behind autograd's back. Writes through `data`
  on shared (viewed) storage are visible to every alias.
- `copyTo` copies the logical elements row-major into `dst`
  (`dst.len` must equal the storage length, else
  `error.InvalidDataLength`); on scalar dtypes it walks strides, so it works
  on non-contiguous views.
- `asRawTensor` exposes the underlying raw tensor pointer **read-only** —
  the escape hatch for shape/stride introspection and for interop with
  exec-layer entry points (§6/§8). Treat it strictly as a borrow: never
  `deinit` through it, never mutate, and never outlive the facade value.

```zig
test "views, contiguity, materialize, copyTo" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var m = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer m.deinit();
    var t = try m.permuteTo(&ctx, .{ .col, .row }); // zero-copy strided view
    defer t.deinit();
    try std.testing.expectError(error.UnsupportedView, t.dataConst());
    var tm = try t.materialize(&ctx); // contiguous copy in the new order
    defer tm.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, try tm.dataConst());

    var mid = try m.narrow(&ctx, .col, 1, 2); // view into m's storage
    defer mid.deinit();
    var buf: [4]f32 = undefined;
    try mid.copyTo(&buf); // stride-aware; works on non-contiguous views
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 5, 6 }, &buf);
}
```

### 3.6 Shape and tag introspection (`src/ag/tensor.zig`, `src/tags.zig`)

```zig
pub fn shape(self: *const Self) [tensor_rank]usize   // runtime sizes, tag order
pub fn dim(self: *const Self, comptime tag: Tag) usize
pub fn axis(comptime tag: Tag) usize                 // comptime tag → axis index
pub fn hasTag(comptime tag: Tag) bool                // comptime membership test
```

`axis` is a compile error for an unknown tag (`tagIndexOrCompileError`);
`hasTag` is the non-failing probe generic code uses before calling `axis`.
`dim(tag)` is `shape()[axis(tag)]`. The comptime constants `axis_tags`,
`tag_count`, `tensor_rank`, `dtype` (§3.1) complete the introspection
surface. The tag algebra itself — how ops compute *result* tags — is §7.

### 3.7 Views and structural ops (`src/ag/tensor.zig`, `src/tagged.zig`, `src/exec/gather_scatter.zig`)

Two families. **Zero-copy views** re-describe existing storage (shape/stride
arithmetic plus a refcount retain — nothing is moved); **copying ops**
produce new storage. On the f32 branch every one of these is differentiable
(each records a backward node routing gradients through the inverse
transform; mechanics in §5); on constant branches the same names exist where
listed in §3.10 and simply produce constants.

#### Zero-copy views

```zig
pub fn withTags(self, ctx, comptime new_tags_spec) !Tensor(...)   // rename only, same rank
pub fn alignTo(self, ctx, comptime target_tags_spec) !Tensor(...) // permute + insert missing tags as size-1 axes
pub fn permuteTo(self, ctx, comptime target_tags_spec) !Tensor(...) // pure permutation (same tag set)
pub fn transpose(self, ctx, comptime target_tags_spec) !Tensor(...) // alias of permuteTo
pub fn insertAxis(self, ctx, comptime tag, comptime axis_index) !Tensor(...) // new size-1 axis
pub fn squeeze(self, ctx, comptime tag) !Tensor(...)              // drop a size-1 axis
pub fn split(self, ctx, comptime tag, comptime split_tags_spec, split_shape) !Tensor(...)
pub fn merge(self, ctx, comptime out_tag, comptime merge_tags_spec) !Tensor(...)
pub fn broadcastTo(self, ctx, comptime target_tags_spec, target_shape) !Tensor(...)
pub fn narrow(self, ctx, comptime tag, start: usize, length: usize) !Self
pub fn select(self, ctx, comptime tag, index: isize) !Tensor(...)   // one position, axis removed
pub fn viewWithStrides(self, ctx, comptime new_tags_spec, raw_shape, raw_strides) !Tensor(...)
```

- `withTags` requires the same rank; it is the bridge between numeric-tag
  (`._0`) tensors from generic loaders and named-tag model code.
- `permuteTo`/`transpose` require the same tag *set* (comptime-checked);
  `alignTo` additionally inserts missing target tags as size-1, stride-0
  axes.
- `split` factors one axis into several (`split_shape` product must equal
  the axis length, else `error.InvalidShape`); `merge` fuses *adjacent*
  axes (tags must be contiguous and in tensor order — comptime error
  otherwise; stride-incompatible layouts fail with
  `error.UnsupportedView`).
- `broadcastTo` produces stride-0 axes for missing or size-1 tags;
  mismatched non-1 sizes fail with `error.ShapeMismatch`.
- `squeeze` fails with `error.InvalidShape` if the axis size is not 1.
- `select` is torch.select / `x[i]`: one position of `tag` with the axis
  removed (the single-slice sibling of `unbindInto`; composed narrow →
  squeeze, so the result is a zero-copy view aliasing the selected row).
  `index` counts from the end when negative (torch convention);
  out-of-range errors with `IndexOutOfBounds`. Scope-required under
  gradients (§5); the gradient is the exact scatter — unselected
  positions receive zero.
- `viewWithStrides` is the audited escape hatch for layouts no tag
  operation can express: explicit raw shape + strides with a new tag set,
  bounds-checked against the underlying buffer
  (`error.InvalidDataLength` when the view would overrun).

```zig
test "split and merge" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .seq, .qkv })
        .fromSlice(&ctx, .{ 2, 6 }, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });
    defer x.deinit();
    var heads = try x.split(&ctx, .qkv, .{ .head, .d }, .{ 2, 3 }); // [seq, head, d] view
    defer heads.deinit();
    try std.testing.expectEqual([3]usize{ 2, 2, 3 }, heads.shape());
    var back = try heads.merge(&ctx, .qkv, .{ .head, .d }); // [seq, qkv] again
    defer back.deinit();
    try std.testing.expectEqualSlices(f32, try x.dataConst(), try back.dataConst());
}
```

```zig
test "tag-directed axis views" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var row = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer row.deinit();
    var aligned = try row.alignTo(&ctx, .{ .batch, .d }); // missing tag => size-1 axis
    defer aligned.deinit();
    try std.testing.expectEqual([2]usize{ 1, 3 }, aligned.shape());
    var grid = try row.broadcastTo(&ctx, .{ .batch, .d }, .{ 2, 3 }); // stride-0 view
    defer grid.deinit();
    try std.testing.expectEqual(@as(usize, 2), grid.dim(.batch));

    var col = try row.insertAxis(&ctx, .b, 0); // [b=1, d]
    defer col.deinit();
    var flat = try col.squeeze(&ctx, .b); // back to [d]
    defer flat.deinit();
    var renamed = try flat.withTags(&ctx, .{.feature}); // pure re-tag, same storage
    defer renamed.deinit();
    try std.testing.expectEqual(@as(usize, 3), renamed.dim(.feature));
}
```

```zig
test "viewWithStrides escape hatch" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var v = try fucina.Tensor(.{.flat}).fromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer v.deinit();
    var m = try v.viewWithStrides(&ctx, .{ .row, .col }, .{ 2, 2 }, .{ 2, 1 });
    defer m.deinit();
    try std.testing.expectEqual(@as(f32, 3), (try m.dataConst())[2]);
}
```

#### Copying structural ops

```zig
pub fn concat(self, ctx, comptime tag, others: []const *const Self) !Self
pub fn gather(self, ctx, comptime tag, indices: []const usize, comptime out_tag) !Tensor(...)
pub fn indexSelect(self, ctx, comptime tag, indices, comptime out_tag) !Tensor(...)
pub fn setSlice(self, ctx, comptime tag, start: usize, update: *const Self) !Self
pub fn setRows(self, ctx, comptime tag, indices: []const usize, update: *const Self) !Self
pub fn flatten(self, ctx, comptime out_tag) !Tensor(.{out_tag})
pub fn pad(self, ctx, comptime tag, before: usize, after: usize, fill: f32) !Self
pub fn zeroPad2d(self, ctx, comptime h_tag, comptime w_tag, padding: anytype) !Self
pub fn constantPad2d(self, ctx, comptime h_tag, comptime w_tag, padding: anytype, fill: f32) !Self
pub fn zeroSlice(self, ctx, comptime tag, start: usize, length: usize) !Self
pub fn zeroRows(self, ctx, comptime tag, indices: []const usize) !Self
pub fn flip(self, ctx, comptime tag) !Self
pub fn roll(self, ctx, comptime tag, shift: isize) !Self
pub fn repeatAxis(self, ctx, comptime tag, n: usize) !Self
pub fn stack(self, ctx, comptime new_tag, comptime axis_index, others: []const *const Self) !Tensor(...)
pub fn unbindInto(self, ctx, comptime tag, out: []Tensor(...)) !void
pub fn maskedSelect(self, ctx, mask, comptime out_tag) !Tensor(.{out_tag})
pub fn maskedScatter(self, ctx, mask, comptime values_tag, values: *const Tensor(.{values_tag})) !Self
pub fn rollBy(self, ctx, comptime tag, offsets: []const isize) !Self
pub fn shiftBy(self, ctx, comptime tag, offsets: []const isize, fill: f32) !Self
pub fn reshape(self, ctx, comptime new_tags_spec, new_shape: [...]usize) !Tensor(normalizeTags(new_tags_spec))
pub fn sliceStep(self, ctx, comptime tag, start: usize, length: usize, step: usize) !Self
pub fn slice(self, ctx, spec) !Self  // multi-axis basic slicing over a per-tag range struct
pub fn diagonal(self, ctx, comptime tag_a, comptime tag_b, comptime out_tag) !Tensor(...)
pub fn trace(self, ctx, comptime tag_a, comptime tag_b) !Tensor(...)
pub fn diag(self, ctx, comptime out_tags_spec) !Tensor(normalizeTags(out_tags_spec))
pub fn nonzero(self, allocator: std.mem.Allocator) ![]usize
pub fn indexAdd(self, ctx, comptime tag, indices: []const usize, update: *const Self) !Self
pub fn takeAlongAxis(self, ctx, comptime tag, indices) !Self
pub fn scatterAdd(self, ctx, comptime tag, indices, src: *const Self) !Self
pub fn scatter(self, ctx, comptime tag, indices, src: *const Self) !Self
```

- `concat` joins along an existing tag (all other dims must match); the
  result owns fresh storage. `gather` copies the rows selected by `indices`
  along `tag`, re-tagging that axis `out_tag` (`out_tag` may equal `tag`);
  out-of-range indices error with `IndexOutOfBounds`. `indexSelect` is
  torch.index_select — `gather` with a rank-1 **i64** index tensor (the
  argmax/topK/sort convention; other dtypes are compile errors), read
  host-side into the same `[]usize` path; entries outside `[0, dim(tag))`
  error with `IndexOutOfBounds` (no wrapping), duplicate reads accumulate
  their gradients, and the index tensor is control data outside the graph.
- `setSlice`/`setRows` are *functional* scatter updates: they return a copy
  of `self` with the range `[start, start+len)` / the given rows along
  `tag` overwritten by `update`; the originals are untouched. Gradients
  flow to both `self` (masked) and `update`. `setRows` requires unique
  in-range indices (`IndexOutOfBounds` / `InvalidShape` on duplicates), and
  `update` must match `self` except along `tag`, where it must have
  `indices.len` rows. `zeroSlice`/`zeroRows` are the fill-with-zero variants
  (no `update` operand).
- `flatten` reshapes to rank-1 under a new tag, materializing first only if
  the source is non-contiguous.
- `pad` grows one axis by `before + after` positions holding `fill`;
  `flip` reverses an axis; `roll` rotates it by `shift` (negative allowed);
  `repeatAxis` tiles the axis `n` times (`n == 0` errors with
  `InvalidShape`, `n == 1` is a zero-copy identity view).
- `zeroPad2d`/`constantPad2d` are torch nn.ZeroPad2d/nn.ConstantPad2d over
  named axes: `padding` is an integer (all four sides) or a 4-tuple/array
  in the torch order `(left, right, top, bottom)` — left/right grow
  `w_tag`, top/bottom grow `h_tag`; negative entries CROP that side (the
  F.pad constant-mode rule). Cropping an axis to zero size or below errors
  `InvalidShape` — the one deliberate divergence: torch returns an empty
  tensor at exactly zero (zero-size tensors are not representable here)
  and errors only below zero. Any rank carrying both tags works; pad
  positions drop their gradient and cropped source positions receive zero
  gradient. Forward/backward semantics are pinned against torch 2.0
  vectors (nn.ZeroPad2d / nn.ConstantPad2d / F.pad, gradients included).
- `stack` inserts a new axis on every input and concatenates along it
  (torch.stack); `unbindInto` fills a caller-provided slice with the
  `dim(tag)` sub-tensors of `self` with `tag` removed — the caller owns and
  deinits every filled entry (under an exec scope they are scope-owned
  borrows and `deinit` is a no-op); `out.len` must equal `dim(tag)`.
  `maskedSelect` returns the elements where `mask` is nonzero as a rank-1
  tensor. Selecting nothing errors with the dedicated `EmptySelection`
  (zero-size tensors are not representable) — distinct from the shape
  errors, so the data-dependent no-match outcome is catchable apart from
  caller bugs; pre-counting with a mask sum avoids the error path entirely
  (snippet below).
- `maskedScatter` is `maskedSelect`'s inverse (torch masked_scatter with an
  exact-count contract): it returns a copy of `self` with the rank-1
  `values` written into the nonzero-mask positions in row-major order.
  `values` must hold exactly `count(mask != 0)` elements (`InvalidShape`
  otherwise) and the mask must select at least one (`EmptySelection`
  otherwise, as in `maskedSelect`); the mask follows the
  `where`/`maskedFill` convention (non-grad `!= 0`, `self`'s shape,
  contiguous). Differentiable in `self` (grad zeroed where scattered) and
  `values` (grad gathered from the selected positions).
- `rollBy`/`shiftBy` generalize `roll` to one shift per *section* (the
  sub-vector obtained by fixing all axes except `tag`), keeping `roll`'s
  sign convention. `offsets` is host-side control data like `gather`
  indices: one `isize` per section, row-major over the remaining axes in
  tag order (`offsets.len == numel / dim(tag)`, else `InvalidShape`; a
  rank-1 tensor takes a single offset and `rollBy` matches `roll`). `rollBy`
  wraps (exact permutation gradient); `shiftBy` is non-circular — shifted-in
  positions hold the constant `fill` (no gradient) and shifted-out source
  positions receive zero gradient.
- `reshape` is torch.reshape over named axes: an arbitrary row-major
  reinterpretation to `new_tags_spec`/`new_shape` (element counts must
  match, `InvalidShape` otherwise) with the torch view-or-materialize
  rule — a contiguous source stays a zero-copy view, a strided one
  materializes first (composed flatten → split; a rank-1 target
  degenerates to plain `flatten`).
- `sliceStep` is `narrow` with a step (torch `x[start::step]` on one
  axis): a zero-copy strided view on no-grad tensors; under gradients it
  lowers to `gather` over the stepped indices (a copy with the exact
  scatter-add record — the `flip`/`roll` precedent).
- `slice` is multi-axis basic slicing (torch/numpy `x[1:-1, ::2]`,
  positive steps): `spec` is a struct literal naming the tags to slice —
  `.{ .h = .{ .start = 1, .end = -1 }, .w = .{ .step = 2 } }` — each
  field a `fucina.SliceRange`-shaped range (`start`/`end`/`step`, each
  optional); unnamed axes pass through whole, and naming a tag not on the
  tensor is a compile error. torch bounds semantics: negatives count from
  the end, `end = null` means the axis dim, out-of-range bounds clamp;
  `step == 0` or an empty result error with `InvalidShape` (zero-size
  tensors are not representable). Negative steps are deliberately
  unsupported (torch rejects them in basic indexing too; strides are
  unsigned, so a reversed view cannot exist — compose `flip`). Lowered to
  per-axis `narrow`/`sliceStep` in tag order: step-1 ranges stay zero-copy
  views with exact scatter gradients; stepped axes follow the `sliceStep`
  contract. Scope-required under gradients when more than one axis is
  sliced.
- `diagonal` views the main diagonal over a (`tag_a`, `tag_b`) plane
  (torch.diagonal, offset 0, any rank carrying both tags): length
  `min(dim_a, dim_b)`, both tags removed, the diagonal appended LAST as
  `out_tag`; zero-copy and differentiable. `trace` is composed diagonal →
  sum; `diag` embeds a rank-1 tensor as the diagonal of an `[n, n]`
  matrix (composed zeros → setRows → reshape).
- `nonzero` returns the row-major flat indices of the nonzero elements
  (NaN counts) as a HOST `[]usize` the caller frees — data-dependent
  cardinality stays host-side by design (ARCHITECTURE.md), so a no-match
  result is just an empty slice, and the indices feed straight into
  `gather`/`setRows`/`indexAdd`/`oneHot`.
- `indexAdd` is torch.index_add: a copy of `self` with `update`'s rows
  ADDED at host-side `indices` along `tag` — unlike `setRows` duplicates
  are allowed and accumulate; differentiable in both (identity /
  row-gather).
- `takeAlongAxis`/`scatterAdd`/`scatter` are the per-ELEMENT indexed ops
  (torch gather / scatter_add / scatter): the index operand is a
  same-tagged i64 tensor in the argmax/topK/sort index convention, so
  selection-op outputs feed them directly (§4.16-17).
- `unbindInto`, `select`, `slice` (more than one sliced axis),
  `maskedSelect`, `maskedScatter`, `rollBy`, `shiftBy`, `stack`,
  `zeroPad2d`, `constantPad2d`, `reshape` (multi-tag targets), `trace`,
  and `diag` are *composed* ops: when gradients are tracked they require
  an active exec scope and error with `error.ActiveExecScopeRequired`
  otherwise (§5).
- Quantized branch: `concat` (rank-2, row axis only) and
  `getRows(ctx, tag, indices, out_tag)` — a fused gather+dequantize
  returning an **f32** tensor (see the snippet in §3.3); both comptime-reject
  other configurations.

```zig
test "maskedSelect no-match outcome: count first or catch EmptySelection" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, -2, 3 });
    defer x.deinit();
    var mask = try x.compare(&ctx, .gt, 10); // nothing matches
    defer mask.deinit();

    // Count-first idiom: the .bool mask sums to the selection count (i64).
    var count = try mask.sumAll(&ctx);
    defer count.deinit();
    try std.testing.expectEqual(@as(i64, 0), try count.item());

    // Or catch the dedicated error — EmptySelection is the recoverable
    // no-match outcome; shape errors (caller bugs) stay loud.
    var picked = x.maskedSelect(&ctx, mask, .m) catch |err| switch (err) {
        error.EmptySelection => null,
        else => return err,
    };
    defer if (picked) |*p| p.deinit();
    try std.testing.expect(picked == null);
}
```

```zig
test "zeroPad2d follows the torch (left, right, top, bottom) order" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .h, .w }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    // nn.ZeroPad2d((1, 0, 0, 1)): one zero column left, one zero row below.
    var padded = try x.zeroPad2d(&ctx, .h, .w, .{ 1, 0, 0, 1 });
    defer padded.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        0, 1, 2,
        0, 3, 4,
        0, 0, 0,
    }, try padded.dataConst());

    // Negative padding crops (the F.pad constant-mode rule).
    var cropped = try x.zeroPad2d(&ctx, .h, .w, .{ -1, 0, 0, 0 });
    defer cropped.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4 }, try cropped.dataConst());
}
```

```zig
test "concat, gather, setSlice, setRows" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 1, 2 }, &.{ 5, 6 });
    defer b.deinit();
    var cat = try a.concat(&ctx, .row, &.{&b}); // [3, 2]
    defer cat.deinit();
    try std.testing.expectEqual(@as(usize, 3), cat.dim(.row));

    var picked = try cat.gather(&ctx, .row, &.{ 2, 0 }, .sel); // rows 2 and 0
    defer picked.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 1, 2 }, try picked.dataConst());

    var patch = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 1, 2 }, &.{ 9, 9 });
    defer patch.deinit();
    var replaced = try cat.setSlice(&ctx, .row, 1, &patch); // overwrite row 1
    defer replaced.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 9, 9, 5, 6 }, try replaced.dataConst());

    var scattered = try cat.setRows(&ctx, .row, &.{2}, &patch); // overwrite row 2
    defer scattered.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 9, 9 }, try scattered.dataConst());
}
```

The same structural surface works on typed constants:

```zig
test "integer constants: structural ops only" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var ids = try fucina.Tensor(.{ .dtype = .i64, .rank = 2 })
        .fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ids.deinit();
    comptime std.debug.assert(@TypeOf(ids).axis_tags[0] == ._0); // auto numeric tags
    var named = try ids.withTags(&ctx, .{ .batch, .seq });
    defer named.deinit();
    var t = try named.transpose(&ctx, .{ .seq, .batch });
    defer t.deinit();
    var out: [6]i64 = undefined;
    try t.copyTo(&out);
    try std.testing.expectEqualSlices(i64, &.{ 1, 4, 2, 5, 3, 6 }, &out);
    comptime std.debug.assert(@hasDecl(@TypeOf(ids), "add")); // wrapping int math (§4.19)
    comptime std.debug.assert(!@hasDecl(@TypeOf(ids), "softmax")); // float NN ops stay off ints
}
```

### 3.8 Casting: `to(dtype)` (`src/ag/tensor.zig`, `src/exec/convert.zig`)

```zig
pub fn to(self: *const Self, ctx: *ExecContext, comptime target_dtype: DType)
    !Tensor(.{ .dtype = target_dtype, .tags = axis_tags })
```

`to` always copies (a new tensor; the source is untouched). Supported
conversions per branch:

| Source branch | Targets | Notes |
|---|---|---|
| f32 | any scalar dtype | `to(.f32)` is a differentiable copy; `to(.f16)`/`to(.bf16)` are DIFFERENTIABLE narrows (the mixed-precision seam, §5.1: the backward is the identity on the f32 upstream gradient); every other target requires no-grad and fails with `error.GradientCastUnsupported` on a `requiresGrad()` tensor — `to(.f64)` is a float↔float cast, non-float targets follow the `castScalar` semantics in the int/bool row |
| f16/bf16 | any scalar dtype | `to(.f32)` is a DIFFERENTIABLE widen when the source requires grad (the f32 gradient flows back unchanged); casts to any non-f32 target require no-grad (`error.GradientCastUnsupported`) — float targets are float↔float casts, non-float targets follow the `castScalar` semantics in the int/bool row |
| f64 constant | any scalar dtype | always no-grad (an f64 constant never carries a gradient); float targets are float↔float casts, non-float targets follow the `castScalar` semantics in the int/bool row |
| int/bool constant | any scalar dtype | no-grad `castScalar` semantics: integer↔integer WRAPS (two's complement); integer→float is exact where representable; float→integer truncates toward zero and SATURATES at the target bounds with NaN → 0; anything→bool is `!= 0` (NaN → true); bool→number is 0/1 |
| block-quantized | `.f32` only | dequantization; other targets are a compile error |

```zig
test "f16 constants: forward math, reductions widen" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const H = fucina.Tensor(.{ .dtype = .f16, .tags = .{ .m, .k } });
    var a = try H.fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var twice = try a.add(&ctx, &a);
    defer twice.deinit();
    comptime std.debug.assert(@TypeOf(twice).dtype == .f16); // pointwise keeps dtype
    var s = try a.sum(&ctx, .k);
    defer s.deinit();
    comptime std.debug.assert(@TypeOf(s).dtype == .f32); // reductions widen to f32
    var dense = try a.to(&ctx, .f32);
    defer dense.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, try dense.dataConst());
}
```

```zig
test "detach and cast rules" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variable(&ctx, try ctx.fromSlice(&.{2}, &.{ 1, 2 }));
    defer x.deinit();
    var frozen = try x.detach(&ctx); // shares storage, drops grad tracking
    defer frozen.deinit();
    try std.testing.expect(!frozen.requiresGrad());
    try std.testing.expectError(error.InvalidShape, frozen.item()); // not single-element
    try std.testing.expectError(error.GradientCastUnsupported, x.to(&ctx, .f64));
    var narrowed = try x.to(&ctx, .f16); // differentiable narrow: stays in the graph
    defer narrowed.deinit();
    try std.testing.expect(narrowed.requiresGrad());
    var half = try frozen.to(&ctx, .f16); // constants cast freely between floats
    defer half.deinit();
    comptime std.debug.assert(@TypeOf(half).dtype == .f16);
}
```

### 3.9 Gradient accessors (`src/ag/tensor.zig`, `src/ag/core.zig`; mechanics in §5)

f32-branch surface. `requiresGrad` also exists on every other branch:
hard-wired `false` on the scalar/quantized/f64 constants, a real
`grad_state != null` check on f16/bf16 — whose leaf-autograd accessors
(`variable`, `grad`, `gradView`, `zeroGrad`, `detach`) mirror the f32 ones
with f32-dtype gradient results (§5.1).

```zig
pub fn requiresGrad(self: *const Self) bool
pub fn backward(self: *const Self, ctx: *ExecContext) !void  // error.NoGradientGraph on constants
pub fn backwardWithGrad(self: *const Self, ctx: *ExecContext, grad_output: *const Self) !void // explicit output gradient
pub fn grad(self: *const Self, ctx: *ExecContext) !?Self     // owned CLONE of the gradient, or null
pub fn gradView(self: *const Self, ctx: *ExecContext) !?Self // refcounted VIEW of the live gradient
pub fn zeroGrad(self: *const Self) void                      // drop accumulated grad; no-op on constants
```

`grad`/`gradView` return `null` before any `backward` has produced a
gradient (and again after `zeroGrad`). Both returns are no-grad tensors the
caller must `deinit`; `gradView` shares storage with the accumulator *as of
that moment* — a later `backward` pass accumulates into a fresh private
buffer (the held view defeats copy-on-write, §5.3), so the view keeps the
stale value; use `grad` to observe later passes. Training loops call
`zeroGrad` between steps so gradients
do not accumulate across them. Graph construction, `fucina.noGrad`,
checkpointing, and the traversal/seeding contract of `backward` and
`backwardWithGrad` (a non-scalar output needs the explicit output gradient;
one backward per graph) are §5.

```zig
test "grad accessors on the facade" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variable(&ctx, try ctx.fromSlice(&.{3}, &.{ 1, 2, 3 }));
    defer x.deinit();
    try std.testing.expect((try x.grad(&ctx)) == null); // no backward run yet
    var loss = try x.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var g = (try x.grad(&ctx)).?; // owned clone; caller deinits
    defer g.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1 }, try g.dataConst());
    x.zeroGrad(); // drop the accumulated gradient
    try std.testing.expect((try x.grad(&ctx)) == null);
}
```

### 3.10 Facade surface index

Complete public surface of `Tensor(spec)`, split by owning section. §3
methods are documented above; §4 covers every math/NN op in depth.

**f32 branch — §3 (construction / lifetime / data / structure):**
`variable`, `variableFromSlice`, `constant`, `fromTensor`, `fromSlice`,
`fromBorrowedSlice`, `fromBorrowedConstSlice`, `empty`, `zeros`, `ones`,
`full`, `scalar`, `emptyLike`, `zerosLike`, `onesLike`, `fullLike`,
`arange`, `linspace`, `oneHot`, `rand`, `uniform`, `randn`, `normal`,
`bernoulli`,
`deinit`, `asRawTensor`, `item`, `data`, `dataConst`,
`copyTo`, `detach`, `materialize`, `contiguous`, `requiresGrad`, `zeroGrad`,
`backward`, `backwardWithGrad`, `grad`, `gradView`, `axis`, `hasTag`, `dim`,
`shape`, `to`,
`withTags`, `viewWithStrides`, `alignTo`, `permuteTo`, `transpose`,
`insertAxis`, `squeeze`, `split`, `merge`, `reshape`, `broadcastTo`,
`narrow`, `select`, `slice`, `sliceStep`,
`flatten`, `gather`, `indexSelect`, `maskedSelect`, `maskedScatter`, `nonzero`, `flip`,
`roll`, `rollBy`, `shiftBy`, `concat`, `stack`, `unbindInto`, `repeatAxis`,
`pad`, `zeroPad2d`, `constantPad2d`, `setSlice`, `setRows`, `indexAdd`,
`takeAlongAxis`, `scatterAdd`, `scatter`, `zeroSlice`,
`zeroRows`, `diagonal`, `diag`, `trace`; consts
`axis_tags`, `tag_count`, `tensor_rank`, `dtype`; fields `value`,
`grad_state`, `scope_owned`.

**f32 branch — §4 (math / NN):**
`add`, `sub`, `mul`, `div`, `scale`, `addScalar`, `subScalar`, `divScalar`,
`powScalar`, `log1p`, `takeAddNoGrad`, `takeScaleNoGrad`,
`addAxisVectorInPlace`, `addAxisVectorUnaryInPlace`, `addScaledInPlace`,
`biasAdd`, `where`, `maskedFill`, `compare`, `logicalAnd`, `logicalOr`,
`logicalXor`, `logicalNot`, `unary`, `elementalUnary`, `elementalBinary`,
`relu`, `leakyRelu`, `exp`, `sqrt`,
`rsqrt`, `sigmoid`, `silu`, `log`, `neg`, `abs`, `sin`, `cos`, `tanh`,
`fastTanh`, `softcap30`, `softcap15`, `gelu`, `quickGelu`, `elu`, `geluErf`,
`floor`, `ceil`, `round`, `sign`, `reciprocal`, `clamp`,
`maximum`, `minimum`, `pow`, `isnan`, `isinf`, `isfinite`,
`dropout`, `gated`, `glu`, `swiglu`, `geglu`, `splitGated`, `sum`, `mean`,
`cumsum`, `prod`, `cumprod`, `variance`, `standardizeAxis`, `sumAll`,
`sumMany`, `any`, `all`, `anyAll`, `allAll`, `norm`, `normAll`,
`logsumexp`, `logSoftmax`, `argmax`,
`max`, `min`, `topK`, `sort`, `argsort`, `routerTopK`, `softmax`, `rmsNorm`,
`rmsNormMul`, `rmsNormMulAdd`, `rmsNormMulRopeHalfPrepared`, `layerNorm`,
`groupNorm`, `crossEntropy`, `crossEntropyExt`, `linearCrossEntropyExt`,
`mseLoss`, `huberLoss`,
`bceLoss`, `klDivLoss`, `nllLoss`, `l2Normalize`, `cosineSimilarity`,
`rope`, `matmul`, `dot`, `einsum`, `dotTernarySte`, `dotPacked`,
`rmsNormMulDotPacked`,
`splitSwiGluDotPacked`, `gegluQuantDotPacked`, `groupedAttention`,
`conv2d`, `conv2dRelu`, `prepareConv2dWeights`, `conv2dPrepared`,
`conv2dPreparedRelu`, `maxPool2d`, `avgPool2d`, `upsample2xNearest`,
`prelu`, `channelAffine`, `relposShift`, `causalDepthwiseConv1d`,
`causalConv1d`, `groupedCausalConv1d`, `conv1d`, `convTranspose1d`,
`snake`. The root free function `fucina.einsumMany(ctx, out_tags, operands)`
is the N-ary companion of `einsum` (§4.8).

**Typed scalar-constant branch** (`.bool`/ints): `constant`, `fromTensor`,
`fromSlice`, `fromBorrowedConstSlice`, `empty`, `zeros`, `ones`,
`emptyLike`, `zerosLike`, `onesLike`, `item`,
`data`, `dataConst`, `copyTo`, `axis`, `hasTag`, `deinit`, `asRawTensor`,
`requiresGrad`, `dim`, `shape`, `materialize`, `withTags`, `alignTo`,
`permuteTo`, `transpose`, `insertAxis`, `squeeze`, `broadcastTo`, `gather`,
`narrow`, `concat`, `setSlice`, `setRows` — all §3 — plus `to` (§3.8), the
integer forward math `add`, `sub`, `mul`, `maximum`, `minimum`,
`divTrunc`, `divFloor`, `sum`, `sumAll` (§4.19; on `.bool` the arithmetic
entries are compile errors — `to` and the counting `sum`/`sumAll` apply),
integer `compare` (§4.6, exact at any magnitude), and — on `.bool` only —
the mask combinators `logicalAnd`, `logicalOr`, `logicalXor`,
`logicalNot`.

**Typed float branch** (`.f16`/`.bf16`/`.f64`): everything in the
scalar-constant branch's §3 list, plus `requiresGrad`, `zeroGrad`, and
`detach` on every typed float dtype (on f64 no leaf can exist, so
`requiresGrad` is `false`, `zeroGrad` no-ops and
`detach` just returns a no-grad view), plus — f16/bf16 only, compile
errors on f64 — the leaf-autograd surface `variable`, `variableFromSlice`,
`grad`, `gradView` (f32 gradients; §5.1), plus `to`
(§3.8), the forward-only math `add`,
`sub`, `mul`, `div`, `sum`, `mean`, `sumAll`, `dot`, `scale`, `divScalar`,
and the structural set `split`, `merge`, `flatten`, `reshape`, `sliceStep`,
`flip`, `roll`, `stack`, `repeatAxis`. **f16/bf16 only** (each computes
through f32 and narrows once; a compile error on f64 — §4.19): `unary` and
the named unary aliases (`relu`, `exp`, `sqrt`, `rsqrt`, `sigmoid`, `silu`,
`log`, `log1p`, `neg`, `abs`, `sin`, `cos`, `tanh`, `fastTanh`, `softcap30`,
`softcap15`, `gelu`, `quickGelu`, `elu`, `geluErf`, `floor`, `ceil`,
`round`, `sign`, `reciprocal`), `leakyRelu`, `clamp`, `addScalar`,
`subScalar`, `powScalar`, `maximum`, `minimum`, `gated`, `glu`, `swiglu`,
`geglu`, `softmax` (plain `.{}` options), `logSoftmax`, `rmsNorm`,
`rmsNormMul`, `layerNorm` (plain `.{}` options), `cumsum`, `cumprod`,
`where`, `maskedFill`, `compare`, `pad`, `einsum`, the widened
reductions `max`, `min`, `prod`, `variance`, `logsumexp` (f32 results,
§8.3), and `argmax` (i64 result, §4.16).

**Block-quantized branch:** `constant`, `fromTensor`, `fromBlocks`,
`fromStorageSlice`, `fromBorrowedBlocks`, `deinit`, `asRawTensor`, `data`,
`dataConst`, `copyTo`, `requiresGrad`, `axis`, `hasTag`, `dim`, `shape`,
`withTags`, `to`, `materialize`, `concat`, `getRows` — all §3 — plus
`packRhs`, `packRhsLayout` (packed matmul RHS containers; §10, used by
`dotPacked` in §4). The root helper `fucina.PackedRhs(dtype)` names
`packRhs`'s return type (§10).

## 4. Tensor operations

This section is the reference for the math/NN operation surface of the public
autograd `fucina.Tensor(tags_spec)` facade (`src/ag/tensor.zig`), the
tag-semantics lowering library behind it (`src/tagged.zig`), and the public
option types those operations take (`src/exec.zig`). Construction, data
access, and structural views (`withTags`, `permuteTo`, `transpose`,
`alignTo`, `insertAxis`, `squeeze`, `split`, `merge`, `broadcastTo`,
`viewWithStrides`, `flatten`, `materialize`, `detach`) are covered in §3; the
tag algebra itself in §7; the autograd engine driving the backward records
named here in §5.

### 4.1 The common operation contract (`src/ag/tensor.zig`)

Every operation below shares one contract, implemented by the shared tails
`finishOp`/`finishNoGrad`:

- **Signature shape.** Ops are methods on `Tensor(tags)` taking
  `ctx: *ExecContext` as the first runtime argument. Axes are chosen by
  comptime tag (`comptime tag: Tag`); misnaming a tag that the tensor does
  not carry is a **compile error**, never a runtime error. Shape problems the
  type system cannot see (mismatched dims, bad lengths) are recoverable
  `TensorError`s (`ShapeMismatch`, `InvalidShape`, `InvalidDataLength`,
  `IndexOutOfBounds`, integer division's `DivisionByZero`); the
  data-dependent no-match outcome of
  `maskedSelect`/`maskedScatter` gets the dedicated `EmptySelection` so it
  stays catchable apart from those.
- **Ownership.** Each op allocates and returns a **new owned tensor**; the
  caller `deinit`s it. Operands are borrowed via `*const` and never consumed
  (the two `take*` ops in §4.3 are the documented exception). While an exec
  scope is open on the context, returned tensors are scope-owned borrows and
  their `deinit` is a safe no-op (§6).
- **Gradients.** A backward record is attached iff at least one operand
  `requiresGrad()` and gradients are globally enabled (`fucina.noGrad`, §5).
  Families that are no-grad by design, or that restrict which operands
  receive gradients, say so inline; grad-incompatible calls fail with
  `error.UnsupportedGradient` (or a more specific error named per family).
  Ops **composed** from other facade ops (`nllLoss`, `l2Normalize`,
  `cosineSimilarity`, `maskedSelect`, `stack`, `unbindInto`) additionally
  require an active exec scope when gradients are tracked — their
  intermediate graph nodes are function-local and only a scope can own them
  until `backward`; they fail with `error.ActiveExecScopeRequired` otherwise.
  No-grad use works unscoped.
- **Thread-safety.** A context is single-threaded at the API surface: run
  ops on one `ExecContext` from one thread (§6). Kernels parallelize
  internally through the context's work pool (§9).
- **Option types.** Options are passed as literals, so their type names are
  rarely written out. The ones re-exported at the `fucina` root are
  `UnaryOp`, `Reduction`, `CrossEntropyOptions`, `StandardizeOptions`,
  `StandardizeAccumulation`, `StandardizeEpsMode`, `RouterTopKOptions`,
  `RopeMode`, `RopeTable`, `RopeTheta`, `MoeRhs`, `MoeBatchProfile`,
  `PackedRhs`, `PackedRhsLayout`, `GatedOp`. The remaining option types
  named in this section (`CompareOp`, `MatmulKind`, `MseOptions`,
  `HuberOptions`, `BceOptions`, `KlDivOptions`, `SoftmaxExtOptions`,
  `NormOrder`) live in
  `src/exec.zig` and are reached through enum/struct literals at call sites
  (`.swiglu`, `.lt`, `.trans_b`, `.{ .reduction = .none }`).

Snippets in this section are runnable test blocks and assume:

```zig
const std = @import("std");
const fucina = @import("fucina");
```

### 4.2 Pointwise binary ops and tag-driven broadcasting (`src/ag/tensor.zig`, `src/tagged.zig`)

```zig
pub fn add(self: *const Self, ctx: *ExecContext, other: anytype)
    !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags))
// same shape for: sub, mul, div; TensorObject unwraps pointer operands
```

`add`, `sub`, `mul`, `div` broadcast **by tag name**, not by position. The
result tag set is `pointwiseResultTags`: `self`'s tags in order, followed by
`other`'s tags that `self` does not carry. Per shared tag the dims must be
equal or one of them 1; a tag missing from one operand behaves as dim 1.
Broadcasting is a zero-stride view (no materialization); when both operands
have identical tags and shapes the kernel runs directly with no view step.
`other` may be a tensor value or pointer. Backward: full two-operand VJP; a
gradient flowing into a broadcast operand is reduced back over the broadcast
axes (§5).

Three more binary pointwise ops share the same tag-broadcast rule and
two-operand VJP:

- `maximum(ctx, other)` / `minimum(ctx, other)` — torch.maximum/minimum,
  full `.max`/`.min` members of the binary kernel enum (pooled SIMD, the
  `add`/`mul` tier): NaN in either operand propagates NaN (NOT the IEEE
  maxNum rule bare `@max` follows); the gradient goes to the winning
  operand and is split evenly on exact ties (torch's subgradient, ±inf
  ties included).
- `pow(ctx, other)` — `self ^ other` with `std.math.pow` domain semantics
  (negative base + non-integer exponent is NaN, `0^0 = 1`); `powScalar`
  (§4.3) is the scalar-exponent fast path. Implemented over the elemental
  tier (§4.4) — `std.math.pow` has no portable SIMD form with these
  domain semantics. The exponent-side gradient `ln(a)·a^b` is meaningful
  only for positive bases, as in torch.

```zig
test "pointwise add broadcasts by tag" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var b = try fucina.Tensor(.{.col}).variableFromSlice(&ctx, .{2}, &.{ 10, 20 });
    defer b.deinit();

    var y = try x.add(&ctx, &b); // result tags .{ .row, .col }; b broadcasts over .row
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 13, 24 }, try y.dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gb = (try b.grad(&ctx)).?; // broadcast VJP: gradient reduced back to .{ .col }
    defer gb.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 2 }, try gb.dataConst());
}
```

### 4.3 Scalar variants and in-place/no-grad helpers (`src/ag/tensor.zig`)

Scalar variants (all differentiable, all return a new tensor):

| Method | Value | Backward |
|---|---|---|
| `scale(ctx, s)` | `x·s` | `ScaleBackward` |
| `addScalar(ctx, s)` | `x + s` | pass-through |
| `subScalar(ctx, s)` | `addScalar(-s)` | pass-through |
| `divScalar(ctx, s)` | `scale(1/s)` | via `scale` |
| `powScalar(ctx, e)` | `x^e` (positive `x`) | `e·x^(e−1)` |

```zig
test "scalar op variants" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer x.deinit();
    var shifted = try x.addScalar(&ctx, 1); // {4, 5}
    defer shifted.deinit();
    var squared = try shifted.powScalar(&ctx, 2); // {16, 25}
    defer squared.deinit();
    var y = try squared.scale(&ctx, 0.5); // {8, 12.5}
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 8, 12.5 }, try y.dataConst());
}
```

Inference-oriented helpers (the in-place and consuming ones reject
grad-requiring operands with `error.UnsupportedGradient`; `biasAdd` does
not — see its bullet):

- `addAxisVectorInPlace(ctx, bias, axis_tag)` — adds a `[axis_dim]` f32 row
  vector along the **last** axis `axis_tag`, mutating `self` in place.
- `addAxisVectorUnaryInPlace(ctx, op, bias, axis_tag)` — fused bias-add +
  `UnaryOp` activation, in place.
- `addScaledInPlace(ctx, other, alpha)` — `self += alpha·other` (same
  shape), in place.
- `takeAddNoGrad(ctx, other)` / `takeScaleNoGrad(ctx, s)` — **consume**
  `self` (it becomes `undefined`) and return the result, reusing `self`'s
  storage when the runtime can take it in place. They additionally fail with
  `error.ActiveExecScopeUnsupported` on a scope-owned borrow.
- `biasAdd(ctx, bias, axis_tag)` — the out-of-place variant: a new tensor,
  `self` unchanged. It accepts grad-requiring input: `self`'s gradient
  passes through identity; the raw `bias` slice receives none. For a
  trainable bias, use broadcast `add`.

### 4.4 Unary ops (`src/ag/tensor.zig`, `src/backend/ops.zig`)

```zig
pub fn unary(self: *const Self, ctx: *ExecContext, comptime op: UnaryOp) !Self
```

`exec.UnaryOp` is the closed kernel enum; most values also have a direct
method alias:

| `UnaryOp` | Method | Notes |
|---|---|---|
| `.relu` | `relu` | dedicated backward record |
| `.exp` | `exp` | |
| `.sqrt` | `sqrt` | |
| `.rsqrt` | `rsqrt` | |
| `.sigmoid` | `sigmoid` | |
| `.silu` | `silu` | |
| `.log` | `log` | |
| `.log1p` | `log1p` | `log(1 + x)` |
| `.softplus` | — (`unary(.softplus)` only) | `log(1 + e^x)`, sign-stable (torch softplus, pre-threshold regime) |
| `.neg` | `neg` | |
| `.abs` | `abs` | |
| `.sin` | `sin` | |
| `.cos` | `cos` | |
| `.tanh` | `tanh` | |
| `.fast_tanh` | `fastTanh` | NAM rational approximation |
| `.gelu` | `gelu` | tanh approximation (exact form of `.gelu_quant`) |
| `.quick_gelu` | `quickGelu` | `x·sigmoid(1.702x)` |
| `.softcap_30` | `softcap30` | `30·tanh(x/30)` logit softcap |
| `.softcap_15` | `softcap15` | `15·tanh(x/15)` logit softcap |
| `.gelu_quant` | — (`unary(.gelu_quant)` only) | ggml GGML_GELU_FP16 parity: f16-rounded tanh-gelu with hard clamps |
| `.elu` | `elu` | alpha = 1, matches `ggml_vec_elu_f32` |
| `.gelu_erf` | `geluErf` | exact-erf GELU (musl `erff` translation, matches `ggml_vec_gelu_erf_f32`) |
| `.floor` | `floor` | zero gradient a.e. (torch convention) |
| `.ceil` | `ceil` | zero gradient a.e. |
| `.round` | `round` | round-half-to-EVEN (torch.round), NOT half-away; 2^23 magic-number trick, scalar and SIMD legs bit-identical; zero gradient a.e. |
| `.sign` | `sign` | ±0 preserved, NaN propagates (numpy/torch); zero gradient a.e. |
| `.reciprocal` | `reciprocal` | `1/x`; output-derivative backward (`-out²`, like tanh) |

All unary ops are differentiable (§5). Related parameterized elementwise
ops:

- `leakyRelu(ctx, negative_slope)` — differentiable, dedicated backward.
- `clamp(ctx, min_value, max_value)` — differentiable; gradient is zero
  outside `[min, max]`.
- `dropout(ctx, p, seed)` — inverted dropout: keeps `x[i]/(1−p)` iff the
  per-element counter RNG at `(seed, i)` draws below `1−p`. The mask is
  never stored: forward, backward, and checkpoint recompute regenerate it
  from `(seed, index)`, so the op is a pure function of `(input, p, seed)`.
  Requires `0 <= p < 1`; `p == 0` returns an identity view. Pass a fresh
  seed per call (reusing a seed reuses the mask); eval mode is caller-side —
  do not call dropout at eval.

```zig
test "unary ops" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ -1, 0, 1 });
    defer x.deinit();
    var r = try x.relu(&ctx);
    defer r.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1 }, try r.dataConst());
    var s = try x.silu(&ctx); // same as x.unary(&ctx, .silu)
    defer s.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), (try s.dataConst())[2], 1e-6);
}
```

#### Elemental ops: user-defined scalar functions (`src/ag/elemental.zig`)

```zig
pub fn elementalUnary(self, ctx, comptime Op: type, extra: anytype) !Self
pub fn elementalBinary(self, ctx, other: anytype, comptime Op: type, extra: anytype)
    !Tensor(pointwiseResultTags(...))  // the standard pointwise tag rule
```

`UnaryOp` is a closed enum; `elementalUnary`/`elementalBinary` are the
user-extensible escape hatch — a convenience tier over `customVjp` (§5.6)
that lifts a comptime scalar `Op` to a differentiable eager tensor op. The
user writes scalar math only; the adapter owns buffer plumbing,
strided-input materialization, tag-driven broadcasting, broadcast-gradient
sum-reduction, `needs_grad` pruning, and the worker-team chunking of the
scalar loops (bitwise thread-count-neutral: disjoint pure writes).

<!-- snippet: helper -->
```zig
const Square = struct {
    pub fn forward(x: f32, extra: void) f32 {
        _ = extra;
        return x * x;
    }
    // Returns the propagated dL/dx, NOT the local dy/dx.
    pub fn backward(x: f32, y: f32, grad_y: f32, extra: void) f32 {
        _ = y;
        _ = extra;
        return 2 * x * grad_y;
    }
};
```

Binary `Op` declares `forward(a, b, extra)` plus `backwardA`/`backwardB`
returning dL/da and dL/db evaluated elementwise at the broadcast result
shape; broadcast operands get their gradient sum-reduced back to their own
tags/shape exactly like `add`/`mul`. Missing declarations are compile
errors. Both ops are f32-branch only, accept strided views, and return
owned contiguous results; `extra` is captured **by value** in the backward
node (the `customVjp` lifetime contract: pointees must outlive backward).
Validate a new `Op` with `fucina.gradcheck` (§5.7).

```zig
test "elementalUnary" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, -2, 3 });
    defer x.deinit();
    var y = try x.elementalUnary(&ctx, Square, {});
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 9 }, try y.dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, -4, 6 }, try gx.dataConst());
}
```

### 4.5 Gated activations (`src/ag/tensor.zig`)

```zig
pub fn gated(self, ctx, other: anytype, comptime op: GatedOp) !Tensor(...)  // pointwise result tags
pub fn splitGated(self, ctx, comptime op: GatedOp, comptime tag: Tag, comptime out_tag: Tag)
    !Tensor(replaceTag(tags, tag, out_tag))
```

`exec.GatedOp` is `{ .glu, .swiglu, .geglu, .swiglu_clamp10 }`. The
two-operand form computes `self * act(other)` — the **second** operand is
the gate — with the same tag-broadcast rule as §4.2: `glu` =
`self·sigmoid(other)`, `swiglu` = `self·silu(other)`, `geglu` =
`self·gelu(other)` (tanh approximation; Gemma's GeGLU).
`glu`/`swiglu`/`geglu` are direct aliases of `gated(..., op)`.
Differentiable in both operands. `.swiglu_clamp10` (DeepSeek V4's clamped
SwiGLU: the gate is `min(gate, 10)` before SiLU, `up` is clamped to
`[-10, 10]`) is inference-only — it has no backward and no split kernel,
so `gated` and `splitGated` reject it at compile time; it exists for the
MoE entries (§4.18).

`splitGated` halves axis `tag` and gates one half with the other in a single
fused kernel; the gate-half conventions differ deliberately (ggml parity):
`.swiglu` gates with the **first** half (`silu(first)·second`), `.glu` with
the **second** (`first·sigmoid(second)`). `out_tag == tag` is allowed.
`.geglu` is a compile error (no split-geglu kernel exists). Differentiable
in `self`.

```zig
test "gated pointwise and split-gated" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var up = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{1}, &.{2});
    defer up.deinit();
    var gate = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{1}, &.{1});
    defer gate.deinit();
    var y = try up.swiglu(&ctx, &gate); // up * silu(gate)
    defer y.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2 * 0.7310586), (try y.dataConst())[0], 1e-6);

    var fused = try fucina.Tensor(.{.ff}).fromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer fused.deinit();
    var z = try fused.splitGated(&ctx, .swiglu, .ff, .d); // silu(first half) * second half
    defer z.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586 * 2), (try z.dataConst())[0], 1e-6);
}
```

### 4.6 Masks, comparisons, and conditionals (`src/ag/tensor.zig`)

Mask producers emit `.bool` tensors (torch's comparison dtype); mask
consumers (`where`, `maskedFill`, the logical ops) take a `.bool` mask or
a float tensor read by truthiness (`!= 0`; NaN truthy). Like every typed
constant, `.bool` results are CALLER-owned even under an exec scope. Count
a mask with `sum`/`sumAll` (i64, §4.19) and cast with `to(.f32)` for the
mask-multiply idiom.

- `compare(ctx, op, other)` — `.bool`, true where `self <op> other`
  holds. `op` is `exec.CompareOp` (`.eq .ne .lt .le .gt .ge`); `other` is
  comptime-dispatched: a same-tagged tensor (same shape only) or a
  numeric scalar. **No-grad by design** (constant mask). NaN follows
  IEEE: any comparison involving NaN is false except `.ne`, which is
  true. Also on the typed branches: f16/bf16 compare through f32; INTEGER
  tensors compare natively (exact at any magnitude — token-id masks).
- `logicalAnd`, `logicalOr`, `logicalXor` (`ctx, other`) and
  `logicalNot(ctx)` — elementwise logic over truthiness, `.bool` out.
  Defined on the f32 branch (float `self`, `.bool`-or-float `other`) and
  on the `.bool` branch itself (mask combinators); same shape only.
- `where(ctx, cond, other)` — `cond[i] ? self[i] : other[i]`.
  Differentiable in `self` and `other`; `cond` (`.bool` or float) is a
  non-grad mask.
- `maskedFill(ctx, mask, value)` — `mask[i] ? value : self[i]`.
  Differentiable in `self` (gradient zeroed where filled); `value` is a
  constant.
- `isnan(ctx)` / `isinf(ctx)` / `isfinite(ctx)` — torch's float
  predicates as `.bool` masks, built purely from the IEEE `compare`
  semantics (`isnan` is the self-`.ne` test; `isfinite` is
  `-inf < x < inf`, false for NaN and both infinities).
  Non-differentiable, unscoped-safe.
- `any(ctx, tag)` / `all(ctx, tag)` — `.bool`, true where any/every
  element along `tag` is truthy (NaN is truthy, the torch.any/all
  convention), the tag removed; `anyAll(ctx)`/`allAll(ctx)` are the
  scalar full-tensor forms (torch with no dim). Non-differentiable
  (compare → i64 count → compare), unscoped-safe.

```zig
test "compare produces bool masks for maskedFill and where" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ -1, 0, 2 });
    defer x.deinit();
    var neg = try x.compare(&ctx, .lt, 0); // .bool: {true, false, false}
    defer neg.deinit();
    comptime std.debug.assert(@TypeOf(neg).dtype == .bool);
    var y = try x.maskedFill(&ctx, &neg, 0); // relu by hand
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 2 }, try y.dataConst());
    var n_neg = try neg.sumAll(&ctx); // count the mask: i64
    defer n_neg.deinit();
    try std.testing.expectEqual(@as(i64, 1), try n_neg.item());
}
```

### 4.7 Reductions and scans (`src/ag/tensor.zig`)

Axis reductions remove the reduced tag from the result type; `sumAll`
returns the scalar `Tensor(.{})`.

- `sum(ctx, tag)` / `mean(ctx, tag)` — differentiable.
- `sumMany(ctx, reduce_tags_spec)` — sums away several tags (innermost
  first); result tags `removeTags(tags, reduce_tags)`.
- `sumAll(ctx)` — full reduction to `Tensor(.{})`; read with `item()`.
- `variance(ctx, tag, ddof)` — `ddof: u1`: 0 = biased (LayerNorm
  convention), 1 = Bessel-corrected (`torch.var` default). Differentiable.
- `cumsum(ctx, tag)` — inclusive prefix sum, shape-preserving;
  differentiable (gradient = reversed suffix sum). Both passes are serial
  per row by default, so results are bitwise deterministic for any thread
  count AND sequence-exact; `-Dvector-scan` (§2.2) vectorizes both
  (non-last axes stay bitwise identical; the last axis reassociates like
  `sum`'s SIMD lanes).
- `max(ctx, tag)` / `min(ctx, tag)` — extremum values with the tag removed
  (indices come from `argmax`/`topK`, §4.16). The gradient flows only to the
  **first** occurrence of the extremum along the axis (strict-comparison
  tie-break, matching `torch.max` over a dim).
- `prod(ctx, tag)` — product with the tag removed (torch.prod over a
  dim), at `sum`'s kernel tier: rank-1 reduces through the pooled SIMD
  `prodInto`, a last-axis reduction runs one vectorized `prodSlice` per
  row (like `sum`, the SIMD lane order fixes the multiplication order per
  backend). Differentiable with torch's zero-handling: zero-free rows get
  `g·(Πx)/x_i`, exactly one zero routes the whole gradient to the zero
  slot, two or more kill the row's gradient.
- `cumprod(ctx, tag)` — inclusive running product, shape-preserving
  (torch.cumprod); serial per row by default, vectorized under
  `-Dvector-scan` (§2.2, the `cumsum` gating). Differentiable: zero-free rows use the
  O(n) reverse-scan closed form; rows containing a zero fall back to an
  exact division-free O(n²) expansion (torch semantics).
- `norm(ctx, tag, order)` / `normAll(ctx, order)` — vector norm along
  `tag` / over all elements (torch.linalg.vector_norm), `order` in
  `exec.NormOrder`: `.l1` = Σ|x|, `.l2` = sqrt(Σx²), `.inf` = max|x|.
  Composed from existing differentiable ops (scope-required under
  gradients); like torch, the `.l2` gradient at an all-zero vector is NaN
  (`sqrt'(0)`).

Dtype policy: on the f32 facade everything is f32 in and out. On the typed
constant tensors (§4.19) reductions widen per `outputDType(.reduction, ·)` —
f16/bf16 reduce into f32, f64 stays f64 — while pointwise and matmul keep
the input dtype; see §8 for the full dtype/storage matrix.

```zig
test "axis reductions and sumAll" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var s = try x.sum(&ctx, .col); // Tensor(.{ .row })
    defer s.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 7 }, try s.dataConst());
    var v = try x.variance(&ctx, .col, 0); // biased (ddof 0)
    defer v.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.25 }, try v.dataConst());
    var total = try x.sumAll(&ctx); // Tensor(.{}) scalar
    defer total.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 10), try total.item(), 1e-6);
}

test "cumsum keeps the axis" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var y = try x.cumsum(&ctx, .d);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 6 }, try y.dataConst());
}
```

### 4.8 `dot`: tag-directed contraction (`src/ag/tensor.zig`, `src/tagged.zig`)

```zig
pub fn dot(self: *const Self, ctx: *ExecContext, other: anytype, comptime contract_tag: Tag)
    !Tensor(dotResultTags(tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag))
// TensorObject unwraps pointer operands (`a.dot(&ctx, &b, .k)`)
```

`dot` is the workhorse contraction: it sums over the **named** tag
`contract_tag`, which must appear in both operands with equal dims. Tag
roles are decided at comptime:

- **contract** — `contract_tag`; removed from the result.
- **batch** — every other tag shared by both operands; dims must match
  exactly (batch tags do not broadcast).
- **free** — tags private to one operand.

The result tag order is `batch ++ left-free ++ right-free` (each group in
its operand's order). Because tags name axes, no `transpose` calls are ever
needed around `dot` — layout is handled by the lowering.

**Lowering**: `dot` is the single-contract-tag special case of `einsum` —
`taggedDot` delegates to `taggedEinsum` with the canonical dot result order
as the equation, so kernel selection is the einsum lowering's (§7.9): each
operand is aligned to the kernel layout as a zero-copy view, and at runtime
each side picks plain or transposed orientation by contiguity, so classic
layouts (`[m,k]·[k,n]`, NT weights `[out,in]`, batched `[b..,m,k]·[b..,k,n]`
and their trans permutations) dispatch straight to
`matmul2D`/`matmulTransA`/`matmulTransB`/`bmm`/`bmmTransA`/`bmmTransB` with
no data movement, vector operands ride along as size-1 GEMM axes, and a
full contraction runs `ctx.dot`. At most one of transA/transB is available
per call, so a layout where BOTH operands want their transposed orientation
(and no output-order swap covers it) materializes the smaller operand;
layouts no orientation can express materialize at most once per operand.

**Mixed-precision and quantized weights.** `other`'s dtype is
comptime-dispatched:

- f32 RHS: full two-operand backward (`DotBackward`).
- f16 / bf16 RHS (`ConstRhsDotBackward`): a CONSTANT RHS is a frozen weight
  — gradient flows to `self` only; a grad-requiring 16-bit RHS **variable**
  (§5.1) also receives its own gradient, as f32 (gradients are always f32;
  dW is the plain f32 einsum of the upstream gradient with the saved f32
  LHS). With no batch tags, one RHS free axis, and RHS storage
  `[free, contract]` (weight tags `{.out, .in}`-style) the forward hits the
  dedicated trans-B mixed kernels that widen in-register; otherwise the RHS
  is cast to f32 once and the f32 path runs.
- block-quantized RHS (q8_0, q4_k, ... — §10): the quantized-RHS GEMM;
  gradient to `self` only. Requires RHS storage `[free, contract]`, one RHS
  free axis, no batch tags (compile errors otherwise). When a GPU backend is
  active and `self` needs no gradient the GEMM may be offloaded (§9).

```zig
test "dot with a shared batch tag lowers to bmm" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{ .b, .m, .k }).fromSlice(&ctx, .{ 2, 1, 2 }, &.{ 1, 2, 5, 6 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .b, .k, .n }).fromSlice(&ctx, .{ 2, 2, 1 }, &.{ 3, 4, 7, 8 });
    defer b.deinit();
    var y = try a.dot(&ctx, &b, .k); // .b is shared (batch), result .{ .b, .m, .n }
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 83 }, try y.dataConst());
}

test "dot with an f16 constant RHS stays mixed-precision" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .t, .in }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1, 1 });
    defer x.deinit();
    var w = try fucina.Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } })
        .fromSlice(&ctx, .{ 1, 2 }, &.{ 2, 3 });
    defer w.deinit();
    var y = try x.dot(&ctx, &w, .in); // f32 result, [free, contract] fast path
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{5}, try y.dataConst());
}
```

#### `einsum` and `einsumMany`: multi-index contraction

```zig
pub fn einsum(self: *const Self, ctx: *ExecContext, other: anytype, comptime out_tags: anytype)
    !Tensor(normalizeTags(out_tags))
pub fn einsumMany(ctx: *ExecContext, comptime out_tags: anytype, operands: anytype)
    !Tensor(normalizeTags(out_tags))   // free function at the fucina root
```

`einsum` generalizes `dot` from one contraction tag to a whole Einstein
equation. Because both operands already carry named axes, the output tag
tuple **is** the equation:

```
result[out_tags] = Σ over every tag not in out_tags of self ⊙ other
```

Tag roles are decided at comptime purely from membership:

- **batch** — shared tags kept in `out_tags`; dims must match exactly.
- **contract** — shared tags dropped from `out_tags`; dims must match
  (`ShapeMismatch` otherwise). Any number of contraction tags.
- **free** — operand-private tags kept in `out_tags`.
- **summed** — operand-private tags dropped from `out_tags`; summed away
  before the contraction (their gradient is a broadcast).

The result axis order is exactly `out_tags` (unlike `dot`, whose result
order is fixed); every output tag must exist in an operand (compile error
`einsum output tag not found in any operand`). `self` is f32; an f16/bf16 `other`
is widened to f32 once per call (forward and backward) — a constant RHS
routes gradient to `self` only, and a grad-requiring 16-bit RHS variable
also receives its own f32 gradient, exactly dot's widened fallback
contract — and
a quantized `other` is a compile error directing to `dot`, whose packed
kernels require the `[free, contract]` weight layout. Duplicate tags within
one operand remain impossible, so there are no trace/diagonal semantics. `dot(other, .k)` and
`einsum(other, dotResultTags(...))` compute the same thing; `einsum` is the
one to reach for the moment an equation has several contraction axes,
several free axes on one side, or a specific output order.

**Lowering** (`taggedEinsum`, §7.9): summed-private axes are pre-reduced,
then both operands are aligned (zero-copy permute views) to an
`out_tags`-derived group-nested order and each side independently picks the
plain or transposed kernel layout at runtime — the orientation whose aligned
view is already contiguous wins, because a trans GEMM is free while
materializing costs a copy pass. Classic layouts therefore dispatch to
`matmul2D`/`bmm` (and trans variants) with zero copies; at most one of
transA/transB is available per call (both-want-trans materializes the
smaller operand), and layouts no orientation can express materialize at
most once per operand. When
`out_tags` nests as `[batch][right free][left free]`, the operands swap
kernel roles, so "double-transposed" layouts (e.g. `x[k,m] · y[n,k] ->
[n,m]`) run as one plain GEMM with zero copies. An `out_tags` order that
interleaves the three groups costs one extra output materialization —
prefer group-nested orders.

**Backward.** Contractions are closed under differentiation: each operand's
gradient is another einsum (the output gradient contracted with the other
operand), broadcast over any forward-summed axes — so both VJP branches
stay on GEMM kernels for every tag structure (`EinsumBackward`, which
`DotBackward` delegates to, and its const-RHS variant
`ConstRhsEinsumBackward`, which `ConstRhsDotBackward` delegates to). This
retired the old
broadcast-multiply backward fallback for dots with more than one free tag
on the opposite side (`zig build bench-einsum` measured that case at two
orders of magnitude).

`einsumMany` folds two or more operands left-to-right through binary
`einsum`, keeping at each step exactly the tags still needed by the
remaining operands or the output. Contraction order is the operand order —
order the tuple so early intermediates stay small. As with other composed
facade ops, tracking gradients through it requires an active exec scope
(§6.3; `error.ActiveExecScopeRequired`).

```zig
test "einsum: one equation for a grouped-attention-style contraction" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // q[g, i, d] x k[j, d] -> scores[g, i, j]: two free axes on the left and
    // an NT-layout right operand, contracted over .d in one equation.
    var q = try fucina.Tensor(.{ .g, .i, .d }).fromSlice(&ctx, .{ 2, 1, 2 }, &.{ 1, 2, 3, 4 });
    defer q.deinit();
    var k = try fucina.Tensor(.{ .j, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 0, 0, 1 });
    defer k.deinit();
    var scores = try q.einsum(&ctx, &k, .{ .g, .i, .j });
    defer scores.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, try scores.dataConst());
}

test "einsum: dropped tags are summed — shared become contractions, private are pre-summed" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // y[n] = sum over s and k of a[s,k] * b[k,n]: .k is shared (contraction),
    // .s is private to `a` and simply summed away.
    var a = try fucina.Tensor(.{ .s, .k }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .k, .n }).fromSlice(&ctx, .{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer b.deinit();
    var y = try a.einsum(&ctx, &b, .{.n});
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 62, 72 }, try y.dataConst());
}

test "einsumMany: a LoRA delta as one three-operand equation" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .s, .i }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1, 1 });
    defer x.deinit();
    var a = try fucina.Tensor(.{ .r, .i }).fromSlice(&ctx, .{ 1, 2 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .o, .r }).fromSlice(&ctx, .{ 2, 1 }, &.{ 1, -1 });
    defer b.deinit();

    // x[s,i] · A[r,i] · B[o,r] -> [s,o], contraction order = operand order.
    var y = try fucina.einsumMany(&ctx, .{ .s, .o }, .{ &x, &a, &b });
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, -5 }, try y.dataConst());
}
```

### 4.9 Explicit matmul, ternary STE, and packed-RHS GEMMs (`src/ag/tensor.zig`)

```zig
pub fn matmul(self, ctx, other: anytype, comptime kind: exec.MatmulKind, comptime out_tags: anytype)
    !Tensor(out_tags)
```

`matmul` bypasses the tag algebra: the caller names the result axes and
picks `exec.MatmulKind` (`.plain`, `.trans_a`, `.trans_b`). Routing is
comptime on rank: both operands rank-2 → the 2-D GEMM entries (`.plain`:
`[m,k]·[k,n]`; `.trans_b`: `[m,k]·[n,k]ᵀ`); anything else → the batched bmm
entries with stride-0 broadcast leading batch axes (mixed-rank operands
broadcast rather than error). `.trans_a` exists only on the batched path —
rank-2 `.trans_a` is a compile error directing to `dot`, whose tag algebra
reaches the 2-D trans-A kernel. f32 only, full two-operand gradients. Unlike
`dot` there is no materialize fallback: the operands' storage order **is**
the kernel layout.

```zig
test "explicit matmul with a transposed RHS" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .m, .k }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1, 2 });
    defer x.deinit();
    var w = try fucina.Tensor(.{ .n, .k }).fromSlice(&ctx, .{ 2, 2 }, &.{ 3, 4, 5, 6 });
    defer w.deinit();
    var y = try x.matmul(&ctx, &w, .trans_b, .{ .m, .n }); // [m,k] · [n,k]ᵀ
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 17 }, try y.dataConst());
}
```

**`dotTernarySte(ctx, weight, contract_tag)`** — trainable ternary linear
(BitNet b1.58 straight-through estimator). Every forward encodes the f32
latent `weight` (tags `{.out, .in}`, per-tensor absmean scale, round-clip to
{−1, 0, +1}) to TQ2_0 and contracts with the mul-free kernel. Backward: `dx`
flows through the **quantized** weight; `dW` is the straight-through
estimate (plain matmul VJP against the latent weight). The contract dim must
be a multiple of 256 (`error.TernaryContractDimNotBlockAligned` otherwise);
no shared batch tags, one weight free axis, weight storage
`[free, contract]`, and lhs storage `[..., contract]` (contract tag last) —
all four are compile errors. See `TERNARY.md` and §10/§11.

```zig
test "dotTernarySte encodes the latent weight per call" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const k = 256; // contract dim must be a multiple of the TQ2_0 block size
    const x_values = [_]f32{1} ** k;
    var x = try fucina.Tensor(.{ .t, .in }).fromSlice(&ctx, .{ 1, k }, &x_values);
    defer x.deinit();
    var w = try fucina.Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 1, k }, &x_values);
    defer w.deinit();
    var y = try x.dotTernarySte(&ctx, &w, .in); // absmean scale 1, all-ones encode
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{k}, try y.dataConst());
}
```

**Packed-RHS entries** (quantization detail in §10):

- `dotPacked(ctx, rhs, contract_tag, out_tag)` — 2-D `[free, contract]` lhs
  against a pre-packed quantized RHS container (`*const` q8_0x4 / q6_kx4 /
  q4_kx8 / q4_kx2mmla / q5_kx8, comptime-dispatched from the pointer type).
  No gradients: fails with `error.GradientQuantizedMatmulUnsupported`.
- `rmsNormMulDotPacked(ctx, norm_weight, eps, rhs, contract_tag, out_tag)` —
  fused `rmsNormMul(self, norm_weight) · rhsᵀ` without materializing the
  normalized tensor (`self` is the pre-norm `[free, contract]` input,
  `norm_weight` the `[contract]` scale row); matches `rmsNormMul` +
  `dotPacked` to ≤ 1 ulp (q8_0x4 / q4_kx8 / q5_kx8 / q6_kx4; q4_kx2mmla is
  a deliberate compile error — MMLA targets use the unfused path).
- `splitSwiGluDotPacked(ctx, rhs, split_tag, out_tag)` — fused split-SwiGLU
  + packed down-projection GEMM without materializing the gated tensor
  (q8_0x4 / q4_kx8 / q5_kx8 / q6_kx4; q4_kx2mmla is a deliberate compile
  error — MMLA targets use the unfused `splitGated` + `dotPacked`).
- `gegluQuantDotPacked(ctx, up, rhs, in_tag, out_tag)` — fused
  `(up · geluQuant(self)) @ rhs`; q8_0x4 only.
- On block-quantized tensors: `packRhs(ctx)` packs a rank-2 weight into the
  ISA-best layout for its dtype — q8_0→x4, q6_k→x4, q5_k→x8, q4_k→x2mmla on
  aarch64+i8mm targets else x8 (the return type is
  `fucina.PackedRhs(dtype)`); `packRhsLayout(ctx, layout)` forces a specific
  `fucina.PackedRhsLayout` instead.

### 4.10 Softmax family (`src/ag/tensor.zig`, `src/exec/softmax.zig`)

```zig
pub fn softmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag, options: anytype) !Self
```

Softmax over `tag`, shape-preserving. `options` is a comptime-validated
struct literal (unknown fields are compile errors); an empty `.{}` routes to
the lean plain kernel. The fused extensions mirror ggml's `soft_max_ext`
(the exec-level option struct is `exec.SoftmaxExtOptions`); the effective
pre-softmax logit is `x·scale + slope·mask`:

- `.scale = s` — logit multiplier (attention `1/sqrt(d)` without a separate
  pass).
- `.mask = &m` — **additive** tag-broadcast tensor (not −inf masking): the
  mask is aligned to `self`'s tags and expanded by zero-stride broadcast, so
  a `[q, k]` mask serves every head of a `[head, q, k]` score tensor. The
  mask must not require grad (`error.UnsupportedGradient`).
- `.max_bias = b` with `.head_tag` — ALiBi: a per-head slope multiplies the
  mask, following the ggml slope schedule (powers of `2^(−b/h)` with `h` the
  head count rounded down to a power of two; `src/exec/shape.zig`
  `alibiSlope`). Requires `.mask` and `.head_tag` (`InvalidShape`
  otherwise).
- `.sinks = slice` — per-head attention sinks: one extra logit per head that
  joins the running max and the denominator only, so row probabilities sum
  to less than 1 (the sink absorbs the remaining mass). Needs `.head_tag`
  (one sink per head; a single-element slice is accepted without one).
- `.causal = .{ .query_tag, .source_offset }` — fused causal masking: query
  row `q` normalizes over sources `[0, source_offset + q]`; positions beyond
  are exactly 0. Validates `source_offset + query_dim <= source_dim`.

Backward: `SoftmaxBackward`/`SoftmaxExtBackward` (§5) — the unified backward
re-derives from the output and `scale` (mask/sinks/ALiBi contribute no
gradient).

Two log-domain companions (max-shifted for stability, torch semantics),
FUSED single-node kernels sharing softmax's row machinery — SIMD max scan
+ vexpf sum per row, task-parallel over rows, scalar strided fallback on
non-last axes; no materialized intermediates and no exec-scope
requirement:

- `logsumexp(ctx, tag)` — `log(Σ exp(x))` with the tag removed
  (torch.logsumexp). Rows whose max is ±inf are shifted by 0 instead, so
  an all(-inf) row yields -inf and a +inf entry yields +inf rather than
  NaN (the torch convention). Backward is the saved-output identity
  `exp(x − lse)·g` (the row softmax).
- `logSoftmax(ctx, tag)` — `x − logsumexp(x)` broadcast, shape-preserving
  (torch.log_softmax), same non-finite-max handling. Prefer
  `crossEntropy` when the next step is an NLL loss (fused with the loss,
  saved-stats backward). Backward is the saved-output identity
  `g − exp(y)·Σg`.

```zig
test "softmax and the fused causal extension" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .seq, .src }).zeros(&ctx, .{ 2, 2 });
    defer x.deinit();
    var p = try x.softmax(&ctx, .src, .{});
    defer p.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.5, 0.5, 0.5 }, try p.dataConst());

    var c = try x.softmax(&ctx, .src, .{ .causal = .{ .query_tag = .seq } });
    defer c.deinit();
    // query 0 attends source 0 only; masked-out tail is exactly 0
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0.5, 0.5 }, try c.dataConst());
}
```

### 4.11 Normalization family (`src/ag/tensor.zig`)

All norms normalize over one named tag and preserve shape; all are
differentiable in every tensor operand (statistics are recomputed in the
backward — nothing extra is saved from the forward).

- `rmsNorm(ctx, tag, eps)` — `x / sqrt(mean(x²) + eps)`.
- `rmsNormMul(ctx, tag, weight, eps)` — fused `rmsNorm(x)·weight`;
  `weight: *const Tensor(.{tag})`.
- `rmsNormMulAdd(ctx, tag, weight, residual, eps)` — fused
  `rmsNorm(x)·weight + residual` (`residual` same tags as `self`).
- `rmsNormMulRopeHalfPrepared(ctx, position_tag, feature_tag, weight, eps, table)`
  — fused rmsNorm·weight followed by half-mode RoPE from a prepared
  `*const exec.RopeTable` (the QK-norm + RoPE step of the model families'
  attention blocks, §14). The inference-only `rmsNormMulDotPacked`
  (rmsNormMul fused into a packed quantized GEMM) is documented with the
  packed-RHS entries in §4.9.
- `layerNorm(ctx, tag, eps, options)` — PyTorch semantics:
  `(x − μ)/sqrt(σ² + eps)` with biased variance. `options` is `.{}` for the
  plain form or `.{ .weight = &w, .bias = &b }` for the fused affine — the
  fused kernel requires **both** together (compile error otherwise).
  Weight/bias are rank-1 `[tag_dim]` tensors, either tagged `.{tag}`
  (comptime-checked) or numeric-tag `Tensor(1)` values (`._0`; runtime
  length check).
- `groupNorm(ctx, channel_tag, groups, eps, weight, bias)` — ggml GroupNorm
  over rank-2 `[time, channel]` storage: per channel group, f64-accumulated
  mean and biased variance over all `time × (C/groups)` elements, eps inside
  the sqrt, then the optional per-channel affine (`?*const Tensor(.{channel_tag})`,
  independently optional) applied after normalization.
- `standardizeAxis(ctx, tag, options)` — `(x − mean)/denom` over `tag`.
  `options` accepts every `exec.StandardizeOptions` field plus an optional
  `.valid_len` (unknown fields are compile errors): standardize only the
  first `valid_len` elements; the suffix is returned as zeros and receives
  zero gradient. `StandardizeOptions`: `ddof: u1 = 0`, `eps: f32 = 0`,
  `eps_mode: StandardizeEpsMode = .outside_sqrt` (`sqrt(var) + eps`,
  Parakeet's frontend convention) or `.inside_sqrt` (`sqrt(var + eps)`,
  LayerNorm placement), `accumulation: StandardizeAccumulation = .f32` or
  `.f64`.
- `l2Normalize(ctx, tag, eps)` — `x·rsqrt(Σx² + eps)`. The eps is added to
  the **squared** norm (the rmsNorm convention) — deliberately not torch
  `F.normalize`'s `x/max(‖x‖₂, eps)`. Composed op: requires an active exec
  scope when gradients are tracked (§4.1).
- `snake(ctx, channel_tag, alpha, inv_b)` — per-channel Snake activation
  (DAC codec): `y = x + inv_b[c]·sin(alpha[c]·x)²` over `[time, channel]`
  storage. `alpha` and `inv_b` are independent operands at this level; no
  gradient flows through the loader's `inv_b = 1/(alpha + 1e-9)` relation.

```zig
test "layerNorm and rmsNorm" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .t, .d }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1, 3 });
    defer x.deinit();
    var ln = try x.layerNorm(&ctx, .d, 0, .{}); // (x - mean) / sqrt(biased var + eps)
    defer ln.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -1, 1 }, try ln.dataConst());

    var rn = try x.rmsNorm(&ctx, .d, 0); // x / sqrt(mean(x²) + eps)
    defer rn.deinit();
    const rms = @sqrt((1.0 + 9.0) / 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0 / rms), (try rn.dataConst())[1], 1e-6);
}

test "standardizeAxis zero-mean unit-variance" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .t, .d }).fromSlice(&ctx, .{ 1, 4 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var y = try x.standardizeAxis(&ctx, .d, .{}); // ddof 0, eps 0
    defer y.deinit();
    const out = try y.dataConst();
    try std.testing.expectApproxEqAbs(@as(f32, 0), out[0] + out[1] + out[2] + out[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5 / @sqrt(1.25)), out[3], 1e-6);
}
```

### 4.12 Rotary position embedding (`src/ag/tensor.zig`, `src/exec/rope.zig`)

```zig
pub fn rope(self, ctx, comptime position_tag: Tag, comptime feature_tag: Tag,
            source: anytype, comptime mode: RopeMode) !Self
```

Rotates feature pairs by position-dependent angles over
(`position_tag`, `feature_tag`). `mode` is comptime `exec.RopeMode`:
`.half` pairs feature `i` with `i + d/2` (NEOX/Llama layout);
`.interleaved` pairs adjacent features. `source` selects the factor source
at comptime (a closed set; anything else is a compile error):

- `*const exec.RopeTable` — prepared factors, the production path. Build
  with `ctx.prepareRopeTable(positions, feature_dim, theta_base, inverse)`
  or `ctx.prepareRopeTableFactors(..., freq_factors)` — the latter is ggml's
  `rope_ext` frequency scaling (Llama-3 long-context, Gemma global layers).
  The table's `feature_dim` is the **authoritative rotary span**: equal to
  `dim(feature_tag)` rotates fully; smaller rotates the leading
  `feature_dim` features and passes the tail through unchanged (partial NEOX
  RoPE). `RopeTable` owns its buffers; `table.deinit()` releases them.
- `exec.RopeTheta` / `.{ .positions = p, .theta_base = t }` — on-the-fly
  factors (`positions: []const i32`), full rotation only. Pair `i` at
  position `p` rotates by `p / theta_base^(2i/d)`.

Differentiable in `self`; the backward applies the inverse rotation (§5).
Negative positions rotate backwards, so re-roping cached values to a new
offset is a valid pattern.

```zig
test "rope rotates feature pairs by position" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .seq, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 0, 1, 0 });
    defer x.deinit();
    var y = try x.rope(&ctx, .seq, .d, .{ .positions = &.{ 0, 1 }, .theta_base = 10000 }, .half);
    defer y.deinit();
    const out = try y.dataConst();
    try std.testing.expectApproxEqAbs(@as(f32, 1), out[0], 1e-6); // position 0: identity
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1)), out[2], 1e-6); // position 1: angle 1 rad
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1)), out[3], 1e-6);
}
```

### 4.13 Attention (`src/ag/tensor.zig`)

```zig
pub fn groupedAttention(self, ctx, k: anytype, v: anytype, kv_head_for_head: []const usize,
                        comptime out_tag: Tag, scale_value: f32, opts: anytype)
    !Tensor(.{ .seq, out_tag })
```

Grouped-query (GQA) flash-style attention. `self` is the query and **must**
be tagged `.{ .seq, .head, .d }` (compile error otherwise); the result is
`[seq, head·d]` tagged `.{ .seq, out_tag }`. `kv_head_for_head[h]` maps each
query head to its KV head. `scale_value` multiplies the scores. The KV
representation is comptime-dispatched from `@TypeOf(k)` (k and v must
match):

| `k`/`v` type | Use case | Gradients |
|---|---|---|
| `*Tensor(.{ .seq, .kv_head, .d })` (f32) | training / f32 caches | full q/k/v backward (windowed re-masks to the window) |
| same tags, f16 | decode KV cache | q-grad only (K/V widened to f32 once for the backward) |
| `[]const BlockQ8_0` | q8_0 raw-block cache, layout `[kv_seq, kv_heads, d/32]` | q-grad only, causal only |
| `[]const []const f16` | ragged multi-stream decode | inference-only |
| `[]const []const BlockQ8_0` | ragged multi-stream decode (q8_0) | inference-only |

`opts` is a comptime-validated struct literal with a per-representation
field whitelist (a misspelled option is a compile error, never
silently-full-causal attention):

- `.mask = .causal` (default) | `.bidirectional` — f32/f16 KV only.
  Bidirectional has no windowed kernel by design (realize SWA reach by
  narrowing the K/V views).
- `.window = w` — runtime sliding window, 0 = full causal (query `p`
  attends `[max(0, p−w+1), p]`); causal only.
- `.bias = &b` — rank-2 `[q_seq, kv_seq]` additive f32 bias on the scaled
  scores pre-softmax (ggml `soft_max_ext` semantics); bidirectional + f32 KV
  only; inference-only — any grad-requiring operand returns
  `error.UnsupportedGradient`.
- `.kv_seq = n, .kv_heads = h` — required for the q8_0-block representation
  (raw blocks carry no shape).
- `.lens = lens, .kv_heads = h` — required for the multi-stream
  representations; q's `.seq` tag is reinterpreted as the **stream** axis
  (one query row per stream, row `s` attending `lens[s]` cached positions);
  per-stream results are bit-identical to N single-stream calls.

Related: `relposShift(ctx, t_k, out_tags)` — Transformer-XL relative-shift
("skew") of a rank-3 `[H, Tq, P]` score tensor to `[H, Tq, Tk]` with
`out[h,q,j] = self[h, q, j+(Tq−1)−q]` (`P >= Tk+Tq−1`); differentiable
(scatter VJP).

```zig
test "groupedAttention over a single cached position returns v" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var q = try fucina.Tensor(.{ .seq, .head, .d }).fromSlice(&ctx, .{ 1, 1, 2 }, &.{ 1, 0 });
    defer q.deinit();
    var k = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 1, 1, 2 }, &.{ 1, 1 });
    defer k.deinit();
    var v = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 1, 1, 2 }, &.{ 5, 7 });
    defer v.deinit();

    var y = try q.groupedAttention(&ctx, &k, &v, &.{0}, .out, 1.0, .{});
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 7 }, try y.dataConst());
}
```

### 4.14 Convolution and channel-last vision ops (`src/ag/tensor.zig`)

**1-D family** (input storage `[time, channel]`, enforced at comptime; bias
is deliberately not fused — compose it with broadcast `add`):

- `conv1d(ctx, time_tag, in_tag, tap_tag, out_tag, weight, stride, padding, dilation, groups)`
  — PyTorch `Conv1d` semantics (cross-correlation); `weight` is
  `*const Tensor(.{ tap_tag, in_tag, out_tag })` stored `[tap, in/groups, out]`;
  result `[t_out, out]` with
  `t_out = (T + 2·pad − dilation·(taps−1) − 1)/stride + 1`. Differentiable
  in input and weight.
- `causalConv1d(ctx, time_tag, in_tag, tap_tag, out_tag, weight, dilation, state)`
  — causal orientation (tap `taps−1` is the newest sample); `state`, when
  given, supplies the `dilation·(taps−1)` input rows preceding `x` (oldest
  first, `[row, in]`); absent rows read as zeros; no gradient into `state`.
- `groupedCausalConv1d(ctx, time_tag, in_tag, tap_tag, in_per_group_tag, out_tag, weight, dilation, groups, state)`
  — grouped variant, weight `[tap, in_per_group, out]`.
- `causalDepthwiseConv1d(ctx, time_tag, channel_tag, tap_tag, kernel, state)`
  — depthwise; `kernel: *const Tensor(.{ channel_tag, tap_tag })`.
- `convTranspose1d(ctx, time_tag, in_tag, kout_tag, out_tag, weight2, bias, out_channels, taps, stride, padding, output_pad)`
  — GEMM + col2im_1d (ggml decomposition); `weight2` is the load-time
  repacked `[K·OC, IC]` matrix (k fastest within each oc block), `bias` is
  `?*const Tensor(.{out_tag})`. The `output_pad` trailing rows are
  bias-only (ggml/omnivoice convention, not exact PyTorch when pad > 0).
  Differentiable in input, weight2, and bias; the weight gradient is with
  respect to the **packed** layout.

**Channel-last 2-D family** (rank-3 `[H, W, C]` inputs; used by §14's face
detection stack):

- `conv2d(ctx, weight, bias, stride, padding, groups, out_tags)` — `weight`
  rank-4 `[Cout, kH, kW, Cin/groups]`, `bias` is `null` or rank-1 `[Cout]`;
  result `[oH, oW, Cout]` tagged `out_tags`. Differentiable in all tensor
  operands.
- `conv2dRelu(...)` — conv2d with the relu fused into the epilogue on the
  no-grad path (identical values to `conv2d` then `relu`; on the Winograd
  route it folds into the output transform). Falls back to the
  differentiable composition when any operand requires gradients.
- `prepareConv2dWeights(ctx)` — on the rank-4 weight: load-time Winograd
  F2/F4 weight-transform planes, built once so the prepared entries below
  skip the per-call weight transform. Returns `fucina.PreparedConvWeights`
  (caller `deinit`s); a weight that can never take the Winograd route
  returns `.empty`, which is inert on every conv route. No gradient
  support (the `dotPacked` policy — prepared planes live outside the
  graph): fails with `error.GradientPreparedConv2dUnsupported` on a
  grad-requiring weight.
- `conv2dPrepared(ctx, weight, prepared, bias, stride, padding, groups, out_tags)`
  — no-grad conv2d against the prepared planes: bitwise-identical values to
  `conv2d`, minus the per-call weight transform on the Winograd route
  (every other route ignores `prepared`). Fails with
  `error.GradientPreparedConv2dUnsupported` when any operand requires grad.
- `conv2dPreparedRelu(...)` — `conv2dPrepared` with the relu fused into the
  conv epilogue; same no-grad contract.
- `maxPool2d(ctx, kernel, stride, padding)` — `[h, w]`-ordered params; the
  zero-pad border reads as −inf. `avgPool2d(...)` averages valid taps only
  (ONNX `count_include_pad=0`). Both differentiable.
- `upsample2xNearest(ctx)` — `[H, W, C] → [2H, 2W, C]`; VJP = 2×2 stride-2
  sum-pool.
- `prelu(ctx, alpha)` — learnable per-channel slope, `alpha` rank-1 `[C]`
  (channel innermost); differentiable in `x` and `alpha`.
- `channelAffine(ctx, scale_t, shift_t)` — fused per-channel
  `x·scale[c] + shift[c]` (frozen-stats inference BatchNorm); differentiable
  in all three.

```zig
test "conv1d computes a moving weighted sum" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .time, .in }).fromSlice(&ctx, .{ 3, 1 }, &.{ 1, 2, 3 });
    defer x.deinit();
    var w = try fucina.Tensor(.{ .tap, .in, .out }).fromSlice(&ctx, .{ 2, 1, 1 }, &.{ 1, 1 });
    defer w.deinit();
    var y = try x.conv1d(&ctx, .time, .in, .tap, .out, &w, 1, 0, 1, 1);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 5 }, try y.dataConst());
}

test "conv2d channel-last and maxPool2d" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .h, .w, .c }).fromSlice(&ctx, .{ 2, 2, 1 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var weight = try fucina.Tensor(.{ .oc, .kh, .kw, .ic }).fromSlice(&ctx, .{ 1, 1, 1, 1 }, &.{2});
    defer weight.deinit();
    var y = try x.conv2d(&ctx, &weight, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .oh, .ow, .oc });
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6, 8 }, try y.dataConst());

    var pooled = try x.maxPool2d(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 });
    defer pooled.deinit();
    try std.testing.expectEqualSlices(f32, &.{4}, try pooled.dataConst());
}
```

### 4.15 Losses and similarity (`src/ag/tensor.zig`, `src/exec/loss.zig`)

**Cross-entropy** (fused log-softmax + NLL over a named class tag; labels
are `[]const usize`, one per position, positions ordered with the class axis
removed, remaining axes row-major):

```zig
pub fn crossEntropy(self, ctx, comptime class_tag: Tag, labels: []const usize) !Tensor(.{})
pub fn crossEntropyExt(self, ctx, comptime class_tag: Tag, labels: []const usize,
                       comptime options: exec.CrossEntropyOptions)
    !Tensor(if (options.reduction == .none) removeTag(tags, class_tag) else .{})
```

`exec.Reduction` is `{ .mean, .sum, .none }`. `exec.CrossEntropyOptions`
(PyTorch parity): `ignore_index: ?usize = null` (matching positions
contribute zero loss/gradient and leave the `.mean` denominator; when
*every* position is ignored the loss is 0, a deliberate divergence from
PyTorch's NaN), `reduction: Reduction = .mean` (`.none` returns per-position
losses with the class tag removed), `label_smoothing: f32 = 0` (in `[0,1)`,
PyTorch semantics). Labels must be `< class_count` or equal to
`ignore_index` (`IndexOutOfBounds` otherwise). Differentiable in the logits.

**Fused linear + cross-entropy** — `crossEntropyExt(self·weightᵀ)` as ONE
differentiable op:

```zig
pub fn linearCrossEntropyExt(self, ctx, weight: anytype, labels: []const usize,
                             comptime options: exec.CrossEntropyOptions)
    !Tensor(if (options.reduction == .none) removeTag(tags, tags[1]) else .{})
```

`self` is rank-2 `[row, shared]` and `weight` rank-2 `[class, shared]`
(both f32, shared tag last on both, the class tag absent from `self` —
all comptime-checked). The logits exist only inside the op: they are
computed once and saved on the backward record with the forward's per-row
softmax statistics, and the VJP folds block-built probability panels
straight into dx and dweight, so the `[rows, classes]` logit **gradient**
is never materialized. Differentiable in **both** operands; same
options/reduction contract as `crossEntropyExt` (`.none` returns per-row
losses tagged by the row tag).

**Elementwise losses** vs a same-tagged `target`, all differentiable in
**both** operands, all sharing the reduction/result-type contract above
(`.mean` divides by the **total** element count):

- `mseLoss(ctx, target, options)` — `exec.MseOptions{ .reduction }`.
- `huberLoss(ctx, target, options)` — `exec.HuberOptions{ .delta = 1.0, .reduction }`;
  quadratic for `|x−t| <= delta`, linear beyond.
- `bceLoss(ctx, target, options)` — `exec.BceOptions{ .reduction, .from_logits = false }`.
  With `from_logits` the input is a raw logit and the stable
  `max(x,0) − x·y + log1p(exp(−|x|))` form is used; otherwise a probability
  clamped to `[bce_eps, 1−bce_eps]` (`exec/loss.zig`'s `bce_eps = 1e-7`;
  gradient defined as 0 outside the open interval — deliberate divergence
  from torch's huge boundary gradients).
- `klDivLoss(ctx, target, options)` — `exec.KlDivOptions{ .reduction, .log_target = false }`.
  `self` holds **log**-probabilities; no `.batchmean` exists — `.mean` is
  torch's total-element mean, `.sum` the mathematical divergence.

**Composed** (require an active exec scope when gradients are tracked,
§4.1):

- `nllLoss(ctx, class_tag, labels, comptime reduction)` — NLL over
  **log-probabilities** (one-hot → mul → sum → negate). Prefer
  `crossEntropy`/`crossEntropyExt` when starting from logits.
- `cosineSimilarity(ctx, other, tag, eps)` — torch `F.cosine_similarity`:
  `Σxy / max(‖x‖·‖y‖, eps)` with `tag` reduced away; differentiable in both.

```zig
test "crossEntropy on uniform logits is ln(K)" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var logits = try fucina.Tensor(.{ .batch, .class }).zeros(&ctx, .{ 1, 4 });
    defer logits.deinit();
    var loss = try logits.crossEntropy(&ctx, .class, &.{2});
    defer loss.deinit();
    try std.testing.expectApproxEqAbs(@log(@as(f32, 4)), try loss.item(), 1e-6);

    var per_pos = try logits.crossEntropyExt(&ctx, .class, &.{2}, .{ .reduction = .none });
    defer per_pos.deinit(); // Tensor(.{ .batch }): class tag removed
    try std.testing.expectApproxEqAbs(@log(@as(f32, 4)), (try per_pos.dataConst())[0], 1e-6);
}

test "mseLoss mean over all elements" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer x.deinit();
    var t = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 3, 2 });
    defer t.deinit();
    var loss = try x.mseLoss(&ctx, &t, .{});
    defer loss.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2), try loss.item(), 1e-6); // ((-2)² + 0) / 2
}
```

### 4.16 Selection: argmax, topK, sort, routerTopK (`src/ag/tensor.zig`, `src/exec/topk.zig`)

Index outputs across the library are constant **i64** tensors (the
repo-wide index convention — torch's index dtype, exact for any axis
length) and carry no gradient. As typed constants they are CALLER-owned
even under an exec scope (§6.3): pair them with `deinit` — an f32
`values` arm of the same call remains a scope-owned borrow.

- `argmax(ctx, tag)` — indices of the per-row maximum, tag removed, as
  `Tensor(.{ .dtype = .i64, .tags = ... })`. **No-grad by design** (like
  sampling).
- `topK(ctx, tag, k, out_tag)` — returns `TopKResult(replaceTag(tags, tag, out_tag))`,
  a struct of `values` and `indices` with a single `deinit()` releasing
  both. `values` **is** differentiable (the gradient scatters back through
  the saved indices); `indices` is a constant i64 tensor.
- `sort(ctx, tag, descending)` — full sort (`TopKResult(tags)`): values +
  source index per output position. **Unstable** sort; NaN sorts **last**
  regardless of direction (documented divergence from `torch.sort`, which
  puts NaN first when descending). Values differentiable, indices constant.
- `argsort(ctx, tag, descending)` — the indices arm alone (i64); no-grad.
- `routerTopK(ctx, expert_tag, k, options, selected, weights)` — the MoE
  router primitive: fills caller-provided `selected: []usize` /
  `weights: []f32` (both `rows·k` long) with the per-row top-k experts and
  their softmax probabilities computed over the **full** expert axis.
  `exec.RouterTopKOptions{ .normalize_selected: bool = true }` renormalizes
  the selected mass to sum to 1. Requires rank-2 `[row, expert]` logits with
  the expert tag last (compile errors) and a no-grad input
  (`error.UnsupportedGradient`).

```zig
test "argmax, topK, and routerTopK" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var logits = try fucina.Tensor(.{ .row, .expert }).fromSlice(&ctx, .{ 1, 4 }, &.{ 1, 3, 2, 0 });
    defer logits.deinit();

    var best = try logits.argmax(&ctx, .expert); // i64 indices, no grad
    defer best.deinit();
    try std.testing.expectEqualSlices(i64, &.{1}, try best.dataConst());

    var top = try logits.topK(&ctx, .expert, 2, .k);
    defer top.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 2 }, try top.values.dataConst());

    var selected: [2]usize = undefined;
    var weights: [2]f32 = undefined;
    try logits.routerTopK(&ctx, .expert, 2, .{}, &selected, &weights);
    try std.testing.expectEqual(@as(usize, 1), selected[0]);
    // normalize_selected renormalizes the top-k softmax mass to 1
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), weights[0], 1e-6);
}
```

### 4.17 Indexing, assembly, and functional updates (`src/ag/tensor.zig`)

These produce owned tensors, differentiable in every tensor operand unless
noted; gradients route exactly through the index maps (gather scatters-adds,
slices narrow, pads drop the border). All materialize copies except
`narrow`, `select`, `slice` on step-1 ranges, `unbindInto`'s entries, and
`repeatAxis` with `n == 1`, which alias the source storage (see their
bullets).

- `gather(ctx, tag, indices: []const usize, out_tag)` — select rows along
  `tag`; the axis is retagged `out_tag` (which may equal `tag`). This IS
  torch.index_select with host-side indices (indices live host-side by
  design — ARCHITECTURE.md); `indexSelect` below is the tensor-valued
  spelling and lowers to this. For per-ELEMENT index tensors use
  `takeAlongAxis` below.
- `indexSelect(ctx, tag, indices, out_tag)` — torch.index_select: `gather`
  with a rank-1 **i64** index tensor (the argmax/topK/sort index
  convention — their outputs feed it directly; any other dtype is a
  compile error), read host-side into the same `[]usize` path. Entries
  outside `[0, dim(tag))` error with `IndexOutOfBounds` (negatives are not
  wrapped); duplicate indices accumulate their gradients (the scatter-add
  adjoint); the index tensor is control data outside the graph and can be
  released after the call.
- `getRows(ctx, tag, indices, out_tag)` — **block-quantized tensors only**:
  fused gather + dequantize of rows from a rank-2 weight into an f32 tensor
  (the embedding-lookup path; §10). No-grad (quantized tensors are
  constants).
- `narrow(ctx, tag, start, length)` — contiguous sub-range as a zero-copy
  **view**: it retains the source buffer and aliases its memory, so a
  mutation of the source through `data()` is visible through the result.
- `select(ctx, tag, index: isize)` — torch.select / `x[i]`: one position
  of `tag` with the axis removed (composed narrow → squeeze — a zero-copy
  view; scope-required under gradients, §4.1). Negative `index` counts
  from the end; out of range errors with `IndexOutOfBounds`. The gradient
  is the exact scatter — unselected positions receive zero.
- `slice(ctx, spec)` — multi-axis basic slicing (torch/numpy
  `x[1:-1, ::2]`, positive steps): `spec` names the tags to slice, each
  field a `fucina.SliceRange`-shaped range — §3.7 has the full bounds
  contract (negatives from the end, `end = null` = dim, clamping, no
  negative steps — compose `flip`). Composed per-axis `narrow`/`sliceStep`
  in tag order; scope-required under gradients when more than one axis is
  sliced.
- `pad(ctx, tag, before, after, fill)` — constant padding on one axis; pad
  positions hold `fill` and drop their gradient.
- `concat(ctx, tag, others: []const *const Self)` — concatenation along an
  existing tag; one multi-parent node, differentiable in all inputs.
- `stack(ctx, new_tag, axis_index, others)` — stack along a **new** axis
  (composed insertAxis + concat; scope-required under gradients, §4.1).
- `unbindInto(ctx, tag, out: []Tensor(removeTag(tags, tag)))` — fill a
  caller-provided slice with the `dim(tag)` slices, each with `tag` removed
  (composed narrow + squeeze; scope-required under gradients). The caller
  owns and deinits every filled entry; on error, already-filled entries have
  been released.
- `repeatAxis(ctx, tag, n)` — tile the axis n times (`n == 1` is a
  zero-copy identity view; `n == 0` is `InvalidShape`); gradients from all
  copies accumulate.
- `flip(ctx, tag)` / `roll(ctx, tag, shift)` — reverse / rotate one axis
  (gather with a permutation; exact gradient).
- `setSlice(ctx, tag, start, update)` / `setRows(ctx, tag, indices, update)`
  — functional overwrite of a range / of specific rows; differentiable in
  both `self` (gradient zeroed where overwritten) and `update`.
- `zeroSlice(ctx, tag, start, length)` / `zeroRows(ctx, tag, indices)` —
  copy with a range/rows zeroed; the zeroed positions receive zero gradient.
- `maskedSelect(ctx, mask, out_tag)` — `torch.masked_select`: rank-1 tensor
  of the elements where `mask` is nonzero, row-major. The mask must be
  input-shaped, contiguous, and non-grad; `self` is differentiable (composed
  flatten + gather, scope-required under gradients). Selecting nothing is
  the dedicated `EmptySelection` (zero-size tensors are not representable),
  distinct from the shape errors so the no-match case is recoverable with a
  targeted `catch`; pre-counting with a mask sum avoids the error path
  (see the guard snippet in §3.7).
- `indexAdd(ctx, tag, indices: []const usize, update)` —
  torch.index_add: `setRows`'s accumulating sibling — duplicates allowed,
  each occurrence adds. Differentiable in both (`self` identity, `update`
  row-gather).
- `takeAlongAxis(ctx, tag, indices)` — torch.gather /
  np.take_along_axis: per-element row selection along `tag`. `indices` is
  a same-tagged i64 tensor (the argmax/topK/sort index convention —
  their outputs feed it directly; any other dtype is a compile error),
  matching `self` on every other axis; the result takes `indices`' shape. Parallel over outer
  slices (disjoint writes — bitwise identical for any thread count);
  differentiable in `self` (exact scatter-add adjoint, duplicate reads
  accumulate).
- `scatterAdd(ctx, tag, indices, src)` / `scatter(ctx, tag, indices,
  src)` — torch.scatter_add / torch.scatter: functional per-element
  accumulate/overwrite at `indices` along `tag` (`indices` shaped exactly
  like `src`). Duplicates accumulate in `scatterAdd`; in `scatter` they
  resolve deterministically to the LAST row-major write (torch leaves the
  order unspecified; this pins it). Parallel over outer slices —
  duplicates only collide within a slice, where serial row-major order is
  preserved, so accumulation order and last-write-wins stay bitwise
  identical for any thread count. Differentiable in both operands
  (overwrite zeroes `self`'s gradient at every written slot; on
  duplicates every writer receives the winning slot's gradient — the
  torch formula).
- `nonzero(allocator)` — host-side `[]usize` of row-major nonzero flat
  indices (§3.7); pairs with `gather`/`indexAdd`/`oneHot`.

```zig
test "takeAlongAxis pairs with argsort indices" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = fucina.Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 30, 10, 20, 5, 15, 0 });
    defer x.deinit();
    // argsort emits i64 indices — takeAlongAxis consumes them directly,
    // reordering each row (here: torch.gather(x, 1, x.argsort(1))).
    var order = try x.argsort(&ctx, .col, false);
    defer order.deinit();
    var sorted = try x.takeAlongAxis(&ctx, .col, &order);
    defer sorted.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30, 0, 5, 15 }, try sorted.dataConst());
}
```

```zig
test "gather, flip, and narrow copy with exact gradients" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer x.deinit();
    var picked = try x.gather(&ctx, .d, &.{ 2, 0 }, .g); // tag .d becomes .g
    defer picked.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 30, 10 }, try picked.dataConst());

    var reversed = try x.flip(&ctx, .d);
    defer reversed.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 30, 20, 10 }, try reversed.dataConst());

    var mid = try x.narrow(&ctx, .d, 1, 2);
    defer mid.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 20, 30 }, try mid.dataConst());
}
```

```zig
test "select, multi-axis slice, and tensor-valued indexSelect" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = fucina.Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 3, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    var last_row = try x.select(&ctx, .row, -1); // Tensor(.{ .col }), torch x[-1]
    defer last_row.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 9, 10, 11, 12 }, try last_row.dataConst());

    // torch x[1:, 1:-1] — omitted bounds default, negatives count from the end.
    var inner = try x.slice(&ctx, .{ .row = .{ .start = 1 }, .col = .{ .start = 1, .end = -1 } });
    defer inner.deinit();
    var inner_mat = try inner.materialize(&ctx);
    defer inner_mat.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 7, 10, 11 }, try inner_mat.dataConst());

    // index_select with an i64 tensor — argmax/topK/sort outputs feed it directly.
    const I = fucina.Tensor(.{ .dtype = .i64, .tags = .{.pick} });
    var idx = try I.fromSlice(&ctx, .{2}, &.{ 2, 0 });
    defer idx.deinit();
    var rows = try x.indexSelect(&ctx, .row, &idx, .pick); // Tensor(.{ .pick, .col })
    defer rows.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 9, 10, 11, 12, 1, 2, 3, 4 }, try rows.dataConst());
}
```

### 4.18 MoE facade entries (`src/exec/moe.zig`, `src/exec.zig`)

Routed expert FFNs run below the tag facade, directly on `ExecContext`
(inference-only; §10 covers the quantized layouts, §13 the LLM integration):

```zig
pub const MoeRhs = union(enum) { q4_k: ..., q5_k: ..., q6_k: ..., q8_0: ...,
    tq2_0: ..., q2_k: ..., iq2_xxs: ..., iq3_xxs: ..., streamed: ... };  // fucina.MoeRhs
pub fn moeExpertFfn(self: *ExecContext, x: *const Tensor,
    gate: *const MoeRhs, up: *const MoeRhs, down: *const MoeRhs,
    selected: []const usize, weights: []const f32,
    out_pe: usize, act: GatedOp, io: ?std.Io, profile: ?*MoeBatchProfile) !Tensor
pub fn moeExpertFfnBatch(..., top_k: usize, ...) !Tensor
```

`fucina.MoeRhs` stacks all experts of one layer's gate/up/down projection
into a single compact-block RHS (experts are row-contiguous zero-copy
sub-views; the resident arms cover q4_k/q5_k/q6_k/q8_0/tq2_0/q2_k/
iq2_xxs/iq3_xxs, plus a `streamed` arm whose expert blocks resolve
through the disk-backed expert store (`src/exec/expert_store.zig`)
instead of one resident buffer).
`moeExpertFfn` computes the route-weighted sum over the selected experts of
`down(act(gate(x), up(x)))` for a single token; `moeExpertFfnBatch` is the
batched-prefill variant taking the per-token `selected`/`weights` produced
by `routerTopK` (§4.16). `fucina.MoeBatchProfile` is an optional wall-clock
breakdown the caller can pass to profile a run.

`ExecContext.moe_chain` re-exports the shared batched-MoE scheduling
scaffolding of `src/exec/moe_chain.zig` — the expert-grouped route plan,
the phase-chain machinery, chunk helpers, and the profile timer pair — so
in-tree LLM-band MoE engines (§13) reach the exact same types through the
`fucina` root.

`fucina.expert_store` (`src/exec/expert_store.zig`) is the out-of-core
expert tier behind the `streamed` arm: `fucina.ExpertStore`
(`create`/`destroy`, `addLayer`/`finalize`) opens the GGUF part files and
resolves expert blocks through a pinned → LRU → `pread` hierarchy inside
a caller-driven `acquire`/`release` window, so MoE models larger than RAM
decode against the same kernels as the resident arms. `streamedRhs` hands
out the per-projection `StreamedMoeRhs` handle the `streamed` arm wraps;
`pilotHint` enqueues router-lookahead readahead on a dedicated I/O thread;
`saveUsage` persists the expert-usage histogram to a `<gguf>.experts`
sidecar that auto-pin turns into a pinned hot tier at the next startup,
and `repinPass` adapts that tier live at generation boundaries.

### 4.19 Math on non-f32 tensors (`src/ag/tensor.zig`)

`Tensor(.{ .dtype = dt, .tags = ... })` instantiates typed constant tensors
(§3, §8). Their math surface is forward-only, always no-grad:

- **f16 / bf16 / f64** (`supportsForwardFloatMath`) — native typed kernels:
  `to` (cast), `add`, `sub`, `mul`, `div` (same dtype both sides — cast
  explicitly), `sum`, `mean`, `sumAll`, `dot`, `scale`, `divScalar`, plus
  the full structural set (`split`, `merge`, `flatten`, `reshape`,
  `sliceStep`, `flip`, `roll`, `stack`, `repeatAxis`, and the indexing
  subset `gather`, `narrow`, `concat`, `setSlice`, `setRows`,
  `broadcastTo`, views). Pointwise and `dot` keep the input dtype;
  reductions widen f16/bf16 results to f32 (§4.7, §8).
- **f16 / bf16 only — the widened forward set.** Ops with no native typed
  kernel lower to widen → f32 kernel → narrow-once (f32 arithmetic and
  accumulation with a single final round, the §8.3 policy; on f64 they are
  a compile error — f64 math must not round through f32). Shape-preserving
  ops keep the input dtype: the unary family (`unary(op)` and every named
  alias listed in §3.10), `leakyRelu`, `clamp`, `addScalar`, `subScalar`,
  `powScalar`, `maximum`, `minimum`, `gated`/`glu`/`swiglu`/`geglu`,
  `softmax` and `layerNorm` (plain `.{}` options only — cast to f32 for
  the ext/affine paths), `logSoftmax`, `rmsNorm`, `rmsNormMul` (same-dtype
  weight), `cumsum`, `cumprod`, `where`/`maskedFill` (`.bool` or
  same-dtype masks), `compare` (`.bool` result), `pad`, and `einsum`
  (same-dtype operands, f32 GEMM
  lowering — the typed `dot` contract). The widened reductions `max`,
  `min`, `prod`, `variance`, `logsumexp` return **f32** like the native
  typed `sum`/`mean` (§8.3); `argmax` returns i64 (§4.16).
- **Block-quantized** (q8_0, q4_k, ...): no arithmetic — `to(.f32)`
  (dequantize), `getRows` (§4.17), row-axis `concat`, `packRhs` /
  `packRhsLayout` (§4.9), and constructors/views (§3, §10). Their main math
  role is as the constant RHS of `dot` (§4.8) and `dotPacked` (§4.9).
- **Integer dtypes** (e.g. token-id tensors) — ordinary integer forward
  math, plain exec loops (integers are never the hot path): wrapping
  two's-complement `add`/`sub`/`mul` (torch's narrowing behavior),
  `maximum`/`minimum`, and EXPLICIT division — `divTrunc` (toward zero)
  and `divFloor` (toward −inf), `error.DivisionByZero` on a zero divisor,
  minInt/−1 wrapping to minInt. There is deliberately no integer `div`:
  torch's `/` silently promotes integers to float, and Fucina keeps
  promotion explicit (documented divergence). `sum`/`sumAll` accumulate
  in i64 and RETURN `.i64` (torch's integer-sum dtype). `to` casts to any
  scalar dtype (§3.8).
- **`.bool`**: no pointwise arithmetic (compile error — cast first);
  `to` and the counting `sum`/`sumAll` (i64) apply, plus the structural
  subset.

The typed forward ops are no-grad by design: an operand that requires
gradients is REJECTED with `error.UnsupportedGradient` instead of silently
dropping its graph. The differentiable ways into and out of the 16-bit
world are `to` (§3.8) and the mixed-RHS `dot`/`einsum` (§4.8); widen with
`to(.f32)` for everything else in a trained path.

Because the widened ops run the identical f32 kernels and round once on
store, their results are bit-identical to "cast up, run the f32 op, cast
down" — pinned by parity tests in `src/ag/tensor_tests.zig`:

```zig
test "bf16 forward ops compute through f32 and narrow once" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .seq, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -1.25, 2.0, 3.5 });
    defer x.deinit();
    var half = try x.to(&ctx, .bf16);
    defer half.deinit();

    var activated = try half.gelu(&ctx); // widen -> f32 gelu -> narrow
    defer activated.deinit();
    comptime std.debug.assert(@TypeOf(activated).dtype == .bf16);

    var reference_f32 = try x.gelu(&ctx);
    defer reference_f32.deinit();
    var reference = try reference_f32.to(&ctx, .bf16);
    defer reference.deinit();
    try std.testing.expectEqualSlices(u16, try reference.dataConst(), try activated.dataConst());

    // Widened reductions keep the f32 accumulator dtype, like sum/mean.
    var spread = try half.variance(&ctx, .d, 0);
    defer spread.deinit();
    comptime std.debug.assert(@TypeOf(spread).dtype == .f32);
}
```

```zig
test "integer math wraps, divides explicitly, and reduces to i64" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const Ids = fucina.Tensor(.{ .dtype = .i8, .tags = .{.d} });
    var a = try Ids.fromSlice(&ctx, .{2}, &.{ 127, -7 });
    defer a.deinit();
    var b = try Ids.fromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer b.deinit();

    var wrapped = try a.add(&ctx, &b); // two's-complement wrap
    defer wrapped.deinit();
    try std.testing.expectEqualSlices(i8, &.{ -128, -5 }, try wrapped.dataConst());

    var quotient = try a.divFloor(&ctx, &b); // explicit: no integer `div`
    defer quotient.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 127, -4 }, try quotient.dataConst());

    var total = try a.sumAll(&ctx); // integer reductions return i64
    defer total.deinit();
    comptime std.debug.assert(@TypeOf(total).dtype == .i64);
    try std.testing.expectEqual(@as(i64, 120), try total.item());
}
```

The f32 `to(ctx, target_dtype)` cast is differentiable for `.f32 → .f32`
and for the mixed-precision narrows `.f32 → .f16`/`.bf16` (§3.8, §5.1 —
the backward is the identity on the f32 upstream gradient); casting a
grad-requiring tensor to any other dtype fails with
`error.GradientCastUnsupported`.

## 5. Automatic differentiation

Fucina's autograd is eager and backward-only: every op computes its value
immediately through the same public code path, and — when an operand requires
gradients — attaches a backward record for the reverse pass. There is no
graph compiler, no tape replay, and no separate "raw" autograd surface (a
guard test in `src/ag_tests.zig` asserts the legacy `Function`/`Node`/`Engine`
declarations stay removed from the core). The module root is `src/ag.zig`;
the user-facing surface in this section (`Tensor`, `checkpoint`,
`checkpointWithContext`, `noGrad`, `isGradEnabled`, `NoGradScope`,
`customVjp`, `gradcheck` and its option/result types) is re-exported at the
`fucina` root (§1). The engine internals documented below — `GradState`,
`BackwardFunction`, `AgError`, the `backwardGrad*` entry points,
`BlockOutput`/`BlockOutputWithContext` — deliberately are not.

### 5.1 The gradient model (`src/ag/tensor.zig`, `src/ag/core.zig`)

The f32 public tensor (§3) owns exactly one raw value and at most one
gradient state:

```zig
value: RawTensor,
grad_state: ?*GradState = null,
scope_owned: bool = false, // exec-scope borrow flag, see §6
```

- **Constants** (`constant`, `fromSlice`, `fromTensor`, `zeros`, `ones`,
  `full`, `scalar`, `empty`, the borrowed-slice constructors) have
  `grad_state == null`. They participate in any op but never accumulate
  gradients.
- **Variables** (`variable`, `variableFromSlice`) attach a leaf `GradState`
  allocated from `ctx.allocator`. `deinit` on the tensor releases both the
  value and the state (unless `scope_owned`, in which case `deinit` is a
  no-op and the exec scope releases everything at `closeExecScope`, §6).
- The f32 branch carries the full graph machinery. The f16/bf16 branch is
  a LEAF-capable participant: `variable`/`variableFromSlice` create
  trainable 16-bit leaves whose accumulated gradient is ALWAYS an f32 raw
  tensor (there is no 16-bit gradient anywhere in the engine), and the
  differentiable entries into/out of the 16-bit world are `to` (both cast
  directions, §3.8) and the mixed-RHS `dot`/`einsum` (§4.8). Every OTHER
  typed forward op is no-grad by design and rejects a grad-requiring
  operand with `error.UnsupportedGradient` (§4.19). f64, integer, and
  block-quantized constant branches have no gradient support; their
  `requiresGrad()` never returns `true` (integer and block-quantized
  branches hard-wire `false`; on f64 `variable`/`variableFromSlice`/
  `grad`/`gradView` exist but are compile errors, while `zeroGrad` no-ops
  and `detach` just returns a no-grad view).

`requiresGrad` is simply:

```zig
pub fn requiresGrad(self: *const Self) bool // grad_state != null
```

Every differentiable op funnels through one private tail (`finishOp`): if no
operand requires gradients — or a `noGrad` scope is active (§5.4) — the
result is a plain no-grad tensor and no graph state is retained; otherwise
the eager value is wrapped together with a VJP record from
`src/ag/backward.zig` inside a fresh `GradState`. Because forward always
takes the identical kernel path, training and inference produce identical
values.

A `GradState` (`src/ag/core.zig`) holds the accumulated gradient, the
backward record, and the scheduling state:

```zig
pub const GradState = struct {
    allocator: Allocator,
    grad: ?Tensor = null,              // raw accumulated gradient
    grad_fn: ?BackwardFunction = null, // null for leaves
    state: std.atomic.Value(u8),       // idle | pending | ongoing
    pending_grads: std.atomic.Value(u32),
    grad_mutex: thread.Mutex,
    backward_done: bool,               // completed pass consumed this output (§5.2)
};
```

`BackwardFunction` is the type-erased VJP record interface:

```zig
pub const BackwardFunction = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        operands: *const fn (*const anyopaque) []const ?*GradState,
        backward: *const fn (*const anyopaque, *ExecContext, *const Tensor,
                             []const bool, []?Tensor) anyerror!void,
        deinit: *const fn (*anyopaque, Allocator) void,
        prefer_async_backward: bool = false,
        estimated_work: ?*const fn (*const anyopaque) usize = null,
    };
};
```

`operands()` returns one slot per forward operand (`null` for operands that
were constants); `backward(ctx, gy, needs_grad, out)` must write an owned raw
gradient into `out[i]` for every true `needs_grad[i]`. The engine consumes
and deinits those tensors. This interface is internal; user-defined
differentiable ops go through `customVjp` (§5.6), which implements it for
you.

```zig
test "backward and grad read" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var c = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer c.deinit(); // constant: no grad state

    var y = try x.mul(&ctx, &c);
    defer y.deinit();
    var loss = try y.sumAll(&ctx); // scalar output: implicit seed 1
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?; // deep copy; caller deinits
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30 }, try gx.dataConst());
    try std.testing.expect((try c.grad(&ctx)) == null); // constants never accumulate
}
```

### 5.2 Running backward (`src/ag/core.zig`)

The facade exposes a single-output entry point, in an implicitly-seeded and
an explicitly-seeded form:

```zig
pub fn backward(self: *const Self, ctx: *ExecContext) !void
pub fn backwardWithGrad(self: *const Self, ctx: *ExecContext, grad_output: *const Self) !void
```

Both error with `error.NoGradientGraph` when called on a tensor without a
`grad_state` and otherwise delegate to the engine's
`core.backwardGradOne(ctx, state, &self.value)`. `backwardWithGrad` first
installs `grad_output` as the output gradient: it is same-tagged (checked
at comptime) and must match `self`'s shape (`error.ShapeMismatch`); it is
read as a *value* — its own gradient state, if any, is ignored — and
replaces any gradient already held by `self`. The engine itself is
multi-output-capable but internal (not re-exported at the `fucina` root):

```zig
pub fn backwardGrad(ctx: *ExecContext, outputs: []const *GradState,
                    output_values: []const *const Tensor) !void
pub fn backwardGradSerial(ctx: *ExecContext, ...) !void // same, node-serial
pub fn backwardGradOne(ctx: *ExecContext, output: *GradState,
                       output_value: *const Tensor) !void
```

`pub const AgError = error{ MissingOutputGradient, MissingBackwardGradient, BackwardAlreadyRun };`

**Seeding rules.** Before any scheduling state exists, every output is
validated and its implicit seed pre-allocated:

- A **scalar** output (one element total — a `{1,1}` tensor counts) with no
  gradient present receives the implicit seed `1`.
- A **non-scalar** output with no gradient present fails with
  `error.MissingOutputGradient` — seed it with `backwardWithGrad` (or
  install a gradient through the low-level `setGrad`, §5.3).
- An output whose `GradState` **already holds a gradient** (installed by
  `backwardWithGrad`, by `setGrad` (§5.3), or by the checkpoint recompute)
  is respected as-is — the implicit `+1` is *not* added on top.
- In a multi-output pass, a scalar output whose gradient appears only
  **mid-pass** (an earlier output's backward already contributed to it)
  still accumulates its own seed on top of that contribution.

```zig
test "non-scalar output needs a seed" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer x.deinit();
    var y = try x.scale(&ctx, 2);
    defer y.deinit();

    // Unseeded non-scalar output: fails before any scheduling state exists,
    // so the SAME graph runs once a seed is supplied.
    try std.testing.expectError(error.MissingOutputGradient, y.backward(&ctx));

    var grad_output = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 1, 10 });
    defer grad_output.deinit();
    try y.backwardWithGrad(&ctx, &grad_output); // shape-checked; read as a value
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 20 }, try gx.dataConst());
}
```

**Pending-counter scheduling.** `prepareBackwardPass` walks the graph once,
incrementing each state's `pending_grads` by one per consumer edge and
flipping it `idle → pending` on first visit. A node's VJP executes only when
its counter drains to zero — i.e. after *all* downstream contributions have
arrived — so a value consumed by several branches (a shared activation, a
weight used twice) accumulates the complete upstream gradient before
propagating it exactly once.

```zig
test "shared branch accumulates" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 3, 5 });
    defer x.deinit();

    var sq = try x.mul(&ctx, &x); // x feeds the node twice
    defer sq.deinit();
    var lin = try x.scale(&ctx, 4); // second consumer of x
    defer lin.deinit();
    var both = try sq.add(&ctx, &lin);
    defer both.deinit();
    var loss = try both.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    // d/dx (x^2 + 4x) = 2x + 4: contributions from every branch summed.
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 14 }, try gx.dataConst());
}
```

**Accumulation mechanics.** Contributions add in place under the per-state
`grad_mutex`, so concurrent branches on pool threads are safe. Before the
engine mutates an accumulator (or installs a first contribution that will be
added to later), it checks the raw tensor's exclusive-ownership predicate and
materializes a private copy when the buffer is shared — a VJP may therefore
hand back cheap refcounted *views* of `gy` without risking cross-state
aliasing (`src/ag/core_tests.zig` pins this copy-on-write behavior).

**`needs_grad` pruning.** Each VJP receives `needs_grad: []const bool`
(true where the operand slot has a `GradState`), so gradients for constant
operands — frozen weights, masks, cached KV — are never computed.

**Parallel backward vs `backwardGradSerial`.** The engine grabs the
context's work pool (`ctx.tryWorkPool()`, §9). When several *independent*
states become ready at once, it may spawn all but one onto pool threads —
but only for records that opt in through the vtable: `prefer_async_backward
= true` (no in-tree record currently sets it) or an `estimated_work()` at or
above `parallel.backward_async_work_threshold` (`256 * 1024 * 1024` work
units; provided by the attention, causal-conv1d-family, gather,
linear-cross-entropy, `Conv1d`/`Conv2d`, `Dot`, and ternary-STE-dot
records). Node-level spawning is
additionally gated at comptime by `exec.parallel_dot_backward_branches`
(native backend with BLAS, §9) — on scalar or no-BLAS builds every node runs
inline on the calling thread. (`DotBackward` additionally parallelizes its
two operand branches internally, via the context's dot-backward worker.)
`backwardGradSerial` forces `pool = null` so the whole pass is node-serial
regardless; kernel-level `parallelChunks` parallelism *inside* a VJP is
unaffected. Serial mode is required whenever a threadlocal guard must
observe the entire pass on one thread — the checkpoint recompute (§5.5) is
the in-tree case.

**Error exits are re-runnable.** Seeds are validated and allocated *before*
any pending counter is installed — a stranded counter would make the next
backward over the same states stop at their `.pending` check and report
success with missing gradients — so a seeding failure (as in the snippet
above) leaves zero scheduling debris and the same graph runs correctly once
the seed is supplied (`src/ag/core_tests.zig`, "failed output seeding leaves
the graph re-runnable"). When a VJP fails mid-pass, the engine deinits any
gradients it produced, releases the pending counters of the failing node's
operands (returning them to `idle`), records the first error, drains all
in-flight tasks, and returns that error. Re-runnability restores
*scheduling* state, not values: gradient contributions delivered before the
failure remain accumulated — call `zeroGrad` on the leaves before retrying
if exact values matter.

**One backward per graph.** Gradients accumulate in every `GradState` they
touch, including interior op results, and a completed pass leaves them
there. Re-running over the *same* retained graph would therefore compound:
interior states would receive new contributions on top of their previous
gradients, which then flow downstream multiplied (two passes over
`loss = sum(x·x)` would yield `3·(2x)`, not `2·(2x)`). A completed pass
therefore marks its outputs consumed, and a repeated
`backward`/`backwardWithGrad` over them fails with
`error.BackwardAlreadyRun` before any scheduling state is installed —
`zeroGrad` resets gradients, not the consumed graph. Only a *completed*
pass consumes: the failed-seeding retry above stays re-runnable, and a leaf
output (a bare variable) has no graph to consume and is never marked. The
supported accumulation idiom is one backward per freshly built forward
graph over shared leaves (§5.3).

### 5.3 Reading, seeding, and resetting gradients (`src/ag/tensor.zig`, `src/ag/core.zig`)

```zig
pub fn grad(self: *const Self, ctx: *ExecContext) !?Self     // deep copy
pub fn gradView(self: *const Self, ctx: *ExecContext) !?Self // refcounted view
pub fn zeroGrad(self: *const Self) void                      // drop accumulated grad
pub fn detach(self: *const Self, ctx: *ExecContext) !Self    // no-grad view of value
```

- `grad` returns `null` for constants and for variables with no accumulated
  gradient; otherwise a caller-owned no-grad tensor holding a deep copy
  (allocated from `ctx.allocator`). `gradView` is the zero-copy variant: the
  result aliases the accumulator *as of that moment*. A later backward pass
  does **not** mutate it — the held reference defeats the engine's
  copy-on-write check (`canTakeInPlace` needs a unique buffer), so the
  engine accumulates into a fresh private buffer and the view silently keeps
  the stale pre-pass value. Use `gradView` for immediate reads, `grad` for
  anything that must observe later passes. Both are taken under the state's
  mutex.
- `zeroGrad` frees the accumulated gradient (no-op on constants). Training
  loops call it between optimizer steps; `optim.OptimizerSet.zeroGrad` (§11)
  fans it out over registered parameters.
- `detach` returns a no-grad constant sharing the same storage
  (refcounted view): the value flows on, the graph is cut.
- `data()` refuses mutable access on a grad-carrying tensor with
  `error.MutableDataRequiresNoGrad` (mutating a recorded value would
  invalidate the graph); `dataConst()`/`item()`/`copyTo()` are always
  allowed.

Direct gradient state access goes through the public `grad_state` field
(`?*GradState`); its methods are thread-safe under the per-state mutex:

```zig
pub fn setGrad(self: *GradState, grad: Tensor) void        // takes ownership, replaces
pub fn zeroGrad(self: *GradState) void
pub fn gradClone(self: *GradState, allocator: Allocator) !?Tensor
pub fn gradView(self: *GradState) !?Tensor
```

`setGrad` consumes a *raw* tensor (e.g. produced by `ctx.fromSlice`, §6) and
replaces any existing gradient. For seeding an output before `backward`,
prefer `backwardWithGrad` (§5.2) — it stays in facade tensors and
shape-checks the output gradient; `setGrad` is the unchecked low-level hook
underneath it (the checkpoint recompute seeds through it too, §5.5, and gradient
clipping rewrites accumulated gradients with it, §11.4).
`GradState.leaf`/`deinit` and the `createNode`/`BackwardNode` record
co-allocation exist for internal wiring and are managed by the facade.

**Accumulation across backward calls** — the micro-batch idiom: build a
fresh forward graph per micro-batch over the same leaf variables and call
`backward` once per graph; leaf gradients sum. (A repeat over the *same*
graph fails with `error.BackwardAlreadyRun`, §5.2.)

```zig
test "micro-batch accumulation and zeroGrad" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var w = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer w.deinit();

    for (0..2) |_| { // one fresh graph per micro-batch
        var y = try w.scale(&ctx, 3);
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gw = (try w.grad(&ctx)).?; // 3 + 3: sums across the two passes
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 6 }, try gw.dataConst());

    w.zeroGrad(); // training loops reset between optimizer steps
    try std.testing.expect((try w.grad(&ctx)) == null);
}
```

When gradients are tracked, interior op results carry live `GradState`s that
downstream records point at; keep them alive until after `backward` (the
defer-deinit idiom above) or run the forward under an exec scope, which owns
them for you (§6, [TRAINING.md](TRAINING.md)). The *composed* facade ops
(`nllLoss`, `l2Normalize`, `cosineSimilarity`, `norm`, `normAll`,
`maskedSelect`, `maskedScatter`, `select`, `slice` (more than one sliced
axis), `reshape` (multi-tag targets), `rollBy`, `shiftBy`, `trace`,
`diag`, `constantPad2d`/`zeroPad2d`, `stack`, `unbindInto`, `einsumMany`)
create function-local graph nodes and therefore require an active exec
scope when any operand requires gradients — they fail loudly with
`error.ActiveExecScopeRequired` instead of dangling operand pointers; the
no-grad composition works unscoped.

### 5.4 noGrad scopes (`src/ag/control.zig`)

```zig
pub fn noGrad() NoGradScope
pub fn isGradEnabled() bool
pub const NoGradScope = struct {
    pub fn close(self: *NoGradScope) void
};
```

`noGrad()` increments a **threadlocal** depth counter and returns a scope
handle; `close()` decrements it (asserting the depth is nonzero) and is
idempotent — a handle closed early makes the deferred `close()` a no-op, as
in `src/ag/control.zig`'s own test. Scopes nest arbitrarily;
`isGradEnabled()` is true only at depth zero, per thread.

While disabled, every op takes the identical forward path but skips backward
record creation even when operands are variables — the standard evaluation
mode wrapper. `customVjp` honors it too. `checkpoint` does **not**: it keys
on its inputs' `requiresGrad` alone, so a checkpoint call inside a `noGrad`
scope still records its backward node (the block body itself always runs
grad-free; only the outer node is affected).

```zig
test "noGrad suppresses recording" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer x.deinit();

    var scope = fucina.noGrad();
    defer scope.close();
    try std.testing.expect(!fucina.isGradEnabled());

    var y = try x.scale(&ctx, 2); // same op path, but no backward node
    defer y.deinit();
    try std.testing.expect(!y.requiresGrad());
}
```

### 5.5 Activation checkpointing (`src/ag/checkpoint.zig`)

```zig
pub fn checkpoint(ctx: *ExecContext, comptime block: anytype, inputs: anytype)
    !BlockOutput(block, @TypeOf(inputs))
pub fn checkpointWithContext(ctx: *ExecContext, comptime block: anytype,
    extra: anytype, inputs: anytype)
    !BlockOutputWithContext(block, @TypeOf(extra), @TypeOf(inputs))
```

Recompute-in-backward: the forward run executes `block` on grad-free
constants inside an inner exec scope and retains only refcounted views of
the block **inputs** plus one deep copy of the block **output** — every
intermediate is freed the moment the scope closes. When gradients reach the
checkpoint node during backward, the block is re-run on the stored input
views to rebuild the subgraph, the incoming gradient is installed on the
recomputed output with `setGrad` (which is why pre-seeded outputs are never
topped up with `+1`, §5.2), a full inner backward runs, and the resulting
input gradients are handed to the outer engine. Memory per checkpoint is
O(inputs + output) instead of O(intermediates).

`BlockOutput`/`BlockOutputWithContext` (pub in `src/ag/checkpoint.zig`, not
re-exported at the root) compute the result type: the block's return type
with the error union stripped.

Contract for `block` (violations are compile errors where detectable):

- a comptime function `fn (*ExecContext, ...inputs) !Tensor(..)` — for
  `checkpointWithContext`, `fn (*ExecContext, extra, ...inputs) !Tensor(..)`
  — whose parameters after the lead are single-item pointers to **f32**
  facade tensors matching the `inputs` tuple, and whose result is produced
  by facade ops on those inputs. The block always runs under an exec scope,
  so the defer-deinit forward idiom works unchanged inside it (deinits of
  scope-owned results are no-ops).
- **deterministic and pure in its inputs**: the recompute must rebuild the
  exact forward values. RNG-using ops must derive their stream from explicit
  stored seeds — `dropout(p, seed)` qualifies by construction (its mask is a
  pure function of `(seed, element index)` and is never stored); ambient RNG
  state does not.
- **no nested `checkpoint` inside a block**: the recompute lock is not
  reentrant; a nested recompute is rejected at backward time with
  `error.NestedCheckpointRecompute` instead of deadlocking.

Contract for `extra` (`checkpointWithContext` only): it sits between
`*ExecContext` and the inputs in the block signature, is stored **by value**
in the backward node, and is passed verbatim to both the forward run and the
recompute. It is the channel for everything non-differentiable — frozen
quantized/f16/bf16 constant tensors, RoPE tables, config values, layer
struct pointers. Anything reachable through it must remain valid until
backward completes (the node keeps only the value bits, no deep copy or
refcount) and is treated as constant: tensors reachable through `extra`
never receive gradients — trainable tensors must travel through `inputs`.
A `{}` (void) `extra` degenerates to plain `checkpoint`.

Runtime behavior and constraints:

- With no grad-requiring input, checkpoint degenerates to a no-grad forward
  (same adoption tail as any facade op). The result follows the standard
  ownership contract: caller-owned with no scope open, a `scope_owned`
  borrow otherwise.
- Recomputes are serialized by one process-wide mutex (re-run facade ops
  adopt into `ctx` scope entries, which is not thread-safe), and the inner
  backward runs via `backwardGradSerial` so the threadlocal nested-recompute
  guard can see any nested node. Checkpoint nodes themselves always execute
  synchronously on the scheduling thread.
- Block runs (forward and recompute) disable GPU quant-dot offload for their
  duration (an internal `disableQuantDotGpu` scope) so both runs take the
  same kernels.
- The recompute errors with `error.CheckpointOutputNotDifferentiable` if the
  re-run block's output carries no graph, and
  `error.CheckpointMissingInputGradient` if a grad-requiring input received
  none.

```zig
fn ckptLayer(
    ctx: *fucina.ExecContext,
    x: *const fucina.Tensor(.{ .batch, .in }),
    w: *const fucina.Tensor(.{ .out, .in }),
) !fucina.Tensor(.{ .batch, .out }) {
    var z = try x.dot(ctx, w, .in); // intermediates are scope-owned:
    defer z.deinit(); //             deinit is a safe no-op inside the block
    return z.tanh(ctx);
}

test "checkpointed layer backward" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 1, 2 }, &.{ 1, 2 });
    defer x.deinit();
    var w = try fucina.Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 1, 2 }, &.{ 0.5, -0.25 });
    defer w.deinit();

    var y = try fucina.checkpoint(&ctx, ckptLayer, .{ &x, &w });
    defer y.deinit(); // only inputs + this output are retained
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx); // block re-runs here to rebuild the subgraph

    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    // z = 0, tanh'(0) = 1 -> dL/dw = x
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, try gw.dataConst());
}
```

Frozen state through `extra`:

```zig
const Frozen = struct {
    w: *const fucina.Tensor(.{ .out, .in }), // constant: never receives grads
    alpha: f32,
};

fn frozenLayer(
    ctx: *fucina.ExecContext,
    extra: Frozen,
    x: *const fucina.Tensor(.{ .batch, .in }),
) !fucina.Tensor(.{ .batch, .out }) {
    var z = try x.dot(ctx, extra.w, .in);
    defer z.deinit();
    return z.scale(ctx, extra.alpha);
}

test "checkpointWithContext carries frozen state" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 1, 2 }, &.{ 1, 2 });
    defer x.deinit();
    var w = try fucina.Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 1, 2 }, &.{ 3, 4 });
    defer w.deinit(); // frozen weight rides in `extra`, not `inputs`

    var y = try fucina.checkpointWithContext(&ctx, frozenLayer, Frozen{ .w = &w, .alpha = 0.5 }, .{&x});
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1.5, 2 }, try gx.dataConst()); // alpha * w
}
```

The gradients are **bitwise identical** to the non-checkpointed forward: the
recompute runs the identical ops on the identical input views
(`src/ag/checkpoint_tests.zig` asserts parity to the bit).

### 5.6 Custom VJPs (`src/ag/custom.zig`)

```zig
pub fn customVjp(ctx: *ExecContext, comptime Spec: type, extra: anytype,
                 inputs: anytype) !Spec.Output
```

`customVjp` admits a user-defined differentiable op. (For plain elementwise
scalar functions, prefer the `elementalUnary`/`elementalBinary` convenience
tier built on this adapter — §4.4 — which needs no raw-tensor code.) The
public contract stays in f32 facade tensors — `inputs` is a tuple of
pointers to f32 facade tensors, `Spec.Output` an f32 facade tensor type —
while the `Spec` computes on raw tensors (`fucina.internal.RawTensor`, §8)
inside the adapter:

```zig
const Spec = struct {
    pub const Output = fucina.Tensor(.{ ... });
    pub fn forward(ctx: *ExecContext, extra: E,
                   inputs: []const *const RawTensor) !RawTensor { ... }
    pub fn backward(ctx: *ExecContext, extra: E,
                    inputs: []const *const RawTensor,
                    output: *const RawTensor, gy: *const RawTensor,
                    needs_grad: []const bool, out: []?RawTensor) !void { ... }
};
```

Missing `Output`/`forward`/`backward` declarations, non-f32 facade types, or
a non-tuple `inputs` are compile errors. Semantics:

- `forward` returns an owned raw tensor; `customVjp` validates it against
  `Output`'s tag rank and wraps it.
- If no input requires gradients, or a `noGrad` scope is active, the result
  is a plain no-grad tensor and `backward` is never referenced at runtime.
- Otherwise the node captures refcounted **views** of every input value and
  of the output (cheap, no copies), the input `GradState` pointers as
  operands, and `extra` **by value** (same lifetime contract as checkpoint's
  `extra`: pointees must outlive backward).
- At backward time the adapter passes the saved views, the saved output, and
  the upstream `gy`; `backward` must write an *owned* raw tensor into
  `out[i]` for every true `needs_grad[i]` (the engine consumes and deinits
  them; leaving a required slot `null` surfaces as
  `error.MissingBackwardGradient`, §5.2). Each produced gradient is
  shape-checked against its input; a mismatch fails the pass with
  `TensorError.ShapeMismatch`.

<!-- snippet: helper -->
```zig
const RawTensor = fucina.internal.RawTensor;

const ScaledSquare = struct {
    pub const Output = fucina.Tensor(.{.d});

    pub fn forward(ctx: *fucina.ExecContext, extra: f32, inputs: []const *const RawTensor) !RawTensor {
        var sq = try ctx.mulRank(1, inputs[0], inputs[0]);
        defer sq.deinit();
        return ctx.scale(&sq, extra); // y = extra * x^2
    }

    pub fn backward(
        ctx: *fucina.ExecContext,
        extra: f32,
        inputs: []const *const RawTensor,
        output: *const RawTensor,
        gy: *const RawTensor,
        needs_grad: []const bool,
        out: []?RawTensor,
    ) !void {
        _ = output;
        if (needs_grad[0]) {
            var slope = try ctx.scale(inputs[0], 2 * extra); // dy/dx = 2*extra*x
            defer slope.deinit();
            out[0] = try ctx.mulRank(1, gy, &slope); // engine consumes out[0]
        }
    }
};
```

### 5.7 Gradient checking (`src/ag/gradcheck.zig`)

```zig
pub fn gradcheck(ctx: *ExecContext, comptime loss_fn: anytype,
                 inputs: anytype, options: Options) !Result

// root re-exports (src/ag.zig / src/fucina.zig):
pub const GradcheckOptions = gradcheck_mod.Options;
pub const GradcheckResult = gradcheck_mod.Result;
```

Central finite-difference validation of a deterministic scalar loss.
`loss_fn` must be `fn (*ExecContext, ...input ptrs) !Tensor(.{})` (one
pointer parameter per tuple entry; a non-scalar return is a compile error).
`inputs` is a tuple of **mutable** pointers to f32 facade tensors: variable
inputs are checked — the harness perturbs their owned storage element by
element, so they must be contiguous — while constants may appear and are
ignored. All inputs' gradients are zeroed before the check and again on exit
(the accumulated analytical gradients do not leak into the caller's
training state). The analytical backward and every loss evaluation run under
their own exec scope, so composed ops inside the loss are fine.

| `Options` field   | default | meaning                                        |
|-------------------|---------|------------------------------------------------|
| `eps`             | `1e-3`  | central-difference step (`f64`)                 |
| `abs_tol`         | `1e-3`  | absolute tolerance floor                        |
| `rel_tol`         | `1e-2`  | relative tolerance factor                       |
| `print_mismatch`  | `true`  | `std.debug.print` the first failing element     |

Per element the check is `|g_num − g_ana| ≤ abs_tol + rel_tol·|g_ana|`.
`Result` reports `checked` (element count), `max_abs_error`, and
`max_rel_error`. Errors: `error.InvalidGradcheckOptions` (non-finite or
non-positive `eps`; negative or non-finite tolerances), `error.NoVariableInputs`,
`error.MissingAnalyticalGradient` (backward produced nothing for a
variable), `error.GradientShapeMismatch`, and `error.GradientMismatch` (the
tolerance failure). Any error from the loss itself propagates.

```zig
fn squareLoss(ctx: *fucina.ExecContext, x: *const fucina.Tensor(.{.d})) !fucina.Tensor(.{}) {
    var y = try fucina.customVjp(ctx, ScaledSquare, @as(f32, 0.5), .{x});
    defer y.deinit();
    return y.sumAll(ctx);
}

test "customVjp validated by gradcheck" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, -2, 3 });
    defer x.deinit();

    const result = try fucina.gradcheck(&ctx, squareLoss, .{&x}, .{});
    try std.testing.expectEqual(@as(usize, 3), result.checked);
    try std.testing.expect(result.max_abs_error < 1e-2);
}
```

`gradcheck` is the oracle used throughout `src/ag/gradcheck_tests.zig` to
validate both built-in VJPs (conv2d, losses, norms) and custom ops; use it
for every new `customVjp` spec.

### 5.8 VJP coverage inventory (`src/ag/backward.zig`)

Every differentiable facade op attaches a concrete VJP record from
`src/ag/backward.zig`. Coverage by family (op names as on the facade, §4):

| Family | Differentiable ops | Notes |
|---|---|---|
| Pointwise arithmetic | `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `pow`, `scale`, `addScalar`, `subScalar`, `divScalar`, `powScalar`, `biasAdd` | broadcast operands reduce gradients back to source tags; `maximum`/`minimum` split the gradient evenly on exact ties; `biasAdd`'s slice bias is constant |
| Selection / masking | `where` (grads to both value operands), `maskedFill` (grad zeroed where filled), `clamp`, `dropout` | `cond`/`mask` are non-grad; dropout regenerates its mask from `(seed, index)` in forward, backward, and recompute |
| Unary activations | `relu`, `leakyRelu`, `exp`, `sqrt`, `rsqrt`, `sigmoid`, `silu`, `log`, `log1p`, `neg`, `abs`, `sin`, `cos`, `tanh`, `fastTanh`, `softcap30`, `softcap15`, `gelu`, `quickGelu`, `elu`, `geluErf`, `floor`, `ceil`, `round`, `sign`, `reciprocal`, `unary`, `snake`, `prelu` | `prelu`/`snake` also differentiate their parameters; `floor`/`ceil`/`round`/`sign` have zero gradient a.e. (torch convention) |
| Gated units | `gated`, `glu`, `swiglu`, `geglu`, `splitGated` | fused split+gate VJPs |
| Reductions / statistics | `sum`, `sumMany`, `sumAll`, `mean`, `variance`, `prod`, `cumsum`, `cumprod`, `logsumexp`, `standardizeAxis`, `norm`, `normAll`, `max`, `min`, `topK` (values arm), `sort` (values arm) | `max`/`min` route gradient to the first extremum (strict tie-break); `topK`/`sort` values scatter back through the saved indices |
| Structure / views | `withTags`, `permuteTo`, `transpose`, `alignTo`, `insertAxis`, `squeeze`, `split`, `merge`, `reshape`, `viewWithStrides`, `materialize`, `contiguous`, `broadcastTo`, `flatten`, `narrow`, `select`, `slice`, `sliceStep`, `pad`, `zeroPad2d`, `constantPad2d`, `concat`, `stack`, `unbindInto`, `repeatAxis`, `flip`, `roll`, `rollBy`, `shiftBy`, `diagonal`, `trace`, `diag`, `gather`, `indexSelect`, `takeAlongAxis`, `indexAdd`, `scatterAdd`, `scatter`, `maskedSelect`, `maskedScatter`, `setSlice`, `setRows`, `zeroSlice`, `zeroRows`, `relposShift`, `to` (f32/f16/bf16 targets, §3.8) | view VJPs scatter through the saved layout; `detach` deliberately cuts the graph |
| Norms / softmax | `softmax` (all fused options; `.mask` must not require grad), `logSoftmax`, `rmsNorm`, `rmsNormMul`, `rmsNormMulAdd`, `rmsNormMulRopeHalfPrepared`, `layerNorm` (plain + affine), `groupNorm`, `l2Normalize`, `cosineSimilarity` | |
| Losses | `crossEntropy`, `crossEntropyExt`, `linearCrossEntropyExt`, `mseLoss`, `huberLoss`, `bceLoss`, `klDivLoss`, `nllLoss` | `linearCrossEntropyExt` differentiates both the input and the classifier weight without materializing the logit gradient (§4.15) |
| Contractions | `dot` (f32×f32: both operands; quantized RHS: lhs-only, the RHS is a frozen constant; f16/bf16 RHS: lhs always, plus an f32 dW when the RHS is a grad-requiring 16-bit variable), `einsum` (f32×f32: both operands; f16/bf16 RHS: same variable-RHS contract as dot; each gradient is itself an einsum — GEMM-lowered for every tag structure, broadcast over forward-summed axes; `DotBackward`/`ConstRhsDotBackward` delegate to the einsum records), `einsumMany` (composes binary einsum records), `matmul` (2-D GEMM `.plain`/`.trans_b`, batched bmm all kinds; rank-2 `.trans_a` is a compile error directing to `dot`), `dotTernarySte` (straight-through estimator: dx through the quantized weight, dW as-if-unquantized) | |
| Convolutions / pooling | `conv1d`, `convTranspose1d`, `causalConv1d`, `groupedCausalConv1d`, `causalDepthwiseConv1d`, `conv2d`, `conv2dRelu`, `maxPool2d`, `avgPool2d`, `upsample2xNearest`, `channelAffine` | `conv2d` differentiates input, weight, and bias; `conv2dRelu` falls back to the composed differentiable path when any operand requires grad |
| Position / attention | `rope` (table and on-the-fly sources, both modes), `groupedAttention` | attention grad matrix: f32 KV = full q/k/v; f16 or q8_0 KV = q-only (caches are constants); `.bias` or multi-stream KV = inference-only (`error.UnsupportedGradient`) |

Intentionally no-grad (result is a constant; grad-requiring operands are
either irrelevant or rejected):

- **Constant results by nature**: `argmax`, `argsort`, the `indices` arm of
  `topK` and `sort`, `compare`, `isnan`, `isinf`, `isfinite`, `any`, `all`,
  `anyAll`, `allAll`, `logicalAnd`, `logicalOr`, `logicalXor`,
  `logicalNot` — index/mask outputs; gradients are undefined.
- **Inference-only fused kernels** — fail with
  `error.GradientQuantizedMatmulUnsupported` when an operand requires grad:
  `dotPacked`, `rmsNormMulDotPacked`, `splitSwiGluDotPacked`,
  `gegluQuantDotPacked` (§10). For a *trainable* path over quantized
  weights use `dot` with a quantized RHS (lhs-grad) or `dotTernarySte`.
- **Prepared-conv entries** — fail with
  `error.GradientPreparedConv2dUnsupported` when an operand requires grad:
  `prepareConv2dWeights`, `conv2dPrepared`, `conv2dPreparedRelu` (§4.14 —
  the prepared Winograd planes live outside the graph; use `conv2d`/
  `conv2dRelu` for the trainable path).
- **In-place / storage-consuming helpers** — fail with
  `error.UnsupportedGradient`: `addAxisVectorInPlace`,
  `addAxisVectorUnaryInPlace`, `addScaledInPlace`, `takeAddNoGrad`,
  `takeScaleNoGrad`, `routerTopK`.
- **Casts off the float seam**: `to` with a target other than `.f32`,
  `.f16`, or `.bf16` rejects grad-carrying inputs with
  `error.GradientCastUnsupported` (`to(.f32)` and the f16/bf16 narrows are
  differentiable, §3.8).
- The typed and quantized constant tensor branches (§3, §10) never carry
  gradients at all.

## 6. The execution runtime: ExecContext and the memory model

`fucina.ExecContext` is the eager runtime boundary. Every op call, every
tensor allocation, and every gradient pass goes through one context: it owns
the allocator wrapper, the backend instance, the transient-buffer pool, the
lazily-created worker team, and the exec-scope stack. There is no graph
object and no deferred execution — an op call runs the kernel and returns an
owned result immediately. This section covers the context itself and the
memory model it implements; the op surface it exposes is catalogued in §4,
autograd semantics in §5, backend selection in §9.

### 6.1 ExecContext: role and lifecycle (`src/exec.zig`, `src/exec/runtime.zig`)

`ExecContext` is a thin forwarding facade over `Runtime`
(`src/exec/runtime.zig`), the generic substrate that owns
allocation/thread/scope machinery. Domain op implementations live in leaf
modules under `src/exec/` (`elementwise.zig`, `matmul.zig`, `conv.zig`,
`attention.zig`, `moe.zig`, …) and receive `*Runtime` explicitly; the facade
methods on `ExecContext` forward to them. The only domain state on the facade
itself is MoE-decode scratch.

```zig
pub const ExecContext = struct {
    rt: Runtime,                     // substrate: allocator, backend, pool, team, scopes
    moe_scratch: MoeDecodeScratch,   // domain state, facade-owned
    allocator: Allocator,            // cached copy of rt.allocator (thread-safe wrapper)

    pub fn init(self: *ExecContext, allocator: Allocator) void
    pub fn deinit(self: *ExecContext) void
};
```

`Runtime` fields (all reached through `ctx.rt`, internal but observable):

| Field | Type | Created |
|---|---|---|
| `thread_safe_allocator` | `thread.ThreadSafeAllocator` | at `init` (wraps the caller's allocator in a mutex) |
| `allocator` | `std.mem.Allocator` | at `init` (fat pointer into `thread_safe_allocator`) |
| `backend` | `Backend` | at `init` (see §9) |
| `buffers` | `BufferPool` | at `init` (§6.5) |
| `work_pool` | `thread.Pool` | lazily, on first `tryWorkPool` (§6.6) |
| `dot_backward_worker` | `thread.OneShotWorker` | lazily, on first `dotBackwardWorker` (§6.6) |
| `scope_entries`, `scope_depth` | scope stack | at `init`, empty (§6.3) |

**The init(self-pointer) pattern.** `init` takes `self: *ExecContext` and
returns `void` instead of returning a value: the context is self-referential
(`rt.allocator` is a fat pointer into `rt.thread_safe_allocator`, and
`ctx.allocator` is a cached copy of it), so it must be initialized in place
at its final address and must never be moved or copied afterwards. The idiom:

```zig
test "context lifecycle" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc); // in place: the context is self-referential, never move it
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var y = try x.scale(&ctx, 2.0);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6, 8 }, try y.dataConst());
}
```

`init` cannot fail (it allocates nothing). `deinit` tears down in order:
MoE scratch, then the one-shot worker and the worker team (if they were ever
created; `backend.setWorkPool(null)` first), then any exec scopes still open
(defensive release), then the buffer pool. `BufferPool.deinit` asserts that
no pooled buffer is still outstanding — a tensor leaked past context teardown
fails this assertion in safety builds rather than silently leaking. After
`deinit` the struct is `undefined`.

**Substrate methods on the facade** (everything else on `ExecContext` is an
op, see §4):

```zig
pub fn execScopeActive(self: *const ExecContext) bool
pub fn openExecScope(self: *ExecContext) ExecScope
pub fn closeExecScope(self: *ExecContext, mark: ExecScope) void
pub fn reserveScopeSlot(self: *ExecContext) !void
pub fn adoptScopeValueAssumeCapacity(self: *ExecContext, value: Tensor,
    node: ?*anyopaque, destroy_node: ScopeNodeDestroy) void
pub fn adoptScopeNodeAssumeCapacity(self: *ExecContext, node: *anyopaque,
    destroy_node: ScopeNodeDestroy) void
pub fn tryWorkPool(self: *ExecContext) !*thread.Pool
pub fn workPool(self: *ExecContext) ?*thread.Pool
pub fn dotBackwardWorker(self: *ExecContext) ?*thread.OneShotWorker
pub fn classify(_: *const ExecContext, x: *const Tensor) LayoutClass
pub fn replace(self: *ExecContext, old: anytype, new_value: anytype) @TypeOf(new_value)
pub fn broadcastTo(self: *ExecContext, x: *const Tensor, shape: []const usize) !Tensor
pub fn broadcastToRank(self: *ExecContext, comptime rank: usize, x: *const Tensor, shape: [rank]usize) !Tensor
```

`classify` buckets a raw tensor's layout into
`LayoutClass = enum { contiguous, scalar, tail_broadcast, arbitrary }` — the
dispatch key elementwise kernels use to pick a fast path. `broadcastTo` /
`broadcastToRank` return zero-copy views (refcounted aliases, §6.2). The
scope and pool methods are covered below.

**MoE decode scratch** (`moe_scratch`, ops in `src/exec/moe.zig`). A
grow-only, mutex-guarded scratch region backing the single-row MoE decode
ops: the per-token region sizes are model constants, so after the first
token the hot path performs no allocations — one uncontended lock instead of
several allocator/pool round-trips per layer. The discipline is
`lockMoeDecodeScratch()` … carve … run … `unlockMoeDecodeScratch()`, holding
the lock for the whole op because the expert tasks write into the carved
slices. `carveMoeDecodeScratch(QgBlock, Task, hidden_blocks, top_k, out_pe,
hidden, blocks_per_g)` returns a `MoeDecodeScratchView(QgBlock, Task)` —
borrowed slices carved from the region (`qx` Q8_K activation blocks,
`gate_buf`/`up_buf`/`g_buf`, `qg`, `outs`, `tasks`), valid only while the
lock is held; `carveMoeDecodeChainScratch` adds a `states` slice and a
caller-sized task count for dependency-chained decode
(`MoeDecodeChainScratchView`). Every carved type must align to ≤ 8 (compile
error otherwise). This is the seam in-tree LLM-band code uses to build
custom MoE decode paths (`src/llm/gemma/moe.zig`, §13); `deinit` frees the
scratch with the context.

### 6.2 The memory model: who owns an op result (`docs/MEMORY-MODEL.md`)

The contract, in one sentence: **every tensor an op returns is owned by the
caller and must be deinitialized exactly once — unless an exec scope is open,
in which case op results are scope-owned borrows and `deinit` on them is a
safe no-op.** The full rationale (including why a generic arena allocator was
evaluated and rejected) is recorded in [MEMORY-MODEL.md](MEMORY-MODEL.md);
this subsection restates the operative rules.

Ownership by construction source:

| Tensor came from | Owner | `deinit` required? |
|---|---|---|
| any facade op result, **no scope open** | caller | yes, exactly once |
| any facade op result, **scope open** | innermost exec scope | no (safe no-op); never use after the scope closes |
| explicit constructors: `variable`, `constant`, `fromSlice`, `fromTensor`, `empty`, `zeros`, `ones`, `full`, `scalar`, … (§3) | caller | yes — even inside a scope |
| fetched gradients: `grad`, `gradView` (§5) | caller | yes — even inside a scope |
| `ctx.*` raw construction helpers (§6.4) | caller | yes — never scope-adopted |
| typed/quantized-constant tensor results (§3, §10) | caller | yes — typed ops are not scope-adopted |

**`deinit` is the recycling driver, not a naive free.** The chain is
`tensor.deinit()` → `buffer.release()` (atomic refcount decrement) → refcount
hits 0 → the buffer's release hook returns it to the `BufferPool` free list.
A released transient is immediately reusable by the next op — same-sized
successive allocations get the *same address* back, which keeps the hot
working buffer warm in cache. This is asserted behavior:

```zig
test "deinit recycles transient buffers through the pool" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try ctx.fromSlice(&.{3}, &.{ 1, 2, 3 });
    defer a.deinit();

    var first = try ctx.add(&a, &a);
    const first_ptr = first.dataConst().ptr;
    first.deinit(); // storage returns to the pool free list

    var second = try ctx.add(&a, &a); // same size: the pool hands back the same address
    defer second.deinit();
    try std.testing.expectEqual(first_ptr, second.dataConst().ptr);
}
```

**Two lifetime regimes.** Inference tensors are constants
(`grad_state == null`); their `deinit` releases storage immediately, so the
pool behaves arena-like *within* a forward pass — the idiomatic
`var x = try someOp(ctx, ...); defer x.deinit();` gives an O(1) working set.
Training variables retain their inputs: backward functions store operand
*views* at op-execution time, and a view bumps the storage refcount, so those
buffers cannot return to the pool until the tape node is destroyed in/after
`backward()`. Holding activations for backward is inherent to training, not
pool overhead.

**Views are refcounted aliases.** Every view operation (`cloneView`,
`reshape`, `broadcastTo`, `narrow`, strided views — §3, §8) retains the
source buffer and releases it on its own `deinit`; a view's lifetime is
independent of its parent's. Deinitializing a source tensor while views on it
live is safe — the storage survives until the last reference drops. This is
also why the runtime cannot use region-reset arenas: a zero-copy `narrow`
into a session-lifetime KV cache (§13) has a per-object lifetime no region
reset can express.

**The carried-value pattern: `ctx.replace`.** Residual streams and other
accumulators advance a single binding through many ops. `replace`
deinitializes the old value and returns the new one in one statement:

```zig
test "ctx.replace advances a carried tensor" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer x.deinit(); // runs on whatever x holds at scope exit
    for (0..3) |_| {
        // frees the old x and rebinds — on error the old x stays valid
        x = try ctx.replace(x, x.scale(&ctx, 2.0));
    }
    try std.testing.expectEqualSlices(f32, &.{ 8, 16 }, try x.dataConst());
}
```

`new_value` must be an error union of `@TypeOf(old)` (compile error
otherwise). On error the old value is *not* consumed — the caller's binding
and `defer`/`errdefer` arms stay valid and the error propagates. On success
the old value is released (one reference) and the new value returned for
rebinding. Inside an exec scope the release is a safe no-op on scope-owned
results, so the same forward code is training-safe. `replace` is generic
over any owned value with a `deinit` method (tagged tensors, projection
structs).

**Why not an arena.** Summarizing MEMORY-MODEL.md §4: a per-forward reset
arena would (i) balloon peak memory from the working set (~6–12 live
transients per block) to the sum of all intermediates, (ii) destroy the
address-reuse cache locality shown above, (iii) be unable to express
refcounted views and KV-cache aliasing, and (iv) be incorrect for training,
where activations must outlive the forward. The `BufferPool` already
delivers allocation amortization — the only real arena advantage — plus
intra-pass reuse and a bounded cap.

### 6.3 Exec scopes: implicit ownership for training (`src/exec.zig`, `src/exec/runtime.zig`)

Training breaks deinit-ASAP: every intermediate on the path from the
parameters to the loss must stay alive until `backward()` returns, because
each differentiable result owns a single-owner `GradState` graph node that
`deinit` destroys unconditionally (see §5 and
[TRAINING.md](TRAINING.md) §2). Exec scopes make the context itself the
owner of those intermediates, so training forward passes look like inference
code.

```zig
pub const ExecScope = struct { index: usize };
pub const ScopeNodeDestroy = *const fn (*anyopaque) void;

pub fn openExecScope(self: *ExecContext) ExecScope
pub fn closeExecScope(self: *ExecContext, mark: ExecScope) void
pub fn execScopeActive(self: *const ExecContext) bool
```

Semantics:

- **While a scope is open, every tensor returned by a facade op is adopted
  by the innermost scope.** The value the caller receives is a borrow with
  its `scope_owned` flag set: `deinit` on it is a safe no-op (arena-style),
  and using it after the scope closes is use-after-free. Adoption covers
  both differentiable results and no-grad f32 results (eval on constants,
  the `values` arm of `topK`/`sort`, …) — it is wired into the op tails
  (`finishOp` / `finishNoGrad` in `src/ag/tensor.zig`). The i64 INDEX
  outputs (`argmax`, `argsort`, the `indices` arms) are typed constants
  and stay caller-owned (below).
- **What stays caller-owned even inside a scope:** tensors created
  explicitly (`variable`, `constant`, `fromSlice`, and the other §3
  constructors), fetched gradients (`grad` / `gradView`), the raw `ctx.*`
  construction helpers of §6.4, and results of typed/quantized-constant
  tensor methods (weights and caches have explicit lifetimes).
- **`closeExecScope(mark)` releases every tensor adopted since `mark`,
  newest first**, destroying each adopted graph node through its registered
  destructor. Close a scope only when no `backward()` over tensors adopted
  in it is pending.
- **Scopes nest with strict stack discipline** — close in reverse order of
  opening. A nested scope releases only its own suffix; values adopted by an
  outer scope survive an inner close.
- **Error safety for free:** if an op fails mid-forward, the scope already
  owns the prefix of results, so model code inside a scope needs no
  `errdefer` chains.
- **Not thread-safe:** open/close and the ops between them run on one thread,
  like every other context mutation (§6.9).

The canonical training-step pattern — a per-iteration scope, no keeps, no
defers on intermediates:

```zig
test "training step under an exec scope" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var w = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer w.deinit(); // parameters stay caller-owned

    for (0..2) |_| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope); // releases all adopted intermediates, newest first
        const y = try w.mul(&ctx, &w); // scope-owned borrow: no defer needed
        const loss = try y.sumAll(&ctx);
        try loss.backward(&ctx);

        var gw = (try w.grad(&ctx)).?; // fetched gradients stay caller-owned
        defer gw.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, try gw.dataConst());
        w.zeroGrad();
    }
}
```

**Write-once forward code.** Because `deinit` on a scope-owned result is a
no-op, defer-deinit forward code — the inference idiom, including
`ctx.replace` for residual streams — runs unchanged under a scope. Write the
forward once; train it by opening a scope around it:

```zig
test "deinit on a scope-owned result is a safe no-op" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer x.deinit();

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    var y = try x.add(&ctx, &x);
    defer y.deinit(); // no-op: the scope owns y — the same code runs unscoped
    try std.testing.expectEqualSlices(f32, &.{ 6, 8 }, try y.dataConst());
}
```

Scope-related errors (recoverable, not panics):

- `error.ActiveExecScopeRequired` — facade-level *composed* differentiable
  ops (`nllLoss`, `l2Normalize`, `cosineSimilarity`, §4) build function-local
  intermediate graph nodes; when gradients are tracked, only a scope can own
  them until backward, so calling them with gradients enabled and no scope
  open is a loud error instead of undefined behavior. No-grad composition
  works unscoped.
- `error.ActiveExecScopeUnsupported` — the storage-consuming
  `takeAddNoGrad` / `takeScaleNoGrad` (§4) refuse a scope-owned operand:
  consuming a borrow would double-free at close.

**Scopes are a training tool, not an inference optimization.** A held scope
inverts the pool discipline: with deinit-ASAP a chain of same-shaped ops
recycles ~2 pooled buffers (O(1) working set, warm addresses), while a scope
keeps every intermediate live until close (O(N), cold addresses) — measured
2 vs 32 distinct buffers on a 32-op chain
([MEMORY-MODEL.md](MEMORY-MODEL.md) §5; the behavior is test-pinned in
`src/ag/tensor_tests.zig`). For pure inference, deinit-ASAP with no scope is
the discipline; scopes are correct where holding the graph *is* the
semantics (training), and harmless on cold no-grad paths.

**Extension point.** Op implementers (e.g. `fucina.customVjp`, §5) use the
two-phase adoption API so op construction stays infallible after its
"consumes the value on success" point: `reserveScopeSlot()` (fallible)
*before* building the result, then `adoptScopeValueAssumeCapacity(value,
node, destroy_node)` (infallible) after. `node` is a type-erased per-op
payload (the autograd facade stores its backward node there) released via
`destroy_node` at scope close — the exec layer itself knows nothing about
autograd types.

### 6.4 Raw construction and copy helpers on ctx (`src/exec.zig`, `src/exec/runtime.zig`)

These methods build *raw* tensors — the internal, tag-free, no-grad tensor
type (§8; deliberately not exported at the `fucina` root). Application code
normally uses the tagged facade constructors of §3, which wrap these; the
raw helpers appear in public signatures wherever a facade constructor takes
a `RawTensor` (e.g. `Tensor(spec).variable(&ctx, try ctx.fromSlice(...))`)
and throughout runtime-extension code. Results are **always caller-owned and
never scope-adopted**; pair each with `deinit` (or hand ownership to a
facade constructor, which consumes the value on success and leaves it with
the caller on error).

Allocation (uninitialized / filled), all pool-backed:

| Function | Result | Notes |
|---|---|---|
| `empty(shape)` / `emptyRank(rank, shape)` | f32, uninitialized | slice-shape vs comptime-rank array-shape variants |
| `emptyRankTyped(dtype, rank, shape)` | `TensorOf(dtype)`, uninitialized | non-f32 dtypes route to the slab arm (§6.5) |
| `zeros(shape)` / `zerosTyped(dtype, shape)` | zero-filled | |
| `ones(shape)` / `onesRank(rank, shape)` / `onesTyped(dtype, shape)` / `onesRankTyped(dtype, rank, shape)` | one-filled | |
| `full(shape, value)` / `fullTyped(dtype, shape, value)` | filled with `value` | typed variant takes `Scalar(dtype)` |
| `scalar(value)` | shape `{1}` f32 | |

Copy-in from caller data:

| Function | Semantics |
|---|---|
| `fromSlice(shape, values)` / `fromSliceRank(rank, shape, values)` | copy `[]const f32` into pooled storage |
| `fromSliceTyped(dtype, shape, values)` / `fromSliceRankTyped(dtype, rank, shape, values)` | copy `[]const Scalar(dtype)` |
| `fromStorageSliceRankTyped(dtype, rank, shape, values)` | copy `[]const Storage(dtype)` (block-quantized payloads, §10) |
| `fromBorrowedSliceRank(rank, shape, values)` | **zero-copy** wrap of caller-owned `[]f32`; the tensor borrows — keep the slice alive and unmoved until the tensor's `deinit`, which frees only the header |
| `fromBorrowedSliceRankTyped(dtype, rank, shape, values)` / `fromBorrowedStorageSliceRankTyped(dtype, rank, shape, values)` | typed zero-copy wraps, same borrow contract |

Copy/materialize existing tensors:

| Function | Semantics |
|---|---|
| `materialize(x)` / `materializeTyped(dtype, x)` | contiguous pooled copy of a (possibly strided/broadcast) view |
| `clone(x)` | alias for `materialize` |

Errors: `TensorError.InvalidDataLength` when `values.len` does not match the
shape's element count; `TensorError.InvalidShape` for rank 0, rank above the
max, or any zero dimension (and, for block-quantized dtypes, an innermost
dimension not divisible by the block size); `error.Overflow` on element-count
overflow; `error.OutOfMemory` from the pool. Borrowed wraps allocate only a
buffer header and never enter the pool's free lists.

```zig
test "fromSlice copies; fromBorrowedSliceRank wraps caller storage" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var source = [_]f32{ 1, 2, 3, 4 };
    var copied = try ctx.fromSlice(&.{ 2, 2 }, &source); // pooled copy, caller-owned
    defer copied.deinit();
    source[0] = 99;
    try std.testing.expectEqual(@as(f32, 1), copied.dataConst()[0]);

    var borrowed = try ctx.fromBorrowedSliceRank(2, .{ 2, 2 }, source[0..]); // zero-copy
    defer borrowed.deinit(); // frees only the header; `source` stays caller-owned
    try std.testing.expectEqual(@as(f32, 99), borrowed.dataConst()[0]);
}
```

Internal substrate helpers on `Runtime` (not forwarded to the facade; for
runtime extenders): `emptyTyped`, `scalarTyped`, `zerosRank`,
`zerosRankTyped`, `cloneTyped`, and the contiguity-preparation pair
`prepareContiguous` / `prepareContiguousTyped`
returning `PreparedTensor` / `PreparedTensorOf(dtype)` — a
borrowed-or-owned union whose `deinit` is a no-op on the borrowed arm, so
hot paths can `defer prepared.deinit()` unconditionally.

**The raw op surface and its naming grammar.** Beyond these constructors,
`ExecContext` carries the full raw op surface — roughly 300 `pub fn`s in
`src/exec.zig`, whose recurring suffixes follow one grammar: `*Rank`
entries take a comptime rank parameter first and raw-tensor pointers as
arguments (`addRank`, `mulRank`, `gluRank`); `*AxisRank` entries add a
comptime axis index
(`splitSwiGluAxisRank`, `conv1dAxisRank`); `*Typed` entries take a comptime
`DType` for non-f32 storage (`addRankTyped`, `castTyped`, `scaleTyped`); and
`*Backward*` entries are the VJP kernels `src/ag/backward.zig` dispatches to
(`conv2dBackwardInput`, `dropoutBackward`, `splitSwiGluBackwardAxisRank`).
This is the surface `customVjp` forward/backward specs (§5.6) are written
against. `src/exec.zig` is the source of truth; the domain modules under
`src/exec/` it forwards to are not public API.

### 6.5 BufferPool: transient reuse and scratch leases (`src/exec/buffer_pool.zig`)

`BufferPool` (re-exported as `exec.BufferPool`; one instance per context at
`ctx.rt.buffers`) recycles owned, refcounted storage buffers across ops.
Kernels never allocate — the `Runtime` allocation primitives of §6.4 are the
only source of transient tensors, and all of them draw from the pool. Two
arms share one byte budget:

- **The f32 arm** — a free list of `*storage.Buffer`, serving every
  default-dtype tensor (`acquire(len)`). In an LLM forward essentially all
  transient activations are f32, so this arm covers the hot path.
- **The byte-slab arm** — a free list of 64-byte-aligned, 4096-byte-rounded
  raw slabs serving every other storage dtype (`acquireTyped(dtype, len)`
  wraps a slab in a typed buffer header whose release hook returns the slab)
  plus non-DType packed block scratch via `acquireScratch(T, len)`. Slabs are
  reused across dtypes: an f16 slab released by one op can serve q8_k
  scratch in the next.

```zig
pub const slab_align = 64;          // covers max element alignment + cache line
pub const slab_size_quantum = 4096; // slab byte-size rounding

pub const BufferPool = struct {
    pub fn init(allocator: Allocator) BufferPool
    pub fn deinit(self: *BufferPool) void  // asserts outstanding == 0
    pub fn acquire(self: *BufferPool, len: usize) !*storage.Buffer
    pub fn acquireTyped(self: *BufferPool, comptime dtype: storage.DType, storage_len: usize) !*storage.BufferOf(dtype)
    pub fn acquireScratch(self: *BufferPool, comptime T: type, len: usize) !ScratchLease(T)
    pub fn cachedBuffers(self: *BufferPool) usize
    pub fn cachedSlabs(self: *BufferPool) usize
    pub fn cachedBytes(self: *BufferPool) usize
    pub fn outstandingBuffers(self: *const BufferPool) usize
};
```

Behavior users should know:

- **Size rounding.** f32 requests round to the next power of two up to 1024
  elements, then to the next 1024-element multiple; slab requests round to
  the next 4096-byte multiple. Rounding collapses nearby sizes into shared
  buckets, which helps reuse; a handed-back buffer may be larger than asked
  (tensors use the shape-covered prefix).
- **First-fit over an ascending free list.** `acquire` returns the smallest
  cached buffer whose capacity fits; on a miss it allocates fresh (releasing
  the pool mutex first). Releases insert *before* existing same-length
  entries, so within a size class reuse is LIFO — the most recently released
  buffer is handed back first. Same-size acquire/release cycles return the
  same address — the cache-locality property of §6.2.
- **Bounded retention.** `max_cached_bytes` (default 1 GiB, shared by both
  arms) caps *cached* (free-list) bytes, not live bytes: a released buffer
  that alone exceeds the cap, or would push the cache over it, is destroyed
  instead of cached. Steady-state retention is bounded by the actual peak
  transient set of the workload.
- **Leak detection.** An atomic `outstanding` counter tracks live pooled
  buffers; `BufferPool.deinit` (run by `ExecContext.deinit`) asserts it is
  zero.
- **Scratch leases.** `acquireScratch(T, len)` returns a
  `ScratchLease(T) { pool, slab, items: []T }` — borrowed pooled scratch for
  non-DType block types (the packed quantized-LHS layouts of §10). Call
  `lease.release()` exactly once; `items` is valid until then. Release may
  run on any thread.
- **Thread safety.** Both arms are mutex-guarded; buffer release hooks run
  on whatever thread drops the last reference. See §6.9.
- **What never enters the pool:** `fromBorrowed*` wraps, load-time weight
  packs, and backend-tier LHS-quantization scratch below the exec seam (a
  deliberate, documented exception).
- **Teardown retention.** Session-lifetime typed buffers (KV-cache f16
  layers, resident bf16 weights) are pool-backed, so tearing down a model
  *session* while keeping the context alive returns their slabs to the free
  list — retained up to the cap so the next session reuses warm slabs.
  `ExecContext.deinit` frees everything.

### 6.6 The worker team (`src/thread.zig`, `src/parallel.zig`)

CPU kernels parallelize over a persistent fork-join team owned by the
context. Everything is lazy: `Runtime.tryWorkPool` creates the
`thread.Pool` on first request (with `cpuThreadCount(vector_max_threads) - 1`
workers, so the dispatching thread itself is participant 0) and hands it to
the backend via `setWorkPool`; the pool in turn spawns its worker threads
only on the first parallel dispatch. A context that only ever runs
small/serial ops never starts a thread.

```zig
pub fn tryWorkPool(self: *ExecContext) !*thread.Pool   // creates on first call
pub fn workPool(self: *ExecContext) ?*thread.Pool      // tryWorkPool catch null
pub fn dotBackwardWorker(self: *ExecContext) ?*thread.OneShotWorker
```

The team (`BarrierPool` in `src/thread.zig`) is spin-then-park: after each
dispatch a worker spins on a generation counter for a bounded budget
(default 32768 `spinLoopHint` iterations — the measured M1 tuning, which
survived an x86 sweep), then parks on a futex, so a dense op stream (a
transformer forward) pays atomics instead of kernel round-trips while a
long-idle team consumes no CPU. The dispatcher runs chunk 0 of every
parallel op and pure-spins on the completion counter — it owns a core until
the join. On macOS, workers and dispatcher pin to performance-core QoS
(`pthread_set_qos_class_self_np`); elsewhere the pin compiles to nothing.
`dotBackwardWorker` is a single lazily-started `OneShotWorker` used to
overlap the two branches of matmul backward on native-BLAS builds (§5, §9).

Thread-count knobs, in precedence order:

| Knob | Kind | Effect |
|---|---|---|
| `-Dmax-threads=N` (1–64, default 8 = the M1 Max P-core count) | build option | comptime ceiling for the team and stack task arrays, AND the runtime default team size (`fucina.parallel.vector_max_threads`). Servers with more cores must raise it at build time |
| `fucina.parallel.setMaxThreads(n)` | runtime API | **replaces** the detected CPU count (mirrors llama.cpp `-t`) — it can also *raise* the team size above the detected count, up to the `-Dmax-threads` build ceiling; call once at startup before any parallel work; `n == 0` ignored; wins over the env var by pre-seeding the cache |
| `FUCINA_MAX_THREADS` | env var | read once on the first `cpuThreadCount` call; applied as `@min` against the detected count, so it **only lowers**; `0`/invalid = no override |
| `FUCINA_SPIN_BUDGET` | env var | overrides the spin-then-park window, read once per team init; workload-coupled and U-shaped — override only with measurements (short budgets, ~512, favor encode-style workloads with serial host sections; the default favors dense LLM op streams) |

The effective thread count is `parallel.cpuThreadCount(vector_max_threads)`
= `max(1, min(count, vector_max_threads))`, where `count` is the
`setMaxThreads` value verbatim when one was set (detection is bypassed),
else the detected CPU count — clamped to the physical-core count on SMT
machines, so hyperthreads are never double-booked; `setMaxThreads` remains
the escape hatch for deliberate oversubscription — lowered by
`FUCINA_MAX_THREADS`. No single
value wins every workload (measured: prefill fastest at all P-cores when
cool, decode often faster one or two threads lower), hence the runtime
knobs. The env parsers behind these knobs are themselves public:
`parallel.envPositiveUsize(name)` implements the positive-usize knob
contract (libc `getenv`, or a libc-free `/proc/self/environ` scan on static
Linux; unset/invalid/`0` ⇒ `null`), and `parallel.envSpinBudget()` is the
`FUCINA_SPIN_BUDGET` read consulted once per team init.

```zig
test "worker-team sizing knobs" {
    // Comptime team ceiling from -Dmax-threads (1-64, default 8).
    try std.testing.expect(fucina.parallel.vector_max_threads >= 1);
    // Runtime cap (mirrors llama.cpp -t and FUCINA_MAX_THREADS); call once
    // at startup, before any parallel work.
    fucina.parallel.setMaxThreads(2);
    const n = fucina.parallel.cpuThreadCount(fucina.parallel.vector_max_threads);
    try std.testing.expect(n >= 1 and n <= 2);
}
```

For direct use of the pool (custom parallel sections):

```zig
pub const Pool = struct {
    pub fn init(self: *Pool, options: InitOptions) !void;   // .allocator, .max_workers
    pub fn deinit(self: *Pool) void;
    pub fn parallelChunks(self: *Pool, comptime Task: type,
        tasks: []const Task, comptime run: fn (*const Task) void) void;
    pub fn parallelChained(self: *Pool, comptime Task: type, tasks: []Task,
        initial_count: usize, comptime run: fn (*Task, *const Chain) void) bool;
    pub fn spawnWg / trySpawnWg / waitAndWork;               // std.Io-executor tasks
};
```

`parallelChunks` is the default substrate for splitting a numeric kernel:
fork-join over the hot team, the caller executing chunk 0 and the team the
rest, rendezvousing before return. Degradation is always safe: no barrier,
zero workers, or a team already mid-dispatch (`parallel_chunks_active`)
runs the tasks serially on the caller. `parallelChained` is
dependency-chained fork-join: `tasks[0..initial_count)` start runnable and
a running task makes successors runnable via `chain.enqueue(i)`; it returns
`false` when the team is unavailable or busy (the caller must run the graph
itself). The enqueue contract is strict: across one dispatch, every index
in `[0, tasks.len)` must become runnable **exactly once** (seeds plus
enqueues). Debug/ReleaseSafe builds instrument the contract and panic on
double-enqueue or a stalled under-enqueue; ReleaseFast compiles the checks
out, where a violation corrupts the intrusive Treiber stack (the pop is
ABA-unsafe) or spins forever. `spawnWg` / `trySpawnWg` / `waitAndWork`
route general async tasks through `std.Io`'s executor with a
`thread.WaitGroup` — unlike the hot team, each spawn heap-allocates a task
node and parks/wakes via futex syscalls. Per-kernel threading policy (work
thresholds such as `parallel.vector_matmul_work_threshold`, claim-chunk
sizing) and the backend-side pool handshake are §9 material (§9.4, §9.8).
`ParallelConfig` is an internal backend/vector type, not part of the public
surface.

### 6.7 RhsLifetime: address-keyed caching of RHS operands (`src/exec/quant_matmul.zig`)

```zig
pub const RhsLifetime = enum {
    transient,       // default: no address-keyed caching beyond this dispatch
    stable_process,  // caller guarantees stable bytes; backends may cache wraps
    pub fn isCacheable(self: RhsLifetime) bool // true iff .stable_process
};
```

Re-exported as `fucina.RhsLifetime`. GPU backends (§9) avoid re-wrapping and
re-uploading a quantized weight on every matmul by caching device wraps
keyed on the RHS byte address. That is only sound if the bytes at that
address never change, which the type states explicitly:

- `.transient` — ordinary tensor/temporary storage. The backend may still
  use the GPU for the dispatch, but must not retain an address-keyed wrap.
- `.stable_process` — the caller guarantees the RHS bytes stay mapped at the
  same address for the process lifetime (an mmap'd weight file kept mapped),
  or are device-resident storage from
  `fucina.internal.gpu.allocResidentBytes` whose owner evicts cached wraps
  via `freeResidentBytes` before freeing. Violating the promise is
  use-after-free on the GPU side.

The hard rule: **pooled storage must never be marked `.stable_process`.**
The slab arm makes address reuse routine, so a cached wrap keyed on a pooled
transient's address would silently read stale data after the slab is
recycled. Every in-tree `.stable_process` caller wraps device-resident or
mmap'd weight bytes (`src/llm/weights.zig` threads the flag through the
quantized-weight loaders via `QuantizedMatmulOptions.rhs_lifetime`). This is
about storage stability, not about whether the operand is a model weight.

### 6.8 Determinism and the RNG contract (`src/rng.zig`)

`fucina.rng` is the repo-owned deterministic RNG (splitmix64-based). Its
(seed → values) mappings are **checkpoint contracts**: consumers store a
seed and regenerate values instead of serializing them, so none of these
functions may ever change behavior or depend on `std.Random` internals
(which are free to change across Zig releases).

```zig
pub fn splitmix64(state: *u64) u64                     // one sequential step
pub fn at(seed: u64, i: u64) u64                       // i-th output, O(1), counter-based
pub fn gaussianFill(seed: u64, out: []f32, scale: f32) void
pub fn gaussianFillAt(seed: u64, first: u64, out: []f32, scale: f32) void
pub fn gaussianFillAtFast(seed: u64, first: u64, out: []f32, scale: f32) void
pub fn uniformFill(seed: u64, out: []f32, lo: f32, hi: f32) void
pub fn kaimingUniformFill(seed: u64, out: []f32, fan_in: usize) void
pub fn normalFill(seed: u64, out: []f32, mean: f32, std_dev: f32) void
```

- `at(seed, i)` computes the i-th output of the stream started at `seed`
  directly — every element is a pure function of `(seed, i)`, independent of
  the preceding ones.
- `gaussianFill` is splitmix64 + Box-Muller (two stream outputs per value
  pair); `gaussianFillAt` is its counter-based form: filling elements
  `first .. first + out.len` of the same stream, bitwise identical to the
  sequential fill under **any** range decomposition.
- `gaussianFillAtFast` is a vectorized variant with f32 polynomial
  transcendentals. It is a **distinct** (seed → values) mapping — values
  agree with `gaussianFillAt` to a few ulps but are not bitwise equal — and
  therefore a separate checkpoint contract. It is equally
  chunking-invariant.
- `uniformFill` maps one output per value onto `[lo, hi)` (half-open bound
  kept exact by clamping the rare round-up); `kaimingUniformFill` is the
  PyTorch `nn.Linear`/LoRA-A default init; `normalFill` adds explicit
  moments on top of `gaussianFill`.

```zig
test "counter-based rng reproduces the sequential stream chunk by chunk" {
    const rng = fucina.rng;

    var state: u64 = 42;
    for (0..8) |i| {
        const sequential = rng.splitmix64(&state);
        try std.testing.expectEqual(sequential, rng.at(42, i)); // O(1) random access
    }

    var whole: [6]f32 = undefined;
    rng.gaussianFillAt(7, 0, &whole, 1.0);
    var parts: [6]f32 = undefined;
    rng.gaussianFillAt(7, 0, parts[0..2], 1.0);
    rng.gaussianFillAt(7, 2, parts[2..], 1.0); // any chunking, identical bits
    try std.testing.expectEqualSlices(f32, &whole, &parts);
}
```

Where the contract is load-bearing:

- **Dropout** (§4, §5): the mask is never stored — forward, backward, and
  `checkpoint` recompute all regenerate it from `(seed, element index)` via
  `at`, so the op is a deterministic pure function of `(input, p, seed)` and
  parallel kernels are bitwise-stable regardless of chunking. Pass a fresh
  seed per call (reusing a seed reuses the mask); eval mode is simply not
  calling dropout.
- **APOLLO** (§11, `src/optim.zig`): low-rank projections are regenerated
  from their stored seed at checkpoint restore via `gaussianFill`, not
  serialized.
- **Evolution strategies** (§11, `src/es.zig`): member perturbations are
  regenerated from member seeds via `gaussianFillAtFast` in parallel chunks,
  with results independent of the chunking.

These are the determinism guarantees the runtime makes: seed-driven ops are
bitwise reproducible across runs, thread counts, and chunk decompositions.
No blanket bitwise-reproducibility claim is made for every parallel
reduction at every thread count; where a kernel guarantees serial/parallel
parity, the guarantee is pinned by its tests (§9).

### 6.9 The thread-safety contract

What is thread-safe inside a context:

- **The allocator**: `Runtime` wraps the caller's allocator in
  `thread.ThreadSafeAllocator` (a mutex around alloc/resize/remap/free), so
  internal allocations and frees may happen on worker threads. `ctx.allocator`
  is this wrapper.
- **The BufferPool**: both arms are mutex-guarded, `outstanding` is atomic,
  and buffer/slab release hooks run on whatever thread drops the last
  reference (storage refcounts are atomic).
- **Lazy initialization**: `tryWorkPool` and `dotBackwardWorker` are
  mutex-guarded and idempotent.

What is not:

- **Op execution, scope open/close, and every other context mutation are
  single-threaded**: drive one `ExecContext` from one thread at a time. CPU
  ops fan work out to the team and join before returning. Eligible f32/f16/dense-quant
  GPU ops submit before return and keep program order through their provider
  queue/stream; a later CPU access performs the storage readiness wait. The
  external call order remains serial.
- **Sharing tensor handles across threads is unspecified.** The runtime
  makes no promise about concurrent reads or writes through tensor handles
  on different threads (storage refcounts are atomic, but the handle structs
  are mutable value types with interior pointers). Confine a tensor and its
  views to the thread driving its context, or synchronize externally.
  CPU parallelism inside the runtime — kernel chunking, ES perturbation fills,
  dot-backward branches — is always mediated by the context's own team and
  joins before the op returns. Submitted GPU completion is the explicitly
  documented exception (`GPU-OFFLOAD.md`).

## 7. Named axes: the tag algebra

Fucina names tensor axes with **tags** — Zig enum literals such as `.batch`,
`.seq`, `.d_model` — and every axis-level decision (broadcasting, contraction,
reduction, permutation) is made by tag identity, never by axis position at the
call site. Tags are **comptime-only data**: there is no runtime tag
representation and no tagged tensor *type*. The single runtime tensor currency
is the raw tensor (§8); the public `Tensor(tags_spec)` facade (§3, §4) carries
its tag tuple purely in the type and re-attaches result tags at comptime after
each op.

Two internal modules implement this:

- `src/tags.zig` — the pure comptime tuple algebra: spec normalization,
  lookup, uniqueness/subset constraints, and result-tag computation. Every
  function here runs at compile time and violations are compile errors.
- `src/tagged.zig` — the tag-semantics op library: functions that take
  comptime tag tuples plus `*const` raw tensors and return **owned** raw
  tensors (tag-directed views, tag-driven broadcasting, multi-axis reduction,
  and `taggedDot` lowering onto the ExecContext matmul/bmm kernels, §6).

Neither module is re-exported at the public root (`src/fucina.zig`): users
consume these semantics through `Tensor` methods, and the autograd VJPs
(`ag/backward.zig`, §5) call the same library directly on raw gradients. This
section is the semantics contract for the public surface and the reference for
the internal library. Snippets demonstrate the semantics through the public
facade.

### 7.1 Tags, tag specs, and normalization (`src/tags.zig`)

```zig
pub const Tag = @TypeOf(.tag);                    // the enum-literal type
pub const inserted_axis = std.math.maxInt(usize); // axis-map sentinel (§7.3)
```

Any enum literal is a `Tag`; two tags are equal iff their spellings are equal
(`tagEqual` compares `@tagName` strings at comptime). Tag names carry no
built-in meaning — `._0`…`._7` are ordinary tags that happen to be generated
for rank specs.

Everything that accepts axes accepts a **tag spec**, normalized by:

```zig
pub fn normalizeTags(comptime tags_spec: anytype) [tagSpecLen(tags_spec)]Tag
pub fn tagSpecLen(comptime tags_spec: anytype) usize
pub fn dtypeFromSpec(comptime tags_spec: anytype) DType   // defaults to .f32
pub fn isTensorSpec(comptime tags_spec: anytype) bool
pub fn isRankSpec(comptime tags_spec: anytype) bool
pub fn rankFromSpec(comptime rank_spec: anytype) usize
pub fn autoTags(comptime rank: usize) [rank]Tag           // ._0, ._1, ... ._7
```

| Spec form | Example | Normalizes to |
|---|---|---|
| Tag tuple | `.{ .batch, .d }` | the tuple itself |
| Integer rank | `3` | `autoTags(3)` = `.{ ._0, ._1, ._2 }` |
| Struct spec | `.{ .dtype = .u16, .tags = .{ .batch, .seq } }` | the `tags` field |
| Struct spec (rank) | `.{ .dtype = .i64, .rank = 2 }` | `autoTags(2)` |

`isTensorSpec` recognizes a non-tuple struct with a `dtype`, `tags`, or `rank`
field; a struct spec with neither `tags` nor `rank` is a compile error
(`"tensor dtype specs must include tags or rank"`). `dtypeFromSpec` reads the
optional `.dtype` field and defaults to `.f32` — this is how non-f32 typed
tensors get tags (§3). `rankFromSpec` rejects negative ranks (`"tensor rank
must be non-negative"`), non-integer specs (`"tensor tags must be a tag tuple
or a comptime rank"`), and ranks above `max_rank = 8` (`src/tensor.zig`,
`"too many tensor tags"`).

The public facade exposes the normalized result as comptime type members:
`Tensor(spec).axis_tags`, `.tag_count`, `.tensor_rank`, plus per-tag lookup
`axis(tag)`, `hasTag(tag)`, and runtime `dim(tag)` / `shape()` (§3).

```zig
test "tag specs and comptime introspection" {
    const M = fucina.Tensor(.{ .batch, .d }); // explicit tag tuple
    const R = fucina.Tensor(2); // rank spec: auto tags ._0, ._1
    const S = fucina.Tensor(.{}); // scalar: empty tag tuple, raw rank 1
    comptime {
        std.debug.assert(M.axis(.d) == 1); // axis position by tag
        std.debug.assert(M.hasTag(.batch) and !M.hasTag(.channel));
        std.debug.assert(R.axis(._1) == 1 and R.tag_count == 2);
        std.debug.assert(S.tag_count == 0 and S.tensor_rank == 1);
    }
}
```

### 7.2 Lookup, equality, and constraint helpers (`src/tags.zig`)

```zig
pub fn tagEqual(comptime a: anytype, comptime b: anytype) bool
pub fn tagsEqual(comptime a: anytype, comptime b: anytype) bool   // elementwise + length
pub fn tagIndex(comptime tags: anytype, comptime tag: anytype) ?usize
pub fn tagIndexOrCompileError(comptime tags: anytype, comptime tag: anytype) usize
pub fn validateUniqueTags(comptime tags: anytype) void
pub fn validateSameTagSet(comptime source_tags: anytype, comptime target_tags: anytype) void
pub fn rawRank(comptime tag_count: usize) usize   // 0 -> 1, else tag_count
```

- `tagIndex` returns the axis position of a tag within a tuple, or `null`;
  `tagIndexOrCompileError` fails compilation with `"tensor tag not found"`.
- `validateUniqueTags` enforces the global uniqueness invariant — a tag tuple
  never repeats a tag (`"duplicate tensor tag"`). `Tensor(spec)` validates
  this at type construction, so no public tensor type can carry duplicates.
- `validateSameTagSet` is the permutation precondition: same length
  (`"permutation requires the same rank"`) and same membership
  (`"permutation target must contain the same tags"`).
- `rawRank` maps the tag count to the raw tensor rank: the empty tag tuple
  (scalar) is stored as a rank-1 raw tensor of shape `{1}` — there are no
  rank-0 raw tensors (§8).

### 7.3 Tuple rewrites and axis maps (`src/tags.zig`)

These compute the *result* tag tuple of an op; the facade uses them directly
in return types, so shape errors in tag terms surface as compile errors at the
call site.

```zig
pub fn removeTag(comptime tags: anytype, comptime tag: Tag) [tags.len - 1]Tag
pub fn removeTags(comptime tags: anytype, comptime remove_tags: anytype) [tags.len - remove_tags.len]Tag
pub fn replaceTag(comptime tags: anytype, comptime old_tag: Tag, comptime new_tag: Tag) [tags.len]Tag
pub fn insertTagAt(comptime tags: anytype, comptime tag: Tag, comptime axis_index: usize) [tags.len + 1]Tag
pub fn splitTags(comptime tags: anytype, comptime tag: Tag, comptime split_tags: anytype) [tags.len + split_tags.len - 1]Tag
pub fn mergeTags(comptime tags: anytype, comptime out_tag: Tag, comptime merge_tags: anytype) [tags.len - merge_tags.len + 1]Tag
pub fn mergeStartAxis(comptime tags: anytype, comptime merge_tags: anytype) usize
pub fn reduceAxesDescending(comptime tags: anytype, comptime reduce_tags: anytype) [reduce_tags.len]usize
```

Constraints (all compile errors):

| Helper | Enforced constraint |
|---|---|
| `removeTag` / `removeTags` | removed tags must exist; `remove_tags` unique |
| `replaceTag` | `old_tag` must exist; `new_tag` must not already exist elsewhere (`"replacement tensor tag already exists"`) |
| `insertTagAt` | index ≤ rank (`"insert axis out of bounds"`); tag must be new (`"inserted tensor tag already exists"`); result ≤ `max_rank` |
| `splitTags` | split axis must exist; `split_tags` non-empty, unique, and each must be new or equal to the split tag (`"split output tag already exists"`); result ≤ `max_rank` |
| `mergeTags` / `mergeStartAxis` | `merge_tags` non-empty, unique, and must appear **contiguously and in order** in `tags` (`"merge tags must be contiguous and in tensor order"`; a run overflowing the end of the tuple reports `"merge tags must be contiguous"`); `out_tag` must not collide with a retained tag unless it is one of the merged tags (`"merge output tag already exists"`) |

`reduceAxesDescending` maps reduce tags to axis indices sorted descending, so
a multi-axis reduction can strip one axis at a time without invalidating the
remaining indices (§7.8).

Axis-map helpers translate tag decisions into per-axis permutation vectors
consumed by the view machinery; the sentinel `inserted_axis` means "no source
axis — inject a size-1, stride-0 axis here":

```zig
pub fn identityAxes(comptime rank: usize) [rank]usize
pub fn alignAxes(comptime source_tags: anytype, comptime target_tags: anytype) [target_tags.len]usize
pub fn insertAxes(comptime rank: usize, comptime axis_index: usize) [rank + 1]usize
pub fn squeezeAxes(comptime rank: usize, comptime axis_index: usize) [rank - 1]usize
```

`alignAxes` requires the target tuple to be unique and a **superset** of the
source (`"target tags must include all source tags"`); each target position
maps to its source axis or to `inserted_axis`. These back the facade's
`withTags`, `alignTo`, `permuteTo`/`transpose`, `insertAxis`, and `squeeze`
(§4).

### 7.4 Result-tag computation: pointwise and dot (`src/tags.zig`)

```zig
pub fn pointwiseResultTags(comptime left_tags: anytype, comptime right_tags: anytype) [pointwiseResultLen(...)]Tag
pub fn pointwiseResultLen(comptime left_tags: anytype, comptime right_tags: anytype) usize
```

Pointwise result tags are the **union in operand order**: all left tags in
left order, then every right-only tag appended in right order. Operand order
therefore determines the physical layout of the materialized result
(`{.d}` + `{.batch, .d}` produces tags `{.d, .batch}`, not `{.batch, .d}`).
The union must fit `max_rank`.

For a contraction over `contract_tag`, the operands' tags partition into
three comptime classes:

- **batch tags** — present in both operands and not the contract tag
  (`dotBatchTags`/`dotBatchLen`), in left-operand order;
- **left free tags** — left-only, non-contract
  (`dotLeftFreeTags`/`dotLeftFreeLen`);
- **right free tags** — right-only, non-contract
  (`dotRightFreeTags`/`dotRightFreeLen`).

```zig
pub fn dotResultTags(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) [dotResultLen(...)]Tag
pub fn dotResultLen(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) usize
```

`dotResultTags` = batch ++ left free ++ right free. The contract tag must be
present in **both** operands (compile error otherwise), and the result must
fit `max_rank`. Contracting a vector against a vector yields the empty tuple —
a scalar tensor.

Four canonical **storage orders** drive kernel selection in `taggedDot`
(§7.9): they describe, for each operand, the tag order a direct kernel expects.

```zig
// All: (comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag)
pub fn dotLeftOrder(...) [left_tags.len]Tag        // batch ++ left_free ++ contract
pub fn dotLeftTransAOrder(...) [left_tags.len]Tag  // batch ++ contract ++ left_free
pub fn dotRightOrder(...) [right_tags.len]Tag      // batch ++ contract ++ right_free
pub fn dotRightTransBOrder(...) [right_tags.len]Tag// batch ++ right_free ++ contract
```

The einsum generalization derives every axis role from tag membership alone
(shared vs private × kept vs dropped):

```zig
pub const EinsumPart = enum { batch, contract, left_free, right_free, left_summed, right_summed };
pub fn einsumClassOfLeft(comptime right_tags: anytype, comptime out_tags: anytype, comptime tag: Tag) EinsumPart
pub fn einsumClassOfRight(comptime left_tags: anytype, comptime out_tags: anytype, comptime tag: Tag) EinsumPart
pub fn einsumPartTags(comptime left_tags, right_tags, out_tags: anytype, comptime part: EinsumPart) [einsumPartLen(...)]Tag
pub fn einsumPartLen(comptime left_tags, right_tags, out_tags: anytype, comptime part: EinsumPart) usize
pub fn einsumValidate(comptime left_tags: anytype, comptime right_tags: anytype, comptime out_tags: anytype) void
```

`einsumPartTags` reports each part in the owning operand's axis order; the
shared parts (batch, contract) are reported in LEFT order, matching the
`dot*` convention. `einsumValidate` compile-errors on duplicate output tags,
an output tag missing from both operands, or a result rank past `max_rank`.
The set helpers `unionTags`/`unionTagsLen` (the first tuple followed by
the second's tags not already present — a membership set, deliberately not
capped by `max_rank`) and `intersectTags`/`intersectTagsLen` (tags of the
first tuple also present in the second, first-tuple order) support the
einsum lowering and are general-purpose.

### 7.5 The op library contract (`src/tagged.zig`)

Every runtime function below takes comptime tag tuples plus `*const` raw
tensors and returns an **owned** raw tensor: the caller must `deinit` it.
Results that are views (`align`/`permute`/`broadcast`/`split`/`merge`, and
`cloneView` fast paths) retain the source's underlying buffer, so the view
stays valid even after the source tensor value is deinitialized (buffer
refcounting, §8) — but they share storage: writing through one aliases the
other. Functions hold no state of their own; allocation and kernel dispatch go
through the `*ExecContext` argument, whose concurrency contract applies (§6).
All failures are recoverable Zig errors; nothing in this layer panics.

Rank validation is shared:

```zig
pub fn validateTensorRank(comptime tags: anytype, value: *const RawTensor) !void
pub fn validateTensorRankOf(comptime tensor_dtype: DType, comptime tags: anytype,
                            value: *const TensorOf(tensor_dtype)) !void
```

An empty tag tuple requires a scalar value (`value.isScalar()`, i.e.
`len() == 1`); otherwise the raw rank must equal `tags.len`. Violations return
`TensorError.InvalidShape`. (`RawTensor` is `src/tensor.zig`'s `Tensor`,
spelled `fucina.internal.RawTensor` in-tree; `TensorOf` is its dtype-generic
form — §8.)

Runtime error summary for the whole library:

| Condition | Error |
|---|---|
| value rank ≠ tag count (or non-scalar with empty tags) | `TensorError.InvalidShape` |
| broadcast dim conflict (both ≠ 1 and unequal) | `TensorError.ShapeMismatch` |
| dot contract or batch dim mismatch | `TensorError.ShapeMismatch` |
| split factors don't multiply to the axis dim, or a zero factor | `TensorError.InvalidShape` |
| merge over stride-incompatible axes | `TensorError.UnsupportedView` |

### 7.6 Alignment, permutation, and broadcast views (`src/tagged.zig`)

```zig
pub fn alignTensorTo(comptime source_tags: anytype, source: *const RawTensor,
                     comptime target_tags: anytype) !RawTensor
pub fn alignTensorToOf(comptime tensor_dtype: DType, comptime source_tags: anytype,
                       source: *const TensorOf(tensor_dtype),
                       comptime target_tags: anytype) !TensorOf(tensor_dtype)
pub fn permuteTensorTo(comptime source_tags: anytype, source: *const RawTensor,
                       comptime target_tags: anytype) !RawTensor
pub fn broadcastTensorTo(comptime source_tags: anytype, source: *const RawTensor,
                         comptime target_tags: anytype,
                         target_shape: [target_tags.len]usize) !RawTensor
pub fn broadcastTensorToOf(comptime tensor_dtype: DType, ...) !TensorOf(tensor_dtype)
```

`alignTensorTo` is the workhorse view: it reorders axes into `target_tags`
order and, for every target tag absent from the source, injects a **size-1,
zero-stride axis**. Zero-copy always. Comptime preconditions: target unique,
target ⊇ source, target ≤ `max_rank`. An empty target returns a `cloneView`
(only reachable for scalar sources, since a non-empty source cannot be a
subset of an empty target).

`permuteTensorTo` adds `validateSameTagSet`: same tags, same rank — a pure
axis permutation with no injection.

`broadcastTensorTo` aligns first, then expands the aligned view to
`target_shape`: axes of size 1 (including injected ones) stretch to any size
with stride 0; a non-1 axis whose dim differs from the target returns
`TensorError.ShapeMismatch`. Still zero-copy. An empty target with a non-empty
source is a compile error (`"scalar broadcast target cannot drop source
tags"`) — broadcasting never drops axes.

Facade equivalents (§4): `alignTo`, `permuteTo`/`transpose`, `broadcastTo`
(differentiable, `BroadcastBackward`), plus `withTags` (relabel, same rank),
`insertAxis`, and `squeeze` built on the axis maps of §7.3.

```zig
test "alignTo reorders and injects singleton axes" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var y = try x.alignTo(&ctx, .{ .d, .batch, .channel }); // .channel absent -> size-1 axis
    defer y.deinit();
    const shape = y.shape();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 1 }, &shape);

    var copied = [_]f32{0} ** 6;
    try y.copyTo(&copied); // transposed traversal of the same buffer
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, &copied);
}
```

```zig
test "broadcastTo expands missing tags without copying" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var bias = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();

    var wide = try bias.broadcastTo(&ctx, .{ .batch, .d }, .{ 2, 3 });
    defer wide.deinit();
    var copied = [_]f32{0} ** 6;
    try wide.copyTo(&copied);
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30, 10, 20, 30 }, &copied);
}
```

### 7.7 Pointwise and gated broadcasting (`src/tagged.zig`)

```zig
pub const PointwiseOp = enum { add, sub, mul, div, max, min };

pub fn pointwise(comptime op: PointwiseOp,
                 comptime left_tags: anytype, left: *const RawTensor, ctx: *ExecContext,
                 comptime right_tags: anytype, right: *const RawTensor) !RawTensor
pub fn gatedPointwise(comptime op: GatedOp, ...same signature...) !RawTensor
```

Broadcasting is entirely tag-driven. Any two tag sets are compatible at
comptime as long as their union fits `max_rank`; per-axis compatibility is a
runtime check. For each tag of `pointwiseResultTags(left_tags, right_tags)`
(§7.4):

1. an operand missing the tag contributes dim 1;
2. equal dims pass through; a dim of 1 broadcasts to the other's dim
   (zero-stride, no copy); two unequal non-1 dims return
   `TensorError.ShapeMismatch`.

Both operands are broadcast to the result shape as views, then the
rank-matched ExecContext kernel runs (`addRank`/`subRank`/`mulRank`/`divRank`/
`maxRank`/`minRank`,
or `gatedRank` for `gatedPointwise`). `GatedOp` (§4) is
`enum { glu, swiglu, geglu, swiglu_clamp10 }` and computes `left * σ(right)`,
`left * silu(right)`, `left * gelu(right)`, and — for `.swiglu_clamp10` —
`left * silu(min(right, 10))` respectively (the gate side only; the
`up`-side clamp of §4.5's full clamped SwiGLU exists only in the fused MoE
kernels, §4.18). The output is always
a newly materialized contiguous tensor in result-tag order.

The shape half is exposed separately for callers that need it (VJPs, facade):

```zig
pub fn pointwiseShape(comptime result_tags: anytype,
                      comptime left_tags: anytype, left: *const RawTensor,
                      comptime right_tags: anytype, right: *const RawTensor) ![rawRank(result_tags.len)]usize
pub fn pointwiseShapeOf(comptime tensor_dtype: DType, ...) ![rawRank(result_tags.len)]usize
```

These report the **raw-rank** shape — a scalar result reports `{1}` — and
perform the same dim-by-dim validation.

On the facade this is `add`/`sub`/`mul`/`div`/`maximum`/`minimum` and
`gated`/`glu`/`swiglu`/`geglu`; the result type is
`Tensor(pointwiseResultTags(...))` (§4).

```zig
test "pointwise broadcasts by tag" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();

    var y = try x.add(&ctx, &bias); // .d aligns; missing .batch broadcasts
    defer y.deinit();
    comptime std.debug.assert(@TypeOf(y).axis_tags.len == 2); // tags {.batch, .d}
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, try y.dataConst());
}
```

Disjoint tag sets broadcast to their union — an outer product without any
reshape ceremony:

```zig
test "disjoint tags broadcast to the union" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var rows = try fucina.Tensor(.{.batch}).fromSlice(&ctx, .{2}, &.{ 1, 10 });
    defer rows.deinit();
    var cols = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer cols.deinit();

    var outer = try rows.mul(&ctx, &cols); // result tags {.batch, .d}
    defer outer.deinit();
    const shape = outer.shape();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, &shape);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 10, 20, 30 }, try outer.dataConst());
}
```

Same-tag axes with conflicting sizes fail at runtime:

```zig
test "incompatible dims fail with ShapeMismatch" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer a.deinit();
    var b = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 10, 20 });
    defer b.deinit();

    try std.testing.expectError(error.ShapeMismatch, a.add(&ctx, &b));
    try std.testing.expectError(error.ShapeMismatch, a.dot(&ctx, &b, .d));
}
```

### 7.8 Split, merge, flatten, and multi-axis reduction (`src/tagged.zig`)

```zig
pub fn splitAxisView(comptime source_tags: anytype, source: *const RawTensor,
                     comptime tag: Tag, comptime split_tags: anytype,
                     split_shape: [split_tags.len]usize) !RawTensor
pub fn mergeAxesView(comptime source_tags: anytype, source: *const RawTensor,
                     comptime out_tag: Tag, comptime merge_tags: anytype) !RawTensor
pub fn flattenTensor(ctx: *ExecContext, source: *const RawTensor) !RawTensor
pub fn sumManyTensor(comptime tags: anytype, source: *const RawTensor,
                     ctx: *ExecContext, comptime reduce_tags: anytype) !RawTensor
```

**`splitAxisView`** factors one axis into several named factor axes, zero-copy
on **any** source layout: factor strides derive from the split axis's own
stride (`stride(axis) × suffix-product of remaining factors`), so strided
views split fine. The factor dims must be non-zero and multiply exactly to the
source axis dim (`TensorError.InvalidShape` otherwise). Tag-level constraints
come from `splitTags` (§7.3).

**`mergeAxesView`** is the inverse: it collapses adjacent axes into one,
zero-copy — but only when the merged axes are **stride-compatible**, i.e. laid
out as an unsplit axis: for each adjacent pair,
`stride(i) == shape(i+1) × stride(i+1)`. A transposed or otherwise gapped
layout returns `TensorError.UnsupportedView` (no silent materialization; make
the tensor contiguous first, e.g. facade `materialize`, §4). The merged axis
takes the stride of the last merged axis; the merged dim product is
overflow-checked. Tag-level contiguity (`mergeTags`, §7.3) is checked at
comptime; stride compatibility is the runtime half of the same rule.

**`flattenTensor`** returns a rank-1 tensor of all elements in logical order:
a zero-copy reshape when the source is contiguous, otherwise it materializes
through the ExecContext first (owned result either way).

**`sumManyTensor`** reduces away `reduce_tags` (comptime: unique, subset of
`tags`). Fast paths: an empty reduce set returns a `cloneView`; reducing every
tag lowers to the full reduction `ctx.sum` (scalar `{1}`). Otherwise it strips
one axis per step with `ctx.sumAxisRank`, innermost-first via
`reduceAxesDescending` (§7.3) so remaining axis indices stay valid.

Facade equivalents (§4): `split`, `merge` (both differentiable view ops),
`flatten(ctx, out_tag)` → `Tensor(.{out_tag})`, `sumMany` →
`Tensor(removeTags(tags, reduce_tags))`, and `sumAll` → `Tensor(.{})`.

```zig
test "split and merge rename factor axes as views" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .d_model }).fromSlice(&ctx, .{ 2, 6 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    var heads = try x.split(&ctx, .d_model, .{ .head, .head_dim }, .{ 2, 3 });
    defer heads.deinit(); // tags {.batch, .head, .head_dim}, shape {2,2,3}
    const hs = heads.shape();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3 }, &hs);

    var flat = try heads.merge(&ctx, .features, .{ .head, .head_dim });
    defer flat.deinit(); // tags {.batch, .features}, shape {2,6}
    try std.testing.expectEqualSlices(f32, try x.dataConst(), try flat.dataConst());
}
```

```zig
test "merge rejects stride-incompatible layouts" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .a, .b }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var t = try x.permuteTo(&ctx, .{ .b, .a }); // zero-copy transposed view
    defer t.deinit();

    // Tag-contiguous, but the transposed strides cannot collapse into one axis.
    try std.testing.expectError(error.UnsupportedView, t.merge(&ctx, .m, .{ .b, .a }));
}
```

```zig
test "sumMany reduces several named axes at once" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .seq, .d }).fromSlice(&ctx, .{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    var d_totals = try x.sumMany(&ctx, .{ .batch, .seq }); // tags {.d}
    defer d_totals.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 22, 26, 30 }, try d_totals.dataConst());
}
```

### 7.9 `taggedDot`: tag-directed contraction and its lowering (`src/tagged.zig`)

```zig
pub fn taggedDot(comptime left_tags: anytype, left: *const RawTensor, ctx: *ExecContext,
                 comptime right_tags: anytype, right: *const RawTensor,
                 comptime contract_tag: Tag) !RawTensor
```

Semantics: contract the `contract_tag` axis of both operands; every shared
non-contracted tag is a **batch axis**; the result tags are
`dotResultTags` = batch ++ left free ++ right free (§7.4). Because the
contraction is named, the operands' physical axis order never changes the
mathematical result — it only selects the kernel.

Validation happens before any compute, via:

```zig
pub fn dotResultShapeOf(comptime left_dtype: DType, comptime right_dtype: DType,
                        comptime left_tags: anytype, left: *const TensorOf(left_dtype),
                        comptime right_tags: anytype, right: *const TensorOf(right_dtype),
                        comptime contract_tag: Tag) ![rawRank(dotResultTags(...).len)]usize
```

which requires the contract dims to be equal and every batch tag's dim to
match on both sides (`TensorError.ShapeMismatch`), and returns the raw-rank
result shape (`{1}` for a scalar result). Dot batching is exact-match — batch
dims do **not** broadcast (unlike pointwise); use the facade `matmul` with
explicit `out_tags` for stride-0 broadcast batching (§4).

Lowering: `taggedDot` is a one-line delegation to `taggedEinsum` with
`dotResultTags(left_tags, right_tags, contract_tag)` as the equation — see
the taggedEinsum subsection below for the full pipeline (role assignment,
runtime orientation selection, batch collapse). The classic dot dispatches
fall out of it: vector·vector runs `ctx.dot`; 2-D layouts pick
`matmul2D`/`matmulTransA`/`matmulTransB` by whichever aligned orientation is
contiguous (the `(0,1)` layout wants both transposes — only one is available
per call, so the smaller operand materializes); canonical batched layouts
hit `bmm`/`bmmTransA`/`bmmTransB` with no data movement; vectors ride along
as size-1 GEMM axes; everything else aligns and materializes at most once
per operand. Kernel outputs are contiguous; a final zero-copy reshape
restores the canonical per-axis result shape.

`taggedDot` itself is the f32 path. The facade `dot` (§4) has the same tag
semantics for every RHS dtype but routes quantized-block, `f16`, and `bf16`
RHS tensors to dedicated kernels (§9, §10), and attaches `DotBackward` /
`ConstRhsDotBackward` for autograd (§5). Its result type is
`Tensor(dotResultTags(tags, other_tags, contract_tag))`.

```zig
test "dot treats shared non-contracted tags as batch axes" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{ .batch, .m, .k }).fromSlice(&ctx, .{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .batch, .k, .n }).fromSlice(&ctx, .{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer b.deinit();

    var c = try a.dot(&ctx, &b, .k); // result tags {.batch, .m, .n}
    defer c.deinit();
    const shape = c.shape();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, &shape);
    try std.testing.expectEqualSlices(f32, &.{ 22, 28, 49, 64, 220, 244, 301, 334 }, try c.dataConst());
}
```

```zig
test "dot contracts by tag regardless of physical axis order" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{ .m, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .n, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 7, 9, 11, 8, 10, 12 });
    defer b.deinit();

    var c = try a.dot(&ctx, &b, .k); // lowers to the trans-B matmul kernel
    defer c.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, try c.dataConst());
}
```

```zig
test "contracting the only tag yields a scalar tensor" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer a.deinit();
    var b = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 4, 5, 6 });
    defer b.deinit();

    var y = try a.dot(&ctx, &b, .d); // Tensor(.{}): no tags, raw shape {1}
    defer y.deinit();
    comptime std.debug.assert(@TypeOf(y).tag_count == 0);
    try std.testing.expectEqual(@as(f32, 32), try y.item());
}
```

#### `taggedEinsum`: multi-index contraction lowering

```zig
pub fn taggedEinsum(comptime left_tags: anytype, left: *const RawTensor, ctx: *ExecContext,
                    comptime right_tags: anytype, right: *const RawTensor,
                    comptime out_tags: anytype) !RawTensor
pub fn einsumResultShapeOf(comptime left_dtype: DType, comptime right_dtype: DType,
                           comptime left_tags: anytype, left: *const TensorOf(left_dtype),
                           comptime right_tags: anytype, right: *const TensorOf(right_dtype),
                           comptime out_tags: anytype) ![rawRank(out_tags.len)]usize
```

`taggedEinsum` is the raw lowering behind the facade `einsum` AND `dot`
(§4.8) — `taggedDot` delegates here with the canonical dot result order as
the equation. The output tag tuple is the whole equation, axis roles come
from `einsumPartTags` (§7.4), and every shared dim is validated by
`einsumResultShapeOf` before compute. The lowering, in order:

1. **Pre-sum** — operand-private dropped tags are reduced away with
   `sumManyTensor`, so every remaining axis is batch/free/contract.
2. **Scalar output** — flatten both operands (right aligned to the left's
   axis order) and run `ctx.dot`.
3. **Role assignment** (comptime) — when `out_tags` nests as
   `[batch][left free][right free]` the operands keep their roles; the
   swapped nesting `[batch][right free][left free]` swaps kernel-left and
   kernel-right (so "double-transposed" layouts run as one plain GEMM); an
   interleaved `out_tags` contracts in canonical order and pays one output
   materialization (`permuteTensorTo` + materialize) at the end.
4. **Orientation selection** (runtime) — both operands are aligned to the
   group-nested order as zero-copy permute views, and each side
   independently picks the plain or transposed kernel layout by probing
   which aligned view is already contiguous (a trans GEMM is free; a
   materialize costs a copy pass). At most one of transA/transB
   can be taken per call — when both operands prefer transposed, the larger
   keeps it and the smaller is materialized. Groups then collapse by
   zero-copy reshape into `[batch…,m,k]·[batch…,k,n]` (or the trans
   permutations) and one `matmul2D`/`matmulTransA`/`matmulTransB` (no
   batch) or `bmm`/`bmmTransA`/`bmmTransB` runs; a side whose no
   orientation is contiguous materializes once.

The batch group collapses into a single bmm batch axis before the kernel
call, so any batch count the operands can represent is lowerable (there is
no rank-(batch+2) cap). The facade attaches `EinsumBackward` (§5.8), whose two
branches are einsums themselves — the gradient of a contraction is a
contraction, so no pointwise fallback exists anywhere on the contraction
backward paths (`DotBackward` and `ConstRhsDotBackward` delegate to the
einsum records).

### 7.10 Shared dtype-generic helpers (`src/tagged.zig`)

```zig
pub fn contiguousForReshapeOf(comptime tensor_dtype: DType, ctx: *ExecContext,
                              value: *const TensorOf(tensor_dtype)) !TensorOf(tensor_dtype)
pub fn productRangeOf(comptime tensor_dtype: DType, value: *const TensorOf(tensor_dtype),
                      comptime start: usize, comptime count: usize) usize
```

`contiguousForReshapeOf` returns a `cloneView` when the value is already
contiguous, otherwise a materialized copy via the ExecContext — an owned
tensor either way, so callers can `defer deinit` unconditionally.
`productRangeOf` multiplies a comptime-bounded run of dims; the dot paths use
it to collapse free/batch axis groups. Both are used by the facade's typed dot
paths (`ag/tensor.zig`) as well as the library itself.

## 8. Data types, storage, and the raw tensor layer (internal)

This section documents the substrate under the public `Tensor` facade: the
dtype system (`src/dtype.zig`), refcounted storage (`src/storage.zig`), and
the raw tensor value (`src/tensor.zig`). **None of this is a stable public
API.** The module root deliberately does not export the raw tensor type — a
comptime guard makes that a compile error (§8.6) — and the sanctioned
in-tree names for it are `fucina.internal.RawTensor` and, for
microbenchmarks only, `bench_raw.RawTensor` (§2). It is documented here
because it is load-bearing for everything else: the dtype policy in §8.3
explains every output dtype in §4, the buffer refcount explains the memory
model in §6 and [MEMORY-MODEL.md](MEMORY-MODEL.md), and anyone extending the
library (new ops, backend kernels, model loaders) works directly against
these types. Expect this surface to change without compatibility notice.

### 8.1 The `DType` enum (`src/dtype.zig`)

```zig
pub const DType = enum {
    bool, u8, u16, i8, i16, i32, i64, f16, bf16, f32, f64,
    q1_0, q4_0, q4_1, q5_0, q5_1, q8_0, q8_1,
    q2_k, q3_k, q4_k, q5_k, q6_k, q8_k,
    iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs,
    tq1_0, tq2_0, mxfp4, nvfp4,
};

pub const DTypeKind = enum { scalar, block_quantized };
```

`DType` is the logical format tag carried by every buffer and tensor type. It
is re-exported at the public root as `fucina.DType`. Every dtype falls into
one of two kinds (`kind(dtype)`, `isScalar`, `isBlockQuantized`):

- **Scalar** dtypes store one storage element per logical tensor element.
- **Block-quantized** dtypes store one packed block struct per `blockSize`
  logical elements, always along the last logical axis.

Scalar dtypes:

| `DType` | `Scalar(dtype)` | Size | Notes |
|---|---|---|---|
| `.bool` | `bool` | 1 B | `zero()` is `false`, `one()` is `true` |
| `.u8` | `u8` | 1 B | |
| `.u16` | `u16` | 2 B | token-id workhorse |
| `.i8` | `i8` | 1 B | |
| `.i16` | `i16` | 2 B | |
| `.i32` | `i32` | 4 B | |
| `.i64` | `i64` | 8 B | |
| `.f16` | `f16` | 2 B | IEEE binary16, native Zig float |
| `.bf16` | `u16` | 2 B | **raw bfloat16 bits**, not a float type; `one(.bf16) == 0x3f80` |
| `.f32` | `f32` | 4 B | the only differentiable public dtype |
| `.f64` | `f64` | 8 B | |

Block-quantized dtypes (GGML-compatible wire formats; the packed `extern
struct` layouts are byte-exact against ggml and pinned by `comptime` size
asserts in `src/dtype.zig`). Encoding/decoding semantics per format are §10;
this table is the storage geometry:

| `DType` | Block struct | Elems/block (`blockSize`) | Bytes/block (`blockByteSize`) |
|---|---|---|---|
| `.q1_0` | `BlockQ1_0` | 128 | 18 |
| `.q4_0` | `BlockQ4_0` | 32 | 18 |
| `.q4_1` | `BlockQ4_1` | 32 | 20 |
| `.q5_0` | `BlockQ5_0` | 32 | 22 |
| `.q5_1` | `BlockQ5_1` | 32 | 24 |
| `.q8_0` | `BlockQ8_0` | 32 | 34 |
| `.q8_1` | `BlockQ8_1` | 32 | 36 |
| `.q2_k` | `BlockQ2_K` | 256 | 84 |
| `.q3_k` | `BlockQ3_K` | 256 | 110 |
| `.q4_k` | `BlockQ4_K` | 256 | 144 |
| `.q5_k` | `BlockQ5_K` | 256 | 176 |
| `.q6_k` | `BlockQ6_K` | 256 | 210 |
| `.q8_k` | `BlockQ8_K` | 256 | 292 |
| `.iq1_s` | `BlockIQ1_S` | 256 | 50 |
| `.iq1_m` | `BlockIQ1_M` | 256 | 56 |
| `.iq2_xxs` | `BlockIQ2_XXS` | 256 | 66 |
| `.iq2_xs` | `BlockIQ2_XS` | 256 | 74 |
| `.iq2_s` | `BlockIQ2_S` | 256 | 82 |
| `.iq3_xxs` | `BlockIQ3_XXS` | 256 | 98 |
| `.iq3_s` | `BlockIQ3_S` | 256 | 110 |
| `.iq4_nl` | `BlockIQ4_NL` | 32 | 18 |
| `.iq4_xs` | `BlockIQ4_XS` | 256 | 136 |
| `.tq1_0` | `BlockTQ1_0` | 256 | 54 |
| `.tq2_0` | `BlockTQ2_0` | 256 | 66 |
| `.mxfp4` | `BlockMXFP4` | 32 | 17 |
| `.nvfp4` | `BlockNVFP4` | 64 (16-elem subblocks) | 36 |

The block structs and the size constants (`q1_0_block_size`,
`q4_0_block_size`, `q4_1_block_size`, `q5_0_block_size`, `q5_1_block_size`,
`q8_0_block_size`, `q8_1_block_size`, `qk_k_block_size` = 256,
`k_scale_size` = 12, `iq4_nl_block_size`, `mxfp4_block_size`,
`nvfp4_block_size`, `nvfp4_subblock_size`, `iq3s_n_scale`) are `pub` in
`src/dtype.zig`; the block structs are also re-exported at the public root
(`fucina.BlockQ4_K`, `fucina.BlockTQ2_0`, ...) because loaders and format
code legitimately handle raw blocks.

A block-quantized tensor of shape `[..., n]` requires `n` to be a nonzero
multiple of `blockSize(dtype)` and stores
`prefix_product * n / blockSize(dtype)` block structs
(`storageElementCount`, §8.5.7). Both `blockSize` and `blockByteSize` are
comptime functions; calling `blockSize` on a scalar dtype is a compile error.

### 8.2 Storage mapping and dtype predicates (`src/dtype.zig`)

```zig
pub fn Scalar(comptime dtype: DType) type       // scalar dtypes only
pub fn Storage(comptime dtype: DType) type      // any dtype
pub fn Accumulator(comptime dtype: DType) type  // scalar dtypes only
```

- `Scalar(dtype)` is the per-logical-element type. It is a compile error for
  block-quantized dtypes ("block-quantized dtypes do not have one scalar
  storage element per logical tensor element"). Note `Scalar(.bf16) == u16`:
  bf16 is stored and passed as raw bits everywhere.
- `Storage(dtype)` is the per-storage-element type: `Scalar(dtype)` for
  scalar dtypes, the block struct for block-quantized dtypes. Buffers and
  raw tensors are sized in `Storage(dtype)` units.
- `Accumulator(dtype)` is the reduction accumulator type: `f32` for
  `.f16`/`.bf16`/`.f32`, `f64` for `.f64`, `u64` for `.bool`/`.u8`/`.u16`,
  `i64` for the signed integers. Compile error for block dtypes.

Classification predicates (all comptime, all `pub`):

| Function | True for |
|---|---|
| `kind(dtype)` | returns `.scalar` or `.block_quantized` |
| `isScalar` / `isBlockQuantized` | kind shorthands |
| `isFloat` | `.f16`, `.bf16`, `.f32`, `.f64` |
| `isInteger` | `.u8`, `.u16`, `.i8`, `.i16`, `.i32`, `.i64` |
| `isSignedInteger` / `isUnsignedInteger` | the obvious subsets |
| `supportsGrad` | `== isFloat` (only float tensors can carry gradients; in practice only `.f32` does, §5) |
| `supportsIntMath` | `== isInteger` (wrapping integer pointwise math and i64-accumulated reductions; `.bool` reduces but has no pointwise math) |
| `supportsForwardFloatMath` | `== isFloat` (forward-only math on the typed facade, §3) |
| `supportsToFloat` | floats plus every block-quantized dtype (dequantizable) |
| `supportsQuantizedMatmulRhs` | every block dtype **except** `.q8_1` and `.q8_k` (those two are activation-side dot-product formats, §10) |
| `supportsQuantizedGetRows` | `== isBlockQuantized` (embedding-row gather) |
| `logicalDType` | blocks map to `.f32`, scalars map to themselves |

Scalar constant/conversion helpers: `zero(dtype)`, `one(dtype)`,
`name(dtype)` (the tag name), `toF32`/`toF64`/`fromF32`/`fromF64` (float
dtypes only; compile error otherwise), and
`castFloat(source_dtype, target_dtype, value)`, which routes through `f64`
when the target is `.f64` and through `f32` otherwise.
`castScalar(source_dtype, target_dtype, value)` is the general scalar cast
across the non-block dtypes: float↔float delegates to `castFloat`, and the
int/bool legs follow the semantics quoted in §3.8's `to` conversion table
(integer↔integer wraps two's-complement, float→integer truncates toward
zero and saturates with NaN → 0, anything→bool is `!= 0`, bool→number is
0/1). `isTruthy(dtype, value)` is mask truthiness: `!= 0`, with bf16 read
through the value bridge so `-0.0` stays falsy and NaN is truthy.
`toAccumulator`/`fromAccumulator` convert between `Scalar(dtype)` and
`Accumulator(dtype)` (bool maps to 0/1). `bf16ToF32(bits: u16) f32` and
`f32ToBf16(value: f32) u16` implement the bf16 bridge: round-to-nearest-even
on narrowing, with ggml-compatible NaN quieting (a NaN payload never
truncates to infinity; `src/dtype_tests.zig` pins this).

### 8.3 Float compute/output dtype policy (`src/dtype.zig`)

```zig
pub const FloatOp = enum { pointwise, reduction, matmul };

pub fn computeDType(comptime op: FloatOp, comptime input_dtype: DType) DType
pub fn outputDType(comptime op: FloatOp, comptime input_dtype: DType) DType
```

Forward float math has one explicit, comptime-resolved policy. `computeDType`
names the arithmetic/accumulation dtype, `outputDType` the result storage
dtype. For block-quantized dtypes both functions return the input unchanged;
integers and `.bool` follow the integer rows in the table:

| Op family | Input | Computes in | Returns |
|---|---|---|---|
| pointwise | `.f16` | `f16` | `.f16` |
| pointwise | `.bf16` | `f32` | `.bf16` |
| pointwise | `.f32` | `f32` | `.f32` |
| pointwise | `.f64` | `f64` | `.f64` |
| pointwise | integers | input dtype (wrapping) | input dtype |
| reduction | `.f16`, `.bf16`, `.f32` | `f32` | **`.f32`** |
| reduction | `.f64` | `f64` | `.f64` |
| reduction | integers, `.bool` | `i64` (wrapping) | **`.i64`** |
| dot/matmul | `.f16`, `.bf16`, `.f32` | `f32` (accumulate) | input dtype |
| dot/matmul | `.f64` | `f64` | `.f64` |

Three rules fall out of the table:

- **bf16 computes through f32 always** — it is stored as raw `u16` bits, so
  even pointwise ops widen via `bf16ToF32`, compute in `f32`, and narrow
  back with round-to-nearest-even on store (except reductions, which return
  `f32` outright).
- **Reductions on 16-bit floats return f32.** Summing `f16`/`bf16` into a
  16-bit result would lose the accumulator's precision, so the widened
  result dtype is kept.
- **Explicit casts are required for anything else.** No op silently promotes
  across operand dtypes: mixed-dtype pointwise math on the typed facade is a
  compile error (`"typed pointwise requires matching dtypes; cast
  explicitly"` in `src/ag/tensor.zig`). Casting is an explicit op — `to(ctx,
  target_dtype)` on the public facade (§3), `castTyped` on `ExecContext`
  (§6).

The policy is visible directly in public result types:

```zig
test "float dtype policy: f16 reduction returns f32" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const Half = fucina.Tensor(.{ .dtype = .f16, .tags = .{ .row, .col } });
    var x = try Half.fromSlice(&ctx, .{ 2, 2 }, &.{ 1.5, 2.5, 3.0, 4.0 });
    defer x.deinit();

    // Pointwise keeps the input dtype: f16 + f16 -> f16.
    var y = try x.add(&ctx, &x);
    defer y.deinit();
    comptime std.debug.assert(@TypeOf(y).dtype == .f16);

    // Reductions on f16 accumulate in f32 and *return* f32.
    var s = try x.sum(&ctx, .col);
    defer s.deinit();
    comptime std.debug.assert(@TypeOf(s).dtype == .f32);
    try std.testing.expectEqualSlices(f32, &.{ 4.0, 7.0 }, try s.dataConst());

    // Any other output dtype requires an explicit cast.
    var wide = try x.to(&ctx, .f32);
    defer wide.deinit();
    comptime std.debug.assert(@TypeOf(wide).dtype == .f32);
}
```

### 8.4 Refcounted storage: `BufferOf(dtype)` (`src/storage.zig`)

```zig
pub fn BufferOf(comptime buffer_dtype: DType) type {
    return struct {
        allocator: Allocator,
        data: []Elem,                     // Elem == dtype.Storage(buffer_dtype)
        refs: std.atomic.Value(u32),
        release_ctx: ?*anyopaque = null,
        release_fn: ?*const fn (*anyopaque, *Self) void = null,
        pending_work: std.atomic.Value(?*accelerator.Work) = .init(null),
        pending_use: std.atomic.Value(?*accelerator.Work) = .init(null),
        accelerator_resource: std.atomic.Value(?*accelerator.Resource) = .init(null),

        pub const dtype = buffer_dtype;
        pub const Element = Elem;
        ...
    };
}
pub const Buffer = BufferOf(.f32);
```

A buffer is a heap-allocated header (`allocator.create(Self)`) plus a typed
data slice, shared by pointer and lifetime-managed by an atomic refcount.
Every raw tensor holds exactly one reference to exactly one buffer; views
share the buffer by taking additional references.

Constructors (all return `!*Self` with `refs == 1`):

| Constructor | Data ownership | At `refs == 0` |
|---|---|---|
| `create(allocator, len)` | owned, uninitialized `len` elements | `destroy()`: free data + header |
| `createWithRelease(allocator, len, release_ctx, release_fn)` | owned | `release_fn(release_ctx, self)` |
| `fromSlice(allocator, values)` | owned copy of `values` | `destroy()` |
| `fromBorrowedSlice(allocator, values)` | **aliases** caller memory | destroy header only; caller keeps the bytes |
| `fromBorrowedSliceWithRelease(allocator, values, release_fn)` | aliases | `release_fn(self, self)` — full cleanup duty |
| `fromBorrowedSliceWithReleaseCtx(allocator, values, release_ctx, release_fn)` | aliases | `release_fn(release_ctx, self)` — full cleanup duty |

Refcount operations:

- `retain()` — `fetchAdd(1, .monotonic)`. Safe from any thread.
- `release()` — `fetchSub(1, .acq_rel)`; debug-asserts against
  over-release. When the count reaches zero it invokes `release_fn` if set,
  otherwise `destroy()`. Exactly one caller observes zero, so the hook fires
  **exactly once** (`src/storage_tests.zig` pins this).
- `isUnique()` — acquire-load snapshot `refs == 1`. **Snapshot only**: it is
  meaningful only when the caller already has exclusive access to the tensor
  handle pointing at this buffer (the basis of `canTakeInPlace`, §8.5.5).
- `resetRefs()` — stores 1; valid only under exclusive ownership. Used by the
  `ExecContext` buffer pool when recycling a cached buffer (§6).
- `destroy()` — unconditionally frees data + header, bypassing the refcount.
  Only for owners that know no references remain (pool teardown).
- `waitReady()` / `discardPending()` — complete already-submitted GPU output
  work, respectively making host bytes visible or skipping an unused D2H.
- `setPendingUse()` / `waitUnused()` / `waitMutable()` — track the latest
  submitted GPU reader of this allocation. Const host reads may overlap a
  device read; mutable access waits so post-call input mutation cannot race
  Metal zero-copy reads or CUDA async upload. Provider queue order lets the
  latest token subsume earlier readers. Final release always completes both
  output and reader work before storage can be recycled.
- `acceleratorResource` — provider mapping metadata tied to this allocation's
  lifetime (Metal's pooled page-wrapper cache; CUDA host page registration).
  It survives ordinary pool release/reacquire and is destroyed with the
  backing allocation.

**Release-hook contract.** For the borrowed-with-release variants the hook
runs once, at the final `release()`, and takes *full* cleanup responsibility:
it must dispose of the external data by whatever means created it **and**
free the header with `buffer.destroyHeader()` (which also releases any
accelerator resource). Capture the external slice, call `destroyHeader()`
first, then free/unmap the bytes: provider teardown may still need the live
address to unregister it. The two in-tree
production uses:

- **GPU device-resident bytes** — `src/llm/weights.zig` wraps managed device
  allocations from `internal.gpu.allocResidentBytes` so the final buffer
  release (counting `cloneView`'d weights that share it) frees them via
  `internal.gpu.freeResidentBytes` (§8.6, §9).
- **pooled slabs** — `src/exec/buffer_pool.zig` uses the `Ctx` variant so a
  typed buffer's release returns its byte slab to the pool free list instead
  of freeing it (§6).

Note the tree does **not** use release hooks for mmap'd GGUF bytes: that
lifetime is holder-managed — `gguf.File.deinit` munmaps, or ownership moves
via `takeMapping` to a `MappedRegion` the holder must keep alive while
anything borrows tensor data from it (§12). The hook mechanism remains the
right tool when *user* code wants refcount-driven cleanup of an external
mapping, as below:

```zig
fn wrapMappedWeights(alloc: std.mem.Allocator, mapped: []f32) !fucina.internal.RawTensor {
    const RawTensor = fucina.internal.RawTensor;
    const Buffer = std.meta.Child(@FieldType(RawTensor, "buffer")); // BufferOf(.f32)

    const hook = struct {
        fn releaseMapped(_: *anyopaque, buffer: *Buffer) void {
            // Full cleanup responsibility: the external bytes AND the header.
            const bytes = std.mem.sliceAsBytes(buffer.data);
            buffer.destroyHeader();
            std.posix.munmap(@alignCast(bytes));
        }
    };

    const buffer = try Buffer.fromBorrowedSliceWithRelease(alloc, mapped, hook.releaseMapped);
    return RawTensor.fromOwnedBuffer(buffer, &.{ 4, 8 }) catch |err| {
        buffer.release(); // still owns the one reference on failure
        return err;
    };
    // Later: the final tensor deinit drops refs to 0 and fires the hook once.
}
```

The buffer type is not separately exported; internal code names it through
the tensor's field type, as above (the `src/llm/weights.zig` idiom).

**Thread-safety.** `retain`/`release` are atomic and may race freely; the
data slice is not synchronized — concurrent reads are fine, and writers need
external coordination (the runtime's parallel kernels partition disjoint
ranges, §9). `isUnique` and `resetRefs` are only correct under exclusive
access as described above.

### 8.5 The raw tensor: `TensorOf(dtype)` (`src/tensor.zig`)

```zig
pub const max_rank = 8;

pub const TensorError = error{
    ShapeMismatch, InvalidShape, InvalidDataLength, IndexOutOfBounds, UnsupportedView,
    EmptySelection, DivisionByZero,
};

pub const Shape = struct {
    len: u8,
    dims: [max_rank]usize,
    pub fn init(values: []const usize) !Shape        // rejects rank 0/>8 and zero dims
    pub fn initStrides(values: []const usize) !Shape // zeros allowed (broadcast strides)
    pub fn slice(self: *const Shape) []const usize
    pub fn at(self: *const Shape, i: usize) usize
};

pub fn TensorOf(comptime tensor_dtype: DType) type {
    return struct {
        buffer: *BufferOf(tensor_dtype),
        shape: Shape,
        strides: Shape,
        offset: usize = 0,

        pub const dtype = tensor_dtype;   // and: pub const Element = Storage(dtype)
        ...
    };
}
pub const Tensor = TensorOf(.f32);        // == fucina.internal.RawTensor
```

A raw tensor is a plain value: a buffer pointer plus **inline** shape/stride
metadata (two fixed `[8]usize` arrays — no allocation per view) and a start
`offset`, all measured in *storage elements* (`Storage(dtype)` units — for
block-quantized dtypes, strides count blocks). Rank is runtime (1 to
`max_rank` = 8; there is no rank-0 shape, which is why the facade's
scalar-tag tensor is a rank-1 `{1}` raw tensor). Copying the struct does
**not** retain the buffer; every legitimately owned tensor value carries
exactly one buffer reference, and `deinit()` releases it and poisons the
struct (`self.* = undefined` — not idempotent).

The raw tensor appears inside public signatures (`ctx.fromSlice` returns
one; `Tensor(spec).variable(&ctx, raw)` consumes one), but the type itself is
only nameable as `fucina.internal.RawTensor` /
`fucina.internal.tensor_mod.TensorOf(dtype)`. For convenience `tensor.zig`
re-exports `DType`, `Scalar`, and `Storage` from `dtype.zig` (and
`storage.zig` re-exports `DType`), so `tensor_mod` alone is enough for most
raw-layer work.

#### 8.5.1 Construction and ownership

| Constructor | Dtypes | Semantics |
|---|---|---|
| `zeros(allocator, shape)` / `ones(allocator, shape)` | scalar only (compile error otherwise) | fresh owned buffer, filled |
| `fromSlice(allocator, shape, values: []const Scalar)` | scalar only | owned copy; `InvalidDataLength` unless `values.len == elementCount(shape)` |
| `fromBorrowedSlice(allocator, shape, values: []Scalar)` | scalar only | aliases caller memory (borrowed buffer; caller keeps ownership of the bytes and must outlive the tensor) |
| `fromStorageSlice(allocator, shape, values: []const Element)` | any | owned copy in storage elements; for block dtypes `values.len` must equal `storageElementCount` |
| `fromBorrowedStorageSlice(allocator, shape, values: []Element)` | any | borrowed, in storage elements |
| `fromOwnedBuffer(buffer, shape)` | any | **consumes one reference** to `buffer`; the caller must not release that reference after success — `deinit` does. Accepts oversized buffers (`data.len >= storageElementCount`), which is how pooled buffers are wrapped; `InvalidDataLength` if too small. On error the reference stays with the caller |
| `scalar(allocator, value)` | scalar only | shape `{1}` |
| `clone(allocator)` | any | materializing deep copy into a fresh contiguous buffer (see 8.5.4) |

All errors are `TensorError` members, plus `error.Overflow` from checked
element-count multiplication and `error.OutOfMemory` from every allocating
constructor (only `fromOwnedBuffer` is allocation-free). The allocator
passed at construction is stored in the buffer and used for its teardown.

#### 8.5.2 Geometry queries

`rank()`, `len()` (logical element count), `storageLen()` (storage element
count; `len()/blockSize` per trailing axis for block dtypes), `rows()` /
`cols()` (rank-2 only, else `InvalidShape`), `isScalar()` (`len() == 1`),
and `isContiguous()` — true when strides are exactly the row-major strides
of the shape (a broadcast `{1}` scalar with stride 0 is *not* contiguous).

#### 8.5.3 Views

All view constructors `retain()` the buffer and return a new tensor value
that must be `deinit`ed independently; shape/stride metadata is copied
inline, and `offset` is preserved (or extended). Writes through any view are
visible through every alias of the same buffer.

```zig
pub fn cloneView(self: *const Self) !Self
pub fn reshape(self: *const Self, new_shape: []const usize) !Self
pub fn viewWithStrides(self: *const Self, shape: []const usize, strides: []const usize) !Self
pub fn viewWithStridesOffset(self: *const Self, shape: []const usize, strides: []const usize, offset_delta: usize) !Self
pub fn broadcastTo(self: *const Self, target_shape: []const usize) !Self
pub fn broadcastToRank(self: *const Self, comptime target_rank: usize, target_shape: [target_rank]usize) !Self
```

- `cloneView` — identical view, one more reference. This is how weight
  tensors are shared across module structs.
- `reshape` — **requires contiguity** (`UnsupportedView` otherwise) and a
  matching element count (`InvalidShape`); the result is a retained view
  over the same storage, never a copy. Non-contiguous tensors must be
  materialized first (`clone`, or `ExecContext.materialize*` / the tagged
  layer's `contiguousForReshapeOf`, §6/§7).
- `viewWithStrides` / `viewWithStridesOffset` — arbitrary strided
  (sub)views, **checked**: `strides.len` must match `shape.len`
  (`InvalidShape`) and the maximal reachable index
  `offset + offset_delta + Σ (dim-1)·stride` must lie inside the buffer
  (`InvalidDataLength`). `offset_delta` advances the view's start; this is
  the raw narrowing primitive (there is no dedicated `narrow` on the raw
  type — `ExecContext.narrowAxisRank`/`narrowAxisRankTyped` in §6 and the
  facade's `narrow` in §4 are built on it).
- `broadcastTo` / `broadcastToRank` — zero-stride broadcast views,
  right-aligned like NumPy: the source rank must not exceed the target rank
  (`ShapeMismatch`), new leading axes get stride 0, matching axes keep
  their stride, size-1 axes get stride 0, and any other mismatch is
  `ShapeMismatch`. `broadcastTo` dispatches over runtime target rank 1–8
  (`InvalidShape` beyond); `broadcastToRank` takes the rank at comptime.

**Block-quantized restriction:** for block dtypes, `reshape` and the
`viewWithStrides*` family accept only the identity view (same shape, same
strides, `offset_delta == 0`) and otherwise return `UnsupportedView`; blocks
are indivisible, so only whole-tensor aliasing is a view. Broadcasting
follows the generic path but is only meaningful on non-trailing axes.

```zig
fn rawViewTour(alloc: std.mem.Allocator, blocks: []const fucina.BlockQ8_0) !void {
    const RawTensor = fucina.internal.RawTensor; // TensorOf(.f32)

    var x = try RawTensor.fromSlice(alloc, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    // Transposed view: same buffer, swapped strides, not contiguous.
    var t = try x.viewWithStrides(&.{ 3, 2 }, &.{ 1, 3 });
    defer t.deinit();
    std.debug.assert(t.buffer == x.buffer);
    std.debug.assert(!t.isContiguous());

    // t.data() would panic here; the checked accessor reports an error.
    try std.testing.expectError(error.UnsupportedView, t.dataChecked());

    // clone materializes any view into fresh contiguous storage.
    var m = try t.clone(alloc);
    defer m.deinit();
    std.debug.assert(m.isContiguous() and m.canTakeInPlace());

    // Zero-stride broadcast view; reshape of a contiguous tensor is a view.
    var b = try x.broadcastTo(&.{ 4, 2, 3 });
    defer b.deinit();
    var flat = try x.reshape(&.{6});
    defer flat.deinit();

    // Non-f32 raw tensors: fucina.internal.tensor_mod.TensorOf(dtype).
    var ids = try fucina.internal.tensor_mod.TensorOf(.u16)
        .fromSlice(alloc, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ids.deinit();

    // Block-quantized raw tensor: last axis in logical elements, storage in
    // blocks (here 4*64/32 == 8 BlockQ8_0 storage elements).
    var w = try fucina.internal.tensor_mod.TensorOf(.q8_0)
        .fromStorageSlice(alloc, &.{ 4, 64 }, blocks);
    defer w.deinit();
}
```

#### 8.5.4 Data access and materialization

```zig
pub fn data(self: *Self) []Elem                    // PANICS on non-contiguous
pub fn dataConst(self: *const Self) []const Elem   // PANICS on non-contiguous
pub fn dataChecked(self: *Self) ![]Elem            // error.UnsupportedView instead
pub fn dataConstChecked(self: *const Self) ![]const Elem
pub fn item(self: *const Self) Elem                // scalar dtypes; asserts len() == 1
pub fn copyTo(self: *const Self, dst: []Elem) !void
pub fn clone(self: *const Self, allocator: Allocator) !Self
```

`data`/`dataConst` return `buffer.data[offset .. offset + storageLen()]` and
**panic** on non-contiguous tensors (`"Tensor.data requires a contiguous
tensor; materialize or use dataChecked"`) — they are for hot paths that have
already established contiguity. `dataChecked`/`dataConstChecked` are the
recoverable variants (`UnsupportedView`). `item()` debug-asserts a
single-element tensor and reads through `dataConst` (so it also requires
contiguity — a zero-stride broadcast scalar panics).

`copyTo(dst)` writes the logical contents into a caller slice of exactly
`storageLen()` elements (`InvalidDataLength` otherwise): a straight `memcpy`
when contiguous, an odometer copy for non-contiguous scalar tensors (the
maximal row-major-contiguous axis suffix moves as whole `memcpy` runs, a
strided innermost axis as a stride-increment loop — never a per-element
division; dim-1 axes are absorbed, so a spuriously non-contiguous singleton
view still copies as one `memcpy`), and `UnsupportedView` for non-contiguous
block tensors. `copyRangeTo(dst, linear_start, count)` is the range form
(scalar dtypes): disjoint ranges of the row-major linearization may be
copied concurrently, which is how the exec runtime parallelizes large
materializations (§6). `clone(allocator)` is the materialization path: it
allocates a fresh contiguous buffer and `copyTo`s into it — the result is
always contiguous with `offset == 0`, regardless of the source view.

#### 8.5.5 In-place helpers and `canTakeInPlace`

Scalar-dtype-only mutators (compile error for block dtypes):
`addInPlace(other)` (`ShapeMismatch` unless shapes match; both operands go
through `data()`/`dataConst()` and hence panic when non-contiguous),
`scaleInPlace(scalar_value)`, and `fill(value)`.

```zig
// Safe only when the caller owns exclusive access to this Tensor value.
// The refcount proves no other retained Tensor aliases the buffer now; it
// is not a lock against another thread retaining the same Tensor later.
pub fn canTakeInPlace(self: *const Self) bool  // offset == 0 and isContiguous() and buffer.isUnique()
```

`canTakeInPlace` is the **ownership optimization** used by consuming ops to
steal an operand's buffer and write the result in place instead of
allocating. The exact contract: it returns true only for a full-buffer
(`offset == 0`), contiguous, uniquely-referenced tensor, and the answer is
trustworthy **only while the caller has exclusive access to the tensor
handle** — the refcount check proves no *other* view aliases the buffer at
that instant; it is *not* synchronization, and another thread that could
still retain/read the same handle invalidates the optimization by contract,
not by any runtime check.

#### 8.5.6 Fixed-rank views

```zig
pub fn RankedTensorOf(comptime tensor_dtype: DType, comptime rank: usize) type {
    return struct {
        tensor: *const TensorOf(tensor_dtype),
        shape: [rank]usize,
        strides: [rank]usize,
        pub fn dim(self: @This(), comptime axis: usize) usize
        pub fn len(self: @This()) usize
        pub fn isContiguous(self: @This()) bool
    };
}
pub fn RankedTensor(comptime rank: usize) type  // f32 alias

pub fn rankView(self: *const Self, comptime rank_value: usize) !RankedTensorOf(dtype, rank_value)
```

`rankView` copies the runtime shape/strides into comptime-sized arrays
(`InvalidShape` when the tensor's rank differs). The result **borrows** the
tensor — no retain; it must not outlive it. This is the bridge from
runtime-rank tensors to rank-specialized kernels: with `[rank]usize` in
hand, loops unroll at comptime (`inline while`/`inline for`), which is how
the exec layer's `*Rank` entry points (§6) and backend kernels (§9) are
written.

#### 8.5.7 Shape arithmetic (free functions)

`pub` helpers in `src/tensor.zig`, shared by the exec and tagged layers:

- `requireSameShape(a, b)` / `requireSameShapeOf(dtype, a, b)` —
  `ShapeMismatch` unless shapes are equal.
- `elementCount(shape)` / `elementCountArray(rank, shape)` — logical element
  count; `InvalidShape` for rank 0/>8 or zero dims; overflow-checked.
- `storageElementCount(dtype, shape)` / `storageElementCountArray(...)` —
  storage element count; for block dtypes the last axis must be a nonzero
  multiple of `blockSize(dtype)` (`InvalidShape` otherwise).
- `elementCountArrayAssumeValid(rank, shape)` — unchecked product for
  already-validated shapes.

**Thread-safety.** Raw tensors have no interior locking. The only atomic
state is the buffer refcount; concurrent readers of one buffer are safe,
concurrent writers (or a writer racing readers) need external coordination.
The runtime never mutates shared storage concurrently except by partitioning
disjoint ranges across the worker team (§9).

### 8.6 The `fucina.internal` escape hatch (`src/fucina.zig`)

The public root deliberately does **not** export the raw tensor type, and an
anti-regression guard makes reintroducing it a compile error on any build
that analyzes the module root (every test, example, and tool — not just
`zig build test`):

```zig
comptime {
    if (@hasDecl(@This(), "RawTensor")) @compileError(
        "fucina.RawTensor must not be exported at the public root; raw tensors are internal. " ++
            "Use fucina.internal.RawTensor (in-tree raw naming) or bench_raw.RawTensor (microbench).",
    );
}
```

The rationale is API shape, not capability: the no-grad `Tensor` facade has
negligible forward overhead, so model and example code carries
`fucina.Tensor(spec)` end-to-end, and a public raw type would split the
ecosystem into two tensor vocabularies. Code that genuinely needs the raw
layer names it through `fucina.internal`; raw-kernel microbenchmarks use the
separate `bench_raw` module (`src/bench_raw.zig`), which the guard does not
affect (it inspects only the root's own decls).

```zig
pub const internal = struct {
    pub const backend_mod = backend;      // src/backend.zig
    pub const tensor_mod = tensor;        // src/tensor.zig
    pub const thread_mod = thread;        // src/thread.zig
    pub const gpu = struct { ... };       // GPU hooks, see below
    pub const RawTensor = tensor.Tensor;  // TensorOf(.f32)
};
```

- `backend_mod`, `tensor_mod`, `thread_mod` — the internal surface for
  sibling modules (notably `fucina_llm`, §13) that need **exact core type
  identity** without importing a second copy of the backend/exec files: a
  `TensorOf(.q4_k)` from a re-imported `tensor.zig` would be a distinct,
  incompatible type. `tensor_mod` gives typed raw tensors
  (`TensorOf(dtype)`, `RankedTensorOf`, the shape helpers); `thread_mod` the
  thread-pool primitives (`Pool`, `WaitGroup`, `Mutex`, ..., §9);
  `backend_mod` the kernel entry points and packed-RHS types (§9).
- `RawTensor` — the canonical internal name for the raw no-grad f32 tensor.
  Intended users: runtime/backend internals, raw-kernel benchmarks,
  serialization/format byte work, and tests targeting raw runtime behavior.
- `gpu` — hooks for model loaders and benchmark instrumentation,
  deliberately kept off the public root: users keep ordinary eager `Tensor`
  values; residency and tracing are backend-owned details (§9).

| Hook | Type / signature | Purpose |
|---|---|---|
| `enabled` | `bool` (comptime) | true on GPU builds (`-Dgpu=metal` or `-Dgpu=cuda`, §2): GPU GEMM offload is compiled in |
| `has_quant_gemm` | `bool` (comptime) | provider implements dequant-in-kernel quantized GEMM (dense + grouped MoE). Loaders that reshape CPU representations for the GPU quant path key on this, **not** on `enabled` — a provider can be enabled while its quantized arms are still CPU-only |
| `allocResidentBytes` | `fn (len: usize) ?[]u8` | device-owned bytes for GPU-build loaders; `null` when unavailable (no device context, `len == 0`, or too large) |
| `freeResidentBytes` | `fn (bytes: []const u8) void` | release bytes returned by `allocResidentBytes`; safe no-op when the device context is gone or the slice is foreign |
| `traceEnabled` | `fn () bool` | opt-in dispatch tracing, enabled by `FUCINA_GPU_TRACE=1` |
| `traceReset` | `fn () void` | reset trace counters (call before a warm measurement window); no-op when tracing is off |
| `traceDump` | `fn () void` | print the accumulated dispatch/time breakdown to stderr; no-op when tracing is off |

The hooks resolve at comptime through `src/backend/gpu.zig` to the active
provider (`src/backend/metal.zig` or `src/backend/cuda.zig`; `-Dgpu=none`
resolves to `metal.zig` with `enabled == false`). Call sites gate
shim-touching calls on `comptime gpu.enabled` so CPU-only builds
comptime-elide every provider reference; `traceReset`/`traceDump` may be
called unconditionally on GPU builds since they no-op when tracing is off.

```zig
fn residentScratch(len: usize) !void {
    const gpu = fucina.internal.gpu;
    if (comptime gpu.enabled) {
        // Device-owned bytes for GPU-build loaders; null when unavailable.
        const bytes = gpu.allocResidentBytes(len) orelse return error.OutOfMemory;
        defer gpu.freeResidentBytes(bytes);

        // Dispatch tracing (FUCINA_GPU_TRACE=1); reset/dump no-op when off.
        gpu.traceReset();
        if (gpu.traceEnabled()) gpu.traceDump();
    }
}
```

For inspection (as opposed to construction), the public facade already
crosses the boundary: every public tensor exposes `asRawTensor()`, returning
`*const` raw tensor whose metadata can be read without owning anything:

```zig
test "asRawTensor exposes raw shape/stride metadata" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .rows, .cols }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    const raw = x.asRawTensor(); // *const fucina.internal.RawTensor
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, raw.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 3, 1 }, raw.strides.slice());
    try std.testing.expectEqual(@as(usize, 0), raw.offset);
    try std.testing.expect(raw.isContiguous());

    // Scalar-tag tensors are rank-1 shape {1} at the raw layer (no rank 0).
    var total = try x.sumAll(&ctx);
    defer total.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, total.asRawTensor().shape.slice());
}
```

## 9. Backends: CPU SIMD, BLAS, threading, and GPU offload

A backend is the layer that owns numeric kernels and nothing else. `ExecContext`
(§6) validates shapes, allocates outputs, and dispatches; the backend fills
caller-supplied buffers. There are exactly two backends — a scalar reference
and the production `native` backend — selected at build time, plus two optional
accelerator tiers layered *inside* the native backend: platform BLAS for large
f32 GEMM and a GPU GEMM provider (Metal or CUDA). Nothing about backend choice
is visible in tensor types or the public API surface beyond a handful of
comptime constants.

### 9.1 Build-time selection and the facade constants (`src/backend.zig`, `build.zig`)

```zig
pub const Kind = enum { scalar, native };

pub const active_kind: Kind = switch (build_options.backend_kind) {
    .scalar, .cpu => .scalar,
    .native => .native,
};

const active = switch (build_options.backend_kind) {
    .scalar, .cpu => scalar_impl,   // src/backend/cpu.zig
    .native => native_impl,         // src/backend/native.zig
};
```

Selection is a comptime `switch` over `build_options.backend_kind`
(`-Dbackend=native|scalar|cpu`; `cpu` is a deprecated alias for `scalar`).
The inactive implementation is never dispatched to — every `Backend` method
forwards to `active.<fn>` and the other module's code paths are dead — and the
switches are exhaustive, so adding a backend variant forces edits at every
dispatch site rather than silently falling through. The same pattern selects
the GPU provider (`src/backend/gpu.zig`): a comptime switch over
`build_options.gpu_kind` resolves `gpu_impl` to `metal.zig` or `cuda.zig`, and
the unselected provider is parsed but never semantically analyzed, so it costs
nothing and needs none of its target's libraries (`cuda.zig` is fully inert on
macOS builds and vice versa). A `comptime` guard in `gpu.zig` verifies
`build_options.use_gpu == (gpu_kind != .none)`.

Build options that shape the backend (see §2 for the full option list):

| Option | Values | Default | Effect |
|---|---|---|---|
| `-Dbackend` | `native`, `scalar`, `cpu` | `native` | backend implementation; `cpu` = deprecated alias for `scalar` |
| `-Dblas` | `none`, `accelerate`, `openblas`, `mkl`, `blis`, `nvpl`, `blas` | `accelerate` on macOS, else `none` | CBLAS provider for large f32 GEMM; `none` selects the pure-Zig blocked packed GEMM |
| `-Daccelerate` | bool | — | compatibility alias: `false` ≡ `-Dblas=none`, `true` selects Accelerate on macOS |
| `-Dblas-threads` | `u32` | 0 | pin the provider's thread count once at first GEMM; 0 keeps the provider default |
| `-Dmax-threads` | 1–64 | 8 | comptime worker-team ceiling *and* runtime default team size (`parallel.vector_max_threads`) |
| `-Dgpu` | `none`, `metal`, `cuda` | `none` | GPU GEMM offload provider; `metal` requires macOS, `cuda` targets Linux (build panics otherwise) |

The facade (`src/fucina.zig`) re-exports the backend's build facts as comptime
constants:

```zig
pub const Backend = backend.Backend;                    // the kernel dispatch struct
pub const BackendKind = backend.Kind;                   // enum { scalar, native }
pub const active_backend_kind = backend.active_kind;
pub const native_blas_kind = backend.native_blas_kind;  // the -Dblas enum value
pub const native_uses_blas = backend.native_uses_blas;  // blas_kind != .none
pub const native_uses_accelerate = backend.native_uses_accelerate;
pub const native_blas_threads = backend.native_blas_threads;
pub const supports_q4_k_mmla = backend.supports_q4_k_mmla;
```

`supports_q4_k_mmla` is `true` on aarch64 targets whose feature set includes
`i8mm` (`src/backend/quant/common.zig` `has_aarch64_i8mm`); it decides which
packed Q4_K RHS layout `packRhs` produces (§9.7, §10). All of these are
comptime values — branches on them fold away:

```zig
test "backend build facts" {
    // All comptime-known — fixed by -Dbackend / -Dblas at build time.
    switch (fucina.active_backend_kind) {
        .native => {}, // default: @Vector kernels + optional BLAS/GPU
        .scalar => {}, // reference backend (-Dbackend=scalar)
    }
    if (fucina.native_uses_blas) {
        try std.testing.expect(fucina.native_blas_kind != .none);
    } else {
        try std.testing.expect(fucina.native_blas_kind == .none);
    }
    if (fucina.native_uses_accelerate)
        try std.testing.expect(fucina.native_uses_blas);
    // -Dblas-threads pin (0 = provider default) and the aarch64+i8mm
    // Q4_K smmla capability, both comptime constants.
    const blas_threads: u32 = fucina.native_blas_threads;
    const q4k_mmla: bool = fucina.supports_q4_k_mmla;
    _ = blas_threads;
    _ = q4k_mmla;
}
```

### 9.2 The `Backend` struct and the kernel contract (`src/backend.zig`)

```zig
pub const Backend = struct {
    pub const kind = active_kind;
    parallel_pool: std.atomic.Value(?*thread.Pool) = .init(null),

    pub fn init() Backend { return .{}; }
    pub fn setWorkPool(self: *Backend, pool: ?*thread.Pool) void;
    // ~90 kernel entry points, all forwarding to the active implementation
};
```

`Backend` is a thin dispatch value embedded in the exec `Runtime` (one per
`ExecContext`). Its only state is `parallel_pool`, an atomic pointer to the
worker team. The atomicity is load-bearing: kernels may dispatch from other
threads (e.g. dot-backward's `OneShotWorker`) while a lazy `tryWorkPool` retry
publishes the pool; the store is `release` and the load `acquire` so a racing
first observer also sees `Pool.init`'s writes. Every `...WithConfig` kernel
receives the pool via a `ParallelConfig{ .pool = ... }` snapshot taken per
call.

Method naming encodes the checking tier — with one caveat:

- `...Into(out, ...) !void` — validates shapes itself (`rankView`, dim
  checks) and returns `TensorError.ShapeMismatch` on disagreement. This
  holds for the elementwise/reduction/dot/matmul families (`addInto`,
  `sumInto`, `dotInto`, `matmulInto`, …). The conv/pool/norm families
  (`conv2dInto`, `im2colInto`, `pool2dInto`, `upsample2xNearestInto`,
  `conv1dInto`, `col2im1dInto`, `snakeInto`, `groupNormInto`,
  `causalDepthwiseConv1dInto`, …) are `...Into`-named but plain `void` and
  **unchecked** — the exec layer validates geometry before calling them.
- `...IntoUnchecked` / `...SliceUnchecked` — `void`; the caller (exec) has
  already validated shape and contiguity. Passing wrong geometry is illegal
  (out-of-bounds slice panics in safe builds, UB in ReleaseFast).
- `...Typed` variants take a comptime `DType` and typed tensors
  (`TensorOf(dtype)`), with output dtype derived by
  `dtype_mod.outputDType(...)`.

The full method inventory, grouped (every name is a `Backend` method; the
`WithConfig` suffix appears only on the implementation-module twins):

| Family | Methods |
|---|---|
| elementwise | `addInto`, `addContiguousIntoUnchecked`, `subInto`, `subContiguousIntoUnchecked`, `mulInto`, `mulContiguousIntoUnchecked`, `maximumContiguousIntoUnchecked`, `minimumContiguousIntoUnchecked`, `elementwiseContiguousIntoTyped`, `scaleInto`, `unaryContiguousIntoUnchecked`, `leakyReluContiguousIntoUnchecked`, `clampContiguousIntoUnchecked`, `gatedContiguousIntoUnchecked` |
| row/slice helpers | `addScaledSliceUnchecked`, `addRowVectorSliceUnchecked`, `addRowVectorUnarySliceUnchecked`, `unaryRowSliceUnchecked`, `mulRowSliceUnchecked`, `preluChannelsIntoUnchecked`, `preluChannelsBackwardInputIntoUnchecked`, `preluChannelsBackwardAlphaIntoUnchecked`, `channelAffineIntoUnchecked` |
| reductions | `sumInto`, `sumSlice`, `prodInto`, `prodSlice`, `sumSliceTyped`, `dotInto`, `dotIntoTyped` |
| 1-D conv | `causalDepthwiseConv1dInto` (+`BackwardInputInto`, `BackwardKernelInto`), `causalConv1dInto` (+`BackwardInputInto`, `BackwardWeightInto`), `groupedCausalConv1dInto` (+`BackwardInputInto`, `BackwardWeightInto`), `conv1dInto` (+`BackwardInputInto`, `BackwardWeightInto`), `col2im1dInto`, `col2im1dBackwardInto` |
| 2-D conv / image | `conv2dInto`, `conv2dBackwardInputInto`, `conv2dBackwardWeightInto`, `im2colInto`, `col2imInto`, `pool2dInto`, `avgPool2dBackwardInto`, `maxPool2dBackwardInto`, `upsample2xNearestInto` |
| Winograd transforms | `winogradF2WeightTransformInto`, `winogradF2InputTransformInto`, `winogradF2OutputTransformInto`, `winogradF4WeightTransformInto`, `winogradF4InputTransformInto`, `winogradF4OutputTransformInto` |
| norm / activation kernels | `groupNormInto`, `groupNormBackwardInto`, `snakeInto`, `snakeBackwardInputInto`, `snakeBackwardParamsInto` |
| dense GEMM | `matmulInto`, `matmul2DIntoUnchecked`, `matmul2DIntoUncheckedTyped`, `matmulTransAInto`, `matmulTransA2DIntoUnchecked`, `matmulTransBInto`, `matmulTransB2DIntoUnchecked`, `matmulTransB2DIntoUncheckedF16Operands`, `matmulTransB2DIntoUncheckedBf16Rhs` |
| batched GEMM | `matmulBatched2DIntoUnchecked`, `matmulBatchedTransA2DIntoUnchecked`, `matmulBatchedTransB2DIntoUnchecked` |
| packed dense RHS | `packMatmulRhsTyped`, `matmul2DIntoUncheckedPackedRhsTyped` |
| quantized RHS | `quantizeMatmulRhsBlockwiseI8`, `quantizeMatmulRhsQ4_0`, `quantizeMatmulRhsQ8_0`, `supportsQuantizedMatmulRhs`, `matmul2DQuantizedRhs`, `matmul2DQuantizedRhsQ8_0x4`, `matmul2DQuantizedRhsQ6_Kx4`, `matmul2DQuantizedRhsQ4_Kx4`, `matmul2DQuantizedRhsQ4_Kx8`, `matmul2DQuantizedRhsQ4_Kx2Mmla`, `matmul2DQuantizedRhsQ5_Kx8`, `matmul2DPackedQ8_0x4LhsRhs`, `matmul2DPackedPaddedQ8_0x4LhsRhs`, `matmulPackedQ4_Kx8Q8_Kx4Slice`, `matmulPackedQ4_Kx8RowsSlice`, `matmulPackedQ5_Kx8Q8_Kx4Slice`, `matmulPackedQ5_Kx8RowsSlice`, `matmulPackedQ6_Kx4RowsSlice` |

Geometry structs re-exported through `backend.zig` (and used in the
signatures above): `Conv2dDims` (channel-last `[H,W,Cin] → [OH,OW,Cout]`;
fields `h, w, cin, oh, ow, cout, kh, kw, stride_h, stride_w, pad_h, pad_w,
groups`), `Conv1dDims` (`seq, out_len, in_channels, out_channels, taps,
stride, pad, dilation, groups`), `Pool2dDims` (`h, w, c, oh, ow, kh, kw,
stride_h, stride_w, pad_h, pad_w`), `PoolKind = enum { avg, max, sum }`, and
`WinogradF2Dims` (shared by the F2 and F4 transforms).

**The allocation contract**, precisely scoped:

- Output buffers are always supplied by `ExecContext`; no backend allocates
  tensor outputs. The vector/quant compute leaves (`src/backend/vector/*`,
  the dot kernels in `src/backend/quant/*`) are allocation-free.
- The quantized-RHS dispatch tier (`matmul2DQuantizedRhs*` in
  `native.zig`/`cpu.zig`) deliberately takes an allocator for per-call LHS
  quantization scratch (f32 activation rows → `Q8_0`/`Q8_1`/`Q8_K` blocks);
  the Q8_0 arms have a 512-block stack fast path
  (`q8_0_lhs_stack_blocks = 512`) so decode-sized calls allocate nothing.
  RHS pack preparation (the x4/x8 lane packs) allocates at load time, not
  per matmul. The exec-tier packed-LHS scratch above this seam is pooled
  (§6); pooling the backend-tier scratch below the seam is an open,
  bench-gated task.
- Direct native vector kernels accept a `ParallelConfig` so the execution
  context controls thread-pool ownership — a kernel never creates threads
  and never assumes a pool exists (`.pool = null` runs serially).

`src/backend/ops.zig` defines the shared op vocabulary both backends compile
against: `ElementwiseOp` (`add, sub, mul, div, max, min`), `UnaryOp` (`relu,
exp, sqrt, rsqrt, sigmoid, silu, log, log1p, softplus, neg, abs, sin, cos,
tanh, fast_tanh, gelu, quick_gelu, softcap_30, softcap_15, gelu_quant, elu,
gelu_erf, floor, ceil, round, sign, reciprocal`), `GatedOp` (`glu, swiglu,
geglu, swiglu_clamp10`), and `CompareOp` (`eq, ne, lt, le, gt, ge` — exec-level only, no
backend kernel), plus the scalar reference semantics (`unaryScalar`,
`gatedActivationScalar`, `compareScalar` with IEEE-754 NaN rules, and `erff`,
a faithful musl translation so `gelu_erf` matches `ggml_vec_gelu_erf_f32`).
`gelu_quant` reproduces ggml's f16-LUT GELU bit-for-bit (input and output
rounded through f16, hard clamps at ±10) for llama.cpp numeric parity; `gelu`
is the exact tanh-approximation form.

### 9.3 The scalar backend and the parity contract (`src/backend/cpu.zig`, `src/backend/parity_test.zig`)

The scalar backend is the numeric reference: plain serial loops, no SIMD, no
BLAS, no GPU. It exposes the same `ParallelConfig`-taking signatures as the
native backend but ignores the config (every kernel is serial). Where a
kernel is *routing shared by both backends* rather than divergent numerics —
`im2col`, the Winograd weight/input/output transforms, the conv2d backward
loops, the pool2d backward scatters — the scalar backend reuses the shared
correctness-first implementation serially, so the two backends compute
identical values on those paths by construction; everything with a real SIMD
counterpart (elementwise, reductions, GEMM, pool2d forward, prelu/affine) is
an independent scalar implementation.

`src/backend/parity_test.zig` imports **both** `cpu.zig` and `native.zig`
directly (independent of `-Dbackend`), so `zig build test` always runs the
cross-backend parity suite. What it guarantees:

- `addInto`, `subInto`, `mulInto`, `scaleInto` agree within `1e-6` absolute
  over lengths `{1, 3, 7, 8, 15, 16, 17, 31, 64, 128, 257, 1024}` (edge cases
  around every vector width) plus a 300 000-element case for
  `addInto`/`mulInto`/`scaleInto` that crosses the parallel-split thresholds.
- `sumInto`, `dotInto` agree within `1e-6·n` (the SIMD pairwise/parallel
  reduction reassociates; tolerance scales with the accumulation count).
- `matmulInto`, `matmulTransAInto`, `matmulTransBInto` agree within `1e-5·k`
  over shapes up to `64×64×64` plus `48×192×128`.
- The batched GEMM triple (`matmulBatched2DIntoUnchecked`,
  `...TransA...`, `...TransB...`) agrees over batch counts `{1, 2, 5, 8}`,
  including broadcast RHS (`stride_b = 0`) and shared LHS (`stride_a = 0`).
- `pool2dIntoWithConfig` (max/avg/sum, odd channel counts to exercise SIMD
  remainders), `upsample2xNearestIntoWithConfig`, `preluChannels*`, and
  `channelAffineIntoWithConfig` (with and without shift) agree within `1e-6`.

The suite pins the semantics the native backend must preserve while its
kernels are rewritten for speed; anything not covered by shared routing or a
parity test is covered by the op-level tests in `backend_tests.zig` and the
exec-layer suites.

### 9.4 Native backend: portable `@Vector` kernels (`src/backend/vector/`)

The native backend's non-GEMM work is pure Zig `@Vector` code, portable
across NEON, AVX2/AVX-512, and WASM SIMD. The vector width is chosen at
comptime by `std.simd.suggestVectorLength(f32) orelse 4` — 4 lanes on NEON,
8 on AVX2, 16 on AVX-512 — with separate widths for f16 and f64
(`vector_len_f16`, `vector_len_f64`). Module map:

| Module | Contents |
|---|---|
| `vector/common.zig` | shared leaf: `ParallelConfig`, `V*` width aliases, thread-count gates, contiguous-data accessors |
| `vector/primitives.zig` | `@Vector` leaves: `vecAdd/Sub/Mul/Scale/AddScaled`, `vecUnary/AddUnary/LeakyRelu/Clamp/Gated`, `vecSum/Dot`, f16/bf16/f64 typed twins, `vexpf`, bf16 bit converters |
| `vector/elementwise.zig` | elementwise/reduction entry points, parallel dispatch, snake/groupNorm/prelu/channelAffine kernels |
| `vector/gemm.zig` | dense f32/f16/f64/bf16 GEMM (NN/TN/NT), register-tiled row kernels, `gemmNNRange/gemmTNRange/gemmNTRange` |
| `vector/gemm_blocked.zig` | BLIS-style cache-blocked packed f32 GEMM (§9.5) |
| `vector/matmul_quant.zig` | quantized matmul dispatch + row/column parallel splitters (kernels live in `backend/quant/*`) |
| `vector/batched.zig` | batched dense GEMM (reuses the `gemm*Range` kernels per batch) |
| `vector/conv.zig` | causal depthwise/general/grouped 1-D conv, dense conv1d/col2im1d, channel-last conv2d + im2col, `Conv1dDims`/`Conv2dDims` |
| `vector/pool.zig` | channel-last pool2d (max/avg/sum), pool backwards, `upsample2xNearest` |
| `vector/winograd.zig` | F(2×2,3×3) and F(4×4,3×3) transform kernels |

`ParallelConfig` is one field: `pool: ?*thread.Pool = null`. Whether a kernel
splits is decided by the thread-count gates in `common.zig`
(`elementwiseThreadCount`, `matmulThreadCount`, `columnThreadCount`,
`i8ColumnThreadCount`, `batchedThreadCount`, `depthwiseConvThreadCount`,
`generalConvThreadCount`) against the tuned thresholds in `src/parallel.zig`:

| Constant | Value | Meaning |
|---|---|---|
| `vector_max_threads` | `-Dmax-threads` (default 8) | comptime team ceiling and stack-array bound |
| `vector_elementwise_len_threshold` | 256 Ki elements | below this, elementwise/conv kernels stay serial |
| `vector_matmul_work_threshold` | 1 Mi (m·n·k) | row-split GEMM gate |
| `vector_batched_work_threshold` | 2 Mi | batched GEMM gate |
| `vector_column_min_m` / `vector_column_min_n` | 32 / 128 | column splits are chosen for decode-shaped GEMMs with `m <` the m constant **and** `n ≥` the n constant; at `m ≥ 32` splitting is by rows |
| `vector_column_chunk` | 64 | columns per task in column splits |
| `vector_column_work_multiplier` | 1 | scales the column-split work gate: `columnThreadCount` stays serial below `multiplier × vector_matmul_work_threshold` m·n·k |
| `backward_matmul_work_threshold` | 262 144 | autograd-side pool-enable gate (§5) |
| `backward_async_work_threshold` | 256 Mi | dot-backward async offload gate (§5) |
| `bmm_loop_work_threshold` | 262 144 (= `backward_matmul_work_threshold`) | total m·n·k·batches above which a multi-batch matmul loop splits batches across the pool (`src/exec/matmul.zig`) |
| `bmm_loop_max_chunks` | 16 | chunk cap and stack task-array bound for that batched-loop split |

Parallel splits are deterministic: tasks own disjoint output ranges, so the
threaded result is bit-identical to the serial path for elementwise, conv,
pool, and Winograd kernels (reductions and GEMM state their reassociation
tolerance instead — see §9.3).

Elementwise entry points operate on `f32` tensors by default; the `*Typed`
twins (`elementwiseContiguousIntoTypedWithConfig`, `sumSliceTypedWithConfig`,
`dotIntoTypedWithConfig`, `matmul2DIntoUncheckedTypedWithConfig`) accept
`.f16`, `.bf16`, and `.f64` with the compute/output dtype policy from §8
(f16/bf16 accumulate sums and dots in f32).

### 9.5 GEMM: dispatch precedence, BLAS, and the blocked packed kernel (`src/backend/native.zig`, `vector/gemm.zig`, `vector/gemm_blocked.zig`)

Every dense f32 GEMM entry in the native backend dispatches in a fixed order,
each tier compiled in only when its build flag is set:

1. **GPU** (`-Dgpu≠none`): if `gpu.shouldUseGpu(m, n, k)` passes and the
   provider's `gemmF32` returns `true`, done. A `false` return (gate refusal,
   init failure, kill switch, driver error) falls through — correctness never
   depends on the GPU.
2. **BLAS** (`-Dblas≠none`): if `shouldUseBlas(m, n, k)` — all of `m, n, k ≥
   16` and each dimension fits in `c_int` — the call goes to `cblas_sgemm`
   (row-major, `alpha = 1`, `beta = 0`, overwrite). Batched entries require
   `batch_count > 1` on top and loop `cblas_sgemm` per matrix.
3. **Pure-Zig vector GEMM** otherwise.

BLAS providers are linked per `-Dblas`; on first GEMM the native backend pins
the provider's thread count to `-Dblas-threads` (when nonzero) exactly once,
under a mutex, via the provider-specific setter (`openblas_set_num_threads`,
`bli_thread_set_num_threads`, `mkl_set_num_threads`,
`nvpl_blas_set_num_threads`; Accelerate and the generic `blas` provider have
no setter and are left alone).

Within the pure-Zig tier there are two paths:

- **Register-tiled row kernels** (`gemmNNRange`/`gemmTNRange`/`gemmNTRange`):
  loop orders chosen so the inner loop is contiguous streams (NN/TN
  broadcast-FMA into C rows; NT is a per-element two-stream dot). Parallel
  over row ranges, or over column ranges for decode-shaped `m < 32`.
- **The BLIS-style blocked packed GEMM** (`gemm_blocked.zig`) — the
  `-Dblas=none` answer to training-shaped sizes. Gate:
  `shouldUseBlocked(m, n, k)` = `m ≥ 32 and n ≥ 32 and k ≥ 16` and
  `m·n·k ≥ 192 Mi` (`blocked_work_threshold`). Classic three-level loop nest
  `jc(nc) → pc(kc) → ic(mc)` with packed `A~`/`B~` panels and an `mr×nr`
  register microkernel; TransA/TransB are absorbed by the pack loops so all
  three orientations share the microkernel. Comptime microkernel shape:
  `mr = 8, nr = 12` on aarch64 (24 four-wide accumulators), `mr = 6,
  nr = 2·vector_len` elsewhere. Default `BlockParams`: `kc = 128` (aarch64) /
  `512` (x86, `x86_default_kc`), `mc = 128`, `nc = 1024`, bounded by
  `kc_max = 512`, `mc_max = 256`, `nc_max = 1024`;
  `gemmBlockedWithParams` **panics** on out-of-bounds params (they would
  overrun the static workspace, and the bench sweep feeds runtime params in
  ReleaseFast where asserts vanish). Because backend kernels must stay
  allocation-free and these entries are infallible, the pack panels live in a
  static BSS workspace guarded by `workspace_lock`; concurrent blocked GEMMs
  serialize on it (an accepted trade — pool workers never re-enter the path).
  Parallelism is an (ic-block × column-chunk) cell grid over the persistent
  team; each C tile is written by exactly one task, so results are
  deterministic and thread-count-independent.

f16 GEMM policy (`vector/gemm.zig`): on aarch64 the f16×f16 `@mulAdd` arms
are native `fmla.8h`, so half-precision accumulation is the fast path and
output is bit-stable across releases; every other ISA takes widened twins
(each f16 load converted once, f32 accumulation — strictly more accurate, and
different from the aarch64 bit pattern). `matmulTransB2DIntoUncheckedBf16Rhs`
dots f32 activations against a bf16 RHS without materializing f32 weights.

Batched GEMM (`vector/batched.zig`) parallelizes across batches
(`batchedThreadCount`) and reuses the row-kernel ranges per matrix; `stride_b
= 0` broadcasts one RHS across the batch and `stride_a = 0` shares one LHS.

Threshold provenance and the re-tuning protocol (cool-state runs, thermal
discipline, paired A/B) live in [BENCHMARK.md](BENCHMARK.md); the sweep tool
is `zig build bench-gemm` (`-Dblas=none -- --sweep` for block params,
`-Dgpu=metal` for GPU crossover points).

### 9.6 Convolution, pooling, and image kernels (`vector/conv.zig`, `vector/pool.zig`, `vector/winograd.zig`)

All 2-D image kernels are channel-last: input `[H, W, Cin]`, weights
`[Cout, KH, KW, Cin]`, output `[OH, OW, Cout]`, so every window step is a
contiguous `C`-wide vector op. Forward kernels parallelize over output rows
(bit-identical to serial); the conv2d backward kernels split the same way
(input rows for backward-input, output channels for backward-weight — each
task owns disjoint output, so parallel is bit-identical to serial), while
the pool2d backward scatters are correctness-first serial. Pooling
semantics: `.max` skips out-of-range taps (−inf border, the
ONNX convention), `.avg` averages over *valid* taps only
(`count_include_pad=0`), `.sum` sums valid taps (the upsample VJP; not
exposed publicly). The 1-D families cover causal depthwise FIR
(DeltaNet-style, `[time, channel]` × `[channel, tap]`), channel-mixing dilated
causal conv (`[tap, in, out]` weights so every kernel runs on contiguous
out-channel rows), grouped causal conv, and general non-causal conv1d with
`col2im1d` for transposed conv.

How a conv2d reaches these kernels is exec-side routing
(`src/exec/conv.zig`), but its gates are backend facts worth stating here.
For the dominant shape — 3×3, stride 1, `pad ≤ 1`, `groups == 1`,
`cin ≥ 4`, `oh, ow ≥ 2` — the op takes the **Winograd route**: weight
transform `U = G·g·Gᵀ`, input transform `V = Bᵀ·d·B`, per-plane tile GEMMs
through the ordinary matmul dispatch (BLAS / blocked / row kernels), output
transform with bias folded in and an optional fused ReLU epilogue. Two tiers:

- **F(2×2,3×3)**: 16 coefficient planes, ~2.25× fewer MACs than im2col,
  ~1e-6-relative drift vs the direct kernel (reassociated 3×3 reduction).
- **F(4×4,3×3)**: 36 planes, ~4× fewer MACs, ~1e-5-relative drift; selected
  for large shallow maps — `min(oh, ow) ≥ 14` and `cin ≤ 56` by default
  (bench-tuned: detector-class maps win ~22%, deep-channel recognizer stacks
  lose ~30% and stay on F2).

Gating is read once and cached: the route defaults **on for `-Dblas=none`**
builds and **off when a platform BLAS backs the matmul**
(`winograd_default_on = !native_uses_blas` — Accelerate's AMX prefers one big
im2col GEMM). Runtime overrides: `FUCINA_WINOGRAD=1` forces on,
`FUCINA_NO_WINOGRAD=1` forces off, `FUCINA_NO_WINOGRAD_F4=1` pins large maps
to F2, `FUCINA_WINOGRAD_F4_MIN` / `FUCINA_WINOGRAD_F4_MAXCIN` retune the F4
shape gate. Ineligible convs fall to im2col + one GEMM (`groups == 1`) or the
direct grouped kernel; 1×1 stride-1 pad-0 convs lower to a plain NT matmul.

### 9.7 Quantized matmul dispatch, packed RHS, and the int8 dot arms

The quantized *kernels* belong to §10; this section covers what the backend
tier owns: dispatch, scratch, and ISA selection.

`matmul2DQuantizedRhsWithConfig` switches over `AnyQuantizedMatmulRhs` (one
arm per GGML format) and, per call, quantizes the f32 activation rows into
the format's activation blocks — `Q8_0`/`Q8_1` for legacy and
`IQ4_NL`/`MXFP4`/`NVFP4` formats, `Q8_K` for K-quants and the `IQ*`/`TQ*`
table formats — using the caller's allocator (the deliberate scratch tier
from §9.2). The x4/x8 interleaved fast paths add row-shape policy, tuned so
every row's math stays bit-identical to the kernel that owns it:

- `Q8_0x4`: `m % 4 == 0` goes straight to the packed kernel; `12 ≤ m < 32`
  uses a padded-LHS variant; `m ≥ 32` splits into a multiple-of-4 bulk (x4
  kernel) plus a 1–3-row remainder (row kernel); small odd `m` takes the
  per-row path.
- `Q4_Kx8` engages the x4 activation packing for every `m ≥ 4`
  (`q4_k_x4_min_rows`; its padded-group kernel makes one pass over the packed
  weights); `Q5_Kx8` has no padded kernel, so its bulk+tail split only pays
  at `m ≥ 128` (`q5_k_x4_prefix_min_rows`).
- `Q4_Kx2Mmla` (aarch64 `smmla`) processes row pairs with a `Q8_Kx2` LHS
  packing and a row-kernel tail.

The `matmulPacked*Slice` entries are the same kernels with a
**pre-quantized LHS** — exec's fused split-activation FFN paths quantize
activation rows themselves, so these skip the allocator tier entirely.

**Packed RHS, user-facing story.** Packing a weight once at load time and
matmul-ing against the pack is the hot inference path. Two tiers exist:

- Block-quantized weights: `fucina.PackedRhsLayout = enum { q8_0x4, q6_kx4,
  q4_kx4, q4_kx8, q4_kx2mmla, q5_kx8 }` names the interleaved layouts;
  `backend.PackedRhsFor(layout)` maps a layout to its container type, and the
  facade's `fucina.PackedRhs(dtype)` picks the ISA-best container per dtype
  (`q8_0→x4`, `q6_k→x4`, `q5_k→x8`, `q4_k→x2mmla` when
  `supports_q4_k_mmla` else `x8`). Model code calls
  `weights.packRhs(ctx)` / `packRhsLayout(ctx, .q4_kx8)` on a rank-2
  quantized tensor and feeds the pack to `dotPacked` — full semantics in §10.
- f16/bf16 dense weights: `backend/packed.zig` defines `PackedMatmulFormat =
  enum { f16_rhs_f32, bf16_rhs_f32 }` and `PackedMatmulRhsFor(dtype)`; the
  pack widens the RHS to f32 once (`packRhs` → owns an f32 tensor; caller
  `deinit()`s the container). `matmul2DIntoUncheckedPackedRhsTyped` then runs
  f32 GEMM with widen/narrow bridges, with a dedicated `m == 1` GEMV fast
  path that dots the f16/bf16 activation row directly against the packed f32
  columns (column-parallel over the pool). This tier is reached through
  `ExecContext.packMatmulRhsTyped` / `matmul2DWithPackedRhsTyped` (§6).

**Arch-gated int8 dot arms.** The K-quant/Q8 kernels select their inner dot
at comptime: aarch64 `sdot` inline asm (all aarch64), aarch64 `smmla` behind
`has_aarch64_i8mm` (Graviton3+/Grace-class; selects the `Q4_Kx2Mmla`
layout), x86 AVX2 via the `vpmaddubsw`+`vpsignb` sign trick, AVX-VNNI via the
VEX-encoded `vpdpbusd` (the `{vex}` prefix is mandatory — LLVM's asm parser
does not feature-check and a bare `vpdpbusd` assembles to the EVEX form and
SIGILLs on Alder/Raptor Lake), AVX512-VNNI via EVEX `vpdpbusd`, and a
portable widening tier everywhere else. Debug builds run the portable twins
(the stage2-assembler gate, `common.has_llvm_asm`) — ISA-arm coverage
requires ReleaseFast/ReleaseSafe.

`src/x86dot_check.zig` is the standalone cross-ISA parity checker for these
arms plus the Q4_K/Q8_0/TQ2_0 dot kernels: a self-contained `main` that runs
kernel-vs-scalar semantic asserts on deterministic randomized and extreme
inputs, exits nonzero on mismatch, and prints FNV-1a checksums of the raw
result bits so runs from different machines/emulators can be diffed for
bit-exactness. `zig build x86dot-check` builds and runs it natively
(ReleaseSafe) and additionally compile-checks — never runs — one leg per
feature gate no local substrate can execute (`x86_64_v3`, `alderlake`,
`znver4`, `neoverse_v1`). The file's header carries the dated per-arm
execution attestation table (which arms have actually executed on which
hardware/emulator) and the emulator caveats (qemu ≥ 9.2 required; qemu 7.0
executes AVX2 silently wrong).

### 9.8 Threading: the worker team (`src/thread.zig`, `src/parallel.zig`)

The team itself is context infrastructure and is documented in §6.6: pool
creation and sizing (`cpuThreadCount`, `setMaxThreads`,
`FUCINA_MAX_THREADS`, `FUCINA_SPIN_BUDGET`), the spin-then-park worker
lifecycle, and the `parallelChunks`/`parallelChained` dispatch contracts.
This subsection covers the backend seam.

**The cross-thread handshake.** The pool is created lazily:
`Runtime.tryWorkPool` (under `work_pool_mutex`) initializes one `thread.Pool`
per `ExecContext` with `cpuThreadCount(vector_max_threads) - 1` workers and
publishes it with `Backend.setWorkPool(&pool)` — the atomic
release-store/acquire-load pair from §9.2, because a kernel dispatched on
another thread may race the publication. `Runtime.deinit` unpublishes
(`setWorkPool(null)`) before destroying the pool. Exec ops call
`enableNative*PoolForWork(work, threshold)` helpers so the team is only
instantiated once an op is actually big enough to split. `thread.zig` also
provides `Mutex`/`Condition` (thin `std.Io` wrappers), `WaitGroup`,
`ThreadSafeAllocator` (mutex-guarded child allocator), `Chain`, and
`OneShotWorker` — a single persistent futex-parked thread used by
dot-backward to overlap the two gradient GEMMs (§5).

### 9.9 GPU offload (`src/backend/gpu.zig`, `metal.zig`, `cuda.zig`)

Both providers implement the same eager accelerator contract:

- **Gates decide, dispatchers run.** Cheap `shouldUse*` gates (no device
  init) sit at the dispatch sites; every dispatch entry returns
  `false`/`null` when the GPU did not run, and the caller falls through to
  BLAS/vector — correctness never depends on the GPU.
- **Submit eagerly; synchronize at host visibility.** Dense f32 commands are
  encoded and submitted before the op returns. Output storage carries only a
  completion/lifetime token: GPU consumers remain queue-ordered and a CPU
  data accessor/kernel performs the deferred wait. F16 and quantized panel
  staging remain blocking. The public `Tensor` API still has no device type or
  location state, and no op description/graph is retained.
- **Lazy init.** The device/library/context is created on first
  above-threshold use, double-checked under a mutex so concurrent
  `ExecContext`s share one device. Config (env vars) is read once, separately
  from device init, so below-threshold probes stay cheap.
- **Shape gates.** General GEMM requires `m ≥ 32, n ≥ 32, k ≥ 16`.
  Resident dense-f32 GEMV/small-m GEMM has the separate `m ≤ 8`,
  `n,k ≥ 256`, `FUCINA_GPU_MIN_WORK_GEMV` gate.

The `FUCINA_GPU*` runtime knobs — kill switch, per-kind work gates, the
MoE fill gate, tracing, TF32, the transient floor, VRAM budget, kernel
source, and the decode opt-in — are all read once at first use and are
tabulated with their defaults in §2.6.

The eager f32/f16/dense-quant implementation and its ordering/teardown contract are described
in [GPU-OFFLOAD.md](GPU-OFFLOAD.md). Both providers keep their queue/streams
and library state open across calls. Metal additionally caches storage page
wrappers; CUDA uses a bounded eight-slot device pool, registers pooled host
allocations once, and connects persistent upload/compute/download lanes with
events. A pending producer's device address passes directly to a dependent
GEMM.

#### 9.9.1 Metal (`src/backend/metal.zig`, `-Dgpu=metal`, macOS)

The kernels are vendored MSL compiled once at lazy init from embedded source
by the ObjC shim (`src/backend/metal/shim.m`): the MLX "steel" f32/f16 GEMM
(`metal/mlx_gemm.metal`, MIT, Apple) and the llama.cpp quantized `mul_mm`
(`metal/ggml_mul_mm.metal`, dequant-in-kernel). What offloads:

- **Dense f32 GEMM/GEMV** `nn`/`tn`/`nt` and strided-batched
  (`gemmF32Async`/`gemmBatchedF32Async`; a batched call is ONE dispatch with
  grid depth = batch), behind the general or resident-small-m gate. The direct
  slice `gemmF32` twin is blocking for parity/benchmarks.
- **f16 NT GEMM** (`gemmF16NtAsync`) behind `shouldUseGpuF16ForRhs`: the
  mixed steel instantiation reads f16 operands and writes the public f32
  output directly. It commits immediately and attaches a Work; there is no
  shared staging buffer, f16 lock, CPU widen pass, or result re-rounding. The
  old direct-slice `gemmF16Nt` remains blocking only for low-level parity
  tests/bench callers.
- **Dense quantized prefill** (Q4_K/Q6_K/Q8_0): exec's
  `denseQuantMatmulGpu` seam (`src/exec/quant_matmul.zig`) offloads
  `m ≥ 32` stable-weight matmuls behind the compact/raw or packed-CPU
  per-format gate when
  `k % QFormat.kMultiple() == 0` (32 for q8_0, 256 for q4_k/q6_k) and
  `n % 4 == 0`. `gemmQuantNtAsync` binds input/output tensor storage
  directly and copies its ≤4 KiB tile table into command-owned bytes (up to
  8192 rows); shared-input batches encode multiple weight matrices without
  replicating input rows. Transient RHS or longer prompts retain the blocking
  chunk fallback.
- **MoE expert FFN** (llm tier, `src/llm/gemma/moe.zig`): CPU gathers
  activation rows into shared staging panels (`qmoeStage`, grow-only
  MTLBuffers), dispatches grouped tile-table GEMMs (`gemmQGroupedNt`, one
  `QMMTile` per 32-row output tile per expert), and reads results back — all
  under the process-global `qmoe_lock`. Gated by `shouldUseGpuQMoe` (total
  m·n·k across both projections) *and* `qmoeFillAcceptable` (per-tile GPU
  cost is fill-independent, so below ~50% occupancy the CPU wins).

Weight residency: `allocResidentBytes(len)` returns device-owned,
page-aligned unified-memory bytes the CPU reads through the same slice; this
is a performance cache, not a correctness precondition — pageable client
wraps are re-wired into the GPU address space on every commit (~45 µs/MB), so
stable weights should live resident. The provider keeps a bounded
address-keyed registry (512 ranges) so dispatch paths recognize resident
operands without caller flags. Dense f32/f16 inputs and f32 outputs own one
page wrapper on each storage allocation; pooled reuse changes values, not the
mapping, and the wrapper is evicted before the allocation is freed. Quantized
stable weight pages retain the bounded address-keyed shim cache. The stale-pages rule
is absolute: only bytes whose address is process-lifetime-stable may be
flagged cacheable
(`RhsLifetime.stable_process`) — a cached wrap of a freed-and-reused page
reads stale data. `freeResidentBytes` unregisters and releases.

What does **not** offload on Metal, per current tree: **quantized** decode GEMV
(`decodeGemvEnabled()` is hard-`false`; resident dense-f32 GEMV is separate),
attention (`shouldUseGpuAttn`/`attnPrefillF16` return
`false`; the CPU tiled kernel runs), and the ES parameter-update device arm
(`esPerturb`/`esUpdate`/`esAnchor` are stubs returning `false` — on unified
memory the CPU kernels already mutate the shared pages the GPU reads
zero-copy). Quantized matmuls reached from *trainable* autograd inputs pass
`allow_gpu = false` (`QuantizedMatmulOptions`), keeping the training path on
CPU unless a gradient-aware GPU policy is added deliberately.

#### 9.9.2 CUDA (`src/backend/cuda.zig`, `-Dgpu=cuda`, Linux)

Host binding is `dlopen` (`src/backend/cuda/api.zig`): `libcuda.so.1` and
`libcublas` are loaded at runtime, so **no CUDA SDK is needed at build time**
and `-Dgpu=cuda -Dtarget=x86_64-linux-gnu` cross-compiles from any machine.
Missing libraries degrade per capability (no cuBLAS ⇒ only the f32/f16 GEMM
arms are disabled). Quantized/GEMV/ES/attention kernels are vendored CUDA C
(`cuda/kernels.cu`) shipped as committed PTX (`cuda/kernels.ptx`, driver JIT,
disk-cached, ~26 ms cold), with an NVRTC recompile fallback
(`FUCINA_GPU_KERNELS=src` forces it). Persistent upload, compute, and download
streams use reusable events; cuBLAS stays bound to compute and submission holds
a short dispatch lock. Pooled f32 host allocations are registered once for
direct asynchronous DMA. What offloads:

- **f32 GEMM/GEMV** nn/tn/nt + strided-batched via cuBLAS, strict FP32 math by
  default (`FUCINA_GPU_TF32=1` opts into TF32). The f32 gates add a
  *transient floor* on top of `min_work`: non-resident operands stream over
  PCIe (measured ~10.6 GB/s pageable), so shapes below `2^33` m·n·k or
  `m < 128` are refused even when the plain gate passes (trace counts these
  separately as re-tuning evidence). An already device-resident RHS instead
  uses the `2^27` resident gate; resident `m≤8` uses the separate GEMV gate.
  Pending outputs pass their device address directly to a dependent op. H2D,
  compute, and D2H overlap on their three lanes; the final D2H lands directly
  in the ordinary exec-owned output (a pinned-stage fallback remains for a
  host allocator the driver cannot register).
- **f16 NT GEMM** via `cublasGemmEx` (f16 operands, direct f32 output and f32
  accumulation) through the same eight async slots and three persistent lanes
  as f32. Resident RHS decode uses the separate `2^20` gate; transient decode
  is refused, while streamed prefill retains the `2^27` gate.
- **Dense quantized prefill** (Q4_K/Q6_K/Q8_0): stable RHS bytes resolve to one
  managed resident allocation. `gemmQuantNtAsync` reuses the slot's activation,
  output, pinned-tile, and device-tile buffers; a pending f32 producer passes
  its device address directly. Shared-input batches launch each weight matrix
  on the same stream without copying activation rows. On tensor-core devices,
  adaptive N32/N64 WMMA kernels consume the same half-rounded dequantized
  operands as the scalar fallback and accumulate f32; N32 is selected only for
  severe grid underfill. Full output tiles store directly and partial tiles use
  guarded staging. `FUCINA_GPU_QUANT_MMA=0` retains the scalar path for
  compatibility/diagnosis. Host download is deferred to the output Work.
- **Grouped MoE** uses the same tile kernel but keeps its required CPU phase
  boundaries (`qmoeStage`, `qmoe_lock`). Panel/tile H2D, compute, and panel D2H
  are now event-chained across persistent streams; the CPU performs one final
  fence when GeGLU/scatter needs the result, rather than synchronizing compute
  and then starting a blocking download.
- **Fused prefill attention** (`attnPrefillF16`): online-softmax grouped
  attention over f16 KV with the CPU tiled kernel's exact semantics
  (absolute positions, pre-clamped sliding window, causal or bidirectional,
  per-head KV mapping, `d ≤ 256`). Stateless and blocking — Q/K/V stream in,
  the output streams back. Gated by `shouldUseGpuAttn` on q·kv·heads·d ≥
  `2^28`; decode never reaches this seam, and the KV cache itself stays in
  host memory.
- **Decode GEMV** (opt-in, `FUCINA_GPU_DECODE=1`): warp-per-row dequant-dot
  for `m ≤ 8` against **resident-or-adoptable weights only** — at decode
  shapes the op is bytes-bound and streaming weights per token is a strict
  loss, so a registry miss on transient RHS refuses and the caller stays on
  the CPU int8 kernels. Kept opt-in pending a parity-oracle pass on
  sampled-token streams.
- **ES device arm** (`esPerturb`, `esUpdate`, `esAnchor`): seeded
  perturbation/update/anchor kernels for evolution-strategies training (§11)
  that write the caller's live resident storage (never an adopted snapshot)
  and reproduce the CPU noise contract bitwise.

Residency: `allocResidentBytes` = `cuMemAllocManaged` + `READ_MOSTLY` advice
+ prefetch-on-first-use — unified addressing means a resident RHS dispatches
with zero weight transfer while the CPU fallback reads the same pointer.
Stable (`RhsLifetime.stable_process`) RHS bytes are additionally *adopted*
into the managed registry on first use — the analog of Metal's wrap cache,
same stale-pages rule: mmap'd weights cross PCIe once per process, not per
dispatch. `FUCINA_GPU_VRAM_BUDGET` bounds tracked allocations (default ~80%
of free VRAM at init); over-budget requests return `null` and callers fall
back to host bytes + transient (the Metal OOM path). Residency requires
`CONCURRENT_MANAGED_ACCESS` (absent on WSL2/some Jetson targets → residency
disabled, transient/CPU fallback with a one-time warning).

#### 9.9.3 `internal.gpu` hooks and tracing (`src/fucina.zig`)

Users keep ordinary eager `Tensor` values; residency and tracing are
backend-owned details deliberately kept off the public root, under
`fucina.internal.gpu`:

```zig
pub const gpu = struct {
    pub const enabled = backend.gpu_impl.enabled;             // comptime: -Dgpu build?
    pub const has_quant_gemm = backend.gpu_impl.has_quant_gemm; // dequant-in-kernel GEMM?
    pub const allocResidentBytes = backend.gpu_impl.allocResidentBytes;
    pub const freeResidentBytes = backend.gpu_impl.freeResidentBytes;
    pub const traceEnabled = backend.gpu_impl.traceEnabled;
    pub const traceReset = backend.gpu_impl.traceReset;
    pub const traceDump = backend.gpu_impl.traceDump;
};
```

`has_quant_gemm` is the capability loaders key on when reshaping CPU-side
weight representations for the GPU quant path (e.g. `src/llm/gemma/gemma4.zig`
copying mmap'd expert tensors into resident storage) — a provider can be
`enabled` while its quantized arms are stubs. `RhsLifetime`
(`fucina.RhsLifetime`) is how callers communicate the storage-stability
guarantee that authorizes address-keyed caching of an RHS; the enum, the
cacheability rule, and the pooled-storage prohibition are §6.7.

Tracing workflow: run with `FUCINA_GPU_TRACE=1`, warm the workload, call
`traceReset()` at the start of the measurement window and `traceDump()` at
the end. The dump (stderr) breaks down per-kind dispatch counts and wall/GPU/
scheduling time (f32/f16/quant, plus gemv/attn and H2D/D2H bytes on CUDA),
resident-vs-streamed RHS counts, resident-bytes allocations, and — most
useful for threshold tuning — the gate-decision counters (pass, below-gate,
shape-reject, transient-floor, shim/CUDA errors) and the top shapes by
dispatch wall time (Metal). `traceReset`/`traceDump` are no-ops when tracing
is off, so instrumentation can stay in place unconditionally:

```zig
fn snippetGpuTraceAndResidency(weight_bytes: []const u8) void {
    const gpu = fucina.internal.gpu;
    // The trace hooks are callable on every build; they no-op unless the
    // process runs with FUCINA_GPU_TRACE=1 on a -Dgpu build.
    gpu.traceReset();
    // ... run the measured (warm) workload ...
    gpu.traceDump(); // per-kind dispatch counts/wall/GPU time to stderr

    if (comptime gpu.enabled) {
        // Device-owned weight storage: stable bytes dispatch with zero
        // per-call transfer; release through the same hook.
        if (gpu.allocResidentBytes(weight_bytes.len)) |dev| {
            @memcpy(dev, weight_bytes);
            // ... hand `dev` to the model; when the owner drops it:
            gpu.freeResidentBytes(dev);
        }
    }
} // requires a -Dgpu build to actually offload
```

GPU parity is tested in-tree: the provider modules carry their own test
blocks (f32 orientation/edge-tile parity vs a f64 reference, f16 NT, the
quantized formats vs dequantized references, grouped expert tiles, the fill
gate arithmetic), compiled and run only on `-Dgpu` builds — `zig build test
-Dgpu=metal` on an Apple Silicon machine, `-Dgpu=cuda` on a CUDA box.
Threshold defaults are measurement-backed; the protocol and the recorded
numbers live in [BENCHMARK.md](BENCHMARK.md).

## 10. Quantization

Fucina treats ggml/llama.cpp block-quantized weights as first-class tensor
data: every GGUF quantized wire format is a `DType` whose storage element is
the exact ggml block struct, decoders and encoders are byte-for-byte ports of
`ggml-quants.c`, and matmuls against quantized weights run int8 dot-product
kernels over dynamically quantized activations. The stack has three tiers:

1. **dtype/block tier** (`src/dtype.zig`, §8) — block structs, block-size
   constants, and the comptime predicates (`isBlockQuantized`,
   `supportsQuantizedMatmulRhs`, `supportsQuantizedGetRows`, `blockSize`,
   `logicalDType`).
2. **kernel tier** (`src/backend/quant.zig` and `src/backend/quant/*.zig`) —
   encoders, decoders, activation quantizers, dot/matmul kernels, packed RHS
   layouts, and the format-trait table. Reachable in-tree as
   `fucina.internal.backend_mod.quantized_matmul`; application code normally
   never calls it directly.
3. **facade tier** (`src/ag/tensor.zig`, `src/exec/quant_matmul.zig`) —
   `fucina.Tensor(.{ .dtype = .q4_k, ... })` constant tensors, `dot` with a
   quantized RHS, `getRows`, `to(.f32)`, `packRhs`/`dotPacked`, and the
   ternary STE training op `dotTernarySte`.

GGUF container mechanics (parsing, `tensorByteLen`, zero-copy tensor bytes)
are §12; the LLM weight wrappers that own packed RHS containers are §13; the
CPU/GPU dispatch machinery underneath is §9.

### 10.1 Format inventory (`src/backend/quant/types.zig`, `src/dtype.zig`)

Each quantized `DType` stores rows as a contiguous sequence of fixed-size
blocks; a row of `k` logical elements is `k / block_size` blocks (`k` must
divide exactly — `QuantizedFormatError.InvalidQuantizedLength` otherwise).
The block structs are `extern struct`s matching ggml's wire layout exactly
and are re-exported at the root (`fucina.BlockQ4_K`, ...).

| DType | Block struct | Elems/block | Bytes/block | f32 encoder | Matmul kernel | LHS activation |
|---|---|---|---|---|---|---|
| `.q1_0` | `BlockQ1_0` | 128 | 18 | — | cold | Q8_0 |
| `.q4_0` | `BlockQ4_0` | 32 | 18 | yes | cold | Q8_0 |
| `.q4_1` | `BlockQ4_1` | 32 | 20 | yes | cold | Q8_1 |
| `.q5_0` | `BlockQ5_0` | 32 | 22 | yes | cold | Q8_0 |
| `.q5_1` | `BlockQ5_1` | 32 | 24 | yes | cold | Q8_1 |
| `.q8_0` | `BlockQ8_0` | 32 | 34 | yes | **hot** (+ x4 packed) | Q8_0 |
| `.q8_1` | `BlockQ8_1` | 32 | 36 | yes | — (activation format) | — |
| `.q2_k` | `BlockQ2_K` | 256 | 84 | — | cold | Q8_K |
| `.q3_k` | `BlockQ3_K` | 256 | 110 | — | cold | Q8_K |
| `.q4_k` | `BlockQ4_K` | 256 | 144 | yes | **hot** (+ x8 / x2mmla packed) | Q8_K |
| `.q5_k` | `BlockQ5_K` | 256 | 176 | yes | **hot** (+ x8 packed) | Q8_K |
| `.q6_k` | `BlockQ6_K` | 256 | 210 | yes | **hot** (+ x4 packed) | Q8_K |
| `.q8_k` | `BlockQ8_K` | 256 | 292 | yes | — (activation format) | — |
| `.iq1_s` | `BlockIQ1_S` | 256 | 50 | — | cold | Q8_K |
| `.iq1_m` | `BlockIQ1_M` | 256 | 56 | — | cold | Q8_K |
| `.iq2_xxs` | `BlockIQ2_XXS` | 256 | 66 | — | cold | Q8_K |
| `.iq2_xs` | `BlockIQ2_XS` | 256 | 74 | — | cold | Q8_K |
| `.iq2_s` | `BlockIQ2_S` | 256 | 82 | — | cold | Q8_K |
| `.iq3_xxs` | `BlockIQ3_XXS` | 256 | 98 | — | cold | Q8_K |
| `.iq3_s` | `BlockIQ3_S` | 256 | 110 | — | cold | Q8_K |
| `.iq4_nl` | `BlockIQ4_NL` | 32 | 18 | — | cold | Q8_0 |
| `.iq4_xs` | `BlockIQ4_XS` | 256 | 136 | — | cold | Q8_K |
| `.tq1_0` | `BlockTQ1_0` | 256 | 54 | — | cold | Q8_K |
| `.tq2_0` | `BlockTQ2_0` | 256 | 66 | yes | **hot** (mul-free ternary) | Q8_K |
| `.mxfp4` | `BlockMXFP4` | 32 | 17 | — | cold | Q8_0 |
| `.nvfp4` | `BlockNVFP4` | 64 | 36 | — | cold | Q8_0 |

Reading the table:

- **f32 encoder** — the format has a `quantizeRowForDType` prong (§10.6).
  Every format decodes to f32 (`dequantizeRowForDType` covers all 26), so
  "encoder = —" means *decode/matmul-only*: usable as loaded weights, never
  producible in-process or by `gguf.encodeF32`.
- **Matmul kernel** — *hot* formats have dedicated SIMD kernels plus packed
  (column-interleaved) RHS layouts tuned per ISA; *cold* formats
  (`src/backend/quant/cold.zig`) have a generic per-block dot path that is
  correct and tested but not benchmark-tuned. `.q8_1` and `.q8_k` have no
  matmul kernel at all (`supports_matmul == false`): they exist as the
  *activation* side of the int8 dots and as decodable tensor data.
- **LHS activation** — the block format the f32 activations are dynamically
  quantized to before the int8 dot (§10.5): Q8_0 for the 32-element weight
  families plus `.q1_0` (128) and `.nvfp4` (64), Q8_1 for the offset formats
  `.q4_1`/`.q5_1` (the offset term needs the per-block activation sum Q8_1
  carries), Q8_K for all 256-element formats.
- First-class end-to-end (encoder + hot kernel + GGUF export): `.q8_0`,
  `.q4_k`, `.q5_k`, `.q6_k`, `.tq2_0` — the first four additionally have
  packed (column-interleaved) RHS layouts; `.tq2_0` has none
  (`PackedRhsLayout` has no ternary member; `packRhs` on a `.tq2_0` tensor
  is a compile error — its dedicated kernels work from plain blocks).

The kernel tier's trait layer describes each format programmatically:
`QuantizedMatmulFormat` (enum: `fucina_w8a8_rhs` plus `ggml_*` per dtype),
`QuantizedMatmulTraits` (block size, byte size, storage/scale layout,
`supports_from_float`/`supports_to_float`/`supports_matmul`,
`matmul_kernel`), `matmulTraits` (comptime) / `matmulTraitsRuntime`,
`formatForDType`, `supportsMatmul`, and the `QuantizedMatmulKernel`,
`QuantizedStorageLayout`, `QuantizedScaleLayout` enums. Block-size constants
(`q8_0_block_size` = 32, `qk_k_block_size` = 256, `k_scale_size` = 12,
`iq4_nl_block_size`/`mxfp4_block_size` = 32, `nvfp4_block_size` = 64,
`nvfp4_subblock_size` = 16, `q1_0_block_size` = 128, `q4_0_block_size`,
`q4_1_block_size`, `q5_0_block_size`, `q5_1_block_size`,
`q8_1_block_size`) originate in `src/dtype.zig`; `fucina.q8_0_block_size`
is re-exported at the root.

Separate from the ggml formats, the kernel tier also ships a Fucina-native
**W8A8** container (`QuantizedMatmulRhsI8`, format `.fucina_w8a8_rhs`):
symmetric per-(column, group) int8 weights stored transposed `[n][k]` with
f32 scales (`default_i8_group_size` = 32), built by `quantizeRhsBlockwiseI8`
and multiplied by `matmulI8BlockwiseTile`/`matmulI8BlockwiseRange` against
per-row int8 activations (`quantizeActivationsPerRowI8`). It is not a tensor
dtype and has no facade surface; it exists for W8A8 experiments below the
facade.

### 10.2 The block-quantized public tensor (`src/ag/tensor.zig`)

`fucina.Tensor(.{ .dtype = <quant dtype>, .tags = ... })` instantiates a
*quantized constant tensor*: a tagged wrapper over the typed raw tensor whose
element type is the block struct. It deliberately exposes **only** quantized
operations — no autograd (`requiresGrad()` is always `false`, there is no
`variable`), no float math (`add`, `softmax`, ... are absent at comptime).
The public surface:

```zig
// constructors (all validate rank/shape; shapes are LOGICAL element counts)
pub fn constant(ctx: *ExecContext, value: RawTypedTensor) !Self          // consumes value on success
pub fn fromTensor(ctx: *ExecContext, value: RawTypedTensor) !Self        // alias of constant
pub fn fromBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self   // copies blocks
pub fn fromStorageSlice(...) !Self                                       // alias of fromBlocks
pub fn fromBorrowedBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []Elem) !Self // zero-copy borrow

// accessors / structure
pub fn deinit(self: *Self) void
pub fn data(self: *Self) ![]Elem
pub fn dataConst(self: *const Self) ![]const Elem
pub fn copyTo(self: *const Self, dst: []Elem) !void
pub fn asRawTensor(self: *const Self) *const RawTypedTensor
pub fn axis(comptime tag: Tag) usize
pub fn hasTag(comptime tag: Tag) bool
pub fn dim(self: *const Self, comptime tag: Tag) usize
pub fn shape(self: *const Self) [tensor_rank]usize
pub fn requiresGrad(_: *const Self) bool                                 // always false
pub fn withTags(self, ctx, comptime new_tags_spec) !Tensor(...)          // retag view, same rank

// quantized operations
pub fn to(self, ctx, comptime target_dtype: DType) !Tensor(...)          // .f32 only: dequantize
pub fn materialize(self, ctx) !Self                                      // copy into owned storage
pub fn concat(self, ctx, comptime tag, others: []const *const Self) !Self// rank-2, row axis only
pub fn getRows(self, ctx, comptime tag, indices: []const usize, comptime out_tag) !Tensor(f32 ...)
pub fn packRhs(self, ctx) !PackedRhs(dtype)                              // §10.3
pub fn packRhsLayout(self, ctx, comptime layout: PackedRhsLayout) !PackedRhsFor(layout)
```

Semantics:

- **Shapes are logical.** A `[n, k]` quantized tensor holds
  `n * (k / block_size)` blocks; `fromBlocks` fails with
  `TensorError.InvalidDataLength` when the slice length disagrees and with
  `TensorError.InvalidShape` when `k` is not a multiple of the block size
  (the shape check fires before the kernel tier's
  `QuantizedFormatError.InvalidQuantizedLength` is ever reached on this
  path). Weight tensors follow the ggml convention
  `[out, in]` = `[n, k]`: row `r` of blocks is *output column* `r`.
- **Ownership.** `fromBlocks` copies into context-owned storage;
  `fromBorrowedBlocks` borrows — the caller keeps ownership and the blocks
  must outlive the tensor (this is the mmap'd-GGUF path: the loader
  reinterprets mapped tensor bytes as a block slice and borrows them
  zero-copy; see §12). `materialize` converts a borrowed tensor into an
  owned copy. `constant`/`fromTensor` consume the raw tensor on success;
  on error, ownership stays with the caller. Every constructor's result is
  released with `deinit`.
- **`to(.f32)`** dequantizes the whole tensor (rows × columns) into a fresh
  f32 tensor with the same tags; any target other than `.f32` is a compile
  error. **`getRows`** gathers rows from the *first* axis by index and
  dequantizes only those rows — the embedding-lookup path (token embeddings
  stay quantized; only the looked-up ids are widened). Errors:
  `TensorError.InvalidShape` for empty `indices`,
  `TensorError.IndexOutOfBounds` for an index ≥ row count. `getRows` is
  comptime-restricted to rank-2 tensors; `to(.f32)` requires a rank-2
  value at runtime (`TensorError.InvalidShape` otherwise).
- **`concat`** joins quantized tensors along the row axis without
  dequantizing (rank-2, axis 0 only — comptime-checked).
- **Thread-safety.** Quantized tensors are immutable after construction;
  concurrent reads are safe. All ops take an `ExecContext`, which is
  single-threaded externally (§6) — internal parallelism comes from the
  context's work pool.

An asset-free construction path exists because the first-class encoders run
in-process: `fucina.gguf.encodeF32` (§10.6) turns f32 data into wire blocks,
which `fromBlocks` accepts directly:

```zig
test "encode f32 rows to Q8_0 blocks and dequantize them back" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // Two 32-element rows -> one BlockQ8_0 per row.
    var src: [2 * fucina.q8_0_block_size]f32 = undefined;
    for (&src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.25;

    var blocks: [2]fucina.BlockQ8_0 = undefined;
    try fucina.gguf.encodeF32(.q8_0, &src, std.mem.sliceAsBytes(&blocks));

    const W = fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });
    var w = try W.fromBlocks(&ctx, .{ 2, fucina.q8_0_block_size }, &blocks);
    defer w.deinit();

    var dense = try w.to(&ctx, .f32); // block-wise dequantize
    defer dense.deinit();
    for (try dense.dataConst(), src) |got, want|
        try std.testing.expectApproxEqAbs(want, got, 0.01);
}
```

#### f32 × quantized-RHS `dot`

The f32 tensor's `dot` (§4) dispatches on the RHS dtype at comptime: when the
RHS is a block-quantized tensor whose dtype satisfies
`supportsQuantizedMatmulRhs` (everything in the table except `.q8_1` and
`.q8_k`), the contraction runs the quantized matmul instead of a dense GEMM.
Requirements, all comptime-checked:

- the RHS must have exactly one free axis and be stored **[free, contract]**
  (e.g. weight tags `{ .out, .in }` contracted over `.in`) — the ggml weight
  layout, never transposed at runtime;
- no shared batch tags between LHS and RHS;
- the LHS may have any number of free axes (they are flattened to `m` rows
  around the contraction).

The runtime path quantizes the LHS activations (§10.5), runs the format's
kernel, and reshapes back. Gradients: the quantized weight is a constant
(it never receives grad); the LHS gradient is supported and flows through
the **dequantized** weight — the backward node holds a view of the block
data and dequantizes it transiently (`ConstRhsDotBackward`,
`src/ag/backward.zig`). On GPU builds the forward may offload to the
dense-quant GEMM provider (q4_k/q6_k/q8_0 only, §9); the facade
automatically disables offload when the LHS requires grad, so training
numerics stay on the CPU path.

```zig
test "f32 activations contract against a Q8_0 weight tensor" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var wsrc: [2 * fucina.q8_0_block_size]f32 = undefined;
    for (wsrc[0..fucina.q8_0_block_size]) |*v| v.* = 1.0;
    for (wsrc[fucina.q8_0_block_size..]) |*v| v.* = 2.0;
    var blocks: [2]fucina.BlockQ8_0 = undefined;
    try fucina.gguf.encodeF32(.q8_0, &wsrc, std.mem.sliceAsBytes(&blocks));

    const W = fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });
    var w = try W.fromBlocks(&ctx, .{ 2, fucina.q8_0_block_size }, &blocks);
    defer w.deinit();

    const x_values = [_]f32{1} ** fucina.q8_0_block_size;
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 1, fucina.q8_0_block_size }, &x_values);
    defer x.deinit();

    var y = try x.dot(&ctx, &w, .in); // y: .{ .batch, .out }
    defer y.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 32), (try y.dataConst())[0], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 64), (try y.dataConst())[1], 1e-2);

    var row = try w.getRows(&ctx, .out, &.{1}, .seq); // f32 gather of weight row 1
    defer row.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2), (try row.dataConst())[0], 1e-2);
}
```

### 10.3 RHS containers and packed layouts (`src/backend/quant/types.zig`, `src/exec/quant_matmul.zig`)

Two families of weight-side containers exist below the tensor facade, both
re-exported at the root:

**Plain per-format containers** — `fucina.QuantizedMatmulRhsQ2_K`,
`fucina.QuantizedMatmulRhsQ4_K`, `fucina.QuantizedMatmulRhsQ5_K`,
`fucina.QuantizedMatmulRhsQ6_K` (root); the kernel tier additionally
defines `QuantizedMatmulRhsQ8_0`, `QuantizedMatmulRhsQ4_0`,
`QuantizedMatmulRhsQ3_K`, the
`QuantizedMatmulRhsRowsFor(dtype)` generic (instantiated as
`QuantizedMatmulRhs{Q1_0,Q4_1,Q5_0,Q5_1,IQ1_S,IQ1_M,IQ2_XXS,IQ2_XS,IQ2_S,
IQ3_XXS,IQ3_S,IQ4_NL,IQ4_XS,TQ1_0,TQ2_0,MXFP4,NVFP4}`), the
row wrappers (`QuantizedRowsQ8_1` instantiates the `QuantizedRowsFor(dtype)`
generic; `QuantizedRowsQ8_0` and `QuantizedRowsQ4_0` are hand-written, and
`QuantizedRowsQ4_0`'s allocator is non-optional — no borrow support), and
the type-erased `AnyQuantizedMatmulRhs` union the backends dispatch on. A
plain container is blocks + `k`/`n` dims; the hot-format containers (`Q8_0`,
`Q4_K`, `Q5_K`, `Q6_K`), the `Q2_K` container, and the
`QuantizedRowsFor` generic carry an
**optional allocator**:
`allocator = null` means the blocks are *borrowed* (mmap'd GGUF, packed ES
genomes) and `deinit` frees nothing. Ordinary users never build these —
`ExecContext` wraps tensor blocks in stack-allocated borrow containers per
dispatch, and the LLM MoE loader borrows expert blocks through
`fucina.MoeRhs` (§13).

**Packed containers** — column-interleaved copies of a quantized weight,
laid out so the innermost kernel loop feeds the target's int8 dot
instruction (`sdot`/`smmla` on aarch64, VNNI/AVX2 on x86). Root exports:

```zig
pub const PackedRhsLayout = enum { q8_0x4, q6_kx4, q4_kx4, q4_kx8, q4_kx2mmla, q5_kx8 };
pub fn PackedRhs(comptime dt: DType) type   // ISA-best layout for a dtype
pub const QuantizedMatmulRhsQ8_0x4;         // + Q4_Kx4, Q4_Kx8, Q4_Kx2Mmla, Q5_Kx8, Q6_Kx4
pub const supports_q4_k_mmla: bool;         // aarch64 + i8mm target feature
```

`PackedRhs(dt)` maps q8_0→x4, q6_k→x4, q5_k→x8, and q4_k→x2mmla on
aarch64+i8mm targets, x8 otherwise. Each packed container owns its blocks
(non-optional allocator; `deinit` frees), holds `k`/`n`, and carries a
comptime `layout` tag that `dotPacked` dispatches on. `PackedRhsFor(layout)`
(kernel tier) maps a layout back to its container type. The `q4_kx4` layout
exists only for kernel comparisons and has no facade entry (comptime error).

Packing and consuming happen on the facade:

- `w.packRhs(ctx)` / `w.packRhsLayout(ctx, layout)` — pack a rank-2
  contiguous quantized tensor (`TensorError.UnsupportedView` when not
  contiguous). The tensor can be released after packing; the packed
  container is independent. `packRhsLayout` is the escape hatch to force a
  non-default layout (e.g. x8 on MMLA hardware to exercise the fused
  kernels). Equivalent `ExecContext` entries: `packMatmulRhsQ8_0x4`,
  `packMatmulRhsQ6_Kx4`, `packMatmulRhsQ4_Kx4`, `packMatmulRhsQ4_Kx8`,
  `packMatmulRhsQ4_Kx2Mmla`, `packMatmulRhsQ5_Kx8`.
- `x.dotPacked(ctx, &packed, contract_tag, out_tag)` — rank-2 f32 LHS stored
  `[free, contract]`; returns `[free, out_tag]`. **No gradient support**:
  returns `error.GradientQuantizedMatmulUnsupported` when `self` requires
  grad.
- `x.rmsNormMulDotPacked(ctx, &norm_weight, eps, &packed, contract_tag, out_tag)`
  — fused pre-norm + packed GEMM: normalizes up to 4 rows at a time into
  task-private scratch with the exact `rmsNormMulRows` kernel and quantizes
  with the fused packers, so the normalized tensor is never materialized;
  matches `rmsNormMul` + `dotPacked` to ≤ 1 ulp observed (the packed
  matmul's internal LHS quantizer arrangement may differ in the last ulp,
  the `splitSwiGluDotPacked` precedent). Kernels exist for
  `q8_0x4`/`q4_kx8`/`q5_kx8`/`q6_kx4`; `q4_kx2mmla` is a deliberate
  comptime error — callers fall back to the unfused pair.
- `gate_up.splitSwiGluDotPacked(ctx, &packed, split_tag, out_tag)` — fused
  split-SwiGLU activation + down-projection GEMM without materializing the
  gated tensor; kernels exist for `q8_0x4`/`q4_kx8`/`q5_kx8`/`q6_kx4`
  (`q4_kx2mmla` is a deliberate comptime error — callers guard with
  `comptime !fucina.supports_q4_k_mmla` and fall back to unfused).
- `gate.gegluQuantDotPacked(ctx, &up, &packed, in_tag, out_tag)` — fused
  GeGLU + down projection, `q8_0x4` only.

At the LLM layer (§13), `fucina_llm`'s `weights.zig` wraps each quantized
projection as a struct holding the original blocks plus a
`fucina.PackedRhs(dtype)` built once at load; this is the intended pattern
for any model code with reused weights — pack once, `dotPacked` per step.

```zig
test "packed RHS matmul matches the unpacked quantized dot" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const m = 3;
    const n = 8;
    const k = 2 * fucina.q8_0_block_size;

    var wsrc: [n * k]f32 = undefined;
    for (&wsrc, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 13 + 5) % 17)) * 0.1 - 0.8;
    var blocks: [n * 2]fucina.BlockQ8_0 = undefined;
    try fucina.gguf.encodeF32(.q8_0, &wsrc, std.mem.sliceAsBytes(&blocks));

    const W = fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });
    var w = try W.fromBlocks(&ctx, .{ n, k }, &blocks);
    defer w.deinit();
    var packed_rhs = try w.packRhs(&ctx); // fucina.PackedRhs(.q8_0)
    defer packed_rhs.deinit();

    var xv: [m * k]f32 = undefined;
    for (&xv, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7 + 3) % 11)) * 0.2 - 1.0;
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ m, k }, &xv);
    defer x.deinit();

    var fast = try x.dotPacked(&ctx, &packed_rhs, .in, .out);
    defer fast.deinit();
    var reference = try x.dot(&ctx, &w, .in);
    defer reference.deinit();
    for (try fast.dataConst(), try reference.dataConst()) |a, b|
        try std.testing.expectApproxEqAbs(b, a, 5e-2);
}
```

### 10.4 `RhsLifetime` and the address-keyed caching rule (`src/exec/quant_matmul.zig`)

The enum itself, the address-keyed cacheability rule, and the
pooled-storage prohibition are §6.7; this subsection covers how the flag
rides through quantized matmul.

```zig
pub const QuantizedMatmulOptions = struct {
    allow_gpu: bool = true,
    rhs_lifetime: RhsLifetime = .transient,
};
```

The option rides on the `ExecContext` raw-tensor entry points —
`matmul2DWithQuantizedTensorRhs` / `matmul2DWithQuantizedTensorRhsOptions`
and `matmul2DWithQuantizedBlocksRhs` /
`matmul2DWithQuantizedBlocksRhsOptions` (the blocks-slice variants accept
q8_0/q4_k/q5_k/q6_k only). The facade `dot` always uses the default
transient/`allow_gpu`-when-not-training options — a `.transient` RHS may
still use the provider's blocking GPU path, but no address-keyed wrap survives
the call and the borrowed bytes cannot be retained by an async command.
`fucina_llm`'s weight wrappers thread `.stable_process` through for resident
or mmap'd weights (§13); that lifetime first tries the direct-output async
path (Metal and CUDA support up to 8192 activation rows per dense-quant
submission) and falls back to balanced blocking chunks of at most 2048 rows
when necessary. Q4_K/Q6_K/Q8_0 prefill uses provider- and format-specific work
gates calibrated against the actual compact/raw or load-time-packed CPU
fallback. Decode `m <= 8` remains behind the provider's explicit GEMV opt-in.
The complete GPU contract is §9.

### 10.5 LHS activation quantization (`src/backend/quant/q8k.zig`, `src/backend/cpu.zig`, `src/backend/native.zig`)

Quantized matmuls quantize the f32 activations on the fly, per weight-format
family (all ggml-parity):

- **Q8_0** (`quantizeRowQ8_0Into`, `quantizeRowsQ8_0Into`,
  `quantizeRowsQ8_0`): per-32 symmetric absmax — `d = amax/127` stored as
  f16, `q = round(x/d)` clamped to i8. Consumed by q1_0/q4_0/q5_0/q8_0 and
  the table formats iq4_nl/mxfp4/nvfp4.
- **Q8_1** (`quantizeRowQ8_1Into`, `quantizeRowsQ8_1` — defined in
  `src/backend/quant/cold.zig`, re-exported through `quant.zig`): Q8_0 plus
  the per-block activation sum (`s = d·Σq` as f16) that the offset formats
  q4_1/q5_1 need to fold their block minimum into the dot.
- **Q8_K** (`quantizeRowQ8_KInto`, `quantizeRowsQ8_K`): per-256 with an f32
  scale derived from the signed maximum (`inv_scale = -127/max`) and
  per-16-element partial sums (`bsums`) — the K-quant and ternary kernels
  consume the bsums to fold block minima / the ternary `−Σa` term for free.
  Packed-LHS variants (`quantizeRowsQ8_Kx4Into`,
  `quantizeRowsQ8_Kx4PaddedInto`, `quantizeRowsQ8_Kx2MmlaInto`,
  `packRowsQ8_Kx4`) interleave 4 (or 2 for MMLA) rows for the x4/x2mmla
  kernels.

Scratch policy: only the **Q8_0/Q8_0x4** LHS buffer has a stack fast path —
a fixed 512-block array (`q8_0_lhs_stack_blocks`) covers GEMVs and small
batches with zero allocation, falling back to a context-allocator heap
allocation when `m × blocks_per_row` exceeds it. The Q8_K path (every
K-quant, IQ\*, and TQ\* weight) and the Q8_1 path heap-allocate their block
buffers unconditionally (`quantizeRowsQ8_K`/`quantizeRowsQ8_1` take the
allocator). The native backend's fused K-quant split-SwiGLU and GeGLU paths
(§10.3) lease buffers from the context's reusable scratch pool
(`buffers.acquireScratch`) — the q8_0x4 fused split-SwiGLU arm keeps the
stack-array-plus-alloc-fallback scheme instead — and parallelize row-group
quantization across the work pool once `m·k` reaches **one eighth** of
`parallel.vector_elementwise_len_threshold`. Failure mode: allocation
failure is the only runtime error; quantization itself is total for finite
input.

`ExecContext` also exposes the Q8_0 activation codec directly —
`quantizeF32RowsToQ8_0Into(x, dst_blocks)` and
`dequantizeQ8_0RowsInto(dst, blocks)` — used to maintain Q8_0 KV caches for
`groupedAttention`'s quantized-KV representation (§4).

### 10.6 Encoders, `gguf.encodeF32`, and ggml parity (`src/backend/quant.zig`, `src/gguf.zig`)

The kernel tier's encoder dispatch:

```zig
pub fn quantizeRowForDType(comptime tensor_dtype: DType, dst: []dtype_mod.Storage(tensor_dtype), src: []const f32) !void
pub fn dequantizeRowForDType(comptime tensor_dtype: DType, dst: []f32, blocks: []const dtype_mod.Storage(tensor_dtype)) !void
pub fn blockCountForDType(comptime tensor_dtype: DType, len: usize) !usize
```

`quantizeRowForDType` covers `.q4_0`, `.q4_1`, `.q5_0`, `.q5_1`, `.q8_0`,
`.q8_1`, `.q4_k`, `.q5_k`, `.q6_k`, `.q8_k`, `.tq2_0`; any other dtype is a
compile error. `src.len` must be a whole number of blocks and `dst.len` must
equal `blockCountForDType(dtype, src.len)` —
`QuantizedFormatError.InvalidQuantizedLength` otherwise. Inputs are assumed
**finite** (same contract as ggml's encoders; debug asserts mirror ggml's
`nearest_int` bound). The K-quant encoders are operation-for-operation ports
of `quantize_row_{q4,q5,q6}_K_ref`: the shared iterative scale-search
helpers `makeQxQuants` (symmetric, Q6_K) and `makeQkx2Quants` (asymmetric
scale+min grid search, Q4_K/Q5_K) plus `nearestInt` (round-to-nearest-even
via the 1.5·2²³ magic constant) and `group_max_eps` reproduce ggml's f32
arithmetic exactly. Per-block entries also exist
(`quantizeBlockQ4_KInto`/`Q5_K`/`Q6_K`, `dequantizeBlockQ4_KInto`/`Q5_K`/
`Q6_K`/`Q8_K`/`Q2_K`/`Q3_K`, `getScaleMinK4`).

The public seam is `fucina.gguf.encodeF32(ggml_type, src, dst)` (§12): it
validates `dst.len == tensorByteLen(...)`, rejects non-finite input on the
block formats with `error.NonFiniteValue` (release builds too — the same
guard llama.cpp applies at its quantize seam), requires `dst` to be aligned
for the block struct, and dispatches to `quantizeRowForDType`. Supported
block targets: q4_0, q4_1, q5_0, q5_1, q8_0, q4_k, q5_k, q6_k, tq2_0
(scalar f32/f16/bf16 cast element-wise); everything else returns
`error.EncoderUnavailable`. `gguf.decodeF32` is the exact mirror
(`error.DecoderUnavailable` for formats it does not cover).

Parity evidence, all in-tree and run by `zig build test`:

- `src/backend/quant/encode_golden_test.zig` — embedded goldens generated
  once by a C harness linking ggml's reference encoders over 8 adversarial
  input vectors (ramp, alternating, near-zero, wide-range, all-equal,
  denormals, random, zeros); the Zig encoders for Q4_K/Q5_K/Q6_K/Q4_1/Q5_0/
  Q5_1 match **byte-for-byte**, and the oracle was verified stable across
  three compiler/FP-contraction configurations.
- `src/backend/quant/cold_tests.zig` — embedded ggml-golden dequantize
  fixtures reproduced **bit-for-bit** for every cold decode format
  (Q2_K/Q3_K, all IQ*, TQ*, MXFP4/NVFP4), plus behavioral matmul tests for
  the table-dot paths.
- `src/backend/quant_tests.zig` and the per-format
  `quant/{q4_k,q5_k,q6_k,q8_0,common}_tests.zig` — hot-kernel vs scalar
  reference equivalence; `src/backend/quant/ternary_tests.zig` pins the
  ternary hot kernels **bitwise** against the cold scalar reference.

### 10.7 Ternary: TQ2_0 first-class, TQ1_0 decode-only (`src/backend/quant/ternary.zig`, [TERNARY.md](TERNARY.md))

TQ2_0 (BitNet b1.58: weights in {−1, 0, +1}, 2.0625 bits/weight — 256-element
blocks, 64 bytes of 2-bit crumbs storing `w+1`, inline f16 scale) is promoted
to a first-class format: encoders, tuned mul-free kernels on both the int8
and f32 activation paths, a facade training op, ternary-native ES (§11), and
GGUF export interop with llama.cpp. The contract dimension `k` must be a
multiple of 256 everywhere.

Encoders:

- `quantizeRowTQ2_0Into` — ggml `quantize_row_tq2_0_ref` parity: per-block
  absmax `d`, round-half-away(x/d). This is the **only** encoder behind the
  generic seams: `quantizeRowForDType(.tq2_0, ...)` dispatches to it, and
  `gguf.encodeF32(.tq2_0, ...)` therefore realizes per-block absmax too.
- `ternaryAbsmeanScale` + `quantizeRowTQ2_0ScaledInto` — the b1.58 recipe:
  per-tensor `d = max(mean|W|, 1e-5)`, `clamp(round(W/d), −1, +1)`; every
  block stores the same `d`, so the output is plain valid TQ2_0. This
  scaled encoder takes an explicit `d` and is *not* reachable through
  `quantizeRowForDType`/`gguf.encodeF32` — it is driven by
  `quantizedMatmulRhsTQ2_0FromF32Absmean` (the `dotTernarySte` forward,
  §4.9) and the ternary ES paths (§11).

Kernels: the int8 flagship exploits `dot(w, a) = Σ(w+1)·a − Σa` — the Q8_K
activation `bsums` supply `Σa`, so the hot loop is shift/mask plus int8
group dots with **no weight multiplications** (`matmulTQ2_0RhsTile`/
`matmulTQ2_0RhsRange`); all ISA arms accumulate the exact per-block integer
and are cross-ISA bitwise identical to the cold reference. The f32 path
(`dotTQ2_0F32`, `matmulTQ2_0F32RhsTile`/`Range`) is exact in IEEE f32 via
the sign-plane/zero-plane identity `w·x = (x XOR s) AND m` with a fixed
4-lane accumulation order — bitwise reproducible on every target, used for
STE training forwards. RHS constructors: `quantizedMatmulRhsTQ2_0FromBlocks`
(copies), `quantizedMatmulRhsTQ2_0FromBorrowedBlocks` (borrows, `deinit`
frees nothing), `quantizedMatmulRhsTQ2_0FromF32` (absmax) and
`quantizedMatmulRhsTQ2_0FromF32Absmean` (b1.58). Measured: ~4.2x the cold
path and ~2.1x Q4_K per byte-ratio on M1 Max; ~5.1x cold and ~4.8x Q4_K on
Raptor Lake AVX-VNNI ([TERNARY.md](TERNARY.md)).

As inference weights, `.tq2_0` tensors go through the ordinary facade:

```zig
test "TQ2_0 ternary weights are a first-class matmul RHS" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const k = 256;
    var wsrc: [2 * k]f32 = undefined;
    for (wsrc[0..k]) |*v| v.* = 0.5; // encodes exactly: d = 0.5, trit = +1
    for (wsrc[k..]) |*v| v.* = -0.5;
    var blocks: [2]fucina.BlockTQ2_0 = undefined;
    try fucina.gguf.encodeF32(.tq2_0, &wsrc, std.mem.sliceAsBytes(&blocks));

    const W = fucina.Tensor(.{ .dtype = .tq2_0, .tags = .{ .out, .in } });
    var w = try W.fromBlocks(&ctx, .{ 2, k }, &blocks);
    defer w.deinit();

    const x_values = [_]f32{1} ** k;
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 1, k }, &x_values);
    defer x.deinit();

    var y = try x.dot(&ctx, &w, .in); // mul-free int8 ternary kernel
    defer y.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 128), (try y.dataConst())[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -128), (try y.dataConst())[1], 1e-3);
}
```

#### `dotTernarySte` — the straight-through-estimator training op

```zig
pub fn dotTernarySte(self: *const Self, ctx: *ExecContext, weight: anytype,
                     comptime contract_tag: Tag) !Tensor(...)
```

Trainable ternary linear on the f32 facade tensor (`src/ag/tensor.zig`).
Every forward encodes the **f32 latent weight** (tags `{ .out, .in }`,
per-tensor absmean scale, round-clip to {−1, 0, +1}) to TQ2_0 and contracts
`self` (`[..., in]`) against it with the mul-free f32 kernel — the same
`@Vector` kernel on both backend kinds, so scalar and native builds share
bitwise-identical numerics. Backward: `dx` flows through the **quantized**
weight (dequantize-then-matmul); `dW` is the straight-through estimate — the
plain matmul VJP against the latent weight, identity through the quantizer,
no clipping or masking (exactly BitNet's `w + (Q(w) − w).detach()`). The
encoded blocks live in the backward node and are freed with it; the op works
under exec scopes.

Comptime requirements: f32 latent weight, storage order `[free, contract]`
on the weight and `[..., contract]` on the LHS, one weight free axis, no
shared batch tags. Runtime: `error.TernaryContractDimNotBlockAligned` when
`k` is 0 or not a multiple of 256. The latent weight is re-encoded every
forward (inherent to STE).

```zig
test "dotTernarySte trains a b1.58 ternary linear" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const k = 256; // contract dim must be a multiple of 256 (TQ2_0 blocks)
    var xv: [k]f32 = undefined;
    for (&xv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) * 0.1 - 0.2;
    var wv: [2 * k]f32 = undefined;
    for (&wv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 9)) * 0.1 - 0.4;

    var x = try fucina.Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 1, k }, &xv);
    defer x.deinit();
    var w = try fucina.Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, k }, &wv);
    defer w.deinit();

    var y = try x.dotTernarySte(&ctx, &w, .in); // encode-then-mul-free forward
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    // STE identity: dW is the plain matmul VJP; with gy = 1, each row of dW = x.
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    const gw_data = try gw.dataConst();
    for (0..k) |i| try std.testing.expectApproxEqAbs(xv[i], gw_data[i], 1e-6);
}
```

The gradient-free alternative is **ternary-native evolution strategies**
(§11): ES genomes that *are* packed `[]BlockTQ2_0` — no latent floats, so the
trained state is byte-for-byte the served state and every member evaluation
runs the real int8 inference kernels (`examples/es_ternary_spirals.zig` is
the end-to-end demo). `.tq1_0` (1.6875 bits/weight, base-3⁵ packing of five
trits per byte) remains decode/cold-matmul only.

### 10.8 Cold decode rules: IQ*, FP4, and friends (`src/backend/quant/cold.zig`, `src/backend/quant_tables.zig`)

`quant_tables.zig` holds the lookup tables generated from ggml's
`ggml-common.h` (`iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`,
`iq3s_grid`, `iq1s_grid`, `ksigns_iq2xs`/`kmask_iq2xs`, `kvalues_iq4nl`,
`kvalues_mxfp4`). Decode semantics, one line each:

- **Q1_0** — pure sign bits: bit set → `+d`, clear → `−d` (f16 per-block
  scale, 128 weights/block).
- **IQ2_XXS / IQ2_XS / IQ2_S** — 2.06–2.5 bpw: 8-element groups are indices
  into a shared 256/512/1024-entry codebook grid of magnitude patterns
  {8, 25, 43}, with separate sign words (`ksigns_iq2xs`) and 4-bit group
  scales times the f16 block scale.
- **IQ3_XXS / IQ3_S** — same grid-codebook construction at ~3 bpw with
  4-element grids (`iq3xxs_grid`/`iq3s_grid`) and explicit sign bits.
- **IQ1_S / IQ1_M** — 1.56/1.75 bpw: 11-bit indices into the 2048-entry
  8-element grid `iq1s_grid`, per-32 3-bit scales, and a whole-group
  ±0.125 delta added to every element (IQ1_M drops the f16 block scale and
  packs all scales into nibble fields).
- **IQ4_NL / IQ4_XS** — 4-bit **nonlinear** codebook: nibbles index the
  16-entry `kvalues_iq4nl` table (an asymmetric nonlinear ladder from −127
  to 113) times an f16 block scale; IQ4_XS adds per-32 sub-scales inside
  256-blocks.
- **MXFP4** — OCP MX: one shared **E8M0** exponent byte per 32 elements
  (decoded as a pure power of two, halved to compensate the
  integer-doubled table) times the 16-entry FP4 **E2M1** codebook
  `kvalues_mxfp4` = {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}.
- **NVFP4** — NVIDIA FP4: four 16-element sub-blocks per 64-element block,
  each with a **UE4M3** (unsigned e4m3) scale byte, same E2M1 codebook.
- **TQ1_0** — five trits packed per byte in base-3 (with a small `qh`
  tail), f16 scale.

All of these dequantize (`to(.f32)`, `getRows`) and matmul through the
generic cold dot path with Q8_0- or Q8_K-quantized activations per the
§10.1 table; none has an encoder — they enter Fucina only as loaded GGUF
weights (§12) and their decode output is pinned bit-for-bit to ggml goldens
(`cold_tests.zig`).

### 10.9 PTQTP: multi-plane ternary decomposition (`src/ptqtp.zig`, [PTQTP.md](PTQTP.md))

`fucina.ptqtp` decomposes a dense f32 `[n][k]` weight matrix into
K ∈ {1, 2, 3} ternary planes, `W ≈ Σₖ diag(αₖ)Tₖ` with `Tₖ ∈ {−1, 0, +1}`
and one scale per plane per 256-column group — data-free post-training
quantization (arXiv:2509.16989; no calibration inputs, no gradients). Each
group solves independently by alternating a closed-form K×K ridge
regression for the scales with an exhaustive 3ᴷ-way per-element trit
search, in a pinned candidate order that makes the whole solve bitwise
reproducible for any thread count. The group size is the TQ2_0 block
width, so **each plane is a standalone byte-valid `.tq2_0` tensor** whose
per-block f16 `d` is the group scale: decorated inference is K stock
ternary matmuls (§10.7) plus adds — no new kernels. `k % 256 == 0` is
required.

```zig
pub const Options = struct {
    planes: u8 = 2,            // K; 3 adds a residual plane (+2.06 bpw)
    group_size: usize = 256,   // quantizeMatrix requires 256 (the packable size)
    max_iterations: usize = 50, epsilon: f32 = 1e-4,
    lambda0: f32 = 1e-8, lambda_max: f32 = 1.0, kappa_max: f32 = 1e6,
};
pub fn quantizeMatrix(ctx: *ExecContext, weights: []const f32, n: usize, k: usize,
                      options: Options) !PlanePair
pub fn solveGroup(w: []const f32, t1: []i8, t2: []i8, t3: []i8,
                  options: Options) GroupResult   // pure, allocation-free
pub fn reconstructReference(allocator, weights, n, k, options, dst: []f32) !MatrixStats
```

`PlanePair` owns up to three `[]BlockTQ2_0` planes (`plane2`/`plane3`
empty below the built count) plus `stats: MatrixStats` (rel Frobenius
error of the served reconstruction — fp16-rounded scales — per-plane zero
fractions, iteration/convergence counts). `rhs(plane)` returns a borrowed
backend matmul view; `reconstructInto(dst)` writes the dequantized plane
sum; `deinit(allocator)` frees the planes. `reconstructReference` solves
at arbitrary group sizes with f32 scales (unpacked) for fidelity studies.
Non-finite weights are excluded from the regression and forced to zero
trits in every plane. K = 1 is a least-squares upgrade over the absmean
b1.58 encoder (§10.7); planes are separable, so serving fewer planes than
were solved is valid.

```zig
test "PTQTP: planes reconstruct and multiply as plain TQ2_0 tensors" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const n = 2;
    const k = 256;
    var w: [n * k]f32 = undefined; // any f32 [n][k] with k % 256 == 0
    for (&w, 0..) |*v, i| v.* = 0.02 * @sin(@as(f32, @floatFromInt(i)));

    var pair = try fucina.ptqtp.quantizeMatrix(&ctx, &w, n, k, .{}); // K = 2
    defer pair.deinit(ctx.allocator);
    try std.testing.expect(pair.stats.rel_frob_err < 0.25); // 9-level regime

    // Each plane is a byte-valid TQ2_0 tensor; the decorated product is
    // the sum of per-plane products through the stock ternary kernels.
    const W = fucina.Tensor(.{ .dtype = .tq2_0, .tags = .{ .out, .in } });
    var p1 = try W.fromBlocks(&ctx, .{ n, k }, pair.plane1);
    defer p1.deinit();
    var p2 = try W.fromBlocks(&ctx, .{ n, k }, pair.plane2);
    defer p2.deinit();

    const ones = [_]f32{1} ** k;
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 1, k }, &ones);
    defer x.deinit();
    var y1 = try x.dot(&ctx, &p1, .in);
    defer y1.deinit();
    var y2 = try x.dot(&ctx, &p2, .in);
    defer y2.deinit();
    var y = try y1.add(&ctx, &y2);
    defer y.deinit();

    var rec: [n * k]f32 = undefined; // the exact Ŵ the stats measured
    try pair.reconstructInto(&rec);
    for (try y.dataConst(), 0..) |yi, r| {
        var want: f32 = 0;
        for (rec[r * k ..][0..k]) |v| want += v;
        try std.testing.expectApproxEqAbs(want, yi, 1e-2);
    }
}
```

At the LLM layer, `LinearWeight.toPtqtp` decorates a loaded GGUF linear in
place from any source dtype and `llm.qwen3.model.Model.decoratePtqtp`
walks a whole model (§13.2.1); `zig build ptqtp-spirals` and
`zig build ptqtp-qwen3` are the acceptance/measurement examples. Decorated
models persist to GGUF as per-plane standalone TQ2_0 tensors and load back
bitwise through plane pair-detection in the qwen3 loaders
(`llm.ptqtp_gguf`, §13.2.1), so the solve runs once per model. Measured
accuracy/speed tables and
configuration guidance (plane counts, source precision, lm_head economics
per ISA): [PTQTP.md](PTQTP.md).

## 11. Training: optimizers, evolution strategies, LoRA, and checkpoints

Training on the eager runtime is five modules, all reachable from the root
export: `fucina.optim` (gradient-descent optimizers, param groups, LR
schedules, clipping, optimizer-state persistence), `fucina.es`
(gradient-free evolution strategies), `fucina.lora` (LoRA adapters over
frozen linears), `fucina.ParamRegistry` + `fucina.state_dict` (named
parameter collection and safetensors state dicts), and
`fucina.training_checkpoint` (the resumable checkpoint directory layout).
The contract document is [TRAINING.md](TRAINING.md); everything below is
exercised end-to-end by `zig build spirals`, `zig build es-spirals`,
`zig build finetune`, and `zig build es-finetune` (`examples/`). Autograd
semantics are §5; the exec-scope memory model the training loop leans on is
§6. Snippets in this section assume
`const std = @import("std"); const fucina = @import("fucina"); const optim = fucina.optim;`.

### 11.1 The shape of a training step

Training differs from inference in exactly one rule: every tensor on the
path from the parameters to the loss must stay alive until `backward()`
returns (§5 — `GradState` is single-owner, and consumers hold raw pointers
into the graph). Exec scopes (§6.3) make that rule implicit: open a scope
around the step, write the forward in the ordinary deinit-ASAP style (deinit
on scope-owned results is a safe no-op), and close the scope after the
optimizer step. The canonical step order is `backward` → `clipGradNorm` →
`step` → `zeroGrad`:

```zig
test "one training step: forward, backward, clip, step, zero" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var w = try fucina.Tensor(.{ .class, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.1, -0.2, 0.3, 0.4 });
    defer w.deinit();
    var b = try fucina.Tensor(.{.class}).variableFromSlice(&ctx, .{2}, &.{ 0, 0 });
    defer b.deinit();
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, -1, 0.5 });
    defer x.deinit();
    const labels = [_]usize{ 0, 1 };

    var opt = optim.AdamW.init(alloc, .{ .lr = 0.05, .weight_decay = 0.01 });
    defer opt.deinit();
    try opt.addParam(&w); // params must outlive the optimizer
    try opt.addParam(&b);

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..20) |i| {
        const scope = ctx.openExecScope(); // the scope owns the step's graph
        defer ctx.closeExecScope(scope);
        const z = try x.dot(&ctx, &w, .in);
        const logits = try z.add(&ctx, &b);
        const loss = try logits.crossEntropy(&ctx, .class, &labels);
        try loss.backward(&ctx);
        _ = try opt.clipGradNorm(&ctx, 1.0); // after backward, before step
        try opt.step(&ctx);
        opt.zeroGrad();
        if (i == 0) first = try loss.item();
        last = try loss.item();
    }
    try std.testing.expect(last < first);
}
```

Gradient accumulation needs no extra machinery: `backward()` ADDS into each
parameter's persisted gradient, leaf gradients live outside exec scopes,
`step()` reads them non-destructively, and `zeroGrad()` is the only clear.
N micro-batches + one `step()` is the accumulation recipe; scale the LOSS
(not the gradients) by the window normalizer, clip once after the window,
key LR schedules by the macro step, and checkpoint only at window
boundaries (accumulated gradients are never serialized). The full recipe,
its normalization arms, and its determinism contract are
[TRAINING.md](TRAINING.md) §4.

### 11.2 Optimizers (`src/optim.zig`)

Each optimizer is a faithful port of a reference implementation, pinned by
golden parity tests against the actual references (PyTorch 2.12, Keller
Jordan's muon.py, apollo_torch — `src/optim_tests.zig`):

| Type | Config | Reference | State per param element |
|---|---|---|---|
| `optim.SGD` | `SgdConfig` | `torch.optim.SGD` (single-tensor) | 0 B; 4 B with momentum (2 B bf16) |
| `optim.Adam` | `AdamConfig` | `torch.optim.Adam` (coupled L2 decay) | 8 B m+v (down to 4 B bf16) |
| `optim.AdamW` | `AdamWConfig` | `torch.optim.AdamW` (decoupled decay) | 8 B m+v (down to 4 B bf16) |
| `optim.Muon` | `MuonConfig` | Keller Jordan reference + Moonlight scale | 4 B momentum (2 B bf16) + embedded AdamW fallback |
| `optim.Apollo` | `ApolloConfig` | apollo_torch (arXiv 2412.05270) | ~`8·rank·max(dim)` B moments + `4·rank·min(dim)` B resident projection per matrix (always f32) |

All five share one surface (`Muon`/`Apollo` add the fallback registrars):

```zig
pub fn init(allocator: Allocator, config: Config) Self      // SGD panics here on bad nesterov
pub fn deinit(self: *Self) void                             // frees slots + state; params stay caller-owned
pub fn addParam(self: *Self, t: anytype) !void              // t: pointer to an f32/f16/bf16 autograd variable
pub fn addParamNamed(self: *Self, t: anytype, name: []const u8) !void
pub fn addFallbackParam(self: *Self, t: anytype) !void      // Muon/Apollo only
pub fn addFallbackParamNamed(self: *Self, t: anytype, name: []const u8) !void
pub fn step(self: *Self, ctx: *ExecContext) !void
pub fn zeroGrad(self: *Self) void
pub fn clipGradNorm(self: *Self, ctx: *ExecContext, max_norm: f32) !f32
pub fn gradSquaredNorm(self: *Self, ctx: *ExecContext) !f64
pub fn scaleGradients(self: *Self, ctx: *ExecContext, factor: f32) !void
pub fn saveState(self: *const Self, writer: *std.Io.Writer) !void
pub fn loadState(self: *Self, reader: *std.Io.Reader) !void
pub fn collectGradStates(self: *const Self, set: *GradStateSet, allocator: Allocator) !void // OptimizerSet plumbing (private set type)
```

**Ownership.** `addParam` goes through `optim.Param.of`: it retains a
refcounted view of the variable's storage plus the raw `*GradState` pointer,
so the facade struct may move by value but the parameter must OUTLIVE the
optimizer (the tensor owns the GradState).

**16-bit params and f32 master weights.** `addParam` also accepts f16/bf16
variables (§5.1: their gradients are f32). Each 16-bit slot allocates an
optimizer-owned f32 MASTER copy at registration (widened once from the
param values): every update kernel steps the master — so the step math is
identical to the f32 path, and updates below 16-bit resolution accumulate
instead of rounding away — and the master is narrowed back into the 16-bit
storage after each step. Ordering contract: load parameter values BEFORE
registering the param (or restore via `loadState`, whose v5 frames carry
the master); a value load after registration leaves the master stale.

```zig
test "bf16 params train through f32 masters" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const W = fucina.Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } });
    var w = try W.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0x3f80, 0xc000, 0x3f00, 0x4040 }); // 1, -2, 0.5, 3
    defer w.deinit();
    var x = try fucina.Tensor(.{ .t, .in }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1, 2 });
    defer x.deinit();

    var opt = fucina.optim.AdamW.init(alloc, .{ .lr = 0.05 });
    defer opt.deinit();
    try opt.addParam(&w); // allocates + fills the f32 master

    const scope = ctx.openExecScope();
    var y = try x.dot(&ctx, &w, .in); // native mixed GEMM; dW arrives as f32
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    ctx.closeExecScope(scope);

    try opt.step(&ctx); // steps the master, narrows back into w's storage
    opt.zeroGrad();
    try std.testing.expect(w.requiresGrad());
    try std.testing.expect((try w.grad(&ctx)) == null); // cleared
}
```

Errors from `addParam`: `error.NotAVariable`
(constant or grad-free tensor), `error.NonContiguousParam`,
`error.DuplicateParam` (same variable twice in ONE instance — registering it
with two *different* instances is undetectable per-instance and silently
double-steps; `OptimizerSet.add` closes that gap, §11.3). Names passed to
`addParamNamed` are BORROWED and must outlive the optimizer (string
literals and model-struct fields qualify). Optimizer state (moments,
momentum, projections) is optimizer-owned and freed by `deinit`.

**`optim.Param`** is the type-erased slot handle (`value` — an
f32/f16/bf16 storage union, `grad_state`, `rows`, `cols`, `raw_rank`,
optional `name`, the f32 `master` for 16-bit slots; `pub fn of(t) !Param`,
`pub fn len(self) usize`). `rows`/`cols` describe the matrix view the
matrix-aware optimizers use: dim 0 by the product of the remaining dims —
Keller's conv-filter flattening `[d0, d1*d2*...]`.

**`step` semantics.** Parameters whose accumulated gradient is null are
skipped (PyTorch behavior); a gradient whose element count disagrees with
the parameter is `error.GradShapeMismatch`. Elementwise updates are fused
single-pass, element-independent maps chunked over the worker pool at
2^17 elements and above — bitwise identical to the serial loop for any thread count
(reductions such as norms run over a fixed chunk grid with a pinned
combine order, equally thread-count-invariant). Scalar prep
(bias corrections, step sizes) runs in f64 and rounds once to f32, matching
torch's Python-float scalars to within a few f32 ulps.

**`optim.StateDType`** (`enum(u8) { f32 = 0, bf16 = 1 }`; the values are v4
checkpoint wire tags — never renumbered). Every elementwise optimizer can
store its moment/momentum buffers in bf16 via `state_dtype` (and, for
Adam/AdamW, the separate `second_moment_dtype`): step math stays f32
(widen-on-read, NaN-guarded round-to-nearest-even narrow-on-write), updates
stay element-independent, and since the step is memory-bound the narrower
state is measurably *faster* (bench: `zig build bench-optim`). First
moments/momentum tolerate bf16 well; AdamW/Adam `v` is precision-sensitive
(with beta2 = 0.999 its ~0.1 %/step change sits below bf16's ~0.39 %
resolution, so the EMA can stall) — hence the separate opt-in. The f32
default keeps every existing checkpoint byte-identical.

#### Config structs and defaults

```zig
pub const SgdConfig = struct {
    lr: f32 = 1e-3,
    momentum: f32 = 0,          // 0 = no momentum buffer at all
    dampening: f32 = 0,
    weight_decay: f32 = 0,      // COUPLED L2 (g += wd*p), PyTorch SGD
    nesterov: bool = false,     // requires momentum > 0, dampening == 0
    state_dtype: StateDType = .f32,
};
pub const AdamConfig  = struct { lr: f32 = 1e-3, beta1: f32 = 0.9, beta2: f32 = 0.999,
    eps: f32 = 1e-8, weight_decay: f32 = 0,     // coupled decay
    state_dtype: StateDType = .f32, second_moment_dtype: StateDType = .f32 };
pub const AdamWConfig = struct { lr: f32 = 1e-3, beta1: f32 = 0.9, beta2: f32 = 0.999,
    eps: f32 = 1e-8, weight_decay: f32 = 0.01,  // decoupled, applied BEFORE the step
    state_dtype: StateDType = .f32, second_moment_dtype: StateDType = .f32 };
```

`SGD.init` **panics** (in every build mode, deliberately not a debug assert)
when `nesterov` is set with zero momentum or nonzero dampening — the PyTorch
constructor rule. With momentum, the buffer is initialized on the first step
to a clone of the first (decayed) gradient, not zeros (`buf = d_p.clone()`);
under bf16 state that clone is stored narrowed. `AdamW` applies decay to the
parameter before the moment update and adds `eps` after dividing `sqrt(v)`
by the bias correction — the exact `_single_tensor_adam` order; `Adam`
differs only in coupling the decay into the gradient.

```zig
pub const MuonScale = enum { spectral, match_rms_adamw };
pub const MuonConfig = struct {
    lr: f32 = 0.02, momentum: f32 = 0.95, nesterov: bool = true,
    ns_steps: u32 = 5, weight_decay: f32 = 0,
    scale: MuonScale = .spectral,
    state_dtype: StateDType = .f32,
    fallback: AdamWConfig = .{ .lr = 3e-4, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-10, .weight_decay = 0 },
};
```

Muon runs lerp-form momentum, Newton-Schulz-5 orthogonalization (f32; the
reference uses bf16 on GPU), and a shape-dependent scale: `.spectral` is
Keller's `sqrt(max(1, rows/cols))`, `.match_rms_adamw` is Moonlight's
`0.2*sqrt(max(rows, cols))` (reuse AdamW-tuned lr/wd). Routing: `addParam`
sends rank ≥ 2 params to the Muon path and auto-routes 0D/1D params (biases,
norms) to the embedded AdamW fallback; embeddings and output/classifier
heads are 2D but must NOT be orthogonalized — route them explicitly with
`addFallbackParam`/`addFallbackParamNamed`. Newton-Schulz transients come
from the ExecContext `BufferPool`. The iteration itself is public:

```zig
pub fn newtonSchulz5(ctx: *ExecContext, u: *const RawTensor, steps: u32) !RawTensor
```

Frobenius-normalize, then iterate the tuned quintic; when rows > cols the
iteration runs on the transpose so the Gram matrix has the small dimension.
The result approximates `U·V^T` with singular values in roughly (0.5, 1.5) —
by design. `u` must be rank-2 and is never mutated; the caller owns the
result.

```zig
pub const ApolloScaleType = enum { channel, tensor };
pub const ApolloConfig = struct {
    lr: f32 = 0.01, beta1: f32 = 0.9, beta2: f32 = 0.999,
    eps: f32 = 1e-6,                 // legacy-HF default; HF Trainer overrides to 1e-8
    weight_decay: f32 = 0,
    rank: usize = 128, update_proj_gap: u64 = 200,
    scale: f32 = 1.0, scale_type: ApolloScaleType = .channel,
    correct_bias: bool = true, scale_front: bool = false,
    disable_norm_growth_limiter: bool = false,
    seed: u64 = 0,
    pub fn mini() ApolloConfig      // rank 1, .tensor scaling, scale 128 (APOLLO-Mini)
};
```

**APOLLO rank/projection specifics.** Only rank-2 params take the low-rank
path (`addParam` auto-routes everything else to the fallback; use
`addFallbackParam` for embeddings/heads). Per matrix: the gradient is
projected into a rank-`rank` space (tall params `rows ≥ cols` project the
column space with P `(rank, cols)`, compressed grad `R = G·P^T` of shape
`(rows, rank)`; wide params project the row space, `R = P^T·G` of shape
`(rank, cols)`), AdamW moments run in the compressed space, channel- or
tensor-wise scaling factors are computed from the un-bias-corrected
`m/(sqrt(v)+eps)` vs the raw `R` (f64 accumulators, `+1e-8` division guard),
the full-size update is the raw gradient rescaled per channel, the Fira
norm-growth limiter clips per-step `||U||_F` growth at gamma 1.01 (its norm
reduction is the deterministic fixed-chunk one), and the final step is scaled SGD with
decoupled decay applied AFTER the step at the raw lr — the reference's
legacy-HF order, deliberately different from `AdamW`. The fallback path is
likewise the legacy-HF AdamW (`denom = sqrt(v) + eps`, bias correction
folded into the scalar, decay after the step).

**APOLLO RNG contract.** Projections are REGENERATED, never stored: P is a
deterministic function of `(slot seed, step / update_proj_gap)` through the
repo-owned `rng.gaussianFill` (splitmix64 + Box-Muller, §"RNG" in
[TRAINING.md](TRAINING.md) §6) with i.i.d. `N(0, 1/rank)` entries; the
per-param seed is `config.seed +% (1-based rank-slot index)`. The
(seed → values) mapping is a checkpoint contract — deliberately not
`std.Random`, so it survives toolchain upgrades. Regeneration uses the
pre-increment step counter (chunks `[0,T), [T,2T), ...`); moments are NOT
reset on regeneration; `loadState` restores the stored per-slot seed and
forces regeneration on the next step. Note the APOLLO recipes train with
global clipping disabled — the norm-growth limiter replaces it
(`clipGradNorm` is still provided for mixed setups).

### 11.3 Param groups: `OptimizerSet` (`src/optim.zig`)

A param group is exactly {hyperparams, params, state} — one optimizer
instance. `optim.OptimizerSet` makes N instances feel like one optimizer,
type-erased through `optim.AnyOptimizer`:

```zig
pub const AnyOptimizer = struct {
    ptr: *anyopaque, vtable: *const VTable,
    pub const VTable = struct { step, zeroGrad, gradSquaredNorm, scaleGradients, saveState, loadState };
    // pub fn step / zeroGrad / gradSquaredNorm / scaleGradients / saveState / loadState
};
pub fn anyOptimizer(opt: anytype) AnyOptimizer   // wrap a *SGD / *Adam / *AdamW / *Muon / *Apollo

pub const OptimizerSet = struct {
    pub fn init(allocator: Allocator) OptimizerSet
    pub fn deinit(self: *OptimizerSet) void       // frees only the set; members stay caller-owned
    pub fn add(self: *OptimizerSet, opt: anytype) !void
    pub fn step(self: *OptimizerSet, ctx: *ExecContext) !void
    pub fn zeroGrad(self: *OptimizerSet) void
    pub fn gradSquaredNorm(self: *OptimizerSet, ctx: *ExecContext) !f64
    pub fn scaleGradients(self: *OptimizerSet, ctx: *ExecContext, factor: f32) !void
    pub fn clipGradNorm(self: *OptimizerSet, ctx: *ExecContext, max_norm: f32) !f32 // GLOBAL norm
    pub fn saveState(self: *const OptimizerSet, writer: *std.Io.Writer) !void       // FZO3 frame
    pub fn loadState(self: *OptimizerSet, reader: *std.Io.Reader) !void
};
```

Members are BORROWED via raw pointer (they must not move or be freed while
the set is in use). `add` calls the member's `collectGradStates` to check
every parameter against all previously-added members: registering the same
variable into two groups returns `error.DuplicateParam` and leaves the set
unchanged — the cross-instance double-step the per-optimizer guard cannot
see. Mixing optimizer types in one set works (Muon for the trunk, AdamW for
an adapter). `clipGradNorm` is global across all groups — the
`clip_grad_norm_(model.parameters())` contract. `loadState` checks the
member count (`error.CheckpointShapeMismatch`) and loads members in order,
transactionally per member.

### 11.4 Gradient clipping and LR schedules (`src/optim.zig`)

**Clipping** is `torch.nn.utils.clip_grad_norm_` semantics: compute
`total = sqrt(sum ||g||^2)` over every registered param (deterministic
fixed-chunk f64 reduction); if `total > max_norm`, scale every gradient by
`max_norm / (total + 1e-6)`; return the PRE-clip norm. Call after
`backward()`, before `step()`. `scaleGradients(ctx, factor)` is the raw
primitive (also the grad-side accumulation normalizer); `gradSquaredNorm`
exposes the sum of squares for custom policies. On `Muon`/`Apollo` the norm
spans the matrix path AND the fallback.

**Schedules** are one small hook plus pure factor functions:

```zig
pub const LrSchedule = struct {
    pub fn init(allocator: Allocator) LrSchedule
    pub fn deinit(self: *LrSchedule) void
    pub fn attach(self: *LrSchedule, lr: *f32) !void   // captures *lr as the base
    pub fn apply(self: *const LrSchedule, factor: f64) void // lr = base * factor
};
pub fn warmupCosineFactor(step: u64, total_steps: u64, warmup_steps: u64, min_factor: f64) f64
```

`attach` points at the public `config.lr` field of any optimizer (and of a
Muon fallback: `&muon.fallback.config.lr`) and captures the current value as
the base — attach before the first `apply`, or the in-effect factor gets
baked into the base; re-attaching the same pointer refreshes its base
instead of duplicating the entry. The pointee must outlive the schedule.
Because the factor is a pure function of the step, resuming from a
checkpoint just re-applies it — this is why lr is deliberately NOT a
validated field of optimizer checkpoints. `warmupCosineFactor` is linear
warmup from `1/warmup_steps` to 1 over `warmup_steps` (0-based `step`), then
cosine decay to `min_factor` over the remaining steps:

```zig
test "warmupCosineFactor endpoints" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), optim.warmupCosineFactor(0, 100, 10, 0.1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), optim.warmupCosineFactor(9, 100, 10, 0.1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), optim.warmupCosineFactor(100, 100, 10, 0.1), 1e-9);
}
```

Groups, schedule, and clipping compose — the standard LLM recipe in
miniature (`examples/spirals.zig` `groupsDemo` proves this composition
resumes bit-exactly):

```zig
test "param groups under one OptimizerSet with a warmup-cosine schedule" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var w = try fucina.Tensor(.{ .class, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.1, -0.2, 0.3, 0.4 });
    defer w.deinit();
    var b = try fucina.Tensor(.{.class}).variableFromSlice(&ctx, .{2}, &.{ 0, 0 });
    defer b.deinit();
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, -1, 0.5 });
    defer x.deinit();
    const labels = [_]usize{ 0, 1 };

    var decay = optim.AdamW.init(alloc, .{ .lr = 2e-2, .weight_decay = 0.1 });
    defer decay.deinit();
    var no_decay = optim.AdamW.init(alloc, .{ .lr = 2e-2, .weight_decay = 0 });
    defer no_decay.deinit();
    try decay.addParam(&w); // matrices: decayed group
    try no_decay.addParam(&b); // biases/norms: no-decay group

    var set = optim.OptimizerSet.init(alloc);
    defer set.deinit();
    try set.add(&decay);
    try set.add(&no_decay);

    var sched = optim.LrSchedule.init(alloc);
    defer sched.deinit();
    try sched.attach(&decay.config.lr); // captures 2e-2 as the base
    try sched.attach(&no_decay.config.lr);

    for (0..4) |step_i| {
        sched.apply(optim.warmupCosineFactor(step_i, 100, 10, 0.1));
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const logits = try (try x.dot(&ctx, &w, .in)).add(&ctx, &b);
        const loss = try logits.crossEntropy(&ctx, .class, &labels);
        try loss.backward(&ctx);
        _ = try set.clipGradNorm(&ctx, 1.0); // GLOBAL norm across both groups
        try set.step(&ctx);
        set.zeroGrad();
    }
    // Linear warmup: factor at step 3 of a 10-step warmup is 4/10.
    try std.testing.expectApproxEqAbs(@as(f32, 2e-2 * 0.4), decay.config.lr, 1e-9);
}
```

### 11.5 Optimizer-state persistence: FZT1 snapshots vs named state dicts (`src/optim.zig`)

Parameter values have two formats; optimizer internals a third.

**Positional FZT1 (legacy).** f32-only, order-based (stream magic `FZT1`;
layout in §12.7):

```zig
pub fn saveTensors(writer: *std.Io.Writer, tensors: anytype) !void
pub fn loadTensors(allocator: Allocator, reader: *std.Io.Reader, tensors: anytype) !void
```

`tensors` is a tuple of pointers to contiguous f32 facade tensors (variables
or constants). The loading program must list the same tensors in the same
order; shapes are validated (`error.CheckpointShapeMismatch`), and the load
is transactional — the whole stream is staged and validated before any
tensor is written, so a truncated or mismatched stream leaves every
destination byte-unchanged. Use it only for closed-world snapshots where the
saving and loading code are the same program; new code should prefer the
named form.

**Named, dtype-aware state dicts.** Re-exported from `fucina.state_dict`
(§11.7) for convenience: `optim.NamedTensor`, `optim.NamedTensorMut`,
`optim.LoadOptions`, `optim.saveStateDict`, `optim.loadStateDict`. Entries
carry a unique name, dtype (f32/f16/bf16, raw byte passthrough), shape, and
bytes; the wire format is a valid safetensors file; the load matches stream
entries BY NAME so entry order is free. This is the portable format — it is
what `model.safetensors`/`adapters.safetensors` in a checkpoint directory
contain, and any safetensors consumer can read it.

**Optimizer internals** — moments, momentum, per-slot step counters, APOLLO
seeds/limiter memory — serialize through each optimizer's
`saveState`/`loadState` (and `OptimizerSet`'s, which concatenates member
frames under `FZO3`). Frame magics:

| Optimizer | all-f32 state (v3) | any bf16 state (v4) | any 16-bit param (v5) |
|---|---|---|---|
| Adam | `FZAD` | `FZD4` | `FZD5` |
| AdamW | `FZA3` | `FZA4` | `FZA5` |
| Muon | `FZM3` (fallback frames follow) | `FZM4` | `FZM5` |
| SGD | `FZS3` | `FZS4` | `FZS5` |
| APOLLO | `FZP3` (state always f32) | — | `FZP5` |

v5 frames additionally persist each 16-bit slot's f32 MASTER weights
(per-slot presence flag + raw f32 bytes): resuming from the narrowed
values instead would re-round the master and lose the sub-16-bit update
accumulation. `loadState` installs the checkpoint master and narrows it
into the param storage; when a v3/v4 frame (or a v5 slot without a master)
loads into a 16-bit slot, the master re-widens from the current param
values instead.

Writers emit v3 whenever every state buffer is f32 — byte-identical to
pre-`StateDType` builds, so older builds keep reading new f32 checkpoints —
and v4 otherwise (identical layout except each state buffer is prefixed by
one u8 `StateDType` tag). Loaders accept both and require the stored dtype
to match the configured one EXACTLY (`error.CheckpointDtypeMismatch`; v3
implies f32 everywhere; an unknown tag is
`error.CheckpointUnsupportedDtype`). There is deliberately no implicit
f32↔bf16 conversion: it would silently break bit-exact resume.

Load-time validation: 4-byte magic (`error.CheckpointMagicMismatch`),
structural config fields (`error.CheckpointConfigMismatch` — Muon validates
scale/nesterov/ns_steps; SGD momentum/dampening/nesterov; Apollo
rank/update_proj_gap/scale/scale_type/correct_bias/scale_front/limiter
flag; lr is deliberately NOT validated — schedules legitimately change it),
per-slot dims (`error.CheckpointShapeMismatch`). Slots are matched BY NAME:
the explicit `addParamNamed` name, else the auto-name `"param<i>"` from the
slot's index within its slot list (Muon/Apollo fallback lists number
independently). Named params may therefore re-register in ANY order within
their list; unnamed params must keep their **absolute slot indices** to
reproduce their auto-names — the auto-name is `"param<i>"` from the slot's
position in the whole list, so reordering *named* entries around an unnamed
one also breaks it (`error.CheckpointUnknownName` on load). Name errors:
`error.CheckpointInvalidName`
(empty/too long/NUL/invalid UTF-8), `error.CheckpointDuplicateName` (also
raised at SAVE time when an explicit name collides with an auto-name),
`error.CheckpointUnknownName` (stream record matches no slot),
`error.CheckpointMissingEntry` (a slot left unfilled).

Every `loadState`/`loadTensors` is transactional per optimizer instance:
records decode into scratch and the whole stream validates before the first
live byte is written, so a bad stream is a no-op (`Muon` commits its own
slots only after its embedded fallback loads, keeping the pair atomic;
`OptimizerSet` is transactional per member — treat a failed set load as
fatal for the whole run). **Bit-exact resume**: state restore is byte-exact
and updates are thread-count-invariant, so resume replays bit-exactly as
long as the surrounding forward/backward is deterministic (the one caveat:
a tensor fed by 3+ heavy async backward branches accumulates in completion
order — [TRAINING.md](TRAINING.md) §8). `zig build spirals` asserts
bit-identical final parameters after a halfway-checkpoint resume for SGD,
AdamW, Muon, APOLLO, and APOLLO-Mini, and for the groups+schedule+clip
combo (plain `Adam`'s FZAD/FZD4 resume path is not covered by that gate).

```zig
test "optimizer state: name-matched slots round-trip; structural config is validated" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();
    var w = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer w.deinit();
    var b = try fucina.Tensor(.{.e}).variableFromSlice(&ctx, .{2}, &.{ 5, 6 });
    defer b.deinit();

    var opt = optim.SGD.init(alloc, .{ .lr = 0.1, .momentum = 0.9 });
    defer opt.deinit();
    try opt.addParamNamed(&w, "w");
    try opt.addParamNamed(&b, "b");
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try w.sumAll(&ctx);
        try loss.backward(&ctx);
        try opt.step(&ctx); // populates the momentum buffer
        opt.zeroGrad();
    }
    var buf: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try opt.saveState(&writer);

    // Resume: named slots may re-register in any order.
    var opt2 = optim.SGD.init(alloc, .{ .lr = 0.1, .momentum = 0.9 });
    defer opt2.deinit();
    try opt2.addParamNamed(&b, "b");
    try opt2.addParamNamed(&w, "w");
    var reader = std.Io.Reader.fixed(writer.buffered());
    try opt2.loadState(&reader);

    // Structural config fields must match the stored ones (lr is not one).
    var opt3 = optim.SGD.init(alloc, .{ .lr = 0.1, .momentum = 0.5 });
    defer opt3.deinit();
    var reader3 = std.Io.Reader.fixed(writer.buffered());
    try std.testing.expectError(error.CheckpointConfigMismatch, opt3.loadState(&reader3));
}
```

**When to use which.** FZT1 `saveTensors` for quick same-program f32
snapshots; named state dicts for anything that must survive refactoring,
mixed dtypes, or foreign consumers (they are plain safetensors);
`saveState`/`loadState` alongside the state dict whenever training must
RESUME (moments and step counts are not reconstructible). The complete
error set is `optim.OptimError`.

### 11.6 `ParamRegistry` (`src/param_registry.zig`)

`fucina.ParamRegistry` is the named-parameter seam between models,
checkpoints, and trainers. It owns no model tensors: it BORROWS named
f32/f16/bf16 facade tensors, retaining refcounted storage views
(dtype-erased), so the original tensors and their GradStates must outlive
the registry and any optimizer it registers into. Names are COPIED
(registry-owned).

```zig
pub const ParamRegistry = struct {
    pub fn init(allocator: Allocator) ParamRegistry
    pub fn deinit(self: *ParamRegistry) void
    pub fn addParam(self: *ParamRegistry, name: []const u8, t: anytype) !void
    pub fn collect(self: *ParamRegistry, model: anytype) !void
    pub fn collectPrefixed(self: *ParamRegistry, prefix: []const u8, model: anytype) !void
    pub fn parameterCount(self: *const ParamRegistry) usize
    pub fn view(self: *const ParamRegistry, index: usize) ParamView
    pub fn zeroGrad(self: *ParamRegistry) void
    pub fn addParamsTo(self: *const ParamRegistry, opt: anytype) !void
    pub fn saveStateDict(self: *const ParamRegistry, writer: *std.Io.Writer) !void
    pub fn loadStateDict(self: *ParamRegistry, reader: *std.Io.Reader, options: state_dict.LoadOptions) !void
};
pub const ParamView = struct { name, dtype, shape, bytes: []u8, trainable: bool };
```

- `addParam` registers one tensor under an explicit name. Variables
  (`grad_state != null`) are trainable; constants and grad-free typed
  f16/bf16 tensors register as FROZEN entries — saved and loaded, but
  skipped by `addParamsTo` and `zeroGrad`. Errors:
  `CheckpointInvalidName`, `CheckpointDuplicateName`,
  `error.NonContiguousParam`; unsupported dtypes are a compile error.
- `collect` reflectively registers every f32/f16/bf16 tensor field of a
  model (mutable struct pointer), naming by field path: nested structs get
  dotted names (`"encoder.weight"`), arrays and slices index with dots
  (`"layers.0.weight"`), mutable single-item pointers are followed
  transparently, optionals descend into the payload, and tagged unions
  descend into the ACTIVE arm under the same prefix (exactly one arm is
  live, so the checkpoint path stays stable across storage-variant arms —
  e.g. an f16 vs bf16 weight union). Const pointers, comptime fields,
  scalars, and unsupported-dtype tensors are skipped. `collectPrefixed`
  prepends a root prefix.
- `view(i)` returns a borrowed per-entry view in registration order —
  `bytes` aliases the live (mutable) storage; this is the seam
  gradient-free consumers use (`es.Trainer.addRegistry`, §11.11), frozen
  entries included.
- `addParamsTo(opt)` forwards each TRAINABLE entry (f32/f16/bf16) to
  `opt.addParamNamed(&param, name)` — so trainers delegate registration and
  checkpoint identity to the registry in one call, and optimizer slot names
  automatically equal the state-dict paths.
- `saveStateDict`/`loadStateDict` wrap `fucina.state_dict` over the full
  entry set (frozen included).

**Names are the on-disk schema.** A registered name is a checkpoint field
path: strict loading matches by exact name, so RENAMING a parameter path
silently orphans old checkpoints (`CheckpointUnknownName` for the stream
entry, `CheckpointMissingEntry` for the renamed destination). When a rename
is unavoidable, do NOT loosen `strict` — pass a
`state_dict.LoadOptions.aliases` rule
(`.{ .old = "enc.w", .new = "encoder.w" }`) so old checkpoints load into
the new path while keeping the one-to-one guarantee.

Round trip through a directory (the trainable model saves; a gradient-free
constant twin loads — spirals' inference phase):

```zig
test "ParamRegistry: collect, save to a directory, load by name" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const Model = struct { w: fucina.Tensor(.{ .out, .in }) };
    var model = Model{ .w = try fucina.Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 }) };
    defer model.w.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    {
        var registry = fucina.ParamRegistry.init(alloc);
        defer registry.deinit();
        try registry.collect(&model); // names by field path: "w"
        var file = try tmp.dir.createFile(io, "model.safetensors", .{});
        defer file.close(io);
        var fbuf: [4096]u8 = undefined;
        var writer = file.writer(io, &fbuf);
        try registry.saveStateDict(&writer.interface);
        try writer.interface.flush();
    }
    // Inference twin: a CONSTANT registers as a frozen entry and still loads.
    var restored = Model{ .w = try fucina.Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0, 0, 0, 0, 0, 0 }) };
    defer restored.w.deinit();
    {
        var registry = fucina.ParamRegistry.init(alloc);
        defer registry.deinit();
        try registry.collect(&restored);
        var file = try tmp.dir.openFile(io, "model.safetensors", .{});
        defer file.close(io);
        var fbuf: [4096]u8 = undefined;
        var reader = file.reader(io, &fbuf);
        try registry.loadStateDict(&reader.interface, .{});
    }
    try std.testing.expectEqualSlices(f32, try model.w.dataConst(), try restored.w.dataConst());
}
```

### 11.7 State dicts (`src/state_dict.zig`)

`fucina.state_dict` is the neutral named-tensor serialization layer: models,
LoRA adapters, and optimizers all speak in named entries without depending
on each other. The wire format is Hugging Face safetensors (§12); a state
dict written here is a valid standalone safetensors file.

```zig
pub const NamedTensor    = struct { name, dtype, shape, bytes: []const u8;
    pub fn of(name: []const u8, t: anytype) !NamedTensor };     // borrowed name + storage
pub const NamedTensorMut = struct { name, dtype, shape, bytes: []u8;
    pub fn of(name: []const u8, t: anytype) !NamedTensorMut };  // requires a mutable tensor pointer
pub const Alias = struct { old: []const u8, new: []const u8 };
pub const LoadOptions = struct { strict: bool = true, aliases: []const Alias = &.{} };
pub fn saveStateDict(allocator, writer: *std.Io.Writer, entries: []const NamedTensor) !void
pub fn loadStateDict(allocator, reader: *std.Io.Reader, entries: []const NamedTensorMut, options: LoadOptions) !void
```

`NamedTensor.of` accepts a pointer to any contiguous f32/f16/bf16 facade
tensor (variable or constant; other dtypes are compile errors;
non-contiguous is `error.NonContiguousParam`). `NamedTensorMut.of`
additionally requires a mutable tensor pointer — a `*const` argument is a
compile error. Both the name and the storage are BORROWED — they must
outlive the entry.

`saveStateDict` validates everything before writing a byte: names must be
non-empty, NUL-free, valid UTF-8, not `"__metadata__"`, and unique; entry
byte lengths must match dtype×shape; a hand-built `NamedTensor` whose dtype
is outside f32/f16/bf16 fails with `CheckpointUnsupportedDtype`. Raw bytes
are then written — no conversion. `loadStateDict` reads one safetensors
prefix from the reader and matches stream entries to destinations BY NAME
(any order), after applying the `aliases` remap to each STREAM name (first
matching rule wins). Shape and dtype must match the destination exactly
(`CheckpointShapeMismatch`/`CheckpointDtypeMismatch` — no conversion is
ever performed). Strict mode (the default) demands a one-to-one match: an
unmatched stream entry is `CheckpointUnknownName`, an unfilled destination
`CheckpointMissingEntry`. Non-strict skips unknown STREAM entries and
leaves destinations absent from the stream unchanged. The load is two-pass
transactional: every stream entry validates against its destination before
any destination byte is written; pass 2 commits with plain `@memcpy`s, so
any error leaves every destination byte-unchanged. Error set:
`state_dict.Error` (the same checkpoint error names as `optim.OptimError`).

```zig
test "state_dict: named save/load round-trip" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var w = try fucina.Tensor(.{ .out, .in }).constant(&ctx, try ctx.fromSlice(&.{ 2, 2 }, &.{ 3, -1, 4, 1 }));
    defer w.deinit();

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try fucina.state_dict.saveStateDict(alloc, &writer, &.{
        try fucina.state_dict.NamedTensor.of("enc.w", &w),
    });

    var dst = try fucina.Tensor(.{ .out, .in }).constant(&ctx, try ctx.zeros(&.{ 2, 2 }));
    defer dst.deinit();
    var reader = std.Io.Reader.fixed(writer.buffered());
    try fucina.state_dict.loadStateDict(alloc, &reader, &.{
        try fucina.state_dict.NamedTensorMut.of("enc.w", &dst),
    }, .{});
    try std.testing.expectEqualSlices(f32, try w.dataConst(), try dst.dataConst());
}
```

### 11.8 safetensors read/write surface (`src/safetensors.zig`)

`fucina.state_dict` sits on `fucina.safetensors`, which is also usable
directly: `File` (`parse`, `parseOwned`, `load`, `loadMmap`, `deinit`,
`tensor`, `maybeTensor`, `names`, `tensorNames`, `len`, `isEmpty`),
`TensorInfo` (+ `sliceBytesAlloc` with `TensorInfo.Slice` ranges),
`readPrefix` (one safetensors payload from a stream — what `loadStateDict`
uses), `serialize` / `serializeAlloc` / `saveFileAtomic`, `Tensor`,
`MetadataEntry`, `DType` (+ `bitsize`, `string`), `dtypeFromFucina` /
`dtypeToFucina`, `max_header_size`, `Error`. Container format, dtype
coverage, and mmap semantics are §12.

### 11.9 Checkpoint directories (`src/training_checkpoint.zig`)

Canonical resumable checkpoints are DIRECTORIES: the portable tensor
artifact is a clean safetensors file, Fucina-only resume state lives beside
it, and a small JSON sentinel commits the whole thing.

```text
checkpoint/
  model.safetensors        # or adapters.safetensors (LoRA runs)
  optimizer.fucina         # native optimizer frames (§11.5)
  trainer_state.json       # written LAST; the commit sentinel
```

```zig
pub const model_state_file     = "model.safetensors";
pub const adapters_state_file  = "adapters.safetensors";
pub const optimizer_state_file = "optimizer.fucina";
pub const trainer_state_file   = "trainer_state.json";
pub const Error = error{ InvalidTrainerState, UnsupportedTrainerStateVersion };

pub const TrainerState = struct {
    version: u32 = 1, step: u64 = 0, seed: u64 = 0,
    lora_rank: ?u64, lora_alpha: ?f64, lora_dropout_p: ?f64, learning_rate: ?f64,
    accum_steps: ?u64,                       // window size; step % accum_steps == 0 at save
    data_seed: ?u64, data_epoch: ?u64, data_index: ?u64,   // llm.data.Loader.State
    es_sigma: ?f64, es_alpha: ?f64, es_population: ?u64,
    es_noise: ?u64,                          // STABLE mapping: 0 = iid, 1 = correlated
    es_antithetic: ?u64,                     // 1 = mirrored pairs
    es_anchor_decay: ?u64, es_anchor_lambda: ?f64,          // 0/absent none, 1 l1, 2 l2
    es_ternary_flip_rate: ?f64, es_ternary_update_fraction: ?f64, es_ternary_update_decay: ?f64,
    es_iteration: ?u64,
};

pub fn pathJoin(allocator, dir_path: []const u8, leaf: []const u8) ![]u8
pub fn beginSave(allocator, io: std.Io, dir_path: []const u8) !void
pub fn writeFileAtomic(io: std.Io, path: []const u8, context: anytype,
    comptime writeFn: fn (@TypeOf(context), *std.Io.Writer) anyerror!void) !void
pub fn saveTrainerState(allocator, io: std.Io, dir_path: []const u8, state: TrainerState) !void
pub fn loadTrainerState(allocator, io: std.Io, dir_path: []const u8) !TrainerState
```

**Crash-consistency protocol.** `beginSave` creates the directory and
DELETES `trainer_state.json` first — the sentinel — so a checkpoint being
rewritten is visibly uncommitted. Each payload file is then written through
`writeFileAtomic` (temp file + atomic rename, so no reader ever sees a
partial file). `saveTrainerState` writes the sentinel LAST, itself
atomically. Consequence: a directory with a parseable `trainer_state.json`
is a complete, committed checkpoint; a crash mid-save leaves a sentinel-less
directory that resume logic must treat as absent (the previous sentinel was
deleted up front, so a torn save can never masquerade as committed). All optional `TrainerState` fields
serialize only when set and parse to `null` when absent (older checkpoints
stay readable); `format` must be `"fucina.training_checkpoint"` and
`version` 1 (`UnsupportedTrainerStateVersion` otherwise). With gradient
accumulation, save only at window boundaries — accumulated gradients live
only in GradStates and are never serialized ([TRAINING.md](TRAINING.md)
§4). The es_* fields make an ES checkpoint self-describing without any
`optimizer.fucina` (§11.11). `examples/spirals.zig` `saveCheckpoint` /
`loadCheckpoint` is the reference composition of these helpers with
`ParamRegistry` and `saveState`.

### 11.10 LoRA adapters (`src/lora.zig`)

For a frozen linear weight `W: [out, in]` — f32, f16, bf16, or a
block-quantized constant, anything `dot` accepts as a frozen RHS (§5 routes
gradients to the f32 LHS only; constants carry no GradState) — an adapter
learns the additive update

```text
y = base(x) + (alpha / r) * dropout(x) · A^T · B^T
```

with `A: [r, in]` kaiming-uniform (seeded, deterministic — the PyTorch
`nn.Linear`/LoRA-A init) and `B: [out, r]` zeros, so the initial delta is
exactly zero. Only A and B train; the base never changes.

```zig
pub const rank_tag: Tag = .lora_r;                  // reserved rank-axis tag
pub const LoraError = error{ InvalidRank, InvalidDropout };
pub const Config = struct { rank: usize, alpha: f32, dropout_p: f32 = 0 };

pub fn Adapter(comptime in_tag: Tag, comptime out_tag: Tag) type {
    // in_tag != out_tag; neither may be .lora_r (compile errors)
    pub const ATensor = Tensor(.{ rank_tag, in_tag });
    pub const BTensor = Tensor(.{ out_tag, rank_tag });
    pub const Config = ...;                          // re-export
    a: ATensor, b: BTensor, scale: f32, dropout_p: f32,

    pub fn init(ctx: *ExecContext, in_dim: usize, out_dim: usize, config: Config, seed: u64) !Self
    pub fn deinit(self: *Self) void
    pub fn Delta(comptime XPtr: type) type           // x's tags with in_tag -> out_tag
    pub fn delta(self: *const Self, ctx: *ExecContext, x: anytype, dropout_seed: ?u64) !Delta(@TypeOf(x))
    pub fn apply(self: *const Self, ctx: *ExecContext, x: anytype, base: anytype, dropout_seed: ?u64) !Delta(@TypeOf(x))
    pub fn registerParams(self: *Self, opt: anytype, comptime name_prefix: []const u8) !void
    pub fn namedTensors(self: *const Self, comptime name_prefix: []const u8) ![2]optim.NamedTensor
    pub fn namedTensorsMut(self: *Self, comptime name_prefix: []const u8) ![2]optim.NamedTensorMut
    pub fn mergeInto(self: *const Self, ctx: *ExecContext, w: anytype) !void
    pub fn mergeF16(self: *const Self, ctx: *ExecContext, w: anytype) !W // W = w's f16 tensor type; NEW tensor
}
```

- `init` validates `1 <= rank <= min(in_dim, out_dim)`
  (`LoraError.InvalidRank`) and `0 <= dropout_p < 1`
  (`LoraError.InvalidDropout`). `seed` drives A's fill deterministically
  (same seed → bitwise-identical A). A and B are caller-owned VARIABLES —
  never scope-adopted; keep them alive as long as any optimizer or
  state-dict entry borrows them; pair with `deinit`. The effective scaling
  `alpha/rank` makes `alpha` transfer across ranks.
- `delta` computes `scale * dropout(x)·A^T·B^T`; `apply` adds a
  caller-supplied `base` (an f32 facade tensor carrying exactly the delta's
  tags — e.g. the frozen-path output `x.dot(ctx, &w, in_tag)`).
  `dropout_seed` selects the mode: a fresh per-step seed trains (consumed
  only when `dropout_p > 0`; derive per step/layer with `rng.at` — reusing
  a seed reuses the mask), `null` is eval and skips dropout entirely
  (identical to the `p == 0` zero-copy identity path). Input validation is
  comptime: x must be f32, carry `in_tag`, and carry neither `out_tag` nor
  `.lora_r`.
- **Composite-op contract**: `delta`/`apply` build a multi-op chain and
  release interior tensors on return, so any call whose result will be
  `backward()`'d MUST run under an exec scope (§6.3) — without one the
  released interior graph nodes dangle (UB). Without backward (eval), no
  scope is needed.
- `registerParams(opt, "prefix")` registers A/B via `addParamNamed` as
  `"prefix.lora_a"` / `"prefix.lora_b"`; `namedTensors`/`namedTensorsMut`
  produce the matching state-dict entries. Adapter names double as the
  on-disk schema (§11.6).
- `mergeInto(ctx, &w)` folds the adapter into an f32 base IN PLACE
  (`w += scale·B·A`; `w: [out_tag, in_tag]`, dims checked at runtime →
  `error.ShapeMismatch`). It goes through the facade's `data()` gate, which
  only grants mutable access to no-grad tensors — a variable base returns
  `error.MutableDataRequiresNoGrad`, exactly the right fence: only frozen
  bases merge. `mergeF16` widens an f16 base to f32, merges, casts back,
  and returns a NEW f16 tensor (caller-owned). Quantized bases are NOT
  mergeable in place, deliberately: dequantize→merge→re-encode compounds
  quantization error. In memory, dequantize to f32 (`.to(ctx, .f32)`) and
  merge into the copy; for files, merge into a dense f32/f16/bf16 base and
  quantize the RESULT (below).

```zig
test "LoRA: zero delta at init; eval forward; f32 merge parity" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var adapter = try fucina.lora.Adapter(.in, .out).init(&ctx, 8, 4, .{ .rank = 2, .alpha = 4 }, 42);
    defer adapter.deinit();

    var x_vals: [16]f32 = undefined;
    fucina.rng.uniformFill(1, &x_vals, -1, 1);
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 8 }, &x_vals);
    defer x.deinit();
    var w_vals: [32]f32 = undefined;
    fucina.rng.uniformFill(2, &w_vals, -1, 1);
    var w = try fucina.Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 4, 8 }, &w_vals);
    defer w.deinit();

    var base = try x.dot(&ctx, &w, .in); // frozen-base forward
    defer base.deinit();
    var y0 = try adapter.apply(&ctx, &x, &base, null); // null seed = eval
    defer y0.deinit();
    // B is zero-initialized: apply returns the base bitwise.
    try std.testing.expectEqualSlices(f32, try base.dataConst(), try y0.dataConst());

    fucina.rng.uniformFill(3, adapter.b.value.data(), -0.5, 0.5); // stand-in for training
    var y1 = try adapter.apply(&ctx, &x, &base, null);
    defer y1.deinit();

    try adapter.mergeInto(&ctx, &w); // w += (alpha/rank) * B*A, in place
    var y2 = try x.dot(&ctx, &w, .in); // the merged weight alone
    defer y2.deinit();
    for (try y1.dataConst(), try y2.dataConst()) |expected, got| {
        try std.testing.expectApproxEqAbs(expected, got, 1e-4); // fp order differs, not bitwise
    }
}
```

**Fine-tune → merge → quantize → serve.** LLM-scale LoRA fine-tuning lives
in `llm.qwen3.train` (§13): `Trainer(targets)` puts adapters on selected
projections of a frozen GGUF model, `saveAdapters`/`loadAdapters` persist
them as `adapters.safetensors` (names `layers.<i>.<target>.lora_{a,b}`;
`loadAdaptersWithOptions` threads `state_dict.LoadOptions`). The loop back
to a servable model is `zig build export-gguf` (merge and quantize are
separate passes BY DESIGN — one combined pass would chain-requantize):

```sh
zig build finetune -Doptimize=ReleaseFast -- \
    --model models/Qwen3-0.6B-f16.gguf --steps 30 --save /tmp/qwen3-lora
zig build export-gguf -Doptimize=ReleaseFast -- \
    --from-gguf models/Qwen3-0.6B-f16.gguf --adapters /tmp/qwen3-lora \
    --alpha 16 --out /tmp/qwen3-tuned-f16.gguf          # merge (dense f32/f16/bf16 base only)
zig build export-gguf -Doptimize=ReleaseFast -- \
    --from-gguf /tmp/qwen3-tuned-f16.gguf --dtype q4_k --out /tmp/qwen3-tuned-q4_k.gguf
zig build qwen3 -Doptimize=ReleaseFast -- /tmp/qwen3-tuned-q4_k.gguf --chat "..."
# or: llama-cli -m /tmp/qwen3-tuned-q4_k.gguf            # any GGUF consumer
```

The adapter checkpoint stores A/B but not alpha — pass the training-time
value to `--alpha`. Details, transcode policy, and gradient-verification
evidence: [TRAINING.md](TRAINING.md) §9.

### 11.11 Evolution strategies (`src/es.zig`)

`fucina.es` trains WITHOUT gradients — a faithful reimplementation of
ES-at-scale (arXiv:2509.24372; algorithm reimplemented from the paper, the
reference code being under a noncommercial license): deliberately vanilla
OpenAI-ES with the reference's simplifications kept intact —

```text
eps_n ~ N(0, I)                               n = 1..population
R_n   = reward(theta + sigma * eps_n)         (forward passes only)
C_n   = (R_n - mean(R)) / (std(R) + 1e-8)     (z-score, f64 stats, ddof = 0)
theta += (alpha / population) * sum_n C_n * eps_n
```

No antithetic pairs, no rank shaping, no optimizer state, and no 1/sigma in
the update (folded into alpha; the reference default is `alpha = sigma/2`).
Because the signal is one scalar reward per member, ES composes with
anything scoreable from a forward pass, and every parameter is fair game —
no `GradState` needed: f32 variables, f32 constants, typed f16/bf16
tensors, whole registries, and packed ternary genomes all register.

```zig
pub const EsError = error{ InvalidConfig, AnchorMissing, UnsupportedDType,
    NonContiguousParam, DuplicateParam, NoParams, RewardCountMismatch,
    MemberActive, MemberNotActive, ReplicaShapeMismatch };
pub const NoiseScheme = enum { iid, correlated };
pub const RestoreMode = enum { regenerate, snapshot };
pub const AnchorDecay = enum { none, l1, l2 };
pub const RewardNorm  = enum { z_score, centered_ranks, none };
pub const Stats = struct { mean_reward: f64, std_reward: f64, min_reward: f32, max_reward: f32 };
pub const BlockTQ2_0 = ...;                    // re-export (ternary genomes, §10)

pub const Config = struct {
    sigma: f32 = 0.001,
    alpha: ?f32 = null,                        // null = sigma/2
    population: usize = 30,
    antithetic: bool = false,                  // mirrored (+eps, -eps) pairs; even population
    noise: NoiseScheme = .iid,
    restore_mode: RestoreMode = .regenerate,
    cache_streams: bool = false,
    anchor_decay: AnchorDecay = .none, anchor_lambda: f32 = 0,
    reward_norm: RewardNorm = .z_score,
    ternary_flip_rate: f32 = 0.001, ternary_update_fraction: f32 = 0.005,
    ternary_update_decay: f32 = 0.0,
    seed: u64 = 42,
};
```

```zig
pub const Trainer = struct {
    iteration: u64 = 0,                        // advances once per update; persist to resume
    pub fn init(allocator: Allocator, config: Config) !Trainer
    pub fn deinit(self: *Self) void
    pub fn alphaValue(self: *const Self) f32
    pub fn addParam(self: *Self, t: anytype) !void
    pub fn addParamNamed(self: *Self, t: anytype, name: ?[]const u8) !void
    pub fn addRegistry(self: *Self, registry: *const ParamRegistry) !usize
    pub fn addTernaryParam(self: *Self, blocks: []BlockTQ2_0, len: usize) !void
    pub fn addTernaryParamNamed(self: *Self, blocks: []BlockTQ2_0, len: usize, name: ?[]const u8) !void
    pub fn captureAnchor(self: *Self) !void
    pub fn paramCount(self: *const Self) usize
    pub fn elementCount(self: *const Self) usize
    pub fn memberSeed(self: *const Self, member: usize) u64
    pub fn ternaryMemberSeed(self: *const Self, member: usize) u64
    pub fn ternarySlotStreamSeed(member_seed: u64, slot_index: usize) u64
    pub fn ternaryFlipCount(self: *const Self, len: usize) usize
    pub fn perturb(self: *Self, ctx: *ExecContext, member: usize) !void
    pub fn restore(self: *Self, ctx: *ExecContext, member: usize) !void
    pub fn materializeMember(self: *const Self, member: usize, dst: []const []u8) !void
    pub fn materializeTernaryMember(self: *const Self, member: usize, dst: []const []BlockTQ2_0) !void
    pub fn update(self: *Self, ctx: *ExecContext, rewards: []const f32) !Stats
    pub fn step(self: *Self, ctx: *ExecContext, evaluator: anytype) !Stats
    pub fn evaluateMembers(self: *const Self, evaluator: anytype, rewards: []f32, workers: usize) !void
};
```

**Registration.** `addParam`/`addParamNamed` accept a pointer to any
f32/f16/bf16 facade tensor — autograd variable, constant, or grad-free
typed tensor; ES treats them all the same (compile error otherwise;
`NonContiguousParam` at runtime). The trainer retains a refcounted storage
view (the facade may move by value); names are borrowed. `addRegistry`
registers every entry of a `ParamRegistry` — trainable AND frozen —
BORROWING buffers and names (the registry must outlive the trainer);
entries whose storage is already registered are SKIPPED, not rejected
(tied weights perturb once, matching torch `named_parameters()`
deduplication), and the added-slot count is returned. Duplicate storage via
`addParam` is `DuplicateParam`; adding any slot while a member is applied
in place is `MemberActive`. `init` validates the config
(`InvalidConfig`): sigma/alpha positive-finite, population ≥ 2, even under
`antithetic`, ternary knobs in range, and for AWD a positive-finite lambda
with `alpha*lambda < 1` for l2.

**Seed-regenerated noise (the scale trick).** Noise is never stored: a
member's perturbation is a pure function of
`(config.seed, iteration, member, slot, element)` through the counter-based
gaussian `rng.gaussianFillAtFast` (vectorized; a distinct checkpoint
contract from the scalar `gaussianFillAt` mapping that APOLLO stays on), so
`perturb`, `restore`, and `update` regenerate it on the fly — O(1) memory
beyond the parameters. `memberSeed(member)` exposes the derivation
(domain-separated `rng.at`); the mapping may never change once checkpoints
exist. `NoiseScheme.iid` (default) gives every (member, slot) an
independent stream; `.correlated` reuses ONE stream per member across slots
— same-length slots get identical noise — mirroring the reference library's
acknowledged reseeding artifact (kept for reference-faithful runs). Both
are checkpoint contracts: `(config.seed, iteration)` plus population and
the scheme knobs fully regenerate the population, so resume needs only the
iteration counter (`TrainerState.es_*` fields — there is no
`optimizer.fucina` in an ES checkpoint; on resume, validate
sigma/alpha/population and restore `es_iteration`).

**Two evaluation shapes.**

- *In place* (big models): `perturb(ctx, member)` mutates the registered
  buffers (`theta += sigma*eps`, chunk-parallel); exactly one member may be
  active (`MemberActive` tripwire; `restore` of the wrong member is
  `MemberNotActive`). `restore` undoes it: `.regenerate` (default)
  subtracts the regenerated noise — exact up to `(x+t)-t` float drift — or
  `.snapshot` memcpys parameter bytes back (bitwise, costs one parameter
  copy). `step(ctx, evaluator)` is the sequential driver:
  perturb → `evaluator.eval(member) !f32` → restore per member, then one
  `update`; on an eval error it restores before propagating.
- *Member-parallel replicas* (small parameter sets): `materializeMember`
  writes `theta + sigma*eps_member` into caller-owned replica buffers
  (`dst[k]` = slot k in registration order, byte lengths validated —
  `ReplicaShapeMismatch`; buffers must be scalar-aligned) without touching
  shared theta; `evaluateMembers(evaluator, rewards, workers)` fans members
  out over OS threads pulling from a shared counter, calling
  `evaluator.evalMember(worker, member) !f32` — `worker` indexes the
  caller's replica table. `workers` ≥ 1 (clamped to population and 64); the
  evaluator must be thread-safe across distinct workers; the first error
  stops the fan-out and is returned after all workers join.
  `examples/es_spirals.zig` is this shape end-to-end (each worker owns a
  replica model + its own ExecContext; only scalar rewards cross threads).

**The update.** `update(ctx, rewards)` (`rewards.len == population`, else
`RewardCountMismatch`; `NoParams` with nothing registered) computes reward
stats in f64 (ddof 0, sequential summation), shapes coefficients per
`reward_norm` — `.z_score` (affine: preserves outlier magnitude, so one
catastrophic member can dominate with unbounded rewards; self-stops exactly
on all-equal rewards), `.centered_ranks` (Salimans centered ranks in
[-0.5, 0.5], monotone-invariant, outlier-immune; ties break by member
index, a pinned total order; all-equal rewards still take a mean-zero
phantom step), `.none` (raw coefficients) — then applies
`theta += (alpha/population)·Σ C_n·eps_n` chunk-parallel with fp32
accumulation and pinned rounding placement (mul-then-add, no FMA; member
order fixed inside each element), narrowing once to the parameter dtype.
Under `antithetic`, pairs fold to `(C_2k - C_2k+1)·eps_k`, halving
update-side regeneration; division stays by the full population. `update`
is mutate-last: every fallible step runs before the first parameter byte
changes, so a failed update is a no-op. It returns the pre-normalization
`Stats` and advances `iteration`. Determinism: all kernels are
element-independent maps, bitwise identical to the serial loop for ANY
thread count; `cache_streams = true` regenerates each stream once per
iteration into a RAM cache and replays it, bitwise-neutral (worthwhile only
when replays dominate — large populations, small parameter sets).

**Anchored weight decay** (arXiv:2605.30148): with
`anchor_decay = .l1|.l2` and `anchor_lambda`, each `update` ends with a
proximal pull toward a fixed anchor — l2 shrinks `theta - theta_0` by
`(1 - alpha*lambda)`, l1 soft-thresholds it at `alpha*lambda` (exact
zeroing; sparsity in the fine-tuning delta) — counteracting random-walk
drift in reward-irrelevant directions. `captureAnchor` snapshots the
CURRENT float parameters as theta_0: call once after registration while
the parameters still hold the pretrained values, in particular BEFORE
loading a checkpoint on resume (the anchor is never serialized;
`update` without it is `AnchorMissing`). Fine-tuning only — anchoring a
random init pins the model to noise. Reference values: l2 lambda 10, l1
lambda 0.01 at alpha 5e-4.

```zig
test "ES: perturb/evaluate/restore/update shrinks the sphere objective" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // ES needs no gradients: a plain constant is a first-class parameter.
    var theta = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{4}, &.{ 1, -2, 0.5, 3 });
    defer theta.deinit();

    var trainer = try fucina.es.Trainer.init(alloc, .{ .sigma = 0.05, .population = 8, .seed = 7 });
    defer trainer.deinit();
    try trainer.addParamNamed(&theta, "theta");

    const sumSq = struct {
        fn of(t: anytype) !f32 {
            var s: f32 = 0;
            for (try t.dataConst()) |v| s += v * v;
            return s;
        }
    }.of;

    const before = try sumSq(&theta);
    var rewards: [8]f32 = undefined;
    for (0..200) |_| {
        for (&rewards, 0..) |*r, member| {
            try trainer.perturb(&ctx, member); // theta += sigma * eps_member
            r.* = -(try sumSq(&theta)); // reward from a plain forward pass
            try trainer.restore(&ctx, member);
        }
        _ = try trainer.update(&ctx, &rewards); // z-score + fp32-accumulated step
    }
    try std.testing.expect(try sumSq(&theta) < before);
    try std.testing.expectEqual(@as(u64, 200), trainer.iteration);
}
```

**Parity evidence.** Three layers: `tools/gen_es_goldens.py` replicates the
repo RNG bit-level and the update algebra in numpy (tolerance goldens in
`src/es_tests.zig` — the generator's f64 libm gaussian sits a few f32 ulps
from the `gaussianFillAtFast` polynomials); a
test-local straight-line serial reference pins the chunk-parallel kernels
BITWISE; and `tools/check_es_parity.py` runs the ACTUAL reference code
(es-at-scale perturb/restore/z-score/update, and es-awd's decay kernels)
against torch transcriptions of es.zig's algebra on identical noise —
bitwise `torch.equal` on f32/f16/bf16, both noise schemes. The one
deliberate substitution is the RNG itself (repo-owned splitmix64 instead of
torch Philox — a checkpoint contract). On `-Dgpu=cuda` builds, GPU-resident
slots run perturb/restore/update/anchor as device kernels bitwise-identical
to the CPU path for any launch geometry, so checkpoints are
device-independent; non-resident slots, bf16, and active stream caches fall
back per slot ([TRAINING.md](TRAINING.md) §13).

**Practical notes.** Rewards must be finite — one NaN/Inf poisons the
z-score for the whole iteration (clamp in the evaluator; the reference
scores failed rollouts 0.0). Prefer BOUNDED rewards, or
`reward_norm = .centered_ranks` for unbounded ones (raw −CE) — neither
normalization shrinks the step near an optimum, so the practical brakes are
saturating rewards, conservative sigma, AWD, and eval-selected checkpoints.
Changing `population`, `seed`, `noise`, or `antithetic` mid-run breaks the
noise contract exactly like editing an optimizer checkpoint.
`zig build es-spirals` (from-scratch two-spirals training, no backward
anywhere; self-verifying against `--target`, default 0.90 accuracy — 100 %
is the typically observed result) and `zig build es-finetune` (finetune.zig's
gradient-free twin: `--mode lora` perturbs adapters, `--mode full` every
resident float weight — quantized blocks cannot take noise; rewards
`rule`/`acc`/`nll`) are the reference applications
([TRAINING.md](TRAINING.md) §13).

**Ternary-native ES.** `addTernaryParam`/`addTernaryParamNamed` register
BORROWED packed TQ2_0 genomes (`len` a positive multiple of 256,
`blocks.len == len/256`, every 2-bit crumb a valid ternary code — corrupt
code 3 is rejected at registration): the packed blocks ARE the training
state, so every member and every checkpoint is a servable ternary model
(training = packed inference model). Perturbation is sparse trit flips from
the dedicated `es_trits` counter-RNG domain (`ternaryMemberSeed` /
`ternarySlotStreamSeed` / `ternaryFlipCount` expose the pinned mappings;
float slots stay bitwise unchanged when ternary slots join); restore
replays a per-slot undo log in reverse (clamping at the rails is lossy, so
regenerate-subtract cannot work); the update is EGGROLL-style
vote-and-threshold top-K one-bin moves; `materializeTernaryMember` is the
replica twin. Block scales (`d`) are never touched; ternary slots skip
snapshots, stream caches, and AWD. The three `ternary_*` config knobs are
checkpoint contracts (`es_ternary_*` in `TrainerState`). Quantization
background and the TQ2_0 layout: §10; design record:
[TERNARY.md](TERNARY.md); acceptance demo: `zig build es-ternary-spirals`.

## 12. Model I/O: GGUF and safetensors

Fucina speaks two interchange formats and two native sidecar formats:

| Format | Module (facade export) | Role |
|---|---|---|
| GGUF v2/v3 | `fucina.gguf` (`src/gguf.zig`) | llama.cpp-ecosystem model interop: read quantized weights, re-emit/transcode/export |
| safetensors | `fucina.safetensors` (`src/safetensors.zig`) | Hugging Face tensor container: neutral named-tensor payloads |
| state-dict stream | `fucina.state_dict` (`src/state_dict.zig`) | named, dtype-aware checkpoint entries; wire format IS safetensors |
| checkpoint directory | `fucina.training_checkpoint` (`src/training_checkpoint.zig`) | resumable-training layout: safetensors payloads + native `optimizer.fucina` frames + JSON sentinel |

All integers in both binary formats are little-endian. None of the types in
this section have internal locking: a parsed `File` is safe to read from many
threads once construction returns, but `deinit`, `takeMapping`, and every
writer mutation require external serialization.

### 12.1 GGUF reader (`src/gguf.zig`)

#### 12.1.1 Opening and lifetime

```zig
pub const File = struct {
    allocator: Allocator,
    bytes: []u8,                          // entire file (heap copy or mmap)
    tensors: []TensorInfo,
    index: std.StringHashMap(usize),      // tensor name -> tensors[i]
    metadata: std.StringHashMap(Value),
    alignment: usize,                     // data-section alignment (default 32)
    data_offset: usize,                   // file offset of the tensor data section
    is_mmap: bool = false,
    extra_bytes: [][]u8 = &.{},           // mappings of split parts 2..N (part 1 = bytes)
    part_data_offsets: []u64 = &.{},      // each part's data-section offset

    pub fn load(allocator: Allocator, io: std.Io, path: []const u8) !File
    pub fn loadMmap(allocator: Allocator, io: std.Io, path: []const u8) !File
    pub fn loadMmapAuto(allocator: Allocator, io: std.Io, path: []const u8) !File
    pub fn splitPartPaths(allocator: Allocator, path: []const u8) !?[][]u8
    pub fn parseOwned(allocator: Allocator, bytes: []u8) !File
    pub fn deinit(self: *File) void
    pub fn takeMapping(self: *File) ?MappedRegion
    pub fn isSplit(self: *const File) bool
    pub fn partDataOffset(self: *const File, part: u16) u64
};
```

Four constructors, one parse core:

- `File.load` reads the whole file into a heap buffer, then parses. Errors:
  `error.IsDir` for non-regular files, `error.EndOfStream` on a short read,
  plus the parse errors below. Prefer for small files (tests, tools).
- `File.loadMmap` maps the file read-only (`PROT_READ`, `MAP_PRIVATE`) and
  parses in place; the fd is closed immediately (POSIX keeps the mapping
  valid). An empty file is `Error.InvalidMagic`. This is the path for
  multi-GB models: pages are file-backed and evictable, and no heap copy
  coexists with the materialized weights.
- `File.loadMmapAuto` is `loadMmap` that transparently follows llama.cpp
  split GGUFs: when the path names the first `-00001-of-0000N` part
  (`splitPartPaths` enumerates all part paths; null for any other path),
  every part is mapped and parsed into one merged `File` — part 1's
  metadata (splits carry the full metadata there), the union of all parts'
  tensors (each tagged with its `TensorInfo.part`), one index over all of
  them. The mappings of parts 2..N live in `extra_bytes`; `isSplit`
  reports a split load, and `partDataOffset(part)` is each part's
  data-section offset within its own file. For a non-split path it behaves
  exactly like `loadMmap`.
- `File.parseOwned` takes ownership of caller-provided bytes; on parse
  failure the bytes are freed (exactly once — callers must not also free).

Parsing accepts GGUF versions 2 and 3 (`Error.UnsupportedVersion` otherwise),
magic `"GGUF"` (`Error.InvalidMagic` otherwise). Header counts larger than
the file length are rejected before any allocation (`Error.InvalidTensorInfo`).
A tensor whose declared extent runs past EOF logs a self-diagnosing message
(name, shortfall in bytes/GB — the signature of a truncated download) and
returns `Error.InvalidTensorInfo`.

**Ownership.** Everything a `File` hands out — metadata strings, array
payloads, `TensorInfo.name`, `TensorInfo.data` — is a zero-copy slice into
`File.bytes`. All of it dies at `deinit`, which frees the hash maps and the
`tensors` slice, then frees (heap) or `munmap`s (mmap) the bytes.

```zig
// nested inside File — the symbol is gguf.File.MappedRegion
pub const MappedRegion = struct {
    bytes: []const u8,
    pub fn deinit(self: *MappedRegion) void   // munmap
};
```

`takeMapping` transfers ownership of the underlying mmap to the caller and
returns `null` when the file was heap-read or split-loaded (a split's
tensors point into all part mappings, but a `MappedRegion` can carry only
one). After a successful `takeMapping`,
`File.deinit` no longer unmaps; every previously parsed slice (metadata,
`TensorInfo.data`) stays valid for as long as the returned `MappedRegion`
lives. This is how a model borrows quantized weight blocks straight from the
mapping instead of copying them (the `fucina_llm` loaders do this for large
expert tensors — §13). The absolute addresses of tensor payloads are aligned
(page-aligned mapping base plus `alignment`-multiple offsets), so borrowed
block slices satisfy the block-struct alignment that §10's kernels assume.

```zig
pub fn prefetch(data: []const u8) void
```

`prefetch` issues `madvise(WILLNEED)` over a mapped region about to be read
in full, letting OS readahead run ahead of a sequential copy/pack loop — the
dominant cold-load cost. It is a silent no-op on heap buffers and wherever
the advice call fails. Deliberately not called for borrowed (zero-copy)
blocks, which stay lazily paged.

#### 12.1.2 Metadata

```zig
pub const Value = union(enum) {
    int: i64, float: f64, boolean: bool,
    string: []const u8, array: Array,
};
pub const Array = struct {
    item_type: u32,        // wire value-type code of the elements
    len: usize,
    data: []const u8,      // raw bytes spanning all elements
    pub fn stringSlices(self: Array, allocator: Allocator) ![][]const u8
};
```

The parser widens scalars: every integer wire type (u8/i8/u16/i16/u32/i32/
u64/i64) becomes `i64`, every float becomes `f64`. A wire `uint64 >= 2^63`
cannot be represented and fails the whole parse with
`Error.MetadataValueOutOfRange`. Strings and arrays are zero-copy. Wire
value-type codes (ggml's `GGUF_TYPE_*`, also the writer-side `MetaType`
enum): 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string,
9 array, 10 u64, 11 i64, 12 f64. Nested arrays (array-of-array) are
`Error.UnsupportedValueType`.

`Array.stringSlices` decodes a string array (item_type 8, otherwise
`Error.UnsupportedValueType`) into a caller-freed outer slice; the inner
strings still borrow the file bytes.

Typed lookups (all `null` when the key is absent or the wrong kind):

```zig
pub fn meta(self: *const File, key: []const u8) ?Value
pub fn getString(self: *const File, key: []const u8) ?[]const u8
pub fn getInt(self: *const File, key: []const u8) ?i64
pub fn getFloat(self: *const File, key: []const u8) ?f64   // also widens int
pub fn getBool(self: *const File, key: []const u8) ?bool   // also accepts int != 0
pub fn getArray(self: *const File, key: []const u8) ?Array
```

`general.alignment` is special-cased: it is validated straight from the wire
value, before the lossy widening — a non-integer, negative, zero,
non-power-of-two, or out-of-range (`>= 2^63` or `> 2^20`) alignment returns
`Error.InvalidAlignment` instead of reaching undefined behavior at a cast or
`alignForward`. The validated value replaces the default alignment of 32.

#### 12.1.3 Tensor directory

```zig
pub const TensorInfo = struct {
    name: []const u8,
    dims: [4]usize,        // GGUF ne[] order: innermost/fastest-varying FIRST
    n_dims: usize,         // 1..4
    ggml_type: GgmlType,
    offset: usize,         // relative to the data section (data_offset)
    data: []const u8,      // exact wire bytes, borrowed from File.bytes
    part: u16 = 0,         // split part holding this tensor (0 = single-file)

    pub fn dim(self: TensorInfo, index: usize) !usize            // InvalidTensorInfo past n_dims
    pub fn logicalMatrixShape(self: TensorInfo) ![2]usize        // 2-D only: {dims[1], dims[0]}
};

pub fn get(self: *const File, name: []const u8) !*const TensorInfo   // Error.TensorNotFound
pub fn maybeGet(self: *const File, name: []const u8) ?*const TensorInfo
```

`dims` follow ggml's `ne[]` convention — the innermost axis first — so a
Fucina row-major logical `[out, in]` matrix appears as `dims = { in, out }`.
`logicalMatrixShape` performs that swap for rank-2 tensors and errors with
`Error.InvalidTensorInfo` for any other rank.

```zig
pub const GgmlType = enum(u32) { f32 = 0, f16 = 1, q4_0 = 2, ... };
pub fn dtypeForGgmlType(value: GgmlType) ?DType
pub fn tensorByteLen(ggml_type: GgmlType, dims: []const usize) !usize
```

`GgmlType` carries the ggml wire codes for: `f32`, `f16`, `bf16`, `f64`,
`i8`, `i16`, `i32`, `i64`, the block quants `q1_0 q4_0 q4_1 q5_0 q5_1 q8_0
q8_1 q2_k q3_k q4_k q5_k q6_k q8_k`, the i-quants `iq1_s iq1_m iq2_xxs
iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs`, the ternaries `tq1_0 tq2_0`, and
the microscaling floats `mxfp4 nvfp4`. An unknown wire code fails parsing
with `Error.UnsupportedGgmlType`. `dtypeForGgmlType` maps every quantized
and float format to the corresponding core `DType` (§8); only the integer
scalars (`i8 i16 i32 i64`) and `f64` return `null` (they have no core dtype
— their bytes are still readable via `TensorInfo.data`).

`tensorByteLen` computes exact wire size: scalar formats multiply the element
count by the element size; block formats require the innermost dim to be a
whole number of blocks (`dims[0] % blockSize == 0`, like `ggml_row_size` —
`Error.InvalidTensorInfo` otherwise) and multiply block count by
`blockByteSize` (§10). A zero-length dimension is a legitimate empty ggml
tensor and yields 0 bytes.

**Layout rules.** The tensor data section starts at
`data_offset = alignForward(header_end, alignment)`; each `TensorInfo.offset`
is relative to that. The parser bounds-checks `data_offset + offset +
tensorByteLen` against the file and slices `data` accordingly.

Reading metadata and the directory of a real model file:

```zig
fn snippetInspectGguf(alloc: std.mem.Allocator, io: std.Io) !void {
    var file = try fucina.gguf.File.loadMmap(alloc, io, "models/Qwen3-0.6B-Q4_K_S.gguf");
    defer file.deinit(); // unmaps; every borrowed slice dies here

    const arch = file.getString("general.architecture").?;
    const n_layers = file.getInt("qwen3.block_count").?;
    const tokens = file.getArray("tokenizer.ggml.tokens").?;
    const vocab = try tokens.stringSlices(alloc); // slices point into the mapping
    defer alloc.free(vocab);

    const embd = try file.get("token_embd.weight");
    fucina.gguf.prefetch(embd.data); // OS readahead before a sequential copy
    const dt = fucina.gguf.dtypeForGgmlType(embd.ggml_type); // core DType (§8)
    _ = .{ arch, n_layers, dt };
} // requires model assets to run
```

### 12.2 GGUF writer (`src/gguf.zig`)

```zig
pub const Writer = struct {
    pub fn init(allocator: Allocator) Writer
    pub fn deinit(self: *Writer) void

    pub fn addMetaString(self: *Writer, key: []const u8, value: []const u8) !void
    pub fn addMetaInt(self: *Writer, key: []const u8, comptime Int: type, value: Int) !void
    pub fn addMetaFloat(self: *Writer, key: []const u8, comptime Float: type, value: Float) !void
    pub fn addMetaBool(self: *Writer, key: []const u8, value: bool) !void
    pub fn addMetaArray(self: *Writer, key: []const u8, comptime Elem: type, values: []const Elem) !void
    pub fn addMetaStringArray(self: *Writer, key: []const u8, values: []const []const u8) !void
    pub fn addMetaCopy(self: *Writer, from: *const File, key: []const u8) !void
    pub fn copyAllMetadata(self: *Writer, from: *const File, skip_keys: []const []const u8) !void
    pub fn copyAllMetadataRaw(self: *Writer, file_bytes: []const u8, skip_keys: []const []const u8) !void

    pub fn addTensor(self: *Writer, name: []const u8, ggml_type: GgmlType,
                     dims: []const usize, data: []const u8) !void
    pub fn finish(self: *const Writer, out: *std.Io.Writer) !void
};
pub const MetaType = enum(u32) { uint8 = 0, int8, uint16, int16, uint32, int32,
                                 float32, boolean, string, array, uint64, int64, float64 };
```

The writer buffers metadata KVs and tensor declarations, then `finish`
serializes everything in one pass: `"GGUF"` magic, version 3, tensor count,
KV count, the KV section, the tensor-info section, zero padding to
`alignment`, then each tensor's data padded to `alignment` — including the
last tensor, matching ggml's writer byte-for-byte.

**llama.cpp-exact offsets.** Tensor offsets are precomputed as the running
padded total, relative to the data-section start — llama.cpp's reader
rejects files whose offsets are not exactly that value, so nothing here is
optional. Re-parsing a `finish` output and re-emitting it reproduces the
file byte-identically, and a real-model re-emit preserves every KV and
tensor payload verbatim (both asserted in `src/gguf_tests.zig`).

**Ownership.** Metadata keys/payloads and tensor names are duplicated into
the writer (`deinit` frees them). Tensor `data` is **borrowed** and must stay
alive until `finish` returns. `finish` is `*const` and repeatable.

**Metadata semantics.**

- `addMetaInt`/`addMetaFloat`/`addMetaArray` select the exact wire type from
  the comptime scalar type (u8/i8/u16/i16/u32/i32/u64/i64, f32/f64; anything
  else is a compile error). This matters: llama.cpp type-checks many keys, so
  passthrough-adjacent metadata must keep its original width.
- Re-adding an existing key replaces its value **in place** — file order is
  preserved and GGUF keys stay unique.
- `addMetaCopy` copies one KV **byte-verbatim** from a parsed `File`,
  preserving the exact wire type that the parser's widened `Value` map drops
  (it re-reads the raw file bytes). Absent key: `Error.KeyNotFound`.
  `copyAllMetadata` does the same for every KV except `skip_keys`, in the
  source file's order. Both require `from` to still own its bytes — call
  them before `from.deinit()`/`takeMapping()`. `copyAllMetadataRaw` is
  `copyAllMetadata` over a raw GGUF byte region, for callers whose `File`
  transferred its mapping away via `takeMapping` while the region is still
  alive.
- `general.alignment` is tracked no matter how it is added: it must be wire
  type uint32, a power of two, and `<= 2^20`, else `Error.InvalidAlignment`;
  it changes the padding rule used by `finish` (mirroring the parser).

**Tensor declaration.** `addTensor` dims are ne-order (innermost first),
exactly as the parser surfaces them — re-emitting a parsed tensor is
`addTensor(info.name, info.ggml_type, info.dims[0..info.n_dims], info.data)`.
Validation: non-empty name, 1–4 dims, `data.len == tensorByteLen(...)`
(`Error.InvalidTensorInfo` otherwise), unique name
(`Error.DuplicateTensorName`).

Write-then-read round-trip:

```zig
test "gguf: write, reopen, verify" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    var w = fucina.gguf.Writer.init(alloc);
    defer w.deinit();
    try w.addMetaString("general.name", "demo");
    try w.addMetaInt("demo.heads", u32, 8);

    const values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var wire: [24]u8 = undefined;
    try fucina.gguf.encodeF32(.f32, &values, &wire);
    try w.addTensor("w", .f32, &.{ 3, 2 }, &wire); // ne order: logical [2, 3]

    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "doc_demo_{d}.gguf", .{std.Io.Clock.real.now(io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    {
        var out = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer out.close(io);
        var buf: [4096]u8 = undefined;
        var writer = out.writer(io, &buf);
        try w.finish(&writer.interface);
        try writer.interface.flush();
    }

    var file = try fucina.gguf.File.load(alloc, io, path);
    defer file.deinit();
    try std.testing.expectEqualStrings("demo", file.getString("general.name").?);
    try std.testing.expectEqual(@as(i64, 8), file.getInt("demo.heads").?);
    const info = try file.get("w");
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, &(try info.logicalMatrixShape()));
    try std.testing.expectEqualSlices(u8, &wire, info.data);
}
```

### 12.3 The f32 transcode seam: `encodeF32` / `decodeF32` (`src/gguf.zig`)

```zig
pub fn encodeF32(ggml_type: GgmlType, src: []const f32, dst: []u8) !void
pub fn decodeF32(ggml_type: GgmlType, src: []const u8, dst: []f32) !void
```

`encodeF32` is the writer-side quantize seam: it encodes f32 values as
`ggml_type` wire bytes. `decodeF32` is its exact mirror. Both enforce the
length contract — the byte slice must equal
`tensorByteLen(ggml_type, &.{float_slice.len})`, else
`Error.InvalidTensorInfo`.

| Format group | `encodeF32` | `decodeF32` | Path |
|---|---|---|---|
| `f32`, `f16`, `bf16` | yes | yes | element-wise cast (f16 may overflow to inf on out-of-range values, matching ggml's scalar conversion) |
| `q4_0 q4_1 q5_0 q5_1 q8_0 q4_k q5_k q6_k tq2_0` | yes | yes | byte-exact ggml-parity block codecs: `quantizeRowForDType` / `dequantizeRowForDType` (§10) |
| everything else (`q2_k`, `q3_k`, i-quants, `mxfp4`, ...) | `Error.EncoderUnavailable` | `Error.DecoderUnavailable` | — |

Additional block-format contracts:

- **Finite input only:** the block encoders assume finite input, so
  `encodeF32` rejects any NaN/inf in `src` with `Error.NonFiniteValue`
  (release builds included) — the same seam llama.cpp guards with
  `ggml_validate_row_data`. Scalar casts stay unguarded.
- **Alignment:** for block formats, `dst`/`src` must be aligned to the block
  struct (`@alignOf(Storage(dt))`, §8), else `Error.InvalidTensorInfo`.
  Little-endian targets only (the same assumption as the parser's zero-copy
  blocks).

```zig
test "gguf: q8_0 quantize seam round-trip" {
    var src: [64]f32 = undefined;
    for (&src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 - 3.2;

    // tensorByteLen(.q8_0, &.{64}) = 2 blocks x 34 bytes.
    var wire: [68]u8 align(2) = undefined;
    try fucina.gguf.encodeF32(.q8_0, &src, &wire);

    var back: [64]f32 = undefined;
    try fucina.gguf.decodeF32(.q8_0, &wire, &back);
    for (src, back) |a, b| try std.testing.expectApproxEqAbs(a, b, 0.05);
}
```

### 12.4 The export-gguf tool (`tools/export_gguf.zig`)

`zig build export-gguf` builds and runs the exporter (installed as
`fucina-zig-export-gguf`), which closes the train → export → serve-anywhere
loop on top of `Writer` + `encodeF32`/`decodeF32`:

```sh
# (a) re-emit / transcode
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf Qwen3-0.6B-f16.gguf --out Qwen3-0.6B-Q4_K_S.gguf --dtype q4_k

# (b) merge Fucina LoRA adapters (safetensors, as saved by `zig build finetune`)
#     into dense f32/f16/bf16 base weights and re-emit
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf base-f16.gguf --adapters ckpt-dir --alpha 16 --out merged.gguf
```

| Flag | Meaning |
|---|---|
| `--from-gguf PATH` | input model (mmap-loaded); required |
| `--out PATH` | output path; required |
| `--dtype MODE` | global transcode target: `verbatim` (default), `f32`, `f16`, `bf16`, `q8_0`, `q4_k`, `q5_k`, `q6_k`, `tq2_0` |
| `--experts-dtype MODE` | override for tensors named `*_exps.weight` only; may requantize a quantized source |
| `--adapters DIR_OR_FILE` | checkpoint directory containing `adapters.safetensors`, or a safetensors file directly |
| `--alpha F` | LoRA scaling; **required** with `--adapters` (the safetensors checkpoint stores A/B but not alpha; finetune default 16) |

All metadata passes through byte-verbatim (`copyAllMetadata`); a non-verbatim
`--dtype` additionally sets `general.file_type` to the matching llama.cpp
`llama_ftype` code (uniform K-quant exports report the `_S` variants).

**Transcode policy** (llama.cpp-convention): only matrix weights transcode —
`n_dims >= 2`, name ends `.weight`, name does not contain `norm` — including
`token_embd`/`output`. Norms and 1-D tensors keep their stored type. Sources
must be f32/f16/bf16: transcoding an already-quantized source would
chain-requantize, so the global `--dtype` errors
(`error.QuantizedSourceUnsupported`); re-emit those verbatim instead.
Block-divisibility rules: a `q8_0` target with `dims[0] % 32 != 0`, or a
K-quant/`tq2_0` target with `dims[0] % 256 != 0`, keeps the **source** dtype
(more conservative than llama-quantize's smaller-quant fallback: no extra
quantization loss, small size cost). A tensor containing NaN/inf is refused
at the `encodeF32` seam with a named diagnostic.

The `--experts-dtype` override (experts-only quantization) IS allowed to
requantize pre-quantized expert tensors (dequant → re-encode through
`decodeF32`/`encodeF32`): shipped MoE GGUFs store experts pre-quantized, and
experts are where shrinking bytes pays most in decode bandwidth at lowest
quality risk. Block divisibility still rules.

**Merge policy**: adapters named `layers.<i>.<q|k|v|o|gate|up|down>.lora_a/b`
merge into the matching `blk.<i>.attn_*/ffn_*.weight` tensors via
`lora.Adapter.mergeInto`/`mergeF16` (§11). Quantized bases error
(`error.QuantizedBaseUnsupported`): merge on an f32/f16/bf16 base, then
`--dtype`-transcode in a second pass. `--adapters` cannot be combined with
`--dtype`/`--experts-dtype` in one run. A `.lora_b` without its `.lora_a`
(or vice versa) is a hard error, as is an empty adapter set.

### 12.5 safetensors (`src/safetensors.zig`)

#### 12.5.1 Format and dtypes

Layout (current upstream contract): a u64 little-endian JSON header length,
the UTF-8 JSON header, then one contiguous tensor data buffer. Tensor
`data_offsets` are relative to the start of that buffer and must cover it
exactly, in ascending order, with no holes or overlap. When writing, the
header is padded to an 8-byte multiple with spaces so the first data byte is
naturally aligned for scalar dtypes. `pub const max_header_size` is
100,000,000 bytes, enforced both ways.

```zig
pub const DType = enum {
    BOOL, F4, F6_E2M3, F6_E3M2, U8, I8,
    F8_E5M2, F8_E4M3, F8_E8M0, F8_E4M3FNUZ, F8_E5M2FNUZ,
    I16, U16, F16, BF16, I32, U32, F32, C64, F64, I64, U64,
    pub fn bitsize(self: DType) usize
    pub fn string(self: DType) []const u8
};
pub fn dtypeFromFucina(dtype: fucina.DType) !DType   // f32/f16/bf16 only
pub fn dtypeToFucina(dtype: DType) !fucina.DType     // F32/F16/BF16 only
```

Every upstream dtype tag round-trips as raw bytes, including the sub-byte
`F4` (4-bit) and `F6_*` (6-bit) types — for those, the total bit count of a
tensor must land on a byte boundary (`Error.MisalignedSlice` otherwise).
Only `F32`/`F16`/`BF16` map to core `DType`s; both direction functions
return `Error.UnsupportedDtype` for the rest.

#### 12.5.2 Reading

```zig
pub const File = struct {
    allocator: Allocator,
    bytes: []const u8,
    tensors: []TensorInfo,                       // sorted by data offset
    metadata: std.StringHashMap([]const u8),     // __metadata__ entries (owned copies)
    index: std.StringHashMap(usize),
    ownership: Ownership = .borrowed,            // borrowed | owned | mmap

    pub fn parse(allocator: Allocator, bytes: []const u8) !File        // borrows bytes
    pub fn parseOwned(allocator: Allocator, bytes: []u8) !File         // takes ownership
    pub fn load(allocator: Allocator, io: std.Io, path: []const u8) !File
    pub fn loadMmap(allocator: Allocator, io: std.Io, path: []const u8) !File
    pub fn deinit(self: *File) void
    pub fn tensor(self: *const File, name: []const u8) !*const TensorInfo   // Error.TensorNotFound
    pub fn maybeTensor(self: *const File, name: []const u8) ?*const TensorInfo
    pub fn names(self: *const File) []const TensorInfo    // the full info slice, not just names
    pub fn tensorNames(self: *const File, allocator: Allocator) ![][]const u8
    pub fn len(self: *const File) usize
    pub fn isEmpty(self: *const File) bool
};
pub fn readPrefix(allocator: Allocator, reader: *std.Io.Reader) !File
```

Ownership mirrors GGUF with one extra mode: `parse` **borrows** the input
bytes (the caller keeps them alive until `deinit` and frees them itself),
`parseOwned`/`load` own a heap buffer, `loadMmap` owns a read-only mapping.
`deinit` always frees the per-tensor `name`/`shape` copies, the metadata
copies, and the maps, then releases the bytes according to the mode.
`readPrefix` consumes exactly one safetensors frame from a stream reader —
header length, header, then precisely the data the header describes —
leaving the reader positioned after it (multiple frames can share a stream);
the result is an `owned` `File`.

Unlike GGUF, tensor **names and shapes are allocated copies** (they survive
into error paths cleanly); only `TensorInfo.data` borrows `File.bytes`.

```zig
pub const TensorInfo = struct {
    name: []const u8,
    dtype: DType,
    shape: []usize,
    data_offsets: [2]usize,      // [begin, end) relative to the data buffer
    data: []const u8,            // borrowed from File.bytes

    pub const Slice = struct { start: usize = 0, end: ?usize = null };
    pub fn sliceBytesAlloc(self: *const TensorInfo, allocator: Allocator,
                           ranges: []const Slice) ![]u8
};
```

`sliceBytesAlloc` gathers a row-major sub-block into a caller-owned byte
buffer: one `Slice` per leading axis (missing trailing axes take the full
extent; `ranges.len > rank` is `Error.InvalidSlice`, `start > end` or
`end > dim` likewise). Byte-aligned dtypes only (`Error.MisalignedSlice`
for `F4`/`F6_*`); a rank-0 tensor returns a copy of its whole payload.

Validation on parse (all before any tensor is exposed): UTF-8 header
(`InvalidHeader`), JSON well-formedness and schema
(`InvalidHeaderDeserialization` — including duplicate JSON keys), header
length sanity (`HeaderTooSmall`/`HeaderTooLarge`/`InvalidHeaderLength`),
offsets contiguous-ascending and matching `dtype x shape` byte length
(`InvalidOffset`/`TensorInvalidInfo`), buffer covered exactly — trailing
polyglot bytes or missing data are `MetadataIncompleteBuffer` — duplicate
tensor names (`DuplicateTensorName`), string-only `__metadata__`
(`InvalidMetadata`), and checked arithmetic throughout
(`ValidationOverflow`). Zero-sized tensors (a 0 in the shape) are legal.

#### 12.5.3 Writing

```zig
pub const Tensor = struct { name: []const u8, dtype: DType,
                            shape: []const usize, data: []const u8 };
pub const MetadataEntry = struct { key: []const u8, value: []const u8 };

pub fn serialize(allocator: Allocator, writer: *std.Io.Writer,
                 tensors: []const Tensor, metadata: ?[]const MetadataEntry) !void
pub fn serializeAlloc(allocator: Allocator, tensors: []const Tensor,
                      metadata: ?[]const MetadataEntry) ![]u8
pub fn saveFileAtomic(allocator: Allocator, io: std.Io, path: []const u8,
                      tensors: []const Tensor, metadata: ?[]const MetadataEntry) !void
```

All three build the same bytes (golden-pinned against upstream safetensors
output in `src/safetensors_tests.zig`): everything is validated before the
first byte is written. Tensors are sorted by descending `DType` declaration
order, then ascending name — input order does not matter and is not
preserved. (Declaration order is not bit width: `BOOL` is declared first,
so under the descending sort its tensors land last, after the narrower
F4/F6 types.) Names must be non-empty UTF-8 and not `__metadata__`
(`InvalidTensorName`); duplicate names are `DuplicateTensorName`; each
tensor's `data.len` must equal its `dtype x shape` byte length
(`TensorInvalidInfo`). Metadata is an optional flat string map: unique
UTF-8 keys/values (`InvalidMetadata`), emitted as `__metadata__`.

`saveFileAtomic` writes to `PATH.tmp.<nanotimestamp>` (preallocated via
`setLength`, `F_NOCACHE` on macOS to skip the page cache on a one-shot
sequential write) and renames over `path`; on rename failure the temp file
is removed. Readers never observe a half-written file.

```zig
test "safetensors: serialize and parse back" {
    const alloc = std.testing.allocator;
    const st = fucina.safetensors;

    const values = [_]f32{ 1.5, -2.0, 0.25, 8.0 };
    const tensors = [_]st.Tensor{.{
        .name = "layer.weight",
        .dtype = .F32,
        .shape = &.{ 2, 2 },
        .data = std.mem.sliceAsBytes(&values),
    }};
    const meta = [_]st.MetadataEntry{.{ .key = "format", .value = "pt" }};

    const bytes = try st.serializeAlloc(alloc, &tensors, &meta);
    defer alloc.free(bytes);

    var file = try st.File.parse(alloc, bytes); // borrows `bytes`
    defer file.deinit();
    const t = try file.tensor("layer.weight");
    try std.testing.expectEqual(st.DType.F32, t.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, t.shape);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&values), t.data);
    try std.testing.expectEqualStrings("pt", file.metadata.get("format").?);
}
```

### 12.6 Named state dicts (`src/state_dict.zig`)

The API — `NamedTensor`/`NamedTensorMut`, `Alias`/`LoadOptions`,
`saveStateDict`/`loadStateDict`, name validation, alias remapping, strict
one-to-one matching, and the two-pass transactional load — is §11.7; the
consumers (`fucina.ParamRegistry`, the `fucina.optim` re-exports) and the
schema-stability contract for registered names are §11.5–§11.6. What
belongs here is the wire format: a state dict is exactly one safetensors
frame of §12.5 with no `__metadata__` entry — there is no bespoke stream
format for state dicts. Entry names become the safetensors header keys
unchanged (hence the §11.7 name rules: non-empty, NUL-free, unique UTF-8,
not `"__metadata__"`); dtypes map through `dtypeFromFucina`/`dtypeToFucina`
(§12.5), with only F32/F16/BF16 produced or accepted; tensor payloads are
raw little-endian storage bytes, no conversion in either direction.
`loadStateDict` consumes one frame from a stream via `readPrefix` (§11.8),
so a state dict can be embedded in a longer stream. Any safetensors
consumer can read the file; GGUF remains a separate interop/export codec.

### 12.7 Training-checkpoint directory and native optimizer frames (`src/training_checkpoint.zig`, `src/optim.zig`)

The directory layout, the four file-name constants, the save/load API
(`pathJoin`, `beginSave`, `writeFileAtomic`, `saveTrainerState`,
`loadTrainerState`), the sentinel commit protocol, and the `TrainerState`
fields are §11.9; the frame-magic table and all load-time validation and
resume semantics are §11.5. This subsection documents the on-disk formats.

**`trainer_state.json`.** A flat JSON object with the fixed marker
`"format": "fucina.training_checkpoint"` and `"version": 1` (anything else
is `Error.InvalidTrainerState` / `Error.UnsupportedTrainerStateVersion`).
`step` and `seed` (u64) are always present; every optional field is simply
omitted when null and parses to `null` when absent. Enum-like fields
(`es_noise`, `es_anchor_decay`) serialize through stable on-disk integer
mappings — never `@intFromEnum` of an in-memory enum.

**`optimizer.fucina`.** The raw byte stream produced by one optimizer's
`saveState` (or `OptimizerSet.saveState`) from `src/optim.zig`; the
magic ↔ optimizer table is §11.5. Common frame shape: 4-byte magic,
optional optimizer-config scalars (Muon, Apollo, SGD write theirs right
after the magic), a u32 slot count, then per-slot records:

- **name** — u16 length + bytes; the explicit `addParamNamed` name or the
  auto-name `param<i>`.
- **dims** — u64 rows, u64 cols.
- **scalars** — u64 step (Adam/AdamW/Apollo/SGD); Apollo main slots add a
  u64 projection seed and a u32 f32-bit `prev_norm`.
- **state buffers** — in v3 frames, raw f32 bytes; in v4 and v5 frames each
  buffer is prefixed by one u8 `StateDType` tag (0 = f32, 1 = bf16;
  wire-stable) followed by the raw storage bytes. Apollo state buffers stay
  raw f32 in every version.
- **master record** — v5 frames only: each slot ends with a u8 presence
  flag, then the slot's raw f32 master weights when the flag is set.

A Muon frame is immediately followed by its embedded AdamW fallback frame;
an Apollo frame carries a second u32 count + slot list for its fallback
slots inside the same `FZP3`/`FZP5` frame; an `FZO3` container is the magic,
a u32 member count, then each member's frame in registration order.

**`FZT1`** (`optim.saveTensors`/`loadTensors`) is the minimal positional
format for parameter values only: magic, u32 tensor count, then per tensor a
u32 rank, rank u64 dims, and the raw f32 little-endian data. It carries no
names and no dtypes (usage rules and the named alternative are §11.5).

## 13. The LLM stack (fucina_llm)

`fucina_llm` is a second Zig module layered on top of the `fucina` facade (its
only module dependency; see §2 for build wiring). It contains everything a
transformer inference/fine-tuning runner needs that is not a tensor op:
GGUF-to-weight binding, KV caching, tokenizers, sampling, SFT data plumbing,
multi-turn chat, and lossless draft-free speculative decoding. Import it as:

```zig
const fucina = @import("fucina");
const llm = @import("fucina_llm");
```

### 13.1 Module layout (`src/llm.zig`)

Model families live in subdirectories and are exposed as namespaces; generic,
family-agnostic helpers stay flat:

| Namespace | Contents | Files |
|---|---|---|
| `llm.qwen3` | `model`, `train` — Qwen3 dense + LoRA fine-tuning | `llm/qwen3/` |
| `llm.qwen35` | `model` — Qwen3.5 Gated-DeltaNet hybrid | `llm/qwen35/` |
| `llm.gemma` | `gemma4`, `gemma4_train`, `moe`, `moe_route`, `moe_route_tensor` | `llm/gemma/` |
| `llm.diffusion_gemma` | `model` — block text-diffusion on the gemma4 backbone | `llm/diffusion_gemma/` |
| `llm.parakeet` | `loader`, `frontend`, `subsampling`, `encoder`, `weights`, `decoder`, `tokenizer`, `streaming`, `transcription` — NeMo FastConformer/RNN-T ASR | `llm/parakeet/` |
| `llm.speculative` | `core`, `sam_index`, `recycling`, `cascade`, `constrained` | `llm/speculative/` |
| `llm.deepseek2` | `model` — DeepSeek-V2 MLA + fine-grained MoE with shared experts | `llm/deepseek2/` |
| `llm.glm4moe` | `model` — GLM-4.5 MoE with native MTP (`nextn`) self-speculation | `llm/glm4moe/` |
| `llm.deepseek4` | `model` — DeepSeek V4 Flash (hyper-connections, compressed-KV MQA, streamed experts, MTP) | `llm/deepseek4/` |

| Flat helper | Purpose | Section |
|---|---|---|
| `llm.weights` | GGUF tensor → typed linear weight binding | §13.2 |
| `llm.ptqtp_gguf` | PTQTP plane persistence — `<name>.ptqtp0/1/2` writer + pair-detecting loader | §13.2 |
| `llm.gguf_meta` | metadata readers + parallel layer loader | §13.3 |
| `llm.kv_cache` | per-layer K/V store for autoregressive decode | §13.4 |
| `llm.kv_persist` | crash-safe append-only KV-cache sidecar: conversations reopen warm | §13.4 |
| `llm.tokenizer` | byte-level BPE (GPT-2/Qwen) | §13.5 |
| `llm.spm_tokenizer` | SentencePiece Unigram (Gemma/llama-vocab) | §13.5 |
| `llm.unicode_categories` | generated `\p{L}`/`\p{N}`/`\s` tables (byte-BPE pretokenizer; shared with out-of-module tokenizers) | §13.5 |
| `llm.sampler` | greedy/temperature/top-k/top-p/min-p/penalties + logit-processor seam | §13.6 |
| `llm.logit_processor` | pluggable logit-transform interface (grammar masks, bias lists) | §13.6 |
| `llm.llguidance` | grammar/JSON-schema constrained decoding (vendored engine, `-Dllguidance`) | §13.6 |
| `llm.data` | SFT pairs, encodePair, deterministic Loader | §13.7 |
| `llm.chat` | templates + generic `Conversation(Model, Tok)` | §13.8 |

The family namespaces are covered in §14 (deepseek2/glm4moe/deepseek4 by
their module doc comments); this section documents the shared stack they
are built from.

### 13.2 Weight loading (`src/llm/weights.zig`)

`weights.zig` turns raw GGUF tensor payloads (§12) into typed, immediately
usable linear weights. Its error set is
`Error = error{ InvalidWeightShape, UnsupportedWeightType, GradUnsupported }`.

#### 13.2.1 `LinearWeight`

```zig
pub const LinearWeight = union(enum) {
    f32: WeightF32,     // fucina.Tensor(.{ .out, .in })
    f16: WeightF16,     // fucina.Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } })
    bf16: WeightBf16,   // fucina.Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } })
    q8_0: WeightQ8_0, q4_k: WeightQ4_K, q5_k: WeightQ5_K, q6_k: WeightQ6_K,
    // plus one QuantWeight(dtype) arm per remaining GGUF block format:
    // q1_0, q4_0, q4_1, q5_0, q5_1, q2_k, q3_k, iq1_s, iq1_m, iq2_xxs, iq2_xs,
    // iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs, tq1_0, tq2_0, mxfp4, nvfp4
    ptqtp: WeightPtqtp, // 1-3 packed TQ2_0 trit-planes (PTQTP, section 10.9)
};
```

Every arm is a `[.out, .in]`-tagged tensor kept **resident in its source
precision** — nothing is widened to f32 at load time:

| Arm | Resident form | Forward path |
|---|---|---|
| `f32` | f32 tensor (f64 sources are narrowed) | plain `dot` |
| `f16` | f16 tensor, 2 B/weight | f16-operands GEMM (§9); GPU-resident on `-Dgpu=metal` |
| `bf16` | raw u16 bit patterns, 2 B/weight | mixed f32×bf16 TransB kernel, exact in-register widening |
| `q8_0`, `q4_k`, `q5_k`, `q6_k` | raw GGUF blocks **plus** a pre-packed matmul RHS | `dotPacked` on the CPU quantized hot path (§10); q4_k/q6_k/q8_0 additionally try the dequant-in-kernel Metal GEMM |
| all other quant arms | raw GGUF blocks (`QuantWeight(dtype)`) | tagged `dot` through the generic quantized matmul |
| `ptqtp` | up to three `.tq2_0` plane tensors (§10.9) — built in place by `toPtqtp`, or rebuilt bitwise from persisted `<name>.ptqtp0/1/2` plane tensors (`llm.ptqtp_gguf` pair-detection; [PTQTP.md](PTQTP.md)) | fused multi-plane entry: ONE Q8_K activation quantization + ONE worker-team dispatch computing every plane and summing in fixed plane order (bitwise equal to per-plane facade dots, which remain the gradient-path fallback) |

`pub fn QuantWeight(comptime dtype: DType) type` returns
`fucina.Tensor(.{ .dtype = dtype, .tags = .{ .out, .in } })`. The four hot
K-quant/Q8 formats get dedicated wrapper structs — `WeightQ4_K`, `WeightQ5_K`,
`WeightQ6_K`, `WeightQ8_0` — each holding `value` (the raw block tensor) and
`packed_rhs: fucina.PackedRhs(dtype)` built once at init, with
`init`/`deinit`/`cloneView`/`concat` (and, except `WeightQ5_K`,
`initWithRhsLifetime` plus a `rhs_lifetime: fucina.RhsLifetime` field that
tells GPU dispatch whether the block bytes are process-stable).

Binding a GGUF tensor:

```zig
pub fn load(ctx: *ExecContext, info: *const gguf.TensorInfo,
            expected_rows: usize, expected_cols: usize) !LinearWeight
pub fn loadForFusion(...same args...) !LinearWeight
pub fn loadWithOptions(...same args..., options: LoadOptions) !LinearWeight

pub const LoadOptions = struct { gpu_resident: bool = true };
```

- The tensor's `logicalMatrixShape()` must equal
  `(expected_rows, expected_cols)` = `(out, in)`, else
  `Error.InvalidWeightShape`; a GGML type without an arm is
  `Error.UnsupportedWeightType`.
- `load` calls `gguf.prefetch` on the payload first (readahead for cold-mmapped
  bytes) and copies/repacks it, so the result **does not borrow** the
  `gguf.File` — the file may be freed after loading (MoE borrow mode below is
  the exception).
- `LoadOptions.gpu_resident` (default `true`): on `-Dgpu=metal` builds,
  f16/q4_k/q6_k/q8_0 payloads are copied into device-owned storage
  (`internal.gpu.allocResidentBytes`) so GPU matmuls read them with zero
  per-call transfer. The storage buffer OWNS the device bytes through a
  release hook: when the last tensor reference (including `cloneView`s sharing
  the buffer) drops, the hook frees the device allocation and evicts the GPU
  shim's cached wrap. The bytes stay CPU-readable (and, for dense f16/f32,
  CPU-writable in place — in-place trainers mutate resident weights and GPU
  dispatch reads the live values). If the device budget is exhausted the load
  silently falls back to heap storage with `.transient` RHS lifetime.
- `loadForFusion` is `loadWithOptions(..., .{ .gpu_resident = false })`: a
  weight loaded only to be consumed by `fuseLinear` skips the per-part device
  copy, because the fused result re-acquires residency itself — per-part
  copies would be alloc+memcpy+free waste. If fusion later declines, the parts
  remain fully usable on the CPU packed path.

Pre-fusion:

```zig
pub fn fuseLinear(ctx: *ExecContext, parts: []const *LinearWeight) !?LinearWeight
```

Concatenates 2–4 same-format weights along `.out` into one stacked matrix
(one GEMM instead of N on the forward path). Supported formats:
f32/f16/bf16/q4_k/q5_k/q6_k/q8_0, plus `ptqtp` parts with a uniform plane
count (planes concatenate plane-wise — byte-identical to decorating the
fused matrix; mixed plane counts return `null` like mixed formats). On
success the parts are **consumed**
(deinitialized) and the fused weight is returned; when the parts' formats
differ, or the format has no fused fast path, it returns `null` and leaves
the parts untouched; fewer than 2 or more than 4 parts is
`Error.InvalidWeightShape`. Fused dense f32/f16 and quant
q4_k/q6_k/q8_0 results re-acquire GPU residency on Metal builds.

Forward/apply entry points on `LinearWeight`:

```zig
pub fn linearSeq(self, ctx, input: anytype, comptime in_tag: Tag, comptime out_tag: Tag)
    !fucina.Tensor(.{ .seq, out_tag })
pub fn linearSeqNormed(self, ctx, x: anytype, norm_weight: anytype, eps: f32,
    comptime in_tag: Tag, comptime out_tag: Tag) !fucina.Tensor(.{ .seq, out_tag })
pub fn supportsNormedFusion(self, m: usize) bool
pub fn getRowsAs(self, ctx, token_ids: []const usize, comptime out_tag: Tag)
    !fucina.Tensor(.{ .seq, out_tag })
pub fn toResidentF16(self: *LinearWeight, ctx: *ExecContext) !void
pub fn toPtqtp(self: *LinearWeight, ctx: *ExecContext, options: fucina.ptqtp.Options)
    !fucina.ptqtp.MatrixStats     // requires ptqtpEligible; drops the source storage
pub fn ptqtpEligible(self: *const LinearWeight) bool  // non-ptqtp arm, inDim % 256 == 0
pub fn outDim(self) usize / pub fn inDim(self) usize
pub fn cloneView(self, ctx) !LinearWeight   // shares storage, fresh tags/packed RHS
pub fn deinit(self: *LinearWeight) void
```

- `linearSeq` computes `input · Wᵀ` with the format's fastest route: packed
  quantized kernels for q4_k/q5_k/q6_k/q8_0 (with a GPU attempt first for
  q4_k/q6_k/q8_0 — declined when the input requires gradients or the exec
  gate says the shape is too small, falling back to the CPU packed path; at
  decode shapes (`seq < 4`, no gradients) q5_k and q6_k instead contract
  against the resident GGUF-native compact blocks — bitwise-equal outputs,
  ~1.57x/1.30x fewer weight bytes streamed than the byte-expanded packed
  layout; default on x86_64, off on aarch64, with
  `FUCINA_Q5K_DECODE_COMPACT`/`FUCINA_NO_Q5K_DECODE_COMPACT` and the Q6K
  pair forcing the route on/off and `setQ5kDecodeCompact`/
  `setQ6kDecodeCompact` as the programmatic overrides), and a tagged `dot`
  for everything else. The per-format helpers `linearSeqQ8_0`,
  `linearSeqQ4_K`, `linearSeqQ5_K`, `linearSeqQ6_K` are also `pub` for
  callers that hold the wrapper struct directly.
- `linearSeqNormed` is `linearSeq` over `rmsNormMul(x, norm_weight, eps)`:
  on the packed CPU q4_k/q5_k/q6_k/q8_0 routes at prefill shapes
  (`seq >= 4`, no gradients; q4_k only on non-MMLA targets) the normalized
  tensor is never materialized — the fused kernel normalizes into
  task-private scratch and quantizes in place, matching the unfused pair to
  f32 roundoff. Every other arm — GPU builds, decode shapes, and
  `FUCINA_NO_NORM_QUANT_FUSED=1` (`FUCINA_NORM_QUANT_FUSED=1` forces the
  fused route; `setNormQuantFused` is the programmatic override) —
  normalizes and delegates. `supportsNormedFusion(m)` reports whether the
  fused route applies for an m-row input; callers fanning one normalized
  input into several projections should require it for every projection —
  the fallback re-normalizes per call.
- `getRowsAs` gathers rows by index (the embedding-lookup shape) and returns
  f32; f16/bf16 rows are widened, quantized rows dequantized. Dedicated arms
  exist for f32/f16/bf16/q4_k/q5_k/q6_k/q8_0, and an `inline else` arm
  routes every remaining block-quantized format through the generic
  quantized row gather — all `LinearWeight` forms work.
- `toPtqtp` dequantizes the weight row-chunk-wise through `getRowsAs` — so
  every loadable source dtype quantizes through one code path — solves the
  trit-planes (§10.9), and replaces the arm in place, dropping the source
  storage. On the `ptqtp` arm, `getRowsAs` returns the dequantized plane
  sum (so `toResidentF16` doubles as un-decorate). `decoratePtqtpInto` +
  `PtqtpReport` aggregate per-tensor solver stats over model walks;
  `llm.qwen3.model.Model.decoratePtqtp(ctx, options)` walks attention
  q/k/v (split or fused), o_proj, and dense FFN projections, with
  `DecoratePtqtpOptions` covering per-projection plane overrides
  (`down_planes`/`o_planes`) and data-free edge-layer skip
  (`skip_first_layers`/`skip_last_layers`); embeddings, lm_head, and norms
  are not walked (decorate `model.output` directly for a ternary head).
  `Model.savePtqtpGguf(ctx, io, src_file, out_path)` persists the decorated
  model: `llm.ptqtp_gguf` writes one standalone TQ2_0 tensor per plane —
  `<name>.ptqtp0/1/2` replaces `<name>`, fused weights row-slicing back to
  their source tensor names — plus a `fucina.ptqtp.version` metadata key,
  everything else byte-verbatim; the qwen3 loaders pair-detect planes and
  rebuild the arm bitwise (re-fusing via `fuseLinear`'s ptqtp arm — other
  families do not read decorated files yet), so decoration runs once and
  the saved file serves through the ordinary qwen3 runners
  ([PTQTP.md](PTQTP.md)).
- `toResidentF16` replaces the weight in place with a dequantized resident-f16
  copy (2 B/weight — the f16 GEMM/GPU-offload operand format), dequantizing in
  4096-row chunks through the same row gather so the transient peak stays a
  few MB. No-op when already f16.

```zig
fn snippetLinearWeight(ctx: *fucina.ExecContext, file: *const fucina.gguf.File, row: []const f32) !void {
    const info = try file.get("blk.0.attn_q.weight");
    var w = try llm.weights.LinearWeight.load(ctx, info, 1024, 1024); // expected [out, in]
    defer w.deinit();

    var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ 1, w.inDim() }, row);
    defer x.deinit();
    var y = try w.linearSeq(ctx, &x, .embed, .attn_q); // format-dispatched matmul
    defer y.deinit();

    var rows = try w.getRowsAs(ctx, &.{ 0, 2 }, .embed); // dequantized f32 row gather
    defer rows.deinit();
} // requires model assets to run
```

```zig
fn snippetFuseLinear(ctx: *fucina.ExecContext, file: *const fucina.gguf.File) !void {
    var gate = try llm.weights.LinearWeight.loadForFusion(ctx, try file.get("blk.0.ffn_gate.weight"), 3072, 1024);
    errdefer gate.deinit();
    var up = try llm.weights.LinearWeight.loadForFusion(ctx, try file.get("blk.0.ffn_up.weight"), 3072, 1024);
    errdefer up.deinit();
    if (try llm.weights.fuseLinear(ctx, &.{ &gate, &up })) |fused| {
        var owned = fused; // one [6144, 1024] weight; gate/up were consumed
        defer owned.deinit();
    } else {
        gate.deinit(); // mixed formats: parts untouched, use them individually
        up.deinit();
    }
} // requires model assets to run
```

#### 13.2.2 Vectors, MoE, and borrowed linears

```zig
pub fn loadVector(ctx: *ExecContext, info: *const gguf.TensorInfo,
                  expected_len: usize, comptime tag: Tag) !fucina.Tensor(.{tag})
pub fn layerName(buf: []u8, layer_i: usize, suffix: []const u8) ![]const u8
```

`loadVector` reads a 1-D tensor (f32/f16/bf16/f64 sources) into an f32 vector;
wrong rank/length is `Error.InvalidWeightShape`. `layerName` formats
`"blk.{d}.{s}"` into a caller buffer — the GGUF per-layer naming convention.

```zig
pub fn loadMoeRhs(ctx: *ExecContext, info: *const gguf.TensorInfo,
    expected_in_dim: usize, expected_out_dim: usize, expected_n_expert: usize,
    borrow: bool) !fucina.MoeRhs
pub fn moeSwiGluFfnSeq(ctx, input: *const Tensor(.{ .seq, .embed }),
    gate: *const fucina.MoeRhs, up: ..., down: ...,
    selected: []const usize, routing_weights: []const f32, top_k: usize,
    out_pe: usize, io: ?std.Io, profile: ?*fucina.MoeBatchProfile)
    !fucina.Tensor(.{ .seq, .embed })
```

`loadMoeRhs` binds one stacked-expert 3-D tensor
(`blk.N.ffn_{gate,up,down}_exps.weight`, GGUF shape `[in, out, n_expert]`) as
a single packed matmul RHS; the fused MoE kernel slices each expert as a
zero-copy row block. Supported expert formats: the K-quants
(q2_k/q4_k/q5_k/q6_k) plus q8_0 (llama.cpp's fallback when an expert dim is
not a 256 multiple), iq2_xxs, iq3_xxs, and tq2_0 —
other formats are `Error.UnsupportedWeightType`. With `borrow = true` the
blocks are borrowed straight from the (mmapped) GGUF, skipping the multi-GB
copy; the caller must then keep the mapping alive for the model's lifetime
(`gguf.File.takeMapping`, §12). `moeSwiGluFfnSeq` is the tensor-valued
Qwen-style SwiGLU MoE FFN over those RHS values; it refuses gradient-tracked
inputs (`Error.GradUnsupported`) and internally splits decode (`seq == 1`)
from batched prefill. `moeGatedFfnSeq` is the same entry with the gated
activation chosen by the caller (`act: fucina.GatedOp`; deepseek4 routes
through the clamped SwiGLU). `loadMoeRhsStreamed(store, file, layer_i,
gate_info, up_info, down_info, expected_in_dim, expected_out_dim,
expected_n_expert)` is the streamed counterpart of three `loadMoeRhs` calls:
it registers one layer's gate/up/down stacked expert tensors with the
`fucina.ExpertStore` (which `pread`s individual experts on demand) and
returns a `StreamedMoeFfnRhs{ gate, up, down }` of `.streamed` RHS values —
only the geometry is validated, nothing of the expert stacks is read.

Zero-copy linears over caller-owned immutable bytes (used by runners that keep
weights mmapped):

```zig
pub fn linearSeqBorrowedF16(ctx, input: anytype, bytes: []const u8, shape: [2]usize,
    comptime in_tag: Tag, comptime out_tag: Tag) !fucina.Tensor(.{ .seq, out_tag })
pub fn linearSeqBorrowedQuantized(comptime dtype: DType, ctx, input: anytype,
    bytes: []const u8, shape: [2]usize, options: BorrowedQuantLinearOptions,
    comptime in_tag: Tag, comptime out_tag: Tag) !fucina.Tensor(.{ .seq, out_tag })

pub const BorrowedQuantLinearOptions = struct {
    allow_gpu: bool = true,
    rhs_lifetime: RhsLifetime = .transient,
};
```

The quantized variant is comptime-restricted to q8_0/q4_k/q5_k/q6_k, rejects
gradient-tracked inputs (`Error.GradUnsupported`), and validates
`input.dim(in_tag) == shape[1]` (`Error.InvalidWeightShape`). Neither takes
ownership of `bytes`.

Metal-residency utilities shared by loaders and eager dispatch batching:

- `ResidentByteRegistry` (`init`/`deinit`/`bytes`): a session/model-owned map
  from host byte pointers to one-time device copies. `bytes(src)` returns the
  device-resident alias (still CPU-readable) on GPU builds, or `src` verbatim
  on non-GPU builds and on any allocation failure; `deinit` frees all device
  copies. Not thread-safe.
- `QuantByteStackPart`, `QuantByteStackOptions{ prefer_device = true,
  require_device = false }`, `QuantByteStack`
  (`deinit(allocator)`/`bytesPerRow`/`totalOutRows`) and
  `makeQuantByteStack(comptime dtype, allocator, parts, options) !?QuantByteStack`:
  copy same-shaped quantized weights into one contiguous stack with the same
  residency policy as the loaders. Returns `null` for empty `parts` or when
  `require_device` is set and no device storage is available;
  mismatched part shapes are `Error.InvalidWeightShape`. Device-capable
  dtypes: q4_k/q6_k/q8_0.

### 13.3 GGUF metadata glue (`src/llm/gguf_meta.zig`)

Flat helpers shared by every model family's loader. Error set:
`Error = error{InvalidConfig}`.

```zig
pub const ZeroPolicy = enum { reject_zero, accept_zero };
pub fn metaInt(file: *const gguf.File, arch: []const u8, suffix: []const u8, zero: ZeroPolicy) Error!usize
pub fn metaIntOpt(file, arch, suffix, zero: ZeroPolicy) ?usize
pub fn metaFloat(file, arch, suffix) Error!f32
pub fn metaFloatOpt(file, arch, suffix) ?f32
```

All read the key `"<arch>.<suffix>"`. Missing keys, negative integers, and —
under `.reject_zero` — present-but-zero integers are invalid: the `Opt`
variants read them as `null`, the strict variants return
`Error.InvalidConfig`. `ZeroPolicy` exists because families disagree about
zero on purpose: qwen3 treats a zero-valued key like a missing key everywhere,
while gemma reads legitimately-zero keys such as
`attention.shared_kv_layers`. A key that overflows the internal 128-byte
format buffer reads as absent.

```zig
test "gguf_meta: zero-valued keys split by policy" {
    const alloc = std.testing.allocator;
    var w = fucina.gguf.Writer.init(alloc);
    defer w.deinit();
    try w.addMetaInt("arch.block_count", u32, 24);
    try w.addMetaInt("arch.shared_kv_layers", u32, 0);
    var buf: [4096]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try w.finish(&sink);
    var file = try fucina.gguf.File.parseOwned(alloc, try alloc.dupe(u8, sink.buffered()));
    defer file.deinit();

    const meta = llm.gguf_meta;
    try std.testing.expectEqual(@as(usize, 24), try meta.metaInt(&file, "arch", "block_count", .reject_zero));
    try std.testing.expectError(error.InvalidConfig, meta.metaInt(&file, "arch", "shared_kv_layers", .reject_zero));
    try std.testing.expectEqual(@as(usize, 0), try meta.metaInt(&file, "arch", "shared_kv_layers", .accept_zero));
    try std.testing.expectEqual(@as(?usize, null), meta.metaIntOpt(&file, "arch", "missing", .accept_zero));
}
```

```zig
pub fn parallelLoadLayers(comptime Layer: type, comptime Loader: type,
    ctx: *ExecContext, loader: Loader, layers: []Layer) !void
```

Loads all model layers in parallel across the exec work pool (§9) when one is
available, serially otherwise — layer loads are independent and the
`ExecContext` allocator and buffer pool are thread-safe, so the multi-GB
copy+pack becomes an N-core job (the dominant chunk of model load time).
`Loader` is a small per-family adapter value providing
`fn load(self, layer_i: usize) !Layer` and
`fn deinitLayer(self, layer: *Layer) void`. On failure, only the layers that
DID load are deinitialized, and the first error **in layer order** is
returned (deterministic even under parallel execution).

### 13.4 KV cache (`src/llm/kv_cache.zig`)

```zig
pub const KvTensor = fucina.Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } });
pub const KvInput  = fucina.Tensor(.{ .seq, .kv_head, .d });   // f32 rows handed to append
pub const KvDtype  = enum { f16, q8_0 };
pub const Error = error{ KvCacheOverflow, KvCacheShapeMismatch, KvCacheHeadDimNotBlockAligned };
```

`KvCache` is the per-layer post-RoPE key/value store for autoregressive
decode, shared by every family and by the speculative decoder. Layout: one
contiguous `[capacity, kv_heads, head_dim]` tensor per layer for K and one for
V — exactly the `[.seq, .kv_head, .d]` layout the attention kernels consume,
so the active prefix `[0..len]` is a zero-copy narrow. K is stored **after**
RoPE (V has no RoPE), so past positions are never re-rotated.

- **f16 default**: 2 B/element — half the f32 footprint and per-step
  bandwidth; the attention kernel widens to f32 in-register. Matches
  llama.cpp's default cache type.
- **Opt-in q8_0** (`initWithDtype`/`initPerLayerWithDtype` with `.q8_0`;
  llama.cpp's `--cache-type-k/v q8_0`): each (position, kv_head) row is stored
  as `head_dim/32` `BlockQ8_0` — 34 bytes per 32 elements, roughly halving f16
  again at a small quantization loss. Requires `head_dim % 32 == 0` (checked
  at init: `Error.KvCacheHeadDimNotBlockAligned`); q8_0 layers are raw block
  slices, consumed via `kBlocks`/`vBlocks` and the attention kernels'
  q8_0-block KV arm.

```zig
pub fn init(ctx: *ExecContext, num_layers: usize, kv_heads: usize, head_dim: usize, capacity: usize) !KvCache
pub fn initWithDtype(...same..., dtype: KvDtype) !KvCache
pub fn initPerLayer(ctx, kv_heads_per_layer: []const usize, head_dims: []const usize, capacity: usize) !KvCache
pub fn initPerLayerWithDtype(...same..., dtype: KvDtype) !KvCache
pub fn deinit(self: *KvCache) void
```

The per-layer variants size each layer's slot independently — Gemma 4
interleaves local sliding-window layers (kv_heads 8, head_dim 256) with global
layers (kv_heads 2, head_dim 512). The cache itself has no window logic: every
position is appended and retained, and windowed models apply their sliding
window at read time through the windowed attention kernels (which also keeps
`truncate` rewind trivially correct). Allocations use `ctx.allocator`; the
caller owns the cache and must `deinit` it.

Decode-loop API:

```zig
pub fn appendLayer(self: *KvCache, ctx: *ExecContext, layer_i: usize,
                   k_rows: *const KvInput, v_rows: *const KvInput) !void
pub fn advance(self: *KvCache, m: usize) void
pub fn reset(self: *KvCache) void                // len = 0, buffers retained
pub fn truncate(self: *KvCache, keep_len: usize) void
pub fn kSlice(self, layer_i: usize, len: usize) ![]const f16   // f16 mode
pub fn vSlice(self, layer_i: usize, len: usize) ![]const f16
pub fn kBlocks(self, layer_i: usize, len: usize) []const fucina.BlockQ8_0  // q8_0 mode
pub fn vBlocks(self, layer_i: usize, len: usize) []const fucina.BlockQ8_0
pub fn byteSize(self) usize
```

- `appendLayer` converts the new tokens' f32 K/V rows to the cache dtype and
  writes them at offset `len`, in one pass with no temporaries. Shape
  mismatches against the layer's geometry are `Error.KvCacheShapeMismatch`;
  exceeding `capacity` is `Error.KvCacheOverflow`. It does **not** advance
  `len` — every layer appends at the same base offset; call `advance(m)` once
  per step after all layers have been written.
- `truncate(keep_len)` rewinds to the first `keep_len` positions (a value at
  or above `len` is a no-op). Decrementing `len` suffices for both storage
  modes: buffers are pre-allocated at `capacity`, every position occupies
  whole per-(position, kv_head) rows, and every reader and `appendLayer`
  address rows strictly from `len` — the next append overwrites the abandoned
  rows. This is the speculative decoder's rewind primitive (§13.9): rejected
  draft positions are dropped with one integer store.

```zig
test "kv cache: append, advance, truncate rewind" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // 1 layer, 2 kv heads, head_dim 4, capacity 8 positions (f16 storage).
    var cache = try llm.kv_cache.KvCache.init(&ctx, 1, 2, 4, 8);
    defer cache.deinit();

    var k = try llm.kv_cache.KvInput.fromSlice(&ctx, .{ 3, 2, 4 }, &([_]f32{0.5} ** 24));
    defer k.deinit();
    var v = try llm.kv_cache.KvInput.fromSlice(&ctx, .{ 3, 2, 4 }, &([_]f32{0.25} ** 24));
    defer v.deinit();
    try cache.appendLayer(&ctx, 0, &k, &v); // writes at offset len, does not advance
    cache.advance(3); // once per step, after all layers
    try std.testing.expectEqual(@as(usize, 3), cache.len);
    try std.testing.expectEqual(@as(usize, 3 * 2 * 4), (try cache.kSlice(0, cache.len)).len);

    cache.truncate(1); // speculative rewind: drop rejected positions
    try std.testing.expectEqual(@as(usize, 1), cache.len);
}
```

`llm.kv_persist` (`src/llm/kv_persist.zig`) persists the cache to a
crash-safe append-only sidecar file so a conversation reopens warm across
process restarts, with zero re-prefill. The sidecar is a fixed header —
magic `FUXKV001`, a record count, and a per-layer geometry guard (any
mismatch with the opening cache ignores the file wholesale) — followed by
one record per position: the token id plus every layer's K/V row bytes
(both cache dtypes round-trip). `reset(io, allocator, path, kv)` arms a
fresh sidecar for the cache's geometry. `appendRange(io, allocator, path,
kv, tokens)` writes the positions the file does not hold yet — record data
first, the header's record count last, so a torn append is invisible;
`tokens.len != kv.len` is `Error.KvPersistTokenMismatch`. `load(io,
allocator, path, kv)` resumes into an empty cache: it applies up to the
stored count (stopping early at a torn tail — the prefix stays usable),
sets `kv.len`, and returns the caller-owned token history, or null when
nothing usable exists (absent file, foreign geometry, or a history beyond
capacity). `chat.Conversation.enablePersistence` (§13.8.2) is the turnkey
consumer.

### 13.5 Tokenizers

#### 13.5.1 Byte-level BPE (`src/llm/tokenizer.zig`)

`llm.tokenizer.Tokenizer` is a native byte-level BPE tokenizer (GPT-2/Qwen
family) built entirely from a model's GGUF metadata
(`tokenizer.ggml.{tokens,merges,pre,bos_token_id,eos_token_id,add_bos_token,add_eos_token}`)
— no external tokenizer dependency, no per-model hardcoding. Error set:
`error{ NoTokenizerVocab, UnsupportedTokenizerFormat, TokenizerTooLarge } || Allocator.Error`.

```zig
pub const SpecialTokens = struct {
    bos: ?u32 = null, eos: ?u32 = null,
    prepend_bos: bool = false, append_eos: bool = false,
};
pub fn initFromGguf(allocator: Allocator, file: *const gguf.File, overrides: SpecialTokens) !Tokenizer
pub fn initFromParts(allocator, vocab_strings: []const []const u8,
                     merge_strings: []const []const u8, special: SpecialTokens) !Tokenizer
pub fn deinit(self: *Tokenizer) void
```

- `initFromGguf` requires a string-array vocab and non-empty merges, and
  refuses SentencePiece-scored models
  (`tokenizer.ggml.scores` present → `Error.UnsupportedTokenizerFormat` — use
  `spm_tokenizer` instead). Special tokens default from metadata; non-null
  `overrides` fields replace them, and `prepend_bos`/`append_eos` in the
  overrides can only force the policy **on** (a `false` leaves the metadata
  value in effect).
- The tokenizer copies all vocab/merge bytes into owned blobs, so it stays
  valid after the source `gguf.File` is freed. Duplicate token bytes resolve
  to the lowest id.
- **Pretokenizer parity**: the chunker is a faithful port of llama.cpp's
  hand-rolled qwen2 pretokenizer loop, backed by generated Unicode category
  tables — on valid UTF-8 input it chunks and encodes **token-ID-exact**
  against llama.cpp for qwen2-pre models (malformed UTF-8 is the one
  documented deviation). A GGUF declaring `"joyai-llm"` (the DeepSeek-V4
  family's byte-oriented splitter) selects that chunker instead, via the
  `pre: Pre = .qwen2` field (`Pre = enum { qwen2, joyai_llm }`). If the
  GGUF declares a pretokenizer other than an implemented chunker, encoding
  still proceeds with the qwen2 rules, but the id is recorded in the
  `pre_mismatch: ?[]u8` field and a warning is logged once — token-ID
  parity is then not guaranteed.

Encode/decode surface:

```zig
pub fn encode(self, allocator, text: []const u8) ![]u32       // BOS/EOS policy applied
pub fn encodeRaw(self, allocator, text: []const u8) ![]u32    // no BOS/EOS (templates own structure)
pub fn encodePlainAppend(self, allocator, text, out: *std.ArrayList(u32)) !void // no marker resolution
pub fn decode(self, allocator, ids: []const u32) ![]u8
pub fn decodeAppend(self, allocator, id: u32, out: *std.ArrayList(u8)) !void
pub fn tokenId(self, token: []const u8) ?u32
pub fn vocabSize(self) usize
pub fn eosId(self) ?u32 / pub fn bosId(self) ?u32 / pub fn isEos(self, id: u32) bool
```

`encode`/`encodeRaw` resolve `<|...|>` special-token markers to their ids
atomically (a `<|` that does not open a known marker is left to normal
pretokenization, matching llama.cpp's partitioning); `encodePlainAppend` skips
marker resolution entirely for callers with their own control-token sets.
Returned slices are owned by the caller.

```zig
test "byte-level BPE: merges, special markers, round-trip" {
    const alloc = std.testing.allocator;
    const vocab = [_][]const u8{ "<|im_end|>", "h", "i", "hi" };
    const merges = [_][]const u8{"h i"};
    var tok = try llm.tokenizer.Tokenizer.initFromParts(alloc, &vocab, &merges, .{});
    defer tok.deinit();

    const ids = try tok.encode(alloc, "hi<|im_end|>");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 3, 0 }, ids); // "hi" merged, marker resolved

    const text = try tok.decode(alloc, ids);
    defer alloc.free(text);
    try std.testing.expectEqualStrings("hi<|im_end|>", text);
}
```

`StreamDecoder` handles token-by-token generation where one token can end in
the middle of a multi-byte UTF-8 character:

```zig
pub const StreamDecoder = struct {
    pub fn init(tokenizer: *const Tokenizer) StreamDecoder
    pub fn deinit(self: *StreamDecoder, allocator: Allocator) void
    pub fn reset(self: *StreamDecoder) void
    pub fn push(self, allocator, id: u32, writer: *std.Io.Writer) !void
    pub fn flush(self, writer: *std.Io.Writer) !void
};
```

`push` emits only the complete-UTF-8 prefix and holds the incomplete tail
until a later token finishes it; `flush` emits any remainder when generation
ends. The sink is any `*std.Io.Writer` (stdout, an SSE response, an in-memory
buffer).

#### 13.5.2 SentencePiece (`src/llm/spm_tokenizer.zig`)

`llm.spm_tokenizer.Tokenizer` is the Gemma-family counterpart: a faithful port
of llama.cpp's `llm_tokenizer_spm` Unigram model, driven by per-token
**scores** rather than merge ranks. Encoding seeds a max-heap of adjacent
symbol pairs keyed by the score of the token they would form, repeatedly
merges the highest-scoring pair, resegments, and byte-falls-back to `<0xXX>`
tokens for anything the vocabulary cannot cover. Special/control tokens
(`<start_of_turn>`, `<bos>`, …) are partitioned out of the raw text first
(longest marker wins), so they map to single ids. Error set:
`error{ NoTokenizerVocab, UnsupportedTokenizerFormat, TokenizerArrayTooShort, TokenizerTooLarge } || Allocator.Error`.

```zig
pub const Attr = enum(i32) { undef, normal, unknown, control, user_defined, unused, byte, _ };
pub const Options = struct {
    bos: ?u32 = null, eos: ?u32 = null, unk: ?u32 = null,
    add_bos: ?bool = null, add_eos: ?bool = null, add_space_prefix: ?bool = null,
};
pub fn initFromGguf(allocator, file: *const gguf.File, overrides: Options) !Tokenizer
pub fn initFromSlices(allocator, vocab_strings: []const []const u8,
    scores: []const f32, attrs: ?[]const Attr, opts: Options) !Tokenizer
```

`initFromGguf` requires `tokenizer.ggml.scores` (its absence means a byte-BPE
vocab → `Error.UnsupportedTokenizerFormat`); `tokenizer.ggml.token_type` is
optional (absent = every token NORMAL). Defaults follow llama.cpp's SPM
defaults when metadata is silent: `bos=1, eos=2, unk=0, add_bos=true,
add_eos=false, add_space_prefix=true`. `Attr` mirrors llama.cpp's
`LLAMA_TOKEN_TYPE_*` numbering and controls encode partitioning and decode
rendering (NORMAL unescapes `▁`, BYTE emits the raw byte, CONTROL/UNKNOWN are
suppressed). The public shape matches the byte-BPE tokenizer — `encode`,
`encodeRaw`, `decode`, `decodeAppend`, `tokenId`, `vocabSize`, `eosId`,
`bosId`, `isEos`, `deinit`, and an identical `StreamDecoder` — so a runner
picks one module per architecture and the rest of the stack (chat, data) is
generic over either.

```zig
test "SPM: score-driven merges and byte fallback" {
    const alloc = std.testing.allocator;
    const vocab = [_][]const u8{ "<unk>", "a", "b", "ab", "abc", "c", "<0x78>" };
    const scores = [_]f32{ 0, -1, -1, -3, -2.5, -1, -5 };
    var tok = try llm.spm_tokenizer.Tokenizer.initFromSlices(alloc, &vocab, &scores, null, .{
        .add_bos = false,
        .add_space_prefix = false,
    });
    defer tok.deinit();

    const ids = try tok.encode(alloc, "abcx"); // "abc" outscores "ab"; 'x' byte-falls-back
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 4, 6 }, ids);
}
```

#### 13.5.3 Unicode tables (`src/llm/unicode_categories.zig`)

Generated (do not edit) `\p{L}`/`\p{N}`/`\s` classification tables
(`isLetter`/`isNumber`/`isWhitespace`) matching llama.cpp's tokenizer data for
token-ID-exact pretokenizer parity; regenerate with
`python3 tools/gen_unicode_categories.py > src/llm/unicode_categories.zig`
(the generator writes to stdout). Re-exported from `llm.zig` as
`llm.unicode_categories` so out-of-module consumers (nanochat's
example-local tokenizer) share the tables.

### 13.6 Sampling (`src/llm/sampler.zig`)

```zig
pub const Config = struct {
    temperature: f32 = 0,        // <= 0 selects greedy (argmax)
    top_k: usize = 0,            // 0 = internal cap of 256 candidates
    top_p: f32 = 1.0,            // nucleus: smallest prefix with cum. prob >= top_p
    min_p: f32 = 0,              // keep tokens with p >= min_p * p(best); 0 disables
    repeat_penalty: f32 = 1.0,   // llama.cpp penalty_repeat; 1.0 disables
    freq_penalty: f32 = 0,       // llama.cpp penalty_freq (per-count subtraction)
    presence_penalty: f32 = 0,   // llama.cpp penalty_present (flat subtraction)
    repeat_last_n: usize = 64,   // penalty window over the most recent tokens
    seed: u64 = 0,
    pub fn isGreedy(self: Config) bool;  // temperature <= 0
};

pub const Sampler = struct {
    processor: ?LogitProcessor = null,   // optional pre-sampling logit transform
    pub fn init(config: Config) Sampler;
    pub fn next(self: *Sampler, ctx: *ExecContext,
                logits: *fucina.Tensor(.{ .seq, .vocab }),   // shape [1, vocab]
                history: []const usize) !usize;
};
```

`next` implements the llama.cpp-compatible pipeline in order: logit
processor → penalties → greedy shortcut → top-k truncation → temperature
softmax → top-p → min-p → categorical draw. Semantics worth pinning:

- **Penalties mutate `logits` in place**, applied once per unique token in the
  last `repeat_last_n` tokens of `history` (with `count` = occurrences in the
  window, matching `llama_sampler_penalties`).
- With `temperature <= 0` (the default) the call is a deterministic argmax and
  the RNG is never touched — benchmarking and greedy decode share the path.
- Sampling uses a `std.Random.DefaultPrng` seeded once from `config.seed` at
  `init`: the draw sequence is a pure function of the seed, which the chat and
  speculative layers rely on (one draw per committed token, §13.8/§13.9).
  The candidate set is capped at 256 even when `top_k = 0`.
- A `Sampler` is single-stream mutable state (RNG + config): not thread-safe,
  one per decode stream.
- With a `processor` set, its `process` hook mutates the logits row before
  everything else and its `commit` hook observes the selected token on every
  path (greedy included, exactly once per `next`); a processor that masks out
  every candidate is `error.AllTokensMasked`.

```zig
test "sampler: greedy default, seed-deterministic sampling" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var logits = try fucina.Tensor(.{ .seq, .vocab }).fromSlice(&ctx, .{ 1, 5 }, &.{ 0.1, 0.2, 0.9, 0.3, 0.0 });
    defer logits.deinit();

    var greedy = llm.sampler.Sampler.init(.{}); // temperature 0 => argmax
    try std.testing.expectEqual(@as(usize, 2), try greedy.next(&ctx, &logits, &.{}));

    var a = llm.sampler.Sampler.init(.{ .temperature = 0.8, .top_k = 3, .seed = 42 });
    var b = llm.sampler.Sampler.init(.{ .temperature = 0.8, .top_k = 3, .seed = 42 });
    for (0..8) |_| { // same seed -> same draw sequence
        try std.testing.expectEqual(try a.next(&ctx, &logits, &.{}), try b.next(&ctx, &logits, &.{}));
    }
}
```

#### Logit processors (`src/llm/logit_processor.zig`)

`LogitProcessor` is the injectable pre-sampling transform — the seam
grammar-constrained decoding plugs into, and the hook for any custom logit
policy (bias lists, banned-token rules, watermarking). It follows the
`DraftSource` vtable pattern (§13.9):

```zig
pub const LogitProcessor = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        process: *const fn (ptr, logits: []f32, history: []const usize) anyerror!void,
        commit: *const fn (ptr, token: usize) anyerror!void,
        reset: ?*const fn (ptr) anyerror!void = null,
        // structural hooks (optional; pure deterministic lookahead):
        forcedTokens: ?*const fn (ptr, buf: []usize) usize = null,
        validPrefixLen: ?*const fn (ptr, tokens: []const usize) usize = null,
    };
    pub fn process(...) / commit(...) / reset(...)
    pub fn hasStructure(self) bool  // both structural hooks present
    pub fn forcedTokens(self, buf: []usize) usize / validPrefixLen(self, tokens) usize
};
```

`process` mutates one `[vocab]` logits row in place before the sampler's own
pipeline (a mask writes `-inf` over forbidden tokens); `commit` observes the
selected token, exactly once per `Sampler.next`; the optional `reset` re-arms
state for a fresh constrained region (`chat.Conversation` calls it at every
turn start). Because the seam lives inside the `Sampler`, every decode path
that samples through one — `chat.send`/`sendBatch`, the speculative
decoder's plain and verify steps, hand-rolled runner loops — picks the
processor up with no loop changes.

The two **structural hooks** let a processor expose what its state machine
knows beyond a mask: `forcedTokens` writes the unique legal continuation
(grammar-mandated JSON punctuation, a forced literal) and `validPrefixLen`
reports how many leading tokens of a candidate sequence the state accepts.
Both must be deterministic pure lookaheads. When present
(`hasStructure()`), the speculative layer turns them into drafts —
§13.9's `ConstrainedSource`.

**The seam is speculative-safe by construction**: the verify loop samples
each row only after that row's prefix is committed, and every sampled row
token is itself committed (accepted draft, correction, or bonus — §13.9), so
`commit` keeps processor state exactly in step with history and no rollback
hook is needed. A draft token the mask forbids simply loses the
`sampled == draft` comparison and is rejected; the constrained speculative
stream is token-for-token identical to the constrained plain stream (proven
greedy + sampled in `chat_tests.zig`). One processor per decode stream, like
the sampler that hosts it.

```zig
test "logit processor: mask before sampling, observe the selection" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const OddMask = struct {
        commits: usize = 0,
        fn process(ptr: *anyopaque, logits: []f32, history: []const usize) anyerror!void {
            _ = ptr;
            _ = history;
            for (logits, 0..) |*l, tok| {
                if (tok % 2 == 1) l.* = -std.math.inf(f32);
            }
        }
        fn commit(ptr: *anyopaque, token: usize) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            _ = token;
            self.commits += 1;
        }
    };
    var mask = OddMask{};

    var logits = try fucina.Tensor(.{ .seq, .vocab }).fromSlice(&ctx, .{ 1, 4 }, &.{ 0.1, 0.9, 0.5, 0.8 });
    defer logits.deinit();
    var s = llm.sampler.Sampler.init(.{}); // greedy
    s.processor = .{ .ptr = &mask, .vtable = &.{ .process = OddMask.process, .commit = OddMask.commit } };
    // Unmasked argmax would be token 1; the mask forces the best even id.
    try std.testing.expectEqual(@as(usize, 2), try s.next(&ctx, &logits, &.{}));
    try std.testing.expectEqual(@as(usize, 1), mask.commits);
}
```

#### Constrained decoding: llguidance (`src/llm/llguidance.zig`, `-Dllguidance`)

`llm.llguidance.Constraint` compiles a grammar with the vendored
[llguidance](https://github.com/guidance-ai/llguidance) engine
(`vendor/llguidance`, MIT — version/update procedure in its
[README](../vendor/llguidance/README.md)) and adapts it to the
`LogitProcessor` seam: JSON-schema/regex/Lark-constrained generation for any
runner built on the shared sampler, ~50 µs of pure CPU mask work per token.
Requires `-Dllguidance=true` (§2.2; cargo builds the Rust staticlib); without
it the module still compiles and `Constraint.init` returns
`error.LlguidanceNotEnabled`.
[CONSTRAINED-DECODING.md](CONSTRAINED-DECODING.md) is the full design record
(seam adjudication, tokenizer-bridge details, the no-rollback speculation
argument, measured results).

```zig
pub const enabled: bool;                 // build-flag mirror
pub fn version() []const u8;             // "llguidance@X.Y.Z derivre@..."

pub const Grammar = union(enum) {
    json_schema: []const u8,  // stringified JSON schema
    regex: []const u8,        // Rust-syntax regex the reply must match
    lark: []const u8,         // llguidance's Lark-variant grammar
    llguidance: []const u8,   // composite JSON list form
};
pub const Options = struct {
    eos_token: ?u32 = null,      // forced when the grammar completes; default tokenizer eosId()
    extra_eos: []const u32 = &.{},
    n_vocab: ?usize = null,      // model vocab when padded larger than the tokenizer's
    log_level: u32 = 1,          // 0 silent, 1 warnings, 2 info
};

pub const Constraint = struct {
    pub fn init(allocator, tokenizer: anytype, grammar: Grammar, options: Options) Error!Constraint
    pub fn deinit(self: *Constraint) void
    pub fn clone(self: *const Constraint) Error!Constraint  // independent per-stream twin
    pub fn processor(self: *Constraint) LogitProcessor  // install on a Sampler / chat.Options
    pub fn isStopped(self: *const Constraint) bool      // grammar terminated
    pub fn isAccepting(self: *Constraint) bool          // tokens so far form a complete sentence
    pub fn reset(self: *Constraint) Error!void          // re-arm for a fresh reply
    pub fn ffTokens(self: *Constraint, buf: []u32) Error!usize // grammar-forced continuation
};
```

- `tokenizer` is `*const llm.tokenizer.Tokenizer` (byte-BPE) or
  `*const llm.spm_tokenizer.Tokenizer` (SPM) — both borrowed. The bridge
  hands llguidance every token's RAW bytes: BPE tokens byte-decoded, SPM
  pieces unescaped (`▁` → space) and `<0xXX>` byte tokens as their byte.
  Control tokens (BPE: the `<|...|>` marker shape; SPM: `control`/`unknown`
  attrs) carry toktrie's `0xFF` special marker, so a grammar whose text could
  spell `<|im_end|>` can never steer the model into emitting the actual
  control token. Padding ids past the tokenizer vocab (set
  `n_vocab = config.vocab_size`) get empty bytes and are never allowed.
- **Stop forcing**: when the grammar completes, the mask allows only
  `eos_token` — pass the chat template's stop-marker id so a finished grammar
  ends the turn through the existing stop handling; a matcher failure
  mid-decode also degrades to the forced stop (details logged at
  `log_level >= 1`). An invalid grammar fails `init` loudly instead.
- One `Constraint` per decode stream; do not move it after `processor()` is
  taken. `chat.Conversation` re-arms it per turn via the `reset` hook.
  Multi-stream decode (`sendBatch`, `--streams`) gives each stream a
  `clone()` — a deep-cloned matcher over the refcounted tokenizer with the
  tokenize bridge borrowed from the original (which must outlive the
  clones); no vocab rebuild or grammar recompilation.
- The processor exposes the §13.6 structural hooks (`forcedTokens` /
  `validPrefixLen`, backed by llguidance's fast-forward and
  token-validation lookaheads), so speculation routes it through the
  grammar-aware draft source automatically (§13.9): grammar-forced spans
  draft themselves and are accepted with certainty. Measured on
  Qwen3-0.6B-Q8_0 with the JSON-schema example below: 0% draft acceptance
  (cost gate mutes speculation) without the hooks, 83% acceptance at
  1.24 tok/step with them — output byte-identical either way.
- The runner flags (qwen3 + gemma4, §14.2/§14.4): `--json-schema JSON|@FILE`,
  `--lark GRAMMAR|@FILE`, `--regex PATTERN` — combine with `--no-think` on
  reasoning models (the grammar governs the whole reply, thinking channel
  included). Composes with `--spec` (output identical to the plain run) and
  with qwen3's `--streams` (per-stream clones; batch == sequential
  token-for-token).

```zig
test "llguidance: JSON-schema constrained greedy decode" {
    if (!llm.llguidance.enabled) return error.SkipZigTest; // -Dllguidance=true builds only
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const vocab = [_][]const u8{ "{", "}", "\"", "a", ":", "1", "<|end|>" };
    var tok = try llm.tokenizer.Tokenizer.initFromParts(alloc, &vocab, &.{}, .{ .eos = 6 });
    defer tok.deinit();

    var constraint = try llm.llguidance.Constraint.init(alloc, &tok, .{
        .json_schema =
        \\{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}
    }, .{});
    defer constraint.deinit();

    var s = llm.sampler.Sampler.init(.{}); // greedy
    s.processor = constraint.processor();

    // The model "wants" '}' everywhere; the mask walks it through a valid
    // object instead — '}' only becomes samplable once {"a":1 is complete —
    // then the finished grammar forces the stop token.
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    var steps: usize = 0;
    while (!constraint.isStopped() and steps < 16) : (steps += 1) {
        var logits = try fucina.Tensor(.{ .seq, .vocab }).fromSlice(&ctx, .{ 1, 7 }, &.{ 0, 1, 0, 0, 0, 0.5, 0 });
        defer logits.deinit();
        const next = try s.next(&ctx, &logits, &.{});
        if (next == 6) break;
        try tok.decodeAppend(alloc, @intCast(next), &out);
    }
    try std.testing.expectEqualStrings("{\"a\":1}", out.items);
}
```

### 13.7 SFT data (`src/llm/data.zig`)

Minimal supervised-fine-tuning helpers, generic across model families. Error
set: `error{ MalformedJsonl, SampleTooLong, EmptyDataset, InvalidLoaderState }`.

```zig
pub const Pair = struct { instruction: []const u8, response: []const u8 };
pub const SftText = struct {
    pairs: []const Pair,
    blob: ?[]u8 = null,
    pub const JsonlOptions = struct {
        instruction_key: []const u8 = "instruction",
        response_key: []const u8 = "response",
    };
    pub fn fromPairs(pairs: []const Pair) SftText;                    // zero-copy borrow
    pub fn fromJsonl(allocator, io: std.Io, path: []const u8, opts: JsonlOptions) !SftText;
    pub fn deinit(self: *SftText, allocator: Allocator) void;
};
```

`fromPairs` borrows caller-owned pairs (`deinit` frees nothing). `fromJsonl`
loads one JSON object per line, reading strings under the configured keys into
one owned blob (so the result outlives the file); blank lines are skipped, and
any malformed line fails with `Error.MalformedJsonl` after logging the path
and line number.

```zig
pub const Sample = struct {
    inputs: []usize,   // full sequence minus its final token
    labels: []usize,   // next-token shift; prompt positions masked
    pub fn deinit(self: *Sample, allocator: Allocator) void;
};
pub const EncodeOptions = struct {
    seq_max: usize = 256,
    ignore_index: usize = std.math.maxInt(usize),  // the trainer's mask sentinel
    mask_prompt: bool = true,
    system: ?[]const u8 = null,
    think_off: bool = true,
};
pub fn encodePrompt(allocator, tokenizer: anytype, template: chat.Template,
                    instruction: []const u8, opts: EncodeOptions) ![]usize
pub fn encodePair(allocator, tokenizer: anytype, template: chat.Template,
                  pair: Pair, opts: EncodeOptions) !Sample
```

`encodePair` renders one user turn through the chat template, tokenizes, and
builds the shifted training pair: `inputs` = prompt ++ response tokens minus
the last; `labels[i-1]` = token `i`, with all prompt positions replaced by
`opts.ignore_index` unless `mask_prompt` is off. The response (plus the
template's stop marker) is encoded **separately** from the prompt —
concatenating the text first would move BPE chunk boundaries across the join
and change token ids. Samples are truncated to `seq_max` input positions; a
window that leaves no supervised token is `Error.SampleTooLong`. The
`tokenizer` parameter is duck-typed over `encodeRaw` — byte-BPE and SPM both
satisfy it — and `ignore_index` is injected so this module never imports a
trainer (§11).

```zig
test "encodePair: render + tokenize + shift + prompt mask" {
    const alloc = std.testing.allocator;
    const vocab = [_][]const u8{
        "<|im_start|>", "<|im_end|>", "u", "s", "e", "r", "a", "n", "t", "i",
        "h",            "k",          "y", "o", "m", "<", ">", "/", "\xC4\x8A", // Ċ = byte-level '\n'
    };
    var tok = try llm.tokenizer.Tokenizer.initFromParts(alloc, &vocab, &.{}, .{});
    defer tok.deinit();

    const chatml = llm.chat.Template{ .format = .chatml };
    var sample = try llm.data.encodePair(alloc, &tok, chatml, .{
        .instruction = "hi",
        .response = "yo",
    }, .{ .seq_max = 64, .ignore_index = 9999 });
    defer sample.deinit(alloc);

    try std.testing.expectEqual(sample.inputs.len, sample.labels.len);
    try std.testing.expectEqual(@as(usize, 9999), sample.labels[0]); // prompt masked
    try std.testing.expectEqual(@as(usize, 1), sample.labels[sample.labels.len - 1]); // stop marker supervised
}
```

```zig
pub const Loader = struct {
    pub const Order = enum { sequential, shuffled };
    pub const State = struct { seed: u64, epoch: u64, index: u64 };
    pub fn init(allocator, n: usize, order: Order, seed: u64) !Loader;   // n == 0 => EmptyDataset
    pub fn deinit(self: *Loader, allocator: Allocator) void;
    pub fn next(self: *Loader) usize;
    pub fn state(self: *const Loader) State;
    pub fn restore(self: *Loader, s: State) !void;   // out-of-range index => InvalidLoaderState
};
```

`Loader` is a deterministic sample-order iterator. `.sequential` is plain
round-robin; `.shuffled` draws each epoch as a fresh permutation that is a
**pure function of `(seed, epoch)`** — a checkpoint contract: the permutation
is identity order followed by a Fisher–Yates pass driven by a splitmix64
stream seeded with `rng.at(seed, epoch)` (`j = splitmix64 % (i+1)` for `i`
from `n-1` down to `1`). The formula is golden-pinned in `data_tests.zig` and
may never change once checkpoints exist against it; `restore` regenerates the
exact stream position from a saved `State` (u64 fields, so it round-trips
through `trainer_state.json` unchanged — §11).

```zig
test "Loader: (seed, epoch) -> permutation is a checkpoint contract" {
    const alloc = std.testing.allocator;
    var loader = try llm.data.Loader.init(alloc, 8, .shuffled, 42);
    defer loader.deinit(alloc);
    // Golden-pinned: this exact order may never change once checkpoints exist.
    try std.testing.expectEqualSlices(usize, &.{ 3, 6, 0, 7, 1, 2, 5, 4 }, loader.perm);

    for (0..3) |_| _ = loader.next();
    const s = loader.state();
    var expect: [8]usize = undefined;
    for (&expect) |*e| e.* = loader.next(); // crosses the epoch boundary

    var replay = try llm.data.Loader.init(alloc, 8, .shuffled, 0);
    defer replay.deinit(alloc);
    try replay.restore(s); // seed/epoch/index come from the saved State
    for (expect) |want| try std.testing.expectEqual(want, replay.next());
}
```

### 13.8 Chat (`src/llm/chat.zig`)

#### 13.8.1 Templates

```zig
pub const Format = enum { chatml, llama3, gemma, gemma4 };
pub const Template = struct {
    format: Format,
    pub fn detect(chat_template: ?[]const u8) ?Template;
    pub fn stopMarker(self: Template) []const u8;
    pub fn renderTurn(self, allocator, buf: *std.ArrayList(u8),
        system: ?[]const u8, user: []const u8, first: bool, think_off: bool) !void;
};
```

`detect` sniffs the format from a GGUF `tokenizer.chat_template` string
(`<|im_start|>` → chatml, `<|start_header_id|>` → llama3, `<|turn>` → gemma4,
`<start_of_turn>` → gemma; anything else → `null`). `stopMarker` is the token
text that ends an assistant turn (`<|im_end|>`, `<|eot_id|>`,
`<end_of_turn>`, `<turn|>`). `renderTurn` appends the text to feed for one
user turn: `first` emits the conversation-start (bos/system) scaffolding,
otherwise it first closes the previous assistant turn; `think_off` suppresses
the reasoning channel on ChatML (empty `<think>` block) and Gemma 4
(primed-empty thought channel). Gemma 1–3 has no system role — the system
prompt is folded into the first user turn.

```zig
test "chat template: detect from GGUF metadata, render a turn" {
    const alloc = std.testing.allocator;
    const t = llm.chat.Template.detect("... {{ '<|im_start|>' }} ...").?;
    try std.testing.expectEqual(llm.chat.Format.chatml, t.format);
    try std.testing.expectEqualStrings("<|im_end|>", t.stopMarker());

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    try t.renderTurn(alloc, &buf, "Be terse.", "Hi", true, false);
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        buf.items,
    );
}
```

`renderMessages` is `renderTurn`'s stateless twin: it renders a FULL message
history for a fresh conversation, ending with the assistant-turn opener — the
shape a messages-array API server receives on every request (the lmserve
example, `examples/lmserve/`).

```zig
pub const Message = struct {
    role: Role, // enum { system, user, assistant }
    content: []const u8, // borrowed
};
pub fn renderMessages(self, allocator, buf: *std.ArrayList(u8),
    messages: []const Message, think_off: bool) !void;
```

ChatML and Llama 3 render every message as its own role block, any order.
The Gemma formats have a single conversation-start system slot: leading
system messages merge into it (Gemma 1–3: folded into the first user turn),
and a later one is `error.SystemMidConversation`. An empty list is
`error.EmptyMessages`; a trailing assistant message is
`error.TrailingAssistantMessage` (rendering would open a SECOND assistant
turn after it rather than continue it). Historical assistant contents have
their reasoning block stripped (ChatML `<think>…</think>`, Gemma 4
`<|channel>thought…<channel|>`) — the reference templates drop prior-turn
reasoning, and stateless clients replay content without it. First-turn output
is byte-identical to `renderTurn`'s, so a stateless render prefills the same
KV prefix as an incrementally driven conversation.

```zig
test "chat template: render a full message history (stateless server shape)" {
    const alloc = std.testing.allocator;
    const t = llm.chat.Template{ .format = .chatml };
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    try t.renderMessages(alloc, &buf, &.{
        .{ .role = .system, .content = "Be terse." },
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "<think>\nhm\n</think>\n\nHello!" },
        .{ .role = .user, .content = "Bye" },
    }, true);
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nBe terse.<|im_end|>\n" ++
            "<|im_start|>user\nHi<|im_end|>\n" ++
            "<|im_start|>assistant\nHello!<|im_end|>\n" ++ // <think> stripped
            "<|im_start|>user\nBye<|im_end|>\n" ++
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        buf.items,
    );
}
```

#### 13.8.2 `Conversation(Model, Tok)`

```zig
pub fn Conversation(comptime Model: type, comptime Tok: type) type
```

Comptime-generic multi-turn chat over a model family and a tokenizer module.
The duck-typed contract:

- `Model` exposes `config.vocab_size`,
  `initKvCache(ctx, capacity) !KvCache` (over the shared §13.4 cache), and the
  decode entries with the qwen3/gemma4 signatures:
  `forwardStep(ctx, kv, token_ids, pos0) !Tensor(.{ .seq, .vocab })`
  (last-token logits) and `forwardStepAllLogits` (same signature, all-row
  logits). The latter is a hard compile-time requirement even with
  speculation permanently off — `send` unconditionally references the
  speculative path, so a `Model` without it fails to instantiate; it is
  only *executed* when speculation is enabled. `sendBatch`
  additionally requires
  `forwardStepBatch(ctx, caches: []const *KvCache, token_ids: []const usize)`;
  the requirement is comptime-gated, so families without it (gemma4 today)
  still instantiate the type and get `error.BatchDecodeUnsupported` at
  runtime.
- `Tok` is the tokenizer **module** (`llm.tokenizer` or `llm.spm_tokenizer`):
  it must provide a `Tokenizer` type with
  `tokenId`/`eosId`/`encodeRaw`/`decodeAppend` and a `StreamDecoder`.

```zig
pub const Options = struct {
    system: ?[]const u8 = null,
    capacity: usize = 4096,               // total KV size; the whole conversation must fit
    max_response_tokens: usize = 1024,    // per-reply cap
    think_off: bool = false,
    sampler: sampler.Config = .{},
    extra_stop_ids: []const u32 = &.{},   // borrowed
    stop_sequences: []const []const u8 = &.{},  // borrowed; incompatible with speculation
    logit_processor: ?sampler.LogitProcessor = null,  // borrowed; §13.6
    speculation: bool = false,
    spec_options: speculative.Options = .{},
    io: ?std.Io = null,                   // clock for the decoder's live cost gate
};

pub fn init(ctx: *ExecContext, model: *const Model, tokenizer: *const Tok.Tokenizer,
            template: Template, options: Options) !Self
pub fn deinit(self: *Self) void
pub fn send(self: *Self, user: []const u8, writer: *std.Io.Writer) !usize
pub fn sendRendered(self: *Self, rendered: []const u8, writer: *std.Io.Writer) !usize
pub fn sendBatch(convos: []const *Self, users: []const []const u8,
                 writers: []const *std.Io.Writer, produced: []usize) !void
pub fn addSpecReference(self: *Self, tokens: []const usize) !void
pub fn enablePersistence(self: *Self, io: std.Io, path: []const u8) !usize
pub fn specStats(self: *const Self) ?speculative.Stats
```

Semantics:

- `init` resolves the stop id as `tokenizer.tokenId(template.stopMarker())
  orelse tokenizer.eosId()`, builds the KV cache via the model's own
  `initKvCache`, and — with `speculation` on — heap-allocates the speculative
  state (a `SpeculationIndex` cascade plus a `SpeculativeDecoder(Model)`,
  §13.9), wiring the stop id into `spec_options.stop_token` and aligning the
  cascade's `accounting_min_draft` with the decoder's `min_draft`.
  **`stop_sequences` plus `speculation` fails loudly with
  `error.StopSequencesWithSpeculation`**: the token completing a text stop
  sequence could be accepted mid-verify-batch, breaking the
  one-RNG-draw-per-committed-token contract. The `Conversation` borrows `ctx`,
  `model`, `tokenizer`, `system`, `extra_stop_ids`, and `stop_sequences` —
  they must outlive it; `deinit` releases the cache, history, stream decoder,
  and speculative state.
- `send` renders the turn, tokenizes it (`error.ContextFull` if the prefix
  does not fit the remaining KV capacity), prefills at the current cache
  position (one KV cache persists across turns — each turn prefills only the
  new tokens), then decodes token by token: sample → stop check → stream
  through the `StreamDecoder` (flushed per token) → forward. It returns the
  number of response tokens produced. A turn ends on the template stop marker,
  any of `extra_stop_ids`, the `max_response_tokens` budget, or KV exhaustion.
  With `stop_sequences`, generation stops **before streaming** the token whose
  decoded reply text completes a sequence (the completing token is not
  committed).
- `sendRendered` is `send` over caller-provided pre-rendered template text —
  the stateless-API entry: `Template.renderMessages` renders a full message
  history and a FRESH conversation prefills it in one turn (`Options.system`
  is not consulted; the caller rendered everything). Streaming, stop
  handling, speculation, and the logit processor behave exactly as in `send`;
  the equivalence with an incrementally driven conversation is proven in
  `chat_tests.zig`.
- With speculation on, `send` routes through the decoder with a turn-boundary
  gate that stops streaming/index-learning at the stop marker and trims any
  verify-batch overshoot from history **and** the KV cache — unconditionally,
  on error paths included — so the post-turn state matches the plain path's
  exactly. The equivalence (token-for-token, draw-for-draw across a persistent
  sampler, greedy and sampled) is proven in `chat_tests.zig`.
- `logit_processor` installs a §13.6 processor (e.g. a `llm.llguidance`
  grammar constraint) on the conversation's sampler and re-arms it via its
  `reset` hook at every turn start, so the same constraint governs each
  assistant reply independently — on the plain, speculative, and `sendBatch`
  paths alike (constrained plain == constrained speculative is part of the
  `chat_tests.zig` equivalence proofs). When the processor exposes the
  structural hooks (`hasStructure()`), speculation automatically routes
  through the grammar-aware draft source (13.9.6): forced grammar spans
  draft themselves. `sendBatch` requires one processor instance per stream
  (a shared pointer is `error.SharedBatchProcessor` — single-stream state;
  llguidance streams use `Constraint.clone`).
- `sendBatch` decodes one message on each of N sibling conversations in
  lockstep: every step forwards one token per live stream through
  `forwardStepBatch` (one m=N weight pass instead of N GEMVs), then samples
  each stream from its own logits row with its own sampler/history. Per-stream
  semantics match a plain `send` exactly; below the m-dependent kernel
  thresholds the produced tokens are bit-identical to N sequential sends,
  beyond them rows can differ by ~1e-6 reassociation drift. Requirements,
  checked up front: non-empty batch (`error.EmptyBatch`), equal slice lengths
  (`error.BatchLengthMismatch`), speculation off on every stream
  (`error.SpeculationWithBatch`), one shared `ctx`/`model`
  (`error.MixedBatchModels`), distinct conversations
  (`error.DuplicateBatchConversation`). Ownership contract on error: the
  batch aborts, `produced` is left unwritten, and **every** stream's history
  is trimmed back to its KV cache, so healthy siblings of the failing stream
  remain internally consistent and resendable; bytes already streamed are not
  recalled. Turn prefills run per stream.
- `addSpecReference(tokens)` injects a tokenized reference document into the
  speculation index (the RAG seam); `error.SpeculationDisabled` when
  speculation is off. `specStats` returns the decoder's lifetime
  `speculative.Stats` (null when off).
- `enablePersistence(io, path)` arms KV persistence (`kv_persist.zig`) on a
  fresh conversation — once, before any send. A compatible saved conversation
  at `path` resumes into it (token history and KV cache restored, zero
  re-prefill); otherwise the file is reset so a stale or foreign prefix
  cannot become this conversation's. It returns the number of resumed
  positions (0 = fresh start). Every subsequent `send`/`sendRendered` turn
  appends its new positions to the append-only sidecar — the record count is
  published last, so a crash mid-append leaves a consistent prefix.
- A `Conversation` is single-threaded mutable state; `sendBatch` runs on the
  caller's thread over all streams.

```zig
fn snippetConversation(ctx: *fucina.ExecContext, io: std.Io, out: *std.Io.Writer) !void {
    const alloc = ctx.allocator;
    var file = try fucina.gguf.File.loadMmap(alloc, io, "qwen3-0.6b.gguf");
    defer file.deinit();
    var model = try llm.qwen3.model.Model.loadGgufFromFile(ctx, &file, try llm.qwen3.model.Config.fromGguf(&file));
    defer model.deinit();
    var tok = try llm.tokenizer.Tokenizer.initFromGguf(alloc, &file, .{});
    defer tok.deinit();
    const template = llm.chat.Template.detect(file.getString("tokenizer.chat_template")) orelse
        llm.chat.Template{ .format = .chatml };

    const Convo = llm.chat.Conversation(llm.qwen3.model.Model, llm.tokenizer);
    var convo = try Convo.init(ctx, &model, &tok, template, .{
        .capacity = 4096,
        .sampler = .{ .temperature = 0.7, .top_k = 20, .seed = 42 },
        .speculation = true, // lossless draft-free speculative decoding
    });
    defer convo.deinit();
    _ = try convo.send("Why is the sky blue?", out); // streams tokens to `out`
} // requires model assets to run
```

### 13.9 Speculative decoding (`src/llm/speculative/`)

Training-free, **draft-model-free** speculative decoding: drafts come from
cheap deterministic indexes over text the model has already seen — no extra
weights. [`SPECULATIVE.md`](SPECULATIVE.md) is the full design record (proof
obligations, verify economics, adjudicated alternatives); this section covers
the public surface.

**The lossless contract** (`core.zig` header is normative): because every
draft source is deterministic (a one-hot proposal distribution), rejection
sampling degenerates to running the FULL sampling pipeline on the target
logits at each verified position, conditioned on the hypothetical prefix —
accept while `sampled == draft[i]`; at the first mismatch the sampled token
IS the correction token; on full acceptance the (k+1)-th row yields a free
bonus token. Greedy is the same code path (temperature ≤ 0 makes the sampler
an argmax). Token IDs are compared, never probabilities. Exactly **one RNG
draw is consumed per committed token** — the same pattern as a plain run —
and committing `Options.stop_token` ends the verify row loop immediately, so
a persistent sampler never desyncs. Given bitwise-identical logits the output
stream equals the non-speculative run's; logits are computed in verify
batches of m = 1+draft rows, which match bitwise below the m-dependent kernel
thresholds and can drift ~1e-6 beyond them ("same distribution always; same
sample stream whenever the logits match bitwise").

#### 13.9.1 Core (`speculative/core.zig`)

```zig
pub const TopKRow = struct { token: usize, topk: []const u32 };  // borrowed per call

pub const DraftSource = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        suggest: *const fn (ptr, context: []const usize, buf: []usize) usize,
        observe: *const fn (ptr, committed: []const usize) void,
        observeTopK: ?*const fn (ptr, positions: []const TopKRow) void = null,
        truncatePending: ?*const fn (ptr, new_len: usize) void = null,
    };
    pub fn suggest(...) usize / observe(...) void / observeTopK(...) void
    pub fn wantsTopK(self) bool
    pub fn truncatePending(self, new_len: usize) void
};
```

`DraftSource` is the injectable proposer interface: `suggest` writes up to
`buf.len` continuation tokens for the committed `context` (0 = no draft),
`observe` feeds newly committed tokens back, and the optional `observeTopK`
receives per-position top-K candidates from the verification logits — when
null, the decoder skips computing the top-k entirely. Sources must be
deterministic; the decoder clamps a lying `suggest` return value at runtime.
The optional `truncatePending` lets a wrapper (the chat turn-boundary gate,
the 13.9.6 grammar filter) tell the source its just-returned draft was
shortened, so pending acceptance accounting shrinks to the prefix the
decoder will actually verify.

```zig
pub const Options = struct {
    max_draft: usize = 16,
    min_draft: usize = 2,           // shorter drafts fall back to a plain step
    enabled: bool = true,
    topk_feedback: usize = 8,       // candidates per verified position
    stop_token: ?usize = null,
    // cost-aware auto-off gate:
    rate_window: usize = 16, min_window_drafted: usize = 8,
    min_speedup: f32 = 1.0, probe_margin: f32 = 0.10,
    reprobe_after: usize = 128, reprobe_max: usize = 1024, probe_steps: usize = 4,
    cost_table: []const CostPoint = &default_cost_table,
    adapt_budget: bool = true,
};
pub const CostPoint = struct { draft_len: usize, cost: f32 };
pub const default_cost_table: [4]CostPoint;   // measured Qwen3-0.6B-Q4_K_S economics
pub fn tableCost(table: []const CostPoint, draft_len: usize) f32;
```

`CostGate` (public type; driven internally by the decoder, `estSpeedup()` is
its public read) estimates the rolling true speedup — committed tokens per
plain-step-equivalent of verify cost — over a window of verify steps, turns
speculation off when it drops below `min_speedup`, re-probes with exponential
backoff (`reprobe_after` → `reprobe_max`) and `probe_margin` hysteresis, and
adapts the draft budget to the rolling acceptance rate. With a clock
(`io` set on the decoder), measured verify/plain ratios continuously rescale
the static `cost_table` through a clamped EWMA; without one, the table
applies as-is. Gating decides WHEN speculation runs, never WHAT is committed.

```zig
pub const Stats = struct {
    steps, spec_steps, fallback_steps, disabled_steps,
    drafted, accepted, rejected_steps, bonus, committed: usize,
    pub fn tokensPerStep(self) f64 / acceptanceRate(self) f64
    pub fn writeSummary(self, writer: *std.Io.Writer) !void
};
pub const TokenSink = struct { ptr: *anyopaque, func: *const fn (ptr, token: usize) anyerror!void, pub fn emit(...) };
pub const VerifyRowHook = struct { ... };  // test/debug: every pre-penalty logits row

pub fn SpeculativeDecoder(comptime Model: type) type {
    // fields: source, options, stats, gate, io: ?std.Io = null, on_verify_row
    pub fn init(allocator, source: DraftSource, options: Options) !Self
    pub fn deinit(self: *Self) void
    pub fn step(self, ctx, model: *const Model, kv: *KvCache,
                sampler: *Sampler, history: *std.ArrayList(usize), sink: TokenSink) !usize
}
```

- `init` validates the configuration loudly instead of leaving ReleaseFast UB:
  a top-K-consuming source with `topk_feedback == 0` is
  `error.TopKFeedbackDisabled`; degenerate gate options are
  `error.RateWindowTooSmall` / `error.ProbeStepsZero` / `error.CostTableEmpty`
  / `error.ReprobeAfterZero`.
- `step` runs one decode iteration under the invariant
  `history.items.len == kv.len + 1` (every committed token in `history`, the
  last one not yet forwarded into the cache); a violated invariant is
  `error.InvalidDecodeState` at runtime. `history` must be allocated with
  `ctx.allocator` — the decoder appends committed tokens to it. Each committed
  token is emitted through `sink`; returns the number committed (≥ 1). Verify
  passes run one batched `forwardStepAllLogits` over `[carried token,
  draft...]`, and `kv.truncate` drops rejected rows — on error-unwind paths
  too (`errdefer` restores the invariant).
- `Model` is duck-typed: `forwardStep` + `forwardStepAllLogits` over the
  shared `KvCache` (qwen3 and gemma4 today; qwen35's recurrent cache cannot
  rewind and is out of scope).

#### 13.9.2 Suffix-automaton index (`speculative/sam_index.zig`)

`SamIndex` is an online suffix automaton over a token stream — the
exact-match draft source (SAM-Decoding/SuffixDecoding lineage). It gives O(1)
amortized online extension, an exact self-match-excluded longest-suffix-match
length (drafts follow the most recent **prior** occurrence, never the current
one), and doubles as a frozen index over reference documents.

```zig
pub const SamIndex = struct {
    pub const max_stream_len: usize = 1 << 29;
    min_match: usize = 2, max_draft: usize = 16,     // policy fields
    pub fn init(allocator: Allocator) !SamIndex / deinit(self) void
    pub fn append(self, new_tokens: []const usize) !void   // O(1) amortized per token
    pub fn matchLen(self) usize
    pub fn draft(self, buf: []usize) usize
    pub fn tokenCount/stateCount/transitionCount(self) usize
    // frozen mode (RAG):
    pub fn freeze(self) void
    pub const Cursor = struct { state: u32 = 0, len: u32 = 0 };
    pub fn advance(self, cursor: *Cursor, token: usize) void
    pub fn draftFrom(self, cursor: Cursor, buf: []usize) usize
    // DraftSource method shapes:
    pub fn suggest(self, context, buf) usize / observe(self, committed) void / observeTopK(...)
};
pub const FrozenSource = struct { index: *const SamIndex, cursor: Cursor,
                                  pub fn suggest/observe/observeTopK };
```

A failed `append` **poisons** the index (`degraded`): all queries return 0
forever and further appends fail — a half-applied append must never serve
drafts (`observe` swallows errors into this degradation instead of
propagating). `freeze()` ends appends and makes external `Cursor`s safe
(appends can split states; only the internal cursor gets the clone fix-up).
`FrozenSource` owns a per-conversation cursor over one frozen document and
borrows the index.

```zig
test "SamIndex: longest self-excluded suffix match drives the draft" {
    const alloc = std.testing.allocator;
    var sam = try llm.speculative.sam_index.SamIndex.init(alloc);
    defer sam.deinit();
    try sam.append(&.{ 5, 6, 7, 5, 6 });
    // Longest suffix with an occurrence ending strictly before the end: [5,6].
    try std.testing.expectEqual(@as(usize, 2), sam.matchLen());
    var buf: [8]usize = undefined;
    const n = sam.draft(&buf); // tokens after the prior occurrence: {7,5,6}
    try std.testing.expectEqualSlices(usize, &.{ 7, 5, 6 }, buf[0..n]);
}
```

#### 13.9.3 Token recycling (`speculative/recycling.zig`)

```zig
pub fn TokenRecycling(comptime K: usize) type {
    pub const k = K;
    pub const sentinel: u32 = std.math.maxInt(u32);
    pub fn init(allocator, vocab: usize) !Self / deinit(self) void
    pub fn topkOf(self, token: usize) []const u32
    pub fn update(self, token: usize, topk: []const u32) void
    pub fn draftChain(self, last_token: usize, buf: []usize) usize
    pub fn suggest/observe/observeTopK   // DraftSource method shapes
}
pub const Recycling = TokenRecycling(8);
```

A `vocab × K` adjacency matrix (Token Recycling, Luo et al. 2024): row `t`
holds the most recent top-K next-token candidates observed at any verified
position whose input token was `t` (≈4.6 MiB for the Qwen3 vocab at K=8).
Drafting walks the top-1 chain until an unseen row or the budget stops it.
`observe` promotes each committed bigram to slot 0 (ground truth beats stale
logits); `observeTopK` overwrites whole rows from verification feedback. The
struct owns `m`; `update`/`observe*` copy, no slice is retained.

```zig
test "TokenRecycling: top-1 chain drafting" {
    const alloc = std.testing.allocator;
    var rec = try llm.speculative.recycling.Recycling.init(alloc, 32); // vocab 32, K = 8
    defer rec.deinit();
    rec.update(3, &.{ 7, 9 }); // most recent top-K observed after token 3
    rec.update(7, &.{5});
    var buf: [4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 2), rec.draftChain(3, &buf)); // 3 -> 7 -> 5, then unseen
    try std.testing.expectEqualSlices(usize, &.{ 7, 5 }, buf[0..2]);
}
```

#### 13.9.4 Cascade (`speculative/cascade.zig`)

`SpeculationIndex` is the user-facing orchestrator behind one `DraftSource`:
it composes (1) an online conversation `SamIndex` over every committed token,
(2) any number of frozen reference `SamIndex` documents injected via
`addReference` (the RAG seam), and (3) the `Recycling` matrix as the
self-draft fallback and `observeTopK` consumer.

```zig
pub const gate_window: usize = 64;
pub const Gate = struct { ..., pub fn muted(self) bool };   // per-source rolling acceptance
pub const FrozenRef = struct { index: SamIndex, cursor: SamIndex.Cursor, gate: Gate };

pub const SpeculationIndex = struct {
    // policy fields (defaults): beta = 2, min_match = 2, recycling_chain = 8,
    // mute_acceptance = 0.20, mute_commits = 128, accounting_min_draft = 2
    pub fn init(allocator: Allocator, vocab: usize) !SpeculationIndex / deinit(self) void
    pub fn addReference(self, tokens: []const usize) !void
    pub fn clearReferences(self) void
    pub fn suggest(self, context, buf) usize / observe(self, committed) void / observeTopK(...)
    pub fn asDraftSource(self: *SpeculationIndex) DraftSource
    pub fn truncatePending(self, new_len: usize) void
    pub fn writeSourceSummary(self, writer: *std.Io.Writer) !void
};
```

Suggest policy: the source with the longest current match wins (ties break
toward the conversation; among references, first-added), with draft budget
`min(buf.len, beta * (1 + match_len))`; matches shorter than `min_match` fall
back to the recycling top-1 chain (≤ `recycling_chain` tokens); an unseen
recycling row drafts 0 and the decoder takes a plain step. Each source keeps
a rolling acceptance gate over its last `gate_window` drafted tokens: below
`mute_acceptance` it is muted for `mute_commits` committed tokens, then
re-probed — muted sources keep observing so they stay in sync. Acceptance is
settled at the `suggest`→`observe` seam (longest common prefix of draft and
next committed slice); `accounting_min_draft` mirrors the decoder's
`min_draft` so unverified short drafts never skew the gates.
`addReference` catches the new document's cursor up over the already-observed
stream, so a mid-conversation injection can match existing context
immediately.

```zig
test "SpeculationIndex: observe committed tokens, suggest a draft" {
    const alloc = std.testing.allocator;
    var index = try llm.speculative.cascade.SpeculationIndex.init(alloc, 1024);
    defer index.deinit();

    index.observe(&.{ 1, 2, 3, 4, 1, 2 }); // [1,2] recurred; earlier it was followed by [3,4]
    var buf: [16]usize = undefined;
    const n = index.suggest(&.{ 1, 2, 3, 4, 1, 2 }, &buf);
    try std.testing.expect(n >= 2);
    try std.testing.expectEqualSlices(usize, &.{ 3, 4 }, buf[0..2]);

    try index.addReference(&.{ 7, 8, 9, 10 }); // RAG seam: frozen document index
    const source = index.asDraftSource(); // hand this to SpeculativeDecoder
    try std.testing.expect(source.wantsTopK()); // the recycling matrix consumes logits feedback
}
```

#### 13.9.5 Enabling speculation in a runner

The turnkey path is `chat.Options{ .speculation = true }` (§13.8), which owns
the index/decoder lifecycle, stop-token wiring, and turn trimming. A custom
runner drives the decoder directly:

```zig
fn snippetDecoderLoop(
    ctx: *fucina.ExecContext,
    model: *const llm.qwen3.model.Model,
    kv: *llm.kv_cache.KvCache,
    index: *llm.speculative.cascade.SpeculationIndex,
    history: *std.ArrayList(usize),
    sink: llm.speculative.core.TokenSink,
) !void {
    const Decoder = llm.speculative.core.SpeculativeDecoder(llm.qwen3.model.Model);
    var decoder = try Decoder.init(ctx.allocator, index.asDraftSource(), .{ .max_draft = 16 });
    defer decoder.deinit();
    var sampler = llm.sampler.Sampler.init(.{});
    // Invariant: history.len == kv.len + 1 (last committed token not yet forwarded).
    while (history.items.len < 128) {
        _ = try decoder.step(ctx, model, kv, &sampler, history, sink);
    }
} // requires model assets to run
```

#### 13.9.6 Grammar-constrained drafting (`speculative/constrained.zig`)

`ConstrainedSource` makes a grammar constraint *accelerate* speculation
instead of muting it ([CONSTRAINED-DECODING.md](CONSTRAINED-DECODING.md) §5
is the design record). It wraps any inner `DraftSource` with a
`LogitProcessor` that exposes the §13.6 structural hooks
(`hasStructure()`), and must sit on the same processor instance installed
on the stream's sampler:

```zig
pub const ConstrainedSource = struct {
    pub fn init(processor: LogitProcessor, inner: DraftSource) ConstrainedSource
    pub fn source(self: *ConstrainedSource) DraftSource
};
```

- **Forced spans draft themselves.** When the grammar mandates a unique
  continuation (JSON structure, a forced literal), `suggest` proposes it
  directly — and because the sampler's mask allows exactly that token at
  each row, the whole span verifies with acceptance probability 1.
- **Certainly-rejected drafts die early.** Otherwise the inner source
  proposes and the draft is truncated at its first grammar-invalid token
  (`validPrefixLen`) — those tokens would be masked to `-inf` at their
  verify row, so proposing them only wastes verify compute and drags the
  inner source's acceptance gates down. The truncation is forwarded through
  `DraftSource.truncatePending` so the inner accounting matches what is
  actually verified.

Losslessness is untouched (drafts never decide WHAT is committed, and both
hooks are deterministic lookaheads); the constrained speculative stream is
token-for-token identical to the constrained plain stream, proven greedy +
sampled in `chat_tests.zig`. Wiring is automatic everywhere: with
`chat.Options{ .speculation = true, .logit_processor = ... }` the
conversation wraps its cascade in a `ConstrainedSource` whenever the
processor has structure, and the qwen3 runner does the same for
`--spec` + a grammar flag. Measured effect on Qwen3-0.6B-Q8_0 JSON-schema
chat: draft acceptance 0% → 83%, with the cost gate staying on (§13.6).

```zig
test "constrained source: forced spans preempt, invalid drafts truncate" {
    const Fixed = struct {
        // state = 1: the grammar forces {7, 8}; state = 0: free choice with
        // ids >= 3 invalid. process/commit are irrelevant on the suggest side.
        fn process(_: *anyopaque, _: []f32, _: []const usize) anyerror!void {}
        fn commit(_: *anyopaque, _: usize) anyerror!void {}
        fn forcedTokens(ptr: *anyopaque, buf: []usize) usize {
            const forcing: *u8 = @ptrCast(ptr);
            if (forcing.* == 0) return 0;
            buf[0] = 7;
            buf[1] = 8;
            return 2;
        }
        fn validPrefixLen(_: *anyopaque, tokens: []const usize) usize {
            for (tokens, 0..) |t, i| if (t >= 3) return i;
            return tokens.len;
        }
        fn suggest(_: *anyopaque, _: []const usize, buf: []usize) usize {
            buf[0] = 1; // an inner source that always drafts {1, 5}
            buf[1] = 5;
            return 2;
        }
        fn observe(_: *anyopaque, _: []const usize) void {}
    };
    var forcing: u8 = 1;
    const processor = llm.logit_processor.LogitProcessor{ .ptr = &forcing, .vtable = &.{
        .process = Fixed.process,
        .commit = Fixed.commit,
        .forcedTokens = Fixed.forcedTokens,
        .validPrefixLen = Fixed.validPrefixLen,
    } };
    const inner = llm.speculative.core.DraftSource{ .ptr = &forcing, .vtable = &.{
        .suggest = Fixed.suggest,
        .observe = Fixed.observe,
    } };

    var cs = llm.speculative.constrained.ConstrainedSource.init(processor, inner);
    var buf: [8]usize = undefined;
    // Forced state: the grammar span wins over the inner source.
    try std.testing.expectEqual(@as(usize, 2), cs.source().suggest(&.{0}, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 7, 8 }, buf[0..2]);
    // Free choice: the inner draft {1, 5} truncates at the invalid 5.
    forcing = 0;
    try std.testing.expectEqual(@as(usize, 1), cs.source().suggest(&.{0}, &buf));
    try std.testing.expectEqual(@as(usize, 1), buf[0]);
}
```

## 14. Model families and example applications

The `fucina_llm` module root (`src/llm.zig`) exposes each model family as a
namespace — `llm.qwen3.{model,train}`, `llm.qwen35.model`,
`llm.gemma.{gemma4,gemma4_train,moe,moe_route,moe_route_tensor}`,
`llm.diffusion_gemma.model`, `llm.deepseek2.model`, `llm.glm4moe.model`,
`llm.deepseek4.model`, `llm.parakeet.*`, `llm.speculative.*` — while the
generic helpers (`llm.weights`, `llm.kv_cache`, `llm.kv_persist`, `llm.tokenizer`,
`llm.spm_tokenizer`, `llm.sampler`, `llm.logit_processor`, `llm.llguidance`,
`llm.chat`, `llm.data`, `llm.gguf_meta`, `llm.ptqtp_gguf`,
`llm.unicode_categories`) stay flat and are covered in §13. This section
documents the per-family model
APIs, their runner CLIs, and the example applications under `examples/`.
Weight containers (`LinearWeight` and its quant arms), `KvCache`, tokenizers,
sampling, chat orchestration, and speculative decoding are §13 material;
GGUF parsing is §12; LoRA/optimizer/ES mechanics are §11.

### 14.1 Conventions shared by every family

(`src/llm/*/model.zig`, `src/llm/gguf_meta.zig`)

**Config from GGUF metadata.** Each family's `Config.fromGguf(file)` reads
hyperparameters from the standard GGUF key convention: the value of
`general.architecture` (e.g. `"qwen3"`, `"qwen3moe"`, `"qwen35"`, `"gemma4"`,
`"diffusion-gemma"`, `"parakeet"`) prefixes every key —
`<arch>.block_count`, `<arch>.embedding_length`,
`<arch>.attention.head_count`, `<arch>.rope.freq_base`, and so on — so one
loader covers every size of a family without hardcoding. The vocab size comes
from the `token_embd.weight` tensor shape, not a key. The `llm.gguf_meta`
helpers (`metaInt`, `metaIntOpt`, `metaFloat`, `metaFloatOpt`) implement the
prefixing plus a per-family zero policy: qwen3/qwen35 reject present-but-zero
required ints (`.reject_zero` — every config int is structurally positive),
gemma accepts zeros (`.accept_zero` — keys like
`attention.shared_kv_layers` are legitimately 0). Missing or malformed keys
surface as `Error.InvalidConfig` (parakeet: `Error.MissingMetadata`;
parakeet also departs from the tensor-shape vocab convention — its
`Config.fromGguf` reads `parakeet.vocab_size` from metadata).

**Loader entry points.** The qwen3, qwen35, gemma4 and diffusion_gemma
families expose the same pair (the deepseek2, glm4moe, and deepseek4
loaders read their `Config` from the file internally, taking
`max_positions` and/or a `LoadOptions` instead):

```zig
pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8, config: Config) !Model
pub fn loadGgufFromFile(ctx: *ExecContext, file: *gguf.File, config: Config) !Model
// qwen35 takes `file: *const gguf.File` (it never calls takeMapping);
// a mutable pointer coerces, so caller code is unaffected
```

`loadGguf` opens the file itself (mmap via `gguf.File.loadMmap` for qwen3,
gemma4 and diffusion_gemma; qwen35's convenience arm uses the eager
`gguf.File.load`). `loadGgufFromFile` takes an already-parsed `gguf.File` so
the caller can build a tokenizer from the same file's metadata without a
second read — the pattern every runner uses:

```zig
var file = try fucina.gguf.File.loadMmap(alloc, io, path);
defer file.deinit();
const config = try llm.qwen3.model.Config.fromGguf(&file);
var model = try llm.qwen3.model.Model.loadGgufFromFile(&ctx, &file, config);
defer model.deinit();
var tok = try llm.tokenizer.Tokenizer.initFromGguf(alloc, &file, .{});
defer tok.deinit();
```

Weights are materialized through `llm.weights.LinearWeight.load` (§13), which
keeps the GGUF dtype resident — f32/f16/bf16 and the quant forms (Q8_0,
Q4_K/Q5_K/Q6_K and the other ggml types) run their own packed kernels; no
global dequantization happens at load. Sibling projections that share a dtype
and layout are fused at load (`weights.fuseLinear`): q/k/v into one QKV
matrix, gate/up into one gate_up matrix — one wider GEMM per block instead of
two or three. Layer loading is parallelized across the work pool
(`gguf_meta.parallelLoadLayers`). Ownership: `Model.deinit` releases every
weight; when expert blocks borrow from the mmap (qwen3 MoE when a
single-file GGUF is mmap'd — split GGUFs' experts are copied, and opt-in
expert disk streaming (`LoadOptions.moe_stream`) leaves the mapping with the
caller; gemma4/diffusion_gemma under `borrow_experts`) the model takes the
mapping via `file.takeMapping()` and unmaps it **last** in `deinit`. On GPU
builds nothing changes at this API level — offload decisions are per-GEMM
work-gates inside the shared kernels (§9); the two model-level GPU knobs are
gemma-MoE's raw expert representation (14.4) and diffusion_gemma's
`convertDenseWeightsToF16` (14.5).

**Forward/decode surface.** The autoregressive families share one contract:

- `forwardLastLogits(ctx, token_ids)` — cacheless whole-sequence forward;
  returns the last position's `[1, vocab]` logits (caller deinits). Empty
  input is `Error.InvalidSequenceLength`.
- `initKvCache(ctx, capacity)` / qwen35's `initCache` — build the streaming
  cache sized for `capacity` positions. This is the duck-typed construction
  seam the generic `llm.chat.Conversation` embedder uses (§13).
- `forwardStep(ctx, kv, token_ids, pos0)` — process `token_ids` at absolute
  positions `pos0..pos0+len`, append their K/V, advance the cache by `len`,
  return the last row's logits. **Contract:** `kv.len == pos0` or
  `Error.InvalidSequenceLength`; `kv.len + len <= kv.capacity` or
  `kv_cache.Error.KvCacheOverflow`. Prefill is one call on a fresh cache with
  `pos0 == 0` (last-token logits equal `forwardLastLogits`); decode is a
  one-token call at `pos0 == kv.len`.
- `forwardStepAllLogits` (qwen3, gemma4) — same KV semantics, but returns
  `[len, vocab]` logits for **every** appended position: the
  speculative-decoding verify entry (§13 — one batched pass scores all
  draft positions for ~one step's weight traffic).
- `forwardStepBatch` (qwen3 only) — lockstep multi-stream decode, 14.2.
- `generate(ctx, kv, prompt_tokens, out_tokens, options)` (qwen3, gemma4;
  not qwen35) — greedy loop
  (argmax; `GenerateOptions{ .max_new_tokens, .stop_token = null }`); resets
  `kv` first, returns the count written. diffusion_gemma's block-diffusion
  `generate` has its own options and returns a `GenerateResult` (14.5).
  Sampled decoding is composed by the callers from `llm.sampler` (§13).
- `forward*Profiled` variants take `io: std.Io` and a family-specific
  `ForwardProfile` accumulator (per-block wall-clock buckets; the `--profile`
  runner flag).

All forward entries take `*const Model` and mutate only the `ExecContext`,
the cache, and (profiled) the profile struct; a loaded model is read-only.
`ExecContext` is single-threaded (§6), so concurrent streams over one model
need one context and one cache per thread. Returned logits are caller-owned
constants (`deinit` them); no exec scope is required for inference.

### 14.2 Qwen3 — dense and MoE (`src/llm/qwen3/model.zig`)

The reference transformer family and the most complete runner: standard GQA
attention with per-head q/k RMSNorm and full RoPE, SwiGLU FFN — dense, or a
top-k routed mixture (`qwen3moe`, e.g. 30B-A3B) selected purely by GGUF
metadata.

```zig
pub const Config = struct {
    vocab_size, hidden_size, intermediate_size, num_layers,
    num_attention_heads, num_key_value_heads, head_dim: usize,
    rms_norm_eps, rope_theta: f32,
    num_experts: usize = 0,          // 0 = dense
    num_experts_used: usize = 0,
    moe_intermediate_size: usize = 0,
    norm_topk_prob: bool = true,
    moe_expert_top_p: f32 = 1.0,     // adaptive expert top-p; 1.0 = full top-k (runtime knob, not GGUF)
    pub fn isMoe(self: Config) bool
    pub fn qwen3_0_6b() Config       // hardcoded 0.6B reference config
    pub fn fromGguf(file: *const gguf.File) !Config
};
```

`fromGguf` reads `<arch>.{embedding_length, feed_forward_length, block_count,
attention.head_count, attention.head_count_kv, attention.key_length,
attention.layer_norm_rms_epsilon, rope.freq_base}` plus the MoE trio
`{expert_count, expert_used_count, expert_feed_forward_length}`; a model with
no `expert_count` key stays dense. Validation (at load) rejects zero heads,
non-divisible GQA grouping, odd `head_dim`, and inconsistent MoE fields with
`Error.InvalidConfig`.

```zig
test "qwen3 reference config" {
    const cfg = llm.qwen3.model.Config.qwen3_0_6b();
    try std.testing.expect(!cfg.isMoe());
    try std.testing.expectEqual(@as(usize, 28), cfg.num_layers);
    try std.testing.expectEqual(@as(usize, 8), cfg.num_key_value_heads);
}
```

`pub const Error = weights.Error || error{ InvalidConfig,
InvalidSequenceLength, MismatchedKvCaches }`. Public surface on `Model`:
`loadGguf`, `loadGgufOptions`, `loadGgufFromFile`, `loadGgufFromFileOptions`
(opt-in MoE expert disk streaming, `LoadOptions.moe_stream`), `deinit`,
`forwardLastLogits`, `forwardLastLogitsProfiled`, `initKvCache`,
`forwardStep`, `forwardStepProfiled`, `forwardStepAllLogits`,
`forwardStepBatch`, `generate`, `decoratePtqtp`, `savePtqtpGguf` (§10.9);
plus `GenerateOptions`, `ForwardProfile`, `MoeStreamOptions`, `LoadOptions`,
and `applyExpertTopP` at module level.

Load specifics: when the GGUF is mmap'd, MoE expert stacks
(`ffn_{gate,up,down}_exps.weight`) are **borrowed zero-copy** from the
mapping (`weights.loadMoeRhs` with `borrow = file.is_mmap and
!file.isSplit()` — split GGUFs cannot hand over their multiple mappings, so
their experts are copied) instead of copying multi-GB tensors; the model
owns the mapping and unmaps it last.
A missing `output.weight` means tied embeddings (`token_embedding.cloneView`).
The MoE FFN routes on the host (`routerTopK` with
`normalize_selected = norm_topk_prob`) and runs the router-weighted SwiGLU
mixture through `weights.moeSwiGluFfnSeq`: decode (seq 1) uses a fused
expert-parallel GEMV, prefill groups tokens by expert so each expert's
weights are read once per batch.

`initKvCache` builds a uniform-geometry f16 `KvCache`
(`KvCache.init(ctx, num_layers, num_key_value_heads, head_dim, capacity)`).
Qwen3 is the **only** family whose attention also accepts a q8_0 cache
(construct it with `kv_cache.KvCache.initWithDtype(..., .q8_0)`, §13; the
runner flag is `--cache-type q8_0` — half the KV memory, a capacity option
rather than a speed one).

```zig
var file = try fucina.gguf.File.loadMmap(alloc, io, "models/Qwen3-0.6B-Q8_0.gguf");
defer file.deinit();
const config = try llm.qwen3.model.Config.fromGguf(&file);
var model = try llm.qwen3.model.Model.loadGgufFromFile(&ctx, &file, config);
defer model.deinit();

var kv = try model.initKvCache(&ctx, 512);
defer kv.deinit();
var prefill = try model.forwardStep(&ctx, &kv, &.{ 151644, 872, 198 }, 0); // [1, vocab]
defer prefill.deinit();
var step = try model.forwardStep(&ctx, &kv, &.{9707}, kv.len); // one decode step
defer step.deinit();
// requires model assets to run
```

**Lockstep batch decode.** `forwardStepBatch(ctx, caches, token_ids)` decodes
one new token per stream, each stream backed by its own sibling cache from
this model's `initKvCache` (same dtype, distinct pointers, layer count
matching the model — violations return `Error.MismatchedKvCaches`; a full
cache returns `KvCacheOverflow`). Row `s` of the returned
`[n_streams, vocab]` logits is stream `s`'s next-token distribution and every
cache advances by one. The dense trunk (QKV/O-proj, FFN or MoE mixture,
lm_head) runs as ONE m=n pass — weights are read once for all streams, the
batch-decode bandwidth win — while RoPE positions, KV appends and attention
stay per-stream (ragged, each row against its own cache at its own position).
Per-row numerics match per-stream `forwardStep` bit-for-bit below the
m-dependent kernel thresholds (quantized x4-packed kernels engage at n >= 4,
fused FFN at seq >= 12, tiled attention at seq >= 48); beyond them rows can
differ by ~1e-6 reassociation drift. The same thresholds bound
`forwardStepAllLogits` against per-token steps.

```zig
var kv_a = try model.initKvCache(ctx, 256);
defer kv_a.deinit();
var kv_b = try model.initKvCache(ctx, 256);
defer kv_b.deinit();
var a = try model.forwardStep(ctx, &kv_a, &.{ 151644, 872 }, 0);
a.deinit();
var b = try model.forwardStep(ctx, &kv_b, &.{ 151644, 8948 }, 0);
b.deinit();
// One m=2 weight pass decodes both streams; row s = stream s's logits.
var logits = try model.forwardStepBatch(ctx, &.{ &kv_a, &kv_b }, &.{ 9707, 3838 });
defer logits.deinit();
// requires model assets to run
```

**Speculative decoding** is available on this family: the runner's `--spec`
drives the draft-model-free SAM + Token-Recycling cascade from
`llm.speculative` (§13) with `forwardStepAllLogits` as the verify pass;
`--spec-ref doc.txt` injects a reference document the drafter can copy spans
from. Output is lossless (greedy streams verified identical with and without
`--spec`).

**Runner** (`examples/qwen3.zig`, the full CLI surface is documented in
[RUNNING-MODELS.md](RUNNING-MODELS.md)):

```sh
# chat / REPL, sampling flags, GPU offload
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "What is the capital of France?" --no-think \
  --temp 0.7 --top-k 40 --top-p 0.9 --seed 42
zig build qwen3 -Dgpu=metal -Doptimize=ReleaseFast -- models/Qwen3-0.6B-f16.gguf --repl

# raw completion, speculative decode, lockstep multi-stream bench
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  --prompt "The capital of France is" --gen 64 --spec
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  151644,872,198,9707 --gen 64 --bench 3 --streams 4

# q8_0 KV cache; tokenizer / logit parity oracles
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "..." --gen 256 --cache-type q8_0
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --tokenize input.txt

# constrained decoding (§13.6; needs -Dllguidance=true): the reply must
# satisfy the JSON schema / regex / Lark grammar; composes with --spec
zig build qwen3 -Dllguidance=true -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "Give me facts about Paris." --no-think \
  --json-schema '{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer","maximum":99999999}},"required":["city","population"],"additionalProperties":false}'
zig build qwen3 -Dllguidance=true -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "The answer is" --gen 32 --regex ' (yes|no)\.'
# --json-schema @schema.json / --lark @grammar.lark read the grammar from a file
```

#### 14.2.1 LoRA fine-tuning (`src/llm/qwen3/train.zig`)

`llm.qwen3.train` trains LoRA adapters over a frozen, possibly quantized
`qwen3.Model` (dense only — MoE configs return `Error.MoeUnsupported`). The
trainer mirrors the inference forward op-for-op but routes every frozen
projection through the differentiable frozen-RHS `dot` (gradients flow to f32
activations only; weight memory stays quantized/f16) and adds trainable A/B
deltas on the projections selected at comptime. Mechanics — adapters,
optimizers, checkpoints, exec scopes — live in §11; this is the entry-point
map.

```zig
pub const Targets = struct { q: bool = true, k: bool = false, v: bool = true,
                             o, gate, up, down: bool = false };
pub const ignore_index: usize = std.math.maxInt(usize);
pub fn Trainer(comptime targets: Targets) type
```

Module-level symbols: `Error` (`MoeUnsupported`, `ExecScopeRequired`,
`InvalidSequenceLength`, `LabelLengthMismatch`, `InvalidLayerRange`,
`InvalidInjection`), `Targets`, `ignore_index`, `ModelLayer` (test seam:
the model's per-block layer type), `Hidden`
(`fucina.Tensor(.{ .seq, .embed })`), `Injection` (`{ pos, row }` — a
differentiable single-row embedding override), `ForwardOptions`
(`{ start_layer = 0, layer_count = null, inject = null }`).

`Trainer(targets)` members: `init(ctx, model, lora.Config, seed)` /
`deinit`; `registerAllParams(opt)` (registers every A/B under
`layers.<i>.<target>.lora_{a,b}` on anything with `addParamNamed`; the
trainer must outlive the optimizer — params and names are borrowed);
`saveAdapters(writer)` / `loadAdapters(reader)` /
`loadAdaptersWithOptions(reader, optim.LoadOptions)` (clean safetensors state
dict, strict one-to-one on load); `loss(ctx, tokens, labels)` /
`lossExt(..., LossOptions)` (mean CE against pre-shifted labels,
`ignore_index` masks; **must** run inside an open exec scope —
`Error.ExecScopeRequired` otherwise — and returns a scope-owned borrow;
`LossOptions{ .reduction = .mean|.sum, .loss_scale = 1 }` is the gradient-
accumulation seam, TRAINING.md §4); `lossInjected(...)`;
`evalLastLogits` / `evalLogits` / `evalLastLogitsExt` (dropout off, no step
advance, run under their own scope, return caller-owned constants);
`forwardHidden(ctx, tokens, step, opts)` (raw residual stream, scope
required); the `checkpoint_layers` field enables recompute-in-backward per
layer; `n_enabled` and `LayerAdapters` are the comptime target plumbing.
Dropout is deterministic per (step, layer, projection) from the base seed;
RoPE tables are cached per sequence length and freed only in `deinit`.

```zig
const Trainer = llm.qwen3.train.Trainer(.{ .q = true, .v = true });
var trainer = try Trainer.init(ctx, model, .{ .rank = 8, .alpha = 16 }, 42);
defer trainer.deinit();

var opt = fucina.optim.AdamW.init(ctx.allocator, .{ .lr = 1e-3 });
defer opt.deinit();
try trainer.registerAllParams(&opt);

const scope = ctx.openExecScope();
defer ctx.closeExecScope(scope);
const tokens: []const usize = &.{ 1, 2, 3, 4 };
const labels: []const usize = &.{ 2, 3, 4, llm.qwen3.train.ignore_index };
var loss = try trainer.loss(ctx, tokens, labels);
try loss.backward(ctx);
try opt.step(ctx);
opt.zeroGrad();
// requires model assets to run
```

The end-to-end loop — fine-tune (`zig build finetune`), merge adapters into
dense weights (`zig build export-gguf -- --adapters ... --alpha ...`),
re-quantize, serve — is scripted in [RUNNING-MODELS.md](RUNNING-MODELS.md)
("Fine-tune → merge → serve loop"); the gradient-free twin is
`zig build es-finetune` (§11, TRAINING.md §13).

### 14.3 Qwen3.5 — Gated-DeltaNet hybrid (`src/llm/qwen35/model.zig`)

The `qwen35` GGUF arch is a **hybrid linear-attention transformer** (sibling
of qwen3next, not a Qwen3 variant): every `full_attention_interval`-th block
is full GQA attention (fused Q+gate projection, per-head q/k RMSNorm,
multi-section/partial RoPE, sigmoid output gate); the rest are **DeltaNet
linear** blocks — a causal depthwise conv1d feeding a gated delta-rule
recurrent scan over per-v-head state matrices. Both feed a SiLU dense FFN.

Config adds, on top of the usual attention keys: `rope.dimension_count`
(`rope_n_rot` — partial RoPE when < `head_dim`), `rope.dimension_sections`
(`rope_sections: [4]i32`), `full_attention_interval` (default 4), and the SSM
dims `ssm.{conv_kernel, inner_size, state_size, time_step_rank, group_count}`
(`ssm_d_conv/d_inner/d_state/dt_rank/n_group`), plus `nextn_predict_layers`
and `expert_count`. `Config.isRecurrent(il)` implements the block schedule;
`isMoe()` mirrors qwen3. Validation rejects `qwen35moe` and MTP/NextN
variants with `Error.UnsupportedVariant` (dense text path only).

```zig
test "qwen35 hybrid layer pattern" {
    const cfg = llm.qwen35.model.Config{
        .vocab_size = 151_936, .hidden_size = 1024, .intermediate_size = 4096,
        .num_layers = 24, .num_attention_heads = 16, .num_key_value_heads = 2,
        .head_dim = 256, .rms_norm_eps = 1e-6, .rope_theta = 1_000_000,
        .rope_n_rot = 64, .rope_sections = .{ 11, 11, 10, 0 },
        .full_attention_interval = 4,
        .ssm_d_conv = 4, .ssm_d_inner = 4096, .ssm_d_state = 128,
        .ssm_dt_rank = 32, .ssm_n_group = 16,
    };
    // Every 4th block is full attention; the rest run the DeltaNet scan.
    try std.testing.expect(cfg.isRecurrent(0));
    try std.testing.expect(!cfg.isRecurrent(3));
    try std.testing.expect(cfg.isRecurrent(4));
}
```

`pub const Error = weights.Error || error{ InvalidConfig,
InvalidSequenceLength, UnsupportedVariant, UnsupportedKvCacheDtype }`.
Model surface: `loadGguf`, `loadGgufFromFile`, `deinit`, `blockCounts`
(`.{ attn, linear }` counts for `--info`), `forwardLastLogits` (cacheless,
chunked DeltaNet scan; logit-parity-validated against llama.cpp on
Qwen3.5-0.8B), `initCache`, `forwardStep`, `forwardStepWithScanMode`,
`forwardStepProfiled`, `forwardStepProfiledWithScanMode`; module-level
`LinearScanMode` and `ForwardProfile`.

The streaming state is `Cache`, not a bare `KvCache`: an f16 attention KV
cache (q8_0 caches are rejected with `Error.UnsupportedKvCacheDtype`) plus,
per linear layer, a conv window (`(d_conv-1)*conv_dim` floats) and the
recurrent state matrix (`H*Sd*Sd` floats) — O(1) state per linear layer
regardless of context. `Cache.deinit`, `Cache.reset` (zero all carried
state), `Cache.len()` (current position). `LinearScanMode` selects the
DeltaNet prefill path: `.chunked` (default — exact batched chunked-GEMM) or
`.recurrent` (exact token-by-token scan, forced even for prefill); both are
exact, the choice is performance/validation.

```zig
var file = try fucina.gguf.File.loadMmap(alloc, io, "models/Qwen3.5-0.8B-Q8_0.gguf");
defer file.deinit();
const config = try llm.qwen35.model.Config.fromGguf(&file);
var model = try llm.qwen35.model.Model.loadGgufFromFile(&ctx, &file, config);
defer model.deinit();

var cache = try model.initCache(&ctx, 256); // KV + conv/SSM state
defer cache.deinit();
var prefill = try model.forwardStep(&ctx, &cache, &.{ 9707, 11, 1879 }, 0);
defer prefill.deinit();
var step = try model.forwardStepWithScanMode(&ctx, &cache, &.{0}, cache.len(), .recurrent);
defer step.deinit();
// requires model assets to run
```

No training entry, no `forwardStepAllLogits`/`forwardStepBatch`, no
speculative decoding on this family. The CLI is a loader/parity harness:

```sh
zig build qwen35 -Doptimize=ReleaseFast -- models/Qwen3.5-0.8B-Q8_0.gguf
zig build qwen35 -Doptimize=ReleaseFast -- models/Qwen3.5-0.8B-Q8_0.gguf --info
zig build qwen35 -Doptimize=ReleaseFast -- models/Qwen3.5-0.8B-Q8_0.gguf --linear-scan chunked
```

### 14.4 Gemma 4 — text + MoE (`src/llm/gemma/`)

`gemma4` (26B-A4B class) is the geometry-heavy family: 16 query heads over
**per-layer** KV geometry — interleaved local sliding-window (SWA) and global
layers with different head dims, KV-head counts and RoPE bases, trailing
layers that **share** an earlier layer's K/V (`shared_kv_layers`), optional
per-layer embeddings (PLE), per-layer output scales, GeGLU FFNs (shared dense
MLP + a 128-expert top-8 MoE), and final logit softcapping.

Config keys beyond the common set: `attention.key_length_swa`,
`attention.sliding_window`, `attention.shared_kv_layers`,
`rope.freq_base_swa`, `expert_count`/`expert_used_count`/
`expert_feed_forward_length`, `embedding_length_per_layer_input` (PLE width,
0 = disabled), `final_logit_softcapping`, plus the per-layer arrays
`gemma4.attention.sliding_window_pattern` and
`gemma4.attention.head_count_kv` (read by `readU32OrBoolArray`, which
broadcasts a scalar across layers like llama.cpp's `get_key_or_arr`).
`Config.fromGguf` wraps `Config.fromGgufArch(file, "gemma4")`; the `arch`
argument exists because diffusion-gemma shares the identical hparam key set
under its own prefix. `Config.borrow_experts` is a **load-time policy field**,
not a GGUF hparam: `true` (the `--experts=borrow` flag) borrows MoE experts
zero-copy from the mmap on CPU builds — load in seconds at ~half the RSS
instead of x4-packing ~20 GB — at some decode-throughput cost; the default
packed path favors peak CPU throughput. Numerically identical either way.

`deriveGeometry(allocator, n_layer, swa_pattern, kv_heads_in,
shared_kv_layers, head_dim_global, head_dim_swa) !LayerGeometry` computes the
per-layer view (`is_swa`, `head_dim`, `kv_heads`, `has_kv`, `kv_ref`;
`LayerGeometry.deinit(allocator)` frees it): the trailing `shared_kv_layers`
layers store no K/V and instead reference the last same-type writer (offset
2 for SWA, 1 for global).

```zig
test "gemma4 shared-KV geometry" {
    const alloc = std.testing.allocator;
    var geom = try llm.gemma.gemma4.deriveGeometry(
        alloc,
        4, // n_layer
        &.{ true, true, false, true }, // SWA pattern (false = global)
        &.{ 4, 4, 8, 4 }, // per-layer KV heads
        1, // shared_kv_layers: the last layer stores no K/V
        256, // head_dim_global
        128, // head_dim_swa
    );
    defer geom.deinit(alloc);
    try std.testing.expect(!geom.has_kv[3]);
    try std.testing.expectEqual(@as(usize, 1), geom.kv_ref[3]); // reuses layer 1
    try std.testing.expectEqual(@as(usize, 128), geom.head_dim[3]);
}
```

`pub const Error = weights.Error || error{ InvalidConfig,
InvalidSequenceLength, MissingMetadata, UnsupportedExpertType,
UnsupportedKvCacheDtype }`. Model surface: `loadGguf`, `loadGgufFromFile`,
`deinit`, `initKvCache` (per-layer geometry:
`KvCache.initPerLayer(ctx, geom.kv_heads, geom.head_dim, capacity)`),
`forwardLastLogits`, `forwardLastLogitsProfiled`, `forwardStep`,
`forwardStepProfiled`, `forwardStepAllLogits` (speculative verify entry —
softcapping applies to every row), `generate` + `GenerateOptions`,
`ForwardProfile`. Only f16 caches are accepted (`requireF16KvCache` returns
`Error.UnsupportedKvCacheDtype` for q8_0). Final logits are softcapped when
`final_logit_softcapping != 0` (a fused `softcap30` kernel serves the
model's actual 30.0 value). The remaining public symbols are loader/forward
plumbing reused by diffusion_gemma and the trainer: `max_heads` (64),
`metaInt`/`metaIntOpt`/`metaFloat`/`metaFloatOpt`, `LayerGeometry`,
`deriveGeometry`, `readU32OrBoolArray`, `MoeFfn`, `PerLayerInject`,
`SeparateAttentionProjection`, `FusedAttentionProjectionKind`,
`FusedAttentionProjection`, `AttentionProjectionResult`,
`AttentionProjection` (with `toResidentF16` and `project`), `Layer`,
`loadLayers` (the pub wrapper over a file-private `LayerLoader`),
`requireF16KvCache`,
`attnBlock`, `ffnBlock`.

```zig
var file = try fucina.gguf.File.loadMmap(alloc, io, "models/gemma-4-26B-A4B-it-UD-Q6_K.gguf");
defer file.deinit();
var config = try llm.gemma.gemma4.Config.fromGguf(&file);
config.borrow_experts = true; // zero-copy experts from the mmap (--experts=borrow)
var model = try llm.gemma.gemma4.Model.loadGgufFromFile(&ctx, &file, config);
defer model.deinit();

var kv = try model.initKvCache(&ctx, 512); // per-layer geometry
defer kv.deinit();
var prefill = try model.forwardStep(&ctx, &kv, &.{ 2, 651, 235 }, 0);
defer prefill.deinit();
var step = try model.forwardStep(&ctx, &kv, &.{651}, kv.len);
defer step.deinit();
// requires model assets to run
```

**MoE expert kernels** (`moe.zig`, `moe_route.zig`, `moe_route_tensor.zig`,
survey depth). The expert FFN has two weight representations: per-expert
**packed** RHS (`MoeFfn.gate/up/down`, the tested Q6_K/Q8_0 packed kernels —
peak CPU throughput) and **raw** GGUF-layout blocks
(`RawExpertWeights{ gu: .q6_k|.q4_k, dn_blocks, device_owned, borrowed }`,
plus `guBlockCount`), used on `-Dgpu=metal` builds (grouped dequant-in-kernel
Metal GEMMs read them; the loader then keeps ONE representation — tens of
seconds and ~24 GB saved at load), on Q4_K-transcoded experts, under
`--experts=borrow`, and by the trainer. Four entry pairs cover the
(decode | batch) x (packed | raw) matrix: `decodePackedTensor` /
`batchPackedTensor` / `decodeRawTensor` / `batchRawTensor` (tagged-tensor
wrappers) over `decodePacked` / `batchPacked` / `decodeRaw` / `batchRaw`.
Batch entries consume the shared counting-sort route plan re-exported by
`moe_route` (`Plan`, `BuildResult`, `build`) with the gemma-specific
expert-major scatter (`moe_route.scatterInto`, deliberately serial to keep
each token's summation order fixed against parity oracles);
`moe_route_tensor.scatterGrouped` / `recordBatch` are the tensor-level
scatter and profile hooks.

**LoRA fine-tuning** (`gemma4_train.zig`, pointer depth — §11).
`llm.gemma.gemma4_train.Trainer(targets)` mirrors the qwen3 trainer over the
gemma4 forward: identical `Targets` struct and defaults (q, v), identical
`ignore_index`, and the same member set — `init(ctx, model, lora.Config,
seed)`, `deinit`, `registerAllParams`, `saveAdapters`, `loadAdapters`,
`loadAdaptersWithOptions`, `loss`, `lossExt` + `LossOptions`,
`evalLastLogits`, `n_enabled`, `LayerAdapters` (per-layer geometry sizes the
k/v adapters). Its `Error` set encodes the intentional exclusions checked in
`init`: `PleUnsupported` (PLE models rejected), `SharedKvUnsupported`
(any layer with `has_kv == false`), `RawMoeWeightsRequired` (MoE layers must
retain raw expert blocks — load with `--experts=borrow` or a raw-expert
build; the packed inference-only RHS cannot take gradients), plus
`ExecScopeRequired`, `InvalidSequenceLength`, `LabelLengthMismatch`.

**Runner** (`examples/gemma4.zig` — chat/REPL over the SPM tokenizer
(`llm.spm_tokenizer`) and the generic `llm.chat.Conversation`; sampling
defaults come from the GGUF):

```sh
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --chat "Why is the sky blue?" --experts=borrow
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --repl --system "Answer tersely." --think
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --chat "Why is the sky blue?" --spec          # lossless speculative decoding
zig build gemma4 -Dllguidance=true -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --chat "List three facts about the sky." \
  --json-schema '{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":3}'  # constrained reply (§13.6)
zig build gemma4 -Dgpu=metal -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --chat "..."                                  # MoE expert FFN on the GPU
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  2,651,235 --bench 3 --profile                 # prefill/decode benchmark
```

### 14.5 DiffusionGemma — block text-diffusion (`src/llm/diffusion_gemma/model.zig`)

The `diffusion-gemma` arch is **not autoregressive**: the transformer is
exactly gemma4 (this module reuses gemma4's layer loader and attn/ffn
blocks), but generation denoises fixed-length token canvases and commits them
block-autoregressively. Two forward modes share one weight set:

- `encodeStep(ctx, kv, token_ids, pos0) !void` — causal prefix pass over the
  prompt or a finalized canvas; exists only for its K/V side effect (the lm
  head is skipped), appends and advances the cache, applies the per-layer
  `enc_layer_output_scale`. Same `pos0`/capacity contract as `forwardStep`.
- `canvasForward(ctx, kv, canvas_ids, sc) ![seq, vocab]` — one
  **bidirectional** denoiser pass over the canvas at absolute positions
  `[kv.len, kv.len + C)`. Canvas K/V are written into the cache's scratch
  region past `kv.len` WITHOUT advancing it (the next step overwrites), so
  the cache is read-only from the caller's perspective; logits are returned
  for every row (softcapped). `sc` is the previous step's self-conditioning
  signal (null on the first step); passing one on a GGUF without the
  `self_cond_*` MLP is `Error.SelfConditioningUnavailable`.

`Config = { base: gemma4.Config, canvas_length, eb: EbParams }`;
`Config.fromGguf` reads the gemma4 keys under the `diffusion-gemma.` prefix
plus `diffusion.canvas_length` (required — `Error.MissingCanvasLength`) and
the optional `diffusion.eb_*` overrides of `EbParams` (defaults are the
reference generation_config: `max_steps = 48`, `t_min = 0.4`, `t_max = 0.8`,
`entropy_bound = 0.1`, `stability_threshold = 1`,
`confidence_threshold = 0.005`). Loading additionally requires both per-layer
scales (`layer_output_scale` via the gemma4 layer loader, the diffusion-only
`enc_layer_output_scale` into `Model.enc_scale`) — `Error.MissingLayerScale`
otherwise — and rejects PLE configs. `pub const Error = gemma4.Error ||
error{ MissingCanvasLength, MissingLayerScale, CanvasLengthMismatch,
KvCapacityTooSmall, SelfConditioningUnavailable }`.

Model surface: `loadGguf`, `loadGgufFromFile`, `deinit`, `initKvCache`
(per-layer geometry; capacity must cover prefix + one canvas),
`convertDenseWeightsToF16` (dequantize attention q/k/v/o, the shared dense
FFN, the self-conditioning MLP and the lm head to resident f16 so the
m = 256 canvas GEMMs take the f16 GPU path — the `--gpu-f16` flag; ~4.6 GB
extra resident on 26B-A4B, pointless without `-Dgpu=metal`), `encodeStep`,
`canvasForward`.

The **entropy-bound sampler** is exposed as free functions over the canvas
logits: `SamplerOptions` / `SamplerPass` (owns `results` + an optional
`ScSignal`; `deinit(allocator)`) / `samplerPass(ctx, logits, temp, u,
options)` (per-position argmax, entropy of softmax(z/t) and one multinomial
draw, parallelized over positions with caller-pre-drawn uniforms so results
are thread-count independent; also collects the sparse self-conditioning
candidate lists), `ScSignal` (sparse per-row id/prob lists; `deinit`),
`entropyBoundAccept(results, entropy_bound, order, accepted)` (accept
positions by ascending entropy while the cumulative entropy of the
strictly-lower set stays within the bound). The loop drivers:
`denoiseCanvas(model, ctx, kv, canvas, DenoiseOptions) !DenoiseResult`
(denoise one canvas in place — uniform-random init, temperature schedule
t_max→t_min, per-step acceptance + renoise, stable-and-confident adaptive
stop; `DenoiseOptions{ .eb, .seed = 0, .self_conditioning = true, .sampler,
.on_step, .on_step_user }` with `StepInfo` snapshots feeding the runner's
live inline visualization) and
`generate(model, ctx, kv, prompt_tokens, out_tokens, GenerateOptions)
!GenerateResult` (encode the prompt once, then per block: denoise, trim at
the first EOG token — default ids `{1, 106, 50}` — or a repetition-loop
onset, append the kept tokens, encoder-pass the canvas back into the cache;
`.on_block` callback; returns `{ produced, steps, blocks }`).

```zig
const dg = llm.diffusion_gemma.model;
var file = try fucina.gguf.File.loadMmap(alloc, io, "models/diffusiongemma-26B-A4B-it-Q6_K.gguf");
defer file.deinit();
const config = try dg.Config.fromGguf(&file); // gemma4 hparams + canvas_length + EB sampler
var model = try dg.Model.loadGgufFromFile(&ctx, &file, config);
defer model.deinit();

const prompt: []const usize = &.{ 2, 651, 235 };
var kv = try model.initKvCache(&ctx, prompt.len + 2 * config.canvas_length);
defer kv.deinit();
var out: [512]usize = undefined;
const result = try dg.generate(&model, &ctx, &kv, prompt, &out, .{
    .denoise = .{ .eb = config.eb, .seed = 42 },
    .max_new_tokens = 256,
});
_ = out[0..result.produced];
// requires model assets to run
```

No training entry and no speculative decoding (there is no autoregressive
draft/verify seam). Runner (`examples/diffusion_gemma.zig`; on a TTY the
reply denoises live inline — `--no-visual` disables):

```sh
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "Why is the sky blue? Answer in two sentences." --max 256 --seed 42 --experts=borrow
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --repl --system "Answer tersely."
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "..." --steps 32 --entropy-bound 0.2 --t-max 0.9 --t-min 0.4   # sampler knobs
zig build diffusion-gemma -Dgpu=metal -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "..." --gpu-f16
```

### 14.6 Parakeet ASR (`src/llm/parakeet/`)

NVIDIA NeMo FastConformer speech recognition (110M hybrid TDT+CTC through
0.6B multilingual TDT), ported stage-for-stage against parakeet.cpp/NeMo.
The pipeline is a chain of free functions over one `gguf.File` — there is no
monolithic `Model` struct; `ParakeetWeights` is a lazy name-keyed cache:

| stage | module | role |
| --- | --- | --- |
| front end | `frontend.zig` | WAV → 16 kHz mono f32 → preemphasis → STFT power → log-mel (+ per-feature normalization) |
| subsampling | `subsampling.zig` | stride-2 conv2d stack (`subsampling_factor`, 8x on the shipped models), mel → `[T/8, d_model]` |
| encoder | `encoder.zig` | Conformer layers: rel-pos MHA + conv module + macaron half-step FFNs |
| decoder | `decoder.zig` | CTC argmax-collapse, or LSTM predictor + joint (RNNT/TDT greedy) |
| text | `tokenizer.zig`, `transcription.zig` | SentencePiece detokenize; word grouping, timestamps, JSON |
| streaming | `streaming.zig` | cache-aware chunked encoder + carried-state RNN-T session |

**Config and loading** (`loader.zig`). `Config.fromGguf` requires
`general.architecture == "parakeet"` and reads flat `parakeet.*` keys:
`arch` (a `DecoderArch`: `ctc`, `rnnt`, `tdt`, `hybrid_tdt_ctc`,
`hybrid_rnnt_ctc`, with `hasCtc`/`hasTransducer`/`isTdt` predicates —
describes which weights exist, not which decoder runs), the encoder set
`encoder.{d_model, n_layers, n_heads, ff_dim, feat_in, conv_kernel,
conv_norm_type, subsampling_factor, subsampling_conv_channels,
pos_emb_max_len, xscaling}`, the mel front end
`preprocessor.{sample_rate, n_mels, n_fft, win_length, hop_length, preemph,
mag_power, log_zero_guard, normalize}`, `vocab_size`/`blank_id`, the
predictor `decoder.{pred_hidden, pred_rnn_layers}` +
`decoding.max_symbols`, the joint `joint.{joint_hidden, activation}`, and the
TDT duration table `parakeet.tdt.durations` (required iff the arch is TDT;
`max_durations = 16`). Derived accessors: `vPlus` / `checkedVPlus` (joint
output width = vocab + blank + durations), `subsampledFreq` /
`checkedSubsampledFreq`, `durationsSlice`. Supporting enums: `ConvNorm`,
`Normalize`, `JointActivation`. Streaming-variant models additionally carry
`StreamingConfig.fromGguf` (null for offline models): `att_context_left/
right/style` (`AttContextStyle.regular|chunked_limited`), the
`[step0, step>=1]` schedules `chunk_size`/`shift_size`/
`pre_encode_cache_size` (+ `stepIdx`), `cache_drop_size`,
`last_channel_cache_size`, `valid_out_len`, `drop_extra_pre_encoded`.
Multilingual prompt-conditioned models carry `PromptConfig.fromGguf` (null
otherwise) with `resolveLang` mapping a locale to its one-hot index.
`expectTensor`/`TensorClass` and `validateTensors` gate the tensor inventory
at load (`f32_required` vs `quantizable` — the GGUFs ship f16/q8_0/q6_k/
q5_k/q4_k variants of the big matmuls; norms, biases and the featurizer stay
f32). `loadFeaturizer` returns the `Featurizer` (mel filterbank `fb` +
window, **borrowed zero-copy from the mapping** — valid only while the
`gguf.File` lives); `loadPieces` decodes the SentencePiece table (outer
slice caller-freed, pieces borrow the mapping). `Error` covers
`NotParakeet`, `UnsupportedArch`, `UnsupportedConvNorm`, `InvalidConfig`,
`MissingMetadata`, `TensorNotFound`, `TensorShapeMismatch`,
`TensorDtypeMismatch`.

**Weights** (`weights.zig`). `ParakeetWeights.init(ctx, file)` /
`deinit` — a lazy cache mapping tensor names to `LinearWeight`s, built on
first use; `enableF32Blas` pre-converts f32 GEMM operands for the BLAS path
(the `--f32-cache` flag); accessors `getLinear`, `getLinearF32`, `linear`,
`linearD`, `linearQkvD` (fused QKV), `linearPosAllD` (all layers'
`linear_pos` in one GEMM); free `borrowF32` (alignment-checked zero-copy f32
view of mapped bytes). Sessions borrow the weights struct; it must outlive
them.

**Front end** (`frontend.zig`): `Audio` (+ `deinit`), `loadWav16kMono` /
`loadWav16kMonoFile` (PCM16/24/32/f32, stereo downmix, linear resample to
16 kHz via `resampleLinear`), `preemphasis`, `StftParams`, `Spectrogram`,
`DftBasis` (precomputed direct-DFT basis for the `melSpectrogramFast*`
variants), `stftPower`, `MelParams`, `MelSpectrogram` (feat-major
`feats[m * n_frames + t]`), `melSpectrogram`, `melSpectrogramFast`,
`melSpectrogramFastWithBasis`. NeMo-exact: constant-pad STFT, f64
accumulation, per-feature z-score over the valid frames.

**Subsampling** (`subsampling.zig`): `subsample` / `subsampleWithWeights`
(the offline stride-2 conv stack + linear proj), `streamingSubsample` (the
causal variant), `conv2dPublic` (the shared conv2d entry, also exercised by
tests). **Encoder** (`encoder.zig`): `encode` / `encodeWithWeights` (mel
`[n_mels, T]` → `[T/8, d_model]`, caller-owned `Tensor(2)`), built from
`conformerLayer` = `relposAttention` (Transformer-XL relative-position
attention) + `convModule` + `feedForwardT`, with helpers `relPosEncoding`,
`layerNorm`, `layerNormByName`, `layerNormByNameT`, `linearWT`, `f32Data`,
`attnName`, `convName`.

**Decoders** (`decoder.zig`): `ctcDecode` (frame argmax → `ctcCollapse` /
`ctcCollapseWithMeta`); `tdtDecode` / `tdtDecodeWithWeights` (greedy TDT:
LSTM `Predictor` (`init/deinit/step`) + `Joint`
(`init/deinit/encProjAll/step`), duration head skips frames);
`rnntDecodeFrames` + `RnntDecodeState` (`init/deinit/reset`) — the
carried-state per-chunk variant streaming uses. The batch decoders return
caller-freed `[]i32` token ids; `rnntDecodeFrames` returns `!void` and
appends into a caller-provided `*std.ArrayList(i32)`.
`TokenInfo`/`TokenMeta` optionally collect per-token frame
indices and confidences. **Text**: `tokenizer.detokenize` (SentencePiece
piece join, `▁` → space); `transcription.Word`, `groupWords`, `freeWords`,
`toJson` (per-word timestamps from token frames x `frame_sec`).

Offline transcription end-to-end:

```zig
const pk = llm.parakeet;
var file = try fucina.gguf.File.loadMmap(alloc, io, "models/parakeet/tdt_ctc-110m-f16.gguf");
defer file.deinit();
const cfg = try pk.loader.Config.fromGguf(&file);
const feat = try pk.loader.loadFeaturizer(&file, cfg); // borrows the mmap

var audio = try pk.frontend.loadWav16kMonoFile(alloc, io, "clip.wav");
defer audio.deinit(alloc);
var mel = try pk.frontend.melSpectrogram(alloc, audio.samples, .{
    .stft = .{ .n_fft = cfg.n_fft, .hop = cfg.hop_length, .win_length = cfg.win_length,
               .mag_power = cfg.mag_power, .preemph = cfg.preemph },
    .n_mels = cfg.n_mels,
    .log_guard = cfg.log_zero_guard,
    .normalize_per_feature = cfg.normalize == .per_feature,
}, feat.fb, feat.window);
defer mel.deinit(alloc);

var w = pk.weights.ParakeetWeights.init(&ctx, &file);
defer w.deinit();
var enc = try pk.encoder.encodeWithWeights(&ctx, &file, cfg, mel.feats, cfg.n_mels, mel.n_frames, &w);
defer enc.deinit();
const ids = try pk.decoder.tdtDecodeWithWeights(&ctx, cfg, &enc, alloc, &w, null);
defer alloc.free(ids);

const pieces = try pk.loader.loadPieces(&file, alloc);
defer alloc.free(pieces);
const text = try pk.tokenizer.detokenize(alloc, pieces, ids);
defer alloc.free(text);
// requires model assets to run
```

**Streaming API** (`streaming.zig`). Two layers:

- `StreamingEncoder` (`init(allocator, cfg, StreamingConfig)` / `deinit` /
  `reset` / `step` / `layerStack`) runs the full cache-aware encoder on one
  mel chunk: causal subsampling → drop `drop_extra_pre_encoded` leading
  frames (steps >= 1) → the layer stack with carried caches → slice to
  `valid_out_len` (all frames on the last chunk). Per-layer state is a
  `ConvCache` (depthwise conv tail, `init/deinit/reset`) and a
  `ChannelCache` (attention K/V left-context window,
  `init/deinit/reset/advance`); the windowed attention itself is
  `streamingAttnMask` + `streamingRelposAttention` +
  `streamingConformerLayer`, with `streamingDepthwiseConv` for the conv
  module and `applyPromptKernel` for the multilingual one-hot projection.
- `StreamingSession` (`init(allocator, file, cfg, sc, weights, lang)` /
  `deinit`) owns the encoder caches, the LSTM predictor + joint, the carried
  `RnntDecodeState` and the accumulated output. Feed granularities:
  `feedMel(ctx, file, w, mel, n_mels, t)` windows a whole clip through the
  chunk schedule; `feedMelChunk` processes one pre-windowed mel chunk;
  `encodeChunkPrompted` returns a chunk's encoder frames (+ prompt kernel);
  `feedEncoderFrames` greedy-decodes frames you encoded yourself.
  Non-special tokens accumulate in `session.tokens` (set
  `collect_meta = true` to align `token_meta` for timestamps); `<EOU>`/
  `<EOB>` events are counted in `eou_events` and reset the decoder state for
  the next utterance (decoder-only, matching the reference). `init` returns
  `error.UnknownLang` if a prompt-conditioned model cannot resolve `lang`.

No training entry, no speculative decoding (not autoregressive text).
Runner (`examples/parakeet.zig`):

```sh
zig build parakeet -Doptimize=ReleaseFast -- --model models/parakeet/tdt_ctc-110m-f16.gguf \
  --audio clip.wav --transcribe                       # offline; --json --timestamps for word timing
zig build parakeet -Doptimize=ReleaseFast -- --model models/parakeet/tdt_ctc-110m-f16.gguf \
  --audio clip.wav --stream                           # cache-aware chunked pipeline
zig build parakeet -Dparakeet-mic -Doptimize=ReleaseFast -- \
  --model models/parakeet/tdt_ctc-110m-f16.gguf --mic # live microphone
zig build parakeet -Doptimize=ReleaseFast -- --model ... --manifest files.txt --decoder ctc
```

`--decoder tdt|ctc` picks the head on hybrid models; `--lang XX` selects the
prompt locale on multilingual models; `--threads N` caps the worker team.

### 14.7 Example applications

Beyond the family runners, six applications exercise the library end to end.

**nanochat** (`examples/nanochat/`,
[README](../examples/nanochat/README.md)) is a from-scratch CPU port of
karpathy/nanochat: BPE tokenizer training (rustbpe-equivalent), GPT base
pretraining (grad-accum loop, Muon+AdamW, checkpoint/resume), supervised
fine-tuning on the task mixture, bits-per-byte evaluation, and an
interactive chat CLI with a calculator tool. The port is example-local —
everything composes from the public facade — and every stage is validated
against the fp32 Python reference under a tiered parity ladder.
`zig build nanochat -- tok-train|base-train|sft|eval-bpb|chat ...`.

**lmserve** (`examples/lmserve/`) is an OpenAI-compatible HTTP server over
the in-tree language models: Chat Completions (`POST /v1/chat/completions`)
plus the stateless Responses API (`POST /v1/responses`), with SSE streaming,
JSON-schema/regex/Lark constrained output (`-Dllguidance=true` builds), and
a bounded request queue in front of one sequential inference worker. The
GGUF's `general.architecture` picks the backend (qwen3 / qwen3moe / gemma4 /
diffusion-gemma); `--nanochat <dir>` serves a nanochat checkpoint.
`zig build lmserve -- <model.gguf> [--host H] [--port N]`.

**facedetect** (`examples/facedetect/`,
[README](../examples/facedetect/README.md)) runs the insightface
**buffalo_l** pack — SCRFD det_10g detection (boxes + 5-point landmarks),
ArcFace IResNet-50 recognition (512-d embeddings, 1:1 verification),
GenderAge MobileNet-0.25 attributes, a MiniFASNet x2 anti-spoof ensemble, and
2d106det/1k3d68 dense landmarks — from self-contained GGUFs, as a pure-Zig
port of face-detect.cpp. It is the CNN workout for the core op set: the
hand-mapped nets (`recognizer.zig`, `scrfd.zig`, `genderage.zig`) drive the
public tagged-`Tensor` facade with channel-last `[h, w, c]` `conv2d`,
`pool2d`, `prelu`, `channelAffine` and `upsample` (§4), with GGUF dequant,
layout repack and BatchNorm folding at load; the anti-spoof and landmark nets
replay a GGUF-embedded node list through an app-level graph dispatcher over
`ExecContext` ops. Decision-critical control paths (cv2-exact letterbox,
umeyama alignment, NMS) are verbatim scalar ports; detect/analyze JSON is
byte-identical to the reference, embeddings agree at cosine >= 0.999999.
`zig build facedetect -- detect|embed|verify|analyze|bench ...`.

**locate_anything** (`examples/locate_anything/`,
[README](../examples/locate_anything/README.md)) runs NVIDIA
**LocateAnything-3B** open-vocabulary detection — MoonViT vision tower +
MLP projector + Qwen2.5-3B, detection in token space via coordinate tokens —
from one GGUF, ported from locate-anything.cpp and validated stage by stage
(token streams id-exact in all three decode modes, detections JSON
byte-identical at f32). Everything numeric runs on stock tensor ops: the
interleaved 2D vision RoPE is a hand-filled `RopeTable`, ViT attention is the
bidirectional grouped-attention arm, the MTP block-diffusion mask rides the
additive-bias attention arm, and every linear goes through `LinearWeight`
(so f16/quant arms and BLAS/Metal/CUDA GEMM dispatch apply unchanged).
Decode modes `hybrid` (parallel box decoding with AR fallback), `slow`
(pure autoregressive), `fast` (MTP-only). ~1.2-2.4x faster than the
reference CLI depending on ISA and dtype.
`zig build locate-anything -- detect --model ... --input scene.png --prompt '...'`.

**nam** (`examples/nam/`, [README](../examples/nam/README.md)) is a complete
Neural Amp Modeler ecosystem port: load any upstream-format 0.5.0-0.7.x
`.nam` guitar-amp capture (WaveNet incl. gated/grouped/FiLM variants, LSTM,
ConvNet, Linear, SlimmableContainer), render offline or **play live** through
real audio devices (vendored miniaudio, allocation-free lock-free callback,
MIDI control), append cabinet IRs and multi-stage chains, train new profiles
from a reamp pair (classic/A2/packed WaveNet recipes, ESR validation) and
exchange them losslessly with upstream tooling (GGUF interchange recovers a
byte-identical `.nam`). It exercises the streaming-convolution regime: tiny
L1-resident models where per-block latency dominates — standard WaveNet at
~49 us per 64-frame block on one x86 P-core (~27x realtime), numerically
within 2.3e-6 of upstream `tools/render` with a strict-contract SIMD tanh.
`zig build nam -- live|render|train|profile|bench|devices ...`.

**omnivoice** (`examples/omnivoice/`,
[README](../examples/omnivoice/README.md)) is multilingual zero-shot TTS
(646 languages): a **MaskGIT non-autoregressive decoder** on a Qwen3-0.6B
backbone drives the Higgs Audio v2 codec — HuBERT semantic encoder + DAC
acoustic codec + 8-codebook RVQ at 25 fps / 24 kHz — for auto voice, voice
design (attribute prompts) and voice cloning (reference WAV + transcript).
RVQ codes and MaskGIT token streams are byte-exact vs omnivoice.cpp at F32
with a fixed seed; decoded audio cosine >= 0.99999; 2.3-4.6x faster than the
reference's CPU backend on M1 Max. The example doubles as a library
(`pipeline.synthesize`/`synthesizeStream`, ring-buffered `play.Player`) and
ships a WAV↔RVQ codec tool. `zig build omnivoice -- tts --model ... --codec
... --lang English -o out.wav` (see 14.8 for the flag shape).

**The didactic set** (single files under `examples/`):

- `smoke.zig` (`zig build run`) — the minimal facade round trip: two
  variables, `dot`, `sumAll`, `backward`, gradients printed.
- `spirals.zig` (`zig build spirals`) — two-spirals MLP trained with every
  optimizer (SGD/AdamW/Muon/APOLLO/APOLLO-Mini) + param groups, lr schedule,
  clipping; proves bit-exact checkpoint/resume (§11).
- `finetune.zig` (`zig build finetune`) — the qwen3 LoRA loop of 14.2.1 on a
  built-in pirate SFT set; `--data PATH.jsonl`, `--accum-steps`,
  `--verify-grads` gradient-evidence audit.
- `es_finetune.zig` (`zig build es-finetune`) — the gradient-free twin:
  `fucina.es` over the same trainer forward; `--mode lora|full`,
  `--reward rule|acc|nll`, anchored weight decay.
- `es_spirals.zig` (`zig build es-spirals`) — from-scratch ES on two spirals;
  member-parallel evaluation (one ExecContext + model replica per worker);
  self-verifying (fails below `--target` accuracy).
- `es_ternary_spirals.zig` (`zig build es-ternary-spirals`) — the
  ternary-native ES flagship: packed TQ2_0 genomes are the inference model
  (§10, §11), trained by trit flips on the real int8 kernels; self-verifying.

### 14.8 Example → features → run command

Weights are never bundled; [RUNNING-MODELS.md](RUNNING-MODELS.md) lists the
download source and full flag set for every model row. The table omits
`-Doptimize=ReleaseFast` for width — add it to every real run.

| example | demonstrates | run |
| --- | --- | --- |
| `smoke` | facade: tensors, `dot`, autograd (§3-§5) | `zig build run` |
| `spirals` | optimizers, schedules, checkpoint/resume (§11) | `zig build spirals` |
| `es_spirals` | from-scratch ES, member-parallel eval (§11) | `zig build es-spirals` |
| `es_ternary_spirals` | ternary-native ES on TQ2_0 kernels (§10, §11) | `zig build es-ternary-spirals` |
| `finetune` | qwen3 LoRA SFT, accumulation, data loader (§11, 14.2.1) | `zig build finetune -- --model models/Qwen3-0.6B-Q4_K_S.gguf --steps 30 --rank 8 --alpha 16 --save /tmp/qwen3-lora` |
| `es_finetune` | gradient-free LLM fine-tuning (§11) | `zig build es-finetune -- --model models/Qwen3-0.6B-Q4_K_S.gguf --reward acc --iterations 100 --population 8` |
| `qwen3` | dense+MoE decode, KV cache (f16/q8_0), batch decode, speculative (§13, 14.2) | `zig build qwen3 -- models/Qwen3-0.6B-Q8_0.gguf --chat "..." --no-think` |
| `qwen35` | Gated-DeltaNet hybrid, recurrent cache (14.3) | `zig build qwen35 -- models/Qwen3.5-0.8B-Q8_0.gguf --info` |
| `gemma4` | per-layer KV geometry, MoE experts, SPM chat, speculative (14.4) | `zig build gemma4 -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf --chat "..." --experts=borrow` |
| `diffusion_gemma` | block text-diffusion, EB sampler, live denoise UI (14.5) | `zig build diffusion-gemma -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf --chat "..." --seed 42` |
| `parakeet` | ASR pipeline, streaming encoder, mic capture (14.6) | `zig build parakeet -- --model models/parakeet/tdt_ctc-110m-f16.gguf --audio clip.wav --transcribe` |
| `omnivoice` | MaskGIT TTS, HuBERT/DAC/RVQ codec, streaming WAV | `zig build omnivoice -- tts --model models/omnivoice/omnivoice-base-Q8_0.gguf --codec models/omnivoice/omnivoice-tokenizer-F32.gguf --lang English -o out.wav` |
| `nam` | streaming conv nets, live audio/MIDI, `.nam`/GGUF interchange | `zig build nam -- live profile.nam --ir cab.wav` |
| `facedetect` | channel-last conv2d/pool2d/prelu/upsample CNNs (§4) | `zig build facedetect -- detect --model models/buffalo_l.gguf --input face.png --json` |
| `locate_anything` | VLM: ViT + projector + LM, token-space detection | `zig build locate-anything -- detect --model models/locate-anything-f32.gguf --input scene.png --prompt '...'` |
| `nanochat` | end-to-end GPT: BPE tokenizer training, pretraining, SFT, bpb eval, chat (14.7) | `zig build nanochat -- chat -i <ckpt dir> --tokenizer <tokenizer.bin> -p "..."` |
| `lmserve` | OpenAI-compatible HTTP server: chat completions + responses, SSE streaming, constrained output (14.7) | `zig build lmserve -- models/Qwen3-0.6B-Q8_0.gguf --port 8080` |
| `deepseek2` | DeepSeek V2/V3: MLA compressed KV cache, MoE decode | `zig build deepseek2 -- models/DeepSeek-V2-Lite-Chat.Q8_0.gguf --prompt "..." --gen 64` |
| `glm4moe` | GLM-4.5 family: native MTP speculative decode, streamed experts | `zig build glm4moe -- models/glm45-air/GLM-4.5-Air-Q6_K-00001-of-00002.gguf --prompt "..." --gen 64 --mtp` |
| `deepseek4` | DeepSeek V4 Flash: CSA/HCA trunk, streamed experts, MTP sidecar | `zig build deepseek4 -- <model.gguf> --chat --prompt "..." --moe-stream` |
| `ptqtp_spirals` | float-trained MLP decorated post-training with trit-planes, self-verifying (§10.9) | `zig build ptqtp-spirals` |
| `ptqtp_qwen3` | PTQTP-decorate a Qwen3 GGUF in place, NLL before/after, `--save` GGUF (§10.9, §13.2.1) | `zig build ptqtp-qwen3 -- models/Qwen3-0.6B-Q4_K_S.gguf --planes 2` |
| (tool) `export-gguf` | transcode/re-emit GGUF, merge LoRA adapters (§11, §12) | `zig build export-gguf -- --from-gguf in.gguf --out out.gguf --dtype q8_0` |
