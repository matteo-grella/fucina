<!-- docs-nav: group="Project" title="Architecture" weight=50 -->
# Fucina Zig Architecture

This document describes the current Zig implementation in this tree (`src/`,
`examples/`, `apps/`, `bench/`, `tools/`). It is derived from the actual
source layout and behavior, not from historical design notes. Structure lives
here; the command cheat sheet lives in `AGENTS.md`, and per-model recipes live
in the per-target `examples/<name>/README.md` and `apps/<name>/README.md`
guides indexed by `RUNNING-MODELS.md`. Last reconciled against the tree:
2026-08-24.

## Status

Fucina is an eager, close-to-metal CPU tensor/autograd runtime plus an LLM/ASR
inference stack, written in Zig 0.16. It is multi-dtype: bool/integer scalar
dtypes, `f16`/`bf16`/`f32`/`f64`, and the GGML block-quantized formats (see
`src/dtype.zig`). The public API is the tagged autograd `Tensor` facade
exposed from `src/fucina.zig`. Model families (Qwen3 dense + MoE, Qwen3.5,
Gemma 4, DiffusionGemma, DeepSeek V2 + V4 Flash, GLM-4.5, Inkling, Parakeet
ASR, plus the OmniVoice TTS, LocateAnything VLM, and NAM ports in
`apps/`) run from GGUF weights through the sibling `fucina_models` module
(`src/models.zig`) and the runners in `examples/` and `apps/`. Execution is CPU-first with optional
Metal/CUDA callable-accelerator offload (`-Dgpu=metal|cuda`,
`src/backend/{metal,cuda}.zig`).

The core architecture is internally coherent: production dependencies are
acyclic and direction-banded (machine-enforced — see *Layering And
Enforcement*), execution is eager and explicit, buffers are owned
deterministically, and the autograd surface is unified around public `Tensor`.
What it is not: a general ML framework product with a stable, versioned
external API (see *Current Production Gaps*).

## Layer Stack

Top-down; a band may depend only on bands at or below it:

| Band | Contents |
| --- | --- |
| apps | `examples/**`, `apps/**`, `tools/**`, `bench/**`, `src/bench_raw.zig`, `src/x86dot_check.zig` |
| models | `src/models.zig`, `src/models/**` (the `fucina_models` module) |
| facade | `src/fucina.zig` (the `fucina` module root) |
| ag + training/serialization | `src/ag.zig`, `src/ag/**`, `src/optim.zig`, `src/optim/**`, `src/es.zig`, `src/ptqtp.zig`, `src/gguf.zig`, `src/lora.zig`, `src/safetensors.zig`, `src/state_dict.zig`, `src/training_checkpoint.zig`, `src/param_registry.zig`, `src/weights.zig`, `src/weights/**`, `src/gguf_meta.zig`, `src/ptqtp_gguf.zig` (model I/O) |
| tagged | `src/tag_ops.zig` (tag-ops library) |
| exec | `src/exec.zig`, `src/exec/**` (eager runtime) |
| store | `src/store/**` (disk-streamed block stores: `expert_store.zig`, the out-of-core MoE expert tier) |
| backend | `src/backend.zig`, `src/backend/**` (build-selected numeric kernels behind the `backend/interface.zig` kernel set; the single-implementation fused kernels live beside their ops in `exec/`) |
| tags | `src/tags.zig` (comptime tag algebra) |
| tensor | `src/tensor.zig` (raw tensor) |
| primitives | `src/thread.zig`, `src/parallel.zig`, `src/tuning.zig` (the tuning table over `parallel`'s env readers) |
| core | `src/dtype.zig`, `src/storage.zig`, `src/accelerator.zig`, `src/rng.zig`, and the std-only leaves `src/fpenv.zig`, `src/caching_allocator.zig`, `src/streamconv.zig` |

## Public Surface

`src/fucina.zig` exports:

- `Tensor`: the public tagged/autograd tensor constructor from `src/ag.zig`.
- `ExecContext` (plus `RhsLifetime`, the MoE/`RouterTopKOptions`/`Reduction`/
  `CrossEntropyOptions`/`StandardizeOptions` option types, `UnaryOp`, and
  `RopeMode`/`RopeTable`) from `src/exec.zig`, plus the `expert_store`
  disk-streamed MoE expert tier (`src/store/expert_store.zig`) and the
  `fakequant` low-precision grid round-trips.
- `DType` and the GGML block types (`BlockQ4_K`, `BlockIQ2_XS`, ...) from
  `src/dtype.zig`, plus the quantized RHS container types from the backend.
- `BackendKind` and the backend build/runtime constants
  (`active_backend_kind`, `native_uses_blas`, ...).
- Autograd framework pillars: `checkpoint`/`checkpointWithContext`,
  `noGrad`/`isGradEnabled`/`NoGradScope`, `customVjp`, and
  `gradcheck`/`GradcheckOptions`/`GradcheckResult`.
- `einsumMany`: N-ary multi-index contraction (comptime left-fold of the
  binary `Tensor.einsum`).
- Training/persistence namespaces: `optim`, `es`, `lora`, `gguf`, `rng`,
  `parallel`, `ParamRegistry`, `state_dict`, `safetensors`,
  `training_checkpoint`, plus the `ptqtp` trit-plane PTQ namespace.

The root intentionally does not export the raw tensor or raw autograd
internals. A comptime guard in `src/fucina.zig` makes re-exporting `RawTensor`
at the public root a compile error; in-tree code that genuinely needs the raw
type names it through the `fucina.internal` escape hatch
(`internal.RawTensor`, plus `internal.backend_mod`/`tensor_mod`/`thread_mod`
for exact type identity in the `fucina_models` module, and `internal.gpu` for the
Metal residency/tracing hooks). Microbenchmarks use the separate `bench_raw`
module (`src/bench_raw.zig`).

`Tensor(tags_or_rank)` is the user-facing tensor type. It supports named tags
(`Tensor(.{ .batch, .hidden })`), a numeric rank (`Tensor(2)`, generating axis
tags `._0`, `._1`, ...), and dtype specs such as
`Tensor(.{ .dtype = .u16, .tags = .{ .batch, .seq } })` or
`Tensor(.{ .dtype = .i64, .rank = 2 })`. Rank is part of the public tensor
type at comptime; dimension sizes remain runtime values. `DType` is the public
logical format tag, not a promise that every tensor has one scalar storage
element per logical element: scalar dtypes store `[]Scalar(dtype)`;
block-quantized dtypes store `[]Storage(dtype)` blocks over the last logical
axis.

The `.f32` public tensor branch is the differentiable autograd tensor.
Non-`f32` scalar public tensors are constant typed tensors: storage, tags,
views, broadcasting, gather, narrow/slice, concat, and slice/row updates.
Floating non-`f32` tensors also expose forward-only math; integer and bool
tensors do not expose float math. Block-quantized public tensors are constant
inference tensors: loaded-block construction, `to(.f32)`, embedding-style
`getRows`, and f32 x quantized-RHS tagged dot/matmul when the RHS is stored as
`[free, contract]` — no generic pointwise math, softmax, norms, or autograd.
These boundaries are enforced by the Zig type system through separate public
tensor branches, not by runtime dtype checks. A scalar-tag tensor,
`Tensor(.{})`, is represented internally as a rank-1 raw tensor with shape
`{1}` because the raw tensor layer has no rank-0 shape.

Forward float dtype policy is explicit per operation family
(`computeDType`/`outputDType` in `src/dtype.zig`):

- Pointwise ops preserve input/output dtype. `bf16` computes through `f32`
  because it is stored as bits; `f16` computes as `f16`; `f64` as `f64`.
- Reductions on `f16`/`bf16` compute in `f32` and return `f32`; `f64`
  reductions compute and return `f64`.
- Dot/matmul on `f16`/`bf16`/`f32` accumulates in `f32` and returns the input
  dtype; `f64` matmul computes and returns `f64`.
- Explicit casts are required when a caller wants a different output dtype.

## Source Layout

Core value types and substrate:

- `src/dtype.zig`: comptime dtype metadata, scalar storage mapping,
  block-quantized storage block definitions, float compute/output policy.
- `src/storage.zig`: refcounted typed buffer storage (`BufferOf(dtype)`),
  including borrowed-slice storage with an optional release hook
  (`fromBorrowedSliceWithRelease`) used for device-resident weight bytes;
  storage also owns optional submitted-writer/latest-reader accelerator
  fences and storage-lifetime mapping resources.
- `src/accelerator.zig`: backend-neutral lifetime tokens for
  already-submitted eager GPU work (`Work`) and per-storage mapping caches
  (`Resource`). They contain no operation description or compute graph.
- `src/tensor.zig`: raw tensor value (`TensorOf(dtype)`), shape/stride
  metadata, views, broadcast, reshape, materialization, fixed-rank views.
- `src/tags.zig`: comptime tag/rank algebra (no runtime representation).
- `src/rng.zig`: repo-owned deterministic RNG; the (seed → values) mapping is
  a checkpoint contract (APOLLO projections, dropout masks).
- `src/parallel.zig`: thresholds and CPU-count helpers.
- `src/thread.zig`: thread pool. The worker team stays hot between dispatches
  (spin-then-park); the dependency-chained fork-join mode carries a documented
  exactly-once/exact-count enqueue contract with safety-build-only accounting
  (duplicate detection, stall diagnostics — comptime-elided in ReleaseFast).

Execution runtime:

- `src/exec.zig`: `ExecContext`, the public runtime boundary. One struct
  declares the substrate state (thread-safe allocator, published worker
  team, `BufferPool`, exec-scope stack, MoE decode scratch) and
  carries every op as an alias line (`pub const add = exec_elementwise.add;`)
  grouped by domain; the struct body is the registry, the bodies live under
  `src/exec/`.
- `src/exec/runtime.zig`: the substrate functions of `ExecContext`
  (lifecycle, exec scopes, worker team, tensor allocation primitives,
  `replace`), each taking `*ExecContext` first. Domain modules receive
  `*ExecContext` explicitly (never `self: anytype`), so their code is
  monomorphic; `src/exec.zig` and `src/exec/*.zig` form the one permitted
  root-anchored import cycle (see *Layering And Enforcement*).
- `src/exec/buffer_pool.zig`: the reusable transient-buffer pool leaf.
- `src/exec/` domain modules: `attention.zig`, `matmul.zig`,
  `quant_matmul.zig`, `moe.zig`, `moe_chain.zig`,
  `fakequant.zig` (FP8/FP4/f16 grid round-trips), `elementwise.zig`,
  `row_ops.zig`, `norm.zig`, `softmax.zig`, `loss.zig`, `reduce.zig`,
  `topk.zig`, `stats.zig`, `gather_scatter.zig`, `rope.zig`, `convert.zig`,
  `conv.zig`, `pool.zig`, `shape.zig`. These are not public API; `src/exec.zig` remains
  the runtime boundary.
- `src/exec/moe_gu.zig`: fused gate|up MoE expert kernels over raw GGUF
  stack layouts (packed Q6_Kx4/Q8_0x4 arms, raw Q6_K/Q4_K blocks, GPU
  batch path), reached through the `moeGu*` facade methods; the gemma
  family's tagged wrappers live in `models/gemma/moe.zig`.
- `src/exec/moe_chain.zig`: shared batched-MoE scheduling scaffolding
  (expert-grouped route plan, gather → gate/up → act → down phase-chain
  machinery, chunking helpers, profile timers). Consumed by `exec/moe.zig`
  and by the fused gate|up kernels in `moe_gu.zig`, so scheduler fixes
  land once for every
  family.

Backends:

- `src/backend.zig`: build-selected backend facade; `backend.kernels` is
  the selected provider's kernel set and the shared kernel vocabulary
  (ops, block and RHS types) is re-exported from here.
- `src/backend/interface.zig`: the kernel set by name (`names`,
  `generic_names`, `pool_free_names`) and `conform`, the comptime check
  that both providers export exactly that set with matching signatures.
- `src/backend/cpu.zig` (scalar reference) and `src/backend/native.zig`
  (Zig `@Vector` kernels plus optional CBLAS for GEMM), each exporting
  `pub const kernels`; `src/backend/parity_test.zig` keeps them in
  agreement numerically.
- `src/backend/vector.zig` + `src/backend/vector/`: portable SIMD kernels,
  addressed by child module (`vector.gemm.matmul2DIntoUnchecked`):
  `primitives.zig`, `gemm.zig`, `gemm_blocked.zig` — the BLIS-style blocked
  packed f32 GEMM for the no-BLAS path, `gemm_packed.zig`, `matmul_quant.zig`,
  `elementwise.zig`, `conv.zig`, `pool.zig` — channel-last pool2d/upsample2x,
  `winograd.zig` — F(2×2,3×3) conv transforms for the no-BLAS conv route,
  `batched.zig`; `common.zig` holds `ParallelConfig`, the vector-width aliases
  and the thread-count gates. Every pool-taking kernel takes `pc` first.
- `src/backend/quant.zig` + `src/backend/quant/`: GGML-compatible block
  helpers, dequantization, quantized-RHS containers and dot kernels, addressed
  by child module (`quant.q4_k.matmulQ4_Kx8RhsTile`): `q4_k.zig`, `q5_k.zig`,
  `q6_k.zig`, `q8_0.zig`, `q8k.zig`, `ternary.zig`, `mxfp4.zig`, `cold.zig`
  for the rare formats, `types.zig` (the block and RHS types; `quant.zig`
  forwards these outward), `common.zig` (the shared SIMD primitives). The
  f32 → quantized row encoders live in the format modules behind
  `quantizeRowForDType` in `quant.zig` (byte-exact ggml parity).
- `src/backend/packed.zig` (dense packed-RHS helpers for `f16`/`bf16`
  matmul), `src/backend/ops.zig` (shared op enums),
  `src/backend/quant_tables.zig` (GGML lookup tables).
- `src/backend/metal.zig` + `src/backend/metal/`: the `-Dgpu=metal` GPU GEMM
  provider — Zig host (lazy init, persistent queue, eager-async f32/f16/dense-quant
  completion, work-threshold gates, device-owned weight storage,
  storage-lifetime page wrappers) plus the ObjC shim (`shim.m`) and
  vendored kernels (`mlx_gemm.metal` f32/f16, `ggml_mul_mm.metal`
  dequant-in-kernel).
- `src/backend/cuda.zig` + `src/backend/cuda/`: the Linux/NVIDIA provider —
  dlopen'd driver/cuBLAS, persistent upload/compute/download streams, a
  bounded reusable in-flight slot pool and storage-lifetime host registration
  for eager-async f32/f16/dense quant, managed weight residency, and vendored PTX
  quant/GEMV/attention kernels.
- `src/x86dot_check.zig`: standalone cross-ISA parity checker for the int8 dot
  primitives + Q4_K/Q8_0 dot kernels (per-arm coverage table in its header).

Autograd:

- `src/tag_ops.zig`: tag-semantics op library over raw tensors (see *Tagged
  Tensor Semantics*).
- `src/ag.zig`: autograd module root, exporting the public `Tensor` and the
  framework pillars.
- `src/tuning.zig`: tuning policy — the typed table of every FUCINA_* route
  gate and numeric crossover (read-once env load, measured defaults,
  programmatic pins), and the per-context `Overrides` carried by
  `ExecContext` (`setTuning`).
- `src/ag/tensor.zig`: public tagged/autograd tensor facade — the `Tensor`
  dispatcher, the f32 struct core, and one-line aliases onto the per-domain
  method mixins in `src/ag/tensor/float/` (elementwise, matmul, reduce,
  shape, ...); `src/ag/tensor/typed_constant.zig`: the non-f32 constant
  tensor band; `src/ag/tensor/plumbing.zig`: shared result-finishing and
  dispatch helpers. The `src/ag/tensor/` files never import the facade
  back — they receive it as a comptime parameter (`Self.ag_root` /
  `Mod(ag_tensor)`), keeping the import graph acyclic.
- `src/ag/backward.zig`: re-export facade over the per-domain VJP modules
  in `src/ag/backward/` (mirroring `src/exec/`'s taxonomy, over a shared
  `common.zig`); `src/ag/core.zig`:
  backward-only gradient state and scheduling engine; `src/ag/checkpoint.zig`:
  activation checkpointing (recompute-in-backward); `src/ag/control.zig`:
  no-grad scopes; `src/ag/custom.zig`: the `customVjp` adapter;
  `src/ag/gradcheck.zig`: the finite-difference gradient oracle.

Training and persistence (see *Training And Persistence*): `src/optim.zig`,
`src/es.zig`, `src/ptqtp.zig`, `src/param_registry.zig`, `src/state_dict.zig`,
`src/safetensors.zig`, `src/training_checkpoint.zig`, `src/lora.zig`,
`src/gguf.zig`.

LLM stack (see *LLM Stack*): `src/models.zig` + `src/models/`.

Apps band: `examples/` (single-file teaching programs, one `main.zig` per
directory: `smoke/`, `spirals/`, `es_spirals/`, `gemma4/`, `qwen35/`,
`engram/`, and more), `apps/` (product- and port-shaped programs with their
own tests, shims, and goldens: `run/` (the registry runner), `qwen3/`,
`deepseek4/`, `diffusion_gemma/`, `lmserve/`, `parakeet/`, `omnivoice/`,
`nam/`, `voiceagent/`, `facedetect/`, `locate_anything/`, `nanochat/`,
`finetune/`, `es_finetune/`, `cartridge/`, `cartridge_fleet/`), `tools/`
(`export_gguf.zig`, `check_import_graph.zig`, `check_doc_links.zig`, plus the
benchmark/parity helper scripts), `bench/` (microbenchmarks plus the shared
`alloc.zig`/`timer.zig` helpers).

## Layering And Enforcement

The intended production dependency direction inside the `fucina` module:

```text
fucina.zig
  -> ag.zig, exec.zig, backend.zig, tag_ops.zig, tensor.zig, storage.zig,
     dtype.zig, thread.zig, and the training/persistence + model-I/O
     modules (gguf, weights, gguf_meta, ptqtp_gguf, optim, es, ptqtp,
     lora, rng, parallel, tuning, param_registry, state_dict,
     safetensors, training_checkpoint)

ag/tensor.zig
  -> ag/tensor/ (float/ method mixins, typed_constant.zig, plumbing.zig),
     ag/{core,backward,control}.zig, tags.zig, tag_ops.zig, exec.zig,
     backend.zig, tensor.zig, dtype.zig

ag/tensor/*.zig, ag/tensor/float/*.zig
  -> ag/{core,backward,control,elemental}.zig, tags.zig, tag_ops.zig,
     exec.zig, backend.zig, tensor.zig, dtype.zig, rng.zig
     (never ag/tensor.zig — the facade arrives as a comptime parameter)

ag/backward.zig
  -> ag/backward/ (per-domain VJP modules + common.zig)

ag/backward/*.zig
  -> ag/core.zig, tags.zig, tag_ops.zig, exec.zig, backend.zig (ops),
     tensor.zig, dtype.zig, parallel.zig

tag_ops.zig
  -> exec.zig, tags.zig, tensor.zig

exec.zig
  -> exec/*.zig, backend.zig, tensor.zig, thread.zig, tuning.zig, fpenv.zig

exec/runtime.zig, exec/<domain>.zig
  -> exec.zig (the `ExecContext` type), exec/buffer_pool.zig, backend.zig,
     dtype.zig, parallel.zig, storage.zig, tensor.zig, thread.zig
     (the root-anchored cycle with exec.zig: struct body in the root,
     function bodies in the children)

backend.zig
  -> backend/{ops,packed,quant,cpu,native,gpu,vector}.zig, dtype.zig,
     tensor.zig, thread.zig

backend/gpu.zig (comptime -Dgpu selector, a leaf so native.zig can reach it)
  -> backend/{gpu_provider,metal,cuda}.zig

tags.zig -> tensor.zig
tensor.zig -> storage.zig, dtype.zig
storage.zig -> accelerator.zig, dtype.zig
```

The `fucina_models` module (`src/models.zig` + `src/models/`) sits above the facade:
its files import the `fucina` module (public surface plus `fucina.internal`),
never individual `src/*.zig` files. `build.zig` wires the models module against
the `fucina` module root only (the `bench_raw`/`raw_backend` microbench
modules are separate, apps-band roots).

Enforcement:

- `zig build arch-check` runs `tools/check_import_graph.zig` over the
  production (non-test) import graph of `src/`, `examples/`, `apps/`,
  `bench/`, and `tools/`, and enforces three invariants: no forbidden
  strongly-connected components, zero band inversions, and every sibling
  test file forwarded from a production file. An SCC is permitted only
  when every member is in the same band and one member is the directory
  root of another (`P.zig` with a member under `P/`: `src/exec.zig` with
  `src/exec/*.zig`, the struct-body-in-the-root shape `std.zig` and
  `std/array_list.zig` share). Children cycling without their root, or
  any cycle that crosses a band, fail the build. The apps-band roots are
  scanned because the `apps/` entries are complete model ports rather than
  snippets, and an unforwarded test file there is just as silently dead as
  one in `src/`. A file whose name ends in `_tests.zig`
  but which declares `pub fn main` is an executable root, not a suite
  (`tools/gen_snippet_tests.zig` generates tests), and is exempt from the
  forwarding rule.
  The checker is AST-based and test-aware: `@import`s inside `test`
  declarations, and inside non-pub file-scope decls reachable only from
  tests, are excluded, so sibling-test forwarding stanzas and private test
  helpers do not count as production edges. The step prints the current
  file/edge count; the counts grow with the tree, so they are not pinned
  here; the root-anchored SCC count is printed too (1 today: `exec`).

- The direction bands are the *Layer Stack* table, encoded as `band_table`
  in that same tool: every production file belongs to exactly one band, and
  every production import must point at a band at or below the importer's.
  A file in no band fails the check, so a new `src/` root cannot silently
  acquire whatever band its neighbours have; a production layer inversion is
  a failed build, not a review catch. (The sibling `<name>_tests.zig` files
  intentionally form benign 2-cycles with their sources through the
  forwarding-stanza pattern — see *Build And Verification* — which is why
  any cycle check over this tree must be test-aware, as `arch-check` is.)

## Tensor And Storage Model

The raw tensor layer is intentionally small:

- Differentiable math is `f32`. Raw storage/view helpers are generic over
  comptime dtype, and the executor has typed data movement/indexing kernels
  plus forward-only float kernels for non-`f32` tensors.
- Maximum rank is `8` (`tensor.max_rank`).
- No zero-size or zero-rank raw tensors: every dimension is >= 1
  (`Shape.init` rejects 0) and scalars store as rank-1 `{1}`. This is a
  deliberate torch divergence, not a gap: emptiness fails loud at the
  construction boundary instead of surfacing as torch's empty-reduction
  contract (`mean` -> NaN, `min`/`max` -> runtime error) deep in a graph;
  data-dependent cardinality lives host-side, where Zig represents and
  guards it natively (slices of `[]usize` indices, optionals — the one
  data-dependent op pair, `maskedSelect`/`maskedScatter`, signals no-match
  with the recoverable `error.EmptySelection`); and op/backend contracts
  are defined and parity-pinned only over non-degenerate shapes, which
  keeps that surface small for every current and future backend.
- Shape and stride metadata are inline arrays; no heap allocation for
  metadata.
- Buffers are reference-counted through `storage.BufferOf(dtype)`. Borrowed
  storage (mmap'd GGUF tensors, device-resident bytes) uses the borrow
  constructors; `fromBorrowedSliceWithRelease` attaches a release hook that
  runs when the last reference drops (the Metal weight path uses this to free
  device bytes and evict the shim wrap-cache slot).
- `cloneView()` retains the same buffer and preserves shape/stride/offset.
- `reshape()` is a retained view and requires contiguity.
- `viewWithStrides()` creates checked retained views.
- `broadcastTo()` uses zero strides for broadcast dimensions.
- `data()` and `dataConst()` require contiguity and panic on arbitrary views;
  recoverable callers use `dataChecked()`/`dataConstChecked()`.
- `canTakeInPlace()` is an ownership optimization, not a synchronization
  primitive: valid only when the caller has exclusive access to the handle.

Raw tensors are internal to the public API; `Tensor.asRawTensor()` exposes a
read-only pointer for inspection and interop, and `fucina.internal.RawTensor`
is the canonical in-tree name for allowed-raw zones.

## Execution Runtime

`ExecContext` is the eager runtime boundary. The struct, declared in
`src/exec.zig`, owns:

- a thread-safe allocator wrapper,
- the published worker team (`parallel_pool`, an atomic pointer that
  `pc()` snapshots into the `ParallelConfig` every kernel call receives),
- a reusable `BufferPool` (`src/exec/buffer_pool.zig`),
- a lazily initialized `thread.Pool` (spin-then-park hot worker team),
- the exec-scope stack (`openExecScope`/`closeExecScope`: implicit ownership
  of training intermediates; see `MEMORY-MODEL.md` and `TRAINING.md`),
- the grow-only MoE decode scratch (`moe_scratch`).

The substrate functions that operate on these fields (lifecycle, scopes,
worker team, tensor allocation primitives) are free functions in
`src/exec/runtime.zig`; the struct body aliases them, so `ctx.empty(.f32, ...)`
and `exec_runtime.empty(ctx, ...)` are the same call.

The execution context is responsible for allocating outputs, reusing buffers,
classifying layouts, materializing non-contiguous inputs when required
(large strided materializations are chunked across the worker team over the
raw tensor's run-based `copyRangeTo`), and calling backend kernels only
after validation. Each op is an alias line in the `ExecContext` struct
body that resolves to its domain module; domain modules take `*ExecContext`
plus validated arguments.

Important execution paths:

- Elementwise ops support dynamic-rank dispatch and fixed-rank APIs; fast
  contiguous paths call unchecked backend kernels after shape validation;
  tail-broadcast paths avoid materializing simple broadcast views.
- `take*` APIs reuse unique contiguous inputs in place when safe.
- Reductions, narrow/concat/gather/scatter-add, set-slice/set-rows,
  argmax/topK, softmax/`softmaxExt` (score scaling, additive masks,
  sink mass, ALiBi-style `max_bias`; broadcast masks read by stride), RMSNorm
  (+ fused mul/add and backward variants), LayerNorm/statistics, RoPE (with
  precomputed `RopeTable`), convolutions, and cross-entropy execute as
  first-class eager tensor operations, each specialized by comptime rank/axis.
- `matmul2D`/`matmulTransA2D`/`matmulTransB2D` validate shape, prepare
  contiguous inputs, allocate output, and call backend GEMM variants; `bmm`
  variants support broadcasted leading batch dimensions without materializing
  expanded tensors.
- Attention (`src/exec/attention.zig`) is a tiled flash-style grouped kernel
  with windowed and f16/quantized-KV variants.
- Quantized/fused matmul (`src/exec/quant_matmul.zig`) owns the
  quantized-RHS dispatch, the fused K-quant FFN paths, and the GPU offload
  seams. `RhsLifetime` distinguishes `transient` RHS bytes from
  `stable_process` ones (process-lifetime mmap or registered device-resident
  storage) — only the latter may be cached address-keyed by a backend.
- MoE (`src/exec/moe.zig` + `moe_chain.zig`) executes batched expert FFNs as
  a phase chain over the hot team; the route plan is a counting sort shared
  across families.

The runtime is local and eager. It does not fuse operations, build a planner,
or preallocate an entire model execution schedule.

## Backend Model

Backend selection is build-time (`-Dbackend=native|scalar`; `native` is
the default, `scalar` the reference).
Dispatch is compiled away; adding a variant forces edits through exhaustive
switches.

The two providers meet at one interface, `src/backend/interface.zig`: a
comptime-checked namespace, not a struct of function pointers, because many
kernels are generic over a `comptime` dtype or op. It lists every kernel by
name (`names`), the generic subset (`generic_names`) and the subset that
takes no pool (`pool_free_names`); `cpu.zig` and `native.zig` each export
`pub const kernels = struct { ... }` with exactly that set, and
`backend.zig` runs `interface.conform` on both at comptime (name, parameter
types, return payload, parameter count for generics, and the `pc` rule).
`backend.kernels` is the selected set. The signature rule: a kernel that
needs the worker pool takes `pc: ParallelConfig` as its first parameter;
one that does not use the pool does not take it; the scalar reference
accepts `pc` wherever the native kernel threads on it and ignores it. Exec
calls `kernels.X(ctx.pc(), ...)`, where `ExecContext.pc()` snapshots the
published worker team.

The native backend uses portable Zig `@Vector` kernels for elementwise ops,
reductions, dot, and fallback GEMM; optional CBLAS for large GEMM
(`-Dblas=none|accelerate|openblas|mkl|blis|nvpl|blas`; Accelerate is the macOS
default, `-Dblas=none` selects the pure-Zig blocked packed GEMM); and
arch-gated int8 dot kernels (NEON sdot/smmla, AVX2/AVX-VNNI/AVX512-VNNI) for
the quantized paths. On `-Dgpu=metal` builds, f32/f16 GEMM gates in
`native.zig` and the quantized/MoE entries in the exec layer offload
above-threshold work to `src/backend/metal.zig`.

Dense f32, f16, and stable-weight quantized GPU calls (Q4_K/Q6_K/Q8_0 on
Metal; those plus Q5_K on CUDA) are eagerly
submitted but are not synchronously joined at every op return. Output storage
carries a completion token: another GPU GEMM stays queue-ordered (CUDA can
consume the producer device pointer), while the first CPU data access waits
for host visibility. F16 kernels write the public f32 output directly; dense
quantized linears bind exec input/output storage instead of copying through
the grouped-MoE panels. Final release waits before recycling but skips an
unused D2H. Metal caches a page wrapper per storage allocation; CUDA pools
eight in-flight typed device/tile slots behind persistent
upload/compute/download streams and uses storage-lifetime page registration so
DMA lands directly in exec-owned tensors. CUDA quantized prefill selects
adaptive N32/N64 f16-input/f32-accumulate tensor-core tiles on capable devices;
underfilled dense grids split K into a grow-only per-slot partial buffer and
queue their fixed-order reduction on the same persistent stream. This fills
idle SMs without adding a host fence, graph node, or steady-state allocation.
The same eager tile-table ABI and scalar-FFMA fallback remain. Reusable events
and one cuBLAS handle order the lanes. Grouped MoE still fences at its CPU
gather/GeGLU/scatter data dependencies, but CUDA transfers/kernel/download are
event-chained before the one required host fence. This is completion tracking
for commands that already
exist, not deferred execution or a graph; see `docs/GPU-OFFLOAD.md` for the
ordering proof and measurements.

The allocation contract, precisely scoped:

- Output buffers are always supplied by `ExecContext`; no backend allocates
  tensor outputs. The vector/quant compute leaves (`backend/vector/*`,
  the dot kernels in `backend/quant/*`) are allocation-free.
- The quantized-RHS dispatch tier (`matmul2DQuantizedRhs` in
  `native.zig`/`cpu.zig`) deliberately takes an allocator for per-call LHS
  quantization scratch (f32 activations → Q8_0/Q8_1/Q8_K blocks); the Q8_0
  arm has a 512-block stack fast path (`q8_0_lhs_stack_blocks`). RHS pack
  preparation (x4/x8 lane packs) allocates at load time, not per matmul.
  The exec-tier packed-LHS scratch above this seam is pooled
  (`BufferPool.acquireScratch` byte-slab leases; the pool's byte-slab arm
  also backs all non-f32 `empty` transients); pooling the backend-tier
  scratch below the seam remains an open, bench-gated design task.
- Direct native vector kernels accept a `ParallelConfig` so the execution
  context controls thread-pool ownership.

## Quantized Matmul Boundary

Dense `.i8` is a scalar tensor dtype, not a quantized format. The public
quantized inference path is Tensor-backed: `DType` includes the GGML
block-quantized formats (legacy `q1_0`/`q4_0`/`q4_1`/`q5_0`/`q5_1`/`q8_0`/
`q8_1`, K-quants `q2_k`..`q8_k`, the cold table/nonlinear/FP4 formats
`iq1_s`..`iq4_xs`, `tq1_0`/`tq2_0`, `mxfp4`/`nvfp4`, and the Bonsai ternary
`q2_0`), and
`Tensor(.{ .dtype = .q4_k, ... })` stores logical shape plus GGML-compatible
blocks over the last axis. These tensors are constant inference tensors:
loaded-block construction, dequantize to `f32`, embedding-style `getRows`, and
f32 matmul RHS when the dtype has a registered RHS dot kernel and the tensor
is stored `[free, contract]`.

`DType` is the only identity of a storage format. `src/dtype.zig` defines
every GGML block struct (`dtype.BlockQ4_K`, ...) and the comptime
`block_formats` registry, one row per block-quantized dtype (its block
struct and its logical elements per block); `blockSize`, `blockByteSize`,
`Storage`, `isBlockQuantized`, and `supportsQuantizedMatmulRhs` derive
from it, and GGUF derives its type mapping from the same rows. No second
enum names these formats: every RHS container carries `pub const dtype:
DType` (the W8A8 `QuantizedMatmulRhsI8` is not a block format and carries
only its group-size policy), `AnyQuantizedMatmulRhs`'s union tags are
`DType` tags, and the per-provider GPU `KernelFormatTag` is a kernel-side
ABI value reached through the provider's `kernelTag(dt)`, not an identity.
Layout choices are comptime functions of `DType` and the target:
`backend.PackedRhsFor(dt)` (aliased as `fucina.PackedRhs`) is the one
dtype-to-packed-container map (dense f32/f16/bf16 share the f32 panel;
q8_0 → x4, q6_k → x4, q5_k → x8, q4_k → x2mmla on aarch64+i8mm else
x8), and facade ops dispatch on the container's `dtype` plus its type for
the Q4_K ISA split.

The raw tensor dtype layer owns scalar and block storage. `ExecContext` owns
validation, materialization, allocation, and dispatch. Backends own numeric
kernels. `backend/quant.zig` owns block helpers, dequantization, loaded-block
row access, the interleaved pack layouts and RHS containers, and the
portable kernels shared by both backends; backend dispatch consumes
`AnyQuantizedMatmulRhs` internally. Exec and kernel entries are spelled
once per operation with the format as a comptime parameter or inferred
from the container type: `ExecContext.packMatmulRhs(dt, &w)`,
`matmulPacked(&a, &rhs)`, `rmsNormMulMatmulPacked`,
`splitSwiGluMatmulPacked`, and `gegluQuantMatmulPacked` on the exec side;
`kernels.matmulQuantizedRhs` (plain K-quant containers),
`kernels.matmulPacked` (packed containers), and `kernels.matmulPackedSlice`
(pre-quantized LHS slices) on the backend side. K-quants and the `IQ*`/`TQ*` formats dot
against `Q8_K` activation blocks; `IQ4_NL`, `MXFP4`, and `NVFP4` (like the
legacy formats) use `Q8_0`/`Q8_1` activation blocks. Decode follows GGML
lookup tables, nonlinear codebooks, and E8M0/UE4M3 FP4 scale rules; every
cold decode format is verified bit-exactly against embedded ggml-golden
fixtures (`src/backend/quant/cold_tests.zig`). Matmul uses direct
integer/table dot kernels at the dtype/backend boundary, so these paths do
not materialize dense f32 RHS blocks in the inner loop. Encoders (f32 →
blocks) exist for the K-quants (Q4_K/Q5_K/Q6_K) and legacy formats
(`quantizeRowForDType`, surfaced by `gguf.encodeF32`); the cold formats decode
and matmul but do not encode.

## Tagged Tensor Semantics

`src/tag_ops.zig` is the tag-semantics op library. It applies the comptime
axis-tag algebra from `src/tags.zig` to runtime raw tensors so the public
autograd tensor and the VJPs can delegate tag alignment and named-operation
semantics without duplicating raw view logic. There is intentionally no tagged
tensor *type*: tags are comptime-only data, the single runtime currency stays
the raw tensor, and the library's functions take comptime tag tuples plus
`*const` raw tensors and return owned raw tensors.

Library behavior includes `alignTensorTo`/`permuteTensorTo` view reordering
with zero-stride singleton injection, `broadcastTensorTo`,
`splitAxisView`/`mergeAxesView`, tag-driven broadcasting `pointwise` and
`gatedPointwise`, `sumManyTensor`/`flattenTensor`, and `taggedEinsum` — the
single contraction lowering: the output tag tuple is the whole einsum
equation (shared tags are batch axes when kept, contraction axes when
dropped; operand-private tags are free when kept, pre-summed when dropped),
operands align to an output-derived order as zero-copy views, each side
picks its plain or transposed GEMM/BMM layout at runtime by contiguity, and
the batch group collapses into one bmm axis; `taggedDot` is its
single-contract-tag special case — plus the shared dtype-generic
shape/validation helpers (`pointwiseShape`, `dotResultShape`,
`einsumResultShape`, ...). The public autograd `Tensor` (`ag/tensor.zig`)
implements the named-op surface once and calls into this library; the VJPs
(`ag/backward/`) call the same functions directly on raw gradients.

## Autograd Model

The public `Tensor` in `src/ag/tensor.zig` owns exactly one raw tensor value
and optionally one gradient state:

```zig
value: RawTensor,
grad_state: ?*GradState = null,
```

Constants have no gradient state; variables attach a leaf `GradState`.
Forward execution always happens through the same public tensor operation
path. When no operand requires gradients (or a `noGrad` scope is active),
operations return a no-grad public tensor without retaining graph state.

When gradients are required, `ag/tensor.zig` computes the eager forward
value, creates a backward record from `ag/backward/`, and wraps it in a
`GradState` from `ag/core.zig`. `ag/core.zig` is backward-only: there is no
public `Node`, no `Function.forward`, and no separate raw autograd surface (a
guard test in `src/ag_tests.zig` asserts the legacy declarations stay
removed).

Backward execution (`backwardGrad`/`backwardGradSerial` in `ag/core.zig`):

- Validates every output and pre-allocates the implicit scalar seeds *before*
  `prepareBackwardPass` installs any pending counter, so an error exit during
  seeding leaves the graph re-runnable instead of stranding counters.
- Seeds non-scalar outputs only if a gradient is already present (the facade's
  `backwardWithGrad`, the checkpoint recompute, external `setGrad`); explicitly
  pre-seeded outputs are respected without an implicit `+1` on top. Scalar
  outputs whose gradient appears only mid-pass still accumulate their own
  seed.
- Marks outputs consumed once their pass completes: interior states retain
  their accumulated gradients, so a repeat backward over the same graph would
  compound them — it fails with `AgError.BackwardAlreadyRun` instead (failed
  passes stay unmarked and re-runnable).
- Recursively discovers dependencies, uses per-state pending-gradient
  counters for shared branches, and schedules a state only when all
  downstream contributions are present.
- Uses the `ExecContext` thread pool for async-capable backward records;
  `backwardGradSerial` disables node-level spawning (required by the
  checkpoint recompute's threadlocal nesting guard).
- Passes `needs_grad` into backward records so unnecessary gradients are not
  computed, and accumulates gradients in-place under a per-state mutex.

Backward coverage spans the pointwise/reduction/view/norm/softmax/RoPE/
cross-entropy surface plus conv1d/convTranspose1d, snake, groupNorm,
quantized/f16 dot, windowed and f16-KV attention, and the tagged
contractions (`einsum`/`dot`), whose VJPs exploit closure — the gradient of
a contraction is another contraction, so every contraction backward is
GEMM-lowered (`DotBackward` and the constant-RHS records delegate to the
einsum records); sampling helpers (argmax/topK selection) are intentionally
no-grad. `fucina.checkpoint`/`checkpointWithContext` provide activation
checkpointing (recompute-in-backward); `customVjp` (`ag/custom.zig`) admits
user-defined differentiable ops with raw-tensor forward/backward specs; and
`gradcheck` (`ag/gradcheck.zig`) is the finite-difference oracle used to
validate both built-in VJPs and custom ops.

## LLM Stack

`src/models.zig` is the root of the separate `fucina_models` module (wired in
`build.zig`; it consumes only the `fucina` module).

### Where shared model code lives

Code shared across families has three homes, and the SUBJECT of the code
decides which. The rule is stated at the top of `src/weights.zig`; in short:

| Home | Band | Subject |
| --- | --- | --- |
| `fucina.weights` | core | a weight CONTAINER: building one from GGUF bytes, and multiplying by it (`LinearWeight`, `MoeRhs`, `linearSeq*`, `moe*FfnSeq`) |
| `models/model_common.zig` | models | a GGUF FILE's layout: which tensor names a family's layer trio has, how an embed/head/norm set is read |
| `models/host_ops.zig` | models | raw f32 HOST SLICES, for the host-reference ports that run below the Tensor facade |
| `models/train/lora_trainer.zig` | models | a LoRA TARGET SELECTION: the per-layer adapter set, its A/B tuple, and the dropout seed stream |

`linearSeq*` is forward compute inside the model-I/O module on purpose:
`LinearWeight.linearSeq` is a union dispatch into the per-format arms and
those arms take the container types back, so the container and its multiply
are one mutually-dependent unit. A helper that fits none of these rows wants
a new home with a stated subject, not a fifth un-ruled one.

### The decoder contract and the architecture registry

Every autoregressive text family speaks one comptime-checked surface,
declared in `models/decoder.zig` and asserted by `assertDecoder(Model)` at the
top of the generic layers (`chat.Conversation`,
`speculative.SpeculativeDecoder`, `serving.gguf_chat.GgufChatBackend`,
`models.text.generate`): a `Cache` type (`len()`/`reset()`/`deinit()`, plus
`truncate` iff `caps.rewind`), a `caps: decoder.Caps` value (`rewind`,
`batch`), `initCache(self, ctx, capacity)`, and
`forwardStep(self, ctx, cache, tokens, pos0)` returning the LAST row's
logits as a caller-owned `[1, vocab]` tensor (`forwardStepAllLogits`
returns every row iff `caps.rewind`; `forwardStepBatch` decodes N streams
in lockstep iff `caps.batch`). Conforming: qwen3/qwen3moe, gemma4, SHINE's
`AdaptedModel`, qwen35, deepseek2, deepseek4, glm4moe, inkling; kimi3
(research tier, cache-less whole-sequence forward) stays outside.

`models/registry.zig` is the architecture registry: one comptime table from a
GGUF's `general.architecture` string to the family module (`Family` decls
in each family's `model.zig`: `Model`, the tokenizer module, `load`,
`tokenizer`, `template_fallback`). `serving.open` dispatches over it with
one `inline for`; `registry.familyFor` is the comptime lookup.
`models/text/generate.zig` is the reference generation loop over the contract
(prefill, sample, emit through a `TokenSink` until a stop id, the budget,
or capacity); the qwen35 and inkling chat engines and
`gemma.Model.generate` call it.

Model families live in subdirectories and are exposed as namespaces:

- `models.qwen3.{model,train}` — Qwen3 dense/MoE inference + LoRA fine-tuning;
  `forwardStepBatch` is the batch-N lockstep decode entry (one m=N weight
  pass over N per-stream KV caches).
- `models.qwen35.{model,chat,serving}` — Qwen3.5/Qwen3.6 Gated-DeltaNet hybrid
  plus its ChatML chat/generation engine and serving adapter.
- `models.gemma.{model,train,moe}` — Gemma 4 text + MoE; the MoE kernels
  live in the exec band (`exec/moe_gu.zig`), `gemma.moe` is the tagged
  family surface over them.
- `models.diffusion_gemma.model` — block text-diffusion on the gemma4 backbone.
- `models.deepseek2.model` — DeepSeek-V2 family (MLA + MoE).
- `models.deepseek4.{model,serving}` — DeepSeek V4 Flash (CSA/HCA attention +
  streamed experts via `expert_store`) and its serving adapter.
- `models.glm4moe.model` — GLM-4.5 family, with native multi-token-prediction
  speculative decode.
- `models.inkling.{model,mmproj,chat,serving}` — Inkling hybrid SWA/global
  decoder with banded relative-position bias, plus the image/audio mmproj
  towers, chat glue, and serving adapter.
- `models.parakeet.*` — NeMo FastConformer ASR (frontend → subsampling →
  encoder → CTC/TDT decoder → transcription/streaming).
- `models.text.speculative.{core,mtp,sam_index,recycling,cascade,constrained}` —
  lossless draft-model-free speculative decoding (see `SPECULATIVE.md`),
  including grammar-constrained drafting and native-MTP drafting behind
  the `DraftSource` vtable (`mtp.MtpDraftSource`, glm4moe).
- `models.research.*` — the research tier, one namespace so the facade states
  it: `subq` (decode-path attention evaluator, installed through the
  runner's `AttentionOverride` seam; `SUBQUADRATIC-ATTENTION.md`), `engram`
  (conditional n-gram memory, grafted through the qwen3 trainer's
  `residual_hook` seam; `ENGRAM.md`), `shine`/`shine_train` (context-to-LoRA
  adapters, served by `models.qwen3.shine_serving`), and `kimi3.model` (the
  Kimi-K3 port).

Generic helpers stay flat in `src/models/`:

- `weights.zig`: GGUF weight binding — `LinearWeight` over resident
  f32/f16/bf16 and quantized forms; `LoadOptions{ .gpu_resident }` with
  `loadWithOptions`/`loadForFusion` so pre-fusion parts skip transient device
  residency; device-resident quant weights are owned via storage release
  hooks that free device bytes and evict the Metal wrap-cache slot.
- `ptqtp_gguf.zig`: PTQTP GGUF persistence (docs/PTQTP.md) — decorated
  models save as one standalone TQ2_0 tensor per trit-plane
  (`<name>.ptqtp0/1/2` replaces `<name>`, everything else byte-verbatim)
  behind a `fucina.ptqtp.version` metadata gate; loader pair-detection
  (wired in the qwen3 loaders) rebuilds `.ptqtp` arms bitwise, with fused
  weights row-sliced to source names on save and re-fused through
  `fuseLinear`'s ptqtp arm on load.
- `gguf_meta.zig`: flat loader glue — `metaInt`/`metaFloat`(+`Opt`) readers
  with an explicit `ZeroPolicy` (families disagree on zero-valued keys on
  purpose), plus the comptime-generic `parallelLoadLayers`.
- `kv_cache.zig`: f16-default KV cache (opt-in q8_0 as a capacity option);
  `truncate` is the speculative rewind. `kv_persist.zig`: crash-safe
  append-only KV-cache sidecar, so conversations reopen warm across process
  restarts.
- `cartridge.zig` / `cartridge_fleet.zig`: trainable KV-prefix cartridges and
  per-document cartridge fleets (`CARTRIDGES.md`).
- `engram.zig`: conditional n-gram memory — hashed suffix n-gram tables gated
  into the residual stream of a frozen backbone (`ENGRAM.md`); exposed as
  `models.research.engram`.
- `logit_processor.zig` + `llguidance.zig`: the in-place logit-processing seam
  and the vendored llguidance grammar/JSON-schema engine behind it
  (`CONSTRAINED-DECODING.md`).
- `tokenizer.zig` (byte-level BPE; token-ID-exact pretokenizer chunkers:
  qwen2, qwen35, glm4, joyai), `spm_tokenizer.zig` (Gemma SPM),
  `unicode_categories.zig` (generated tables), `sampler.zig`.
- `data.zig`: SFT dataset/dataloader — `SftText` JSONL/static pairs,
  `encodePair` (template + tokenize + shift + mask), and a deterministic
  `Loader` whose `(seed, epoch) → permutation` mapping is a golden-pinned
  checkpoint contract; the tokenizer parameter is duck-typed so BPE and SPM
  both fit.
- `chat.zig`: `Conversation(comptime Model, comptime Tok)` — genuinely
  generic multi-turn chat over any decoder-contract family with
  `caps.rewind` over the shared `KvCache`, paired with a tokenizer module; `Template` renders ChatML/Llama 3/Gemma 1-3/Gemma 4;
  `Options` includes `extra_stop_ids`, `stop_sequences`, and `speculation`
  (`stop_sequences` compose with speculation: the accept gate scans the
  committed text and the completing token is trimmed, preserving the
  lossless one-draw contract; the init errors are
  `SpeculationWithBatch`/`SpeculationWithReuse`). `sendBatch` runs lockstep batch-N decode
  over N sibling conversations sharing one model via `Model.forwardStepBatch`
  (speculation excluded; ownership contract in
  [§13.8](reference/13-the-model-stack-fucina_models.md#138-chat-srcmodelstextchatzig)).
- `serving.zig` + `serving/`: the serving band. `serving/contract.zig` is
  the model-agnostic contract (`GenerateRequest`/`GenerateResult`, `Caps`,
  the per-family `Backend` vtable); `serving/http.zig` (accept loop, SSE
  stream pipe, Host guard; needs libc on Linux for the `std.c.recv`
  hang-up probe), `serving/scheduler.zig` (bounded FIFO + single inference
  worker), `serving/emitter.zig` (per-dialect delta framing),
  `serving/openai.zig` + `serving/anthropic.zig` (wire dialects), and
  `serving/toolcall.zig` (hermes tool calling) are the transport;
  `serving/gguf_chat.zig` is the generic `GgufChatBackend` engine
  (constraint cache, KV reuse slots + disk tier, RAM guard) for any
  `Conversation`-hosted family; `serving/open.zig` is the load-and-serve
  entry (`serving.open`: GGUF in, ready `Backend` out), dispatched through
  the architecture registry: the `Conversation`-hosted set (qwen3,
  qwen3moe, gemma4) shares one generic engine box, and the engine-hosted
  set (qwen35, qwen35moe, inkling, deepseek4) routes to the family serving
  adapters (`qwen35/serving.zig`, `inkling/serving.zig`,
  `deepseek4/serving.zig`). `apps/lmserve` is the CLI front end and
  keeps only the two non-registry backends (diffusion-gemma, nanochat).

- `src/optim.zig` (facade) + `src/optim/`: SGD/AdamW/Muon/APOLLO, grad clipping, LR schedules,
  `OptimizerSet` param groups; positional `FZT1` tensor snapshots plus named,
  dtype-aware safetensors state dicts with name-matched optimizer state.
  Golden-parity-tested against torch references.
- `src/es.zig`: evolution strategies at scale (gradient-free ES-at-scale,
  arXiv:2509.24372): seed-regenerated gaussian perturbations over registered
  f32/f16/bf16 parameters (facade tensors or a whole `ParamRegistry`, frozen
  entries included — ES needs no GradState), in-place perturb/restore plus
  member-parallel replica materialization, z-scored update with fp32
  accumulation, chunk-parallel kernels bitwise-deterministic for any thread
  count. Golden-pinned by `tools/gen_es_goldens.py` and cross-checked
  bitwise against the reference implementation by `tools/check_es_parity.py`
  (see `TRAINING.md` §13).
- `src/param_registry.zig`: borrows named f32/f16/bf16 tensors for
  checkpointing and optimizer registration; a registered name is an on-disk
  schema path (renames go through `state_dict.LoadOptions.aliases`, never by
  loosening strictness). The trainers (`models/qwen3/train.zig`,
  `models/gemma/train.zig`) delegate their parameter plumbing here.
- `src/state_dict.zig` + `src/safetensors.zig`: the named checkpoint stream
  and its safetensors container.
- `src/training_checkpoint.zig`: canonical checkpoint directory
  (`model.safetensors`/`adapters.safetensors`, native `optimizer.fucina`,
  JSON `trainer_state.json` commit sentinel); the state codec is generic
  over the caller's struct, and the LLM trainers' concrete state is
  `fucina_models.train.trainer_state.TrainerState`.
- `src/lora.zig`: `Adapter(in_tag, out_tag)` over frozen weights; named
  persistence; f32/f16 merge (the fine-tune → merge → quantize → serve loop
  is documented in `TRAINING.md`).
- `src/ptqtp.zig`: post-training quantization to trit-planes — K ∈ {1,2,3}
  ternary planes with per-group scales over packed TQ2_0 (`PTQTP.md`;
  GGUF persistence in `src/ptqtp_gguf.zig`).
- `src/gguf.zig`: GGUF parser + writer (byte-verbatim metadata passthrough,
  llama.cpp-exact offsets; `encodeF32` is the writer-side quantize seam).

## Build And Verification

`build.zig` wires two library modules (`fucina` from `src/fucina.zig`,
`fucina_models` from `src/models.zig`) plus the `bench_raw` (`src/bench_raw.zig`)
and `raw_backend` (rooted at `src/backend.zig`) microbench modules.
`build.zig.zon` names the package `.fucina`, so both library modules are
consumable from another project via `zig fetch` + `b.dependency`
([§2.5](reference/02-toolchain-build-and-project-wiring.md#25-consuming-fucina-from-another-project)). The full step list and options live in `AGENTS.md`;
the verification-relevant steps are:

- `zig build test` (+ `-Dbackend=scalar`, `-Dblas=none`, optimize variants):
  drives every test root — `src/fucina.zig`, `src/models.zig`, and the
  `apps/{lmserve,nam,parakeet,omnivoice,locate_anything,facedetect,voiceagent,nanochat}/main.zig`
  roots (`zig build test-fucina` runs the fucina root alone, the routine
  `-Dbackend=scalar` leg). Parity suites needing local model/reference assets
  are env-gated (e.g. `OMNIVOICE_PARITY`) and skip by default.
- `zig build arch-check`: the production import-graph gate (see *Layering And
  Enforcement*).
- `zig build x86dot-check`: runs the cross-ISA dot parity checker natively
  (ReleaseSafe, follows `-Dtarget`) and builds four compile-only legs
  (x86_64_v3 AVX2, alderlake AVX-VNNI, znver4 AVX512-VNNI, neoverse_v1
  smmla) to catch bit-rot of arms no local substrate can execute.
- `zig build doc-check`: fails when `AGENTS.md`'s doc index names a `.md`
  (root-level, `docs/<name>.md`, `examples/<name>/README.md`, or
  `apps/<name>/README.md`) that does not exist, and existence-checks the
  `examples/<name>/README.md` and `apps/<name>/README.md` references in
  `RUNNING-MODELS.md` the same way (`tools/check_doc_links.zig`).
- The model runners (`run`, `qwen3`, `qwen35`, `gemma4`, `deepseek4`,
  `diffusion-gemma`, `parakeet`, `omnivoice`, `locate-anything`,
  `facedetect`, `nam`, `finetune`, `export-gguf`) double as
  parity/oracle harnesses; `bench*` steps are the perf protocol vehicles
  (`BENCHMARK.md`).

Test organization: behavioral tests live in sibling `<name>_tests.zig` files;
each source file keeps only a one-line forwarding stanza
(`test { _ = @import("<name>_tests.zig"); }`) so the sibling is reachable from
its test root. The one sanctioned exception is tests that must touch non-pub
symbols, which stay inline next to those symbols (e.g. the non-pub dot
kernels in `src/backend/quant/cold.zig`). The forwarding stanzas are why
production files still contain `test` blocks; they are not license for inline
behavioral tests, and `arch-check` ignores the imports inside them.

## Known limitations

- No stable external API contract: the package manifest and 0.x tags give
  consumers a pin (`zig fetch --save git+...#v0.3.0`), not a semver
  stability promise — the public API may change between tags.
- The CUDA backend (`-Dgpu=cuda`, Linux) covers f32/f16 GEMM + quantized
  dense/MoE prefill + opt-in decode GEMV; no attention/KV offload and no
  distributed execution. Mixed-precision training (16-bit params and
  activations, f32 gradients, optimizer master weights) is CPU-side —
  `TRAINING.md` §10.
- No graph fusion or compiler layer — deliberate for now (`AGENTS.md` house
  rules); don't add one without a concrete design.
- Quantized encoder coverage stops at K-quants + legacy formats; the cold
  formats (Q2_K/Q3_K/IQ*/TQ*/FP4) are decode/matmul-only.
- The descriptor runner (`models.qwen3.runner`, docs/RUNNER.md) unifies the qwen3
  family and the glm4moe trunk, and `serving.open` is the load-and-serve
  entry for the Conversation families; the OTHER families still wire their
  own config/loader/decoder (the shared seams are `weights.zig`,
  `gguf_meta.zig`, `chat.zig`, `host_ops.zig`, and `moe_chain`), and no
  descriptor covers recurrent/MLA/hyper-connection vocabularies yet.
- No documented thread-safety contract for users sharing tensor handles
  across threads (the runtime's internal pools are thread-safe; handle
  sharing is not specified).
