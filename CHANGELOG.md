# Changelog

Fucina versions as `0.MINOR.PATCH`: a MINOR release may change public API,
and every change that does is listed here with its one-line rewrite. Per
`docs/DEVELOPMENT.md` §6 there is no alias step before 1.0: renames land
directly and the entry here is the migration guide. The changelog starts at
this point; earlier history is `git log`.

## Stability tiers

- **Stable** (every public change gets a rewrite entry): `tensor`, `ag` (the public
  `Tensor` and autograd pillars), `exec`/`ExecContext`, `gguf`, `weights`,
  `gguf_meta`, `safetensors`, `optim`, `lora`, `parallel`, `tuning`,
  `models.text.serving` (the contract: request/result types and the `Backend`
  vtable), `models.text.chat`, `models.text.tokenizer`, `models.text.kv_cache`.
- **Experimental** (changelog entry only): `es`, `ptqtp`, `models.qwen3.runner`, `models.registry`, the `fucina_serving` module
  (`http`, `scheduler`, `emitter`, the wire dialects, `wire_json`,
  `toolcall`), the `models.text.serving` engine (`gguf_chat`, `open`,
  `adapter_common`, `fleet_serve`),
  `models.text.speculative`, the `models.research` namespace (SubQ, Engram,
  SHINE, kimi3), cartridges, and
  every model family's internal layout.

## Unreleased

### Changed

- Batched matmul parallelizes on the worker team alone: the BLAS arm of
  `kernels.gemmBatched` splits batch ranges over the team (each range runs
  its batches through BLAS on its own thread) instead of running every
  batch sequentially on the caller, and the exec-side executor chunk loop
  (`bmmLoopParallel`, `parallel.bmm_loop_work_threshold`,
  `bmm_loop_max_chunks`) is gone with it. Chunks of one batch used to miss
  the batched-BLAS gate and fall to the single-threaded vector kernel; a
  few-large-batch `bmm` now takes BLAS. The split applies below
  `parallel.blas_batch_split_max_work` per batch; above it BLAS threads
  each batch itself. `Pool.parallelChunksOpts`/`tile.forRangeOpts` take a
  `DispatchOptions{ .park_after }` hint: workers park right after such a
  dispatch instead of spinning, so a range that hands its work to BLAS
  does not compete with BLAS's threads for the cores.
- The contraction backward's right branch runs as one task on the work
  pool (`Pool.spawnWg`) instead of on a dedicated thread: the runtime's
  `dot_backward_worker` and `ExecContext.dotBackwardWorker` are gone, and
  `thread.OneShotWorker` remains the expert store's blocking-I/O reader
  only.
- The two-operand contraction backward split (left branch on the calling
  thread, right branch on the dot-backward worker) is one helper,
  `ag/backward/common.zig` `runContractionBranches`, used by the einsum/dot
  and addDot records. Its gate moved from `exec.parallel_dot_backward_branches`
  to `ag.core.parallel_dot_backward_branches`, beside its only readers.
- MoE is its own band, `fucina.moe` (`src/moe.zig`, `src/moe/`), above
  exec: `moe.MoeRhs`, `moe.MoeBatchProfile`, `moe.expertFfn(ctx, ...)`,
  `moe.expertFfnBatch(ctx, ...)`, `moe.chain` (the shared phase-chain
  scheduling) and the decode-scratch views (`lockDecodeScratch`,
  `carveDecodeScratch`, `carveDecodeChainScratch`, `DecodeScratchView`,
  `DecodeChainScratchView`). `ExecContext.moeExpertFfn`/`moeExpertFfnBatch`,
  `ExecContext.MoeRhs`, `ExecContext.moe_chain` and the `*MoeDecodeScratch*`
  decls are gone (`fucina.MoeRhs` and `fucina.MoeBatchProfile` still name
  the types). The runtime keeps one generic grow-only decode scratch arena
  (`ExecContext.decode_scratch`, `exec.ScratchArena`) that the MoE band
  carves; it no longer knows MoE.
- `Tensor.linearDistill` is gone from the core: the fused linear +
  sparse-soft-target distillation loss is `models.text.linear_distill.
  linearDistill(ctx, x, weight, rows, classes, probs, options)`, a custom
  VJP beside the cartridge trainer (its only consumer). With it go
  `exec.LinearDistillOptions`/`LinearDistillForward`, the exec
  `linearDistillLossStats`/`linearDistillBackwardUpstream` entries, the
  `distillStatsRows`/`distillBackwardRows` kernels and the
  `LinearDistillBackward` record. `customVjp` now releases an `extra`
  that declares `deinit(allocator)` when the record is dropped (or after a
  forward that records no gradient).
- The exec registry's "model-serving" exception group is gone: `snakeRows`,
  `relposShift` and `standardizeValidPrefix` are listed with their
  families (elementwise, gather/scatter, stats).
- The subquadratic evaluator's f16 row-block attention kernels
  (`scoreRows4F16`, `weightedAccumRows4F16`, `vecExpAffineSumInPlace`,
  `vecMaxReduce`) live beside their one consumer as
  `models.research.subq_kernels`; `backend/vector/primitives.zig` keeps
  `dotF32F16` (still `fucina.simd`-adjacent) and the f16 lane widener.
- Shape and stride arithmetic lives in one core leaf, `src/shape.zig`:
  the `Shape` value (with `Shape.from` normalizing array, tuple, pointer
  and slice spellings), `elementCount`/`storageElementCount` over slices,
  `contiguousStrides`, `dispatchRank`, and the `AxisGeometry` decomposition.
  `src/exec/shape.zig` is gone (its conv validators moved into
  `exec/conv.zig`, the ALiBi slope into `exec/softmax.zig`). Internal raw
  tensor decls removed with it: `tensor.elementCountArray`,
  `storageElementCountArray`, `elementCountArrayAssumeValid`
  (`shape.elementCount(&array)` / `shape.product`), `requireSameShapeOf`
  (`tensor.requireSameShape` is dtype-generic), and the deprecated
  `rows`/`cols` accessors.
- The exec axis-op skeleton is stated once per family: `ExecContext.
  dispatchRangeOr` is the one pool-or-serial tail (it synthesizes the pool
  adapter, so the `run*Task` twins beside the row kernels are gone);
  `reduce.zig` folds sum/int-sum/prod through one `reduceAxis`,
  `softmax.zig` drives softmax/logSoftmax/logsumexp through one row-family
  driver, `norm.zig` has one rmsNorm body over optional weight and
  residual and one backward-input body, the split-gated backward pair is
  one body, argmax is the extremum core, and the plain hand-rolled pool
  splits in exec and ag ride the runtime helper. `LogRowsTask`,
  `SplitGluTask` and `SplitGluBackwardTask` are aliases of their twins. The
  exec tail-broadcast and in-place elementwise paths are one row body
  (`opRow`) under one row driver (`tailBroadcastRows`).
- Quantized matmul routing decisions are made once: the rowwise pin is
  the context scope alone (`exec.QuantMatmul.numerics` and its `Numerics`
  enum are gone; the field was never set by a caller, open
  `ctx.pinRowwiseNumerics()` instead); the x4 prefix policy of the K-quant
  lane packs is `backend.quantized_matmul.x4PrefixRows`, shared by the
  exec fused engine and the native tier (the exec engine now takes every
  Q4_K batch of 4 or more rows through the padded x4 kernel, as the native
  tier already did); a lane-packed container carries the loader's raw
  blocks as `raw: ?RawRhs`, so `ExecContext.matmulQuant` makes the
  accelerator attempt on every route and `weights.dense` no longer tries
  the GPU before the dot. `RhsLifetime` is defined beside the containers
  (`backend.quantized_matmul.types.RhsLifetime`; `exec.RhsLifetime` and
  `fucina.RhsLifetime` still name it); the packed weight's lifetime is
  read through `rhsLifetime()`/`setRhsLifetime()`.
- The kernel matrix is spelled once: each `backend/quant/<fmt>.zig` exports
  a `kernels` table pairing an `ops.QuantGemm` request with its tile body,
  `backend.quantized_matmul.gemm` dispatches on that table, and
  `backend.quantized_matmul.supported`/`check` read it
  (`ops.QuantGemm.supported`/`check` and the per-format `gemm` switches are
  gone; `hasCompactColOuter` derives from the table).
- The quantized RHS containers are two generics in
  `src/backend/quant/types.zig`: `CompactRhs(dtype)` (every `.rows`
  container: `allocator: ?Allocator`, `blocks: []const Storage(dtype)`,
  `k`, `n`, `blocks_per_column`, `columnBlocks`) and `LanePackedRhs(dtype,
  pack)` over `PackedBlock(dtype, pack)`. Every `fucina.quant.
  QuantizedMatmulRhs*` name is an alias of one of them, so the rewrite is
  field-level: `QuantizedMatmulRhsQ8_0`/`Q4_0` and the cold formats lose
  their nested `rows` (`.rows.blocks` -> `.blocks`, `.rows.blocks_per_row`
  -> `.blocks_per_column`, `.rows.rowBlocks(i)` -> `.columnBlocks(i)`;
  `QuantizedMatmulRhsQ4_0.allocator` is optional now).
  `AnyQuantizedMatmulRhs` is derived from `dtype.block_formats`; its
  `innerDim`/`outputDim` methods (no callers) are gone. `ag.isPackedRhsType`
  reads the container's `pack`.

## 0.4.0 - 2026-08-27

### Added

- `fucina.Error` and `fucina.TensorError`: the public error vocabulary at
  the root. `TensorError` re-exports the shape/data domain set and gains
  `InvalidArgument` (a non-shape argument failing its own validity check);
  `Error` is the merge of `TensorError`, the graph-control names
  (`UnsupportedGradient`, `MutableDataRequiresNoGrad`, `NoGradientGraph`,
  `ActiveExecScopeUnsupported`), the backward engine's domain
  (`MissingOutputGradient`, `MissingBackwardGradient`,
  `BackwardAlreadyRun`), and `OutOfMemory`, for wrappers that
  thread any fucina error upward. Facade methods keep their precise
  inferred error sets; docs/reference/04 §4.1 documents the vocabulary.

- `ExecContext.matmulQuant` / `matmulQuantInto` (`exec.QuantMatmul`,
  `exec.QuantMatmulLhs`): the one quantized matmul request. `prologue`
  names the fused activation (`.split_swiglu`/`.rms_norm_mul`/
  `.geglu_quant`), `placement` the accelerator policy, `rhs_lifetime` the
  RHS storage guarantee, and `numerics = .rowwise` the per-call
  kernel-pinned batch mode. The body is one row-pinned fallback, one
  fused prologue, one accelerator attempt and one backend call.
- `ExecContext.pinRowwiseNumerics()` / `rowwiseNumericsPinned()`
  (`RowwiseNumericsScope`): the context-scoped rowwise-numerics pin — a
  nesting scope count forcing `QuantMatmul.numerics = .rowwise` on every
  quant matmul (and the batched MoE op) while open. The
  speculative-verify pin, replacing the `pin_rowwise_kernels` bool +
  `pinRowwiseKernels()` setter (see Removed).
- `ExecContext.compactMatmulRhs` / `compactMatmulRhsFromBlocks`
  (`exec.CompactRhsFor(dt)`): the borrowing compact `.rows` container over
  a block-quantized weight tensor's blocks or a raw block slice, for
  `matmulQuant`'s compact arm (any `supportsQuantizedMatmulRhs` dtype; the
  container borrows the blocks and needs no deinit).
- `backend/ops.zig` `QuantGemm` (`{ weight, rhs: RhsPack, lhs: LhsForm,
  order: LoopOrder }` with `supported()` as the one kernel matrix, plus
  `Tile` and the `LhsOf`/`RhsOf` operand types): the quantized GEMM
  request at the kernel seam. Each `quant/<fmt>.zig` exports one
  `gemm(comptime g, out, lhs, rhs, tile)`; `quant.gemm` dispatches on
  `g.weight`; every packed RHS container carries `pub const pack`.
- `optim`: `addFallbackParam`/`addFallbackParamNamed` exist on every
  optimizer type, not only `Muon`/`Apollo`. Without an embedded fallback
  (SGD/Adam/AdamW) they are `addParam`/`addParamNamed` — the one path is
  exactly where a fallback-routed param lands — so a generic registration
  function may call them unconditionally; existing
  `@hasDecl(T, "addFallbackParam")` guards keep compiling, with both
  branches now equivalent on the fallback-less optimizers.
- `state_dict.loadStateDictFromFile` (re-exported as
  `optim.loadStateDictFromFile`) and `ParamRegistry.loadStateDictFromFile`:
  the named state-dict load over a parsed `safetensors.File`, copying out of
  the file's tensor bytes with the same matching, alias remap, strictness,
  and transactional guarantee as the stream form. Over a `File.loadMmap`
  mapping a full-model checkpoint resumes at one model of resident memory
  (the stream form stages the whole frame first); `es-finetune --mode full`
  resumes through it.
- `es.Trainer.applyResumedConfig(config)`: the one way to replace a
  trainer's configuration after registration (a checkpoint resume). It
  re-runs the `init` validation on the whole config, re-sizes every ternary
  slot's undo log to the new flip count, and drops the per-iteration noise
  cache; field assignment into `trainer.config` bypassed all three.
  `es-finetune --load` resumes through it.
- `ExecContext.ScopeStack` + `installScopeStack`/`restoreScopeStack`: the
  exec-scope stack is a value (`ctx.scopes`, replacing the `scope_entries`
  + `scope_depth` fields), and a thread can install a stack of its own for
  scope traffic. The checkpoint recompute runs on such a frame instead of
  under a process-wide lock, so recomputes on different contexts (and
  independent checkpoint nodes of one backward) no longer serialize, and a
  nested `checkpoint` inside a block now recomputes on a nested frame
  instead of failing with `error.NestedCheckpointRecompute` (removed).
- `ExecContext.disableQuantDotGpu()` / `quantDotGpuEnabled()`: the
  quantized-RHS dot's GPU pin is a per-context atomic depth count
  (`ctx.quant_dot_gpu_disabled`) instead of the thread-local
  `ag.control.disableQuantDotGpu` / `isQuantDotGpuEnabled` (removed).
- `fucina-run --spec`: draft-model-free speculative decode for every
  rewind-capable registered family (the cascade SAM + token-recycling
  draft source over the shared `SpeculativeDecoder`; greedy-only, same
  contract as the qwen3 app's `--spec`). Families without KV rewind
  (deepseek2, deepseek4) are rejected with a message.
- `tuning.wasSet(path)`: true when a table leaf carries an explicit value
  (environment variable or programmatic pin) rather than its measured
  default; the loader now keeps the env-supplied leaves as a separate
  shadow. `tuning.gpuQ6Seeding` states the dense-Q6/packed-Q6/qmoe seeding
  rule once for both GPU providers, and programmatic pins now count as
  explicit for it (previously only the raw environment variables did).
- `tuning.Table.pool_profile` (`FUCINA_POOL_PROFILE`): the worker-team
  profiling switch rides the table instead of a direct env read.
- `ExecContext.parallelMap`: the chunked parallel map with a serial
  fallback that `optim` and `es` each carried privately, now one context
  method over `dispatchRangeCapped`.

- `fucina_serving`: the serving transport is its own exported module
  (`src/serving.zig` + `src/serving/`): the HTTP server, SSE emitter,
  OpenAI/Anthropic wire dialects, hermes tool calling, and the request
  scheduler, moved out of `fucina_models` (`src/models/text/serving/`).
  Model-free by construction: it depends on `fucina_models` only for the
  serving contract and chat message types, and the arch-check band table
  enforces the direction. The contract, the generic GGUF chat engine
  (`gguf_chat`), and the `open` load-and-serve entry stay on
  `models.text.serving`. Consumers of the moved namespaces switch from
  `@import("fucina_models").text.serving.http` (etc.) to
  `@import("fucina_serving").http`; `zig build test-serving` is the
  band's solo test root.
- `apps/`: the example tree splits into two homes. `examples/<name>/` is
  teaching code (one `main.zig`, no test files, no C shims, no vendored
  assets); `apps/<name>/` is everything with product or port shape
  (multi-file, own tests, shims, goldens, or a real CLI surface). Under
  `apps/`: omnivoice, nam, nanochat, voiceagent, facedetect,
  locate_anything, qwen3, parakeet, cartridge, cartridge_fleet, lmserve,
  finetune, es_finetune, deepseek4, diffusion_gemma. Build-step names,
  executables, and test-root steps (`test-nam`, `test-omnivoice`, ...) are
  unchanged; only the paths moved.
- `apps/run` (fucina-run): one GGUF runner over `models.registry`; arch
  auto-detected, decoder-contract completion/chat/REPL/NLL/bench plus a
  generic parity surface (`--tokenize`, `--logits-out`, `--compare-logits`,
  `--step1`), the shared `--moe-*` levers, the deepseek2 MLA/DSA dials,
  glm4moe `--mtp` (the library `MtpDraftSource` + `SpeculativeDecoder`
  loop), and the inkling multimodal tower and wire-format chat. It replaces
  the deepseek2, glm4moe, and inkling single-file runners; rewrites:
  `zig build deepseek2 -- ...` -> `zig build run -- ...`,
  `zig build glm4moe -- ... [--mtp]` -> `zig build run -- ... [--mtp]`,
  `zig build inkling -- ...` -> `zig build run -- ...` (same flags). The
  `run` step previously aliased `smoke`; `zig build smoke` remains and
  `zig build run` is now the runner.
- `models.decoder`: the autoregressive text-decoder contract — `Caps`
  (`rewind`, `batch`) plus the comptime `assertDecoder(Model)` the generic
  layers (`models.text.chat.Conversation`, `models.text.speculative.SpeculativeDecoder`,
  `models.text.serving.gguf_chat.GgufChatBackend`, `models.text.generate`) now assert. A
  conforming model exposes `Cache` (`len()`/`reset()`/`deinit()`, plus
  `truncate` iff `caps.rewind`), `caps`, `initCache(self, ctx, capacity)`,
  `forwardStep(self, ctx, cache, tokens, pos0)` returning the LAST row's
  logits as a caller-owned `[1, vocab]` tensor, `forwardStepAllLogits`
  iff `caps.rewind`, and `forwardStepBatch` iff `caps.batch`. Conforming:
  qwen3/qwen3moe, gemma4, SHINE's `AdaptedModel`, qwen35, deepseek2,
  deepseek4, glm4moe, inkling; kimi3 (research tier) stays outside.
- `models.registry`: the architecture registry — one comptime table from a
  GGUF's `general.architecture` to the family module (`Family` decls in
  each family's `model.zig`: `Model`, the tokenizer module `Tok` +
  `Tokenizer`, `load(ctx, file, options)`, `tokenizer(allocator, file)`,
  `template_fallback`); `registry.familyFor(arch)` is the comptime
  lookup and `serving.open` dispatches over the table.
- `models.text.speculative.mtp.MtpDraftSource(Model)`: native MTP (`nextn`)
  drafting behind the `DraftSource` vtable, so the shared decoder's
  verify loop drives glm4moe self-speculation (`zig build run -- <gguf> --mtp`
  decodes through `SpeculativeDecoder` instead of a hand-rolled
  draft/verify/commit/rewind loop). deepseek4's MTP sidecar keeps its own
  loop: its `Session` rewinds by snapshot/restore, not `truncate`.
- Family serving adapters in the library: `models.qwen35.serving`,
  `models.inkling.serving`, `models.deepseek4.serving` (moved from
  `apps/lmserve/backend_{qwen35,inkling,deepseek4}.zig`, which are
  deleted); `serving.openFromFile` now serves the qwen35, qwen35moe,
  inkling, and deepseek4 architectures. `models.text.serving.OpenOptions` gains
  `moe_stream` (the deepseek4 streamed-experts levers) and
  `models.text.serving.Opened` gains `expert_store` (the host's exit-time report
  seam); both types now live in `serving/contract.zig` with
  `samplingFromGguf` (the `models.text.serving.*` re-export paths are unchanged).

- `fucina.ParamView`: the per-entry view `ParamRegistry.view` returns
  (name, dtype, shape, mutable byte view, trainability) is now nameable
  at the root; it was returned but unexported.
- `models.qwen3.runner.Error.WrongBlockStyle`: the fused entries (`forwardStep*`,
  `forwardLastLogits*`, `initCache`) reject a `.host_reference` model
  instead of silently running zero layers and returning embedding-only
  logits; `hostStep`/`initHostCache` answer it (was `InvalidConfig`) on a
  `.fused` model.
- qwen35: the multi-token fused-down FFN fast path now covers Q4_K (x8
  targets), Q5_K, and Q6_K down projections (it was Q8_0-only), and every
  projection loader detects persisted PTQTP planes (decorated qwen35
  GGUFs previously loaded their base tensors, dropping the decoration).
- `exec/backend`: `div` joins the backend elementwise surface
  (`divContiguousIntoUnchecked`), so the contiguous binary dispatch no
  longer special-cases it through an exec-local kernel.

### Removed

- The deprecated `ExecContext` quantized matmul spellings, each a thin
  wrapper over `matmulQuant`/`matmulQuantInto` (one entry per name; on
  the tensor/blocks forms the runtime `QuantizedMatmulOptions` map onto
  the request as `allow_gpu = false` → `.placement = .cpu` and
  `rhs_lifetime` carried over unchanged):
  - `ctx.matmulPacked(a, rhs)` →
    `ctx.matmulQuant(.{ .plain = a }, rhs, .{})`.
  - `ctx.matmulPackedInto(out, a, rhs)` →
    `ctx.matmulQuantInto(out, .{ .plain = a }, rhs, .{})`.
  - `ctx.splitSwiGluMatmulPacked(gate_up, rhs)` →
    `ctx.matmulQuant(.{ .plain = gate_up }, rhs, .{ .prologue = .split_swiglu })`.
  - `ctx.rmsNormMulMatmulPacked(x, w, eps, rhs)` →
    `ctx.matmulQuant(.{ .rms_norm = .{ .x = x, .weight = w, .eps = eps } }, rhs, .{ .prologue = .rms_norm_mul })`.
  - `ctx.gegluQuantMatmulPacked(gate, up, rhs)` →
    `ctx.matmulQuant(.{ .gate_up = .{ .gate = gate, .up = up } }, rhs, .{ .prologue = .geglu_quant })`.
  - `ctx.matmul2DWithQuantizedTensorRhs(dt, a, rhs, opts)` →
    `const compact = try ctx.compactMatmulRhs(dt, rhs);` then
    `ctx.matmulQuant(.{ .plain = a }, &compact, .{ ... })`.
  - `ctx.matmul2DWithQuantizedBlocksRhs(dt, a, blocks, n, k, opts)` →
    `const compact = try ctx.compactMatmulRhsFromBlocks(dt, blocks, n, k);`
    then `ctx.matmulQuant(.{ .plain = a }, &compact, .{ ... })`.
  - `exec.QuantizedMatmulOptions` (the wrappers' runtime options) →
    state the request on `exec.QuantMatmul`.
  - `exec_quant_matmul.MatmulPackedOutput` (the `matmulPacked` return-type
    helper) → the entries return `!Tensor`.
  - `kernels.matmul2DTQ2_0F32RhsInto(pc, out, lhs, rhs, m, n, k)` →
    `kernels.gemm2D(pc, .{ .weight = .tq2_0, .lhs = .f32 }, out, lhs, rhs, m, n, k)`
    (`gemm2D`, the parallel `ops.QuantGemm` request entry, joins the
    conformed `kernels` set).

  The `try*` GPU attempts (`tryMatmulQuantRhs`, `tryMatmulTernaryFolded`,
  `tryMatmulQuantRhsSharedInput`) stay: they take raw quantized bytes and
  a decline-to-CPU contract that `matmulQuant` does not subsume.
- `ExecContext.pin_rowwise_kernels` and `pinRowwiseKernels(on)`: the raw
  bool + setter give way to the request vocabulary. Rewrite: pass
  `.numerics = .rowwise` on the `QuantMatmul` request of the calls that
  must pin, or — around a whole verify forward whose model internals you
  do not thread options through — open the context scope:
  `var pin = ctx.pinRowwiseNumerics(); defer pin.close();` (the query is
  `ctx.rowwiseNumericsPinned()`; the backing field is the open-scope
  count `rowwise_numerics_pinned`). Semantics are unchanged: batched
  quant matmuls reproduce the m == 1 numerics bitwise while pinned, the
  lossless-speculation property.
- `fucina.native_uses_accelerate`: derivable from the facts that remain.
  Rewrite: `fucina.native_blas_kind == .accelerate`.

- Single-family ops leave the public `ExecContext` for the models band
  (`fucina_models`), each next to its one consumer. Rewrites:
  `ExecContext.moe_gu` / `ctx.moeGuDecodePacked` / `ctx.moeGuBatchPacked` /
  `ctx.moeGuDecodeRaw` / `ctx.moeGuBatchRaw` →
  `models.gemma.moe.moe_gu` / `models.gemma.moe.decodePacked` /
  `.batchPacked` / `.decodeRaw` / `.batchRaw` (free functions taking
  `*ExecContext` first; `RawExpertWeights` stays
  `models.gemma.moe.RawExpertWeights`); `exec.delta_attention` /
  `ctx.kdaRecurrent` →
  `models.research.kimi3.model.delta_attention` and its `.kdaRecurrent`
  (`src/models/research/kimi3/delta_attention.zig`);
  `ctx.yarnBlendInvFreqsF64` → `models.ops.yarnBlendInvFreqsF64(ctx, ...)`
  (`src/models/ops.zig`). The shared MoE DECODE/BATCH engines
  (`exec/moe.zig`, `exec/moe_chain.zig`, `MoeRhs`) stay on `ExecContext`.
- `dtype.isFloat` (module-internal spelling of the same predicate).
  Rewrite: `dtype.supportsForwardFloatMath(dt)`.
- `ExecContext.reserveScopeSlot`, `adoptScopeValueAssumeCapacity`,
  `adoptScopeNodeAssumeCapacity`, and `ScopeNodeDestroy`. Rewrite:
  `try ctx.adopt(&.{ ctx.bufferEntry(dtype, buffer), ... })` with
  `ScopeEntry{ .ptr, .release }` values, before the value hand-off.
- `error.ActiveExecScopeRequired` and the exec-scope requirement of the
  composed facade ops (`nllLoss`, `l2Normalize`, `cosineSimilarity`, `norm`,
  `normAll`, `maskedSelect`, `maskedScatter`, `select`, multi-axis `slice`,
  multi-tag `reshape`, `rollBy`, `shiftBy`, `trace`, `diag`, `diagEmbed`,
  `constantPad2d`/`zeroPad2d`, `stack`, `unbindInto`, `einsumMany`,
  `conv2dRelu`): they differentiate with or without a scope. Rewrite: delete
  the `openExecScope` you added only to satisfy it.
- `models.train.trainer_state.TrainerState.{es_ternary_flip_rate,
  es_ternary_update_fraction, es_ternary_update_decay}`: no trainer wrote
  or read them. The ternary knobs stay checkpoint contracts that a resumed
  run re-passes identically (`docs/TERNARY.md`); an older checkpoint
  carrying the keys still loads (unknown keys are ignored).
- Every deprecated alias, ahead of the one-MINOR schedule: the policy is now
  "no alias step before 1.0" (`docs/DEVELOPMENT.md` §6). Rewrites:
  `fucina.BlockQ*` / `fucina.BlockIQ*` / `fucina.BlockTQ*` /
  `fucina.BlockMXFP4` / `fucina.BlockNVFP4` / `fucina.QuantizedMatmulRhs*`
  / `fucina.q8_0_block_size` / `fucina.supports_q4_k_mmla` → the same
  names under `fucina.quant`; `fucina.simd.{vecScale, vecMaxReduce,
  dotF32F16, scoreRows4F16, vecExpAffineSumInPlace, weightedAccumRows4F16}`
  → `fucina.internal.backend_mod.vector_impl.*` (backend internals);
  `fucina.weights.GroupedQ8_0RhsX4` → `fucina.quant.QuantizedMatmulRhsQ8_0x4`;
  `models.gemma.gemma4` / `models.gemma.gemma4_train` / `models.pockettts.pocket` →
  `models.gemma.model` / `models.gemma.train` / `models.pockettts.model`; build
  options `-Dbackend=cpu` → `-Dbackend=scalar`, `-Daccelerate=true|false`
  → `-Dblas=accelerate|none`.
- `fucina.PackedRhsLayout` and the `layout` decl on every packed RHS
  container: `DType` is the one identity of a storage format, and the
  container type is the layout. Every RHS container carries
  `pub const dtype: DType` instead; rewrite a `switch (rhs.layout)` to a
  `switch (@TypeOf(rhs).dtype)`, and a `.q4_kx2mmla` vs `.q4_kx8` check to
  `@TypeOf(rhs) == fucina.quant.QuantizedMatmulRhsQ4_Kx2Mmla` (or compare
  against `fucina.PackedRhs(dt)`, the ISA-best container for `dt`).
  `Tensor.packRhsLayout(ctx, layout)` → `Tensor.packRhsAs(ctx, Rhs)` with
  the container type (`fucina.quant.QuantizedMatmulRhsQ4_Kx8`).
- Kernel-tier format enums, none reachable from the `fucina` root:
  `QuantizedMatmulFormat`, `QuantizedMatmulKernel`, `QuantizedMatmulTraits`,
  `QuantizedStorageLayout`, `QuantizedScaleLayout`, `matmulTraits`,
  `matmulTraitsRuntime`, `formatForDType`, `supportsMatmul`, the
  `format`/`traits` decls on the RHS containers (`X.dtype`,
  `dtype.blockSize(X.dtype)`, `dtype.blockByteSize(X.dtype)`,
  `dtype.supportsQuantizedMatmulRhs(dt)`), `PackedMatmulFormat` /
  `preferredRhsFormat` in `backend/packed.zig` (`PackedDenseRhs` carries
  `dtype`), and `backend/packed_layout.zig`. The
  `AnyQuantizedMatmulRhs` union tags are `DType` names (`.ggml_q4_k` →
  `.q4_k`; `.fucina_w8a8_rhs` unchanged). The GGML block structs have two
  paths, `fucina.quant.BlockQ4_K` (public) and `dtype.BlockQ4_K`
  (definition); the `backend.zig` and `backend/quant.zig` forwards are gone
  (`fucina.internal.backend_mod.BlockQ4_K` → `fucina.quant.BlockQ4_K`).

### Changed

- `gelu_quant`'s SIMD lanes evaluate `vtanhf`, a musl-faithful tanhf port
  on `@Vector` lanes over a shared musl expm1f body (`vexpm1f`), instead
  of the scalar body per lane: every lane still reproduces the ggml
  f16-LUT bytes (the sweep test now also pins every adjacent-f16 midpoint
  and dense off-grid f32 bands, validating the input rounding step), at
  4.7 ms per 1M lanes single-thread against 12.8 ms for the per-lane form.
  The `elu` lanes ride `vexpm1f` the same way (byte parity with
  `std.math.expm1`, swept over every finite f16, the saturation cut, and
  the specials).

- The elementwise VJP map splitter (`ag/backward/elementwise.zig`) rides
  `backend.tile.forRange` instead of its own hand-rolled Task array: the
  gate (length threshold, pool presence, thread count) stays with the VJP
  helper, the proportional `i * total / n` boundaries are unchanged, so
  the split is bitwise-neutral. `backend.tile` is the range splitter's
  exported seam.

- The quantized accumulate scaffolding is spelled once: the per-format
  tier-dispatch ladders and lane-rows loop shells of q4_k/q5_k/q6_k/q8_0
  ride `quant/common.zig`'s `accumulateTier` and `accumulateLaneRows`
  (comptime-resolved; per-format bodies stay beside their formats).
  Codegen is unchanged: the q4_k x8 and q8_0 x4 tile streams and the whole
  proof binary are byte-identical before/after, and the x86dot-check
  checksum is unchanged.

- `ExecContext` embeds its runtime substrate as one struct field:
  `rt: exec.Runtime` (declared in `src/exec/runtime.zig`) carries
  `thread_safe_allocator`, `allocator`, `parallel_pool`, `buffers`,
  `tuning`, the `work_pool` triple, the `dot_backward_worker` pair,
  `scopes`, and `fp_env_at_init`; the model/session state
  (`quant_dot_gpu_disabled`, `rowwise_numerics_pinned`, `moe_scratch`)
  stays a direct field set. The PINNED property (self-referential
  allocator; never copy or move an initialized context) is stated on
  `Runtime`. Rewrites: `ctx.allocator` → `ctx.allocator()` (the one
  forwarding accessor, kept because the reference documents the
  spelling in user code); every other substrate field access
  `ctx.<field>` → `ctx.rt.<field>` (`ctx.buffers` → `ctx.rt.buffers`,
  `ctx.tuning` → `ctx.rt.tuning`, `ctx.scopes` → `ctx.rt.scopes`, ...).
  The method surface (`workPool`, `pc`, `setTuning`, `openExecScope`,
  ...) is unchanged.
- The fused row kernels (softmax/logsumexp/log-softmax rows and their
  strided inner-lane arms, layer/RMS-norm rows and backward stats,
  cross-entropy and distillation rows, dropout, scatter-add, the gated
  activations and the fused activation+quantize workers) live in
  `backend/vector/rows.zig`, no longer in `exec/row_ops.zig`: the serial
  kernel entries are registered in the conformed `backend.kernels` table,
  the Task payloads and `run*Task` pool adapters are the `backend.rows`
  seam, and on `-Dbackend=scalar` builds every SIMD entry selects its
  serial twin in `rows.scalar` (held to the entries by
  `backend/parity_test.zig`; the inner-lane twins reproduce the entries'
  bytes). The `exec` domain modules keep validation, layout, allocation
  and dispatch only. Native builds are bitwise-unchanged (the moved
  bodies are textually identical; codegen histograms are identical on
  `softmaxRows` and the attention decode head kernel).
- The attention kernel bodies (the per-query head/pair units over
  f32/f16/q8_0 KV, the query-tiled online-softmax prefill kernel, the
  tiled backward and its BLAS-strip variant, the multi-stream ragged
  decode walker, the dK/dV plane reduction) live in
  `backend/vector/attention.zig`, no longer in `exec/attention.zig`: the
  kernel entries join the conformed `backend.kernels` table, the Task
  payloads/adapters/tile constants are the `backend.attention` seam, and
  on `-Dbackend=scalar` builds the f32/f16 forward entries and both
  backward routes select serial three-pass twins in `attention.scalar`
  (the q8_0 per-query arms compose the quant kernels' own scalar tiers);
  `backend/parity_test.zig` holds entries and twins together.
  `exec/attention.zig` keeps `groupedAttention`/`groupedAttentionBackward`
  and the dispatch/validation tier. Native builds are bitwise-unchanged
  (identical codegen histograms on the decode head kernel).
- The remaining exec `@Vector` bodies are backend kernels too: the
  vectorized dtype casts (`castF32ToF16`/`castF16ToF32`/`castF32ToBf16`/
  `castBf16ToF32`, now `backend.kernels` entries — `optim`'s 16-bit
  master-weight mirrors call them there, no longer through
  `exec/convert.zig`), the last-axis extremum and variance row kernels,
  the fused rms-norm+rope contiguous pair strip, the directed inclusive
  scans (`kernels.scanRows`/`scanColumns`, with the `-Dvector-scan` gate
  inside the kernel and `ops.ScanOp` as the shared vocabulary), and the
  masked-reduce select row. Each carries its serial reference arm; the
  elementwise ones are bit-exact against it by construction.
- Every production reach past the backend facade is gone: the ~55 sites
  in exec/ag/weights/models/store that named `backend.quantized_matmul.
  <fmt>.<fn>` or `backend.vector_impl` internals now go through the
  conformed `backend.kernels` table (31 per-format quantized
  row/pack/tile kernels and `matmul2DTQ2_0F32RhsInto` registered),
  `backend.ParallelConfig`, `quantized_matmul.blockCountForDType` (the
  `q8k.qkBlockCount`/`q8_0BlockCount` spellings), or the curated
  `backend.simd` seam (`dotF32F16` added for the SubQ research kernels).
- `backend/vector/tile.zig` is the one range splitter behind the vector
  kernels' parallel dispatch (`forRange` for disjoint-write splits,
  `reduceRange` for partial folds): the 39 hand-rolled per-kernel Task
  structs and their `runParallel*`/`run*Task` splitters in
  `vector/elementwise.zig`, `vector/gemm.zig` and `vector/conv.zig` are
  gone (Task-struct decls 19/10/10 to 0/0/0). Split points
  (`i * total / thread_count`), per-chunk iteration order and every
  measured thread-count gate are unchanged, so results are bit-identical;
  `matmul_quant.zig`'s `QuantizedRhsParallel` keeps its grouped/gated
  policy as the richer sibling.
- `store/expert_store.zig` is now the facade over five concern files —
  `store/io.zig` (platform I/O shims + the store error set),
  `store/geometry.zig` (`StreamedQuant`/`Proj`/`ProjSpec` and the layout
  math), `store/tiers.zig` (slot/LRU+heat state, the pinned tier, the
  pilot staging ring, the striped L2 tier and its CL2F index),
  `store/policy.zig` (mirror routing, cache-aware selection, repin,
  `memAvailableBytes`), `store/persist.zig` (the FUCEXPT1/FUCTRCE1
  formats). `fucina.ExpertStore` and every `fucina.expert_store` re-export
  keep their names; `readExpert`/`findCached`/`routeCopy`/`copyFd` became
  `pub` module-internal seams (documented as not part of the consumer
  contract).
- `weights.LinearWeight` is a five-container union — `dense` (f32/f16/bf16),
  `quant` (every other GGUF block format, dtype-erased behind a vtable
  built at load), `packed_quant` (q4_k/q5_k/q6_k/q8_0 with their packed
  RHS), `ptqtp`, `tq2_0_fx4` — instead of one union arm per GGUF format.
  Method names and signatures are unchanged; per-format pattern matches
  respell: `.f32 => |*w|` becomes `.dense => |*d| switch (d.*) { .f32 =>
  ... }`, `.q4_k => |*down|` becomes `.packed_quant => |*p| switch (p.*)
  { .q4_k => ... }`; the `.ptqtp` and `.tq2_0_fx4` spellings are
  unchanged. New: `LinearWeight.dtype()` returns the loaded block format
  at runtime; the containers are exported as `weights.DenseWeight`,
  `weights.PackedWeight`, `weights.ColdQuantWeight`.
- `weights.QuantByteStack.device_owned: bool` is now
  `rhs_lifetime: fucina.RhsLifetime` (`.stable_process` = provider-owned
  bytes), so stack consumers pass the lifetime through instead of
  respelling the bool.
- The CBLAS provider is one leaf, `src/backend/blas.zig`: the single
  `cblas_sgemm` extern, the vendor thread setters (one comptime switch),
  the once-only `-Dblas-threads` configuration, the MKL nested scope, the
  `fitsCblas` dimension gate and the Accelerate packed-kernel preference
  move there from `native.zig`; the five raw `cblas_sgemm` spellings in
  `native.zig` collapse onto `blas.gemm` (orientation request) and
  `blas.gemmStrided` (explicit leading dimensions). Same calls, same
  argument values, no numeric change. `backend.blas` re-exports the seam;
  its member names (`sgemmStrided`, `beginNestedScope`/`endNestedScope`,
  `matmulFoldedx4`) are unchanged.
- The backend kernel interface is derived from the declarations:
  `backend/interface.zig` and its three hand-maintained name lists
  (`names`/`generic_names`/`pool_free_names`) are gone. `native.zig`'s
  `kernels` declaration list is the inventory; a kernel that takes no
  `pc: ParallelConfig` carries a `pool_free_<name>` marker beside it, and
  `backend.zig`'s `conformKernels` checks the `pc`-first/pool-free
  contract from the signatures at comptime. Rewrite:
  `backend.interface.names` and the count assertions over the lists become
  reflection over `@typeInfo(backend.kernels).@"struct".decls`.
- One CPU kernel provider: `backend/cpu.zig` is gone, and
  `-Dbackend=scalar` no longer swaps provider modules. Each kernel entry
  selects its scalar reference arm internally on the reference build
  (`backend/isa.zig` `reference`): the independent scalar bodies live in
  `pub const scalar` namespaces inside `backend/vector/{elementwise,conv,
  pool,gemm,batched,matmul_quant}.zig`, the shared-core routes (im2col/
  col2im, conv2d backwards, pool2d backwards, the Winograd transforms) run
  serially there via `vector/common.zig` `refSerial`, and
  `backend.kernels` is always `native.zig`'s set. Rewrite for provider-name
  users: `backend.scalar_impl.kernels.X(pc, ...)` becomes the twin
  `backend.vector_impl.<domain>.scalar.X(...)` (no `pc`);
  `backend.native_impl` is unchanged.
- `AxisRange` is declared in the exec band (`exec/rope.zig`, beside its one
  consumer, the rope table builders); `fucina.AxisRange` is unchanged. The
  raw layer's `tensor.AxisRange` spelling is gone. Internal.
- Optimizer construction is fallible: `Optimizer(Kernel).init` (every
  `optim.SGD`/`Adam`/`AdamW`/`Muon`/`MuonH`/`Apollo` constructor) returns
  `!Self`, and an invalid config is `error.InvalidOptimizerConfig` instead
  of a panic (SGD's nesterov rule, still checked in every build mode).
  Rewrite: `var opt = optim.AdamW.init(alloc, cfg)` becomes
  `var opt = try optim.AdamW.init(alloc, cfg)`.
- The raw tensor's `rows()`/`cols()` rank-2 helpers are deprecated (no
  in-tree callers); read `shape.at(0)`/`shape.at(1)` instead. Internal
  (`fucina.internal.RawTensor`).
- `variance` takes an options struct: `variance(ctx, tag, ddof)` is now
  `variance(ctx, tag, options)` with `fucina.VarianceOptions`
  (`ddof: u1 = 1`, the torch.var default). Rewrite:
  `x.variance(ctx, .d, 0)` becomes `x.variance(ctx, .d, .{ .ddof = 0 })`;
  `x.variance(ctx, .d, 1)` becomes `x.variance(ctx, .d, .{})`.
- `groupNorm` takes an options struct for the affine pair:
  `groupNorm(ctx, channel_tag, groups, eps, weight, bias)` is now
  `groupNorm(ctx, channel_tag, groups, eps, options)` with
  `GroupNormOptions(channel_tag)` (both fields default null). Rewrite:
  `..., null, null)` becomes `..., .{})`; `..., &w, &b)` becomes
  `..., .{ .weight = &w, .bias = &b })`.
- `nonzero` takes the context like every other method:
  `nonzero(allocator)` is now `nonzero(ctx)`; the returned host slice is
  owned by the caller and freed with `ctx.allocator`. Rewrite:
  `x.nonzero(alloc)` becomes `x.nonzero(&ctx)`.
- `backward`, `backwardWithGrad`, and `zeroGrad` take a mutable receiver
  (`*Self`): they mutate shared gradient state, and the signature now says
  so. Rewrite: bind the loss (or the tensor whose gradient you reset) with
  `var` instead of `const`; call sites are otherwise unchanged.
- Argument-validity failures error with the new `error.InvalidArgument`
  instead of `error.InvalidShape`: dropout `p` outside `[0, 1)`, `softcap`
  cap not positive, `clamp` with `min > max`, cross-entropy
  `label_smoothing` outside `[0, 1)`, Huber `delta` not positive and
  finite, `standardize` negative `eps`, `topk` with `k == 0`, softmax
  `.max_bias` without `.head_tag`/`.mask`, `setRows`/`indexCopy` duplicate
  indices, `arange` with `step == 0`, `bernoulli` `p` outside `[0, 1]`,
  and `randint` with `low >= high`. Shape and layout failures keep
  `InvalidShape`/`ShapeMismatch`. Rewrite: catch or expect
  `error.InvalidArgument` at those call sites; inferred error sets absorb
  the change everywhere else.

- The public bf16/f8 tensor branches speak VALUE types instead of raw bit
  patterns: `item`/`data`/`dataConst`/`copyTo`/`fromSlice`/
  `fromBorrowedConstSlice`/`variableFromSlice` on a `.bf16` tensor now take
  and return the new `fucina.Bf16` (`packed struct(u16)` with `.bits`,
  `toF32`, `fromF32`; `.f8_e4m3`/`.f8_e5m2` likewise via `fucina.F8E4M3` /
  `fucina.F8E5M2`, and `dtype.Element(dtype)` is the mapping). The RAW
  tensor layer is unchanged (`Scalar(.bf16)` stays `u16` bits;
  `asRawTensor()` still exposes bits, as do state-dict bytes). The value
  structs are layout-identical to their bits, so the rewrite is one cast at
  the seam: `f32ToBf16(x)` in a facade element list becomes
  `Bf16.fromF32(x)` (or `.{ .bits = raw }`), a `[]const u16` read of
  `dataConst()` becomes `@ptrCast(try t.dataConst())` (or read `.bits` /
  `v.toF32()` per element).
- Exec scopes are one uniform arena of borrows. An adopted result is one
  `ExecContext.ScopeEntry{ ptr, release }` per reference (the value buffer
  of any dtype, and the graph node when there is one), taken by one
  fallible `ctx.adopt(entries)` before the value is handed to the returned
  handle; the two-phase `reserveScopeSlot` + `adoptScope*AssumeCapacity`
  protocol and the type-erased `TypedScopePayload` for 16-bit results are
  gone. Every op result of every dtype is adopted alike: the i64 index
  outputs (`argmax`, `topK.indices`, `argsort`), `.bool` masks, 16-bit
  casts, and the quantized branch's views and row gathers are now scope
  borrows too (`scope_owned` on every branch, `deinit` a uniform no-op),
  where they used to be caller-owned under a scope. Rewrite: nothing for
  the `defer x.deinit()` idiom (a no-op on a borrow); a typed result
  computed under a scope must not be used after the scope closes. Scopes
  manage lifetimes only and are never required for correctness (the graph
  is reference-counted).
- The backward record contract is derived (`core.recordVTable`): the
  operand slots come from a `parents` array or a `states` slice, `vjp`
  takes `*Self` and no `needs_grad` slice (`core.needs(self, i)` reads the
  slot; the engine sizes `out` to the operand count, so the 127
  `needs_grad.len > i` guards are gone), and `BackwardFunction.VTable.backward`
  takes a mutable record pointer (the two `@constCast`s in the linear
  losses go). `customVjp` and `checkpoint` records use the same vtable
  synthesis instead of hand-written shims; the public `customVjp` Spec
  still receives its `needs_grad` slice. Internal.
- Backward records are typed struct literals built by the op
  (`finishOp(tags, ctx, value, Record{ .parents = ..., ... })`;
  `core.createNode(allocator, record)` moves the record into the
  co-allocated node). The 84 `Record.init(self, allocator, ...)` constructors
  and the positional `create_args` tuple forwarded by `@call` are gone: the op
  takes the views its record saves with `cloneView` and holds them under
  `errdefer` until the node exists, node allocation is the last fallible step
  of an op tail, and the no-grad path (`plumbing.recordsGrad`) clones
  nothing. A swapped pair of parents no longer compiles. Internal
  (`src/ag/backward`, `src/ag/core.zig`); the public tensor surface is
  unchanged.
- `GradState` is reference-counted (`src/ag/core.zig`): a facade handle holds
  one reference, every backward record holds one per non-null operand (taken
  when the node is created, dropped when the record is destroyed), and an
  exec scope holds one per adopted result. Releasing an intermediate before
  `backward` is safe scoped or unscoped; previously an unscoped release
  before backward left the consumer records with dangling parent pointers
  (documented as undefined behavior). `GradState.deinit` is
  `GradState.release`, with `retain` alongside; `core.createNode` retains
  the record's operands and `core.releaseParents` is the matching head of
  every record vtable deinit. The cost is one atomic per operand per node
  in each direction.
- Backend kernel seam, dense GEMM: the nineteen name variants
  (`matmul2DIntoUnchecked`, `matmul2DAccIntoUnchecked`,
  `matmul2DIntoUncheckedTyped`, `matmulTransA2DIntoUnchecked`,
  `matmulTransB2DIntoUnchecked`, `matmulTransB2DIntoUncheckedF16Operands`,
  `matmulTransB2DIntoUncheckedBf16Rhs`, the three `matmulBatched*`, the
  validating `matmulInto`/`matmulTransAInto`/`matmulTransBInto`,
  `dotInto`/`dotIntoTyped`, `packDenseMatmulRhsTyped`/`packMatmulRhsTyped`,
  `matmul2DIntoUncheckedPackedDenseRhs`/`matmul2DIntoUncheckedPackedRhsTyped`)
  are five entries whose variation is a parameter: `gemm(pc, request, out,
  a, b, m, n, k)` over an `ops.Gemm` request (orientation, operand and
  output dtypes, store or accumulate; unsupported combinations are compile
  errors), `gemmBatched(pc, kind, ...)`, `dot(pc, dtype, ...)`,
  `packDenseRhs`, and `matmulPacked` now covering the f32 panel and the
  quantized packs by container type. The
  vector kernels (`vector.gemm.gemm`, `vector.batched.gemmBatched`) take
  slices and the same request. One orientation enum, `ops.MatmulKind`
  (`.plain`/`.trans_a`/`.trans_b`; `exec.MatmulKind` is that type), now
  serves the facade, exec, the blocked kernel (was `Orientation.nn/tn/nt`)
  and the GPU providers (was `Orient.nn/tn/nt`). In-tree consumers of the
  raw kernel set (`bench/`, `fucina.internal.backend_mod`) rewrite to the
  request form; the public `ExecContext` and `Tensor` surfaces are
  unchanged.
- Rope seam, `ExecContext`: the five table constructors
  (`prepareRopeTable`, `prepareRopeTableFactors`, `prepareRopeTableRange`,
  `prepareRopeTableFactorsRange`, `prepareRopeTableInvFreqsF64`) are one
  `prepareRopeTable(spec)` over an `exec.RopeTableSpec`: `positions`
  (`.explicit = []const i32` or `.range = AxisRange`), `feature_dim`,
  `freqs` (`.theta = .{ .base, .factors = null }` in f32, or
  `.inv_freq_f64 = []const f64` accumulated in f64) and `inverse`
  (default false). Rewrites: `prepareRopeTableRange(r, d, t, false)` is
  `prepareRopeTable(.{ .positions = .{ .range = r }, .feature_dim = d,
  .freqs = .{ .theta = .{ .base = t } } })`;
  `prepareRopeTableInvFreqsF64(pos0, n, f, inv)` is
  `prepareRopeTable(.{ .positions = .{ .range = .{ .origin = pos0, .len =
  n } }, .feature_dim = 2 * f.len, .freqs = .{ .inv_freq_f64 = f },
  .inverse = inv })`. Tables are bitwise what the old entries built.
  `ropeWithTable` now covers every rotary span (`ropePartialWithTable` and
  `ropePartial` are gone: the table's `feature_dim` was already the span);
  `rope` takes its on-the-fly source as one `exec.RopeTheta`. `RopeTable`
  drops the unused `theta_base` field. The `Tensor.rope` facade is
  unchanged.
- Norm seam, `ExecContext`: the affine and residual terms and the
  requested gradients are options. `rmsNorm(rank, x, axis, eps, .{ .weight,
  .residual })` replaces `rmsNorm`/`rmsNormMul`/`rmsNormMulAdd`;
  `rmsNormBackward(rank, x, gy, axis, eps, .{ .weight, .need_input,
  .need_weight })` returns an `exec.RmsNormBackwardResult` and replaces
  `rmsNormBackward`/`rmsNormMulBackwardInput`/`rmsNormMulBackwardWeight`;
  `layerNorm(rank, x, axis, eps, .{ .weight, .bias })` replaces
  `layerNorm`/`layerNormAffine` (weight and bias are now independently
  optional); `layerNormBackward(rank, x, gy, axis, eps, .{ .weight,
  .need_input, .need_weight, .need_bias })` returns an
  `exec.AffineBackwardResult` and replaces
  `layerNormBackward`/`layerNormAffineBackward`; `layerNormRows(input,
  rows, cols, eps, .{ .weight, .bias })` (slices) replaces
  `layerNormAffineRows`; `groupNorm`/`groupNormBackward` take the same
  `exec.AffineOptions`/`exec.AffineBackwardOptions`.
  `LayerNormAffineBackwardResult` and `GroupNormBackwardResult` are the one
  `exec.AffineBackwardResult`. Every combination runs the kernel it ran
  before, so results are bitwise unchanged. The `Tensor` facade
  (`rmsNorm`, `rmsNormMul`, `rmsNormMulAdd`, `layerNorm`, `groupNorm`) is
  unchanged.
- Ownership seam, `ExecContext`: the per-op spellings `addInPlace`,
  `subInPlace`, `mulInPlace`, `divInPlace`, `takeAdd`, `takeSub`, `takeMul`,
  `takeDiv`, `takeRelu`, `takeSilu` are the three entries they wrapped, with
  the op as the parameter: `elementwiseInPlace(.add, target, other)`,
  `takeElementwise(.add, target, other)`, `takeUnary(.relu, target)`
  (`takeScale` stays: scaling is not an `ElementwiseOp`). The two contracts
  stay two entries because they differ in signature, not in spelling: in
  place mutates a contiguous `*Tensor` and returns nothing, take consumes
  it and returns the result. `addAxisVectorInPlace(rank, op, target,
  row_vector, axis)` takes the optional activation that
  `addAxisVectorUnaryInPlace` spelled (`null` = plain bias add), and the
  backend pair `addRowVectorSlice`/`addRowVectorUnarySlice` is one kernel
  `addRowVectorSlice(op, ...)`. The `Tensor` facade (`takeAddNoGrad`,
  `takeScaleNoGrad`, `addAxisVectorInPlace`, `addAxisVectorUnaryInPlace`)
  is unchanged.
- Dtype policy at the exec seam, elementwise family. `dtype.FloatOp` gains
  the `widened` class (an f32 kernel only: f16/bf16 widen on entry, the
  result narrows once on store; f64 stays f64 or is a compile error where
  no kernel exists), and `ExecContext.prepareAs(dtype, compute, x)` /
  `storeAs(compute, out, value)` are its two helpers. The elementwise exec
  entries take the storage dtype and apply the policy themselves:
  `unary(dtype, op, x)`, `leakyRelu(dtype, x, slope)`, `clamp(dtype, x,
  lo, hi)`, `addScalar(dtype, x, v)`, `powScalar(dtype, x, e)`,
  `where(dtype, cond_dtype, x, cond, y)`, `maskedFill(dtype, mask_dtype,
  x, mask, v)`, `gated(dtype, rank, op, a, b)`, `splitGated(dtype, rank,
  op, x, axis)`, `elementwise(dtype, op, a, b)`; `max`/`min` on 16-bit
  floats now run through the same policy instead of a compile error. The
  op-as-name exec wrappers are gone: `relu`, `exp`, `sqrt`, `rsqrt`,
  `sigmoid`, `silu`, `log`, `neg`, `abs`, `sin`, `cos`, `tanh`, `gelu`,
  `quickGelu` are `unary(dtype, .op, x)`; `glu`/`swiglu`/`geglu` are
  `gated(dtype, rank, .op, a, b)`; `splitSwiGlu`/`splitGlu` are
  `splitGated(dtype, rank, .swiglu | .glu, x, axis)`.
  `tag_ops.gatedPointwise` takes the dtype first. On the facade the
  elementwise mixin is dtype-generic (`src/ag/tensor/elementwise.zig`,
  moved from `float/`): the f32 methods are unchanged; the f16/bf16 branch
  now takes the same methods from it (and gains `clampMin`, `clampMax`,
  `situ`, `splitGated`), the integer branch takes `add`/`sub`/`mul`/
  `maximum`/`minimum` from it. Results are bitwise what the widened facade
  computed.
- Dtype policy at the exec seam, the rest of the widened family. The same
  contract for `softmax(dtype, rank, x, axis)`, `logSoftmax`, `logsumexp`,
  `cumsum`/`cumsumReverse`/`cumprod`, `prod`, `varAxis`, `argmax`,
  `maxAxis`/`minAxis`, `pad` (data movement on the storage dtype) and the
  norms: `rmsNorm(dtype, rank, x, axis, eps, options)` with
  `exec.RmsNormOptions(dtype)` (weight and residual of the storage dtype),
  `layerNorm(dtype, rank, x, axis, eps, options)` with
  `exec.AffineOptions(dtype)`; `groupNorm` keeps `AffineOptions(.f32)`,
  the backward entries stay f32. On the facade the softmax, reduce, stats,
  shape and norm mixins serve the f16/bf16 branch for these ops
  (`typed/widened.zig` keeps `compare` and `einsum` only); the 16-bit
  branch gains `rmsNormMulAdd` and the affine `layerNorm`
  (`.{ .weight, .bias }` of its own dtype). f32 callers of the exec
  entries pass `.f32` first; results are bitwise unchanged.
- `compare(dtype, ...)`/`compareScalar(dtype, ...)` apply the same policy
  to 16-bit floats (were a compile error outside f32 and the integers);
  `compareScalar` takes `exec.CompareScalar(dtype)` (the exact element
  for integers and bool, f32 for the floats). The comparison family
  (`compare`, `logicalAnd`/`Or`/`Xor`/`Not`, `isnan`/`isinf`/`isfinite`)
  is served to the f16/bf16 branch by the shared elementwise mixin;
  `typed/widened.zig` keeps `einsum` only.
- Matmul seam, `ExecContext`: `matmul(dtype, kind, a, b)` replaces
  `matmul(dtype, a, b)`, `matmulTransA`, `matmulTransB` and
  `matmul2DDispatch`; `bmm(dtype, kind, a, b)` replaces `bmm`, `bmmTransA`,
  `bmmTransB` and `bmmDispatch`; `matmulAdd` was `matmul2DAdd`;
  `matmulHalfRhs(dtype, a, b)` was `matmulTransB2DWithHalfRhs`. Rewrites:
  `matmulTransB(&a, &b)` is `matmul(.f32, .trans_b, &a, &b)`, `bmm(&a, &b)`
  is `bmm(.f32, .plain, &a, &b)`. The typed plain `matmul` runs the typed
  GEMM as before; the other typed kinds and typed `bmm` follow the
  `.widened` policy. `tag_ops.taggedEinsum`/`taggedDot` take the dtype
  first and apply the same policy (16-bit operands widen once, the result
  narrows to the matmul output dtype). On the facade `matmul` and `einsum`
  are served to the f16/bf16 branch by the shared matmul mixin;
  `src/ag/tensor/typed/widened.zig` is gone.
- The accelerator is a provider behind one seam, `backend.offload`
  (`src/backend/offload.zig`), instead of `if (gpu_impl.enabled)` arms in
  exec, weights, es and the facade. The seam owns the capability queries
  (`enabled`, `has_quant_gemm`, `supportsQuant(QuantFormat)`,
  `supportsQuantDType`), resident storage, tracing, and the offload
  entries with their decisions built in: `quantGemmAccepts`/`gemmQuant`,
  `quantGemmSharedInputAccepts`/`gemmQuantSharedInput`, `attnPrefillF16`,
  `attentionFwd(KvElem, ...)`, `qmoeAccepts`/`QMoeSession`
  (`begin`/`gemmGrouped`/`end`), and the ES `flatPerturb`/
  `flatWeightedUpdate`/`flatAnchorDecay`. `ExecContext` renames its
  optional offload entries to say what they are: `tryMatmulQuantRhs` (was
  `denseQuantMatmulGpu`), `tryMatmulTernaryFolded` (was
  `foldedTernaryMatmulGpu`), `tryMatmulQuantRhsSharedInput` (was
  `denseQuantMatmulGpuSharedInputBatch`); `null` means the caller's CPU
  path runs. `fucina.internal.gpu` reads the same seam; the provider
  contract (`gpu_provider.zig`) and the provider files are unchanged.
- `CachingAllocator` uses four size classes per octave (geometric steps of
  2^(1/4) from 64 KB) instead of powers of two. On the Qwen3-0.6B Q8_0
  LoRA step at 1280 tokens the class rounding was 7.7 GB of a 28 GB
  class-accounted peak and is 2.1 GB now; maximum resident memory went
  from 24.7 GB to 23.6 GB with the step time unchanged. Same freelist
  semantics (a block serves its own class only, nothing is released
  before `deinit`).
- Autograd releases an interior gradient as soon as the node's own backward
  has consumed it (`GradState.pass_output` marks the outputs a pass was
  asked for; their gradients and the leaves' stay). The backward pass no
  longer holds a second copy of the forward until the exec scope closes:
  on the Qwen3-0.6B Q8_0 LoRA step at 1280 tokens the maximum resident
  memory goes from 23.6 GB to 15.6 GB with identical losses and the step
  time unchanged. Reading an interior gradient after `backward` now
  requires passing that tensor as an output of the pass.
- The logit softcap takes its cap as a parameter: `Tensor.softcap(ctx, cap)`
  and `ExecContext.softcap(dtype, x, cap)` over one kernel entry
  (`softcapContiguousIntoUnchecked`), with an output-form VJP
  (`1 - (y/cap)^2`). The model constants `UnaryOp.softcap_30` (Gemma) and
  `UnaryOp.softcap_15` (nanochat) and the facade aliases `softcap30`/
  `softcap15` are gone: `x.softcap30(ctx)` is `x.softcap(ctx, 30)`. The
  forward is bitwise what the constant kernels computed; Gemma's
  `final_logit_softcapping` now reaches the fused kernel at any cap.
- The MoE entries (`moeExpertFfn`, `moeExpertFfnBatch`,
  `weights.moeGatedFfnSeq`) take the activation with its parameter,
  `exec.Gated{ .op, .clamp }`, and `GatedOp.swiglu_clamp10` is gone:
  DeepSeek V4's clamped SwiGLU is `.{ .op = .swiglu, .clamp = 10 }`, the
  plain `.swiglu` argument is `.{ .op = .swiglu }`. Same numerics (the
  clamp is the same `min`/`clamp` pair the member computed).
- `Tensor(spec)`: one set of shared method mixins behind the four branches.
  Views and data movement (`materialize`, `contiguous`, `detach`,
  `withTags`, `viewWithStrides`, `alignTo`, `permuteTo`, `transpose`,
  `insertAxis`, `squeeze`, `split`, `merge`, `reshape`, `broadcastTo`,
  `flatten`, `flip`, `roll`, `rollBy`, `narrow`, `select`, `sliceStep`,
  `slice`, `gather`, `setSlice`, `setRows`, `concat`, `stack`,
  `unbindInto`, `repeatAxis`) are written once over the dtype
  (`src/ag/tensor/views.zig`) and are now the same set on the f32, typed
  float, and typed scalar branches; integer, bool, and f8 tensors gain the
  entries they lacked (`split`, `merge`, `flatten`, `reshape`,
  `sliceStep`, `flip`, `roll`, `stack`, `repeatAxis`, and the composed
  `contiguous`/`select`/`slice`/`unbindInto`/`rollBy`/`viewWithStrides`,
  which the typed float branch gains too). On the typed branches these
  ops stay no-grad constants: a grad-requiring operand is
  `error.UnsupportedGradient`. The lifetime/accessor methods share one
  implementation (`common.zig`), so `data()` on a 16-bit leaf that
  requires gradients now returns `error.MutableDataRequiresNoGrad`, the
  f32 rule. The dispatcher normalizes the spec before instantiating a
  branch, so every spelling of the same (dtype, tags) is the same type by
  construction. A comptime check in `src/ag/tensor.zig` asserts that every
  `pub` entry of every mixin is aliased by the branches that use it (the
  exception lists there are the complete statement of where the branches
  differ), and the facade mixins import their VJP domain files directly
  (`src/ag/backward.zig` no longer carries the 94-line alias table).
- `ExecContext` cross-entropy: the eight entries collapse into two.
  `crossEntropyLoss(ctx, rank, logits, axis, labels, options)` takes the
  options directly (`.{}` for the defaults); `CrossEntropyOptions` gains
  `row_stats: ?[]f32`, the per-position `{max, sum_exp}` slot the forward
  fills and the backward reads, so one options value drives both
  directions. `crossEntropyBackward(ctx, rank, logits, axis, labels,
  options, upstream)` takes a `CrossEntropyUpstream` union: `.{ .scale = s
  }` (mean/sum), `.{ .rows = .{ .per_row = g, .scale = s } }` (`.none`),
  or `.{ .tensor = &gy }` (the autograd form). Rewrites:
  `crossEntropyLoss(r, &x, a, l)` -> `crossEntropyLoss(r, &x, a, l, .{})`;
  `crossEntropyLossEx(..., o)` -> `crossEntropyLoss(..., o)`;
  `crossEntropyLossExStats(..., o, stats)` -> `crossEntropyLoss(..., o with
  .row_stats = stats)`; `crossEntropyBackward(..., s)` ->
  `crossEntropyBackward(..., .{}, .{ .scale = s })`;
  `crossEntropyBackwardEx(..., o, s, rows)` -> `crossEntropyBackward(...,
  o, .{ .rows = .{ .per_row = rows, .scale = s } })` (or `.{ .scale = s }`
  when `rows` was null); the `*ExStats` twins fold the stats into
  `o.row_stats`; `crossEntropyBackwardExUpstream[Stats](..., o, &gy[,
  stats])` -> `crossEntropyBackward(..., o, .{ .tensor = &gy })`. The
  fused `linearCrossEntropyBackwardUpstream` keeps its explicit
  `row_stats` parameter (a stats-only kernel).
- `ExecContext` packed matmul: `matmulPacked(a, rhs)` is the one entry
  over every pre-packed RHS; the container type selects the arm
  (`MatmulPackedOutput` names the LHS/output dtypes). Rewrites:
  `matmul2DWithPackedDenseRhs(&a, &rhs)` -> `matmulPacked(&a, &rhs)`;
  `matmul2DWithPackedDenseRhsInto(&out, &a, &rhs)` ->
  `matmulPackedInto(&out, &a, &rhs)`; `matmul2DWithPackedRhs(.f16, &a,
  &rhs)` -> `matmulPacked(&a, &rhs)`. The two mixed-precision twins merge:
  `matmulTransB2DWithF16Rhs(&a, &b)` / `matmulTransB2DWithBf16Rhs(&a,
  &b)` -> `matmulTransB2DWithHalfRhs(.f16 | .bf16, &a, &b)`.

The LLM band is renamed to the model band. The module `fucina_llm` is now
`fucina_models` (root `src/models.zig`); the one-line consumer rewrite is
`@import("fucina_llm")` -> `@import("fucina_models")`. Dependency-context
wiring passes the options module as `models_build_options` (was
`llm_build_options`), and the `test-llm` build step is renamed
`test-models`. The band taxonomy is stated in the module root: families in
`models/<family>/`, the modality-agnostic text runtime in `models/text/`,
training helpers in `models/train/`, research in `models/research/`.
Family namespaces keep their names; `decoder` and `registry` stay
top-level. Spelling table for the moved flat files:

| old | new |
| --- | --- |
| `llm.tokenizer` | `models.text.tokenizer` |
| `llm.spm_tokenizer` | `models.text.spm_tokenizer` |
| `llm.unicode_categories` | `models.text.unicode_categories` |
| `llm.sampler` | `models.text.sampler` |
| `llm.logit_processor` | `models.text.logit_processor` |
| `llm.llguidance` | `models.text.llguidance` |
| `llm.chat` | `models.text.chat` |
| `llm.generate` | `models.text.generate` |
| `llm.kv_cache` | `models.text.kv_cache` |
| `llm.kv_persist` | `models.text.kv_persist` |
| `llm.data` | `models.text.data` |
| `llm.cartridge` | `models.text.cartridge` |
| `llm.cartridge_fleet` | `models.text.cartridge_fleet` |
| `llm.speculative` | `models.text.speculative` |
| `llm.serving` | `models.text.serving` |
| `llm.lora_trainer` | `models.train.lora_trainer` |
| `llm.trainer_state` | `models.train.trainer_state` |
| `llm.runner` | `models.qwen3.runner` |
| `llm.subq` | `models.research.subq` (research namespace unchanged) |
| `llm.decoder` | `models.decoder` |
| `llm.registry` | `models.registry` |
| `llm.moe_router` | `models.moe_router` |
| `llm.moe_stream_cli` | `models.moe_stream_cli` |

The module root additionally exports the band-level helpers
`models.model_common`, `models.host_ops`, and `models.test_support`.

`ExecContext` carries exactly one spelling per op: the `*Rank`, `*AxisRank`,
and `*Typed` variant suffixes are gone from the exec surface (`src/exec.zig`
and the `src/exec/` bodies), and `src/tag_ops.zig` drops its `*Of` split.
Kernels, routing, and numerics are unchanged; comptime rank/axis
monomorphization is preserved everywhere. Rewrite table, grouped by rule:

- **Shape argument carries the rank (creation ops).** `shape` is `anytype`:
  a `[rank]usize` array or a tuple of sizes takes the comptime-rank arm, a
  `[]const usize` slice the runtime-rank arm. Each op also takes a leading
  comptime `DType` (f32 call sites pass `.f32`).
  `empty`/`emptyRank`/`emptyTyped`/`emptyRankTyped` are `empty(dt, shape)`;
  the same collapse applies to `zeros`, `ones`, `full` (`fullTyped`),
  `scalar` (`scalarTyped`), `fromSlice` (`fromSliceRank`, `fromSliceTyped`,
  `fromSliceRankTyped`), `fromBorrowedSlice` (`fromBorrowedSliceRank`,
  `fromBorrowedSliceRankTyped`), `fromStorageSlice`
  (`fromStorageSliceRankTyped`), and `fromBorrowedStorageSlice`
  (`fromBorrowedStorageSliceRankTyped`). `broadcastTo`/`broadcastToRank`
  are `broadcastTo(x, shape)`; `reduceBroadcast`/`reduceBroadcastRank` are
  `reduceBroadcast(x, target_shape)`.
- **Dtype is an explicit comptime parameter, once.** Where an f32 spelling
  and a `*Typed` spelling coexisted, the generic survives under the base
  name and every f32 call site gains an explicit `.f32`:
  `materialize(dt, x)` (`materializeTyped`), `clone(dt, x)` (`cloneTyped`),
  `prepareContiguous(dt, x)` (`prepareContiguousTyped`),
  `cast(src_dt, dst_dt, x)` (`castTyped`), `scale(dt, x, s)` (`scaleTyped`),
  `sum(dt, x)` (`sumTyped`), `dot(dt, a, b)` (`dotTyped`),
  `matmul(dt, a, b)` (`matmul2D`, `matmul2DTyped`, `matmulTyped`; the
  duplicate `matmul2D` alias is deleted),
  `where(cond_dt, x, cond, y)` (`whereTyped`),
  `maskedFill(mask_dt, x, mask, v)` (`maskedFillTyped`),
  `compare(dt, op, a, b)` (`compareIntTyped`),
  `compareScalar(dt, op, x, v)` (`compareIntScalarTyped`),
  `logical` (`logicalTyped`), `logicalNot` (`logicalNotTyped`),
  `add`/`sub`/`mul`/`div`/`max`/`min` `(dt, rank, a, b)`
  (`addRank`/`addRankTyped` and kin),
  `divTrunc`/`divFloor`/`rem`/`mod`/`bitwise` `(dt, rank, a, b)`
  (`divTruncRankTyped`, `divFloorRankTyped`, `remRankTyped`,
  `modRankTyped`, `bitwiseRankTyped`),
  `sumAxis`/`meanAxis` `(dt, rank, x, axis)` (`sumAxisRank`,
  `sumAxisRankTyped`, `meanAxisRank`, `meanAxisRankTyped`),
  `narrowAxis`, `concatAxis`, `gatherAxis`, `setSliceAxis`, `setRows`
  (their `AxisRank`/`AxisRankTyped` pairs),
  `dequantizeTensor` (`dequantizeTensorTyped`), `getRowsQuantized`
  (`getRowsQuantizedTyped`), `concatQuantizedRows`
  (`concatQuantizedRowsTyped`), `packDenseMatmulRhs`
  (`packDenseMatmulRhsTyped`), `matmul2DWithPackedRhs`
  (`matmul2DWithPackedRhsTyped`), and
  `enableNativeMatmulPoolForWork(dt, m, n, k)`
  (`enableNativeTypedMatmulPoolForWork`; `dt` names the storage the kernel
  walks, and the f32 dense arm keeps its BLAS bail).
  `packMatmulRhsTyped` folds into `packMatmulRhs(dt, rhs)`, the quantized
  block pack for `dt` (`backend.PackedRhsFor(dt)`); dense f32/f16/bf16
  weights pack through `packDenseMatmulRhs` into the one f32 panel, and
  the separate same-dtype f16/bf16 panel (`packHalfRhs`,
  `matmulHalfPanel`) is gone.
- **`AxisRank` dropped; the comptime rank/axis parameters stay.**
  `softmax`, `logsumexp`, `logSoftmax`, `softmaxExt`, `rmsNorm`,
  `rmsNormMul`, `rmsNormMulAdd`, `rmsNormMulBackwardInput`,
  `rmsNormMulBackwardWeight`, `rmsNormMulRopeWithTable`, `rmsNormBackward`,
  `layerNorm`, `layerNormAffine`, `layerNormBackward`,
  `layerNormAffineBackward`, `groupNorm`, `groupNormBackward`,
  `crossEntropyLoss` (+`Ex`, `ExStats`), `crossEntropyBackward` (+`Ex`,
  `ExStats`, `ExUpstream`, `ExUpstreamStats`), `rope`, `ropeWithTable`,
  `ropePartial`, `ropePartialWithTable`, `cumsum`, `cumsumReverse`,
  `cumprod`, `prod`, `segmentSum`, `segmentBroadcast`, `linearRecurrence`,
  `linearRecurrenceBackward`, `sumMasked`, `meanMasked`, `pad`, `setRows`,
  `zeroSlice`, `zeroRows`, `sliceGradient`, `scatterAdd`, `takeAlong`,
  `scatterAddAlong`, `scatterAlong`, `argmax`, `maxAxis`, `minAxis`,
  `maxMasked`, `minMasked`, `varAxis`, `standardize`,
  `standardizeValidPrefix`, `standardizeBackward`, `topK`, `sort`,
  `conv1d`, `conv1dBackwardInput`, `conv1dBackwardWeight`, `causalConv1d`
  (+`BackwardInput`, `BackwardWeight`), `causalDepthwiseConv1d`
  (+`BackwardInput`, `BackwardKernel`), `groupedCausalConv1d`
  (+`BackwardInput`, `BackwardWeight`), `col2im1d`, `col2im1dBackward`,
  `splitSwiGlu`, `splitGlu`, `splitSwiGluBackward`, `splitGluBackward`,
  `addAxisVectorInPlace`, `addAxisVectorUnaryInPlace`, `gated`, `glu`,
  `swiglu`, `geglu` (from `gatedRank`/`gluRank`/`swigluRank`/`gegluRank`),
  and `relposShift` (`relposShiftRank3`; the rank-3 contract is documented,
  not spelled). Base-name decisions where a whole-tensor sibling exists:
  `sum` stays the all-elements reduction and the axis form is `sumAxis`
  (likewise `meanAxis`, `narrowAxis`, `concatAxis`, `gatherAxis`,
  `setSliceAxis`); `maxAxis`/`minAxis`/`varAxis` keep `Axis` because
  `max`/`min` are the elementwise binaries and `var` is a keyword.
- **Runtime-rank elementwise entry.** `ctx.add(&a, &b)` and kin (runtime
  rank dispatch) are `ctx.elementwise(.add, &a, &b)` with
  `op: ElementwiseOp`; the base names `add`/`sub`/`mul`/`div`/`max`/`min`
  are the comptime-rank generics above. Same kernels either way.
- **Absorbed variants.** `softmaxBackward(rank, y, gy, axis, scale)`
  absorbs `softmaxExtBackwardAxisRank` (plain callers pass `1`);
  `matmul2DWithQuantizedTensorRhs` and `matmul2DWithQuantizedBlocksRhs`
  absorb their zero-caller `*Options` twins (the options struct is now the
  trailing parameter); `conv2dExt`/`conv2dPreparedExt` are folded into the
  private shared implementation behind the four public conv2d entries.
  `softmaxExt` stays a separate op: its masked/sink/ALiBi row kernel is not
  the plain softmax kernel.
- **`tag_ops`: the dtype-explicit form holds the base name.** The `*Of`
  spellings are renamed and the f32 convenience twins are deleted; callers
  of the old convenience forms pass `.f32` first. The 11 renames:
  `splitAxisViewOf`, `mergeAxesViewOf`, `flattenTensorOf`,
  `alignTensorToOf`, `broadcastTensorToOf`, `pointwiseShapeOf`,
  `validateTensorRankOf`, `contiguousForReshapeOf`, `dotResultShapeOf`,
  `einsumResultShapeOf`, `productRangeOf`, each to the same name without
  `Of`.

- `Model.initKvCache` is `Model.initCache` on every family (runner/qwen3,
  gemma4, diffusion_gemma, SHINE's `AdaptedModel`), and every family's
  cache constructor takes `(self, ctx, capacity)` (deepseek2, glm4moe,
  and inkling gain the `ctx` parameter and ignore it; deepseek4's
  `initCache` now builds the decoder-contract `Session`, with the raw
  layer-state constructor renamed `initRawCache`).
- Cache length is the method `len()` on every cache type; the state field
  is `count` (`models.text.kv_cache.KvCache`, `models.qwen3.runner.HostCache`,
  `deepseek2.Cache`, `deepseek4.Cache`, `inkling.Cache`; qwen35's `Cache`
  already had the method). Rewrite `cache.len` reads to `cache.len()`
  and direct field writes to `cache.count`. `HostCache`, `deepseek4.Cache`
  and `inkling.Cache` gain `reset()`; `deepseek4.Session` gains `len()`,
  `reset()`, and a parameterless `deinit()` (was `deinit(model)`).
- The family decode entries speak the contract: deepseek2 gains
  `forwardStep` (chunked `stepBatch` prefill keeping the last logits;
  `step`/`stepBatch` remain the hot paths), deepseek4 gains a
  `forwardStep` method over `Session` returning a caller-owned tensor
  (the session-owned `step`/`stepBatch*` slices remain for the MTP
  runner), glm4moe gains `forwardStepAllLogits` (its `step` IS the
  all-row forward) and `forwardStep` (last row), and inkling gains
  `forwardStep` (its `step` already returns the last row).
- `models.text.generate` is the one reference generation loop over the contract
  (`generate`/`generateOutcome` with a `Sampler`, stop ids, and a
  `TokenSink`; `greedy` keeps the slice-filling argmax convenience).
  `models.qwen3.generate` is deleted — rewrite `models.qwen3.generate.greedy` to
  `models.text.generate.greedy` (same arguments). The qwen35 and inkling chat
  engines decode through the shared loop, and their `StreamDecoder` now
  comes from the engine's `TokMod` parameter.
- `serving/open.zig` dispatches through `models.registry` (one `inline for`
  in place of the arch string ladder): `registry.Entry` gains `Serving`,
  the row's serving wiring (the family's `serving.zig`: `openFromFile`
  plus a `conversation_hosted` flag; `models.qwen3.serving` and
  `models.gemma.serving` are new, wired over the generic engine box).
  Rows without a wiring return `error.UnsupportedArchitecture`, stated
  per row: deepseek2 (the MLA cache has no rewind, so no `Conversation`,
  and no recognized chat template) and glm4moe (mutable-model
  `forwardStep` against the engine's const model pointer, and no
  recognized template).
- The family serving adapters (qwen35, inkling, deepseek4, the SHINE
  fleet) share one skeleton: `serving/adapter_common.zig` builds the
  heap-pinned engine box (tokenizer, template policy, sampling policy,
  `Family.load`, file handoff) from a per-family comptime `Wiring`, and
  `serving/fleet_serve.zig` holds the `--fleet` state and the cartridge
  loader; the `Conversation`-hosted `openChat` path lives beside them.
  qwen35's template fallback is now the checked detect-or-fallback policy
  in place of an unchecked `.?` (structural; cannot fire on a qwen35
  GGUF, whose family always carries the ChatML fallback).
- The OpenAI and Anthropic wire parsers share `wire_json.Head(ErrorInfo)`:
  parser state, the typed `opt*` field accessors, `ToolSet`, and the
  forced-tool_choice plumbing live once in `serving/wire_json.zig`, and
  each dialect `Parser` forwards to its head one line per method
  (`forceNamed` bakes in the dialect's noun). Wire behavior is unchanged;
  the intentionally divergent pieces stay in the dialect files.

- One tuning table replaces the per-gate switches: `fucina.tuning.Table`
  holds every FUCINA_* route gate and numeric crossover as a typed field,
  and each leaf derives its env variable from its field path (`FUCINA_`
  plus the upper-cased segments; a leaf named `base` or `enabled` names
  its group, so `gpu.min_work.base` is `FUCINA_GPU_MIN_WORK` and
  `gpu.enabled` is `FUCINA_GPU`). API rewrites: `tuning.Switch(cfg)` and
  `tuning.Threshold(name, default)` and `tuning.defaults` are gone;
  read `tuning.get().<field>`, pin via `tuning.setField("<path>", value)`
  (null re-arms the env/default value) or `tuning.set(patch)`, and take
  one uncached read with `tuning.load()`. `tuning.Overrides` is now the
  optional shadow of the whole table (every leaf `?T = null`, not just
  the two shadow fields), so `ExecContext.setTuning` can override any
  field per context; routes that support per-context policy consult
  `tuning.resolve(&ctx.tuning, "<path>")`. Env spellings: the boolean
  off switch is `FUCINA_X=0` in place of the removed `FUCINA_NO_X=1`
  form. Removed → replacement: `FUCINA_NO_WINOGRAD=1` →
  `FUCINA_WINOGRAD=0`; `FUCINA_NO_WINOGRAD_F4=1` → `FUCINA_WINOGRAD_F4=0`;
  `FUCINA_NO_NORM_QUANT_FUSED=1` → `FUCINA_NORM_QUANT_FUSED=0`;
  `FUCINA_NO_DECODE_COMPACT=1` → `FUCINA_DECODE_COMPACT=0`;
  `FUCINA_NO_ATTN_BWD_STATS=1` → `FUCINA_ATTN_BWD_STATS=0`;
  `FUCINA_NO_ATTN_BWD_BLAS=1` → `FUCINA_ATTN_BWD_BLAS=0`;
  `FUCINA_NO_CONV_BWD_GEMM=1` → `FUCINA_CONV_BWD_GEMM=0`;
  `FUCINA_NO_FUSED_DISTILL=1` → `FUCINA_FUSED_DISTILL=0`;
  `FUCINA_PTQTP_NO_FOLD=1` → `FUCINA_PTQTP_FOLD=0`. Every other spelling
  is unchanged (they already match the derivation rule), including the
  whole `FUCINA_GPU_*` family, whose values now flow to the providers
  through the table; `FUCINA_SPIN_BUDGET` keeps its semantics (0 = park
  immediately) as the table's `spin_budget` leaf; `FUCINA_MAX_THREADS`
  (bootstrap) and `FUCINA_GPU_KERNELS` (string-valued) stay direct env
  reads. With no environment set, every default is byte-identical to the
  previous per-gate declarations.
- One exec attention entry with a typed KV view and typed options: the 13
  `ExecContext.grouped*Attention*` entries are now
  `ctx.groupedAttention(q, kv: KvView, kv_head_for_head, scale, opts: AttentionOptions)`
  plus `ctx.groupedAttentionBackward` (the former
  `groupedCausalAttentionBackward`, arguments unchanged). `exec.KvView`
  names the cache representation (`.f32`, `.f16`, `.q8`, `.multi_f16`,
  `.multi_q8`), `exec.AttentionOptions` the variant (`.mask`, `.window`,
  `.bias`, `.stats_out`); option combinations with no kernel return
  `error.UnsupportedAttentionVariant`. Rewrites:
  `ctx.groupedCausalAttention(q, k, v, map, s)` →
  `ctx.groupedAttention(q, .{ .f32 = .{ .k = k, .v = v } }, map, s, .{})`;
  `ctx.groupedCausalAttentionWindowed(..., w)` → `.{ .window = w }`;
  `ctx.groupedBidirectionalAttention(...)` → `.{ .mask = .bidirectional }`;
  `ctx.groupedBidirectionalAttentionBiased(..., b)` →
  `.{ .mask = .bidirectional, .bias = b }`;
  `ctx.groupedCausalAttentionStatsOut(..., w, causal, st)` →
  `.{ .mask = if (causal) .causal else .bidirectional, .window = w, .stats_out = st }`;
  `ctx.groupedCausalAttentionF16Kv`/`...F16KvWindowed`/
  `ctx.groupedBidirectionalAttentionF16Kv` → the `.f16` view with the same
  options; `ctx.groupedCausalAttentionQ8Kv[Windowed](q, kb, vb, n, h, map, s[, w])`
  → `ctx.groupedAttention(q, .{ .q8 = .{ .k = kb, .v = vb, .kv_seq = n, .kv_heads = h } }, map, s, .{ .window = w })`;
  `ctx.groupedCausalAttentionMulti{F16,Q8}Kv(q, ks, vs, lens, h, map, s)`
  → `ctx.groupedAttention(q, .{ .multi_f16 = .{ .k = ks, .v = vs, .lens = lens, .kv_heads = h } }, map, s, .{})`
  (`.multi_q8` for the q8_0 caches). Every view selects the identical
  kernel path with identical arguments; numerics and the GPU attention
  gates are unchanged. The `Tensor.groupedAttention` facade (signature,
  options, comptime diagnostics) is unchanged.
- Per-format plumbing is comptime-parameterized: each packed quantized
  matmul entry is spelled once, with the format a comptime parameter or
  inferred from the RHS container type. Rewrites:
  `ctx.packMatmulRhsQ4_Kx8(&t)` → `ctx.packMatmulRhs(.q4_k, &t)` (the
  dtype-default container; `ctx.packMatmulRhsAs(Rhs, &t)` forces a
  specific one); `ctx.matmul2DWithPackedQ8_0x4Rhs(&a, &rhs)` →
  `ctx.matmulPacked(&a, &rhs)`; `ctx.rmsNormMulMatmul2DWithPacked*Rhs` →
  `ctx.rmsNormMulMatmulPacked`; `ctx.splitSwiGluMatmul2DWithPacked*Rhs` →
  `ctx.splitSwiGluMatmulPacked`; `ctx.gegluQuantMatmul2DWithPackedQ8_0x4Rhs`
  → `ctx.gegluQuantMatmulPacked`; `weights.linearSeqQ6_K(&w, ...)` →
  `weights.linearSeq(&w, ...)` (one generic over the packed weight
  containers). One decode-route gate replaces the three per-format
  gates: `FUCINA_Q4K_DECODE_COMPACT`/`FUCINA_Q5K_DECODE_COMPACT`/
  `FUCINA_Q6K_DECODE_COMPACT` (and their `FUCINA_NO_*` twins) →
  `FUCINA_DECODE_COMPACT=1/0`, and
  `weights.setQ5kDecodeCompact`/`setQ6kDecodeCompact` →
  `weights.setDecodeCompact`. Kernel-set entries follow the same rule:
  `kernels.matmul2DQuantizedRhsQ8_0x4`/`...Q4_Kx4`/`...Q4_Kx8`/
  `...Q4_Kx2Mmla`/`...Q5_Kx8`/`...Q6_Kx4` → `kernels.matmulPacked`,
  the per-dtype plain K-quant forwards → `kernels.matmulQuantizedRhs`,
  and the `matmulPacked*Slice` family → `kernels.matmulPackedSlice`.
  Every collapsed entry calls the identical kernel with identical
  arguments; numerics are unchanged.
- `fucina.Backend` is gone: the kernel dispatch struct carried no state
  beyond the worker-pool pointer, which now lives on `ExecContext`
  (`parallel_pool`, read through `ctx.pc()`). `fucina.BackendKind` and
  `fucina.active_backend_kind` are unchanged; rewrite `fucina.Backend.kind`
  -> `fucina.active_backend_kind`, and a direct kernel call
  `backend.X(args)` -> `fucina.internal.backend_mod.kernels.X(.{}, args)`.
  Internally the kernel set is `backend.kernels`, named once in
  `backend/interface.zig` (`names`, `generic_names`, `pool_free_names`)
  and checked on both providers at comptime by `interface.conform`; every
  kernel signature takes `pc: ParallelConfig` first when it uses the pool
  and drops the `WithConfig` suffix (`scaleIntoWithConfig(out, a, s,
  config)` -> `scaleInto(pc, out, a, s)`); the config-less twins are gone.
- The research modules live under one namespace, so the facade states the
  tier: `models.research.subq` → `models.research.subq`, `models.engram` →
  `models.research.engram`, `models.qwen3.shine`/`models.qwen3.shine_train` →
  `models.research.shine`/`models.research.shine_train`, `models.kimi3` →
  `models.research.kimi3`.
- The runner's SubQ entry is a generic seam: `Model.forwardStepSubq(ctx,
  kv, ids, pos, &sq)` is removed in favor of the type-erased
  `Model.attention_override` hook: install
  `model.attention_override = models.research.subq.attentionOverride(&sq)`
  and call the stock `forwardStep`. The override receives the same
  arguments the internal SubQ glue took and returns null to fall through,
  so numerics are unchanged.
- The qwen3 trainer's Engram graft is a generic seam:
  `ForwardOptions.engram = .{ .model, .rows }` (and `EngramOptions`) →
  `ForwardOptions.residual_hook = adapter.hook()` with
  `const adapter = models.research.engram.ResidualGraft{ .model, .rows }`.
  The hook validates before any compute and adds the same tensor the
  inline graft added; `error.InvalidEngram` now surfaces from the adapter
  (it left `qwen3.train.Error`).
- SHINE fleet serving is its own entry: `serving.OpenOptions.shine_fleet_dir`
  is removed, and `models.qwen3.shine_serving.open(ctx, io, allocator,
  gguf_path, fleet_dir, options, stderr)` / `openFromFile(...)` mirror
  `serving.open`/`openFromFile` with the fleet directory as an explicit
  argument (same `OpenOptions`, same `Opened` result).
- The streamed-experts CLI and routing policy live with the runners:
  `fucina.weights.MoeStreamCli` / `parseMirrorWeights` /
  `reportAndSaveMoeStream` → `models.moe_stream_cli.*`, and
  `fucina.weights.cacheRouteSel` / `pilotHintTopK` → `models.moe_router.*`.
  `fucina.weights.MoeStreamOptions` (the options struct the loaders
  consume) is unchanged.
- The trainer resume state is LLM-band:
  `fucina.training_checkpoint.TrainerState` →
  `fucina_models.train.trainer_state.TrainerState`, and the checkpoint frame is
  generic over the state struct: `saveTrainerState(allocator, io, dir,
  state)` takes any struct with the version/step/seed header and optional
  `?u64`/`?f64` fields, `loadTrainerState(State, allocator, io, dir)`
  takes the state type first, and the JSON codec pair
  (`writeTrainerStateJson`, `parseTrainerState`) is exported. The on-disk
  format is byte-identical.

- Internal reorganization (no public spelling changes; details in `git
  log`): the `Runtime` substrate merged into `ExecContext` (domain modules
  take `*ExecContext`; `zig build arch-check` permits and counts the one
  root-anchored `exec.zig` <-> `exec/*.zig` SCC); the `backend/quant/` and
  `backend/vector/` children addressed by module, with the kernel set named
  once in `backend/interface.zig`; the GPU providers' kernel ABI tag
  (`KernelFormatTag`, was `QFormat`) and the `expert_store` streamable-quant
  enum derived from `dtype.block_formats`; the new `src/store/` band
  (`src/exec/expert_store.zig` moved there) and the `-Dgpu=none` null
  provider `backend/gpu_none.zig`; the shared model load band
  (`src/models/model_common.zig`, `weights/moe_stream.zig`, the
  `exec/moe_gu.zig` chain skeleton); the `optim/` facade split; named
  `parallel.*` pool-gate thresholds; and the docs' per-file band table.
  Kernels, numerics, recorded goldens, checkpoint bytes, and every public
  spelling are unchanged.

### Deprecated

- `fucina.simd.{vecScale,vecMaxReduce,dotF32F16,scoreRows4F16,vecExpAffineSumInPlace,weightedAccumRows4F16}`
  — backend internals that were published under the elemental-op
  namespace; in-tree consumers use `fucina.internal.backend_mod.vector_impl`.
  Aliases kept; removal in the next MINOR release.
- `fucina.weights.GroupedQ8_0RhsX4` — alias of
  `fucina.quant.QuantizedMatmulRhsQ8_0x4` (the same type); removal in the
  next MINOR release.

### Fixed

- `ExpertStore.release` after an abandoned wave-split acquire (an error
  between `acquireStart` and `acquireFinish`) no longer promotes the
  unread working slots into the LRU: a promoted `invalid_eid` slot both
  evicted a real cached expert for garbage and made the next release's
  heat victim scan index `heat[invalid_eid]` out of bounds.
- qwen35: a layer leak on the load error path (a failing
  `ExpertStore.finalize` after the layers had loaded freed the layer
  slice but not the layers).

## 0.3.0 — 2026-08-23

### Added

- `llm.serving` transport and engine, promoted from `apps/lmserve`:
  `serving.http` (server, SSE stream pipe, Host guard), `serving.scheduler`
  (bounded FIFO + single inference worker), `serving.emitter`, the
  `serving.openai`/`serving.anthropic` wire dialects, `serving.toolcall`
  (hermes tool calling), and `serving.gguf_chat` (the generic
  `GgufChatBackend` engine: constraint cache, KV reuse slots + disk tier,
  KV RAM guard). A package consumer now gets a working server, not only
  the `Backend` vtable.
- `llm.serving.open` / `llm.serving.openFromFile`: load a GGUF and return
  a ready `serving.Backend` for the `Conversation`-hosted families (qwen3,
  qwen3moe, gemma4) with the full engine option surface
  (`serving.OpenOptions`); architectures whose adapters stay with
  `apps/lmserve` (nanochat, diffusion-gemma, inkling, qwen35,
  qwen35moe, deepseek4) return `error.UnsupportedArchitecture`. lmserve is now a thin
  CLI front end over the band, and the voice agent hosts the engine
  through `llm.serving` (the `lmserve` build module is removed; in-process
  `--chat` covers the `serving.open` families, `--chat-url` covers the
  rest).
- `fucina.quant`: the quantized-format namespace — every GGML `Block*`
  struct, the eleven root-exported `QuantizedMatmulRhs*` container types,
  `q8_0_block_size`, and `supports_q4_k_mmla` move under one name.
- The module root now states its membership rule in its header (pillar
  types, receiver-less graph control, subsystem namespaces, backend
  constants, scalar format converters), and every root export carries a
  doc comment.
- `llm.generate`: the greedy generation driver, shared by the tensor-band
  families (duck-typed on `forwardStep` + `KvCache`). `llm.qwen3.generate`
  re-exports it; `gemma.model.Model.generate` forwards to it.
- `KvCache.requireF16` (+ `kv_cache.Error.UnsupportedKvCacheDtype`): the
  f16-views-only guard the gemma4/qwen35/diffusion_gemma forwards share.
  Replaces the per-family `requireF16KvCache` copies (experimental tier,
  no alias).

### Changed

- The gemma-family MoE kernels move to the exec band where they belong
  (`exec/moe_gu.zig`, `ExecContext.moeGu*` facade): they were the one
  below-facade kernel body living in `llm/`. `llm.gemma.moe` remains the
  family surface (tagged wrappers + re-exported raw entries);
  `llm.gemma.moe_route` / `moe_route_tensor` are gone (their code folded
  into the kernels; experimental tier).
- Host-band `step` entries return the tensor-band logits shape
  `fucina.Tensor(.{ .seq, .vocab })` (caller deinits) instead of host
  slices: `glm4moe.model.step` (was `[][]f32` with per-row dupes),
  `deepseek2.model.step`/`stepBatch` and `inkling.model.step`/`stepMixed`
  (were caller-freed `[]f32` dupes of an internal tensor). Semantics and
  numerics unchanged; the dupe copies are gone. `deepseek4` keeps its
  session-owned protocol (its MTP out-rows contract differs by design).
  Rewrites: `const row = try model.step(...); defer allocator.free(row)` →
  `var logits = try model.step(...); defer logits.deinit();` and read via
  `try logits.dataConst()`; glm4moe's per-row `rows[i]` becomes
  `flat[i * vocab ..][0 .. vocab]` over the one returned tensor.
- Family namespace shape unified to `family.model`: `llm.gemma.model` /
  `llm.gemma.train` (files `llm/gemma/{model,train}.zig`) and
  `llm.pockettts.model`. `gemma.gemma4`, `gemma.gemma4_train`, and
  `pockettts.pocket` remain as deprecated aliases for one MINOR release.
- The glm4moe family's trunk now runs on the runner's host_reference
  band: `llm.glm4moe.model` keeps the MTP (`nextn`) head, draft step, and
  the `step` verify API, and drives `runner.hostLayerForward` (which gains
  a nullable `mtp_cache` seam for the MTP stream's single-layer cache).
  `Config` is `runner.Descriptor`; `Cache` is `runner.HostCache`. The
  ~530-line verbatim block copy is gone; bit-exactness is pinned by the
  glm fixture's recorded logits hash and greedy chain in
  `runner_tests.zig`.
- The qwen3 family now runs directly on the descriptor runner:
  `llm.qwen3.model` is an alias surface over `llm.runner` (`Config` is
  `runner.Descriptor`, `Model` is `runner.Model`), removing the ~1,300-line
  verbatim copy the extraction had left behind. The runner gains the
  batched decode entries (`forwardStepBatch`, `forwardStepBatchSpans`) and
  the SubQ research seam; every public name and method of
  `llm.qwen3.model` is preserved. Bit-exactness of the consolidation is
  pinned by recorded-logits gates in `runner_tests.zig` (real Qwen3-0.6B
  Q8_0/Q4_K_M plus the synthetic fixtures, captured before the merge).
- Qwen3-TTS parity fixtures moved out of the shipped package:
  `src/llm/qwen3tts/goldens/` → `testdata/qwen3tts/` (5.5 MB of binary
  dumps that only in-repo parity tests read; `build.zig.zon` ships all of
  `src/`, so the move halves the fetched package). The tests already skip
  when fixtures are absent; dependency builds are unaffected.

### Deprecated

- The flat root quant spellings (`fucina.Block*`,
  `fucina.QuantizedMatmulRhs*`, `fucina.q8_0_block_size`,
  `fucina.supports_q4_k_mmla`) — aliases of the same names under
  `fucina.quant`; removal in the next MINOR release.
- `llm.gemma.gemma4` / `llm.gemma.gemma4_train` / `llm.pockettts.pocket`
  — aliases of `gemma.model` / `gemma.train` / `pockettts.model`; removal
  in the next MINOR release.
- Build options `-Dbackend=cpu` (alias of `scalar`) and `-Daccelerate`
  (compatibility alias of `-Dblas`): both predate this changelog and are
  now on the ledger; removal in the next MINOR release.

### Removed

- The 0.2.0-deprecated aliases, on schedule: `sumExt` / `meanExt` /
  `maxExt` / `minExt` / `crossEntropyExt` / `linearCrossEntropyExt` /
  `linearDistillExt` (use the base names, same signatures) and
  `llm.weights` / `llm.ptqtp_gguf` / `llm.gguf_meta` (use the `fucina.*`
  roots).
- `llm.gemma.moe_route` / `llm.gemma.moe_route_tensor` (experimental
  tier): their code folded into the exec-band kernels
  (`exec/moe_gu.zig`); `llm.gemma.moe` remains the family surface.
- The `lmserve` build module (voiceagent now hosts the engine through
  `llm.serving`), and four consumer-less `PackedMatmul*` /
  `QuantizedMatmulFormat` re-exports on the exec facade (unreachable via
  `fucina.*`; the RHS container vocabulary lives on `fucina.quant`).

### Fixed

- `-Dgpu=metal` builds compile again: the model-I/O promotion had left
  `weights.zig`'s internal shim without `RawTensor` and
  `gpu.has_quant_gemm`, breaking every Metal build since; the leg is now
  exercised (build + llm tests on Metal).
- Two `sum` call sites in `bench/facade.zig` missed the 0.2.0 trailing
  options migration; benches are never compiled by the default build, so
  the subq research tools and this gap are now inside `bench-check`.

## 0.2.0 — 2026-08-22

### Added

- `fucina.tuning`: the shared shape of every `FUCINA_*` route gate
  (`Switch`/`Threshold`: read-once env cache + measured default +
  programmatic `set`), the numeric route defaults, and per-context
  `Overrides` — `ExecContext.setTuning` lets two contexts in one process
  run different route policy (first consumer: the CPU f32 weight-shadow
  route).
- `llm.serving`: the model-agnostic serving contract (`GenerateRequest`,
  `GenerateResult`, `Caps`, the per-family `Backend` vtable), promoted from
  `apps/lmserve` so an out-of-tree server consumes it without vendoring
  the example.
- `llm.runner` (experimental): the descriptor runner — one
  family-independent decoder driven by a runtime `Descriptor`, with two
  block styles (`.fused`: the qwen3/qwen3moe vocabulary; `.host_reference`:
  the GLM/DeepSeek-MoE vocabulary — biased QKV, partial interleaved rope,
  sigmoid noaux MoE + shared experts). `Descriptor.fromGguf` reads
  qwen3-shaped and glm4moe GGUF metadata. Three bitwise parity gates in
  `runner_tests.zig`: real Qwen3-0.6B vs the hand qwen3 port, a synthetic
  in-test MoE GGUF (the CI-safe small-MoE fixture), and a synthetic
  glm4moe GGUF loaded purely from its metadata vs the hand glm4moe port
  (docs/RUNNER.md).
- `dtype.block_formats`: the block-quantization format registry.
  `Storage`, `kind`, `blockSize`, and GGUF's type mapping derive from the
  one table; a comptime completeness check makes an unregistered `DType`
  tag a compile error.
- `lmserve --cors-origin O`: cross-origin browser access is opt-in per
  origin (`*` for any).

### Changed

- **One op per name**: the `Ext` reduction/loss variants folded into their
  base ops, which now take a trailing options argument (`.{}` for
  defaults). Rewrites: `x.sum(ctx, tag)` → `x.sum(ctx, tag, .{})`, same
  for `mean`/`max`/`min`; `x.crossEntropy(ctx, tag, labels)` →
  `x.crossEntropy(ctx, tag, labels, .{})`;
  `sumExt`/`meanExt`/`maxExt`/`minExt`/`crossEntropyExt` →
  the base name, same signature; `linearCrossEntropyExt` →
  `linearCrossEntropy`; `linearDistillExt` → `linearDistill`. The typed
  (non-f32) `sum`/`mean`/`max`/`min` take the same trailing `.{}`.
- Route-gate env switches read through `parallel.envFlag` uniformly (the
  getenv-truthiness contract, first character not `'0'`); previously three
  gates keyed on presence-of-any-value, and one
  (`FUCINA_NO_ATTN_BWD_BLAS`) read `std.c.getenv` directly, which
  libc-free Linux builds cannot compile.
- `src/ag` reorganized: `backward.zig` is a re-export facade over
  per-domain VJP modules (`ag/backward/`), `ag/tensor.zig` a facade over
  per-domain method mixins (`ag/tensor/float/`) plus the typed-constant
  and plumbing bands — public API unchanged.
- Root golden fixtures moved beside their consumers
  (`apps/voiceagent/goldens/`, `src/llm/qwen3tts/goldens/`).

### Deprecated

- `sumExt`, `meanExt`, `maxExt`, `minExt`, `crossEntropyExt`,
  `linearCrossEntropyExt`, `linearDistillExt` — aliases of the merged ops
  above; removal in the next MINOR release.
- `llm.weights` / `llm.ptqtp_gguf` / `llm.gguf_meta` — model I/O lives at
  the root: `fucina.weights` / `fucina.ptqtp_gguf` / `fucina.gguf_meta`;
  removal in the next MINOR release.

### Security

- `lmserve` no longer sends `access-control-allow-origin: *` by default:
  with no API key configured, any web page could read responses from a
  loopback bind on browsers without Private Network Access enforcement.
  CORS is now off unless `--cors-origin` is given.

## 0.1.0 — 2026-08-17

First tagged package: `build.zig.zon` + `zig fetch` consumption of the
`fucina` and `fucina_llm` modules, module-carried BLAS/GPU linkage.
