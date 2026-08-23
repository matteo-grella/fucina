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
  `llm.serving` (the contract: request/result types and the `Backend`
  vtable), `llm.chat`, `llm.tokenizer`, `llm.kv_cache`.
- **Experimental** (changelog entry only): `es`, `ptqtp`, `llm.runner`, the `llm.serving` transport/engine
  band (`http`, `scheduler`, `emitter`, wire dialects, `gguf_chat`,
  `open`), `llm.speculative`, `llm.subq`, the
  research features under model families (SHINE, cartridges, Engram), and
  every model family's internal layout.

## Unreleased

### Added

- `fucina.ParamView`: the per-entry view `ParamRegistry.view` returns
  (name, dtype, shape, mutable byte view, trainability) is now nameable
  at the root; it was returned but unexported.
- `llm.runner.Error.WrongBlockStyle`: the fused entries (`forwardStep*`,
  `forwardLastLogits*`, `initKvCache`) reject a `.host_reference` model
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

- Every deprecated alias, ahead of the one-MINOR schedule: the policy is now
  "no alias step before 1.0" (`docs/DEVELOPMENT.md` §6). Rewrites:
  `fucina.BlockQ*` / `fucina.BlockIQ*` / `fucina.BlockTQ*` /
  `fucina.BlockMXFP4` / `fucina.BlockNVFP4` / `fucina.QuantizedMatmulRhs*`
  / `fucina.q8_0_block_size` / `fucina.supports_q4_k_mmla` → the same
  names under `fucina.quant`; `fucina.simd.{vecScale, vecMaxReduce,
  dotF32F16, scoreRows4F16, vecExpAffineSumInPlace, weightedAccumRows4F16}`
  → `fucina.internal.backend_mod.vector_impl.*` (backend internals);
  `fucina.weights.GroupedQ8_0RhsX4` → `fucina.quant.QuantizedMatmulRhsQ8_0x4`;
  `llm.gemma.gemma4` / `llm.gemma.gemma4_train` / `llm.pockettts.pocket` →
  `llm.gemma.model` / `llm.gemma.train` / `llm.pockettts.model`; build
  options `-Dbackend=cpu` → `-Dbackend=scalar`, `-Daccelerate=true|false`
  → `-Dblas=accelerate|none`.

### Changed

- Internal layout, no public spelling changes: `src/llm/model_common.zig`
  holds the PTQTP-aware projection loader, the dense-FFN containers, the
  MoE expert-trio loader, the embed/norm/lm-head trio, and the GQA head
  map the runner and qwen35 previously each carried; `weights/moe_stream.zig`
  holds `MoeStreamOptions` and the `--moe-*` argv band; `exec/moe_gu.zig`'s
  packed and raw batch bodies share one chain-wiring skeleton; `optim.zig`
  is a facade over `optim/{common,frame,moment_pair,muon,apollo,sgd,schedule,set}.zig`
  (every `fucina.optim.*` spelling and every checkpoint byte unchanged);
  the three core model-I/O files import their home modules directly
  instead of a shadow `fucina` struct. Recorded goldens and llama.cpp parity gates are
  unchanged.
- `parallel.row_kernel_len_threshold` and `parallel.attention_work_threshold`
  name the halved pool gates the fused row kernels and attention paths
  use (previously 27 inline `threshold / 2` sites); values unchanged.
- docs/ARCHITECTURE.md: the band table now lists every `src/*.zig` file,
  and the kernel-boundary wording in `exec.zig`/`backend.zig` states where
  the fused kernels actually live (exec, backend-independent).
- `ExecContext` is one type: the `Runtime` substrate is merged into it
  (its fields are `ExecContext` fields, its methods are free functions in
  `exec/runtime.zig` aliased into the struct); domain modules take
  `*ExecContext`; `ctx.rt.X` -> `ctx.X`. `Runtime` is not public API, so
  no public spelling changes. `zig build arch-check` permits one SCC
  shape, a same-band SCC anchored on a directory root (`exec.zig` <->
  `exec/*.zig`), and reports it in its summary line; every other SCC
  stays an error.
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

### Deprecated

- `fucina.simd.{vecScale,vecMaxReduce,dotF32F16,scoreRows4F16,vecExpAffineSumInPlace,weightedAccumRows4F16}`
  — backend internals that were published under the elemental-op
  namespace; in-tree consumers use `fucina.internal.backend_mod.vector_impl`.
  Aliases kept; removal in the next MINOR release.
- `fucina.weights.GroupedQ8_0RhsX4` — alias of
  `fucina.quant.QuantizedMatmulRhsQ8_0x4` (the same type); removal in the
  next MINOR release.

### Fixed

- qwen35: a layer leak on the load error path (a failing
  `ExpertStore.finalize` after the layers had loaded freed the layer
  slice but not the layers).

## 0.3.0 — 2026-08-23

### Added

- `llm.serving` transport and engine, promoted from `examples/lmserve`:
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
  `examples/lmserve` (nanochat, diffusion-gemma, inkling, qwen35,
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
  `examples/lmserve` so an out-of-tree server consumes it without vendoring
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
  (`examples/voiceagent/goldens/`, `src/llm/qwen3tts/goldens/`).

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
