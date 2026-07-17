# AGENTS.md — Fucina

Fucina is a close-to-metal **CPU tensor / autograd runtime + LLM inference engine** written in
**Zig 0.16**. North Star: **match or beat llama.cpp on CPU**. It runs Qwen3
dense and the Qwen3-MoE (`qwen3moe`) family, Gemma 4, and several other model families from GGUF
weights (see `docs/RUNNING-MODELS.md`; model weights are not part of the repo). It is CPU-first, with an
optional Metal GPU GEMM offload via `-Dgpu=metal` (see the build options + `src/backend/metal.zig`
below). There is no ggml graph runtime and no C/CMake build — pure Zig vector kernels plus optional
CBLAS for GEMM (the Metal `shim.m`/`.metal` kernels are vendored, not a CMake build).

This file is the working guide for contributors and coding agents: toolchain, build/test commands,
repo map, house rules, and the doc index.

## Toolchain

- Pinned to **Zig 0.16.0** (`zig version` → `0.16.0`).
- No `build.zig.zon` / package manifest — modules are wired directly in `build.zig`.

## Build, test, run, bench

```sh
zig build test                 # unit tests — NINE roots: src/fucina.zig, src/llm.zig, examples/lmserve.zig, examples/nam.zig, examples/parakeet.zig, examples/omnivoice.zig, examples/locate_anything.zig, examples/facedetect.zig, examples/nanochat.zig
zig build test -Dbackend=scalar   # reference scalar backend
zig build test-fucina -Dbackend=scalar  # routine scalar leg: fucina root only (the kernel/spec surface); run the full nine-root scalar matrix pre-merge
zig build test -Dblas=none        # native backend via pure Zig vector kernels (no CBLAS)
zig build arch-check           # production-only src import graph (AST-based, test-aware): enforces 0 SCCs
zig build doc-check            # doc-index link check: every doc named in AGENTS.md's doc index must exist (tools/check_doc_links.zig)
zig build snippet-check        # REFERENCE.md snippet gate: every runnable ```zig snippet (named test block) extracted and run against the real fucina/fucina_llm modules (tools/gen_snippet_tests.zig)
zig build x86dot-check         # cross-ISA int8/Q4_K/Q8_0/TQ2_0 dot parity checker (follows -Dtarget) + compile-only AVX2/VNNI/smmla bit-rot legs (src/x86dot_check.zig)
zig build cuda-check           # compile-only -Dgpu=cuda legs (x86_64-linux-gnu fucina/llm roots + NVRTC PTX generator, not run): CUDA-provider bit-rot gate for GPU-less machines
zig build run                  # smoke example (examples/smoke.zig)
zig build qwen3 -- <args>      # Qwen3 GGUF inference (examples/qwen3.zig; --spec/--spec-ref = lossless speculative decode, --tokenize = tokenizer-parity oracle)
zig build gemma4 -- <args>     # Gemma 4 GGUF inference / logit-parity harness; --chat/--repl/--spec (examples/gemma4.zig)
zig build qwen35 -- <args>     # Qwen3.5 (qwen35 hybrid Gated-DeltaNet) GGUF — loader/parity harness (examples/qwen35.zig; see examples/qwen35/README.md)
zig build deepseek2 -- <args>  # DeepSeek-V2 family (MLA + fine-grained MoE) GGUF inference (examples/deepseek2.zig)
zig build glm4moe -- <args>    # GLM-4.5 family GGUF inference; --mtp = native multi-token-prediction speculative decode (examples/glm4moe.zig)
zig build deepseek4 -- <args>  # DeepSeek V4 Flash GGUF inference (hyper-connections, compressed KV, streamed experts, MTP; examples/deepseek4.zig)
zig build inkling -- <args>    # Inkling GGUF inference / parity harness (hybrid rel-bias attention, shortconv sites, sink-shared MoE; examples/inkling.zig)
zig build lmserve -- <args>    # OpenAI-compatible LM server (chat completions + stateless responses, SSE, JSON-schema constrained output w/ -Dllguidance=true) over qwen3/gemma4/diffusion-gemma GGUFs + nanochat checkpoints (examples/lmserve.zig; see docs/LMSERVER.md)
zig build parakeet -- <args>   # Parakeet ASR (NeMo FastConformer): WAV → text; --stream/--manifest/--mic (needs -Dparakeet-mic), --compare parity harness (examples/parakeet.zig)
zig build omnivoice -- <args>  # OmniVoice MaskGIT TTS: voice cloning/design/auto, codec encode/decode, parity oracles (examples/omnivoice.zig)
zig build facedetect -- <args> # buffalo_l face pipeline (face-detect.cpp port): info/detect/embed/verify/analyze + bench paired CPU harness (examples/facedetect.zig)
zig build nanochat -- <args>   # nanochat port (karpathy/nanochat): tok-train / base-train / sft / eval-bpb / chat — full CPU pipeline, GPT pretraining + SFT + chat w/ calculator tool (examples/nanochat.zig)
zig build diffusion-gemma -- <args>  # DiffusionGemma block text-diffusion: --eval parity harness vs llama.cpp PR 24423, --chat EB decoding (examples/diffusion_gemma.zig)
zig build locate-anything -- <args>  # LocateAnything-3B open-vocabulary detection: detect/info CLI + exit-code parity gates vs reference dumps (examples/locate_anything.zig)
zig build spirals              # two-spirals training demo: SGD/AdamW/Muon/APOLLO + checkpoints (examples/spirals.zig)
zig build nam -- <args>        # Neural Amp Modeler: .nam profile import/run/train/export, GGUF interchange, live amp sim (examples/nam.zig)
zig build finetune -- <args>   # LoRA fine-tune a Qwen3 GGUF on CPU (examples/finetune.zig)
zig build cartridge -- <args>  # Cartridges (arXiv 2506.06266): train a corpus into a reusable KV prefix by in-process self-study distillation + serve it (examples/cartridge.zig; see docs/CARTRIDGES.md)
zig build es-finetune -- <args>  # gradient-free ES fine-tune of a Qwen3 GGUF (examples/es_finetune.zig; --mode lora|full, --reward rule|nll|acc)
zig build es-spirals           # two-spirals MLP trained FROM SCRATCH by ES (examples/es_spirals.zig; self-verifying, member-parallel replicas)
zig build es-ternary-spirals   # two-spirals MLP with PACKED TERNARY (TQ2_0) hidden/output layers trained by ternary-native ES — training state IS the int8 inference model (examples/es_ternary_spirals.zig; see docs/TERNARY.md)
zig build ptqtp-spirals        # float-train a two-spirals MLP, then post-training-quantize it to DUAL TRIT-PLANES (PTQTP, arXiv:2509.16989: packed TQ2_0 plane pairs; self-verifying — examples/ptqtp_spirals.zig, docs/PTQTP.md)
zig build ptqtp-qwen3 -- <gguf>  # PTQTP-decorate a Qwen3 GGUF's linears in place (any source dtype) + teacher-forced NLL before/after + greedy completion; --planes 1|2|3, --down-planes/--o-planes N = selective third plane, --skip-first/--skip-last N = edge layers stay source precision (examples/ptqtp_qwen3.zig)
zig build export-gguf -- <args>  # export a GGUF: re-emit/transcode (incl. --dtype tq2_0 ternary), merge LoRA adapters into dense weights, or shard-streaming PTQTP quantization (--ptqtp[=K], one tensor at a time — models bigger than RAM; tools/export_gguf.zig)
zig build bench                # MLP-shaped inference/backward benchmarks
zig build bench-gate           # paired Fucina-vs-llama benchmark gate (tools/bench_gate.py; protocol in docs/BENCHMARK.md)
zig build bench-optim          # optimizer step kernels at LLM shapes (bench/optim.zig)
zig build bench-ce             # softmax / cross-entropy / layerNorm row kernels at LLM shapes (bench/ce.zig)
zig build bench-scatter        # scatter-add (embedding-gradient) kernel at vocab x dim shapes (bench/scatter.zig)
zig build bench-backend        # scalar vs native backends on representative ops
zig build bench-f16gemm        # f16 TransB GEMM parallel-efficiency microbench
zig build bench-gemm           # large-shape f32 GEMM: row kernels vs blocked packed kernel vs BLAS dispatch (bench/gemm.zig)
zig build bench-packed-gemm    # pack-once dense f32/f16/bf16 RHS GEMM at skinny-m inference shapes (bench/packed_gemm.zig)
zig build bench-gpu-dispatch  # CPU BLAS vs blocking/async eager GPU GEMM/GEMV latency + queued throughput
zig build bench-gpu-formats   # packed CPU vs eager GPU f16/Q4_K/Q5_K/Q6_K/Q8_0 LLM-linear latency + queued throughput
zig build bench-q5kmoe         # Q5_K MoE-expert matmul: per-row vs 4-row lane-packed col-outer (bench/q5kmoe.zig)
zig build bench-ternary        # TQ2_0 ternary matmul: hot sdot/vpdpbusd tiles vs cold table path, mul-free f32 path, Q4_K, dense f32 (bench/ternary.zig)
zig build bench-attention-backward  # grouped causal attention backward (bench/attention_backward.zig)
zig build bench-facade         # raw tensor ops vs public no-grad Tensor facade
zig build bench-einsum         # einsum vs hand-written dot/permute contraction pipelines (parity + advantage cases)
zig build bench-backward-diamond  # serial vs manual-parallel independent GEMM VJPs
```

Build options (consumed at comptime via `build_options`):

- `-Dbackend=native|scalar|cpu` — `native` (default) = Zig SIMD + optional BLAS; `scalar` = reference;
  `cpu` is a deprecated alias for `scalar`.
- `-Dblas=none|accelerate|openblas|mkl|blis|nvpl|blas` — CBLAS provider for GEMM. Default `accelerate`
  on macOS, `none` elsewhere. `none` keeps the native backend on its pure Zig vector kernels.
- `-Dblas-threads=N` — pin vendor BLAS threads (`0` = provider default).
- `-Dmax-threads=N` — comptime worker-team ceiling *and* runtime default thread count (1–64,
  default 8 = M1 Max P-cores; `src/parallel.zig`). `FUCINA_MAX_THREADS` still only lowers it at
  runtime (works on static/non-libc Linux too, via `/proc/self/environ`) — many-core servers
  must raise the ceiling at build time.
- `-Dgpu=none|metal|cuda` — GPU GEMM offload. **metal** (macOS): big f32 GEMMs (cold single-op gate 2^32
  m·n·k work, `FUCINA_GPU_MIN_WORK` override, `FUCINA_GPU=0` kill switch) run on the GPU via the
  vendored MLX steel kernel, and the Gemma/Diffusion MoE expert FFN runs as grouped
  dequant-in-kernel Q6_K/Q8_0 GEMMs (vendored ggml mul_mm; `FUCINA_GPU_MIN_WORK_QMOE` work gate +
  `FUCINA_GPU_QMOE_MIN_FILL` tile-occupancy gate (default 50% — small-m expert batches whose
  32-row tiles would run mostly empty stay on CPU; 0 = old behavior), raw-block CPU fallback —
  gpu builds keep ONE raw expert representation instead of the x4 packs). **Dense quantized
  linears** (Q4_K/Q6_K/Q8_0 — e.g. the qwen3/gemma prefill projections) also offload via the same
  `gemmQuantNtAsync` dequant-in-kernel GEMM (`weights.linearSeqQ*` → `ExecContext.denseQuantMatmulGpu`,
  per-format `FUCINA_GPU_MIN_WORK_DENSE_Q4/Q6/Q8` gates against the CPU packed-kernel fallback,
  stable RHS residency, ~+33% pp on 0.6B-Q4_K);
  decode (m=1, below the gate) and training (grad path) stay on CPU.
  On both providers, eligible **dense f32, f16, and provider-supported stable quantized** commands submit eagerly
  to persistent provider lanes and synchronize only at a CPU visibility boundary; pending CUDA outputs
  pass their device pointer directly to dependent GEMMs. CUDA registers pooled host allocations once
  and overlaps upload/compute/download; resident ordinary GEMM uses `FUCINA_GPU_MIN_WORK_RESIDENT`
  (default 2^27). Resident f32 `m≤8` uses the separate
  `FUCINA_GPU_MIN_WORK_GEMV` gate (default 2^24), and resident CUDA f16 uses
  `FUCINA_GPU_MIN_WORK_F16_RESIDENT` (default 2^20). This is completion tracking, not a graph; see
  `docs/GPU-OFFLOAD.md`.
  **cuda** (Linux/NVIDIA): no CUDA SDK at build time — dlopen'd
  cuBLAS + vendored PTX kernels; cross-compiles from macOS with `-Dtarget=x86_64-linux-gnu`.
  Covers big f32 GEMMs (strict FP32; `FUCINA_GPU_TF32=1` opts into TF32 tensor cores,
  `FUCINA_GPU_MIN_WORK_TRANSIENT` floors non-resident operands), f16 NT GEMM, dense quantized
  Q4_K/Q5_K/Q6_K/Q8_0 prefill (Q5_K is CUDA-only; adaptive N32/N64 f16-input/f32-accumulate tensor-core kernels;
  underfilled dense grids use on-stream split-K/reduction, disabled with
  `FUCINA_GPU_QUANT_SPLIT_K=0`; scalar fallback with `FUCINA_GPU_QUANT_MMA=0`) + the grouped MoE expert FFN (same tile-table
  protocol and gates as metal; stable RHS bytes are adopted into a managed-memory registry — one PCIe crossing per
  weight per process), and opt-in quantized decode (`FUCINA_GPU_DECODE=1`, m≤8, resident weights
  only; Q5_K uses GEMV for m<4 and tiled MMA for m=4..8, with a measured work gate).
  `FUCINA_GPU_VRAM_BUDGET` bounds residency;
  `FUCINA_GPU_KERNELS=src` NVRTC-recompiles the vendored kernels (dev loop;
  `tools/gen_cuda_ptx.sh` regenerates the committed PTX through the same NVRTC frontend). `zig build cuda-check` is the
  compile-only bit-rot leg (fucina + llm test roots and the NVRTC PTX generator for x86_64-linux-gnu).
- `-Dvector-scan=bool` — vectorize the scan kernels (`cumsum`/`cumprod` + cumsum's reverse VJP
  pass; default `false` = the documented serial-per-row scans). On: non-last-axis scans vectorize
  across independent columns (bitwise identical to serial); last-axis scans use an in-register
  prefix scan — still bitwise deterministic for any thread count, but the accumulation order
  differs from the serial default (the sum-SIMD-lanes rounding class; exact for integer data).
- `-Daccelerate=bool` — compatibility alias (`false` ≈ `-Dblas=none`).
- Standard `-Doptimize=Debug|ReleaseSafe|ReleaseFast|ReleaseSmall` and `-Dtarget=...`.
- **CPU targeting: native by default.** With no `-Dtarget`, Zig targets the compiling machine's
  exact CPU (full detected feature set, like `-march=native`), and the kernels' comptime feature
  gates (`src/backend/quant/common.zig:13-31`) compile in the matching arms — NEON/sdot on Apple
  Silicon, AVX2/AVX-VNNI on modern x86, smmla on I8MM-class ARM servers, portable vectors
  elsewhere; unused arms are compiled out entirely (no runtime dispatch). Cross-compiling with
  `-Dtarget=...` drops to that architecture's BASELINE unless `-Dcpu=...` names a model
  (`x86_64_v3`, `alderlake`, `znver4`, `neoverse_v1`, …) — a bare `-Dtarget` binary silently
  loses the fast kernels. Build on the machine that will run it, or pin `-Dcpu` to match it.

## Repo map

| Path | Role |
| --- | --- |
| `src/fucina.zig` | Public facade (the `fucina` module root). |
| `src/tensor.zig` | Raw internal tensor (shape/stride/offset, rank ≤ 8). |
| `src/tagged.zig`, `src/tags.zig` | Tag-semantics op library over raw tensors + comptime rank/axis-tag metaprogramming (no tagged tensor type). |
| `src/exec.zig` | `ExecContext` — forwarding facade over `src/exec/`: embeds one `Runtime` as `rt`, forwards every op to its domain module, and re-exports the option/result types. Exec scopes live here too (openExecScope/closeExecScope: implicit ownership of training intermediates). |
| `src/exec/` | The eager-runtime implementation. `runtime.zig` = leaf `Runtime` substrate (allocator, worker team, exec-scope stack, tensor allocation primitives; domain modules take an explicit `*Runtime`, never `self: anytype`); `buffer_pool.zig` = the reusable transient-buffer pool (see `docs/MEMORY-MODEL.md`); domain modules (`attention`, `matmul`, `quant_matmul`, `moe`, `norm`, `rope`, `softmax`, `loss`, `stats`, `topk`, `reduce`, `gather_scatter`, `conv`, `pool`, `convert`, `elementwise`, `shape`, + the `row_ops` kernel leaf); `expert_store.zig` = disk-backed MoE expert store (pinned set + per-layer LRU + pread readahead — out-of-core experts for models larger than RAM); `moe_chain.zig` = shared batched-MoE scheduling leaf (expert-grouped route plan, phase-chain machinery, chunk helpers, profile timers) consumed by `exec/moe.zig` and — via the `ExecContext.moe_chain` re-export — by the gemma MoE engines. |
| `src/backend.zig`, `src/backend/` | Final numeric kernels (`native.zig`, `cpu.zig`, `ops.zig`, `vector/`, `quant/`, `packed.zig`; `vector/gemm_blocked.zig` = BLIS-style blocked packed f32 GEMM for the no-BLAS path; `quant/` also holds the f32→quantized row ENCODERS — Q4_K/Q5_K/Q6_K/TQ2_0 + legacy, byte-exact ggml parity, `quantizeRowForDType` dispatch; `quant/ternary.zig` = the hot TQ2_0 ternary {-1,0,+1} kernels — int8 sdot/vpdpbusd flagship + mul-free f32 path + b1.58 absmean encoder, see `docs/TERNARY.md`). |
| `src/x86dot_check.zig` | Standalone cross-ISA parity checker for the int8 dot primitives + Q4_K/Q8_0/TQ2_0 dot kernels (Rosetta/qemu/x86-hardware validation vehicle; per-arm execution-coverage table + build matrix in its header). |
| `src/storage.zig` | Refcounted owned storage. |
| `src/accelerator.zig` | Backend-neutral lifetime tokens for already-submitted eager GPU work and storage-lifetime mapping resources (completion tracking only; no compute graph). |
| `src/dtype.zig` | Scalar + block-quantized dtype definitions. |
| `src/parallel.zig`, `src/thread.zig` | Thread pool + parallel-chunk helpers. |
| `src/gguf.zig` | GGUF parser + writer (`Writer`: byte-verbatim metadata passthrough, llama.cpp-exact offsets/padding — a 449 MiB verbatim re-emit is byte-identical; `encodeF32` = the writer-side quantize seam onto the `quant/` encoders). |
| `src/rng.zig` | Repo-owned deterministic RNG: splitmix64, counter-based `at(seed, i)`, uniform/gaussian/kaiming/normal fills. The (seed→values) mapping is a checkpoint contract (APOLLO projections, dropout masks). |
| `src/ag.zig`, `src/ag/` | Autograd: `tensor.zig` (facade), `backward.zig` (VJPs), `core.zig` (scheduling), `checkpoint.zig` (activation checkpointing — recompute-in-backward, `checkpoint`/`checkpointWithContext`). |
| `src/optim.zig` | Optimizers (SGD/AdamW/Muon/APOLLO), grad clipping, LR schedule, OptimizerSet (param groups), checkpoint save/load (positional FZT1 + named/dtype-aware safetensors state dicts, name-matched optimizer state v3, `addParamNamed`; native frames FZAD/FZA3/FZM3/FZP3/FZS3/FZO3). Golden-parity-tested vs the torch references. |
| `src/training_checkpoint.zig` | Canonical training checkpoint directory helper: clean `model.safetensors`/`adapters.safetensors`, native `optimizer.fucina`, and JSON `trainer_state.json` commit sentinel. |
| `src/lora.zig` | LoRA adapters over frozen weights: `Adapter(in_tag, out_tag)`, kaiming-A/zero-B init, delta/apply, named persistence, f32/f16 merge. |
| `src/es.zig` | Evolution strategies at scale (gradient-free ES-at-scale, arXiv:2509.24372): seed-regenerated noise (vectorized fast-gaussian contract; opt-in antithetic pairs, centered-rank shaping, anchored weight decay) over f32/f16/bf16 params (facade tensors or a whole `ParamRegistry`, frozen entries included), in-place perturb/restore + member-parallel replicas, z-scored or centered-rank (Salimans-style) update, chunk-parallel deterministic kernels; on `-Dgpu=cuda`, resident params perturb/update on the device via bitwise-identical kernels (kernels.cu). Goldens: `tools/gen_es_goldens.py`; reference cross-check: `tools/check_es_parity.py` (refs/es-at-scale). Also hosts the ternary-native strategy: packed TQ2_0 genomes (sparse trit-flip perturbations, EGGROLL-style top-K vote-and-threshold one-bin updates, `es_trits` RNG domain) so training state == the int8 inference model — see `docs/TERNARY.md`. See `docs/TRAINING.md` §13. |
| `src/param_registry.zig` | `ParamRegistry` — comptime-reflective named parameter registry over the tagged facade (borrows tensors; names are checkpoint field paths; bridges `OptimizerSet` + the state-dict layer). Used by `examples/spirals.zig` and both LLM trainers. |
| `src/ptqtp.zig` | PTQTP trit-plane PTQ (arXiv:2509.16989, implemented from the paper's formulas): data-free `W ≈ α₁T₁ + α₂T₂` per 256-column group via alternating adaptive-λ ridge regression + exhaustive trit search (pinned tie-break order = deterministic + symmetric-init breaker); packs each plane as a standalone valid TQ2_0 tensor (per-block fp16 `d` = the group scale → inference is two stock ternary matmuls + add), `reconstructReference` = arbitrary-G fidelity path. LLM seam: `LinearWeight.toPtqtp` / qwen3 `Model.decoratePtqtp`. See `docs/PTQTP.md`. |
| `src/state_dict.zig`, `src/safetensors.zig` | Neutral named-tensor state-dict serialization (`LoadOptions.aliases` = name remapping) over the Hugging Face safetensors reader/writer. GGUF stays a separate LLM interop codec. |
| `src/bench_raw.zig` | Internal raw-surface module for `bench/` (wired as the `bench_raw` import in `build.zig`; not part of the public facade). |
| `src/llm.zig`, `src/llm/` | LLM/ASR module root. **Model families grouped in `llm/<family>/`, exposed as namespaces** by `src/llm.zig`: `llm.qwen3.{model,train}`, `llm.qwen35.model`, `llm.gemma.{gemma4,gemma4_train,moe,…}`, `llm.diffusion_gemma.model`, `llm.deepseek2.model`, `llm.glm4moe.model`, `llm.deepseek4.model`, `llm.inkling.model`, `llm.parakeet.{decoder,loader,encoder,…}`, `llm.speculative.{core,sam_index,recycling,cascade,constrained}`. **Generic helpers stay flat in `src/llm/`**: `kv_cache.zig` (f16 default + opt-in q8_0 — the capacity option, see `docs/BENCHMARK.md`; `truncate` = the speculative rewind), `kv_persist.zig` (crash-safe append-only KV-cache sidecar — conversations reopen warm), `ptqtp_gguf.zig` (PTQTP plane persistence: `<name>.ptqtp0/1/2` writer + pair-detecting loader), `tokenizer.zig` (byte-level BPE + faithful qwen2 pretokenizer — token-ID-exact vs llama-tokenize), `spm_tokenizer.zig` (gemma SPM), `sampler.zig` (llama.cpp-compatible pipeline + the `LogitProcessor` hook), `logit_processor.zig` (pluggable pre-sampling logit transform: process/commit/reset + optional structural hooks — the constrained-decoding seam, speculative-safe by the every-sample-is-committed argument), `llguidance.zig` (grammar/JSON-schema token masking over the vendored llguidance engine, `-Dllguidance=true`; stub otherwise; `Constraint.clone` = the per-stream primitive), `weights.zig` (incl. resident-bf16 arm), `gguf_meta.zig` (flat GGUF loader glue: `metaInt`/`metaFloat`(+`Opt`) with `ZeroPolicy` + comptime-generic `parallelLoadLayers`; qwen3/qwen35/gemma4 use it, parakeet/omnivoice keep their parity-bound variants), `chat.zig` (genuinely generic `Conversation(Model, Tok)` chat/REPL engine — qwen3 AND gemma4; `Options.speculation` wires the speculative decoder, `extra_stop_ids`/`stop_sequences`; `stop_sequences` + speculation is an init error — the lossless one-draw contract; `sendBatch` = lockstep batch-N decode over N sibling conversations via `Model.forwardStepBatch`, speculation excluded, per-stream `logit_processor`s enforced distinct; `Options.logit_processor` = per-turn-reset constrained decoding on every path), `data.zig` (SFT dataset/dataloader: `SftText` JSONL/static pairs, `encodePair` template+tokenize+shift+mask, deterministic `Loader` — the `(seed, epoch) → permutation` mapping is a golden-pinned checkpoint contract; tokenizer is duck-typed so BPE and SPM both fit), `cartridge.zig` (Cartridges, arXiv 2506.06266: trained KV-prefix corpus compression — per-layer post-RoPE K/V rows with a frozen attention sink, teacher top-k distillation loss, safetensors persistence, `writeToCache` serving; see `docs/CARTRIDGES.md`), `unicode_categories.zig`. Training Trainers: `qwen3/train.zig`, `gemma/gemma4_train.zig` (param plumbing via `ParamRegistry`; `lossExt` = reduction/scale knobs for gradient accumulation, qwen3 adds `forwardHidden`/`lossInjected` — truncated-forward and injected-embedding seams for representation experiments; exercised by the `research/nla` branch). `build.zig` only references the `src/llm.zig` root. |
| `src/llm/speculative/` | Speculative decoding subsystem (`llm.speculative.*`): `core.zig` (`DraftSource` vtable, `SpeculativeDecoder` — one lossless verify path for greedy AND sampled, `CostGate` never-a-loss auto-off), `cascade.zig` (`SpeculationIndex` cascade: conversation SAM + injectable frozen refs (`addReference`, RAG seam) + recycling fallback), `sam_index.zig` (online suffix automaton, ~110 B/token), `recycling.zig` (Token-Recycling adjacency matrix), `constrained.zig` (`ConstrainedSource`: grammar-forced spans draft themselves, invalid drafts pruned pre-verify — see `docs/CONSTRAINED-DECODING.md`). See `docs/SPECULATIVE.md`. |
| `src/llm/diffusion_gemma/model.zig` | DiffusionGemma (`diffusion-gemma`): block text-diffusion on the gemma4 backbone — causal encode + bidirectional canvas forwards over one weight set, sparse self-conditioning, entropy-bound sampler, block-AR generate. Parity harness targets llama.cpp PR 24423. |
| `src/backend/metal.zig`, `src/backend/metal/` | `-Dgpu=metal` GPU GEMM provider: Zig host (lazy init, work-threshold gates, device-owned weight storage) + ObjC shim (`shim.m`) + vendored MLX steel kernel (`mlx_gemm.metal`, f32/f16) + vendored ggml `kernel_mul_mm` (`ggml_mul_mm.metal`, dequant-in-kernel Q6_K/Q8_0 with a CPU-built grouped tile table — the MoE prefill path); f32/f16 gates sit in front of the BLAS arms in `native.zig`; the quantized grouped MoE entry is `batchRawGpu` in `src/llm/gemma/moe.zig` (dispatched from `batchRaw`), driving `fucina.internal.backend_mod.gpu_impl` directly from the llm layer; the dense-quant arm goes through `ExecContext.denseQuantMatmulGpu` (`src/exec.zig` → `src/exec/quant_matmul.zig`). |
| `src/backend/gpu.zig`, `src/backend/cuda.zig`, `src/backend/cuda/` | `gpu.zig` = comptime provider selector (`gpu_impl` resolves to metal.zig or cuda.zig; dead arm never analyzed). `cuda.zig` = `-Dgpu=cuda` provider (Linux/NVIDIA): same contract/decl surface as metal.zig — f32/f16 GEMM via cuBLAS, adaptive tensor-core quantized dense/grouped-MoE GEMM (underfilled dense grids use graphless on-stream split-K/reduction; scalar fallback) + decode GEMV + fused prefill attention via the vendored kernels (`cuda/kernels.cu` → committed `cuda/kernels.ptx`, driver JIT, NVRTC dev fallback; regen: `tools/gen_cuda_ptx.sh`), managed-memory weight residency with a stable-RHS adoption cache; dlopen host binding in `cuda/api.zig` (function-pointer prototypes via `std.DynLib`, soname ladders, zero CUDA SDK at build). |
| `src/llm/unicode_categories.zig` | Generated \p{L}/\p{N}/\s tables matching llama.cpp's unicode-data (regen: `tools/gen_unicode_categories.py`) — token-ID-exact pretokenizer parity. |
| `tools/export_gguf.zig` | `zig build export-gguf`: GGUF re-emit / transcode (`--dtype f16/bf16/f32/q8_0/q4_k/q5_k/q6_k/verbatim`; `--experts-dtype` = experts-only override for `*_exps.weight` tensors, may requantize a quantized source via `gguf.decodeF32`) + LoRA-adapter merge into dense weights (`--adapters`, safetensors → `blk.*`) + shard-streaming PTQTP quantization (`--ptqtp[=K]`, tensor-at-a-time incl. per-expert-slice 3D MoE stacks via `ptqtp_gguf.quantizeMoeStack`; `--dry-run` plan; docs/PTQTP.md). |
| `examples/` | `smoke.zig`, `qwen3.zig` (incl. `--spec`/`--spec-ref`/`--spec-bench` speculative decoding + `--tokenize` parity oracle), `gemma4.zig` (chat/REPL on the generic `Conversation`, incl. `--spec`), `qwen35.zig`, `diffusion_gemma.zig`, `parakeet.zig` (ASR CLI over the `llm.parakeet` family), `spirals.zig` (training end-to-end), `finetune.zig` (LoRA SFT on a Qwen3 GGUF end-to-end; `--verify-grads` = the real-model gradient-evidence audit), `es_finetune.zig` (its gradient-free ES twin: same data/checkpoint plumbing, `--mode lora|full`, `--reward rule|nll|acc`), `es_spirals.zig` (two-spirals FROM SCRATCH by ES — the gradient-free from-random-init acceptance demo, self-verifying, member-parallel replica evaluation). |
| `examples/omnivoice.zig`, `examples/omnivoice/` | OmniVoice port: MaskGIT non-autoregressive TTS (k2-fsa/OmniVoice via omnivoice.cpp) — Qwen3-0.6B backbone with bidirectional attention + additive 0/1 bias, hybrid 8-codebook audio embedding, Higgs Audio v2 codec (HuBERT + DAC + RVQ), PyTorch-CUDA-aligned Philox, torchaudio-exact resampler, pydub-parity postproc. Voice cloning / voice design / auto voice, single-shot + chunked. Parity: byte-exact tokens + RVQ codes vs the C++ reference (F32), audio cosine ≥ 0.99999; 2.3–32.7× faster on CPU (M1 Max). Tests run under `zig build test` (parity suites gated by `OMNIVOICE_PARITY`). |
| `examples/locate_anything.zig`, `examples/locate_anything/` | LocateAnything-3B port (open-vocabulary detection / visual-grounding VLM, via mudler/locate-anything.cpp): MoonViT tower + MLP projector + Qwen2.5-3B LM + MTP parallel box decoding, entirely on stock tensor ops — interleaved 2D RoPE through a hand-filled `RopeTable`, bidirectional grouped attention (+ the additive-bias arm for the MTP block-diffusion mask), resident f32 KV, `LinearWeight` linears (BLAS/Metal/CUDA dispatch unchanged); PIL-exact bicubic preproc, pure-Zig PNG IO, reference-exact special-token BPE over `llm.tokenizer`, verbatim-scalar MTP decode heuristics, byte-compatible detections JSON. Parity: `compare` exit-code gates vs reference dumps regenerable with `tools/ref-patches/la_dump.cpp` (token-ID-exact tokenizer/prompt, byte-exact preproc, tight-f32 tower/logits, exact slow/hybrid/fast streams). |
| `examples/facedetect.zig`, `examples/facedetect/` | face-detect.cpp port: the buffalo_l pack (SCRFD det_10g detector + ArcFace R50 recognizer + genderage + MiniFASNet anti-spoof) plus 2d106/1k3d68 dense landmarks — channel-last conv2d/pool2d/prelu/channelAffine over the public facade, load-once Model structs (weights dequant/repack/BN-fold once, forwards are pure compute), an app-level compiled replay (`graph.zig`) for the interpreter-driven nets, cv2-exact letterbox and umeyama align. Parity: byte-identical detect/analyze JSON vs the reference CLI, embed cosine 0.999999, anti-spoof real_prob exact, landmarks ≤0.03px (goldens + regeneration recipe in `examples/facedetect/goldens/README.md`). Tests run under `zig build test`. |
| `examples/nanochat.zig`, `examples/nanochat/` | nanochat port (karpathy/nanochat): the full CPU pipeline — rustbpe-equivalent BPE tokenizer training, GPT pretraining (Muon+AdamW, the MuonAdamW variant: Polar-Express orthogonalization + NorMuon variance reduction + cautious WD, reusing `fucina.optim.AdamW` for the Adam groups), SFT, bits-per-byte eval, and a chat CLI with a calculator tool over a KV-cached decode path. Entirely example-local over the public facade (no new core ops): half-split RoPE (inverse-built table), grouped causal attention, `crossEntropyExt`, gather/scatter embeddings, relu² MLP, tanh softcap. Parity vs the Python reference (CPU fp32): tokenizer encode token-ID-exact + trainer byte-identical to rustbpe, dataloader batches byte-identical, forward per-layer ≤1e-5, optimizer-step + loss-trace within a drift budget, greedy decode token-exact vs a trained reference checkpoint (goldens + regen recipe in `examples/nanochat/goldens/README.md`). `NANOCHAT_PARITY`-gated suites run under `zig build test`. |
| `examples/nam.zig`, `examples/nam/` | Neural Amp Modeler port: `.nam` reader/writer with upstream-exact weight ordering, streaming WaveNet/LSTM/ConvNet/Linear engines (golden parity vs NeuralAmpModelerCore render), classic-recipe trainer over the core `causalConv1d` op, lossless GGUF interchange, vendored-miniaudio live device I/O, CoreMIDI control of the live knobs (`midi.zig`/`midi_shim.c` — hot-plug needs the runloop pump in the shim). Tests run under `zig build test`. |
| `bench/` | Microbenchmarks (`mlp`, `backend`, `f16gemm`, `gemm`, `gpu_dispatch`, `gpu_formats`, `q5kmoe`, `attention_backward`, `facade`, `einsum`, `backward_diamond`, `optim`, `ce`, `scatter`) + shared helpers (`alloc.zig`, `timer.zig`). |
| `refs/` (untracked, by convention) | Optional local clones of the upstream reference repos (llama.cpp, omnivoice.cpp, parakeet.cpp, NeuralAmpModelerCore, …) plus locally captured parity goldens (e.g. `refs/omnivoice-research/goldens/`, consumed by the env-gated parity suites, which skip cleanly when absent). Source comments and `docs/BENCHMARK.md` cite `refs/<repo>/<file>:<line>` into these clones as parity provenance; `tools/fetch_refs.sh` clones every reference at the snapshot's pinned commit (`--build` also builds llama.cpp CPU-only). Nothing in the default build or test run depends on them. |

**Placement policy (ports and families).** Engines intended as reusable `src/llm` families —
anything other src/llm consumers or the chat/session layer should import — go in
`src/llm/<family>/` (parakeet is the precedent). Single-purpose parity ports and their DSP/IO
plumbing stay example-local in `examples/<name>/` (nam, omnivoice — the parity suites pin their
layout). Family-specific kernel orchestration lives in `src/llm/<family>/` over the
`fucina.internal` seam (e.g. the gemma MoE engines in `src/llm/gemma/moe.zig`), never inside the
generic exec runtime. Audio-IO helpers get promoted to a shared home once a second consumer
appears; until then reuse across examples is explicit in `build.zig`.

## House rules (repo-specific)

- **Benchmark before "done".** Kernel/perf changes are not complete until measured against the
  protocol in `docs/BENCHMARK.md`. SOTA CPU perf is the point; bench in `ReleaseFast`, validate in
  `Debug`/`ReleaseSafe`.
- **Backend outputs are exec-supplied.** Kernels never allocate or retain output tensors — results
  go into buffers supplied by `ExecContext`/`Runtime`, and the vector/quant compute leaves are
  allocation-free and infallible. One deliberate exception: the quantized-RHS *dispatch* tier
  (`matmul2DQuantizedRhs*` in `backend/native.zig`/`cpu.zig`) takes an explicit allocator for
  per-call LHS-quantization scratch (stack fast paths keep decode heap-free); RHS pack-prep
  allocates at load time. Don't add allocation below that tier.
- **Explicit ownership.** Storage is refcounted and owned; `[]T` slices/tensor views *borrow*. State
  who owns vs. borrows; pair every allocation with deterministic `errdefer`/`defer` cleanup.
- **Deinit convention.** `deinit(self)` for structs whose members carry their own ctx/allocator or
  that store one; `deinit(self, allocator)` for POD-ish/array-held structs holding raw slices or
  unmanaged stdlib containers. Either way, end with `self.* = undefined` (the Debug use-after-deinit
  tripwire) unless `self` is taken by value.
- **Validate, then call an unchecked kernel.** Hot paths check shape/stride/alignment/contiguity in
  the caller/runtime, then dispatch to a small unchecked, allocation-free backend kernel.
- **Eager and local.** No global graph object, no fusion/compiler layer. Don't add one without a
  concrete design.
- **Build-time backend selection** — dispatch is compiled away; prefer exhaustive `switch` over
  dtype/backend so adding a variant forces edits everywhere.
- **Surgical changes.** Match existing style; touch only what the task needs; don't refactor or
  delete pre-existing dead code unasked.

## Zig 0.16 notes (version-delta traps)

Full reference: the official language documentation at
https://ziglang.org/documentation/0.16.0/. The deltas most likely to trip code (or a model)
trained on older Zig:

- **`usingnamespace` was removed.** This repo uses none — do not introduce it. Compose with explicit
  `pub const` re-exports / namespacing instead.
- **Type constructors are builtins:** `@Int`, `@Vector`, `@Struct`, `@Union`, `@Enum`, `@Fn`,
  `@Pointer`, `@Tuple` (return `type`).
- **`@addWithOverflow`/`@subWithOverflow`/`@mulWithOverflow`** return an anonymous `struct{ T, u1 }`
  tuple — destructure it (`const r, const ov = @addWithOverflow(a, b);`).
- **`@splat(scalar)`** takes only the scalar and infers the vector type from the result location
  (no length argument).
- **Result-location inference** drives many casts: `@intCast`, `@ptrCast`, `@alignCast`, `@enumFromInt`
  infer their destination type from context (use `@as(T, x)` when you need to state the target type
  explicitly) — keep the target type obvious.
- **Build API shape:** modules via `b.addModule` / `b.createModule` with `.root_module`; options via
  `b.addOptions()` + `module.addOptions("build_options", ...)`; targets/optimize via
  `b.standardTargetOptions` / `b.standardOptimizeOption`.
- **ReleaseFast/ReleaseSmall drop safety checks.** A kernel that only behaves because Debug catches
  it is broken; prove invariants, don't rely on checks as logic.

## Doc index

- `docs/ARCHITECTURE.md` — the current Zig architecture from the actual source layout. Start here for structure.
- `docs/REFERENCE.md` — the detailed API reference: the full public surface with exact semantics (ownership, errors, defaults, thread-safety) and machine-verified example snippets for every important feature. Start here to *use* the library.
- `docs/RUNNING-MODELS.md` — model/example index: the verified weight-download table with license notes (weights are not bundled) plus the shared cross-runner machinery (MoE expert streaming, native MTP drafting, constrained decoding, GPU offload, global thread/BLAS knobs). Per-example getting-started guides live next to each entry file (e.g. `examples/qwen3/README.md`).
- `docs/LMSERVER.md` — the lmserve example: OpenAI chat-completions + stateless responses mapping tables (honored/rejected/ignored), the accept-concurrently/generate-sequentially architecture, streaming contracts (SSE chunk + semantic-event skeletons), constrained-output plumbing, the per-model Backend matrix.
- `docs/BENCHMARK.md` — benchmark protocol for the Qwen GGUF runner, plus dated measurement snapshots/addenda. Read before making perf claims.
- `docs/GPU-OFFLOAD.md` — graphless eager GPU completion design: persistent queues/streams, storage fences/resources, transfer/device-buffer reuse, sync rules, gates, and Metal/CUDA measurements.
- `docs/MEMORY-MODEL.md` — why transient memory uses per-tensor `defer deinit` + `BufferPool` (not an arena); rationale, file:line evidence, the optional "frame" helper, and sharp edges.
- `docs/TRAINING.md` — training guide: tensor-lifetime rules, exec scopes, optimizers/param groups/LR schedules/clipping, gradient accumulation, cross-entropy options, dropout + the deterministic-RNG contract, gradient checkpointing, checkpoint directory contracts, LoRA + Qwen3 GGUF fine-tuning, gradient verification, the fine-tune→merge→quantize→serve export loop, bf16 policy, bench numbers, evolution strategies (gradient-free ES-at-scale, §13).
- `docs/DEVELOPMENT.md` — the development method: the design invariants (with enforcement and violation smells), the check-before-you-build capability inventory, the per-task template table, the gate matrix, and the delivery/reporting loop. Start here before writing new code.
- `docs/PORTING.md` — the porting method: oracle-first staging, the tiered tolerance policy, the LLM parity ladder, two-way interop proof, perf-after-parity ratchet, new-ISA discipline.
- `docs/SPECULATIVE.md` — design record: lossless draft-model-free speculative decoding (DraftSource vtable → SpeculationIndex cascade → SpeculativeDecoder), the losslessness proof obligations, verify economics + CostGate, SAM/recycling design, RAG injection, bench results with caveats.
- `docs/CONSTRAINED-DECODING.md` — design record: grammar/JSON-schema constrained decoding — the `LogitProcessor` seam on the shared sampler, the vendored llguidance engine + tokenizer bridge, why speculation composes without rollback, grammar-driven drafting (`ConstrainedSource`), `Constraint.clone` multi-stream, adjudications.
- `docs/CARTRIDGES.md` — design record: trained KV-prefix corpus compression (Cartridges, arXiv 2506.06266) — reference-pinned semantics (post-RoPE rows, sink freeze, teacher top-k distillation), the qwen3 trainer seams, in-process self-study, serving via KvCache preload, the prefill-equivalence gate (bitwise on Qwen3-0.6B), follow-ups incl. Cartridges-at-Scale.
- `docs/TERNARY.md` — design record: TQ2_0 ternary {-1,0,+1} weights as a first-class citizen — the mul-free int8 flagship + f32 kernels, b1.58 encoders, STE training op, ternary-native ES (training = inference), GGUF interop, bench numbers.
- `docs/PTQTP.md` — design record: PTQTP dual trit-plane post-training quantization (arXiv:2509.16989) over the TQ2_0 substrate — the alternating ridge/search solver, the G=256 packing identity and measured deltas, the `LinearWeight.toPtqtp` decoration seam, spirals + Qwen3-0.6B results, deferred GGUF pair persistence.
- `docs/THIRD-PARTY-NOTICES.md` — provenance and licenses of the vendored third-party code.
- `README.md` — overview, build, scope.
- `CONTRIBUTING.md` — the contribution bar: human-owned PRs, the two regression tracks (correctness + speed), reporting requirements, provenance rules.
