# AGENTS.md — Fucina

Fucina is a close-to-metal **CPU tensor / autograd runtime + LLM inference engine** written in
**Zig 0.16**. North Star: **match or beat llama.cpp on CPU**. It runs Qwen3 dense and MoE, Gemma 4,
and several other model families from GGUF weights (`docs/RUNNING-MODELS.md`; weights are not part
of the repo). CPU-first, with optional GPU GEMM offload via `-Dgpu=metal|cuda`; no ggml graph
runtime and no C/CMake build — pure Zig vector kernels plus optional CBLAS for GEMM (the vendored
Metal/CUDA kernels and audio shims are compiled by `build.zig` itself). This file is the working
guide for contributors and coding agents: toolchain, commands, repo map, house rules, doc index.

## Toolchain

- Pinned to **Zig 0.16.0** (`zig version` → `0.16.0`). `build.zig.zon` is the package manifest (`.version`, `.paths` = the shipped surface; `zig build doc-check` asserts README's fetch pin matches its version); modules are wired in `build.zig` and exported as `fucina` / `fucina_models`.

## Build, test, run, bench

```sh
zig build test                 # unit tests, every test root: src/fucina.zig, src/models.zig, and the test-carrying apps; each root also has a solo step (test-fucina, test-models, test-<app>)
zig build test-fucina -Dbackend=scalar  # THE scalar leg: fucina root only (the kernel/spec surface). The scalar backend verifies kernels/math; real-model golden forwards are native-only by design, so the full test matrix is a native gate
zig build test -Dblas=none        # native backend via pure Zig vector kernels (no CBLAS)
zig build arch-check           # production-only import graph over src/ + examples/ + apps/ + bench/ + tools/ (AST-based, test-aware): 0 forbidden SCCs (the root-anchored exec.zig <-> exec/*.zig SCC is permitted and counted), 0 band inversions (the ARCHITECTURE.md Layer Stack, encoded as band_table in the tool), every sibling test file forwarded
zig build doc-check            # doc rot gate: docs/README.md (the index) matches the docs-nav set; intra-doc links, anchors, and src:line citations resolve; README's zig-fetch pin matches build.zig.zon (tools/check_doc_links.zig)
zig build snippet-check        # reference snippet gate: every runnable ```zig snippet (named test block) in docs/reference/ extracted and run against the real fucina/fucina_models modules (tools/gen_snippet_tests.zig)
zig build x86dot-check         # cross-ISA int8/Q4_K/Q8_0/TQ2_0 dot parity checker (follows -Dtarget) + compile-only AVX2/VNNI/smmla bit-rot legs (src/x86dot_check.zig)
zig build bench-check          # compile every bench executable and the subq research tools without running them
zig build cuda-check           # compile-only -Dgpu=cuda legs (x86_64-linux-gnu fucina/models roots + NVRTC PTX generator, not run): CUDA-provider bit-rot gate for GPU-less machines
zig build smoke                # smoke example (examples/smoke/main.zig)
zig build run -- <gguf> [args]   # fucina-run, the registry runner: any registered GGUF arch (qwen3/qwen3moe/gemma4/qwen35/qwen35moe/inkling/deepseek2/deepseek4/glm4moe): completion/chat/REPL, sampling, NLL, parity dumps, --moe-* streaming, glm4moe --mtp, inkling multimodal (apps/run/main.zig)
zig build qwen3 -- <args>      # Qwen3 GGUF inference (apps/qwen3/main.zig; --spec/--spec-ref = lossless speculative decode, --tokenize = tokenizer-parity oracle)
zig build gemma4 -- <args>     # Gemma 4 GGUF inference / logit-parity harness; --chat/--repl/--spec (examples/gemma4/main.zig)
zig build qwen35 -- <args>     # Qwen3.5 (qwen35 hybrid Gated-DeltaNet) GGUF — loader/parity harness (examples/qwen35/main.zig; see examples/qwen35/README.md)
zig build deepseek4 -- <args>  # DeepSeek V4 Flash GGUF inference (hyper-connections, compressed KV, streamed experts, MTP; apps/deepseek4/main.zig)
zig build lmserve -- <args>    # OpenAI-compatible LM server (chat completions + stateless responses, SSE, JSON-schema constrained output w/ -Dllguidance=true) over qwen3/qwen35/gemma4/diffusion-gemma/inkling/deepseek4 GGUFs + nanochat checkpoints (apps/lmserve/main.zig; see docs/LMSERVER.md)
zig build parakeet -- <args>   # Parakeet ASR (NeMo FastConformer): WAV → text; --stream/--manifest/--mic (needs -Dparakeet-mic), --compare parity harness (apps/parakeet/main.zig)
zig build omnivoice -- <args>  # OmniVoice MaskGIT TTS: voice cloning/design/auto, codec encode/decode, parity oracles (apps/omnivoice/main.zig)
zig build facedetect -- <args> # buffalo_l face pipeline (face-detect.cpp port): info/detect/embed/verify/analyze + bench paired CPU harness (apps/facedetect/main.zig)
zig build nanochat -- <args>   # nanochat port (karpathy/nanochat): tok-train / base-train / sft / eval-bpb / chat — full CPU pipeline, GPT pretraining + SFT + chat w/ calculator tool (apps/nanochat/main.zig)
zig build diffusion-gemma -- <args>  # DiffusionGemma block text-diffusion: --eval parity harness vs llama.cpp PR 24423, --chat EB decoding (apps/diffusion_gemma/main.zig)
zig build locate-anything -- <args>  # LocateAnything-3B open-vocabulary detection: detect/info CLI + exit-code parity gates vs reference dumps (apps/locate_anything/main.zig)
zig build spirals              # two-spirals training demo: SGD/AdamW/Muon/APOLLO + checkpoints (examples/spirals/main.zig)
zig build nam -- <args>        # Neural Amp Modeler: .nam profile import/run/train/export, GGUF interchange, live amp sim (apps/nam/main.zig)
zig build finetune -- <args>   # LoRA fine-tune a Qwen3 GGUF on CPU (apps/finetune/main.zig)
zig build cartridge -- <args>  # Cartridges (arXiv 2506.06266): train a corpus into a reusable KV prefix by in-process self-study distillation + serve it (apps/cartridge/main.zig; see docs/CARTRIDGES.md)
zig build cartridge-fleet -- <args>  # per-document cartridge fleets: joint training, budget manager, cosine cartridge-RAG (apps/cartridge_fleet/main.zig)
zig build engram -- <args>     # conditional n-gram memory graft trained on a frozen Qwen3 GGUF (examples/engram/main.zig; see docs/ENGRAM.md)
zig build es-finetune -- <args>  # gradient-free ES fine-tune of a Qwen3 GGUF (apps/es_finetune/main.zig; --mode lora|full, --reward rule|nll|acc)
zig build es-spirals           # two-spirals MLP trained FROM SCRATCH by ES (examples/es_spirals/main.zig; self-verifying, member-parallel replicas)
zig build es-ternary-spirals   # two-spirals MLP with PACKED TERNARY (TQ2_0) hidden/output layers trained by ternary-native ES — training state IS the int8 inference model (examples/es_ternary_spirals/main.zig; see docs/TERNARY.md)
zig build ptqtp-spirals        # float-train a two-spirals MLP, then post-training-quantize it to DUAL TRIT-PLANES (PTQTP, arXiv:2509.16989: packed TQ2_0 plane pairs; self-verifying — examples/ptqtp_spirals/main.zig, docs/PTQTP.md)
zig build ptqtp-qwen3 -- <gguf>  # PTQTP-decorate a Qwen3 GGUF's linears in place (any source dtype) + teacher-forced NLL before/after + greedy completion; --planes 1|2|3, --down-planes/--o-planes N = selective third plane, --skip-first/--skip-last N = edge layers stay source precision (examples/ptqtp_qwen3/main.zig)
zig build export-gguf -- <args>  # export a GGUF: re-emit/transcode (incl. --dtype tq2_0 ternary), merge LoRA adapters into dense weights, or shard-streaming PTQTP quantization (--ptqtp[=K], one tensor at a time — models bigger than RAM; tools/export_gguf.zig)
zig build bench                # MLP-shaped inference/backward benchmarks
zig build bench-gate           # paired Fucina-vs-llama benchmark gate (tools/bench_gate.py; protocol in docs/BENCHMARK.md)
zig build bench-optim          # optimizer step kernels at LLM shapes (bench/optim.zig)
zig build bench-ce             # softmax / cross-entropy / layerNorm row kernels at LLM shapes (bench/ce.zig)
zig build bench-conv           # conv2d forward/backward-input/backward-weight at CNN shapes (bench/conv.zig)
zig build bench-scatter        # scatter-add (embedding-gradient) kernel at vocab x dim shapes (bench/scatter.zig)
zig build bench-masked-reduce  # masked reductions: fused vs maskedFill+reduce vs unmasked (bench/masked_reduce.zig)
zig build bench-backend        # scalar vs native backends on representative ops
zig build bench-f16gemm        # f16 TransB GEMM parallel-efficiency microbench
zig build bench-gemm           # large-shape f32 GEMM: row kernels vs blocked packed kernel vs BLAS dispatch (bench/gemm.zig)
zig build bench-packed-gemm    # pack-once dense f32/f16/bf16 RHS GEMM at skinny-m inference shapes (bench/packed_gemm.zig)
zig build bench-train-step     # end-to-end GPT autograd training step on a fixed synthetic sequence; --inference = eval-mode forward, --dump <dir> feeds tools/torch_train_step.py (bench/train_step.zig)
zig build bench-gpu-dispatch  # CPU BLAS vs blocking/async eager GPU GEMM/GEMV latency + queued throughput
zig build bench-gpu-formats   # packed CPU vs eager GPU f16/Q4_K/Q5_K/Q6_K/Q8_0 LLM-linear latency + queued throughput
zig build bench-q5kmoe         # Q5_K MoE-expert matmul: per-row vs 4-row lane-packed col-outer (bench/q5kmoe.zig)
zig build bench-q8gemv         # q8_0 skinny-m decode GEMV: per-row vs x4 interleaved vs lane-packed LHS (bench/q8gemv.zig)
zig build bench-ternary        # TQ2_0 ternary matmul: hot sdot/vpdpbusd tiles vs x4 column-interleaved pack (interleaved A/B decision pair) vs cold table path, mul-free f32 path, Q4_K, dense f32 (bench/ternary.zig)
zig build bench-membw          # measured DRAM read-bandwidth ceiling, single-thread + all-core — the roofline denominator for weight-stream GB/s (bench/membw.zig)
zig build bench-attention-backward  # grouped causal attention backward (bench/attention_backward.zig)
zig build bench-facade         # raw tensor ops vs public no-grad Tensor facade
zig build bench-einsum         # einsum vs hand-written dot/permute contraction pipelines (parity + advantage cases)
zig build bench-backward-diamond  # serial vs manual-parallel independent GEMM VJPs
```

`zig build --help` lists every step; runner CLIs live in `docs/RUNNING-MODELS.md` and the per-target `examples/<name>/README.md` / `apps/<name>/README.md`.

Build options (consumed at comptime via `build_options`; the full table with defaults and constraints is the reference's toolchain chapter, `docs/reference/02-toolchain-build-and-project-wiring.md`):

- `-Dbackend=native|scalar` — `native` (default) = Zig SIMD + optional BLAS; `scalar` = reference.
- `-Dblas=none|accelerate|openblas|mkl|blis|nvpl|blas` — CBLAS provider for GEMM. Default `accelerate`
  on macOS, auto-detected/`none` elsewhere; `none` keeps the pure Zig vector kernels.
  `-Dblas-threads=N` pins vendor BLAS threads (`0` = provider default).
- `-Dmax-threads=N` — comptime worker-team ceiling *and* runtime default thread count (1–64,
  default 8 = M1 Max P-cores; `src/parallel.zig`). `FUCINA_MAX_THREADS` still only lowers it at
  runtime — many-core servers must raise the ceiling at build time.
- `-Dgpu=none|metal|cuda` — GPU GEMM offload. **metal** (macOS): big f32/f16 GEMMs, dense
  quantized prefill linears (Q4_K/Q6_K/Q8_0, ternary TQ2_0), and the grouped quantized MoE expert
  FFN, behind measured work gates. **cuda** (Linux/NVIDIA; no CUDA SDK at build time — dlopen'd
  cuBLAS + vendored PTX; cross-compiles with `-Dtarget=x86_64-linux-gnu`): the same surface plus
  Q5_K, exec-tier attention forward, and opt-in quantized decode (`FUCINA_GPU_DECODE=1`). Decode
  below the gates and training stay on CPU. Every `FUCINA_GPU_*` gate and default is tabled in
  the reference's runtime-environment section; design and measurements in `docs/GPU-OFFLOAD.md`.
- `-Dvector-scan=bool` — vectorize the scan kernels (default `false` = serial-per-row scans).
- Standard `-Doptimize=Debug|ReleaseSafe|ReleaseFast|ReleaseSmall` and `-Dtarget=...`.
- **CPU targeting: native by default.** With no `-Dtarget`, Zig targets the compiling machine's
  exact CPU and the kernels' comptime feature gates compile in the matching arms (NEON/sdot,
  AVX2/AVX-VNNI, smmla, portable vectors); unused arms are compiled out (no runtime dispatch).
  Cross-compiling with `-Dtarget=...` drops to that architecture's BASELINE unless `-Dcpu=...`
  names a model — a bare `-Dtarget` binary silently loses the fast kernels. Build on the machine
  that will run it, or pin `-Dcpu` to match it.

## Repo map

| Path | Role | Detail |
| --- | --- | --- |
| `src/fucina.zig` | public facade, the `fucina` module root | reference ch 1 |
| `src/tensor.zig`, `src/storage.zig`, `src/dtype.zig` | raw tensor + refcounted storage + dtype/block registry | reference ch 8 |
| `src/tag_ops.zig`, `src/tags.zig` | tag-semantics op library + comptime axis-tag metaprogramming | reference ch 7 |
| `src/exec.zig`, `src/exec/` | `ExecContext` eager runtime: ops, buffer pool, scopes | reference ch 6; `docs/MEMORY-MODEL.md` |
| `src/store/` | disk-streamed block stores (`expert_store` = out-of-core MoE experts) | reference ch 13; `docs/PTQTP.md` |
| `src/backend.zig`, `src/backend/` | final numeric kernels: native/scalar providers, encoders | reference ch 9-10; `docs/TERNARY.md` |
| `src/backend/metal.zig`, `src/backend/metal/` | Metal GPU GEMM provider (vendored MLX + ggml kernels) | `docs/GPU-OFFLOAD.md` |
| `src/backend/gpu.zig`, `src/backend/cuda.zig`, `src/backend/cuda/` | provider selector + CUDA provider | `docs/GPU-OFFLOAD.md` |
| `src/accelerator.zig` | lifetime tokens for submitted eager GPU work (completion tracking only) | `docs/GPU-OFFLOAD.md` |
| `src/parallel.zig`, `src/thread.zig` | thread pool + parallel-chunk helpers | reference ch 9 |
| `src/gguf.zig`, `src/state_dict.zig`, `src/safetensors.zig` | GGUF parser/writer; named state dicts | reference ch 12 |
| `src/fpenv.zig`, `src/rng.zig` | IEEE float-environment guard; deterministic counter-based RNG (checkpoint contract) | reference ch 6 |
| `src/ag.zig`, `src/ag/` | autograd: tensor facade, VJPs, scheduling, activation checkpointing | reference ch 3-5 |
| `src/optim.zig`, `src/optim/` | SGD/AdamW/Muon/APOLLO, clipping, schedules, param groups, checkpoint frames | `docs/TRAINING.md` |
| `src/training_checkpoint.zig`, `src/lora.zig`, `src/param_registry.zig` | checkpoint dirs, LoRA adapters, named params | `docs/TRAINING.md` |
| `src/es.zig` | evolution strategies at scale + the ternary-native strategy | `docs/TRAINING.md` §13; `docs/TERNARY.md` |
| `src/ptqtp.zig` | PTQTP trit-plane PTQ solver (K ∈ {1,2,3}) | `docs/PTQTP.md` |
| `src/models.zig`, `src/models/` | model band: families, text runtime, research tier | reference ch 13-14; `docs/ARCHITECTURE.md` |
| `src/models/text/speculative/` | speculative-decoding subsystem (SAM/recycling/MTP/constrained cascade) | `docs/SPECULATIVE.md` |
| `tools/export_gguf.zig` | GGUF re-emit/transcode, LoRA merge, shard-streaming PTQTP quantizer | `docs/PTQTP.md`; reference ch 12 |
| `examples/` | teaching code: one `main.zig` per directory (a README is fine), no test files, no C shims, no vendored assets; it exists to be read and copied | `docs/RUNNING-MODELS.md` |
| `apps/` | product- and port-shaped programs: multi-file, own tests, shims, goldens, or a real CLI surface; one directory per app rooted at `main.zig`, each README owns its CLI | `docs/RUNNING-MODELS.md` |
| `apps/run/` | fucina-run, the registry GGUF runner (`zig build run -- <gguf>`) | `docs/RUNNING-MODELS.md` |
| `apps/omnivoice/` | OmniVoice MaskGIT TTS port (voice clone/design; byte-exact parity) | `apps/omnivoice/README.md` |
| `apps/locate_anything/` | LocateAnything-3B open-vocabulary detection VLM port | `apps/locate_anything/README.md` |
| `apps/facedetect/` | buffalo_l face pipeline port (detect/recognize/analyze) | `apps/facedetect/README.md` |
| `apps/nanochat/` | nanochat port: BPE training, pretraining, SFT, eval, chat on CPU | `apps/nanochat/README.md` |
| `apps/nam/` | Neural Amp Modeler port: `.nam` engines, training, live audio/MIDI | `apps/nam/README.md` |
| `bench/`, `src/bench_raw.zig` | microbenchmarks + their internal raw-surface module | `docs/BENCHMARK.md` |
| `refs/` (untracked) | reference-repo clones + parity goldens (`tools/fetch_refs.sh`) | `docs/PORTING.md`; `docs/BENCHMARK.md` |

**Placement policy** (reusable family vs example-local port, kernel orchestration, shared audio
helpers): `docs/DEVELOPMENT.md` §1.8.

## House rules (repo-specific)

- **Benchmark before "done".** Kernel/perf changes are not complete until measured against the
  protocol in `docs/BENCHMARK.md`. SOTA CPU perf is the point; bench in `ReleaseFast`, validate in
  `Debug`/`ReleaseSafe`.
- **Backend outputs are exec-supplied.** Kernels never allocate or retain output tensors — results
  go into buffers supplied by `ExecContext`, and the vector/quant compute leaves are
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

The doc index is `docs/README.md` (enforced by `zig build doc-check`); the API reference is
`docs/reference/`, and `docs/ARCHITECTURE.md` is the structural entry point.
